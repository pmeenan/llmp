// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "kernels/ggml/gemma_moe_fusion.h"

#include <algorithm>
#include <cstdint>
#include <limits>

#include "kernels/ggml/graph_read_index.h"
#include "kernels/ggml/validate_util.h"

namespace llmp::kernels::ggml {
namespace {
using namespace detail;
// Borrowed descriptors may be mutated. Bound chains before any storage scan
// or ordinary GGML helper; reject cycles, stale addresses and end overflow.
std::optional<const ggml_tensor*> RootOf(const ggml_tensor* t) {
  for (unsigned depth = 0; t != nullptr && depth < 64; ++depth) {
    if (static_cast<unsigned>(t->type) >= GGML_TYPE_COUNT || !Extent(t)) return std::nullopt;
    if (t->view_src == nullptr) return t;
    const auto* p = t->view_src;
    if (static_cast<unsigned>(p->type) >= GGML_TYPE_COUNT) return std::nullopt;
    const auto child = Extent(t), parent = Extent(p);
    if (!child || !parent || t->view_offs > *parent || *child > *parent - t->view_offs)
      return std::nullopt;
    const auto address = reinterpret_cast<std::uintptr_t>(p->data);
    if (p->data == nullptr || t->view_offs > std::numeric_limits<std::uint64_t>::max() - address ||
        reinterpret_cast<std::uintptr_t>(t->data) != address + t->view_offs)
      return std::nullopt;
    t = p;
  }
  return std::nullopt;
}
bool Edges(const ggml_tensor* t, std::initializer_list<const ggml_tensor*> sources) {
  std::size_t i = 0;
  for (const auto* source : sources)
    if (t->src[i++] != source) return false;
  for (; i < GGML_MAX_SRC; ++i)
    if (t->src[i] != nullptr) return false;
  return true;
}
bool Shape(const ggml_tensor* t, ggml_type type, std::array<std::int64_t, 4> ne,
           bool packed = true) {
  return t != nullptr && t->type == type && std::ranges::equal(std::span(t->ne), ne) &&
         RootOf(t).has_value() && (!packed || Packed(t));
}
bool Reshape(const ggml_tensor* t, const ggml_tensor* source) {
  return t->op == GGML_OP_RESHAPE && Edges(t, {source}) && t->view_offs == 0 &&
         RootOf(t) == RootOf(source);
}
template <std::size_t N>
bool Window(GraphNodes graph, std::size_t index, std::array<ggml_tensor*, N>& nodes) {
  if (index > graph.size() || N > graph.size() - index) return false;
  std::copy_n(graph.begin() + static_cast<std::ptrdiff_t>(index), N, nodes.begin());
  for (std::size_t i = 0; i < N; ++i) {
    const auto* t = nodes[i];
    if (t == nullptr || !RootOf(t) ||
        (t->op != GGML_OP_VIEW && t->op != GGML_OP_RESHAPE && t->view_src != nullptr) ||
        std::find(nodes.begin(), nodes.begin() + static_cast<std::ptrdiff_t>(i), t) !=
            nodes.begin() + static_cast<std::ptrdiff_t>(i))
      return false;
  }
  return true;
}
template <std::size_t N>
bool Outside(const ggml_tensor* input, const std::array<ggml_tensor*, N>& nodes) {
  const auto root = RootOf(input);
  return root && std::ranges::none_of(nodes, [&](const auto* n) { return RootOf(n) == root; });
}
// Validate every graph/keep view chain before inspecting readers. Logical
// storage identity, rather than coincident addresses, identifies a dependency:
// placement may reuse an unrelated producer's slot at a different lifetime.
template <std::size_t N>
bool Private(GraphNodes graph, std::span<ggml_tensor* const> keep,
             const std::array<ggml_tensor*, N>& nodes, const ggml_tensor* output,
             const ggml_tensor* selected_ids, const GraphReadIndex* reads) {
  if (reads != nullptr && reads->Matches(graph, keep))
    return reads->Private(nodes, output, selected_ids);
  const auto out = RootOf(output);
  if (!out) return false;
  const auto internal = [&](const ggml_tensor* t) {
    return std::ranges::find(nodes, t) != nodes.end();
  };
  const auto readable = [&](const ggml_tensor* t) { return RootOf(t) == out || t == selected_ids; };
  const auto elided = [&](const ggml_tensor* t) {
    const auto root = RootOf(t);
    return root && std::ranges::any_of(nodes, [&](const auto* n) { return RootOf(n) == root; });
  };
  for (const auto* t : nodes)
    if ((t->flags & GGML_TENSOR_FLAG_OUTPUT) != 0 && !readable(t)) return false;
  for (const auto* t : keep) {
    if (t == nullptr) continue;
    if (!RootOf(t) || (elided(t) && !readable(t))) return false;
  }
  for (const auto* node : graph) {
    if (node == nullptr || !RootOf(node)) return false;
    for (const auto* source : node->src)
      if (source != nullptr && !RootOf(source)) return false;
    if (internal(node)) continue;
    if (elided(node) && !readable(node)) return false;
    for (const auto* source : node->src)
      if (source != nullptr && elided(source) && !readable(source)) return false;
  }
  // The last graph node is an implicit kept output.
  return graph.empty() || !elided(graph.back()) || readable(graph.back());
}
}  // namespace

std::optional<GemmaRoutingFusionNodes> GemmaRoutingFusionAt(GraphNodes graph, std::size_t index,
                                                            std::span<ggml_tensor* const> keep,
                                                            const detail::GraphReadIndex* reads) {
  GemmaRoutingFusionNodes f{};
  if (!Window(graph, index, f.nodes)) return std::nullopt;
  const auto& n = f.nodes;
  constexpr std::array ops = {GGML_OP_SOFT_MAX, GGML_OP_RESHAPE, GGML_OP_ARGSORT,  GGML_OP_VIEW,
                              GGML_OP_GET_ROWS, GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP,
                              GGML_OP_DIV,      GGML_OP_RESHAPE};
  for (std::size_t i = 0; i < n.size(); ++i)
    if (n[i]->op != ops[i]) return std::nullopt;
  const auto rows = n[0]->ne[1];
  if (rows <= 0 || rows > kGemmaMoeMaxRows || !Shape(n[0], GGML_TYPE_F32, {128, rows, 1, 1}) ||
      !Shape(n[0]->src[0], GGML_TYPE_F32, {128, rows, 1, 1}) || !Edges(n[0], {n[0]->src[0]}) ||
      ParamF32(n[0], 0) != 1.0f || ParamF32(n[0], 1) != 0.0f ||
      !Shape(n[1], GGML_TYPE_F32, {1, 128, rows, 1}) || !Reshape(n[1], n[0]) ||
      !Shape(n[2], GGML_TYPE_I32, {128, rows, 1, 1}) || !Edges(n[2], {n[0]}) ||
      n[2]->op_params[0] != GGML_SORT_ORDER_DESC ||
      !Shape(n[3], GGML_TYPE_I32, {8, rows, 1, 1}, false) || !Edges(n[3], {n[2]}) ||
      n[3]->view_src != n[2] || n[3]->view_offs != 0 ||
      !Shape(n[4], GGML_TYPE_F32, {1, 8, rows, 1}) || !Edges(n[4], {n[1], n[3]}) ||
      !Shape(n[5], GGML_TYPE_F32, {8, rows, 1, 1}) || !Reshape(n[5], n[4]) ||
      !Shape(n[6], GGML_TYPE_F32, {1, rows, 1, 1}) || !Edges(n[6], {n[5]}) ||
      !Shape(n[7], GGML_TYPE_F32, {1, rows, 1, 1}) || !Edges(n[7], {n[6]}) ||
      ParamF32(n[7], 0) != kGemmaRouteClamp ||
      ParamF32(n[7], 1) != std::numeric_limits<float>::infinity() ||
      !Shape(n[8], GGML_TYPE_F32, {8, rows, 1, 1}) || !Edges(n[8], {n[5], n[7]}) ||
      !Shape(n[9], GGML_TYPE_F32, {1, 8, rows, 1}) || !Reshape(n[9], n[8]))
    return std::nullopt;
  const auto r = static_cast<std::uint64_t>(rows);
  f.operands = {{n[0]->src[0], base::Bytes(r * 128 * 4)},
                {n[9], base::Bytes(r * 8 * 4)},
                {n[3], base::Bytes(r * 128 * 4)}};
  if (!Outside(f.operands.logits.tensor, n) || !CheckGemmaRouting(f.operands) ||
      !Private(graph, keep, n, n[9], n[3], reads))
    return std::nullopt;
  return f;
}

std::optional<GemmaReductionFusionNodes> GemmaReductionFusionAt(
    GraphNodes graph, std::size_t index, std::span<ggml_tensor* const> keep,
    const detail::GraphReadIndex* reads) {
  GemmaReductionFusionNodes f{};
  if (!Window(graph, index, f.nodes)) return std::nullopt;
  const auto& n = f.nodes;
  const auto rows = n[0]->ne[2];
  if (rows <= 0 || rows > kGemmaMoeMaxRows || n[0]->op != GGML_OP_MUL || n[1]->op != GGML_OP_MUL ||
      !Shape(n[0], GGML_TYPE_F32, {2816, 8, rows, 1}) ||
      !Shape(n[0]->src[0], GGML_TYPE_F32, {2816, 8, rows, 1}) ||
      !Shape(n[0]->src[1], GGML_TYPE_F32, {1, 8, rows, 1}) ||
      !Edges(n[0], {n[0]->src[0], n[0]->src[1]}) ||
      !Shape(n[1], GGML_TYPE_F32, {2816, 8, rows, 1}) ||
      !Shape(n[1]->src[1], GGML_TYPE_F32, {1, 8, rows, 1}) || !Edges(n[1], {n[0], n[1]->src[1]}))
    return std::nullopt;
  for (std::size_t i = 0; i < 8; ++i) {
    const auto* v = n[2 + i];
    if (v->op != GGML_OP_VIEW || !Shape(v, GGML_TYPE_F32, {2816, rows, 1, 1}, false) ||
        !Edges(v, {n[1]}) || v->view_src != n[1] || v->view_offs != i * 2816 * 4 || v->nb[0] != 4 ||
        v->nb[1] != 2816 * 8 * 4 || v->nb[2] != static_cast<std::uint64_t>(rows) * v->nb[1] ||
        v->nb[3] != v->nb[2])
      return std::nullopt;
  }
  for (std::size_t i = 0; i < 7; ++i) {
    const auto* left = i == 0 ? n[2] : n[9 + i];
    if (n[10 + i]->op != GGML_OP_ADD || !Shape(n[10 + i], GGML_TYPE_F32, {2816, rows, 1, 1}) ||
        !Edges(n[10 + i], {left, n[3 + i]}))
      return std::nullopt;
  }
  const auto r = static_cast<std::uint64_t>(rows);
  f.operands = {{n[0]->src[0], base::Bytes(r * 2816 * 8 * 4)},
                {n[0]->src[1], base::Bytes(r * 8 * 4)},
                {n[1]->src[1], base::Bytes(r * 8 * 4)},
                {n[16], base::Bytes(r * 2816 * 4)}};
  if (!Outside(f.operands.experts.tensor, n) || !Outside(f.operands.scales.tensor, n) ||
      !Outside(f.operands.weights.tensor, n) || !CheckGemmaScaledReduction(f.operands) ||
      !Private(graph, keep, n, n[16], nullptr, reads))
    return std::nullopt;
  return f;
}
}  // namespace llmp::kernels::ggml
