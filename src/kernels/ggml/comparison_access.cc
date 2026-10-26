// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#include "kernels/ggml/comparison_access.h"

#include <algorithm>
#include <bit>
#include <span>

#include "kernels/ggml/capabilities.h"
#include "kernels/ggml/comparison.h"
#include "kernels/ggml/comparison_schema.h"
#include "kernels/ggml/dsv4_hc_norm.h"
#include "kernels/ggml/llmp_ops.h"
#include "kernels/ggml/ops.h"
#include "kernels/ggml/validate.h"
#include "kernels/ggml/validate_ext.h"
namespace llmp::kernels::ggml {
namespace {
using Nodes = std::span<const ggml_tensor* const>;
using Result = std::optional<ComparisonExplicitAccesses>;
void Add(ComparisonExplicitAccesses& out, const ggml_tensor* t, bool read, bool written,
         std::uint64_t alignment = 0) {
  out.values[out.count++] = {.tensor = t,
                             .bytes = static_cast<std::uint64_t>(ggml_nbytes(t)),
                             .alignment = alignment ? alignment : ggml_type_size(t->type),
                             .read = read,
                             .written = written};
}
Result UnaryReadWrite(const ggml_tensor* t, std::uint64_t alignment = 0) {
  ComparisonExplicitAccesses out;
  Add(out, t->src[0], true, false, alignment);
  Add(out, t, false, true, alignment);
  return out;
}
Result BinaryReadWrite(const ggml_tensor* t, std::uint64_t alignment = 0) {
  ComparisonExplicitAccesses out;
  Add(out, t->src[0], true, false, alignment);
  Add(out, t->src[1], true, false, alignment);
  Add(out, t, false, true, alignment);
  return out;
}
Result FourSourceReadWrite(const ggml_tensor* t, std::uint64_t src_alignment,
                           std::uint64_t dst_alignment) {
  ComparisonExplicitAccesses out;
  for (std::size_t i = 0; i < 4; ++i) Add(out, t->src[i], true, false, src_alignment);
  Add(out, t, false, true, dst_alignment);
  return out;
}
Result HcCombineAccess(Nodes n, ComparisonAccessScope) {
  const auto* combine = n[0];
  if (!CheckHcCombine(combine) || !HasNoExtraSources(*combine, 3)) return {};
  ComparisonExplicitAccesses out;
  Add(out, combine->src[0], true, false, 16);
  Add(out, combine->src[1], true, false, 16);
  Add(out, combine->src[2], true, false, 4);
  Add(out, combine, false, true, 16);
  return out;
}
Result GdnGatesAccess(Nodes n, ComparisonAccessScope) {
  const auto* gates = n[0];
  if (!CheckGdnGates(gates) || !HasNoExtraSources(*gates, 4)) return {};
  return FourSourceReadWrite(gates, 4, 4);
}
Result HcPostAccess(Nodes n, ComparisonAccessScope) {
  const auto* post = n[0];
  if (!CheckHcPost(post) || !HasNoExtraSources(*post, 4)) return {};
  return FourSourceReadWrite(post, 4, 4);
}
Result HcPostNormAccess(Nodes n, ComparisonAccessScope scope) {
  const auto* post = n[0];
  const auto* norm = n[1];
  if ((scope == ComparisonAccessScope::kLocal && norm->type != GGML_TYPE_F32) ||
      !CheckDsv4HcPostNormF16(post, norm))
    return {};
  const auto* flat = norm->src[0];
  if (flat->op != GGML_OP_RESHAPE || !HasTypedDescriptorSchema(*flat) || flat->type != post->type ||
      flat->view_src != post || flat->view_offs != 0 || flat->src[0] != post ||
      !HasNoExtraSources(*flat, 1))
    return {};
  auto out = FourSourceReadWrite(post, 4, 16);
  Add(*out, norm, false, true, 16);
  return out;
}
Result Binary(Nodes n, ComparisonAccessScope scope, ggml_op op) {
  const auto* t = n[0];
  if ((op == GGML_OP_ADD && scope == ComparisonAccessScope::kLocal && t->type != GGML_TYPE_F32) ||
      !CheckBinary(t, op))
    return {};
  return BinaryReadWrite(t);
}
Result AddAccess(Nodes n, ComparisonAccessScope s) { return Binary(n, s, GGML_OP_ADD); }
Result MulAccess(Nodes n, ComparisonAccessScope s) { return Binary(n, s, GGML_OP_MUL); }
Result RmsAccess(Nodes n, ComparisonAccessScope) {
  if (!CheckRmsNorm(n[0])) return {};
  return UnaryReadWrite(n[0]);
}
Result RmsMulAccess(Nodes n, ComparisonAccessScope) {
  if (!CheckRmsNormMul(n[0], n[1])) return {};
  ComparisonExplicitAccesses out;
  Add(out, n[0]->src[0], true, false);
  Add(out, n[1]->src[0] == n[0] ? n[1]->src[1] : n[1]->src[0], true, false);
  Add(out, n[1], false, true);
  out.elided[0] = n[0];
  out.elided_count = 1;
  return out;
}
Result RmsThenMulAccess(Nodes n, ComparisonAccessScope) {
  if (!CheckRmsNormThenMul(n[0], n[1])) return {};
  auto out = UnaryReadWrite(n[0]);
  Add(*out, n[1]->src[0], true, false);
  Add(*out, n[1]->src[1], true, false);
  Add(*out, n[1], false, true);
  for (std::size_t i = 2; i < out->count; ++i) out->values[i].phase = 1;
  return out;
}
Result RopeExtAccess(Nodes n, ComparisonAccessScope scope) {
  const auto* t = n[0];
  if (t->op != GGML_OP_ROPE || !CheckRopeExt(t) || !HasNoExtraSources(*t, 3)) return {};
  const auto real = [&](int word) { return std::bit_cast<float>(t->op_params[word]); };
  // Preserve the existing checked frequency-NEOX class. The additional class
  // is plain forward NORMAL/NEOX rotation, including a nonzero section offset.
  const bool frequency_neox =
      t->src[2] && t->op_params[11] == 0 && real(7) >= 0.0F && CheckRope(t).has_value();
  const bool partial =
      !t->src[2] && (t->op_params[2] == 0 || t->op_params[2] == GGML_ROPE_TYPE_NEOX) &&
      t->op_params[11] == 0 && real(6) == 1.0F && real(7) == 0.0F && real(8) == 1.0F;
  if (scope == ComparisonAccessScope::kLocal && !frequency_neox && !partial) return {};
  ComparisonExplicitAccesses out;
  Add(out, t->src[0], true, false, 4);
  Add(out, t->src[1], true, false, 4);
  if (t->src[2]) Add(out, t->src[2], true, false, 4);
  Add(out, t, false, true, 8);
  return out;
}
Result GammaConvertAccess(Nodes n, ComparisonAccessScope) {
  const auto* t = n[0];
  if (!HasNoExtraSources(*t, 2) || !CheckConvert(t) || t->src[0]->type != GGML_TYPE_BF16 ||
      t->type != GGML_TYPE_F32 || t->src[1]->type != GGML_TYPE_F32 || t->view_src != t->src[1] ||
      t->view_offs != 0 || !std::ranges::equal(t->nb, t->src[1]->nb))
    return {};
  ComparisonExplicitAccesses out;
  Add(out, t->src[0], true, false, 2);
  Add(out, t->src[1], false, true, 4);
  return out;
}
Result ContAccess(Nodes n, ComparisonAccessScope scope) {
  const auto* t = n[0];
  if (!HasNoExtraSources(*t, 1) || !CheckCont(t) ||
      (scope == ComparisonAccessScope::kLocal && t->type != GGML_TYPE_I32 &&
       t->type != GGML_TYPE_F32))
    return {};
  return UnaryReadWrite(t);
}
Result NormRopeSectionsAccess(Nodes n, ComparisonAccessScope) {
  const auto* t = n[0];
  const auto checked = CheckNormRopeSections(t);
  if (!checked || t->type != GGML_TYPE_F32 || t->src[0]->type != GGML_TYPE_F32 ||
      checked->sections < 1 || !HasNoExtraSources(*t, 2 + checked->sections))
    return {};
  ComparisonExplicitAccesses out;
  Add(out, t->src[0], true, false);
  Add(out, t->src[1], true, false);
  for (std::size_t section = 0; section < checked->sections; ++section)
    Add(out, checked->positions[section], true, false);
  Add(out, t, false, true);
  return out;
}
Result SpatialChannelNormAccess(Nodes n, ComparisonAccessScope) {
  const auto* t = n[0];
  if (!CheckSpatialChannelNorm(t) || !HasNoExtraSources(*t, 2)) return {};
  return BinaryReadWrite(t, 4);
}
Result L2NormAccess(Nodes n, ComparisonAccessScope) {
  const auto* t = n[0];
  if (!CheckL2Norm(t) || !HasNoExtraSources(*t, 1)) return {};
  return UnaryReadWrite(t, 4);
}
Result LayerNormAccess(Nodes n, ComparisonAccessScope) {
  if (!CheckLayerNorm(n[0])) return {};
  return UnaryReadWrite(n[0]);
}
Result ScaleAccess(Nodes n, ComparisonAccessScope) {
  if (!CheckScale(n[0])) return {};
  return UnaryReadWrite(n[0]);
}
Result UnaryAccess(Nodes n, ComparisonAccessScope) {
  if (!CheckUnary(n[0])) return {};
  return UnaryReadWrite(n[0]);
}
Result RepeatAccess(Nodes n, ComparisonAccessScope) {
  if (n[0]->type != GGML_TYPE_F32 || !CheckRepeat(n[0])) return {};
  return UnaryReadWrite(n[0]);
}
Result ConcatAccess(Nodes n, ComparisonAccessScope) {
  if (n[0]->type != GGML_TYPE_F32 || !CheckConcat(n[0])) return {};
  return BinaryReadWrite(n[0]);
}
bool Aligned(const ggml_tensor* t, std::uint64_t alignment) {
  return reinterpret_cast<std::uintptr_t>(t->data) % alignment == 0;
}
Result ModulateAccess(Nodes n, ComparisonAccessScope) {
  const auto* t = n[0];
  if (!CheckLayerNormModulateSplit(t)) return {};
  ComparisonExplicitAccesses out;
  Add(out, t->src[0], true, false, 4);
  Add(out, t->src[1], true, false, 4);
  Add(out, t->src[2], true, false, 4);
  Add(out, t, false, true, 4);
  out.dispatch = t->ne[0] == 4096 && Aligned(t->src[0], 16) && Aligned(t->src[1], 16) &&
                         Aligned(t->src[2], 16) && Aligned(t, t->type == GGML_TYPE_BF16 ? 8 : 16)
                     ? 1U
                     : 0U;
  return out;
}
Result GatedAccess(Nodes n, ComparisonAccessScope) {
  const auto* t = n[0];
  if (!CheckGatedResidualSplit(t)) return {};
  ComparisonExplicitAccesses out;
  Add(out, t->src[0], true, false);
  Add(out, t->src[1], true, false);
  Add(out, t->src[2], true, false, 4);
  Add(out, t->src[3], true, false, 4);
  Add(out, t, false, true);
  out.dispatch = 0;
  return out;
}
Result JoinedAccess(Nodes n, ComparisonAccessScope) {
  const auto* r = n[0];
  const auto* m = n[1];
  if (!CheckGatedResidualModulateSplit(r, m)) return {};
  ComparisonExplicitAccesses out;
  Add(out, r->src[0], true, false);
  Add(out, r->src[1], true, false);
  Add(out, r->src[2], true, false, 4);
  Add(out, r->src[3], true, false, 4);
  Add(out, r, false, true);
  if (r->ne[0] > 8192) Add(out, r, true, false, 4);
  Add(out, m->src[1], true, false, 4);
  Add(out, m->src[2], true, false, 4);
  Add(out, m, false, true, 4);
  for (std::size_t i = 5; i < out.count; ++i) out.values[i].phase = 1;
  const auto row_aligned = [](const ggml_tensor* t) {
    return Aligned(t, t->type == GGML_TYPE_BF16 ? 8 : 16);
  };
  const bool wide = r->ne[0] == 4096 && row_aligned(r->src[0]) && row_aligned(r->src[1]) &&
                    row_aligned(r) && row_aligned(m) && Aligned(r->src[2], 16) &&
                    Aligned(r->src[3], 16) && Aligned(m->src[1], 16) && Aligned(m->src[2], 16);
  out.dispatch = wide ? 1U : (r->ne[0] > 8192 ? 2U : 0U);
  return out;
}
struct Entry {
  std::string_view name;
  Result (*describe)(Nodes, ComparisonAccessScope);
};
#define LLMP_GGML_NORM_RECIPE(NAME, VARIANT, CHECK, RUN)
#define LLMP_GGML_RECIPE(NAME, KEY, VARIANT, ARITY, CHECK, RUN)
#define LLMP_GGML_ACCESS_RECIPE(NAME, DESCRIBE) {NAME, DESCRIBE},
constexpr Entry kEntries[] = {
#include "kernels/ggml/recipe_entries.inc"
};
#undef LLMP_GGML_ACCESS_RECIPE
#undef LLMP_GGML_RECIPE
#undef LLMP_GGML_NORM_RECIPE
}  // namespace
bool HasComparisonAccessRecipe(std::string_view name) {
  return std::ranges::find(kEntries, name, &Entry::name) != std::end(kEntries);
}
Result DescribeComparisonAccess(const PlanStep& step, ComparisonAccessScope scope) {
  const auto at = std::ranges::find(kEntries, step.implementation, &Entry::name);
  if (at == std::end(kEntries)) return {};
  const auto declarations = RecipeDeclarations();
  const auto declaration =
      std::ranges::find(declarations, step.implementation, &RecipeDeclaration::name);
  if (declaration == declarations.end() || declaration->operation != step.operation ||
      declaration->arity != step.nodes.size() || step.nodes.size() > 2)
    return {};
  std::array<const ggml_tensor*, 2> nodes{};
  for (std::size_t i = 0; i < step.nodes.size(); ++i) {
    nodes[i] = step.nodes[i];
    if (!nodes[i] || !HasTypedDescriptorSchema(*nodes[i])) return {};
  }
  auto result = at->describe(std::span(nodes).first(step.nodes.size()), scope);
  if (!result) return {};
  for (const auto& fact : std::span(result->values).first(result->count))
    if (!fact.tensor || !fact.tensor->data || fact.bytes == 0 || fact.alignment == 0 ||
        reinterpret_cast<std::uintptr_t>(fact.tensor->data) % fact.alignment != 0)
      return {};
  return result;
}
}  // namespace llmp::kernels::ggml
