// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "planner/select.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

#include "execution/registry.h"
#include "ir/graph.h"
#include "ir/op.h"
#include "planner/pattern.h"
#include "planner/policy.h"

namespace llmp::planner {
namespace {

using ir::FusedKind;
using ir::Op;

bool Leaf(ir::Op op) { return op == Op::kInput || op == Op::kWeight || op == Op::kState; }
struct RootIdentity {
  ir::TensorId id{};
  ir::Binding binding{};
  std::uint32_t input = ir::TensorId::kNone;
  std::uint32_t computation = ir::NodeId::kNone;
  std::uint8_t output = 0;
  ir::Role role{};
  format::TensorFormat format{};
  std::array<std::uint64_t, 4> nb{};
  std::uint8_t segment = ir::kJoinedSegment;
  bool operator==(const RootIdentity&) const = default;
};
std::vector<RootIdentity> RootSeal(const ir::Graph& graph) {
  std::vector<RootIdentity> seal;
  std::uint32_t computation = 0;
  for (const auto& node : graph.nodes()) {
    const bool leaf = Leaf(node.key.op);
    for (std::uint8_t slot = 0; slot < node.nout; ++slot) {
      const auto id = node.out[slot];
      const auto& desc = graph.tensor(id);
      if (desc.is_view()) continue;
      RootIdentity entry{.id = id,
                         .binding = desc.binding,
                         .output = slot,
                         .role = desc.role,
                         .format = desc.format,
                         .nb = desc.nb,
                         .segment = desc.segment};
      if (!leaf)
        entry.computation = computation;
      else if (std::holds_alternative<std::monostate>(desc.binding)) {
        const auto inputs = graph.inputs();
        const auto it = std::ranges::find(inputs, id);
        entry.input = static_cast<std::uint32_t>(it - inputs.begin());
      }
      seal.push_back(std::move(entry));
    }
    if (!leaf) ++computation;
  }
  return seal;
}

// Its rows: every dimension past the first.
std::uint64_t Rows(const ir::TensorDesc& t) { return t.ne(1) * t.ne(2) * t.ne(3); }

const ir::TensorDesc* At(std::span<const ir::TensorDesc* const> list, std::size_t i) {
  return i < list.size() ? list[i] : nullptr;
}

// A product of weights `w` [K, N, ...] and input `x` [K, M, ...].
QueryShape Product(const ir::TensorDesc* w, const ir::TensorDesc* x) {
  if (w == nullptr || x == nullptr) {
    return {};
  }
  return {.rows = Rows(*x), .k = x->ne(0), .n = w->ne(1)};
}

}  // namespace

RootAlignment GraphRootAlignments(const ir::Graph& graph,
                                  std::span<const RootAlignmentFact> facts) {
  std::vector<std::uint64_t> alignments(graph.tensor_count(), 0);
  for (const auto& fact : facts) {
    if (!fact.root.valid() || fact.root.v >= graph.tensor_count() ||
        graph.tensor(fact.root).is_view() || fact.alignment == 0 ||
        (fact.alignment & (fact.alignment - 1)) != 0 || alignments[fact.root.v] != 0)
      return {};
    alignments[fact.root.v] = fact.alignment;
  }
  return [digest = graph.Digest(), seal = RootSeal(graph), alignments = std::move(alignments)](
             const ir::Graph& current, ir::TensorId root) {
    if (!root.valid() || root.v >= alignments.size() || alignments[root.v] == 0 ||
        root.v >= current.tensor_count() || current.tensor(root).is_view() ||
        current.Digest() != digest || RootSeal(current) != seal)
      return std::uint64_t{0};
    return alignments[root.v];
  };
}

QueryShape ShapeOfNode(ir::OpKey key, std::span<const ir::TensorDesc* const> sources,
                       std::span<const ir::TensorDesc* const> outputs) {
  const ir::TensorDesc* s0 = At(sources, 0);
  if (key.op == Op::kFused) {
    switch (key.fused) {
      case FusedKind::kMatMulGlu:
        return Product(s0, At(sources, 2));
      case FusedKind::kMatMulIdGlu:
      case FusedKind::kMoeExperts:
        return Product(s0, At(sources, 2));
      case FusedKind::kMatMulAdd:
        return Product(s0, At(sources, 1));
      case FusedKind::kPreparedProducts:
        return sources.empty() ? QueryShape{} : Product(s0, sources.back());
      case FusedKind::kStateBankAttention: {
        const ir::TensorDesc* raw = At(sources, 1);
        return s0 == nullptr || raw == nullptr
                   ? QueryShape{}
                   : QueryShape{.rows = s0->ne(1) * s0->ne(3), .k = s0->ne(0), .n = raw->ne(0)};
      }
      case FusedKind::kOwnerAttention: {
        const ir::TensorDesc* v = At(sources, 2 + ((sources.size() - 2) / 2));
        if (s0 == nullptr || v == nullptr) {
          return {};
        }
        return {.rows = s0->ne(1) * s0->ne(3), .k = s0->ne(0), .n = v->ne(0)};
      }
      case FusedKind::kRopeStore:
        return s0 == nullptr
                   ? QueryShape{}
                   : QueryShape{.rows = s0->ne(2) * s0->ne(3), .k = s0->ne(0) * s0->ne(1)};
      case FusedKind::kGroupedStores: {
        // A batch's member is one store (cache, src, idx).
        const ir::TensorDesc* src = At(sources, 1);
        return src == nullptr ? QueryShape{} : QueryShape{.rows = Rows(*src), .k = src->ne(0)};
      }
      default:
        break;
    }
  }
  switch (key.op) {
    case Op::kMatMul:
      return Product(s0, At(sources, 1));
    case Op::kMatMulId: {
      const ir::TensorDesc* b = At(sources, 1);
      if (s0 == nullptr || b == nullptr) {
        return {};
      }
      return {.rows = b->ne(2), .k = b->ne(0), .n = s0->ne(1)};
    }
    case Op::kAttention: {
      const ir::TensorDesc* v = At(sources, 2);
      if (s0 == nullptr || v == nullptr) {
        return {};
      }
      return {.rows = s0->ne(1) * s0->ne(3), .k = s0->ne(0), .n = v->ne(0)};
    }
    case Op::kCacheStore: {
      const ir::TensorDesc* src = At(sources, 1);
      return src == nullptr ? QueryShape{} : QueryShape{.rows = Rows(*src), .k = src->ne(0)};
    }
    default:
      break;
  }
  const ir::TensorDesc* t = s0 != nullptr ? s0 : At(outputs, 0);
  return t == nullptr ? QueryShape{} : QueryShape{.rows = Rows(*t), .k = t->ne(0)};
}

bool LaunchesNothing(ir::OpKey key) {
  switch (key.op) {
    case Op::kInput:
    case Op::kWeight:
    case Op::kState:
    case Op::kView:
    case Op::kReshape:
    case Op::kPermute:
      return true;
    default:
      return false;
  }
}

SelectionView::SelectionView(const execution::Registry& registry, const DeviceFacts& device,
                             const Policy& policy, bool row_invariant, bool capturable,
                             StateSpans state_spans, RootAlignment root_alignment)
    : registry_(registry),
      device_(device),
      policy_(policy),
      row_invariant_(row_invariant || policy.row_invariant),
      capturable_(capturable),
      state_spans_(std::move(state_spans)),
      root_alignment_(std::move(root_alignment)),
      ranks_(RanksFor(policy, device.device_class)) {}

std::vector<std::size_t> SelectionView::Candidates(ir::OpKey key,
                                                   std::span<const ir::TensorDesc* const> sources,
                                                   std::span<const ir::TensorDesc* const> outputs,
                                                   std::span<const std::byte> params,
                                                   bool state_span) const {
  // Hypothetical descriptors have no authenticated current graph/root.
  return ForOperands(key, sources, outputs, params, state_span, {});
}

std::vector<std::size_t> SelectionView::ForOperands(
    ir::OpKey key, std::span<const ir::TensorDesc* const> sources,
    std::span<const ir::TensorDesc* const> outputs, std::span<const std::byte> params,
    bool state_span, std::span<const std::uint64_t> alignments) const {
  // A composite op whose own implementations the policy leaves off
  // (pattern.h route_reduce) has none: it decomposes.
  if (!CompositeImplemented(policy_, key)) {
    return {};
  }
  std::vector<execution::OperandFormat> operands;
  operands.reserve(sources.size());
  for (std::size_t i = 0; i < sources.size(); ++i) {
    if (sources[i] == nullptr) {
      continue;
    }
    const format::TensorFormat& f = sources[i]->format;
    operands.push_back(
        {.operand = static_cast<std::uint8_t>(i),
         .encoding = f.encoding,
         .layout = f.layout,
         .params = f.params,
         .ne = {f.shape.ne.at(0), f.shape.ne.at(1), f.shape.ne.at(2), f.shape.ne.at(3)},
         .memory = execution::OperandMemory{
             .nb = sources[i]->nb,
             .view_offset = sources[i]->view_offset,
             .root_alignment = i < alignments.size() ? alignments[i] : 0}});
  }
  // The outputs' formats, for rules and predicates on the result.
  for (std::size_t i = 0; i < outputs.size(); ++i) {
    if (outputs[i] == nullptr) {
      continue;
    }
    const format::TensorFormat& f = outputs[i]->format;
    operands.push_back(
        {.operand = static_cast<std::uint8_t>(execution::kOutputOperand + i),
         .encoding = f.encoding,
         .layout = f.layout,
         .params = f.params,
         .ne = {f.shape.ne.at(0), f.shape.ne.at(1), f.shape.ne.at(2), f.shape.ne.at(3)},
         .memory = execution::OperandMemory{.nb = outputs[i]->nb,
                                            .view_offset = outputs[i]->view_offset,
                                            .root_alignment = sources.size() + i < alignments.size()
                                                                  ? alignments[sources.size() + i]
                                                                  : 0}});
  }
  const QueryShape shape = ShapeOfNode(key, sources, outputs);
  const execution::CapabilityQuery query{
      .op = key,
      .operands = operands,
      .rows = shape.rows,
      .k = shape.k,
      .n = shape.n,
      .arch = device_.arch,
      .features = device_.features,
      .row_invariant = row_invariant_,
      .capturable = capturable_,
      .policy = {.max_decode_rows = policy_.max_decode_rows, .allow_lossy = policy_.allow_lossy},
      .measured = ranks_,
      .tuned = device_.tuned,
      .params = params,
      .state_span = state_span};
  return execution::Candidates(registry_, query);
}

std::vector<std::size_t> SelectionView::Candidates(const ir::Graph& graph,
                                                   const ir::Node& node) const {
  return Candidates(graph, node, node.key);
}

std::vector<std::size_t> SelectionView::Candidates(const ir::Graph& graph, const ir::Node& node,
                                                   ir::OpKey key) const {
  std::vector<const ir::TensorDesc*> sources;
  for (const ir::TensorId s : node.sources()) {
    sources.push_back(s.valid() ? &graph.tensor(s) : nullptr);
  }
  std::vector<const ir::TensorDesc*> outputs;
  for (const ir::TensorId o : node.outputs()) {
    outputs.push_back(&graph.tensor(o));
  }
  bool state_span = false;
  if (state_spans_ && key == ir::Fused(FusedKind::kStateBankAttention)) {
    if (const auto slices = StateBankSlices(graph, node.source(1), node.source(2))) {
      const auto fact = state_spans_(*slices);
      const auto& raw = graph.tensor(node.src[1]);
      const auto& compressed = graph.tensor(node.src[2]);
      state_span = fact && fact->bank.encoding == format::Encoding::kF16 &&
                   fact->bank.cell_bytes == raw.nb[2] &&
                   fact->bank.range.bytes == (raw.ne(2) + compressed.ne(2)) * raw.nb[2] &&
                   fact->bank.range.offset <= fact->region_bytes &&
                   fact->bank.range.bytes <= fact->region_bytes - fact->bank.range.offset;
    }
  }
  std::vector<std::uint64_t> alignments;
  const auto alignment = [&](ir::TensorId tensor) {
    return root_alignment_ && tensor.valid() ? root_alignment_(graph, graph.root(tensor)) : 0;
  };
  for (const auto tensor : node.sources()) alignments.push_back(alignment(tensor));
  for (const auto tensor : node.outputs()) alignments.push_back(alignment(tensor));
  return ForOperands(key, sources, outputs, node.params, state_span, alignments);
}

}  // namespace llmp::planner
