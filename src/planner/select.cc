// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "planner/select.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "execution/registry.h"
#include "ir/graph.h"
#include "ir/op.h"
#include "planner/policy.h"

namespace llmp::planner {
namespace {

using ir::FusedKind;
using ir::Op;

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

QueryShape ShapeOfNode(ir::OpKey key, std::span<const ir::TensorDesc* const> sources,
                       std::span<const ir::TensorDesc* const> outputs) {
  const ir::TensorDesc* s0 = At(sources, 0);
  if (key.op == Op::kFused) {
    switch (key.fused) {
      case FusedKind::kMatMulGlu:
        return Product(s0, At(sources, 2));
      case FusedKind::kMatMulAdd:
        return Product(s0, At(sources, 1));
      case FusedKind::kPreparedProducts:
        return sources.empty() ? QueryShape{} : Product(s0, sources.back());
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
                             const Policy& policy, bool row_invariant, bool capturable)
    : registry_(registry),
      device_(device),
      policy_(policy),
      row_invariant_(row_invariant || policy.row_invariant),
      capturable_(capturable),
      ranks_(RanksFor(policy, device.device_class)) {}

std::vector<std::size_t> SelectionView::Candidates(ir::OpKey key,
                                                   std::span<const ir::TensorDesc* const> sources,
                                                   std::span<const ir::TensorDesc* const> outputs,
                                                   std::span<const std::byte> params) const {
  std::vector<execution::OperandFormat> operands;
  operands.reserve(sources.size());
  for (std::size_t i = 0; i < sources.size(); ++i) {
    if (sources[i] == nullptr) {
      continue;
    }
    const format::TensorFormat& f = sources[i]->format;
    operands.push_back({.operand = static_cast<std::uint8_t>(i),
                        .encoding = f.encoding,
                        .layout = f.layout,
                        .params = f.params});
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
      .params = params};
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
  return Candidates(key, sources, outputs, node.params);
}

}  // namespace llmp::planner
