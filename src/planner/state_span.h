// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Immutable layout facts supplied by the engine, independently checked by
// lowerings. A fact proves virtual offsets; backing still belongs to the
// step's catalog closure. No provider addresses are serialized in the IR.
#ifndef LLMP_PLANNER_STATE_SPAN_H_
#define LLMP_PLANNER_STATE_SPAN_H_

#include <array>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>

#include "ir/graph.h"
#include "model/state_contract.h"

namespace llmp::planner {
struct StateCellSlice {
  ir::StateRef ref{};
  std::uint64_t first = 0;
  std::uint64_t cells = 0;
};
struct StateSpanFact {
  model::StateBankSpan bank{};
  std::uint64_t region_bytes = 0;
};
// All slices must resolve to owned items of the same stage and owner.
// Empty means no proof: retain the primitive copy. The engine also checks
// supplied runtime bindings before a pattern chooses a bank recipe; the
// lowering checks the actual bound descriptors again.
using StateSpans = std::function<std::optional<StateSpanFact>(std::span<const StateCellSlice>)>;
// The zero-copy attention recipe's exact boundary: whole raw cells and
// a compressed prefix, both direct immutable views of packed F16 state
// matrices. The formula itself remains valid for more general operands;
// those retain its primitive concatenation.
inline std::optional<StateCellSlice> StateCellPrefix(const ir::Graph& graph, ir::TensorId tensor,
                                                     bool whole) {
  if (!tensor.valid()) return std::nullopt;
  const auto& d = graph.tensor(tensor);
  const auto& node = graph.node(graph.producer(tensor));
  const auto root = graph.root(tensor);
  const auto& rd = graph.tensor(root);
  const auto* ref = std::get_if<ir::StateRef>(&rd.binding);
  if (!d.is_view() || node.key != ir::Primitive(ir::Op::kView) || node.src[0] != root ||
      graph.node(graph.producer(root)).key != ir::Primitive(ir::Op::kState) || ref == nullptr ||
      ref->segment == ir::kJoinedSegment || d.segment != ref->segment ||
      rd.role != ir::Role::kState || rd.format.encoding != format::Encoding::kF16 ||
      rd.format.layout != format::Layout::kDense || !ir::IsContiguous(rd) || d.view_offset != 0 ||
      d.ne(0) != rd.ne(0) || d.ne(1) != 1 || d.ne(3) != 1 || rd.ne(2) != 1 || rd.ne(3) != 1 ||
      d.ne(2) > rd.ne(1) || (whole && d.ne(2) != rd.ne(1)) ||
      d.nb != std::array<std::uint64_t, 4>{rd.nb[0], rd.nb[1], rd.nb[1], rd.nb[2]})
    return std::nullopt;
  return StateCellSlice{.ref = *ref, .cells = d.ne(2)};
}
inline std::optional<std::array<StateCellSlice, 2>> StateBankSlices(const ir::Graph& graph,
                                                                    ir::TensorId raw,
                                                                    ir::TensorId compressed) {
  const auto a = StateCellPrefix(graph, raw, true);
  const auto b = StateCellPrefix(graph, compressed, false);
  if (!a || !b || a->ref.stage != b->ref.stage || a->ref.segment != b->ref.segment ||
      a->ref.layer != b->ref.layer || a->ref.tensor == b->ref.tensor ||
      graph.tensor(raw).ne(0) != graph.tensor(compressed).ne(0))
    return std::nullopt;
  return std::array{*a, *b};
}
}  // namespace llmp::planner
#endif  // LLMP_PLANNER_STATE_SPAN_H_
