// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef LLMP_PLANNER_CONSTRUCTION_H_
#define LLMP_PLANNER_CONSTRUCTION_H_

#include <cstdint>
#include <limits>
#include <optional>

#include "planner/pattern.h"

namespace llmp::planner {
// Preserve the reviewed graph/node/name/selection allowance, including its
// original 64-byte Applied slot. New numerical provenance grows that slot;
// this delta belongs to transient rewrite inventories, not primitive graphs
// retained by observation fixtures. Callers retain their phase/capacity factors.
inline constexpr std::uint64_t kConstructionAppliedGrowth = sizeof(Applied) > 64
                                                                ? sizeof(Applied) - 64
                                                                : 0;
// Three node-capacity copies plus the reviewed 72-byte name/selection slot.
// Parameter growth follows Node's actual layout, rather than its former size.
inline constexpr std::uint64_t kConstructionGraphNodeBytes =
    3 * sizeof(ir::Node) + 72 + kConstructionAppliedGrowth;
// Recipient nodes are disjoint within each round. A compound GLU node has
// two independently authenticated weight operands. Two rewrite rounds, with
// at most three-capacity old/current growth overlap each, bound all retained
// operand proofs and the temporary site/missing lists. Pay this category once per
// compile, independently of the caller's copied-graph/phase multipliers.
inline constexpr std::uint64_t kConstructionNumericalBytesPerNode =
    2 * 3 * (2 * sizeof(NumericalProof) + 2 * sizeof(ir::NodeId));
constexpr std::optional<std::uint64_t> NumericalConstructionBytes(std::uint64_t nodes) {
  if (nodes > std::numeric_limits<std::uint64_t>::max() / kConstructionNumericalBytesPerNode)
    return std::nullopt;
  return nodes * kConstructionNumericalBytesPerNode;
}
inline constexpr std::uint64_t kConstructionGraphTensorBytes = 192;
}  // namespace llmp::planner

#endif  // LLMP_PLANNER_CONSTRUCTION_H_
