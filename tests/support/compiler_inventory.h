// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef LLMP_TESTS_SUPPORT_COMPILER_INVENTORY_H_
#define LLMP_TESTS_SUPPORT_COMPILER_INVENTORY_H_

#include <cstdint>

#include "ir/graph.h"
#include "kernels/ggml/lower.h"
#include "planner/construction.h"

namespace llmp::test_support::compiler_inventory {
inline constexpr auto kGraphNodeBytes = planner::kConstructionGraphNodeBytes;
// These constexpr expressions are for the fixtures' fixed, checked inventories.
// The native construction category retains the reviewed arena512 + planner1024
// + traversal32 bytes/descriptor. Physical tables add two vector capacities:
// one retained inventory plus conservative temporary/reallocation overlap.
constexpr std::uint64_t NativeBytes(std::uint64_t descriptors, std::uint64_t tensors) {
  return descriptors * (512 + 1024 + 32 + 2 * sizeof(kernels::ggml::PhysicalTensor)) +
         tensors * 2 * sizeof(kernels::ggml::PhysicalRoot);
}
// Legacy first-screen derivation has seven overlapping graph inventories and
// three-capacity growth. It already pays its old native arena/planner category;
// the native arena is independently refused above 65,536 descriptors.
inline constexpr std::uint64_t kLegacyCompilerGrowth =
    7 * 3 * ir::Graph::kMaxNodes * planner::kConstructionAppliedGrowth +
    planner::NumericalConstructionBytes(ir::Graph::kMaxNodes).value() +
    2 * ir::Graph::kMaxTensors * sizeof(kernels::ggml::PhysicalRoot) +
    2 * 65536ULL * sizeof(kernels::ggml::PhysicalTensor);
// All source-specific fixtures below use the original4*3 graph phase/growth
// bound. Numerical vectors are a separate actual-site/round category.
constexpr std::uint64_t GraphBytes(std::uint64_t nodes, std::uint64_t tensors) {
  return 12 * (nodes * kGraphNodeBytes + tensors * planner::kConstructionGraphTensorBytes) +
         planner::NumericalConstructionBytes(nodes).value();
}
}  // namespace llmp::test_support::compiler_inventory

#endif  // LLMP_TESTS_SUPPORT_COMPILER_INVENTORY_H_
