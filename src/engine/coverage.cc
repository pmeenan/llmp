// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "engine/coverage.h"

#include <format>
#include <optional>

#include "catalog/catalog.h"
#include "engine/paged_node.h"

namespace llmp::engine {

namespace {

catalog::MemoryClass ClassOf(const kernels::TensorUse& use) {
  switch (use.role) {
    case ir::Role::kWeight:
      return use.routed_experts ? catalog::MemoryClass::kRoutedExperts
                                : catalog::MemoryClass::kWeights;
    case ir::Role::kState:
      return catalog::MemoryClass::kLiveState;
    case ir::Role::kConstant:
      return catalog::MemoryClass::kRuntime;
    case ir::Role::kActivation:
    case ir::Role::kInput:
    case ir::Role::kOutput:
      break;
  }
  return catalog::MemoryClass::kScratch;
}

}  // namespace

void CheckCoverage(const PagedNode& node, int owner, std::span<const kernels::TensorUse> uses,
                   std::string_view what, Coverage& coverage) {
  for (const kernels::TensorUse& use : uses) {
    ++coverage.tensors;
    const catalog::MemoryClass expected = ClassOf(use);
    const std::optional<catalog::MemoryClass> covered =
        node.Covered(use.address, use.bytes, owner, expected != catalog::MemoryClass::kLiveState);
    if (covered != expected && coverage.violations++ == 0) {
      coverage.first_violation = std::format("{}{} of {} bytes at {:#x}", what,
                                             ir::RoleName(use.role), use.bytes, use.address);
    }
  }
}

}  // namespace llmp::engine
