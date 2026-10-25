// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef LLMP_KERNELS_GGML_COMPARISON_REGION_H_
#define LLMP_KERNELS_GGML_COMPARISON_REGION_H_

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

#include "components/comparison_boundary.h"
#include "execution/registry.h"
#include "kernels/common/comparison_region.h"
#include "kernels/ggml/comparison.h"
#include "kernels/ggml/lower.h"

namespace llmp::kernels::ggml {

class LaunchContext;
struct ComparisonQ8Dispatch;

struct ComparisonRegionAnalysis {
  ComparisonRegionStatus status = ComparisonRegionStatus::kUnproved;
  std::size_t first = 0;
  std::size_t end = 0;
  bool inputs_preserved = false;
  ComparisonRegionRecipeSeal seal{};
  std::span<const ir::NodeId> required_members{};
  std::span<const ComparisonRegionValuePort> value_ports{};
  std::string_view reason{};
};

enum class ComparisonRegionPhase : std::uint8_t { kDescriptor, kBound };
enum class ComparisonRegionPurpose : std::uint8_t {
  kExecutableComparison,
  kProductionCertificate,
  // Remainder equality requires contiguous closure, even though its local
  // certificates share production's ordinary full-program binding contract.
  kProductionRemainder
};

// The first-domain RMS/Mul access recipe preserves GGML's typed minimum
// alignment for default tensor accesses. More specialized calls supply their
// independently checked alignment explicitly.
std::uint64_t ComparisonRegionDefaultAccessAlignment(ggml_type type);

// The write-liveness decision used by the actual remainder checker. A true
// result means the write preserves this value; malformed intervals fail
// closed. Kept internal to GGML comparison validation, not execution authority.
bool ComparisonWritePreservesLifetime(std::uintptr_t write_address, std::uint64_t write_bytes,
                                      std::size_t call, std::uint8_t subcall,
                                      const ComparisonLifetimeFact& value);

// Projects a common owned source boundary onto the already lowered actual
// schedule. This never plans or selects a second graph. Its result spans and
// port records borrow only LoweredStep's preallocated paid relation.
ComparisonRegionAnalysis AnalyzeComparisonRegion(
    const LoweredStep& lowered, const components::OwnedComparisonRegionBoundary& boundary,
    const execution::Registry& registry, const kernels::ComparisonRecipeSeal& whole_recipe,
    ComparisonRegionPhase phase, const LaunchContext* launch_context = nullptr,
    const ComparisonQ8Dispatch* q8_dispatch = nullptr,
    ComparisonRegionPurpose purpose = ComparisonRegionPurpose::kExecutableComparison,
    ComparisonRemainderCallFactsFn describe_call_facts = nullptr);

// Compare the actual retained calls outside the same closed source region.
// Any expansion borrows `current`'s paid source-sized origin scratch.
ComparisonRemainderResult CompareComparisonRemainder(
    const LoweredStep& current, const execution::Registry& current_registry,
    const kernels::ComparisonRecipeSeal& current_recipe, const LaunchContext* current_context,
    const LoweredStep& candidate, const execution::Registry& candidate_registry,
    const kernels::ComparisonRecipeSeal& candidate_recipe, const LaunchContext* candidate_context,
    const components::OwnedComparisonRegionBoundary& boundary,
    kernels::ComparisonCertificatePhase phase,
    ComparisonRemainderCallFactsFn describe_call_facts = nullptr);

}  // namespace llmp::kernels::ggml

#endif  // LLMP_KERNELS_GGML_COMPARISON_REGION_H_
