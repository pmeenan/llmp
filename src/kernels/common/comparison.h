// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef LLMP_KERNELS_COMMON_COMPARISON_H_
#define LLMP_KERNELS_COMMON_COMPARISON_H_

#include <cstdint>
#include <optional>
#include <span>

#include "base/sha256.h"
#include "components/comparison_boundary.h"
#include "components/step.h"
#include "format/encoding.h"
#include "ir/graph.h"

namespace llmp::kernels {

enum class ComparisonRunMode : std::uint8_t { kFull, kWarm };
enum class ComparisonProofState : std::uint8_t {
  kUnproved,
  kDescriptorVerified,
  kBoundVerified,
};
enum class ComparisonMissing : std::uint8_t {
  kNone,
  kBoundary,
  kPhysicalClosure,
  kOperationSchema,
  kSelectedCall,
  kBindings,
  kSourceIdentity,
};

struct ComparisonBindingRange {
  ir::TensorId original_root{};
  std::optional<format::PartRole> part{};
  std::uint32_t allocation_class = 0;
  std::uint64_t allocation_bytes = 0;
  std::uint64_t offset = 0;
  std::uint64_t bytes = 0;
  std::uint64_t alignment = 0;
  std::uint64_t base = 0;  // transient authentication only; never hashed
  std::optional<base::Sha256Digest> resource_part_identity{};
  std::optional<components::ComparisonRowStorage> row_storage{};
};

// Component-source adapters may represent a disjoint staged input as its own
// bounded suballocation: base is the actual planned virtual slice, allocation
// bytes is that checked slice extent, and offset is zero. This is valid only
// after proving the slices disjoint. Exact aliases share a class; any partial
// overlap without one authenticated common allocation/offset representation
// must remain unproved. The base is live authentication only and is never
// hashed into a comparison key.

// Whole native proof authenticates one complete ordinary program. It grants
// no local cut, access, or remainder authority.
enum class ComparisonProofScope : std::uint8_t { kTypedAccessClosure, kWholeProgramNative };

// Whole-program method is derived from the actual backend declarations. An
// ordinary-only selected call cannot borrow a captured-program measurement.
enum class ComparisonExecutionMethod : std::uint8_t { kCaptured, kOrdinaryOnly };

struct ComparisonRequest {
  ComparisonRunMode mode = ComparisonRunMode::kFull;
  ComparisonProofScope scope = ComparisonProofScope::kTypedAccessClosure;
  std::span<const ComparisonBindingRange> bindings{};
  std::span<const components::StepState> state{};
  std::uint16_t active_stage = 0;
  const components::OwnedComparisonBoundary* expected_boundary = nullptr;
};

struct ComparisonRecipeSeal {
  std::uint32_t version = 4;
  ComparisonRunMode mode = ComparisonRunMode::kFull;
  ComparisonProofScope scope = ComparisonProofScope::kTypedAccessClosure;
  ComparisonExecutionMethod execution_method = ComparisonExecutionMethod::kCaptured;
  ComparisonProofState state = ComparisonProofState::kUnproved;
  ComparisonMissing missing = ComparisonMissing::kBoundary;
  base::Sha256Digest subject{};
  base::Sha256Digest boundary{};
  base::Sha256Digest planned_graph{};
  base::Sha256Digest physical_recipe{};
  base::Sha256Digest policy{};
  base::Sha256Digest registry{};
  base::Sha256Digest backend_recipe{};
  base::Sha256Digest backend_build{};
  base::Sha256Digest backend_selection{};
  // Minted only by the concrete backend alongside descriptor proof authority.
  base::Sha256Digest backend_device{};
  std::uint64_t activation_bytes = 0;
  std::uint64_t pool_bytes = 0;
  std::uint64_t private_pool_bytes = 0;
  std::uint64_t lanes = 0;
  std::uint64_t lane_scratch_bytes = 0;
  std::uint64_t scheduled_calls = 0;
  friend bool operator==(const ComparisonRecipeSeal&, const ComparisonRecipeSeal&) = default;
};

}  // namespace llmp::kernels

#endif  // LLMP_KERNELS_COMMON_COMPARISON_H_
