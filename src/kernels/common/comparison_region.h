// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef LLMP_KERNELS_COMMON_COMPARISON_REGION_H_
#define LLMP_KERNELS_COMMON_COMPARISON_REGION_H_

#include <cstddef>
#include <cstdint>
#include <expected>
#include <optional>
#include <span>
#include <string_view>
#include <utility>

#include "base/sha256.h"
#include "components/comparison_boundary.h"
#include "ir/graph.h"
#include "kernels/common/comparison.h"

namespace llmp::kernels {

class StepLaunch;
struct StepFailure;
}  // namespace llmp::kernels

namespace llmp::planner {
class ComparisonOriginDag;
struct ComparisonFinalNodeOrigins;
}  // namespace llmp::planner

namespace llmp::kernels {

// Borrowed only for synchronous lowering. The lowered program copies the
// complete per-planned-node origin relation into its paid construction; it
// never retains these pointers or only the seed membership.
struct ComparisonRegionCapture {
  const components::OwnedComparisonRegionBoundary* seed = nullptr;
  planner::ComparisonOriginDag* origins = nullptr;
  const planner::ComparisonFinalNodeOrigins* final_nodes = nullptr;
};

enum class ComparisonRegionStatus : std::uint8_t { kClosed, kExpand, kUnproved };

// The production certificate never creates an executable region capability.
// Bound is an assertion to revalidate a previous descriptor on a current
// bound program; callers cannot use it to manufacture binding authority.
enum class ComparisonCertificatePhase : std::uint8_t { kDescriptor, kBound };

// Transient equality of the actual calls outside one already authenticated
// local region. This is discovery evidence only; it never authorizes running
// or selecting either program.
enum class ComparisonRemainderStatus : std::uint8_t { kSame, kExpand, kUnproved };

struct ComparisonRemainderResult {
  ComparisonRemainderStatus status = ComparisonRemainderStatus::kUnproved;
  // Expansion members borrow paid scratch owned by the reporting program and
  // remain valid until its next comparison request or destruction.
  std::span<const ir::NodeId> required_members{};
  std::string_view reason{};
};

enum class ComparisonRegionValueKind : std::uint8_t { kInput, kOutput };

// Actual crossing values for a closed local call span. `source_tensor` is a
// transient lookup key into the authenticated source boundary; `local_tensor`
// is canonical only within this projection. Address/extent/view facts come
// from the bound backend descriptors and are for staging/readback, not seal
// identity or caller-minted range authority.
struct ComparisonRegionValuePort {
  ComparisonRegionValueKind kind = ComparisonRegionValueKind::kInput;
  // Index in the requested OwnedComparisonRegionBoundary::ports() list;
  // distinct from that boundary record's member-local operand ordinal.
  std::uint32_t ordinal = 0;
  ir::TensorId source_tensor{};
  std::uint32_t local_tensor = 0;
  ir::Role role = ir::Role::kActivation;
  format::TensorFormat format{};
  std::uint64_t address = 0;
  std::uint64_t bytes = 0;
  std::uint64_t root_offset = 0;
  bool read = false;
  bool written = false;
};

// A local key is deliberately distinct from ComparisonRecipeSeal's
// whole-program digest. The coordinator uses the independent planned and
// physical domains, along with their implementation and selection owners.
struct ComparisonRegionRecipeSeal {
  std::uint32_t version = 3;
  ComparisonProofState state = ComparisonProofState::kUnproved;
  ComparisonMissing missing = ComparisonMissing::kBoundary;
  base::Sha256Digest subject{};
  base::Sha256Digest boundary{};
  base::Sha256Digest planned_region{};
  base::Sha256Digest physical_recipe{};
  base::Sha256Digest policy{};
  base::Sha256Digest registry{};
  base::Sha256Digest backend_recipe{};
  base::Sha256Digest backend_build{};
  base::Sha256Digest backend_selection{};
  // Producer-owned immutable device identity, independent of mutable tuning
  // rows and never supplied by the comparison caller.
  base::Sha256Digest backend_device{};
};

// Opaque authority minted by one bound program for one actual, contiguous
// call span. The span and live binding facts are intentionally not exposed;
// only the canonical recipe is public. The program reauthenticates owner,
// binding generation, and projection serial before either execution method.
class ComparisonRegionProjection {
 public:
  ComparisonRegionProjection(ComparisonRegionProjection&& other) noexcept
      : program_nonce_(std::exchange(other.program_nonce_, 0)),
        generation_(std::exchange(other.generation_, 0)),
        serial_(std::exchange(other.serial_, 0)),
        first_(std::exchange(other.first_, 0)),
        end_(std::exchange(other.end_, 0)),
        inputs_preserved_(std::exchange(other.inputs_preserved_, false)),
        boundary_(other.boundary_),
        recipe_(other.recipe_),
        seal_(other.seal_) {
    other.boundary_ = {};
    other.recipe_ = {};
    other.seal_ = {};
  }
  ComparisonRegionProjection& operator=(ComparisonRegionProjection&&) noexcept = delete;
  ComparisonRegionProjection(const ComparisonRegionProjection&) = delete;
  ComparisonRegionProjection& operator=(const ComparisonRegionProjection&) = delete;

  const base::Sha256Digest& recipe() const { return recipe_; }
  const base::Sha256Digest& boundary() const { return boundary_; }
  const ComparisonRegionRecipeSeal& seal() const { return seal_; }
  bool inputs_preserved() const { return inputs_preserved_; }

 private:
  friend class StepProgram;
  ComparisonRegionProjection(std::uint64_t program_nonce, std::uint64_t generation,
                             std::uint64_t serial, std::size_t first, std::size_t end,
                             ComparisonRegionRecipeSeal seal, bool inputs_preserved)
      : program_nonce_(program_nonce),
        generation_(generation),
        serial_(serial),
        first_(first),
        end_(end),
        inputs_preserved_(inputs_preserved),
        boundary_(seal.boundary),
        recipe_(seal.physical_recipe),
        seal_(seal) {}

  std::uint64_t program_nonce_ = 0;
  std::uint64_t generation_ = 0;
  std::uint64_t serial_ = 0;
  std::size_t first_ = 0;
  std::size_t end_ = 0;
  bool inputs_preserved_ = false;
  base::Sha256Digest boundary_{};
  base::Sha256Digest recipe_{};
  ComparisonRegionRecipeSeal seal_{};
};

// Expansion members borrow program-owned, preallocated storage and remain
// valid only until the next projection request or program destruction.
// Closed capabilities own no heap memory and cannot be created by callers.
struct ComparisonRegionResult {
  ComparisonRegionStatus status = ComparisonRegionStatus::kUnproved;
  std::optional<ComparisonRegionProjection> projection{};
  std::span<const ir::NodeId> required_members{};
  // Program-owned, preallocated port inventory. It is valid only while the
  // matching capability's program/generation/serial remains current.
  std::span<const ComparisonRegionValuePort> value_ports{};
  std::uint64_t program_nonce = 0;
  std::uint64_t generation = 0;
  std::uint64_t serial = 0;
  std::string_view reason{};
};

// Descriptor-only region projection for exact pre-admission selection. Its
// expansion list borrows paid program-owned storage until the next projection
// request or program destruction. It carries no executable authority and
// publishes no device addresses.
struct ComparisonRegionDescriptorResult {
  ComparisonRegionStatus status = ComparisonRegionStatus::kUnproved;
  ComparisonRegionRecipeSeal seal{};
  std::span<const ir::NodeId> required_members{};
  std::string_view reason{};
};

}  // namespace llmp::kernels

#endif  // LLMP_KERNELS_COMMON_COMPARISON_REGION_H_
