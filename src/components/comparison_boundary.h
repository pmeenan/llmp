// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef LLMP_COMPONENTS_COMPARISON_BOUNDARY_H_
#define LLMP_COMPONENTS_COMPARISON_BOUNDARY_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "base/sha256.h"
#include "components/graph_io.h"
#include "ir/storage_admission.h"

namespace llmp::components {

// Persistent comparison identity treats caller-facing inputs, internal
// activations, and returned values as one value class. Live role facts remain
// untouched and continue to participate in validation.
enum class ComparisonPersistentRole : std::uint8_t {
  kValue,
  kWeight,
  kState,
  kConstant,
};

constexpr ComparisonPersistentRole CanonicalComparisonRole(ir::Role role) noexcept {
  switch (role) {
    case ir::Role::kInput:
    case ir::Role::kActivation:
    case ir::Role::kOutput:
      return ComparisonPersistentRole::kValue;
    case ir::Role::kWeight:
      return ComparisonPersistentRole::kWeight;
    case ir::Role::kState:
      return ComparisonPersistentRole::kState;
    case ir::Role::kConstant:
      return ComparisonPersistentRole::kConstant;
  }
  return ComparisonPersistentRole::kValue;
}

inline constexpr std::uint32_t kComparisonRegionCanonicalizerVersion = 2;
inline constexpr std::uint32_t kComparisonFrameworkComponent =
    std::numeric_limits<std::uint16_t>::max();

// Static provenance for an owner-backed row-table slot mapping. The actual
// virtual slot address and byte range remain in BoundedAddress; this value
// describes the logical artifact slice and its checked row representation.
struct ComparisonRowStorage {
  ir::WeightRef ref{};
  format::PartRole part = format::PartRole::kData;
  format::TensorFormat logical_format{};
  base::Sha256Digest content_identity{};
  std::uint32_t resource = 0;
  std::uint32_t group = 0;
  std::uint32_t shard = 0;
  std::uint64_t group_bytes = 0;
  std::uint64_t resource_offset = 0;
  std::uint64_t resource_bytes = 0;
  std::uint64_t logical_rows = 0;
  std::uint64_t slot_capacity = 0;
  std::uint64_t row_bytes = 0;
  bool operator==(const ComparisonRowStorage&) const = default;
};

// Shared portable validation for owner issuance, source bindings and backend
// physical-leaf checks. `physical_bytes` must be the exact capacity-sized slot
// extent, never the source table's full logical extent.
bool ValidComparisonRowStorage(const ComparisonRowStorage& storage, const ir::WeightRef& ref,
                               format::PartRole part, const format::TensorFormat& logical_format,
                               std::uint64_t physical_bytes) noexcept;

// Caller-owned facts about an original external allocation. Class labels are
// arbitrary input tokens; the owned boundary canonicalizes their partition.
struct ComparisonExternalBinding {
  ir::TensorId original_root{};
  std::optional<format::PartRole> part{};
  std::uint32_t allocation_class = 0;
  std::uint64_t allocation_bytes = 0;
  std::uint64_t offset = 0;
  std::uint64_t bytes = 0;
  std::uint64_t alignment = 0;
  std::optional<base::Sha256Digest> resource_part_identity{};
  std::optional<ComparisonRowStorage> row_storage{};
};

// A source input whose values have an authenticated non-storage realization.
// This is intentionally a closed enum: callers cannot declare arbitrary
// omitted inputs or provide a synthetic address for them.
enum class ComparisonUnstoredInputReason : std::uint8_t {
  kCellPositionsFoldedIntoMasks,
};

struct ComparisonUnstoredInput {
  ir::TensorId original_input{};
  ComparisonUnstoredInputReason reason =
      ComparisonUnstoredInputReason::kCellPositionsFoldedIntoMasks;
};

struct ComparisonBoundaryUnstoredInput {
  std::uint32_t canonical_root = 0;
  ir::TensorId original_input{};  // lookup key, never hashed
  ComparisonUnstoredInputReason reason =
      ComparisonUnstoredInputReason::kCellPositionsFoldedIntoMasks;
};

struct ComparisonBoundaryView {
  GraphBuildIO io;
  std::span<const StepState> state{};
  std::uint16_t active_stage = 0;
  std::span<const ComparisonExternalBinding> external{};
  std::span<const ComparisonUnstoredInput> unstored_inputs{};
};

struct ComparisonBoundaryDigest {
  std::uint32_t version = 2;
  base::Sha256Digest logical{};
  base::Sha256Digest external{};
  base::Sha256Digest combined{};
};

struct ComparisonBoundaryError {
  enum Code : std::uint8_t { kInvalid, kAdmission } code = kInvalid;
  std::string detail;
};

enum class ComparisonPortKind : std::uint8_t { kInput, kOutput, kKeep };

struct ComparisonBoundaryRoot {
  std::uint32_t canonical = 0;
  ir::TensorId original_root{};  // lookup key, never hashed
  ir::Role role = ir::Role::kActivation;
  format::TensorFormat format{};
  std::array<std::uint64_t, format::kMaxDims> strides{};
  std::uint64_t storage_bytes = 0;
  base::Sha256Digest constant_content{};
  ir::Binding binding{};
  std::uint8_t segment = ir::kJoinedSegment;
};

struct ComparisonBoundaryPort {
  ComparisonPortKind kind = ComparisonPortKind::kInput;
  std::uint32_t ordinal = 0;
  std::uint32_t canonical_root = 0;
  ir::TensorId original_tensor{};  // lookup key, never hashed
  ir::Role role = ir::Role::kActivation;
  format::TensorFormat format{};
  std::array<std::uint64_t, format::kMaxDims> strides{};
  std::uint64_t root_offset = 0;
  std::uint64_t storage_bytes = 0;
  ir::Binding binding{};
  std::uint8_t segment = ir::kJoinedSegment;
};

struct ComparisonBoundaryWeightPart {
  std::uint32_t canonical_root = 0;
  ir::TensorId original_root{};  // lookup key, never hashed
  ir::WeightRef ref{};
  std::string name;
  format::PartRole part = format::PartRole::kData;
};

struct ComparisonBoundaryState {
  std::uint32_t canonical_root = 0;
  StepState facts{};
};

struct ComparisonBoundaryExternal {
  std::uint32_t canonical_root = 0;
  std::optional<format::PartRole> part{};
  std::uint32_t canonical_allocation_class = 0;
  std::uint64_t allocation_bytes = 0;
  std::uint64_t offset = 0;
  std::uint64_t bytes = 0;
  std::uint64_t alignment = 0;
  std::optional<base::Sha256Digest> resource_part_identity{};
  std::optional<ComparisonRowStorage> row_storage{};
};

// A region is an ordered collection of source-member groups. `member_offsets`
// has group_count+1 entries and slices `members`; group order and each group's
// member order are semantic, while source node IDs are lookup keys only.
struct ComparisonRegionBoundaryView {
  ComparisonBoundaryView source;
  std::span<const std::uint32_t> member_offsets;
  std::span<const ir::NodeId> members;
};

struct ComparisonRegionNode {
  std::uint32_t canonical = 0;
  std::uint32_t group = 0;
  ir::NodeId original_node{};  // occurrence lookup key, never hashed
  ir::OpKey key{};
  std::uint16_t component = 0;
  ir::Batch batch = ir::Batch::kJoined;
  std::array<std::byte, ir::kMaxParamBytes> params{};
  std::array<std::uint32_t, ir::kMaxSources> sources{};
  std::array<std::uint32_t, ir::kMaxOutputs> outputs{};
  std::uint8_t nsrc = 0;
  std::uint8_t nout = 0;
};

// Zero-based first-occurrence component identity shared by portable and
// backend recipe serializers. The framework sentinel remains unchanged;
// invalid indices return UINT32_MAX. No allocation is performed.
inline std::uint32_t CanonicalComparisonComponent(std::span<const ComparisonRegionNode> nodes,
                                                  std::size_t index) noexcept {
  if (index >= nodes.size()) return UINT32_MAX;
  const auto value = nodes[index].component;
  if (value == kComparisonFrameworkComponent) return kComparisonFrameworkComponent;
  std::uint32_t next = 0;
  for (std::size_t i = 0; i <= index; ++i) {
    const auto candidate = nodes[i].component;
    if (candidate == kComparisonFrameworkComponent) continue;
    bool seen = false;
    for (std::size_t prior = 0; prior < i; ++prior)
      if (nodes[prior].component == candidate) {
        seen = true;
        break;
      }
    if (!seen) {
      if (candidate == value) return next;
      ++next;
    }
  }
  return UINT32_MAX;
}

struct ComparisonRegionTensor {
  std::uint32_t canonical = 0;
  std::uint32_t canonical_root = 0;
  ir::Role role = ir::Role::kActivation;
  format::TensorFormat format{};
  std::array<std::uint64_t, format::kMaxDims> strides{};
  std::uint64_t view_offset = 0;
  ir::Binding binding{};
  std::uint8_t segment = ir::kJoinedSegment;
  std::uint64_t storage_bytes = 0;
};

struct ComparisonRegionRoot {
  std::uint32_t canonical = 0;
  ir::TensorId original_root{};  // lookup key, never hashed
  ir::Role role = ir::Role::kActivation;
  format::TensorFormat format{};
  std::array<std::uint64_t, format::kMaxDims> strides{};
  std::uint64_t storage_bytes = 0;
  base::Sha256Digest constant_content{};
  ir::Binding binding{};
  std::uint8_t segment = ir::kJoinedSegment;
};

enum class ComparisonRegionPortKind : std::uint8_t {
  kInput,
  kOutput,
  kKeep,
  kStateRead,
  kStateWrite
};

struct ComparisonRegionPort {
  ComparisonRegionPortKind kind = ComparisonRegionPortKind::kInput;
  std::uint32_t member = 0;
  std::uint16_t ordinal = 0;
  std::uint32_t canonical_tensor = 0;
  std::uint32_t canonical_root = 0;
  ir::TensorId original_tensor{};  // lookup key, never hashed
  std::uint64_t view_offset = 0;
};

struct ComparisonRegionWeightPart {
  std::uint32_t canonical_root = 0;
  ir::WeightRef ref{};
  std::string name;
  format::PartRole part = format::PartRole::kData;
};

struct ComparisonRegionState {
  std::uint32_t canonical_root = 0;
  StepState facts{};
};

struct ComparisonRegionExternal {
  std::uint32_t canonical_root = 0;
  std::optional<format::PartRole> part{};
  std::uint32_t canonical_allocation_class = 0;
  std::uint64_t allocation_bytes = 0;
  std::uint64_t offset = 0;
  std::uint64_t bytes = 0;
  std::uint64_t alignment = 0;
  std::optional<base::Sha256Digest> resource_part_identity{};
  std::optional<ComparisonRowStorage> row_storage{};
};

class OwnedComparisonRegionBoundary {
 public:
  OwnedComparisonRegionBoundary() = default;
  OwnedComparisonRegionBoundary(OwnedComparisonRegionBoundary&& other) noexcept
      : loan_(std::move(other.loan_)),
        digest_(other.digest_),
        source_identity_(std::exchange(other.source_identity_, 0)),
        active_stage_(std::exchange(other.active_stage_, 0)),
        nodes_(std::move(other.nodes_)),
        roots_(std::move(other.roots_)),
        tensors_(std::move(other.tensors_)),
        ports_(std::move(other.ports_)),
        weight_parts_(std::move(other.weight_parts_)),
        state_(std::move(other.state_)),
        external_(std::move(other.external_)) {
    other.digest_ = {};
  }
  OwnedComparisonRegionBoundary& operator=(OwnedComparisonRegionBoundary&&) noexcept = delete;
  OwnedComparisonRegionBoundary(const OwnedComparisonRegionBoundary&) = delete;
  OwnedComparisonRegionBoundary& operator=(const OwnedComparisonRegionBoundary&) = delete;
  const ComparisonBoundaryDigest& digest() const { return digest_; }
  std::uint64_t source_identity() const { return source_identity_; }
  std::uint16_t active_stage() const { return active_stage_; }
  std::uint64_t storage_bytes() const { return loan_.bytes(); }
  // Transfers the live vector storage to an enclosing externally paid owner.
  // The caller must keep that owner alive until this boundary is destroyed.
  void ConstructionPaid() { loan_.Reset(); }
  std::span<const ComparisonRegionNode> nodes() const { return nodes_; }
  std::span<const ComparisonRegionRoot> roots() const { return roots_; }
  std::span<const ComparisonRegionTensor> tensors() const { return tensors_; }
  std::span<const ComparisonRegionPort> ports() const { return ports_; }
  std::span<const ComparisonRegionWeightPart> weight_parts() const { return weight_parts_; }
  std::span<const ComparisonRegionState> state() const { return state_; }
  std::span<const ComparisonRegionExternal> external() const { return external_; }

 private:
  friend std::expected<OwnedComparisonRegionBoundary, ComparisonBoundaryError>
  MakeComparisonRegionBoundary(const ComparisonRegionBoundaryView&,
                               std::shared_ptr<const ir::StorageFunder>, ir::DiagnosticAdmission*);
  ir::StorageLoan loan_{};
  ComparisonBoundaryDigest digest_{};
  std::uint64_t source_identity_ = 0;
  std::uint16_t active_stage_ = 0;
  std::vector<ComparisonRegionNode> nodes_;
  std::vector<ComparisonRegionRoot> roots_;
  std::vector<ComparisonRegionTensor> tensors_;
  std::vector<ComparisonRegionPort> ports_;
  std::vector<ComparisonRegionWeightPart> weight_parts_;
  std::vector<ComparisonRegionState> state_;
  std::vector<ComparisonRegionExternal> external_;
};

class OwnedComparisonBoundary {
 public:
  OwnedComparisonBoundary() = default;
  OwnedComparisonBoundary(OwnedComparisonBoundary&& other) noexcept;
  OwnedComparisonBoundary& operator=(OwnedComparisonBoundary&&) noexcept = delete;
  OwnedComparisonBoundary(const OwnedComparisonBoundary&) = delete;
  OwnedComparisonBoundary& operator=(const OwnedComparisonBoundary&) = delete;

  const ComparisonBoundaryDigest& digest() const { return digest_; }
  const base::Sha256Digest& source_digest() const { return source_digest_; }
  std::uint64_t source_identity() const { return source_identity_; }
  std::uint32_t source_tensor_count() const { return source_tensor_count_; }
  std::uint32_t source_input_count() const { return source_input_count_; }
  std::uint32_t source_output_count() const { return source_output_count_; }
  std::uint32_t source_keep_count() const { return source_keep_count_; }
  std::uint16_t active_stage() const { return active_stage_; }
  std::span<const ComparisonBoundaryRoot> roots() const { return roots_; }
  std::span<const ComparisonBoundaryPort> ports() const { return ports_; }
  std::span<const ComparisonBoundaryWeightPart> weight_parts() const { return weight_parts_; }
  std::span<const ComparisonBoundaryState> state() const { return state_; }
  std::span<const StepState> state_facts() const { return state_facts_; }
  std::span<const ComparisonBoundaryExternal> external() const { return external_; }
  std::span<const ComparisonBoundaryUnstoredInput> unstored_inputs() const {
    return unstored_inputs_;
  }

 private:
  friend std::expected<OwnedComparisonBoundary, ComparisonBoundaryError> MakeComparisonBoundary(
      const ComparisonBoundaryView&, std::shared_ptr<const ir::StorageFunder>,
      ir::DiagnosticAdmission*);

  // Loan precedes vectors so vectors are destroyed before their admission.
  ir::StorageLoan loan_{};
  ComparisonBoundaryDigest digest_{};
  base::Sha256Digest source_digest_{};
  std::uint64_t source_identity_ = 0;  // identity token only; never hashed
  std::uint32_t source_tensor_count_ = 0;
  std::uint32_t source_input_count_ = 0;
  std::uint32_t source_output_count_ = 0;
  std::uint32_t source_keep_count_ = 0;
  std::uint16_t active_stage_ = 0;
  std::vector<ComparisonBoundaryRoot> roots_;
  std::vector<ComparisonBoundaryPort> ports_;
  std::vector<ComparisonBoundaryWeightPart> weight_parts_;
  std::vector<ComparisonBoundaryState> state_;
  std::vector<StepState> state_facts_;
  std::vector<ComparisonBoundaryExternal> external_;
  std::vector<ComparisonBoundaryUnstoredInput> unstored_inputs_;
};

std::expected<std::uint64_t, ComparisonBoundaryError> ComparisonBoundaryStorageBound(
    const ComparisonBoundaryView& view, ir::DiagnosticAdmission* diagnostic = nullptr);
std::expected<OwnedComparisonBoundary, ComparisonBoundaryError> MakeComparisonBoundary(
    const ComparisonBoundaryView& view, std::shared_ptr<const ir::StorageFunder> funder,
    ir::DiagnosticAdmission* diagnostic = nullptr);
std::expected<void, ComparisonBoundaryError> ValidateMappedComparisonBoundary(
    const OwnedComparisonBoundary& source, const ir::Graph& planned,
    std::span<const ir::TensorId> original_to_planned,
    ir::DiagnosticAdmission* diagnostic = nullptr);
std::expected<OwnedComparisonRegionBoundary, ComparisonBoundaryError> MakeComparisonRegionBoundary(
    const ComparisonRegionBoundaryView& view, std::shared_ptr<const ir::StorageFunder> funder,
    ir::DiagnosticAdmission* diagnostic = nullptr);

}  // namespace llmp::components

#endif  // LLMP_COMPONENTS_COMPARISON_BOUNDARY_H_
