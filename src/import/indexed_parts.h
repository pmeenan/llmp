// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#ifndef LLMP_IMPORT_INDEXED_PARTS_H_
#define LLMP_IMPORT_INDEXED_PARTS_H_

#include <cstddef>
#include <cstdint>
#include <expected>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "import/checkpoint.h"
#include "import/route.h"

namespace llmp::import {
struct SemanticReceipt {
  std::uint32_t source = 0;
  Extent extent{};
  base::Sha256Digest sha256{};
};

// Canonical tensor index is the position in this vector. This explicit join
// lets a generic publisher consume geometry without interpreting source names.
struct NormalizedPartBinding {
  std::uint32_t output = 0;
  format::PartRole part = format::PartRole::kData;
};

// A newly owned transform; publication records it only after all authenticated routes finish.
struct PlannedPartFold {
  std::string fold{};
  std::vector<std::uint32_t> outputs{};
};

// Bytes the importer derives from one authenticated whole source file and
// writes as a canonical tensor's part (a selected head's token map, I32,
// from its list). `tensor` follows every bound canonical tensor.
struct GeneratedPart {
  std::uint32_t tensor = 0;
  std::uint32_t file = 0;
  std::vector<std::byte> bytes{};
};

// Geometry only: a model mapping must independently authorize each source
// role and exclusion. Neither these declarations nor a successful plan grant
// model admission, authenticate payloads or publish an artifact.
enum class PartAssemblyKind : std::uint8_t {
  kCopy,
  kDenseReshape,
  kRows,
  kMatrices,
  kSharedTensorScale,
  // One F32 element of a dense [sources] vector per source: each source's
  // 4-byte F32 tensor-scale part, in order (a routed-expert table's
  // per-matrix tensor scales gathered beside its relaid-out table).
  kScalars,
  // A row-interleaved table's data from split row blocks (a relayout from
  // kSplit): sources are pairs, a block's codes then its block scales, each
  // row's codes and scales side by side in the output (PartRoutePlan::
  // Strided); the blocks' rows in order. Its tensor scale is shared.
  kInterleavedRows
};
enum class SourceDispositionKind : std::uint8_t { kRouted, kSemantic, kAuxiliary };
struct SourceDisposition {
  SourceDispositionKind kind = SourceDispositionKind::kRouted;
  // Nonzero mapping rule ID for semantic/auxiliary entries; zero for routed.
  std::uint32_t rule = 0;
};
struct PartAssembly {
  format::PartRole part = format::PartRole::kData;
  PartAssemblyKind kind = PartAssemblyKind::kCopy;
  // Numeric output order, deliberately independent of file/name order.
  std::vector<std::uint32_t> sources{};
};
struct TensorAssembly {
  std::string name{};
  format::TensorFormat format{};
  std::vector<PartAssembly> parts{};
  // One shared permutation for every part; only single-source copy/reshape.
  // Must describe this exact output format.
  std::optional<HeadPermutationPlan> permutation{};
  // A scalar fold of one dense BF16 source into this dense F32 output of the
  // same shape (a single kCopy part; with a permutation, the reorder of its
  // BF16 source, PartRoutePlan::PermuteFold): FoldBf16's exact conversion
  // (PartRoutePlan::Fold). The importer that asks for it records the
  // applied fold (D-108 as amended, D-113).
  std::optional<ScalarFold> fold{};
  // A lossless relayout (D-108, D-113) from `relayout_from` (kSplit): a
  // kMatrices assembly in format::Layout::kScaleAtoms takes format.stacked
  // source matrices a slice, in order (gate then up), at the same bytes:
  // codes copied, block scales by PartRoutePlan::ScaleAtoms, the sources'
  // tensor scales not this output's (kScalars outputs); or a row table's
  // kInterleavedRows assembly in format::Layout::kRowInterleaved.
  std::optional<format::Layout> relayout_from{};
};
struct IndexedPartCost {
  std::uint64_t metadata_bytes = 0;
  std::uint64_t work_units = 0;
  std::uint64_t members = 0;
};
struct IndexedOutput {
  std::string name{};
  format::TensorFormat format{};
  format::PartList parts{};
  // The layout its sources held when it is a relayout (the writer records
  // it as an applied relayout, D-113).
  std::optional<format::Layout> relayout_from{};
};
struct IndexedMember {
  std::uint32_t source = 0;
  std::uint32_t output = 0;
  format::PartRole part = format::PartRole::kData;
  Extent extent{};
  std::uint64_t destination = 0;
  PartRoutePlan route;
};

// An owned, complete source-to-part assembly. Source tensor metadata remains
// byte-for-byte intact. All routes are encoding-preserving and allocation-free
// at execution. Output offsets are part-relative; the writer adds destination
// to each route sink offset. It must authenticate every source and acknowledge
// every member before any publication. Readable padding is not source payload.
class IndexedPartPlan {
 public:
  IndexedPartPlan(const IndexedPartPlan&) = delete;
  IndexedPartPlan& operator=(const IndexedPartPlan&) = delete;
  IndexedPartPlan(IndexedPartPlan&&) noexcept = default;
  IndexedPartPlan& operator=(IndexedPartPlan&&) noexcept = default;
  // Allocation-free preflight. Bounds this planner's simultaneous allocations,
  // including sort metadata and string allocations. Caller separately funds
  // the already-built Checkpoint and borrowed declarations through Make; they
  // are not hidden in this cost. Caps: kMaxTensors entries, 256 files.
  static std::expected<IndexedPartCost, Error> Bound(const Checkpoint& source,
                                                     std::span<const TensorAssembly> outputs);
  // Refusal before admission leaves the caller's source ownership intact.
  // No implicit deep copy can precede prepayment.
  static std::expected<IndexedPartPlan, Error> Make(Checkpoint&& source,
                                                    std::span<const TensorAssembly> outputs,
                                                    std::span<const SourceDisposition> dispositions,
                                                    IndexedPartCost prepaid);

  const Checkpoint& source() const { return source_; }
  std::span<const IndexedOutput> outputs() const { return outputs_; }
  std::span<const IndexedMember> members() const { return members_; }
  std::span<const SourceDisposition> dispositions() const { return dispositions_; }
  // Member indices ordered by physical file and byte offset.
  std::span<const std::uint32_t> physical_order() const { return physical_order_; }
  std::optional<std::uint32_t> FindSource(std::string_view name) const;
  IndexedPartCost cost() const { return cost_; }

 private:
  IndexedPartPlan() = default;
  Checkpoint source_{};
  std::vector<IndexedOutput> outputs_{};
  std::vector<IndexedMember> members_{};
  std::vector<SourceDisposition> dispositions_{};
  std::vector<std::uint32_t> names_{}, physical_order_{};
  IndexedPartCost cost_{};
};
}  // namespace llmp::import
#endif  // LLMP_IMPORT_INDEXED_PARTS_H_
