// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#ifndef LLMP_IMPORT_NORMALIZE_H_
#define LLMP_IMPORT_NORMALIZE_H_

#include <cstddef>
#include <cstdint>
#include <expected>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "import/checkpoint.h"
#include "import/source_mapping.h"
#include "spec/spec.h"

namespace llmp::import {
// Caller-owned, validated artifact target. Normalization never manufactures a
// target checkpoint profile; the target's stored spec and tensors are authoritative.
struct NormalizationTarget {
  const spec::Mapping* mapping = nullptr;
  spec::ModelSpec model;
  spec::Source source;
};
std::expected<void, Error> ValidateTarget(const NormalizationTarget& target,
                                          const spec::Mapping& drafter);

struct NormalizationLimits {
  // New requested container storage and conservative byte-comparison work.
  // The caller separately owns/funds the moved Checkpoint. Neither is a
  // process admission or a promise that allocation cannot fail.
  std::uint64_t metadata_bytes = 0;
  std::uint64_t work_units = 0;
};
struct NormalizationCost {
  std::uint64_t metadata_bytes = 0, work_units = 0, payload_bytes = 0;
  std::uint32_t layers = 0;
};
std::expected<NormalizationCost, Error> NormalizationBound(const Checkpoint& source);

enum class MetadataClass : std::uint8_t { kSelected, kStructural, kAuxiliary };
struct MetadataUse {
  std::uint32_t source = 0;
  MetadataClass classification = MetadataClass::kAuxiliary;
  // Selected: index in values(); auxiliary: index in auxiliary_namespaces().
  std::uint32_t destination = 0;
};
struct CopyRoute {
  std::uint32_t source_tensor = 0;
  Extent extent{};
  std::uint64_t readable = 0;  // output padding obligation; never source bytes
};
struct SourceCopy {
  std::uint32_t file = 0;
  std::uint64_t source_offset = 0, output_offset = 0, bytes = 0;
};
class SourceCopyCursor {
 public:
  std::expected<std::optional<SourceCopy>, Error> Next(std::uint64_t maximum_bytes);

 private:
  friend class NormalizedPlan;
  explicit SourceCopyCursor(Extent extent) : extent_(extent) {}
  Extent extent_{};
  std::uint64_t next_ = 0;
};
struct AppliedSourceFold {
  std::string fold{};
  // An assertion of the compiled container mapping, never a transform command.
  std::string container{};
};
struct OwnedAuxiliaryNamespace {
  std::string prefix{}, reason{};
};
class NormalizedPlan;
namespace normalization_detail {
// Internal construction seam for compiled-table tests. The public entry selects
// only FindSourceMapping. The same validator and interpreter serve both.
std::expected<NormalizationCost, Error> Bound(const Checkpoint&, const SourceMapping&);
std::expected<NormalizedPlan, Error> Build(Checkpoint, const SourceMapping&, NormalizationLimits,
                                           const NormalizationTarget* = nullptr);
}  // namespace normalization_detail

class NormalizedPlan {
 public:
  NormalizedPlan(NormalizedPlan&&) noexcept = default;
  NormalizedPlan& operator=(NormalizedPlan&&) noexcept = default;
  NormalizedPlan(const NormalizedPlan&) = delete;
  NormalizedPlan& operator=(const NormalizedPlan&) = delete;
  const Checkpoint& source() const { return source_; }
  const spec::Source& canonical() const { return canonical_; }
  const spec::ModelSpec& spec() const { return spec_; }
  std::span<const spec::SourceValue> values() const { return values_; }
  std::span<const MetadataUse> metadata() const { return metadata_; }
  std::span<const OwnedAuxiliaryNamespace> auxiliary_namespaces() const { return auxiliary_; }
  std::span<const AppliedSourceFold> applied_folds() const { return folds_; }
  std::span<const CopyRoute> routes() const { return routes_; }
  // Canonical output indices sorted by physical source file/offset.
  std::span<const std::uint32_t> source_order() const { return order_; }
  std::uint16_t normalization_version() const { return version_; }
  std::uint16_t architecture_version() const { return architecture_version_; }
  const NormalizationCost& cost() const { return cost_; }
  std::expected<SourceCopyCursor, Error> CursorFor(std::size_t output) const;

 private:
  friend std::expected<NormalizedPlan, Error> normalization_detail::Build(
      Checkpoint, const SourceMapping&, NormalizationLimits, const NormalizationTarget*);
  NormalizedPlan() = default;
  Checkpoint source_{};
  spec::Source canonical_{};
  spec::ModelSpec spec_{};
  std::vector<spec::SourceValue> values_{};
  std::vector<MetadataUse> metadata_{};
  std::vector<OwnedAuxiliaryNamespace> auxiliary_{};
  std::vector<AppliedSourceFold> folds_{};
  std::vector<CopyRoute> routes_{};
  std::vector<std::uint32_t> order_{};
  std::uint16_t version_ = 0, architecture_version_ = 0;
  NormalizationCost cost_{};
};
// Consumes metadata only. No payload reads, writes, transformation, output
// artifact, or successful-import receipt is implied by a returned plan.
std::expected<NormalizedPlan, Error> Normalize(Checkpoint source, NormalizationLimits limits,
                                               const NormalizationTarget* target = nullptr);
}  // namespace llmp::import
#endif  // LLMP_IMPORT_NORMALIZE_H_
