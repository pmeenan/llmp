// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#ifndef LLMP_IMPORT_NORMALIZE_H_
#define LLMP_IMPORT_NORMALIZE_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "base/sha256.h"
#include "import/checkpoint.h"
#include "import/indexed_parts.h"
#include "import/source_mapping.h"
#include "spec/spec.h"

namespace llmp::test_support {
class PartPublicationFixture;
}

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

// A selected draft head's token list (`llmp import --draft-head`): one
// decimal token ID a line, strictly ascending (the order keeps argmax's
// tie-breaking), each below the vocabulary; at most `vocab` lines. Read
// whole as untrusted input; the file becomes one of the drafter's sources,
// its whole content its parsed receipt, so publication authenticates the
// bytes these IDs were read from.
struct DraftHeadList {
  SourceFile file{};
  std::vector<std::int32_t> ids{};
};
std::expected<DraftHeadList, Error> ReadDraftHeadList(const std::filesystem::path& path,
                                                      std::uint32_t vocab);

struct NormalizeBoundOptions {
  std::string_view component{};
  std::size_t draft_head_rows = 0;
};
struct NormalizeOptions {
  std::string component{};
  std::optional<DraftHeadList> draft_head{};
};

// Where a component's tensors are stored when the graph's bindings do not
// state their group (spatial component recipes derive this from their nodes).
struct ComponentGroup {
  enum class Kind : std::uint8_t { kTable, kGlobal, kLayer, kHead };
  Kind kind = Kind::kGlobal;
  std::uint32_t layer = 0;
};

std::expected<NormalizationCost, Error> NormalizationBound(const Checkpoint& source,
                                                           NormalizeBoundOptions options = {});

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
// only FindImportMapping. The same validator and interpreter serve both.
std::expected<NormalizationCost, Error> Bound(const Checkpoint&, const spec::Mapping&,
                                              NormalizeBoundOptions = {});
std::expected<NormalizedPlan, Error> Build(Checkpoint&&, const spec::Mapping&, NormalizationLimits,
                                           const NormalizationTarget* = nullptr,
                                           NormalizeOptions = {});
std::expected<NormalizedPlan, Error> Build(const Checkpoint&, const spec::Mapping&,
                                           NormalizationLimits,
                                           const NormalizationTarget* = nullptr,
                                           NormalizeOptions = {});
}  // namespace normalization_detail

class NormalizedPlan {
 public:
  NormalizedPlan(NormalizedPlan&&) noexcept = default;
  NormalizedPlan& operator=(NormalizedPlan&&) noexcept = default;
  NormalizedPlan(const NormalizedPlan&) = delete;
  NormalizedPlan& operator=(const NormalizedPlan&) = delete;
  const Checkpoint& source() const { return parts_ ? parts_->source() : source_; }
  bool is_indexed() const { return parts_.has_value(); }
  const IndexedPartPlan& parts() const { return *parts_; }
  const spec::Source& canonical() const { return canonical_; }
  const spec::ModelSpec& spec() const { return spec_; }
  std::span<const spec::SourceValue> values() const { return values_; }
  std::span<const MetadataUse> metadata() const { return metadata_; }
  std::span<const OwnedAuxiliaryNamespace> auxiliary_namespaces() const { return auxiliary_; }
  std::span<const AppliedSourceFold> applied_folds() const { return folds_; }
  std::span<const CopyRoute> routes() const { return routes_; }
  std::span<const ComponentGroup> groups() const { return groups_; }
  // Canonical output indices sorted by physical source file/offset.
  std::span<const std::uint32_t> source_order() const { return order_; }
  std::span<const NormalizedPartBinding> bindings() const { return bindings_; }
  std::span<const SemanticReceipt> semantic() const { return semantic_; }
  std::span<const PlannedPartFold> planned_folds() const { return planned_folds_; }
  std::span<const GeneratedPart> generated() const { return generated_; }
  std::uint64_t selected_rows() const { return selected_rows_; }
  const std::string& options() const { return options_; }
  bool has_options() const { return has_options_; }
  std::string_view mapping_architecture() const {
    return mapping_architecture_.empty() ? std::string_view(canonical_.architecture)
                                         : std::string_view(mapping_architecture_);
  }
  std::uint16_t normalization_version() const { return architecture_version_; }
  std::uint16_t architecture_version() const { return architecture_version_; }
  const NormalizationCost& cost() const { return cost_; }
  std::expected<SourceCopyCursor, Error> CursorFor(std::size_t output) const;

 private:
  friend class llmp::test_support::PartPublicationFixture;
  friend std::expected<NormalizedPlan, Error> normalization_detail::Build(
      Checkpoint&&, const spec::Mapping&, NormalizationLimits, const NormalizationTarget*,
      NormalizeOptions);
  friend std::expected<NormalizedPlan, Error> normalization_detail::Build(
      const Checkpoint&, const spec::Mapping&, NormalizationLimits, const NormalizationTarget*,
      NormalizeOptions);
  NormalizedPlan() = default;
  Checkpoint source_{};
  std::optional<IndexedPartPlan> parts_{};
  spec::Source canonical_{};
  spec::ModelSpec spec_{};
  std::vector<spec::SourceValue> values_{};
  std::vector<MetadataUse> metadata_{};
  std::vector<OwnedAuxiliaryNamespace> auxiliary_{};
  std::vector<AppliedSourceFold> folds_{};
  std::vector<CopyRoute> routes_{};
  std::vector<ComponentGroup> groups_{};
  std::vector<std::uint32_t> order_{};
  std::vector<NormalizedPartBinding> bindings_{};
  std::vector<SemanticReceipt> semantic_{};
  std::vector<PlannedPartFold> planned_folds_{};
  std::vector<GeneratedPart> generated_{};
  std::uint64_t selected_rows_ = 0;
  std::string options_{};
  bool has_options_ = false;
  std::string mapping_architecture_{};
  std::uint16_t architecture_version_ = 0;
  NormalizationCost cost_{};
};

// Consumes metadata and explicitly declared authenticated semantic inputs. No
// payload writes or successful-import receipt is implied by a returned plan.
std::expected<NormalizedPlan, Error> Normalize(Checkpoint&& source, NormalizationLimits limits,
                                               const NormalizationTarget* target = nullptr,
                                               NormalizeOptions options = {});
std::expected<NormalizedPlan, Error> Normalize(const Checkpoint& source, NormalizationLimits limits,
                                               const NormalizationTarget* target = nullptr,
                                               NormalizeOptions options = {});
}  // namespace llmp::import
#endif  // LLMP_IMPORT_NORMALIZE_H_
