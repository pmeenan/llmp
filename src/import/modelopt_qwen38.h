// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#ifndef LLMP_IMPORT_MODELOPT_QWEN38_H_
#define LLMP_IMPORT_MODELOPT_QWEN38_H_
#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "base/sha256.h"
#include "import/indexed_parts.h"
#include "import/normalize.h"

namespace llmp::test_support {
class PartPublicationFixture;
}

namespace llmp::import {
// The native target's canonical source and values as NormalizeModelOptTarget
// gives them, without reading a checkpoint: the target a drafter's spec
// composes with at import, and admission's reference.
struct ModelOptTargetForm {
  spec::Source source{};
  std::vector<spec::SourceValue> values{};
};
std::expected<ModelOptTargetForm, Error> ModelOptNativeTargetForm();
class ModelOptTargetPlan;
// Metadata-only, allocation-free admission for the closed source domain.
// Caller separately owns/funds Checkpoint and its reader before calling this.
std::expected<NormalizationCost, Error> ModelOptTargetBound(const Checkpoint& source);
// Reads exactly 280 bytes after admission through authenticated source identities.
// No full payload hashing, publication, runtime admission or vision import; the
// MTP drafter is NormalizeModelOptDrafter's, below.
std::expected<ModelOptTargetPlan, Error> NormalizeModelOptTarget(Checkpoint&& source,
                                                                 NormalizationLimits prepaid);
// Unified dispatch for indexed target normalizers (ModelOpt and MLX).
bool IsIndexedTarget(const Checkpoint& source);
std::expected<NormalizationCost, Error> IndexedTargetBound(const Checkpoint& source);
std::expected<ModelOptTargetPlan, Error> NormalizeIndexedTarget(Checkpoint&& source,
                                                                NormalizationLimits prepaid);
class ModelOptTargetPlan {
 public:
  ModelOptTargetPlan(const ModelOptTargetPlan&) = delete;
  ModelOptTargetPlan& operator=(const ModelOptTargetPlan&) = delete;
  ModelOptTargetPlan(ModelOptTargetPlan&&) noexcept = default;
  ModelOptTargetPlan& operator=(ModelOptTargetPlan&&) noexcept = default;
  const IndexedPartPlan& parts() const { return *parts_; }
  const spec::Source& canonical() const { return canonical_; }
  const spec::ModelSpec& spec() const { return spec_; }
  std::span<const spec::SourceValue> values() const { return values_; }
  std::span<const NormalizedPartBinding> bindings() const { return bindings_; }
  std::span<const SemanticReceipt> semantic() const { return semantic_; }
  std::span<const PlannedPartFold> planned_folds() const { return planned_folds_; }
  NormalizationCost cost() const { return cost_; }

 private:
  friend class llmp::test_support::PartPublicationFixture;
  friend std::expected<ModelOptTargetPlan, Error> NormalizeModelOptTarget(
      Checkpoint&& source, NormalizationLimits prepaid);
  friend std::expected<ModelOptTargetPlan, Error> NormalizeMlxTarget(Checkpoint&& source,
                                                                     NormalizationLimits prepaid);
  ModelOptTargetPlan() = default;
  std::optional<IndexedPartPlan> parts_{};
  spec::Source canonical_{};
  spec::ModelSpec spec_{};
  std::vector<spec::SourceValue> values_{};
  std::vector<NormalizedPartBinding> bindings_{};
  std::array<SemanticReceipt, 3> semantic_{};
  std::vector<PlannedPartFold> planned_folds_{};
  NormalizationCost cost_{};
};

// The closed source domain's vocabulary (config.json's text_config).
inline constexpr std::uint32_t kModelOptQwen38Vocab = 248320;

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

class ModelOptDrafterPlan;
// The MTP drafter (`llmp import --component mtp`) of the same closed source
// domain: its config contract, every mtp.* tensor by its pinned descriptor,
// and with a head list lm_head.weight. Metadata only; the head list was read
// by ReadDraftHeadList.
std::expected<NormalizationCost, Error> ModelOptDrafterBound(const Checkpoint& source,
                                                             std::size_t head_ids);
// Publishes from the files that hold the drafter's tensors (and the head's
// rows) beside the configuration documents, and the head list: no other
// shard is read. Its tensors keep their source encodings (BF16 linears and
// zero-centered norms; NVFP4 routed experts relaid out into the scale-atom
// layout, import/expert_relayout.h); a selected head's BF16
// rows are gathered from lm_head.weight in the list's order, with its I32
// token map [1, rows]. The drafter carries no spec of its own (D-113): it is
// composed with a target's at serving; its spec here is that composition
// with the native target's, to validate and place it.
std::expected<ModelOptDrafterPlan, Error> NormalizeModelOptDrafter(
    const Checkpoint& source, std::optional<DraftHeadList> head, NormalizationLimits prepaid);
class ModelOptDrafterPlan {
 public:
  ModelOptDrafterPlan(const ModelOptDrafterPlan&) = delete;
  ModelOptDrafterPlan& operator=(const ModelOptDrafterPlan&) = delete;
  ModelOptDrafterPlan(ModelOptDrafterPlan&&) noexcept = default;
  ModelOptDrafterPlan& operator=(ModelOptDrafterPlan&&) noexcept = default;
  const IndexedPartPlan& parts() const { return *parts_; }
  const spec::Source& canonical() const { return canonical_; }
  const spec::ModelSpec& spec() const { return spec_; }
  std::span<const spec::SourceValue> values() const { return values_; }
  std::span<const NormalizedPartBinding> bindings() const { return bindings_; }
  std::span<const GeneratedPart> generated() const { return generated_; }
  // None: the drafter's norms keep their source form.
  std::span<const PlannedPartFold> planned_folds() const { return planned_folds_; }
  // The selected head's rows; 0 without one.
  std::uint64_t selected_rows() const { return selected_rows_; }
  // The identity-affecting import options, as the manifest's importer hashes
  // them (empty without a head).
  const std::string& options() const { return options_; }
  NormalizationCost cost() const { return cost_; }

 private:
  friend class llmp::test_support::PartPublicationFixture;
  friend std::expected<ModelOptDrafterPlan, Error> NormalizeModelOptDrafter(
      const Checkpoint& source, std::optional<DraftHeadList> head, NormalizationLimits prepaid);
  ModelOptDrafterPlan() = default;
  std::optional<IndexedPartPlan> parts_{};
  spec::Source canonical_{};
  spec::ModelSpec spec_{};
  std::vector<spec::SourceValue> values_{};
  std::vector<NormalizedPartBinding> bindings_{};
  std::vector<GeneratedPart> generated_{};
  std::vector<PlannedPartFold> planned_folds_{};
  std::uint64_t selected_rows_ = 0;
  std::string options_{};
  NormalizationCost cost_{};
};
}  // namespace llmp::import
#endif  // LLMP_IMPORT_MODELOPT_QWEN38_H_
