// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#ifndef LLMP_IMPORT_MODELOPT_QWEN38_H_
#define LLMP_IMPORT_MODELOPT_QWEN38_H_
#include <array>
#include <cstdint>
#include <expected>
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
// Newly owned transforms, not historical source-converter assertions. The
// publisher resolves each logical output through bindings(), and records the
// fold as applied only after every authenticated route completes.
struct PlannedPartFold {
  std::string fold{};
  std::vector<std::uint32_t> outputs{};
};
class ModelOptTargetPlan;
// Metadata-only, allocation-free admission for the closed source domain.
// Caller separately owns/funds Checkpoint and its reader before calling this.
std::expected<NormalizationCost, Error> ModelOptTargetBound(const Checkpoint& source);
// Reads exactly 280 bytes after admission through authenticated source identities.
// No full payload hashing, publication, runtime admission or MTP/vision import.
std::expected<ModelOptTargetPlan, Error> NormalizeModelOptTarget(Checkpoint&& source,
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
}  // namespace llmp::import
#endif  // LLMP_IMPORT_MODELOPT_QWEN38_H_
