// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#ifndef LLMP_IMPORT_NORMALIZERS_H_
#define LLMP_IMPORT_NORMALIZERS_H_

#include <span>
#include <string_view>

#include "import/modelopt_qwen38.h"

namespace llmp::import {
// Trusted normalizer implementations remain separate from the data-only
// architecture mappings. Recognition and CLI dispatch use this single table.
struct Normalizer {
  std::string_view container, architecture, quantization;
  std::expected<NormalizationCost, Error> (*target_bound)(const Checkpoint&);
  std::expected<ModelOptTargetPlan, Error> (*target)(Checkpoint&&, NormalizationLimits);
  std::expected<NormalizationCost, Error> (*drafter_bound)(const Checkpoint&, std::size_t);
  std::expected<ModelOptDrafterPlan, Error> (*drafter)(const Checkpoint&,
                                                       std::optional<DraftHeadList>,
                                                       NormalizationLimits,
                                                       const NormalizationTarget*);
};
std::span<const Normalizer> Normalizers();
const Normalizer* FindNormalizer(const Checkpoint& source);
}  // namespace llmp::import
#endif  // LLMP_IMPORT_NORMALIZERS_H_
