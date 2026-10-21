// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>

#include "components/component.h"
#include "components/defs.h"

namespace llmp::components {

std::span<const ComponentDef* const> Catalog() {
  // Sorted by (kind, version).
  static const std::array catalog{
      &defs::AffineChannel(),
      &defs::AttentionBorrowed(),
      &defs::AttentionDense(),
      &defs::AttentionGdn(),
      &defs::AttentionMqa(),
      &defs::AttentionQsa(),
      &defs::AttentionSpatial(),
      &defs::ConcatSpatial(),
      &defs::ConditionRows(),
      &defs::Conv2d(),
      &defs::EmbeddingInject(),
      &defs::EmbeddingMtp(),
      &defs::EmbeddingProjected(),
      &defs::EmbeddingToken(),
      &defs::FfnGlu(),
      &defs::FfnMoe(),
      &defs::HcCombine(),
      &defs::HcMix(),
      &defs::HeadLinear(),
      &defs::NormAdaptive(),
      &defs::NormChannel(),
      &defs::NormRms(),
      &defs::ProjectionSpatial(),
      &defs::ReshapeSpatial(),
      &defs::ResidualGated(),
      &defs::ResidualHc(),
      &defs::ResidualPle(),
      &defs::ResidualScaled(),
      &defs::ResidualSpatial(),
      &defs::ResidualUpsample(),
      &defs::ScaleLearned(),
      &defs::SideHash(),
      &defs::SideNgram(),
      &defs::SliceSpatial(),
      &defs::UpsampleNearest(),
  };
  return catalog;
}

const ComponentDef* Find(std::string_view kind, std::uint16_t version) {
  for (const ComponentDef* def : Catalog()) {
    if (def->kind == kind && def->version == version) return def;
  }
  return nullptr;
}

std::optional<std::uint16_t> RoleIndex(const ComponentDef& def, std::string_view name) {
  const auto at = std::ranges::find(def.roles, name, &WeightRole::name);
  if (at == def.roles.end()) return std::nullopt;
  return static_cast<std::uint16_t>(at - def.roles.begin());
}

}  // namespace llmp::components
