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
  static const std::array<const ComponentDef*, 22> catalog{
      &defs::AttentionBorrowed(),
      &defs::AttentionDense(),
      &defs::AttentionGdn(),
      &defs::AttentionMqa(),
      &defs::AttentionQsa(),
      &defs::AttentionQsaReplay(),
      &defs::EmbeddingInject(),
      &defs::EmbeddingMtp(),
      &defs::EmbeddingMtpRepeat(),
      &defs::EmbeddingProjected(),
      &defs::EmbeddingToken(),
      &defs::FfnGlu(),
      &defs::FfnMoe(),
      &defs::HcCombine(),
      &defs::HcMix(),
      &defs::HeadLinear(),
      &defs::NormRms(),
      &defs::ResidualHc(),
      &defs::ResidualPle(),
      &defs::ScaleLearned(),
      &defs::SideHash(),
      &defs::SideNgram(),
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
