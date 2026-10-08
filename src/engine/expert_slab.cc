// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "engine/expert_slab.h"

#include <cstdint>
#include <format>
#include <numeric>
#include <optional>

#include "engine/support.h"
#include "ggml.h"
#include "kernels/ggml/dsv4_graph.h"  // GgmlTypeOf

namespace llmp::engine {

namespace {

using support::Error;
using support::Round;

}  // namespace

std::expected<SlabSpec, std::string> ExpertSlab(
    const artifact::Artifact& artifact,
    std::span<const std::pair<std::uint32_t, std::string_view>> arrays, std::uint32_t count,
    std::uint64_t alignment, std::uint32_t layer) {
  std::uint64_t unit = 16;
  std::optional<std::uint32_t> first_group;
  for (const auto& [index, type_name] : arrays) {
    const auto& a = artifact.expert_arrays()[index];
    auto type = kernels::ggml::GgmlTypeOf(type_name);
    if (!type) {
      return Error(type.error().detail);
    }
    unit = std::lcm(unit, static_cast<std::uint64_t>(ggml_type_size(*type)));
    if (first_group && *first_group != a.first_group) {
      return Error(std::format("layer {}'s expert arrays do not share their groups", layer));
    }
    first_group = a.first_group;
  }
  const std::uint32_t g = first_group.value_or(0);
  if (g >= artifact.groups().size()) {
    return Error(std::format("layer {}'s expert arrays name no group", layer));
  }
  return SlabSpec{.first_group = g,
                  .count = count,
                  .stride = Round(artifact.groups()[g].stored.value(), unit),
                  .alignment = alignment};
}

}  // namespace llmp::engine
