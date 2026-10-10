// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "engine/expert_slab.h"

#include <cstdint>
#include <format>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <vector>

#include "artifact/representation.h"
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

std::expected<LayerSlabs, std::string> ExpertSlabsOf(const artifact::Artifact& artifact,
                                                     std::uint64_t alignment) {
  LayerSlabs out;
  const auto arrays = artifact.expert_arrays();
  out.places.assign(artifact.groups().size(), GroupPlace::kDevice);
  out.pitch.assign(arrays.size(), 0);
  std::map<std::uint32_t, std::vector<std::pair<std::uint32_t, std::string_view>>> layers;
  for (std::uint32_t i = 0; i < arrays.size(); ++i) {
    layers[arrays[i].layer].emplace_back(i, arrays[i].repr.type);
  }
  std::vector<bool> covered(artifact.groups().size(), false);
  for (const auto& [layer, list] : layers) {
    auto slab = ExpertSlab(artifact, list, arrays[list.front().first].count, alignment, layer);
    if (!slab) return std::unexpected(slab.error());
    std::uint64_t quantum = alignment;
    for (const auto& [index, type] : list) {
      if (arrays[index].count != slab->count) {
        return Error(std::format("layer {}'s expert arrays differ in their experts", layer));
      }
      const auto* info = artifact::FindGgmlType(type);
      if (info == nullptr)
        return Error(std::format("layer {}'s experts have type {}", layer, type));
      quantum = std::lcm(quantum, std::uint64_t{info->block_bytes});
    }
    if (slab->stride > std::numeric_limits<std::uint64_t>::max() - (quantum - 1)) {
      return Error(std::format("layer {}'s expert pitch overflows", layer));
    }
    slab->stride = Round(slab->stride, quantum);
    for (const auto& [index, type] : list) out.pitch[index] = slab->stride;
    for (std::uint32_t e = 0; e < slab->count; ++e) {
      if (slab->first_group + e < covered.size()) covered[slab->first_group + e] = true;
    }
    out.slabs.push_back(*slab);
  }
  for (std::size_t g = 0; g < out.places.size(); ++g) {
    if (artifact.groups()[g].kind != artifact::GroupKind::kExpert) continue;
    if (!covered[g]) return Error(std::format("expert group {} is in no slab", g));
    out.places[g] = GroupPlace::kNone;
  }
  return out;
}

}  // namespace llmp::engine
