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
#include "format/tensor_format.h"
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
  std::map<std::uint32_t, std::vector<std::uint32_t>> layers;
  for (std::uint32_t i = 0; i < arrays.size(); ++i) layers[arrays[i].layer].push_back(i);
  if (alignment == 0) return Error("expert slab alignment is zero");
  std::vector<bool> covered(artifact.groups().size(), false);
  for (const auto& [layer, list] : layers) {
    std::uint64_t unit = 16, quantum = alignment;
    const auto& first = arrays[list.front()];
    for (const auto index : list) {
      const auto& array = arrays[index];
      if (array.first_group != first.first_group || array.count != first.count)
        return Error(std::format("layer {}'s expert arrays differ in groups or experts", layer));
      std::uint64_t element_bytes = 0, block_bytes = 0;
      if (artifact.format_version() == 1) {
        // Native descriptors are authoritative. Never populate or consult the
        // absent legacy Representation of a v1 expert array.
        if (!array.format || !format::Validate(*array.format) ||
            array.part != format::PartRole::kData || array.format->shape.rank != 2 ||
            (array.format->layout != format::Layout::kGgmlBlocks &&
             array.format->layout != format::Layout::kDense))
          return Error("native expert slab needs a valid GGML-compatible data slice");
        const auto& info = format::Info(array.format->encoding);
        if (!info.ggml) return Error("native expert slice has no GGML block geometry");
        element_bytes = block_bytes = info.ggml->block_bytes;
      } else {
        auto type = kernels::ggml::GgmlTypeOf(array.repr.type);
        if (!type) return Error(type.error().detail);
        const auto* info = artifact::FindGgmlType(array.repr.type);
        if (!info) return Error("legacy expert slice has unknown GGML block geometry");
        element_bytes = static_cast<std::uint64_t>(ggml_type_size(*type));
        block_bytes = info->block_bytes;
      }
      const auto combine = [](std::uint64_t current,
                              std::uint64_t block) -> std::optional<std::uint64_t> {
        if (block == 0 ||
            current / std::gcd(current, block) > std::numeric_limits<std::uint64_t>::max() / block)
          return std::nullopt;
        return std::lcm(current, block);
      };
      const auto next_unit = combine(unit, element_bytes),
                 next_quantum = combine(quantum, block_bytes);
      if (!next_unit || !next_quantum) return Error("expert slab block alignment overflows");
      unit = *next_unit;
      quantum = *next_quantum;
    }
    const auto g = first.first_group;
    if (g >= artifact.groups().size() || first.count > artifact.groups().size() - g)
      return Error(std::format("layer {}'s expert arrays name missing groups", layer));
    const auto stored = artifact.groups()[g].stored.value();
    if (stored > std::numeric_limits<std::uint64_t>::max() - (unit - 1))
      return Error("expert slab stored size overflows");
    const auto stride = Round(stored, unit);
    if (stride > std::numeric_limits<std::uint64_t>::max() - (quantum - 1))
      return Error(std::format("layer {}'s expert pitch overflows", layer));
    const SlabSpec slab{.first_group = g,
                        .count = first.count,
                        .stride = Round(stride, quantum),
                        .alignment = alignment};
    for (const auto index : list) out.pitch[index] = slab.stride;
    for (std::uint32_t e = 0; e < slab.count; ++e) {
      if (slab.first_group + e < covered.size()) covered[slab.first_group + e] = true;
    }
    out.slabs.push_back(slab);
  }
  for (std::size_t g = 0; g < out.places.size(); ++g) {
    if (artifact.groups()[g].kind != artifact::GroupKind::kExpert) continue;
    if (!covered[g]) return Error(std::format("expert group {} is in no slab", g));
    out.places[g] = GroupPlace::kNone;
  }
  return out;
}

}  // namespace llmp::engine
