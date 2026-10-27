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

#include "engine/support.h"
#include "format/encoding.h"
#include "format/layout.h"
#include "format/tensor_format.h"

namespace llmp::engine {

namespace {

using support::Error;
using support::Round;

}  // namespace

std::expected<LayerSlabs, std::string> ExpertSlabsOf(const artifact::Artifact& artifact,
                                                     std::uint64_t alignment) {
  LayerSlabs out;
  const auto arrays = artifact.expert_arrays();
  out.places.assign(artifact.groups().size(), GroupPlace::kDevice);
  out.pitch.assign(arrays.size(), 0);
  std::map<std::uint32_t, std::vector<std::uint32_t>> layers;
  for (std::uint32_t i = 0; i < arrays.size(); ++i) layers[arrays[i].layer].push_back(i);
  if (alignment == 0 || alignment % 16 != 0)
    return Error("expert slab alignment is not a positive multiple of 16");
  std::vector<bool> covered(artifact.groups().size(), false);
  for (const auto& [layer, list] : layers) {
    // A multiple of the alignment (and so of the 16-byte expert base) and
    // of every array's block bytes.
    std::uint64_t quantum = alignment;
    const auto& first = arrays[list.front()];
    for (const auto index : list) {
      const auto& array = arrays[index];
      if (array.first_group != first.first_group || array.count != first.count)
        return Error(std::format("layer {}'s expert arrays differ in groups or experts", layer));
      std::uint64_t block_bytes = 0;
      // A split or scale-atom encoding's parts (codes, block scales, a
      // tensor scale; an affine encoding's biases or zero points) are
      // separate arrays of one expert group: each part's slice is placed at
      // the writer's member alignment and read by the format's decoders or
      // the block-scaled grouped GEMM, not as whole blocks; the slab keeps
      // 16-byte expert bases.
      const bool split =
          array.format &&
          (array.format->layout == format::Layout::kSplit ||
           array.format->layout == format::Layout::kScaleAtoms) &&
          (array.part == format::PartRole::kCodes || array.part == format::PartRole::kBlockScales ||
           array.part == format::PartRole::kTensorScale ||
           array.part == format::PartRole::kBiases || array.part == format::PartRole::kZeros);
      if (!array.format || !format::Validate(*array.format) || array.format->shape.rank != 2 ||
          (!split && (array.part != format::PartRole::kData ||
                      !format::IsSinglePartBlockOrDense(array.format->layout))))
        return Error("native expert slab needs a valid single-part block or split-part slice");
      if (split) {
        block_bytes = 16;
      } else {
        const auto block = format::BlockBytes(array.format->encoding);
        if (!block) return Error("native expert slice has no block geometry");
        block_bytes = *block;
      }
      const auto combine = [](std::uint64_t current,
                              std::uint64_t block) -> std::optional<std::uint64_t> {
        if (block == 0 ||
            current / std::gcd(current, block) > std::numeric_limits<std::uint64_t>::max() / block)
          return std::nullopt;
        return std::lcm(current, block);
      };
      const auto next = combine(quantum, block_bytes);
      if (!next) return Error("expert slab block alignment overflows");
      quantum = *next;
    }
    const auto g = first.first_group;
    if (g >= artifact.groups().size() || first.count > artifact.groups().size() - g)
      return Error(std::format("layer {}'s expert arrays name missing groups", layer));
    const auto stored = artifact.groups()[g].stored.value();
    if (stored > std::numeric_limits<std::uint64_t>::max() - (quantum - 1))
      return Error(std::format("layer {}'s expert pitch overflows", layer));
    const SlabSpec slab{.first_group = g,
                        .count = first.count,
                        .stride = Round(stored, quantum),
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
