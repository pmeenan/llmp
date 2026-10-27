// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// A model's slabs of routed experts (paged_weights.h SlabSpec), from the
// native tensor formats of its artifact's expert arrays. In the engine core
// (llmp_engine_core), every profile.

#ifndef LLMP_ENGINE_EXPERT_SLAB_H_
#define LLMP_ENGINE_EXPERT_SLAB_H_

#include <cstdint>
#include <expected>
#include <string>
#include <vector>

#include "artifact/artifact.h"
#include "engine/paged_weights.h"

namespace llmp::engine {

// Every layer's routed experts as one slab (the artifact's expert arrays of
// that layer, ExpertArray::layer, by their native formats: a single-part
// block or dense encoding's block bytes, or a split encoding's parts at
// 16-byte expert bases), its pitch rounded up to `alignment` (a positive
// multiple of 16) and to every array's block bytes, so each expert's base
// keeps both (Gemma 4's runner's rule: a shard change then falls on an
// alignable page boundary); the expert groups placed by their slabs alone
// (GroupPlace::kNone), every other group on the device. Refused if an
// expert group is in no slab.
struct LayerSlabs {
  std::vector<GroupPlace> places;    // by group
  std::vector<SlabSpec> slabs;       // by layer with experts, ascending
  std::vector<std::uint64_t> pitch;  // by expert array: its slab's
};
std::expected<LayerSlabs, std::string> ExpertSlabsOf(const artifact::Artifact& artifact,
                                                     std::uint64_t alignment);

}  // namespace llmp::engine

#endif  // LLMP_ENGINE_EXPERT_SLAB_H_
