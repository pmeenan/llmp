// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// A GGML runner's slab of a layer's routed experts (paged_weights.h
// SlabSpec), from the GGML types of its expert arrays. CUDA builds only
// (llmp_engine): it reads GGML's type table, which the engine core does not
// name.

#ifndef LLMP_ENGINE_EXPERT_SLAB_H_
#define LLMP_ENGINE_EXPERT_SLAB_H_

#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <string_view>
#include <utility>

#include "artifact/artifact.h"
#include "engine/paged_weights.h"

namespace llmp::engine {

// The slab of a layer whose expert arrays (each an index into the
// artifact's expert arrays and its GGML type's name) share their groups:
// its first group, and a stride of the stored bytes rounded up to 16 and to
// every array's element size. Refused if the arrays' groups differ.
std::expected<SlabSpec, std::string> ExpertSlab(
    const artifact::Artifact& artifact,
    std::span<const std::pair<std::uint32_t, std::string_view>> arrays, std::uint32_t count,
    std::uint64_t alignment, std::uint32_t layer);

}  // namespace llmp::engine

#endif  // LLMP_ENGINE_EXPERT_SLAB_H_
