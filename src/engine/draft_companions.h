// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Shared prefill/decode/scored target chunks with prompt batching.
// Decode and score require their own numerical-use permissions. Prefill
// requires prompt permission.
// The history views contain every token from zero through each chunk's end;
// they are required only for an opted-in owner.

#ifndef LLMP_ENGINE_DRAFT_COMPANIONS_H_
#define LLMP_ENGINE_DRAFT_COMPANIONS_H_

#include <cstdint>
#include <span>

#include "engine/wave_composer.h"

namespace llmp::engine {

WaveReport RunPromptWave(ModelRunner& runner, const WaveComposer& composer,
                         std::span<const PendingWork> pending, std::span<const SegmentWork> work,
                         std::span<const std::span<const std::int32_t>> histories,
                         const AheadFor& ahead = {});

}  // namespace llmp::engine

#endif  // LLMP_ENGINE_DRAFT_COMPANIONS_H_
