// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The prompt wave: one wave of target chunks (prefill, decode or scored rows)
// over the wave composer, with prompt batching where the runner declares it
// (PromptBatch). Decode and score require their own numerical-use
// permissions; prefill requires prompt permission. The history views hold
// every token from zero through each chunk's end, for a stage whose steps
// read them (side-row lookups); empty otherwise.

#ifndef LLMP_ENGINE_PROMPT_WAVE_H_
#define LLMP_ENGINE_PROMPT_WAVE_H_

#include <cstdint>
#include <span>

#include "engine/wave_composer.h"

namespace llmp::engine {

WaveReport RunPromptWave(ModelRunner& runner, const WaveComposer& composer,
                         std::span<const PendingWork> pending, std::span<const SegmentWork> work,
                         std::span<const std::span<const std::int32_t>> histories,
                         const AheadFor& ahead = {});

}  // namespace llmp::engine

#endif  // LLMP_ENGINE_PROMPT_WAVE_H_
