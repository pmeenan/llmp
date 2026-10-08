// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef LLMP_ENGINE_GEMMA4_WAVE_H_
#define LLMP_ENGINE_GEMMA4_WAVE_H_

#include <cstdint>

namespace llmp::engine {
// Vendor-free serving bound for the checked one-row product policy.
inline constexpr std::uint32_t kGemma4InvariantWaveRows = 8;
// Ordinary products admit the full serving cohort; checked invariant sums
// retain their separately validated eight-column bound.
inline constexpr std::uint32_t kGemma4OrdinaryWaveRows = 12;
constexpr std::uint32_t Gemma4WaveRows(bool row_invariant) {
  return row_invariant ? kGemma4InvariantWaveRows : kGemma4OrdinaryWaveRows;
}
}  // namespace llmp::engine
#endif  // LLMP_ENGINE_GEMMA4_WAVE_H_
