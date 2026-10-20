// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: MIT AND Apache-2.0

// Scalar reference rotation for src/components: implements ir/op.h's kRope
// specification directly from ir::RopeParams, optional per-pair frequency
// factors, and section positions, independent of kernels/reference/exec.cc.

#ifndef LLMP_COMPONENTS_ROTATION_REFERENCE_H_
#define LLMP_COMPONENTS_ROTATION_REFERENCE_H_

#include <array>
#include <cstdint>
#include <span>

#include "ir/op.h"

namespace llmp::components::defs {

// Rotates `heads` heads of `head_dim` values in `row` in place as ir/op.h's
// kRope defines it (each angle step rounded to F32), at the four section
// positions `positions` and optional per-pair frequency factors `freqs`.
void RotateHeads(std::span<double> row, std::uint64_t heads, std::uint64_t head_dim,
                 const ir::RopeParams& params, std::array<std::int64_t, 4> positions,
                 std::span<const float> freqs = {});

inline void RotateHeads(std::span<double> row, std::uint64_t heads, std::uint64_t head_dim,
                        const ir::RopeParams& params, std::int64_t position,
                        std::span<const float> freqs = {}) {
  RotateHeads(row, heads, head_dim, params, {position, position, position, position}, freqs);
}

}  // namespace llmp::components::defs

#endif  // LLMP_COMPONENTS_ROTATION_REFERENCE_H_
