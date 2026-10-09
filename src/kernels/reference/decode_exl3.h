// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The EXL3 trellis's CPU reference, private to the reference module
// (decode.h says what it computes and from which upstream functions):
// decode.cc binds a tensor's parts and checks its ranges, then calls these
// on one matrix's share of each part.

#ifndef LLMP_KERNELS_REFERENCE_DECODE_EXL3_H_
#define LLMP_KERNELS_REFERENCE_DECODE_EXL3_H_

#include <cstddef>
#include <cstdint>
#include <span>

#include "format/tensor_format.h"

namespace llmp::kernels::reference::exl3 {

// One matrix's parts: the trellis ([K / 16][N / 16][16 · k_bits] I16),
// suh (K F16) and svh (N F16).
struct Matrix {
  std::span<const std::byte> codes{};
  std::span<const std::byte> suh{};
  std::span<const std::byte> svh{};
};

// ExLlamaV3's 128-point Hadamard blocks: both sides of every EXL3 linear
// are multiples of 128 (its quantizer's and kernels' rule), which the
// original-basis decode needs; the rotated decode needs only whole tiles.
inline constexpr std::uint64_t kHadamardBlock = 128;
bool Transformable(const format::TensorFormat& format);

// Columns [first, first + count) of the rotated weights, multiples of 16,
// as F16 bits: K rows of `count`, row-major.
void Rotated(const format::TensorFormat& format, const Matrix& matrix, std::uint64_t first,
             std::uint64_t count, std::span<std::uint16_t> out);

// Rows (output features) [first, first + count) of the original-basis
// weights, each K F32 values, into `out` (count · K values).
void Rows(const format::TensorFormat& format, const Matrix& matrix, std::uint64_t first,
          std::uint64_t count, std::span<float> out);

}  // namespace llmp::kernels::reference::exl3

#endif  // LLMP_KERNELS_REFERENCE_DECODE_EXL3_H_
