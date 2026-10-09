// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// CPU reference decoders (D-108): any tensor format's rows as F32, the
// definition every device decoder and product is checked against. Built in
// every profile; no device code.
//
// GGML's block encodings decode through GGML's own dequantization at the
// pin (ggml-base's type traits: the encoding's definition), except Q8_1
// and Q8_K, GGML's activation-side blocks, which it gives none and which
// are d · q here. NVFP4, MXFP4 and MXFP8 in every layout, and the dense
// elements, are llmpalooza's:
//   - E2M1 codes are ±{0, 0.5, 1, 1.5, 2, 3, 4, 6}, the sign in bit 3;
//   - E4M3 is OCP's (bias 7, subnormals, no infinities, 0x7F and 0xFF
//     NaN), E5M2 an F16's upper byte, E8M0 2^(x - 127) with 0xFF NaN;
//   - a value is code · block scale (· tensor scale), rounded once per
//     product in F32, so where the exact value is an F32 the result is it.
// GGML's own NVFP4 and MXFP4 dequantization agree for every scale a valid
// checkpoint holds; they differ where GGML reads an E4M3 scale as unsigned
// and 0x7F as zero (NaN here), and an E8M0 0xFF as 2^128 (NaN here).
// The EXL3 trellis is not decoded here yet (kUnsupported; the EXL3 slice).

#ifndef LLMP_KERNELS_REFERENCE_DECODE_H_
#define LLMP_KERNELS_REFERENCE_DECODE_H_

#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <string_view>

#include "format/encoding.h"
#include "format/tensor_format.h"

namespace llmp::kernels::reference {

// One part's bytes (all of the tensor's matrices).
struct PartView {
  format::PartRole role = format::PartRole::kData;
  std::span<const std::byte> bytes{};
};

enum class DecodeError : std::uint8_t {
  kUnsupported,  // a format this reference does not decode (or encode)
  kFormat,       // not a valid format (format::Validate)
  kParts,        // a part missing, repeated, extra, or shorter than its bytes
  kRange,        // a matrix or rows outside the tensor, or output of the wrong size
};
std::string_view DecodeErrorName(DecodeError error);

// Rows [first, first + count) of matrix `matrix` (of format::Matrices) as
// F32, count · ne[0] values row after row. With a tensor scale, row r of
// the matrix takes scale matrix · stacked + r / (ne[1] / stacked).
std::expected<void, DecodeError> DecodeRows(const format::TensorFormat& format,
                                            std::span<const PartView> parts, std::uint64_t matrix,
                                            std::uint64_t first, std::uint64_t count,
                                            std::span<float> out);

// The reverse into a single-part format's data (all its matrices' bytes):
// rows.size() / ne[0] rows from `first` of matrix `matrix`. Dense F32, F16
// and BF16 (round to nearest even), and GGML's block encodings through
// GGML's reference quantization where it needs no importance matrix; for
// tests and lossy state. Everything else is kUnsupported.
std::expected<void, DecodeError> EncodeRows(const format::TensorFormat& format,
                                            std::uint64_t matrix, std::uint64_t first,
                                            std::span<const float> rows, std::span<std::byte> data);

// Element conversions, as DecodeRows makes them.
float F16ToF32(std::uint16_t bits);
std::uint16_t F32ToF16(float value);  // round to nearest even
float Bf16ToF32(std::uint16_t bits);
std::uint16_t F32ToBf16(float value);  // round to nearest even; NaN stays NaN
float E4m3ToF32(std::uint8_t bits);
float E5m2ToF32(std::uint8_t bits);
float E8m0ToF32(std::uint8_t bits);
float E2m1ToF32(std::uint8_t code);  // the low four bits

}  // namespace llmp::kernels::reference

#endif  // LLMP_KERNELS_REFERENCE_DECODE_H_
