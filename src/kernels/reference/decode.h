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
// The affine encoding (MLX's affine mode; AWQ's and GPTQ's arithmetic) is
// its definition: code q (layout.h's bit stream) of group g, scale s and
// bias b of the scales' type, gives s · q + b rounded once to F32 (a fused
// multiply-add), or with a zero point z, s · (q − z) (exact for F16 and
// BF16 scales). MLX's own kernels compute s · q + b in the activations'
// type, rounding the product and the sum there (bfloat16 for its BF16
// checkpoints), so they agree within that type's rounding.
//
// GGML's own NVFP4 and MXFP4 dequantization agree for every scale a valid
// checkpoint holds; they differ where GGML reads an E4M3 scale as unsigned
// and 0x7F as zero (NaN here), and an E8M0 0xFF as 2^128 (NaN here).
//
// The EXL3 trellis is ExLlamaV3's, at the locked pin (exllamav3_ext/quant):
//   - a 16 × 16 tile of the weights is 256 · k_bits bits of a circular
//     stream (uint32 words, most significant bit first), position t's
//     16-bit state the bits ending at (t + 1) · k_bits (exl3_dq.cuh's dq
//     and dq4), every integer k_bits from 1 to 8;
//   - a state's value is its codebook's F16 (codebook.cuh's decode_3inst:
//     3inst, mcg and mul1, rounded as __hadd and __hfma round), placed in
//     the tile where reconstruct_tile (reconstruct.cu) stores it;
//   - those are the rotated weights Ŵ, K × N; the weights are W =
//     diag(suh) · H · Ŵ · H · diag(svh) / 128, with H the 128-point
//     Sylvester Hadamard over each block of 128 inputs and of 128 outputs
//     (hadamard_inner.cuh, reconstruct_had_kernel), so a row of W (an
//     output feature) needs its block of Ŵ.
// DecodeExl3Rotated gives Ŵ exactly as reconstruct_kernel writes it, F16
// bit for bit. DecodeRows gives W computed in F64, exact for 3inst and mcg
// and rounded once to F32, where upstream's kernels round the transforms
// to F16 or F32 along the way (its reconstruct_had_kernel in F16 butterflies,
// the packed and reconstruction paths with the input in F16): they agree
// within that arithmetic, not bit for bit. Half-integer rates (mul1's
// K + 0.5, upstream's HALF tiles) have no format yet.

#ifndef LLMP_KERNELS_REFERENCE_DECODE_H_
#define LLMP_KERNELS_REFERENCE_DECODE_H_

#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <string>
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
// the matrix takes scale matrix · stacked + r / (ne[1] / stacked). An EXL3
// trellis whose K or N is not a multiple of 128 is kUnsupported (upstream's
// transforms and kernels take none).
std::expected<void, DecodeError> DecodeRows(const format::TensorFormat& format,
                                            std::span<const PartView> parts, std::uint64_t matrix,
                                            std::uint64_t first, std::uint64_t count,
                                            std::span<float> out);

// The reverse into a single-part format's data (all its matrices' bytes):
// rows.size() / ne[0] rows from `first` of matrix `matrix`. Dense F32, F16
// and BF16 (round to nearest even), and GGML's block encodings through
// GGML's reference quantization where it needs no importance matrix; for
// tests and state, every one of format::StateEncodings() among them.
// Everything else is kUnsupported.
std::expected<void, DecodeError> EncodeRows(const format::TensorFormat& format,
                                            std::uint64_t matrix, std::uint64_t first,
                                            std::span<const float> rows, std::span<std::byte> data);

// Reference component cache row: round complete blocks through EncodeRows /
// DecodeRows, not per-scalar rounding. At most 256 values; no retained spans.
std::expected<void, std::string> StoreStateRow(format::Encoding encoding,
                                               std::span<const double> source,
                                               std::span<double> destination);

// An EXL3 trellis's rotated weights Ŵ, F16 bits: columns [first, first +
// count) of matrix `matrix`, multiples of 16, for every one of the K rows,
// row-major (K · count values), as reconstruct_kernel writes them.
std::expected<void, DecodeError> DecodeExl3Rotated(const format::TensorFormat& format,
                                                   std::span<const PartView> parts,
                                                   std::uint64_t matrix, std::uint64_t first,
                                                   std::uint64_t count,
                                                   std::span<std::uint16_t> out);

// A 16-bit trellis state's value in a codebook, as F16 bits (NaN for
// kNone).
std::uint16_t Exl3CodebookValue(format::Exl3Codebook codebook, std::uint16_t state);

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
