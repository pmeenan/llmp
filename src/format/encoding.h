// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Tensor encodings (D-108; docs/engine-components.md#weight-and-state-formats):
// a tensor's numeric content, named independently of any kernel library.
// An encoding fixes what the values are: codes, scales and their formats,
// and the block geometry along a row. How its bytes are arranged is the
// layout (layout.h), which a lossless relayout may change; an encoding is
// changed only by a transcode, which changes quality.
//
// GGML's block encodings are defined by GGML's own dequantization at the
// locked pin (ggml-quants.c); NVFP4, MXFP4 and MXFP8 by NVIDIA's and the
// OCP Microscaling specification's element and scale formats; the EXL3
// trellis by ExLlamaV3. Where GGML has a type for an encoding, its number
// and block struct are recorded as a lowering hint and as the geometry of
// GGML's layouts, never as the encoding's meaning.

#ifndef LLMP_FORMAT_ENCODING_H_
#define LLMP_FORMAT_ENCODING_H_

#include <cstdint>
#include <optional>
#include <span>
#include <string_view>

namespace llmp::format {

enum class Encoding : std::uint16_t {
  // Dense elements, one per value.
  kF32,
  kF16,
  kBF16,
  kF64,
  kI8,
  kI16,
  kI32,
  kI64,
  kU8,
  kU16,
  kU32,
  kU64,
  kBool,
  kF8E4M3,  // OCP E4M3: bias 7, no infinities, 0x7F and 0xFF NaN
  kF8E5M2,  // OCP E5M2: the upper byte of an F16
  kE8M0,    // OCP E8M0: 2^(x - 127), 0xFF NaN; Microscaling's block scale
  // GGML's block encodings (ggml-common.h at the locked pin).
  kQ1_0,
  kQ2_0,
  kQ4_0,
  kQ4_1,
  kQ5_0,
  kQ5_1,
  kQ8_0,
  kQ8_1,
  kQ2_K,
  kQ3_K,
  kQ4_K,
  kQ5_K,
  kQ6_K,
  kQ8_K,
  kIQ1_S,
  kIQ1_M,
  kIQ2_XXS,
  kIQ2_XS,
  kIQ2_S,
  kIQ3_XXS,
  kIQ3_S,
  kIQ4_NL,
  kIQ4_XS,
  kTQ1_0,
  kTQ2_0,
  // Block-scaled floating point: E2M1 codes with an E8M0 scale per 32
  // (MXFP4), E4M3 codes with an E8M0 scale per 32 (MXFP8), E2M1 codes with
  // an E4M3 scale per 16 and an optional F32 scale per matrix (NVFP4).
  kMxfp4,
  kMxfp8,
  kNvfp4,
  // ExLlamaV3's trellis: k_bits per weight, a codebook, and the input and
  // output sign vectors of its Hadamard rotations (EncodingParams).
  kExl3Trellis,
};

// The parts a multi-part encoding stores: codes and their block scales, a
// scale per matrix, EXL3's sign vectors and codebook marker. A single-part
// arrangement (a dense tensor, GGML's blocks, interleaved rows) holds its
// codes and scales together as kData.
enum class PartRole : std::uint8_t {
  kData,
  kCodes,
  kBlockScales,
  kTensorScale,
  kInputSigns,      // EXL3's suh
  kOutputSigns,     // EXL3's svh
  kCodebookMarker,  // EXL3's codebook marker
};
std::string_view PartRoleName(PartRole role);

// A part of an encoding stored in parts: its role, the dense encoding of
// its elements, and whether every tensor of the encoding has it.
struct PartSpec {
  PartRole role = PartRole::kData;
  Encoding element = Encoding::kU8;
  bool required = true;
};

enum class Exl3Codebook : std::uint8_t { kNone, kMcg, kMul1, k3Inst };

// What an encoding leaves to the tensor. Zero (none) wherever the encoding
// does not use a field.
struct EncodingParams {
  std::uint8_t k_bits = 0;                      // EXL3: bits per weight, 1 to 8
  Exl3Codebook codebook = Exl3Codebook::kNone;  // EXL3
  bool tensor_scale = false;                    // NVFP4: an F32 scale per matrix
  bool operator==(const EncodingParams&) const = default;
};

// GGML's type for an encoding and its block struct at the locked pin.
struct GgmlBlock {
  std::uint32_t type = 0;            // enum ggml_type's value
  std::uint32_t block_elements = 0;  // along ne[0]
  std::uint32_t block_bytes = 0;
};

struct EncodingInfo {
  Encoding id = Encoding::kF32;
  // Lower-case letters, digits and '_' ("q4_k", "nvfp4", "exl3_trellis"):
  // the name a manifest and an implementation's name use.
  std::string_view name{};
  // The values one block covers along ne[0], which ne[0] is a multiple of:
  // a scale's values (a GGML type's block, NVFP4's 16, Microscaling's 32),
  // the trellis's 16-value tile edge; 1 for dense encodings.
  std::uint32_t block_elements = 1;
  // Bits per element of a dense encoding, or per code of MXFP4, MXFP8 and
  // NVFP4; 0 where codes are not uniform (GGML's blocks, the trellis).
  std::uint32_t element_bits = 0;
  bool quantized = false;
  bool floating = false;  // a dense floating-point encoding
  // The parts a split arrangement stores (layout.h); empty for encodings
  // stored as one part.
  std::span<const PartSpec> parts{};
  std::optional<GgmlBlock> ggml{};
};

// Every encoding, in enum order.
std::span<const EncodingInfo> Encodings();
// Info(e).id == e for every enumerator; an out-of-range value is a bug.
const EncodingInfo& Info(Encoding encoding);
std::optional<Encoding> EncodingByName(std::string_view name);

// Whether every value of `from` is exactly a value of `to`, so that
// converting state from one to the other loses nothing. False where the
// conversion rounds, or where exactness is not established (a quantized
// encoding into anything but itself).
bool Lossless(Encoding from, Encoding to);

}  // namespace llmp::format

#endif  // LLMP_FORMAT_ENCODING_H_
