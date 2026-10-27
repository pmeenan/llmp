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
// trellis by ExLlamaV3; the affine encoding by MLX's documented affine
// quantization (mx.quantize: w = s · ŵ + β) and, with zero points, AWQ's
// and GPTQ's s · (q − z). Where GGML has a type for an encoding, its number
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
  // Affine integer quantization (MLX's affine mode; AWQ's and GPTQ's
  // arithmetic): unsigned codes q of `bits` bits with a scale s per group
  // of `group` values along a row, and per group either a bias b of the
  // scales' element type (MLX: s · q + b) or an unsigned zero point z of
  // `bits` bits (AWQ, GPTQ: s · (q − z)). All of it is EncodingParams.
  kAffine,
  // E4M3FN values with an F32 multiplicative scale per 2-D block.
  kFp8Scaled,
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
  kCodebookMarker,  // EXL3's codebook marker (mcg and mul1 only)
  kBiases,          // the affine encoding's bias per group (MLX's biases)
  kZeros,           // the affine encoding's zero point per group (AWQ, GPTQ)
};
std::string_view PartRoleName(PartRole role);

// A part of an encoding stored in parts: its role, the dense encoding of
// its elements, and whether every tensor of the encoding has it. The
// affine encoding's scales and biases are of the element type its
// parameters name (EncodingParams::scale; listed as F16), and it has
// either biases or zero points.
struct PartSpec {
  PartRole role = PartRole::kData;
  Encoding element = Encoding::kU8;
  bool required = true;
};

enum class Exl3Codebook : std::uint8_t { kNone, kMcg, kMul1, k3Inst };

// The element type of the affine encoding's scales and biases.
enum class AffineScale : std::uint8_t { kNone, kF16, kBF16, kF32 };
// Its dense encoding (F16, BF16 or F32; F32 for kNone, which no valid
// format has).
Encoding AffineScaleEncoding(AffineScale scale);

// What an encoding leaves to the tensor. Zero (none) wherever the encoding
// does not use a field.
struct EncodingParams {
  std::uint8_t k_bits = 0;                      // EXL3: bits per weight, 1 to 8
  Exl3Codebook codebook = Exl3Codebook::kNone;  // EXL3
  bool tensor_scale = false;                    // NVFP4: an F32 scale per matrix
  // Affine: bits a code (1 to 8), values a group (a multiple of 8, which
  // ne[0] is a multiple of), the scales' (and biases') element type, and
  // zero points (kZeros) rather than biases (kBiases).
  std::uint8_t bits = 0;
  std::uint32_t group = 0;
  AffineScale scale = AffineScale::kNone;
  bool zero_point = false;
  // fp8_scaled: positive block dimensions in logical [rows, columns] order.
  std::uint32_t block_rows = 0;
  std::uint32_t block_cols = 0;
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
  // the trellis's 16-value tile edge; 1 for dense encodings; for the
  // affine encoding 8, the least group (a tensor's group is a parameter).
  std::uint32_t block_elements = 1;
  // Bits per element of a dense encoding, or per code of MXFP4, MXFP8 and
  // NVFP4; 0 where codes are not uniform (GGML's blocks, the trellis) or
  // a parameter (the affine encoding's).
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

// The byte size of one block of a single-part encoding (1-element dense
// encodings included); nullopt for multi-part encodings.
std::optional<std::uint32_t> BlockBytes(Encoding encoding);
// An encoding's whole block struct where it has one (GGML's block types at
// the pin: layout.h kGgmlBlocks, and NVFP4's kGgmlNvfp4Blocks): its block's
// values along ne[0] and its bytes; nothing for any other encoding.
struct BlockStruct {
  std::uint32_t elements = 0;
  std::uint32_t bytes = 0;
};
std::optional<BlockStruct> BlockStructOf(Encoding encoding);
// Case-insensitive lookup by encoding or container/GGML type name.
std::optional<Encoding> EncodingByTypeName(std::string_view name);
// A GGML type name of the pinned table, and a safetensors container dtype,
// as encodings.
std::optional<Encoding> EncodingOfGgmlType(std::string_view type);
std::optional<std::string_view> GgmlTypeName(std::uint32_t type_id);
std::optional<Encoding> EncodingOfDtype(std::string_view dtype);
// The container dtype of a dense encoding (EncodingOfDtype's inverse);
// nothing for an encoding safetensors has no dtype for (E8M0, every
// quantized one).
std::optional<std::string_view> DtypeOfEncoding(Encoding encoding);

// Whether every value of `from` is exactly a value of `to`, so that
// converting state from one to the other loses nothing. False where the
// conversion rounds, or where exactness is not established (a quantized
// encoding, the affine one among them, into anything but itself; within
// one encoding, formats of other parameters are not compared here).
bool Lossless(Encoding from, Encoding to);

// The encodings the format layer admits for a state tensor (D-108; a KV
// cache, a compressed or indexer cache, a recurrent or convolution
// state), each of which the CPU reference encodes and decodes: F32, F16
// and BF16, and the block encodings GGML's attention reads at the locked
// pin, Q4_0, Q4_1, Q5_0, Q5_1 and Q8_0 (ggml-cuda/fattn.cu,
// ggml_cuda_fattn_kv_type_supported: its vector kernel's K and V types;
// its tile and tensor-core kernels convert them to F16 first), each of
// which GGML's set_rows also stores. Membership says nothing about a
// device: whether a backend reads or stores one is its attention's and
// cache store's capability (on CUDA, Q8_0 and Q4_0 at head sizes 64 and
// 256: kernels/ggml/kv_quant.h; docs/engine-components.md), so a state
// may use one only where its plan finds both. Whether one is exact for a state is
// Lossless(the state's own encoding, the stored one); a lossy one is an
// opt-in per alias, off by default.
std::span<const Encoding> StateEncodings();
bool IsStateEncoding(Encoding encoding);

// A block-scaled encoding's scale byte. NVFP4's is UE4M3 (NVIDIA's PTX
// `.ue4m3`, CUTLASS's float_ue4m3_t): E4M3's exponent and mantissa, bias 7,
// without a sign bit; 0x00-0x7E are its finite values (E4M3's for those
// bytes) and 0x7F its NaN. MXFP4's and MXFP8's is E8M0 (OCP Microscaling):
// 0x00-0xFE are 2^(x - 127) and 0xFF NaN. Only those finite bytes mean one
// value to every reader: GGML's CPU reference ignores bit 7 and reads 0x7F
// as 0, its CUDA dequantizers honour bit 7 and read 0x7F and 0xFF as 0, and
// tensor cores take the raw byte. So llmp's own decoders (the CPU reference
// and the format ladder's device decoders, in every layout) read every
// other byte as NaN, and the importer refuses a checkpoint that holds one
// (docs/engine-components.md, "Block scales").
enum class ScaleCode : std::uint8_t { kNone, kUe4m3, kE8m0 };
// The scale code of `encoding`'s block scales (kNone: it has none of these).
ScaleCode ScaleCodeOf(Encoding encoding);
// Whether `byte` is a finite scale of `code` (every byte for kNone).
constexpr bool DefinedScale(ScaleCode code, std::uint8_t byte) {
  switch (code) {
    case ScaleCode::kUe4m3:
      return byte < 0x7F;
    case ScaleCode::kE8m0:
      return byte != 0xFF;
    case ScaleCode::kNone:
      return true;
  }
  return true;
}

}  // namespace llmp::format

#endif  // LLMP_FORMAT_ENCODING_H_
