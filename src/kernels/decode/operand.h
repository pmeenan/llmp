// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// What the device decoders (decoders.cuh) read, shared by the host launch
// code and the device code, vendor-free: the decoders this module has, one
// per encoding and layout, the geometry each assumes, and one matrix's
// operand as every decoder takes it. The geometry is a local copy of the
// format layer's (format/encoding.h, layout.h), since device code cannot
// call its host functions; decode_test checks the copy against
// format::Info() and format::ScaleAtomOffset.

#ifndef LLMP_KERNELS_DECODE_OPERAND_H_
#define LLMP_KERNELS_DECODE_OPERAND_H_

#include <cstdint>

#ifdef __CUDACC__
#define LLMP_DECODE_HD __host__ __device__
#else
#define LLMP_DECODE_HD
#endif

namespace llmp::kernels::decode {

// The decoders, one per encoding and layout (decoders.cuh).
enum class DecoderId : std::uint8_t {
  kBf16Dense,
  kF16Dense,
  kQ8_0Blocks,        // GGML's block_q8_0: an F16 d, then 32 I8 (34 bytes)
  kNvfp4Split,        // codes K / 2 bytes a row, E4M3 scales K / 16 a row
  kNvfp4Atoms,        // codes as split, scales in CUTLASS's SM1xx atoms
  kNvfp4Interleaved,  // each row's K / 2 code bytes, then its K / 16 scales
  kNvfp4GgmlBlocks,   // GGML's block_nvfp4: four E4M3 scales, then 32 bytes
  kMxfp8Split,        // E4M3 codes K bytes a row, E8M0 scales K / 32 a row
  kMxfp8Atoms,        // codes as split, scales in CUTLASS's SM1xx atoms
  kCount
};

// Values one Load decodes: a segment of a row, at a multiple of it.
inline constexpr std::uint32_t kSegment = 16;

// The block geometry the decoders assume (format/encoding.h's).
inline constexpr std::uint32_t kNvfp4Block = 16;      // E2M1 codes a scale
inline constexpr std::uint32_t kMxfp8Block = 32;      // E4M3 codes a scale
inline constexpr std::uint32_t kQ8_0Block = 32;       // GGML's QK8_0
inline constexpr std::uint32_t kQ8_0BlockBytes = 34;  // sizeof(block_q8_0)
inline constexpr std::uint32_t kGgmlNvfp4Block = 64;  // GGML's QK_NVFP4
inline constexpr std::uint32_t kGgmlNvfp4BlockBytes = 36;

// CUTLASS's SM1xx scale atoms (format/layout.h's).
inline constexpr std::uint64_t kAtomRows = 128;
inline constexpr std::uint64_t kAtomBlocks = 4;
inline constexpr std::uint64_t kAtomBytes = 512;

// format::ScaleAtomOffset, for device code: row `row`'s scale of block
// `block` in a matrix with `blocks` scales a row (a multiple of 4).
LLMP_DECODE_HD constexpr std::uint64_t AtomOffset(std::uint64_t row, std::uint64_t block,
                                                  std::uint64_t blocks) {
  return ((((row / kAtomRows) * (blocks / kAtomBlocks)) + (block / kAtomBlocks)) * kAtomBytes) +
         ((row % 32) * 16) + (((row % kAtomRows) / 32) * 4) + (block % kAtomBlocks);
}

// One matrix of a weight, as every decoder reads it: its parts' addresses
// at the matrix's share, its row length, and its tensor scale. Trivially
// copyable, the same on the host and the device: a grouped product reads
// an array of them, one per expert, from device memory.
struct Operand {
  // kData, or kCodes for the split arrangements (split, atoms).
  const std::uint8_t* data = nullptr;
  // kBlockScales for the split arrangements; unused otherwise.
  const std::uint8_t* scales = nullptr;
  // F32 tensor scales (NVFP4's per matrix), or null: row r takes
  // tensor_scale[r / rows_per_scale].
  const float* tensor_scale = nullptr;
  std::uint32_t k = 0;               // values a row
  std::uint32_t rows = 0;            // rows of the matrix (ne[1])
  std::uint32_t rows_per_scale = 0;  // ne[1] / stacked
  std::uint32_t pad = 0;
  std::uint64_t row_bytes = 0;  // bytes of a row of `data`
};

}  // namespace llmp::kernels::decode

#endif  // LLMP_KERNELS_DECODE_OPERAND_H_
