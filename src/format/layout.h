// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Tensor layouts (D-108): how an encoding's bytes are arranged, which a
// lossless relayout at import may change. Every layout of an encoding
// holds the same values; tensor_format.h's size rule says which encodings
// each layout takes and how many bytes each part holds.
//
// Along a matrix, values are ne[0] (K) to a row and ne[1] (N) rows; a
// tensor of more dimensions is ne[2] · ne[3] such matrices one after the
// other in each part.
//
//   kDense              one part, K elements a row, rows in order.
//   kGgmlBlocks         one part: GGML's block structs, K / block a row
//                       (ggml-common.h at the pin). GGML's CUDA kernels
//                       may read past such a tensor whose K is not a
//                       multiple of 512 (over_read_row_step); that
//                       over-read is part of its readable range.
//   kGgmlNvfp4Blocks    NVFP4 in GGML's block_nvfp4: 64 values a block,
//                       its four E4M3 scales, then 32 code bytes, byte j of
//                       a 16-value sub-block holding values j (low nibble)
//                       and j + 8 (high). GGML's over-read rule applies.
//   kSplit              codes and block scales as separate row-major parts
//                       (ModelOpt, Microscaling and MLX checkpoints): a
//                       row's codes as a little-endian bit stream (code i
//                       at bit i · bits from the row's first byte's least
//                       significant bit: 4-bit codes two a byte, value 2i
//                       in the low nibble; MLX's 3-, 5- and 6-bit packs
//                       alike), and in the next part its K / block scales.
//                       The affine encoding adds a third part, a row's
//                       K / group biases (scale elements) or zero points
//                       (a bit stream of `bits`-bit codes, padded to a
//                       byte).
//   kRowInterleaved     one part: each row's codes, then its scales
//                       (Qwen3.8's n-gram table).
//   kScaleAtoms         codes as kSplit; block scales in scale atoms of
//                       128 rows by 4 scales (512 bytes), atoms along K
//                       first, rows padded to a whole atom
//                       (ScaleAtomOffset): the tile order block-scaled
//                       tensor-core products read (on the GB10, CUTLASS's
//                       SM1xx grouped GEMM). Any block-scaled encoding of
//                       one-byte scales fits it; a stacked tensor's
//                       matrices take whole atoms each when their rows are
//                       a multiple of 128. The importer relays routed
//                       experts out into it by role and encoding
//                       (import/expert_relayout.h); the IR prepares MXFP8
//                       activations in it.
//   kExl3Tiles          ExLlamaV3's trellis, I16 [K / 16, N / 16, 16 · k_bits]
//                       in container order, with its sign vectors and
//                       codebook marker as further parts.

#ifndef LLMP_FORMAT_LAYOUT_H_
#define LLMP_FORMAT_LAYOUT_H_

#include <cstdint>
#include <optional>
#include <span>
#include <string_view>

namespace llmp::format {

enum class Layout : std::uint16_t {
  kDense,
  kGgmlBlocks,
  kGgmlNvfp4Blocks,
  kSplit,
  kRowInterleaved,
  kScaleAtoms,
  kExl3Tiles,
};

struct LayoutInfo {
  Layout id = Layout::kDense;
  std::string_view name{};  // lower-case letters, digits and '_'
  // GGML's MATRIX_ROW_PADDING at the pin where its kernels may read past a
  // block-quantized tensor (row_size(step - K mod step) bytes when K is not
  // a multiple of it); 0 where nothing is read past a tensor.
  std::uint32_t over_read_row_step = 0;
  // A row of each part starts K-proportionally into the part, so rows are
  // read alone (a table's lookups, a row range's decode). False where a
  // part is tiled across rows (scale atoms, trellis tiles).
  bool rows_addressable = true;
  // Holds codes and scales in two parts (the encoding's PartSpec list),
  // rather than together as one.
  bool split_parts = false;
};

std::span<const LayoutInfo> Layouts();
const LayoutInfo& Info(Layout layout);
std::optional<Layout> LayoutByName(std::string_view name);

// Whether `layout` is a single-part dense or block arrangement.
constexpr bool IsSinglePartBlockOrDense(Layout layout) {
  return layout == Layout::kDense || layout == Layout::kGgmlBlocks;
}

inline constexpr std::uint64_t kGgmlRowPadding = 512;  // MATRIX_ROW_PADDING at the pin

// The scale atoms' geometry (kScaleAtoms).
inline constexpr std::uint64_t kScaleAtomRows = 128;
inline constexpr std::uint64_t kScaleAtomBlocks = 4;
inline constexpr std::uint64_t kScaleAtomBytes = 512;

// The byte offset of row `row`'s scale of block `block` in the scale atoms
// of a matrix with `blocks` scales a row (a multiple of kScaleAtomBlocks):
// atoms of 128 rows by 4 scales, along K first; in an atom, row r's four
// scales at (r mod 32) · 16 + (r mod 128) / 32 · 4.
constexpr std::uint64_t ScaleAtomOffset(std::uint64_t row, std::uint64_t block,
                                        std::uint64_t blocks) {
  return ((((row / kScaleAtomRows) * (blocks / kScaleAtomBlocks)) + (block / kScaleAtomBlocks)) *
          kScaleAtomBytes) +
         ((row % 32) * 16) + (((row % kScaleAtomRows) / 32) * 4) + (block % kScaleAtomBlocks);
}

}  // namespace llmp::format

#endif  // LLMP_FORMAT_LAYOUT_H_
