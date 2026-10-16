// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Tensor formats (format/, D-108): the encoding and layout tables, the size
// rule on hand-computed shapes of every layout, its refusals, the scale
// atoms' offsets against the CUTLASS expert layout's, and which state
// conversions are lossless. Parity with v0's size rule (representation.cc)
// is artifact_format_map_test's.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <set>
#include <string>
#include <string_view>
#include <vector>

#include "format/encoding.h"
#include "format/layout.h"
#include "format/tensor_format.h"
#include "kernels/ggml/moe_cutlass.h"
#include "kernels/ggml/moe_layout.h"

namespace {

namespace format = llmp::format;
using format::Encoding;
using format::FormatError;
using format::Layout;
using format::PartRole;
using format::TensorFormat;

TensorFormat Matrix(Encoding e, Layout l, std::uint64_t k, std::uint64_t n,
                    std::uint64_t matrices = 1) {
  TensorFormat f{.encoding = e, .layout = l};
  f.shape = {.ne = {k, n, matrices, 1}, .rank = static_cast<std::uint8_t>(matrices > 1 ? 3 : 2)};
  return f;
}

format::PartList Parts(const TensorFormat& f) {
  const auto parts = format::PartsOf(f);
  EXPECT_TRUE(parts.has_value()) << (parts ? "" : format::FormatErrorName(parts.error()));
  return parts.value_or(format::PartList{});
}

FormatError Refusal(const TensorFormat& f) {
  const auto parts = format::PartsOf(f);
  EXPECT_FALSE(parts.has_value());
  return parts ? FormatError::kUnknown : parts.error();
}

TEST(FormatTest, TablesAreCompleteAndNamed) {
  std::set<std::string_view> names;
  for (std::size_t i = 0; i < format::Encodings().size(); ++i) {
    const format::EncodingInfo& info = format::Encodings()[i];
    EXPECT_EQ(static_cast<std::size_t>(info.id), i);
    EXPECT_EQ(&format::Info(info.id), &info);
    EXPECT_EQ(format::EncodingByName(info.name), info.id) << info.name;
    EXPECT_TRUE(names.insert(info.name).second) << info.name;
    for (const char c : info.name) {
      EXPECT_TRUE((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_') << info.name;
    }
    EXPECT_GE(info.block_elements, 1U);
    if (!info.quantized) {
      EXPECT_EQ(info.block_elements, 1U) << info.name;
      EXPECT_EQ(info.element_bits % 8, 0U) << info.name;
      EXPECT_TRUE(info.parts.empty()) << info.name;
    }
    if (info.ggml && info.quantized && info.id != Encoding::kNvfp4) {
      EXPECT_EQ(info.ggml->block_elements, info.block_elements) << info.name;
    }
  }
  EXPECT_EQ(format::Encodings().size(), static_cast<std::size_t>(Encoding::kAffine) + 1);
  EXPECT_FALSE(format::EncodingByName("Q4_K"));  // names are lower-case
  EXPECT_EQ(format::EncodingByName("q4_k"), Encoding::kQ4_K);
  EXPECT_EQ(format::Info(Encoding::kNvfp4).ggml->block_elements, 64U);
  EXPECT_EQ(format::Info(Encoding::kNvfp4).block_elements, 16U);
  for (std::size_t i = 0; i < format::Layouts().size(); ++i) {
    const format::LayoutInfo& info = format::Layouts()[i];
    EXPECT_EQ(static_cast<std::size_t>(info.id), i);
    EXPECT_EQ(format::LayoutByName(info.name), info.id) << info.name;
  }
  EXPECT_EQ(format::PartRoleName(PartRole::kBlockScales), "block_scales");
}

TEST(FormatTest, DenseAndGgmlBlocksSizeAsTheirElementsAndBlocks) {
  const auto dense = Parts(Matrix(Encoding::kBF16, Layout::kDense, 2560, 7, 3));
  ASSERT_EQ(dense.size(), 1U);
  EXPECT_EQ(dense[0],
            (format::PartExtent{PartRole::kData, 2560ULL * 7 * 3 * 2, 2560ULL * 7 * 3 * 2}));
  // Q4_K, K = 2560: 10 blocks of 144 bytes a row; 2560 mod 512 = 0, so
  // nothing is read past it.
  const auto q4k = Parts(Matrix(Encoding::kQ4_K, Layout::kGgmlBlocks, 2560, 10));
  EXPECT_EQ(q4k[0].bytes, 14400U);
  EXPECT_EQ(q4k[0].readable, 14400U);
  // Q8_0, K = 640: 640 mod 512 = 128, so GGML may read 384 values' row
  // (12 blocks of 34 bytes) past the last row.
  const auto q8 = Parts(Matrix(Encoding::kQ8_0, Layout::kGgmlBlocks, 640, 3));
  EXPECT_EQ(q8[0].bytes, 3U * 20 * 34);
  EXPECT_EQ(q8[0].readable, (3U * 20 * 34) + (12U * 34));
  // GGML's NVFP4 blocks: 64 values in 36 bytes, the same over-read rule.
  const auto nv = Parts(Matrix(Encoding::kNvfp4, Layout::kGgmlNvfp4Blocks, 640, 2560));
  EXPECT_EQ(nv[0].bytes, 2560U * 10 * 36);
  EXPECT_EQ(nv[0].readable, (2560U * 10 * 36) + (6U * 36));
  EXPECT_EQ(format::RowBytes(Matrix(Encoding::kNvfp4, Layout::kGgmlNvfp4Blocks, 640, 2560),
                             PartRole::kData),
            360U);
  // A scalar: rank 0.
  const auto scalar = Parts({.encoding = Encoding::kI32});
  EXPECT_EQ(scalar[0].bytes, 4U);
}

TEST(FormatTest, SplitInterleavedAndAtomLayoutsSizeTheirParts) {
  // MXFP8 [K = 2560, N = 512]: E4M3 codes and an E8M0 scale per 32.
  const TensorFormat mx = Matrix(Encoding::kMxfp8, Layout::kSplit, 2560, 512);
  const auto mx_parts = Parts(mx);
  ASSERT_EQ(mx_parts.size(), 2U);
  EXPECT_EQ(mx_parts[0], (format::PartExtent{PartRole::kCodes, 2560U * 512, 2560U * 512}));
  EXPECT_EQ(mx_parts[1], (format::PartExtent{PartRole::kBlockScales, 80U * 512, 80U * 512}));
  EXPECT_EQ(format::RowBytes(mx, PartRole::kCodes), 2560U);
  EXPECT_EQ(format::RowBytes(mx, PartRole::kBlockScales), 80U);
  EXPECT_EQ(format::RowBytes(mx, PartRole::kData).error(), FormatError::kParts);

  // NVFP4 split with a tensor scale per stacked matrix.
  TensorFormat nv = Matrix(Encoding::kNvfp4, Layout::kSplit, 2560, 1280, 4);
  nv.params.tensor_scale = true;
  nv.stacked = 2;
  const auto nv_parts = Parts(nv);
  ASSERT_EQ(nv_parts.size(), 3U);
  EXPECT_EQ(nv_parts[0].bytes, 4U * 1280 * 1280);
  EXPECT_EQ(nv_parts[1].bytes, 4U * 1280 * 160);
  EXPECT_EQ(nv_parts[2], (format::PartExtent{PartRole::kTensorScale, 32, 32}));
  EXPECT_EQ(format::RowBytes(nv, PartRole::kTensorScale).error(), FormatError::kLayout);

  // Qwen3.8's n-gram table: 160 values a row, 80 code bytes then 10 scales.
  const TensorFormat table = Matrix(Encoding::kNvfp4, Layout::kRowInterleaved, 160, 37);
  EXPECT_EQ(Parts(table)[0].bytes, 37U * 90);
  EXPECT_EQ(format::RowBytes(table, PartRole::kData), 90U);

  // The CUTLASS expert layout (kernels/ggml/moe_layout.h): gate and up of
  // f = 640 rows each over w = 2560, and down.
  const TensorFormat gate_up = Matrix(Encoding::kNvfp4, Layout::kScaleAtoms, 2560, 1280);
  const llmp::kernels::ggml::moe::ExpertLayout expert{.ffn = 640, .width = 2560};
  const auto gu = Parts(gate_up);
  EXPECT_EQ(gu[0].bytes, expert.gate_up_scales());
  EXPECT_EQ(gu[1].bytes, expert.down_codes() - expert.gate_up_scales());
  const auto down = Parts(Matrix(Encoding::kNvfp4, Layout::kScaleAtoms, 640, 2560));
  EXPECT_EQ(down[0].bytes, expert.down_scales() - expert.down_codes());
  EXPECT_EQ(down[1].bytes, expert.bytes() - expert.down_scales());
  // Scales pad rows to a whole 128-row atom.
  const auto padded = Parts(Matrix(Encoding::kMxfp8, Layout::kScaleAtoms, 256, 130));
  EXPECT_EQ(padded[1].bytes, 256U * 8);
  EXPECT_EQ(format::RowBytes(gate_up, PartRole::kBlockScales).error(), FormatError::kLayout);
}

TEST(FormatTest, TheTrellisHasItsCodesSignsAndMarker) {
  TensorFormat t = Matrix(Encoding::kExl3Trellis, Layout::kExl3Tiles, 896, 4864);
  t.params.k_bits = 5;
  t.params.codebook = format::Exl3Codebook::kMcg;
  const auto parts = Parts(t);
  ASSERT_EQ(parts.size(), 4U);
  // I16 [896 / 16, 4864 / 16, 16 · 5].
  EXPECT_EQ(parts[0],
            (format::PartExtent{PartRole::kCodes, 56U * 304 * 80 * 2, 56U * 304 * 80 * 2}));
  EXPECT_EQ(parts[1].bytes, 896U * 2);
  EXPECT_EQ(parts[2].bytes, 4864U * 2);
  EXPECT_EQ(parts[3], (format::PartExtent{PartRole::kCodebookMarker, 4, 4}));
  EXPECT_EQ(format::RowBytes(t, PartRole::kCodes).error(), FormatError::kLayout);
  t.params.k_bits = 0;
  EXPECT_EQ(Refusal(t), FormatError::kParams);
  t.params.k_bits = 9;
  EXPECT_EQ(Refusal(t), FormatError::kParams);
  t.params.k_bits = 4;
  t.params.codebook = format::Exl3Codebook::kNone;
  EXPECT_EQ(Refusal(t), FormatError::kParams);
  t.params.codebook = format::Exl3Codebook::kMcg;
  t.shape.ne[1] = 4865;
  EXPECT_EQ(Refusal(t), FormatError::kBlock);
  // mul1 has its marker too; a 3inst trellis has none.
  t.shape.ne[1] = 4864;
  t.params.codebook = format::Exl3Codebook::kMul1;
  EXPECT_EQ(Parts(t).size(), 4U);
  t.params.codebook = format::Exl3Codebook::k3Inst;
  const auto three = Parts(t);
  ASSERT_EQ(three.size(), 3U);
  EXPECT_EQ(three[2].role, PartRole::kOutputSigns);
}

// The affine encoding (MLX's affine mode, AWQ's and GPTQ's arithmetic):
// codes K · bits / 8 bytes a row, K / group scales of the scale type, and
// as many biases, or zero points of `bits` bits padded to a byte.
TEST(FormatTest, AffineSizesAsItsBitsGroupsAndScales) {
  const auto affine = [](std::uint64_t k, std::uint64_t n, std::uint8_t bits, std::uint32_t group,
                         format::AffineScale scale, bool zeros = false,
                         std::uint64_t matrices = 1) {
    TensorFormat f = Matrix(Encoding::kAffine, Layout::kSplit, k, n, matrices);
    f.params.bits = bits;
    f.params.group = group;
    f.params.scale = scale;
    f.params.zero_point = zeros;
    return f;
  };
  // MLX's 4-bit, group 32, BF16 (Qwen3.8's q_proj: [12288, 320] U32
  // codes, [12288, 80] scales and biases).
  const TensorFormat q = affine(2560, 12288, 4, 32, format::AffineScale::kBF16);
  const auto parts = Parts(q);
  ASSERT_EQ(parts.size(), 3U);
  EXPECT_EQ(parts[0], (format::PartExtent{PartRole::kCodes, 12288U * 320 * 4, 12288U * 320 * 4}));
  EXPECT_EQ(parts[1],
            (format::PartExtent{PartRole::kBlockScales, 12288U * 80 * 2, 12288U * 80 * 2}));
  EXPECT_EQ(parts[2], (format::PartExtent{PartRole::kBiases, 12288U * 80 * 2, 12288U * 80 * 2}));
  EXPECT_EQ(format::RowBytes(q, PartRole::kCodes).value(), 1280U);
  EXPECT_EQ(format::RowBytes(q, PartRole::kBlockScales).value(), 160U);
  EXPECT_EQ(format::RowBytes(q, PartRole::kBiases).value(), 160U);
  EXPECT_EQ(format::RowBytes(q, PartRole::kZeros).error(), FormatError::kParts);
  // Every width packs a row's codes as one bit stream: 3, 5 and 6 bits too.
  for (std::uint8_t bits = 1; bits <= 8; ++bits) {
    EXPECT_EQ(Parts(affine(64, 3, bits, 64, format::AffineScale::kF16))[0].bytes, 3U * 8 * bits)
        << static_cast<int>(bits);
  }
  // F32 scales; expert arrays; zero points padded to a byte a row.
  EXPECT_EQ(Parts(affine(128, 2, 4, 64, format::AffineScale::kF32))[1].bytes, 2U * 2 * 4);
  EXPECT_EQ(Parts(affine(2560, 640, 4, 32, format::AffineScale::kBF16, false, 512))[0].bytes,
            512U * 640 * 1280);
  const TensorFormat z = affine(384, 5, 3, 128, format::AffineScale::kF16, true);
  const auto zp = Parts(z);
  ASSERT_EQ(zp.size(), 3U);
  EXPECT_EQ(zp[2], (format::PartExtent{PartRole::kZeros, 5U * 2, 5U * 2}));  // 9 bits: 2 bytes
  EXPECT_EQ(format::RowBytes(z, PartRole::kZeros).value(), 2U);
  // Refused: a row of partial groups, groups of partial bytes, widths and
  // scale types outside the encoding, its parameters on another encoding,
  // and layouts other than kSplit.
  EXPECT_EQ(Refusal(affine(96, 2, 4, 64, format::AffineScale::kF16)), FormatError::kBlock);
  EXPECT_EQ(Refusal(affine(96, 2, 4, 12, format::AffineScale::kF16)), FormatError::kParams);
  EXPECT_EQ(Refusal(affine(96, 2, 4, 0, format::AffineScale::kF16)), FormatError::kParams);
  EXPECT_EQ(Refusal(affine(96, 2, 0, 32, format::AffineScale::kF16)), FormatError::kParams);
  EXPECT_EQ(Refusal(affine(96, 2, 9, 32, format::AffineScale::kF16)), FormatError::kParams);
  EXPECT_EQ(Refusal(affine(96, 2, 4, 32, format::AffineScale::kNone)), FormatError::kParams);
  EXPECT_EQ(Refusal(affine(96, 2, 4, 32, static_cast<format::AffineScale>(9))),
            FormatError::kParams);
  TensorFormat scaled = affine(96, 2, 4, 32, format::AffineScale::kF16);
  scaled.params.tensor_scale = true;
  EXPECT_EQ(Refusal(scaled), FormatError::kParams);
  TensorFormat other = Matrix(Encoding::kMxfp8, Layout::kSplit, 64, 2);
  other.params.group = 32;
  EXPECT_EQ(Refusal(other), FormatError::kParams);
  other = Matrix(Encoding::kF16, Layout::kDense, 64, 2);
  other.params.scale = format::AffineScale::kF16;
  EXPECT_EQ(Refusal(other), FormatError::kParams);
  for (const Layout l : {Layout::kDense, Layout::kGgmlBlocks, Layout::kRowInterleaved,
                         Layout::kScaleAtoms, Layout::kExl3Tiles}) {
    TensorFormat f = affine(256, 128, 4, 32, format::AffineScale::kF16);
    f.layout = l;
    EXPECT_EQ(Refusal(f), FormatError::kLayout) << static_cast<int>(l);
  }
  // Past 64 bits.
  EXPECT_EQ(Refusal(affine(std::uint64_t{1} << 40U, std::uint64_t{1} << 30U, 8, 64,
                           format::AffineScale::kF32)),
            FormatError::kOverflow);
  // Quantized: exact only into itself.
  EXPECT_TRUE(format::Lossless(Encoding::kAffine, Encoding::kAffine));
  EXPECT_FALSE(format::Lossless(Encoding::kAffine, Encoding::kF32));
  EXPECT_FALSE(format::Lossless(Encoding::kQ4_1, Encoding::kAffine));
  EXPECT_FALSE(format::IsStateEncoding(Encoding::kAffine));
  EXPECT_EQ(format::EncodingByName("affine"), Encoding::kAffine);
  EXPECT_EQ(format::PartRoleName(PartRole::kBiases), "biases");
  EXPECT_EQ(format::PartRoleName(PartRole::kZeros), "zeros");
}

TEST(FormatTest, RefusesWhatTheRuleDoesNotTake) {
  // Layouts that do not hold the encoding.
  EXPECT_EQ(Refusal(Matrix(Encoding::kQ4_K, Layout::kDense, 256, 1)), FormatError::kLayout);
  EXPECT_EQ(Refusal(Matrix(Encoding::kF32, Layout::kGgmlBlocks, 256, 1)), FormatError::kLayout);
  EXPECT_EQ(Refusal(Matrix(Encoding::kNvfp4, Layout::kGgmlBlocks, 256, 1)), FormatError::kLayout);
  EXPECT_EQ(Refusal(Matrix(Encoding::kQ8_0, Layout::kGgmlNvfp4Blocks, 256, 1)),
            FormatError::kLayout);
  EXPECT_EQ(Refusal(Matrix(Encoding::kQ4_K, Layout::kSplit, 256, 1)), FormatError::kLayout);
  EXPECT_EQ(Refusal(Matrix(Encoding::kMxfp8, Layout::kExl3Tiles, 256, 16)), FormatError::kLayout);
  // Rows that are not whole blocks, scales not whole atoms.
  EXPECT_EQ(Refusal(Matrix(Encoding::kQ4_K, Layout::kGgmlBlocks, 300, 1)), FormatError::kBlock);
  EXPECT_EQ(Refusal(Matrix(Encoding::kNvfp4, Layout::kGgmlNvfp4Blocks, 48, 1)),
            FormatError::kBlock);
  EXPECT_EQ(Refusal(Matrix(Encoding::kNvfp4, Layout::kScaleAtoms, 32, 128)), FormatError::kBlock);
  // Shapes.
  TensorFormat f = Matrix(Encoding::kF16, Layout::kDense, 4, 4);
  f.shape.ne[0] = 0;
  EXPECT_EQ(Refusal(f), FormatError::kShape);
  f = Matrix(Encoding::kF16, Layout::kDense, 4, 4);
  f.shape.ne[3] = 2;  // beyond rank 2
  EXPECT_EQ(Refusal(f), FormatError::kShape);
  f = Matrix(Encoding::kF16, Layout::kDense, 4, 5);
  f.stacked = 2;
  EXPECT_EQ(Refusal(f), FormatError::kShape);
  f.stacked = 0;
  EXPECT_EQ(Refusal(f), FormatError::kShape);
  // Parameters an encoding does not take.
  f = Matrix(Encoding::kMxfp8, Layout::kSplit, 64, 4);
  f.params.tensor_scale = true;
  EXPECT_EQ(Refusal(f), FormatError::kParams);
  f = Matrix(Encoding::kQ8_0, Layout::kGgmlBlocks, 64, 4);
  f.params.k_bits = 4;
  EXPECT_EQ(Refusal(f), FormatError::kParams);
  // Outside the enums.
  f = Matrix(Encoding::kF16, Layout::kDense, 4, 4);
  f.encoding = static_cast<Encoding>(200);
  EXPECT_EQ(Refusal(f), FormatError::kUnknown);
  f = Matrix(Encoding::kF16, Layout::kDense, 4, 4);
  f.layout = static_cast<Layout>(200);
  EXPECT_EQ(Refusal(f), FormatError::kUnknown);
  // Sizes past 64 bits.
  EXPECT_EQ(Refusal(Matrix(Encoding::kF32, Layout::kDense, 1ULL << 40, 1ULL << 30)),
            FormatError::kOverflow);
  EXPECT_EQ(
      Refusal(Matrix(Encoding::kQ8_0, Layout::kGgmlBlocks, 1ULL << 40, 1ULL << 30, 1ULL << 20)),
      FormatError::kOverflow);
  EXPECT_FALSE(format::ShapeOf(std::vector<std::uint64_t>{1, 2, 3, 4, 5}));
  EXPECT_EQ(format::ShapeOf(std::vector<std::uint64_t>{7, 3})->ne,
            (std::array<std::uint64_t, 4>{7, 3, 1, 1}));
}

TEST(FormatTest, ScaleAtomOffsetsAreTheCutlassLayouts) {
  namespace moe = llmp::kernels::ggml::moe;
  for (const std::uint64_t blocks : {4ULL, 40ULL, 160ULL}) {
    std::set<std::uint64_t> seen;
    for (std::uint64_t row = 0; row < 384; ++row) {
      for (std::uint64_t b = 0; b < blocks; ++b) {
        const std::uint64_t at = format::ScaleAtomOffset(row, b, blocks);
        ASSERT_EQ(at, moe::SfOffset(row, b, blocks)) << row << " " << b;
        seen.insert(at);
      }
    }
    // A permutation of the padded scale bytes.
    EXPECT_EQ(seen.size(), 384 * blocks);
    EXPECT_EQ(*seen.rbegin(), (384 * blocks) - 1);
  }
}

TEST(FormatTest, LosslessIsExactValueInclusion) {
  EXPECT_TRUE(format::Lossless(Encoding::kF16, Encoding::kF16));
  EXPECT_TRUE(format::Lossless(Encoding::kF16, Encoding::kF32));
  EXPECT_TRUE(format::Lossless(Encoding::kBF16, Encoding::kF32));
  EXPECT_TRUE(format::Lossless(Encoding::kF32, Encoding::kF64));
  EXPECT_TRUE(format::Lossless(Encoding::kF8E4M3, Encoding::kF16));
  EXPECT_TRUE(format::Lossless(Encoding::kF8E4M3, Encoding::kBF16));
  EXPECT_TRUE(format::Lossless(Encoding::kF8E5M2, Encoding::kF16));
  EXPECT_TRUE(format::Lossless(Encoding::kE8M0, Encoding::kF32));
  EXPECT_TRUE(format::Lossless(Encoding::kI8, Encoding::kF16));
  EXPECT_TRUE(format::Lossless(Encoding::kU8, Encoding::kBF16));
  EXPECT_TRUE(format::Lossless(Encoding::kI16, Encoding::kF32));
  EXPECT_TRUE(format::Lossless(Encoding::kI32, Encoding::kF64));
  EXPECT_TRUE(format::Lossless(Encoding::kI8, Encoding::kI16));
  EXPECT_TRUE(format::Lossless(Encoding::kU8, Encoding::kI16));
  EXPECT_TRUE(format::Lossless(Encoding::kBool, Encoding::kU8));

  EXPECT_FALSE(format::Lossless(Encoding::kF32, Encoding::kF16));
  EXPECT_FALSE(format::Lossless(Encoding::kF16, Encoding::kBF16));
  EXPECT_FALSE(format::Lossless(Encoding::kBF16, Encoding::kF16));
  EXPECT_FALSE(format::Lossless(Encoding::kF8E5M2, Encoding::kF8E4M3));  // infinities
  EXPECT_FALSE(format::Lossless(Encoding::kE8M0, Encoding::kF16));
  EXPECT_FALSE(format::Lossless(Encoding::kF16, Encoding::kE8M0));
  EXPECT_FALSE(format::Lossless(Encoding::kI32, Encoding::kF32));
  EXPECT_FALSE(format::Lossless(Encoding::kU16, Encoding::kF16));
  EXPECT_FALSE(format::Lossless(Encoding::kU8, Encoding::kI8));
  EXPECT_FALSE(format::Lossless(Encoding::kI8, Encoding::kU8));
  EXPECT_FALSE(format::Lossless(Encoding::kF16, Encoding::kI32));
  // Quantized encodings: not established, either way.
  EXPECT_FALSE(format::Lossless(Encoding::kF16, Encoding::kQ8_0));
  EXPECT_FALSE(format::Lossless(Encoding::kQ8_0, Encoding::kF32));
  EXPECT_TRUE(format::Lossless(Encoding::kQ8_0, Encoding::kQ8_0));
}

// The state encodings: the dense floats and the blocks GGML's attention
// reads at the pin, each storable through GGML's own quantization (a
// single-part format of whole blocks); exact or lossy for a state by
// Lossless from its own encoding: an F16 KV cache stores exactly as F16 or
// F32, lossily as BF16 or any block; an F32 recurrent state only as F32.
TEST(FormatTest, StateEncodingsAreGgmlsAttentionTypes) {
  const std::vector<Encoding> want = {Encoding::kF32,  Encoding::kF16,  Encoding::kBF16,
                                      Encoding::kQ4_0, Encoding::kQ4_1, Encoding::kQ5_0,
                                      Encoding::kQ5_1, Encoding::kQ8_0};
  const auto states = format::StateEncodings();
  EXPECT_EQ(std::vector<Encoding>(states.begin(), states.end()), want);
  for (const format::EncodingInfo& info : format::Encodings()) {
    const bool state = std::ranges::find(want, info.id) != want.end();
    EXPECT_EQ(format::IsStateEncoding(info.id), state) << info.name;
    if (!state) {
      continue;
    }
    // One part, and a GGML type to lower to.
    EXPECT_TRUE(info.parts.empty()) << info.name;
    EXPECT_TRUE(info.ggml.has_value()) << info.name;
    const bool exact_for_f16 = info.id == Encoding::kF16 || info.id == Encoding::kF32;
    EXPECT_EQ(format::Lossless(Encoding::kF16, info.id), exact_for_f16) << info.name;
    EXPECT_EQ(format::Lossless(Encoding::kF32, info.id), info.id == Encoding::kF32) << info.name;
  }
  EXPECT_TRUE(format::Lossless(Encoding::kBF16, Encoding::kF32));
  EXPECT_FALSE(format::Lossless(Encoding::kBF16, Encoding::kF16));
}

}  // namespace
