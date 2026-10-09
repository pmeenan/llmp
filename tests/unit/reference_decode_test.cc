// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The CPU reference decoders (kernels/reference/decode.h, D-108): element
// conversions against GGML's and hand-computed values; GGML's block
// encodings through its own dequantization, at any matrix and row; GGML's
// reference quantization round trips; NVFP4 in its four layouts (one set
// of values, permuted) and against GGML's dequantization of its blocks;
// MXFP4 likewise; MXFP8 exactly, as qwen38_ops_test's device fixtures
// check it; tensor scales; the EXL3 trellis's codebooks, tile layout and
// transforms against values and products derived apart from the
// reference (exl3_trellis_test checks it against ExLlamaV3's own kernels);
// every state encoding's encode and decode; refusals; and, where the M3 store holds them (a
// Spark), the routed experts of both Qwen3.8 imports (GGML's blocks and the
// CUTLASS layout) decoding to the same values.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <limits>
#include <random>
#include <span>
#include <string>
#include <vector>

#include "artifact/artifact.h"
#include "artifact/format_map.h"
#include "format/encoding.h"
#include "format/layout.h"
#include "format/tensor_format.h"
#include "ggml.h"
#include "kernels/reference/decode.h"

namespace {

namespace format = llmp::format;
namespace ref = llmp::kernels::reference;
using format::Encoding;
using format::Layout;
using format::PartRole;
using format::TensorFormat;

// OCP E4M3 (bias 7, no infinities, 0x7F and 0xFF NaN), as qwen38_ops_test.
double E4m3(std::uint8_t b) {
  const int e = (b >> 3) & 0xF;
  const int m = b & 7;
  if (e == 15 && m == 7) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  const double v = e == 0 ? std::ldexp(m / 8.0, -6) : std::ldexp(1.0 + (m / 8.0), e - 7);
  return (b & 0x80) != 0 ? -v : v;
}

double E2m1(std::uint8_t code) {
  constexpr std::array<double, 8> kValues = {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0};
  return ((code & 8) != 0 ? -1.0 : 1.0) * kValues.at(code & 7);
}

bool SameBits(float a, float b) {
  return std::bit_cast<std::uint32_t>(a) == std::bit_cast<std::uint32_t>(b) ||
         (std::isnan(a) && std::isnan(b));
}

// Equal values: GGML's FP4 tables hold no negative zero (its E2M1 values
// are int8 twice the value), so its -0 codes decode to +0.
bool SameValue(float a, float b) { return a == b || (std::isnan(a) && std::isnan(b)); }

std::vector<std::byte> RandomBytes(std::uint64_t seed, std::size_t n) {
  std::mt19937_64 random(seed);
  std::vector<std::byte> out(n);
  for (std::byte& b : out) {
    b = static_cast<std::byte>(random() & 0xFF);
  }
  return out;
}

std::vector<float> Normal(std::uint64_t seed, std::size_t n) {
  std::mt19937_64 random(seed);
  std::normal_distribution<float> normal(0.0f, 0.05f);
  std::vector<float> out(n);
  for (float& v : out) {
    v = normal(random);
  }
  return out;
}

TensorFormat Matrix(Encoding e, Layout l, std::uint64_t k, std::uint64_t n,
                    std::uint64_t matrices = 1) {
  TensorFormat f{.encoding = e, .layout = l};
  f.shape = {.ne = {k, n, matrices, 1}, .rank = static_cast<std::uint8_t>(matrices > 1 ? 3 : 2)};
  return f;
}

std::vector<float> Decode(const TensorFormat& f, std::span<const ref::PartView> parts,
                          std::uint64_t matrix, std::uint64_t first, std::uint64_t count) {
  std::vector<float> out(count * f.shape.ne[0]);
  const auto ok = ref::DecodeRows(f, parts, matrix, first, count, out);
  EXPECT_TRUE(ok.has_value()) << (ok ? "" : ref::DecodeErrorName(ok.error()));
  return out;
}

TEST(ReferenceDecodeTest, ElementConversionsAreGgmlsAndOcps) {
  for (std::uint32_t h = 0; h < 0x10000; ++h) {
    const auto bits = static_cast<std::uint16_t>(h);
    ASSERT_TRUE(SameBits(ref::F16ToF32(bits), ggml_fp16_to_fp32(bits))) << h;
    ggml_bf16_t bf{bits};
    ASSERT_TRUE(SameBits(ref::Bf16ToF32(bits), ggml_bf16_to_fp32(bf))) << h;
  }
  std::mt19937 random(3);  // NOLINT(bugprone-random-generator-seed): reproducible
  std::vector<std::uint32_t> samples = {0x00000000, 0x80000000, 0x7F800000, 0xFF800000,
                                        0x7FC00001, 0x477FE000, 0x477FEFFF, 0x477FF000,
                                        0x38800000, 0x387FFFFF, 0x33000000, 0x33000001,
                                        0x32FFFFFF, 0x337FFFFF, 0x00000001, 0x3F800000};
  for (int i = 0; i < 200000; ++i) {
    samples.push_back(static_cast<std::uint32_t>(random()));
  }
  for (const std::uint32_t x : samples) {
    const auto f = std::bit_cast<float>(x);
    if (std::isnan(f)) {  // NaN stays NaN; payloads may differ by platform
      EXPECT_TRUE(std::isnan(ref::F16ToF32(ref::F32ToF16(f))));
      EXPECT_TRUE(std::isnan(ref::Bf16ToF32(ref::F32ToBf16(f))));
      continue;
    }
    ASSERT_EQ(ref::F32ToF16(f), ggml_fp32_to_fp16(f)) << std::hex << x;
    ASSERT_EQ(ref::F32ToBf16(f), ggml_fp32_to_bf16(f).bits) << std::hex << x;
  }
  for (std::uint32_t b = 0; b < 256; ++b) {
    const auto byte = static_cast<std::uint8_t>(b);
    const double e4m3 = E4m3(byte);
    if (std::isnan(e4m3)) {
      EXPECT_TRUE(std::isnan(ref::E4m3ToF32(byte))) << b;
    } else {
      EXPECT_EQ(static_cast<double>(ref::E4m3ToF32(byte)), e4m3) << b;
    }
    EXPECT_TRUE(
        SameBits(ref::E5m2ToF32(byte), ggml_fp16_to_fp32(static_cast<ggml_fp16_t>(b << 8U))))
        << b;
    if (b == 0xFF) {
      EXPECT_TRUE(std::isnan(ref::E8m0ToF32(byte)));
    } else {
      EXPECT_EQ(static_cast<double>(ref::E8m0ToF32(byte)),
                std::ldexp(1.0, static_cast<int>(b) - 127));
    }
  }
  for (std::uint8_t code = 0; code < 16; ++code) {
    EXPECT_EQ(static_cast<double>(ref::E2m1ToF32(code)), E2m1(code));
  }
}

// Every GGML block encoding with a CPU dequantization decodes, at any
// matrix and row, as GGML's own to_float of those bytes.
TEST(ReferenceDecodeTest, GgmlBlockEncodingsAreGgmlsDequantization) {
  constexpr std::uint64_t kRows = 7;
  constexpr std::uint64_t kMatrices = 3;
  int checked = 0;
  for (const format::EncodingInfo& info : format::Encodings()) {
    if (!info.quantized || !info.ggml || info.id == Encoding::kNvfp4 ||
        info.id == Encoding::kMxfp4) {
      continue;
    }
    SCOPED_TRACE(std::string(info.name));
    const std::uint64_t k = 2 * std::max<std::uint64_t>(info.ggml->block_elements, 256);
    const TensorFormat f = Matrix(info.id, Layout::kGgmlBlocks, k, kRows, kMatrices);
    const auto parts = format::PartsOf(f);
    ASSERT_TRUE(parts.has_value());
    const std::vector<std::byte> data = RandomBytes(info.ggml->type, (*parts)[0].bytes);
    const ref::PartView view{.role = PartRole::kData, .bytes = data};
    const std::vector<float> got = Decode(f, {&view, 1}, 1, 2, 4);
    const auto type = static_cast<ggml_type>(info.ggml->type);
    const std::size_t row = ggml_row_size(type, static_cast<std::int64_t>(k));
    std::vector<float> want(4 * k);
    const ggml_type_traits* traits = ggml_get_type_traits(type);
    for (std::uint64_t r = 0; r < 4; ++r) {
      const std::byte* src = data.data() + (((1 * kRows) + 2 + r) * row);
      if (traits->to_float != nullptr) {
        traits->to_float(src, want.data() + (r * k), static_cast<std::int64_t>(k));
        continue;
      }
      // Q8_1 and Q8_K: d · q (an F16 d and 32 values, or an F32 d and 256).
      const bool q8_1 = info.id == Encoding::kQ8_1;
      const std::uint64_t block = q8_1 ? 32 : 256;
      for (std::uint64_t b = 0; b < k / block; ++b) {
        const std::byte* x = src + (b * info.ggml->block_bytes);
        float d = 0.0f;
        if (q8_1) {
          std::uint16_t h = 0;
          std::memcpy(&h, x, 2);
          d = ggml_fp16_to_fp32(h);
        } else {
          std::memcpy(&d, x, 4);
        }
        for (std::uint64_t j = 0; j < block; ++j) {
          want[(r * k) + (b * block) + j] =
              d * static_cast<float>(static_cast<std::int8_t>(std::to_integer<int>(x[4 + j])));
        }
      }
    }
    for (std::size_t i = 0; i < got.size(); ++i) {
      ASSERT_TRUE(SameBits(got[i], want[i])) << i << ": " << got[i] << " " << want[i];
    }
    ++checked;
  }
  EXPECT_EQ(checked, 25);
}

// EncodeRows writes exactly GGML's reference quantization of those rows,
// and the dense conversions; what it does not take it refuses.
TEST(ReferenceDecodeTest, EncodeRowsIsGgmlsQuantization) {
  constexpr std::uint64_t kRows = 5;
  for (const Encoding e :
       {Encoding::kQ4_0, Encoding::kQ4_1, Encoding::kQ5_0, Encoding::kQ5_1, Encoding::kQ8_0,
        Encoding::kQ2_K, Encoding::kQ3_K, Encoding::kQ4_K, Encoding::kQ5_K, Encoding::kQ6_K,
        Encoding::kIQ4_NL, Encoding::kIQ4_XS, Encoding::kMxfp4, Encoding::kTQ2_0}) {
    const format::EncodingInfo& info = format::Info(e);
    SCOPED_TRACE(std::string(info.name));
    const TensorFormat f = Matrix(e, Layout::kGgmlBlocks, 512, kRows, 2);
    const auto parts = format::PartsOf(f);
    ASSERT_TRUE(parts.has_value());
    std::vector<std::byte> data((*parts)[0].bytes);
    const std::vector<float> rows = Normal(static_cast<std::uint64_t>(e), 2 * 512);
    ASSERT_TRUE(ref::EncodeRows(f, 1, 3, rows, data).has_value());
    const auto type = static_cast<ggml_type>(info.ggml->type);
    const std::size_t row = ggml_row_size(type, 512);
    std::vector<std::byte> want(2 * row);
    ggml_quantize_chunk(type, rows.data(), want.data(), 0, 2, 512, nullptr);
    EXPECT_EQ(std::memcmp(data.data() + ((kRows + 3) * row), want.data(), want.size()), 0);
    // And decodes as GGML dequantizes it.
    const ref::PartView view{.role = PartRole::kData, .bytes = data};
    const std::vector<float> back = Decode(f, {&view, 1}, 1, 3, 2);
    std::vector<float> theirs(2 * 512);
    ggml_get_type_traits(type)->to_float(want.data(), theirs.data(), 2 * 512);
    for (std::size_t i = 0; i < back.size(); ++i) {
      ASSERT_TRUE(SameValue(back[i], theirs[i])) << i << ": " << back[i] << " " << theirs[i];
    }
  }
  // Dense.
  const std::vector<float> rows = Normal(1, 3 * 64);
  for (const Encoding e : {Encoding::kF32, Encoding::kF16, Encoding::kBF16}) {
    const TensorFormat f = Matrix(e, Layout::kDense, 64, 3);
    std::vector<std::byte> data(format::PartsOf(f).value()[0].bytes);
    ASSERT_TRUE(ref::EncodeRows(f, 0, 0, rows, data).has_value());
    const ref::PartView view{.role = PartRole::kData, .bytes = data};
    const std::vector<float> back = Decode(f, {&view, 1}, 0, 0, 3);
    for (std::size_t i = 0; i < rows.size(); ++i) {
      const float want = e == Encoding::kF32   ? rows[i]
                         : e == Encoding::kF16 ? ggml_fp16_to_fp32(ggml_fp32_to_fp16(rows[i]))
                                               : ggml_bf16_to_fp32(ggml_fp32_to_bf16(rows[i]));
      ASSERT_TRUE(SameBits(back[i], want)) << i;
    }
  }
  // Refused: an importance-matrix type, the activation blocks, split parts.
  std::vector<std::byte> data(1 << 20);
  const std::vector<float> row = Normal(2, 256);
  EXPECT_EQ(
      ref::EncodeRows(Matrix(Encoding::kIQ2_XXS, Layout::kGgmlBlocks, 256, 1), 0, 0, row, data)
          .error(),
      ref::DecodeError::kUnsupported);
  EXPECT_EQ(ref::EncodeRows(Matrix(Encoding::kQ8_K, Layout::kGgmlBlocks, 256, 1), 0, 0, row, data)
                .error(),
            ref::DecodeError::kUnsupported);
  EXPECT_EQ(
      ref::EncodeRows(Matrix(Encoding::kMxfp8, Layout::kSplit, 256, 1), 0, 0, row, data).error(),
      ref::DecodeError::kUnsupported);
  EXPECT_EQ(ref::EncodeRows(Matrix(Encoding::kQ8_0, Layout::kGgmlBlocks, 256, 1), 0, 1, row, data)
                .error(),
            ref::DecodeError::kRange);
}

// One NVFP4 tensor, its codes and scales permuted into each layout:
// every layout decodes to the same values, and each value is
// E2M1(code) · E4M3(scale) · tensor scale.
TEST(ReferenceDecodeTest, Nvfp4LayoutsHoldTheSameValues) {
  constexpr std::uint64_t kK = 256;
  constexpr std::uint64_t kN = 160;  // the atoms pad it to 256 rows
  constexpr std::uint64_t kBlocks = kK / 16;
  const std::vector<std::byte> codes = RandomBytes(5, kN * kK / 2);
  std::vector<std::byte> scales = RandomBytes(6, kN * kBlocks);
  for (std::byte& s : scales) {
    s &= std::byte{0x7F};  // positive, as a checkpoint's
    if (s == std::byte{0x7F}) {
      s = std::byte{0x38};
    }
  }
  const float tensor_scale = 0.375f;
  const std::array<float, 1> ts_value = {tensor_scale};
  const std::span<const std::byte> ts = std::as_bytes(std::span(ts_value));
  auto code = [&](std::uint64_t r, std::uint64_t j) {
    const auto byte = std::to_integer<std::uint8_t>(codes[(r * kK / 2) + (j / 2)]);
    return static_cast<std::uint8_t>(j % 2 == 0 ? byte & 15 : byte >> 4);
  };
  auto scale = [&](std::uint64_t r, std::uint64_t b) {
    return std::to_integer<std::uint8_t>(scales[(r * kBlocks) + b]);
  };
  // The values, as qwen38_ops_test's Nvfp4TableRowsDequantizeExactly.
  std::vector<float> want(kN * kK);
  for (std::uint64_t r = 0; r < kN; ++r) {
    for (std::uint64_t j = 0; j < kK; ++j) {
      want[(r * kK) + j] = static_cast<float>(E2m1(code(r, j))) *
                           static_cast<float>(E4m3(scale(r, j / 16))) * tensor_scale;
    }
  }
  // Row-interleaved, scale atoms, and GGML's block_nvfp4.
  std::vector<std::byte> interleaved;
  std::vector<std::byte> atoms((kN + 127) / 128 * 128 * kBlocks);
  std::vector<std::byte> ggml(kN * (kK / 64) * 36);
  for (std::uint64_t r = 0; r < kN; ++r) {
    interleaved.insert(interleaved.end(), codes.begin() + static_cast<std::ptrdiff_t>(r * kK / 2),
                       codes.begin() + static_cast<std::ptrdiff_t>((r + 1) * kK / 2));
    interleaved.insert(interleaved.end(), scales.begin() + static_cast<std::ptrdiff_t>(r * kBlocks),
                       scales.begin() + static_cast<std::ptrdiff_t>((r + 1) * kBlocks));
    for (std::uint64_t b = 0; b < kBlocks; ++b) {
      atoms[format::ScaleAtomOffset(r, b, kBlocks)] = std::byte{scale(r, b)};
    }
    for (std::uint64_t g = 0; g < kK / 64; ++g) {
      std::byte* block = ggml.data() + (((r * (kK / 64)) + g) * 36);
      for (std::uint64_t s = 0; s < 4; ++s) {
        block[s] = std::byte{scale(r, (g * 4) + s)};
        for (std::uint64_t t = 0; t < 8; ++t) {
          const std::uint64_t j = (g * 64) + (s * 16) + t;
          block[4 + (s * 8) + t] =
              std::byte{static_cast<std::uint8_t>(code(r, j) | (code(r, j + 8) << 4))};
        }
      }
    }
  }
  struct Case {
    Layout layout;
    std::vector<ref::PartView> parts;
  };
  const std::vector<Case> cases = {
      {Layout::kSplit,
       {{PartRole::kCodes, codes}, {PartRole::kBlockScales, scales}, {PartRole::kTensorScale, ts}}},
      {Layout::kRowInterleaved, {{PartRole::kData, interleaved}, {PartRole::kTensorScale, ts}}},
      {Layout::kCutlassScaleAtoms,
       {{PartRole::kCodes, codes}, {PartRole::kBlockScales, atoms}, {PartRole::kTensorScale, ts}}},
      {Layout::kGgmlNvfp4Blocks, {{PartRole::kData, ggml}, {PartRole::kTensorScale, ts}}},
  };
  for (const Case& c : cases) {
    SCOPED_TRACE(std::string(format::Info(c.layout).name));
    TensorFormat f = Matrix(Encoding::kNvfp4, c.layout, kK, kN);
    f.params.tensor_scale = true;
    const std::vector<float> all = Decode(f, c.parts, 0, 0, kN);
    for (std::size_t i = 0; i < all.size(); ++i) {
      ASSERT_TRUE(SameBits(all[i], want[i])) << i;
    }
    const std::vector<float> some = Decode(f, c.parts, 0, 131, 3);
    for (std::size_t i = 0; i < some.size(); ++i) {
      ASSERT_TRUE(SameBits(some[i], want[(131 * kK) + i])) << i;
    }
  }
  // GGML's own dequantization of its blocks (scales it reads as unsigned,
  // here all positive) agrees, before the tensor scale.
  const ggml_type_traits* traits = ggml_get_type_traits(GGML_TYPE_NVFP4);
  std::vector<float> theirs(kN * kK);
  traits->to_float(ggml.data(), theirs.data(), static_cast<std::int64_t>(kN * kK));
  const ref::PartView view{.role = PartRole::kData, .bytes = ggml};
  const std::vector<float> ours =
      Decode(Matrix(Encoding::kNvfp4, Layout::kGgmlNvfp4Blocks, kK, kN), {&view, 1}, 0, 0, kN);
  for (std::size_t i = 0; i < ours.size(); ++i) {
    ASSERT_TRUE(SameValue(ours[i], theirs[i])) << i << ": " << ours[i] << " " << theirs[i];
  }
  // Where they differ: E4M3's NaN scale, which GGML reads as zero.
  std::vector<std::byte> nan_block(36, std::byte{0x22});
  nan_block[0] = std::byte{0x7F};
  const ref::PartView nan_view{.role = PartRole::kData, .bytes = nan_block};
  const std::vector<float> nan_row =
      Decode(Matrix(Encoding::kNvfp4, Layout::kGgmlNvfp4Blocks, 64, 1), {&nan_view, 1}, 0, 0, 1);
  EXPECT_TRUE(std::isnan(nan_row[0]));
  std::vector<float> ggml_nan(64);
  traits->to_float(nan_block.data(), ggml_nan.data(), 64);
  EXPECT_EQ(ggml_nan[0], 0.0f);
}

// MXFP4: GGML's block_mxfp4 and the split layout of the same values; and
// GGML's dequantization for every scale but 0xFF.
TEST(ReferenceDecodeTest, Mxfp4LayoutsHoldTheSameValues) {
  constexpr std::uint64_t kK = 128;
  constexpr std::uint64_t kN = 9;
  std::vector<std::byte> ggml = RandomBytes(8, kN * (kK / 32) * 17);
  for (std::size_t b = 0; b < ggml.size(); b += 17) {
    if (ggml[b] == std::byte{0xFF}) {
      ggml[b] = std::byte{0x80};
    }
  }
  std::vector<std::byte> codes(kN * kK / 2);
  std::vector<std::byte> scales(kN * kK / 32);
  for (std::uint64_t r = 0; r < kN; ++r) {
    for (std::uint64_t g = 0; g < kK / 32; ++g) {
      const std::byte* block = ggml.data() + (((r * (kK / 32)) + g) * 17);
      scales[(r * (kK / 32)) + g] = block[0];
      for (std::uint64_t w = 0; w < 32; ++w) {
        const auto byte = std::to_integer<std::uint8_t>(block[1 + (w % 16)]);
        const auto c = static_cast<std::uint8_t>(w < 16 ? byte & 15 : byte >> 4);
        const std::uint64_t j = (g * 32) + w;
        std::byte& out = codes[(r * kK / 2) + (j / 2)];
        out |= std::byte{static_cast<std::uint8_t>(j % 2 == 0 ? c : c << 4)};
      }
    }
  }
  const ref::PartView blocks{.role = PartRole::kData, .bytes = ggml};
  const std::vector<float> a =
      Decode(Matrix(Encoding::kMxfp4, Layout::kGgmlBlocks, kK, kN), {&blocks, 1}, 0, 0, kN);
  const std::vector<ref::PartView> split = {{PartRole::kCodes, codes},
                                            {PartRole::kBlockScales, scales}};
  const std::vector<float> b =
      Decode(Matrix(Encoding::kMxfp4, Layout::kSplit, kK, kN), split, 0, 0, kN);
  std::vector<float> theirs(kN * kK);
  ggml_get_type_traits(GGML_TYPE_MXFP4)
      ->to_float(ggml.data(), theirs.data(), static_cast<std::int64_t>(kN * kK));
  for (std::size_t i = 0; i < a.size(); ++i) {
    ASSERT_TRUE(SameBits(a[i], b[i])) << i;
    ASSERT_TRUE(SameValue(a[i], theirs[i])) << i << ": " << a[i] << " " << theirs[i];
  }
}

// MXFP8 (as qwen38_ops_test's Mxfp8DequantizationIsExact): each value is
// exactly E4M3(code) · 2^(scale - 127), split or in scale atoms; NaN codes
// and scales are NaN.
TEST(ReferenceDecodeTest, Mxfp8IsExact) {
  constexpr std::uint64_t kK = 256;
  constexpr std::uint64_t kN = 24;
  std::mt19937_64 random(5);  // NOLINT(bugprone-random-generator-seed): reproducible
  std::vector<std::byte> codes(kK * kN);
  for (std::byte& c : codes) {
    c = static_cast<std::byte>(random() & 0xFF);
  }
  std::vector<std::byte> scales(kK / 32 * kN);
  for (std::size_t i = 0; i < scales.size(); ++i) {
    scales[i] = static_cast<std::byte>(110 + (i % 30));
  }
  scales[3] = std::byte{0xFF};
  std::vector<std::byte> atoms(128 * (kK / 32));
  for (std::uint64_t r = 0; r < kN; ++r) {
    for (std::uint64_t b = 0; b < kK / 32; ++b) {
      atoms[format::ScaleAtomOffset(r, b, kK / 32)] = scales[(r * (kK / 32)) + b];
    }
  }
  const std::vector<ref::PartView> split = {{PartRole::kCodes, codes},
                                            {PartRole::kBlockScales, scales}};
  const std::vector<ref::PartView> atom = {{PartRole::kCodes, codes},
                                           {PartRole::kBlockScales, atoms}};
  const std::vector<float> a =
      Decode(Matrix(Encoding::kMxfp8, Layout::kSplit, kK, kN), split, 0, 0, kN);
  const std::vector<float> b =
      Decode(Matrix(Encoding::kMxfp8, Layout::kCutlassScaleAtoms, kK, kN), atom, 0, 0, kN);
  for (std::size_t i = 0; i < a.size(); ++i) {
    const auto code = std::to_integer<std::uint8_t>(codes[i]);
    const auto s = std::to_integer<std::uint8_t>(scales[i / 32]);
    ASSERT_TRUE(SameBits(a[i], b[i])) << i;
    if (s == 0xFF || (code & 0x7F) == 0x7F) {
      EXPECT_TRUE(std::isnan(a[i])) << i;
      continue;
    }
    ASSERT_EQ(static_cast<double>(a[i]), E4m3(code) * std::ldexp(1.0, s - 127)) << i;
  }
}

// A tensor scale per stacked matrix: gate's rows take gate's, up's up's,
// in each of the tensor's matrices.
TEST(ReferenceDecodeTest, TensorScalesFollowTheStackedMatrices) {
  constexpr std::uint64_t kK = 32;
  constexpr std::uint64_t kN = 8;  // 4 gate rows, 4 up rows
  TensorFormat f = Matrix(Encoding::kNvfp4, Layout::kSplit, kK, kN, 2);
  f.stacked = 2;
  f.params.tensor_scale = true;
  const std::vector<std::byte> codes(2 * kN * kK / 2, std::byte{0x22});    // 1.0 everywhere
  const std::vector<std::byte> scales(2 * kN * kK / 16, std::byte{0x38});  // E4M3 1.0
  const std::array<float, 4> ts = {2.0f, 3.0f, 5.0f, 7.0f};
  const std::vector<ref::PartView> parts = {{PartRole::kCodes, codes},
                                            {PartRole::kBlockScales, scales},
                                            {PartRole::kTensorScale, std::as_bytes(std::span(ts))}};
  const std::vector<float> m1 = Decode(f, parts, 1, 0, kN);
  for (std::uint64_t r = 0; r < kN; ++r) {
    EXPECT_EQ(m1[r * kK], r < 4 ? 5.0f : 7.0f) << r;
  }
  EXPECT_EQ(Decode(f, parts, 0, 3, 2)[kK], 3.0f);
}

// ---------------------------------------------------------------- EXL3 trellis

// An EXL3 trellis's format and random parts: words, and F16 side vectors
// of mixed signs and magnitudes.
struct Trellis {
  TensorFormat format;
  std::vector<std::byte> codes, suh, svh, marker;
  std::vector<ref::PartView> parts() const {
    return {{PartRole::kCodes, codes},
            {PartRole::kInputSigns, suh},
            {PartRole::kOutputSigns, svh},
            {PartRole::kCodebookMarker, marker}};
  }
};

std::vector<std::byte> HalfBytes(std::uint64_t seed, std::size_t n) {
  std::mt19937_64 random(seed);
  std::uniform_real_distribution<float> magnitude(0.25f, 2.0f);
  std::vector<std::byte> out(2 * n);
  for (std::size_t i = 0; i < n; ++i) {
    const float v = ((random() & 1U) != 0 ? -1.0f : 1.0f) * magnitude(random);
    const std::uint16_t h = ref::F32ToF16(v);
    std::memcpy(out.data() + (2 * i), &h, 2);
  }
  return out;
}

Trellis RandomTrellis(std::uint64_t seed, std::uint64_t k, std::uint64_t n, std::uint8_t bits,
                      format::Exl3Codebook codebook, std::uint64_t matrices = 1) {
  Trellis t;
  t.format = Matrix(Encoding::kExl3Trellis, Layout::kExl3Tiles, k, n, matrices);
  t.format.params.k_bits = bits;
  t.format.params.codebook = codebook;
  const auto parts = format::PartsOf(t.format);
  EXPECT_TRUE(parts.has_value());
  t.codes = RandomBytes(seed, (*parts)[0].bytes);
  t.suh = HalfBytes(seed + 1, k * matrices);
  t.svh = HalfBytes(seed + 2, n * matrices);
  t.marker.resize(4 * matrices);
  return t;
}

// The values codebook.cuh's decode_3inst gives, computed apart from the
// reference: in Python, from each codebook's integer steps, the two F16
// halves summed exactly (or mul1's h · 0x1EEE + 0xC931 exactly) and
// rounded once to F16 with Python's half-even packing (struct 'e').
TEST(ReferenceDecodeTest, Exl3CodebooksAreUpstreams) {
  struct Case {
    std::uint16_t state, three_inst, mcg, mul1;
  };
  constexpr std::array<Case, 6> kCases = {{{0x0000, 0x3A25, 0x3F60, 0xC2E8},
                                           {0x0001, 0xBB5B, 0x304E, 0x3921},
                                           {0x1234, 0xC066, 0x3E0F, 0xB89E},
                                           {0x8000, 0xB7F8, 0xB915, 0xBAF2},
                                           {0xBEEF, 0x424B, 0xC293, 0x3124},
                                           {0xFFFF, 0xB110, 0x3ACD, 0xB936}}};
  for (const Case& c : kCases) {
    EXPECT_EQ(ref::Exl3CodebookValue(format::Exl3Codebook::k3Inst, c.state), c.three_inst)
        << c.state;
    EXPECT_EQ(ref::Exl3CodebookValue(format::Exl3Codebook::kMcg, c.state), c.mcg) << c.state;
    EXPECT_EQ(ref::Exl3CodebookValue(format::Exl3Codebook::kMul1, c.state), c.mul1) << c.state;
  }
  EXPECT_EQ(ref::Exl3CodebookValue(format::Exl3Codebook::kNone, 0), 0x7E00);
}

// A tile's 256 states and where they land, derived apart from the
// reference's simulation of the kernels: state t is the 16 bits of the
// tile's circular bit stream (each uint32 most significant bit first)
// ending at bit (t + 1) · K; position 8L + i (L a lane, i < 8) of a lane
// without bit 2 lands at row (L % 4) · 2 + {0, 1, 8, 9}[i % 4] and column
// 2 · (L / 8 + 4 · (i / 4)), and lane L + 4's same position one column
// right. Every rate, tiles of two tile rows and columns, column ranges.
TEST(ReferenceDecodeTest, Exl3TilesAreReconstructTilesLayout) {
  constexpr std::uint64_t kK = 32;
  constexpr std::uint64_t kN = 48;
  for (std::uint8_t bits = 1; bits <= 8; ++bits) {
    SCOPED_TRACE(static_cast<int>(bits));
    const Trellis t = RandomTrellis(100 + bits, kK, kN, bits, format::Exl3Codebook::kMcg);
    std::vector<std::uint16_t> want(kK * kN);
    const std::uint64_t tile_bytes = std::uint64_t{32} * bits;
    for (std::uint64_t kt = 0; kt < kK / 16; ++kt) {
      for (std::uint64_t nt = 0; nt < kN / 16; ++nt) {
        const std::span<const std::byte> tile =
            std::span(t.codes).subspan(((kt * (kN / 16)) + nt) * tile_bytes, tile_bytes);
        const std::uint64_t length = 256 * bits;
        auto bit = [&](std::uint64_t j) {
          std::uint32_t word = 0;
          std::memcpy(&word, tile.data() + (4 * (j / 32)), 4);
          return (word >> (31 - (j % 32))) & 1U;
        };
        for (std::uint64_t p = 0; p < 256; ++p) {
          std::uint32_t state = 0;
          const std::uint64_t end = (p + 1) * bits;
          for (std::uint64_t i = 0; i < 16; ++i) {
            state = (state << 1U) | bit((end + length - 16 + i) % length);
          }
          const std::uint64_t group = p / 8;
          const bool partner = (group & 4) != 0;
          const std::uint64_t lane = partner ? group - 4 : group;
          const std::uint64_t i = p % 8;
          constexpr std::array<std::uint64_t, 4> kRow = {0, 1, 8, 9};
          const std::uint64_t row = ((lane % 4) * 2) + kRow.at(i % 4);
          const std::uint64_t column = (2 * ((lane / 8) + (4 * (i / 4)))) + (partner ? 1 : 0);
          want[(((kt * 16) + row) * kN) + (nt * 16) + column] =
              ref::Exl3CodebookValue(format::Exl3Codebook::kMcg, static_cast<std::uint16_t>(state));
        }
      }
    }
    std::vector<std::uint16_t> got(kK * kN);
    ASSERT_TRUE(ref::DecodeExl3Rotated(t.format, t.parts(), 0, 0, kN, got).has_value());
    ASSERT_EQ(got, want);
    // A column range: columns [16, 48) of every row.
    std::vector<std::uint16_t> part(kK * 32);
    ASSERT_TRUE(ref::DecodeExl3Rotated(t.format, t.parts(), 0, 16, 32, part).has_value());
    for (std::uint64_t r = 0; r < kK; ++r) {
      for (std::uint64_t c = 0; c < 32; ++c) {
        ASSERT_EQ(part[(r * 32) + c], want[(r * kN) + 16 + c]) << r << " " << c;
      }
    }
  }
}

// W = diag(suh) · H · Ŵ · H · diag(svh) / 128, computed here by explicit
// products with the Hadamard matrices (not the reference's fast
// transform), in F64 and rounded once to F32: equal for the 3inst and mcg
// codebooks, whose sums are exact, and within F64 rounding for mul1; at
// every rate, any row range, and the second of two matrices.
TEST(ReferenceDecodeTest, Exl3RowsAreTheTransformedRotatedWeights) {
  constexpr std::uint64_t kK = 256;
  constexpr std::uint64_t kN = 384;
  auto h = [](std::uint64_t i, std::uint64_t j) {
    return std::popcount((i % 128) & (j % 128)) % 2 == 0 ? 1.0 : -1.0;
  };
  struct Case {
    std::uint8_t bits;
    format::Exl3Codebook codebook;
  };
  std::vector<Case> cases;
  for (std::uint8_t bits = 1; bits <= 8; ++bits) {
    cases.push_back({bits, format::Exl3Codebook::kMcg});
  }
  cases.push_back({3, format::Exl3Codebook::k3Inst});
  cases.push_back({4, format::Exl3Codebook::kMul1});
  for (const Case& c : cases) {
    SCOPED_TRACE(std::format("K {} codebook {}", c.bits, static_cast<int>(c.codebook)));
    const Trellis t = RandomTrellis(200 + c.bits, kK, kN, c.bits, c.codebook, 2);
    std::vector<std::uint16_t> rotated(kK * kN);
    ASSERT_TRUE(ref::DecodeExl3Rotated(t.format, t.parts(), 1, 0, kN, rotated).has_value());
    // A = H · Ŵ, then W = A · H, block by block.
    std::vector<double> a(kK * kN);
    for (std::uint64_t k = 0; k < kK; ++k) {
      for (std::uint64_t n = 0; n < kN; ++n) {
        double sum = 0;
        for (std::uint64_t j = k / 128 * 128; j < (k / 128 * 128) + 128; ++j) {
          sum += h(k, j) * ref::F16ToF32(rotated[(j * kN) + n]);
        }
        a[(k * kN) + n] = sum;
      }
    }
    std::vector<float> want(kN * kK);  // rows of W: output n, then input k
    for (std::uint64_t k = 0; k < kK; ++k) {
      std::uint16_t su = 0;
      std::memcpy(&su, t.suh.data() + (2 * (kK + k)), 2);
      for (std::uint64_t n = 0; n < kN; ++n) {
        double sum = 0;
        for (std::uint64_t j = n / 128 * 128; j < (n / 128 * 128) + 128; ++j) {
          sum += a[(k * kN) + j] * h(j, n);
        }
        std::uint16_t sv = 0;
        std::memcpy(&sv, t.svh.data() + (2 * (kN + n)), 2);
        want[(n * kK) + k] =
            static_cast<float>(sum * ref::F16ToF32(su) * ref::F16ToF32(sv) / 128.0);
      }
    }
    const std::vector<float> got = Decode(t.format, t.parts(), 1, 0, kN);
    const std::vector<float> some = Decode(t.format, t.parts(), 1, 100, 200);
    for (std::size_t i = 0; i < got.size(); ++i) {
      if (c.codebook == format::Exl3Codebook::kMul1) {
        ASSERT_NEAR(got[i], want[i], 1e-6 * std::max(1.0f, std::abs(want[i]))) << i;
      } else {
        ASSERT_TRUE(SameBits(got[i], want[i])) << i << ": " << got[i] << " " << want[i];
      }
      if (i >= 100 * kK && i < 300 * kK) {
        ASSERT_TRUE(SameBits(some[i - (100 * kK)], got[i])) << i;
      }
    }
  }
}

TEST(ReferenceDecodeTest, Exl3RefusesWhatUpstreamDoesNotTake) {
  const Trellis t = RandomTrellis(7, 128, 256, 4, format::Exl3Codebook::kMcg);
  std::vector<float> row(128);
  // Missing or short side vectors.
  std::vector<ref::PartView> parts = t.parts();
  parts.erase(parts.begin() + 1);
  EXPECT_EQ(ref::DecodeRows(t.format, parts, 0, 0, 1, row).error(), ref::DecodeError::kParts);
  parts = t.parts();
  parts[2].bytes = parts[2].bytes.first(10);
  EXPECT_EQ(ref::DecodeRows(t.format, parts, 0, 0, 1, row).error(), ref::DecodeError::kParts);
  // K or N off the Hadamard's 128 (whole tiles, so a valid format).
  const Trellis narrow = RandomTrellis(8, 128, 144, 4, format::Exl3Codebook::kMcg);
  ASSERT_TRUE(format::Validate(narrow.format).has_value());
  EXPECT_EQ(ref::DecodeRows(narrow.format, narrow.parts(), 0, 0, 1, row).error(),
            ref::DecodeError::kUnsupported);
  std::vector<std::uint16_t> rotated(128 * 144);
  EXPECT_TRUE(
      ref::DecodeExl3Rotated(narrow.format, narrow.parts(), 0, 0, 144, rotated).has_value());
  // Rotated columns off the tiles, past the matrix, or the wrong size.
  std::vector<std::uint16_t> some(128 * 16);
  EXPECT_EQ(ref::DecodeExl3Rotated(t.format, t.parts(), 0, 8, 16, some).error(),
            ref::DecodeError::kRange);
  EXPECT_EQ(ref::DecodeExl3Rotated(t.format, t.parts(), 0, 256, 16, some).error(),
            ref::DecodeError::kRange);
  EXPECT_EQ(ref::DecodeExl3Rotated(t.format, t.parts(), 1, 0, 16, some).error(),
            ref::DecodeError::kRange);
  EXPECT_EQ(ref::DecodeExl3Rotated(t.format, t.parts(), 0, 0, 32, some).error(),
            ref::DecodeError::kRange);
  EXPECT_EQ(ref::DecodeExl3Rotated(Matrix(Encoding::kF16, Layout::kDense, 128, 16), t.parts(), 0, 0,
                                   16, some)
                .error(),
            ref::DecodeError::kUnsupported);
  // No codebook, or a rate past 8: not a format.
  TensorFormat none = t.format;
  none.params.codebook = format::Exl3Codebook::kNone;
  EXPECT_EQ(ref::DecodeRows(none, t.parts(), 0, 0, 1, row).error(), ref::DecodeError::kFormat);
  TensorFormat nine = t.format;
  nine.params.k_bits = 9;
  EXPECT_EQ(ref::DecodeRows(nine, t.parts(), 0, 0, 1, row).error(), ref::DecodeError::kFormat);
  // Nothing encodes a trellis.
  std::vector<std::byte> data(1 << 16);
  EXPECT_EQ(ref::EncodeRows(t.format, 0, 0, row, data).error(), ref::DecodeError::kUnsupported);
}

// ---------------------------------------------------------------- state encodings

// Every state encoding encodes and decodes on the CPU (what a reference
// cache store writes and a reference attention reads): F32 exactly, F16
// and BF16 rounded to nearest even, GGML's blocks within their step (Q8_0
// within amax / 128 of each block, Q5_x amax / 16, Q4_x amax / 8, the F16
// scale's rounding included); and a lossy one is never Lossless from F16.
TEST(ReferenceDecodeTest, StateEncodingsEncodeAndDecode) {
  constexpr std::uint64_t kK = 256;
  constexpr std::uint64_t kRows = 6;
  const std::vector<float> rows = Normal(31, kK * kRows);
  for (const Encoding e : format::StateEncodings()) {
    const format::EncodingInfo& info = format::Info(e);
    SCOPED_TRACE(std::string(info.name));
    const TensorFormat f =
        Matrix(e, info.quantized ? Layout::kGgmlBlocks : Layout::kDense, kK, kRows);
    std::vector<std::byte> data(format::PartsOf(f).value()[0].bytes);
    ASSERT_TRUE(ref::EncodeRows(f, 0, 0, rows, data).has_value());
    const ref::PartView view{.role = PartRole::kData, .bytes = data};
    const std::vector<float> back = Decode(f, {&view, 1}, 0, 0, kRows);
    if (!info.quantized) {
      for (std::size_t i = 0; i < rows.size(); ++i) {
        const float want = e == Encoding::kF32   ? rows[i]
                           : e == Encoding::kF16 ? ref::F16ToF32(ref::F32ToF16(rows[i]))
                                                 : ref::Bf16ToF32(ref::F32ToBf16(rows[i]));
        ASSERT_TRUE(SameBits(back[i], want)) << i;
      }
      continue;
    }
    EXPECT_FALSE(format::Lossless(Encoding::kF16, e));
    const double step = e == Encoding::kQ8_0                           ? 1.0 / 128
                        : e == Encoding::kQ5_0 || e == Encoding::kQ5_1 ? 1.0 / 16
                                                                       : 1.0 / 8;
    const std::uint64_t block = info.block_elements;
    for (std::size_t b = 0; b < rows.size() / block; ++b) {
      double amax = 0;
      for (std::size_t i = b * block; i < (b + 1) * block; ++i) {
        amax = std::max(amax, std::abs(static_cast<double>(rows[i])));
      }
      for (std::size_t i = b * block; i < (b + 1) * block; ++i) {
        ASSERT_LE(std::abs(static_cast<double>(back[i]) - rows[i]), 1.01 * step * amax) << i;
      }
    }
  }
}

TEST(ReferenceDecodeTest, RefusesPartsAndRangesThatDoNotFit) {
  const TensorFormat f = Matrix(Encoding::kMxfp8, Layout::kSplit, 64, 4);
  const std::vector<std::byte> codes(256);
  const std::vector<std::byte> scales(8);
  const std::vector<std::byte> short_scales(7);
  std::vector<float> out(64);
  auto refused = [&](std::vector<ref::PartView> parts, std::uint64_t matrix, std::uint64_t first,
                     std::span<float> o) {
    const auto r = ref::DecodeRows(f, parts, matrix, first, o.size() / 64, o);
    return r ? ref::DecodeError::kUnsupported : r.error();
  };
  const ref::PartView c{PartRole::kCodes, codes};
  const ref::PartView s{PartRole::kBlockScales, scales};
  EXPECT_TRUE(ref::DecodeRows(f, std::vector{c, s}, 0, 3, 1, out).has_value());
  EXPECT_EQ(refused({c}, 0, 0, out), ref::DecodeError::kParts);
  EXPECT_EQ(refused({c, c}, 0, 0, out), ref::DecodeError::kParts);
  EXPECT_EQ(refused({c, {PartRole::kBlockScales, short_scales}}, 0, 0, out),
            ref::DecodeError::kParts);
  EXPECT_EQ(refused({c, s, {PartRole::kTensorScale, scales}}, 0, 0, out), ref::DecodeError::kParts);
  EXPECT_EQ(refused({c, s}, 1, 0, out), ref::DecodeError::kRange);
  EXPECT_EQ(refused({c, s}, 0, 4, out), ref::DecodeError::kRange);
  std::vector<float> two(128);
  EXPECT_EQ(refused({c, s}, 0, 3, two), ref::DecodeError::kRange);
  EXPECT_EQ(ref::DecodeRows(f, std::vector{c, s}, 0, 0, 2, out).error(), ref::DecodeError::kRange);
  TensorFormat bad = f;
  bad.shape.ne[0] = 48;
  EXPECT_EQ(ref::DecodeRows(bad, std::vector{c, s}, 0, 0, 1, out).error(),
            ref::DecodeError::kFormat);
  TensorFormat trellis = Matrix(Encoding::kExl3Trellis, Layout::kExl3Tiles, 64, 16);
  trellis.params.k_bits = 4;
  trellis.params.codebook = format::Exl3Codebook::kMcg;
  EXPECT_EQ(ref::DecodeRows(trellis, std::vector{c, s}, 0, 0, 1, out).error(),
            ref::DecodeError::kUnsupported);
}

// ---------------------------------------------------------------- real artifacts

namespace fs = std::filesystem;
namespace ar = llmp::artifact;

std::vector<std::byte> ReadPlaced(const fs::path& root, const ar::Artifact& a,
                                  const ar::Placement& p) {
  const ar::Group& group = a.groups()[p.group];
  const ar::Shard& shard = a.shards()[group.shard];
  std::ifstream in(root / shard.path, std::ios::binary);
  in.seekg(static_cast<std::streamoff>(shard.data_offset.value() + group.offset.value() +
                                       p.offset.value()));
  std::vector<std::byte> out(p.bytes.value());
  in.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(out.size()));
  EXPECT_TRUE(in.good());
  return out;
}

struct Slice {
  TensorFormat format;
  std::vector<std::byte> bytes;
};

Slice ExpertSlice(const fs::path& root, const ar::Artifact& a, const std::string& name,
                  std::uint32_t expert) {
  const auto index = a.FindExpertArray(name);
  EXPECT_TRUE(index.has_value()) << name;
  if (!index) {
    return {};
  }
  const ar::ExpertArray& array = a.expert_arrays()[*index];
  const auto placed = a.SlicePlacement(*index, expert);
  EXPECT_TRUE(placed.has_value()) << name;
  const auto f = ar::FormatOf(array.repr);
  EXPECT_TRUE(f.has_value()) << name;
  return {.format = f.value_or(TensorFormat{}), .bytes = ReadPlaced(root, a, *placed)};
}

// The first Qwen3.8 import holds the routed experts in GGML's block_nvfp4
// (gate, up and down apiece), the later one in the CUTLASS layout (gate
// and up stacked, codes then atoms); both from one checkpoint, losslessly
// (docs/artifact-format.md). Each expert decodes to the same values in
// both.
TEST(ReferenceDecodeArtifactTest, Qwen38ExpertsDecodeAlikeInBothLayouts) {
  const char* home = std::getenv("HOME");  // NOLINT(concurrency-mt-unsafe)
  const fs::path store = fs::path(home != nullptr ? home : "/") / ".local/share/llmp/m3-artifacts";
  const fs::path ggml_root =
      store / "67617f87f2199408faa6a3663ca86f6d7aafb258a687a8d0a11092ffd74741b6";
  const fs::path cutlass_root =
      store / "c4fb47a911207c11f935f932d05196dc1701aa0d886eac1b5e91934e554b5a93";
  std::error_code error;
  if (!fs::is_directory(ggml_root, error) || !fs::is_directory(cutlass_root, error)) {
    GTEST_SKIP() << "no Qwen3.8 imports in " << store << " (they are on the Sparks)";
  }
  const auto ggml_artifact = ar::Artifact::Open(ggml_root);
  const auto cutlass_artifact = ar::Artifact::Open(cutlass_root);
  ASSERT_TRUE(ggml_artifact.has_value());
  ASSERT_TRUE(cutlass_artifact.has_value());
  // The first and last layers with routed experts.
  std::vector<std::uint32_t> layers;
  for (const ar::ExpertArray& array : ggml_artifact->expert_arrays()) {
    if (array.name.ends_with(".ffn_gate_exps.weight")) {
      layers.push_back(array.layer);
    }
  }
  ASSERT_FALSE(layers.empty());
  std::ranges::sort(layers);
  for (const std::uint32_t layer : {layers.front(), layers.back()}) {
    for (const std::uint32_t expert : {0U, 1U, 511U}) {
      SCOPED_TRACE(std::to_string(layer) + "/" + std::to_string(expert));
      const std::string blk = "blk." + std::to_string(layer) + ".";
      const Slice gate =
          ExpertSlice(ggml_root, *ggml_artifact, blk + "ffn_gate_exps.weight", expert);
      const Slice up = ExpertSlice(ggml_root, *ggml_artifact, blk + "ffn_up_exps.weight", expert);
      const Slice down =
          ExpertSlice(ggml_root, *ggml_artifact, blk + "ffn_down_exps.weight", expert);
      ASSERT_EQ(gate.format.layout, Layout::kGgmlNvfp4Blocks);
      // The CUTLASS arrays, assembled into their logical tensors.
      const auto& c = *cutlass_artifact;
      auto repr = [&](const std::string& name) {
        return &c.expert_arrays()[c.FindExpertArray(name).value()].repr;
      };
      const std::vector<ar::V0Part> gu_parts = {
          {PartRole::kCodes, repr(blk + "ffn_gate_up_exps.codes")},
          {PartRole::kBlockScales, repr(blk + "ffn_gate_up_exps.scales")}};
      const auto gate_up = ar::Assemble(Encoding::kNvfp4, Layout::kCutlassScaleAtoms, gu_parts, 2);
      ASSERT_TRUE(gate_up.has_value());
      const std::vector<ar::V0Part> down_parts = {
          {PartRole::kCodes, repr(blk + "ffn_down_exps.codes")},
          {PartRole::kBlockScales, repr(blk + "ffn_down_exps.scales")}};
      const auto down_c = ar::Assemble(Encoding::kNvfp4, Layout::kCutlassScaleAtoms, down_parts);
      ASSERT_TRUE(down_c.has_value());
      const Slice gu_codes = ExpertSlice(cutlass_root, c, blk + "ffn_gate_up_exps.codes", expert);
      const Slice gu_scales = ExpertSlice(cutlass_root, c, blk + "ffn_gate_up_exps.scales", expert);
      const Slice d_codes = ExpertSlice(cutlass_root, c, blk + "ffn_down_exps.codes", expert);
      const Slice d_scales = ExpertSlice(cutlass_root, c, blk + "ffn_down_exps.scales", expert);

      const std::uint64_t f = gate.format.shape.ne[1];
      const ref::PartView g{PartRole::kData, gate.bytes};
      const ref::PartView u{PartRole::kData, up.bytes};
      const ref::PartView d{PartRole::kData, down.bytes};
      const std::vector<float> gate_values = Decode(gate.format, {&g, 1}, 0, 0, f);
      const std::vector<float> up_values = Decode(up.format, {&u, 1}, 0, 0, f);
      const std::vector<float> down_values =
          Decode(down.format, {&d, 1}, 0, 0, down.format.shape.ne[1]);
      const std::vector<ref::PartView> gu = {{PartRole::kCodes, gu_codes.bytes},
                                             {PartRole::kBlockScales, gu_scales.bytes}};
      const std::vector<ref::PartView> dc = {{PartRole::kCodes, d_codes.bytes},
                                             {PartRole::kBlockScales, d_scales.bytes}};
      const std::vector<float> gate_c = Decode(*gate_up, gu, 0, 0, f);
      const std::vector<float> up_c = Decode(*gate_up, gu, 0, f, f);
      const std::vector<float> down_cv = Decode(*down_c, dc, 0, 0, down_c->shape.ne[1]);
      ASSERT_EQ(gate_values.size(), gate_c.size());
      ASSERT_EQ(down_values.size(), down_cv.size());
      for (std::size_t i = 0; i < gate_values.size(); ++i) {
        ASSERT_TRUE(SameBits(gate_values[i], gate_c[i])) << "gate " << i;
        ASSERT_TRUE(SameBits(up_values[i], up_c[i])) << "up " << i;
      }
      for (std::size_t i = 0; i < down_values.size(); ++i) {
        ASSERT_TRUE(SameBits(down_values[i], down_cv[i])) << "down " << i;
      }
    }
  }
}

}  // namespace
