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
// the affine encoding against hand-computed values, MLX's bit order and
// its documented quantization; every state encoding's encode and decode;
// and refusals.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <format>
#include <initializer_list>
#include <limits>
#include <random>
#include <span>
#include <string>
#include <vector>

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

TEST(ReferenceDecode, ComponentStateRowReusesWholeBlockCodec) {
  for (const std::size_t width : {64U, 256U}) {
    std::vector<double> input(width), output(width, -999.0);
    std::vector<float> values(width), decoded(width);
    for (std::size_t i = 0; i < width; ++i)
      input[i] = values[i] = static_cast<float>(std::sin(static_cast<double>(i) * 0.73));
    const TensorFormat f{.encoding = Encoding::kQ8_0,
                         .layout = Layout::kGgmlBlocks,
                         .shape = {.ne = {width, 1, 1, 1}, .rank = 2}};
    std::vector<std::byte> encoded(width / 32 * 34);
    ASSERT_TRUE(ref::EncodeRows(f, 0, 0, values, encoded));
    const ref::PartView part{.role = PartRole::kData, .bytes = encoded};
    ASSERT_TRUE(ref::DecodeRows(f, std::span(&part, 1), 0, 0, 1, decoded));
    ASSERT_TRUE(ref::StoreStateRow(Encoding::kQ8_0, input, output));
    for (std::size_t i = 0; i < width; ++i) EXPECT_EQ(output[i], decoded[i]);
    std::ranges::fill(output, -999.0);
    EXPECT_FALSE(ref::StoreStateRow(Encoding::kQ4_0, input, output));
    EXPECT_FALSE(ref::StoreStateRow(Encoding::kQ8_0, std::span(input).first(width - 1), output));
    EXPECT_TRUE(std::ranges::all_of(output, [](double x) { return x == -999.0; }));
  }
}

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
      {Layout::kScaleAtoms,
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

// NVFP4's scales are UE4M3 and MX's E8M0 (format::ScaleCode): every layout
// decodes a finite scale byte to E4M3's (or 2^(x - 127)) and every other
// byte (UE4M3's 0x7F and sign-bit bytes, E8M0's 0xFF) to NaN, as the device
// decoders do (decode_gpu_test's EdgeCodesDecodeAsTheReference) and the
// importer refuses them (format::FirstUndefinedScale).
TEST(ReferenceDecodeTest, UndefinedBlockScalesAreNanInEveryLayout) {
  for (unsigned b = 0; b < 256; ++b) {
    const auto byte = static_cast<std::uint8_t>(b);
    EXPECT_EQ(std::isnan(ref::Ue4m3ToF32(byte)),
              !format::DefinedScale(format::ScaleCode::kUe4m3, byte))
        << b;
    EXPECT_EQ(std::isnan(ref::E8m0ToF32(byte)),
              !format::DefinedScale(format::ScaleCode::kE8m0, byte))
        << b;
    if (b < 0x7F) EXPECT_TRUE(SameBits(ref::Ue4m3ToF32(byte), ref::E4m3ToF32(byte))) << b;
  }
  constexpr std::array<std::uint8_t, 8> kScales = {0x00, 0x01, 0x38, 0x7E, 0x7F, 0x80, 0xB8, 0xFF};
  constexpr std::uint64_t kK = 128;
  constexpr std::uint64_t kN = 4;
  constexpr std::uint64_t kBlocks = kK / 16;
  const std::vector<std::byte> codes(kN * kK / 2, std::byte{0x22});  // 1.0 everywhere
  std::vector<std::byte> scales(kN * kBlocks);
  for (std::size_t i = 0; i < scales.size(); ++i)
    scales[i] = std::byte{kScales[i % kScales.size()]};
  std::vector<std::byte> interleaved;
  std::vector<std::byte> atoms(128 * kBlocks);
  std::vector<std::byte> ggml(kN * (kK / 64) * 36);
  for (std::uint64_t r = 0; r < kN; ++r) {
    interleaved.insert(interleaved.end(), codes.begin() + static_cast<std::ptrdiff_t>(r * kK / 2),
                       codes.begin() + static_cast<std::ptrdiff_t>((r + 1) * kK / 2));
    interleaved.insert(interleaved.end(), scales.begin() + static_cast<std::ptrdiff_t>(r * kBlocks),
                       scales.begin() + static_cast<std::ptrdiff_t>((r + 1) * kBlocks));
    for (std::uint64_t b = 0; b < kBlocks; ++b)
      atoms[format::ScaleAtomOffset(r, b, kBlocks)] = scales[(r * kBlocks) + b];
    for (std::uint64_t g = 0; g < kK / 64; ++g) {
      std::byte* block = ggml.data() + (((r * (kK / 64)) + g) * 36);
      for (std::uint64_t s = 0; s < 4; ++s) block[s] = scales[(r * kBlocks) + (g * 4) + s];
      std::fill_n(block + 4, 32, std::byte{0x22});
    }
  }
  struct Case {
    Layout layout;
    std::vector<ref::PartView> parts;
  };
  const std::vector<Case> cases = {
      {Layout::kSplit, {{PartRole::kCodes, codes}, {PartRole::kBlockScales, scales}}},
      {Layout::kRowInterleaved, {{PartRole::kData, interleaved}}},
      {Layout::kScaleAtoms, {{PartRole::kCodes, codes}, {PartRole::kBlockScales, atoms}}},
      {Layout::kGgmlNvfp4Blocks, {{PartRole::kData, ggml}}},
  };
  for (const Case& c : cases) {
    SCOPED_TRACE(std::string(format::Info(c.layout).name));
    const std::vector<float> got =
        Decode(Matrix(Encoding::kNvfp4, c.layout, kK, kN), c.parts, 0, 0, kN);
    for (std::uint64_t i = 0; i < got.size(); ++i) {
      const auto s = std::to_integer<std::uint8_t>(scales[i / 16]);
      if (s >= 0x7F) {
        ASSERT_TRUE(std::isnan(got[i])) << i << " scale " << unsigned{s};
      } else {
        ASSERT_EQ(static_cast<double>(got[i]), E4m3(s)) << i << " scale " << unsigned{s};
      }
    }
  }
  // E8M0: 0xFF is NaN in MXFP4's split layout and in GGML's block_mxfp4.
  const std::vector<std::byte> fp4(kK / 2, std::byte{0x22});
  const std::array<std::byte, 4> e8m0 = {std::byte{0x00}, std::byte{0x7F}, std::byte{0xFE},
                                         std::byte{0xFF}};
  std::vector<std::byte> blocks;
  for (std::uint64_t g = 0; g < kK / 32; ++g) {
    blocks.push_back(e8m0[g]);
    blocks.insert(blocks.end(), 16, std::byte{0x22});
  }
  const std::vector<ref::PartView> split = {{PartRole::kCodes, fp4},
                                            {PartRole::kBlockScales, e8m0}};
  const ref::PartView block_view{.role = PartRole::kData, .bytes = blocks};
  for (const auto& got :
       {Decode(Matrix(Encoding::kMxfp4, Layout::kSplit, kK, 1), split, 0, 0, 1),
        Decode(Matrix(Encoding::kMxfp4, Layout::kGgmlBlocks, kK, 1), {&block_view, 1}, 0, 0, 1)}) {
    for (std::uint64_t i = 0; i < kK; ++i) {
      EXPECT_EQ(std::isnan(got[i]), i / 32 == 3) << i;
    }
  }
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

// MXFP8: each value is
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
      Decode(Matrix(Encoding::kMxfp8, Layout::kScaleAtoms, kK, kN), atom, 0, 0, kN);
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

TEST(ReferenceDecodeTest, Fp8ScaledUsesF32GridAcrossStacksMatricesAndSlices) {
  TensorFormat f = Matrix(Encoding::kFp8Scaled, Layout::kSplit, 5, 10, 2);
  f.stacked = 2;
  f.params.block_rows = 3;
  f.params.block_cols = 2;
  std::vector<std::byte> codes(5 * 10 * 2);
  std::vector<std::byte> scales(3 * 4 * 4 * 2);
  for (std::size_t i = 0; i < codes.size(); ++i)
    codes[i] = static_cast<std::byte>((i % 2 == 0) ? 0x38 : 0x40);  // 1 and 2 in E4M3FN
  for (std::uint64_t m = 0; m < 2; ++m)
    for (std::uint64_t row = 0; row < 4; ++row)
      for (std::uint64_t col = 0; col < 3; ++col) {
        const float scale = static_cast<float>(2 + m * 100 + row * 10 + col);
        std::memcpy(scales.data() + (m * 12 + row * 3 + col) * 4, &scale, 4);
      }
  const std::array parts = {ref::PartView{PartRole::kCodes, codes},
                            ref::PartView{PartRole::kBlockScales, scales}};
  std::vector<float> decoded(6 * 5);
  ASSERT_TRUE(ref::DecodeRows(f, parts, 1, 2, 6, decoded));
  for (std::uint64_t r = 2; r < 8; ++r) {
    const std::uint64_t projection = r / 5;
    const std::uint64_t local_row = r % 5;
    const std::uint64_t scale_row = projection * 2 + local_row / 3;
    for (std::uint64_t col = 0; col < 5; ++col) {
      const std::uint64_t source = (1 * 10 + r) * 5 + col;
      const float code = (source % 2 == 0) ? 1.0f : 2.0f;
      const float scale = static_cast<float>(2 + 100 + scale_row * 10 + col / 2);
      EXPECT_EQ(decoded[(r - 2) * 5 + col], code * scale) << r << "," << col;
    }
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
  // A 3inst trellis has no marker (upstream's modules name mcg and mul1 by
  // their marker tensors).
  std::vector<ref::PartView> parts() const {
    std::vector<ref::PartView> out = {
        {PartRole::kCodes, codes}, {PartRole::kInputSigns, suh}, {PartRole::kOutputSigns, svh}};
    if (format.params.codebook != format::Exl3Codebook::k3Inst) {
      out.push_back({PartRole::kCodebookMarker, marker});
    }
    return out;
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

// ---------------------------------------------------------------- affine

TensorFormat Affine(std::uint64_t k, std::uint64_t n, std::uint8_t bits, std::uint32_t group,
                    format::AffineScale scale, bool zeros = false, std::uint64_t matrices = 1) {
  TensorFormat f = Matrix(Encoding::kAffine, Layout::kSplit, k, n, matrices);
  f.params.bits = bits;
  f.params.group = group;
  f.params.scale = scale;
  f.params.zero_point = zeros;
  return f;
}

std::vector<std::byte> Bf16s(std::initializer_list<float> values) {
  std::vector<std::byte> out;
  for (const float v : values) {
    const std::uint16_t b = ref::F32ToBf16(v);
    out.push_back(static_cast<std::byte>(b & 0xFFU));
    out.push_back(static_cast<std::byte>(b >> 8U));
  }
  return out;
}

// Codes as one little-endian bit stream, code i at bit i · bits.
std::vector<std::byte> Pack(const std::vector<std::uint32_t>& codes, std::uint32_t bits) {
  std::vector<std::byte> out(((codes.size() * bits) + 7) / 8);
  for (std::size_t i = 0; i < codes.size(); ++i) {
    for (std::uint32_t b = 0; b < bits; ++b) {
      if (((codes[i] >> b) & 1U) != 0) {
        const std::size_t bit = (i * bits) + b;
        out[bit / 8] |= static_cast<std::byte>(1U << (bit % 8));
      }
    }
  }
  return out;
}

// A row's codes, a scale of 1 and a bias of 0 a group: the codes' values.
std::vector<float> Codes(std::span<const std::byte> codes, std::uint64_t k, std::uint8_t bits,
                         std::uint32_t group) {
  std::vector<float> ones(k / group, 1.0f);
  std::vector<float> zeros(k / group, 0.0f);
  const std::vector<ref::PartView> parts = {
      {PartRole::kCodes, codes},
      {PartRole::kBlockScales, std::as_bytes(std::span(ones))},
      {PartRole::kBiases, std::as_bytes(std::span(zeros))}};
  return Decode(Affine(k, 1, bits, group, format::AffineScale::kF32), parts, 0, 0, 1);
}

// Hand-computed values: MLX's s · q + b with BF16 scales and biases, in
// the bit order mx.quantize documents ("packed in an unsigned 32-bit
// integer from the lower to upper bits"), at any row of any matrix; and
// AWQ's and GPTQ's s · (q − z).
TEST(ReferenceDecodeTest, AffineIsScaleTimesCodePlusBias) {
  // Row 0: codes 0 to 15 (bytes 0x10, 0x32, ... 0xFE); row 1: all 15.
  const TensorFormat f = Affine(16, 2, 4, 8, format::AffineScale::kBF16);
  std::vector<std::byte> codes = {std::byte{0x10}, std::byte{0x32}, std::byte{0x54},
                                  std::byte{0x76}, std::byte{0x98}, std::byte{0xBA},
                                  std::byte{0xDC}, std::byte{0xFE}};
  codes.resize(16, std::byte{0xFF});
  const auto scales = Bf16s({0.5f, -2.0f, 1.0f, 1.0f});
  const auto biases = Bf16s({1.0f, 0.25f, -7.5f, 0.0f});
  const std::vector<ref::PartView> parts = {
      {PartRole::kCodes, codes}, {PartRole::kBlockScales, scales}, {PartRole::kBiases, biases}};
  const std::vector<float> rows = Decode(f, parts, 0, 0, 2);
  for (std::uint64_t j = 0; j < 16; ++j) {
    const auto q = static_cast<float>(j);
    EXPECT_EQ(rows[j], j < 8 ? (0.5f * q) + 1.0f : (-2.0f * q) + 0.25f) << j;
    EXPECT_EQ(rows[16 + j], j < 8 ? 7.5f : 15.0f) << j;
  }
  EXPECT_EQ(Decode(f, parts, 0, 1, 1)[3], 7.5f);
  // One rounding: 3 · (1 + 2^-7) + 2^-9 is exact in F32 (BF16 scales keep
  // the product exact), and 3 · (1 + 2^-7) + 2^-30 rounds once to the
  // product.
  const auto odd_scales = Bf16s({1.0f + 0x1p-7f, 1.0f, 1.0f, 1.0f});
  const auto odd_biases = Bf16s({0x1p-9f, 0.0f, 0.0f, 0.0f});
  const std::vector<ref::PartView> odd = {{PartRole::kCodes, codes},
                                          {PartRole::kBlockScales, odd_scales},
                                          {PartRole::kBiases, odd_biases}};
  EXPECT_EQ(Decode(f, odd, 0, 0, 1)[3], 3.0f + (3.0f * 0x1p-7f) + 0x1p-9f);
  const auto tiny_biases = Bf16s({0x1p-30f, 0.0f, 0.0f, 0.0f});
  const std::vector<ref::PartView> tiny = {{PartRole::kCodes, codes},
                                           {PartRole::kBlockScales, odd_scales},
                                           {PartRole::kBiases, tiny_biases}};
  EXPECT_EQ(Decode(f, tiny, 0, 0, 1)[3], 3.0f + (3.0f * 0x1p-7f));

  // Zero points: F16 scales, z of 4 bits a group.
  const TensorFormat z = Affine(16, 1, 4, 8, format::AffineScale::kF16, true);
  const std::array<std::uint16_t, 2> half = {ref::F32ToF16(0.25f), ref::F32ToF16(-1.0f)};
  const std::vector<std::byte> zeros = {std::byte{0x38}};  // z = 8, then 3
  const std::vector<ref::PartView> zp = {{PartRole::kCodes, std::span(codes).first(8)},
                                         {PartRole::kBlockScales, std::as_bytes(std::span(half))},
                                         {PartRole::kZeros, zeros}};
  const std::vector<float> zr = Decode(z, zp, 0, 0, 1);
  for (std::uint64_t j = 0; j < 16; ++j) {
    const auto q = static_cast<float>(j);
    EXPECT_EQ(zr[j], j < 8 ? 0.25f * (q - 8.0f) : -1.0f * (q - 3.0f)) << j;
  }

  // The second matrix of an expert array.
  const TensorFormat two = Affine(16, 1, 4, 8, format::AffineScale::kBF16, false, 2);
  EXPECT_EQ(Decode(two, parts, 1, 0, 1)[0], 7.5f);
  // A part missing, short, or biases where zero points belong.
  std::vector<float> row(16);
  EXPECT_EQ(ref::DecodeRows(f, std::span(parts).first(2), 0, 0, 1, row).error(),
            ref::DecodeError::kParts);
  const std::vector<ref::PartView> shorter = {{PartRole::kCodes, codes},
                                              {PartRole::kBlockScales, scales},
                                              {PartRole::kBiases, std::span(biases).first(6)}};
  EXPECT_EQ(ref::DecodeRows(f, shorter, 0, 0, 1, row).error(), ref::DecodeError::kParts);
  EXPECT_EQ(ref::DecodeRows(z, parts, 0, 0, 1, row).error(), ref::DecodeError::kParts);
  // Not an encoding EncodeRows writes.
  std::vector<std::byte> out(64);
  EXPECT_EQ(ref::EncodeRows(f, 0, 0, row, out).error(), ref::DecodeError::kUnsupported);
}

// MLX's 3-, 5- and 6-bit packs (bytes, not words) are the same bit
// stream: its CPU unpacking (mlx/backend/cpu/quantized.cpp extract_bits)
// reads 3 bits' w[2] = ((w0 & 0xc0) >> 6) + ((w1 & 0x1) << 2), 5 bits'
// w[4] = ((w2 & 0xf0) >> 4) + ((w3 & 0x1) << 4), and 6 bits' w[1] = ((w0
// >> 6) & 0x03) + ((w1 & 0x0f) << 2). Codes 1, 2, ... 7, 0 at 3 bits are
// the bytes 0xD1, 0x58, 0x1F.
TEST(ReferenceDecodeTest, AffineCodesAreOneBitStreamAtEveryWidth) {
  const std::vector<std::byte> three = {std::byte{0xD1}, std::byte{0x58}, std::byte{0x1F}};
  EXPECT_EQ(Pack({1, 2, 3, 4, 5, 6, 7, 0}, 3), three);
  EXPECT_EQ(Codes(three, 8, 3, 8), (std::vector<float>{1, 2, 3, 4, 5, 6, 7, 0}));
  const auto w = RandomBytes(51, 6);
  const auto u8 = [&](std::size_t i) { return std::to_integer<unsigned>(w[i]); };
  EXPECT_EQ(Codes(std::span(w).first(3), 8, 3, 8)[2],
            static_cast<float>(((u8(0) & 0xC0U) >> 6U) + ((u8(1) & 0x1U) << 2U)));
  EXPECT_EQ(Codes(std::span(w).first(5), 8, 5, 8)[4],
            static_cast<float>(((u8(2) & 0xF0U) >> 4U) + ((u8(3) & 0x1U) << 4U)));
  EXPECT_EQ(Codes(w, 8, 6, 8)[1],
            static_cast<float>(((u8(0) >> 6U) & 0x03U) + ((u8(1) & 0x0FU) << 2U)));
  // Random codes at every width round-trip through the packing.
  std::mt19937_64 random(52);
  for (std::uint32_t bits = 1; bits <= 8; ++bits) {
    std::vector<std::uint32_t> q(64);
    for (std::uint32_t& c : q) {
      c = static_cast<std::uint32_t>(random() & ((1U << bits) - 1U));
    }
    const std::vector<float> got = Codes(Pack(q, bits), 64, static_cast<std::uint8_t>(bits), 32);
    for (std::size_t i = 0; i < q.size(); ++i) {
      ASSERT_EQ(got[i], static_cast<float>(q[i])) << bits << " " << i;
    }
  }
}

// MLX's documented affine quantization (mx.quantize: per group, s = (α −
// β) / (2^b − 1) with α and β its maximum and minimum, ŵ = round((w − β)
// / s)), written here with F32 scales and biases: the reference's s · ŵ +
// β is within s / 2 of every value (and an F32 rounding), at every width
// MLX's affine mode takes (2, 3, 4, 5, 6, 8) and its group sizes 32, 64
// and 128.
TEST(ReferenceDecodeTest, AffineDecodesMlxsDocumentedQuantization) {
  constexpr std::uint64_t kK = 256;
  constexpr std::uint64_t kN = 3;
  const std::vector<float> w = Normal(53, kK * kN);
  for (const std::uint8_t bits : std::initializer_list<std::uint8_t>{2, 3, 4, 5, 6, 8}) {
    for (const std::uint32_t group : {32U, 64U, 128U}) {
      SCOPED_TRACE(std::format("bits {} group {}", bits, group));
      const std::uint64_t groups = kK / group * kN;
      std::vector<std::uint32_t> q(kK * kN);
      std::vector<float> s(groups);
      std::vector<float> b(groups);
      for (std::uint64_t g = 0; g < groups; ++g) {
        const auto first = w.begin() + static_cast<std::ptrdiff_t>(g * group);
        const auto [lo, hi] = std::minmax_element(first, first + group);
        s[g] = (*hi - *lo) / static_cast<float>((1U << bits) - 1U);
        b[g] = *lo;
        for (std::uint64_t j = g * group; j < (g + 1) * group; ++j) {
          q[j] = static_cast<std::uint32_t>(std::lround((w[j] - b[g]) / s[g]));
        }
      }
      const auto codes = Pack(q, bits);
      const std::vector<ref::PartView> parts = {
          {PartRole::kCodes, codes},
          {PartRole::kBlockScales, std::as_bytes(std::span(s))},
          {PartRole::kBiases, std::as_bytes(std::span(b))}};
      const std::vector<float> got =
          Decode(Affine(kK, kN, bits, group, format::AffineScale::kF32), parts, 0, 0, kN);
      for (std::uint64_t i = 0; i < w.size(); ++i) {
        const float step = s[i / group];
        ASSERT_LE(std::abs(got[i] - w[i]), (step * 0.5f * (1.0f + 1e-5f)) + 1e-7f) << i;
        ASSERT_EQ(got[i], std::fma(step, static_cast<float>(q[i]), b[i / group])) << i;
      }
    }
  }
}

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

}  // namespace
