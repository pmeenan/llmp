// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cfenv>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string_view>
#include <vector>

#include "base/sha256.h"
#include "format/encoding.h"
#include "format/layout.h"
#include "format/tensor_format.h"
#include "import/transforms.h"

namespace llmp::import {
namespace {

using format::Encoding;
using format::Layout;
using format::PartRole;

std::array<std::byte, 2> Bf16(std::uint16_t bits) {
  return {static_cast<std::byte>(bits & 0xFFU), static_cast<std::byte>(bits >> 8U)};
}

std::uint32_t Bits(std::span<const std::byte> bytes) {
  std::uint32_t bits = 0;
  for (std::size_t i = 0; i < 4; ++i) {
    bits |= std::to_integer<std::uint32_t>(bytes[i]) << (i * 8);
  }
  return bits;
}

TEST(ImportTransformsTest, HistoricalScalarValuesAndSpecials) {
  struct Sample {
    std::uint16_t source;
    std::uint32_t norm;
    std::uint32_t a;
  };
  // Pinned historical converter, Python 3.14.7 on spark-b, 2026-10-11.
  constexpr std::array samples = {
      Sample{0x0000, 0x3F800000, 0xBF800000}, Sample{0x8000, 0x3F800000, 0xBF800000},
      Sample{0x3F00, 0x3FC00000, 0xBFD3094C}, Sample{0xBF80, 0x00000000, 0xBEBC5AB2},
      Sample{0x3F80, 0x40000000, 0xC02DF854}, Sample{0x42B1, 0x42B30000, 0xFF4CDCC4},
      Sample{0x42B2, 0x42B40000, 0xFF800000}, Sample{0x4431, 0x44314000, 0xFF800000},
      Sample{0x7F80, 0x7F800000, 0xFF800000}, Sample{0xFF80, 0xFF800000, 0x80000000},
      Sample{0x7F81, 0x7FC10000, 0xFFC10000}, Sample{0xFF81, 0xFFC10000, 0x7FC10000},
      Sample{0x7FC1, 0x7FC10000, 0xFFC10000}, Sample{0xFFC1, 0xFFC10000, 0x7FC10000},
  };
  for (const Sample& s : samples) {
    std::array<std::byte, 4> out{};
    ASSERT_TRUE(FoldBf16(ScalarFold::kNormPlusOne, Bf16(s.source), out).has_value());
    EXPECT_EQ(Bits(out), s.norm) << s.source;
    ASSERT_TRUE(FoldBf16(ScalarFold::kANegExp, Bf16(s.source), out).has_value());
    EXPECT_EQ(Bits(out), s.a) << s.source;
  }
}

TEST(ImportTransformsTest, EveryBf16WordMatchesHistoricalReference) {
  // Each independently generated historical table is 65,536 five-byte
  // records in BF16-word order: tag 0 + F32 little-endian bytes, or tag 1
  // + four zeros for Python OverflowError. These are the hashes of the
  // actual Python tables on the Spark's libm, not hashes computed by the
  // C++ implementation. Other libm implementations are not qualified by
  // this source alone; a mismatch requires investigation, not a new hash.
  // Converter modelopt_qwen38.py SHA-256:
  // 59457799c83abe75dd21fe76c310686a1d7b5fa6e2d94ee52122bb2efe2bffbb.
  constexpr std::array folds = {ScalarFold::kNormPlusOne, ScalarFold::kANegExp};
  constexpr std::array<std::string_view, 2> hashes = {
      "f487a87ca80e4753a5049b19c02ad5af3dfa395436dfe52b8f062f3996e45b28",
      "f0d4497f94e1cc6eaea0640f4a516fbefe6cd794bd612ca000f5d97b7ffc5822"};
  constexpr std::array<std::uint32_t, 2> refused = {0, 15182};
  for (std::size_t f = 0; f < folds.size(); ++f) {
    base::Sha256 hash;
    std::uint32_t failures = 0;
    std::uint32_t partial_writes = 0;
    for (std::uint32_t word = 0; word < 65536; ++word) {
      std::array<std::byte, 4> out;
      out.fill(std::byte{0x5A});
      const auto result = FoldBf16(folds[f], Bf16(static_cast<std::uint16_t>(word)), out);
      std::array<std::byte, 5> record{};
      if (result) {
        std::ranges::copy(out, record.begin() + 1);
      } else {
        ++failures;
        if (!std::ranges::all_of(out, [](std::byte b) { return b == std::byte{0x5A}; })) {
          ++partial_writes;
        }
        record[0] = std::byte{1};
      }
      hash.Update(record);
    }
    EXPECT_EQ(failures, refused[f]);
    EXPECT_EQ(partial_writes, 0U);
    EXPECT_EQ(base::ToHex(hash.Finish()), hashes[f]) << "fold " << f;
  }
}

TEST(ImportTransformsTest, NonzeroSubnormalsAndUnderflowKeepReferenceBits) {
  // Binary64 exp is normal for every nonzero F32 result here. Encoding
  // its F32 bits as integers avoids any hardware F32 flush-to-zero cast.
  // These fixed results straddle the normal/subnormal and zero boundaries.
  constexpr std::array<std::uint16_t, 10> words = {0xC2AE, 0xC2AF, 0xC2B0, 0xC2B1, 0xC2B2,
                                                   0xC2B4, 0xC2CE, 0xC2CF, 0xC2D0, 0xC2D1};
  constexpr std::array<std::uint32_t, 10> expected = {
      0x80B33687, 0x806CB2BC, 0x8041EDC4, 0x8027FCE2, 0x801840FC,
      0x8008EC28, 0x80000001, 0x80000001, 0x80000000, 0x80000000};
  for (std::size_t i = 0; i < words.size(); ++i) {
    std::array<std::byte, 4> out{};
    ASSERT_TRUE(FoldBf16(ScalarFold::kANegExp, Bf16(words[i]), out).has_value());
    EXPECT_EQ(Bits(out), expected[i]) << words[i];
  }
}

TEST(ImportTransformsTest, ScalarRefusalsDoNotWriteAnyDestinationByte) {
  std::array<std::byte, 8> out;
  out.fill(std::byte{0x35});
  const auto original = out;
  const std::array<std::byte, 4> later_overflow = {std::byte{0}, std::byte{0}, std::byte{0x32},
                                                   std::byte{0x44}};
  EXPECT_FALSE(FoldBf16(ScalarFold::kANegExp, later_overflow, out).has_value());
  EXPECT_EQ(out, original);
  EXPECT_FALSE(FoldBf16(static_cast<ScalarFold>(255), later_overflow, out).has_value());
  EXPECT_EQ(out, original);
  EXPECT_FALSE(
      FoldBf16(ScalarFold::kNormPlusOne, std::span(later_overflow).first(3), out).has_value());
  EXPECT_EQ(out, original);
  EXPECT_FALSE(
      FoldBf16(ScalarFold::kNormPlusOne, later_overflow, std::span(out).first(7)).has_value());
  EXPECT_EQ(out, original);
  EXPECT_FALSE(FoldBf16(ScalarFold::kNormPlusOne, std::span(out).first(4), out).has_value());
  EXPECT_EQ(out, original);
  const int old_rounding = std::fegetround();
  ASSERT_EQ(std::fesetround(FE_DOWNWARD), 0);
  const auto rounded = FoldBf16(ScalarFold::kNormPlusOne, later_overflow, out);
  const int restored = std::fesetround(old_rounding);
  ASSERT_EQ(restored, 0);
  EXPECT_FALSE(rounded.has_value());
  EXPECT_EQ(out, original);
  EXPECT_TRUE(FoldBf16(ScalarFold::kNormPlusOne, {}, {}).has_value());
}

TEST(ImportTransformsTest, ScalarChunksKeepTheirMeaningAndGuardBytes) {
  const std::array<std::byte, 8> input = {std::byte{0},    std::byte{0},   std::byte{0x80},
                                          std::byte{0xBF}, std::byte{0},   std::byte{0x3F},
                                          std::byte{0x80}, std::byte{0x3F}};
  for (const ScalarFold fold : {ScalarFold::kNormPlusOne, ScalarFold::kANegExp}) {
    std::array<std::byte, 16> together{};
    std::array<std::byte, 18> guarded;
    guarded.fill(std::byte{0x6B});
    ASSERT_TRUE(FoldBf16(fold, input, together).has_value());
    for (std::size_t i = 0; i < input.size(); i += 2) {
      ASSERT_TRUE(
          FoldBf16(fold, std::span(input).subspan(i, 2), std::span(guarded).subspan(1 + i * 2, 4))
              .has_value());
    }
    EXPECT_TRUE(std::ranges::equal(together, std::span(guarded).subspan(1, 16)));
    EXPECT_EQ(guarded.front(), std::byte{0x6B});
    EXPECT_EQ(guarded.back(), std::byte{0x6B});
  }
}

format::TensorFormat Matrix(Encoding encoding, std::uint64_t columns, std::uint64_t rows) {
  return {.encoding = encoding,
          .layout = encoding == Encoding::kF32 ? Layout::kDense : Layout::kSplit,
          .shape = {.ne = {columns, rows, 1, 1}, .rank = 2}};
}

// Tiny test-only byte coverage, independently constructing destination ->
// source indices. The production cursor keeps constant metadata, no bitmap.
void CheckPermutation(const HeadPermutationPlan& plan, HeadAxis axis, std::uint64_t keys,
                      std::uint64_t heads, std::uint64_t width, std::uint64_t first) {
  // Visit scales before codes to prove there is no global part-order dependency.
  for (std::size_t pi = plan.parts().size(); pi > 0; --pi) {
    const format::PartExtent part = plan.parts()[pi - 1];
    const auto row_bytes = format::RowBytes(plan.format(), part.role);
    ASSERT_TRUE(row_bytes.has_value());
    const auto n = static_cast<std::size_t>(part.bytes);
    ASSERT_LE(n, 65536U);
    std::vector<std::byte> source(n);
    for (std::size_t i = 0; i < n; ++i) {
      source[i] = static_cast<std::byte>((i * 7 + pi * 19) % 251);
    }
    std::vector<std::byte> expected(n);
    for (std::uint64_t dst = 0; dst < part.bytes; ++dst) {
      std::uint64_t src = dst;
      if (axis == HeadAxis::kRows) {
        const auto row = dst / *row_bytes;
        if (row >= first && row < first + heads * width) {
          const auto out_head = (row - first) / width;
          const auto in_head = (out_head % keys) * (heads / keys) + out_head / keys;
          src = (first + in_head * width + (row - first) % width) * *row_bytes + dst % *row_bytes;
        }
      } else {
        const auto head_bytes = *row_bytes / heads;
        const auto column = dst % *row_bytes;
        const auto out_head = column / head_bytes;
        const auto in_head = (out_head % keys) * (heads / keys) + out_head / keys;
        src = (dst / *row_bytes) * *row_bytes + in_head * head_bytes + column % head_bytes;
      }
      expected[static_cast<std::size_t>(dst)] = source[static_cast<std::size_t>(src)];
    }
    for (const std::uint64_t maximum : {std::uint64_t{1}, std::uint64_t{13}, std::uint64_t{37},
                                        std::numeric_limits<std::uint64_t>::max()}) {
      auto cursor = plan.CursorFor(part.role);
      ASSERT_TRUE(cursor.has_value());
      EXPECT_FALSE(cursor->Next(0).has_value());
      std::vector<std::byte> result(n);
      std::vector<bool> written(n, false);
      std::uint64_t consumed = 0;
      for (;;) {
        auto next = cursor->Next(maximum);
        ASSERT_TRUE(next.has_value());
        if (!*next) break;
        const CopyRun run = **next;
        ASSERT_EQ(run.part, part.role);
        ASSERT_EQ(run.source, consumed);
        ASSERT_GT(run.bytes, 0U);
        ASSERT_LE(run.bytes, maximum);
        ASSERT_LE(run.source, part.bytes);
        ASSERT_LE(run.bytes, part.bytes - run.source);
        ASSERT_LE(run.destination, part.bytes);
        ASSERT_LE(run.bytes, part.bytes - run.destination);
        for (std::uint64_t j = 0; j < run.bytes; ++j) {
          const auto dest = static_cast<std::size_t>(run.destination + j);
          ASSERT_FALSE(written[dest]);
          written[dest] = true;
          result[dest] = source[static_cast<std::size_t>(run.source + j)];
        }
        consumed += run.bytes;
      }
      EXPECT_EQ(consumed, part.bytes);
      EXPECT_TRUE(std::ranges::all_of(written, [](bool v) { return v; }));
      EXPECT_EQ(result, expected);
      auto end = cursor->Next(maximum);
      ASSERT_TRUE(end.has_value());
      EXPECT_FALSE(end->has_value());
    }
  }
}

TEST(ImportTransformsTest, HeadRowsAndColumnsMoveCompletePairedParts) {
  for (const auto encoding : {Encoding::kF32, Encoding::kMxfp8}) {
    const auto rows_format = Matrix(encoding, 64, 17);
    auto rows = HeadPermutationPlan::Make(rows_format, HeadAxis::kRows, 2, 6, 2, 3);
    ASSERT_TRUE(rows.has_value());
    EXPECT_EQ(rows->format(), rows_format);
    CheckPermutation(*rows, HeadAxis::kRows, 2, 6, 2, 3);
    const auto columns_format = Matrix(encoding, 192, 2);
    auto columns = HeadPermutationPlan::Make(columns_format, HeadAxis::kColumns, 2, 6, 32);
    ASSERT_TRUE(columns.has_value());
    EXPECT_EQ(columns->format(), columns_format);
    CheckPermutation(*columns, HeadAxis::kColumns, 2, 6, 32, 0);
  }
  auto vector = Matrix(Encoding::kF32, 6, 1);
  vector.shape.rank = 1;
  auto plan = HeadPermutationPlan::Make(vector, HeadAxis::kColumns, 2, 6, 1);
  ASSERT_TRUE(plan.has_value());
  CheckPermutation(*plan, HeadAxis::kColumns, 2, 6, 1, 0);
  EXPECT_FALSE(plan->CursorFor(PartRole::kBlockScales).has_value());
}

TEST(ImportTransformsTest, HeadFactoryRefusesUnprovablePermutations) {
  auto f = Matrix(Encoding::kMxfp8, 192, 16);
  EXPECT_FALSE(HeadPermutationPlan::Make(f, HeadAxis::kRows, 0, 6, 1).has_value());
  EXPECT_FALSE(HeadPermutationPlan::Make(f, HeadAxis::kRows, 2, 0, 1).has_value());
  EXPECT_FALSE(HeadPermutationPlan::Make(f, HeadAxis::kRows, 4, 6, 1).has_value());
  EXPECT_FALSE(HeadPermutationPlan::Make(f, HeadAxis::kRows, 2, 6, 0).has_value());
  EXPECT_FALSE(HeadPermutationPlan::Make(f, HeadAxis::kRows, 2, 6, 3).has_value());
  EXPECT_FALSE(HeadPermutationPlan::Make(f, HeadAxis::kRows, 2, 6, 2, 5).has_value());
  EXPECT_FALSE(
      HeadPermutationPlan::Make(f, HeadAxis::kRows, 1, 2, std::numeric_limits<std::uint64_t>::max())
          .has_value());
  EXPECT_FALSE(HeadPermutationPlan::Make(f, HeadAxis::kColumns, 2, 6, 31).has_value());
  EXPECT_FALSE(HeadPermutationPlan::Make(f, HeadAxis::kColumns, 2, 12, 16).has_value());
  EXPECT_FALSE(HeadPermutationPlan::Make(f, HeadAxis::kColumns, 2, 6, 32, 1).has_value());
  EXPECT_FALSE(HeadPermutationPlan::Make(f, static_cast<HeadAxis>(255), 2, 6, 1).has_value());
  f.shape.ne[0] = 191;
  EXPECT_FALSE(HeadPermutationPlan::Make(f, HeadAxis::kRows, 2, 6, 1).has_value());
  f = Matrix(Encoding::kF16, 192, 16);
  f.layout = Layout::kDense;
  EXPECT_FALSE(HeadPermutationPlan::Make(f, HeadAxis::kRows, 2, 6, 1).has_value());
  f = Matrix(Encoding::kMxfp8, 192, 16);
  f.params.tensor_scale = true;
  EXPECT_FALSE(HeadPermutationPlan::Make(f, HeadAxis::kRows, 2, 6, 1).has_value());
  f = Matrix(Encoding::kF32, 192, 16);
  f.stacked = 2;
  EXPECT_FALSE(HeadPermutationPlan::Make(f, HeadAxis::kRows, 2, 6, 1).has_value());
  f = Matrix(Encoding::kF32, 192, 16);
  f.shape.rank = 3;
  f.shape.ne[2] = 2;
  EXPECT_FALSE(HeadPermutationPlan::Make(f, HeadAxis::kRows, 2, 6, 1).has_value());
  f = Matrix(Encoding::kF32, std::numeric_limits<std::uint64_t>::max(), 16);
  EXPECT_FALSE(HeadPermutationPlan::Make(f, HeadAxis::kRows, 2, 6, 1).has_value());
}

}  // namespace
}  // namespace llmp::import
