// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <utility>
#include <vector>

#include "format/layout.h"
#include "import/route.h"

namespace llmp::import {
namespace {
using format::Encoding;
using format::Layout;
using format::PartRole;

std::expected<void, Error> Stop() {
  return std::unexpected(Error{.rule = Rule::kIo, .reason = "test sink refused"});
}
void Check(const PartRoutePlan& plan, std::span<const std::byte> source,
           std::span<const std::byte> expected, std::size_t chunk,
           std::span<std::byte> workspace = {}) {
  auto run = plan.Start(workspace);
  ASSERT_TRUE(run);
  EXPECT_FALSE(run->Finish());
  std::vector<std::byte> guarded(expected.size() + 2, std::byte{0xAB});
  auto output = std::span(guarded).subspan(1, expected.size());
  std::vector<unsigned> coverage(expected.size(), 0);
  const PartSink sink = [&](std::uint64_t at,
                            std::span<const std::byte> bytes) -> std::expected<void, Error> {
    EXPECT_LE(at, output.size());
    if (at > output.size() || bytes.size() > output.size() - at) return Stop();
    for (std::size_t i = 0; i < bytes.size(); ++i) {
      const auto dest = static_cast<std::size_t>(at) + i;
      ++coverage[dest];
      output[dest] = bytes[i];
    }
    return {};
  };
  for (std::size_t at = 0; at < source.size();) {
    const auto count = std::min(chunk, source.size() - at);
    ASSERT_TRUE(run->Consume(at, source.subspan(at, count), sink));
    at += count;
    EXPECT_EQ(run->received(), at);
  }
  ASSERT_TRUE(run->Finish());
  EXPECT_TRUE(std::ranges::equal(output, expected));
  EXPECT_TRUE(std::ranges::all_of(coverage, [](unsigned n) { return n == 1; }));
  EXPECT_EQ(guarded.front(), std::byte{0xAB});
  EXPECT_EQ(guarded.back(), std::byte{0xAB});
}

TEST(ImportRouteTest, CopiesOddFragmentsAndMovesAcknowledgedProgress) {
  std::array<std::byte, 1031> bytes{};
  for (std::size_t i = 0; i < bytes.size(); ++i) bytes[i] = static_cast<std::byte>(i % 251);
  auto plan = PartRoutePlan::Copy(bytes.size());
  ASSERT_TRUE(plan);
  for (const std::size_t chunk : {1U, 3U, 37U, 4096U}) Check(*plan, bytes, bytes, chunk);
  auto first = plan->Start();
  ASSERT_TRUE(first);
  std::size_t written = 0;
  const PartSink sink = [&](std::uint64_t at,
                            std::span<const std::byte> part) -> std::expected<void, Error> {
    EXPECT_EQ(at, written);
    EXPECT_TRUE(std::ranges::equal(part, std::span(bytes).subspan(written, part.size())));
    written += part.size();
    return {};
  };
  ASSERT_TRUE(first->Consume(0, std::span(bytes).first(1), sink));
  auto moved = std::move(*first);
  EXPECT_FALSE(first->Finish());
  EXPECT_FALSE(first->Consume(1, std::span(bytes).subspan(1), sink));
  EXPECT_EQ(written, 1U);
  ASSERT_TRUE(moved.Consume(1, std::span(bytes).subspan(1), sink));
  EXPECT_TRUE(moved.Finish());
  EXPECT_EQ(written, bytes.size());
}

TEST(ImportRouteTest, PermutationsCoverEveryByteInEitherPartOrder) {
  // Two groups of three heads become three pairs. This fixed reference
  // ordering is independent of the cursor's source-to-destination formula.
  constexpr std::array<std::size_t, 6> source_heads{0, 3, 1, 4, 2, 5};
  for (const auto encoding :
       {Encoding::kF32, Encoding::kBF16, Encoding::kMxfp8, Encoding::kAffine}) {
    for (const auto axis : {HeadAxis::kRows, HeadAxis::kColumns}) {
      format::TensorFormat format{
          .encoding = encoding,
          .layout = encoding == Encoding::kMxfp8 || encoding == Encoding::kAffine ? Layout::kSplit
                                                                                  : Layout::kDense,
          .shape = {.ne = axis == HeadAxis::kRows ? std::array<std::uint64_t, 4>{64, 9, 1, 1}
                                                  : std::array<std::uint64_t, 4>{192, 3, 1, 1},
                    .rank = 2}};
      if (encoding == Encoding::kAffine)
        format.params = {.bits = 4, .group = 32, .scale = format::AffineScale::kBF16};
      auto permutation = HeadPermutationPlan::Make(
          format, axis, 2, 6, axis == HeadAxis::kRows ? 1 : 32, axis == HeadAxis::kRows ? 2 : 0);
      ASSERT_TRUE(permutation);
      EXPECT_FALSE(PartRoutePlan::Permute(*permutation, PartRole::kTensorScale));
      for (const bool reverse : {false, true}) {
        for (std::size_t i = 0; i < permutation->parts().size(); ++i) {
          const auto part = permutation->parts()[reverse ? permutation->parts().size() - 1 - i : i];
          const auto row_bytes = format::RowBytes(format, part.role);
          ASSERT_TRUE(row_bytes);
          const auto row = static_cast<std::size_t>(*row_bytes);
          std::vector<std::byte> source(static_cast<std::size_t>(part.bytes));
          for (std::size_t j = 0; j < source.size(); ++j)
            source[j] = static_cast<std::byte>((j * 7 + i * 23) % 251);
          auto expected = source;
          if (axis == HeadAxis::kRows) {
            for (std::size_t h = 0; h < source_heads.size(); ++h)
              std::copy_n(source.begin() + static_cast<std::ptrdiff_t>((2 + source_heads[h]) * row),
                          row, expected.begin() + static_cast<std::ptrdiff_t>((2 + h) * row));
          } else {
            const auto head = row / 6;
            for (std::size_t r = 0; r < 3; ++r)
              for (std::size_t h = 0; h < source_heads.size(); ++h)
                std::copy_n(
                    source.begin() + static_cast<std::ptrdiff_t>(r * row + source_heads[h] * head),
                    head, expected.begin() + static_cast<std::ptrdiff_t>(r * row + h * head));
          }
          const auto original = source;
          auto plan = PartRoutePlan::Permute(*permutation, part.role);
          ASSERT_TRUE(plan);
          for (const std::size_t chunk : {1U, 3U, 37U, 4096U})
            Check(*plan, source, expected, chunk);
          EXPECT_EQ(source, original);
        }
      }
    }
  }
}

TEST(ImportRouteTest, ScalarCarryMatchesHistoricalBytesWithPaidWorkspace) {
  constexpr std::array<std::uint16_t, 4> words{0x0000, 0x8000, 0x3F00, 0xBF80};
  constexpr std::array<std::uint32_t, 4> norm{0x3F800000, 0x3F800000, 0x3FC00000, 0x00000000};
  constexpr std::array<std::uint32_t, 4> exp{0xBF800000, 0xBF800000, 0xBFD3094C, 0xBEBC5AB2};
  std::array<std::byte, 8> source{};
  for (std::size_t i = 0; i < words.size(); ++i)
    for (std::size_t b = 0; b < 2; ++b)
      source[2 * i + b] = static_cast<std::byte>(words[i] >> (8 * b));
  for (const auto fold : {ScalarFold::kNormPlusOne, ScalarFold::kANegExp}) {
    auto plan = PartRoutePlan::Fold(fold, words.size());
    ASSERT_TRUE(plan);
    std::array<std::byte, 16> expected{};
    const auto& values = fold == ScalarFold::kNormPlusOne ? norm : exp;
    for (std::size_t i = 0; i < values.size(); ++i)
      for (std::size_t b = 0; b < 4; ++b)
        expected[4 * i + b] = static_cast<std::byte>(values[i] >> (8 * b));
    for (const std::size_t workspace_size : {4U, 5U, 65536U}) {
      std::vector<std::byte> workspace(workspace_size);
      for (const std::size_t chunk : {1U, 3U, 7U, 4096U})
        Check(*plan, source, expected, chunk, workspace);
    }
  }
}

// Row-major one-byte block scales into the scale atoms (format/layout.h
// ScaleAtomOffset), losslessly, whatever the input's fragments.
TEST(ImportRouteTest, ScaleAtomsSwizzleWholeBandsLosslessly) {
  constexpr std::uint64_t kRows = 384, kBlocks = 12;
  std::vector<std::byte> source(kRows * kBlocks);
  for (std::size_t i = 0; i < source.size(); ++i)
    source[i] = static_cast<std::byte>((i * 131 + 7) % 251);
  std::vector<std::byte> expected(source.size());
  for (std::uint64_t r = 0; r < kRows; ++r)
    for (std::uint64_t b = 0; b < kBlocks; ++b)
      expected[format::ScaleAtomOffset(r, b, kBlocks)] = source[(r * kBlocks) + b];
  auto plan = PartRoutePlan::ScaleAtoms(kRows, kBlocks);
  ASSERT_TRUE(plan);
  EXPECT_EQ(plan->input_bytes(), source.size());
  EXPECT_EQ(plan->output_bytes(), source.size());
  std::vector<std::byte> workspace(65536);
  for (const std::size_t chunk : {1U, 3U, 37U, 1536U, 4096U})
    Check(*plan, source, expected, chunk, workspace);
  // Whole 128-row atoms of 4 to 256 scales only; two bands of workspace.
  EXPECT_FALSE(PartRoutePlan::ScaleAtoms(100, 12));
  EXPECT_FALSE(PartRoutePlan::ScaleAtoms(128, 6));
  EXPECT_FALSE(PartRoutePlan::ScaleAtoms(128, 260));
  EXPECT_FALSE(PartRoutePlan::ScaleAtoms(0, 4));
  EXPECT_FALSE(plan->Start(std::span(workspace).first(2 * 128 * kBlocks - 1)));
  EXPECT_TRUE(plan->Start(std::span(workspace).first(2 * 128 * kBlocks)));
}

// Two parts' rows side by side: each lane's rows at its offset in a row,
// as they arrive.
TEST(ImportRouteTest, StridedLanesInterleaveRows) {
  constexpr std::uint64_t kRows = 37, kCodes = 8, kScales = 2, kStride = kCodes + kScales;
  std::vector<std::byte> codes(kRows * kCodes), scales(kRows * kScales);
  for (std::size_t i = 0; i < codes.size(); ++i) codes[i] = static_cast<std::byte>(i % 251);
  for (std::size_t i = 0; i < scales.size(); ++i) scales[i] = static_cast<std::byte>(200 + i % 50);
  for (const bool scale_lane : {false, true}) {
    const auto& source = scale_lane ? scales : codes;
    const auto row = scale_lane ? kScales : kCodes;
    auto plan = PartRoutePlan::Strided(kRows, row, kStride, scale_lane ? kCodes : 0);
    ASSERT_TRUE(plan);
    EXPECT_EQ(plan->output_bytes(), source.size());
    EXPECT_EQ(plan->span(), kRows * kStride);
    for (const std::size_t chunk : {1U, 3U, 9U, 4096U}) {
      auto run = plan->Start();
      ASSERT_TRUE(run);
      std::vector<std::byte> out(kRows * kStride, std::byte{0xEE});
      const PartSink sink = [&](std::uint64_t at,
                                std::span<const std::byte> bytes) -> std::expected<void, Error> {
        if (at > out.size() || bytes.size() > out.size() - at) return Stop();
        std::ranges::copy(bytes, out.begin() + static_cast<std::ptrdiff_t>(at));
        return {};
      };
      for (std::size_t at = 0; at < source.size(); at += chunk)
        ASSERT_TRUE(run->Consume(
            at, std::span(source).subspan(at, std::min(chunk, source.size() - at)), sink));
      ASSERT_TRUE(run->Finish());
      for (std::uint64_t r = 0; r < kRows; ++r)
        for (std::uint64_t b = 0; b < kStride; ++b) {
          const bool mine = scale_lane ? b >= kCodes : b < kCodes;
          const auto want =
              !mine ? std::byte{0xEE} : source[(r * row) + (scale_lane ? b - kCodes : b)];
          ASSERT_EQ(out[(r * kStride) + b], want) << r << ':' << b;
        }
    }
  }
  EXPECT_FALSE(PartRoutePlan::Strided(4, 8, 6, 0));   // a row past its stride
  EXPECT_FALSE(PartRoutePlan::Strided(4, 2, 10, 9));  // a lane past its stride
  EXPECT_FALSE(PartRoutePlan::Strided(0, 2, 10, 0));
}

// A fold under a head reorder (A: -exp(A_log) per value head, tiled): each
// value folded into its reordered place.
TEST(ImportRouteTest, PermutedFoldsFoldEachHeadIntoItsPlace) {
  constexpr std::array<std::size_t, 6> source_heads{0, 3, 1, 4, 2, 5};
  const format::TensorFormat bf16{.encoding = Encoding::kBF16,
                                  .layout = Layout::kDense,
                                  .shape = {.ne = {6, 1, 1, 1}, .rank = 2}};
  auto permutation = HeadPermutationPlan::Make(bf16, HeadAxis::kColumns, 2, 6, 1, 0);
  ASSERT_TRUE(permutation);
  constexpr std::array<std::uint16_t, 6> words{0x0000, 0x3F00, 0xBF80, 0x4000, 0x3E80, 0xC040};
  std::array<std::byte, 12> source{};
  for (std::size_t i = 0; i < words.size(); ++i)
    for (std::size_t b = 0; b < 2; ++b)
      source[2 * i + b] = static_cast<std::byte>(words[i] >> (8 * b));
  for (const auto fold : {ScalarFold::kNormPlusOne, ScalarFold::kANegExp}) {
    std::array<std::byte, 24> expected{};
    for (std::size_t h = 0; h < source_heads.size(); ++h)
      ASSERT_TRUE(FoldBf16(fold, std::span(source).subspan(2 * source_heads[h], 2),
                           std::span(expected).subspan(4 * h, 4)));
    auto plan = PartRoutePlan::PermuteFold(*permutation, PartRole::kData, fold);
    ASSERT_TRUE(plan);
    EXPECT_EQ(plan->output_bytes(), 24U);
    std::vector<std::byte> workspace(65536);
    for (const std::size_t chunk : {1U, 5U, 4096U})
      Check(*plan, source, expected, chunk, workspace);
    EXPECT_FALSE(plan->Start(std::span(workspace).first(35)));
  }
  EXPECT_FALSE(
      PartRoutePlan::PermuteFold(*permutation, PartRole::kData, static_cast<ScalarFold>(255)));
}

TEST(ImportRouteTest, RejectsGeometryWorkspaceAndNoncontiguousInputBeforeSink) {
  EXPECT_FALSE(PartRoutePlan::Copy(0));
  EXPECT_FALSE(PartRoutePlan::Fold(static_cast<ScalarFold>(255), 1));
  EXPECT_FALSE(PartRoutePlan::Fold(ScalarFold::kNormPlusOne, 0));
  EXPECT_FALSE(
      PartRoutePlan::Fold(ScalarFold::kNormPlusOne, std::numeric_limits<std::uint64_t>::max()));
  auto plan = PartRoutePlan::Fold(ScalarFold::kNormPlusOne, 2);
  ASSERT_TRUE(plan);
  std::array<std::byte, 8> bytes{};
  EXPECT_FALSE(plan->Start(std::span(bytes).first(3)));
  std::size_t calls = 0;
  const PartSink sink = [&](std::uint64_t,
                            std::span<const std::byte>) -> std::expected<void, Error> {
    ++calls;
    return {};
  };
  for (const int bad : {0, 1, 2}) {
    auto run = plan->Start(bytes);
    ASSERT_TRUE(run);
    const std::array<std::byte, 5> source{};
    if (bad == 0) EXPECT_FALSE(run->Consume(1, std::span(source).first(2), sink));
    if (bad == 1) EXPECT_FALSE(run->Consume(0, source, sink));
    if (bad == 2) EXPECT_FALSE(run->Consume(0, std::span(bytes).first(2), sink));
    EXPECT_FALSE(run->Finish());
    EXPECT_FALSE(run->Consume(0, {}, sink));
  }
  EXPECT_EQ(calls, 0U);
}

TEST(ImportRouteTest, SinkFailureCancellationReentryAndAssignmentAreTerminal) {
  auto plan = PartRoutePlan::Copy(4);
  ASSERT_TRUE(plan);
  std::array<std::byte, 4> bytes{};
  for (const int failure : {0, 1, 2, 3}) {
    auto run = plan->Start();
    ASSERT_TRUE(run);
    std::size_t calls = 0;
    const PartSink sink = [&](std::uint64_t,
                              std::span<const std::byte>) -> std::expected<void, Error> {
      ++calls;
      EXPECT_FALSE(run->Finish());
      if (failure == 0) return Stop();
      if (failure == 1) run->Cancel();
      if (failure == 2) {
        const PartSink nested = [](std::uint64_t, std::span<const std::byte>) { return Stop(); };
        EXPECT_FALSE(run->Consume(0, {}, nested));
      }
      if (failure == 3) {
        // Replacing an active destination used to overwrite its four-byte
        // bound with one, then let the outer Consume advance past that bound.
        auto replacement_plan = PartRoutePlan::Copy(1);
        EXPECT_TRUE(replacement_plan);
        if (!replacement_plan) return Stop();
        auto replacement = replacement_plan->Start();
        EXPECT_TRUE(replacement);
        if (!replacement) return Stop();
        *run = std::move(*replacement);
      }
      return {};
    };
    EXPECT_FALSE(run->Consume(0, bytes, sink));
    EXPECT_FALSE(run->Finish());
    EXPECT_FALSE(run->Consume(0, bytes, sink));
    EXPECT_FALSE(run->Consume(4, bytes, sink));
    EXPECT_EQ(calls, 1U);
    EXPECT_EQ(run->received(), 0U);
  }
  auto fold = PartRoutePlan::Fold(ScalarFold::kANegExp, 2);
  ASSERT_TRUE(fold);
  std::array<std::byte, 4> workspace{};
  auto run = fold->Start(workspace);
  ASSERT_TRUE(run);
  // First scalar writes privately, second overflows binary64. No retry or
  // completion may certify this partially written output.
  const std::array source{std::byte{0}, std::byte{0}, std::byte{0x7F}, std::byte{0x7F}};
  std::size_t written = 0;
  EXPECT_FALSE(run->Consume(
      0, source,
      [&](std::uint64_t, std::span<const std::byte> output) -> std::expected<void, Error> {
        written += output.size();
        return {};
      }));
  EXPECT_EQ(written, 4U);
  EXPECT_EQ(run->received(), 2U);
  EXPECT_FALSE(run->Finish());
}
}  // namespace
}  // namespace llmp::import
