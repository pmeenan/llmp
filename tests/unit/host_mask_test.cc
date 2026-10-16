// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "model/host_mask.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

namespace {
namespace md = llmp::model;

std::uint16_t OldRing(std::uint64_t cell, std::uint64_t position, std::uint64_t end,
                      std::uint64_t capacity, std::uint64_t window) {
  if (cell >= end) return 0xFC00;
  const auto held = cell + ((end - 1 - cell) / capacity) * capacity;
  return held <= position && position - held < window ? 0 : 0xFC00;
}

TEST(HostMaskTest, RingIntervalsMatchOldFormulaAndAuthenticateEveryBit) {
  for (const std::uint64_t capacity : {1U, 3U, 16U, 257U}) {
    for (const auto end :
         std::array<std::uint64_t, 4>{1, capacity, capacity + 1, capacity * 3 + 7}) {
      for (const auto position : std::array<std::uint64_t, 3>{0, end / 2, end - 1}) {
        for (const auto window : std::array<std::uint64_t, 4>{1, 5, capacity, capacity + 3}) {
          for (const auto width : std::array<std::uint64_t, 3>{0, capacity / 2, capacity}) {
            auto runs = md::HostMaskRing(width, position, end, capacity, window);
            ASSERT_TRUE(runs);
            std::vector<std::uint16_t> row(width, 0xFC00);
            ASSERT_TRUE(md::FillHostMaskVisible(std::span{row}, *runs, std::uint16_t{0}));
            for (std::size_t c = 0; c < row.size(); ++c)
              ASSERT_EQ(row[c], OldRing(c, position, end, capacity, window));
            ASSERT_TRUE(md::MatchHostMask(row, *runs));
            for (std::size_t c = 0; c < row.size(); ++c) {
              row[c] ^= 1;  // poison even a single low bit, including blocked cells
              ASSERT_FALSE(md::MatchHostMask(row, *runs));
              row[c] ^= 1;
            }
          }
        }
      }
    }
  }
  const auto limit = std::numeric_limits<std::uint64_t>::max();
  auto near_limit = md::HostMaskRing(17, limit - 1, limit, 17, 13);
  ASSERT_TRUE(near_limit);
  std::vector<std::uint16_t> row(17, 0xFC00);
  ASSERT_TRUE(md::FillHostMaskVisible(std::span{row}, *near_limit, std::uint16_t{0}));
  for (std::size_t c = 0; c < row.size(); ++c)
    EXPECT_EQ(row[c], OldRing(c, limit - 1, limit, 17, 13));
}

TEST(HostMaskTest, InvalidDescriptorsRefuseBeforeChangingOutput) {
  EXPECT_FALSE(md::HostMaskRing(8, 0, 1, 0, 8));
  EXPECT_FALSE(md::HostMaskRing(9, 0, 1, 8, 8));
  EXPECT_FALSE(md::HostMaskRing(8, 0, 1, 8, 0));
  EXPECT_FALSE(md::HostMaskRing(8, 1, 1, 8, 8));
  EXPECT_FALSE(md::HostMaskRing(8, UINT64_MAX, UINT64_MAX, 8, 8));
  EXPECT_FALSE(md::HostMaskPrefix(8, 9));
  EXPECT_FALSE(md::HostMaskPrefix(8, UINT64_MAX));
  std::array<std::uint16_t, 8> output{};
  output.fill(0x1234);
  const auto original = output;
  for (const auto invalid :
       {md::HostMaskIntervals{.width = 9},
        md::HostMaskIntervals{.width = 8, .visible = {{{2, 5}, {4, 7}}}, .count = 2},
        md::HostMaskIntervals{.width = 8, .visible = {{{0, 9}, {}}}, .count = 1},
        md::HostMaskIntervals{.width = 8, .count = 3}}) {
    EXPECT_FALSE(
        md::FillHostMaskVisible(std::span<std::uint16_t>{output}, invalid, std::uint16_t{0}));
    EXPECT_EQ(output, original);
    EXPECT_FALSE(md::MatchHostMask(output, invalid));
  }
  std::array<float, 4> bias{1, 2, 3, 4};
  const auto saved = bias;
  EXPECT_FALSE(md::FillHostQsaBias(bias, 5, 0));
  EXPECT_EQ(bias, saved);
}

// DeepSeek V4 Flash's raw window masks at its geometry (the M3 chunk
// inputs, retired with their runner, built them so): a 16,384-position
// state in chunks of 512, its 128-position window a ring of pad(128 + 512,
// 256) cells or a cell a position; a ring's chunk reads every cell, a full
// cache's the positions up to the next 256 (at least 256).
TEST(HostMaskTest, DeepSeekRawMasksMatchTheOldFormula) {
  constexpr std::uint64_t kWindow = 128;
  for (const bool ring : {false, true}) {
    const std::uint64_t cells = ring ? 768 : 16384;
    for (const auto past : {0U, 257U, 4090U, 8191U}) {
      for (const auto rows : {1U, 17U, 511U}) {
        const std::uint64_t end = std::uint64_t{past} + rows;
        const std::uint64_t width =
            ring ? cells
                 : std::min<std::uint64_t>(
                       cells,
                       std::max<std::uint64_t>(256, (std::min(end, cells) + 255) / 256 * 256));
        for (std::uint32_t r = 0; r < rows; ++r) {
          const auto runs = md::HostMaskRing(width, past + r, end, cells, kWindow);
          ASSERT_TRUE(runs);
          std::vector<std::uint16_t> row(width, 0xFC00);
          ASSERT_TRUE(md::FillHostMaskVisible(std::span{row}, *runs, std::uint16_t{0}));
          for (std::uint64_t c = 0; c < width; ++c)
            ASSERT_EQ(row[c], OldRing(c, past + r, end, cells, kWindow));
          ASSERT_TRUE(md::MatchHostMask(row, *runs));
        }
      }
    }
  }
}

TEST(HostMaskTest, QwenDenseAndThreeValuedQsaBuffersMatchOldBits) {
  constexpr std::uint32_t kIndexerRatio = 4;  // Qwen3.8 Flash's
  for (const auto past : {0U, 2048U, 2300U, 4095U}) {
    for (const auto rows : {1U, 17U, 257U}) {
      const auto end = past + rows;
      const auto read = (end + 255) / 256 * 256;
      const auto full = end / kIndexerRatio;
      const auto blocks = (read + kIndexerRatio - 1) / kIndexerRatio;
      for (std::uint32_t r = 0; r < rows; ++r) {
        auto prefix = md::HostMaskPrefix(read, past + r + 1);
        ASSERT_TRUE(prefix);
        std::vector<std::uint16_t> mask(read, 0xFC00);
        ASSERT_TRUE(md::FillHostMaskVisible(std::span{mask}, *prefix, std::uint16_t{0}));
        for (std::uint32_t c = 0; c < read; ++c) {
          EXPECT_EQ(mask[c], c <= past + r ? 0 : 0xFC00);
        }
        if (full == 0) continue;
        const auto tail = (std::uint64_t{past} + r + 1) / kIndexerRatio;
        std::vector<float> bias(blocks);
        ASSERT_TRUE(md::FillHostQsaBias(std::span{bias}, full, tail));
        for (std::uint32_t b = 0; b < blocks; ++b) {
          auto expected = b >= full                  ? -std::numeric_limits<float>::infinity()
                          : std::uint64_t{b} >= tail ? 1e9f
                                                     : 0.0f;
          if (full < blocks && b == full) expected = 1e9f;
          EXPECT_EQ(std::bit_cast<std::uint32_t>(bias[b]), std::bit_cast<std::uint32_t>(expected));
        }
      }
    }
  }
}
}  // namespace
