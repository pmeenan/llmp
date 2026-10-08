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

#include "model/dsv4.h"
#include "model/gemma2.h"
#include "model/gemma3.h"
#include "model/gemma4.h"
#include "model/qwen38.h"

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

template <typename Segment, typename Profile, typename StateBuilder, typename ChunkBuilder>
void GemmaOracle(const Profile& profile, StateBuilder state_builder, ChunkBuilder chunk_builder) {
  auto state = state_builder(profile, 8192, 512);
  ASSERT_TRUE(state);
  for (const auto past : {0U, profile.window - 3, 5000U}) {
    for (const auto rows : {1U, 17U, 511U}) {
      std::vector<std::int32_t> tokens(rows, 1);
      const std::array segments{Segment{.slot = 0, .n_past = past, .tokens = tokens}};
      auto chunk = chunk_builder(profile, *state, segments, true, 256U, 0U);
      ASSERT_TRUE(chunk);
      const auto& in = chunk->segments[0];
      for (std::uint32_t r = 0; r < rows; ++r) {
        for (std::uint32_t c = 0; c < in.global_n_kv; ++c)
          ASSERT_EQ(in.global_mask[std::size_t{r} * in.global_n_kv + c],
                    c <= past + r ? 0 : 0xFC00);
        for (std::uint32_t c = 0; c < in.local_n_kv; ++c)
          ASSERT_EQ(in.local_mask[std::size_t{r} * in.local_n_kv + c],
                    OldRing(c, past + r, past + rows, state->local_cells, profile.window));
      }
    }
  }
}

TEST(HostMaskTest, AllGemmaBuildersMatchCompleteOldGlobalAndWindowMasks) {
  GemmaOracle<md::Gemma2Segment>(md::Gemma2_2B(), md::Gemma2State, md::Gemma2Chunk);
  GemmaOracle<md::Gemma3Segment>(md::Gemma3_4BQat(), md::Gemma3State, md::Gemma3Chunk);
  GemmaOracle<md::Gemma4Segment>(
      md::Gemma4_26BA4B(), md::Gemma4State,
      [](const auto& p, const auto& state, auto segments, bool masks, auto align, auto) {
        return md::Gemma4Chunk(p, state, segments, masks, align);
      });
  GemmaOracle<md::Gemma4Segment>(
      md::Gemma4_31B(), md::Gemma4State,
      [](const auto& p, const auto& state, auto segments, bool masks, auto align, auto) {
        return md::Gemma4Chunk(p, state, segments, masks, align);
      });
}

TEST(HostMaskTest, DeepSeekRawAndAllCompressedMasksMatchCompleteOldFormula) {
  const auto& profile = md::Dsv4Flash();
  for (const auto mode : {md::Dsv4Window::kFull, md::Dsv4Window::kRing}) {
    auto state = md::Dsv4State(profile, 16384, 512, mode);
    ASSERT_TRUE(state);
    for (const auto past : {0U, 257U, 4090U, 8191U}) {
      for (const auto rows : {1U, 17U, 511U}) {
        auto in = md::Dsv4Chunk(profile, *state, past, rows, true, true);
        ASSERT_TRUE(in);
        for (std::uint32_t r = 0; r < rows; ++r)
          for (std::uint32_t c = 0; c < in->raw_n_kv; ++c)
            ASSERT_EQ(in->raw_mask[std::size_t{r} * in->raw_n_kv + c],
                      OldRing(c, past + r, past + rows, state->raw_cells, profile.window));
        const auto comp = [&](const auto& plan, const auto& mask) {
          ASSERT_EQ(mask.size(), std::size_t{rows} * plan.n_kv);
          for (std::uint32_t r = 0; r < rows; ++r)
            for (std::uint32_t c = 0; c < plan.n_kv; ++c)
              ASSERT_EQ(mask[std::size_t{r} * plan.n_kv + c],
                        std::int64_t{c} < plan.n_visible[r] ? 0 : 0xFC00);
        };
        comp(in->csa, in->csa_mask);
        comp(in->hca, in->hca_mask);
        comp(in->lid, in->lid_mask);
      }
    }
  }
}

TEST(HostMaskTest, QwenDenseAndThreeValuedQsaBuffersMatchOldBits) {
  const auto& p = md::Qwen38Flash();
  for (const auto past : {0U, 2048U, 2300U, 4095U}) {
    for (const auto rows : {1U, 17U, 257U}) {
      const auto end = past + rows;
      const auto read = (end + 255) / 256 * 256;
      auto in = md::Qwen38Rows(p, 8192, past, rows, read, true, true);
      ASSERT_TRUE(in);
      for (std::uint32_t r = 0; r < rows; ++r) {
        for (std::uint32_t c = 0; c < read; ++c) {
          EXPECT_EQ(in->mask[std::size_t{r} * read + c], c <= past + r ? 0 : 0xFC00);
          const auto expected = c <= past + r ? 0.0f : -std::numeric_limits<float>::infinity();
          EXPECT_EQ(std::bit_cast<std::uint32_t>(in->mask_f32[std::size_t{r} * read + c]),
                    std::bit_cast<std::uint32_t>(expected));
        }
        if (!in->qsa_select) continue;
        const auto full = end / p.indexer_ratio;
        const auto tail = (std::uint64_t{past} + r + 1) / p.indexer_ratio * p.indexer_ratio;
        for (std::uint32_t b = 0; b < in->qsa.blocks; ++b) {
          auto expected = b >= full ? -std::numeric_limits<float>::infinity()
                          : std::uint64_t{b} * p.indexer_ratio >= tail ? 1e9f
                                                                       : 0.0f;
          if (full < in->qsa.blocks && b == full) expected = 1e9f;
          EXPECT_EQ(std::bit_cast<std::uint32_t>(in->qsa.bias[std::size_t{r} * in->qsa.blocks + b]),
                    std::bit_cast<std::uint32_t>(expected));
        }
      }
    }
  }
}
}  // namespace
