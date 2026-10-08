// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Metadata-only controls: providers/artifact payload are never opened.
#include "engine/gemma2_runner.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <limits>
#include <vector>

#include "engine/gemma3_runner.h"
#include "engine/support.h"

namespace en = jitllm::engine;
TEST(Gemma2Runner, UninitializedLifecycleRefusesWithoutPublishingOrOwningCopies) {
  EXPECT_TRUE(en::Gemma2Options{}.device_masks);
  en::PagedNode node({});
  en::Gemma2Runner runner(node, {}, 0, 0);
  en::LiveState::CopyRetirement retirement = en::LiveState::CopyRetirement::kUnproven;
  EXPECT_FALSE(runner.CopyState(0, nullptr, {}, false, &retirement));
  EXPECT_EQ(retirement, en::LiveState::CopyRetirement::kProven);
  EXPECT_FALSE(runner.PrepareRestore(0, 1, {}, ""));
  EXPECT_FALSE(runner.CompleteRestore(0, 1));
  EXPECT_FALSE(runner.Adopt(0, 1, {}, "malformed"));
  EXPECT_EQ(node.StateCapacity(), 0U);
  EXPECT_FALSE(runner.Held(0));
}
TEST(Gemma2Runner, CheckpointFootprintRequiresOrderedWholeExtentsAtRingAndContextBoundaries) {
  const auto& profile = jitllm::model::Gemma2_2B();
  auto layout = jitllm::model::Gemma2State(profile, 8192, 128);
  ASSERT_TRUE(layout);
  for (const auto positions : {1U, 4096U, 4224U, 4353U, 8192U}) {
    auto needed = jitllm::model::Gemma2UsedState(profile, *layout, positions);
    ASSERT_TRUE(needed);
    std::vector<en::LiveState::Range> footprint;
    for (const auto& range : *needed)
      for (auto extent = range.offset / en::kPagedExtent;
           extent <= (range.offset + range.bytes - 1) / en::kPagedExtent; ++extent)
        footprint.push_back(
            {0, extent * en::kPagedExtent,
             std::min(en::kPagedExtent, layout->bytes - extent * en::kPagedExtent)});
    std::ranges::sort(footprint, {}, &en::LiveState::Range::offset);
    footprint.erase(std::unique(footprint.begin(), footprint.end(),
                                [](const auto& a, const auto& b) { return a.offset == b.offset; }),
                    footprint.end());
    ASSERT_TRUE(en::Gemma2CheckpointFootprint(profile, *layout, positions, footprint));
    auto bad = footprint;
    bad.pop_back();
    EXPECT_FALSE(en::Gemma2CheckpointFootprint(profile, *layout, positions, bad));
    bad = footprint;
    ++bad[0].offset;
    EXPECT_FALSE(en::Gemma2CheckpointFootprint(profile, *layout, positions, bad));
    bad = footprint;
    --bad[0].bytes;
    EXPECT_FALSE(en::Gemma2CheckpointFootprint(profile, *layout, positions, bad));
    bad = footprint;
    bad[0].region = 1;
    EXPECT_FALSE(en::Gemma2CheckpointFootprint(profile, *layout, positions, bad));
    bad = footprint;
    bad.push_back(bad.back());
    EXPECT_FALSE(en::Gemma2CheckpointFootprint(profile, *layout, positions, bad));
    bad = footprint;
    std::ranges::reverse(bad);
    EXPECT_FALSE(en::Gemma2CheckpointFootprint(profile, *layout, positions, bad));
    bad = footprint;
    bad.back().offset = std::numeric_limits<std::uint64_t>::max();
    EXPECT_FALSE(en::Gemma2CheckpointFootprint(profile, *layout, positions, bad));
    EXPECT_FALSE(en::Gemma2CheckpointFootprint(profile, *layout, 0, footprint));
    EXPECT_FALSE(en::Gemma2CheckpointFootprint(profile, *layout, 8193, footprint));
  }
  EXPECT_TRUE(en::Gemma2CheckpointFootprint(profile, *layout, 0, {}));
}

TEST(Gemma2Runner, SharedPreparationMeasuresEverySmallOwnerCompositionAndKeepsBudgets) {
  EXPECT_TRUE(en::Gemma2Options{}.shared_q8);
  EXPECT_TRUE(en::Gemma3Options{}.shared_q8);
  for (const std::uint32_t max_rows : {1U, 3U, 128U}) {
    for (const std::uint32_t budget : {2U, 8U, 256U}) {
      for (const std::uint32_t owners : {1U, 2U, 3U, 4U, 8U, 12U}) {
        const auto ordinary = en::support::ChunkMeasurementRows(max_rows, budget, owners, false);
        const auto prepared = en::support::ChunkMeasurementRows(max_rows, budget, owners, true);
        for (const auto& row : ordinary)
          EXPECT_NE(std::ranges::find(prepared, row), prepared.end());
        for (const auto& row : prepared) {
          EXPECT_EQ(row.size(), owners);
          std::uint32_t total = 0;
          for (const auto n : row) {
            EXPECT_GE(n, 1U);
            EXPECT_LE(n, max_rows);
            total += n;
          }
          EXPECT_LE(total, budget);
        }
        // Independently enumerate every bounded ordered row composition.
        if (owners > budget || owners > 8) continue;
        std::vector<std::uint32_t> row(owners, 1);
        const auto check = [&](auto&& self, std::uint32_t owner, std::uint32_t total) -> void {
          if (owner == owners) {
            EXPECT_NE(std::ranges::find(prepared, row), prepared.end());
            return;
          }
          const auto left = std::min(8U, budget) - total;
          for (std::uint32_t n = 1; n <= std::min(max_rows, left); ++n) {
            if (total + n + owners - owner - 1 > std::min(8U, budget)) break;
            row[owner] = n;
            self(self, owner + 1, total + n);
          }
        };
        check(check, 0, 0);
      }
    }
  }
}
