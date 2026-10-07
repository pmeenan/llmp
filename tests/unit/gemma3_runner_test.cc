// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Metadata-only controls: providers/artifact payload are never opened.
#include "engine/gemma3_runner.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <vector>

namespace en = jitllm::engine;
TEST(Gemma3Runner, UninitializedLifecycleRefusesWithoutPublishingOrOwningCopies) {
  en::PagedNode node({});
  en::Gemma3Runner runner(node, {}, 0, 0);
  en::LiveState::CopyRetirement retirement = en::LiveState::CopyRetirement::kUnproven;
  EXPECT_FALSE(runner.CopyState(0, nullptr, {}, false, &retirement));
  EXPECT_EQ(retirement, en::LiveState::CopyRetirement::kProven);
  EXPECT_FALSE(runner.PrepareRestore(0, 1, {}, ""));
  EXPECT_FALSE(runner.CompleteRestore(0, 1));
  EXPECT_FALSE(runner.Adopt(0, 1, {}, "malformed"));
  EXPECT_EQ(node.StateCapacity(), 0U);
  EXPECT_FALSE(runner.Held(0));
}
TEST(Gemma3Runner, CheckpointFootprintRequiresOrderedWholeExtentsAtRingAndContextBoundaries) {
  const auto& profile = jitllm::model::Gemma3_4BQat();
  for (const auto context : {4096U, 8448U, 131072U}) {
    auto layout = jitllm::model::Gemma3State(profile, context, 128);
    ASSERT_TRUE(layout);
    for (const auto positions : {1U, 1024U, 1152U, 1281U, context - 64U, context}) {
      auto needed = jitllm::model::Gemma3UsedState(profile, *layout, positions);
      ASSERT_TRUE(needed);
      std::vector<en::LiveState::Range> footprint;
      for (const auto& range : *needed)
        for (auto extent = range.offset / en::kPagedExtent;
             extent <= (range.offset + range.bytes - 1) / en::kPagedExtent; ++extent)
          footprint.push_back(
              {0, extent * en::kPagedExtent,
               std::min(en::kPagedExtent, layout->bytes - extent * en::kPagedExtent)});
      std::ranges::sort(footprint, {}, &en::LiveState::Range::offset);
      footprint.erase(
          std::unique(footprint.begin(), footprint.end(),
                      [](const auto& a, const auto& b) { return a.offset == b.offset; }),
          footprint.end());
      ASSERT_TRUE(en::Gemma3CheckpointFootprint(profile, *layout, positions, footprint));
      auto bad = footprint;
      bad.pop_back();
      EXPECT_FALSE(en::Gemma3CheckpointFootprint(profile, *layout, positions, bad));
      bad = footprint;
      ++bad[0].offset;
      EXPECT_FALSE(en::Gemma3CheckpointFootprint(profile, *layout, positions, bad));
      bad = footprint;
      --bad[0].bytes;
      EXPECT_FALSE(en::Gemma3CheckpointFootprint(profile, *layout, positions, bad));
      bad = footprint;
      bad[0].region = 1;
      EXPECT_FALSE(en::Gemma3CheckpointFootprint(profile, *layout, positions, bad));
      bad = footprint;
      bad.push_back(bad.back());
      EXPECT_FALSE(en::Gemma3CheckpointFootprint(profile, *layout, positions, bad));
      EXPECT_FALSE(en::Gemma3CheckpointFootprint(profile, *layout, 0, footprint));
      EXPECT_FALSE(en::Gemma3CheckpointFootprint(profile, *layout, context + 1U, footprint));
    }
    EXPECT_TRUE(en::Gemma3CheckpointFootprint(profile, *layout, 0, {}));
  }
}
