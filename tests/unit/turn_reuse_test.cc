// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "runtime/turn_reuse.h"

#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <limits>
#include <string>

#include "chat/chat.h"

namespace {
namespace rt = llmp::runtime;
namespace ch = llmp::chat;
using Clock = std::chrono::steady_clock;

TEST(TurnReuseTest, CheckpointBudgetCountsHeldFilesAndCannotWrap) {
  EXPECT_TRUE(rt::TurnCheckpointFits(100, 20, 50, 30));
  EXPECT_FALSE(rt::TurnCheckpointFits(100, 20, 50, 31));
  EXPECT_FALSE(rt::TurnCheckpointFits(0, 0, 0, 4096));
  EXPECT_FALSE(rt::TurnCheckpointFits(100, 101, 0, 0));
  const auto most = std::numeric_limits<std::uint64_t>::max();
  EXPECT_TRUE(rt::TurnCheckpointFits(most, most - 2, 1, 1));
  EXPECT_FALSE(rt::TurnCheckpointFits(most, most - 2, 1, 2));
  EXPECT_FALSE(rt::TurnCheckpointFits(most, most, most, 2));
  EXPECT_FALSE(rt::TurnCheckpointFits(100, 0, most, most));
}

TEST(TurnReuseTest, CheckpointDeletionCountsStateSpilledByItsKeptRecordUpdate) {
  std::uint64_t spilled = 0, checkpoints = 80;
  std::size_t attempts = 0;
  const bool fits = rt::TrimTurnCheckpointBudget(
      100, 30, [&] { return spilled; }, [&] { return checkpoints; },
      [&] {
        if (++attempts != 1) return false;  // the remaining files are held
        checkpoints -= 20;
        // Rewriting the kept record needs token storage. Its synchronous
        // reclaim spills another conversation before deletion returns.
        spilled += 30;
        return true;
      });
  EXPECT_FALSE(fits);  // 30 spilled + 60 held checkpoints + 30 incoming
  EXPECT_EQ(attempts, 2U);
}

TEST(TurnReuseTest, CheckpointDeletionCountsFilesRemovedByANestedTrim) {
  std::uint64_t checkpoints = 80;
  std::size_t attempts = 0;
  const bool fits = rt::TrimTurnCheckpointBudget(
      100, 70, [] { return 0; }, [&] { return checkpoints; },
      [&] {
        ++attempts;
        if (attempts != 1) return false;
        checkpoints -= 20;
        // A nested reclaim also removes other optional checkpoint files.
        checkpoints -= 40;
        return true;
      });
  EXPECT_TRUE(fits);
  EXPECT_EQ(attempts, 1U);  // no unnecessary second deletion/refusal
  EXPECT_EQ(checkpoints, 20U);
}

TEST(TurnReuseTest, SelectsTheNearestCheckpointInsideTheExactCommonPrefix) {
  const std::array<std::int32_t, 6> old = {1, 2, 3, 4, 5, 6};
  const std::array<std::int32_t, 6> changed = {1, 2, 3, 9, 5, 6};
  EXPECT_EQ(rt::CommonPrefix(old, changed), 3);
  const auto now = Clock::now();
  const std::array<rt::TurnBoundary, 2> boundaries = {
      rt::TurnBoundary{.position = 2, .created = now},
      rt::TurnBoundary{.position = 5, .created = now}};
  EXPECT_EQ(rt::MatchingTurnBoundary(boundaries, 3, changed.size(), now), 0);
  EXPECT_EQ(rt::MatchingTurnBoundary(boundaries, 6, old.size(), now), 1);
  EXPECT_FALSE(rt::MatchingTurnBoundary(boundaries, 1, changed.size(), now));
}

TEST(TurnReuseTest, DefaultRetentionKeepsAnIdleWeekendButExpiresAtSevenDays) {
  const auto created = Clock::time_point{} + std::chrono::hours(1);
  const std::array boundaries = {rt::TurnBoundary{.position = 2, .created = created}};
  EXPECT_EQ(rt::MatchingTurnBoundary(boundaries, 3, 4, created + std::chrono::hours(72)), 0);
  EXPECT_EQ(rt::MatchingTurnBoundary(
                boundaries, 3, 4, created + std::chrono::hours(168) - std::chrono::nanoseconds(1)),
            0);
  // Looking up a checkpoint does not renew its lifetime.
  EXPECT_FALSE(rt::MatchingTurnBoundary(boundaries, 3, 4, created + std::chrono::hours(168)));
  EXPECT_FALSE(rt::MatchingTurnBoundary(boundaries, 3, 4, created + std::chrono::hours(169)));
}

TEST(TurnReuseTest, ExpiredBoundariesAndPromptEndsCannotSupplyMissingLogits) {
  const auto now = Clock::now();
  const std::array<rt::TurnBoundary, 2> boundaries = {
      rt::TurnBoundary{.position = 2, .created = now - rt::kTurnCheckpointRetention},
      rt::TurnBoundary{.position = 5, .created = now}};
  EXPECT_FALSE(rt::MatchingTurnBoundary(boundaries, 6, 5, now));
  EXPECT_EQ(rt::MatchingTurnBoundary(boundaries, 6, 6, now), 1);
  auto recent = boundaries;
  recent[0].created += std::chrono::nanoseconds(1);
  EXPECT_EQ(rt::MatchingTurnBoundary(recent, 6, 5, now), 0);
  recent[0].created = now + std::chrono::nanoseconds(1);
  EXPECT_FALSE(rt::MatchingTurnBoundary(recent, 3, 6, now));
}

TEST(TurnReuseTest, StableRendererBoundarySurvivesRemovalOfAssistantReasoning) {
  for (const auto render : {ch::RenderDeepSeekV4, ch::RenderDeepSeekV4ChatV2, ch::RenderQwen38}) {
    ch::Conversation first;
    first.enable_thinking = true;
    first.messages.push_back({.role = ch::Role::kUser,
                              .content = "Inspect this code.",
                              .reasoning_content = {},
                              .tool_calls = {}});
    auto before = render(first);
    ASSERT_TRUE(before);
    std::size_t stable = 0;
    for (const ch::Boundary& boundary : before->boundaries) {
      if (boundary.kind == ch::BoundaryKind::kGenerationPrompt) {
        stable = boundary.offset;
      }
    }
    ASSERT_GT(stable, 0);
    ASSERT_LT(stable, before->text.size());
    bool marked = false;
    for (const auto& special : before->specials) {
      marked = marked || special.offset == stable;
    }
    EXPECT_TRUE(marked);
    auto second = first;
    second.messages.push_back({.role = ch::Role::kAssistant,
                               .content = "The function is correct.",
                               .reasoning_content = {},
                               .tool_calls = {}});
    second.messages.push_back({.role = ch::Role::kUser,
                               .content = "Now add error handling.",
                               .reasoning_content = {},
                               .tool_calls = {}});
    auto after = render(second);
    ASSERT_TRUE(after);
    EXPECT_EQ(before->text.substr(0, stable), after->text.substr(0, stable));
    // The assistant's empty reasoning acquires a closing marker; Qwen's
    // adjacent newlines may also merge into another token at this opening.
    EXPECT_EQ(before->text.substr(stable).find("</think>"), std::string::npos);
    EXPECT_NE(after->text.substr(stable).find("</think>"), std::string::npos);
  }
}

}  // namespace
