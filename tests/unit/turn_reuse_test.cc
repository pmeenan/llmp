// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "runtime/turn_reuse.h"

#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <string>

#include "chat/chat.h"

namespace {
namespace rt = jitllm::runtime;
namespace ch = jitllm::chat;
using Clock = std::chrono::steady_clock;

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
