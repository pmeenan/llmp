// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0
#include "engine/gemma4_greedy.h"

#include <gtest/gtest.h>

#include <array>
#include <limits>
#include <vector>
namespace en = jitllm::engine;
TEST(Gemma4Greedy, EveryAcceptedLengthCarriesOnlyThePendingNextAnchor) {
  const std::array<std::int32_t, 3> drafts{1, 2, 3};
  for (std::uint32_t keep = 1; keep <= 4; ++keep) {
    std::array<float, 20> heads{};
    for (std::uint32_t row = 0; row < 4; ++row)
      heads[row * 5 + (row < keep - 1 ? std::uint32_t(drafts[row]) : 4U)] = 1;
    auto decision = en::JudgeGemma4Greedy(drafts, heads, 5);
    ASSERT_TRUE(decision);
    EXPECT_EQ(decision->keep, keep);
    EXPECT_EQ(decision->next_anchor, 4);
  }
}
TEST(Gemma4Greedy, LowestIdTieAndDepthOneRemainTargetAuthoritative) {
  const std::array<std::int32_t, 1> drafts{0};
  const std::array<float, 6> heads{2, 2, 1, 0, 4, 4};
  auto decision = en::JudgeGemma4Greedy(drafts, heads, 3);
  ASSERT_TRUE(decision);
  EXPECT_EQ(decision->keep, 2U);
  EXPECT_EQ(decision->next_anchor, 1);
}
TEST(Gemma4Greedy, RefusesInvalidDepthVocabularyAndIncompleteRows) {
  const std::array<std::int32_t, 4> drafts{0, 0, 0, 0};
  const std::array<float, 10> heads{};
  EXPECT_FALSE(en::JudgeGemma4Greedy({}, {}, 2));
  EXPECT_FALSE(en::JudgeGemma4Greedy(drafts, heads, 2));
  EXPECT_FALSE(en::JudgeGemma4Greedy(std::span(drafts).first(1), heads, 0));
  EXPECT_FALSE(en::JudgeGemma4Greedy(std::span(drafts).first(1), heads, 262145));
  EXPECT_FALSE(en::JudgeGemma4Greedy(std::span(drafts).first(1), std::span(heads).first(3), 2));
}
TEST(Gemma4Greedy, RefusesNoncanonicalDraftEvenAfterAnEarlierMismatch) {
  const std::array<std::int32_t, 2> negative{1, -1}, outside{1, 2};
  const std::array<float, 6> heads{2, 0, 2, 0, 2, 0};
  EXPECT_FALSE(en::JudgeGemma4Greedy(negative, heads, 2));
  EXPECT_FALSE(en::JudgeGemma4Greedy(outside, heads, 2));
}
TEST(Gemma4Greedy, RefusesNonfiniteUnusedTailAndFinalRows) {
  const std::array<std::int32_t, 2> drafts{1, 1};
  for (const auto invalid :
       {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity()}) {
    std::array<float, 6> heads{2, 0, 2, 0, 2, 0};
    heads[5] = invalid;
    EXPECT_FALSE(en::JudgeGemma4Greedy(drafts, heads, 2));
  }
}
