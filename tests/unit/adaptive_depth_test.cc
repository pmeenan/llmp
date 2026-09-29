// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "execution/adaptive_depth.h"

#include <gtest/gtest.h>

#include <cstdint>

namespace {
using jitllm::execution::AdaptiveDepth;

TEST(AdaptiveDepthTest, StartsAtLongerDepthBeforeLearningTheShorterCost) {
  AdaptiveDepth policy;
  for (std::uint32_t i = 0; i < 4; ++i) {
    EXPECT_EQ(policy.Choose(), 3U);
    policy.Observe(policy.Choose(), 4);
  }
  for (std::uint32_t i = 0; i < 4; ++i) {
    EXPECT_EQ(policy.Choose(), 2U);
    policy.Observe(policy.Choose(), 3);
  }
  EXPECT_EQ(policy.Choose(), 3U);
}

TEST(AdaptiveDepthTest, HighAcceptanceSelectsLongerDepthAndLowAcceptanceSelectsShorter) {
  for (const bool accepts : {false, true}) {
    AdaptiveDepth policy(3, 1.25);
    for (std::uint32_t i = 0; i < 38; ++i) {
      const auto d = policy.Choose();
      policy.Observe(d, accepts ? d + 1 : 1);
    }
    EXPECT_EQ(policy.Choose(), accepts ? 3U : 2U);
    const auto chosen = policy.Choose();
    policy.Observe(chosen, accepts ? chosen + 1 : 1);
    EXPECT_EQ(policy.Choose(), accepts ? 2U : 3U);  // bounded probe
    for (std::uint32_t i = 0; i < 4; ++i) {
      const auto d = policy.Choose();
      EXPECT_EQ(d, accepts ? 2U : 3U);
      policy.Observe(d, accepts ? d + 1 : 1);
    }
    EXPECT_EQ(policy.Choose(), accepts ? 3U : 2U);
  }
}

TEST(AdaptiveDepthTest, ProbesAllowThePolicyToFollowChangingAcceptance) {
  AdaptiveDepth policy;
  for (std::uint32_t i = 0; i < 96; ++i) {
    const auto d = policy.Choose();
    policy.Observe(d, d + 1);
  }
  for (std::uint32_t i = 0; i < 512; ++i) {
    const auto d = policy.Choose();
    policy.Observe(d, 1);
  }
  std::uint32_t short_steps = 0;
  for (std::uint32_t i = 0; i < 64; ++i) {
    const auto d = policy.Choose();
    short_steps += d == 2 ? 1 : 0;
    policy.Observe(d, 1);
  }
  EXPECT_GE(short_steps, 48U);
}

TEST(AdaptiveDepthTest, TruncatedOrInvalidStepsDoNotChooseTheDepth) {
  AdaptiveDepth policy;
  for (std::uint32_t i = 0; i < 8; ++i) {
    const auto d = policy.Choose();
    policy.Observe(d, d + 1);
  }
  EXPECT_EQ(policy.Choose(), 3U);
  for (std::uint32_t i = 0; i < 100; ++i) {
    policy.Observe(3, 1, false);
    policy.Observe(0, 1);
    policy.Observe(3, 5);
    policy.Observe(3, 0);
  }
  EXPECT_EQ(policy.Choose(), 3U);
}

TEST(AdaptiveDepthTest, ACheckpointRestoresTheSameChoicesThroughRejectionsAndProbes) {
  AdaptiveDepth policy;
  for (std::uint32_t i = 0; i < 91; ++i) {
    const auto d = policy.Choose();
    policy.Observe(d, i % 7 == 0 ? 1 : d + 1);
  }
  const auto saved = policy;
  auto uninterrupted = policy;
  for (std::uint32_t i = 0; i < 20; ++i) {
    policy.Observe(policy.Choose(), 1);
  }
  policy = saved;
  for (std::uint32_t i = 0; i < 256; ++i) {
    const auto d = uninterrupted.Choose();
    EXPECT_EQ(policy.Choose(), d);
    const auto kept = i % 5 == 0 ? 1 : d + 1;
    uninterrupted.Observe(d, kept);
    policy.Observe(d, kept);
    EXPECT_EQ(policy, uninterrupted);
  }
}
}  // namespace
