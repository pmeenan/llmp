// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "execution/adaptive_depth.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <span>

#include "execution/adaptive_wave_mode.h"

namespace {
using llmp::execution::AdaptiveDepth;

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

using llmp::execution::AdaptiveWaveMode;
using Mode = AdaptiveWaveMode::Mode;

// A calibration like DeepSeek's (model_settings.h kDsv4WaveCosts; its
// first, without wave lanes): a draft-verify wave costs about 1.9, 2.2 and
// 2.9 plain waves at widths 2, 3 and 4.
constexpr AdaptiveWaveMode::Costs kCost = {0, 0, 1.94, 2.21, 2.90};

// `waves` waves of `width`: a draft-verify wave commits `kept` tokens a
// request, a plain one one. Returns how many ran plain.
std::uint32_t RunWaves(AdaptiveWaveMode& policy, std::uint32_t width, std::uint32_t waves,
                       std::uint32_t kept) {
  std::uint32_t plain = 0;
  for (std::uint32_t i = 0; i < waves; ++i) {
    const Mode m = policy.Choose(width);
    plain += m == Mode::kPlain ? 1 : 0;
    policy.Observe(width, m, kept * width, width);
  }
  return plain;
}

TEST(AdaptiveWaveModeTest, ALoneRequestOrAnUncalibratedWidthSpeculates) {
  AdaptiveWaveMode policy(kCost);
  RunWaves(policy, 4, 8, 1);  // acceptance nil
  EXPECT_EQ(policy.Choose(4), Mode::kPlain);
  EXPECT_EQ(policy.Choose(1), Mode::kSpeculative);
  EXPECT_EQ(policy.Choose(5), Mode::kSpeculative);
  AdaptiveWaveMode uncalibrated;
  RunWaves(uncalibrated, 4, 8, 1);
  EXPECT_EQ(uncalibrated.Choose(4), Mode::kSpeculative);
}

TEST(AdaptiveWaveModeTest, ChoosesByAcceptanceAgainstEachWidthsCost) {
  // Three tokens a request: past 2.9 at width 4, so speculation; 2.5: past
  // widths 2 and 3's costs, not width 4's.
  AdaptiveWaveMode high(kCost);
  EXPECT_EQ(RunWaves(high, 4, 3, 3), 0U);  // speculation first
  EXPECT_EQ(high.Choose(4), Mode::kSpeculative);
  AdaptiveWaveMode mid(kCost);
  RunWaves(mid, 2, 3, 2);
  for (std::uint32_t i = 0; i < 40; ++i) {
    mid.Observe(4, Mode::kSpeculative, 10, 4);  // 2.5 a request
  }
  EXPECT_NEAR(mid.kept(), 2.5, 0.01);
  EXPECT_EQ(mid.Choose(2), Mode::kSpeculative);
  EXPECT_EQ(mid.Choose(3), Mode::kSpeculative);
  EXPECT_EQ(mid.Choose(4), Mode::kPlain);
}

TEST(AdaptiveWaveModeTest, SamplingAndAForcedFormOverride) {
  AdaptiveWaveMode policy(kCost);
  RunWaves(policy, 4, 8, 1);
  EXPECT_EQ(policy.Choose(4), Mode::kPlain);
  EXPECT_EQ(policy.Choose(4, /*sampling=*/true), Mode::kSpeculative);
  const AdaptiveWaveMode plain(kCost, AdaptiveWaveMode::Force::kPlain);
  EXPECT_EQ(plain.Choose(2), Mode::kPlain);
  EXPECT_EQ(plain.Choose(1), Mode::kSpeculative);
  AdaptiveWaveMode spec(kCost, AdaptiveWaveMode::Force::kSpeculative);
  RunWaves(spec, 4, 8, 1);
  EXPECT_EQ(spec.Choose(4), Mode::kSpeculative);
}

TEST(AdaptiveWaveModeTest, ProbesFollowAcceptanceAndTheScheduleRepeats) {
  AdaptiveWaveMode policy(kCost);
  // Low acceptance: plain, with two draft-verify waves after every 64.
  EXPECT_GE(RunWaves(policy, 4, 132, 1), 124U);
  // Acceptance rises: the probes see it and speculation returns (the
  // average needs about nine draft-verify waves, five probes).
  EXPECT_LE(RunWaves(policy, 4, 5 * 66, 4), 5U * 64U);
  EXPECT_EQ(RunWaves(policy, 4, 50, 4), 0U);
  // No clock: the same tokens give the same choices.
  AdaptiveWaveMode a(kCost);
  AdaptiveWaveMode b(kCost);
  for (std::uint32_t i = 0; i < 300; ++i) {
    const std::uint32_t kept = 1 + ((i / 37) % 3);
    ASSERT_EQ(a.Choose(4), b.Choose(4));
    RunWaves(a, 4, 1, kept);
    RunWaves(b, 4, 1, kept);
    EXPECT_EQ(a, b);
  }
  // Empty observations, and a draft-verify wave without a complete verify,
  // are ignored.
  const AdaptiveWaveMode before = a;
  a.Observe(4, Mode::kSpeculative, 0, 4);
  a.Observe(4, Mode::kSpeculative, 7, 0);
  a.Observe(4, Mode::kSpeculative, 7, 5);
  a.Observe(1, Mode::kSpeculative, 3, 1);
  EXPECT_EQ(a, before);
}

// Past four requests (more request slots, D-104) a DeepSeek verify takes
// its share of a wave's sixteen rows: three a request at width 5, two at 6
// to 8 (serving.cc CutRows), against widths 5 to 8's measured costs.
constexpr AdaptiveWaveMode::Costs kWideCost = {0, 0, 1.94, 2.21, 2.90, 2.58, 2.10, 2.20, 2.25};
constexpr AdaptiveWaveMode::Rows kCutRows = {0, 0, 0, 0, 0, 3, 2, 2, 2};

TEST(AdaptiveWaveModeTest, AWidthWhoseCutVerifyCannotPayIsPlainFromTheStart) {
  AdaptiveWaveMode policy(kWideCost, AdaptiveWaveMode::Force::kNone, kCutRows);
  // Two tokens a request at most, against 2.1 to 2.25 plain waves: plain
  // before any acceptance is seen, and no probe makes it speculate.
  for (std::uint32_t w = 6; w <= 8; ++w) {
    EXPECT_EQ(policy.Choose(w), Mode::kPlain) << w;
    EXPECT_EQ(RunWaves(policy, w, 200, 1), 200U) << w;
  }
  EXPECT_EQ(policy.Choose(6, /*sampling=*/true), Mode::kSpeculative);
  const AdaptiveWaveMode spec(kWideCost, AdaptiveWaveMode::Force::kSpeculative, kCutRows);
  EXPECT_EQ(spec.Choose(8), Mode::kSpeculative);
  // Narrower widths still speculate first.
  EXPECT_EQ(policy.Choose(4), Mode::kSpeculative);
}

TEST(AdaptiveWaveModeTest, ACutVerifyCountsAgainstItsRowsAndDoesNotFeedTheAverage) {
  // Full verifies keeping four tokens a request: past width 5's cost even
  // at its three rows, so width 5 speculates.
  AdaptiveWaveMode high(kWideCost, AdaptiveWaveMode::Force::kNone, kCutRows);
  RunWaves(high, 4, 6, 4);
  EXPECT_NEAR(high.kept(), 4.0, 1e-9);
  EXPECT_EQ(high.Choose(5), Mode::kSpeculative);
  // Width 5's own waves (at most three a request) leave the average alone.
  const double before = high.kept();
  high.Observe(5, Mode::kSpeculative, 5, 5);
  EXPECT_EQ(high.kept(), before);
  // 2.5 a request: past widths 2 and 3's costs, below width 5's 2.58.
  AdaptiveWaveMode mid(kWideCost, AdaptiveWaveMode::Force::kNone, kCutRows);
  RunWaves(mid, 2, 3, 2);
  for (std::uint32_t i = 0; i < 40; ++i) {
    mid.Observe(4, Mode::kSpeculative, 10, 4);
  }
  EXPECT_EQ(mid.Choose(3), Mode::kSpeculative);
  EXPECT_EQ(mid.Choose(5), Mode::kPlain);
  // It never probes: plain throughout while narrower waves say so.
  EXPECT_EQ(RunWaves(mid, 5, 200, 3), 200U);
  // Before any full verify it runs plain (measured faster, docs/
  // experiments/request-slots), not an uninformed draft-verify wave.
  const AdaptiveWaveMode fresh(kWideCost, AdaptiveWaveMode::Force::kNone, kCutRows);
  EXPECT_EQ(fresh.Choose(5), Mode::kPlain);
  EXPECT_EQ(fresh.Choose(4), Mode::kSpeculative);
}

// A kept conversation's record carries the depth's whole state (D-105):
// loaded back it makes the same choices as the one saved, through its
// probes; words no state could have are refused.
TEST(AdaptiveDepthTest, ItsStateSavesAndLoadsWhole) {
  AdaptiveDepth depth(3);
  for (std::uint32_t i = 0; i < 45; ++i) {
    depth.Observe(depth.Choose(), (i % 3) + 1);
  }
  const auto words = depth.Save();
  const auto load = AdaptiveDepth::Load(words);
  ASSERT_TRUE(load.has_value());
  AdaptiveDepth loaded = load.value_or(AdaptiveDepth(1));
  EXPECT_EQ(loaded, depth);
  for (std::uint32_t i = 0; i < 80; ++i) {
    ASSERT_EQ(loaded.Choose(), depth.Choose()) << i;
    loaded.Observe(loaded.Choose(), (i % 4) + 1);
    depth.Observe(depth.Choose(), (i % 4) + 1);
  }
  EXPECT_FALSE(AdaptiveDepth::Load(std::span(words).first(10)).has_value());
  auto bad = words;
  bad[0] = 9;  // a depth past 7
  EXPECT_FALSE(AdaptiveDepth::Load(bad).has_value());
  bad = words;
  bad[2] = 5;  // a preferred depth that is neither
  EXPECT_FALSE(AdaptiveDepth::Load(bad).has_value());
  bad = words;
  bad[8] = 0x7ff8000000000000ULL;  // a NaN average
  EXPECT_FALSE(AdaptiveDepth::Load(bad).has_value());
  bad = words;
  bad[6] = 5;  // a probe longer than any
  EXPECT_FALSE(AdaptiveDepth::Load(bad).has_value());
  bad = words;
  bad[5] = 0;  // a probe still to run, of no depth (Choose would give 0)
  bad[6] = 2;
  EXPECT_FALSE(AdaptiveDepth::Load(bad).has_value());
}

TEST(AdaptiveWaveModeTest, OnlyCompleteVerifiesFeedTheAverage) {
  // Two of four requests verified their full rows and kept 3 each; the
  // others' cut-short verifies do not dilute the average.
  AdaptiveWaveMode policy(kCost);
  for (std::uint32_t i = 0; i < 6; ++i) {
    policy.Observe(4, Mode::kSpeculative, 6, 2);
  }
  EXPECT_NEAR(policy.kept(), 3.0, 1e-9);
  EXPECT_EQ(policy.Choose(4), Mode::kSpeculative);
}
}  // namespace
