// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// No device/provider initialization: refusal must precede native allocation.
#include "engine/gemma4_runner.h"

#include <gtest/gtest.h>

#include <array>
#include <tuple>
#include <vector>

namespace en = jitllm::engine;
TEST(Gemma4Runner, InvalidEnvelopesRefuseBeforeOpeningProvidersOrArtifacts) {
  for (const auto [context, rows, slots, masks] :
       {std::tuple{4096U, 128U, 0U, true}, std::tuple{4096U, 128U, 17U, true},
        std::tuple{4096U, 1U, 2U, true}, std::tuple{128U, 128U, 1U, true},
        std::tuple{262145U, 128U, 1U, true}, std::tuple{4096U, 128U, 1U, false}}) {
    en::PagedNode node({});
    en::Gemma4Runner runner(node,
                            {.artifact = "/missing",
                             .context = context,
                             .max_rows = rows,
                             .slots = slots,
                             .reference_masks = masks},
                            0, 0);
    EXPECT_FALSE(runner.Setup());
    EXPECT_EQ(node.StateCapacity(), 0U);
    EXPECT_EQ(runner.activations_needed(), 0U);
  }
}
TEST(Gemma4Runner, InvalidHeadCapacityRefusesBeforeArtifactsProvidersAndReservations) {
  for (const auto variant : {en::Gemma4Variant::k26BA4B, en::Gemma4Variant::k31B}) {
    for (const auto [slots, cap] :
         {std::pair{4U, 3U}, std::pair{1U, 129U}, std::pair{4U, 1U}, std::pair{1U, UINT32_MAX}}) {
      en::PagedNode node({});
      en::Gemma4Runner runner(
          node, {.artifact = "/missing", .variant = variant, .slots = slots, .max_head_rows = cap},
          0, 0);
      const auto setup = runner.Setup();
      ASSERT_FALSE(setup);
      EXPECT_EQ(setup.error(), "Gemma4 head capacity must cover slots within max_rows");
      EXPECT_EQ(node.StateCapacity(), 0U);
      EXPECT_EQ(node.host_counted(), 0U);
      EXPECT_EQ(runner.activations_needed(), 0U);
      EXPECT_EQ(runner.pool_needed(), 0U);
      EXPECT_FALSE(runner.request_slot(0));
    }
  }
}
TEST(Gemma4Runner, HeadCapacityDefaultsToLegacyInputRows) {
  EXPECT_EQ(en::Gemma4Options{}.max_head_rows, 0U);
  // Valid caps reach the artifact check without opening native providers.
  for (const auto [slots, cap] :
       {std::pair{4U, 0U}, std::pair{4U, 4U}, std::pair{1U, 1U}, std::pair{4U, 128U}}) {
    en::PagedNode node({});
    en::Gemma4Runner runner(node, {.artifact = "/missing", .slots = slots, .max_head_rows = cap}, 0,
                            0);
    auto setup = runner.Setup();
    ASSERT_FALSE(setup);
    EXPECT_NE(setup.error(), "Gemma4 head capacity must cover slots within max_rows");
    EXPECT_EQ(node.StateCapacity(), 0U);
    EXPECT_FALSE(runner.request_slot(0));
  }
}
TEST(Gemma4Runner, UninitializedRunnerCannotDispatchSelectRestoreOrCopy) {
  en::PagedNode node({});
  en::Gemma4Runner runner(node, {}, 0, 0);
  std::vector<float> logits{17};
  const std::array<std::int32_t, 1> token{2};
  EXPECT_FALSE(runner.Chunk(0, token, logits));
  EXPECT_EQ(logits, std::vector<float>{17});
  EXPECT_FALSE(runner.request_slot(0));
  EXPECT_FALSE(runner.request_slot(UINT32_MAX));
  EXPECT_FALSE(runner.SelectSlots(std::array<std::uint32_t, 1>{0}));
  EXPECT_FALSE(runner.Register());
  EXPECT_FALSE(runner.Bind());
  EXPECT_FALSE(runner.CopyState(0, nullptr, {}, true));
  EXPECT_FALSE(runner.RestoreCheckpoint(0, 1, nullptr, {}, ""));
}
TEST(Gemma4Runner, UnreadableArtifactRefusesBeforeStateReservations) {
  for (auto variant : {en::Gemma4Variant::k26BA4B, en::Gemma4Variant::k31B}) {
    en::PagedNode node({});
    en::Gemma4Runner runner(node, {.artifact = "/missing-gemma4-artifact", .variant = variant}, 0,
                            0);
    EXPECT_FALSE(runner.Setup());
    EXPECT_EQ(runner.profile().layers, variant == en::Gemma4Variant::k31B ? 60U : 30U);
    EXPECT_EQ(node.StateCapacity(), 0U);
    EXPECT_FALSE(runner.request_slot(0));
  }
}
TEST(Gemma4Runner, UnknownVariantRefusesBeforeOpeningArtifactsOrAllocatingState) {
  en::PagedNode node({});
  en::Gemma4Runner runner(
      node, {.artifact = "/missing", .variant = static_cast<en::Gemma4Variant>(UINT8_MAX)}, 0, 0);
  auto setup = runner.Setup();
  ASSERT_FALSE(setup);
  EXPECT_EQ(setup.error(), "Gemma4 runner variant is not approved");
  EXPECT_EQ(node.StateCapacity(), 0U);
  EXPECT_EQ(runner.activations_needed(), 0U);
  EXPECT_FALSE(runner.request_slot(0));
}
TEST(Gemma4Runner, ActualLayerFourteenShardCutPreservesEveryReadableMember) {
  // Native prepared artifact: stored3719168, cut67/128, Q5_1 down at0,
  // Q4_K gate/up at1487104. Minimal144 pitch fails the256-aligned delta.
  constexpr std::uint64_t stored = 3719168;
  const auto pitch = en::Gemma4ExpertPitch(stored, std::array<std::string_view, 2>{"Q5_1", "Q4_K"});
  ASSERT_TRUE(pitch);
  EXPECT_EQ(*pitch % 256, 0U);
  EXPECT_EQ(*pitch % 24, 0U);
  EXPECT_EQ(*pitch % 144, 0U);
  std::vector<std::uint32_t> shard(128);
  std::vector<std::uint64_t> file(128);
  for (std::size_t i = 0; i < file.size(); ++i) {
    shard[i] = i < 67 ? 1 : 2;
    file[i] = 4096 + (i < 67 ? i : i - 67) * stored;
  }
  EXPECT_FALSE(en::LayOutSlab(shard, file, stored, ((stored + 143) / 144) * 144, 256));
  auto layout = en::LayOutSlab(shard, file, stored, *pitch, 256);
  ASSERT_TRUE(layout);
  std::uint64_t copied = 0;
  for (std::size_t page = 0; page < layout->pages.size(); ++page) {
    const auto& p = layout->pages[page];
    for (std::size_t piece = 0; piece < p.pieces; ++piece) {
      copied += p.bytes[piece];
      const auto relative = page * en::kPagedExtent + p.page_offset[piece] - layout->delta;
      const auto expert = relative / *pitch;
      ASSERT_LT(expert, 128U);
      EXPECT_EQ(p.shard, shard[expert]);
      EXPECT_EQ(p.file_offset + p.slot_offset[piece], file[expert] + relative % *pitch);
      EXPECT_LE(relative % *pitch + p.bytes[piece], stored);
    }
  }
  EXPECT_EQ(copied, 128 * stored);
  EXPECT_FALSE(en::Gemma4ExpertPitch(UINT64_MAX, std::array<std::string_view, 1>{"Q4_K"}));
  EXPECT_FALSE(en::Gemma4ExpertPitch(stored, std::array<std::string_view, 1>{"unknown"}));
}
TEST(Gemma4Runner, CheckpointLedgerRequiresEveryFundedExtentWithoutInferringPosition) {
  for (const auto& profile : {jitllm::model::Gemma4_26BA4B(), jitllm::model::Gemma4_31B()}) {
    auto layout = jitllm::model::Gemma4State(profile, 4096, 16);
    ASSERT_TRUE(layout);
    auto needed = jitllm::model::Gemma4UsedState(profile, *layout, 6);
    ASSERT_TRUE(needed);
    std::vector<en::LiveState::Range> footprint;
    for (const auto& r : *needed)
      for (auto e = r.offset / en::kPagedExtent; e <= (r.offset + r.bytes - 1) / en::kPagedExtent;
           ++e)
        footprint.push_back({0, e * en::kPagedExtent,
                             std::min(en::kPagedExtent, layout->bytes - e * en::kPagedExtent)});
    EXPECT_TRUE(en::Gemma4CheckpointFootprint(profile, *layout, 6, footprint));
    // The same initialized pages can fund several positions. Only owned
    // logical metadata selects which one was actually computed.
    EXPECT_TRUE(en::Gemma4CheckpointFootprint(profile, *layout, 5, footprint));
    EXPECT_FALSE(en::Gemma4CheckpointFootprint(profile, *layout, 4097, footprint));
    auto bad = footprint;
    bad.pop_back();
    EXPECT_FALSE(en::Gemma4CheckpointFootprint(profile, *layout, 6, bad));
    bad = footprint;
    ++bad[0].offset;
    EXPECT_FALSE(en::Gemma4CheckpointFootprint(profile, *layout, 6, bad));
    bad = footprint;
    --bad[0].bytes;
    EXPECT_FALSE(en::Gemma4CheckpointFootprint(profile, *layout, 6, bad));
    bad = footprint;
    bad[0].region = 1;
    EXPECT_FALSE(en::Gemma4CheckpointFootprint(profile, *layout, 6, bad));
    bad = footprint;
    bad.push_back(bad.back());
    EXPECT_FALSE(en::Gemma4CheckpointFootprint(profile, *layout, 6, bad));
    EXPECT_TRUE(en::Gemma4CheckpointFootprint(profile, *layout, 0, {}));
    EXPECT_FALSE(en::Gemma4CheckpointFootprint(profile, *layout, 0, footprint));
  }
}
