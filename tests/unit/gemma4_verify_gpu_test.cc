// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <memory>
#include <numeric>
#include <span>
#include <tuple>
#include <vector>

#include "base/sha256.h"
#include "engine/gemma4_runner.h"
#include "engine/support.h"

namespace en = jitllm::engine;
namespace base = jitllm::base;
class Gemma4VerifyGpu : public ::testing::TestWithParam<en::Gemma4Variant> {
 protected:
  void Start(bool invariant = false) {
    const auto artifact =
        std::filesystem::path("/home/pmeenan/.local/share/jitllm/m3-artifacts") /
        (GetParam() == en::Gemma4Variant::k31B
             ? "32c92e077a6816b54aa988e2dee61a3639c958fd510ea99e25f3621f10b2aa08"
             : "4ddb360c9ce08f1e984ab304b6af918be44246d52346734066b06443f7c249d3");
    ASSERT_TRUE(std::filesystem::exists(artifact));
    en::Gemma4Options options{.artifact = artifact,
                              .out = "/tmp/jitllm-gemma4-verify-control",
                              .variant = GetParam(),
                              .context = 1536,
                              .max_rows = 16,
                              .slots = 2,
                              .max_head_rows = 4,
                              .max_verify_rows = 4,
                              .retain_features = true,
                              .shared_q8 = invariant,
                              .fuse_norms = invariant,
                              .row_invariant = invariant};
    runner = std::make_unique<en::Gemma4Runner>(node, std::move(options), 0, 0);
    ASSERT_TRUE(node.Open());
    entered.push_back(runner.get());
    // All caller output vectors and small state/range containers are funded
    // before growth. State payload diagnostics use separately cataloged pinned
    // memory, not uncharged vector copies.
    ASSERT_TRUE(runner->Setup());
    ASSERT_TRUE(node.MapWorkspace(runner->activations_needed(), runner->pool_needed()));
    const auto fixed = node.catalog().OccupancyOf(node.domain()).Total().value();
    ASSERT_TRUE(node.Start(base::Bytes(fixed + runner->weights().size() * en::kPagedExtent +
                                       2 * node.StateCapacity())));
    ASSERT_TRUE(runner->Register());
    ASSERT_TRUE(runner->Bind());
    node.Run();
    ASSERT_TRUE(node.ChargeHost(64ULL << 20, false));
    ASSERT_TRUE(runner->SelectSlots(std::array<std::uint32_t, 2>{0, 1}));
    const auto capacity = 4 * runner->layout().tensors.size();
    EXPECT_EQ((*runner->request_slot(0))->state().saved().capacity(), capacity);
  }
  void TearDown() override {
    auto retired = node.TearDown(entered);
    EXPECT_TRUE(retired) << (retired ? "" : retired.error());
    if (!retired) std::ignore = lifetime.release();
  }
  en::Status Held(const std::function<en::Status()>& body) {
    return node.WithRequest(0, runner->closure(), "Gemma4 verify control", body);
  }
  en::Status Single(std::uint32_t slot, std::uint32_t past, std::span<const std::int32_t> tokens,
                    std::vector<float>& heads) {
    const en::Gemma4Runner::Work work{slot, past, tokens, &heads};
    return runner->Wave(std::span(&work, 1));
  }
  en::Status Prefix(std::uint32_t slot, std::uint32_t rows, std::vector<float>& head) {
    std::array<std::int32_t, 16> tokens{};
    for (std::uint32_t past = 0; past < rows;) {
      const auto count = std::min(16U, rows - past);
      for (std::uint32_t i = 0; i < count; ++i) tokens[i] = past + i == 0 ? 2 : 818;
      if (auto r = Single(slot, past, std::span(tokens).first(count), head); !r) return r;
      past += count;
    }
    return {};
  }
  std::expected<base::Sha256Digest, std::string> StateHash(std::uint32_t slot,
                                                           std::uint32_t positions) {
    auto ranges = runner->CheckpointRanges(positions);
    if (!ranges) return std::unexpected(ranges.error());
    return RangeHash(slot, *ranges);
  }
  std::expected<base::Sha256Digest, std::string> RangeHash(
      std::uint32_t slot, std::span<const en::LiveState::Range> ranges) {
    std::uint64_t bytes = 0;
    for (const auto& r : ranges) bytes += r.bytes;
    std::vector<jitllm::catalog::ExtentId> staging;
    if (bytes == 0) return base::Sha256{}.Finish();
    auto pinned = node.Pinned(bytes, 0, staging);
    if (!pinned) return std::unexpected(pinned.error());
    auto copied = runner->CopyState(slot, *pinned, ranges, true);
    if (!copied) {
      node.KeepPinned(*pinned);
      return std::unexpected(copied.error());
    }
    base::Sha256 hash;
    hash.Update(std::span(static_cast<const std::byte*>(*pinned), bytes));
    const auto digest = hash.Finish();
    if (auto r = node.FreePinned(*pinned); !r) return std::unexpected(r.error());
    return digest;
  }
  en::Status Feature(std::uint32_t slot, std::uint32_t position, std::vector<float>& out) {
    const auto bytes = std::uint64_t{runner->profile().width} * sizeof(float);
    std::vector<jitllm::catalog::ExtentId> staging;
    auto pinned = node.Pinned(bytes, 0, staging);
    if (!pinned) return en::support::Error(pinned.error());
    if (auto r = runner->CopyFeatures(slot, position, 1, *pinned); !r) return r;
    const auto* values = static_cast<const float*>(*pinned);
    out.assign(values, values + runner->profile().width);
    return node.FreePinned(*pinned);
  }
  void Exact(std::span<const float> a, std::span<const float> b) {
    ASSERT_EQ(a.size(), b.size());
    EXPECT_EQ(std::memcmp(a.data(), b.data(), a.size_bytes()), 0);
    EXPECT_TRUE(std::ranges::all_of(a, [](float x) { return std::isfinite(x); }));
  }
  struct Lifetime {
    en::PagedNode node{{.slot_bytes = en::kSlabSlotBytes}};
    std::unique_ptr<en::Gemma4Runner> runner;
    std::vector<en::PagedModel*> entered;
  };
  std::unique_ptr<Lifetime> lifetime = std::make_unique<Lifetime>();
  en::PagedNode& node = lifetime->node;
  std::unique_ptr<en::Gemma4Runner>& runner = lifetime->runner;
  std::vector<en::PagedModel*>& entered = lifetime->entered;
  const std::array<std::int32_t, 4> draft{45518, 107, 101, 818};
};

TEST_P(Gemma4VerifyGpu, OrdinaryRepeatDiscardAndActualFeatureCommit) {
  Start();
  ASSERT_TRUE(runner);
  auto result = Held([&]() -> en::Status {
    std::vector<float> head, peer, feature_before, feature_after, first_heads, first_features;
    if (auto r = Prefix(0, 6, head); !r) return r;
    // An over-capacity verify refuses before allocation or state mutation.
    // Host-pressure admission is not inferred from this descriptor guard.
    const std::array<std::int32_t, 5> too_many{45518, 107, 101, 818, 529};
    std::vector<float> refused_heads{17}, refused_features{19};
    auto refused = runner->Verify(1, 0, too_many, refused_heads, refused_features);
    EXPECT_FALSE(refused);
    EXPECT_EQ(refused_heads, std::vector<float>{17});
    EXPECT_EQ(refused_features, std::vector<float>{19});
    EXPECT_EQ((*runner->request_slot(1))->completed_positions(), 0U);
    EXPECT_TRUE((*runner->request_slot(1))->state_usable());
    EXPECT_TRUE((*runner->request_slot(0))->state_usable());
    if (auto r = Prefix(1, 6, peer); !r) return r;
    if (auto r = Feature(0, 5, feature_before); !r) return r;
    if (auto r = runner->ReserveStateThrough(0, 10); !r) return r;
    auto before = StateHash(0, 10), peer_before = StateHash(1, 6);
    if (!before || !peer_before) return en::support::Error("initial state copy failed");
    for (unsigned repeat = 0; repeat < 3; ++repeat) {
      std::vector<float> heads, features;
      if (auto r = runner->Verify(0, 6, draft, heads, features); !r) return r;
      EXPECT_EQ((*runner->request_slot(0))->completed_positions(), 6U);
      EXPECT_EQ(heads.size(), 4 * runner->profile().vocab);
      EXPECT_EQ(features.size(), 4 * runner->profile().width);
      if (repeat == 0) {
        first_heads = heads;
        first_features = features;
      } else {
        Exact(heads, first_heads);
        Exact(features, first_features);
      }
      if (auto r = Feature(0, 5, feature_after); !r) return r;
      Exact(feature_after, feature_before);
      EXPECT_FALSE(runner->BorrowFrozen(0, draft[0]));
      EXPECT_FALSE(Single(0, 6, std::span(draft).first(1), head));
      EXPECT_FALSE(runner->Clear(0));
      EXPECT_FALSE(runner->Spill(0));
      EXPECT_FALSE(runner->ReserveStateThrough(0, 11));
      EXPECT_FALSE(runner->AcceptVerify(0, 0));
      EXPECT_FALSE(runner->AcceptVerify(0, 5));
      const std::array<std::uint32_t, 1> peer_only{1};
      EXPECT_FALSE(runner->SelectSlots(peer_only));
      if (repeat < 2) {
        if (auto r = runner->DiscardVerify(0); !r) return r;
        auto restored = StateHash(0, 10);
        if (!restored) return en::support::Error(restored.error());
        EXPECT_EQ(*restored, *before);
        auto peer_after = StateHash(1, 6);
        if (!peer_after) return en::support::Error(peer_after.error());
        EXPECT_EQ(*peer_before, *peer_after);
      } else {
        // A pending verify does not prevent an independent peer's next job.
        if (auto r = Single(1, 6, std::span(draft).first(1), peer); !r) return r;
        if (auto r = runner->AcceptVerify(0, 2); !r) return r;
        EXPECT_EQ((*runner->request_slot(0))->completed_positions(), 8U);
        if (auto r = Feature(0, 7, feature_after); !r) return r;
        Exact(feature_after,
              std::span(first_features).subspan(runner->profile().width, runner->profile().width));
        auto borrow = runner->BorrowFrozen(0, draft[2]);
        if (!borrow) return en::support::Error(borrow.error());
        EXPECT_FALSE(runner->Verify(0, 8, draft, heads, features));
      }
    }
    return {};
  });
  ASSERT_TRUE(result) << (result ? "" : result.error());
  EXPECT_GT(runner->graph_stats().replayed, 0U);
}

TEST_P(Gemma4VerifyGpu, AcceptedPrefixesEqualIndependentFourRowWave) {
  Start(true);
  ASSERT_TRUE(runner);
  auto result = Held([&]() -> en::Status {
    for (std::uint32_t keep = 1; keep <= 4; ++keep) {
      if (auto r = runner->Clear(0); !r) return r;
      if (auto r = runner->Clear(1); !r) return r;
      std::vector<float> head, control, heads, features, accepted, manual;
      if (auto r = Prefix(0, 6, head); !r) return r;
      if (auto r = Prefix(1, 6, control); !r) return r;
      if (auto r = runner->ReserveStateThrough(0, 10); !r) return r;
      std::vector<en::LiveState::Range> rejected;
      if (keep < 4) {
        auto writes = jitllm::model::Gemma4ChunkWrites(runner->profile(), runner->layout(),
                                                       6 + keep, 4 - keep);
        if (!writes) return en::support::Error(writes.error());
        for (const auto& r : *writes)
          rejected.push_back({.region = 0, .offset = r.offset, .bytes = r.bytes});
      }
      auto rejected_before = RangeHash(0, rejected);
      if (!rejected_before) return en::support::Error(rejected_before.error());
      if (auto r = runner->Verify(0, 6, draft, heads, features); !r) return r;
      // The independent ordinary wave has the same four-query arithmetic.
      // Scalar attention is a different policy even with invariant products.
      const en::Gemma4Runner::Work work{1, 6, draft, &control};
      if (auto r = runner->Wave(std::span(&work, 1), true, true); !r) return r;
      Exact(heads, control);
      if (auto r = runner->AcceptVerify(0, keep); !r) return r;
      EXPECT_EQ((*runner->request_slot(0))->completed_positions(), 6U + keep);
      // CheckpointRanges includes aligned future cells, which differ because
      // the manual wave keeps all four rows and acceptance restores the tail.
      auto writes =
          jitllm::model::Gemma4ChunkWrites(runner->profile(), runner->layout(), 0, 6 + keep);
      if (!writes) return en::support::Error(writes.error());
      std::vector<en::LiveState::Range> prefix;
      for (const auto& r : *writes)
        prefix.push_back({.region = 0, .offset = r.offset, .bytes = r.bytes});
      auto state = RangeHash(0, prefix), reference = RangeHash(1, prefix);
      if (!state || !reference) return en::support::Error("accepted state copy failed");
      EXPECT_EQ(*state, *reference);
      if (auto r = Feature(0, 5 + keep, accepted); !r) return r;
      if (auto r = Feature(1, 5 + keep, manual); !r) return r;
      Exact(accepted, manual);
      Exact(accepted, std::span(features).subspan((keep - 1) * runner->profile().width,
                                                  runner->profile().width));
      auto rejected_after = RangeHash(0, rejected);
      if (!rejected_after) return en::support::Error(rejected_after.error());
      EXPECT_EQ(*rejected_before, *rejected_after);
      // Project the independent four-row checkpoint to the accepted prefix;
      // this tests the continuation without changing its computed KV values.
      if (keep < 4) {
        auto ranges = runner->CheckpointRanges(6 + keep);
        if (!ranges) return en::support::Error(ranges.error());
        std::uint64_t bytes = 0;
        for (const auto& range : *ranges) bytes += range.bytes;
        std::vector<jitllm::catalog::ExtentId> staging;
        auto saved = node.Pinned(bytes, 0, staging);
        if (!saved) return en::support::Error(saved.error());
        if (auto r = runner->CopyState(1, *saved, *ranges, true); !r) return r;
        if (auto r = runner->Clear(1); !r) return r;
        if (auto r = runner->RestoreCheckpoint(1, 6 + keep, *saved, *ranges,
                                               runner->CheckpointLayoutId());
            !r)
          return r;
        if (auto r = node.FreePinned(*saved); !r) return r;
      }
      const std::array<std::int32_t, 1> next{529};
      if (auto r = Single(0, 6 + keep, next, head); !r) return r;
      if (auto r = Single(1, 6 + keep, next, control); !r) return r;
      Exact(head, control);
    }
    return {};
  });
  ASSERT_TRUE(result) << (result ? "" : result.error());
}

TEST_P(Gemma4VerifyGpu, RingWrapWholeDiscardAndCleanTailRefusal) {
  Start();
  ASSERT_TRUE(runner);
  auto result = Held([&]() -> en::Status {
    std::vector<float> head, heads, features, old_feature, restored_feature;
    // Local ring is 1280 cells. These real writes cross its wrap boundary.
    EXPECT_EQ(runner->layout().local_cells, 1280U);
    if (auto r = Prefix(0, 1279, head); !r) return r;
    if (auto r = runner->ReserveStateThrough(0, 1283); !r) return r;
    auto before = StateHash(0, 1283);
    if (!before) return en::support::Error(before.error());
    if (auto r = Feature(0, 1278, old_feature); !r) return r;
    if (auto r = runner->Verify(0, 1279, draft, heads, features); !r) return r;
    if (auto r = runner->DiscardVerify(0); !r) return r;
    auto after = StateHash(0, 1283);
    if (!after) return en::support::Error(after.error());
    EXPECT_EQ(*before, *after);
    if (auto r = Feature(0, 1278, restored_feature); !r) return r;
    Exact(old_feature, restored_feature);
    EXPECT_EQ((*runner->request_slot(0))->completed_positions(), 1279U);
    const std::array<std::int32_t, 5> too_many{2, 818, 529, 101, 107};
    EXPECT_FALSE(runner->Verify(0, 1279, too_many, heads, features));
    EXPECT_FALSE(runner->Verify(0, 1278, draft, heads, features));
    EXPECT_FALSE(runner->Verify(2, 1279, draft, heads, features));
    if (auto r = runner->Clear(1); !r) return r;
    if (auto r = Prefix(1, 1534, head); !r) return r;
    auto tail_before = StateHash(1, 1534);
    if (!tail_before) return en::support::Error(tail_before.error());
    EXPECT_FALSE(runner->Verify(1, 1534, draft, heads, features));
    auto tail_after = StateHash(1, 1534);
    if (!tail_after) return en::support::Error(tail_after.error());
    EXPECT_EQ(*tail_before, *tail_after);
    EXPECT_EQ((*runner->request_slot(1))->completed_positions(), 1534U);
    // Ordinary continuation remains explicit; no accepted scalar parity is
    // inferred from a multirow ordinary verify.
    if (auto r = Single(0, 1279, std::span(draft).first(1), head); !r) return r;
    EXPECT_EQ((*runner->request_slot(0))->completed_positions(), 1280U);
    return {};
  });
  ASSERT_TRUE(result) << (result ? "" : result.error());
}

INSTANTIATE_TEST_SUITE_P(ApprovedProfiles, Gemma4VerifyGpu,
                         ::testing::Values(en::Gemma4Variant::k26BA4B, en::Gemma4Variant::k31B));

TEST(Gemma4VerifyOptions, RefusesImplicitOrUndersizedHeadEnvelopeBeforeArtifactOpen) {
  for (const auto& options :
       {en::Gemma4Options{.max_verify_rows = 5},
        en::Gemma4Options{.max_head_rows = 4, .max_verify_rows = 4},
        en::Gemma4Options{.max_head_rows = 1, .max_verify_rows = 4, .retain_features = true},
        en::Gemma4Options{.max_verify_rows = 4, .retain_features = true}}) {
    en::PagedNode node{{.slot_bytes = en::kSlabSlotBytes}};
    en::Gemma4Runner runner(node, options, 0, 0);
    auto setup = runner.Setup();
    ASSERT_FALSE(setup);
    EXPECT_EQ(setup.error().find("verify"), 7U);
  }
  EXPECT_EQ(en::Gemma4Options{}.max_verify_rows, 0U);
}
