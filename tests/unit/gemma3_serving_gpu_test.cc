// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <expected>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "base/bytes.h"
#include "base/sha256.h"
#include "engine/gemma3_runner.h"
#include "engine/support.h"
#include "scheduler/scheduler.h"

namespace en = jitllm::engine;
class Gemma3ServingGpu : public ::testing::Test {
 protected:
  struct PreparationObserver final : jitllm::scheduler::PageInObserver {
    jitllm::catalog::Catalog* catalog = nullptr;
    jitllm::catalog::ExtentId target;
    std::uint64_t generation = 0;
    jitllm::catalog::RegistrationId registration;
    bool held = false;
    void Staged(jitllm::catalog::ExtentId extent, jitllm::scheduler::PageInEvent event) override {
      if (held || extent != target || event != jitllm::scheduler::PageInEvent::kResident) return;
      const auto view = catalog->Describe(extent);
      if (view && view->content_generation == generation && !view->discarded &&
          view->descriptor.memory_class == jitllm::catalog::MemoryClass::kLiveState) {
        if (auto acquired = catalog->AddRegistration(extent); acquired) {
          registration = *acquired;
          held = true;
        }
      }
    }
  };
  virtual bool PrepareState() const { return false; }
  struct Lifetime {
    Lifetime() : node({.slot_bytes = en::kSlabSlotBytes, .observer = &observer}) {}
    PreparationObserver observer;
    en::PagedNode node;
    std::unique_ptr<en::Gemma3Runner> runner;
    std::vector<en::PagedModel*> entered;
  };
  std::unique_ptr<Lifetime> life = std::make_unique<Lifetime>();
  PreparationObserver& observer = life->observer;
  en::PagedNode& node = life->node;
  std::unique_ptr<en::Gemma3Runner>& runner = life->runner;
  const std::array<std::int32_t, 6> prompt{2, 818, 5279, 529, 7001, 563};
  void SetUp() override {
    runner = std::make_unique<en::Gemma3Runner>(
        node,
        en::Gemma3Options{.artifact =
                              "/home/pmeenan/.local/share/jitllm/gemma3-import-20261007/artifacts/"
                              "8c7103418a6608022e5eda50a0dcc4b7688a0d59ef239813c9de0984161397fb",
                          .out = "/tmp/jitllm-gemma3-serving-control",
                          .slots = 2,
                          .max_wave_rows = 256,
                          .max_head_rows = 2,
                          .owner_decode = true,
                          .packed_prefill = true,
                          .fuse_norms = true,
                          .fuse_quant_glu = true,
                          .fuse_norm_rope = true,
                          .fuse_norm_add = true,
                          .prepare_state = PrepareState()},
        0, 0);
    ASSERT_TRUE(node.Open());
    observer.catalog = &node.catalog();
    life->entered.push_back(runner.get());
    ASSERT_TRUE(runner->Setup());
    ASSERT_TRUE(node.MapWorkspace(runner->activations_needed(), runner->pool_needed()));
    const auto fixed = node.catalog().OccupancyOf(node.domain()).Total().value();
    node.SetHostFloor(runner->host_input_bytes() + runner->plan_floor_bytes() + (16ULL << 20U));
    ASSERT_TRUE(node.Start(jitllm::base::Bytes(fixed + runner->weights().size() * en::kPagedExtent +
                                               4 * node.StateCapacity())));
    ASSERT_TRUE(runner->Register());
    ASSERT_TRUE(runner->Bind());
    node.Run();
    ASSERT_TRUE(runner->SelectSlots(std::array<std::uint32_t, 2>{0, 1}));
  }
  void TearDown() override {
    const auto retired = node.TearDown(life->entered);
    EXPECT_TRUE(retired) << (retired ? "" : retired.error());
    if (!retired) std::ignore = life.release();
  }
  en::Status Single(std::uint32_t slot, std::uint32_t past, std::span<const std::int32_t> ids,
                    std::vector<float>& head) {
    const en::Gemma3Runner::Work work{slot, past, ids, &head};
    return runner->Wave(std::span(&work, 1));
  }
  en::Status Held(const std::function<en::Status()>& body) {
    return node.WithRequest(0, runner->closure(), "Gemma3 serving lifecycle", body);
  }
  std::expected<jitllm::base::Sha256Digest, std::string> StateHash(std::uint32_t id) {
    const auto slot = runner->request_slot(id);
    if (!slot) return en::support::Error(slot.error());
    std::vector<jitllm::catalog::ExtentId> staging;
    auto buffer = node.Pinned(1U << 20U, 0, staging);
    if (!buffer) return en::support::Error(buffer.error());
    jitllm::base::Sha256 hash;
    for (const auto& range : (*slot)->state().used_ranges()) {
      for (std::uint64_t at = 0; at < range.bytes; at += 1U << 20U) {
        const en::LiveState::Range part{range.region, range.offset + at,
                                        std::min<std::uint64_t>(1U << 20U, range.bytes - at)};
        en::LiveState::CopyRetirement retirement = en::LiveState::CopyRetirement::kUnproven;
        const auto copied = runner->CopyState(id, *buffer, std::span(&part, 1), &retirement);
        if (retirement == en::LiveState::CopyRetirement::kUnproven) node.KeepPinned(*buffer);
        if (!copied) {
          if (retirement == en::LiveState::CopyRetirement::kProven) (void)node.FreePinned(*buffer);
          return en::support::Error(copied.error());
        }
        hash.Update(std::span(static_cast<const std::byte*>(*buffer), part.bytes));
      }
    }
    if (auto released = node.FreePinned(*buffer); !released)
      return en::support::Error(released.error());
    return hash.Finish();
  }
  void Exact(std::span<const float> a, std::span<const float> b) {
    ASSERT_EQ(a.size(), b.size());
    EXPECT_EQ(std::memcmp(a.data(), b.data(), a.size_bytes()), 0);
  }
};
TEST_F(Gemma3ServingGpu, ServingRestoreNeedsProvenCompleteCopiesAndProtectsThePeer) {
  const auto source_layout = runner->CheckpointLayoutId();
  ASSERT_TRUE(runner->SelectSlots(std::array<std::uint32_t, 2>{0, 1}));
  auto r = Held([&]() -> en::Status {
    std::vector<float> baseline, peer;
    if (auto x = Single(0, 0, std::span(prompt).first(5), baseline); !x) return x;
    if (auto x = Single(0, 5, std::span(prompt).last(1), baseline); !x) return x;
    if (auto x = Single(1, 0, std::span(prompt).first(5), peer); !x) return x;
    if (auto x = Single(1, 5, std::span(prompt).last(1), peer); !x) return x;
    auto slot = runner->request_slot(0);
    if (!slot) return en::support::Error(slot.error());
    const auto footprint = (*slot)->state().used_ranges();
    const auto saved_bytes = (*slot)->used_state_bytes();
    std::vector<jitllm::catalog::ExtentId> staging;
    auto pinned = node.Pinned(saved_bytes, 0, staging);
    if (!pinned) return en::support::Error(pinned.error());
    auto exercised = [&]() -> en::Status {
      if (auto x = runner->CopyState(0, *pinned, footprint, true); !x) return x;
      const auto occupancy = node.catalog().OccupancyOf(node.domain()).Total().value();
      EXPECT_FALSE(runner->PrepareRestore(0, 5, footprint, "gemma3-wrong-generation"));
      EXPECT_FALSE(runner->Adopt(0, 5, footprint, source_layout));  // Held, nonempty.
      EXPECT_EQ(node.catalog().OccupancyOf(node.domain()).Total().value(), occupancy);
      auto invalid = footprint;
      ++invalid[0].offset;
      EXPECT_FALSE(runner->PrepareRestore(0, 5, invalid, source_layout));
      EXPECT_EQ((*slot)->completed_positions(), 6U);
      // A capacity refusal before any copy preserves the completed ledger
      // and the peer's held state, even when the target needs more extents.
      std::vector<en::LiveState::Range> larger;
      for (std::uint64_t offset = 0; offset < (*slot)->state().bytes(0); offset += en::kPagedExtent)
        larger.push_back(
            {0, offset, std::min(en::kPagedExtent, (*slot)->state().bytes(0) - offset)});
      const auto available = node.FreeBytes();
      if (!available || *available <= (16U << 20U)) return en::support::Error("test capacity");
      const auto pressure = *available - (16U << 20U);
      if (!node.ChargeHost(pressure, false)) return en::support::Error("test pressure");
      const auto refused_prepare = runner->PrepareRestore(0, 1025, larger, source_layout);
      node.UnchargeHost(pressure);
      EXPECT_FALSE(refused_prepare);
      EXPECT_TRUE((*slot)->state_usable());
      EXPECT_TRUE((*slot)->refused_state_growth());
      EXPECT_EQ((*slot)->completed_positions(), 6U);
      EXPECT_EQ((*runner->request_slot(1))->completed_positions(), 6U);
      // Compare the identical whole-extent packing, including its padding.
      // Logical CheckpointRanges use a different packing and cannot be
      // compared directly with this snapshot's complete extent footprint.
      auto preserved = node.Pinned(saved_bytes, 0, staging);
      if (!preserved) return en::support::Error(preserved.error());
      const auto copied = runner->CopyState(0, *preserved, footprint, true);
      if (copied) EXPECT_EQ(std::memcmp(*pinned, *preserved, saved_bytes), 0);
      const auto released = node.FreePinned(*preserved);
      if (!copied) return copied;
      if (!released) return released;
      // The saved bytes are still the completed prefix, not a partial restore.
      if (auto x = runner->PrepareRestore(0, 5, footprint, source_layout); !x) return x;
      EXPECT_TRUE(runner->SelectSlots(std::array<std::uint32_t, 2>{0, 1}));
      EXPECT_FALSE(runner->CompleteRestore(0, 5));
      EXPECT_FALSE(runner->CompleteRestore(0, 6));
      EXPECT_EQ((*slot)->completed_positions(), 6U);
      EXPECT_FALSE((*slot)->state_usable());
      std::vector<float> refused{17};
      EXPECT_FALSE(Single(0, 6, std::span(prompt).first(1), refused));
      EXPECT_FALSE(runner->ReserveStateThrough(0, 7));
      EXPECT_FALSE(runner->Spill(0));
      EXPECT_FALSE(runner->CopyState(0, *pinned, footprint, true));
      en::LiveState::CopyRetirement retirement = en::LiveState::CopyRetirement::kUnproven;
      EXPECT_FALSE(runner->CopyState(0, nullptr, footprint, false, &retirement));
      EXPECT_EQ(retirement, en::LiveState::CopyRetirement::kProven);
      EXPECT_FALSE(runner->CompleteRestore(0, 5));
      EXPECT_EQ(refused, std::vector<float>{17});
      // A separately retired later extent cannot stand in for the first one.
      const auto& last = footprint.back();
      if (auto x = runner->CopyState(0, static_cast<std::byte*>(*pinned) + saved_bytes - last.bytes,
                                     std::span(footprint).last(1), false);
          !x)
        return x;
      EXPECT_FALSE(runner->CompleteRestore(0, 5));
      if (auto x = runner->CopyState(0, *pinned, std::span(footprint).first(1), false); !x)
        return x;
      EXPECT_FALSE(runner->CompleteRestore(0, 5));
      // A full, independently retired snapshot copy funds all logical rows.
      if (auto x = runner->CopyState(0, *pinned, footprint, false); !x) return x;
      if (auto x = runner->CompleteRestore(0, 5); !x) return x;
      EXPECT_EQ((*slot)->completed_positions(), 5U);
      EXPECT_TRUE((*slot)->state_usable());
      std::vector<float> restored;
      if (auto x = Single(0, 5, std::span(prompt).last(1), restored); !x) return x;
      Exact(baseline, restored);
      EXPECT_EQ((*runner->request_slot(1))->completed_positions(), 6U);
      const std::array<std::int32_t, 1> anchor{563};
      std::vector<float> a, b;
      if (auto x = Single(0, 6, anchor, a); !x) return x;
      if (auto x = Single(1, 6, anchor, b); !x) return x;
      Exact(a, b);
      return {};
    }();
    if (!exercised) {
      auto freed = node.FreePinned(*pinned);
      return freed ? exercised : freed;
    }
    return node.FreePinned(*pinned);
  });
  ASSERT_TRUE(r) << (r ? "" : r.error());
}

TEST_F(Gemma3ServingGpu, PendingRestoreCanBeCancelledAndCannotPublishPeerOrStaleCursors) {
  const auto status = Held([&]() -> en::Status {
    std::vector<float> a, b;
    if (auto x = Single(0, 0, prompt, a); !x) return x;
    if (auto x = Single(1, 0, prompt, b); !x) return x;
    const auto footprint = (*runner->request_slot(0))->state().used_ranges();
    if (auto x = runner->PrepareRestore(0, 5, footprint, runner->CheckpointLayoutId()); !x)
      return x;
    EXPECT_FALSE(runner->PrepareRestore(0, 6, footprint, runner->CheckpointLayoutId()));
    EXPECT_FALSE((*runner->request_slot(0))->state_usable());
    EXPECT_TRUE((*runner->request_slot(1))->state_usable());
    if (auto x = runner->Clear(0); !x) return x;
    EXPECT_FALSE(runner->CompleteRestore(0, 5));
    EXPECT_TRUE((*runner->request_slot(0))->state_usable());
    EXPECT_EQ((*runner->request_slot(0))->completed_positions(), 0U);
    EXPECT_EQ((*runner->request_slot(1))->completed_positions(), 6U);
    if (auto x = Single(0, 0, prompt, a); !x) return x;
    Exact(a, b);
    const std::array<std::int32_t, 1> anchor{563};
    if (auto x = Single(0, 6, anchor, a); !x) return x;
    if (auto x = Single(1, 6, anchor, b); !x) return x;
    Exact(a, b);
    return {};
  });
  ASSERT_TRUE(status) << (status ? "" : status.error());
}

TEST_F(Gemma3ServingGpu, HeldSelectionPreservesAdmissionAndRefreshesChangedOwners) {
  const std::array<std::uint32_t, 2> both{0, 1};
  // Outside a held request selection must still refresh the closure.
  ASSERT_TRUE(runner->SelectSlots(both));
  ASSERT_FALSE(node.InRequest(0));
  const auto status = Held([&]() -> en::Status {
    std::vector<float> a, b;
    if (auto x = Single(0, 0, prompt, a); !x) return x;
    if (auto x = Single(1, 0, prompt, b); !x) return x;
    const auto occupancy = node.catalog().OccupancyOf(node.domain()).Total().value();
    for (int i = 0; i < 32; ++i) {
      if (auto x = runner->SelectSlots(both); !x) return x;
    }
    EXPECT_FALSE(runner->SelectSlots(std::array<std::uint32_t, 2>{0, 0}));
    EXPECT_FALSE(runner->SelectSlots(std::array<std::uint32_t, 2>{0, 2}));
    EXPECT_EQ(node.catalog().OccupancyOf(node.domain()).Total().value(), occupancy);
    EXPECT_EQ((*runner->request_slot(0))->completed_positions(), 6U);
    EXPECT_EQ((*runner->request_slot(1))->completed_positions(), 6U);
    if (auto x = runner->SelectSlots(std::array<std::uint32_t, 1>{1}); !x) return x;
    std::vector<float> refused{17};
    EXPECT_FALSE(Single(0, 6, std::span(prompt).last(1), refused));
    EXPECT_EQ(refused, std::vector<float>{17});
    if (auto x = runner->SelectSlots(both); !x) return x;
    if (auto x = Single(0, 6, std::span(prompt).last(1), a); !x) return x;
    if (auto x = Single(1, 6, std::span(prompt).last(1), b); !x) return x;
    Exact(a, b);
    return {};
  });
  ASSERT_TRUE(status) << (status ? "" : status.error());
  ASSERT_TRUE(runner->SelectSlots(both));
}

TEST_F(Gemma3ServingGpu, JointPrefillFundsTwoFullChunksAndReplaysFreshIndependentInputs) {
  EXPECT_EQ(runner->CheckpointLayoutId(), "gemma3-4b-f16-kv-device-v1:4096:128:4096:1280");
  std::array<std::vector<float>, 2> saved_heads[2];
  std::array<jitllm::base::Sha256Digest, 2> saved_states[2];
  const auto status = Held([&]() -> en::Status {
    for (std::uint32_t pass = 0; pass < 4; ++pass) {
      for (std::uint32_t id = 0; id < 2; ++id)
        if (auto cleared = runner->Clear(id); !cleared) return cleared;
      if (auto selected = runner->SelectSlots(std::array<std::uint32_t, 2>{0, 1}); !selected)
        return selected;
      std::array<std::vector<std::int32_t>, 2> tokens;
      for (std::uint32_t id = 0; id < 2; ++id) {
        tokens[id].resize(128);
        for (std::size_t row = 0; row < 128; ++row)
          tokens[id][row] = prompt[(row + id + pass % 2) % prompt.size()];
      }
      std::array<std::vector<float>, 2> heads;
      std::array<en::Gemma3Runner::Work, 2> work{
          {{0, 0, tokens[0], &heads[0]}, {1, 0, tokens[1], &heads[1]}}};
      if (auto ran = runner->WavePrefill(work, false); !ran) return ran;
      EXPECT_TRUE(heads[0].empty());
      EXPECT_TRUE(heads[1].empty());
      for (auto& owner : work) owner.n_past = 128;
      if (auto ran = runner->WavePrefill(work, true); !ran) return ran;
      for (std::uint32_t past = 256; past < 259; ++past) {
        for (std::uint32_t id = 0; id < 2; ++id)
          work[id] = {id, past, std::span(tokens[id]).last(1), &heads[id]};
        if (auto ran = runner->Wave(work); !ran) return ran;
      }
      std::array<jitllm::base::Sha256Digest, 2> states;
      for (std::uint32_t id = 0; id < 2; ++id) {
        EXPECT_EQ((*runner->request_slot(id))->completed_positions(), 259U);
        auto hash = StateHash(id);
        if (!hash) return en::support::Error(hash.error());
        states[id] = *hash;
      }
      if (pass < 2) {
        saved_heads[pass] = heads;
        saved_states[pass] = states;
      } else {
        for (std::uint32_t id = 0; id < 2; ++id) Exact(heads[id], saved_heads[pass % 2][id]);
        EXPECT_EQ(states, saved_states[pass % 2]);
      }
      work[0].n_past = work[1].n_past = 259;
      work[1].logits = &heads[0];
      EXPECT_FALSE(runner->WavePrefill(work, true));
      work[1].logits = &heads[1];
      EXPECT_FALSE(runner->WavePrefill(work, false));  // one-row checkpoint chunks stay scalar
      work[0].tokens = tokens[0];
      work[1].tokens = std::span(tokens[1]).first(64);
      EXPECT_FALSE(runner->WavePrefill(work, true));  // different final row counts
      const std::vector<std::int32_t> oversized(129, 2);
      work[0].tokens = oversized;
      EXPECT_FALSE(runner->WavePrefill(work, true));
      for (std::uint32_t id = 0; id < 2; ++id) {
        EXPECT_EQ((*runner->request_slot(id))->completed_positions(), 259U);
        auto hash = StateHash(id);
        if (!hash) return en::support::Error(hash.error());
        EXPECT_EQ(*hash, states[id]);
      }
    }
    EXPECT_NE(saved_states[0], saved_states[1]);
    // Real initialized prefixes now straddle different padded read buckets.
    // A failed joint admission must preserve both slots and output rows.
    const std::vector<std::int32_t> continuation(128, 2);
    std::array<std::vector<float>, 2> heads;
    if (auto ran = Single(1, 259, continuation, heads[1]); !ran) return ran;
    if (auto ran = Single(1, 387, continuation, heads[1]); !ran) return ran;
    std::array<jitllm::base::Sha256Digest, 2> before;
    for (std::uint32_t id = 0; id < 2; ++id) {
      auto hash = StateHash(id);
      if (!hash) return en::support::Error(hash.error());
      before[id] = *hash;
    }
    const auto head_before = heads;
    const std::array<en::Gemma3Runner::Work, 2> unequal{
        {{0, 259, continuation, &heads[0]}, {1, 515, continuation, &heads[1]}}};
    EXPECT_FALSE(runner->WavePrefill(unequal, true));
    EXPECT_EQ(heads, head_before);
    EXPECT_EQ((*runner->request_slot(0))->completed_positions(), 259U);
    EXPECT_EQ((*runner->request_slot(1))->completed_positions(), 515U);
    for (std::uint32_t id = 0; id < 2; ++id) {
      auto hash = StateHash(id);
      if (!hash) return en::support::Error(hash.error());
      EXPECT_EQ(*hash, before[id]);
    }
    EXPECT_GT(runner->plan_selections().owner_prefill_attention, 0U);
    EXPECT_GT(runner->graph_stats().replayed, 0U);
    return {};
  });
  ASSERT_TRUE(status) << (status ? "" : status.error());
}
TEST_F(Gemma3ServingGpu, PrefillHintsBuildAndCaptureAheadWithoutChangingStateOrHeads) {
  // Six 128-row chunks: read widths 256, 256, 512, 512, 768, 768, the last
  // with a head. Unhinted, hinted, wrongly hinted and warm runs must leave
  // byte-identical state and heads; correct hints replay every chunk after
  // the first (the first run of each later width included).
  constexpr std::uint32_t kRows = 128, kChunks = 6;
  std::vector<std::int32_t> tokens(kRows * kChunks);
  for (std::size_t i = 0; i < tokens.size(); ++i)
    tokens[i] = prompt[(i * 7 + i / 5) % prompt.size()];
  const auto status = Held([&]() -> en::Status {
    std::vector<float> reference_head;
    jitllm::base::Sha256Digest reference_state{};
    // 0: no hints; 1: hints; 2: hints whose second chunk is wrong; 3: hints, warm.
    for (std::uint32_t pass = 0; pass < 4; ++pass) {
      if (auto cleared = runner->Clear(0); !cleared) return cleared;
      if (pass < 3) runner->DropPlans();
      const auto before = runner->graph_stats();
      const auto ahead_before = runner->lookahead_stats();
      std::vector<float> head;
      for (std::uint32_t chunk = 0; chunk < kChunks; ++chunk) {
        const en::Gemma3Runner::Work work{0, chunk * kRows,
                                          std::span(tokens).subspan(chunk * kRows, kRows), &head};
        const bool last = chunk + 1 == kChunks;
        en::Gemma3Runner::PrefillNext hint{0, last ? 0U : kRows, chunk + 2 < kChunks ? kRows : 0U};
        if (pass == 2 && hint.after != 0) hint.after = 64;
        const bool hinted = pass != 0 && !last;
        if (auto ran =
                runner->WavePrefill(std::span(&work, 1), last, std::span(&hint, hinted ? 1U : 0U),
                                    chunk + 2 == kChunks, chunk + 3 == kChunks);
            !ran)
          return ran;
        EXPECT_EQ(head.empty(), !last);
      }
      auto state = StateHash(0);
      if (!state) return en::support::Error(state.error());
      if (pass == 0) {
        reference_head = head;
        reference_state = *state;
      } else {
        Exact(head, reference_head);
        EXPECT_EQ(*state, reference_state) << pass;
      }
      const auto& after = runner->graph_stats();
      const auto& ahead = runner->lookahead_stats();
      const auto replayed = after.replayed - before.replayed;
      if (pass == 1) {
        EXPECT_EQ(replayed, kChunks - 1);
        EXPECT_EQ(ahead.captured_first - ahead_before.captured_first, 1U);
        EXPECT_EQ(ahead.captured_ahead - ahead_before.captured_ahead, 3U);
        EXPECT_GT(ahead.cached, ahead_before.cached);
        EXPECT_EQ(ahead.built, ahead.cached);
      }
      if (pass == 3) EXPECT_EQ(replayed, kChunks);
      EXPECT_EQ(ahead.dropped_ahead, 0U);
      EXPECT_EQ(after.refused, 0U);
    }
    return {};
  });
  ASSERT_TRUE(status) << (status ? "" : status.error());
}

TEST_F(Gemma3ServingGpu, SuppressedNextHeadsStillCaptureTheCorrectAfterOwnerPosition) {
  constexpr std::uint32_t kRows = 128;
  std::array<std::vector<std::int32_t>, 2> tokens;
  for (std::size_t id = 0; id < 2; ++id) {
    tokens[id].resize(id == 0 ? 256 : 384);
    for (std::size_t i = 0; i < tokens[id].size(); ++i)
      tokens[id][i] = prompt[(i + id * 3 + i / 7) % prompt.size()];
  }
  const auto status = Held([&]() -> en::Status {
    std::array<std::vector<float>, 2> expected;
    std::array<jitllm::base::Sha256Digest, 2> expected_state{};
    for (const bool hinted : {false, true}) {
      for (std::uint32_t id = 0; id < 2; ++id)
        if (auto r = runner->Clear(id); !r) return r;
      runner->DropPlans();
      std::array<std::vector<float>, 2> heads;
      const std::array<en::Gemma3Runner::Work, 2> initial{
          {{0, 0, std::span(tokens[0]).first(kRows), &heads[0]},
           {1, 0, std::span(tokens[1]).first(kRows), &heads[1]}}};
      const std::array<en::Gemma3Runner::PrefillNext, 2> hints{{{0, kRows, 0}, {1, kRows, kRows}}};
      const auto before = runner->lookahead_stats();
      // Next heads disagree: owner0 finishes, owner1 does not. The after
      // cohort contains only owner1, at256 rather than128, and wants a head.
      if (auto r = runner->WavePrefill(initial, false, std::span(hints).first(hinted ? 2U : 0U),
                                       std::nullopt, true);
          !r)
        return r;
      EXPECT_TRUE(heads[0].empty());
      EXPECT_TRUE(heads[1].empty());
      if (hinted) EXPECT_EQ(runner->lookahead_stats().cached - before.cached, 1U);
      const en::Gemma3Runner::Work finished{0, kRows, std::span(tokens[0]).last(kRows), &heads[0]};
      if (auto r = runner->WavePrefill(std::span(&finished, 1), true); !r) return r;
      const en::Gemma3Runner::Work middle{1, kRows, std::span(tokens[1]).subspan(kRows, kRows),
                                          &heads[1]};
      const en::Gemma3Runner::PrefillNext next{1, kRows, 0};
      if (auto r = runner->WavePrefill(std::span(&middle, 1), false,
                                       std::span(&next, hinted ? 1U : 0U), true);
          !r)
        return r;
      if (hinted) EXPECT_EQ(runner->lookahead_stats().captured_ahead - before.captured_ahead, 1U);
      const auto replayed = runner->graph_stats().replayed;
      const en::Gemma3Runner::Work last{1, 2 * kRows, std::span(tokens[1]).last(kRows), &heads[1]};
      if (auto r = runner->WavePrefill(std::span(&last, 1), true); !r) return r;
      if (hinted) EXPECT_EQ(runner->graph_stats().replayed - replayed, 1U);
      EXPECT_EQ((*runner->request_slot(0))->completed_positions(), 256U);
      EXPECT_EQ((*runner->request_slot(1))->completed_positions(), 384U);
      for (std::uint32_t id = 0; id < 2; ++id) {
        auto state = StateHash(id);
        if (!state) return en::support::Error(state.error());
        if (!hinted) {
          expected[id] = heads[id];
          expected_state[id] = *state;
        } else {
          Exact(heads[id], expected[id]);
          EXPECT_EQ(*state, expected_state[id]);
        }
      }
      EXPECT_EQ(runner->lookahead_stats().dropped_ahead, 0U);
      EXPECT_EQ(runner->coverage().violations, 0U);
    }
    return {};
  });
  ASSERT_TRUE(status) << (status ? "" : status.error());
}

class Gemma3PreparationGpu : public Gemma3ServingGpu {
 protected:
  bool PrepareState() const override { return true; }
};

TEST_F(Gemma3PreparationGpu, SuccessfulWrappedWaveWithHeldFutureQuarantinesEveryCurrentOwner) {
  constexpr std::uint32_t kRows = 128, kPast = 1920;
  ASSERT_LT(runner->layout().local_cells, kPast);
  const auto result = Held([&]() -> en::Status {
    std::array<std::vector<std::int32_t>, 2> tokens;
    std::array<std::vector<float>, 2> heads;
    for (std::uint32_t id = 0; id < 2; ++id) {
      tokens[id].resize(kRows);
      for (std::size_t i = 0; i < kRows; ++i) tokens[id][i] = prompt[(i + id) % prompt.size()];
    }
    std::array<en::Gemma3Runner::Work, 2> work{
        {{0, 0, tokens[0], &heads[0]}, {1, 0, tokens[1], &heads[1]}}};
    // Establish a real wrapped prefix without predictions. The tested unit
    // writes the old ring while preparing a disjoint new global-cache page.
    for (std::uint32_t past = 0; past < kPast; past += kRows) {
      for (auto& w : work) w.n_past = past;
      if (auto ran = runner->WavePrefill(work, false); !ran) return ran;
    }
    auto request = runner->request_slot(0);
    if (!request) return en::support::Error(request.error());
    const auto& tensors = runner->layout().tensors;
    const auto global = std::ranges::find_if(tensors, [](const auto& t) { return !t.local; });
    if (global == tensors.end()) return en::support::Error("test global cache");
    const auto index = (global->offset + 2 * en::kPagedExtent) / en::kPagedExtent;
    const auto future = (*request)->state().reserved_extents()[index];
    if (auto armed = node.Call(
            [&]() -> en::Status {
              const auto view = node.catalog().Describe(future);
              if (!view || view->state != jitllm::catalog::ExtentState::kNonresident)
                return en::support::Error("test future must be unpublished");
              observer.target = future;
              observer.generation = view->content_generation;
              return {};
            },
            "hold one completed future preparation");
        !armed)
      return armed;
    (void)node.TakeTimes(0);
    for (auto& w : work) w.n_past = kPast;
    const std::array next = {en::Gemma3Runner::PrefillNext{0, kRows, 0},
                             en::Gemma3Runner::PrefillNext{1, kRows, 0}};
    const auto failed = runner->WavePrefill(work, false, next, false);
    EXPECT_FALSE(failed);
    EXPECT_EQ(node.TakeTimes(0).steps, 1U);  // current GPU Job really completed
    EXPECT_GT(runner->state_preparation_stats().submitted, 0U);
    for (std::uint32_t id = 0; id < 2; ++id) {
      EXPECT_FALSE((*runner->request_slot(id))->state_usable());
      EXPECT_EQ((*runner->request_slot(id))->completed_positions(), kPast);
    }
    if (auto checked = node.Call(
            [&]() -> en::Status {
              EXPECT_TRUE(observer.held);
              const auto view = node.catalog().Describe(future);
              EXPECT_EQ(view->state, jitllm::catalog::ExtentState::kResident);
              EXPECT_FALSE(view->discarded);
              EXPECT_EQ(view->content_generation, observer.generation);
              EXPECT_EQ(view->registrations, 1U);
              observer.target = {};
              if (!node.catalog().RetireRegistration(observer.registration))
                return en::support::Error("test future registration release");
              return {};
            },
            "release held future after failed collection");
        !checked)
      return checked;
    for (std::uint32_t id = 0; id < 2; ++id) {
      if (auto cleared = runner->Clear(id); !cleared) return cleared;
      EXPECT_TRUE((*runner->request_slot(id))->state_usable());
      EXPECT_EQ((*runner->request_slot(id))->completed_positions(), 0U);
    }
    return {};
  });
  ASSERT_TRUE(result) << (result ? "" : result.error());
}
