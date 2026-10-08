// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <format>
#include <memory>
#include <tuple>
#include <vector>

#include "base/bytes.h"
#include "base/sha256.h"
#include "engine/gemma2_runner.h"
#include "engine/support.h"
#include "platform/kept_files.h"

namespace en = jitllm::engine;
class Gemma2CheckpointGpu : public ::testing::Test {
 protected:
  struct Lifetime {
    en::PagedNode node{{.slot_bytes = en::kSlabSlotBytes}};
    std::unique_ptr<en::Gemma2Runner> runner;
    std::vector<en::PagedModel*> entered;
  };
  std::unique_ptr<Lifetime> life = std::make_unique<Lifetime>();
  en::PagedNode& node = life->node;
  std::unique_ptr<en::Gemma2Runner>& runner = life->runner;
  const std::array<std::int32_t, 6> prompt{2, 818, 5279, 529, 7001, 563};
  std::filesystem::path scratch;
  int spill_directory = -1;
  bool retired_primary = false, retirement_failed = false;
  virtual bool JoinedLookahead() const { return false; }
  static std::uint64_t HostFloor(const en::Gemma2Runner& model) {
    return model.host_input_bytes() + model.plan_floor_bytes() + (16ULL << 20U);
  }
  en::Status Start(Lifetime& target, bool keep) {
    auto& n = target.node;
    auto& r = target.runner;
    r = std::make_unique<en::Gemma2Runner>(
        n,
        en::Gemma2Options{
            .artifact = "/home/pmeenan/.local/share/jitllm/gemma2-import-20261007/artifacts/"
                        "eb18d30d0a7de3a95c7b6994b65a12a057ffbf42866add6f128873de8b7aa870",
            .context = 8192,
            .slots = 2,
            .max_wave_rows = JoinedLookahead() ? 256U : 0U,
            .max_head_rows = 2,
            .owner_decode = true,
            .packed_prefill = JoinedLookahead(),
            .bounded_roots = JoinedLookahead(),
            .fuse_norms = true,
            .fuse_quant_glu = true,
            .fuse_norm_rope = false,
            .fuse_norm_add = true,
            .spill_place =
                [fd = spill_directory, keep](std::uint32_t slot) {
                  return en::LiveState::SpillPlace{.directory = {},
                                                   .dir = fd,
                                                   .name = std::format("slot{}.kv", slot),
                                                   .keep = keep};
                }},
        0, 0);
    if (auto x = n.Open(); !x) return x;
    target.entered.push_back(r.get());
    if (auto x = r->Setup(); !x) return x;
    if (auto x = n.MapWorkspace(r->activations_needed(), r->pool_needed()); !x) return x;
    const auto fixed = n.catalog().OccupancyOf(n.domain()).Total().value();
    n.SetHostFloor(HostFloor(*r));
    if (auto x = n.Start(jitllm::base::Bytes(fixed + r->weights().size() * en::kPagedExtent +
                                             4 * n.StateCapacity()));
        !x)
      return x;
    if (auto x = r->Register(); !x) return x;
    if (auto x = r->Bind(); !x) return x;
    n.Run();
    return r->SelectSlots(std::array<std::uint32_t, 2>{0, 1});
  }
  void SetUp() override {
    std::string name = "/home/pmeenan/.cache/jitllm-gemma2-checkpoint-XXXXXX";
    ASSERT_NE(mkdtemp(name.data()), nullptr);
    scratch = name;
    auto opened = jitllm::platform::OpenPrivateDirectory(-1, scratch.c_str());
    ASSERT_TRUE(opened);
    spill_directory = *opened;
    const auto started = Start(*life, false);
    ASSERT_TRUE(started) << (started ? "" : started.error());
  }
  en::Status RetirePrimary() {
    if (retired_primary) return en::support::Error("primary retirement already attempted");
    retired_primary = true;
    auto retired = node.TearDown(life->entered);
    retirement_failed = !retired;
    return retired;
  }
  void TearDown() override {
    if (!retired_primary) {
      const auto retired = RetirePrimary();
      EXPECT_TRUE(retired) << (retired ? "" : retired.error());
    }
    if (retirement_failed) {
      // Preserve the node, open spill directory and private files until exit.
      std::ignore = life.release();
      return;
    }
    if (spill_directory >= 0) close(spill_directory);
    std::filesystem::remove_all(scratch);
  }
  en::Status Single(std::uint32_t slot, std::uint32_t past, std::span<const std::int32_t> ids,
                    std::vector<float>& head) {
    const en::Gemma2Runner::Work work{slot, past, ids, &head};
    return runner->Wave(std::span(&work, 1));
  }
  en::Status Held(const std::function<en::Status()>& body) {
    return node.WithRequest(0, runner->closure(), "Gemma2 serving lifecycle", body);
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
    ASSERT_FALSE(a.empty());
    ASSERT_TRUE(std::ranges::all_of(a, [](float x) { return std::isfinite(x); }));
    ASSERT_TRUE(std::ranges::all_of(b, [](float x) { return std::isfinite(x); }));
    EXPECT_EQ(std::memcmp(a.data(), b.data(), a.size_bytes()), 0);
  }
};
TEST_F(Gemma2CheckpointGpu, ServingRestoreNeedsProvenCompleteCopiesAndProtectsThePeer) {
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
      EXPECT_FALSE(runner->PrepareRestore(0, 5, footprint, "gemma2-wrong-generation"));
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
      EXPECT_FALSE(runner->SelectSlots(std::array<std::uint32_t, 1>{1}));
      EXPECT_FALSE(runner->Restore(0));
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

TEST_F(Gemma2CheckpointGpu, PendingRestoreCanBeCancelledAndCannotPublishPeerOrStaleCursors) {
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

TEST_F(Gemma2CheckpointGpu, HeldSelectionPreservesAdmissionAndRefreshesChangedOwners) {
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

TEST_F(Gemma2CheckpointGpu, WrappedRingCheckpointAndKeptAdoptionReplayTheExactPeer) {
  constexpr std::uint32_t prefix = 4355;
  EXPECT_EQ(runner->layout().local_cells, 4352U);
  std::vector<std::int32_t> tokens(prefix);
  for (std::size_t i = 0; i < tokens.size(); ++i) tokens[i] = prompt[i % prompt.size()];
  const std::array<std::int32_t, 1> anchor{563};
  std::vector<en::LiveState::Range> footprint;
  std::vector<float> expected_next;
  const auto source_layout = runner->CheckpointLayoutId();
  auto exercised = Held([&]() -> en::Status {
    std::vector<float> a, b;
    for (std::uint32_t past = 0; past < prefix;) {
      const auto rows = std::min(128U, prefix - past);
      if (auto x = Single(0, past, std::span(tokens).subspan(past, rows), a); !x) return x;
      if (auto x = Single(1, past, std::span(tokens).subspan(past, rows), b); !x) return x;
      past += rows;
    }
    Exact(a, b);
    auto* slot = *runner->request_slot(0);
    footprint = slot->state().used_ranges();
    const auto bytes = slot->used_state_bytes();
    std::vector<jitllm::catalog::ExtentId> staging;
    auto pinned = node.Pinned(bytes, 0, staging);
    if (!pinned) return en::support::Error(pinned.error());
    auto copied = [&]() -> en::Status {
      if (auto x = runner->CopyState(0, *pinned, footprint, true); !x) return x;
      if (auto x = Single(0, prefix, anchor, a); !x) return x;
      if (auto x = Single(1, prefix, anchor, b); !x) return x;
      Exact(a, b);
      if (auto x = runner->PrepareRestore(0, prefix, footprint, runner->CheckpointLayoutId()); !x)
        return x;
      EXPECT_FALSE(runner->CompleteRestore(0, prefix));
      if (auto x = runner->CopyState(0, *pinned, footprint, false); !x) return x;
      if (auto x = runner->CompleteRestore(0, prefix); !x) return x;
      std::vector<float> restored;
      if (auto x = Single(0, prefix, anchor, restored); !x) return x;
      Exact(a, restored);
      if (auto x = runner->Spill(0); !x) return x;
      EXPECT_TRUE(slot->kept_whole());
      // A live peer supplies the independent same-history continuation oracle.
      return Single(1, prefix + 1, anchor, expected_next);
    }();
    // Submitted copies have a completion result before this node-owned buffer
    // is freed. An unproven failure faults the cohort and makes TearDown sticky.
    auto freed = node.FreePinned(*pinned);
    return copied ? freed : copied;
  });
  ASSERT_TRUE(exercised) << (exercised ? "" : exercised.error());
  const auto primary = RetirePrimary();
  ASSERT_TRUE(primary) << (primary ? "" : primary.error());
  auto restarted = std::make_unique<Lifetime>();
  auto check = [&]() -> en::Status {
    if (auto x = Start(*restarted, true); !x) return x;
    auto& target = *restarted->runner;
    auto* slot = *target.request_slot(0);
    EXPECT_EQ(slot->completed_positions(), 0U);
    EXPECT_EQ(slot->used_state_bytes(), 0U);
    EXPECT_EQ(target.CheckpointLayoutId(), source_layout);
    EXPECT_FALSE(target.Adopt(0, prefix + 1, footprint, "gemma2-wrong-layout"));
    auto incomplete = footprint;
    incomplete.pop_back();
    EXPECT_FALSE(target.Adopt(0, prefix + 1, incomplete, source_layout));
    if (auto x = target.Adopt(0, prefix + 1, footprint, source_layout); !x) return x;
    EXPECT_TRUE(slot->is_spilled());
    EXPECT_GT(slot->spilled_bytes(), 0U);
    EXPECT_FALSE(slot->kept_whole());
    EXPECT_FALSE(target.Adopt(0, prefix + 1, footprint, source_layout));
    if (auto x = target.Restore(0); !x) return x;
    EXPECT_FALSE(slot->is_spilled());
    EXPECT_TRUE(slot->kept_whole());
    return restarted->node.WithRequest(
        0, target.closure(), "Gemma2 kept adoption", [&]() -> en::Status {
          std::vector<float> head;
          const en::Gemma2Runner::Work work{0, prefix + 1, anchor, &head};
          if (auto x = target.Wave(std::span(&work, 1)); !x) return x;
          Exact(head, expected_next);
          EXPECT_FALSE(slot->kept_whole());
          EXPECT_EQ(slot->completed_positions(), prefix + 2);
          return {};
        });
  }();
  const auto retired = restarted->node.TearDown(restarted->entered);
  EXPECT_TRUE(retired) << (retired ? "" : retired.error());
  if (!retired) {
    retirement_failed = true;
    std::ignore = restarted.release();
  }
  ASSERT_TRUE(check) << (check ? "" : check.error());
}

class Gemma2LookaheadGpu : public Gemma2CheckpointGpu {
 protected:
  bool JoinedLookahead() const override { return true; }
};

TEST_F(Gemma2LookaheadGpu, SuppressedNextHeadsStillCaptureTheCorrectAfterOwnerPosition) {
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
      const std::array<en::Gemma2Runner::Work, 2> initial{
          {{0, 0, std::span(tokens[0]).first(kRows), &heads[0]},
           {1, 0, std::span(tokens[1]).first(kRows), &heads[1]}}};
      const std::array<en::Gemma2Runner::PrefillNext, 2> hints{{{0, kRows, 0}, {1, kRows, kRows}}};
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
      const en::Gemma2Runner::Work finished{0, kRows, std::span(tokens[0]).last(kRows), &heads[0]};
      if (auto r = runner->WavePrefill(std::span(&finished, 1), true); !r) return r;
      const en::Gemma2Runner::Work middle{1, kRows, std::span(tokens[1]).subspan(kRows, kRows),
                                          &heads[1]};
      const en::Gemma2Runner::PrefillNext next{1, kRows, 0};
      if (auto r = runner->WavePrefill(std::span(&middle, 1), false,
                                       std::span(&next, hinted ? 1U : 0U), true);
          !r)
        return r;
      if (hinted) EXPECT_EQ(runner->lookahead_stats().captured_ahead - before.captured_ahead, 1U);
      const auto replayed = runner->graph_stats().replayed;
      const en::Gemma2Runner::Work last{1, 2 * kRows, std::span(tokens[1]).last(kRows), &heads[1]};
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

TEST_F(Gemma2LookaheadGpu, LookaheadRefusalAndAbandonedPredictionKeepTheCompletedPrefix) {
  ASSERT_TRUE(runner->SelectSlots(std::array<std::uint32_t, 1>{0}));
  struct Charge {
    en::PagedNode& node;
    std::uint64_t bytes;
    ~Charge() { node.UnchargeHost(bytes); }
  };
  const auto output_bytes = std::uint64_t{runner->profile().vocab} * sizeof(float);
  ASSERT_TRUE(node.ChargeHost(output_bytes, false));
  const Charge output_charge{node, output_bytes};
  std::vector<float> row;
  row.reserve(runner->profile().vocab);
  auto ran = Held([&]() -> en::Status {
    runner->DropPlans();
    const auto baseline_host = node.host_counted();
    const auto overcharges = node.host_overcharges();
    const auto refused = runner->lookahead_stats().refused;
    const auto cached = runner->lookahead_stats().cached;
    const en::Gemma2Runner::PrefillNext hint{0, 2};
    en::Gemma2Runner::Work work{0, 0, std::span(prompt).first(1), &row};
    // Seed the required plan and padded state. Pressure affects only optional
    // capture/lookahead, with no new state backing needed by the next row.
    if (auto r = runner->WavePrefill(std::span(&work, 1), false); !r) return r;
    // Consume the unused startup host floor as well as free device budget.
    // Small Gemma2 plans may still fit inside that pre-funded host allowance.
    auto free = node.FreeBytes();
    if (!free || *free == 0)
      return en::support::Error("lookahead control cannot fund input-only pressure");
    const auto floor = HostFloor(*runner);
    const auto unused_floor = floor > node.host_counted() ? floor - node.host_counted() : 0U;
    const auto fill = *free + unused_floor;
    if (!node.ChargeHost(fill, false))
      return en::support::Error("lookahead control cannot take its pressure charge");
    {
      const Charge pressure{node, fill};
      auto remaining = node.FreeBytes();
      if (!remaining || *remaining != 0)
        return en::support::Error("lookahead pressure did not exhaust the budget");
      work.n_past = 1;
      row.assign(1, 123.0F);
      if (auto r = runner->WavePrefill(std::span(&work, 1), false, std::span(&hint, 1)); !r)
        return r;
      EXPECT_TRUE(row.empty());
      EXPECT_EQ((*runner->request_slot(0))->completed_positions(), 2U);
      EXPECT_EQ(runner->lookahead_stats().refused, refused + 1);
      EXPECT_EQ(runner->lookahead_stats().cached, cached);
      EXPECT_EQ(node.host_overcharges(), overcharges);
    }
    work.n_past = 2;
    if (auto r = runner->WavePrefill(std::span(&work, 1), false, std::span(&hint, 1)); !r) return r;
    EXPECT_EQ(runner->lookahead_stats().cached, cached + 1);
    EXPECT_EQ((*runner->request_slot(0))->completed_positions(), 3U);
    // Cancel before the predicted chunk: its plan is ordinary reclaimable
    // cache, and dropping it advances no cursor and leaves no host allowance.
    runner->DropPlans();
    EXPECT_EQ(node.host_counted(), baseline_host);
    EXPECT_EQ((*runner->request_slot(0))->completed_positions(), 3U);
    if (auto r = runner->Clear(0); !r) return r;
    EXPECT_EQ((*runner->request_slot(0))->completed_positions(), 0U);
    EXPECT_EQ((*runner->request_slot(0))->used_state_bytes(), 0U);
    EXPECT_EQ(node.host_counted(), baseline_host);
    return {};
  });
  ASSERT_TRUE(ran) << (ran ? "" : ran.error());
}
