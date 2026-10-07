// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <array>
#include <cstring>
#include <filesystem>
#include <memory>
#include <tuple>
#include <vector>

#include "base/bytes.h"
#include "engine/gemma3_runner.h"
#include "engine/support.h"

namespace en = jitllm::engine;
class Gemma3ServingGpu : public ::testing::Test {
 protected:
  struct Lifetime {
    en::PagedNode node{{.slot_bytes = en::kSlabSlotBytes}};
    std::unique_ptr<en::Gemma3Runner> runner;
    std::vector<en::PagedModel*> entered;
  };
  std::unique_ptr<Lifetime> life = std::make_unique<Lifetime>();
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
                          .max_head_rows = 2,
                          .owner_decode = true,
                          .fuse_norms = true,
                          .fuse_quant_glu = true,
                          .fuse_norm_rope = true,
                          .fuse_norm_add = true},
        0, 0);
    ASSERT_TRUE(node.Open());
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
