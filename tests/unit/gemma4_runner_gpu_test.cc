// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <numeric>
#include <span>
#include <vector>

#include "base/bytes.h"
#include "engine/gemma4_runner.h"
#include "engine/support.h"

namespace en = jitllm::engine;
class Gemma4RunnerGpu : public ::testing::Test {
 protected:
  virtual bool Invariant() const { return false; }
  virtual bool NormChains() const { return false; }
  virtual en::Gemma4Variant Variant() const { return en::Gemma4Variant::k26BA4B; }
  void SetUp() override {
    // Explicit fixture path, matching other real-model engine controls.
    const auto artifact =
        std::filesystem::path("/home/pmeenan/.local/share/jitllm/m3-artifacts") /
        (Variant() == en::Gemma4Variant::k31B
             ? "32c92e077a6816b54aa988e2dee61a3639c958fd510ea99e25f3621f10b2aa08"
             : "4ddb360c9ce08f1e984ab304b6af918be44246d52346734066b06443f7c249d3");
    ASSERT_TRUE(std::filesystem::exists(artifact));
    runner = std::make_unique<en::Gemma4Runner>(
        node,
        en::Gemma4Options{.artifact = artifact,
                          .out = "/tmp/jitllm-gemma4-runner-control",
                          .variant = Variant(),
                          .slots = 4,
                          .shared_q8 = Invariant(),
                          .fuse_norms = Invariant(),
                          .row_invariant = Invariant(),
                          .fuse_norm_rope = NormChains(),
                          .fuse_norm_add = NormChains()},
        0, 0);
    ASSERT_TRUE(node.Open());
    entered.push_back(runner.get());
    auto setup = runner->Setup();
    ASSERT_TRUE(setup) << (setup ? "" : setup.error());
    ASSERT_TRUE(node.MapWorkspace(runner->activations_needed(), runner->pool_needed()));
    const auto fixed = node.catalog().OccupancyOf(node.domain()).Total().value();
    node.SetHostFloor(runner->host_input_bytes() + runner->plan_floor_bytes());
    ASSERT_TRUE(node.Start(jitllm::base::Bytes(fixed + runner->weights().size() * en::kPagedExtent +
                                               2 * node.StateCapacity())));
    ASSERT_TRUE(runner->Register());
    ASSERT_TRUE(runner->Bind());
    node.Run();
  }
  void TearDown() override {
    auto retired = node.TearDown(entered);
    EXPECT_TRUE(retired) << (retired ? "" : retired.error());
  }
  en::Status Single(std::uint32_t slot, std::uint32_t past, std::span<const std::int32_t> tokens,
                    std::vector<float>& logits) {
    const en::Gemma4Runner::Work work{slot, past, tokens, &logits};
    return runner->Wave(std::span(&work, 1));
  }
  en::Status Held(const std::function<en::Status()>& body) {
    return node.WithRequest(0, runner->closure(), "Gemma4 lifecycle control", body);
  }
  void Exact(std::span<const float> a, std::span<const float> b) {
    ASSERT_EQ(a.size(), b.size());
    EXPECT_EQ(std::memcmp(a.data(), b.data(), a.size_bytes()), 0);
  }
  void WrongSourceLayouts(std::uint32_t positions, void* saved,
                          std::span<const en::LiveState::Range> ranges) {
    const auto other = Variant() == en::Gemma4Variant::k31B ? "gemma26" : "gemma31";
    const std::string wrong = std::string(other) + "-f16-kv-scalar-device-v1:4096:128:4096:1280";
    for (const std::string_view tag :
         {std::string_view(wrong), std::string_view("malformed"), std::string_view("")}) {
      const auto occupancy = node.catalog().OccupancyOf(node.domain()).Total().value();
      const auto extents = runner->state();
      std::array<std::uint32_t, 4> cursors{};
      for (std::uint32_t i = 0; i < 4; ++i)
        cursors[i] = (*runner->request_slot(i))->completed_positions();
      EXPECT_FALSE(runner->PrepareRestore(0, positions, ranges, tag));
      EXPECT_FALSE(runner->Adopt(0, positions, ranges, tag));
      EXPECT_FALSE(runner->RestoreCheckpoint(0, positions, saved, ranges, tag));
      EXPECT_EQ(node.catalog().OccupancyOf(node.domain()).Total().value(), occupancy);
      EXPECT_EQ(runner->state(), extents);
      for (std::uint32_t i = 0; i < 4; ++i) {
        EXPECT_EQ((*runner->request_slot(i))->completed_positions(), cursors[i]);
        EXPECT_TRUE((*runner->request_slot(i))->state_usable());
      }
    }
  }
  void JoinedDifference(std::span<const float> a, std::span<const float> b, std::uint32_t count,
                        std::uint32_t slot, std::uint32_t step) {
    ASSERT_EQ(a.size(), b.size());
    double largest = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
      ASSERT_TRUE(std::isfinite(a[i]) && std::isfinite(b[i]));
      largest = std::max(largest, std::abs(double(a[i]) - b[i]));
    }
    const auto ai = std::max_element(a.begin(), a.end()) - a.begin();
    const auto bi = std::max_element(b.begin(), b.end()) - b.begin();
    EXPECT_EQ(ai, bi);
    std::cout << "GEMMA_JOIN count=" << count << " slot=" << slot << " step=" << step
              << " max_logit_delta=" << largest << " argmax=" << ai << '/' << bi << '\n';
  }
  en::Status StateBytes(std::uint32_t slot, std::uint32_t past, std::vector<std::byte>& bytes) {
    auto ranges = runner->CheckpointRanges(past);
    if (!ranges) return en::support::Error(ranges.error());
    std::uint64_t count = 0;
    for (const auto& r : *ranges) count += r.bytes;
    std::vector<jitllm::catalog::ExtentId> staging;
    auto pinned = node.Pinned(count, 0, staging);
    if (!pinned) return en::support::Error(pinned.error());
    auto copied = runner->CopyState(slot, *pinned, *ranges, true);
    if (copied) {
      const auto* begin = static_cast<const std::byte*>(*pinned);
      bytes.assign(begin, begin + count);
    }
    auto freed = node.FreePinned(*pinned);
    return copied ? freed : copied;
  }
  en::PagedNode node{{.slot_bytes = en::kSlabSlotBytes}};
  std::unique_ptr<en::Gemma4Runner> runner;
  std::vector<en::PagedModel*> entered;
  const std::array<std::int32_t, 6> prompt{2, 818, 5279, 529, 7001, 563};
};
class Gemma4RunnerPolicy : public Gemma4RunnerGpu, public ::testing::WithParamInterface<bool> {
 protected:
  bool Invariant() const override { return GetParam(); }
};
class Gemma31RunnerGpu : public Gemma4RunnerGpu {
 protected:
  en::Gemma4Variant Variant() const override { return en::Gemma4Variant::k31B; }
  void ReplayOwnState();
};
class Gemma31NormRunnerGpu : public Gemma31RunnerGpu {
 protected:
  bool NormChains() const override { return true; }
};
void Gemma31RunnerGpu::ReplayOwnState() {
  ASSERT_EQ(runner->profile().layers, 60U);
  ASSERT_EQ(runner->profile().experts, 0U);
  EXPECT_EQ(runner->slab_padding(), 0U);
  EXPECT_EQ(runner->pitch_padding(), 0U);
  const auto source_layout = runner->CheckpointLayoutId();
  EXPECT_EQ(source_layout, "gemma31-f16-kv-scalar-device-v1:4096:128:4096:1280");
  const std::string wrong_layout = "gemma26-f16-kv-scalar-device-v1:4096:128:4096:1280";
  // Distinct identity rejects even when the caller supplies the correct31 ranges.
  EXPECT_FALSE(runner->PrepareRestore(0, 0, {}, wrong_layout));
  EXPECT_FALSE(runner->Adopt(0, 0, {}, wrong_layout));
  EXPECT_FALSE(runner->RestoreCheckpoint(0, 0, nullptr, {}, wrong_layout));
  for (const auto count : {1U, 2U, 4U}) {
    const std::array<std::uint32_t, 4> slots{0, 1, 2, 3};
    ASSERT_TRUE(runner->SelectSlots(std::span(slots).first(count)));
    auto ran = Held([&]() -> en::Status {
      constexpr std::uint64_t output_bytes = 8ULL * 262144 * sizeof(float);
      if (!node.ChargeHost(output_bytes, false)) return en::support::Error("publication refused");
      struct Charge {
        en::PagedNode& node;
        ~Charge() { node.UnchargeHost(output_bytes); }
      } charge{node};
      std::array<std::vector<float>, 4> output, baseline;
      for (std::uint32_t slot = 0; slot < 4; ++slot) {
        output[slot].reserve(262144);
        baseline[slot].reserve(262144);
        if (output[slot].capacity() != 262144 || baseline[slot].capacity() != 262144)
          return en::support::Error("unexpected publication capacity");
      }
      const std::int32_t next = 563;
      const auto prefix = [&]() -> en::Status {
        for (std::uint32_t slot = 0; slot < count; ++slot) {
          if (auto r = runner->Clear(slot); !r) return r;
          if (auto r = Single(slot, 0, prompt, output[slot]); !r) return r;
        }
        return {};
      };
      if (auto r = prefix(); !r) return r;
      const auto wave = [&]() -> en::Status {
        std::array<en::Gemma4Runner::Work, 4> work{};
        for (std::uint32_t slot = 0; slot < count; ++slot)
          work[slot] = {slot, 6, std::span(&next, 1), &output[slot]};
        return runner->Wave(std::span(work).first(count));
      };
      if (auto r = wave(); !r) return r;
      for (std::uint32_t slot = 0; slot < count; ++slot) baseline[slot] = output[slot];
      const auto& policy = runner->last_built_policy();
      EXPECT_FALSE(policy.norm_fused || policy.rope_store || policy.shared_vecq ||
                   policy.row_products || policy.lane_steps);
      EXPECT_EQ(policy.norm_rope != 0, NormChains());
      EXPECT_EQ(policy.norm_add != 0, NormChains());
      auto ranges = runner->CheckpointRanges(7);
      if (!ranges) return en::support::Error(ranges.error());
      std::uint64_t bytes = 0;
      for (const auto& r : *ranges) bytes += r.bytes;
      std::vector<jitllm::catalog::ExtentId> staging;
      auto saved = node.Pinned(bytes, 0, staging);
      if (!saved) return en::support::Error(saved.error());
      struct Pinned {
        en::PagedNode& node;
        void* memory;
        ~Pinned() { std::ignore = node.FreePinned(memory); }
      } pinned{node, *saved};
      if (auto r = runner->CopyState(0, *saved, *ranges, true); !r) return r;
      WrongSourceLayouts(7, *saved, *ranges);
      if (auto r = prefix(); !r) return r;
      if (auto r = wave(); !r) return r;
      for (std::uint32_t slot = 0; slot < count; ++slot) Exact(baseline[slot], output[slot]);
      auto repeated = node.Pinned(bytes, 0, staging);
      if (!repeated) return en::support::Error(repeated.error());
      Pinned repeated_pinned{node, *repeated};
      if (auto r = runner->CopyState(0, *repeated, *ranges, true); !r) return r;
      EXPECT_EQ(std::memcmp(*saved, *repeated, bytes), 0);
      // Save the common continuation before clear/restore/spill. Peers remain active.
      if (auto r = Single(0, 7, std::span(&next, 1), baseline[0]); !r) return r;
      if (auto r = runner->Clear(0); !r) return r;
      EXPECT_FALSE(runner->RestoreCheckpoint(0, 7, *saved, *ranges, wrong_layout));
      EXPECT_EQ((*runner->request_slot(0))->completed_positions(), 0U);
      if (auto r = runner->RestoreCheckpoint(0, 7, *saved, *ranges, source_layout); !r) return r;
      if (auto r = runner->Spill(0); !r) return r;
      if (auto r = runner->Restore(0); !r) return r;
      if (auto r = Single(0, 7, std::span(&next, 1), output[0]); !r) return r;
      Exact(baseline[0], output[0]);
      std::cout << "GEMMA31_STATE count=" << count << " checkpoint_bytes=" << bytes
                << " completed=" << (*runner->request_slot(0))->completed_positions() << '\n';
      return {};
    });
    ASSERT_TRUE(ran) << (ran ? "" : ran.error());
  }
  EXPECT_GT(runner->graph_stats().captured, 0U);
  EXPECT_GT(runner->graph_stats().replayed, 0U);
  EXPECT_EQ(runner->coverage().violations, 0U);
}
TEST_F(Gemma31RunnerGpu, OrdinarySoloAndWavesReplayAndRestoreOwnedCheckpoints) { ReplayOwnState(); }
TEST_F(Gemma31NormRunnerGpu, NormChainsReplayAndRestoreOwnedCheckpoints) { ReplayOwnState(); }
TEST_P(Gemma4RunnerPolicy, SoloTwoFourJoinedDecodeHasStrictGreediesAndExactSamePolicyReplay) {
  for (const auto count : {1U, 2U, 4U}) {
    std::array<std::array<std::vector<float>, 3>, 4> baseline;
    std::array<std::array<std::vector<float>, 3>, 4> joined_baseline;
    std::array<std::vector<float>, 4> joined;
    std::array<std::vector<std::byte>, 4> state;
    std::array<std::vector<std::int32_t>, 4> prompts;
    std::array<std::int32_t, 4> anchors{};
    std::array<std::uint32_t, 4> ids{0, 1, 2, 3};
    for (std::uint32_t slot = 0; slot < count; ++slot) {
      prompts[slot].assign(prompt.begin(), prompt.end());
      for (std::uint32_t i = 0; i < slot; ++i) prompts[slot].push_back(563);
      ASSERT_TRUE(runner->SelectSlots(std::span(ids).subspan(slot, 1)));
      auto ran = Held([&]() -> en::Status {
        if (auto r = runner->Clear(slot); !r) return r;
        std::vector<float> first;
        if (auto r = Single(slot, 0, prompts[slot], first); !r) return r;
        anchors[slot] =
            static_cast<std::int32_t>(std::max_element(first.begin(), first.end()) - first.begin());
        for (std::uint32_t step = 0; step < 3; ++step) {
          if (auto r = Single(slot, static_cast<std::uint32_t>(prompts[slot].size()) + step,
                              std::span(&anchors[slot], 1), baseline[slot][step]);
              !r)
            return r;
        }
        return StateBytes(slot, static_cast<std::uint32_t>(prompts[slot].size()) + 3, state[slot]);
      });
      ASSERT_TRUE(ran) << (ran ? "" : ran.error());
    }
    ASSERT_TRUE(runner->SelectSlots(std::span(ids).first(count)));
    auto ran = Held([&]() -> en::Status {
      for (std::uint32_t slot = 0; slot < count; ++slot) {
        if (auto r = runner->Clear(slot); !r) return r;
        std::vector<float> ignored;
        if (auto r = Single(slot, 0, prompts[slot], ignored); !r) return r;
      }
      for (std::uint32_t step = 0; step < 3; ++step) {
        std::array<en::Gemma4Runner::Work, 4> work{};
        for (std::uint32_t slot = 0; slot < count; ++slot)
          work[slot] = {slot, static_cast<std::uint32_t>(prompts[slot].size()) + step,
                        std::span(&anchors[slot], 1), &joined[slot]};
        if (auto r = runner->Wave(std::span(work).first(count)); !r) return r;
        if (step == 0) {
          const auto& p = runner->last_built_policy();
          std::cout << "GEMMA_JOIN_POLICY optimized=" << Invariant() << " rows=" << p.rows
                    << " segments=" << p.segments << " norm_fused=" << p.norm_fused
                    << " rope_store=" << p.rope_store << " shared_vecq=" << p.shared_vecq
                    << " row_products=" << p.row_products << " lane_steps=" << p.lane_steps << '\n';
        }
        for (std::uint32_t slot = 0; slot < count; ++slot) {
          JoinedDifference(baseline[slot][step], joined[slot], count, slot, step);
          if (Invariant()) Exact(baseline[slot][step], joined[slot]);
          joined_baseline[slot][step] = joined[slot];
        }
      }
      for (std::uint32_t slot = 0; slot < count; ++slot) {
        std::vector<std::byte> actual;
        if (auto r = StateBytes(slot, static_cast<std::uint32_t>(prompts[slot].size()) + 3, actual);
            !r)
          return r;
        const auto changed =
            std::inner_product(actual.begin(), actual.end(), state[slot].begin(), std::size_t{0},
                               std::plus<>(), [](std::byte a, std::byte b) { return a != b; });
        std::cout << "GEMMA_JOIN_STATE count=" << count << " slot=" << slot
                  << " changed_bytes=" << changed << " total_bytes=" << actual.size() << '\n';
        if (Invariant()) EXPECT_EQ(actual, state[slot]);
        state[slot] = std::move(actual);
      }
      // Same joined policy, fresh state, fresh positions: cached graph runs
      // must reproduce all logits and initialized KV bytes exactly.
      for (std::uint32_t slot = 0; slot < count; ++slot) {
        if (auto r = runner->Clear(slot); !r) return r;
        std::vector<float> ignored;
        if (auto r = Single(slot, 0, prompts[slot], ignored); !r) return r;
      }
      for (std::uint32_t step = 0; step < 3; ++step) {
        std::array<en::Gemma4Runner::Work, 4> work{};
        for (std::uint32_t slot = 0; slot < count; ++slot)
          work[slot] = {slot, static_cast<std::uint32_t>(prompts[slot].size()) + step,
                        std::span(&anchors[slot], 1), &joined[slot]};
        if (auto r = runner->Wave(std::span(work).first(count)); !r) return r;
        for (std::uint32_t slot = 0; slot < count; ++slot)
          Exact(joined_baseline[slot][step], joined[slot]);
      }
      for (std::uint32_t slot = 0; slot < count; ++slot) {
        std::vector<std::byte> actual;
        if (auto r = StateBytes(slot, static_cast<std::uint32_t>(prompts[slot].size()) + 3, actual);
            !r)
          return r;
        EXPECT_EQ(actual, state[slot]);
      }
      return {};
    });
    ASSERT_TRUE(ran) << (ran ? "" : ran.error());
  }
  EXPECT_GT(runner->graph_stats().captured, 0U);
  EXPECT_GT(runner->graph_stats().replayed, 0U);
  EXPECT_EQ(runner->coverage().violations, 0U);
  EXPECT_GT(runner->coverage().tensors, 0U);
}
INSTANTIATE_TEST_SUITE_P(ProductPolicies, Gemma4RunnerPolicy, ::testing::Values(false, true));
TEST_F(Gemma4RunnerGpu, MaximumRaggedPrefillFitsEnvelopeAndReplaysIndependentStates) {
  std::array<std::uint32_t, 4> ids{0, 1, 2, 3};
  ASSERT_TRUE(runner->SelectSlots(ids));
  std::array<std::vector<std::int32_t>, 4> inputs;
  inputs[0].resize(125);
  for (std::size_t i = 0; i < inputs[0].size(); ++i) inputs[0][i] = prompt[i % prompt.size()];
  for (std::size_t i = 1; i < inputs.size(); ++i) inputs[i] = {2};
  std::array<std::vector<float>, 4> expected, actual;
  std::array<std::vector<std::byte>, 4> state;
  auto ran = Held([&]() -> en::Status {
    for (std::uint32_t repeat = 0; repeat < 3; ++repeat) {
      for (const auto id : ids)
        if (auto r = runner->Clear(id); !r) return r;
      std::array<en::Gemma4Runner::Work, 4> work{};
      for (const auto id : ids) work[id] = {id, 0, inputs[id], &actual[id]};
      if (auto r = runner->Wave(work); !r) return r;
      for (const auto id : ids) {
        std::vector<std::byte> bytes;
        if (auto r = StateBytes(id, static_cast<std::uint32_t>(inputs[id].size()), bytes); !r)
          return r;
        if (repeat == 0) {
          expected[id] = actual[id];
          state[id] = std::move(bytes);
        } else {
          Exact(expected[id], actual[id]);
          EXPECT_EQ(bytes, state[id]);
        }
      }
    }
    return {};
  });
  ASSERT_TRUE(ran) << (ran ? "" : ran.error());
  EXPECT_GT(runner->graph_stats().captured, 0U);
  EXPECT_GT(runner->graph_stats().replayed, 0U);
}
TEST_F(Gemma4RunnerGpu, SpillCheckpointRestoreReplayAndPeerClearPreserveExactContinuations) {
  EXPECT_EQ(runner->CheckpointLayoutId(), "gemma26-f16-kv-scalar-device-v1:4096:128:4096:1280");
  ASSERT_TRUE(runner->SelectSlots(std::array<std::uint32_t, 2>{0, 1}));
  auto ran = Held([&]() -> en::Status {
    std::vector<float> logits, peer;
    if (auto r = Single(0, 0, prompt, logits); !r) return r;
    if (auto r = Single(1, 0, prompt, peer); !r) return r;
    const auto source_layout = runner->CheckpointLayoutId();
    auto ranges = runner->CheckpointRanges(6);
    if (!ranges) return en::support::Error(ranges.error());
    std::uint64_t bytes = 0;
    for (const auto& r : *ranges) bytes += r.bytes;
    std::vector<jitllm::catalog::ExtentId> staging;
    auto saved = node.Pinned(bytes, 0, staging);
    if (!saved) return en::support::Error(saved.error());
    if (auto r = runner->CopyState(0, *saved, *ranges, true); !r) return r;
    WrongSourceLayouts(6, *saved, *ranges);
    const auto next =
        static_cast<std::int32_t>(std::max_element(logits.begin(), logits.end()) - logits.begin());
    std::vector<float> baseline, restored;
    if (auto r = Single(0, 6, std::span(&next, 1), baseline); !r) return r;
    if (auto r = runner->Clear(0); !r) return r;
    if (auto r = runner->RestoreCheckpoint(0, 6, *saved, *ranges, source_layout); !r) return r;
    if (auto r = runner->Spill(0); !r) return r;
    if (auto r = runner->Restore(0); !r) return r;
    runner->DropPlans();
    EXPECT_EQ(runner->plans_bytes(), 0U);
    if (auto r = Single(0, 6, std::span(&next, 1), restored); !r) return r;
    Exact(baseline, restored);
    const auto peer_next =
        static_cast<std::int32_t>(std::max_element(peer.begin(), peer.end()) - peer.begin());
    if (auto r = Single(1, 6, std::span(&peer_next, 1), peer); !r) return r;
    Exact(baseline, peer);
    // Invalid continuation and duplicate slots must publish nothing.
    std::vector<float> canary{17};
    const en::Gemma4Runner::Work bad{0, 6, std::span(&next, 1), &canary};
    EXPECT_FALSE(runner->Wave(std::span(&bad, 1)));
    EXPECT_EQ(canary, std::vector<float>{17});
    auto slot = runner->request_slot(0);
    EXPECT_TRUE(slot && (*slot)->state_usable());
    EXPECT_EQ((*slot)->completed_positions(), 7U);
    if (auto r = node.FreePinned(*saved); !r) return r;
    return {};
  });
  ASSERT_TRUE(ran) << (ran ? "" : ran.error());
}
TEST_F(Gemma4RunnerGpu, PagedOutWeightsAdmissionRefusesWithoutLosingCompletedPrefix) {
  ASSERT_TRUE(runner->SelectSlots(std::array<std::uint32_t, 1>{0}));
  std::vector<float> first, baseline, resumed;
  ASSERT_TRUE(Single(0, 0, prompt, first));
  const auto next =
      static_cast<std::int32_t>(std::max_element(first.begin(), first.end()) - first.begin());
  const auto source_layout = runner->CheckpointLayoutId();
  auto ranges = runner->CheckpointRanges(6);
  ASSERT_TRUE(ranges);
  std::uint64_t bytes = 0;
  for (const auto& range : *ranges) bytes += range.bytes;
  std::vector<jitllm::catalog::ExtentId> staging;
  auto checkpoint = node.Pinned(bytes, 0, staging);
  ASSERT_TRUE(checkpoint);
  ASSERT_TRUE(runner->CopyState(0, *checkpoint, *ranges, true));
  ASSERT_TRUE(Single(0, 6, std::span(&next, 1), baseline));
  ASSERT_TRUE(runner->Clear(0));
  ASSERT_TRUE(runner->RestoreCheckpoint(0, 6, *checkpoint, *ranges, source_layout));
  // A separate clean state-growth refusal under a held weight/state lease.
  auto growth = Held([&]() -> en::Status {
    auto available = node.FreeBytes();
    if (!available || *available <= (16U << 20U)) return en::support::Error("test capacity");
    const auto reservation = *available - (16U << 20U);
    if (!node.ChargeHost(reservation, false)) return en::support::Error("test pressure");
    auto refused_growth = runner->ReserveStateThrough(0, 1025);
    node.UnchargeHost(reservation);
    EXPECT_FALSE(refused_growth);
    auto slot = runner->request_slot(0);
    EXPECT_TRUE(slot && (*slot)->state_usable());
    EXPECT_TRUE(slot && (*slot)->refused_state_growth());
    EXPECT_EQ((*slot)->completed_positions(), 6U);
    return {};
  });
  ASSERT_TRUE(growth) << (growth ? "" : growth.error());
  // Materialize the next-row padded read footprint before removing weights:
  // this refusal must come from Job admission, not state growth.
  ASSERT_TRUE(runner->ReserveStateThrough(0, 7));
  ASSERT_TRUE(node.Evict(runner->weights()));
  auto free = node.FreeBytes();
  ASSERT_TRUE(free);
  ASSERT_GT(*free, 16U << 20U);
  const auto pressure = *free - (16U << 20U);
  ASSERT_TRUE(node.ChargeHost(pressure, false));
  resumed = {17};
  auto refused = Single(0, 6, std::span(&next, 1), resumed);
  node.UnchargeHost(pressure);
  if (!refused) std::cout << "GEMMA_ADMISSION_REFUSAL " << refused.error() << '\n';
  EXPECT_FALSE(refused);
  EXPECT_EQ(resumed, std::vector<float>{17});
  auto slot = runner->request_slot(0);
  ASSERT_TRUE(slot);
  EXPECT_EQ((*slot)->completed_positions(), 6U);
  EXPECT_TRUE((*slot)->state_usable());
  EXPECT_FALSE((*slot)->refused_state_growth());
  // Admission capacity restored: same initialized state and exact continuation.
  std::vector<std::byte> state;
  ASSERT_TRUE(StateBytes(0, 6, state));
  const auto* saved = static_cast<const std::byte*>(*checkpoint);
  EXPECT_EQ(std::memcmp(saved, state.data(), state.size()), 0);
  ASSERT_TRUE(Single(0, 6, std::span(&next, 1), resumed));
  Exact(baseline, resumed);
  ASSERT_TRUE(node.FreePinned(*checkpoint));
}
TEST_F(Gemma4RunnerGpu, ServingRestoreNeedsProvenCompleteCopiesAndProtectsThePeer) {
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
      EXPECT_FALSE(runner->CompleteRestore(0, 5));
      EXPECT_FALSE(runner->CompleteRestore(0, 6));
      EXPECT_EQ((*slot)->completed_positions(), 6U);
      EXPECT_FALSE((*slot)->state_usable());
      std::vector<float> refused{17};
      EXPECT_FALSE(Single(0, 6, std::span(prompt).first(1), refused));
      EXPECT_EQ(refused, std::vector<float>{17});
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

TEST_F(Gemma4RunnerGpu, MaximumAllHeadRowsReplayWithSeparatelyFundedCallerVectors) {
  ASSERT_TRUE(runner->SelectSlots(std::array<std::uint32_t, 1>{0}));
  const auto bytes = 128ULL * 262144 * sizeof(float);
  // Pinned native outputs were funded during Setup. These two caller-owned
  // copies have their own host grant before either output vector can grow.
  ASSERT_TRUE(node.ChargeHost(2 * bytes, false));
  struct Grant {
    en::PagedNode& node;
    std::uint64_t bytes;
    ~Grant() { node.UnchargeHost(bytes); }
  } grant{node, 2 * bytes};
  std::vector<float> expected, repeated;
  std::vector<std::int32_t> tokens(128);
  for (std::size_t i = 0; i < tokens.size(); ++i) tokens[i] = prompt[i % prompt.size()];
  const en::Gemma4Runner::Work first{0, 0, tokens, &expected};
  ASSERT_TRUE(runner->Wave(std::span(&first, 1), true));
  EXPECT_EQ(expected.size(), 128U * 262144);
  EXPECT_TRUE(std::ranges::all_of(expected, [](float x) { return std::isfinite(x); }));
  for (unsigned run = 0; run < 3; ++run) {
    ASSERT_TRUE(runner->Clear(0));
    const en::Gemma4Runner::Work again{0, 0, tokens, &repeated};
    ASSERT_TRUE(runner->Wave(std::span(&again, 1), true));
    Exact(expected, repeated);
  }
  EXPECT_GT(runner->graph_stats().replayed, 0U);
}
