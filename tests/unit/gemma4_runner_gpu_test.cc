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
#include <tuple>
#include <vector>

#include "base/bytes.h"
#include "engine/gemma4_runner.h"
#include "engine/support.h"

namespace en = jitllm::engine;
class Gemma4RunnerGpu : public ::testing::Test {
 protected:
  virtual std::uint32_t Slots() const { return 4; }
  virtual std::uint32_t MaxRows() const { return 128; }
  virtual bool CaptureAhead() const { return false; }
  virtual std::uint32_t HeadRows() const { return 0; }
  virtual bool Invariant() const { return false; }
  virtual bool DefaultNorms() const { return false; }
  virtual bool RetainFeatures() const { return false; }
  virtual bool NormChains() const { return false; }
  virtual bool MoeChains() const { return false; }
  virtual en::Gemma4Variant Variant() const { return en::Gemma4Variant::k26BA4B; }
  void SetUp() override {
    // Explicit fixture path, matching other real-model engine controls.
    const auto artifact =
        std::filesystem::path("/home/pmeenan/.local/share/jitllm/m3-artifacts") /
        (Variant() == en::Gemma4Variant::k31B
             ? "32c92e077a6816b54aa988e2dee61a3639c958fd510ea99e25f3621f10b2aa08"
             : "4ddb360c9ce08f1e984ab304b6af918be44246d52346734066b06443f7c249d3");
    ASSERT_TRUE(std::filesystem::exists(artifact));
    en::Gemma4Options options{.artifact = artifact,
                              .out = "/tmp/jitllm-gemma4-runner-control",
                              .variant = Variant(),
                              .max_rows = MaxRows(),
                              .slots = Slots(),
                              .max_head_rows = HeadRows(),
                              .retain_features = RetainFeatures(),
                              .shared_q8 = Invariant(),
                              .row_invariant = Invariant(),
                              .fuse_norm_rope = NormChains(),
                              .fuse_norm_add = NormChains(),
                              .fuse_gemma_route = MoeChains(),
                              .fuse_gemma_reduce = MoeChains(),
                              .capture_ahead = CaptureAhead(),
                              .prefill_lookahead_capacity = CaptureAhead() ? 2U : 1U};
    // Historical ordinary fixtures stay explicitly off; default fixtures use
    // the production default without overriding the option.
    if (!DefaultNorms()) options.fuse_norms = Invariant();
    runner = std::make_unique<en::Gemma4Runner>(node, std::move(options), 0, 0);
    ASSERT_TRUE(node.Open());
    entered.push_back(runner.get());
    auto setup = runner->Setup();
    ASSERT_TRUE(setup) << (setup ? "" : setup.error());
    ASSERT_TRUE(node.MapWorkspace(runner->activations_needed(), runner->pool_needed()));
    const auto fixed = node.catalog().OccupancyOf(node.domain()).Total().value();
    node.SetHostFloor(runner->host_input_bytes() +
                      (CaptureAhead() ? 2U : 1U) * runner->plan_floor_bytes());
    ASSERT_TRUE(node.Start(jitllm::base::Bytes(fixed + runner->weights().size() * en::kPagedExtent +
                                               2 * node.StateCapacity())));
    ASSERT_TRUE(runner->Register());
    ASSERT_TRUE(runner->Bind());
    node.Run();
  }
  void TearDown() override {
    auto retired = node.TearDown(entered);
    EXPECT_TRUE(retired) << (retired ? "" : retired.error());
    // Retain all execution owners when completion cannot be proved.
    if (!retired) std::ignore = lifetime.release();
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
  void HeldSelectionControl();
  void StateOnlyControl();
  void HeadCapacityControl();
  struct Lifetime {
    en::PagedNode node{{.slot_bytes = en::kSlabSlotBytes}};
    std::unique_ptr<en::Gemma4Runner> runner;
    std::vector<en::PagedModel*> entered;
  };
  std::unique_ptr<Lifetime> lifetime = std::make_unique<Lifetime>();
  en::PagedNode& node = lifetime->node;
  std::unique_ptr<en::Gemma4Runner>& runner = lifetime->runner;
  std::vector<en::PagedModel*>& entered = lifetime->entered;
  const std::array<std::int32_t, 6> prompt{2, 818, 5279, 529, 7001, 563};
};
class Gemma26MoeRunnerGpu : public Gemma4RunnerGpu {
 protected:
  bool NormChains() const override { return true; }
  bool MoeChains() const override { return true; }
};
TEST_F(Gemma26MoeRunnerGpu, IndependentJoinedReplayCheckpointAndSpillKeepCompletedStateExact) {
  const std::array<std::uint32_t, 4> ids{0, 1, 2, 3};
  const std::array<std::int32_t, 3> anchors{45518, 107, 101};
  auto maximum_ranges = runner->CheckpointRanges(10);
  ASSERT_TRUE(maximum_ranges);
  std::uint64_t one_state = 0;
  for (const auto& range : *maximum_ranges) one_state += range.bytes;
  // At most four retained snapshots plus one current copy, twelve saved
  // heads, four working heads and two continuation heads. Pinned transfer
  // storage remains separately catalog-backed by StateBytes/CopyState.
  const auto host_bytes = 5 * one_state + 18ULL * 262144 * sizeof(float);
  ASSERT_TRUE(node.ChargeHost(host_bytes, false));
  struct Copies {
    en::PagedNode& node;
    std::uint64_t bytes;
    ~Copies() { node.UnchargeHost(bytes); }
  } copies{node, host_bytes};
  for (const auto count : {1U, 2U, 4U}) {
    ASSERT_TRUE(runner->SelectSlots(std::span(ids).first(count)));
    std::array<std::array<std::vector<float>, 3>, 4> expected;
    std::array<std::vector<std::byte>, 4> state;
    auto ran = Held([&]() -> en::Status {
      for (unsigned repeat = 0; repeat < 3; ++repeat) {
        std::array<std::vector<float>, 4> output;
        for (std::uint32_t id = 0; id < count; ++id) {
          if (auto r = runner->Clear(id); !r) return r;
          if (auto r = Single(id, 0, prompt, output[id]); !r) return r;
        }
        for (std::uint32_t step = 0; step < 3; ++step) {
          std::array<en::Gemma4Runner::Work, 4> work{};
          for (std::uint32_t id = 0; id < count; ++id)
            work[id] = {id, 6 + step, std::span(&anchors[step], 1), &output[id]};
          if (auto r = runner->Wave(std::span(work).first(count)); !r) return r;
          const auto& selected = runner->last_built_policy();
          EXPECT_EQ(selected.gemma_route, 30U);
          EXPECT_EQ(selected.gemma_reduce, 30U);
          EXPECT_GT(selected.norm_rope, 0U);
          EXPECT_GT(selected.norm_add, 0U);
          for (std::uint32_t id = 0; id < count; ++id)
            if (repeat == 0)
              expected[id][step] = output[id];
            else
              Exact(expected[id][step], output[id]);
        }
        for (std::uint32_t id = 0; id < count; ++id) {
          std::vector<std::byte> current;
          if (auto r = StateBytes(id, 9, current); !r) return r;
          if (repeat == 0)
            state[id] = std::move(current);
          else
            EXPECT_EQ(current, state[id]);
        }
      }
      auto ranges = runner->CheckpointRanges(9);
      if (!ranges) return en::support::Error(ranges.error());
      std::uint64_t bytes = 0;
      for (const auto& range : *ranges) bytes += range.bytes;
      std::vector<jitllm::catalog::ExtentId> staging;
      auto saved = node.Pinned(bytes, 0, staging);
      if (!saved) return en::support::Error(saved.error());
      if (auto r = runner->CopyState(0, *saved, *ranges, true); !r) return r;
      std::vector<float> baseline, resumed;
      if (auto r = Single(0, 9, std::span(&anchors[0], 1), baseline); !r) return r;
      if (auto r = runner->Clear(0); !r) return r;
      if (auto r = runner->RestoreCheckpoint(0, 9, *saved, *ranges, runner->CheckpointLayoutId());
          !r)
        return r;
      if (auto r = runner->Spill(0); !r) return r;
      if (auto r = runner->Restore(0); !r) return r;
      runner->DropPlans();
      if (auto r = Single(0, 9, std::span(&anchors[0], 1), resumed); !r) return r;
      Exact(baseline, resumed);
      for (std::uint32_t peer = 1; peer < count; ++peer) {
        std::vector<std::byte> current;
        if (auto r = StateBytes(peer, 9, current); !r) return r;
        EXPECT_EQ(current, state[peer]);
      }
      return node.FreePinned(*saved);
    });
    ASSERT_TRUE(ran) << (ran ? "" : ran.error());
  }
  EXPECT_GT(runner->graph_stats().captured, 0U);
  EXPECT_GT(runner->graph_stats().replayed, 0U);
  EXPECT_EQ(runner->coverage().violations, 0U);
}

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
class Gemma26DefaultNormRunnerGpu : public Gemma4RunnerGpu {
 protected:
  bool DefaultNorms() const override { return true; }
};
class Gemma31DefaultNormRunnerGpu : public Gemma31RunnerGpu {
 protected:
  bool DefaultNorms() const override { return true; }
};
TEST(Gemma4RunnerDefaults, CheckedPlainNormsDoNotEnableOtherExperiments) {
  const en::Gemma4Options options;
  EXPECT_TRUE(options.fuse_norms);
  EXPECT_FALSE(options.shared_q8 || options.row_invariant || options.rope_store ||
               options.fuse_norm_rope || options.fuse_norm_add || options.fuse_gemma_route ||
               options.fuse_gemma_reduce || options.owner_attention);
}
void Gemma4RunnerGpu::StateOnlyControl() {
  ASSERT_TRUE(runner->SelectSlots(std::array<std::uint32_t, 2>{0, 1}));
  auto ranges = runner->CheckpointRanges(7);
  ASSERT_TRUE(ranges);
  std::uint64_t state_bytes = 0;
  for (const auto& range : *ranges) state_bytes += range.bytes;
  const auto host_bytes = 5 * state_bytes + 4ULL * runner->profile().vocab * sizeof(float);
  ASSERT_TRUE(node.ChargeHost(host_bytes, false));
  struct Grant {
    en::PagedNode& node;
    std::uint64_t bytes;
    ~Grant() { node.UnchargeHost(bytes); }
  } grant{node, host_bytes};
  std::vector<float> full, selected;
  std::vector<std::byte> full_state, selected_state, current_state;
  std::array<std::vector<std::byte>, 2> continued_state;
  std::array<std::vector<float>, 2> continued_heads;
  for (unsigned repeat = 0; repeat < 3; ++repeat) {
    auto result = Held([&]() -> en::Status {
      runner->DropPlans();
      if (auto r = runner->Clear(0); !r) return r;
      if (auto r = runner->Clear(1); !r) return r;
      for (std::uint32_t past = 0; past < prompt.size(); past += 2) {
        const auto tokens = std::span(prompt).subspan(past, 2);
        if (auto r = Single(0, past, tokens, full); !r) return r;
        const en::Gemma4Runner::Work work{1, past, tokens, &selected};
        const en::Gemma4Runner::PrefillNext hint{1, 2};
        if (auto r = runner->WavePrefill(std::span(&work, 1), past + 2 == prompt.size(),
                                         std::span(&hint, past + 2 < prompt.size() ? 1U : 0U),
                                         past + 4 == prompt.size());
            !r)
          return r;
        if (past + 2 == prompt.size())
          Exact(full, selected);
        else
          EXPECT_TRUE(selected.empty());
        if (auto r = StateBytes(0, past + 2, full_state); !r) return r;
        if (auto r = StateBytes(1, past + 2, selected_state); !r) return r;
        EXPECT_EQ(full_state, selected_state);
        // A clean pre-dispatch refusal must keep the previously completed output
        // and position. In particular it must not clear a caller's sentinel.
        selected.assign(1, 123.0F);
        const en::Gemma4Runner::Work bad{1, past, tokens, &selected};
        EXPECT_FALSE(runner->WavePrefill(std::span(&bad, 1), false));
        EXPECT_EQ(selected, (std::vector<float>{123.0F}));
        EXPECT_EQ((*runner->request_slot(1))->completed_positions(), past + 2);
      }
      const std::int32_t next = 529;
      if (auto r = Single(0, 6, std::span(&next, 1), full); !r) return r;
      if (auto r = Single(1, 6, std::span(&next, 1), selected); !r) return r;
      Exact(full, selected);
      if (auto r = StateBytes(0, 7, full_state); !r) return r;
      if (auto r = StateBytes(1, 7, selected_state); !r) return r;
      EXPECT_EQ(full_state, selected_state);
      return {};
    });
    ASSERT_TRUE(result) << (result ? "" : result.error());
  }
  // A uniform no-head wave retains independent unequal-row owners. Compare
  // against the same batched rows and product policy, rather than separate solos.
  for (unsigned repeat = 0; repeat < 3; ++repeat) {
    auto result = Held([&]() -> en::Status {
      const std::array<en::Gemma4Runner::Work, 2> work{
          {{0, 0, std::span(prompt).first(2), &full},
           {1, 0, std::span(prompt).first(4), &selected}}};
      if (auto r = runner->Clear(0); !r) return r;
      if (auto r = runner->Clear(1); !r) return r;
      if (auto r = runner->Wave(work); !r) return r;
      if (auto r = StateBytes(0, 2, full_state); !r) return r;
      if (auto r = StateBytes(1, 4, selected_state); !r) return r;
      const std::array<std::int32_t, 2> next{529, 7001};
      const std::array<en::Gemma4Runner::Work, 2> continuation{
          {{0, 2, std::span(next).first(1), &full}, {1, 4, std::span(next).last(1), &selected}}};
      if (auto r = runner->Wave(continuation); !r) return r;
      continued_heads[0] = full;
      continued_heads[1] = selected;
      if (auto r = StateBytes(0, 3, continued_state[0]); !r) return r;
      if (auto r = StateBytes(1, 5, continued_state[1]); !r) return r;
      if (auto r = runner->Clear(0); !r) return r;
      if (auto r = runner->Clear(1); !r) return r;
      runner->DropPlans();
      const auto host_before = node.host_counted();
      std::array<en::Gemma4Runner::PrefillNext, 2> hints{{{0, 1}, {1, 1}}};
      // A bad prediction is discarded without refusing or changing this wave.
      if (repeat == 2) hints[1].slot = en::kMaxRequestSlots;
      if (auto r = runner->WavePrefill(work, false, hints, true); !r) return r;
      EXPECT_TRUE(full.empty());
      EXPECT_TRUE(selected.empty());
      for (std::uint32_t slot = 0; slot < 2; ++slot) {
        if (auto r = StateBytes(slot, slot == 0 ? 2U : 4U, current_state); !r) return r;
        const auto& expected = slot == 0 ? full_state : selected_state;
        EXPECT_EQ(current_state.size(), expected.size());
        if (current_state.size() == expected.size())
          EXPECT_EQ(std::memcmp(current_state.data(), expected.data(), expected.size()), 0);
      }
      // Refuse the entire wave before its valid first owner can advance.
      full.assign(1, 123.0F);
      selected.assign(1, 456.0F);
      auto bad = continuation;
      bad[1].n_past = 3;
      EXPECT_FALSE(runner->WavePrefill(bad, false));
      EXPECT_EQ(full, (std::vector<float>{123.0F}));
      EXPECT_EQ(selected, (std::vector<float>{456.0F}));
      EXPECT_EQ((*runner->request_slot(0))->completed_positions(), 2U);
      EXPECT_EQ((*runner->request_slot(1))->completed_positions(), 4U);
      if (auto r = runner->WavePrefill(continuation, true); !r) return r;
      Exact(full, continued_heads[0]);
      Exact(selected, continued_heads[1]);
      for (std::uint32_t slot = 0; slot < 2; ++slot) {
        if (auto r = StateBytes(slot, slot == 0 ? 3U : 5U, current_state); !r) return r;
        const auto& expected = continued_state[slot];
        EXPECT_EQ(current_state.size(), expected.size());
        if (current_state.size() == expected.size())
          EXPECT_EQ(std::memcmp(current_state.data(), expected.data(), expected.size()), 0);
      }
      runner->DropPlans();
      EXPECT_EQ(node.host_counted(), host_before);
      return {};
    });
    ASSERT_TRUE(result) << (result ? "" : result.error());
  }
  EXPECT_GT(runner->graph_stats().captured, 0U);
  EXPECT_GT(runner->graph_stats().replayed, 0U);
  EXPECT_GT(runner->lookahead_stats().built, 0U);
  EXPECT_EQ(runner->lookahead_stats().built, runner->lookahead_stats().cached);
}
TEST_F(Gemma4RunnerGpu, StateOnlyChunksPreserveExactKvContinuationAndCapturedReplay) {
  StateOnlyControl();
}
TEST_F(Gemma31RunnerGpu, StateOnlyChunksPreserveExactKvContinuationAndCapturedReplay) {
  StateOnlyControl();
}
TEST_F(Gemma26MoeRunnerGpu, StateOnlyChunksPreserveExactKvContinuationAndCapturedReplay) {
  StateOnlyControl();
}
TEST_F(Gemma31NormRunnerGpu, StateOnlyChunksPreserveExactKvContinuationAndCapturedReplay) {
  StateOnlyControl();
}
TEST_F(Gemma26DefaultNormRunnerGpu, StateOnlyChunksPreserveExactKvContinuationAndCapturedReplay) {
  StateOnlyControl();
  EXPECT_GT(runner->last_built_policy().norm_fused, 0U);
}
TEST_F(Gemma31DefaultNormRunnerGpu, StateOnlyChunksPreserveExactKvContinuationAndCapturedReplay) {
  StateOnlyControl();
  EXPECT_GT(runner->last_built_policy().norm_fused, 0U);
}
TEST_F(Gemma31RunnerGpu, LookaheadRefusalAndAbandonedPredictionKeepTheCompletedPrefix) {
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
    const en::Gemma4Runner::PrefillNext hint{0, 2};
    en::Gemma4Runner::Work work{0, 0, std::span(prompt).first(1), &row};
    // Seed the required plan and padded state. Pressure affects only optional
    // capture/lookahead, with no new state backing needed by the next row.
    if (auto r = runner->WavePrefill(std::span(&work, 1), false); !r) return r;
    // The budget full: a wave's host inputs lie beside it, uncharged.
    auto free = node.FreeBytes();
    if (!free || *free == 0)
      return en::support::Error("lookahead control cannot fund input-only pressure");
    const auto fill = *free;
    if (!node.ChargeHost(fill, false))
      return en::support::Error("lookahead control cannot take its pressure charge");
    {
      const Charge pressure{node, fill};
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
      // Greedy waves, fresh state: the device's tokens are the rows' host
      // greedy choices, and the KV bytes are unchanged. Mixed outputs refuse.
      for (std::uint32_t slot = 0; slot < count; ++slot) {
        if (auto r = runner->Clear(slot); !r) return r;
        std::vector<float> ignored;
        if (auto r = Single(slot, 0, prompts[slot], ignored); !r) return r;
      }
      if (count > 1) {
        std::int32_t chosen = -1;
        const std::array<en::Gemma4Runner::Work, 2> mixed{
            en::Gemma4Runner::Work{0, static_cast<std::uint32_t>(prompts[0].size()),
                                   std::span(&anchors[0], 1), &joined[0]},
            en::Gemma4Runner::Work{1, static_cast<std::uint32_t>(prompts[1].size()),
                                   std::span(&anchors[1], 1), nullptr, &chosen}};
        if (runner->Wave(mixed)) return std::unexpected(std::string("a mixed wave ran"));
      }
      for (std::uint32_t step = 0; step < 3; ++step) {
        std::array<std::int32_t, 4> tokens{-1, -1, -1, -1};
        std::array<en::Gemma4Runner::Work, 4> work{};
        for (std::uint32_t slot = 0; slot < count; ++slot)
          work[slot] = {slot, static_cast<std::uint32_t>(prompts[slot].size()) + step,
                        std::span(&anchors[slot], 1), nullptr, &tokens[slot]};
        if (auto r = runner->Wave(std::span(work).first(count)); !r) return r;
        for (std::uint32_t slot = 0; slot < count; ++slot) {
          const auto& row = joined_baseline[slot][step];
          EXPECT_EQ(tokens[slot], std::max_element(row.begin(), row.end()) - row.begin())
              << "count " << count << " slot " << slot << " step " << step;
        }
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
void Gemma4RunnerGpu::HeldSelectionControl() {
  const std::array<std::uint32_t, 2> both{0, 1};
  ASSERT_TRUE(runner->SelectSlots(both));
  ASSERT_FALSE(node.InRequest(0));
  const auto bytes = std::uint64_t{2} * runner->profile().vocab * sizeof(float);
  ASSERT_TRUE(node.ChargeHost(bytes, false));
  struct Grant {
    en::PagedNode& node;
    std::uint64_t bytes;
    ~Grant() { node.UnchargeHost(bytes); }
  } grant{node, bytes};
  auto status = Held([&]() -> en::Status {
    std::vector<float> a, b;
    if (auto r = Single(0, 0, prompt, a); !r) return r;
    if (auto r = Single(1, 0, prompt, b); !r) return r;
    const auto occupancy = node.catalog().OccupancyOf(node.domain()).Total().value();
    for (unsigned i = 0; i < 32; ++i)
      if (auto r = runner->SelectSlots(both); !r) return r;
    EXPECT_FALSE(runner->SelectSlots(std::array<std::uint32_t, 2>{0, 0}));
    EXPECT_FALSE(runner->SelectSlots(std::array<std::uint32_t, 2>{0, 4}));
    EXPECT_EQ(node.catalog().OccupancyOf(node.domain()).Total().value(), occupancy);
    EXPECT_EQ((*runner->request_slot(0))->completed_positions(), prompt.size());
    EXPECT_EQ((*runner->request_slot(1))->completed_positions(), prompt.size());
    // Clear refreshes the held closure; unchanged selection must retain it.
    if (auto r = runner->Clear(0); !r) return r;
    if (auto r = runner->SelectSlots(both); !r) return r;
    if (auto r = Single(0, 0, prompt, a); !r) return r;
    Exact(a, b);
    if (auto r = runner->SelectSlots(std::array<std::uint32_t, 1>{1}); !r) return r;
    a = {17};
    EXPECT_FALSE(Single(0, 6, std::span(prompt).last(1), a));
    EXPECT_EQ(a, std::vector<float>{17});
    if (auto r = runner->SelectSlots(both); !r) return r;
    if (auto r = Single(0, 6, std::span(prompt).last(1), a); !r) return r;
    if (auto r = Single(1, 6, std::span(prompt).last(1), b); !r) return r;
    Exact(a, b);
    return {};
  });
  ASSERT_TRUE(status) << (status ? "" : status.error());
  // The same selection outside a request must still rebuild its closure.
  ASSERT_FALSE(node.InRequest(0));
  ASSERT_TRUE(runner->SelectSlots(both));
}
TEST_F(Gemma4RunnerGpu, HeldSelectionPreservesAdmissionAndRefreshesChangedOwners) {
  HeldSelectionControl();
}
TEST_F(Gemma31RunnerGpu, HeldSelectionPreservesAdmissionAndRefreshesChangedOwners) {
  HeldSelectionControl();
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

class Gemma4HeadCapGpu : public Gemma4RunnerGpu {
 protected:
  std::uint32_t HeadRows() const override { return 4; }
};
class Gemma31HeadCapGpu : public Gemma4HeadCapGpu {
 protected:
  en::Gemma4Variant Variant() const override { return en::Gemma4Variant::k31B; }
};
void Gemma4RunnerGpu::HeadCapacityControl() {
  ASSERT_TRUE(runner->SelectSlots(std::array<std::uint32_t, 4>{0, 1, 2, 3}));
  auto maximum = runner->CheckpointRanges(4);
  ASSERT_TRUE(maximum);
  std::uint64_t state_bytes = 0;
  for (const auto& range : *maximum) state_bytes += range.bytes;
  const auto head_bytes = std::uint64_t{runner->profile().vocab} * sizeof(float);
  const auto host_bytes = 3 * state_bytes + 14 * head_bytes;
  ASSERT_TRUE(node.ChargeHost(host_bytes, false));
  struct Grant {
    en::PagedNode& node;
    std::uint64_t bytes;
    ~Grant() { node.UnchargeHost(bytes); }
  } grant{node, host_bytes};
  std::array<std::vector<float>, 2> output, expected_heads, expected_continued;
  std::array<std::vector<std::byte>, 2> saved;
  std::vector<std::byte> current;
  const std::array<en::Gemma4Runner::Work, 2> boundary{
      {{0, 0, std::span(prompt).first(3), &output[0]},
       {1, 0, std::span(prompt).first(1), &output[1]}}};
  const std::array<en::Gemma4Runner::Work, 2> continuation{
      {{0, 3, std::span(prompt).subspan(3, 1), &output[0]},
       {1, 1, std::span(prompt).subspan(1, 1), &output[1]}}};
  auto result = Held([&]() -> en::Status {
    for (unsigned repeat = 0; repeat < 3; ++repeat) {
      for (std::uint32_t slot = 0; slot < 4; ++slot)
        if (auto r = runner->Clear(slot); !r) return r;
      if (auto r = runner->Wave(boundary, true); !r) return r;
      EXPECT_EQ(output[0].size(), 3U * runner->profile().vocab);
      EXPECT_EQ(output[1].size(), runner->profile().vocab);
      for (std::uint32_t slot = 0; slot < 2; ++slot) {
        if (repeat == 0)
          expected_heads[slot] = output[slot];
        else
          Exact(expected_heads[slot], output[slot]);
        EXPECT_TRUE(std::ranges::all_of(output[slot], [](float v) { return std::isfinite(v); }));
        if (auto r = StateBytes(slot, slot == 0 ? 3U : 1U, saved[slot]); !r) return r;
      }
      // Five heads exceed cap4 although both segments fit max_rows128. The
      // second owner must not cause partial progress in the first or any peer.
      const std::array<en::Gemma4Runner::Work, 2> over{
          {{0, 3, std::span(prompt).first(2), &output[0]},
           {1, 1, std::span(prompt).first(3), &output[1]}}};
      const auto occupancy = node.catalog().OccupancyOf(node.domain()).Total().value();
      const auto counted = node.host_counted();
      const auto charged = node.host_charged();
      const auto plans = runner->plan_count();
      const auto plan_bytes = runner->plans_bytes();
      const auto graphs = runner->graph_count();
      const auto stats = runner->graph_stats();
      const auto extents = runner->state();
      std::array<std::uint64_t, 4> used{};
      for (std::uint32_t slot = 0; slot < 4; ++slot)
        used[slot] = (*runner->request_slot(slot))->used_state_bytes();
      auto refused = runner->Wave(over, true);
      EXPECT_FALSE(refused);
      if (!refused) EXPECT_EQ(refused.error(), "Gemma4 wave exceeds head publication capacity");
      EXPECT_EQ(node.catalog().OccupancyOf(node.domain()).Total().value(), occupancy);
      EXPECT_EQ(node.host_counted(), counted);
      EXPECT_EQ(node.host_charged(), charged);
      EXPECT_EQ(runner->plan_count(), plans);
      EXPECT_EQ(runner->plans_bytes(), plan_bytes);
      EXPECT_EQ(runner->graph_count(), graphs);
      EXPECT_EQ(runner->graph_stats().eager, stats.eager);
      EXPECT_EQ(runner->graph_stats().captured, stats.captured);
      EXPECT_EQ(runner->graph_stats().replayed, stats.replayed);
      EXPECT_EQ(runner->graph_stats().refused, stats.refused);
      EXPECT_EQ(runner->state(), extents);
      for (std::uint32_t slot = 0; slot < 4; ++slot) {
        EXPECT_EQ((*runner->request_slot(slot))->completed_positions(), slot == 0   ? 3U
                                                                        : slot == 1 ? 1U
                                                                                    : 0U);
        EXPECT_EQ((*runner->request_slot(slot))->used_state_bytes(), used[slot]);
      }
      for (std::uint32_t slot = 0; slot < 2; ++slot) {
        Exact(expected_heads[slot], output[slot]);
        if (auto r = StateBytes(slot, slot == 0 ? 3U : 1U, current); !r) return r;
        EXPECT_EQ(current, saved[slot]);
      }
      if (auto r = runner->Wave(continuation); !r) return r;
      for (std::uint32_t slot = 0; slot < 2; ++slot)
        if (repeat == 0)
          expected_continued[slot] = output[slot];
        else
          Exact(expected_continued[slot], output[slot]);
    }
    // Restore the cap-boundary prefixes at the same stable state addresses,
    // then reuse the same joined continuation and captured publication buffer.
    for (std::uint32_t slot = 0; slot < 2; ++slot) {
      if (auto r = runner->Clear(slot); !r) return r;
      auto ranges = runner->CheckpointRanges(slot == 0 ? 3U : 1U);
      if (!ranges) return en::support::Error(ranges.error());
      std::vector<jitllm::catalog::ExtentId> staging;
      auto pinned = node.Pinned(saved[slot].size(), 0, staging);
      if (!pinned) return en::support::Error(pinned.error());
      std::memcpy(*pinned, saved[slot].data(), saved[slot].size());
      en::LiveState::CopyRetirement retirement;
      auto restored = runner->RestoreCheckpoint(slot, slot == 0 ? 3U : 1U, *pinned, *ranges,
                                                runner->CheckpointLayoutId(), &retirement);
      if (retirement == en::LiveState::CopyRetirement::kUnproven) {
        node.KeepPinned(*pinned);
        return restored;
      }
      auto freed = node.FreePinned(*pinned);
      if (!restored) return restored;
      if (!freed) return freed;
    }
    if (auto r = runner->Wave(continuation); !r) return r;
    for (std::uint32_t slot = 0; slot < 2; ++slot) Exact(expected_continued[slot], output[slot]);
    // Full input capacity remains available with only four frontier heads,
    // including ragged owners; the same shape can also omit all heads.
    std::array<std::int32_t, 128> tokens{};
    for (std::size_t i = 0; i < tokens.size(); ++i) tokens[i] = prompt[i % prompt.size()];
    std::array<std::vector<float>, 4> frontier;
    std::array<en::Gemma4Runner::Work, 4> wide{};
    for (std::uint32_t slot = 0; slot < 4; ++slot) {
      if (auto r = runner->Clear(slot); !r) return r;
      wide[slot] = {slot, 0, std::span(tokens).first(slot == 0 ? 125U : 1U), &frontier[slot]};
    }
    if (auto r = runner->Wave(wide); !r) return r;
    for (const auto& head : frontier) EXPECT_EQ(head.size(), runner->profile().vocab);
    for (std::uint32_t slot = 0; slot < 4; ++slot)
      if (auto r = runner->Clear(slot); !r) return r;
    if (auto r = runner->WavePrefill(wide, false); !r) return r;
    for (const auto& head : frontier) EXPECT_TRUE(head.empty());
    return {};
  });
  ASSERT_TRUE(result) << (result ? "" : result.error());
  EXPECT_GT(runner->graph_stats().captured, 0U);
  EXPECT_GT(runner->graph_stats().replayed, 0U);
}
TEST_F(Gemma4HeadCapGpu, RaggedCapBoundaryRefusalReplayRestoreAndMaximumFrontiers) {
  HeadCapacityControl();
}
TEST_F(Gemma31HeadCapGpu, RaggedCapBoundaryRefusalReplayRestoreAndMaximumFrontiers) {
  HeadCapacityControl();
}
TEST_F(Gemma4HeadCapGpu, StateOnlyChunksKeepExactKvAndContinuationUnderReducedHeadCapacity) {
  StateOnlyControl();
}
TEST(Gemma4HeadCapacityGpu, CatalogCountsBothLegacyAndCappedPinnedOutputAllocations) {
  struct Lifetime {
    en::PagedNode node{{.slot_bytes = en::kSlabSlotBytes}};
    std::array<std::unique_ptr<en::Gemma4Runner>, 2> runners;
    std::vector<en::PagedModel*> entered;
  };
  auto owner = std::make_unique<Lifetime>();
  auto& node = owner->node;
  ASSERT_TRUE(node.Open());
  const auto artifact = std::filesystem::path("/home/pmeenan/.local/share/jitllm/m3-artifacts") /
                        "4ddb360c9ce08f1e984ab304b6af918be44246d52346734066b06443f7c249d3";
  std::array<std::uint64_t, 2> charged{};
  const auto staging_class = static_cast<std::size_t>(jitllm::catalog::MemoryClass::kStaging);
  bool ready = true;
  for (std::size_t i = 0; i < owner->runners.size(); ++i) {
    auto& runner = owner->runners[i];
    runner = std::make_unique<en::Gemma4Runner>(
        node,
        en::Gemma4Options{.artifact = artifact, .slots = 4, .max_head_rows = i == 0 ? 0U : 4U}, 0,
        0);
    owner->entered.push_back(runner.get());
    const auto before = node.catalog().OccupancyOf(node.domain()).by_class[staging_class].value();
    const auto setup = runner->Setup();
    EXPECT_TRUE(setup) << (setup ? "" : setup.error());
    if (!setup) {
      ready = false;
      break;
    }
    charged[i] = node.catalog().OccupancyOf(node.domain()).by_class[staging_class].value() - before;
  }
  if (ready) {
    const auto row_bytes = std::uint64_t{owner->runners[0]->profile().vocab} * sizeof(float);
    EXPECT_GE(charged[0], 128 * row_bytes);
    EXPECT_GE(charged[1], 4 * row_bytes);
    EXPECT_LT(charged[1], 128 * row_bytes);
    // All other pinned staging is no larger under the cap. This compares
    // actual catalog backing, not an estimate of resident GPU activation peak.
    EXPECT_GE(charged[0] - charged[1], 124 * row_bytes);
  }
  const auto retired = node.TearDown(owner->entered);
  EXPECT_TRUE(retired) << (retired ? "" : retired.error());
  if (!retired) std::ignore = owner.release();
}
class Gemma4SoloHeadCapGpu : public Gemma4RunnerGpu {
 protected:
  std::uint32_t Slots() const override { return 1; }
  std::uint32_t HeadRows() const override { return 1; }
};
TEST_F(Gemma4SoloHeadCapGpu, SoloFrontierAcceptsMaximumInputsAndRefusesTwoAllHeadRows) {
  ASSERT_TRUE(runner->SelectSlots(std::array<std::uint32_t, 1>{0}));
  const auto host_bytes = std::uint64_t{runner->profile().vocab} * sizeof(float);
  ASSERT_TRUE(node.ChargeHost(host_bytes, false));
  struct Grant {
    en::PagedNode& node;
    std::uint64_t bytes;
    ~Grant() { node.UnchargeHost(bytes); }
  } grant{node, host_bytes};
  auto result = Held([&]() -> en::Status {
    std::vector<float> head{123};
    const en::Gemma4Runner::Work two{0, 0, std::span(prompt).first(2), &head};
    EXPECT_FALSE(runner->Wave(std::span(&two, 1), true));
    EXPECT_EQ(head, (std::vector<float>{123}));
    EXPECT_EQ((*runner->request_slot(0))->completed_positions(), 0U);
    const en::Gemma4Runner::Work one{0, 0, std::span(prompt).first(1), &head};
    if (auto r = runner->Wave(std::span(&one, 1), true); !r) return r;
    EXPECT_EQ(head.size(), runner->profile().vocab);
    if (auto r = runner->Clear(0); !r) return r;
    std::array<std::int32_t, 128> tokens{};
    for (std::size_t i = 0; i < tokens.size(); ++i) tokens[i] = prompt[i % prompt.size()];
    const en::Gemma4Runner::Work maximum{0, 0, tokens, &head};
    if (auto r = runner->Wave(std::span(&maximum, 1)); !r) return r;
    EXPECT_EQ(head.size(), runner->profile().vocab);
    return {};
  });
  ASSERT_TRUE(result) << (result ? "" : result.error());
}

class Gemma4FeatureGpu : public Gemma4RunnerGpu {
 protected:
  bool RetainFeatures() const override { return true; }
};
class Gemma4FeatureCaptureGpu : public Gemma4FeatureGpu {
 protected:
  std::uint32_t Slots() const override { return 2; }
  std::uint32_t MaxRows() const override { return 256; }
  bool CaptureAhead() const override { return true; }
  bool NormChains() const override { return true; }
  bool MoeChains() const override { return true; }
};
TEST_F(Gemma4FeatureCaptureGpu, CapturedFrontierMatchesOrdinaryAndFreshRestoreContinuation) {
  ASSERT_TRUE(runner->SelectSlots(std::array<std::uint32_t, 2>{0, 1}));
  auto ranges = runner->CheckpointRanges(769);
  ASSERT_TRUE(ranges);
  std::uint64_t state_bytes = 0;
  for (const auto& range : *ranges) state_bytes += range.bytes;
  const auto feature_bytes = std::uint64_t{runner->profile().width} * sizeof(float);
  const auto host_bytes =
      4 * state_bytes + 10ULL * runner->profile().vocab * sizeof(float) + 8 * feature_bytes;
  ASSERT_TRUE(node.ChargeHost(host_bytes, false));
  struct Grant {
    en::PagedNode& node;
    std::uint64_t bytes;
    ~Grant() { node.UnchargeHost(bytes); }
  } grant{node, host_bytes};
  std::array<std::int32_t, 769> tokens{};
  for (std::size_t i = 0; i < tokens.size(); ++i) tokens[i] = prompt[i % prompt.size()];
  std::array<std::vector<float>, 3> expected_heads, expected_features;
  std::vector<float> next_head, next_feature;
  std::vector<std::byte> expected_state, next_state;
  auto ran = Held([&]() -> en::Status {
    const auto feature = [&](std::uint32_t first, std::vector<float>& values) -> en::Status {
      std::vector<jitllm::catalog::ExtentId> staging;
      auto pinned = node.Pinned(feature_bytes, 0, staging);
      if (!pinned) return en::support::Error(pinned.error());
      if (auto copied = runner->CopyFeatures(0, first, 1, *pinned); !copied) return copied;
      const auto* data = static_cast<const float*>(*pinned);
      values.assign(data, data + runner->profile().width);
      EXPECT_TRUE(std::ranges::all_of(values, [](float v) { return std::isfinite(v); }));
      return node.FreePinned(*pinned);
    };
    for (const bool hinted : {false, true}) {
      for (const auto slot : {0U, 1U})
        if (auto r = runner->Clear(slot); !r) return r;
      runner->DropPlans();
      const auto before = runner->lookahead_stats();
      const auto graphs = runner->graph_stats();
      std::vector<float> head, features;
      for (std::uint32_t chunk = 0; chunk < 3; ++chunk) {
        const auto past = chunk * 256;
        if (auto r = runner->ChunkPrefill(past, std::span(tokens).subspan(past, 256), head, true,
                                          hinted && chunk < 2 ? 256U : 0U, true,
                                          hinted && chunk == 0 ? 256U : 0U, true);
            !r)
          return r;
        if (auto r = feature(past + 255, features); !r) return r;
        EXPECT_EQ(head.size(), runner->profile().vocab);
        EXPECT_TRUE(std::ranges::all_of(head, [](float v) { return std::isfinite(v); }));
        if (!hinted) {
          expected_heads[chunk] = head;
          expected_features[chunk] = features;
        } else {
          Exact(expected_heads[chunk], head);
          Exact(expected_features[chunk], features);
        }
      }
      const auto after = runner->lookahead_stats();
      EXPECT_EQ(after.refused, before.refused);
      EXPECT_EQ(after.dropped_ahead, before.dropped_ahead);
      if (hinted) {
        EXPECT_GT(after.built_pairs, before.built_pairs);
        EXPECT_GT(after.cached_pairs, before.cached_pairs);
        EXPECT_GT(after.captured_ahead, before.captured_ahead);
        EXPECT_GT(runner->graph_stats().replayed, graphs.replayed);
      } else {
        EXPECT_EQ(after.captured_ahead, before.captured_ahead);
        EXPECT_EQ(runner->graph_stats().replayed, graphs.replayed);
      }
      std::vector<std::byte> state;
      if (auto r = StateBytes(0, 768, state); !r) return r;
      if (!hinted)
        expected_state = state;
      else
        EXPECT_EQ(state, expected_state);
      if (hinted) {
        auto borrow = runner->BorrowFrozen(0, tokens[767]);
        if (!borrow) return en::support::Error(borrow.error());
        EXPECT_EQ(borrow->prefix(), 768U);
        EXPECT_FALSE(runner->Clear(0));
        EXPECT_FALSE(runner->Spill(0));
        std::vector<float> sentinel{123}, peer;
        EXPECT_FALSE(runner->Chunk(768, std::span(tokens).last(1), sentinel));
        EXPECT_EQ(sentinel, (std::vector<float>{123}));
        if (auto r = Single(1, 0, prompt, peer); !r) return r;
        if (auto r = runner->CheckBorrow(*borrow); !r) return r;
        if (auto r = feature(767, features); !r) return r;
        Exact(features, expected_features[2]);
        if (auto r = StateBytes(0, 768, state); !r) return r;
        EXPECT_EQ(state, expected_state);
        *borrow = en::Gemma4Runner::FrozenBorrow{};
        if (auto r = runner->Spill(0); !r) return r;
        if (auto r = runner->Restore(0); !r) return r;
        EXPECT_FALSE(runner->BorrowFrozen(0, tokens[767]));
        if (auto r = StateBytes(0, 768, state); !r) return r;
        EXPECT_EQ(state, expected_state);
      }
      if (auto r = runner->Chunk(768, std::span(tokens).last(1), head); !r) return r;
      if (auto r = feature(768, features); !r) return r;
      if (auto r = StateBytes(0, 769, state); !r) return r;
      if (!hinted) {
        next_head = head;
        next_feature = features;
        next_state = state;
      } else {
        Exact(head, next_head);
        Exact(features, next_feature);
        EXPECT_EQ(state, next_state);
        auto fresh = runner->BorrowFrozen(0, tokens.back());
        if (!fresh) return en::support::Error(fresh.error());
        EXPECT_EQ(fresh->prefix(), 769U);
        EXPECT_EQ((*runner->request_slot(1))->completed_positions(), prompt.size());
      }
    }
    return {};
  });
  ASSERT_TRUE(ran) << (ran ? "" : ran.error());
}
class Gemma4FeatureHeadCapGpu : public Gemma4FeatureGpu {
 protected:
  std::uint32_t HeadRows() const override { return 4; }
};
TEST_F(Gemma4FeatureHeadCapGpu, FullFeatureRowsRemainFundedAndNoHeadHintUsesFrontierCapacity) {
  ASSERT_TRUE(runner->SelectSlots(std::array<std::uint32_t, 4>{0, 1, 2, 3}));
  const auto feature_bytes = std::uint64_t{125} * runner->profile().width * sizeof(float);
  const auto host_bytes = std::uint64_t{4} * runner->profile().vocab * sizeof(float);
  ASSERT_TRUE(node.ChargeHost(host_bytes, false));
  struct Grant {
    en::PagedNode& node;
    std::uint64_t bytes;
    ~Grant() { node.UnchargeHost(bytes); }
  } grant{node, host_bytes};
  std::vector<jitllm::catalog::ExtentId> staging;
  auto pinned = node.Pinned(feature_bytes, 0, staging);
  ASSERT_TRUE(pinned);
  auto result = Held([&]() -> en::Status {
    std::array<std::int32_t, 128> tokens{};
    for (std::size_t i = 0; i < tokens.size(); ++i) tokens[i] = prompt[i % prompt.size()];
    std::array<std::vector<float>, 4> heads;
    std::array<en::Gemma4Runner::Work, 4> work{};
    for (std::uint32_t slot = 0; slot < 4; ++slot)
      work[slot] = {slot, 0, std::span(tokens).first(slot == 0 ? 125U : 1U), &heads[slot]};
    if (auto r = runner->Wave(work, false, true); !r) return r;
    auto copied = runner->CopyFeatures(0, 0, 125, *pinned);
    if (!copied) return copied;
    const auto* values = static_cast<const float*>(*pinned);
    EXPECT_TRUE(std::all_of(values, values + feature_bytes / sizeof(float),
                            [](float v) { return std::isfinite(v); }));
    for (std::uint32_t slot = 0; slot < 4; ++slot) {
      if (auto r = runner->Clear(slot); !r) return r;
    }
    if (auto r = runner->WavePrefill(work, false); !r) return r;
    for (std::uint32_t slot = 0; slot < 4; ++slot) {
      EXPECT_EQ(heads[slot].size(), runner->profile().vocab);
      auto borrowed = runner->BorrowFrozen(slot, tokens[(slot == 0 ? 125U : 1U) - 1]);
      if (!borrowed) return en::support::Error(borrowed.error());
      EXPECT_EQ(borrowed->prefix(), slot == 0 ? 125U : 1U);
    }
    return {};
  });
  ASSERT_TRUE(result) << (result ? "" : result.error());
  ASSERT_TRUE(node.FreePinned(*pinned));
}
TEST_F(Gemma4FeatureGpu, PrefillWithoutAHeadRequestStillPublishesRequiredFeatures) {
  ASSERT_TRUE(runner->SelectSlots(std::array<std::uint32_t, 1>{0}));
  const auto bytes = std::uint64_t{runner->profile().vocab} * sizeof(float);
  ASSERT_TRUE(node.ChargeHost(bytes, false));
  struct Grant {
    en::PagedNode& node;
    std::uint64_t bytes;
    ~Grant() { node.UnchargeHost(bytes); }
  } grant{node, bytes};
  auto status = Held([&]() -> en::Status {
    std::vector<float> head;
    const en::Gemma4Runner::Work work{0, 0, prompt, &head};
    if (auto r = runner->WavePrefill(std::span(&work, 1), false); !r) return r;
    EXPECT_EQ(head.size(), runner->profile().vocab);
    EXPECT_TRUE(std::ranges::all_of(head, [](float x) { return std::isfinite(x); }));
    auto borrowed = runner->BorrowFrozen(0, prompt.back());
    if (!borrowed) return en::support::Error(borrowed.error());
    EXPECT_EQ(borrowed->prefix(), prompt.size());
    return {};
  });
  ASSERT_TRUE(status) << (status ? "" : status.error());
}
TEST_F(Gemma4FeatureGpu, FrozenFeatureSurvivesPeerProgressPlansAndRejectsSameSlotMutation) {
  ASSERT_TRUE(runner->SelectSlots(std::array<std::uint32_t, 2>{0, 1}));
  const auto feature_bytes = std::uint64_t{2} * runner->profile().width * sizeof(float);
  const auto host_bytes =
      std::uint64_t{3} * runner->profile().vocab * sizeof(float) + feature_bytes * 2;
  ASSERT_TRUE(node.ChargeHost(host_bytes, false));
  struct Grant {
    en::PagedNode& node;
    std::uint64_t bytes;
    ~Grant() { node.UnchargeHost(bytes); }
  } grant{node, host_bytes};
  std::vector<jitllm::catalog::ExtentId> staging;
  auto saved = node.Pinned(feature_bytes, 0, staging);
  ASSERT_TRUE(saved);
  auto status = Held([&]() -> en::Status {
    EXPECT_FALSE(runner->BorrowFrozen(0, 2));
    std::vector<float> head, peer, sentinel{123};
    const en::Gemma4Runner::Work first{0, 0, std::span(prompt).first(2), &head};
    if (auto r = runner->Wave(std::span(&first, 1), false, true); !r) return r;
    EXPECT_EQ(head.size(), 262144U);
    if (auto r = runner->CopyFeatures(0, 0, 2, *saved); !r) return r;
    const auto* values = static_cast<const float*>(*saved);
    std::vector<float> expected(values, values + feature_bytes / sizeof(float));
    EXPECT_TRUE(std::ranges::all_of(expected, [](float v) { return std::isfinite(v); }));
    auto borrowed = runner->BorrowFrozen(0, 818);
    if (!borrowed) return en::support::Error(borrowed.error());
    EXPECT_EQ(borrowed->prefix(), 2U);
    EXPECT_FALSE(runner->BorrowFrozen(0, 818));
    EXPECT_FALSE(runner->BorrowFrozen(1, -1));
    EXPECT_FALSE(runner->Clear(0));
    EXPECT_FALSE(runner->ClearIdle(0));
    EXPECT_FALSE(runner->Spill(0));
    EXPECT_FALSE(runner->ReserveStateThrough(0, 3));
    if (auto r = runner->SelectSlots(std::array<std::uint32_t, 2>{0, 1}); !r) return r;
    EXPECT_FALSE(runner->SelectSlots(std::array<std::uint32_t, 1>{1}));
    const en::Gemma4Runner::Work forbidden{0, 2, std::span(prompt).first(1), &sentinel};
    EXPECT_FALSE(runner->Wave(std::span(&forbidden, 1)));
    EXPECT_EQ(sentinel, (std::vector<float>{123}));
    if (auto r = Single(1, 0, std::span(prompt).first(2), peer); !r) return r;
    if (auto r = runner->CheckBorrow(*borrowed); !r) return r;
    runner->DropPlans();
    if (auto r = runner->CopyFeatures(0, 0, 2, *saved); !r) return r;
    EXPECT_EQ(std::memcmp(saved.value(), expected.data(), feature_bytes), 0);
    if (auto r = runner->CheckBorrow(*borrowed); !r) return r;
    // Release only the scoped guard; completed target bytes remain intact.
    *borrowed = en::Gemma4Runner::FrozenBorrow{};
    auto again = runner->BorrowFrozen(0, 818);
    if (!again) return en::support::Error(again.error());
    *again = en::Gemma4Runner::FrozenBorrow{};
    if (auto r = runner->Spill(0); !r) return r;
    if (auto r = runner->Restore(0); !r) return r;
    EXPECT_FALSE(runner->BorrowFrozen(0, 818));  // no feature serialized with KV
    if (auto r = Single(0, 2, std::span(prompt).subspan(2, 1), head); !r) return r;
    auto restored = runner->BorrowFrozen(0, 818);
    if (!restored) return en::support::Error(restored.error());
    EXPECT_EQ(restored->prefix(), 3U);
    *restored = en::Gemma4Runner::FrozenBorrow{};
    const auto footprint = (*runner->request_slot(0))->state().used_ranges();
    if (auto r = runner->PrepareRestore(0, 3, footprint, runner->CheckpointLayoutId()); !r)
      return r;
    EXPECT_FALSE(runner->BorrowFrozen(0, 818));  // before any restoring copy
    EXPECT_FALSE(runner->CopyFeatures(0, 2, 1, *saved));
    if (auto r = runner->Clear(0); !r) return r;
    EXPECT_FALSE(runner->BorrowFrozen(0, 818));
    EXPECT_EQ((*runner->request_slot(1))->completed_positions(), 2U);
    return {};
  });
  ASSERT_TRUE(status) << (status ? "" : status.error());
  ASSERT_TRUE(node.FreePinned(*saved));
}

TEST_F(Gemma4FeatureGpu, FailedFeatureCopyRetainsItsNodeOwnedPinnedDestination) {
  constexpr auto host_bytes = std::uint64_t{2} << 20U;
  ASSERT_TRUE(node.ChargeHost(host_bytes, false));
  struct Grant {
    en::PagedNode& node;
    ~Grant() { node.UnchargeHost(host_bytes); }
  } grant{node};
  ASSERT_TRUE(runner->SelectSlots(std::array<std::uint32_t, 1>{0}));
  std::vector<float> head;
  ASSERT_TRUE(Single(0, 0, prompt, head));
  std::vector<jitllm::catalog::ExtentId> staging;
  auto pinned = node.Pinned(2816 * sizeof(float), 0, staging);
  ASSERT_TRUE(pinned);
  std::memset(*pinned, 0x5a, 2816 * sizeof(float));
  ASSERT_TRUE(node.Evict(runner->weights()));
  auto free = node.FreeBytes();
  ASSERT_TRUE(free);
  ASSERT_GT(*free, 16U << 20U);
  const auto pressure = *free - (16U << 20U);
  ASSERT_TRUE(node.ChargeHost(pressure, false));
  const auto kept = node.kept_pinned();
  auto copied = runner->CopyFeatures(0, 5, 1, *pinned);
  node.UnchargeHost(pressure);
  EXPECT_FALSE(copied);
  EXPECT_EQ(node.kept_pinned(), kept + 1);
  EXPECT_FALSE(node.FreePinned(*pinned));
  EXPECT_TRUE(
      std::ranges::all_of(std::span(static_cast<const std::byte*>(*pinned), 2816 * sizeof(float)),
                          [](std::byte b) { return b == std::byte{0x5a}; }));
  auto slot = runner->request_slot(0);
  ASSERT_TRUE(slot);
  EXPECT_TRUE((*slot)->state_usable());
  EXPECT_EQ((*slot)->completed_positions(), 6U);
}
