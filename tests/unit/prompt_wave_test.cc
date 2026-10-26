// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "engine/prompt_wave.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <vector>

#include "base/bytes.h"
#include "engine/batch_inventory.h"
#include "engine/decoding.h"
#include "engine/paged_node.h"
#include "engine/prefill.h"
#include "engine/support.h"
#include "expected_error.h"
#include "fake_step_source.h"
#include "kernels/common/recording_launch.h"
#include "model/state_contract.h"
#include "providers/device_runtime.h"
#include "providers/fake/fake_device_runtime.h"

namespace {
namespace en = llmp::engine;
namespace md = llmp::model;
namespace pl = llmp::planner;
namespace ts = llmp::test_support;
using C = md::StateTensorContract;
constexpr std::uint32_t kContext = 8192, kRows = 8;
constexpr std::uint64_t kExtent = en::kPagedExtent;

std::vector<en::StageState> Stages() {
  const auto layout = [](std::vector<C> tensors) {
    const std::vector<std::pair<std::uint32_t, md::StateContract>> layers{
        {0, {.tensors = std::move(tensors)}}};
    return md::Layout(layers, kContext, kRows).value();
  };
  return {{.name = "target", .layout = layout({{.name = "kv", .cell_bytes = 1024}})},
          {.name = "next",
           .layout = layout({{.name = "kv", .cell_bytes = 1024, .replayable = true},
                             {.name = "tap",
                              .kind = md::StateKind::kFeatures,
                              .encoding = md::Encoding::kF32,
                              .cell_bytes = 256,
                              .cells = md::CellRule::kPerRow,
                              .row_carry = 1,
                              .rollback = md::Rollback::kSnapshot,
                              .target_writes = true}})}};
}

class PromptWaveTest : public ::testing::Test {
 protected:
  en::PagedNode node_{
      en::NodeSettings{.compute_streams = 1, .slots = 2, .copy_lane = true, .slot_bytes = kExtent}};
  std::unique_ptr<en::ModelRunner> runner_;
  ts::FakeStepSource* source_ = nullptr;
  std::optional<en::WaveComposer> composer_;
  std::array<std::vector<std::int32_t>, 2> history_;
  std::array<std::vector<ts::FakeRow>, 2> drafts_;
  std::uint32_t keep_ = 3;
  bool discard_ = false;

  en::SlotLifecycle& slots() { return runner_->slots(); }
  std::int32_t Token(std::uint32_t slot, std::uint32_t q) {
    return static_cast<std::int32_t>(1000 * (slot + 1) + q);
  }
  void History(std::uint32_t slot, std::uint32_t through) {
    auto& h = history_[slot];
    while (h.size() <= through) h.push_back(Token(slot, static_cast<std::uint32_t>(h.size())));
  }
  virtual void Configure(en::RunnerConfig&) {}
  virtual void ConfigureBatching(md::Batching&) {}
  virtual bool BorrowsTarget() const { return false; }
  virtual void CheckDraftBorrow(std::uint32_t) {}
  virtual std::vector<en::StageState> StateStages() const { return Stages(); }
  virtual std::uint64_t BudgetExtents() const { return 64; }
  void SetUp() override {
    // NOLINTNEXTLINE(concurrency-mt-unsafe): test environment is immutable
    const char* scratch = std::getenv("LLMP_TEST_SCRATCH");
    ASSERT_NE(scratch, nullptr);
    std::filesystem::create_directories(scratch);
    ASSERT_TRUE(node_.Open());
    const auto fixed = node_.catalog().OccupancyOf(node_.domain()).Total().value();
    auto stages = StateStages();
    ts::FakeSourceOptions options;
    for (const auto& stage : stages) options.layouts.push_back(stage.layout);
    options.features = true;
    options.extra_inputs = [](const pl::StepShape& shape) {
      std::uint64_t rows = 0;
      for (const auto& segment : shape.segments) rows += segment.rows;
      return shape.stage == 1
                 ? std::vector<llmp::kernels::InputSlot>{{.bytes = rows * 4,
                                                          .kind =
                                                              llmp::kernels::InputKind::kSideRows,
                                                          .component = 17}}
                 : std::vector<llmp::kernels::InputSlot>{};
    };
    options.side_rows = [](std::uint16_t stage, std::uint16_t component,
                           std::span<const std::int32_t> history, std::uint32_t position,
                           std::span<std::int32_t> out) -> en::Status {
      if (stage != 1 || component != 17 || out.size() != 1 || position >= history.size())
        return std::unexpected("wrong shifted lookup coordinates");
      out[0] = history[position];
      return {};
    };
    auto source = std::make_unique<ts::FakeStepSource>(std::move(options));
    source_ = source.get();
    en::RunnerConfig config;
    config.slots = {.model = "Repeat",
                    .slots = 2,
                    .stages = stages,
                    .spill_place = {},
                    .spill_directory = scratch,
                    .invalidate = {},
                    .drop_plans = {}};
    for (std::uint16_t stage = 0; stage < stages.size(); ++stage) {
      for (const auto phase : {pl::Phase::kPrefill, pl::Phase::kVerify, pl::Phase::kDraft}) {
        if ((stage == 0) == (phase == pl::Phase::kDraft)) continue;
        pl::StepShape shape{.stage = stage, .phase = phase, .segments = {}, .read_align = 256};
        for (std::uint8_t slot = 0; slot < 2; ++slot)
          shape.segments.push_back({.slot = slot,
                                    .rows = kRows,
                                    .past = kContext - kRows,
                                    .read_cells = kContext,
                                    .outputs = kRows});
        config.setup_shapes.push_back(std::move(shape));
      }
    }
    config.launch = [](std::uint64_t)
        -> std::expected<std::unique_ptr<llmp::kernels::StepLaunch>, std::string> {
      return std::make_unique<llmp::kernels::RecordingLaunch>();
    };
    config.max_verify_rows = 4;
    Configure(config);
    auto made = en::ModelRunner::Create(node_, std::move(config), std::move(source), 0, 0);
    ASSERT_TRUE(made) << ts::Failed(made).value_or("");
    runner_ = std::move(*made);
    auto set = runner_->Setup();
    ASSERT_TRUE(set) << ts::Failed(set).value_or("");
    ASSERT_TRUE(node_.MapWorkspace(runner_->activations_needed(),
                                   std::max(kExtent, runner_->pool_needed())));
    ASSERT_TRUE(node_.Start(llmp::base::Bytes(fixed + BudgetExtents() * kExtent)));
    ASSERT_TRUE(runner_->Register());
    node_.Run();
    ASSERT_TRUE(runner_->Bind());
    ASSERT_TRUE(slots().SelectSlots(std::array<std::uint32_t, 2>{0, 1}));
    ASSERT_TRUE(node_.BeginRequest(0, slots().execution(), "repeat tests"));
    md::Batching caps;
    caps.max_segments = 2;
    ConfigureBatching(caps);
    auto composer = en::WaveComposer::For(caps, slots());
    ASSERT_TRUE(composer);
    composer_.emplace(std::move(*composer));
  }
  void TearDown() override {
    llmp::providers::fake::ClearCopyFailure();
    if (!runner_) return;
    if (node_.InRequest(0)) EXPECT_TRUE(node_.EndRequest(0));
    std::array<en::PagedModel*, 1> models{runner_.get()};
    auto closed = node_.TearDown(models);
    EXPECT_TRUE(closed) << ts::Failed(closed).value_or("");
  }
  en::WaveReport Prompt(std::span<const std::uint32_t> owners, std::uint32_t count) {
    std::vector<en::PendingWork> pending;
    std::vector<en::SegmentWork> work;
    std::vector<std::span<const std::int32_t>> histories;
    for (const auto slot : owners) {
      const auto past = slots().positions(slot);
      History(slot, past + count);
      pending.push_back({.slot = slot,
                         .stage = 0,
                         .phase = pl::Phase::kPrefill,
                         .past = past,
                         .rows = count,
                         .outputs = 1});
      work.push_back({.tokens = std::span(history_[slot]).subspan(past, count)});
      histories.push_back(history_[slot]);
    }
    return en::RunPromptWave(*runner_, *composer_, pending, work, histories);
  }
  void Prefill(std::uint32_t slot, std::uint32_t count) {
    const std::array owners{slot};
    while (count != 0) {
      const auto take = std::min(count, kRows);
      auto ran = Prompt(owners, take);
      ASSERT_FALSE(ran.outcomes.front().refusal) << ran.outcomes.front().refusal->what;
      count -= take;
    }
  }
  en::DecodingOrchestrator Decoder() {
    en::DecodingMode mode{.kind = en::DecodingMode::Kind::kSpeculative,
                          .drafters = {{.stage = 1,
                                        .max_depth = 3,
                                        .catch_up = true,
                                        .borrows_target = BorrowsTarget(),
                                        .features = true}}};
    en::DecodingPolicy policy;
    policy.propose = [this](
                         std::uint32_t slot, std::size_t, std::span<const std::byte> bytes,
                         std::uint32_t) -> std::expected<std::vector<std::int32_t>, std::string> {
      CheckDraftBorrow(slot);
      drafts_[slot].emplace_back();
      std::memcpy(&drafts_[slot].back(), bytes.data(), sizeof(ts::FakeRow));
      const auto& staged = source_->options().seen_inputs->at(0);
      std::vector<std::int32_t> ids(staged.size() / sizeof(std::int32_t));
      std::memcpy(ids.data(), staged.data(), staged.size());
      EXPECT_NE(std::ranges::find(ids, drafts_[slot].back().token), ids.end());
      return std::vector<std::int32_t>{drafts_[slot].back().token + 1};
    };
    policy.judge = [this](std::uint32_t, std::span<const std::int32_t>,
                          std::span<const std::byte>) -> std::expected<en::Verdict, std::string> {
      if (discard_) return std::unexpected("discard test");
      return en::Verdict{.keep = keep_};
    };
    return en::DecodingOrchestrator::Make(*runner_, *composer_, std::move(mode), std::move(policy))
        .value();
  }
  en::DecodeReport Decode(en::DecodingOrchestrator& decoder,
                          std::span<const std::uint32_t> owners) {
    std::vector<en::DecodeWork> work;
    for (const auto slot : owners) {
      History(slot, slots().positions(slot));
      drafts_[slot].clear();
      work.push_back(
          {.slot = slot, .history = std::span(history_[slot]).first(slots().positions(slot) + 1)});
    }
    return decoder.Step(work);
  }
};

class JoinedPromptPromptWaveTest : public PromptWaveTest {
 protected:
  void ConfigureBatching(md::Batching& caps) override { caps.join_prefill = true; }
};

TEST_F(JoinedPromptPromptWaveTest, JoinedTargetWavePublishesBothOwners) {
  const std::array owners{0U, 1U};
  (void)node_.TakeTimes(0);
  bool joined_target = false;
  source_->options().on_build = [&](const pl::StepShape& shape, bool measuring) {
    if (!measuring && shape.stage == 0) joined_target = shape.segments.size() == 2;
  };
  auto report = Prompt(owners, 1);
  source_->options().on_build = {};
  ASSERT_TRUE(joined_target);
  ASSERT_EQ(report.steps, 1U);
  EXPECT_EQ(node_.TakeTimes(0).steps, report.steps);
  for (const auto owner : owners) {
    ASSERT_FALSE(report.outcomes[owner].refusal) << report.outcomes[owner].refusal->what;
    EXPECT_EQ(slots().positions(owner), 1U);
  }
}

TEST_F(JoinedPromptPromptWaveTest, RejectedJoinedTargetNeverPublishes) {
  const std::array owners{0U, 1U};
  source_->options().fail_phase = pl::Phase::kPrefill;
  source_->options().fail_run = llmp::kernels::StepError::kRejected;
  auto refused = Prompt(owners, 1);
  for (const auto owner : owners) {
    ASSERT_TRUE(refused.outcomes[owner].refusal);
    EXPECT_EQ(slots().positions(owner), 0U);
  }
  EXPECT_EQ(node_.kept_pinned(), 0U);
}

class OutputPaymentPromptWaveTest : public PromptWaveTest {
 protected:
  void Configure(en::RunnerConfig&) override {
    // Size/measure actual outputs during normal Setup, rather than changing
    // an installed plan's advertised result after admission.
    source_->options().logits = 16384;
  }
};

TEST_F(OutputPaymentPromptWaveTest, OutputPaymentRefusalPrecedesAnyProgramOrStateSubmission) {
  const std::array owners{0U};
  auto warm = Prompt(owners, 1);
  ASSERT_FALSE(warm.outcomes[0].refusal) << warm.outcomes[0].refusal->what;
  ASSERT_TRUE(slots().Clear(0));  // Keeps stable places and cached plans for the exact retry shape.
  warm = {};
  node_.SetHostFloor(0);
  ASSERT_TRUE(node_.TryChargeHost(0));
  auto free = node_.FreeBytes();
  ASSERT_TRUE(free);
  const auto counted = node_.host_counted();
  const auto limit = (node_.host_charged() + *free) / kExtent * kExtent;
  constexpr std::uint64_t kPermissionHeadroom = 16U << 10U;
  ASSERT_GT(limit, counted + kPermissionHeadroom);
  // Existing permission descriptors fit; the actual 64 KiB output does not.
  const auto filler = limit - counted - kPermissionHeadroom;
  ASSERT_TRUE(node_.ChargeHost(filler, false));
  const auto executed = source_->executed();
  const auto positions = slots().positions(0);
  auto refused = Prompt(owners, 1);  // Real companion authority, no fabricated StepRequest.
  node_.UnchargeHost(filler);
  ASSERT_TRUE(refused.outcomes[0].refusal);
  EXPECT_TRUE(refused.outcomes[0].refusal->clean);
  EXPECT_NE(refused.outcomes[0].refusal->what.find("owned host result"), std::string::npos);
  EXPECT_EQ(source_->executed(), executed);
  EXPECT_EQ(slots().positions(0), positions);
  EXPECT_TRUE(slots().usable(0));
}

TEST_F(PromptWaveTest, RetainedOutputsSurviveLaterStepsAndResultMoves) {
  const std::array owners{std::uint32_t{0}, std::uint32_t{1}};
  auto first = Prompt(owners, 2);
  ASSERT_EQ(first.outcomes.size(), 2U);
  ASSERT_FALSE(first.outcomes[0].refusal);
  auto retained = first.outcomes[0].outputs;
  const std::vector<std::byte> snapshot(retained.begin(), retained.end());
  const auto paid = retained.paid_bytes();
  ASSERT_GT(paid, 0U);
  auto moved = std::move(first);
  moved = {};
  auto later = Prompt(owners, 2);
  ASSERT_EQ(later.outcomes.size(), 2U);
  ASSERT_FALSE(later.outcomes[0].refusal);
  EXPECT_TRUE(std::ranges::equal(retained, snapshot));
  EXPECT_FALSE(std::ranges::equal(retained, later.outcomes[0].outputs));
  EXPECT_EQ(retained.paid_bytes(), paid);
}

TEST_F(PromptWaveTest, FeatureCopyBeforeARejectedDraftRetiresBeforePeerContinues) {
  const std::array owners{0U};
  Prefill(0, 7);
  auto decoder = Decoder();
  decoder.Reset(0);
  const auto executed = source_->executed();
  source_->options().fail_run = llmp::kernels::StepError::kRejected;
  const auto refused = Decode(decoder, owners);
  ASSERT_TRUE(refused.outcomes[0].refusal);
  EXPECT_EQ(source_->executed(), executed);  // no rejected program body ran
  EXPECT_EQ(slots().positions(0), 7U);
  bool copied = false;
  runner_->plans().ForEach([&](const auto& entry) {
    if (entry.key.shape.stage != 1) return;
    for (const auto& input : entry.planned->program->inputs()) {
      if (input.kind != llmp::kernels::InputKind::kFeatures) continue;
      ASSERT_GE(input.bytes, sizeof(ts::FakeCell));
      ts::FakeCell first;
      const auto* address = reinterpret_cast<const void*>(input.device_address);
      std::memcpy(&first, address, sizeof(first));
      EXPECT_EQ(first.token, Token(0, 6));
      EXPECT_EQ(first.position, 6);
      copied = true;
    }
  });
  EXPECT_TRUE(copied);  // actual asynchronous feature payload reached its input
  source_->options().fail_phase.reset();
  source_->options().fail_run.reset();
  const std::array peer{1U};
  auto prompt = Prompt(peer, 7);
  ASSERT_FALSE(prompt.outcomes[0].refusal) << prompt.outcomes[0].refusal->what;
}

// A plain target step (here a certain span) commits rows the MTP drafter's
// cursor never followed. Its next catch-up starts at the oldest feature the
// target's tap still holds, so drafting resumes rather than being refused
// for the rest of the reply.
TEST_F(PromptWaveTest, DraftingResumesAfterACertainSpan) {
  const std::array owners{0U};
  Prefill(0, 7);
  auto decoder = Decoder();
  decoder.Reset(0);
  const auto speculated = Decode(decoder, owners);
  ASSERT_FALSE(speculated.outcomes[0].refusal) << speculated.outcomes[0].refusal->what;
  ASSERT_EQ(speculated.outcomes[0].drafts.size(), 1U);
  EXPECT_FALSE(speculated.outcomes[0].drafts[0].refusal);
  EXPECT_GT(speculated.outcomes[0].drafts[0].proposed, 0U);
  // A forced span: the anchor and three certain tokens, one target step.
  const auto past = slots().positions(0);
  History(0, past + 4);
  const auto certain = std::span(history_[0]).subspan(past + 1, 3);
  const std::array work{en::DecodeWork{
      .slot = 0, .history = std::span(history_[0]).first(past + 1), .certain = certain}};
  const auto forced = decoder.Step(work);
  ASSERT_FALSE(forced.outcomes[0].refusal) << forced.outcomes[0].refusal->what;
  EXPECT_FALSE(forced.speculated);
  ASSERT_EQ(slots().positions(0), past + 4);
  const auto resumed = Decode(decoder, owners);
  ASSERT_FALSE(resumed.outcomes[0].refusal) << resumed.outcomes[0].refusal->what;
  ASSERT_EQ(resumed.outcomes[0].drafts.size(), 1U);
  EXPECT_FALSE(resumed.outcomes[0].drafts[0].refusal)
      << resumed.outcomes[0].drafts[0].refusal->what;
  EXPECT_GT(resumed.outcomes[0].drafts[0].proposed, 0U);
  // Its first pass caught up from the forced span's first row: the oldest
  // feature its tap held (the row before it, the tap's carry).
  EXPECT_EQ(decoder.cursor(0, 0), past + 4);
  ASSERT_FALSE(drafts_[0].empty());
}

}  // namespace
