// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "engine/draft_companions.h"

#include <gtest/gtest.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <format>
#include <memory>
#include <optional>
#include <span>
#include <vector>

#include "base/bytes.h"
#include "engine/decoding.h"
#include "engine/paged_node.h"
#include "engine/prefill.h"
#include "engine/support.h"
#include "expected_error.h"
#include "fake_step_source.h"
#include "kernels/common/recording_launch.h"
#include "model/state_contract.h"
#include "platform/kept_files.h"
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
           .layout = layout(
               {{.name = "kv", .cell_bytes = 1024, .replayable = true},
                {.name = "tap",
                 .kind = md::StateKind::kFeatures,
                 .encoding = md::Encoding::kF32,
                 .cell_bytes = 256,
                 .cells = md::CellRule::kPerRow,
                 .window_or_ratio = 1,
                 .rollback = md::Rollback::kSnapshot,
                 .target_writes = true},
                md::DraftProgressControl("progress", md::DraftRelation::kTokenAfterCache)})}};
}

class DraftCompanionsTest : public ::testing::Test {
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
  virtual bool JoinedQualified() const { return true; }
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
    auto source = std::make_unique<ts::FakeStepSource>(std::move(options), JoinedQualified());
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
    config.launch = []() -> std::expected<std::unique_ptr<llmp::kernels::StepLaunch>, std::string> {
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
  std::vector<std::byte> Read(std::uint32_t slot, std::span<const en::LiveState::Range> ranges) {
    std::uint64_t bytes = 0;
    for (const auto& range : ranges) bytes += range.bytes;
    std::vector<llmp::catalog::ExtentId> staging;
    auto host = node_.Pinned(bytes, 0, staging);
    EXPECT_TRUE(host);
    if (!host) return {};
    auto copied = slots().CopyState(slot, *host, ranges, true);
    EXPECT_TRUE(copied) << ts::Failed(copied).value_or("");
    if (!copied) return {};
    auto* begin = static_cast<const std::byte*>(*host);
    return {begin, begin + bytes};
  }
  struct Checkpoint {
    std::vector<en::LiveState::Range> footprint, pages;
    void* host = nullptr;
  };
  Checkpoint Save(std::uint32_t slot) {
    Checkpoint out;
    for (std::size_t stage = 0; stage < slots().stages(); ++stage)
      for (auto range : slots().live(slot, stage).used_ranges()) {
        range.region = stage;
        out.footprint.push_back(range);
      }
    auto pages = slots().CheckpointRanges(slot, slots().positions(slot));
    EXPECT_TRUE(pages);
    if (!pages) return out;
    out.pages = std::move(*pages);
    std::uint64_t bytes = 0;
    for (const auto& page : out.pages) bytes += page.bytes;
    std::vector<llmp::catalog::ExtentId> staging;
    auto host = node_.Pinned(bytes, 0, staging);
    EXPECT_TRUE(host);
    if (!host) return out;
    out.host = *host;
    EXPECT_TRUE(slots().CopyState(slot, out.host, out.pages, true));
    return out;
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
    en::DecodingMode mode{
        .kind = en::DecodingMode::Kind::kSpeculative,
        .drafters = {{.stage = 1, .max_depth = 3, .catch_up = true, .features = true}}};
    en::DecodingPolicy policy;
    policy.propose = [this](
                         std::uint32_t slot, std::size_t, std::span<const std::byte> bytes,
                         std::uint32_t) -> std::expected<std::vector<std::int32_t>, std::string> {
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

class UnqualifiedDraftCompanionsTest : public DraftCompanionsTest {
 protected:
  bool JoinedQualified() const override { return false; }
  void Configure(en::RunnerConfig& config) override {
    std::vector<pl::StepShape> scalar;
    for (const auto& shape : config.setup_shapes) {
      for (const auto& segment : shape.segments) {
        auto one = shape;
        one.segments = {segment};
        scalar.push_back(std::move(one));
      }
    }
    config.setup_shapes = std::move(scalar);
  }
  struct State {
    std::vector<llmp::catalog::ExtentId> extents{};
    std::vector<std::byte> bytes{};
    std::uint64_t used = 0;
    md::DraftProgress completed, pending;
    en::DraftProgressLedger::State ledger;
    std::uint32_t positions = 0;
    bool owed = false;
    bool operator==(const State&) const = default;
  };
  State Observe(std::uint32_t slot) {
    const auto* progress = slots().progress(slot, 1);
    State out{.used = slots().used_bytes(slot),
              .completed = progress->completed(),
              .pending = progress->after_owed(),
              .ledger = progress->state(),
              .positions = slots().positions(slot),
              .owed = slots().owed(slot)};
    for (std::size_t stage = 0; stage < slots().stages(); ++stage) {
      const auto& live = slots().live(slot, stage);
      const auto extents = live.extents();
      out.extents.insert(out.extents.end(), extents.begin(), extents.end());
      for (const auto& range : live.used_ranges()) {
        std::vector<llmp::catalog::ExtentId> host_extents;
        auto host = node_.Pinned(range.bytes, 0, host_extents);
        EXPECT_TRUE(host);
        if (!host) return out;
        // Deliberately inspect physical bytes without settling owed work.
        auto copied = node_.Copy(runner_->stream(), slots().execution(),
                                 live.base(range.region) + range.offset, *host, range.bytes, true,
                                 "qualification refusal bytes");
        EXPECT_TRUE(copied);
        if (!copied) {
          node_.KeepPinned(*host);
          return out;
        }
        const auto* begin = static_cast<const std::byte*>(*host);
        out.bytes.insert(out.bytes.end(), begin, begin + range.bytes);
        EXPECT_TRUE(node_.FreePinned(*host));
      }
    }
    return out;
  }
  void Refused(const en::StepRefusal& refusal) {
    EXPECT_NE(refusal.what.find("joined repeat-pending execution is not qualified"),
              std::string::npos)
        << refusal.what;
    EXPECT_TRUE(refusal.clean);
    EXPECT_FALSE(refusal.state_refused);
    EXPECT_FALSE(refusal.widening);
  }
};

TEST_F(UnqualifiedDraftCompanionsTest, DeclaredPolicyRefusesBeforeColdStateAdmission) {
  EXPECT_EQ(slots().progress(0, 1)->completed(), md::DraftProgress{});
  EXPECT_EQ(slots().used_bytes(0), 0U);
  EXPECT_EQ(slots().used_bytes(1), 0U);
  const auto builds = source_->built().size();
  const auto queued = source_->queued();
  const std::array owners{0U, 1U};
  auto result = Prompt(owners, 7);
  EXPECT_EQ(result.steps, 0U);
  for (const auto& out : result.outcomes) {
    ASSERT_TRUE(out.refusal);
    Refused(*out.refusal);
  }
  EXPECT_EQ(slots().used_bytes(0), 0U);
  EXPECT_EQ(slots().used_bytes(1), 0U);
  EXPECT_EQ(source_->built().size(), builds);
  EXPECT_EQ(source_->queued(), queued);
}

TEST_F(UnqualifiedDraftCompanionsTest, EveryBoundaryPreservesOwedStateAndExistingUnit) {
  Prefill(0, 7);
  Prefill(1, 5);
  auto decoder = Decoder();
  const std::array first{0U};
  ASSERT_FALSE(Decode(decoder, first).outcomes[0].refusal);
  ASSERT_TRUE(slots().owed(0));
  auto unit = slots().BeginDraftUnit(1, 1);
  ASSERT_TRUE(unit);
  const std::array before{Observe(0), Observe(1)};
  const auto builds = source_->built().size(), queued = source_->queued(),
             executed = source_->executed();
  const auto graphs = runner_->graph_stats();
  const auto lookahead = runner_->lookahead_stats();
  std::vector<en::PendingWork> pending;
  std::vector<en::SegmentWork> input;
  std::vector<en::DecodeWork> decode;
  std::vector<en::PromptWork> prompts;
  std::vector<std::span<const std::int32_t>> histories;
  for (const auto slot : {0U, 1U}) {
    const auto n = slots().positions(slot);
    History(slot, n + 1);
    const auto tokens = std::span(history_[slot]).subspan(n, 1);
    pending.push_back({.slot = slot, .past = n});
    input.push_back({.tokens = tokens});
    decode.push_back({.slot = slot, .history = std::span(history_[slot]).first(n + 1)});
    prompts.push_back({.slot = slot, .tokens = tokens, .past = n, .history = history_[slot]});
    histories.push_back(history_[slot]);
  }
  const auto composition = composer_->Compose(pending);
  ASSERT_EQ(composition.steps.size(), 1U);
  const auto joined = composition.steps[0].shape;
  auto scalar = joined;
  scalar.segments.resize(1);
  en::StepAhead ahead;
  ahead.shapes[1] = joined;  // second hint must not escape the first-hint check
  ahead.prepare = ahead.capture_ahead = true;
  ahead.build = 2;
  const en::AheadFor predict = [&](const en::ComposedStep&, std::span<const std::size_t>) {
    return ahead;
  };
  const auto unchanged = [&] {
    EXPECT_EQ(Observe(0), before[0]);
    EXPECT_EQ(Observe(1), before[1]);
    EXPECT_TRUE(slots().progress(1, 1)->CheckUnit(*unit));
    EXPECT_EQ(source_->built().size(), builds);
    EXPECT_EQ(source_->queued(), queued);
    EXPECT_EQ(source_->executed(), executed);
    const auto now = runner_->graph_stats();
    EXPECT_EQ(now.eager, graphs.eager);
    EXPECT_EQ(now.captured, graphs.captured);
    EXPECT_EQ(now.replayed, graphs.replayed);
    EXPECT_EQ(now.dropped, graphs.dropped);
    const auto future = runner_->lookahead_stats();
    EXPECT_EQ(future.attempted, lookahead.attempted);
    EXPECT_EQ(future.built, lookahead.built);
    EXPECT_EQ(future.captured_ahead, lookahead.captured_ahead);
    EXPECT_EQ(future.prepared, lookahead.prepared);
  };
  const auto refused_report = [&](const auto& report) {
    EXPECT_EQ(report.steps, 0U);
    for (const auto& out : report.outcomes) {
      ASSERT_TRUE(out.refusal);
      Refused(*out.refusal);
    }
    unchanged();
  };
  refused_report(decoder.Step(decode));
  auto plain = en::DecodingOrchestrator::Make(*runner_, *composer_, {}, {});
  ASSERT_TRUE(plain);
  refused_report(plain->Step(decode));
  en::PrefillOrchestrator prefill(*runner_, *composer_, {});
  refused_report(prefill.Step(prompts));
  refused_report(en::RunPromptWave(*runner_, *composer_, pending, input, histories));
  refused_report(en::RunWave(*runner_, *composer_, pending, input));
  for (const auto stage : {0U, 1U}) {
    auto shape = joined;
    shape.stage = static_cast<std::uint16_t>(stage);
    shape.phase = stage == 0 ? pl::Phase::kPrefill : pl::Phase::kDraft;
    auto result = runner_->RunStep({.shape = shape, .work = input, .advance = false, .ahead = {}});
    ASSERT_FALSE(result);
    Refused(result.error());
    unchanged();
  }
  auto result = runner_->RunStep(
      {.shape = scalar, .work = std::span(input).first(1), .advance = false, .ahead = ahead});
  ASSERT_FALSE(result);
  Refused(result.error());
  unchanged();
  refused_report(en::RunWave(*runner_, *composer_, std::span(pending).first(1),
                             std::span(input).first(1), false, predict));
  refused_report(en::RunPromptWave(*runner_, *composer_, std::span(pending).first(1),
                                   std::span(input).first(1), std::span(histories).first(1),
                                   predict));
  // The untouched dirty owner still settles and continues with its exact K.
  ASSERT_TRUE(slots().Settle(0));
  ASSERT_FALSE(Decode(decoder, first).outcomes[0].refusal);
}

TEST_F(DraftCompanionsTest, PromptAndPendingPassKeepShiftedCoordinatesForTwoOwners) {
  const std::array owners{0U, 1U};
  auto prompt = Prompt(owners, 7);
  for (const auto& out : prompt.outcomes) ASSERT_FALSE(out.refusal) << out.refusal->what;
  for (const auto slot : owners) {
    EXPECT_EQ(slots().progress(slot, 1)->completed(), (md::DraftProgress{7, 6, 0, 7, 0}));
    EXPECT_EQ(slots().positions(slot), 7U);
  }
  auto decoder = Decoder();
  auto first = Decode(decoder, owners);
  for (const auto& out : first.outcomes) ASSERT_FALSE(out.refusal) << out.refusal->what;
  for (const auto slot : owners) {
    ASSERT_FALSE(drafts_[slot].empty());
    EXPECT_EQ(drafts_[slot].front().position, 6);
    EXPECT_EQ(drafts_[slot].front().token, Token(slot, 7));
    EXPECT_EQ(drafts_[slot].front().tap, Token(slot, 6));
    EXPECT_TRUE(slots().owed(slot));
    EXPECT_FALSE(slots().kept_whole(slot));
    EXPECT_EQ(decoder.cursor(slot, 0), 7U);
  }
  auto second = Decode(decoder, owners);
  for (const auto& out : second.outcomes) ASSERT_FALSE(out.refusal) << out.refusal->what;
  for (const auto slot : owners) {
    EXPECT_EQ(drafts_[slot].front().position, 9);
    EXPECT_EQ(drafts_[slot].front().token, Token(slot, 10));
    EXPECT_EQ(drafts_[slot].front().tap, Token(slot, 9));
  }
}

TEST_F(DraftCompanionsTest, WholeDiscardAndCleanPromptRefusalRetryOriginalK) {
  Prefill(0, 7);
  const std::array owners{0U};
  auto decoder = Decoder();
  auto accepted = Decode(decoder, owners);
  ASSERT_FALSE(accepted.outcomes[0].refusal) << accepted.outcomes[0].refusal->what;
  ASSERT_TRUE(slots().Settle(0));
  const auto before = slots().progress(0, 1)->completed();
  ASSERT_EQ(before.pending_first, 7U);
  ASSERT_EQ(before.positions, 10U);
  source_->options().refuse_build = [](const pl::StepShape& shape) {
    return shape.stage == 0 && shape.phase == pl::Phase::kPrefill;
  };
  runner_->DropPlans();
  auto refused = Prompt(owners, 2);
  ASSERT_TRUE(refused.outcomes[0].refusal);
  EXPECT_TRUE(refused.outcomes[0].refusal->clean);
  EXPECT_EQ(slots().progress(0, 1)->completed(), before);
  EXPECT_EQ(slots().positions(0), 10U);
  source_->options().refuse_build = {};
  discard_ = true;
  auto discarded = Decode(decoder, owners);
  ASSERT_TRUE(discarded.outcomes[0].discarded);
  EXPECT_TRUE(slots().usable(0));
  EXPECT_EQ(slots().progress(0, 1)->completed(), before);
  EXPECT_EQ(drafts_[0].front().position, 9);
  EXPECT_EQ(drafts_[0].front().tap, Token(0, 9));
  discard_ = false;
  auto retried = Decode(decoder, owners);
  ASSERT_FALSE(retried.outcomes[0].refusal) << retried.outcomes[0].refusal->what;
  EXPECT_EQ(drafts_[0].front().position, 9);
  EXPECT_EQ(drafts_[0].front().tap, Token(0, 9));
}

TEST_F(DraftCompanionsTest, SpillWithOwedPublicationKeepsPendingSpanAndCursor) {
  Prefill(0, 7);
  auto decoder = Decoder();
  const std::array owners{0U};
  auto accepted = Decode(decoder, owners);
  ASSERT_FALSE(accepted.outcomes[0].refusal) << accepted.outcomes[0].refusal->what;
  ASSERT_TRUE(slots().Spill(0));
  ASSERT_TRUE(slots().Restore(0));
  decoder.Reset(0);
  EXPECT_EQ(decoder.cursor(0, 0), 7U);
  auto resumed = Decode(decoder, owners);
  ASSERT_FALSE(resumed.outcomes[0].refusal) << resumed.outcomes[0].refusal->what;
  EXPECT_EQ(drafts_[0].front().position, 9);
  EXPECT_EQ(drafts_[0].front().tap, Token(0, 9));
}

TEST_F(DraftCompanionsTest, PublicTargetCannotOverwritePendingFeatures) {
  Prefill(0, 7);
  History(0, 7);
  en::PendingWork p{.slot = 0, .phase = pl::Phase::kDecode, .past = 7};
  en::SegmentWork w{.tokens = std::span(history_[0]).subspan(7, 1)};
  const auto executed = source_->executed();
  auto refused = en::RunWave(*runner_, *composer_, std::span(&p, 1), std::span(&w, 1));
  ASSERT_TRUE(refused.outcomes[0].refusal);
  EXPECT_TRUE(refused.outcomes[0].refusal->clean);
  EXPECT_EQ(source_->executed(), executed);
  EXPECT_EQ(slots().positions(0), 7U);
}

TEST_F(DraftCompanionsTest, CheckpointRestoresTheOriginalPendingFeaturesAndSchedule) {
  Prefill(0, 7);
  const std::array owners{0U};
  auto decoder = Decoder();
  ASSERT_FALSE(Decode(decoder, owners).outcomes[0].refusal);
  ASSERT_TRUE(slots().Settle(0));
  const auto original = slots().progress(0, 1)->completed();
  std::vector<en::LiveState::Range> footprint;
  for (std::size_t s = 0; s < slots().stages(); ++s)
    for (auto range : slots().live(0, s).used_ranges()) {
      range.region = s;
      footprint.push_back(range);
    }
  auto pages = slots().CheckpointRanges(0, 10);
  ASSERT_TRUE(pages) << ts::Failed(pages).value_or("");
  std::uint64_t bytes = 0;
  for (const auto& page : *pages) bytes += page.bytes;
  std::vector<llmp::catalog::ExtentId> staging;
  auto host = node_.Pinned(bytes, 0, staging);
  ASSERT_TRUE(host);
  ASSERT_TRUE(slots().CopyState(0, *host, *pages, true));
  auto newer = Prompt(owners, 4);
  ASSERT_FALSE(newer.outcomes[0].refusal) << newer.outcomes[0].refusal->what;
  EXPECT_NE(slots().progress(0, 1)->completed(), original);
  ASSERT_TRUE(slots().PrepareRestore(0, 10, footprint, slots().kept_layout_id()));
  EXPECT_FALSE(slots().usable(0));
  EXPECT_FALSE(slots().kept_whole(0));
  EXPECT_FALSE(slots().CompleteRestore(0, 10));
  ASSERT_TRUE(slots().CopyState(0, *host, *pages, false));
  ASSERT_TRUE(slots().CompleteRestore(0, 10));
  EXPECT_EQ(slots().progress(0, 1)->completed(), original);
  // The caller supplies a new anchor after restoring an earlier branch.
  history_[0][10] = 9999;
  decoder.Reset(0);
  auto resumed = Decode(decoder, owners);
  ASSERT_FALSE(resumed.outcomes[0].refusal) << resumed.outcomes[0].refusal->what;
  EXPECT_EQ(drafts_[0].front().position, 9);
  EXPECT_EQ(drafts_[0].front().token, 9999);
  EXPECT_EQ(drafts_[0].front().tap, Token(0, 9));
}

TEST_F(DraftCompanionsTest, CheckpointRequiresTheReplayedPageBeforeTheTwoMiBBoundary) {
  Prefill(0, 4093);
  const std::array owners{0U};
  auto decoder = Decoder();
  ASSERT_FALSE(Decode(decoder, owners).outcomes[0].refusal);
  ASSERT_TRUE(slots().Settle(0));
  ASSERT_EQ(slots().positions(0), 4096U);
  ASSERT_EQ(slots().progress(0, 1)->completed().pending_first, 4093U);
  std::vector<en::LiveState::Range> footprint;
  for (std::size_t s = 0; s < slots().stages(); ++s)
    for (auto range : slots().live(0, s).used_ranges()) {
      range.region = s;
      footprint.push_back(range);
    }
  auto pages = slots().CheckpointRanges(0, 4096);
  ASSERT_TRUE(pages);
  const auto previous = std::ranges::find_if(*pages, [](const auto& p) {
    return p.region == 1 && p.offset == kExtent && p.bytes == kExtent;
  });
  ASSERT_NE(previous, pages->end());
  std::uint64_t bytes = 0;
  for (const auto& p : *pages) bytes += p.bytes;
  std::vector<llmp::catalog::ExtentId> staging;
  auto host = node_.Pinned(bytes, 0, staging);
  ASSERT_TRUE(host);
  ASSERT_TRUE(slots().CopyState(0, *host, *pages, true));
  // Public draft work cannot write an immutable old page, even with a
  // feature from the current retained chunk. Both RunWave's admission and
  // direct RunStep reject it before allocating or executing anything.
  EXPECT_TRUE(
      std::ranges::none_of(*pages, [](const auto& p) { return p.region == 1 && p.offset == 0; }));
  History(0, 4096);
  const auto executed = source_->executed();
  const auto used = slots().used_bytes(0);
  const auto record = slots().progress(0, 1)->completed();
  const auto& items = slots().layout(1).items;
  const auto control = std::ranges::find_if(items, [](const auto& item) {
    return item.contract.kind == md::StateKind::kDrafterProgress;
  });
  ASSERT_NE(control, items.end());
  const std::array immutable{
      en::LiveState::Range{.region = 1, .offset = 0, .bytes = kExtent},
      en::LiveState::Range{
          .region = 1, .offset = control->offset, .bytes = md::kDraftProgressBytes}};
  const auto immutable_before = Read(0, immutable);
  ASSERT_EQ(immutable_before.size(), kExtent + md::kDraftProgressBytes);
  for (const auto past : {0U, 4093U}) {
    en::PendingWork bad{.slot = 0, .stage = 1, .phase = pl::Phase::kDraft, .past = past};
    en::SegmentWork input{.tokens = std::span(history_[0]).subspan(4096, 1),
                          .feature = {.kind = en::FeatureFrom::Kind::kTap, .position = 4095}};
    auto refusal =
        en::RunWave(*runner_, *composer_, std::span(&bad, 1), std::span(&input, 1), false);
    ASSERT_TRUE(refusal.outcomes[0].refusal);
    EXPECT_TRUE(refusal.outcomes[0].refusal->clean);
    auto shape = composer_->Compose(std::span(&bad, 1)).steps.front().shape;
    auto direct = runner_->RunStep(
        {.shape = shape, .work = std::span(&input, 1), .advance = false, .ahead = {}});
    ASSERT_FALSE(direct);
    EXPECT_TRUE(direct.error().clean);
  }
  EXPECT_EQ(source_->executed(), executed);
  EXPECT_EQ(slots().used_bytes(0), used);
  EXPECT_EQ(slots().progress(0, 1)->completed(), record);
  EXPECT_EQ(Read(0, immutable), immutable_before);
  ASSERT_FALSE(Decode(decoder, owners).outcomes[0].refusal);
  ASSERT_TRUE(slots().Settle(0));
  ASSERT_TRUE(slots().PrepareRestore(0, 4096, footprint, slots().kept_layout_id()));
  auto* read = static_cast<std::byte*>(*host);
  std::byte* missing = nullptr;
  for (const auto& page : *pages) {
    if (&page == &*previous)
      missing = read;
    else
      ASSERT_TRUE(slots().CopyState(0, read, std::span(&page, 1), false));
    read += page.bytes;
  }
  ASSERT_NE(missing, nullptr);
  EXPECT_FALSE(slots().CompleteRestore(0, 4096));
  EXPECT_FALSE(slots().usable(0));
  ASSERT_TRUE(slots().CopyState(0, missing, std::span(&*previous, 1), false));
  ASSERT_TRUE(slots().CompleteRestore(0, 4096));
  EXPECT_EQ(slots().progress(0, 1)->completed().pending_first, 4093U);
  auto resumed = Decode(decoder, owners);
  ASSERT_FALSE(resumed.outcomes[0].refusal) << resumed.outcomes[0].refusal->what;
  EXPECT_EQ(drafts_[0].front().position, 4095);
  EXPECT_EQ(drafts_[0].front().tap, Token(0, 4095));
}

TEST_F(DraftCompanionsTest, RestoredBytesNeedCompletedControlHydration) {
  Prefill(0, 7);
  const std::array owners{0U};
  auto decoder = Decoder();
  ASSERT_FALSE(Decode(decoder, owners).outcomes[0].refusal);
  ASSERT_TRUE(slots().Settle(0));
  const auto saved = Save(0);
  ASSERT_NE(saved.host, nullptr);
  ASSERT_FALSE(Prompt(owners, 4).outcomes[0].refusal);
  const auto before = slots().progress(0, 1)->completed();

  ASSERT_TRUE(slots().PrepareRestore(0, 10, saved.footprint, slots().kept_layout_id()));
  ASSERT_TRUE(slots().CopyState(0, saved.host, saved.pages, false));
  // Numeric pages are present; a failed readback cannot publish a cursor.
  llmp::providers::fake::FailNextCopy(llmp::providers::CopyKind::kDeviceToHost,
                                      md::kDraftProgressBytes);
  EXPECT_FALSE(slots().CompleteRestore(0, 10));
  EXPECT_FALSE(slots().usable(0));
  EXPECT_FALSE(slots().kept_whole(0));
  EXPECT_EQ(slots().progress(0, 1)->completed(), before);
}

TEST_F(DraftCompanionsTest, RestoredBytesNeedAControlMatchingTheirDeclaredRelation) {
  Prefill(0, 7);
  const std::array owners{0U};
  auto decoder = Decoder();
  ASSERT_FALSE(Decode(decoder, owners).outcomes[0].refusal);
  ASSERT_TRUE(slots().Settle(0));
  const auto saved = Save(0);
  ASSERT_NE(saved.host, nullptr);
  ASSERT_FALSE(Prompt(owners, 4).outcomes[0].refusal);
  const auto before = slots().progress(0, 1)->completed();
  const auto& control = slots().layout(1).items.back();
  ASSERT_EQ(control.contract.kind, md::StateKind::kDrafterProgress);
  auto* bytes = static_cast<std::byte*>(saved.host);
  bool changed = false;
  for (const auto& page : saved.pages) {
    if (page.region == 1 && control.offset >= page.offset &&
        control.offset + md::kDraftProgressBytes <= page.offset + page.bytes) {
      md::DraftProgressWords words{};
      std::memcpy(&words, bytes + control.offset - page.offset, sizeof(words));
      words[6] = 0;  // the layout declares token-after-cache
      std::memcpy(bytes + control.offset - page.offset, &words, sizeof(words));
      changed = true;
    }
    bytes += page.bytes;
  }
  ASSERT_TRUE(changed);
  ASSERT_TRUE(slots().PrepareRestore(0, 10, saved.footprint, slots().kept_layout_id()));
  ASSERT_TRUE(slots().CopyState(0, saved.host, saved.pages, false));

  EXPECT_FALSE(slots().CompleteRestore(0, 10));
  EXPECT_FALSE(slots().usable(0));
  EXPECT_FALSE(slots().kept_whole(0));
  EXPECT_EQ(slots().progress(0, 1)->completed(), before);
}

TEST_F(DraftCompanionsTest, FailedControlCopyAfterAcceptedDataNeverPublishesTheNewRecord) {
  Prefill(0, 7);
  const auto original = slots().progress(0, 1)->completed();
  const std::array owners{0U};
  auto decoder = Decoder();
  ASSERT_FALSE(Decode(decoder, owners).outcomes[0].refusal);
  ASSERT_TRUE(slots().owed(0));
  llmp::providers::fake::FailNextCopy(llmp::providers::CopyKind::kHostToDevice,
                                      md::kDraftProgressBytes);
  auto refused = slots().Settle(0);
  ASSERT_FALSE(refused);
  EXPECT_EQ(slots().progress(0, 1)->completed(), original);
  EXPECT_FALSE(slots().usable(0));
  EXPECT_FALSE(slots().kept_whole(0));
  EXPECT_FALSE(slots().CheckpointRanges(0, 10));
  EXPECT_FALSE(slots().live(0, 1).owed());  // numeric restore preceded the control call
  EXPECT_EQ(node_.kept_pinned(), 1U);
}

TEST_F(DraftCompanionsTest, QueuedButUnprovenControlCopyRetainsItsBufferAndQuarantines) {
  Prefill(0, 7);
  const auto original = slots().progress(0, 1)->completed();
  const std::array owners{0U};
  auto decoder = Decoder();
  ASSERT_FALSE(Decode(decoder, owners).outcomes[0].refusal);
  llmp::providers::fake::FailNextCopy(llmp::providers::CopyKind::kHostToDevice,
                                      md::kDraftProgressBytes, true);
  ASSERT_FALSE(slots().Settle(0));
  EXPECT_EQ(slots().progress(0, 1)->completed(), original);
  EXPECT_FALSE(slots().usable(0));
  EXPECT_EQ(node_.kept_pinned(), 1U);
  EXPECT_FALSE(slots().kept_whole(0));
}

class KeptDraftCompanionsTest : public DraftCompanionsTest {
 protected:
  int kept_dir_ = -1;
  std::filesystem::path kept_path_;
  void Configure(en::RunnerConfig& config) override {
    kept_path_ = config.slots.spill_directory / std::format("repeat-kept-{}", ::getpid());
    auto directory = llmp::platform::OpenPrivateDirectory(-1, kept_path_.c_str());
    ASSERT_TRUE(directory);
    kept_dir_ = *directory;
    const auto scratch = config.slots.spill_directory;
    config.slots.spill_place = [this, scratch](std::uint32_t slot, std::uint32_t stage) {
      return slot == 0 ? en::LiveState::SpillPlace{.directory = scratch,
                                                   .dir = kept_dir_,
                                                   .name = std::format("stage-{}", stage)}
                       : en::LiveState::SpillPlace{.directory = scratch, .name = {}};
    };
  }
  void TearDown() override {
    DraftCompanionsTest::TearDown();
    if (kept_dir_ >= 0) (void)::close(kept_dir_);
    std::filesystem::remove_all(kept_path_);
  }
};

TEST_F(KeptDraftCompanionsTest, ANewRunnerHydratesPendingProgressBeforeAdoptedStateCanRun) {
  Prefill(0, 7);
  const std::array owners{0U};
  auto decoder = Decoder();
  ASSERT_FALSE(Decode(decoder, owners).outcomes[0].refusal);
  ASSERT_TRUE(slots().Settle(0));
  const auto original = slots().progress(0, 1)->completed();
  const auto saved = Save(0);
  ASSERT_TRUE(slots().Spill(0));
  ASSERT_TRUE(slots().kept_whole(0));
  std::vector<llmp::platform::FileIdentity> identities;
  for (std::size_t stage = 0; stage < slots().stages(); ++stage) {
    ASSERT_TRUE(slots().live(0, stage).spill_identity());
    identities.push_back(*slots().live(0, stage).spill_identity());
  }
  en::PagedNode restarted_node{
      en::NodeSettings{.compute_streams = 1, .slots = 2, .copy_lane = true, .slot_bytes = kExtent}};
  ASSERT_TRUE(restarted_node.Open());
  const auto fixed = restarted_node.catalog().OccupancyOf(restarted_node.domain()).Total().value();
  auto source = std::make_unique<ts::FakeStepSource>(source_->options());
  en::RunnerConfig config;
  config.slots = {.model = "Repeat",
                  .slots = 2,
                  .stages = Stages(),
                  .spill_place = {},
                  .spill_directory = kept_path_,
                  .invalidate = {},
                  .drop_plans = {}};
  config.slots.spill_place = [&](std::uint32_t slot, std::uint32_t stage) {
    return slot == 0 ? en::LiveState::SpillPlace{.directory = kept_path_,
                                                 .dir = kept_dir_,
                                                 .name = std::format("stage-{}", stage),
                                                 .keep = true,
                                                 .expect = identities[stage]}
                     : en::LiveState::SpillPlace{.directory = kept_path_, .name = {}};
  };
  for (const auto& built : source_->built())
    if (built.measuring) config.setup_shapes.push_back(built.shape);
  config.max_verify_rows = 4;
  config.launch = []() -> std::expected<std::unique_ptr<llmp::kernels::StepLaunch>, std::string> {
    return std::make_unique<llmp::kernels::RecordingLaunch>();
  };
  auto made = en::ModelRunner::Create(restarted_node, std::move(config), std::move(source), 0, 0);
  ASSERT_TRUE(made);
  auto restarted = std::move(*made);
  const auto run = [&]() -> en::Status {
    if (auto x = restarted->Setup(); !x) return x;
    if (auto x = restarted_node.MapWorkspace(restarted->activations_needed(),
                                             std::max(kExtent, restarted->pool_needed()));
        !x)
      return x;
    if (auto x = restarted_node.Start(llmp::base::Bytes(fixed + 64 * kExtent)); !x) return x;
    if (auto x = restarted->Register(); !x) return x;
    restarted_node.Run();
    if (auto x = restarted->Bind(); !x) return x;
    auto& state = restarted->slots();
    if (auto x = state.Adopt(0, 10, saved.footprint, slots().kept_layout_id()); !x) return x;
    EXPECT_FALSE(state.usable(0));
    EXPECT_FALSE(state.kept_whole(0));
    EXPECT_FALSE(state.BeginDraftUnit(0, 1));
    EXPECT_EQ(state.progress(0, 1)->completed(), md::DraftProgress{});
    if (auto x = state.Restore(0); !x) return x;
    EXPECT_EQ(state.progress(0, 1)->completed(), original);
    EXPECT_TRUE(state.usable(0));
    EXPECT_TRUE(state.kept_whole(0));
    if (auto x = state.SelectSlots(owners); !x) return x;
    md::Batching caps;
    auto composer = en::WaveComposer::For(caps, state);
    if (!composer) return std::unexpected(composer.error());
    bool saw = false;
    en::DecodingPolicy policy;
    policy.propose = [&](std::uint32_t, std::size_t, std::span<const std::byte> bytes,
                         std::uint32_t) -> std::expected<std::vector<std::int32_t>, std::string> {
      ts::FakeRow row;
      std::memcpy(&row, bytes.data(), sizeof(row));
      EXPECT_EQ(row.position, 9);
      EXPECT_EQ(row.token, 7777);
      EXPECT_EQ(row.tap, Token(0, 9));
      saw = true;
      return std::vector<std::int32_t>{row.token + 1};
    };
    policy.judge = [](std::uint32_t, std::span<const std::int32_t>,
                      std::span<const std::byte>) -> std::expected<en::Verdict, std::string> {
      return std::unexpected("whole discard");
    };
    auto resumed = en::DecodingOrchestrator::Make(
        *restarted, *composer,
        {.kind = en::DecodingMode::Kind::kSpeculative,
         .drafters = {{.stage = 1, .max_depth = 1, .catch_up = true, .features = true}}},
        std::move(policy));
    if (!resumed) return std::unexpected(resumed.error());
    History(0, 10);
    history_[0][10] = 7777;
    const en::DecodeWork work{.slot = 0, .history = std::span(history_[0]).first(11)};
    auto report = resumed->Step(std::span(&work, 1));
    if (!report.outcomes[0].discarded)
      return std::unexpected(report.outcomes[0].refusal ? report.outcomes[0].refusal->what
                                                        : "resumed unit was not discarded");
    EXPECT_TRUE(saw);
    EXPECT_TRUE(report.outcomes[0].discarded);
    EXPECT_EQ(state.progress(0, 1)->completed(), original);
    return {};
  }();
  EXPECT_TRUE(run) << ts::Failed(run).value_or("");
  std::array<en::PagedModel*, 1> models{restarted.get()};
  EXPECT_TRUE(restarted_node.TearDown(models));
}

TEST_F(DraftCompanionsTest, LeadingCarryMutationBeforeRejectedTargetQuarantines) {
  Prefill(0, 7);
  const std::array owners{0U};
  source_->options().fail_phase = pl::Phase::kPrefill;
  source_->options().fail_run = llmp::kernels::StepError::kRejected;
  auto refused = Prompt(owners, 2);
  ASSERT_TRUE(refused.outcomes[0].refusal);
  EXPECT_FALSE(refused.outcomes[0].refusal->clean);
  EXPECT_FALSE(slots().usable(0));
  EXPECT_EQ(slots().positions(0), 7U);
}

class PartialBlockDraftTest : public DraftCompanionsTest {
 protected:
  std::uint64_t BudgetExtents() const override { return 256; }
  std::vector<en::StageState> StateStages() const override {
    auto stages = Stages();
    md::StateContract contract;
    for (const auto& item : stages[1].layout.items) contract.tensors.push_back(item.contract);
    contract.tensors.push_back({.name = "partial_blocks",
                                .kind = md::StateKind::kBlockKeys,
                                .cell_bytes = kExtent,
                                .cells = md::CellRule::kPerBlock,
                                .window_or_ratio = 4,
                                .cell_pad = 1,
                                .dummy_last = true,
                                .partial_blocks = true,
                                .rollback = md::Rollback::kSnapshot,
                                .replayable = true});
    const std::array layers{std::pair{0U, std::move(contract)}};
    stages[1].layout = md::Layout(layers, kContext, kRows).value();
    return stages;
  }
};
TEST_F(PartialBlockDraftTest, CheckpointRestoresThePartialBlockAndDistantDummyWithoutAllContext) {
  Prefill(0, 7);
  keep_ = 4;
  auto decoder = Decoder();
  const std::array owners{0U};
  ASSERT_FALSE(Decode(decoder, owners).outcomes[0].refusal);
  ASSERT_TRUE(slots().Settle(0));
  ASSERT_EQ(slots().positions(0), 11U);
  ASSERT_EQ(slots().progress(0, 1)->completed().pending_first, 7U);
  const auto original = slots().progress(0, 1)->completed();
  const auto& block = slots().layout(1).items.back();
  ASSERT_TRUE(block.contract.partial_blocks && block.contract.dummy_last);
  const std::array marks{
      en::LiveState::Range{
          .region = 1, .offset = block.offset + block.contract.cell_bytes, .bytes = 1},
      en::LiveState::Range{.region = 1,
                           .offset = block.offset + (block.cells - 1) * block.contract.cell_bytes,
                           .bytes = 1}};
  const auto fill = [&](std::uint8_t value) {
    if (auto written = slots().RecordWrite(0, marks); !written) return written;
    return node_.Job(
        slots().slot_fence(0),
        [&](llmp::providers::NativeStream stream) {
          for (const auto& range : marks)
            if (!llmp::providers::FillAsync(
                     stream, en::support::Pointer(slots().live(0, 1).base(0) + range.offset), value,
                     range.bytes)
                     .ok())
              return llmp::scheduler::JobResult::kUnknown;
          return llmp::scheduler::JobResult::kQueued;
        },
        "mutable block cells", 0);
  };
  ASSERT_TRUE(fill(0x41));
  const auto checkpoint = Save(0);
  ASSERT_NE(checkpoint.host, nullptr);
  for (const auto& mark : marks)
    EXPECT_TRUE(std::ranges::any_of(checkpoint.pages, [&](const auto& page) {
      return page.region == mark.region && mark.offset >= page.offset &&
             mark.offset + mark.bytes <= page.offset + page.bytes;
    }));
  // Cell 1 is replayed at q=7 although ordinary forward writes start in cell 2.
  // Two-MiB cells isolate it from both that forward page and fixed/control pages.
  const auto earlier_page = (marks[0].offset / kExtent) * kExtent;
  const auto forward_page = ((block.offset + 2 * block.contract.cell_bytes) / kExtent) * kExtent;
  EXPECT_LT(earlier_page, forward_page);
  EXPECT_GT(earlier_page, (block.offset / kExtent) * kExtent);
  const auto middle = (block.offset / kExtent + 128) * kExtent;
  EXPECT_TRUE(std::ranges::none_of(checkpoint.pages, [&](const auto& page) {
    return page.region == 1 && page.offset == middle;
  }));
  ASSERT_TRUE(fill(0xB9));
  ASSERT_EQ(Read(0, marks), (std::vector<std::byte>{std::byte{0xB9}, std::byte{0xB9}}));
  ASSERT_TRUE(slots().PrepareRestore(0, 11, checkpoint.footprint, slots().kept_layout_id()));
  ASSERT_TRUE(slots().CopyState(0, checkpoint.host, checkpoint.pages, false));
  ASSERT_TRUE(slots().CompleteRestore(0, 11));
  EXPECT_EQ(Read(0, marks), (std::vector<std::byte>{std::byte{0x41}, std::byte{0x41}}));
  EXPECT_EQ(slots().progress(0, 1)->completed(), original);
}

class MultipleDraftConsumersTest : public DraftCompanionsTest {
 protected:
  std::uint32_t choice_offset_ = 0;
  bool plain_ = false;
  std::array<std::array<std::vector<ts::FakeRow>, 2>, 2> seen_;
  std::vector<en::StageState> StateStages() const override {
    auto stages = Stages();
    stages.push_back(stages[1]);
    stages.back().name = "alternative";
    return stages;
  }
  en::DecodingOrchestrator Alternatives() {
    en::DecodingPolicy policy;
    policy.choose = [this](std::uint32_t slot) {
      return std::optional<std::size_t>((slot + choice_offset_) % 2);
    };
    policy.speculate = [this](std::uint32_t) { return !plain_; };
    policy.propose = [this](
                         std::uint32_t slot, std::size_t drafter, std::span<const std::byte> bytes,
                         std::uint32_t) -> std::expected<std::vector<std::int32_t>, std::string> {
      if (bytes.size() < sizeof(ts::FakeRow)) return std::unexpected("missing row");
      auto& row = seen_[slot][drafter].emplace_back();
      std::memcpy(&row, bytes.data(), sizeof(row));
      return std::vector<std::int32_t>{row.token + 1};
    };
    policy.judge = [this](std::uint32_t, std::span<const std::int32_t>,
                          std::span<const std::byte>) -> std::expected<en::Verdict, std::string> {
      if (discard_) return std::unexpected("discard alternatives");
      return en::Verdict{.keep = keep_};
    };
    return en::DecodingOrchestrator::Make(
               *runner_, *composer_,
               {.kind = en::DecodingMode::Kind::kSpeculative,
                .drafters = {{.stage = 1, .max_depth = 3, .catch_up = true, .features = true},
                             {.stage = 2, .max_depth = 3, .catch_up = true, .features = true}},
                .combine = en::DecodingMode::Drafters::kAlternatives},
               std::move(policy))
        .value();
  }
};

TEST_F(MultipleDraftConsumersTest, UnselectedConsumersKeepTheOriginalRepeatSchedule) {
  const std::array owners{0U, 1U};
  ASSERT_FALSE(Prompt(owners, 7).outcomes[0].refusal);
  auto decoder = Alternatives();
  auto first = Decode(decoder, owners);
  for (const auto slot : owners) {
    ASSERT_FALSE(first.outcomes[slot].refusal) << first.outcomes[slot].refusal->what;
    ASSERT_TRUE(slots().Settle(slot));
    for (const auto stage : {1U, 2U}) {
      const auto& p = slots().progress(slot, stage)->completed();
      EXPECT_EQ(p.positions, 10U);
      EXPECT_EQ(p.pending_first, 7U);
    }
  }
  choice_offset_ = 1;
  discard_ = true;
  auto discarded = Decode(decoder, owners);
  for (const auto slot : owners) {
    ASSERT_TRUE(discarded.outcomes[slot].discarded);
    const auto chosen = (slot + choice_offset_) % 2;
    ASSERT_FALSE(seen_[slot][chosen].empty());
    const auto& first_row = seen_[slot][chosen].front();
    EXPECT_EQ(first_row.position, 9);
    EXPECT_EQ(first_row.token, Token(slot, 10));
    EXPECT_EQ(first_row.tap, Token(slot, 9));
    for (const auto stage : {1U, 2U}) {
      EXPECT_EQ(slots().progress(slot, stage)->completed().pending_first, 7U);
      const auto& kv = slots().layout(stage).items.front();
      const en::LiveState::Range tail{.region = stage,
                                      .offset = kv.offset + 9 * kv.contract.cell_bytes,
                                      .bytes = sizeof(ts::FakeCell)};
      const auto bytes = Read(slot, std::span(&tail, 1));
      ASSERT_EQ(bytes.size(), sizeof(ts::FakeCell));
      ts::FakeCell row;
      std::memcpy(&row, bytes.data(), sizeof(row));
      EXPECT_EQ(row.slot, static_cast<std::int32_t>(slot));
      EXPECT_EQ(row.position, 9);
      EXPECT_EQ(row.token, Token(slot, 10));
    }
  }
  discard_ = false;
  keep_ = 2;
  auto retried = Decode(decoder, owners);
  for (const auto slot : owners) {
    ASSERT_FALSE(retried.outcomes[slot].refusal) << retried.outcomes[slot].refusal->what;
    ASSERT_TRUE(slots().Settle(slot));
    for (const auto stage : {1U, 2U}) {
      EXPECT_EQ(slots().progress(slot, stage)->completed().positions, 12U);
      EXPECT_EQ(slots().progress(slot, stage)->completed().pending_first, 10U);
    }
  }
  plain_ = true;
  auto fallback = Decode(decoder, owners);
  for (const auto slot : owners) {
    ASSERT_FALSE(fallback.outcomes[slot].refusal) << fallback.outcomes[slot].refusal->what;
    ASSERT_TRUE(slots().Settle(slot));
    for (const auto stage : {1U, 2U}) {
      EXPECT_EQ(slots().progress(slot, stage)->completed().positions, 13U);
      EXPECT_EQ(slots().progress(slot, stage)->completed().pending_first, 12U);
    }
  }
}

TEST_F(MultipleDraftConsumersTest, ASecondInvalidHydrationPublishesNoFirstStageMirror) {
  Prefill(0, 7);
  const auto checkpoint = Save(0);
  ASSERT_NE(checkpoint.host, nullptr);
  Prefill(0, 4);
  const auto first_before = slots().progress(0, 1)->completed();
  const auto second_before = slots().progress(0, 2)->completed();
  const auto& control = slots().layout(2).items.back();
  ASSERT_EQ(control.contract.kind, md::StateKind::kDrafterProgress);
  auto* bytes = static_cast<std::byte*>(checkpoint.host);
  bool changed = false;
  for (const auto& page : checkpoint.pages) {
    if (page.region == 2 && control.offset >= page.offset &&
        control.offset + md::kDraftProgressBytes <= page.offset + page.bytes) {
      md::DraftProgressWords words{};
      std::memcpy(&words, bytes + control.offset - page.offset, sizeof(words));
      words[6] = 0;
      std::memcpy(bytes + control.offset - page.offset, &words, sizeof(words));
      changed = true;
    }
    bytes += page.bytes;
  }
  ASSERT_TRUE(changed);
  ASSERT_TRUE(slots().PrepareRestore(0, 7, checkpoint.footprint, slots().kept_layout_id()));
  ASSERT_TRUE(slots().CopyState(0, checkpoint.host, checkpoint.pages, false));
  EXPECT_FALSE(slots().CompleteRestore(0, 7));
  EXPECT_FALSE(slots().usable(0));
  EXPECT_FALSE(slots().kept_whole(0));
  EXPECT_EQ(slots().progress(0, 1)->completed(), first_before);
  EXPECT_EQ(slots().progress(0, 2)->completed(), second_before);
}

TEST_F(MultipleDraftConsumersTest, ASecondPublicationCopyFailureExposesNeitherNewMirror) {
  Prefill(0, 7);
  const auto first_before = slots().progress(0, 1)->completed();
  const auto second_before = slots().progress(0, 2)->completed();
  auto decoder = Alternatives();
  const std::array owners{0U};
  ASSERT_FALSE(Decode(decoder, owners).outcomes[0].refusal);
  ASSERT_TRUE(slots().owed(0));
  llmp::providers::fake::FailNextCopy(llmp::providers::CopyKind::kHostToDevice,
                                      md::kDraftProgressBytes, false, 1);
  EXPECT_FALSE(slots().Settle(0));
  EXPECT_EQ(slots().progress(0, 1)->completed(), first_before);
  EXPECT_EQ(slots().progress(0, 2)->completed(), second_before);
  EXPECT_FALSE(slots().live(0, 1).owed());
  EXPECT_FALSE(slots().live(0, 2).owed());
  EXPECT_FALSE(slots().usable(0));
  EXPECT_FALSE(slots().kept_whole(0));
  EXPECT_FALSE(slots().CheckpointRanges(0, 10));
  EXPECT_EQ(node_.kept_pinned(), 2U);
}

}  // namespace
