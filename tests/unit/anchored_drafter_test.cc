// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// An anchored drafter that reads its target's state frozen and its tapped
// features (engine/decoding.h DrafterSpec::anchored and features; Gemma
// 4's assistant), over the fake step source (tests/support/
// fake_step_source.h with `features`): every pass at the committed
// positions, reading the cells before them (a read through a past that is
// a multiple of 256 is the past itself), its first pass's feature the
// target's tap of the position before, a later one's its own last next
// feature (the runner's carry); the runner's refusals of features nothing
// holds, of a tap ring narrower than a verify or a target's outputs; and
// the computed inputs it stages (a tap's output-row cells, a frozen read's
// mask).

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <expected>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "base/bytes.h"
#include "engine/decoding.h"
#include "engine/model_runner.h"
#include "engine/paged_node.h"
#include "engine/prefill.h"
#include "engine/wave_composer.h"
#include "expected_error.h"
#include "fake_step_source.h"
#include "ir/graph.h"
#include "kernels/common/recording_launch.h"
#include "model/batching.h"
#include "model/state_contract.h"
#include "planner/step_shape.h"

namespace {

namespace en = llmp::engine;
namespace kn = llmp::kernels;
namespace md = llmp::model;
namespace pl = llmp::planner;
namespace ts = llmp::test_support;
using llmp::base::Bytes;
using C = md::StateTensorContract;
using Layers = std::vector<std::pair<std::uint32_t, md::StateContract>>;
constexpr std::uint64_t kExtent = en::kPagedExtent;
constexpr std::uint32_t kContext = 1024;
constexpr std::uint32_t kSlots = 2;
constexpr std::uint32_t kChunk = 64;
constexpr std::uint32_t kTapCells = 16;
constexpr std::uint32_t kDepth = 3;
constexpr std::uint16_t kAssistant = 1;

std::int32_t Truth(std::int64_t q) { return static_cast<std::int32_t>(500 + q); }

std::vector<ts::FakeRow> Rows(std::span<const std::byte> bytes) {
  std::vector<ts::FakeRow> rows(bytes.size() / sizeof(ts::FakeRow));
  if (!rows.empty()) std::memcpy(rows.data(), bytes.data(), rows.size() * sizeof(ts::FakeRow));
  return rows;
}

// The target: a cache that truncates (Gemma's). The assistant: a view of
// the target's cache and the target's feature tap, a position ring.
std::vector<en::StageState> Stages(bool chunk = false) {
  const Layers target = {{0,
                          {.tensors = {C{.name = "kv",
                                         .kind = md::StateKind::kKv,
                                         .cell_bytes = 256,
                                         .rollback = md::Rollback::kTruncate}}}}};
  const Layers assistant = {
      {0,
       {.tensors = {C{.name = "kv",
                      .kind = md::StateKind::kKv,
                      .cell_bytes = 256,
                      .borrowed_from_target = true,
                      .target_layer = 0},
                    C{.name = "features",
                      .kind = md::StateKind::kFeatures,
                      .encoding = md::Encoding::kF32,
                      .cell_bytes = 256,
                      .cells = chunk ? md::CellRule::kPerRow : md::CellRule::kPositionRing,
                      .fixed_cells = chunk ? 0U : kTapCells,
                      .row_carry = chunk ? 1U : 0U,
                      .rollback = chunk ? md::Rollback::kSnapshot : md::Rollback::kTruncate,
                      .target_writes = true}}}}};
  return {{.name = "target", .layout = md::Layout(target, kContext, kChunk).value()},
          {.name = "assistant", .layout = md::Layout(assistant, kContext, kChunk).value()}};
}

pl::StepShape Wide(std::uint16_t stage, pl::Phase phase, std::uint32_t rows) {
  pl::StepShape s{.stage = stage, .phase = phase, .segments = {}, .read_align = 256};
  for (std::uint8_t slot = 0; slot < kSlots; ++slot) {
    s.segments.push_back({.slot = slot,
                          .rows = rows,
                          .past = kContext - rows,
                          .read_cells = kContext,
                          .outputs = rows,
                          .want_head = true});
  }
  return s;
}

// A slot's three rows after 40, the last two its outputs: the computed
// inputs' step.
pl::StepShape ComputedShape() {
  return {
      .stage = 0,
      .phase = pl::Phase::kPrefill,
      .segments =
          {{.slot = 0, .rows = 3, .past = 40, .read_cells = 256, .outputs = 2, .want_head = true}},
      .read_align = 256};
}

class AnchoredDrafterTest : public ::testing::Test {
 protected:
  en::PagedNode node_{
      en::NodeSettings{.compute_streams = 1, .slots = 2, .copy_lane = true, .slot_bytes = kExtent}};
  std::unique_ptr<en::ModelRunner> runner_;
  ts::FakeStepSource* source_ = nullptr;
  std::optional<en::WaveComposer> composer_;
  std::array<std::vector<std::int32_t>, kSlots> history_{};
  std::array<std::vector<ts::FakeRow>, kSlots> drafts_{};

  // The verify rows the runner takes (Setup refuses past the tap's cells).
  virtual std::uint32_t VerifyRows() const { return kDepth + 1; }
  virtual bool ChunkTap() const { return false; }

  void SetUp() override {
    // NOLINTNEXTLINE(concurrency-mt-unsafe): test environment is immutable
    const char* scratch = std::getenv("LLMP_TEST_SCRATCH");
    ASSERT_NE(scratch, nullptr);
    std::filesystem::create_directories(scratch);
    ASSERT_TRUE(node_.Open());
    const std::uint64_t fixed = node_.catalog().OccupancyOf(node_.domain()).Total().value();
    const auto stages = Stages(ChunkTap());
    ts::FakeSourceOptions o;
    for (const auto& s : stages) o.layouts.push_back(s.layout);
    o.features = true;
    // A computed tap cells' and frozen mask's slots on one test's shape
    // (measured at setup, so the staging holds them).
    o.extra_inputs = [](const pl::StepShape& shape) {
      std::vector<kn::InputSlot> extra;
      if (shape.stage != 0 || shape != ComputedShape()) return extra;
      extra.push_back({.bytes = 2 * sizeof(std::int64_t),
                       .segment = 0,
                       .kind = kn::InputKind::kStoreCells,
                       .cells = {.capacity = kTapCells, .ring = true},
                       .outputs = 2});
      extra.push_back({.bytes = 256 * 32 * sizeof(std::uint16_t),
                       .segment = 0,
                       .kind = kn::InputKind::kMask,
                       .cells = {.capacity = kContext, .ring = false, .frozen = true},
                       .read = 256,
                       .mask_rows = 32});
      return extra;
    };
    auto source = std::make_unique<ts::FakeStepSource>(std::move(o));
    source_ = source.get();
    en::RunnerConfig config;
    config.slots = {.model = "Fake",
                    .slots = kSlots,
                    .stages = stages,
                    .spec = {},
                    .read_align = 256,
                    .spill_place = {},
                    .spill_directory = scratch,
                    .invalidate = {},
                    .drop_plans = {}};
    config.setup_shapes = {Wide(0, pl::Phase::kPrefill, kChunk),
                           Wide(0, pl::Phase::kVerify, kDepth + 1),
                           Wide(kAssistant, pl::Phase::kDraft, ChunkTap() ? kDepth + 1 : 1)};
    // A target's prompt chunk returns its last row only.
    for (pl::Segment& s : config.setup_shapes.front().segments) s.outputs = 1;
    config.setup_shapes.push_back(ComputedShape());
    config.launch =
        [](std::uint64_t) -> std::expected<std::unique_ptr<kn::StepLaunch>, std::string> {
      return std::make_unique<kn::RecordingLaunch>();
    };
    config.max_verify_rows = VerifyRows();
    auto runner = en::ModelRunner::Create(node_, std::move(config), std::move(source), 0, 0);
    ASSERT_TRUE(runner.has_value()) << runner.error();
    runner_ = std::move(*runner);
    auto set = runner_->Setup();
    if (VerifyRows() > kTapCells) {
      ASSERT_FALSE(set);
      setup_refusal_ = ts::Failed(set).value_or("");
      EXPECT_TRUE(runner_->Release());
      runner_.reset();
      return;
    }
    ASSERT_TRUE(set) << ts::Failed(set).value_or("");
    ASSERT_TRUE(node_.MapWorkspace(runner_->activations_needed(),
                                   std::max(runner_->pool_needed(), kExtent)));
    ASSERT_TRUE(node_.Start(Bytes(fixed + (32 * kExtent))));
    ASSERT_TRUE(runner_->Register());
    node_.Run();
    ASSERT_TRUE(runner_->Bind());
    ASSERT_TRUE(runner_->slots().SelectSlots(std::vector<std::uint32_t>{0, 1}));
    ASSERT_TRUE(node_.BeginRequest(0, runner_->slots().execution(), "test request"));
    md::Batching caps;
    caps.max_segments = kSlots;
    auto composer = en::WaveComposer::For(caps, runner_->slots());
    ASSERT_TRUE(composer.has_value()) << composer.error();
    composer_.emplace(std::move(*composer));
  }
  void TearDown() override {
    if (runner_ == nullptr) {
      // A refused Setup leaves an opened node that never started.
      auto closed = node_.TearDown({});
      EXPECT_TRUE(closed) << ts::Failed(closed).value_or("");
      return;
    }
    if (node_.InRequest(0)) EXPECT_TRUE(node_.EndRequest(0));
    std::array<en::PagedModel*, 1> models = {runner_.get()};
    auto closed = node_.TearDown(models);
    EXPECT_TRUE(closed) << ts::Failed(closed).value_or("");
  }

  // Each slot's prompt of `lengths[slot]` truths, its anchor the next.
  void Prefill(std::span<const std::uint32_t> lengths) {
    en::PrefillOrchestrator prefill(*runner_, *composer_,
                                    {.chunk_rows = kChunk,
                                     .lookahead = 0,
                                     .capture_ahead = false,
                                     .prepare_ahead = false,
                                     .state_only = true});
    std::vector<en::PromptWork> prompts;
    for (std::uint32_t slot = 0; slot < lengths.size(); ++slot) {
      history_[slot].clear();
      for (std::uint32_t q = 0; q < lengths[slot]; ++q) history_[slot].push_back(Truth(q));
      prompts.push_back({.slot = slot, .stage = 0, .tokens = history_[slot], .want_head = true});
    }
    // A unit a chunk, until every prompt's last.
    std::vector<bool> done(lengths.size(), false);
    while (!std::ranges::all_of(done, [](bool d) { return d; })) {
      std::vector<en::PromptWork> open;
      std::vector<std::uint32_t> of;
      for (std::uint32_t slot = 0; slot < lengths.size(); ++slot) {
        if (done[slot]) continue;
        en::PromptWork w = prompts[slot];
        w.past = runner_->slots().positions(slot);
        w.tokens = std::span<const std::int32_t>(history_[slot]).subspan(w.past);
        open.push_back(w);
        of.push_back(slot);
      }
      const en::PrefillReport report = prefill.Step(open);
      ASSERT_FALSE(report.refusal);
      for (std::size_t j = 0; j < of.size(); ++j) {
        ASSERT_FALSE(report.outcomes()[j].refusal.has_value())
            << report.outcomes()[j].refusal->what;
        done[of[j]] = report.outcomes()[j].last;
      }
    }
    for (std::uint32_t slot = 0; slot < lengths.size(); ++slot) {
      history_[slot].push_back(Truth(lengths[slot]));
    }
  }

  en::DecodingOrchestrator Assistant() {
    en::DecodingPolicy policy;
    // Each pass proposes the truth after its row's token (anchored rows
    // all sit at the anchor's position).
    policy.propose =
        [this](std::uint32_t slot, std::size_t, std::span<const std::byte> out,
               std::uint32_t want) -> std::expected<std::vector<std::int32_t>, std::string> {
      const auto rows = Rows(out.first(std::min<std::size_t>(out.size(), sizeof(ts::FakeRow))));
      drafts_.at(slot).insert(drafts_.at(slot).end(), rows.begin(), rows.end());
      std::vector<std::int32_t> proposed;
      for (const auto& row : rows) proposed.push_back(row.token + 1);
      if (proposed.size() > want) proposed.resize(want);
      return proposed;
    };
    policy.judge = [](std::uint32_t, std::span<const std::int32_t> verified,
                      std::span<const std::byte> out) -> std::expected<en::Verdict, std::string> {
      const auto rows = Rows(out);
      if (rows.size() != verified.size()) return std::unexpected(std::string("rows differ"));
      const std::int64_t at = rows.front().position;
      std::uint32_t accepted = 0;
      while (accepted + 1 < verified.size() && verified[accepted + 1] == Truth(at + accepted + 1)) {
        ++accepted;
      }
      en::Verdict v{.keep = accepted + 1, .tokens = {}};
      v.tokens.assign(verified.begin() + 1, verified.begin() + 1 + accepted);
      v.tokens.push_back(Truth(at + accepted + 1));
      return v;
    };
    auto made = en::DecodingOrchestrator::Make(
        *runner_, *composer_,
        {.kind = en::DecodingMode::Kind::kSpeculative,
         .drafters = {{.stage = kAssistant,
                       .shape = en::DrafterSpec::Shape::kAutoregressive,
                       .max_depth = kDepth,
                       .borrows_target = true,
                       .anchored = true,
                       .features = true}}},
        std::move(policy));
    EXPECT_TRUE(made.has_value()) << made.error();
    return std::move(*made);
  }

  en::DecodeReport Unit(en::DecodingOrchestrator& o) {
    std::vector<en::DecodeWork> work;
    for (std::uint32_t slot = 0; slot < kSlots; ++slot) {
      work.push_back({.slot = slot, .history = history_[slot]});
    }
    en::DecodeReport report = o.Step(work);
    for (std::uint32_t slot = 0; slot < kSlots; ++slot) {
      const auto& out = report.outcomes[slot];
      if (out.refusal || out.kept == 0) continue;
      history_[slot].insert(history_[slot].end(), out.tokens.begin(), out.tokens.end());
    }
    return report;
  }

  std::string setup_refusal_;
};

TEST_F(AnchoredDrafterTest, PassesSitAtTheAnchorReadingTheTapThenTheirCarry) {
  // One prompt ends at a multiple of 256: its drafts read exactly its past.
  const std::array<std::uint32_t, kSlots> prompts = {40, 256};
  Prefill(prompts);
  auto o = Assistant();
  for (int unit = 0; unit < 2; ++unit) {
    const std::size_t built = source_->built().size();
    std::array<std::uint32_t, kSlots> past{};
    for (std::uint32_t slot = 0; slot < kSlots; ++slot) {
      past[slot] = runner_->slots().positions(slot);
      drafts_[slot].clear();
    }
    const en::DecodeReport report = Unit(o);
    for (std::uint32_t slot = 0; slot < kSlots; ++slot) {
      const en::DecodeOutcome& out = report.outcomes[slot];
      ASSERT_FALSE(out.refusal.has_value()) << out.refusal->what;
      ASSERT_EQ(out.drafts.size(), 1U);
      EXPECT_EQ(out.drafts.front().proposed, kDepth);
      // Every draft accepted: the anchor and three drafts kept.
      EXPECT_EQ(out.kept, kDepth + 1);
      EXPECT_EQ(runner_->slots().positions(slot), past[slot] + kDepth + 1);
      const auto& rows = drafts_[slot];
      ASSERT_EQ(rows.size(), kDepth);
      for (std::uint32_t pass = 0; pass < kDepth; ++pass) {
        // At the anchor's position, its token the anchor, then each proposal.
        EXPECT_EQ(rows[pass].position, static_cast<std::int32_t>(past[slot])) << pass;
        EXPECT_EQ(rows[pass].token, Truth(past[slot] + pass)) << pass;
        // The target's feature of the position before, then its own last
        // next feature.
        EXPECT_EQ(rows[pass].tap, pass == 0 ? Truth(past[slot] - 1)
                                            : Truth(past[slot] + pass - 1) + ts::kFakeFeatureStep)
            << "slot " << slot << " pass " << pass;
      }
    }
    // The drafts' shapes: one joined step a pass, each owner at its anchor,
    // reading the target's cells before it.
    std::size_t drafts = 0;
    for (std::size_t i = built; i < source_->built().size(); ++i) {
      const auto& b = source_->built()[i];
      if (b.measuring || b.shape.stage != kAssistant) continue;
      ++drafts;
      ASSERT_EQ(b.shape.segments.size(), kSlots);
      for (const pl::Segment& s : b.shape.segments) {
        EXPECT_EQ(s.past, past[s.slot]);
        EXPECT_EQ(s.rows, 1U);
        EXPECT_EQ(s.read_cells, (past[s.slot] + 255) / 256 * 256);
      }
    }
    // Built once a unit at most: its shape repeats in every pass.
    EXPECT_LE(drafts, 1U) << unit;
  }
  // Slot 1's drafts read exactly its past of 256 (above); its history grew a unit's tokens twice.
  EXPECT_EQ(history_[1].size(), 256U + (2 * (kDepth + 1)) + 1);
}

TEST_F(AnchoredDrafterTest, FeaturesNothingHoldsAreRefused) {
  const std::array<std::uint32_t, kSlots> prompts = {40, 41};
  Prefill(prompts);
  std::vector<ts::FakeRow> drafted;
  const auto draft = [&](en::FeatureFrom from) -> std::string {
    auto borrow = runner_->slots().Borrow(0, 0);
    EXPECT_TRUE(borrow.has_value());
    const std::uint32_t past = runner_->slots().positions(0);
    const std::array<std::int32_t, 1> token = {history_[0].back()};
    const std::array<en::SegmentWork, 1> work = {
        en::SegmentWork{.tokens = token, .bytes = {}, .feature = from}};
    const pl::StepShape shape{.stage = kAssistant,
                              .phase = pl::Phase::kDraft,
                              .segments = {{.slot = 0,
                                            .rows = 1,
                                            .past = past,
                                            .read_cells = 256,
                                            .outputs = 1,
                                            .want_head = true}},
                              .read_align = 256};
    auto ran = runner_->RunStep(
        {.shape = shape, .work = work, .step_bytes = {}, .advance = false, .ahead = {}});
    if (ran) {
      const auto rows = Rows(std::span(ran->outputs.front()).first(sizeof(ts::FakeRow)));
      drafted.insert(drafted.end(), rows.begin(), rows.end());
    }
    return ran ? std::string() : ran.error().what;
  };
  EXPECT_NE(draft({}).find("does not name"), std::string::npos);
  EXPECT_NE(draft({.kind = en::FeatureFrom::Kind::kCarry}).find("carries no features"),
            std::string::npos);
  // Not yet committed, and long since out of the ring.
  EXPECT_NE(draft({.kind = en::FeatureFrom::Kind::kTap, .position = 40}).find("tap holds no"),
            std::string::npos);
  EXPECT_NE(draft({.kind = en::FeatureFrom::Kind::kTap, .position = 3}).find("tap holds no"),
            std::string::npos);
  // The tap of the last committed position, then the carry it leaves.
  EXPECT_EQ(draft({.kind = en::FeatureFrom::Kind::kTap, .position = 39}), "");
  EXPECT_EQ(draft({.kind = en::FeatureFrom::Kind::kCarry}), "");
  // Spilling preserves both the logical state and the drafter's runtime
  // carry, so catch-up can continue from exactly its previous features.
  ASSERT_TRUE(runner_->slots().Spill(0));
  ASSERT_TRUE(runner_->slots().Restore(0));
  EXPECT_EQ(draft({.kind = en::FeatureFrom::Kind::kCarry}), "");
  ASSERT_GE(drafted.size(), 3U);
  EXPECT_EQ(drafted.back().tap, drafted[drafted.size() - 2].token + ts::kFakeFeatureStep);
  // A slot whose logical state was replaced carries nothing.
  ASSERT_TRUE(runner_->slots().Clear(0));
  EXPECT_NE(draft({.kind = en::FeatureFrom::Kind::kCarry}).find("carries no features"),
            std::string::npos);
}

// A drafter's rows reach its slot's committed positions (a catch-up runs
// through them, a pass sits at or past them): a draft step wholly behind
// them is refused before anything runs; one at them is admitted.
TEST_F(AnchoredDrafterTest, ADraftWhollyBehindTheCommittedPositionsIsRefused) {
  const std::array<std::uint32_t, kSlots> prompts = {40, 41};
  Prefill(prompts);
  auto borrow = runner_->slots().Borrow(0, 0);
  ASSERT_TRUE(borrow.has_value());
  const std::uint32_t positions = runner_->slots().positions(0);
  const auto draft = [&](std::uint32_t past) -> std::string {
    const std::array<std::int32_t, 1> token = {history_[0].back()};
    const std::array<en::SegmentWork, 1> work = {en::SegmentWork{
        .tokens = token,
        .bytes = {},
        .feature = {.kind = en::FeatureFrom::Kind::kTap, .position = positions - 1}}};
    const pl::StepShape shape{.stage = kAssistant,
                              .phase = pl::Phase::kDraft,
                              .segments = {{.slot = 0,
                                            .rows = 1,
                                            .past = past,
                                            .read_cells = 256,
                                            .outputs = 1,
                                            .want_head = true}},
                              .read_align = 256};
    auto ran = runner_->RunStep(
        {.shape = shape, .work = work, .step_bytes = {}, .advance = false, .ahead = {}});
    return ran ? std::string() : ran.error().what;
  };
  EXPECT_NE(draft(positions - 2).find("behind its"), std::string::npos);
  EXPECT_NE(draft(0).find("behind its"), std::string::npos);
  EXPECT_EQ(draft(positions), "");
}

TEST_F(AnchoredDrafterTest, ATargetsOutputsFitItsDraftersTap) {
  const std::array<std::int32_t, 20> tokens{};
  const std::array<en::SegmentWork, 1> work = {en::SegmentWork{.tokens = tokens}};
  pl::StepShape shape{
      .stage = 0,
      .phase = pl::Phase::kScore,
      .segments =
          {{.slot = 0, .rows = 20, .past = 0, .read_cells = 256, .outputs = 20, .want_head = true}},
      .read_align = 256};
  auto ran = runner_->RunStep(
      {.shape = shape, .work = work, .step_bytes = {}, .advance = true, .ahead = {}});
  ASSERT_FALSE(ran.has_value());
  EXPECT_NE(ran.error().what.find("feature tap of 16 cells"), std::string::npos)
      << ran.error().what;
  shape.segments.front().outputs = kTapCells;
  EXPECT_TRUE(
      runner_
          ->RunStep({.shape = shape, .work = work, .step_bytes = {}, .advance = true, .ahead = {}})
          .has_value());
}

class WideVerifyTest : public AnchoredDrafterTest {
 protected:
  std::uint32_t VerifyRows() const override { return kTapCells + 1; }
};

TEST_F(WideVerifyTest, AVerifyWiderThanTheTapIsRefusedAtSetup) {
  EXPECT_NE(setup_refusal_.find("feature tap of 16 cells"), std::string::npos) << setup_refusal_;
}

// The computed inputs a runner stages: a tap's store cells for the output
// rows alone, and a frozen read's mask over the cells before its rows.
TEST_F(AnchoredDrafterTest, ComputedInputsStageTapCellsAndFrozenMasks) {
  const std::array<std::uint32_t, kSlots> prompts = {40, 41};
  Prefill(prompts);
  const std::array<std::int32_t, 3> tokens = {Truth(40), Truth(41), Truth(42)};
  const std::array<en::SegmentWork, 1> work = {en::SegmentWork{.tokens = tokens}};
  const pl::StepShape shape = ComputedShape();
  auto ran = runner_->RunStep(
      {.shape = shape, .work = work, .step_bytes = {}, .advance = true, .ahead = {}});
  ASSERT_TRUE(ran.has_value()) << ran.error().what;
  const auto& seen = *source_->options().seen_inputs;
  ASSERT_EQ(seen.size(), 2U);
  // The last two rows' positions (41, 42), in a ring of 16.
  std::array<std::int64_t, 2> cells{};
  ASSERT_EQ(seen[0].size(), sizeof(cells));
  std::memcpy(cells.data(), seen[0].data(), sizeof(cells));
  EXPECT_EQ(cells, (std::array<std::int64_t, 2>{41 % kTapCells, 42 % kTapCells}));
  // Frozen: every row sees the cells before the segment's rows only.
  std::vector<std::uint16_t> mask(256 * 32);
  ASSERT_EQ(seen[1].size(), mask.size() * sizeof(std::uint16_t));
  std::memcpy(mask.data(), seen[1].data(), seen[1].size());
  for (std::uint32_t r = 0; r < 3; ++r) {
    for (std::uint32_t c = 0; c < 256; ++c) {
      EXPECT_EQ(mask[(r * 256) + c], c < 40 ? 0x0000 : 0xFC00) << "row " << r << " cell " << c;
    }
  }
}

class ChunkFeatureTest : public AnchoredDrafterTest {
 protected:
  bool ChunkTap() const override { return true; }
  std::expected<en::StepReport, en::StepRefusal> Target(std::uint32_t slot, std::uint32_t rows,
                                                        bool verify = false) {
    const auto past = runner_->slots().positions(slot);
    std::vector<std::int32_t> tokens;
    for (std::uint32_t r = 0; r < rows; ++r)
      tokens.push_back(Truth(past + r) + 1000 * static_cast<std::int32_t>(slot));
    const std::array<en::SegmentWork, 1> work{{{.tokens = tokens}}};
    return runner_->RunStep({.shape = {.stage = 0,
                                       .phase = verify ? pl::Phase::kVerify : pl::Phase::kPrefill,
                                       .segments = {{.slot = static_cast<std::uint8_t>(slot),
                                                     .rows = rows,
                                                     .past = past,
                                                     .read_cells = 256,
                                                     .outputs = rows,
                                                     .want_head = true}},
                                       .read_align = 256},
                             .work = work,
                             .advance = !verify,
                             .ahead = {}});
  }
  std::expected<en::StepReport, en::StepRefusal> Draft(std::uint32_t slot, std::uint32_t first,
                                                       std::uint32_t rows) {
    EXPECT_TRUE(runner_->slots().Settle(slot));
    auto borrow = runner_->slots().Borrow(slot, 0);
    EXPECT_TRUE(borrow);
    std::vector<std::int32_t> tokens(rows, 7);
    const std::array<en::SegmentWork, 1> work{
        {{.tokens = tokens, .feature = {.kind = en::FeatureFrom::Kind::kTap, .position = first}}}};
    return runner_->RunStep({.shape = {.stage = kAssistant,
                                       .phase = pl::Phase::kDraft,
                                       .segments = {{.slot = static_cast<std::uint8_t>(slot),
                                                     .rows = rows,
                                                     .past = first + 1,
                                                     .read_cells = 256,
                                                     .outputs = rows,
                                                     .want_head = true}},
                                       .read_align = 256},
                             .work = work,
                             .advance = false,
                             .ahead = {}});
  }
  void ExpectFeatures(std::uint32_t slot, std::uint32_t first, std::uint32_t rows) {
    auto result = Draft(slot, first, rows);
    ASSERT_TRUE(result) << result.error().what;
    const auto values = Rows(std::span(result->outputs.front()).first(rows * sizeof(ts::FakeRow)));
    ASSERT_EQ(values.size(), rows);
    for (std::uint32_t r = 0; r < rows; ++r)
      EXPECT_EQ(values[r].tap, Truth(first + r) + 1000 * static_cast<std::int32_t>(slot));
  }
};

TEST_F(ChunkFeatureTest, CatchupReadsCommittedChunkRowsAndPreviousTailAcrossAcceptAndDiscard) {
  ASSERT_TRUE(Target(0, 4));
  ASSERT_TRUE(Target(1, 5));
  ExpectFeatures(0, 0, 4);
  ASSERT_TRUE(Target(0, 3, true));
  ASSERT_TRUE(runner_->slots().Accept(0, 2));
  // Persist and restore even while the accepted verify owes restoration.
  ASSERT_TRUE(runner_->slots().Spill(0));
  ASSERT_TRUE(runner_->slots().Restore(0));
  // A different owner overwrites shared activations, never this tap.
  ASSERT_TRUE(Target(1, 2));
  ExpectFeatures(0, 3, 3);
  ExpectFeatures(1, 4, 3);
  auto rejected = Draft(0, 2, 1);
  ASSERT_FALSE(rejected);
  EXPECT_TRUE(rejected.error().clean);
  ASSERT_TRUE(Target(0, 3, true));
  ASSERT_TRUE(runner_->slots().DiscardVerify(0));
  ExpectFeatures(0, 3, 3);
  ASSERT_TRUE(Target(0, 2));
  ExpectFeatures(0, 5, 3);
}

TEST_F(ChunkFeatureTest, CheckpointReplacementRefusesAnUnrestoredLogicalTapSpan) {
  ASSERT_TRUE(Target(0, 4));
  std::vector<en::LiveState::Range> footprint;
  for (std::size_t stage = 0; stage < runner_->slots().stages(); ++stage) {
    for (auto range : runner_->slots().live(0, stage).used_ranges()) {
      range.region = stage;
      footprint.push_back(range);
    }
  }
  auto pages = runner_->slots().CheckpointRanges(0, 4);
  ASSERT_TRUE(pages);
  std::uint64_t bytes = 0;
  for (const auto& page : *pages) bytes += page.bytes;
  std::vector<llmp::catalog::ExtentId> staging;
  auto host = node_.Pinned(bytes, 0, staging);
  ASSERT_TRUE(host);
  ASSERT_TRUE(runner_->slots().CopyState(0, *host, *pages, true));
  ASSERT_TRUE(Target(0, 3));
  ExpectFeatures(0, 6, 1);
  ASSERT_TRUE(runner_->slots().PrepareRestore(0, 4, footprint, runner_->slots().kept_layout_id()));
  ASSERT_TRUE(runner_->slots().CopyState(0, *host, *pages, false));
  ASSERT_TRUE(runner_->slots().CompleteRestore(0, 4));
  ASSERT_TRUE(node_.FreePinned(*host));
  const auto queued = source_->queued();
  const auto refused = Draft(0, 3, 1);
  ASSERT_FALSE(refused);
  EXPECT_TRUE(refused.error().clean);
  EXPECT_EQ(refused.error().what, "a chunk tap does not hold the requested features");
  EXPECT_EQ(source_->queued(), queued);
  EXPECT_EQ(runner_->slots().positions(0), 4U);
  // A new target chunk publishes a new valid span after replacement.
  ASSERT_TRUE(Target(0, 2));
  ExpectFeatures(0, 4, 2);
}

// A restore of exactly the captured state may put back the tap bookkeeping
// recorded with it: the tap's cells are state the checkpoint restores, so the
// drafter catches up from the captured features, as it would have before the
// capture (the swap table's continuation after a diagnostic snapshot restore
// drafted otherwise without them, and so differed in its logits' bits).
TEST_F(ChunkFeatureTest, CheckpointRestoreWithItsTapRowsReadsTheCapturedFeatures) {
  ASSERT_TRUE(Target(0, 4));
  std::vector<en::LiveState::Range> footprint;
  for (std::size_t stage = 0; stage < runner_->slots().stages(); ++stage) {
    for (auto range : runner_->slots().live(0, stage).used_ranges()) {
      range.region = stage;
      footprint.push_back(range);
    }
  }
  auto pages = runner_->slots().CheckpointRanges(0, 4);
  ASSERT_TRUE(pages);
  std::uint64_t bytes = 0;
  for (const auto& page : *pages) bytes += page.bytes;
  std::vector<llmp::catalog::ExtentId> staging;
  auto host = node_.Pinned(bytes, 0, staging);
  ASSERT_TRUE(host);
  ASSERT_TRUE(runner_->slots().CopyState(0, *host, *pages, true));
  const auto taps = runner_->SaveTapRows(0);
  ASSERT_EQ(taps.stages, runner_->slots().stages());
  EXPECT_EQ(taps.positions, 4U);
  ASSERT_TRUE(runner_->OldestTapFeature(0, kAssistant));
  const auto oldest = *runner_->OldestTapFeature(0, kAssistant);
  // A later chunk overwrites the tap's cells, then the restore puts them back.
  ASSERT_TRUE(Target(0, 3));
  ExpectFeatures(0, 6, 1);
  ASSERT_TRUE(runner_->slots().PrepareRestore(0, 4, footprint, runner_->slots().kept_layout_id()));
  ASSERT_TRUE(runner_->slots().CopyState(0, *host, *pages, false));
  ASSERT_TRUE(runner_->slots().CompleteRestore(0, 4));
  ASSERT_TRUE(node_.FreePinned(*host));
  EXPECT_FALSE(runner_->OldestTapFeature(0, kAssistant));
  // Recorded for another state: refused, nothing put back.
  auto other = taps;
  other.positions = 5;
  EXPECT_FALSE(runner_->RestoreTapRows(0, other));
  EXPECT_FALSE(runner_->RestoreTapRows(1, taps));
  EXPECT_FALSE(runner_->OldestTapFeature(0, kAssistant));
  ASSERT_TRUE(runner_->RestoreTapRows(0, taps));
  EXPECT_EQ(runner_->SaveTapRows(0), taps);
  ASSERT_TRUE(runner_->OldestTapFeature(0, kAssistant));
  EXPECT_EQ(*runner_->OldestTapFeature(0, kAssistant), oldest);
  ExpectFeatures(0, 3, 1);
  ExpectFeatures(0, 0, 4);
}

}  // namespace
