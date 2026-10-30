// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Decoding modes (engine/decoding.h) on the fake device backend, over the
// fake step source (tests/support/fake_step_source.h) with a target and
// three drafter stages: an autoregressive drafter with its own cache and a
// feature tap the target writes (MTP-like), a block drafter whose ring the
// target's injection writes (DSpark-like) and an assistant that reads the
// target's cache borrowed (Gemma4-like). Two drafters as a cascade and as
// alternatives; a verify's rows accepted all, some or none, rolled back by
// truncation or by a snapshot and the model's commit; a verify refused in
// a wave, a failed one undone whole, a discarded one; one slot's drafter
// refused while its peers draft; the adaptive depth changing between
// units; the stages' admission rules and per-stage reads; verify waves at
// their owners' lone reads; the single-pass modes; and the prefill's
// outcomes keeping their place in a joined step. Then the challenges: a
// target that saves nothing, a build binding state its stage may not
// write, a cascade's second drafter refused after its first ran, an
// alternative left behind past its chunk, adaptive depth at the runner's
// verify rows, failed drafts and commits, owed work and a borrow across a
// recovery, a slot spilled between its drafts and its verify, and a failed
// verify of a target whose drafter's tap truncates (undone by nothing).

#include "engine/decoding.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <expected>
#include <filesystem>
#include <format>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <span>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "base/bytes.h"
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
#include "providers/fake/fake_device_execution.h"
#include "providers/fake/fake_device_runtime.h"
#include "scheduler/scheduler.h"

namespace {

namespace en = llmp::engine;
namespace ir = llmp::ir;
namespace kn = llmp::kernels;
namespace md = llmp::model;
namespace pl = llmp::planner;
namespace ts = llmp::test_support;
using llmp::base::Bytes;
using C = md::StateTensorContract;
using Layers = std::vector<std::pair<std::uint32_t, md::StateContract>>;
constexpr std::uint64_t kExtent = en::kPagedExtent;
constexpr std::uint32_t kContext = 1024;
constexpr std::uint32_t kSlots = 3;
constexpr std::uint32_t kMaxSlots = 4;
constexpr std::uint32_t kChunk = 64;
constexpr std::uint32_t kVerifyRows = 8;
constexpr std::int32_t kFill = -7;
constexpr std::uint32_t kAll = std::numeric_limits<std::uint32_t>::max();

TEST(DrafterHeadTest, SourceRowsUseImmutableSelectedVocabulary) {
  for (unsigned protocol = 0; protocol != 3; ++protocol) {
    en::DrafterSpec drafter;
    drafter.anchored = protocol == 0;
    drafter.shape =
        protocol == 1 ? en::DrafterSpec::Shape::kBlock : en::DrafterSpec::Shape::kAutoregressive;
    drafter.head_ids = std::make_shared<const std::vector<std::int32_t>>(
        std::initializer_list<std::int32_t>{913, 7, 602});
    drafter.head_rows = 3;
    auto retained = drafter;
    drafter.head_ids.reset();
    const auto first = en::DrafterHeadToken(retained, 0, 1000);
    const auto last = en::DrafterHeadToken(retained, 2, 1000);
    ASSERT_TRUE(first);
    ASSERT_TRUE(last);
    EXPECT_EQ(*first, 913);
    EXPECT_EQ(*last, 602);
    EXPECT_FALSE(en::DrafterHeadToken(retained, 3, 1000));
    EXPECT_FALSE(en::DrafterHeadToken(retained, 0, 600));
  }
}

TEST(DrafterHeadTest, OrdinaryHeadRetainsIdentityAndBounds) {
  en::DrafterSpec drafter;
  const auto ordinary = en::DrafterHeadToken(drafter, 17, 32);
  ASSERT_TRUE(ordinary);
  EXPECT_EQ(*ordinary, 17);
  EXPECT_FALSE(en::DrafterHeadToken(drafter, 32, 32));
  EXPECT_FALSE(en::DrafterHeadToken(drafter, 0, 0));
  drafter.head_ids = std::make_shared<const std::vector<std::int32_t>>();
  EXPECT_EQ(en::DrafterHeadToken(drafter, 17, 32).value_or(-1), 17);
  drafter.head_rows = 8;
  EXPECT_EQ(en::DrafterHeadRows(drafter, 32).value_or(0), 8U);
  EXPECT_EQ(en::DrafterHeadToken(drafter, 7, 32).value_or(-1), 7);
  EXPECT_FALSE(en::DrafterHeadToken(drafter, 8, 32));
  drafter.head_rows = 33;
  EXPECT_FALSE(en::DrafterHeadRows(drafter, 32));
}

TEST(DrafterHeadTest, RefusesInvalidMappedTokenAndDomain) {
  en::DrafterSpec drafter;
  drafter.head_ids = std::make_shared<const std::vector<std::int32_t>>(
      std::initializer_list<std::int32_t>{-1, 32, 7});
  drafter.head_rows = 2;
  EXPECT_FALSE(en::DrafterHeadRows(drafter, 32));
  drafter.head_rows = 3;
  EXPECT_FALSE(en::DrafterHeadToken(drafter, 0, 32));
  EXPECT_FALSE(en::DrafterHeadToken(drafter, 1, 32));
  EXPECT_FALSE(en::DrafterHeadToken(drafter, 2, 2));
  EXPECT_FALSE(en::DrafterHeadToken(drafter, 0, UINT32_MAX));
}

// The stages.
constexpr std::uint16_t kMtp = 1, kBlock = 2, kAssistant = 3;

// What the fake drafters and the judge agree is the token at position q.
std::int32_t Truth(std::int64_t q) { return static_cast<std::int32_t>(500 + q); }

std::vector<ts::FakeRow> Rows(std::span<const std::byte> bytes) {
  std::vector<ts::FakeRow> rows(bytes.size() / sizeof(ts::FakeRow));
  if (!rows.empty()) std::memcpy(rows.data(), bytes.data(), rows.size() * sizeof(ts::FakeRow));
  return rows;
}

std::uint64_t Hash(std::span<const std::int32_t> tokens) {
  std::uint64_t h = 0;
  for (const std::int32_t t : tokens) h = ts::FakeHash(h, t);
  return h;
}

// The target: a cache (rolled back as the fixture says) and a fixed state
// rewritten by every row (snapshot, rebuilt by the commit). The MTP-like
// drafter: its own cache (truncated: its catch-up rewrites it) and a tap
// of the target's features (a position ring the target writes), read at
// 512. The DSpark-like drafter: a ring both it and the target's injection
// write. The assistant: a small ring of its own and a view of the
// target's cache.
std::vector<en::StageState> StageStates(md::Rollback kv, bool fixed,
                                        md::Rollback tap = md::Rollback::kSnapshot,
                                        md::Rollback state = md::Rollback::kSnapshot) {
  Layers target = {
      {0,
       {.tensors = {C{.name = "kv", .kind = md::StateKind::kKv, .cell_bytes = 256, .rollback = kv},
                    C{.name = "state",
                      .kind = md::StateKind::kRecurrent,
                      .encoding = md::Encoding::kF32,
                      .cell_bytes = 256,
                      .cells = md::CellRule::kFixed,
                      .fixed_cells = 1,
                      .rollback = state}}}}};
  if (!fixed) target[0].second.tensors.pop_back();
  const Layers mtp = {{0,
                       {.tensors = {C{.name = "kv", .kind = md::StateKind::kKv, .cell_bytes = 256},
                                    C{.name = "streams",
                                      .kind = md::StateKind::kFeatures,
                                      .encoding = md::Encoding::kF32,
                                      .cell_bytes = 256,
                                      .cells = md::CellRule::kPositionRing,
                                      .fixed_cells = 64,
                                      .rollback = tap,
                                      .target_writes = true}}}}};
  const Layers block = {{0,
                         {.tensors = {C{.name = "ring",
                                        .kind = md::StateKind::kDrafterRing,
                                        .cell_bytes = 256,
                                        .cells = md::CellRule::kPositionRing,
                                        .fixed_cells = 256,
                                        .rollback = md::Rollback::kSnapshot,
                                        .target_writes = true}}}}};
  const Layers assistant = {{0,
                             {.tensors = {C{.name = "feature",
                                            .kind = md::StateKind::kFeatures,
                                            .cell_bytes = 256,
                                            .cells = md::CellRule::kPositionRing,
                                            .fixed_cells = 16},
                                          C{.name = "kv",
                                            .kind = md::StateKind::kKv,
                                            .cell_bytes = 256,
                                            .borrowed_from_target = true,
                                            .target_layer = 0},
                                          C{.name = "rows",
                                            .kind = md::StateKind::kFeatures,
                                            .cell_bytes = 256,
                                            .cells = md::CellRule::kPerRow,
                                            .row_carry = 1}}}}};
  return {
      {.name = "target", .layout = md::Layout(target, kContext, kChunk).value(), .read_align = 0},
      {.name = "mtp", .layout = md::Layout(mtp, kContext, kChunk).value(), .read_align = 512},
      {.name = "block", .layout = md::Layout(block, kContext, kChunk).value(), .read_align = 0},
      {.name = "assistant",
       .layout = md::Layout(assistant, kContext, kChunk).value(),
       .read_align = 0}};
}

pl::StepShape Shape(std::uint16_t stage, pl::Phase phase, std::vector<pl::Segment> segments,
                    std::uint32_t align = 256, bool row_invariant = false) {
  return {.stage = stage,
          .phase = phase,
          .segments = std::move(segments),
          .row_invariant = row_invariant,
          .read_align = align};
}

pl::Segment Seg(std::uint8_t slot, std::uint32_t past, std::uint32_t rows, std::uint32_t align) {
  return {.slot = slot,
          .rows = rows,
          .past = past,
          .read_cells = static_cast<std::uint32_t>(std::min<std::uint64_t>(
              (std::uint64_t{past} + rows + align - 1) / align * align, kContext)),
          .outputs = rows,
          .group = 0,
          .want_head = true,
          .greedy = false};
}

md::Batching Caps(std::uint32_t slots = kSlots) {
  md::Batching caps;
  caps.max_segments = slots;
  return caps;
}

en::DrafterSpec Mtp(std::uint32_t depth) {
  return {.stage = kMtp,
          .shape = en::DrafterSpec::Shape::kAutoregressive,
          .max_depth = depth,
          .catch_up = true};
}
en::DrafterSpec Block(std::uint32_t depth) {
  return {
      .stage = kBlock, .shape = en::DrafterSpec::Shape::kBlock, .max_depth = depth, .fill = kFill};
}

class DecodingFixture : public ::testing::Test {
 protected:
  explicit DecodingFixture(bool direct = true)
      : node_(en::NodeSettings{.compute_streams = 1,
                               .slots = 2,
                               .copy_lane = true,
                               .slot_bytes = kExtent,
                               .direct_steps = direct}) {}
  virtual std::uint32_t Slots() const { return kSlots; }

  en::PagedNode node_;
  std::unique_ptr<en::ModelRunner> runner_;
  ts::FakeStepSource* source_ = nullptr;
  std::uint64_t fixed_ = 0;
  std::optional<en::WaveComposer> composer_;

  // The fake judge's and drafters' record.
  std::array<std::uint32_t, kMaxSlots> accept_{kAll, kAll, kAll, kAll};  // drafts a verify may keep
  bool refuse_judge_ = false;
  std::array<std::vector<std::int32_t>, kMaxSlots> verified_{};
  std::array<std::vector<ts::FakeRow>, kMaxSlots> verify_rows_{};
  std::map<std::pair<std::uint32_t, std::size_t>, std::vector<ts::FakeRow>> draft_rows_;
  std::function<void(std::uint32_t slot, std::size_t drafter)> in_propose_;
  std::uint64_t commits_ = 0;
  bool fail_commit_ = false;  // the commit refuses (nothing queued)
  // Each slot's tokens through its anchor.
  std::array<std::vector<std::int32_t>, kMaxSlots> history_{};

  virtual md::Rollback KvRollback() const = 0;
  // The target's fixed state (a snapshot its commit rebuilds); without it
  // the target has no commit.
  virtual bool TargetFixed() const { return true; }
  // The MTP's tap's rollback.
  virtual md::Rollback TapRollback() const { return md::Rollback::kSnapshot; }
  // The target's fixed state's.
  virtual md::Rollback FixedRollback() const { return md::Rollback::kSnapshot; }
  // The runner's Setup refuses these stages (setup_refusal_ says why).
  virtual bool RefusesSetup() const { return false; }
  std::string setup_refusal_;

  void SetUp() override {
    // NOLINTNEXTLINE(concurrency-mt-unsafe): test environment is immutable
    const char* scratch = std::getenv("LLMP_TEST_SCRATCH");
    ASSERT_NE(scratch, nullptr);
    std::filesystem::create_directories(scratch);
    ASSERT_TRUE(node_.Open());
    fixed_ = node_.catalog().OccupancyOf(node_.domain()).Total().value();
    const auto stages = StageStates(KvRollback(), TargetFixed(), TapRollback(), FixedRollback());
    ts::FakeSourceOptions o;
    for (const auto& s : stages) o.layouts.push_back(s.layout);
    auto source = std::make_unique<ts::FakeStepSource>(std::move(o));
    source_ = source.get();
    const std::uint32_t slots = Slots();
    en::RunnerConfig config;
    config.slots = {.model = "Fake",
                    .slots = slots,
                    .stages = stages,
                    .spec = {},
                    .read_align = 256,
                    .spill_place = {},
                    .spill_directory = scratch,
                    .invalidate = {},
                    .drop_plans = {}};
    const auto wide = [slots](std::uint16_t stage, pl::Phase phase, std::uint32_t rows,
                              std::uint32_t align) {
      std::vector<pl::Segment> segments;
      for (std::uint8_t s = 0; s < slots; ++s)
        segments.push_back(Seg(s, kContext - rows, rows, align));
      return Shape(stage, phase, std::move(segments), align);
    };
    config.setup_shapes = {wide(0, pl::Phase::kPrefill, kChunk, 256),
                           wide(0, pl::Phase::kVerify, kVerifyRows, 256),
                           wide(kMtp, pl::Phase::kDraft, kChunk, 512),
                           wide(kBlock, pl::Phase::kDraft, kVerifyRows, 256)};
    config.launch =
        [](std::uint64_t) -> std::expected<std::unique_ptr<kn::StepLaunch>, std::string> {
      return std::make_unique<kn::RecordingLaunch>();
    };
    config.max_verify_rows = kVerifyRows;
    // The target's commit: the fixed state, restored to its state before
    // the verify, folded over the kept rows' tokens.
    config.commit = [this](std::uint32_t slot, std::size_t stage) -> en::LiveState::Commit {
      if (stage != 0 || !TargetFixed()) return {};
      return [this, slot](llmp::providers::NativeStream stream,
                          std::uint32_t keep) -> en::LiveState::Queued {
        if (fail_commit_) {
          return std::unexpected(
              en::LiveState::QueueFailure{.unknown = false, .detail = "the commit refuses"});
        }
        const auto& layout = runner_->slots().layout(0);
        const std::uint64_t at = runner_->slots().live(slot, 0).base(0) + layout.items[1].offset;
        const auto rows =
            static_cast<std::ptrdiff_t>(std::min<std::size_t>(keep, verified_[slot].size()));
        std::vector<std::int32_t> kept(verified_[slot].begin(), verified_[slot].begin() + rows);
        ++commits_;
        auto launched = llmp::providers::fake::FakeDeviceExecution::Launch(stream, [at, kept] {
          std::uint64_t hash = 0;
          std::memcpy(&hash, reinterpret_cast<const void*>(at), sizeof(hash));  // NOLINT
          for (const std::int32_t t : kept) hash = ts::FakeHash(hash, t);
          std::memcpy(reinterpret_cast<void*>(at), &hash, sizeof(hash));  // NOLINT
        });
        if (!launched) {
          return std::unexpected(
              en::LiveState::QueueFailure{.unknown = false, .detail = "the fake refused"});
        }
        return {};
      };
    };
    auto runner = en::ModelRunner::Create(node_, std::move(config), std::move(source), 0, 0);
    ASSERT_TRUE(runner.has_value()) << runner.error();
    runner_ = std::move(*runner);
    auto set = runner_->Setup();
    if (RefusesSetup()) {
      ASSERT_FALSE(set);
      setup_refusal_ = llmp::test_support::Failed(set).value_or("");
      EXPECT_TRUE(runner_->Release());
      runner_.reset();
      return;
    }
    ASSERT_TRUE(set) << llmp::test_support::Failed(set).value_or("");
    ASSERT_TRUE(node_.MapWorkspace(runner_->activations_needed(),
                                   std::max(runner_->pool_needed(), kExtent)));
    auto started = node_.Start(Bytes(fixed_ + (64 * kExtent)));
    ASSERT_TRUE(started) << llmp::test_support::Failed(started).value_or("");
    auto registered = runner_->Register();
    ASSERT_TRUE(registered) << llmp::test_support::Failed(registered).value_or("");
    node_.Run();
    auto bound = runner_->Bind();
    ASSERT_TRUE(bound) << llmp::test_support::Failed(bound).value_or("");
    std::vector<std::uint32_t> selected_slots;
    for (std::uint32_t s = 0; s < slots; ++s) selected_slots.push_back(s);
    auto selected = runner_->slots().SelectSlots(selected_slots);
    ASSERT_TRUE(selected) << llmp::test_support::Failed(selected).value_or("");
    ASSERT_TRUE(node_.BeginRequest(0, runner_->slots().execution(), "test request"));
    auto composer = en::WaveComposer::For(Caps(slots), runner_->slots());
    ASSERT_TRUE(composer.has_value()) << composer.error();
    composer_.emplace(std::move(*composer));
  }
  void TearDown() override {
    if (runner_ == nullptr) {
      // A refused Setup leaves an opened node that never started.
      auto closed = node_.TearDown({});
      EXPECT_TRUE(closed) << llmp::test_support::Failed(closed).value_or("");
      return;
    }
    if (node_.InRequest(0)) EXPECT_TRUE(node_.EndRequest(0));
    std::array<en::PagedModel*, 1> models = {runner_.get()};
    auto closed = node_.TearDown(models);
    EXPECT_TRUE(closed) << llmp::test_support::Failed(closed).value_or("");
  }

  en::SlotLifecycle& slots() { return runner_->slots(); }

  en::DecodingPolicy Policy() {
    en::DecodingPolicy p;
    p.propose = [this](
                    std::uint32_t slot, std::size_t drafter, std::span<const std::byte> out,
                    std::uint32_t want) -> std::expected<std::vector<std::int32_t>, std::string> {
      if (in_propose_) in_propose_(slot, drafter);
      const auto rows = Rows(out);
      auto& kept = draft_rows_[{slot, drafter}];
      kept.insert(kept.end(), rows.begin(), rows.end());
      std::vector<std::int32_t> proposed;
      for (const auto& row : rows) proposed.push_back(Truth(row.position + 1));
      if (proposed.size() > want) proposed.resize(want);
      return proposed;
    };
    p.judge = [this](std::uint32_t slot, std::span<const std::int32_t> verified,
                     std::span<const std::byte> out) -> std::expected<en::Verdict, std::string> {
      if (refuse_judge_) return std::unexpected(std::string("the judge refuses"));
      const auto rows = Rows(out);
      if (rows.size() != verified.size()) return std::unexpected(std::string("rows differ"));
      verified_[slot].assign(verified.begin(), verified.end());
      verify_rows_[slot] = rows;
      // Every draft that agrees with the truth, up to what the test allows.
      const std::int64_t at = rows.front().position;
      std::uint32_t accepted = 0;
      while (accepted + 1 < verified.size() && accepted < accept_[slot] &&
             verified[accepted + 1] == Truth(at + accepted + 1)) {
        ++accepted;
      }
      en::Verdict v{.keep = accepted + 1, .tokens = {}};
      v.tokens.assign(verified.begin() + 1, verified.begin() + 1 + accepted);
      v.tokens.push_back(Truth(at + accepted + 1));
      return v;
    };
    return p;
  }

  en::DecodingOrchestrator Make(en::DecodingMode mode, en::DecodingPolicy policy = {}) {
    if (!policy.propose) policy = Policy();
    auto made =
        en::DecodingOrchestrator::Make(*runner_, *composer_, std::move(mode), std::move(policy));
    EXPECT_TRUE(made.has_value()) << made.error();
    return std::move(*made);
  }

  // Each slot's prompt (`lengths[slot]` tokens of the truth) prefilled,
  // its anchor the next token.
  void Prefill(std::span<const std::uint32_t> lengths, en::DecodingOrchestrator* o = nullptr) {
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
      prompts.push_back({.slot = slot,
                         .stage = 0,
                         .tokens = history_[slot],
                         .past = 0,
                         .want_head = true,
                         .all_outputs = false,
                         .boundary = 0});
    }
    const en::PrefillReport report = prefill.Step(prompts);
    ASSERT_FALSE(report.refusal);
    for (std::uint32_t slot = 0; slot < lengths.size(); ++slot) {
      ASSERT_FALSE(report.outcomes()[slot].refusal.has_value())
          << report.outcomes()[slot].refusal->what;
      ASSERT_TRUE(report.outcomes()[slot].last);
      history_[slot].push_back(Truth(lengths[slot]));
      if (o != nullptr) o->Reset(slot);
    }
  }

  // One unit for these slots; each slot's history follows what it kept.
  en::DecodeReport Unit(en::DecodingOrchestrator& o, std::vector<std::uint32_t> on,
                        std::uint32_t left = kAll) {
    std::vector<en::DecodeWork> work;
    for (const std::uint32_t slot : on) {
      work.push_back({.slot = slot, .history = history_[slot], .left = left});
    }
    en::DecodeReport report = o.Step(work);
    for (std::size_t i = 0; i < on.size(); ++i) {
      const auto& out = report.outcomes[i];
      if (out.refusal || out.kept == 0) continue;
      auto& h = history_[on[i]];
      h.insert(h.end(), out.tokens.begin(), out.tokens.end());
    }
    return report;
  }

  // A stage's tensor's cells [first, first + n), read back (after what the
  // slot owes has run).
  std::vector<ts::FakeCell> Read(std::uint32_t slot, std::size_t stage, std::size_t item,
                                 std::uint64_t first, std::uint64_t n) {
    EXPECT_TRUE(slots().Settle(slot));
    const auto& it = slots().layout(stage).items.at(item);
    const std::uint64_t cell = it.contract.cell_bytes;
    const std::vector<en::LiveState::Range> ranges = {
        {.region = stage, .offset = it.offset + (first * cell), .bytes = n * cell}};
    std::vector<llmp::catalog::ExtentId> staging;
    auto host = node_.Pinned(((n * cell) + kExtent - 1) / kExtent * kExtent, 0, staging);
    EXPECT_TRUE(host.has_value());
    if (!host) return {};
    auto copied = slots().CopyState(slot, *host, ranges, true);
    EXPECT_TRUE(copied) << llmp::test_support::Failed(copied).value_or("");
    std::vector<ts::FakeCell> cells(n);
    for (std::uint64_t i = 0; i < n; ++i) {
      std::memcpy(&cells[i], static_cast<const std::byte*>(*host) + (i * cell),
                  sizeof(ts::FakeCell));
    }
    (void)node_.FreePinned(*host);
    return cells;
  }
  bool Holds(const ts::FakeCell& cell, std::uint32_t slot, std::int64_t position,
             std::int32_t token) {
    return cell.magic == ts::kFakeMagic && cell.slot == static_cast<std::int32_t>(slot) &&
           cell.position == position && cell.token == token;
  }
  std::uint64_t Built(pl::Phase phase, std::uint16_t stage, std::size_t from) const {
    std::uint64_t n = 0;
    for (std::size_t i = from; i < source_->built().size(); ++i) {
      const auto& b = source_->built()[i];
      n += !b.measuring && b.shape.phase == phase && b.shape.stage == stage ? 1 : 0;
    }
    return n;
  }
};

class DecodingTest : public DecodingFixture {
 protected:
  using DecodingFixture::DecodingFixture;
  md::Rollback KvRollback() const override { return md::Rollback::kSnapshot; }
};
class TruncatingTest : public DecodingFixture {
 protected:
  md::Rollback KvRollback() const override { return md::Rollback::kTruncate; }
};
// A target whose every tensor truncates: it saves nothing and commits
// nothing (Gemma's caches).
class TruncatingOnlyTest : public DecodingFixture {
 protected:
  md::Rollback KvRollback() const override { return md::Rollback::kTruncate; }
  bool TargetFixed() const override { return false; }
};

// Besides, the MTP's tap truncates: the target writes that stage and saves
// nothing in it.
class TruncatingTapTest : public TruncatingOnlyTest {
 protected:
  md::Rollback TapRollback() const override { return md::Rollback::kTruncate; }
};

en::DecodingMode Cascade() {
  return {.kind = en::DecodingMode::Kind::kSpeculative,
          .drafters = {Mtp(2), Block(2)},
          .combine = en::DecodingMode::Drafters::kCascade,
          .row_invariant = false};
}

// --------------------------------------------------------------- the cascade

TEST_F(DecodingTest, ACascadeOfTwoDraftersExtendsOneProposalForOneVerify) {
  auto o = Make(Cascade());
  EXPECT_EQ(o.mode().verify_rows(), 5U);
  const std::array<std::uint32_t, 3> prompts = {40, 41, 42};
  Prefill(prompts, &o);
  const std::size_t from = source_->built().size();
  const auto report = Unit(o, {0, 1, 2});
  // Two MTP passes and a DSpark block, joined across the slots, then one
  // joined verify. The passes are one shape: one plan.
  EXPECT_EQ(report.steps, 4U);
  EXPECT_EQ(Built(pl::Phase::kDraft, kMtp, from), 1U);
  EXPECT_EQ(Built(pl::Phase::kDraft, kBlock, from), 1U);
  EXPECT_EQ(Built(pl::Phase::kVerify, 0, from), 1U);
  for (std::uint32_t slot = 0; slot < kSlots; ++slot) {
    const auto& out = report.outcomes[slot];
    ASSERT_FALSE(out.refusal.has_value()) << out.refusal->what;
    const std::uint32_t v = prompts[slot];
    ASSERT_EQ(out.drafts.size(), 2U);
    EXPECT_EQ(out.drafts[0].drafter, 0U);
    EXPECT_EQ(out.drafts[0].proposed, 2U);
    EXPECT_EQ(out.drafts[0].steps, 2U);
    EXPECT_EQ(out.drafts[1].drafter, 1U);
    EXPECT_EQ(out.drafts[1].proposed, 2U);
    EXPECT_EQ(out.drafts[1].steps, 1U);
    // The block extends the MTP's proposals: from the last one known.
    EXPECT_EQ(out.proposed,
              (std::vector<std::int32_t>{Truth(v + 1), Truth(v + 2), Truth(v + 3), Truth(v + 4)}));
    EXPECT_EQ(out.kept, 5U);
    EXPECT_EQ(slots().positions(slot), v + 5);
    EXPECT_FALSE(slots().verify_pending(slot));
    // The block's rows: its anchor, then the fill.
    const auto& block = draft_rows_[{slot, 1}];
    ASSERT_EQ(block.size(), 2U);
    EXPECT_EQ(block[0].position, static_cast<std::int32_t>(v + 2));
    EXPECT_EQ(block[0].token, Truth(v + 2));
    EXPECT_EQ(block[1].token, kFill);
    // The MTP's first pass at the anchor read the target's tap of the
    // position before (written by the prefill); its next, at a proposal,
    // none (the target has not run it).
    const auto& mtp = draft_rows_[{slot, 0}];
    ASSERT_EQ(mtp.size(), 2U);
    EXPECT_EQ(mtp[0].position, static_cast<std::int32_t>(v));
    EXPECT_EQ(mtp[0].tap, Truth(v - 1));
    EXPECT_EQ(mtp[1].tap, -1);
    // The verify's first row continues the committed hash.
    EXPECT_EQ(verify_rows_[slot].front().hash,
              Hash(std::span(history_[slot]).first(std::size_t{v} + 1)));
    // The MTP's catch-up starts at the verify's first row next time.
    EXPECT_EQ(o.cursor(slot, 0), v);
  }
  // The next unit: the MTP catches up the five kept rows and the anchor in
  // its first pass, reading the taps the verify left.
  draft_rows_.clear();
  const auto next = Unit(o, {0, 1, 2});
  for (std::uint32_t slot = 0; slot < kSlots; ++slot) {
    ASSERT_FALSE(next.outcomes[slot].refusal.has_value()) << next.outcomes[slot].refusal->what;
    const std::uint32_t v = prompts[slot] + 5;
    const auto& mtp = draft_rows_[{slot, 0}];
    ASSERT_EQ(mtp.size(), 2U);
    EXPECT_EQ(mtp[0].position, static_cast<std::int32_t>(v));
    EXPECT_EQ(mtp[0].tap, Truth(v - 1));
    EXPECT_EQ(next.outcomes[slot].kept, 5U);
    // The commit rebuilt the fixed state for the kept rows.
    EXPECT_EQ(verify_rows_[slot].front().hash,
              Hash(std::span(history_[slot]).first(std::size_t{v} + 1)));
  }
  bool caught_up = false;
  for (const auto& b : source_->built()) {
    caught_up = caught_up || (b.shape.phase == pl::Phase::kDraft && b.shape.stage == kMtp &&
                              b.shape.segments.front().rows == 6);
  }
  EXPECT_TRUE(caught_up);
  EXPECT_EQ(commits_, 3U);  // each slot's first Accept, run by the next unit's first job
}

TEST_F(DecodingTest, AUnitTheWaveFormMakesPlainDecodesARowEachAndFeedsTheTaps) {
  auto policy = Policy();
  policy.speculate = [](std::uint32_t width) { return width < kSlots; };
  auto o = Make(Cascade(), std::move(policy));
  const std::array<std::uint32_t, 3> prompts = {40, 41, 42};
  Prefill(prompts, &o);
  const auto plain = Unit(o, {0, 1, 2});
  EXPECT_FALSE(plain.speculated);
  EXPECT_EQ(plain.steps, 1U);
  for (std::uint32_t slot = 0; slot < kSlots; ++slot) {
    const auto& out = plain.outcomes[slot];
    ASSERT_FALSE(out.refusal.has_value()) << out.refusal->what;
    EXPECT_EQ(out.kept, 1U);
    EXPECT_TRUE(out.drafts.empty());
    EXPECT_EQ(slots().positions(slot), prompts[slot] + 1);
    history_[slot].push_back(Truth(prompts[slot] + 1));  // the sampler's choice
  }
  // Narrower, it speculates: the MTP catches up the decoded row, whose tap
  // the plain step wrote.
  const auto spec = Unit(o, {0, 1});
  EXPECT_TRUE(spec.speculated);
  for (std::uint32_t slot = 0; slot < 2; ++slot) {
    ASSERT_FALSE(spec.outcomes[slot].refusal.has_value()) << spec.outcomes[slot].refusal->what;
    const auto& mtp = draft_rows_[{slot, 0}];
    ASSERT_FALSE(mtp.empty());
    EXPECT_EQ(mtp.front().position, static_cast<std::int32_t>(prompts[slot] + 1));
    EXPECT_EQ(mtp.front().tap, Truth(prompts[slot]));
    EXPECT_EQ(spec.outcomes[slot].kept, 5U);
  }
}

// ------------------------------------------------------------ alternatives

TEST_F(DecodingTest, AlternativesTakeOneDrafterAUnitByTheAliasPolicyOrWhatEachEarns) {
  auto policy = Policy();
  policy.choose = [](std::uint32_t slot) -> std::optional<std::size_t> {
    return slot == 2 ? std::optional<std::size_t>(1) : std::nullopt;
  };
  auto o = Make({.kind = en::DecodingMode::Kind::kSpeculative,
                 .drafters = {Mtp(2), Block(3)},
                 .combine = en::DecodingMode::Drafters::kAlternatives,
                 .row_invariant = false},
                std::move(policy));
  EXPECT_EQ(o.mode().verify_rows(), 4U);
  const std::array<std::uint32_t, 3> prompts = {30, 31, 32};
  Prefill(prompts, &o);
  std::vector<std::size_t> chosen;
  for (int unit = 0; unit < 10; ++unit) {
    const auto report = Unit(o, {0, 1, 2});
    for (std::uint32_t slot = 0; slot < kSlots; ++slot) {
      const auto& out = report.outcomes[slot];
      ASSERT_FALSE(out.refusal.has_value()) << out.refusal->what;
      ASSERT_EQ(out.drafts.size(), 1U);
      if (slot == 2) EXPECT_EQ(out.drafts[0].drafter, 1U);  // the alias policy's
      if (slot == 0) chosen.push_back(out.drafts[0].drafter);
      EXPECT_EQ(out.kept, out.drafts[0].drafter == 0 ? 3U : 4U);
    }
  }
  // Each explored for four units, then the one committing more a unit.
  EXPECT_EQ(chosen, (std::vector<std::size_t>{0, 0, 0, 0, 1, 1, 1, 1, 1, 1}));
  EXPECT_DOUBLE_EQ(o.choice(0).kept(0), 3.0);
  EXPECT_DOUBLE_EQ(o.choice(0).kept(1), 4.0);
  EXPECT_EQ(o.choice(2).seen(0), 0U);
  // The MTP's catch-up waits where its last unit left it.
  EXPECT_EQ(o.cursor(0, 0), prompts[0] + (3 * 3));
}

// ------------------------------------------------- accept, and roll back

TEST_F(DecodingTest, AcceptAllSomeOrNoneRestoresTheRejectedRowsAndCommitsTheKept) {
  auto o = Make(Cascade());
  const std::array<std::uint32_t, 3> prompts = {40, 40, 40};
  Prefill(prompts, &o);
  accept_ = {4, 1, 0};
  const auto report = Unit(o, {0, 1, 2});
  const std::uint32_t v = 40;
  const std::array<std::uint32_t, 3> kept = {5, 2, 1};
  for (std::uint32_t slot = 0; slot < kSlots; ++slot) {
    const auto& out = report.outcomes[slot];
    ASSERT_FALSE(out.refusal.has_value()) << out.refusal->what;
    EXPECT_EQ(out.kept, kept[slot]);
    EXPECT_EQ(slots().positions(slot), v + kept[slot]);
    EXPECT_TRUE(slots().owed(slot));
    // Saved: each row's cache cell (a snapshot), the fixed state, and each
    // row's tap in both drafters' stages.
    EXPECT_EQ(slots().live(slot, 0).saved().size(), 6U);
    EXPECT_EQ(slots().live(slot, kMtp).saved().size(), 5U);
    EXPECT_EQ(slots().live(slot, kBlock).saved().size(), 5U);
    EXPECT_TRUE(slots().live(slot, kAssistant).saved().empty());
    // Settled, the target's cache holds the kept rows and, past them, what
    // it held before the verify (nothing); each drafter's tap the same (the
    // block's ring: its own block's rows at positions v + 2 and v + 3).
    const auto cache = Read(slot, 0, 0, v, 5);
    const auto taps = Read(slot, kMtp, 1, v, 5);
    const auto ring = Read(slot, kBlock, 0, v, 5);
    EXPECT_FALSE(slots().owed(slot));
    for (std::uint32_t r = 0; r < 5; ++r) {
      const auto what = std::format("slot {} row {}", slot, r);
      if (r < kept[slot]) {
        const std::int32_t token = verified_[slot][r];
        EXPECT_TRUE(Holds(cache[r], slot, v + r, token)) << what;
        EXPECT_TRUE(Holds(taps[r], slot, v + r, token)) << what;
        EXPECT_TRUE(Holds(ring[r], slot, v + r, token)) << what;
        continue;
      }
      EXPECT_EQ(cache[r].magic, 0) << what;
      EXPECT_EQ(taps[r].magic, 0) << what;
      if (r == 2 || r == 3) {
        EXPECT_TRUE(Holds(ring[r], slot, v + r, r == 2 ? Truth(v + 2) : kFill)) << what;
      } else {
        EXPECT_EQ(ring[r].magic, 0) << what;
      }
    }
  }
  // The commit made the fixed state the kept rows': the next verify's rows
  // continue it.
  accept_ = {kAll, kAll, kAll};
  const auto next = Unit(o, {0, 1, 2});
  for (std::uint32_t slot = 0; slot < kSlots; ++slot) {
    ASSERT_FALSE(next.outcomes[slot].refusal.has_value());
    const std::uint32_t at = v + kept[slot];
    EXPECT_EQ(verify_rows_[slot].front().position, static_cast<std::int32_t>(at));
    EXPECT_EQ(verify_rows_[slot].front().hash,
              Hash(std::span(history_[slot]).first(std::size_t{at} + 1)));
  }
  EXPECT_EQ(commits_, 3U);
}

TEST_F(TruncatingTest, ATruncatedCacheSavesNothingAndKeepsItsRejectedRowsUnread) {
  auto o = Make(Cascade());
  const std::array<std::uint32_t, 1> prompts = {40};
  Prefill(prompts, &o);
  accept_[0] = 1;
  const auto report = Unit(o, {0});
  ASSERT_FALSE(report.outcomes[0].refusal.has_value());
  EXPECT_EQ(report.outcomes[0].kept, 2U);
  // Only the fixed state saved: the cache truncates.
  EXPECT_EQ(slots().live(0, 0).saved().size(), 1U);
  const auto cache = Read(0, 0, 0, 40, 5);
  for (std::uint32_t r = 0; r < 5; ++r) {
    // The rejected rows' cells stay as the verify wrote them, past the
    // positions, never read.
    EXPECT_TRUE(Holds(cache[r], 0, 40 + r, verified_[0][r])) << r;
  }
  EXPECT_EQ(slots().positions(0), 42U);
  // Overwritten by the next unit's rows, and the hash continues.
  const auto next = Unit(o, {0});
  ASSERT_FALSE(next.outcomes[0].refusal.has_value());
  EXPECT_EQ(verify_rows_[0].front().hash, Hash(std::span(history_[0]).first(43)));
  const auto after = Read(0, 0, 0, 42, 5);
  for (std::uint32_t r = 0; r < 5; ++r) EXPECT_TRUE(Holds(after[r], 0, 42 + r, verified_[0][r]));
}

TEST_F(DecodingTest, ADiscardedVerifyIsUndoneWholeAndAcceptChecksItsRows) {
  auto o = Make(Cascade());
  const std::array<std::uint32_t, 2> prompts = {40, 40};
  Prefill(prompts, &o);
  // No verify: nothing to accept or discard.
  EXPECT_FALSE(slots().Accept(0, 1));
  EXPECT_FALSE(slots().DiscardVerify(0));
  refuse_judge_ = true;
  const auto report = Unit(o, {0, 1});
  for (std::uint32_t slot = 0; slot < 2; ++slot) {
    const auto& out = report.outcomes[slot];
    ASSERT_TRUE(out.refusal.has_value());
    EXPECT_TRUE(out.discarded);
    EXPECT_EQ(out.kept, 0U);
    EXPECT_EQ(slots().positions(slot), 40U);
    EXPECT_FALSE(slots().verify_pending(slot));
    EXPECT_FALSE(slots().owed(slot));  // run at once
    EXPECT_TRUE(slots().usable(slot));
    for (const auto& cell : Read(slot, 0, 0, 40, 5)) EXPECT_EQ(cell.magic, 0);
    for (const auto& cell : Read(slot, kMtp, 1, 40, 5)) EXPECT_EQ(cell.magic, 0);
  }
  // The same unit again, judged: the state was as before it.
  refuse_judge_ = false;
  const auto again = Unit(o, {0, 1});
  for (std::uint32_t slot = 0; slot < 2; ++slot) {
    ASSERT_FALSE(again.outcomes[slot].refusal.has_value());
    EXPECT_EQ(again.outcomes[slot].kept, 5U);
    EXPECT_EQ(verify_rows_[slot].front().hash, Hash(std::span(history_[slot]).first(41)));
  }
  // A verify of three rows run directly: an Accept of none or of more rows
  // than it ran is refused with nothing changed; of two, taken.
  EXPECT_TRUE(slots().Settle(0));
  const std::uint32_t at = slots().positions(0);
  const std::vector<std::int32_t> rows = {history_[0].back(), Truth(at + 1), Truth(at + 2)};
  const std::array<en::SegmentWork, 1> work = {en::SegmentWork{.tokens = rows, .bytes = {}}};
  auto verified = runner_->RunStep({.shape = Shape(0, pl::Phase::kVerify, {Seg(0, at, 3, 256)}),
                                    .work = work,
                                    .step_bytes = {},
                                    .advance = false,
                                    .ahead = {}});
  ASSERT_TRUE(verified.has_value()) << verified.error().what;
  EXPECT_EQ(slots().verify_rows(0), 3U);
  EXPECT_FALSE(slots().Accept(0, 0));
  EXPECT_FALSE(slots().Accept(0, 4));
  EXPECT_EQ(slots().verify_rows(0), 3U);
  EXPECT_EQ(slots().positions(0), at);
  verified_[0] = rows;
  ASSERT_TRUE(slots().Accept(0, 2));
  EXPECT_EQ(slots().positions(0), at + 2);
  EXPECT_FALSE(slots().verify_pending(0));
  EXPECT_TRUE(slots().owed(0));
  EXPECT_TRUE(slots().Settle(0));
  EXPECT_FALSE(slots().owed(0));
  EXPECT_GE(commits_, 1U);
}

// ------------------------------------------------------------- isolation

TEST_F(DecodingTest, AVerifyRefusedMidWaveLeavesItsPeersVerified) {
  auto o = Make(Cascade());
  const std::array<std::uint32_t, 3> prompts = {40, 41, 42};
  Prefill(prompts, &o);
  source_->options().refuse_build = [](const pl::StepShape& shape) {
    return shape.phase == pl::Phase::kVerify &&
           std::ranges::any_of(shape.segments, [](const pl::Segment& s) { return s.slot == 1; });
  };
  const auto report = Unit(o, {0, 1, 2});
  ASSERT_TRUE(report.outcomes[1].refusal.has_value());
  EXPECT_TRUE(report.outcomes[1].refusal->clean);
  EXPECT_EQ(slots().positions(1), 41U);
  EXPECT_FALSE(slots().verify_pending(1));
  EXPECT_TRUE(slots().usable(1));
  EXPECT_EQ(o.cursor(1, 0), 41U);
  for (const std::uint32_t slot : {0U, 2U}) {
    ASSERT_FALSE(report.outcomes[slot].refusal.has_value()) << report.outcomes[slot].refusal->what;
    EXPECT_EQ(report.outcomes[slot].kept, 5U);
  }
  source_->options().refuse_build = {};
  const auto next = Unit(o, {0, 1, 2});
  for (std::uint32_t slot = 0; slot < kSlots; ++slot) {
    ASSERT_FALSE(next.outcomes[slot].refusal.has_value()) << next.outcomes[slot].refusal->what;
    EXPECT_EQ(next.outcomes[slot].kept, 5U);
  }
  EXPECT_EQ(slots().positions(1), 46U);
}

TEST_F(DecodingTest, ARejectedVerifyPublishesItsRetiredPrefixWithoutItsOwnNumericEffects) {
  auto o = Make(Cascade());
  const std::array<std::uint32_t, 3> prompts = {40, 41, 42};
  Prefill(prompts, &o);
  ASSERT_FALSE(Unit(o, {0, 1, 2}).outcomes[0].refusal.has_value());
  // Each slot owes its Accept's restore and commit; a verify of the anchor
  // alone (one token left: no drafts) queues them, then its run is refused.
  source_->options().fail_run = kn::StepError::kRejected;
  source_->options().fail_phase = pl::Phase::kVerify;
  runner_->DropPlans();  // Failure injection must reach Run, not replay a prior recording.
  const auto executed = source_->executed();
  const auto commits = commits_;
  const auto failed = Unit(o, {0, 1, 2}, 1);
  EXPECT_EQ(source_->executed(), executed);  // No rejected verify executed.
  EXPECT_EQ(commits_ - commits, kSlots);     // Every prior Accept actually retired.
  for (std::uint32_t slot = 0; slot < kSlots; ++slot) {
    const auto& out = failed.outcomes[slot];
    ASSERT_TRUE(out.refusal.has_value());
    EXPECT_TRUE(out.refusal->clean);  // Complete prior work published; this verify wrote nothing.
    EXPECT_NE(out.refusal->what.find("the fake refuses to run"), std::string::npos);
    EXPECT_TRUE(out.drafts.empty());
    EXPECT_EQ(slots().positions(slot), prompts[slot] + 5);
    EXPECT_FALSE(slots().verify_pending(slot));
    // The rejected verify's snapshots now owe an undo, distinct from the
    // prior Accept's commits whose execution was counted above.
    EXPECT_TRUE(slots().owed(slot));
    EXPECT_TRUE(slots().usable(slot));  // The saves undo whole; no rejected numeric writes.
  }
  source_->options().fail_run.reset();
  source_->options().fail_phase.reset();
  const auto next = Unit(o, {0, 1, 2});
  for (std::uint32_t slot = 0; slot < kSlots; ++slot) {
    ASSERT_FALSE(next.outcomes[slot].refusal.has_value()) << next.outcomes[slot].refusal->what;
    const std::uint32_t at = prompts[slot] + 5;
    EXPECT_EQ(verify_rows_[slot].front().hash,
              Hash(std::span(history_[slot]).first(std::size_t{at} + 1)));
  }
}

TEST_F(DecodingTest, ADrafterRefusedForOneSlotLeavesItsUnitAndItsPeers) {
  auto o = Make(Cascade());
  const std::array<std::uint32_t, 3> prompts = {40, 41, 42};
  Prefill(prompts, &o);
  source_->options().refuse_build = [](const pl::StepShape& shape) {
    return shape.phase == pl::Phase::kDraft && shape.stage == kMtp &&
           std::ranges::any_of(shape.segments, [](const pl::Segment& s) { return s.slot == 1; });
  };
  const auto report = Unit(o, {0, 1, 2});
  const auto& lone = report.outcomes[1];
  ASSERT_FALSE(lone.refusal.has_value()) << lone.refusal->what;
  ASSERT_EQ(lone.drafts.size(), 2U);
  EXPECT_TRUE(lone.drafts[0].refusal.has_value());
  EXPECT_EQ(lone.drafts[0].proposed, 0U);
  // The block drafts from its anchor instead.
  EXPECT_EQ(lone.drafts[1].proposed, 2U);
  EXPECT_EQ(lone.proposed, (std::vector<std::int32_t>{Truth(42), Truth(43)}));
  EXPECT_EQ(lone.kept, 3U);
  for (const std::uint32_t slot : {0U, 2U}) {
    ASSERT_FALSE(report.outcomes[slot].refusal.has_value());
    EXPECT_FALSE(report.outcomes[slot].drafts[0].refusal.has_value());
    EXPECT_EQ(report.outcomes[slot].kept, 5U);
  }
}

// --------------------------------------------------------- adaptive depth

TEST_F(DecodingTest, AdaptiveDepthChangesBetweenUnitsFromCompleteStepsOnly) {
  en::DrafterSpec mtp = Mtp(3);
  mtp.adaptive = true;
  auto o = Make({.kind = en::DecodingMode::Kind::kSpeculative,
                 .drafters = {mtp},
                 .combine = en::DecodingMode::Drafters::kCascade,
                 .row_invariant = false});
  const std::array<std::uint32_t, 1> prompts = {40};
  Prefill(prompts, &o);
  std::vector<std::uint32_t> depths;
  // A unit cut by the reply's end teaches nothing.
  auto cut = Unit(o, {0}, 2);
  ASSERT_FALSE(cut.outcomes[0].refusal.has_value());
  depths.push_back(cut.outcomes[0].drafts.at(0).depth);
  for (int unit = 0; unit < 6; ++unit) {
    const auto report = Unit(o, {0});
    ASSERT_FALSE(report.outcomes[0].refusal.has_value()) << report.outcomes[0].refusal->what;
    depths.push_back(report.outcomes[0].drafts.at(0).depth);
  }
  // Four complete units at the deepest, then the shallower explored.
  EXPECT_EQ(depths, (std::vector<std::uint32_t>{1, 3, 3, 3, 3, 2, 2}));
  const auto saved = o.depth(0, 0).Save();
  EXPECT_EQ(saved[7], 2U);  // the shallower's units seen
  EXPECT_EQ(saved[9], 4U);  // the deepest's
}

// --------------------------------------------------------- borrowed state

TEST_F(DecodingTest, AnchoredDepthUsesActualVerifyBoundsRatherThanTheMeasuredDefault) {
  en::DecodingMode mode{
      .kind = en::DecodingMode::Kind::kSpeculative,
      .drafters = {{.stage = kAssistant, .max_depth = 4, .borrows_target = true, .anchored = true}},
      .row_invariant = false};
  auto o = Make(mode);
  const std::array<std::uint32_t, 1> prompts{40};
  Prefill(prompts, &o);
  const auto report = Unit(o, {0});
  ASSERT_EQ(report.outcomes.size(), 1U);
  ASSERT_FALSE(report.outcomes[0].refusal);
  ASSERT_EQ(report.outcomes[0].drafts.size(), 1U);
  EXPECT_EQ(report.outcomes[0].drafts[0].depth, 4U);
  mode.drafters[0].max_depth = 17;
  const auto bounded = en::DecodingOrchestrator::Make(*runner_, *composer_, mode, Policy());
  ASSERT_FALSE(bounded);
  EXPECT_NE(bounded.error().find("runner's 8"), std::string::npos);
  mode.drafters[0].max_depth = slots().layout(kAssistant).max_rows;
  const auto refused = en::DecodingOrchestrator::Make(*runner_, *composer_, mode, Policy());
  EXPECT_FALSE(refused);
}

TEST_F(DecodingTest, AnOrdinaryReaderRequiresItsExactAcceptedPrefixPermit) {
  auto o = Make({.kind = en::DecodingMode::Kind::kSpeculative,
                 .drafters = {{.stage = kAssistant, .max_depth = 2, .borrows_target = true}},
                 .combine = en::DecodingMode::Drafters::kCascade,
                 .row_invariant = false});
  const std::array<std::uint32_t, 2> prompts{40, 41};
  Prefill(prompts, &o);
  auto first = Unit(o, {0, 1});
  for (const auto& out : first.outcomes) ASSERT_FALSE(out.refusal);
  ASSERT_TRUE(slots().owed(0));
  const auto past = slots().positions(0);
  auto borrowed = slots().BorrowAfterOwed(0, kAssistant, past, 1);
  ASSERT_TRUE(borrowed) << ts::Failed(borrowed).value_or("");
  ASSERT_TRUE(borrowed->permit());
  const auto permit = *borrowed->permit();
  EXPECT_TRUE(slots().borrowed(0));
  EXPECT_FALSE(slots().borrowed(0, 0));
  EXPECT_TRUE(slots().CheckBorrowPermit(permit, 0, kAssistant, past, 1));
  EXPECT_FALSE(slots().CheckBorrowPermit(permit, 1, kAssistant, past, 1));
  EXPECT_FALSE(slots().CheckBorrowPermit(permit, 0, kAssistant, past + 1, 1));
  EXPECT_FALSE(slots().SetPositions(0, past + 1));
  EXPECT_FALSE(slots().ReserveStateThrough(0, past + 1));
  const std::array token{Truth(past)};
  const std::array work{en::SegmentWork{.tokens = token}};
  const pl::StepShape shape{
      .stage = kAssistant,
      .phase = pl::Phase::kDraft,
      .segments = {{.slot = 0, .rows = 1, .past = past, .read_cells = past + 1, .outputs = 1}},
      .read_align = 256};
  const auto executed = source_->executed();
  auto refused = runner_->RunStep({.shape = shape, .work = work, .advance = false, .ahead = {}});
  ASSERT_FALSE(refused);
  EXPECT_TRUE(refused.error().clean);
  EXPECT_NE(refused.error().what.find("ordered borrow"), std::string::npos);
  EXPECT_EQ(source_->executed(), executed);
  borrowed->Return();
  EXPECT_FALSE(slots().borrowed(0));
  auto fresh = slots().BorrowAfterOwed(0, kAssistant, past, 1);
  ASSERT_TRUE(fresh);
  EXPECT_FALSE(slots().CheckBorrowPermit(permit, 0, kAssistant, past, 1));
  fresh->Return();
}

TEST_F(DecodingTest, UnknownOrdinaryReaderRetainsItsOrderedGuardUntilRecovery) {
  auto o = Make({.kind = en::DecodingMode::Kind::kSpeculative,
                 .drafters = {{.stage = kAssistant, .max_depth = 2, .borrows_target = true}},
                 .combine = en::DecodingMode::Drafters::kCascade,
                 .row_invariant = false});
  const std::array<std::uint32_t, 1> prompts{40};
  Prefill(prompts, &o);
  ASSERT_FALSE(Unit(o, {0}).outcomes[0].refusal);
  ASSERT_TRUE(slots().owed(0));
  source_->options().fail_phase = pl::Phase::kDraft;
  source_->options().fail_run = kn::StepError::kUnknown;
  runner_->DropPlans();  // A replay does not call the host Run failure hook.
  const auto queued = source_->queued();
  const auto failed = Unit(o, {0});
  EXPECT_GT(source_->queued(), queued);  // The injected reader really submitted work.
  ASSERT_TRUE(failed.outcomes[0].refusal);
  ASSERT_FALSE(failed.outcomes[0].drafts.empty());
  ASSERT_TRUE(failed.outcomes[0].drafts[0].refusal);
  EXPECT_NE(failed.outcomes[0].drafts[0].refusal->what.find("the fake faulted"), std::string::npos);
  EXPECT_FALSE(failed.outcomes[0].refusal->clean);
  EXPECT_FALSE(slots().usable(0));
  EXPECT_TRUE(slots().borrowed(0));
  EXPECT_FALSE(slots().ReserveStateThrough(0, slots().positions(0) + 1));
  EXPECT_FALSE(slots().Spill(0));
}

TEST_F(DecodingTest, AnAssistantDraftsFromTheTargetsStateBorrowedFrozen) {
  auto o = Make({.kind = en::DecodingMode::Kind::kSpeculative,
                 .drafters = {{.stage = kAssistant,
                               .shape = en::DrafterSpec::Shape::kAutoregressive,
                               .max_depth = 2,
                               .borrows_target = true}},
                 .combine = en::DecodingMode::Drafters::kCascade,
                 .row_invariant = false});
  const std::array<std::uint32_t, 2> prompts = {40, 41};
  Prefill(prompts, &o);
  std::uint32_t checked = 0;
  in_propose_ = [&](std::uint32_t slot, std::size_t /*drafter*/) {
    // Frozen: the target's state cannot grow, the assistant's may.
    EXPECT_TRUE(slots().borrowed(slot, 0));
    EXPECT_FALSE(slots().borrowed(slot, kAssistant));
    EXPECT_FALSE(slots().ReserveStateThrough(slot, slots().positions(slot) + 1));
    EXPECT_TRUE(slots().ReserveStateThrough(slot, slots().positions(slot) + 1, 0, kAssistant));
    ++checked;
  };
  for (int unit = 0; unit < 2; ++unit) {
    draft_rows_.clear();
    const std::array<std::uint32_t, 2> at = {slots().positions(0), slots().positions(1)};
    if (unit != 0) {
      EXPECT_TRUE(slots().owed(0));
      EXPECT_TRUE(slots().owed(1));
    }
    (void)node_.TakeTimes(0);
    const auto report = Unit(o, {0, 1});
    EXPECT_EQ(node_.TakeTimes(0).steps, report.steps);  // Both prefixes fold into useful work.
    for (std::uint32_t slot = 0; slot < 2; ++slot) {
      ASSERT_FALSE(report.outcomes[slot].refusal.has_value())
          << report.outcomes[slot].refusal->what;
      EXPECT_EQ(report.outcomes[slot].kept, 3U);
      EXPECT_FALSE(slots().borrowed(slot));
      // Its first pass read every committed cell of the target's cache.
      const auto& rows = draft_rows_[{slot, 0}];
      ASSERT_FALSE(rows.empty());
      EXPECT_EQ(rows.front().borrowed, static_cast<std::int32_t>(at[slot]));
    }
  }
  EXPECT_EQ(checked, 8U);
  std::size_t alignment_checks = 0;
  for (const auto& built : source_->built()) {
    if (built.shape.stage != kAssistant) continue;
    EXPECT_TRUE(built.bound_alignments_hold);
    EXPECT_TRUE(built.invalid_alignment_refs_unknown);
    ASSERT_FALSE(built.borrowed_alignments.empty());
    const auto& borrowed = slots().layout(kAssistant).borrowed;
    ASSERT_EQ(built.borrowed_alignments.size(), built.shape.segments.size() * borrowed.size());
    std::size_t at = 0;
    for (const auto& owner : built.shape.segments) {
      (void)owner;
      for (const auto& ref : borrowed) {
        const auto* actual = md::BorrowedItem(slots().layout(0), ref);
        ASSERT_NE(actual, nullptr);
        EXPECT_EQ(built.borrowed_alignments[at++],
                  std::gcd(built.activation_alignment, actual->offset));
        ++alignment_checks;
      }
    }
  }
  EXPECT_GT(alignment_checks, 0U);
  // Its step without the borrow is refused.
  const std::vector<std::int32_t> token = {1};
  const std::array<en::SegmentWork, 1> work = {en::SegmentWork{.tokens = token, .bytes = {}}};
  auto refused = runner_->RunStep(
      {.shape = Shape(kAssistant, pl::Phase::kDraft, {Seg(0, slots().positions(0), 1, 256)}),
       .work = work,
       .step_bytes = {},
       .advance = false,
       .ahead = {}});
  ASSERT_FALSE(refused.has_value());
  EXPECT_TRUE(refused.error().clean);
  EXPECT_EQ(refused.error().slot, 0U);
}

// ------------------------------------------------- the stages' rules

TEST_F(DecodingTest, StepsKeepTheirStagesRulesAndRowInvariantVerifiesReadAlone) {
  const std::array<std::uint32_t, 3> prompts = {40, 40, 40};
  Prefill(prompts);
  const auto run = [&](const pl::StepShape& shape, bool advance) {
    std::vector<std::vector<std::int32_t>> tokens;
    std::vector<en::SegmentWork> work;
    tokens.reserve(shape.segments.size());
    for (const auto& s : shape.segments) tokens.emplace_back(s.rows, 9);
    for (const auto& t : tokens) work.push_back({.tokens = t, .bytes = {}});
    return runner_->RunStep(
        {.shape = shape, .work = work, .step_bytes = {}, .advance = advance, .ahead = {}});
  };
  // A verify never advances and starts at the committed positions.
  EXPECT_FALSE(run(Shape(0, pl::Phase::kVerify, {Seg(0, 40, 2, 256)}), true).has_value());
  auto wrong = run(Shape(0, pl::Phase::kVerify, {Seg(0, 39, 2, 256)}), false);
  ASSERT_FALSE(wrong.has_value());
  EXPECT_EQ(wrong.error().slot, 0U);
  EXPECT_FALSE(run(Shape(0, pl::Phase::kVerify, {Seg(0, 40, kVerifyRows + 1, 256)}), false));
  // A drafter never commits, and drafts within its chunk bound.
  EXPECT_FALSE(run(Shape(kMtp, pl::Phase::kDraft, {Seg(0, 40, 1, 512)}), true).has_value());
  auto far = run(Shape(kMtp, pl::Phase::kDraft, {Seg(0, 40 + kChunk, 1, 512)}), false);
  ASSERT_FALSE(far.has_value());
  EXPECT_EQ(far.error().slot, 0U);
  // Behind the committed positions (a catch-up), at its stage's 512.
  auto caught = run(Shape(kMtp, pl::Phase::kDraft, {Seg(0, 30, 11, 512)}), false);
  ASSERT_TRUE(caught.has_value()) << caught.error().what;
  EXPECT_EQ(slots().positions(0), 40U);
  EXPECT_EQ(slots().read_align(kMtp), 512U);
  EXPECT_EQ(slots().read_align(0), 256U);

  // The composer: a stage's own alignment alone, the joined one together,
  // and a row-invariant verify at each owner's lone reads, never its
  // peers'.
  md::Batching caps = Caps();
  caps.joined_read_align = 2048;
  caps.common_read = true;
  auto composer = en::WaveComposer::For(caps, slots());
  ASSERT_TRUE(composer.has_value());
  EXPECT_EQ(composer->LoneRead(kMtp, 30, 11), 512U);
  const auto draft = composer->Compose(std::vector<en::PendingWork>{
      {.slot = 0, .stage = kMtp, .phase = pl::Phase::kDraft, .past = 30, .rows = 11}});
  ASSERT_EQ(draft.steps.size(), 1U);
  EXPECT_EQ(draft.steps[0].shape.read_align, 512U);
  std::vector<en::PendingWork> verify;
  for (std::uint32_t slot = 0; slot < kSlots; ++slot) {
    verify.push_back({.slot = slot,
                      .stage = 0,
                      .phase = pl::Phase::kVerify,
                      .past = slot == 2 ? 300U : 40U,
                      .rows = 3,
                      .outputs = 3,
                      .row_invariant = true});
  }
  const auto exact = composer->Compose(verify);
  ASSERT_EQ(exact.steps.size(), 1U);
  EXPECT_EQ(exact.steps[0].shape.read_align, 256U);
  EXPECT_EQ(exact.steps[0].shape.segments[0].read_cells, 256U);
  EXPECT_EQ(exact.steps[0].shape.segments[2].read_cells, 512U);
  for (auto& v : verify) v.row_invariant = false;
  const auto batched = composer->Compose(verify);
  EXPECT_EQ(batched.steps[0].shape.read_align, 2048U);
  EXPECT_EQ(batched.steps[0].shape.segments[0].read_cells, kContext);  // 2,048, at most the context
  // The runner refuses a row-invariant step whose reads follow its peers.
  auto joined = run(
      Shape(0, pl::Phase::kVerify, {Seg(0, 40, 2, 2048), Seg(1, 40, 2, 2048)}, 2048, true), false);
  ASSERT_FALSE(joined.has_value());
  EXPECT_TRUE(joined.error().clean);
  EXPECT_FALSE(slots().verify_pending(0));
  // A composer whose stage reads otherwise than the state's runs nothing.
  auto other = en::WaveComposer::Make(
      Caps(), std::vector<en::StageReads>{
                  {kContext, 256}, {kContext, 256}, {kContext, 256}, {kContext, 256}});
  ASSERT_TRUE(other.has_value());
  const std::vector<std::int32_t> one = {9};
  const std::array<en::SegmentWork, 1> work = {en::SegmentWork{.tokens = one, .bytes = {}}};
  const auto refused = en::RunWave(
      *runner_, *other,
      std::vector<en::PendingWork>{{.slot = 0, .stage = 0, .past = 40, .rows = 1}}, work);
  EXPECT_TRUE(refused.outcomes[0].refusal.has_value());
  EXPECT_FALSE(en::DecodingOrchestrator::Make(*runner_, *other, Cascade(), Policy()).has_value());
}

// ----------------------------------------------------- the other modes

TEST_F(DecodingTest, SinglePassModesAndRefusedModes) {
  using K = en::DecodingMode::Kind;
  const auto refused = [&](en::DecodingMode mode) {
    return !en::DecodingOrchestrator::Make(*runner_, *composer_, std::move(mode), Policy())
                .has_value();
  };
  EXPECT_TRUE(refused(en::DecodingMode{.kind = K::kBlockDiffusion}));  // declared, not built
  EXPECT_TRUE(refused(en::DecodingMode{.kind = K::kPlain, .drafters = {Mtp(1)}}));
  EXPECT_TRUE(refused(en::DecodingMode{.kind = K::kSpeculative}));
  EXPECT_TRUE(refused(en::DecodingMode{.kind = K::kSpeculative, .drafters = {{.stage = 0}}}));
  EXPECT_TRUE(refused(en::DecodingMode{.kind = K::kSpeculative, .drafters = {Mtp(1), Mtp(1)}}));
  EXPECT_TRUE(refused(
      en::DecodingMode{.kind = K::kSpeculative, .drafters = {Mtp(4), Block(4)}}));  // 9 rows
  // The assistant reads the target's state: only borrowing it.
  EXPECT_TRUE(
      refused(en::DecodingMode{.kind = K::kSpeculative, .drafters = {{.stage = kAssistant}}}));
  EXPECT_FALSE(en::DecodingOrchestrator::Make(*runner_, *composer_, Cascade(), {}).has_value());

  const std::array<std::uint32_t, 3> prompts = {40, 41, 42};
  Prefill(prompts);
  auto plain = Make({.kind = K::kPlain});
  std::vector<en::DecodeWork> work;
  for (std::uint32_t slot = 0; slot < kSlots; ++slot) {
    work.push_back({.slot = slot, .history = history_[slot], .greedy = true});
  }
  const auto decoded = plain.Step(work);
  EXPECT_EQ(decoded.steps, 1U);
  for (std::uint32_t slot = 0; slot < kSlots; ++slot) {
    const auto& out = decoded.outcomes[slot];
    ASSERT_FALSE(out.refusal.has_value()) << out.refusal->what;
    EXPECT_EQ(out.kept, 1U);
    EXPECT_EQ(slots().positions(slot), prompts[slot] + 1);
    // The device's choice: the segment's own output (the lowering's form).
    ASSERT_EQ(out.outputs.size(), sizeof(std::int32_t));
    std::int32_t chosen = 0;
    std::memcpy(&chosen, out.outputs.data(), sizeof(chosen));
    EXPECT_EQ(chosen, Truth(prompts[slot]) + 1000);
  }
  // Scored: every given token's row, from the positions.
  auto scored = Make({.kind = K::kScored});
  const std::vector<std::int32_t> score = {1, 2, 3, 4, 5};
  const std::array<en::DecodeWork, 1> one = {en::DecodeWork{.slot = 0, .score = score}};
  const auto rows = scored.Step(one);
  ASSERT_FALSE(rows.outcomes[0].refusal.has_value());
  EXPECT_EQ(rows.outcomes[0].kept, 5U);
  EXPECT_EQ(Rows(rows.outcomes[0].outputs).size(), 5U);
  EXPECT_EQ(slots().positions(0), prompts[0] + 6);
  // Work that does not continue its slot's positions is refused alone.
  auto sampled = Make({.kind = K::kSampled});
  const std::array<en::DecodeWork, 2> mixed = {
      en::DecodeWork{.slot = 1, .history = std::span(history_[1]).first(5)},
      en::DecodeWork{.slot = 2, .history = history_[2]}};
  history_[2].push_back(0);  // its anchor: the sampler's choice
  const auto sampled_out = sampled.Step(
      std::array<en::DecodeWork, 2>{mixed[0], en::DecodeWork{.slot = 2, .history = history_[2]}});
  EXPECT_TRUE(sampled_out.outcomes[0].refusal.has_value());
  ASSERT_FALSE(sampled_out.outcomes[1].refusal.has_value());
  EXPECT_EQ(slots().positions(2), prompts[2] + 2);
}

TEST_F(DecodingTest, PrefillOutcomesKeepTheirPlaceInAJoinedStep) {
  md::Batching caps = Caps();
  caps.join_prefill = true;
  auto composer = en::WaveComposer::For(caps, slots());
  ASSERT_TRUE(composer.has_value());
  en::PrefillOrchestrator prefill(*runner_, *composer,
                                  {.chunk_rows = kChunk,
                                   .lookahead = 0,
                                   .capture_ahead = false,
                                   .prepare_ahead = false,
                                   .state_only = true});
  std::array<std::vector<std::int32_t>, kSlots> tokens;
  std::vector<en::PromptWork> prompts;
  for (std::uint32_t slot = 0; slot < kSlots; ++slot) {
    for (std::uint32_t q = 0; q < 20 + slot; ++q) tokens[slot].push_back(Truth(q));
    prompts.push_back({.slot = slot, .stage = 0, .tokens = tokens[slot], .all_outputs = slot == 1});
  }
  const auto report = prefill.Step(prompts);
  EXPECT_EQ(report.steps, 1U);
  std::uint32_t rows = 0, outputs = 0;
  for (std::uint32_t slot = 0; slot < kSlots; ++slot) {
    const auto& out = report.outcomes()[slot];
    ASSERT_FALSE(out.refusal.has_value()) << out.refusal->what;
    EXPECT_EQ(out.segment, slot);
    EXPECT_EQ(out.first_row, rows);
    EXPECT_EQ(out.first_output, outputs);
    rows += 20 + slot;
    outputs += slot == 1 ? 21 : 1;
  }
}

// -------------------------------------------------------------- challenges

TEST_F(TruncatingOnlyTest, ATargetSavingNothingAwaitsItsAcceptAndUndoesAFailedVerify) {
  auto o = Make(Cascade());
  const std::array<std::uint32_t, 2> prompts = {40, 41};
  Prefill(prompts, &o);
  accept_ = {kAll, 1, kAll};
  const auto report = Unit(o, {0, 1});
  const std::array<std::uint32_t, 2> kept = {5, 2};
  for (std::uint32_t slot = 0; slot < 2; ++slot) {
    const auto& out = report.outcomes[slot];
    ASSERT_FALSE(out.refusal.has_value()) << out.refusal->what;
    EXPECT_EQ(out.kept, kept[slot]);
    EXPECT_EQ(slots().positions(slot), prompts[slot] + kept[slot]);
    EXPECT_TRUE(slots().live(slot, 0).saved().empty());  // every tensor truncates
  }
  // A verify failing after its job queued work: undone whole (the target
  // by nothing, the drafters' taps by their saves), every slot usable.
  accept_ = {kAll, kAll, kAll};
  source_->options().fail_run = kn::StepError::kRejected;
  source_->options().fail_phase = pl::Phase::kVerify;
  const auto failed = Unit(o, {0, 1});
  for (std::uint32_t slot = 0; slot < 2; ++slot) {
    ASSERT_TRUE(failed.outcomes[slot].refusal.has_value());
    EXPECT_EQ(slots().positions(slot), prompts[slot] + kept[slot]);
    EXPECT_FALSE(slots().verify_pending(slot));
    EXPECT_TRUE(slots().usable(slot));
  }
  source_->options().fail_run.reset();
  source_->options().fail_phase.reset();
  const auto next = Unit(o, {0, 1});
  for (std::uint32_t slot = 0; slot < 2; ++slot) {
    ASSERT_FALSE(next.outcomes[slot].refusal.has_value()) << next.outcomes[slot].refusal->what;
    EXPECT_EQ(next.outcomes[slot].kept, 5U);
    EXPECT_EQ(slots().positions(slot), prompts[slot] + kept[slot] + 5);
  }
}

TEST_F(DecodingTest, ABuildBindsOnlyTheStateItsStageMayWrite) {
  const std::array<std::uint32_t, 1> prompts = {40};
  Prefill(prompts);
  std::optional<ir::StateRef> stray;
  source_->options().extra_state = [&](const pl::StepShape&) {
    return stray ? std::vector<ir::StateRef>{*stray} : std::vector<ir::StateRef>{};
  };
  const std::vector<std::int32_t> token = {9};
  const std::array<en::SegmentWork, 1> work = {en::SegmentWork{.tokens = token, .bytes = {}}};
  const auto run = [&](std::uint16_t stage, ir::StateRef ref) {
    stray = ref;
    const std::uint32_t align = slots().read_align(stage);
    const std::uint32_t past = slots().positions(0);
    return runner_->RunStep(
        {.shape = Shape(stage, stage == 0 ? pl::Phase::kDecode : pl::Phase::kDraft,
                        {Seg(0, past, 1, align)}, align),
         .work = work,
         .step_bytes = {},
         .advance = stage == 0,
         .ahead = {}});
  };
  const auto refused = [&](std::uint16_t stage, ir::StateRef ref) {
    auto ran = run(stage, ref);
    ASSERT_FALSE(ran.has_value()) << stage << " binds " << ref.stage << "/" << ref.tensor;
    EXPECT_TRUE(ran.error().clean);
    EXPECT_EQ(slots().positions(0), 40U);
    EXPECT_TRUE(slots().usable(0));
  };
  // The target: never a drafter's own state, nor its view of the target's.
  refused(0, {.stage = kMtp, .tensor = 0, .layer = 0, .segment = 0});
  refused(0, {.stage = kAssistant, .tensor = 0, .layer = 0, .segment = 0});
  refused(0, {.stage = kAssistant, .tensor = 1, .layer = 0, .segment = 0});
  // A drafter: never the target's state, nor another drafter's.
  refused(kMtp, {.stage = 0, .tensor = 0, .layer = 0, .segment = 0});
  refused(kMtp, {.stage = kBlock, .tensor = 0, .layer = 0, .segment = 0});
  // The drafters' taps the target writes; a drafter's own stage.
  auto tap = run(0, {.stage = kMtp, .tensor = 1, .layer = 0, .segment = 0});
  ASSERT_TRUE(tap.has_value()) << tap.error().what;
  auto own = run(kMtp, {.stage = kMtp, .tensor = 1, .layer = 0, .segment = 0});
  ASSERT_TRUE(own.has_value()) << own.error().what;
}

TEST_F(DecodingTest, ACascadesSecondDrafterRefusedAfterItsFirstRanVerifiesWhatItHas) {
  auto o = Make(Cascade());
  const std::array<std::uint32_t, 3> prompts = {40, 41, 42};
  Prefill(prompts, &o);
  source_->options().refuse_build = [](const pl::StepShape& shape) {
    return shape.stage == kBlock &&
           std::ranges::any_of(shape.segments, [](const pl::Segment& s) { return s.slot == 1; });
  };
  const auto report = Unit(o, {0, 1, 2});
  const auto& lone = report.outcomes[1];
  ASSERT_FALSE(lone.refusal.has_value()) << lone.refusal->what;
  ASSERT_EQ(lone.drafts.size(), 2U);
  EXPECT_FALSE(lone.drafts[0].refusal.has_value());
  EXPECT_TRUE(lone.drafts[1].refusal.has_value());
  EXPECT_EQ(lone.proposed, (std::vector<std::int32_t>{Truth(42), Truth(43)}));
  EXPECT_EQ(lone.kept, 3U);
  EXPECT_EQ(o.cursor(1, 0), 41U);
  for (const std::uint32_t slot : {0U, 2U}) {
    ASSERT_FALSE(report.outcomes[slot].refusal.has_value());
    EXPECT_EQ(report.outcomes[slot].kept, 5U);
  }
  // The block's ring holds what its peers' blocks wrote and nothing of slot
  // 1's; the next unit drafts from the kept rows.
  source_->options().refuse_build = {};
  draft_rows_.clear();
  const auto next = Unit(o, {0, 1, 2});
  for (std::uint32_t slot = 0; slot < kSlots; ++slot) {
    ASSERT_FALSE(next.outcomes[slot].refusal.has_value()) << next.outcomes[slot].refusal->what;
    EXPECT_EQ(next.outcomes[slot].kept, 5U);
    EXPECT_EQ(verify_rows_[slot].front().hash,
              Hash(std::span(history_[slot]).first(std::size_t{slots().positions(slot)} - 4)));
  }
  const auto& mtp = draft_rows_[{1, 0}];
  ASSERT_FALSE(mtp.empty());
  EXPECT_EQ(mtp.front().position, 44);
  EXPECT_EQ(mtp.front().tap, Truth(43));
}

TEST_F(DecodingTest, AnAlternativeLeftBehindPastItsChunkCatchesUpItsLastChunk) {
  std::size_t pick = 1;
  auto policy = Policy();
  policy.choose = [&](std::uint32_t) -> std::optional<std::size_t> { return pick; };
  auto o = Make({.kind = en::DecodingMode::Kind::kSpeculative,
                 .drafters = {Mtp(2), Block(3)},
                 .combine = en::DecodingMode::Drafters::kAlternatives,
                 .row_invariant = false},
                std::move(policy));
  const std::array<std::uint32_t, 1> prompts = {30};
  Prefill(prompts, &o);
  // The block for 20 units: 80 positions, past the MTP's chunk of 64.
  for (int unit = 0; unit < 20; ++unit) {
    const auto report = Unit(o, {0});
    ASSERT_FALSE(report.outcomes[0].refusal.has_value()) << report.outcomes[0].refusal->what;
    ASSERT_EQ(report.outcomes[0].kept, 4U);
  }
  ASSERT_EQ(slots().positions(0), 110U);
  EXPECT_EQ(o.cursor(0, 0), 30U);
  // Then the MTP: it catches up its last chunk's rows and drafts.
  pick = 0;
  draft_rows_.clear();
  const std::size_t from = source_->built().size();
  const auto report = Unit(o, {0});
  const auto& out = report.outcomes[0];
  ASSERT_FALSE(out.refusal.has_value()) << out.refusal->what;
  ASSERT_EQ(out.drafts.size(), 1U);
  EXPECT_FALSE(out.drafts[0].refusal.has_value()) << out.drafts[0].refusal->what;
  EXPECT_EQ(out.drafts[0].proposed, 2U);
  EXPECT_EQ(out.kept, 3U);
  EXPECT_EQ(o.cursor(0, 0), 110U);
  const auto& mtp = draft_rows_[{0, 0}];
  ASSERT_FALSE(mtp.empty());
  EXPECT_EQ(mtp.front().position, 110);
  EXPECT_EQ(mtp.front().tap, Truth(109));
  // One pass of the chunk's rows, built now.
  std::vector<std::uint32_t> passes;
  for (std::size_t i = from; i < source_->built().size(); ++i) {
    const auto& b = source_->built()[i];
    if (!b.measuring && b.shape.phase == pl::Phase::kDraft && b.shape.stage == kMtp) {
      passes.push_back(b.shape.segments.front().rows);
    }
  }
  EXPECT_EQ(passes, (std::vector<std::uint32_t>{kChunk, 1}));
}

// F15a: the prompt's rows of a catch-up drafter's own cache are written as
// the prompt commits (its attention reads the prompt as the target saw it),
// not left stale while its first pass starts at the committed positions.
TEST_F(DecodingTest, APromptCatchUpWritesTheDraftersPromptRowsBeforeItsFirstPass) {
  auto o = Make({.kind = en::DecodingMode::Kind::kSpeculative,
                 .drafters = {Mtp(2)},
                 .combine = en::DecodingMode::Drafters::kCascade,
                 .row_invariant = false});
  const std::array<std::uint32_t, 2> prompts = {30, 40};
  Prefill(prompts);
  // Slot 1 as a reset leaves it: its state claimed through its positions.
  o.Reset(1);
  const std::size_t from = source_->built().size();
  const std::array<en::PromptCatchUp, 1> work = {
      {{.slot = 0, .first = 0, .history = history_[0], .restart = true}}};
  EXPECT_EQ(o.CatchUpPrompt(work), 1U);
  EXPECT_EQ(o.cursor(0, 0), 30U);
  EXPECT_EQ(o.cursor(1, 0), 40U);
  // One headless pass of the prompt's rows.
  std::vector<std::uint32_t> passes;
  for (std::size_t i = from; i < source_->built().size(); ++i) {
    const auto& b = source_->built()[i];
    if (!b.measuring && b.shape.phase == pl::Phase::kDraft && b.shape.stage == kMtp) {
      passes.push_back(b.shape.segments.front().rows);
      EXPECT_FALSE(b.shape.segments.front().want_head);
    }
  }
  EXPECT_EQ(passes, (std::vector<std::uint32_t>{30}));
  const auto caught = Read(0, kMtp, 0, 0, 30);
  for (std::uint32_t q = 0; q < 30; ++q)
    EXPECT_TRUE(Holds(caught[q], 0, q, Truth(q))) << "row " << q;
  // The reset slot's prompt rows were never written: what the drafter's
  // attention read before this fix.
  for (const auto& cell : Read(1, kMtp, 0, 0, 40)) EXPECT_EQ(cell.magic, 0);
  // Each first pass then starts at the committed positions: one row.
  const auto report = Unit(o, {0});
  ASSERT_FALSE(report.outcomes[0].refusal.has_value()) << report.outcomes[0].refusal->what;
  EXPECT_EQ(report.outcomes[0].kept, 3U);
  const auto& mtp = draft_rows_[{0, 0}];
  ASSERT_FALSE(mtp.empty());
  EXPECT_EQ(mtp.front().position, 30);
  // A chunk that continues a prompt catches up from where the last stopped
  // (the verify's anchor here), never before it.
  history_[0].resize(slots().positions(0));
  const std::uint32_t cursor = o.cursor(0, 0);
  EXPECT_EQ(cursor, 30U);
  const std::array<en::PromptCatchUp, 1> next = {
      {{.slot = 0, .first = slots().positions(0), .history = history_[0], .restart = false}}};
  EXPECT_EQ(o.CatchUpPrompt(next), 1U);
  EXPECT_EQ(o.cursor(0, 0), slots().positions(0));
  const auto rewritten = Read(0, kMtp, 0, cursor, slots().positions(0) - cursor);
  for (std::uint32_t r = 0; r < rewritten.size(); ++r)
    EXPECT_TRUE(Holds(rewritten[r], 0, cursor + r, history_[0][cursor + r])) << "row " << r;
}

// F15a: the prefill lookahead predicts the catch-up drafter's rows of this
// chunk and of the next (all but each chunk's first) and builds them beside
// the target's chunk, instead of planning them on use between chunks.
TEST_F(DecodingTest, PrefillPredictsTheDraftersCatchUpOfThisChunkAndTheNext) {
  en::PrefillOrchestrator prefill(*runner_, *composer_, {.chunk_rows = kChunk, .lookahead = 1});
  const std::array<en::PrefillUnit, 1> units{{{.current = {.slot = 0,
                                                           .stage = 0,
                                                           .phase = pl::Phase::kPrefill,
                                                           .past = 0,
                                                           .rows = 40,
                                                           .outputs = 0,
                                                           .want_head = false},
                                               .work = {},
                                               .next = {.rows = 30}}}};
  const en::PrefillContinuation continuation{.catch_up_stage = kMtp};
  auto ahead = prefill.Predict(units, {}, continuation, std::array<std::size_t, 1>{0});
  ASSERT_TRUE(ahead) << llmp::test_support::Failed(ahead).value_or("");
  std::vector<std::pair<std::uint32_t, std::uint32_t>> drafted;
  std::size_t last = 0;
  for (std::size_t i = 0; i < ahead->shapes().size(); ++i) {
    const auto& shape = *ahead->shapes()[i];
    if (shape.stage != kMtp) continue;
    EXPECT_EQ(shape.phase, pl::Phase::kDraft);
    ASSERT_EQ(shape.segments.size(), 1U);
    EXPECT_FALSE(shape.segments[0].want_head);
    drafted.emplace_back(shape.segments[0].past, shape.segments[0].rows);
    last = i + 1;
  }
  EXPECT_EQ(drafted, (std::vector<std::pair<std::uint32_t, std::uint32_t>>{{1, 39}, {41, 29}}));
  // Built beside the chunk even past the lookahead's one future.
  EXPECT_GE(ahead->build, last);
  // Without a catch-up drafter, nothing of its stage is predicted.
  auto plain = prefill.Predict(units, {}, {}, std::array<std::size_t, 1>{0});
  ASSERT_TRUE(plain);
  for (const auto& shape : plain->shapes()) EXPECT_EQ(shape->stage, 0U);
}

TEST_F(DecodingTest, ASegmentWiderThanItsStagesPerRowStateIsRefused) {
  const std::array<std::uint32_t, 1> prompts = {kChunk};
  Prefill(prompts);
  auto borrow = slots().Borrow(0, 0);
  ASSERT_TRUE(borrow.has_value()) << borrow.error();
  const auto run = [&](std::uint32_t rows) {
    const std::vector<std::int32_t> tokens(rows, 9);
    const std::array<en::SegmentWork, 1> work = {en::SegmentWork{.tokens = tokens, .bytes = {}}};
    return runner_->RunStep({.shape = Shape(kAssistant, pl::Phase::kDraft, {Seg(0, 20, rows, 256)}),
                             .work = work,
                             .step_bytes = {},
                             .advance = false,
                             .ahead = {}});
  };
  // The assistant's per-row cells hold one chunk of 64 rows.
  auto wide = run(kChunk + 1);
  ASSERT_FALSE(wide.has_value());
  EXPECT_TRUE(wide.error().clean);
  auto fits = run(kChunk);
  ASSERT_TRUE(fits.has_value()) << fits.error().what;
}

TEST_F(DecodingTest, AnAdaptiveDrafterAtTheRunnersVerifyRows) {
  en::DrafterSpec mtp = Mtp(kVerifyRows - 1);
  mtp.adaptive = true;
  auto o = Make({.kind = en::DecodingMode::Kind::kSpeculative,
                 .drafters = {mtp},
                 .combine = en::DecodingMode::Drafters::kCascade,
                 .row_invariant = false});
  EXPECT_EQ(o.mode().verify_rows(), kVerifyRows);
  const std::array<std::uint32_t, 1> prompts = {40};
  Prefill(prompts, &o);
  std::uint32_t at = 40;
  for (int unit = 0; unit < 6; ++unit) {
    const auto report = Unit(o, {0});
    const auto& out = report.outcomes[0];
    ASSERT_FALSE(out.refusal.has_value()) << out.refusal->what;
    ASSERT_EQ(out.drafts.size(), 1U);
    EXPECT_LE(out.drafts[0].depth, kVerifyRows - 1);
    EXPECT_LE(verified_[0].size(), kVerifyRows);
    EXPECT_EQ(out.kept, out.drafts[0].depth + 1);
    at += out.kept;
    EXPECT_EQ(slots().positions(0), at);
    EXPECT_EQ(verify_rows_[0].front().hash, Hash(std::span(history_[0]).first(at - out.kept + 1)));
  }
}

TEST_F(DecodingTest, AFailedDraftQuarantinesItsOwnStageNeverTheTargets) {
  auto o = Make(Cascade());
  const std::array<std::uint32_t, 3> prompts = {40, 41, 42};
  Prefill(prompts, &o);
  ASSERT_FALSE(Unit(o, {0, 1, 2}).outcomes[0].refusal.has_value());
  source_->options().fail_run = kn::StepError::kUnknown;
  source_->options().fail_phase = pl::Phase::kDraft;
  runner_->DropPlans();  // Reach Run and its post-submission failure hook.
  const auto executed = source_->executed();
  const auto commits = commits_;
  const auto failed = Unit(o, {0});
  EXPECT_GT(source_->executed(), executed);  // Actual draft writes, not only input staging.
  EXPECT_EQ(commits_ - commits, 1U);         // Prior target commitment positively retired.
  ASSERT_TRUE(failed.outcomes[0].refusal.has_value());
  ASSERT_FALSE(failed.outcomes[0].drafts.empty());
  EXPECT_TRUE(failed.outcomes[0].drafts[0].refusal.has_value());
  EXPECT_NE(failed.outcomes[0].drafts[0].refusal->what.find("the fake faulted"), std::string::npos);
  EXPECT_EQ(slots().positions(0), 45U);
  EXPECT_FALSE(slots().verify_pending(0));
  // The draft's job ran what the Accept owed whole, then its run failed
  // after queueing: only the MTP's stage may hold its writes.
  EXPECT_FALSE(slots().owed(0));
  EXPECT_FALSE(slots().live(0, 0).quarantined());
  EXPECT_TRUE(slots().live(0, kMtp).quarantined());
  EXPECT_FALSE(slots().live(0, kBlock).quarantined());
  EXPECT_FALSE(slots().live(0, kAssistant).quarantined());
  EXPECT_FALSE(slots().usable(0));
  EXPECT_TRUE(slots().usable(1));
  source_->options().fail_run.reset();
  source_->options().fail_phase.reset();
  // A recovery discards that slot alone; its peers continue.
  ASSERT_TRUE(node_.EndRequest(0));
  auto recovered = slots().RecoverInPlace({});
  ASSERT_TRUE(recovered.has_value()) << recovered.error();
  EXPECT_EQ(*recovered, 0b001U);
  ASSERT_TRUE(slots().SelectSlots(std::vector<std::uint32_t>{0, 1, 2}));
  ASSERT_TRUE(node_.BeginRequest(0, slots().execution(), "test request"));
  const std::array<std::uint32_t, 1> again = {40};
  Prefill(again, &o);
  const auto next = Unit(o, {0, 1, 2});
  for (std::uint32_t slot = 0; slot < kSlots; ++slot) {
    ASSERT_FALSE(next.outcomes[slot].refusal.has_value()) << next.outcomes[slot].refusal->what;
    EXPECT_EQ(next.outcomes[slot].kept, 5U);
    const std::uint32_t before = slots().positions(slot) - 5;
    EXPECT_EQ(verify_rows_[slot].front().hash,
              Hash(std::span(history_[slot]).first(std::size_t{before} + 1)));
  }
}

TEST_F(DecodingTest, UnknownInputTransportNeverClaimsTheProgramsNarrowWriteScope) {
  auto decoder = Make(Cascade());
  const std::array<std::uint32_t, 3> prompts{40, 41, 42};
  Prefill(prompts, &decoder);
  ASSERT_FALSE(Unit(decoder, {0, 1, 2}).outcomes[0].refusal);
  runner_->DropPlans();
  bool armed = false;
  source_->options().on_build = [&](const pl::StepShape& shape, bool measuring) {
    if (!measuring && !armed && shape.phase == pl::Phase::kDraft) {
      armed = true;
      llmp::providers::fake::FailNextCopy(llmp::providers::CopyKind::kHostToDevice,
                                          shape.rows() * sizeof(std::int32_t), true);
    }
  };
  const auto executed = source_->executed();
  const auto failed = Unit(decoder, {0});
  llmp::providers::fake::ClearCopyFailure();
  source_->options().on_build = {};
  ASSERT_TRUE(armed);
  ASSERT_TRUE(failed.outcomes[0].refusal);
  ASSERT_FALSE(failed.outcomes[0].drafts.empty());
  ASSERT_TRUE(failed.outcomes[0].drafts[0].refusal);
  EXPECT_NE(failed.outcomes[0].drafts[0].refusal->what.find("an input copy"), std::string::npos);
  EXPECT_FALSE(failed.outcomes[0].refusal->clean);
  EXPECT_EQ(source_->executed(), executed);  // The transport failed before program Run.
  EXPECT_FALSE(slots().faulted());           // A retired transport error is local to this owner.
  for (std::size_t stage = 0; stage < slots().stages(); ++stage)
    EXPECT_TRUE(slots().live(0, stage).quarantined());
  EXPECT_TRUE(slots().usable(1));
  EXPECT_TRUE(slots().usable(2));
}

// The provider deliberately withholds retirement. Keep the complete node,
// source and runner alive until the subprocess exits; a failed teardown or
// a test destructor would not grant permission to release their allocations.
class UnprovenDraftProbe final : public DecodingTest {
 public:
  explicit UnprovenDraftProbe(bool direct) : DecodingTest(direct) {}
  void TestBody() override {}
  void Exercise() {
    SetUp();
    if (HasFatalFailure()) return;
    auto decoder = Make(Cascade());
    const std::array<std::uint32_t, 3> prompts{40, 41, 42};
    Prefill(prompts, &decoder);
    if (HasFatalFailure()) return;
    ASSERT_FALSE(Unit(decoder, {0, 1, 2}).outcomes[0].refusal);
    runner_->DropPlans();
    source_->options().fail_run = kn::StepError::kUnknown;
    source_->options().fail_phase = pl::Phase::kDraft;
    bool armed = false;
    auto& execution = static_cast<llmp::providers::fake::FakeDeviceExecution&>(node_.execution());
    source_->options().on_build = [&](const pl::StepShape& shape, bool measuring) {
      if (!measuring && !armed && shape.phase == pl::Phase::kDraft) {
        armed = true;
        execution.FailNextQuery(llmp::providers::ProviderError::kUnknown);
      }
    };
    const auto queued = source_->queued();
    const auto executed = source_->executed();
    const auto failed = Unit(decoder, {0});
    std::fprintf(stderr,
                 "lost-fence probe: direct=%d armed=%d queued=%llu->%llu executed=%llu->%llu\n",
                 node_.direct() ? 1 : 0, armed ? 1 : 0, static_cast<unsigned long long>(queued),
                 static_cast<unsigned long long>(source_->queued()),
                 static_cast<unsigned long long>(executed),
                 static_cast<unsigned long long>(source_->executed()));
    if (failed.outcomes[0].refusal)
      std::fprintf(stderr, "lost-fence outcome: %s\n", failed.outcomes[0].refusal->what.c_str());
    ASSERT_TRUE(armed);
    EXPECT_GT(source_->queued(), queued);  // Own draft was submitted before the lost fence.
    ASSERT_TRUE(failed.outcomes[0].refusal);
    ASSERT_FALSE(failed.outcomes[0].drafts.empty());
    ASSERT_TRUE(failed.outcomes[0].drafts[0].refusal);
    EXPECT_NE(failed.outcomes[0].drafts[0].refusal->what.find("the fake faulted"),
              std::string::npos);
    EXPECT_FALSE(failed.outcomes[0].refusal->clean);
    for (std::size_t stage = 0; stage < slots().stages(); ++stage)
      EXPECT_TRUE(slots().live(0, stage).quarantined());
    if (node_.direct()) {
      // DirectStep owns its fence outside the scheduler: the complete held
      // request and retained runner transaction protect every active peer.
      EXPECT_FALSE(node_.scheduler().fault());
      EXPECT_FALSE(slots().faulted());
      EXPECT_TRUE(node_.InRequest(0));
      EXPECT_TRUE(slots().usable(1));
      EXPECT_TRUE(slots().usable(2));
    } else {
      EXPECT_EQ(node_.scheduler().fault(), llmp::scheduler::Fault::kUnproven);
      EXPECT_TRUE(slots().faulted());
      for (std::uint32_t slot = 1; slot < kSlots; ++slot)
        for (std::size_t stage = 0; stage < slots().stages(); ++stage)
          EXPECT_TRUE(slots().live(slot, stage).quarantined());
    }
    const auto before_peer = source_->queued();
    const auto peer_past = slots().positions(1);
    const std::array token{Truth(peer_past)};
    const std::array work{en::SegmentWork{.tokens = token, .history = history_[1]}};
    const auto peer =
        runner_->RunStep({.shape = Shape(0, pl::Phase::kDecode, {Seg(1, peer_past, 1, 256)}),
                          .work = work,
                          .ahead = {}});
    ASSERT_FALSE(peer);
    EXPECT_FALSE(peer.error().clean);
    EXPECT_NE(peer.error().what.find("unproven effects"), std::string::npos);
    EXPECT_EQ(source_->queued(), before_peer);
    const auto held = runner_->plans().size();
    ASSERT_GT(held, 0U);
    runner_->DropPlans();
    EXPECT_EQ(runner_->plans().size(), held);  // Unproven transaction still protects its plans.
    if (!node_.direct()) return;               // the scheduler faulted: nothing less than a restart
    // D-102's rung 2: a later fence of the stream, seen complete, proves the
    // lost step's end. Acknowledged, the runner drops its plans and steps
    // again; the recovery discards the slot the failed work touched.
    source_->options().fail_run.reset();
    source_->options().fail_phase.reset();
    source_->options().on_build = {};
    auto fenced = node_.RecoverAfterFence(*runner_);
    ASSERT_TRUE(fenced) << fenced.error();
    EXPECT_FALSE(node_.InRequest(0));
    EXPECT_EQ(runner_->plans().size(), 0U);
    auto recovered = slots().RecoverInPlace({});
    ASSERT_TRUE(recovered.has_value()) << recovered.error();
    EXPECT_EQ(*recovered & 1U, 1U);
    ASSERT_TRUE(slots().SelectSlots(std::vector<std::uint32_t>{0, 1, 2}));
    ASSERT_TRUE(node_.BeginRequest(0, slots().execution(), "test request"));
    const std::array<std::uint32_t, 1> again{40};
    Prefill(again, &decoder);
    if (HasFatalFailure()) return;
    const auto next = Unit(decoder, {0, 1, 2});
    for (std::uint32_t slot = 0; slot < kSlots; ++slot)
      ASSERT_FALSE(next.outcomes[slot].refusal.has_value()) << next.outcomes[slot].refusal->what;
  }
};

void CheckUnprovenDraft(bool direct) {
  EXPECT_EXIT(
      {
        auto* retained = new UnprovenDraftProbe(direct);
        retained->Exercise();
        const auto* result = ::testing::UnitTest::GetInstance()->current_test_info()->result();
        for (int i = 0; i < result->total_part_count(); ++i) {
          const auto& part = result->GetTestPartResult(i);
          if (part.failed())
            std::fprintf(stderr, "%s:%d: %s\n", part.file_name() ? part.file_name() : "child",
                         part.line_number(), part.message());
        }
        std::fflush(stderr);
        std::_Exit(::testing::Test::HasFailure() ? 1 : 0);
      },
      ::testing::ExitedWithCode(0), "");
}

TEST(DecodingUnprovenDeathTest, DirectOwnProgramUnknownRetainsEveryAffectedStageAndCompleteOwner) {
  CheckUnprovenDraft(true);
}

TEST(DecodingUnprovenDeathTest,
     ScheduledOwnProgramUnknownRetainsEveryAffectedStageAndCompleteOwner) {
  CheckUnprovenDraft(false);
}

TEST_F(DecodingTest, ARefusedCommitAfterItsRestoreFailsItsSlotAlone) {
  auto o = Make(Cascade());
  const std::array<std::uint32_t, 3> prompts = {40, 41, 42};
  Prefill(prompts, &o);
  ASSERT_FALSE(Unit(o, {0}).outcomes[0].refusal.has_value());
  ASSERT_TRUE(slots().owed(0));
  // The next unit's first job queues the restore, then the commit refuses:
  // the state is neither the verify's nor the kept rows'.
  fail_commit_ = true;
  const auto failed = Unit(o, {0});
  fail_commit_ = false;
  ASSERT_TRUE(failed.outcomes[0].refusal.has_value());
  EXPECT_EQ(slots().positions(0), 45U);
  EXPECT_FALSE(slots().usable(0));
  EXPECT_TRUE(slots().live(0, 0).quarantined());
  EXPECT_FALSE(slots().faulted());
  EXPECT_TRUE(slots().usable(1));
  EXPECT_TRUE(slots().usable(2));
  ASSERT_TRUE(node_.EndRequest(0));
  auto recovered = slots().RecoverInPlace({});
  ASSERT_TRUE(recovered.has_value()) << recovered.error();
  EXPECT_EQ(*recovered, 0b001U);
  ASSERT_TRUE(slots().SelectSlots(std::vector<std::uint32_t>{0, 1, 2}));
  ASSERT_TRUE(node_.BeginRequest(0, slots().execution(), "test request"));
  EXPECT_EQ(slots().positions(1), 41U);
  const auto next = Unit(o, {1, 2});
  for (std::uint32_t i = 0; i < 2; ++i) {
    ASSERT_FALSE(next.outcomes[i].refusal.has_value()) << next.outcomes[i].refusal->what;
    EXPECT_EQ(next.outcomes[i].kept, 5U);
  }
}

TEST_F(DecodingTest, OwedWorkAndAStaleBorrowOutliveARecoveryThatSparesTheirSlots) {
  auto o = Make(Cascade());
  const std::array<std::uint32_t, 3> prompts = {40, 41, 42};
  Prefill(prompts, &o);
  accept_[0] = 1;
  ASSERT_FALSE(Unit(o, {0}).outcomes[0].refusal.has_value());
  ASSERT_TRUE(slots().owed(0));
  auto stale = slots().Borrow(1, 0);
  ASSERT_TRUE(stale.has_value()) << stale.error();
  // A recovery that touched neither slot: slot 0 still owes its restore
  // and commit; slot 1's borrow is gone, and its stale return inert.
  ASSERT_TRUE(node_.EndRequest(0));
  auto recovered = slots().RecoverInPlace({});
  ASSERT_TRUE(recovered.has_value()) << recovered.error();
  EXPECT_EQ(*recovered, 0U);
  EXPECT_TRUE(slots().owed(0));
  EXPECT_FALSE(slots().borrowed(1));
  ASSERT_TRUE(slots().SelectSlots(std::vector<std::uint32_t>{0, 1, 2}));
  ASSERT_TRUE(node_.BeginRequest(0, slots().execution(), "test request"));
  auto fresh = slots().Borrow(1, 0);
  ASSERT_TRUE(fresh.has_value()) << fresh.error();
  stale->Return();
  EXPECT_TRUE(slots().borrowed(1, 0));
  fresh->Return();
  EXPECT_FALSE(slots().borrowed(1));
  // The owed work runs first in the next unit's job: the kept rows' state.
  accept_[0] = kAll;
  const auto next = Unit(o, {0, 1});
  for (std::uint32_t slot = 0; slot < 2; ++slot) {
    ASSERT_FALSE(next.outcomes[slot].refusal.has_value()) << next.outcomes[slot].refusal->what;
  }
  EXPECT_EQ(verify_rows_[0].front().position, 42);
  EXPECT_EQ(verify_rows_[0].front().hash, Hash(std::span(history_[0]).first(43)));
  const auto cache = Read(0, 0, 0, 40, 2);
  EXPECT_TRUE(Holds(cache[0], 0, 40, Truth(40)));
  EXPECT_TRUE(Holds(cache[1], 0, 41, Truth(41)));
}

TEST_F(DecodingTest, ASlotSpilledBetweenItsDraftsAndItsVerifyIsRefusedAloneAndResumes) {
  auto o = Make(Cascade());
  const std::array<std::uint32_t, 3> prompts = {40, 41, 42};
  Prefill(prompts, &o);
  ASSERT_FALSE(Unit(o, {0, 1, 2}).outcomes[1].refusal.has_value());
  bool spilled = false;
  in_propose_ = [&](std::uint32_t slot, std::size_t drafter) {
    if (slot != 1 || drafter != 1 || spilled) return;
    spilled = true;
    auto spill = slots().Spill(1);
    EXPECT_TRUE(spill) << llmp::test_support::Failed(spill).value_or("");
  };
  const auto report = Unit(o, {0, 1, 2});
  in_propose_ = {};
  ASSERT_TRUE(spilled);
  ASSERT_TRUE(report.outcomes[1].refusal.has_value());
  EXPECT_TRUE(report.outcomes[1].refusal->clean);
  EXPECT_EQ(slots().positions(1), 46U);
  EXPECT_FALSE(slots().verify_pending(1));
  for (const std::uint32_t slot : {0U, 2U}) {
    ASSERT_FALSE(report.outcomes[slot].refusal.has_value());
    EXPECT_EQ(report.outcomes[slot].kept, 5U);
  }
  auto restored = slots().Restore(1);
  ASSERT_TRUE(restored) << llmp::test_support::Failed(restored).value_or("");
  const auto next = Unit(o, {0, 1, 2});
  for (std::uint32_t slot = 0; slot < kSlots; ++slot) {
    ASSERT_FALSE(next.outcomes[slot].refusal.has_value()) << next.outcomes[slot].refusal->what;
    EXPECT_EQ(next.outcomes[slot].kept, 5U);
    const std::uint32_t before = slots().positions(slot) - 5;
    EXPECT_EQ(verify_rows_[slot].front().hash,
              Hash(std::span(history_[slot]).first(std::size_t{before} + 1)));
  }
}

TEST_F(TruncatingTapTest, AFailedVerifyUndoesAStageItWritesAndSavesNothingInByNothing) {
  auto o = Make(Cascade());
  const std::array<std::uint32_t, 2> prompts = {40, 41};
  Prefill(prompts, &o);
  accept_ = {kAll, 1, kAll};
  const auto report = Unit(o, {0, 1});
  const std::array<std::uint32_t, 2> kept = {5, 2};
  for (std::uint32_t slot = 0; slot < 2; ++slot) {
    ASSERT_FALSE(report.outcomes[slot].refusal.has_value()) << report.outcomes[slot].refusal->what;
    EXPECT_EQ(report.outcomes[slot].kept, kept[slot]);
    EXPECT_TRUE(slots().live(slot, 0).saved().empty());
    EXPECT_TRUE(slots().live(slot, kMtp).saved().empty());  // its tap truncates
  }
  // A verify failing after its job queued work writes the MTP's tap past
  // the kept prefix, which is never read: undone whole by nothing, as the
  // target's truncating cache is, never quarantined.
  accept_ = {kAll, kAll, kAll};
  source_->options().fail_run = kn::StepError::kRejected;
  source_->options().fail_phase = pl::Phase::kVerify;
  const auto failed = Unit(o, {0, 1});
  source_->options().fail_run.reset();
  source_->options().fail_phase.reset();
  for (std::uint32_t slot = 0; slot < 2; ++slot) {
    ASSERT_TRUE(failed.outcomes[slot].refusal.has_value());
    EXPECT_EQ(slots().positions(slot), prompts[slot] + kept[slot]);
    EXPECT_FALSE(slots().verify_pending(slot));
    EXPECT_FALSE(slots().live(slot, kMtp).quarantined());
    EXPECT_TRUE(slots().usable(slot));
  }
  const auto next = Unit(o, {0, 1});
  for (std::uint32_t slot = 0; slot < 2; ++slot) {
    ASSERT_FALSE(next.outcomes[slot].refusal.has_value()) << next.outcomes[slot].refusal->what;
    EXPECT_EQ(next.outcomes[slot].kept, 5U);
    EXPECT_EQ(slots().positions(slot), prompts[slot] + kept[slot] + 5);
  }
}

// A fixed state that truncates: a verify rewrites it whole, so a partly
// accepted one would leave the rejected rows' state (before the refusal,
// the next verify's first row read the hash of every verified row).
class TruncatedFixedTest : public DecodingFixture {
 protected:
  md::Rollback KvRollback() const override { return md::Rollback::kSnapshot; }
  md::Rollback FixedRollback() const override { return md::Rollback::kTruncate; }
  bool RefusesSetup() const override { return true; }
};

TEST_F(TruncatedFixedTest, AVerifyOfStateItRewritesWholeCannotTruncateIt) {
  EXPECT_NE(setup_refusal_.find("cannot truncate"), std::string::npos) << setup_refusal_;
}

TEST_F(DecodingTest, CompletePassScheduleTruncatesOnlyTheVerifyAndDoesNotTeachTail) {
  auto spec = Mtp(3);
  spec.draft_before_limit = true;
  spec.adaptive = true;
  auto policy = Policy();
  std::uint32_t chosen = 3;
  std::vector<std::tuple<std::uint32_t, std::uint32_t, bool>> observations;
  policy.depth = [&](std::uint32_t, std::size_t) { return chosen; };
  policy.observe = [&](std::uint32_t, std::size_t, std::uint32_t depth, std::uint32_t kept,
                       bool complete) { observations.emplace_back(depth, kept, complete); };
  auto decoding =
      Make({.kind = en::DecodingMode::Kind::kSpeculative, .drafters = {spec}}, std::move(policy));
  const std::array<std::uint32_t, 1> prompt{40};
  Prefill(prompt, &decoding);
  const auto tail = Unit(decoding, {0}, 2);
  ASSERT_FALSE(tail.outcomes[0].refusal);
  ASSERT_EQ(tail.outcomes[0].drafts.size(), 1U);
  EXPECT_EQ(tail.outcomes[0].drafts[0].steps, 3U);
  EXPECT_EQ(tail.outcomes[0].drafts[0].proposed, 3U);
  EXPECT_EQ(verified_[0].size(), 2U);
  ASSERT_EQ(observations.size(), 1U);
  EXPECT_EQ(observations[0], (std::tuple{3U, 2U, false}));
  chosen = 2;
  const auto whole = Unit(decoding, {0});
  ASSERT_FALSE(whole.outcomes[0].refusal);
  EXPECT_EQ(whole.outcomes[0].drafts[0].steps, 2U);
  EXPECT_EQ(verified_[0].size(), 3U);
  EXPECT_EQ(observations.back(), (std::tuple{2U, 3U, true}));
}

// Near the context's end a complete pass schedule shrinks to the rows left
// (E8), rather than refusing the unit, and teaches nothing.
TEST_F(DecodingTest, CompletePassScheduleShrinksAtTheContextTail) {
  auto spec = Mtp(3);
  spec.draft_before_limit = true;
  auto policy = Policy();
  std::uint32_t observations = 0;
  bool taught = false;
  policy.observe = [&](std::uint32_t, std::size_t, std::uint32_t, std::uint32_t, bool complete) {
    ++observations;
    taught = taught || complete;
  };
  auto decoding =
      Make({.kind = en::DecodingMode::Kind::kSpeculative, .drafters = {spec}}, std::move(policy));
  en::PrefillOrchestrator prefill(*runner_, *composer_,
                                  {.chunk_rows = kChunk,
                                   .lookahead = 0,
                                   .capture_ahead = false,
                                   .prepare_ahead = false,
                                   .state_only = true});
  for (std::uint32_t q = 0; q < kContext - 2; ++q) history_[0].push_back(Truth(q));
  for (std::uint32_t past = 0; past < kContext - 2;) {
    const en::PromptWork prompt{.slot = 0,
                                .stage = 0,
                                .tokens = std::span(history_[0]).subspan(past),
                                .past = past,
                                .want_head = true,
                                .all_outputs = false,
                                .boundary = 0,
                                .history = history_[0]};
    const auto report = prefill.Step(std::span(&prompt, 1));
    ASSERT_EQ(report.outcomes().size(), 1U);
    ASSERT_FALSE(report.outcomes()[0].refusal) << report.outcomes()[0].refusal->what;
    ASSERT_EQ(report.outcomes()[0].rows, std::min(kChunk, kContext - 2 - past));
    past += report.outcomes()[0].rows;
    ASSERT_EQ(report.outcomes()[0].last, past == kContext - 2);
    ASSERT_EQ(runner_->slots().positions(0), past);
  }
  history_[0].push_back(Truth(kContext - 2));
  decoding.Reset(0);
  const auto tail = Unit(decoding, {0}, 8);
  ASSERT_FALSE(tail.outcomes[0].refusal) << tail.outcomes[0].refusal->what;
  ASSERT_EQ(tail.outcomes[0].drafts.size(), 1U);
  EXPECT_EQ(tail.outcomes[0].drafts[0].depth, 2U);  // the two rows left, not 3
  EXPECT_EQ(tail.outcomes[0].drafts[0].steps, 2U);
  EXPECT_EQ(tail.outcomes[0].drafts[0].proposed, 2U);
  EXPECT_EQ(verified_[0].size(), 2U);  // the anchor and the one draft that fits
  EXPECT_EQ(runner_->slots().positions(0), kContext);
  EXPECT_EQ(observations, 1U);
  EXPECT_FALSE(taught);
}

// A full-block verify holds every proposal of its block even past what the
// reply may still emit; its judge sees only the rows within `left` (E7).
TEST_F(DecodingTest, AFullBlockVerifyJudgesOnlyTheRowsTheReplyMayEmit) {
  auto spec = Block(3);
  spec.draft_before_limit = true;
  auto decoding =
      Make({.kind = en::DecodingMode::Kind::kSpeculative, .drafters = {spec}}, Policy());
  const std::array<std::uint32_t, 1> prompt{40};
  Prefill(prompt, &decoding);
  const auto tail = Unit(decoding, {0}, 2);
  ASSERT_FALSE(tail.outcomes[0].refusal) << tail.outcomes[0].refusal->what;
  EXPECT_EQ(tail.outcomes[0].proposed.size(), 3U);  // the whole block verified
  EXPECT_EQ(verified_[0].size(), 2U);               // only `left` rows judged
  EXPECT_LE(tail.outcomes[0].kept, 2U);
  EXPECT_LE(tail.outcomes[0].tokens.size(), 2U);
}

TEST_F(DecodingTest, CompletePassScheduleRefusesMultiDrafterDeclaration) {
  auto spec = Mtp(3);
  spec.draft_before_limit = true;
  const auto made = en::DecodingOrchestrator::Make(
      *runner_, *composer_,
      {.kind = en::DecodingMode::Kind::kSpeculative, .drafters = {spec, Block(1)}}, Policy());
  EXPECT_FALSE(made);
}

TEST_F(DecodingTest, SharedWaveDepthCapsMultiSlotWavesAndLeavesLoneWavesAdaptive) {
  auto invalid_depth = Mtp(2);
  invalid_depth.shared_wave_depth = 3;
  EXPECT_FALSE(en::DecodingOrchestrator::Make(
      *runner_, *composer_,
      {.kind = en::DecodingMode::Kind::kSpeculative, .drafters = {invalid_depth}}, Policy()));
  auto invalid_block = Block(3);
  invalid_block.shared_wave_depth = 2;
  EXPECT_FALSE(en::DecodingOrchestrator::Make(
      *runner_, *composer_,
      {.kind = en::DecodingMode::Kind::kSpeculative, .drafters = {invalid_block}}, Policy()));

  auto spec = Mtp(3);
  spec.shared_wave_depth = 2;
  spec.draft_before_limit = true;
  spec.adaptive = true;
  auto policy = Policy();
  std::vector<std::tuple<std::uint32_t, std::uint32_t, std::uint32_t, bool>> observations;
  policy.observe = [&](std::uint32_t slot, std::size_t, std::uint32_t depth, std::uint32_t kept,
                       bool complete) { observations.emplace_back(slot, depth, kept, complete); };
  auto decoding =
      Make({.kind = en::DecodingMode::Kind::kSpeculative, .drafters = {spec}}, std::move(policy));
  const std::array<std::uint32_t, 2> prompts{40, 41};
  Prefill(prompts, &decoding);

  const auto lone = Unit(decoding, {0});
  ASSERT_FALSE(lone.outcomes[0].refusal);
  ASSERT_EQ(lone.outcomes[0].drafts.size(), 1U);
  EXPECT_EQ(lone.outcomes[0].drafts[0].depth, 3U);
  EXPECT_EQ(lone.outcomes[0].drafts[0].steps, 3U);
  EXPECT_EQ(lone.outcomes[0].drafts[0].proposed, 3U);
  EXPECT_EQ(verified_[0].size(), 4U);
  ASSERT_EQ(observations.size(), 1U);
  EXPECT_EQ(observations[0], (std::tuple{0U, 3U, 4U, true}));

  observations.clear();
  const auto shared = Unit(decoding, {0, 1});
  for (std::uint32_t slot = 0; slot < 2; ++slot) {
    ASSERT_FALSE(shared.outcomes[slot].refusal);
    ASSERT_EQ(shared.outcomes[slot].drafts.size(), 1U);
    EXPECT_EQ(shared.outcomes[slot].drafts[0].depth, 2U);
    EXPECT_EQ(shared.outcomes[slot].drafts[0].steps, 2U);
    EXPECT_EQ(shared.outcomes[slot].drafts[0].proposed, 2U);
    EXPECT_EQ(verified_[slot].size(), 3U);
  }
  ASSERT_EQ(observations.size(), 2U);
  EXPECT_EQ(observations[0], (std::tuple{0U, 2U, 3U, true}));
  EXPECT_EQ(observations[1], (std::tuple{1U, 2U, 3U, true}));

  observations.clear();
  const auto shared_tail = Unit(decoding, {0, 1}, 2);
  for (std::uint32_t slot = 0; slot < 2; ++slot) {
    ASSERT_FALSE(shared_tail.outcomes[slot].refusal);
    ASSERT_EQ(shared_tail.outcomes[slot].drafts.size(), 1U);
    EXPECT_EQ(shared_tail.outcomes[slot].drafts[0].depth, 2U);
    EXPECT_EQ(shared_tail.outcomes[slot].drafts[0].steps, 2U);
    EXPECT_EQ(shared_tail.outcomes[slot].drafts[0].proposed, 2U);
    EXPECT_EQ(verified_[slot].size(), 2U);
  }
  ASSERT_EQ(observations.size(), 2U);
  EXPECT_EQ(observations[0], (std::tuple{0U, 2U, 2U, false}));
  EXPECT_EQ(observations[1], (std::tuple{1U, 2U, 2U, false}));
}

class FourSlotDecodingTest : public DecodingTest {
 protected:
  std::uint32_t Slots() const override { return 4; }
};

TEST_F(FourSlotDecodingTest, DraftWaveMaxExecutesAloneDraftsWhileVerifyJoins) {
  auto spec = Mtp(2);
  spec.draft_wave_max = 2;
  spec.catch_up = true;
  auto decoding = Make({.kind = en::DecodingMode::Kind::kSpeculative, .drafters = {spec}});
  const std::array<std::uint32_t, 4> prompts{40, 41, 42, 43};
  Prefill(prompts, &decoding);

  // Unit 1: First speculative unit across all 4 slots.
  // 4 active slots > draft_wave_max (2), so every draft step runs alone (1 segment).
  // Target verify runs with all 4 slots joined into 1 step (4 segments).
  // 4 alone pass 0 drafts + 4 alone pass 1 drafts + 1 joined verify = 9 steps total.
  const std::size_t from_first = source_->built().size();
  const auto unit1 = Unit(decoding, {0, 1, 2, 3});
  EXPECT_EQ(unit1.steps, 9U);
  for (std::uint32_t slot = 0; slot < 4; ++slot) {
    ASSERT_FALSE(unit1.outcomes[slot].refusal);
    EXPECT_EQ(unit1.outcomes[slot].kept, 3U);
    EXPECT_EQ(unit1.outcomes[slot].drafts.size(), 1U);
    EXPECT_EQ(unit1.outcomes[slot].drafts[0].steps, 2U);
  }

  std::uint32_t first_draft_steps = 0;
  std::uint32_t first_verify_steps = 0;
  for (std::size_t i = from_first; i < source_->built().size(); ++i) {
    const auto& b = source_->built()[i];
    if (b.measuring) continue;
    if (b.shape.phase == pl::Phase::kDraft && b.shape.stage == kMtp) {
      EXPECT_EQ(b.shape.segments.size(), 1U);
      ++first_draft_steps;
    } else if (b.shape.phase == pl::Phase::kVerify && b.shape.stage == 0) {
      EXPECT_EQ(b.shape.segments.size(), 4U);
      ++first_verify_steps;
    }
  }
  EXPECT_GT(first_draft_steps, 0U);
  EXPECT_GT(first_verify_steps, 0U);

  // Unit 2: The next speculative unit across all 4 slots.
  // The first pass of MTP is a catch-up step that catches up the kept tokens from unit 1
  // (rows = 4) and proposes the first draft token.
  // Because 4 active slots > draft_wave_max (2), catch-up draft steps also run alone (1 segment).
  // 4 alone pass 0 catch-up drafts + 4 alone pass 1 drafts + 1 joined verify = 9 steps total.
  const std::size_t from_second = source_->built().size();
  const auto unit2 = Unit(decoding, {0, 1, 2, 3});
  EXPECT_EQ(unit2.steps, 9U);
  for (std::uint32_t slot = 0; slot < 4; ++slot) {
    ASSERT_FALSE(unit2.outcomes[slot].refusal);
    EXPECT_EQ(unit2.outcomes[slot].kept, 3U);
    EXPECT_EQ(unit2.outcomes[slot].drafts.size(), 1U);
    EXPECT_EQ(unit2.outcomes[slot].drafts[0].steps, 2U);
  }

  std::uint32_t second_catchup_steps = 0;
  for (std::size_t i = from_second; i < source_->built().size(); ++i) {
    const auto& b = source_->built()[i];
    if (b.measuring) continue;
    if (b.shape.phase == pl::Phase::kDraft && b.shape.stage == kMtp) {
      EXPECT_EQ(b.shape.segments.size(), 1U);
      EXPECT_EQ(b.shape.segments.front().rows, 4U);
      ++second_catchup_steps;
    }
  }
  EXPECT_GT(second_catchup_steps, 0U);
}

TEST_F(DecodingTest, DuplicateSlotsRefusedInSingleCertainAndMixedWaves) {
  auto plain = Make({.kind = en::DecodingMode::Kind::kPlain});
  const std::array<std::uint32_t, 2> prompts{10, 12};
  Prefill(prompts, &plain);

  // 1. Plain Single wave with duplicate slot 0: first unit executes, duplicate is refused.
  {
    const std::array<en::DecodeWork, 2> dup_single{
        en::DecodeWork{.slot = 0, .history = history_[0], .left = kAll},
        en::DecodeWork{.slot = 0, .history = history_[0], .left = kAll}};
    const auto report = plain.Step(dup_single);
    EXPECT_EQ(report.steps, 1U);
    ASSERT_EQ(report.outcomes.size(), 2U);
    EXPECT_FALSE(report.outcomes[0].refusal);
    EXPECT_EQ(report.outcomes[0].kept, 1U);
    ASSERT_TRUE(report.outcomes[1].refusal);
    EXPECT_NE(report.outcomes[1].refusal->what.find("no unit of its own"), std::string::npos);
    EXPECT_EQ(slots().positions(0), 11U);
    history_[0].push_back(Truth(11));
  }
  // 2. Certain wave with duplicate slot 0: first unit executes, duplicate is refused.
  {
    const std::array<std::int32_t, 1> certain_toks{Truth(12)};
    const std::array<en::DecodeWork, 2> dup_certain{
        en::DecodeWork{.slot = 0, .history = history_[0], .left = kAll, .certain = certain_toks},
        en::DecodeWork{.slot = 0, .history = history_[0], .left = kAll, .certain = certain_toks}};
    const auto report = plain.Step(dup_certain);
    EXPECT_EQ(report.steps, 1U);
    ASSERT_EQ(report.outcomes.size(), 2U);
    EXPECT_FALSE(report.outcomes[0].refusal);
    EXPECT_EQ(report.outcomes[0].kept, 2U);
    ASSERT_TRUE(report.outcomes[1].refusal);
    EXPECT_NE(report.outcomes[1].refusal->what.find("no unit of its own"), std::string::npos);
    EXPECT_EQ(slots().positions(0), 13U);
    history_[0].push_back(Truth(12));
    history_[0].push_back(Truth(13));
  }
  // 3. Mixed Certain + Single wave with duplicate slot 0 across sub-waves.
  {
    const std::array<std::int32_t, 1> certain_toks{Truth(14)};
    const std::array<en::DecodeWork, 2> dup_mixed{
        en::DecodeWork{.slot = 0, .history = history_[0], .left = kAll, .certain = certain_toks},
        en::DecodeWork{.slot = 0, .history = history_[0], .left = kAll}};
    const auto report = plain.Step(dup_mixed);
    EXPECT_EQ(report.steps, 1U);
    ASSERT_EQ(report.outcomes.size(), 2U);
    EXPECT_FALSE(report.outcomes[0].refusal);
    EXPECT_EQ(report.outcomes[0].kept, 2U);
    ASSERT_TRUE(report.outcomes[1].refusal);
    EXPECT_NE(report.outcomes[1].refusal->what.find("no unit of its own"), std::string::npos);
    EXPECT_EQ(slots().positions(0), 15U);
    history_[0].push_back(Truth(14));
    history_[0].push_back(Truth(15));
  }
  // Slot 0 state remains consistent and usable.
  const auto ok = Unit(plain, {0});
  EXPECT_EQ(ok.steps, 1U);
  EXPECT_FALSE(ok.outcomes[0].refusal);
  EXPECT_EQ(slots().positions(0), 16U);
}

}  // namespace
