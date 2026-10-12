// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The LLM serving adapter (runtime/engine_llm.h) on the fake device
// backend: Llm's own conversation code (prompt sessions, generation
// sessions, waves, spills, turn checkpoints, kept conversations) over the
// shared runner, its step programs the fake step source's in its logits
// mode (tests/support/fake_step_source.h). A head row there names a token
// that follows from the row's token and position alone, but only while the
// target's state holds every earlier position; so an uninterrupted chain
// of choices is what every generation must reproduce exactly: plain,
// greedy on the device, speculative with two drafters (an MTP-like one
// that catches up and a DSpark-like block one, cascaded), joined in waves,
// after a spill, a turn checkpoint's restore, a recovery, and a restart
// that adopts the conversation from its kept record (D-105, version 2: a
// spill file a stage), where an old record of version 1 is refused.

#include "runtime/engine_llm.h"

#include <gtest/gtest.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <expected>
#include <filesystem>
#include <format>
#include <fstream>
#include <functional>
#include <iterator>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <system_error>
#include <tuple>
#include <utility>
#include <vector>

#include "base/bytes.h"
#include "base/sha256.h"
#include "engine/paged_node.h"
#include "expected_error.h"
#include "fake_step_source.h"
#include "kernels/common/recording_launch.h"
#include "model/batching.h"
#include "model/state_contract.h"
#include "planner/step_shape.h"
#include "platform/kept_files.h"
#include "runtime/kept_record.h"
#include "runtime/serving.h"
#include "runtime/state_keeper.h"

namespace {

namespace en = llmp::engine;
namespace kn = llmp::kernels;
namespace md = llmp::model;
namespace pl = llmp::planner;
namespace rt = llmp::runtime;
namespace ts = llmp::test_support;
namespace fs = std::filesystem;
namespace kept = llmp::runtime::kept;
namespace pf = llmp::platform;
using llmp::base::Bytes;
using llmp::test_support::Failed;
using C = md::StateTensorContract;
using Layers = std::vector<std::pair<std::uint32_t, md::StateContract>>;
constexpr std::uint64_t kExtent = en::kPagedExtent;
constexpr std::uint32_t kContext = 1024;
constexpr std::uint32_t kChunk = 64;
constexpr std::uint32_t kSlots = 2;
constexpr std::uint32_t kVerifyRows = 8;
constexpr std::uint32_t kWidth = 61;  // the head's logits; tokens below it
constexpr std::int32_t kFill = 59;
constexpr std::uint16_t kMtp = 1, kBlock = 2;

std::string Why(const rt::Status& s) { return Failed(s).value_or(""); }

// The target: a cache truncated by a verify. The MTP-like drafter: its own
// cache and a tap of the target's features (a position ring the target
// writes). The DSpark-like drafter: a ring both it and the target write.
std::vector<en::StageState> StageStates(bool drafters) {
  const Layers target = {{0,
                          {.tensors = {C{.name = "kv",
                                         .kind = md::StateKind::kKv,
                                         .cell_bytes = 256,
                                         .rollback = md::Rollback::kTruncate}}}}};
  const Layers mtp = {{0,
                       {.tensors = {C{.name = "kv", .kind = md::StateKind::kKv, .cell_bytes = 256},
                                    C{.name = "streams",
                                      .kind = md::StateKind::kFeatures,
                                      .encoding = md::Encoding::kF32,
                                      .cell_bytes = 256,
                                      .cells = md::CellRule::kPositionRing,
                                      .window_or_ratio = 64,
                                      .rollback = md::Rollback::kSnapshot,
                                      .target_writes = true}}}}};
  const Layers block = {{0,
                         {.tensors = {C{.name = "ring",
                                        .kind = md::StateKind::kDrafterRing,
                                        .cell_bytes = 256,
                                        .cells = md::CellRule::kPositionRing,
                                        .window_or_ratio = 256,
                                        .rollback = md::Rollback::kSnapshot,
                                        .target_writes = true}}}}};
  std::vector<en::StageState> stages = {
      {.name = "target", .layout = md::Layout(target, kContext, kChunk).value(), .read_align = 0}};
  if (drafters) {
    stages.push_back(
        {.name = "mtp", .layout = md::Layout(mtp, kContext, kChunk).value(), .read_align = 512});
    stages.push_back(
        {.name = "block", .layout = md::Layout(block, kContext, kChunk).value(), .read_align = 0});
  }
  return stages;
}

pl::StepShape Wide(std::uint16_t stage, pl::Phase phase, std::uint32_t rows, std::uint32_t align) {
  pl::StepShape shape{
      .stage = stage, .phase = phase, .segments = {}, .row_invariant = false, .read_align = align};
  for (std::uint8_t s = 0; s < kSlots; ++s) {
    shape.segments.push_back({.slot = s,
                              .rows = rows,
                              .past = kContext - rows,
                              .read_cells = kContext,
                              .outputs = rows,
                              .group = 0,
                              .want_head = true,
                              .greedy = false});
  }
  return shape;
}

// The token a chain of greedy choices gives after `prompt`: each from the
// one before it at its position (the fake's head).
std::vector<std::int32_t> Chain(std::span<const std::int32_t> prompt, std::size_t n) {
  std::vector<std::int32_t> out;
  std::int32_t token = prompt.back();
  auto position = static_cast<std::int32_t>(prompt.size() - 1);
  for (std::size_t k = 0; k < n; ++k) {
    token = ts::FakeLogit(token, position++, kWidth);
    out.push_back(token);
  }
  return out;
}

std::vector<std::int32_t> Prompt(std::size_t n, std::int32_t seed) {
  std::vector<std::int32_t> out;
  for (std::size_t i = 0; i < n; ++i) {
    out.push_back(
        static_cast<std::int32_t>((static_cast<std::int64_t>(i) * 13 + seed) % (kWidth - 1)));
  }
  return out;
}

std::int32_t ArgmaxOf(std::span<const float> row) {
  return static_cast<std::int32_t>(std::ranges::max_element(row) - row.begin());
}

class EngineLlmTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // NOLINTNEXTLINE(concurrency-mt-unsafe): test environment is immutable
    const char* scratch = std::getenv("LLMP_TEST_SCRATCH");
    ASSERT_NE(scratch, nullptr);
    root_ = fs::path(scratch) / std::format("engine-llm-{}", ::getpid());
    fs::remove_all(root_);
    fs::create_directories(root_);
    fs::permissions(root_, fs::perms::owner_all, fs::perm_options::replace);
    roles_ = {
        .installed = root_ / "installed", .spill = root_, .state = root_ / "state", .warnings = {}};
    entry_.name = "fake";
    spec_ = llmp::base::Sha256().Update(std::string_view("an engine-backed fake")).Finish();
  }
  void TearDown() override {
    Stop();
    fs::remove_all(root_);
  }

  rt::ModelSettings Settings(bool speculate) const {
    rt::ModelSettings s;
    s.name = "fake";
    s.architecture = "fake";
    s.context.value = kContext;
    s.prefill_chunk.value = kChunk;
    s.max_slots.value = kSlots;
    s.speculation.value = speculate;
    s.depth_cost_ratio.value = 1.16;
    return s;
  }

  rt::EngineBuild Build(bool drafters) {
    return [this, drafters](const rt::ModelSettings& settings)
               -> std::expected<rt::EngineLlmParts, std::string> {
      EXPECT_EQ(settings.context.value, kContext);
      rt::EngineLlmParts parts;
      std::vector<en::StageState> stages = StageStates(drafters);
      if (kept_progress_) {
        stages.resize(2);
        const Layers layers{{0,
                             {.tensors = {C{.name = "kv", .cell_bytes = 256, .replayable = true},
                                          C{.name = "streams",
                                            .kind = md::StateKind::kFeatures,
                                            .encoding = md::Encoding::kF32,
                                            .cell_bytes = 256,
                                            .cells = md::CellRule::kPerRow,
                                            .window_or_ratio = 1,
                                            .rollback = md::Rollback::kSnapshot,
                                            .target_writes = true},
                                          md::DraftProgressControl(
                                              "progress", md::DraftRelation::kTokenAfterCache)}}}};
        stages[1].layout = md::Layout(layers, kContext, kChunk).value();
      }
      ts::FakeSourceOptions o;
      for (const auto& s : stages) o.layouts.push_back(s.layout);
      o.logits = kWidth;
      o.features = native_outputs_;
      o.spec = spec_;
      auto source = std::make_unique<ts::FakeStepSource>(std::move(o));
      source_ = source.get();
      parts.config.slots = {.model = "ignored",
                            .slots = 1,
                            .stages = stages,
                            .spec = spec_,
                            .read_align = 256,
                            .spill_place = {},
                            .spill_directory = {},
                            .invalidate = {},
                            .drop_plans = {}};
      parts.config.setup_shapes = {Wide(0, pl::Phase::kPrefill, kChunk, 256)};
      if (drafters) {
        parts.config.setup_shapes.push_back(Wide(0, pl::Phase::kVerify, kVerifyRows, 256));
        parts.config.setup_shapes.push_back(Wide(kMtp, pl::Phase::kDraft, kChunk, 512));
        if (!kept_progress_)
          parts.config.setup_shapes.push_back(Wide(kBlock, pl::Phase::kDraft, kVerifyRows, 256));
      }
      parts.config.launch = []() -> std::expected<std::unique_ptr<kn::StepLaunch>, std::string> {
        return std::make_unique<kn::RecordingLaunch>();
      };
      parts.config.max_verify_rows = drafters ? kVerifyRows : 0;
      parts.source = std::move(source);
      parts.caps.max_segments = kSlots;
      parts.caps.join_prefill = true;
      if (drafters) {
        parts.mode = {.kind = en::DecodingMode::Kind::kSpeculative,
                      .drafters = {{.stage = kMtp,
                                    .shape = en::DrafterSpec::Shape::kAutoregressive,
                                    .max_depth = 2,
                                    .catch_up = true},
                                   {.stage = kBlock,
                                    .shape = en::DrafterSpec::Shape::kBlock,
                                    .max_depth = 2,
                                    .fill = kFill}},
                      .combine = en::DecodingMode::Drafters::kCascade,
                      .row_invariant = false};
      }
      if (drafters && native_outputs_) {
        parts.mode.drafters.resize(1);
        auto& draft = parts.mode.drafters.front();
        draft.max_depth = 3;
        draft.features = true;
        draft.device_tokens = true;
        draft.draft_before_limit = true;
        draft.adaptive = true;
        parts.device_verify = true;
        parts.branch_depth = 3;
      }
      parts.logit_width = kWidth;
      parts.device_greedy = true;
      parts.prefill = {.chunk_rows = kChunk,
                       .lookahead = 2,
                       .capture_ahead = true,
                       .prepare_ahead = true,
                       .state_only = true};
      parts.architecture = "fake";
      return parts;
    };
  }

  // A node and the model on it, as the server registers it: Setup, the
  // workspace, the node's start, then `before_register` (the kept
  // conversations' places, D-105), Register, the node runs, Bind.
  void Start(bool drafters, bool speculate,
             const std::function<void(rt::EngineLlm&)>& before_register = {}) {
    node_ = std::make_unique<en::PagedNode>(en::NodeSettings{
        .compute_streams = 1, .slots = 2, .copy_lane = true, .slot_bytes = kExtent});
    ASSERT_TRUE(node_->Open());
    const std::uint64_t fixed = node_->catalog().OccupancyOf(node_->domain()).Total().value();
    llm_ = std::make_unique<rt::EngineLlm>(*node_, entry_, Settings(speculate), roles_, 0,
                                           Build(drafters));
    llm_->set_turn_checkpoints(save_checkpoints_);
    auto set = llm_->Setup();
    ASSERT_TRUE(set) << Why(set);
    ASSERT_TRUE(
        node_->MapWorkspace(llm_->activations_needed(), std::max(llm_->pool_needed(), kExtent)));
    auto started = node_->Start(Bytes(fixed + (128 * kExtent)));
    ASSERT_TRUE(started) << Why(started);
    if (before_register) before_register(*llm_);
    auto registered = llm_->Register();
    ASSERT_TRUE(registered) << Why(registered);
    node_->Run();
    auto bound = llm_->Bind();
    ASSERT_TRUE(bound) << Why(bound);
  }

  void Stop() {
    if (node_ == nullptr) return;
    if (node_->InRequest(0)) EXPECT_TRUE(node_->EndRequest(0));
    if (llm_ != nullptr) {
      std::array<en::PagedModel*, 1> models = {&llm_->paged()};
      auto closed = node_->TearDown(models);
      EXPECT_TRUE(closed) << Why(closed);
    }
    llm_.reset();
    node_.reset();
  }

  rt::Llm::Branch& BranchAt(std::size_t index) { return **llm_->branch(index); }

  void Open(std::vector<rt::Llm::Branch*> branches) {
    auto selected = llm_->SelectBranches(branches);
    ASSERT_TRUE(selected) << Why(selected);
    auto begun = node_->BeginRequest(0, llm_->request_closure(), "a test turn");
    ASSERT_TRUE(begun) << Why(begun);
  }
  void Close() {
    auto ended = node_->EndRequest(0);
    EXPECT_TRUE(ended) << Why(ended);
  }

  struct Turned {
    rt::Generation out;
    std::uint32_t reused = 0;
  };
  // One turn on a branch, in a request of its own, as the chat route runs
  // one: its prompt prepared (reuse, chunks, a checkpoint at `boundary`),
  // then its reply generated.
  Turned Turn(std::size_t index, const std::vector<std::int32_t>& prompt, std::uint32_t max_tokens,
              bool keep_logits = false, std::uint32_t boundary = 0) {
    Turned t;
    rt::Llm::Branch& branch = BranchAt(index);
    Open({&branch});
    std::vector<float> last;
    auto prepared = branch.PreparePrompt(prompt, boundary, last, t.reused);
    EXPECT_TRUE(prepared) << Why(prepared);
    rt::GenerateOptions options;
    options.max_tokens = max_tokens;
    options.keep_logits = keep_logits;
    if (prepared) {
      auto generated = branch.Generate(last, options, t.out);
      EXPECT_TRUE(generated) << Why(generated);
    }
    Close();
    return t;
  }

  // What follows a branch's history: its next prompt continues it.
  std::vector<std::int32_t> Continue(std::size_t index, const rt::Generation& out,
                                     const std::vector<std::int32_t>& more) {
    std::vector<std::int32_t> prompt = BranchAt(index).history();
    prompt.push_back(out.tokens.back());  // the last reply token, not yet in the state
    prompt.insert(prompt.end(), more.begin(), more.end());
    return prompt;
  }

  std::vector<std::byte> NativeImage(std::uint32_t owner) const {
    std::vector<std::byte> image;
    const auto append = [&](const void* data, std::size_t bytes) {
      const auto* first = static_cast<const std::byte*>(data);
      image.insert(image.end(), first, first + bytes);
    };
    const auto& slots = llm_->runner().slots();
    const auto positions = slots.positions(owner);
    append(&positions, sizeof(positions));
    for (std::size_t stage = 0; stage < slots.stages(); ++stage) {
      const auto& live = slots.live(owner, stage);
      for (const auto& range : live.used_ranges()) {
        append(&range.region, sizeof(range.region));
        append(&range.offset, sizeof(range.offset));
        append(&range.bytes, sizeof(range.bytes));
        append(reinterpret_cast<const void*>(live.base(range.region) + range.offset), range.bytes);
      }
    }
    return image;
  }

  fs::path root_;
  llmp::config::RuntimeRoles roles_;
  llmp::config::ModelEntry entry_;
  llmp::base::Sha256Digest spec_{};
  std::unique_ptr<en::PagedNode> node_;
  std::unique_ptr<rt::EngineLlm> llm_;
  ts::FakeStepSource* source_ = nullptr;
  bool native_outputs_ = false;
  bool kept_progress_ = false;
  bool save_checkpoints_ = true;  // Llm::set_turn_checkpoints at Start
};

// Snapshot payload and variable metadata have one catalog charge. These are
// integration controls over the real slot copy/restore path on the fake node;
// no test receives a writable pointer into the opaque snapshot.
TEST_F(EngineLlmTest, OpaqueSnapshotRestoresCapturedBytesAndHistoryWithOneCharge) {
  Start(false, false);
  auto& branch = BranchAt(0);
  Open({&branch});
  const auto prompt = Prompt(17, 3);
  std::vector<float> logits;
  ASSERT_TRUE(branch.Prefill(prompt, logits));
  const auto occupancy = [&] {
    std::uint64_t value = 0;
    EXPECT_TRUE(node_->Call(
        [&]() -> rt::Status {
          value = node_->catalog().OccupancyOf(node_->domain()).Total().value();
          return {};
        },
        "snapshot test occupancy"));
    return value;
  };
  const auto before = occupancy();
  {
    rt::SnapshotStore store(*node_, 8 * kExtent);
    auto saved = branch.CaptureState(store);
    ASSERT_TRUE(saved) << Failed(saved).value_or("");
    auto original = store.Digest(*saved);
    ASSERT_TRUE(original) << Failed(original).value_or("");
    EXPECT_EQ(occupancy() - before, store.capacity());
    const auto tail = Prompt(3, 7);
    ASSERT_TRUE(branch.Prefill(tail, logits));
    EXPECT_NE(branch.history(), prompt);
    ASSERT_TRUE(branch.RestoreState(store, *saved));
    EXPECT_EQ(branch.history(), prompt);
    EXPECT_EQ(llm_->runner().slots().positions(0), prompt.size());
    // Capture again through the same owned allocation: old capability expires,
    // while the actual copied bytes match the first completed capture.
    auto again = branch.CaptureState(store);
    ASSERT_TRUE(again) << Failed(again).value_or("");
    EXPECT_FALSE(store.Digest(*saved));
    auto repeated = store.Digest(*again);
    ASSERT_TRUE(repeated) << Failed(repeated).value_or("");
    EXPECT_EQ(*repeated, *original);
  }
  // Plans/state may grow with the tail, so only the snapshot's exact charge is
  // checked while its lifetime is isolated, above. Continuation checks actual
  // logical state rather than only the diagnostic byte hash.
  rt::GenerateOptions options;
  options.max_tokens = 4;
  rt::Generation generated;
  // The captured frontier's logits are reconstructed by a one-row score-free
  // continuation prompt; the state prefix remains the captured history.
  const auto next = Prompt(1, 9);
  ASSERT_TRUE(branch.Prefill(next, logits));
  auto continued = prompt;
  continued.insert(continued.end(), next.begin(), next.end());
  ASSERT_TRUE(branch.Generate(logits, options, generated));
  EXPECT_EQ(generated.tokens, Chain(continued, 4));
  Close();
}

TEST_F(EngineLlmTest, OpaqueSnapshotRejectsForeignStoreAndOwnerBeforeAnyRestore) {
  Start(false, false);
  auto& a = BranchAt(0);
  auto& b = BranchAt(1);
  Open({&a, &b});
  std::vector<float> logits;
  ASSERT_TRUE(a.Prefill(Prompt(7, 1), logits));
  ASSERT_TRUE(b.Prefill(Prompt(7, 8), logits));
  rt::SnapshotStore first(*node_, 8 * kExtent);
  rt::SnapshotStore second(*node_, 8 * kExtent);
  auto saved = a.CaptureState(first);
  ASSERT_TRUE(saved) << Failed(saved).value_or("");
  const auto a_history = a.history();
  const auto b_history = b.history();
  const auto a_position = llm_->runner().slots().positions(0);
  const auto b_position = llm_->runner().slots().positions(1);
  EXPECT_FALSE(second.Digest(*saved));
  EXPECT_FALSE(a.RestoreState(second, *saved));
  EXPECT_FALSE(b.RestoreState(first, *saved));
  EXPECT_EQ(a.history(), a_history);
  EXPECT_EQ(b.history(), b_history);
  EXPECT_EQ(llm_->runner().slots().positions(0), a_position);
  EXPECT_EQ(llm_->runner().slots().positions(1), b_position);
  EXPECT_FALSE(llm_->runner().slots().owed(0));
  EXPECT_FALSE(llm_->runner().slots().owed(1));
  EXPECT_TRUE(first.Digest(*saved));
  Close();
}

TEST_F(EngineLlmTest, OpaqueSnapshotCanRecaptureAfterIssuerExpiryWithoutRevivingOldCapability) {
  Start(false, false);
  auto& branch = BranchAt(0);
  Open({&branch});
  std::vector<float> logits;
  ASSERT_TRUE(branch.Prefill(Prompt(9, 2), logits));
  rt::SnapshotStore store(*node_, 8 * kExtent);
  auto saved = branch.CaptureState(store);
  ASSERT_TRUE(saved) << Failed(saved).value_or("");
  const auto capacity = store.capacity();
  branch.InvalidateStateSnapshot();
  EXPECT_FALSE(store.Digest(*saved));
  EXPECT_FALSE(branch.RestoreState(store, *saved));
  auto current = branch.CaptureState(store);
  ASSERT_TRUE(current) << Failed(current).value_or("");
  EXPECT_EQ(store.capacity(), capacity);
  EXPECT_TRUE(store.Digest(*current));
  EXPECT_FALSE(store.Digest(*saved));
  Close();
}

TEST_F(EngineLlmTest, OpaqueSnapshotStoreAndCapabilityMayOutliveNodeAndBranch) {
  Start(false, false);
  auto& branch = BranchAt(0);
  Open({&branch});
  std::vector<float> logits;
  ASSERT_TRUE(branch.Prefill(Prompt(9, 2), logits));
  auto store = std::make_unique<rt::SnapshotStore>(*node_, 8 * kExtent);
  auto saved = branch.CaptureState(*store);
  ASSERT_TRUE(saved) << Failed(saved).value_or("");
  Close();
  const auto observer = node_->staging_lifetime().lock();
  ASSERT_TRUE(observer);
  Stop();  // node and branch retire while the non-owning capability remains
  EXPECT_FALSE(observer->live());
  EXPECT_FALSE(store->usable());
  EXPECT_FALSE(store->Digest(*saved));
  store.reset();  // must neither call the expired node nor reread freed bytes
}

TEST_F(EngineLlmTest, OpaqueSnapshotCeilingIncludesMetadataAndAllocatesNothingOnRefusal) {
  Start(false, false);
  auto& branch = BranchAt(0);
  Open({&branch});
  std::vector<float> logits;
  ASSERT_TRUE(branch.Prefill(Prompt(7, 4), logits));
  rt::SnapshotStore store(*node_, 1);
  const auto history = branch.history();
  auto refused = branch.CaptureState(store);
  EXPECT_FALSE(refused);
  EXPECT_EQ(store.capacity(), 0U);
  EXPECT_EQ(branch.history(), history);
  EXPECT_EQ(llm_->runner().slots().positions(0), history.size());
  Close();
}

// Prompts in chunks (state-only cuts, the last headed) and plain decoding,
// its greedy tokens chosen on the device or, with logits kept, by the host
// from each row; the next turn reuses the whole live history.
TEST_F(EngineLlmTest, PromptUnitsAndPlainGenerationFollowTheChain) {
  Start(/*drafters=*/false, /*speculate=*/false);
  EXPECT_EQ(llm_->KeptVersion(), kept::kStageFilesVersion);
  EXPECT_TRUE(llm_->KeptLayout().starts_with("llmp-state-v2:"));
  EXPECT_FALSE(llm_->speculative());
  const auto prompt = Prompt(150, 3);
  const Turned first = Turn(0, prompt, 20);
  EXPECT_EQ(first.reused, 0U);
  EXPECT_EQ(first.out.tokens, Chain(prompt, 20));
  EXPECT_GE(llm_->counts().greedy_tokens, 19U);
  EXPECT_EQ(BranchAt(0).history().size(), prompt.size() + 19);

  const auto next = Continue(0, first.out, Prompt(30, 7));
  const std::size_t history = BranchAt(0).history().size();
  const Turned second = Turn(0, next, 12, /*keep_logits=*/true);
  EXPECT_EQ(second.reused, history);
  EXPECT_EQ(second.out.tokens, Chain(next, 12));
  ASSERT_EQ(second.out.logits.size(), second.out.tokens.size());
  for (std::size_t i = 0; i < second.out.tokens.size(); ++i) {
    EXPECT_EQ(ArgmaxOf(second.out.logits[i]), second.out.tokens[i]) << i;
  }
  // Prefill was planned and run as steps of the runner.
  EXPECT_GT(llm_->runner().graph_stats().eager + llm_->runner().graph_stats().replayed, 0U);
}

// Two conversations joined: their prompts' chunks in joined prompt waves
// (with the next chunks as futures), their decode rows in generation waves;
// each follows its own chain.
TEST_F(EngineLlmTest, PromptAndGenerationWavesJoinTheSlots) {
  Start(false, false);
  EXPECT_EQ(llm_->prefill_wave_capacity(), kSlots);
  const std::array<std::vector<std::int32_t>, 2> prompts = {Prompt(128, 1), Prompt(128, 29)};
  rt::Llm::Branch& a = BranchAt(0);
  rt::Llm::Branch& b = BranchAt(1);
  Open({&a, &b});
  std::array<std::unique_ptr<rt::Llm::PromptSession>, 2> sessions;
  for (std::size_t i = 0; i < 2; ++i) {
    auto begun = BranchAt(i).BeginPrompt(prompts[i]);
    ASSERT_TRUE(begun.has_value()) << begun.error();
    sessions[i] = std::move(*begun);
    // Each one's reuse first, alone.
    while (!sessions[i]->done() && sessions[i]->NextUnit().has_value() &&
           sessions[i]->NextUnit()->phase != rt::Llm::PromptSession::Phase::kChunk) {
      auto advanced = sessions[i]->Advance();
      ASSERT_TRUE(advanced) << Why(advanced);
    }
  }
  const std::array<rt::PrefillGoOn, 2> go_on{};
  while (!sessions[0]->done() && !sessions[1]->done()) {
    const std::array<rt::Llm::PromptSession*, 2> both = {sessions[0].get(), sessions[1].get()};
    auto ran = llm_->RunPromptWave(both, go_on);
    ASSERT_TRUE(ran) << Why(ran);
  }
  EXPECT_TRUE(sessions[0]->done() && sessions[1]->done());
  EXPECT_GE(llm_->counts().joined_prefill_steps, 1U);
  std::array<std::vector<float>, 2> lasts = {sessions[0]->last(), sessions[1]->last()};
  for (auto& session : sessions) {
    auto finished = session->Finish();
    ASSERT_TRUE(finished) << Why(finished);
  }
  rt::GenerateOptions options;
  options.max_tokens = 16;
  std::array<rt::Generation, 2> outs;
  std::array<std::unique_ptr<rt::Llm::GenerationSession>, 2> generations;
  for (std::size_t i = 0; i < 2; ++i) {
    auto begun = BranchAt(i).BeginGeneration(lasts[i], options, outs[i]);
    ASSERT_TRUE(begun.has_value()) << begun.error();
    generations[i] = std::move(*begun);
  }
  for (int guard = 0; guard < 64; ++guard) {
    std::vector<rt::Llm::GenerationSession*> active;
    for (auto& g : generations) {
      if (!g->done()) active.push_back(g.get());
    }
    if (active.empty()) break;
    auto ran = llm_->RunGenerationWave(active);
    ASSERT_TRUE(ran) << Why(ran);
  }
  for (std::size_t i = 0; i < 2; ++i) {
    auto finished = generations[i]->Finish();
    ASSERT_TRUE(finished) << Why(finished);
    EXPECT_EQ(outs[i].tokens, Chain(prompts[i], 16)) << i;
  }
  Close();
  EXPECT_GE(llm_->counts().joined_waves, 1U);
}

// A joined prompt step whose owners' last chunks differ in rows (they step
// apart) builds both of those steps while it runs, so neither is planned
// on its own first run; the prompts still end on the chain.
TEST_F(EngineLlmTest, UnequalLastChunksAreBuiltAhead) {
  Start(false, false);
  const std::array<std::vector<std::int32_t>, 2> prompts = {Prompt(kChunk + 20, 3),
                                                            Prompt(kChunk + 40, 17)};
  rt::Llm::Branch& a = BranchAt(0);
  rt::Llm::Branch& b = BranchAt(1);
  Open({&a, &b});
  std::array<std::unique_ptr<rt::Llm::PromptSession>, 2> sessions;
  for (std::size_t i = 0; i < 2; ++i) {
    auto begun = BranchAt(i).BeginPrompt(prompts[i]);
    ASSERT_TRUE(begun.has_value()) << begun.error();
    sessions[i] = std::move(*begun);
    while (!sessions[i]->done() && sessions[i]->NextUnit().has_value() &&
           sessions[i]->NextUnit()->phase != rt::Llm::PromptSession::Phase::kChunk) {
      auto advanced = sessions[i]->Advance();
      ASSERT_TRUE(advanced) << Why(advanced);
    }
  }
  const std::uint64_t before = llm_->runner().lookahead_stats().built;
  // The first chunks joined; then each prompt's last chunk alone.
  const std::array<rt::Llm::PromptSession*, 2> both = {sessions[0].get(), sessions[1].get()};
  const std::array<rt::PrefillGoOn, 2> go_on{};
  auto ran = llm_->RunPromptWave(both, go_on);
  ASSERT_TRUE(ran) << Why(ran);
  EXPECT_EQ(llm_->counts().joined_prefill_steps, 1U);
  EXPECT_EQ(llm_->runner().lookahead_stats().built - before, 2U);
  const std::size_t built = source_->built().size();
  for (auto& session : sessions) {
    while (!session->done()) {
      auto advanced = session->Advance();
      ASSERT_TRUE(advanced) << Why(advanced);
    }
  }
  // Nothing more was built: both last chunks ran on plans built ahead.
  EXPECT_EQ(source_->built().size(), built);
  for (std::size_t i = 0; i < 2; ++i) {
    EXPECT_EQ(ArgmaxOf(sessions[i]->last()), Chain(prompts[i], 1).front()) << i;
    auto finished = sessions[i]->Finish();
    ASSERT_TRUE(finished) << Why(finished);
  }
  Close();
}

// A joined prompt wave ran with both owners' next chunks predicted (built,
// captured and their state prepared ahead); then the first owner's
// callback stops it. Its peer runs on alone and ends on its chain; the
// stopped owner keeps the prefix its completed chunk wrote (an abandoned
// prediction leaves no pending preparation), and its next turn reuses that
// prefix and follows its own chain.
TEST_F(EngineLlmTest, AStoppedOwnerKeepsItsCompletedPrefixAndItsPeerRunsOn) {
  Start(false, false);
  const std::array<std::vector<std::int32_t>, 2> prompts = {Prompt(3 * kChunk, 5),
                                                            Prompt(3 * kChunk, 23)};
  rt::Llm::Branch& a = BranchAt(0);
  rt::Llm::Branch& b = BranchAt(1);
  Open({&a, &b});
  std::array<std::unique_ptr<rt::Llm::PromptSession>, 2> sessions;
  for (std::size_t i = 0; i < 2; ++i) {
    auto begun = BranchAt(i).BeginPrompt(prompts[i]);
    ASSERT_TRUE(begun.has_value()) << begun.error();
    sessions[i] = std::move(*begun);
    while (!sessions[i]->done() && sessions[i]->NextUnit().has_value() &&
           sessions[i]->NextUnit()->phase != rt::Llm::PromptSession::Phase::kChunk) {
      auto advanced = sessions[i]->Advance();
      ASSERT_TRUE(advanced) << Why(advanced);
    }
  }
  const std::array<rt::Llm::PromptSession*, 2> both = {sessions[0].get(), sessions[1].get()};
  const std::array<rt::PrefillGoOn, 2> go_on{};
  auto ran = llm_->RunPromptWave(both, go_on);
  ASSERT_TRUE(ran) << Why(ran);
  EXPECT_EQ(llm_->counts().joined_prefill_steps, 1U);
  EXPECT_GE(llm_->runner().lookahead_stats().built, 1U);
  ASSERT_EQ(a.history().size(), kChunk);
  const std::array<rt::PrefillGoOn, 2> stop_first = {[](std::uint32_t) { return false; },
                                                     rt::PrefillGoOn{}};
  ran = llm_->RunPromptWave(both, stop_first);
  ASSERT_TRUE(ran) << Why(ran);
  EXPECT_TRUE(sessions[0]->done());
  EXPECT_FALSE(node_->has_pending_state_preparation());
  auto finished = sessions[0]->Finish();
  ASSERT_TRUE(finished) << Why(finished);
  EXPECT_EQ(a.history(),
            std::vector<std::int32_t>(prompts[0].begin(), prompts[0].begin() + kChunk));
  while (!sessions[1]->done()) {
    const std::array<rt::Llm::PromptSession*, 1> alone = {sessions[1].get()};
    const std::array<rt::PrefillGoOn, 1> on{};
    ran = llm_->RunPromptWave(alone, on);
    ASSERT_TRUE(ran) << Why(ran);
  }
  EXPECT_EQ(ArgmaxOf(sessions[1]->last()), Chain(prompts[1], 1).front());
  finished = sessions[1]->Finish();
  ASSERT_TRUE(finished) << Why(finished);
  EXPECT_EQ(b.history(), prompts[1]);
  EXPECT_FALSE(node_->has_pending_state_preparation());
  Close();
  const Turned next = Turn(0, prompts[0], 6);
  EXPECT_EQ(next.reused, kChunk);
  EXPECT_EQ(next.out.tokens, Chain(prompts[0], 6));
}

// Speculation with two drafters in a cascade: the MTP-like drafter's two
// proposals and the block's first follow the chain, the block's second
// does not, so a verify keeps four of its five rows; the tokens are the
// plain chain's all the same, alone and in a joined wave, with their rows.
TEST_F(EngineLlmTest, SpeculationWithTwoDraftersKeepsThePlainChain) {
  Start(/*drafters=*/true, /*speculate=*/true);
  EXPECT_TRUE(llm_->speculative());
  const auto prompt = Prompt(200, 5);
  const Turned first = Turn(0, prompt, 40);
  EXPECT_EQ(first.out.tokens, Chain(prompt, 40));
  EXPECT_GT(first.out.drafted, 0U);
  EXPECT_GT(first.out.accepted, 0U);
  EXPECT_LT(first.out.accepted, first.out.drafted);  // the block's fill rows miss
  EXPECT_GT(llm_->counts().speculative_units, 0U);

  const auto other = Prompt(90, 17);
  const Turned rows = Turn(1, other, 25, /*keep_logits=*/true);
  EXPECT_EQ(rows.out.tokens, Chain(other, 25));
  ASSERT_EQ(rows.out.logits.size(), rows.out.tokens.size());
  for (std::size_t i = 0; i < rows.out.tokens.size(); ++i) {
    EXPECT_EQ(ArgmaxOf(rows.out.logits[i]), rows.out.tokens[i]) << i;
  }

  // Both continued in speculative waves.
  const auto next0 = Continue(0, first.out, Prompt(10, 2));
  const auto next1 = Continue(1, rows.out, Prompt(12, 4));
  rt::Llm::Branch& a = BranchAt(0);
  rt::Llm::Branch& b = BranchAt(1);
  Open({&a, &b});
  std::array<std::vector<float>, 2> lasts;
  for (std::size_t i = 0; i < 2; ++i) {
    std::uint32_t reused = 0;
    auto prepared = BranchAt(i).PreparePrompt(i == 0 ? next0 : next1, 0, lasts[i], reused);
    ASSERT_TRUE(prepared) << Why(prepared);
    EXPECT_GT(reused, 0U);
  }
  rt::GenerateOptions options;
  options.max_tokens = 20;
  std::array<rt::Generation, 2> outs;
  std::array<std::unique_ptr<rt::Llm::GenerationSession>, 2> generations;
  for (std::size_t i = 0; i < 2; ++i) {
    auto begun = BranchAt(i).BeginGeneration(lasts[i], options, outs[i]);
    ASSERT_TRUE(begun.has_value()) << begun.error();
    generations[i] = std::move(*begun);
  }
  const std::uint64_t waves = llm_->counts().joined_waves;
  for (int guard = 0; guard < 64; ++guard) {
    std::vector<rt::Llm::GenerationSession*> active;
    for (auto& g : generations) {
      if (!g->done()) active.push_back(g.get());
    }
    if (active.empty()) break;
    auto ran = llm_->RunGenerationWave(active);
    ASSERT_TRUE(ran) << Why(ran);
  }
  for (std::size_t i = 0; i < 2; ++i) {
    auto finished = generations[i]->Finish();
    ASSERT_TRUE(finished) << Why(finished);
  }
  Close();
  EXPECT_EQ(outs[0].tokens, Chain(next0, 20));
  EXPECT_EQ(outs[1].tokens, Chain(next1, 20));
  EXPECT_GT(llm_->counts().joined_waves, waves);
}

// An idle conversation spilled (every stage's file) and restored by its
// next turn's first unit: the turn reuses the whole history and continues
// the chain.
TEST_F(EngineLlmTest, ASpilledConversationComesBackExactly) {
  Start(true, true);
  const auto prompt = Prompt(180, 9);
  const Turned first = Turn(0, prompt, 12);
  EXPECT_EQ(first.out.tokens, Chain(prompt, 12));
  rt::Llm::Branch& branch = BranchAt(0);
  const std::size_t history = branch.history().size();
  auto spilled = llm_->SpillIdle(branch);
  ASSERT_TRUE(spilled) << Why(spilled);
  EXPECT_TRUE(branch.spilled());
  EXPECT_GT(llm_->SpilledStateBytes(branch), 0U);
  EXPECT_EQ(llm_->ResidentStateBytes(branch), 0U);
  const auto next = Continue(0, first.out, Prompt(20, 1));
  const Turned second = Turn(0, next, 15);
  EXPECT_EQ(second.reused, history);
  EXPECT_EQ(second.out.tokens, Chain(next, 15));
  EXPECT_FALSE(branch.spilled());
  EXPECT_EQ(llm_->spill_stats().spills, 1U);
  EXPECT_EQ(llm_->spill_stats().restores, 1U);
}

// A turn checkpoint at the renderer's boundary, restored when a later
// prompt diverges after it: the lifecycle's three-step restore with its
// positions and footprint, every stage's pages.
TEST_F(EngineLlmTest, ATurnCheckpointRestoresItsPrefix) {
  Start(true, true);
  const auto prompt = Prompt(300, 4);
  const Turned first = Turn(0, prompt, 8, false, /*boundary=*/200);
  EXPECT_EQ(first.out.tokens, Chain(prompt, 8));
  ASSERT_EQ(BranchAt(0).turn_checkpoints(), 1U);
  std::vector<std::int32_t> diverged(prompt.begin(), prompt.begin() + 250);
  const auto tail = Prompt(30, 11);
  diverged.insert(diverged.end(), tail.begin(), tail.end());
  const Turned second = Turn(0, diverged, 10);
  EXPECT_EQ(second.reused, 200U);
  EXPECT_EQ(second.out.tokens, Chain(diverged, 10));
}

// The speed gate's diagnostic (ServingOptions::diagnostic_skip_turn_
// checkpoints): the turn saves no checkpoint and is otherwise the same (its
// tokens and the steps it ran); a later divergent prompt then has none to
// restore and prefills, with the same reply.
TEST_F(EngineLlmTest, TheGateDiagnosticSkipsOnlyTheTurnCheckpointSave) {
  const auto prompt = Prompt(300, 4);
  std::vector<std::int32_t> diverged(prompt.begin(), prompt.begin() + 250);
  const auto tail = Prompt(30, 11);
  diverged.insert(diverged.end(), tail.begin(), tail.end());
  struct Seen {
    std::vector<std::int32_t> tokens;
    std::uint64_t queued = 0, executed = 0;
    std::size_t checkpoints = 0;
    std::uint32_t reused = 0;
  };
  std::array<Seen, 2> seen;
  for (const bool save : {true, false}) {
    save_checkpoints_ = save;
    Start(false, false);
    Seen& s = seen.at(save ? 0 : 1);
    const Turned first = Turn(0, prompt, 8, false, /*boundary=*/200);
    s.tokens = first.out.tokens;
    s.queued = source_->queued();
    s.executed = source_->executed();
    s.checkpoints = BranchAt(0).turn_checkpoints();
    const Turned second = Turn(0, diverged, 10);
    EXPECT_EQ(second.out.tokens, Chain(diverged, 10));
    s.reused = second.reused;
    Stop();
  }
  EXPECT_EQ(seen[0].tokens, Chain(prompt, 8));
  EXPECT_EQ(seen[1].tokens, seen[0].tokens);
  EXPECT_EQ(seen[1].queued, seen[0].queued);
  EXPECT_EQ(seen[1].executed, seen[0].executed);
  EXPECT_EQ(seen[0].checkpoints, 1U);
  EXPECT_EQ(seen[1].checkpoints, 0U);
  EXPECT_EQ(seen[0].reused, 200U);
  EXPECT_EQ(seen[1].reused, 0U);
}

// D-102's rung 2: a step that failed after it queued its work faults the
// cohort; recovery discards what the failure may have touched (the resident
// conversation, forgotten) and keeps the spilled one, which continues.
TEST_F(EngineLlmTest, RecoveryKeepsTheSpilledConversationAndForgetsTheRest) {
  Start(false, false);
  const auto kept_prompt = Prompt(100, 2);
  const Turned kept_turn = Turn(1, kept_prompt, 6);
  rt::Llm::Branch& spilled = BranchAt(1);
  ASSERT_TRUE(llm_->SpillIdle(spilled));
  const std::size_t history = spilled.history().size();

  rt::Llm::Branch& failing = BranchAt(0);
  Open({&failing});
  std::vector<float> last;
  std::uint32_t reused = 0;
  ASSERT_TRUE(failing.PreparePrompt(Prompt(70, 3), 0, last, reused));
  source_->options().fail_run = kn::StepError::kUnknown;
  rt::GenerateOptions options;
  options.max_tokens = 4;
  rt::Generation out;
  EXPECT_FALSE(failing.Generate(last, options, out));
  source_->options().fail_run.reset();
  std::ignore = node_->EndRequest(0);
  // The stream fenced, as the server's recovery does first.
  ASSERT_TRUE(node_->Job(
      llmp::catalog::Closure{},
      [](llmp::providers::NativeStream) { return llmp::scheduler::JobResult::kQueued; },
      "fencing the test's stream", 0));
  auto recovered = llm_->RecoverInPlace();
  ASSERT_TRUE(recovered) << Why(recovered);
  EXPECT_TRUE(failing.history().empty());
  EXPECT_TRUE(spilled.spilled());
  EXPECT_EQ(spilled.history().size(), history);

  const auto next = Continue(1, kept_turn.out, Prompt(9, 8));
  const Turned resumed = Turn(1, next, 7);
  EXPECT_EQ(resumed.reused, history);
  EXPECT_EQ(resumed.out.tokens, Chain(next, 7));
  const auto fresh = Prompt(40, 21);
  const Turned again = Turn(0, fresh, 5);
  EXPECT_EQ(again.out.tokens, Chain(fresh, 5));
}

// Cold repeat-pending metadata is restorable but cannot execute until its
// exact disk-backed progress record and initialized state have hydrated.
TEST_F(EngineLlmTest, AdoptedRepeatPendingHydratesWithoutRecomputingItsPrefix) {
  kept_progress_ = true;
  native_outputs_ = true;
  save_checkpoints_ = false;
  const fs::path directory = root_ / "repeat-kept";
  auto opened = pf::OpenPrivateDirectory(-1, directory.c_str());
  ASSERT_TRUE(opened.has_value());
  const int dir = *opened;
  kept::Identity identity{.build = "test build", .artifact = "abcdef", .drafter = "", .layout = ""};
  std::vector<std::int32_t> history;
  std::vector<std::byte> image;
  llmp::base::Sha256Digest digest{};
  {
    rt::StateKeeper keeper(1, {});
    const auto model = keeper.AddModel(::dup(dir));
    Start(true, true, [&](rt::EngineLlm& llm) {
      identity.layout = llm.KeptLayout();
      llm.set_kept(
          {.keeper = &keeper, .model = model, .directory = dir, .identity = identity, .adopt = {}});
    });
    const auto turn = Turn(0, Prompt(17, 6), 8);
    ASSERT_EQ(turn.out.tokens.size(), 8U);
    ASSERT_GT(turn.out.drafted, 0U);
    auto& branch = BranchAt(0);
    Open({&branch});
    {
      rt::SnapshotStore store(*node_, 8 * kExtent);
      auto captured = branch.CaptureState(store);
      ASSERT_TRUE(captured) << Failed(captured).value_or("");
      auto value = store.Digest(*captured);
      ASSERT_TRUE(value) << Failed(value).value_or("");
      digest = *value;
      image = NativeImage(0);
    }
    Close();
    history = branch.history();
    ASSERT_TRUE(llm_->SpillIdle(branch));
    ASSERT_TRUE(keeper.Drain(rt::Clock::now() + std::chrono::seconds(30)));
    ASSERT_TRUE(keeper.Kept(model, 0));
    llm_->PreserveKeptFiles();
    Stop();
  }
  auto text = pf::ReadPrivateFile(dir, "slot-0.record", kept::kMostRecordBytes);
  ASSERT_TRUE(text.has_value());
  auto record = kept::Decode(*text);
  ASSERT_TRUE(record) << Failed(record).value_or("");
  ASSERT_EQ(rt::VerifyKeptState(dir, *record), "");
  rt::StateKeeper keeper(1, {});
  const auto model = keeper.AddModel(::dup(dir));
  Start(true, true, [&](rt::EngineLlm& llm) {
    EXPECT_EQ(llm.KeptLayout(), identity.layout);
    std::vector<kept::FileId> ids;
    for (const auto& file : record->files) ids.push_back(file.id);
    llm.set_kept({.keeper = &keeper,
                  .model = model,
                  .directory = dir,
                  .identity = identity,
                  .adopt = {0},
                  .files = {ids}});
  });
  auto& branch = BranchAt(0);
  rt::MemoryCharge charge;
  ASSERT_TRUE(llm_->Adopt(branch, *record, charge));
  EXPECT_EQ(branch.history(), history);
  EXPECT_TRUE(llm_->runner().slots().restorable(0));
  EXPECT_FALSE(llm_->runner().slots().usable(0));
  EXPECT_FALSE(llm_->runner().slots().CheckActive(0));
  const auto executed = source_->executed();
  Open({&branch});
  // Leave room for the target page alone. A refused second-stage hydrate
  // must retain its cold record and history, then resume after room returns.
  const auto occupied = node_->catalog().OccupancyOf(node_->domain()).Total().value();
  ASSERT_GT(node_->budget().value(), occupied + kExtent);
  std::vector<llmp::catalog::ExtentId> staging;
  auto pad = node_->Pinned(node_->budget().value() - occupied - kExtent, 0, staging);
  ASSERT_TRUE(pad) << Failed(pad).value_or("");
  auto opened_prompt = branch.BeginPrompt(history, 0, false, true);
  ASSERT_TRUE(opened_prompt) << Failed(opened_prompt).value_or("");
  auto prompt = std::move(*opened_prompt);
  const auto first = prompt->NextUnit();
  ASSERT_TRUE(first) << Failed(first).value_or("");
  ASSERT_EQ(first->phase, rt::Llm::PromptSession::Phase::kReuse);
  EXPECT_FALSE(prompt->Advance({}, true));
  EXPECT_TRUE(prompt->refused());
  EXPECT_FALSE(prompt->done());
  EXPECT_EQ(branch.history(), history);
  EXPECT_EQ(source_->executed(), executed);
  EXPECT_TRUE(llm_->runner().slots().state_refused(0));
  EXPECT_TRUE(llm_->runner().slots().restorable(0));
  EXPECT_FALSE(llm_->runner().slots().usable(0));
  EXPECT_FALSE(llm_->runner().slots().live(0, 0).used_ranges().empty());
  EXPECT_TRUE(llm_->runner().slots().live(0, 1).used_ranges().empty());
  ASSERT_TRUE(node_->FreePinned(*pad));
  ASSERT_TRUE(prompt->Advance({}, true));
  ASSERT_TRUE(prompt->done()) << "a kept prefix must not fall back to model recomputation";
  EXPECT_EQ(prompt->reused(), history.size());
  ASSERT_TRUE(prompt->Finish());
  prompt.reset();
  EXPECT_EQ(source_->executed(), executed);
  EXPECT_FALSE(branch.spilled());
  EXPECT_FALSE(llm_->runner().slots().restorable(0));
  EXPECT_TRUE(llm_->runner().slots().usable(0));
  EXPECT_EQ(NativeImage(0), image);
  {
    rt::SnapshotStore store(*node_, 8 * kExtent);
    auto captured = branch.CaptureState(store);
    ASSERT_TRUE(captured) << Failed(captured).value_or("");
    EXPECT_EQ(store.Digest(*captured), digest);
  }
  EXPECT_EQ(source_->executed(), executed);
  Close();
  Stop();
  ::close(dir);
}

// Kept across a restart (D-105) with a spill file a stage (record version
// 2): the first process spills a conversation with a turn checkpoint and
// its record is written; the second checks the record and every stage's
// file, adopts it, and its next turn restores the state from the files and
// continues the chain. A record of version 1 for the other slot is refused
// and removed.
TEST_F(EngineLlmTest, AConversationIsKeptAcrossARestartInStageFiles) {
  const fs::path directory = root_ / "conversations";
  auto opened = pf::OpenPrivateDirectory(-1, directory.c_str());
  ASSERT_TRUE(opened.has_value());
  const int dir = *opened;
  kept::Identity identity{
      .build = "a test build", .artifact = "abcdef", .drafter = "", .layout = ""};
  std::vector<std::int32_t> history;
  rt::Generation reply;
  std::string layout;
  {
    rt::StateKeeper keeper(1, {});
    const std::size_t model = keeper.AddModel(::dup(dir));
    Start(true, true, [&](rt::EngineLlm& llm) {
      layout = llm.KeptLayout();
      identity.layout = layout;
      llm.set_kept(
          {.keeper = &keeper, .model = model, .directory = dir, .identity = identity, .adopt = {}});
    });
    const auto prompt = Prompt(220, 6);
    const Turned first = Turn(0, prompt, 10, false, /*boundary=*/150);
    EXPECT_EQ(first.out.tokens, Chain(prompt, 10));
    reply = first.out;
    rt::Llm::Branch& branch = BranchAt(0);
    ASSERT_TRUE(llm_->SpillIdle(branch));
    history = branch.history();
    ASSERT_TRUE(keeper.Drain(std::chrono::steady_clock::now() + std::chrono::seconds(30)));
    EXPECT_TRUE(keeper.Kept(model, 0));
    llm_->PreserveKeptFiles();
    Stop();
  }
  for (const char* name :
       {"slot-0.state", "slot-0.stage-1.state", "slot-0.stage-2.state", "slot-0.record"}) {
    EXPECT_TRUE(fs::exists(directory / name)) << name;
  }
  auto text = pf::ReadPrivateFile(dir, "slot-0.record", kept::kMostRecordBytes);
  ASSERT_TRUE(text.has_value());
  auto record = kept::Decode(*text);
  ASSERT_TRUE(record.has_value()) << record.error();
  EXPECT_EQ(record->version, kept::kStageFilesVersion);
  ASSERT_EQ(record->files.size(), 3U);
  EXPECT_EQ(record->files[2].name, "slot-0.stage-2.state");
  EXPECT_EQ(record->tokens, history);
  EXPECT_EQ(record->checkpoints.size(), 1U);

  // The next process: the record checked against what it would write, its
  // files hashed, then adopted.
  rt::StateKeeper keeper(1, {});
  const std::size_t model = keeper.AddModel(::dup(dir));
  kept::Record old;
  Start(true, true, [&](rt::EngineLlm& llm) {
    EXPECT_EQ(llm.KeptLayout(), layout);
    kept::Expected expected{.version = llm.KeptVersion(),
                            .identity = identity,
                            .slot = 0,
                            .regions = llm.KeptRegions(),
                            .layouts = llm.KeptLayouts(),
                            .context = llm.usable_context(),
                            .vocabulary = kWidth,
                            .now_unix_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                               std::chrono::system_clock::now().time_since_epoch())
                                               .count(),
                            .retention_ms = 24LL * 3600 * 1000};
    auto checked = kept::Check(*record, expected);
    ASSERT_TRUE(checked.has_value()) << checked.error();
    EXPECT_EQ(rt::VerifyKeptState(dir, *record), "");
    for (const kept::Checkpoint& c : record->checkpoints) {
      EXPECT_EQ(
          rt::VerifyKeptFile(dir, c.file, c.id, c.file_bytes, kept::CheckpointPlaces(c), c.digests),
          "");
    }
    EXPECT_EQ(rt::EmptyUnlisted(dir, *record), "");
    // The other slot's record of the old version: decoded, refused.
    old = *record;
    old.version = kept::kOneFileVersion;
    old.slot = 1;
    old.files = {{.name = kept::StateFileName(1), .id = record->files[0].id, .bytes = 0}};
    old.checkpoints.clear();
    for (const std::uint64_t bytes : old.regions) old.files[0].bytes += bytes;
    expected.slot = 1;
    auto refused = kept::Check(old, expected);
    ASSERT_FALSE(refused.has_value());
    EXPECT_NE(refused.error().find("format version 1"), std::string::npos) << refused.error();
    std::vector<kept::FileId> ids;
    for (const kept::StateFile& file : record->files) ids.push_back(file.id);
    llm.set_kept({.keeper = &keeper,
                  .model = model,
                  .directory = dir,
                  .identity = identity,
                  .adopt = {0},
                  .files = {ids}});
  });
  rt::MemoryCharge charge;
  rt::Llm::Branch& branch = BranchAt(0);
  auto adopted = llm_->Adopt(branch, *record, charge);
  ASSERT_TRUE(adopted) << Why(adopted);
  EXPECT_EQ(branch.history(), history);
  EXPECT_TRUE(branch.spilled());
  EXPECT_EQ(branch.turn_checkpoints(), 1U);
  EXPECT_TRUE(keeper.Kept(model, 0));

  // The old record, written where the other slot's would be: refused at
  // adoption too, and removed.
  ASSERT_TRUE(pf::ReplacePrivateFile(dir, kept::RecordFileName(1).c_str(), kept::Encode(old)));
  auto old_text = pf::ReadPrivateFile(dir, kept::RecordFileName(1).c_str(), kept::kMostRecordBytes);
  ASSERT_TRUE(old_text.has_value());
  auto old_record = kept::Decode(*old_text);
  ASSERT_TRUE(old_record.has_value()) << old_record.error();
  rt::MemoryCharge old_charge;
  auto old_adopted = llm_->Adopt(BranchAt(1), *old_record, old_charge);
  ASSERT_FALSE(old_adopted);
  EXPECT_NE(Why(old_adopted).find("format version"), std::string::npos) << Why(old_adopted);
  EXPECT_FALSE(fs::exists(directory / kept::RecordFileName(1)));
  EXPECT_TRUE(BranchAt(1).history().empty());
  // Only that slot's record went: the adopted slot's record and stage files
  // stay as they were.
  for (const char* name :
       {"slot-0.state", "slot-0.stage-1.state", "slot-0.stage-2.state", "slot-0.record"}) {
    EXPECT_TRUE(fs::exists(directory / name)) << name;
  }
  EXPECT_TRUE(keeper.Kept(model, 0));

  // Its next turn restores every stage from its files and continues.
  std::vector<std::int32_t> next = history;
  next.push_back(reply.tokens.back());
  const auto more = Prompt(15, 2);
  next.insert(next.end(), more.begin(), more.end());
  const Turned resumed = Turn(0, next, 12);
  EXPECT_EQ(resumed.reused, history.size());
  EXPECT_EQ(resumed.out.tokens, Chain(next, 12));
  EXPECT_FALSE(keeper.Kept(model, 0));  // live again: its record went
  Stop();
  (void)::close(dir);
}

// A file put under a kept name after the start checked it (between
// VerifyKeptState and the reopens that adopt it) is not the one its record
// names: a stage's spill file found so refuses the slot's adoption and a
// turn checkpoint found so is left out, and the file reached through the
// name is never adopted, written nor emptied (D-105 as amended).
TEST_F(EngineLlmTest, AFileSwappedInAfterTheChecksIsNeverAdoptedNorWritten) {
  const fs::path directory = root_ / "conversations";
  auto opened = pf::OpenPrivateDirectory(-1, directory.c_str());
  ASSERT_TRUE(opened.has_value());
  const int dir = *opened;
  kept::Identity identity{
      .build = "a test build", .artifact = "abcdef", .drafter = "", .layout = ""};
  {
    rt::StateKeeper keeper(1, {});
    const std::size_t model = keeper.AddModel(::dup(dir));
    Start(true, true, [&](rt::EngineLlm& llm) {
      identity.layout = llm.KeptLayout();
      llm.set_kept(
          {.keeper = &keeper, .model = model, .directory = dir, .identity = identity, .adopt = {}});
    });
    for (const std::size_t slot : {std::size_t{0}, std::size_t{1}}) {
      const auto prompt = Prompt(220, 6 + static_cast<std::int32_t>(slot));
      (void)Turn(slot, prompt, 10, false, /*boundary=*/150);
      ASSERT_TRUE(llm_->SpillIdle(BranchAt(slot)));
    }
    ASSERT_TRUE(keeper.Drain(std::chrono::steady_clock::now() + std::chrono::seconds(30)));
    llm_->PreserveKeptFiles();
    Stop();
  }
  std::vector<kept::Record> records;
  for (const std::uint32_t slot : {0U, 1U}) {
    auto text =
        pf::ReadPrivateFile(dir, kept::RecordFileName(slot).c_str(), kept::kMostRecordBytes);
    ASSERT_TRUE(text.has_value()) << slot;
    auto record = kept::Decode(*text);
    ASSERT_TRUE(record.has_value()) << record.error();
    ASSERT_EQ(record->checkpoints.size(), 1U);
    // The start's checks pass: the files are the ones named.
    EXPECT_EQ(rt::VerifyKeptState(dir, *record), "");
    records.push_back(std::move(*record));
  }
  // Then each of two names becomes a link to a copy: the same bytes and
  // size, another file.
  const auto contents = [](const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  };
  const auto swap = [&](const std::string& name) {
    const fs::path foreign = root_ / ("foreign-" + name);
    fs::copy_file(directory / name, foreign);
    fs::remove(directory / name);
    fs::create_symlink(foreign, directory / name);
    return foreign;
  };
  const fs::path stage = swap("slot-0.stage-1.state");
  const fs::path checkpoint = swap(records[1].checkpoints[0].file);
  const std::string stage_bytes = contents(stage);
  const std::string checkpoint_bytes = contents(checkpoint);
  ASSERT_FALSE(stage_bytes.empty());
  ASSERT_FALSE(checkpoint_bytes.empty());

  // The next process reopens the files the start checked (PrepareKept
  // passes their identities).
  std::vector<std::vector<kept::FileId>> files;
  for (const kept::Record& record : records) {
    files.emplace_back();
    for (const kept::StateFile& file : record.files) files.back().push_back(file.id);
  }
  rt::StateKeeper keeper(1, {});
  const std::size_t model = keeper.AddModel(::dup(dir));
  Start(true, true, [&](rt::EngineLlm& llm) {
    llm.set_kept({.keeper = &keeper,
                  .model = model,
                  .directory = dir,
                  .identity = identity,
                  .adopt = {0, 1},
                  .files = files});
  });
  rt::MemoryCharge first_charge;
  auto first = llm_->Adopt(BranchAt(0), records[0], first_charge);
  EXPECT_FALSE(first);
  if (!first) {
    EXPECT_NE(Why(first).find("not the one its record names"), std::string::npos) << Why(first);
    EXPECT_TRUE(BranchAt(0).ReleaseIdleState());  // as the server does with one refused
  }
  EXPECT_FALSE(fs::is_symlink(directory / "slot-0.stage-1.state"));
  rt::MemoryCharge second_charge;
  auto second = llm_->Adopt(BranchAt(1), records[1], second_charge);
  ASSERT_TRUE(second) << Why(second);
  EXPECT_EQ(BranchAt(1).turn_checkpoints(), 0U);
  // Both slots used and spilled again: neither file reached through a
  // swapped name changes.
  for (const std::size_t slot : {std::size_t{0}, std::size_t{1}}) {
    const auto prompt = Prompt(90, 30 + static_cast<std::int32_t>(slot));
    const Turned turned = Turn(slot, prompt, 6);
    EXPECT_EQ(turned.out.tokens, Chain(prompt, 6));
    ASSERT_TRUE(llm_->SpillIdle(BranchAt(slot)));
  }
  ASSERT_TRUE(keeper.Drain(std::chrono::steady_clock::now() + std::chrono::seconds(30)));
  EXPECT_TRUE(contents(stage) == stage_bytes);
  EXPECT_TRUE(contents(checkpoint) == checkpoint_bytes);
  Stop();
  (void)::close(dir);
}

// A slot set to be adopted without its record's file identities keeps
// none of its files: it gets new empty ones, so a file put under a name
// since the process before (here a link to a copy of the same size) is
// never used, nor emptied when the adoption is refused.
TEST_F(EngineLlmTest, AKeptSlotWithoutItsFileIdentitiesStartsFresh) {
  const fs::path directory = root_ / "conversations";
  auto opened = pf::OpenPrivateDirectory(-1, directory.c_str());
  ASSERT_TRUE(opened.has_value());
  const int dir = *opened;
  kept::Identity identity{
      .build = "a test build", .artifact = "abcdef", .drafter = "", .layout = ""};
  {
    rt::StateKeeper keeper(1, {});
    const std::size_t model = keeper.AddModel(::dup(dir));
    Start(true, true, [&](rt::EngineLlm& llm) {
      identity.layout = llm.KeptLayout();
      llm.set_kept(
          {.keeper = &keeper, .model = model, .directory = dir, .identity = identity, .adopt = {}});
    });
    (void)Turn(0, Prompt(220, 6), 10, false);
    ASSERT_TRUE(llm_->SpillIdle(BranchAt(0)));
    ASSERT_TRUE(keeper.Drain(std::chrono::steady_clock::now() + std::chrono::seconds(30)));
    llm_->PreserveKeptFiles();
    Stop();
  }
  auto text = pf::ReadPrivateFile(dir, kept::RecordFileName(0).c_str(), kept::kMostRecordBytes);
  ASSERT_TRUE(text.has_value());
  auto record = kept::Decode(*text);
  ASSERT_TRUE(record.has_value()) << record.error();
  const std::string name = kept::StateFileName(0, 0);
  const fs::path foreign = root_ / "foreign";
  fs::copy_file(directory / name, foreign);
  fs::remove(directory / name);
  fs::create_symlink(foreign, directory / name);
  const auto contents = [](const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  };
  const std::string bytes = contents(foreign);
  ASSERT_FALSE(bytes.empty());
  ASSERT_NE(bytes.find_first_not_of('\0'), std::string::npos);

  rt::StateKeeper keeper(1, {});
  const std::size_t model = keeper.AddModel(::dup(dir));
  Start(true, true, [&](rt::EngineLlm& llm) {
    llm.set_kept(
        {.keeper = &keeper, .model = model, .directory = dir, .identity = identity, .adopt = {0}});
  });
  rt::MemoryCharge charge;
  auto adopted = llm_->Adopt(BranchAt(0), *record, charge);
  ASSERT_FALSE(adopted);
  EXPECT_NE(Why(adopted).find("not the one its record names"), std::string::npos) << Why(adopted);
  EXPECT_TRUE(BranchAt(0).ReleaseIdleState());  // as the server does with one refused
  EXPECT_FALSE(fs::is_symlink(directory / name));
  EXPECT_TRUE(contents(foreign) == bytes);
  Stop();
  (void)::close(dir);
}

// A model's kept directory made unwritable after the start's checks, and
// an adopted slot's name made a link to nothing meanwhile: the model still
// registers, each slot spilling to an unnamed file (said, and kept for
// this run only), the adoption is refused and nothing in the directory is
// touched; the slots serve as fresh ones.
TEST_F(EngineLlmTest, ADirectoryUnwritableAfterTheChecksKeepsNothingAndStarts) {
  const fs::path directory = root_ / "conversations";
  auto opened = pf::OpenPrivateDirectory(-1, directory.c_str());
  ASSERT_TRUE(opened.has_value());
  const int dir = *opened;
  kept::Identity identity{
      .build = "a test build", .artifact = "abcdef", .drafter = "", .layout = ""};
  {
    rt::StateKeeper keeper(1, {});
    const std::size_t model = keeper.AddModel(::dup(dir));
    Start(true, true, [&](rt::EngineLlm& llm) {
      identity.layout = llm.KeptLayout();
      llm.set_kept(
          {.keeper = &keeper, .model = model, .directory = dir, .identity = identity, .adopt = {}});
    });
    (void)Turn(0, Prompt(220, 6), 10, false);
    ASSERT_TRUE(llm_->SpillIdle(BranchAt(0)));
    ASSERT_TRUE(keeper.Drain(std::chrono::steady_clock::now() + std::chrono::seconds(30)));
    llm_->PreserveKeptFiles();
    Stop();
  }
  auto text = pf::ReadPrivateFile(dir, kept::RecordFileName(0).c_str(), kept::kMostRecordBytes);
  ASSERT_TRUE(text.has_value());
  auto record = kept::Decode(*text);
  ASSERT_TRUE(record.has_value()) << record.error();
  EXPECT_EQ(rt::VerifyKeptState(dir, *record), "");
  std::vector<kept::FileId> ids;
  for (const kept::StateFile& file : record->files) ids.push_back(file.id);
  // After the checks: the stage-0 name a link to nothing, the directory
  // closed to writes.
  const std::string name = kept::StateFileName(0, 0);
  fs::remove(directory / name);
  fs::create_symlink(root_ / "nothing", directory / name);
  const auto listing = [&] {
    std::vector<std::string> names;
    for (const auto& e : fs::directory_iterator(directory)) {
      names.push_back(e.path().filename().string());
    }
    std::ranges::sort(names);
    return names;
  };
  const std::vector<std::string> before = listing();

  // Writable again however the test ends, so TearDown removes it.
  struct Reopen {
    fs::path path;
    explicit Reopen(fs::path p) : path(std::move(p)) {}
    Reopen(const Reopen&) = delete;
    Reopen& operator=(const Reopen&) = delete;
    Reopen(Reopen&&) = delete;
    Reopen& operator=(Reopen&&) = delete;
    ~Reopen() {
      std::error_code ignored;
      fs::permissions(path, fs::perms::owner_all, fs::perm_options::replace, ignored);
    }
  } const reopen(directory);
  rt::StateKeeper keeper(1, {});
  const std::size_t model = keeper.AddModel(::dup(dir));
  Start(true, true, [&](rt::EngineLlm& llm) {
    llm.set_kept({.keeper = &keeper,
                  .model = model,
                  .directory = dir,
                  .identity = identity,
                  .adopt = {0},
                  .files = {ids}});
    fs::permissions(directory, fs::perms::owner_read | fs::perms::owner_exec,
                    fs::perm_options::replace);
  });
  const std::vector<std::string> unkept = llm_->UnkeptSlots();
  ASSERT_EQ(unkept.size(), std::size_t{kSlots});
  EXPECT_NE(unkept[0].find("slot 0: the spill file"), std::string::npos) << unkept[0];
  rt::MemoryCharge charge;
  auto adopted = llm_->Adopt(BranchAt(0), *record, charge);
  ASSERT_FALSE(adopted);
  EXPECT_NE(Why(adopted).find("not the one its record names"), std::string::npos) << Why(adopted);
  EXPECT_TRUE(BranchAt(0).ReleaseIdleState());  // as the server does with one refused
  for (const std::size_t slot : {std::size_t{0}, std::size_t{1}}) {
    const auto prompt = Prompt(90, 30 + static_cast<std::int32_t>(slot));
    const Turned turned = Turn(slot, prompt, 6);
    EXPECT_EQ(turned.out.tokens, Chain(prompt, 6));
    ASSERT_TRUE(llm_->SpillIdle(BranchAt(slot)));
  }
  ASSERT_TRUE(keeper.Drain(std::chrono::steady_clock::now() + std::chrono::seconds(30)));
  fs::permissions(directory, fs::perms::owner_all, fs::perm_options::replace);
  EXPECT_EQ(listing(), before);
  EXPECT_TRUE(fs::is_symlink(directory / name));
  Stop();
  (void)::close(dir);
}

// An adapter without its parts, or whose parts are refused, refuses Setup
// with its reason; nothing else runs.
TEST_F(EngineLlmTest, SetupRefusesWhatItCannotBuild) {
  en::PagedNode node(
      en::NodeSettings{.compute_streams = 1, .slots = 2, .copy_lane = true, .slot_bytes = kExtent});
  ASSERT_TRUE(node.Open());
  rt::EngineLlm unbuilt(node, entry_, Settings(false), roles_, 0, {});
  EXPECT_FALSE(unbuilt.Setup());
  rt::EngineLlm refused(
      node, entry_, Settings(false), roles_, 0,
      [](const rt::ModelSettings&) -> std::expected<rt::EngineLlmParts, std::string> {
        return std::unexpected(std::string("no step source yet"));
      });
  auto set = refused.Setup();
  ASSERT_FALSE(set);
  EXPECT_NE(Why(set).find("no step source yet"), std::string::npos);
  EXPECT_FALSE(refused.SelectBranches({}));
  // Speculation on with parts that have no drafter: refused, never plain
  // in its place (M3's classes refused it too).
  rt::EngineLlm plain(node, entry_, Settings(true), roles_, 0, Build(false));
  auto speculating = plain.Setup();
  ASSERT_FALSE(speculating);
  EXPECT_NE(Why(speculating).find("speculation is on but its parts have no drafter"),
            std::string::npos);
  // No runner: a server stopping after this failed start leaves it out of
  // the node's teardown (Server::TearDown) instead of asking for it.
  EXPECT_FALSE(unbuilt.has_paged());
  EXPECT_FALSE(refused.has_paged());
}

TEST_F(EngineLlmTest, CompleteDraftScheduleReservesOnlyItsEnabledDepth) {
  struct Case {
    bool enabled;
    std::uint32_t depth;
    bool complete;
    bool adaptive;
    std::uint32_t target_context;
    std::uint32_t usable;
  };
  for (const auto c :
       {Case{true, 2, true, false, kContext, 1022}, Case{true, 3, true, true, kContext, 1021},
        Case{false, 3, true, true, kContext, 1024}, Case{true, 3, false, true, kContext, 1024},
        Case{true, 2, true, false, 2, 0}}) {
    SCOPED_TRACE(c.enabled);
    SCOPED_TRACE(c.depth);
    SCOPED_TRACE(c.target_context);
    en::PagedNode node(en::NodeSettings{
        .compute_streams = 1, .slots = 2, .copy_lane = true, .slot_bytes = kExtent});
    ASSERT_TRUE(node.Open());
    auto build = Build(true);
    rt::EngineLlm llm(
        node, entry_, Settings(c.enabled), roles_, 0,
        [build,
         c](const rt::ModelSettings& settings) -> std::expected<rt::EngineLlmParts, std::string> {
          auto parts = build(settings);
          if (!parts) return parts;
          parts->mode.drafters.resize(1);
          auto& draft = parts->mode.drafters.front();
          draft.max_depth = c.depth;
          draft.draft_before_limit = c.complete;
          draft.adaptive = c.adaptive;
          parts->branch_depth = 3;
          parts->config.slots.stages.front().layout.context = c.target_context;
          return parts;
        });
    const auto set = llm.Setup();
    if (c.usable == 0) {
      ASSERT_FALSE(set);
      EXPECT_NE(Why(set).find("complete draft schedule does not fit"), std::string::npos);
      EXPECT_FALSE(llm.has_paged());
    } else {
      ASSERT_TRUE(set) << Why(set);
      EXPECT_EQ(llm.usable_context(), c.usable);
    }
  }
}

TEST_F(EngineLlmTest, NativeVerifyTokenRangeRefusesBeforeAcceptance) {
  native_outputs_ = true;
  Start(true, true);
  Open({&BranchAt(0)});
  const auto prompt = Prompt(37, 7);
  std::vector<float> head;
  ASSERT_TRUE(BranchAt(0).Prefill(prompt, head));
  source_->options().native_token_phase = pl::Phase::kVerify;
  rt::GenerateOptions options;
  options.max_tokens = 8;
  options.stop = false;
  for (const auto token : {-1, static_cast<std::int32_t>(kWidth)}) {
    source_->options().native_token = token;
    rt::Generation bad;
    const auto run = BranchAt(0).Generate(head, options, bad);
    ASSERT_FALSE(run);
    EXPECT_NE(run.error().find("outside the target vocabulary"), std::string::npos);
    EXPECT_EQ(bad.tokens.size(), 1U);  // only the already valid prefill choice
    EXPECT_EQ(llm_->runner().slots().positions(0), prompt.size());
    EXPECT_TRUE(llm_->runner().slots().usable(0));
  }
  source_->options().native_token.reset();
  rt::Generation good;
  ASSERT_TRUE(BranchAt(0).Generate(head, options, good));
  EXPECT_EQ(good.tokens, Chain(prompt, 8));
  Close();
}

// Native proposal tokens (followed by next features) and multirow verify
// tokens use the ordinary serving sampler/state path, including a truncated
// final verify and the branch-owned adaptive state preserved by snapshots.
TEST_F(EngineLlmTest, NativeDraftAndVerifyTokensPreserveRowsAndSnapshotContinuation) {
  native_outputs_ = true;
  Start(true, true);
  const auto prompt = Prompt(37, 7);
  const auto tokens = Turn(0, prompt, 18);
  EXPECT_EQ(tokens.out.tokens, Chain(prompt, 18));
  EXPECT_GT(tokens.out.drafted, 0U);
  const auto rows = Turn(1, prompt, 18, true);
  EXPECT_EQ(rows.out.tokens, tokens.out.tokens);
  ASSERT_EQ(rows.out.logits.size(), 18U);
  for (std::size_t i = 0; i < rows.out.tokens.size(); ++i)
    EXPECT_EQ(ArgmaxOf(rows.out.logits[i]), rows.out.tokens[i]);
  Open({&BranchAt(0)});
  rt::SnapshotStore store(*node_, 8 * kExtent);
  auto captured = BranchAt(0).CaptureState(store);
  ASSERT_TRUE(captured) << captured.error();
  const auto original = BranchAt(0).history();
  auto next = original;
  next.push_back(tokens.out.tokens.back());
  std::vector<float> head;
  std::uint32_t reused = 0;
  ASSERT_TRUE(BranchAt(0).PreparePrompt(next, 0, head, reused));
  rt::GenerateOptions options;
  options.max_tokens = 8;
  options.stop = false;
  rt::Generation first;
  ASSERT_TRUE(BranchAt(0).Generate(head, options, first));
  ASSERT_TRUE(BranchAt(0).RestoreState(store, *captured));
  EXPECT_EQ(BranchAt(0).history(), original);
  ASSERT_TRUE(BranchAt(0).PreparePrompt(next, 0, head, reused));
  rt::Generation again;
  ASSERT_TRUE(BranchAt(0).Generate(head, options, again));
  EXPECT_EQ(again.tokens, first.tokens);
  EXPECT_EQ(again.drafted, first.drafted);
  EXPECT_EQ(again.accepted, first.accepted);
  Close();
  // Changed prompt inputs still use the captured step programs.
  const auto changed = Turn(0, Prompt(37, 11), 18);
  EXPECT_EQ(changed.out.tokens, Chain(Prompt(37, 11), 18));
  EXPECT_NE(changed.out.tokens, tokens.out.tokens);
}

}  // namespace
