// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The one model runner (engine/model_runner.h), its wave composer
// (engine/wave_composer.h) and its prefill orchestrator (engine/prefill.h)
// on the fake device backend, over a step source whose programs write
// deterministic bytes (tests/support/fake_step_source.h): the order of its
// life and its release (recordings before the launch), the measured
// workspace, staging by input slot, a shape captured on its second run and
// replayed, plan and graph reclaim, the coverage check, the composer's
// joins, groups, alignments and caps, every owner's outputs its own, one
// owner's refusal (inactive, at other positions, out of state capacity)
// never failing its peers, a refused joined plan run owner by owner, and
// prompts chunked, joined, cut state-only and finished with their heads,
// with the lookahead's futures built, captured ahead and their fresh state
// prepared, giving the same state as one unchunked run.

#include "engine/model_runner.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <expected>
#include <filesystem>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "artifact/format_map.h"
#include "base/bytes.h"
#include "engine/paged_node.h"
#include "engine/ple_rows.h"
#include "engine/prefill.h"
#include "engine/runner_memory.h"
#include "engine/step_runs.h"
#include "engine/wave_composer.h"
#include "expected_error.h"
#include "fake_step_source.h"
#include "kernels/common/recording_launch.h"
#include "model/batching.h"
#include "model/state_contract.h"
#include "planner/step_shape.h"
#include "providers/fake/fake_device_runtime.h"
#include "providers/fake/fake_storage.h"

namespace {

namespace en = llmp::engine;
namespace kn = llmp::kernels;
namespace md = llmp::model;
namespace pl = llmp::planner;
namespace ts = llmp::test_support;
using llmp::base::Bytes;
constexpr std::uint64_t kExtent = en::kPagedExtent;
constexpr std::uint32_t kContext = 1024;
constexpr std::uint32_t kSlots = 3;

// Exercise the exact pre-allocation source gate with the physical v0 table
// representation: 37 rows of 90 U8 bytes encode 160 NVFP4 values per row.
TEST(RowTableSourceTest, CompletePhysicalPartsMustReconstructTheLogicalTable) {
  namespace ar = llmp::artifact;
  namespace fm = llmp::format;
  std::array<ar::Resource, 2> resources{};
  resources[0].repr = {.family = ar::Family::kPlain,
                       .type = "U8",
                       .dims = {37, 90},
                       .role = {},
                       .codebook = {},
                       .bytes = 3330,
                       .readable = 3330};
  resources[1].repr = {.family = ar::Family::kPlain,
                       .type = "F32",
                       .dims = {1},
                       .role = {},
                       .codebook = {},
                       .bytes = 4,
                       .readable = 4};
  for (auto& resource : resources) {
    resource.bytes = Bytes(resource.repr.bytes);
    auto physical = ar::FormatOf(resource.repr);
    ASSERT_TRUE(physical);
    resource.format = *physical;
  }
  const std::array parts{ar::V0Part{fm::PartRole::kData, &resources[0].repr},
                         ar::V0Part{fm::PartRole::kTensorScale, &resources[1].repr}};
  auto logical = ar::Assemble(fm::Encoding::kNvfp4, fm::Layout::kRowInterleaved, parts);
  ASSERT_TRUE(logical);
  EXPECT_EQ(logical->shape.ne[0], 160U);
  EXPECT_EQ(logical->shape.ne[1], 37U);
  EXPECT_NE(*resources[0].format, *logical);
  en::RunnerConfig::RowTable table{
      .weight = {},
      .resource = 0,
      .format = *logical,
      .rows_per_token = 2,
      .source_parts = {{fm::PartRole::kData, 0}, {fm::PartRole::kTensorScale, 1}}};
  EXPECT_TRUE(en::CheckRowTableSource(table, 0, resources));
  auto bad = table;
  bad.source_parts.clear();
  EXPECT_FALSE(en::CheckRowTableSource(bad, 0, resources));
  bad = table;
  bad.source_parts.pop_back();
  EXPECT_FALSE(en::CheckRowTableSource(bad, 0, resources));
  bad = table;
  bad.source_parts.push_back(bad.source_parts.front());
  EXPECT_FALSE(en::CheckRowTableSource(bad, 0, resources));
  bad = table;
  bad.source_parts[1].second = 99;
  EXPECT_FALSE(en::CheckRowTableSource(bad, 0, resources));
  bad = table;
  bad.source_parts[1].second = 0;
  EXPECT_FALSE(en::CheckRowTableSource(bad, 0, resources));
  bad = table;
  bad.resource = 1;
  EXPECT_FALSE(en::CheckRowTableSource(bad, 0, resources));
  bad = table;
  --bad.format.shape.ne[1];
  EXPECT_FALSE(en::CheckRowTableSource(bad, 0, resources));
  resources[1].repr.type = "I32";
  EXPECT_FALSE(en::CheckRowTableSource(table, 0, resources));
  resources[1].repr.type = "F32";
  resources[0].bytes = Bytes(3329);
  EXPECT_FALSE(en::CheckRowTableSource(table, 0, resources));
}

TEST(RowTableSourceTest, OrdinaryV0AndStoredV1KeepExactFormatAndPartChecks) {
  namespace ar = llmp::artifact;
  namespace fm = llmp::format;
  ar::Resource resource;
  resource.repr = {.family = ar::Family::kPlain,
                   .type = "F32",
                   .dims = {3, 8},
                   .role = {},
                   .codebook = {},
                   .bytes = 96,
                   .readable = 96};
  auto format = ar::FormatOf(resource.repr);
  ASSERT_TRUE(format);
  resource.format = *format;
  resource.bytes = Bytes(96);
  const std::span resources(&resource, 1);
  en::RunnerConfig::RowTable table{.weight = {}, .format = *format, .rows_per_token = 1};
  for (const std::uint32_t version : {0U, 1U}) {
    EXPECT_TRUE(en::CheckRowTableSource(table, version, resources));
    auto bad = table;
    bad.format.encoding = fm::Encoding::kI32;
    EXPECT_FALSE(en::CheckRowTableSource(bad, version, resources));
    bad = table;
    bad.part = fm::PartRole::kBlockScales;
    EXPECT_FALSE(en::CheckRowTableSource(bad, version, resources));
    bad = table;
    bad.resource = 1;
    EXPECT_FALSE(en::CheckRowTableSource(bad, version, resources));
  }
  table.source_parts = {{fm::PartRole::kData, 0}};
  EXPECT_TRUE(en::CheckRowTableSource(table, 0, resources));
  EXPECT_FALSE(en::CheckRowTableSource(table, 1, resources));
}

// These guards must run before any copy, program call or whole-graph launch.
// A retained routed recording alone must not be mistaken for an ordinary graph.
class RoutedGuardProgram final : public kn::StepProgram, public kn::RoutedProgram {
 public:
  bool expose_route = true;
  mutable unsigned calls = 0;
  std::uint64_t scratch_bytes() const override { return 0; }
  std::uint64_t host_bytes() const override { return 0; }
  std::uint64_t launches() const override { return 0; }
  std::span<const kn::InputSlot> inputs() const override { return {}; }
  std::span<const kn::OutputSlot> outputs() const override { return {}; }
  std::span<const kn::TensorUse> tensors() const override { return {}; }
  kn::RoutedProgram* routed() override { return expose_route ? this : nullptr; }
  const kn::RoutedProgram* routed() const override { return expose_route ? this : nullptr; }
  std::expected<void, kn::StepFailure> Bind(kn::StepLaunch&) override { return {}; }
  std::expected<void, kn::StepFailure> Run(kn::StepLaunch&,
                                           llmp::providers::NativeStream) const override {
    ++calls;
    return {};
  }
  std::span<const kn::RoutedWeight> weights() const override { return {}; }
  std::size_t groups() const override { return 1; }
  const kn::RoutedLayout& layout(std::size_t = 0) const override { return layout_; }
  std::span<const std::size_t> members(std::size_t) const override { return {}; }
  std::uint64_t pinned_bytes() const override { return 0; }
  const llmp::base::Sha256Digest& routing_identity() const override { return identity_; }
  const llmp::base::Sha256Digest& graph_identity() const override { return identity_; }
  std::expected<void, kn::StepFailure> RunSegment(const kn::RoutedAccess&, kn::StepLaunch&,
                                                  llmp::providers::NativeStream) override {
    ++calls;
    return {};
  }
  std::expected<void, kn::StepFailure> Prepare(const kn::RoutedAccess&,
                                               std::span<std::byte>) override {
    ++calls;
    return {};
  }
  std::expected<void, kn::StepFailure> CompleteSegment(const kn::RoutedAccess&) override {
    ++calls;
    return {};
  }
  std::expected<std::unique_ptr<kn::RoutedRecording>, kn::StepFailure> RecordSegment(
      const kn::RoutedAccess&, kn::StepLaunch&, llmp::providers::NativeStream,
      const kn::StepLaunch::Body&) override {
    ++calls;
    return std::unexpected(kn::StepFailure{.detail = "unexpected routed recording"});
  }
  std::expected<void, kn::StepFailure> ReplaySegment(const kn::RoutedAccess&, kn::StepLaunch&,
                                                     llmp::providers::NativeStream,
                                                     const kn::RoutedRecording&) override {
    ++calls;
    return {};
  }

 private:
  kn::RoutedLayout layout_;
  llmp::base::Sha256Digest identity_{};
};
class GuardRoutedRecording final : public kn::RoutedRecording {
 public:
  explicit GuardRoutedRecording(RoutedGuardProgram& program) : program_(program) {}
  const kn::RoutedProgram* owner() const override { return &program_; }
  std::size_t segment() const override { return 0; }
  std::size_t nodes() const override { return 0; }
  double capture_seconds() const override { return 0; }
  double instantiate_seconds() const override { return 0; }
  std::expected<void, kn::StepFailure> Replay(const kn::RoutedAccess&, kn::StepLaunch&,
                                              llmp::providers::NativeStream) const override {
    ++program_.calls;
    return {};
  }

 private:
  RoutedGuardProgram& program_;
};
TEST(StepRuns, OrderedRouteProgressRequiresActualCompletionAndFreshPreparation) {
  kn::RoutedProgress progress;
  EXPECT_FALSE(progress.Start(0, 2));
  EXPECT_FALSE(progress.Start(1, 0));
  ASSERT_TRUE(progress.Start(1, 2));
  // Recording may check readiness repeatedly, but supplies no completion.
  EXPECT_TRUE(progress.Ready(1, 0));
  EXPECT_TRUE(progress.Ready(1, 0));
  EXPECT_FALSE(progress.Complete(1, 0));
  EXPECT_FALSE(progress.Prepared(1, 0));
  EXPECT_FALSE(progress.Ready(1, 1));
  ASSERT_TRUE(progress.Submit(1, 0));
  EXPECT_FALSE(progress.Submit(1, 0));
  EXPECT_FALSE(progress.Complete(1, 1));
  ASSERT_TRUE(progress.Complete(1, 0));
  EXPECT_FALSE(progress.Complete(1, 0));
  EXPECT_FALSE(progress.Prepared(1, 1));
  ASSERT_TRUE(progress.Prepared(1, 0));
  ASSERT_TRUE(progress.Submit(1, 1));
  ASSERT_TRUE(progress.Complete(1, 1));
  EXPECT_FALSE(progress.Ready(1, 2));
  ASSERT_TRUE(progress.Prepared(1, 1));
  ASSERT_TRUE(progress.Submit(1, 2));
  ASSERT_TRUE(progress.Complete(1, 2));
  EXPECT_FALSE(progress.Ready(1, 2));
  EXPECT_FALSE(progress.Start(1, 2));
  ASSERT_TRUE(progress.Start(2, 2));
  EXPECT_FALSE(progress.Ready(1, 0));
  EXPECT_TRUE(progress.Ready(2, 0));
}

TEST(StepRuns, UnknownSecondSegmentCannotAdvanceOrRestartRouteProgress) {
  kn::RoutedProgress progress;
  ASSERT_TRUE(progress.Start(1, 2));
  ASSERT_TRUE(progress.Submit(1, 0));
  ASSERT_TRUE(progress.Complete(1, 0));
  ASSERT_TRUE(progress.Prepared(1, 0));
  ASSERT_TRUE(progress.Submit(1, 1));
  // A failed/unknown job is not a completion certificate. This fake protocol
  // control deliberately withholds Complete; it injects no GPU failure.
  EXPECT_TRUE(progress.awaiting());
  EXPECT_FALSE(progress.Prepared(1, 1));
  EXPECT_FALSE(progress.Ready(1, 2));
  EXPECT_FALSE(progress.Start(2, 2));
  EXPECT_FALSE(progress.Complete(2, 1));
  EXPECT_TRUE(progress.awaiting());
}

TEST(StepRuns, OrdinaryQueueAndCaptureRefuseRoutedProgramsAndRetainedSegmentsBeforeEffects) {
  RoutedGuardProgram program;
  for (bool graphs : {false, true}) {
    en::StepRuns steps(graphs);
    kn::RecordingLaunch launch;
    steps.SetLaunch(&launch);
    for (unsigned mode = 0; mode < 3; ++mode) {
      program.expose_route = mode != 2;
      en::StepPlanRuns runs;
      runs.routed_graphs.resize(2);
      if (mode != 0) runs.routed_graphs[0] = std::make_unique<GuardRoutedRecording>(program);
      const auto* retained = runs.routed_graphs[0].get();
      en::GraphStats stats;
      unsigned between = 0;
      for (bool capture : {false, true}) {
        const auto queued = steps.Queue(runs, {},
                                        [&](auto) {
                                          ++between;
                                          return true;
                                        },
                                        program, {}, capture, stats, {});
        ASSERT_FALSE(queued.result);
        EXPECT_EQ(queued.result.error().error, kn::StepError::kRejected);
        EXPECT_EQ(queued.result.error().detail,
                  "ordinary queue refuses routed programs or recordings");
        EXPECT_FALSE(queued.before);
      }
      const auto ahead = steps.CaptureAhead(runs, {}, program, {}, stats, {});
      ASSERT_FALSE(ahead);
      EXPECT_EQ(ahead.error().error, kn::StepError::kRejected);
      EXPECT_EQ(ahead.error().detail, "ordinary capture refuses routed programs or recordings");
      EXPECT_EQ(runs.routed_graphs[0].get(), retained);
      EXPECT_EQ(runs.graph, nullptr);
      EXPECT_EQ(runs.eager_runs, 0U);
      EXPECT_EQ(program.calls, 0U);
      EXPECT_EQ(between, 0U);
      EXPECT_EQ(launch.captures(), 0U);
      EXPECT_EQ(launch.launches(), 0U);
    }
  }
}

// The pinned-memory API promises no direct-I/O block alignment. Force the fake's
// allowed 256-byte alignment to prove that the shared allocation owns both
// its exact charged backing and the distinct 4 KiB landing view.
static_assert(sizeof(void*) <= 8);
static_assert(sizeof(en::AlignedPinned) <= 32);
static_assert(sizeof(std::expected<void*, std::string>) <= 48);
static_assert(sizeof(std::expected<en::AlignedPinned, std::string>) <= 48);

class RunnerAlignedPinnedTest : public ::testing::Test {
 protected:
  en::PagedNode node_{{.compute_streams = 1, .slots = 1, .slot_bytes = kExtent}};
  bool retired_ = false;
  void SetUp() override { ASSERT_TRUE(node_.Open()); }
  void TearDown() override {
    llmp::providers::fake::ClearPinnedMisalignment();
    llmp::providers::fake::ClearPinnedFreeHook();
    if (!retired_) EXPECT_TRUE(node_.TearDown({}));
  }
};

TEST_F(RunnerAlignedPinnedTest, UnalignedOwnerFundsAndCoversTheCompleteDirectReadView) {
  namespace pf = llmp::providers::fake;
  constexpr std::uint64_t bytes = 8192;
  constexpr std::uint64_t charged = bytes + en::kPleBlock - 1;
  en::RunnerMemory memory(node_, 0, 0);
  const auto before = node_.catalog().OccupancyOf(node_.domain()).Total().value();
  pf::MisalignNextPinned(charged);
  auto held = memory.PinnedAligned(bytes, en::kPleBlock);
  ASSERT_TRUE(held) << held.error();
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(held->allocation) % en::kPleBlock, 256U);
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(held->data) % en::kPleBlock, 0U);
  ASSERT_NE(held->allocation, held->data);
  const auto offset = held->data - static_cast<std::byte*>(held->allocation);
  EXPECT_EQ(offset, 3840);
  EXPECT_LE(static_cast<std::uint64_t>(offset) + bytes, held->allocation_bytes);
  EXPECT_EQ(held->bytes, bytes);
  EXPECT_EQ(held->allocation_bytes, charged);
  EXPECT_EQ(memory.pinned_bytes(), charged);
  EXPECT_EQ(node_.catalog().OccupancyOf(node_.domain()).Total().value() - before, charged);
  ASSERT_EQ(memory.extents().size(), 1U);
  auto extent = node_.catalog().Describe(memory.extents().front());
  ASSERT_TRUE(extent);
  EXPECT_EQ(extent->descriptor.size.value(), charged);

  pf::FakeStorage storage(4, en::kPleBlock);
  const int fd = storage.AddFile(std::vector<std::byte>(bytes, std::byte{0x5a}));
  const en::PleTable table{.fd = fd, .rows = 80, .row_bytes = 90, .file_bytes = bytes};
  // Row 45 crosses the 4 KiB boundary, requiring both complete blocks.
  const std::array<std::int32_t, 1> rows{45};
  auto plan = en::PlanPleRows(table, rows, bytes, 1);
  ASSERT_TRUE(plan) << plan.error();
  ASSERT_EQ(plan->landing_bytes, bytes);
  EXPECT_FALSE(en::ReadPleRows(storage, fd, *plan, static_cast<std::byte*>(held->allocation)));
  ASSERT_EQ(storage.in_flight(), 0U);
  ASSERT_TRUE(en::ReadPleRows(storage, fd, *plan, held->data));
  EXPECT_EQ(held->data[0], std::byte{0x5a});
  EXPECT_EQ(held->data[bytes - 1], std::byte{0x5a});
  bool freed = false;
  pf::AfterNextPinnedFree(held->allocation, [&] { freed = true; });
  auto retired = node_.TearDown({});
  retired_ = true;
  EXPECT_TRUE(retired) << llmp::test_support::Failed(retired).value_or("");
  EXPECT_TRUE(freed);
}

TEST_F(RunnerAlignedPinnedTest, StalledInteriorReadRetainsItsRawOwnerPastTeardown) {
  namespace pf = llmp::providers::fake;
  constexpr std::uint64_t bytes = 8192;
  en::RunnerMemory memory(node_, 0, 0);
  pf::MisalignNextPinned(bytes + en::kPleBlock - 1);
  auto held = memory.PinnedAligned(bytes, en::kPleBlock);
  ASSERT_TRUE(held) << held.error();
  ASSERT_NE(held->allocation, held->data);
  std::memset(held->data, 0, bytes);
  auto storage = std::make_unique<pf::FakeStorage>(4, en::kPleBlock);
  auto* ring = storage.get();
  const int fd = ring->AddFile(std::vector<std::byte>(bytes, std::byte{0x6b}));
  ring->ScriptNext(
      {.submission = llmp::providers::Submission::kAccepted, .result = std::nullopt, .hold = true});
  ASSERT_EQ(ring->Submit({.token = 7,
                          .kind = llmp::providers::IoKind::kRead,
                          .fd = fd,
                          .offset = 0,
                          .memory = held->data,
                          .length = bytes,
                          .segments = {}}),
            llmp::providers::Submission::kAccepted);
  bool freed = false;
  pf::AfterNextPinnedFree(held->allocation, [&] { freed = true; });
  const std::array<void*, 1> raw_owner{held->allocation};
  ASSERT_TRUE(node_.RetireRing(std::move(storage), raw_owner));
  EXPECT_EQ(node_.kept_pinned(), 1U);
  auto retired = node_.TearDown({});
  retired_ = true;
  EXPECT_TRUE(retired) << llmp::test_support::Failed(retired).value_or("");
  EXPECT_FALSE(freed);
  ASSERT_TRUE(ring->Release(7));
  std::array<llmp::providers::IoCompletion, 1> done;
  ASSERT_EQ(ring->Harvest(done, false), 1U);
  EXPECT_EQ(done[0].result, bytes);
  EXPECT_EQ(held->data[bytes - 1], std::byte{0x6b});
  EXPECT_EQ(ring->in_flight(), 0U);
  // The production retention rule is process lifetime even after a late
  // completion; the test never manually frees the retained allocation.
  pf::ClearPinnedFreeHook();
}

TEST_F(RunnerAlignedPinnedTest, InvalidGeometryAndUnfundedSlackAllocateNothing) {
  en::RunnerMemory memory(node_, 0, 0);
  const auto before = node_.catalog().OccupancyOf(node_.domain()).Total().value();
  EXPECT_FALSE(memory.PinnedAligned(0, 4096));
  EXPECT_FALSE(memory.PinnedAligned(8192, 0));
  EXPECT_FALSE(memory.PinnedAligned(8192, 3));
  EXPECT_FALSE(memory.PinnedAligned(std::numeric_limits<std::uint64_t>::max(), 4096));
  EXPECT_EQ(memory.pinned_bytes(), 0U);
  EXPECT_TRUE(memory.extents().empty());
  EXPECT_EQ(node_.catalog().OccupancyOf(node_.domain()).Total().value(), before);
  // The payload alone fits; its full alignment allowance does not.
  ASSERT_TRUE(node_.Start(Bytes(before + 8192)));
  node_.Run();
  EXPECT_FALSE(memory.PinnedAligned(8192, 4096));
  EXPECT_EQ(memory.pinned_bytes(), 0U);
  EXPECT_TRUE(memory.extents().empty());
  std::uint64_t after = 0;
  ASSERT_TRUE(node_.Call(
      [&]() -> en::Status {
        after = node_.catalog().OccupancyOf(node_.domain()).Total().value();
        return {};
      },
      "aligned-pinned refusal occupancy"));
  EXPECT_EQ(after, before);
}

// The target: 4 KiB a position (tensor 0: 512 positions an extent, two
// extents) and 4 KiB of fixed state (tensor 1, a third extent).
std::vector<en::StageState> Stages() {
  using C = md::StateTensorContract;
  const std::vector<std::pair<std::uint32_t, md::StateContract>> layers = {
      {0,
       {.tensors = {C{.name = "kv", .kind = md::StateKind::kKv, .cell_bytes = 4096},
                    C{.name = "state",
                      .kind = md::StateKind::kRecurrent,
                      .encoding = md::Encoding::kF32,
                      .cell_bytes = 4096,
                      .cells = md::CellRule::kFixed,
                      .window_or_ratio = 1,
                      .rollback = md::Rollback::kSnapshot}}}}};
  return {{.name = "target", .layout = md::Layout(layers, kContext, 256).value()}};
}

pl::Segment Seg(std::uint8_t slot, std::uint32_t past, std::uint32_t rows, std::uint32_t outputs,
                std::uint32_t align = 256) {
  const auto cells = static_cast<std::uint32_t>(
      std::min<std::uint64_t>((std::uint64_t{past} + rows + align - 1) / align * align, kContext));
  return {.slot = slot,
          .rows = rows,
          .past = past,
          .read_cells = cells,
          .outputs = outputs,
          .group = 0,
          .want_head = outputs != 0,
          .greedy = false};
}

pl::StepShape Shape(pl::Phase phase, std::vector<pl::Segment> segments, std::uint32_t align = 256) {
  return {.stage = 0,
          .phase = phase,
          .segments = std::move(segments),
          .row_invariant = false,
          .read_align = align};
}

std::vector<std::int32_t> Tokens(std::uint32_t n, std::int32_t from) {
  std::vector<std::int32_t> out(n);
  std::iota(out.begin(), out.end(), from);
  return out;
}

std::uint64_t Hash(std::span<const std::int32_t> tokens) {
  std::uint64_t h = 0;
  for (const std::int32_t t : tokens) h = ts::FakeHash(h, t);
  return h;
}

std::vector<ts::FakeRow> Rows(std::span<const std::byte> bytes) {
  std::vector<ts::FakeRow> rows(bytes.size() / sizeof(ts::FakeRow));
  if (!rows.empty()) std::memcpy(rows.data(), bytes.data(), rows.size() * sizeof(ts::FakeRow));
  return rows;
}

// A launch that logs what it destroys, around RecordingLaunch.
class LoggedLaunch final : public kn::StepLaunch {
 public:
  explicit LoggedLaunch(std::vector<std::string>* log) : log_(log) {}
  ~LoggedLaunch() override { log_->push_back("launch"); }
  std::expected<std::unique_ptr<kn::Recorded>, kn::StepFailure> Capture(
      llmp::providers::NativeStream stream, const Body& body) override {
    auto inner = inner_.Capture(stream, body);
    if (!inner) return std::unexpected(inner.error());
    return std::make_unique<Logged>(this, std::move(*inner), log_);
  }
  std::expected<void, kn::StepFailure> Launch(llmp::providers::NativeStream stream,
                                              const kn::Recorded& recorded) override {
    if (recorded.owner() != this) {
      return std::unexpected(kn::StepFailure{.error = kn::StepError::kRejected, .detail = "?"});
    }
    return inner_.Launch(stream, *static_cast<const Logged&>(recorded).inner);
  }

 private:
  struct Logged final : kn::Recorded {
    Logged(const kn::StepLaunch* owner, std::unique_ptr<kn::Recorded> recorded,
           std::vector<std::string>* log)
        : kn::Recorded(owner), inner(std::move(recorded)), log(log) {}
    ~Logged() override { log->push_back("recording"); }
    std::size_t nodes() const override { return 1; }
    double capture_seconds() const override { return inner->capture_seconds(); }
    double instantiate_seconds() const override { return inner->instantiate_seconds(); }
    std::unique_ptr<kn::Recorded> inner;
    std::vector<std::string>* log;
  };
  kn::RecordingLaunch inner_;
  std::vector<std::string>* log_;
};

class ModelRunnerTest : public ::testing::Test {
 protected:
  en::PagedNode node_{
      en::NodeSettings{.compute_streams = 1, .slots = 2, .copy_lane = true, .slot_bytes = kExtent}};
  std::unique_ptr<en::ModelRunner> runner_;
  ts::FakeStepSource* source_ = nullptr;
  std::vector<std::string> log_;
  std::uint64_t fixed_ = 0;

  virtual std::vector<en::StageState> StateStages() const { return Stages(); }
  virtual ts::FakeSourceOptions SourceOptions() {
    ts::FakeSourceOptions o;
    for (const en::StageState& s : StateStages()) o.layouts.push_back(s.layout);
    return o;
  }
  virtual std::vector<pl::StepShape> SetupShapes() {
    return {Shape(pl::Phase::kPrefill, {Seg(0, kContext - 512, 512, 512)}),
            Shape(pl::Phase::kDecode, {Seg(0, 0, 4, 4), Seg(1, 0, 4, 4), Seg(2, 0, 4, 4)}, 2048),
            Shape(pl::Phase::kPrefill, {Seg(0, 0, 512, 1), Seg(1, 0, 512, 1)})};
  }
  virtual std::uint64_t BudgetExtents() const { return 64; }
  virtual void Configure(en::RunnerConfig& /*config*/) {}

  void SetUp() override {
    // NOLINTNEXTLINE(concurrency-mt-unsafe): test environment is immutable
    const char* scratch = std::getenv("LLMP_TEST_SCRATCH");
    ASSERT_NE(scratch, nullptr);
    std::filesystem::create_directories(scratch);
    ASSERT_TRUE(node_.Open());
    fixed_ = node_.catalog().OccupancyOf(node_.domain()).Total().value();
    auto source = std::make_unique<ts::FakeStepSource>(SourceOptions());
    source_ = source.get();
    en::RunnerConfig config;
    config.slots = {.model = "Fake",
                    .slots = kSlots,
                    .stages = StateStages(),
                    .spec = {},
                    .read_align = 256,
                    .spill_place = {},
                    .spill_directory = scratch,
                    .invalidate = {},
                    .drop_plans = {}};
    config.setup_shapes = SetupShapes();
    config.launch = [this]() -> std::expected<std::unique_ptr<kn::StepLaunch>, std::string> {
      return std::make_unique<LoggedLaunch>(&log_);
    };
    Configure(config);
    auto runner = en::ModelRunner::Create(node_, std::move(config), std::move(source), 0, 0);
    ASSERT_TRUE(runner.has_value()) << runner.error();
    runner_ = std::move(*runner);
    // Out of order: refused.
    EXPECT_FALSE(runner_->Register());
    EXPECT_FALSE(runner_->Bind());
    auto set = runner_->Setup();
    ASSERT_TRUE(set) << llmp::test_support::Failed(set).value_or("");
    EXPECT_FALSE(runner_->Setup());
    EXPECT_FALSE(runner_->Bind());
    ASSERT_TRUE(node_.MapWorkspace(runner_->activations_needed(),
                                   std::max(runner_->pool_needed(), kExtent)));
    auto started = node_.Start(Bytes(fixed_ + (BudgetExtents() * kExtent)));
    ASSERT_TRUE(started) << llmp::test_support::Failed(started).value_or("");
    auto registered = runner_->Register();
    ASSERT_TRUE(registered) << llmp::test_support::Failed(registered).value_or("");
    node_.Run();
    auto bound = runner_->Bind();
    ASSERT_TRUE(bound) << llmp::test_support::Failed(bound).value_or("");
    auto selected = runner_->slots().SelectSlots(std::vector<std::uint32_t>{0, 1, 2});
    ASSERT_TRUE(selected) << llmp::test_support::Failed(selected).value_or("");
    // Several active slots step under one held request (request_cohort.h).
    ASSERT_TRUE(node_.BeginRequest(0, runner_->slots().execution(), "test request"));
  }
  void TearDown() override {
    if (runner_ == nullptr) return;
    if (node_.InRequest(0)) EXPECT_TRUE(node_.EndRequest(0));
    std::array<en::PagedModel*, 1> models = {runner_.get()};
    auto closed = node_.TearDown(models);
    EXPECT_TRUE(closed) << llmp::test_support::Failed(closed).value_or("");
  }

  std::expected<en::StepReport, en::StepRefusal> Run(const pl::StepShape& shape,
                                                     std::vector<std::vector<std::int32_t>> tokens,
                                                     en::StepAhead ahead = {}) {
    tokens_ = std::move(tokens);
    std::vector<en::SegmentWork> work;
    for (const auto& t : tokens_) work.push_back({.tokens = t, .bytes = {}});
    const en::StepRequest request{
        .shape = shape, .work = work, .step_bytes = {}, .advance = true, .ahead = std::move(ahead)};
    return runner_->RunStep(request);
  }
  // A slot's cells [0, n), read back, each checked for its position.
  std::vector<ts::FakeCell> Cells(std::uint32_t slot, std::uint32_t n) {
    const auto& layout = runner_->slots().layout(0);
    const std::uint64_t cell = layout.items[0].contract.cell_bytes;
    std::vector<en::LiveState::Range> ranges = {
        {.region = 0, .offset = layout.items[0].offset, .bytes = std::uint64_t{n} * cell}};
    std::vector<llmp::catalog::ExtentId> staging;
    auto host = node_.Pinned((((n * cell) + kExtent - 1) / kExtent) * kExtent, 0, staging);
    EXPECT_TRUE(host.has_value());
    if (!host) return {};
    auto copied = runner_->slots().CopyState(slot, *host, ranges, true);
    EXPECT_TRUE(copied) << llmp::test_support::Failed(copied).value_or("");
    std::vector<ts::FakeCell> cells(n);
    for (std::uint32_t i = 0; i < n; ++i) {
      std::memcpy(&cells[i], static_cast<const std::byte*>(*host) + (std::uint64_t{i} * cell),
                  sizeof(ts::FakeCell));
    }
    (void)node_.FreePinned(*host);
    return cells;
  }
  void ExpectPrompt(std::uint32_t slot, std::span<const std::int32_t> tokens) {
    const auto cells = Cells(slot, static_cast<std::uint32_t>(tokens.size()));
    for (std::size_t i = 0; i < cells.size(); ++i) {
      EXPECT_EQ(cells[i].magic, ts::kFakeMagic) << "slot " << slot << " position " << i;
      EXPECT_EQ(cells[i].slot, static_cast<std::int32_t>(slot));
      EXPECT_EQ(cells[i].position, static_cast<std::int32_t>(i));
      EXPECT_EQ(cells[i].token, tokens[i]);
    }
  }
  // Pinned memory leaving `room` extents of the budget free.
  void Leave(std::uint64_t room) {
    auto free = node_.FreeBytes();
    ASSERT_TRUE(free.has_value());
    ASSERT_GT(*free / kExtent, room);
    std::vector<llmp::catalog::ExtentId> staging;
    auto taken = node_.Pinned(((*free / kExtent) - room) * kExtent, 0, staging);
    ASSERT_TRUE(taken.has_value()) << taken.error();
  }

  std::vector<std::vector<std::int32_t>> tokens_;
};

TEST_F(ModelRunnerTest, MeasuresTheWidestShapesAtSetup) {
  // The widest shape's inputs and outputs, a quarter's margin, and the
  // plan floor one program's host bytes.
  EXPECT_GE(runner_->activations_needed(), kExtent);
  EXPECT_EQ(runner_->plan_floor_bytes(), 4096U);
  EXPECT_GT(runner_->host_input_bytes(), 0U);
  ASSERT_EQ(source_->built().size(), 3U);
  for (const auto& built : source_->built()) EXPECT_TRUE(built.measuring);
  EXPECT_NE(runner_->report_json().find(R"("plan_floor":4096)"), std::string::npos);
}

TEST_F(ModelRunnerTest, PrefillsWritesItsCellsAndReturnsItsFrontier) {
  const auto tokens = Tokens(100, 7);
  auto ran = Run(Shape(pl::Phase::kPrefill, {Seg(0, 0, 100, 1)}), {tokens});
  ASSERT_TRUE(ran.has_value()) << ran.error().what;
  EXPECT_EQ(runner_->slots().positions(0), 100U);
  const auto rows = Rows(ran->outputs.at(0));
  ASSERT_EQ(rows.size(), 1U);
  EXPECT_EQ(rows[0].slot, 0);
  EXPECT_EQ(rows[0].position, 99);
  EXPECT_EQ(rows[0].token, 106);
  EXPECT_EQ(rows[0].cells, 100);
  EXPECT_EQ(rows[0].hash, Hash(tokens));
  ExpectPrompt(0, tokens);
  // A step at the wrong positions is that slot's refusal, clean.
  auto wrong = Run(Shape(pl::Phase::kDecode, {Seg(0, 50, 1, 1)}), {{1}});
  ASSERT_FALSE(wrong.has_value());
  EXPECT_TRUE(wrong.error().clean);
  EXPECT_EQ(wrong.error().slot, 0U);
  EXPECT_EQ(runner_->slots().positions(0), 100U);
}

TEST_F(ModelRunnerTest, CapturesOnTheSecondRunAndReplays) {
  std::vector<std::int32_t> all;
  std::vector<en::RunPath> paths;
  for (std::uint32_t i = 0; i < 6; ++i) {
    const auto token = static_cast<std::int32_t>(40 + i);
    all.push_back(token);
    auto ran = Run(Shape(pl::Phase::kDecode, {Seg(0, i, 1, 1)}), {{token}});
    ASSERT_TRUE(ran.has_value()) << ran.error().what;
    paths.push_back(ran->path);
    const auto rows = Rows(ran->outputs.at(0));
    ASSERT_EQ(rows.size(), 1U);
    // A replay reads what was staged for it.
    EXPECT_EQ(rows[0].position, static_cast<std::int32_t>(i));
    EXPECT_EQ(rows[0].token, token);
    EXPECT_EQ(rows[0].cells, static_cast<std::int32_t>(i + 1));
    EXPECT_EQ(rows[0].hash, Hash(all));
  }
  EXPECT_EQ(paths[0], en::RunPath::kEager);
  EXPECT_EQ(paths[1], en::RunPath::kCaptured);
  for (std::size_t i = 2; i < paths.size(); ++i) EXPECT_EQ(paths[i], en::RunPath::kReplayed);
  EXPECT_EQ(runner_->plans().size(), 1U);  // past is data, not the plan
  EXPECT_EQ(runner_->plans().graphs(), 1U);
  EXPECT_EQ(runner_->graph_stats().replayed, 4U);
  ExpectPrompt(0, all);
}

class FirstCaptureTest : public ModelRunnerTest {
 protected:
  std::vector<en::StageState> StateStages() const override {
    // This control writes append-only cells. The general fake's fixed hash
    // requires a separate commit program for Verify and is not used here.
    const std::vector<std::pair<std::uint32_t, md::StateContract>> layers = {
        {0, {.tensors = {{.name = "kv", .kind = md::StateKind::kKv, .cell_bytes = 4096}}}}};
    return {{.name = "target", .layout = md::Layout(layers, kContext, 256).value()}};
  }
  void Configure(en::RunnerConfig& config) override {
    config.max_verify_rows = 1;
    config.capture_first_phases = 1U << static_cast<unsigned>(pl::Phase::kVerify);
  }
  std::expected<en::StepReport, en::StepRefusal> Verify(std::uint32_t past) {
    const std::array<std::int32_t, 1> tokens{static_cast<std::int32_t>(40 + past)};
    const std::array<en::SegmentWork, 1> work{{{.tokens = tokens}}};
    return runner_->RunStep({.shape = Shape(pl::Phase::kVerify, {Seg(0, past, 1, 1)}),
                             .work = work,
                             .advance = false,
                             .ahead = {}});
  }
};

TEST_F(FirstCaptureTest, FirstVerifyWarmsOnceThenCapturesAndNextReplays) {
  for (std::uint32_t past = 0; past < 2; ++past) {
    auto ran = Verify(past);
    ASSERT_TRUE(ran) << ran.error().what;
    EXPECT_EQ(ran->path, past == 0 ? en::RunPath::kCaptured : en::RunPath::kReplayed);
    EXPECT_EQ(Rows(ran->outputs.at(0)).at(0).token, static_cast<std::int32_t>(40 + past));
    ASSERT_TRUE(runner_->slots().Accept(0, 1));
  }
  EXPECT_EQ(source_->executed(), 2U);  // recording never executes the state update twice
  EXPECT_EQ(runner_->plans().graphs(), 1U);
  EXPECT_EQ(runner_->lookahead_stats().captured_first, 1U);
  auto ordinary = Run(Shape(pl::Phase::kDecode, {Seg(0, 2, 1, 1)}), {{42}});
  ASSERT_TRUE(ordinary);
  EXPECT_EQ(ordinary->path, en::RunPath::kEager);  // another phase retains D-090
  ExpectPrompt(0, std::array<std::int32_t, 3>{40, 41, 42});
}

TEST_F(FirstCaptureTest, CleanCaptureRefusalKeepsExactlyOneEagerExecution) {
  source_->options().refuse_record = true;
  for (std::uint32_t past = 0; past < 2; ++past) {
    auto ran = Verify(past);
    ASSERT_TRUE(ran) << ran.error().what;
    EXPECT_EQ(ran->path, en::RunPath::kEager);
    ASSERT_TRUE(runner_->slots().Accept(0, 1));
  }
  EXPECT_EQ(source_->executed(), 2U);
  EXPECT_EQ(runner_->graph_stats().refused, 1U);
  EXPECT_EQ(runner_->plans().graphs(), 0U);
  std::vector<llmp::memory::ReclaimCandidate> candidates;
  runner_->ReclaimCandidates(0, true, candidates);
  ASSERT_EQ(candidates.size(), 1U);
  EXPECT_EQ(candidates[0].kind, llmp::memory::ReclaimKind::kPlan);
  EXPECT_EQ(candidates[0].bytes, 4096U);  // no retained graph charge
}

TEST_F(FirstCaptureTest, UnknownCaptureDoesNotRetryOrCommitTheWarmRun) {
  source_->options().fail_record = kn::StepError::kUnknown;
  auto failed = Verify(0);
  ASSERT_FALSE(failed);
  EXPECT_FALSE(failed.error().clean);
  EXPECT_EQ(source_->queued(), 1U);  // only the warm execution; no fallback or replay
  EXPECT_FALSE(runner_->slots().usable(0));
  EXPECT_EQ(runner_->slots().positions(0), 0U);
}

class FirstCaptureDisabledTest : public FirstCaptureTest {
 protected:
  void Configure(en::RunnerConfig& config) override {
    FirstCaptureTest::Configure(config);
    config.capture_phases = 0;
  }
};

TEST_F(FirstCaptureDisabledTest, FirstUseMaskDoesNotOverrideDisabledPhase) {
  auto ran = Verify(0);
  ASSERT_TRUE(ran);
  EXPECT_EQ(ran->path, en::RunPath::kEager);
  EXPECT_EQ(runner_->plans().graphs(), 0U);
}

class FirstCapturePositionedTest : public FirstCaptureTest {
 protected:
  ts::FakeSourceOptions SourceOptions() override {
    auto options = FirstCaptureTest::SourceOptions();
    options.positioned = true;
    return options;
  }
};

TEST_F(FirstCapturePositionedTest, PositionedProgramStillRunsWithoutARecording) {
  auto ran = Verify(0);
  ASSERT_TRUE(ran);
  EXPECT_EQ(ran->path, en::RunPath::kEager);
  EXPECT_EQ(runner_->graph_stats().captured, 0U);
}

class FirstCaptureUnpaidTest : public FirstCaptureTest {
 protected:
  void Configure(en::RunnerConfig& config) override {
    FirstCaptureTest::Configure(config);
    node_.SetHostFloor(0);  // enable real catalog host charges, unlike the general fake fixture
  }
  ts::FakeSourceOptions SourceOptions() override {
    auto options = FirstCaptureTest::SourceOptions();
    options.launches = 1U << 20;  // graph exceeds the actual fake node's paid capacity
    return options;
  }
};

TEST_F(FirstCaptureUnpaidTest, RefusedGraphPaymentLeavesOneEagerExecution) {
  auto ran = Verify(0);
  ASSERT_TRUE(ran) << ran.error().what;
  EXPECT_EQ(ran->path, en::RunPath::kEager);
  EXPECT_EQ(source_->executed(), 1U);
  EXPECT_EQ(runner_->plans().graphs(), 0U);
  EXPECT_EQ(runner_->graph_stats().captured, 0U);
}

TEST_F(ModelRunnerTest, RecordingsAreReleasedBeforeTheLaunch) {
  for (std::uint32_t i = 0; i < 2; ++i) {
    ASSERT_TRUE(Run(Shape(pl::Phase::kDecode, {Seg(1, i, 1, 1)}), {{3}}).has_value());
  }
  ASSERT_EQ(runner_->plans().graphs(), 1U);
  std::array<en::PagedModel*, 1> models = {runner_.get()};
  auto closed = node_.TearDown(models);
  ASSERT_TRUE(closed) << llmp::test_support::Failed(closed).value_or("");
  EXPECT_EQ(log_, (std::vector<std::string>{"recording", "launch"}));
  EXPECT_EQ(runner_->plans().size(), 0U);
  runner_.reset();
}

// The kernels' own memory (RunnerConfig::kernel_memory): mapped at Setup
// before the measurements, pinned and charged as the runner's, handed to
// `prepare` in order; `retire` runs at Release after the recordings and
// the launch, before the memory goes.
class KernelMemoryTest : public ModelRunnerTest {
 protected:
  void Configure(en::RunnerConfig& config) override {
    config.kernel_memory = {{.name = "a kernel's workspace", .bytes = kExtent},
                            {.name = "another's", .bytes = (2 * kExtent) + 1}};
    config.prepare =
        [this](std::span<const std::uint64_t> bases) -> std::expected<void, std::string> {
      bases_.assign(bases.begin(), bases.end());
      measured_before_ = source_->built().size();
      log_.emplace_back("prepare");
      return {};
    };
    config.retire = [this] { log_.emplace_back("retire"); };
  }
  std::vector<std::uint64_t> bases_;
  std::size_t measured_before_ = 1;
};

TEST_F(KernelMemoryTest, IsMappedBeforeTheMeasurementsAndRetiredAfterTheLaunch) {
  ASSERT_EQ(bases_.size(), 2U);
  EXPECT_NE(bases_[0], 0U);
  EXPECT_NE(bases_[1], 0U);
  EXPECT_NE(bases_[0], bases_[1]);
  EXPECT_EQ(measured_before_, 0U);
  // Charged: the node's occupancy holds its four extents beside the rest.
  EXPECT_GE(node_.catalog().OccupancyOf(node_.domain()).Total().value(), fixed_ + (4 * kExtent));
  for (std::uint32_t i = 0; i < 2; ++i) {
    ASSERT_TRUE(Run(Shape(pl::Phase::kDecode, {Seg(1, i, 1, 1)}), {{3}}).has_value());
  }
  ASSERT_EQ(runner_->plans().graphs(), 1U);
  if (node_.InRequest(0)) EXPECT_TRUE(node_.EndRequest(0));
  std::array<en::PagedModel*, 1> models = {runner_.get()};
  auto closed = node_.TearDown(models);
  ASSERT_TRUE(closed) << llmp::test_support::Failed(closed).value_or("");
  EXPECT_EQ(log_, (std::vector<std::string>{"prepare", "recording", "launch", "retire"}));
  runner_.reset();
}

// A `prepare` that fails fails the Setup; what it made is still retired.
TEST(KernelMemory, AFailedPrepareIsStillRetired) {
  en::PagedNode node{
      en::NodeSettings{.compute_streams = 1, .slots = 2, .copy_lane = true, .slot_bytes = kExtent}};
  ASSERT_TRUE(node.Open());
  std::vector<std::string> log;
  ts::FakeSourceOptions o;
  for (const en::StageState& s : Stages()) o.layouts.push_back(s.layout);
  en::RunnerConfig config;
  // NOLINTNEXTLINE(concurrency-mt-unsafe): test environment is immutable
  const char* scratch = std::getenv("LLMP_TEST_SCRATCH");
  ASSERT_NE(scratch, nullptr);
  config.slots = {.model = "Fake",
                  .slots = kSlots,
                  .stages = Stages(),
                  .spec = {},
                  .read_align = 256,
                  .spill_place = {},
                  .spill_directory = scratch,
                  .invalidate = {},
                  .drop_plans = {}};
  config.setup_shapes = {Shape(pl::Phase::kDecode, {Seg(0, 0, 1, 1)})};
  config.kernel_memory = {{.name = "a kernel's workspace", .bytes = kExtent}};
  config.prepare = [&log](std::span<const std::uint64_t>) -> std::expected<void, std::string> {
    log.emplace_back("prepare");
    return std::unexpected("no handle");
  };
  config.retire = [&log] { log.emplace_back("retire"); };
  config.launch = []() -> std::expected<std::unique_ptr<kn::StepLaunch>, std::string> {
    return std::make_unique<kn::RecordingLaunch>();
  };
  auto runner = en::ModelRunner::Create(node, std::move(config),
                                        std::make_unique<ts::FakeStepSource>(std::move(o)), 0, 0);
  ASSERT_TRUE(runner.has_value()) << runner.error();
  auto set = (*runner)->Setup();
  ASSERT_FALSE(set);
  EXPECT_NE(llmp::test_support::Failed(set).value_or("").find("no handle"), std::string::npos);
  std::array<en::PagedModel*, 1> models = {runner->get()};
  auto closed = node.TearDown(models);
  EXPECT_TRUE(closed) << llmp::test_support::Failed(closed).value_or("");
  EXPECT_EQ(log, (std::vector<std::string>{"prepare", "retire"}));
}

TEST_F(ModelRunnerTest, ReclaimsGraphsAndPlansOutsideAStep) {
  for (std::uint32_t i = 0; i < 2; ++i) {
    ASSERT_TRUE(Run(Shape(pl::Phase::kDecode, {Seg(0, i, 1, 1)}), {{3}}).has_value());
  }
  std::vector<llmp::memory::ReclaimCandidate> candidates;
  runner_->ReclaimCandidates(0, true, candidates);
  ASSERT_EQ(candidates.size(), 2U);  // the plan (with its graph) and its graph
  const auto graph = std::ranges::find(candidates, llmp::memory::ReclaimKind::kGraph,
                                       &llmp::memory::ReclaimCandidate::kind);
  ASSERT_NE(graph, candidates.end());
  EXPECT_GT(runner_->Reclaim(llmp::memory::ReclaimKind::kGraph, graph->id), 0U);
  EXPECT_EQ(runner_->plans().graphs(), 0U);
  // Its next run captures again (it ran launch by launch before).
  auto again = Run(Shape(pl::Phase::kDecode, {Seg(0, 2, 1, 1)}), {{3}});
  ASSERT_TRUE(again.has_value());
  EXPECT_EQ(again->path, en::RunPath::kCaptured);
  candidates.clear();
  runner_->ReclaimCandidates(0, true, candidates);
  const auto plan = std::ranges::find(candidates, llmp::memory::ReclaimKind::kPlan,
                                      &llmp::memory::ReclaimCandidate::kind);
  ASSERT_NE(plan, candidates.end());
  EXPECT_GT(runner_->Reclaim(llmp::memory::ReclaimKind::kPlan, plan->id), 0U);
  EXPECT_EQ(runner_->plans().size(), 0U);
  const auto builds = source_->built().size();
  ASSERT_TRUE(Run(Shape(pl::Phase::kDecode, {Seg(0, 3, 1, 1)}), {{3}}).has_value());
  EXPECT_EQ(source_->built().size(), builds + 1);  // planned again
  EXPECT_EQ(runner_->plans().reclaimed_plans(), 1U);
  EXPECT_EQ(runner_->plans().reclaimed_graphs(), 1U);
}

TEST_F(ModelRunnerTest, StagesInputsBySlotAtTheirBoundaries) {
  en::StepRuns runs(true);
  std::vector<std::byte> staging(1024);
  runs.SetStaging(staging.data(), staging.size());
  const std::array<kn::InputSlot, 3> inputs = {
      kn::InputSlot{.device_address = 0x1000, .bytes = 10},
      kn::InputSlot{.device_address = 0x2000, .bytes = 300},
      kn::InputSlot{.device_address = 0x3000, .bytes = 4}};
  auto copies = runs.Stage(
      inputs,
      [](std::size_t slot, std::span<std::byte> out) -> std::expected<void, std::string> {
        std::ranges::fill(out, static_cast<std::byte>(slot + 1));
        return {};
      },
      0);
  ASSERT_TRUE(copies.has_value());
  ASSERT_EQ(copies->size(), 3U);
  EXPECT_EQ((*copies)[0], (en::RunCopy{0x1000, 10, 0}));
  EXPECT_EQ((*copies)[1], (en::RunCopy{0x2000, 300, 256}));
  EXPECT_EQ((*copies)[2], (en::RunCopy{0x3000, 4, 768}));
  EXPECT_EQ(staging[256], std::byte{2});
  EXPECT_EQ(staging[768], std::byte{3});
  const std::array<kn::InputSlot, 2> wide = {kn::InputSlot{.bytes = 800},
                                             kn::InputSlot{.bytes = 300}};
  EXPECT_FALSE(runs.Layout(wide, 0).has_value());
}

class CoverageTest : public ModelRunnerTest {
 protected:
  ts::FakeSourceOptions SourceOptions() override {
    auto o = ModelRunnerTest::SourceOptions();
    o.stray_use = true;
    return o;
  }
};

TEST_F(CoverageTest, RefusesAPlanThatBindsUncatalogedMemory) {
  auto ran = Run(Shape(pl::Phase::kDecode, {Seg(0, 0, 1, 1)}), {{3}});
  ASSERT_FALSE(ran.has_value());
  EXPECT_NE(ran.error().what.find("coverage"), std::string::npos);
  EXPECT_TRUE(ran.error().clean);
  EXPECT_EQ(runner_->plans().size(), 0U);
  EXPECT_EQ(source_->queued(), 0U);
}

class StrayOutputTest : public ModelRunnerTest {
 protected:
  ts::FakeSourceOptions SourceOptions() override {
    auto o = ModelRunnerTest::SourceOptions();
    o.stray_output = true;
    return o;
  }
};

TEST_F(StrayOutputTest, RefusesAPlanWhoseSlotsNameAnotherShapesSegments) {
  // An output naming a segment the shape lacks would be another owner's.
  auto ran = Run(Shape(pl::Phase::kDecode, {Seg(0, 0, 1, 1)}), {{3}});
  ASSERT_FALSE(ran.has_value());
  EXPECT_NE(ran.error().what.find("names no segment"), std::string::npos);
  EXPECT_TRUE(ran.error().clean);
  EXPECT_EQ(runner_->plans().size(), 0U);
  EXPECT_EQ(source_->queued(), 0U);
}

TEST_F(ModelRunnerTest, ARefusedBuildOrBindLeavesItsOwnerUsable) {
  const auto shape = Shape(pl::Phase::kDecode, {Seg(0, 0, 1, 1)});
  source_->options().refuse_build = [](const pl::StepShape&) { return true; };
  auto built = Run(shape, {{3}});
  ASSERT_FALSE(built.has_value());
  EXPECT_TRUE(built.error().clean);
  EXPECT_FALSE(built.error().slot.has_value());
  source_->options().refuse_build = {};
  source_->options().refuse_bind = true;
  auto bound = Run(shape, {{3}});
  ASSERT_FALSE(bound.has_value());
  EXPECT_TRUE(bound.error().clean);
  EXPECT_NE(bound.error().what.find("binding"), std::string::npos);
  EXPECT_EQ(runner_->plans().size(), 0U);
  EXPECT_EQ(source_->queued(), 0U);
  EXPECT_EQ(runner_->slots().positions(0), 0U);
  EXPECT_TRUE(runner_->slots().usable(0));
  source_->options().refuse_bind = false;
  auto ran = Run(shape, {{3}});
  ASSERT_TRUE(ran.has_value()) << ran.error().what;
  EXPECT_EQ(runner_->slots().positions(0), 1U);
  ExpectPrompt(0, std::vector<std::int32_t>{3});
}

TEST_F(ModelRunnerTest, ABorrowedOwnerIsRefusedCleanlyAndStaysUsable) {
  // Its state may not grow while borrowed: refused before anything ran,
  // clean and its own, not a capacity refusal.
  auto borrow = runner_->slots().Borrow(1);
  ASSERT_TRUE(borrow);
  auto ran = Run(Shape(pl::Phase::kDecode, {Seg(1, 0, 1, 1)}), {{3}});
  ASSERT_FALSE(ran.has_value());
  EXPECT_TRUE(ran.error().clean);
  EXPECT_EQ(ran.error().slot, 1U);
  EXPECT_FALSE(ran.error().state_refused);
  EXPECT_TRUE(runner_->slots().usable(1));
  borrow->Return();
  ASSERT_TRUE(Run(Shape(pl::Phase::kDecode, {Seg(1, 0, 1, 1)}), {{3}}).has_value());
}

TEST_F(ModelRunnerTest, AJobRefusedOnceItsCohortFaultedIsNotClean) {
  // Slot 2 holds state. A joined step of slots 0 and 1 is admitted; then,
  // while its plan builds, the request ends, slot 2's extent is evicted and
  // the budget filled, so the job's own materialization is refused before
  // anything runs. Nothing was written, but three active slots ran without
  // a held request: CheckFailedJob faults the cohort, so the refusal must
  // not invite a retry.
  ASSERT_TRUE(Run(Shape(pl::Phase::kDecode, {Seg(2, 0, 1, 1)}), {{5}}).has_value());
  bool armed = true;
  source_->options().on_build = [&](const pl::StepShape&, bool measuring) {
    if (measuring || !armed) return;
    armed = false;
    ASSERT_TRUE(node_.EndRequest(0));
    const auto extents = runner_->slots().live(2, 0).extents();
    ASSERT_FALSE(extents.empty());
    auto evicted = node_.Evict({extents.front()});
    ASSERT_TRUE(evicted) << llmp::test_support::Failed(evicted).value_or("");
    Leave(0);
  };
  auto ran = Run(Shape(pl::Phase::kDecode, {Seg(0, 0, 1, 1), Seg(1, 0, 1, 1)}), {{1}, {2}});
  ASSERT_FALSE(ran.has_value());
  EXPECT_FALSE(armed);
  EXPECT_TRUE(runner_->slots().faulted());
  EXPECT_FALSE(ran.error().clean) << ran.error().what;
  EXPECT_FALSE(ran.error().slot.has_value());
  EXPECT_EQ(source_->queued(), 1U);  // slot 2's step alone
  EXPECT_EQ(runner_->slots().positions(0), 0U);
}

TEST_F(ModelRunnerTest, AReclaimDuringAStepSparesWhatTheStepHolds) {
  const auto decode = [](std::uint8_t slot, std::uint32_t past) {
    return Shape(pl::Phase::kDecode, {Seg(slot, past, 1, 1)});
  };
  // Slot 0's decode plan with its recording, slot 2's decode plan, and
  // slot 1's four-row prefill plan.
  ASSERT_TRUE(Run(decode(0, 0), {{1}}).has_value());
  ASSERT_TRUE(Run(decode(0, 1), {{2}}).has_value());
  ASSERT_TRUE(Run(decode(2, 0), {{5}}).has_value());
  ASSERT_TRUE(Run(Shape(pl::Phase::kPrefill, {Seg(1, 0, 4, 1)}), {{1, 2, 3, 4}}).has_value());
  ASSERT_EQ(runner_->plans().size(), 3U);
  ASSERT_EQ(runner_->plans().graphs(), 1U);
  // Slot 0's third decode replays; its near future (slot 1's decode) is
  // built in its job's meanwhile and its far one (slot 2's) is cached.
  // While the future builds, everything reclaimable is reclaimed: only
  // the prefill plan, which the step does not hold, goes.
  std::uint64_t reclaimed = 0;
  bool reclaiming = false;
  source_->options().on_build = [&](const pl::StepShape&, bool measuring) {
    if (measuring || reclaiming) return;
    reclaiming = true;
    std::vector<llmp::memory::ReclaimCandidate> candidates;
    runner_->ReclaimCandidates(0, true, candidates);
    for (const auto& c : candidates) reclaimed += runner_->Reclaim(c.kind, c.id);
  };
  en::StepAhead ahead;
  ahead.shapes[0] = decode(1, 4);
  ahead.shapes[1] = decode(2, 1);
  ahead.build = 2;
  auto ran = Run(decode(0, 2), {{3}}, ahead);
  ASSERT_TRUE(ran.has_value()) << ran.error().what;
  EXPECT_TRUE(reclaiming);
  EXPECT_EQ(ran->path, en::RunPath::kReplayed);
  EXPECT_EQ(ran->built_ahead, 1U);
  EXPECT_EQ(ran->cached_ahead, 1U);
  const auto rows = Rows(ran->outputs.at(0));
  ASSERT_EQ(rows.size(), 1U);
  EXPECT_EQ(rows[0].position, 2);
  EXPECT_EQ(rows[0].cells, 3);
  EXPECT_GT(reclaimed, 0U);
  EXPECT_EQ(runner_->plans().reclaimed_plans(), 1U);
  EXPECT_EQ(runner_->plans().reclaimed_graphs(), 0U);
  EXPECT_EQ(runner_->plans().size(), 3U);    // slot 0's, slot 2's and the built future
  EXPECT_EQ(runner_->plans().graphs(), 1U);  // slot 0's recording kept
  // Both futures run without planning again.
  const auto builds = source_->built().size();
  ASSERT_TRUE(Run(decode(1, 4), {{9}}).has_value());
  ASSERT_TRUE(Run(decode(2, 1), {{9}}).has_value());
  EXPECT_EQ(source_->built().size(), builds);
}

TEST_F(ModelRunnerTest, LookaheadUsesDerivedDimensionsForBuildAndCaptureIdentity) {
  source_->options().shape_parameters = [](const pl::StepShape& shape) {
    return std::vector<std::uint64_t>{shape.segments.front().past / 2};
  };
  const auto shape = [](std::uint32_t past) {
    return Shape(pl::Phase::kDecode, {Seg(0, past, 1, 1)});
  };
  ASSERT_EQ(shape(0), shape(2));
  en::StepAhead ahead;
  ahead.shapes = {shape(2), shape(3)};  // one future dimension, unlike the current one
  ahead.build = 2;
  ahead.capture_ahead = true;
  auto first = Run(shape(0), {{4}}, ahead);
  ASSERT_TRUE(first) << first.error().what;
  EXPECT_EQ(runner_->lookahead_stats().built, 1U);
  EXPECT_EQ(runner_->lookahead_stats().captured_first, 0U);
  EXPECT_EQ(runner_->plans().size(), 2U);
  const auto built = source_->built().size();
  ASSERT_TRUE(Run(shape(1), {{5}}, ahead));
  EXPECT_EQ(source_->built().size(), built);
  auto next = Run(shape(2), {{6}});
  ASSERT_TRUE(next) << next.error().what;
  EXPECT_EQ(source_->built().size(), built);
  EXPECT_EQ(runner_->plans().size(), 2U);
  EXPECT_EQ(Rows(next->outputs.at(0))[0].position, 2);
}

class SideHistoryTest : public ModelRunnerTest {
 protected:
  ts::FakeSourceOptions SourceOptions() override {
    auto o = ModelRunnerTest::SourceOptions();
    o.extra_inputs = [](const pl::StepShape& shape) {
      std::uint64_t rows = 0;
      for (const auto& segment : shape.segments) rows += segment.rows;
      return std::vector<kn::InputSlot>{
          {.bytes = rows * 8, .kind = kn::InputKind::kSideRows, .component = 17}};
    };
    o.side_rows = [](std::uint16_t stage, std::uint16_t component,
                     std::span<const std::int32_t> history, std::uint32_t position,
                     std::span<std::int32_t> out) -> en::Status {
      if (stage != 0 || component != 17 || out.size() != 2 || position >= history.size())
        return std::unexpected("wrong lookup coordinates");
      out[0] = history[position];
      out[1] = position == 0 ? 0 : history[position - 1];
      return {};
    };
    return o;
  }
};

TEST_F(SideHistoryTest, LookupsUseEachOwnersHistoryAndNeverReplacePartialExplicitRows) {
  const auto shape = Shape(pl::Phase::kPrefill, {Seg(0, 0, 2, 1), Seg(1, 0, 2, 1)});
  const std::array<std::int32_t, 3> a{31, 32, 33}, b{81, 82, 83};
  const std::array work{en::SegmentWork{.tokens = std::span(a).first(2), .history = a},
                        en::SegmentWork{.tokens = std::span(b).first(2), .history = b}};
  const auto run = [&](std::span<const std::int32_t> explicit_rows) {
    return runner_->RunStep(
        {.shape = shape, .work = work, .side_rows = explicit_rows, .ahead = {}});
  };
  const std::array<std::int32_t, 1> incomplete{99};
  const auto executed = source_->executed();
  const auto initialized = runner_->slots().used_bytes(0);
  auto refused = run(incomplete);
  ASSERT_FALSE(refused);
  EXPECT_TRUE(refused.error().clean);
  EXPECT_EQ(source_->executed(), executed);
  EXPECT_EQ(runner_->slots().used_bytes(0), initialized);
  const std::array<std::int32_t, 8> invalid{1, 2, 3, 4, 5, 6, 7, -1};
  EXPECT_FALSE(run(invalid));
  EXPECT_EQ(runner_->slots().used_bytes(0), initialized);
  const auto fresh = Shape(pl::Phase::kPrefill, {Seg(2, 0, 2, 1)});
  const auto fresh_used = runner_->slots().used_bytes(2);
  for (const auto history : {std::span<const std::int32_t>{}, std::span<const std::int32_t>(b)}) {
    en::SegmentWork bad_history{.tokens = std::span(a).first(2), .history = history};
    auto bad = runner_->RunStep({.shape = fresh, .work = std::span(&bad_history, 1), .ahead = {}});
    ASSERT_FALSE(bad);
    EXPECT_TRUE(bad.error().clean);
    EXPECT_EQ(runner_->slots().used_bytes(2), fresh_used);
    EXPECT_EQ(source_->executed(), executed);
  }
  ASSERT_TRUE(run({}));
  auto bytes = source_->options().seen_inputs->at(0);
  std::vector<std::int32_t> ids(bytes.size() / 4);
  std::memcpy(ids.data(), bytes.data(), bytes.size());
  EXPECT_EQ(ids, (std::vector<std::int32_t>{31, 0, 32, 31, 81, 0, 82, 81}));
  const auto next = Shape(pl::Phase::kDecode, {Seg(0, 2, 1, 1)});
  en::SegmentWork mismatch{.tokens = std::span(a).last(1), .history = b};
  const auto after = source_->executed();
  auto bad = runner_->RunStep({.shape = next, .work = std::span(&mismatch, 1), .ahead = {}});
  ASSERT_FALSE(bad);
  EXPECT_TRUE(bad.error().clean);
  EXPECT_EQ(source_->executed(), after);
  EXPECT_EQ(runner_->slots().positions(0), 2U);
}

TEST_F(SideHistoryTest, DuplicateComponentSlotsRefuseBeforeLookupOrStateAdmission) {
  const auto joined = source_->options().extra_inputs;
  source_->options().extra_inputs = [](const pl::StepShape& shape) {
    std::vector<kn::InputSlot> inputs;
    for (std::size_t i = 0; i < shape.segments.size(); ++i)
      inputs.push_back({.bytes = std::uint64_t{shape.segments[i].rows} * 8,
                        .segment = static_cast<std::uint8_t>(i),
                        .kind = kn::InputKind::kSideRows,
                        .component = 17});
    return inputs;
  };
  const auto lookup = source_->options().side_rows;
  std::uint32_t lookups = 0;
  source_->options().side_rows = [&](auto stage, auto component, auto history, auto pos, auto ids) {
    ++lookups;
    return lookup(stage, component, history, pos, ids);
  };
  runner_->DropPlans();
  const auto shape = Shape(pl::Phase::kPrefill, {Seg(0, 0, 1, 1), Seg(1, 0, 1, 1)});
  const std::array<std::int32_t, 1> a{5}, b{9};
  const std::array work{en::SegmentWork{.tokens = a, .history = a},
                        en::SegmentWork{.tokens = b, .history = b}};
  const auto before0 = runner_->slots().used_bytes(0), before1 = runner_->slots().used_bytes(1);
  const auto ran = runner_->RunStep({.shape = shape, .work = work, .ahead = {}});
  ASSERT_FALSE(ran);
  EXPECT_TRUE(ran.error().clean);
  EXPECT_EQ(ran.error().what, "side-table components require one joined row input");
  EXPECT_EQ(lookups, 0U);
  EXPECT_EQ(source_->queued(), 0U);
  EXPECT_EQ(runner_->slots().used_bytes(0), before0);
  EXPECT_EQ(runner_->slots().used_bytes(1), before1);
  // The same component and owners succeed with one joined row slot.
  source_->options().extra_inputs = joined;
  runner_->DropPlans();
  EXPECT_TRUE(runner_->RunStep({.shape = shape, .work = work, .ahead = {}}));
  EXPECT_EQ(lookups, 2U);
}

class CompressorInputsTest : public ModelRunnerTest {
 protected:
  std::vector<pl::StepShape> SetupShapes() override {
    return {Shape(pl::Phase::kPrefill, {Seg(0, 0, 2, 1)})};
  }
  ts::FakeSourceOptions SourceOptions() override {
    auto o = ModelRunnerTest::SourceOptions();
    o.extra_inputs = [](const pl::StepShape&) {
      return std::vector<kn::InputSlot>{
          {.bytes = 8, .kind = kn::InputKind::kPositionPhases, .ratio = 4},
          {.bytes = 8,
           .segment = 0,
           .kind = kn::InputKind::kKeptRows,
           .cells = {.capacity = 4, .ring = true, .recent = true}},
          {.bytes = 16,
           .segment = 0,
           .kind = kn::InputKind::kKeptCells,
           .cells = {.capacity = 4, .ring = true, .recent = true}},
          {.bytes = 16,
           .segment = 0,
           .kind = kn::InputKind::kPoolReads,
           .cells = {.capacity = 4, .ring = true, .recent = true},
           .ratio = 4,
           .ring = 4},
          {.bytes = 8,
           .segment = 0,
           .kind = kn::InputKind::kBlockCells,
           .cells = {.capacity = 16, .block = 4, .positions = true},
           .ratio = 4},
          {.bytes = 4, .segment = 0, .kind = kn::InputKind::kBlockStarts, .ratio = 4},
          {.bytes = 16, .kind = kn::InputKind::kHadamard, .read = 2},
          {.bytes = 4 * 32 * 2,
           .segment = 0,
           .kind = kn::InputKind::kMask,
           .cells = {.capacity = 4, .ring = true, .recent = true},
           .read = 4,
           .window = 3,
           .mask_rows = 32,
           .ahead = true},
          {.bytes = 16 * 32 * 2,
           .segment = 0,
           .kind = kn::InputKind::kMask,
           .cells = {.capacity = 16, .block = 4, .positions = true},
           .read = 16,
           .mask_rows = 32},
          {.bytes = 8,
           .segment = 0,
           .kind = kn::InputKind::kVisibleCells,
           .cells = {.capacity = 16, .block = 4, .positions = true},
           .read = 16,
           .visible_capacity = 3}};
    };
    return o;
  }
};

TEST_F(CompressorInputsTest, TypedCompressorInputsAndAheadMasksRefreshOnReusedPlans) {
  const auto values = [&]<typename T>(std::size_t at) {
    const auto& bytes = source_->options().seen_inputs->at(at);
    std::vector<T> out(bytes.size() / sizeof(T));
    std::memcpy(out.data(), bytes.data(), bytes.size());
    return out;
  };
  ASSERT_TRUE(Run(Shape(pl::Phase::kPrefill, {Seg(0, 0, 2, 1)}), {{3, 4}}));
  EXPECT_EQ(values.template operator()<std::int32_t>(0), (std::vector<std::int32_t>{0, 1}));
  EXPECT_EQ(values.template operator()<std::int32_t>(1), (std::vector<std::int32_t>{0, 1}));
  EXPECT_EQ(values.template operator()<std::int64_t>(2), (std::vector<std::int64_t>{0, 1}));
  EXPECT_EQ(values.template operator()<std::int32_t>(3), (std::vector<std::int32_t>{4, 4, 4, 4}));
  EXPECT_EQ(values.template operator()<std::int64_t>(4), (std::vector<std::int64_t>{15}));
  EXPECT_EQ(values.template operator()<std::int32_t>(5), (std::vector<std::int32_t>{0}));
  const float h = 1.0F / std::sqrt(2.0F);
  EXPECT_EQ(values.template operator()<float>(6), (std::vector<float>{h, h, h, -h}));
  auto mask = values.template operator()<std::uint16_t>(7);
  EXPECT_EQ(mask[0], 0U);
  EXPECT_EQ(mask[1], 0U);  // row 0 sees future row 1
  EXPECT_EQ(mask[2], 0xFC00U);
  EXPECT_EQ(mask[8], 0xFC00U);  // padded query rows stay hidden
  auto compressed = values.template operator()<std::uint16_t>(8);
  EXPECT_TRUE(std::ranges::all_of(compressed, [](auto x) { return x == 0xFC00U; }));
  ASSERT_TRUE(Run(Shape(pl::Phase::kPrefill, {Seg(0, 2, 2, 1)}), {{5, 6}}));
  EXPECT_EQ(values.template operator()<std::int32_t>(0), (std::vector<std::int32_t>{2, 3}));
  EXPECT_EQ(values.template operator()<std::int64_t>(2), (std::vector<std::int64_t>{2, 3}));
  EXPECT_EQ(values.template operator()<std::int32_t>(3), (std::vector<std::int32_t>{0, 1, 4, 5}));
  EXPECT_EQ(values.template operator()<std::int64_t>(4), (std::vector<std::int64_t>{0}));
  mask = values.template operator()<std::uint16_t>(7);
  EXPECT_EQ(mask[3], 0U);       // row 2 sees future row 3
  EXPECT_EQ(mask[4], 0xFC00U);  // row 3 cannot see position 0 outside its window
  compressed = values.template operator()<std::uint16_t>(8);
  EXPECT_EQ(compressed[0], 0xFC00U);  // row 2 cannot see the block completed by row 3
  EXPECT_EQ(compressed[16], 0U);
  EXPECT_EQ(compressed[17], 0xFC00U);  // no padded/unfinished compressed cell is visible
  EXPECT_EQ(compressed[31], 0xFC00U);  // the last dummy cell is never visible
  EXPECT_EQ(compressed[32], 0xFC00U);  // no padded query row is visible
  EXPECT_EQ(values.template operator()<std::int32_t>(9), (std::vector<std::int32_t>{0, 1}));
  for (std::uint32_t past = 4; past <= 14; past += 2) {
    const auto ran = Run(Shape(pl::Phase::kPrefill, {Seg(0, past, 2, 1)}), {{7, 8}});
    ASSERT_TRUE(ran) << (ran ? "" : ran.error().what);
  }
  EXPECT_EQ(values.template operator()<std::int32_t>(9), (std::vector<std::int32_t>{3, 3}));
  EXPECT_EQ(runner_->plans().size(), 1U);
}

class VisibleCellsRunnerTest : public ModelRunnerTest {
 protected:
  ts::FakeSourceOptions SourceOptions() override {
    auto options = ModelRunnerTest::SourceOptions();
    options.extra_inputs = [](const pl::StepShape&) {
      return std::vector<kn::InputSlot>{{.bytes = 4,
                                         .segment = 0,
                                         .kind = kn::InputKind::kVisibleCells,
                                         .cells = {.capacity = 256, .block = 4, .positions = true},
                                         .read = 2,
                                         .visible_capacity = 3}};
    };
    return options;
  }
};

TEST_F(VisibleCellsRunnerTest, CountsRefreshBeforeTheSameCapturedGraphReplays) {
  for (std::uint32_t past = 0; past < 13; ++past) {
    auto ran = Run(Shape(pl::Phase::kDecode, {Seg(0, past, 1, 1)}), {{7}});
    ASSERT_TRUE(ran) << (ran ? "" : ran.error().what);
    if (past >= 2) EXPECT_EQ(ran->path, en::RunPath::kReplayed);
    const auto& bytes = source_->options().seen_inputs->at(0);
    ASSERT_EQ(bytes.size(), sizeof(std::int32_t));
    std::int32_t count = -1;
    std::memcpy(&count, bytes.data(), bytes.size());
    EXPECT_EQ(count, static_cast<std::int32_t>(std::min((past + 1) / 4, 2U)));
  }
  EXPECT_EQ(runner_->plans().size(), 1U);
  EXPECT_EQ(runner_->plans().graphs(), 1U);
}

class PositionedTest : public ModelRunnerTest {
 protected:
  ts::FakeSourceOptions SourceOptions() override {
    auto o = ModelRunnerTest::SourceOptions();
    o.positioned = true;
    return o;
  }
};

TEST_F(PositionedTest, AProgramThatTakesItsPositionsIsNeverRecorded) {
  // The program writes at the positions Position gave it: every run must
  // be launch by launch, positioned for that step, never a stale replay.
  std::vector<std::int32_t> all;
  for (std::uint32_t i = 0; i < 5; ++i) {
    const auto token = static_cast<std::int32_t>(60 + i);
    all.push_back(token);
    en::StepAhead ahead;
    ahead.shapes[0] = Shape(pl::Phase::kDecode, {Seg(0, i + 1, 1, 1)});
    ahead.capture_ahead = true;
    auto ran = Run(Shape(pl::Phase::kDecode, {Seg(0, i, 1, 1)}), {{token}}, ahead);
    ASSERT_TRUE(ran.has_value()) << ran.error().what;
    EXPECT_EQ(ran->path, en::RunPath::kEager);
    const auto rows = Rows(ran->outputs.at(0));
    ASSERT_EQ(rows.size(), 1U);
    EXPECT_EQ(rows[0].position, static_cast<std::int32_t>(i));
    EXPECT_EQ(rows[0].cells, static_cast<std::int32_t>(i + 1));
  }
  EXPECT_EQ(source_->positioned(), 5U);
  EXPECT_EQ(runner_->plans().size(), 1U);
  EXPECT_EQ(runner_->plans().graphs(), 0U);
  EXPECT_EQ(runner_->graph_stats().captured, 0U);
  EXPECT_EQ(runner_->graph_stats().refused, 0U);
  ExpectPrompt(0, all);
}

// ------------------------------------------------------------ composition

md::Batching Caps() {
  return {.max_segments = 4,
          .max_segment_rows = 512,
          .max_step_rows = 16,
          .group_rows = 0,
          .group_segments = 0,
          .invariant_group_segments = 0,
          .joined_read_align = 2048,
          .common_read = false,
          .join_prefill = false,
          .equal_prefill = false,
          .mixed_heads = false};
}

en::PendingWork Decode(std::uint32_t slot, std::uint32_t past, std::uint32_t rows = 1) {
  return {.slot = slot, .past = past, .rows = rows, .outputs = rows};
}

en::PendingWork Prefill(std::uint32_t slot, std::uint32_t past, std::uint32_t rows,
                        bool head = true) {
  return {.slot = slot,
          .phase = pl::Phase::kPrefill,
          .past = past,
          .rows = rows,
          .outputs = head ? 1U : 0U,
          .want_head = head};
}

TEST(WaveComposerTest, JoinsByTheCapabilityAlone) {
  auto composer = en::WaveComposer::Make(Caps(), kContext, 256);
  ASSERT_TRUE(composer.has_value());
  // Five decoding owners, at most four a step: two steps, a joined one at
  // the joined alignment and a lone one at the owner's.
  const std::vector<en::PendingWork> pending = {Decode(3, 10), Decode(0, 300), Decode(1, 5),
                                                Decode(2, 7), Decode(4, 9)};
  const auto c = composer->Compose(pending);
  ASSERT_TRUE(c.refused.empty());
  ASSERT_EQ(c.steps.size(), 2U);
  const auto& joined = c.steps[0].shape;
  ASSERT_EQ(joined.segments.size(), 4U);
  EXPECT_EQ(joined.read_align, 2048U);
  EXPECT_EQ(joined.segments[0].slot, 0U);              // slots ascending
  EXPECT_EQ(joined.segments[0].read_cells, kContext);  // 2,048 clamped to the context
  EXPECT_EQ(c.steps[0].work, (std::vector<std::size_t>{1, 2, 3, 0}));
  EXPECT_EQ(c.steps[1].shape.read_align, 256U);
  EXPECT_EQ(c.steps[1].shape.segments[0].read_cells, 256U);
  // A step's rows cap: 8 + 8 fit 16, the third goes alone.
  const auto rows = composer->Compose(
      std::vector<en::PendingWork>{Decode(0, 0, 8), Decode(1, 0, 8), Decode(2, 0, 8)});
  ASSERT_EQ(rows.steps.size(), 2U);
  EXPECT_EQ(rows.steps[0].shape.rows(), 16U);
}

TEST(WaveComposerTest, GroupsReadsAndPromptJoins) {
  auto caps = Caps();
  caps.max_segments = 8;
  caps.max_step_rows = 0;
  caps.group_rows = 4;
  caps.group_segments = 3;
  caps.invariant_group_segments = 2;
  caps.common_read = true;
  caps.joined_read_align = 0;
  auto composer = en::WaveComposer::Make(caps, kContext, 256);
  ASSERT_TRUE(composer.has_value());
  const auto c = composer->Compose(std::vector<en::PendingWork>{
      Decode(0, 0, 2), Decode(1, 300, 1), Decode(2, 0, 1), Decode(3, 0, 1), Decode(4, 0, 4)});
  ASSERT_EQ(c.steps.size(), 1U);
  const auto& s = c.steps[0].shape.segments;
  // Rows 2+1+1 fit four and three owners fill the group; 1+4 rows pass
  // four, so the last opens a third.
  EXPECT_EQ(s[0].group, 0U);
  EXPECT_EQ(s[1].group, 0U);
  EXPECT_EQ(s[2].group, 0U);
  EXPECT_EQ(s[3].group, 1U);
  EXPECT_EQ(s[4].group, 2U);
  for (const auto& seg : s) EXPECT_EQ(seg.read_cells, 512U);  // the common read
  // A row-invariant step takes the invariant group cap.
  std::vector<en::PendingWork> verify = {Decode(0, 0), Decode(1, 0), Decode(2, 0)};
  for (auto& v : verify) {
    v.phase = pl::Phase::kVerify;
    v.row_invariant = true;
  }
  const auto inv = composer->Compose(verify);
  ASSERT_EQ(inv.steps.size(), 1U);
  EXPECT_EQ(inv.steps[0].shape.segments[2].group, 1U);
  // Prompts join only when the capability says so, with equal rows and
  // reads when it asks, and never across head intents.
  EXPECT_EQ(composer->Compose(std::vector{Prefill(0, 0, 64), Prefill(1, 0, 64)}).steps.size(), 2U);
  caps.join_prefill = true;
  caps.equal_prefill = true;
  composer = en::WaveComposer::Make(caps, kContext, 256);
  ASSERT_TRUE(composer.has_value());
  EXPECT_EQ(composer->Compose(std::vector{Prefill(0, 0, 64), Prefill(1, 0, 64)}).steps.size(), 1U);
  EXPECT_EQ(composer->Compose(std::vector{Prefill(0, 0, 64), Prefill(1, 0, 32)}).steps.size(), 2U);
  EXPECT_EQ(composer->Compose(std::vector{Prefill(0, 0, 64), Prefill(1, 256, 64)}).steps.size(),
            2U);
  EXPECT_EQ(
      composer->Compose(std::vector{Prefill(0, 0, 64), Prefill(1, 0, 64, false)}).steps.size(), 2U);
}

TEST(WaveComposerTest, AnOwnerSteppingAloneLeavesItsPeersJoined) {
  auto composer = en::WaveComposer::Make(Caps(), kContext, 256);
  ASSERT_TRUE(composer.has_value());
  auto alone = Decode(1, 0);
  alone.alone = true;
  const auto c = composer->Compose(std::vector<en::PendingWork>{Decode(0, 0), alone, Decode(2, 0)});
  ASSERT_EQ(c.steps.size(), 2U);
  EXPECT_EQ(c.steps[0].work, (std::vector<std::size_t>{0, 2}));
  EXPECT_EQ(c.steps[0].shape.read_align, 2048U);
  EXPECT_EQ(c.steps[1].work, (std::vector<std::size_t>{1}));
  EXPECT_EQ(c.steps[1].shape.read_align, 256U);
}

TEST(WaveComposerTest, RefusesMalformedWorkAlone) {
  auto composer = en::WaveComposer::Make(Caps(), kContext, 256);
  ASSERT_TRUE(composer.has_value());
  auto bad = Decode(2, kContext);
  const auto c = composer->Compose(std::vector<en::PendingWork>{Decode(0, 0), bad, Decode(0, 1),
                                                                Prefill(1, 0, 600), Decode(3, 0)});
  ASSERT_EQ(c.refused.size(), 3U);
  EXPECT_EQ(c.refused[0].first, 1U);  // outside the context
  EXPECT_EQ(c.refused[1].first, 2U);  // the slot's second work
  EXPECT_EQ(c.refused[2].first, 3U);  // past an owner's rows
  ASSERT_EQ(c.steps.size(), 1U);
  EXPECT_EQ(c.steps[0].work, (std::vector<std::size_t>{0, 4}));
  EXPECT_FALSE(en::WaveComposer::Make(Caps(), kContext, 100).has_value());
  auto too_many = Caps();
  too_many.max_segments = 17;
  EXPECT_FALSE(en::WaveComposer::Make(too_many, kContext, 256).has_value());
}

// ---------------------------------------------------------------- waves

class WaveTest : public ModelRunnerTest {
 protected:
  en::WaveReport Wave(const std::vector<en::PendingWork>& pending,
                      const std::vector<std::vector<std::int32_t>>& tokens,
                      md::Batching caps = Caps()) {
    auto composer = en::WaveComposer::Make(caps, kContext, 256);
    EXPECT_TRUE(composer.has_value());
    std::vector<en::SegmentWork> work;
    for (const auto& t : tokens) work.push_back({.tokens = t, .bytes = {}});
    return en::RunWave(*runner_, *composer, pending, work);
  }
};

TEST_F(WaveTest, EveryOwnerGetsItsOwnOutputs) {
  const auto report =
      Wave({Decode(0, 0, 2), Decode(1, 0, 3), Decode(2, 0, 1)}, {{10, 11}, {20, 21, 22}, {30}});
  EXPECT_EQ(report.steps, 1U);
  for (std::uint32_t slot = 0; slot < 3; ++slot) {
    const auto& outcome = report.outcomes[slot];
    ASSERT_FALSE(outcome.refusal.has_value()) << outcome.refusal->what;
    const auto rows = Rows(outcome.outputs);
    ASSERT_EQ(rows.size(), slot == 0 ? 2U : slot == 1 ? 3U : 1U);
    for (std::size_t r = 0; r < rows.size(); ++r) {
      EXPECT_EQ(rows[r].slot, static_cast<std::int32_t>(slot));
      EXPECT_EQ(rows[r].position, static_cast<std::int32_t>(r));
      EXPECT_EQ(rows[r].token, static_cast<std::int32_t>((10 * (slot + 1)) + r));
      EXPECT_EQ(rows[r].cells, static_cast<std::int32_t>(rows.size()));
    }
  }
  // The joined step read at the joined alignment; its state is backed there.
  ASSERT_FALSE(source_->built().empty());
  EXPECT_EQ(source_->built().back().shape.read_align, 2048U);
  ExpectPrompt(1, std::vector<std::int32_t>{20, 21, 22});
}

TEST_F(WaveTest, EachOwnerFindsItsOwnOutputsAndPlaceInTheStep) {
  // Greedy owners: each segment's own output is its chosen token (its last
  // token + 1000). Pending out of slot order, each outcome says where it
  // sat in the step.
  std::vector<en::PendingWork> pending = {Decode(2, 0, 1), Decode(0, 0, 2), Decode(1, 0, 3)};
  for (auto& p : pending) p.greedy = true;
  const auto report = Wave(pending, {{30}, {10, 11}, {20, 21, 22}});
  EXPECT_EQ(report.steps, 1U);
  const std::array<std::uint32_t, 3> segment = {2, 0, 1};
  const std::array<std::uint32_t, 3> before = {5, 0, 2};  // rows (and outputs) of earlier owners
  const std::array<std::int32_t, 3> last = {30, 11, 22};
  for (std::size_t i = 0; i < pending.size(); ++i) {
    const auto& outcome = report.outcomes[i];
    ASSERT_FALSE(outcome.refusal.has_value()) << outcome.refusal->what;
    EXPECT_EQ(outcome.segment, segment[i]);
    EXPECT_EQ(outcome.first_row, before[i]);
    EXPECT_EQ(outcome.first_output, before[i]);
    EXPECT_TRUE(outcome.joined.empty());
    ASSERT_EQ(outcome.outputs.size(), sizeof(std::int32_t));
    std::int32_t chosen = 0;
    std::memcpy(&chosen, outcome.outputs.data(), sizeof(chosen));
    EXPECT_EQ(chosen, last[i] + 1000);
  }
}

TEST_F(WaveTest, AComposerAtAnotherAlignmentThanTheStateIsRefused) {
  // Its lone reads would not be what the runner backs first for a joined
  // step's owners: nothing runs.
  auto composer = en::WaveComposer::Make(Caps(), kContext, 512);
  ASSERT_TRUE(composer.has_value());
  const std::vector<std::int32_t> one = {1}, two = {2};
  const std::vector<en::SegmentWork> work = {{.tokens = one, .bytes = {}},
                                             {.tokens = two, .bytes = {}}};
  const auto report =
      en::RunWave(*runner_, *composer, std::vector{Decode(0, 0), Decode(1, 0)}, work);
  EXPECT_EQ(report.steps, 0U);
  for (const auto& outcome : report.outcomes) {
    ASSERT_TRUE(outcome.refusal.has_value());
    EXPECT_NE(outcome.refusal->what.find("alignment"), std::string::npos);
  }
  EXPECT_EQ(source_->queued(), 0U);
  EXPECT_EQ(runner_->slots().used_bytes(0), 0U);
}

TEST_F(WaveTest, OneOwnersRefusalNeverFailsItsPeers) {
  // Slot 1 is at other positions, slot 2 quarantined: each refused alone.
  runner_->slots().live(2, 0).Quarantine();
  const auto report = Wave({Decode(0, 0), Decode(1, 5), Decode(2, 0)}, {{1}, {2}, {3}});
  ASSERT_FALSE(report.outcomes[0].refusal.has_value());
  ASSERT_TRUE(report.outcomes[1].refusal.has_value());
  EXPECT_EQ(report.outcomes[1].refusal->slot, 1U);
  ASSERT_TRUE(report.outcomes[2].refusal.has_value());
  EXPECT_EQ(report.outcomes[2].refusal->slot, 2U);
  EXPECT_EQ(Rows(report.outcomes[0].outputs).at(0).token, 1);
  EXPECT_EQ(runner_->slots().positions(0), 1U);
  EXPECT_EQ(runner_->slots().positions(1), 0U);
  EXPECT_GE(report.recomposed, 1U);
  EXPECT_TRUE(runner_->slots().usable(1));
}

TEST_F(WaveTest, ARefusedJoinedPlanRunsItsOwnersAlone) {
  source_->options().refuse_build = [](const pl::StepShape& shape) {
    return shape.segments.size() > 1;
  };
  const auto report = Wave({Decode(0, 0), Decode(1, 0), Decode(2, 0)}, {{1}, {2}, {3}});
  EXPECT_EQ(report.steps, 3U);
  for (std::uint32_t slot = 0; slot < 3; ++slot) {
    ASSERT_FALSE(report.outcomes[slot].refusal.has_value());
    EXPECT_EQ(Rows(report.outcomes[slot].outputs).at(0).token, static_cast<std::int32_t>(slot + 1));
  }
}

TEST_F(WaveTest, ACommonReadIsBackedForEveryOwner) {
  // Slot 0 holds 600 positions, slot 1 none. With a common read both read
  // 768 cells: slot 1's cells past 512 lie in its cache's second extent,
  // which must be backed before the step reads it.
  auto caps = Caps();
  caps.common_read = true;
  caps.joined_read_align = 0;
  ASSERT_TRUE(Run(Shape(pl::Phase::kPrefill, {Seg(0, 0, 512, 0)}), {Tokens(512, 1)}).has_value());
  ASSERT_TRUE(Run(Shape(pl::Phase::kPrefill, {Seg(0, 512, 88, 0)}), {Tokens(88, 513)}).has_value());
  const auto report = Wave({Decode(0, 600), Decode(1, 0)}, {{7}, {8}}, caps);
  EXPECT_EQ(report.steps, 1U);
  for (std::uint32_t slot = 0; slot < 2; ++slot) {
    ASSERT_FALSE(report.outcomes[slot].refusal.has_value()) << report.outcomes[slot].refusal->what;
  }
  EXPECT_EQ(source_->built().back().shape.segments[1].read_cells, 768U);
  EXPECT_EQ(Rows(report.outcomes[0].outputs).at(0).cells, 601);
  EXPECT_EQ(Rows(report.outcomes[1].outputs).at(0).cells, 1);
}

TEST_F(WaveTest, AJoinedStepThatFailedAfterQueueingFailsItsOwnersTogether) {
  // The program refuses after the step's input copies were queued: every
  // owner refused together (not clean), quarantined, its positions not
  // published, and no owner runs the step again alone.
  source_->options().fail_run = kn::StepError::kRejected;
  const auto report = Wave({Decode(0, 0), Decode(1, 0)}, {{1}, {2}});
  EXPECT_EQ(report.steps, 0U);
  EXPECT_EQ(report.recomposed, 0U);
  for (std::uint32_t slot = 0; slot < 2; ++slot) {
    ASSERT_TRUE(report.outcomes[slot].refusal.has_value());
    EXPECT_FALSE(report.outcomes[slot].refusal->clean);
    EXPECT_EQ(runner_->slots().positions(slot), 0U);
    EXPECT_FALSE(runner_->slots().usable(slot));
  }
  EXPECT_EQ(source_->queued(), 0U);
  EXPECT_TRUE(runner_->slots().usable(2));
}

TEST_F(WaveTest, AJoinedStepOfUnknownEffectFailsItsOwnersTogether) {
  source_->options().fail_run = kn::StepError::kUnknown;
  const auto report = Wave({Decode(0, 0), Decode(1, 0)}, {{1}, {2}});
  EXPECT_EQ(report.steps, 0U);
  EXPECT_EQ(report.recomposed, 0U);
  for (std::uint32_t slot = 0; slot < 2; ++slot) {
    ASSERT_TRUE(report.outcomes[slot].refusal.has_value());
    EXPECT_FALSE(report.outcomes[slot].refusal->clean);
    EXPECT_EQ(runner_->slots().positions(slot), 0U);
    EXPECT_FALSE(runner_->slots().usable(slot));
  }
  EXPECT_EQ(source_->queued(), 1U);  // the joined run, never an owner's alone
}

class CapacityTest : public WaveTest {
 protected:
  std::uint64_t BudgetExtents() const override { return 24; }
};

TEST_F(CapacityTest, AnOwnerOutOfStateCapacityIsRefusedAlone) {
  // Alone each owner needs two extents (its first cache extent and its
  // fixed state's): room for two owners' own, so the third's growth is
  // refused cleanly (alone it needs the same) and its peers run without it.
  Leave(5);
  const auto report = Wave({Decode(0, 0), Decode(1, 0), Decode(2, 0)}, {{1}, {2}, {3}});
  ASSERT_FALSE(report.outcomes[0].refusal.has_value()) << report.outcomes[0].refusal->what;
  ASSERT_FALSE(report.outcomes[1].refusal.has_value()) << report.outcomes[1].refusal->what;
  ASSERT_TRUE(report.outcomes[2].refusal.has_value());
  EXPECT_TRUE(report.outcomes[2].refusal->state_refused);
  EXPECT_EQ(report.outcomes[2].refusal->slot, 2U);
  EXPECT_TRUE(runner_->slots().state_refused(2));
  EXPECT_TRUE(runner_->slots().usable(2));
  EXPECT_EQ(Rows(report.outcomes[1].outputs).at(0).token, 2);
}

TEST_F(CapacityTest, AnOwnerOutOfCapacityAtTheJoinedAlignmentStepsAloneAtItsOwn) {
  // Room for eight extents: every owner's own two first (its first cache
  // extent and its fixed state), then slots 0 and 1 widen to 2,048 (a third
  // each), and slot 2's third does not fit; alone at 256 it needs only its
  // own two, and runs after its peers' joined step.
  Leave(8);
  const auto report = Wave({Decode(0, 0), Decode(1, 0), Decode(2, 0)}, {{1}, {2}, {3}});
  for (std::uint32_t slot = 0; slot < 3; ++slot) {
    ASSERT_FALSE(report.outcomes[slot].refusal.has_value()) << report.outcomes[slot].refusal->what;
    EXPECT_EQ(Rows(report.outcomes[slot].outputs).at(0).token, static_cast<std::int32_t>(slot + 1));
  }
  EXPECT_EQ(report.steps, 2U);
  EXPECT_EQ(source_->built().back().shape.read_align, 256U);
  EXPECT_EQ(source_->built().back().shape.segments.at(0).slot, 2U);
}

TEST_F(CapacityTest, AnOwnerOutOfCapacityForACommonReadStepsAloneAtItsOwn) {
  // Slot 0 holds 600 positions. Joined with it under a common read, slot 1
  // reads 768 cells (two cache extents and its fixed state's), where two
  // extents are left; alone it reads 256 (two extents) and runs. Refused
  // instead, it would be refused in every wave while slot 0 stays long.
  auto caps = Caps();
  caps.common_read = true;
  caps.joined_read_align = 0;
  ASSERT_TRUE(Run(Shape(pl::Phase::kPrefill, {Seg(0, 0, 512, 0)}), {Tokens(512, 1)}).has_value());
  ASSERT_TRUE(Run(Shape(pl::Phase::kPrefill, {Seg(0, 512, 88, 0)}), {Tokens(88, 513)}).has_value());
  Leave(2);
  const auto report = Wave({Decode(0, 600), Decode(1, 0)}, {{7}, {8}}, caps);
  for (std::uint32_t slot = 0; slot < 2; ++slot) {
    ASSERT_FALSE(report.outcomes[slot].refusal.has_value()) << report.outcomes[slot].refusal->what;
  }
  EXPECT_EQ(report.steps, 2U);
  EXPECT_EQ(Rows(report.outcomes[0].outputs).at(0).cells, 601);
  EXPECT_EQ(Rows(report.outcomes[1].outputs).at(0).cells, 1);
  EXPECT_EQ(source_->built().back().shape.segments.at(0).read_cells, 256U);
  EXPECT_EQ(runner_->slots().positions(1), 1U);
}

TEST_F(CapacityTest, EveryOwnersOwnCellsComeBeforeAPeersCommonRead) {
  // Slot 1 holds 512 positions: its next row reads 768 cells, its second
  // cache extent. Slot 0 holds none: alone it needs its first cache extent
  // and its fixed state; through slot 1's common read, its second too.
  // Three extents are left: both owners' own cells fit, slot 0's widening
  // does not. Widened first, slot 0 would take slot 1's extent and keep it,
  // and slot 1 would be refused in every wave while slot 0 lives.
  auto caps = Caps();
  caps.common_read = true;
  caps.joined_read_align = 0;
  ASSERT_TRUE(Run(Shape(pl::Phase::kPrefill, {Seg(1, 0, 512, 0)}), {Tokens(512, 1)}).has_value());
  Leave(3);
  for (std::uint32_t wave = 0; wave < 2; ++wave) {
    const auto report =
        Wave({Decode(0, runner_->slots().positions(0)), Decode(1, runner_->slots().positions(1))},
             {{7}, {8}}, caps);
    for (std::uint32_t slot = 0; slot < 2; ++slot) {
      EXPECT_FALSE(report.outcomes[slot].refusal.has_value())
          << "wave " << wave << " slot " << slot << ": " << report.outcomes[slot].refusal->what;
    }
    if (report.outcomes[1].refusal) continue;
    EXPECT_EQ(Rows(report.outcomes[1].outputs).at(0).cells, static_cast<std::int32_t>(513 + wave));
  }
  EXPECT_EQ(runner_->slots().positions(0), 2U);
  EXPECT_EQ(runner_->slots().positions(1), 514U);
}

TEST_F(CapacityTest, EveryOwnersOwnCellsComeBeforeAPeersJoinedAlignment) {
  // Two fresh owners: alone each needs two extents (its first cache extent
  // and its fixed state), at the joined 2,048 alignment three. Four are
  // left. Slot 0 backed through 2,048 first would leave slot 1 one extent,
  // too few even alone, in every wave while slot 0 lives.
  Leave(4);
  for (std::uint32_t wave = 0; wave < 2; ++wave) {
    const auto report =
        Wave({Decode(0, runner_->slots().positions(0)), Decode(1, runner_->slots().positions(1))},
             {{1}, {2}});
    for (std::uint32_t slot = 0; slot < 2; ++slot) {
      EXPECT_FALSE(report.outcomes[slot].refusal.has_value())
          << "wave " << wave << " slot " << slot << ": " << report.outcomes[slot].refusal->what;
    }
  }
  EXPECT_EQ(runner_->slots().positions(0), 2U);
  EXPECT_EQ(runner_->slots().positions(1), 2U);
}

TEST_F(CapacityTest, ARefusalAfterACapacityRefusalIsNotOneUnlessItIs) {
  // Slot 1 is refused for capacity, then borrowed: its next refusal is the
  // borrow's (clean, its own), not a stale capacity refusal a caller would
  // wait on memory for.
  Leave(1);
  auto refused = Run(Shape(pl::Phase::kDecode, {Seg(1, 0, 1, 1)}), {{3}});
  ASSERT_FALSE(refused.has_value());
  ASSERT_TRUE(refused.error().state_refused);
  auto borrow = runner_->slots().Borrow(1);
  ASSERT_TRUE(borrow);
  auto borrowed = Run(Shape(pl::Phase::kDecode, {Seg(1, 0, 1, 1)}), {{3}});
  borrow->Return();
  ASSERT_FALSE(borrowed.has_value());
  EXPECT_TRUE(borrowed.error().clean);
  EXPECT_EQ(borrowed.error().slot, 1U);
  EXPECT_FALSE(borrowed.error().state_refused);
  EXPECT_TRUE(runner_->slots().usable(1));
}

TEST_F(CapacityTest, AnOwnerShortForItsOwnRowsIsNotRetriedAlone) {
  // Alone each owner needs two extents; five are left. Slots 0 and 1 take
  // their own, slot 2's are refused: alone it needs the same two, so it is
  // refused at once rather than asking for them again alone. Slot 0 widens
  // to 2,048 (a third extent); slot 1's widening is refused and steps alone.
  std::uint32_t asked = 0;
  node_.SetReclaimer(
      [&](std::uint64_t, en::PagedNode::ReclaimFor what, std::span<const llmp::catalog::ExtentId>) {
        asked += what == en::PagedNode::ReclaimFor::kMaterialize ? 1 : 0;
        return std::uint64_t{0};
      });
  Leave(5);
  const auto report = Wave({Decode(0, 0), Decode(1, 0), Decode(2, 0)}, {{1}, {2}, {3}});
  node_.SetReclaimer({});
  ASSERT_FALSE(report.outcomes[0].refusal.has_value()) << report.outcomes[0].refusal->what;
  ASSERT_FALSE(report.outcomes[1].refusal.has_value()) << report.outcomes[1].refusal->what;
  ASSERT_TRUE(report.outcomes[2].refusal.has_value());
  EXPECT_TRUE(report.outcomes[2].refusal->state_refused);
  EXPECT_EQ(asked, 2U);  // slot 2's own rows, slot 1's widening
}

// The target's state with a compressed tensor beside it: a cell per four
// positions, 64 KiB each, unpadded, so a state of n positions uses
// ceil(n / 4) blocks (32 an extent) whatever the read alignment.
std::vector<en::StageState> BlockStages() {
  using C = md::StateTensorContract;
  const std::vector<std::pair<std::uint32_t, md::StateContract>> layers = {
      {0,
       {.tensors = {C{.name = "kv", .kind = md::StateKind::kKv, .cell_bytes = 4096},
                    C{.name = "state",
                      .kind = md::StateKind::kRecurrent,
                      .encoding = md::Encoding::kF32,
                      .cell_bytes = 4096,
                      .cells = md::CellRule::kFixed,
                      .window_or_ratio = 1,
                      .rollback = md::Rollback::kSnapshot},
                    C{.name = "blocks",
                      .kind = md::StateKind::kCompressed,
                      .cell_bytes = 65536,
                      .cells = md::CellRule::kPerBlock,
                      .window_or_ratio = 4,
                      .cell_pad = 1,
                      .basis = md::BlockBasis::kPositions,
                      .rollback = md::Rollback::kSnapshot}}}}};
  return {{.name = "target", .layout = md::Layout(layers, kContext, 256).value()}};
}

class BlockCapacityTest : public CapacityTest {
 protected:
  std::vector<en::StageState> StateStages() const override { return BlockStages(); }
};

TEST_F(BlockCapacityTest, AnOwnersOwnBackingIsWhatItReadsAlone) {
  // A fresh owner's one row reads 256 cells alone: its first cache extent,
  // and 64 blocks (three extents, the first beside the fixed state). Ten
  // extents are left: slots 0 and 1 take their own four, slot 2's are
  // refused. Slot 0's widening to 1,024 cells is refused, and alone it needs
  // nothing more, so both lower slots run. Backed first only through their
  // one position (one block), every owner would pass, each would need two
  // extents more alone, and slot 2 would take what a lower slot then lacks.
  Leave(10);
  const auto report = Wave({Decode(0, 0), Decode(1, 0), Decode(2, 0)}, {{1}, {2}, {3}});
  for (std::uint32_t slot = 0; slot < 2; ++slot) {
    ASSERT_FALSE(report.outcomes[slot].refusal.has_value())
        << "slot " << slot << ": " << report.outcomes[slot].refusal->what;
    EXPECT_EQ(runner_->slots().used_bytes(slot), 4 * kExtent);
    EXPECT_EQ(runner_->slots().positions(slot), 1U);
  }
  ASSERT_TRUE(report.outcomes[2].refusal.has_value());
  EXPECT_TRUE(report.outcomes[2].refusal->state_refused);
  EXPECT_EQ(runner_->slots().positions(2), 0U);
}

// --------------------------------------------------------------- prefill

class PrefillTest : public WaveTest {
 protected:
  // Runs `prompts` (slot, tokens) to their ends, a unit at a time; the
  // last unit's outputs by slot.
  std::vector<std::vector<ts::FakeRow>> Prefill(
      const std::vector<std::pair<std::uint32_t, std::vector<std::int32_t>>>& prompts,
      md::Batching caps, en::PrefillOptions options, std::uint32_t* units = nullptr) {
    auto composer = en::WaveComposer::Make(caps, kContext, 256);
    EXPECT_TRUE(composer.has_value());
    en::PrefillOrchestrator prefill(*runner_, *composer, options);
    std::vector<std::uint32_t> done(prompts.size(), 0);
    std::vector<std::vector<ts::FakeRow>> last(kSlots);
    for (std::uint32_t unit = 0; unit < 64; ++unit) {
      std::vector<en::PromptWork> work;
      std::vector<std::size_t> of;
      for (std::size_t i = 0; i < prompts.size(); ++i) {
        if (done[i] == prompts[i].second.size()) continue;
        work.push_back({.slot = prompts[i].first,
                        .stage = 0,
                        .tokens = std::span(prompts[i].second).subspan(done[i]),
                        .past = done[i],
                        .want_head = true,
                        .all_outputs = false,
                        .boundary = 0});
        of.push_back(i);
      }
      if (work.empty()) {
        if (units != nullptr) *units = unit;
        return last;
      }
      const auto report = prefill.Step(work);
      for (std::size_t j = 0; j < work.size(); ++j) {
        const auto& outcome = report.outcomes[j];
        EXPECT_FALSE(outcome.refusal.has_value()) << outcome.refusal->what;
        if (outcome.refusal) return last;
        done[of[j]] += outcome.rows;
        // Nonfinal chunks are cut state-only: no head.
        EXPECT_EQ(outcome.outputs.empty(), !outcome.last);
        if (outcome.last) last[work[j].slot] = Rows(outcome.outputs);
      }
    }
    ADD_FAILURE() << "the prompts did not finish";
    return last;
  }
};

TEST_F(PrefillTest, ChunkedJoinedPromptsMatchOneUnchunkedRun) {
  const auto a = Tokens(700, 1000);
  const auto b = Tokens(600, 2000);
  auto caps = Caps();
  caps.join_prefill = true;
  caps.max_step_rows = 0;
  caps.joined_read_align = 0;
  std::uint32_t units = 0;
  const auto chunked = Prefill({{0, a}, {1, b}}, caps,
                               {.chunk_rows = 256,
                                .lookahead = 2,
                                .capture_ahead = true,
                                .prepare_ahead = true,
                                .state_only = true},
                               &units);
  EXPECT_EQ(units, 3U);
  // The same prompt in two chunks on slot 2 ends at the same state.
  const auto whole = Prefill({{2, a}}, caps,
                             {.chunk_rows = 512,
                              .lookahead = 0,
                              .capture_ahead = false,
                              .prepare_ahead = false,
                              .state_only = true});
  ASSERT_EQ(chunked[0].size(), 1U);
  ASSERT_EQ(chunked[1].size(), 1U);
  ASSERT_EQ(whole[2].size(), 1U);
  EXPECT_EQ(chunked[0][0].hash, whole[2][0].hash);
  EXPECT_EQ(chunked[0][0].hash, Hash(a));
  EXPECT_EQ(chunked[1][0].hash, Hash(b));
  EXPECT_EQ(chunked[0][0].cells, 700);
  EXPECT_EQ(chunked[1][0].cells, 600);
  EXPECT_EQ(chunked[0][0].position, 699);
  EXPECT_EQ(runner_->slots().positions(0), 700U);
  EXPECT_EQ(runner_->slots().positions(1), 600U);
  ExpectPrompt(0, a);
  ExpectPrompt(1, b);
  // The lookahead: the first unit built the second's and the third's plans
  // (both futures) while it ran; the second captured the third's recording
  // beside its own run and prepared its fresh extent (positions past 512),
  // so the third replayed on its first run.
  const auto& stats = runner_->lookahead_stats();
  EXPECT_EQ(stats.built, 2U);
  EXPECT_EQ(stats.cached, 2U);
  EXPECT_GE(stats.captured_ahead, 1U);
  EXPECT_GE(stats.prepared, 1U);
  EXPECT_GE(runner_->graph_stats().replayed, 1U);
}

TEST_F(PrefillTest, AShapeTheNextChunkRepeatsIsCapturedOnItsFirstRun) {
  // Chunks of 64 rows read 256 cells through position 256: the first
  // chunk's shape repeats, so it is captured as it runs and the next two
  // replay; the last returns its head, a shape of its own, built while the
  // second ran and captured ahead beside the third, so it replays too.
  const auto a = Tokens(256, 5);
  const auto last = Prefill({{0, a}}, Caps(),
                            {.chunk_rows = 64,
                             .lookahead = 2,
                             .capture_ahead = true,
                             .prepare_ahead = false,
                             .state_only = true});
  ASSERT_EQ(last[0].size(), 1U);
  EXPECT_EQ(last[0][0].hash, Hash(a));
  ExpectPrompt(0, a);
  EXPECT_EQ(runner_->lookahead_stats().captured_first, 1U);
  EXPECT_EQ(runner_->lookahead_stats().captured_ahead, 1U);
  EXPECT_EQ(runner_->graph_stats().replayed, 3U);
}

TEST_F(PrefillTest, RefusedRecordingsGiveTheirChargesBack) {
  // Every capture is refused (beside a run, on a first run, ahead): the
  // prompt runs launch by launch to the same state, and once no step holds
  // them no plan keeps a graph's charge.
  source_->options().refuse_record = true;
  const auto a = Tokens(256, 5);
  const auto last = Prefill({{0, a}}, Caps(),
                            {.chunk_rows = 64,
                             .lookahead = 2,
                             .capture_ahead = true,
                             .prepare_ahead = false,
                             .state_only = true});
  ASSERT_EQ(last[0].size(), 1U);
  EXPECT_EQ(last[0][0].hash, Hash(a));
  ExpectPrompt(0, a);
  EXPECT_EQ(runner_->plans().graphs(), 0U);
  EXPECT_EQ(runner_->graph_stats().replayed, 0U);
  EXPECT_GE(runner_->graph_stats().refused, 2U);
  std::vector<llmp::memory::ReclaimCandidate> candidates;
  runner_->ReclaimCandidates(0, true, candidates);
  ASSERT_FALSE(candidates.empty());
  for (const auto& c : candidates) {
    EXPECT_EQ(c.kind, llmp::memory::ReclaimKind::kPlan);
    EXPECT_EQ(c.bytes, 4096U);  // its host bytes alone
  }
}

class PositionedPrefillTest : public PrefillTest {
 protected:
  ts::FakeSourceOptions SourceOptions() override {
    auto o = PrefillTest::SourceOptions();
    o.positioned = true;
    return o;
  }
};

TEST_F(PositionedPrefillTest, ProgramsThatTakeTheirPositionsAreNeverRecordedAhead) {
  // The prefill lookahead's every path: the chunk shape that repeats (a
  // capture on its first run), the last chunk's own shape built ahead and
  // installed after the job (a capture ahead beside the third chunk), and
  // its fresh state prepared. Positioned programs take none of the
  // recordings, and each run is positioned, including the built future's.
  const auto a = Tokens(256, 5);
  const auto last = Prefill({{0, a}}, Caps(),
                            {.chunk_rows = 64,
                             .lookahead = 2,
                             .capture_ahead = true,
                             .prepare_ahead = true,
                             .state_only = true});
  ASSERT_EQ(last[0].size(), 1U);
  EXPECT_EQ(last[0][0].hash, Hash(a));
  EXPECT_EQ(last[0][0].position, 255);
  ExpectPrompt(0, a);
  EXPECT_GE(runner_->lookahead_stats().built, 1U);
  EXPECT_EQ(runner_->lookahead_stats().captured_ahead, 0U);
  EXPECT_EQ(runner_->lookahead_stats().captured_first, 0U);
  EXPECT_EQ(runner_->plans().graphs(), 0U);
  EXPECT_EQ(runner_->graph_stats().captured, 0U);
  EXPECT_EQ(runner_->graph_stats().replayed, 0U);
  EXPECT_EQ(runner_->graph_stats().refused, 0U);
  EXPECT_EQ(source_->positioned(), 4U);
}

TEST_F(PrefillTest, AllRowsOutputsAndCheckpointBoundaries) {
  auto composer = en::WaveComposer::Make(Caps(), kContext, 256);
  ASSERT_TRUE(composer.has_value());
  en::PrefillOrchestrator prefill(*runner_, *composer,
                                  {.chunk_rows = 128,
                                   .lookahead = 0,
                                   .capture_ahead = false,
                                   .prepare_ahead = false,
                                   .state_only = true});
  const auto a = Tokens(100, 1);
  const en::PromptWork cut{.slot = 0,
                           .stage = 0,
                           .tokens = a,
                           .past = 0,
                           .want_head = true,
                           .all_outputs = true,
                           .boundary = 40};
  EXPECT_EQ(prefill.ChunkRows(cut, 0), 40U);
  EXPECT_EQ(prefill.ChunkRows(cut, 40), 60U);
  auto first = prefill.Step(std::vector{cut});
  ASSERT_FALSE(first.outcomes[0].refusal.has_value());
  EXPECT_EQ(first.outcomes[0].rows, 40U);
  EXPECT_TRUE(first.outcomes[0].outputs.empty());  // a state-only cut at the boundary
  en::PromptWork rest = cut;
  rest.tokens = std::span(a).subspan(40);
  rest.past = 40;
  rest.boundary = 0;
  auto second = prefill.Step(std::vector{rest});
  ASSERT_FALSE(second.outcomes[0].refusal.has_value());
  EXPECT_TRUE(second.outcomes[0].last);
  const auto rows = Rows(second.outcomes[0].outputs);
  ASSERT_EQ(rows.size(), 60U);  // every row's outputs (scoring)
  EXPECT_EQ(rows.front().position, 40);
  EXPECT_EQ(rows.back().hash, Hash(a));
}

// Generated component commits keep each owner's verified rows alive until
// that owner next runs, even if a peer runs and cache pressure intervenes.
class ComponentCommitTest : public ModelRunnerTest {
 protected:
  std::vector<en::StageState> StateStages() const override {
    auto stages = Stages();
    stages.front().layout.items[1].contract.rollback = md::Rollback::kCommit;
    return stages;
  }
  ts::FakeSourceOptions SourceOptions() override {
    auto options = ModelRunnerTest::SourceOptions();
    options.commits = true;
    return options;
  }
  void Configure(en::RunnerConfig& config) override { config.max_verify_rows = 3; }
  std::expected<en::StepReport, en::StepRefusal> Verify() {
    const std::array<std::int32_t, 3> a{11, 12, 13}, b{21, 22, 23};
    const std::array<en::SegmentWork, 2> work{{{.tokens = a}, {.tokens = b}}};
    return runner_->RunStep({.shape = Shape(pl::Phase::kVerify, {Seg(0, 0, 3, 3), Seg(1, 0, 3, 3)}),
                             .work = work,
                             .advance = false,
                             .ahead = {}});
  }
};

TEST_F(ComponentCommitTest, JoinedSavesSurviveAPeerStepAndLatePreparationFailure) {
  auto verified = Verify();
  ASSERT_TRUE(verified) << verified.error().what;
  ASSERT_TRUE(runner_->slots().Accept(0, 1));
  ASSERT_TRUE(runner_->slots().Accept(1, 2));
  // An unrelated owner's work may reuse all transient activations.
  ASSERT_TRUE(Run(Shape(pl::Phase::kDecode, {Seg(2, 0, 1, 1)}), {{99}}));
  const auto next = Shape(pl::Phase::kDecode, {Seg(0, 1, 1, 1), Seg(1, 2, 1, 1)});
  source_->options().refuse_build = [](const pl::StepShape& shape) {
    return shape.phase == pl::Phase::kCommit && shape.segments.front().slot == 1;
  };
  const auto before = source_->executed();
  en::StepAhead ahead;
  ahead.prepare = true;
  ahead.shapes[0] = Shape(pl::Phase::kPrefill, {Seg(0, 510, 4, 1)});
  auto refused = Run(next, {{14}, {24}}, ahead);
  ASSERT_FALSE(refused);
  EXPECT_TRUE(refused.error().clean);
  EXPECT_EQ(source_->executed(), before);
  EXPECT_FALSE(runner_->slots().live(0, 0).preparing());
  EXPECT_GT(runner_->slots().live(0, 0).preparation_stats().attempted, 0U);
  EXPECT_TRUE(runner_->slots().live(0, 0).owed());
  EXPECT_TRUE(runner_->slots().live(1, 0).owed());
  EXPECT_EQ(runner_->slots().positions(0), 1U);
  EXPECT_EQ(runner_->slots().positions(1), 2U);
  source_->options().refuse_build = {};
  // Reclaim every eligible plan while the later commit is built. The
  // prepared earlier commit and current step must remain pinned.
  source_->options().on_build = [&](const pl::StepShape&, bool measuring) {
    if (measuring) return;
    std::vector<llmp::memory::ReclaimCandidate> candidates;
    runner_->ReclaimCandidates(0, true, candidates);
    for (const auto& c : candidates) (void)runner_->Reclaim(c.kind, c.id);
  };
  auto retried = Run(next, {{14}, {24}});
  ASSERT_TRUE(retried) << retried.error().what;
  ASSERT_EQ(retried->outputs.size(), 2U);
  EXPECT_EQ(Rows(retried->outputs[0]).front().hash, Hash(std::array<std::int32_t, 2>{11, 14}));
  EXPECT_EQ(Rows(retried->outputs[1]).front().hash, Hash(std::array<std::int32_t, 3>{21, 22, 24}));
  EXPECT_FALSE(runner_->slots().live(0, 0).owed());
  EXPECT_FALSE(runner_->slots().live(1, 0).owed());
}

TEST_F(ComponentCommitTest, SettlingAnOwnerPreparesItsCommitUnderPlanPressure) {
  ASSERT_TRUE(Verify());
  ASSERT_TRUE(runner_->slots().Accept(0, 2));
  ASSERT_TRUE(runner_->slots().DiscardVerify(1));
  source_->options().on_build = [&](const pl::StepShape&, bool measuring) {
    if (measuring) return;
    std::vector<llmp::memory::ReclaimCandidate> candidates;
    runner_->ReclaimCandidates(0, true, candidates);
    for (const auto& c : candidates) (void)runner_->Reclaim(c.kind, c.id);
  };
  ASSERT_TRUE(runner_->slots().Settle(0));
  auto ran = Run(Shape(pl::Phase::kDecode, {Seg(0, 2, 1, 1)}), {{14}});
  ASSERT_TRUE(ran) << ran.error().what;
  EXPECT_EQ(Rows(ran->outputs[0]).front().hash, Hash(std::array<std::int32_t, 3>{11, 12, 14}));
}

TEST_F(ComponentCommitTest, CommitInputsStayDistinctThroughCaptureAndReplay) {
  std::array<std::uint64_t, 2> hashes{};
  for (std::uint32_t round = 0; round < 4; ++round) {
    const std::array<std::int32_t, 3> a{101 + static_cast<std::int32_t>(round), 202, 303};
    const std::array<std::int32_t, 3> b{404 + static_cast<std::int32_t>(round), 505, 606};
    const std::array<en::SegmentWork, 2> work{{{.tokens = a}, {.tokens = b}}};
    const auto past0 = runner_->slots().positions(0), past1 = runner_->slots().positions(1);
    auto verified = runner_->RunStep(
        {.shape = Shape(pl::Phase::kVerify, {Seg(0, past0, 3, 3), Seg(1, past1, 3, 3)}),
         .work = work,
         .advance = false,
         .ahead = {}});
    ASSERT_TRUE(verified) << verified.error().what;
    ASSERT_TRUE(runner_->slots().Accept(0, 1));
    ASSERT_TRUE(runner_->slots().Accept(1, 2));
    hashes[0] = ts::FakeHash(hashes[0], a[0]);
    hashes[1] = ts::FakeHash(ts::FakeHash(hashes[1], b[0]), b[1]);
    const auto ran =
        Run(Shape(pl::Phase::kDecode, {Seg(0, past0 + 1, 1, 1), Seg(1, past1 + 2, 1, 1)}),
            {{707}, {808}});
    ASSERT_TRUE(ran) << ran.error().what;
    hashes[0] = ts::FakeHash(hashes[0], 707);
    hashes[1] = ts::FakeHash(hashes[1], 808);
    EXPECT_EQ(Rows(ran->outputs[0]).front().hash, hashes[0]);
    EXPECT_EQ(Rows(ran->outputs[1]).front().hash, hashes[1]);
  }
  // Two current-step shapes plus two distinct owner commits, all captured.
  EXPECT_GE(runner_->plans().graphs(), 4U);
}

TEST_F(ComponentCommitTest, AnInternalCommitCannotBeCapturedAhead) {
  en::StepAhead ahead;
  ahead.shapes[0] = Shape(pl::Phase::kCommit, {Seg(0, 0, 1, 0)});
  ahead.build = 1;
  ahead.capture_ahead = true;
  const auto before = source_->queued();
  auto refused = Run(Shape(pl::Phase::kDecode, {Seg(0, 0, 1, 1)}), {{17}}, ahead);
  ASSERT_FALSE(refused);
  EXPECT_TRUE(refused.error().clean);
  EXPECT_EQ(source_->queued(), before);
  EXPECT_EQ(runner_->slots().positions(0), 0U);
}

TEST_F(ComponentCommitTest, AnInternalCommitCannotRunWithoutTheOwedFlow) {
  const auto before = source_->queued();
  const auto refused = Run(Shape(pl::Phase::kCommit, {Seg(0, 0, 1, 0)}), {{17}});
  ASSERT_FALSE(refused);
  EXPECT_TRUE(refused.error().clean);
  EXPECT_EQ(source_->queued(), before);
  EXPECT_EQ(runner_->slots().positions(0), 0U);
  EXPECT_FALSE(runner_->slots().live(0, 0).owed());
}

TEST_F(ModelRunnerTest, PhysicalAlignmentFactsMatchMeasurementAndExecutionBindings) {
  const auto base_alignment = node_.memory().ReservationAlignment().value();
  ASSERT_NE(base_alignment, 0U);
  const auto check = [&] {
    for (const auto& built : source_->built()) {
      EXPECT_EQ(built.activation_alignment, base_alignment);
      EXPECT_TRUE(built.bound_alignments_hold);
      EXPECT_TRUE(built.invalid_alignment_refs_unknown);
      const auto& layout = runner_->slots().layout(built.shape.stage);
      ASSERT_EQ(built.state_alignments.size(), built.shape.segments.size() * layout.items.size());
      std::size_t at = 0;
      for (const auto& owner : built.shape.segments) {
        (void)owner;
        for (const auto& item : layout.items)
          EXPECT_EQ(built.state_alignments[at++], std::gcd(base_alignment, item.offset));
      }
    }
  };
  check();
  auto ran = Run(Shape(pl::Phase::kPrefill, {Seg(1, 0, 3, 1)}), {Tokens(3, 42)});
  ASSERT_TRUE(ran) << ran.error().what;
  ASSERT_FALSE(source_->built().back().measuring);
  check();
}

}  // namespace
