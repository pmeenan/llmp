// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The shared slot lifecycle (engine/slot_lifecycle.h) on the fake device
// backend: three request slots, each a target and two drafters' state (a
// per-position cache with fixed state, a drafter's cache, a drafter's
// ring). Incremental spill and restore, a checkpoint's three-step restore
// and its refusals, a clean capacity refusal, adoption of kept files with
// the layout and footprint checks, a clear with its peers held, the
// selection guards (pending restore, borrow, verify awaiting its Accept),
// quarantine, and recovery in place; and the challenges: a growth or a
// checkpoint restore refused in a later stage, an adopted restore refused
// partway, a verify's owed commit run after a spill, a checkpoint restore
// (and its copies in) refused until such a commit settled, a borrow or a
// copy out refused until a verify settled, recovery discarding a slot
// whose Accept or checkpoint restore the failure interrupted, a stage's
// saves kept to its own bytes, no step mid-restore, no write record or
// positions for a spilled slot, an owed commit into another stage, and the
// serving sequences around an Accept the guards must leave open.

#include "engine/slot_lifecycle.h"

#include <gtest/gtest.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <expected>
#include <filesystem>
#include <format>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "base/bytes.h"
#include "engine/coverage.h"
#include "engine/paged_node.h"
#include "engine/runner_memory.h"
#include "engine/support.h"
#include "expected_error.h"
#include "model/state_contract.h"
#include "platform/kept_files.h"
#include "providers/device_runtime.h"

namespace {

namespace en = llmp::engine;
namespace md = llmp::model;
namespace pr = llmp::providers;
using llmp::base::Bytes;
using Range = en::LiveState::Range;
constexpr std::uint64_t kExtent = en::kPagedExtent;
constexpr std::uint32_t kContext = 1024;

// The target: a cache of 8 KiB a position (256 positions an extent, four
// extents), then 4 KiB of fixed state in a fifth extent. A drafter's cache
// of 4 KiB a position (two extents), and a drafter's 64 KiB ring.
std::vector<en::StageState> Stages() {
  using C = md::StateTensorContract;
  const auto layout = [](std::vector<C> tensors) {
    const std::vector<std::pair<std::uint32_t, md::StateContract>> layers = {
        {0, {.tensors = std::move(tensors)}}};
    return md::Layout(layers, kContext, 64).value();
  };
  return {
      {.name = "target",
       .layout = layout({{.name = "kv", .kind = md::StateKind::kKv, .cell_bytes = 8192},
                         {.name = "state",
                          .kind = md::StateKind::kRecurrent,
                          .encoding = md::Encoding::kF32,
                          .cell_bytes = 4096,
                          .cells = md::CellRule::kFixed,
                          .fixed_cells = 1,
                          .rollback = md::Rollback::kSnapshot}})},
      {.name = "mtp",
       .layout = layout({{.name = "kv", .kind = md::StateKind::kKv, .cell_bytes = 4096}})},
      {.name = "ring",
       .layout = layout({{.name = "ring",
                          .kind = md::StateKind::kDrafterRing,
                          .encoding = md::Encoding::kF32,
                          .cell_bytes = 65536,
                          .cells = md::CellRule::kFixed,
                          .fixed_cells = 1,
                          .rollback = md::Rollback::kSnapshot}})},
  };
}

class Runner final : public en::PagedModel {
 public:
  Runner(en::PagedNode& node, en::SlotLifecycleOptions options)
      : slots(node, std::move(options), 0, 0), node_(node) {}
  en::SlotLifecycle slots;
  std::uint32_t stream() const override { return 0; }
  const llmp::catalog::Closure& fence_closure() const override { return slots.fence(); }
  std::vector<llmp::catalog::ExtentId> managed_extents() const override {
    auto all = slots.state();
    const auto kept = slots.kept_state();
    all.insert(all.end(), kept.begin(), kept.end());
    return all;
  }
  en::Status Release() override {
    std::vector<std::string> problems;
    slots.Release(node_.memory(), problems);
    if (!problems.empty()) return std::unexpected(problems.front());
    return {};
  }

 private:
  en::PagedNode& node_;
};

class SlotLifecycleTest : public ::testing::Test {
 protected:
  en::PagedNode node_{
      en::NodeSettings{.compute_streams = 1, .slots = 2, .copy_lane = true, .slot_bytes = kExtent}};
  std::unique_ptr<Runner> runner_;
  std::uint64_t fixed_ = 0;
  std::filesystem::path kept_;
  int kept_dir_ = -1;
  std::vector<std::string> invalidated_;
  int dropped_ = 0;
  void* host_ = nullptr;
  bool running_ = false;

  virtual std::uint64_t BudgetExtents() const { return 48; }
  virtual std::vector<en::StageState> MakeStages() const { return Stages(); }
  virtual std::optional<llmp::base::Sha256Digest> ExecutionIdentity() const { return std::nullopt; }
  std::string unbound_layout_;
  en::SlotLifecycle& slots() { return runner_->slots; }

  void SetUp() override {
    // NOLINTNEXTLINE(concurrency-mt-unsafe): test environment is immutable
    const char* scratch = std::getenv("LLMP_TEST_SCRATCH");
    ASSERT_NE(scratch, nullptr);
    std::filesystem::create_directories(scratch);
    ASSERT_TRUE(node_.Open());
    fixed_ = node_.catalog().OccupancyOf(node_.domain()).Total().value();
    en::SlotLifecycleOptions options{.model = "Test",
                                     .slots = 3,
                                     .stages = MakeStages(),
                                     .spec = {},
                                     .read_align = 256,
                                     .spill_place = {},
                                     .spill_directory = scratch,
                                     .invalidate = {},
                                     .drop_plans = {}};
    options.spec[0] = 0x42;
    options.invalidate = [this](std::uint32_t slot) {
      invalidated_.push_back(std::format("{}", slot));
    };
    options.drop_plans = [this] { ++dropped_; };
    // Slot 2's state lives in named files kept from "the process before"
    // (D-105): marked bytes at known places.
    kept_ = std::filesystem::path(scratch) / std::format("slot-lifecycle-{}", ::getpid());
    auto dir = llmp::platform::OpenPrivateDirectory(-1, kept_.c_str());
    ASSERT_TRUE(dir.has_value());
    kept_dir_ = *dir;
    const auto stages = MakeStages();
    for (std::uint32_t s = 0; s < stages.size(); ++s) {
      auto file =
          llmp::platform::OpenPrivateFile(kept_dir_, std::format("stage-{}", s).c_str(),
                                          {.write = true, .create = true, .truncate = true});
      ASSERT_TRUE(file.has_value());
      const std::uint64_t size = (stages[s].layout.bytes + kExtent - 1) / kExtent * kExtent;
      const char mark = static_cast<char>(0x50 + s);
      ASSERT_EQ(::ftruncate(file->fd, static_cast<off_t>(size)), 0);
      ASSERT_EQ(::pwrite(file->fd, &mark, 1, 7), 1);
      (void)::close(file->fd);
    }
    options.spill_place = [this, scratch](std::uint32_t slot, std::uint32_t stage) {
      if (slot != 2) {
        return en::LiveState::SpillPlace{
            .directory = scratch, .dir = -1, .name = {}, .keep = false};
      }
      return en::LiveState::SpillPlace{
          .directory = {}, .dir = kept_dir_, .name = std::format("stage-{}", stage), .keep = true};
    };
    runner_ = std::make_unique<Runner>(node_, std::move(options));
    auto set = slots().Setup();
    ASSERT_TRUE(set) << llmp::test_support::Failed(set).value_or("");
    unbound_layout_ = slots().kept_layout_id();
    const auto* storage = slots().kept_layout_id().data();
    ASSERT_TRUE(slots().SealExecutionIdentity(ExecutionIdentity()));
    EXPECT_EQ(slots().kept_layout_id().data(), storage);
    const auto sealed = slots().kept_layout_id();
    EXPECT_FALSE(slots().SealExecutionIdentity(ExecutionIdentity()));
    EXPECT_EQ(slots().kept_layout_id(), sealed);
    auto started = node_.Start(Bytes(fixed_ + (BudgetExtents() * kExtent)));
    ASSERT_TRUE(started) << llmp::test_support::Failed(started).value_or("");
    auto registered = slots().Register();
    ASSERT_TRUE(registered) << llmp::test_support::Failed(registered).value_or("");
    (void)::close(kept_dir_);
    kept_dir_ = -1;
    ASSERT_TRUE(node_.scheduler().PinPlaces(slots().reserved_extents()));
    node_.Run();
    running_ = true;
    auto bound = slots().Bind({}, {});
    ASSERT_TRUE(bound) << llmp::test_support::Failed(bound).value_or("");
    std::vector<llmp::catalog::ExtentId> staging;
    auto host = node_.Pinned(8 * kExtent, 0, staging);
    ASSERT_TRUE(host.has_value()) << host.error();
    host_ = *host;
  }
  void TearDown() override {
    if (runner_ != nullptr) {
      if (node_.InRequest(0)) EXPECT_TRUE(node_.EndRequest(0));
      std::array<en::PagedModel*, 1> models = {runner_.get()};
      auto closed = node_.TearDown(models);
      EXPECT_TRUE(closed) << llmp::test_support::Failed(closed).value_or("");
    }
    if (kept_dir_ >= 0) (void)::close(kept_dir_);
    std::filesystem::remove_all(kept_);
  }

  void Select(std::vector<std::uint32_t> which) {
    auto selected = slots().SelectSlots(which);
    ASSERT_TRUE(selected) << llmp::test_support::Failed(selected).value_or("");
  }
  // Every used range of a slot's stages, the slot's interface's form.
  std::vector<Range> Used(std::uint32_t slot) {
    std::vector<Range> out;
    for (std::size_t s = 0; s < slots().stages(); ++s) {
      for (Range r : slots().live(slot, s).used_ranges()) {
        r.region = s;
        out.push_back(r);
      }
    }
    return out;
  }
  // `ranges` of a slot's state filled with `value` (one job).
  void Fill(std::uint32_t slot, std::span<const Range> ranges, std::uint8_t value) {
    std::vector<std::pair<std::uint64_t, std::uint64_t>> at;
    for (const Range& r : ranges) {
      at.emplace_back(slots().live(slot, r.region).base(0) + r.offset, r.bytes);
    }
    auto filled = node_.Job(
        slots().slot_fence(slot),
        [at, value](pr::NativeStream native) {
          for (const auto& [base, bytes] : at) {
            if (!pr::FillAsync(native, en::support::Pointer(base), value, bytes).ok()) {
              return llmp::scheduler::JobResult::kUnknown;
            }
          }
          return llmp::scheduler::JobResult::kQueued;
        },
        "test fill", 0);
    ASSERT_TRUE(filled) << llmp::test_support::Failed(filled).value_or("");
  }
  // Whether every byte of `ranges` is `value` (a copy to the host).
  bool All(std::uint32_t slot, std::span<const Range> ranges, std::uint8_t value) {
    std::uint64_t bytes = 0;
    for (const Range& r : ranges) bytes += r.bytes;
    EXPECT_LE(bytes, 8 * kExtent);
    auto copied = slots().CopyState(slot, host_, ranges, true);
    EXPECT_TRUE(copied) << llmp::test_support::Failed(copied).value_or("");
    const auto* p = static_cast<const std::uint8_t*>(host_);
    return std::all_of(p, p + bytes, [value](std::uint8_t b) { return b == value; });
  }
  // Pinned memory leaving `room` extents of the budget free (null if it
  // already leaves no more).
  void* Leave(std::uint64_t room) {
    const std::uint64_t budget = fixed_ + (BudgetExtents() * kExtent);
    const std::uint64_t used = Occupancy();
    EXPECT_GE(budget, used + (room * kExtent));
    if (budget <= used + (room * kExtent)) return nullptr;
    std::vector<llmp::catalog::ExtentId> staging;
    auto pad = node_.Pinned(budget - used - (room * kExtent), 0, staging);
    EXPECT_TRUE(pad.has_value()) << pad.error();
    return pad.value_or(nullptr);
  }
  // A verify from `past` (recorded) in the target's stage whose Accept of
  // one row owes the fixed state's commit, which fills it with `value`.
  void OweCommit(std::uint32_t slot, std::uint32_t past, std::uint8_t value) {
    const std::uint64_t at = slots().live(slot, 0).base(0) + (4 * kExtent);
    slots().live(slot, 0).SetCommit(
        [at, value](pr::NativeStream native, std::uint32_t) -> en::LiveState::Queued {
          if (!pr::FillAsync(native, en::support::Pointer(at), value, 4096).ok()) {
            return std::unexpected(en::LiveState::QueueFailure{.unknown = true, .detail = "fill"});
          }
          return {};
        });
    ASSERT_TRUE(slots().RecordWrite(slot, past));
    slots().live(slot, 0).Verified(2);
    ASSERT_TRUE(slots().live(slot, 0).Accept(1));
    ASSERT_TRUE(slots().owed(slot));
  }
  // A job that runs what the slot's stage owes (as a step's QueueOwed
  // does, before its own work), the step recorded as writing from `from`.
  void StepRunsOwed(std::uint32_t slot, std::size_t stage, std::uint32_t from) {
    ASSERT_TRUE(slots().CheckActive(slot));
    ASSERT_TRUE(slots().RecordWrite(slot, from));
    en::LiveState& live = slots().live(slot, stage);
    auto ran = node_.Job(
        slots().slot_fence(slot),
        [&live](pr::NativeStream native) {
          return live.QueueOwed(native) ? llmp::scheduler::JobResult::kQueued
                                        : llmp::scheduler::JobResult::kUnknown;
        },
        "test step", 0);
    ASSERT_TRUE(ran) << llmp::test_support::Failed(ran).value_or("");
    ASSERT_FALSE(slots().owed(slot));
  }
  std::uint64_t Occupancy() {
    std::uint64_t bytes = 0;
    EXPECT_TRUE(node_.Call(
        [&]() -> en::Status {
          bytes = node_.catalog().OccupancyOf(node_.domain()).Total().value();
          return {};
        },
        "test occupancy"));
    return bytes;
  }
};

TEST_F(SlotLifecycleTest, NullExecutionIdentityPreservesLegacyLayoutAndCannotBeRebound) {
  EXPECT_EQ(slots().kept_layout_id(), unbound_layout_);
  const auto recipe = llmp::base::Sha256().Update("another execution").Finish();
  EXPECT_FALSE(slots().SealExecutionIdentity(recipe));
  EXPECT_EQ(slots().kept_layout_id(), unbound_layout_);
}

TEST_F(SlotLifecycleTest, CursorPublicationChecksEveryOwnerBeforeChangingAnyCursor) {
  const std::array<std::pair<std::uint32_t, std::uint32_t>, 2> invalid{std::pair{0U, 1U},
                                                                       std::pair{1U, kContext + 1}};
  EXPECT_FALSE(slots().SetPositions(invalid));
  EXPECT_EQ(slots().positions(0), 0U);
  EXPECT_EQ(slots().positions(1), 0U);
  const std::array<std::pair<std::uint32_t, std::uint32_t>, 2> duplicated{std::pair{0U, 1U},
                                                                          std::pair{0U, 2U}};
  EXPECT_FALSE(slots().SetPositions(duplicated));
  EXPECT_EQ(slots().positions(0), 0U);
  const std::array<std::pair<std::uint32_t, std::uint32_t>, 2> valid{std::pair{0U, 1U},
                                                                     std::pair{1U, 2U}};
  ASSERT_TRUE(slots().SetPositions(valid));
  EXPECT_EQ(slots().positions(0), 1U);
  EXPECT_EQ(slots().positions(1), 2U);
}

class QualifiedSlotLifecycleTest : public SlotLifecycleTest {
 protected:
  std::optional<llmp::base::Sha256Digest> ExecutionIdentity() const override {
    return llmp::base::Sha256().Update("qualified execution").Finish();
  }
};

TEST_F(QualifiedSlotLifecycleTest, UnboundLayoutRefusesBeforeRestoreOrAdoptionChangesState) {
  const auto sealed = slots().kept_layout_id();
  EXPECT_NE(sealed, unbound_layout_);
  EXPECT_EQ(sealed.size(), unbound_layout_.size());
  Select({0});
  ASSERT_TRUE(slots().ReserveStateThrough(0, 1));
  ASSERT_TRUE(slots().SetPositions(0, 1));
  const auto footprint = Used(0);
  ASSERT_TRUE(slots().ValidateFootprint(1, footprint));
  const auto occupied = Occupancy();
  const auto invalidations = invalidated_.size();
  auto restore = slots().PrepareRestore(0, 1, footprint, unbound_layout_);
  ASSERT_FALSE(restore);
  EXPECT_NE(restore.error().find("source layout differs"), std::string::npos);
  auto adopt = slots().Adopt(2, 1, footprint, unbound_layout_);
  ASSERT_FALSE(adopt);
  EXPECT_NE(adopt.error().find("layout differs"), std::string::npos);
  EXPECT_EQ(slots().positions(0), 1U);
  EXPECT_EQ(slots().positions(2), 0U);
  EXPECT_TRUE(std::ranges::equal(Used(0), footprint, [](const Range& a, const Range& b) {
    return a.region == b.region && a.offset == b.offset && a.bytes == b.bytes;
  }));
  EXPECT_FALSE(slots().adopted(2));
  EXPECT_EQ(invalidated_.size(), invalidations);
  EXPECT_EQ(Occupancy(), occupied);
  EXPECT_FALSE(slots().SealExecutionIdentity(std::nullopt));
  EXPECT_EQ(slots().kept_layout_id(), sealed);
}

TEST_F(SlotLifecycleTest, AllStagesGrowSpillIncrementallyAndRestoreExactly) {
  Select({0});
  ASSERT_TRUE(slots().ReserveStateThrough(0, 300));
  // Target: cache extents 0 and 1, fixed state's extent 4; each drafter one.
  EXPECT_EQ(slots().live(0, 0).extents().size(), 3U);
  EXPECT_EQ(slots().live(0, 1).extents().size(), 1U);
  EXPECT_EQ(slots().live(0, 2).extents().size(), 1U);
  ASSERT_TRUE(slots().SetPositions(0, 300));
  const auto used = Used(0);
  Fill(0, used, 0xA1);
  // Never written back: a spill writes everything.
  EXPECT_EQ(slots().spill_write_bytes(0), 5 * kExtent);
  const auto before = Occupancy();
  ASSERT_TRUE(slots().Spill(0));
  EXPECT_TRUE(slots().spilled(0));
  EXPECT_EQ(before - Occupancy(), 5 * kExtent);
  EXPECT_EQ(slots().spilled_bytes(0), 5 * kExtent);
  EXPECT_TRUE(slots().kept_whole(0));
  EXPECT_FALSE(slots().CheckActive(0));  // spilled: restore first
  ASSERT_TRUE(slots().Restore(0));
  EXPECT_FALSE(slots().spilled(0));
  EXPECT_TRUE(slots().kept_whole(0));
  EXPECT_TRUE(All(0, used, 0xA1));
  // Unchanged since: nothing to write. A write from position 300 changes
  // the cache's second extent, the fixed state and both drafters', not
  // the cache's first.
  EXPECT_EQ(slots().spill_write_bytes(0), 0U);
  ASSERT_TRUE(slots().RecordWrite(0, 300));
  EXPECT_EQ(slots().spill_write_bytes(0), 4 * kExtent);
  EXPECT_FALSE(slots().kept_whole(0));  // its files are behind it now
  const std::array<Range, 1> late = {Range{.region = 0, .offset = 300 * 8192, .bytes = 8192}};
  Fill(0, late, 0xB2);
  ASSERT_TRUE(slots().Spill(0));
  ASSERT_TRUE(slots().Restore(0));
  EXPECT_TRUE(All(0, late, 0xB2));
  const std::array<Range, 1> early = {Range{.region = 0, .offset = 0, .bytes = 300 * 8192}};
  EXPECT_TRUE(All(0, early, 0xA1));
  const std::array<Range, 2> drafters = {Range{.region = 1, .offset = 0, .bytes = kExtent},
                                         Range{.region = 2, .offset = 0, .bytes = 65536}};
  EXPECT_TRUE(All(0, drafters, 0xA1));
  EXPECT_TRUE(slots().CheckPlaces());
}

// A restored slot's spill files stay for its next incremental spill and
// occupy disk, which the spill budget counts; released, they hold nothing,
// the state stays as it was, and the next spill writes it whole.
TEST_F(SlotLifecycleTest, ARestoredSlotsRetainedFilesCountAndCanBeReleased) {
  Select({0});
  ASSERT_TRUE(slots().ReserveStateThrough(0, 300));
  ASSERT_TRUE(slots().SetPositions(0, 300));
  const auto used = Used(0);
  Fill(0, used, 0xC3);
  EXPECT_EQ(slots().retained_spill_bytes(0), 0U);  // nothing written back yet
  ASSERT_TRUE(slots().Spill(0));
  EXPECT_EQ(slots().retained_spill_bytes(0), 0U);  // spilled: spilled_bytes counts it
  ASSERT_TRUE(slots().Restore(0));
  EXPECT_GE(slots().retained_spill_bytes(0), kExtent);
  EXPECT_EQ(slots().spill_write_bytes(0), 0U);
  ASSERT_TRUE(slots().ReleaseSpillFiles(0));
  EXPECT_EQ(slots().retained_spill_bytes(0), 0U);
  EXPECT_TRUE(All(0, used, 0xC3));
  EXPECT_FALSE(slots().kept_whole(0));
  EXPECT_EQ(slots().spill_write_bytes(0), 5 * kExtent);
  ASSERT_TRUE(slots().Spill(0));
  ASSERT_TRUE(slots().Restore(0));
  EXPECT_TRUE(All(0, used, 0xC3));
  EXPECT_TRUE(slots().CheckPlaces());
}

TEST_F(SlotLifecycleTest, ACheckpointRestoresInThreeStepsAndRefusesOutOfOrder) {
  Select({0});
  ASSERT_TRUE(slots().ReserveStateThrough(0, 300));
  ASSERT_TRUE(slots().SetPositions(0, 300));
  Fill(0, Used(0), 0xA1);
  const auto footprint = Used(0);
  auto pages = slots().CheckpointRanges(0, 300);
  ASSERT_TRUE(pages) << pages.error();
  // The target's cache extent 0 holds positions 0..255 only: immutable.
  ASSERT_EQ(pages->size(), 4U);
  EXPECT_EQ((*pages)[0].offset, kExtent);
  std::uint64_t page_bytes = 0;
  for (const auto& p : *pages) page_bytes += p.bytes;
  ASSERT_TRUE(slots().CopyState(0, host_, *pages, true));
  std::vector<std::uint8_t> saved(static_cast<const std::uint8_t*>(host_),
                                  static_cast<const std::uint8_t*>(host_) + page_bytes);
  // The conversation goes on: written from 300, grown to 900.
  ASSERT_TRUE(slots().RecordWrite(0, 300));
  Fill(0, *pages, 0xB2);
  ASSERT_TRUE(slots().ReserveStateThrough(0, 900));
  ASSERT_TRUE(slots().SetPositions(0, 900));
  EXPECT_GT(Used(0).size(), footprint.size());
  // Refusals before anything changes: another layout, a footprint short of
  // its positions, completion with nothing pending, a copy in outside one.
  EXPECT_FALSE(slots().PrepareRestore(0, 300, footprint, "gemma2-2b-f16-kv-device-v1:4096"));
  const std::vector<Range> short_print(footprint.begin() + 1, footprint.end());
  EXPECT_FALSE(slots().PrepareRestore(0, 300, short_print, slots().kept_layout_id()));
  EXPECT_FALSE(slots().CompleteRestore(0, 300));
  EXPECT_FALSE(slots().CopyState(0, host_, *pages, false));
  EXPECT_FALSE(slots().restoring(0));
  invalidated_.clear();
  auto prepared = slots().PrepareRestore(0, 300, footprint, slots().kept_layout_id());
  ASSERT_TRUE(prepared) << llmp::test_support::Failed(prepared).value_or("");
  EXPECT_TRUE(slots().restoring(0));
  EXPECT_EQ(invalidated_.size(), 1U);
  // Exactly the footprint again: the newer tail dropped.
  const auto now = Used(0);
  ASSERT_EQ(now.size(), footprint.size());
  for (std::size_t i = 0; i < now.size(); ++i) {
    EXPECT_EQ(now[i].region, footprint[i].region);
    EXPECT_EQ(now[i].offset, footprint[i].offset);
  }
  // While restoring: no growth, spill, second restore, copy out, or
  // selection dropping it.
  EXPECT_FALSE(slots().ReserveStateThrough(0, 400));
  EXPECT_FALSE(slots().Spill(0));
  EXPECT_FALSE(slots().PrepareRestore(0, 300, footprint, slots().kept_layout_id()));
  EXPECT_FALSE(slots().CopyState(0, host_, *pages, true));
  EXPECT_FALSE(slots().SelectSlots(std::vector<std::uint32_t>{1}));
  EXPECT_FALSE(slots().Restore(0));
  // Pages copied in order: completion refused until the last one lands.
  std::memcpy(host_, saved.data(), saved.size());
  auto* source = static_cast<std::byte*>(host_);
  for (std::size_t i = 0; i < pages->size(); ++i) {
    if (i + 1 == pages->size()) EXPECT_FALSE(slots().CompleteRestore(0, 300));
    ASSERT_TRUE(slots().CopyState(0, source, std::span(*pages).subspan(i, 1), false));
    source += (*pages)[i].bytes;
  }
  EXPECT_FALSE(slots().CompleteRestore(0, 299));
  ASSERT_TRUE(slots().CompleteRestore(0, 300));
  EXPECT_EQ(slots().positions(0), 300U);
  EXPECT_FALSE(slots().restoring(0));
  EXPECT_TRUE(All(0, Used(0), 0xA1));
}

TEST_F(SlotLifecycleTest, KeptFilesAreAdoptedOnlyWhenTheLayoutAndFootprintMatch) {
  // Positions 300: the target's extents 0, 1 and 4 (4 KiB), each drafter's 0.
  const auto stages = Stages();
  const std::vector<Range> footprint = {
      {.region = 0, .offset = 0, .bytes = kExtent},
      {.region = 0, .offset = kExtent, .bytes = kExtent},
      {.region = 0, .offset = 4 * kExtent, .bytes = stages[0].layout.bytes - (4 * kExtent)},
      {.region = 1, .offset = 0, .bytes = kExtent},
      {.region = 2, .offset = 0, .bytes = stages[2].layout.bytes}};
  const auto id = slots().kept_layout_id();
  EXPECT_TRUE(id.starts_with("llmp-state-v2:"));
  EXPECT_FALSE(slots().Adopt(2, 300, footprint, "qwen38-state/1;context=1024"));
  EXPECT_FALSE(slots().Adopt(2, 0, footprint, id));
  EXPECT_FALSE(slots().Adopt(2, 600, footprint, id));  // short of 600 positions
  auto unordered = footprint;
  std::swap(unordered[0], unordered[1]);
  EXPECT_FALSE(slots().Adopt(2, 300, unordered, id));
  auto partial = footprint;
  partial[2].bytes = kExtent;  // past the layout
  EXPECT_FALSE(slots().Adopt(2, 300, partial, id));
  // Not into a slot holding state.
  Select({1});
  ASSERT_TRUE(slots().ReserveStateThrough(1, 10));
  EXPECT_FALSE(slots().Adopt(1, 300, footprint, id));
  // Nor into one whose clear kept zeroed backing, which growth would take
  // back instead of reading the files.
  ASSERT_TRUE(slots().Clear(1));
  EXPECT_EQ(slots().used_bytes(1), 0U);
  EXPECT_FALSE(slots().kept_state().empty());
  EXPECT_FALSE(slots().Adopt(1, 300, footprint, id));
  // Nor into an unusable one.
  slots().live(0, 1).Quarantine();
  EXPECT_FALSE(slots().Adopt(0, 300, footprint, id));
  invalidated_.clear();
  auto adopted = slots().Adopt(2, 300, footprint, id);
  ASSERT_TRUE(adopted) << llmp::test_support::Failed(adopted).value_or("");
  EXPECT_EQ(invalidated_.size(), 1U);
  EXPECT_TRUE(slots().spilled(2));
  EXPECT_TRUE(slots().adopted(2));
  EXPECT_FALSE(slots().kept_whole(2));
  EXPECT_EQ(slots().positions(2), 300U);
  EXPECT_EQ(slots().spilled_bytes(2), 5 * kExtent);
  EXPECT_EQ(slots().used_bytes(2), 0U);
  ASSERT_TRUE(slots().Restore(2));
  EXPECT_FALSE(slots().adopted(2));
  EXPECT_TRUE(slots().kept_whole(2));  // its files hold it as it is
  EXPECT_EQ(Used(2).size(), footprint.size());
  Select({2});
  const std::array<Range, 3> marks = {Range{.region = 0, .offset = 7, .bytes = 1},
                                      Range{.region = 1, .offset = 7, .bytes = 1},
                                      Range{.region = 2, .offset = 7, .bytes = 1}};
  ASSERT_TRUE(slots().CopyState(2, host_, marks, true));
  const auto* p = static_cast<const std::uint8_t*>(host_);
  EXPECT_EQ(p[0], 0x50);
  EXPECT_EQ(p[1], 0x51);
  EXPECT_EQ(p[2], 0x52);
}

TEST_F(SlotLifecycleTest, AClearKeepsItsHeldPeersAndResetsTheSlot) {
  Select({0, 1});
  ASSERT_TRUE(node_.BeginRequest(0, slots().execution(), "test request"));
  ASSERT_TRUE(slots().ReserveStateThrough(0, 300));
  ASSERT_TRUE(slots().ReserveStateThrough(1, 600));
  ASSERT_TRUE(slots().SetPositions(0, 300));
  Fill(0, Used(0), 0xA1);
  Fill(1, Used(1), 0xC3);
  const auto peer = Used(1);
  // Borrowed, or awaiting a verify's Accept: refused.
  auto borrow = slots().Borrow(0);
  ASSERT_TRUE(borrow);
  EXPECT_FALSE(slots().Clear(0));
  borrow->Return();
  slots().live(0, 0).Verified(2);
  EXPECT_FALSE(slots().Clear(0));
  EXPECT_FALSE(slots().ClearIdle(0));  // and held besides
  ASSERT_TRUE(slots().live(0, 0).Accept(1));
  auto cleared = slots().Clear(0);
  ASSERT_TRUE(cleared) << llmp::test_support::Failed(cleared).value_or("");
  EXPECT_EQ(slots().used_bytes(0), 0U);
  EXPECT_EQ(slots().positions(0), 0U);
  EXPECT_TRUE(slots().usable(0));
  // Its peer still held, every byte as it was.
  EXPECT_TRUE(slots().held(1));
  EXPECT_TRUE(All(1, peer, 0xC3));
  // Regrowth reuses the zeroed backing.
  ASSERT_TRUE(slots().ReserveStateThrough(0, 300));
  EXPECT_TRUE(All(0, Used(0), 0x00));
  ASSERT_TRUE(node_.EndRequest(0));
}

TEST_F(SlotLifecycleTest, SelectionNeverDropsAPendingRestoreABorrowOrAVerify) {
  Select({0, 1});
  ASSERT_TRUE(node_.BeginRequest(0, slots().execution(), "test request"));
  ASSERT_TRUE(slots().ReserveStateThrough(1, 300));
  ASSERT_TRUE(slots().SetPositions(1, 300));
  const auto footprint = Used(1);
  const std::vector<std::uint32_t> only_zero = {0};
  // A borrow.
  auto borrow = slots().Borrow(1);
  ASSERT_TRUE(borrow);
  EXPECT_FALSE(slots().SelectSlots(only_zero));
  EXPECT_FALSE(slots().ReserveStateThrough(1, 400));
  EXPECT_FALSE(slots().Spill(1));
  borrow->Return();
  // A verify awaiting its Accept, in a drafter's state.
  slots().live(1, 1).Verified(1);
  EXPECT_FALSE(slots().SelectSlots(only_zero));
  EXPECT_FALSE(slots().Spill(1));
  ASSERT_TRUE(slots().live(1, 1).Accept(1));
  // A pending restore.
  ASSERT_TRUE(slots().PrepareRestore(1, 300, footprint, slots().kept_layout_id()));
  EXPECT_FALSE(slots().SelectSlots(only_zero));
  // Keeping it is fine, and so is dropping it once it completed.
  const std::vector<std::uint32_t> both = {0, 1};
  EXPECT_TRUE(slots().SelectSlots(both));
  auto pages = slots().CheckpointRanges(1, 300);
  EXPECT_FALSE(pages.has_value());  // refused while restoring
  ASSERT_TRUE(node_.EndRequest(0));
  Select({1});
  ASSERT_TRUE(node_.BeginRequest(0, slots().execution(), "test request"));
  // Zeros copied in for every page a checkpoint at 300 holds.
  std::vector<Range> all;
  for (std::size_t s = 0; s < slots().stages(); ++s) {
    auto writes = md::CheckpointWrites(slots().layout(s), 300);
    ASSERT_TRUE(writes.has_value());
    for (const Range& r : footprint) {
      if (r.region != s) continue;
      const bool mutable_page = std::ranges::any_of(*writes, [&](const md::StateRange& w) {
        return w.bytes != 0 && w.offset < r.offset + r.bytes && r.offset < w.offset + w.bytes;
      });
      if (mutable_page) all.push_back(r);
    }
  }
  std::memset(host_, 0, 8 * kExtent);
  ASSERT_TRUE(slots().CopyState(1, host_, all, false));
  ASSERT_TRUE(slots().CompleteRestore(1, 300));
  ASSERT_TRUE(node_.EndRequest(0));
  EXPECT_TRUE(slots().SelectSlots(only_zero));
}

TEST_F(SlotLifecycleTest, AQuarantinedSlotRefusesWorkUntilClearedAndItsPeersRunOn) {
  Select({0});
  ASSERT_TRUE(slots().ReserveStateThrough(0, 300));
  slots().live(0, 2).Quarantine();  // a drafter's job failed after writing
  EXPECT_FALSE(slots().usable(0));
  EXPECT_FALSE(slots().CheckActive(0));
  EXPECT_FALSE(slots().ReserveStateThrough(0, 400));
  EXPECT_FALSE(slots().state_refused(0));
  EXPECT_FALSE(slots().Spill(0));
  EXPECT_FALSE(slots().kept_whole(0));
  EXPECT_FALSE(slots().CheckpointRanges(0, 300).has_value());
  ASSERT_TRUE(slots().Clear(0));
  EXPECT_TRUE(slots().usable(0));
  ASSERT_TRUE(slots().ReserveStateThrough(0, 10));
}

TEST_F(SlotLifecycleTest, RecoveryKeepsUntouchedSpilledSlotsAndDiscardsTheRest) {
  Select({1});
  ASSERT_TRUE(slots().ReserveStateThrough(1, 300));
  Fill(1, Used(1), 0xD4);
  const auto spilled = Used(1);
  ASSERT_TRUE(slots().Spill(1));
  Select({0});
  ASSERT_TRUE(slots().ReserveStateThrough(0, 300));
  ASSERT_TRUE(slots().SetPositions(0, 300));
  // A failure the cohort cannot prove local.
  slots().Fault();
  EXPECT_TRUE(slots().faulted());
  EXPECT_FALSE(slots().SelectSlots(std::vector<std::uint32_t>{0}));
  EXPECT_FALSE(slots().Restore(1));
  std::vector<std::uint32_t> told;
  dropped_ = 0;
  auto recovered = slots().RecoverInPlace([&](std::uint32_t slot) { told.push_back(slot); });
  ASSERT_TRUE(recovered) << recovered.error();
  EXPECT_EQ(*recovered, 0b001U);
  EXPECT_EQ(told, std::vector<std::uint32_t>{0});
  EXPECT_EQ(dropped_, 1);
  EXPECT_FALSE(slots().faulted());
  EXPECT_EQ(slots().used_bytes(0), 0U);
  EXPECT_EQ(slots().positions(0), 0U);
  EXPECT_TRUE(slots().spilled(1));
  ASSERT_TRUE(slots().Restore(1));
  Select({1});
  EXPECT_TRUE(All(1, spilled, 0xD4));
}

TEST_F(SlotLifecycleTest, AVerifysOwedCommitKeepsTheFilesBehindUntilTheNextWholeSpill) {
  Select({0});
  ASSERT_TRUE(slots().ReserveStateThrough(0, 300));
  ASSERT_TRUE(slots().SetPositions(0, 300));
  Fill(0, Used(0), 0xA1);
  ASSERT_TRUE(slots().Spill(0));
  ASSERT_TRUE(slots().Restore(0));
  ASSERT_EQ(slots().spill_write_bytes(0), 0U);
  // A verify from 300 (recorded) whose Accept owes the fixed state's
  // commit (the model's rebuild of the kept rows).
  const Range fixed{.region = 0, .offset = 4 * kExtent, .bytes = 4096};
  const std::uint64_t at = slots().live(0, 0).base(0) + fixed.offset;
  slots().live(0, 0).SetCommit(
      [at](pr::NativeStream native, std::uint32_t) -> en::LiveState::Queued {
        if (!pr::FillAsync(native, en::support::Pointer(at), 0xEE, 4096).ok()) {
          return std::unexpected(en::LiveState::QueueFailure{.unknown = true, .detail = "fill"});
        }
        return {};
      });
  ASSERT_TRUE(slots().RecordWrite(0, 300));
  slots().live(0, 0).Verified(2);
  // A swap's write-back while it awaits its Accept, or owes its commit,
  // leaves the slot's files behind the state that follows (every stage's
  // extents written by the next spill: a commit may write any stage).
  slots().StateWrittenBack(true);
  EXPECT_EQ(slots().spill_write_bytes(0), 5 * kExtent);
  ASSERT_TRUE(slots().live(0, 0).Accept(1));
  ASSERT_TRUE(slots().owed(0));
  slots().StateWrittenBack(true);
  EXPECT_EQ(slots().spill_write_bytes(0), 5 * kExtent);
  EXPECT_FALSE(slots().kept_whole(0));
  // Spilled and restored with the commit still owed, which then runs as a
  // job of its own that records no write (a settle before a checkpoint):
  // the next spill still writes it, so it comes back.
  ASSERT_TRUE(slots().Spill(0));
  ASSERT_TRUE(slots().Restore(0));
  auto settled = slots().live(0, 0).Rollback(node_, slots().slot_fence(0), 0, "test settle");
  ASSERT_TRUE(settled) << llmp::test_support::Failed(settled).value_or("");
  EXPECT_FALSE(slots().owed(0));
  EXPECT_EQ(slots().spill_write_bytes(0), 5 * kExtent);
  ASSERT_TRUE(slots().Spill(0));
  ASSERT_TRUE(slots().Restore(0));
  EXPECT_TRUE(All(0, std::array{fixed}, 0xEE));
  EXPECT_TRUE(slots().kept_whole(0));
}

TEST_F(SlotLifecycleTest, ACheckpointRestoreWaitsForAnOwedCommitToSettle) {
  Select({0});
  ASSERT_TRUE(slots().ReserveStateThrough(0, 300));
  ASSERT_TRUE(slots().SetPositions(0, 300));
  Fill(0, Used(0), 0xA1);
  const auto footprint = Used(0);
  auto pages = slots().CheckpointRanges(0, 300);
  ASSERT_TRUE(pages) << pages.error();
  std::uint64_t page_bytes = 0;
  for (const auto& p : *pages) page_bytes += p.bytes;
  ASSERT_TRUE(slots().CopyState(0, host_, *pages, true));
  std::vector<std::uint8_t> saved(static_cast<const std::uint8_t*>(host_),
                                  static_cast<const std::uint8_t*>(host_) + page_bytes);
  // A verify from 300 whose Accept owes the fixed state's commit.
  const Range fixed{.region = 0, .offset = 4 * kExtent, .bytes = 4096};
  const std::uint64_t at = slots().live(0, 0).base(0) + fixed.offset;
  slots().live(0, 0).SetCommit(
      [at](pr::NativeStream native, std::uint32_t) -> en::LiveState::Queued {
        if (!pr::FillAsync(native, en::support::Pointer(at), 0xEE, 4096).ok()) {
          return std::unexpected(en::LiveState::QueueFailure{.unknown = true, .detail = "fill"});
        }
        return {};
      });
  ASSERT_TRUE(slots().RecordWrite(0, 300));
  slots().live(0, 0).Verified(2);
  ASSERT_TRUE(slots().live(0, 0).Accept(1));
  ASSERT_TRUE(slots().owed(0));
  // Owed, the commit would run after the restore, over the checkpoint's
  // bytes (or into its dropped tail): refused, nothing changed.
  invalidated_.clear();
  EXPECT_FALSE(slots().PrepareRestore(0, 300, footprint, slots().kept_layout_id()));
  EXPECT_FALSE(slots().restoring(0));
  EXPECT_TRUE(invalidated_.empty());
  // Settled first, the checkpoint comes back exactly and stays.
  ASSERT_TRUE(slots().live(0, 0).Rollback(node_, slots().slot_fence(0), 0, "test settle"));
  EXPECT_TRUE(All(0, std::array{fixed}, 0xEE));
  ASSERT_TRUE(slots().PrepareRestore(0, 300, footprint, slots().kept_layout_id()));
  std::memcpy(host_, saved.data(), saved.size());
  ASSERT_TRUE(slots().CopyState(0, host_, *pages, false));
  ASSERT_TRUE(slots().CompleteRestore(0, 300));
  EXPECT_FALSE(slots().owed(0));
  EXPECT_TRUE(All(0, Used(0), 0xA1));
}

TEST_F(SlotLifecycleTest, AVerifyUnsettledKeepsItsStateFromReadersAndCopies) {
  Select({0});
  ASSERT_TRUE(slots().ReserveStateThrough(0, 300));
  ASSERT_TRUE(slots().SetPositions(0, 300));
  Fill(0, Used(0), 0xA1);
  const std::array<Range, 1> fixed = {Range{.region = 0, .offset = 4 * kExtent, .bytes = 4096}};
  // Awaiting its Accept, in the target's or a drafter's stage: the state
  // still holds rows that may be rejected.
  for (const std::size_t stage : {std::size_t{0}, std::size_t{2}}) {
    slots().live(0, stage).Verified(1);
    EXPECT_FALSE(slots().Borrow(0));
    EXPECT_FALSE(slots().borrowed(0));
    EXPECT_FALSE(slots().CopyState(0, host_, fixed, true));
    ASSERT_TRUE(slots().live(0, stage).Accept(1));
  }
  // Owing a commit: the state is not yet what the kept rows leave.
  OweCommit(0, 300, 0xEE);
  EXPECT_FALSE(slots().Borrow(0));
  EXPECT_FALSE(slots().borrowed(0));
  EXPECT_FALSE(slots().CopyState(0, host_, fixed, true));
  // Settled, both proceed and read what the commit left.
  ASSERT_TRUE(slots().live(0, 0).Rollback(node_, slots().slot_fence(0), 0, "test settle"));
  auto borrow = slots().Borrow(0);
  ASSERT_TRUE(borrow);
  EXPECT_TRUE(All(0, fixed, 0xEE));
  borrow->Return();
}

TEST_F(SlotLifecycleTest, ACheckpointRestoreTakesNoPageWhileAVerifyOwes) {
  Select({0});
  ASSERT_TRUE(slots().ReserveStateThrough(0, 300));
  ASSERT_TRUE(slots().SetPositions(0, 300));
  Fill(0, Used(0), 0xA1);
  const auto footprint = Used(0);
  auto pages = slots().CheckpointRanges(0, 300);
  ASSERT_TRUE(pages) << pages.error();
  std::uint64_t page_bytes = 0;
  for (const auto& p : *pages) page_bytes += p.bytes;
  ASSERT_TRUE(slots().CopyState(0, host_, *pages, true));
  std::vector<std::uint8_t> saved(static_cast<const std::uint8_t*>(host_),
                                  static_cast<const std::uint8_t*>(host_) + page_bytes);
  ASSERT_TRUE(slots().RecordWrite(0, 300));
  Fill(0, *pages, 0xB2);
  ASSERT_TRUE(slots().PrepareRestore(0, 300, footprint, slots().kept_layout_id()));
  // A verify run while restoring (the runner's to refuse) owes a commit:
  // a page copied in now would be overwritten by its settle, which
  // completion would then accept. Refused until settled.
  OweCommit(0, 300, 0xEE);
  std::memcpy(host_, saved.data(), saved.size());
  EXPECT_FALSE(slots().CopyState(0, host_, *pages, false));
  EXPECT_FALSE(slots().CompleteRestore(0, 300));
  ASSERT_TRUE(slots().live(0, 0).Rollback(node_, slots().slot_fence(0), 0, "test settle"));
  ASSERT_TRUE(slots().CopyState(0, host_, *pages, false));
  ASSERT_TRUE(slots().CompleteRestore(0, 300));
  EXPECT_TRUE(All(0, Used(0), 0xA1));
}

TEST_F(SlotLifecycleTest, RecoveryDiscardsASlotWhoseAcceptTheFailureSkipped) {
  Select({0, 1});
  ASSERT_TRUE(node_.BeginRequest(0, slots().execution(), "test request"));
  ASSERT_TRUE(slots().ReserveStateThrough(0, 300));
  ASSERT_TRUE(slots().ReserveStateThrough(1, 300));
  Fill(0, Used(0), 0xC3);
  const auto kept = Used(0);
  // Slot 1's verify ran; the failure came before its Accept.
  slots().live(1, 1).Verified(1);
  ASSERT_TRUE(node_.EndRequest(0));
  std::vector<std::uint32_t> told;
  auto recovered = slots().RecoverInPlace([&](std::uint32_t slot) { told.push_back(slot); });
  ASSERT_TRUE(recovered) << recovered.error();
  EXPECT_EQ(*recovered, 0b010U);
  EXPECT_EQ(told, std::vector<std::uint32_t>{1});
  EXPECT_FALSE(slots().verify_pending(1));
  EXPECT_EQ(slots().used_bytes(1), 0U);
  // Otherwise no selection without it could ever be made again.
  Select({0});
  EXPECT_TRUE(All(0, kept, 0xC3));
}

TEST_F(SlotLifecycleTest, RecoveryDiscardsACheckpointRestoreTheFailureInterrupted) {
  // Into state, or into nothing.
  for (const std::uint32_t positions : {300U, 0U}) {
    Select({1});
    std::vector<Range> footprint;
    if (positions != 0) {
      ASSERT_TRUE(slots().ReserveStateThrough(1, positions));
      footprint = Used(1);
    }
    ASSERT_TRUE(slots().PrepareRestore(1, positions, footprint, slots().kept_layout_id()));
    auto recovered = slots().RecoverInPlace();
    ASSERT_TRUE(recovered) << recovered.error();
    EXPECT_EQ(*recovered, 0b010U);
    EXPECT_FALSE(slots().restoring(1));
    Select({0});
  }
}

TEST_F(SlotLifecycleTest, ALocalFailureDiscardsOnlyItsSlotAndDropsStaleBorrows) {
  Select({0, 1});
  ASSERT_TRUE(node_.BeginRequest(0, slots().execution(), "test request"));
  ASSERT_TRUE(slots().ReserveStateThrough(0, 300));
  ASSERT_TRUE(slots().ReserveStateThrough(1, 300));
  Fill(1, Used(1), 0xC3);
  const auto peer = Used(1);
  auto stale = slots().Borrow(1);  // a reader whose Return the failure skipped
  ASSERT_TRUE(stale);
  slots().live(0, 1).Quarantine();  // slot 0's drafter job failed after writing
  ASSERT_TRUE(node_.EndRequest(0));
  std::vector<std::uint32_t> told;
  auto recovered = slots().RecoverInPlace([&](std::uint32_t slot) { told.push_back(slot); });
  ASSERT_TRUE(recovered) << recovered.error();
  EXPECT_EQ(*recovered, 0b001U);
  EXPECT_EQ(told, std::vector<std::uint32_t>{0});
  EXPECT_EQ(slots().used_bytes(0), 0U);
  EXPECT_TRUE(slots().usable(0));
  EXPECT_FALSE(slots().borrowed(1));
  Select({1});
  EXPECT_TRUE(All(1, peer, 0xC3));
  // The stale reader's return, after a new reader's borrow, returns
  // nothing: the new borrow holds.
  auto fresh = slots().Borrow(1);
  ASSERT_TRUE(fresh) << fresh.error();
  stale->Return();
  EXPECT_TRUE(slots().borrowed(1));
  EXPECT_FALSE(slots().Spill(1));
  fresh->Return();
  EXPECT_FALSE(slots().borrowed(1));
  EXPECT_TRUE(slots().Spill(1));
}

// L1: a verify's snapshot saves only its own stage's bytes. The file-current
// mark is per stage (a stage owing a restore is written whole by the next
// spill); a save of another stage's bytes would be restored behind that
// stage's file.
TEST_F(SlotLifecycleTest, AStagesVerifySavesOnlyItsOwnStagesBytes) {
  Select({0});
  ASSERT_TRUE(slots().ReserveStateThrough(0, 300));
  en::RunnerMemory memory(node_, 0, 0);
  constexpr std::uint64_t kSnapshot = 4096;
  auto snapshot = memory.Pinned(kSnapshot);
  ASSERT_TRUE(snapshot) << snapshot.error();
  en::LiveState& ring = slots().live(0, 2);
  ring.SnapshotAt(reinterpret_cast<std::uint64_t>(*snapshot), kSnapshot);
  ASSERT_TRUE(ring.AllocateSnapshot(memory, 8));
  const std::uint64_t own = ring.base(0);
  const std::uint64_t bytes = slots().layout(2).bytes;
  ring.BeginSaves();
  EXPECT_TRUE(ring.Save(own, 256, 0));
  EXPECT_TRUE(ring.Save(own + bytes - 256, 256, -1));
  // The target's bytes, the other drafter's, another slot's, and ranges
  // straddling its own region's ends: refused.
  EXPECT_FALSE(ring.Save(slots().live(0, 0).base(0), 256, 0));
  EXPECT_FALSE(ring.Save(slots().live(0, 1).base(0), 256, 0));
  EXPECT_FALSE(ring.Save(slots().live(1, 2).base(0), 256, 0));
  EXPECT_FALSE(ring.Save(own - 256, 512, 0));
  EXPECT_FALSE(ring.Save(own + bytes - 256, 512, 0));
  EXPECT_FALSE(ring.Save(own + bytes, 256, 0));
  EXPECT_EQ(ring.saved().size(), 2U);
  ring.BeginSaves();
}

// L2: between a checkpoint restore's Prepare and Complete the state is
// part old, part checkpoint: no step's work, positions, borrow or
// checkpoint, only the restore's own copies and completion.
TEST_F(SlotLifecycleTest, NoStepRunsBetweenACheckpointRestoresPrepareAndComplete) {
  Select({0});
  ASSERT_TRUE(slots().ReserveStateThrough(0, 300));
  ASSERT_TRUE(slots().SetPositions(0, 300));
  Fill(0, Used(0), 0xA1);
  const auto footprint = Used(0);
  auto pages = slots().CheckpointRanges(0, 300);
  ASSERT_TRUE(pages) << pages.error();
  std::uint64_t page_bytes = 0;
  for (const auto& p : *pages) page_bytes += p.bytes;
  ASSERT_TRUE(slots().CopyState(0, host_, *pages, true));
  std::vector<std::uint8_t> saved(static_cast<const std::uint8_t*>(host_),
                                  static_cast<const std::uint8_t*>(host_) + page_bytes);
  ASSERT_TRUE(slots().RecordWrite(0, 300));
  Fill(0, *pages, 0xB2);
  ASSERT_TRUE(slots().PrepareRestore(0, 300, footprint, slots().kept_layout_id()));
  EXPECT_FALSE(slots().CheckActive(0));
  EXPECT_FALSE(slots().SetPositions(0, 301));
  EXPECT_FALSE(slots().Borrow(0));
  EXPECT_FALSE(slots().CheckpointRanges(0, 300).has_value());
  EXPECT_FALSE(slots().usable(0));
  // Its own copies in and completion run; then steps do again.
  std::memcpy(host_, saved.data(), saved.size());
  ASSERT_TRUE(slots().CopyState(0, host_, *pages, false));
  ASSERT_TRUE(slots().CompleteRestore(0, 300));
  EXPECT_TRUE(slots().CheckActive(0));
  EXPECT_TRUE(slots().SetPositions(0, 300));
  EXPECT_TRUE(All(0, Used(0), 0xA1));
  // A clear (the runtime's answer to a failed or cancelled restore) is not
  // a step: it runs mid-restore and ends it.
  ASSERT_TRUE(slots().PrepareRestore(0, 300, footprint, slots().kept_layout_id()));
  ASSERT_TRUE(slots().Clear(0));
  EXPECT_FALSE(slots().restoring(0));
  EXPECT_TRUE(slots().CheckActive(0));
}

// L2: a spilled slot, or an adopted one not yet read back, is in no
// closure: no job writes it, so a write record or positions for it are
// refused (an adopted slot's Restore would otherwise drop the record
// unseen), and nothing changes.
TEST_F(SlotLifecycleTest, ASpilledOrAdoptedSlotTakesNoWriteRecordOrPositions) {
  Select({0});
  ASSERT_TRUE(slots().ReserveStateThrough(0, 300));
  ASSERT_TRUE(slots().SetPositions(0, 300));
  Fill(0, Used(0), 0xA1);
  ASSERT_TRUE(slots().Spill(0));
  const std::array<Range, 1> late = {Range{.region = 0, .offset = 300 * 8192, .bytes = 8192}};
  EXPECT_FALSE(slots().RecordWrite(0, 300));
  EXPECT_FALSE(slots().RecordWrite(0, late));
  EXPECT_FALSE(slots().SetPositions(0, 400));
  EXPECT_EQ(slots().positions(0), 300U);
  EXPECT_TRUE(slots().kept_whole(0));
  ASSERT_TRUE(slots().Restore(0));
  EXPECT_EQ(slots().spill_write_bytes(0), 0U);
  // A range naming no stage is refused too.
  const std::array<Range, 1> nowhere = {Range{.region = 3, .offset = 0, .bytes = 256}};
  EXPECT_FALSE(slots().RecordWrite(0, nowhere));
  // Adopted.
  const auto stages = Stages();
  const std::vector<Range> footprint = {
      {.region = 0, .offset = 0, .bytes = kExtent},
      {.region = 0, .offset = kExtent, .bytes = kExtent},
      {.region = 0, .offset = 4 * kExtent, .bytes = stages[0].layout.bytes - (4 * kExtent)},
      {.region = 1, .offset = 0, .bytes = kExtent},
      {.region = 2, .offset = 0, .bytes = stages[2].layout.bytes}};
  ASSERT_TRUE(slots().Adopt(2, 300, footprint, slots().kept_layout_id()));
  EXPECT_FALSE(slots().RecordWrite(2, 0));
  EXPECT_FALSE(slots().SetPositions(2, 0));
  EXPECT_EQ(slots().positions(2), 300U);
  ASSERT_TRUE(slots().Restore(2));
  EXPECT_TRUE(slots().kept_whole(2));
  ASSERT_TRUE(slots().RecordWrite(2, 299));
  EXPECT_FALSE(slots().kept_whole(2));
}

// A stage's owed commit is the model's own code: one that writes another
// stage's state (as a target's commit of a drafter's rows could) must not
// leave that stage's file marked current by a write-back while it is owed.
TEST_F(SlotLifecycleTest, AnOwedCommitIntoAnotherStageKeepsThatStagesFileBehind) {
  Select({0});
  ASSERT_TRUE(slots().ReserveStateThrough(0, 300));
  ASSERT_TRUE(slots().SetPositions(0, 300));
  Fill(0, Used(0), 0xA1);
  ASSERT_TRUE(slots().Spill(0));
  ASSERT_TRUE(slots().Restore(0));
  const Range drafter{.region = 1, .offset = 0, .bytes = 4096};
  const std::uint64_t at = slots().live(0, 1).base(0);
  slots().live(0, 0).SetCommit(
      [at](pr::NativeStream native, std::uint32_t) -> en::LiveState::Queued {
        if (!pr::FillAsync(native, en::support::Pointer(at), 0xEE, 4096).ok()) {
          return std::unexpected(en::LiveState::QueueFailure{.unknown = true, .detail = "fill"});
        }
        return {};
      });
  ASSERT_TRUE(slots().RecordWrite(0, 300));
  slots().live(0, 0).Verified(2);
  ASSERT_TRUE(slots().live(0, 0).Accept(1));
  slots().StateWrittenBack(true);  // a swap's, while the commit is owed
  ASSERT_TRUE(slots().live(0, 0).Rollback(node_, slots().slot_fence(0), 0, "test settle"));
  ASSERT_TRUE(slots().Spill(0));
  ASSERT_TRUE(slots().Restore(0));
  EXPECT_TRUE(All(0, std::array{drafter}, 0xEE));
}

// The guards leave every legitimate serving sequence open: a swap's
// write-back after an Accept, with the owed work run by the next step; a
// borrow once what an Accept owed has run; a checkpoint restore after a
// settle; and recovery after a hang, a slot owing work kept (resident and
// untouched, or spilled) and settled afterwards.
TEST_F(SlotLifecycleTest, ServingSequencesAroundAnAcceptStayOpen) {
  Select({0, 1});
  ASSERT_TRUE(node_.BeginRequest(0, slots().execution(), "test request"));
  ASSERT_TRUE(slots().ReserveStateThrough(0, 300));
  ASSERT_TRUE(slots().ReserveStateThrough(1, 300));
  ASSERT_TRUE(slots().SetPositions(0, 300));
  ASSERT_TRUE(slots().SetPositions(1, 300));
  Fill(0, Used(0), 0xA1);
  Fill(1, Used(1), 0xC3);
  const auto footprint = Used(0);
  auto pages = slots().CheckpointRanges(0, 300);
  ASSERT_TRUE(pages) << pages.error();
  std::uint64_t page_bytes = 0;
  for (const auto& p : *pages) page_bytes += p.bytes;
  ASSERT_TRUE(slots().CopyState(0, host_, *pages, true));
  std::vector<std::uint8_t> saved(static_cast<const std::uint8_t*>(host_),
                                  static_cast<const std::uint8_t*>(host_) + page_bytes);
  const std::array<Range, 1> fixed = {Range{.region = 0, .offset = 4 * kExtent, .bytes = 4096}};
  // A swap's write-back after an Accept; the next step runs what it owes.
  OweCommit(0, 300, 0xEE);
  ASSERT_TRUE(node_.EndRequest(0));
  slots().StateWrittenBack(true);
  ASSERT_TRUE(node_.BeginRequest(0, slots().execution(), "test request"));
  StepRunsOwed(0, 0, 301);
  // Borrowed once settled.
  auto borrow = slots().Borrow(0);
  ASSERT_TRUE(borrow);
  EXPECT_TRUE(All(0, fixed, 0xEE));
  borrow->Return();
  ASSERT_TRUE(node_.EndRequest(0));
  ASSERT_TRUE(slots().Spill(0));
  ASSERT_TRUE(slots().Restore(0));
  ASSERT_TRUE(node_.BeginRequest(0, slots().execution(), "test request"));
  EXPECT_TRUE(All(0, fixed, 0xEE));
  // A checkpoint restore after a settle.
  OweCommit(0, 300, 0xDD);
  ASSERT_TRUE(slots().live(0, 0).Rollback(node_, slots().slot_fence(0), 0, "test settle"));
  auto prepared = slots().PrepareRestore(0, 300, footprint, slots().kept_layout_id());
  ASSERT_TRUE(prepared) << llmp::test_support::Failed(prepared).value_or("");
  std::memcpy(host_, saved.data(), saved.size());
  ASSERT_TRUE(slots().CopyState(0, host_, *pages, false));
  ASSERT_TRUE(slots().CompleteRestore(0, 300));
  EXPECT_TRUE(All(0, Used(0), 0xA1));
  // A hang with work owed: slot 1 owes a commit, resident and untouched by
  // the failure (slot 0's drafter job, which quarantined it).
  OweCommit(1, 300, 0xEE);
  slots().live(0, 1).Quarantine();
  ASSERT_TRUE(node_.EndRequest(0));
  auto recovered = slots().RecoverInPlace();
  ASSERT_TRUE(recovered) << recovered.error();
  EXPECT_EQ(*recovered, 0b001U);
  EXPECT_TRUE(slots().owed(1));
  Select({1});
  StepRunsOwed(1, 0, 300);
  EXPECT_TRUE(All(1, fixed, 0xEE));
  // Spilled while owing (allowed: the owed work is runtime memory), then a
  // fault the cohort cannot prove local: kept, restored, settled, exact.
  OweCommit(1, 300, 0xBB);
  ASSERT_TRUE(slots().Spill(1));
  Select({0});
  slots().Fault();
  recovered = slots().RecoverInPlace();
  ASSERT_TRUE(recovered) << recovered.error();
  EXPECT_EQ(*recovered & 0b010U, 0U);
  EXPECT_TRUE(slots().spilled(1));
  EXPECT_TRUE(slots().owed(1));
  ASSERT_TRUE(slots().Restore(1));
  Select({1});
  ASSERT_TRUE(slots().live(1, 0).Rollback(node_, slots().slot_fence(1), 0, "test settle"));
  ASSERT_TRUE(slots().Spill(1));
  ASSERT_TRUE(slots().Restore(1));
  EXPECT_TRUE(All(1, fixed, 0xBB));
  const std::array<Range, 1> early = {Range{.region = 0, .offset = 0, .bytes = 300 * 8192}};
  EXPECT_TRUE(All(1, early, 0xC3));
}

// A Gemma-like target: per layer a K and a V cache, a global one a cell a
// position and a local one a ring (window 4,096, chunk 128: 4,352 cells),
// at a context of 8,192 and 2 KiB a cell, so that every tensor spans
// several extents.
class GemmaFootprintTest : public SlotLifecycleTest {
 protected:
  static constexpr std::uint32_t kGemmaContext = 8192, kRows = 128, kWindow = 4096;
  std::vector<en::StageState> MakeStages() const override {
    using C = md::StateTensorContract;
    const auto kv = [](const char* name, bool local) {
      return C{.name = name,
               .kind = local ? md::StateKind::kRing : md::StateKind::kKv,
               .encoding = md::Encoding::kF16,
               .cell_bytes = 2048,
               .cells = local ? md::CellRule::kRing : md::CellRule::kPerPosition,
               .window = local ? kWindow : 0,
               .rollback = local ? md::Rollback::kNone : md::Rollback::kTruncate};
    };
    const std::vector<std::pair<std::uint32_t, md::StateContract>> layers = {
        {0, {.tensors = {kv("k", true), kv("v", true)}}},
        {1, {.tensors = {kv("k", false), kv("v", false)}}}};
    std::vector<en::StageState> out;
    out.push_back(
        {.name = "target", .layout = md::Layout(layers, kGemmaContext, kRows, kExtent).value()});
    return out;
  }
  // The whole extents of the stage's layout that `positions` positions read.
  std::vector<Range> Footprint(std::uint32_t positions) const {
    const auto layout = MakeStages().front().layout;
    std::vector<Range> out;
    for (const auto& r : md::UsedThrough(layout, positions, 256).value()) {
      if (r.bytes == 0) continue;
      for (std::uint64_t e = r.offset / kExtent; e <= (r.offset + r.bytes - 1) / kExtent; ++e) {
        out.push_back({0, e * kExtent, std::min(kExtent, layout.bytes - (e * kExtent))});
      }
    }
    std::ranges::sort(out, {}, &Range::offset);
    out.erase(std::unique(out.begin(), out.end(),
                          [](const Range& a, const Range& b) { return a.offset == b.offset; }),
              out.end());
    return out;
  }
};

// A restored footprint must be an ordered set of whole extents that funds
// what the completed positions read, at the ring's and the context's
// boundaries (Gemma's local layers' ring wraps at 4,352 cells).
TEST_F(GemmaFootprintTest, ARestoredFootprintIsOrderedWholeExtentsAtTheRingAndContextBoundaries) {
  for (const std::uint32_t positions : {1U, 4096U, 4224U, 4353U, 8192U}) {
    SCOPED_TRACE(positions);
    const auto footprint = Footprint(positions);
    ASSERT_FALSE(footprint.empty());
    ASSERT_TRUE(slots().ValidateFootprint(positions, footprint));
    auto bad = footprint;
    bad.pop_back();
    EXPECT_FALSE(slots().ValidateFootprint(positions, bad));  // unfunded
    bad = footprint;
    ++bad[0].offset;
    EXPECT_FALSE(slots().ValidateFootprint(positions, bad));  // not an extent
    bad = footprint;
    --bad[0].bytes;
    EXPECT_FALSE(slots().ValidateFootprint(positions, bad));  // not a whole one
    bad = footprint;
    bad[0].region = 1;
    EXPECT_FALSE(slots().ValidateFootprint(positions, bad));  // no such stage
    bad = footprint;
    bad.push_back(bad.back());
    EXPECT_FALSE(slots().ValidateFootprint(positions, bad));  // repeated
    bad = footprint;
    std::ranges::reverse(bad);
    EXPECT_FALSE(slots().ValidateFootprint(positions, bad));  // out of order
    bad = footprint;
    bad.back().offset = std::numeric_limits<std::uint64_t>::max();
    EXPECT_FALSE(slots().ValidateFootprint(positions, bad));                // past the layout
    EXPECT_FALSE(slots().ValidateFootprint(0, footprint));                  // backing for nothing
    EXPECT_FALSE(slots().ValidateFootprint(kGemmaContext + 1, footprint));  // past the context
  }
  EXPECT_TRUE(slots().ValidateFootprint(0, {}));
}

// A raw/compressed bank begins inside an extent and crosses its boundary.
// A fact proves its virtual layout; only admission proves live backing.
class StateBankFootprintTest : public SlotLifecycleTest {
 protected:
  std::vector<en::StageState> MakeStages() const override {
    const std::vector<std::pair<std::uint32_t, md::StateContract>> layers = {
        {0,
         {.tensors = {{.name = "prefix",
                       .cell_bytes = 3 * kExtent / 4,
                       .cells = md::CellRule::kFixed,
                       .fixed_cells = 1},
                      {.name = "raw",
                       .kind = md::StateKind::kRing,
                       .cell_bytes = 1024,
                       .cells = md::CellRule::kRing,
                       .window = 128,
                       .ring_reads_whole = true},
                      {.name = "compressed",
                       .kind = md::StateKind::kCompressed,
                       .cell_bytes = 1024,
                       .cells = md::CellRule::kPerBlock,
                       .block_ratio = 4,
                       .basis = md::BlockBasis::kPositions,
                       .dummy_last = true}}}}};
    return {{.name = "target", .layout = md::Layout(layers, 65536, 512).value()}};
  }
};

TEST_F(StateBankFootprintTest, WholeSpanCoverageRequiresOnlyTheAdmittedExtents) {
  namespace kn = llmp::kernels;
  const auto layout = MakeStages().front().layout;
  ASSERT_EQ(layout.items[1].offset, 3 * kExtent / 4);
  const std::array slices{md::StateSlice{.tensor = 1, .cells = layout.items[1].cells},
                          md::StateSlice{.tensor = 2, .cells = 512}};
  const auto bank = md::ContiguousStateSpan(layout, slices);
  ASSERT_TRUE(bank);
  Select({0});
  const auto base = slots().live(0, 0).base(0);
  const std::array use{kn::TensorUse{.address = base + bank->range.offset,
                                     .bytes = bank->range.bytes,
                                     .role = llmp::ir::Role::kState}};
  en::Coverage before;
  en::CheckCoverage(node_, 0, use, "unbacked bank ", before);
  EXPECT_EQ(before.violations, 0U);  // state coverage checks its virtual reservation
  EXPECT_FALSE(node_.Covered(use[0].address, use[0].bytes, 0, true));
  ASSERT_TRUE(slots().ReserveStateThrough(0, 2048, 2048));
  ASSERT_TRUE(slots().SetPositions(0, 2048));
  // The bank intersects extents 0 and 1; the last dummy is in extent 9.
  // The rest of the virtual cache remains unbacked.
  EXPECT_EQ(slots().live(0, 0).extents().size(), 3U);
  en::Coverage admitted;
  en::CheckCoverage(node_, 0, use, "admitted bank ", admitted);
  EXPECT_EQ(admitted.violations, 0U) << admitted.first_violation;
  EXPECT_EQ(node_.Covered(use[0].address, use[0].bytes, 0, true),
            llmp::catalog::MemoryClass::kLiveState);
  const std::array ranges{
      Range{.region = 0, .offset = bank->range.offset, .bytes = bank->range.bytes}};
  Fill(0, ranges, 0x7E);
  EXPECT_TRUE(All(0, ranges, 0x7E));
  // A wider prefix is physically contiguous but reaches absent extent 2.
  // The resident-coverage audit rejects it despite a valid layout fact.
  // The ordinary state coverage check still recognizes its reservation.
  auto wider = slices;
  wider[1].cells = 2048;
  const auto outside = md::ContiguousStateSpan(layout, wider);
  ASSERT_TRUE(outside);
  const std::array absent{kn::TensorUse{.address = base + outside->range.offset,
                                        .bytes = outside->range.bytes,
                                        .role = llmp::ir::Role::kState}};
  en::Coverage rejected;
  en::CheckCoverage(node_, 0, absent, "unadmitted bank ", rejected);
  EXPECT_EQ(rejected.violations, 0U);
  EXPECT_FALSE(node_.Covered(absent[0].address, absent[0].bytes, 0, true));
  EXPECT_EQ(slots().live(0, 0).extents().size(), 3U);
}

// Native scoring loads complete 64-key tiles within the declared C, even
// when visibility is zero. Admission follows the typed key view, not counts.
class IndexKeyFootprintTest : public SlotLifecycleTest {
 protected:
  std::vector<en::StageState> MakeStages() const override {
    const std::vector<std::pair<std::uint32_t, md::StateContract>> layers = {
        {0,
         {.tensors = {{.name = "prefix",
                       .cell_bytes = kExtent - 256,
                       .cells = md::CellRule::kFixed,
                       .fixed_cells = 1},
                      {.name = "index",
                       .kind = md::StateKind::kIndex,
                       .cell_bytes = 256,
                       .cells = md::CellRule::kPerBlock,
                       .block_ratio = 4,
                       .basis = md::BlockBasis::kPositions,
                       .dummy_last = true}}}}};
    return {{.name = "target", .layout = md::Layout(layers, 131072, 512).value()}};
  }
};

TEST_F(IndexKeyFootprintTest, PaddedKeyClosureIsBackedIndependentlyOfVisibleCounts) {
  const auto layout = MakeStages().front().layout;
  const auto& key = layout.items[1];
  ASSERT_EQ(key.offset, kExtent - 256);
  ASSERT_EQ((key.offset + (key.cells - 1) * key.contract.cell_bytes) / kExtent, 4U);
  const auto used = md::UsedThrough(layout, 2048, 2048);
  ASSERT_TRUE(used);
  constexpr std::uint64_t cells = 512;
  EXPECT_TRUE(std::ranges::any_of(*used, [&](const md::StateRange& r) {
    return r.offset <= key.offset && key.offset + cells * 256 <= r.offset + r.bytes;
  }));
  Select({0});
  const auto address = slots().live(0, 0).base(0) + key.offset;
  EXPECT_FALSE(node_.Covered(address, cells * 256, 0, true));
  ASSERT_TRUE(slots().ReserveStateThrough(0, 2048, 2048));
  for (const std::int32_t visible : {0, 1, 511, 512}) {
    SCOPED_TRACE(visible);
    EXPECT_EQ(node_.Covered(address, cells * 256, 0, true), llmp::catalog::MemoryClass::kLiveState);
  }
  // Prefix+keys cross extent0/1, and only the last dummy needs extent4.
  EXPECT_EQ(slots().live(0, 0).extents().size(), 3U);
  EXPECT_FALSE(node_.Covered(slots().live(0, 0).base(0) + 2 * kExtent, 256, 0, true));
  EXPECT_FALSE(node_.Covered(address, 12288 * 256, 0, true));  // absent extent2
  EXPECT_EQ(slots().live(0, 0).extents().size(), 3U);
}

class LowCapacitySlotTest : public SlotLifecycleTest {
 protected:
  std::uint64_t BudgetExtents() const override { return 16; }
};

TEST_F(LowCapacitySlotTest, ARestoreRefusedForCapacityLeavesTheSlotSpilledAndUsable) {
  // Slot 0's state (5 extents) spilled; slot 1's (8) held in a request,
  // beside the host buffer (4): no room to bring slot 0 back.
  Select({0});
  ASSERT_TRUE(slots().ReserveStateThrough(0, 300));
  Fill(0, Used(0), 0xE5);
  const auto used = Used(0);
  ASSERT_TRUE(slots().Spill(0));
  Select({1});
  ASSERT_TRUE(node_.BeginRequest(0, slots().execution(), "test request"));
  ASSERT_TRUE(slots().ReserveStateThrough(1, 1000));
  auto refused = slots().Restore(0);
  EXPECT_FALSE(refused);
  EXPECT_TRUE(slots().state_refused(0));
  EXPECT_TRUE(slots().spilled(0));
  EXPECT_TRUE(slots().usable(0));
  EXPECT_FALSE(slots().faulted());
  // Room made (the runtime's reclaim order spills the idle peer), it comes
  // back exactly.
  ASSERT_TRUE(node_.EndRequest(0));
  ASSERT_TRUE(slots().Spill(1));
  auto restored = slots().Restore(0);
  ASSERT_TRUE(restored) << llmp::test_support::Failed(restored).value_or("");
  EXPECT_FALSE(slots().state_refused(0));
  Select({0});
  EXPECT_TRUE(All(0, used, 0xE5));
}

TEST_F(LowCapacitySlotTest, AGrowthRefusedInALaterStageKeepsTheEarlierStagesFreshPagesHeld) {
  Select({0});
  ASSERT_TRUE(node_.BeginRequest(0, slots().execution(), "test request"));
  ASSERT_TRUE(slots().ReserveStateThrough(0, 300));
  ASSERT_TRUE(slots().SetPositions(0, 300));
  const auto before = Used(0);
  Fill(0, before, 0xA1);
  // To 1000 positions the target's cache needs two more extents and the
  // drafter's one: room for the target's alone.
  void* pad = Leave(2);
  ASSERT_NE(pad, nullptr);
  EXPECT_FALSE(slots().ReserveStateThrough(0, 1000));
  EXPECT_TRUE(slots().state_refused(0));
  EXPECT_TRUE(slots().usable(0));
  EXPECT_FALSE(slots().faulted());
  EXPECT_EQ(slots().live(0, 0).extents().size(), 5U);
  EXPECT_EQ(slots().live(0, 1).extents().size(), 1U);
  // The target's fresh pages are leased with the rest of the slot.
  for (const auto id : slots().live(0, 0).extents()) {
    EXPECT_TRUE(std::ranges::any_of(slots().execution().extents,
                                    [id](const auto& e) { return e.first == id; }));
  }
  EXPECT_TRUE(All(0, before, 0xA1));
  // Room made, it grows whole, its contents as they were.
  ASSERT_TRUE(node_.FreePinned(pad));
  ASSERT_TRUE(slots().ReserveStateThrough(0, 1000));
  EXPECT_FALSE(slots().state_refused(0));
  EXPECT_EQ(slots().live(0, 1).extents().size(), 2U);
  EXPECT_TRUE(All(0, before, 0xA1));
  ASSERT_TRUE(node_.EndRequest(0));
}

TEST_F(LowCapacitySlotTest, ACheckpointRestoreRefusedInALaterStageDropsNothing) {
  Select({0});
  ASSERT_TRUE(node_.BeginRequest(0, slots().execution(), "test request"));
  ASSERT_TRUE(slots().ReserveStateThrough(0, 300));
  ASSERT_TRUE(slots().SetPositions(0, 300));
  const auto before = Used(0);
  Fill(0, before, 0xA1);
  // A footprint at 200 positions that drops the target's second cache
  // extent and adds the drafter cache's second: the drafter's growth
  // has no room.
  const auto stages = Stages();
  const std::vector<Range> footprint = {
      {.region = 0, .offset = 0, .bytes = kExtent},
      {.region = 0, .offset = 4 * kExtent, .bytes = stages[0].layout.bytes - (4 * kExtent)},
      {.region = 1, .offset = 0, .bytes = kExtent},
      {.region = 1, .offset = kExtent, .bytes = kExtent},
      {.region = 2, .offset = 0, .bytes = stages[2].layout.bytes}};
  void* pad = Leave(0);
  invalidated_.clear();
  EXPECT_FALSE(slots().PrepareRestore(0, 200, footprint, slots().kept_layout_id()));
  EXPECT_TRUE(invalidated_.empty());  // what its runner derived still holds
  EXPECT_TRUE(slots().state_refused(0));
  EXPECT_FALSE(slots().restoring(0));
  EXPECT_TRUE(slots().usable(0));
  // Nothing dropped: the target's second extent is still there, as it was.
  EXPECT_EQ(Used(0).size(), before.size());
  EXPECT_TRUE(All(0, before, 0xA1));
  // Room made, the footprint exactly.
  if (pad != nullptr) ASSERT_TRUE(node_.FreePinned(pad));
  auto prepared = slots().PrepareRestore(0, 200, footprint, slots().kept_layout_id());
  ASSERT_TRUE(prepared) << llmp::test_support::Failed(prepared).value_or("");
  const auto now = Used(0);
  ASSERT_EQ(now.size(), footprint.size());
  for (std::size_t i = 0; i < now.size(); ++i) {
    EXPECT_EQ(now[i].region, footprint[i].region);
    EXPECT_EQ(now[i].offset, footprint[i].offset);
  }
  std::memset(host_, 0x5A, 8 * kExtent);
  ASSERT_TRUE(slots().CopyState(0, host_, footprint, false));
  ASSERT_TRUE(slots().CompleteRestore(0, 200));
  EXPECT_TRUE(All(0, footprint, 0x5A));
  ASSERT_TRUE(node_.EndRequest(0));
}

TEST_F(LowCapacitySlotTest, AnAdoptedRestoreRefusedPartwayResumesFromTheFiles) {
  // The kept state at 300 positions: the target's three extents, then
  // each drafter's one; room for the target's alone.
  const auto stages = Stages();
  const std::vector<Range> footprint = {
      {.region = 0, .offset = 0, .bytes = kExtent},
      {.region = 0, .offset = kExtent, .bytes = kExtent},
      {.region = 0, .offset = 4 * kExtent, .bytes = stages[0].layout.bytes - (4 * kExtent)},
      {.region = 1, .offset = 0, .bytes = kExtent},
      {.region = 2, .offset = 0, .bytes = stages[2].layout.bytes}};
  ASSERT_TRUE(slots().Adopt(2, 300, footprint, slots().kept_layout_id()));
  void* pad = Leave(3);
  ASSERT_NE(pad, nullptr);
  EXPECT_FALSE(slots().Restore(2));
  EXPECT_TRUE(slots().state_refused(2));
  EXPECT_TRUE(slots().spilled(2));
  EXPECT_TRUE(slots().adopted(2));
  EXPECT_TRUE(slots().usable(2));
  EXPECT_FALSE(slots().kept_whole(2));
  // The target's pages read and held while the drafter grew: not taken
  // as its victims.
  EXPECT_EQ(slots().live(2, 0).extents().size(), 3U);
  EXPECT_TRUE(slots().live(2, 1).extents().empty());
  // Evicted meanwhile (written back to its file), then the rest made room.
  ASSERT_TRUE(node_.Evict(slots().live(2, 0).extents()));
  ASSERT_TRUE(node_.FreePinned(pad));
  auto restored = slots().Restore(2);
  ASSERT_TRUE(restored) << llmp::test_support::Failed(restored).value_or("");
  EXPECT_FALSE(slots().adopted(2));
  EXPECT_FALSE(slots().spilled(2));
  EXPECT_TRUE(slots().kept_whole(2));
  Select({2});
  const std::array<Range, 3> marks = {Range{.region = 0, .offset = 7, .bytes = 1},
                                      Range{.region = 1, .offset = 7, .bytes = 1},
                                      Range{.region = 2, .offset = 7, .bytes = 1}};
  ASSERT_TRUE(slots().CopyState(2, host_, marks, true));
  const auto* p = static_cast<const std::uint8_t*>(host_);
  EXPECT_EQ(p[0], 0x50);
  EXPECT_EQ(p[1], 0x51);
  EXPECT_EQ(p[2], 0x52);
}

}  // namespace
