// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "engine/live_state.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <expected>
#include <filesystem>
#include <span>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "base/bytes.h"
#include "engine/checkpoint_file.h"
#include "engine/paged_node.h"
#include "engine/support.h"
#include "expected_error.h"
#include "platform/direct_io.h"
#include "providers/device_runtime.h"
#include "runtime/intake_limits.h"
#include "scheduler/commands.h"

namespace jitllm::engine {
struct LiveStatePlacementTestAccess {
  static bool Check(LiveState& state, const scheduler::Scheduler& scheduler, PlaceCheck& check) {
    return state.CheckPlacesImpl(scheduler, check);
  }
};
}  // namespace jitllm::engine

namespace {

namespace en = jitllm::engine;
namespace pr = jitllm::providers;
using jitllm::base::Bytes;
constexpr std::uint64_t kExtent = en::kPagedExtent;

class BareState final : public en::PagedModel {
 public:
  explicit BareState(en::PagedNode& node) : node_(node) {}
  en::LiveState live{"test"};

  std::uint32_t stream() const override { return 0; }
  const jitllm::catalog::Closure& fence_closure() const override { return fence_; }
  std::vector<jitllm::catalog::ExtentId> managed_extents() const override { return live.extents(); }
  en::Status Refresh() {
    return node_.Call(
        [&]() -> en::Status {
          auto closure = node_.catalog().ClosureOfExtents(live.extents());
          if (!closure) {
            return std::unexpected("test state closure");
          }
          fence_ = std::move(*closure);
          return {};
        },
        "test state closure");
  }
  en::Status Release() override {
    std::vector<std::string> errors;
    live.Release(node_.memory(), errors);
    if (!errors.empty()) {
      return std::unexpected(errors.front());
    }
    return {};
  }

 private:
  en::PagedNode& node_;
  jitllm::catalog::Closure fence_;
};

class LiveStateTest : public ::testing::Test {
 protected:
  en::PagedNode node_{
      en::NodeSettings{.compute_streams = 1, .slots = 2, .copy_lane = true, .slot_bytes = kExtent}};
  BareState model_{node_};
  std::uint64_t fixed_ = 0;
  bool running_ = false;
  virtual std::uint64_t BudgetExtents() const { return 16; }

  void SetUp() override {
    auto opened = node_.Open();
    ASSERT_TRUE(opened) << jitllm::test_support::Failed(opened).value_or("");
    fixed_ = node_.catalog().OccupancyOf(node_.domain()).Total().value();
    auto added = model_.live.AddGrowing(node_, "a cache", 8 * kExtent, 0);
    ASSERT_TRUE(added) << jitllm::test_support::Failed(added).value_or("");
    auto started = node_.Start(Bytes(fixed_ + (BudgetExtents() * kExtent)));
    ASSERT_TRUE(started) << jitllm::test_support::Failed(started).value_or("");
    // NOLINTNEXTLINE(concurrency-mt-unsafe): test environment is immutable
    const char* scratch = std::getenv("JITLLM_TEST_SCRATCH");
    ASSERT_NE(scratch, nullptr);
    auto registered = model_.live.RegisterSpill(node_, std::filesystem::path(scratch));
    ASSERT_TRUE(registered) << jitllm::test_support::Failed(registered).value_or("");
    node_.Run();
    running_ = true;
  }
  void TearDown() override {
    if (running_) {
      auto refreshed = model_.Refresh();
      EXPECT_TRUE(refreshed) << jitllm::test_support::Failed(refreshed).value_or("");
    }
    std::array<en::PagedModel*, 1> models = {&model_};
    auto closed = node_.TearDown(models);
    EXPECT_TRUE(closed) << jitllm::test_support::Failed(closed).value_or("");
  }

  bool PlacementMemoHit() {
    bool hit = false;
    en::PlaceCheck check;
    auto checked = node_.Call(
        [&]() -> en::Status {
          hit = en::LiveStatePlacementTestAccess::Check(model_.live, node_.scheduler(), check);
          return {};
        },
        "test state placement check");
    EXPECT_TRUE(checked) << jitllm::test_support::Failed(checked).value_or("");
    EXPECT_EQ(check.moved, 0U) << check.first;
    return hit;
  }

  jitllm::catalog::ExtentView Describe(jitllm::catalog::ExtentId id) {
    jitllm::catalog::ExtentView view;
    auto read = node_.Call(
        [&]() -> en::Status {
          view = node_.catalog().Describe(id).value();
          return {};
        },
        "test extent view");
    EXPECT_TRUE(read) << jitllm::test_support::Failed(read).value_or("");
    return view;
  }

  // Every one of `ids` nonresident: its backing released.
  bool Released(const std::vector<jitllm::catalog::ExtentId>& ids) {
    return std::ranges::all_of(ids, [&](auto id) {
      return Describe(id).state == jitllm::catalog::ExtentState::kNonresident;
    });
  }

  // Every used extent filled with `value` (a job over the state's fence).
  en::Status Pattern(std::uint8_t value) {
    std::vector<std::uint64_t> bases;
    for (const auto& range : model_.live.used_ranges())
      bases.push_back(model_.live.base(range.region) + range.offset);
    return node_.Job(
        model_.fence_closure(),
        [bases, value](pr::NativeStream native) {
          for (const auto base : bases)
            if (!pr::FillAsync(native, en::support::Pointer(base), value, kExtent).ok())
              return jitllm::scheduler::JobResult::kUnknown;
          return jitllm::scheduler::JobResult::kQueued;
        },
        "test pattern", 0);
  }

  // The used state's bytes, packed.
  std::vector<std::byte> Read() {
    std::vector<std::byte> copy;
    std::array<std::vector<std::byte>*, 1> output = {&copy};
    auto read = model_.live.Read(node_, model_.fence_closure(), 0, "test read", output);
    EXPECT_TRUE(read) << jitllm::test_support::Failed(read).value_or("");
    return copy;
  }
  std::byte Contents(std::size_t at) {
    const auto copy = Read();
    return at < copy.size() ? copy[at] : std::byte{0xFF};
  }
  bool AllZero() {
    const auto copy = Read();
    return !copy.empty() &&
           std::ranges::all_of(copy, [](std::byte b) { return b == std::byte{0}; });
  }

  std::uint64_t Occupancy() {
    std::uint64_t bytes = 0;
    auto read = node_.Call(
        [&]() -> en::Status {
          bytes = node_.catalog().OccupancyOf(node_.domain()).Total().value();
          return {};
        },
        "test occupancy");
    EXPECT_TRUE(read) << jitllm::test_support::Failed(read).value_or("");
    return bytes;
  }
};

TEST_F(LiveStateTest, PlacementMemoRechecksGrowthTrimClearAndRestoreWithoutCachingResidency) {
  // Runners pin registered source places during Setup. This does not make the
  // pages resident or acquire a request lease.
  ASSERT_TRUE(node_.Call(
      [&]() -> en::Status {
        if (auto r = node_.scheduler().PinPlaces(model_.live.reserved_extents()); !r)
          return std::unexpected("test pin state places");
        return {};
      },
      "test pin state places"));
  EXPECT_FALSE(PlacementMemoHit());
  EXPECT_TRUE(PlacementMemoHit());
  const std::array<en::LiveState::Range, 2> ranges = {
      en::LiveState::Range{.region = 0, .offset = 0, .bytes = 16},
      en::LiveState::Range{.region = 0, .offset = 4 * kExtent, .bytes = 16}};
  ASSERT_TRUE(model_.live.Use(node_, ranges));
  ASSERT_TRUE(model_.Refresh());
  EXPECT_FALSE(PlacementMemoHit());
  EXPECT_TRUE(PlacementMemoHit());
  auto again = model_.live.Use(node_, ranges);
  ASSERT_TRUE(again);
  EXPECT_FALSE(*again);
  EXPECT_TRUE(PlacementMemoHit());  // Already initialized: no local mutation.

  ASSERT_TRUE(node_.Evict(model_.live.extents()));
  EXPECT_EQ(Occupancy(), fixed_);
  EXPECT_TRUE(PlacementMemoHit());  // Sources/pins still match; pages are absent.
  ASSERT_TRUE(model_.Refresh());
  void* pinned = model_.live.HostCopy(node_, 16);
  ASSERT_NE(pinned, nullptr);
  ASSERT_TRUE(
      model_.live.Copy(node_, model_.fence_closure(), 0, pinned, std::span(ranges).first(1), true));
  EXPECT_TRUE(PlacementMemoHit());  // Actual page-in still does not relocate.
  std::memset(pinned, 0x5A, 16);
  ASSERT_TRUE(model_.live.Copy(node_, model_.fence_closure(), 0, pinned, std::span(ranges).first(1),
                               false));
  EXPECT_FALSE(PlacementMemoHit());
  EXPECT_TRUE(PlacementMemoHit());
  std::memset(pinned, 0, 16);
  ASSERT_TRUE(
      model_.live.Copy(node_, model_.fence_closure(), 0, pinned, std::span(ranges).first(1), true));
  EXPECT_TRUE(std::all_of(static_cast<std::byte*>(pinned), static_cast<std::byte*>(pinned) + 16,
                          [](std::byte b) { return b == std::byte{0x5A}; }));

  ASSERT_TRUE(model_.live.Retain(node_, std::span(ranges).first(1)));
  EXPECT_FALSE(PlacementMemoHit());
  EXPECT_TRUE(PlacementMemoHit());
  EXPECT_EQ(model_.live.extents().size(), 1U);
  ASSERT_TRUE(model_.live.Clear(node_, model_.fence_closure(), 0, "clear memo test"));
  EXPECT_FALSE(PlacementMemoHit());
  EXPECT_TRUE(PlacementMemoHit());
  EXPECT_TRUE(model_.live.extents().empty());
  ASSERT_TRUE(model_.live.Use(node_, std::span(ranges).first(1)));
  ASSERT_TRUE(model_.Refresh());
  EXPECT_FALSE(PlacementMemoHit());
  EXPECT_TRUE(PlacementMemoHit());
  ASSERT_TRUE(
      model_.live.Copy(node_, model_.fence_closure(), 0, pinned, std::span(ranges).first(1), true));
  EXPECT_TRUE(std::all_of(static_cast<std::byte*>(pinned), static_cast<std::byte*>(pinned) + 16,
                          [](std::byte b) { return b == std::byte{0}; }));
  ASSERT_TRUE(node_.Call(
      [&]() -> en::Status {
        node_.scheduler().UnpinPlaces(model_.live.reserved_extents());
        return {};
      },
      "test unpin state places"));
}

TEST_F(LiveStateTest, OnlyUsedExtentsAreMappedAndTheirFirstContentsAreZero) {
  EXPECT_TRUE(model_.live.extents().empty());
  EXPECT_EQ(model_.live.reserved_extents().size(), 8);
  EXPECT_EQ(Occupancy(), fixed_);
  const std::array<en::LiveState::Range, 2> ranges = {
      en::LiveState::Range{.region = 0, .offset = 7 * kExtent, .bytes = 16},
      en::LiveState::Range{.region = 0, .offset = (7 * kExtent) + 8, .bytes = 16}};
  auto used = model_.live.Use(node_, ranges);
  ASSERT_TRUE(used) << jitllm::test_support::Failed(used).value_or("");
  EXPECT_TRUE(*used);
  EXPECT_EQ(model_.live.extents().size(), 1);
  EXPECT_EQ(Occupancy(), fixed_ + kExtent);
  ASSERT_TRUE(model_.Refresh());
  std::vector<std::byte> copy;
  std::array<std::vector<std::byte>*, 1> output = {&copy};
  ASSERT_TRUE(model_.live.Read(node_, model_.fence_closure(), 0, "test state", output));
  EXPECT_EQ(copy.size(), 8 * kExtent);
  EXPECT_TRUE(std::ranges::all_of(copy, [](std::byte b) { return b == std::byte{0}; }));
  auto again = model_.live.Use(node_, ranges);
  ASSERT_TRUE(again);
  EXPECT_FALSE(*again);
}

// An admission's estimate (Llm::StateBytesThrough): what used_bytes() would
// be with only these ranges used, each extent once, materializing nothing.
TEST_F(LiveStateTest, UsedBytesOfIsWhatUseWouldLeaveAndMapsNothing) {
  const std::array<en::LiveState::Range, 3> ranges = {
      en::LiveState::Range{.region = 0, .offset = 0, .bytes = 16},
      en::LiveState::Range{.region = 0, .offset = 8, .bytes = 16},  // the same extent
      en::LiveState::Range{.region = 0, .offset = (3 * kExtent) + 5, .bytes = kExtent}};
  auto estimate = model_.live.UsedBytesOf(ranges);
  ASSERT_TRUE(estimate) << jitllm::test_support::Failed(estimate).value_or("");
  EXPECT_EQ(*estimate, 3 * kExtent);
  EXPECT_TRUE(model_.live.extents().empty());
  EXPECT_EQ(Occupancy(), fixed_);
  ASSERT_TRUE(model_.live.Use(node_, ranges));
  EXPECT_EQ(model_.live.used_bytes(), *estimate);
  const std::array<en::LiveState::Range, 1> outside = {
      en::LiveState::Range{.region = 0, .offset = 8 * kExtent, .bytes = 1}};
  EXPECT_FALSE(model_.live.UsedBytesOf(outside));
}

TEST_F(LiveStateTest, InvalidRangesCannotPartiallyGrowState) {
  const std::array<en::LiveState::Range, 2> ranges = {
      en::LiveState::Range{.region = 0, .offset = 0, .bytes = 1},
      en::LiveState::Range{.region = 0, .offset = 8 * kExtent, .bytes = 1}};
  EXPECT_FALSE(model_.live.Use(node_, ranges));
  EXPECT_TRUE(model_.live.extents().empty());
  EXPECT_EQ(Occupancy(), fixed_);
}

TEST_F(LiveStateTest, GrowthAndEvictionPreserveTheInitializedRangesAtTheirAddresses) {
  const std::uint64_t address = model_.live.base(0);
  const std::array<en::LiveState::Range, 1> first = {
      en::LiveState::Range{.region = 0, .offset = 0, .bytes = kExtent}};
  ASSERT_TRUE(model_.live.Use(node_, first));
  ASSERT_TRUE(model_.Refresh());
  ASSERT_TRUE(node_.Job(
      model_.fence_closure(),
      [&](pr::NativeStream stream) {
        return pr::FillAsync(stream, en::support::Pointer(address), 0x5A, kExtent).ok()
                   ? jitllm::scheduler::JobResult::kQueued
                   : jitllm::scheduler::JobResult::kUnknown;
      },
      "test pattern", 0));
  const std::array<en::LiveState::Range, 1> later = {
      en::LiveState::Range{.region = 0, .offset = 4 * kExtent, .bytes = kExtent}};
  ASSERT_TRUE(model_.live.Use(node_, later));
  EXPECT_EQ(model_.live.base(0), address);
  EXPECT_EQ(model_.live.extents().size(), 2);
  ASSERT_TRUE(node_.Evict(model_.live.extents()));
  EXPECT_EQ(Occupancy(), fixed_);
  ASSERT_TRUE(model_.Refresh());
  std::vector<std::byte> copy;
  std::array<std::vector<std::byte>*, 1> output = {&copy};
  ASSERT_TRUE(model_.live.Read(node_, model_.fence_closure(), 0, "restored state", output));
  EXPECT_TRUE(std::all_of(copy.begin(), copy.begin() + kExtent,
                          [](std::byte b) { return b == std::byte{0x5A}; }));
  EXPECT_TRUE(std::all_of(copy.begin() + kExtent, copy.end(),
                          [](std::byte b) { return b == std::byte{0}; }));
  // Two live pages plus their cataloged packed host copy.
  EXPECT_EQ(Occupancy(), fixed_ + (4 * kExtent));
}

TEST_F(LiveStateTest, ClearReleasesResidentAndSavedPagesAndRegrowthStartsWithZeros) {
  const std::array<en::LiveState::Range, 1> first = {
      en::LiveState::Range{.region = 0, .offset = 0, .bytes = kExtent}};
  ASSERT_TRUE(model_.live.Use(node_, first));
  ASSERT_TRUE(model_.Refresh());
  ASSERT_TRUE(node_.Job(
      model_.fence_closure(),
      [&](pr::NativeStream stream) {
        return pr::FillAsync(stream, en::support::Pointer(model_.live.base(0)), 0x5A, kExtent).ok()
                   ? jitllm::scheduler::JobResult::kQueued
                   : jitllm::scheduler::JobResult::kUnknown;
      },
      "test pattern", 0));
  ASSERT_TRUE(node_.Evict(model_.live.extents()));
  const std::array<en::LiveState::Range, 1> second = {
      en::LiveState::Range{.region = 0, .offset = 4 * kExtent, .bytes = kExtent}};
  ASSERT_TRUE(model_.live.Use(node_, second));
  ASSERT_TRUE(model_.Refresh());
  const auto used = model_.live.extents();
  ASSERT_TRUE(node_.Evict({used.front()}));
  ASSERT_TRUE(model_.live.Clear(node_, model_.fence_closure(), 0, "clear test"));
  EXPECT_TRUE(model_.live.extents().empty());
  EXPECT_EQ(Occupancy(), fixed_);
  ASSERT_TRUE(model_.live.Use(node_, first));
  ASSERT_TRUE(model_.Refresh());
  std::vector<std::byte> copy;
  std::array<std::vector<std::byte>*, 1> output = {&copy};
  ASSERT_TRUE(model_.live.Read(node_, model_.fence_closure(), 0, "cleared state", output));
  EXPECT_TRUE(std::ranges::all_of(copy, [](std::byte b) { return b == std::byte{0}; }));
}

// A clear zeroed for reuse keeps resident backing out of the state: the
// slot holds nothing, earlier closures and the saved copy cannot return,
// and growth takes the same backing back with no load.
TEST_F(LiveStateTest, ZeroedClearKeepsBackingOutsideTheStateAndGrowthTakesItBack) {
  const std::array<en::LiveState::Range, 2> both = {
      en::LiveState::Range{.region = 0, .offset = 0, .bytes = 1},
      en::LiveState::Range{.region = 0, .offset = 4 * kExtent, .bytes = 1}};
  ASSERT_TRUE(model_.live.Use(node_, both));
  ASSERT_TRUE(model_.Refresh());
  ASSERT_TRUE(Pattern(0x5A));
  // Written back and restored: the place saved these contents.
  ASSERT_TRUE(node_.Evict(model_.live.extents()));
  ASSERT_TRUE(model_.Refresh());
  ASSERT_EQ(Contents(0), std::byte{0x5A});
  const auto ids = model_.live.extents();
  const jitllm::catalog::Closure stale = model_.fence_closure();
  const std::uint64_t backing = Describe(ids.front()).backing_generation;
  const std::uint64_t occupied = Occupancy();
  auto zeroed = model_.live.ZeroForReuse(node_, model_.fence_closure(), 0);
  ASSERT_TRUE(zeroed) << jitllm::test_support::Failed(zeroed).value_or("");
  EXPECT_TRUE(*zeroed);
  ASSERT_TRUE(model_.live.DiscardGrowingState(node_, true));
  EXPECT_TRUE(model_.live.extents().empty());
  EXPECT_EQ(model_.live.used_bytes(), 0U);
  EXPECT_EQ(model_.live.kept_extents(), ids);
  EXPECT_EQ(Occupancy(), occupied);
  EXPECT_TRUE(Describe(ids.front()).discarded);
  EXPECT_FALSE(node_.Job(
      stale, [](pr::NativeStream) { return jitllm::scheduler::JobResult::kQueued; },
      "stale closure", 0));
  ASSERT_TRUE(model_.live.Use(node_, both));
  EXPECT_EQ(model_.live.extents(), ids);
  EXPECT_TRUE(model_.live.kept_extents().empty());
  EXPECT_EQ(Describe(ids.front()).backing_generation, backing);
  EXPECT_FALSE(Describe(ids.front()).discarded);
  EXPECT_EQ(Occupancy(), occupied);
  ASSERT_TRUE(model_.Refresh());
  EXPECT_TRUE(AllZero());
  // The saved copy is gone and the generation moved: an "unchanged"
  // eviction writes, and the restore is still zero.
  ASSERT_TRUE(node_.Evict(model_.live.extents(), {.unchanged = true}));
  ASSERT_TRUE(model_.Refresh());
  EXPECT_TRUE(AllZero());
}

// Kept backing is a free victim; reclaimed, growth loads zeros as before.
// Anything changing the state between the zeroing and the discard cancels
// the keep, and a discard not zeroed for reuse releases kept backing too.
TEST_F(LiveStateTest, KeptBackingIsReclaimableAndOnlyAnUnchangedZeroingKeepsIt) {
  const std::array<en::LiveState::Range, 2> both = {
      en::LiveState::Range{.region = 0, .offset = 0, .bytes = 1},
      en::LiveState::Range{.region = 0, .offset = 4 * kExtent, .bytes = 1}};
  ASSERT_TRUE(model_.live.Use(node_, both));
  ASSERT_TRUE(model_.Refresh());
  const auto ids = model_.live.extents();
  ASSERT_TRUE(Pattern(0x5A));
  ASSERT_TRUE(model_.live.ZeroForReuse(node_, model_.fence_closure(), 0).value_or(false));
  ASSERT_TRUE(model_.live.DiscardGrowingState(node_, true));
  ASSERT_TRUE(node_.Evict(model_.live.kept_extents()));  // as the reclaim order would
  EXPECT_TRUE(Released(ids));
  ASSERT_TRUE(model_.live.Use(node_, both));
  EXPECT_TRUE(model_.live.kept_extents().empty());
  ASSERT_TRUE(model_.Refresh());
  EXPECT_TRUE(AllZero());
  // Grown after the zeroing: the discard evicts everything.
  ASSERT_TRUE(Pattern(0x5A));
  ASSERT_TRUE(model_.live.ZeroForReuse(node_, model_.fence_closure(), 0).value_or(false));
  const std::array<en::LiveState::Range, 1> third = {
      en::LiveState::Range{.region = 0, .offset = 6 * kExtent, .bytes = 1}};
  ASSERT_TRUE(model_.live.Use(node_, third));
  ASSERT_TRUE(model_.live.DiscardGrowingState(node_, true));
  EXPECT_TRUE(model_.live.kept_extents().empty());
  EXPECT_TRUE(Released(model_.live.reserved_extents()));
  // A nonresident extent: no zeroing, the discard evicts as before.
  ASSERT_TRUE(model_.live.Use(node_, both));
  ASSERT_TRUE(model_.Refresh());
  ASSERT_TRUE(node_.Evict({model_.live.extents().front()}));
  auto declined = model_.live.ZeroForReuse(node_, model_.fence_closure(), 0);
  ASSERT_TRUE(declined);
  EXPECT_FALSE(*declined);
  ASSERT_TRUE(model_.live.DiscardGrowingState(node_, true));
  EXPECT_TRUE(model_.live.kept_extents().empty());
  EXPECT_TRUE(Released(model_.live.reserved_extents()));
  // Zeroed, but the discard does not ask to keep: released.
  ASSERT_TRUE(model_.live.Use(node_, both));
  ASSERT_TRUE(model_.Refresh());
  ASSERT_TRUE(model_.live.ZeroForReuse(node_, model_.fence_closure(), 0).value_or(false));
  ASSERT_TRUE(model_.live.DiscardGrowingState(node_));
  EXPECT_TRUE(model_.live.kept_extents().empty());
  EXPECT_TRUE(Released(model_.live.reserved_extents()));
  // Kept, half taken back, then discarded plainly: all released.
  ASSERT_TRUE(model_.live.Use(node_, both));
  ASSERT_TRUE(model_.Refresh());
  ASSERT_TRUE(model_.live.ZeroForReuse(node_, model_.fence_closure(), 0).value_or(false));
  ASSERT_TRUE(model_.live.DiscardGrowingState(node_, true));
  ASSERT_TRUE(model_.live.Use(node_, std::span(both).first(1)));
  EXPECT_EQ(model_.live.kept_extents().size(), 1);
  ASSERT_TRUE(model_.live.DiscardGrowingState(node_));
  EXPECT_TRUE(model_.live.kept_extents().empty());
  EXPECT_TRUE(Released(model_.live.reserved_extents()));
}

TEST_F(LiveStateTest, TrimDiscardsSavedTailAndRegrowthCannotRestoreItsOldContents) {
  const std::array<en::LiveState::Range, 2> both = {
      en::LiveState::Range{.region = 0, .offset = 0, .bytes = 1},
      en::LiveState::Range{.region = 0, .offset = 4 * kExtent, .bytes = 1}};
  ASSERT_TRUE(model_.live.Use(node_, both));
  ASSERT_TRUE(model_.Refresh());
  ASSERT_TRUE(node_.Job(
      model_.fence_closure(),
      [&](pr::NativeStream native) {
        return pr::FillAsync(native, en::support::Pointer(model_.live.base(0)), 0x5A, kExtent)
                           .ok() &&
                       pr::FillAsync(native,
                                     en::support::Pointer(model_.live.base(0) + (4 * kExtent)),
                                     0x7B, kExtent)
                           .ok()
                   ? jitllm::scheduler::JobResult::kQueued
                   : jitllm::scheduler::JobResult::kUnknown;
      },
      "trim patterns", 0));
  ASSERT_TRUE(node_.Evict(model_.live.extents()));
  ASSERT_TRUE(model_.live.Retain(node_, std::span(both).first(1)));
  EXPECT_EQ(model_.live.extents().size(), 1);
  ASSERT_TRUE(model_.live.Use(node_, both));
  ASSERT_TRUE(model_.Refresh());
  std::vector<std::byte> copy;
  std::array<std::vector<std::byte>*, 1> output = {&copy};
  ASSERT_TRUE(model_.live.Read(node_, model_.fence_closure(), 0, "trimmed state", output));
  EXPECT_EQ(copy.front(), std::byte{0x5A});
  EXPECT_TRUE(std::ranges::all_of(std::span(copy).subspan(4 * kExtent, kExtent),
                                  [](std::byte b) { return b == std::byte{0}; }));
}

TEST_F(LiveStateTest, PackedSnapshotCopiesOnlyNamedPagesAndDynamicPinnedStagingRetires) {
  const std::array<en::LiveState::Range, 1> tail = {
      en::LiveState::Range{.region = 0, .offset = 7 * kExtent, .bytes = 16}};
  ASSERT_TRUE(model_.live.Use(node_, tail));
  ASSERT_TRUE(model_.Refresh());
  std::vector<jitllm::catalog::ExtentId> staging;
  auto host = node_.Pinned(256, 0, staging);
  ASSERT_TRUE(host);
  EXPECT_EQ(Occupancy(), fixed_ + kExtent + 256);
  std::ranges::fill(std::span(static_cast<std::byte*>(*host), 256), std::byte{0x6C});
  ASSERT_TRUE(model_.live.Copy(node_, model_.fence_closure(), 0, *host, tail, false));
  std::ranges::fill(std::span(static_cast<std::byte*>(*host), 256), std::byte{0});
  ASSERT_TRUE(model_.live.Copy(node_, model_.fence_closure(), 0, *host, tail, true));
  EXPECT_TRUE(std::ranges::all_of(std::span(static_cast<std::byte*>(*host), 16),
                                  [](std::byte b) { return b == std::byte{0x6C}; }));
  EXPECT_EQ(static_cast<std::byte*>(*host)[16], std::byte{0});
  jitllm::catalog::ResourceId resource;
  ASSERT_TRUE(node_.Call(
      [&]() -> en::Status {
        const std::array<jitllm::catalog::Range, 1> ranges = {jitllm::catalog::Range{
            .extent = staging.front(), .offset = Bytes(0), .length = Bytes(16)}};
        auto added = node_.catalog().AddResource(ranges);
        if (!added) {
          return std::unexpected("test pinned resource");
        }
        resource = *added;
        return {};
      },
      "test pinned resource"));
  EXPECT_FALSE(node_.FreePinned(*host));
  EXPECT_EQ(Occupancy(), fixed_ + kExtent + 256);
  ASSERT_TRUE(node_.Call(
      [&]() -> en::Status {
        return node_.catalog().RemoveResource(resource) ? en::Status{}
                                                        : std::unexpected("test pinned removal");
      },
      "removing test pinned resource"));
  ASSERT_TRUE(node_.FreePinned(*host));
  EXPECT_EQ(Occupancy(), fixed_ + kExtent);
  EXPECT_FALSE(node_.FreePinned(*host));
}

class LowCapacityStateTest : public LiveStateTest {
 protected:
  std::uint64_t BudgetExtents() const override { return 1; }
};

TEST_F(LowCapacityStateTest, CleanCapacityRefusalPreservesTheExistingPrefix) {
  const std::array<en::LiveState::Range, 1> first = {
      en::LiveState::Range{.region = 0, .offset = 0, .bytes = 16}};
  ASSERT_TRUE(model_.live.Use(node_, first));
  ASSERT_TRUE(model_.Refresh());
  const std::array<en::LiveState::Range, 1> next = {
      en::LiveState::Range{.region = 0, .offset = kExtent, .bytes = 16}};
  EXPECT_FALSE(model_.live.Use(node_, next));
  EXPECT_TRUE(model_.live.Usable());
  EXPECT_EQ(model_.live.extents().size(), 1);
  EXPECT_EQ(Occupancy(), fixed_ + kExtent);
  std::vector<jitllm::catalog::ExtentId> staging;
  EXPECT_FALSE(node_.Pinned(256, 0, staging));
  EXPECT_TRUE(staging.empty());
}

// The refusal a cohort may wait out (Llm::StateRefusedFor, the runners'
// Slot::state_refused) is typed: only the budget's refusal of a growth,
// the state usable as it was; never an invalid range's.
TEST_F(LowCapacityStateTest, OnlyTheBudgetsRefusalIsTypedAsCapacity) {
  const std::array<en::LiveState::Range, 1> first = {
      en::LiveState::Range{.region = 0, .offset = 0, .bytes = 16}};
  bool over_budget = true;
  ASSERT_TRUE(model_.live.Use(node_, first, nullptr, &over_budget));
  EXPECT_FALSE(over_budget);
  ASSERT_TRUE(model_.Refresh());
  const std::array<en::LiveState::Range, 1> next = {
      en::LiveState::Range{.region = 0, .offset = kExtent, .bytes = 16}};
  EXPECT_FALSE(model_.live.Use(node_, next, nullptr, &over_budget));
  EXPECT_TRUE(over_budget);
  EXPECT_TRUE(model_.live.Usable());
  const std::array<en::LiveState::Range, 1> outside = {
      en::LiveState::Range{.region = 0, .offset = 8 * kExtent, .bytes = 16}};
  EXPECT_FALSE(model_.live.Use(node_, outside, nullptr, &over_budget));
  EXPECT_FALSE(over_budget);
  // Once the held state is released, the same growth fits.
  ASSERT_TRUE(model_.live.DiscardGrowingState(node_));
  ASSERT_TRUE(model_.Refresh());
  EXPECT_TRUE(model_.live.Use(node_, next, nullptr, &over_budget));
  EXPECT_FALSE(over_budget);
}

TEST_F(LiveStateTest, DiskCheckpointRestoresTheWholeMutablePageAndDropsNewTailPages) {
  const std::array<en::LiveState::Range, 1> prefix = {
      en::LiveState::Range{.region = 0, .offset = 0, .bytes = 3 * kExtent}};
  ASSERT_TRUE(model_.live.Use(node_, prefix));
  ASSERT_TRUE(model_.Refresh());
  ASSERT_TRUE(node_.Job(
      model_.fence_closure(),
      [&](pr::NativeStream stream) {
        return pr::FillAsync(stream, en::support::Pointer(model_.live.base(0)), 0x21, 3 * kExtent)
                       .ok()
                   ? jitllm::scheduler::JobResult::kQueued
                   : jitllm::scheduler::JobResult::kUnknown;
      },
      "checkpoint prefix", 0));
  const auto footprint = model_.live.used_ranges();
  const std::array<en::LiveState::Range, 1> writes = {en::LiveState::Range{
      .region = 0, .offset = (2 * kExtent) + 256, .bytes = (6 * kExtent) - 256}};
  auto pages = en::CheckpointPages(footprint, writes);
  ASSERT_TRUE(pages);
  ASSERT_EQ(pages->size(), 1);
  EXPECT_EQ(pages->front().bytes, kExtent);
  // NOLINTNEXTLINE(concurrency-mt-unsafe): test environment is immutable
  const auto directory = std::filesystem::path(std::getenv("JITLLM_TEST_SCRATCH"));
  auto checkpoint = en::CheckpointFile::Capture(
      node_, directory, *pages, [&](void* host, std::span<const en::LiveState::Range> ranges) {
        return model_.live.Copy(node_, model_.fence_closure(), 0, host, ranges, true);
      });
  ASSERT_TRUE(checkpoint);
  EXPECT_EQ(Occupancy(), fixed_ + (3 * kExtent));  // staging is gone between turns
  const std::array<en::LiveState::Range, 1> grown = {
      en::LiveState::Range{.region = 0, .offset = 0, .bytes = 5 * kExtent}};
  ASSERT_TRUE(model_.live.Use(node_, grown));
  ASSERT_TRUE(model_.Refresh());
  ASSERT_TRUE(node_.Job(
      model_.fence_closure(),
      [&](pr::NativeStream stream) {
        return pr::FillAsync(stream, en::support::Pointer(model_.live.base(0) + (2 * kExtent)),
                             0x62, 3 * kExtent)
                       .ok()
                   ? jitllm::scheduler::JobResult::kQueued
                   : jitllm::scheduler::JobResult::kUnknown;
      },
      "a newer branch", 0));
  ASSERT_TRUE(node_.Evict(model_.live.extents()));
  auto restored = checkpoint->Restore(
      node_,
      [&]() -> en::Status {
        if (auto retained = model_.live.Retain(node_, footprint); !retained) {
          return retained;
        }
        return model_.Refresh();
      },
      [&](void* host, std::span<const en::LiveState::Range> ranges) {
        return model_.live.Copy(node_, model_.fence_closure(), 0, host, ranges, false);
      });
  ASSERT_TRUE(restored);
  EXPECT_EQ(model_.live.extents().size(), 3);
  std::vector<std::byte> copy;
  const std::array<std::vector<std::byte>*, 1> output = {&copy};
  ASSERT_TRUE(model_.live.Read(node_, model_.fence_closure(), 0, "restored checkpoint", output));
  EXPECT_TRUE(std::ranges::all_of(std::span(copy).first(3 * kExtent),
                                  [](std::byte b) { return b == std::byte{0x21}; }));
  EXPECT_TRUE(std::ranges::all_of(std::span(copy).subspan(3 * kExtent),
                                  [](std::byte b) { return b == std::byte{0}; }));
}

TEST_F(LiveStateTest, DiskCheckpointPadsFileTransfersButRestoresOnlyLogicalBytes) {
  const std::array<en::LiveState::Range, 2> ranges = {
      en::LiveState::Range{.region = 0, .offset = 0, .bytes = 123},
      en::LiveState::Range{.region = 0, .offset = kExtent, .bytes = 16}};
  // NOLINTNEXTLINE(concurrency-mt-unsafe): test environment is immutable
  const auto directory = std::filesystem::path(std::getenv("JITLLM_TEST_SCRATCH"));
  auto checkpoint = en::CheckpointFile::Capture(
      node_, directory, ranges,
      [](void* host, std::span<const en::LiveState::Range> page) -> en::Status {
        std::memset(host, page.front().offset == 0 ? 0x35 : 0x72, page.front().bytes);
        return {};
      });
  ASSERT_TRUE(checkpoint);
  EXPECT_EQ(checkpoint->bytes(), 139);
  std::size_t copied = 0;
  auto restored = checkpoint->Restore(
      node_, [] { return en::Status{}; },
      [&](void* host, std::span<const en::LiveState::Range> page) -> en::Status {
        const auto want = page.front().offset == 0 ? std::byte{0x35} : std::byte{0x72};
        EXPECT_TRUE(
            std::ranges::all_of(std::span(static_cast<const std::byte*>(host), page.front().bytes),
                                [&](std::byte byte) { return byte == want; }));
        copied += page.front().bytes;
        return {};
      });
  ASSERT_TRUE(restored);
  EXPECT_EQ(copied, 139);
  EXPECT_EQ(Occupancy(), fixed_);
}

TEST_F(LowCapacityStateTest, OptionalCheckpointAllocationRefusesBeforeStateMutation) {
  const std::array<en::LiveState::Range, 1> range = {
      en::LiveState::Range{.region = 0, .offset = 0, .bytes = 16}};
  ASSERT_TRUE(model_.live.Use(node_, range));
  // NOLINTNEXTLINE(concurrency-mt-unsafe): test environment is immutable
  const auto directory = std::filesystem::path(std::getenv("JITLLM_TEST_SCRATCH"));
  bool copied = false;
  auto checkpoint = en::CheckpointFile::Capture(
      node_, directory, range, [&](void*, std::span<const en::LiveState::Range>) -> en::Status {
        copied = true;
        return {};
      });
  ASSERT_FALSE(checkpoint);
  EXPECT_FALSE(checkpoint.error().invalid_state);
  EXPECT_FALSE(copied);
  EXPECT_TRUE(model_.live.Usable());
  EXPECT_EQ(Occupancy(), fixed_ + kExtent);
}

TEST_F(LiveStateTest, FailedCheckpointDeviceCopyRetainsItsOriginalStagingAllocation) {
  const std::array<en::LiveState::Range, 1> range = {
      en::LiveState::Range{.region = 0, .offset = 0, .bytes = 16}};
  // NOLINTNEXTLINE(concurrency-mt-unsafe): test environment is immutable
  const auto directory = std::filesystem::path(std::getenv("JITLLM_TEST_SCRATCH"));
  auto checkpoint = en::CheckpointFile::Capture(
      node_, directory, range, [](void*, std::span<const en::LiveState::Range>) -> en::Status {
        return std::unexpected("copy completion is unknown");
      });
  ASSERT_FALSE(checkpoint);
  EXPECT_TRUE(checkpoint.error().invalid_state);
  EXPECT_EQ(node_.kept_pinned(), 1);
  EXPECT_EQ(Occupancy(), fixed_ + kExtent + jitllm::platform::kDirectIoAlignment);
}

TEST_F(LiveStateTest, CancelledCheckpointCaptureDropsItsCandidateAfterTheCurrentPage) {
  const std::array<en::LiveState::Range, 2> ranges = {
      en::LiveState::Range{.region = 0, .offset = 0, .bytes = 16},
      en::LiveState::Range{.region = 0, .offset = kExtent, .bytes = 16}};
  // NOLINTNEXTLINE(concurrency-mt-unsafe): test environment is immutable
  const auto directory = std::filesystem::path(std::getenv("JITLLM_TEST_SCRATCH"));
  std::size_t copied = 0;
  auto checkpoint = en::CheckpointFile::Capture(
      node_, directory, ranges,
      [&](void* host, std::span<const en::LiveState::Range>) -> en::Status {
        std::memset(host, 0x45, 16);
        ++copied;
        return {};
      },
      [&]() { return copied == 0; });
  ASSERT_FALSE(checkpoint);
  EXPECT_TRUE(checkpoint.error().cancelled);
  EXPECT_FALSE(checkpoint.error().invalid_state);
  EXPECT_EQ(copied, 1);
  EXPECT_EQ(Occupancy(), fixed_);
  EXPECT_EQ(node_.kept_pinned(), 0);
}

TEST_F(LiveStateTest, RestoreCancellationReportsWhetherTheBranchNeedsClearing) {
  const std::array<en::LiveState::Range, 2> ranges = {
      en::LiveState::Range{.region = 0, .offset = 0, .bytes = 16},
      en::LiveState::Range{.region = 0, .offset = kExtent, .bytes = 16}};
  // NOLINTNEXTLINE(concurrency-mt-unsafe): test environment is immutable
  const auto directory = std::filesystem::path(std::getenv("JITLLM_TEST_SCRATCH"));
  auto checkpoint = en::CheckpointFile::Capture(
      node_, directory, ranges,
      [](void* host, std::span<const en::LiveState::Range>) -> en::Status {
        std::memset(host, 0x45, 16);
        return {};
      });
  ASSERT_TRUE(checkpoint);
  bool prepared = false;
  std::size_t copied = 0;
  const auto prepare = [&]() -> en::Status {
    prepared = true;
    return {};
  };
  const auto copy = [&](void* host, std::span<const en::LiveState::Range>) -> en::Status {
    EXPECT_EQ(static_cast<const std::byte*>(host)[0], std::byte{0x45});
    ++copied;
    return {};
  };
  auto early = checkpoint->Restore(node_, prepare, copy, [] { return false; });
  ASSERT_FALSE(early);
  EXPECT_TRUE(early.error().cancelled);
  EXPECT_FALSE(early.error().invalid_state);
  EXPECT_FALSE(prepared);
  EXPECT_EQ(copied, 0);
  auto partial = checkpoint->Restore(node_, prepare, copy, [&]() { return copied == 0; });
  ASSERT_FALSE(partial);
  EXPECT_TRUE(partial.error().cancelled);
  EXPECT_TRUE(partial.error().invalid_state);
  EXPECT_TRUE(prepared);
  EXPECT_EQ(copied, 1);
  EXPECT_EQ(Occupancy(), fixed_);
  EXPECT_EQ(node_.kept_pinned(), 0);
  copied = 0;
  ASSERT_TRUE(checkpoint->Restore(node_, prepare, copy));
  EXPECT_EQ(copied, 2);  // the immutable checkpoint remains usable
}

class CheckpointCapacityStateTest : public LiveStateTest {
 protected:
  std::uint64_t BudgetExtents() const override { return 2; }
};

TEST_F(CheckpointCapacityStateTest, RestoreStagingRefusalLeavesTheBranchUntouched) {
  const std::array<en::LiveState::Range, 1> range = {
      en::LiveState::Range{.region = 0, .offset = 0, .bytes = 16}};
  // NOLINTNEXTLINE(concurrency-mt-unsafe): test environment is immutable
  const auto directory = std::filesystem::path(std::getenv("JITLLM_TEST_SCRATCH"));
  auto checkpoint = en::CheckpointFile::Capture(
      node_, directory, range, [](void* host, std::span<const en::LiveState::Range>) -> en::Status {
        std::memset(host, 0, 16);
        return {};
      });
  ASSERT_TRUE(checkpoint);
  ASSERT_TRUE(model_.live.Use(node_, range));
  bool prepared = false;
  auto restored = checkpoint->Restore(
      node_,
      [&]() -> en::Status {
        prepared = true;
        return {};
      },
      [](void*, std::span<const en::LiveState::Range>) { return en::Status{}; });
  ASSERT_FALSE(restored);
  EXPECT_FALSE(restored.error().invalid_state);
  EXPECT_FALSE(prepared);
  EXPECT_TRUE(model_.live.Usable());
  EXPECT_EQ(Occupancy(), fixed_ + kExtent);
}

// A budget full of what the reclaim order can give back (plans and graphs
// charged past their floor, PagedNode::ChargeHost) never refuses a turn
// checkpoint: its staging has the node's reclaimer free room first.
TEST_F(CheckpointCapacityStateTest, ABudgetFullOfPlansAndGraphsStillCheckpoints) {
  node_.SetHostFloor(0);
  ASSERT_TRUE(node_.ChargeHost(2 * kExtent, true));
  EXPECT_EQ(Occupancy(), fixed_ + (2 * kExtent));
  std::uint64_t asked = 0;
  node_.SetReclaimer([&](std::uint64_t needed, en::PagedNode::ReclaimFor what,
                         std::span<const jitllm::catalog::ExtentId>) {
    EXPECT_EQ(what, en::PagedNode::ReclaimFor::kStaging);  // a state need: whole
    asked += needed;
    const std::uint64_t freed =
        std::min(node_.host_counted(), (needed + kExtent - 1) / kExtent * kExtent);
    node_.UnchargeHost(freed);
    return freed;
  });
  const std::array<en::LiveState::Range, 1> range = {
      en::LiveState::Range{.region = 0, .offset = 0, .bytes = 16}};
  // NOLINTNEXTLINE(concurrency-mt-unsafe): test environment is immutable
  const auto directory = std::filesystem::path(std::getenv("JITLLM_TEST_SCRATCH"));
  auto checkpoint = en::CheckpointFile::Capture(
      node_, directory, range, [](void* host, std::span<const en::LiveState::Range>) -> en::Status {
        std::memset(host, 0x5a, 16);
        return {};
      });
  ASSERT_TRUE(checkpoint);
  EXPECT_GT(asked, 0U);
  node_.SetReclaimer({});
}

// The runtime sets the checkpoints' staging apart at start
// (PagedNode::ReserveStaging): with the budget past full and nothing left
// to reclaim, a capture and a restore still run on it, one at a time, and
// it stays reserved between them.
TEST_F(CheckpointCapacityStateTest, ReservedStagingServesCheckpointsWithNothingToReclaim) {
  ASSERT_TRUE(node_.ReserveStaging(en::CheckpointStagingBytes()));
  node_.SetHostFloor(0);
  ASSERT_TRUE(node_.ChargeHost(2 * kExtent, true));  // required: past the budget
  const std::uint64_t full = Occupancy();
  EXPECT_GT(full, fixed_ + (2 * kExtent));
  std::vector<jitllm::catalog::ExtentId> staging;
  EXPECT_FALSE(node_.Pinned(256, 0, staging));  // nothing else fits
  const std::array<en::LiveState::Range, 1> range = {
      en::LiveState::Range{.region = 0, .offset = 0, .bytes = 16}};
  // NOLINTNEXTLINE(concurrency-mt-unsafe): test environment is immutable
  const auto directory = std::filesystem::path(std::getenv("JITLLM_TEST_SCRATCH"));
  auto checkpoint = en::CheckpointFile::Capture(
      node_, directory, range, [](void* host, std::span<const en::LiveState::Range>) -> en::Status {
        std::memset(host, 0x5a, 16);
        return {};
      });
  ASSERT_TRUE(checkpoint);
  EXPECT_EQ(Occupancy(), full);
  std::size_t copied = 0;
  auto restored = checkpoint->Restore(
      node_, [] { return en::Status{}; },
      [&](void* host, std::span<const en::LiveState::Range> page) -> en::Status {
        EXPECT_EQ(static_cast<const std::byte*>(host)[0], std::byte{0x5a});
        copied += page.front().bytes;
        return {};
      });
  ASSERT_TRUE(restored);
  EXPECT_EQ(copied, 16U);
  EXPECT_EQ(Occupancy(), full);
  node_.UnchargeHost(2 * kExtent);
}

TEST(LiveStateBufferTest, UnprovenCopyRefusesReuseAndKeepsItsDestinationUntilExit) {
  en::PagedNode node(en::NodeSettings{});
  ASSERT_TRUE(node.Open());
  const auto fixed = node.catalog().OccupancyOf(node.domain()).Total().value();
  ASSERT_TRUE(node.Start(Bytes(fixed + kExtent)));
  node.Run();
  en::LiveState state("test");
  ASSERT_NE(state.HostCopy(node, 256), nullptr);
  state.HostCopyUnproven();
  EXPECT_EQ(state.HostCopy(node, 256), nullptr);
  EXPECT_EQ(state.HostCopy(node, 512), nullptr);
  EXPECT_EQ(node.kept_pinned(), 1);
  ASSERT_TRUE(node.TearDown({}));
  // Deliberately retained, as a real unknown-completion destination is.
}

}  // namespace

class TokenCapacityStateTest : public LiveStateTest {
  std::uint64_t BudgetExtents() const override { return 2; }
};

TEST_F(TokenCapacityStateTest, NativeHistoryReclaimReleasesRoundedCatalogOccupancy) {
  namespace rt = jitllm::runtime;
  rt::RequestMemory memory(0, 8 * kExtent, true);
  rt::MemoryCharge idle_charge;
  std::vector<std::int32_t> idle;
  unsigned reclaims = 0;
  memory.SetDriver(std::this_thread::get_id(), [&](std::uint64_t incoming) {
    std::uint64_t shortfall = 0;
    if (node_.SetTokenCharge(memory.used() + incoming, &shortfall)) {
      return true;
    }
    ++reclaims;
    rt::ReleaseTokenStorage(idle, idle_charge);
    memory.Settle();
    return node_.SetTokenCharge(memory.used() + incoming, &shortfall);
  });
  ASSERT_TRUE(
      rt::ReserveTokenStorage(idle, idle_charge, memory, 3 * kExtent / (2 * sizeof(std::int32_t))));
  EXPECT_EQ(memory.used(), 3 * kExtent / 2);
  EXPECT_EQ(node_.token_charged(), 2 * kExtent);
  EXPECT_EQ(Occupancy(), fixed_ + 2 * kExtent);
  rt::MemoryCharge active_charge;
  std::vector<std::int32_t> active;
  ASSERT_TRUE(
      rt::ReserveTokenStorage(active, active_charge, memory, kExtent / sizeof(std::int32_t)));
  EXPECT_EQ(reclaims, 1U);
  EXPECT_TRUE(idle.empty());
  EXPECT_EQ(memory.used(), kExtent);
  EXPECT_EQ(memory.grant(), kExtent);
  EXPECT_EQ(node_.token_charged(), kExtent);
  EXPECT_EQ(Occupancy(), fixed_ + kExtent);
  rt::ReleaseTokenStorage(active, active_charge);
  memory.Settle();
  EXPECT_EQ(node_.token_charged(), 0U);
  EXPECT_EQ(Occupancy(), fixed_);
}

class TokenBoundaryStateTest : public LiveStateTest {
  std::uint64_t BudgetExtents() const override { return 1; }
};

TEST_F(TokenBoundaryStateTest, TinyIdleHistoryAvoidsAnUnfundedProspectiveExtent) {
  namespace rt = jitllm::runtime;
  rt::RequestMemory memory(0, 4 * kExtent, true);
  rt::MemoryCharge idle_charge;
  std::vector<std::int32_t> idle;
  rt::MemoryCharge protected_charge;
  std::vector<std::int32_t> protected_tokens;
  unsigned reclaims = 0;
  memory.SetDriver(std::this_thread::get_id(), [&](std::uint64_t incoming) {
    std::uint64_t shortfall = 0;
    if (node_.SetTokenCharge(memory.used() + incoming, &shortfall)) {
      return true;
    }
    const std::array<std::uint64_t, 1> capacities{idle_charge.bytes()};
    const auto groups =
        rt::GroupTokenReclaim(capacities, memory.used() + incoming, kExtent, shortfall);
    if (groups.empty()) {
      return false;
    }
    ++reclaims;
    rt::ReleaseTokenStorage(idle, idle_charge);
    memory.Settle();
    return node_.SetTokenCharge(memory.used() + incoming, &shortfall);
  });
  ASSERT_TRUE(rt::ReserveTokenStorage(idle, idle_charge, memory, 8));
  ASSERT_TRUE(rt::ReserveTokenStorage(protected_tokens, protected_charge, memory,
                                      (kExtent - 32) / sizeof(std::int32_t)));
  EXPECT_EQ(memory.used(), kExtent);
  EXPECT_EQ(node_.token_charged(), kExtent);
  rt::MemoryCharge incoming_charge;
  std::vector<std::int32_t> incoming;
  ASSERT_TRUE(rt::ReserveTokenStorage(incoming, incoming_charge, memory, 8));
  EXPECT_EQ(reclaims, 1U);
  EXPECT_TRUE(idle.empty());
  EXPECT_EQ(memory.used(), kExtent);
  EXPECT_EQ(node_.token_charged(), kExtent);
  EXPECT_EQ(Occupancy(), fixed_ + kExtent);
  rt::ReleaseTokenStorage(incoming, incoming_charge);
  rt::ReleaseTokenStorage(protected_tokens, protected_charge);
  memory.Settle();
  EXPECT_EQ(Occupancy(), fixed_);
}
