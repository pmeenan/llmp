// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "engine/live_state.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <expected>
#include <filesystem>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "base/bytes.h"
#include "engine/paged_node.h"
#include "engine/support.h"
#include "expected_error.h"
#include "providers/device_runtime.h"
#include "scheduler/commands.h"

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
