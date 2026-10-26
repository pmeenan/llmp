// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// ReleaseMapped (engine/paged_node.h) over fake device memory: a region
// whose creation failed part way, or before its reservation, releases what
// it holds and stays releasable. CPU only; no CUDA context is opened.
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <format>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "artifact_writer.h"
#include "base/bytes.h"
#include "engine/live_state.h"
#include "engine/paged_node.h"
#include "engine/paged_weights.h"
#include "expected_error.h"
#include "providers/fake/fake_device_memory.h"

namespace llmp::scheduler {
struct SchedulerPlacementTestAccess {
  static std::uint64_t Token(std::atomic<std::uint64_t>& next) {
    return Scheduler::TakePlacementInstance(next);
  }
  static void Epoch(Scheduler& scheduler, std::uint64_t value) {
    scheduler.placement_stamp_.epoch = value;
  }
  static void Instance(Scheduler& scheduler, std::uint64_t value) {
    scheduler.placement_stamp_.instance = value;
  }
};
}  // namespace llmp::scheduler

namespace llmp::engine {
struct LiveStatePlacementTestAccess {
  static void Expected(LiveState& state, catalog::ExtentId extent,
                       const scheduler::PageSource& source) {
    state.regions_.resize(1);
    auto& region = state.regions_.front();
    region.mapped.name = "state test";
    region.mapped.extents = {extent};
    region.sources = {source};
  }
  static bool Check(LiveState& state, const scheduler::Scheduler& scheduler, PlaceCheck& check) {
    return state.CheckPlacesImpl(scheduler, check);
  }
  static void MissingSource(LiveState& state) { state.regions_.front().sources.clear(); }
  static void Reservation(LiveState& state, providers::ReservationId reservation) {
    state.regions_.front().mapped.reservation = reservation;
  }
};

struct PagedWeightsTestAccess {
  // Exact pre-optimization scan, retained only as the semantic oracle.
  static std::expected<std::size_t, std::string> Scan(const PagedWeights& weights,
                                                      std::uint64_t address, std::uint64_t bytes,
                                                      std::span<catalog::ExtentId> destination) {
    std::uint64_t end = 0;
    if (address == 0 || bytes == 0 || __builtin_add_overflow(address, bytes, &end) ||
        weights.sources_.size() != weights.extents_.size())
      return std::unexpected("a weight interval is empty or overflows");
    // Pages are disjoint by Reserve. Count their exact intersection; no hole in
    // an unplaced group or a host reservation may be treated as a device page.
    std::uint64_t covered = 0;
    std::size_t count = 0;
    for (const PagedWeights::Source& source : weights.sources_) {
      if (source.host) continue;
      std::uint64_t page_end = 0;
      if (__builtin_add_overflow(source.address, kPagedExtent, &page_end))
        return std::unexpected("a reserved weight page overflows");
      const auto lo = std::max(address, source.address);
      const auto hi = std::min(end, page_end);
      if (lo >= hi) continue;
      if (__builtin_add_overflow(covered, hi - lo, &covered))
        return std::unexpected("a weight interval coverage overflows");
      ++count;
    }
    if (covered != bytes || count > destination.size())
      return std::unexpected(
          "a weight interval is not completely placed or its output is too small");
    std::size_t next = 0;
    for (std::size_t i = 0; i < weights.sources_.size(); ++i) {
      const PagedWeights::Source& source = weights.sources_[i];
      if (!source.host && source.address < end && address < source.address + kPagedExtent)
        destination[next++] = weights.extents_[i];
    }
    return next;
  }
  static void FakeNode(PagedNode& node) {
    node.memory_ = std::make_unique<providers::fake::FakeDeviceMemory>(
        base::Bytes(kPagedExtent), base::Bytes(4 * kPagedExtent), providers::fake::Contents::kNone);
    node.domain_ = node.catalog_.AddDomain("ordinal weights");
    for (std::size_t i = 0; i < node.memory_->Classes().size(); ++i) {
      if (node.memory_->Classes()[i].kind == providers::BackingKind::kDevice)
        node.device_class_ = i;
      else
        node.host_class_ = i;
    }
  }
  static providers::ReservationId Reservation(const PagedWeights& weights) {
    return weights.reservation_;
  }
  static std::uint64_t Base(const PagedWeights& weights) {
    return weights.sources_.front().address;
  }
  template <typename F>
  static void Edit(PagedWeights& weights, F&& edit) {
    edit(weights.sources_, weights.extents_, weights.bytes_, weights.host_bytes_);
  }
  // Scaling screen metadata only: no artifact payload or backing is created.
  // Small Open/Reserve controls below independently establish this geometry.
  static void Linear(PagedWeights& weights, providers::ReservationId reservation,
                     std::uint64_t base, std::size_t count, bool host) {
    weights.reservation_ = reservation;
    weights.bytes_ = count * kPagedExtent;
    weights.host_bytes_ = host ? kPagedExtent : 0;
    weights.sources_.reserve(count);
    weights.extents_.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
      const auto offset = i * kPagedExtent;
      weights.sources_.push_back(
          {.address = base + offset,
           .host = host && i % 2 == 0,
           .source = {.read = {},
                      .backing = scheduler::BackingPlace{.reservation = reservation,
                                                         .offset = base::Bytes(offset),
                                                         .size = base::Bytes(kPagedExtent)}}});
      // Deliberately descending opaque IDs: index/address order is not ID order.
      weights.extents_.emplace_back(static_cast<std::uint32_t>(count - i), 1);
    }
  }
  static std::uint64_t MetadataBound(std::size_t count) {
    static_assert(sizeof(PagedWeights::Source) <= 512 && sizeof(catalog::ExtentId) <= 16);
    return count * (sizeof(PagedWeights::Source) + sizeof(catalog::ExtentId));
  }
  static void Expected(PagedWeights& weights, catalog::ExtentId extent,
                       const scheduler::PageSource& source) {
    weights.extents_ = {extent};
    weights.sources_ = {{.source = source}};
  }
  static bool Check(PagedWeights& weights, const scheduler::Scheduler& scheduler,
                    PlaceCheck& check) {
    return weights.CheckPlacesImpl(scheduler, check);
  }
  static void Reservation(PagedWeights& weights, providers::ReservationId reservation) {
    weights.reservation_ = reservation;
  }
};
}  // namespace llmp::engine

namespace {
namespace engine = llmp::engine;
namespace providers = llmp::providers;
namespace fake = providers::fake;
using llmp::base::operator""_MiB;
using llmp::test_support::FailedCode;
namespace scheduler = llmp::scheduler;
using WeightsAccess = engine::PagedWeightsTestAccess;
using SchedulerAccess = scheduler::SchedulerPlacementTestAccess;

// Descriptor-only weights and scheduler: no device context, paging, or model.
struct WeightsFixture {
  llmp::catalog::Catalog catalog;
  llmp::base::WakeFlag wake;
  scheduler::CompletionBoard board{16, wake};
  std::array<std::byte, 64> bytes{};
  std::optional<scheduler::Scheduler> scheduled;
  engine::PagedWeights weights;
  llmp::catalog::ExtentId extent;
  scheduler::PageSource source;

  WeightsFixture() {
    const auto domain = catalog.AddDomain("weights test");
    extent = catalog
                 .AddExtent({.domain = domain,
                             .memory_class = llmp::catalog::MemoryClass::kWeights,
                             .recovery = llmp::catalog::Recovery::kFromArtifact,
                             .size = llmp::base::Bytes(32),
                             .content = {}})
                 .value();
    source.read.memory = bytes.data();
    source.read.length = 32;
    ReconstructScheduler();
    WeightsAccess::Expected(weights, extent, source);
  }
  void ReconstructScheduler() {
    scheduled.emplace(catalog, board, wake, scheduler::Lanes{}, scheduler::SchedulerSettings{});
    EXPECT_TRUE(scheduled->SetSource(extent, source));
    EXPECT_TRUE(scheduled->PinPlaces(std::span(&extent, 1)));
  }
  bool Check(engine::PlaceCheck& check) { return WeightsAccess::Check(weights, *scheduled, check); }
};

TEST(WeightsPlacement, ReusesOnlySuccessfulLocalChecksAndPreservesPriorErrors) {
  WeightsFixture fixture;
  engine::PlaceCheck prior;
  prior.Missing("another resource");
  EXPECT_FALSE(fixture.Check(prior));
  EXPECT_EQ(prior.moved, 1U);
  EXPECT_EQ(prior.first, "another resource");
  EXPECT_TRUE(fixture.Check(prior));  // The actual production skip branch.
  EXPECT_EQ(prior.moved, 1U);
  EXPECT_EQ(prior.first, "another resource");
  fixture.scheduled->UnpinPlaces(std::span(&fixture.extent, 1));
  for (int repeat = 0; repeat < 2; ++repeat) {
    engine::PlaceCheck failed;
    EXPECT_FALSE(fixture.Check(failed));
    EXPECT_EQ(failed.moved, 1U);
    EXPECT_EQ(failed.first, "extent " + std::to_string(fixture.extent.index()));
  }
}

TEST(WeightsPlacement, SuccessfulSamePlaceReplacementInvalidatesButRefusalsDoNot) {
  WeightsFixture fixture;
  engine::PlaceCheck check;
  EXPECT_FALSE(fixture.Check(check));
  EXPECT_TRUE(fixture.Check(check));
  auto reread = fixture.source;
  reread.read.offset = 4096;  // Same place, new source metadata.
  ASSERT_TRUE(fixture.scheduled->SetSource(fixture.extent, reread));
  EXPECT_FALSE(fixture.Check(check));
  EXPECT_EQ(check.moved, 0U);
  EXPECT_TRUE(fixture.Check(check));
  auto invalid = reread;
  invalid.read.length = 0;
  const auto unchanged = fixture.scheduled->placement_stamp();
  EXPECT_FALSE(fixture.scheduled->SetSource(fixture.extent, invalid));
  auto moved = reread;
  moved.read.memory = fixture.bytes.data() + 16;
  EXPECT_FALSE(fixture.scheduled->SetSource(fixture.extent, moved));  // Pinned.
  EXPECT_FALSE(fixture.scheduled->PinPlaces(std::array{llmp::catalog::ExtentId{}}));
  EXPECT_EQ(fixture.scheduled->placement_stamp(), unchanged);
  EXPECT_TRUE(fixture.Check(check));
  fixture.scheduled->UnpinPlaces(std::span(&fixture.extent, 1));
  ASSERT_TRUE(fixture.scheduled->SetSource(fixture.extent, moved));
  ASSERT_TRUE(fixture.scheduled->PinPlaces(std::span(&fixture.extent, 1)));
  for (int repeat = 0; repeat < 2; ++repeat) {
    engine::PlaceCheck mismatch;
    EXPECT_FALSE(fixture.Check(mismatch));
    EXPECT_EQ(mismatch.moved, 1U);
  }
}

TEST(WeightsPlacement, DuplicatePinsAndActualUnpinsInvalidateWithoutDroppingRemainingPin) {
  WeightsFixture fixture;
  engine::PlaceCheck check;
  EXPECT_FALSE(fixture.Check(check));
  ASSERT_TRUE(fixture.scheduled->PinPlaces(std::array{fixture.extent, fixture.extent}));
  EXPECT_FALSE(fixture.Check(check));
  EXPECT_TRUE(fixture.Check(check));
  fixture.scheduled->UnpinPlaces(std::array{fixture.extent, fixture.extent});
  EXPECT_TRUE(fixture.scheduled->PlacePinned(fixture.extent));
  EXPECT_FALSE(fixture.Check(check));
  EXPECT_EQ(check.moved, 0U);
  fixture.scheduled->UnpinPlaces(std::span(&fixture.extent, 1));
  EXPECT_FALSE(fixture.Check(check));
  EXPECT_EQ(check.moved, 1U);
  const auto stamp = fixture.scheduled->placement_stamp();
  fixture.scheduled->UnpinPlaces(std::span(&fixture.extent, 1));
  EXPECT_EQ(fixture.scheduled->placement_stamp(), stamp);
}

TEST(WeightsPlacement, SchedulerAtReusedAddressHasANewLifetime) {
  WeightsFixture fixture;
  engine::PlaceCheck check;
  EXPECT_FALSE(fixture.Check(check));
  EXPECT_TRUE(fixture.Check(check));
  const auto stamp = fixture.scheduled->placement_stamp();
  const auto* address = &*fixture.scheduled;
  fixture.ReconstructScheduler();
  EXPECT_EQ(&*fixture.scheduled, address);
  EXPECT_NE(fixture.scheduled->placement_stamp().instance, stamp.instance);
  EXPECT_EQ(fixture.scheduled->placement_stamp().epoch, stamp.epoch);
  EXPECT_FALSE(fixture.Check(check));
  EXPECT_TRUE(fixture.Check(check));
  EXPECT_EQ(check.moved, 0U);
}

TEST(WeightsPlacement, ExhaustedEpochAndInstanceAlwaysRecheckWithoutWrapping) {
  WeightsFixture fixture;
  engine::PlaceCheck check;
  SchedulerAccess::Epoch(*fixture.scheduled, UINT64_MAX - 1);
  EXPECT_FALSE(fixture.Check(check));
  EXPECT_TRUE(fixture.Check(check));
  ASSERT_TRUE(fixture.scheduled->SetSource(fixture.extent, fixture.source));
  EXPECT_EQ(fixture.scheduled->placement_stamp().epoch, UINT64_MAX);
  ASSERT_TRUE(fixture.scheduled->SetSource(fixture.extent, fixture.source));
  EXPECT_EQ(fixture.scheduled->placement_stamp().epoch, UINT64_MAX);
  EXPECT_FALSE(fixture.Check(check));
  EXPECT_FALSE(fixture.Check(check));
  std::atomic<std::uint64_t> next{UINT64_MAX - 1};
  EXPECT_EQ(SchedulerAccess::Token(next), UINT64_MAX - 1);
  EXPECT_EQ(SchedulerAccess::Token(next), UINT64_MAX);
  EXPECT_EQ(SchedulerAccess::Token(next), UINT64_MAX);
  EXPECT_EQ(next.load(), UINT64_MAX);
  SchedulerAccess::Epoch(*fixture.scheduled, 0);
  for (const auto token : {std::uint64_t{0}, std::uint64_t{UINT64_MAX}}) {
    SchedulerAccess::Instance(*fixture.scheduled, token);
    EXPECT_FALSE(fixture.Check(check));
    EXPECT_FALSE(fixture.Check(check));
  }
  EXPECT_EQ(check.moved, 0U);
}

TEST(WeightsPlacement, FailedOpenReserveReleaseAndReregisterDiscardMemo) {
  WeightsFixture fixture;
  engine::PlaceCheck check;
  EXPECT_FALSE(fixture.Check(check));
  EXPECT_TRUE(fixture.Check(check));
  EXPECT_FALSE(fixture.weights.Open("/nonexistent-weights-placement-test"));
  EXPECT_FALSE(fixture.Check(check));
  engine::PagedNode unopened(engine::NodeSettings{});
  EXPECT_FALSE(fixture.weights.Reserve(unopened, {}, {}));
  EXPECT_FALSE(fixture.Check(check));
  fake::FakeDeviceMemory memory(2_MiB, 4_MiB);
  const auto reservation = memory.Reserve(2_MiB);
  const auto backing = memory.Create(0, 2_MiB);
  ASSERT_TRUE(reservation && backing);
  ASSERT_TRUE(memory.Map(*reservation, 0_MiB, *backing));
  WeightsAccess::Reservation(fixture.weights, *reservation);
  EXPECT_FALSE(fixture.weights.Release(memory));  // Still mapped.
  EXPECT_FALSE(fixture.Check(check));
  ASSERT_TRUE(memory.Unmap(*reservation, 0_MiB, 2_MiB));
  ASSERT_TRUE(memory.Release(*backing));
  ASSERT_TRUE(fixture.weights.Release(memory));
  EXPECT_FALSE(fixture.Check(check));

  // An empty component registers without touching the unopened node. Its
  // registration boundary must still invalidate a prior successful scan.
  engine::PagedWeights empty;
  EXPECT_FALSE(WeightsAccess::Check(empty, *fixture.scheduled, check));
  EXPECT_TRUE(WeightsAccess::Check(empty, *fixture.scheduled, check));
  ASSERT_TRUE(empty.Register(unopened, 0));
  EXPECT_FALSE(WeightsAccess::Check(empty, *fixture.scheduled, check));
}

// The same descriptor-only scheduler fixture exercises mutable-state source/pin
// validation. Contents, residency and request leases are separate checks.
struct StateFixture : WeightsFixture {
  engine::LiveState live{"state test"};
  StateFixture() { engine::LiveStatePlacementTestAccess::Expected(live, extent, source); }
  bool Check(engine::PlaceCheck& check) {
    return engine::LiveStatePlacementTestAccess::Check(live, *scheduled, check);
  }
};

TEST(StatePlacement, ReusesOnlySuccessfulLocalChecksAndPreservesPriorErrors) {
  StateFixture fixture;
  engine::PlaceCheck prior;
  prior.Missing("another resource");
  EXPECT_FALSE(fixture.Check(prior));
  EXPECT_TRUE(fixture.Check(prior));
  EXPECT_EQ(prior.moved, 1U);
  EXPECT_EQ(prior.first, "another resource");
  fixture.scheduled->UnpinPlaces(std::span(&fixture.extent, 1));
  for (int repeat = 0; repeat < 2; ++repeat) {
    engine::PlaceCheck failed;
    EXPECT_FALSE(fixture.Check(failed));
    EXPECT_EQ(failed.moved, 1U);
    EXPECT_EQ(failed.first, "extent " + std::to_string(fixture.extent.index()));
  }
}

TEST(StatePlacement, SourceAndPinChangesRecheckEvenWhenThePlaceRemainsEqual) {
  StateFixture fixture;
  engine::PlaceCheck check;
  EXPECT_FALSE(fixture.Check(check));
  EXPECT_TRUE(fixture.Check(check));
  auto source = fixture.source;
  source.read.offset = 4096;
  ASSERT_TRUE(fixture.scheduled->SetSource(fixture.extent, source));
  EXPECT_FALSE(fixture.Check(check));
  EXPECT_TRUE(fixture.Check(check));
  ASSERT_TRUE(fixture.scheduled->PinPlaces(std::span(&fixture.extent, 1)));
  fixture.scheduled->UnpinPlaces(std::span(&fixture.extent, 1));
  EXPECT_TRUE(fixture.scheduled->PlacePinned(fixture.extent));
  EXPECT_FALSE(fixture.Check(check));
  EXPECT_TRUE(fixture.Check(check));
  fixture.scheduled->UnpinPlaces(std::span(&fixture.extent, 1));
  source.read.memory = fixture.bytes.data() + 16;
  ASSERT_TRUE(fixture.scheduled->SetSource(fixture.extent, source));
  ASSERT_TRUE(fixture.scheduled->PinPlaces(std::span(&fixture.extent, 1)));
  for (int repeat = 0; repeat < 2; ++repeat) {
    engine::PlaceCheck mismatch;
    EXPECT_FALSE(fixture.Check(mismatch));
    EXPECT_EQ(mismatch.moved, 1U);
  }
}

TEST(StatePlacement, MissingRegionSourcesCannotAuthorizeAMemo) {
  StateFixture fixture;
  engine::LiveStatePlacementTestAccess::MissingSource(fixture.live);
  for (int repeat = 0; repeat < 2; ++repeat) {
    engine::PlaceCheck check;
    EXPECT_FALSE(fixture.Check(check));
    EXPECT_EQ(check.moved, 1U);
    EXPECT_EQ(check.first, "state test's write-back places");
  }
}

TEST(StatePlacement, SchedulerLifetimeAndSaturationCannotReuseAnOldCheck) {
  StateFixture fixture;
  engine::PlaceCheck check;
  EXPECT_FALSE(fixture.Check(check));
  EXPECT_TRUE(fixture.Check(check));
  const auto* address = &*fixture.scheduled;
  const auto instance = fixture.scheduled->placement_stamp().instance;
  fixture.ReconstructScheduler();
  EXPECT_EQ(&*fixture.scheduled, address);
  EXPECT_NE(fixture.scheduled->placement_stamp().instance, instance);
  EXPECT_FALSE(fixture.Check(check));
  EXPECT_TRUE(fixture.Check(check));
  SchedulerAccess::Epoch(*fixture.scheduled, UINT64_MAX - 1);
  EXPECT_FALSE(fixture.Check(check));
  EXPECT_TRUE(fixture.Check(check));
  ASSERT_TRUE(fixture.scheduled->SetSource(fixture.extent, fixture.source));
  EXPECT_EQ(fixture.scheduled->placement_stamp().epoch, UINT64_MAX);
  EXPECT_FALSE(fixture.Check(check));
  EXPECT_FALSE(fixture.Check(check));
  SchedulerAccess::Epoch(*fixture.scheduled, 0);
  for (const auto token : {std::uint64_t{0}, std::uint64_t{UINT64_MAX}}) {
    SchedulerAccess::Instance(*fixture.scheduled, token);
    EXPECT_FALSE(fixture.Check(check));
    EXPECT_FALSE(fixture.Check(check));
  }
  EXPECT_EQ(check.moved, 0U);
}

TEST(StatePlacement, RefusedRestoreTrimSpillAndDiscardInvalidateBeforeReturning) {
  StateFixture fixture;
  engine::PagedNode unopened(engine::NodeSettings{});
  const auto seed = [&] {
    engine::PlaceCheck check;
    (void)fixture.Check(check);
    EXPECT_EQ(check.moved, 0U);
    EXPECT_TRUE(fixture.Check(check));
  };
  const auto invalidated = [&] {
    engine::PlaceCheck check;
    EXPECT_FALSE(fixture.Check(check));
    EXPECT_EQ(check.moved, 0U);
  };
  const std::array<engine::LiveState::Range, 1> invalid = {
      engine::LiveState::Range{.region = 1, .bytes = 1}};
  seed();
  EXPECT_FALSE(fixture.live.Copy(unopened, {}, 0, nullptr, invalid, false));
  invalidated();
  seed();
  EXPECT_FALSE(fixture.live.Retain(unopened, invalid));
  invalidated();
  seed();
  EXPECT_FALSE(fixture.live.RegisterSpill(
      unopened,
      engine::LiveState::SpillPlace{.directory = {}, .dir = 2147483647, .name = "missing"}));
  invalidated();
  seed();
  EXPECT_FALSE(fixture.live.DiscardGrowingState(unopened));
  invalidated();
}

TEST(StatePlacement, ReleaseFailureAndSuccessBothDiscardTheMemo) {
  StateFixture fixture;
  engine::PlaceCheck check;
  EXPECT_FALSE(fixture.Check(check));
  EXPECT_TRUE(fixture.Check(check));
  fake::FakeDeviceMemory memory(2_MiB, 4_MiB);
  const auto reservation = memory.Reserve(2_MiB);
  const auto backing = memory.Create(0, 2_MiB);
  ASSERT_TRUE(reservation && backing);
  ASSERT_TRUE(memory.Map(*reservation, 0_MiB, *backing));
  engine::LiveStatePlacementTestAccess::Reservation(fixture.live, *reservation);
  std::vector<std::string> problems;
  fixture.live.Release(memory, problems);
  EXPECT_EQ(problems.size(), 1U);  // Cannot free a mapped reservation.
  EXPECT_FALSE(fixture.Check(check));
  EXPECT_TRUE(fixture.Check(check));
  ASSERT_TRUE(memory.Unmap(*reservation, 0_MiB, 2_MiB));
  ASSERT_TRUE(memory.Release(*backing));
  problems.clear();
  fixture.live.Release(memory, problems);
  EXPECT_TRUE(problems.empty());
  EXPECT_FALSE(fixture.Check(check));
}

TEST(ReleaseMapped, PartialCreateFailureReleasesCompletedBackingAndReservation) {
  fake::FakeDeviceMemory memory(2_MiB, 4_MiB);
  engine::Mapped mapped;
  mapped.name = "partially created region";
  mapped.bytes = (4_MiB).value();
  const auto reservation = memory.Reserve(4_MiB);
  ASSERT_TRUE(reservation.has_value());
  mapped.reservation = *reservation;
  const auto range = memory.RangeOf(mapped.reservation);
  ASSERT_TRUE(range.has_value());
  mapped.base = range->base;
  const auto first = memory.Create(0, 2_MiB);
  ASSERT_TRUE(first.has_value());
  mapped.backings.push_back(*first);
  ASSERT_TRUE(memory.Map(mapped.reservation, 0_MiB, *first).has_value());
  ASSERT_TRUE(memory.SetAccess(mapped.reservation, 0_MiB, 2_MiB, providers::Access::kReadWrite)
                  .has_value());
  memory.FailNext(fake::Operation::kCreate, providers::ProviderError::kOutOfMemory);
  EXPECT_EQ(FailedCode(memory.Create(0, 2_MiB)), providers::ProviderError::kOutOfMemory);
  EXPECT_EQ(memory.in_use(), 2_MiB);
  EXPECT_EQ(memory.backings(), 1U);
  EXPECT_EQ(memory.reservations(), 1U);
  ASSERT_TRUE(engine::ReleaseMapped(memory, mapped));
  EXPECT_FALSE(mapped.reservation.valid());
  EXPECT_TRUE(mapped.backings.empty());
  EXPECT_EQ(memory.in_use(), 0_MiB);
  EXPECT_EQ(memory.backings(), 0U);
  EXPECT_EQ(memory.reservations(), 0U);
  EXPECT_EQ(FailedCode(memory.RangeOf(*reservation)), providers::ProviderError::kInvalid);
  EXPECT_TRUE(engine::ReleaseMapped(memory, mapped));
  EXPECT_EQ(memory.in_use(), 0_MiB);
  EXPECT_EQ(memory.backings(), 0U);
  EXPECT_EQ(memory.reservations(), 0U);
}

TEST(ReleaseMapped, FailedReservationLeavesAnEmptyIdempotentOwner) {
  fake::FakeDeviceMemory memory(2_MiB, 4_MiB);
  engine::Mapped mapped;
  mapped.name = "region whose reservation failed";
  mapped.bytes = (4_MiB).value();
  memory.FailNext(fake::Operation::kReserve, providers::ProviderError::kOutOfMemory);
  EXPECT_EQ(FailedCode(memory.Reserve(4_MiB)), providers::ProviderError::kOutOfMemory);
  EXPECT_FALSE(mapped.reservation.valid());
  EXPECT_TRUE(engine::ReleaseMapped(memory, mapped));
  EXPECT_TRUE(engine::ReleaseMapped(memory, mapped));
  EXPECT_EQ(memory.in_use(), 0_MiB);
  EXPECT_EQ(memory.backings(), 0U);
  EXPECT_EQ(memory.reservations(), 0U);
}

TEST(ReleaseMapped, FailedUnmapRetainsBackingsUntilUnmapSucceeds) {
  fake::FakeDeviceMemory memory(2_MiB, 4_MiB);
  engine::Mapped mapped;
  mapped.name = "region whose unmap fails once";
  mapped.bytes = (2_MiB).value();
  auto reservation = memory.Reserve(2_MiB);
  ASSERT_TRUE(reservation.has_value());
  mapped.reservation = *reservation;
  auto backing = memory.Create(0, 2_MiB);
  ASSERT_TRUE(backing.has_value());
  ASSERT_TRUE(memory.Map(mapped.reservation, 0_MiB, *backing).has_value());
  mapped.backings.push_back(*backing);

  memory.FailNext(fake::Operation::kUnmap, providers::ProviderError::kInvalid);
  EXPECT_FALSE(engine::ReleaseMapped(memory, mapped));
  EXPECT_EQ(mapped.backings.size(), 1U);
  EXPECT_EQ(memory.backings(), 1U);
  EXPECT_EQ(memory.reservations(), 1U);

  // A subsequent ReleaseMapped succeeds and releases both the backing and reservation.
  EXPECT_TRUE(engine::ReleaseMapped(memory, mapped));
  EXPECT_TRUE(mapped.backings.empty());
  EXPECT_EQ(memory.backings(), 0U);
  EXPECT_EQ(memory.reservations(), 0U);
}

// A small real writer/Open/Reserve fixture proves device/slab/host ordering.
// No node Open/Start, scheduler, stream, page-in or device work is performed.
class WeightIntervals : public testing::Test {
 protected:
  static constexpr std::uint64_t extent = engine::kPagedExtent;
  engine::PagedNode node{engine::NodeSettings{}};
  engine::PagedWeights weights;
  std::filesystem::path scratch;
  void SetUp() override {
    WeightsAccess::FakeNode(node);
    const auto* root = std::getenv("LLMP_TEST_SCRATCH");  // NOLINT(concurrency-mt-unsafe)
    scratch = std::filesystem::path(root ? root : testing::TempDir()) / "weight-intervals" /
              testing::UnitTest::GetInstance()->current_test_info()->name();
    // Fixed construction admission: all three dense + four expert payloads are
    // 4096B; bounded JSON/source objects and reader copies fit this 4MiB ceiling.
    static_assert(sizeof(engine::PagedWeights) < 4096);
    llmp::test::V1Plan plan;
    const llmp::format::TensorFormat format{
        .encoding = llmp::format::Encoding::kF32,
        .layout = llmp::format::Layout::kDense,
        .shape = *llmp::format::ShapeOf(std::array<std::uint64_t, 2>{1024, 1})};
    plan.expert_count = 4;
    plan.groups = {{.layer = 0, .resources = {{.name = "before", .format = format}}},
                   {.kind = llmp::artifact::GroupKind::kExpert,
                    .layer = 0,
                    .arrays = {{.name = "experts", .format = format}}},
                   {.layer = 1, .resources = {{.name = "after", .format = format}}},
                   {.layer = 2, .resources = {{.name = "unplaced", .format = format}}}};
    const auto opened = weights.Open(llmp::test::Publish(scratch, llmp::test::Render(plan)));
    ASSERT_TRUE(opened) << llmp::test_support::Failed(opened).value_or("");
  }
  void TearDown() override {
    // No queued work exists. Free all reservations, including those abandoned
    // by the intentionally unsupported retry sequences, via the fake's owner.
    EXPECT_TRUE(weights.Release(node.memory()));
    std::filesystem::remove_all(scratch);
  }
  engine::Status Reserve(bool host = false) {
    std::array<engine::GroupPlace, 7> places{};
    places[0] = engine::GroupPlace::kDevice;
    places[5] = host ? engine::GroupPlace::kHost : engine::GroupPlace::kDevice;
    const std::array slabs{
        engine::SlabSpec{.first_group = 1, .count = 4, .stride = 8192, .alignment = 4096}};
    return weights.Reserve(node, places, slabs);
  }
  void Same(std::uint64_t address, std::uint64_t bytes, std::size_t capacity = 8) {
    const llmp::catalog::ExtentId sentinel{UINT32_MAX, 7};
    std::array<llmp::catalog::ExtentId, 8> want, got;
    want.fill(sentinel);
    got.fill(sentinel);
    const auto expected =
        WeightsAccess::Scan(weights, address, bytes, std::span(want).first(capacity));
    const auto actual = weights.ExtentsTouching(address, bytes, std::span(got).first(capacity));
    ASSERT_EQ(actual.has_value(), expected.has_value());
    if (actual)
      EXPECT_EQ(*actual, *expected);
    else {
      EXPECT_EQ(actual.error(), expected.error());
      EXPECT_TRUE(std::ranges::all_of(got, [&](auto id) { return id == sentinel; }));
    }
    EXPECT_EQ(got, want);
  }
};

TEST_F(WeightIntervals, RealDenseSlabRangesAndPayloadGapsKeepExactPageOrder) {
  ASSERT_TRUE(Reserve());
  const auto base = WeightsAccess::Base(weights);
  ASSERT_EQ(weights.extents().size(), 3U);
  for (const auto& query : std::array<std::pair<std::uint64_t, std::uint64_t>, 7>{
           {{base, 1},
            {base + extent - 1, 2},
            {base, 3 * extent},
            {weights.group_address(1) + 4096, 4096},  // Unwritten slab gap is still a page.
            {weights.group_address(2), 4096},
            {base + 3 * extent - 1, 1},
            {base, 1}}})
    Same(query.first, query.second);
  Same(base, 3 * extent, 2);
  Same(base + 3 * extent, 1);
  Same(0, 1);
  Same(base, 0);
  Same(UINT64_MAX - 3, 8);
  EXPECT_EQ(weights.group_address(6), 0U);
}

TEST_F(WeightIntervals, HostPlacementAndReleaseRetryRetainLiteralScanSemantics) {
  ASSERT_TRUE(Reserve(true));
  const auto base = WeightsAccess::Base(weights);
  Same(base, 2 * extent);
  Same(weights.group_address(5), 1);  // A host page never supplies device coverage.
  ASSERT_TRUE(weights.Release(node.memory()));
  Same(base, 1);           // Existing helper is geometry-only even after release.
  ASSERT_TRUE(Reserve());  // Existing API retains old arrays/counters; use fallback.
  Same(base, 1);
  Same(weights.group_address(0), 1);
}

TEST_F(WeightIntervals, FailedReleaseAndReserveDoNotMintCurrentReservationAuthority) {
  ASSERT_TRUE(Reserve());
  const auto base = WeightsAccess::Base(weights);
  const auto reservation = WeightsAccess::Reservation(weights);
  auto backing = node.memory().Create(node.device_class(), llmp::base::Bytes(extent));
  ASSERT_TRUE(backing);
  ASSERT_TRUE(node.memory().Map(reservation, llmp::base::Bytes(0), *backing));
  EXPECT_FALSE(weights.Release(node.memory()));
  Same(base, extent);
  ASSERT_TRUE(node.memory().Unmap(reservation, llmp::base::Bytes(0), llmp::base::Bytes(extent)));
  ASSERT_TRUE(node.memory().Release(*backing));
  ASSERT_TRUE(weights.Release(node.memory()));
  static_cast<fake::FakeDeviceMemory&>(node.memory())
      .FailNext(fake::Operation::kReserve, providers::ProviderError::kOutOfMemory);
  EXPECT_FALSE(Reserve());
  Same(base, extent);
}

TEST_F(WeightIntervals, IneligibleMetadataFallsBackBeforeAnyDestinationWrite) {
  ASSERT_TRUE(Reserve());
  const auto base = WeightsAccess::Base(weights);
  WeightsAccess::Edit(weights, [&](auto& sources, auto& extents, auto& bytes, auto&) {
    const auto saved = sources[1];
    sources[1].source.backing->offset = llmp::base::Bytes(0);
    Same(base + extent, 1);
    sources[1] = saved;
    sources[1].source.backing.reset();
    Same(base + extent, 1);
    sources[1] = saved;
    sources[1].source.backing->reservation = {};
    Same(base + extent, 1);
    sources[1] = saved;
    sources[1].source.backing->size = llmp::base::Bytes(1);
    Same(base + extent, 1);
    sources[1] = saved;
    sources[1].address += 1;
    Same(base + extent, 1);
    sources[1] = saved;
    const auto tail = sources.back();
    const auto tail_id = extents.back();
    sources.pop_back();  // A partial catalog prefix is not the whole reservation.
    extents.pop_back();
    Same(base, 2 * extent);
    sources.push_back(tail);
    Same(base, 1);  // Mismatched source/extent cardinality refuses unchanged.
    extents.push_back(tail_id);
    const auto full = bytes;
    bytes = UINT64_MAX;
    Same(base, 1);
    bytes = full;
  });
}

TEST(WeightIntervalSlab, NonzeroDeltaAndSharedExpertPagesKeepPaddingGeometry) {
  constexpr std::uint64_t extent = engine::kPagedExtent;
  const std::array<std::uint32_t, 4> shards{0, 0, 1, 1};
  const std::array<std::uint64_t, 4> files{4096, 8192, 4096, 8192};
  const auto slab = engine::LayOutSlab(shards, files, 4096, 8192, 4096);
  ASSERT_TRUE(slab);
  ASSERT_NE(slab->delta, 0U);
  ASSERT_EQ(slab->pages.size(), 2U);
  fake::FakeDeviceMemory memory(llmp::base::Bytes(extent), llmp::base::Bytes(extent),
                                fake::Contents::kNone);
  const auto reservation = memory.Reserve(llmp::base::Bytes(2 * extent));
  ASSERT_TRUE(reservation);
  const auto range = memory.RangeOf(*reservation);
  ASSERT_TRUE(range);
  engine::PagedWeights weights;
  WeightsAccess::Linear(weights, *reservation, range->base, 2, false);
  // These are the actual two-shard layout's pieces, at the same addresses
  // Reserve uses; no payload is initialized or read by a geometry lookup.
  WeightsAccess::Edit(weights, [&](auto& sources, auto&, auto&, auto&) {
    for (std::size_t i = 0; i < slab->pages.size(); ++i) {
      const auto& page = slab->pages[i];
      auto& source = sources[i].source;
      source.piece_count = page.pieces;
      for (std::size_t part = 0; part < page.pieces; ++part)
        source.pieces.at(part) = {
            .slot_offset = page.slot_offset.at(part),
            .destination = range->base + i * extent + page.page_offset.at(part),
            .length = llmp::base::Bytes(page.bytes.at(part))};
    }
  });
  std::array<llmp::catalog::ExtentId, 2> expected{}, actual{};
  for (std::uint64_t expert = 0; expert != 4; ++expert) {
    const auto address = range->base + slab->delta + expert * 8192;
    ASSERT_EQ(weights.ExtentsTouching(address, 4096, actual),
              WeightsAccess::Scan(weights, address, 4096, expected));
    EXPECT_EQ(actual, expected);
    ASSERT_EQ(weights.ExtentsTouching(address + 4096, 4096, actual),
              WeightsAccess::Scan(weights, address + 4096, 4096, expected));
    EXPECT_EQ(actual, expected);
  }
  ASSERT_TRUE(weights.Release(memory));
}

// Manual screen target only; not registered with CTest. Parent and candidate
// time the real production helper, never the copied semantic Scan oracle.
#if defined(LLMP_WEIGHT_ORDINAL_SCREEN)
TEST(WeightsOrdinalScreen, FixedQueriesUseTheActualHelper) {
  constexpr std::uint64_t extent = engine::kPagedExtent;
  constexpr std::size_t queries = 256;
  // Paid before constructing any inventory: Source/Extent arrays are the only
  // size-dependent heap objects. Fake VMM reserves addresses without backing.
  constexpr std::uint64_t host_bound = 256ULL << 20U;
  ASSERT_LT(WeightsAccess::MetadataBound(16384) + (4ULL << 20U), host_bound);
  fake::FakeDeviceMemory memory(llmp::base::Bytes(extent), llmp::base::Bytes(extent),
                                fake::Contents::kNone);
  std::uint64_t checksum = 0;
  for (const auto count : {std::size_t{64}, std::size_t{1024}, std::size_t{16384}}) {
    for (const bool host : {false, true}) {
      const auto reservation = memory.Reserve(llmp::base::Bytes(count * extent));
      ASSERT_TRUE(reservation);
      const auto range = memory.RangeOf(*reservation);
      ASSERT_TRUE(range);
      engine::PagedWeights weights;
      WeightsAccess::Linear(weights, *reservation, range->base, count, host);
      for (const auto touched : {std::size_t{1}, std::size_t{8}, std::size_t{64}}) {
        if (host && touched != 1) continue;  // Successful fallback calls allocate no errors.
        std::array<std::uint64_t, queries> addresses{};
        std::array<llmp::catalog::ExtentId, 64> output{}, reference{};
        for (std::size_t i = 0; i < queries; ++i) {
          const auto page = host ? 1 + 2 * (i % (count / 2)) : (i * 31) % (count - touched + 1);
          addresses[i] = range->base + page * extent;
          const auto expected =
              WeightsAccess::Scan(weights, addresses[i], touched * extent, reference);
          const auto actual = weights.ExtentsTouching(addresses[i], touched * extent, output);
          ASSERT_EQ(actual.has_value(), expected.has_value());
          if (actual) {
            ASSERT_EQ(*actual, *expected);
            ASSERT_EQ(output, reference);
          } else {
            ASSERT_EQ(actual.error(), expected.error());
          }
        }
        for (std::size_t block = 0; block < 6; ++block) {
          std::size_t accepted = 0;
          const auto cpu_start = std::clock();
          ASSERT_NE(cpu_start, std::clock_t{-1});
          const auto start = std::chrono::steady_clock::now();
          for (const auto address : addresses) {
            const auto result = weights.ExtentsTouching(address, touched * extent, output);
            if (result) {
              ++accepted;
              for (std::size_t j = 0; j < *result; ++j) checksum += output[j].index();
            }
          }
          const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                   std::chrono::steady_clock::now() - start)
                                   .count();
          const auto cpu_end = std::clock();
          ASSERT_NE(cpu_end, std::clock_t{-1});
          ASSERT_EQ(accepted, queries);
          if (block != 0) {
            const auto key =
                std::format("pages{}_touch{}_host{}_block{}", count, touched, host, block);
            RecordProperty(key + "_wall_ns", std::to_string(elapsed));
            RecordProperty(
                key + "_cpu_seconds",
                std::to_string(static_cast<double>(cpu_end - cpu_start) / CLOCKS_PER_SEC));
            RecordProperty(key + "_accepted", std::to_string(accepted));
          }
        }
      }
      ASSERT_TRUE(weights.Release(memory));
    }
  }
  RecordProperty("output_checksum", std::to_string(checksum));
  RecordProperty("queries_per_cell", static_cast<int>(queries));
}

#endif  // LLMP_WEIGHT_ORDINAL_SCREEN

}  // namespace
