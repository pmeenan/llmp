// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// ReleaseMapped (engine/paged_node.h) over fake device memory: a region
// whose creation failed part way, or before its reservation, releases what
// it holds and stays releasable. CPU only; no CUDA context is opened.
#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>

#include "base/bytes.h"
#include "engine/paged_node.h"
#include "engine/paged_weights.h"
#include "expected_error.h"
#include "providers/fake/fake_device_memory.h"

namespace jitllm::scheduler {
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
}  // namespace jitllm::scheduler

namespace jitllm::engine {
struct PagedWeightsTestAccess {
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
}  // namespace jitllm::engine

namespace {
namespace engine = jitllm::engine;
namespace providers = jitllm::providers;
namespace fake = providers::fake;
using jitllm::base::operator""_MiB;
using jitllm::test_support::FailedCode;
namespace scheduler = jitllm::scheduler;
using WeightsAccess = engine::PagedWeightsTestAccess;
using SchedulerAccess = scheduler::SchedulerPlacementTestAccess;

// Descriptor-only weights and scheduler: no device context, paging, or model.
struct WeightsFixture {
  jitllm::catalog::Catalog catalog;
  jitllm::base::WakeFlag wake;
  scheduler::CompletionBoard board{16, wake};
  std::array<std::byte, 64> bytes{};
  std::optional<scheduler::Scheduler> scheduled;
  engine::PagedWeights weights;
  jitllm::catalog::ExtentId extent;
  scheduler::PageSource source;

  WeightsFixture() {
    const auto domain = catalog.AddDomain("weights test");
    extent = catalog
                 .AddExtent({.domain = domain,
                             .memory_class = jitllm::catalog::MemoryClass::kWeights,
                             .recovery = jitllm::catalog::Recovery::kFromArtifact,
                             .size = jitllm::base::Bytes(32),
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
  EXPECT_FALSE(fixture.scheduled->PinPlaces(std::array{jitllm::catalog::ExtentId{}}));
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
}  // namespace
