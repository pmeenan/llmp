// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Physical memory modes (D-124 as amended 2026-10-25): the handle pool held
// within the budget's headroom (invariant 5: what it holds beside the
// occupancy never exceeds the budget), best-effort preallocation, the pool
// given back first by a drain, the budget as a scheduler command clamped to
// outstanding claims, the device-aware policy and the budget's headroom on a
// discrete device.

#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "base/bytes.h"
#include "config/node_config.h"
#include "engine/paged_node.h"
#include "providers/device_runtime.h"
#include "providers/fake/fake_device_memory.h"
#include "providers/fake/fake_device_runtime.h"
#include "runtime/memory_policy.h"
#include "runtime/pressure_trim.h"
#include "scheduler/services.h"

namespace {

namespace en = llmp::engine;
namespace sc = llmp::scheduler;
namespace pr = llmp::providers;
namespace rt = llmp::runtime;
using llmp::base::Bytes;
using llmp::base::operator""_MiB;
using pr::ProviderError;

constexpr std::uint64_t kExtent = en::kPagedExtent;

// One VMM command through a lane driven by the test, and whether it succeeded.
class Lane {
 public:
  Lane(sc::BackingService& service, sc::CompletionBoard& board)
      : service_(service), board_(board) {}
  bool Run(sc::BackingWork::Kind kind, pr::ReservationId place, std::uint64_t at) {
    const auto operation = board_.Open();
    if (service_.Submit(sc::DeviceCommand{.operation = operation,
                                          .work = sc::BackingWork{.kind = kind,
                                                                  .reservation = place,
                                                                  .offset = Bytes(at * kExtent),
                                                                  .size = Bytes(kExtent),
                                                                  .allocation_class = 0,
                                                                  .retain = false,
                                                                  .reuse = false}}) !=
        llmp::base::PushResult::kAccepted) {
      return false;
    }
    (void)service_.Turn();
    const auto seen = board_.Harvest(4);
    (void)board_.Close(operation);
    return seen.size() == 1 && seen[0].terminal.has_value() &&
           seen[0].terminal->outcome == sc::Outcome::kSucceeded;
  }

 private:
  sc::BackingService& service_;
  sc::CompletionBoard& board_;
};

sc::ReserveSettings Pool(std::size_t ceiling) {
  return {.allocation_class = 0, .size = Bytes(kExtent), .count = ceiling};
}

TEST(BackingServicePool, PreallocatedHandlesAreTakenAndKeptOneForOne) {
  pr::fake::FakeDeviceMemory memory(2_MiB, 32_MiB);
  const auto place = memory.Reserve(Bytes(4 * kExtent)).value();
  llmp::base::WakeFlag wake;
  sc::CompletionBoard board(64, wake);
  sc::BackingService service(&memory, board, {.capacity = 64, .reserved = 16, .batch = 8}, Pool(8));
  Lane lane(service, board);

  std::string failure;
  EXPECT_EQ(service.Preallocate(4, &failure), 4U);
  EXPECT_TRUE(failure.empty());
  service.ManageReserve(0);  // the headroom is exactly what it holds
  EXPECT_EQ(service.ReserveAvailable(), 4U);
  EXPECT_EQ(service.create_stats().startup_pool_creates, 4U);
  EXPECT_EQ(memory.backings(), 4U);

  // A map takes one and a plain unmap keeps it again: no create or release.
  ASSERT_TRUE(lane.Run(sc::BackingWork::Kind::kMap, place, 0));
  EXPECT_EQ(service.ReserveAvailable(), 3U);
  EXPECT_FALSE(service.Turn());  // managed with no grant: no refill
  EXPECT_EQ(memory.created_backings(), 4U);
  ASSERT_TRUE(lane.Run(sc::BackingWork::Kind::kUnmap, place, 0));
  EXPECT_EQ(service.ReserveAvailable(), 4U);
  EXPECT_EQ(memory.created_backings(), 4U);
  EXPECT_EQ(memory.released_backings(), 0U);
  EXPECT_EQ(service.create_stats().post_init_creates(), 0U);

  service.Close();
  service.Run();
  EXPECT_TRUE(service.ReleaseReserve());
  EXPECT_EQ(memory.backings(), 0U);
  ASSERT_TRUE(memory.Free(place).has_value());
}

// D-124 as amended: a device short of memory gives a smaller pool, never a
// refused start; what was made is kept, and the pool grows no further.
TEST(BackingServicePool, PreallocationIsBestEffort) {
  pr::fake::FakeDeviceMemory memory(2_MiB, 4_MiB);  // two extents at most
  llmp::base::WakeFlag wake;
  sc::CompletionBoard board(64, wake);
  sc::BackingService service(&memory, board, {.capacity = 64, .reserved = 16, .batch = 8}, Pool(4));
  std::string failure;
  EXPECT_EQ(service.Preallocate(4, &failure), 2U);
  EXPECT_FALSE(failure.empty());
  EXPECT_EQ(service.ReserveAvailable(), 2U);
  EXPECT_EQ(service.create_stats().startup_pool_creates, 2U);
  EXPECT_EQ(memory.backings(), 2U);
  service.ManageReserve(2);
  EXPECT_FALSE(service.Turn());  // the ceiling fell to what it holds
  EXPECT_EQ(service.ReserveTarget(), 2U);
  service.Close();
  service.Run();
  EXPECT_TRUE(service.ReleaseReserve());
  EXPECT_EQ(memory.backings(), 0U);
}

// A create of unknown outcome ends the preallocation too: the backing it
// may have made counts against the ceiling, and nothing already made is
// shed for it.
TEST(BackingServicePool, AnUnknownPreallocationKeepsWhatItMade) {
  pr::fake::FakeDeviceMemory memory(2_MiB, 32_MiB);
  llmp::base::WakeFlag wake;
  sc::CompletionBoard board(64, wake);
  sc::BackingService service(&memory, board, {.capacity = 64, .reserved = 16, .batch = 8}, Pool(8));
  EXPECT_EQ(service.Preallocate(2), 2U);
  memory.FailNext(pr::fake::Operation::kCreate, ProviderError::kUnknown, /*applied=*/true);
  std::string failure;
  EXPECT_EQ(service.Preallocate(2, &failure), 0U);
  EXPECT_FALSE(failure.empty());
  EXPECT_EQ(service.ReserveAvailable(), 2U);
  service.ManageReserve(4);
  EXPECT_FALSE(service.Turn());  // nothing shed, and no refill past the ceiling
  EXPECT_EQ(service.ReserveAvailable(), 2U);
  EXPECT_EQ(memory.released_backings(), 0U);
  service.Close();
  service.Run();
  EXPECT_TRUE(service.ReleaseReserve());
}

// The pool holds at most the budget's headroom: past it, it sheds (unless
// paging in flight may count handles twice); within it, it may refill.
TEST(BackingServicePool, RebalanceShedsPastTheHeadroomAndGrantsWithinIt) {
  pr::fake::FakeDeviceMemory memory(2_MiB, 64_MiB);
  llmp::base::WakeFlag wake;
  sc::CompletionBoard board(64, wake);
  sc::BackingService service(&memory, board, {.capacity = 64, .reserved = 16, .batch = 8}, Pool(8));
  ASSERT_EQ(service.Preallocate(6), 6U);
  service.ManageReserve(0);

  // Four page-ins and evictions in flight: the six may all be counted twice.
  service.Rebalance(Bytes(2 * kExtent), 4);
  EXPECT_FALSE(service.Turn());
  EXPECT_EQ(service.ReserveAvailable(), 6U);

  // None in flight: the four past the headroom go, released by the lane.
  service.Rebalance(Bytes(2 * kExtent), 0);
  EXPECT_TRUE(service.Turn());
  EXPECT_EQ(service.ReserveAvailable(), 2U);
  EXPECT_EQ(memory.released_backings(), 4U);
  EXPECT_EQ(memory.backings(), 2U);

  // The headroom grows to seven: the pool may make five more, lazily.
  service.Rebalance(Bytes(7 * kExtent), 0);
  EXPECT_EQ(service.ReserveTarget(), 7U);
  while (service.Turn()) {
  }
  EXPECT_EQ(service.ReserveAvailable(), 7U);
  EXPECT_EQ(service.create_stats().reserve_attempts, 5U);

  // Never past its ceiling, however large the headroom.
  service.Rebalance(Bytes(64 * kExtent), 0);
  EXPECT_EQ(service.ReserveTarget(), 8U);

  service.Close();
  service.Run();
  EXPECT_TRUE(service.ReleaseReserve());
  EXPECT_EQ(memory.backings(), 0U);
}

// A drain gives the pool's empty handles back, waits for the provider to
// release them on the running lane, and keeps the pool from refilling (an
// unmap's backing is then released) until it is restored.
TEST(BackingServicePool, DrainReleasesOnARunningLaneAndLowersTheCeiling) {
  pr::fake::FakeDeviceMemory memory(2_MiB, 32_MiB);
  const auto place = memory.Reserve(Bytes(2 * kExtent)).value();
  llmp::base::WakeFlag wake;
  sc::CompletionBoard board(64, wake);
  sc::BackingService service(&memory, board, {.capacity = 64, .reserved = 16, .batch = 8}, Pool(8));
  ASSERT_EQ(service.Preallocate(4), 4U);
  service.ManageReserve(0);

  service.Starting();  // before its thread: a drain meanwhile waits for the lane
  std::thread worker([&] { service.Run(); });
  EXPECT_EQ(service.DrainReserve(1), 3U);
  EXPECT_EQ(service.ReserveAvailable(), 1U);
  EXPECT_EQ(memory.released_backings(), 3U);
  service.Close();
  worker.join();
  EXPECT_EQ(board.publications(), 0U);  // the lane's wakes publish nothing
  EXPECT_TRUE(board.Harvest(16).empty());

  // Lowered: a map takes the one it holds, and its unmap releases it.
  sc::BackingService again(&memory, board, {.capacity = 64, .reserved = 16, .batch = 8}, Pool(8));
  Lane lane(again, board);
  ASSERT_EQ(again.Preallocate(1), 1U);
  again.ManageReserve(0);
  EXPECT_EQ(again.DrainReserve(0), 1U);  // no lane running: released here
  ASSERT_TRUE(lane.Run(sc::BackingWork::Kind::kMap, place, 0));
  ASSERT_TRUE(lane.Run(sc::BackingWork::Kind::kUnmap, place, 0));
  EXPECT_EQ(again.ReserveAvailable(), 0U);
  again.RestoreReserve();
  ASSERT_TRUE(lane.Run(sc::BackingWork::Kind::kMap, place, 0));
  ASSERT_TRUE(lane.Run(sc::BackingWork::Kind::kUnmap, place, 0));
  EXPECT_EQ(again.ReserveAvailable(), 1U);  // restored: kept again
  again.Close();
  again.Run();
  EXPECT_TRUE(again.ReleaseReserve());
  EXPECT_TRUE(service.ReleaseReserve());
  EXPECT_EQ(memory.backings(), 0U);
  ASSERT_TRUE(memory.Free(place).has_value());
}

// A drain that keeps some handles (pressure from outside) keeps the pool
// from growing back until restored: a load taking one leaves room under the
// lowered ceiling, but the pool never refills it with a create; only an
// unmap's backing may come back into it, which adds no physical memory.
TEST(BackingServicePool, ADrainedPoolCreatesNothingUntilRestored) {
  pr::fake::FakeDeviceMemory memory(2_MiB, 64_MiB);
  const auto place = memory.Reserve(Bytes(2 * kExtent)).value();
  llmp::base::WakeFlag wake;
  sc::CompletionBoard board(64, wake);
  sc::BackingService service(&memory, board, {.capacity = 64, .reserved = 16, .batch = 8}, Pool(8));
  Lane lane(service, board);
  ASSERT_EQ(service.Preallocate(4), 4U);
  service.ManageReserve(0);
  EXPECT_EQ(service.DrainReserve(2), 2U);  // no lane running: released here
  EXPECT_EQ(service.ReserveAvailable(), 2U);

  ASSERT_TRUE(lane.Run(sc::BackingWork::Kind::kMap, place, 0));
  EXPECT_EQ(service.ReserveAvailable(), 1U);
  service.Rebalance(Bytes(16 * kExtent), 0);  // headroom to spare
  while (service.Turn()) {
  }
  EXPECT_EQ(memory.created_backings(), 4U);  // no refill while drained
  EXPECT_EQ(service.ReserveAvailable(), 1U);
  ASSERT_TRUE(lane.Run(sc::BackingWork::Kind::kUnmap, place, 0));
  EXPECT_EQ(service.ReserveAvailable(), 2U);  // kept: back under the ceiling
  EXPECT_EQ(memory.released_backings(), 2U);
  ASSERT_TRUE(lane.Run(sc::BackingWork::Kind::kMap, place, 0));
  ASSERT_TRUE(lane.Run(sc::BackingWork::Kind::kMap, place, 1));
  ASSERT_TRUE(lane.Run(sc::BackingWork::Kind::kUnmap, place, 0));
  ASSERT_TRUE(lane.Run(sc::BackingWork::Kind::kUnmap, place, 1));
  EXPECT_EQ(service.ReserveAvailable(), 2U);  // never past the lowered ceiling
  EXPECT_EQ(memory.created_backings(), 4U);
  EXPECT_EQ(memory.backings(), 2U);

  service.RestoreReserve();
  service.Rebalance(Bytes(6 * kExtent), 0);
  while (service.Turn()) {
  }
  EXPECT_EQ(service.ReserveAvailable(), 6U);  // grows back within the headroom
  service.Close();
  service.Run();
  EXPECT_TRUE(service.ReleaseReserve());
  EXPECT_EQ(memory.backings(), 0U);
  ASSERT_TRUE(memory.Free(place).has_value());
}

// A provider whose creates wait, once armed, until the test opens them:
// a refill held in flight while a drain begins.
class GatedMemory final : public pr::DeviceMemory {
 public:
  explicit GatedMemory(pr::fake::FakeDeviceMemory& inner) : inner_(inner) {}
  void Arm() {
    std::lock_guard lock(mutex_);
    armed_ = true;
  }
  void WaitEntered() {
    std::unique_lock lock(mutex_);
    cv_.wait(lock, [&] { return entered_; });
  }
  void Open() {
    {
      std::lock_guard lock(mutex_);
      armed_ = false;
    }
    cv_.notify_all();
  }

  std::span<const pr::AllocationClass> Classes() const override { return inner_.Classes(); }
  Bytes Granularity() const override { return inner_.Granularity(); }
  Bytes ReservationAlignment() const override { return inner_.ReservationAlignment(); }
  std::expected<pr::ReservationId, pr::Failure> Reserve(Bytes size) override {
    return inner_.Reserve(size);
  }
  std::expected<pr::AddressRange, pr::Failure> RangeOf(pr::ReservationId r) const override {
    return inner_.RangeOf(r);
  }
  std::expected<void, pr::Failure> Free(pr::ReservationId r) override { return inner_.Free(r); }
  std::expected<pr::BackingId, pr::Failure> Create(std::size_t allocation_class,
                                                   Bytes size) override {
    {
      std::unique_lock lock(mutex_);
      if (armed_) {
        entered_ = true;
        cv_.notify_all();
        cv_.wait(lock, [&] { return !armed_; });
      }
    }
    return inner_.Create(allocation_class, size);
  }
  std::expected<void, pr::Failure> Release(pr::BackingId backing) override {
    return inner_.Release(backing);
  }
  std::expected<void, pr::Failure> Map(pr::ReservationId r, Bytes offset,
                                       pr::BackingId backing) override {
    return inner_.Map(r, offset, backing);
  }
  std::expected<void, pr::Failure> SetAccess(pr::ReservationId r, Bytes offset, Bytes size,
                                             pr::Access access) override {
    return inner_.SetAccess(r, offset, size, access);
  }
  std::expected<void, pr::Failure> Unmap(pr::ReservationId r, Bytes offset, Bytes size) override {
    return inner_.Unmap(r, offset, size);
  }
  std::optional<pr::BackingId> MappedAt(pr::ReservationId r, Bytes offset) const override {
    return inner_.MappedAt(r, offset);
  }

 private:
  pr::fake::FakeDeviceMemory& inner_;
  std::mutex mutex_;
  std::condition_variable cv_;
  bool armed_ = false;
  bool entered_ = false;
};

// Three pooled, a refill's create held in flight on the running lane, and
// a drain to two begun meanwhile (seen once the pool's target falls to its
// ceiling); then the create is let through.
struct InFlightDrain {
  pr::fake::FakeDeviceMemory fake{2_MiB, 64_MiB};
  GatedMemory memory{fake};
  llmp::base::WakeFlag wake;
  sc::CompletionBoard board{64, wake};
  sc::BackingService service{&memory, board, {.capacity = 64, .reserved = 16, .batch = 8}, Pool(8)};
  std::thread worker;
  std::size_t released = 0;

  void Run(bool refuse_a_release) {
    ASSERT_EQ(service.Preallocate(3), 3U);
    service.ManageReserve(0);
    service.Starting();
    worker = std::thread([&] { service.Run(); });
    memory.Arm();
    service.Rebalance(Bytes(8 * kExtent), 0);  // grants refills: the lane creates
    memory.WaitEntered();
    if (refuse_a_release) fake.FailNext(pr::fake::Operation::kRelease, ProviderError::kFailed);
    std::thread drain([&] { released = service.DrainReserve(2); });
    while (service.ReserveTarget() != 2) std::this_thread::yield();  // drained
    memory.Open();
    drain.join();
  }
  void Stop() {
    service.Close();
    worker.join();
    EXPECT_TRUE(service.ReleaseReserve());
    EXPECT_EQ(fake.backings(), 0U);
  }
};

// The create in flight is not counted against the lowered ceiling (it is
// shed when it ends), so the drain keeps the two it was asked to, and it
// waits for that create: once it returns, the provider holds only them.
TEST(BackingServicePool, ADrainShedsACreateInFlightAndWaitsForIt) {
  InFlightDrain run;
  run.Run(false);
  EXPECT_EQ(run.released, 2U);  // one pooled, and the one created meanwhile
  EXPECT_EQ(run.service.ReserveAvailable(), 2U);
  EXPECT_EQ(run.fake.backings(), 2U);
  EXPECT_EQ(run.fake.created_backings(), 4U);
  run.Stop();
}

// A create in flight shed by a drain goes through the pool's release, so a
// refused release keeps it (released at teardown, never leaked), and the
// scheduler's next Rebalance after the lane's next command sheds again
// what the refusal left past the ceiling, even with its inputs unchanged.
TEST(BackingServicePool, ARefusedReleaseOfAShedCreateIsKeptAndShedAgain) {
  InFlightDrain run;
  run.Run(true);
  EXPECT_EQ(run.service.ReserveAvailable(), 3U);  // the refused one came back
  EXPECT_EQ(run.fake.backings(), 3U);
  run.service.Rebalance(Bytes(8 * kExtent), 0);  // refused: no shed before a command
  EXPECT_EQ(run.service.ReserveAvailable(), 3U);

  const auto operation = run.board.Open();  // a command that is not VMM work
  ASSERT_EQ(run.service.Submit(sc::DeviceCommand{.operation = operation, .work = sc::DeviceWork{}}),
            llmp::base::PushResult::kAccepted);
  bool seen = false;
  while (!seen) {
    for (const auto& o : run.board.Harvest(8)) seen = seen || o.operation == operation;
    if (!seen) std::this_thread::yield();
  }
  (void)run.board.Close(operation);
  run.service.Rebalance(Bytes(8 * kExtent), 0);  // the same inputs: shed again
  const auto give_up = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (run.fake.backings() != 2U && std::chrono::steady_clock::now() < give_up) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  EXPECT_EQ(run.service.ReserveAvailable(), 2U);
  EXPECT_EQ(run.fake.backings(), 2U);
  run.Stop();
}

// The pool's bookkeeping under every thread at once: the lane running,
// the scheduler's thread mapping and unmapping through it while it moves
// the headroom, and the driver draining and restoring. No handle is lost
// or released twice (every map and unmap succeeds, and the provider's
// backings are always what the pool and the mapped places hold once it
// settles), and the settled pool is within the last headroom.
TEST(BackingServicePool, ConcurrentPagingRebalancesAndDrainsLoseNoHandle) {
  constexpr std::uint64_t kPlaces = 8;
  pr::fake::FakeDeviceMemory memory(2_MiB, 128_MiB);
  const auto place = memory.Reserve(Bytes(kPlaces * kExtent)).value();
  llmp::base::WakeFlag wake;
  sc::CompletionBoard board(256, wake);
  sc::BackingService service(&memory, board, {.capacity = 64, .reserved = 16, .batch = 8},
                             Pool(24));
  ASSERT_EQ(service.Preallocate(12), 12U);
  service.ManageReserve(0);
  service.Starting();  // before its thread: a drain meanwhile waits for the lane
  std::thread worker([&] { service.Run(); });

  std::atomic<bool> paging = true;
  std::atomic<std::uint64_t> failures = 0;
  std::array<bool, kPlaces> mapped{};
  std::thread scheduler([&] {
    std::uint64_t seed = 12345;
    const auto next = [&seed] {
      seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
      return seed >> 33U;
    };
    for (int round = 0; round < 4000; ++round) {
      const std::uint64_t at = next() % kPlaces;
      const auto kind = mapped[at] ? sc::BackingWork::Kind::kUnmap : sc::BackingWork::Kind::kMap;
      const auto operation = board.Open();
      if (service.Submit(sc::DeviceCommand{.operation = operation,
                                           .work = sc::BackingWork{.kind = kind,
                                                                   .reservation = place,
                                                                   .offset = Bytes(at * kExtent),
                                                                   .size = Bytes(kExtent),
                                                                   .allocation_class = 0,
                                                                   .retain = false,
                                                                   .reuse = false}}) !=
          llmp::base::PushResult::kAccepted) {
        ++failures;
        (void)board.Close(operation);
        continue;
      }
      bool done = false;
      while (!done) {
        for (const auto& seen : board.Harvest(8)) {
          if (seen.operation != operation) continue;
          if (seen.acceptance == sc::Acceptance::kNotStarted) {
            done = true;  // refused: nothing changed
            ++failures;
          } else if (seen.terminal) {
            done = true;
            if (seen.terminal->outcome != sc::Outcome::kSucceeded) ++failures;
          }
        }
        service.Rebalance(Bytes((next() % 20) * kExtent), next() % 3);
        if (!done) std::this_thread::yield();
      }
      (void)board.Close(operation);
      mapped[at] = !mapped[at];
    }
    paging = false;
  });
  std::uint64_t drains = 0;
  while (paging) {
    (void)service.DrainReserve(drains % 5);
    service.RestoreReserve();
    ++drains;
    std::this_thread::yield();
  }
  scheduler.join();
  EXPECT_EQ(failures.load(), 0U);
  EXPECT_GT(drains, 0U);

  // Settled at a headroom of five: the pool sheds to it or grows to it.
  std::size_t held = 0;
  for (const bool at : mapped) held += at ? 1 : 0;
  const auto give_up = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  bool settled = false;
  while (!settled && std::chrono::steady_clock::now() < give_up) {
    service.Rebalance(Bytes(5 * kExtent), 0);
    settled = service.ReserveAvailable() == 5 && memory.backings() == held + 5;
    if (!settled) std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  EXPECT_TRUE(settled) << service.ReserveAvailable() << " pooled, " << memory.backings()
                       << " backings, " << held << " mapped";
  service.Close();
  worker.join();
  EXPECT_TRUE(service.ReleaseReserve());
  EXPECT_EQ(memory.backings(), held);
}

// ---- the paged node's pool against its budget

class PagedNodePool : public ::testing::Test {
 protected:
  void Open(std::optional<en::NodePool> pool, std::uint64_t extents) {
    node_ = std::make_unique<en::PagedNode>(
        en::NodeSettings{.compute_streams = 1, .slots = 1, .slot_bytes = kExtent});
    ASSERT_TRUE(node_->Open());
    fixed_ = node_->catalog().OccupancyOf(node_->domain()).Total().value();
    fixed_backings_ = node_->LiveBackings();
    budget_ = fixed_ + extents * kExtent;
    ASSERT_TRUE(node_->Start(Bytes(budget_), pool));
  }
  void TearDown() override {
    if (node_ != nullptr) {
      EXPECT_TRUE(node_->TearDown({}));
      EXPECT_EQ(node_->LiveBackings(), 0U);
    }
  }
  std::uint64_t Occupancy() {
    std::uint64_t occupancy = 0;
    EXPECT_TRUE(node_->Call(
        [&]() -> en::Status {
          occupancy = node_->catalog().OccupancyOf(node_->domain()).Total().value();
          return {};
        },
        "reading the occupancy"));
    return occupancy;
  }
  // Invariant 5 with the pool counted once: what it holds beside the
  // occupancy within the budget. Waits (a few seconds at most) for the
  // scheduler's turns and the lane to settle.
  bool WithinBudget() {
    const auto give_up = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (std::chrono::steady_clock::now() < give_up) {
      const std::uint64_t pooled = node_->ReserveAvailable() * kExtent;
      if (Occupancy() + pooled <= node_->budget().value() &&
          node_->LiveBackings() == fixed_backings_ + node_->ReserveAvailable()) {
        return true;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return false;
  }
  bool Holds(std::size_t handles) {
    const auto give_up = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (std::chrono::steady_clock::now() < give_up) {
      (void)Occupancy();  // a scheduler turn
      if (node_->ReserveAvailable() == handles &&
          node_->LiveBackings() == fixed_backings_ + handles) {
        return true;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return false;
  }

  std::unique_ptr<en::PagedNode> node_;
  std::uint64_t fixed_ = 0;
  std::uint64_t budget_ = 0;
  std::size_t fixed_backings_ = 0;
};

TEST_F(PagedNodePool, ADedicatedPoolFillsTheHeadroomAtStart) {
  Open(en::NodePool{.ceiling = Bytes(64 * kExtent), .fill = true}, 6);
  EXPECT_TRUE(node_->pool_shortfall().empty());
  EXPECT_EQ(node_->ReserveAvailable(), 6U);
  EXPECT_EQ(node_->ReserveTarget(), 6U);
  EXPECT_EQ(node_->LiveBackings(), fixed_backings_ + 6U);
  EXPECT_EQ(node_->backing_create_counts().startup_pool_creates, 6U);
  node_->Run();
  EXPECT_TRUE(WithinBudget());
}

// A2: charges outside the pool (plans, graphs, token and request memory,
// pinned staging) take budget the pool held: it gives that much back, so
// physical use never exceeds the budget as they grow, and grows back as
// they shrink.
TEST_F(PagedNodePool, ChargesOutsideThePoolShrinkItAndPhysicalUseStaysWithinTheBudget) {
  Open(en::NodePool{.ceiling = Bytes(64 * kExtent), .fill = true}, 6);
  node_->Run();
  const std::uint64_t released = node_->ReleasedBackings();
  std::uint64_t shortfall = 0;
  ASSERT_TRUE(node_->SetTokenCharge(2 * kExtent, &shortfall));
  ASSERT_TRUE(node_->SetRequestCharge(kExtent, &shortfall));
  EXPECT_TRUE(Holds(3));
  EXPECT_TRUE(WithinBudget());
  EXPECT_EQ(node_->ReleasedBackings(), released + 3);
  EXPECT_EQ(Occupancy(), fixed_ + 3 * kExtent);

  // Given back: the pool refills lazily, within the headroom again.
  ASSERT_TRUE(node_->SetTokenCharge(0, &shortfall));
  EXPECT_TRUE(Holds(5));
  EXPECT_TRUE(WithinBudget());
  ASSERT_TRUE(node_->SetRequestCharge(0, &shortfall));
  EXPECT_TRUE(Holds(6));
}

// A lower budget sheds the pool too; the budget never drops below what
// the domain's extents hold (D-050).
TEST_F(PagedNodePool, ALowerBudgetShedsThePoolAndStopsAtTheClaims) {
  Open(en::NodePool{.ceiling = Bytes(64 * kExtent), .fill = true}, 6);
  node_->Run();
  const auto lowered = node_->SetBudget(Bytes(fixed_ + 2 * kExtent));
  ASSERT_TRUE(lowered);
  EXPECT_EQ(lowered->value(), fixed_ + 2 * kExtent);
  EXPECT_TRUE(Holds(2));
  EXPECT_TRUE(WithinBudget());
  // Below the pinned fixed memory: clamped to it.
  const auto clamped = node_->SetBudget(Bytes(fixed_ / 2));
  ASSERT_TRUE(clamped);
  EXPECT_EQ(clamped->value(), fixed_);
  EXPECT_EQ(node_->budget().value(), fixed_);
  EXPECT_TRUE(Holds(0));
  ASSERT_TRUE(node_->SetBudget(Bytes(budget_)));
  EXPECT_TRUE(Holds(6));
}

// A3: the release route and pressure from outside free physical memory in
// dedicated mode: the drain releases the pool's handles to the system, and
// the pool stays empty until restored.
TEST_F(PagedNodePool, TheDrainReleasesThePoolUntilRestored) {
  Open(en::NodePool{.ceiling = Bytes(64 * kExtent), .fill = true}, 6);
  node_->Run();
  const std::uint64_t released = node_->ReleasedBackings();
  EXPECT_EQ(node_->DrainPool(4), 2U);
  EXPECT_EQ(node_->ReserveAvailable(), 4U);
  EXPECT_EQ(node_->DrainPool(0), 4U);
  EXPECT_EQ(node_->ReleasedBackings(), released + 6);
  EXPECT_EQ(node_->LiveBackings(), fixed_backings_);
  EXPECT_TRUE(Holds(0));  // the scheduler's turns do not refill it
  node_->RestorePool();
  EXPECT_TRUE(Holds(6));
  EXPECT_TRUE(WithinBudget());
}

TEST_F(PagedNodePool, SharedModeKeepsNoPool) {
  Open(std::nullopt, 6);
  EXPECT_EQ(node_->ReserveAvailable(), 0U);
  EXPECT_EQ(node_->ReserveTarget(), 0U);
  EXPECT_EQ(node_->LiveBackings(), fixed_backings_);
  EXPECT_EQ(node_->DrainPool(0), 0U);
  node_->Run();
  EXPECT_TRUE(Holds(0));
}

// A device short of memory starts the node with a smaller pool.
TEST_F(PagedNodePool, APoolTheDeviceCannotFillStartsSmaller) {
  pr::fake::ConfigureFakeDevice({.capacity = Bytes(64 * kExtent)});
  struct Restore {
    ~Restore() { pr::fake::ConfigureFakeDevice({}); }
  } restore;
  node_ = std::make_unique<en::PagedNode>(
      en::NodeSettings{.compute_streams = 1, .slots = 1, .slot_bytes = kExtent});
  ASSERT_TRUE(node_->Open());
  fixed_ = node_->catalog().OccupancyOf(node_->domain()).Total().value();
  fixed_backings_ = node_->LiveBackings();
  const std::size_t room = 64 - fixed_backings_;
  budget_ = fixed_ + (room + 16) * kExtent;
  ASSERT_TRUE(node_->Start(Bytes(budget_), en::NodePool{.ceiling = Bytes(1024 * kExtent)}));
  EXPECT_FALSE(node_->pool_shortfall().empty());
  EXPECT_EQ(node_->ReserveAvailable(), room);
}

TEST(PagedNodeModes, TheBudgetBeforeRunIsAppliedAtOnce) {
  en::PagedNode node(en::NodeSettings{.compute_streams = 1, .slots = 1, .slot_bytes = kExtent});
  ASSERT_TRUE(node.Open());
  const auto fixed = node.catalog().OccupancyOf(node.domain()).Total().value();
  const Bytes original(fixed + 10 * kExtent);
  ASSERT_TRUE(node.Start(original));
  EXPECT_EQ(node.budget(), original);
  const Bytes trimmed(fixed + 4 * kExtent);
  EXPECT_EQ(node.SetBudget(trimmed).value_or(Bytes()), trimmed);
  EXPECT_EQ(node.budget(), trimmed);
  EXPECT_EQ(node.SetBudget(original).value_or(Bytes()), original);
  EXPECT_EQ(node.budget(), original);
  EXPECT_TRUE(node.TearDown({}));
}

// ---- the runtime's policy

TEST(MemoryPolicy, TheDeviceChoosesUnlessConfigured) {
  llmp::config::MemoryConfig memory;
  const auto unified = rt::ChooseMemoryPolicy(memory, true);
  EXPECT_EQ(unified.mode, llmp::config::MemoryMode::kDedicated);
  EXPECT_FALSE(unified.configured);
  EXPECT_TRUE(unified.pool);
  EXPECT_FALSE(unified.pageable_workspace);
  EXPECT_FALSE(unified.idle_release);
  EXPECT_FALSE(unified.shrink_budget);

  const auto discrete = rt::ChooseMemoryPolicy(memory, false);
  EXPECT_EQ(discrete.mode, llmp::config::MemoryMode::kShared);
  EXPECT_FALSE(discrete.pool);
  EXPECT_TRUE(discrete.pageable_workspace);
  EXPECT_TRUE(discrete.idle_release);
  EXPECT_TRUE(discrete.shrink_budget);
  EXPECT_TRUE(discrete.release_under_pressure);

  memory.mode = {llmp::config::MemoryMode::kDedicated, true};
  const auto configured = rt::ChooseMemoryPolicy(memory, false);
  EXPECT_EQ(configured.mode, llmp::config::MemoryMode::kDedicated);
  EXPECT_TRUE(configured.configured);
  EXPECT_TRUE(configured.pool);

  memory.mode = {llmp::config::MemoryMode::kShared, true};
  memory.idle_release_seconds = {0, true};
  const auto shared = rt::ChooseMemoryPolicy(memory, true);
  EXPECT_EQ(shared.mode, llmp::config::MemoryMode::kShared);
  EXPECT_FALSE(shared.idle_release);  // 0 disables it
}

TEST(MemoryPolicy, TheHeadroomOnAUnifiedDeviceIsTheHostsAlone) {
  const auto headroom = rt::BudgetHeadroom(
      {.host_available = 100 * kExtent + 5, .host_reserve = 10 * kExtent, .extent = kExtent});
  ASSERT_TRUE(headroom.has_value());
  EXPECT_EQ(*headroom, 90 * kExtent);  // whole extents
  EXPECT_EQ(
      rt::BudgetHeadroom({.host_available = 4 * kExtent, .host_reserve = 10 * kExtent}).value_or(1),
      0U);
  EXPECT_FALSE(rt::BudgetHeadroom({}).has_value());  // nothing known
}

// A5: on a discrete device the budget is sized from the device's free
// memory, not the host's MemAvailable (the fake device reports a small one).
TEST(MemoryPolicy, ADiscreteDevicesBudgetIsSizedFromItsFreeMemory) {
  pr::fake::ConfigureFakeDevice({.capacity = Bytes(std::uint64_t{12} << 30U),
                                 .unified = false,
                                 .free = Bytes(std::uint64_t{6} << 30U)});
  struct Restore {
    ~Restore() { pr::fake::ConfigureFakeDevice({}); }
  } restore;
  const auto facts = pr::QueryDeviceFacts(0);
  ASSERT_TRUE(facts.has_value());
  EXPECT_FALSE(facts->unified);
  const auto info = pr::QueryDeviceMemory();
  ASSERT_TRUE(info.has_value());
  EXPECT_EQ(info->free, std::uint64_t{6} << 30U);

  llmp::config::MemoryConfig memory;
  EXPECT_EQ(rt::ChooseMemoryPolicy(memory, facts->unified).mode, llmp::config::MemoryMode::kShared);
  // 120 GiB of host memory available, 6 GiB free on the device: the
  // device's free memory less its reserve bounds the budget.
  const std::uint64_t reserve = (std::uint64_t{512} << 20U) + rt::kDeviceUncountedMargin;
  const auto headroom = rt::BudgetHeadroom({.host_available = std::uint64_t{120} << 30U,
                                            .host_reserve = std::uint64_t{8} << 30U,
                                            .device_free = info->free,
                                            .device_reserve = reserve,
                                            .extent = kExtent});
  ASSERT_TRUE(headroom.has_value());
  EXPECT_EQ(*headroom, (info->free - reserve) / kExtent * kExtent);
  // Host memory short of the device's: the host bounds it.
  EXPECT_EQ(rt::BudgetHeadroom({.host_available = std::uint64_t{3} << 30U,
                                .host_reserve = std::uint64_t{1} << 30U,
                                .device_free = info->free,
                                .device_reserve = reserve,
                                .extent = kExtent})
                .value_or(0),
            std::uint64_t{2} << 30U);
}

TEST(PressureTrimModes, ObserveAndHysteresisRecovery) {
  rt::PressureTrim trim;
  rt::PressureTrim::Clock::time_point now{};
  constexpr auto kLook = std::chrono::milliseconds(250);

  EXPECT_FALSE(trim.Observe({.available = 4096 * (1ULL << 20U), .stall = {}}, now).now);
  EXPECT_FALSE(trim.pressed());
  now += kLook;
  auto t = trim.Observe({.available = 300 * (1ULL << 20U), .stall = {}}, now);
  ASSERT_TRUE(t.now);
  EXPECT_TRUE(trim.pressed());
  EXPECT_GT(t.needed, 0U);
  EXPECT_FALSE(trim.Trimmed(t.needed, t.needed, now));
  now += kLook;
  EXPECT_FALSE(trim.Observe({.available = 900 * (1ULL << 20U), .stall = {}}, now).now);
  EXPECT_TRUE(trim.pressed());
  now += kLook;
  EXPECT_FALSE(trim.Observe({.available = 1600 * (1ULL << 20U), .stall = {}}, now).now);
  EXPECT_FALSE(trim.pressed());
}

}  // namespace
