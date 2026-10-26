// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Physical memory modes (dedicated node pool + shared-machine mode):
// verifies dedicated pool pre-allocation and handle recycling, shared-machine
// demand allocation, dynamic budget adjustments, and external memory pressure trimming.

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "base/bytes.h"
#include "engine/paged_node.h"
#include "providers/fake/fake_device_memory.h"
#include "runtime/pressure_trim.h"
#include "scheduler/services.h"

namespace {

namespace en = llmp::engine;
namespace sc = llmp::scheduler;
namespace pr = llmp::providers;
namespace rt = llmp::runtime;
using llmp::base::Bytes;
using llmp::base::operator""_MiB;

TEST(BackingServicePool, PreallocatesAndRecyclesHandles) {
  pr::fake::FakeDeviceMemory memory(2_MiB, 32_MiB);
  llmp::base::WakeFlag wake;
  sc::CompletionBoard board(64, wake);
  sc::BackingService service(&memory, board, {.capacity = 64, .reserved = 16, .batch = 8});

  // Pre-allocate 4 handles into reserve pool.
  auto res = service.SetReserve(0, 2_MiB, 4, /*preallocate_now=*/true);
  ASSERT_TRUE(res.has_value()) << res.error();

  EXPECT_EQ(service.ReserveAvailable(), 4U);
  EXPECT_EQ(service.ReserveTarget(), 4U);
  EXPECT_EQ(service.create_stats().startup_pool_creates, 4U);
  EXPECT_EQ(service.create_stats().post_init_creates(), 0U);
  EXPECT_EQ(memory.created_backings(), 4U);
  EXPECT_EQ(memory.released_backings(), 0U);
  EXPECT_EQ(memory.backings(), 4U);

  // Take a handle from reserve.
  auto h1 = service.TakeFromReserve(0, 2_MiB);
  ASSERT_TRUE(h1.has_value());
  EXPECT_EQ(service.ReserveAvailable(), 3U);
  EXPECT_EQ(memory.backings(), 4U);  // still alive, just mapped/taken

  // A pre-allocated pool recycles handles on eviction and does not create extra
  // handles in the background while handles are mapped out of the pool.
  EXPECT_FALSE(service.Turn());
  EXPECT_EQ(service.ReserveAvailable(), 3U);
  EXPECT_EQ(memory.created_backings(), 4U);
  EXPECT_EQ(service.create_stats().post_init_creates(), 0U);

  // Return handle to reserve.
  EXPECT_TRUE(service.ReturnToReserve(*h1));
  EXPECT_EQ(service.ReserveAvailable(), 4U);

  // Drain reserve handles down to 0.
  const std::size_t drained = service.DrainReserve(0);
  EXPECT_EQ(drained, 4U);
  EXPECT_EQ(service.ReserveAvailable(), 0U);
  EXPECT_EQ(memory.released_backings(), 4U);
  EXPECT_EQ(memory.backings(), 0U);
}

TEST(BackingServicePool, RollsBackOnAllocationFailureWithoutLeaking) {
  // Capacity of 4 MiB allows only 2 extents of 2 MiB.
  pr::fake::FakeDeviceMemory memory(2_MiB, 4_MiB);
  llmp::base::WakeFlag wake;
  sc::CompletionBoard board(64, wake);
  sc::BackingService service(&memory, board, {.capacity = 64, .reserved = 16, .batch = 8});

  // Requesting 4 extents should fail and roll back all pre-allocated handles.
  auto res = service.SetReserve(0, 2_MiB, 4, /*preallocate_now=*/true);
  EXPECT_FALSE(res.has_value());
  EXPECT_EQ(service.ReserveAvailable(), 0U);
  EXPECT_EQ(service.ReserveTarget(), 0U);
  EXPECT_EQ(service.create_stats().startup_pool_creates, 0U);
  EXPECT_EQ(memory.backings(), 0U);
}

TEST(PagedNodeModes, DedicatedNodePoolPreallocatesAndDrains) {
  en::NodeSettings settings{.compute_streams = 1, .slots = 1, .slot_bytes = en::kPagedExtent};
  settings.memory_mode = en::MemoryMode::kDedicated;
  settings.preallocate_pool = true;

  en::PagedNode node(settings);
  ASSERT_TRUE(node.Open());
  const auto initial_created = node.CreatedBackings();
  const auto fixed = node.catalog().OccupancyOf(node.domain()).Total().value();
  ASSERT_TRUE(node.Start(Bytes(fixed + 6 * en::kPagedExtent)));

  EXPECT_EQ(node.memory_mode(), en::MemoryMode::kDedicated);
  EXPECT_EQ(node.ReserveTarget(), 6U);
  EXPECT_EQ(node.ReserveAvailable(), 6U);
  EXPECT_EQ(node.CreatedBackings(), initial_created + 6U);
  EXPECT_EQ(node.ReleasedBackings(), 0U);
  EXPECT_EQ(node.LiveBackings(), initial_created + 6U);

  // In dedicated mode, the dedicated pool is preserved: ReleaseUnusedBacking
  // only drains excess handles above the reserve target.
  const std::size_t drained = node.ReleaseUnusedBacking();
  EXPECT_EQ(drained, 0U);
  EXPECT_EQ(node.ReserveAvailable(), 6U);
  EXPECT_EQ(node.LiveBackings(), initial_created + 6U);
  EXPECT_EQ(node.ReleasedBackings(), 0U);

  EXPECT_TRUE(node.TearDown({}));
  EXPECT_EQ(node.LiveBackings(), 0U);
}

TEST(PagedNodeModes, ReleaseUnusedBackingSafeWithRunningWorkerThreads) {
  en::NodeSettings settings{.compute_streams = 1, .slots = 1, .slot_bytes = en::kPagedExtent};
  settings.memory_mode = en::MemoryMode::kDedicated;
  settings.preallocate_pool = true;

  en::PagedNode node(settings);
  ASSERT_TRUE(node.Open());
  const auto fixed = node.catalog().OccupancyOf(node.domain()).Total().value();
  ASSERT_TRUE(node.Start(Bytes(fixed + 4 * en::kPagedExtent)));
  node.Run();  // Worker threads (including backing_lane_) are now running!

  // Calling ReleaseUnusedBacking concurrently with active worker threads must not race
  // or violate VmmProvider single-threaded ownership.
  EXPECT_EQ(node.ReleaseUnusedBacking(), 0U);
  EXPECT_EQ(node.ReserveAvailable(), 4U);

  EXPECT_TRUE(node.TearDown({}));
  EXPECT_EQ(node.LiveBackings(), 0U);
}

TEST(PagedNodeModes, SharedMachineDrainExcessWithRunningWorkerThreads) {
  en::NodeSettings settings{.compute_streams = 1, .slots = 1, .slot_bytes = en::kPagedExtent};
  settings.memory_mode = en::MemoryMode::kShared;
  settings.preallocate_pool = false;

  en::PagedNode node(settings);
  ASSERT_TRUE(node.Open());
  const auto fixed = node.catalog().OccupancyOf(node.domain()).Total().value();
  ASSERT_TRUE(node.Start(Bytes(fixed + 4 * en::kPagedExtent)));
  node.Run();  // Worker threads running

  // In shared mode, reserve target is 0. Calling ReleaseUnusedBacking drains all handles.
  EXPECT_EQ(node.ReleaseUnusedBacking(), 0U);
  EXPECT_EQ(node.ReserveAvailable(), 0U);

  EXPECT_TRUE(node.TearDown({}));
  EXPECT_EQ(node.LiveBackings(), 0U);
}

TEST(PagedNodeModes, SharedMachineDemandAllocationDoesNotPreallocate) {
  en::NodeSettings settings{.compute_streams = 1, .slots = 1, .slot_bytes = en::kPagedExtent};
  settings.memory_mode = en::MemoryMode::kShared;
  settings.preallocate_pool = false;

  en::PagedNode node(settings);
  ASSERT_TRUE(node.Open());
  const auto initial_created = node.CreatedBackings();
  const auto fixed = node.catalog().OccupancyOf(node.domain()).Total().value();
  ASSERT_TRUE(node.Start(Bytes(fixed + 6 * en::kPagedExtent)));

  EXPECT_EQ(node.memory_mode(), en::MemoryMode::kShared);
  EXPECT_EQ(node.ReserveTarget(), 0U);
  EXPECT_EQ(node.ReserveAvailable(), 0U);
  EXPECT_EQ(node.CreatedBackings(), initial_created);
  EXPECT_EQ(node.LiveBackings(), initial_created);
  EXPECT_EQ(node.ReleasedBackings(), 0U);

  EXPECT_TRUE(node.TearDown({}));
  EXPECT_EQ(node.LiveBackings(), 0U);
}

TEST(PagedNodeModes, DynamicBudgetAdjustments) {
  en::NodeSettings settings{.compute_streams = 1, .slots = 1, .slot_bytes = en::kPagedExtent};
  en::PagedNode node(settings);
  ASSERT_TRUE(node.Open());
  const auto fixed = node.catalog().OccupancyOf(node.domain()).Total().value();
  const Bytes orig_budget(fixed + 10 * en::kPagedExtent);
  ASSERT_TRUE(node.Start(orig_budget));
  EXPECT_EQ(node.budget(), orig_budget);

  // External memory pressure: reduce budget.
  const Bytes trimmed_budget(fixed + 4 * en::kPagedExtent);
  node.SetBudget(trimmed_budget);
  EXPECT_EQ(node.budget(), trimmed_budget);

  // Pressure subsides: restore budget.
  node.SetBudget(orig_budget);
  EXPECT_EQ(node.budget(), orig_budget);

  EXPECT_TRUE(node.TearDown({}));
}

TEST(PressureTrimModes, ObserveAndHysteresisRecovery) {
  rt::PressureTrim trim;
  rt::PressureTrim::Clock::time_point now{};
  constexpr auto kLook = std::chrono::milliseconds(250);

  // Normal conditions: plenty of memory available (4 GiB).
  EXPECT_FALSE(trim.Observe({.available = 4096 * (1ULL << 20U), .stall = {}}, now).now);
  EXPECT_FALSE(trim.pressed());

  // Severe memory pressure occurs (available drops to 300 MiB).
  now += kLook;
  auto t = trim.Observe({.available = 300 * (1ULL << 20U), .stall = {}}, now);
  ASSERT_TRUE(t.now);
  EXPECT_TRUE(trim.pressed());
  EXPECT_GT(t.needed, 0U);

  // Trimming executed.
  EXPECT_FALSE(trim.Trimmed(t.needed, t.needed, now));

  // Memory rises above the low mark (512 MiB) to 900 MiB, but below target (1536 MiB).
  // Remains pressed (hysteresis).
  now += kLook;
  EXPECT_FALSE(trim.Observe({.available = 900 * (1ULL << 20U), .stall = {}}, now).now);
  EXPECT_TRUE(trim.pressed());

  // Memory fully recovers to 1600 MiB (above 1536 MiB target headroom).
  now += kLook;
  EXPECT_FALSE(trim.Observe({.available = 1600 * (1ULL << 20U), .stall = {}}, now).now);
  EXPECT_FALSE(trim.pressed());
}

TEST(BackingServicePool, DrainReserveOnRunningServiceDoesNotEmitZeroOperationCompletion) {
  pr::fake::FakeDeviceMemory memory(2_MiB, 32_MiB);
  llmp::base::WakeFlag wake;
  sc::CompletionBoard board(64, wake);
  sc::BackingService service(&memory, board, {.capacity = 64, .reserved = 16, .batch = 8});

  ASSERT_TRUE(service.SetReserve(0, 2_MiB, 4, /*preallocate_now=*/true).has_value());
  EXPECT_EQ(service.ReserveAvailable(), 4U);

  std::thread worker([&] { service.Run(); });
  EXPECT_EQ(service.DrainReserve(1), 3U);
  EXPECT_EQ(service.ReserveAvailable(), 1U);
  service.Close();
  worker.join();

  EXPECT_EQ(board.publications(), 0U);
  EXPECT_TRUE(board.Harvest(16).empty());
  EXPECT_EQ(service.DrainReserve(0), 1U);
  EXPECT_EQ(service.ReserveAvailable(), 0U);
}

}  // namespace
