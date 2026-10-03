// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// A runner's plan and graph caps (engine/planned.h), host-only: a cache
// keeps its least recently used shapes out (a Find counts as a use), counts
// the host bytes each plan was charged, several caches share one plan cap
// (each request slot's chunk plans) and one graph cap (every graph a model
// keeps, its drafter's and its waves' too), each dropping the least
// recently used across them.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <utility>
#include <vector>

#include "engine/graph_runs.h"
#include "engine/planned.h"
#include "ggml.h"
#include "kernels/ggml/tensors.h"

namespace {

namespace en = jitllm::engine;

using Cache = en::PlanCache<int, int>;

std::unique_ptr<int> Plan(int v) { return std::make_unique<int>(v); }

TEST(PlanCacheTest, KeepsTheMostRecentlyUsedAndCountsTheirBytes) {
  Cache cache(3);
  cache.Add(1, Plan(1), 100);
  cache.Add(2, Plan(2), 200);
  cache.Add(3, Plan(3), 300);
  EXPECT_EQ(cache.host_bytes(), 600U);
  // A use of the oldest keeps it: the next Add drops the least recently used.
  ASSERT_NE(cache.Find(1), nullptr);
  cache.Add(4, Plan(4), 400);
  EXPECT_EQ(cache.size(), 3U);
  EXPECT_NE(cache.Find(1), nullptr);
  EXPECT_EQ(cache.Find(2), nullptr);
  EXPECT_EQ(cache.host_bytes(), 100U + 300U + 400U);
  // The least recently used is now 3 (1 and 4 were used since).
  const std::uint64_t oldest = cache.OldestUse();
  cache.DropOldest();
  EXPECT_EQ(cache.Find(3), nullptr);
  EXPECT_GT(cache.OldestUse(), oldest);
  cache.Clear();
  EXPECT_EQ(cache.size(), 0U);
  EXPECT_EQ(cache.host_bytes(), 0U);
  EXPECT_EQ(cache.OldestUse(), Cache::kNever);
  cache.DropOldest();  // nothing to drop
  EXPECT_EQ(cache.graphs(), 0U);
  EXPECT_FALSE(cache.DropOldestGraph());
  EXPECT_EQ(cache.OldestGraphUse(), Cache::kNever);
}

TEST(PlanCacheTest, SlotsShareOnePlanCapDroppingTheLeastRecentlyUsedOfAny) {
  // Four slots' chunk plans, each cache able to hold the whole cap alone.
  std::array<Cache, 4> slots{Cache(4), Cache(4), Cache(4), Cache(4)};
  std::array<Cache*, 4> caches{slots.data(), &slots[1], &slots[2], &slots[3]};
  const std::span<Cache* const> all(caches);
  const auto held = [&]() {
    std::size_t n = 0;
    for (const Cache& c : slots) {
      n += c.size();
    }
    return n;
  };
  for (int i = 0; i < 4; ++i) {
    en::RoomForPlan(4, all);
    slots[static_cast<std::size_t>(i)].Add(i, Plan(i));
  }
  EXPECT_EQ(held(), 4U);
  // Slot 0's plan was used last: slot 1's goes for slot 2's next shape.
  ASSERT_NE(slots[0].Find(0), nullptr);
  en::RoomForPlan(4, all);
  EXPECT_EQ(held(), 3U);
  EXPECT_EQ(slots[1].size(), 0U);
  slots[2].Add(10, Plan(10));
  EXPECT_EQ(held(), 4U);
  EXPECT_EQ(slots[2].size(), 2U);
  // Below the cap nothing goes.
  slots[3].DropOldest();
  en::RoomForPlan(4, all);
  EXPECT_EQ(held(), 3U);
  // A cap of zero empties them all; empty caches end the loop.
  en::RoomForPlan(0, all);
  EXPECT_EQ(held(), 0U);
  en::RoomForPlan(0, all);
}

// A cache as RoomForGraphs sees it: graphs, each with its plan's last use
// and the bytes counted for it.
struct FakeGraphs {
  std::vector<std::pair<std::uint64_t, std::uint64_t>> graphs_;  // use, bytes
  std::size_t graphs() const { return graphs_.size(); }
  std::uint64_t graph_bytes() const {
    std::uint64_t n = 0;
    for (const auto& g : graphs_) {
      n += g.second;
    }
    return n;
  }
  std::uint64_t OldestGraphUse() const {
    std::uint64_t oldest = std::numeric_limits<std::uint64_t>::max();
    for (const auto& g : graphs_) {
      oldest = std::min(oldest, g.first);
    }
    return oldest;
  }
  bool DropOldestGraph() {
    if (graphs_.empty()) {
      return false;
    }
    graphs_.erase(std::ranges::min_element(graphs_));
    return true;
  }
};

constexpr std::uint64_t kNoBudget = std::numeric_limits<std::uint64_t>::max();

TEST(PlanCacheTest, EveryGraphCountsTowardOneCapDroppingTheLeastRecentlyUsed) {
  // A wave cache, and two slots' draft caches (whose graphs count too).
  FakeGraphs waves{.graphs_ = {{5, 0}, {9, 0}}};
  FakeGraphs draft_a{.graphs_ = {{3, 0}}};
  FakeGraphs draft_b{.graphs_ = {{7, 0}}};
  en::GraphStats stats;
  // Four kept, room for two captures under a cap of five: one goes, the
  // least recently used of all (draft A's).
  en::RoomForGraphs(5, kNoBudget, 2, 0, stats, waves, draft_a, draft_b);
  EXPECT_EQ(waves.graphs() + draft_a.graphs() + draft_b.graphs(), 3U);
  EXPECT_EQ(draft_a.graphs(), 0U);
  EXPECT_EQ(stats.dropped, 1U);
  // Room for three under a cap of four: two more go (5, then 7).
  en::RoomForGraphs(4, kNoBudget, 3, 0, stats, waves, draft_a, draft_b);
  ASSERT_EQ(waves.graphs(), 1U);
  EXPECT_EQ(waves.graphs_.front().first, 9U);
  EXPECT_EQ(draft_b.graphs(), 0U);
  EXPECT_EQ(stats.dropped, 3U);
  // Already within the cap: nothing goes. More captures than the cap are
  // refused, dropping nothing; one more past a full cap drops the last.
  en::RoomForGraph(2, stats, waves, draft_a, draft_b);
  EXPECT_EQ(stats.dropped, 3U);
  EXPECT_FALSE(en::RoomForGraphs(1, kNoBudget, 4, 0, stats, waves, draft_a, draft_b));
  EXPECT_EQ(waves.graphs(), 1U);
  EXPECT_EQ(stats.dropped, 3U);
  EXPECT_TRUE(en::RoomForGraphs(1, kNoBudget, 1, 0, stats, waves, draft_a, draft_b));
  EXPECT_EQ(waves.graphs(), 0U);
  EXPECT_EQ(stats.dropped, 4U);
}

TEST(PlanCacheTest, GraphsStayWithinTheirByteBudget) {
  FakeGraphs waves{.graphs_ = {{2, 300}, {6, 300}}};
  FakeGraphs chunks{.graphs_ = {{4, 100}, {8, 100}}};
  en::GraphStats stats;
  // 800 kept; a 250-byte capture under a 1,000-byte budget: the least
  // recently used goes (300 at use 2), then 500 + 250 fits.
  EXPECT_TRUE(en::RoomForGraphs(16, 1000, 1, 250, stats, waves, chunks));
  EXPECT_EQ(waves.graph_bytes() + chunks.graph_bytes(), 500U);
  EXPECT_EQ(stats.dropped, 1U);
  // Within both caps nothing goes.
  EXPECT_TRUE(en::RoomForGraphs(16, 1000, 1, 500, stats, waves, chunks));
  EXPECT_EQ(stats.dropped, 1U);
  // A capture alone past the budget is refused, dropping nothing: the kept
  // graphs never hold more than the budget.
  EXPECT_FALSE(en::RoomForGraphs(16, 1000, 1, 1200, stats, waves, chunks));
  EXPECT_EQ(waves.graph_bytes() + chunks.graph_bytes(), 500U);
  EXPECT_EQ(stats.dropped, 1U);
}

TEST(PlanCacheTest, APlansHostBytesAreItsArenaAndItsLaunchedNodes) {
  en::PlannedBase empty;
  EXPECT_EQ(en::PlannedHostBytes(empty), 0U);
  EXPECT_EQ(en::PlannedNodes(empty), 0U);
  auto arena = jitllm::kernels::ggml::TensorArena::Create(16);
  ASSERT_TRUE(arena.has_value());
  const std::uint64_t arena_bytes = arena->bytes();
  EXPECT_GE(arena_bytes, 16U * sizeof(ggml_tensor));
  en::PlannedBase planned;
  planned.arena.emplace(std::move(*arena));
  planned.plan.steps.emplace_back().nodes = {nullptr, nullptr};
  planned.plan.steps.emplace_back().nodes = {nullptr};
  EXPECT_EQ(en::PlannedNodes(planned), 3U);
  EXPECT_EQ(en::PlannedHostBytes(planned), arena_bytes + (3 * en::kPlanNodeHostBytes));
  // Uses are ordered across caches and never repeat.
  const std::uint64_t a = en::NextPlanUse();
  EXPECT_GT(en::NextPlanUse(), a);
}

}  // namespace
