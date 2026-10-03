// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// A runner's plans and graphs (engine/planned.h), host-only: a cache keeps
// every shape (no fixed number) with its measured planning time, charges
// each plan's bytes to its account as it is added (and a graph's before
// its capture, refused when the account refuses it), gives each back as it
// goes, and offers its plans and graphs to the node's reclaim order
// (memory/reclaim.h) as candidates, least recently used ones first, never
// one a step under way holds (PlanStep). Entries never move. A plan's
// arena is sized to what its graph uses (SizedArena).

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <utility>
#include <vector>

#include "engine/graph_runs.h"
#include "engine/planned.h"
#include "ggml.h"
#include "kernels/ggml/tensors.h"
#include "memory/reclaim.h"

namespace {

namespace en = jitllm::engine;
using jitllm::memory::ReclaimCandidate;
using jitllm::memory::ReclaimKind;

using Cache = en::PlanCache<int, int>;

std::unique_ptr<int> Plan(int v) { return std::make_unique<int>(v); }

// An account that records its charges and refuses past a budget.
struct Ledger {
  std::uint64_t budget = UINT64_MAX;
  std::uint64_t charged = 0;
  std::vector<std::pair<std::uint64_t, bool>> asked;  // bytes, required
  en::PlanAccount account;
  Ledger() {
    account.Bind(
        [this](std::uint64_t bytes, bool required) {
          asked.emplace_back(bytes, required);
          if (!required && charged + bytes > budget) {
            return false;
          }
          charged += bytes;
          return true;
        },
        [this](std::uint64_t bytes) { charged -= bytes; });
  }
};

TEST(PlanCacheTest, KeepsEveryShapeAndChargesWhatEachHolds) {
  Ledger ledger;
  Cache cache(&ledger.account);
  for (int i = 0; i < 100; ++i) {
    cache.Add(i, Plan(i), 100, 0, 0.01);
  }
  // No fixed number: every one is kept, every one charged, as required.
  EXPECT_EQ(cache.size(), 100U);
  EXPECT_EQ(cache.host_bytes(), 10000U);
  EXPECT_EQ(ledger.charged, 10000U);
  EXPECT_EQ(ledger.account.bytes(), 10000U);
  EXPECT_TRUE(ledger.asked.front().second);
  ASSERT_NE(cache.Find(42), nullptr);
  EXPECT_EQ(*cache.Find(42)->planned, 42);
  EXPECT_EQ(cache.Find(1000), nullptr);
  cache.Clear();
  EXPECT_EQ(cache.size(), 0U);
  EXPECT_EQ(ledger.charged, 0U);
  EXPECT_EQ(ledger.account.bytes(), 0U);
}

TEST(PlanCacheTest, AGraphIsChargedBeforeItsCaptureAndGivenBackWithoutOne) {
  Ledger ledger;
  Cache cache(&ledger.account);
  Cache::Entry& entry = cache.Add(1, Plan(1), 100, 10);
  // A graph of 10 launched nodes: refused past the budget (not required),
  // so it is not captured and nothing is charged.
  ledger.budget = 100 + (10 * en::kGraphNodeHostBytes) - 1;
  EXPECT_FALSE(cache.ChargeGraph(entry));
  EXPECT_FALSE(ledger.asked.back().second);
  EXPECT_EQ(ledger.charged, 100U);
  ledger.budget = UINT64_MAX;
  // A refused charge is not asked again at once: the next use skips it
  // without asking (1, then 2, 4 … uses after each refusal).
  const std::size_t asked_before = ledger.asked.size();
  EXPECT_FALSE(cache.ChargeGraph(entry));
  EXPECT_EQ(ledger.asked.size(), asked_before);
  {
    // Within the step that will capture it, the charge stays even with no
    // graph yet, and is made once.
    const en::PlanStep step;
    ASSERT_NE(cache.Find(1), nullptr);
    ASSERT_TRUE(cache.ChargeGraph(entry));
    EXPECT_EQ(ledger.charged, 100U + (10 * en::kGraphNodeHostBytes));
    const std::size_t asked = ledger.asked.size();
    EXPECT_TRUE(cache.ChargeGraph(entry));
    EXPECT_EQ(ledger.asked.size(), asked);
    ASSERT_NE(cache.Find(1), nullptr);
    EXPECT_EQ(ledger.charged, 100U + (10 * en::kGraphNodeHostBytes));
  }
  // The capture did not happen (no graph): once no step holds the plan,
  // its next use gives the charge back.
  ASSERT_NE(cache.Find(1), nullptr);
  EXPECT_EQ(ledger.charged, 100U);
  EXPECT_EQ(cache.graphs(), 0U);
  // A variant past the cache's is refused.
  EXPECT_FALSE(cache.ChargeGraph(entry, 1));
}

TEST(PlanCacheTest, ARefusedGraphChargeBacksOffDoublingToItsMost) {
  Ledger ledger;
  Cache cache(&ledger.account);
  Cache::Entry& entry = cache.Add(1, Plan(1), 100, 10);
  ledger.budget = 100;  // no room for any graph
  std::vector<std::size_t> asked_at;
  for (std::size_t use = 0; use < 1000; ++use) {
    const std::size_t before = ledger.asked.size();
    EXPECT_FALSE(cache.ChargeGraph(entry));
    if (ledger.asked.size() != before) {
      asked_at.push_back(use);
    }
  }
  // Asked at 0, then after 1, 2, 4 … 256 uses skipped, then every 257th.
  ASSERT_GE(asked_at.size(), 3U);
  EXPECT_EQ(asked_at[0], 0U);
  EXPECT_EQ(asked_at[1], 2U);
  EXPECT_EQ(asked_at[2], 5U);
  EXPECT_LE(asked_at.size(), 12U);
  // Once it fits, the wait resets.
  ledger.budget = UINT64_MAX;
  while (!cache.ChargeGraph(entry)) {
  }
  EXPECT_EQ(entry.graph_backoff, 0U);
}

TEST(PlanCacheTest, PlansAreCandidatesOnceNoStepHoldsThem) {
  Ledger ledger;
  Cache cache(&ledger.account);
  cache.Add(1, Plan(1), 100, 0, 0.02);
  cache.Add(2, Plan(2), 300, 0, 0.05);
  std::vector<ReclaimCandidate> candidates;
  cache.Collect(7, true, candidates);
  ASSERT_EQ(candidates.size(), 2U);  // plans only: neither has a graph
  EXPECT_EQ(candidates[0].kind, ReclaimKind::kPlan);
  EXPECT_EQ(candidates[0].owner, 7U);
  EXPECT_EQ(candidates[0].bytes, 100U);
  EXPECT_DOUBLE_EQ(candidates[0].restore_seconds, 0.02);
  EXPECT_TRUE(candidates[0].running);
  EXPECT_LT(candidates[0].last_use, candidates[1].last_use);
  {
    // A step that found plan 1 holds it: it is no candidate, and a reclaim
    // of it (from a charge inside the step) is refused.
    const en::PlanStep step;
    EXPECT_NE(en::PlanStepStart(), en::kNoStep);
    {
      const en::PlanStep nested;  // a nested step keeps the outer one's start
      ASSERT_NE(cache.Find(1), nullptr);
    }
    candidates.clear();
    cache.Collect(7, true, candidates);
    ASSERT_EQ(candidates.size(), 1U);
    EXPECT_EQ(candidates[0].bytes, 300U);
    Cache::Entry* held = cache.Find(1);
    ASSERT_NE(held, nullptr);
    EXPECT_EQ(cache.Reclaim(ReclaimKind::kPlan, held->serial), 0U);
    // The other goes; the held entry has not moved.
    EXPECT_EQ(cache.Reclaim(ReclaimKind::kPlan, candidates[0].id), 300U);
    EXPECT_EQ(cache.Find(1), held);
    EXPECT_EQ(cache.reclaimed_plans(), 1U);
  }
  EXPECT_EQ(en::PlanStepStart(), en::kNoStep);
  EXPECT_EQ(ledger.charged, 100U);
  // Between steps plan 1 is a candidate again; a graph reclaim of a plan
  // without one frees nothing; an unknown serial nothing.
  candidates.clear();
  cache.Collect(0, false, candidates);
  ASSERT_EQ(candidates.size(), 1U);
  EXPECT_EQ(cache.Reclaim(ReclaimKind::kGraph, candidates[0].id), 0U);
  EXPECT_EQ(cache.Reclaim(ReclaimKind::kPlan, 0), 0U);
  EXPECT_EQ(cache.Reclaim(ReclaimKind::kPlan, candidates[0].id), 100U);
  EXPECT_EQ(cache.size(), 0U);
  EXPECT_EQ(ledger.charged, 0U);
}

TEST(PlanCacheTest, AReclaimFindsItsPlanAmongARunnersCaches) {
  Ledger ledger;
  Cache a(&ledger.account);
  en::PlanCache<int, int, 2> b(&ledger.account);
  a.Add(1, Plan(1), 10);
  auto& kept = b.Add(1, Plan(2), 20);
  std::array<en::PlanCacheBase*, 2> caches = {&a, &b};
  std::vector<ReclaimCandidate> candidates;
  en::CollectPlans(caches, 3, false, candidates);
  ASSERT_EQ(candidates.size(), 2U);
  EXPECT_EQ(en::ReclaimPlan(caches, ReclaimKind::kPlan, kept.serial), 20U);
  EXPECT_EQ(b.size(), 0U);
  EXPECT_EQ(a.size(), 1U);
  EXPECT_EQ(en::ReclaimPlan(caches, ReclaimKind::kPlan, kept.serial), 0U);
  EXPECT_EQ(ledger.charged, 10U);
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

// A graph of `tensors` leaf tensors built on an arena (false: no room).
bool Build(jitllm::kernels::ggml::TensorArena& arena, int tensors) {
  for (int i = 0; i < tensors; ++i) {
    if (!arena.Reserve(1)) {
      return false;
    }
    (void)ggml_new_tensor_1d(arena.context(), GGML_TYPE_F32, 4);
  }
  return true;
}

TEST(PlanCacheTest, APlansArenaHoldsWhatItsGraphUsesNotTheEstimate) {
  auto sized =
      en::SizedArena(1000, [](jitllm::kernels::ggml::TensorArena& a) { return Build(a, 10); });
  ASSERT_TRUE(sized.has_value());
  // The estimate's 1,000 tensors are not kept: ten and one's slack.
  EXPECT_LE(sized->bytes(), 11 * ggml_tensor_overhead());
  EXPECT_EQ(sized->used(), 0U);
  // Its checks answer for the estimate the build was checked against, so
  // the same build passes them; sealed, it answers for its own room.
  EXPECT_TRUE(sized->Reserve(1000).has_value());
  EXPECT_TRUE(Build(*sized, 10));
  sized->Seal();
  EXPECT_TRUE(sized->Reserve(1).has_value());
  EXPECT_FALSE(sized->Reserve(2).has_value());
  // A build that fails keeps the estimate, for the caller's own error.
  auto failed =
      en::SizedArena(8, [](jitllm::kernels::ggml::TensorArena& a) { return Build(a, 10); });
  ASSERT_TRUE(failed.has_value());
  EXPECT_EQ(failed->bytes(), 8 * ggml_tensor_overhead());
}

}  // namespace
