// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// A runner's plans and graphs (engine/plan_cache.h), host-only: a cache
// keeps every shape (no fixed number) with its measured planning time,
// finds each by its key's hash, charges each plan's bytes to its account
// as it is added (and a graph's before its capture, refused when the
// account refuses it), gives each back as it goes, and offers its plans and
// graphs to the node's reclaim order (memory/reclaim.h) as candidates,
// least recently used ones first, never one a step under way holds
// (PlanStep). Entries never move. And the prefill lookahead's plans
// (engine/prefill_lookahead.h), funded under their own grant.

#include "engine/plan_cache.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <format>
#include <limits>
#include <memory>
#include <set>
#include <span>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "engine/prefill_lookahead.h"
#include "engine/support.h"
#include "execution/step_shape.h"
#include "memory/reclaim.h"

namespace {

namespace en = llmp::engine;
using llmp::memory::ReclaimCandidate;
using llmp::memory::ReclaimKind;

// A plan's runs as a cache holds them: none here captures a graph.
struct TestRuns {
  double seconds = 0;
  std::uint64_t measured_bytes = 0;
  bool has_graph() const { return false; }
  void DropGraph() {}
};
template <typename Planned, std::size_t Variants = 1>
using PlanCache = en::BasicPlanCache<int, Planned, Variants, TestRuns>;
using Cache = PlanCache<int>;

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

// Records whether a future plan was destroyed while its host bytes were
// still funded. Real plans own arenas/vectors whose teardown has the same
// order. Its host bytes grow with its nodes (a 2-node plan is past a
// 512-byte grant).
struct FuturePlan {
  Ledger& ledger;
  std::uint64_t minimum;
  int& destroyed;
  std::uint64_t bytes;
  FuturePlan(Ledger& l, std::uint64_t m, int& d, std::size_t nodes = 1)
      : ledger(l), minimum(m), destroyed(d), bytes(300 * nodes) {}
  FuturePlan(const FuturePlan&) = delete;
  FuturePlan& operator=(const FuturePlan&) = delete;
  FuturePlan(FuturePlan&&) = delete;
  FuturePlan& operator=(FuturePlan&&) = delete;
  ~FuturePlan() {
    EXPECT_GE(ledger.charged, minimum);
    ++destroyed;
  }
};
// The lookahead's measure of a plan (found by argument-dependent lookup).
std::uint64_t PlannedHostBytes(const FuturePlan& plan) { return plan.bytes; }
using Future = en::PrefillLookahead<FuturePlan>;
using FutureResult = std::expected<std::unique_ptr<FuturePlan>, std::string>;

TEST(PrefillPredictionTest, BoundsCurrentNearFarAndPreservesIntentWithoutPublishingWork) {
  const auto stages = en::PredictPrefill(
      256, 128, 1024, 512,
      {.rows = 256, .want_head = false, .after_rows = 3, .after_want_head = true});
  EXPECT_EQ(stages[0].first, 384U);
  EXPECT_EQ(stages[0].end(), 640U);
  EXPECT_FALSE(stages[0].want_head);
  EXPECT_EQ(stages[1].first, 640U);
  EXPECT_EQ(stages[1].end(), 643U);
  EXPECT_TRUE(stages[1].want_head);
  const auto vanished = en::PredictPrefill(0, 128, 1024, 512, {.after_rows = 256});
  EXPECT_EQ(vanished[0].rows, 0U);
  EXPECT_EQ(vanished[1].rows, 0U);
  for (const auto hint : {en::PrefillHint{.rows = 513, .after_rows = 1},
                          en::PrefillHint{.rows = 512, .after_rows = 512}}) {
    const auto bad = en::PredictPrefill(512, 256, 1024, 512, hint);
    EXPECT_EQ(bad[0].rows, 0U);
    EXPECT_EQ(bad[1].rows, 0U);
  }
  const auto tail = en::PredictPrefill(0, 256, 1024, 512, {.rows = 512, .after_rows = 257});
  EXPECT_EQ(tail[0].rows, 512U);
  EXPECT_EQ(tail[1].rows, 0U);
  const auto limit = std::numeric_limits<std::uint32_t>::max();
  const auto overflow = en::PredictPrefill(limit - 7, 8, limit, 512, {.rows = 1, .after_rows = 1});
  EXPECT_EQ(overflow[0].rows, 0U);
  const auto last = en::PredictPrefill(limit - 7, 1, limit, 512, {.rows = 6, .after_rows = 1});
  EXPECT_EQ(last[0].end(), limit);
  EXPECT_EQ(last[1].rows, 0U);
}

TEST(PrefillLookaheadTest, OptionalRefusalDoesNotBuildOrInstall) {
  Ledger ledger;
  ledger.budget = 0;
  Future future(
      1024, [&](auto bytes, bool required) { return ledger.account.Charge(bytes, required); },
      [&](auto bytes) { ledger.account.Uncharge(bytes); });
  EXPECT_FALSE(future.Fund());
  EXPECT_FALSE(future.Fund());
  EXPECT_FALSE(future.Build([]() -> FutureResult {
    ADD_FAILURE() << "refused lookahead built a plan";
    return std::unexpected("not built");
  }));
  EXPECT_FALSE(future.InstallAfterCompletion([](auto, double, const auto&) {
    ADD_FAILURE() << "refused lookahead installed a plan";
    return true;
  }));
  ASSERT_EQ(ledger.asked.size(), 1U);
  EXPECT_FALSE(ledger.asked.front().second);
  EXPECT_EQ(ledger.charged, 0U);
}

TEST(PrefillLookaheadTest, FailedAndOversizedBuildsStayUnderTheirGrant) {
  Ledger ledger;
  int destroyed = 0;
  {
    Future future(
        512, [&](auto bytes, bool required) { return ledger.account.Charge(bytes, required); },
        [&](auto bytes) { ledger.account.Uncharge(bytes); });
    ASSERT_TRUE(future.Fund());
    EXPECT_FALSE(future.Build(
        [&]() -> FutureResult { return std::make_unique<FuturePlan>(ledger, 512, destroyed, 2); }));
    EXPECT_EQ(destroyed, 1);
    EXPECT_EQ(ledger.charged, 512U);
    EXPECT_FALSE(future.Build([]() -> FutureResult {
      ADD_FAILURE() << "lookahead built twice";
      return std::unexpected("not built");
    }));
  }
  EXPECT_EQ(ledger.charged, 0U);
  {
    Future future(
        512, [&](auto bytes, bool required) { return ledger.account.Charge(bytes, required); },
        [&](auto bytes) { ledger.account.Uncharge(bytes); });
    ASSERT_TRUE(future.Fund());
    EXPECT_FALSE(future.Build([]() -> FutureResult { return std::unexpected("builder refused"); }));
    EXPECT_EQ(ledger.charged, 512U);
  }
  EXPECT_EQ(ledger.charged, 0U);
}

TEST(PrefillLookaheadTest, AbandonedOrFailedCurrentUnitDestroysFutureBeforeReleasingGrant) {
  Ledger ledger;
  int destroyed = 0;
  for (bool explicit_abandon : {false, true}) {
    Future future(
        1024, [&](auto bytes, bool required) { return ledger.account.Charge(bytes, required); },
        [&](auto bytes) { ledger.account.Uncharge(bytes); });
    ASSERT_TRUE(future.Fund());
    ASSERT_TRUE(future.Build(
        [&]() -> FutureResult { return std::make_unique<FuturePlan>(ledger, 1024, destroyed); }));
    // A failed/cancelled current unit returns without calling the installer.
    if (explicit_abandon) {
      future.Abandon();
      EXPECT_EQ(ledger.charged, 0U);
      EXPECT_FALSE(future.Fund());
      EXPECT_FALSE(future.InstallAfterCompletion([](auto, double, const auto&) {
        ADD_FAILURE() << "abandoned lookahead installed a plan";
        return true;
      }));
    }
  }
  EXPECT_EQ(destroyed, 2);
  EXPECT_EQ(ledger.charged, 0U);
}

TEST(PrefillLookaheadTest, InstallationFailureDestroysPlanBeforeReleasingGrant) {
  Ledger ledger;
  int destroyed = 0;
  {
    Future future(
        1024, [&](auto bytes, bool required) { return ledger.account.Charge(bytes, required); },
        [&](auto bytes) { ledger.account.Uncharge(bytes); });
    ASSERT_TRUE(future.Fund());
    ASSERT_TRUE(future.Build(
        [&]() -> FutureResult { return std::make_unique<FuturePlan>(ledger, 1024, destroyed); }));
    EXPECT_FALSE(future.InstallAfterCompletion([&](auto planned, double seconds, const auto&) {
      EXPECT_NE(planned, nullptr);
      EXPECT_GE(seconds, 0.0);
      EXPECT_EQ(ledger.charged, 1024U);
      return false;  // binding/coverage refusal: no transfer or cache insertion
    }));
    EXPECT_EQ(destroyed, 1);
    EXPECT_EQ(ledger.charged, 1024U);
  }
  EXPECT_EQ(ledger.charged, 0U);
}

TEST(PrefillLookaheadTest, TransfersOnceToCacheWhileCurrentPlanAndCaptureStayProtected) {
  Ledger ledger;
  int destroyed = 0;
  Cache current(&ledger.account);
  auto& entry = current.Add(1, Plan(1), 100, 1);
  PlanCache<FuturePlan> next(&ledger.account);
  {
    const en::PlanStep step;
    ASSERT_EQ(current.Find(1), &entry);
    ASSERT_TRUE(current.ChargeGraph(entry));
    const auto held_bytes = ledger.charged;
    // Optional funding may invoke reclaim; neither the current plan nor its
    // charged capture may be taken while the step borrows them.
    Future future(
        1024,
        [&](auto bytes, bool required) {
          EXPECT_FALSE(required);
          EXPECT_EQ(current.Reclaim(ReclaimKind::kPlan, entry.serial), 0U);
          EXPECT_EQ(current.Reclaim(ReclaimKind::kGraph, entry.serial), 0U);
          return ledger.account.Charge(bytes, required);
        },
        [&](auto bytes) { ledger.account.Uncharge(bytes); });
    ASSERT_TRUE(future.Fund());
    ASSERT_TRUE(future.Build(
        [&]() -> FutureResult { return std::make_unique<FuturePlan>(ledger, 512, destroyed); }));
    EXPECT_TRUE(
        future.InstallAfterCompletion([&](auto planned, double seconds, const auto& transfer) {
          EXPECT_EQ(ledger.charged, held_bytes + 1024);
          transfer();
          EXPECT_EQ(ledger.charged, held_bytes);
          next.Add(2, std::move(planned), 512, 1, seconds);
          EXPECT_EQ(ledger.charged, held_bytes + 512);
          return true;
        }));
    EXPECT_FALSE(future.InstallAfterCompletion([](auto, double, const auto&) {
      ADD_FAILURE() << "lookahead installed twice";
      return true;
    }));
    ASSERT_EQ(current.Find(1), &entry);
    EXPECT_EQ(current.Reclaim(ReclaimKind::kPlan, entry.serial), 0U);
  }
  next.Clear();
  current.Clear();
  EXPECT_EQ(destroyed, 1);
  EXPECT_EQ(ledger.charged, 0U);
}

using FutureGroup = en::PrefillLookaheadGroup<FuturePlan, 2>;

TEST(PrefillLookaheadGroupTest, RangeAndSecondGrantRefusalLeaveTheFirstInstallable) {
  Ledger ledger;
  ledger.budget = 1024;
  int destroyed = 0;
  {
    FutureGroup futures(
        1024, [&](auto bytes, bool required) { return ledger.account.Charge(bytes, required); },
        [&](auto bytes) { ledger.account.Uncharge(bytes); });
    EXPECT_FALSE(futures.Fund(2));
    EXPECT_EQ(futures.seconds(2), 0);
    EXPECT_TRUE(ledger.asked.empty());
    ASSERT_TRUE(futures.Fund(0));
    EXPECT_FALSE(futures.Fund(1));
    EXPECT_FALSE(futures.Fund(0));
    const auto built = futures.BuildAll([&](std::size_t i) -> FutureResult {
      EXPECT_EQ(i, 0U);
      return std::make_unique<FuturePlan>(ledger, 1024, destroyed);
    });
    EXPECT_EQ(built, (std::array{true, false}));
    const auto installed = futures.InstallAfterCompletion(
        [&](std::size_t i, auto planned, double seconds, const auto&) {
          EXPECT_EQ(i, 0U);
          EXPECT_GE(seconds, 0);
          EXPECT_NE(planned, nullptr);
          return true;
        });
    EXPECT_EQ(installed, (std::array{true, false}));
    EXPECT_EQ(destroyed, 1);
    EXPECT_EQ(futures.InstallAfterCompletion([](auto, auto, double, const auto&) {
      ADD_FAILURE() << "group installed twice";
      return true;
    }),
              (std::array{false, false}));
  }
  EXPECT_EQ(ledger.charged, 0U);
}

TEST(PrefillLookaheadGroupTest, SecondBuildRefusalAndSecondInstallRefusalAreIndependent) {
  for (const bool refuse_build : {true, false}) {
    Ledger ledger;
    int destroyed = 0;
    {
      FutureGroup futures(
          1024, [&](auto bytes, bool required) { return ledger.account.Charge(bytes, required); },
          [&](auto bytes) { ledger.account.Uncharge(bytes); });
      ASSERT_TRUE(futures.Fund(0));
      ASSERT_TRUE(futures.Fund(1));
      std::vector<std::size_t> order;
      const auto built = futures.BuildAll([&](std::size_t i) -> FutureResult {
        order.push_back(i);
        if (i == 1 && refuse_build) return std::unexpected("far builder refused");
        return std::make_unique<FuturePlan>(ledger, 1024, destroyed);
      });
      EXPECT_EQ(order, (std::vector<std::size_t>{0, 1}));
      EXPECT_EQ(built, (std::array{true, !refuse_build}));
      order.clear();
      const auto installed =
          futures.InstallAfterCompletion([&](std::size_t i, auto planned, double, const auto&) {
            order.push_back(i);
            EXPECT_NE(planned, nullptr);
            return i == 0;  // independent far binding/coverage refusal
          });
      EXPECT_EQ(installed, (std::array{true, false}));
      EXPECT_EQ(order,
                refuse_build ? (std::vector<std::size_t>{0}) : (std::vector<std::size_t>{0, 1}));
    }
    EXPECT_EQ(destroyed, refuse_build ? 1 : 2);
    EXPECT_EQ(ledger.charged, 0U);
  }
}

TEST(PrefillLookaheadGroupTest, FailedCurrentUnitDestroysBothPlansUnderTheirOwnGrants) {
  for (const bool explicit_abandon : {false, true}) {
    Ledger ledger;
    int destroyed = 0;
    {
      FutureGroup futures(
          1024, [&](auto bytes, bool required) { return ledger.account.Charge(bytes, required); },
          [&](auto bytes) { ledger.account.Uncharge(bytes); });
      ASSERT_TRUE(futures.Fund(0));
      ASSERT_TRUE(futures.Fund(1));
      EXPECT_EQ(futures.BuildAll([&](std::size_t) -> FutureResult {
        return std::make_unique<FuturePlan>(ledger, 1024, destroyed);
      }),
                (std::array{true, true}));
      if (explicit_abandon) {
        futures.Abandon();
        EXPECT_EQ(destroyed, 2);
        EXPECT_EQ(ledger.charged, 0U);
      }
    }
    EXPECT_EQ(destroyed, 2);
    EXPECT_EQ(ledger.charged, 0U);
  }
}

TEST(PrefillLookaheadGroupTest, OrderedCacheTransferProtectsTheFirstFromTheSecondCharge) {
  Ledger ledger;
  int destroyed = 0;
  PlanCache<FuturePlan> cache(&ledger.account);
  std::uint64_t first = 0;
  ledger.account.Bind(
      [&](auto bytes, bool required) {
        // The second Add can reclaim, but the first newly installed plan is
        // protected by this current PlanStep, without an extra Find.
        if (required && first != 0) EXPECT_EQ(cache.Reclaim(ReclaimKind::kPlan, first), 0U);
        ledger.charged += bytes;
        return true;
      },
      [&](auto bytes) { ledger.charged -= bytes; });
  {
    const en::PlanStep step;
    FutureGroup futures(
        1024, [&](auto bytes, bool required) { return ledger.account.Charge(bytes, required); },
        [&](auto bytes) { ledger.account.Uncharge(bytes); });
    ASSERT_TRUE(futures.Fund(0));
    ASSERT_TRUE(futures.Fund(1));
    EXPECT_EQ(futures.BuildAll([&](std::size_t) -> FutureResult {
      return std::make_unique<FuturePlan>(ledger, 512, destroyed);
    }),
              (std::array{true, true}));
    EXPECT_EQ(futures.InstallAfterCompletion([&](std::size_t i, auto planned, double seconds,
                                                 const auto& transfer) {
      transfer();
      auto& entry = cache.Add(static_cast<int>(i), std::move(planned), 512, 1, seconds);
      if (i == 0) first = entry.serial;
      return true;
    }),
              (std::array{true, true}));
    EXPECT_EQ(cache.size(), 2U);
    EXPECT_EQ(ledger.charged, 1024U);
  }
  cache.Clear();
  EXPECT_EQ(destroyed, 2);
  EXPECT_EQ(ledger.charged, 0U);
}

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

TEST(PlanCacheTest, PlanStepProtectsPlansAcrossThreads) {
  Ledger ledger;
  Cache cache(&ledger.account);
  cache.Add(1, Plan(1), 100, 0, 0.02);
  cache.Add(2, Plan(2), 300, 0, 0.05);

  std::atomic<bool> step_ready{false};
  std::atomic<bool> check_done{false};

  std::thread worker([&] {
    const en::PlanStep step;
    EXPECT_NE(en::PlanStepStart(), en::kNoStep);
    EXPECT_NE(cache.Find(1), nullptr);
    step_ready.store(true, std::memory_order_release);
    while (!check_done.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
  });

  while (!step_ready.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }

  // From a second thread, PlanStepStart() sees the worker's active step.
  EXPECT_NE(en::PlanStepStart(), en::kNoStep);

  // Collect from the second thread excludes plan 1 (held by worker's step).
  std::vector<ReclaimCandidate> candidates;
  cache.Collect(7, true, candidates);
  ASSERT_EQ(candidates.size(), 1U);
  EXPECT_EQ(candidates[0].bytes, 300U);

  // Reclaim from the second thread refuses plan 1 while worker holds its step.
  Cache::Entry* held = cache.Find(1);
  ASSERT_NE(held, nullptr);
  EXPECT_EQ(cache.Reclaim(ReclaimKind::kPlan, held->serial), 0U);

  check_done.store(true, std::memory_order_release);
  worker.join();

  EXPECT_EQ(en::PlanStepStart(), en::kNoStep);
}

TEST(PlanCacheTest, AReclaimFindsItsPlanAmongARunnersCaches) {
  Ledger ledger;
  Cache a(&ledger.account);
  PlanCache<int, 2> b(&ledger.account);
  a.Add(1, Plan(1), 10);
  // The serial, not the entry: reclaiming frees the entry.
  const auto serial = b.Add(1, Plan(2), 20).serial;
  std::array<en::PlanCacheBase*, 2> caches = {&a, &b};
  std::vector<ReclaimCandidate> candidates;
  en::CollectPlans(caches, 3, false, candidates);
  ASSERT_EQ(candidates.size(), 2U);
  EXPECT_EQ(en::ReclaimPlan(caches, ReclaimKind::kPlan, serial), 20U);
  EXPECT_EQ(b.size(), 0U);
  EXPECT_EQ(a.size(), 1U);
  EXPECT_EQ(en::ReclaimPlan(caches, ReclaimKind::kPlan, serial), 0U);
  EXPECT_EQ(ledger.charged, 10U);
}

// Plans are found by their keys' hashes: a key whose hash collides with
// another's is still told apart by the whole key, and a reclaimed plan's
// key finds nothing.
struct Collide {
  std::size_t operator()(int) const { return 7; }
};
TEST(PlanCacheTest, AHashCollisionIsToldApartByTheWholeKey) {
  Ledger ledger;
  en::BasicPlanCache<int, int, 1, TestRuns, Collide> cache(&ledger.account);
  for (int i = 0; i < 16; ++i) cache.Add(i, Plan(i * 10), 1);
  for (int i = 0; i < 16; ++i) {
    auto* found = cache.Find(i);
    ASSERT_NE(found, nullptr) << i;
    EXPECT_EQ(*found->planned, i * 10);
  }
  EXPECT_EQ(cache.Find(16), nullptr);
  const std::uint64_t serial = cache.Find(5)->serial;
  EXPECT_EQ(cache.Reclaim(ReclaimKind::kPlan, serial), 1U);
  EXPECT_EQ(cache.Find(5), nullptr);
  EXPECT_NE(cache.Find(6), nullptr);
  cache.Clear();
  EXPECT_EQ(cache.Find(6), nullptr);
}

// Finding a plan must not inspect every unrelated plan's graph ownership:
// a hash index alone would still leave linear lookup if it settled them all.
TEST(PlanCacheTest, LookupAndCaptureInspectOnlyTheirOwnGraphs) {
  struct CountedRuns {
    std::size_t* inspected = nullptr;
    double seconds = 0;
    std::uint64_t measured_bytes = 0;
    bool has_graph() const {
      if (inspected != nullptr) ++*inspected;
      return false;
    }
    void DropGraph() {}
  };
  Ledger ledger;
  en::BasicPlanCache<int, int, 1, CountedRuns> cache(&ledger.account);
  std::size_t inspected = 0;
  for (int i = 0; i < 100; ++i) {
    cache.Add(i, Plan(i), 1).runs[0].inspected = &inspected;
  }
  inspected = 0;
  EXPECT_EQ(cache.Find(100), nullptr);
  EXPECT_EQ(inspected, 0U);
  auto* found = cache.Find(50);
  ASSERT_NE(found, nullptr);
  EXPECT_EQ(inspected, 1U);
  EXPECT_TRUE(cache.ChargeGraph(*found));
  EXPECT_EQ(inspected, 2U);
}

// A step's plan key hashes what its equality compares and nothing else: a
// segment's `past` (positions, data to a program) changes neither, so a
// later step of the same shape finds its plan; a change of rows, cells,
// stage or parameters is another key.
TEST(PlanCacheTest, AStepsKeyFindsItsPlanWhateverItsPositions) {
  Ledger ledger;
  en::BasicPlanCache<llmp::planner::PlanKey, int, 1, TestRuns> cache(&ledger.account);
  llmp::planner::PlanKey key;
  key.spec[0] = 1;
  key.policy[3] = 9;
  key.shape.phase = llmp::planner::Phase::kDecode;
  key.shape.segments = {{.slot = 0, .rows = 1, .past = 100, .read_cells = 2048, .outputs = 1},
                        {.slot = 1, .rows = 1, .past = 7, .read_cells = 256, .outputs = 1}};
  key.parameters = {5, 6};
  cache.Add(key, Plan(42), 1);
  llmp::planner::PlanKey later = key;
  later.shape.segments[0].past = 101;
  later.shape.segments[1].past = 8;
  ASSERT_EQ(later, key);
  EXPECT_EQ(llmp::execution::HashOf(later), llmp::execution::HashOf(key));
  ASSERT_NE(cache.Find(later), nullptr);
  EXPECT_EQ(*cache.Find(later)->planned, 42);
  for (int change = 0; change < 5; ++change) {
    llmp::planner::PlanKey other = key;
    if (change == 0) other.shape.segments[1].rows = 2;
    if (change == 1) other.shape.segments[0].read_cells = 4096;
    if (change == 2) other.shape.stage = 1;
    if (change == 3) other.parameters.back() = 7;
    if (change == 4) other.patterns[0] = 1;
    EXPECT_EQ(cache.Find(other), nullptr) << change;
  }
}

// Every bounded ordered composition of owners' rows a shared-preparation
// measurement can select is measured, within the budget and the chunk.
TEST(ChunkMeasurementRows, EverySmallOwnerCompositionIsMeasuredAndKeepsBudgets) {
  for (const std::uint32_t max_rows : {1U, 3U, 128U}) {
    for (const std::uint32_t budget : {2U, 8U, 256U}) {
      for (const std::uint32_t owners : {1U, 2U, 3U, 4U, 8U, 12U}) {
        const auto ordinary = en::support::ChunkMeasurementRows(max_rows, budget, owners, false);
        const auto prepared = en::support::ChunkMeasurementRows(max_rows, budget, owners, true);
        for (const auto& row : ordinary)
          EXPECT_NE(std::ranges::find(prepared, row), prepared.end());
        for (const auto& row : prepared) {
          EXPECT_EQ(row.size(), owners);
          std::uint32_t total = 0;
          for (const auto n : row) {
            EXPECT_GE(n, 1U);
            EXPECT_LE(n, max_rows);
            total += n;
          }
          EXPECT_LE(total, budget);
        }
        // Independently enumerate every bounded ordered row composition.
        if (owners > budget || owners > 8) continue;
        std::vector<std::uint32_t> row(owners, 1);
        const auto check = [&](auto&& self, std::uint32_t owner, std::uint32_t total) -> void {
          if (owner == owners) {
            EXPECT_NE(std::ranges::find(prepared, row), prepared.end());
            return;
          }
          const auto left = std::min(8U, budget) - total;
          for (std::uint32_t n = 1; n <= std::min(max_rows, left); ++n) {
            if (total + n + owners - owner - 1 > std::min(8U, budget)) break;
            row[owner] = n;
            self(self, owner + 1, total + n);
          }
        };
        check(check, 0, 0);
      }
    }
  }
}

TEST(PlanCacheTest, RegistryBackendAndCapturabilityAreDistinctKeys) {
  llmp::execution::PlanKey key;
  std::array<llmp::execution::PlanKey, 4> keys{key, key, key, key};
  keys[1].registry[0] = 1;
  keys[2].backend[0] = 1;
  keys[3].capturable = true;
  for (std::size_t i = 0; i < keys.size(); ++i)
    for (std::size_t j = i + 1; j < keys.size(); ++j) {
      EXPECT_NE(keys[i], keys[j]);
      EXPECT_NE(llmp::execution::HashOf(keys[i]), llmp::execution::HashOf(keys[j]));
    }
}

// A step's read class (execution::ReadCellsFor), the one rule its readers
// share. Without buckets: its end at the alignment, at most the context
// (every earlier step's read).
TEST(ReadCellsForTest, WithoutBucketsIsTheAlignedEndWithinTheContext) {
  for (const std::uint32_t align : {256U, 512U, 2048U}) {
    for (const std::uint32_t context : {1000U, 4096U, 262144U}) {
      for (std::uint64_t end = 0; end <= context; end += 37) {
        const std::uint64_t aligned = (end + align - 1) / align * align;
        EXPECT_EQ(llmp::execution::ReadCellsFor(end, align, 0, context),
                  std::min<std::uint64_t>(aligned, context))
            << align << " " << context << " " << end;
      }
    }
  }
  EXPECT_EQ(llmp::execution::ReadCellsFor(0, 256, 4, 4096), 0U);
}

// With buckets: monotonic in the end, covering it, a multiple of the
// alignment within the context, idempotent, less than the larger of an
// alignment and end / buckets past the end, every alignment times a power
// of two a class boundary (a kernel's power-of-two cell limit is crossed
// only by an end past it), and the aligned classes through 2 * buckets
// alignments unchanged.
TEST(ReadCellsForTest, BucketsAreMonotonicCoveringBoundedAndKeepPowerOfTwoBoundaries) {
  constexpr std::uint32_t kContext = 262144;
  for (const std::uint32_t align : {256U, 2048U}) {
    for (const std::uint32_t buckets : {1U, 2U, 3U, 4U, 8U, 16U}) {
      SCOPED_TRACE(std::format("align {} buckets {}", align, buckets));
      std::uint32_t last = 0;
      for (std::uint64_t end = 0; end <= kContext; ++end) {
        const std::uint32_t read = llmp::execution::ReadCellsFor(end, align, buckets, kContext);
        ASSERT_GE(read, last) << end;
        ASSERT_GE(read, end);
        ASSERT_LE(read, kContext);
        ASSERT_EQ(read % align, 0U) << end;
        ASSERT_EQ(llmp::execution::ReadCellsFor(read, align, buckets, kContext), read) << end;
        ASSERT_LT(read - end, std::max<std::uint64_t>(align, end / buckets)) << end;
        if (end <= std::uint64_t{2} * buckets * align)
          ASSERT_EQ(read, llmp::execution::ReadCellsFor(end, align, 0, kContext)) << end;
        last = read;
      }
      for (std::uint64_t limit = align; limit <= kContext; limit *= 2)
        EXPECT_EQ(llmp::execution::ReadCellsFor(limit, align, buckets, kContext), limit);
    }
  }
  // Capped by a context that is not a class boundary.
  EXPECT_EQ(llmp::execution::ReadCellsFor(5000, 256, 1, 6000), 6000U);
}

// About 36 decode plans at a 262K context for four buckets of 256, against
// 1,024 aligned; the 2,048-aligned families' 128 become 24.
TEST(ReadCellsForTest, CountsTheClassesOfALongContext) {
  const auto classes = [](std::uint32_t align, std::uint32_t buckets, std::uint32_t context) {
    std::set<std::uint32_t> seen;
    for (std::uint64_t end = 1; end <= context; ++end)
      seen.insert(llmp::execution::ReadCellsFor(end, align, buckets, context));
    return seen.size();
  };
  EXPECT_EQ(classes(256, 0, 262144), 1024U);
  EXPECT_EQ(classes(256, 4, 262144), 36U);
  EXPECT_EQ(classes(256, 8, 262144), 64U);
  EXPECT_EQ(classes(256, 16, 262144), 112U);
  EXPECT_EQ(classes(256, 4, 131072), 32U);
  EXPECT_EQ(classes(2048, 0, 262144), 128U);
  EXPECT_EQ(classes(2048, 4, 262144), 24U);
}

}  // namespace
