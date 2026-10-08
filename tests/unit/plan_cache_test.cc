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
#include "engine/prefill_lookahead.h"
#include "ggml.h"
#include "kernels/ggml/graph_read_index.h"
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

// Records whether a future plan was destroyed while its host bytes were
// still funded. Real plans own arenas/vectors whose teardown has the same order.
struct FuturePlan : en::PlannedBase {
  Ledger& ledger;
  std::uint64_t minimum;
  int& destroyed;
  FuturePlan(Ledger& l, std::uint64_t m, int& d, std::size_t nodes = 1)
      : ledger(l), minimum(m), destroyed(d) {
    plan.steps.resize(1);
    plan.steps.front().nodes.resize(nodes);
  }
  ~FuturePlan() {
    EXPECT_GE(ledger.charged, minimum);
    ++destroyed;
  }
};
using Future = en::PrefillLookahead<FuturePlan>;
using FutureResult = std::expected<std::unique_ptr<FuturePlan>, std::string>;

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
  en::PlanCache<int, FuturePlan> next(&ledger.account);
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
  // Metadata keeps ten tensors and one slack slot. Traversal storage
  // retains its separate 1,000-descriptor bound and total host charge.
  const auto traversal_bytes = sized->graph_visited().size_bytes();
  EXPECT_EQ(sized->graph_capacity(), 1000U);
  EXPECT_GE(sized->graph_visited().size(), 2000U);
  EXPECT_EQ(sized->bytes(), 11 * ggml_tensor_overhead() + traversal_bytes);
  const auto total_bytes = sized->bytes();
  EXPECT_EQ(sized->used(), 0U);
  // Its checks answer for the estimate the build was checked against, so
  // the same build passes them; sealed, it answers for its own room.
  EXPECT_TRUE(sized->Reserve(1000).has_value());
  EXPECT_TRUE(Build(*sized, 10));
  sized->Seal();
  EXPECT_EQ(sized->bytes(), total_bytes);
  EXPECT_EQ(sized->graph_visited().size_bytes(), traversal_bytes);
  EXPECT_EQ(sized->graph_capacity(), 1000U);
  EXPECT_TRUE(sized->Reserve(1).has_value());
  EXPECT_FALSE(sized->Reserve(2).has_value());
  // A build that fails keeps the estimate, for the caller's own error.
  auto failed =
      en::SizedArena(8, [](jitllm::kernels::ggml::TensorArena& a) { return Build(a, 10); });
  ASSERT_TRUE(failed.has_value());
  EXPECT_EQ(failed->graph_capacity(), 8U);
  EXPECT_EQ(failed->graph_visited().size(), 16U);
  EXPECT_EQ(failed->bytes(), 8 * ggml_tensor_overhead() + failed->graph_visited().size_bytes());
}

}  // namespace

TEST(PlanCacheTest, ReaderScratchIsMeasuredOnceAndRuntimeCannotGrowIt) {
  namespace kg = jitllm::kernels::ggml;
  auto arena = kg::TensorArena::Create(8);
  ASSERT_TRUE(arena);
  auto* input = ggml_new_tensor_1d(arena->context(), GGML_TYPE_F32, 4);
  auto* output = ggml_scale(arena->context(), input, 2.0f);
  const std::array nodes{output};
  const std::array inputs{input};
  const std::array keep{output};
  kg::DeviceChoices choices;
  choices.fuse_norms = true;
  en::PlannedBase measured;
  auto measure = en::PlaceAndPlan(measured, nodes, inputs, keep, choices, 0, 0);
  ASSERT_TRUE(measure);
  const auto envelope = en::ScratchArenaBytes();
  const auto allowance = kg::detail::GraphReadIndex::ScratchBytes(nodes, keep);
  ASSERT_TRUE(allowance);
  EXPECT_GE(envelope, *allowance);
  en::PlannedBase placed;
  auto place =
      en::PlaceAndPlan(placed, nodes, inputs, keep, choices, 1ULL << 40, measured.placement.extent);
  ASSERT_TRUE(place);
  EXPECT_TRUE(kg::SamePlan(measured.plan, placed.plan));
  EXPECT_EQ(en::ScratchArenaBytes(), envelope);
  // A caller cannot silently enlarge the startup envelope, even by duplicate
  // keep entries. Refusal precedes address rebinding and index allocation.
  std::vector<ggml_tensor*> oversized_keep(envelope / 1024 + 1, output);
  const auto address = output->data;
  en::PlannedBase refused;
  EXPECT_FALSE(en::PlaceAndPlan(refused, nodes, inputs, oversized_keep, choices, 1ULL << 42,
                                measured.placement.extent));
  EXPECT_EQ(output->data, address);
  EXPECT_EQ(en::ScratchArenaBytes(), envelope);
}
