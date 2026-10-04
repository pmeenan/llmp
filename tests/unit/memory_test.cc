// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// The commitment ledger against D-050's worked cases
// (docs/reservation-policy.md#worked-cases-and-implementation-gates), and
// the victim-selection baseline.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

#include "base/bytes.h"
#include "catalog/catalog.h"
#include "expected_error.h"
#include "memory/commitment.h"
#include "memory/materialize.h"
#include "memory/reclaim.h"
#include "memory/victims.h"

namespace {

using jitllm::base::Bytes;
using jitllm::test_support::Failed;
using jitllm::base::operator""_MiB;
using jitllm::catalog::Catalog;
using jitllm::catalog::ExtentId;
using jitllm::catalog::MemoryClass;
using jitllm::catalog::Recovery;
using jitllm::memory::CommitmentError;
using jitllm::memory::CommitmentLedger;
using jitllm::memory::Envelope;
using jitllm::memory::GrantId;
using ::testing::ElementsAre;

Envelope E(std::uint64_t retained, std::uint64_t phase) {
  return {.retained = Bytes(retained), .phase = Bytes(phase)};
}

constexpr jitllm::catalog::DomainId kSpark(0, 1);
constexpr jitllm::catalog::DomainId kSparkB(1, 1);
constexpr Bytes kBudget = 1024_MiB;

jitllm::memory::CommitmentTotals Totals(const CommitmentLedger& ledger) {
  const auto totals = ledger.Totals(kSpark);
  EXPECT_TRUE(totals.has_value());
  return totals.value_or(jitllm::memory::CommitmentTotals{});
}

CommitmentLedger Ledger(std::uint64_t budget) {
  CommitmentLedger ledger;
  EXPECT_TRUE(ledger.AddDomain(kSpark, Bytes(budget)).has_value());
  return ledger;
}

// B=100, F=10, J=0: A and B (R=20, E=50) fit serially; a third (1, 1) is
// deferred; (41, 50) is impossible even alone.
TEST(CommitmentLedger, TheWorkedCases) {
  CommitmentLedger ledger = Ledger(100);
  ASSERT_TRUE(ledger.AddFixed(kSpark, Bytes(10)).has_value());
  const GrantId a = ledger.Grant(kSpark, E(20, 50)).value();
  const GrantId b = ledger.Grant(kSpark, E(20, 50)).value();
  EXPECT_EQ(Totals(ledger).required, Bytes(100));
  EXPECT_EQ(Failed(ledger.Grant(kSpark, E(1, 1))), CommitmentError::kDoesNotFit);
  // Both phases at once would need 10 + 40 + 100.
  const std::vector<GrantId> cohort = {a, b};
  EXPECT_EQ(Failed(ledger.SetCohort(kSpark, cohort)), CommitmentError::kDoesNotFit);
  const std::vector<GrantId> alone = {a};
  EXPECT_TRUE(ledger.SetCohort(kSpark, alone).has_value());

  CommitmentLedger empty = Ledger(100);
  ASSERT_TRUE(empty.AddFixed(kSpark, Bytes(10)).has_value());
  EXPECT_EQ(Failed(empty.Grant(kSpark, E(41, 50))), CommitmentError::kDoesNotFit);
}

TEST(CommitmentLedger, ReplacementIsAtomicAndRetirementNeverRefused) {
  CommitmentLedger ledger = Ledger(100);
  const GrantId a = ledger.Grant(kSpark, E(20, 50)).value();
  EXPECT_EQ(Failed(ledger.Replace(a, E(60, 50))), CommitmentError::kDoesNotFit);
  EXPECT_EQ(ledger.EnvelopeOf(a).value_or(Envelope{}).retained, Bytes(20));  // the old grant stands
  ASSERT_TRUE(ledger.Replace(a, E(30, 50)).has_value());
  EXPECT_EQ(Totals(ledger).required, Bytes(80));
  ASSERT_TRUE(ledger.Retire(a).has_value());
  EXPECT_EQ(Failed(ledger.Retire(a)), CommitmentError::kUnknownGrant);
  EXPECT_EQ(Failed(ledger.Replace(a, E(1, 1))), CommitmentError::kUnknownGrant);
  EXPECT_EQ(Totals(ledger).required, Bytes(0));
}

// Every change is checked against the active cohort, not only the serial
// rule (docs/reservation-policy.md: "recheck the serial and any
// active-cohort inequalities").
TEST(CommitmentLedger, ChangesAreCheckedAgainstTheActiveCohort) {
  CommitmentLedger ledger = Ledger(100);
  const GrantId a = ledger.Grant(kSpark, E(0, 40)).value();
  const GrantId b = ledger.Grant(kSpark, E(0, 40)).value();
  const std::vector<GrantId> both = {a, b};
  ASSERT_TRUE(ledger.SetCohort(kSpark, both).has_value());
  EXPECT_EQ(Failed(ledger.Replace(a, E(0, 70))), CommitmentError::kDoesNotFit);  // 70 + 40
  EXPECT_EQ(Failed(ledger.AddFixed(kSpark, Bytes(21))), CommitmentError::kDoesNotFit);
  EXPECT_EQ(Failed(ledger.SetBudget(kSpark, Bytes(79))), CommitmentError::kDoesNotFit);
  // Retiring a member leaves the cohort, and never fails.
  ASSERT_TRUE(ledger.Retire(b).has_value());
  EXPECT_EQ(Totals(ledger).cohort_phase, Bytes(40));
  EXPECT_TRUE(ledger.Replace(a, E(0, 70)).has_value());
}

// A budget reduction waits or is refused; it never drops below claims.
TEST(CommitmentLedger, TheBudgetNeverDropsBelowClaims) {
  CommitmentLedger ledger = Ledger(100);
  ASSERT_TRUE(ledger.AddBackground(kSpark, Bytes(10)).has_value());
  ASSERT_TRUE(ledger.Grant(kSpark, E(20, 50)).has_value());
  EXPECT_EQ(Failed(ledger.SetBudget(kSpark, Bytes(79))), CommitmentError::kDoesNotFit);
  EXPECT_EQ(Totals(ledger).budget, Bytes(100));
  ASSERT_TRUE(ledger.SetBudget(kSpark, Bytes(80)).has_value());
  // F and J increases are checked too; releases are not.
  EXPECT_EQ(Failed(ledger.AddFixed(kSpark, Bytes(1))), CommitmentError::kDoesNotFit);
  EXPECT_EQ(Failed(ledger.AddBackground(kSpark, Bytes(1))), CommitmentError::kDoesNotFit);
  ASSERT_TRUE(ledger.ReleaseBackground(kSpark, Bytes(10)).has_value());
  EXPECT_EQ(Failed(ledger.ReleaseBackground(kSpark, Bytes(1))), CommitmentError::kUnderflow);
  EXPECT_EQ(Failed(ledger.ReleaseFixed(kSpark, Bytes(1))), CommitmentError::kUnderflow);
}

TEST(CommitmentLedger, ArithmeticOverflowIsRefused) {
  CommitmentLedger ledger = Ledger(UINT64_MAX);
  ASSERT_TRUE(ledger.Grant(kSpark, E(UINT64_MAX - 1, 1)).has_value());
  EXPECT_EQ(Failed(ledger.Grant(kSpark, E(1, 0))), CommitmentError::kOverflow);
  EXPECT_EQ(Failed(ledger.AddFixed(kSpark, Bytes(UINT64_MAX))), CommitmentError::kOverflow);
}

TEST(CommitmentLedger, ResumptionOverflowDefersAChangeThatFitsNow) {
  CommitmentLedger ledger = Ledger(UINT64_MAX);
  const GrantId paused = ledger.Grant(kSpark, E(0, UINT64_MAX - 10)).value();
  const GrantId peer = ledger.Grant(kSpark, E(0, 10)).value();
  const GrantId substitute = ledger.Grant(kSpark, E(0, 1)).value();
  const std::vector<GrantId> running = {peer, substitute};
  const std::vector<GrantId> resuming = {paused, peer};
  ASSERT_TRUE(ledger.SetCohort(kSpark, running).has_value());
  ASSERT_TRUE(ledger.SetResumption(kSpark, resuming, substitute).has_value());
  EXPECT_EQ(Failed(ledger.Replace(peer, E(0, 20))), CommitmentError::kBreaksResumption);
  const auto envelope = ledger.EnvelopeOf(peer);
  ASSERT_TRUE(envelope.has_value());
  EXPECT_EQ(envelope.value_or(Envelope{}).phase, Bytes(10));
}

// Domains are separate budgets, and a grant belongs to one of them.
TEST(CommitmentLedger, DomainsAreSeparate) {
  CommitmentLedger ledger = Ledger(100);
  ASSERT_TRUE(ledger.AddDomain(kSparkB, Bytes(50)).has_value());
  EXPECT_EQ(Failed(ledger.AddDomain(kSparkB, Bytes(50))), CommitmentError::kUnknownDomain);
  const GrantId here = ledger.Grant(kSpark, E(0, 80)).value();
  EXPECT_EQ(Failed(ledger.Grant(kSparkB, E(0, 80))), CommitmentError::kDoesNotFit);
  const GrantId there = ledger.Grant(kSparkB, E(0, 40)).value();
  EXPECT_NE(here, there);
  EXPECT_EQ(ledger.DomainOf(there), kSparkB);
  const std::vector<GrantId> mixed = {here, there};
  EXPECT_EQ(Failed(ledger.SetCohort(kSpark, mixed)), CommitmentError::kUnknownGrant);
  const std::vector<GrantId> twice = {here, here};
  EXPECT_EQ(Failed(ledger.SetCohort(kSpark, twice)), CommitmentError::kUnknownGrant);
  EXPECT_EQ(Failed(ledger.Grant(jitllm::catalog::DomainId(9, 1), E(1, 1))),
            CommitmentError::kUnknownDomain);
}

class VictimTest : public ::testing::Test {
 protected:
  void SetUp() override { domain_ = catalog_.AddDomain("spark"); }

  ExtentId Resident(MemoryClass memory_class, Recovery recovery, std::uint32_t chunk,
                    std::uint64_t used) {
    jitllm::catalog::ExtentDescriptor descriptor{.domain = domain_,
                                                 .memory_class = memory_class,
                                                 .recovery = recovery,
                                                 .size = 2_MiB,
                                                 .content = {}};
    descriptor.content.chunk = chunk;
    const ExtentId extent = catalog_.AddExtent(descriptor).value();
    const auto ticket = catalog_.BeginLoad(extent, kBudget).value();
    EXPECT_TRUE(catalog_.CompleteLoad(ticket).has_value());
    if (used != 0) {
      const std::vector<ExtentId> only = {extent};
      const auto lease = catalog_.AcquireLease(catalog_.ClosureOfExtents(only).value()).value();
      EXPECT_TRUE(catalog_.RecordUse(lease, used).has_value());
      EXPECT_TRUE(catalog_.ReleaseLease(lease).has_value());
    }
    return extent;
  }

  Catalog catalog_;
  jitllm::catalog::DomainId domain_;
};

TEST_F(VictimTest, DiscardedFirstThenLeastRecentlyUsedThenContent) {
  const ExtentId recent = Resident(MemoryClass::kWeights, Recovery::kFromArtifact, 0, 9);
  const ExtentId old_b = Resident(MemoryClass::kWeights, Recovery::kFromArtifact, 5, 2);
  const ExtentId old_a = Resident(MemoryClass::kWeights, Recovery::kFromArtifact, 3, 2);
  const ExtentId scratch = Resident(MemoryClass::kScratch, Recovery::kDiscardable, 0, 20);
  const ExtentId never = Resident(MemoryClass::kWeights, Recovery::kFromArtifact, 7, 0);
  auto plan = jitllm::memory::SelectVictims(catalog_, domain_, 10_MiB);
  std::vector<ExtentId> order;
  order.reserve(plan.victims.size());
  for (const auto& victim : plan.victims) {
    order.push_back(victim.extent);
  }
  EXPECT_THAT(order, ElementsAre(scratch, never, old_a, old_b, recent));
  EXPECT_TRUE(plan.sufficient);
  EXPECT_EQ(plan.credited, 10_MiB);
  EXPECT_EQ(plan.passed_over, 0U);
  // Only what is needed; the rest are passed over.
  plan = jitllm::memory::SelectVictims(catalog_, domain_, 3_MiB);
  ASSERT_EQ(plan.victims.size(), 2U);
  EXPECT_EQ(plan.passed_over, 3U);
  EXPECT_EQ(plan.victims[0].memory_class, MemoryClass::kScratch);
}

TEST_F(VictimTest, NeverHeldPinnedOrProtected) {
  const ExtentId leased = Resident(MemoryClass::kWeights, Recovery::kFromArtifact, 0, 1);
  const ExtentId state = Resident(MemoryClass::kLiveState, Recovery::kPreserve, 1, 1);
  const ExtentId pending = Resident(MemoryClass::kWeights, Recovery::kFromArtifact, 2, 1);
  const ExtentId free = Resident(MemoryClass::kWeights, Recovery::kFromArtifact, 3, 1);
  const std::vector<ExtentId> only = {leased};
  (void)catalog_.AcquireLease(catalog_.ClosureOfExtents(only).value()).value();
  (void)catalog_
      .AddExtent({.domain = domain_,
                  .memory_class = MemoryClass::kUnknown,
                  .recovery = Recovery::kPinned,
                  .size = 2_MiB,
                  .content = {}},
                 /*resident=*/true)
      .value();
  (void)state;
  const std::vector<ExtentId> protect = {pending};
  const auto plan = jitllm::memory::SelectVictims(catalog_, domain_, 8_MiB, protect);
  ASSERT_EQ(plan.victims.size(), 1U);
  EXPECT_EQ(plan.victims[0].extent, free);
  EXPECT_FALSE(plan.sufficient);  // not enough: nothing should be evicted for it
}

TEST_F(VictimTest, TheChoiceIsDeterministic) {
  for (std::uint32_t chunk = 0; chunk < 16; ++chunk) {
    (void)Resident(MemoryClass::kWeights, Recovery::kFromArtifact, 15 - chunk, 1);
  }
  const auto first = jitllm::memory::SelectVictims(catalog_, domain_, 8_MiB);
  const auto second = jitllm::memory::SelectVictims(catalog_, domain_, 8_MiB);
  ASSERT_EQ(first.victims.size(), 4U);
  for (std::size_t i = 0; i < first.victims.size(); ++i) {
    EXPECT_EQ(first.victims[i].extent, second.victims[i].extent);
    EXPECT_EQ(catalog_.Describe(first.victims[i].extent).value().descriptor.content.chunk, i);
  }
}

class MaterializeTest : public VictimTest {
 protected:
  ExtentId Absent(std::uint32_t chunk) {
    jitllm::catalog::ExtentDescriptor descriptor{.domain = domain_,
                                                 .memory_class = MemoryClass::kWeights,
                                                 .recovery = Recovery::kFromArtifact,
                                                 .size = 2_MiB,
                                                 .content = {}};
    descriptor.content.chunk = chunk;
    return catalog_.AddExtent(descriptor).value();
  }

  jitllm::catalog::Closure Of(const std::vector<ExtentId>& extents) {
    return catalog_.ClosureOfExtents(extents).value();
  }
};

TEST_F(MaterializeTest, AResidentClosureIsReady) {
  const ExtentId a = Resident(MemoryClass::kWeights, Recovery::kFromArtifact, 0, 1);
  const auto plan = jitllm::memory::PlanMaterialization(catalog_, domain_, 2_MiB, Of({a}));
  EXPECT_TRUE(plan.feasible);
  EXPECT_TRUE(plan.ready);
  EXPECT_EQ(plan.missing, Bytes());
}

// Only the missing dependencies are materialized, and victims are chosen
// only for the shortfall against B, never from the closure or protected.
TEST_F(MaterializeTest, OnlyTheShortfallIsReclaimed) {
  const ExtentId needed = Resident(MemoryClass::kWeights, Recovery::kFromArtifact, 0, 1);
  const ExtentId kept = Resident(MemoryClass::kWeights, Recovery::kFromArtifact, 1, 2);
  const ExtentId old = Resident(MemoryClass::kWeights, Recovery::kFromArtifact, 2, 3);
  const ExtentId newer = Resident(MemoryClass::kWeights, Recovery::kFromArtifact, 3, 4);
  const ExtentId x = Absent(10);
  const ExtentId y = Absent(11);
  const std::vector<ExtentId> protect = {kept};
  // 8 MiB occupied, 4 MiB missing, B = 10 MiB: 2 MiB short.
  auto plan =
      jitllm::memory::PlanMaterialization(catalog_, domain_, 10_MiB, Of({needed, x, y}), protect);
  EXPECT_THAT(plan.load, ElementsAre(x, y));
  EXPECT_EQ(plan.missing, 4_MiB);
  EXPECT_EQ(plan.shortfall, 2_MiB);
  ASSERT_EQ(plan.victims.victims.size(), 1U);
  EXPECT_EQ(plan.victims.victims[0].extent, old);  // not `needed` (older) or `kept`
  EXPECT_TRUE(plan.feasible);
  EXPECT_FALSE(plan.ready);
  // Loads wait for the victim's eviction to complete.
  EXPECT_EQ(catalog_.BeginLoad(x, 10_MiB).value_or(jitllm::catalog::Ticket{}).extent, x);
  EXPECT_EQ(Failed(catalog_.BeginLoad(y, 10_MiB)), jitllm::catalog::CatalogError::kOverBudget);
  auto evict = catalog_.BeginEvict(old).value();
  EXPECT_EQ(Failed(catalog_.BeginLoad(y, 10_MiB)), jitllm::catalog::CatalogError::kOverBudget);
  ASSERT_TRUE(catalog_.CompleteEvict(evict).has_value());
  EXPECT_TRUE(catalog_.BeginLoad(y, 10_MiB).has_value());
  plan =
      jitllm::memory::PlanMaterialization(catalog_, domain_, 10_MiB, Of({needed, x, y}), protect);
  EXPECT_THAT(plan.loading, ElementsAre(x, y));
  EXPECT_EQ(plan.shortfall, Bytes());
  EXPECT_TRUE(plan.feasible);
  (void)newer;
}

TEST_F(MaterializeTest, InsufficientVictimsMakeItInfeasible) {
  const ExtentId leased = Resident(MemoryClass::kWeights, Recovery::kFromArtifact, 0, 1);
  const std::vector<ExtentId> only = {leased};
  (void)catalog_.AcquireLease(catalog_.ClosureOfExtents(only).value()).value();
  const ExtentId x = Absent(10);
  const auto plan = jitllm::memory::PlanMaterialization(catalog_, domain_, 3_MiB, Of({x}));
  EXPECT_EQ(plan.shortfall, 1_MiB);
  EXPECT_FALSE(plan.victims.sufficient);
  EXPECT_FALSE(plan.feasible);
}

TEST_F(MaterializeTest, InFlightQuarantinedAndStaleExtents) {
  const ExtentId evicting = Resident(MemoryClass::kWeights, Recovery::kFromArtifact, 0, 1);
  const ExtentId loading = Absent(1);
  const ExtentId broken = Absent(2);
  (void)catalog_.BeginEvict(evicting).value();
  (void)catalog_.BeginLoad(loading, kBudget).value();
  const auto failed = catalog_.BeginLoad(broken, kBudget).value();
  auto plan =
      jitllm::memory::PlanMaterialization(catalog_, domain_, kBudget, Of({evicting, loading}));
  EXPECT_THAT(plan.cancel, ElementsAre(evicting));
  EXPECT_THAT(plan.loading, ElementsAre(loading));
  EXPECT_TRUE(plan.feasible);
  EXPECT_FALSE(plan.ready);
  ASSERT_TRUE(catalog_.FailLoad(failed, /*completion_known=*/false).has_value());
  plan = jitllm::memory::PlanMaterialization(catalog_, domain_, kBudget, Of({broken}));
  EXPECT_THAT(plan.quarantined, ElementsAre(broken));
  EXPECT_FALSE(plan.feasible);
  // Invalidated state cannot be restored by loading.
  const ExtentId state = Resident(MemoryClass::kLiveState, Recovery::kPreserve, 3, 0);
  const auto taken = Of({state});
  ASSERT_TRUE(catalog_.InvalidateContents(state).has_value());
  plan = jitllm::memory::PlanMaterialization(catalog_, domain_, kBudget, taken);
  EXPECT_THAT(plan.stale, ElementsAre(state));
  EXPECT_FALSE(plan.feasible);
}

// The node's one reclaim order (memory/reclaim.h): GreedyDual-Size over
// each kind's measured cost to restore a byte, so with equal inflation the
// cheapest kind first; strict least recent use within a kind (never largest
// first), the running model's last.
using jitllm::memory::KindCosts;
using jitllm::memory::ProtectFloor;
using jitllm::memory::ReclaimCandidate;
using jitllm::memory::ReclaimKind;
using jitllm::memory::ReclaimOrder;
using jitllm::memory::RunReclaim;
using jitllm::memory::SelectReclaim;

constexpr std::uint64_t kGiB = std::uint64_t{1} << 30U;

// The costs measured on a GB10 (docs/experiments/memory-pressure), a GiB
// each: weights paged in at 13.3 GB/s, idle state written back and read
// again at 11.0 and 14.5 GB/s, a decode graph's capture (9 ms for ~40 MiB),
// a decode plan's planning (23 ms for ~2.4 MiB).
std::vector<ReclaimCandidate> Measured() {
  return {
      {.kind = ReclaimKind::kPlan,
       .owner = 0,
       .id = 1,
       .bytes = 2516582,
       .last_use = 1,
       .restore_seconds = 0.023},
      {.kind = ReclaimKind::kGraph,
       .owner = 0,
       .id = 1,
       .bytes = 41943040,
       .last_use = 1,
       .restore_seconds = 0.009},
      {.kind = ReclaimKind::kIdleState,
       .owner = 0,
       .id = 2,
       .bytes = kGiB,
       .last_use = 5,
       .restore_seconds = (1.0737 / 11.0) + (1.0737 / 14.5)},
      {.kind = ReclaimKind::kIdleWeights,
       .owner = 1,
       .id = 3,
       .bytes = kGiB,
       .last_use = 9,
       .restore_seconds = 1.0737 / 13.3},
  };
}

TEST(ReclaimOrder, KindsGoByTheirMeasuredCostToRestoreAByte) {
  const auto candidates = Measured();
  const auto cost = KindCosts(candidates);
  EXPECT_NEAR(cost[static_cast<std::size_t>(ReclaimKind::kIdleWeights)], 0.081, 0.001);
  EXPECT_NEAR(cost[static_cast<std::size_t>(ReclaimKind::kIdleState)], 0.172, 0.001);
  EXPECT_NEAR(cost[static_cast<std::size_t>(ReclaimKind::kGraph)], 0.230, 0.001);
  EXPECT_NEAR(cost[static_cast<std::size_t>(ReclaimKind::kPlan)], 9.81, 0.01);
  // Weights, then idle state (spilled), then graphs, then plans.
  EXPECT_THAT(ReclaimOrder(candidates), ElementsAre(3U, 2U, 1U, 0U));
  // A measurement that makes plans cheaper per byte moves them first.
  auto cheap = candidates;
  cheap[0].restore_seconds = 0.0001;
  EXPECT_EQ(ReclaimOrder(cheap).front(), 0U);
}

TEST(ReclaimOrder, WithinAKindTheLeastRecentlyUsedGoesFirstNeverTheLargest) {
  std::vector<ReclaimCandidate> plans = {
      {.kind = ReclaimKind::kPlan,
       .owner = 0,
       .id = 1,
       .bytes = 100,
       .last_use = 9,
       .restore_seconds = 0.05},
      {.kind = ReclaimKind::kPlan,
       .owner = 0,
       .id = 2,
       .bytes = 900,
       .last_use = 3,
       .restore_seconds = 0.05},
      {.kind = ReclaimKind::kPlan,
       .owner = 1,
       .id = 3,
       .bytes = 5000,
       .last_use = 6,
       .restore_seconds = 0.05},
      // The running model's goes after every other model's of its kind.
      {.kind = ReclaimKind::kPlan,
       .owner = 1,
       .id = 4,
       .bytes = 50,
       .last_use = 1,
       .restore_seconds = 0.05,
       .running = true},
  };
  EXPECT_THAT(ReclaimOrder(plans), ElementsAre(1U, 2U, 0U, 3U));
  // Enough is taken in that order and no more.
  const auto plan = SelectReclaim(plans, 950);
  EXPECT_THAT(plan.victims, ElementsAre(1U, 2U));
  EXPECT_EQ(plan.bytes, 5900U);
  EXPECT_TRUE(plan.sufficient);
  const auto all = SelectReclaim(plans, 1U << 20U);
  EXPECT_EQ(all.victims.size(), 4U);
  EXPECT_FALSE(all.sufficient);
}

TEST(ReclaimOrder, TheChoiceIsDeterministicAndIgnoresEmptyCandidates) {
  std::vector<ReclaimCandidate> c = {
      {.kind = ReclaimKind::kGraph, .owner = 2, .id = 1, .bytes = 10, .last_use = 4},
      {.kind = ReclaimKind::kGraph, .owner = 1, .id = 7, .bytes = 10, .last_use = 4},
      {.kind = ReclaimKind::kGraph, .owner = 1, .id = 5, .bytes = 0, .last_use = 1},
      {.kind = ReclaimKind::kGraph, .owner = 1, .id = 6, .bytes = 10, .last_use = 4},
  };
  EXPECT_THAT(ReclaimOrder(c), ElementsAre(3U, 1U, 0U));
  EXPECT_TRUE(SelectReclaim(c, 0).victims.empty());
  EXPECT_TRUE(SelectReclaim({}, 1).victims.empty());
  EXPECT_FALSE(SelectReclaim({}, 1).sufficient);
}

// GreedyDual-Size: a candidate's priority is the inflation value at its last
// use plus its kind's cost a GiB, and every reclaim raises the inflation to
// what it took. A costly kind outlasts a cheap one while both are in use; one
// left unused falls behind cheap ones used after reclaims raised the value.
TEST(ReclaimOrder, AStaleCostlyEntryFallsBehindFreshCheapOnesAsReclaimsGoOn) {
  std::vector<ReclaimCandidate> c = {
      {.kind = ReclaimKind::kPlan,
       .owner = 0,
       .id = 1,
       .bytes = kGiB / 64,
       .last_use = 1,
       .inflation = 0,
       .restore_seconds = 0.08},  // 5.12 s a GiB
      {.kind = ReclaimKind::kIdleState,
       .owner = 0,
       .id = 2,
       .bytes = kGiB,
       .last_use = 8,
       .inflation = 0,
       .restore_seconds = 0.17},
  };
  // Both used at the same inflation: the cheap idle state goes first.
  EXPECT_THAT(ReclaimOrder(c), ElementsAre(1U, 0U));
  // Idle state used again once reclaims raised the value past the plan's
  // cost: the plan, unused since, goes first.
  c[1].inflation = 5.0;
  EXPECT_THAT(ReclaimOrder(c), ElementsAre(0U, 1U));
  const auto plan = SelectReclaim(c, 1);
  ASSERT_EQ(plan.priorities.size(), 1U);
  EXPECT_NEAR(plan.priorities[0], 5.12, 0.01);
  // The value only rises.
  const double before = jitllm::memory::ReclaimInflation();
  jitllm::memory::RaiseReclaimInflation(before + 1.5);
  EXPECT_DOUBLE_EQ(jitllm::memory::ReclaimInflation(), before + 1.5);
  jitllm::memory::RaiseReclaimInflation(before);
  EXPECT_DOUBLE_EQ(jitllm::memory::ReclaimInflation(), before + 1.5);
}

// Staleness decays the expected cost directly: a plan (about 6 s a GiB)
// unused for a handful of reclaims, or for minutes, ranks below an idle
// conversation used just now (0.2 s a GiB), whatever the inflation did;
// within a kind the order stays strict least recent use.
TEST(ReclaimOrder, AStalePlanFallsBelowAFreshConversationWithinAFewReclaims) {
  const ReclaimCandidate plan{.kind = ReclaimKind::kPlan,
                              .owner = 0,
                              .id = 1,
                              .bytes = kGiB / 64,
                              .last_use = 1,
                              .restore_seconds = 6.0 / 64};
  const ReclaimCandidate fresh{.kind = ReclaimKind::kIdleState,
                               .owner = 0,
                               .id = 2,
                               .bytes = kGiB,
                               .last_use = 9,
                               .restore_seconds = 0.2};
  // Used at the same time: the plan stays.
  std::vector<ReclaimCandidate> c = {plan, fresh};
  EXPECT_THAT(ReclaimOrder(c), ElementsAre(1U, 0U));
  // Unused for k reclaims since (each raising the inflation by what it
  // took, about 0.2): the first k at which the plan goes first.
  std::uint64_t k = 0;
  for (; k < 64; ++k) {
    c[0].reclaims_since = k;
    c[1].inflation = 0.2 * static_cast<double>(k);
    if (ReclaimOrder(c).front() == 0U) {
      break;
    }
  }
  EXPECT_LE(k, 5U);
  // Unused for minutes with no reclaim at all: it goes first too.
  c[0].reclaims_since = 0;
  c[1].inflation = 0;
  c[0].idle_seconds = 30 * 60;
  EXPECT_EQ(ReclaimOrder(c).front(), 0U);
  // Within a kind, staler is never ranked after fresher.
  std::vector<ReclaimCandidate> plans = {plan, plan};
  plans[0].id = 1;
  plans[0].reclaims_since = 0;
  plans[0].idle_seconds = 1;
  plans[1].id = 2;
  plans[1].reclaims_since = 3;
  plans[1].idle_seconds = 90;
  EXPECT_THAT(ReclaimOrder(plans), ElementsAre(1U, 0U));
}

// No more than needed: a few KiB do not cost an idle conversation when a
// later candidate covers them at no more absolute cost to restore.
TEST(ReclaimOrder, ATinyNeedTakesASmallSufficientCandidateNotAConversation) {
  std::vector<ReclaimCandidate> c = {
      {.kind = ReclaimKind::kIdleState,
       .owner = 0,
       .id = 1,
       .bytes = 451 * (kGiB >> 10U),
       .last_use = 1,
       .restore_seconds = 0.08},
      {.kind = ReclaimKind::kPlan,
       .owner = 0,
       .id = 2,
       .bytes = 3 * (kGiB >> 10U),
       .last_use = 1,
       .restore_seconds = 0.03},
  };
  ASSERT_THAT(ReclaimOrder(c), ElementsAre(0U, 1U));  // the conversation is cheaper a byte
  const auto tiny = SelectReclaim(c, 4096);
  EXPECT_THAT(tiny.victims, ElementsAre(1U));
  EXPECT_TRUE(tiny.sufficient);
  // A need the plan does not cover, or a plan that costs more to rebuild
  // than the conversation to spill, keeps the order.
  EXPECT_THAT(SelectReclaim(c, 64 * (kGiB >> 10U)).victims, ElementsAre(0U));
  c[1].restore_seconds = 0.2;
  EXPECT_THAT(SelectReclaim(c, 4096).victims, ElementsAre(0U));
}

// A small need costs less to cover from several small entries of a costlier
// kind than from one large cheap one: 55 MiB for a request slot takes seven
// stale 8 MiB graphs (about 0.02 s to capture again), not a 704 MiB idle
// conversation (0.12 s to spill and restore), though the order ranks the
// conversation's GiB cheaper. The graphs go in strict least recent use;
// a need the graphs cannot cover takes the conversation.
TEST(ReclaimOrder, ASmallNeedTakesTheCheapestTotalToRestoreNotACheapGiB) {
  constexpr std::uint64_t kMiB = std::uint64_t{1} << 20U;
  std::vector<ReclaimCandidate> c = {
      {.kind = ReclaimKind::kIdleState,
       .owner = 0,
       .id = 1,
       .bytes = 704 * kMiB,
       .last_use = 100,
       .restore_seconds = (704.0 / 1024) * 0.17},
  };
  for (std::uint64_t g = 0; g < 10; ++g) {
    c.push_back({.kind = ReclaimKind::kGraph,
                 .owner = 0,
                 .id = 10 + g,
                 .bytes = 8 * kMiB,
                 .last_use = 50 - g,  // the last pushed is the oldest
                 .restore_seconds = (8.0 / 1024) * 0.36});
  }
  ASSERT_EQ(ReclaimOrder(c).front(), 0U);  // the conversation is cheaper a GiB
  const auto small = SelectReclaim(c, 55 * kMiB);
  EXPECT_TRUE(small.sufficient);
  EXPECT_EQ(small.bytes, 56 * kMiB);
  EXPECT_THAT(small.victims, ElementsAre(10U, 9U, 8U, 7U, 6U, 5U, 4U));
  // Ten graphs hold 80 MiB: a need past them takes the conversation.
  const auto large = SelectReclaim(c, 100 * kMiB);
  EXPECT_THAT(large.victims, ElementsAre(0U));
  // Graphs that cost more in all than the conversation keep the order.
  for (std::size_t i = 1; i < c.size(); ++i) {
    c[i].restore_seconds = 0.05;
  }
  EXPECT_THAT(SelectReclaim(c, 55 * kMiB).victims, ElementsAre(0U));
}

// Within a kind too: 100 MiB beside an old 704 MiB conversation, a newer
// 30 MiB one and ten 8 MiB graphs takes the 30 MiB conversation and nine
// graphs (about 0.03 s to restore), not the old 704 MiB one (0.12 s); the
// old one gives way only because a later conversation fits what is left.
TEST(ReclaimOrder, AnOldLargeEntryGivesWayToALaterOneOfItsKindThatFits) {
  constexpr std::uint64_t kMiB = std::uint64_t{1} << 20U;
  std::vector<ReclaimCandidate> c = {
      {.kind = ReclaimKind::kIdleState,
       .owner = 0,
       .id = 1,
       .bytes = 704 * kMiB,
       .last_use = 100,
       .restore_seconds = (704.0 / 1024) * 0.17},
      {.kind = ReclaimKind::kIdleState,
       .owner = 0,
       .id = 2,
       .bytes = 30 * kMiB,
       .last_use = 200,
       .restore_seconds = (30.0 / 1024) * 0.17},
  };
  for (std::uint64_t g = 0; g < 10; ++g) {
    c.push_back({.kind = ReclaimKind::kGraph,
                 .owner = 0,
                 .id = 10 + g,
                 .bytes = 8 * kMiB,
                 .last_use = 50 - g,  // the last pushed is the oldest
                 .restore_seconds = (8.0 / 1024) * 0.36});
  }
  const auto plan = SelectReclaim(c, 100 * kMiB);
  EXPECT_TRUE(plan.sufficient);
  EXPECT_EQ(plan.bytes, 102 * kMiB);
  EXPECT_THAT(plan.victims, ElementsAre(1U, 11U, 10U, 9U, 8U, 7U, 6U, 5U, 4U, 3U));
  // A need only the old one covers takes it.
  EXPECT_THAT(SelectReclaim(c, 200 * kMiB).victims, ElementsAre(0U));
}

// An optional charge reclaims only what costs less to restore than what it
// charges, and all of it or nothing; a plan and its graph count once.
TEST(ReclaimOrder, AReclaimTakesOnlyWhatIsCheaperAndCountsAPlanWithItsGraphOnce) {
  const std::vector<ReclaimCandidate> c = {
      {.kind = ReclaimKind::kGraph,
       .owner = 0,
       .id = 1,
       .bytes = 40,
       .last_use = 1,
       .restore_seconds = 0.000001},
      {.kind = ReclaimKind::kPlan,
       .owner = 0,
       .id = 1,
       .bytes = 50,  // 10 of its own beside its graph's 40
       .last_use = 1,
       .restore_seconds = 0.01},
      {.kind = ReclaimKind::kIdleState,
       .owner = 0,
       .id = 2,
       .bytes = 1000,
       .last_use = 2,
       .restore_seconds = 0.0000001},
  };
  // Everything: the idle state, the graph, then the plan's own 10.
  const auto all = SelectReclaim(c, 2000);
  EXPECT_EQ(all.bytes, 1050U);
  EXPECT_FALSE(all.sufficient);
  // Below the graphs' priority: only the idle state.
  const auto cost = KindCosts(c);
  const double graph = jitllm::memory::ReclaimPriority(c[0], cost);
  const auto cheaper = SelectReclaim(c, 2000, graph);
  EXPECT_THAT(cheaper.victims, ElementsAre(2U));
  EXPECT_FALSE(cheaper.sufficient);
  EXPECT_TRUE(SelectReclaim(c, 500, graph).sufficient);
}

// The running model's in-use floor: its most recently used plans up to its
// plan_floor_bytes (their own bytes, not their graphs'), with their graphs,
// are never candidates; its older ones and every other model's stay.
TEST(ReclaimOrder, TheRunningModelsInUseFloorIsNeverTaken) {
  const auto plan = [](std::uint32_t owner, std::uint64_t id, std::uint64_t bytes,
                       std::uint64_t used, bool running) {
    return ReclaimCandidate{.kind = ReclaimKind::kPlan,
                            .owner = owner,
                            .id = id,
                            .bytes = bytes,
                            .last_use = used,
                            .restore_seconds = 0.05,
                            .running = running};
  };
  const auto graph = [](std::uint32_t owner, std::uint64_t id, std::uint64_t bytes,
                        std::uint64_t used, bool running) {
    return ReclaimCandidate{.kind = ReclaimKind::kGraph,
                            .owner = owner,
                            .id = id,
                            .bytes = bytes,
                            .last_use = used,
                            .restore_seconds = 0.01,
                            .running = running};
  };
  const std::vector<ReclaimCandidate> all = {
      plan(0, 1, 100, 1, true),                                // the oldest: not in the floor
      plan(0, 2, 1100, 5, true),                               // 100 of its own beside its graph
      graph(0, 2, 1000, 5, true),   plan(0, 3, 100, 9, true),  // the newest
      plan(1, 4, 1100, 10, false),  // another model's: 100 of its own beside its graph
      graph(1, 4, 1000, 10, false),
  };
  // A floor of 150: the newest (100), then the next (100 of its own) reaches it.
  std::vector<ReclaimCandidate> c = all;
  ProtectFloor(c, 150);
  ASSERT_EQ(c.size(), 3U);
  EXPECT_EQ(c[0].id, 1U);
  EXPECT_EQ(c[1].id, 4U);
  EXPECT_EQ(c[2].id, 4U);
  // A floor its newest plan alone covers keeps only that one.
  c = all;
  ProtectFloor(c, 100);
  EXPECT_EQ(c.size(), all.size() - 1);
  EXPECT_TRUE(std::ranges::none_of(c, [](const ReclaimCandidate& x) { return x.id == 3; }));
  // A floor of 0 keeps nothing back; one past everything keeps every
  // running plan and graph back, never another model's.
  c = all;
  ProtectFloor(c, 0);
  EXPECT_EQ(c.size(), all.size());
  c = all;
  ProtectFloor(c, kGiB);
  ASSERT_EQ(c.size(), 2U);
  EXPECT_FALSE(c[0].running);
  EXPECT_FALSE(c[1].running);
  // So a reclaim asking for everything takes the rest, not the floor (the
  // other model's plan and graph counted once).
  c = all;
  ProtectFloor(c, 150);
  EXPECT_EQ(SelectReclaim(c, kGiB).bytes, 100U + 100U + 1000U);
}

// One reclaim's rounds: all of what is needed or nothing. A selection that
// cannot cover it takes nothing (with `partial`, what there is). A victim
// that gives back nothing is never offered again: the next round selects
// for the rest without it, and when the rest cannot be covered the reclaim
// ends short with what it took.
TEST(ReclaimOrder, AReclaimSelectsAgainWithoutAVictimThatGaveNothing) {
  const auto graph = [](std::uint64_t id) {
    return ReclaimCandidate{.kind = ReclaimKind::kGraph,
                            .owner = 0,
                            .id = id,
                            .bytes = 100,
                            .last_use = id,
                            .restore_seconds = 0.01};
  };
  std::vector<std::uint64_t> failing;
  std::vector<std::uint64_t> taken;
  std::vector<std::uint64_t> gone;
  int rounds = 0;
  const auto gather = [&](std::vector<ReclaimCandidate>& out, double&) {
    ++rounds;
    for (const std::uint64_t id : {1U, 2U, 3U}) {
      if (std::ranges::find(gone, id) == gone.end()) {
        out.push_back(graph(id));
      }
    }
  };
  const auto take = [&](const ReclaimCandidate& c) -> std::uint64_t {
    taken.push_back(c.id);
    if (std::ranges::find(failing, c.id) != failing.end()) {
      return 0;  // held: still a candidate next round, were it offered
    }
    gone.push_back(c.id);
    return c.bytes;
  };
  const auto reset = [&](std::vector<std::uint64_t> fail) {
    failing = std::move(fail);
    taken.clear();
    gone.clear();
    rounds = 0;
  };
  // Every victim gives back its count: the least recently used, once.
  reset({});
  auto run = RunReclaim(150, false, gather, take);
  EXPECT_EQ(run.freed, 200U);
  EXPECT_FALSE(run.short_of_need);
  EXPECT_THAT(taken, ElementsAre(1U, 2U));
  EXPECT_EQ(rounds, 1);
  // The oldest gives nothing: the next round covers the rest without it.
  reset({1});
  run = RunReclaim(150, false, gather, take);
  EXPECT_EQ(run.freed, 200U);
  EXPECT_FALSE(run.short_of_need);
  EXPECT_THAT(taken, ElementsAre(1U, 2U, 3U));
  // Two give nothing: what was taken stays, and the reclaim ends short.
  reset({1, 3});
  run = RunReclaim(150, false, gather, take);
  EXPECT_EQ(run.freed, 100U);
  EXPECT_TRUE(run.short_of_need);
  EXPECT_THAT(taken, ElementsAre(1U, 2U, 3U));
  // More than there is: nothing taken; with `partial`, all there is.
  reset({});
  run = RunReclaim(400, false, gather, take);
  EXPECT_EQ(run.freed, 0U);
  EXPECT_TRUE(run.short_of_need);
  EXPECT_TRUE(taken.empty());
  reset({});
  run = RunReclaim(400, true, gather, take);
  EXPECT_EQ(run.freed, 300U);
  EXPECT_TRUE(run.short_of_need);
  EXPECT_THAT(taken, ElementsAre(1U, 2U, 3U));
  // Nothing gives anything back: each victim tried once, then the one
  // left cannot cover the need and is not taken.
  reset({1, 2, 3});
  run = RunReclaim(150, false, gather, take);
  EXPECT_EQ(run.freed, 0U);
  EXPECT_TRUE(run.short_of_need);
  EXPECT_THAT(taken, ElementsAre(1U, 2U));
  EXPECT_EQ(rounds, 2);
  EXPECT_LE(rounds, jitllm::memory::kReclaimRounds);
}

}  // namespace
