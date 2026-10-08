// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// A chat cohort's state-capacity policy (runtime/cohort_capacity.h): who
// waits, who fails, who is preempted and restarted, and that a cohort
// sharing a capacity always finishes, failing only what cannot fit alone.

#include "runtime/cohort_capacity.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <optional>
#include <random>
#include <vector>

namespace {

namespace rt = llmp::runtime;
using rt::CapacityWait;
using ::testing::ElementsAre;
using ::testing::IsEmpty;

rt::Cohort Members(std::initializer_list<std::uint64_t> admitted) {
  rt::Cohort cohort{};
  std::size_t slot = 0;
  for (const std::uint64_t order : admitted) {
    if (order != 0) {
      cohort[slot] = {
          .present = true, .terminal = false, .admitted = order, .wait = CapacityWait::kNone};
    }
    ++slot;
  }
  return cohort;
}

TEST(CohortCapacity, ARefusalWaitsWhileAPeerHoldsState) {
  rt::Cohort cohort = Members({1, 2});
  const std::array<std::size_t, 1> refused = {1};
  const auto decision = rt::OnCapacityRefused(cohort, refused, false);
  EXPECT_THAT(decision.refuse, IsEmpty());
  EXPECT_FALSE(decision.preempt);
  EXPECT_EQ(cohort[1].wait, CapacityWait::kBlocked);
  EXPECT_FALSE(cohort[1].terminal);
  EXPECT_TRUE(rt::Runnable(cohort[0]));
  EXPECT_FALSE(rt::Runnable(cohort[1]));
  EXPECT_FALSE(rt::AdmissionOpen(cohort));  // FIFO: no newcomer before it
  EXPECT_FALSE(rt::NextRestart(cohort));    // its peer can still go on
  EXPECT_EQ(cohort[1].wait, CapacityWait::kBlocked);
  // Its peer finishes: it retries.
  cohort[0] = {};
  rt::OnMemberRetired(cohort);
  EXPECT_TRUE(rt::Runnable(cohort[1]));
  EXPECT_TRUE(rt::AdmissionOpen(cohort));
}

TEST(CohortCapacity, ARefusalWithNoPeerHoldingStateCannotFitAlone) {
  rt::Cohort alone = Members({1});
  const std::array<std::size_t, 1> first = {0};
  auto decision = rt::OnCapacityRefused(alone, first, false);
  EXPECT_THAT(decision.refuse, ElementsAre(0U));
  EXPECT_TRUE(alone[0].terminal);
  EXPECT_FALSE(decision.preempt);

  // A preempted peer holds no state: waiting for it frees nothing.
  rt::Cohort with_preempted = Members({1, 2});
  with_preempted[1].wait = CapacityWait::kPreempted;
  decision = rt::OnCapacityRefused(with_preempted, first, false);
  EXPECT_THAT(decision.refuse, ElementsAre(0U));
  EXPECT_FALSE(decision.preempt);
  EXPECT_EQ(with_preempted[1].wait, CapacityWait::kPreempted);
  // Once it retires, the preempted member restarts.
  with_preempted[0] = {};
  rt::OnMemberRetired(with_preempted);
  EXPECT_EQ(rt::NextRestart(with_preempted), 1U);
  EXPECT_TRUE(rt::Runnable(with_preempted[1]));
}

TEST(CohortCapacity, AMemberAboutToRetireIsWaitedForNotPreempted) {
  rt::Cohort cohort = Members({1, 2});
  cohort[0].terminal = true;  // cancelled or finished: its state frees at retirement
  const std::array<std::size_t, 1> refused = {1};
  const auto decision = rt::OnCapacityRefused(cohort, refused, false);
  EXPECT_THAT(decision.refuse, IsEmpty());
  EXPECT_FALSE(decision.preempt);
  EXPECT_EQ(cohort[1].wait, CapacityWait::kBlocked);
  EXPECT_TRUE(rt::AdmissionOpen(Members({1})));
}

TEST(CohortCapacity, WhenEveryHolderWaitsTheYoungestIsPreemptedAndTheRestRetry) {
  // Slots in admission order 3, 1, 4, 2: slot 2 is the youngest.
  rt::Cohort cohort = Members({3, 1, 4, 2});
  const std::array<std::size_t, 2> early = {0, 3};
  auto decision = rt::OnCapacityRefused(cohort, early, false);
  EXPECT_FALSE(decision.preempt);
  const std::array<std::size_t, 2> rest = {2, 1};
  decision = rt::OnCapacityRefused(cohort, rest, false);
  EXPECT_THAT(decision.refuse, IsEmpty());
  ASSERT_TRUE(decision.preempt);
  EXPECT_EQ(decision.preempt.value_or(rt::kCohortSlots), 2U);
  EXPECT_EQ(cohort[2].wait, CapacityWait::kPreempted);
  for (const std::size_t slot : {0U, 1U, 3U}) {
    EXPECT_TRUE(rt::Runnable(cohort[slot])) << slot;
  }
  EXPECT_FALSE(rt::AdmissionOpen(cohort));
  // The preempted member restarts only once no holder can go on.
  EXPECT_FALSE(rt::NextRestart(cohort));
  for (const std::size_t slot : {0U, 1U, 3U}) {
    cohort[slot] = {};
  }
  EXPECT_EQ(rt::NextRestart(cohort), 2U);
  EXPECT_TRUE(rt::Runnable(cohort[2]));
  EXPECT_TRUE(rt::AdmissionOpen(cohort));
}

TEST(CohortCapacity, ACancelledWaiterRetiresAndNeverBlocksAdmission) {
  rt::Cohort cohort = Members({1, 2, 3});
  const std::array<std::size_t, 1> refused = {2};
  (void)rt::OnCapacityRefused(cohort, refused, false);
  ASSERT_EQ(cohort[2].wait, CapacityWait::kBlocked);
  cohort[2].terminal = true;  // its client left, or its deadline passed
  EXPECT_TRUE(rt::AdmissionOpen(cohort));
  cohort[2] = {};
  rt::OnMemberRetired(cohort);
  EXPECT_TRUE(rt::Runnable(cohort[0]));
  EXPECT_TRUE(rt::Runnable(cohort[1]));
}

TEST(CohortCapacity, RestartsTheOldestPreemptedAfterWakingAnyWaiter) {
  rt::Cohort cohort = Members({5, 6, 7});
  cohort[0].wait = CapacityWait::kPreempted;
  cohort[1].wait = CapacityWait::kBlocked;
  cohort[2].wait = CapacityWait::kPreempted;
  EXPECT_FALSE(rt::NextRestart(cohort));  // the waiter retries first
  EXPECT_TRUE(rt::Runnable(cohort[1]));
  cohort[1] = {};
  EXPECT_EQ(rt::NextRestart(cohort), 0U);
  EXPECT_FALSE(rt::NextRestart(cohort));
  EXPECT_EQ(cohort[2].wait, CapacityWait::kPreempted);
}

TEST(CohortCapacity, IdleStateIsReclaimedBeforeAnyoneWaits) {
  rt::Cohort cohort = Members({1, 2});
  cohort[0].wait = CapacityWait::kBlocked;
  const std::array<std::size_t, 1> refused = {1};
  const auto decision = rt::OnCapacityRefused(cohort, refused, true);
  EXPECT_TRUE(decision.reclaim);
  EXPECT_THAT(decision.refuse, IsEmpty());
  EXPECT_FALSE(decision.preempt);
  EXPECT_TRUE(rt::Runnable(cohort[0]));
  EXPECT_TRUE(rt::Runnable(cohort[1]));
  // Alone, idle state is still reclaimed before a refusal.
  rt::Cohort alone = Members({1});
  const std::array<std::size_t, 1> only = {0};
  EXPECT_TRUE(rt::OnCapacityRefused(alone, only, true).reclaim);
  EXPECT_FALSE(alone[0].terminal);
}

// A cohort over a shared state capacity, as serve_api.cc drives it. Each
// branch holds cells: a member grows its branch a cell per unit to its
// need, and a unit that would pass the capacity is refused. Conversation
// state is never evicted: a retired member's cells stay on its idle branch
// until reclaimed, or cleared by the next member admitted there (its first
// unit); a preempted member's cells are discarded and it rebuilds from
// nothing. Every member that fits alone finishes, every one that does not
// is refused, nothing else fails, and the cohort always ends.
TEST(CohortCapacity, ASharedCapacityAlwaysFinishesAndFailsOnlyWhatCannotFitAlone) {
  std::mt19937 rng(20261002);  // NOLINT(bugprone-random-generator-seed): reproducible
  int waits = 0;
  int preemptions = 0;
  int refusals = 0;
  int reclaims = 0;
  for (int trial = 0; trial < 2000; ++trial) {
    const auto capacity = static_cast<int>(rng() % 24) + 4;
    const auto requests = static_cast<int>(rng() % 9) + 1;
    std::vector<int> need(static_cast<std::size_t>(requests));
    for (int& n : need) {
      n = static_cast<int>(rng() % static_cast<unsigned>(capacity + 6)) + 1;
    }
    std::deque<int> queue;
    for (int r = 0; r < requests; ++r) {
      queue.push_back(r);
    }
    rt::Cohort cohort{};
    std::array<int, rt::kCohortSlots> request{};
    std::array<int, rt::kCohortSlots> held{};      // cells on each branch
    std::array<int, rt::kCohortSlots> progress{};  // cells the member has built
    std::array<bool, rt::kCohortSlots> fresh{};    // admitted, its first unit not run
    std::array<bool, rt::kCohortSlots> failed{};
    std::vector<int> outcome(static_cast<std::size_t>(requests), 0);  // 1 done, -1 refused
    const std::size_t slots = (rng() % rt::kCohortSlots) + 1;
    std::uint64_t admissions = 0;
    std::size_t next = 0;
    int units = 0;
    const auto used = [&] {
      int total = 0;
      for (std::size_t s = 0; s < slots; ++s) {
        total += held[s];
      }
      return total;
    };
    const auto largest_idle = [&]() -> std::optional<std::size_t> {
      std::optional<std::size_t> idle;
      for (std::size_t s = 0; s < slots; ++s) {
        if (!cohort[s].present && held[s] > 0 && (!idle || held[s] > held[*idle])) {
          idle = s;
        }
      }
      return idle;
    };
    while (!queue.empty() || std::ranges::any_of(cohort, &rt::CohortMember::present)) {
      ASSERT_LT(++units, 100000) << "trial " << trial << ": the cohort never ended";
      // Retire terminal members (finished or refused), as the server does;
      // their cells stay on the branch.
      for (std::size_t s = 0; s < slots; ++s) {
        if (cohort[s].present && cohort[s].terminal) {
          outcome[static_cast<std::size_t>(request[s])] = failed[s] ? -1 : 1;
          cohort[s] = {};
          rt::OnMemberRetired(cohort);
          if (const auto restart = rt::NextRestart(cohort)) {
            progress[*restart] = 0;
          }
        }
      }
      // Admit while a slot is free and nobody waits.
      for (std::size_t s = 0; s < slots && !queue.empty(); ++s) {
        if (!cohort[s].present && rt::AdmissionOpen(cohort)) {
          request[s] = queue.front();
          queue.pop_front();
          cohort[s] = {.present = true,
                       .terminal = false,
                       .admitted = ++admissions,
                       .wait = CapacityWait::kNone};
          progress[s] = 0;
          fresh[s] = true;
          failed[s] = false;
        }
      }
      if (!std::ranges::any_of(cohort, &rt::CohortMember::present)) {
        continue;
      }
      // One unit: the next runnable member, round robin.
      std::optional<std::size_t> chosen;
      for (std::size_t k = 0; k < slots && !chosen; ++k) {
        const std::size_t s = (next + k) % slots;
        if (rt::Runnable(cohort[s])) {
          chosen = s;
        }
      }
      if (!chosen) {
        // Only terminal members remain: they retire next round.
        ASSERT_TRUE(std::ranges::any_of(
            cohort, [](const rt::CohortMember& m) { return m.present && m.terminal; }))
            << "trial " << trial << ": no member can go on";
        continue;
      }
      next = *chosen + 1;
      const std::size_t s = *chosen;
      if (fresh[s]) {
        fresh[s] = false;  // its reuse unit clears the branch (no prefix reused)
        held[s] = 0;
        continue;
      }
      if (used() + 1 > capacity) {
        const std::array<std::size_t, 1> refused = {s};
        const auto idle = largest_idle();
        const auto decision = rt::OnCapacityRefused(cohort, refused, idle.has_value());
        if (decision.reclaim) {
          ASSERT_TRUE(idle);
          held[idle.value_or(0)] = 0;
          ++reclaims;
          continue;
        }
        waits += cohort[s].wait == CapacityWait::kBlocked ? 1 : 0;
        preemptions += decision.preempt ? 1 : 0;
        refusals += static_cast<int>(decision.refuse.size());
        for (const std::size_t r : decision.refuse) {
          failed[r] = true;
          // Refused only while alone: no other branch holds a cell.
          EXPECT_EQ(used(), held[r]) << "trial " << trial;
          EXPECT_GT(need[static_cast<std::size_t>(request[r])], capacity) << "trial " << trial;
        }
        if (decision.preempt) {
          held[*decision.preempt] = 0;  // the preemption clears its branch
        }
        if (const auto restart = rt::NextRestart(cohort)) {
          progress[*restart] = 0;
        }
        continue;
      }
      ++held[s];
      if (++progress[s] >= need[static_cast<std::size_t>(request[s])]) {
        cohort[s].terminal = true;
      }
    }
    for (int r = 0; r < requests; ++r) {
      EXPECT_EQ(outcome[static_cast<std::size_t>(r)],
                need[static_cast<std::size_t>(r)] > capacity ? -1 : 1)
          << "trial " << trial << " request " << r;
    }
  }
  // The trials exercised every path.
  EXPECT_GT(waits, 100);
  EXPECT_GT(preemptions, 100);
  EXPECT_GT(refusals, 100);
  EXPECT_GT(reclaims, 100);
}

// Admission by memory (docs/runtime-serving.md#request-slots): a prompt's
// state to come is its state less what its branch holds; once a request
// found no room beside its peers none is looked at again until a member
// retires, and a request alone always starts.
TEST(CohortCapacity, AdmissionByMemoryWaitsForARetirementNeverAlone) {
  EXPECT_EQ(rt::StateToCome(700, 200), 500U);
  EXPECT_EQ(rt::StateToCome(200, 700), 0U);  // a shorter prompt on a fuller branch
  rt::MemoryWait wait;
  EXPECT_FALSE(wait.Blocked(3));
  wait.NoRoom();
  EXPECT_TRUE(wait.Blocked(3));
  EXPECT_FALSE(wait.Blocked(0));  // alone, it starts
  wait.MemberRetired();
  EXPECT_FALSE(wait.Blocked(2));
}

}  // namespace
