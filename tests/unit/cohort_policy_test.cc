// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "scheduler/cohort_policy.h"

#include <gtest/gtest.h>

#include <chrono>
#include <deque>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace llmp::scheduler {
namespace {
using namespace std::chrono_literals;
using Head = CohortPolicy::Head;
using Role = CohortPolicy::Role;
constexpr CohortPolicy::Clock::time_point kStart{};

TEST(CohortPolicyTest, RefillFollowsCurrentHeadAndCapacityWithoutRefusingWork) {
  CohortPolicy policy(30s, kStart);
  policy.BeginPass(Head::kCompatible);
  EXPECT_TRUE(policy.CanRefill(1, 4, 0));
  EXPECT_FALSE(policy.CanRefill(4, 4, 0));
  EXPECT_FALSE(policy.CanRefill(1, 4, 4));
  EXPECT_FALSE(policy.AdmitHead(Head::kIncompatible));
  EXPECT_TRUE(policy.BlocksWaiters(false, false));
  EXPECT_FALSE(policy.CanRefill(1, 4, 0));
  policy.BeginPass(Head::kCompatible);  // incompatible request departed
  EXPECT_TRUE(policy.CanRefill(1, 4, 0));
  policy.Deferred();
  EXPECT_FALSE(policy.CanRefill(1, 4, 0));
  EXPECT_TRUE(policy.BlocksWaiters(false, false));
  policy.BeginPass(Head::kEmpty);
  EXPECT_FALSE(policy.AdmitHead(Head::kEmpty));
  EXPECT_FALSE(policy.BlocksWaiters(false, false));
  EXPECT_TRUE(policy.BlocksWaiters(true, false));
  EXPECT_TRUE(policy.BlocksWaiters(false, true));
}

TEST(CohortPolicyTest, FullCohortYieldsAtItsTurnOnlyForAnotherModel) {
  CohortPolicy policy(30s, kStart);
  policy.UnitCompleted(kStart + 1s, CohortPolicy::Unit::kProgress);
  policy.BeginPass(Head::kCompatible);
  EXPECT_FALSE(policy.YieldForSwitch(kStart + 60s));
  policy.BeginPass(Head::kOtherModel);
  EXPECT_FALSE(policy.CanRefill(1, 4, 0));
  EXPECT_FALSE(policy.YieldForSwitch(kStart + 29s));
  EXPECT_TRUE(policy.YieldForSwitch(kStart + 30s));
  EXPECT_EQ(policy.SwitchDeadline(), kStart + 30s);
  // A request disappearing before any owner retired cancels the switch.
  policy.BeginPass(Head::kEmpty);
  EXPECT_FALSE(policy.YieldForSwitch(kStart + 60s));
  EXPECT_EQ(policy.SwitchDeadline(), CohortPolicy::Clock::time_point::max());
}

TEST(CohortPolicyTest, PartialRetirementFinishesDrainingEvenWhenCompetingClientLeaves) {
  CohortPolicy policy(30s, kStart);
  policy.ContinuationRetired();
  policy.BeginPass(Head::kEmpty);
  EXPECT_TRUE(policy.YieldForSwitch(kStart));
  EXPECT_FALSE(policy.CanRefill(1, 4, 0));
  EXPECT_FALSE(policy.RunSubstitute(Head::kEmpty, false));
  EXPECT_FALSE(policy.RunSubstitute(Head::kCompatible, false));
  EXPECT_TRUE(policy.RunSubstitute(Head::kOtherModel, false));
}

TEST(CohortPolicyTest, CompetingHeadFoundDuringRefillArmsThePausedReaderDeadline) {
  CohortPolicy policy(30s, kStart);
  policy.UnitCompleted(kStart + 1s, CohortPolicy::Unit::kProgress);
  policy.BeginPass(Head::kCompatible);
  ASSERT_TRUE(policy.AdmitHead(Head::kCompatible));
  // Starting that peer exposes B at the head, after the pass's first look.
  // If every A reader pauses now, no completed unit will refresh the head.
  EXPECT_FALSE(policy.AdmitHead(Head::kOtherModel));
  EXPECT_EQ(policy.SwitchDeadline(), kStart + 30s);
  EXPECT_TRUE(policy.YieldForSwitch(kStart + 30s));
  policy.BeginPass(Head::kEmpty);  // B may still leave before the deadline.
  EXPECT_EQ(policy.SwitchDeadline(), CohortPolicy::Clock::time_point::max());
}

TEST(CohortPolicyTest, SwapCostIsExcludedAndAResumedTurnMustAdvanceBeforeYielding) {
  CohortPolicy policy(30s, kStart);
  policy.BeginPass(Head::kOtherModel);
  EXPECT_FALSE(policy.YieldForSwitch(kStart + 300s));
  policy.UnitCompleted(kStart + 300s, CohortPolicy::Unit::kSwap);
  EXPECT_FALSE(policy.YieldForSwitch(kStart + 600s));
  policy.UnitCompleted(kStart + 301s, CohortPolicy::Unit::kReadiness);
  EXPECT_FALSE(policy.YieldForSwitch(kStart + 600s));
  policy.UnitCompleted(kStart + 301s, CohortPolicy::Unit::kProgress);
  EXPECT_FALSE(policy.YieldForSwitch(kStart + 329s));
  EXPECT_TRUE(policy.YieldForSwitch(kStart + 330s));
}

TEST(CohortPolicyTest, SubstituteBatchesInitialPeersThenYieldsToPausedOriginalWithoutQueueHead) {
  CohortPolicy policy(30s, kStart, Role::kSubstitute);
  policy.BeginPass(Head::kCompatible);
  EXPECT_TRUE(policy.CanRefill(1, 4, 0));
  EXPECT_TRUE(policy.AdmitHead(Head::kCompatible));
  policy.AdmissionFinished();
  policy.BeginPass(Head::kCompatible);
  EXPECT_FALSE(policy.CanRefill(1, 4, 0));
  policy.UnitCompleted(kStart + 1s, CohortPolicy::Unit::kProgress);
  policy.BeginPass(Head::kEmpty);
  EXPECT_FALSE(policy.YieldForSwitch(kStart + 29s));
  EXPECT_TRUE(policy.YieldForSwitch(kStart + 30s));
  EXPECT_EQ(policy.SwitchDeadline(), kStart + 30s);
  policy.ContinuationRetired();
  // A third model never nests another substitute call (maximum depth two).
  EXPECT_FALSE(policy.RunSubstitute(Head::kOtherModel, false));
}

TEST(CohortPolicyTest, AnUnswitchablePausedMemberWaitsForReadersInsteadOfSpinning) {
  CohortPolicy policy(30s, kStart, Role::kSubstitute);
  policy.UnitCompleted(kStart + 1s, CohortPolicy::Unit::kProgress);
  EXPECT_TRUE(policy.YieldForSwitch(kStart + 30s));
  policy.SwitchAttempted();  // backend refused; every member is now reader-paused
  EXPECT_EQ(policy.SwitchDeadline(), CohortPolicy::Clock::time_point::max());
  policy.BeginPass(Head::kEmpty);
  EXPECT_EQ(policy.SwitchDeadline(), CohortPolicy::Clock::time_point::max());
  policy.UnitCompleted(kStart + 31s, CohortPolicy::Unit::kProgress);
  EXPECT_TRUE(policy.YieldForSwitch(kStart + 31s));
  EXPECT_EQ(policy.SwitchDeadline(), kStart + 30s);
}

struct Request {
  std::string model;
  std::unique_ptr<int> owner;
};
Request Owned(std::string model, int id) {
  return {.model = std::move(model), .owner = std::make_unique<int>(id)};
}
const auto kModel = [](const Request& request) -> std::string_view { return request.model; };
std::vector<int> Owners(const std::deque<Request>& requests) {
  std::vector<int> owners;
  for (const auto& request : requests) {
    owners.push_back(*request.owner);
  }
  return owners;
}

TEST(CohortPolicyTest, FailedOrStoppedTurnRestoresAllRetiredOwnersForNormalCancellation) {
  CohortPolicy primary(30s, kStart);
  primary.ContinuationRetired();
  EXPECT_FALSE(primary.RunSubstitute(Head::kOtherModel, true));
  std::deque<Request> queue;
  queue.push_back(Owned("B", 3));
  std::deque<Request> suspended;
  suspended.push_back(Owned("A", 1));
  suspended.push_back(Owned("A", 2));
  primary.Restore(queue, suspended, kModel);
  EXPECT_TRUE(suspended.empty());
  EXPECT_EQ(Owners(queue), (std::vector<int>{1, 2, 3}));
}

TEST(CohortPolicyTest, YieldedSubstituteRotatesBehindOtherModelsAheadOfItsFreshArrivals) {
  CohortPolicy primary(30s, kStart);
  CohortPolicy substitute(30s, kStart, Role::kSubstitute);
  std::deque<Request> queue;
  queue.push_back(Owned("B", 5));
  queue.push_back(Owned("C", 6));
  queue.push_back(Owned("B", 7));
  queue.push_back(Owned("D", 8));
  std::deque<Request> a;
  a.push_back(Owned("A", 1));
  a.push_back(Owned("A", 2));
  std::deque<Request> b;
  b.push_back(Owned("B", 3));
  b.push_back(Owned("B", 4));
  substitute.Restore(queue, b, kModel);
  primary.Restore(queue, a, kModel);
  EXPECT_EQ(Owners(queue), (std::vector<int>{1, 2, 6, 8, 3, 4, 5, 7}));
  EXPECT_TRUE(a.empty());
  EXPECT_TRUE(b.empty());
}

TEST(CohortPolicyTest, RepeatedModelTurnsKeepAllContinuationsAndReachEveryWaitingCohort) {
  // A remains the primary; B/C/D are indefinitely long substitute cohorts.
  // Each yielded substitute must rotate, rather than an A/B pair blocking C.
  std::deque<Request> queue;
  for (int id = 1; id <= 4; ++id) {
    queue.push_back(Owned(std::string(1, static_cast<char>('A' + id - 1)), id));
  }
  std::vector<int> ran;
  for (int turn = 0; turn < 9; ++turn) {
    CohortPolicy primary(30s, kStart);
    std::deque<Request> a;
    a.push_back(std::move(queue.front()));
    queue.pop_front();
    ASSERT_EQ(*a.front().owner, 1);
    primary.UnitCompleted(kStart + 1s, CohortPolicy::Unit::kProgress);
    primary.BeginPass(Head::kOtherModel);
    ASSERT_TRUE(primary.YieldForSwitch(kStart + 30s));
    primary.ContinuationRetired();
    ASSERT_TRUE(primary.RunSubstitute(Head::kOtherModel, false));
    CohortPolicy substitute(30s, kStart, Role::kSubstitute);
    std::deque<Request> b;
    b.push_back(std::move(queue.front()));
    queue.pop_front();
    ran.push_back(*b.front().owner);
    substitute.UnitCompleted(kStart + 1s, CohortPolicy::Unit::kProgress);
    ASSERT_TRUE(substitute.YieldForSwitch(kStart + 30s));
    substitute.ContinuationRetired();
    ASSERT_FALSE(substitute.RunSubstitute(Head::kOtherModel, false));
    substitute.Restore(queue, b, kModel);
    primary.Restore(queue, a, kModel);
    ASSERT_EQ(queue.size(), 4U);
  }
  EXPECT_EQ(ran, (std::vector<int>{2, 3, 4, 2, 3, 4, 2, 3, 4}));
  EXPECT_EQ(Owners(queue), (std::vector<int>{1, 2, 3, 4}));
}

}  // namespace
}  // namespace llmp::scheduler
