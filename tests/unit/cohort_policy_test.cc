// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "scheduler/cohort_policy.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <deque>
#include <expected>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace llmp::scheduler {
namespace {
using namespace std::chrono_literals;
using Head = CohortPolicy::Head;
using Role = CohortPolicy::Role;
using Point = CohortPolicy::Point;
using Unit = CohortPolicy::Unit;
using Action = CohortPolicy::Action;
constexpr CohortPolicy::Clock::time_point kStart{};
constexpr auto kNever = CohortPolicy::Clock::time_point::max();
constexpr std::size_t kCapacity = 4;

// The queue at one look: its head, and where another model's first request waits.
CohortPolicy::Queue Q(Head head, std::optional<std::size_t> other = std::nullopt) {
  if (head == Head::kOtherModel) other = 0;
  return {.head = head, .other = other};
}

// One decision, as the executor (Server::ServeCooperative) makes it.
struct Step {
  Point point = Point::kPass;
  CohortPolicy::Queue queue{};
  std::chrono::seconds at{};         // kPass's now, from kStart
  std::size_t active = 1;            // kAdmit, kWait
  std::size_t attempts = 0;          // kAdmit
  bool release = false;              // kPass, kAdmit
  bool suspended_elsewhere = false;  // kWait
  bool stopped = false;              // kEnd
};
std::expected<Action, std::string> Ask(CohortPolicy& policy, const Step& step) {
  return policy.Decide({.point = step.point,
                        .queue = step.queue,
                        .active = step.active,
                        .capacity = kCapacity,
                        .attempts = step.attempts,
                        .suspended_elsewhere = step.suspended_elsewhere,
                        .release_pending = step.release,
                        .stopped = step.stopped,
                        .now = kStart + step.at});
}
// One the executor asks in its order: a decision, never an error.
Action Decide(CohortPolicy& policy, const Step& step) {
  auto action = Ask(policy, step);
  EXPECT_TRUE(action.has_value()) << action.error();
  return action.value_or(Action{});
}
Step Pass(CohortPolicy::Queue queue, std::chrono::seconds at = {}, bool release = false) {
  return {.point = Point::kPass, .queue = queue, .at = at, .release = release};
}
Step Admit(CohortPolicy::Queue queue, std::size_t active = 1, std::size_t attempts = 0,
           bool release = false) {
  return {.point = Point::kAdmit,
          .queue = queue,
          .active = active,
          .attempts = attempts,
          .release = release};
}
Step Wait(std::size_t active = 1, bool suspended_elsewhere = false) {
  return {.point = Point::kWait, .active = active, .suspended_elsewhere = suspended_elsewhere};
}
Step End(CohortPolicy::Queue queue, bool stopped = false) {
  return {.point = Point::kEnd, .queue = queue, .stopped = stopped};
}

// A table row: a step and the action expected of it (only the field the
// step's point decides is compared).
struct Row {
  Row(Step s, bool e = false, std::optional<std::chrono::seconds> w = std::nullopt)
      : step(s), expected(e), wake(w) {}
  Step step;
  bool expected = false;                     // yield, admit, blocks or substitute
  std::optional<std::chrono::seconds> wake;  // kWait: the turn's end, none: never
};
void Expect(CohortPolicy& policy, const std::vector<Row>& rows) {
  for (std::size_t i = 0; i < rows.size(); ++i) {
    SCOPED_TRACE(i);
    const Row& row = rows[i];
    const Action action = Decide(policy, row.step);
    switch (row.step.point) {
      case Point::kPass:
        EXPECT_EQ(action.yield_for_switch, row.expected);
        break;
      case Point::kAdmit:
        EXPECT_EQ(action.admit, row.expected);
        break;
      case Point::kWait:
        EXPECT_EQ(action.blocks_waiters, row.expected);
        EXPECT_EQ(action.wake, row.wake ? kStart + *row.wake : kNever);
        break;
      case Point::kEnd:
        EXPECT_EQ(action.run_substitute, row.expected);
        break;
    }
  }
}

TEST(CohortPolicyTest, RefillFollowsCurrentHeadAndCapacityWithoutRefusingWork) {
  CohortPolicy policy(30s, kStart);
  Expect(policy, {
                     {Pass(Q(Head::kCompatible))},
                     {Admit(Q(Head::kCompatible)), true},
                     {Admit(Q(Head::kCompatible), kCapacity), false},
                     {Admit(Q(Head::kCompatible), 1, kCapacity), false},
                     {Admit(Q(Head::kIncompatible)), false},
                     {Admit(Q(Head::kCompatible)), false},  // drains for the incompatible head
                     {Wait(), true},
                     {Pass(Q(Head::kCompatible))},  // the incompatible request departed
                     {Admit(Q(Head::kCompatible)), true},
                 });
  policy.Deferred();
  Expect(policy, {
                     {Admit(Q(Head::kCompatible)), false},
                     {Wait(), true},
                     {Pass(Q(Head::kEmpty))},
                     {Admit(Q(Head::kEmpty)), false},
                     {Wait(), false},
                     {Pass(Q(Head::kEmpty))},
                     {Wait(kCapacity), true},
                     {Pass(Q(Head::kEmpty))},
                     {Wait(1, true), true},
                 });
}

TEST(CohortPolicyTest, FullCohortYieldsAtItsTurnOnlyForAnotherModel) {
  CohortPolicy policy(30s, kStart);
  policy.UnitCompleted(kStart + 1s, Unit::kProgress);
  Expect(policy, {
                     {Pass(Q(Head::kCompatible), 60s), false},
                     {Pass(Q(Head::kOtherModel), 29s), false},
                     {Admit(Q(Head::kOtherModel)), false},
                     {Wait(), true, 30s},
                     {Pass(Q(Head::kOtherModel), 30s), true},
                     // A request disappearing before any owner retired cancels the switch.
                     {Pass(Q(Head::kEmpty), 60s), false},
                     {Wait(), false, std::nullopt},
                 });
}

TEST(CohortPolicyTest, AnotherModelBehindACompatibleHeadEndsTheTurnWithoutBlockingFifoAdmission) {
  CohortPolicy policy(30s, kStart);
  policy.UnitCompleted(kStart + 1s, Unit::kProgress);
  const auto behind = Q(Head::kCompatible, 2);
  Expect(policy, {
                     {Pass(behind, 29s), false},
                     // Admission stays FIFO: the compatible head arrived first.
                     {Admit(behind), true},
                     {Wait(2), false, 30s},
                     {Pass(behind, 30s), true},
                 });
  policy.ContinuationRetired();
  // The substitute is that request, wherever it waits.
  Expect(policy, {{End(behind), true}});
}

TEST(CohortPolicyTest, PartialRetirementFinishesDrainingEvenWhenCompetingClientLeaves) {
  const auto retired = [] {
    CohortPolicy policy(30s, kStart);
    policy.ContinuationRetired();
    return policy;
  };
  auto policy = retired();
  Expect(policy, {
                     {Pass(Q(Head::kEmpty)), true},
                     {Admit(Q(Head::kCompatible)), false},
                     {Wait(), true, std::nullopt},
                     {End(Q(Head::kEmpty)), false},
                 });
  auto same = retired();
  Expect(same, {{End(Q(Head::kCompatible)), false}});
  auto other = retired();
  Expect(other, {{End(Q(Head::kOtherModel)), true}});
}

TEST(CohortPolicyTest, CompetingHeadFoundDuringRefillArmsThePausedReaderDeadline) {
  CohortPolicy policy(30s, kStart);
  policy.UnitCompleted(kStart + 1s, Unit::kProgress);
  Expect(policy, {
                     {Pass(Q(Head::kCompatible))},
                     {Admit(Q(Head::kCompatible)), true},
                     // Starting that peer exposes B at the head, after the pass's first
                     // look. If every A reader pauses now, no completed unit will refresh it.
                     {Admit(Q(Head::kOtherModel), 2), false},
                     {Wait(2), true, 30s},
                     {Pass(Q(Head::kOtherModel), 30s), true},
                     {Pass(Q(Head::kEmpty), 30s), false},  // B may still leave before then
                     {Wait(2), false, std::nullopt},
                 });
}

TEST(CohortPolicyTest, SwapCostIsExcludedAndAResumedTurnMustAdvanceBeforeYielding) {
  CohortPolicy policy(30s, kStart);
  const auto other = Q(Head::kOtherModel);
  Expect(policy, {{Pass(other, 300s), false}});
  policy.UnitCompleted(kStart + 300s, Unit::kSwap);
  Expect(policy, {{Pass(other, 600s), false}});
  policy.UnitCompleted(kStart + 301s, Unit::kReadiness);
  Expect(policy, {{Pass(other, 600s), false}});
  policy.UnitCompleted(kStart + 301s, Unit::kProgress);
  Expect(policy, {{Pass(other, 329s), false}, {Pass(other, 330s), true}});
}

TEST(CohortPolicyTest, SubstituteBatchesInitialPeersThenYieldsToPausedOriginalWithoutQueueHead) {
  CohortPolicy policy(30s, kStart, Role::kSubstitute);
  Expect(policy, {
                     {Pass(Q(Head::kCompatible))},
                     {Admit(Q(Head::kCompatible)), true},
                     {Wait(2), false, std::nullopt},
                     {Pass(Q(Head::kCompatible))},
                     {Admit(Q(Head::kCompatible), 2), false},
                     {Wait(2), true, std::nullopt},
                 });
  policy.UnitCompleted(kStart + 1s, Unit::kProgress);
  Expect(policy, {
                     {Pass(Q(Head::kEmpty), 29s), false},
                     {Pass(Q(Head::kEmpty), 30s), true},
                     {Wait(2), true, 30s},
                 });
  policy.ContinuationRetired();
  // A third model never nests another substitute call (maximum depth two).
  Expect(policy, {{End(Q(Head::kOtherModel)), false}});
}

TEST(CohortPolicyTest, AnUnswitchablePausedMemberWaitsForReadersInsteadOfSpinning) {
  CohortPolicy policy(30s, kStart, Role::kSubstitute);
  policy.UnitCompleted(kStart + 1s, Unit::kProgress);
  Expect(policy, {{Pass(Q(Head::kEmpty), 30s), true}});
  policy.SwitchAttempted();  // backend refused; every member is now reader-paused
  Expect(policy, {
                     {Wait(), false, std::nullopt},
                     {Pass(Q(Head::kEmpty), 30s), true},
                     {Wait(), true, std::nullopt},
                 });
  policy.UnitCompleted(kStart + 31s, Unit::kProgress);
  Expect(policy, {{Pass(Q(Head::kEmpty), 31s), true}, {Wait(), true, 30s}});
}

// A pending release changes neither admission nor turns (the starvation
// bound comes first): the cohort refills and rotates as it would, and the
// release is answered busy at the turn's end.
TEST(CohortPolicyTest, APendingReleaseNeitherDrainsNorSuspendsTurns) {
  CohortPolicy policy(30s, kStart);
  policy.UnitCompleted(kStart + 1s, Unit::kProgress);
  const auto other = Q(Head::kOtherModel);
  // Another model waits past its turn: members pause for it with a release
  // waiting too, and the release is answered busy at that turn's end.
  auto pass = Decide(policy, Pass(other, 60s, true));
  EXPECT_TRUE(pass.yield_for_switch);
  EXPECT_TRUE(pass.release_busy);
  policy.SwitchAttempted();
  Expect(policy, {{Wait(), true, std::nullopt}});

  // Same model only: refills go on, and the wait for readers is bounded by
  // the turn's end, where the release is answered.
  CohortPolicy same(30s, kStart);
  same.UnitCompleted(kStart + 1s, Unit::kProgress);
  pass = Decide(same, Pass(Q(Head::kCompatible), 10s, true));
  EXPECT_FALSE(pass.yield_for_switch);
  EXPECT_FALSE(pass.release_busy);
  Expect(same, {
                   {Admit(Q(Head::kCompatible), 1, 0, true), true},
                   {Wait(2), false, 30s},
               });
  pass = Decide(same, Pass(Q(Head::kEmpty), 30s, true));
  EXPECT_FALSE(pass.yield_for_switch);  // no other model: nothing to pause for
  EXPECT_TRUE(pass.release_busy);
  Expect(same, {
                   {Wait(2), false, std::nullopt},  // answered: no turn end to wait for
                   {Pass(Q(Head::kEmpty), 31s), false},
               });

  // Before any useful work (a swap's), no turn runs out for it.
  CohortPolicy fresh(30s, kStart);
  pass = Decide(fresh, Pass(Q(Head::kCompatible), 60s, true));
  EXPECT_FALSE(pass.release_busy);
  Expect(fresh, {{Admit(Q(Head::kCompatible), 1, 0, true), true}, {Wait(2), false, std::nullopt}});
}

// A decision asked out of the executor's order is the executor's fault: an
// error the cohort ends on (Server::ServeCooperative's fatal), never an abort
// of the serving process.
TEST(CohortPolicyTest, ADecisionOutOfTheExecutorsOrderIsAnErrorNotAnAbort) {
  const auto refused = [](std::expected<Action, std::string> action) {
    ASSERT_FALSE(action.has_value());
    EXPECT_THAT(action.error(), ::testing::HasSubstr("out of its order"));
  };
  {
    CohortPolicy policy(30s, kStart);
    refused(Ask(policy, Admit(Q(Head::kCompatible))));
    refused(Ask(policy, Wait()));
    EXPECT_TRUE(Ask(policy, Pass(Q(Head::kEmpty))).has_value());  // the order still holds
  }
  {
    CohortPolicy policy(30s, kStart);
    (void)Decide(policy, Pass(Q(Head::kEmpty)));
    (void)Decide(policy, Admit(Q(Head::kEmpty)));
    refused(Ask(policy, Pass(Q(Head::kEmpty))));  // a pass's admissions end with its wait
    refused(Ask(policy, End(Q(Head::kEmpty))));
    EXPECT_TRUE(Ask(policy, Wait()).has_value());
  }
  {
    CohortPolicy policy(30s, kStart);
    (void)Decide(policy, End(Q(Head::kEmpty)));
    refused(Ask(policy, Pass(Q(Head::kEmpty))));
    refused(Ask(policy, End(Q(Head::kEmpty))));
  }
}

TEST(CohortPolicyTest, LookClassifiesTheHeadAndFindsAnotherModelAnywhere) {
  const auto key = [](const std::string& model) -> std::string_view { return model; };
  const auto supports = [](const std::string& model) { return model != "A-serial"; };
  const auto look = [&](std::deque<std::string> queue) {
    return CohortPolicy::Look(queue, std::string_view("A"), key, supports);
  };
  const auto empty = look({});
  EXPECT_EQ(empty.head, Head::kEmpty);
  EXPECT_FALSE(empty.other);
  const auto same = look({"A", "A"});
  EXPECT_EQ(same.head, Head::kCompatible);
  EXPECT_FALSE(same.other);
  const auto behind = look({"A", "A", "B", "C"});
  EXPECT_EQ(behind.head, Head::kCompatible);
  EXPECT_EQ(behind.other, 2U);
  const auto first = look({"B", "A"});
  EXPECT_EQ(first.head, Head::kOtherModel);
  EXPECT_EQ(first.other, 0U);
  // An incompatible head is classified by `supports`; the model key decides
  // whether a request is another model's.
  const auto serial = CohortPolicy::Look(std::deque<std::string>{"A-serial"},
                                         std::string_view("A-serial"), key, supports);
  EXPECT_EQ(serial.head, Head::kIncompatible);
  EXPECT_FALSE(serial.other);
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
  Expect(primary, {{End(Q(Head::kOtherModel), true), false}});
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
  // Each yielded substitute must rotate, rather than an A/B pair blocking C:
  // the primary runs every other turn, each other model one in 2n.
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
    primary.UnitCompleted(kStart + 1s, Unit::kProgress);
    ASSERT_TRUE(Decide(primary, Pass(Q(Head::kOtherModel), 30s)).yield_for_switch);
    primary.ContinuationRetired();
    ASSERT_TRUE(Decide(primary, End(Q(Head::kOtherModel))).run_substitute);
    CohortPolicy substitute(30s, kStart, Role::kSubstitute);
    std::deque<Request> b;
    b.push_back(std::move(queue.front()));
    queue.pop_front();
    ran.push_back(*b.front().owner);
    substitute.UnitCompleted(kStart + 1s, Unit::kProgress);
    ASSERT_TRUE(Decide(substitute, Pass(Q(Head::kOtherModel), 30s)).yield_for_switch);
    substitute.ContinuationRetired();
    ASSERT_FALSE(Decide(substitute, End(Q(Head::kOtherModel))).run_substitute);
    substitute.Restore(queue, b, kModel);
    primary.Restore(queue, a, kModel);
    ASSERT_EQ(queue.size(), 4U);
  }
  EXPECT_EQ(ran, (std::vector<int>{2, 3, 4, 2, 3, 4, 2, 3, 4}));
  EXPECT_EQ(Owners(queue), (std::vector<int>{1, 2, 3, 4}));
}

}  // namespace
}  // namespace llmp::scheduler
