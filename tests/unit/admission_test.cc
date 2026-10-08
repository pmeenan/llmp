// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Admission and the switching policy against the M2 rows of D-050's
// adversarial matrix and D-069's pause cases
// (docs/reservation-policy.md#worked-cases-and-implementation-gates), on
// logical time with no backend.

#include "scheduler/admission.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <optional>
#include <vector>

#include "base/bytes.h"
#include "expected_error.h"
#include "memory/commitment.h"

namespace {

using llmp::base::Bytes;
using llmp::memory::CommitmentError;
using llmp::scheduler::Admission;
using llmp::scheduler::AdmissionError;
using llmp::scheduler::Decision;
using llmp::scheduler::PolicySettings;
using llmp::scheduler::RequestClass;
using llmp::scheduler::RequestId;
using llmp::scheduler::RequestSpec;
using llmp::scheduler::RequestState;
using llmp::scheduler::SwitchingPolicy;
using llmp::test_support::Failed;
using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::Pair;

RequestSpec Spec(std::uint64_t retained, std::uint64_t phase,
                 RequestClass request_class = RequestClass::kInteractive, std::uint64_t work = 10) {
  return {.request_class = request_class,
          .envelope = {.retained = Bytes(retained), .phase = Bytes(phase)},
          .work = work,
          .switch_cost = 1,
          .deadline = std::nullopt};
}

constexpr llmp::catalog::DomainId kSpark(0, 1);

llmp::memory::CommitmentLedger LedgerWith(std::uint64_t budget) {
  llmp::memory::CommitmentLedger ledger;
  EXPECT_TRUE(ledger.AddDomain(kSpark, Bytes(budget)).has_value());
  return ledger;
}

// A ledger for one domain, with admission over it.
struct Node {
  explicit Node(std::uint64_t budget, PolicySettings settings = {})
      : ledger(LedgerWith(budget)), admission(ledger, kSpark, settings) {}
  llmp::memory::CommitmentLedger ledger;
  Admission admission;
};

RequestId Admit(Admission& admission, const RequestSpec& spec, std::uint64_t now = 0) {
  auto submitted = admission.Submit(spec, now);
  EXPECT_TRUE(submitted.has_value()) << (submitted ? "" : ToString(submitted.error()));
  return submitted ? submitted->id : RequestId{};
}

// B=100, F=10: A and B (20, 50) both admitted; A holds the slot through all
// its phases and B, of the same class, starts only after A retires (no
// per-step alternation). A third (1, 1) is deferred; (41, 50) is impossible.
TEST(Admission, SameClassRunsToCompletion) {
  Node node(100, PolicySettings{});
  Admission& admission = node.admission;
  ASSERT_TRUE(admission.AddFixed(Bytes(10), 0).has_value());
  const RequestId a = Admit(admission, Spec(20, 50));
  const RequestId b = Admit(admission, Spec(20, 50));
  EXPECT_THAT(admission.Running(), ElementsAre(a));
  EXPECT_EQ(admission.StateOf(b), RequestState::kWaiting);
  for (std::uint64_t tick = 1; tick < 10; ++tick) {
    EXPECT_FALSE(admission.Boundary(a, tick, tick).value().paused.has_value());
  }
  const RequestId c = Admit(admission, Spec(1, 1));
  EXPECT_EQ(admission.StateOf(c), RequestState::kQueued);  // 101 with A and B
  EXPECT_EQ(Failed(admission.Submit(Spec(41, 50), 10)), AdmissionError::kImpossible);
  const auto decision = admission.Retire(a, 10).value();
  EXPECT_EQ(decision.run, b);
  EXPECT_EQ(admission.StateOf(c), RequestState::kWaiting);  // admitted once A's allowance is gone
}

// An interactive request pauses a background one only at a completed
// boundary, after its minimum run; the paused request's state stays
// charged; the paused request resumes next, ahead of other waiting work.
TEST(Admission, InteractivePausesBackgroundAtABoundary) {
  PolicySettings settings{.minimum_run = 5, .pause_cap = 1};
  Node node(100, settings);
  Admission& admission = node.admission;
  const RequestId a = Admit(admission, Spec(20, 50, RequestClass::kBackground, 100));
  const RequestId b = Admit(admission, Spec(20, 50, RequestClass::kInteractive), 1);
  EXPECT_FALSE(admission.Boundary(a, 1, 3).value().paused.has_value());  // before its minimum run
  auto decision = admission.Boundary(a, 2, 6).value();
  EXPECT_EQ(decision.paused, a);
  EXPECT_EQ(decision.run, b);
  EXPECT_EQ(admission.StateOf(a), RequestState::kPaused);
  // Still charged: R(G) holds both retained bounds.
  EXPECT_EQ(admission.Totals().retained, Bytes(40));
  // Another interactive request waits behind the resumption.
  const RequestId c = Admit(admission, Spec(10, 10, RequestClass::kInteractive), 7);
  decision = admission.Retire(b, 8).value();
  EXPECT_EQ(decision.run, a);
  EXPECT_EQ(admission.StateOf(c), RequestState::kWaiting);
  // The pause cap: A is not paused again.
  EXPECT_FALSE(admission.Boundary(a, 3, 20).value().paused.has_value());
}

TEST(Admission, RunToCompletionNeverPauses) {
  Node node(100, PolicySettings{.policy = SwitchingPolicy::kRunToCompletion});
  Admission& admission = node.admission;
  const RequestId a = Admit(admission, Spec(20, 50, RequestClass::kBackground));
  (void)Admit(admission, Spec(20, 50, RequestClass::kInteractive));
  EXPECT_FALSE(admission.Boundary(a, 1, 100).value().paused.has_value());
}

// Time-slicing alternates, bounded by the quantum and the pause cap.
TEST(Admission, TimeSlicingIsBounded) {
  Node node(100, PolicySettings{
                     .policy = SwitchingPolicy::kTimeSlicing, .pause_cap = 2, .time_slice = 4});
  Admission& admission = node.admission;
  const RequestId a = Admit(admission, Spec(20, 30));
  const RequestId b = Admit(admission, Spec(20, 30));
  EXPECT_FALSE(admission.Boundary(a, 1, 2).value().paused.has_value());  // within the quantum
  EXPECT_EQ(admission.Boundary(a, 2, 4).value().paused, a);
  // B is A's substitute: it is not paused while A is paused for it.
  EXPECT_FALSE(admission.Boundary(b, 1, 100).value().paused.has_value());
  EXPECT_EQ(admission.Retire(b, 101).value().run, a);
}

// A request with a known deadline is not paused if its remaining work, the
// substitute's program and both switches would miss it.
TEST(Admission, APauseMayNotCostTheDeadline) {
  Node node(100, PolicySettings{});
  Admission& admission = node.admission;
  RequestSpec background = Spec(20, 50, RequestClass::kBackground, 20);
  background.deadline = 30;
  const RequestId a = Admit(admission, background);
  (void)Admit(admission, Spec(20, 50, RequestClass::kInteractive, 10), 1);
  // At tick 5 with 5 done: 15 remaining + 10 + 1 + 1 = 27, finishing at 32 > 30.
  EXPECT_FALSE(admission.Boundary(a, 5, 5).value().paused.has_value());
  // At tick 5 with 10 done: 10 + 10 + 2 = 22, finishing at 27.
  EXPECT_EQ(admission.Boundary(a, 10, 5).value().paused, a);
}

// An unservable known deadline is refused before admission, never after.
TEST(Admission, AnUnmeetableDeadlineIsRefusedUpFront) {
  Node node(100, PolicySettings{});
  Admission& admission = node.admission;
  (void)Admit(admission, Spec(20, 50, RequestClass::kInteractive, 50));
  RequestSpec late = Spec(10, 10, RequestClass::kInteractive, 10);
  late.deadline = 40;
  EXPECT_EQ(Failed(admission.Submit(late, 0)), AdmissionError::kDeadline);
  late.deadline = 70;
  EXPECT_TRUE(admission.Submit(late, 0).has_value());
}

// B=100, F+R(G)=10 (here, F=10 with zero retained), cohort A (80) and C
// (10); A paused for substitute B (30). C's replacement to 50 passes the
// serial and cohort checks (10 + 30 + 50) but would break the promised
// resumption (10 + 80 + 50): it waits. So do a grant, an F or J increase
// and a budget reduction.
TEST(Admission, ChangesDuringAPauseKeepTheResumption) {
  Node node(100, PolicySettings{});
  Admission& admission = node.admission;
  ASSERT_TRUE(admission.AddFixed(Bytes(10), 0).has_value());
  const RequestId a = Admit(admission, Spec(0, 80, RequestClass::kBackground));
  const RequestId c = Admit(admission, Spec(0, 10, RequestClass::kBackground));
  ASSERT_TRUE(admission.Join(c, 0).has_value());
  EXPECT_THAT(admission.Running(), ElementsAre(a, c));
  const RequestId b = Admit(admission, Spec(0, 30, RequestClass::kInteractive), 1);
  EXPECT_EQ(admission.Boundary(a, 1, 2).value().paused, a);
  EXPECT_THAT(admission.Running(), ElementsAre(c, b));
  EXPECT_EQ(Failed(admission.ReplaceEnvelope(c, {.retained = Bytes(0), .phase = Bytes(50)}, 0)),
            AdmissionError::kDeferred);
  EXPECT_EQ(Failed(admission.AddFixed(Bytes(1), 0)), AdmissionError::kDeferred);
  EXPECT_EQ(Failed(admission.AddBackground(Bytes(1), 0)), AdmissionError::kDeferred);
  EXPECT_EQ(Failed(admission.SetBudget(Bytes(99), 0)), AdmissionError::kDeferred);
  const RequestId d = Admit(admission, Spec(1, 1, RequestClass::kBackground), 3);
  EXPECT_EQ(admission.StateOf(d), RequestState::kQueued);
  // After the resumption the same change is judged normally (and fails).
  EXPECT_EQ(admission.Retire(b, 4).value().run, a);
  EXPECT_EQ(Failed(admission.ReplaceEnvelope(c, {.retained = Bytes(0), .phase = Bytes(50)}, 0)),
            AdmissionError::kRefused);
}

// The substitute's own change is checked without its allowance and, if it
// still fails, refused, never deferred; a second pause is not taken while
// one is open.
TEST(Admission, TheSubstitutesChangesAreRefusedNotDeferred) {
  Node node(100, PolicySettings{});
  Admission& admission = node.admission;
  const RequestId a = Admit(admission, Spec(20, 40, RequestClass::kBackground));
  const RequestId b = Admit(admission, Spec(20, 40, RequestClass::kInteractive), 1);
  EXPECT_EQ(admission.Boundary(a, 1, 2).value().paused, a);
  EXPECT_EQ(Failed(admission.ReplaceEnvelope(b, {.retained = Bytes(20), .phase = Bytes(70)}, 0)),
            AdmissionError::kRefused);
  EXPECT_TRUE(
      admission.ReplaceEnvelope(b, {.retained = Bytes(20), .phase = Bytes(60)}, 0).has_value());
  EXPECT_FALSE(admission.Boundary(b, 1, 3).value().paused.has_value());
}

// D-050's row: while one pause is open, a still-running cohort member is
// not paused, although an interactive request waits whose phase would fit
// in its place. Once the first pause has closed, the same boundary pauses
// it (the control: the guard, not the fit, held it).
TEST(Admission, ASecondPauseIsNotTakenWhileOneIsOpen) {
  Node node(100, PolicySettings{});
  Admission& admission = node.admission;
  ASSERT_TRUE(admission.AddFixed(Bytes(10), 0).has_value());
  const RequestId a = Admit(admission, Spec(0, 60, RequestClass::kBackground));
  const RequestId c = Admit(admission, Spec(0, 10, RequestClass::kBackground));
  ASSERT_TRUE(admission.Join(c, 0).has_value());
  const RequestId b = Admit(admission, Spec(0, 20, RequestClass::kInteractive), 1);
  ASSERT_EQ(admission.Boundary(a, 1, 2).value().paused, a);
  const RequestId e = Admit(admission, Spec(0, 5, RequestClass::kInteractive), 3);
  EXPECT_EQ(admission.StateOf(e), RequestState::kWaiting);
  EXPECT_FALSE(admission.Boundary(c, 1, 4).value().paused.has_value());
  EXPECT_EQ(admission.StateOf(c), RequestState::kRunning);
  EXPECT_EQ(admission.StateOf(a), RequestState::kPaused);
  // B retires and A resumes next: no pause is open any more.
  EXPECT_EQ(admission.Retire(b, 5).value().run, a);
  EXPECT_EQ(admission.Boundary(c, 1, 6).value().paused, c);
}

// D-050's "continuation expires while an admitted request is suspended"
// (its M2 lifetime part; expiry itself is M6's retention): nothing in
// admission ages an admitted request's allowance out. However long a
// pause lasts, the paused request's retained allowance stays committed, a
// newcomer that would need it waits, and the paused request resumes.
TEST(Admission, APausedRequestKeepsItsAllowanceHoweverLongItWaits) {
  Node node(100, PolicySettings{});
  Admission& admission = node.admission;
  ASSERT_TRUE(admission.AddFixed(Bytes(10), 0).has_value());
  const RequestId a = Admit(admission, Spec(30, 30, RequestClass::kBackground));
  const RequestId b = Admit(admission, Spec(10, 20, RequestClass::kInteractive), 1);
  ASSERT_EQ(admission.Boundary(a, 1, 2).value().paused, a);
  const Bytes retained = admission.Totals().retained;
  EXPECT_EQ(retained, Bytes(40));
  constexpr std::uint64_t kLater = 1'000'000'000;
  const RequestId d = Admit(admission, Spec(30, 30, RequestClass::kBackground), kLater);
  EXPECT_EQ(admission.StateOf(d), RequestState::kQueued);
  EXPECT_EQ(admission.StateOf(a), RequestState::kPaused);
  EXPECT_EQ(admission.Totals().retained, retained);
  EXPECT_EQ(admission.Retire(b, kLater + 1).value().run, a);
  EXPECT_EQ(admission.StateOf(a), RequestState::kRunning);
}

// Endless higher-class arrivals cannot starve a waiting background request.
TEST(Admission, AgingPreventsStarvation) {
  Node node(1000, PolicySettings{.policy = SwitchingPolicy::kRunToCompletion, .aging_limit = 2});
  Admission& admission = node.admission;
  RequestId running = Admit(admission, Spec(10, 10, RequestClass::kInteractive));
  const RequestId background = Admit(admission, Spec(10, 10, RequestClass::kBackground));
  std::uint64_t now = 1;
  int served_interactive = 0;
  while (admission.StateOf(background) != RequestState::kRunning) {
    (void)Admit(admission, Spec(10, 10, RequestClass::kInteractive), now);
    running = admission.Retire(running, ++now).value().run.value_or(RequestId{});
    ++served_interactive;
    ASSERT_LT(served_interactive, 10);
  }
  EXPECT_LE(served_interactive, 3);
}

TEST(Admission, RetirementIsNeverRefused) {
  Node node(100, PolicySettings{});
  Admission& admission = node.admission;
  const RequestId a = Admit(admission, Spec(20, 50, RequestClass::kBackground));
  const RequestId b = Admit(admission, Spec(20, 50, RequestClass::kInteractive), 1);
  EXPECT_EQ(admission.Boundary(a, 1, 2).value().paused, a);
  // The paused request ends while paused: its substitute runs on.
  EXPECT_FALSE(admission.Retire(a, 3).value().run.has_value());
  EXPECT_FALSE(admission.PausedRequest().has_value());
  EXPECT_TRUE(admission.Retire(b, 4).has_value());
  EXPECT_EQ(Failed(admission.Retire(b, 5)), AdmissionError::kUnknownRequest);
  EXPECT_THAT(admission.Running(), IsEmpty());
  EXPECT_EQ(admission.Totals().required, Bytes(0));
}

TEST(Admission, TheQueueIsBounded) {
  Node node(100, PolicySettings{.queue_limit = 1});
  Admission& admission = node.admission;
  (void)Admit(admission, Spec(50, 50));
  (void)Admit(admission, Spec(50, 50));  // queued
  EXPECT_EQ(Failed(admission.Submit(Spec(50, 50), 1)), AdmissionError::kQueueFull);
}

// A queued request that can no longer fit alone, after a budget reduction
// or a larger F, is refused explicitly instead of blocking the queue.
TEST(Admission, QueuedRequestsThatCanNoLongerRunAreRefused) {
  Node node(100, PolicySettings{});
  Admission& admission = node.admission;
  const RequestId x = Admit(admission, Spec(30, 50));
  const RequestId q = Admit(admission, Spec(40, 50), 1);  // 120 with X: queued
  const RequestId small = Admit(admission, Spec(1, 1), 2);
  EXPECT_EQ(admission.StateOf(q), RequestState::kQueued);
  const Decision decision = admission.SetBudget(Bytes(80), 3).value();
  EXPECT_THAT(decision.refused, ElementsAre(Pair(q, AdmissionError::kImpossible)));
  EXPECT_FALSE(admission.StateOf(q).has_value());
  EXPECT_EQ(admission.StateOf(small), RequestState::kQueued);  // 81 with X
  EXPECT_EQ(admission.Retire(x, 4).value().run, small);
}

TEST(Admission, QueuedRequestsTimeOutExplicitly) {
  Node node(100, PolicySettings{.queue_wait_limit = 5});
  Admission& admission = node.admission;
  (void)Admit(admission, Spec(50, 50, RequestClass::kInteractive, 100));
  const RequestId first = Admit(admission, Spec(50, 50), 1);
  const RequestId second = Admit(admission, Spec(50, 50), 3);
  EXPECT_THAT(admission.Drain(6).refused, IsEmpty());
  // At 7 the first has waited 6, the second 4.
  EXPECT_THAT(admission.Drain(7).refused, ElementsAre(Pair(first, AdmissionError::kQueueTimeout)));
  EXPECT_EQ(admission.StateOf(second), RequestState::kQueued);
}

TEST(Admission, AQueuedDeadlineIsRecheckedBeforeAdmission) {
  Node node(100, PolicySettings{});
  Admission& admission = node.admission;
  const RequestId a = Admit(admission, Spec(50, 50, RequestClass::kInteractive, 100));
  RequestSpec timed = Spec(50, 50, RequestClass::kInteractive, 10);
  timed.deadline = 120;  // 100 + 1 + 10 + 1 = 112 at tick 0
  const RequestId late = Admit(admission, timed);
  ASSERT_TRUE(admission.Boundary(a, 5, 20).has_value());  // A is slower than estimated
  EXPECT_THAT(admission.Drain(20).refused, ElementsAre(Pair(late, AdmissionError::kDeadline)));
}

// Releases drain the queue at once.
TEST(Admission, ReleasesAdmitQueuedRequests) {
  Node node(100, PolicySettings{});
  Admission& admission = node.admission;
  const RequestId a = Admit(admission, Spec(0, 90));
  const RequestId q = Admit(admission, Spec(20, 5));
  EXPECT_EQ(admission.StateOf(q), RequestState::kQueued);
  ASSERT_TRUE(
      admission.ReplaceEnvelope(a, {.retained = Bytes(0), .phase = Bytes(10)}, 1).has_value());
  EXPECT_EQ(admission.StateOf(q), RequestState::kWaiting);
}

// Background work J drains, so it defers a request instead of making it
// impossible.
TEST(Admission, BackgroundWorkDefersRatherThanRefuses) {
  Node node(100, PolicySettings{});
  Admission& admission = node.admission;
  ASSERT_TRUE(admission.AddBackground(Bytes(30), 0).has_value());
  const RequestId r = Admit(admission, Spec(20, 60));
  EXPECT_EQ(admission.StateOf(r), RequestState::kQueued);
  EXPECT_EQ(admission.ReleaseBackground(Bytes(30), 1).value().run, r);
}

// The promised resumption lives in the ledger, so a change made directly
// on the ledger cannot break it either.
TEST(Admission, TheLedgerItselfKeepsTheResumption) {
  Node node(100, PolicySettings{});
  Admission& admission = node.admission;
  ASSERT_TRUE(admission.AddFixed(Bytes(10), 0).has_value());
  const RequestId a = Admit(admission, Spec(0, 80, RequestClass::kBackground));
  const RequestId c = Admit(admission, Spec(0, 10, RequestClass::kBackground));
  ASSERT_TRUE(admission.Join(c, 0).has_value());
  const RequestId b = Admit(admission, Spec(0, 30, RequestClass::kInteractive), 1);
  EXPECT_EQ(admission.Boundary(a, 1, 2).value().paused, a);
  EXPECT_EQ(Failed(node.ledger.AddBackground(kSpark, Bytes(5))),
            CommitmentError::kBreaksResumption);
  EXPECT_EQ(Failed(node.ledger.Grant(kSpark, {.retained = Bytes(1), .phase = Bytes(1)})),
            CommitmentError::kBreaksResumption);
  // A cohort peer retires during the pause; then the substitute.
  ASSERT_TRUE(admission.Retire(c, 3).has_value());
  EXPECT_EQ(admission.Retire(b, 4).value().run, a);
  EXPECT_THAT(admission.Running(), ElementsAre(a));
  EXPECT_EQ(admission.Totals().required, Bytes(90));
}

// Joining is deferred while a pause is open and refused when the cohort
// does not fit; a pause the cohort inequality forbids is not taken.
TEST(Admission, JoinsAndPausesPassTheCohortCheck) {
  Node node(100, PolicySettings{});
  Admission& admission = node.admission;
  const RequestId a = Admit(admission, Spec(0, 40, RequestClass::kBackground));
  const RequestId c = Admit(admission, Spec(0, 40, RequestClass::kBackground));
  ASSERT_TRUE(admission.Join(c, 0).has_value());
  const RequestId big = Admit(admission, Spec(0, 30, RequestClass::kBackground));
  EXPECT_EQ(Failed(admission.Join(big, 0)), AdmissionError::kRefused);  // 110
  EXPECT_EQ(Failed(admission.Join(a, 0)), AdmissionError::kWrongState);
  ASSERT_TRUE(admission.Retire(big, 0).has_value());  // a waiting request retires
  const RequestId b = Admit(admission, Spec(0, 70, RequestClass::kInteractive), 1);
  EXPECT_FALSE(admission.Boundary(a, 1, 2).value().paused.has_value());  // C and B: 110
  ASSERT_TRUE(admission.Retire(c, 3).has_value());
  EXPECT_EQ(admission.Boundary(a, 2, 4).value().paused, a);  // B alone: 70
  const RequestId d = Admit(admission, Spec(0, 10, RequestClass::kInteractive), 5);
  EXPECT_EQ(Failed(admission.Join(d, 5)), AdmissionError::kDeferred);
  (void)b;
}

// A request the policy would pause for is not refused for a deadline the
// running work alone would miss.
TEST(Admission, TheUpFrontDeadlineCountsThePause) {
  Node node(100, PolicySettings{.minimum_run = 3});
  Admission& admission = node.admission;
  (void)Admit(admission, Spec(20, 30, RequestClass::kBackground, 1000));
  RequestSpec urgent = Spec(20, 30, RequestClass::kInteractive, 10);
  urgent.deadline = 50;  // 3 + 1 + 10 + 1 = 15
  EXPECT_TRUE(admission.Submit(urgent, 0).has_value());
}

TEST(Admission, ADeadlineCannotAssumeAPauseWithoutCapacity) {
  Node node(100);
  Admission& admission = node.admission;
  (void)Admit(admission, Spec(60, 40, RequestClass::kBackground, 1000));
  RequestSpec urgent = Spec(60, 40, RequestClass::kInteractive, 10);
  urgent.deadline = 50;
  EXPECT_EQ(Failed(admission.Submit(urgent, 0)), AdmissionError::kDeadline);
}

TEST(Admission, ADeadlineCannotAssumeAnInfeasibleReplacementCohort) {
  Node node(100);
  Admission& admission = node.admission;
  (void)Admit(admission, Spec(0, 40, RequestClass::kBackground, 1000));
  const RequestId peer = Admit(admission, Spec(0, 40, RequestClass::kBackground, 1000));
  ASSERT_TRUE(admission.Join(peer, 0).has_value());
  RequestSpec urgent = Spec(0, 70, RequestClass::kInteractive, 10);
  urgent.deadline = 50;  // its grant fits, but replacing either member needs 110
  EXPECT_EQ(Failed(admission.Submit(urgent, 0)), AdmissionError::kDeadline);
}

TEST(Admission, ADeadlineCannotSpendAnotherSubstitutesPause) {
  Node node(100);
  Admission& admission = node.admission;
  (void)Admit(admission, Spec(20, 30, RequestClass::kBackground, 1000));
  (void)Admit(admission, Spec(20, 30, RequestClass::kInteractive, 10));
  RequestSpec urgent = Spec(20, 30, RequestClass::kInteractive, 10);
  urgent.deadline = 50;  // the earlier interactive request gets the sole pause
  EXPECT_EQ(Failed(admission.Submit(urgent, 0)), AdmissionError::kDeadline);
}

TEST(Admission, ADeadlineIncludesLowerClassQueuePredecessors) {
  Node node(100);
  Admission& admission = node.admission;
  (void)Admit(admission, Spec(50, 50, RequestClass::kInteractive, 10));
  (void)Admit(admission, Spec(50, 50, RequestClass::kBackground, 1000));
  RequestSpec urgent = Spec(50, 50, RequestClass::kInteractive, 10);
  urgent.deadline = 50;  // the lower-class queue head cannot be bypassed
  EXPECT_EQ(Failed(admission.Submit(urgent, 0)), AdmissionError::kDeadline);
}

TEST(Admission, ADeadlineIncludesLowerClassGrantsThatBlockAdmission) {
  Node node(100);
  Admission& admission = node.admission;
  (void)Admit(admission, Spec(0, 20, RequestClass::kInteractive, 10));
  (void)Admit(admission, Spec(60, 30, RequestClass::kBackground, 1000));
  RequestSpec urgent = Spec(50, 30, RequestClass::kInteractive, 10);
  urgent.deadline = 50;  // the waiting background grant must retire first
  EXPECT_EQ(Failed(admission.Submit(urgent, 0)), AdmissionError::kDeadline);
}

TEST(Admission, AQueuedDeadlineCountsHigherClassSuccessorsGrantedInTheSameDrain) {
  Node node(100);
  Admission& admission = node.admission;
  ASSERT_TRUE(admission.AddBackground(Bytes(90), 0).has_value());
  RequestSpec timed = Spec(20, 10, RequestClass::kBackground, 5);
  timed.deadline = 10;
  const RequestId first = Admit(admission, timed);
  const RequestId urgent = Admit(admission, Spec(20, 10, RequestClass::kInteractive, 100));
  ASSERT_EQ(admission.StateOf(first), RequestState::kQueued);
  ASSERT_EQ(admission.StateOf(urgent), RequestState::kQueued);
  const Decision decision = admission.ReleaseBackground(Bytes(90), 0).value();
  EXPECT_THAT(decision.refused, ElementsAre(Pair(first, AdmissionError::kDeadline)));
  EXPECT_EQ(decision.run, urgent);
}

TEST(Admission, ADeadlineIncludesRequestsAgedByEarlierSelections) {
  Node node(1000, PolicySettings{.policy = SwitchingPolicy::kRunToCompletion, .aging_limit = 1});
  Admission& admission = node.admission;
  (void)Admit(admission, Spec(0, 10, RequestClass::kInteractive, 1));
  (void)Admit(admission, Spec(0, 10, RequestClass::kBackground, 1000));
  (void)Admit(admission, Spec(0, 10, RequestClass::kInteractive, 1));
  RequestSpec urgent = Spec(0, 10, RequestClass::kInteractive, 1);
  urgent.deadline = 50;  // the first waiting interactive request ages the background one
  EXPECT_EQ(Failed(admission.Submit(urgent, 0)), AdmissionError::kDeadline);
}

TEST(Admission, AFeasibleImmediatePauseCanPassLowerClassWaitingWork) {
  Node node(100);
  Admission& admission = node.admission;
  const RequestId running = Admit(admission, Spec(20, 30, RequestClass::kBackground, 1000));
  (void)Admit(admission, Spec(20, 30, RequestClass::kBackground, 1000));
  RequestSpec urgent = Spec(20, 30, RequestClass::kInteractive, 10);
  urgent.deadline = 50;
  const RequestId substitute = Admit(admission, urgent);
  EXPECT_EQ(admission.Boundary(running, 1, 1).value().run, substitute);
}

TEST(Admission, TickOverflowCannotMeetADeadline) {
  Node node(100);
  Admission& admission = node.admission;
  RequestSpec too_long = Spec(20, 30, RequestClass::kBackground, UINT64_MAX);
  too_long.deadline = UINT64_MAX;
  EXPECT_EQ(Failed(admission.Submit(too_long, 0)), AdmissionError::kDeadline);
  too_long.work = UINT64_MAX - 1;  // exactly fits before any pause
  const RequestId running = Admit(admission, too_long);
  (void)Admit(admission, Spec(20, 30, RequestClass::kInteractive, 1));
  EXPECT_FALSE(admission.Boundary(running, 0, 1).value().paused.has_value());
}

TEST(Admission, ABackwardsClockDoesNotSkipTheMinimumRun) {
  Node node(100, PolicySettings{.minimum_run = 5});
  Admission& admission = node.admission;
  const RequestId a = Admit(admission, Spec(20, 30, RequestClass::kBackground), 10);
  (void)Admit(admission, Spec(20, 30, RequestClass::kInteractive), 10);
  EXPECT_FALSE(admission.Boundary(a, 1, 3).value().paused.has_value());
}

// Time-slicing gives the slot to whatever would be chosen next.
TEST(Admission, TimeSlicingPicksTheNextChoice) {
  Node node(1000, PolicySettings{
                      .policy = SwitchingPolicy::kTimeSlicing, .pause_cap = 2, .time_slice = 1});
  Admission& admission = node.admission;
  const RequestId a = Admit(admission, Spec(10, 10, RequestClass::kBackground));
  (void)Admit(admission, Spec(10, 10, RequestClass::kBackground));
  const RequestId interactive = Admit(admission, Spec(10, 10, RequestClass::kInteractive));
  EXPECT_EQ(admission.Boundary(a, 1, 2).value().run, interactive);
}

// D-069: under random arrivals, boundaries and retirements, the rule holds
// after every event, and every admitted request runs or is refused
// explicitly; nothing waits forever once the running work retires.
TEST(Admission, EveryAdmittedRequestMakesProgress) {
  for (const SwitchingPolicy policy :
       {SwitchingPolicy::kPriorityAware, SwitchingPolicy::kRunToCompletion,
        SwitchingPolicy::kTimeSlicing}) {
    Node node(100, PolicySettings{.policy = policy,
                                  .minimum_run = 2,
                                  .pause_cap = 2,
                                  .time_slice = 3,
                                  .queue_limit = 16,
                                  .queue_wait_limit = 400});
    Admission& admission = node.admission;
    std::uint64_t state = 12345;
    const auto next = [&state](std::uint64_t bound) {
      state = (state * 6364136223846793005ULL) + 1442695040888963407ULL;
      return (state >> 33) % bound;
    };
    std::vector<RequestId> live;
    std::vector<RequestId> ran;
    std::uint64_t now = 0;
    const auto forget = [&](const Decision& decision) {
      for (const auto& [id, why] : decision.refused) {
        live.erase(std::ranges::remove(live, id).begin(), live.end());
      }
    };
    for (int step = 0; step < 3000; ++step) {
      now += 1 + next(3);
      const std::uint64_t what = next(10);
      if (what < 4) {
        auto submitted = admission.Submit(
            Spec(next(30), 1 + next(50),
                 next(2) == 0 ? RequestClass::kBackground : RequestClass::kInteractive,
                 1 + next(20)),
            now);
        if (submitted) {
          live.push_back(submitted->id);
          forget(submitted->decision);
        }
      } else if (!admission.Running().empty()) {
        const RequestId running = admission.Running()[next(admission.Running().size())];
        auto decision =
            what < 7 ? admission.Boundary(running, now, now) : admission.Retire(running, now);
        ASSERT_TRUE(decision.has_value());
        forget(*decision);
        if (what >= 7) {
          live.erase(std::ranges::remove(live, running).begin(), live.end());
        }
      } else {
        forget(admission.Drain(now));
      }
      for (const RequestId id : admission.Running()) {
        if (std::ranges::find(ran, id) == ran.end()) {
          ran.push_back(id);
        }
      }
      const auto totals = admission.Totals();
      ASSERT_LE(totals.required, totals.budget) << "step " << step;
    }
    // Drain: retire whatever runs until nothing is left.
    for (int guard = 0; !live.empty() && guard < 10000; ++guard) {
      now += 1;
      if (admission.Running().empty()) {
        forget(admission.Drain(now));
        if (admission.Running().empty() && admission.PausedRequest()) {
          FAIL() << "a paused request with nothing running";
        }
        continue;
      }
      const RequestId running = admission.Running().front();
      forget(admission.Retire(running, now).value());
      live.erase(std::ranges::remove(live, running).begin(), live.end());
    }
    EXPECT_THAT(live, IsEmpty());
    EXPECT_EQ(admission.Totals().required, Bytes(0));
  }
}

}  // namespace
