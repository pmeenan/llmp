// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef LLMP_SCHEDULER_COHORT_POLICY_H_
#define LLMP_SCHEDULER_COHORT_POLICY_H_

#include <chrono>
#include <cstddef>
#include <deque>
#include <utility>

namespace llmp::scheduler {

// Model-turn policy, independent of transport, backend ownership and clocks.
// The caller supplies completed-unit times and the current queue head. Unit
// selection within a model lives in cohort_schedule.h; turn expiry only asks
// for a yield at a completed boundary, never cancellation or request refusal.
class CohortPolicy {
 public:
  using Clock = std::chrono::steady_clock;
  enum class Unit { kSwap, kReadiness, kProgress };
  enum class Role { kPrimary, kSubstitute };
  enum class Head { kEmpty, kCompatible, kIncompatible, kOtherModel };

  CohortPolicy(Clock::duration turn, Clock::time_point now, Role role = Role::kPrimary)
      : turn_(turn), turn_since_(now), role_(role) {}

  // Refresh on each pass: a departed competing head must not keep a cohort
  // draining. Once a continuation retired, finish draining its peers even
  // if that head left. Backend retirement, not a yield request, records this.
  void BeginPass(Head head);
  bool YieldForSwitch(Clock::time_point now) const;
  Clock::time_point SwitchDeadline() const;
  // A work may refuse to yield. If all members then wait for their readers,
  // do not spin on an expired turn deadline; readers/intake/backpressure wake
  // the caller. Completing another unit permits another deadline wake.
  void SwitchAttempted() { switch_attempted_ = true; }
  void ContinuationRetired() { suspended_ = true; }
  void UnitCompleted(Clock::time_point now, Unit unit);

  // Capacity and the admission-attempt budget are the caller's bounded work
  // storage, not client admission limits. An incompatible head is never skipped.
  bool CanRefill(std::size_t active, std::size_t capacity, std::size_t attempts) const;
  bool AdmitHead(Head head);
  void Deferred() { drain_ = true; }
  void AdmissionFinished() { formed_ = true; }
  bool BlocksWaiters(bool full, bool suspended_elsewhere) const;

  // At most one substitute call: a substitute returns its continuations to
  // its caller instead of nesting another turn. Stop/failure still restores
  // those owners for the caller's normal cancellation/retirement path.
  bool RunSubstitute(Head head, bool stopped_or_failed) const;

  // Return original owners first. A yielded substitute rotates behind other
  // models, carrying its fresh queued arrivals with it: they must never take
  // a slot held by a continuation. This prevents an A/B pair starving C while
  // preserving admission order within every model. Key returns model identity.
  template <typename Request, typename Key>
  void Restore(std::deque<Request>& queue, std::deque<Request>& suspended, Key key) const {
    if (suspended.empty()) {
      return;
    }
    if (role_ == Role::kPrimary) {
      while (!suspended.empty()) {
        queue.push_front(std::move(suspended.back()));
        suspended.pop_back();
      }
      return;
    }
    std::deque<Request> later;
    for (auto remaining = queue.size(); remaining != 0; --remaining) {
      auto request = std::move(queue.front());
      queue.pop_front();
      if (key(request) == key(suspended.front())) {
        later.push_back(std::move(request));
      } else {
        queue.push_back(std::move(request));
      }
    }
    for (auto* group : {&suspended, &later}) {
      while (!group->empty()) {
        queue.push_back(std::move(group->front()));
        group->pop_front();
      }
    }
  }

 private:
  Clock::duration turn_;
  Clock::time_point turn_since_;
  Role role_;
  bool other_model_ = false;
  bool formed_ = false;
  bool suspended_ = false;
  bool progressed_ = false;
  bool switch_attempted_ = false;
  bool drain_ = false;
};

}  // namespace llmp::scheduler

#endif  // LLMP_SCHEDULER_COHORT_POLICY_H_
