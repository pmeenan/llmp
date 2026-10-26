// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef LLMP_SCHEDULER_COHORT_POLICY_H_
#define LLMP_SCHEDULER_COHORT_POLICY_H_

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <expected>
#include <optional>
#include <string>
#include <utility>

namespace llmp::scheduler {

// A node's model-turn policy for one cooperative cohort, independent of
// transport, backend ownership and clocks (docs/runtime-serving.md, "Pending
// model switches"). It is node-local: a conductor places models across nodes
// and leaves each node's turns to it. Unit selection within a model lives in
// cohort_schedule.h; turn expiry only asks for a yield at a completed
// boundary, never cancellation or request refusal.
//
// One decision function, Decide(view) -> action, with the executor
// (Server::ServeCooperative) performing each action. Each pass of the
// executor asks at its points in this order, which Decide checks:
//
//   kPass   the queue looked at: yield every member for a switch?
//   kAdmit  (repeated) the queue's head looked at again: admit it?
//   kWait   admission over: does the cohort block waiters, and until when
//           may a wait for paused readers last?
//   ...     a unit runs (UnitCompleted), then the next pass: kPass again.
//   kEnd    the cohort ended with members set aside for a switch: run a
//           substitute cohort for another model?
//
// The executor reports what happened between decisions: SwitchAttempted,
// ContinuationRetired, Deferred and UnitCompleted.
//
// Turns. A turn ends (its members pause at their next completed unit) once
// it has done useful work for model_turn and a request for another model
// waits anywhere in the queue, not only at its head. A pending working-set
// release changes neither turns nor admission (the starvation bound comes
// first, D-102): it waits at most until the turn's end (model_turn of useful
// work, as for another model), where the node is still busy and it is
// answered so (Action::release_busy); a cohort that ends sooner leaves the
// release to the executor's idle point. Admission stays FIFO: an
// incompatible or other-model head stops refills, a compatible head behind
// which another model waits is still admitted. The primary cohort's model
// alternates with substitutes: with n other models waiting, the primary runs
// every other turn and each other model at least one turn in 2n, so every
// waiting model is reached within 2n turns (Restore's rotation).
class CohortPolicy {
 public:
  using Clock = std::chrono::steady_clock;
  enum class Unit { kSwap, kReadiness, kProgress };
  enum class Role { kPrimary, kSubstitute };
  enum class Head { kEmpty, kCompatible, kIncompatible, kOtherModel };
  enum class Point : std::uint8_t { kPass, kAdmit, kWait, kEnd };

  // The queue as a cohort sees it at one look (Look).
  struct Queue {
    Head head = Head::kEmpty;
    // The first request for another model, anywhere in the queue: a turn
    // ends for it, and it is the substitute a primary runs.
    std::optional<std::size_t> other{};
  };
  struct View {
    Point point = Point::kPass;
    Queue queue{};
    // kAdmit and kWait: the active works, the caller's bounded storage for
    // them and this pass's admission attempts (capacity bounds both; they are
    // not client admission limits).
    std::size_t active = 0;
    std::size_t capacity = 0;
    std::size_t attempts = 0;
    bool suspended_elsewhere = false;  // kWait: continuations of an enclosing cohort wait
    bool release_pending = false;      // kPass/kAdmit: a working-set release waits
    bool stopped = false;              // kEnd: the runtime stops or the backend failed
    Clock::time_point now{};           // kPass
  };
  struct Action {
    bool yield_for_switch = false;  // kPass: ask every member to pause for a switch
    // kPass: the turn ended with a release waiting: answer it busy (409).
    bool release_busy = false;
    bool admit = false;           // kAdmit: start the head
    bool blocks_waiters = false;  // kWait: paused members yield their place once due
    // kWait: the latest a wait for paused readers lasts (a turn's end, when
    // another model or a release waits).
    Clock::time_point wake = Clock::time_point::max();
    // kEnd: run a substitute cohort for the request at View::queue.other.
    // Stop or failure restores the set-aside owners for the caller's normal
    // cancellation/retirement path instead.
    bool run_substitute = false;
  };

  CohortPolicy(Clock::duration turn, Clock::time_point now, Role role = Role::kPrimary)
      : turn_(turn), turn_since_(now), role_(role) {}

  std::expected<Action, std::string> Decide(const View& view);

  // A member may refuse to pause. If all members then wait for their
  // readers, a wait does not spin on an expired turn: readers, intake and
  // backpressure wake it. Completing another unit permits another wake.
  void SwitchAttempted() { switch_attempted_ = true; }
  // A member retired with a continuation for a switch: its peers finish
  // pausing, even if the competing request leaves.
  void ContinuationRetired() { suspended_ = true; }
  // The admitted head could not start (capacity): no further refill.
  void Deferred() { drain_ = true; }
  void UnitCompleted(Clock::time_point now, Unit unit);

  // The queue as a cohort of `model` sees it: its head's class (another
  // model, or for this model whether `supports` admits it to a cohort) and
  // the first request for another model. `key` gives a request's model.
  // O(queue), under the caller's queue lock.
  template <typename Request, typename Model, typename Key, typename Supports>
  static Queue Look(const std::deque<Request>& queue, const Model& model, Key key,
                    Supports supports) {
    Queue seen;
    for (std::size_t i = 0; i < queue.size(); ++i) {
      if (key(queue[i]) != model) {
        seen.other = i;
        break;
      }
    }
    if (!queue.empty()) {
      seen.head = seen.other == std::size_t{0} ? Head::kOtherModel
                  : supports(queue.front())    ? Head::kCompatible
                                               : Head::kIncompatible;
    }
    return seen;
  }

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
  // Whether useful work for another model's sake (or a substitute's, for
  // its paused original) runs against the turn clock.
  bool Turning() const;

  Clock::duration turn_;
  Clock::time_point turn_since_;
  Role role_;
  std::optional<Point> last_;  // the previous decision's point (its order)
  bool other_model_ = false;
  bool release_ = false;
  bool formed_ = false;
  bool suspended_ = false;
  bool progressed_ = false;
  bool switch_attempted_ = false;
  bool drain_ = false;
};

}  // namespace llmp::scheduler

#endif  // LLMP_SCHEDULER_COHORT_POLICY_H_
