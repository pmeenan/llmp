// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "scheduler/cohort_policy.h"

namespace llmp::scheduler {
namespace {

// The executor's order (cohort_policy.h): a pass, its admissions, its wait,
// then the next pass; the end once.
bool Follows(std::optional<CohortPolicy::Point> last, CohortPolicy::Point next) {
  using P = CohortPolicy::Point;
  if (!last) return next == P::kPass || next == P::kEnd;
  switch (*last) {
    case P::kPass:
      return true;
    case P::kAdmit:
      return next == P::kAdmit || next == P::kWait;
    case P::kWait:
      return next == P::kPass || next == P::kEnd;
    case P::kEnd:
      return false;
  }
  return false;
}

const char* Name(std::optional<CohortPolicy::Point> point) {
  if (!point) return "the start";
  switch (*point) {
    case CohortPolicy::Point::kPass:
      return "a pass";
    case CohortPolicy::Point::kAdmit:
      return "an admission";
    case CohortPolicy::Point::kWait:
      return "a wait";
    case CohortPolicy::Point::kEnd:
      return "the end";
  }
  return "?";
}

}  // namespace

bool CohortPolicy::Turning() const {
  return progressed_ && (other_model_ || role_ == Role::kSubstitute);
}

std::expected<CohortPolicy::Action, std::string> CohortPolicy::Decide(const View& view) {
  if (!Follows(last_, view.point)) {
    return std::unexpected(std::string("a cohort policy decision out of its order: ") +
                           Name(view.point) + " after " + Name(last_));
  }
  last_ = view.point;
  const Head head = view.queue.head;
  Action action;
  switch (view.point) {
    case Point::kPass:
      // Refreshed each pass: a departed competing request does not keep a
      // cohort draining, unless a continuation already retired for it.
      other_model_ = view.queue.other.has_value();
      release_ = view.release_pending;
      drain_ = head == Head::kOtherModel || head == Head::kIncompatible || suspended_ ||
               (role_ == Role::kSubstitute && formed_);
      {
        const bool over = progressed_ && view.now - turn_since_ >= turn_;
        action.yield_for_switch = suspended_ || (Turning() && over);
        // A release waits at most for the turn's end, and the node is not
        // idle at it (its members still run, or pause for another model):
        // answered busy, the turn going on or rotating as it would without it.
        action.release_busy = release_ && over;
        if (action.release_busy) release_ = false;
      }
      break;
    case Point::kAdmit:
      // A competing request found during refill arms the turn as well.
      other_model_ = view.queue.other.has_value();
      release_ = release_ || view.release_pending;
      if (head == Head::kOtherModel || head == Head::kIncompatible) drain_ = true;
      action.admit = !drain_ && head == Head::kCompatible && view.active < view.capacity &&
                     view.attempts < view.capacity;
      break;
    case Point::kWait:
      formed_ = true;
      action.blocks_waiters = drain_ || view.active >= view.capacity || view.suspended_elsewhere;
      // A wait for paused readers lasts at most to the turn's end: there
      // the members pause for another model, or a waiting release is answered.
      if ((!switch_attempted_ && Turning()) || (progressed_ && release_)) {
        action.wake = turn_since_ + turn_;
      }
      break;
    case Point::kEnd:
      // At most one substitute call: a substitute returns its continuations
      // to its caller instead of nesting another turn.
      action.run_substitute =
          role_ == Role::kPrimary && suspended_ && !view.stopped && view.queue.other.has_value();
      break;
  }
  return action;
}

void CohortPolicy::UnitCompleted(Clock::time_point now, Unit unit) {
  switch_attempted_ = false;
  if (unit == Unit::kSwap) {
    turn_since_ = now;  // residency cost is not useful work
    progressed_ = false;
  } else if (unit == Unit::kProgress) {
    progressed_ = true;  // a resumed turn must advance before yielding again
  }
}

}  // namespace llmp::scheduler
