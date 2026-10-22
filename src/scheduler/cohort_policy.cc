// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "scheduler/cohort_policy.h"

namespace llmp::scheduler {

void CohortPolicy::BeginPass(Head head) {
  other_model_ = head == Head::kOtherModel;
  drain_ = other_model_ || head == Head::kIncompatible || suspended_ ||
           (role_ == Role::kSubstitute && formed_);
}

bool CohortPolicy::YieldForSwitch(Clock::time_point now) const {
  return suspended_ || (progressed_ && (other_model_ || role_ == Role::kSubstitute) &&
                        now - turn_since_ >= turn_);
}

CohortPolicy::Clock::time_point CohortPolicy::SwitchDeadline() const {
  return !switch_attempted_ && progressed_ && (other_model_ || role_ == Role::kSubstitute)
             ? turn_since_ + turn_
             : Clock::time_point::max();
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

bool CohortPolicy::CanRefill(std::size_t active, std::size_t capacity, std::size_t attempts) const {
  return !drain_ && active < capacity && attempts < capacity;
}

bool CohortPolicy::AdmitHead(Head head) {
  other_model_ = head == Head::kOtherModel;
  if (head == Head::kOtherModel || head == Head::kIncompatible) {
    drain_ = true;
  }
  return !drain_ && head == Head::kCompatible;
}

bool CohortPolicy::BlocksWaiters(bool full, bool suspended_elsewhere) const {
  return drain_ || full || suspended_elsewhere;
}

bool CohortPolicy::RunSubstitute(Head head, bool stopped_or_failed) const {
  return role_ == Role::kPrimary && suspended_ && !stopped_or_failed && head == Head::kOtherModel;
}

}  // namespace llmp::scheduler
