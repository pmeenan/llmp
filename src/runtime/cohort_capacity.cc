// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "runtime/cohort_capacity.h"

#include <algorithm>

namespace llmp::runtime {
namespace {

// A member that retires or runs next: the cohort is not stuck.
bool CanProgress(const Cohort& cohort) {
  return std::ranges::any_of(cohort, [](const CohortMember& m) {
    return m.present && (m.terminal || m.wait == CapacityWait::kNone);
  });
}

void WakeBlocked(Cohort& cohort) {
  for (CohortMember& m : cohort) {
    if (m.present && m.wait == CapacityWait::kBlocked) {
      m.wait = CapacityWait::kNone;
    }
  }
}

}  // namespace

CapacityDecision OnCapacityRefused(Cohort& cohort, std::span<const std::size_t> refused,
                                   bool reclaimed) {
  CapacityDecision decision;
  if (reclaimed) {
    // The reclaim order made room: the refused members
    // stay runnable, and so does anyone waiting.
    decision.reclaim = true;
    WakeBlocked(cohort);
    return decision;
  }
  std::vector<std::size_t> order;
  for (const std::size_t slot : refused) {
    if (slot < cohort.size() && cohort[slot].present && !cohort[slot].terminal &&
        std::ranges::find(order, slot) == order.end()) {
      order.push_back(slot);
    }
  }
  std::ranges::sort(order, {}, [&](std::size_t slot) { return cohort[slot].admitted; });
  for (const std::size_t slot : order) {
    // A peer holds state while it is present and not preempted, including a
    // blocked peer or one about to retire: waiting frees it (a retired
    // member's state becomes idle, and then reclaimable).
    bool peer_holds = false;
    for (std::size_t other = 0; other < cohort.size(); ++other) {
      peer_holds = peer_holds || (other != slot && cohort[other].present &&
                                  cohort[other].wait != CapacityWait::kPreempted);
    }
    if (peer_holds) {
      cohort[slot].wait = CapacityWait::kBlocked;
    } else {
      cohort[slot].wait = CapacityWait::kNone;
      cohort[slot].terminal = true;
      decision.refuse.push_back(slot);
    }
  }
  if (!CanProgress(cohort)) {
    // Every member holding state waits for another: free the youngest's.
    std::optional<std::size_t> youngest;
    for (std::size_t slot = 0; slot < cohort.size(); ++slot) {
      const CohortMember& m = cohort[slot];
      if (m.present && m.wait == CapacityWait::kBlocked &&
          (!youngest || m.admitted > cohort[*youngest].admitted)) {
        youngest = slot;
      }
    }
    if (youngest) {
      cohort[*youngest].wait = CapacityWait::kPreempted;
      decision.preempt = youngest;
      WakeBlocked(cohort);
    }
  }
  return decision;
}

void OnMemberRetired(Cohort& cohort) { WakeBlocked(cohort); }

std::optional<std::size_t> NextRestart(Cohort& cohort) {
  if (CanProgress(cohort)) {
    return std::nullopt;
  }
  if (std::ranges::any_of(cohort, [](const CohortMember& m) {
        return m.present && m.wait == CapacityWait::kBlocked;
      })) {
    WakeBlocked(cohort);
    return std::nullopt;
  }
  std::optional<std::size_t> oldest;
  for (std::size_t slot = 0; slot < cohort.size(); ++slot) {
    const CohortMember& m = cohort[slot];
    if (m.present && m.wait == CapacityWait::kPreempted &&
        (!oldest || m.admitted < cohort[*oldest].admitted)) {
      oldest = slot;
    }
  }
  if (oldest) {
    cohort[*oldest].wait = CapacityWait::kNone;
  }
  return oldest;
}

bool AdmissionOpen(const Cohort& cohort) {
  return std::ranges::none_of(cohort, [](const CohortMember& m) {
    return m.present && !m.terminal && m.wait != CapacityWait::kNone;
  });
}

}  // namespace llmp::runtime
