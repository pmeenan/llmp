// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// How a cooperative chat cohort shares conversation-state capacity
// (docs/runtime-serving.md#state-capacity-in-a-cohort). Vendor-free and
// host-only, so the CPU tests drive it; serve_api.cc applies its decisions
// to the members' sessions.
//
// Each member's state grows as its prompt and generation run, and is
// leased while the cohort runs. A member's growth (or its spilled state's
// restore) can be refused for capacity (Llm::StateRefusedFor) when the
// budget is full. A refusal leaves the member's state usable as it was.
// The policy:
//
// - Reclaim first: the node's one reclaim order (Server::Reclaim, memory/
//   reclaim.h) frees what the refused unit asked for, cheapest to restore
//   a byte first: plans and graphs, and idle conversations outside the
//   cohort, spilled (not cleared: their next turn restores them exactly).
//   When it freed anything, the refused member simply runs again.
// - Then wait, never fail, while a peer holds state: the refused member is
//   blocked, its state retained and its completed prefix kept, and runs
//   again once a peer retired (its state then reclaimable) or was preempted.
// - Refuse only what cannot fit alone: a member refused while no other
//   branch holds state (every other member is preempted, or there is none,
//   and nothing is left to reclaim) fails with the refusal, as a lone
//   request.
// - No deadlock: when no member can make progress (none is runnable and
//   none is about to retire), the youngest blocked member is set aside
//   (preempted): its sessions end at their completed unit and its state is
//   spilled (Llm::SpillSetAside), and it restarts later from that state
//   restored, exactly (cleared and prefilled again only when it could not
//   be spilled). The rest retry at once.
// - FIFO: the youngest is preempted first; a preempted member restarts, the
//   oldest first, only once no other member can run or is blocked; no new
//   member is admitted while any member waits.

#ifndef JITLLM_RUNTIME_COHORT_CAPACITY_H_
#define JITLLM_RUNTIME_COHORT_CAPACITY_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace jitllm::runtime {

inline constexpr std::size_t kCohortSlots = 4;

enum class CapacityWait : std::uint8_t {
  kNone,       // runnable (or terminal)
  kBlocked,    // refused for capacity; its state retained, it retries when capacity frees
  kPreempted,  // its state discarded; it restarts from its host tokens later
};

// One slot of a cohort, as the backend sees it between completed units.
struct CohortMember {
  bool present = false;        // a claimed slot
  bool terminal = false;       // finished, failed or cancelled: retires at the next boundary
  std::uint64_t admitted = 0;  // admission order: smaller is older
  CapacityWait wait = CapacityWait::kNone;
};
using Cohort = std::array<CohortMember, kCohortSlots>;

struct CapacityDecision {
  bool reclaim = false;                // something was reclaimed: the refused run again
  std::vector<std::size_t> refuse;     // cannot fit alone: end with the refusal
  std::optional<std::size_t> preempt;  // set it aside (its state spilled); it restarts later
};

// After one unit in which the members at `refused` were refused for
// capacity (their state usable and unchanged). `reclaimed`: the node's
// reclaim order freed something for them (Server::Reclaim). Marks the members,
// and returns whether they run again, which fail and which member
// to preempt; woken members are marked runnable. Members to refuse are
// marked terminal; the one to preempt, kPreempted.
CapacityDecision OnCapacityRefused(Cohort& cohort, std::span<const std::size_t> refused,
                                   bool reclaimed);

// After a member retired: blocked members retry (its state is now idle).
void OnMemberRetired(Cohort& cohort);

// Between units: when no member can make progress, blocked members retry,
// or else the oldest preempted member restarts (returned; the caller begins
// its prompt again and marks it runnable, or ends it).
std::optional<std::size_t> NextRestart(Cohort& cohort);

// Whether a new member may join: none waits for capacity.
bool AdmissionOpen(const Cohort& cohort);

// Whether a member may run its next unit.
inline bool Runnable(const CohortMember& member) {
  return member.present && !member.terminal && member.wait == CapacityWait::kNone;
}

}  // namespace jitllm::runtime

#endif  // JITLLM_RUNTIME_COHORT_CAPACITY_H_
