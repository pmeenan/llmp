// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Admission and the execution slot for one memory domain (D-050, D-069;
// docs/reservation-policy.md). A request is either refused (impossible, or
// a known deadline it cannot meet behind the queue), queued (it does not fit
// now), or admitted with a guaranteed envelope in the commitment ledger.
// One admitted request holds the execution slot from its first phase until
// it retires, unless the switching policy pauses it at a completed phase
// boundary for one substitute. Others may join it as a concurrent cohort
// only if the cohort inequality passes; joining never pauses anyone (M6
// validates concurrent execution; the policy is here now).
//
// The switching policies (D-069):
//   - priority-aware (the default): same-class requests run to completion;
//     an interactive request pauses a background one at a completed
//     boundary after that one's minimum run;
//   - run-to-completion for every class;
//   - time-slicing: any waiting request may pause the running one after a
//     quantum.
// Every policy has these guards: a minimum run, a cap on pauses per
// request, one open pause at a time with one substitute, the paused
// request resuming next when its substitute ends, and a request with a
// known deadline paused only if its remaining work, the substitute's
// program bound and both switching costs fit before its deadline.
//
// While a request is paused, the ledger holds the promised resumption (the
// paused request back in the running set, the substitute gone), and every
// capacity-changing transaction (a new grant, an envelope replacement, an
// increase to F or J, a budget reduction) must leave it feasible, whoever
// makes it. One that fits now but not at the resumption is deferred; the
// substitute's own changes never threaten it (its allowance is gone by
// then), so if they do not fit they are refused, never deferred.
//
// Queued requests hold no grant. Whenever capacity changes, the queue is
// drained in arrival order, and a queued request that can no longer fit
// alone, has waited past the queue-wait limit, or can no longer meet its
// deadline is refused explicitly (Decision::refused), so nothing waits
// behind a request that can never run.
//
// Pure policy on logical time: the scheduler thread owns it (D-048), feeds
// it events and carries out its decisions. It maps, loads and evicts
// nothing: a grant commits lazily (D-007).

#ifndef LLMP_SCHEDULER_ADMISSION_H_
#define LLMP_SCHEDULER_ADMISSION_H_

#include <cstdint>
#include <deque>
#include <expected>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "base/bytes.h"
#include "base/ids.h"
#include "memory/commitment.h"

namespace llmp::scheduler {

using base::Bytes;

struct RequestTag {
  static constexpr const char* kName = "request";
};
using RequestId = base::Id<RequestTag>;

// Logical time, in the scheduler's ticks.
using Tick = std::uint64_t;

enum class RequestClass : std::uint8_t { kBackground, kInteractive };

enum class SwitchingPolicy : std::uint8_t { kPriorityAware, kRunToCompletion, kTimeSlicing };

struct PolicySettings {
  SwitchingPolicy policy = SwitchingPolicy::kPriorityAware;
  // Guards (D-069); numbers are placeholders until M6 measures them.
  Tick minimum_run = 0;
  std::uint32_t pause_cap = 1;
  Tick time_slice = 0;  // the quantum, for time-slicing
  // A waiting request of a lower class is chosen after this many higher-
  // class selections pass it (starvation prevention).
  std::uint32_t aging_limit = 4;
  // The most requests that may wait in the queue, and for how long.
  std::size_t queue_limit = 64;
  std::optional<Tick> queue_wait_limit = std::nullopt;
};

struct RequestSpec {
  RequestClass request_class = RequestClass::kInteractive;
  memory::Envelope envelope;
  // The finite program's remaining work, in ticks (an estimate from
  // program bounds and measured rates).
  Tick work = 0;
  // Switching to and from this request, in ticks (catalog estimates).
  Tick switch_cost = 0;
  std::optional<Tick> deadline;  // absolute, from configuration (D-047)
};

enum class AdmissionError : std::uint8_t {
  kImpossible,  // cannot fit even alone at this budget and fixed overhead
  kDeadline,    // a known deadline the queue ahead makes unmeetable
  kQueueFull,
  kQueueTimeout,  // waited in the queue longer than the configured limit
  kDeferred,      // a capacity change that must wait for a resumption
  kRefused,       // a capacity change that cannot fit, not even later
  kUnknownRequest,
  kWrongState,
};

std::string ToString(AdmissionError error);

enum class RequestState : std::uint8_t { kQueued, kWaiting, kRunning, kPaused, kRetired };

// What the scheduler should do after an event.
struct Decision {
  // The request that should hold the slot now, if it changed.
  std::optional<RequestId> run;
  // The request paused at this boundary, if any.
  std::optional<RequestId> paused;
  // Queued requests refused by this event, which the caller answers.
  std::vector<std::pair<RequestId, AdmissionError>> refused;
};

struct Submitted {
  RequestId id;
  Decision decision;
};

class Admission {
 public:
  // Admission for one domain of ledger, which must already hold it. The
  // ledger outlives this object; its grants for the domain are this
  // object's to make and retire.
  Admission(memory::CommitmentLedger& ledger, memory::DomainId domain, PolicySettings settings);

  // A new request at `now`. Admitted (waiting for the slot, or running if
  // the slot is free), queued, or refused.
  std::expected<Submitted, AdmissionError> Submit(const RequestSpec& spec, Tick now);

  // The running request reached a completed phase boundary, with `done`
  // ticks of its work behind it. The policy may pause it for a waiting
  // request. Nothing here checks that the boundary is a completed one: the
  // caller does (execution::ProgramCursor::AtBoundary, execution/program.h).
  std::expected<Decision, AdmissionError> Boundary(RequestId id, Tick done, Tick now);

  // A request retired or was terminated explicitly, whatever its state.
  // Never refused for an admitted request. The slot passes on.
  std::expected<Decision, AdmissionError> Retire(RequestId id, Tick now);

  // Capacity-changing transactions (D-050, D-069). Each drains the queue
  // afterwards: a release may admit queued requests, and a budget
  // reduction or a larger F may leave some unable to fit, which are refused.
  std::expected<Decision, AdmissionError> ReplaceEnvelope(RequestId id,
                                                          const memory::Envelope& envelope,
                                                          Tick now);
  std::expected<Decision, AdmissionError> AddFixed(Bytes bytes, Tick now);
  std::expected<Decision, AdmissionError> AddBackground(Bytes bytes, Tick now);
  std::expected<Decision, AdmissionError> SetBudget(Bytes budget, Tick now);
  std::expected<Decision, AdmissionError> ReleaseFixed(Bytes bytes, Tick now);
  std::expected<Decision, AdmissionError> ReleaseBackground(Bytes bytes, Tick now);

  // An admitted, waiting request joins the running cohort if the cohort
  // inequality passes with it. Deferred while a pause is open.
  std::expected<void, AdmissionError> Join(RequestId id, Tick now);

  // Refuses queued requests that can no longer run (see above), admits
  // those that now fit, in order, and gives a free slot. Also the timer
  // tick for queue-wait limits.
  Decision Drain(Tick now);

  std::optional<RequestState> StateOf(RequestId id) const;
  const std::vector<RequestId>& Running() const { return running_; }
  std::optional<RequestId> PausedRequest() const { return paused_; }
  memory::CommitmentTotals Totals() const;

 private:
  struct Request {
    RequestSpec spec;
    RequestState state = RequestState::kQueued;
    std::optional<memory::GrantId> grant;
    Tick submitted = 0;
    Tick run_started = 0;  // when it last got the slot
    Tick done = 0;         // work behind it
    std::uint32_t pauses = 0;
    std::uint32_t passed_over = 0;  // selections that skipped it
    std::uint64_t sequence = 0;     // arrival order
  };

  // Whether it fits alone at the budget and fixed overhead. Background
  // work J drains, so it defers a request rather than refusing it.
  bool FitsAlone(const RequestSpec& spec) const;
  // The ledger's active cohort follows the running set, and its promised
  // resumption the open pause.
  void SyncCohort();
  memory::GrantId GrantOf(RequestId id) const;
  std::vector<memory::GrantId> Grants(const std::vector<RequestId>& requests) const;
  // The admitted request that should get a free slot, without side
  // effects; ChooseNext also ages the requests it passes over.
  std::optional<RequestId> PeekNext() const;
  bool Before(const Request& candidate, const Request& other) const;
  std::optional<RequestId> ChooseNext();
  Decision GiveSlot(Tick now);
  Decision Drained(Decision decision, Tick now);
  static bool DeadlineAllows(const Request& running, const Request& substitute, Tick now);
  // A conservative wait estimate. Credit an immediate pause only if this
  // request is next and both its grant and replacement cohort fit. Otherwise
  // count current work to completion, queue predecessors and aged selections.
  // `sequence` places a queued request among the queue; new requests go last.
  // No value means the wait exceeds the representable tick range.
  std::optional<Tick> WaitAhead(const RequestSpec& spec, std::uint64_t sequence, Tick now) const;
  std::optional<AdmissionError> Unrunnable(const Request& request, Tick now) const;

  memory::CommitmentLedger& ledger_;
  memory::DomainId domain_;
  PolicySettings settings_;
  base::SlotTable<RequestTag, Request> requests_;
  std::deque<RequestId> queued_;         // not yet admitted, in arrival order
  std::vector<RequestId> waiting_;       // admitted, waiting for the slot
  std::vector<RequestId> running_;       // the slot's holder, and its cohort
  std::optional<RequestId> paused_;      // the one open pause
  std::optional<RequestId> substitute_;  // what it was paused for
  std::uint64_t next_sequence_ = 0;
};

}  // namespace llmp::scheduler

#endif  // LLMP_SCHEDULER_ADMISSION_H_
