// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "scheduler/admission.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "base/bytes.h"
#include "base/check.h"
#include "memory/commitment.h"

namespace llmp::scheduler {
namespace {

// A wait beyond the tick range cannot meet a representable deadline.
std::optional<Tick> Plus(Tick a, Tick b) {
  return a > UINT64_MAX - b ? std::nullopt : std::optional<Tick>(a + b);
}

// Time since `then`; a clock that went backwards counts as none.
Tick Since(Tick now, Tick then) { return now > then ? now - then : 0; }

void Erase(std::vector<RequestId>& list, RequestId id) {
  list.erase(std::ranges::remove(list, id).begin(), list.end());
}

// A change the ledger refused: one that breaks only the promised
// resumption waits for it; anything else does not fit.
AdmissionError FromLedger(memory::CommitmentError error) {
  return error == memory::CommitmentError::kBreaksResumption ? AdmissionError::kDeferred
                                                             : AdmissionError::kRefused;
}

}  // namespace

std::string ToString(AdmissionError error) {
  switch (error) {
    case AdmissionError::kImpossible:
      return "the request cannot fit even alone at this budget";
    case AdmissionError::kDeadline:
      return "the request's deadline cannot be met behind the queue";
    case AdmissionError::kQueueFull:
      return "the queue is full";
    case AdmissionError::kQueueTimeout:
      return "the request waited in the queue longer than the limit";
    case AdmissionError::kDeferred:
      return "the change must wait until the paused request resumes";
    case AdmissionError::kRefused:
      return "the change does not fit";
    case AdmissionError::kUnknownRequest:
      return "stale or unknown request";
    case AdmissionError::kWrongState:
      return "the request is not in the state this needs";
  }
  return "unknown admission error";
}

Admission::Admission(memory::CommitmentLedger& ledger, memory::DomainId domain,
                     PolicySettings settings)
    : ledger_(ledger), domain_(domain), settings_(settings) {
  base::Check(ledger_.Totals(domain_).has_value(),
              "admission for a domain the ledger does not hold");
}

memory::CommitmentTotals Admission::Totals() const {
  const std::optional<memory::CommitmentTotals> totals = ledger_.Totals(domain_);
  base::Check(totals.has_value(), "admission for a domain the ledger does not hold");
  return totals.value_or(memory::CommitmentTotals{});
}

memory::GrantId Admission::GrantOf(RequestId id) const {
  const Request* request = requests_.Find(id);
  base::Check(request != nullptr && request->grant.has_value(), "an admitted request has no grant");
  return request != nullptr ? request->grant.value_or(memory::GrantId{}) : memory::GrantId{};
}

std::vector<memory::GrantId> Admission::Grants(const std::vector<RequestId>& requests) const {
  std::vector<memory::GrantId> grants;
  grants.reserve(requests.size());
  for (const RequestId id : requests) {
    grants.push_back(GrantOf(id));
  }
  return grants;
}

void Admission::SyncCohort() {
  // Every change to the running set, and every change during a pause, was
  // checked before it was made, so these cannot fail.
  base::Check(ledger_.ClearResumption(domain_).has_value(), "the ledger lost the domain");
  base::Check(ledger_.SetCohort(domain_, Grants(running_)).has_value(),
              "the running set does not fit");
  if (paused_ && substitute_) {
    std::vector<RequestId> resumption = running_;
    Erase(resumption, *substitute_);
    resumption.push_back(*paused_);
    base::Check(
        ledger_.SetResumption(domain_, Grants(resumption), GrantOf(*substitute_)).has_value(),
        "the promised resumption does not fit");
  }
}

bool Admission::FitsAlone(const RequestSpec& spec) const {
  const memory::CommitmentTotals totals = Totals();
  const std::optional<Bytes> required =
      totals.fixed.Plus(spec.envelope.retained).and_then([&](Bytes b) {
        return b.Plus(spec.envelope.phase);
      });
  return required && *required <= totals.budget;
}

std::optional<Tick> Admission::WaitAhead(const RequestSpec& spec, std::uint64_t sequence,
                                         Tick now) const {
  const auto remaining = [](const Request& request) {
    return request.spec.work > request.done ? request.spec.work - request.done : 0;
  };
  const auto add_work = [&](std::optional<Tick> wait, const Request& request) {
    return wait.and_then([&](Tick value) { return Plus(value, remaining(request)); })
        .and_then([&](Tick value) { return Plus(value, request.spec.switch_cost); });
  };
  Request target;
  target.spec = spec;
  target.sequence = sequence;
  std::optional<Tick> ahead{0};
  bool immediate = true;
  const bool queued_before = std::ranges::any_of(
      queued_, [&](RequestId id) { return requests_.Find(id)->sequence < sequence; });
  if (queued_before || !ledger_.Check(domain_, memory::Change{.add = spec.envelope})) {
    // No bypass at admission: every queue predecessor may need to finish.
    // If the new grant cannot coexist with current grants, even a lower-
    // class waiting request's retained state can block admission.
    immediate = false;
    for (const RequestId id : waiting_) {
      ahead = add_work(ahead, *requests_.Find(id));
    }
    for (const RequestId id : queued_) {
      const Request& other = *requests_.Find(id);
      if (other.sequence < sequence) {
        ahead = add_work(ahead, other);
      }
    }
  } else {
    // Replay known waiting choices, including aging caused by each choice.
    // A lower-class request can move ahead even if it is not aged yet.
    std::vector<Request> choices;
    choices.reserve(waiting_.size() + 1);
    for (const RequestId id : waiting_) {
      choices.push_back(*requests_.Find(id));
    }
    choices.push_back(target);
    while (!choices.empty()) {
      const auto next = std::ranges::min_element(
          choices, [&](const Request& a, const Request& b) { return Before(a, b); });
      if (next->sequence == sequence) {
        break;
      }
      immediate = false;
      ahead = add_work(ahead, *next);
      const RequestClass chosen_class = next->spec.request_class;
      choices.erase(next);
      for (Request& other : choices) {
        if (other.spec.request_class < chosen_class && other.passed_over < UINT32_MAX) {
          ++other.passed_over;
        }
      }
    }
  }
  // Drain can grant several queued requests before choosing a runner. A
  // later higher-class request may then overtake this one in the same
  // batch, despite being behind it in the admission queue.
  for (const RequestId id : queued_) {
    const Request& other = *requests_.Find(id);
    if (other.sequence > sequence && other.spec.request_class > spec.request_class) {
      immediate = false;
      ahead = add_work(ahead, other);
    }
  }
  // A later substitute cannot promise this request a pause: the current
  // request resumes next and may have exhausted its pause cap by then.
  std::optional<Tick> slot{0};
  std::optional<Tick> pause_point;
  for (const RequestId id : running_) {
    const Request& running = *requests_.Find(id);
    const std::optional<Tick> finish = add_work(0, running);
    slot = slot && finish ? std::optional<Tick>(std::max(*slot, *finish)) : std::nullopt;
    const bool higher = spec.request_class > running.spec.request_class;
    const bool pausable = immediate && !paused_ && running.pauses < settings_.pause_cap &&
                          (settings_.policy == SwitchingPolicy::kTimeSlicing ||
                           (settings_.policy == SwitchingPolicy::kPriorityAware && higher));
    if (!pausable || !DeadlineAllows(running, target, now)) {
      continue;
    }
    std::vector<RequestId> peers = running_;
    Erase(peers, id);
    if (!ledger_.Check(
            domain_,
            memory::Change{.add = spec.envelope, .add_to_cohort = true, .cohort = Grants(peers)})) {
      continue;
    }
    const Tick hold = settings_.policy == SwitchingPolicy::kTimeSlicing
                          ? std::max(settings_.minimum_run, settings_.time_slice)
                          : settings_.minimum_run;
    const std::optional<Tick> point =
        Plus(hold - std::min(hold, Since(now, running.run_started)), running.spec.switch_cost);
    if (point) {
      pause_point = std::min(pause_point.value_or(*point), *point);
    }
  }
  std::optional<Tick> wait = pause_point && (!slot || *pause_point < *slot) ? pause_point : slot;
  if (paused_) {
    // The paused request resumes next, before anyone else.
    wait = add_work(wait, *requests_.Find(*paused_));
  }
  return wait.and_then(
      [&](Tick value) { return ahead.and_then([&](Tick prior) { return Plus(value, prior); }); });
}

std::optional<AdmissionError> Admission::Unrunnable(const Request& request, Tick now) const {
  if (!FitsAlone(request.spec)) {
    return AdmissionError::kImpossible;
  }
  if (settings_.queue_wait_limit && Since(now, request.submitted) > *settings_.queue_wait_limit) {
    return AdmissionError::kQueueTimeout;
  }
  if (request.spec.deadline) {
    const std::optional<Tick> finish =
        WaitAhead(request.spec, request.sequence, now)
            .and_then([&](Tick wait) { return Plus(now, wait); })
            .and_then([&](Tick value) { return Plus(value, request.spec.work); })
            .and_then([&](Tick value) { return Plus(value, request.spec.switch_cost); });
    if (!finish || *finish > *request.spec.deadline) {
      return AdmissionError::kDeadline;  // still refused before admission (D-069)
    }
  }
  return std::nullopt;
}

std::expected<Submitted, AdmissionError> Admission::Submit(const RequestSpec& spec, Tick now) {
  Request request;
  request.spec = spec;
  request.submitted = now;
  request.sequence = next_sequence_;
  if (const std::optional<AdmissionError> refused = Unrunnable(request, now)) {
    return std::unexpected(*refused);
  }
  // Admitted at once only if nothing is queued ahead of it. A grant during
  // a pause must also keep the promised resumption (the ledger checks).
  if (queued_.empty()) {
    if (auto grant = ledger_.Grant(domain_, spec.envelope); grant) {
      request.grant = *grant;
      request.state = RequestState::kWaiting;
    }
  }
  if (!request.grant && queued_.size() >= settings_.queue_limit) {
    return std::unexpected(AdmissionError::kQueueFull);
  }
  const RequestId id = requests_.Insert(request);
  if (!id.valid()) {
    if (request.grant) {
      base::Check(ledger_.Retire(*request.grant).has_value(), "a fresh grant is missing");
    }
    return std::unexpected(AdmissionError::kQueueFull);
  }
  ++next_sequence_;
  if (!request.grant) {
    queued_.push_back(id);
    return Submitted{.id = id, .decision = {}};
  }
  waiting_.push_back(id);
  return Submitted{.id = id, .decision = GiveSlot(now)};
}

bool Admission::Before(const Request& candidate, const Request& other) const {
  const bool candidate_aged = candidate.passed_over >= settings_.aging_limit;
  const bool other_aged = other.passed_over >= settings_.aging_limit;
  if (candidate_aged != other_aged) {
    return candidate_aged;
  }
  if (!candidate_aged && candidate.spec.request_class != other.spec.request_class) {
    return candidate.spec.request_class > other.spec.request_class;
  }
  return candidate.sequence < other.sequence;
}

std::optional<RequestId> Admission::PeekNext() const {
  RequestId chosen;
  for (const RequestId id : waiting_) {
    if (!chosen.valid() || Before(*requests_.Find(id), *requests_.Find(chosen))) {
      chosen = id;
    }
  }
  return chosen.valid() ? std::optional<RequestId>(chosen) : std::nullopt;
}

std::optional<RequestId> Admission::ChooseNext() {
  const std::optional<RequestId> chosen = PeekNext();
  if (!chosen) {
    return std::nullopt;
  }
  const RequestClass chosen_class = requests_.Find(*chosen)->spec.request_class;
  for (const RequestId id : waiting_) {
    Request& request = *requests_.Find(id);
    if (id != *chosen && request.spec.request_class < chosen_class &&
        request.passed_over < UINT32_MAX) {
      ++request.passed_over;
    }
  }
  return chosen;
}

Decision Admission::GiveSlot(Tick now) {
  if (!running_.empty() || paused_) {
    return {};
  }
  const std::optional<RequestId> next = ChooseNext();
  if (!next) {
    return {};
  }
  Erase(waiting_, *next);
  Request& request = *requests_.Find(*next);
  request.state = RequestState::kRunning;
  request.run_started = now;
  request.passed_over = 0;
  running_.push_back(*next);
  SyncCohort();
  return Decision{.run = next, .paused = std::nullopt, .refused = {}};
}

bool Admission::DeadlineAllows(const Request& running, const Request& substitute, Tick now) {
  if (!running.spec.deadline) {
    return true;
  }
  const Tick remaining = running.spec.work > running.done ? running.spec.work - running.done : 0;
  const std::optional<Tick> finish =
      Plus(now, remaining)
          .and_then([&](Tick value) { return Plus(value, substitute.spec.work); })
          .and_then([&](Tick value) { return Plus(value, running.spec.switch_cost); })
          .and_then([&](Tick value) { return Plus(value, substitute.spec.switch_cost); });
  return finish && *finish <= *running.spec.deadline;
}

std::expected<Decision, AdmissionError> Admission::Boundary(RequestId id, Tick done, Tick now) {
  Request* request = requests_.Find(id);
  if (request == nullptr) {
    return std::unexpected(AdmissionError::kUnknownRequest);
  }
  if (request->state != RequestState::kRunning) {
    return std::unexpected(AdmissionError::kWrongState);
  }
  request->done = std::max(request->done, done);
  // One pause at a time, and a substitute is not itself paused.
  if (paused_ || settings_.policy == SwitchingPolicy::kRunToCompletion || waiting_.empty()) {
    return Decision{};
  }
  const Tick ran = Since(now, request->run_started);
  if (ran < settings_.minimum_run || request->pauses >= settings_.pause_cap) {
    return Decision{};
  }
  // The candidate is whatever would be chosen next: under priority-aware it
  // must be of a higher class; under time-slicing the quantum must be up.
  const std::optional<RequestId> candidate = PeekNext();
  if (!candidate) {
    return Decision{};
  }
  const bool eligible =
      settings_.policy == SwitchingPolicy::kTimeSlicing
          ? ran >= settings_.time_slice
          : requests_.Find(*candidate)->spec.request_class > request->spec.request_class;
  if (!eligible || !DeadlineAllows(*request, *requests_.Find(*candidate), now)) {
    return Decision{};
  }
  // The running set after the pause must pass the cohort inequality.
  std::vector<RequestId> after = running_;
  Erase(after, id);
  after.push_back(*candidate);
  if (!ledger_.Check(domain_, memory::Change{.cohort = Grants(after)})) {
    return Decision{};
  }
  const std::optional<RequestId> chosen = ChooseNext();  // ages the others
  base::Check(chosen == candidate, "the choice changed between peeking and choosing");
  Erase(running_, id);
  request->state = RequestState::kPaused;
  ++request->pauses;
  paused_ = id;
  substitute_ = candidate;
  Erase(waiting_, *candidate);
  Request& substitute = *requests_.Find(*candidate);
  substitute.state = RequestState::kRunning;
  substitute.run_started = now;
  substitute.passed_over = 0;
  running_.push_back(*candidate);
  SyncCohort();
  return Decision{.run = candidate, .paused = id, .refused = {}};
}

std::expected<Decision, AdmissionError> Admission::Retire(RequestId id, Tick now) {
  Request* request = requests_.Find(id);
  if (request == nullptr) {
    return std::unexpected(AdmissionError::kUnknownRequest);
  }
  Decision decision;
  const RequestState state = request->state;
  if (request->grant) {
    // Never refused; the grant also leaves the ledger's cohort.
    base::Check(ledger_.Retire(*request->grant).has_value(),
                "an admitted request's grant is missing");
  }
  switch (state) {
    case RequestState::kQueued:
      queued_.erase(std::ranges::remove(queued_, id).begin(), queued_.end());
      break;
    case RequestState::kWaiting:
      Erase(waiting_, id);
      break;
    case RequestState::kPaused:
      // Ended while paused: its substitute simply runs on.
      paused_.reset();
      substitute_.reset();
      SyncCohort();
      break;
    case RequestState::kRunning:
      Erase(running_, id);
      if (substitute_ == id) {
        // The paused request resumes next, ahead of every waiting one.
        const RequestId resumed = paused_.value_or(RequestId{});
        base::Check(resumed.valid(), "a substitute without a paused request");
        Request& paused = *requests_.Find(resumed);
        paused.state = RequestState::kRunning;
        paused.run_started = now;
        running_.push_back(resumed);
        paused_.reset();
        substitute_.reset();
        decision.run = resumed;
      }
      // Every change during the pause kept the resumption feasible.
      SyncCohort();
      break;
    case RequestState::kRetired:
      return std::unexpected(AdmissionError::kWrongState);
  }
  (void)requests_.Erase(id);
  return Drained(std::move(decision), now);
}

Decision Admission::Drain(Tick now) {
  Decision decision;
  // Refuse, in arrival order, what can no longer run; each refusal can only
  // shorten the waits of those behind it.
  for (std::size_t i = 0; i < queued_.size();) {
    const RequestId id = queued_[i];
    if (const std::optional<AdmissionError> why = Unrunnable(*requests_.Find(id), now)) {
      decision.refused.emplace_back(id, *why);
      queued_.erase(queued_.begin() + static_cast<std::ptrdiff_t>(i));
      (void)requests_.Erase(id);
    } else {
      ++i;
    }
  }
  // Admit in arrival order; the first that does not fit yet keeps its
  // place (bounded bypass: none), and can still run once capacity returns.
  while (!queued_.empty()) {
    const RequestId id = queued_.front();
    Request& request = *requests_.Find(id);
    auto grant = ledger_.Grant(domain_, request.spec.envelope);
    if (!grant) {
      break;
    }
    request.grant = *grant;
    request.state = RequestState::kWaiting;
    queued_.pop_front();
    waiting_.push_back(id);
  }
  decision.run = GiveSlot(now).run;
  return decision;
}

Decision Admission::Drained(Decision decision, Tick now) {
  Decision drained = Drain(now);
  if (!decision.run) {
    decision.run = drained.run;
  }
  for (auto& refusal : drained.refused) {
    decision.refused.push_back(refusal);
  }
  return decision;
}

std::expected<void, AdmissionError> Admission::Join(RequestId id, Tick now) {
  Request* request = requests_.Find(id);
  if (request == nullptr) {
    return std::unexpected(AdmissionError::kUnknownRequest);
  }
  if (request->state != RequestState::kWaiting) {
    return std::unexpected(AdmissionError::kWrongState);
  }
  if (paused_) {
    return std::unexpected(AdmissionError::kDeferred);  // one pause, one substitute
  }
  std::vector<RequestId> cohort = running_;
  cohort.push_back(id);
  if (auto fits = ledger_.Check(domain_, memory::Change{.cohort = Grants(cohort)}); !fits) {
    return std::unexpected(FromLedger(fits.error()));
  }
  Erase(waiting_, id);
  request->state = RequestState::kRunning;
  request->run_started = now;
  running_.push_back(id);
  SyncCohort();
  return {};
}

std::expected<Decision, AdmissionError> Admission::ReplaceEnvelope(RequestId id,
                                                                   const memory::Envelope& envelope,
                                                                   Tick now) {
  Request* request = requests_.Find(id);
  if (request == nullptr) {
    return std::unexpected(AdmissionError::kUnknownRequest);
  }
  if (!request->grant) {
    return std::unexpected(AdmissionError::kWrongState);
  }
  // The ledger checks the promised resumption too. The substitute's
  // allowance is gone by the resumption, so its own change never breaks
  // it: if it does not fit it is refused, never deferred.
  if (auto replaced = ledger_.Replace(*request->grant, envelope); !replaced) {
    return std::unexpected(FromLedger(replaced.error()));  // the old envelope stands
  }
  request->spec.envelope = envelope;
  return Drained({}, now);
}

std::expected<Decision, AdmissionError> Admission::AddFixed(Bytes bytes, Tick now) {
  if (auto added = ledger_.AddFixed(domain_, bytes); !added) {
    return std::unexpected(FromLedger(added.error()));
  }
  return Drained({}, now);
}

std::expected<Decision, AdmissionError> Admission::AddBackground(Bytes bytes, Tick now) {
  if (auto added = ledger_.AddBackground(domain_, bytes); !added) {
    return std::unexpected(FromLedger(added.error()));
  }
  return Drained({}, now);
}

std::expected<Decision, AdmissionError> Admission::SetBudget(Bytes budget, Tick now) {
  if (auto set = ledger_.SetBudget(domain_, budget); !set) {
    return std::unexpected(FromLedger(set.error()));
  }
  return Drained({}, now);
}

std::expected<Decision, AdmissionError> Admission::ReleaseFixed(Bytes bytes, Tick now) {
  if (!ledger_.ReleaseFixed(domain_, bytes)) {
    return std::unexpected(AdmissionError::kRefused);
  }
  return Drained({}, now);
}

std::expected<Decision, AdmissionError> Admission::ReleaseBackground(Bytes bytes, Tick now) {
  if (!ledger_.ReleaseBackground(domain_, bytes)) {
    return std::unexpected(AdmissionError::kRefused);
  }
  return Drained({}, now);
}

std::optional<RequestState> Admission::StateOf(RequestId id) const {
  const Request* request = requests_.Find(id);
  return request != nullptr ? std::optional<RequestState>(request->state) : std::nullopt;
}

}  // namespace llmp::scheduler
