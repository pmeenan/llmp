// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "memory/commitment.h"

#include <algorithm>
#include <expected>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "base/bytes.h"

namespace llmp::memory {

std::string ToString(CommitmentError error) {
  switch (error) {
    case CommitmentError::kDoesNotFit:
      return "the commitments would exceed the execution budget";
    case CommitmentError::kOverflow:
      return "the commitments' arithmetic would overflow";
    case CommitmentError::kUnknownDomain:
      return "unknown memory domain";
    case CommitmentError::kUnknownGrant:
      return "stale or unknown grant";
    case CommitmentError::kUnderflow:
      return "releasing more than is committed";
    case CommitmentError::kExhausted:
      return "the grant table is exhausted";
    case CommitmentError::kBreaksResumption:
      return "the change would not fit when the paused request resumes";
  }
  return "unknown commitment error";
}

std::expected<CommitmentTotals, CommitmentError> CommitmentLedger::Evaluate(
    DomainId domain, const Change& change) const {
  const auto found = domains_.find(domain);
  if (found == domains_.end()) {
    return std::unexpected(CommitmentError::kUnknownDomain);
  }
  const Domain& state = found->second;
  CommitmentTotals totals{.budget = change.budget.value_or(state.budget),
                          .fixed = change.fixed.value_or(state.fixed),
                          .background = change.background.value_or(state.background),
                          .retained = {},
                          .max_phase = {},
                          .cohort_phase = {},
                          .required = {}};
  if (change.replace) {
    const GrantRecord* replaced = grants_.Find(*change.replace);
    if (replaced == nullptr || replaced->domain != domain) {
      return std::unexpected(CommitmentError::kUnknownGrant);
    }
  }
  const std::vector<GrantId>& cohort = change.cohort ? *change.cohort : state.cohort;
  std::optional<Bytes> retained{Bytes()};
  std::optional<Bytes> cohort_phase{Bytes()};
  const auto count = [&](const Envelope& envelope, bool in_cohort) {
    retained = retained.and_then([&](Bytes b) { return b.Plus(envelope.retained); });
    totals.max_phase = std::max(totals.max_phase, envelope.phase);
    if (in_cohort) {
      cohort_phase = cohort_phase.and_then([&](Bytes b) { return b.Plus(envelope.phase); });
    }
  };
  grants_.ForEach([&](GrantId id, const GrantRecord& grant) {
    if (grant.domain != domain || std::ranges::find(change.exclude, id) != change.exclude.end()) {
      return;
    }
    const Envelope& envelope =
        change.replace && *change.replace == id ? change.replacement : grant.envelope;
    count(envelope, std::ranges::find(cohort, id) != cohort.end());
  });
  if (change.add) {
    count(*change.add, change.add_to_cohort);
  }
  if (!retained || !cohort_phase) {
    return std::unexpected(CommitmentError::kOverflow);
  }
  totals.retained = *retained;
  totals.cohort_phase = *cohort_phase;
  const std::optional<Bytes> required =
      totals.fixed.Plus(totals.background)
          .and_then([&](Bytes b) { return b.Plus(totals.retained); })
          .and_then(
              [&](Bytes b) { return b.Plus(std::max(totals.max_phase, totals.cohort_phase)); });
  if (!required) {
    return std::unexpected(CommitmentError::kOverflow);
  }
  totals.required = *required;
  return totals;
}

std::expected<void, CommitmentError> CommitmentLedger::Check(DomainId domain,
                                                             const Change& change) const {
  auto totals = Evaluate(domain, change);
  if (!totals) {
    return std::unexpected(totals.error());
  }
  if (totals->required > totals->budget) {
    return std::unexpected(CommitmentError::kDoesNotFit);
  }
  const std::optional<Resumption>& resumption = domains_.at(domain).resumption;
  if (!resumption) {
    return {};
  }
  // The same change, at the resumption: its cohort running, the
  // substitute gone.
  Change resumed = change;
  resumed.cohort = resumption->cohort;
  resumed.add_to_cohort = false;  // a newcomer is not in the promised resumption
  resumed.exclude.push_back(resumption->substitute);
  totals = Evaluate(domain, resumed);
  if (!totals) {
    return std::unexpected(totals.error() == CommitmentError::kOverflow
                               ? CommitmentError::kBreaksResumption
                               : totals.error());
  }
  if (totals->required > totals->budget) {
    return std::unexpected(CommitmentError::kBreaksResumption);
  }
  return {};
}

std::expected<void, CommitmentError> CommitmentLedger::AddDomain(DomainId domain, Bytes budget) {
  if (!domain.valid() || domains_.contains(domain)) {
    return std::unexpected(CommitmentError::kUnknownDomain);
  }
  domains_.emplace(domain, Domain{.budget = budget,
                                  .fixed = {},
                                  .background = {},
                                  .cohort = {},
                                  .resumption = std::nullopt});
  return {};
}

std::expected<void, CommitmentError> CommitmentLedger::SetBudget(DomainId domain, Bytes budget) {
  if (auto fits = Check(domain, Change{.budget = budget}); !fits) {
    return fits;
  }
  domains_.at(domain).budget = budget;
  return {};
}

std::expected<void, CommitmentError> CommitmentLedger::AddFixed(DomainId domain, Bytes bytes) {
  const auto found = domains_.find(domain);
  if (found == domains_.end()) {
    return std::unexpected(CommitmentError::kUnknownDomain);
  }
  const std::optional<Bytes> fixed = found->second.fixed.Plus(bytes);
  if (!fixed) {
    return std::unexpected(CommitmentError::kOverflow);
  }
  if (auto fits = Check(domain, Change{.fixed = fixed}); !fits) {
    return fits;
  }
  found->second.fixed = *fixed;
  return {};
}

std::expected<void, CommitmentError> CommitmentLedger::ReleaseFixed(DomainId domain, Bytes bytes) {
  const auto found = domains_.find(domain);
  if (found == domains_.end()) {
    return std::unexpected(CommitmentError::kUnknownDomain);
  }
  const std::optional<Bytes> fixed = found->second.fixed.Minus(bytes);
  if (!fixed) {
    return std::unexpected(CommitmentError::kUnderflow);
  }
  found->second.fixed = *fixed;
  return {};
}

std::expected<void, CommitmentError> CommitmentLedger::AddBackground(DomainId domain, Bytes bytes) {
  const auto found = domains_.find(domain);
  if (found == domains_.end()) {
    return std::unexpected(CommitmentError::kUnknownDomain);
  }
  const std::optional<Bytes> background = found->second.background.Plus(bytes);
  if (!background) {
    return std::unexpected(CommitmentError::kOverflow);
  }
  if (auto fits = Check(domain, Change{.background = background}); !fits) {
    return fits;
  }
  found->second.background = *background;
  return {};
}

std::expected<void, CommitmentError> CommitmentLedger::ReleaseBackground(DomainId domain,
                                                                         Bytes bytes) {
  const auto found = domains_.find(domain);
  if (found == domains_.end()) {
    return std::unexpected(CommitmentError::kUnknownDomain);
  }
  const std::optional<Bytes> background = found->second.background.Minus(bytes);
  if (!background) {
    return std::unexpected(CommitmentError::kUnderflow);
  }
  found->second.background = *background;
  return {};
}

std::expected<GrantId, CommitmentError> CommitmentLedger::Grant(DomainId domain,
                                                                const Envelope& envelope) {
  if (auto fits = Check(domain, Change{.add = envelope}); !fits) {
    return std::unexpected(fits.error());
  }
  const GrantId id = grants_.Insert(GrantRecord{.domain = domain, .envelope = envelope});
  if (!id.valid()) {
    return std::unexpected(CommitmentError::kExhausted);
  }
  return id;
}

std::expected<void, CommitmentError> CommitmentLedger::Replace(GrantId grant,
                                                               const Envelope& envelope) {
  const GrantRecord* found = grants_.Find(grant);
  if (found == nullptr) {
    return std::unexpected(CommitmentError::kUnknownGrant);
  }
  if (auto fits = Check(found->domain, Change{.replace = grant, .replacement = envelope}); !fits) {
    return fits;  // the old envelope stands
  }
  grants_.Find(grant)->envelope = envelope;
  return {};
}

std::expected<void, CommitmentError> CommitmentLedger::Retire(GrantId grant) {
  const GrantRecord* found = grants_.Find(grant);
  if (found == nullptr) {
    return std::unexpected(CommitmentError::kUnknownGrant);
  }
  Domain& state = domains_.at(found->domain);
  state.cohort.erase(std::ranges::remove(state.cohort, grant).begin(), state.cohort.end());
  if (state.resumption) {
    std::vector<GrantId>& resumed = state.resumption->cohort;
    resumed.erase(std::ranges::remove(resumed, grant).begin(), resumed.end());
  }
  (void)grants_.Erase(grant);
  return {};
}

std::optional<std::vector<GrantId>> CommitmentLedger::Members(
    DomainId domain, std::span<const GrantId> grants) const {
  std::vector<GrantId> members(grants.begin(), grants.end());
  std::ranges::sort(members);
  if (std::ranges::adjacent_find(members) != members.end()) {
    return std::nullopt;  // a member named twice
  }
  for (const GrantId member : members) {
    const GrantRecord* grant = grants_.Find(member);
    if (grant == nullptr || grant->domain != domain) {
      return std::nullopt;
    }
  }
  return members;
}

std::expected<void, CommitmentError> CommitmentLedger::SetCohort(DomainId domain,
                                                                 std::span<const GrantId> cohort) {
  if (!domains_.contains(domain)) {
    return std::unexpected(CommitmentError::kUnknownDomain);
  }
  std::optional<std::vector<GrantId>> members = Members(domain, cohort);
  if (!members) {
    return std::unexpected(CommitmentError::kUnknownGrant);
  }
  if (auto fits = Check(domain, Change{.cohort = members}); !fits) {
    return fits;
  }
  domains_.at(domain).cohort = *std::move(members);
  return {};
}

std::expected<void, CommitmentError> CommitmentLedger::SetResumption(
    DomainId domain, std::span<const GrantId> cohort, GrantId substitute) {
  const auto found = domains_.find(domain);
  if (found == domains_.end()) {
    return std::unexpected(CommitmentError::kUnknownDomain);
  }
  std::optional<std::vector<GrantId>> members = Members(domain, cohort);
  const GrantRecord* excluded = grants_.Find(substitute);
  if (!members || excluded == nullptr || excluded->domain != domain ||
      std::ranges::binary_search(*members, substitute)) {
    return std::unexpected(CommitmentError::kUnknownGrant);
  }
  // Checked as a change with no pause open.
  std::optional<Resumption> previous = std::move(found->second.resumption);
  found->second.resumption.reset();
  auto fits = Check(domain, Change{.cohort = members, .exclude = {substitute}});
  if (!fits) {
    found->second.resumption = std::move(previous);
    return fits;
  }
  found->second.resumption = Resumption{.cohort = *std::move(members), .substitute = substitute};
  return {};
}

std::expected<void, CommitmentError> CommitmentLedger::ClearResumption(DomainId domain) {
  const auto found = domains_.find(domain);
  if (found == domains_.end()) {
    return std::unexpected(CommitmentError::kUnknownDomain);
  }
  found->second.resumption.reset();
  return {};
}

std::optional<Envelope> CommitmentLedger::EnvelopeOf(GrantId grant) const {
  const GrantRecord* found = grants_.Find(grant);
  return found != nullptr ? std::optional<Envelope>(found->envelope) : std::nullopt;
}

std::optional<DomainId> CommitmentLedger::DomainOf(GrantId grant) const {
  const GrantRecord* found = grants_.Find(grant);
  return found != nullptr ? std::optional<DomainId>(found->domain) : std::nullopt;
}

std::optional<CommitmentTotals> CommitmentLedger::Totals(DomainId domain) const {
  auto totals = Evaluate(domain, Change{});
  return totals ? std::optional<CommitmentTotals>(*totals) : std::nullopt;
}

}  // namespace llmp::memory
