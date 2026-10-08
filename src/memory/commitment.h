// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The commitment ledger (D-050;
// docs/reservation-policy.md#admission-rule-and-separate-ledgers): promises,
// not bytes in use, keyed by memory domain from the start (D-026). Each
// domain holds its execution budget B, fixed overhead F, non-revocable
// background work J, each guaranteed request's envelope (its retained-state
// bound R_i and its largest phase peak E_i) and the active cohort C, the
// requests running together. Every change that adds a commitment is checked,
// with checked arithmetic, against
//
//   F + R(G) + J + max(max(E_i for i in G), sum(E_i for i in C)) <= B,
//
// which with no cohort is D-050's serial rule. Retiring a grant or
// releasing F or J is never refused. A budget reduction that would break
// the rule is refused: the budget never drops below outstanding claims.
//
// While a request is paused (D-069), the domain also holds a promised
// resumption: the cohort that will run when the substitute ends, with the
// substitute's grant gone. Every change is then checked twice, against the
// state now and against that resumption, so no transaction, whoever makes
// it, can break the promise; one that fits now but not at the resumption
// is refused with kBreaksResumption, which admission defers.
// Occupancy (catalog::Occupancy) is a separate ledger: cached bytes are
// never added here, and a grant maps nothing.

#ifndef LLMP_MEMORY_COMMITMENT_H_
#define LLMP_MEMORY_COMMITMENT_H_

#include <expected>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "base/bytes.h"
#include "base/ids.h"
#include "catalog/catalog.h"

namespace llmp::memory {

using base::Bytes;
using catalog::DomainId;

struct GrantTag {
  static constexpr const char* kName = "grant";
};
// Unique across the ledger's domains.
using GrantId = base::Id<GrantTag>;

// A guaranteed request's envelope in one domain.
struct Envelope {
  Bytes retained;  // R_i: retained state and its maximum future growth
  Bytes phase;     // E_i: the largest additional phase working set
};

enum class CommitmentError : std::uint8_t {
  kDoesNotFit,  // the checked inequality would fail
  kOverflow,    // the arithmetic itself would overflow
  kUnknownDomain,
  kUnknownGrant,
  kUnderflow,  // releasing more F or J than is committed
  kExhausted,
  kBreaksResumption,  // fits now, but not when the paused request resumes
};

std::string ToString(CommitmentError error);

// The commitments' sums in one domain, as the inequality uses them.
struct CommitmentTotals {
  Bytes budget;
  Bytes fixed;
  Bytes background;
  Bytes retained;      // R(G)
  Bytes max_phase;     // max E_i
  Bytes cohort_phase;  // sum of E_i over the active cohort
  // F + R(G) + J + max(max E_i, cohort phase)
  Bytes required;
};

// A hypothetical change, checked without applying it. The explicit member
// initializers let callers name only the fields they change without
// -Wmissing-designated-field-initializers, which readability-redundant-
// member-init would otherwise remove.
// NOLINTBEGIN(readability-redundant-member-init)
struct Change {
  std::optional<GrantId> replace = {};              // this grant's envelope ...
  Envelope replacement = {};                        // ... becomes this
  std::optional<Envelope> add = {};                 // a new grant
  bool add_to_cohort = false;                       // include that grant's phase in the cohort
  std::optional<std::vector<GrantId>> cohort = {};  // a new active cohort
  std::vector<GrantId> exclude = {};                // grants left out entirely
  std::optional<Bytes> fixed = {};
  std::optional<Bytes> background = {};
  std::optional<Bytes> budget = {};
};
// NOLINTEND(readability-redundant-member-init)

class CommitmentLedger {
 public:
  // A domain and its execution budget. Refused if already present.
  std::expected<void, CommitmentError> AddDomain(DomainId domain, Bytes budget);

  // Refused if the current commitments would no longer fit.
  std::expected<void, CommitmentError> SetBudget(DomainId domain, Bytes budget);

  std::expected<void, CommitmentError> AddFixed(DomainId domain, Bytes bytes);
  std::expected<void, CommitmentError> ReleaseFixed(DomainId domain, Bytes bytes);
  std::expected<void, CommitmentError> AddBackground(DomainId domain, Bytes bytes);
  std::expected<void, CommitmentError> ReleaseBackground(DomainId domain, Bytes bytes);

  // Admits a guaranteed request in domain if the rule still passes.
  std::expected<GrantId, CommitmentError> Grant(DomainId domain, const Envelope& envelope);
  // Atomic envelope replacement at a completed boundary: the old envelope
  // stands unless the new one fits, with the active cohort.
  std::expected<void, CommitmentError> Replace(GrantId grant, const Envelope& envelope);
  // Never refused; the grant also leaves the active cohort.
  std::expected<void, CommitmentError> Retire(GrantId grant);

  // Makes these grants, all in domain, its active cohort, if they fit
  // running together. An empty cohort always fits.
  std::expected<void, CommitmentError> SetCohort(DomainId domain, std::span<const GrantId> cohort);

  // Records the promised resumption of an open pause: `cohort` runs
  // together once `substitute`'s grant is gone. Refused unless it fits.
  std::expected<void, CommitmentError> SetResumption(DomainId domain,
                                                     std::span<const GrantId> cohort,
                                                     GrantId substitute);
  // The pause ended (resumed, or the paused request retired).
  std::expected<void, CommitmentError> ClearResumption(DomainId domain);

  // Whether a change would fit, without making it.
  std::expected<void, CommitmentError> Check(DomainId domain, const Change& change) const;

  std::optional<Envelope> EnvelopeOf(GrantId grant) const;
  std::optional<DomainId> DomainOf(GrantId grant) const;
  std::optional<CommitmentTotals> Totals(DomainId domain) const;

 private:
  struct Resumption {
    std::vector<GrantId> cohort;
    GrantId substitute;
  };
  struct Domain {
    Bytes budget;
    Bytes fixed;
    Bytes background;
    std::vector<GrantId> cohort;
    std::optional<Resumption> resumption;
  };

  // The grants, sorted, if every one is live in domain and none repeats.
  std::optional<std::vector<GrantId>> Members(DomainId domain,
                                              std::span<const GrantId> grants) const;
  struct GrantRecord {
    DomainId domain;
    Envelope envelope;
  };

  std::expected<CommitmentTotals, CommitmentError> Evaluate(DomainId domain,
                                                            const Change& change) const;

  std::map<DomainId, Domain> domains_;
  base::SlotTable<GrantTag, GrantRecord> grants_;
};

}  // namespace llmp::memory

#endif  // LLMP_MEMORY_COMMITMENT_H_
