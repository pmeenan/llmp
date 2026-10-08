// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// The node's one reclaim order (D-055 as amended 2026-10-02,
// docs/retention-policy.md#victims-spill-and-exhaustion): everything that
// can give memory back under pressure, whoever owns it, chosen by one rule.
// It only chooses; each victim's owner reclaims it (the runtime's
// reclaimer drops a plan or a graph, spills an idle conversation), between
// completed units, and never anything a unit in flight uses.
//
// The rule is GreedyDual-Size over expected costs per byte: each
// candidate's priority is H = L + c × p, where c is its kind's measured
// cost to restore a GiB freed, p the chance it is used again, and L the
// inflation value (ReclaimInflation) when the candidate was last used.
// The lowest H goes first, and each reclaim raises L to the priority it
// took. The chance of reuse decays with staleness directly: it halves
// with every reclaim since the candidate's last use and with every
// kReuseHalfLifeSeconds of idleness, so a costly entry left unused (a
// one-off prompt's plan) falls behind a fresh cheap one (an idle
// conversation) within a few reclaims or minutes, not the dozens of
// reclaims L alone would take to rise past their cost difference. Both
// terms only fall with staleness, so among candidates of one cost (one
// kind) the order is strict least recent use, never largest first; a
// costly kind (plans) outlasts a cheap one (idle state) while both are in
// use. A model that is not running goes before the running one at equal
// priority. Costs are measured, not assumed: a plan's is the seconds its
// planning took (and its graphs' captures), a graph's its capture and
// instantiation, idle conversation state the seconds its spill (only what
// changed since its last spill) and its restore take at the node's
// measured rates, idle weights their page-in at the measured rate. A
// kind's cost a GiB is its candidates' seconds over their GiB.
//
// Vendor-free and host-only: the CPU tests hold the order.

#ifndef JITLLM_MEMORY_RECLAIM_H_
#define JITLLM_MEMORY_RECLAIM_H_

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <span>
#include <string_view>
#include <vector>

namespace jitllm::memory {

// The chance of reuse halves with every reclaim since a candidate's last
// use and with every this many seconds idle (ReclaimPriority).
inline constexpr double kReuseHalfLifeReclaims = 1.0;
inline constexpr double kReuseHalfLifeSeconds = 300.0;

// What an owner records at each use of something reclaimable: the
// inflation value, the reclaims taken so far and the time.
struct ReclaimStamp {
  double inflation = 0;
  std::uint64_t reclaims = 0;
  std::chrono::steady_clock::time_point at;
};
// Now (process-wide counters, as engine::NextPlanUse's ticks are).
ReclaimStamp StampUse();

enum class ReclaimKind : std::uint8_t {
  kIdleWeights,   // clean weights of a model between its requests: paged in again
  kGraph,         // a captured graph, its plan kept: captured again from it
  kPlan,          // a planned shape with its graphs: planned again at its next use
  kIdleState,     // an idle conversation's resident state: spilled, restored at its next turn
  kTokenHistory,  // idle host tokens and state: discarded, recomputed at next turn
};
inline constexpr std::size_t kReclaimKinds = 5;

std::string_view ToString(ReclaimKind kind);

struct ReclaimCandidate {
  ReclaimKind kind = ReclaimKind::kPlan;
  std::uint32_t owner = 0;     // the model's registration index
  std::uint64_t id = 0;        // the owner's handle for it
  std::uint64_t bytes = 0;     // what reclaiming it frees (a plan's includes its graphs')
  std::uint64_t last_use = 0;  // a tick comparable within the kind: smaller is older
  double inflation = 0;        // ReclaimInflation() at its last use
  double restore_seconds = 0;  // what bringing it back costs, measured
  bool running = false;        // its model is the one running
  // Staleness: reclaims taken since its last use, and seconds idle since.
  std::uint64_t reclaims_since = 0;
  double idle_seconds = 0;
};

// Sets a candidate's inflation and staleness from its owner's stamp, now.
void SetUse(ReclaimCandidate& candidate, const ReclaimStamp& stamp);

// GreedyDual's inflation value L, process-wide (as engine::NextPlanUse's
// ticks are): owners record it at each use of a candidate; Raise sets it
// to a reclaimed candidate's priority if that is higher (never decreases)
// and counts one reclaim taken.
double ReclaimInflation();
void RaiseReclaimInflation(double priority);
std::uint64_t ReclaimsTaken();

// Each kind's measured cost: restore seconds per GiB freed over its
// candidates (0 for a kind with none).
std::array<double, kReclaimKinds> KindCosts(std::span<const ReclaimCandidate> candidates);

// A candidate's priority H under `cost` (KindCosts): its inflation at its
// last use plus its kind's cost a GiB times its chance of reuse (halved
// per kReuseHalfLifeReclaims reclaims and per kReuseHalfLifeSeconds idle).
double ReclaimPriority(const ReclaimCandidate& candidate,
                       const std::array<double, kReclaimKinds>& cost);

// Every candidate's index, in reclaim order (above). Deterministic: ties
// break by kind, the running model's last, last use, owner, then id.
std::vector<std::size_t> ReclaimOrder(std::span<const ReclaimCandidate> candidates);

// A candidate's chance of reuse: halved per kReuseHalfLifeReclaims
// reclaims and per kReuseHalfLifeSeconds idle since its last use.
double ReuseChance(const ReclaimCandidate& candidate);

// Candidates in reclaim order until their bytes cover `needed`, only those
// whose priority is below `below` (an optional charge reclaims only what
// costs less to restore than what it charges). A plan and its graph are
// counted once. The order ranks a GiB, so the same is taken over each
// smaller set of the kinds present too, and the selection that covers the
// need at the least total expected cost to restore (each victim's restore
// seconds times its chance of reuse) wins: a small need takes a few small
// plans and graphs rather than a whole idle conversation when they cost
// less to bring back. Within a kind the order, strict least recent use,
// stands, but that an entry larger than what is still needed may give way
// to a later one of its kind that fits. Victims chosen this way can stand
// later in the full order, so raising the inflation to their priorities
// (RaiseReclaimInflation) can raise it more than the full order's first
// victims would have. `sufficient` says whether they cover it; a caller
// that needs all of it reclaims nothing otherwise.
struct ReclaimPlan {
  std::vector<std::size_t> victims;  // indices into the candidates
  std::vector<double> priorities;    // each victim's priority, for ReclaimInflation
  std::uint64_t bytes = 0;
  bool sufficient = false;
};
ReclaimPlan SelectReclaim(std::span<const ReclaimCandidate> candidates, std::uint64_t needed,
                          double below = std::numeric_limits<double>::infinity());

// One reclaim's rounds (the runtime's Server::Reclaim). Each round has
// `gather` collect the candidates afresh (and set `below`, SelectReclaim's
// bound), selects for what is still needed, and has `take` reclaim the
// victims in order until it is covered; `take` returns what a victim gave
// back, and each one that gave something raises the inflation to its
// priority (RaiseReclaimInflation). A selection that cannot cover what is
// still needed takes nothing (with `partial`, whatever it has). A victim
// that gives back less than it counted (held, or gone meanwhile) leaves
// the round short: the next one, at most kReclaimRounds in all, selects for
// the rest without any victim tried before. So a reclaim takes all it was
// asked for or nothing, but for what earlier rounds took before a victim
// failed, which stays taken (nothing reclaimed comes back): it then ends
// short of `needed`, and its caller, which compares what it freed with
// what it asked for, waits or refuses as for nothing.
inline constexpr int kReclaimRounds = 4;
struct ReclaimRun {
  std::uint64_t freed = 0;
  bool short_of_need = false;  // a selection could not cover what was still needed
};
using GatherReclaim = std::function<void(std::vector<ReclaimCandidate>& out, double& below)>;
using TakeReclaim = std::function<std::uint64_t(const ReclaimCandidate& victim)>;
// Consecutive selected idle-weight victims may be evicted in one owner call.
// `freed` has the same size, initially zero: publish each victim's actual
// completed release (at most its bytes), including releases before a failure.
using TakeWeightBatch =
    std::function<void(std::span<const ReclaimCandidate> victims, std::span<std::uint64_t> freed)>;
ReclaimRun RunReclaim(std::uint64_t needed, bool partial, const GatherReclaim& gather,
                      const TakeReclaim& take, const TakeWeightBatch& weights = {});

// Takes the running model's in-use floor out of `candidates`: its most
// recently used plans, newest first, until their own bytes (without their
// graphs) reach `floor_bytes` (what one of its steps holds at once,
// Served::plan_floor_bytes), with those plans' graphs. Its next step then
// finds them, so no reclaim (a capacity refusal's, a swap's, pressure from
// outside) has it plan and capture them again at once. What a step under
// way uses never is a candidate (engine/planned.h, PlanStep). A floor of 0
// keeps nothing.
void ProtectFloor(std::vector<ReclaimCandidate>& candidates, std::uint64_t floor_bytes);

}  // namespace jitllm::memory

#endif  // JITLLM_MEMORY_RECLAIM_H_
