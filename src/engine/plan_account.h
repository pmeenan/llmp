// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// What a runner's plans and graphs hold, whatever their kernels
// (docs/engine.md): the charge against the node beside the catalog
// (PlanAccount, D-090 as amended 2026-10-02), the order plans were last
// used in and the step under way that a reclaim spares (PlanStep), and
// what the node's one reclaim order (memory/reclaim.h) sees of a plan
// cache (PlanCacheBase), which the plan cache (plan_cache.h) builds on.

#ifndef LLMP_ENGINE_PLAN_ACCOUNT_H_
#define LLMP_ENGINE_PLAN_ACCOUNT_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <span>
#include <utility>
#include <vector>

#include "memory/reclaim.h"

namespace llmp::engine {

// The order a runner's plans were last used in, across its caches
// (PlanCache's least recently used; one counter for the process).
std::uint64_t NextPlanUse();

// A runner's step (a chunk, a draft, a verify, a wave) borrows plans from
// its caches from its first Find to its job's end. While one is under way
// on this thread (from a PlanStep's construction to its destruction; they
// nest), a reclaim spares every plan used since it began: a charge that
// asks the node to reclaim (PlanAccount) never drops what the step holds.
// Between steps nothing is borrowed, and every plan is a candidate.
class PlanStep {
 public:
  PlanStep();
  PlanStep(const PlanStep&) = delete;
  PlanStep& operator=(const PlanStep&) = delete;
  PlanStep(PlanStep&&) = delete;
  PlanStep& operator=(PlanStep&&) = delete;
  ~PlanStep();
};
// The first use a reclaim must spare: the outermost step's start, or
// kNoStep between steps.
inline constexpr std::uint64_t kNoStep = std::numeric_limits<std::uint64_t>::max();
std::uint64_t PlanStepStart();

// What a model's plans and graphs hold, as the node charges it
// (PagedNode::ChargeHost; D-090 as amended 2026-10-02): counted inside the
// execution budget beside the catalog's extents, past a floor the start's
// guard sets apart (what one step of the model needs at once). A charge
// that does not fit has the node reclaim through its one order first. A
// required charge (a plan the step under way needs) is always taken; one
// that is not (a graph's capture) is refused when it still does not fit,
// and the plan then runs launch by launch. Unbound (a host test, or before
// the node binds it), it only counts.
class PlanAccount {
 public:
  using ChargeFn = std::function<bool(std::uint64_t bytes, bool required)>;
  using ReleaseFn = std::function<void(std::uint64_t bytes)>;
  void Bind(ChargeFn charge, ReleaseFn release) {
    charge_ = std::move(charge);
    release_ = std::move(release);
  }
  bool Charge(std::uint64_t bytes, bool required) {
    if (bytes != 0 && charge_ && !charge_(bytes, required)) {
      return false;
    }
    bytes_ += bytes;
    return true;
  }
  void Uncharge(std::uint64_t bytes) {
    bytes = std::min(bytes, bytes_);
    bytes_ -= bytes;
    if (bytes != 0 && release_) {
      release_(bytes);
    }
  }
  std::uint64_t bytes() const { return bytes_; }

 private:
  ChargeFn charge_;
  ReleaseFn release_;
  std::uint64_t bytes_ = 0;
};

// What the reclaimer sees of a plan cache, whatever its key and plan type:
// its plans and graphs as candidates (none a step under way holds), and a
// candidate's reclaim.
class PlanCacheBase {
 public:
  PlanCacheBase() = default;
  PlanCacheBase(const PlanCacheBase&) = delete;
  PlanCacheBase& operator=(const PlanCacheBase&) = delete;
  PlanCacheBase(PlanCacheBase&&) = delete;
  PlanCacheBase& operator=(PlanCacheBase&&) = delete;
  virtual ~PlanCacheBase() = default;
  // Each plan (with its graphs: kPlan) and each plan's graphs (kGraph)
  // not used since PlanStepStart(), `owner`'s, `running` if its model is.
  virtual void Collect(std::uint32_t owner, bool running,
                       std::vector<memory::ReclaimCandidate>& out) = 0;
  // Drops the plan `serial` with its graphs (kPlan) or only its graphs
  // (kGraph), between jobs: the bytes it freed, 0 if it is not this
  // cache's or a step under way holds it.
  virtual std::uint64_t Reclaim(memory::ReclaimKind kind, std::uint64_t serial) = 0;
  virtual std::size_t size() const = 0;
  virtual std::size_t graphs() const = 0;
  virtual std::uint64_t host_bytes() const = 0;
  virtual std::uint64_t graph_bytes() const = 0;
  // What its graphs took of the device's free memory at their captures
  // (PlanRuns::measured_bytes), to check the count against.
  virtual std::uint64_t graph_measured_bytes() const = 0;
  virtual void Clear() = 0;
  // Plans and graphs dropped by Reclaim since it was made.
  std::uint64_t reclaimed_plans() const { return reclaimed_plans_; }
  std::uint64_t reclaimed_graphs() const { return reclaimed_graphs_; }

 protected:
  std::uint64_t reclaimed_plans_ = 0;
  std::uint64_t reclaimed_graphs_ = 0;
};

// Over a runner's caches: every candidate, and one candidate's reclaim.
void CollectPlans(std::span<PlanCacheBase* const> caches, std::uint32_t owner, bool running,
                  std::vector<memory::ReclaimCandidate>& out);
std::uint64_t ReclaimPlan(std::span<PlanCacheBase* const> caches, memory::ReclaimKind kind,
                          std::uint64_t serial);

}  // namespace llmp::engine

#endif  // LLMP_ENGINE_PLAN_ACCOUNT_H_
