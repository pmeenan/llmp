// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "memory/reclaim.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <span>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

namespace jitllm::memory {
namespace {

std::atomic<double> inflation{0.0};
std::atomic<std::uint64_t> reclaims{0};

}  // namespace

double ReclaimInflation() { return inflation.load(std::memory_order_relaxed); }

std::uint64_t ReclaimsTaken() { return reclaims.load(std::memory_order_relaxed); }

void RaiseReclaimInflation(double priority) {
  reclaims.fetch_add(1, std::memory_order_relaxed);
  double now = inflation.load(std::memory_order_relaxed);
  while (priority > now &&
         !inflation.compare_exchange_weak(now, priority, std::memory_order_relaxed)) {
  }
}

ReclaimStamp StampUse() {
  return {.inflation = ReclaimInflation(),
          .reclaims = ReclaimsTaken(),
          .at = std::chrono::steady_clock::now()};
}

void SetUse(ReclaimCandidate& candidate, const ReclaimStamp& stamp) {
  candidate.inflation = stamp.inflation;
  const std::uint64_t taken = ReclaimsTaken();
  candidate.reclaims_since = taken > stamp.reclaims ? taken - stamp.reclaims : 0;
  const auto idle = std::chrono::steady_clock::now() - stamp.at;
  candidate.idle_seconds = stamp.at == std::chrono::steady_clock::time_point{}
                               ? 0
                               : std::max(0.0, std::chrono::duration<double>(idle).count());
}

std::string_view ToString(ReclaimKind kind) {
  switch (kind) {
    case ReclaimKind::kIdleWeights:
      return "idle weights";
    case ReclaimKind::kGraph:
      return "graph";
    case ReclaimKind::kPlan:
      return "plan";
    case ReclaimKind::kIdleState:
      return "idle conversation state";
  }
  return "unknown";
}

std::array<double, kReclaimKinds> KindCosts(std::span<const ReclaimCandidate> candidates) {
  std::array<double, kReclaimKinds> seconds{};
  std::array<double, kReclaimKinds> bytes{};
  for (const ReclaimCandidate& c : candidates) {
    const auto k = static_cast<std::size_t>(c.kind);
    if (k < kReclaimKinds && c.bytes != 0) {
      seconds[k] += std::max(c.restore_seconds, 0.0);
      bytes[k] += static_cast<double>(c.bytes);
    }
  }
  std::array<double, kReclaimKinds> cost{};
  for (std::size_t k = 0; k < kReclaimKinds; ++k) {
    cost[k] = bytes[k] > 0 ? seconds[k] / (bytes[k] / static_cast<double>(1ULL << 30U)) : 0;
  }
  return cost;
}

double ReclaimPriority(const ReclaimCandidate& candidate,
                       const std::array<double, kReclaimKinds>& cost) {
  const auto k = static_cast<std::size_t>(candidate.kind);
  const double reuse =
      std::exp2(-((static_cast<double>(candidate.reclaims_since) / kReuseHalfLifeReclaims) +
                  (candidate.idle_seconds / kReuseHalfLifeSeconds)));
  return candidate.inflation + ((k < kReclaimKinds ? cost[k] : 0.0) * reuse);
}

std::vector<std::size_t> ReclaimOrder(std::span<const ReclaimCandidate> candidates) {
  const std::array<double, kReclaimKinds> cost = KindCosts(candidates);
  std::vector<std::size_t> order;
  order.reserve(candidates.size());
  for (std::size_t i = 0; i < candidates.size(); ++i) {
    if (candidates[i].bytes != 0 && static_cast<std::size_t>(candidates[i].kind) < kReclaimKinds) {
      order.push_back(i);
    }
  }
  const auto key = [&](std::size_t i) {
    const ReclaimCandidate& c = candidates[i];
    return std::tuple(ReclaimPriority(c, cost), static_cast<std::size_t>(c.kind), c.running,
                      c.last_use, c.owner, c.id);
  };
  std::ranges::sort(order, [&](std::size_t a, std::size_t b) { return key(a) < key(b); });
  return order;
}

double ReuseChance(const ReclaimCandidate& candidate) {
  return std::exp2(-((static_cast<double>(candidate.reclaims_since) / kReuseHalfLifeReclaims) +
                     (candidate.idle_seconds / kReuseHalfLifeSeconds)));
}

namespace {

// The order's selection (SelectReclaim) among the candidates whose kind is
// in `kinds` (bit k: kind k), and in `expected` the victims' expected cost
// to restore (each its restore seconds times its chance of reuse). With
// `skip_large`, an entry larger than what is still needed is passed over
// while a later entry of its kind would fit in what is left (a newer small
// conversation before an old large one); least recent use stands
// otherwise.
ReclaimPlan SelectFrom(std::span<const ReclaimCandidate> candidates,
                       std::span<const std::size_t> all_order, std::uint64_t needed, double below,
                       const std::array<double, kReclaimKinds>& cost, unsigned kinds,
                       bool skip_large, double& expected) {
  ReclaimPlan plan;
  expected = 0;
  // A plan's bytes include its graphs': a graph taken before its plan is
  // not counted again, and a graph whose plan was taken is skipped.
  std::vector<std::pair<std::uint32_t, std::uint64_t>> plans;
  std::vector<std::pair<std::uint32_t, std::uint64_t>> graphs;
  const auto graph_of = [&](const ReclaimCandidate& plan_candidate) -> const ReclaimCandidate* {
    for (const ReclaimCandidate& c : candidates) {
      if (c.kind == ReclaimKind::kGraph && c.owner == plan_candidate.owner &&
          c.id == plan_candidate.id) {
        return &c;
      }
    }
    return nullptr;
  };
  std::vector<std::size_t> order;
  for (const std::size_t i : all_order) {
    if ((kinds & (1U << static_cast<unsigned>(candidates[i].kind))) != 0) {
      order.push_back(i);
    }
  }
  std::vector<bool> taken(order.size(), false);
  for (std::size_t p = 0; p < order.size(); ++p) {
    if (plan.bytes >= needed) {
      break;
    }
    if (taken[p]) {
      continue;
    }
    std::size_t at = p;
    // No more than needed: a candidate far larger than what is left (an
    // idle conversation for a plan's few KiB) gives way to a later one of
    // another kind that covers the rest at no more absolute cost to
    // restore (within a kind, least recent use stands).
    const std::uint64_t left = needed - plan.bytes;
    if (skip_large && candidates[order[p]].bytes > left) {
      bool later_fits = false;
      for (std::size_t q = p + 1; q < order.size() && !later_fits; ++q) {
        const ReclaimCandidate& later = candidates[order[q]];
        later_fits = !taken[q] && later.kind == candidates[order[p]].kind && later.bytes <= left;
      }
      if (later_fits) {
        continue;
      }
    }
    if (candidates[order[p]].bytes / 8 > left) {
      for (std::size_t q = p + 1; q < order.size(); ++q) {
        const ReclaimCandidate& alt = candidates[order[q]];
        if (!taken[q] && alt.kind != candidates[order[p]].kind && alt.bytes >= left &&
            alt.bytes < candidates[order[p]].bytes &&
            alt.restore_seconds <= candidates[order[p]].restore_seconds &&
            ReclaimPriority(alt, cost) < below) {
          at = q;
          break;
        }
      }
    }
    taken[at] = true;
    if (at != p) {
      --p;  // the one passed over stays next in the order
    }
    const std::size_t i = order[at];
    const ReclaimCandidate& c = candidates[i];
    const double priority = ReclaimPriority(c, cost);
    if (!(priority < below)) {
      break;  // the rest cost as much to restore as what is charged, or more
    }
    std::uint64_t b = c.bytes;
    double seconds = std::max(c.restore_seconds, 0.0);
    const auto handle = std::pair(c.owner, c.id);
    if (c.kind == ReclaimKind::kGraph) {
      if (std::ranges::find(plans, handle) != plans.end()) {
        continue;  // gone with its plan
      }
      graphs.push_back(handle);
    } else if (c.kind == ReclaimKind::kPlan) {
      if (std::ranges::find(graphs, handle) != graphs.end()) {
        if (const ReclaimCandidate* graph = graph_of(c); graph != nullptr) {
          b -= std::min(b, graph->bytes);
          seconds = std::max(0.0, seconds - std::max(graph->restore_seconds, 0.0));
        }
      }
      plans.push_back(handle);
    }
    plan.victims.push_back(i);
    plan.priorities.push_back(priority);
    plan.bytes = b > UINT64_MAX - plan.bytes ? UINT64_MAX : plan.bytes + b;
    expected += seconds * ReuseChance(c);
  }
  plan.sufficient = plan.bytes >= needed;
  return plan;
}

}  // namespace

ReclaimPlan SelectReclaim(std::span<const ReclaimCandidate> candidates, std::uint64_t needed,
                          double below) {
  const std::array<double, kReclaimKinds> cost = KindCosts(candidates);
  const std::vector<std::size_t> order = ReclaimOrder(candidates);
  unsigned present = 0;
  for (const std::size_t i : order) {
    present |= 1U << static_cast<unsigned>(candidates[i].kind);
  }
  // The order over every kind, and over each smaller set of the kinds
  // present: the cheapest total expected cost to restore that covers the
  // need wins (ties keep the larger set's, every kind's first). The order
  // ranks a GiB; a small need can cost less to restore from a costlier
  // kind's few small entries than from one large cheap one (a whole idle
  // conversation for a few MiB). Each selection keeps the order within its
  // kinds, but for one variant: an entry larger than what is still needed
  // may give way to a later, smaller one of its kind that fits (an old
  // large conversation for a newer small one).
  double best_expected = 0;
  ReclaimPlan best =
      SelectFrom(candidates, order, needed, below, cost, present, false, best_expected);
  for (unsigned kinds = present; kinds != 0; kinds = (kinds - 1) & present) {
    for (const bool skip_large : {false, true}) {
      if (kinds == present && !skip_large) {
        continue;
      }
      double expected = 0;
      ReclaimPlan plan =
          SelectFrom(candidates, order, needed, below, cost, kinds, skip_large, expected);
      if (plan.sufficient && (!best.sufficient || expected < best_expected)) {
        best = std::move(plan);
        best_expected = expected;
      }
    }
  }
  return best;
}

ReclaimRun RunReclaim(std::uint64_t needed, bool partial, const GatherReclaim& gather,
                      const TakeReclaim& take) {
  ReclaimRun run;
  // Every victim tried so far, by kind, owner and handle: taken (gone) or
  // failed, never offered again within this reclaim.
  std::vector<std::tuple<ReclaimKind, std::uint32_t, std::uint64_t>> tried;
  for (int round = 0; round < kReclaimRounds && run.freed < needed; ++round) {
    std::vector<ReclaimCandidate> candidates;
    double below = std::numeric_limits<double>::infinity();
    gather(candidates, below);
    std::erase_if(candidates, [&](const ReclaimCandidate& c) {
      return std::ranges::find(tried, std::tuple(c.kind, c.owner, c.id)) != tried.end();
    });
    const ReclaimPlan plan = SelectReclaim(candidates, needed - run.freed, below);
    if (!plan.sufficient) {
      run.short_of_need = true;
      if (!partial || plan.victims.empty()) {
        break;
      }
    }
    std::uint64_t got_round = 0;
    for (std::size_t v = 0; v < plan.victims.size() && run.freed < needed; ++v) {
      const ReclaimCandidate& c = candidates[plan.victims[v]];
      tried.emplace_back(c.kind, c.owner, c.id);
      const std::uint64_t got = take(c);
      if (got != 0) {
        RaiseReclaimInflation(plan.priorities[v]);
        run.freed += got;
        got_round += got;
      }
    }
    // With `partial`, a selection short of the need was all there was.
    if (!plan.sufficient && got_round != 0) {
      break;
    }
  }
  return run;
}

void ProtectFloor(std::vector<ReclaimCandidate>& candidates, std::uint64_t floor_bytes) {
  if (floor_bytes == 0) {
    return;
  }
  std::vector<std::size_t> plans;
  for (std::size_t i = 0; i < candidates.size(); ++i) {
    if (candidates[i].running && candidates[i].kind == ReclaimKind::kPlan) {
      plans.push_back(i);
    }
  }
  const auto newest = [&](std::size_t a, std::size_t b) {
    const ReclaimCandidate& x = candidates[a];
    const ReclaimCandidate& y = candidates[b];
    return std::tuple(x.last_use, x.owner, x.id) > std::tuple(y.last_use, y.owner, y.id);
  };
  std::ranges::sort(plans, newest);
  const auto graph_bytes = [&](const ReclaimCandidate& plan) -> std::uint64_t {
    for (const ReclaimCandidate& c : candidates) {
      if (c.running && c.kind == ReclaimKind::kGraph && c.owner == plan.owner && c.id == plan.id) {
        return c.bytes;
      }
    }
    return 0;
  };
  std::vector<std::pair<std::uint32_t, std::uint64_t>> kept;
  std::uint64_t held = 0;
  for (const std::size_t i : plans) {
    if (held >= floor_bytes) {
      break;
    }
    const ReclaimCandidate& plan = candidates[i];
    held += plan.bytes - std::min(graph_bytes(plan), plan.bytes);
    kept.emplace_back(plan.owner, plan.id);
  }
  std::erase_if(candidates, [&](const ReclaimCandidate& c) {
    return c.running && (c.kind == ReclaimKind::kPlan || c.kind == ReclaimKind::kGraph) &&
           std::ranges::find(kept, std::pair(c.owner, c.id)) != kept.end();
  });
}

}  // namespace jitllm::memory
