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

ReclaimPlan SelectReclaim(std::span<const ReclaimCandidate> candidates, std::uint64_t needed,
                          double below) {
  ReclaimPlan plan;
  const std::array<double, kReclaimKinds> cost = KindCosts(candidates);
  // A plan's bytes include its graphs': a graph taken before its plan is
  // not counted again, and a graph whose plan was taken is skipped.
  std::vector<std::pair<std::uint32_t, std::uint64_t>> plans;
  std::vector<std::pair<std::uint32_t, std::uint64_t>> graphs;
  const auto graph_bytes = [&](const ReclaimCandidate& plan_candidate) -> std::uint64_t {
    for (const ReclaimCandidate& c : candidates) {
      if (c.kind == ReclaimKind::kGraph && c.owner == plan_candidate.owner &&
          c.id == plan_candidate.id) {
        return c.bytes;
      }
    }
    return 0;
  };
  const std::vector<std::size_t> order = ReclaimOrder(candidates);
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
    const auto handle = std::pair(c.owner, c.id);
    if (c.kind == ReclaimKind::kGraph) {
      if (std::ranges::find(plans, handle) != plans.end()) {
        continue;  // gone with its plan
      }
      graphs.push_back(handle);
    } else if (c.kind == ReclaimKind::kPlan) {
      if (std::ranges::find(graphs, handle) != graphs.end()) {
        b -= std::min(b, graph_bytes(c));
      }
      plans.push_back(handle);
    }
    plan.victims.push_back(i);
    plan.priorities.push_back(priority);
    plan.bytes = b > UINT64_MAX - plan.bytes ? UINT64_MAX : plan.bytes + b;
  }
  plan.sufficient = plan.bytes >= needed;
  return plan;
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
