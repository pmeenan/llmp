// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "engine/plan_account.h"

#include <atomic>
#include <cstdint>
#include <span>
#include <vector>

#include "memory/reclaim.h"

namespace llmp::engine {

std::uint64_t NextPlanUse() {
  static std::atomic<std::uint64_t> next{0};
  return next.fetch_add(1, std::memory_order_relaxed) + 1;
}

namespace {
// The steps under way on this thread (PlanStep): how deep, and the first
// one's start.
thread_local std::uint32_t step_depth = 0;
thread_local std::uint64_t step_start = kNoStep;
}  // namespace

PlanStep::PlanStep() {
  if (step_depth++ == 0) {
    step_start = NextPlanUse();
  }
}

PlanStep::~PlanStep() {
  if (--step_depth == 0) {
    step_start = kNoStep;
  }
}

std::uint64_t PlanStepStart() { return step_start; }

void CollectPlans(std::span<PlanCacheBase* const> caches, std::uint32_t owner, bool running,
                  std::vector<memory::ReclaimCandidate>& out) {
  for (PlanCacheBase* cache : caches) {
    if (cache != nullptr) {
      cache->Collect(owner, running, out);
    }
  }
}

std::uint64_t ReclaimPlan(std::span<PlanCacheBase* const> caches, memory::ReclaimKind kind,
                          std::uint64_t serial) {
  for (PlanCacheBase* cache : caches) {
    if (cache != nullptr) {
      if (const std::uint64_t freed = cache->Reclaim(kind, serial); freed != 0) {
        return freed;
      }
    }
  }
  return 0;
}

}  // namespace llmp::engine
