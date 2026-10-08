// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "memory/victims.h"

#include <algorithm>
#include <optional>
#include <span>
#include <tuple>
#include <utility>
#include <vector>

#include "base/bytes.h"
#include "catalog/catalog.h"

namespace llmp::memory {

VictimPlan SelectVictims(const catalog::Catalog& catalog, catalog::DomainId domain, Bytes needed,
                         std::span<const catalog::ExtentId> protect) {
  std::vector<catalog::ExtentId> protected_extents(protect.begin(), protect.end());
  std::ranges::sort(protected_extents);
  std::vector<std::pair<catalog::ExtentId, catalog::ExtentView>> candidates;
  for (auto& [id, view] : catalog.ExtentsOf(domain)) {
    if (catalog::Catalog::Evictable(view) && !std::ranges::binary_search(protected_extents, id)) {
      candidates.emplace_back(id, view);
    }
  }
  // Discarded contents first, then least recent actual use, then content,
  // then identity.
  const auto key = [](const std::pair<catalog::ExtentId, catalog::ExtentView>& candidate) {
    const catalog::ExtentView& view = candidate.second;
    const int order =
        view.discarded || view.descriptor.recovery == catalog::Recovery::kDiscardable ? 0 : 1;
    return std::tuple(order, view.last_use, view.descriptor.content, candidate.first);
  };
  std::ranges::sort(candidates, [&key](const auto& a, const auto& b) { return key(a) < key(b); });

  VictimPlan plan;
  for (const auto& [id, view] : candidates) {
    if (plan.credited >= needed) {
      ++plan.passed_over;
      continue;
    }
    plan.victims.push_back(Victim{
        .extent = id, .memory_class = view.descriptor.memory_class, .bytes = view.descriptor.size});
    // Every candidate's backing exists, so the domain's occupancy bounds the sum.
    plan.credited = plan.credited.Plus(view.descriptor.size).value_or(Bytes(UINT64_MAX));
  }
  plan.sufficient = plan.credited >= needed;
  return plan;
}

}  // namespace llmp::memory
