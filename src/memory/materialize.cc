// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "memory/materialize.h"

#include <optional>
#include <span>
#include <vector>

#include "base/bytes.h"
#include "catalog/catalog.h"
#include "memory/victims.h"

namespace llmp::memory {

MaterializationPlan PlanMaterialization(const catalog::Catalog& catalog, catalog::DomainId domain,
                                        Bytes budget, const catalog::Closure& closure,
                                        std::span<const catalog::ExtentId> protect,
                                        bool select_victims) {
  MaterializationPlan plan;
  std::vector<catalog::ExtentId> keep(protect.begin(), protect.end());
  bool overflow = false;
  for (const auto& [extent, content_generation] : closure.extents) {
    keep.push_back(extent);
    const auto view = catalog.Describe(extent);
    if (!view) {
      plan.stale.push_back(extent);  // the extent itself is gone
      continue;
    }
    if (view->descriptor.domain != domain) {
      continue;
    }
    if (view->content_generation != content_generation || view->discarded) {
      plan.stale.push_back(extent);
      continue;
    }
    switch (view->state) {
      case catalog::ExtentState::kResident:
        break;
      case catalog::ExtentState::kNonresident: {
        plan.load.push_back(extent);
        const std::optional<Bytes> missing = plan.missing.Plus(view->descriptor.size);
        overflow = overflow || !missing;
        plan.missing = missing.value_or(plan.missing);
        break;
      }
      case catalog::ExtentState::kLoading:
        plan.loading.push_back(extent);
        break;
      case catalog::ExtentState::kEvicting:
        plan.cancel.push_back(extent);
        break;
      case catalog::ExtentState::kQuarantined:
        plan.quarantined.push_back(extent);
        break;
    }
  }
  const std::optional<Bytes> after = catalog.OccupancyOf(domain).Total().Plus(plan.missing);
  if (overflow || !after) {
    return plan;  // not feasible: more than any budget
  }
  if (*after > budget) {
    plan.shortfall = after->Minus(budget).value_or(Bytes());
    if (select_victims) plan.victims = SelectVictims(catalog, domain, plan.shortfall, keep);
  }
  plan.feasible = plan.quarantined.empty() && plan.stale.empty() &&
                  (plan.shortfall == Bytes() || plan.victims.sufficient);
  plan.ready = plan.feasible && plan.load.empty() && plan.loading.empty() && plan.cancel.empty();
  return plan;
}

}  // namespace llmp::memory
