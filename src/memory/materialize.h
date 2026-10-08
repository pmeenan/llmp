// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Planning a closure's materialization in one domain
// (docs/reservation-policy.md#admission-rule-and-separate-ledgers,
// docs/architecture.md#page-in-and-eviction-lifecycles-8): on a real miss,
// select eligible victims and materialize only the missing dependencies,
// with actual occupancy checked against the execution budget B. Capacity an
// intended eviction will free counts as occupied until that eviction
// completes, so a plan that needs victims loads nothing until they are
// gone; the catalog's BeginLoad refuses anything that would still exceed B.
//
// It only plans, against a snapshot of the catalog: the memory manager
// begins each eviction and load through the catalog and replans when
// completions arrive. It chooses nothing that the closure needs, and
// nothing the caller protects.

#ifndef LLMP_MEMORY_MATERIALIZE_H_
#define LLMP_MEMORY_MATERIALIZE_H_

#include <span>
#include <vector>

#include "base/bytes.h"
#include "catalog/catalog.h"
#include "memory/victims.h"

namespace llmp::memory {

struct MaterializationPlan {
  // The closure's extents in this domain, by what they need.
  std::vector<catalog::ExtentId> load;         // nonresident: load them
  std::vector<catalog::ExtentId> loading;      // a load in progress: join it
  std::vector<catalog::ExtentId> cancel;       // evicting: cancel the eviction
  std::vector<catalog::ExtentId> quarantined;  // unusable until resolved
  // Extents whose contents differ from what the closure recorded, or were
  // discarded: loading cannot restore them.
  std::vector<catalog::ExtentId> stale;
  Bytes missing;    // the bytes `load` materializes
  Bytes shortfall;  // what occupancy plus `missing` exceeds B by
  // Victims for the shortfall; empty when there is none.
  VictimPlan victims;
  // The closure can be materialized in this domain: nothing quarantined or
  // stale, and the shortfall, if any, covered by victims.
  bool feasible = false;
  // Every extent is resident with the recorded contents: lease it now.
  bool ready = false;
};

// Plans the closure's extents in `domain` against budget B, never choosing
// the closure's own extents or `protect` as victims.
MaterializationPlan PlanMaterialization(const catalog::Catalog& catalog, catalog::DomainId domain,
                                        Bytes budget, const catalog::Closure& closure,
                                        std::span<const catalog::ExtentId> protect = {},
                                        bool select_victims = true);

}  // namespace llmp::memory

#endif  // LLMP_MEMORY_MATERIALIZE_H_
