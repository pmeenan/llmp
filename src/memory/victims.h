// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The initial victim-selection baseline
// (docs/architecture.md#victim-selection-initial-baseline, D-055's victim
// order): deliberately simple, deterministic, and the yardstick later
// policies must beat on recorded traces. It only chooses; the memory
// manager begins, completes or abandons each eviction through the catalog.
//
// Victims are chosen only when an admitted phase needs capacity. Eligible
// extents are resident, unleased and unregistered, not quarantined, and
// either clean (restorable from their artifact) or discardable. Those whose
// contents are already discarded (released scratch, invalidated state) go
// first, then clean weights; within each, least recent actual use first,
// ties broken by content (artifact, group, chunk) and then identity, so
// replays and fake-backend runs are deterministic. Each victim is credited
// with the whole extent it frees; a pending phase's own dependencies are
// never chosen. Retained entries and their M_state cap (D-055) join the
// order with the retention cache.

#ifndef LLMP_MEMORY_VICTIMS_H_
#define LLMP_MEMORY_VICTIMS_H_

#include <cstddef>
#include <span>
#include <vector>

#include "base/bytes.h"
#include "catalog/catalog.h"

namespace llmp::memory {

using base::Bytes;

struct Victim {
  catalog::ExtentId extent;
  catalog::MemoryClass memory_class = catalog::MemoryClass::kUnknown;
  Bytes bytes;
};

// The choice, and what the eviction event reports: the victims with their
// classes and expected bytes, and how many eligible alternatives were
// passed over.
struct VictimPlan {
  std::vector<Victim> victims;
  Bytes credited;
  std::size_t passed_over = 0;
  // Whether the victims free at least what was needed; if not, nothing
  // should be evicted for it.
  bool sufficient = false;
};

// Chooses victims in domain to free `needed` bytes, never choosing an
// extent in `protect`.
VictimPlan SelectVictims(const catalog::Catalog& catalog, catalog::DomainId domain, Bytes needed,
                         std::span<const catalog::ExtentId> protect = {});

}  // namespace llmp::memory

#endif  // LLMP_MEMORY_VICTIMS_H_
