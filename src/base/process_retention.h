// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Ownership that ends with the process. Work whose end could not be proven
// may still touch what it was given (docs/engine.md, "Unproven work"), so
// its owners are never freed: a storage ring with reads in flight, a
// supplemental run's state. Kept here instead of dropped, such an owner stays
// reachable, which says what it is (and LeakSanitizer does not report it).

#ifndef LLMP_BASE_PROCESS_RETENTION_H_
#define LLMP_BASE_PROCESS_RETENTION_H_

#include <cstddef>

namespace llmp::base {

// Keeps `owner` reachable until the process ends; it is never freed.
// Thread-safe; a null `owner` is ignored.
void RetainForProcess(const void* owner);

// How many owners RetainForProcess has kept.
std::size_t ProcessRetained();

}  // namespace llmp::base

#endif  // LLMP_BASE_PROCESS_RETENTION_H_
