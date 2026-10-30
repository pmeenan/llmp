// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "base/process_retention.h"

#include <cstddef>
#include <mutex>
#include <vector>

namespace llmp::base {
namespace {

struct Retained {
  std::mutex mutex;
  std::vector<const void*> owners;
};

// Never destroyed: an owner stays reachable through exit.
Retained& Kept() {
  static auto* const kept = new Retained();  // NOLINT(cppcoreguidelines-owning-memory)
  return *kept;
}

}  // namespace

void RetainForProcess(const void* owner) {
  if (owner == nullptr) return;
  auto& kept = Kept();
  const std::lock_guard lock(kept.mutex);
  kept.owners.push_back(owner);
}

std::size_t ProcessRetained() {
  auto& kept = Kept();
  const std::lock_guard lock(kept.mutex);
  return kept.owners.size();
}

}  // namespace llmp::base
