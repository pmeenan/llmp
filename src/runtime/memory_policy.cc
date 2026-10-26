// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "runtime/memory_policy.h"

#include <algorithm>
#include <cstdint>
#include <optional>

#include "config/node_config.h"

namespace llmp::runtime {

MemoryPolicy ChooseMemoryPolicy(const config::MemoryConfig& memory, bool unified) {
  MemoryPolicy policy;
  policy.configured = memory.mode.set;
  policy.mode = memory.mode.set ? memory.mode.value
                : unified       ? config::MemoryMode::kDedicated
                                : config::MemoryMode::kShared;
  const bool shared = policy.mode == config::MemoryMode::kShared;
  policy.pool = !shared;
  policy.pageable_workspace = shared;
  policy.idle_release = shared && memory.idle_release_seconds.value > 0;
  policy.shrink_budget = shared;
  policy.release_under_pressure = shared;
  return policy;
}

std::optional<std::uint64_t> BudgetHeadroom(const HeadroomInputs& inputs) {
  const std::uint64_t extent = std::max<std::uint64_t>(inputs.extent, 1);
  const auto whole = [extent](std::uint64_t bytes) { return bytes / extent * extent; };
  std::optional<std::uint64_t> headroom;
  if (inputs.host_available != 0) {
    headroom = whole(inputs.host_available > inputs.host_reserve
                         ? inputs.host_available - inputs.host_reserve
                         : 0);
  }
  if (inputs.device_free) {
    const std::uint64_t device = whole(*inputs.device_free > inputs.device_reserve
                                           ? *inputs.device_free - inputs.device_reserve
                                           : 0);
    headroom = headroom ? std::min(*headroom, device) : device;
  }
  return headroom;
}

}  // namespace llmp::runtime
