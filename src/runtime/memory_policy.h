// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The node's physical memory policy (D-124 as amended 2026-10-25;
// docs/runtime-serving.md, "Physical memory modes"): what `[memory] mode`
// means for the server, decided once at its start, and the budget's
// physical headroom. Vendor-free and host-only, so the CPU tests drive it
// with readings of their own.
//
// - Dedicated (the default on a unified-memory device, the GB10's): the
//   runtime owns the node's memory. A node pool of empty device handles
//   (engine::NodePool) is filled at the start, within what the budget leaves
//   beside the occupancy, and kept there; the workspace stays resident when
//   it fits.
// - Shared (the default on a discrete GPU until M6's device budget, and for
//   workstations): no pool; the workspace pages with the active model; the
//   working set is released after `idle_release_seconds` idle, and pressure
//   from outside lowers the budget and, when a trim falls short, releases
//   the working set.
// Pressure from outside and the release route give empty pool handles back
// first in either mode, then cache through the reclaim order.

#ifndef LLMP_RUNTIME_MEMORY_POLICY_H_
#define LLMP_RUNTIME_MEMORY_POLICY_H_

#include <cstdint>
#include <optional>
#include <string_view>

#include "config/node_config.h"

namespace llmp::runtime {

struct MemoryPolicy {
  config::MemoryMode mode = config::MemoryMode::kDedicated;
  bool configured = false;              // the mode was set; otherwise chosen by the device
  bool pool = false;                    // the node pool, filled at the start
  bool pageable_workspace = false;      // the workspace's default: paged with the active model
  bool idle_release = false;            // the working set released once idle
  bool shrink_budget = false;           // the budget lowered under pressure from outside
  bool release_under_pressure = false;  // the working set released when a trim falls short

  std::string_view name() const { return config::ToString(mode); }
};

// `[memory] mode`, or unset, the device's: dedicated where the device's
// memory is the host's (`unified`), shared on a discrete GPU.
MemoryPolicy ChooseMemoryPolicy(const config::MemoryConfig& memory, bool unified);

// What the budget may take beside the fixed memory (D-050's B less F), in
// whole extents of `extent` bytes: the host's available memory less what the
// start's guard sets apart outside the budget (`host_reserve`), and on a
// discrete device (D-082) no more than its free memory less `device_reserve`
// either, since the paged extents live there. None when neither reading is
// known (`host_available` 0 and no `device_free`).
struct HeadroomInputs {
  std::uint64_t host_available = 0;  // MemAvailable; 0: unknown
  std::uint64_t host_reserve = 0;
  std::optional<std::uint64_t> device_free = std::nullopt;  // a discrete device's free memory
  std::uint64_t device_reserve = 0;
  std::uint64_t extent = std::uint64_t{2} << 20U;
};
std::optional<std::uint64_t> BudgetHeadroom(const HeadroomInputs& inputs);

// What a discrete device keeps outside the catalog beside a step's plans
// (the driver's context, library workspaces): a bound, not a measurement,
// until M6 measures the device budget (D-082).
inline constexpr std::uint64_t kDeviceUncountedMargin = std::uint64_t{1} << 30U;

}  // namespace llmp::runtime

#endif  // LLMP_RUNTIME_MEMORY_POLICY_H_
