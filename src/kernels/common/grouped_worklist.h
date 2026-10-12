// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Checked inversion of a completed route into RunGrouped's worklist. No
// allocation, device access or residency authority: callers pay for all spans
// before creating them and hold their source/lease through execution.
#ifndef LLMP_KERNELS_COMMON_GROUPED_WORKLIST_H_
#define LLMP_KERNELS_COMMON_GROUPED_WORKLIST_H_

#include <cstdint>
#include <expected>
#include <span>

#include "base/bytes.h"

namespace llmp::kernels {

struct GroupedRouteShape {
  std::uint64_t experts = 0;
  std::uint64_t used = 0;
  std::uint64_t tokens = 0;
  // Activations per token: 1 for token input, used for routed input.
  std::uint64_t input_lanes = 0;
};
enum class GroupedRouteError : std::uint8_t { kShape, kExtent, kAlias, kRoute };
struct GroupedRouteSize {
  std::uint64_t pairs = 0;
  base::Bytes inputs{};   // all five completed host input spans
  base::Bytes outputs{};  // offsets and two pair arrays
  base::Bytes scratch{};  // one byte per pair, may change on failure
};
std::expected<GroupedRouteSize, GroupedRouteError> GroupedRouteBound(GroupedRouteShape shape);

struct CompletedGroupedRoute {
  std::span<const std::int32_t> ids{};       // original [used,tokens]
  std::span<const std::int32_t> offsets{};   // experts+1, starts at zero
  std::span<const std::int32_t> position{};  // original slot -> sorted row
  std::span<const std::int32_t> token{};     // sorted row -> token
  std::span<const std::int32_t> expert{};    // sorted row -> expert
};
struct GroupedWorklistStorage {
  std::span<std::int32_t> offsets{};
  std::span<std::int32_t> rows{};
  std::span<std::int32_t> original_slots{};
  std::span<std::uint8_t> seen{};
};
// Exact span sizes are required. All writable spans must be disjoint from
// each other and the inputs. On refusal only seen may change. The caller
// must keep inputs immutable through this synchronous validation/copy.
std::expected<void, GroupedRouteError> ValidateGroupedRouteInto(GroupedRouteShape shape,
                                                                CompletedGroupedRoute route,
                                                                GroupedWorklistStorage into);

}  // namespace llmp::kernels
#endif  // LLMP_KERNELS_COMMON_GROUPED_WORKLIST_H_
