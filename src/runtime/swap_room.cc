// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "runtime/swap_room.h"

#include <algorithm>
#include <cstdint>
#include <expected>
#include <format>
#include <functional>
#include <limits>
#include <map>
#include <string>

namespace llmp::runtime {

std::expected<std::vector<catalog::ExtentId>, std::string> ReleaseForHandoff(
    catalog::DomainId domain, std::uint64_t occupancy, std::uint64_t budget,
    std::span<const SwapBackingExtent> missing, std::span<const SwapBackingExtent> selected) {
  std::map<SwapBackingKey, std::uint64_t> demand;
  std::map<SwapBackingKey, std::uint64_t> supply;
  // Authenticate all typed identities with contiguous scratch rather than
  // allocating one tree node for every missing/selected extent. Release
  // order still follows `selected`; sorting this scratch changes no policy.
  std::vector<catalog::ExtentId> identities;
  if (missing.size() > identities.max_size() ||
      selected.size() > identities.max_size() - missing.size())
    return std::unexpected("too many backing identities in a swap");
  identities.reserve(missing.size() + selected.size());
  std::uint64_t incoming = 0;
  std::uint64_t outgoing = 0;
  const auto add = [](std::uint64_t& value, std::uint64_t bytes) {
    if (bytes > std::numeric_limits<std::uint64_t>::max() - value) return false;
    value += bytes;
    return true;
  };
  const auto valid = [&](const SwapBackingExtent& extent) {
    if (!extent.extent.valid() || extent.bytes == 0 ||
        (extent.backing &&
         (extent.backing->domain != domain || extent.backing->size != extent.bytes)))
      return false;
    identities.push_back(extent.extent);
    return true;
  };
  for (const auto& extent : missing) {
    if (!valid(extent) || !add(incoming, extent.bytes) ||
        (extent.backing && !add(demand[*extent.backing], extent.bytes))) {
      return std::unexpected("invalid missing backing in a swap");
    }
  }
  for (const auto& extent : selected) {
    if (!valid(extent) || !add(outgoing, extent.bytes) ||
        (extent.backing && !add(supply[*extent.backing], extent.bytes))) {
      return std::unexpected("invalid selected backing in a swap");
    }
  }
  std::ranges::sort(identities);
  if (std::ranges::adjacent_find(identities) != identities.end())
    return std::unexpected("duplicate backing identities in a swap");
  std::uint64_t after = occupancy;
  if (!add(after, incoming)) return std::unexpected("a swap's occupancy overflows");
  if (after > budget && after - budget > outgoing) {
    return std::unexpected("the selected extents cannot make room for a swap");
  }
  std::uint64_t compatible = 0;
  for (auto& [key, bytes] : supply) {
    const auto matched = std::min(bytes, demand[key]);
    compatible += matched;  // bounded by incoming
    bytes -= matched;       // only this surplus may release without losing a donor
  }
  const auto uncredited = after - compatible;
  const std::uint64_t needed = uncredited > budget ? uncredited - budget : 0;
  std::uint64_t released = 0;
  std::vector<catalog::ExtentId> release;
  // No managed backing cannot park. Include it in the actual release credit.
  for (const auto& extent : selected) {
    if (!extent.backing) {
      release.push_back(extent.extent);
      released += extent.bytes;
    }
  }
  for (const auto& extent : selected) {
    if (released >= needed) break;
    if (extent.backing && supply[*extent.backing] >= extent.bytes) {
      supply[*extent.backing] -= extent.bytes;
      release.push_back(extent.extent);
      released += extent.bytes;
    }
  }
  if (released < needed) return std::unexpected("a swap lacks releasable incompatible backing");
  return release;
}

std::expected<SwapRoomSteps, std::string> MakeRoom(
    const std::function<std::uint64_t()>& shortfall,
    const std::function<std::uint64_t(std::uint64_t)>& reclaim, std::uint64_t unit, int attempts) {
  SwapRoomSteps steps;
  for (;;) {
    const std::uint64_t short_by = shortfall();
    if (short_by == 0) {
      return steps;
    }
    if (steps.reclaims >= attempts) {
      return std::unexpected(
          std::format("{} bytes short of room for the incoming model after {} reclaims", short_by,
                      steps.reclaims));
    }
    const std::uint64_t ask = unit == 0 ? short_by : (short_by + unit - 1) / unit * unit;
    const std::uint64_t freed = reclaim(ask);
    ++steps.reclaims;
    steps.asked += ask;
    steps.freed += freed;
    if (freed == 0) {
      return std::unexpected(std::format(
          "{} bytes short of room for the incoming model, and nothing left to reclaim", short_by));
    }
  }
}

}  // namespace llmp::runtime
