// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "kernels/common/grouped_worklist.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>

namespace llmp::kernels {
namespace {
struct Range {
  std::uintptr_t begin = 0;
  std::uintptr_t end = 0;
};
template <class T>
std::expected<Range, GroupedRouteError> RangeOf(std::span<T> span) {
  const auto begin = reinterpret_cast<std::uintptr_t>(span.data());
  if (begin == 0 || begin % alignof(T) != 0 ||
      span.size_bytes() > std::numeric_limits<std::uintptr_t>::max() - begin)
    return std::unexpected(GroupedRouteError::kExtent);
  return Range{begin, begin + span.size_bytes()};
}
bool Overlap(Range a, Range b) { return a.begin < b.end && b.begin < a.end; }
}  // namespace

std::expected<GroupedRouteSize, GroupedRouteError> GroupedRouteBound(GroupedRouteShape s) {
  constexpr auto max = static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max());
  if (s.experts == 0 || s.experts > 4096 || s.used == 0 || s.used > 64 || s.used > s.experts ||
      s.tokens == 0 || (s.input_lanes != 1 && s.input_lanes != s.used) ||
      s.tokens > (max / 4) / s.used)
    return std::unexpected(GroupedRouteError::kShape);
  const auto pairs = s.used * s.tokens;
  // Same representable route domain as MoeRoute, before any storage creation.
  const auto route_ints = 2 * (s.experts + 1) + 3 * pairs + ((s.tokens + 63) / 64) * s.experts;
  if (route_ints > max) return std::unexpected(GroupedRouteError::kShape);
  return GroupedRouteSize{
      .pairs = pairs,
      .inputs = base::Bytes((s.experts + 1 + 4 * pairs) * sizeof(std::int32_t)),
      .outputs = base::Bytes((s.experts + 1 + 2 * pairs) * sizeof(std::int32_t)),
      .scratch = base::Bytes(pairs)};
}

std::expected<void, GroupedRouteError> ValidateGroupedRouteInto(GroupedRouteShape s,
                                                                CompletedGroupedRoute r,
                                                                GroupedWorklistStorage out) {
  const auto size = GroupedRouteBound(s);
  if (!size) return std::unexpected(size.error());
  const auto p = size->pairs;
  if (r.ids.size() != p || r.position.size() != p || r.token.size() != p || r.expert.size() != p ||
      r.offsets.size() != s.experts + 1 || out.offsets.size() != s.experts + 1 ||
      out.rows.size() != p || out.original_slots.size() != p || out.seen.size() != p)
    return std::unexpected(GroupedRouteError::kExtent);
  const std::array inputs{RangeOf(r.ids), RangeOf(r.offsets), RangeOf(r.position), RangeOf(r.token),
                          RangeOf(r.expert)};
  const std::array outputs{RangeOf(out.offsets), RangeOf(out.rows), RangeOf(out.original_slots),
                           RangeOf(out.seen)};
  for (const auto& range : inputs)
    if (!range) return std::unexpected(range.error());
  for (std::size_t i = 0; i != outputs.size(); ++i) {
    if (!outputs[i]) return std::unexpected(outputs[i].error());
    for (const auto& input : inputs)
      if (Overlap(*outputs[i], *input)) return std::unexpected(GroupedRouteError::kAlias);
    for (std::size_t j = 0; j != i; ++j)
      if (Overlap(*outputs[i], *outputs[j])) return std::unexpected(GroupedRouteError::kAlias);
  }
  if (r.offsets.front() != 0 || r.offsets.back() != static_cast<std::int32_t>(p))
    return std::unexpected(GroupedRouteError::kRoute);
  for (std::size_t e = 0; e != s.experts; ++e)
    if (r.offsets[e] < 0 || r.offsets[e] > r.offsets[e + 1])
      return std::unexpected(GroupedRouteError::kRoute);
  std::ranges::fill(out.seen, 0);
  for (std::size_t slot = 0; slot != p; ++slot) {
    const auto id = r.ids[slot];
    const auto pos = r.position[slot];
    if (id < 0 || static_cast<std::uint64_t>(id) >= s.experts || pos < 0 ||
        static_cast<std::uint64_t>(pos) >= p)
      return std::unexpected(GroupedRouteError::kRoute);
    const auto e = static_cast<std::size_t>(id);
    const auto row = static_cast<std::size_t>(pos);
    if (out.seen[row] != 0 || pos < r.offsets[e] || pos >= r.offsets[e + 1] ||
        r.token[row] != static_cast<std::int32_t>(slot / s.used) || r.expert[row] != id)
      return std::unexpected(GroupedRouteError::kRoute);
    out.seen[row] = 1;
  }
  std::ranges::copy(r.offsets, out.offsets.begin());
  for (std::size_t slot = 0; slot != p; ++slot) {
    const auto row = static_cast<std::size_t>(r.position[slot]);
    out.rows[row] = static_cast<std::int32_t>(s.input_lanes == 1 ? slot / s.used : slot);
    out.original_slots[row] = static_cast<std::int32_t>(slot);
  }
  return {};
}
}  // namespace llmp::kernels
