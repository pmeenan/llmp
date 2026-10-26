// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "planner/lanes.h"

#include <bits/stl_tree.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <string_view>
#include <utility>
#include <vector>

#include "execution/registry.h"
#include "ir/graph.h"
#include "planner/select.h"

namespace llmp::planner {
namespace {

constexpr std::uint64_t kVectorEnvelope = ir::kAdmissionHeapBlockOverhead;

bool AddExact(std::uint64_t count, std::uint64_t stride, std::uint64_t& total) {
  return ir::AddVectorAdmissionBytes(count, stride, total);
}

bool AddGrowing(std::uint64_t elements, std::uint64_t allocations, std::uint64_t stride,
                std::uint64_t& total) {
  if (elements == 0) return true;
  const auto max = std::numeric_limits<std::uint64_t>::max();
  if (stride == 0 || stride > max / 3 || elements > max / (3 * stride) ||
      allocations > max / (2 * kVectorEnvelope))
    return false;
  const std::uint64_t payload = 3 * elements * stride;
  const std::uint64_t overhead = 2 * allocations * kVectorEnvelope;
  if (payload > max - overhead || total > max - payload - overhead) return false;
  total += payload + overhead;
  return true;
}

bool AddOneShotVectors(std::uint64_t elements, std::uint64_t allocations, std::uint64_t stride,
                       std::uint64_t& total) {
  if (elements == 0) return true;
  const auto max = std::numeric_limits<std::uint64_t>::max();
  if (stride == 0 || elements > max / stride || allocations > max / kVectorEnvelope) return false;
  const std::uint64_t payload = elements * stride;
  const std::uint64_t overhead = allocations * kVectorEnvelope;
  if (payload > max - overhead || total > max - payload - overhead) return false;
  total += payload + overhead;
  return true;
}

bool AddTreeNodes(std::uint64_t nodes, std::uint64_t node_bytes, std::uint64_t& total) {
  const auto max = std::numeric_limits<std::uint64_t>::max();
  if (nodes == 0) return true;
  if (node_bytes > max - kVectorEnvelope || nodes > (max - total) / (node_bytes + kVectorEnvelope))
    return false;
  total += nodes * (node_bytes + kVectorEnvelope);
  return true;
}

}  // namespace

std::expected<std::uint64_t, std::string> LaneMetadataBytes(std::span<const Selected> steps,
                                                            const LaneTags& source_tags,
                                                            std::size_t current_node_count) {
  if (source_tags.empty()) return std::uint64_t{0};
  using TagValue = std::pair<const std::uint32_t, LaneTag>;
  using RegionValue = std::pair<const std::uint32_t, Region>;
  using TagTreeNode = std::_Rb_tree_node<TagValue>;
  using RegionTreeNode = std::_Rb_tree_node<RegionValue>;
  // Derive map storage from the pinned libstdc++ node type (tree links/color
  // plus the concrete value), then add the shared allocator envelope.
  static_assert(sizeof(TagTreeNode) >= sizeof(std::_Rb_tree_node_base) + sizeof(TagValue));
  static_assert(sizeof(RegionTreeNode) >= sizeof(std::_Rb_tree_node_base) + sizeof(RegionValue));

  const auto max = std::numeric_limits<std::uint64_t>::max();
  const std::uint64_t nodes = current_node_count;
  std::uint64_t bytes = 0;
  std::uint64_t max_lane_source = 0;
  for (const auto& [node, _] : source_tags)
    if (node.valid()) max_lane_source = std::max<std::uint64_t>(max_lane_source, node.v);
  if (max_lane_source == max) return std::unexpected("lane tag index overflows");
  const std::uint64_t lookup_count = max_lane_source + 1;
  constexpr std::uint64_t word_bits = std::numeric_limits<unsigned long>::digits;
  if (lookup_count > max - (word_bits - 1) ||
      !AddGrowing(nodes, nodes, sizeof(LaneTags::value_type), bytes) ||
      !AddExact(lookup_count, sizeof(LaneTag), bytes) ||
      !AddExact((lookup_count + word_bits - 1) / word_bits, sizeof(unsigned long), bytes) ||
      !AddTreeNodes(nodes, sizeof(TagTreeNode), bytes))
    return std::unexpected("lane lookup inventory overflows");

  std::uint64_t batch_members = 0;
  std::uint64_t batch_text = 0;
  for (const Selected& step : steps) {
    const std::uint64_t members = step.nodes.size();
    if (members > max - batch_members) return std::unexpected("lane batch inventory overflows");
    batch_members += members;
    if (!step.batch.empty()) {
      const std::uint64_t per_copy =
          static_cast<std::uint64_t>(step.batch.size()) + 1 + kVectorEnvelope;
      if (per_copy > max / std::max<std::uint64_t>(members, 1) ||
          batch_text > max - members * per_copy)
        return std::unexpected("lane batch string inventory overflows");
      batch_text += members * per_copy;
    }
  }
  if (!AddGrowing(nodes, nodes, sizeof(Selected), bytes) ||
      !AddOneShotVectors(batch_members, nodes, sizeof(ir::NodeId), bytes) ||
      !AddOneShotVectors(batch_members, nodes, sizeof(std::size_t), bytes) ||
      !AddOneShotVectors(batch_members, nodes, sizeof(std::string_view), bytes) ||
      batch_text > max - bytes || !AddTreeNodes(nodes, sizeof(RegionTreeNode), bytes))
    return std::unexpected("lane split inventory overflows");
  bytes += batch_text;
  if (!AddExact(nodes, sizeof(Region), bytes) || !AddGrowing(nodes, nodes, sizeof(Region), bytes))
    return std::unexpected("lane region inventory overflows");
  return bytes;
}

std::vector<Region> AssignLanes(std::vector<Selected>& steps, const LaneTags& tags,
                                const execution::Registry& registry,
                                const std::function<bool(std::string_view)>& on_stream) {
  std::vector<Region> regions;
  for (Selected& step : steps) {
    step.lane = 0;
  }
  if (tags.empty()) {
    return regions;
  }
  std::map<std::uint32_t, LaneTag> tag;
  for (const auto& [node, t] : tags) {
    tag.emplace(node.v, t);
  }
  const auto tag_of = [&](ir::NodeId node) {
    const auto found = tag.find(node.v);
    return found == tag.end() ? LaneTag{} : found->second;
  };
  // A batch never joins nodes of separate lanes or regions: split it into
  // runs of one tag, each a batch again or its member's own step.
  std::vector<Selected> split;
  split.reserve(steps.size());
  for (Selected& step : steps) {
    const bool mixed = !step.batch.empty() && std::ranges::any_of(step.nodes, [&](ir::NodeId n) {
      return tag_of(n) != tag_of(step.nodes.front());
    });
    if (!mixed) {
      split.push_back(std::move(step));
      continue;
    }
    for (std::size_t first = 0; first < step.nodes.size();) {
      std::size_t end = first + 1;
      while (end < step.nodes.size() && tag_of(step.nodes[end]) == tag_of(step.nodes[first])) {
        ++end;
      }
      Selected run;
      run.nodes.assign(step.nodes.begin() + static_cast<std::ptrdiff_t>(first),
                       step.nodes.begin() + static_cast<std::ptrdiff_t>(end));
      if (end - first > 1) {
        run.implementation = step.implementation;
        run.group = step.group;
        run.batch = step.batch;
        run.singles.assign(step.singles.begin() + static_cast<std::ptrdiff_t>(first),
                           step.singles.begin() + static_cast<std::ptrdiff_t>(end));
        run.single_groups.assign(step.single_groups.begin() + static_cast<std::ptrdiff_t>(first),
                                 step.single_groups.begin() + static_cast<std::ptrdiff_t>(end));
      } else {
        run.implementation = step.singles.at(first);
        run.group = step.single_groups.at(first);
      }
      split.push_back(std::move(run));
      first = end;
    }
  }
  steps = std::move(split);
  const auto stays = [&](std::size_t implementation) {
    return on_stream && on_stream(registry.at(implementation).name);
  };
  std::map<std::uint32_t, Region> spans;
  for (std::size_t s = 0; s < steps.size(); ++s) {
    Selected& step = steps[s];
    for (const ir::NodeId node : step.nodes) {
      const LaneTag t = tag_of(node);
      if (t.region == 0) {
        continue;
      }
      const auto at = static_cast<std::uint32_t>(s);
      const auto [it, added] = spans.try_emplace(t.region, Region{at, at});
      if (!added) {
        it->second.first = std::min(it->second.first, at);
        it->second.last = std::max(it->second.last, at);
      }
      if (t.lane <= kMaxLanes && !stays(step.implementation)) {
        step.lane = t.lane;
      }
      break;
    }
  }
  std::vector<Region> ordered;
  ordered.reserve(spans.size());
  for (const auto& [region, span] : spans) {
    ordered.push_back(span);
  }
  std::ranges::sort(ordered, {}, &Region::first);
  for (const Region& span : ordered) {
    if (!regions.empty() && span.first <= regions.back().last) {
      regions.back().last = std::max(regions.back().last, span.last);
    } else {
      regions.push_back(span);
    }
  }
  // A lane step outside every region would have no fork or join.
  std::size_t r = 0;
  for (std::size_t s = 0; s < steps.size(); ++s) {
    while (r < regions.size() && regions[r].last < s) {
      ++r;
    }
    if (r == regions.size() || s < regions[r].first) {
      steps[s].lane = 0;
    }
  }
  return regions;
}

}  // namespace llmp::planner
