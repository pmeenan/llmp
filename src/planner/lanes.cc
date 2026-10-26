// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "planner/lanes.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <string_view>
#include <utility>
#include <vector>

#include "execution/registry.h"
#include "ir/graph.h"
#include "planner/select.h"

namespace llmp::planner {

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
