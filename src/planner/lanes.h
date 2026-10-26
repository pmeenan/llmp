// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Concurrent lanes for a plan's steps (a port of kernels/ggml/graph_plan.h
// AssignLanes onto IR steps). A caller may tag nodes that run independently
// of one another (a wave's owners' attention and state work) with a lane
// and a region. AssignLanes gives each step its first tagged node's lane
// and records each region's span of steps: over a span, the lowering
// queues each lane's steps on a stream of its own, waiting where a step
// reads or writes what another lane last touched, and joins every lane at
// the span's end. Placement (placement.h) keeps everything a span's steps
// compute or read live for the whole span. A batch whose members carry
// different tags is split into runs first (a run of one takes the member's
// own implementation), and a step whose implementation `on_stream` names
// stays on the stream (one that borrows the context stream's library
// handle, such as cuBLAS). There is no default, as the legacy AssignLanes
// had (its module's cuBLAS and BF16 GEMM): a lowering whose module has
// such implementations passes `on_stream`.

#ifndef LLMP_PLANNER_LANES_H_
#define LLMP_PLANNER_LANES_H_

#include <cstdint>
#include <functional>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

#include "execution/registry.h"
#include "ir/graph.h"
#include "planner/select.h"

namespace llmp::planner {

inline constexpr std::uint32_t kMaxLanes = 4;

struct LaneTag {
  std::uint8_t lane = 0;     // 1 to kMaxLanes; 0 keeps the step on the stream
  std::uint32_t region = 0;  // 1 on; 0 is no region
  bool operator==(const LaneTag&) const = default;
};
using LaneTags = std::vector<std::pair<ir::NodeId, LaneTag>>;

// A region's first and last steps.
struct Region {
  std::uint32_t first = 0;
  std::uint32_t last = 0;
  bool operator==(const Region&) const = default;
};

// Gives `steps` their lanes and returns the regions' spans, in order and
// disjoint (overlapping spans joined).
std::vector<Region> AssignLanes(std::vector<Selected>& steps, const LaneTags& tags,
                                const execution::Registry& registry,
                                const std::function<bool(std::string_view)>& on_stream = {});

}  // namespace llmp::planner

#endif  // LLMP_PLANNER_LANES_H_
