// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "engine/planned.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <functional>
#include <limits>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "catalog/catalog.h"
#include "engine/paged_node.h"
#include "engine/support.h"
#include "kernels/ggml/graph_read_index.h"
#include "kernels/ggml/implementations.h"

namespace jitllm::engine {

namespace {

namespace kg = jitllm::kernels::ggml;
using catalog::MemoryClass;
using support::Address;
using support::Error;
using support::Round;

// SizedArena's scratch (one for the process: plans are built on the
// driver's thread and, at setup, on the starting one).
struct Scratch {
  std::mutex mutex;
  std::optional<kg::TensorArena> arena;
  std::size_t tensors = 0;
  std::atomic<std::uint64_t> bytes{0};
  std::atomic<std::uint64_t> index_bytes{0};
};
Scratch& TheScratch() {
  static Scratch scratch;
  return scratch;
}

}  // namespace

std::uint64_t ScratchArenaBytes() {
  return TheScratch().bytes.load() + TheScratch().index_bytes.load();
}

std::expected<void, std::string> PlaceAndPlan(
    PlannedBase& out, std::span<ggml_tensor* const> nodes, std::span<ggml_tensor* const> inputs,
    std::span<ggml_tensor* const> keep, const kg::DeviceChoices& choices, std::uint64_t activations,
    std::uint64_t activation_bytes, const kg::LaneTags* lanes) {
  // Measurement precedes the node startup budget. Admit one bounded index
  // envelope for all measured shapes; runtime plans cannot grow it. Both
  // planning passes are sequential and their indices are destroyed on return.
  const auto index_bytes = kg::NeedsGraphReadIndex(false, choices)
                               ? kg::detail::GraphReadIndex::ScratchBytes(nodes, keep)
                               : std::optional<std::uint64_t>{0};
  if (!index_bytes) return Error("graph reader scratch allowance overflow or invalid node");
  auto& scratch = TheScratch();
  if (activations == 0) {
    const std::scoped_lock lock(scratch.mutex);
    if (*index_bytes > std::numeric_limits<std::uint64_t>::max() - scratch.bytes.load())
      return Error("combined graph scratch allowance overflow");
    scratch.index_bytes.store(std::max(scratch.index_bytes.load(), *index_bytes));
  }
  const auto index_limit = scratch.index_bytes.load();
  if (*index_bytes > index_limit)
    return Error("graph reader index exceeds the startup scratch allowance");
  constexpr std::uint64_t kDistinct = std::uint64_t{1} << 46U;
  std::uint64_t leaf = kDistinct - (std::uint64_t{1} << 40U);
  for (ggml_tensor* input : inputs) {
    kg::TensorArena::Bind(input, leaf);
    leaf += Round(ggml_nbytes(input), 256) + 256;
  }
  kg::BindDistinct(nodes, kDistinct);
  // What the caller keeps is read after the graph: never overwritten in place.
  auto first = kg::PlanGraph(nodes, /*fusion=*/false, choices, keep, index_limit);
  if (!first) {
    return Error(first.error().detail);
  }
  if (lanes != nullptr) {
    kg::AssignLanes(*first, *lanes, kg::UsesCublas);
  }
  auto placement = kg::PlaceActivations(nodes, *first, inputs, 256, keep);
  if (!placement) {
    return Error(placement.error().detail);
  }
  out.placement = std::move(*placement);
  for (ggml_tensor* input : inputs) {
    out.inputs_bytes += Round(ggml_nbytes(input), 256);  // as GraphRuns::Stage places them
  }
  if (activations == 0) {
    out.plan = std::move(*first);
    return {};
  }
  if (out.placement.extent > activation_bytes) {
    return Error(std::format("the activations ({} bytes) exceed their region ({} bytes)",
                             out.placement.extent, activation_bytes));
  }
  for (const auto& [tensor, offset] : out.placement.offsets) {
    kg::TensorArena::Bind(tensor, activations + offset);
  }
  kg::BindViews(nodes);
  auto second = kg::PlanGraph(nodes, false, choices, keep, index_limit);
  if (!second) {
    return Error(second.error().detail);
  }
  if (lanes != nullptr) {
    kg::AssignLanes(*second, *lanes, kg::UsesCublas);
  }
  if (!kg::SamePlan(*first, *second)) {
    return Error("the plan changed once the activations were placed");
  }
  out.plan = std::move(*second);
  return {};
}

std::expected<void, std::string> BindPlanned(PlannedBase& planned, kg::LaunchContext& launch,
                                             const execution::Registry& registry,
                                             std::string_view what) {
  if (launch.lanes() < kg::kMaxLanes && !planned.plan.regions.empty()) {
    // A context without lanes runs everything on its stream. The placement
    // stays as made for lanes, which only lives longer.
    planned.plan.regions.clear();
    for (kg::PlanStep& step : planned.plan.steps) {
      step.lane = 0;
    }
  }
  auto scratch = kg::PlanScratch(launch, planned.plan);
  if (!scratch) {
    return Error(scratch.error().detail);
  }
  if (*scratch > launch.scratch_size(0).value()) {
    return Error(std::format("{}'s scratch ({} bytes) exceeds the pool ({} bytes)", what, *scratch,
                             launch.scratch_size(0).value()));
  }
  auto lane_scratch = kg::PlanLaneScratch(launch, planned.plan);
  if (!lane_scratch) {
    return Error(lane_scratch.error().detail);
  }
  if (*lane_scratch > launch.scratch_size(1).value()) {
    return Error(std::format("{}'s lane scratch ({} bytes) exceeds a lane's pool ({} bytes)", what,
                             *lane_scratch, launch.scratch_size(1).value()));
  }
  planned.scratch = *scratch;
  auto bound = kg::BoundGraph::Bind(registry, planned.plan);
  if (!bound) {
    return Error(bound.error().detail);
  }
  planned.bound.emplace(std::move(*bound));
  return {};
}

std::uint64_t PlannedNodes(const PlannedBase& planned) {
  std::uint64_t nodes = 0;
  for (const kg::PlanStep& step : planned.plan.steps) {
    nodes += step.nodes.size();
  }
  return nodes;
}

std::uint64_t PlannedHostBytes(const PlannedBase& planned) {
  const std::uint64_t arena = planned.arena ? planned.arena->bytes() : 0;
  return arena + (PlannedNodes(planned) * kPlanNodeHostBytes);
}

std::uint64_t NextPlanUse() {
  static std::atomic<std::uint64_t> next{0};
  return next.fetch_add(1, std::memory_order_relaxed) + 1;
}

namespace {
// The steps under way on this thread (PlanStep): how deep, and the first
// one's start.
thread_local std::uint32_t step_depth = 0;
thread_local std::uint64_t step_start = kNoStep;
}  // namespace

PlanStep::PlanStep() {
  if (step_depth++ == 0) {
    step_start = NextPlanUse();
  }
}

PlanStep::~PlanStep() {
  if (--step_depth == 0) {
    step_start = kNoStep;
  }
}

std::uint64_t PlanStepStart() { return step_start; }

void CollectPlans(std::span<PlanCacheBase* const> caches, std::uint32_t owner, bool running,
                  std::vector<memory::ReclaimCandidate>& out) {
  for (PlanCacheBase* cache : caches) {
    if (cache != nullptr) {
      cache->Collect(owner, running, out);
    }
  }
}

std::uint64_t ReclaimPlan(std::span<PlanCacheBase* const> caches, memory::ReclaimKind kind,
                          std::uint64_t serial) {
  for (PlanCacheBase* cache : caches) {
    if (cache != nullptr) {
      if (const std::uint64_t freed = cache->Reclaim(kind, serial); freed != 0) {
        return freed;
      }
    }
  }
  return 0;
}

std::expected<kg::TensorArena, std::string> SizedArena(
    std::size_t estimate, const std::function<bool(kg::TensorArena&)>& build) {
  // The process's scratch arena, grown to the largest estimate seen: the
  // first build's tensors are only counted, then forgotten.
  Scratch& s = TheScratch();
  const std::scoped_lock lock(s.mutex);
  if (!s.arena || s.tensors < estimate) {
    s.arena.reset();
    s.bytes.store(0);
    auto made = kg::TensorArena::Create(estimate);
    if (!made) {
      return Error(made.error().detail);
    }
    s.arena.emplace(std::move(*made));
    s.tensors = estimate;
    if (s.arena->bytes() > std::numeric_limits<std::uint64_t>::max() - s.index_bytes.load())
      return Error("combined arena/index scratch allowance overflow");
    s.bytes.store(s.arena->bytes());
  }
  kg::TensorArena& scratch = *s.arena;
  scratch.Reset();
  // Within the estimate, as a build in an arena of that size would be.
  const bool built = build(scratch) && scratch.used() <= estimate * ggml_tensor_overhead();
  const std::size_t used = scratch.used();
  scratch.Reset();
  // A build that failed is the caller's own to report, over an arena of
  // the estimate as before.
  auto sized = built ? kg::TensorArena::CreateBytes(used + ggml_tensor_overhead(),
                                                    estimate * ggml_tensor_overhead())
                     : kg::TensorArena::Create(estimate);
  if (!sized) {
    return Error(sized.error().detail);
  }
  return std::move(*sized);
}

void CheckCoverage(const PagedNode& node, int owner, std::span<ggml_tensor* const> nodes,
                   const TensorClasses& classes, Coverage& coverage) {
  const auto in = [](const auto& list, const ggml_tensor* t) {
    return std::ranges::find(list, t) != list.end();
  };
  const auto kind_of = [&](const ggml_tensor* t) {
    const ggml_tensor* base = t->view_src != nullptr ? t->view_src : t;
    if (in(classes.state, base)) {
      return MemoryClass::kLiveState;
    }
    if (in(classes.runtime, base)) {
      return MemoryClass::kRuntime;
    }
    if (in(classes.scratch, base)) {
      return MemoryClass::kScratch;
    }
    return base->op == GGML_OP_NONE && !in(classes.inputs, base) ? MemoryClass::kWeights
                                                                 : MemoryClass::kScratch;
  };
  const auto expect = [&](const ggml_tensor* t, const ggml_tensor* consumer) {
    ++coverage.tensors;
    const MemoryClass expected = kind_of(t);
    const std::optional<MemoryClass> covered =
        node.Covered(Address(t->data), ggml_nbytes(t), owner, expected != MemoryClass::kLiveState);
    if (covered != expected && coverage.violations++ == 0) {
      coverage.first_violation = std::format(
          "{}{} ({} {} [{}, {}, {}, {}], {} bytes at {:#x}, read by {} {})", classes.what, t->name,
          ggml_op_desc(t), ggml_type_name(t->type), t->ne[0], t->ne[1], t->ne[2], t->ne[3],
          ggml_nbytes(t), Address(t->data), consumer != nullptr ? ggml_op_desc(consumer) : "-",
          consumer != nullptr ? consumer->name : "");
    }
  };
  for (const ggml_tensor* n : nodes) {
    expect(n, nullptr);
    if (classes.fill_reads_nothing && n->op == GGML_OP_FILL) {
      continue;
    }
    for (const ggml_tensor* src : n->src) {
      if (src != nullptr) {
        expect(src, n);
      }
    }
  }
}

}  // namespace jitllm::engine
