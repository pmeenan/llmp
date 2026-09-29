// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "engine/planned.h"

#include <algorithm>
#include <format>
#include <optional>
#include <utility>

#include "catalog/catalog.h"
#include "engine/paged_node.h"
#include "engine/support.h"

namespace jitllm::engine {

namespace {

namespace kg = jitllm::kernels::ggml;
using catalog::MemoryClass;
using support::Address;
using support::Error;
using support::Round;

}  // namespace

std::expected<void, std::string> PlaceAndPlan(PlannedBase& out, std::span<ggml_tensor* const> nodes,
                                              std::span<ggml_tensor* const> inputs,
                                              std::span<ggml_tensor* const> keep,
                                              const kg::DeviceChoices& choices,
                                              std::uint64_t activations,
                                              std::uint64_t activation_bytes) {
  constexpr std::uint64_t kDistinct = std::uint64_t{1} << 46U;
  std::uint64_t leaf = kDistinct - (std::uint64_t{1} << 40U);
  for (ggml_tensor* input : inputs) {
    kg::TensorArena::Bind(input, leaf);
    leaf += Round(ggml_nbytes(input), 256) + 256;
  }
  kg::BindDistinct(nodes, kDistinct);
  auto first = kg::PlanGraph(nodes, /*fusion=*/false, choices);
  if (!first) {
    return Error(first.error().detail);
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
  auto second = kg::PlanGraph(nodes, false, choices);
  if (!second) {
    return Error(second.error().detail);
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
  auto scratch = kg::PlanScratch(launch, planned.plan);
  if (!scratch) {
    return Error(scratch.error().detail);
  }
  if (*scratch > launch.workspace().size.value()) {
    return Error(std::format("{}'s scratch ({} bytes) exceeds the pool ({} bytes)", what, *scratch,
                             launch.workspace().size.value()));
  }
  planned.scratch = *scratch;
  auto bound = kg::BoundGraph::Bind(registry, planned.plan);
  if (!bound) {
    return Error(bound.error().detail);
  }
  planned.bound.emplace(std::move(*bound));
  return {};
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
