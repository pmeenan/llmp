// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "engine/graph_plan.h"

#include <algorithm>
#include <format>
#include <utility>
#include <vector>

namespace llmp::engine {

std::expected<planner::StepPlan, std::string> PlanComponentGraph(
    const GraphBuildIO& io, const kernels::StepBackend& backend, const planner::Policy& policy,
    const kernels::Addresses& addresses, planner::PlanOptions options) {
  if (backend.registry == nullptr || !std::ranges::equal(io.inputs, io.graph.inputs()) ||
      !std::ranges::equal(io.outputs, io.graph.outputs()))
    return std::unexpected("a component graph needs its exact typed interface and registry");
  for (const auto& weight : io.weights) {
    if (!weight.tensor.valid() || weight.tensor.v >= io.graph.tensor_count())
      return std::unexpected("a component weight names no graph tensor");
    const auto& desc = io.graph.tensor(weight.tensor);
    const auto* ref = std::get_if<ir::WeightRef>(&desc.binding);
    if (desc.is_view() ||
        io.graph.node(io.graph.producer(weight.tensor)).key.op != ir::Op::kWeight ||
        ref == nullptr || *ref != weight.ref)
      return std::unexpected("a component weight differs from its graph root binding");
  }
  const auto power_of_two = [](std::uint64_t n) { return n != 0 && (n & (n - 1)) == 0; };
  if (addresses.activation_alignment != 0 && !power_of_two(addresses.activation_alignment))
    return std::unexpected("the activation reservation alignment is not a power of two");
  if (!addresses.measuring() && addresses.activation_alignment != 0 &&
      addresses.activations % addresses.activation_alignment != 0)
    return std::unexpected("the activation base breaks its reservation alignment promise");
  std::vector<std::pair<ir::TensorDesc, std::uint64_t>> physical_facts;
  if (addresses.state_alignment) {
    for (const auto& node : io.graph.nodes()) {
      if (node.key.op != ir::Op::kState) continue;
      const auto& desc = io.graph.tensor(node.out[0]);
      const auto* ref = std::get_if<ir::StateRef>(&desc.binding);
      if (ref == nullptr) continue;
      const auto alignment = addresses.state_alignment(*ref);
      if (alignment == 0) continue;
      if (!power_of_two(alignment))
        return std::unexpected("a state reservation alignment is not a power of two");
      const auto address = addresses.state ? addresses.state(*ref) : 0;
      if (address != 0 && address % alignment != 0)
        return std::unexpected("a state base breaks its reservation alignment promise");
      physical_facts.emplace_back(desc, alignment);
    }
  }
  if (addresses.weight_alignment) {
    for (const auto& weight : io.weights) {
      // A multipart logical root has no single physical address. Its parts
      // remain unknown here; no first-part alignment is promoted to the root.
      if (weight.parts.size() != 1) continue;
      const auto& desc = io.graph.tensor(weight.tensor);
      const auto parts = format::PartsOf(desc.format);
      if (!parts || parts->size() != 1 || (*parts)[0].role != weight.parts[0].second)
        return std::unexpected("a weight alignment does not name its single format part");
      const auto part = weight.parts[0].second;
      const auto alignment = addresses.weight_alignment(weight.ref, part);
      if (alignment == 0) continue;
      if (!power_of_two(alignment))
        return std::unexpected("a weight reservation alignment is not a power of two");
      const auto address = addresses.weight ? addresses.weight(weight.ref, part) : 0;
      if (address == 0 || address % alignment != 0)
        return std::unexpected("a weight base breaks its reservation alignment promise");
      physical_facts.emplace_back(desc, alignment);
    }
  }
  planner::RootAlignment root_alignment;
  if (!physical_facts.empty()) {
    root_alignment = [facts = std::move(physical_facts)](const ir::Graph& graph,
                                                         ir::TensorId root) -> std::uint64_t {
      if (!root.valid() || root.v >= graph.tensor_count()) return 0;
      const auto& desc = graph.tensor(root);
      const auto op = graph.node(graph.producer(root)).key.op;
      if (desc.is_view() || (op != ir::Op::kState && op != ir::Op::kWeight)) return 0;
      // Exact semantic leaf identity survives authorized rewrites. The format
      // also fixes the single part checked above; no unrelated binding gains it.
      for (const auto& [saved, alignment] : facts) {
        if (desc.binding == saved.binding && desc.format == saved.format && desc.nb == saved.nb &&
            desc.role == saved.role && desc.segment == saved.segment)
          return alignment;
      }
      return 0;
    };
  }
  options.root_alignment = std::move(root_alignment);
  options.activation_alignment = addresses.activation_alignment;
  auto plan = planner::Plan(io.graph.Clone(), *backend.registry, backend.device, policy, options);
  if (!plan)
    return std::unexpected(std::format("planning a step ({}): {}",
                                       planner::PlanErrorName(plan.error().error),
                                       plan.error().detail));
  return std::move(*plan);
}

std::expected<std::unique_ptr<kernels::StepProgram>, std::string> BuildComponentProgram(
    const GraphBuildIO& io, const kernels::StepBackend& backend, const planner::Policy& policy,
    const kernels::Addresses& addresses, planner::PlanOptions options,
    std::optional<kernels::TokenLowerInput> token, const kernels::ProgramValidator& validate) {
  if (!backend.lower) return std::unexpected("a component graph needs a backend lowerer");
  if (token && (&token->step.graph != &io.graph || options.shape != &token->shape))
    return std::unexpected("token lowering metadata differs from the planned graph or shape");
  if (!token && (options.shape != nullptr || options.state_spans))
    return std::unexpected("a spatial graph cannot borrow token shape or state facts");
  auto plan = PlanComponentGraph(io, backend, policy, addresses, std::move(options));
  if (!plan) return std::unexpected(std::move(plan.error()));
  const kernels::LowerInput input{*plan, io, addresses, token};
  auto lowered = backend.lower(input);
  if (!lowered) return lowered;
  if (!*lowered) return std::unexpected("the backend returned no step program");
  if (validate) {
    auto valid = validate(input, **lowered);
    if (!valid) return std::unexpected(valid.error());
  }
  return lowered;
}

}  // namespace llmp::engine
