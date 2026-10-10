// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "engine/component_source.h"

#include <format>
#include <utility>
#include <vector>

#include "ir/graph.h"
#include "planner/plan.h"

namespace llmp::engine {

std::expected<std::unique_ptr<ComponentStepSource>, std::string> ComponentStepSource::Create(
    ComponentSourceOptions options) {
  if (options.spec == nullptr || options.bindings.state == nullptr) {
    return std::unexpected("a component step source needs its spec and state layout");
  }
  if (options.backend.registry == nullptr || !options.backend.lower) {
    return std::unexpected("a component step source needs a kernel source");
  }
  planner::Policy policy;
  if (options.policy) {
    policy = *options.policy;
  } else if (options.mapping != nullptr) {
    // The mapping's rule for this source's values; none: the planner's
    // default.
    auto text = spec::DefaultPolicy(*options.mapping, options.values);
    if (!text) {
      return std::unexpected(std::format("mapping {}'s planning policy: {}",
                                         options.mapping->architecture, text.error()));
    }
    if (!text->empty()) {
      auto parsed = planner::ParsePolicy(*text);
      if (!parsed) {
        return std::unexpected(std::format("mapping {}'s planning policy: {}",
                                           options.mapping->architecture, parsed.error()));
      }
      policy = std::move(*parsed);
    }
  } else {
    return std::unexpected("a component step source needs a planning policy or a mapping");
  }
  if (auto valid = planner::Validate(policy); !valid) {
    return std::unexpected(std::format("the planning policy: {}", valid.error()));
  }
  return std::unique_ptr<ComponentStepSource>(
      new ComponentStepSource(std::move(options), std::move(policy)));
}

planner::PlanKey ComponentStepSource::KeyBase() const {
  return planner::KeyOf(o_.bound_digest, {}, policy_);
}

std::expected<std::unique_ptr<kernels::StepProgram>, std::string> ComponentStepSource::Build(
    const planner::StepShape& shape, const kernels::Addresses& addresses) {
  if (shape.stage != o_.stage) {
    return std::unexpected(
        std::format("a step of stage {} from stage {}'s source", shape.stage, o_.stage));
  }
  auto step = components::BuildStep(*o_.spec, o_.stage, shape, o_.bindings);
  if (!step) return std::unexpected(std::move(step.error()));
  // Every input is staged but the cells' positions, which only masks read:
  // a lowering builds them on the device or folds them into a host mask.
  std::vector<ir::TensorId> staged;
  for (const components::StepInput& input : step->inputs) {
    if (input.kind != components::InputKind::kCellPositions) staged.push_back(input.tensor);
  }
  auto plan = planner::Plan(step->graph.Clone(), *o_.backend.registry, o_.backend.device, policy_,
                            {.row_invariant = shape.row_invariant,
                             .capturable = true,
                             .staged = std::move(staged),
                             .on_stream = o_.backend.on_stream,
                             .written = o_.backend.written,
                             .shape = &shape});
  if (!plan) {
    return std::unexpected(std::format("planning a step ({}): {}",
                                       planner::PlanErrorName(plan.error().error),
                                       plan.error().detail));
  }
  const kernels::LowerInput input{.plan = *plan,
                                  .step = *step,
                                  .shape = shape,
                                  .addresses = addresses,
                                  .context = o_.bindings.state->context};
  return o_.backend.lower(input);
}

}  // namespace llmp::engine
