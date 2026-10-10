// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "engine/component_source.h"

#include <algorithm>
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
    auto text =
        spec::DefaultPolicy(*options.mapping, options.values,
                            {.context = options.bindings.state->context, .slots = options.slots});
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
  auto instances = components::InstancesOf(*options.spec, options.stage);
  if (!instances) return std::unexpected(instances.error());
  auto source = std::unique_ptr<ComponentStepSource>(
      new ComponentStepSource(std::move(options), std::move(policy)));
  const auto add = [&](const components::Instance& in) {
    if (in.def->shape_parameters != nullptr) source->shaped_.push_back(in);
  };
  if (instances->embedding) add(*instances->embedding);
  for (const auto& in : instances->side) add(in);
  if (instances->mixing) add(*instances->mixing);
  for (const auto& layer : instances->layers)
    for (const auto& in : layer) add(in);
  if (instances->final_norm) add(*instances->final_norm);
  if (instances->output) add(*instances->output);
  return source;
}

planner::PlanKey ComponentStepSource::KeyBase() const {
  return planner::KeyOf(o_.bound_digest, {}, policy_);
}

std::vector<std::uint64_t> ComponentStepSource::ShapeParameters(
    const planner::StepShape& shape) const {
  std::vector<std::uint64_t> out;
  for (const auto& in : shaped_)
    for (const auto& segment : shape.segments)
      in.def->shape_parameters(in, segment.rows, segment.past, shape.read_align, out);
  return out;
}

std::expected<std::vector<CommitBuffer>, std::string> ComponentStepSource::CommitBuffers(
    std::uint16_t stage, std::uint32_t rows) const {
  if (stage != o_.stage || rows == 0 || rows > o_.bindings.state->context) {
    return std::unexpected("a commit buffer's stage or rows");
  }
  const bool commits = std::ranges::any_of(o_.bindings.state->items, [](const auto& item) {
    return item.contract.rollback == model::Rollback::kCommit;
  });
  if (!commits) return std::vector<CommitBuffer>{};
  planner::StepShape shape{
      .stage = stage,
      .phase = planner::Phase::kVerify,
      .segments = {{.rows = rows, .read_cells = o_.bindings.state->context, .outputs = rows}}};
  auto step = components::BuildStep(*o_.spec, stage, shape, o_.bindings);
  if (!step) return std::unexpected(step.error());
  std::vector<CommitBuffer> buffers;
  for (const components::CommitSave& save : step->saves) {
    const auto& tensor = step->graph.tensor(save.tensor);
    std::uint64_t bytes = sizeof(float);
    for (const std::uint64_t extent : tensor.shape().ne) bytes *= extent;
    if (bytes % rows != 0) return std::unexpected("a commit save is not row-shaped");
    buffers.push_back({.layer = save.layer,
                       .component = save.component,
                       .what = save.what,
                       .row_bytes = bytes / rows});
  }
  return buffers;
}

std::expected<std::unique_ptr<kernels::StepProgram>, std::string> ComponentStepSource::Build(
    const planner::StepShape& shape, const kernels::Addresses& addresses) {
  if (shape.stage != o_.stage) {
    return std::unexpected(
        std::format("a step of stage {} from stage {}'s source", shape.stage, o_.stage));
  }
  // The narrowed final layer is the policy's (planner/pattern.h
  // narrow_final).
  auto step = shape.phase == planner::Phase::kCommit
                  ? components::BuildCommit(*o_.spec, o_.stage, shape, o_.bindings)
                  : components::BuildStep(*o_.spec, o_.stage, shape, o_.bindings,
                                          {.narrow = !planner::Disabled(policy_, "narrow_final")});
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
                             .within = o_.backend.within,
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

std::expected<std::unique_ptr<StagedStepSource>, std::string> StagedStepSource::Create(
    std::vector<std::unique_ptr<ComponentStepSource>> stages) {
  if (stages.empty()) return std::unexpected("a staged step source needs a stage");
  base::Sha256 patterns, policies;
  for (std::size_t i = 0; i < stages.size(); ++i) {
    if (stages[i] == nullptr || stages[i]->stage() != i) {
      return std::unexpected(std::format("stage {}'s step source is not its own", i));
    }
    if (stages[i]->bound_digest() != stages.front()->bound_digest()) {
      return std::unexpected(std::format("stage {}'s step source plans another bound spec", i));
    }
    const planner::PlanKey key = stages[i]->KeyBase();
    patterns.Update(std::as_bytes(std::span(key.patterns)));
    policies.Update(std::as_bytes(std::span(key.policy)));
  }
  planner::PlanKey key = stages.front()->KeyBase();
  key.patterns = patterns.Finish();
  key.policy = policies.Finish();
  return std::unique_ptr<StagedStepSource>(new StagedStepSource(std::move(stages), key));
}

std::vector<std::uint64_t> StagedStepSource::ShapeParameters(
    const planner::StepShape& shape) const {
  return shape.stage < stages_.size() ? stages_[shape.stage]->ShapeParameters(shape)
                                      : std::vector<std::uint64_t>{};
}

std::expected<std::unique_ptr<kernels::StepProgram>, std::string> StagedStepSource::Build(
    const planner::StepShape& shape, const kernels::Addresses& addresses) {
  if (shape.stage >= stages_.size()) {
    return std::unexpected(std::format("a step of stage {}, which the model lacks", shape.stage));
  }
  return stages_[shape.stage]->Build(shape, addresses);
}

std::expected<std::vector<CommitBuffer>, std::string> StagedStepSource::CommitBuffers(
    std::uint16_t stage, std::uint32_t rows) const {
  if (stage >= stages_.size()) return std::unexpected("a commit names no stage");
  return stages_[stage]->CommitBuffers(stage, rows);
}

}  // namespace llmp::engine
