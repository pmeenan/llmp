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
    if (in.def->side_rows != nullptr) source->side_.push_back(in);
  };
  if (instances->embedding) add(*instances->embedding);
  for (const auto& in : instances->side) add(in);
  if (instances->mixing) add(*instances->mixing);
  for (const auto& layer : instances->layers)
    for (const auto& in : layer) add(in);
  if (instances->final_norm) add(*instances->final_norm);
  if (instances->output) add(*instances->output);
  base::Sha256 layout_hash;
  layout_hash.Update(model::KeptLayoutId({}, std::span(source->o_.bindings.state, 1)));
  source->layout_identity_ = layout_hash.Finish();
  return source;
}

planner::PlanKey ComponentStepSource::KeyBase() const {
  return planner::KeyOf(o_.bound_digest, {}, policy_);
}

std::vector<std::uint64_t> ComponentStepSource::ShapeParameters(
    const planner::StepShape& shape) const {
  std::vector<std::uint64_t> out;
  if (!planner::Disabled(policy_, "state_bank_attention")) {
    out.push_back(0x737461746562616eULL);  // state-bank layout identity, fixed four words
    for (std::size_t word = 0; word < 4; ++word) {
      std::uint64_t value = 0;
      for (std::size_t byte = 0; byte < 8; ++byte)
        value |= std::uint64_t{layout_identity_[word * 8 + byte]} << (byte * 8);
      out.push_back(value);
    }
  }
  for (const auto& in : shaped_)
    for (const auto& segment : shape.segments)
      in.def->shape_parameters(in, segment.rows, segment.past, shape.read_align, out);
  return out;
}

planner::StateSpans ComponentStepSource::StateSpans(const planner::StepShape& shape,
                                                    const kernels::Addresses& addresses) const {
  return [this, &shape, &addresses](std::span<const planner::StateCellSlice> slices)
             -> std::optional<planner::StateSpanFact> {
    if (slices.empty()) return std::nullopt;
    const auto owner = slices.front().ref.segment;
    if (owner >= shape.segments.size()) return std::nullopt;
    std::vector<model::StateSlice> requested;
    for (const auto& slice : slices) {
      if (slice.ref.stage != o_.stage || slice.ref.segment != owner) return std::nullopt;
      requested.push_back({.layer = slice.ref.layer,
                           .tensor = slice.ref.tensor,
                           .first = slice.first,
                           .cells = slice.cells});
    }
    const auto& layout = *o_.bindings.state;
    auto bank = model::ContiguousStateSpan(layout, requested);
    if (!bank) return std::nullopt;
    // Addresses are of whole items. Prove one stage/slot base, not merely
    // adjacency of two unrelated allocations. Missing measurement addresses
    // are replaced together from this same immutable region by the lowering.
    std::optional<std::uint64_t> base;
    bool missing = false;
    for (const auto& slice : slices) {
      const auto item = std::ranges::find_if(layout.items, [&](const auto& item) {
        return item.layer == slice.ref.layer && item.tensor == slice.ref.tensor;
      });
      if (item == layout.items.end()) return std::nullopt;
      const auto address = addresses.state ? addresses.state(slice.ref) : 0;
      if (address == 0) {
        missing = true;
        continue;
      }
      if (address < item->offset) return std::nullopt;
      const auto candidate = address - item->offset;
      std::uint64_t end = 0;
      if (__builtin_add_overflow(candidate, layout.bytes, &end) || (base && *base != candidate))
        return std::nullopt;
      base = candidate;
    }
    if (missing && (!addresses.measuring() || base)) return std::nullopt;
    return planner::StateSpanFact{.bank = *bank, .region_bytes = layout.bytes};
  };
}

bool ComponentStepSource::UsesSideRows(std::uint16_t stage) const {
  return stage == o_.stage && !side_.empty();
}

Status ComponentStepSource::SideRows(std::uint16_t stage, std::uint16_t component,
                                     std::span<const std::int32_t> history, std::uint32_t position,
                                     std::span<std::int32_t> out) const {
  if (stage != o_.stage) return std::unexpected("side rows name another stage");
  const auto found = std::ranges::find(side_, component, &components::Instance::index);
  if (found == side_.end()) return std::unexpected("side rows name no table component");
  return found->def->side_rows(*found, history, position, out);
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
  // Keep the existing placement set. Cell positions are synthesized by
  // masks; commit rows retain their legacy reserved place but are bound to
  // external saved rows, so receive no arena alignment guarantee.
  std::vector<ir::TensorId> staged, aligned_staged;
  for (const components::StepInput& input : step->inputs) {
    if (input.kind != components::InputKind::kCellPositions) staged.push_back(input.tensor);
    if (input.kind != components::InputKind::kCellPositions &&
        input.kind != components::InputKind::kCommitRows)
      aligned_staged.push_back(input.tensor);
  }
  const auto power_of_two = [](std::uint64_t n) { return n != 0 && (n & (n - 1)) == 0; };
  if (addresses.activation_alignment != 0 && !power_of_two(addresses.activation_alignment))
    return std::unexpected("the activation reservation alignment is not a power of two");
  if (!addresses.measuring() && addresses.activation_alignment != 0 &&
      addresses.activations % addresses.activation_alignment != 0)
    return std::unexpected("the activation base breaks its reservation alignment promise");
  std::vector<std::pair<ir::TensorDesc, std::uint64_t>> state_facts;
  if (addresses.state_alignment) {
    for (const auto& node : step->graph.nodes()) {
      if (node.key.op != ir::Op::kState) continue;
      const auto& desc = step->graph.tensor(node.out[0]);
      const auto* ref = std::get_if<ir::StateRef>(&desc.binding);
      if (ref == nullptr) continue;
      const auto alignment = addresses.state_alignment(*ref);
      if (alignment == 0) continue;
      if (!power_of_two(alignment))
        return std::unexpected("a state reservation alignment is not a power of two");
      const auto address = addresses.state ? addresses.state(*ref) : 0;
      if (address != 0 && address % alignment != 0)
        return std::unexpected("a state base breaks its reservation alignment promise");
      state_facts.emplace_back(desc, alignment);
    }
  }
  planner::RootAlignment root_alignment;
  if (!state_facts.empty()) {
    root_alignment = [facts = std::move(state_facts)](const ir::Graph& graph,
                                                      ir::TensorId root) -> std::uint64_t {
      if (!root.valid() || root.v >= graph.tensor_count()) return 0;
      const auto& desc = graph.tensor(root);
      if (desc.is_view() || graph.node(graph.producer(root)).key.op != ir::Op::kState) return 0;
      for (const auto& [saved, alignment] : facts) {
        if (desc.binding == saved.binding && desc.format == saved.format && desc.nb == saved.nb &&
            desc.role == saved.role && desc.segment == saved.segment)
          return alignment;
      }
      return 0;
    };
  }
  const auto state_spans = StateSpans(shape, addresses);
  auto plan = planner::Plan(step->graph.Clone(), *o_.backend.registry, o_.backend.device, policy_,
                            {.row_invariant = shape.row_invariant,
                             .capturable = true,
                             .staged = std::move(staged),
                             .on_stream = o_.backend.on_stream,
                             .written = o_.backend.written,
                             .within = o_.backend.within,
                             .shape = &shape,
                             .state_spans = state_spans,
                             .root_alignment = std::move(root_alignment),
                             .activation_alignment = addresses.activation_alignment,
                             .alignment_staged = std::move(aligned_staged)});
  if (!plan) {
    return std::unexpected(std::format("planning a step ({}): {}",
                                       planner::PlanErrorName(plan.error().error),
                                       plan.error().detail));
  }
  const kernels::LowerInput input{.plan = *plan,
                                  .step = *step,
                                  .shape = shape,
                                  .addresses = addresses,
                                  .context = o_.bindings.state->context,
                                  .state_spans = state_spans};
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

bool StagedStepSource::JoinedRepeatPending(std::uint16_t stage) const {
  return stage < stages_.size() && stages_[stage]->JoinedRepeatPending(stage);
}

bool StagedStepSource::UsesSideRows(std::uint16_t stage) const {
  return stage < stages_.size() && stages_[stage]->UsesSideRows(stage);
}

Status StagedStepSource::SideRows(std::uint16_t stage, std::uint16_t component,
                                  std::span<const std::int32_t> history, std::uint32_t position,
                                  std::span<std::int32_t> out) const {
  if (stage >= stages_.size()) return std::unexpected("side rows name no stage");
  return stages_[stage]->SideRows(stage, component, history, position, out);
}

std::expected<std::vector<CommitBuffer>, std::string> StagedStepSource::CommitBuffers(
    std::uint16_t stage, std::uint32_t rows) const {
  if (stage >= stages_.size()) return std::unexpected("a commit names no stage");
  return stages_[stage]->CommitBuffers(stage, rows);
}

}  // namespace llmp::engine
