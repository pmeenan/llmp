// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "engine/component_source.h"

#include <algorithm>
#include <array>
#include <format>
#include <utility>
#include <vector>

#include "format/tensor_format.h"
#include "ir/graph.h"
#include "planner/plan.h"

namespace llmp::engine {

std::expected<std::unique_ptr<ComponentStepSource>, std::string> ComponentStepSource::Create(
    ComponentSourceOptions options) {
  if (options.backend.owner) {
    auto owner = options.backend.owner;
    // Retain just the owned table, avoiding another allocation/copy of its
    // device/tuning vectors. Caller-supplied copied fields have no authority.
    options.backend = {.owner = std::move(owner)};
  }
  if (static_cast<bool>(options.backend.owner) != static_cast<bool>(options.execution_domain))
    return std::unexpected("a required source needs both its backend owner and complete domain");
  if (options.spec == nullptr || options.bindings.state == nullptr) {
    return std::unexpected("a component step source needs its spec and state layout");
  }
  const auto& backend = options.backend.owner ? options.backend.owner->backend() : options.backend;
  if (backend.registry == nullptr || !backend.lower) {
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
  if (!planner::Disabled(policy, "rounded_indexer")) {
    if (!options.backend.owner || !options.execution_domain || options.stage != 0 ||
        options.slots != 1 || options.spec->stages.size() != 1 ||
        options.execution_domain->stages().size() != 1 || options.bindings.state->context < 4 ||
        options.bindings.state->context > 4096 || !options.bindings.state->borrowed.empty())
      return std::unexpected("rounded indexer requires an owned single-target execution domain");
    const auto& stage = options.execution_domain->stages().front();
    if (stage.stage != 0 || stage.name != options.spec->stages.front().name)
      return std::unexpected("rounded indexer domain names another target");
    for (const auto use : stage.uses)
      if (use != NumericalUse::kPrompt && use != NumericalUse::kDecode &&
          use != NumericalUse::kScore)
        return std::unexpected("rounded indexer does not admit companion or speculative modes");
    for (const auto& phase : stage.phases)
      if ((phase.phase != planner::Phase::kPrefill && phase.phase != planner::Phase::kDecode &&
           phase.phase != planner::Phase::kScore) ||
          phase.max_owners != 1 || phase.max_owner_rows > 8 || phase.max_total_rows > 8 ||
          phase.row_invariant)
        return std::unexpected("rounded indexer domain exceeds its bounded ordinary phases");
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

struct ComponentStepSource::ExecutionState {
  std::vector<std::shared_ptr<const kernels::BackendOwner>> owners;
  std::vector<execution::BackendDeclaration> declarations;
  base::Sha256Digest identity{};
};

std::expected<std::shared_ptr<ComponentStepSource::ExecutionState>, std::string>
ComponentStepSource::SealExecution(std::span<ComponentStepSource* const> stages) {
  if (stages.empty()) return std::unexpected("an execution composition has no stages");
  const auto& domain = stages.front()->o_.execution_domain;
  if (!domain) {
    for (const auto* stage : stages)
      if (stage->o_.execution_domain || stage->o_.backend.owner)
        return std::unexpected("a required execution composition omits a stage");
    return std::shared_ptr<ExecutionState>{};
  }
  if (domain->stages().size() != stages.size() ||
      stages.front()->o_.spec->stages.size() != stages.size())
    return std::unexpected("a required execution composition must include every source stage");
  std::uint64_t bytes = sizeof(ExecutionState) + 512;
  const auto add = [&](std::uint64_t n) { return !__builtin_add_overflow(bytes, n, &bytes); };
  const auto text = [&](std::string_view value) {
    std::uint64_t n = 0;
    return !__builtin_add_overflow(static_cast<std::uint64_t>(value.size()),
                                   sizeof(std::string) + std::uint64_t{128}, &n) &&
           !__builtin_mul_overflow(n, std::uint64_t{4}, &n) && add(n);
  };
  // Count all new copies and canonical text/hash construction before making
  // StageExecutionInput/ExecutionContract. The domain and policies supplied
  // by the caller already belong to their existing source allowance.
  for (std::size_t i = 0; i < stages.size(); ++i) {
    const auto* stage = stages[i];
    if (!stage || stage->o_.stage != i || !stage->o_.backend.owner || !stage->o_.execution_domain ||
        stage->o_.execution_domain->identity() != domain->identity() ||
        stage->o_.spec->stages.size() != stages.size() ||
        stage->o_.spec->stages[i].name != domain->stages()[i].name)
      return std::unexpected("required source stages disagree or lack their backend owner");
    if (stage->execution_) return std::unexpected("an execution source cannot be prepared twice");
    auto bound = planner::OwnedDeviceFacts::Bound(stage->Backend().device);
    if (!bound || !add(*bound) || !add(sizeof(StageExecutionInput) + 1024) ||
        !text(domain->stages()[i].name))
      return std::unexpected("execution metadata bound overflows");
    for (const auto& name : stage->policy_.enabled)
      if (!text(name)) return std::unexpected("execution policy bound overflows");
    for (const auto& name : stage->policy_.disabled)
      if (!text(name)) return std::unexpected("execution policy bound overflows");
    for (const auto& rank : stage->policy_.ranks)
      if (!text(rank.device_class) || !text(rank.implementation) || !add(sizeof(rank) + 128))
        return std::unexpected("execution rank bound overflows");
    const auto& declared = domain->stages()[i];
    // Known unique enums bound the vectors to the finite enum inventory.
    if (!add(declared.phases.size() * (sizeof(NumericalPhase) + 64)) ||
        !add(declared.uses.size() * (sizeof(NumericalUse) + 64)))
      return std::unexpected("execution domain bound overflows");
  }
  const auto funder = stages.front()->o_.backend.owner;
  if (!funder->Charge(bytes)) return std::unexpected("execution metadata was not funded");
  auto state =
      std::shared_ptr<ExecutionState>(new ExecutionState, [funder, bytes](ExecutionState* value) {
        delete value;
        funder->Release(bytes);
      });
  state->owners.reserve(stages.size());
  state->owners.push_back(funder);
  state->declarations.reserve(stages.size());
  std::vector<StageExecutionInput> inputs;
  inputs.reserve(stages.size());
  for (std::size_t i = 0; i < stages.size(); ++i) {
    const auto* stage = stages[i];
    const auto& backend = stage->Backend();
    auto actual = stage->o_.backend.owner->Ready();
    if (!actual) return std::unexpected(actual.error());
    if (i != 0) state->owners.push_back(stage->o_.backend.owner);
    state->declarations.push_back(*actual);
    const auto key = planner::KeyOf(stage->o_.bound_digest, {}, stage->policy_);
    inputs.push_back({.stage = stage->o_.stage,
                      .name = domain->stages()[i].name,
                      .policy = stage->policy_,
                      .patterns = key.patterns,
                      .registry = backend.registry,
                      .device = backend.device,
                      .backend = *actual});
  }
  auto contract = ExecutionContract::Create(*domain, inputs);
  if (!contract) return std::unexpected(contract.error());
  base::Sha256 hash;
  hash.Update("llmp-component-execution-v2");
  hash.Update(std::as_bytes(std::span(contract->identity())));
  for (const auto* stage : stages) {
    hash.Update(std::as_bytes(std::span(stage->o_.bound_digest)));
    hash.Update(std::as_bytes(std::span(stage->layout_identity_)));
    hash.Update(
        std::format("{}:{}:{};", stage->o_.stage, stage->o_.slots, stage->o_.require_capture));
  }
  state->identity = hash.Finish();
  return state;
}

Status ComponentStepSource::PrepareExecution() {
  const std::array<ComponentStepSource*, 1> sources{this};
  auto sealed = SealExecution(sources);
  if (!sealed) return std::unexpected(sealed.error());
  execution_ = std::move(*sealed);
  return {};
}

std::expected<std::optional<base::Sha256Digest>, std::string>
ComponentStepSource::ExecutionIdentity() const {
  if (!o_.execution_domain) return std::optional<base::Sha256Digest>{};
  if (!execution_) return std::unexpected("the required execution source is not prepared");
  for (std::size_t i = 0; i < execution_->owners.size(); ++i) {
    auto actual = execution_->owners[i]->Ready();
    if (!actual) return std::unexpected(actual.error());
    if (*actual != execution_->declarations[i])
      return std::unexpected("the execution backend changed after preparation");
  }
  return std::optional{execution_->identity};
}

Status ComponentStepSource::CheckExecutionUse(NumericalUse use) const {
  if (!o_.execution_domain) return {};
  if (auto identity = ExecutionIdentity(); !identity) return std::unexpected(identity.error());
  return o_.execution_domain->CheckMode(use);
}

Status ComponentStepSource::CheckExecutionShape(const planner::StepShape& shape) const {
  if (!o_.execution_domain) return {};
  if (auto identity = ExecutionIdentity(); !identity) return std::unexpected(identity.error());
  if (auto checked = o_.execution_domain->CheckShape(shape); !checked) return checked;
  if (!planner::Disabled(policy_, "rounded_indexer")) {
    const auto& own = shape.segments.front();  // CheckShape proved the bounded owner set.
    if (own.slot != 0 || own.past > o_.bindings.state->context ||
        own.rows > o_.bindings.state->context - own.past || own.read_cells == 0 ||
        own.read_cells > o_.bindings.state->context)
      return std::unexpected("rounded indexer owner exceeds its admitted context or rows");
  }
  return {};
}

planner::PlanKey ComponentStepSource::KeyBase() const {
  auto key = planner::KeyOf(o_.bound_digest, {}, policy_);
  if (execution_) {
    base::Sha256 hash;
    hash.Update(std::as_bytes(std::span(key.patterns)));
    hash.Update(std::as_bytes(std::span(execution_->identity)));
    key.patterns = hash.Finish();
  }
  return key;
}

std::vector<std::uint64_t> ComponentStepSource::ShapeParameters(
    const planner::StepShape& shape) const {
  std::vector<std::uint64_t> out;
  if (!o_.require_capture) out.push_back(0x6e6f636170747572ULL);  // uncaptured selection identity
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
  if (!planner::Disabled(policy_, "rounded_indexer"))
    return std::unexpected("rounded indexer has no speculative commit buffers");
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
  if (auto checked = CheckExecutionShape(shape); !checked) return std::unexpected(checked.error());
  if (shape.stage != o_.stage) {
    return std::unexpected(
        std::format("a step of stage {} from stage {}'s source", shape.stage, o_.stage));
  }
  // The narrowed final layer is the policy's (planner/pattern.h
  // narrow_final).
  auto step = shape.phase == planner::Phase::kCommit
                  ? components::BuildCommit(*o_.spec, o_.stage, shape, o_.bindings)
                  : components::BuildStep(
                        *o_.spec, o_.stage, shape, o_.bindings,
                        {.narrow = !planner::Disabled(policy_, "narrow_final"),
                         .rounded_indexer = !planner::Disabled(policy_, "rounded_indexer")});
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
  std::vector<std::pair<ir::TensorDesc, std::uint64_t>> physical_facts;
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
      physical_facts.emplace_back(desc, alignment);
    }
  }
  if (addresses.weight_alignment) {
    for (const auto& weight : step->weights) {
      // A multipart logical root has no single physical address. Its parts
      // remain unknown here; no first-part alignment is promoted to the root.
      if (weight.parts.size() != 1) continue;
      const auto& desc = step->graph.tensor(weight.tensor);
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
  const auto state_spans = StateSpans(shape, addresses);
  auto plan = planner::Plan(step->graph.Clone(), *Backend().registry, Backend().device, policy_,
                            {.row_invariant = shape.row_invariant,
                             .capturable = o_.require_capture,
                             .staged = std::move(staged),
                             .on_stream = Backend().on_stream,
                             .written = Backend().written,
                             .within = Backend().within,
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
  auto lowered = Backend().lower(input);
  if (!lowered) return lowered;
  if (!*lowered) return std::unexpected("the backend returned no step program");
  if (o_.validate_program) {
    auto valid = o_.validate_program(input, **lowered);
    if (!valid) return std::unexpected(valid.error());
  }
  return lowered;
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

Status StagedStepSource::PrepareExecution() {
  if (!stages_.front()->o_.execution_domain) {
    for (const auto& stage : stages_)
      if (stage->o_.execution_domain || stage->o_.backend.owner)
        return std::unexpected("a required execution composition omits a stage");
    return {};
  }
  const auto funder = stages_.front()->o_.backend.owner;
  std::uint64_t temporary = 0;
  if (__builtin_mul_overflow(static_cast<std::uint64_t>(stages_.size()),
                             sizeof(ComponentStepSource*), &temporary) ||
      __builtin_add_overflow(temporary, std::uint64_t{128}, &temporary) || !funder ||
      !funder->Charge(temporary))
    return std::unexpected("execution composition pointers were not funded");
  std::vector<ComponentStepSource*> views;
  views.reserve(stages_.size());
  for (const auto& stage : stages_) views.push_back(stage.get());
  auto sealed = ComponentStepSource::SealExecution(views);
  decltype(views){}.swap(views);
  funder->Release(temporary);
  if (!sealed) return std::unexpected(sealed.error());
  for (auto& stage : stages_) stage->execution_ = *sealed;
  if (*sealed) {
    base::Sha256 hash;
    hash.Update(std::as_bytes(std::span(key_.patterns)));
    hash.Update(std::as_bytes(std::span((*sealed)->identity)));
    key_.patterns = hash.Finish();
  }
  return {};
}

std::expected<std::optional<base::Sha256Digest>, std::string> StagedStepSource::ExecutionIdentity()
    const {
  return stages_.front()->ExecutionIdentity();
}

Status StagedStepSource::CheckExecutionUse(NumericalUse use) const {
  return stages_.front()->CheckExecutionUse(use);
}

Status StagedStepSource::CheckExecutionShape(const planner::StepShape& shape) const {
  if (shape.stage >= stages_.size()) return std::unexpected("execution shape names no stage");
  return stages_[shape.stage]->CheckExecutionShape(shape);
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
