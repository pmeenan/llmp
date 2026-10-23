// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "engine/component_source.h"

#include <algorithm>
#include <array>
#include <format>
#include <utility>
#include <vector>

#include "base/check.h"
#include "engine/graph_plan.h"
#include "format/tensor_format.h"
#include "ir/graph.h"
#include "planner/plan.h"

namespace llmp::engine {

SharedCompilerAllowance::Use::Use(Use&& other) noexcept
    : owner_(std::move(other.owner_)), bytes_(other.bytes_), thread_(other.thread_) {}
SharedCompilerAllowance::Use::~Use() {
  if (owner_) {
    base::Check(thread_ == std::this_thread::get_id(), "compiler charge leaves its driver thread");
    owner_->Release(bytes_);
  }
}
Status SharedCompilerAllowance::Reserve(catalog::Catalog& catalog, catalog::DomainId domain,
                                        std::uint64_t total, std::uint64_t cap,
                                        std::uint64_t available, std::uint64_t margin) {
  // Check the immutable thread identity before touching unsynchronized local
  // usage or catalog state. No worker may race the driver's accounting.
  if (thread_ != std::this_thread::get_id())
    return std::unexpected("compiler admission belongs to its serving driver");
  if (owner_ && (owner_->used() != 0 || !owner_->HeldBy(catalog, domain)))
    return std::unexpected("compiler admission has live use or another catalog owner");
  if (total <= ExecutionMetadataAllowance::OverheadBytes())
    return std::unexpected("compiler admission omits its owner overhead");
  const auto occupied = catalog.OccupancyOf(domain).Total().value();
  if (occupied > cap) return std::unexpected("compiler admission exceeds the node cap");
  if (owner_)
    return owner_->Grow(catalog, domain, total - ExecutionMetadataAllowance::OverheadBytes(), cap,
                        available, margin);
  if (total > cap - occupied) return std::unexpected("compiler admission exceeds the node cap");
  auto owner = ExecutionMetadataAllowance::Admit(catalog, domain, total, available, margin);
  if (!owner) return std::unexpected(owner.error());
  owner_ = std::move(*owner);
  return {};
}
std::expected<SharedCompilerAllowance::Use, std::string> SharedCompilerAllowance::Acquire(
    std::uint64_t bytes) {
  if (thread_ != std::this_thread::get_id())
    return std::unexpected("compiler use belongs to its serving driver");
  if (bytes == 0 || !owner_ || !owner_->Charge(bytes))
    return std::unexpected("compiler construction exceeds its shared payment");
  return Use(owner_, bytes);
}

std::expected<std::unique_ptr<ComponentStepSource>, std::string> ComponentStepSource::Create(
    ComponentSourceOptions options) {
  if (static_cast<bool>(options.compiler) != (options.compiler_bytes != 0))
    return std::unexpected("a compiler source needs both its owner and byte bound");
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
  if ((backend.lanes.count == 0 && backend.lanes.scratch_bytes_per_lane != 0) ||
      backend.lanes.count > planner::kMaxLanes ||
      (backend.lanes.count != 0 &&
       (backend.lanes.scratch_bytes_per_lane == 0 ||
        backend.lanes.scratch_bytes_per_lane % 256 != 0 ||
        backend.lanes.scratch_bytes_per_lane > UINT64_MAX / backend.lanes.count)))
    return std::unexpected("the backend's lane configuration is invalid");
  planner::Policy policy;
  if (options.policy) {
    policy = *options.policy;
  } else if (options.mapping != nullptr) {
    // The mapping's rule for this source's values; none: the planner's
    // default.
    auto text =
        spec::DefaultPolicy(*options.mapping, options.values,
                            {.context = options.bindings.state->context, .slots = options.slots},
                            options.policy_context);
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
  // Facts were authenticated once at admission; source construction scopes
  // them to this actual stage, never a caller's stale stage ordinal.
  policy.active_stage = options.stage;
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
  auto key = planner::KeyOf(source->o_.bound_digest, {}, source->policy_);
  const auto& source_backend =
      source->o_.backend.owner ? source->o_.backend.owner->backend() : source->o_.backend;
  key.registry = source_backend.registry->Digest();
  key.capturable = source->o_.require_capture;
  base::Sha256 facts;
  facts.Update("llmp-plan-backend-v2");
  facts.Update(std::format("lanes:{}:{};", source_backend.lanes.count,
                           source_backend.lanes.scratch_bytes_per_lane));
  facts.Update(std::format("{}:", source_backend.device.device_class.size()));
  facts.Update(source_backend.device.device_class);
  facts.Update(std::format(":{}:{}:{}:{}:", source_backend.device.arch,
                           source_backend.device.features, source_backend.device.multiprocessors,
                           source_backend.device.device_model.size()));
  facts.Update(source_backend.device.device_model);
  for (const auto& name : source_backend.device.preferred) {
    facts.Update(std::format("preferred:{}:", name.size()));
    facts.Update(name);
  }
  for (const auto& tune : source_backend.device.tuned) {
    facts.Update(std::format("{}:", tune.implementation.size()));
    facts.Update(tune.implementation);
    facts.Update(std::format(":{}:{}:{}:{}:{}:{};", tune.rows, tune.k, tune.n,
                             tune.output_encoding ? static_cast<int>(*tune.output_encoding) : -1,
                             tune.weight_bits, tune.all_rows));
  }
  key.backend = facts.Finish();
  source->key_base_ = std::move(key);
  source->semantic_identity_ = source->o_.spec->Digest();
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
    for (const auto change : stage->policy_.numerical_candidates)
      if (!add(sizeof(change) + 128) || !text(planner::NumericalChangeName(change)))
        return std::unexpected("candidate numerical policy bound overflows");
    for (const auto& name : stage->policy_.enabled)
      if (!text(name)) return std::unexpected("execution policy bound overflows");
    for (const auto& name : stage->policy_.disabled)
      if (!text(name)) return std::unexpected("execution policy bound overflows");
    for (const auto& rule : stage->policy_.rules)
      if (!text(rule.pattern) || !add(sizeof(rule) + 128))
        return std::unexpected("execution step rule bound overflows");
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

StageExecutionFacts ComponentStepSource::StageFacts(std::uint16_t stage) const {
  return {.uses_side_rows = UsesSideRows(stage),
          .commit_buffers = [this, stage](std::uint32_t max_rows) {
            return CommitBuffers(stage, max_rows);
          }};
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
  auto key = key_base_;
  if (execution_) {
    base::Sha256 hash;
    hash.Update(std::as_bytes(std::span(key.patterns)));
    hash.Update(std::as_bytes(std::span(execution_->identity)));
    key.patterns = hash.Finish();
  }
  return key;
}

void ComponentStepSource::EmitShapeParameters(const planner::StepShape& shape,
                                              model::ShapeParameterSink& out) const {
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
}

std::expected<SpecializationChange, SpecializationError>
ComponentStepSource::NextSpecializationChange(const SpecializationRequest& request) const {
  if (!ValidSpecializationRequest(request) || request.shape.stage != o_.stage ||
      request.context != o_.bindings.state->context || request.shape.read_align == 0 ||
      request.shape.read_align % 256 != 0 ||
      (o_.slots != 0 && request.shape.segments.size() > o_.slots))
    return std::unexpected(SpecializationError::kRequest);
  const auto& own = request.shape.segments[request.owner];
  const bool frozen = model::ReadsFrozen(*o_.bindings.state);
  const auto read_end = std::uint64_t{own.past} + (frozen ? 0U : own.rows);
  if (own.rows > o_.bindings.state->max_rows || own.read_cells > request.context ||
      own.read_cells < read_end)
    return std::unexpected(SpecializationError::kRequest);
  // Fixed source/capture/layout words cannot change within this immutable
  // source. Every position-dependent component word is observed through the
  // SAME callback used by EmitShapeParameters; unknown annotations fail closed.
  auto sink = model::ShapeParameterSink::Observe(request.after, request.through);
  for (const auto& in : shaped_) {
    if (!in.def->shape_parameter_intervals) return SpecializationChange{};
    in.def->shape_parameters(in, own.rows, own.past, request.shape.read_align, sink);
  }
  const auto observed = sink.Next();
  if (!observed)
    return std::unexpected(observed.error() == model::ShapeParameterSink::Error::kOverflow
                               ? SpecializationError::kOverflow
                               : SpecializationError::kProvider);
  if (!observed->complete) return SpecializationChange{};
  SpecializationChange result{.coverage = SpecializationChange::Coverage::kComplete};
  if (observed->next)
    result.next = SpecializationBoundary{.position = *observed->next, .fact = observed->word + 1};
  // read_cells is already part of the actual plan key. The first owner whose
  // true frontier exceeds its currently padded/common plane changes that key.
  // For a shared common plane, other owners may remain inside it longer; the
  // predictor takes the earliest independently authenticated owner cutoff.
  if (own.read_cells < request.context) {
    const auto offset = frozen ? 0U : own.rows;
    const auto next = std::uint64_t{own.read_cells} - offset + 1;
    if (next > request.after && next <= request.through &&
        (!result.next || next < result.next->position))
      result.next =
          SpecializationBoundary{.position = static_cast<std::uint32_t>(next), .fact = UINT64_MAX};
  }
  return result;
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
  std::optional<SharedCompilerAllowance::Use> compiler;
  if (o_.compiler) {
    auto paid = o_.compiler->Acquire(o_.compiler_bytes);
    if (!paid) return std::unexpected(paid.error());
    compiler.emplace(std::move(*paid));
  }
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
  return BuildWith(shape, addresses, nullptr);
}
std::expected<std::unique_ptr<kernels::StepProgram>, std::string> ComponentStepSource::BuildPrivate(
    const planner::StepShape& shape, const kernels::Addresses& addresses,
    const kernels::ExecutionResources& resources) {
  if (!o_.backend.owner || resources.parent() != o_.backend.owner.get())
    return std::unexpected("private step resources belong to another canonical source backend");
  return BuildWith(shape, addresses, &resources);
}
std::expected<std::unique_ptr<kernels::StepProgram>, std::string> ComponentStepSource::BuildWith(
    const planner::StepShape& shape, const kernels::Addresses& addresses,
    const kernels::ExecutionResources* resources) {
  if (auto checked = CheckExecutionShape(shape); !checked) return std::unexpected(checked.error());
  if (shape.stage != o_.stage) {
    return std::unexpected(
        std::format("a step of stage {} from stage {}'s source", shape.stage, o_.stage));
  }
  std::optional<SharedCompilerAllowance::Use> compiler;
  if (o_.compiler) {
    auto paid = o_.compiler->Acquire(o_.compiler_bytes);
    if (!paid) return std::unexpected(paid.error());
    compiler.emplace(std::move(*paid));
  }
  // The narrowed final layer is the policy's (planner/pattern.h
  // narrow_final).
  auto step = shape.phase == planner::Phase::kCommit
                  ? components::BuildCommit(*o_.spec, o_.stage, shape, o_.bindings)
                  : components::BuildStep(
                        *o_.spec, o_.stage, shape, o_.bindings,
                        {.narrow = !planner::Disabled(policy_, "narrow_final"),
                         .rounded_indexer = !planner::Disabled(policy_, "rounded_indexer"),
                         .rounded_index_mask = !planner::Disabled(policy_, "rounded_index_mask")});
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
  planner::LaneTags lane_tags;
  if (Backend().lanes.count != 0 &&
      (shape.phase == planner::Phase::kDecode || shape.phase == planner::Phase::kVerify ||
       shape.phase == planner::Phase::kDraft) &&
      shape.segments.size() > 1) {
    std::uint32_t region = 0;
    std::uint16_t last_component = components::kFrameworkComponent;
    for (std::uint32_t i = 0; i < step->graph.nodes().size(); ++i) {
      const auto& node = step->graph.nodes()[i];
      if (node.component == components::kFrameworkComponent) continue;
      if (node.batch == ir::Batch::kPerOwner && node.nout > 0) {
        const auto segment = step->graph.tensor(node.out[0]).segment;
        if (segment != ir::kJoinedSegment) {
          if (node.component != last_component) {
            ++region;
            last_component = node.component;
          }
          lane_tags.push_back({ir::NodeId{i}, planner::LaneTag{
                                                  .lane = static_cast<std::uint8_t>(
                                                      (segment % Backend().lanes.count) + 1),
                                                  .region = region,
                                              }});
        }
      }
    }
  }
  const auto state_spans = StateSpans(shape, addresses);
  // The policy's own step rules (its mapping's data: a fast form taken
  // only by small solo steps, a tighter row bound past some rows).
  auto ruled = planner::ForStep(policy_, shape.rows(), shape.segments.size());
  if (!ruled) return std::unexpected(std::move(ruled.error()));
  const planner::Policy step_policy = std::move(*ruled);
  const GraphBuildIO io{step->graph, step->graph.inputs(), step->graph.outputs(), step->weights};
  auto program = BuildComponentProgram(
      io, Backend(), step_policy, addresses,
      {.row_invariant = shape.row_invariant,
       .capturable = o_.require_capture,
       .staged = std::move(staged),
       .lanes = std::move(lane_tags),
       .on_stream = Backend().on_stream,
       .written = Backend().written,
       .within = Backend().within,
       .shape = &shape,
       .state_spans = state_spans,
       .alignment_staged = std::move(aligned_staged),
       .observed_bound_model = o_.bound_digest,
       .observed_model_spec = semantic_identity_},
      kernels::TokenLowerInput{*step, shape, o_.bindings.state->context, state_spans},
      o_.validate_program, o_.backend.owner, resources);
  if (program && compiler) {
    auto held = std::make_shared<SharedCompilerAllowance::Use>(std::move(*compiler));
    if (!(**program).HoldConstruction(std::move(held)))
      return std::unexpected("lowered program already has a construction owner");
  }
  return program;
}

std::expected<std::unique_ptr<StagedStepSource>, std::string> StagedStepSource::Create(
    std::vector<std::unique_ptr<ComponentStepSource>> stages) {
  if (stages.empty()) return std::unexpected("a staged step source needs a stage");
  base::Sha256 patterns, policies, registries, backends;
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
    registries.Update(std::as_bytes(std::span(key.registry)));
    backends.Update(std::as_bytes(std::span(key.backend)));
    backends.Update(key.capturable ? "capturable" : "ordinary");
  }
  planner::PlanKey key = stages.front()->KeyBase();
  key.patterns = patterns.Finish();
  key.policy = policies.Finish();
  key.registry = registries.Finish();
  key.backend = backends.Finish();
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

void StagedStepSource::EmitShapeParameters(const planner::StepShape& shape,
                                           model::ShapeParameterSink& out) const {
  if (shape.stage < stages_.size()) stages_[shape.stage]->EmitShapeParameters(shape, out);
}

std::expected<std::unique_ptr<kernels::StepProgram>, std::string> StagedStepSource::Build(
    const planner::StepShape& shape, const kernels::Addresses& addresses) {
  if (shape.stage >= stages_.size()) {
    return std::unexpected(std::format("a step of stage {}, which the model lacks", shape.stage));
  }
  return stages_[shape.stage]->Build(shape, addresses);
}

StageExecutionFacts StagedStepSource::StageFacts(std::uint16_t stage) const {
  if (stage < stages_.size()) return stages_[stage]->StageFacts(stage);
  return {
      .commit_buffers = [](std::uint32_t) -> std::expected<std::vector<CommitBuffer>, std::string> {
        return std::unexpected("a commit names no stage");
      }};
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
