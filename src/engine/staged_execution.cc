// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "engine/staged_execution.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <format>
#include <functional>
#include <limits>
#include <new>
#include <utility>

#include "base/check.h"
#include "engine/graph_plan.h"
#include "engine/program_lifecycle.h"
#include "engine/support.h"
#include "ir/infer.h"
#include "providers/device_runtime.h"
#include "scheduler/programs.h"

namespace llmp::engine {
namespace {
namespace en = engine;
namespace kn = kernels;
using en::support::Error;
using en::support::Pointer;
using en::support::Round;

std::expected<std::uint64_t, std::string> Bytes(const format::TensorFormat& format) {
  const auto parts = format::PartsOf(format);
  if (!parts || parts->size() != 1 || (*parts)[0].role != format::PartRole::kData ||
      format.layout != format::Layout::kDense)
    return Error("an image port requires a complete dense format");
  return (*parts)[0].bytes;
}
std::expected<std::uint64_t, std::string> Margin(std::uint64_t bytes, std::uint64_t align) {
  std::uint64_t result = 0;
  if (__builtin_add_overflow(bytes, bytes / 4, &result) ||
      result > std::numeric_limits<std::uint64_t>::max() - align)
    return Error("image workspace margin overflows");
  return Round(std::max(result, align), align);
}
void Unique(std::vector<catalog::ExtentId>& values) {
  std::ranges::sort(values);
  values.erase(std::unique(values.begin(), values.end()), values.end());
}
// A need's prefix of a region of the node's shared workspace: the need (in
// whole extents, at least one) within what the node cataloged. The image's
// steps are bound to it and its closures name only it, so a pageable
// workspace brings in only what the running model uses (paged_node.h).
std::uint64_t WorkspacePrefix(std::uint64_t need, const en::Mapped& mapped) {
  return std::min(std::max<std::uint64_t>(need, en::kPagedExtent), mapped.bytes);
}
void AddWorkspacePrefix(std::vector<catalog::ExtentId>& ids, std::uint64_t need,
                        const en::Mapped& mapped) {
  const auto count = std::min<std::size_t>(
      mapped.extents.size(),
      static_cast<std::size_t>(WorkspacePrefix(need, mapped) / en::kPagedExtent));
  ids.insert(ids.end(), mapped.extents.begin(),
             mapped.extents.begin() + static_cast<std::ptrdiff_t>(count));
}
}  // namespace

struct StagedExecution::State {
  struct Key {
    base::Sha256Digest composition, graph;
    execution::BackendDeclaration backend;
    std::uint32_t stage = 0;
    bool operator==(const Key&) const = default;
  };
  // The plan cache's index: the digests' first words and the stage (the
  // backend compared with the whole key).
  struct KeyHash {
    std::size_t operator()(const Key& key) const noexcept {
      std::uint64_t h = key.stage;
      for (const base::Sha256Digest* d : {&key.composition, &key.graph}) {
        std::uint64_t word = 0;
        for (std::size_t i = 0; i < 8; ++i) word |= std::uint64_t{(*d)[i]} << (8U * i);
        h = (h * 0x9e3779b97f4a7c15ULL) ^ word;
      }
      return static_cast<std::size_t>(h);
    }
  };
  // Two runs a plan: a cold run's, and a warm run's (a run right after a
  // run of the same plan, skipping its prologue and invariant inputs).
  using Plans = en::BasicPlanCache<Key, en::StepPlanned, 2, en::StepPlanRuns, KeyHash>;
  struct Model {
    StagedExecutionParts parts;
    en::RunnerMemory memory;
    en::ProgramLifecycle programs;
    std::vector<std::unique_ptr<en::Mapped>> kernel;
    bool prepared = false, registered = false, released = false;
    Model(en::PagedNode& node, int owner, StagedExecutionParts source)
        : parts(std::move(source)),
          memory(node, owner, static_cast<std::uint32_t>(owner)),
          programs(node, owner, parts.graphs) {}
  };
  // One complete model/handle/launch owner precedes geometry-only borrowers.
  std::shared_ptr<Model> model;
  StagedGeometry parts;
  en::RunnerMemory memory;
  en::ProgramLifecycle& programs;
  Plans plans;
  std::vector<std::unique_ptr<en::Mapped>> ports;
  std::vector<std::vector<catalog::ExtentId>> stage_weights;
  std::vector<catalog::Closure> closures;
  catalog::Closure all, fence;
  execution::BackendDeclaration declaration;
  en::ProgramEnvelope measured;
  kn::PlanCoverage plan_coverage;
  std::vector<en::ProgramEnvelope> stage_envelopes;
  std::vector<std::uint64_t> stage_capture_nodes;
  // Each stage's graph digest, its plans' key: the geometry's typed graphs
  // are immutable for this state's life, hashed once when it is prepared.
  std::vector<base::Sha256Digest> stage_digests;
  en::GraphStats stats;
  std::byte* input = nullptr;
  std::byte* output = nullptr;
  std::uint64_t input_bytes = 0, output_bytes = 0, activation_bytes = 0, pool_bytes = 0;
  std::uint64_t plan_floor = 0;
  std::size_t next = 0;
  bool setup = false, registered = false, bound = false, released = false;
  // Work this state queued whose end is unproven (unproven_work.h).
  UnprovenMark unproven;
  bool generation_complete = false;
  // A pending job's source, captures, plan and storage survive a failed wait:
  // the job's own retirement releases it, or a fence of the stream does
  // (AcknowledgeFence, RetireAfterFence).
  RetainedFlight<State> flight;
  // A refused geometry replacement's candidates, kept with this state until
  // a fence lets their retirement be retried; `retiring`: this state's own
  // retirement for an installed candidate failed part way.
  std::vector<std::shared_ptr<State>> transition_holds;
  bool retiring = false;

  State(en::PagedNode& node, int owner, StagedExecutionParts source)
      : model(std::make_shared<Model>(node, owner, std::move(source))),
        parts{.keep = model->parts.geometry_keep ? std::move(model->parts.geometry_keep)
                                                 : model->parts.keep,
              .identity = model->parts.identity,
              .stages = std::move(model->parts.stages),
              .request = std::move(model->parts.request)},
        memory(node, owner, static_cast<std::uint32_t>(owner)),
        programs(model->programs) {}
  State(std::shared_ptr<Model> retained, StagedGeometry geometry)
      : model(std::move(retained)),
        parts(std::move(geometry)),
        memory(model->memory.node(), model->memory.owner(), model->memory.stream()),
        programs(model->programs) {}

  kn::Addresses Addresses(std::size_t stage, bool measuring) const {
    auto result = parts.stages.at(stage).addresses;
    result.activations = measuring ? 0 : memory.node().activations().base;
    result.activation_bytes =
        measuring ? 0 : WorkspacePrefix(activation_bytes, memory.node().activations());
    result.activation_alignment = en::kPagedExtent;
    return result;
  }
  std::expected<std::unique_ptr<kn::StepProgram>, std::string> Build(std::size_t stage,
                                                                     bool measuring) const {
    ir::InferenceScope inference(parts.stages.at(stage).compiler_memo_bytes);
    const auto& graph = *parts.stages.at(stage).graph;
    const en::GraphBuildIO io{graph.graph(), graph.graph().inputs(), graph.graph().outputs(),
                              graph.weights()};
    const auto& backend = model->parts.backend->owner->backend();
    auto policy = model->parts.policy;
    policy.numerical = parts.stages.at(stage).numerical;
    policy.active_stage = static_cast<std::uint16_t>(stage);
    return en::BuildComponentProgram(
        io, backend, policy, Addresses(stage, measuring),
        {.staged = std::vector(io.inputs.begin(), io.inputs.end()),
         .written = backend.written,
         .within = backend.within,
         .alignment_staged = std::vector(io.inputs.begin(), io.inputs.end()),
         .invariant = Invariant(stage)},
        std::nullopt, {}, model->parts.backend->owner);
  }
  // The stage's inputs its source keeps the same within a generation.
  std::vector<ir::TensorId> Invariant(std::size_t stage) const {
    const auto& descriptor = parts.stages.at(stage);
    const auto inputs = descriptor.graph->graph().inputs();
    std::vector<ir::TensorId> out;
    for (const std::uint32_t i : descriptor.invariant)
      if (i < inputs.size()) out.push_back(inputs[i]);
    return out;
  }
  Key KeyOf(std::size_t stage) const {
    return {parts.identity, stage_digests[stage], declaration, static_cast<std::uint32_t>(stage)};
  }
  // A stage's plan: the cached one, or built, bound and added.
  std::expected<Plans::Entry*, std::string> Plan(std::size_t stage,
                                                 const kn::PlanRecordSink& report_sink) {
    const Key key = KeyOf(stage);
    if (auto* entry = plans.Find(key)) return entry;
    auto program = Build(stage, false);
    if (!program) return Error(program.error());
    auto planned = std::make_unique<en::StepPlanned>();
    planned->program = std::move(*program);
    auto bind = programs.BindPrepared(*planned, [&](const kn::StepProgram& actual) -> Status {
      auto envelope = en::ProgramEnvelope::Measure(actual, true);
      const auto& paid = stage_envelopes[stage];
      if (!envelope || envelope->scratch > paid.scratch || envelope->pool > paid.pool ||
          envelope->host > paid.host || envelope->launches > paid.launches ||
          envelope->inputs > paid.inputs || envelope->outputs > paid.outputs)
        return Error("bound image program exceeds its measured envelope");
      return Ports(stage, actual);
    });
    if (!bind) return Error(bind.error());
    const auto bytes = planned->program->host_bytes();
    auto* entry = &plans.Add(key, std::move(planned), bytes, stage_capture_nodes[stage], 0);
    entry->planned->program->ConstructionPaid();
    for (auto& runs : entry->runs) runs.uncapturable = entry->planned->positioned;
    EmitPlanReport(*entry, report_sink);
    return entry;
  }
  void ObservePlanCoverage(const kn::StepProgram& program) {
    kn::ObservePlanCoverage(plan_coverage, program);
  }
  static void EmitPlanReport(const Plans::Entry& entry, const kn::PlanRecordSink& sink) {
    if (!sink) return;
    std::array<char, 512> header{};
    const auto written = std::format_to_n(
        header.data(), header.size(),
        R"({{"kind":"installed_plan","pipeline":"image","id":{},"stage":{},"host_bytes":{}}})",
        entry.serial, entry.key.stage, entry.host_bytes);
    base::Check(static_cast<std::size_t>(written.size) <= header.size(),
                "bounded installed image-plan report header");
    sink(kn::PlanReportEvent::kBegin, {});
    sink(kn::PlanReportEvent::kFragment,
         std::string_view(header.data(), static_cast<std::size_t>(written.size)));
    sink(kn::PlanReportEvent::kEnd, {});
    // Explicit record boundaries let the logger frame streamed JSON fragments.
    entry.planned->program->ReportPlanRecords(sink);
  }
  // The staging of a stage's host inputs, in slot order at 256-byte
  // boundaries: a warm run's invariant inputs keep their place but are not
  // copied. `fill` (if set) writes each copied input's bytes.
  std::expected<en::Copies, std::string> Inputs(
      std::size_t stage, const kn::StepProgram& program, bool warm,
      const std::function<Status(std::size_t, std::span<std::byte>)>& fill) const {
    const auto& descriptor = parts.stages[stage];
    en::Copies copies;
    std::uint64_t cursor = 0;
    for (std::size_t i = 0; i < program.inputs().size(); ++i) {
      const auto& slot = program.inputs()[i];
      if (descriptor.inputs[i]) continue;
      if (!(warm && slot.invariant)) {
        if (cursor > input_bytes || slot.bytes > input_bytes - cursor)
          return Error("image host input has no paid source");
        if (fill) {
          if (auto filled = fill(i, {input + cursor, static_cast<std::size_t>(slot.bytes)});
              !filled)
            return Error(filled.error());
        }
        copies.push_back({slot.device_address, slot.bytes, cursor});
      }
      cursor += Round(slot.bytes, 256);
    }
    return copies;
  }
  // The device ports a stage reads, copied in after its host inputs; a
  // warm run's invariant ones are in place already.
  bool PortsIn(std::size_t stage, const kn::StepProgram& program, providers::NativeStream stream,
               bool warm) const {
    const auto& descriptor = parts.stages[stage];
    for (std::size_t i = 0; i < descriptor.inputs.size(); ++i) {
      if (!descriptor.inputs[i] || (warm && program.inputs()[i].invariant)) continue;
      if (!providers::CopyAsync(stream, Pointer(program.inputs()[i].device_address),
                                Pointer(ports[*descriptor.inputs[i]]->base),
                                program.inputs()[i].bytes, providers::CopyKind::kDeviceToDevice)
               .ok())
        return false;
    }
    return true;
  }
  // The observation every run of a stage copies out (part of its recording).
  en::Copies Observed(std::size_t stage, const kn::StepProgram& program) const {
    const auto& observed = program.outputs()[parts.stages[stage].observation];
    return {{en::support::Address(output), observed.device_address, observed.bytes}};
  }
  std::expected<std::uint64_t, std::string> Request(const StagedRequest& request) const {
    if (!request.keep || request.layout == base::Sha256Digest{} ||
        request.identity == base::Sha256Digest{} || request.sequence.empty() ||
        request.first_output >= request.sequence.size() - 1)
      return Error("image request needs a paid owner, identities and complete sequence");
    std::vector<bool> initialized;
    std::uint64_t initial_bytes = 0;
    for (const auto& port : request.ports) {
      const auto bytes = Bytes(port.format);
      if (!bytes || (!port.initial.empty() && port.initial.size() != *bytes))
        return Error("image initial port encoding or length differs");
      initialized.push_back(!port.initial.empty());
      if (!port.initial.empty() &&
          (*bytes > UINT64_MAX - 255 ||
           __builtin_add_overflow(initial_bytes, Round(*bytes, 256), &initial_bytes)))
        return Error("image initial port staging overflows");
    }
    for (const auto& step : request.sequence) {
      if (step.stage >= parts.stages.size()) return Error("image invocation names no stage");
      const auto& stage = parts.stages[step.stage];
      for (const auto& input_port : stage.inputs) {
        if (!input_port && !request.fill)
          return Error("image request lacks a required input callback");
        if (input_port && (*input_port >= initialized.size() || !initialized[*input_port]))
          return Error("image sequence reads an unwritten port");
      }
      for (const auto& output_port : stage.outputs)
        if (output_port) {
          if (*output_port >= initialized.size())
            return Error("image sequence names no output port");
          initialized[*output_port] = true;
        }
    }
    return initial_bytes;
  }
  Status Ports(std::size_t stage, const kn::StepProgram& program) const {
    const auto& descriptor = parts.stages.at(stage);
    const auto& graph = descriptor.graph->graph();
    if (program.routed() || program.inputs().size() != graph.inputs().size() ||
        program.outputs().size() != graph.outputs().size())
      return Error("image program typed port count or routed contract differs");
    const auto check = [&](const auto& slots, auto ids, const auto& bindings) -> Status {
      for (std::size_t i = 0; i < ids.size(); ++i) {
        const auto& tensor = graph.tensor(ids[i]);
        const auto bytes = Bytes(tensor.format);
        if (!bytes || slots[i].tensor != ids[i] || slots[i].bytes != *bytes ||
            slots[i].segment != tensor.segment || !ir::IsContiguous(tensor))
          return Error("image program changed an original typed graph port");
        if (bindings[i]) {
          const auto& port = parts.request.ports.at(*bindings[i]);
          if (port.format != tensor.format || port.owner != tensor.segment)
            return Error("image device handoff format, geometry or owner differs");
        }
      }
      return {};
    };
    if (auto inputs = check(program.inputs(), graph.inputs(), descriptor.inputs); !inputs)
      return inputs;
    return check(program.outputs(), graph.outputs(), descriptor.outputs);
  }
};

struct StagedExecution::Lifetime final : kernels::ResourceDependency {
  Lifetime(HostPayment paid, PagedNode& node, int owner)
      : payment(std::move(paid)), execution(node, owner) {
    execution.lifetime_ = this;
    registered.model = &execution;
  }
  kernels::ResourceKeep Publish() { return Adopt(); }
  void Destroy() override {
    auto paid = std::move(payment);
    delete this;
    paid.Reset();
  }
  HostPayment payment;
  StagedExecution execution;
  RetainedModelRoot registered;
};
std::expected<StagedExecutionOwner, std::string> StagedExecution::Create(PagedNode& node,
                                                                         int owner) {
  if (owner < 0) return Error("staged execution requires a model owner");
  auto paid = node.PrepayHost(sizeof(Lifetime) + 64);
  if (!paid) return Error(paid.error());
  auto* root = new (std::nothrow) Lifetime(std::move(*paid), node, owner);
  if (!root) return Error("allocating paid staged execution owner failed");
  root->registered.keep = root->Keep();
  node.KeepModelRoot(root->registered);
  return StagedExecutionOwner(root->Publish(), &root->execution);
}
void StagedExecution::RetireRoot() {
  if (!lifetime_) return;
  auto* root = std::exchange(lifetime_, nullptr);
  node_.RetireModelRoot(root->registered);
}
Status StagedExecution::RetireAfterFence(const FenceProof& proof) {
  drained_ = true;
  if (state_) {
    state_->unproven.Clear(proof);
    state_->flight.Clear(proof);
    if (state_->programs.launch()) state_->programs.launch()->Retire();
  }
  for (auto& pending : transitions_) {
    pending->unproven.Clear(proof);
    pending->flight.Clear(proof);
  }
  return {};
}

Status StagedExecution::AcknowledgeFence(const FenceProof& proof) {
  if (!state_ || state_->released) return {};
  auto& s = *state_;
  // The fence proved the stopped stage's job (and every earlier one) ended:
  // what kept the state for it goes, and the generation starts over.
  s.unproven.Clear(proof);
  s.flight.Clear(proof);
  if (s.retiring && !transitions_.empty()) {
    // A replacement whose old geometry stopped retiring part way: resumed,
    // so the candidate it had validated, paid and bound takes its place.
    auto previous = state_;
    auto candidate = transitions_.back();
    if (auto installed = InstallGeometry(previous, candidate); !installed) return installed;
    return {};
  }
  // A refused replacement's candidates: their retirement retried.
  for (auto& pending : transitions_) {
    if (pending->released) continue;
    pending->plans.Clear();
    if (auto retired = pending->memory.Retire(); !retired) {
      s.unproven.Mark();
      s.flight.Retain(state_);
      return Error(
          std::format("an image geometry candidate still cannot retire: {}", retired.error()));
    }
    pending->released = true;
  }
  transitions_.clear();
  s.transition_holds.clear();
  s.next = 0;
  s.generation_complete = false;
  DropPlans();
  return {};
}

bool StagedExecution::unproven() const noexcept { return state_ && state_->unproven.active(); }

StagedExecution::StagedExecution(en::PagedNode& node, int owner) : node_(node), index_(owner) {}
std::uint64_t StagedExecution::ControlMetadataBytes() {
  return sizeof(State) + sizeof(State::Model) + sizeof(StagedExecution);
}
StagedExecution::~StagedExecution() = default;

Status StagedExecution::Setup(StagedExecutionParts parts) {
  if (state_ || drained_ || index_ < 0) return Error("staged setup order or source differs");
  // Reject incomplete source owners before publishing any cleanup borrower.
  for (const auto& source : parts.weights)
    if (!source.weights) return Error("image source weights are missing");
  // All source construction was paid by the factory before these values exist.
  state_ = std::make_shared<State>(node_, index_, std::move(parts));
  auto& s = *state_;
  if (!s.parts.keep || !s.model->parts.backend || !s.model->parts.backend->owner ||
      !s.model->parts.launch || s.parts.stages.empty() || s.parts.request.sequence.empty() ||
      s.parts.request.first_output >= s.parts.request.sequence.size())
    return Error("image needs an owned typed composition and complete sequence");
  for (auto& weights : s.model->parts.weights) {
    if (auto reserved = weights.weights->Reserve(node_, weights.places, weights.slabs); !reserved)
      return reserved;
  }
  std::vector<std::uint64_t> bases;
  for (const auto& allocation : s.model->parts.kernel_memory) {
    auto& mapped = s.model->kernel.emplace_back(std::make_unique<en::Mapped>());
    if (auto r = s.model->memory.Map(*mapped, allocation.name, allocation.bytes,
                                     catalog::MemoryClass::kRuntime);
        !r)
      return r;
    bases.push_back(mapped->base);
  }
  if (s.model->parts.prepare) {
    s.model->prepared = true;
    if (auto prepared = s.model->parts.prepare(bases); !prepared) return prepared;
  }
  auto declaration = s.model->parts.backend->owner->Ready();
  if (!declaration) return Error(declaration.error());
  s.declaration = *declaration;
  return PrepareState(s);
}

Status StagedExecution::PrepareState(State& s) {
  if (!s.parts.keep || s.parts.identity == base::Sha256Digest{} || s.parts.stages.empty())
    return Error("image geometry needs a paid graph owner and identity");
  auto declaration = s.model->parts.backend->owner->Ready();
  if (!declaration) return Error(declaration.error());
  s.declaration = *declaration;
  for (const auto& stage : s.parts.stages) {
    if (!stage.graph || stage.weights >= s.model->parts.weights.size() ||
        stage.inputs.size() != stage.graph->graph().inputs().size() ||
        stage.outputs.size() != stage.graph->graph().outputs().size() ||
        stage.observation >= stage.outputs.size())
      return Error("image stage source or typed port inventory differs");
    for (const auto* bindings : {&stage.inputs, &stage.outputs})
      for (const auto& port : *bindings)
        if (port && *port >= s.parts.request.ports.size())
          return Error("image stage names no device port");
  }
  s.stage_digests.clear();
  s.stage_digests.reserve(s.parts.stages.size());
  for (const auto& stage : s.parts.stages) s.stage_digests.push_back(stage.graph->graph().Digest());
  auto initial_bytes = s.Request(s.parts.request);
  if (!initial_bytes) return Error(initial_bytes.error());
  for (const auto& port : s.parts.request.ports) {
    const auto bytes = Bytes(port.format);
    if (!bytes) return Error(bytes.error());
    auto& mapped = s.ports.emplace_back(std::make_unique<en::Mapped>());
    if (auto r = s.memory.Map(*mapped, "image typed carry", *bytes, catalog::MemoryClass::kRuntime);
        !r)
      return r;
  }
  // Measure each actual stage once. Exact consumed source intervals, not all
  // resources merely sharing its artifact, determine its residency closure.
  for (std::size_t stage = 0; stage < s.parts.stages.size(); ++stage) {
    auto program = s.Build(stage, true);
    if (!program) return Error(std::format("measuring image stage {}: {}", stage, program.error()));
    if (auto ports = s.Ports(stage, **program); !ports) return ports;
    auto envelope = en::ProgramEnvelope::Measure(**program, true);
    if (!envelope) return Error(envelope.error());
    s.ObservePlanCoverage(**program);
    s.measured.Include(*envelope);
    s.stage_envelopes.push_back(*envelope);
    // One H2D or D2D per input and a fixed observation D2H are captured.
    // Device output handoffs occur after the recording, but their launch
    // descriptors are conservatively included in the same prepaid inventory.
    std::uint64_t nodes = envelope->launches, capture_bytes = 0, retained = envelope->host;
    if (__builtin_add_overflow(nodes, (*program)->inputs().size(), &nodes) ||
        __builtin_add_overflow(nodes, (*program)->outputs().size() + 1ULL, &nodes) ||
        __builtin_mul_overflow(nodes, en::kGraphNodeHostBytes, &capture_bytes) ||
        (s.model->parts.graphs && __builtin_add_overflow(retained, capture_bytes, &retained)))
      return Error("image capture inventory overflows");
    s.stage_capture_nodes.push_back(nodes);
    if (__builtin_add_overflow(s.plan_floor, retained, &s.plan_floor))
      return Error("retained image program inventory overflows");
    auto& selected = s.stage_weights.emplace_back();
    const auto& weights = *s.model->parts.weights[s.parts.stages[stage].weights].weights;
    std::vector<catalog::ExtentId> scratch(weights.extents().size());
    for (const auto& use : (*program)->tensors()) {
      if (use.role != ir::Role::kWeight) continue;
      if (use.routed_experts) return Error("stateless image stage has a routed weight");
      auto count = weights.ExtentsTouching(use.address, use.bytes, scratch);
      if (!count) return Error(count.error());
      selected.insert(selected.end(), scratch.begin(),
                      scratch.begin() + static_cast<std::ptrdiff_t>(*count));
    }
    Unique(selected);
  }
  auto activations = Margin(s.measured.scratch, en::kPagedExtent);
  auto pool = s.measured.pool == 0 ? std::expected<std::uint64_t, std::string>(0)
                                   : Margin(s.measured.pool, en::kPagedExtent);
  auto inputs = Margin(std::max(s.measured.inputs, *initial_bytes), 256),
       outputs = Margin(s.measured.outputs, 256);
  if (!activations || !pool || !inputs || !outputs)
    return Error("image measured envelope overflows");
  s.activation_bytes = *activations;
  s.pool_bytes = *pool;
  s.input_bytes = *inputs;
  s.output_bytes = *outputs;
  auto input = s.memory.Pinned(s.input_bytes), output = s.memory.Pinned(s.output_bytes);
  if (!input || !output) return Error(!input ? input.error() : output.error());
  s.input = static_cast<std::byte*>(*input);
  s.output = static_cast<std::byte*>(*output);
  s.setup = true;
  return {};
}

void StagedExecution::SetLog(std::function<void(std::string_view)> log) {
  if (state_) state_->model->parts.log = std::move(log);
}

Status StagedExecution::Register() {
  if (!state_ || !state_->setup || state_->registered) return Error("image register order");
  for (auto& weights : state_->model->parts.weights)
    if (auto r = weights.weights->Register(node_, index_); !r) return r;
  // D-090: every registered source place remains stable for this model's
  // lifetime, including source extents unused by the current geometry.
  if (auto pinned = node_.scheduler().PinPlaces(managed_extents()); !pinned)
    return Error(
        std::format("pinning the image's places: {}", scheduler::ToString(pinned.error())));
  state_->registered = true;
  state_->model->registered = true;
  return {};
}
Status StagedExecution::Bind() {
  if (!state_ || !state_->registered || state_->bound) return Error("image bind order");
  auto& s = *state_;
  if (auto bound = BindState(s); !bound) return bound;
  s.programs.runs().SetStaging(s.input, s.input_bytes);
  return {};
}
Status StagedExecution::BindState(State& s) {
  if (s.activation_bytes > node_.activations().bytes || s.pool_bytes > node_.pool().bytes)
    return Error("image geometry exceeds backed workspace");
  if (!s.programs.launch()) {
    auto launch = s.model->parts.launch();
    if (!launch) return Error(launch.error());
    if (auto r = s.programs.BindLaunch(std::move(*launch)); !r) return r;
  }
  s.plans.set_account(&s.programs.account());
  const auto closures = [&]() -> Status {
    auto common = s.memory.extents();
    const auto model_extents = s.model->memory.extents();
    common.insert(common.end(), model_extents.begin(), model_extents.end());
    AddWorkspacePrefix(common, s.activation_bytes, node_.activations());
    AddWorkspacePrefix(common, s.pool_bytes, node_.pool());
    Unique(common);
    auto fence = node_.catalog().ClosureOfExtents(common);
    if (!fence) return Error("image common closure is incomplete");
    s.fence = *fence;
    auto all = common;
    for (const auto& weights : s.stage_weights) {
      auto ids = common;
      ids.insert(ids.end(), weights.begin(), weights.end());
      Unique(ids);
      auto closure = node_.catalog().ClosureOfExtents(ids);
      if (!closure) return Error("image stage closure is incomplete");
      s.closures.push_back(std::move(*closure));
      all.insert(all.end(), weights.begin(), weights.end());
    }
    Unique(all);
    auto complete = node_.catalog().ClosureOfExtents(all);
    if (!complete) return Error("image complete closure is incomplete");
    s.all = std::move(*complete);
    return {};
  };
  // Before Start catalog access is local; afterwards only its owner reads it.
  if (node_.budget().value() != 0) {
    if (auto read = node_.Call(closures, "binding image geometry closures"); !read) return read;
  } else if (auto read = closures(); !read)
    return read;
  s.bound = true;
  return {};
}

Status StagedExecution::RunThrough(std::size_t end, std::string& sha) {
  if (!state_ || !state_->bound || state_->released || state_->unproven.active() ||
      end > state_->parts.request.sequence.size() || end <= state_->next)
    return Error("image execution order or completion is invalid");
  auto hold = state_;
  auto& s = *hold;
  if (s.activation_bytes > node_.activations().bytes || s.pool_bytes > node_.pool().bytes)
    return Error("image execution exceeds the backed workspace prefix");
  // A request this call opened ends with it; one its caller holds returns
  // to the common fence. Between stages of this call the lease moves
  // straight to the next stage's closure (unchanged between flow steps)
  // rather than ending or dropping to the fence and growing back each step.
  bool began = false, settled = false;
  const auto settle = [&]() -> Status {
    settled = true;
    return began ? node_.EndRequest(stream()) : node_.RefreshRequest(stream(), s.fence);
  };
  // Any other return (a refusal before a stage's work is queued) settles the
  // lease too, and returns an unfinished generation to its start, so it can
  // run again or a later preparation or geometry change is not refused.
  // Unknown work keeps its lease and position (s.unproven, s.flight).
  struct Settle {
    const std::function<void()> run;
    ~Settle() { run(); }
  } settle_on_exit{[&] {
    if (s.unproven.active()) return;
    if (!settled && node_.InRequest(stream()) && !settle()) {
      s.unproven.Mark();
      return;
    }
    if (s.next < end && !s.generation_complete) s.next = 0;
  }};
  // The plan this call ran last: the next run of it is warm (its prologue
  // and invariant inputs left in place). The workspace keeps no contents
  // from one job to the next (paged_node.h MapWorkspace) but here: between
  // this call's consecutive jobs, under the lease the call holds throughout
  // (the request moves from one stage's closure to the next, never ended),
  // nothing evicts the workspace and no other model runs on it. A new call
  // starts cold.
  std::uint64_t last_run = 0;  // the plan's serial (Entry::serial), 0 for none
  for (; s.next < end; ++s.next) {
    const en::PlanStep protect;
    const auto invocation = s.parts.request.sequence[s.next];
    const auto& stage = s.parts.stages[invocation.stage];
    auto ready = s.model->parts.backend->owner->Ready();
    if (!ready || *ready != s.declaration) return Error("image backend identity changed");
    // Catalog content identities are refreshed between completed stages.
    auto ids = s.memory.extents();
    const auto model_extents = s.model->memory.extents();
    ids.insert(ids.end(), model_extents.begin(), model_extents.end());
    AddWorkspacePrefix(ids, s.activation_bytes, node_.activations());
    AddWorkspacePrefix(ids, s.pool_bytes, node_.pool());
    const auto& selected = s.stage_weights[invocation.stage];
    ids.insert(ids.end(), selected.begin(), selected.end());
    Unique(ids);
    auto closure = node_.catalog().ClosureOfExtents(ids);
    if (!closure) return Error("image current stage closure is absent");
    const bool opened = !node_.InRequest(stream());
    // A lease opened here starts the workspace afresh: nothing runs warm.
    if (opened) last_run = 0;
    auto admitted = opened ? node_.BeginRequest(stream(), *closure, "image stage")
                           : node_.RefreshRequest(stream(), *closure);
    began = began || (opened && admitted);
    if (!admitted) {
      // No lease and nothing of this stage queued (a refused page-in, room
      // the closure could not have): the request's task has ended and
      // every earlier stage's job completed, so the generation is known and
      // goes back to its start (settle_on_exit). Only a request this call
      // opened says so; a refreshed one (the caller's) or a lease still
      // held past a failure leaves it unknown.
      if (opened && !node_.InRequest(stream())) return admitted;
      s.unproven.Mark();
      s.flight.Retain(hold);
      return admitted;
    }
    if (s.next == 0) {
      en::Copies initial;
      std::uint64_t at = 0;
      for (std::size_t i = 0; i < s.parts.request.ports.size(); ++i) {
        const auto& bytes = s.parts.request.ports[i].initial;
        if (bytes.empty()) continue;
        std::memcpy(s.input + at, bytes.data(), bytes.size());
        initial.push_back({s.ports[i]->base, bytes.size(), at});
        at += Round(bytes.size(), 256);
      }
      s.flight.Retain(hold);
      const auto uploaded = node_.Job(
          *closure,
          [hold,
           initial = std::move(initial)](providers::NativeStream native) -> scheduler::JobResult {
            for (const auto& copy : initial)
              if (!providers::CopyAsync(native, Pointer(copy[0]), hold->input + copy[2], copy[1],
                                        providers::CopyKind::kHostToDevice)
                       .ok())
                return scheduler::JobResult::kUnknown;
            return scheduler::JobResult::kQueued;
          },
          "image initial ports", stream());
      if (!uploaded) {
        s.unproven.Mark();
        return uploaded;
      }
      s.flight.ReleaseRetired();
    }
    auto planned = s.Plan(invocation.stage, plan_report_sink_);
    Status checked = planned ? Status{} : Error(planned.error());
    if (!checked) {
      auto ended = settle();
      if (!ended) {
        s.unproven.Mark();
        s.flight.Retain(hold);
      }
      return checked;
    }
    auto* entry = *planned;
    auto& program = *entry->planned->program;
    // Eager programs still receive the common between-jobs positioning hook.
    // A stateless typed stage has no token positions; refusal precedes Run.
    if (entry->planned->positioned) {
      if (auto positioned = program.Position({}); !positioned)
        checked = Error(positioned.error().detail);
    }
    const bool warm = program.warm() && last_run != 0 && last_run == entry->serial;
    const std::size_t variant = warm ? 1 : 0;
    en::Copies inputs;
    if (checked) {
      auto staged = s.Inputs(invocation.stage, program, warm,
                             [&](std::size_t slot, std::span<std::byte> out) -> Status {
                               if (!s.parts.request.fill)
                                 return Error("image host input has no paid source");
                               return s.parts.request.fill(invocation, slot, out);
                             });
      if (staged)
        inputs = std::move(*staged);
      else
        checked = Error(staged.error());
    }
    if (!checked) {
      auto ended = settle();
      if (!ended) {
        s.unproven.Mark();
        s.flight.Retain(hold);
      }
      return checked;
    }
    const bool observe = s.next + 1 == end;
    const auto& observed = program.outputs()[stage.observation];
    // Fixed observation addresses are part of the captured plan contract.
    // Always copy this stage output; only requested boundaries publish it.
    en::Copies outputs = s.Observed(invocation.stage, program);
    const bool capture = entry->runs[variant].CaptureDue(s.model->parts.graphs) &&
                         s.plans.ChargeGraph(*entry, variant);
    s.flight.Retain(hold);
    auto queued = std::make_shared<en::StepQueued>();
    auto job = [hold, entry, invocation, inputs = std::move(inputs), outputs = std::move(outputs),
                capture, queued, warm,
                variant](providers::NativeStream native) -> scheduler::JobResult {
      auto& live = *hold;
      const auto& descriptor = live.parts.stages[invocation.stage];
      const auto& actual = *entry->planned->program;
      const auto between = [&](providers::NativeStream stream) {
        return live.PortsIn(invocation.stage, actual, stream, warm);
      };
      *queued = live.programs.runs().Queue(entry->runs[variant], inputs, between, actual, outputs,
                                           capture, live.stats, native, warm);
      if (!queued->result) return scheduler::JobResult::kUnknown;
      // Device handoff follows the actual graph launch, in this same leased
      // job. It is not a DeviceToHost RunCopy and is never replayed separately.
      for (std::size_t i = 0; i < descriptor.outputs.size(); ++i) {
        if (!descriptor.outputs[i]) continue;
        if (!providers::CopyAsync(native, Pointer(live.ports[*descriptor.outputs[i]]->base),
                                  Pointer(actual.outputs()[i].device_address),
                                  actual.outputs()[i].bytes, providers::CopyKind::kDeviceToDevice)
                 .ok()) {
          queued->result =
              std::unexpected(kn::StepFailure{kn::StepError::kUnknown, "image port handoff"});
          return scheduler::JobResult::kUnknown;
        }
      }
      return scheduler::JobResult::kQueued;
    };
    const auto ran = node_.Job(*closure, std::move(job), "image typed program", stream());
    if (!ran || !queued->result || s.programs.launch()->faulted()) {
      s.unproven.Mark();
      return Error(!ran              ? ran.error()
                   : !queued->result ? queued->result.error().detail
                                     : "image launch faulted");
    }
    // The last stage of this call settles the request; an earlier one
    // leaves it for the next stage's refresh.
    auto ended = s.next + 1 < end ? Status{} : settle();
    if (!ended) {
      s.unproven.Mark();
      return ended;
    }
    s.flight.ReleaseRetired();
    en::Count(s.stats, queued->path);
    last_run = entry->serial;
    if (observe) {
      const std::span<const std::byte> bytes(s.output, static_cast<std::size_t>(observed.bytes));
      if (end == s.parts.request.sequence.size() && s.parts.request.finish) {
        auto converted = s.parts.request.finish(bytes);
        if (!converted) {
          // Device work and handoff have completed. A rejected host conversion
          // or optional file observer must not strand an otherwise retired
          // request or report its output as successful.
          ++s.next;
          s.generation_complete = true;
          return Error(converted.error());
        }
        sha = std::move(*converted);
      } else {
        base::Sha256 hash;
        hash.Update(bytes);
        sha = base::ToHex(hash.Finish());
      }
    }
  }
  return {};
}
// After a load: each stage's plan built and bound and, with graphs on, each
// run of the sequence (a stage's cold run, and its warm one when the sequence
// runs it twice in a row) recorded ahead. The first generation then replays
// what it would otherwise plan, load (the libraries a recording's first
// launches load) and run launch by launch. Plans and recordings outlive a
// swap out, so only a model's first load in a process pays for them.
Status StagedExecution::AfterLoad() {
  // A pageable workspace is the active model's: recording while another
  // model is active would lease (and page in) this one's prefix of it.
  if (!state_ || !state_->bound || state_->released || state_->unproven.active() ||
      state_->flight.held() || node_.InRequest(stream()) ||
      (node_.workspace_pageable() && !node_.active(index_)))
    return {};
  // Every plan made or found here is this step's until its recordings are
  // queued: nothing the charges below reclaim may be one of them, nor have
  // its graph's charge settled while its capture is pending.
  const en::PlanStep protect;
  auto hold = state_;
  auto& s = *hold;
  struct Ahead {
    std::size_t stage = 0;
    State::Plans::Entry* entry = nullptr;
    bool warm = false;
    en::Copies inputs, outputs;
  };
  std::vector<Ahead> ahead;
  const auto& sequence = s.parts.request.sequence;
  for (std::size_t i = 0; i < sequence.size(); ++i) {
    const std::size_t stage = sequence[i].stage;
    // A plan this cannot make now is the first request's to make (or
    // refuse): only a recording's fault fails the load.
    auto planned = s.Plan(stage, plan_report_sink_);
    if (!planned) continue;
    auto* entry = *planned;
    const auto& program = *entry->planned->program;
    const bool warm = program.warm() && i > 0 && sequence[i - 1].stage == stage;
    const std::size_t variant = warm ? 1 : 0;
    if (!s.model->parts.graphs || entry->planned->positioned || entry->runs[variant].has_graph() ||
        entry->runs[variant].uncapturable || entry->runs[variant].ahead_refused ||
        std::ranges::any_of(ahead,
                            [&](const Ahead& a) { return a.entry == entry && a.warm == warm; }))
      continue;
    auto inputs = s.Inputs(stage, program, warm, {});
    if (!inputs || !s.plans.ChargeGraph(*entry, variant)) continue;
    ahead.push_back({stage, entry, warm, std::move(*inputs), s.Observed(stage, program)});
  }
  if (ahead.empty()) return {};
  s.flight.Retain(hold);
  // Shared with the job: a failed wait may leave it to run later.
  auto shared = std::make_shared<std::vector<Ahead>>(std::move(ahead));
  auto failure = std::make_shared<std::string>();
  const auto ran = node_.Job(
      s.fence,
      [hold, shared, failure](providers::NativeStream native) -> scheduler::JobResult {
        auto& live = *hold;
        for (auto& a : *shared) {
          const auto& program = *a.entry->planned->program;
          auto& runs = a.entry->runs[a.warm ? 1 : 0];
          const auto between = [&](providers::NativeStream stream) {
            return live.PortsIn(a.stage, program, stream, a.warm);
          };
          auto captured = live.programs.runs().CaptureAhead(runs, a.inputs, program, a.outputs,
                                                            live.stats, native, between, a.warm);
          if (!captured) {
            *failure = captured.error().detail;
            return scheduler::JobResult::kUnknown;
          }
        }
        return scheduler::JobResult::kQueued;
      },
      "image recordings ahead", stream());
  if (!ran || !failure->empty() || s.programs.launch()->faulted()) {
    s.unproven.Mark();
    return Error(std::format("recording the image's stages ahead: {}", !ran ? ran.error()
                                                                       : !failure->empty()
                                                                           ? *failure
                                                                           : "launch faulted"));
  }
  s.flight.ReleaseRetired();
  return {};
}

Status StagedExecution::CheckRequestBoundary() const {
  if (!state_ || !state_->bound || state_->released || state_->unproven.active() ||
      state_->flight.held() || !transitions_.empty() || node_.InRequest(stream()) ||
      !(state_->next == 0 ||
        (state_->generation_complete && state_->next == state_->parts.request.sequence.size())))
    return Error("staged preparation requires an idle initial or completed request");
  return {};
}

Status StagedExecution::ReplaceRequest(StagedRequest request) {
  if (!state_ || !state_->bound || state_->released || state_->unproven.active() ||
      state_->flight.held() ||
      !(state_->next == 0 ||
        (state_->generation_complete && state_->next == state_->parts.request.sequence.size())) ||
      node_.InRequest(stream()))
    return Error("image request replacement requires a completed, unleased generation");
  auto& s = *state_;
  const auto& current = s.parts.request;
  if (request.layout != current.layout || request.ports.size() != current.ports.size())
    return Error("image request geometry requires separate resource preparation");
  for (std::size_t i = 0; i < request.ports.size(); ++i)
    if (request.ports[i].format != current.ports[i].format ||
        request.ports[i].owner != current.ports[i].owner)
      return Error("image request changed a retained typed port");
  auto initial = s.Request(request);
  if (!initial || *initial > s.input_bytes)
    return Error(initial ? "image request exceeds retained initial staging" : initial.error());
  // No callbacks, allocation, device work or catalog mutation follows validation.
  // Old request-only owners retire here; graphs, addresses and recordings remain.
  s.parts.request = std::move(request);
  s.generation_complete = true;
  s.next = s.parts.request.sequence.size();
  return {};
}

Status StagedExecution::ReplaceGeometry(StagedGeometry geometry) {
  if (!state_ || !state_->bound || state_->released || state_->unproven.active() ||
      state_->flight.held() || !transitions_.empty() ||
      !(state_->next == 0 ||
        (state_->generation_complete && state_->next == state_->parts.request.sequence.size())) ||
      node_.InRequest(stream()))
    return Error("image geometry replacement requires a completed, unleased generation");
  auto previous = state_;
  auto candidate = std::make_shared<State>(previous->model, std::move(geometry));
  // Retain candidate ownership even when setup or cleanup refuses midway.
  transitions_.push_back(candidate);
  const auto refuse = [&](Status failure) -> Status {
    candidate->plans.Clear();
    if (auto retired = candidate->memory.Retire(); !retired) {
      previous->unproven.Mark();
      previous->flight.Retain(previous);
      previous->transition_holds = transitions_;
      return Error(std::format("{}; candidate retirement: {}", failure.error(), retired.error()));
    }
    candidate->released = true;
    transitions_.clear();
    return failure;
  };
  if (auto prepared = PrepareState(*candidate); !prepared) return refuse(prepared);
  if (auto grown = node_.GrowWorkspace(candidate->activation_bytes, candidate->pool_bytes); !grown)
    return refuse(grown);
  candidate->registered = previous->model->registered;
  if (auto bound = BindState(*candidate); !bound) return refuse(bound);
  // A pageable workspace's growth is only cataloged. While the image is the
  // active model its new extents come in here, through the reclaim order
  // like any acquisition, so a growth that cannot have its room is refused
  // while the previous geometry still stands, not at the next generation's
  // first stage. While another model is resident (a request prepared
  // before its activation), its activation brings them in with the rest of
  // the image's closure, the outgoing model's weights making the room, and
  // refuses before anything moves if they do not fit.
  if (node_.workspace_pageable() && node_.active(index_)) {
    scheduler::AcquireReport report;
    if (auto paged = node_.Acquire(candidate->fence, report, "the image's grown workspace"); !paged)
      return refuse(paged);
  }
  return InstallGeometry(previous, candidate);
}

Status StagedExecution::InstallGeometry(const std::shared_ptr<State>& previous,
                                        const std::shared_ptr<State>& candidate) {
  // All candidate validation/payment precedes retiring the old geometry. The
  // shared model/launch and node-wide workspace remain alive throughout.
  previous->plans.Clear();
  if (auto retired = previous->memory.Retire(); !retired) {
    // Resumed after a fence (AcknowledgeFence): Retire continues where it
    // stopped.
    previous->unproven.Mark();
    previous->flight.Retain(previous);
    previous->transition_holds = transitions_;
    previous->retiring = true;
    return retired;  // partial retirement is never represented as old usability
  }
  previous->released = true;
  previous->retiring = false;
  previous->transition_holds.clear();
  candidate->programs.runs().SetStaging(candidate->input, candidate->input_bytes);
  candidate->generation_complete = true;
  candidate->next = candidate->parts.request.sequence.size();
  // A smaller geometry of the active image: what of the workspace only the
  // previous one used goes out (scratch: nothing to keep), as a swap to a
  // model that uses less would take it. Its places stay registered.
  std::vector<catalog::ExtentId> spare;
  if (node_.workspace_pageable() && node_.active(index_)) {
    std::vector<catalog::ExtentId> before, after;
    AddWorkspacePrefix(before, previous->activation_bytes, node_.activations());
    AddWorkspacePrefix(before, previous->pool_bytes, node_.pool());
    AddWorkspacePrefix(after, candidate->activation_bytes, node_.activations());
    AddWorkspacePrefix(after, candidate->pool_bytes, node_.pool());
    Unique(after);
    for (const auto extent : before)
      if (!std::ranges::binary_search(after, extent)) spare.push_back(extent);
  }
  state_ = candidate;
  transitions_.clear();
  if (!spare.empty()) {
    // The new geometry and its request are in place already: a failed
    // eviction leaves the spare extents resident (charged, unused; the next
    // swap's room takes them), logged, never the geometry unusable.
    if (auto evicted = node_.Evict(std::move(spare)); !evicted)
      if (state_->model->parts.log)
        state_->model->parts.log(
            std::format("model {}: the workspace its smaller geometry no longer uses stays "
                        "resident: {}",
                        index_, evicted.error()));
  }
  return {};
}

std::optional<base::Sha256Digest> StagedExecution::request_layout() const {
  if (!state_) return std::nullopt;
  return state_->parts.request.layout;
}

Status StagedExecution::FirstOutput(std::string& sha) {
  if (state_ && state_->generation_complete && !state_->unproven.active() &&
      !state_->flight.held() && !state_->released) {
    state_->next = 0;
    state_->generation_complete = false;
  }
  if (!state_ || state_->next != 0) return Error("image first output order");
  return RunThrough(state_->parts.request.first_output + 1, sha);
}
Status StagedExecution::Finish(std::string& sha) {
  if (!state_ || state_->next != state_->parts.request.first_output + 1)
    return Error("image finish order");
  auto result = RunThrough(state_->parts.request.sequence.size(), sha);
  if (result) state_->generation_complete = true;
  return result;
}

Status StagedExecution::Abandon() {
  if (!state_ || state_->unproven.active() || state_->flight.held() || state_->released)
    return Error("image abandon requires an idle, known generation");
  // Only the position: what the stopped generation wrote is overwritten by
  // the next one's stages, from its initial ports on.
  if (!state_->generation_complete) state_->next = 0;
  return {};
}

std::uint64_t StagedExecution::activations_needed() const {
  return state_ ? state_->activation_bytes : 0;
}
std::uint64_t StagedExecution::pool_needed() const { return state_ ? state_->pool_bytes : 0; }
std::uint64_t StagedExecution::host_input_bytes() const {
  return state_ ? state_->input_bytes + state_->output_bytes : 0;
}
std::uint64_t StagedExecution::startup_host_bytes() const {
  const auto payment = state_ ? state_->model->parts.construction : nullptr;
  return payment ? payment->capacity() + en::ExecutionMetadataAllowance::OverheadBytes() : 0;
}
std::uint64_t StagedExecution::plan_floor_bytes() const { return state_ ? state_->plan_floor : 0; }
void StagedExecution::ReclaimCandidates(std::uint32_t owner, bool running,
                                        std::vector<memory::ReclaimCandidate>& out) {
  if (state_ && !state_->unproven.active() && !state_->flight.held() && !state_->released)
    state_->plans.Collect(owner, running, out);
}
std::uint64_t StagedExecution::Reclaim(memory::ReclaimKind kind, std::uint64_t id) {
  return state_ && !state_->unproven.active() && !state_->flight.held() && !state_->released
             ? state_->plans.Reclaim(kind, id)
             : 0;
}
std::vector<catalog::ExtentId> StagedExecution::weights() const {
  std::vector<catalog::ExtentId> result;
  if (state_)
    for (const auto& ids : state_->stage_weights)
      result.insert(result.end(), ids.begin(), ids.end());
  Unique(result);
  return result;
}
std::vector<catalog::ExtentId> StagedExecution::state() const {
  // Typed carries are pinned runtime buffers, not spillable conversation state.
  return {};
}
const catalog::Closure& StagedExecution::everything() const {
  base::Check(state_ != nullptr, "image before setup");
  return state_->all;
}
const catalog::Closure& StagedExecution::fence_closure() const {
  base::Check(state_ != nullptr, "image before setup");
  return state_->fence;
}
std::vector<catalog::ExtentId> StagedExecution::managed_extents() const {
  // Pinned ports/staging are released by RunnerMemory after the node fence;
  // only pageable source extents participate in scheduler eviction.
  std::vector<catalog::ExtentId> result;
  if (state_)
    for (const auto& source : state_->model->parts.weights) {
      if (!source.weights) continue;
      const auto& ids = source.weights->extents();
      result.insert(result.end(), ids.begin(), ids.end());
    }
  Unique(result);
  return result;
}
std::uint64_t StagedExecution::weight_read_bytes() const {
  return weights().size() * en::kPagedExtent;
}
Status StagedExecution::CheckPlaces() {
  if (!state_ || !state_->bound) return Error("image places before bind");
  en::PlaceCheck check;
  for (const auto& source : state_->model->parts.weights)
    source.weights->CheckPlaces(node_.scheduler(), check);
  if (check.moved) return Error("image retained weight place differs");
  return {};
}
void StagedExecution::DropPlans() {
  if (state_ && !state_->unproven.active() && !state_->flight.held()) state_->plans.Clear();
}
std::string StagedExecution::plan_report() const {
  if (!state_ || !state_->setup) return {};
  return kernels::FormatPlanCoverageReport(state_->stage_envelopes.size(), state_->plan_coverage);
}
void StagedExecution::SetPlanReportSink(kernels::PlanRecordSink sink) {
  plan_report_sink_ = std::move(sink);
}
void StagedExecution::ReportPlans(const kernels::PlanRecordSink& sink) const {
  if (!sink) return;
  const auto emit = [&](const State& state) {
    state.plans.ForEach(
        [&](const State::Plans::Entry& entry) { State::EmitPlanReport(entry, sink); });
  };
  if (state_) emit(*state_);
  for (const auto& pending : transitions_) emit(*pending);
}
std::string StagedExecution::extra() const {
  return state_
             ? std::format(R"({{"engine":"component","architecture":"{}","completed_stages":{}}})",
                           state_->model->parts.architecture, state_->next)
             : "{}";
}
Status StagedExecution::Release() {
  if (!drained_) return Error("staged release requires node-authenticated stream drain");
  if (release_failed_)
    return Error("staged physical retirement failed; its owner remains retained");
  if (!state_ || state_->released) {
    RetireRoot();
    return {};
  }
  std::vector<std::string> problems;
  auto model = state_->model;
  const auto release = [&](State& s) {
    if (s.released) return;
    s.released = true;
    s.transition_holds.clear();
    s.plans.Clear();
    s.memory.ReleaseMappings(problems);
  };
  release(*state_);
  for (auto& pending : transitions_) release(*pending);
  transitions_.clear();
  model->programs.ReleaseAfterDrain();
  if (!model->released) {
    model->released = true;
    if (model->prepared && model->parts.retire) model->parts.retire();
    model->memory.ReleaseMappings(problems);
    for (auto& weights : model->parts.weights) {
      if (!weights.weights) continue;
      if (auto r = weights.weights->Release(node_.memory()); !r) problems.push_back(r.error());
    }
  }
  auto result = en::support::Joined(problems);
  if (result)
    RetireRoot();
  else
    release_failed_ = true;
  return result;
}

}  // namespace llmp::engine
