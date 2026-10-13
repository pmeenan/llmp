// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "runtime/engine_image.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <format>
#include <limits>
#include <utility>

#include "base/check.h"
#include "engine/graph_plan.h"
#include "engine/program_lifecycle.h"
#include "engine/support.h"
#include "providers/device_runtime.h"

namespace llmp::runtime {
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
}  // namespace

struct EngineImage::State {
  struct Key {
    base::Sha256Digest composition, graph;
    execution::BackendDeclaration backend;
    std::uint32_t stage = 0;
    bool operator==(const Key&) const = default;
  };
  using Plans = en::BasicPlanCache<Key, en::StepPlanned, 1, en::StepPlanRuns>;
  struct Model {
    EngineImageParts parts;
    en::RunnerMemory memory;
    en::ProgramLifecycle programs;
    std::vector<std::unique_ptr<en::Mapped>> kernel;
    bool prepared = false, registered = false, released = false;
    Model(en::PagedNode& node, int owner, EngineImageParts source)
        : parts(std::move(source)),
          memory(node, owner, static_cast<std::uint32_t>(owner)),
          programs(node, owner, parts.graphs) {}
  };
  // One complete model/handle/launch owner precedes geometry-only borrowers.
  std::shared_ptr<Model> model;
  ImageGeometry parts;
  en::RunnerMemory memory;
  en::ProgramLifecycle& programs;
  Plans plans;
  std::vector<std::unique_ptr<en::Mapped>> ports;
  std::vector<std::vector<catalog::ExtentId>> stage_weights;
  std::vector<catalog::Closure> closures;
  catalog::Closure all, fence;
  execution::BackendDeclaration declaration;
  en::ProgramEnvelope measured;
  std::vector<en::ProgramEnvelope> stage_envelopes;
  std::vector<std::uint64_t> stage_capture_nodes;
  en::GraphStats stats;
  std::byte* input = nullptr;
  std::byte* output = nullptr;
  std::uint64_t input_bytes = 0, output_bytes = 0, activation_bytes = 0, pool_bytes = 0;
  std::uint64_t plan_floor = 0;
  std::size_t next = 0;
  bool setup = false, registered = false, bound = false, released = false;
  bool unknown = false, generation_complete = false;
  // A pending job's source, captures, plan and storage survive a failed wait.
  // Release, called after actual node drain, is the only cycle-breaking path.
  std::shared_ptr<State> flight;
  std::vector<std::shared_ptr<State>> transition_holds;

  State(en::PagedNode& node, int owner, EngineImageParts source)
      : model(std::make_shared<Model>(node, owner, std::move(source))),
        parts{.keep = model->parts.geometry_keep ? std::move(model->parts.geometry_keep)
                                                 : model->parts.keep,
              .identity = model->parts.identity,
              .stages = std::move(model->parts.stages),
              .request = std::move(model->parts.request)},
        memory(node, owner, static_cast<std::uint32_t>(owner)),
        programs(model->programs) {}
  State(std::shared_ptr<Model> retained, ImageGeometry geometry)
      : model(std::move(retained)),
        parts(std::move(geometry)),
        memory(model->memory.node(), model->memory.owner(), model->memory.stream()),
        programs(model->programs) {}

  kn::Addresses Addresses(std::size_t stage, bool measuring) const {
    auto result = parts.stages.at(stage).addresses;
    result.activations = measuring ? 0 : memory.node().activations().base;
    result.activation_bytes = measuring ? 0 : memory.node().activations().bytes;
    result.activation_alignment = en::kPagedExtent;
    return result;
  }
  std::expected<std::unique_ptr<kn::StepProgram>, std::string> Build(std::size_t stage,
                                                                     bool measuring) const {
    const auto& graph = *parts.stages.at(stage).graph;
    const en::GraphBuildIO io{graph.graph(), graph.graph().inputs(), graph.graph().outputs(),
                              graph.weights()};
    const auto& backend = model->parts.backend->owner->backend();
    return en::BuildComponentProgram(
        io, backend, model->parts.policy, Addresses(stage, measuring),
        {.staged = std::vector(io.inputs.begin(), io.inputs.end()),
         .written = backend.written,
         .within = backend.within,
         .alignment_staged = std::vector(io.inputs.begin(), io.inputs.end())});
  }
  std::expected<std::uint64_t, std::string> Request(const ImageRequest& request) const {
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

EngineImage::EngineImage(en::PagedNode& node, const config::ModelEntry& entry,
                         const ModelSettings& settings, int index, EngineImageBuild build)
    : node_(node), index_(index), build_(std::move(build)) {
  name_ = entry.name;
  settings_ = settings;
}
EngineImage::~EngineImage() = default;
en::PagedModel& EngineImage::paged() { return *this; }

Status EngineImage::Setup() {
  if (state_ || !build_ || index_ < 0) return Error("image setup order or source differs");
  auto parts = build_(settings_);
  if (!parts) return Error(parts.error());
  // Reject incomplete source owners before publishing any cleanup borrower.
  for (const auto& source : parts->weights)
    if (!source.weights) return Error("image source weights are missing");
  // All source construction was paid by the factory before these values exist.
  state_ = std::make_shared<State>(node_, index_, std::move(*parts));
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

Status EngineImage::PrepareState(State& s) {
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

Status EngineImage::Register() {
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
Status EngineImage::Bind() {
  if (!state_ || !state_->registered || state_->bound) return Error("image bind order");
  auto& s = *state_;
  if (auto bound = BindState(s); !bound) return bound;
  s.programs.runs().SetStaging(s.input, s.input_bytes);
  return {};
}
Status EngineImage::BindState(State& s) {
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
    for (const auto* mapping : {&node_.activations(), &node_.pool()})
      common.insert(common.end(), mapping->extents.begin(), mapping->extents.end());
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

Status EngineImage::RunThrough(std::size_t end, std::string& sha) {
  if (!state_ || !state_->bound || state_->released || state_->unknown ||
      end > state_->parts.request.sequence.size() || end <= state_->next)
    return Error("image execution order or completion is invalid");
  auto hold = state_;
  auto& s = *hold;
  if (s.activation_bytes > node_.activations().bytes || s.pool_bytes > node_.pool().bytes)
    return Error("image execution exceeds the backed workspace prefix");
  for (; s.next < end; ++s.next) {
    const en::PlanStep protect;
    const auto invocation = s.parts.request.sequence[s.next];
    const auto& stage = s.parts.stages[invocation.stage];
    const auto& graph = stage.graph->graph();
    auto ready = s.model->parts.backend->owner->Ready();
    if (!ready || *ready != s.declaration) return Error("image backend identity changed");
    // Catalog content identities are refreshed between completed stages.
    auto ids = s.memory.extents();
    const auto model_extents = s.model->memory.extents();
    ids.insert(ids.end(), model_extents.begin(), model_extents.end());
    for (const auto* mapped : {&node_.activations(), &node_.pool()})
      ids.insert(ids.end(), mapped->extents.begin(), mapped->extents.end());
    const auto& selected = s.stage_weights[invocation.stage];
    ids.insert(ids.end(), selected.begin(), selected.end());
    Unique(ids);
    auto closure = node_.catalog().ClosureOfExtents(ids);
    if (!closure) return Error("image current stage closure is absent");
    const bool opened = !node_.InRequest(stream());
    auto admitted = opened ? node_.BeginRequest(stream(), *closure, "image stage")
                           : node_.RefreshRequest(stream(), *closure);
    if (!admitted) {
      s.unknown = true;
      s.flight = hold;
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
      s.flight = hold;
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
        s.unknown = true;
        return uploaded;
      }
      s.flight.reset();
    }
    const State::Key key{s.parts.identity, graph.Digest(), s.declaration, invocation.stage};
    auto* entry = s.plans.Find(key);
    Status checked;
    if (!entry) {
      auto program = s.Build(invocation.stage, false);
      if (!program)
        checked = Error(program.error());
      else {
        auto planned = std::make_unique<en::StepPlanned>();
        planned->program = std::move(*program);
        auto bound =
            s.programs.BindPrepared(*planned, [&](const kn::StepProgram& actual) -> Status {
              auto envelope = en::ProgramEnvelope::Measure(actual, true);
              const auto& paid = s.stage_envelopes[invocation.stage];
              if (!envelope || envelope->scratch > paid.scratch || envelope->pool > paid.pool ||
                  envelope->host > paid.host || envelope->launches > paid.launches ||
                  envelope->inputs > paid.inputs || envelope->outputs > paid.outputs)
                return Error("bound image program exceeds its measured envelope");
              return s.Ports(invocation.stage, actual);
            });
        if (!bound)
          checked = Error(bound.error());
        else {
          const auto bytes = planned->program->host_bytes();
          entry = &s.plans.Add(key, std::move(planned), bytes,
                               s.stage_capture_nodes[invocation.stage], 0);
          entry->runs[0].uncapturable = entry->planned->positioned;
        }
      }
    }
    if (!checked) {
      auto ended = opened ? node_.EndRequest(stream()) : node_.RefreshRequest(stream(), s.fence);
      if (!ended) {
        s.unknown = true;
        s.flight = hold;
      }
      return checked;
    }
    auto& program = *entry->planned->program;
    // Eager programs still receive the common between-jobs positioning hook.
    // A stateless typed stage has no token positions; refusal precedes Run.
    if (entry->planned->positioned) {
      if (auto positioned = program.Position({}); !positioned)
        checked = Error(positioned.error().detail);
    }
    en::Copies inputs;
    std::uint64_t cursor = 0;
    for (std::size_t i = 0; checked && i < program.inputs().size(); ++i) {
      const auto& input = program.inputs()[i];
      if (stage.inputs[i]) continue;
      if (!s.parts.request.fill || cursor > s.input_bytes || input.bytes > s.input_bytes - cursor) {
        checked = Error("image host input has no paid source");
        break;
      }
      checked = s.parts.request.fill(invocation, i,
                                     {s.input + cursor, static_cast<std::size_t>(input.bytes)});
      if (!checked) break;
      inputs.push_back({input.device_address, input.bytes, cursor});
      cursor += Round(input.bytes, 256);
    }
    if (!checked) {
      auto ended = opened ? node_.EndRequest(stream()) : node_.RefreshRequest(stream(), s.fence);
      if (!ended) {
        s.unknown = true;
        s.flight = hold;
      }
      return checked;
    }
    const bool observe = s.next + 1 == end;
    const auto& observed = program.outputs()[stage.observation];
    en::Copies outputs;
    // Fixed observation addresses are part of the captured plan contract.
    // Always copy this stage output; only requested boundaries publish it.
    outputs.push_back({en::support::Address(s.output), observed.device_address, observed.bytes});
    const bool capture =
        entry->runs[0].CaptureDue(s.model->parts.graphs) && s.plans.ChargeGraph(*entry);
    s.flight = hold;
    auto queued = std::make_shared<en::StepQueued>();
    auto job = [hold, entry, invocation, inputs = std::move(inputs), outputs = std::move(outputs),
                capture, queued](providers::NativeStream native) -> scheduler::JobResult {
      auto& live = *hold;
      const auto& descriptor = live.parts.stages[invocation.stage];
      const auto& actual = *entry->planned->program;
      const auto between = [&](providers::NativeStream stream) {
        for (std::size_t i = 0; i < descriptor.inputs.size(); ++i) {
          if (!descriptor.inputs[i]) continue;
          const auto from = live.ports[*descriptor.inputs[i]]->base;
          if (!providers::CopyAsync(stream, Pointer(actual.inputs()[i].device_address),
                                    Pointer(from), actual.inputs()[i].bytes,
                                    providers::CopyKind::kDeviceToDevice)
                   .ok())
            return false;
        }
        return true;
      };
      *queued = live.programs.runs().Queue(entry->runs[0], inputs, between, actual, outputs,
                                           capture, live.stats, native);
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
      s.unknown = true;
      return Error(!ran              ? ran.error()
                   : !queued->result ? queued->result.error().detail
                                     : "image launch faulted");
    }
    auto ended = opened ? node_.EndRequest(stream()) : node_.RefreshRequest(stream(), s.fence);
    if (!ended) {
      s.unknown = true;
      return ended;
    }
    s.flight.reset();
    en::Count(s.stats, queued->path);
    if (observe) {
      const std::span<const std::byte> bytes(s.output, static_cast<std::size_t>(observed.bytes));
      if (end == s.parts.request.sequence.size() && s.parts.request.finish) {
        auto converted = s.parts.request.finish(bytes);
        if (!converted) return Error(converted.error());
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
Status EngineImage::ReplaceRequest(ImageRequest request) {
  if (!state_ || !state_->bound || state_->released || state_->unknown || state_->flight ||
      !state_->generation_complete || state_->next != state_->parts.request.sequence.size() ||
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
  s.next = s.parts.request.sequence.size();
  return {};
}

Status EngineImage::ReplaceGeometry(ImageGeometry geometry) {
  if (!state_ || !state_->bound || state_->released || state_->unknown || state_->flight ||
      !transitions_.empty() || !state_->generation_complete ||
      state_->next != state_->parts.request.sequence.size() || node_.InRequest(stream()))
    return Error("image geometry replacement requires a completed, unleased generation");
  auto previous = state_;
  auto candidate = std::make_shared<State>(previous->model, std::move(geometry));
  // Retain candidate ownership even when setup or cleanup refuses midway.
  transitions_.push_back(candidate);
  const auto refuse = [&](Status failure) -> Status {
    candidate->plans.Clear();
    if (auto retired = candidate->memory.Retire(); !retired) {
      previous->unknown = true;
      previous->flight = previous;
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
  // All candidate validation/payment precedes retiring the old geometry. The
  // shared model/launch and node-wide workspace remain alive throughout.
  previous->plans.Clear();
  if (auto retired = previous->memory.Retire(); !retired) {
    previous->unknown = true;
    previous->flight = previous;
    previous->transition_holds = transitions_;
    return retired;  // partial retirement is never represented as old usability
  }
  previous->released = true;
  candidate->programs.runs().SetStaging(candidate->input, candidate->input_bytes);
  candidate->generation_complete = true;
  candidate->next = candidate->parts.request.sequence.size();
  state_ = std::move(candidate);
  transitions_.clear();
  return {};
}

Status EngineImage::FirstOutput(std::string& sha) {
  if (state_ && state_->generation_complete && !state_->unknown && !state_->flight &&
      !state_->released) {
    state_->next = 0;
    state_->generation_complete = false;
  }
  if (!state_ || state_->next != 0) return Error("image first output order");
  return RunThrough(state_->parts.request.first_output + 1, sha);
}
Status EngineImage::Finish(std::string& sha) {
  if (!state_ || state_->next != state_->parts.request.first_output + 1)
    return Error("image finish order");
  auto result = RunThrough(state_->parts.request.sequence.size(), sha);
  if (result) state_->generation_complete = true;
  return result;
}

std::uint64_t EngineImage::activations_needed() const {
  return state_ ? state_->activation_bytes : 0;
}
std::uint64_t EngineImage::pool_needed() const { return state_ ? state_->pool_bytes : 0; }
std::uint64_t EngineImage::host_input_bytes() const {
  return state_ ? state_->input_bytes + state_->output_bytes : 0;
}
std::uint64_t EngineImage::plan_floor_bytes() const { return state_ ? state_->plan_floor : 0; }
void EngineImage::ReclaimCandidates(std::uint32_t owner, bool running,
                                    std::vector<memory::ReclaimCandidate>& out) {
  if (state_ && !state_->unknown && !state_->flight && !state_->released)
    state_->plans.Collect(owner, running, out);
}
std::uint64_t EngineImage::Reclaim(memory::ReclaimKind kind, std::uint64_t id) {
  return state_ && !state_->unknown && !state_->flight && !state_->released
             ? state_->plans.Reclaim(kind, id)
             : 0;
}
std::vector<catalog::ExtentId> EngineImage::weights() const {
  std::vector<catalog::ExtentId> result;
  if (state_)
    for (const auto& ids : state_->stage_weights)
      result.insert(result.end(), ids.begin(), ids.end());
  Unique(result);
  return result;
}
std::vector<catalog::ExtentId> EngineImage::state() const {
  // Typed carries are pinned runtime buffers, not spillable conversation state.
  return {};
}
const catalog::Closure& EngineImage::everything() const {
  base::Check(state_ != nullptr, "image before setup");
  return state_->all;
}
const catalog::Closure& EngineImage::fence_closure() const {
  base::Check(state_ != nullptr, "image before setup");
  return state_->fence;
}
std::vector<catalog::ExtentId> EngineImage::managed_extents() const {
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
std::uint64_t EngineImage::weight_read_bytes() const { return weights().size() * en::kPagedExtent; }
Status EngineImage::CheckPlaces() {
  if (!state_ || !state_->bound) return Error("image places before bind");
  en::PlaceCheck check;
  for (const auto& source : state_->model->parts.weights)
    source.weights->CheckPlaces(node_.scheduler(), check);
  if (check.moved) return Error("image retained weight place differs");
  return {};
}
void EngineImage::DropPlans() {
  if (state_ && !state_->unknown && !state_->flight) state_->plans.Clear();
}
std::string EngineImage::extra() const {
  return state_
             ? std::format(R"({{"engine":"component","architecture":"{}","completed_stages":{}}})",
                           state_->model->parts.architecture, state_->next)
             : "{}";
}
Status EngineImage::Release() {
  if (!state_ || state_->released) return {};
  std::vector<std::string> problems;
  auto model = state_->model;
  const auto release = [&](State& s) {
    if (s.released) return;
    s.released = true;
    s.flight.reset();
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
  return en::support::Joined(problems);
}

}  // namespace llmp::runtime
