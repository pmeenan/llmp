// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "runtime/serving.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <format>
#include <limits>
#include <map>
#include <print>
#include <set>
#include <string>
#include <string_view>
#include <system_error>
#include <tuple>
#include <utility>

#include "artifact/artifact.h"
#include "artifact/composition.h"
#include "base/build_info.h"
#include "base/check.h"
#include "base/report.h"
#include "components/step.h"
#include "engine/component_source.h"
#include "engine/dsv4_plan.h"
#include "engine/dsv4_runner.h"
#include "engine/gemma4_runner.h"
#include "engine/paged_weights.h"
#include "engine/qwen38_runner.h"
#include "engine/qwen_image_runner.h"
#include "execution/adaptive_depth.h"
#include "execution/adaptive_wave_mode.h"
#include "kernels/ggml/cublas.h"
#include "kernels/ggml/step_program.h"
#include "memory/reclaim.h"
#include "model/dsv4.h"
#include "model/gemma2.h"
#include "model/gemma3.h"
#include "model/qwen38.h"
#include "planner/step_shape.h"
#include "platform/crash_policy.h"
#include "platform/files.h"
#include "platform/host_probe.h"
#include "platform/kept_files.h"
#include "platform/memory_pressure.h"
#include "providers/device_runtime.h"
#include "runtime/engine_image.h"
#include "runtime/engine_llm.h"
#include "runtime/gemma_profile.h"
#include "runtime/gemma_wave.h"
#include "runtime/intake_limits.h"
#include "runtime/memory_guard.h"
#include "runtime/model_limits.h"
#include "runtime/prefill.h"
#include "runtime/serving_support.h"
#include "runtime/swap_room.h"
#include "scheduler/programs.h"
#include "spec/mapping.h"
#include "spec/mappings/gemma2.h"
#include "spec/mappings/gemma3.h"
#include "spec/spec.h"

namespace llmp::runtime {
namespace {

using serving_support::Errno;
using serving_support::Error;
using serving_support::KeptRegionsOf;
using serving_support::kExtent;
using serving_support::NowUnixMs;
using serving_support::Seconds;

namespace fs = std::filesystem;
namespace ja = llmp::artifact;

// Extents the page-in observer tracks (the M3 models use about 100,000).
constexpr std::size_t kObservedExtents = std::size_t{1} << 18U;
// Each model's settings (prefill chunk, request slots, drafting, waves,
// floors, sampling) resolve at registration in model_settings.h's three
// layers (D-103); their fallbacks are its constants, which name what the
// engine does too.
static_assert(Llm::kMaxBranches == engine::kMaxRequestSlots);
static_assert(config::kMaxModelSlots == engine::kMaxRequestSlots);
static_assert(kDsv4SpeculativeMostSlots == engine::Dsv4Runner::kWaveRows / 2);
static_assert(config::kMaxWaveCostWidths + 1 == execution::AdaptiveWaveMode::kMaxWidth);

// A model's request slots for the start's log, and where their number came
// from (its settings).
std::string SlotsReport(const ModelSettings& s) {
  const std::uint32_t slots = s.max_slots.value;
  return std::format("{} request slot{} ({}: {})", slots, slots == 1 ? "" : "s",
                     SourceName(s.max_slots.source), s.max_slots.basis);
}

// The component engine's parts for a family moved to it (D-107): its spec
// from its architecture's mapping (a v0 artifact's, with its compiled
// profile's values, D-109's synthesis), its components' state layout and
// batching, and the component step source planned under the mapping's
// default policy for those values (its recipe as data, spec/mapping.h
// DefaultPolicy) over GGML's registry and lowered onto its
// graphs (engine/component_source.h, kernels/ggml/step_program.h). The
// backend runs on the model's own stream (its runner's job stream), with a
// cuBLAS handle over a workspace the runner maps and charges as its own
// (RunnerConfig::kernel_memory, as RunnerResources does for the M3
// runners). A head row is the head's vocabulary of logits; a greedy step
// returns each owner's token chosen on the device. Gemma 2 and Gemma 3
// (their compiled profiles' values); any other architecture, or a drafter,
// is refused at Setup, naming why.
struct ComponentHold {
  spec::ModelSpec spec;
  spec::Source source;
  model::StateLayoutDerived layout;
  std::map<ir::WeightRef, std::uint32_t> resource;
  // Before the backend, whose launch borrows it (the runner's Release
  // destroys that launch, then retires the handle).
  std::unique_ptr<kernels::ggml::CublasHandle> cublas;
  std::unique_ptr<kernels::ggml::GgmlBackend> backend;
};

// A v0 artifact's values for its architecture's mapping: its compiled
// profile's (spec/mappings/*.h).
std::expected<std::vector<spec::SourceValue>, std::string> V0Values(std::string_view architecture) {
  if (architecture == "gemma2") return spec::mappings::Gemma2V0Values(model::Gemma2_2B());
  if (architecture == "gemma3") return spec::mappings::Gemma3V0Values(model::Gemma3_4BQat());
  return Error(
      std::format("the component engine has no recipe for architecture {} yet", architecture));
}

// The architectures served by the component engine's adapter for their
// pipeline kind (engine_llm.h, engine_image.h; D-107): Gemma 2 and Gemma 3
// since their migration. A family joins when its migration passes its gate.
bool OnComponentEngine(std::string_view architecture) {
  return architecture == "gemma2" || architecture == "gemma3";
}

EngineBuild ComponentLlmBuild(engine::PagedNode& node, fs::path store,
                              const config::ModelEntry& entry, int index) {
  return [&node, store = std::move(store), id = entry.artifact.value_or(""),
          drafter = entry.drafter.has_value(),
          index](const ModelSettings& settings) -> std::expected<EngineLlmParts, std::string> {
    if (drafter) {
      return Error(std::format("the component engine has no drafter stage for architecture {} yet",
                               settings.architecture));
    }
    auto values = V0Values(settings.architecture);
    if (!values) return Error(values.error());
    auto artifact = OpenInstalled(store, id);
    if (!artifact) return Error(artifact.error());
    const spec::Mapping* mapping = spec::FindMapping(artifact->model().architecture);
    if (mapping == nullptr) return Error("the artifact's architecture has no mapping");
    auto hold = std::make_shared<ComponentHold>();
    auto synthesized = spec::SynthesizeV0(*artifact, *mapping, *values);
    if (!synthesized) return Error(synthesized.error());
    hold->spec = std::move(*synthesized);
    auto source = spec::SourceOfV0(*artifact);
    if (!source) return Error(source.error());
    hold->source = std::move(*source);
    if (auto valid = components::ValidateSpec(hold->spec, hold->source.tensors); !valid) {
      return Error(valid.error());
    }
    // The spec bound to the artifacts its stages run from (each stage's
    // index into them; D-105's layout identity and the plans' key): a v0
    // artifact's one target stage, so its own ID.
    const std::array<std::string_view, 1> artifacts{id};
    auto bound = spec::BoundDigest(hold->spec, artifacts);
    if (!bound) return Error(bound.error());
    // Each state tensor starts on its own paged extent, as the M3 layouts
    // did: packed tensors would straddle extents, so their growth would
    // cross extent boundaries at staggered positions and a prompt would
    // grow its state (each growth a fixed host cost between steps) every
    // chunk or two instead of once every extent's worth of cells.
    auto layout = components::StageLayout(hold->spec, 0, settings.context.value,
                                          settings.prefill_chunk.value, kExtent);
    if (!layout) return Error(layout.error());
    hold->layout = std::move(*layout);
    auto caps = components::StageBatching(hold->spec, 0);
    if (!caps) return Error(caps.error());
    const std::uint32_t slots = std::clamp<std::uint32_t>(
        settings.max_slots.value, 1, static_cast<std::uint32_t>(Llm::kMaxBranches));
    const std::uint32_t context = hold->layout.context;
    const std::uint32_t chunk = std::max<std::uint32_t>(hold->layout.max_rows, 1);
    // The widest steps (the setup measurement): every owner's rows at the
    // end of its context, reading all of it.
    const auto shape = [&](planner::Phase phase, std::uint32_t owners, std::uint32_t rows,
                           bool greedy) {
      planner::StepShape s{
          .stage = 0, .phase = phase, .segments = {}, .row_invariant = false, .read_align = 256};
      for (std::uint32_t o = 0; o < owners; ++o) {
        s.segments.push_back({.slot = static_cast<std::uint8_t>(o),
                              .rows = rows,
                              .past = context - rows,
                              .read_cells = context,
                              .outputs = 1,
                              .group = 0,
                              .want_head = true,
                              .greedy = greedy});
      }
      return s;
    };
    const std::uint32_t joined = std::min(slots, std::max(caps->max_segments, 1U));
    // Each weight's resource, by what a step binds (the head's included),
    // and a head row's width: its logits' (the head's vocabulary).
    std::uint32_t logit_width = 0;
    {
      auto step =
          components::BuildStep(hold->spec, 0, shape(planner::Phase::kPrefill, 1, chunk, false),
                                {.tensors = hold->source.tensors, .state = &hold->layout});
      if (!step) return Error(step.error());
      for (const components::StepWeight& w : step->weights) {
        const auto found = artifact->FindResource(w.parts.front().first);
        if (!found) return Error(std::format("no resource holds {}", w.parts.front().first));
        hold->resource[w.ref] = *found;
      }
      if (!step->logits.valid()) return Error("the spec's target stage has no head");
      const std::uint64_t width = step->graph.tensor(step->logits).shape().ne[0];
      if (width == 0 ||
          width > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
        return Error(std::format("the head's {} logits a row", width));
      }
      logit_width = static_cast<std::uint32_t>(width);
    }
    auto backend = kernels::ggml::GgmlBackend::Create(
        0, node.execution(), node.stream(static_cast<std::uint32_t>(index)), nullptr,
        {.lower = {.device_masks = true}});
    if (!backend) return Error(backend.error());
    hold->backend = std::move(*backend);
    auto component = engine::ComponentStepSource::Create(
        {.spec = &hold->spec,
         .stage = 0,
         .bindings = {.tensors = hold->source.tensors, .state = &hold->layout},
         .bound_digest = *bound,
         .mapping = mapping,
         .values = *values,
         .slots = slots,
         .policy = std::nullopt,
         .backend = hold->backend->Backend()});
    if (!component) return Error(component.error());
    auto weights = std::make_unique<engine::PagedWeights>();
    if (auto opened = weights->Open(store / id); !opened) return Error(opened.error());
    engine::PagedWeights* paged = weights.get();
    EngineLlmParts parts;
    parts.config.slots = {.model = settings.name,
                          .slots = slots,
                          .stages = {{.name = "target", .layout = hold->layout, .read_align = 0}},
                          .spec = *bound,
                          .read_align = 256,
                          .spill_place = {},
                          .spill_directory = {},
                          .invalidate = {},
                          .drop_plans = {}};
    std::vector<engine::GroupPlace> places(paged->artifact().groups().size(),
                                           engine::GroupPlace::kDevice);
    parts.config.weights.push_back(
        {.weights = std::move(weights), .places = std::move(places), .slabs = {}});
    // A prompt chunk and a decode row, joined and alone, with logits and
    // with device tokens: their plans differ (owner attention joins
    // owners), so each is measured.
    std::vector<planner::StepShape>& setup = parts.config.setup_shapes;
    for (const std::uint32_t owners : {1U, joined}) {
      planner::StepShape prompt = shape(planner::Phase::kPrefill, owners, chunk, false);
      if ((owners == 1 || caps->join_prefill) && std::ranges::find(setup, prompt) == setup.end()) {
        setup.push_back(std::move(prompt));
      }
      for (const bool greedy : {false, true}) {
        planner::StepShape row = shape(planner::Phase::kDecode, owners, 1, greedy);
        if (std::ranges::find(setup, row) == setup.end()) setup.push_back(std::move(row));
      }
    }
    // cuBLAS on the model's stream over the runner's workspace (D-053: the
    // engine's handle, never one GGML makes).
    const providers::DeviceFacts facts =
        providers::QueryDeviceFacts(0).value_or(providers::DeviceFacts{});
    const std::uint64_t cublas_bytes =
        kernels::ggml::CublasHandle::UpstreamWorkspace(static_cast<int>(facts.architecture))
            .value();
    parts.config.kernel_memory.push_back(
        {.name = std::format("{} cuBLAS workspace", settings.name), .bytes = cublas_bytes});
    ComponentHold* held = hold.get();
    parts.config.prepare =
        [&node, held, index,
         cublas_bytes](std::span<const std::uint64_t> bases) -> std::expected<void, std::string> {
      if (bases.size() != 1) return Error("the cuBLAS workspace is not mapped");
      auto cublas = kernels::ggml::CublasHandle::Create(
          0, node.execution(), node.stream(static_cast<std::uint32_t>(index)),
          {.base = bases.front(), .size = base::Bytes(cublas_bytes)});
      if (!cublas) return Error(std::format("cuBLAS: {}", cublas.error().detail));
      held->cublas = std::move(*cublas);
      return held->backend->LendCublas(held->cublas.get());
    };
    parts.config.retire = [held] {
      held->backend->LaunchReleased();
      held->cublas.reset();
    };
    parts.config.launch =
        [&node, held]() -> std::expected<std::unique_ptr<kernels::StepLaunch>, std::string> {
      return held->backend->Launch(
          {.base = node.pool().base, .size = base::Bytes(node.pool().bytes)});
    };
    parts.config.weight_address = [held, paged](const ir::WeightRef& ref,
                                                format::PartRole) -> std::uint64_t {
      const auto at = held->resource.find(ref);
      return at == held->resource.end() ? 0 : paged->resource_address(at->second);
    };
    parts.source = std::move(*component);
    parts.caps = *caps;
    parts.logit_width = logit_width;
    parts.device_greedy = true;
    parts.prefill = {.chunk_rows = chunk,
                     .lookahead = 2,
                     .capture_ahead = true,
                     .prepare_ahead = true,
                     .state_only = true};
    parts.architecture = settings.architecture;
    parts.keep = std::move(hold);
    return parts;
  };
}
EngineImageBuild ComponentImageBuild(std::string architecture) {
  return [architecture = std::move(architecture)](
             const ModelSettings&) -> std::expected<EngineImageParts, std::string> {
    return Error(std::format(
        "the component engine has no step source for architecture {} yet (its migration wires it)",
        architecture));
  };
}

// The reclaim order's cost of idle conversation state (Server::Reclaim):
// its write-back and its restore, at the node's rates measured so far, and
// before any at the rates measured on a GB10 (spark-b, DeepSeek's four
// slots' state, 1.42 GB written back at 11.0 GB/s and read again at 14.5
// GB/s; docs/experiments/memory-pressure). Dropping it instead (no spill
// budget) costs its recomputation: its tokens at the model's
// recompute_ms_per_token.
constexpr double kSpillBytesPerSecond = 11.0e9;
constexpr double kRestoreBytesPerSecond = 14.5e9;
// Prepared weight page-in measured on the GB10 (D-055, memory-pressure).
constexpr double kWeightReadBytesPerSecond = 13.3e9;
// A graph's capture and instantiation a GiB counted, measured on a GB10
// (0.33–0.45 s in the services' logs): what a graph's capture weighs
// against before any graph is held (Server::Reclaim).
constexpr double kGraphSecondsPerGiB = 0.36;
// How often the driver's maintenance looks (between units and when idle).
constexpr auto kMaintenanceInterval = std::chrono::milliseconds(250);

// Diagnostic caps alone enable these clocks; ordinary serving adds no clock
// reads or scheduler calls. The driver owns every accumulator.
class SwapPhaseTimer {
 public:
  explicit SwapPhaseTimer(double* total)
      : total_(total), begin_(total != nullptr ? Clock::now() : Clock::time_point{}) {}
  SwapPhaseTimer(const SwapPhaseTimer&) = delete;
  SwapPhaseTimer& operator=(const SwapPhaseTimer&) = delete;
  ~SwapPhaseTimer() {
    if (total_ != nullptr) *total_ += Seconds(Clock::now() - begin_);
  }

 private:
  double* total_;
  Clock::time_point begin_;
};

std::uint64_t WeightHandle(catalog::ExtentId extent) {
  return (static_cast<std::uint64_t>(extent.generation()) << 32U) | extent.index();
}

catalog::ExtentId WeightExtent(std::uint64_t handle) {
  return {static_cast<std::uint32_t>(handle), static_cast<std::uint32_t>(handle >> 32U)};
}

std::string GraphJson(const GraphCounts& g) {
  return std::format(R"({{"eager":{},"captured":{},"replayed":{},"refused":{},"kept":{}}})",
                     g.eager, g.captured, g.replayed, g.refused, g.kept);
}

GraphCounts Sum(const engine::GraphStats& a, const engine::GraphStats& b, std::size_t kept) {
  return {.eager = a.eager + b.eager,
          .captured = a.captured + b.captured,
          .replayed = a.replayed + b.replayed,
          .refused = a.refused + b.refused,
          .kept = kept};
}

// ------------------------------------------- conversations kept (D-105)

// What this build is, for a kept record: its version and commit, and the
// executable file's identity, so another build, or this one rebuilt or
// reinstalled, never adopts what it did not write.
std::string KeptBuild() {
  const base::BuildInfo& info = base::GetBuildInfo();
  const auto stamp = platform::RunningExecutableStamp();
  const std::string exe =
      stamp ? std::format("{}:{}:{}:{}:{}", stamp->device, stamp->inode,
                          stamp->generation ? std::to_string(*stamp->generation) : "-", stamp->size,
                          stamp->modified_ns)
            : std::string("unknown");
  return std::format("{};{}{};{};{};exe={}", info.version, info.commit,
                     info.modified ? "+modified" : "", info.sdk, info.target, exe);
}

// ---------------------------------------------------------------- the models

// DeepSeek V4 Flash (engine/dsv4_runner.h), with DSpark as its drafter.
class Dsv4 final : public Llm {
 public:
  Dsv4(engine::PagedNode& node, const config::ModelEntry& entry, const ModelSettings& settings,
       const config::RuntimeRoles& roles, int index, std::optional<bool> plain_device_tokens,
       const ServingOptions& serving)
      : entry_(entry),
        artifact_id_(entry.artifact.value_or("")),
        drafter_id_(entry.drafter.value_or("")),
        store_(roles.installed),
        runner_(node, options_, index, static_cast<std::uint32_t>(index)) {
    name_ = entry.name;
    settings_ = settings;
    node_ = &node;
    checkpoint_directory_ = roles.spill;
    speculate_ = !drafter_id_.empty() && settings.speculation.value;
    // DSpark's drafts a step, and its verify's rows: the anchor and them.
    options_.draft_rows = settings.draft_rows.value;
    options_.max_verify = settings.draft_rows.value + 1;
    execution::AdaptiveWaveMode::Costs costs{};
    for (std::size_t w = 0; w < settings.wave_costs.value.size() && w + 2 < costs.size(); ++w) {
      costs[w + 2] = settings.wave_costs.value[w];
    }
    wave_mode_ = execution::AdaptiveWaveMode(costs, WaveForce(settings.wave_form.value),
                                             CutRows(options_.max_verify));
    context_ = settings.context.value;
    options_.artifact = roles.installed / artifact_id_;
    options_.out = roles.spill;
    options_.context = context_;
    configured_rows_ = settings.prefill_chunk.value;
    max_rows_ = PrefillChunkRows(context_, std::nullopt, settings.prefill_chunk.value,
                                 model::Dsv4MostRows(model::Dsv4Flash(), context_));
    options_.max_rows = max_rows_;
    options_.graphs = true;
    if (plain_device_tokens) options_.device_tokens = *plain_device_tokens;
    options_.prefill_lookahead = serving.dsv4_prefill_lookahead;
    options_.prepare_state = serving.dsv4_prepare_state;
    options_.state_only_prefill = serving.dsv4_state_only_prefill.value_or(!speculate_);
    prefill_hints_ = serving.predicted_prefill_hints;
    prefill_output_intent_ = serving.prefill_output_intent;
    // Request slots share the weights and workspace, each with its own
    // conversation state: concurrent chat requests decode in waves
    // (engine/dsv4_runner.h), whose row-local products read each weight
    // once for all of them. With DSpark each request's verify joins a wave
    // with at least two rows, so a wave of sixteen holds eight (the
    // settings' cap).
    options_.wave_slots = settings.max_slots.value;
    slots_report_ = SlotsReport(settings);
    // The output-A/HCA prefill: on full 4,096-row chunks, and on every
    // prefill chunk of 64 rows or more, where it passes every registered
    // quality control on both GGUFs under the tie-aware rule
    // (docs/experiments/ds4-output-prefix, "Default-on acceptance"): the
    // served defaults, which an owner may turn off for exactness.
    options_.prefill_outa_hca = settings.prefill_outa_hca.value;
    options_.prefill_outa_hca_partial =
        settings.prefill_outa_hca.value && settings.prefill_outa_hca_partial.value;
    if (speculate_) {
      options_.drafter = roles.installed / drafter_id_;
    }
  }

  engine::PagedModel& paged() override { return runner_; }

  Status Setup() override {
    if (auto opened = OpenArtifacts(); !opened) {
      return opened;
    }
    if (auto setup = runner_.Setup(); !setup) {
      return setup;
    }
    for (std::size_t i = 0; i < runner_.wave_capacity(); ++i) {
      auto slot = runner_.request_slot(i);
      if (!slot) {
        return std::unexpected(slot.error());
      }
      native_slots_[i] = *slot;
    }
    return PrepareBranches(runner_.wave_capacity(), 1);
  }

  Status OpenArtifacts() {
    auto artifact = OpenInstalled(store_, artifact_id_);
    if (!artifact) {
      return std::unexpected(artifact.error());
    }
    if (artifact->model().architecture != "deepseek4") {
      return Error(std::format("artifact {} is a {}, not DeepSeek V4 (deepseek4)", artifact_id_,
                               artifact->model().architecture));
    }
    auto binding = model::BindDsv4(model::Dsv4Flash(), *artifact);
    if (!binding) {
      return std::unexpected(binding.error());
    }
    options_.frontier_head = Dsv4FrontierHeadForServing(*binding);
    if (speculate_) {
      if (auto drafter = OpenInstalled(store_, drafter_id_); !drafter) {
        return std::unexpected(drafter.error());
      }
    }
    // The tokenizer and template from the artifact's kept GGUF metadata
    // (import rule 7): the first shard's.
    return UseChatAssets(*artifact, entry_);
  }
  std::uint64_t activations_needed() const override { return runner_.activations_needed(); }
  std::uint64_t pool_needed() const override { return runner_.pool_needed(); }
  std::uint64_t host_input_bytes() const override { return runner_.host_input_bytes(); }
  std::uint64_t plan_floor_bytes() const override { return runner_.plan_floor_bytes(); }
  std::uint64_t graph_measured_bytes() const override { return runner_.graph_measured_bytes(); }
  void ReclaimCandidates(std::uint32_t owner, bool running,
                         std::vector<memory::ReclaimCandidate>& out) override {
    runner_.ReclaimCandidates(owner, running, out);
  }
  std::uint64_t Reclaim(memory::ReclaimKind kind, std::uint64_t id) override {
    return runner_.Reclaim(kind, id);
  }
  std::string plan_report() const override { return runner_.plan_report(); }
  std::string serial_reason() const override { return runner_.serial_reason(); }
  std::string slots_report() const override { return slots_report_; }
  Status Register() override { return runner_.Register(); }
  Status Bind() override { return runner_.Bind(); }
  std::vector<catalog::ExtentId> weights() const override { return runner_.weights(); }
  std::vector<catalog::ExtentId> state() const override { return runner_.state(); }
  std::vector<catalog::ExtentId> kept_state() const override { return runner_.kept_state(); }
  std::vector<catalog::ExtentId> unchanged_state() const override {
    return runner_.unchanged_state();
  }
  void StateWrittenBack(bool whole) override { runner_.StateWrittenBack(whole); }
  const catalog::Closure& everything() const override { return runner_.everything(); }
  const catalog::Closure& request_closure() const override { return runner_.execution_closure(); }
  bool HasRetainedState() const override { return AnyBranchHasRetainedState(); }
  Status PrepareDefaultRequest() override {
    std::array<Branch*, 1> active{&default_branch()};
    return SelectBranches(active);
  }
  Status SelectBranches(std::span<Branch* const> active) override {
    if (active.size() > runner_.wave_capacity()) {
      return Error("too many DeepSeek native conversation branches");
    }
    std::array<engine::Dsv4Runner::Slot*, engine::Dsv4Runner::kRequestSlots> selected{};
    for (std::size_t i = 0; i < active.size(); ++i) {
      if (active[i] == nullptr || &active[i]->model() != this) {
        return Error("a selected DeepSeek conversation belongs to another model");
      }
      selected[i] = &NativeSlot(*active[i]);
    }
    return runner_.SelectSlots(std::span(selected).first(active.size()));
  }
  bool supports_generation_waves() const override { return runner_.waves_provisioned(); }
  std::size_t generation_wave_capacity() const override { return runner_.wave_capacity(); }
  std::uint64_t StateBytesThrough(std::uint32_t positions) const override {
    return runner_.StateBytesThrough(positions).value_or(0);
  }
  std::uint64_t weight_read_bytes() const override { return runner_.weight_read_bytes(); }
  std::uint32_t logit_width() const override { return runner_.vocab(); }
  Status AfterLoad() override { return runner_.CheckHashRouting(); }
  Status CheckPlaces() override { return runner_.CheckPlaces(); }
  void DropPlans() override { runner_.DropPlans(); }
  double plan_seconds() const override { return runner_.plan_seconds(); }
  GraphCounts graphs() const override {
    return Sum(runner_.graph_stats(), runner_.draft_stats(), runner_.graphs());
  }
  std::string violations() const override {
    return runner_.coverage_violations() == 0
               ? std::string()
               : std::format("{} bound tensors outside cataloged extents of their class; first {}",
                             runner_.coverage_violations(), runner_.first_violation());
  }
  std::string extra() const override {
    return std::format(
        R"({{"turn_checkpoints":{},"turn_checkpoint_bytes":{},"used_state_bytes":{},"architecture":"deepseek4","speculation":{},"coverage_tensors":{},)"
        R"("slab_padding":{},"state_bytes":{},"drafter_state_bytes":{},"device_raw_mask_plans":{},"draft_graph_steps":{},"graphs":{}}})",
        turn_checkpoints(), turn_checkpoint_bytes(), runner_.used_state_bytes(),
        speculate_ ? "\"dspark\"" : "null", runner_.coverage_tensors(), runner_.slab_padding(),
        runner_.state_bytes(), runner_.drafter_state_bytes(), runner_.bound_raw_masks(),
        runner_.draft_stats().eager + runner_.draft_stats().captured +
            runner_.draft_stats().replayed,
        GraphJson(graphs()));
  }
  void Defaults(chat::Conversation& c) const override {
    // As llama-server's /apply-template rendered the references' prompts:
    // its default turns the template's thinking on.
    if (!c.enable_thinking) {
      c.enable_thinking = true;
    }
  }

 protected:
  // The default conversation's scalar entry points forward to its branch.
  Status RunChunk(std::span<const std::int32_t> all, std::uint32_t n_past, bool inject,
                  std::vector<float>& logits) override {
    return RunChunkFor(default_branch(), all, n_past, inject, logits);
  }
  Status SpecStep(std::span<const std::int32_t> all, std::uint32_t pos, std::uint32_t left,
                  std::vector<std::int32_t>& kept, std::vector<std::vector<float>>* logits,
                  std::uint64_t& drafted) override {
    return SpecStepFor(default_branch(), all, pos, left, kept, logits, drafted);
  }
  Status Settle() override { return SettleFor(default_branch()); }
  Status ClearState() override { return ClearStateFor(default_branch()); }
  std::uint64_t target_state_base() const override { return TargetStateBaseFor(default_branch()); }
  std::uint64_t used_state_bytes() const override { return UsedStateBytesFor(default_branch()); }
  bool StateUsable() const override { return StateUsableFor(default_branch()); }
  Status PrepareDecodeState(std::uint32_t pos, std::uint32_t left) override {
    return PrepareDecodeStateFor(default_branch(), pos, left);
  }
  std::vector<engine::LiveState::Range> used_state_ranges() const override {
    return UsedStateRangesFor(default_branch());
  }
  Status SaveUsedState(void* host, std::span<const engine::LiveState::Range> ranges) override {
    return SaveUsedStateFor(default_branch(), host, ranges);
  }
  Status RestoreUsedState(void* host, std::span<const engine::LiveState::Range> ranges) override {
    return RestoreUsedStateFor(default_branch(), host, ranges);
  }
  std::expected<std::vector<engine::LiveState::Range>, std::string> CheckpointRanges(
      std::uint32_t positions) const override {
    return CheckpointRangesFor(default_branch(), positions);
  }
  Status PrepareRestoreState(std::span<const engine::LiveState::Range> footprint) override {
    return PrepareRestoreStateFor(default_branch(), footprint);
  }
  Status CopyCheckpointState(void* host, std::span<const engine::LiveState::Range> ranges,
                             bool to_host) override {
    return CopyCheckpointStateFor(default_branch(), host, ranges, to_host);
  }
  std::uint64_t target_state_bytes() const override {
    return TargetStateBytesFor(default_branch());
  }
  std::uint64_t drafter_state_base() const override {
    return DrafterStateBaseFor(default_branch());
  }
  std::uint64_t drafter_state_bytes() const override {
    return DrafterStateBytesFor(default_branch());
  }

  // Each conversation branch on its own native request slot.
  Status RunChunkFor(Branch& branch, std::span<const std::int32_t> all, std::uint32_t n_past,
                     bool inject, std::vector<float>& logits) override {
    return NativeSlot(branch).Chunk(
        n_past, all.subspan(n_past), logits,
        inject ? engine::Dsv4ChunkKind::kInject : engine::Dsv4ChunkKind::kPlain);
  }
  Status RunPrefillChunkFor(Branch& branch, std::span<const std::int32_t> all, std::uint32_t past,
                            bool inject, bool want_head, std::vector<float>& logits,
                            PrefillHint next) override {
    if (!prefill_hints_) next = {};
    if (!prefill_output_intent_) {
      want_head = true;
      next.want_head = true;
      next.after_want_head = true;
    }
    if (past >= all.size()) return Error("an empty DeepSeek prefill chunk");
    return NativeSlot(branch).PrefillChunk(
        past, all.subspan(past), logits,
        {.rows = next.rows,
         .want_head = next.want_head,
         .after_rows = next.after_rows,
         .after_want_head = next.after_want_head},
        inject ? engine::Dsv4ChunkKind::kInject : engine::Dsv4ChunkKind::kPlain, want_head);
  }
  std::optional<Status> RunGreedyChunkFor(Branch& branch, std::span<const std::int32_t> all,
                                          std::uint32_t pos, std::int32_t& token) override {
    if (!options_.device_tokens || speculate_) return std::nullopt;
    if (all.size() != std::size_t{pos} + 1)
      return Error("a DeepSeek greedy step needs one anchor row");
    return NativeSlot(branch).GreedyChunk(pos, all.subspan(pos), token);
  }
  Status SpecStepFor(Branch& branch, std::span<const std::int32_t> all, std::uint32_t pos,
                     std::uint32_t left, std::vector<std::int32_t>& kept,
                     std::vector<std::vector<float>>* logits, std::uint64_t& drafted,
                     bool* prefix_kept = nullptr) override {
    const std::uint32_t rows = VerifyRows(pos, left, runner_.max_verify());
    std::vector<std::int32_t> drafts;
    std::vector<float> verified;
    if (auto r = NativeSlot(branch).DraftVerify(pos, all.back(), rows, drafts, verified); !r) {
      return r;
    }
    return Undone(branch, Judge(branch, pos, rows, drafts, verified, kept, logits, drafted),
                  prefix_kept);
  }
  Status SettleFor(Branch& branch) override { return NativeSlot(branch).Rollback(); }
  Status ClearStateFor(Branch& branch) override { return NativeSlot(branch).Clear(); }
  // A conversation outside the cohort: its reuse cache cleared for peers
  // short of state capacity (Branch::ReleaseIdleState).
  Status ReleaseIdleStateFor(Branch& branch) override { return NativeSlot(branch).ClearIdle(); }
  Status SpillFor(Branch& branch) override { return NativeSlot(branch).Spill(); }
  Status RestoreFor(Branch& branch) override { return NativeSlot(branch).Restore(); }
  bool SpilledFor(const Branch& branch) const override { return NativeSlot(branch).spilled(); }
  std::uint64_t SpilledBytesFor(const Branch& branch) const override {
    return NativeSlot(branch).spilled_bytes();
  }
  std::uint64_t SpillWriteBytesFor(const Branch& branch) const override {
    return NativeSlot(branch).spill_write_bytes();
  }
  bool LeasedFor(const Branch& branch) const override { return NativeSlot(branch).held(); }
  std::uint64_t RefusedBytesFor(const Branch& branch) const override {
    const auto& slot = NativeSlot(branch);
    return slot.spilled() ? slot.spilled_bytes() : slot.refused_bytes();
  }
  bool StateRefusedFor(const Branch& branch) const override {
    return NativeSlot(branch).state_refused();
  }
  bool StateUsableFor(const Branch& branch) const override {
    return NativeSlot(branch).state_usable();
  }

 public:
  // Kept across a restart (D-105).
  std::string KeptLayout() const override { return runner_.kept_layout(); }
  std::vector<std::uint64_t> KeptRegions() const override {
    return KeptRegionsOf(NativeSlot(default_branch()).live(), true);
  }
  std::vector<std::uint64_t> KeptLayouts() const override {
    return KeptRegionsOf(NativeSlot(default_branch()).live(), false);
  }
  // D-102's rung 2.
  Status RecoverInPlace() override {
    auto discarded = runner_.RecoverInPlace([this](std::uint32_t slot) { UnkeepSlot(slot); });
    if (!discarded) {
      return std::unexpected(discarded.error());
    }
    ForgetDiscarded(*discarded);
    return {};
  }

 protected:
  const engine::LiveState* KeptLiveFor(const Branch& branch) const override {
    return &NativeSlot(branch).live();
  }
  bool KeptWholeFor(const Branch& branch) const override { return NativeSlot(branch).kept_whole(); }
  bool OwedFor(const Branch& branch) const override { return NativeSlot(branch).owed(); }
  Status AdoptFor(Branch& branch, std::span<const engine::LiveState::Range> used) override {
    return NativeSlot(branch).Adopt(used);
  }
  void SetSpillPlaces(
      const std::function<engine::LiveState::SpillPlace(std::uint32_t)>& place) override {
    options_.spill_place = place;
  }
  Status PrepareDecodeStateFor(Branch& branch, std::uint32_t pos, std::uint32_t left) override {
    const std::uint32_t rows = speculate_ ? VerifyRows(pos, left, runner_.max_verify()) : 1U;
    return NativeSlot(branch).ReserveStateThrough(pos + rows);
  }
  std::uint64_t TargetStateBaseFor(const Branch& branch) const override {
    return NativeSlot(branch).state_base();
  }
  std::uint64_t TargetStateBytesFor(const Branch& branch) const override {
    return NativeSlot(branch).state_bytes();
  }
  std::uint64_t DrafterStateBaseFor(const Branch& branch) const override {
    return NativeSlot(branch).drafter_state_base();
  }
  std::uint64_t DrafterStateBytesFor(const Branch& branch) const override {
    return NativeSlot(branch).drafter_state_bytes();
  }
  std::uint64_t UsedStateBytesFor(const Branch& branch) const override {
    return NativeSlot(branch).used_state_bytes();
  }
  std::vector<engine::LiveState::Range> UsedStateRangesFor(const Branch& branch) const override {
    return NativeSlot(branch).used_state_ranges();
  }
  Status SaveUsedStateFor(Branch& branch, void* host,
                          std::span<const engine::LiveState::Range> ranges) override {
    return NativeSlot(branch).SaveUsedState(host, ranges);
  }
  Status RestoreUsedStateFor(Branch& branch, void* host,
                             std::span<const engine::LiveState::Range> ranges) override {
    return NativeSlot(branch).RestoreUsedState(host, ranges);
  }
  std::expected<std::vector<engine::LiveState::Range>, std::string> CheckpointRangesFor(
      const Branch& branch, std::uint32_t positions) const override {
    return NativeSlot(branch).CheckpointRanges(positions);
  }
  Status PrepareRestoreStateFor(Branch& branch,
                                std::span<const engine::LiveState::Range> footprint) override {
    return NativeSlot(branch).PrepareRestoreState(footprint);
  }
  Status CopyCheckpointStateFor(Branch& branch, void* host,
                                std::span<const engine::LiveState::Range> ranges,
                                bool to_host) override {
    return NativeSlot(branch).CopyCheckpointState(host, ranges, to_host);
  }

  // DeepSeek keeps no speculation cursor and no adaptive draft depth: every
  // branch's are the defaults, saved and restored as no state.
  std::uint32_t CursorFor(const Branch& branch) const override {
    (void)BranchIndex(branch);
    return 0;
  }
  void SetCursorFor(Branch& branch, std::uint32_t /*value*/) override { (void)BranchIndex(branch); }
  void SaveDecodingStateFor(Branch& branch) override { SaveBranchDecoding(branch); }
  void RestoreDecodingStateFor(Branch& branch) override { RestoreBranchDecoding(branch); }
  execution::AdaptiveDepth TurnDecodingStateFor(const Branch& branch) const override {
    return BranchDecoding(branch);
  }
  void RestoreTurnDecodingStateFor(Branch& branch, const execution::AdaptiveDepth& state) override {
    BranchDecoding(branch) = state;
  }
  // A step keeps at most its verify's rows: the anchor and every DSpark
  // draft, not the branch's adaptive depth (which DeepSeek leaves at 1).
  std::uint32_t StepTokenBoundFor(const Branch& /*branch*/, std::uint32_t left) const override {
    return std::min(left, runner_.max_verify());
  }

  bool GenerationCohortUsable() const override { return runner_.cohort_usable(); }

  // Several branches' steps in one wave (engine/dsv4_runner.h): a decode
  // step of each, or with DSpark each one's draft and one joined verify of
  // every branch's rows (each branch's share of the wave's rows) or a
  // decode step of each, as AdaptiveWaveMode chooses. A lone branch keeps
  // its ordinary step.
  Status RunPreparedGenerationWave(std::span<PreparedGeneration> prepared) override {
    if (!runner_.waves_provisioned() || prepared.size() == 1) {
      return Llm::RunPreparedGenerationWave(prepared);
    }
    if (!speculate_) {
      const bool greedy =
          options_.device_tokens &&
          std::ranges::all_of(prepared, [](const auto& unit) { return DeviceGreedy(unit); });
      std::array<std::int32_t, engine::Dsv4Runner::kRequestSlots> chosen{};
      std::vector<engine::Dsv4Runner::WaveWork> work;
      work.reserve(prepared.size());
      for (PreparedGeneration& unit : prepared) {
        if (unit.step.all.size() != std::size_t{unit.step.position} + 1) {
          return Error("a DeepSeek decode wave step is not one anchor row");
        }
        work.push_back({.slot = &NativeSlot(*unit.branch),
                        .pos = unit.step.position,
                        .anchor = unit.step.all.back(),
                        .rows = 1,
                        .drafts = nullptr,
                        .logits = greedy ? nullptr : &unit.row,
                        .token = greedy ? &chosen[work.size()] : nullptr});
      }
      if (auto ran = runner_.DecodeWave(work); !ran) return ran;
      if (greedy)
        for (std::size_t i = 0; i < prepared.size(); ++i) prepared[i].chosen = chosen[i];
      return {};
    }
    // With DSpark, a wave of several requests is a draft-verify wave or a
    // plain decode wave (the drafter still fed), as the accepted tokens
    // against the measured cost of each width choose: never by wall time,
    // so the same waves choose the same forms (execution/adaptive_wave_mode.h).
    // A sampling member keeps every wave speculative.
    const auto width = static_cast<std::uint32_t>(prepared.size());
    const bool sampled = std::ranges::any_of(
        prepared, [this](const PreparedGeneration& unit) { return sampling(*unit.branch); });
    const auto mode = wave_cost_exploration_.Choose(wave_mode_.Choose(width, sampled), width,
                                                    sampled, settings_, calibration_record_.known);
    std::uint32_t tokens = 0;
    std::uint32_t complete = 0;
    const double planned = runner_.plan_seconds();
    const std::uint64_t captured = runner_.graph_stats().captured + runner_.draft_stats().captured;
    const auto started = Clock::now();
    auto waved = mode == execution::AdaptiveWaveMode::Mode::kPlain
                     ? PlainWave(prepared)
                     : SpeculativeWave(prepared, tokens, complete);
    const double seconds = Seconds(Clock::now() - started);
    // A sampling member's speculation accepts differently: not observed.
    if (waved && !sampled) {
      wave_mode_.Observe(width, mode, tokens, complete);
    }
    // Each form's wave time at this width: the calibration of its cost
    // (calibration.h), in force from the next start, never this service's.
    // Only steady waves count, matched between the forms: none that
    // planned or captured a graph (a shape's first waves), and a
    // draft-verify wave only when every member's full verify joined it
    // (not one cut by a mask width or the reply's end, or run alone).
    const bool speculative = mode == execution::AdaptiveWaveMode::Mode::kSpeculative;
    if (waved && runner_.plan_seconds() == planned &&
        runner_.graph_stats().captured + runner_.draft_stats().captured == captured &&
        (!speculative || complete == width)) {
      calibration_samples_.Wave(width, speculative, seconds);
    }
    return waved;
  }

 private:
  // A plain decode wave of a speculative model's requests: one row each,
  // its token chosen as an ordinary step's (greedy: the argmax). A failed
  // choice keeps the processed anchor, as an ordinary plain step's does.
  Status PlainWave(std::span<PreparedGeneration> prepared) {
    if (prepared.empty() || prepared.size() > engine::Dsv4Runner::kRequestSlots)
      return Error("a DeepSeek plain wave exceeds its bounded owner outputs");
    const bool greedy =
        options_.device_tokens &&
        std::ranges::all_of(prepared, [](const auto& unit) { return GreedyWithoutRows(unit); });
    std::array<std::int32_t, engine::Dsv4Runner::kRequestSlots> tokens{};
    std::vector<engine::Dsv4Runner::WaveWork> work;
    std::vector<std::vector<float>> rows(prepared.size());
    work.reserve(prepared.size());
    for (std::size_t i = 0; i < prepared.size(); ++i) {
      const PreparedGeneration& unit = prepared[i];
      if (unit.step.all.size() != std::size_t{unit.step.position} + 1) {
        return Error("a DeepSeek decode wave step is not one anchor row");
      }
      work.push_back({.slot = &NativeSlot(*unit.branch),
                      .pos = unit.step.position,
                      .anchor = unit.step.all.back(),
                      .rows = 1,
                      .drafts = nullptr,
                      .logits = greedy ? nullptr : &rows[i],
                      .token = greedy ? &tokens[i] : nullptr});
    }
    if (auto ran = runner_.DecodeWave(work); !ran) {
      return ran;
    }
    // DecodeWave has completed and validated every ID before any owner sees
    // one. These sessions still apply speculative results, with no draft kept.
    for (std::size_t i = 0; i < prepared.size(); ++i) {
      PreparedGeneration& unit = prepared[i];
      if (greedy) {
        unit.kept = {tokens[i]};
        continue;
      }
      auto chosen = Choose(*unit.branch, rows[i], std::uint64_t{unit.step.position} + 1);
      if (!chosen) {
        unit.result = std::unexpected(chosen.error());
        unit.anchor_processed = true;
        continue;
      }
      unit.kept = {*chosen};
      if (unit.step.need_logits) {
        unit.logits.push_back(std::move(rows[i]));
      }
    }
    return {};
  }

  // Each request's draft and one joined verify of every request's rows.
  // `tokens` and `complete`: the tokens kept by, and the count of, the
  // requests whose verify took its full rows in the joined wave (what the
  // wave form's acceptance average counts).
  Status SpeculativeWave(std::span<PreparedGeneration> prepared, std::uint32_t& tokens,
                         std::uint32_t& complete) {
    tokens = 0;
    complete = 0;
    std::vector<engine::Dsv4Runner::WaveWork> work;
    work.reserve(prepared.size());
    // These owners never move once the borrowed descriptors are made.
    struct Frame {
      std::uint32_t rows = 0;
      std::vector<std::int32_t> drafts;
      std::vector<float> verified;
    };
    std::vector<Frame> frames(prepared.size());
    // A verify of one row (a step ending at a mask width, or the last
    // token) runs alone: a wave of wider verifies would give it the
    // multi-row launch, not its own step's arithmetic. So does a lone one.
    const auto share = [](std::size_t count) { return WaveShare(count); };
    std::vector<std::size_t> joined;
    std::vector<std::size_t> alone;
    for (std::size_t i = 0; i < prepared.size(); ++i) {
      const PreparedGeneration& unit = prepared[i];
      const std::uint32_t rows = VerifyRows(unit.step.position, unit.step.left,
                                            std::min(runner_.max_verify(), share(prepared.size())));
      (rows == 1 ? alone : joined).push_back(i);
    }
    if (joined.size() == 1) {
      alone.push_back(joined.front());
      joined.clear();
    }
    for (const std::size_t i : alone) {
      PreparedGeneration& unit = prepared[i];
      unit.result = SpecStepFor(*unit.branch, unit.step.all, unit.step.position, unit.step.left,
                                unit.kept, unit.step.need_logits ? &unit.logits : nullptr,
                                unit.drafted, &unit.failed_prefix_valid);
      if (!unit.result && !runner_.cohort_usable()) {
        return std::unexpected(unit.result.error());
      }
    }
    if (joined.empty()) {
      return {};
    }
    const std::uint32_t full = std::min(runner_.max_verify(), share(joined.size()));
    for (const std::size_t i : joined) {
      PreparedGeneration& unit = prepared[i];
      frames[i].rows = VerifyRows(unit.step.position, unit.step.left, full);
      work.push_back({.slot = &NativeSlot(*unit.branch),
                      .pos = unit.step.position,
                      .anchor = unit.step.all.back(),
                      .rows = frames[i].rows,
                      .drafts = &frames[i].drafts,
                      .logits = &frames[i].verified});
    }
    if (auto ran = runner_.DraftVerifyWave(work); !ran) {
      return ran;
    }
    for (const std::size_t i : joined) {
      PreparedGeneration& unit = prepared[i];
      Frame& frame = frames[i];
      unit.result =
          Judge(*unit.branch, unit.step.position, frame.rows, frame.drafts, frame.verified,
                unit.kept, unit.step.need_logits ? &unit.logits : nullptr, unit.drafted);
      if (unit.result && frame.rows == full) {
        tokens += static_cast<std::uint32_t>(unit.kept.size());
        ++complete;
      }
      if (!unit.result) {
        // The shared wave completed; only this branch's judgement failed.
        // Undo its own verify before keeping its prefix.
        if (auto rolled = NativeSlot(*unit.branch).DiscardVerify(); !rolled) {
          if (!runner_.cohort_usable()) {
            return Error(
                std::format("{}; rolling back failed: {}", unit.result.error(), rolled.error()));
          }
          unit.result = Error(
              std::format("{}; rolling back failed: {}", unit.result.error(), rolled.error()));
        } else {
          unit.failed_prefix_valid = true;
        }
      }
    }
    return {};
  }

  // A failed judgement after the native verify completed: the verify is
  // undone, so the step's starting prefix still holds (`prefix_kept`, as a
  // wave's failed_prefix_valid); an undo that fails says so.
  Status Undone(Branch& branch, Status judged, bool* prefix_kept) {
    if (judged) {
      return judged;
    }
    if (auto rolled = NativeSlot(branch).DiscardVerify(); !rolled) {
      return Error(std::format("{}; rolling back failed: {}", judged.error(), rolled.error()));
    }
    if (prefix_kept != nullptr) {
      *prefix_kept = true;
    }
    return judged;
  }

  // The verify's rows from `pos`: the anchor and its drafts, within the
  // tokens left, `most`, the context and the steps' mask widths (D-092).
  std::uint32_t VerifyRows(std::uint32_t pos, std::uint32_t left, std::uint32_t most) const {
    auto rows = std::min<std::uint32_t>({options_.draft_rows + 1, most, left, context_ - pos});
    rows = std::max<std::uint32_t>(rows, 1);
    while (rows > 1 && !model::Dsv4SameWidths(runner_.state_layout(), pos, rows)) {
      --rows;
    }
    return rows;
  }
  // A completed verify's verdict for `branch`: accept drafts while the
  // target agrees (greedy: its argmax; sampling: speculative sampling's
  // verdict); the first disagreement, or the row after the last draft,
  // gives the next token. Row m predicts the token at pos + m + 1.
  Status Judge(Branch& branch, std::uint32_t pos, std::uint32_t rows,
               std::vector<std::int32_t>& drafts, std::span<const float> verified,
               std::vector<std::int32_t>& kept, std::vector<std::vector<float>>* logits,
               std::uint64_t& drafted) {
    const std::uint32_t vocab = runner_.vocab();
    if (rows == 0 || drafts.size() + 1 < rows || verified.size() != std::size_t{rows} * vocab) {
      return Error("a DeepSeek verify returned inconsistent draft/logit counts");
    }
    drafts.resize(rows - 1);
    const auto row = [&](std::uint32_t i) {
      return verified.subspan(std::size_t{i} * vocab, vocab);
    };
    std::uint32_t m = 0;
    std::int32_t next = -1;
    for (; m < rows - 1; ++m) {
      std::int32_t instead = -1;
      auto keep = Keep(branch, row(m), drafts[m], std::uint64_t{pos} + m + 1, instead);
      if (!keep) {
        return std::unexpected(keep.error());
      }
      if (!*keep) {
        next = instead;
        break;
      }
    }
    if (next < 0) {
      auto chosen = Choose(branch, row(m), std::uint64_t{pos} + m + 1);
      if (!chosen) {
        return std::unexpected(chosen.error());
      }
      next = *chosen;
    }
    if (auto r = NativeSlot(branch).Accept(m + 1); !r) {
      return r;
    }
    kept.assign(drafts.begin(), drafts.begin() + m);
    kept.push_back(next);
    if (logits != nullptr) {
      for (std::uint32_t i = 0; i <= m; ++i) {
        logits->emplace_back(row(i).begin(), row(i).end());
      }
    }
    drafted += rows - 1;
    return {};
  }
  engine::Dsv4Runner::Slot& NativeSlot(Branch& branch) {
    const auto index = BranchIndex(branch);
    base::Check(native_slots_[index] != nullptr, "native conversation slots are not ready");
    return *native_slots_[index];
  }
  const engine::Dsv4Runner::Slot& NativeSlot(const Branch& branch) const {
    const auto index = BranchIndex(branch);
    base::Check(native_slots_[index] != nullptr, "native conversation slots are not ready");
    return *native_slots_[index];
  }
  config::ModelEntry entry_;  // its tokenizer and template files, if named
  std::string artifact_id_;
  std::string drafter_id_;  // empty: none
  fs::path store_;
  std::string slots_report_;
  bool prefill_hints_ = true;
  bool prefill_output_intent_ = true;
  engine::Dsv4Options options_;  // before the runner, which keeps a reference
  engine::Dsv4Runner runner_;
  std::array<engine::Dsv4Runner::Slot*, engine::Dsv4Runner::kRequestSlots> native_slots_{};
  // With DSpark: draft-verify or plain waves, by width (RunPreparedGenerationWave),
  // as the settings' wave_costs (a draft-verify wave's time over a plain
  // decode wave's, by width; model_settings.h kDsv4WaveCosts measured them
  // with each form forced) and wave_form choose.
  // A request's share of a draft-verify wave's rows among `count`: all
  // sixteen's, at least two (so DSpark takes at most eight requests).
  static std::uint32_t WaveShare(std::size_t count) {
    return std::max<std::uint32_t>(
        2, static_cast<std::uint32_t>(engine::Dsv4Runner::kWaveRows / count));
  }
  // By width, a draft-verify wave's verify rows a request where its share
  // cuts them below the full verify's `most` (AdaptiveWaveMode::Rows).
  static execution::AdaptiveWaveMode::Rows CutRows(std::uint32_t most) {
    execution::AdaptiveWaveMode::Rows rows{};
    for (std::uint32_t w = 2; w <= execution::AdaptiveWaveMode::kMaxWidth; ++w) {
      rows[w] = WaveShare(w) < most ? WaveShare(w) : 0;
    }
    return rows;
  }
  // The model's configured wave form (wave_form): chosen, or forced for
  // exactness controls.
  static execution::AdaptiveWaveMode::Force WaveForce(config::WaveForm form) {
    switch (form) {
      case config::WaveForm::kSpeculative:
        return execution::AdaptiveWaveMode::Force::kSpeculative;
      case config::WaveForm::kPlain:
        return execution::AdaptiveWaveMode::Force::kPlain;
      case config::WaveForm::kAuto:
        break;
    }
    return execution::AdaptiveWaveMode::Force::kNone;
  }
  execution::AdaptiveWaveMode wave_mode_;
  WaveCostExploration wave_cost_exploration_;
};

// Approved Gemma26/31 profiles share the serving driver and native slots.
// Ordinary Gemma31 and Gemma26 use their qualified recipes within their bounded
// envelopes; larger configurations and explicit arithmetic diagnostics retain
// their own paths.
class Gemma final : public Llm {
 public:
  Gemma(engine::PagedNode& node, const config::ModelEntry& entry, const ModelSettings& settings,
        const config::RuntimeRoles& roles, int index, engine::Gemma4Variant variant,
        const ServingOptions& serving)
      : entry_(entry),
        artifact_id_(entry.artifact.value_or("")),
        store_(roles.installed),
        profile_(variant == engine::Gemma4Variant::k26BA4B ? model::Gemma4_26BA4B()
                                                           : model::Gemma4_31B()),
        options_(Options(entry, settings, roles, variant, serving.gemma_row_invariant,
                         serving.gemma4_prepare_state)),
        joined_(serving.gemma_joined ||
                (variant == engine::Gemma4Variant::k31B && settings.gemma31_production) ||
                (variant == engine::Gemma4Variant::k26BA4B && settings.gemma26_production)),
        runner_(node, options_, index, static_cast<std::uint32_t>(index)) {
    name_ = entry.name;
    settings_ = settings;
    node_ = &node;
    context_ = options_.context;
    max_rows_ = options_.max_rows;
    configured_rows_ = settings.prefill_chunk.value;
    checkpoint_directory_ = roles.spill;
  }
  engine::PagedModel& paged() override { return runner_; }
  Status Setup() override {
    if (entry_.drafter || settings_.speculation.value)
      return Error("Gemma serving has no qualified assistant or speculative path");
    auto artifact = OpenInstalled(store_, artifact_id_);
    if (!artifact) return Error(artifact.error());
    auto binding = model::BindGemma4(profile_, *artifact);
    if (!binding) return Error(binding.error());
    if (auto r = UseChatAssets(*artifact, entry_); !r) return r;
    // This first serving slice has no generated thought/tool channel parser.
    // Plain chat stops before a channel control can become answer text.
    for (const auto text : {"<|channel>", "<|tool_call>"})
      if (const auto token = tokenizer_->Find(text)) chat_stops_.push_back(*token);
    if (auto r = runner_.Setup(); !r) return r;
    for (std::uint32_t i = 0; i < options_.slots; ++i) {
      auto slot = runner_.request_slot(i);
      if (!slot) return Error(slot.error());
      slots_[i] = *slot;
    }
    return PrepareBranches(options_.slots, 1);
  }
  Status Register() override { return runner_.Register(); }
  Status Bind() override { return runner_.Bind(); }
  std::uint64_t activations_needed() const override { return runner_.activations_needed(); }
  std::uint64_t pool_needed() const override { return runner_.pool_needed(); }
  std::uint64_t host_input_bytes() const override {
    // The startup guard sets this bounded heap workspace apart before any
    // scalar output or per-owner sampling vector grows. One retained frontier
    // and one prepared result per owner, plus the sampling candidate capacity.
    // The independently catalog-backed pinned output holds one row per slot.
    const auto heap = std::uint64_t{options_.slots} * profile_.vocab *
                      (2 * sizeof(float) + 2 * sizeof(execution::SamplingCandidate));
    return runner_.host_input_bytes() + heap;
  }
  std::uint64_t plan_floor_bytes() const override { return runner_.plan_floor_bytes(); }
  std::uint64_t graph_measured_bytes() const override { return runner_.graph_measured_bytes(); }
  std::vector<catalog::ExtentId> weights() const override { return runner_.weights(); }
  std::vector<catalog::ExtentId> state() const override { return runner_.state(); }
  std::vector<catalog::ExtentId> kept_state() const override { return runner_.kept_state(); }
  void StateWrittenBack(bool whole) override { runner_.StateWrittenBack(whole); }
  const catalog::Closure& everything() const override { return runner_.everything(); }
  const catalog::Closure& request_closure() const override { return runner_.closure(); }
  std::uint64_t weight_read_bytes() const override { return runner_.weight_read_bytes(); }
  std::uint32_t logit_width() const override { return profile_.vocab; }
  bool HasRetainedState() const override { return AnyBranchHasRetainedState(); }
  Status CheckPlaces() override { return runner_.CheckPlaces(); }
  void DropPlans() override { runner_.DropPlans(); }
  void ReclaimCandidates(std::uint32_t owner, bool running,
                         std::vector<memory::ReclaimCandidate>& out) override {
    runner_.ReclaimCandidates(owner, running, out);
  }
  std::uint64_t Reclaim(memory::ReclaimKind kind, std::uint64_t id) override {
    return runner_.Reclaim(kind, id);
  }
  GraphCounts graphs() const override {
    const auto& g = runner_.graph_stats();
    return {g.eager, g.captured, g.replayed, g.refused, runner_.graph_count()};
  }
  std::string violations() const override { return runner_.coverage().first_violation; }
  std::string extra() const override {
    const auto& selected = runner_.last_built_policy();
    return std::format(
        R"({{"architecture":"gemma4","dispatch":"{}","optional_optimizations":{},"row_invariant":{},"profile":"{}","coverage_tensors":{},"pitch_padding":{},"joined_groups":{},"joined_units":{},"gemma31_candidate":{},"gemma26_candidate":{},"max_rows":{},"norm_rope_requested":{},"norm_add_requested":{},"owner_attention_requested":{},"last_built_rows":{},"last_built_segments":{},"last_built_norm_rope":{},"last_built_norm_add":{},"last_built_owner_attention":{},"common_owner_reads_requested":{},"bounded_owner_roots_requested":{},"max_bound_bounded_owner_attention":{}}})",
        settings_.gemma31_production || settings_.gemma26_production ? "joined-candidate"
        : joined_                                                    ? "joined-diagnostic"
                                                                     : "scalar-cohort",
        joined_ || options_.row_invariant, options_.row_invariant, profile_.name,
        runner_.coverage().tensors, runner_.pitch_padding(), joined_groups_, joined_units_,
        settings_.gemma31_production, settings_.gemma26_production, options_.max_rows,
        options_.fuse_norm_rope, options_.fuse_norm_add, options_.owner_attention, selected.rows,
        selected.segments, selected.norm_rope, selected.norm_add, selected.owner_attention_steps,
        options_.common_owner_reads, options_.bounded_owner_roots,
        max_bound_bounded_owner_attention_);
  }
  std::string slots_report() const override { return SlotsReport(settings_); }
  std::string KeptLayout() const override { return runner_.CheckpointLayoutId(); }
  std::vector<std::uint64_t> KeptRegions() const override {
    return KeptRegionsOf(NativeSlot(default_branch()).state(), true);
  }
  std::vector<std::uint64_t> KeptLayouts() const override {
    return KeptRegionsOf(NativeSlot(default_branch()).state(), false);
  }
  Status PrepareDefaultRequest() override {
    std::array<Branch*, 1> selected{&default_branch()};
    return SelectBranches(selected);
  }
  Status SelectBranches(std::span<Branch* const> active) override {
    if (active.size() > options_.slots) return Error("too many Gemma conversation owners");
    std::array<std::uint32_t, engine::kMaxRequestSlots> ids{};
    for (std::size_t i = 0; i < active.size(); ++i) {
      if (!active[i] || &active[i]->model() != this) return Error("foreign Gemma conversation");
      ids[i] = BranchIndex(*active[i]);
    }
    return runner_.SelectSlots(std::span(ids).first(active.size()));
  }
  std::span<const std::int32_t> ChatStops() const override { return chat_stops_; }
  bool supports_generation_waves() const override { return true; }
  std::size_t generation_wave_capacity() const override { return options_.slots; }
  std::uint64_t StateBytesThrough(std::uint32_t positions) const override {
    const auto ranges = runner_.CheckpointRanges(positions);
    if (!ranges || !slots_[0]) return 0;
    return slots_[0]->state().UsedBytesOf(*ranges).value_or(0);
  }
  void Defaults(chat::Conversation& c) const override {
    if (!c.enable_thinking) c.enable_thinking = false;
  }

 protected:
  Status CheckConversation(const chat::Conversation& c) const override {
    if (c.enable_thinking.value_or(false))
      return Error("Gemma thought-channel output is not qualified in this serving slice");
    if (!c.tools.empty())
      return Error("Gemma generated tool-call output is not qualified in this serving slice");
    return {};
  }
  Status PrepareSamplingScratchFor(Branch& branch, std::size_t logits) override {
    const auto vocab = profile_.vocab;
    if (logits != vocab) return Error("Gemma sampling needs its complete target vocabulary");
    // TopK may reserve 2*k. Reserve the complete bound from empty storage
    // before the first sample, so subsequent rows never reallocate it.
    ReserveBranchSamplingScratch(branch, 2 * std::size_t{vocab});
    return {};
  }
  void RetireSamplingScratchFor(Branch& branch) override { DropBranchSamplingScratch(branch); }
  bool GenerationCohortUsable() const override { return runner_.cohort_usable(); }
  Status RunPreparedGenerationWave(std::span<PreparedGeneration> units) override {
    // Greedy waves take their tokens from the device: no 1 MiB row back, no
    // host scan. A wave is all one kind (the runner publishes one or the other).
    const bool greedy =
        !units.empty() && std::ranges::all_of(units, [](const auto& u) { return DeviceGreedy(u); });
    if (!joined_ || units.size() == 1) return RunScalarGenerationUnits(units);
    if (greedy)
      for (auto& unit : units) unit.chosen = 0;
    if (units.empty() || units.size() > options_.slots)
      return Error("Gemma joined generation exceeds its funded owner envelope");
    std::array<engine::Gemma4Runner::Work, engine::kMaxRequestSlots> work{};
    std::array<bool, engine::kMaxRequestSlots> seen{};
    for (std::size_t i = 0; i < units.size(); ++i) {
      const auto& unit = units[i];
      if (unit.branch == nullptr || &unit.branch->model() != this || unit.step.speculative ||
          unit.step.all.size() != std::size_t{unit.step.position} + 1)
        return Error("Gemma joined generation needs plain owned one-anchor units");
      const auto id = BranchIndex(*unit.branch);
      if (id >= options_.slots || seen[id] ||
          NativeSlot(*unit.branch).completed_positions() != unit.step.position)
        return Error("Gemma joined generation needs distinct current native cursors");
      seen[id] = true;
      work[i] = {id, unit.step.position, unit.step.all.last(1), greedy ? nullptr : &units[i].row,
                 greedy ? &*units[i].chosen : nullptr};
    }
    // Checked one-row sums keep their eight-column limit. Ordinary products
    // execute the complete admitted cohort without duplicate model passes.
    return RunGemmaGroups(
        units.size(), engine::Gemma4WaveRows(options_.row_invariant),
        [&](std::size_t first, std::size_t count) {
          auto ran = runner_.Wave(std::span(work).subspan(first, count));
          if (ran) {
            ++joined_groups_;
            joined_units_ += count;
            max_bound_bounded_owner_attention_ =
                std::max(max_bound_bounded_owner_attention_,
                         runner_.last_built_policy().bounded_owner_steps);
          }
          return ran;
        },
        [&] { return GenerationCohortUsable(); },
        [&](std::size_t first, std::size_t count, const std::string& error) {
          for (auto& unit : units.subspan(first, count)) {
            unit.result = std::unexpected(error);
            unit.failed_prefix_valid = StateUsableFor(*unit.branch);
          }
        });
  }
  Status RunChunkFor(Branch& branch, std::span<const std::int32_t> all, std::uint32_t past,
                     bool inject, std::vector<float>& logits) override {
    if (inject || past >= all.size()) return Error("Gemma needs a plain nonempty chunk");
    const engine::Gemma4Runner::Work work{BranchIndex(branch), past, all.subspan(past), &logits};
    return runner_.Wave(std::span(&work, 1));
  }
  std::optional<Status> RunGreedyChunkFor(Branch& branch, std::span<const std::int32_t> all,
                                          std::uint32_t past, std::int32_t& token) override {
    if (past >= all.size()) return Error("Gemma needs a plain nonempty chunk");
    const engine::Gemma4Runner::Work work{BranchIndex(branch), past, all.subspan(past), nullptr,
                                          &token};
    return runner_.Wave(std::span(&work, 1));
  }
  Status RunPrefillChunkFor(Branch& branch, std::span<const std::int32_t> all, std::uint32_t past,
                            bool inject, bool want_head, std::vector<float>& logits,
                            PrefillHint next) override {
    if (inject || past >= all.size()) return Error("Gemma needs a plain nonempty prefill chunk");
    const engine::Gemma4Runner::Work work{BranchIndex(branch), past, all.subspan(past), &logits};
    const engine::Gemma4Runner::PrefillNext hint{work.slot, next.rows, next.after_rows};
    return runner_.WavePrefill(std::span(&work, 1), want_head,
                               std::span(&hint, next.rows == 0 ? 0U : 1U), next.want_head,
                               next.after_want_head);
  }
  Status SettleFor(Branch& branch) override {
    return NativeSlot(branch).state_usable() ? Status{} : Error("Gemma state is quarantined");
  }
  Status ClearStateFor(Branch& branch) override { return runner_.Clear(BranchIndex(branch)); }
  bool StateUsableFor(const Branch& branch) const override {
    return NativeSlot(branch).state_usable();
  }
  bool StateRefusedFor(const Branch& branch) const override {
    return NativeSlot(branch).refused_state_growth();
  }
  Status PrepareDecodeStateFor(Branch& branch, std::uint32_t pos, std::uint32_t) override {
    if (pos != NativeSlot(branch).completed_positions() || pos >= context_)
      return Error("Gemma decode position differs from its completed state");
    return runner_.ReserveStateThrough(BranchIndex(branch), pos + 1);
  }
  std::uint64_t TargetStateBaseFor(const Branch& branch) const override {
    return NativeSlot(branch).state().base(0);
  }
  std::uint64_t TargetStateBytesFor(const Branch& branch) const override {
    return NativeSlot(branch).state().bytes(0);
  }
  std::uint64_t DrafterStateBaseFor(const Branch&) const override { return 0; }
  std::uint64_t DrafterStateBytesFor(const Branch&) const override { return 0; }
  std::uint64_t UsedStateBytesFor(const Branch& branch) const override {
    return NativeSlot(branch).used_state_bytes();
  }
  std::vector<engine::LiveState::Range> UsedStateRangesFor(const Branch& branch) const override {
    return NativeSlot(branch).state().used_ranges();
  }
  Status SaveUsedStateFor(Branch& branch, void* host,
                          std::span<const engine::LiveState::Range> ranges) override {
    return runner_.CopyState(BranchIndex(branch), host, ranges, true);
  }
  Status RestoreUsedStateFor(Branch&, void*, std::span<const engine::LiveState::Range>) override {
    return Error("Gemma restore requires owned logical-position metadata");
  }
  std::expected<std::vector<engine::LiveState::Range>, std::string> CheckpointRangesFor(
      const Branch& branch, std::uint32_t pos) const override {
    if (pos != NativeSlot(branch).completed_positions())
      return Error("Gemma checkpoint boundary differs from completed state");
    return runner_.CheckpointRanges(pos);
  }
  Status PrepareRestoreStateFor(Branch&, std::span<const engine::LiveState::Range>) override {
    return Error("Gemma restore requires owned logical-position metadata");
  }
  Status CheckCheckpointMetadataFor(
      const Branch&, std::uint32_t pos, std::uint32_t cursor,
      std::span<const engine::LiveState::Range> ranges) const override {
    if (pos != cursor) return Error("Gemma checkpoint cursor differs from its logical boundary");
    return runner_.ValidateFootprint(pos, ranges);
  }
  Status PreparePositionRestoreFor(Branch& branch, std::uint32_t pos, std::uint32_t cursor,
                                   std::span<const engine::LiveState::Range> ranges,
                                   std::string_view /*layout*/) override {
    if (auto r = CheckCheckpointMetadataFor(branch, pos, cursor, ranges); !r) return r;
    // Llm checks the retained record's layout against KeptLayout before
    // adoption; local snapshots are produced by this same runner.
    return runner_.PrepareRestore(BranchIndex(branch), pos, ranges, KeptLayout());
  }
  Status CompletePositionRestoreFor(Branch& branch, std::uint32_t pos) override {
    return runner_.CompleteRestore(BranchIndex(branch), pos);
  }
  Status CopyCheckpointStateFor(Branch& branch, void* host,
                                std::span<const engine::LiveState::Range> ranges,
                                bool to_host) override {
    return runner_.CopyState(BranchIndex(branch), host, ranges, to_host);
  }
  Status RestoreSnapshotFor(Branch& branch, void* host,
                            std::span<const engine::LiveState::Range> ranges,
                            std::uint32_t pos) override {
    if (auto r = PreparePositionRestoreFor(branch, pos, pos, ranges, KeptLayout()); !r) return r;
    if (auto r = CopyCheckpointStateFor(branch, host, ranges, false); !r) return r;
    return CompletePositionRestoreFor(branch, pos);
  }
  void SaveDecodingStateFor(Branch& branch) override { SaveBranchDecoding(branch); }
  void RestoreDecodingStateFor(Branch& branch) override { RestoreBranchDecoding(branch); }
  execution::AdaptiveDepth TurnDecodingStateFor(const Branch& branch) const override {
    return BranchDecoding(branch);
  }
  void RestoreTurnDecodingStateFor(Branch& branch, const execution::AdaptiveDepth& state) override {
    BranchDecoding(branch) = state;
  }
  std::uint32_t CursorFor(const Branch& branch) const override {
    return NativeSlot(branch).completed_positions();
  }
  void SetCursorFor(Branch&, std::uint32_t) override {}  // checked restore/adopt publishes it
  const engine::LiveState* KeptLiveFor(const Branch& branch) const override {
    return &NativeSlot(branch).state();
  }
  bool KeptWholeFor(const Branch& branch) const override { return NativeSlot(branch).kept_whole(); }
  Status AdoptPositionFor(Branch& branch, std::span<const std::int32_t> tokens, std::uint32_t pos,
                          std::span<const engine::LiveState::Range> ranges,
                          std::string_view /*layout*/) override {
    if (pos != tokens.size() || pos == 0)
      return Error("Gemma kept positions differ from its owned token history");
    return runner_.Adopt(BranchIndex(branch), pos, ranges, KeptLayout());
  }
  void SetSpillPlaces(
      const std::function<engine::LiveState::SpillPlace(std::uint32_t)>& place) override {
    runner_.SetSpillPlaces(place);
  }
  Status ReleaseIdleStateFor(Branch& branch) override {
    return runner_.ClearIdle(BranchIndex(branch));
  }
  Status SpillFor(Branch& branch) override { return runner_.Spill(BranchIndex(branch)); }
  Status RestoreFor(Branch& branch) override { return runner_.Restore(BranchIndex(branch)); }
  bool SpilledFor(const Branch& branch) const override { return NativeSlot(branch).is_spilled(); }
  std::uint64_t SpilledBytesFor(const Branch& branch) const override {
    return NativeSlot(branch).spilled_bytes();
  }
  bool LeasedFor(const Branch& branch) const override { return runner_.Held(BranchIndex(branch)); }
  std::uint64_t RefusedBytesFor(const Branch& branch) const override {
    return NativeSlot(branch).refused_bytes();
  }
  Status RunChunk(std::span<const std::int32_t> all, std::uint32_t past, bool inject,
                  std::vector<float>& logits) override {
    return RunChunkFor(default_branch(), all, past, inject, logits);
  }
  Status SpecStep(std::span<const std::int32_t>, std::uint32_t, std::uint32_t,
                  std::vector<std::int32_t>&, std::vector<std::vector<float>>*,
                  std::uint64_t&) override {
    return Error("Gemma speculation is unavailable");
  }
  Status Settle() override { return SettleFor(default_branch()); }
  Status ClearState() override { return ClearStateFor(default_branch()); }
  bool StateUsable() const override { return StateUsableFor(default_branch()); }
  Status PrepareDecodeState(std::uint32_t pos, std::uint32_t left) override {
    return PrepareDecodeStateFor(default_branch(), pos, left);
  }
  std::uint64_t target_state_base() const override { return TargetStateBaseFor(default_branch()); }
  std::uint64_t target_state_bytes() const override {
    return TargetStateBytesFor(default_branch());
  }
  std::uint64_t drafter_state_base() const override { return 0; }
  std::uint64_t drafter_state_bytes() const override { return 0; }
  std::uint64_t used_state_bytes() const override { return UsedStateBytesFor(default_branch()); }
  std::vector<engine::LiveState::Range> used_state_ranges() const override {
    return UsedStateRangesFor(default_branch());
  }
  Status SaveUsedState(void* host, std::span<const engine::LiveState::Range> ranges) override {
    return SaveUsedStateFor(default_branch(), host, ranges);
  }
  Status RestoreUsedState(void*, std::span<const engine::LiveState::Range>) override {
    return Error("Gemma restore requires owned metadata");
  }
  std::expected<std::vector<engine::LiveState::Range>, std::string> CheckpointRanges(
      std::uint32_t pos) const override {
    return runner_.CheckpointRanges(pos);
  }
  Status PrepareRestoreState(std::span<const engine::LiveState::Range>) override {
    return Error("Gemma restore requires owned metadata");
  }
  Status CopyCheckpointState(void* host, std::span<const engine::LiveState::Range> ranges,
                             bool to_host) override {
    return CopyCheckpointStateFor(default_branch(), host, ranges, to_host);
  }

 private:
  static engine::Gemma4Options Options(const config::ModelEntry& entry,
                                       const ModelSettings& settings,
                                       const config::RuntimeRoles& roles,
                                       engine::Gemma4Variant variant, bool row_invariant,
                                       bool prepare_state) {
    const bool candidate = variant == engine::Gemma4Variant::k31B && settings.gemma31_production;
    const bool candidate26 =
        variant == engine::Gemma4Variant::k26BA4B && settings.gemma26_production;
    const bool bounded = (candidate || candidate26) && settings.context.value <= 4096 &&
                         settings.max_slots.value == 2;
    return {.artifact = roles.installed / entry.artifact.value_or(""),
            .out = roles.spill,
            .variant = variant,
            .context = settings.context.value,
            .max_rows = settings.prefill_chunk.value,
            .slots = settings.max_slots.value,
            .max_head_rows = settings.max_slots.value,
            // Stock Gemma4 gathers after the full final FFN. Narrowing earlier
            // changes quantized product arithmetic even for one published head.
            .frontier_head = !(candidate || candidate26),
            .row_invariant = row_invariant,
            .fuse_norm_rope = candidate || candidate26,
            .fuse_norm_add = candidate || candidate26,
            .fuse_gemma_route = candidate26,
            .fuse_gemma_reduce = candidate26,
            .fuse_quant_glu = candidate,
            .prepare_state = prepare_state,
            .owner_attention = candidate || candidate26,
            .common_owner_reads = bounded,
            .bounded_owner_roots = bounded};
  }
  engine::Gemma4Runner::Slot& NativeSlot(const Branch& branch) const {
    return *slots_[BranchIndex(branch)];
  }
  std::vector<std::int32_t> chat_stops_;
  config::ModelEntry entry_;
  std::string artifact_id_;
  fs::path store_;
  const model::Gemma4Profile& profile_;
  engine::Gemma4Options options_;
  const bool joined_;
  std::uint64_t joined_groups_ = 0;
  std::uint64_t joined_units_ = 0;
  std::uint32_t max_bound_bounded_owner_attention_ = 0;
  engine::Gemma4Runner runner_;
  std::array<engine::Gemma4Runner::Slot*, engine::kMaxRequestSlots> slots_{};
};

// Qwen3.8 Flash Next (engine/qwen38_runner.h), with its MTP block as its
// drafter.
class Qwen38 final : public Llm {
 public:
  Qwen38(engine::PagedNode& node, const config::ModelEntry& entry, const ModelSettings& settings,
         const config::RuntimeRoles& roles, int index, std::optional<bool> plain_device_tokens,
         const ServingOptions& serving)
      : entry_(entry),
        artifact_id_(entry.artifact.value_or("")),
        drafter_id_(entry.drafter.value_or("")),
        store_(roles.installed),
        runner_(node, options_, index, static_cast<std::uint32_t>(index)) {
    name_ = entry.name;
    settings_ = settings;
    node_ = &node;
    speculate_ = !drafter_id_.empty() && settings.speculation.value;
    checkpoint_directory_ = roles.spill;
    context_ = settings.context.value;
    options_.artifact = roles.installed / artifact_id_;
    options_.out = roles.spill;
    options_.context = context_;
    configured_rows_ = settings.prefill_chunk.value;
    // Device QSA selection avoids the dense matrix; short causal matrices
    // are graph-owned. RE-037's dense-consumer bound still applies wherever
    // such a matrix is needed, independently of host materialization.
    max_rows_ = PrefillChunkRows(context_, std::nullopt, settings.prefill_chunk.value,
                                 model::Qwen38MostRows(context_, false));
    options_.max_rows = max_rows_;
    options_.graphs = true;
    if (plain_device_tokens) options_.device_tokens = *plain_device_tokens;
    options_.prefill_lookahead = serving.qwen38_prefill_lookahead;
    options_.prepare_state = serving.qwen38_prepare_state;
    options_.state_only_prefill = serving.qwen38_state_only_prefill.value_or(!speculate_);
    prefill_hints_ = serving.predicted_prefill_hints;
    prefill_output_intent_ = serving.prefill_output_intent;
    // Request slots share weights and workspace while each branch retains
    // its own native state: the requests decode in one wave
    // (engine/qwen38_wave_plan.h), whose row-local products read each weight
    // once for a group of up to sixteen rows (C2/C4 HTTP screens: +9.8%/+20%
    // over two slots of pairs).
    options_.wave_slots = settings.max_slots.value;
    options_.request_slots = options_.wave_slots;
    slots_report_ = SlotsReport(settings);
    options_.wave_read_align = settings.wave_read_align.value;
    options_.wave_lanes = settings.wave_lanes.value;
    depth_cost_ratio_ = settings.depth_cost_ratio.value;
    shared_wave_depth_ = settings.shared_wave_depth.value;
    draft_wave_max_ = settings.draft_wave_max.value;
    if (speculate_) {
      options_.drafter = roles.installed / drafter_id_;
      // A verify takes the anchor and the drafts: a chunk of fewer than
      // four rows holds two drafts at most.
      options_.draft_rows =
          max_rows_ >= 4 ? settings.draft_rows.value : std::min(settings.draft_rows.value, 2U);
      options_.draft_vocab = settings.draft_vocab.value;
    }
  }

  engine::PagedModel& paged() override { return runner_; }

  std::string allocation_report() const override {
    const auto b = runner_.setup_budget();
    const auto workspace = std::format(
        "\"scalar_activation_bytes\":{},\"provisioned_activation_bytes\":{},"
        "\"scalar_scratch_bytes\":{},\"provisioned_scratch_bytes\":{},"
        "\"scalar_input_staging_bytes\":{},\"provisioned_input_staging_bytes\":{},"
        "\"scalar_host_input_bytes\":{},\"provisioned_host_input_bytes\":{}",
        b.scalar_activations, b.activations, b.scalar_scratch, b.scratch, b.scalar_staging_inputs,
        b.staging_inputs, b.scalar_host_inputs, b.host_inputs);
    const auto fixed = std::format(
        "\"scalar_ple_mapped_bytes\":{},\"provisioned_ple_mapped_bytes\":{},"
        "\"scalar_pinned_bytes\":{},\"provisioned_pinned_bytes\":{},"
        "\"wave_output_pinned_bytes\":{},\"scalar_snapshot_per_branch_bytes\":{},"
        "\"provisioned_snapshot_per_branch_bytes\":{},\"runner_mapped_bytes\":{}",
        b.scalar_ple_mapped, b.ple_mapped, b.scalar_pinned, b.pinned, b.wave_output_pinned,
        b.scalar_snapshot_per_branch, b.snapshot_per_branch, b.runner_mapped);
    const auto state = std::format(
        "\"target_layout_per_branch_bytes\":{},\"drafter_layout_per_branch_bytes\":{},"
        "\"virtual_extent_per_branch_bytes\":{},\"initialized_logical_bytes\":{},"
        "\"initialized_extent_bytes\":{}",
        b.target_per_branch, b.drafter_per_branch, b.virtual_per_branch, b.initialized_logical,
        b.initialized_extent_bytes);
    return std::format(
        "{{\"format\":\"llmp-qwen-wave-setup-budget-v1\",\"context\":{},"
        "\"prefill_rows\":{},\"wave_capacity\":{},\"branch_count\":{},"
        "\"speculative\":{},\"draft_rows\":{},{},{},{}}}",
        context_, max_rows_, runner_.wave_capacity(), branches(), speculate_, options_.draft_rows,
        workspace, fixed, state);
  }

  Status Setup() override {
    auto artifact = OpenInstalled(store_, artifact_id_);
    if (!artifact) {
      return std::unexpected(artifact.error());
    }
    if (artifact->model().architecture != "qwen4exp") {
      return Error(std::format("artifact {} is a {}, not Qwen3.8 Flash Next (qwen4exp)",
                               artifact_id_, artifact->model().architecture));
    }
    if (speculate_) {
      if (auto drafter = OpenInstalled(store_, drafter_id_); !drafter) {
        return std::unexpected(drafter.error());
      }
    }
    // The tokenizer and template: kept in the artifact's metadata, or (M3's
    // import kept config.json only) the checkpoint's, which the
    // configuration names.
    if (auto used = UseChatAssets(*artifact, entry_); !used) {
      return used;
    }
    if (auto setup = runner_.Setup(); !setup) {
      return setup;
    }
    for (std::size_t i = 0; i < runner_.request_slots(); ++i) {
      auto slot = runner_.request_slot(i);
      if (!slot) {
        return std::unexpected(slot.error());
      }
      native_slots_[i] = *slot;
    }
    return PrepareBranches(static_cast<std::uint32_t>(runner_.request_slots()), 3);
  }
  std::uint64_t activations_needed() const override { return runner_.activations_needed(); }
  std::uint64_t pool_needed() const override { return runner_.pool_needed(); }
  std::uint64_t host_input_bytes() const override { return runner_.host_input_bytes(); }
  std::uint64_t plan_floor_bytes() const override { return runner_.plan_floor_bytes(); }
  std::uint64_t graph_measured_bytes() const override { return runner_.graph_measured_bytes(); }
  void ReclaimCandidates(std::uint32_t owner, bool running,
                         std::vector<memory::ReclaimCandidate>& out) override {
    runner_.ReclaimCandidates(owner, running, out);
  }
  std::uint64_t Reclaim(memory::ReclaimKind kind, std::uint64_t id) override {
    return runner_.Reclaim(kind, id);
  }
  std::string plan_report() const override { return runner_.plan_report(); }
  std::string slots_report() const override { return slots_report_; }
  Status Register() override { return runner_.Register(); }
  Status Bind() override { return runner_.Bind(); }
  std::vector<catalog::ExtentId> weights() const override { return runner_.weights(); }
  std::vector<catalog::ExtentId> state() const override { return runner_.state(); }
  std::vector<catalog::ExtentId> kept_state() const override { return runner_.kept_state(); }
  std::vector<catalog::ExtentId> unchanged_state() const override {
    return runner_.unchanged_state();
  }
  void StateWrittenBack(bool whole) override { runner_.StateWrittenBack(whole); }
  const catalog::Closure& everything() const override { return runner_.everything(); }
  const catalog::Closure& request_closure() const override { return runner_.execution_closure(); }
  bool HasRetainedState() const override { return AnyBranchHasRetainedState(); }
  Status PrepareDefaultRequest() override {
    std::array<Branch*, 1> active{&default_branch()};
    return SelectBranches(active);
  }
  Status SelectBranches(std::span<Branch* const> active) override {
    if (active.size() > runner_.request_slots()) {
      return Error("too many Qwen native conversation branches");
    }
    std::array<engine::Qwen38Runner::Slot*, engine::Qwen38Runner::kRequestSlots> selected{};
    for (std::size_t i = 0; i < active.size(); ++i) {
      if (active[i] == nullptr || &active[i]->model() != this) {
        return Error("a selected Qwen conversation belongs to another model");
      }
      selected[i] = &NativeSlot(*active[i]);
    }
    return runner_.SelectSlots(std::span(selected).first(active.size()));
  }
  std::uint64_t weight_read_bytes() const override { return runner_.weight_read_bytes(); }
  std::uint32_t logit_width() const override { return runner_.vocab(); }
  Status AfterLoad() override { return runner_.ReadPleHash(); }
  Status CheckPlaces() override { return runner_.CheckPlaces(); }
  void DropPlans() override { runner_.DropPlans(); }
  double plan_seconds() const override { return runner_.plan_seconds(); }
  double demand_seconds() const override { return runner_.ple().seconds; }
  std::uint64_t demand_bytes() const override { return runner_.ple().read_bytes; }
  GraphCounts graphs() const override {
    return Sum(runner_.graph_stats(), runner_.draft_stats(), runner_.graphs());
  }
  std::string violations() const override {
    return runner_.coverage_violations() == 0
               ? std::string()
               : std::format("{} bound tensors outside cataloged extents of their class; first {}",
                             runner_.coverage_violations(), runner_.first_violation());
  }
  std::string extra() const override {
    const engine::PleStats& p = runner_.ple();
    return std::format(
        R"({{"turn_checkpoints":{},"turn_checkpoint_bytes":{},"used_state_bytes":{},"architecture":"qwen4exp","speculation":{},"coverage_tensors":{},)"
        R"("slab_padding":{},"state_bytes":{},"drafter_state_bytes":{},"ple_table_bytes":{},)"
        R"("ple":{{"chunks":{},"lookups":{},"rows":{},"reads":{},"read_bytes":{},)"
        R"("seconds":{:.6f}}},"graphs":{}}})",
        turn_checkpoints(), turn_checkpoint_bytes(), runner_.used_state_bytes(),
        speculate_ ? "\"mtp\"" : "null", runner_.coverage_tensors(), runner_.slab_padding(),
        runner_.state_bytes(), runner_.drafter_state_bytes(), runner_.table_bytes(), p.chunks,
        p.lookups, p.rows, p.reads, p.read_bytes, p.seconds, GraphJson(graphs()));
  }
  void Defaults(chat::Conversation& /*c*/) const override {
    // The template's own defaults (thinking on), as the oracle's /tokenize
    // rendered the references' prompts.
  }
  // A speculative step's drafts must stay inside the context (SpecStep).
  std::uint32_t usable_context() const override {
    return speculate_ ? context_ - runner_.draft_rows() : context_;
  }
  bool supports_generation_waves() const override { return runner_.waves_provisioned(); }
  std::size_t generation_wave_capacity() const override { return runner_.wave_capacity(); }
  std::uint64_t StateBytesThrough(std::uint32_t positions) const override {
    return runner_.StateBytesThrough(positions).value_or(0);
  }

 protected:
  Status RunChunk(std::span<const std::int32_t> all, std::uint32_t n_past, bool inject,
                  std::vector<float>& logits) override {
    return RunChunkFor(default_branch(), all, n_past, inject, logits);
  }
  Status SpecStep(std::span<const std::int32_t> all, std::uint32_t pos, std::uint32_t left,
                  std::vector<std::int32_t>& kept, std::vector<std::vector<float>>* logits,
                  std::uint64_t& drafted) override {
    return SpecStepFor(default_branch(), all, pos, left, kept, logits, drafted);
  }
  Status Settle() override { return SettleFor(default_branch()); }
  Status ClearState() override { return ClearStateFor(default_branch()); }
  bool StateUsable() const override { return StateUsableFor(default_branch()); }
  Status PrepareDecodeState(std::uint32_t pos, std::uint32_t left) override {
    return PrepareDecodeStateFor(default_branch(), pos, left);
  }
  std::uint64_t target_state_base() const override { return TargetStateBaseFor(default_branch()); }
  std::uint64_t target_state_bytes() const override {
    return TargetStateBytesFor(default_branch());
  }
  std::uint64_t drafter_state_base() const override {
    return DrafterStateBaseFor(default_branch());
  }
  std::uint64_t drafter_state_bytes() const override {
    return DrafterStateBytesFor(default_branch());
  }
  std::uint64_t used_state_bytes() const override { return UsedStateBytesFor(default_branch()); }
  std::vector<engine::LiveState::Range> used_state_ranges() const override {
    return UsedStateRangesFor(default_branch());
  }
  Status SaveUsedState(void* host, std::span<const engine::LiveState::Range> ranges) override {
    return SaveUsedStateFor(default_branch(), host, ranges);
  }
  Status RestoreUsedState(void* host, std::span<const engine::LiveState::Range> ranges) override {
    return RestoreUsedStateFor(default_branch(), host, ranges);
  }
  std::expected<std::vector<engine::LiveState::Range>, std::string> CheckpointRanges(
      std::uint32_t positions) const override {
    return CheckpointRangesFor(default_branch(), positions);
  }
  Status PrepareRestoreState(std::span<const engine::LiveState::Range> footprint) override {
    return PrepareRestoreStateFor(default_branch(), footprint);
  }
  Status CopyCheckpointState(void* host, std::span<const engine::LiveState::Range> ranges,
                             bool to_host) override {
    return CopyCheckpointStateFor(default_branch(), host, ranges, to_host);
  }
  std::uint32_t cursor() const override { return CursorFor(default_branch()); }
  void set_cursor(std::uint32_t value) override { SetCursorFor(default_branch(), value); }
  void SaveDecodingState() override { SaveDecodingStateFor(default_branch()); }
  void RestoreDecodingState() override { RestoreDecodingStateFor(default_branch()); }
  execution::AdaptiveDepth TurnDecodingState() const override {
    return TurnDecodingStateFor(default_branch());
  }
  void RestoreTurnDecodingState(const execution::AdaptiveDepth& state) override {
    RestoreTurnDecodingStateFor(default_branch(), state);
  }

  Status RunChunkFor(Branch& branch, std::span<const std::int32_t> all, std::uint32_t n_past,
                     bool inject, std::vector<float>& logits) override {
    return NativeSlot(branch).Chunk(all, n_past, logits, inject);
  }
  Status RunPrefillChunkFor(Branch& branch, std::span<const std::int32_t> all, std::uint32_t past,
                            bool inject, bool want_head, std::vector<float>& logits,
                            PrefillHint next) override {
    if (!prefill_hints_) next = {};
    if (!prefill_output_intent_) {
      want_head = true;
      next.want_head = true;
      next.after_want_head = true;
    }
    return NativeSlot(branch).PrefillChunk(all, past, logits, inject,
                                           {.rows = next.rows,
                                            .want_head = next.want_head,
                                            .after_rows = next.after_rows,
                                            .after_want_head = next.after_want_head},
                                           want_head);
  }
  std::optional<Status> RunGreedyChunkFor(Branch& branch, std::span<const std::int32_t> all,
                                          std::uint32_t pos, std::int32_t& token) override {
    if (!options_.device_tokens || speculate_) return std::nullopt;
    return NativeSlot(branch).GreedyChunk(all, pos, token);
  }
  Status SettleFor(Branch& branch) override { return NativeSlot(branch).Rollback(); }
  bool StateUsableFor(const Branch& branch) const override {
    return NativeSlot(branch).state_usable();
  }
  std::uint64_t TargetStateBaseFor(const Branch& branch) const override {
    return NativeSlot(branch).state_base();
  }
  std::uint64_t TargetStateBytesFor(const Branch& branch) const override {
    return NativeSlot(branch).state_bytes();
  }
  std::uint64_t DrafterStateBaseFor(const Branch& branch) const override {
    return NativeSlot(branch).drafter_state_base();
  }
  std::uint64_t DrafterStateBytesFor(const Branch& branch) const override {
    return NativeSlot(branch).drafter_state_bytes();
  }
  std::uint64_t UsedStateBytesFor(const Branch& branch) const override {
    return NativeSlot(branch).used_state_bytes();
  }
  std::vector<engine::LiveState::Range> UsedStateRangesFor(const Branch& branch) const override {
    return NativeSlot(branch).used_state_ranges();
  }
  Status SaveUsedStateFor(Branch& branch, void* host,
                          std::span<const engine::LiveState::Range> ranges) override {
    return NativeSlot(branch).SaveUsedState(host, ranges);
  }
  Status RestoreUsedStateFor(Branch& branch, void* host,
                             std::span<const engine::LiveState::Range> ranges) override {
    return NativeSlot(branch).RestoreUsedState(host, ranges);
  }
  std::expected<std::vector<engine::LiveState::Range>, std::string> CheckpointRangesFor(
      const Branch& branch, std::uint32_t positions) const override {
    return NativeSlot(branch).CheckpointRanges(positions);
  }
  Status PrepareRestoreStateFor(Branch& branch,
                                std::span<const engine::LiveState::Range> footprint) override {
    return NativeSlot(branch).PrepareRestoreState(footprint);
  }
  Status CopyCheckpointStateFor(Branch& branch, void* host,
                                std::span<const engine::LiveState::Range> ranges,
                                bool to_host) override {
    return NativeSlot(branch).CopyCheckpointState(host, ranges, to_host);
  }
  std::uint32_t CursorFor(const Branch& branch) const override {
    return NativeSlot(branch).pending_rows();
  }
  void SetCursorFor(Branch& branch, std::uint32_t value) override {
    NativeSlot(branch).set_pending_rows(value);
  }

  std::size_t TokenScratchCapacityFor(const Branch& branch, std::size_t tokens,
                                      std::uint32_t left) const override {
    return speculate_ ? tokens + std::min(left, DraftDepthFor(branch)) : 0;
  }

  Status SpecStepFor(Branch& branch, std::span<const std::int32_t> all, std::uint32_t pos,
                     std::uint32_t left, std::vector<std::int32_t>& kept,
                     std::vector<std::vector<float>>* logits, std::uint64_t& drafted,
                     bool* prefix_kept = nullptr) override {
    if (!ReserveTokenScratch(branch, all.size(), left)) {
      if (prefix_kept != nullptr) {
        *prefix_kept = true;
      }
      return Error("native verify history exceeds the execution budget");
    }
    // Sampling's position keys keep their existing fixed draft schedule.
    // Greedy chooses its depth from deterministic acceptance observations.
    const auto depth = DraftDepthFor(branch);
    if (depth > options_.context - pos) {
      return Error("the drafts would pass the context");
    }
    const auto started = Clock::now();
    std::vector<std::int32_t> drafts;
    if (auto r = NativeSlot(branch).Draft(all, drafts, nullptr, depth); !r) {
      return r;
    }
    // The verify: the anchor and its drafts, within the tokens left.
    const auto rows = std::min<std::uint32_t>(
        {static_cast<std::uint32_t>(drafts.size()) + 1, left, options_.context - pos});
    drafts.resize(rows - 1);
    auto& input = VerifyTokens(branch);
    input.assign(all.begin(), all.end());
    input.insert(input.end(), drafts.begin(), drafts.end());
    std::vector<std::int32_t> argmax;
    std::vector<float> verified;
    const bool rows_needed = logits != nullptr || sampling(branch);
    if (auto r = NativeSlot(branch).Verify(input, pos, argmax, rows_needed ? &verified : nullptr);
        !r) {
      return r;
    }
    // A full greedy step's time by its depth: the calibration of the depth
    // cost ratio (calibration.h), in force from the next start.
    if (!sampling(branch) && rows == depth + 1 && pos < CalibrationSamples::kShortContext) {
      calibration_samples_.DraftStep(depth, Seconds(Clock::now() - started));
    }
    return Undone(branch,
                  JudgeVerify(branch, drafts, pos, depth, argmax, verified, kept, logits, drafted),
                  prefix_kept);
  }
  bool StateRefusedFor(const Branch& branch) const override {
    return NativeSlot(branch).state_refused();
  }
  Status PrepareDecodeStateFor(Branch& branch, std::uint32_t pos, std::uint32_t left) override {
    NativeSlot(branch).ForgetRefusal();  // the refusals below are not capacity's
    if (!speculate_) {
      return NativeSlot(branch).ReserveStateThrough(pos + 1);
    }
    const auto depth = DraftDepthFor(branch);
    if (depth > context_ - pos) {
      return Error("the drafts would pass the context");
    }
    const auto verify_rows = std::min({depth + 1, left, context_ - pos});
    // Draft reprocesses tokens ending before the anchor; its final pass
    // ends at pos + depth - 1. Verify includes the anchor itself.
    return NativeSlot(branch).ReserveStateThrough(pos + std::max(depth - 1, verify_rows));
  }

  Status ClearStateFor(Branch& branch) override {
    BranchDecoding(branch) = execution::AdaptiveDepth(3, depth_cost_ratio_);
    return NativeSlot(branch).Clear();
  }
  Status ReleaseIdleStateFor(Branch& branch) override {
    BranchDecoding(branch) = execution::AdaptiveDepth(3, depth_cost_ratio_);
    return NativeSlot(branch).ClearIdle();
  }
  Status SpillFor(Branch& branch) override { return NativeSlot(branch).Spill(); }
  Status RestoreFor(Branch& branch) override { return NativeSlot(branch).Restore(); }
  bool SpilledFor(const Branch& branch) const override { return NativeSlot(branch).spilled(); }
  std::uint64_t SpilledBytesFor(const Branch& branch) const override {
    return NativeSlot(branch).spilled_bytes();
  }
  std::uint64_t SpillWriteBytesFor(const Branch& branch) const override {
    return NativeSlot(branch).spill_write_bytes();
  }
  bool LeasedFor(const Branch& branch) const override { return NativeSlot(branch).held(); }
  std::uint64_t RefusedBytesFor(const Branch& branch) const override {
    const auto& slot = NativeSlot(branch);
    return slot.spilled() ? slot.spilled_bytes() : slot.refused_bytes();
  }

 public:
  // Kept across a restart (D-105).
  std::string KeptLayout() const override { return runner_.kept_layout(); }
  std::vector<std::uint64_t> KeptRegions() const override {
    return KeptRegionsOf(NativeSlot(default_branch()).live(), true);
  }
  std::vector<std::uint64_t> KeptLayouts() const override {
    return KeptRegionsOf(NativeSlot(default_branch()).live(), false);
  }
  // D-102's rung 2.
  Status RecoverInPlace() override {
    auto discarded = runner_.RecoverInPlace([this](std::uint32_t slot) { UnkeepSlot(slot); });
    if (!discarded) {
      return std::unexpected(discarded.error());
    }
    ForgetDiscarded(*discarded);
    for (std::size_t i = 0; i < branches(); ++i) {
      if ((*discarded & (engine::SlotMask{1} << i)) != 0) {
        if (auto b = branch(i); b) {
          // As a cleared state's.
          BranchDecoding(**b) = execution::AdaptiveDepth(3, depth_cost_ratio_);
        }
      }
    }
    return {};
  }

 protected:
  const engine::LiveState* KeptLiveFor(const Branch& branch) const override {
    return &NativeSlot(branch).live();
  }
  bool KeptWholeFor(const Branch& branch) const override { return NativeSlot(branch).kept_whole(); }
  bool OwedFor(const Branch& branch) const override { return NativeSlot(branch).owed(); }
  Status AdoptFor(Branch& branch, std::span<const engine::LiveState::Range> used) override {
    return NativeSlot(branch).Adopt(used);
  }
  void SetSpillPlaces(
      const std::function<engine::LiveState::SpillPlace(std::uint32_t)>& place) override {
    options_.spill_place = place;
  }
  void SaveDecodingStateFor(Branch& branch) override { SaveBranchDecoding(branch); }
  void RestoreDecodingStateFor(Branch& branch) override { RestoreBranchDecoding(branch); }
  execution::AdaptiveDepth TurnDecodingStateFor(const Branch& branch) const override {
    return BranchDecoding(branch);
  }
  void RestoreTurnDecodingStateFor(Branch& branch, const execution::AdaptiveDepth& state) override {
    BranchDecoding(branch) = state;
  }

  bool GenerationCohortUsable() const override { return runner_.cohort_usable(); }

  Status RunPreparedGenerationWave(std::span<PreparedGeneration> prepared) override {
    if (!runner_.waves_provisioned()) {
      return Llm::RunPreparedGenerationWave(prepared);
    }
    if (!speculate_) {
      const bool greedy =
          options_.device_tokens &&
          std::ranges::all_of(prepared, [](const auto& unit) { return DeviceGreedy(unit); });
      std::array<std::int32_t, engine::Qwen38Runner::kRequestSlots> chosen{};
      std::vector<engine::Qwen38Runner::ChunkWork> chunks;
      chunks.reserve(prepared.size());
      for (PreparedGeneration& unit : prepared) {
        chunks.push_back({.slot = &NativeSlot(*unit.branch),
                          .history = unit.step.all,
                          .n_past = unit.step.position,
                          .logits = greedy ? nullptr : &unit.row,
                          .token = greedy ? &chosen[chunks.size()] : nullptr});
      }
      if (auto ran = runner_.ChunkWave(chunks); !ran) return ran;
      if (greedy)
        for (std::size_t i = 0; i < prepared.size(); ++i) prepared[i].chosen = chosen[i];
      return {};
    }

    // These owners never move once the borrowed descriptors are made. Each
    // completed Draft/VerifyWave consumes its inputs before this frame ends.
    struct Frame {
      std::uint32_t depth = 0;
      std::vector<std::int32_t> drafts;
      std::vector<std::int32_t> argmax;
      std::vector<float> verified;
    };
    std::vector<Frame> frames(prepared.size());
    std::vector<engine::Qwen38Runner::DraftWork> drafts;
    drafts.reserve(prepared.size());
    const auto started = Clock::now();
    for (std::size_t i = 0; i < prepared.size(); ++i) {
      PreparedGeneration& unit = prepared[i];
      Frame& frame = frames[i];
      // A shared wave drafts shared_wave_depth tokens a request (two: each
      // verify row past that reads its own routed experts for every member,
      // so the third draft's acceptance no longer pays for itself; C2/C4
      // HTTP screens: +8%/+15% over adaptive depth). A lone request keeps
      // its adaptive depth.
      frame.depth =
          std::min(DraftDepthFor(*unit.branch), prepared.size() > 1 ? shared_wave_depth_ : 3U);
      if (frame.depth > context_ - unit.step.position) {
        return Error("the drafts would pass the context");
      }
      drafts.push_back({.slot = &NativeSlot(*unit.branch),
                        .history = unit.step.all,
                        .drafts = &frame.drafts,
                        .passes = frame.depth});
    }
    // Past draft_wave_max requests (two) each drafts alone, on its own
    // cached graphs: a draft wave's graph keys every slot's pending rows
    // (one to four), so a wider wave's graphs would seldom repeat (C4 HTTP
    // screen: 3.6% faster).
    if (drafts.size() > draft_wave_max_) {
      for (std::size_t i = 0; i < prepared.size(); ++i) {
        if (auto ran = NativeSlot(*prepared[i].branch)
                           .Draft(prepared[i].step.all, frames[i].drafts, nullptr, frames[i].depth);
            !ran) {
          return ran;
        }
      }
    } else if (auto ran = runner_.DraftWave(drafts); !ran) {
      return ran;
    }
    std::vector<engine::Qwen38Runner::VerifyWork> verifies;
    verifies.reserve(prepared.size());
    for (std::size_t i = 0; i < prepared.size(); ++i) {
      PreparedGeneration& unit = prepared[i];
      Frame& frame = frames[i];
      // Keep the scalar schedule: all requested draft passes run, then the
      // verify is truncated independently by this branch's output/context.
      const auto rows =
          std::min<std::uint32_t>({static_cast<std::uint32_t>(frame.drafts.size()) + 1,
                                   unit.step.left, context_ - unit.step.position});
      frame.drafts.resize(rows - 1);
      auto& input = VerifyTokens(*unit.branch);
      input.assign(unit.step.all.begin(), unit.step.all.end());
      input.insert(input.end(), frame.drafts.begin(), frame.drafts.end());
      verifies.push_back(
          {.slot = &NativeSlot(*unit.branch),
           .history = input,
           .n_past = unit.step.position,
           .argmax = &frame.argmax,
           .logits = unit.step.need_logits || sampling(*unit.branch) ? &frame.verified : nullptr});
    }
    if (auto ran = runner_.VerifyWave(verifies); !ran) {
      return ran;
    }
    for (std::size_t i = 0; i < prepared.size(); ++i) {
      PreparedGeneration& unit = prepared[i];
      const Frame& frame = frames[i];
      unit.result = JudgeVerify(*unit.branch, frame.drafts, unit.step.position, frame.depth,
                                frame.argmax, frame.verified, unit.kept,
                                unit.step.need_logits ? &unit.logits : nullptr, unit.drafted);
      if (!unit.result) {
        // The shared Verify completed; only this branch's host judgement
        // failed. Undo its own completed verify before preserving its prefix.
        if (auto rolled = NativeSlot(*unit.branch).DiscardVerify(); !rolled) {
          if (!runner_.cohort_usable()) {
            return Error(
                std::format("{}; rolling back failed: {}", unit.result.error(), rolled.error()));
          }
          unit.result = Error(
              std::format("{}; rolling back failed: {}", unit.result.error(), rolled.error()));
        } else {
          unit.failed_prefix_valid = true;
        }
      }
    }
    // A lone greedy request's full step by its depth: the calibration of
    // the depth cost ratio (calibration.h), in force from the next start.
    if (prepared.size() == 1 && prepared[0].result && !sampling(*prepared[0].branch) &&
        frames[0].drafts.size() == frames[0].depth &&
        prepared[0].step.position < CalibrationSamples::kShortContext) {
      calibration_samples_.DraftStep(frames[0].depth, Seconds(Clock::now() - started));
    }
    return {};
  }

 private:
  // A failed judgement after the native verify completed: the verify is
  // undone, so the step's starting prefix still holds (`prefix_kept`, as a
  // wave's failed_prefix_valid); an undo that fails says so.
  Status Undone(Branch& branch, Status judged, bool* prefix_kept) {
    if (judged) {
      return judged;
    }
    if (auto rolled = NativeSlot(branch).DiscardVerify(); !rolled) {
      return Error(std::format("{}; rolling back failed: {}", judged.error(), rolled.error()));
    }
    if (prefix_kept != nullptr) {
      *prefix_kept = true;
    }
    return judged;
  }

  std::uint32_t DraftDepthFor(const Branch& branch) const {
    return sampling(branch) || runner_.draft_rows() < 3 ? 2U : BranchDecoding(branch).Choose();
  }
  Status JudgeVerify(Branch& branch, std::span<const std::int32_t> drafts, std::uint32_t pos,
                     std::uint32_t depth, std::span<const std::int32_t> argmax,
                     std::span<const float> verified, std::vector<std::int32_t>& kept,
                     std::vector<std::vector<float>>* logits, std::uint64_t& drafted) {
    const auto rows = static_cast<std::uint32_t>(drafts.size()) + 1;
    const std::uint32_t vocab = runner_.vocab();
    // Validate the runner's whole completed result before using borrowed rows.
    if (argmax.size() != rows ||
        ((logits != nullptr || sampling(branch)) && verified.size() != std::size_t{rows} * vocab)) {
      return Error("a Qwen verify returned inconsistent token/logit counts");
    }
    const auto row = [&](std::uint32_t i) {
      return verified.subspan(std::size_t{i} * vocab, vocab);
    };
    std::uint32_t m = 0;
    std::int32_t next = -1;
    if (!sampling(branch)) {
      // Greedy: the verify's own argmaxes.
      for (; m < rows - 1; ++m) {
        if (argmax[m] != drafts[m]) {
          next = argmax[m];
          break;
        }
      }
      if (next < 0) {
        next = argmax[m];
      }
    } else {
      // Speculative sampling over the verify's rows; row m predicts the
      // token at pos + m + 1.
      for (; m < rows - 1; ++m) {
        std::int32_t instead = -1;
        auto keep = Keep(branch, row(m), drafts[m], std::uint64_t{pos} + m + 1, instead);
        if (!keep) {
          return std::unexpected(keep.error());
        }
        if (!*keep) {
          next = instead;
          break;
        }
      }
      if (next < 0) {
        auto chosen = Choose(branch, row(m), std::uint64_t{pos} + m + 1);
        if (!chosen) {
          return std::unexpected(chosen.error());
        }
        next = *chosen;
      }
    }
    if (auto r = NativeSlot(branch).Accept(m + 1); !r) {
      return r;
    }
    kept.assign(drafts.begin(), drafts.begin() + m);
    kept.push_back(next);
    if (logits != nullptr) {
      for (std::uint32_t i = 0; i <= m; ++i) {
        const std::span<const float> kept_row = row(i);
        logits->emplace_back(kept_row.begin(), kept_row.end());
      }
    }
    drafted += rows - 1;
    if (!sampling(branch)) {
      BranchDecoding(branch).Observe(depth, m + 1, rows == depth + 1);
    }
    return {};
  }
  engine::Qwen38Runner::Slot& NativeSlot(Branch& branch) {
    const auto index = BranchIndex(branch);
    base::Check(native_slots_[index] != nullptr, "native conversation slots are not ready");
    return *native_slots_[index];
  }
  const engine::Qwen38Runner::Slot& NativeSlot(const Branch& branch) const {
    const auto index = BranchIndex(branch);
    base::Check(native_slots_[index] != nullptr, "native conversation slots are not ready");
    return *native_slots_[index];
  }
  config::ModelEntry entry_;  // its tokenizer and template files, if named
  std::string artifact_id_;
  std::string drafter_id_;  // empty: none
  fs::path store_;
  std::string slots_report_;
  // A wave's drafts a request, and the most requests drafting in one
  // joined wave (its settings').
  std::uint32_t shared_wave_depth_ = kQwen38SharedWaveDepth;
  std::uint32_t draft_wave_max_ = kQwen38DraftWaveMax;
  bool prefill_hints_ = true;
  bool prefill_output_intent_ = true;
  engine::Qwen38Options options_;  // before the runner, which keeps a reference
  engine::Qwen38Runner runner_;
  std::array<engine::Qwen38Runner::Slot*, engine::Qwen38Runner::kRequestSlots> native_slots_{};
};

// The Qwen-Image-2.1 pipeline (engine/qwen_image_runner.h).
class QwenImage final : public Image {
 public:
  QwenImage(engine::PagedNode& node, const config::ModelEntry& entry, const ModelSettings& settings,
            const config::RuntimeRoles& roles, const ServingOptions& serving, int index)
      : composition_id_(entry.composition.value_or("")),
        store_(roles.installed),
        runner_(node, options_, index, static_cast<std::uint32_t>(index)) {
    name_ = entry.name;
    settings_ = settings;
    options_.store = roles.installed;
    options_.composition = composition_id_;
    options_.noise = serving.image_noise;
    options_.out = roles.spill;
    options_.prompt = serving.image_prompt;
    options_.size = settings.image_size.value;
    options_.steps = settings.image_steps.value;
  }

  engine::PagedModel& paged() override { return runner_; }
  Status Setup() override {
    // Every component opened and checked first.
    const fs::path root = store_ / composition_id_;
    auto composition = ja::OpenComposition(root, composition_id_);
    if (!composition) {
      return Error(
          std::format("composition {}: {}", composition_id_, composition.error().ToString()));
    }
    for (const ja::CompositionComponent& c : composition->components()) {
      if (auto opened = OpenInstalled(store_, c.artifact); !opened) {
        return std::unexpected(opened.error());
      }
    }
    return runner_.Setup();
  }
  std::uint64_t activations_needed() const override { return runner_.activations_needed(); }
  std::uint64_t pool_needed() const override { return engine::QwenImageRunner::pool_needed(); }
  Status Register() override { return runner_.Register(); }
  Status Bind() override { return runner_.Bind(); }
  std::vector<catalog::ExtentId> weights() const override { return runner_.weights(); }
  const catalog::Closure& everything() const override { return runner_.everything(); }
  std::uint64_t weight_read_bytes() const override { return runner_.weight_read_bytes(); }
  std::string extra() const override { return runner_.Report(); }
  std::string plan_report() const override { return runner_.plan_report(); }
  std::uint64_t graph_measured_bytes() const override { return runner_.graph_measured_bytes(); }
  void ReclaimCandidates(std::uint32_t owner, bool running,
                         std::vector<memory::ReclaimCandidate>& out) override {
    runner_.ReclaimCandidates(owner, running, out);
  }
  std::uint64_t Reclaim(memory::ReclaimKind kind, std::uint64_t id) override {
    return runner_.Reclaim(kind, id);
  }
  Status FirstOutput(std::string& sha) override { return runner_.FirstOutput(sha); }
  Status Finish(std::string& sha) override { return runner_.Finish(sha); }

 private:
  std::string composition_id_;
  fs::path store_;
  engine::QwenImageOptions options_;  // before the runner, which keeps a reference
  engine::QwenImageRunner runner_;
};

}  // namespace

// ---------------------------------------------------------------- memory

MemorySampler::MemorySampler() : start_(Available()) {
  low_ = start_;
  all_ = start_;
  thread_ = std::jthread([this] {
    if (auto stack = platform::InstallThreadSignalStack(); !stack) {
      std::println(stderr, "the memory sampler runs without a signal stack: {}", stack.error());
    }
    Loop();
  });
}

MemorySampler::~MemorySampler() {
  stop_ = true;
  if (thread_.joinable()) {
    thread_.join();
  }
}

std::uint64_t MemorySampler::Available() { return platform::AvailableMemoryBytes().value_or(0); }

void MemorySampler::Reset() { low_ = Available(); }

void MemorySampler::Loop() {
  const auto lower = [](std::atomic<std::uint64_t>& to, std::uint64_t now) {
    std::uint64_t seen = to.load();
    while (now < seen && !to.compare_exchange_weak(seen, now)) {
    }
  };
  while (!stop_) {
    const std::uint64_t now = Available();
    if (now != 0) {
      lower(low_, now);
      lower(all_, now);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
}

void ResidentTimes::Staged(catalog::ExtentId extent, scheduler::PageInEvent event) {
  events_.fetch_add(1, std::memory_order_relaxed);
  if (event == scheduler::PageInEvent::kResident && extent.index() < at_.size()) {
    at_[extent.index()] = Clock::now();
  }
}

Clock::time_point ResidentTimes::Latest(std::span<const catalog::ExtentId> extents) const {
  Clock::time_point latest{};
  for (const catalog::ExtentId extent : extents) {
    if (extent.index() < at_.size()) {
      latest = std::max(latest, at_[extent.index()]);
    }
  }
  return latest;
}

// ---------------------------------------------------------------- the server

Server::Server(const config::NodeConfig& config, const config::RuntimeRoles& roles,
               const ServingOptions& options, std::FILE* log)
    : config_(config),
      roles_(roles),
      options_(options),
      log_(log),
      times_(kObservedExtents),
      node_({.compute_streams = std::max<std::size_t>(config.models.size(), 1),
             .slots = engine::kPagedSlots,
             .inline_lanes = false,
             .coalesce = false,
             .copy_lane = true,
             .lazy_handoff = options.lazy_handoff,
             .zero_state = options.zero_state,
             .handle_reserve = options.handle_reserve.value_or(engine::kHandleReserve),
             .slot_bytes = engine::kSlabSlotBytes,
             .observer = &times_,
             .poll_window = std::nullopt,
             .spin_ahead = std::nullopt,
             .quiet = std::chrono::minutes(10),
             .hold_reads = &hold_reads_,
             .hold_cancellable = options.hold_cancellable}),
      retention_(std::chrono::hours(config.memory.retention_hours)),
      spill_budget_(std::uint64_t{config.memory.spill_budget_gib} << 30U) {}

Server::~Server() {
  if (started_ && !torn_down_) {
    if (auto stopped = TearDown(); !stopped) {
      Log("stopping: " + stopped.error());
    }
  }
}

void Server::Log(std::string_view text) {
  const std::string line = std::format("llmp-runtime: {}\n", base::Printable(text));
  (void)std::fwrite(line.data(), 1, line.size(), log_);
  (void)std::fflush(log_);
}

Status Server::Make(const config::ModelEntry& entry, const ModelSettings& settings, int index) {
  // A family on the component engine: the adapter of its spec's pipeline
  // kind, which its architecture's mapping names.
  if (!entry.composition && OnComponentEngine(settings.architecture)) {
    const spec::Mapping* mapping = spec::FindMapping(settings.architecture);
    if (mapping == nullptr) {
      return Error(std::format("model {}: the component engine has no mapping of architecture {}",
                               entry.name, settings.architecture));
    }
    switch (mapping->pipeline) {
      case spec::PipelineKind::kLlm: {
        auto llm =
            std::make_unique<EngineLlm>(node_, entry, settings, roles_, index,
                                        ComponentLlmBuild(node_, roles_.installed, entry, index));
        if (options_.diagnostic_skip_turn_checkpoints) {
          llm->set_turn_checkpoints(false);
          Log(std::format("model {}: saves no turn checkpoint (a speed gate's diagnostic)",
                          entry.name));
        }
        models_.push_back(std::move(llm));
        return {};
      }
      case spec::PipelineKind::kImage:
        models_.push_back(std::make_unique<EngineImage>(
            node_, entry, settings, index, ComponentImageBuild(settings.architecture)));
        return {};
      default:
        return Error(std::format("model {}: no serving adapter for the {} pipeline kind yet",
                                 entry.name, spec::PipelineName(mapping->pipeline)));
    }
  }
  if (entry.composition) {
    if (options_.image_noise.empty()) {
      Log(
          std::format("model {}: not registered: an image pipeline needs --image-noise (its "
                      "initial latents) in this build",
                      entry.name));
      return {};
    }
    models_.push_back(std::make_unique<QwenImage>(node_, entry, settings, roles_, options_, index));
    return {};
  }
  if (settings.architecture == "deepseek4") {
    models_.push_back(std::make_unique<Dsv4>(node_, entry, settings, roles_, index,
                                             options_.diagnostic_plain_device_tokens, options_));
  } else if (settings.architecture == "gemma4") {
    auto artifact = OpenInstalled(roles_.installed, entry.artifact.value_or(""));
    if (!artifact) return Error(artifact.error());
    auto profile = ApprovedGemmaProfile(*artifact);
    if (!profile) return Error(profile.error());
    const auto variant = *profile == &model::Gemma4_26BA4B() ? engine::Gemma4Variant::k26BA4B
                                                             : engine::Gemma4Variant::k31B;
    models_.push_back(
        std::make_unique<Gemma>(node_, entry, settings, roles_, index, variant, options_));
  } else if (settings.architecture == "qwen4exp") {
    models_.push_back(std::make_unique<Qwen38>(node_, entry, settings, roles_, index,
                                               options_.diagnostic_plain_device_tokens, options_));
  } else {
    return Error(
        std::format("model {}: no runner for architecture {}", entry.name, settings.architecture));
  }
  return {};
}

Status Server::SaveSnapshot(Llm& model) {
  if (!snapshot_enabled_ || snapshot_unproven_) {
    return Error("the diagnostic snapshot is disabled or its completion is unproven");
  }
  if (snapshot_model_ != nullptr) {
    snapshot_model_->InvalidateStateSnapshot();
    snapshot_model_ = nullptr;
  }
  const auto bytes = std::max<std::uint64_t>(model.state_snapshot_bytes(), 256);
  if (bytes > snapshot_capacity_) {
    if (auto freed = node_.FreePinned(snapshot_); !freed) {
      return freed;
    }
    snapshot_ = nullptr;
    snapshot_capacity_ = 0;
    // Its staging is charged to the catalog, within the budget (the
    // node's Pinned): the guard's margin, set apart beside the budget, is
    // not counted again (that double count refused DeepSeek with DSpark
    // beside Qwen3.8). Refused for the budget, the reclaim order makes
    // room first.
    std::vector<catalog::ExtentId> staging;
    auto allocated = node_.Pinned(bytes, engine::kShared, staging);
    if (!allocated && Reclaim(bytes, true, "a state snapshot's staging") != 0) {
      staging.clear();
      allocated = node_.Pinned(bytes, engine::kShared, staging);
    }
    if (!allocated) {
      return std::unexpected(allocated.error());
    }
    snapshot_ = *allocated;
    snapshot_capacity_ = bytes;
  }
  auto saved = model.SaveState(snapshot_);
  if (!saved) {
    snapshot_unproven_ = true;
    node_.KeepPinned(snapshot_);
  } else {
    snapshot_model_ = &model;
  }
  return saved;
}

Status Server::RestoreSnapshot(Llm& model) {
  if (!snapshot_enabled_ || snapshot_unproven_ || snapshot_model_ != &model) {
    return Error("the diagnostic snapshot is disabled or its completion is unproven");
  }
  auto restored = model.RestoreState(snapshot_);
  if (!restored) {
    snapshot_unproven_ = true;
    node_.KeepPinned(snapshot_);
  }
  return restored;
}

Status Server::Start(bool snapshot) {
  if (started_) {
    return Error("the server started already");
  }
  started_ = true;
  if (config_.models.empty()) {
    return Error("the configuration names no models ([models.<name>], D-096)");
  }
  // Every model's settings (D-103: derived from its artifacts, calibrated on
  // this machine, or overridden), its checkpoint's ceiling among them,
  // resolved before opening the device node or constructing runners. An
  // invalid model later in the list must not cause any earlier model's
  // resources to be allocated either.
  std::vector<ModelSettings> settings;
  std::vector<Llm::CalibrationRecord> calibrations;
  settings.reserve(config_.models.size());
  if (const std::string note = CalibrationDirectoryNote(roles_.state); !note.empty()) {
    Log(std::format("warning: the calibration directory {}/{}: {}; used as found",
                    roles_.state.string(), config::kCalibrationDirectory, note));
  }
  for (const config::ModelEntry& entry : config_.models) {
    auto facts = ReadArtifactFacts(entry, roles_.installed);
    if (!facts) {
      return Error(std::format("model {}: {}", entry.name, facts.error()));
    }
    if (options_.gemma31_production && options_.gemma_row_invariant &&
        (facts->gemma_profile == &model::Gemma4_31B() ||
         facts->gemma_profile == &model::Gemma4_26BA4B()))
      return Error("the bounded Gemma production recipes require ordinary arithmetic");
    // Explicit internal arithmetic diagnostics retain their existing recipe.
    const bool gemma31_recipe =
        options_.gemma31_production || (!options_.gemma_joined && !options_.gemma_row_invariant);
    // Its calibration on this machine, if one is recorded for this device,
    // driver and build (calibration.h); a reading costs a small file.
    Llm::CalibrationRecord record;
    if (!entry.composition) {
      // Keyed by the settings its measurements depend on, as they resolve
      // without calibration.
      auto uncalibrated = ResolveSettings(entry, *facts, nullptr, options_.plain, gemma31_recipe,
                                          options_.gemma3_trained_max);
      if (!uncalibrated) {
        return Error(std::format("model {}: {}", entry.name, uncalibrated.error()));
      }
      record.key = CalibrationKeyOf(entry, *uncalibrated);
      const CalibrationRead read = ReadCalibration(roles_.state, record.key);
      Log(std::format("model {}: calibration {}", entry.name, read.note));
      if (read.calibration) {
        record.known = *read.calibration;
      }
    }
    auto resolved = ResolveSettings(entry, *facts, record.known.empty() ? nullptr : &record.known,
                                    options_.plain, gemma31_recipe, options_.gemma3_trained_max);
    if (!resolved) {
      return Error(std::format("model {}: {}", entry.name, resolved.error()));
    }
    for (const std::string& ignored : resolved->ignored) {
      Log(std::format("model {}: ignored {}", entry.name, ignored));
    }
    settings.push_back(std::move(*resolved));
    calibrations.push_back(std::move(record));
  }
  if (auto r = node_.Open(); !r) {
    return r;
  }
  // Each model's stream and owner: its place among the registered.
  for (std::size_t i = 0; i < config_.models.size(); ++i) {
    const std::size_t before = models_.size();
    if (auto r = Make(config_.models[i], settings[i], static_cast<int>(models_.size())); !r) {
      return r;
    }
    if (models_.size() > before && models_.back()->llm()) {
      static_cast<Llm&>(*models_.back()).calibration_record() = std::move(calibrations[i]);
    }
  }
  if (models_.empty()) {
    return Error("no configured model could be registered");
  }
  std::uint64_t activations = 0;
  std::uint64_t pool = 0;
  std::uint64_t largest = 0;
  std::uint64_t host_inputs = 0;
  std::uint64_t plans = 0;
  for (const auto& m : models_) {
    const auto started = Clock::now();
    StartProgress();
    if (auto r = m->Setup(); !r) {
      return Error(std::format("model {}: {}", m->name(), r.error()));
    }
    activations = std::max(activations, m->activations_needed());
    host_inputs = std::max(host_inputs, m->host_input_bytes());
    plans = std::max(plans, m->plan_floor_bytes());
    pool = std::max(pool, m->pool_needed());
    largest = std::max<std::uint64_t>(largest, m->weights().size() * kExtent);
    std::string chunks;
    if (m->llm()) {
      const auto& l = static_cast<Llm&>(*m);
      chunks = std::format("; prefill chunks of {} rows", l.max_rows());
      if (l.configured_rows() && *l.configured_rows() != l.max_rows()) {
        chunks += std::format(
            " (prefill_chunk {}: capped at what the model allows below its context of {}, in "
            "whole 8-row tiles)",
            *l.configured_rows(), l.context());
      }
    }
    Log(std::format("model {}: set up in {:.2f} s, {} weight extents ({:.2f} GB read a load){}",
                    m->name(), Seconds(Clock::now() - started), m->weights().size(),
                    static_cast<double>(m->weight_read_bytes()) / 1e9, chunks));
    // Each effective setting and its source (D-103; `llmp-runtime
    // settings` lists them with what each came from).
    Log(std::format("model {}: settings {}", m->name(), m->settings().Summary()));
    if (const auto report = m->allocation_report(); !report.empty()) {
      Log(std::format("allocation model {}: {}", m->name(), report));
    }
    if (const auto report = m->plan_report(); !report.empty()) {
      Log(std::format("model {}: plans {}", m->name(), report));
    }
    if (const auto reason = m->serial_reason(); !reason.empty()) {
      Log(std::format("model {}: serves one request at a time ({})", m->name(), reason));
    } else if (const auto slots = m->slots_report(); !slots.empty()) {
      Log(std::format("model {}: {}", m->name(), slots));
    }
  }
  const auto startup_available = MemorySampler::Available();
  const MemoryGuard startup_guard{.largest = largest,
                                  .host_inputs = host_inputs,
                                  .plans = plans + engine::ScratchArenaBytes(),
                                  .available = startup_available,
                                  .fixed = 0,
                                  .requests = RequestFloor(config_.client)};
  const auto startup_reserve = GuardReserve(startup_guard) + largest + activations + pool;
  // Unknown availability (0) bounds nothing here, as the guard skips it;
  // the execution budget bounds what is adopted once it is set (Start).
  startup_token_capacity_ = startup_available == 0 ? std::numeric_limits<std::uint64_t>::max()
                            : startup_available > startup_reserve
                                ? startup_available - startup_reserve
                                : 0;
  token_memory_.SetDriver(
      std::this_thread::get_id(),
      [this](std::uint64_t incoming) { return SetTokenMemory(incoming); }, 0,
      std::chrono::milliseconds{0});
  // Conversations the process before kept (D-105): read, checked and
  // their files named before the node registers them.
  if (auto kept = PrepareKept(); !kept) {
    return kept;
  }
  if (auto r = node_.MapWorkspace(activations, pool); !r) {
    return r;
  }
  workspace_ = activations + pool;
  snapshot_enabled_ = snapshot;
  // The budget (D-050's B): everything mapped for the node's life (the zone,
  // each model's own memory, the workspace, the staging) and the largest
  // model's weights, so a full swap is the only way in.
  fixed_ = node_.catalog().OccupancyOf(node_.domain()).Total().value();
  budget_ = fixed_ + largest;
  // A chunk's host-built inputs are the node's memory too, though allocated
  // per chunk and outside the catalog: counted here beside the margin, not
  // in the budget the catalog enforces. One model runs chunks at a time, so
  // the most any model's chunk builds, not their sum.
  host_inputs_ = host_inputs;
  // So are its plans and graphs (engine/planned.h), but only the floor is
  // set apart here: what one step of any model holds at once. Every plan
  // and graph past it is charged inside the budget (PagedNode::ChargeHost)
  // and given back through the reclaim order when the budget is needed.
  // Beside it, the scratch arena every plan's first build uses (one for the
  // process, at the largest estimate the models' setups measured).
  const std::uint64_t step_plans = plans;
  plans_ = plans + engine::ScratchArenaBytes();
  plans = plans_;
  const std::uint64_t available = MemorySampler::Available();
  MemoryGuard bounds = {.largest = largest,
                        .host_inputs = host_inputs,
                        .plans = plans,
                        .available = available,
                        .fixed = fixed_,
                        .requests = 0};
  // Requests' host memory (intake_limits.h, D-102): a small floor set apart
  // here, out of the catalog's budget, so the requests a route admits never
  // take the margin the driver and the models' state count on; past it the
  // route charges what requests hold to the budget as it goes
  // (SetRequestMemory), through the reclaim order.
  request_memory_ = RequestFloor(config_.client);
  bounds.requests = request_memory_;
  const auto guard = CheckMemoryGuard(bounds);
  Log(
      std::format("allocation guard: {{\"format\":\"llmp-wave-startup-guard-v1\","
                  "\"fixed_catalog_bytes\":{},\"shared_activation_bytes\":{},"
                  "\"shared_scratch_bytes\":{},\"largest_weight_extent_bytes\":{},"
                  "\"host_input_bytes\":{},\"plan_floor_bytes\":{},\"uncounted_margin_bytes\":{},"
                  "\"available_after_fixed_bytes\":{},\"available_known\":{},"
                  "\"guard_passed\":{},\"registered_state_virtual_extent_bytes\":{},"
                  "\"request_memory_bytes\":{},\"native_token_history_bytes\":{}}}",
                  fixed_, activations, pool, largest, host_inputs, plans, kUncountedMargin,
                  available, available != 0, guard.has_value(), node_.StateCapacity(),
                  request_memory_, token_memory_.used()));
  if (!guard) {
    return std::unexpected(guard.error());
  }
  // State is charged as it grows. Keep the physical execution cap below
  // the measured available memory, leaving host-built inputs, the plans and
  // the uncounted margin outside it. A virtual context ceiling need not fit
  // before it is used; a growth that cannot fit its active closure fails.
  if (available != 0) {
    budget_ = fixed_ + ((available - GuardReserve(bounds)) / kExtent * kExtent);
  } else {
    budget_ += node_.StateCapacity();
  }
  dynamic_budget_ = budget_;
  if (options_.diagnostic_budget_cap_bytes) {
    const auto reserve = options_.handle_reserve.value_or(engine::kHandleReserve);
    if (reserve > std::numeric_limits<std::uint64_t>::max() / kExtent)
      return Error("diagnostic budget cap handle reserve overflows");
    const auto cap = *options_.diagnostic_budget_cap_bytes;
    if (auto checked = CheckDiagnosticBudgetCap(bounds, dynamic_budget_, cap, reserve * kExtent);
        !checked)
      return std::unexpected(checked.error());
    budget_ = cap;
    Log(std::format("diagnostic budget cap: dynamic={} selected={} bytes", dynamic_budget_,
                    budget_));
  }
  if (auto r = node_.Start(base::Bytes(budget_)); !r) {
    return r;
  }
  StartProgress();
  for (const auto& m : models_) {
    if (auto r = m->Register(); !r) {
      return Error(std::format("model {}: {}", m->name(), r.error()));
    }
  }
  for (const auto& m : models_) {
    if (auto r = m->Bind(); !r) {
      return Error(std::format("model {}: {}", m->name(), r.error()));
    }
  }
  node_.Run();
  // Plans and graphs past the floor are charged inside the budget; a charge
  // that does not fit has the reclaim order make room first.
  node_.SetHostFloor(step_plans);
  node_.SetReclaimer([this](std::uint64_t needed, engine::PagedNode::ReclaimFor what,
                            std::span<const catalog::ExtentId> protect) {
    // Cache charges displace only other plans and graphs, never
    // conversation state (a graph's only what costs less to restore than a
    // graph); pinned staging (a checkpoint's, a snapshot's) is a state need
    // and may spill idle conversations.
    using For = engine::PagedNode::ReclaimFor;
    switch (what) {
      case For::kPlan:
        return Reclaim(needed, false, "a plan past the free budget", nullptr, std::nullopt, false,
                       nullptr, 0, {}, nullptr, false);
      case For::kGraph:
        return Reclaim(needed, false, "a graph past the free budget", nullptr,
                       memory::ReclaimKind::kGraph, false, nullptr, 0, {}, nullptr, false);
      case For::kStaging:
        break;
      case For::kMaterialize:
        return Reclaim(needed, false, "state materialization past the free budget", nullptr,
                       std::nullopt, false, nullptr, 0, protect);
    }
    return Reclaim(needed, true, "pinned staging past the free budget");
  });
  // The turn checkpoints' staging, set apart once so a budget full of
  // reclaimable plans, graphs and idle state never starves a capture or a
  // restore (they use it one at a time).
  if (std::ranges::any_of(models_, [](const auto& m) { return m->llm(); })) {
    if (auto r = node_.ReserveStaging(engine::CheckpointStagingBytes()); !r) {
      return Error("setting apart the turn checkpoints' staging: " + r.error());
    }
  }
  // The state room: what the budget leaves the conversations beside the
  // fixed memory and the largest model's weights.
  const std::uint64_t room = budget_ > fixed_ + largest ? budget_ - fixed_ - largest : 0;
  state_room_ = room;
  Log(std::format(
      "serving {} models; budget {:.2f} GiB ({:.2f} GiB fixed, the workspace {:.2f}; host-built "
      "chunk inputs {:.2f} GiB, a step's plans {:.2f} GiB and {:.2f} GiB of request memory "
      "beside it); conversation state room {:.2f} GiB beside the largest weights; {:.2f} GiB "
      "available; idle conversations spill, kept {} h within {} GiB",
      models_.size(), static_cast<double>(budget_) / (1ULL << 30U),
      static_cast<double>(fixed_) / (1ULL << 30U), static_cast<double>(workspace_) / (1ULL << 30U),
      static_cast<double>(host_inputs_) / (1ULL << 30U),
      static_cast<double>(plans_) / (1ULL << 30U),
      static_cast<double>(request_memory_) / (1ULL << 30U),
      static_cast<double>(room) / (1ULL << 30U), static_cast<double>(available) / (1ULL << 30U),
      config_.memory.retention_hours, config_.memory.spill_budget_gib));
  for (const auto& m : models_) {
    if (m->llm()) {
      static_cast<Llm&>(*m).SetTokenMemory(token_memory_);
      static_cast<Llm&>(*m).set_retention(retention_);
      static_cast<Llm&>(*m).set_log([this](std::string_view text) { Log(text); });
    }
  }
  token_catalog_ready_ = true;
  if (!SetTokenMemory(0)) {
    // Only when availability was unknown at the start can what was decoded
    // exceed the budget: those conversations are not adopted (their records
    // removed, their files emptied), rather than every start failing.
    Log("kept conversations' token histories do not fit the execution budget; none is adopted");
    for (PendingAdoption& pending : adoptions_) {
      if (auto branch = pending.model->branch(pending.record.slot); branch) {
        std::ignore = (*branch)->ReleaseIdleState();
      }
    }
    adoptions_.clear();
    if (!SetTokenMemory(0)) {
      return Error("kept token histories do not fit the execution budget");
    }
  }
  AdoptKept();
  return {};
}

Status Server::PrepareKept() {
  if (!options_.keep_conversations) {
    return {};  // a command: what the service kept is left alone
  }
  const bool any_llm = std::ranges::any_of(models_, [](const auto& m) { return m->llm(); });
  const auto stamp = platform::RunningExecutableStamp();
  const bool wanted = config_.memory.keep_across_restart && spill_budget_ != 0;
  const bool keep = any_llm && wanted && stamp.has_value();
  const int spill = ::open(roles_.spill.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  if (spill < 0) {
    return Error(std::format("the spill directory {}: {}", roles_.spill.string(), Errno(errno)));
  }
  const std::string root_name(kept::kDirectory);
  if (!keep) {
    // Nothing is kept with a spill budget of 0 (idle state is dropped) or
    // with [memory] keep_across_restart false, and nothing kept before
    // stays (D-014: its retention is explicit).
    if (any_llm && wanted) {
      Log("conversations are not kept across a restart: the executable's identity is unknown");
    } else if (any_llm && !config_.memory.keep_across_restart) {
      Log("conversations are not kept across a restart ([memory] keep_across_restart false)");
    }
    std::ignore = platform::RemovePrivate(spill, root_name.c_str());
    (void)::close(spill);
    return {};
  }
  auto opened = platform::OpenPrivateDirectory(spill, root_name.c_str());
  (void)::close(spill);
  if (!opened) {
    Log(std::format("conversations are not kept across a restart: {}/{}: {}", roles_.spill.string(),
                    root_name, Errno(opened.error())));
    return {};
  }
  const int root = *opened;
  // Found with other owners or modes it is used as it is (D-105 as amended
  // on 2026-10-09), said once.
  if (const std::string note = platform::PrivacyNote(root); !note.empty()) {
    Log(std::format("warning: the kept conversations directory {}/{}: {}; used as found",
                    roles_.spill.string(), root_name, note));
  }
  // Hashing a record's files: a few cores, off the driver's path.
  keeper_ = std::make_unique<StateKeeper>(4, [this](std::string_view line) { Log(line); });
  const std::string build = KeptBuild();
  const std::int64_t now = NowUnixMs();
  const std::int64_t retention_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(retention_).count();
  std::vector<std::string> kept_dirs;
  std::size_t adopted_count = 0;
  std::uint64_t adopted_bytes = 0;
  for (const auto& m : models_) {
    if (!m->llm()) {
      continue;
    }
    auto& l = static_cast<Llm&>(*m);
    const auto entry = std::ranges::find(config_.models, l.name(), &config::ModelEntry::name);
    const std::string artifact = entry != config_.models.end() ? entry->artifact.value_or("") : "";
    const std::string layout = l.KeptLayout();
    if (!kept::ValidDirectoryName(artifact) || layout.empty()) {
      Log(
          std::format("model {}: its conversations are not kept across a restart (its artifact "
                      "ID names no directory)",
                      l.name()));
      continue;
    }
    auto dir = platform::OpenPrivateDirectory(root, artifact.c_str());
    if (!dir) {
      Log(std::format("model {}: its conversations are not kept across a restart: {}: {}", l.name(),
                      artifact, Errno(dir.error())));
      continue;
    }
    if (const std::string note = platform::PrivacyNote(*dir); !note.empty()) {
      Log(
          std::format("warning: model {}: its kept conversations directory {}/{}/{}: {}; used as "
                      "found",
                      l.name(), roles_.spill.string(), root_name, artifact, note));
    }
    kept_dirs.push_back(artifact);
    // One found that the runtime cannot write (another user's, or its modes
    // changed) keeps nothing and is left as it is: its slots' named spill
    // files could not be made, and the model would not start.
    if (::faccessat(*dir, ".", W_OK | X_OK, AT_EACCESS) != 0) {
      const int error = errno;
      Log(std::format("model {}: its conversations are not kept across a restart: {}/{}/{}: {}",
                      l.name(), roles_.spill.string(), root_name, artifact, Errno(error)));
      (void)::close(*dir);
      continue;
    }
    const std::size_t index = keeper_->AddModel(*dir);
    const kept::Identity identity{
        .build = build,
        .artifact = artifact,
        .drafter = l.speculative() && entry->drafter ? *entry->drafter : std::string(),
        .layout = layout};
    kept::Expected expected{.version = l.KeptVersion(),
                            .identity = identity,
                            .slot = 0,
                            .regions = l.KeptRegions(),
                            .layouts = l.KeptLayouts(),
                            .context = l.usable_context(),
                            .vocabulary = static_cast<std::uint32_t>(l.tokenizer().size()),
                            .now_unix_ms = now,
                            .retention_ms = retention_ms};
    // Each slot's record, checked whole and its files hashed; a record
    // that does not validate is refused and its files emptied.
    std::vector<std::uint32_t> adopt;
    std::vector<std::vector<kept::FileId>> adopt_files;
    std::vector<std::string> keep_files;
    std::vector<PendingAdoption> pending;
    bool deferred = false;
    for (std::uint32_t slot = 0; slot < l.branches(); ++slot) {
      const std::string record_name = kept::RecordFileName(slot);
      auto text = platform::ReadPrivateFile(*dir, record_name.c_str(), kept::kMostRecordBytes);
      if (!text) {
        if (text.error() != ENOENT) {
          Log(std::format("model {}: slot {}'s kept conversation was refused: its record: {}",
                          l.name(), slot, Errno(text.error())));
        }
        continue;
      }
      expected.slot = slot;
      std::vector<std::string> dropped;
      MemoryCharge token_charge;
      bool memory_refused = false;
      auto record = kept::Decode(*text, &token_memory_, &token_charge, &memory_refused);
      std::string why;
      if (memory_refused) {
        // Memory this start lacks, not necessarily a bad record: checked
        // uncharged (bounded as its text is), one this process would adopt
        // keeps the model's records and files for a later start (below).
        std::vector<std::string> unused;
        auto plain = kept::Decode(*text);
        if (!plain) {
          why = plain.error();
        } else if (auto checked = kept::Check(*plain, expected, &unused); !checked) {
          why = checked.error();
        } else {
          Log(std::format("model {}: slot {}'s kept conversation does not fit this start: {}",
                          l.name(), slot, record.error()));
          deferred = true;
          break;
        }
      } else if (!record) {
        why = record.error();
      } else if (auto checked = kept::Check(*record, expected, &dropped); !checked) {
        why = checked.error();
      }
      // The files: the very ones the record names, whole, hashing as it says
      // (version 2: one a stage).
      if (why.empty()) {
        why = VerifyKeptState(*dir, *record);
      }
      // Every extent the record does not list reads as zeros from here on,
      // as a fresh slot's do: a record removed before a crash can come back
      // (its removal not yet on disk), its listed extents still matching
      // while the file's others hold what a later conversation wrote there,
      // which this one would read when it grows into them (D-105). Each
      // adoption empties them again, so their emptying need not be synced.
      if (why.empty()) {
        why = EmptyUnlisted(*dir, *record);
      }
      if (!why.empty()) {
        Log(std::format("model {}: slot {}'s kept conversation was refused: {}", l.name(), slot,
                        why));
        continue;
      }
      std::erase_if(record->checkpoints, [&](const kept::Checkpoint& c) {
        const std::string bad =
            VerifyKeptFile(*dir, c.file, c.id, c.file_bytes, kept::CheckpointPlaces(c), c.digests);
        if (!bad.empty()) {
          dropped.push_back(bad);
        }
        return !bad.empty();
      });
      for (const std::string& line : dropped) {
        Log(std::format("model {}: slot {}: a kept turn checkpoint was left out: {}", l.name(),
                        slot, line));
      }
      adopt.push_back(slot);
      adopt_files.emplace_back();
      for (const kept::StateFile& file : record->files) {
        keep_files.push_back(file.name);
        adopt_files.back().push_back(file.id);
      }
      keep_files.push_back(record_name);
      for (const kept::Checkpoint& c : record->checkpoints) {
        keep_files.push_back(c.file);
      }
      pending.push_back(
          {.model = &l, .token_charge = std::move(token_charge), .record = std::move(*record)});
      StartProgress();
    }
    if (deferred) {
      // Nothing of this model's is adopted, removed or kept this run: its
      // slots spill to unnamed files, so its directory stays as it was
      // until a start with the memory adopts it or its retention ends.
      Log(std::format(
          "model {}: its kept conversations stay for a later start; this run keeps none of its "
          "conversations across a restart",
          l.name()));
      continue;
    }
    for (PendingAdoption& p : pending) {
      adopted_bytes += p.record.extents.size() * kExtent;
      ++adopted_count;
      adoptions_.push_back(std::move(p));
    }
    // Nothing else stays: unadopted slots' files, other records, stale
    // temporary files (D-014: what is kept is explicit).
    if (auto entries = platform::ListPrivateDirectory(*dir); entries) {
      for (const platform::DirectoryEntry& listed : *entries) {
        if (!std::ranges::contains(keep_files, listed.name)) {
          std::ignore = platform::RemovePrivate(*dir, listed.name.c_str());
        }
      }
    }
    l.set_kept({.keeper = keeper_.get(),
                .model = index,
                .directory = *dir,
                .identity = identity,
                .adopt = std::move(adopt),
                .files = std::move(adopt_files)});
  }
  // Models no longer configured keep nothing.
  if (auto entries = platform::ListPrivateDirectory(root); entries) {
    for (const platform::DirectoryEntry& entry : *entries) {
      if (!std::ranges::contains(kept_dirs, entry.name)) {
        std::ignore = platform::RemovePrivate(root, entry.name.c_str());
      }
    }
  }
  (void)::close(root);
  Log(
      std::format("conversations are kept across a restart in {}/{} (D-105): {} adopted ({:.1f} "
                  "MiB of state)",
                  roles_.spill.string(), root_name, adopted_count,
                  static_cast<double>(adopted_bytes) / (1U << 20U)));
  return {};
}

void Server::AdoptKept() {
  // Slots whose spill files could not be named after the start's checks
  // (their directory no longer writable): unnamed, nothing of theirs
  // outlives this run.
  for (const auto& m : models_) {
    if (m->llm()) {
      const auto& l = static_cast<const Llm&>(*m);
      for (const std::string& note : l.UnkeptSlots()) {
        Log(std::format("model {}: {}; its conversations are not kept across a restart", l.name(),
                        note));
      }
    }
  }
  for (PendingAdoption& pending : adoptions_) {
    Llm& l = *pending.model;
    const std::uint32_t slot = pending.record.slot;
    auto branch = l.branch(slot);
    Status adopted = branch ? l.Adopt(**branch, pending.record, pending.token_charge)
                            : Status(std::unexpected(branch.error()));
    if (adopted) {
      Log(std::format(
          "model {}: slot {} adopted a kept conversation of {} tokens ({} turn "
          "checkpoints); its next turn restores it",
          l.name(), slot, (*branch)->history().size(), pending.record.checkpoints.size()));
      continue;
    }
    Log(std::format("model {}: slot {}'s kept conversation was not adopted: {}", l.name(), slot,
                    adopted.error()));
    // Its file was kept as it is: emptied (a slot's fresh state reads zeros
    // from it), and its record removed.
    if (branch) {
      std::ignore = (*branch)->ReleaseIdleState();
    }
  }
  adoptions_.clear();
  token_memory_.Settle(0, std::chrono::milliseconds{0});
}

void Server::Persist(Clock::time_point deadline, const std::function<void()>& progress) {
  if (keeper_ == nullptr || !started_ || torn_down_) {
    return;
  }
  std::size_t spilled = 0;
  std::uint64_t bytes = 0;
  auto told = Clock::now();
  if (resident_ != nullptr && resident_->llm()) {
    auto& l = static_cast<Llm&>(*resident_);
    // The most recently used first, so those that do not fit are the least
    // recently used.
    std::vector<Llm::Branch*> idle;
    for (std::size_t i = 0; i < l.branches(); ++i) {
      if (auto b = l.branch(i);
          b && !(*b)->history().empty() && l.BranchIdle(**b) && l.ResidentStateBytes(**b) != 0) {
        idle.push_back(*b);
      }
    }
    std::ranges::stable_sort(idle, [&l](const Llm::Branch* a, const Llm::Branch* b) {
      return l.LastUsed(*a) > l.LastUsed(*b);
    });
    for (Llm::Branch* branch : idle) {
      if (Clock::now() >= deadline) {
        break;
      }
      const std::uint64_t held = l.ResidentStateBytes(*branch);
      // Within the spill budget, as every spill, but deleting only spilled
      // state used before it (the least recently used first): one that
      // fits only by displacing a more recent conversation is not kept.
      // Its whole state counts once spilled, not only what the spill
      // writes.
      if (held == 0 || held > spill_budget_ ||
          !KeepWithinSpillBudget(held, {}, l.LastUsed(*branch))) {
        continue;
      }
      if (auto r = l.SpillIdle(*branch); r) {
        ++spilled;
        bytes += held;
      }
      if (progress && Clock::now() - told >= std::chrono::seconds(1)) {
        progress();
        told = Clock::now();
      }
    }
  }
  // The models a swap wrote back have their records already queued. The
  // keeper writes them while it makes progress (bytes hashed, records
  // written), however long that takes; a minute with nothing moving ends
  // the wait (a hung drive), as does the deadline.
  bool drained = false;
  const auto moved = [this] {
    const StateKeeper::Stats s = keeper_->stats();
    return s.hashed_bytes + s.written + s.failed + s.stale;
  };
  std::uint64_t seen = moved();
  auto quiet_until = Clock::now() + std::chrono::minutes(1);
  while (!(drained = keeper_->Drain(std::min(deadline, Clock::now() + std::chrono::seconds(1))))) {
    const auto now = Clock::now();
    if (const std::uint64_t at = moved(); at != seen) {
      seen = at;
      quiet_until = now + std::chrono::minutes(1);
    }
    if (now >= deadline || now >= quiet_until) {
      break;
    }
    if (progress) {
      progress();
    }
  }
  const StateKeeper::Stats stats = keeper_->stats();
  Log(
      std::format("kept for the next start: {} conversations spilled ({:.1f} MiB); records {} "
                  "written in all, {} pending{}; {:.1f} MiB hashed in {:.2f} s",
                  spilled, static_cast<double>(bytes) / (1U << 20U), stats.written,
                  keeper_->pending(), drained ? "" : " (the stop's time ran out)",
                  static_cast<double>(stats.hashed_bytes) / (1U << 20U), stats.hash_seconds));
}

bool Server::DrainKept(Clock::time_point deadline) {
  return keeper_ == nullptr || keeper_->Drain(deadline);
}

std::uint64_t Server::context_body_bytes() const {
  std::uint64_t most = 0;
  for (const auto& m : models_) {
    if (m->llm()) {
      const auto& l = static_cast<const Llm&>(*m);
      most = std::max(most, ContextBodyBytes(l.usable_context(), l.longest_token()));
    }
  }
  return most;
}

std::uint64_t Server::request_capacity() const {
  if (config_.client.request_memory_bytes) {
    return *config_.client.request_memory_bytes;
  }
  return request_memory_ + state_room_;
}

bool Server::SetRequestMemory(std::uint64_t to) {
  if (!started_ || torn_down_) {
    return to <= request_memory_;
  }
  // Past the floor, charged inside the budget; what does not fit has the
  // reclaim order free it (plans, graphs, idle conversations spilled; the
  // request memory itself is never reclaimed), all of it or nothing.
  const std::uint64_t past = to > request_memory_ ? to - request_memory_ : 0;
  std::uint64_t shortfall = 0;
  if (node_.SetRequestCharge(past, &shortfall)) {
    return true;
  }
  for (int attempt = 0; attempt < 3 && shortfall != 0; ++attempt) {
    const std::uint64_t ask = ((shortfall + kExtent - 1) / kExtent * kExtent) + kExtent;
    if (Reclaim(ask, true, "request memory") == 0) {
      break;
    }
    shortfall = 0;
    if (node_.SetRequestCharge(past, &shortfall)) {
      return true;
    }
  }
  return false;
}

bool Server::SetTokenMemory(std::uint64_t incoming) {
  if (!started_ || torn_down_) {
    return false;
  }
  if (!token_catalog_ready_) {
    const auto used = token_memory_.used();
    return used <= startup_token_capacity_ && incoming <= startup_token_capacity_ - used;
  }
  std::uint64_t shortfall = 0;
  for (int attempt = 0; attempt < 4; ++attempt) {
    // Reclaim may have retired histories since this grow began.
    const auto target = token_memory_.used() + incoming;
    if (node_.SetTokenCharge(target, &shortfall)) {
      return true;
    }
    if (attempt == 3 || Reclaim(shortfall, true, "native token history", nullptr, std::nullopt,
                                false, nullptr, incoming) == 0) {
      return false;
    }
  }
  return false;
}

Status Server::TearDown() {
  if (torn_down_) {
    return {};
  }
  if (started_) {
    RecordCalibrations();
    // What the incremental spills and swaps left unwritten, and the
    // reclaims that found too little to take, or ended short when a victim
    // gave back less than it counted (counts and bytes only).
    if (auto stats = node_.Stats(); stats) {
      Log(std::format(
          "write-backs released unchanged (nothing written): {} extents, {:.1f} MiB; reclaims "
          "that took nothing (too little to take): {}; that ended short (a victim gave back "
          "less): {}",
          stats->unchanged_writebacks,
          static_cast<double>(stats->unchanged_writeback_bytes) / (1U << 20U), reclaims_short_,
          reclaims_cut_short_));
    }
    // What each model's plans and graphs held at the end, and what bringing
    // them back would have cost (the reclaim order's measured inputs):
    // counts, bytes and seconds only.
    for (std::uint32_t i = 0; i < models_.size(); ++i) {
      std::vector<memory::ReclaimCandidate> held;
      models_[i]->ReclaimCandidates(i, false, held);
      std::array<std::uint64_t, memory::kReclaimKinds> count{};
      std::array<std::uint64_t, memory::kReclaimKinds> bytes{};
      std::array<double, memory::kReclaimKinds> seconds{};
      for (const memory::ReclaimCandidate& c : held) {
        const auto k = static_cast<std::size_t>(c.kind);
        ++count[k];
        bytes[k] += c.bytes;
        seconds[k] += c.restore_seconds;
      }
      const auto p = static_cast<std::size_t>(memory::ReclaimKind::kPlan);
      const auto g = static_cast<std::size_t>(memory::ReclaimKind::kGraph);
      if (count[p] + count[g] != 0) {
        // A plan's count includes its graphs'; a graph may have no plan
        // candidate (the image pipeline's recorded step).
        const std::array<double, memory::kReclaimKinds> cost = memory::KindCosts(held);
        const auto mib = [](std::uint64_t b) { return static_cast<double>(b) / (1U << 20U); };
        std::uint64_t plan_bytes = 0;
        double plan_seconds = 0;
        for (const memory::ReclaimCandidate& c : held) {
          if (c.kind != memory::ReclaimKind::kPlan) {
            continue;
          }
          std::uint64_t own = c.bytes;
          double own_seconds = c.restore_seconds;
          for (const memory::ReclaimCandidate& graph : held) {
            if (graph.kind == memory::ReclaimKind::kGraph && graph.id == c.id) {
              own -= std::min(own, graph.bytes);
              own_seconds -= std::min(own_seconds, graph.restore_seconds);
            }
          }
          plan_bytes += own;
          plan_seconds += own_seconds;
        }
        Log(std::format(
            "model {}: {} plans held {:.1f} MiB (planned in {:.3f} s, {:.2f} s a GiB), "
            "{} graphs {:.1f} MiB counted, {:.1f} MiB measured at their captures (captured in "
            "{:.3f} s, {:.2f} s a GiB)",
            models_[i]->name(), count[p], mib(plan_bytes), plan_seconds, cost[p], count[g],
            mib(bytes[g]), mib(models_[i]->graph_measured_bytes()), seconds[g], cost[g]));
      }
    }
  }
  torn_down_ = true;
  std::vector<engine::PagedModel*> models;
  models.reserve(models_.size());
  for (const auto& m : models_) {
    models.push_back(&m->paged());
    // Kept checkpoints' files stay for the next start (D-105): those no
    // record names it removes.
    if (m->llm()) {
      static_cast<Llm&>(*m).PreserveKeptFiles();
    }
  }
  const auto stopped = node_.TearDown(models);
  if (!stopped && node_.has_pending_state_preparation()) {
    // Server is stack-owned by its frontends. Returning would destroy a model
    // whose independent page-ins have no proven no-access fence. Stop before
    // any owner/resource destructor; a LiveState destructor is too late.
    Log("unretired state preparation; aborting before owner destruction");
    std::abort();
  }
  if (options_.diagnostic_budget_cap_bytes) {
    const auto c = node_.backing_create_stats();
    const auto h = node_.retired_harvest_stats();
    const auto harvest =
        h ? std::format("{{\"indices\":{},\"pop_locks\":{}}}", h->indices, h->pop_locks) : "null";
    Log(std::format(
        "diagnostic final backing creates: {{\"retired\":{},\"reserve_attempts\":{},"
        "\"reserve_failures\":{},\"ordinary_attempts\":{},\"ordinary_failures\":{},"
        "\"last_failure_monotonic_ns\":{},\"harvest\":{}}}",
        stopped.has_value(), c.reserve_attempts, c.reserve_failures, c.ordinary_attempts,
        c.ordinary_failures, c.last_failure_monotonic_ns, harvest));
  }
  return stopped;
}

Served* Server::Find(std::string_view name) {
  for (const auto& m : models_) {
    if (m->name() == name) {
      return m.get();
    }
  }
  return nullptr;
}

Status Server::InRequest(Served& m, const std::function<Status()>& body) {
  if (auto selected = m.PrepareDefaultRequest(); !selected) {
    return selected;
  }
  auto ran = node_.WithRequest(m.paged().stream(), m.request_closure(),
                               std::format("{}'s request", m.name()), body);
  if (ran) return RecordWeightUse(m);
  return ran;
}

Status Server::RecordWeightUse(Served& m) {
  const auto weights = m.weights();
  return node_.Call(
      [&]() -> Status {
        std::uint64_t tick = 0;
        for (const auto extent : weights) {
          if (const auto view = node_.catalog().Describe(extent))
            tick = std::max(tick, view->last_use);
        }
        m.RecordWeightsUsed(tick);
        return {};
      },
      "recording a completed model's weight use");
}

Status Server::SelectRequestBranches(Llm& model, std::span<Llm::Branch* const> active) {
  if (!started_ || torn_down_ || resident_ != &model || active.empty() ||
      std::ranges::none_of(models_, [&](const auto& owned) { return owned.get() == &model; })) {
    return Error("a cooperative request needs its resident model and active branches");
  }
  if (auto selected = model.SelectBranches(active); !selected) {
    return selected;
  }
  const auto stream = model.paged().stream();
  if (node_.InRequest(stream)) {
    return node_.RefreshRequest(stream, model.request_closure());
  }
  return node_.BeginRequest(stream, model.request_closure(),
                            std::format("{}'s request cohort", model.name()));
}

Status Server::EndRequestBranches(Llm& model) {
  if (!started_ || torn_down_ || resident_ != &model) {
    return Error("a retiring request cohort does not own the resident model");
  }
  const auto stream = model.paged().stream();
  // A proven failed job or state helper can already have ended its request.
  // The caller must separately prove that no work still borrows its owners.
  if (node_.InRequest(stream)) {
    if (auto ended = node_.EndRequest(stream); !ended) return ended;
  }
  return RecordWeightUse(model);
}

Server::ReferenceRetirement Server::RetireRequestBranches(Llm& model, bool last_owner) {
  if (!started_ || torn_down_ || resident_ != &model || !model.generation_cohort_usable()) {
    return {.result = Error("the native cohort cannot prove reference retirement"),
            .references_retired = false};
  }
  const auto stream = model.paged().stream();
  bool queued = false;
  auto fenced = node_.Job(
      model.request_closure(),
      [&queued](providers::NativeStream) {
        queued = true;
        // kQueued requires the DeviceLane to record and await its completion
        // fence, even though this job queues no arithmetic or copies itself.
        return scheduler::JobResult::kQueued;
      },
      "retiring native request-owner references", stream);
  if (!fenced || !queued) {
    return {.result = !fenced ? std::move(fenced) : Error("the retirement fence was not queued"),
            .references_retired = false};
  }
  // The known-complete stream fence is the proof. An EndRequest error does
  // not revoke it: EndRequest itself awaits ProgramDone::gone before return.
  return {.result = last_owner ? EndRequestBranches(model) : Status{}, .references_retired = true};
}

Status Server::WaitReleased() {
  // Progress-based (D-102): only evictions that stop finishing for the
  // node's quiet time are given up on, however long the whole takes.
  auto give_up = Clock::now() + node_.quiet();
  std::size_t before = std::numeric_limits<std::size_t>::max();
  for (;;) {
    std::size_t left = 0;
    if (auto r = node_.Call(
            [&]() -> Status {
              left = node_.scheduler().evictions();
              return {};
            },
            "counting evictions");
        !r) {
      return r;
    }
    if (left == 0) {
      return {};
    }
    if (left < before) {
      before = left;
      give_up = Clock::now() + node_.quiet();
    } else if (Clock::now() > give_up) {
      return Error(
          "backing no load took was not released: no eviction finished for the node's "
          "quiet time");
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
}

Status Server::FinishSwap(SwapParts& parts) {
  if (auto r = WaitReleased(); !r) {
    return r;
  }
  parts.release = Seconds(Clock::now() - parts.ready);
  auto after = node_.Stats();
  if (!after) {
    return std::unexpected(after.error());
  }
  parts.handed_off = after->handed_off - swap_before_.handed_off;
  parts.released_unused = after->released_unused - swap_before_.released_unused;
  parts.occupancy_released = after->catalog_occupancy_bytes;
  if (parts.diagnostic.enabled) {
    parts.diagnostic.finished = swap_host_phases_;
    parts.diagnostic.creates_finished = node_.backing_create_stats();
  }
  return {};
}

std::uint64_t Server::SpilledBytes(SpillView view) {
  std::uint64_t bytes = 0;
  for (const auto& m : models_) {
    if (!m->llm()) {
      continue;
    }
    auto& l = static_cast<Llm&>(*m);
    // A model a swap wrote back holds all its state on disk until it returns.
    const bool written_back =
        m.get() != view.in &&
        (m.get() == view.out || std::ranges::find(spilled_, m.get()) != spilled_.end());
    for (std::size_t i = 0; i < l.branches(); ++i) {
      if (auto b = l.branch(i); b) {
        bytes += written_back ? l.StateBytes(**b) : l.SpilledStateBytes(**b);
      }
    }
  }
  return bytes;
}

void Server::ForEachDeletableSpill(
    SpillView view, std::optional<Clock::time_point> older_than,
    const std::function<void(Llm& model, Llm::Branch& branch, std::uint64_t counted)>& each) {
  for (const auto& m : models_) {
    if (!m->llm() || m.get() == view.in) {
      continue;  // the incoming model's stop counting once it is resident
    }
    auto& l = static_cast<Llm&>(*m);
    const bool written_back =
        m.get() == view.out || std::ranges::find(spilled_, m.get()) != spilled_.end();
    for (std::size_t i = 0; i < l.branches(); ++i) {
      auto b = l.branch(i);
      if (!b || !l.BranchIdle(**b) || (*b)->HeldByContinuation() ||
          (older_than && l.LastUsed(**b) >= *older_than)) {
        continue;
      }
      const std::uint64_t counted = written_back ? l.StateBytes(**b) : l.SpilledStateBytes(**b);
      if (counted != 0) {
        each(l, **b, counted);
      }
    }
  }
}

bool Server::SpillBudgetFits(std::uint64_t extra, SpillView view,
                             std::optional<Clock::time_point> older_than) {
  const std::uint64_t held = SpilledBytes(view);
  std::uint64_t deletable = 0;
  ForEachDeletableSpill(view, older_than,
                        [&](Llm&, Llm::Branch&, std::uint64_t counted) { deletable += counted; });
  const std::uint64_t left = held - std::min(held, deletable);
  return left <= spill_budget_ && extra <= spill_budget_ - left;
}

bool Server::KeepWithinSpillBudget(std::uint64_t extra, SpillView view,
                                   std::optional<Clock::time_point> older_than) {
  // Nothing deleted for a conversation (or a swap) that would not fit even
  // with every deletable one gone.
  return SpillBudgetFits(extra, view, older_than) && DeleteSpilledPast(extra, view, older_than);
}

bool Server::DeleteSpilledPast(std::uint64_t extra, SpillView view,
                               std::optional<Clock::time_point> older_than) {
  std::size_t deleted = 0;
  std::uint64_t deleted_bytes = 0;
  bool fits = false;
  for (;;) {
    const std::uint64_t held = SpilledBytes(view);
    if (held <= spill_budget_ && extra <= spill_budget_ - held) {
      fits = true;
      break;
    }
    // The least recently used spilled conversation no request holds.
    Llm* oldest = nullptr;
    Llm::Branch* branch = nullptr;
    ForEachDeletableSpill(view, older_than, [&](Llm& l, Llm::Branch& b, std::uint64_t) {
      if (branch == nullptr || l.LastUsed(b) < oldest->LastUsed(*branch)) {
        oldest = &l;
        branch = &b;
      }
    });
    if (branch == nullptr) {
      break;  // nothing more to delete: still short
    }
    const std::uint64_t bytes = oldest->StateBytes(*branch);
    if (auto released = branch->ReleaseIdleState(); !released) {
      Log(std::format("{}'s spilled conversation could not be deleted: {}", oldest->name(),
                      released.error()));
      break;
    }
    ++deleted;
    deleted_bytes += bytes;
  }
  if (deleted != 0) {
    Log(std::format(
        "deleted {} spilled conversations ({:.1f} MiB), the least recently used, "
        "past the spill budget of {} GiB",
        deleted, static_cast<double>(deleted_bytes) / (1U << 20U), spill_budget_ >> 30U));
  }
  return fits;
}

void AddIdleStateCandidates(Llm& model, std::uint32_t owner, bool running, const Llm::Branch* spare,
                            const IdleStateRates& rates,
                            std::vector<memory::ReclaimCandidate>& out) {
  for (std::size_t slot = 0; slot < model.branches(); ++slot) {
    auto b = model.branch(slot);
    if (!b || *b == spare || !model.BranchIdle(**b) || (*b)->history().empty()) {
      continue;
    }
    const std::uint64_t bytes = model.ResidentStateBytes(**b);
    if (bytes == 0) {
      continue;
    }
    // Spilled whole within the budget (Server::Reclaim); past all of it,
    // dropped, which a continuation holding it refuses.
    const bool dropped = bytes > rates.spill_budget;
    if (dropped && (*b)->HeldByContinuation()) {
      continue;
    }
    // A spill writes only what changed since its last one; a restore reads
    // it all.
    const auto size = static_cast<double>(bytes);
    const auto writes = static_cast<double>(model.SpillWriteBytes(**b));
    out.push_back(
        {.kind = memory::ReclaimKind::kIdleState,
         .owner = owner,
         .id = slot,
         .bytes = bytes,
         .last_use = static_cast<std::uint64_t>(model.LastUsed(**b).time_since_epoch().count()),
         .restore_seconds = dropped ? static_cast<double>((*b)->history().size()) *
                                          model.settings().recompute_ms_per_token.value / 1000.0
                                    : (writes / rates.spill_rate) + (size / rates.restore_rate),
         .running = running});
    memory::SetUse(out.back(), model.LastStamp(**b));
  }
}

bool RoomFor(std::uint64_t needed, std::optional<std::uint64_t> free,
             const std::function<std::uint64_t(std::uint64_t)>& reclaim) {
  if (needed == 0 || !free || *free >= needed) {
    return true;
  }
  return reclaim(needed - *free) >= needed - *free;
}

bool Server::RoomFor(std::uint64_t needed, std::string_view why, const Llm::Branch* spare) {
  if (needed == 0 || !started_ || torn_down_) {
    return true;
  }
  auto free = node_.FreeBytes();
  return runtime::RoomFor(
      needed, free ? std::optional(*free) : std::nullopt, [&](std::uint64_t shortfall) {
        return Reclaim(shortfall, true, why, nullptr, std::nullopt, false, spare);
      });
}

std::uint64_t Server::Reclaim(std::uint64_t needed, bool states, std::string_view why,
                              const Served* running, std::optional<memory::ReclaimKind> below_kind,
                              bool partial, const Llm::Branch* spare, std::uint64_t token_incoming,
                              std::span<const catalog::ExtentId> protect,
                              PendingWeightEvictions* pending, bool allow_weights) {
  if (reclaiming_ || !started_ || torn_down_ || needed == 0) {
    return 0;
  }
  reclaiming_ = true;
  const bool diagnostic = options_.diagnostic_budget_cap_bytes.has_value();
  if (diagnostic) ++swap_host_phases_.reclaim_calls;
  if (running == nullptr) {
    running = resident_;
  }
  // Kept zeroed state backing holds nothing: it goes first, before the
  // order prices anything (other models' before the running one's).
  std::set<catalog::ExtentId> protected_ids(activation_protect_.begin(), activation_protect_.end());
  protected_ids.insert(protect.begin(), protect.end());
  const std::vector<catalog::ExtentId> protected_extents(protected_ids.begin(),
                                                         protected_ids.end());
  const std::uint64_t kept = ReleaseKept(needed, running, protected_extents);
  if (kept >= needed) {
    reclaiming_ = false;
    return kept;
  }
  // Idle state's cost at the rates measured so far (once a GiB has moved).
  double spill_rate = kSpillBytesPerSecond;
  double restore_rate = kRestoreBytesPerSecond;
  {
    Llm::SpillStats all;
    for (const auto& m : models_) {
      if (m->llm()) {
        const Llm::SpillStats& s = static_cast<Llm&>(*m).spill_stats();
        all.spilled_bytes += s.spilled_bytes;
        all.spill_seconds += s.spill_seconds;
        all.restored_bytes += s.restored_bytes;
        all.restore_seconds += s.restore_seconds;
      }
    }
    constexpr std::uint64_t kEnough = std::uint64_t{1} << 30U;
    if (all.spilled_bytes >= kEnough && all.spill_seconds > 0) {
      spill_rate = static_cast<double>(all.spilled_bytes) / all.spill_seconds;
    }
    if (all.restored_bytes >= kEnough && all.restore_seconds > 0) {
      restore_rate = static_cast<double>(all.restored_bytes) / all.restore_seconds;
    }
  }
  const auto k = [](memory::ReclaimKind kind) { return static_cast<std::size_t>(kind); };
  std::array<std::uint64_t, memory::kReclaimKinds> count{};
  std::array<std::uint64_t, memory::kReclaimKinds> freed_by{};
  std::array<double, memory::kReclaimKinds> cost{};
  std::size_t dropped = 0;
  bool took = false;
  Llm* idle_owner = nullptr;
  struct HistoryVictim {
    Llm* model;
    Llm::Branch* branch;
    std::uint32_t owner;
    std::uint64_t slot;
  };
  std::vector<HistoryVictim> histories;
  std::vector<TokenReclaimGroup> history_groups;
  struct WeightView {
    std::uint32_t owner = 0;
    std::uint64_t generation = 0;
    std::uint64_t bytes = 0;
  };
  std::map<catalog::ExtentId, WeightView> weight_views;
  const double weight_rate = page_read_bytes_ >= (std::uint64_t{1} << 30U) && page_read_seconds_ > 0
                                 ? static_cast<double>(page_read_bytes_) / page_read_seconds_
                                 : kWeightReadBytesPerSecond;
  // The candidates afresh each round (memory::RunReclaim): every model's
  // plans and graphs, and with `states` the resident model's idle state.
  const auto gather = [&](std::vector<memory::ReclaimCandidate>& candidates, double& below) {
    const SwapPhaseTimer timer(diagnostic ? &swap_host_phases_.gather_seconds : nullptr);
    idle_owner = nullptr;
    histories.clear();
    history_groups.clear();
    weight_views.clear();
    std::uint32_t resident_index = 0;
    for (std::uint32_t i = 0; i < models_.size(); ++i) {
      models_[i]->ReclaimCandidates(i, models_[i].get() == running, candidates);
      if (states && models_[i]->llm()) {
        auto& model = static_cast<Llm&>(*models_[i]);
        for (std::size_t slot = 0; slot < model.branches(); ++slot) {
          auto branch = model.branch(slot);
          if (!branch || *branch == spare || !model.HistoryReclaimable(**branch)) {
            continue;
          }
          const auto bytes = model.IdleHistoryBytes(**branch);
          if (bytes == 0) {
            continue;
          }
          histories.push_back({.model = &model, .branch = *branch, .owner = i, .slot = slot});
        }
      }
      if (models_[i].get() == resident_) {
        resident_index = i;
      }
    }
    if (options_.partial_weight_eviction && allow_weights) {
      std::set<catalog::ExtentId> pending_ids;
      if (pending != nullptr) {
        pending_ids = pending->rejected;
        for (const auto& selected : pending->selected) pending_ids.insert(selected.extent);
      }
      std::vector<std::vector<catalog::ExtentId>> weights;
      weights.reserve(models_.size());
      for (const auto& model : models_) {
        weights.push_back(model.get() == running ? std::vector<catalog::ExtentId>{}
                                                 : model->weights());
      }
      const auto listed = node_.Call(
          [&]() -> Status {
            for (std::uint32_t i = 0; i < weights.size(); ++i) {
              for (const auto extent : weights[i]) {
                if (protected_ids.contains(extent) || pending_ids.contains(extent)) continue;
                const auto view = node_.catalog().Describe(extent);
                const auto* source = node_.scheduler().SourceOf(extent);
                if (!view || !catalog::Catalog::Evictable(*view) || source == nullptr ||
                    view->descriptor.memory_class != catalog::MemoryClass::kWeights ||
                    view->descriptor.recovery != catalog::Recovery::kFromArtifact)
                  continue;
                if (!weight_views
                         .try_emplace(extent, WeightView{.owner = i,
                                                         .generation = view->content_generation,
                                                         .bytes = view->descriptor.size.value()})
                         .second) {
                  return Error("a prepared weight extent has more than one owner");
                }
                candidates.push_back(
                    {.kind = memory::ReclaimKind::kIdleWeights,
                     .owner = i,
                     .id = WeightHandle(extent),
                     .bytes = view->descriptor.size.value(),
                     .last_use = view->last_use,
                     .restore_seconds = static_cast<double>(source->read.length) / weight_rate});
                memory::SetUse(candidates.back(), models_[i]->weight_use());
              }
            }
            return {};
          },
          "gathering inactive weight reclaim candidates");
      if (!listed) {
        Log(std::format("listing inactive weights: {}", listed.error()));
        std::erase_if(candidates,
                      [](const auto& c) { return c.kind == memory::ReclaimKind::kIdleWeights; });
        weight_views.clear();
      }
    }
    std::ranges::sort(histories, [](const HistoryVictim& a, const HistoryVictim& b) {
      return a.model->LastUsed(*a.branch) < b.model->LastUsed(*b.branch);
    });
    std::vector<std::uint64_t> capacities;
    capacities.reserve(histories.size());
    for (const auto& history : histories) {
      capacities.push_back(history.model->IdleHistoryBytes(*history.branch));
    }
    history_groups =
        GroupTokenReclaim(capacities, token_memory_.used() + token_incoming, kExtent, needed);
    std::size_t first = 0;
    for (const auto& group : history_groups) {
      const auto& oldest = histories[first];
      double seconds = 0;
      bool group_running = false;
      for (std::size_t i = first; i < group.end; ++i) {
        const auto& h = histories[i];
        seconds += static_cast<double>(h.branch->history().size()) *
                   h.model->settings().recompute_ms_per_token.value / 1000.0;
        group_running = group_running || h.model == running;
      }
      candidates.push_back({.kind = memory::ReclaimKind::kTokenHistory,
                            .owner = oldest.owner,
                            .id = oldest.slot,
                            .bytes = group.catalog_bytes,
                            .last_use = static_cast<std::uint64_t>(
                                oldest.model->LastUsed(*oldest.branch).time_since_epoch().count()),
                            .restore_seconds = seconds,
                            .running = group_running});
      memory::SetUse(candidates.back(), oldest.model->LastStamp(*oldest.branch));
      first = group.end;
    }
    // The running model's next step finds what its last ones used.
    memory::ProtectFloor(candidates, running != nullptr ? running->plan_floor_bytes() : 0);
    if (states && resident_ != nullptr && resident_->llm()) {
      idle_owner = static_cast<Llm*>(resident_);
      AddIdleStateCandidates(
          *idle_owner, resident_index, resident_ == running, spare,
          {.spill_rate = spill_rate, .restore_rate = restore_rate, .spill_budget = spill_budget_},
          candidates);
    }
    cost = memory::KindCosts(candidates);
    if (below_kind) {
      // What the charge would cost to restore a GiB: its kind's measured
      // cost, or the GB10's when none is held.
      double charged = cost[k(*below_kind)];
      if (charged == 0) {
        charged = *below_kind == memory::ReclaimKind::kGraph ? kGraphSecondsPerGiB : 0;
      }
      below = memory::ReclaimInflation() + charged;
    }
  };
  const auto take = [&](const memory::ReclaimCandidate& c) -> std::uint64_t {
    const SwapPhaseTimer timer(diagnostic ? &swap_host_phases_.take_seconds : nullptr);
    std::uint64_t got = 0;
    if (c.kind == memory::ReclaimKind::kTokenHistory) {
      std::size_t first = 0;
      const auto prospective = [&]() {
        const auto target = token_memory_.used() + token_incoming;
        return (target / kExtent + (target % kExtent != 0 ? 1U : 0U)) * kExtent;
      };
      const auto before = prospective();
      const auto charged_before = node_.token_charged();
      const auto free_before = node_.FreeBytes();
      for (const auto& group : history_groups) {
        const auto& oldest = histories[first];
        if (oldest.owner == c.owner && oldest.slot == c.id) {
          for (std::size_t i = first; i < group.end; ++i) {
            auto& h = histories[i];
            if (h.branch != spare && h.model->HistoryReclaimable(*h.branch)) {
              std::ignore = h.branch->ReleaseIdleState();
            }
          }
          break;
        }
        first = group.end;
      }
      const auto after = prospective();
      const auto free_after = node_.FreeBytes();
      const auto actual_tokens = charged_before - std::min(charged_before, node_.token_charged());
      // Dropping a history can also release its resident native state. Count
      // the measured catalog gain once, including overlapping idle victims.
      got = free_before && free_after && *free_after >= *free_before ? *free_after - *free_before
                                                                     : actual_tokens;
      if (token_incoming != 0) {
        const auto target_drop = before - std::min(before, after);
        // The incoming allocation can change which rounded boundary matters.
        // Replace actual token release with the prospective target decrease.
        got = TokenReclaimCredit(got, actual_tokens, target_drop);
      }
    } else if (c.kind == memory::ReclaimKind::kIdleState) {
      if (idle_owner == nullptr) {
        return 0;  // only the resident model's idle state is a candidate
      }
      auto b = idle_owner->branch(c.id);
      if (!b) {
        return 0;
      }
      // Spill, not clear: within the spill budget (the least recently used
      // spilled state deleted first; once spilled its whole state counts);
      // past it, or with none, dropped. A state larger than the whole
      // budget deletes nothing else first.
      const std::uint64_t held = idle_owner->ResidentStateBytes(**b);
      if (held == 0) {
        return 0;  // gone meanwhile: nothing to give back
      }
      if (held <= spill_budget_ && KeepWithinSpillBudget(held)) {
        if (auto spilled = idle_owner->SpillIdle(**b); spilled) {
          got = c.bytes;
        } else {
          Log(std::format("{}'s idle conversation in slot {}: {}", idle_owner->name(), c.id,
                          spilled.error()));
          got = idle_owner->ResidentStateBytes(**b) == 0 ? c.bytes : 0;
        }
      } else if (auto released = (*b)->ReleaseIdleState(); released) {
        got = c.bytes;
        ++dropped;
      }
    } else {
      std::optional<std::uint64_t> before_charge;
      if (options_.partial_weight_eviction) {
        std::ignore = node_.Call(
            [&]() -> Status {
              before_charge = node_.catalog().OccupancyOf(node_.domain()).Total().value();
              return {};
            },
            "sizing a cache reclaim's catalog charge");
      }
      got = models_[c.owner]->Reclaim(c.kind, c.id);
      if (before_charge) {
        const auto measured = node_.Call(
            [&]() -> Status {
              const auto after = node_.catalog().OccupancyOf(node_.domain()).Total().value();
              got = *before_charge - std::min(*before_charge, after);
              return {};
            },
            "measuring a cache reclaim's catalog gain");
        if (!measured) got = 0;
      }
    }
    if (got != 0) {
      took = true;
      ++count[k(c.kind)];
      freed_by[k(c.kind)] += got;
    }
    return got;
  };
  const auto take_weights = [&](std::span<const memory::ReclaimCandidate> victims,
                                std::span<std::uint64_t> freed) {
    const SwapPhaseTimer timer(diagnostic ? &swap_host_phases_.take_seconds : nullptr);
    std::vector<catalog::ExtentId> eligible;
    const auto listed = node_.Call(
        [&]() -> Status {
          for (const auto& c : victims) {
            const auto extent = WeightExtent(c.id);
            const auto known = weight_views.find(extent);
            const auto view = node_.catalog().Describe(extent);
            if (known != weight_views.end() && known->second.owner == c.owner && view &&
                view->content_generation == known->second.generation &&
                view->descriptor.size.value() == c.bytes && catalog::Catalog::Evictable(*view) &&
                !protected_ids.contains(extent))
              eligible.push_back(extent);
          }
          return {};
        },
        "rechecking selected idle weights");
    if (!listed || eligible.empty()) return;
    const auto evicted = node_.Evict(eligible);
    if (!evicted) Log(std::format("reclaiming idle weights: {}", evicted.error()));
    const std::set<catalog::ExtentId> attempted(eligible.begin(), eligible.end());
    const auto reconciled = node_.Call(
        [&]() -> Status {
          for (std::size_t i = 0; i < victims.size(); ++i) {
            const auto extent = WeightExtent(victims[i].id);
            const auto view = node_.catalog().Describe(extent);
            if (attempted.contains(extent) && view &&
                view->state == catalog::ExtentState::kNonresident)
              freed[i] = victims[i].bytes;
          }
          return {};
        },
        "reconciling completed idle weight releases");
    if (!reconciled) return;
    for (const auto bytes : freed) {
      if (bytes != 0) {
        took = true;
        ++count[k(memory::ReclaimKind::kIdleWeights)];
        freed_by[k(memory::ReclaimKind::kIdleWeights)] += bytes;
      }
    }
  };
  // All of it or nothing (memory::RunReclaim): a selection that cannot
  // cover what is needed reclaims nothing (the caller waits or refuses);
  // another round, without it, only when a victim gave back less than its
  // count (held, or gone meanwhile), and what the rounds before took then
  // stays taken even when the rest cannot be covered.
  memory::ReclaimRun run;
  if (pending == nullptr) {
    const auto begin = diagnostic ? Clock::now() : Clock::time_point{};
    const auto gather_before = swap_host_phases_.gather_seconds;
    const auto take_before = swap_host_phases_.take_seconds;
    run = memory::RunReclaim(needed - kept, partial, gather, take, take_weights);
    if (diagnostic) {
      swap_host_phases_.selection_bookkeeping_seconds += std::max(
          0.0, Seconds(Clock::now() - begin) - (swap_host_phases_.gather_seconds - gather_before) -
                   (swap_host_phases_.take_seconds - take_before));
    }
  } else {
    std::vector<memory::ReclaimCandidate> candidates;
    double below = std::numeric_limits<double>::infinity();
    gather(candidates, below);
    memory::ReclaimPlan plan;
    {
      const SwapPhaseTimer timer(diagnostic ? &swap_host_phases_.selection_bookkeeping_seconds
                                            : nullptr);
      plan = memory::SelectReclaim(candidates, needed - kept, below);
    }
    if (!plan.sufficient) {
      run.short_of_need = true;
    } else {
      std::uint64_t pending_credit = 0;
      for (std::size_t i = 0; i < plan.victims.size() && run.freed + pending_credit < needed - kept;
           ++i) {
        const auto& c = candidates[plan.victims[i]];
        if (c.kind == memory::ReclaimKind::kIdleWeights) {
          const auto extent = WeightExtent(c.id);
          const auto known = weight_views.find(extent);
          if (known != weight_views.end()) {
            pending->selected.push_back({.extent = extent,
                                         .generation = known->second.generation,
                                         .candidate = c,
                                         .priority = plan.priorities[i]});
            pending_credit += c.bytes;
          }
        } else {
          const auto got = take(c);
          if (got != 0) {
            memory::RaiseReclaimInflation(plan.priorities[i]);
            run.freed += got;
          }
        }
      }
    }
  }
  const std::uint64_t freed = run.freed + kept;
  if (freed == 0 && run.short_of_need) {
    ++reclaims_short_;
  } else if (pending == nullptr && !partial && freed != 0 && freed < needed) {
    ++reclaims_cut_short_;
  }
  if (freed != 0) {
    const SwapPhaseTimer timer(diagnostic ? &swap_host_phases_.heap_release_seconds : nullptr);
    if (diagnostic) ++swap_host_phases_.heap_release_calls;
    platform::ReleaseFreeHeap();
  }
  if (took) {
    const auto mib = [](std::uint64_t b) { return static_cast<double>(b) / (1U << 20U); };
    using K = memory::ReclaimKind;
    // Counts, bytes and measured costs only (D-014).
    Log(std::format(
        "reclaimed {:.1f} MiB of {:.1f} MiB needed for {}: {} graphs ({:.1f} MiB), {} plans "
        "({:.1f} MiB), {} idle weights ({:.1f} MiB), {} idle conversations {} ({:.1f} MiB); "
        "measured s a GiB: graphs {:.2f}, "
        "plans {:.2f}, idle state {:.2f}; {} history groups discarded ({:.1f} MiB capacity)",
        mib(freed), mib(needed), why, count[k(K::kGraph)], mib(freed_by[k(K::kGraph)]),
        count[k(K::kPlan)], mib(freed_by[k(K::kPlan)]), count[k(K::kIdleWeights)],
        mib(freed_by[k(K::kIdleWeights)]), count[k(K::kIdleState)],
        dropped != 0 ? "dropped" : "spilled", mib(freed_by[k(K::kIdleState)]), cost[k(K::kGraph)],
        cost[k(K::kPlan)], cost[k(K::kIdleState)], count[k(K::kTokenHistory)],
        mib(freed_by[k(K::kTokenHistory)])));
  }
  reclaiming_ = false;
  return freed;
}

Status Server::MakeRoomForSwap(Served& m, std::span<const catalog::ExtentId> out,
                               PendingWeightEvictions* pending) {
  const SwapPhaseTimer timer(options_.diagnostic_budget_cap_bytes ? &swap_host_phases_.room_seconds
                                                                  : nullptr);
  // Size physical occupancy afresh, including required charges above B.
  const auto add = [](std::uint64_t& value, std::uint64_t bytes) {
    if (bytes > UINT64_MAX - value) return false;
    value += bytes;
    return true;
  };
  for (int round = 0; round < memory::kReclaimRounds + 1; ++round) {
    std::uint64_t need = 0;
    std::uint64_t credit = 0;
    auto sized = node_.Call(
        [&]() -> Status {
          const auto& catalog = node_.catalog();
          std::uint64_t after = catalog.OccupancyOf(node_.domain()).Total().value();
          for (const auto& [extent, generation] : m.everything().extents) {
            const auto view = catalog.Describe(extent);
            if (!view || view->descriptor.domain != node_.domain() ||
                view->content_generation != generation || view->discarded ||
                view->state == catalog::ExtentState::kQuarantined)
              return Error("a swap's incoming closure is stale or invalid");
            if (view->state != catalog::ExtentState::kResident &&
                !add(after, view->descriptor.size.value()))
              return Error("a swap's incoming occupancy overflows");
          }
          std::set<catalog::ExtentId> counted;
          for (const auto extent : out) {
            const auto view = catalog.Describe(extent);
            if (counted.insert(extent).second && view &&
                view->state == catalog::ExtentState::kResident &&
                !add(credit, view->descriptor.size.value()))
              return Error("a swap's outgoing credit overflows");
          }
          need = after > budget_ ? after - budget_ : 0;
          if (pending != nullptr) {
            std::erase_if(pending->selected, [&](const auto& selected) {
              const auto view = catalog.Describe(selected.extent);
              const bool valid = view && view->content_generation == selected.generation &&
                                 view->descriptor.size.value() == selected.candidate.bytes &&
                                 catalog::Catalog::Evictable(*view) &&
                                 !counted.contains(selected.extent);
              if (!valid) pending->rejected.insert(selected.extent);
              return !valid;
            });
            // Keep only the policy-ordered prefix still needed after actual
            // cache-charge rounding. A single final extent may overshoot.
            std::size_t keep = 0;
            while (keep < pending->selected.size() && credit < need) {
              const auto& selected = pending->selected[keep++];
              if (!add(credit, selected.candidate.bytes))
                return Error("a swap's selected weight credit overflows");
            }
            pending->selected.resize(keep);
          }
          return {};
        },
        "sizing actual missing incoming and selected swap extents");
    if (!sized) return sized;
    if (credit >= need) return {};
    if (round == memory::kReclaimRounds) break;
    const auto selected_before = pending == nullptr ? 0 : pending->selected.size();
    const auto freed = Reclaim(need - credit, false, "a swap's incoming model", &m, std::nullopt,
                               false, nullptr, 0, {}, pending);
    if (freed == 0 && (pending == nullptr || pending->selected.size() == selected_before)) break;
  }
  return Error(std::format("{} to {}: the incoming closure cannot fit without held extents",
                           resident_ != nullptr ? resident_->name() : "a swap", m.name()));
}

bool Server::NodeHealthy() {
  if (!started_ || torn_down_) {
    return false;
  }
  auto checked = node_.Call(
      [&]() -> Status {
        return node_.scheduler().fault() ? Error("the node's scheduler faulted") : Status{};
      },
      "checking the node's health");
  return checked.has_value();
}

std::uint64_t Server::ReleaseKept(std::uint64_t needed, const Served* last,
                                  std::span<const catalog::ExtentId> protect) {
  const std::set<catalog::ExtentId> protected_ids(protect.begin(), protect.end());
  std::vector<catalog::ExtentId> kept;
  for (const bool running : {false, true}) {
    for (const auto& m : models_) {
      if ((m.get() == last) != running) {
        continue;
      }
      const std::vector<catalog::ExtentId> own = m->kept_state();
      kept.insert(kept.end(), own.begin(), own.end());
    }
  }
  if (kept.empty()) {
    return 0;
  }
  std::vector<catalog::ExtentId> resident;
  std::uint64_t bytes = 0;
  auto listed = node_.Call(
      [&]() -> Status {
        const catalog::Catalog& catalog = node_.catalog();
        for (const catalog::ExtentId extent : kept) {
          if (protected_ids.contains(extent)) continue;
          if (bytes >= needed) {
            break;
          }
          if (const auto view = catalog.Describe(extent);
              view && view->state == catalog::ExtentState::kResident && view->discarded &&
              view->leases == 0 && view->registrations == 0) {
            resident.push_back(extent);
            bytes += view->descriptor.size.value();
          }
        }
        return {};
      },
      "listing kept state backing");
  if (!listed || resident.empty() || !node_.Evict(resident)) {
    return 0;
  }
  return bytes;
}

Status Server::EvictPaged(Served& m, bool weights) {
  // Only what a swap moves: its weights and its conversation state (the
  // shared workspace and its own pinned runtime memory stay where they
  // are), with the zeroed backing its clears kept. State is written back
  // to its places.
  std::vector<catalog::ExtentId> own = weights ? m.weights() : std::vector<catalog::ExtentId>{};
  const std::vector<catalog::ExtentId> state = m.state();
  own.insert(own.end(), state.begin(), state.end());
  const std::vector<catalog::ExtentId> kept = m.kept_state();
  own.insert(own.end(), kept.begin(), kept.end());
  std::vector<catalog::ExtentId> resident;
  auto listed = node_.Call(
      [&]() -> Status {
        const catalog::Catalog& catalog = node_.catalog();
        for (const catalog::ExtentId extent : own) {
          if (const auto view = catalog.Describe(extent);
              view && view->state == catalog::ExtentState::kResident) {
            resident.push_back(extent);
          }
        }
        return {};
      },
      "listing a model's resident extents");
  if (!listed) {
    return listed;
  }
  return resident.empty() ? Status{} : node_.Evict(resident);
}

Status Server::FenceModel(Served& m) {
  const std::uint32_t stream = m.paged().stream();
  if (node_.InRequest(stream)) {
    // Its task may have ended with the cancellation: ending it only frees
    // what it held.
    std::ignore = node_.EndRequest(stream);
  }
  return node_.Job(
      catalog::Closure{}, [](providers::NativeStream) { return scheduler::JobResult::kQueued; },
      "fencing a model's stream after a hang", stream);
}

Status Server::RecoverModel(Served& m) {
  if (!NodeHealthy()) {
    return Error("the node's scheduler faulted");
  }
  if (auto fenced = FenceModel(m); !fenced) {
    return Error(std::format("its stream could not be fenced: {}", fenced.error()));
  }
  if (auto recovered = m.RecoverInPlace(); !recovered) {
    return recovered;
  }
  if (m.llm()) {
    // The hung unit may have been timed: no calibration counts it (D-103).
    static_cast<Llm&>(m).calibration_samples().Forget();
  }
  // Evicted, so its next activation loads it whole: its weights, and the
  // state it kept (idle conversations) written back.
  if (auto evicted = EvictPaged(m); !evicted) {
    return Error(std::format("evicting it: {}", evicted.error()));
  }
  if (resident_ == &m) {
    resident_ = nullptr;
  }
  if (m.llm() && m.HasRetainedState() && std::ranges::find(spilled_, &m) == spilled_.end()) {
    // As a swap's write-back: the conversations it kept are wholly on disk,
    // and their records follow (D-105).
    m.StateWrittenBack(true);
    spilled_.push_back(&m);
    auto& l = static_cast<Llm&>(m);
    for (std::size_t i = 0; i < l.branches(); ++i) {
      if (auto b = l.branch(i); b && !(*b)->spilled()) {
        l.KeepBranch(**b);
      }
    }
  }
  return {};
}

Status Server::UndoSwap(Served& out, Served& in, std::span<const catalog::ExtentId> preexisting) {
  const std::set<catalog::ExtentId> kept(preexisting.begin(), preexisting.end());
  std::vector<catalog::ExtentId> newly_resident;
  if (auto listed = node_.Call(
          [&]() -> Status {
            for (const auto& [extent, generation] : in.everything().extents) {
              (void)generation;
              const auto view = node_.catalog().Describe(extent);
              if (!kept.contains(extent) && view && view->state == catalog::ExtentState::kResident)
                newly_resident.push_back(extent);
            }
            return {};
          },
          "listing only newly loaded extents of a failed swap");
      !listed)
    return listed;
  if (!newly_resident.empty())
    if (auto evicted = node_.Evict(newly_resident); !evicted) return evicted;
  for (const auto& [extent, generation] : out.everything().extents) {
    (void)generation;
    activation_protect_.push_back(extent);
  }
  if (auto room = MakeRoomForSwap(out, {}); !room) return room;
  std::vector<catalog::ExtentId> again;
  auto listed = node_.Call(
      [&]() -> Status {
        const catalog::Catalog& catalog = node_.catalog();
        for (const auto& [extent, generation] : out.everything().extents) {
          (void)generation;
          if (const auto view = catalog.Describe(extent);
              view && view->state != catalog::ExtentState::kResident) {
            again.push_back(extent);
          }
        }
        return {};
      },
      "listing a failed swap's extents");
  if (!listed) {
    return listed;
  }
  if (!again.empty()) {
    std::vector<engine::LoadStats> log;
    if (auto r = node_.Load(again, std::format("{} back after a failed swap", out.name()), log);
        !r) {
      return r;
    }
  }
  return out.AfterLoad();  // its checks after a load, as after a swap
}

void Server::RecordCalibrations() {
  for (const auto& m : models_) {
    if (!m->llm()) {
      continue;
    }
    auto& l = static_cast<Llm&>(*m);
    Llm::CalibrationRecord& record = l.calibration_record();
    auto measured = l.calibration_samples().Take(record.known);
    if (!measured) {
      continue;
    }
    // Recorded for the next registration; this service keeps its settings,
    // so a running schedule never changes with wall time (calibration.h).
    if (auto written = WriteCalibration(roles_.state, record.key, *measured); !written) {
      Log(std::format("model {}: calibration not recorded: {}", m->name(), written.error()));
      continue;
    }
    record.known = *measured;
    Log(std::format("model {}: calibration recorded, in force from the next start: {}", m->name(),
                    FormatCalibration(*measured, record.key)));
  }
}

void Server::Maintain() {
  if (!started_ || torn_down_ || reclaiming_) {
    return;
  }
  token_memory_.Settle(0, std::chrono::milliseconds{0});
  const auto now = Clock::now();
  if (now < next_maintenance_) {
    return;
  }
  next_maintenance_ = now + kMaintenanceInterval;
  // Idle conversations past their retention, resident or spilled.
  std::size_t expired = 0;
  std::uint64_t expired_bytes = 0;
  for (const auto& m : models_) {
    if (!m->llm()) {
      continue;
    }
    auto& l = static_cast<Llm&>(*m);
    for (std::size_t i = 0; i < l.branches(); ++i) {
      auto b = l.branch(i);
      if (b) {
        (void)l.ExpireTurnCheckpoints(**b, now);
      }
      if (!b || !l.BranchIdle(**b) || (*b)->HeldByContinuation() || (*b)->history().empty() ||
          now - l.LastUsed(**b) < retention_) {
        continue;
      }
      const std::uint64_t bytes = l.StateBytes(**b);
      if (auto released = (*b)->ReleaseIdleState(); released) {
        ++expired;
        expired_bytes += bytes;
      }
    }
  }
  if (expired != 0) {
    Log(std::format("deleted {} idle conversations ({:.1f} MiB) past their retention of {} h",
                    expired, static_cast<double>(expired_bytes) / (1U << 20U),
                    config_.memory.retention_hours));
  }
  // As far as it can: what a request holds stays, the rest still goes.
  (void)DeleteSpilledPast(0);
  RecordCalibrations();
  // Pressure from outside the runtime: other processes share the node's
  // memory (D-004). One reclaim of what would restore the target headroom,
  // spaced by a back-off while the pressure persists (PressureTrim), never
  // the running model's in-use floor (Reclaim).
  const platform::MemoryPressure pressure = platform::ReadMemoryPressure();
  const PressureTrim::Trim trim = pressure_.Observe(pressure, now);
  if (!trim.now) {
    return;
  }
  const std::uint64_t freed = Reclaim(trim.needed, true, trim.why, nullptr, std::nullopt, true);
  if (pressure_.Trimmed(freed, trim.needed, now)) {
    const auto mib = [](std::uint64_t b) { return static_cast<double>(b) / (1U << 20U); };
    Log(std::format(
        "memory pressure from outside ({:.0f} MiB available, a full stall {:.1f}% of the last 10 "
        "s): gave back {:.1f} MiB of {:.1f} MiB, too little left to give without the running "
        "model's floor; looking again in {} s ({} trims, {} short)",
        mib(pressure.available.value_or(0)), pressure.stall ? pressure.stall->full_avg10 : 0.0,
        mib(freed), mib(trim.needed),
        std::chrono::duration_cast<std::chrono::seconds>(pressure_.backoff()).count(),
        pressure_.trims(), pressure_.short_trims()));
  }
}

Status Server::Activate(Served& m, SwapParts& parts, std::optional<bool> spill_state) {
  // Records' hashing waits while a swap moves the models (StateKeeper::
  // Quiet): their reads and the swap's share the drive.
  const StateKeeper::Quiet quiet(keeper_.get());
  parts = SwapParts{};
  parts.to = m.name();
  if (resident_ == &m) {
    return {};
  }
  parts.requested = Clock::now();
  if (options_.diagnostic_budget_cap_bytes) {
    parts.diagnostic.enabled = true;
    parts.diagnostic.requested = swap_host_phases_;
    parts.diagnostic.creates_requested = node_.backing_create_stats();
    parts.diagnostic.requested_ns = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(parts.requested.time_since_epoch())
            .count());
  }
  auto before = node_.Stats();
  if (!before) {
    return std::unexpected(before.error());
  }
  swap_before_ = *before;
  parts.occupancy_requested = before->catalog_occupancy_bytes;
  Served* const outgoing_model = resident_;
  const auto protected_before = activation_protect_.size();
  struct ProtectedActivation {
    std::vector<catalog::ExtentId>& ids;
    std::size_t before;
    ~ProtectedActivation() { ids.resize(before); }
  } protection{activation_protect_, protected_before};
  for (const auto& [extent, generation] : m.everything().extents) {
    (void)generation;
    activation_protect_.push_back(extent);
  }
  std::vector<catalog::ExtentId> preexisting;
  catalog::Closure saved_state;
  bool state_was_spilled = true;
  if (auto snapshot = node_.Call(
          [&]() -> Status {
            for (const auto& [extent, generation] : m.everything().extents) {
              const auto view = node_.catalog().Describe(extent);
              if (!view || view->content_generation != generation || view->discarded ||
                  view->state == catalog::ExtentState::kQuarantined)
                return Error("the incoming activation closure is stale or invalid");
              if (view->state == catalog::ExtentState::kResident) preexisting.push_back(extent);
            }
            for (const auto extent : m.state()) {
              const auto view = node_.catalog().Describe(extent);
              if (!view) return Error("the incoming state extent is absent");
              saved_state.extents.push_back({extent, view->content_generation});
              state_was_spilled =
                  state_was_spilled && view->state == catalog::ExtentState::kNonresident;
            }
            const std::set<catalog::ExtentId> cached(preexisting.begin(), preexisting.end());
            for (const auto extent : m.weights()) {
              if (cached.contains(extent))
                parts.retained_incoming_weight_bytes +=
                    node_.catalog().Describe(extent)->descriptor.size.value();
            }
            return {};
          },
          "snapshotting incoming cache and saved state before activation");
      !snapshot)
    return snapshot;
  PendingWeightEvictions pending;
  const auto completed_before = node_.counting().read_completed_bytes.load();
  const auto submitted_before = node_.counting().read_submitted_bytes.load();
  const bool restoring = std::ranges::find(spilled_, &m) != spilled_.end();
  // The incoming model's conversations a swap wrote back come back live:
  // their records go before anything of it loads (D-105), not before a
  // refusal that leaves them on disk. Those the reclaim order spilled stay
  // on disk, and kept, until a turn restores them.
  const auto unkeep_incoming = [&m]() {
    if (m.llm()) {
      auto& in = static_cast<Llm&>(m);
      for (std::size_t i = 0; i < in.branches(); ++i) {
        if (auto b = in.branch(i); b && !(*b)->spilled()) {
          in.Unkeep(**b);
        }
      }
    }
  };
  const auto rekeep_incoming = [&]() {
    if (!restoring || !state_was_spilled || saved_state.extents.empty() || !m.llm()) return;
    bool intact = true;
    const auto checked = node_.Call(
        [&]() -> Status {
          for (const auto& [extent, generation] : saved_state.extents) {
            const auto view = node_.catalog().Describe(extent);
            intact = intact && view && view->content_generation == generation &&
                     view->state == catalog::ExtentState::kNonresident && !view->discarded;
          }
          return {};
        },
        "proving a failed incoming load still has whole saved generations");
    if (!checked || !intact) return;
    m.StateWrittenBack(true);
    auto& in = static_cast<Llm&>(m);
    for (std::size_t i = 0; i < in.branches(); ++i)
      if (auto branch = in.branch(i); branch && !(*branch)->spilled()) in.KeepBranch(**branch);
  };
  const auto discard_new_incoming = [&]() -> Status {
    std::vector<catalog::ExtentId> newly_loaded;
    const std::set<catalog::ExtentId> cached(preexisting.begin(), preexisting.end());
    if (auto listed = node_.Call(
            [&]() -> Status {
              for (const auto& [extent, generation] : m.everything().extents) {
                (void)generation;
                const auto view = node_.catalog().Describe(extent);
                if (!cached.contains(extent) && view &&
                    view->state == catalog::ExtentState::kResident)
                  newly_loaded.push_back(extent);
              }
              return {};
            },
            "listing only additions to the incoming activation cache");
        !listed)
      return listed;
    return newly_loaded.empty() ? Status{} : node_.Evict(newly_loaded);
  };
  const auto account_pending = [&]() {
    std::vector<double> priorities;
    std::ignore = node_.Call(
        [&]() -> Status {
          for (const auto& selected : pending.selected) {
            const auto view = node_.catalog().Describe(selected.extent);
            if (view && view->content_generation == selected.generation &&
                (view->state == catalog::ExtentState::kNonresident ||
                 node_.scheduler().EvictionUnmapped(selected.extent))) {
              parts.evicted_weight_bytes += selected.candidate.bytes;
              priorities.push_back(selected.priority);
            }
          }
          return {};
        },
        "accounting only completed or parked selected weight evictions");
    for (const auto priority : priorities) memory::RaiseReclaimInflation(priority);
  };
  bool outgoing_published = false;
  bool outgoing_transfer_complete = false;
  catalog::Closure saved_outgoing;
  const auto publish_outgoing = [&]() {
    if (outgoing_published || outgoing_model == nullptr || !parts.with_state) return;
    outgoing_published = true;
    outgoing_model->StateWrittenBack(true);
    if (std::ranges::find(spilled_, outgoing_model) == spilled_.end())
      spilled_.push_back(outgoing_model);
    auto& out = static_cast<Llm&>(*outgoing_model);
    for (std::size_t i = 0; i < out.branches(); ++i)
      if (auto branch = out.branch(i); branch && !(*branch)->spilled()) out.KeepBranch(**branch);
  };
  const auto keep_outgoing_after_failed_undo = [&]() {
    if (!outgoing_transfer_complete || outgoing_model == nullptr || !parts.with_state ||
        saved_outgoing.extents.empty())
      return;
    std::vector<catalog::ExtentId> restored;
    auto proved = node_.Call(
        [&]() -> Status {
          for (const auto& [extent, generation] : saved_outgoing.extents) {
            const auto view = node_.catalog().Describe(extent);
            const auto* source = node_.scheduler().SourceOf(extent);
            if (!view || view->content_generation != generation ||
                view->saved_generation != generation || view->discarded || source == nullptr ||
                !source->write_back ||
                (view->state != catalog::ExtentState::kResident &&
                 view->state != catalog::ExtentState::kNonresident))
              return Error("outgoing saved generations cannot be proven after failed undo");
            if (view->state == catalog::ExtentState::kResident)
              restored.push_back(extent);
            else if (!view->preserved)
              return Error("outgoing saved state is not preserved");
          }
          return {};
        },
        "proving the completed outgoing spill after failed undo");
    // The successful original transfer saved every extent. The undo only
    // restored it; no outgoing request ran. Release that restored subset
    // unchanged before advertising the model as wholly spilled again.
    if (proved && !restored.empty()) proved = node_.Evict(restored, {.unchanged = true});
    if (proved)
      proved = node_.Call(
          [&]() -> Status {
            for (const auto& [extent, generation] : saved_outgoing.extents) {
              const auto view = node_.catalog().Describe(extent);
              if (!view || view->content_generation != generation ||
                  view->saved_generation != generation || !view->preserved ||
                  view->state != catalog::ExtentState::kNonresident)
                return Error("outgoing state did not retire wholly to its saved generations");
            }
            return {};
          },
          "checking whole outgoing spill retirement after failed undo");
    if (proved)
      publish_outgoing();
    else
      Log(std::format("failed undo's outgoing state was not kept: {}", proved.error()));
  };
  const auto release_subset = [&](std::span<const catalog::ExtentId> selected,
                                  std::vector<catalog::ExtentId>& release) -> Status {
    if (!options_.partial_weight_eviction || !handoff_) return {};
    const SwapPhaseTimer timer(
        options_.diagnostic_budget_cap_bytes ? &swap_host_phases_.partition_seconds : nullptr);
    std::vector<SwapBackingExtent> missing;
    std::vector<SwapBackingExtent> donors;
    std::uint64_t occupancy = 0;
    const auto listed = node_.Call(
        [&]() -> Status {
          const auto describe =
              [&](catalog::ExtentId extent) -> std::expected<SwapBackingExtent, std::string> {
            const auto view = node_.catalog().Describe(extent);
            const auto* source = node_.scheduler().SourceOf(extent);
            if (!view || source == nullptr)
              return Error("a swap extent has no descriptor or source");
            SwapBackingExtent result{
                .extent = extent, .bytes = view->descriptor.size.value(), .backing = std::nullopt};
            if (source->backing && source->backing->size.value() == result.bytes)
              result.backing = SwapBackingKey{.domain = view->descriptor.domain,
                                              .allocation_class = source->backing->allocation_class,
                                              .size = source->backing->size.value()};
            return result;
          };
          for (const auto& [extent, generation] : m.everything().extents) {
            (void)generation;
            const auto view = node_.catalog().Describe(extent);
            if (view && view->state != catalog::ExtentState::kResident) {
              auto entry = describe(extent);
              if (!entry) return std::unexpected(entry.error());
              missing.push_back(*entry);
            }
          }
          for (const auto extent : selected) {
            const auto view = node_.catalog().Describe(extent);
            if (view && view->state == catalog::ExtentState::kResident) {
              auto entry = describe(extent);
              if (!entry) return std::unexpected(entry.error());
              donors.push_back(*entry);
            }
          }
          occupancy = node_.catalog().OccupancyOf(node_.domain()).Total().value();
          return {};
        },
        "partitioning selected donor backing by actual class and size");
    if (!listed) return listed;
    const auto partition = ReleaseForHandoff(node_.domain(), occupancy, budget_, missing, donors);
    if (!partition) return std::unexpected(partition.error());
    release = *partition;
    return {};
  };
  const auto selected_weights = [&]() {
    catalog::Closure closure;
    for (const auto& selected : pending.selected)
      closure.extents.push_back({selected.extent, selected.generation});
    return closure;
  };
  const auto record_eviction_split = [&](const scheduler::SwapReport& report) {
    if (!parts.diagnostic.enabled || report.started == Clock::time_point{}) return;
    parts.diagnostic.eviction_split_available = true;
    const auto ns = [](Clock::time_point at) {
      return static_cast<std::uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(at.time_since_epoch()).count());
    };
    parts.diagnostic.scheduler_started_ns = ns(report.started);
    parts.diagnostic.scheduler_evicted_ns = ns(report.evicted);
    parts.diagnostic.scheduler_loaded_ns = ns(report.loaded);
    parts.diagnostic.eviction_prepare_seconds = Seconds(report.started - parts.requested);
    parts.diagnostic.eviction_retire_seconds = Seconds(report.evicted - report.started);
  };
  Clock::time_point evicted = parts.requested;
  Clock::time_point loaded;
  if (resident_ == nullptr) {
    // Nothing resident: the first load of its weights (its state is
    // resident since setup, or written back by a swap that failed and
    // could not be undone: restored with them).
    // Whatever another model still holds from a swap that failed and could
    // not be undone goes out first (state written back).
    for (const auto& other : models_) {
      if (other.get() != &m) {
        if (auto r = EvictPaged(*other, !options_.partial_weight_eviction); !r) {
          return r;
        }
        if (other->llm() && other->HasRetainedState() &&
            std::ranges::find(spilled_, other.get()) == spilled_.end()) {
          spilled_.push_back(other.get());
        }
      }
    }
    // Every model's plans and graphs stay charged inside the budget: room
    // for the whole load is made through the reclaim order first, as a
    // swap's is, or a load after a failed swap (or a recovery) could be
    // refused for room again and again with nothing to free it.
    if (auto room = MakeRoomForSwap(m, {}, options_.partial_weight_eviction ? &pending : nullptr);
        !room) {
      return room;
    }
    std::vector<engine::LoadStats> log;
    std::vector<catalog::ExtentId> all;
    for (const auto& [extent, generation] : m.everything().extents) {
      (void)generation;
      all.push_back(extent);
    }
    Status loaded_result;
    if (options_.partial_weight_eviction) {
      std::vector<catalog::ExtentId> victims;
      for (const auto& selected : pending.selected) victims.push_back(selected.extent);
      std::vector<catalog::ExtentId> release;
      if (auto partition = release_subset(victims, release); !partition) return partition;
      scheduler::SwapReport report;
      unkeep_incoming();
      loaded_result = node_.Swap(std::move(victims), m.everything(), handoff_, report,
                                 std::move(release), selected_weights());
      account_pending();
      if (loaded_result) record_eviction_split(report);
      evicted = report.evicted == Clock::time_point{} ? evicted : report.evicted;
      parts.evicted = report.evictions;
      parts.loaded = report.loads;
    } else {
      unkeep_incoming();
      loaded_result = node_.Load(all, std::format("{}'s first load", m.name()), log);
      parts.loaded = log.empty() ? 0 : log.front().extents;
    }
    if (!loaded_result) {
      if (auto cleaned = discard_new_incoming(); !cleaned)
        Log(std::format("failed first-load additions could not be cleaned: {}", cleaned.error()));
      rekeep_incoming();
      return loaded_result;
    }
    loaded = Clock::now();
  } else {
    Served& out = *resident_;
    parts.from = out.name();
    // The caller yields only at a completed unit. Release its request even
    // when the incoming closure is cached and no weight victim is needed.
    if (node_.InRequest(out.paged().stream())) {
      if (auto ended = node_.EndRequest(out.paged().stream()); !ended) return ended;
    }
    if (auto used = RecordWeightUse(out); !used) return used;
    const bool conversation = out.llm() && out.HasRetainedState();
    parts.with_state = out.llm() && spill_state.value_or(conversation);
    std::vector<catalog::ExtentId> extents;
    std::vector<catalog::ExtentId> unchanged;
    const SpillView view{.out = &out, .in = &m};
    if (parts.with_state) {
      auto& l = static_cast<Llm&>(out);
      // The spill budget holds a swap's write-back too, counted as the swap
      // leaves it: all the outgoing model's state on disk, the incoming
      // model's resident (its conversations, which then stop counting, are
      // never deleted for it). Past it, the least recently used spilled
      // conversations are deleted first, the outgoing model's among them,
      // but only once the swap's room is made below: one a request holds
      // is not, and a swap that would not fit even with every other one
      // deleted is refused here, before anything moves or is deleted.
      if (!SpillBudgetFits(0, view)) {
        return Error("the swap's retained request state exceeds the spill budget");
      }
      // Kept across a restart (D-105): each conversation's last verify's
      // owed restore runs first (no request holds the stream during a
      // swap), so the write-back holds it whole. Before the room is sized:
      // whatever it charges is counted there.
      if (l.keeps()) {
        for (std::size_t i = 0; i < l.branches(); ++i) {
          if (auto b = l.branch(i); b && l.ResidentStateBytes(**b) != 0) {
            (void)l.SettleIdle(**b);
          }
        }
      }
    }
    if (options_.partial_weight_eviction && parts.with_state) {
      if (auto saved = node_.Call(
              [&]() -> Status {
                for (const auto extent : out.state()) {
                  const auto view = node_.catalog().Describe(extent);
                  if (!view) return Error("the settled outgoing state extent is absent");
                  saved_outgoing.extents.push_back({extent, view->content_generation});
                }
                return {};
              },
              "snapshotting settled outgoing state generations");
          !saved)
        return saved;
    }
    // Its plans and graphs stay (D-090 as amended 2026-10-02): charged
    // inside the budget, they go only when the reclaim order needs their
    // room, here first if the incoming model does not fit beside them. The
    // room is checked again after each reclaim, and a swap it cannot make
    // room for is refused before anything moves. What goes out is its
    // weights and, with its state, all of that (how much a write-back
    // writes, and whether the budget deletes some of it, does not change
    // what the swap frees; the reclaim order spills no idle state here).
    const std::vector<catalog::ExtentId> weights = out.weights();
    const std::vector<catalog::ExtentId> kept = out.kept_state();
    const std::uint64_t graphs_before = out.graphs().kept + m.graphs().kept;
    {
      std::vector<catalog::ExtentId> going =
          options_.partial_weight_eviction ? std::vector<catalog::ExtentId>{} : weights;
      going.insert(going.end(), kept.begin(), kept.end());
      if (parts.with_state) {
        const std::vector<catalog::ExtentId> state = out.state();
        going.insert(going.end(), state.begin(), state.end());
        for (const catalog::ExtentId extent : out.unchanged_state()) {
          if (std::ranges::find(state, extent) == state.end()) {
            going.push_back(extent);
          }
        }
      }
      if (auto room =
              MakeRoomForSwap(m, going, options_.partial_weight_eviction ? &pending : nullptr);
          !room) {
        return room;
      }
    }
    if (parts.with_state) {
      auto& l = static_cast<Llm&>(out);
      // The room is made: the budget's deletions now, which the check above
      // found enough (nothing since spilled or restored; a deletion that
      // fails still refuses the swap before anything moves).
      if (!KeepWithinSpillBudget(0, view)) {
        return Error("the swap's retained request state exceeds the spill budget");
      }
      // Its state first, so its write-backs start first; what nothing wrote
      // since its slot's spill file last held it is released without
      // writing (an incremental spill), before the swap.
      unchanged = out.unchanged_state();
      for (const catalog::ExtentId extent : out.state()) {
        if (std::ranges::find(unchanged, extent) == unchanged.end()) {
          extents.push_back(extent);
        }
      }
      // What it writes back: its resident state (a conversation the reclaim
      // order spilled already is on disk, and passed over).
      for (std::size_t i = 0; i < l.branches(); ++i) {
        if (auto b = l.branch(i); b) {
          parts.spilled_bytes += l.ResidentStateBytes(**b);
        }
      }
    }
    extents.insert(extents.end(), kept.begin(), kept.end());
    if (options_.partial_weight_eviction) {
      for (const auto& selected : pending.selected) extents.push_back(selected.extent);
    } else {
      extents.insert(extents.end(), weights.begin(), weights.end());
    }
    const std::uint64_t graphs_after = out.graphs().kept + m.graphs().kept;
    parts.dropped_graphs = graphs_before > graphs_after ? graphs_before - graphs_after : 0;
    unkeep_incoming();
    if (!unchanged.empty()) {
      if (auto r = node_.Evict(unchanged, {.unchanged = true}); !r) {
        out.StateWrittenBack(false);
        if (auto undone = UndoSwap(out, m, preexisting); !undone) {
          resident_ = nullptr;  // the next activation loads one whole
        }
        rekeep_incoming();
        return r;
      }
    }
    scheduler::SwapReport report;
    std::vector<catalog::ExtentId> release;
    if (auto partition = release_subset(extents, release); !partition) {
      if (auto undone = UndoSwap(out, m, preexisting); !undone) resident_ = nullptr;
      rekeep_incoming();
      return partition;
    }
    const auto swapped = node_.Swap(std::move(extents), m.everything(), handoff_, report,
                                    std::move(release), selected_weights());
    account_pending();
    if (!swapped) {
      const auto& r = swapped;
      if (parts.with_state) {
        out.StateWrittenBack(false);
      }
      // Never the process: whatever of the incoming model came in goes out
      // again and the outgoing one comes back whole, so both stay usable and
      // only the request that needed the swap fails.
      if (auto undone = UndoSwap(out, m, preexisting); undone) {
        Log(std::format("swap {} -> {} failed and was undone: {}", out.name(), m.name(),
                        r.error()));
      } else {
        Log(
            std::format("swap {} -> {} failed ({}) and could not be undone ({}): no model is "
                        "resident until the next one loads whole",
                        out.name(), m.name(), r.error(), undone.error()));
        if (parts.with_state && std::ranges::find(spilled_, &out) == spilled_.end()) {
          spilled_.push_back(&out);
        }
        resident_ = nullptr;
      }
      rekeep_incoming();
      return r;
    }
    outgoing_transfer_complete = true;
    record_eviction_split(report);
    if (!options_.partial_weight_eviction) publish_outgoing();
    evicted = report.evicted;
    loaded = report.loaded;
    parts.evicted = report.evictions;
    parts.loaded = report.loads;
  }
  if (parts.diagnostic.enabled) parts.diagnostic.creates_loaded = node_.backing_create_stats();
  parts.evict = Seconds(evicted - parts.requested);
  Clock::time_point restored = evicted;
  if (restoring) {
    restored = std::max(restored, times_.Latest(m.state()));
    parts.restore = Seconds(restored - evicted);
    if (!options_.partial_weight_eviction) std::erase(spilled_, &m);
  }
  parts.page_in = Seconds(loaded - restored);
  parts.read_bytes = node_.counting().read_completed_bytes.load() - completed_before;
  parts.read_submitted_bytes = node_.counting().read_submitted_bytes.load() - submitted_before;
  if (outgoing_model != nullptr) {
    std::ignore = node_.Call(
        [&]() -> Status {
          for (const auto extent : outgoing_model->weights()) {
            const auto view = node_.catalog().Describe(extent);
            if (view && view->state == catalog::ExtentState::kResident)
              parts.retained_outgoing_weight_bytes += view->descriptor.size.value();
            else if (!options_.partial_weight_eviction && view &&
                     (view->state == catalog::ExtentState::kNonresident ||
                      node_.scheduler().EvictionUnmapped(extent)))
              parts.evicted_weight_bytes += view->descriptor.size.value();
          }
          return {};
        },
        "measuring actual retained outgoing weights");
  }
  resident_ = &m;
  // A model whose checks after its load fail is not left resident: its
  // kernels index what those checks cover unchecked (DeepSeek's hash
  // routing, Qwen3.8's n-gram hash) or replay graphs at its places (D-090).
  // Evicted (its state written back), its next activation loads it whole
  // and checks it again; only the requests that needed it fail (D-102).
  const auto unload = [&](std::string error) -> Status {
    if (options_.partial_weight_eviction && outgoing_model != nullptr) {
      if (auto undone = UndoSwap(*outgoing_model, m, preexisting); undone)
        resident_ = outgoing_model;
      else {
        error += std::format("; undoing it: {}", undone.error());
        resident_ = nullptr;
        keep_outgoing_after_failed_undo();
      }
      rekeep_incoming();
      return Error(std::move(error));
    }
    if (options_.partial_weight_eviction) {
      if (auto cleaned = discard_new_incoming(); !cleaned)
        error += std::format("; cleaning its new extents: {}", cleaned.error());
      resident_ = nullptr;
      rekeep_incoming();
      return Error(std::move(error));
    }
    if (auto evicted = EvictPaged(m); !evicted) {
      error += std::format("; evicting it: {}", evicted.error());
      m.StateWrittenBack(false);
    } else if (m.llm() && m.HasRetainedState() &&
               std::ranges::find(spilled_, &m) == spilled_.end()) {
      // As a swap's write-back: its conversations are wholly on disk, and
      // their records follow (D-105).
      m.StateWrittenBack(true);
      spilled_.push_back(&m);
      auto& l = static_cast<Llm&>(m);
      for (std::size_t i = 0; i < l.branches(); ++i) {
        if (auto b = l.branch(i); b && !(*b)->spilled()) {
          l.KeepBranch(**b);
        }
      }
    }
    resident_ = nullptr;
    return Error(std::move(error));
  };
  if (auto r = m.AfterLoad(); !r) {
    return unload(std::format("{} after its load: {}", m.name(), r.error()));
  }
  // Its places still pinned where it registered them (D-090).
  std::size_t unpinned = 0;
  const std::vector<catalog::ExtentId> managed = m.paged().managed_extents();
  if (auto r = node_.Call(
          [&]() -> Status {
            for (const catalog::ExtentId extent : managed) {
              unpinned += node_.scheduler().PlacePinned(extent) ? 0 : 1;
            }
            return {};
          },
          "checking places");
      !r) {
    return unload(std::format("checking incoming places failed: {}", r.error()));
  }
  if (unpinned != 0) {
    return unload(std::format("{} of {}'s {} extents are not pinned at their places", unpinned,
                              m.name(), managed.size()));
  }
  if (auto r = m.CheckPlaces(); !r) {
    return unload(std::format("{}'s places: {}", m.name(), r.error()));
  }
  publish_outgoing();
  if (restoring) std::erase(spilled_, &m);
  page_read_bytes_ += parts.read_bytes;
  page_read_seconds_ += parts.page_in + parts.restore;
  parts.ready = Clock::now();
  parts.setup = Seconds(parts.ready - loaded);
  parts.total = Seconds(parts.ready - parts.requested);
  if (parts.diagnostic.enabled) {
    parts.diagnostic.ready = swap_host_phases_;
    parts.diagnostic.creates_ready = node_.backing_create_stats();
  }
  return {};
}

}  // namespace llmp::runtime
