// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "runtime/component_build.h"

#include <algorithm>
#include <cstdint>
#include <expected>
#include <format>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "artifact/artifact.h"
#include "base/bytes.h"
#include "components/step.h"
#include "engine/component_source.h"
#include "engine/decoding.h"
#include "engine/expert_slab.h"
#include "engine/paged_weights.h"
#include "kernels/ggml/cublas.h"
#include "kernels/ggml/step_program.h"
#include "model/gemma2.h"
#include "model/gemma3.h"
#include "model/gemma4.h"
#include "model/gemma4_assistant.h"
#include "planner/policy.h"
#include "planner/step_shape.h"
#include "providers/device_runtime.h"
#include "runtime/gemma_profile.h"
#include "runtime/model_settings.h"
#include "runtime/serving_support.h"
#include "spec/mapping.h"
#include "spec/mappings/gemma2.h"
#include "spec/mappings/gemma3.h"
#include "spec/mappings/gemma4.h"
#include "spec/spec.h"

namespace llmp::runtime {
namespace {

using serving_support::Error;
using serving_support::kExtent;

namespace fs = std::filesystem;
namespace ja = llmp::artifact;

struct ComponentHold {
  spec::ModelSpec spec;
  std::vector<spec::Source> sources;  // by artifact: the target's, the drafter's
  std::vector<std::span<const components::BoundTensor>> tables;
  std::vector<model::StateLayoutDerived> layouts;  // by stage
  std::vector<const model::StateLayoutDerived*> stages;
  std::vector<spec::SourceValue> values, drafter_values;
  // Each weight's place: its artifact (the stage it names), whether it is
  // an expert array, and its resource's or array's index.
  struct Place {
    std::uint16_t artifact = 0;
    bool array = false;
    std::uint32_t index = 0;
  };
  std::map<ir::WeightRef, Place> place;
  std::vector<std::uint64_t> pitch;  // the target's expert arrays', by array
  // Before the backend, whose launch borrows it (the runner's Release
  // destroys that launch, then retires the handle).
  std::unique_ptr<kernels::ggml::CublasHandle> cublas;
  std::unique_ptr<kernels::ggml::GgmlBackend> backend;
};

// A v0 artifact's values for its architecture's mapping: its compiled
// profile's (spec/mappings/*.h); Gemma 4's by the approved profile its
// tensors bind.
std::expected<std::vector<spec::SourceValue>, std::string> V0Values(std::string_view architecture,
                                                                    const ja::Artifact& artifact) {
  if (architecture == "gemma2") return spec::mappings::Gemma2V0Values(model::Gemma2_2B());
  if (architecture == "gemma3") return spec::mappings::Gemma3V0Values(model::Gemma3_4BQat());
  if (architecture == "gemma4") {
    auto profile = ApprovedGemmaProfile(artifact);
    if (!profile) return Error(profile.error());
    return spec::mappings::Gemma4V0Values(**profile);
  }
  return Error(
      std::format("the component engine has no recipe for architecture {} yet", architecture));
}

// A drafter's v0 values (its compiled profile's, by its target's): the
// Gemma 4 assistant's of its target's approved profile, its tensors bound
// to that profile.
std::expected<std::vector<spec::SourceValue>, std::string> DrafterV0Values(
    const ja::Artifact& target, const ja::Artifact& drafter) {
  if (drafter.model().architecture == "gemma4-assistant" &&
      target.model().architecture == "gemma4") {
    auto profile = ApprovedGemmaProfile(target);
    if (!profile) return Error(profile.error());
    const model::Gemma4AssistantProfile& assistant =
        *profile == &model::Gemma4_31B() ? model::Gemma4Assistant31() : model::Gemma4Assistant26();
    if (auto bound = model::BindGemma4Assistant(assistant, drafter); !bound) {
      return Error(bound.error());
    }
    return spec::mappings::Gemma4AssistantV0Values(assistant, **profile);
  }
  return Error(std::format("the component engine has no drafter {} for architecture {}",
                           drafter.model().architecture, target.model().architecture));
}

// What of a family's chat template its chat route serves (the M3 classes'
// qualified slices): Gemma 4's thought channel and tool calls, which no
// parser here reads, are off, and their openers end a reply.
EngineLlmParts::ChatSlice ChatSliceOf(std::string_view architecture) {
  if (architecture == "gemma4") {
    return {.thinking = false, .tools = false, .stops = {"<|channel>", "<|tool_call>"}};
  }
  return {};
}

// A Gemma 4 assistant's depth: three drafts a unit at most, its qualified
// transaction's (docs/experiments/gemma-assistant-throughput).
constexpr std::uint32_t kAssistantDepth = 3;

}  // namespace

bool OnComponentEngine(std::string_view architecture) {
  return architecture == "gemma2" || architecture == "gemma3" || architecture == "gemma4";
}

EngineBuild ComponentLlmBuild(engine::PagedNode& node, fs::path store,
                              const config::ModelEntry& entry, int index,
                              ComponentBuildOptions options) {
  return [&node, store = std::move(store), id = entry.artifact.value_or(""),
          drafter_id = entry.drafter.value_or(""), index, options = std::move(options)](
             const ModelSettings& settings) -> std::expected<EngineLlmParts, std::string> {
    auto artifact = OpenInstalled(store, id);
    if (!artifact) return Error(artifact.error());
    auto values = V0Values(settings.architecture, *artifact);
    if (!values) return Error(values.error());
    const spec::Mapping* mapping = spec::FindMapping(artifact->model().architecture);
    if (mapping == nullptr) return Error("the artifact's architecture has no mapping");
    auto hold = std::make_shared<ComponentHold>();
    hold->values = std::move(*values);
    auto synthesized = spec::SynthesizeV0(*artifact, *mapping, hold->values);
    if (!synthesized) return Error(synthesized.error());
    auto source = spec::SourceOfV0(*artifact, mapping);
    if (!source) return Error(source.error());
    hold->sources.push_back(std::move(*source));
    // The drafter, a stage of its own, when the model speculates.
    const bool speculate = !drafter_id.empty() && settings.speculation.value;
    std::optional<ja::Artifact> drafter;
    const spec::Mapping* drafter_mapping = nullptr;
    if (speculate) {
      auto opened = OpenInstalled(store, drafter_id);
      if (!opened) return Error(opened.error());
      drafter.emplace(std::move(*opened));
      drafter_mapping = spec::FindMapping(drafter->model().architecture);
      if (drafter_mapping == nullptr || drafter_mapping->role != spec::StageRole::kDrafter) {
        return Error(std::format("the drafter's architecture {} has no drafter mapping",
                                 drafter->model().architecture));
      }
      auto dvalues = DrafterV0Values(*artifact, *drafter);
      if (!dvalues) return Error(dvalues.error());
      hold->drafter_values = std::move(*dvalues);
      auto composed = spec::ComposeDrafterV0(*synthesized, *mapping, *drafter, *drafter_mapping,
                                             hold->drafter_values);
      if (!composed) return Error(composed.error());
      hold->spec = std::move(*composed);
      auto dsource = spec::SourceOfV0(*drafter, drafter_mapping);
      if (!dsource) return Error(dsource.error());
      hold->sources.push_back(std::move(*dsource));
    } else {
      hold->spec = std::move(*synthesized);
    }
    for (const spec::Source& s : hold->sources) hold->tables.emplace_back(s.tensors);
    if (auto valid = components::ValidateSpec(hold->spec, hold->tables); !valid) {
      return Error(valid.error());
    }
    // The spec bound to the artifacts its stages run from (each stage's
    // index into them; D-105's layout identity and the plans' key): the
    // target's ID, then the drafter's.
    std::vector<std::string_view> artifacts{id};
    if (speculate) artifacts.emplace_back(drafter_id);
    auto bound = spec::BoundDigest(hold->spec, artifacts);
    if (!bound) return Error(bound.error());
    // Each state tensor starts on its own paged extent, as the M3 layouts
    // did: packed tensors would straddle extents, so their growth would
    // cross extent boundaries at staggered positions and a prompt would
    // grow its state (each growth a fixed host cost between steps) every
    // chunk or two instead of once every extent's worth of cells.
    for (std::uint16_t stage = 0; stage < hold->spec.stages.size(); ++stage) {
      auto layout = components::StageLayout(hold->spec, stage, settings.context.value,
                                            settings.prefill_chunk.value, kExtent);
      if (!layout) return Error(layout.error());
      hold->layouts.push_back(std::move(*layout));
    }
    for (const auto& l : hold->layouts) hold->stages.push_back(&l);
    const model::StateLayoutDerived& target = hold->layouts.front();
    auto caps = components::StageBatching(hold->spec, 0);
    if (!caps) return Error(caps.error());
    const std::uint32_t slots = std::clamp<std::uint32_t>(
        settings.max_slots.value, 1, static_cast<std::uint32_t>(Llm::kMaxBranches));
    const std::uint32_t context = target.context;
    const std::uint32_t chunk = std::max<std::uint32_t>(target.max_rows, 1);
    // The recipe for this serving envelope, and how it composes steps.
    auto rule = spec::DefaultRule(*mapping, hold->values, {.context = context, .slots = slots});
    if (!rule) return Error(rule.error());
    const spec::PolicyRule composition = *rule != nullptr ? **rule : spec::PolicyRule{};
    if (composition.prompts_alone) caps->join_prefill = false;
    if (composition.owners_alone) caps->max_segments = 1;
    // A verify's rows: the anchor and every draft.
    const std::uint32_t verify_rows = speculate ? kAssistantDepth + 1 : 0;
    // The widest steps (the setup measurement): every owner's rows at the
    // end of its context, reading all of it.
    const auto shape = [&](std::uint16_t stage, planner::Phase phase, std::uint32_t owners,
                           std::uint32_t rows, std::uint32_t outputs, bool greedy) {
      planner::StepShape s{.stage = stage,
                           .phase = phase,
                           .segments = {},
                           .row_invariant = false,
                           .read_align = 256};
      for (std::uint32_t o = 0; o < owners; ++o) {
        s.segments.push_back({.slot = static_cast<std::uint8_t>(o),
                              .rows = rows,
                              .past = context - rows,
                              .read_cells = context,
                              .outputs = outputs,
                              .group = 0,
                              .want_head = outputs != 0,
                              .greedy = greedy});
      }
      return s;
    };
    const std::uint32_t joined = std::min(slots, std::max(caps->max_segments, 1U));
    // A prompt chunk and a decode row, joined and alone, with logits and
    // with device tokens: their plans differ (owner attention joins
    // owners), so each is measured; with a drafter, a verify and a draft
    // pass too, joined and alone.
    std::vector<planner::StepShape> setup;
    for (const std::uint32_t owners : {1U, joined}) {
      const auto add = [&](planner::StepShape s) {
        if (std::ranges::find(setup, s) == setup.end()) setup.push_back(std::move(s));
      };
      if (owners == 1 || caps->join_prefill) {
        add(shape(0, planner::Phase::kPrefill, owners, chunk, 1, false));
      }
      for (const bool greedy : {false, true}) {
        add(shape(0, planner::Phase::kDecode, owners, 1, 1, greedy));
      }
      if (speculate) {
        add(shape(0, planner::Phase::kVerify, owners, verify_rows, verify_rows, false));
        add(shape(1, planner::Phase::kDraft, owners, 1, 1, false));
      }
    }
    // Each weight's place, by what the steps bind (the head's included;
    // a state-only prompt chunk binds a subset), and a head row's width:
    // its logits' (the head's vocabulary).
    std::uint32_t logit_width = 0;
    for (const planner::StepShape& s : setup) {
      auto step = components::BuildStep(hold->spec, s.stage, s,
                                        {.tensors = hold->tables.at(s.stage),
                                         .state = &hold->layouts.at(s.stage),
                                         .stages = hold->stages,
                                         .artifacts = hold->tables},
                                        {.narrow = true});
      if (!step) return Error(step.error());
      for (const components::StepWeight& w : step->weights) {
        if (hold->place.contains(w.ref)) continue;
        const ja::Artifact& a = w.ref.stage == 0 ? *artifact : *drafter;
        const std::string& name = w.parts.front().first;
        if (w.ref.expert_array) {
          const auto arrays = a.expert_arrays();
          const auto at = std::ranges::find(arrays, name, &ja::ExpertArray::name);
          if (at == arrays.end()) return Error(std::format("no expert array holds {}", name));
          hold->place[w.ref] = {.artifact = w.ref.stage,
                                .array = true,
                                .index = static_cast<std::uint32_t>(at - arrays.begin())};
        } else {
          const auto found = a.FindResource(name);
          if (!found) return Error(std::format("no resource holds {}", name));
          hold->place[w.ref] = {.artifact = w.ref.stage, .array = false, .index = *found};
        }
      }
      if (s.stage == 0 && step->logits.valid() && logit_width == 0) {
        const std::uint64_t width = step->graph.tensor(step->logits).shape().ne[0];
        if (width == 0 ||
            width > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
          return Error(std::format("the head's {} logits a row", width));
        }
        logit_width = static_cast<std::uint32_t>(width);
      }
    }
    if (logit_width == 0) return Error("the spec's target stage has no head");
    // A layer's routed experts in one slab (the M3 runners' pitch).
    auto slabs = engine::ExpertSlabsOf(*artifact, 256);
    if (!slabs) return Error(slabs.error());
    hold->pitch = slabs->pitch;
    ComponentHold* held = hold.get();
    auto backend = kernels::ggml::GgmlBackend::Create(
        0, node.execution(), node.stream(static_cast<std::uint32_t>(index)), nullptr,
        {.lower = {.device_masks = true,
                   // The runner stages a frozen read's host mask (its cells'
                   // positions before the rows); the device's are made
                   // anyway.
                   .frozen_host_masks = true,
                   .expert_stride = [held](const ir::WeightRef& ref) -> std::uint64_t {
                     const auto at = held->place.find(ref);
                     if (at == held->place.end() || !at->second.array || at->second.artifact != 0 ||
                         at->second.index >= held->pitch.size()) {
                       return 0;
                     }
                     return held->pitch[at->second.index];
                   }}});
    if (!backend) return Error(backend.error());
    hold->backend = std::move(*backend);
    // A harness's matched recipe in place of the target's mapping's.
    std::optional<planner::Policy> target_policy;
    if (options.target_policy) {
      auto parsed = planner::ParsePolicy(*options.target_policy);
      if (!parsed) return Error(std::format("the target's policy: {}", parsed.error()));
      target_policy = std::move(*parsed);
    }
    std::vector<std::unique_ptr<engine::ComponentStepSource>> sources;
    for (std::uint16_t stage = 0; stage < hold->spec.stages.size(); ++stage) {
      auto made = engine::ComponentStepSource::Create(
          {.spec = &hold->spec,
           .stage = stage,
           .bindings = {.tensors = hold->tables.at(stage),
                        .state = &hold->layouts.at(stage),
                        .stages = hold->stages,
                        .artifacts = hold->tables},
           .bound_digest = *bound,
           .mapping = stage == 0 ? mapping : drafter_mapping,
           .values = stage == 0 ? std::span<const spec::SourceValue>(hold->values)
                                : std::span<const spec::SourceValue>(hold->drafter_values),
           .slots = slots,
           .policy = stage == 0 ? target_policy : std::nullopt,
           .backend = hold->backend->Backend()});
      if (!made) return Error(made.error());
      sources.push_back(std::move(*made));
    }
    std::unique_ptr<engine::StepSource> step_source;
    if (sources.size() == 1) {
      step_source = std::move(sources.front());
    } else {
      auto staged = engine::StagedStepSource::Create(std::move(sources));
      if (!staged) return Error(staged.error());
      step_source = std::move(*staged);
    }
    auto weights = std::make_unique<engine::PagedWeights>();
    if (auto opened = weights->Open(store / id); !opened) return Error(opened.error());
    engine::PagedWeights* paged = weights.get();
    engine::PagedWeights* drafter_paged = nullptr;
    EngineLlmParts parts;
    parts.config.slots = {.model = settings.name,
                          .slots = slots,
                          .stages = {{.name = "target", .layout = target, .read_align = 0}},
                          .spec = *bound,
                          .read_align = 256,
                          .spill_place = {},
                          .spill_directory = {},
                          .invalidate = {},
                          .drop_plans = {}};
    parts.config.weights.push_back(
        {.weights = std::move(weights), .places = slabs->places, .slabs = slabs->slabs});
    if (speculate) {
      parts.config.slots.stages.push_back(
          {.name = std::string(drafter_mapping->stage), .layout = hold->layouts.at(1)});
      auto dweights = std::make_unique<engine::PagedWeights>();
      if (auto opened = dweights->Open(store / drafter_id); !opened) return Error(opened.error());
      drafter_paged = dweights.get();
      std::vector<engine::GroupPlace> places(drafter_paged->artifact().groups().size(),
                                             engine::GroupPlace::kDevice);
      parts.config.weights.push_back(
          {.weights = std::move(dweights), .places = std::move(places), .slabs = {}});
      parts.config.max_verify_rows = verify_rows;
      parts.mode = {.kind = engine::DecodingMode::Kind::kSpeculative,
                    .drafters = {{.stage = 1,
                                  .shape = engine::DrafterSpec::Shape::kAutoregressive,
                                  .max_depth = kAssistantDepth,
                                  .borrows_target = true,
                                  .anchored = true,
                                  .features = true}},
                    .combine = engine::DecodingMode::Drafters::kCascade,
                    .row_invariant = false};
    }
    parts.config.setup_shapes = std::move(setup);
    // cuBLAS on the model's stream over the runner's workspace (D-053: the
    // engine's handle, never one GGML makes).
    const providers::DeviceFacts facts =
        providers::QueryDeviceFacts(0).value_or(providers::DeviceFacts{});
    const std::uint64_t cublas_bytes =
        kernels::ggml::CublasHandle::UpstreamWorkspace(static_cast<int>(facts.architecture))
            .value();
    parts.config.kernel_memory.push_back(
        {.name = std::format("{} cuBLAS workspace", settings.name), .bytes = cublas_bytes});
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
    parts.config.weight_address = [held, paged, drafter_paged](const ir::WeightRef& ref,
                                                               format::PartRole) -> std::uint64_t {
      const auto at = held->place.find(ref);
      if (at == held->place.end()) return 0;
      const ComponentHold::Place& p = at->second;
      if (p.artifact != 0) {
        return drafter_paged == nullptr || p.array ? 0 : drafter_paged->resource_address(p.index);
      }
      return p.array ? paged->array_address(p.index) : paged->resource_address(p.index);
    };
    parts.source = std::move(step_source);
    parts.caps = *caps;
    parts.logit_width = logit_width;
    parts.device_greedy = true;
    parts.prefill = {.chunk_rows = chunk,
                     .lookahead = composition.prefill_lookahead,
                     .capture_ahead = composition.capture_ahead,
                     .prepare_ahead = true,
                     .state_only = true};
    parts.architecture = settings.architecture;
    parts.chat = ChatSliceOf(settings.architecture);
    parts.keep = std::move(hold);
    return parts;
  };
}

}  // namespace llmp::runtime
