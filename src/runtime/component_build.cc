// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "runtime/component_build.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
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
#include "model/qwen38.h"
#include "planner/policy.h"
#include "planner/step_shape.h"
#include "platform/direct_io.h"
#include "platform/host_probe.h"
#include "providers/device_runtime.h"
#include "runtime/component_admission.h"
#include "runtime/gemma_profile.h"
#include "runtime/memory_guard.h"
#include "runtime/model_settings.h"
#include "runtime/prefill.h"
#include "runtime/serving_support.h"
#include "spec/mapping.h"
#include "spec/mappings/gemma2.h"
#include "spec/mappings/gemma3.h"
#include "spec/mappings/gemma4.h"
#include "spec/mappings/qwen38.h"
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
  std::map<std::pair<ir::WeightRef, format::PartRole>, Place> place;
  std::map<ir::WeightRef, std::uint64_t> pitch;
  // Before the backend, whose launch borrows it (the runner's Release
  // destroys that launch, then retires the handle).
  std::unique_ptr<kernels::ggml::CublasHandle> cublas;
  std::unique_ptr<kernels::ggml::GgmlBackend> backend;
};

// Small integer recipe metadata read through the artifact's authenticated
// shard descriptor. The synchronous direct read retires before its aligned
// stack window is reused. Shape/type and exact byte count precede allocation.
template <class T>
std::expected<std::vector<T>, std::string> RecipeIntegers(const ja::Artifact& artifact,
                                                          const model::Qwen38Tensor& tensor,
                                                          std::uint32_t maximum,
                                                          std::string_view type) {
  const bool vector = type == "I64" && tensor.plain && tensor.ne.size() == 1;
  const bool table = type == "I32" && !tensor.plain && tensor.ne.size() == 2 && tensor.ne[0] == 1;
  if (tensor.type != type || (!vector && !table))
    return Error("integer recipe metadata has an unsupported type or shape");
  const auto count64 = tensor.ne[table ? 1 : 0];
  if (count64 == 0 || count64 > maximum) return Error("integer recipe metadata exceeds its bound");
  const auto count = static_cast<std::size_t>(count64);
  auto place = artifact.ResourcePlacement(tensor.index);
  if (!place || place->bytes.value() != count * sizeof(T))
    return Error("integer recipe metadata has an inconsistent byte count");
  const auto& group = artifact.groups()[place->group];
  const auto& shard = artifact.shards()[group.shard];
  // Validated artifact extents bound every addition by the shard length.
  const auto begin = shard.data_offset.value() + group.offset.value() + place->offset.value();
  const auto end = begin + place->bytes.value();
  auto file = artifact.OpenShardForDirectRead(group.shard);
  if (!file) return Error(file.error().ToString());
  std::vector<T> out(count);
  alignas(4096) std::array<std::byte, 4096> window{};
  for (auto at = begin / window.size() * window.size(); at < end; at += window.size()) {
    if (at > shard.data_offset.value() + shard.data_bytes.value() ||
        window.size() > shard.data_offset.value() + shard.data_bytes.value() - at)
      return Error("integer recipe metadata's direct read passes its shard");
    if (auto read = platform::TransferDirectFile(file->get(), at, window, false); !read)
      return Error(std::format("reading integer recipe metadata: errno {}", read.error()));
    const auto first = std::max(at, begin), last = std::min(at + window.size(), end);
    std::memcpy(reinterpret_cast<std::byte*>(out.data()) + first - begin,
                window.data() + first - at, last - first);
  }
  return out;
}

std::expected<model::Qwen38PleHash, std::string> RecipeHash(const ja::Artifact& artifact,
                                                            const model::Qwen38Binding& binding) {
  const auto& p = model::Qwen38Flash();
  if (binding.gguf()) return model::ReadQwen38GgufHash(p, artifact, binding.ple_table.ne.at(1));
  const auto& layer = binding.layers.at(p.ple_layer);
  auto multipliers = RecipeIntegers<std::int64_t>(artifact, layer.ple_multipliers, p.ngram, "I64");
  auto offsets =
      RecipeIntegers<std::int64_t>(artifact, layer.ple_head_offsets, p.ple_heads(), "I64");
  auto vocab = RecipeIntegers<std::int64_t>(artifact, layer.ple_head_vocab, p.ple_heads(), "I64");
  if (!multipliers) return Error(multipliers.error());
  if (!offsets) return Error(offsets.error());
  if (!vocab) return Error(vocab.error());
  return model::CheckQwen38PleHash(p, *multipliers, *offsets, *vocab, binding.ple_table.ne.at(1));
}

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

// Closed internal scoring experiment, not a serving route. A non-null domain
// alone is not authority to lift the ordinary single-owner diagnostic bound.
bool JoinedScoreEnvelope(const ModelSettings& settings, std::string_view drafter,
                         const ComponentBuildOptions& options) {
  if (!options.diagnostic_solo || !options.execution_domain || !drafter.empty() ||
      settings.speculation.value || settings.max_slots.value != 4 ||
      settings.context.value != 512 || settings.prefill_chunk.value != 2 ||
      settings.wave_read_align.value != 256)
    return false;
  const auto stages = options.execution_domain->stages();
  constexpr std::array uses{engine::NumericalUse::kDecode, engine::NumericalUse::kScore};
  if (stages.size() != 1 || stages[0].stage != 0 || stages[0].name != "target" ||
      !std::ranges::equal(stages[0].uses, uses))
    return false;
  constexpr std::array phases{engine::NumericalPhase{.phase = planner::Phase::kPrefill,
                                                     .max_owners = 4,
                                                     .max_owner_rows = 2,
                                                     .max_total_rows = 8},
                              engine::NumericalPhase{.phase = planner::Phase::kDecode,
                                                     .max_owners = 4,
                                                     .max_owner_rows = 1,
                                                     .max_total_rows = 4},
                              engine::NumericalPhase{.phase = planner::Phase::kScore,
                                                     .max_owners = 4,
                                                     .max_owner_rows = 2,
                                                     .max_total_rows = 8}};
  return std::ranges::equal(stages[0].phases, phases);
}

}  // namespace

bool OnComponentEngine(std::string_view architecture) {
  return architecture == "gemma2" || architecture == "gemma3" || architecture == "gemma4";
}

EngineBuild ComponentLlmBuild(engine::PagedNode& node, fs::path store,
                              const config::ModelEntry& entry, int index,
                              ComponentBuildOptions options) {
  return [&node, store = std::move(store), id = entry.artifact.value_or(""),
          drafter_id = entry.drafter.value_or(""), index, options = std::move(options),
          metadata = std::shared_ptr<engine::ExecutionMetadataAllowance>{}](
             const ModelSettings& settings) mutable -> std::expected<EngineLlmParts, std::string> {
    const bool solo = settings.architecture == "qwen4exp";
    const bool joined_score = solo && JoinedScoreEnvelope(settings, drafter_id, options);
    if (solo && (!options.diagnostic_solo || (settings.max_slots.value != 1 && !joined_score)))
      return Error("the Qwen component qualification requires exactly one configured slot");
    auto artifact = OpenInstalled(store, id);
    if (!artifact) return Error(artifact.error());
    const bool stored = artifact->format_version() == 1;
    if (stored && !drafter_id.empty())
      return Error("native v1 component admission has no qualified drafter composition");
    const spec::Mapping* mapping = spec::FindMapping(artifact->model().architecture);
    if (mapping == nullptr) return Error("the artifact's architecture has no mapping");
    if (settings.architecture != artifact->model().architecture)
      return Error("resolved architecture differs from the opened artifact");
    auto hold = std::make_shared<ComponentHold>();
    std::optional<model::Qwen38Binding> qwen;
    std::optional<spec::PolicyRule> stored_policy;
    std::expected<spec::ModelSpec, std::string> synthesized;
    if (stored) {
      auto admitted = ReadComponentAdmission(*artifact);
      if (!admitted) return Error(admitted.error());
      if (settings.context.value == 0 ||
          settings.context.value > admitted->model.stages.front().context)
        return Error("resolved context exceeds the admitted stored spec");
      auto policy = ComponentAdmissionPolicy(
          admitted->model, admitted->source, artifact->sources(),
          {.context = settings.context.value, .slots = settings.max_slots.value});
      if (!policy) return Error(policy.error());
      stored_policy = *policy;
      synthesized = std::move(admitted->model);
      hold->sources.push_back(std::move(admitted->source));
    } else {
      std::expected<std::vector<spec::SourceValue>, std::string> values;
      if (solo) {
        auto binding = model::BindQwen38(model::Qwen38Flash(), *artifact);
        if (!binding) return Error(binding.error());
        qwen.emplace(std::move(*binding));
        auto hash = RecipeHash(*artifact, *qwen);
        if (!hash) return Error(hash.error());
        values = spec::mappings::Qwen38V0Values(model::Qwen38Flash(), *qwen, *hash);
      } else
        values = V0Values(settings.architecture, *artifact);
      if (!values) return Error(values.error());
      hold->values = std::move(*values);
      synthesized = spec::SynthesizeV0(*artifact, *mapping, hold->values);
      if (!synthesized) return Error(synthesized.error());
      auto source = spec::SourceOfV0(*artifact, mapping);
      if (!source) return Error(source.error());
      hold->sources.push_back(std::move(*source));
    }
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
      std::expected<std::vector<spec::SourceValue>, std::string> dvalues;
      if (solo) {
        if (qwen->gguf()) return Error("the GGUF target has no qualified component MTP recipe");
        auto binding = model::BindQwen38Mtp(model::Qwen38Flash(), *drafter);
        if (!binding) return Error(binding.error());
        if (binding->selected_head()) {
          auto ids = RecipeIntegers<std::int32_t>(*drafter, binding->draft_ids,
                                                  model::Qwen38Flash().vocab, "I32");
          if (!ids) return Error(ids.error());
          if (auto valid = model::CheckQwen38DraftIds(*ids, model::Qwen38Flash().vocab); !valid)
            return Error(valid.error());
        }
        drafter_mapping = &spec::mappings::Qwen38MtpReplayMapping();
        dvalues = spec::mappings::Qwen38MtpV0Values(model::Qwen38Flash(), *binding);
        if (dvalues)
          for (auto& value : *dvalues)
            if (value.key == "qwen4exp-mtp.draft_vocab") {
              const auto rows = binding->selected_head() ? binding->draft_output.ne.at(1)
                                                         : model::Qwen38Flash().vocab;
              value.value = static_cast<std::int64_t>(
                  std::min<std::uint64_t>(settings.draft_vocab.value, rows));
            }
      } else
        dvalues = DrafterV0Values(*artifact, *drafter);
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
    if (joined_score &&
        (hold->spec.stages.size() != 1 || hold->spec.stages.front().name != "target"))
      return Error("joined scoring requires the actual single target stage");
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
    const auto configured_chunk =
        solo ? PrefillChunkRows(settings.context.value, std::nullopt, settings.prefill_chunk.value,
                                model::Qwen38MostRows(settings.context.value, false))
             : settings.prefill_chunk.value;
    const auto depth = solo ? (configured_chunk >= 4 ? settings.draft_rows.value
                                                     : std::min(settings.draft_rows.value, 2U))
                            : kAssistantDepth;
    for (std::uint16_t stage = 0; stage < hold->spec.stages.size(); ++stage) {
      auto layout = components::StageLayout(hold->spec, stage, settings.context.value,
                                            configured_chunk, kExtent);
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
    spec::PolicyRule composition;
    if (stored) {
      composition = *stored_policy;
    } else {
      auto rule = spec::DefaultRule(*mapping, hold->values, {.context = context, .slots = slots});
      if (!rule) return Error(rule.error());
      if (*rule != nullptr) composition = **rule;
    }
    if (composition.prompts_alone) caps->join_prefill = false;
    if (composition.owners_alone) caps->max_segments = 1;
    if (joined_score && caps->max_segments < 4)
      return Error("the actual component capability cannot join the scoring cohort");
    // A verify's rows: the anchor and every draft.
    const std::uint32_t verify_rows = speculate ? depth + 1 : 0;
    const std::uint32_t read_align = solo ? settings.wave_read_align.value : 256;
    // The widest steps (the setup measurement): every owner's rows at the
    // end of its context, reading all of it.
    const auto shape = [&](std::uint16_t stage, planner::Phase phase, std::uint32_t owners,
                           std::uint32_t rows, std::uint32_t outputs, bool greedy) {
      planner::StepShape s{.stage = stage,
                           .phase = phase,
                           .segments = {},
                           .row_invariant = false,
                           .read_align = read_align};
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
        for (const bool greedy : {false, true}) {
          add(shape(0, planner::Phase::kVerify, owners, verify_rows, verify_rows, greedy));
          add(shape(1, planner::Phase::kDraft, owners, 1, 1, solo && greedy));
        }
        if (solo) {
          for (const auto rows : {verify_rows, chunk}) {
            add(shape(1, planner::Phase::kDraft, owners, rows, 1, true));
            add(shape(1, planner::Phase::kDraft, owners, rows, 0, false));
          }
        }
      }
    }
    if (joined_score) {
      // Full-head scored shapes are additional to ordinary prefill/decode
      // measurements. Include each real fallback slot, not just owner zero.
      for (const std::uint32_t rows : {1U, 2U}) {
        auto scored = shape(0, planner::Phase::kScore, 4, rows, rows, false);
        for (auto& segment : scored.segments) {
          segment.past = 256 - rows;
          segment.read_cells = 256;
        }
        setup.push_back(scored);
        for (const auto& segment : scored.segments) {
          auto single = scored;
          single.segments = {segment};
          setup.push_back(std::move(single));
        }
      }
    }
    // Each weight's place, by what the steps bind (the head's included;
    // a state-only prompt chunk binds a subset), and a head row's width:
    // its logits' (the head's vocabulary).
    std::vector<engine::LayerSlabs> placements;
    for (std::size_t a = 0; a < hold->sources.size(); ++a) {
      auto slabs = engine::ExpertSlabsOf(a == 0 ? *artifact : *drafter, solo ? 16 : 256);
      if (!slabs) return Error(slabs.error());
      placements.push_back(std::move(*slabs));
    }
    if (solo)
      placements[0].places.at(artifact->resources()[qwen->ple_table.index].group) =
          engine::GroupPlace::kNone;
    std::optional<engine::RunnerConfig::RowTable> row_table;
    std::uint32_t logit_width = 0;
    for (const planner::StepShape& s : setup) {
      auto step = components::BuildStep(hold->spec, s.stage, s,
                                        {.tensors = hold->tables.at(s.stage),
                                         .state = &hold->layouts.at(s.stage),
                                         .stages = hold->stages,
                                         .artifacts = hold->tables},
                                        {.narrow = true});
      if (!step) return Error(step.error());
      for (const components::StepWeight& weight : step->weights) {
        const ja::Artifact& a = weight.ref.stage == 0 ? *artifact : *drafter;
        for (const auto& [name, role] : weight.parts) {
          const auto found =
              weight.ref.expert_array ? a.FindExpertArray(name) : a.FindResource(name);
          if (!found) return Error(std::format("no weight part holds {}", name));
          hold->place[{weight.ref, role}] = {
              .artifact = weight.ref.stage, .array = weight.ref.expert_array, .index = *found};
          if (weight.ref.expert_array)
            hold->pitch[weight.ref] = placements.at(weight.ref.stage).pitch.at(*found);
          if (solo && weight.ref.stage == 0 && !weight.ref.expert_array &&
              *found == qwen->ple_table.index)
            row_table = {.weight = weight.ref,
                         .resource = *found,
                         .format = step->graph.tensor(weight.tensor).format,
                         .part = role,
                         .rows_per_token = model::Qwen38Flash().ple_heads()};
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
    if (solo && !row_table) return Error("the component recipe has no n-gram row table");
    ComponentHold* held = hold.get();
    auto backend = kernels::ggml::GgmlBackend::Create(
        0, node.execution(), node.stream(static_cast<std::uint32_t>(index)), nullptr,
        {.lower = {.device_masks = true,
                   // The runner stages a frozen read's host mask (its cells'
                   // positions before the rows); the device's are made
                   // anyway.
                   .frozen_host_masks = true,
                   .expert_stride = [held](const ir::WeightRef& ref) -> std::uint64_t {
                     const auto at = held->pitch.find(ref);
                     return at == held->pitch.end() ? 0 : at->second;
                   }}});
    if (!backend) return Error(backend.error());
    hold->backend = std::move(*backend);
    const providers::DeviceFacts facts =
        providers::QueryDeviceFacts(0).value_or(providers::DeviceFacts{});
    const std::uint64_t cublas_bytes =
        kernels::ggml::CublasHandle::UpstreamWorkspace(static_cast<int>(facts.architecture))
            .value();
    if (options.tuning && !options.execution_domain)
      return Error("explicit tuning requires a declared execution source");
    if (options.execution_domain) {
      if (!metadata) {
        auto admitted = engine::ExecutionMetadataAllowance::Admit(
            node.catalog(), node.domain(), options.execution_metadata_bytes,
            platform::AvailableMemoryBytes().value_or(0), kUncountedMargin);
        if (!admitted) return Error(admitted.error());
        metadata = std::move(*admitted);
      }
      const auto tuning =
          options.tuning ? std::span<const execution::TunedShape>(options.tuning->facts().tuned)
                         : std::span<const execution::TunedShape>{};
      if (auto declared = hold->backend->Declare(
              tuning, cublas_bytes,
              [metadata](std::uint64_t bytes) { return metadata->Charge(bytes); },
              [metadata](std::uint64_t bytes) { metadata->Release(bytes); });
          !declared)
        return Error(declared.error());
    }

    // A harness's matched recipe in place of the target's mapping's.
    std::optional<planner::Policy> target_policy;
    if (stored || options.target_policy) {
      auto parsed =
          planner::ParsePolicy(options.target_policy ? *options.target_policy : composition.text);
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
           .backend = hold->backend->Backend(),
           .execution_domain = options.execution_domain,
           .validate_program = options.validate_program});
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
                          .read_align = read_align,
                          .spill_place = {},
                          .spill_directory = {},
                          .invalidate = {},
                          .drop_plans = {}};
    parts.config.weights.push_back({.weights = std::move(weights),
                                    .places = std::move(placements[0].places),
                                    .slabs = std::move(placements[0].slabs)});
    if (row_table) parts.config.row_tables.push_back(*row_table);
    if (solo)
      parts.config.capture_phases &= ~(1U << static_cast<unsigned>(planner::Phase::kPrefill));
    if (speculate) {
      parts.config.slots.stages.push_back(
          {.name = std::string(drafter_mapping->stage), .layout = hold->layouts.at(1)});
      auto dweights = std::make_unique<engine::PagedWeights>();
      if (auto opened = dweights->Open(store / drafter_id); !opened) return Error(opened.error());
      drafter_paged = dweights.get();
      parts.config.weights.push_back({.weights = std::move(dweights),
                                      .places = std::move(placements[1].places),
                                      .slabs = std::move(placements[1].slabs)});
      parts.config.max_verify_rows = verify_rows;
      parts.mode = {.kind = engine::DecodingMode::Kind::kSpeculative,
                    .drafters = {{.stage = 1,
                                  .shape = engine::DrafterSpec::Shape::kAutoregressive,
                                  .max_depth = depth,
                                  .catch_up = solo,
                                  .borrows_target = !solo,
                                  .anchored = !solo,
                                  .features = true,
                                  .adaptive = solo && depth >= 3,
                                  .depth_cost = solo ? settings.depth_cost_ratio.value : 1.16,
                                  .device_tokens = solo,
                                  .draft_before_limit = solo}},
                    .combine = engine::DecodingMode::Drafters::kCascade,
                    .row_invariant = false};
    }
    parts.config.setup_shapes = std::move(setup);
    // cuBLAS on the model's stream over the runner's workspace (D-053: the
    // engine's handle, never one GGML makes).
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
    parts.config.weight_address = [held, paged, drafter_paged](
                                      const ir::WeightRef& ref,
                                      format::PartRole role) -> std::uint64_t {
      const auto at = held->place.find({ref, role});
      if (at == held->place.end()) return 0;
      const ComponentHold::Place& p = at->second;
      const auto* owner = p.artifact == 0 ? paged : drafter_paged;
      if (owner == nullptr) return 0;
      return p.array ? owner->array_address(p.index) : owner->resource_address(p.index);
    };
    parts.source = std::move(step_source);
    parts.caps = *caps;
    parts.logit_width = logit_width;
    parts.device_greedy = true;
    parts.device_verify = solo;
    parts.branch_depth = solo ? 3U : 0U;
    parts.prefill = {.chunk_rows = chunk,
                     .lookahead = composition.prefill_lookahead,
                     .capture_ahead = composition.capture_ahead,
                     .prepare_ahead = true,
                     .state_only = true};
    if (solo) {
      parts.prefill.lookahead = options.prefill_lookahead ? 2 : 0;
      parts.prefill.prepare_ahead = options.prepare_state;
      parts.prefill.state_only = options.state_only_prefill.value_or(!speculate);
    }
    parts.architecture = settings.architecture;
    parts.chat = ChatSliceOf(settings.architecture);
    parts.keep = std::move(hold);
    return parts;
  };
}

}  // namespace llmp::runtime
