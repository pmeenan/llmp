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
#include "components/state_encoding.h"
#include "components/step.h"
#include "engine/component_source.h"
#include "engine/decoding.h"
#include "engine/expert_slab.h"
#include "engine/paged_weights.h"
#include "kernels/ggml/cublas.h"
#include "kernels/ggml/step_program.h"
#include "model/dspark.h"
#include "model/dsv4.h"
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
#include "runtime/component_bindings.h"
#include "runtime/gemma_profile.h"
#include "runtime/memory_guard.h"
#include "runtime/model_settings.h"
#include "runtime/prefill.h"
#include "runtime/serving_support.h"
#include "spec/mapping.h"
#include "spec/mappings/dsv4.h"
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

// The fixed target's current hard-cap compiler and simultaneous factory
// inventory is sealed with its diagnostic. This charge is separate from the
// backend declaration allowance and from all mapped runtime/device storage.
constexpr std::uint64_t kDsv4ConstructionBytes = std::uint64_t{4} << 30U;
// Reader maps use only the remaining space inside the already paid compiler
// ceiling. Other lower metadata includes the current descriptor arena and
// owner indexes, but excludes the old 32 MiB reader-map sub-budget.
constexpr std::uint64_t kDsv4CompilerBytes = 2776629248ULL;
constexpr std::uint64_t kDsv4OtherLowerBytes = 2279950224ULL - 33554432ULL + 92274688ULL;
static_assert(kDsv4OtherLowerBytes < kDsv4CompilerBytes);
constexpr std::uint64_t kDsv4ReaderScratchBytes = kDsv4CompilerBytes - kDsv4OtherLowerBytes;
constexpr std::string_view kDsv4DiagnosticArtifact =
    "8a355bfb27c90e1150fbd7fa62ea6e63f6bf34fcca33934e52d22773f1508234";
constexpr std::string_view kDsparkDiagnosticArtifact =
    "dd2d3f9c66f070fb231d27d5a11f38ff22c78dc8f089cecedbb67721e9b4bec5";
struct ConstructionCharge {
  std::shared_ptr<engine::ExecutionMetadataAllowance> allowance;
  std::uint64_t bytes = 0;
  ~ConstructionCharge() { allowance->Release(bytes); }
};

// The domain and its payment share one owner, including failed source setup.
// Explicit caller domains retain their existing ownership contract.
struct DerivedDomain {
  std::shared_ptr<ConstructionCharge> charge;
  engine::NumericalDomain domain;
};

}  // namespace

std::expected<std::shared_ptr<const engine::NumericalDomain>, std::string> ComponentServingDomain(
    const spec::ModelSpec& model_spec, std::span<const model::StateLayoutDerived> layouts,
    std::uint32_t slots, const engine::DecodingMode& mode,
    const std::shared_ptr<engine::ExecutionMetadataAllowance>& allowance) {
  const bool composed = model_spec.stages.size() > 1;
  std::uint64_t verify = composed ? 1 : 0;
  for (const auto& draft : mode.drafters) {
    if (mode.combine == engine::DecodingMode::Drafters::kCascade)
      verify += draft.max_depth;
    else
      verify = std::max(verify, std::uint64_t{draft.max_depth} + 1);
    if (verify > UINT32_MAX) return Error("serving verify row bound overflows");
  }
  const auto verify_rows = static_cast<std::uint32_t>(verify);
  bool repeat_pending = false;
  if (model_spec.pipeline != spec::PipelineKind::kLlm || model_spec.stages.empty() ||
      model_spec.stages.size() > std::uint64_t{UINT16_MAX} + 1 ||
      model_spec.stages.size() != layouts.size() || slots == 0 || slots > Llm::kMaxBranches ||
      (composed ? mode.kind != engine::DecodingMode::Kind::kSpeculative || verify_rows == 0 ||
                      mode.drafters.size() != layouts.size() - 1
                : mode.kind != engine::DecodingMode::Kind::kPlain || !mode.drafters.empty()) ||
      !allowance)
    return Error("a serving domain needs its complete state and owner envelope");
  std::uint64_t bytes = sizeof(DerivedDomain) + sizeof(ConstructionCharge) + 512;
  for (std::size_t i = 0; i < layouts.size(); ++i) {
    const auto& stage = model_spec.stages[i];
    auto progress = model::DraftBounds(layouts[i]);
    if (!progress ||
        (i == 0 ? stage.role != spec::StageRole::kTarget || progress->has_value()
                : stage.role != spec::StageRole::kDrafter) ||
        layouts[i].max_rows == 0 || (i == 0 && verify_rows > layouts[i].max_rows))
      return Error("a serving domain needs the actual target and drafter bounds");
    if (i != 0) {
      const auto& draft = mode.drafters[i - 1];
      if (draft.stage != i || draft.max_depth == 0 || draft.shared_wave_depth > draft.max_depth ||
          (draft.shared_wave_depth != 0 &&
           draft.shape != engine::DrafterSpec::Shape::kAutoregressive) ||
          verify_rows > layouts[i].max_rows ||
          (!layouts[i].borrowed.empty() && !draft.borrows_target) ||
          (draft.features && model::FeatureTap(layouts[i]) == nullptr) ||
          (progress->has_value() && (draft.shape != engine::DrafterSpec::Shape::kAutoregressive ||
                                     !draft.catch_up || !draft.features || draft.anchored)))
        return Error("a serving domain disagrees with the actual drafter schedule");
      repeat_pending |= progress->has_value();
    }
    // Input vectors, canonical domain, the retained shared owner and the
    // maximum temporary moved representation; no graph or tensor payload.
    const auto stage_bytes = sizeof(engine::NumericalStage) + stage.name.size() + 1 +
                             5 * sizeof(engine::NumericalUse) + 4 * sizeof(engine::NumericalPhase);
    std::uint64_t copy_bytes = 0;
    if (__builtin_mul_overflow(std::uint64_t{4}, static_cast<std::uint64_t>(stage_bytes),
                               &copy_bytes) ||
        __builtin_add_overflow(bytes, copy_bytes, &bytes))
      return Error("serving domain construction bound overflows");
  }
  if (repeat_pending && slots > 1) {
    // The existing joined replay contract requires every companion to own
    // replay progress; block drafters use the ordinary speculative contract.
    for (std::size_t i = 1; i < layouts.size(); ++i) {
      auto progress = model::DraftBounds(layouts[i]);
      if (!progress || !*progress)
        return Error("joined replay requires complete companion progress contracts");
    }
  }
  if (!allowance->Charge(bytes)) return Error("serving domain metadata allowance is insufficient");
  auto charge = std::make_shared<ConstructionCharge>();
  charge->allowance = allowance;
  charge->bytes = bytes;
  std::vector<engine::NumericalStage> stages;
  stages.reserve(layouts.size());
  for (std::size_t i = 0; i < layouts.size(); ++i) {
    engine::NumericalStage stage{
        .stage = static_cast<std::uint16_t>(i),
        .name = model_spec.stages[i].name,
        .uses = {engine::NumericalUse::kPrompt, engine::NumericalUse::kDecode},
        .phases = {}};
    if (composed) {
      stage.uses.push_back(engine::NumericalUse::kPromptCompanions);
      stage.uses.push_back(engine::NumericalUse::kSpeculative);
      if (repeat_pending && slots > 1)
        stage.uses.push_back(engine::NumericalUse::kJoinedRepeatPending);
    }
    stage.phases.reserve(i == 0 ? (composed ? 4 : 2) : 1);
    const auto add = [&](planner::Phase phase, std::uint32_t rows) {
      stage.phases.push_back({.phase = phase,
                              .max_owners = slots,
                              .max_owner_rows = rows,
                              .max_total_rows = std::uint64_t{slots} * rows});
    };
    if (i == 0) {
      add(planner::Phase::kPrefill, layouts[i].max_rows);
      add(planner::Phase::kDecode, 1);
      if (composed) {
        add(planner::Phase::kVerify, verify_rows);
        add(planner::Phase::kCommit, verify_rows);
      }
    } else {
      const auto& draft = mode.drafters[i - 1];
      add(planner::Phase::kDraft, draft.shape == engine::DrafterSpec::Shape::kBlock
                                      ? draft.max_depth
                                      : layouts[i].max_rows);
    }
    stages.push_back(std::move(stage));
  }
  auto domain = engine::NumericalDomain::Create(std::move(stages));
  if (!domain) return Error(domain.error());
  auto owner = std::make_shared<DerivedDomain>(
      DerivedDomain{.charge = std::move(charge), .domain = std::move(*domain)});
  return std::shared_ptr<const engine::NumericalDomain>(owner, &owner->domain);
}

namespace {

// PlanGraph owns one temporary reader index per synchronous lowering; current
// and future retained programs do not retain that index. This named category
// is both paid by automatic construction and enforced before index allocation.
constexpr std::uint64_t kServingReaderScratchBytes =
    4096ULL * (ir::Graph::kMaxNodes + ir::Graph::kMaxTensors);

// Called under the Server's parser loan after source/schema validation, and
// before any BuildStep/lowering. Graph maxima are the existing IR contract;
// commit roots come from the actual component inventory and owner envelope.
struct ServingConstruction {
  std::uint64_t compiler = 0, retained = 0;
};
std::expected<ServingConstruction, std::string> ServingConstructionBytes(
    const spec::ModelSpec& model, const ja::Artifact& target, const ja::Artifact* draft,
    const ModelSettings& settings) {
  auto saves = components::CommitSaveBound(model, settings.max_slots.value);
  if (!saves) return Error(saves.error());
  constexpr std::uint64_t nodes = ir::Graph::kMaxNodes;
  constexpr std::uint64_t tensors = ir::Graph::kMaxTensors;
  // Twelve overlapping graph representations (construction/clone/plan facts
  // and the current/future compiler), reader indexes, and descriptor arena.
  // Cache/recording lifetimes still pay PlanAccount independently.
  std::uint64_t descriptors = tensors + 16 + 31 * nodes;
  std::uint64_t extra = 0;
  if (__builtin_mul_overflow(*saves, std::uint64_t{4}, &extra) ||
      __builtin_add_overflow(descriptors, extra, &descriptors))
    return Error("component saved-root descriptor payment overflows");
  std::uint64_t bytes = 12 * (480 * nodes + 192 * tensors) + kServingReaderScratchBytes;
  const auto add = [&](std::uint64_t count, std::uint64_t unit) {
    std::uint64_t part = 0;
    return !__builtin_mul_overflow(count, unit, &part) &&
           !__builtin_add_overflow(bytes, part, &bytes);
  };
  if (ggml_tensor_overhead() > 512 || !add(descriptors, 512 + 1024 + 32))
    return Error("component descriptor ABI or payment exceeds its construction inventory");
  ServingConstruction result{.compiler = bytes};
  bytes = 0;
  // The complete spec and canonical/digest copies, parameter/role maps and
  // retained source bindings. Serialization is still inside the parser loan.
  const auto canonical = spec::ToJson(model);
  if (!add(canonical.size(), 64) || !add(1, std::uint64_t{64} << 20U))
    return Error("component spec construction payment overflows");
  const auto source = [&](const ja::Artifact& artifact) {
    if (!add(artifact.resources().size() + artifact.expert_arrays().size(), 16384) ||
        !add(artifact.layout().groups.size(), 4096) ||
        !add(artifact.files().size() + artifact.sources().size(), 4096))
      return false;
    for (const auto& file : artifact.files())
      if (file.role != ja::FileRole::kShard && !add(file.bytes.value(), 8)) return false;
    for (const auto& group : artifact.layout().groups)
      if (!add(group.chunks, 4096)) return false;
    for (const auto& array : artifact.expert_arrays())
      if (!add(array.count, 2048)) return false;
    return true;
  };
  if (!source(target) || (draft && !source(*draft)))
    return Error("component retained source inventory overflows");
  result.retained = bytes;
  return result;
}

struct ComponentHold {
  // First: refund only after all graph, binding and backend owners die.
  std::shared_ptr<ConstructionCharge> construction;
  std::shared_ptr<engine::ExecutionMetadataAllowance> automatic_construction;
  std::shared_ptr<engine::SharedCompilerAllowance> compiler;
  std::uint64_t compiler_bytes = 0;
  std::shared_ptr<ConstructionCharge> kv_encoding_charge;
  std::unique_ptr<StoredBindingPayment> stored_payment;
  spec::ModelSpec spec;
  std::vector<spec::Source> sources;  // by artifact: the target's, the drafter's
  std::vector<std::span<const components::BoundTensor>> tables;
  std::vector<model::StateLayoutDerived> layouts;  // by stage
  std::vector<const model::StateLayoutDerived*> stages;
  std::vector<spec::SourceValue> values, drafter_values;
  std::shared_ptr<const std::vector<std::int32_t>> draft_head_ids;
  std::uint32_t draft_head_rows = 0;
  // Each weight's place: its holding stage's artifact, whether it is
  // an expert array, and its resource's or array's index.
  using Place = ComponentWeightPlace;
  ComponentWeightPlaces place;
  std::map<ir::WeightRef, std::uint64_t> pitch;
  // Before the backend, whose launch borrows it (the runner's Release
  // destroys that launch, then retires the handle).
  std::unique_ptr<kernels::ggml::CublasHandle> cublas;
  std::unique_ptr<kernels::ggml::GgmlBackend> backend;
};
static_assert(sizeof(ComponentHold) <= 4096);  // StoredBindingBytes fixed owner inventory.

// Small integer recipe metadata read through the artifact's authenticated
// shard descriptor. The synchronous direct read retires before its aligned
// stack window is reused. Shape/type and exact byte count precede allocation.
template <class T>
std::expected<std::vector<T>, std::string> ReadRecipeIntegers(const ja::Artifact& artifact,
                                                              std::uint32_t resource,
                                                              std::uint64_t count64) {
  if (count64 == 0 || count64 > std::numeric_limits<std::size_t>::max() / sizeof(T))
    return Error("integer recipe metadata exceeds its addressable extent");
  const auto count = static_cast<std::size_t>(count64);
  auto place = artifact.ResourcePlacement(resource);
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

template <class T>
std::expected<std::vector<T>, std::string> RecipeIntegers(const ja::Artifact& artifact,
                                                          const model::Qwen38Tensor& tensor,
                                                          std::uint32_t maximum,
                                                          std::string_view type) {
  const bool vector = type == "I64" && tensor.plain && tensor.ne.size() == 1;
  const bool table = type == "I32" && !tensor.plain && tensor.ne.size() == 2 && tensor.ne[0] == 1;
  if (tensor.type != type || (!vector && !table))
    return Error("integer recipe metadata has an unsupported type or shape");
  const auto count = tensor.ne[table ? 1 : 0];
  if (count == 0 || count > maximum) return Error("integer recipe metadata exceeds its bound");
  return ReadRecipeIntegers<T>(artifact, tensor.index, count);
}

std::expected<void, std::string> CheckHashTables(const ja::Artifact& artifact,
                                                 const model::Dsv4Binding& binding) {
  const auto& profile = model::Dsv4Flash();
  for (std::uint32_t layer = 0; layer < profile.hash_layers; ++layer) {
    const auto& tensor = binding.layers.at(layer).tid2eid;
    if (tensor.type != "I32" || tensor.ne.size() != 2 || tensor.ne[0] != profile.experts_used ||
        tensor.ne[1] != profile.vocab)
      return Error("hash-routing source is not the bound expert-by-token table");
    // Serial reads: one complete table at a time, under the construction
    // allowance. The source table remains encoded; no result cache is kept.
    auto table = ReadRecipeIntegers<std::int32_t>(
        artifact, tensor.index, std::uint64_t{profile.experts_used} * profile.vocab);
    if (!table) return Error(table.error());
    if (auto valid = model::CheckDsv4HashRouting(profile, *table); !valid)
      return Error(valid.error());
  }
  return {};
}

// Native hash tables retain I32 source values. Validate each table through a
// bounded direct-read window before any routed work; no whole-table copy or
// uncharged payload allocation is introduced for stored targets.
std::expected<void, std::string> CheckStoredHashTables(const ja::Artifact& artifact) {
  const auto& profile = model::Dsv4Flash();
  for (std::uint32_t layer = 0; layer < profile.hash_layers; ++layer) {
    auto index = artifact.FindResource(std::format("blk.{}.ffn_gate_tid2eid.weight", layer));
    if (!index) return Error("stored hash-routing table is missing");
    auto place = artifact.ResourcePlacement(*index);
    const auto count = std::uint64_t{profile.experts_used} * profile.vocab;
    if (!place || place->bytes.value() != count * sizeof(std::int32_t))
      return Error("stored hash-routing table has an inconsistent byte count");
    const auto& group = artifact.groups()[place->group];
    const auto& shard = artifact.shards()[group.shard];
    const auto begin = shard.data_offset.value() + group.offset.value() + place->offset.value();
    const auto end = begin + place->bytes.value();
    if (begin % sizeof(std::int32_t)) return Error("stored hash-routing table is unaligned");
    auto file = artifact.OpenShardForDirectRead(group.shard);
    if (!file) return Error(file.error().ToString());
    alignas(4096) std::array<std::byte, 4096> window{};
    for (auto at = begin / window.size() * window.size(); at < end; at += window.size()) {
      if (at > shard.data_offset.value() + shard.data_bytes.value() ||
          window.size() > shard.data_offset.value() + shard.data_bytes.value() - at)
        return Error("stored hash-routing read exceeds its shard");
      if (auto read = platform::TransferDirectFile(file->get(), at, window, false); !read)
        return Error(std::format("reading stored hash-routing table: errno {}", read.error()));
      for (auto offset = std::max(at, begin); offset < std::min(at + window.size(), end);
           offset += sizeof(std::int32_t)) {
        std::int32_t expert = 0;
        std::memcpy(&expert, window.data() + offset - at, sizeof(expert));
        if (expert < 0 || std::cmp_greater_equal(expert, profile.experts))
          return Error("stored hash-routing table names an invalid expert");
      }
    }
  }
  return {};
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
  if (architecture == "deepseek4") {
    auto binding = model::BindDsv4(model::Dsv4Flash(), artifact);
    if (!binding) return Error(binding.error());
    if (auto checked = CheckHashTables(artifact, *binding); !checked) return Error(checked.error());
    return spec::mappings::Dsv4V0Values(model::Dsv4Flash());
  }
  return ComponentV0Values(artifact);
}

// A drafter's v0 values (its compiled profile's, by its target's): the
// Gemma 4 assistant's of its target's approved profile, its tensors bound
// to that profile.
std::expected<std::vector<spec::SourceValue>, std::string> DrafterV0Values(
    const ja::Artifact& target, const ja::Artifact& drafter) {
  if (drafter.model().architecture == "dflash" && target.model().architecture == "deepseek4") {
    const auto& profile = model::Dsv4Flash();
    auto target_binding = model::BindDsv4(profile, target);
    if (!target_binding) return Error(target_binding.error());
    const auto& companion = model::DsparkDeepSeekV4Flash();
    if (auto bound = model::BindDspark(companion, drafter, profile, *target_binding); !bound)
      return Error(bound.error());
    return spec::mappings::DsparkV0Values(companion);
  }
  return ComponentDrafterV0Values(target, drafter);
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

bool DeepSeekDomain(const ComponentBuildOptions& options, bool companion) {
  if (!options.execution_domain) return false;
  const auto stages = options.execution_domain->stages();
  constexpr std::array uses{engine::NumericalUse::kPrompt, engine::NumericalUse::kDecode};
  constexpr std::array phases{engine::NumericalPhase{.phase = planner::Phase::kPrefill,
                                                     .max_owners = 1,
                                                     .max_owner_rows = 512,
                                                     .max_total_rows = 512},
                              engine::NumericalPhase{.phase = planner::Phase::kDecode,
                                                     .max_owners = 1,
                                                     .max_owner_rows = 1,
                                                     .max_total_rows = 1}};
  if (!companion) {
    if (stages.size() != 1 || stages[0].stage != 0 || stages[0].name != "target") return false;
    constexpr std::array scored_uses{engine::NumericalUse::kPrompt, engine::NumericalUse::kDecode,
                                     engine::NumericalUse::kScore};
    constexpr std::array scored_phases{phases[0], phases[1],
                                       engine::NumericalPhase{.phase = planner::Phase::kScore,
                                                              .max_owners = 1,
                                                              .max_owner_rows = 128,
                                                              .max_total_rows = 128}};
    return (std::ranges::equal(stages[0].uses, uses) &&
            std::ranges::equal(stages[0].phases, phases)) ||
           (std::ranges::equal(stages[0].uses, scored_uses) &&
            std::ranges::equal(stages[0].phases, scored_phases));
  }
  constexpr std::array composed_uses{engine::NumericalUse::kPrompt, engine::NumericalUse::kDecode,
                                     engine::NumericalUse::kPromptCompanions,
                                     engine::NumericalUse::kSpeculative};
  constexpr std::array target_phases{phases[0], phases[1],
                                     engine::NumericalPhase{.phase = planner::Phase::kVerify,
                                                            .max_owners = 1,
                                                            .max_owner_rows = 4,
                                                            .max_total_rows = 4}};
  constexpr std::array draft_phases{engine::NumericalPhase{
      .phase = planner::Phase::kDraft, .max_owners = 1, .max_owner_rows = 3, .max_total_rows = 3}};
  return stages.size() == 2 && stages[0].stage == 0 && stages[0].name == "target" &&
         stages[1].stage == 1 && stages[1].name == "dspark" &&
         std::ranges::equal(stages[0].uses, composed_uses) &&
         std::ranges::equal(stages[1].uses, composed_uses) &&
         std::ranges::equal(stages[0].phases, target_phases) &&
         std::ranges::equal(stages[1].phases, draft_phases);
}

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
  constexpr std::array ordinary_uses{engine::NumericalUse::kPrompt, engine::NumericalUse::kDecode,
                                     engine::NumericalUse::kScore};
  if (stages.size() != 1 || stages[0].stage != 0 || stages[0].name != "target" ||
      (!std::ranges::equal(stages[0].uses, uses) &&
       !std::ranges::equal(stages[0].uses, ordinary_uses)))
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

// Derive only from an actual target-only ordinary request. Explicit caller
// domains keep their existing (including closed scoring) admission contract.
bool PlainServingEnvelope(const ModelSettings& settings, std::string_view drafter,
                          const ComponentBuildOptions& options) {
  return options.diagnostic_solo && options.derive_execution_domain && !options.execution_domain &&
         drafter.empty() && !settings.speculation.value && settings.max_slots.value >= 1 &&
         settings.max_slots.value <= Llm::kMaxBranches && settings.context.value != 0 &&
         settings.prefill_chunk.value != 0;
}

// Internal composed diagnostic only. The actual source seal additionally
// validates every stage/layout/phase and binds real prepared backend owners.
// Ordinary Server admission remains single-owner until qualification closes.
bool JoinedMtpEnvelope(const ModelSettings& settings, std::string_view drafter,
                       const ComponentBuildOptions& options) {
  if (!options.diagnostic_solo || (!options.execution_domain && !options.derive_execution_domain) ||
      drafter.empty() || !settings.speculation.value || settings.max_slots.value < 2 ||
      settings.max_slots.value > Llm::kMaxBranches || !settings.context.value ||
      !settings.prefill_chunk.value || !settings.draft_rows.value)
    return false;
  if (options.derive_execution_domain) return !options.execution_domain;
  const auto& domain = *options.execution_domain;
  return domain.stages().size() == 2 &&
         domain.CheckMode(engine::NumericalUse::kJoinedRepeatPending).has_value() &&
         domain.CheckMode(engine::NumericalUse::kSpeculative).has_value() &&
         domain.CheckMode(engine::NumericalUse::kPromptCompanions).has_value();
}

}  // namespace

EngineBuild ComponentLlmBuild(engine::PagedNode& node, fs::path store,
                              const config::ModelEntry& entry, int index,
                              ComponentBuildOptions options) {
  return [&node, store = std::move(store), id = entry.artifact.value_or(""),
          drafter_id = entry.drafter.value_or(""), index, options = std::move(options),
          metadata = std::shared_ptr<engine::ExecutionMetadataAllowance>{},
          stored_metadata = std::shared_ptr<engine::ExecutionMetadataAllowance>{}](
             const ModelSettings& settings) mutable -> std::expected<EngineLlmParts, std::string> {
    const bool solo = settings.architecture == "qwen4exp";
    if (options.derive_execution_domain && (!options.diagnostic_solo || options.execution_domain))
      return Error(
          "derived serving qualification needs an internal request without a caller domain");
    const bool deepseek = settings.architecture == "deepseek4";
    const bool dspark = deepseek && !drafter_id.empty();
    const bool deepseek_serving =
        deepseek && options.derive_execution_domain && options.automatic_startup_cap != 0 &&
        !options.construction_metadata && !options.target_policy && settings.context.value != 0 &&
        settings.prefill_chunk.value != 0 &&
        settings.prefill_chunk.value < settings.context.value && settings.max_slots.value != 0 &&
        settings.max_slots.value <= Llm::kMaxBranches && (!settings.speculation.value || dspark);
    if (deepseek_serving && (settings.prefill_outa_hca.source == SettingSource::kOverride ||
                             settings.prefill_outa_hca_partial.source == SettingSource::kOverride))
      return Error("the shared diagnostic does not consume legacy prefill fusion overrides");
    if (deepseek && !deepseek_serving &&
        (!options.diagnostic_solo || !DeepSeekDomain(options, dspark) ||
         !options.construction_metadata || options.target_policy || settings.max_slots.value != 1 ||
         settings.context.value != 3072 || settings.prefill_chunk.value != 512 ||
         settings.speculation.value != dspark ||
         (dspark && (drafter_id != kDsparkDiagnosticArtifact || settings.draft_rows.value != 3)) ||
         id != kDsv4DiagnosticArtifact))
      return Error("the DeepSeek target bridge requires its paid single-owner diagnostic envelope");
    std::shared_ptr<ConstructionCharge> construction;
    if (deepseek && !deepseek_serving) {
      if (ggml_tensor_overhead() > 512)
        return Error("DeepSeek native descriptor ABI exceeds its construction inventory");
      if (!options.construction_metadata->Charge(kDsv4ConstructionBytes))
        return Error("DeepSeek component construction exceeds its admitted metadata allowance");
      construction = std::make_shared<ConstructionCharge>();
      construction->allowance = options.construction_metadata;
      construction->bytes = kDsv4ConstructionBytes;
    }
    const bool joined_score = solo && JoinedScoreEnvelope(settings, drafter_id, options);
    const bool joined_mtp = solo && JoinedMtpEnvelope(settings, drafter_id, options);
    const bool plain_serving = solo && PlainServingEnvelope(settings, drafter_id, options);
    if (options.derive_execution_domain && !plain_serving && !joined_mtp && !deepseek_serving)
      return Error(
          "derived serving qualification needs an ordinary target or composed replay source");
    if (solo && (!options.diagnostic_solo ||
                 (settings.max_slots.value != 1 && !joined_score && !joined_mtp && !plain_serving)))
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
    // Stored source construction is admitted before parsing/copying its spec,
    // component parameters or role maps. Retain one catalog payment on retries.
    std::unique_ptr<StoredBindingPayment> stored_payment;
    if (stored) {
      if (auto admitted = CheckComponentArtifactAdmission(*artifact); !admitted)
        return Error(admitted.error());
      auto bytes = StoredBindingBytes(*artifact);
      if (!bytes) return Error(bytes.error());
      if (!stored_metadata) {
        std::uint64_t total = 0;
        if (__builtin_add_overflow(*bytes, engine::ExecutionMetadataAllowance::OverheadBytes(),
                                   &total))
          return Error("stored construction catalog payment overflows");
        auto paid = engine::ExecutionMetadataAllowance::Admit(
            node.catalog(), node.domain(), total, platform::AvailableMemoryBytes().value_or(0),
            kUncountedMargin);
        if (!paid) return Error(paid.error());
        stored_metadata = std::move(*paid);
      }
      auto paid = StoredBindingPayment::Create(*artifact, stored_metadata);
      if (!paid) return Error(paid.error());
      stored_payment = std::move(*paid);
    }
    auto hold = std::make_shared<ComponentHold>();
    hold->construction = std::move(construction);
    hold->stored_payment = std::move(stored_payment);
    std::optional<StoredComponentBindings> stored_bindings;
    std::optional<model::Qwen38Binding> qwen;
    std::optional<spec::PolicyRule> stored_policy;
    std::expected<spec::ModelSpec, std::string> synthesized;
    if (stored) {
      auto bindings = ReadStoredComponents(*artifact, std::move(hold->stored_payment));
      if (!bindings) return Error(bindings.error());
      stored_bindings.emplace(std::move(*bindings));
      if (settings.context.value == 0 ||
          settings.context.value > stored_bindings->model.stages.front().context)
        return Error("resolved context exceeds the admitted stored spec");
      auto policy = ComponentAdmissionPolicy(
          stored_bindings->model, stored_bindings->source, artifact->sources(),
          {.context = settings.context.value, .slots = settings.max_slots.value});
      if (!policy) return Error(policy.error());
      stored_policy = *policy;
      if (deepseek)
        if (auto checked = CheckStoredHashTables(*artifact); !checked)
          return Error(checked.error());
      hold->stored_payment = std::move(stored_bindings->payment);
      synthesized = std::move(stored_bindings->model);
      hold->sources.push_back(std::move(stored_bindings->source));
      hold->place = std::move(stored_bindings->places);
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
          const auto width = std::min<std::size_t>(settings.draft_vocab.value, ids->size());
          if (width == 0) return Error("a selected draft head needs a positive source row count");
          ids->resize(width);
          hold->draft_head_ids = std::make_shared<const std::vector<std::int32_t>>(std::move(*ids));
        }
        drafter_mapping = &spec::mappings::Qwen38MtpReplayMapping();
        dvalues = spec::mappings::Qwen38MtpV0Values(model::Qwen38Flash(), *binding);
        if (dvalues)
          for (auto& value : *dvalues)
            if (value.key == "qwen4exp-mtp.draft_vocab") {
              const auto rows = binding->selected_head() ? binding->draft_output.ne.at(1)
                                                         : model::Qwen38Flash().vocab;
              hold->draft_head_rows = static_cast<std::uint32_t>(
                  std::min<std::uint64_t>(settings.draft_vocab.value, rows));
              value.value = static_cast<std::int64_t>(hold->draft_head_rows);
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
    if (settings.kv_cache_encoding.value != "f16") {
      if (settings.kv_cache_encoding.value != "q8_0" || !drafter_id.empty())
        return Error("unsupported KV encoding or drafter composition");
      auto bytes = components::KvCacheEncodingBytes(hold->spec, format::Encoding::kQ8_0);
      if (!bytes) return Error(bytes.error());
      if (!metadata) {
        auto admitted = engine::ExecutionMetadataAllowance::Admit(
            node.catalog(), node.domain(), options.execution_metadata_bytes,
            platform::AvailableMemoryBytes().value_or(0), kUncountedMargin);
        if (!admitted) return Error(admitted.error());
        metadata = std::move(*admitted);
      }
      if (!metadata->Charge(*bytes))
        return Error("Q8_0 KV source metadata allowance is insufficient");
      hold->kv_encoding_charge = std::make_shared<ConstructionCharge>();
      hold->kv_encoding_charge->allowance = metadata;
      hold->kv_encoding_charge->bytes = *bytes;
      if (auto configured =
              components::ApplyKvCacheEncoding(hold->spec, format::Encoding::kQ8_0, *bytes);
          !configured)
        return Error(configured.error());
    } else if (auto configured =
                   components::ApplyKvCacheEncoding(hold->spec, format::Encoding::kF16, 0);
               !configured) {
      return Error(configured.error());
    }
    for (const spec::Source& s : hold->sources) hold->tables.emplace_back(s.tensors);
    if (joined_mtp && options.execution_domain &&
        hold->spec.stages.size() != options.execution_domain->stages().size())
      return Error("joined MTP requires every actual source stage");
    if (joined_score &&
        (hold->spec.stages.size() != 1 || hold->spec.stages.front().name != "target"))
      return Error("joined scoring requires the actual single target stage");
    if (auto valid = components::ValidateSpec(hold->spec, hold->tables); !valid) {
      return Error(valid.error());
    }
    if (options.automatic_startup_cap != 0) {
      // Already migrated component families use the same paid construction
      // inventory without inventing a NumericalDomain. Newly opted families
      // still require their explicit diagnostic request and derived domain.
      if ((!OnComponentEngine(settings.architecture) &&
           (!options.diagnostic_solo || !options.derive_execution_domain)) ||
          hold->construction)
        return Error("automatic construction requires a qualified shared serving route");
      auto bytes =
          ServingConstructionBytes(hold->spec, *artifact, drafter ? &*drafter : nullptr, settings);
      if (!bytes) return Error(bytes.error());
      if (!node.open_for_setup() || !options.shared_compiler)
        return Error("automatic compiler admission requires its stopped Server owner");
      std::uint64_t total = 0;
      if (__builtin_add_overflow(bytes->compiler,
                                 engine::ExecutionMetadataAllowance::OverheadBytes(), &total) ||
          __builtin_add_overflow(total, kExtent - 1, &total))
        return Error("shared compiler payment overflows");
      total = total / kExtent * kExtent;
      if (auto paid = options.shared_compiler->Reserve(
              node.catalog(), node.domain(), total, options.automatic_startup_cap,
              platform::AvailableMemoryBytes().value_or(0), kUncountedMargin);
          !paid)
        return Error(paid.error());
      hold->compiler = options.shared_compiler;
      hold->compiler_bytes = bytes->compiler;
      auto owner =
          serving_support::AdmitStartup(node, bytes->retained, options.automatic_startup_cap);
      if (!owner) return Error(owner.error());
      hold->automatic_construction = std::move(*owner);
    }
    // The spec bound to the artifacts its stages run from (each stage's
    // index into them; D-105's layout identity and the plans' key): the
    // target's ID, then the drafter's.
    std::vector<std::string_view> artifacts{id};
    if (speculate) artifacts.emplace_back(drafter_id);
    auto bound = spec::BoundDigest(hold->spec, artifacts);
    if (!bound) return Error(bound.error());
    if (settings.bound_spec && *settings.bound_spec != *bound)
      return Error("resolved calibration spec differs from the actual component spec");
    // Each state tensor starts on its own paged extent, as the M3 layouts
    // did: packed tensors would straddle extents, so their growth would
    // cross extent boundaries at staggered positions and a prompt would
    // grow its state (each growth a fixed host cost between steps) every
    // chunk or two instead of once every extent's worth of cells.
    const auto configured_chunk =
        solo ? PrefillChunkRows(settings.context.value, std::nullopt, settings.prefill_chunk.value,
                                model::Qwen38MostRows(settings.context.value, false))
             : settings.prefill_chunk.value;
    const auto depth = solo     ? (configured_chunk >= 4 ? settings.draft_rows.value
                                                         : std::min(settings.draft_rows.value, 2U))
                       : dspark ? settings.draft_rows.value
                                : kAssistantDepth;
    if (speculate && dspark &&
        (depth == 0 || depth == UINT32_MAX || depth > model::DsparkDeepSeekV4Flash().block_size))
      return Error("the resolved draft depth exceeds the bound block drafter");
    for (std::uint16_t stage = 0; stage < hold->spec.stages.size(); ++stage) {
      auto layout = components::StageLayout(hold->spec, stage, settings.context.value,
                                            configured_chunk, deepseek_serving ? 256ULL : kExtent);
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
    if (speculate && std::ranges::any_of(hold->layouts, [verify_rows](const auto& layout) {
          return verify_rows > layout.max_rows;
        }))
      return Error("the complete drafter verify exceeds its actual stage row bounds");
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
      if (speculate && dspark) {
        // Block proposals produce every head, including short reply tails;
        // every verify includes its anchor. No autoregressive catch-up pass.
        for (std::uint32_t rows = 1; rows <= depth; ++rows)
          add(shape(1, planner::Phase::kDraft, owners, rows, rows, false));
        for (std::uint32_t rows = 1; rows <= verify_rows; ++rows)
          add(shape(0, planner::Phase::kVerify, owners, rows, rows, false));
      } else if (speculate) {
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
    if (deepseek && !dspark && options.execution_domain &&
        options.execution_domain->CheckMode(engine::NumericalUse::kScore)) {
      // Full-head Score has a different output/workspace envelope from the
      // frontier-only Prefill. Actual caller schedules may add their complete
      // compressor/read classes before Setup; the factory measures the maxima.
      for (const std::uint32_t rows : {128U, 1U})
        setup.push_back(shape(0, planner::Phase::kScore, 1, rows, rows, false));
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
    if (solo && !stored)
      placements[0].places.at(artifact->resources()[qwen->ple_table.index].group) =
          engine::GroupPlace::kNone;
    std::optional<engine::RunnerConfig::RowTable> row_table;
    std::uint32_t logit_width = 0;
    if (stored_bindings) {
      logit_width = stored_bindings->logit_width;
      for (const auto& [key, place] : hold->place)
        if (place.array)
          hold->pitch[key.first] = placements.at(place.artifact).pitch.at(place.index);
      for (const auto& row : stored_bindings->rows)
        placements.at(hold->spec.stages.at(row.weight.stage).artifact)
            .places.at(artifact->resources()[row.resource].group) = engine::GroupPlace::kNone;
    }
    // v0 retains its established synthesis path. Stored v1 maps came from the
    // validated component roles above; no discovery-only graphs are built.
    if (!stored) {
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
                *found == qwen->ple_table.index) {
              row_table = {.weight = weight.ref,
                           .resource = *found,
                           .format = step->graph.tensor(weight.tensor).format,
                           .part = role,
                           .rows_per_token = model::Qwen38Flash().ple_heads()};
              if (weight.parts.size() > format::PartList::kMax)
                return Error("a row table exceeds its format part inventory");
              for (const auto& [part_name, part_role] : weight.parts) {
                const auto part_resource = a.FindResource(part_name);
                if (!part_resource) return Error("a row table source part is absent");
                row_table->source_parts.emplace_back(part_role, *part_resource);
              }
            }
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
    }
    if (logit_width == 0) return Error("the spec's target stage has no head");
    // A layer's routed experts in one slab (the M3 runners' pitch).
    if (solo && !row_table && (!stored_bindings || stored_bindings->rows.empty()))
      return Error("the component recipe has no n-gram row table");
    ComponentHold* held = hold.get();
    auto backend = kernels::ggml::GgmlBackend::Create(
        0, node.execution(), node.stream(static_cast<std::uint32_t>(index)), nullptr,
        {.lower = {.device_masks = true,
                   // The runner stages a frozen read's host mask (its cells'
                   // positions before the rows); the device's are made
                   // anyway.
                   .frozen_host_masks = true,
                   .index_scratch_limit = deepseek_serving ? kServingReaderScratchBytes
                                          : deepseek ? kDsv4ReaderScratchBytes
                                                     : std::numeric_limits<std::uint64_t>::max(),
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
    engine::DecodingMode mode;
    if (speculate) {
      mode = {.kind = engine::DecodingMode::Kind::kSpeculative,
              .drafters = {{.stage = 1,
                            .shape = dspark ? engine::DrafterSpec::Shape::kBlock
                                            : engine::DrafterSpec::Shape::kAutoregressive,
                            .max_depth = depth,
                            .shared_wave_depth =
                                solo ? std::min(settings.shared_wave_depth.value, depth) : 0,
                            .catch_up = solo,
                            .fill = dspark ? model::DsparkDeepSeekV4Flash().mask_token : 0,
                            .borrows_target = !solo && !dspark,
                            .anchored = !solo && !dspark,
                            .features = !dspark,
                            .adaptive = solo && depth >= 3,
                            .depth_cost = solo ? settings.depth_cost_ratio.value : 1.16,
                            .device_tokens = solo,
                            .draft_before_limit = solo,
                            .head_ids = hold->draft_head_ids,
                            .head_rows = hold->draft_head_rows}},
              .combine = engine::DecodingMode::Drafters::kCascade,
              .row_invariant = false};
    }
    auto execution_domain = options.execution_domain;
    if (options.derive_execution_domain && !metadata) {
      auto admitted = engine::ExecutionMetadataAllowance::Admit(
          node.catalog(), node.domain(), options.execution_metadata_bytes,
          platform::AvailableMemoryBytes().value_or(0), kUncountedMargin);
      if (!admitted) return Error(admitted.error());
      metadata = std::move(*admitted);
    }
    if (options.derive_execution_domain) {
      auto domain = ComponentServingDomain(hold->spec, hold->layouts, slots, mode, metadata);
      if (!domain) return Error(domain.error());
      execution_domain = std::move(*domain);
    }
    if (options.tuning && !execution_domain)
      return Error("explicit tuning requires a declared execution source");
    if (execution_domain) {
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
    if ((stored && !composition.text.empty()) || options.target_policy) {
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
           .bindings = {.tensors = hold->tables.at(hold->spec.stages.at(stage).artifact),
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
           .execution_domain = execution_domain,
           .validate_program = options.validate_program,
           .compiler = hold->compiler,
           .compiler_bytes = hold->compiler_bytes});
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
    if (hold->automatic_construction) {
      parts.startup_host_bytes = hold->automatic_construction->capacity() +
                                 engine::ExecutionMetadataAllowance::OverheadBytes();
      parts.startup_cap = options.automatic_startup_cap;
    }
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
    if (stored_bindings) parts.config.row_tables = std::move(stored_bindings->rows);
    if (row_table) parts.config.row_tables.push_back(*row_table);
    if (options.capture_first_verify)
      parts.config.capture_first_phases = 1U << static_cast<unsigned>(planner::Phase::kVerify);
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
      parts.mode = std::move(mode);
    }
    if (deepseek_serving && speculate) {
      execution::AdaptiveWaveMode::Costs costs{};
      for (std::size_t i = 0; i < settings.wave_costs.value.size() && i + 2 < costs.size(); ++i)
        costs[i + 2] = settings.wave_costs.value[i];
      const auto force = settings.wave_form.value == config::WaveForm::kPlain
                             ? execution::AdaptiveWaveMode::Force::kPlain
                         : settings.wave_form.value == config::WaveForm::kSpeculative
                             ? execution::AdaptiveWaveMode::Force::kSpeculative
                             : execution::AdaptiveWaveMode::Force::kNone;
      // The shared composer admits full per-owner verifies or separates them;
      // it has no legacy fixed sixteen-row lane that truncates each owner.
      parts.wave_mode.emplace(costs, force);
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
    if (deepseek && !deepseek_serving) {
      // Closed first target screen: serial completed chunks, with no extra
      // future program owners; normal prefill/decode execution is unchanged.
      parts.prefill.lookahead = 0;
      parts.prefill.capture_ahead = false;
      parts.prefill.prepare_ahead = false;
      parts.prefill.state_only = false;
    }
    if (solo || deepseek_serving) {
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
