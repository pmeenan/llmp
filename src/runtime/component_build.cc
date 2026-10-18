// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "runtime/component_build.h"

#include <algorithm>
#include <array>
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
#include "components/state_encoding.h"
#include "components/step.h"
#include "engine/component_source.h"
#include "engine/decoding.h"
#include "engine/expert_slab.h"
#include "engine/paged_weights.h"
#include "engine/ple_rows.h"
#include "format/tensor_format.h"
#include "kernels/decode/selected_grouped.h"
#include "kernels/decode/selected_rows.h"
#include "kernels/decode/selected_source.h"
#include "kernels/ggml/cublas.h"
#include "kernels/ggml/step_program.h"
#include "planner/policy.h"
#include "planner/step_shape.h"
#include "platform/host_probe.h"
#include "providers/device_runtime.h"
#include "runtime/component_admission.h"
#include "runtime/component_bindings.h"
#include "runtime/memory_guard.h"
#include "runtime/model_settings.h"
#include "runtime/prefill.h"
#include "runtime/serving_support.h"
#include "spec/mapping.h"
#include "spec/spec.h"
#include "spec/stored.h"

namespace llmp::runtime {
namespace {

using serving_support::Error;
using serving_support::kExtent;

namespace fs = std::filesystem;
namespace ja = llmp::artifact;

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
// is both paid by automatic construction and enforced, for every source,
// before index allocation (its inventory bound: 1,024 bytes an occurrence).
// The largest measured (below) is 26.9 MB, DeepSeek V4's.
constexpr std::uint64_t kServingReaderScratchBytes = 128ULL << 20;

// Each of a launch's concurrent lanes' scratch at most (GGML's
// ConfigureLanes), carved from the end of the node's pool.
constexpr std::uint64_t kLaneScratchBytes = 4ULL * 1024ULL * 1024ULL;

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
  // One step's construction at the IR's maxima, whatever the actual graph:
  // its graph (480 bytes a node and 192 a tensor for the built graph, its
  // clone and the plan's facts), the lowering's reader index (enforced at
  // that bound above), and GGML's descriptor arena (a descriptor, its
  // record and pointer for every tensor, node and saved commit root).
  // Compiles run one at a time on the driver, so one such payment (shared
  // across aliases) covers each; what a plan or graph keeps is charged
  // separately (PlanAccount). Measured on spark-b (2026-10-15, VmHWM reset
  // before each ComponentStepSource::Build, the largest growth across a
  // no-diagnostic `llmp-runtime chat` turn of each family; setup shapes
  // included): DeepSeek V4 + DSpark 40.3 MB, Gemma 4 26B-A4B 23.6 MB, 31B
  // 20.4 MB, Gemma 3 8.6 MB, Gemma 2 7.1 MB, Qwen3.8 (NVFP4 + MTP, GGUF) 22.2
  // and 22.3 MB; this charge is about 501 MB.
  std::uint64_t descriptors = tensors + 16 + nodes;
  std::uint64_t extra = 0;
  if (__builtin_mul_overflow(*saves, std::uint64_t{4}, &extra) ||
      __builtin_add_overflow(descriptors, extra, &descriptors))
    return Error("component saved-root descriptor payment overflows");
  std::uint64_t bytes = (480 * nodes + 192 * tensors) + kServingReaderScratchBytes;
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
  // Measured the same way, a model's setup grows the resident set by at
  // most 64 MB (Gemma 4 31B; DeepSeek 38, Qwen3.8 none) past its cataloged
  // pinned and runtime extents, against this charge's 187 MB (Gemma 2) to
  // 384 MB (Gemma 4 31B).
  const auto canonical = spec::ToJson(model);
  if (!add(canonical.size(), 2) || !add(1, std::uint64_t{64} << 20U))
    return Error("component spec construction payment overflows");
  const auto source = [&](const ja::Artifact& artifact) {
    std::uint64_t non_expert = 0;
    for (const auto& r : artifact.resources()) {
      if (r.group < artifact.groups().size() &&
          artifact.groups()[r.group].kind != ja::GroupKind::kExpert) {
        ++non_expert;
      }
    }
    if (!add(non_expert + artifact.expert_arrays().size(), 16384) ||
        !add(artifact.resources().size(), 256) || !add(artifact.layout().groups.size(), 256) ||
        !add(artifact.files().size() + artifact.sources().size(), 4096))
      return false;
    for (const auto& group : artifact.layout().groups)
      if (!add(group.chunks, 128)) return false;
    return true;
  };
  if (!source(target) || (draft && !source(*draft)))
    return Error("component retained source inventory overflows");
  result.retained = bytes;
  return result;
}

struct ComponentHold {
  // First: refund only after all graph, binding and backend owners die.
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
  std::vector<spec::SourceValue> drafter_values;
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

// Closed internal scoring experiment, not a serving route: only a caller's
// explicit execution domain reaches it (the Server derives its own, so no
// served configuration does). Its four owners, 512-token context, 2-row
// chunks and 256-byte read alignment are that experiment's one matched
// shape (tests/unit/qwen38_joined_score_gpu_test.cc), fixed so its domain
// is checked against exactly what it was matched at; a non-null domain
// alone is not authority to lift the single-owner bound.
bool JoinedScoreEnvelope(const ModelSettings& settings, std::string_view drafter,
                         const ComponentBuildOptions& options) {
  if (!options.shared_route || !options.execution_domain || !drafter.empty() ||
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
  return options.shared_route && options.derive_execution_domain && !options.execution_domain &&
         drafter.empty() && !settings.speculation.value && settings.max_slots.value >= 1 &&
         settings.max_slots.value <= Llm::kMaxBranches && settings.context.value != 0 &&
         settings.prefill_chunk.value != 0;
}

// The joined MTP envelope: a composed (speculating) source of two or more
// owners on the shared route, its domain derived from the actual source by
// the Server's serving request, or a caller's explicit domain with every
// joined mode. The actual source seal additionally validates every
// stage/layout/phase and binds real prepared backend owners.
bool JoinedMtpEnvelope(const ModelSettings& settings, std::string_view drafter,
                       const ComponentBuildOptions& options) {
  if (!options.shared_route || (!options.execution_domain && !options.derive_execution_domain) ||
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
    // The architecture's serving recipe (its mapping's data, spec/mapping.h).
    const spec::Mapping* recipe_mapping =
        spec::FindMapping(settings.architecture, settings.container);
    if (recipe_mapping == nullptr || recipe_mapping->serving == nullptr)
      return Error(
          std::format("the component engine serves no architecture {}", settings.architecture));
    const spec::ServingRecipe& recipe = *recipe_mapping->serving;
    // A shared route qualified by its plain or joined envelopes (else one
    // owner), or one served only in its automatic serving shape.
    const bool solo = recipe.shared_route && recipe.qualified_envelopes;
    if (options.derive_execution_domain && (!options.shared_route || options.execution_domain))
      return Error(
          "derived serving qualification needs an internal request without a caller domain");
    const bool automatic_only = recipe.shared_route && !recipe.qualified_envelopes;
    const bool automatic_serving =
        automatic_only && options.derive_execution_domain && options.automatic_startup_cap != 0 &&
        !options.target_policy && settings.context.value != 0 &&
        settings.prefill_chunk.value != 0 &&
        settings.prefill_chunk.value < settings.context.value && settings.max_slots.value != 0 &&
        settings.max_slots.value <= Llm::kMaxBranches &&
        (!settings.speculation.value || !drafter_id.empty());
    if (automatic_only && !automatic_serving)
      return Error(std::format("the {} shared route serves only its automatic serving envelope",
                               recipe.name));
    // The shared route's serving shape.
    const bool shared = solo || automatic_serving;
    const bool joined_score = solo && JoinedScoreEnvelope(settings, drafter_id, options);
    const bool joined_mtp = solo && JoinedMtpEnvelope(settings, drafter_id, options);
    const bool plain_serving = solo && PlainServingEnvelope(settings, drafter_id, options);
    if (options.derive_execution_domain && !plain_serving && !joined_mtp && !automatic_serving)
      return Error(
          "derived serving qualification needs an ordinary target or composed replay source");
    if (solo && (!options.shared_route ||
                 (settings.max_slots.value != 1 && !joined_score && !joined_mtp && !plain_serving)))
      return Error(std::format(
          "the {} component qualification requires exactly one configured slot", recipe.name));
    auto artifact = OpenInstalled(store, id);
    if (!artifact) return Error(artifact.error());

    const spec::Mapping* mapping =
        spec::FindMapping(artifact->model().architecture, artifact->model().container);
    if (mapping == nullptr) return Error("the artifact's architecture has no mapping");
    if (settings.architecture != artifact->model().architecture ||
        settings.container != artifact->model().container)
      return Error("resolved architecture/container differs from the opened artifact");
    // The artifact's prerequisites are admitted before its spec, component
    // parameters or role maps are parsed or copied (a v1 artifact's alone:
    // D-118); those are admitted once read, below. Retain one catalog
    // payment on retries.
    std::unique_ptr<StoredBindingPayment> stored_payment;
    {
      if (auto admitted = CheckArtifactPrerequisites(*artifact); !admitted)
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
    hold->stored_payment = std::move(stored_payment);
    std::optional<StoredComponentBindings> stored_bindings;
    std::optional<spec::PolicyRule> stored_policy;
    std::expected<spec::ModelSpec, std::string> synthesized;
    {
      auto bindings = ReadStoredComponents(*artifact, std::move(hold->stored_payment));
      if (!bindings) return Error(bindings.error());
      stored_bindings.emplace(std::move(*bindings));
      if (settings.context.value == 0 ||
          settings.context.value > stored_bindings->model.stages.front().context)
        return Error("resolved context exceeds the admitted stored spec");
      // Admitted once, as read; its policy from that admission.
      if (auto admitted = CheckComponentAdmission(stored_bindings->model, stored_bindings->source,
                                                  artifact->sources());
          !admitted)
        return Error(admitted.error());
      auto policy =
          AdmittedPolicy(stored_bindings->model, stored_bindings->source,
                         FindApproval(artifact->model().architecture, artifact->sources()),
                         {.context = settings.context.value, .slots = settings.max_slots.value});
      if (!policy) return Error(policy.error());
      stored_policy = *policy;
      if (auto checked = CheckStoredHashTables(*artifact, stored_bindings->model); !checked)
        return Error(checked.error());
      hold->stored_payment = std::move(stored_bindings->payment);
      synthesized = std::move(stored_bindings->model);
      hold->sources.push_back(std::move(stored_bindings->source));
      hold->place = std::move(stored_bindings->places);
    }
    const auto target_vocabulary = stored_bindings->logit_width;
    if (!target_vocabulary) return Error("target stored head has no vocabulary rows");
    // The drafter, a stage of its own, when the model speculates.
    const bool speculate = !drafter_id.empty() && settings.speculation.value;
    std::optional<ja::Artifact> drafter;
    const spec::Mapping* drafter_mapping = nullptr;
    if (speculate) {
      auto opened = OpenInstalled(store, drafter_id);
      if (!opened) return Error(opened.error());
      drafter.emplace(std::move(*opened));
      // Admitted once: its source as checked (its complete tensor
      // inventory its approval's, where it has one).
      auto drafter_admitted = ReadComponentDrafterAdmission(*drafter);
      if (!drafter_admitted) return Error(drafter_admitted.error());
      // The drafter's mapping as composed (its drafting recipe's replay
      // form; a stored one's by its approval).
      drafter_mapping = ComposedDrafterMapping(*drafter);
      if (drafter_mapping == nullptr || drafter_mapping->role != spec::StageRole::kDrafter) {
        return Error(std::format("the drafter's architecture {} has no drafter mapping",
                                 drafter->model().architecture));
      }
      auto dvalues = ComponentDrafterValues(*artifact, *drafter);
      if (!dvalues) return Error(dvalues.error());
      // The values its approval's composition was pinned with, before a
      // setting narrows the draft vocabulary.
      const std::vector<spec::SourceValue> approved_values = *dvalues;
      // The draft vocabulary (the recipe's vocab_key value): the configured
      // draft_vocab, bounded by the selected head's rows (its token map, the
      // recipe's head_ids) or, without a selected head, by the target's
      // vocabulary; the head's token IDs are read and checked.
      if (const spec::DraftRecipe* draft = drafter_mapping->draft;
          draft != nullptr && !draft->vocab_key.empty()) {
        auto map = DraftHeadRows(*drafter, draft->head_ids, target_vocabulary);
        if (!map) return Error(map.error());
        auto width = ConfigureDraftVocabulary(
            *dvalues, *draft, settings.draft_vocab.value, target_vocabulary,
            *map ? std::optional(static_cast<std::uint32_t>((*map)->rows)) : std::nullopt);
        if (!width) return Error(width.error());
        hold->draft_head_rows = *width;
        if (*map) {
          auto ids = ReadRecipeIntegers<std::int32_t>(*drafter, (*map)->resource, (*map)->rows);
          if (!ids) return Error(ids.error());
          // Strictly ascending target token IDs.
          for (std::size_t i = 0; i < ids->size(); ++i)
            if ((*ids)[i] < 0 || std::cmp_greater_equal((*ids)[i], target_vocabulary) ||
                (i != 0 && (*ids)[i] <= (*ids)[i - 1]))
              return Error("a selected draft head's token IDs are not strictly ascending");
          ids->resize(*width);
          hold->draft_head_ids = std::make_shared<const std::vector<std::int32_t>>(std::move(*ids));
        }
      }
      hold->drafter_values = std::move(*dvalues);
      spec::Source* dsource = &drafter_admitted->source;
      if (auto envelope = ComponentDrafterEnvelope(
              {.context = settings.context.value, .slots = settings.max_slots.value});
          !envelope)
        return Error(envelope.error());
      auto composed = spec::ComposeDrafter(*synthesized, *mapping, *drafter_mapping, *dsource,
                                           hold->drafter_values);
      if (!composed) return Error(composed.error());
      if (approved_values == hold->drafter_values) {
        if (auto approved = CheckComposedAdmission(*composed, *artifact, *drafter); !approved)
          return Error(approved.error());
      } else {
        auto pinned = spec::ComposeDrafter(*synthesized, *mapping, *drafter_mapping, *dsource,
                                           approved_values);
        if (!pinned) return Error(pinned.error());
        if (auto approved = CheckComposedAdmission(*pinned, *artifact, *drafter); !approved)
          return Error(approved.error());
      }
      hold->spec = std::move(*composed);
      hold->sources.push_back(std::move(*dsource));
    } else {
      hold->spec = std::move(*synthesized);
    }
    if (settings.kv_cache_encoding.value != "f16") {
      if (settings.kv_cache_encoding.value != "q8_0") return Error("unsupported KV encoding");
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
      // Component families use the same paid construction inventory without
      // inventing a NumericalDomain (their serving recipe is checked above).
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
    // The actual stored/composed contracts and selected emission mode bound
    // every mask and index-score plane's signed int32 stride (RE-037).
    auto prompt_bound =
        components::StagePrefillRows(hold->spec, 0, settings.context.value,
                                     settings.prefill_chunk.value, shared && recipe.lid_mask);
    if (!prompt_bound) return Error(prompt_bound.error());
    const auto configured_chunk = *prompt_bound;
    // How the drafter drafts (its mapping's drafting recipe).
    const spec::DraftRecipe* draft = speculate ? drafter_mapping->draft : nullptr;
    if (speculate && draft == nullptr)
      return Error(std::format("the drafter's architecture {} has no drafting recipe",
                               drafter->model().architecture));
    const bool block = draft != nullptr && draft->shape == spec::DraftRecipe::Shape::kBlock;
    const bool mtp = draft != nullptr && draft->shape == spec::DraftRecipe::Shape::kMtp;
    const auto small_rows = draft != nullptr ? draft->small_chunk_rows : 4U;
    const auto small_depth = draft != nullptr ? draft->small_chunk_max_depth : 2U;
    const auto depth =
        mtp ? (configured_chunk >= small_rows ? settings.draft_rows.value
                                              : std::min(settings.draft_rows.value, small_depth))
        : block ? settings.draft_rows.value
                : (draft != nullptr ? draft->depth : 0U);
    // A block drafter's bound and fill token, from the metadata it keeps.
    std::int32_t block_fill = 0;
    if (block) {
      auto most = KeptInteger(*drafter, draft->depth_key);
      if (!most) return Error(most.error());
      auto fill = KeptInteger(*drafter, draft->fill_key);
      if (!fill) return Error(fill.error());
      if (*fill < 0 || *fill > std::numeric_limits<std::int32_t>::max())
        return Error("a block drafter's fill token is outside the vocabulary");
      block_fill = static_cast<std::int32_t>(*fill);
      if (depth == 0 || depth == UINT32_MAX || std::cmp_greater(depth, *most))
        return Error("the resolved draft depth exceeds the bound block drafter");
    }
    // Each stage's state by model::ExtentLayout's per-tensor rule: a tensor
    // of an extent or more on its own extents (its growth crosses extents
    // with every other's), smaller ones packed, a bank kept contiguous.
    static_assert(model::kStateExtent == kExtent, "state is laid out over the paged extent");
    for (std::uint16_t stage = 0; stage < hold->spec.stages.size(); ++stage) {
      auto layout =
          components::StageLayout(hold->spec, stage, settings.context.value, configured_chunk);
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
    spec::PolicyRule composition = *stored_policy;
    // The shared route's own recipe in place of the admitted one.
    const bool forced = shared && recipe.serving_rule != nullptr;
    if (forced) composition = *recipe.serving_rule;
    if (composition.prompts_alone) caps->join_prefill = false;
    if (composition.owners_alone) caps->max_segments = 1;
    if (shared) {
      caps->common_read = true;
      caps->joined_read_align = settings.wave_read_align.value;
    }
    if (joined_score && caps->max_segments < 4)
      return Error("the actual component capability cannot join the scoring cohort");
    // A verify's rows: the anchor and every draft.
    const std::uint32_t verify_rows = speculate ? depth + 1 : 0;
    if (speculate && std::ranges::any_of(hold->layouts, [verify_rows](const auto& layout) {
          return verify_rows > layout.max_rows;
        }))
      return Error("the complete drafter verify exceeds its actual stage row bounds");
    const std::uint32_t read_align = shared ? settings.wave_read_align.value : 256;
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
      if (block) {
        if (shared) {
          const std::uint32_t max_rows =
              caps->max_segment_rows != 0 ? caps->max_segment_rows / owners : verify_rows;
          const std::uint32_t d_rows = std::min(depth, max_rows);
          const std::uint32_t v_rows = std::min(verify_rows, max_rows);
          if (d_rows > 0) add(shape(1, planner::Phase::kDraft, owners, d_rows, d_rows, true));
          if (v_rows > 0) add(shape(0, planner::Phase::kVerify, owners, v_rows, v_rows, false));
        } else {
          // Block proposals produce every head, including short reply tails;
          // every verify includes its anchor. No autoregressive catch-up pass.
          for (std::uint32_t rows = 1; rows <= depth; ++rows)
            add(shape(1, planner::Phase::kDraft, owners, rows, rows, false));
          for (std::uint32_t rows = 1; rows <= verify_rows; ++rows)
            add(shape(0, planner::Phase::kVerify, owners, rows, rows, false));
        }
      } else if (speculate) {
        for (const bool greedy : {false, true}) {
          add(shape(0, planner::Phase::kVerify, owners, verify_rows, verify_rows, greedy));
          add(shape(1, planner::Phase::kDraft, owners, 1, 1, mtp && greedy));
        }
        if (mtp) {
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
      auto slabs = engine::ExpertSlabsOf(a == 0 ? *artifact : *drafter, shared ? 16 : 256);
      if (!slabs) return Error(slabs.error());
      placements.push_back(std::move(*slabs));
    }
    // A table read by rows only where that holds less than the table (its
    // landing and slots for the most lookups a setup shape makes,
    // engine::RowPagingPays); a smaller one stays resident with its group,
    // so no step waits on its reads.
    const bool had_rows = !stored_bindings->rows.empty();
    {
      const auto& rows = stored_bindings->rows;
      std::vector<engine::RunnerConfig::RowTable> paged;
      for (const auto& row : rows) {
        const ja::Artifact& owner =
            hold->spec.stages.at(row.weight.stage).artifact == 0 ? *artifact : *drafter;
        std::uint64_t lookups = 0, table = 0, row_bytes = 0;
        for (const planner::StepShape& s : setup)
          if (s.stage == row.weight.stage)
            lookups = std::max(lookups, std::uint64_t{s.rows()} * row.rows_per_token);
        for (const auto& part : rows) {
          if (part.weight != row.weight) continue;
          table += owner.resources()[part.resource].bytes.value();
          row_bytes += format::RowBytes(part.format, part.part).value_or(0);
        }
        if (engine::RowPagingPays(table, lookups, row_bytes)) paged.push_back(row);
      }
      stored_bindings->rows = std::move(paged);
    }
    std::uint32_t logit_width = 0;
    {
      logit_width = stored_bindings->logit_width;
      for (const auto& [key, place] : hold->place)
        if (place.array)
          hold->pitch[key.first] = placements.at(place.artifact).pitch.at(place.index);
      for (const auto& row : stored_bindings->rows) {
        const auto art_idx = hold->spec.stages.at(row.weight.stage).artifact;
        const ja::Artifact& owner = art_idx == 0 ? *artifact : *drafter;
        if (row.resource >= owner.resources().size()) {
          return Error("a paged row table names a resource outside its artifact");
        }
        placements.at(art_idx).places.at(owner.resources()[row.resource].group) =
            engine::GroupPlace::kNone;
      }
    }
    // The stored target's maps came from its validated component roles
    // above; only a speculating model builds the setup graphs, to place its
    // drafter's weights.
    if (speculate) {
      for (const planner::StepShape& s : setup) {
        auto step =
            components::BuildStep(hold->spec, s.stage, s,
                                  {.tensors = hold->tables.at(s.stage),
                                   .state = &hold->layouts.at(s.stage),
                                   .stages = hold->stages,
                                   .artifacts = hold->tables},
                                  {.narrow = true, .dsv4_lid_mask = shared && recipe.lid_mask});
        if (!step) return Error(step.error());
        for (const components::StepWeight& weight : step->weights) {
          // A stored target's weights are placed above from its validated
          // roles. A target-stage step also binds the drafter's weights (the
          // feature projection a prompt chunk runs when the model speculates),
          // and those have no other source of places.
          if (weight.ref.stage == 0) continue;
          const auto art_idx = hold->spec.stages.at(weight.ref.stage).artifact;
          const ja::Artifact& a = *drafter;
          for (const auto& [name, role] : weight.parts) {
            const auto found =
                weight.ref.expert_array ? a.FindExpertArray(name) : a.FindResource(name);
            if (!found) return Error(std::format("no weight part holds {}", name));
            hold->place[{weight.ref, role}] = {.artifact = static_cast<std::uint16_t>(art_idx),
                                               .array = weight.ref.expert_array,
                                               .index = *found};
            if (weight.ref.expert_array)
              hold->pitch[weight.ref] = placements.at(art_idx).pitch.at(*found);
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
    if (block && std::cmp_greater_equal(block_fill, logit_width))
      return Error("a block drafter's fill token is outside the vocabulary");
    // A layer's routed experts in one slab (the M3 runners' pitch).
    if (solo && had_rows && stored_bindings->rows.empty())
      return Error("the component recipe has no n-gram row table");
    ComponentHold* held = hold.get();
    kernels::ggml::GgmlBackendOptions backend_options{
        .lower = {.device_masks = true,
                  // The runner stages a frozen read's host mask (its cells'
                  // positions before the rows); the device's are made
                  // anyway.
                  .frozen_host_masks = recipe.frozen_host_masks,
                  // Every source's reader index is bounded by what its
                  // construction pays (ServingConstructionBytes).
                  .index_scratch_limit = kServingReaderScratchBytes,
                  .expert_stride = [held](const ir::WeightRef& ref) -> std::uint64_t {
                    const auto at = held->pitch.find(ref);
                    return at == held->pitch.end() ? 0 : at->second;
                  }}};
    // The format ladder's decoders for what GGML does not read: affine
    // tables, and the native importer's split NVFP4 routed experts; else
    // only what a native artifact's NVFP4 needs: the grouped product for
    // scale-atom experts (the importer's relayout), which a stage without
    // the block-scaled grouped GEMM (a W4A16 checkpoint's policy, a device
    // without FP4) decodes on chip, and the rows of a split NVFP4 table.
    bool has_affine = false, has_atoms = false, has_rows = false;
    bool has_exl3 = false;
    for (const auto& table : hold->tables) {
      for (const auto& tensor : table) {
        if (tensor.format.encoding == format::Encoding::kExl3Trellis) {
          has_exl3 = true;
        }
        const bool nvfp4 = tensor.format.encoding == format::Encoding::kNvfp4;
        if (tensor.format.encoding == format::Encoding::kAffine ||
            (tensor.experts && nvfp4 && tensor.format.layout == format::Layout::kSplit))
          has_affine = true;
        if (nvfp4 && tensor.experts && tensor.format.layout == format::Layout::kScaleAtoms)
          has_atoms = true;
        if (nvfp4 && !tensor.experts && tensor.format.layout == format::Layout::kSplit)
          has_rows = true;
      }
    }
    const std::string_view policy_text = options.target_policy
                                             ? std::string_view(*options.target_policy)
                                             : std::string_view(composition.text);
    if (policy_text.find("llmp.decode.") != std::string_view::npos) {
      has_affine = true;
    }
    if (has_exl3) {
      // exllamav3 is locked with `use: test` in third_party/sources.lock.json
      // (tests/sources/check_closure.py), so llmp-runtime does not link
      // llmp_exl3_selected; test harnesses attach kernels::exl3::SelectedSource
      // directly on ModelRunner.
      return Error(
          "EXL3 trellis execution is test-only until exllamav3 is promoted from use:test in "
          "third_party/sources.lock.json");
    }
    if (has_affine) {
      // Resident experts: the grouped products route on the device.
      auto decode_source = kernels::decode::DecodeResidentSelectedSource();
      if (!decode_source) return Error(decode_source.error());
      backend_options.selected = *decode_source;
    } else if (has_atoms || has_rows) {
      std::vector<std::shared_ptr<const kernels::ggml::SelectedSource>> sources;
      if (has_atoms) {
        auto grouped = kernels::decode::ResidentGroupedSelectedSource(format::Encoding::kNvfp4);
        if (!grouped) return Error(grouped.error());
        sources.push_back(std::move(*grouped));
      }
      if (has_rows) {
        auto rows = kernels::decode::RowsSelectedSource(format::Encoding::kNvfp4);
        if (!rows) return Error(rows.error());
        sources.push_back(std::move(*rows));
      }
      backend_options.selected = kernels::decode::CompositeSelectedSource(std::move(sources));
    }
    auto backend = kernels::ggml::GgmlBackend::Create(
        0, node.execution(), node.stream(static_cast<std::uint32_t>(index)), nullptr,
        std::move(backend_options));
    if (!backend) return Error(backend.error());
    hold->backend = std::move(*backend);
    const providers::DeviceFacts facts =
        providers::QueryDeviceFacts(0).value_or(providers::DeviceFacts{});
    const std::uint64_t cublas_bytes =
        kernels::ggml::CublasHandle::UpstreamWorkspace(static_cast<int>(facts.architecture))
            .value();
    engine::DecodingMode mode;
    if (speculate) {
      mode = {
          .kind = engine::DecodingMode::Kind::kSpeculative,
          .drafters = {{.stage = 1,
                        .shape = block ? engine::DrafterSpec::Shape::kBlock
                                       : engine::DrafterSpec::Shape::kAutoregressive,
                        .max_depth = depth,
                        .shared_wave_depth =
                            mtp ? std::min(settings.shared_wave_depth.value, depth) : 0,
                        .draft_wave_max = mtp ? settings.draft_wave_max.value : 0,
                        .catch_up = mtp,
                        .fill = block_fill,
                        .borrows_target = !mtp && !block,
                        .anchored = !mtp && !block,
                        .features = !block,
                        .adaptive = mtp && depth >= 3,
                        .depth_cost = mtp ? settings.depth_cost_ratio.value
                                          : (draft != nullptr ? draft->default_depth_cost : 1.16),
                        .device_tokens = shared,
                        .draft_before_limit = shared,
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

    // The target's policy: a harness's matched recipe, the shared route's
    // own, or the rule admission chose (AdmittedPolicy); where it chose none
    // (an approval's kNone, or no rule holds), the engine's default. Its
    // step source never chooses a rule again from its mapping.
    planner::Policy target_policy;
    if (!composition.text.empty() || options.target_policy || forced) {
      auto parsed =
          planner::ParsePolicy(options.target_policy ? *options.target_policy : composition.text);
      if (!parsed) return Error(std::format("the target's policy: {}", parsed.error()));
      target_policy = std::move(*parsed);
    }
    std::optional<planner::Policy> drafter_policy;
    if (options.drafter_policy) {
      auto parsed = planner::ParsePolicy(*options.drafter_policy);
      if (!parsed) return Error(std::format("the drafter's policy: {}", parsed.error()));
      drafter_policy = std::move(*parsed);
    }
    if (!drafter_policy && forced && speculate) {
      // The forced route retains its shared fusion rules for the drafter,
      // but each stage's independent product policies come from its mapping.
      auto text = spec::DefaultPolicy(*drafter_mapping, hold->drafter_values,
                                      {.context = settings.context.value, .slots = slots});
      if (!text) return Error(text.error());
      auto own = text->empty() ? std::expected<planner::Policy, std::string>(planner::Policy{})
                               : planner::ParsePolicy(*text);
      if (!own) return Error(own.error());
      drafter_policy = target_policy;
      drafter_policy->products = own->products;
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
           .values = stage == 0 ? std::span<const spec::SourceValue>{}
                                : std::span<const spec::SourceValue>(hold->drafter_values),
           .slots = slots,
           // A drafter stage: the diagnostic's policy, else a forced
           // shared-route recipe's (the target's), else its mapping's.
           .policy = stage == 0       ? std::optional(target_policy)
                     : drafter_policy ? drafter_policy
                     : forced         ? std::optional(target_policy)
                                      : std::nullopt,
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
    parts.config.split_expert_parts = has_affine || has_atoms;
    parts.config.row_tables = std::move(stored_bindings->rows);
    if (options.capture_first_verify)
      parts.config.capture_first_phases = 1U << static_cast<unsigned>(planner::Phase::kVerify);
    if (shared)
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
    if (shared && speculate && recipe.adaptive_waves) {
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
    // Concurrent lanes (a joined wave's owner regions on streams of their
    // own) for the serving plans measured to gain from them (its recipe's).
    // Lanes take their pools from the end of the model's own prefix of the
    // shared pool, so the stream's own scratch shrinks by kMaxLanes of
    // them: a model that runs lanes sets them apart beside what its setup
    // shapes measured (the runner's pool need), and every other model's
    // regions run on its stream.
    const bool lanes = shared && recipe.lanes;
    if (lanes) parts.config.launch_pool_reserve = kernels::ggml::kMaxLanes * kLaneScratchBytes;
    parts.config.launch = [&node, held, lanes](std::uint64_t pool_bytes)
        -> std::expected<std::unique_ptr<kernels::StepLaunch>, std::string> {
      if (pool_bytes == 0 || pool_bytes > node.pool().bytes)
        return Error("a launch's pool lies outside the node's workspace");
      auto made =
          held->backend->Launch({.base = node.pool().base, .size = base::Bytes(pool_bytes)});
      if (!made || !lanes) return made;
      auto* ggml_launch = static_cast<kernels::ggml::GgmlStepLaunch*>(made->get());
      // Each lane its whole scratch: the model's pool holds the reserve for
      // them beside what its steps measured (launch_pool_reserve).
      if (auto r = ggml_launch->context().ConfigureLanes(kernels::ggml::kMaxLanes,
                                                         base::Bytes(kLaneScratchBytes));
          !r) {
        return Error(std::format("ConfigureLanes: {}", r.error().detail));
      }
      return made;
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
    parts.device_verify = shared;
    parts.branch_depth = recipe.branch_depth;
    parts.prefill = {.chunk_rows = chunk,
                     .lookahead = composition.prefill_lookahead,
                     .capture_ahead = composition.capture_ahead,
                     .prepare_ahead = true,
                     .state_only = true};
    if (shared) {
      parts.prefill.lookahead = options.prefill_lookahead ? 2 : 0;
      parts.prefill.prepare_ahead = options.prepare_state;
      parts.prefill.state_only = options.state_only_prefill.value_or(!speculate);
    }
    parts.architecture = settings.architecture;
    parts.chat = {
        .thinking = recipe.chat_thinking,
        .tools = recipe.chat_tools,
        .stops = std::vector<std::string>(recipe.chat_stops.begin(), recipe.chat_stops.end()),
        .default_thinking = recipe.default_thinking};
    parts.keep = std::move(hold);
    return parts;
  };
}

}  // namespace llmp::runtime
