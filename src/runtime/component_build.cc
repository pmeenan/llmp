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
#include <mutex>
#include <numeric>
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
#include "engine/runner_resources.h"
#include "format/tensor_format.h"
#include "planner/construction.h"
#include "planner/plan.h"
#include "planner/policy.h"
#include "planner/step_shape.h"
#include "platform/host_probe.h"
#include "providers/device_runtime.h"
#include "runtime/component_admission.h"
#include "runtime/component_bindings.h"
#include "runtime/llm_policy.h"
#include "runtime/memory_guard.h"
#include "runtime/model_settings.h"
#include "runtime/prefill.h"
#include "runtime/serving.h"
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

using ConstructionCharge = serving_support::Paid;

// PlanGraph owns one temporary reader index per synchronous lowering; current
// and future retained programs do not retain that index. This named category
// is both paid by automatic construction and enforced, for every source,
// before index allocation (its inventory bound: 1,024 bytes an occurrence).
// The largest measured (below) is 26.9 MB, DeepSeek V4's.
constexpr std::uint64_t kServingReaderScratchBytes = 128ULL << 20;

// Each of a launch's concurrent lanes' scratch at most (the backend's
// ConfigureLanes), carved from the end of the node's pool.
constexpr std::uint64_t kLaneScratchBytes = 4ULL * 1024ULL * 1024ULL;

// Called under the Server's parser loan after source/schema validation, and
// before any BuildStep/lowering. Graph maxima are the existing IR contract;
// native lowering admits its own actual operation and saved-root inventory.
struct ServingConstruction {
  std::uint64_t compiler = 0, retained = 0;
};
std::expected<ServingConstruction, std::string> ServingConstructionBytes(
    const spec::ModelSpec& model, const ja::Artifact& target, const ja::Artifact* draft) {
  constexpr std::uint64_t nodes = ir::Graph::kMaxNodes;
  constexpr std::uint64_t tensors = ir::Graph::kMaxTensors;
  // One serialized IR compiler loan, shared across aliases: graph/rewrite
  // inventories and the independently bounded reader index. Native descriptor
  // construction is admitted from the actual immutable LowerInput by the
  // backend funder; retained programs transfer to PlanAccount before release.
  // These coefficients are conservative bounds, not measured memory peaks.
  std::uint64_t bytes = 0;
  const auto add = [&](std::uint64_t count, std::uint64_t unit) {
    std::uint64_t part = 0;
    return !__builtin_mul_overflow(count, unit, &part) &&
           !__builtin_add_overflow(bytes, part, &bytes);
  };
  const auto numerical = planner::NumericalConstructionBytes(nodes);
  if (!numerical || !add(nodes, planner::kConstructionGraphNodeBytes) ||
      !add(tensors, planner::kConstructionGraphTensorBytes) ||
      !add(1, kServingReaderScratchBytes) || !add(1, *numerical))
    return Error("component IR compiler payment overflows");
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
  if (!source(target) || (draft && !source(*draft)) ||
      !add(model.stages.size(), sizeof(model::Batching)) || !add(1, 64))
    return Error("component retained source inventory overflows");
  if (!add(1, kernels::kRegistrationPlanReportMetadataBytes))
    return Error("registration plan-report metadata inventory overflows");
  result.retained = bytes;
  return result;
}

struct ComponentPreparation {
  // Factory copies share one successful preparation. Serialize construction
  // locally (including its mutable payments); never hold a global runtime lock.
  std::mutex mutex;
  std::shared_ptr<const OpenedModelArtifacts> model;
  bool caller_prepared = false;
  std::optional<std::uint32_t> prepared_draft_vocab;
  // These catalog extents are node-lifetime high-water reservations. Reuse
  // them across retries/copies; each live build owns a separate local charge.
  std::shared_ptr<engine::ExecutionMetadataAllowance> metadata;
  std::shared_ptr<engine::ExecutionMetadataAllowance> stored_metadata;
  std::shared_ptr<engine::ExecutionMetadataAllowance> automatic_construction;
};

struct ComponentHold {
  // Shared immutable preparation and per-build working storage outlive every
  // source/spec copy below, including failures and independently held runners.
  std::shared_ptr<const OpenedModelArtifacts> prepared;
  std::unique_ptr<StartupArtifactPayment> structure_payment;
  // Refund this build's charge only after its graph/binding/backend owners die.
  std::shared_ptr<ConstructionCharge> automatic_construction;
  std::shared_ptr<engine::SharedCompilerAllowance> compiler;
  std::uint64_t compiler_bytes = 0;
  std::shared_ptr<ConstructionCharge> kv_encoding_charge;
  std::shared_ptr<ConstructionCharge> setup_charge;
  std::unique_ptr<StoredBindingPayment> stored_payment;
  spec::ModelSpec spec;
  // Actual admitted semantic identities, captured before composition, draft
  // vocabulary narrowing and per-alias state representation transforms.
  std::array<base::Sha256Digest, spec::kMaxStages> numerical_semantics{};
  std::vector<spec::Source> sources;  // by artifact: the target's, the drafter's
  std::vector<std::span<const components::BoundTensor>> tables;
  std::vector<model::StateLayoutDerived> layouts;  // by stage
  std::vector<model::Batching> batching;           // canonical component limits, by stage
  std::vector<const model::StateLayoutDerived*> stages;
  std::vector<spec::SourceValue> drafter_values;
  std::shared_ptr<const std::vector<std::int32_t>> draft_head_ids;
  std::uint32_t draft_head_rows = 0;
  // Each weight's place: its holding stage's artifact, whether it is
  // an expert array, and its resource's or array's index.
  using Place = ComponentWeightPlace;
  ComponentWeightPlaces place;
  std::map<ir::WeightRef, std::uint64_t> pitch;
  std::unique_ptr<kernels::DeviceBackend> backend;
};
static_assert(sizeof(ComponentHold) <= 4096);  // StoredBindingBytes fixed owner inventory.

// Owned preparation phase. The actual stored/composed contracts supply every
// row, state and batching limit; no serving family supplies capabilities.
Status PrepareComponentState(ComponentHold& hold, std::uint32_t context, std::uint32_t max_rows) {
  hold.layouts.reserve(hold.spec.stages.size());
  hold.batching.reserve(hold.spec.stages.size());
  hold.stages.reserve(hold.spec.stages.size());
  for (std::uint16_t stage = 0; stage < hold.spec.stages.size(); ++stage) {
    auto layout = components::StageLayout(hold.spec, stage, context, max_rows);
    if (!layout) return Error(layout.error());
    auto batching = components::StageBatching(hold.spec, stage);
    if (!batching) return Error(batching.error());
    hold.layouts.push_back(std::move(*layout));
    hold.batching.push_back(*batching);
  }
  for (const auto& layout : hold.layouts) hold.stages.push_back(&layout);
  return {};
}

struct PreparedPlacement {
  std::vector<engine::LayerSlabs> slabs;
  std::uint32_t logit_width = 0;
};

std::expected<PreparedPlacement, std::string> PrepareComponentPlacement(
    ComponentHold& hold, const ja::Artifact& artifact, const ja::Artifact* draft_artifact,
    StoredComponentBindings& bindings, std::span<const planner::StepShape> setup,
    bool indexed_masks) {
  // Each weight's place, by what the steps bind (the head's included;
  // a state-only prompt chunk binds a subset), and a head row's width:
  // its logits' (the head's vocabulary).
  // Native expert parts retain their 16-byte bases in each actual encoded
  // slab; this is byte alignment, independent of owner/expert counts.
  constexpr std::uint64_t kExpertAlignment = 16;
  std::vector<engine::LayerSlabs> placements;
  for (std::size_t a = 0; a < hold.sources.size(); ++a) {
    auto slabs = engine::ExpertSlabsOf(a == 0 ? artifact : *draft_artifact, kExpertAlignment);
    if (!slabs) return Error(slabs.error());
    placements.push_back(std::move(*slabs));
  }
  // A table read by rows only where that holds less than the table (its
  // landing and slots for the most lookups a setup shape makes,
  // engine::RowPagingPays); a smaller one stays resident with its group,
  // so no step waits on its reads.
  {
    const auto& rows = bindings.rows;
    std::vector<engine::RunnerConfig::RowTable> paged;
    for (const auto& row : rows) {
      const ja::Artifact& owner =
          hold.spec.stages.at(row.weight.stage).artifact == 0 ? artifact : *draft_artifact;
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
    bindings.rows = std::move(paged);
  }
  std::uint32_t logit_width = 0;
  {
    logit_width = bindings.logit_width;
    for (const auto& [key, place] : hold.place)
      if (place.array) hold.pitch[key.first] = placements.at(place.artifact).pitch.at(place.index);
    for (const auto& row : bindings.rows) {
      const auto art_idx = hold.spec.stages.at(row.weight.stage).artifact;
      const ja::Artifact& owner = art_idx == 0 ? artifact : *draft_artifact;
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
  if (draft_artifact != nullptr) {
    for (const planner::StepShape& s : setup) {
      auto step = components::BuildStep(hold.spec, s.stage, s,
                                        {.tensors = hold.tables.at(s.stage),
                                         .state = &hold.layouts.at(s.stage),
                                         .stages = hold.stages,
                                         .artifacts = hold.tables},
                                        {.narrow = true, .rounded_index_mask = indexed_masks});
      if (!step) return Error(step.error());
      for (const components::StepWeight& weight : step->weights) {
        // A stored target's weights are placed above from its validated
        // roles. A target-stage step also binds the drafter's weights (the
        // feature projection a prompt chunk runs when the model speculates),
        // and those have no other source of places.
        if (weight.ref.stage == 0) continue;
        const auto art_idx = hold.spec.stages.at(weight.ref.stage).artifact;
        const ja::Artifact& a = *draft_artifact;
        for (const auto& [name, role] : weight.parts) {
          const auto found =
              weight.ref.expert_array ? a.FindExpertArray(name) : a.FindResource(name);
          if (!found) return Error(std::format("no weight part holds {}", name));
          hold.place[{weight.ref, role}] = {.artifact = static_cast<std::uint16_t>(art_idx),
                                            .array = weight.ref.expert_array,
                                            .index = *found};
          if (weight.ref.expert_array)
            hold.pitch[weight.ref] = placements.at(art_idx).pitch.at(*found);
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
  return PreparedPlacement{.slabs = std::move(placements), .logit_width = logit_width};
}

std::expected<std::unique_ptr<engine::StepSource>, std::string> SealComponentSources(
    ComponentHold& hold, const ComponentBuildOptions& options, const spec::PolicyRule& composition,
    bool forced, bool speculate, const spec::Mapping* drafter_mapping,
    const base::Sha256Digest& bound, std::uint32_t slots, std::uint32_t context,
    const std::shared_ptr<const engine::NumericalDomain>& execution_domain,
    const ja::Artifact& artifact, const ja::Artifact* drafter) {
  // The target's policy: a harness's matched recipe or the rule admission
  // chose (AdmittedPolicy); where it chose none
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
  const spec::PolicyContext drafter_policy_context =
      hold.sources.size() > 1
          ? spec::PolicyContext{hold.sources[1].architecture, hold.sources[1].container}
          : spec::PolicyContext{};
  if (!drafter_policy && forced && speculate) {
    // The forced route retains its shared fusion rules for the drafter,
    // but each stage's independent product policies come from its mapping.
    auto text = drafter_mapping ? spec::DefaultPolicy(*drafter_mapping, hold.drafter_values,
                                                      {.context = context, .slots = slots},
                                                      drafter_policy_context)
                                : std::expected<std::string_view, std::string>(std::string_view{});
    if (!text) return Error(text.error());
    auto own = text->empty() ? std::expected<planner::Policy, std::string>(planner::Policy{})
                             : planner::ParsePolicy(*text);
    if (!own) return Error(own.error());
    drafter_policy = target_policy;
    drafter_policy->products = own->products;
  }
  const auto backend = hold.backend->Backend();
  std::vector<std::unique_ptr<engine::ComponentStepSource>> sources;
  for (std::uint16_t stage = 0; stage < hold.spec.stages.size(); ++stage) {
    auto stage_policy = stage == 0       ? std::optional(target_policy)
                        : drafter_policy ? drafter_policy
                        : forced         ? std::optional(target_policy)
                                         : std::nullopt;
    if (!stage_policy) {
      auto text =
          drafter_mapping
              ? spec::DefaultPolicy(*drafter_mapping, hold.drafter_values,
                                    {.context = context, .slots = slots}, drafter_policy_context)
              : std::expected<std::string_view, std::string>(std::string_view{});
      if (!text) return Error(text.error());
      auto parsed = text->empty() ? std::expected<planner::Policy, std::string>(planner::Policy{})
                                  : planner::ParsePolicy(*text);
      if (!parsed) return Error(parsed.error());
      stage_policy = std::move(*parsed);
    }
    const auto artifact_index = hold.spec.stages[stage].artifact;
    const auto& admitted_artifact = artifact_index == 0 ? artifact : *drafter;
    auto numerical = AdmittedNumerics(
        hold.spec, hold.sources.at(artifact_index), admitted_artifact.sources(),
        FindApproval(admitted_artifact.model().architecture, admitted_artifact.sources()),
        hold.numerical_semantics[stage], {.context = context, .slots = slots}, stage, backend);
    if (!numerical) return Error(numerical.error());
    stage_policy->numerical = std::move(*numerical);
    auto made = engine::ComponentStepSource::Create(
        {.spec = &hold.spec,
         .stage = stage,
         .bindings = {.tensors = hold.tables.at(hold.spec.stages.at(stage).artifact),
                      .state = &hold.layouts.at(stage),
                      .stages = hold.stages,
                      .artifacts = hold.tables},
         .bound_digest = bound,
         .mapping = stage == 0 ? nullptr : drafter_mapping,
         .policy_context = {hold.sources.at(artifact_index).architecture,
                            hold.sources.at(artifact_index).container},
         .values = stage == 0 ? std::span<const spec::SourceValue>{}
                              : std::span<const spec::SourceValue>(hold.drafter_values),
         .slots = slots,
         // A drafter stage: the diagnostic's policy, else a forced
         // admitted recipe's (the target's), else its mapping's.
         .policy = std::move(stage_policy),
         .backend = backend,
         .execution_domain = execution_domain,
         // Actual programs report capture safety; eager-only implementations
         // remain eligible without disabling capture of ordinary programs.
         .require_capture = false,
         .validate_program = options.validate_program,
         .compiler = hold.compiler,
         .compiler_bytes = hold.compiler_bytes});
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
  return step_source;
}

struct PreparedServing {
  std::shared_ptr<ComponentHold> hold;
  std::unique_ptr<engine::StepSource> source;
  std::vector<engine::LayerSlabs> placements;
  std::vector<engine::RunnerConfig::RowTable> rows;
  std::vector<planner::StepShape> setup;
  engine::DecodingMode mode;
  LlmServingPolicy policy;
  base::Sha256Digest bound;
  std::uint32_t logit_width = 0, verify_rows = 0, branch_depth = 0;
  std::shared_ptr<const ja::Artifact> target_artifact{};
};

std::expected<LlmBuildParts, std::string> AssembleComponentParts(
    engine::PagedNode& node, const fs::path& store, std::string_view id,
    std::string_view drafter_id, [[maybe_unused]] int index, const ModelSettings& settings,
    const spec::ServingRecipe& recipe, const ComponentBuildOptions& options,
    const std::shared_ptr<engine::ExecutionMetadataAllowance>& metadata, PreparedServing built) {
  auto* held = built.hold.get();
  const auto& target = held->layouts.front();
  const auto slots = built.policy.owners();
  const auto read_align = built.policy.read_align();
  const auto chunk = target.max_rows;
  const bool speculate = built.mode.kind == engine::DecodingMode::Kind::kSpeculative;
  auto weights = std::make_unique<engine::PagedWeights>();
  if (auto opened = weights->Open(built.target_artifact, store / id); !opened)
    return Error(opened.error());
  engine::PagedWeights* paged = weights.get();
  engine::PagedWeights* drafter_paged = nullptr;
  LlmBuildParts parts;
  parts.execution_metadata = metadata;
  if (built.hold->automatic_construction) {
    // Report this build's funded footprint, never the shared pool's aggregate.
    parts.startup_host_bytes = built.hold->automatic_construction->bytes;
    parts.startup_cap = options.automatic_startup_cap;
  }
  parts.config.slots = {.model = settings.name,
                        .slots = slots,
                        .stages = {{.name = "target", .layout = target, .read_align = 0}},
                        .spec = built.bound,
                        .read_align = read_align,
                        .spill_place = {},
                        .spill_directory = {},
                        .invalidate = {},
                        .drop_plans = {}};
  parts.config.weights.push_back({.weights = std::move(weights),
                                  .places = std::move(built.placements[0].places),
                                  .slabs = std::move(built.placements[0].slabs)});
  // Always expose the checked physical-part callback. Kernel candidates,
  // rather than a model-table scan, decide which representation they read.
  parts.config.split_expert_parts = true;
  parts.config.row_tables = std::move(built.rows);
  if (options.capture_first_verify)
    parts.config.capture_first_phases = 1U << static_cast<unsigned>(planner::Phase::kVerify);
  parts.config.automatic_capture_phases &= ~(1U << static_cast<unsigned>(planner::Phase::kPrefill));
  if (speculate) {
    parts.config.slots.stages.push_back(
        {.name = built.hold->spec.stages.at(1).name, .layout = built.hold->layouts.at(1)});
    auto dweights = std::make_unique<engine::PagedWeights>();
    if (auto opened = dweights->Open(
            std::shared_ptr<const ja::Artifact>(held->prepared, &*held->prepared->drafter),
            store / drafter_id);
        !opened)
      return Error(opened.error());
    drafter_paged = dweights.get();
    parts.config.weights.push_back({.weights = std::move(dweights),
                                    .places = std::move(built.placements[1].places),
                                    .slabs = std::move(built.placements[1].slabs)});
    parts.config.max_verify_rows = built.verify_rows;
    parts.mode = std::move(built.mode);
  }
  if (speculate) {
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
  parts.config.setup_shapes = std::move(built.setup);
  parts.config.kernel_memory = engine::BackendKernelMemory(*held->backend, settings.name);
  parts.config.prepare =
      [held](std::span<const std::uint64_t> bases) -> std::expected<void, std::string> {
    return held->backend->BindWorkspaces(bases);
  };
  parts.config.retire = [held] { held->backend->ReleaseWorkspaces(); };
  // The backend's immutable future launch configuration also governs
  // measurement. Its lane pools are funded beside measured stream scratch.
  parts.config.launch_pool_reserve = held->backend->lane_pool_reserve();
  parts.config.launch = [&node, held](std::uint64_t pool_bytes)
      -> std::expected<std::unique_ptr<kernels::StepLaunch>, std::string> {
    if (pool_bytes == 0 || pool_bytes > node.pool().bytes)
      return Error("a launch's pool lies outside the node's workspace");
    return held->backend->Launch({.base = node.pool().base, .size = base::Bytes(pool_bytes)});
  };
  const std::uint64_t weight_reservation_alignment = node.memory().ReservationAlignment().value();
  const auto resolve_weight = [held, paged, drafter_paged, weight_reservation_alignment](
                                  const ir::WeightRef& ref, format::PartRole role)
      -> std::optional<std::pair<std::uint64_t, std::optional<kernels::BoundedAddress>>> {
    const auto at = held->place.find({ref, role});
    if (at == held->place.end()) return std::nullopt;
    const ComponentHold::Place& p = at->second;
    const auto* owner = p.artifact == 0 ? paged : drafter_paged;
    if (owner == nullptr) return std::nullopt;
    if (p.array) {
      const std::uint64_t address = owner->array_address(p.index);
      return address == 0
                 ? std::nullopt
                 : std::optional(std::pair{address, std::optional<kernels::BoundedAddress>{}});
    }
    const auto resources = owner->artifact().resources();
    if (p.index >= resources.size()) return std::nullopt;
    const ja::Resource& resource = resources[p.index];
    if (resource.part != role) return std::nullopt;
    const std::uint64_t address = owner->resource_address(p.index);
    if (address == 0) return std::nullopt;
    std::uint64_t end = 0;
    if (resource.readable.value() == 0 || weight_reservation_alignment == 0 ||
        __builtin_add_overflow(address, resource.readable.value(), &end))
      return std::pair{address, std::optional<kernels::BoundedAddress>{}};
    const std::uint64_t alignment = std::gcd(weight_reservation_alignment, address);
    if (alignment == 0) return std::pair{address, std::optional<kernels::BoundedAddress>{}};
    return std::pair{
        address, std::optional(kernels::BoundedAddress{
                     .base = address, .bytes = resource.readable.value(), .alignment = alignment})};
  };
  parts.config.weight_address = [resolve_weight](const ir::WeightRef& ref,
                                                 format::PartRole role) -> std::uint64_t {
    const auto resolved = resolve_weight(ref, role);
    return resolved ? resolved->first : 0;
  };
  parts.config.weight_range = [resolve_weight](
                                  const ir::WeightRef& ref,
                                  format::PartRole role) -> std::optional<kernels::BoundedAddress> {
    const auto resolved = resolve_weight(ref, role);
    return resolved ? resolved->second : std::nullopt;
  };
  parts.source = std::move(built.source);
  parts.caps = built.policy.batching();
  parts.logit_width = built.logit_width;
  parts.device_greedy = built.policy.device_tokens();
  parts.device_verify = built.policy.device_tokens();
  parts.branch_depth = built.branch_depth;
  parts.prefill = {.chunk_rows = chunk,
                   .lookahead = built.policy.lookahead(),
                   .capture_ahead = built.policy.capture_ahead(),
                   .prepare_ahead = built.policy.prepare_ahead(),
                   .state_only = built.policy.state_only()};
  parts.architecture = settings.architecture;
  parts.chat = {
      .thinking = recipe.chat_thinking,
      .tools = recipe.chat_tools,
      .stops = std::vector<std::string>(recipe.chat_stops.begin(), recipe.chat_stops.end()),
      .default_thinking = recipe.default_thinking};
  parts.target_artifact = std::move(built.target_artifact);
  parts.keep = std::move(built.hold);
  return parts;
}

}  // namespace

LlmBuild ComponentLlmBuild(engine::PagedNode& node, fs::path store, const config::ModelEntry& entry,
                           int index, ComponentBuildOptions options) {
  auto preparation = std::make_shared<ComponentPreparation>();
  preparation->caller_prepared = options.prepared != nullptr;
  preparation->model = std::move(options.prepared);
  return [&node, store = std::move(store), entry, preparation = std::move(preparation),
          id = entry.artifact.value_or(""), drafter_id = entry.drafter.value_or(""), index,
          options = std::move(options)](
             const ModelSettings& settings) mutable -> std::expected<LlmBuildParts, std::string> {
    std::lock_guard construction_lock(preparation->mutex);
    auto& metadata = preparation->metadata;
    auto& stored_metadata = preparation->stored_metadata;
    // The architecture's serving recipe (its mapping's data or the default
    // target LLM recipe, spec/mapping.h).
    const auto* recipe_ptr = RecipeOf(settings.architecture, settings.container);
    const auto& recipe = recipe_ptr ? *recipe_ptr : spec::kDefaultServingRecipe;
    const auto metadata_cap =
        [&node, startup_cap =
                    options.automatic_startup_cap]() -> std::expected<std::uint64_t, std::string> {
      if (startup_cap != 0) return startup_cap;
      if (!node.open_for_setup()) return node.budget().value();
      // A stopped diagnostic has no execution budget yet. Physical headroom
      // bounds admission; Start still enforces its eventual configured cap.
      const auto available = platform::AvailableMemoryBytes().value_or(0);
      const auto occupied = node.catalog().OccupancyOf(node.domain()).Total().value();
      std::uint64_t cap = 0;
      if (available <= kUncountedMargin ||
          __builtin_add_overflow(occupied, available - kUncountedMargin, &cap))
        return Error("execution metadata lacks bounded physical headroom");
      return cap;
    };
    const auto charge_metadata =
        [&node, metadata_cap, limit = options.execution_metadata_cap_bytes](
            const std::shared_ptr<engine::ExecutionMetadataAllowance>& paid, std::uint64_t bytes) {
          if (!paid) return false;
          const auto overhead = engine::ExecutionMetadataAllowance::OverheadBytes();
          if (limit != 0 && (limit < overhead || paid->used() > limit - overhead ||
                             bytes > limit - overhead - paid->used()))
            return false;
          auto cap = metadata_cap();
          return cap && serving_support::ChargeExecutionMetadata(paid, bytes, &node, *cap);
        };
    const auto admit_metadata = [&node, metadata_cap,
                                 limit = options.execution_metadata_cap_bytes]()
        -> std::expected<std::shared_ptr<engine::ExecutionMetadataAllowance>, std::string> {
      const auto bytes = engine::ExecutionMetadataAllowance::OverheadBytes() + 1;
      if (limit != 0 && bytes > limit) return Error("execution metadata cap cannot fund its owner");
      auto cap = metadata_cap();
      if (!cap) return Error(cap.error());
      return serving_support::AdmitExecutionMetadata(node, bytes, *cap);
    };
    // All LLMs use the same declared backend and admitted source route.
    // Configuration is validated against actual artifacts/layouts below;
    // it cannot grant numerical transformations or execution capability.
    if (settings.context.value == 0 || settings.prefill_chunk.value == 0 ||
        settings.max_slots.value == 0 || settings.max_slots.value > Llm::kMaxBranches)
      return Error("component serving needs positive context/chunk and a bounded owner count");
    if (settings.speculation.value && drafter_id.empty())
      return Error("component speculation needs an owned drafter artifact");
    auto prepared = preparation->model;
    if (prepared && !preparation->caller_prepared && !drafter_id.empty() &&
        preparation->prepared_draft_vocab != settings.draft_vocab.value) {
      prepared.reset();
    }
    if (!prepared) {
      // Diagnostics take the same path as Server::Start. A failed admission or
      // funding attempt is not cached, so a later resource/I/O retry is valid.
      // Diagnostic callers have no Server bootstrap. Pay the bounded artifact
      // readers before Open; retained preparation receives its own payment
      // before this transient parser loan is released.
      auto parser_bytes = StartupBootstrapBytes(1);
      auto parser_cap = metadata_cap();
      if (!parser_bytes) return Error(parser_bytes.error());
      if (!parser_cap) return Error(parser_cap.error());
      auto parser =
          StartupArtifactPayment::Admit(node.catalog(), node.domain(), *parser_bytes, *parser_cap,
                                        platform::AvailableMemoryBytes().value_or(0));
      if (!parser) return Error(parser.error());
      auto owner = std::make_shared<OpenedModelArtifacts>();
      auto configured = entry;
      configured.overrides["kv_cache_encoding"] = settings.kv_cache_encoding.value;
      configured.overrides["draft_vocab"] = std::int64_t{settings.draft_vocab.value};
      const StartupArtifactFunder funder = [&](const ja::Artifact& target,
                                               const ja::Artifact* drafter) {
        auto bytes = StartupModelBytes(target, drafter);
        auto cap = metadata_cap();
        if (!bytes)
          return std::expected<std::unique_ptr<StartupArtifactPayment>, std::string>(
              Error(bytes.error()));
        if (!cap)
          return std::expected<std::unique_ptr<StartupArtifactPayment>, std::string>(
              Error(cap.error()));
        return StartupArtifactPayment::Admit(node.catalog(), node.domain(), *bytes, *cap,
                                             platform::AvailableMemoryBytes().value_or(0));
      };
      auto facts = ReadArtifactFacts(configured, store, owner.get(), funder);
      if (!facts) return Error(facts.error());
      if (settings.speculation.value && !owner->composition)
        return Error(facts->composed_spec_error);
      // Plain serving may tolerate a drafter composition refusal. Use that
      // target for this build, but do not memoize a failed optional preparation.
      // A later speculative attempt can retry transient selected-head I/O.
      prepared = owner;
      if (!owner->drafter || owner->composition) {
        preparation->model = std::move(owner);
        preparation->prepared_draft_vocab = settings.draft_vocab.value;
      }
    }
    if (!prepared->payment || !prepared->target || prepared->target->id() != id ||
        !prepared->target_admission ||
        (!drafter_id.empty() && (!prepared->drafter || prepared->drafter->id() != drafter_id ||
                                 !prepared->drafter_admission)))
      return Error("prepared model differs from its configured artifacts/admissions");
    const auto artifact = std::shared_ptr<const ja::Artifact>(prepared, &*prepared->target);
    const auto* pre_admitted = &*prepared->target_admission;
    auto structure_bytes = StartupStructureBytes(
        *artifact, settings.speculation.value && prepared->drafter ? &*prepared->drafter : nullptr);
    if (!structure_bytes) return Error(structure_bytes.error());
    auto structure_cap = metadata_cap();
    if (!structure_cap) return Error(structure_cap.error());
    auto structure_payment =
        StartupArtifactPayment::Admit(node.catalog(), node.domain(), *structure_bytes,
                                      *structure_cap, platform::AvailableMemoryBytes().value_or(0));
    if (!structure_payment) return Error(structure_payment.error());

    if (settings.architecture != artifact->model().architecture ||
        settings.container != artifact->model().container)
      return Error("resolved architecture/container differs from the opened artifact");
    // The artifact's prerequisites are admitted before its spec, component
    // parameters or role maps are parsed or copied (a v1 artifact's alone:
    // D-118); those are admitted once in preparation. Each independent
    // build funds the mutable bindings it retains.
    std::unique_ptr<StoredBindingPayment> stored_payment;
    {
      auto bytes = StoredBindingBytes(*artifact);
      if (!bytes) return Error(bytes.error());
      if (!stored_metadata) {
        auto paid = admit_metadata();
        if (!paid) return Error(paid.error());
        stored_metadata = std::move(*paid);
      }
      // Reserve capacity for this build beside every still-live binding owner.
      // Create takes the charge itself; the local mutex excludes another
      // constructor between this capacity admission and that transfer.
      if (!charge_metadata(stored_metadata, *bytes))
        return Error("stored binding metadata allowance is insufficient");
      stored_metadata->Release(*bytes);
      auto paid = StoredBindingPayment::Create(*artifact, stored_metadata);
      if (!paid) return Error(paid.error());
      stored_payment = std::move(*paid);
    }
    auto hold = std::make_shared<ComponentHold>();
    hold->prepared = prepared;
    hold->structure_payment = std::move(*structure_payment);
    hold->stored_payment = std::move(stored_payment);
    std::optional<StoredComponentBindings> stored_bindings;
    std::optional<spec::PolicyRule> stored_policy;
    std::expected<spec::ModelSpec, std::string> synthesized;
    {
      std::optional<std::pair<spec::ModelSpec, spec::Source>> pre_read;
      pre_read.emplace(pre_admitted->model, pre_admitted->source);
      auto bindings =
          ReadStoredComponents(*artifact, std::move(hold->stored_payment), std::move(pre_read));
      if (!bindings) return Error(bindings.error());
      stored_bindings.emplace(std::move(*bindings));
      if (settings.context.value == 0 ||
          settings.context.value > stored_bindings->model.stages.front().context)
        return Error("resolved context exceeds the admitted stored spec");
      // Admission belongs to the immutable preparation; execution policy and
      // physical placements still validate this build's actual envelope.
      hold->numerical_semantics[0] = stored_bindings->model.Digest();
      auto policy =
          AdmittedPolicy(stored_bindings->model, stored_bindings->source, pre_admitted->approval,
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
    const ja::Artifact* drafter = nullptr;
    const spec::Mapping* drafter_mapping = nullptr;
    if (speculate) {
      drafter = &*prepared->drafter;
      drafter_mapping = ComposedDrafterMapping(*drafter);
      if (!prepared->composition)
        return Error("prepared model has no admitted drafter composition");
      if (auto envelope = ComponentDrafterEnvelope(
              {.context = settings.context.value, .slots = settings.max_slots.value});
          !envelope)
        return Error(envelope.error());
      const auto& composition = *prepared->composition;
      if (auto checked = composition.CheckBinding(*artifact, *drafter, *synthesized,
                                                  target_vocabulary, settings.draft_vocab.value);
          !checked)
        return Error(checked.error());
      hold->spec = composition.model();
      hold->numerical_semantics[1] = composition.numerical_semantic();
      hold->draft_head_rows = composition.vocabulary().value_or(0);
      hold->draft_head_ids = composition.head_ids();
      hold->sources.push_back(prepared->drafter_admission->source);
    } else {
      hold->spec = std::move(*synthesized);
    }
    if (settings.kv_cache_encoding.value != "f16") {
      if (settings.kv_cache_encoding.value != "q8_0") return Error("unsupported KV encoding");
      auto bytes = components::KvCacheEncodingBytes(hold->spec, format::Encoding::kQ8_0);
      if (!bytes) return Error(bytes.error());
      if (!metadata) {
        auto admitted = admit_metadata();
        if (!admitted) return Error(admitted.error());
        metadata = std::move(*admitted);
      }
      if (!charge_metadata(metadata, *bytes))
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
    if (auto valid = components::ValidateSpec(hold->spec, hold->tables); !valid) {
      return Error(valid.error());
    }
    if (options.automatic_startup_cap != 0) {
      // Component families use the same paid construction inventory without
      // inventing a NumericalDomain (their serving recipe is checked above).
      auto bytes = ServingConstructionBytes(hold->spec, *artifact, drafter ? &*drafter : nullptr);
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
      // Preserve the conservative whole-extent footprint for each live build,
      // including its local charge owner. Pool capacity is a node-lifetime
      // reservation; failures release local use so retries can reuse it.
      if (__builtin_add_overflow(bytes->retained,
                                 engine::ExecutionMetadataAllowance::OverheadBytes(), &total) ||
          __builtin_add_overflow(total, kExtent - 1, &total))
        return Error("automatic construction payment overflows");
      total = total / kExtent * kExtent;
      auto& pool = preparation->automatic_construction;
      if (!pool) {
        auto owner = admit_metadata();
        if (!owner) return Error(owner.error());
        pool = std::move(*owner);
      }
      if (!charge_metadata(pool, total))
        return Error("automatic construction metadata allowance is insufficient");
      hold->automatic_construction = std::make_shared<ConstructionCharge>();
      hold->automatic_construction->allowance = pool;
      hold->automatic_construction->bytes = total;
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
    // Resolve the admitted composition before choosing any mechanism. Its
    // numerical/source policy is separate from mechanical backend support.
    spec::PolicyRule composition = *stored_policy;
    const bool forced = recipe.serving_rule != nullptr;
    if (forced) composition = *recipe.serving_rule;
    auto indexed = components::StageIndexedMasks(hold->spec, 0);
    if (!indexed) return Error(indexed.error());
    const auto backend_capabilities = engine::DeviceBackendCapabilities();
    const bool indexed_masks = *indexed && backend_capabilities.indexed_masks;
    // The actual stored/composed contracts and selected emission mode bound
    // every mask and index-score plane's signed int32 stride (RE-037).
    auto prompt_bound = components::StagePrefillRows(hold->spec, 0, settings.context.value,
                                                     settings.prefill_chunk.value, indexed_masks);
    if (!prompt_bound) return Error(prompt_bound.error());
    const auto configured_chunk = *prompt_bound;
    // Execution semantics belong to the stored stage. Registry entries only
    // offer optional performance choices for an already admitted graph.
    const spec::DraftRecipe* draft =
        speculate && drafter_mapping ? drafter_mapping->draft : nullptr;
    const spec::StageDrafting* contract = speculate && hold->spec.stages.back().drafting
                                              ? &*hold->spec.stages.back().drafting
                                              : nullptr;
    if (speculate && !contract) return Error("stored drafter execution contract is absent");
    const auto* block_contract = contract ? std::get_if<spec::BlockDraft>(contract) : nullptr;
    const bool block = block_contract != nullptr;
    const bool mtp = contract && std::holds_alternative<spec::MtpDraft>(*contract);
    const auto small_rows = draft != nullptr ? draft->small_chunk_rows : 4U;
    const auto small_depth = draft != nullptr ? draft->small_chunk_max_depth : 2U;
    const auto depth =
        mtp ? (configured_chunk >= small_rows ? settings.draft_rows.value
                                              : std::min(settings.draft_rows.value, small_depth))
        : contract ? settings.draft_rows.value
                   : 0U;
    const std::int32_t block_fill = block_contract ? block_contract->fill_token : 0;
    if (speculate && (depth == 0 || depth == UINT32_MAX))
      return Error("draft depth must leave room for its anchor in uint32 rows");
    if (block_contract && depth > block_contract->max_proposals)
      return Error("the resolved draft depth exceeds the stored block bound");
    // Each stage's state by model::ExtentLayout's per-tensor rule: a tensor
    // of an extent or more on its own extents (its growth crosses extents
    // with every other's), smaller ones packed, a bank kept contiguous.
    static_assert(model::kStateExtent == kExtent, "state is laid out over the paged extent");
    if (auto state = PrepareComponentState(*hold, settings.context.value, configured_chunk); !state)
      return Error(state.error());
    const model::StateLayoutDerived& target = hold->layouts.front();
    const auto caps = hold->batching.front();
    const std::uint32_t slots = std::clamp<std::uint32_t>(
        settings.max_slots.value, 1, static_cast<std::uint32_t>(Llm::kMaxBranches));
    const std::uint32_t context = target.context;
    const std::uint32_t chunk = std::max<std::uint32_t>(target.max_rows, 1);
    // One immutable value controls composition, actual lanes and prefill.
    // A setting requests a mechanism; only the admitted components/backend
    // supply support. Neither this value nor a domain creates numerical grants.
    auto serving_policy = LlmServingPolicy::Resolve(
        {.batching = caps, .context = context, .max_rows = chunk, .indexed_masks = *indexed},
        backend_capabilities,
        {.owners = slots,
         .read_align = settings.wave_read_align.value,
         .wave_lanes = settings.wave_lanes.value,
         .lane_scratch_bytes = kLaneScratchBytes,
         .owners_alone = composition.owners_alone,
         .prompts_alone = composition.prompts_alone,
         .common_read = composition.common_read,
         .lookahead = options.prefill_lookahead ? composition.prefill_lookahead : 0U,
         .capture_ahead = composition.capture_ahead,
         .prepare_ahead = options.prepare_state,
         .state_only = options.state_only_prefill.value_or(!speculate),
         .trial_mixed_outputs = options.trial_mixed_outputs,
         .trial_mixed_causal = options.trial_mixed_causal});
    if (!serving_policy) return Error(serving_policy.error());
    // A verify's rows: the anchor and every draft.
    const std::uint32_t verify_rows = speculate ? depth + 1 : 0;
    if (speculate && std::ranges::any_of(hold->layouts, [verify_rows](const auto& layout) {
          return verify_rows > layout.max_rows;
        }))
      return Error("the complete drafter verify exceeds its actual stage row bounds");
    const std::uint32_t read_align = serving_policy->read_align();
    engine::DecodingMode mode;
    if (speculate)
      mode = {.kind = engine::DecodingMode::Kind::kSpeculative,
              .drafters = {engine::DrafterSpec{
                  .stage = 1,
                  .shape = block ? engine::DrafterSpec::Shape::kBlock
                                 : engine::DrafterSpec::Shape::kAutoregressive,
                  .max_depth = depth,
                  .shared_wave_depth = mtp ? std::min(settings.shared_wave_depth.value, depth) : 0,
                  .draft_wave_max = mtp ? settings.draft_wave_max.value : 0,
                  .catch_up = mtp,
                  .fill = block_fill,
                  .borrows_target = !mtp && !block,
                  .anchored = !mtp && !block,
                  .features = !block,
                  .adaptive = mtp && depth >= 3 && depth <= 7,
                  .depth_cost = mtp ? settings.depth_cost_ratio.value
                                    : (draft != nullptr ? draft->default_depth_cost : 1.16),
                  .device_tokens = serving_policy->device_tokens(),
                  .draft_before_limit = true,
                  .head_ids = hold->draft_head_ids,
                  .head_rows = hold->draft_head_rows}},
              .combine = engine::DecodingMode::Drafters::kCascade,
              .row_invariant = false};
    auto execution_domain = options.execution_domain;
    if (!execution_domain && !metadata) {
      auto admitted = admit_metadata();
      if (!admitted) return Error(admitted.error());
      metadata = std::move(*admitted);
    }
    if (!execution_domain) {
      auto domain =
          ComponentServingDomain(hold->spec, hold->layouts, hold->batching, slots, mode, metadata,
                                 [metadata, charge_metadata](std::uint64_t bytes) {
                                   return charge_metadata(metadata, bytes);
                                 });
      if (!domain) return Error(domain.error());
      execution_domain = std::move(*domain);
    }
    if (auto checked =
            ValidateServingEnvelope(*execution_domain, hold->layouts, hold->batching, slots);
        !checked)
      return Error(checked.error());
    if (!options.diagnostic_setup_shapes.empty()) {
      if (!options.execution_domain)
        return Error("diagnostic setup shapes require an explicit source domain");
      if (!metadata) {
        auto admitted = admit_metadata();
        if (!admitted) return Error(admitted.error());
        metadata = std::move(*admitted);
      }
      // Source/build owner, immutable retained shape copies and their vectors.
      // Standard allocator/control overhead is named separately from payload.
      std::uint64_t bytes = sizeof(ConstructionCharge) + 2 * 64;
      for (const auto& shape : options.diagnostic_setup_shapes) {
        std::uint64_t segments = 0, item = 0;
        if (__builtin_mul_overflow(static_cast<std::uint64_t>(shape.segments.size()),
                                   sizeof(planner::Segment), &segments) ||
            __builtin_add_overflow(segments, sizeof(planner::StepShape) + 2 * 64, &item) ||
            __builtin_add_overflow(bytes, item, &bytes))
          return Error("diagnostic setup inventory overflows");
      }
      if (!charge_metadata(metadata, bytes))
        return Error("diagnostic setup metadata allowance is insufficient");
      hold->setup_charge = std::make_shared<ConstructionCharge>();
      hold->setup_charge->allowance = metadata;
      hold->setup_charge->bytes = bytes;
    }
    auto setup_caps = hold->batching;
    setup_caps.front() = serving_policy->batching();
    auto measured = ComponentSetupShapes(
        hold->layouts, setup_caps, slots, read_align, mode, options.execution_domain.get(),
        options.diagnostic_setup_shapes, serving_policy->device_tokens());
    if (!measured) return Error(measured.error());
    auto setup = std::move(*measured);
    auto placed =
        PrepareComponentPlacement(*hold, *artifact, drafter ? &*drafter : nullptr, *stored_bindings,
                                  setup, serving_policy->indexed_masks());
    if (!placed) return Error(placed.error());
    auto placements = std::move(placed->slabs);
    const auto logit_width = placed->logit_width;
    if (block && std::cmp_greater_equal(block_fill, logit_width))
      return Error("a block drafter's fill token is outside the vocabulary");
    // A layer's routed experts in one slab (the M3 runners' pitch).
    ComponentHold* held = hold.get();
    kernels::DeviceBackendOptions backend_options{
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
                  }},
        .lanes = serving_policy->lanes()};
    auto backend = engine::CreateDeviceBackend(node, static_cast<std::uint32_t>(index),
                                               std::move(backend_options));
    if (!backend) return Error(backend.error());
    hold->backend = std::move(*backend);
    // Native construction is funded for every family, whether or not its
    // execution source requests a numerical domain. The shared allowance and
    // catalog-aware admission outlive all payments returned with programs.
    if (!metadata) {
      auto admitted = admit_metadata();
      if (!admitted) return Error(admitted.error());
      metadata = std::move(*admitted);
    }
    if (auto funded = hold->backend->FundConstruction(
            [metadata, charge_metadata](std::uint64_t bytes) {
              return charge_metadata(metadata, bytes);
            },
            [metadata](std::uint64_t bytes) { metadata->Release(bytes); });
        !funded)
      return Error(funded.error());
    {
      if (!metadata) {
        auto admitted = admit_metadata();
        if (!admitted) return Error(admitted.error());
        metadata = std::move(*admitted);
      }
      const auto tuning =
          options.tuning ? std::span<const execution::TunedShape>(options.tuning->facts().tuned)
                         : std::span<const execution::TunedShape>{};
      if (auto declared = hold->backend->Declare(
              tuning,
              [metadata, charge_metadata](std::uint64_t bytes) {
                return charge_metadata(metadata, bytes);
              },
              [metadata](std::uint64_t bytes) { metadata->Release(bytes); });
          !declared)
        return Error(declared.error());
    }

    auto sealed = SealComponentSources(*hold, options, composition, forced, speculate,
                                       drafter_mapping, *bound, slots, context, execution_domain,
                                       *artifact, drafter ? &*drafter : nullptr);
    if (!sealed) return Error(sealed.error());
    auto step_source = std::move(*sealed);
    return AssembleComponentParts(node, store, id, drafter_id, index, settings, recipe, options,
                                  metadata,
                                  {.hold = std::move(hold),
                                   .source = std::move(step_source),
                                   .placements = std::move(placements),
                                   .rows = std::move(stored_bindings->rows),
                                   .setup = std::move(setup),
                                   .mode = std::move(mode),
                                   .policy = *serving_policy,
                                   .bound = *bound,
                                   .logit_width = logit_width,
                                   .verify_rows = verify_rows,
                                   .branch_depth = mtp ? depth : 0,
                                   .target_artifact = artifact});
  };
}

}  // namespace llmp::runtime
