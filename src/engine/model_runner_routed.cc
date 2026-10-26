// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <format>
#include <limits>
#include <utility>

#include "engine/model_runner.h"
#include "engine/step_transaction.h"
#include "engine/supplemental_run.h"
#include "engine/support.h"
#include "providers/device_runtime.h"

namespace llmp::engine {
namespace {
namespace kn = kernels;
using support::Error;
using support::Pointer;
using support::Round;

bool SameWeight(const kn::RoutedWeight& a, const kn::RoutedWeight& b) {
  return a.ref == b.ref && a.format == b.format && a.parts == b.parts;
}

// This walks actual reserved device pages of the opened artifact. It does not
// inspect resident payload or assume that one expert owns an entire page.
Status PartPages(const PagedWeights& weights, std::string_view name,
                 std::optional<std::uint32_t> expert, std::span<catalog::ExtentId> temporary,
                 const std::function<Status(catalog::ExtentId)>& visit) {
  const auto& artifact = weights.artifact();
  if (const auto array = artifact.FindExpertArray(name)) {
    const auto& a = artifact.expert_arrays()[*array];
    if (a.count == 0 || (expert && *expert >= a.count)) return Error("invalid expert slice");
    const std::uint32_t begin = expert.value_or(0);
    const std::uint32_t end = expert ? begin + 1 : a.count;
    for (std::uint32_t e = begin; e < end; ++e) {
      std::uint64_t address = 0;
      if (__builtin_add_overflow(weights.group_address(a.first_group + e), a.group_offset.value(),
                                 &address))
        return Error("expert slice address overflow");
      const auto n = weights.ExtentsTouching(address, a.readable.value(), temporary);
      if (!n) return Error(n.error());
      for (const auto id : temporary.first(*n))
        if (auto result = visit(id); !result) return result;
    }
    return {};
  }
  if (expert) return Error("a routed part is not an expert array");
  const auto resource = artifact.FindResource(name);
  if (!resource) return Error("a weight reader names no opened source resource");
  const auto& r = artifact.resources()[*resource];
  // Row tables have their separate bounded residency path. They cannot alias
  // these placed dynamic expert pages and are not accepted as an exclusion.
  if (weights.group_address(r.group) == 0 && r.rows) return {};
  const auto n =
      weights.ExtentsTouching(weights.resource_address(*resource), r.readable.value(), temporary);
  if (!n) return Error(n.error());
  for (const auto id : temporary.first(*n))
    if (auto result = visit(id); !result) return result;
  return {};
}
}  // namespace

struct ModelRunner::RoutedState {
  std::shared_ptr<ExecutionMetadataAllowance> allowance;
  std::uint64_t charge = 0;
  base::Sha256Digest execution{};
  planner::PlanKey source_key;
  std::vector<kn::RoutedWeight> weights;
  std::vector<base::Sha256Digest> signatures;
  std::vector<catalog::ExtentId> pages, temporary, excluded;
  std::vector<std::uint8_t> roles;
  void* pinned = nullptr;
  std::uint64_t pinned_bytes = 0, epoch = 0;
  const kn::RoutedProgram* completed_program = nullptr;
  bool finalized = false;
  ~RoutedState() { allowance->Release(charge); }
};

Status ModelRunner::MeasureRouting(const kn::StepProgram& program) {
  if (!config_.routed_metadata) {
    if (program.routed()) return Error("a routed step lacks admitted metadata and complete owner");
    return {};
  }
  const auto reads = program.weight_reads();
  const auto identity = source_->ExecutionIdentity();
  if (!reads || !identity || !*identity || config_.routed_owner.expired())
    return Error("routed source needs complete declared reads, execution identity and owner");
  if (!routed_) {
    if (!program.routed()) return Error("the first measured routed source must contain its group");
    // Retained inventory + transient vector relocation/string copies, page
    // sets/scratch and one complete in-flight continuation. No payload bytes.
    std::uint64_t pages = 0, bytes = 32768;
    const auto add = [&](std::uint64_t n) {
      return !__builtin_add_overflow(bytes, n, &bytes) && bytes <= SIZE_MAX;
    };
    for (const auto& stage : config_.weights) {
      if (__builtin_add_overflow(pages, std::uint64_t{stage.weights->extents().size()}, &pages))
        return Error("routed page inventory overflow");
    }
    if (pages == 0 || pages > (UINT64_MAX - bytes) / 256 || !add(pages * 256))
      return Error("routed page metadata bound overflow");
    for (const auto& weight : *reads) {
      if (!add(4 * sizeof(kn::RoutedWeight)) || weight.parts.size() > (UINT64_MAX - bytes) / 256 ||
          !add(weight.parts.size() * 256))
        return Error("routed source inventory bound overflow");
      for (const auto& part : weight.parts) {
        std::uint64_t string_bytes = 0;
        if (__builtin_add_overflow(std::uint64_t{part.first.size()}, std::uint64_t{1},
                                   &string_bytes) ||
            __builtin_mul_overflow(string_bytes, std::uint64_t{4}, &string_bytes) ||
            !add(string_bytes))
          return Error("routed source name bound overflow");
      }
    }
    const auto source_key = source_->KeyBase();
    if (source_key.parameters.size() > (UINT64_MAX - bytes) / 32 ||
        !add(source_key.parameters.size() * 32))
      return Error("routed source key bound overflow");
    if (config_.setup_shapes.empty() ||
        config_.setup_shapes.size() > (UINT64_MAX - bytes) / (4 * sizeof(base::Sha256Digest)) ||
        !add(config_.setup_shapes.size() * (4 * sizeof(base::Sha256Digest))))
      return Error("routed shape signature inventory overflow");
    if (!config_.routed_metadata->Charge(bytes))
      return Error("routed metadata allowance exhausted");
    auto state = std::make_shared<RoutedState>();
    state->allowance = config_.routed_metadata;
    state->charge = bytes;
    state->execution = **identity;
    state->source_key = source_key;
    state->source_key.shape = {};
    state->weights.assign(reads->begin(), reads->end());
    state->signatures.reserve(config_.setup_shapes.size());
    state->pages.reserve(static_cast<std::size_t>(pages));
    for (const auto& stage : config_.weights)
      state->pages.insert(state->pages.end(), stage.weights->extents().begin(),
                          stage.weights->extents().end());
    std::ranges::sort(state->pages);
    if (std::adjacent_find(state->pages.begin(), state->pages.end()) != state->pages.end())
      return Error("routed stages share ambiguous source extents");
    state->temporary.resize(state->pages.size());
    state->excluded.reserve(state->pages.size());
    state->roles.resize(state->pages.size(), 0);
    routed_ = std::move(state);
  } else {
    if (auto current = CheckRoutingSource(); !current) return current;
    for (const auto& read : *reads) {
      auto old = std::ranges::find(routed_->weights, read.ref, &kn::RoutedWeight::ref);
      if (old == routed_->weights.end() || !SameWeight(*old, read))
        return Error("a measured routed shape changes its exact source bindings");
      // An ordinary shape retaining any reader prevents exclusion across ALL
      // cached shapes, including physical pages shared with other tensors.
      old->dynamic = old->dynamic && read.dynamic;
    }
  }
  if (const auto* route = program.routed()) {
    if (route->groups() == 0 || route->groups() == SIZE_MAX)
      return Error("routed preparation has no representable group inventory");
    std::uint64_t total = 0;
    for (std::size_t g = 0; g != route->groups(); ++g) {
      const auto& layout = route->layout(g);
      if (!kn::GroupedRouteBound(layout.shape) || layout.host_bytes == 0 ||
          layout.host_bytes % 256 != 0 || layout.host_offset != total ||
          __builtin_add_overflow(total, layout.host_bytes, &total) || route->members(g).empty())
        return Error("routed preparation has invalid bounded storage or membership");
      for (const auto member : route->members(g))
        if (member >= route->weights().size() || !route->weights()[member].dynamic)
          return Error("routed group refers to an undeclared dynamic weight");
    }
    if (total != route->pinned_bytes()) return Error("routed pinned sum differs");
    if (!std::ranges::contains(routed_->signatures, route->routing_identity())) {
      if (routed_->signatures.size() == config_.setup_shapes.size())
        return Error("routed shape signatures exceed their prepaid inventory");
      routed_->signatures.push_back(route->routing_identity());
    }
    routed_->pinned_bytes = std::max(routed_->pinned_bytes, total);
  }
  return {};
}

Status ModelRunner::CheckRoutingSource() const {
  if (!routed_) return {};
  const auto identity = source_->ExecutionIdentity();
  auto key = source_->KeyBase();
  key.shape = {};
  if (!identity || !*identity || **identity != routed_->execution || key != routed_->source_key)
    return Error("routed source or effective execution identity changed");
  return {};
}

Status ModelRunner::FinishRoutingSetup() {
  if (!routed_) return {};
  if (auto current = CheckRoutingSource(); !current) return current;
  if (auto bindings = CheckRoutedWeightBindings(routed_->weights); !bindings) return bindings;
  const auto bind = AddressesFor(config_.setup_shapes.front(), true).expert_part;
  for (std::size_t index = 0; index < routed_->weights.size(); ++index) {
    const auto& w = routed_->weights[index];
    if (w.ref.stage >= config_.weights.size() || w.parts.empty())
      return Error("routed source weight has no complete artifact binding");
    for (std::size_t j = 0; j < index; ++j)
      if (routed_->weights[j].ref == w.ref) return Error("duplicate routed source role");
    if (w.dynamic) {
      auto matrix = w.format;
      if (!w.ref.expert_array || matrix.shape.rank != 3)
        return Error("dynamic source is not an expert array");
      const auto experts = matrix.shape.ne[2];
      matrix.shape.rank = 2;
      matrix.shape.ne[2] = 1;
      const auto parts = format::PartsOf(matrix);
      if (!parts || parts->size() != w.parts.size())
        return Error("dynamic source part coverage differs");
      for (std::size_t p = 0; p < w.parts.size(); ++p) {
        const auto& [name, role] = w.parts[p];
        for (std::size_t j = 0; j < p; ++j)
          if (w.parts[j].first == name || w.parts[j].second == role)
            return Error("duplicate dynamic source part");
        const auto actual = bind(w.ref.stage, name);
        const auto array = config_.weights[w.ref.stage].weights->artifact().FindExpertArray(name);
        if (!actual || !array || actual->format != matrix || actual->role != role ||
            config_.weights[w.ref.stage].weights->artifact().expert_arrays()[*array].count !=
                experts ||
            std::ranges::find(parts->parts(), role, &format::PartExtent::role) ==
                parts->parts().end())
          return Error("dynamic source part differs from opened artifact");
      }
    }
    for (const auto& [name, role] : w.parts) {
      (void)role;
      if (auto found =
              PartPages(*config_.weights[w.ref.stage].weights, name, {}, routed_->temporary,
                        [&](catalog::ExtentId id) -> Status {
                          const auto page = std::ranges::lower_bound(routed_->pages, id);
                          if (page == routed_->pages.end() || *page != id)
                            return Error("foreign source page");
                          routed_->roles[static_cast<std::size_t>(page - routed_->pages.begin())] |=
                              w.dynamic ? 1 : 2;
                          return {};
                        });
          !found)
        return found;
    }
  }
  for (std::size_t i = 0; i < routed_->pages.size(); ++i)
    if (routed_->roles[i] == 1) routed_->excluded.push_back(routed_->pages[i]);
  if (routed_->pinned_bytes == 0) return Error("routed source has no measured preparation storage");
  const auto pin = memory_.PinnedAligned(routed_->pinned_bytes, 256);
  if (!pin) return Error(pin.error());
  routed_->pinned = pin->data;
  routed_->finalized = true;
  return {};
}

std::uint64_t ModelRunner::RoutedPinnedBytes() const {
  return routed_ ? routed_->pinned_bytes + 255 : 0;
}

Status ModelRunner::CheckRoutingBindings(const kn::StepProgram& program) const {
  if (!routed_) return program.routed() ? Error("unmeasured routed program") : Status{};
  if (!routed_->finalized) return Error("routed source inventory is not finalized");
  if (auto current = CheckRoutingSource(); !current) return current;
  const auto reads = program.weight_reads();
  if (!reads) return Error("a dynamic source's cached program omitted read authority");
  if (auto bindings = CheckRoutedWeightBindings(*reads); !bindings) return bindings;
  for (const auto& read : *reads) {
    const auto old = std::ranges::find(routed_->weights, read.ref, &kn::RoutedWeight::ref);
    if (old == routed_->weights.end() || !SameWeight(*old, read) || (old->dynamic && !read.dynamic))
      return Error("cached program changed a dynamic source binding or reader");
    if (!read.dynamic) continue;
    // Deferred descriptors do not imply whole-array residency. Preserve
    // catalog ownership/class coverage for every actual consumed source slice,
    // excluding unconsumed pitch padding. Selected residency is admitted later
    // by the supplemental job while the ordinary base lease remains held.
    const auto& weights = *config_.weights[read.ref.stage].weights;
    for (const auto& [name, role] : read.parts) {
      (void)role;
      const auto array = weights.artifact().FindExpertArray(name);
      if (!array) return Error("dynamic part lost its source array");
      const auto& part = weights.artifact().expert_arrays()[*array];
      for (std::uint32_t expert = 0; expert != part.count; ++expert) {
        std::uint64_t address = 0;
        if (__builtin_add_overflow(weights.group_address(part.first_group + expert),
                                   part.group_offset.value(), &address) ||
            node_.Covered(address, part.readable.value(), owner_, false) !=
                catalog::MemoryClass::kWeights)
          return Error("dynamic expert part lacks owned catalog reservation coverage");
      }
    }
  }
  return {};
}

Status ModelRunner::CheckRouting(const kn::StepProgram& program) const {
  if (auto bindings = CheckRoutingBindings(program); !bindings) return bindings;
  if (!routed_) return {};
  if (const auto* route = program.routed();
      route && (route->pinned_bytes() > routed_->pinned_bytes ||
                !std::ranges::contains(routed_->signatures, route->routing_identity())))
    return Error("routed program changed its measured group identity or preparation storage");
  return {};
}

std::vector<catalog::ExtentId> ModelRunner::base_weight_extents() const {
  auto pages = weight_extents();
  if (routed_)
    std::erase_if(pages, [&](catalog::ExtentId id) {
      return std::ranges::binary_search(routed_->excluded, id);
    });
  return pages;
}

struct ModelRunner::RoutedFlight {
  std::shared_ptr<const void> owner;
  std::shared_ptr<RoutedState> state;
  kn::RoutedProgram* program = nullptr;
  StepPlanRuns* runs = nullptr;
  StepRuns execution{false};
  std::span<std::byte> pinned;
  kn::ExecutionResources resources;
  const kn::StepProgram* compiled = nullptr;
  Plans::Entry* cached = nullptr;
  base::Sha256Digest graph_identity{}, routing_identity{};
  PromptBatch batch;
  SupplementalCapacity capacity;
  std::size_t input_capacity = 0, output_capacity = 0, carry_capacity = 0;
  kn::RoutedAccess access;
  Copies inputs, outputs, carries;
  std::vector<catalog::ExtentId> selected;
  catalog::Closure closure;
  std::uint64_t charge = 0;
  bool capture = false, started = false, unknown = false;
  StepQueued prefix, continuation;
  explicit RoutedFlight(kn::RoutedAccess token) : access(token) {}
};

std::expected<std::shared_ptr<ModelRunner::RoutedFlight>, std::string>
ModelRunner::PrepareBatchRouting(Plans::Entry& planned, const kn::ExecutionResources& resources,
                                 std::span<std::byte> pinned, StepRuns& execution,
                                 StepPlanRuns& runs, std::size_t input_copies,
                                 std::size_t output_copies, std::size_t carry_copies) {
  auto& program = *planned.planned->program;
  auto* route = program.routed();
  if (!route) return std::unexpected("batch route has no authoritative routed program");
  if (resources) {
    if (!program.UsesResources(resources) || resources.launch() != execution.launch() ||
        !resources.Ready())
      return std::unexpected("private route requires its actual successful private lowering");
  } else {
    if (plans_.Find(planned.key) != &planned)
      return std::unexpected("ordinary batch route has no installed cache owner");
    if (auto checked = CheckRouting(program); !checked) return std::unexpected(checked.error());
    pinned = {static_cast<std::byte*>(routed_->pinned), routed_->pinned_bytes};
  }
  if (auto bindings = CheckRoutingBindings(program); !bindings)
    return std::unexpected(bindings.error());
  if (!routed_ || route->groups() == 0 || route->groups() == SIZE_MAX ||
      route->pinned_bytes() > pinned.size())
    return std::unexpected("private route exceeds its paid preparation inventory");
  std::uint64_t total = 0;
  for (std::size_t g = 0; g != route->groups(); ++g) {
    const auto& layout = route->layout(g);
    if (!kn::GroupedRouteBound(layout.shape) || layout.host_bytes == 0 ||
        layout.host_bytes % 256 != 0 || layout.host_offset != total ||
        __builtin_add_overflow(total, layout.host_bytes, &total) || route->members(g).empty())
      return std::unexpected("private route has invalid producer preparation storage");
    for (const auto member : route->members(g))
      if (member >= route->weights().size() || !route->weights()[member].dynamic)
        return std::unexpected("private route names no authoritative dynamic source");
  }
  if (total != route->pinned_bytes())
    return std::unexpected("private route preparation sum differs from its actual inventory");
  static_assert(sizeof(RoutedFlight) + 1024 <= 4096);
  std::uint64_t copies = input_copies, bytes = 4096;
  const auto add = [&](std::uint64_t n, std::uint64_t size) {
    std::uint64_t amount = 0;
    return !__builtin_mul_overflow(n, size, &amount) &&
           !__builtin_add_overflow(bytes, amount, &bytes) && bytes <= PTRDIFF_MAX;
  };
  if (__builtin_add_overflow(copies, std::uint64_t{output_copies}, &copies) ||
      __builtin_add_overflow(copies, std::uint64_t{carry_copies}, &copies) || !add(copies, 128) ||
      !add(routed_->pages.size(), 128) ||
      !add(route->groups() + 1, sizeof(std::unique_ptr<kn::RoutedRecording>)) ||
      !routed_->allowance->Charge(bytes))
    return std::unexpected("private routed flight metadata was not funded");
  auto flight = std::shared_ptr<RoutedFlight>(new RoutedFlight(kn::RoutedAccess(route, 0)),
                                              [](RoutedFlight* old) {
                                                auto allowance = old->state->allowance;
                                                const auto charge = old->charge;
                                                delete old;
                                                allowance->Release(charge);
                                              });
  flight->state = routed_;
  flight->charge = bytes;
  flight->compiled = &program;
  flight->program = route;
  flight->resources = resources;
  flight->cached = resources ? nullptr : &planned;
  flight->graph_identity = route->graph_identity();
  flight->routing_identity = route->routing_identity();
  flight->execution = resources ? execution : programs_.runs();
  flight->pinned = pinned.first(static_cast<std::size_t>(total));
  flight->runs = &runs;
  flight->input_capacity = input_copies;
  flight->output_capacity = output_copies;
  flight->carry_capacity = carry_copies;
  flight->inputs.reserve(input_copies);
  flight->outputs.reserve(output_copies);
  flight->carries.reserve(carry_copies);
  flight->selected.reserve(routed_->pages.size());
  runs.routed_graphs.resize(route->groups() + 1);
  runs.routed_uncapturable.resize(route->groups() + 1, 0);
  runs.copies.reserve(input_copies);
  runs.routed_outputs.reserve(output_copies);
  auto capacity =
      node_.PrepareSupplementalCapacity(stream_, routed_->pages.size(), routed_->allowance);
  if (!capacity) return std::unexpected(capacity.error());
  flight->capacity = std::move(*capacity);
  return flight;
}

Status ModelRunner::BeginRouted(kn::StepProgram& program, StepPlanRuns& runs, const Copies& inputs,
                                const Copies& outputs, const Copies& carries, bool capture,
                                std::shared_ptr<RoutedFlight> prepared, const PromptBatch& batch) {
  if (!program.routed()) return {};
  if (!routed_ || unproven_.has_routed() || routed_->epoch == UINT64_MAX)
    return Error("routed execution is unprepared, still retained or exhausted");
  if (prepared) {
    const bool private_view = bool(prepared->resources);
    if (!batch.ready() || prepared->compiled != &program || prepared->program != program.routed() ||
        (private_view
             ? (!prepared->resources.Ready() || !program.UsesResources(prepared->resources))
             : (!prepared->cached || plans_.Find(prepared->cached->key) != prepared->cached ||
                prepared->cached->planned->program.get() != &program)) ||
        prepared->runs != &runs || prepared->started || (private_view && capture) ||
        prepared->graph_identity != program.routed()->graph_identity() ||
        prepared->routing_identity != program.routed()->routing_identity() ||
        inputs.size() > prepared->input_capacity || outputs.size() > prepared->output_capacity ||
        carries.size() > prepared->carry_capacity)
      return Error("private route differs from its producer-minted paid execution capability");
    if (auto checked = private_view ? CheckRoutingBindings(program) : CheckRouting(program);
        !checked)
      return checked;
    if (runs.has_graph() && (runs.copies != inputs || runs.routed_outputs != outputs))
      return Error("prepaid routed cached staging layout changed");
    prepared->capture = capture;
    if (!runs.has_graph()) {
      runs.copies.assign(inputs.begin(), inputs.end());
      runs.routed_outputs.assign(outputs.begin(), outputs.end());
    }
    prepared->batch = batch;
    prepared->inputs.assign(inputs.begin(), inputs.end());
    prepared->outputs.assign(outputs.begin(), outputs.end());
    prepared->carries.assign(carries.begin(), carries.end());
    prepared->access = kn::RoutedAccess(prepared->program, ++routed_->epoch);
    routed_->completed_program = nullptr;
    unproven_.RetainRouted(std::move(prepared));
    return {};
  }
  if (batch) return Error("private batch route lacks its paid producer capability");
  if (auto checked = CheckRouting(program); !checked) return checked;
  if (runs.has_graph() && (runs.copies != inputs || runs.routed_outputs != outputs))
    return Error("routed captured staging layout changed");
  auto owner = config_.routed_owner.lock();
  if (!owner) return Error("routed execution has no complete lifetime owner");
  std::uint64_t copies = inputs.size();
  if (__builtin_add_overflow(copies, std::uint64_t{outputs.size()}, &copies) ||
      __builtin_add_overflow(copies, std::uint64_t{carries.size()}, &copies) ||
      copies > (UINT64_MAX - 4096) / 128)
    return Error("routed flight copy metadata overflows");
  std::uint64_t bytes = 4096 + copies * 128;
  if (routed_->pages.size() > (UINT64_MAX - bytes) / 128)
    return Error("routed flight page metadata overflows");
  bytes += routed_->pages.size() * 128;
  if (!routed_->allowance->Charge(bytes)) return Error("routed flight metadata was not admitted");
  routed_->completed_program = nullptr;
  const kn::RoutedAccess access(program.routed(), ++routed_->epoch);
  auto flight = std::shared_ptr<RoutedFlight>(new RoutedFlight(access), [](RoutedFlight* old) {
    auto allowance = old->state->allowance;
    const auto charge = old->charge;
    delete old;
    allowance->Release(charge);
  });
  flight->owner = std::move(owner);
  flight->state = routed_;
  flight->charge = bytes;
  flight->program = program.routed();
  flight->runs = &runs;
  flight->execution = programs_.runs();
  flight->pinned = {static_cast<std::byte*>(routed_->pinned), routed_->pinned_bytes};
  flight->capture = capture;
  flight->inputs = inputs;
  flight->outputs = outputs;
  flight->carries = carries;
  flight->selected.reserve(routed_->pages.size());
  const auto slots = flight->program->groups() + 1;
  if (!runs.routed_graphs.empty() && runs.routed_graphs.size() != slots)
    return Error("routed recording inventory changed for this cached program");
  // The program's host account includes these retained recording slots.
  runs.routed_graphs.resize(slots);
  runs.routed_uncapturable.resize(slots, 0);
  // Copies retained by recordings are the actual immutable staging layout,
  // distinct from the newly prepared routing values at those same addresses.
  if (!runs.has_routed_graph()) {
    runs.copies = inputs;
    runs.routed_outputs = outputs;
  }
  unproven_.RetainRouted(std::move(flight));
  return {};
}

void ModelRunner::ForgetCompletedRouting() {
  if (routed_) routed_->completed_program = nullptr;
}

std::expected<std::pair<std::uint64_t, std::span<const std::int32_t>>, std::string>
ModelRunner::CompletedRoute(const kn::RoutedProgram& program, std::size_t group) const {
  if (!routed_ || unproven_.has_routed() || routed_->completed_program != &program ||
      routed_->epoch == 0 || group >= program.groups() || !routed_->pinned)
    return std::unexpected("no completed current route to inspect");
  const auto& layout = program.layout(group);
  std::uint64_t count = 0, bytes = 0, offset = 0;
  if (__builtin_mul_overflow(layout.shape.used, layout.shape.tokens, &count) ||
      __builtin_mul_overflow(count, std::uint64_t{sizeof(std::int32_t)}, &bytes) ||
      __builtin_add_overflow(layout.host_offset, layout.ids, &offset) ||
      offset % alignof(std::int32_t) || offset > routed_->pinned_bytes ||
      bytes > routed_->pinned_bytes - offset)
    return std::unexpected("completed route inspection extent mismatch");
  const auto* ids = reinterpret_cast<const std::int32_t*>(
      static_cast<const std::byte*>(routed_->pinned) + offset);
  return std::pair{routed_->epoch, std::span<const std::int32_t>(ids, count)};
}

void ModelRunner::ForgetRouted(bool retired) {
  // A started flight's owners outlive its step only while that step's
  // retirement is unproven (a failed continuation forces that).
  auto flight = unproven_.routed_as<RoutedFlight>();
  if (flight && (!flight->started || retired)) unproven_.ReleaseProvenRouted();
}

StepQueued ModelRunner::QueueRoutedPrefix(providers::NativeStream native) {
  auto flight = unproven_.routed_as<RoutedFlight>();
  if (!flight) return {.result = std::unexpected(kn::StepFailure{.detail = "no routed flight"})};
  const auto bytes = flight->program->layout().host_bytes;
  flight->prefix = flight->execution.QueueRouted(
      *flight->runs, flight->access, *flight->program, flight->inputs, {},
      flight->pinned.first(static_cast<std::size_t>(bytes)), flight->capture, graph_stats_, native);
  flight->unknown =
      !flight->prefix.result && flight->prefix.result.error().error == kn::StepError::kUnknown;
  // Started only once work was (or may have been) queued: a prefix refused
  // before it queued anything leaves an unstarted flight, which goes with its
  // transaction (ForgetRouted) instead of refusing every later step.
  flight->started = flight->prefix.before || flight->prefix.result.has_value() || flight->unknown;
  return flight->prefix;
}

Status ModelRunner::ContinueRouted(RunPath& path, std::shared_ptr<StepTransaction> transaction) {
  auto flight = unproven_.routed_as<RoutedFlight>();
  if (!flight || !flight->started || !flight->prefix.result || flight->unknown)
    return Error("routed prefix has not completed successfully");
  if (auto current = CheckRoutingSource(); !current) return current;
  if (auto complete = flight->program->CompleteSegment(flight->access); !complete)
    return Error(complete.error().detail);
  bool all_replayed = flight->prefix.path == RunPath::kReplayed;
  bool any_captured = flight->prefix.path == RunPath::kCaptured;
  for (std::size_t group = 0; group != flight->program->groups(); ++group) {
    if (auto current = CheckRoutingSource(); !current) return current;
    const auto& layout = flight->program->layout(group);
    flight->access = kn::RoutedAccess(flight->program, flight->access.epoch(), group);
    auto pinned = flight->pinned.subspan(static_cast<std::size_t>(layout.host_offset),
                                         static_cast<std::size_t>(layout.host_bytes));
    if (auto prepared = flight->program->Prepare(flight->access, pinned); !prepared)
      return Error(prepared.error().detail);
    // Prepare proved the complete offsets/bijection and every actual ID, not a
    // caller-supplied count or a prior route's same-address worklist.
    const auto offsets =
        std::span(reinterpret_cast<const std::int32_t*>(pinned.data() + layout.work_offsets),
                  static_cast<std::size_t>(layout.shape.experts + 1));
    std::ranges::fill(routed_->roles, 0);
    for (const auto member : flight->program->members(group)) {
      const auto& weight = flight->program->weights()[member];
      for (std::uint32_t e = 0; e < layout.shape.experts; ++e) {
        if (offsets[e] == offsets[e + 1]) continue;
        for (const auto& [name, role] : weight.parts) {
          (void)role;
          if (auto found = PartPages(
                  *config_.weights[weight.ref.stage].weights, name, e, routed_->temporary,
                  [&](catalog::ExtentId id) -> Status {
                    const auto page = std::ranges::lower_bound(routed_->pages, id);
                    if (page == routed_->pages.end() || *page != id)
                      return Error("foreign selected page");
                    routed_->roles[static_cast<std::size_t>(page - routed_->pages.begin())] = 1;
                    return {};
                  });
              !found)
            return found;
        }
      }
    }
    flight->selected.clear();
    flight->closure = {};
    for (std::size_t i = 0; i < routed_->pages.size(); ++i)
      if (routed_->roles[i] != 0) flight->selected.push_back(routed_->pages[i]);
    if (flight->selected.empty()) return Error("routed group has no selected weight source");
    // Pages not in routed_->excluded are already held in the base lease (base_weight_extents()).
    std::erase_if(flight->selected, [&](catalog::ExtentId id) {
      return !std::ranges::binary_search(routed_->excluded, id);
    });
    const bool needs_lease = !flight->selected.empty();
    if (needs_lease) {
      if (auto closure = node_.CallMetadata(
              [this, flight]() -> Status {
                auto checked = node_.catalog().ClosureOfExtents(flight->selected);
                if (!checked) return Error("selected source closure is not current");
                flight->closure = std::move(*checked);
                return {};
              },
              "binding routed selected closure");
          !closure)
        return closure;
    }
    const auto segment = group + 1;
    flight->access = kn::RoutedAccess(flight->program, flight->access.epoch(), segment);
    const bool final = segment == flight->program->groups();
    JobReport retired;
    const auto queue_continuation = [this, flight, segment, final,
                                     transaction](providers::NativeStream native) {
      const auto next_staging =
          final ? std::span<std::byte>{}
                : flight->pinned.subspan(
                      static_cast<std::size_t>(flight->program->layout(segment).host_offset),
                      static_cast<std::size_t>(flight->program->layout(segment).host_bytes));
      flight->continuation = flight->execution.QueueRouted(
          *flight->runs, flight->access, *flight->program, {},
          final ? std::span<const RunCopy>(flight->outputs) : std::span<const RunCopy>{},
          next_staging, flight->capture, graph_stats_, native);
      if (!flight->continuation.result) {
        flight->unknown = flight->continuation.result.error().error == kn::StepError::kUnknown;
        return flight->unknown ? scheduler::JobResult::kUnknown : scheduler::JobResult::kFailed;
      }
      if (final)
        for (const auto& [to, from, bytes] : flight->carries) {
          const auto copied = providers::CopyAsync(native, Pointer(to), Pointer(from), bytes,
                                                   providers::CopyKind::kDeviceToDevice);
          if (!copied.ok()) {
            flight->unknown = true;
            flight->continuation.result = std::unexpected(
                kn::StepFailure{.error = kn::StepError::kUnknown, .detail = copied.text()});
            return scheduler::JobResult::kUnknown;
          }
        }
      return scheduler::JobResult::kQueued;
    };
    if (!needs_lease && !node_.InRequest(stream_))
      return Error("supplemental work requires a live request");
    bool over_budget = false;
    const auto posted =
        needs_lease ? node_.SupplementalJob(flight->closure, queue_continuation, flight,
                                            routed_->allowance, "routed selected continuation",
                                            stream_, &retired, flight->capacity, &over_budget)
                    : node_.Job({}, queue_continuation, "routed selected continuation", stream_, {},
                                &retired);
    if (!posted || flight->unknown || !flight->continuation.result ||
        flight->execution.launch()->faulted()) {
      // No false inference from an error to retirement. The retained flight
      // keeps the complete caller owner, plan, source, pinned data and launch.
      // A capacity refusal says so: the selection did not fit the budget.
      return Error(!posted && over_budget
                       ? std::format("capacity: the routed selection does not fit the budget ({})",
                                     posted.error())
                   : !posted                      ? posted.error()
                   : !flight->continuation.result ? flight->continuation.result.error().detail
                                                  : "routed continuation faulted");
    }
    // SupplementalJob success includes actual work and page-in retirement;
    // neither queue success nor graph construction advances the ordinal.
    if (auto complete = flight->program->CompleteSegment(flight->access); !complete)
      return Error(complete.error().detail);
    all_replayed = all_replayed && flight->continuation.path == RunPath::kReplayed;
    any_captured = any_captured || flight->continuation.path == RunPath::kCaptured;
  }
  path = all_replayed ? RunPath::kReplayed : any_captured ? RunPath::kCaptured : RunPath::kEager;
  ++flight->runs->eager_runs;
  routed_->completed_program = flight->resources ? nullptr : flight->program;
  unproven_.ReleaseProvenRouted();
  return {};
}

}  // namespace llmp::engine
