// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <limits>
#include <utility>

#include "engine/model_runner.h"
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

Status ModelRunner::CheckRouting(const kn::StepProgram& program) const {
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
  kn::RoutedAccess access;
  Copies inputs, outputs, carries;
  std::vector<catalog::ExtentId> selected;
  catalog::Closure closure;
  std::uint64_t charge = 0;
  bool capture = false, started = false, unknown = false;
  StepQueued prefix, continuation;
  explicit RoutedFlight(kn::RoutedAccess token) : access(token) {}
  ~RoutedFlight() {
    if (state) state->allowance->Release(charge);
  }
};

Status ModelRunner::BeginRouted(kn::StepProgram& program, StepPlanRuns& runs, const Copies& inputs,
                                const Copies& outputs, const Copies& carries, bool capture) {
  if (!program.routed()) return {};
  if (!routed_ || routed_flight_ || routed_->epoch == UINT64_MAX)
    return Error("routed execution is unprepared, still retained or exhausted");
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
  auto flight = std::make_shared<RoutedFlight>(access);
  flight->owner = std::move(owner);
  flight->state = routed_;
  flight->charge = bytes;
  flight->program = program.routed();
  flight->runs = &runs;
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
  // Copies retained by recordings are the actual immutable staging layout,
  // distinct from the newly prepared routing values at those same addresses.
  runs.copies = inputs;
  runs.routed_outputs = outputs;
  routed_flight_ = std::move(flight);
  return {};
}

void ModelRunner::ForgetCompletedRouting() {
  if (routed_) routed_->completed_program = nullptr;
}

std::expected<std::pair<std::uint64_t, std::span<const std::int32_t>>, std::string>
ModelRunner::CompletedRoute(const kn::RoutedProgram& program, std::size_t group) const {
  if (!routed_ || routed_flight_ || routed_->completed_program != &program || routed_->epoch == 0 ||
      group >= program.groups() || !routed_->pinned)
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

void ModelRunner::ForgetUnstartedRouted() {
  if (routed_flight_ && !routed_flight_->started) routed_flight_.reset();
}

StepQueued ModelRunner::QueueRoutedPrefix(providers::NativeStream native) {
  auto flight = routed_flight_;
  if (!flight) return {.result = std::unexpected(kn::StepFailure{.detail = "no routed flight"})};
  flight->started = true;
  const auto bytes = flight->program->layout().host_bytes;
  flight->prefix = runs_.QueueRouted(
      *flight->runs, flight->access, *flight->program, flight->inputs, {},
      {static_cast<std::byte*>(routed_->pinned), bytes}, flight->capture, graph_stats_, native);
  flight->unknown =
      !flight->prefix.result && flight->prefix.result.error().error == kn::StepError::kUnknown;
  return flight->prefix;
}

Status ModelRunner::ContinueRouted(RunPath& path) {
  auto flight = routed_flight_;
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
    auto pinned =
        std::span(static_cast<std::byte*>(routed_->pinned) + layout.host_offset, layout.host_bytes);
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
    // Only the scheduler's catalog thread mints current content generations.
    // SupplementalJob then materializes/leases this exact closure under the
    // still-held base request before invoking the captured payload below.
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
    const auto segment = group + 1;
    flight->access = kn::RoutedAccess(flight->program, flight->access.epoch(), segment);
    const bool final = segment == flight->program->groups();
    const auto posted = node_.SupplementalJob(
        flight->closure,
        [this, flight, segment, final](providers::NativeStream native) {
          const auto next_staging =
              final ? std::span<std::byte>{}
                    : std::span(static_cast<std::byte*>(flight->state->pinned) +
                                    flight->program->layout(segment).host_offset,
                                flight->program->layout(segment).host_bytes);
          flight->continuation = runs_.QueueRouted(
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
        },
        flight, routed_->allowance, "routed selected continuation", stream_);
    if (!posted || flight->unknown || !flight->continuation.result || launch_->faulted()) {
      // No false inference from an error to retirement. The retained flight
      // keeps the complete caller owner, plan, source, pinned data and launch.
      return Error(!posted                        ? posted.error()
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
  routed_->completed_program = flight->program;
  routed_flight_.reset();
  return {};
}

}  // namespace llmp::engine
