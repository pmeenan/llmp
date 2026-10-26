// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "engine/model_runner.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <format>
#include <limits>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

#include "base/check.h"
#include "components/step.h"
#include "engine/copy_sink.h"
#include "engine/prefill_lookahead.h"
#include "engine/step_inputs.h"
#include "engine/step_transaction.h"
#include "engine/support.h"
#include "kernels/paging/paging.h"
#include "model/state_contract.h"
#include "providers/device_runtime.h"
#include "providers/storage.h"
#include "scheduler/scheduler.h"

namespace llmp::engine {

namespace {

namespace kn = llmp::kernels;
namespace pl = llmp::planner;
namespace sc = llmp::scheduler;
using support::Address;
using support::Error;
using support::Round;

std::unexpected<StepRefusal> Refused(std::string what, bool clean = true,
                                     std::optional<std::uint32_t> slot = std::nullopt,
                                     bool state_refused = false) {
  return std::unexpected(StepRefusal{
      .what = std::move(what), .clean = clean, .slot = slot, .state_refused = state_refused});
}

}  // namespace

Status CheckRowTableSource(const RunnerConfig::RowTable& table, std::uint32_t version,
                           std::span<const artifact::Resource> resources) {
  if (table.resource >= resources.size()) return Error("a row table's resource");
  const auto& resource = resources[table.resource];
  if (version != 1 || !table.source_parts.empty() || !resource.format ||
      *resource.format != table.format || resource.part != table.part)
    return Error("a row table's format or part differs from its source resource");
  return {};
}

SlotLifecycleOptions ModelRunner::LifecycleOptions(ModelRunner* self,
                                                   SlotLifecycleOptions options) {
  // A moved place or a recovery drops every plan and recording (they name
  // the old places), then whatever the caller does beside.
  options.drop_plans = [self, also = std::move(options.drop_plans)] {
    self->DropPlans();
    if (also) also();
  };
  // Logical replacement invalidates carried features; an ordinary spill
  // preserves them in runtime memory for the drafter's next step.
  options.content_changed = [self, also = std::move(options.content_changed)](std::uint32_t slot) {
    self->ForgetCarry(slot);
    for (std::size_t stage = 0; stage < self->slots_.stages(); ++stage) {
      const std::size_t at = (std::size_t{slot} * self->slots_.stages()) + stage;
      self->tap_rows_[at] = self->previous_tap_rows_[at] = {};
    }
    if (also) also(slot);
  };
  return options;
}

ModelRunner::ModelRunner(PagedNode& node, RunnerConfig config, std::unique_ptr<StepSource> source,
                         int owner, std::uint32_t stream)
    : node_(node),
      config_(std::move(config)),
      source_(std::move(source)),
      owner_(owner),
      stream_(stream),
      slots_(node, LifecycleOptions(this, std::move(config_.slots)), owner, stream),
      memory_(node, owner, stream),
      programs_(node, owner, config_.graphs) {
  for (std::size_t stage = 0; stage < slots_.stages(); ++stage) {
    frozen_.push_back(stage != 0 && model::ReadsFrozen(slots_.layout(stage)));
  }
  carry_rows_.assign(std::size_t{slots_.slots()} * slots_.stages(), 0);
  tap_rows_.resize(carry_rows_.size());
  previous_tap_rows_.resize(carry_rows_.size());
}

ModelRunner::~ModelRunner() = default;

std::expected<std::unique_ptr<ModelRunner>, std::string> ModelRunner::Create(
    PagedNode& node, RunnerConfig config, std::unique_ptr<StepSource> source, int owner,
    std::uint32_t stream) {
  if (source == nullptr) return Error("a model runner needs a step source");
  if (!config.launch) return Error("a model runner needs a launch");
  if (config.setup_shapes.empty()) return Error("a model runner measures at least one shape");
  if (config.weights.size() > config.slots.stages.size()) {
    return Error("weights name more stages than the slots hold");
  }
  for (const StageWeights& w : config.weights) {
    if (w.weights == nullptr || !w.weights->opened())
      return Error("a stage's weights are not open");
  }
  // NOLINTNEXTLINE(cppcoreguidelines-owning-memory): the constructor is private
  return std::unique_ptr<ModelRunner>(
      new ModelRunner(node, std::move(config), std::move(source), owner, stream));
}

std::vector<catalog::ExtentId> ModelRunner::weight_extents() const {
  std::vector<catalog::ExtentId> out;
  for (const StageWeights& w : config_.weights) {
    out.insert(out.end(), w.weights->extents().begin(), w.weights->extents().end());
  }
  return out;
}

std::uint64_t ModelRunner::weight_read_bytes() const {
  std::uint64_t bytes = 0;
  for (const StageWeights& w : config_.weights) bytes += w.weights->read_bytes();
  return bytes;
}

std::vector<catalog::ExtentId> ModelRunner::managed_extents() const {
  auto result = weight_extents();
  for (const auto& extents : {slots_.state(), slots_.kept_state()}) {
    result.insert(result.end(), extents.begin(), extents.end());
  }
  return result;
}

// ------------------------------------------------------------------ its life

std::expected<void, std::string> ModelRunner::CheckShape(const pl::StepShape& shape) const {
  if (auto identity = source_->ExecutionIdentity(); !identity) return Error(identity.error());
  if (auto qualified = source_->CheckExecutionShape(shape); !qualified) return qualified;
  if (shape.segments.empty()) return Error("a step has no segments");
  if (shape.stage >= slots_.stages()) return Error("a step names no stage");
  if (shape.read_align == 0 || shape.read_align % 256 != 0) {
    return Error("a step's read alignment is not a multiple of 256");
  }
  const std::uint32_t context = slots_.layout(shape.stage).context;
  // A chunk's per-row cells (a stream, a tap) hold one chunk bound of rows:
  // a segment wider than the bound of a stage the step writes them in
  // would write past them (its own stage, and on the target each drafter's
  // taps).
  std::uint32_t row_bound = ~std::uint32_t{0};
  for (std::size_t stage = 0; stage < slots_.stages(); ++stage) {
    const model::StateLayoutDerived& layout = slots_.layout(stage);
    const bool per_row = std::ranges::any_of(layout.items, [&](const auto& item) {
      return item.contract.cells == model::CellRule::kPerRow &&
             (stage == shape.stage || (shape.stage == 0 && item.contract.target_writes));
    });
    if (per_row) row_bound = std::min(row_bound, layout.max_rows);
  }
  bool any_origin = false;
  bool any_primary = false;
  for (std::size_t i = 0; i < shape.segments.size(); ++i) {
    const pl::Segment& s = shape.segments[i];
    if (s.origin_phase &&
        (shape.stage != 0 || shape.phase != planner::Phase::kPrefill || shape.row_invariant ||
         *s.origin_phase != planner::Phase::kDecode || s.rows != 1))
      return Error("mixed causal origin requires an ordinary one-row target decode");
    any_origin = any_origin || s.origin_phase.has_value();
    any_primary = any_primary || !s.origin_phase.has_value();
    if (s.slot >= slots_.slots()) return Error(std::format("a step names slot {}", s.slot));
    if (i > 0 && s.slot <= shape.segments[i - 1].slot) {
      return Error("a step's segments are not in ascending, distinct slots");
    }
    if (s.rows == 0 || s.past > context || s.rows > context - s.past) {
      return Error(std::format("slot {}'s rows are outside its context", s.slot));
    }
    if (s.rows > row_bound) {
      return Error(std::format("slot {}'s {} rows pass the {} its per-row state holds", s.slot,
                               s.rows, row_bound));
    }
    // A stage that reads frozen reads the cells before its rows only.
    const std::uint64_t covers = Frozen(shape.stage) ? s.past : std::uint64_t{s.past} + s.rows;
    if (s.read_cells < covers || s.read_cells > context) {
      return Error(std::format("slot {}'s reads do not cover its rows", s.slot));
    }
    // A target's output rows each write a drafter's tap cell: no more of
    // them than the tap's ring holds (two rows would share a cell).
    if (shape.stage == 0 && s.outputs != 0) {
      for (std::size_t stage = 1; stage < slots_.stages(); ++stage) {
        const auto* tap = model::FeatureTap(slots_.layout(stage));
        if (tap != nullptr && s.outputs > tap->cells) {
          return Error(std::format("slot {}'s {} outputs pass stage {}'s feature tap of {} cells",
                                   s.slot, s.outputs, stage, tap->cells));
        }
      }
    }
    if (s.outputs > s.rows || (!s.want_head && s.outputs != 0)) {
      return Error(std::format("slot {}'s outputs exceed its rows", s.slot));
    }
    // D-092: a row-invariant step (an exact verify) reads what each owner
    // reads alone, never its peers' alignment or cells.
    if (shape.row_invariant && (shape.read_align != slots_.read_align(shape.stage) ||
                                s.read_cells != LoneRead(shape.stage, s.past, s.rows))) {
      return Error(std::format("slot {}'s row-invariant reads follow its peers", s.slot));
    }
  }
  if (any_origin && !any_primary) {
    return Error("a step with only decode segments must use the decode phase");
  }
  return {};
}

std::uint32_t ModelRunner::LoneRead(std::size_t stage, std::uint32_t past,
                                    std::uint32_t rows) const {
  const std::uint64_t end = Frozen(stage) ? past : std::uint64_t{past} + rows;
  return static_cast<std::uint32_t>(
      std::min<std::uint64_t>(Round(end, slots_.read_align(stage)), slots_.context(stage)));
}

bool ModelRunner::Verifies(std::uint32_t slot, std::size_t stage) const {
  if (stage == 0 || commits_.at((std::size_t{slot} * slots_.stages()) + stage)) return true;
  return std::ranges::any_of(slots_.layout(stage).items,
                             [](const auto& item) { return item.contract.target_writes; });
}

std::expected<pl::PlanKey, std::string> ModelRunner::KeyOf(const pl::StepShape& shape) const {
  pl::PlanKey key;
  if (auto filled = FillKey(shape, key); !filled) return Error(filled.error());
  return key;
}

void ModelRunner::NormalizeKeyShape(const pl::StepShape& shape, pl::StepShape& into) const {
  into.stage = shape.stage;
  into.phase = shape.phase;
  into.row_invariant = shape.row_invariant;
  into.read_align = shape.read_align;
  into.segments.assign(shape.segments.begin(), shape.segments.end());
  if (shape.stage < slots_.stages()) {
    if (const auto bound = model::ReadCellsBound(slots_.layout(shape.stage))) {
      into.read_align = 0;
      for (auto& seg : into.segments) {
        seg.read_cells =
            static_cast<decltype(seg.read_cells)>(std::min<std::uint64_t>(seg.read_cells, *bound));
      }
    }
  }
}

Status ModelRunner::FillKey(const pl::StepShape& shape, pl::PlanKey& key,
                            bool fixed_capacity) const {
  if (fixed_capacity && shape.segments.size() > key.shape.segments.capacity())
    return Error("shape exceeds paid key segment storage");
  key.spec = key_base_.spec;
  key.patterns = key_base_.patterns;
  key.policy = key_base_.policy;
  key.registry = key_base_.registry;
  key.backend = key_base_.backend;
  key.capturable = key_base_.capturable;
  key.partition = key_base_.partition;
  NormalizeKeyShape(shape, key.shape);
  auto count = source_->CountShapeParameters(key.shape);
  if (!count) return Error("shape parameter count overflow");
  if (fixed_capacity && *count > key.parameters.capacity())
    return Error("shape exceeds paid key parameter storage");
  key.parameters.resize(*count);
  if (!source_->FillShapeParameters(key.shape, key.parameters))
    return Error("shape parameter emitter changed between count and fill");
  return {};
}

Status ModelRunner::CheckRoutedWeightBindings(std::span<const kn::RoutedWeight> reads) const {
  const auto addresses = AddressesFor(config_.setup_shapes.front(), true);
  for (const auto& w : reads) {
    if (w.ref.stage >= config_.weights.size()) return Error("foreign routed weight stage");
    const auto& weights = *config_.weights[w.ref.stage].weights;
    const auto& artifact = weights.artifact();
    for (const auto& [name, role] : w.parts) {
      if (w.ref.expert_array) {
        auto matrix = w.format;
        if (matrix.shape.rank != 3) return Error("expert source rank changed");
        matrix.shape.rank = 2;
        matrix.shape.ne[2] = 1;
        const auto array = artifact.FindExpertArray(name);
        const auto part =
            addresses.expert_part ? addresses.expert_part(w.ref.stage, name) : std::nullopt;
        if (!array || !part || part->format != matrix || part->role != role ||
            artifact.expert_arrays()[*array].count != w.format.shape.ne[2])
          return Error("actual expert binding differs from its source inventory");
        const auto callback = addresses.weight(w.ref, role);
        if (callback != 0 && callback != part->address)
          return Error("expert callback aliases a different source part");
      } else {
        const auto resource = artifact.FindResource(name);
        if (!resource) return Error("ordinary reader names no source resource");
        const auto& r = artifact.resources()[*resource];
        if (!r.format || *r.format != w.format || r.part != role)
          return Error("ordinary reader format differs from its source resource");
        auto expected = weights.resource_address(*resource);
        if (r.rows && weights.group_address(r.group) == 0) {
          expected = 0;
          for (const auto& table : row_tables_)
            if (table->config.weight == w.ref && table->config.resource == *resource &&
                table->config.part == role)
              expected = table->slots.base;
        }
        if (expected == 0 || addresses.weight(w.ref, role) != expected)
          return Error("ordinary callback aliases a different source resource");
      }
    }
  }
  return {};
}

kn::Addresses ModelRunner::AddressesFor(const pl::StepShape& shape, bool measuring) const {
  std::vector<std::uint8_t> owners;
  owners.reserve(shape.segments.size());
  for (const pl::Segment& s : shape.segments) owners.push_back(s.slot);
  kn::Addresses a;
  a.weight = config_.weight_address
                 ? config_.weight_address
                 : std::function<std::uint64_t(const ir::WeightRef&, format::PartRole)>(
                       [](const ir::WeightRef&, format::PartRole) { return std::uint64_t{0}; });
  if (!row_tables_.empty()) {
    a.weight = [this, ordinary = std::move(a.weight)](const ir::WeightRef& ref,
                                                      format::PartRole role) -> std::uint64_t {
      for (const auto& table : row_tables_) {
        if (table->config.weight == ref && table->config.part == role) return table->slots.base;
      }
      return ordinary(ref, role);
    };
    a.weight_rows = [this](const ir::WeightRef& ref) -> std::uint64_t {
      for (const auto& table : row_tables_) {
        if (table->config.weight == ref) return table->capacity;
      }
      return 0;
    };
  }
  if (config_.routed_metadata || config_.split_expert_parts)
    a.expert_part = [this](std::uint16_t stage,
                           std::string_view name) -> std::optional<kn::ExpertPartBinding> {
      if (stage >= config_.weights.size()) return std::nullopt;
      const PagedWeights& weights = *config_.weights[stage].weights;
      const auto index = weights.artifact().FindExpertArray(name);
      if (!index) return std::nullopt;
      const auto& array = weights.artifact().expert_arrays()[*index];
      if (!array.format || array.count == 0 || array.readable.value() == 0) return std::nullopt;
      const std::uint64_t base = weights.array_address(*index);
      if (base == 0) return std::nullopt;
      std::uint64_t pitch = array.readable.value();
      if (array.count > 1) {
        const auto next = weights.group_address(array.first_group + 1);
        const auto first = weights.group_address(array.first_group);
        if (next <= first) return std::nullopt;
        pitch = next - first;
      }
      if (pitch < array.readable.value()) return std::nullopt;
      for (std::uint32_t e = 0; e < array.count; ++e) {
        std::uint64_t offset = 0, expected = 0, actual = 0;
        if (__builtin_mul_overflow(std::uint64_t{e}, pitch, &offset) ||
            __builtin_add_overflow(base, offset, &expected) ||
            __builtin_add_overflow(weights.group_address(array.first_group + e),
                                   array.group_offset.value(), &actual) ||
            actual != expected)
          return std::nullopt;
      }
      std::uint64_t bytes = 0, end = 0;
      if (__builtin_mul_overflow(std::uint64_t{array.count - 1}, pitch, &bytes) ||
          __builtin_add_overflow(bytes, array.readable.value(), &bytes) ||
          __builtin_add_overflow(base, bytes, &end))
        return std::nullopt;
      return kn::ExpertPartBinding{.address = base,
                                   .bytes = bytes,
                                   .pitch = pitch,
                                   .format = *array.format,
                                   .role = array.part};
    };
  // Only the state the step's stage may write (Admit's rule, which a
  // verify's saves and a failed step's quarantine follow): a target's step
  // binds the target's state and the drafters' tensors it writes
  // (target_writes); a drafter's binds its own stage's (its view of the
  // target's through it). Any other binds nothing: the build is refused.
  struct StatePlace {
    std::uint64_t base = 0;
    std::uint64_t offset = 0;
    std::uint64_t bytes = 0;
  };
  const auto state_place = [this, stage = shape.stage,
                            owners](const ir::StateRef& ref) -> StatePlace {
    if (ref.segment >= owners.size() || owners[ref.segment] >= slots_.slots() ||
        ref.stage >= slots_.stages())
      return {};
    if (ref.stage != stage && stage != 0) return {};
    const model::StateLayoutDerived& layout = slots_.layout(ref.stage);
    for (const auto& item : layout.items) {
      if (item.layer == ref.layer && item.tensor == ref.tensor) {
        if (ref.stage != stage && !item.contract.target_writes) return {};
        std::uint64_t item_end = 0;
        if (item.bytes == 0 || __builtin_add_overflow(item.offset, item.bytes, &item_end) ||
            item_end > layout.bytes)
          return {};
        return {slots_.live(owners[ref.segment], ref.stage).base(0), item.offset, item.bytes};
      }
    }
    if (ref.stage != stage) return {};
    // A drafter's view of the target's tensor: the owner's target state.
    for (const auto& borrowed : layout.borrowed) {
      if (borrowed.layer != ref.layer || borrowed.tensor != ref.tensor) continue;
      const auto* item = model::BorrowedItem(slots_.layout(0), borrowed);
      std::uint64_t item_end = 0;
      if (item == nullptr || item->bytes == 0 ||
          __builtin_add_overflow(item->offset, item->bytes, &item_end) ||
          item_end > slots_.layout(0).bytes)
        return {};
      return StatePlace{slots_.live(owners[ref.segment], 0).base(0), item->offset, item->bytes};
    }
    return {};
  };
  a.state = [state_place](const ir::StateRef& ref) -> std::uint64_t {
    const StatePlace place = state_place(ref);
    std::uint64_t address = 0;
    return place.base != 0 && !__builtin_add_overflow(place.base, place.offset, &address) ? address
                                                                                          : 0;
  };
  const auto alignment = node_.memory().ReservationAlignment().value();
  a.activation_alignment = alignment;
  // Setup reserves ordinary weights and row-table slots before measurement.
  // These are real stable reservation addresses, never lowering placeholders.
  a.weight_alignment = [weight = a.weight, alignment](const ir::WeightRef& ref,
                                                      format::PartRole part) -> std::uint64_t {
    const auto address = weight(ref, part);
    return address != 0 && alignment != 0 ? std::gcd(alignment, address) : 0;
  };
  a.state_alignment = [state_place, alignment](const ir::StateRef& ref) -> std::uint64_t {
    const StatePlace place = state_place(ref);
    std::uint64_t address = 0;
    return place.base != 0 && alignment != 0 &&
                   !__builtin_add_overflow(place.base, place.offset, &address)
               ? std::gcd(alignment, address)
               : 0;
  };
  a.save_rows = std::uint64_t{config_.max_verify_rows} * slots_.slots();
  // A save's address: the owner's (or, joined, the first owner's) rows of the
  // stage's commit buffer, zero unless all of them lie in the mapped buffer.
  a.save = [this, stage = shape.stage, owners](std::uint32_t layer, std::uint16_t component,
                                               std::string_view what,
                                               std::uint8_t segment) -> std::uint64_t {
    if (stage >= commit_places_.size() || commit_saves_.base == 0 || config_.max_verify_rows == 0 ||
        (segment != ir::kJoinedSegment && segment >= owners.size()))
      return 0;
    const std::uint32_t owner = segment == ir::kJoinedSegment ? 0 : owners[segment];
    if (owner >= slots_.slots()) return 0;
    for (const CommitPlace& place : commit_places_[stage]) {
      const CommitBuffer& buffer = place.buffer;
      if (buffer.layer != layer || buffer.component != component || buffer.what != what) continue;
      std::uint64_t owner_stride = 0, offset = 0, address = 0, rows = config_.max_verify_rows;
      std::uint64_t bytes = 0, end = 0, mapped_end = 0;
      if (buffer.row_bytes == 0 ||
          __builtin_mul_overflow(std::uint64_t{config_.max_verify_rows}, buffer.row_bytes,
                                 &owner_stride) ||
          __builtin_mul_overflow(std::uint64_t{owner}, owner_stride, &offset) ||
          __builtin_add_overflow(place.offset, offset, &offset))
        return 0;
      if (segment == ir::kJoinedSegment &&
          __builtin_mul_overflow(rows, std::uint64_t{slots_.slots()}, &rows))
        return 0;
      if (__builtin_mul_overflow(rows, buffer.row_bytes, &bytes) ||
          __builtin_add_overflow(commit_saves_.base, offset, &address) ||
          __builtin_add_overflow(address, bytes, &end) ||
          __builtin_add_overflow(commit_saves_.base, commit_saves_.bytes, &mapped_end) ||
          end > mapped_end)
        return 0;
      std::uint64_t total_rows = 0, buffer_bytes = 0, buffer_end = 0;
      if (__builtin_mul_overflow(std::uint64_t{config_.max_verify_rows},
                                 std::uint64_t{slots_.slots()}, &total_rows) ||
          __builtin_mul_overflow(total_rows, buffer.row_bytes, &buffer_bytes) ||
          __builtin_add_overflow(buffer_bytes, std::uint64_t{255}, &buffer_bytes))
        return 0;
      buffer_bytes = (buffer_bytes / 256) * 256;
      if (__builtin_add_overflow(place.offset, buffer_bytes, &buffer_end) || offset > buffer_end ||
          bytes > buffer_end - offset)
        return 0;
      if (node_.memory().ReservationAlignment().value() == 0) return 0;
      return address;
    }
    return 0;
  };
  if (!measuring) {
    a.activations = node_.activations().base;
    a.activation_bytes = activation_bytes();
  }
  return a;
}

namespace {
// `need` (Setup rounds it to whole extents) and at least one extent, within
// the `mapped` region's cataloged prefix.
std::uint64_t OwnPrefix(std::uint64_t need, const Mapped& mapped) {
  return std::min(std::max<std::uint64_t>(need, kPagedExtent), mapped.bytes);
}
}  // namespace

std::uint64_t ModelRunner::activation_bytes() const {
  return OwnPrefix(activations_needed_, node_.activations());
}
std::uint64_t ModelRunner::pool_bytes() const { return OwnPrefix(pool_needed_, node_.pool()); }

Status ModelRunner::Setup() {
  if (setup_ || released_) return Error("model runner setup is not repeatable");
  if (auto set = slots_.Setup(); !set) return set;
  if (auto tables = SetupRowTables(); !tables) return tables;
  for (StageWeights& w : config_.weights) {
    if (auto reserved = w.weights->Reserve(node_, w.places, w.slabs); !reserved) return reserved;
  }
  // Each verify row's feature has a tap cell of its own.
  for (std::size_t stage = 1; stage < slots_.stages(); ++stage) {
    const auto* tap = model::FeatureTap(slots_.layout(stage));
    if (tap != nullptr && config_.max_verify_rows != 0 &&
        tap->contract.cells == model::CellRule::kPerRow &&
        tap->contract.rollback != model::Rollback::kSnapshot) {
      return Error("a chunk feature tap needs snapshot rollback for verification");
    }
    if (tap != nullptr && config_.max_verify_rows > tap->cells) {
      return Error(std::format("a verify of {} rows passes stage {}'s feature tap of {} cells",
                               config_.max_verify_rows, stage, tap->cells));
    }
  }
  if (auto commits = SetupCommits(); !commits) return commits;
  if (auto snapshots = SetupSnapshots(); !snapshots) return snapshots;
  // The kernels' own memory, then what uses it, before any measurement.
  std::vector<std::uint64_t> bases;
  for (const RunnerConfig::KernelMemory& k : config_.kernel_memory) {
    auto& mapped = kernel_mapped_.emplace_back(std::make_unique<Mapped>());
    if (auto r = memory_.Map(*mapped, k.name, k.bytes, catalog::MemoryClass::kRuntime); !r) {
      return r;
    }
    bases.push_back(mapped->base);
  }
  if (config_.prepare) {
    // Retired at Release even when it fails part way (what it made goes).
    prepared_ = true;
    if (auto prepared = config_.prepare(bases); !prepared) return Error(prepared.error());
  }
  if (auto prepared = source_->PrepareExecution(); !prepared) return prepared;
  auto execution_identity = source_->ExecutionIdentity();
  if (!execution_identity) return Error(execution_identity.error());
  if (auto sealed = slots_.SealExecutionIdentity(*execution_identity); !sealed) return sealed;
  stage_facts_.clear();
  stage_facts_.reserve(slots_.stages());
  for (std::uint16_t stage = 0; stage < slots_.stages(); ++stage) {
    stage_facts_.push_back(source_->StageFacts(stage));
  }
  std::uint64_t activations = 0, pool = 0, inputs = 0, outputs = 0, head_row = 0;
  bool greedy_heads = false, heads_retained = true;
  std::vector<std::uint64_t> carry(slots_.stages(), 0);
  for (const pl::StepShape& shape : config_.setup_shapes) {
    if (auto checked = CheckShape(shape); !checked) {
      return Error(std::format("a setup shape: {}", checked.error()));
    }
    auto program = source_->Build(shape, AddressesFor(shape, true));
    if (!program) return Error(std::format("measuring a step: {}", program.error()));
    const kn::StepProgram& p = **program;
    ObservePlanCoverage(p);
    if (auto routed = MeasureRouting(p); !routed) return routed;
    auto measured = ProgramEnvelope::Measure(p);
    if (!measured) return Error(measured.error());
    activations = std::max(activations, measured->scratch);
    pool = std::max(pool, measured->pool);
    for (const kn::OutputSlot& slot : p.outputs()) {
      // A drafter's next features: what one segment's carry holds at most.
      if (slot.features && shape.stage != 0) {
        carry[shape.stage] = std::max(carry[shape.stage], slot.bytes);
      }
    }
    // Whether each greedy target owner keeps the head logits its device
    // choices came from (StepProgram::retained), for a head row's readback.
    for (std::size_t s = 0; shape.stage == 0 && s < shape.segments.size(); ++s) {
      const pl::Segment& seg = shape.segments[s];
      if (!seg.greedy || seg.outputs == 0) continue;
      greedy_heads = true;
      const auto kept = std::ranges::find_if(
          p.retained(), [&](const kn::OutputSlot& slot) { return slot.segment == s; });
      if (kept == p.retained().end() || kept->bytes == 0 || kept->bytes % seg.outputs != 0)
        heads_retained = false;
      else
        head_row = std::max(head_row, kept->bytes / seg.outputs);
    }
    inputs = std::max(inputs, measured->inputs);
    outputs = std::max(outputs, measured->outputs);
    plan_floor_bytes_ = std::max(plan_floor_bytes_, measured->host);
  }
  // A readback lands in the output staging: one row of it at least.
  retains_head_rows_ = greedy_heads && heads_retained;
  if (retains_head_rows_) outputs = std::max(outputs, head_row);
  std::uint64_t commit_inputs = 0;
  for (std::uint16_t stage = 0; stage < commit_places_.size(); ++stage) {
    if (commit_places_[stage].empty()) continue;
    for (std::uint32_t keep = 1; keep <= config_.max_verify_rows; ++keep) {
      pl::StepShape shape{
          .stage = stage,
          .phase = pl::Phase::kCommit,
          .segments = {{.rows = keep, .read_cells = slots_.context(stage), .want_head = false}}};
      auto p = source_->Build(shape, AddressesFor(shape, true));
      if (!p) return Error(std::format("measuring a commit: {}", p.error()));
      ObservePlanCoverage(**p);
      if (auto routed = MeasureRouting(**p); !routed) return routed;
      auto measured = ProgramEnvelope::Measure(**p, true);
      if (!measured) return Error(measured.error());
      activations = std::max(activations, measured->scratch);
      pool = std::max(pool, measured->pool);
      commit_inputs = std::max(commit_inputs, measured->inputs);
      plan_floor_bytes_ = std::max(plan_floor_bytes_, measured->host);
    }
  }
  if (auto routed = FinishRoutingSetup(); !routed) return routed;
  if (commit_inputs != 0) {
    commit_staging_stride_ = Round(commit_inputs, 256);
    auto staging = memory_.Pinned(commit_staging_stride_ * slots_.slots() * slots_.stages());
    if (!staging) return Error(staging.error());
    commit_staging_ = *staging;
  }
  // A quarter's margin over the widest measured, as every family's.
  activations_needed_ =
      Round(std::max<std::uint64_t>(activations + activations / 4, 1), kPagedExtent);
  // The launch's own part comes out of the pool before any step draws on
  // it (a small step's pool otherwise loses its margin and more to it).
  // Its reserve is set apart even when the measured steps draw nothing from
  // the pool: the launch takes it from the pool's end whatever they measure.
  pool_needed_ = pool == 0 && config_.launch_pool_reserve == 0
                     ? 0
                     : Round(pool + pool / 4 + config_.launch_pool_reserve, kPagedExtent);
  inputs_staging_bytes_ = Round(std::max<std::uint64_t>(inputs + inputs / 4, 256), 256);
  outputs_staging_bytes_ = Round(std::max<std::uint64_t>(outputs + outputs / 4, 256), 256);
  host_input_bytes_ =
      Round(inputs_staging_bytes_ + outputs_staging_bytes_ + RoutedPinnedBytes(), kPagedExtent);
  auto in = memory_.Pinned(inputs_staging_bytes_);
  if (!in) return Error(in.error());
  auto out = memory_.Pinned(outputs_staging_bytes_);
  if (!out) return Error(out.error());
  inputs_staging_ = *in;
  outputs_staging_ = static_cast<std::byte*>(*out);
  programs_.runs().SetStaging(inputs_staging_, inputs_staging_bytes_);
  if (auto carried = SetupCarry(carry); !carried) return carried;
  setup_ = true;
  return {};
}

Status ModelRunner::SetupCarry(std::span<const std::uint64_t> carry_bytes) {
  const std::size_t stages = slots_.stages();
  carry_offset_.assign(stages, 0);
  carry_bytes_.assign(stages, 0);
  std::uint64_t stride = 0;
  for (std::size_t stage = 0; stage < stages && stage < carry_bytes.size(); ++stage) {
    if (carry_bytes[stage] == 0) continue;
    carry_offset_[stage] = stride;
    carry_bytes_[stage] = Round(carry_bytes[stage], 256);
    stride += carry_bytes_[stage];
  }
  carry_stride_ = stride;
  if (stride == 0) return {};
  return memory_.Map(carry_, "the model's carried features", stride * slots_.slots(),
                     catalog::MemoryClass::kRuntime);
}

std::uint64_t ModelRunner::CarryAddress(std::uint32_t slot, std::size_t stage) const {
  return carry_.base + (std::uint64_t{slot} * carry_stride_) + carry_offset_.at(stage);
}

void ModelRunner::ForgetCarry(std::uint32_t slot) {
  for (std::size_t stage = 0; stage < slots_.stages(); ++stage) {
    const std::size_t at = (std::size_t{slot} * slots_.stages()) + stage;
    if (at < carry_rows_.size()) carry_rows_[at] = 0;
  }
}

std::expected<std::vector<RunCopy>, std::string> ModelRunner::FeatureCopies(
    const kn::StepProgram& program, const StepRequest& request) const {
  Copies copies;
  CopySink into(copies);
  if (auto emitted = EmitFeatureCopies(program, request, into); !emitted)
    return Error(emitted.error());
  return copies;
}

Status ModelRunner::EmitFeatureCopies(const kn::StepProgram& program, const StepRequest& request,
                                      CopySink& copies) const {
  const pl::StepShape& shape = request.shape;
  for (const kn::InputSlot& in : program.inputs()) {
    if (in.kind != kn::InputKind::kFeatures) continue;
    std::uint64_t rows = 0;
    for (const pl::Segment& seg : shape.segments) rows += seg.rows;
    if (in.segment != ir::kJoinedSegment || rows == 0 || in.bytes % rows != 0) {
      return Error("a step's features input is not its rows' joined features");
    }
    const std::uint64_t row_bytes = in.bytes / rows;
    std::uint64_t first = 0;
    for (std::size_t i = 0; i < shape.segments.size(); ++i) {
      const pl::Segment& seg = shape.segments[i];
      const FeatureFrom& from = request.work[i].feature;
      const std::uint64_t to = in.device_address + (first * row_bytes);
      first += seg.rows;
      switch (from.kind) {
        case FeatureFrom::Kind::kNone:
          return Error(
              std::format("slot {}'s rows read a feature its work does not name", seg.slot));
        case FeatureFrom::Kind::kTap: {
          const auto* tap = model::FeatureTap(slots_.layout(shape.stage));
          if (tap == nullptr || tap->contract.cell_bytes != row_bytes) {
            return Error(
                std::format("stage {} has no feature tap of its rows' features", shape.stage));
          }
          // Positions the target has committed and its ring still holds.
          const std::uint64_t committed = slots_.positions(seg.slot);
          const std::uint64_t last = std::uint64_t{from.position} + seg.rows;
          if (last > committed || (tap->contract.cells == model::CellRule::kPositionRing &&
                                   committed - from.position > tap->cells)) {
            return Error(std::format("slot {}'s tap holds no feature of positions {} to {}",
                                     seg.slot, from.position, last - 1));
          }
          const std::uint64_t base = slots_.live(seg.slot, shape.stage).base(0) + tap->offset;
          const TapRows held = HeldTapRows(seg.slot, shape.stage);
          if (tap->contract.cells == model::CellRule::kPerRow &&
              (held.count == 0 || std::uint64_t{from.position} + held.preceding < held.first ||
               last > std::uint64_t{held.first} + held.count)) {
            return Error("a chunk tap does not hold the requested features");
          }
          // The rows' cells are contiguous but for one wrap of a position
          // ring: a row whose cell follows the last row's extends its copy,
          // so a segment takes at most two (bits unchanged: the same bytes).
          for (std::uint32_t r = 0; r < seg.rows; ++r) {
            const std::uint64_t cell =
                tap->contract.cells == model::CellRule::kPerRow
                    ? std::uint64_t{tap->contract.row_carry} + from.position + r - held.first
                    : (std::uint64_t{from.position} + r) % tap->cells;
            const std::uint64_t dst = to + (r * row_bytes);
            const std::uint64_t src = base + (cell * tap->contract.cell_bytes);
            if (!copies.Push({dst, src, row_bytes}, r != 0))
              return Error("feature copies exceed their paid capacity");
          }
          break;
        }
        case FeatureFrom::Kind::kCarry: {
          const std::size_t at = (std::size_t{seg.slot} * slots_.stages()) + shape.stage;
          if (carry_rows_.at(at) != seg.rows ||
              seg.rows * row_bytes > carry_bytes_.at(shape.stage)) {
            return Error(std::format("slot {} carries no features of its {} rows on stage {}",
                                     seg.slot, seg.rows, shape.stage));
          }
          if (!copies.Push({to, CarryAddress(seg.slot, shape.stage), seg.rows * row_bytes}))
            return Error("carried feature copies exceed their paid capacity");
          break;
        }
      }
    }
  }
  return {};
}

std::optional<std::uint32_t> ModelRunner::OldestTapFeature(std::uint32_t slot,
                                                           std::uint16_t stage) const {
  if (slot >= slots_.slots() || stage >= slots_.stages()) return std::nullopt;
  const auto* tap = model::FeatureTap(slots_.layout(stage));
  if (tap == nullptr) return std::nullopt;
  const std::uint32_t committed = slots_.positions(slot);
  if (tap->contract.cells == model::CellRule::kPositionRing) {
    return committed > tap->cells ? static_cast<std::uint32_t>(committed - tap->cells) : 0U;
  }
  const TapRows held = HeldTapRows(slot, stage);
  if (held.count == 0) return std::nullopt;
  return held.first > held.preceding ? held.first - held.preceding : 0U;
}

SlotTapRows ModelRunner::SaveTapRows(std::uint32_t slot) const {
  SlotTapRows saved;
  if (slot >= slots_.slots() || slots_.stages() > SlotTapRows::kStages) return saved;
  saved.stages = static_cast<std::uint32_t>(slots_.stages());
  saved.positions = slots_.positions(slot);
  for (std::size_t stage = 0; stage < slots_.stages(); ++stage) {
    const std::size_t at = (std::size_t{slot} * slots_.stages()) + stage;
    saved.held[stage] = tap_rows_.at(at);
    saved.previous[stage] = previous_tap_rows_.at(at);
  }
  return saved;
}

Status ModelRunner::RestoreTapRows(std::uint32_t slot, const SlotTapRows& saved) {
  if (slot >= slots_.slots() || saved.stages != slots_.stages() ||
      saved.positions != slots_.positions(slot))
    return Error("tap rows were recorded for another slot state");
  for (std::size_t stage = 0; stage < saved.stages; ++stage) {
    const std::size_t at = (std::size_t{slot} * slots_.stages()) + stage;
    tap_rows_.at(at) = saved.held[stage];
    previous_tap_rows_.at(at) = saved.previous[stage];
  }
  return {};
}

bool ModelRunner::ReadsHistory(std::uint16_t stage) const {
  if (stage < stage_facts_.size() ? stage_facts_[stage].uses_side_rows
                                  : source_->UsesSideRows(stage))
    return true;
  return std::ranges::any_of(row_tables_, [stage](const auto& table) {
    return table && table->config.weight.stage == stage;
  });
}

TapRows ModelRunner::HeldTapRows(std::uint32_t slot, std::size_t stage) const {
  const std::size_t at = (std::size_t{slot} * slots_.stages()) + stage;
  const std::uint32_t committed = slots_.positions(slot);
  TapRows held = tap_rows_.at(at);
  // A discarded verify restores the previous chunk's cells. Its positions
  // never advanced; an accepted prefix keeps the new chunk's first rows.
  if (held.count != 0 && committed <= held.first) held = previous_tap_rows_.at(at);
  if (committed < held.first) return {};
  held.count = std::min(held.count, committed - held.first);
  return held;
}

std::expected<std::vector<RunCopy>, std::string> ModelRunner::TapCarries(
    const pl::StepShape& shape) const {
  Copies copies;
  CopySink into(copies);
  if (auto emitted = EmitTapCarries(shape, into); !emitted) return Error(emitted.error());
  return copies;
}

Status ModelRunner::EmitTapCarries(const pl::StepShape& shape, CopySink& copies) const {
  if (shape.stage != 0) return {};
  for (const pl::Segment& seg : shape.segments) {
    for (std::size_t stage = 1; stage < slots_.stages(); ++stage) {
      const auto* tap = model::FeatureTap(slots_.layout(stage));
      if (tap == nullptr || tap->contract.cells != model::CellRule::kPerRow) continue;
      const TapRows held = HeldTapRows(seg.slot, stage);
      const std::uint32_t count = std::min(held.count, tap->contract.row_carry);
      if (count == 0 || std::uint64_t{held.first} + held.count != seg.past) continue;
      const std::uint64_t bytes = tap->contract.cell_bytes;
      const std::uint64_t base = slots_.live(seg.slot, stage).base(0) + tap->offset;
      if (!copies.Push({base + ((tap->contract.row_carry - count) * bytes),
                        base + ((tap->contract.row_carry + held.count - count) * bytes),
                        count * bytes}))
        return Error("tap carries exceed their paid capacity");
    }
  }
  return {};
}

std::expected<std::vector<RunCopy>, std::string> ModelRunner::CarryCopies(
    const kn::StepProgram& program, const pl::StepShape& shape) const {
  Copies copies;
  CopySink into(copies);
  if (auto emitted = EmitCarryCopies(program, shape, into); !emitted) return Error(emitted.error());
  return copies;
}

Status ModelRunner::EmitCarryCopies(const kn::StepProgram& program, const pl::StepShape& shape,
                                    CopySink& copies) const {
  for (const kn::OutputSlot& out : program.outputs()) {
    if (!out.features) continue;
    if (shape.stage == 0 || out.segment >= shape.segments.size() ||
        out.bytes > carry_bytes_.at(shape.stage)) {
      return Error("a step's next features pass their carry");
    }
    if (!copies.Push({CarryAddress(shape.segments[out.segment].slot, shape.stage),
                      out.device_address, out.bytes}))
      return Error("output carries exceed their paid capacity");
  }
  return {};
}

Status ModelRunner::SetupSnapshots() {
  const std::size_t stages = slots_.stages();
  commits_.assign(std::size_t{slots_.slots()} * stages, false);
  if (config_.max_verify_rows == 0) return {};
  // Each slot's stages a verify saves in or commits: its saves at most
  // (model::VerifySaveBound), at 256-byte places of one mapping.
  struct Place {
    std::uint32_t slot = 0;
    std::size_t stage = 0;
    std::uint64_t offset = 0, bytes = 0;
    std::uint32_t ranges = 0;
    LiveState::Commit commit;
  };
  std::vector<Place> places;
  std::uint64_t total = 0;
  for (std::uint32_t slot = 0; slot < slots_.slots(); ++slot) {
    for (std::size_t stage = 0; stage < stages; ++stage) {
      LiveState::Commit commit = config_.commit ? config_.commit(slot, stage) : LiveState::Commit{};
      if (!commit_places_[stage].empty()) {
        if (commit) return Error("a stage has both a component commit and a caller's commit");
        commit = [this, slot, stage](providers::NativeStream stream, std::uint32_t) {
          return QueueCommit(slot, static_cast<std::uint16_t>(stage), stream);
        };
      }
      commits_[(std::size_t{slot} * stages) + stage] = static_cast<bool>(commit);
      if (!Verifies(slot, stage)) continue;
      auto bound =
          model::VerifySaveBound(slots_.layout(stage), config_.max_verify_rows,
                                 {.commits = static_cast<bool>(commit), .drafter = stage != 0});
      if (!bound) {
        return Error(std::format("a verify of {} rows: stage {}: {}", config_.max_verify_rows,
                                 stage, bound.error()));
      }
      places.push_back({.slot = slot,
                        .stage = stage,
                        .offset = total,
                        .bytes = bound->bytes,
                        .ranges = bound->ranges,
                        .commit = std::move(commit)});
      total += Round(std::max<std::uint64_t>(bound->bytes, 256), 256);
    }
  }
  if (total == 0) return {};
  if (auto mapped = memory_.Map(snapshots_, "the model's verify snapshots", total,
                                catalog::MemoryClass::kRuntime);
      !mapped) {
    return mapped;
  }
  for (Place& place : places) {
    LiveState& live = slots_.live(place.slot, place.stage);
    live.SnapshotAt(snapshots_.base + place.offset, place.bytes);
    if (auto allocated = live.AllocateSnapshot(memory_, std::max<std::uint32_t>(place.ranges, 1));
        !allocated) {
      return allocated;
    }
    if (place.commit) {
      std::function<Status(std::uint32_t)> prepare;
      if (!commit_places_[place.stage].empty()) {
        prepare = [this, slot = place.slot, stage = place.stage](std::uint32_t keep) {
          return PrepareCommit(slot, static_cast<std::uint16_t>(stage), keep);
        };
      }
      live.SetCommit(std::move(place.commit), std::move(prepare));
    }
  }
  return {};
}

Status ModelRunner::SetupCommits() {
  commit_places_.resize(slots_.stages());
  prepared_commits_.resize(std::size_t{slots_.slots()} * slots_.stages());
  if (config_.max_verify_rows == 0) return {};
  std::uint64_t bytes = 0;
  for (std::uint16_t stage = 0; stage < slots_.stages(); ++stage) {
    const auto facts = source_->StageFacts(stage);
    auto buffers =
        facts.commit_buffers
            ? facts.commit_buffers(config_.max_verify_rows)
            : std::expected<std::vector<CommitBuffer>, std::string>(std::vector<CommitBuffer>{});
    if (!buffers)
      return Error(std::format("stage {}'s commit buffers: {}", stage, buffers.error()));
    for (CommitBuffer& buffer : *buffers) {
      if (buffer.row_bytes == 0 || buffer.what.empty()) {
        return Error("a commit buffer has no rows or name");
      }
      const auto duplicate = std::ranges::any_of(commit_places_[stage], [&](const auto& place) {
        return place.buffer.layer == buffer.layer && place.buffer.component == buffer.component &&
               place.buffer.what == buffer.what;
      });
      if (duplicate) return Error("a commit buffer is repeated");
      const std::uint64_t rows = std::uint64_t{config_.max_verify_rows} * slots_.slots();
      if (buffer.row_bytes > (std::numeric_limits<std::uint64_t>::max() - bytes - 255) / rows) {
        return Error("the commit buffers exceed their address space");
      }
      commit_places_[stage].push_back({.buffer = buffer, .offset = bytes});
      bytes += Round(buffer.row_bytes * rows, 256);
    }
  }
  if (bytes == 0) return {};
  return memory_.Map(commit_saves_, "the model's commit intermediates", bytes,
                     catalog::MemoryClass::kLiveState);
}

Status ModelRunner::SetupRowTables() {
  // Validate the complete multipart declaration before making any table's
  // storage, landing or device slots. Matrix-wide scales remain ordinary weights.
  for (const auto& config : config_.row_tables) {
    if (config.weight.stage >= config_.weights.size() || config.weight.expert_array ||
        config.rows_per_token == 0 || config.format.shape.rank != 2)
      return Error("a row table's weights, shape or lookups");
    const auto parts = format::PartsOf(config.format);
    if (!parts || !format::RowBytes(config.format, config.part))
      return Error("a row table requires a row-addressable format part");
    for (const auto& other : config_.row_tables) {
      if (other.weight != config.weight) continue;
      if (other.format != config.format || other.rows_per_token != config.rows_per_token ||
          other.source_parts != config.source_parts)
        return Error("a row table's parts disagree on logical geometry");
    }
    for (const auto& part : parts->parts()) {
      if (!format::RowBytes(config.format, part.role)) continue;
      if (std::ranges::count_if(config_.row_tables, [&](const auto& other) {
            return other.weight == config.weight && other.part == part.role;
          }) != 1)
        return Error("a row table has missing or duplicate row parts");
    }
    const auto& weights = config_.weights[config.weight.stage];
    const auto& artifact = weights.weights->artifact();
    if (config.resource >= artifact.resources().size()) return Error("a row table's resource");
    const auto& resource = artifact.resources()[config.resource];
    if (auto valid = CheckRowTableSource(config, artifact.format_version(), artifact.resources());
        !valid)
      return valid;
    const auto part = parts->Find(config.part);
    if (!part || part->bytes != resource.bytes.value() || resource.group >= weights.places.size() ||
        weights.places[resource.group] != GroupPlace::kNone)
      return Error("a row table's part extent or nonresident group");
    for (std::uint32_t i = 0; i < artifact.resources().size(); ++i)
      if (i != config.resource && artifact.resources()[i].group == resource.group)
        return Error("a row-paged table shares its group with another resource");
    const auto row_bytes = format::RowBytes(config.format, config.part);
    if (!row_bytes || *row_bytes > std::numeric_limits<std::uint32_t>::max())
      return Error("a row table's row size");
    std::uint64_t capacity = 0;
    for (const auto& shape : config_.setup_shapes) {
      if (shape.stage != config.weight.stage) continue;
      std::uint64_t count = 0;
      if (__builtin_mul_overflow(std::uint64_t{shape.rows()}, config.rows_per_token, &count))
        return Error("a row table's capacity overflows");
      capacity = std::max(capacity, count);
    }
    if (capacity == 0 ||
        capacity > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max()))
      return Error("a row table's lookup count exceeds its slot IDs");
    const auto first =
        artifact::ChunkRangeOf(artifact.layout(), {.group = resource.group, .chunk = 0});
    if (!first) return Error("a row table's file range");
    std::uint64_t offset = 0, end = 0;
    if (__builtin_add_overflow(first->file_offset.value(), resource.offset.value(), &offset) ||
        __builtin_add_overflow(first->file_offset.value(),
                               artifact.groups()[resource.group].stored.value(), &end))
      return Error("a row table's file range overflows");
    const auto valid = PlanPleRows({.file_offset = offset,
                                    .rows = config.format.shape.ne[1],
                                    .row_bytes = *row_bytes,
                                    .chunk_file_offset = first->file_offset.value(),
                                    .file_bytes = end},
                                   {}, PleLandingBound(capacity), capacity);
    if (!valid) return Error(valid.error());
  }
  for (const auto& config : config_.row_tables) {
    auto& weights = config_.weights[config.weight.stage];
    const auto& artifact = weights.weights->artifact();
    const auto& resource = artifact.resources()[config.resource];
    const auto& group = artifact.groups()[resource.group];
    const auto row_bytes = *format::RowBytes(config.format, config.part);
    const auto first =
        *artifact::ChunkRangeOf(artifact.layout(), {.group = resource.group, .chunk = 0});
    auto table = std::make_unique<RowTableState>();
    table->config = config;
    table->shard = first.shard;
    table->table = {.file_offset = first.file_offset.value() + resource.offset.value(),
                    .rows = config.format.shape.ne[1],
                    .row_bytes = row_bytes,
                    .chunk_file_offset = first.file_offset.value(),
                    .file_bytes = first.file_offset.value() + group.stored.value()};
    for (const auto& shape : config_.setup_shapes) {
      if (shape.stage != config.weight.stage) continue;
      table->capacity =
          std::max(table->capacity, std::uint64_t{shape.rows()} * config.rows_per_token);
    }
    if (config_.row_storage) {
      auto ring = config_.row_storage(64);
      if (!ring || *ring == nullptr)
        return Error(std::format("row-table storage: {}", ring ? "none given" : ring.error()));
      table->storage = std::move(*ring);
    } else {
      auto ring = providers::OpenStorage(64);
      if (!ring) return Error(std::format("row-table storage: {}", ring.error().message()));
      table->storage = std::move(*ring);
    }
    row_tables_.push_back(std::move(table));
    auto& held = *row_tables_.back();  // stable through partial setup and release
    std::uint64_t slot_bytes = 0, source_bytes = 0;
    if (__builtin_mul_overflow(held.capacity, row_bytes, &slot_bytes) ||
        __builtin_mul_overflow(held.capacity, std::uint64_t{sizeof(std::uint64_t)},
                               &source_bytes) ||
        __builtin_add_overflow(source_bytes, std::uint64_t{sizeof(std::uint32_t)}, &source_bytes))
      return Error("a row table's allocation size overflows");
    if (auto mapped =
            memory_.Map(held.slots, "row-table slots", slot_bytes, catalog::MemoryClass::kWeights);
        !mapped)
      return mapped;
    auto landing = memory_.PinnedAligned(PleLandingBound(held.capacity), kPleBlock);
    if (!landing) return Error(landing.error());
    held.landing_allocation = landing->allocation;
    held.landing = landing->data;
    if (Address(held.landing) % kPleBlock != 0)
      return Error("a row table's landing is not aligned");
    auto sources = memory_.Pinned(source_bytes);
    if (!sources) return Error(sources.error());
    held.sources = static_cast<std::uint64_t*>(*sources);
    held.count = reinterpret_cast<std::uint32_t*>(held.sources + held.capacity);
    *held.count = 0;
  }
  return {};
}

Status ModelRunner::PrepareCommit(std::uint32_t slot, std::uint16_t stage, std::uint32_t keep) {
  if (keep == 0 || keep > config_.max_verify_rows) return Error("a commit's kept rows");
  pl::StepShape shape{.stage = stage,
                      .phase = pl::Phase::kCommit,
                      .segments = {{.slot = static_cast<std::uint8_t>(slot),
                                    .rows = keep,
                                    .read_cells = slots_.context(stage),
                                    .want_head = false}}};
  auto entry = Planned(shape);
  if (!entry) return Error(entry.error());
  const std::size_t at = (std::size_t{slot} * slots_.stages()) + stage;
  StepRuns staging(false);
  staging.SetStaging(static_cast<std::byte*>(commit_staging_) + (at * commit_staging_stride_),
                     commit_staging_stride_);
  const auto& program = *(*entry)->planned->program;
  auto copies = staging.Stage(
      program.inputs(),
      [&](std::size_t i, std::span<std::byte> out) {
        return step_inputs::WriteComputed(program.inputs()[i], shape, out);
      },
      0);
  if (!copies) return Error(copies.error());
  const bool capture =
      (*entry)->runs[0].CaptureDue(programs_.runs().graphs()) && plans_.ChargeGraph(**entry);
  prepared_commits_[at] = {.plan = *entry, .copies = std::move(*copies), .capture = capture};
  return {};
}

LiveState::Queued ModelRunner::QueueCommit(std::uint32_t slot, std::uint16_t stage,
                                           providers::NativeStream stream) {
  PreparedCommit& prepared = prepared_commits_.at((std::size_t{slot} * slots_.stages()) + stage);
  if (prepared.plan == nullptr) {
    return std::unexpected(LiveState::QueueFailure{.detail = "a commit was not prepared"});
  }
  Plans::Entry& entry = *prepared.plan;
  auto& run = entry.runs[0];
  StepRuns commits(programs_.runs().graphs());
  const std::size_t at = (std::size_t{slot} * slots_.stages()) + stage;
  commits.SetStaging(static_cast<std::byte*>(commit_staging_) + (at * commit_staging_stride_),
                     commit_staging_stride_);
  commits.SetLaunch(programs_.launch());
  const StepQueued queued = commits.Queue(run, prepared.copies, {}, *entry.planned->program, {},
                                          prepared.capture, graph_stats_, stream);
  prepared.plan = nullptr;
  if (!queued.result) {
    return std::unexpected(LiveState::QueueFailure{
        .unknown = queued.before || queued.result.error().error == kn::StepError::kUnknown,
        .detail = queued.result.error().detail});
  }
  return {};
}

Status ModelRunner::Register() {
  if (!setup_ || registered_ || released_) return Error("model runner registration order");
  for (StageWeights& w : config_.weights) {
    if (auto r = w.weights->Register(node_, owner_); !r) return r;
  }
  if (auto r = slots_.Register(); !r) return r;
  std::vector<catalog::ExtentId> pinned = weight_extents();
  const auto reserved = slots_.reserved_extents();
  pinned.insert(pinned.end(), reserved.begin(), reserved.end());
  if (auto r = node_.scheduler().PinPlaces(pinned); !r) {
    return Error(std::format("pinning the runner's places: {}", sc::ToString(r.error())));
  }
  registered_ = true;
  return {};
}

Status ModelRunner::Bind() {
  if (!registered_ || bound_ || released_) return Error("model runner bind order");
  auto launch = config_.launch(pool_bytes());
  if (!launch) return Error(launch.error());
  if (*launch == nullptr) return Error("the launch is missing");
  if (auto bound = programs_.BindLaunch(std::move(*launch)); !bound) return bound;
  plans_.set_account(&programs_.account());
  key_base_ = source_->KeyBase();
  key_base_.shape = {};
  if (auto current = CheckRoutingSource(); !current) return current;
  std::vector<catalog::ExtentId> shared = base_weight_extents();
  // Its own prefix of the shared workspace, not the whole: what another
  // model needs past it is not this model's to hold (a pageable workspace
  // evicts it while this model runs).
  for (const auto& [mapped, bytes] : {std::pair{&node_.activations(), activation_bytes()},
                                      std::pair{&node_.pool(), pool_bytes()}}) {
    const auto count = std::min<std::size_t>(mapped->extents.size(),
                                             static_cast<std::size_t>(bytes / kPagedExtent));
    shared.insert(shared.end(), mapped->extents.begin(),
                  mapped->extents.begin() + static_cast<std::ptrdiff_t>(count));
  }
  const auto own = memory_.extents();
  shared.insert(shared.end(), own.begin(), own.end());
  std::vector<const PagedWeights*> weights;
  for (const StageWeights& w : config_.weights) weights.push_back(w.weights.get());
  if (auto r = slots_.Bind(std::move(shared), std::move(weights)); !r) return r;
  bound_ = true;
  return {};
}

bool ModelRunner::flights_unproven() const noexcept {
  return routed_flight_.held() || (step_flight_.held() && step_flight_->unproven());
}

bool ModelRunner::unproven() const noexcept {
  return flights_unproven() || std::ranges::any_of(row_tables_, [](const auto& table) {
           return table && table->reads_unproven;
         });
}

bool ModelRunner::SettleRowReads(RowTableState& table) {
  if (!table.reads_unproven) return true;
  if (!table.storage) {
    table.reads_unproven = false;
    return true;
  }
  // Polled, never waited on: each completion is a stalled read's own end,
  // and its token an earlier plan's (ReadPleRows restarts them at 0), so it
  // is discarded before the ring is used again.
  std::array<providers::IoCompletion, 64> done{};
  while (table.storage->Harvest(done, false) != 0) {
  }
  if (table.storage->in_flight() != 0) return false;
  table.reads_unproven = false;
  return true;
}

void ModelRunner::ClearAfterFence(const FenceProof& proof) {
  step_flight_.Clear(proof);
  routed_flight_.Clear(proof);
  for (auto& table : row_tables_) {
    if (table) (void)SettleRowReads(*table);
  }
}

Status ModelRunner::RetireAfterFence(const FenceProof& proof) {
  // PagedNode alone authenticates completed stream fences. Internal flights
  // may now release their unknown-work cycles; external views keep the actual
  // environment alive and refuse physical teardown until their last drop.
  closing_ = true;
  if (programs_.launch()) programs_.launch()->Retire();
  ClearAfterFence(proof);
  if (resource_leases_.load(std::memory_order_acquire) != 0)
    return Error("private resource views still retain this runner's launch and backing");
  return {};
}

Status ModelRunner::AcknowledgeFence(const FenceProof& proof) {
  // As RetireAfterFence, without closing: a recovery's fence proved that the
  // work an unproven step or route left queued has ended, so their owners
  // go, and the plans that work may have used are dropped with them.
  if (released_ || closing_) return {};
  if (batch_active_) return Error("a private batch still holds the runner after its fence");
  ClearAfterFence(proof);
  DropPlans();
  return {};
}

Status ModelRunner::Release() {
  if (released_) return {};
  if (resource_leases_.load(std::memory_order_acquire) != 0)
    return Error("private resource views still retain this runner's launch and backing");
  if (lifetime_ && !closing_)
    return Error("a retained runner requires node-authenticated drain before release");
  // Release's contract is node teardown after its stream fences, which
  // cleared the flights with their proof (RetireAfterFence); a flight still
  // unproven here was never fenced, and keeps everything it uses.
  if (flights_unproven())
    return Error("an unproven step or route retains the runner until its stream is fenced");
  released_ = true;
  // A ring's stalled reads that landed since are harvested: then it goes.
  for (auto& table : row_tables_) {
    if (table) (void)SettleRowReads(*table);
  }
  // The recordings name the launch and the memory: they go first (D-090).
  DropPlans();
  programs_.ReleaseAfterDrain();
  // What used the kernels' memory goes before it.
  if (prepared_ && config_.retire) config_.retire();
  std::vector<std::string> problems;
  for (auto& table : row_tables_) {
    const std::array<void*, 1> landings{table->landing_allocation};
    if (node_.RetireRing(std::move(table->storage), landings)) {
      problems.emplace_back("row-table reads remain in flight; their landing and ring are kept");
    }
  }
  memory_.ReleaseMappings(problems);
  slots_.Release(node_.memory(), problems);
  for (StageWeights& w : config_.weights) {
    if (auto r = w.weights->Release(node_.memory()); !r) problems.push_back(r.error());
  }
  auto result = support::Joined(problems);
  if (result) RetireRoot();  // Last: may destroy the adapter-less node-owned root.
  return result;
}

// --------------------------------------------------------------------- plans

void ModelRunner::DropPlans() {
  if (!batch_active_ && !flights_unproven()) {
    ForgetCompletedRouting();
    plans_.Clear();
  }
}

void ModelRunner::ReclaimCandidates(std::uint32_t owner, bool running,
                                    std::vector<memory::ReclaimCandidate>& out) {
  if (released_ || slots_.faulted() || batch_active_) return;
  std::array<PlanCacheBase*, 1> caches{&plans_};
  CollectPlans(caches, owner, running, out);
}

std::uint64_t ModelRunner::Reclaim(memory::ReclaimKind kind, std::uint64_t id) {
  if (released_ || slots_.faulted() || batch_active_ || flights_unproven()) return 0;
  std::array<PlanCacheBase*, 1> caches{&plans_};
  ForgetCompletedRouting();
  return ReclaimPlan(caches, kind, id);
}

Status ModelRunner::CheckPorts(const pl::StepShape& shape, const kn::StepProgram& bound) const {
  // Every lowered port belongs to this token shape; typed image ports use
  // their own graph authority at the same common installation seam.
  const auto segment_ok = [&](std::uint8_t segment) {
    return segment == ir::kJoinedSegment || segment < shape.segments.size();
  };
  for (const auto& in : bound.inputs()) {
    if (!segment_ok(in.segment)) return Error("a step's input names no segment of its shape");
  }
  for (const auto& out : bound.outputs()) {
    if (!segment_ok(out.segment)) return Error("a step's output names no segment of its shape");
  }
  return {};
}

std::expected<ModelRunner::Plans::Entry*, std::string> ModelRunner::Install(
    const pl::StepShape& shape, std::unique_ptr<StepPlanned> planned, double seconds,
    const std::function<void()>& transfer) {
  if (planned == nullptr || planned->program == nullptr) return Error("no program to install");
  kn::StepProgram& program = *planned->program;
  if (auto routed = CheckRouting(program); !routed) return Error(routed.error());
  auto checked = programs_.BindPrepared(
      *planned, [&](const kn::StepProgram& bound) -> Status { return CheckPorts(shape, bound); });
  if (!checked) return Error(checked.error());
  const std::uint64_t host = program.host_bytes();
  const std::uint64_t launches = program.launches();
  auto key = KeyOf(shape);
  if (!key) return Error(key.error());
  if (transfer) transfer();
  ForgetCompletedRouting();
  Plans::Entry& entry = plans_.Add(std::move(*key), std::move(planned), host, launches, seconds);
  if (lookup_key_.shape.segments.capacity() < entry.key.shape.segments.size())
    lookup_key_.shape.segments.reserve(entry.key.shape.segments.size());
  if (lookup_key_.parameters.capacity() < entry.key.parameters.size())
    lookup_key_.parameters.reserve(entry.key.parameters.size());
  // Add's required PlanAccount charge owns all returned program storage before
  // the source's transient compiler use can be returned to another builder.
  entry.planned->program->ConstructionPaid();
  coverage_.tensors += checked->tensors;
  EmitPlanReport(entry, config_.plan_report_sink);
  // Never recorded: no capture beside a run, on its first run or ahead.
  if (entry.planned->positioned) entry.runs[0].uncapturable = true;
  return &entry;
}

std::expected<ModelRunner::Plans::Entry*, std::string> ModelRunner::Planned(
    const pl::StepShape& shape) {
  if (auto identity = source_->ExecutionIdentity(); !identity) return Error(identity.error());
  if (auto current = CheckRoutingSource(); !current) return Error(current.error());
  if (auto filled = FillKey(shape, lookup_key_); !filled) return Error(filled.error());
  if (Plans::Entry* found = plans_.Find(lookup_key_); found != nullptr) return found;
  const auto started = std::chrono::steady_clock::now();
  // Construct over owned stable virtual places. This is not residency or
  // permission to run: the exact program's requirements are admitted below,
  // before binding, state preparation, capture, or submission.
  auto addresses = AddressesFor(shape, false);
  addresses.activation_bytes = node_.activations().capacity_bytes;
  auto program = source_->Build(shape, addresses);
  if (!program) return Error(std::format("planning a step: {}", program.error()));
  if (*program == nullptr) return Error("planning a step gave no program");
  auto envelope = ProgramEnvelope::Measure(**program);
  if (!envelope) return Error(envelope.error());
  const auto pool = pool_bytes();
  if (config_.launch_pool_reserve > pool || envelope->pool > pool - config_.launch_pool_reserve)
    return Error("the selected step exceeds its fixed launch pool capacity");
  if (envelope->inputs > inputs_staging_bytes_ || envelope->outputs > outputs_staging_bytes_)
    return Error("the selected step exceeds its fixed host staging capacity");
  if (envelope->scratch > activation_bytes()) {
    if (envelope->scratch > node_.activations().capacity_bytes)
      return Error("the selected step exceeds its activation virtual reservation");
    const auto needed = Round(envelope->scratch, kPagedExtent);
    if (auto grown = node_.GrowWorkspace(needed, node_.pool().bytes); !grown)
      return Error(grown.error());
    const auto count = static_cast<std::size_t>(needed / kPagedExtent);
    bool over_budget = false;
    if (auto held = slots_.TryExtendShared(std::span(node_.activations().extents).first(count),
                                           &over_budget);
        !held)
      return Error(held.error());
    // Only a positively admitted prefix becomes this model's high water.
    // Existing plans keep their places and the launch/pool never changes.
    activations_needed_ = needed;
  }
  auto planned = std::make_unique<StepPlanned>();
  planned->program = std::move(*program);
  return Install(shape, std::move(planned),
                 support::Seconds(std::chrono::steady_clock::now() - started), {});
}

std::expected<Copies, std::string> ModelRunner::OutputCopies(const kn::StepProgram& program) const {
  return OutputCopies(program, outputs_staging_, outputs_staging_bytes_);
}
std::expected<Copies, std::string> ModelRunner::OutputCopies(const kn::StepProgram& program,
                                                             std::byte* staging,
                                                             std::uint64_t bytes) const {
  Copies copies;
  copies.reserve(program.outputs().size());
  std::uint64_t at = 0;
  for (const kn::OutputSlot& slot : program.outputs()) {
    if (at > bytes || slot.bytes > bytes - at) {
      return Error("a step's outputs exceed their staging");
    }
    copies.push_back({Address(staging + at), slot.device_address, slot.bytes});
    at += Round(slot.bytes, 256);
  }
  return copies;
}

// --------------------------------------------------------------------- steps

Status ModelRunner::PrepareStep(const pl::StepShape& shape) {
  if (!bound_ || released_ || slots_.faulted()) return Error("the model runner is not usable");
  if (auto checked = CheckShape(shape); !checked) return checked;
  auto prepared = Planned(shape);
  if (!prepared) return Error(prepared.error());
  return {};
}

std::expected<void, StepRefusal> ModelRunner::CheckAdmission(
    const pl::Segment& segment, std::uint16_t stage, bool advance,
    const StepAuthorization* authorization) {
  if (!bound_ || released_ || closing_) return Refused("the model runner is not bound");
  const std::uint32_t slot = segment.slot;
  if (slot >= slots_.slots()) return Refused(std::format("no slot {}", slot));
  if (stage >= slots_.stages()) return Refused(std::format("no stage {}", stage));
  const bool exact =
      authorization && authorization->runner_ == this && authorization->slot_ == slot &&
      authorization->stage_ == stage && authorization->past_ == segment.past &&
      authorization->rows_ == segment.rows && authorization->outputs_ == segment.outputs &&
      authorization->want_head_ == segment.want_head && authorization->greedy_ == segment.greedy;
  if (segment.origin_phase) {
    if (*segment.origin_phase != pl::Phase::kDecode || stage != 0 || !advance ||
        segment.rows != 1 || !segment.want_head || segment.outputs != 1 || !exact ||
        authorization->part_ == StepAuthorization::Part::kNone ||
        authorization->phase_ != *segment.origin_phase) {
      return Refused("mixed causal origin requires its exact decode execution permit", true, slot);
    }
  }
  if (auto active = slots_.CheckActive(slot); !active) {
    // Compilation/admission may have exposed loss of the shared request.
    // Recheck canonical cohort authority before inviting any clean retry.
    slots_.CheckFailedJob();
    if (slots_.faulted()) return Refused(active.error(), false);
    return Refused(active.error(), true, slot);
  }
  const std::uint32_t positions = slots_.positions(slot);
  if (stage == 0) {
    // The target's rows start at the committed positions, a verify's (at
    // its anchor, advancing nothing) as a decode's.
    if (segment.past != positions) {
      return Refused(
          std::format("slot {} holds {} positions, not {}", slot, positions, segment.past), true,
          slot);
    }
  } else {
    // A drafter's: never committed, and within one chunk bound of its
    // stage past the committed positions (a draft's proposals); from
    // wherever its catch-up starts, but reaching them: a catch-up runs
    // through the committed positions, a pass sits at or past them (an
    // anchored one at them), so a step wholly behind them (the last
    // committed position not among its rows or before them) re-runs
    // positions its state already holds and drafts from none.
    if (advance) return Refused("a drafter's step never commits positions", true, slot);
    if (std::uint64_t{segment.past} + segment.rows < positions) {
      return Refused(std::format("slot {}'s draft rows end at {}, behind its {} committed "
                                 "positions",
                                 slot, std::uint64_t{segment.past} + segment.rows, positions),
                     true, slot);
    }
    if (std::uint64_t{segment.past} + segment.rows >
        std::uint64_t{positions} + slots_.layout(stage).max_rows) {
      return Refused(std::format("slot {}'s draft rows end past {} positions and its stage's {} "
                                 "rows",
                                 slot, positions, slots_.layout(stage).max_rows),
                     true, slot);
    }
    // Its view of the target's state is read only while that is frozen.
    const bool ordered_borrow = exact && authorization->borrowed_ &&
                                slots_.CheckBorrowPermit(*authorization->borrowed_, slot, stage,
                                                         segment.past, segment.rows);
    if (!slots_.layout(stage).borrowed.empty() && !slots_.borrowed(slot, 0) && !ordered_borrow) {
      return Refused(std::format("slot {}'s drafter reads the target's state, which is not "
                                 "borrowed",
                                 slot),
                     true, slot);
    }
  }
  return {};
}

std::expected<void, StepRefusal> ModelRunner::Admit(const pl::Segment& segment, std::uint16_t stage,
                                                    std::uint32_t read_align, bool advance,
                                                    const StepAuthorization* authorization) {
  if (auto ready = CheckAdmission(segment, stage, advance, authorization); !ready) return ready;
  const auto slot = segment.slot;
  // Backed through every cell the step reads, not only its own rows' (a
  // common read reaches the widest owner's cells): reading absent backing
  // is a fault. Only this growth's refusal says capacity: an earlier one's
  // flag would mislabel a refusal for a borrow, a verify or a restore.
  slots_.ForgetRefusal(slot);
  if (auto reserved = slots_.ReserveStateThrough(
          slot, std::max(segment.past + segment.rows, segment.read_cells), read_align, stage);
      !reserved) {
    // A clean capacity refusal leaves the slot usable and retryable, as
    // does a refusal before any growth (borrowed, a verify pending); a
    // growth failure that quarantined it is not clean. Never its peers.
    const bool refused = slots_.state_refused(slot);
    return Refused(reserved.error(), refused || slots_.usable(slot), slot, refused);
  }
  return {};
}

std::expected<StepReport, StepRefusal> ModelRunner::RunStep(const StepRequest& request) {
  if (batch_active_ && request.batch.data_ != batch_active_)
    return Refused("the runner is inside another owned complete batch");
  if (step_flight_.held()) return Refused("a step retains unproven effects", false);
  auto transaction = StepTransaction::Begin(*this, request);
  if (!transaction) return std::unexpected(std::move(transaction.error()));
  step_flight_.Retain(std::move(*transaction));
  auto result = step_flight_->Run(request);
  if (step_flight_->releasable())
    step_flight_.ReleaseRetired();
  else if (!result)
    result.error().clean = false;
  return result;
}

// -------------------------------------------------------------------- report

void ModelRunner::ObservePlanCoverage(const kernels::StepProgram& program) {
  ++measured_plan_shapes_;
  kernels::ObservePlanCoverage(setup_plan_coverage_, program);
}

std::string ModelRunner::plan_report() const {
  return kernels::FormatPlanCoverageReport(measured_plan_shapes_, setup_plan_coverage_);
}

void ModelRunner::EmitPlanReport(const Plans::Entry& entry,
                                 const kernels::PlanRecordSink& sink) const {
  if (!sink) return;
  // Serialize only on request, streaming the paid typed inventory without
  // allocating a second copy of the complete report.
  std::array<char, 512> header{};
  const auto& shape = entry.key.shape;
  const auto written = std::format_to_n(
      header.data(), header.size(),
      R"({{"kind":"installed_plan","id":{},"stage":{},"phase":{},"rows":{},"segments":{},"host_bytes":{}}})",
      entry.serial, shape.stage, static_cast<unsigned>(shape.phase), shape.rows(),
      shape.segments.size(), entry.host_bytes);
  base::Check(static_cast<std::size_t>(written.size) <= header.size(),
              "bounded installed-plan report header");
  sink(kernels::PlanReportEvent::kBegin, {});
  sink(kernels::PlanReportEvent::kFragment,
       std::string_view(header.data(), static_cast<std::size_t>(written.size)));
  sink(kernels::PlanReportEvent::kEnd, {});
  entry.planned->program->ReportPlanRecords(sink);
}

void ModelRunner::ReportPlans(const kernels::PlanRecordSink& sink) const {
  if (!sink) return;
  plans_.ForEach([&](const Plans::Entry& entry) { EmitPlanReport(entry, sink); });
}

std::string ModelRunner::report_json() const {
  return std::format(
      R"({{"plans":{},"graphs":{},"plan_host_bytes":{},"graph_bytes":{},)"
      R"("reclaimed_plans":{},"reclaimed_graphs":{},)"
      R"("runs":{{"eager":{},"captured":{},"replayed":{},"refused":{}}},)"
      R"("coverage":{{"tensors":{},"violations":{}}},)"
      R"("lookahead":{{"attempted":{},"built":{},"cached":{},"refused":{},)"
      R"("captured_ahead":{},"captured_first":{},"dropped_ahead":{},"prepared":{}}},)"
      R"("activations":{},"pool":{},"host_inputs":{},"plan_floor":{}}})",
      plans_.size(), plans_.graphs(), plans_.host_bytes(), plans_.graph_bytes(),
      plans_.reclaimed_plans(), plans_.reclaimed_graphs(), graph_stats_.eager,
      graph_stats_.captured, graph_stats_.replayed, graph_stats_.refused, coverage_.tensors,
      coverage_.violations, lookahead_.attempted, lookahead_.built, lookahead_.cached,
      lookahead_.refused, lookahead_.captured_ahead, lookahead_.captured_first,
      lookahead_.dropped_ahead, lookahead_.prepared, activations_needed_, pool_needed_,
      host_input_bytes_, plan_floor_bytes_);
}

}  // namespace llmp::engine
