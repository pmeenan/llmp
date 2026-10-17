// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "engine/model_runner.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <format>
#include <limits>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

#include "artifact/format_map.h"
#include "components/step.h"
#include "engine/prefill_lookahead.h"
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

constexpr std::uint32_t PhaseBit(pl::Phase phase) {
  return std::uint32_t{1} << static_cast<std::uint32_t>(phase);
}

// The I32 values of one segment's rows (or every segment's, joined) for a
// tokens or positions input, written into `out` and the rest zeroed.
std::expected<void, std::string> WriteRows(std::span<std::byte> out,
                                           std::span<const std::int32_t> values) {
  const std::uint64_t bytes = values.size_bytes();
  if (bytes > out.size()) return Error("an input's rows exceed its slot");
  if (bytes != 0) std::memcpy(out.data(), values.data(), bytes);
  std::fill(out.begin() + static_cast<std::ptrdiff_t>(bytes), out.end(), std::byte{0});
  return {};
}

// The values of an input the step's shape decides (step_program.h
// kOutputRows, kStoreCells, kMask), by components/step.h InputValues' rules.
std::expected<void, std::string> WriteComputed(const kn::InputSlot& in, const pl::StepShape& shape,
                                               std::span<std::byte> out) {
  namespace co = llmp::components;
  const co::CellKey cells{.capacity = in.cells.capacity,
                          .ring = in.cells.ring,
                          .frozen = in.cells.frozen,
                          .fixed = in.cells.fixed,
                          .block = in.cells.block,
                          .per_row = in.cells.per_row,
                          .first_row = in.cells.first_row,
                          .positions = in.cells.positions,
                          .recent = in.cells.recent};
  co::StepInput input{.segment = in.segment,
                      .cells = cells,
                      .read = in.read,
                      .ratio = in.ratio,
                      .overlap = in.overlap,
                      .ring = in.ring,
                      .visible_capacity = in.visible_capacity};
  switch (in.kind) {
    case kn::InputKind::kHadamard: {
      const auto values = co::HadamardValues(in.read);
      if (values.size() * sizeof(float) != out.size()) return Error("a Hadamard input's size");
      std::memcpy(out.data(), values.data(), out.size());
      return {};
    }
    case kn::InputKind::kVisibleCells:
    case kn::InputKind::kPositionPhases:
    case kn::InputKind::kKeptRows:
    case kn::InputKind::kKeptCells:
    case kn::InputKind::kPoolReads:
    case kn::InputKind::kBlockCells:
    case kn::InputKind::kBlockStarts: {
      switch (in.kind) {
        case kn::InputKind::kVisibleCells:
          input.kind = co::InputKind::kVisibleCells;
          break;
        case kn::InputKind::kPositionPhases:
          input.kind = co::InputKind::kPositionPhases;
          break;
        case kn::InputKind::kKeptRows:
          input.kind = co::InputKind::kKeptRows;
          break;
        case kn::InputKind::kKeptCells:
          input.kind = co::InputKind::kKeptCells;
          break;
        case kn::InputKind::kPoolReads:
          input.kind = co::InputKind::kPoolReads;
          break;
        case kn::InputKind::kBlockCells:
          input.kind = co::InputKind::kBlockCells;
          break;
        case kn::InputKind::kBlockStarts:
          input.kind = co::InputKind::kBlockStarts;
          break;
        default:
          break;
      }
      auto values = co::InputValues(input, shape, {});
      if (!values) return Error(values.error());
      if (in.kind == kn::InputKind::kKeptCells || in.kind == kn::InputKind::kBlockCells) {
        if (values->size() * sizeof(std::int64_t) != out.size())
          return Error("a compressor's cell input size");
        std::memcpy(out.data(), values->data(), out.size());
        return {};
      }
      for (const auto value : *values)
        if (value < std::numeric_limits<std::int32_t>::min() ||
            value > std::numeric_limits<std::int32_t>::max())
          return Error("a compressor's row index exceeds I32");
      const std::vector<std::int32_t> rows(values->begin(), values->end());
      return WriteRows(out, rows);
    }
    case kn::InputKind::kOutputRows: {
      input.kind = co::InputKind::kOutputRows;
      auto values = co::InputValues(input, shape, {});
      if (!values) return Error(values.error());
      std::vector<std::int32_t> rows(values->begin(), values->end());
      return WriteRows(out, rows);
    }
    case kn::InputKind::kBlockPositions: {
      input.kind = co::InputKind::kBlockPositions;
      input.sections = static_cast<std::uint8_t>(in.sections);
      auto values = co::InputValues(input, shape, {});
      if (!values) return Error(values.error());
      std::vector<std::int32_t> rows(values->begin(), values->end());
      return WriteRows(out, rows);
    }
    case kn::InputKind::kStoreCells: {
      // With `outputs`, the segment's last rows' cells only (a target's
      // tap rows, components/step.h kOutputCells).
      if (in.outputs != 0 && (in.segment >= shape.segments.size() ||
                              in.outputs != shape.segments[in.segment].outputs)) {
        return Error("a tap's cells name no segment's outputs");
      }
      input.kind = in.outputs != 0 ? co::InputKind::kOutputCells : co::InputKind::kStoreCells;
      auto values = co::InputValues(input, shape, {});
      if (!values) return Error(values.error());
      const std::uint64_t bytes = values->size() * sizeof(std::int64_t);
      if (bytes > out.size()) return Error("an input's cells exceed its slot");
      if (bytes != 0) std::memcpy(out.data(), values->data(), bytes);
      std::fill(out.begin() + static_cast<std::ptrdiff_t>(bytes), out.end(), std::byte{0});
      return {};
    }
    case kn::InputKind::kMask: {
      // The positions the read cells hold, then each row's view of them.
      input.kind = co::InputKind::kCellPositions;
      input.read = in.read;
      auto held = co::InputValues(input, shape, {});
      if (!held) return Error(held.error());
      const pl::Segment& s = shape.segments.at(in.segment);
      if (held->size() != in.read || in.mask_rows < s.rows ||
          out.size() != in.read * in.mask_rows * sizeof(std::uint16_t)) {
        return Error("a mask's slot does not hold its cells and rows");
      }
      constexpr std::uint16_t kSeen = 0x0000;    // F16 0
      constexpr std::uint16_t kHidden = 0xFC00;  // F16 -inf
      std::vector<std::uint16_t> mask(in.read * in.mask_rows, kHidden);
      for (std::uint32_t r = 0; r < s.rows; ++r) {
        const std::int64_t position = std::int64_t{s.past} + r;
        for (std::uint64_t c = 0; c < in.read; ++c) {
          const std::int64_t cell = (*held)[c];
          const bool seen = cell >= 0 && (in.ahead || cell <= position) &&
                            (in.window == 0 || position - cell < std::int64_t{in.window});
          if (seen) mask[(std::uint64_t{r} * in.read) + c] = kSeen;
        }
      }
      std::memcpy(out.data(), mask.data(), out.size());
      return {};
    }
    default:
      return Error("not a computed input");
  }
}

}  // namespace

Status CheckRowTableSource(const RunnerConfig::RowTable& table, std::uint32_t version,
                           std::span<const artifact::Resource> resources) {
  if (table.resource >= resources.size()) return Error("a row table's resource");
  const auto& resource = resources[table.resource];
  if (version == 1 || table.source_parts.empty()) {
    if ((version != 0 && version != 1) || !table.source_parts.empty() || !resource.format ||
        *resource.format != table.format || resource.part != table.part)
      return Error("a row table's format or part differs from its source resource");
    return {};
  }
  if (version != 0 || table.source_parts.size() > format::PartList::kMax)
    return Error("a row table's source assembly exceeds its format parts");
  std::array<artifact::V0Part, format::PartList::kMax> parts{};
  bool matched = false;
  for (std::size_t i = 0; i < table.source_parts.size(); ++i) {
    const auto [role, index] = table.source_parts[i];
    if (index >= resources.size()) return Error("a row table's source part is absent");
    for (std::size_t j = 0; j < i; ++j)
      if (table.source_parts[j].first == role || table.source_parts[j].second == index)
        return Error("a row table repeats a source part or resource");
    const auto& source = resources[index];
    if (source.repr.bytes != source.bytes.value())
      return Error("a row table's source part extent differs from its representation");
    parts[i] = {.role = role, .rep = &source.repr};
    if (index == table.resource && role == table.part) matched = true;
  }
  const auto assembled =
      artifact::Assemble(table.format.encoding, table.format.layout,
                         std::span(parts).first(table.source_parts.size()), table.format.stacked);
  if (!matched || !assembled || *assembled != table.format)
    return Error("a row table's complete source assembly differs from its logical format");
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
  // A slot's state changed outside its steps: what a drafter step carried
  // for it no longer follows.
  options.invalidate = [self, also = std::move(options.invalidate)](std::uint32_t slot) {
    self->ForgetCarry(slot);
    if (also) also(slot);
  };
  options.content_changed = [self, also = std::move(options.content_changed)](std::uint32_t slot) {
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

Status ModelRunner::CheckOwners(std::size_t owners) const {
  if (owners < 2) return {};
  bool repeat = false;
  for (std::size_t stage = 0; stage < slots_.stages(); ++stage) {
    for (const auto& item : slots_.layout(stage).items)
      repeat |= item.contract.draft.progress == model::DraftProgressPolicy::kRepeatPending;
  }
  if (!repeat) return {};
  // The declared layouts, not a hydrated mirror or current readiness,
  // determine this boundary. Every stage participates in the qualification.
  for (std::size_t stage = 0; stage < slots_.stages(); ++stage) {
    if (!source_->JoinedRepeatPending(static_cast<std::uint16_t>(stage)))
      return Error(
          std::format("joined repeat-pending execution is not qualified by stage {}", stage));
  }
  return {};
}

std::expected<void, std::string> ModelRunner::CheckShape(const pl::StepShape& shape) const {
  if (auto identity = source_->ExecutionIdentity(); !identity) return Error(identity.error());
  if (auto qualified = source_->CheckExecutionShape(shape); !qualified) return qualified;
  if (shape.segments.empty()) return Error("a step has no segments");
  if (auto qualified = CheckOwners(shape.segments.size()); !qualified) return qualified;
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
  for (std::size_t i = 0; i < shape.segments.size(); ++i) {
    const pl::Segment& s = shape.segments[i];
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

pl::PlanKey ModelRunner::KeyOf(const pl::StepShape& shape) const {
  pl::PlanKey key = key_base_;
  key.shape = shape;
  if (shape.stage < slots_.stages()) {
    // A stage whose steps read only part of their read cells (position
    // rings, min(their cells, the read cells)) or none keys only that part.
    if (const auto bound = model::ReadCellsBound(slots_.layout(shape.stage))) {
      key.shape.read_align = 0;
      for (auto& seg : key.shape.segments) {
        seg.read_cells =
            static_cast<decltype(seg.read_cells)>(std::min<std::uint64_t>(seg.read_cells, *bound));
      }
    }
  }
  key.parameters = source_->ShapeParameters(key.shape);
  return key;
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
  const auto state_place = [this, stage = shape.stage, owners](
                               const ir::StateRef& ref) -> std::pair<std::uint64_t, std::uint64_t> {
    if (ref.segment >= owners.size() || ref.stage >= slots_.stages()) return {};
    if (ref.stage != stage && stage != 0) return {};
    const model::StateLayoutDerived& layout = slots_.layout(ref.stage);
    for (const auto& item : layout.items) {
      if (item.layer == ref.layer && item.tensor == ref.tensor) {
        if (ref.stage != stage && !item.contract.target_writes) return {};
        return {slots_.live(owners[ref.segment], ref.stage).base(0), item.offset};
      }
    }
    if (ref.stage != stage) return {};
    // A drafter's view of the target's tensor: the owner's target state.
    for (const auto& borrowed : layout.borrowed) {
      if (borrowed.layer != ref.layer || borrowed.tensor != ref.tensor) continue;
      const auto* item = model::BorrowedItem(slots_.layout(0), borrowed);
      return item == nullptr ? std::pair<std::uint64_t, std::uint64_t>{}
                             : std::pair{slots_.live(owners[ref.segment], 0).base(0), item->offset};
    }
    return {};
  };
  a.state = [state_place](const ir::StateRef& ref) -> std::uint64_t {
    const auto [base, offset] = state_place(ref);
    std::uint64_t address = 0;
    return base != 0 && !__builtin_add_overflow(base, offset, &address) ? address : 0;
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
    const auto [base, offset] = state_place(ref);
    return base != 0 && alignment != 0 ? std::gcd(alignment, offset) : 0;
  };
  a.save_rows = std::uint64_t{config_.max_verify_rows} * slots_.slots();
  a.save = [this, stage = shape.stage, owners = std::move(owners)](
               std::uint32_t layer, std::uint16_t component, std::string_view what,
               std::uint8_t segment) -> std::uint64_t {
    if (stage >= commit_places_.size()) return 0;
    if (segment != ir::kJoinedSegment && segment >= owners.size()) return 0;
    const std::uint32_t owner = segment == ir::kJoinedSegment ? 0 : owners[segment];
    for (const CommitPlace& place : commit_places_[stage]) {
      const CommitBuffer& buffer = place.buffer;
      if (buffer.layer == layer && buffer.component == component && buffer.what == what) {
        return commit_saves_.base + place.offset +
               (std::uint64_t{owner} * config_.max_verify_rows * buffer.row_bytes);
      }
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
  std::uint64_t activations = 0, pool = 0, inputs = 0, outputs = 0;
  std::vector<std::uint64_t> carry(slots_.stages(), 0);
  for (const pl::StepShape& shape : config_.setup_shapes) {
    if (auto checked = CheckShape(shape); !checked) {
      return Error(std::format("a setup shape: {}", checked.error()));
    }
    auto program = source_->Build(shape, AddressesFor(shape, true));
    if (!program) return Error(std::format("measuring a step: {}", program.error()));
    const kn::StepProgram& p = **program;
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
    inputs = std::max(inputs, measured->inputs);
    outputs = std::max(outputs, measured->outputs);
    plan_floor_bytes_ = std::max(plan_floor_bytes_, measured->host);
  }
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
  std::vector<RunCopy> copies;
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
          std::uint64_t committed = slots_.positions(seg.slot);
          if (const auto* own = CompanionFor(seg.slot, shape.stage);
              own && own->part == CompanionPart::kPost) {
            if (const auto tap_span = slots_.progress(seg.slot, shape.stage)->compound_tap())
              committed = tap_span->positions;
          }
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
          for (std::uint32_t r = 0; r < seg.rows; ++r) {
            const std::uint64_t cell =
                tap->contract.cells == model::CellRule::kPerRow
                    ? std::uint64_t{tap->contract.window_or_ratio} + from.position + r - held.first
                    : (std::uint64_t{from.position} + r) % tap->cells;
            copies.push_back(
                {to + (r * row_bytes), base + (cell * tap->contract.cell_bytes), row_bytes});
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
          copies.push_back({to, CarryAddress(seg.slot, shape.stage), seg.rows * row_bytes});
          break;
        }
      }
    }
  }
  return copies;
}

ModelRunner::TapRows ModelRunner::HeldTapRows(std::uint32_t slot, std::size_t stage) const {
  if (const auto* progress = slots_.progress(slot, stage)) {
    const auto provisional = companion_steps_.empty() ? std::nullopt : progress->compound_tap();
    const model::DraftProgress& p = provisional ? *provisional : progress->after_owed();
    return {.first = p.tap_first, .count = p.tap_rows, .preceding = p.preceding};
  }
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
  std::vector<RunCopy> copies;
  if (shape.stage != 0) return copies;
  for (const pl::Segment& seg : shape.segments) {
    for (std::size_t stage = 1; stage < slots_.stages(); ++stage) {
      const auto* tap = model::FeatureTap(slots_.layout(stage));
      if (tap == nullptr || tap->contract.cells != model::CellRule::kPerRow) continue;
      const TapRows held = HeldTapRows(seg.slot, stage);
      const std::uint32_t count = std::min(held.count, tap->contract.window_or_ratio);
      if (count == 0 || std::uint64_t{held.first} + held.count != seg.past) continue;
      const std::uint64_t bytes = tap->contract.cell_bytes;
      const std::uint64_t base = slots_.live(seg.slot, stage).base(0) + tap->offset;
      copies.push_back({base + ((tap->contract.window_or_ratio - count) * bytes),
                        base + ((tap->contract.window_or_ratio + held.count - count) * bytes),
                        count * bytes});
    }
  }
  return copies;
}

std::expected<std::vector<RunCopy>, std::string> ModelRunner::CarryCopies(
    const kn::StepProgram& program, const pl::StepShape& shape) const {
  std::vector<RunCopy> copies;
  for (const kn::OutputSlot& out : program.outputs()) {
    if (!out.features) continue;
    if (shape.stage == 0 || out.segment >= shape.segments.size() ||
        out.bytes > carry_bytes_.at(shape.stage)) {
      return Error("a step's next features pass their carry");
    }
    copies.push_back({CarryAddress(shape.segments[out.segment].slot, shape.stage),
                      out.device_address, out.bytes});
  }
  return copies;
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
    auto buffers = source_->CommitBuffers(stage, config_.max_verify_rows);
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
    if (config.weight.stage >= config_.weights.size() || config.weight.expert_array ||
        config.rows_per_token == 0 || config.format.shape.rank != 2) {
      return Error("a row table's weights, shape or lookups");
    }
    if (std::ranges::any_of(row_tables_, [&](const auto& t) {
          return t->config.weight == config.weight && t->config.part == config.part;
        })) {
      return Error("a row table was declared twice");
    }
    auto& weights = config_.weights[config.weight.stage];
    const auto& artifact = weights.weights->artifact();
    if (config.resource >= artifact.resources().size()) return Error("a row table's resource");
    const auto& resource = artifact.resources()[config.resource];
    const auto& group = artifact.groups()[resource.group];
    auto parts = format::PartsOf(config.format);
    const auto part = parts ? parts->Find(config.part) : std::nullopt;
    if (!part) return Error("a row table's format has no such part");
    if (part->bytes != resource.bytes.value()) {
      return Error("a row table's format does not cover exactly its resource");
    }
    auto row_bytes = format::RowBytes(config.format, config.part);
    if (!row_bytes || *row_bytes > std::numeric_limits<std::uint32_t>::max()) {
      return Error("a row table's row size");
    }
    if (resource.group >= weights.places.size() ||
        weights.places[resource.group] != GroupPlace::kNone) {
      return Error("a row-paged table's group must not be resident");
    }
    for (std::uint32_t i = 0; i < artifact.resources().size(); ++i) {
      if (i != config.resource && artifact.resources()[i].group == resource.group) {
        return Error("a row-paged table shares its group with another resource");
      }
    }
    auto first = artifact::ChunkRangeOf(artifact.layout(), {.group = resource.group, .chunk = 0});
    if (!first) return Error("a row table's file range");
    auto table = std::make_unique<RowTableState>();
    table->config = config;
    table->shard = first->shard;
    table->table = {.file_offset = first->file_offset.value() + resource.offset.value(),
                    .rows = config.format.shape.ne[1],
                    .row_bytes = *row_bytes,
                    .chunk_file_offset = first->file_offset.value(),
                    .file_bytes = first->file_offset.value() + group.stored.value()};
    for (const auto& shape : config_.setup_shapes) {
      if (shape.stage != config.weight.stage) continue;
      table->capacity =
          std::max(table->capacity, std::uint64_t{shape.rows()} * config.rows_per_token);
    }
    // Slot IDs are I32 and counts are U32; byte offsets are U64. A lookup
    // can span at most two blocks. Bound allocation arithmetic separately.
    if (table->capacity == 0 ||
        table->capacity > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
      return Error("a row table's lookup count exceeds its slot IDs");
    }
    const auto valid =
        PlanPleRows(table->table, {}, PleLandingBound(table->capacity), table->capacity);
    if (!valid) return Error(valid.error());
    auto ring = providers::OpenStorage(64);
    if (!ring) return Error(std::format("row-table storage: {}", ring.error().message()));
    table->storage = std::move(*ring);
    row_tables_.push_back(std::move(table));
    auto& held = *row_tables_.back();  // stable through partial setup and release
    std::uint64_t slot_bytes = 0, source_bytes = 0;
    if (__builtin_mul_overflow(held.capacity, *row_bytes, &slot_bytes) ||
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
        return WriteComputed(program.inputs()[i], shape, out);
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

Status ModelRunner::Release() {
  if (released_) return {};
  released_ = true;
  // Release's existing contract is node teardown after successful drain and
  // stream fences. Only that proof may break a retained routed-owner cycle.
  routed_flight_.reset();
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
  return support::Joined(problems);
}

// --------------------------------------------------------------------- plans

void ModelRunner::DropPlans() {
  if (!routed_flight_) {
    ForgetCompletedRouting();
    plans_.Clear();
  }
}

void ModelRunner::ReclaimCandidates(std::uint32_t owner, bool running,
                                    std::vector<memory::ReclaimCandidate>& out) {
  if (released_ || slots_.faulted()) return;
  std::array<PlanCacheBase*, 1> caches{&plans_};
  CollectPlans(caches, owner, running, out);
}

std::uint64_t ModelRunner::Reclaim(memory::ReclaimKind kind, std::uint64_t id) {
  if (released_ || slots_.faulted() || routed_flight_) return 0;
  std::array<PlanCacheBase*, 1> caches{&plans_};
  ForgetCompletedRouting();
  return ReclaimPlan(caches, kind, id);
}

std::expected<ModelRunner::Plans::Entry*, std::string> ModelRunner::Install(
    const pl::StepShape& shape, std::unique_ptr<StepPlanned> planned, double seconds,
    const std::function<void()>& transfer) {
  if (planned == nullptr || planned->program == nullptr) return Error("no program to install");
  kn::StepProgram& program = *planned->program;
  if (auto routed = CheckRouting(program); !routed) return Error(routed.error());
  auto checked = programs_.BindPrepared(*planned, [&](const kn::StepProgram& bound) -> Status {
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
  });
  if (!checked) return Error(checked.error());
  coverage_.tensors += checked->tensors;
  const std::uint64_t host = program.host_bytes();
  const std::uint64_t launches = program.launches();
  if (transfer) transfer();
  ForgetCompletedRouting();
  Plans::Entry& entry = plans_.Add(KeyOf(shape), std::move(planned), host, launches, seconds);
  // Add's required PlanAccount charge owns all returned program storage before
  // the source's transient compiler use can be returned to another builder.
  entry.planned->program->ConstructionPaid();
  // Never recorded: no capture beside a run, on its first run or ahead.
  if (entry.planned->positioned) entry.runs[0].uncapturable = true;
  return &entry;
}

std::expected<ModelRunner::Plans::Entry*, std::string> ModelRunner::Planned(
    const pl::StepShape& shape) {
  if (auto identity = source_->ExecutionIdentity(); !identity) return Error(identity.error());
  if (auto current = CheckRoutingSource(); !current) return Error(current.error());
  if (Plans::Entry* found = plans_.Find(KeyOf(shape)); found != nullptr) return found;
  const auto started = std::chrono::steady_clock::now();
  auto program = source_->Build(shape, AddressesFor(shape, false));
  if (!program) return Error(std::format("planning a step: {}", program.error()));
  if (*program == nullptr) return Error("planning a step gave no program");
  auto planned = std::make_unique<StepPlanned>();
  planned->program = std::move(*program);
  return Install(shape, std::move(planned),
                 support::Seconds(std::chrono::steady_clock::now() - started), {});
}

std::expected<Copies, std::string> ModelRunner::OutputCopies(const kn::StepProgram& program) const {
  Copies copies;
  std::uint64_t at = 0;
  for (const kn::OutputSlot& slot : program.outputs()) {
    if (at > outputs_staging_bytes_ || slot.bytes > outputs_staging_bytes_ - at) {
      return Error("a step's outputs exceed their staging");
    }
    copies.push_back({Address(outputs_staging_ + at), slot.device_address, slot.bytes});
    at += Round(slot.bytes, 256);
  }
  return copies;
}

// --------------------------------------------------------------------- steps

std::expected<void, StepRefusal> ModelRunner::Admit(const pl::Segment& segment, std::uint16_t stage,
                                                    std::uint32_t read_align, bool advance) {
  if (!bound_ || released_) return Refused("the model runner is not bound");
  const std::uint32_t slot = segment.slot;
  if (slot >= slots_.slots()) return Refused(std::format("no slot {}", slot));
  if (stage >= slots_.stages()) return Refused(std::format("no stage {}", stage));
  if (stage != 0 && slots_.progress(slot, stage) && CompanionFor(slot, stage) == nullptr) {
    const auto authorized = std::ranges::find_if(authorized_drafts_, [&](const auto& own) {
      return own.slot == slot && own.stage == stage && own.past == segment.past &&
             own.rows == segment.rows;
    });
    if (authorized == authorized_drafts_.end() ||
        !slots_.progress(slot, stage)->CheckUnit(authorized->unit))
      return Refused("repeat-pending draft requires its owned decoding schedule", true, slot);
  }
  if (stage == 0 && advance && CompanionFor(slot, stage) == nullptr)
    for (std::size_t s = 1; s < slots_.stages(); ++s)
      if (slots_.progress(slot, s))
        return Refused("target step requires its shared prompt companions", true, slot);
  if (auto active = slots_.CheckActive(slot); !active) return Refused(active.error(), true, slot);
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
    // wherever its catch-up starts.
    if (advance) return Refused("a drafter's step never commits positions", true, slot);
    if (std::uint64_t{segment.past} + segment.rows >
        std::uint64_t{positions} + slots_.layout(stage).max_rows) {
      return Refused(std::format("slot {}'s draft rows end past {} positions and its stage's {} "
                                 "rows",
                                 slot, positions, slots_.layout(stage).max_rows),
                     true, slot);
    }
    // Its view of the target's state is read only while that is frozen.
    if (!slots_.layout(stage).borrowed.empty() && !slots_.borrowed(slot, 0)) {
      return Refused(std::format("slot {}'s drafter reads the target's state, which is not "
                                 "borrowed",
                                 slot),
                     true, slot);
    }
  }
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

const ModelRunner::CompanionStep* ModelRunner::CompanionFor(std::uint32_t slot,
                                                            std::uint16_t stage) const {
  for (const auto& step : companion_steps_)
    if (step.slot == slot && step.stage == stage) return &step;
  return nullptr;
}

std::expected<StepReport, StepRefusal> ModelRunner::RunStep(const StepRequest& request) {
  if (routed_flight_) return Refused("a routed step retains unproven effects", false);
  if (routed_ &&
      (request.ahead.build || request.ahead.capture_ahead || request.ahead.prepare ||
       std::ranges::any_of(request.ahead.shapes, [](const auto& s) { return s.has_value(); })))
    return Refused("routed continuation does not admit asynchronous lookahead");
  if (!bound_ || released_) return Refused("the model runner is not bound");
  const pl::StepShape& shape = request.shape;
  if (auto checked = CheckShape(shape); !checked) return Refused(checked.error());
  if (request.work.size() != shape.segments.size()) {
    return Refused("a step's work does not match its segments");
  }
  for (const auto& future : request.ahead.shapes) {
    if (future) {
      if (auto qualified = CheckOwners(future->segments.size()); !qualified)
        return Refused(qualified.error());
    }
    if (future && future->phase == pl::Phase::kCommit)
      return Refused("commits are internal and cannot be lookahead steps");
  }
  const bool verify = shape.phase == pl::Phase::kVerify;
  if (shape.phase == pl::Phase::kCommit) return Refused("commits run only after a verify's Accept");
  if (verify && (shape.stage != 0 || request.advance)) {
    return Refused("a verify runs on the target and leaves its rows to their Accept");
  }
  if (verify && (config_.max_verify_rows == 0 ||
                 std::ranges::any_of(shape.segments, [&](const pl::Segment& s) {
                   return s.rows > config_.max_verify_rows;
                 }))) {
    return Refused(std::format("the runner verifies at most {} rows", config_.max_verify_rows));
  }
  if (shape.stage != 0 && request.advance) {
    return Refused("a drafter's step never commits positions");
  }
  for (std::size_t segment_index = 0; segment_index < shape.segments.size(); ++segment_index) {
    const auto& segment = shape.segments[segment_index];
    const auto* companion = CompanionFor(segment.slot, shape.stage);
    if (companion && (companion->past != segment.past || companion->rows != segment.rows ||
                      request.advance || verify))
      return Refused("step differs from its owned prompt companion");
    if (shape.stage != 0 && !companion) {
      if (const auto* progress = slots_.progress(segment.slot, shape.stage)) {
        const auto& feature = request.work[segment_index].feature;
        const auto authorization = std::ranges::find_if(authorized_drafts_, [&](const auto& own) {
          return own.slot == segment.slot && own.stage == shape.stage && own.past == segment.past &&
                 own.rows == segment.rows && own.feature.kind == feature.kind &&
                 own.feature.position == feature.position;
        });
        if (shape.phase != pl::Phase::kDraft || authorization == authorized_drafts_.end() ||
            !progress->CheckUnit(authorization->unit))
          return Refused("repeat-pending draft requires its owned decoding schedule");
      }
    }
    if (shape.stage == 0 && !verify && !companion) {
      for (std::size_t s = 1; s < slots_.stages(); ++s)
        if (slots_.progress(segment.slot, s))
          return Refused("target step requires its shared prompt companions");
    }
  }
  if (slots_.faulted()) return Refused("the cohort is faulted", false);
  const PlanStep step;
  struct RowRead {
    RowTableState* table = nullptr;
    PleRowPlan plan;
  };
  std::vector<std::vector<std::int32_t>> side_rows;
  const auto plan_rows =
      [&](const kn::StepProgram& program) -> std::expected<std::vector<RowRead>, StepRefusal> {
    std::vector<RowRead> row_reads;
    side_rows.resize(program.inputs().size());
    // One joined slot per component owns its table's landing and row-slot
    // numbering. Independent segmented plans cannot share that storage.
    std::vector<std::uint16_t> components;
    for (const auto& input : program.inputs()) {
      if (input.kind != kn::InputKind::kSideRows) continue;
      if (std::ranges::find(components, input.component) != components.end())
        return Refused("side-table components require one joined row input");
      components.push_back(input.component);
    }
    for (std::size_t index = 0; index < program.inputs().size(); ++index) {
      const auto& input = program.inputs()[index];
      if (input.kind != kn::InputKind::kSideRows) continue;
      auto& ids = side_rows[index];
      if (!request.side_rows.empty()) {
        if (input.bytes != request.side_rows.size_bytes())
          return Refused("the side rows must fill their slot");
        ids.assign(request.side_rows.begin(), request.side_rows.end());
      } else {
        std::uint64_t rows = 0;
        for (std::size_t i = 0; i < shape.segments.size(); ++i)
          if (input.segment == ir::kJoinedSegment || input.segment == i)
            rows += shape.segments[i].rows;
        if (rows == 0 || input.bytes % (rows * sizeof(std::int32_t)) != 0)
          return Refused("side rows have no whole number of heads");
        const auto heads = input.bytes / (rows * sizeof(std::int32_t));
        if (heads == 0) return Refused("side rows have no heads");
        ids.resize(input.bytes / sizeof(std::int32_t));
        std::size_t at = 0;
        for (std::size_t i = 0; i < shape.segments.size(); ++i) {
          if (input.segment != ir::kJoinedSegment && input.segment != i) continue;
          const auto& segment = shape.segments[i];
          const auto& work = request.work[i];
          auto bounds = model::DraftBounds(slots_.layout(shape.stage));
          if (!bounds) return Refused(bounds.error());
          const std::uint64_t token_first =
              std::uint64_t{segment.past} +
              (bounds->has_value() && (**bounds).relation == model::DraftRelation::kTokenAfterCache
                   ? 1U
                   : 0U);
          if (token_first > work.history.size() ||
              segment.rows > work.history.size() - token_first ||
              work.tokens.size() != segment.rows ||
              !std::ranges::equal(work.tokens, work.history.subspan(token_first, segment.rows)))
            return Refused("side rows require the owner's complete matching token history");
          for (std::uint32_t row = 0; row < segment.rows; ++row) {
            auto derived = source_->SideRows(shape.stage, input.component, work.history,
                                             static_cast<std::uint32_t>(token_first + row),
                                             std::span(ids).subspan(at, heads));
            if (!derived) return Refused(derived.error());
            at += heads;
          }
        }
      }
      if (std::ranges::any_of(ids, [](const auto id) { return id < 0; }))
        return Refused("a side row is negative");
      for (auto& table : row_tables_) {
        if (table->config.weight.stage != shape.stage ||
            table->config.weight.component != input.component)
          continue;
        if (table->stalled) return Refused("row-table reads have unproven completion", false);
        auto plan =
            PlanPleRows(table->table, ids, PleLandingBound(table->capacity), table->capacity);
        if (!plan) return Refused(plan.error());
        // One side-row input is staged from its first part's remap. Every
        // other part must agree in lookup order, not just in its unique IDs.
        for (const auto& prior : row_reads) {
          if (prior.table->config.weight.stage == shape.stage &&
              prior.table->config.weight.component == input.component &&
              prior.plan.slots != plan->slots)
            return Refused("a side table's parts disagree on row-slot mapping");
        }
        row_reads.push_back({.table = table.get(), .plan = std::move(*plan)});
      }
    }

    return row_reads;
  };
  // Row-table input validation precedes admission: even zeroing newly
  // backed state is work a malformed set of side rows must not cause.
  // Planning binds stable virtual addresses only; PlanStep pins the plan
  // across admission and the eventual job.
  std::expected<Plans::Entry*, std::string> entry_of = nullptr;
  std::vector<RowRead> row_reads;
  if (!row_tables_.empty() || !request.side_rows.empty() || source_->UsesSideRows(shape.stage)) {
    entry_of = Planned(shape);
    if (!entry_of) return Refused(entry_of.error());
    auto reads = plan_rows(*(*entry_of)->planned->program);
    if (!reads) return std::unexpected(reads.error());
    row_reads = std::move(*reads);
  }
  // Every owner's own cells (what it reads alone: its rows at its state's
  // alignment, exactly as a lone step's segment admits them) before any
  // owner's wider reads for the join (a joined alignment, a common read
  // through a peer's cells): widened first, an early owner could take what
  // a later owner's own rows need and keep it, refusing that owner in every
  // wave while capacity for both owners' own reads was there. A refusal
  // there is the owner's alone too; one in the widening is not (`widening`).
  // A widening refused for a later owner keeps the earlier owners' (clean,
  // leased growth they read once their positions reach it): bounded by the
  // join's reads, not released (docs/engine.md).
  if (shape.segments.size() > 1) {
    for (pl::Segment own : shape.segments) {
      own.read_cells = LoneRead(shape.stage, own.past, own.rows);
      if (auto admitted = Admit(own, shape.stage, 0, request.advance); !admitted) {
        return std::unexpected(std::move(admitted.error()));
      }
    }
  }
  for (const pl::Segment& s : shape.segments) {
    if (auto admitted = Admit(s, shape.stage, shape.read_align, request.advance); !admitted) {
      admitted.error().widening = shape.segments.size() > 1;
      return std::unexpected(std::move(admitted.error()));
    }
  }
  if (auto places = slots_.CheckPlaces(); !places) return Refused(places.error());
  if (*entry_of == nullptr) {
    entry_of = Planned(shape);
    if (!entry_of) return Refused(entry_of.error());
    auto reads = plan_rows(*(*entry_of)->planned->program);
    if (!reads) return std::unexpected(reads.error());
    row_reads = std::move(*reads);
  }
  Plans::Entry& entry = **entry_of;
  kn::StepProgram& program = *entry.planned->program;

  // Inputs, staged by slot.
  const auto fill = [&](std::size_t index,
                        std::span<std::byte> out) -> std::expected<void, std::string> {
    const kn::InputSlot& in = program.inputs()[index];
    const bool joined = in.segment == ir::kJoinedSegment;
    const std::size_t segment = in.segment;
    if (!joined && segment >= shape.segments.size()) return Error("an input names no segment");
    std::vector<std::int32_t> values;
    switch (in.kind) {
      case kn::InputKind::kTokens:
        if (!joined) {
          const auto tokens = request.work[in.segment].tokens;
          if (tokens.size() != shape.segments[in.segment].rows) return Error("a segment's tokens");
          return WriteRows(out, tokens);
        }
        for (std::size_t i = 0; i < shape.segments.size(); ++i) {
          const auto tokens = request.work[i].tokens;
          if (tokens.size() != shape.segments[i].rows) return Error("a segment's tokens");
          values.insert(values.end(), tokens.begin(), tokens.end());
        }
        return WriteRows(out, values);
      case kn::InputKind::kPositions:
        // In `sections` copies, one after another (a multi-section
        // rotation's).
        for (std::uint32_t sec = 0; sec < std::max<std::uint32_t>(in.sections, 1); ++sec) {
          for (std::size_t i = 0; i < shape.segments.size(); ++i) {
            if (!joined && i != segment) continue;
            for (std::uint32_t r = 0; r < shape.segments[i].rows; ++r) {
              values.push_back(static_cast<std::int32_t>(shape.segments[i].past + r));
            }
          }
        }
        return WriteRows(out, values);
      case kn::InputKind::kSideRows: {
        const auto ids = std::span(side_rows[index]);
        if (ids.size_bytes() != out.size()) {
          return Error("the side rows must fill their slot");
        }
        for (const RowRead& read : row_reads) {
          if (read.table->config.weight.component == in.component) {
            return WriteRows(out, read.plan.slots);
          }
        }
        return WriteRows(out, ids);
      }
      case kn::InputKind::kSegmentBytes: {
        if (joined) return Error("a segment's bytes name no segment");
        const auto bytes = request.work[in.segment].bytes;
        if (bytes.size() > out.size()) return Error("a segment's bytes exceed their slot");
        std::ranges::copy(bytes, out.begin());
        std::fill(out.begin() + static_cast<std::ptrdiff_t>(bytes.size()), out.end(), std::byte{0});
        return {};
      }
      case kn::InputKind::kStepBytes:
        if (request.step_bytes.size() > out.size())
          return Error("the step's bytes exceed their slot");
        std::ranges::copy(request.step_bytes, out.begin());
        std::fill(out.begin() + static_cast<std::ptrdiff_t>(request.step_bytes.size()), out.end(),
                  std::byte{0});
        return {};
      case kn::InputKind::kOutputRows:
      case kn::InputKind::kStoreCells:
      case kn::InputKind::kMask:
      case kn::InputKind::kBlockPositions:
      case kn::InputKind::kVisibleCells:
      case kn::InputKind::kPositionPhases:
      case kn::InputKind::kKeptRows:
      case kn::InputKind::kKeptCells:
      case kn::InputKind::kPoolReads:
      case kn::InputKind::kBlockCells:
      case kn::InputKind::kBlockStarts:
      case kn::InputKind::kHadamard:
        // A fixed tensor's cell, and a verify's whole step's rows, are
        // every segment's.
        if (in.kind != kn::InputKind::kOutputRows && in.kind != kn::InputKind::kPositionPhases &&
            in.kind != kn::InputKind::kHadamard && joined && !in.cells.fixed && !in.cells.per_row)
          return Error("an input names no segment");
        if (in.commit_save) {
          std::vector<std::int64_t> rows;
          for (const pl::Segment& s : shape.segments) {
            for (std::uint32_t r = 0; r < s.rows; ++r) {
              rows.push_back(std::int64_t{s.slot} * config_.max_verify_rows + r);
            }
          }
          if (rows.size() * sizeof(std::int64_t) != out.size()) {
            return Error("a verify's joined save indices do not fill their slot");
          }
          std::memcpy(out.data(), rows.data(), out.size());
          return {};
        }
        return WriteComputed(in, shape, out);
      case kn::InputKind::kFeatures:
        return Error("a step's features are copied on the device, never staged");
    }
    return Error("an unknown input kind");
  };
  auto copies = programs_.runs().Stage(program.inputs(), fill, 0);
  if (!copies) return Refused(copies.error());
  auto outputs = OutputCopies(program);
  if (!outputs) return Refused(outputs.error());
  // A drafter's features, copied in on the device before its run, and its
  // next features carried to its slots' carries after it.
  auto features = FeatureCopies(program, request);
  if (!features) return Refused(features.error());
  auto carries = CarryCopies(program, shape);
  if (!carries) return Refused(carries.error());
  auto taps = TapCarries(shape);
  if (!taps) return Refused(taps.error());
  if (entry.planned->positioned) {
    // Between jobs: no run of this program is in flight.
    std::vector<std::uint32_t> past;
    past.reserve(shape.segments.size());
    for (const pl::Segment& s : shape.segments) past.push_back(s.past);
    if (auto positioned = entry.planned->program->Position(past); !positioned) {
      return Refused(std::format("positioning a step: {}", positioned.error().detail));
    }
  }

  // The futures: deduplicated against this shape and each other, and the
  // cached ones protected (found) before any optional funding.
  const StepAhead& ahead = request.ahead;
  std::array<Plans::Entry*, 8> ahead_cached{};
  std::array<std::optional<pl::PlanKey>, 8> ahead_keys{};
  std::array<bool, 8> distinct{};
  for (std::size_t k = 0; k < ahead.shapes.size(); ++k) {
    if (!ahead.shapes[k] || !CheckShape(*ahead.shapes[k])) continue;
    ahead_keys[k] = KeyOf(*ahead.shapes[k]);
    if (*ahead_keys[k] == entry.key) {
      ahead_cached[k] = &entry;
      continue;
    }
    bool duplicate = false;
    for (std::size_t prior = 0; prior < k; ++prior) {
      if (ahead_keys[prior] && *ahead_keys[k] == *ahead_keys[prior]) {
        ahead_cached[k] = ahead_cached[prior];
        duplicate = true;
        break;
      }
    }
    if (duplicate) continue;
    distinct[k] = true;
    ahead_cached[k] = plans_.Find(*ahead_keys[k]);
  }
  StepPlanRuns& runs = entry.runs[0];
  // A recording made ahead holds the layout its plan's inputs predicted;
  // one that differs from what was staged is dropped (its charge settles).
  if (programs_.runs().graphs() && runs.has_graph() && *copies != runs.copies) {
    runs.DropGraph();
    ++lookahead_.dropped_ahead;
  }
  const bool repeats = ahead_keys[0].has_value() && *ahead_keys[0] == entry.key;
  const bool first_capture =
      (config_.capture_first_phases & config_.capture_phases & PhaseBit(shape.phase)) != 0 &&
      programs_.runs().graphs() && runs.eager_runs == 0 && !runs.has_graph() &&
      !runs.uncapturable && !program.routed();
  bool capture = first_capture ||
                 (runs.CaptureDue(programs_.runs().graphs()) &&
                  (config_.capture_phases & PhaseBit(shape.phase)) != 0) ||
                 (ahead.capture_ahead && repeats && programs_.runs().graphs() &&
                  !runs.has_graph() && !runs.uncapturable);
  if (capture && !plans_.ChargeGraph(entry)) capture = false;
  const bool capture_first = capture && runs.eager_runs == 0;
  // The next shape's recording, beside this run, when it differs and is
  // already planned.
  Plans::Entry* upcoming = nullptr;
  Copies upcoming_copies, upcoming_outputs;
  if (ahead.capture_ahead && ahead.shapes[0] && !repeats && programs_.runs().graphs() &&
      ahead_cached[0] != nullptr) {
    upcoming = ahead_cached[0];
    const kn::StepProgram& next = *upcoming->planned->program;
    auto layout = programs_.runs().Layout(next.inputs(), 0);
    auto out = OutputCopies(next);
    if (upcoming->runs[0].has_graph() || upcoming->runs[0].uncapturable ||
        upcoming->runs[0].ahead_refused || !layout || !out || !plans_.ChargeGraph(*upcoming)) {
      upcoming = nullptr;
    } else {
      upcoming_copies = std::move(*layout);
      upcoming_outputs = std::move(*out);
    }
  }
  // Independent optional grants for the distinct missing futures.
  std::array<const pl::StepShape*, 4> build{};
  std::size_t build_count = 0;
  if (node_.threaded()) {
    for (std::size_t k = 0;
         k < ahead.shapes.size() && build_count < std::min<std::size_t>(ahead.build, build.size());
         ++k) {
      if (distinct[k] && ahead_cached[k] == nullptr) build[build_count++] = &*ahead.shapes[k];
    }
  }
  PrefillLookaheadGroup<StepPlanned, 4> future(node_, plan_floor_bytes_);
  bool funded = false;
  for (std::size_t i = 0; i < build_count; ++i) {
    ++lookahead_.attempted;
    if (!future.Fund(i)) {
      ++lookahead_.refused;
      build[i] = nullptr;
    } else {
      funded = true;
    }
  }
  StepReport report;
  const std::function<void()> meanwhile = funded ? std::function<void()>([&] {
    const auto built = future.BuildAll(
        [&](std::size_t i) -> std::expected<std::unique_ptr<StepPlanned>, std::string> {
          if (build[i] == nullptr) return Error("not funded");
          auto program = source_->Build(*build[i], AddressesFor(*build[i], false));
          if (!program) return std::unexpected(std::move(program.error()));
          auto planned = std::make_unique<StepPlanned>();
          planned->program = std::move(*program);
          // Fund already charged this future's allowance. Keep construction
          // ownership until its actual returned storage fits that paid owner;
          // an oversized future is destroyed by the existing Build refusal.
          if (PlannedHostBytes(*planned) <= plan_floor_bytes_) planned->program->ConstructionPaid();
          return planned;
        });
    for (std::size_t i = 0; i < built.size(); ++i) {
      lookahead_.built += built[i];
      report.built_ahead += built[i];
      lookahead_.build_seconds += future.seconds(i);
    }
  })
                                                 : std::function<void()>{};

  // The next shape's fresh state, prepared before the job.
  if (ahead.prepare && ahead.shapes[0] && CheckShape(*ahead.shapes[0])) {
    for (const pl::Segment& s : ahead.shapes[0]->segments) {
      // Through what that step will back (Admit): its reads, common or not.
      auto prepared = slots_.PrepareThrough(s.slot, std::max(s.past + s.rows, s.read_cells),
                                            ahead.shapes[0]->read_align, ahead.shapes[0]->stage);
      if (!prepared) {
        // A drain that cannot prove its page-ins retired quarantined a
        // state: then the refusal is not clean.
        const bool drained = slots_.FinishPreparations().has_value();
        return Refused(prepared.error(), drained);
      }
      report.prepared_ahead = report.prepared_ahead || *prepared;
    }
    lookahead_.prepared += report.prepared_ahead;
  }

  // A verify's saves (D-068): what it writes in each stage, by row, into
  // that stage's snapshot, queued in its job before its run.
  struct Saving {
    LiveState* live = nullptr;
    std::uint32_t rows = 0;
    bool any = false;     // ranges to queue (a committing or truncating stage may have none)
    bool queued = false;  // its saves were queued (or it had none): a failure undoes it whole
  };
  std::vector<Saving> saving;
  if (verify) {
    for (const pl::Segment& s : shape.segments) {
      for (std::size_t stage = 0; stage < slots_.stages(); ++stage) {
        if (!Verifies(s.slot, stage)) continue;
        const bool commits = commits_.at((std::size_t{s.slot} * slots_.stages()) + stage);
        auto writes = model::WritesFor(slots_.layout(stage), s.past, s.rows,
                                       {.commits = commits, .drafter = stage != 0});
        if (!writes) {
          const bool drained = slots_.FinishPreparations().has_value();
          return Refused(std::format("a verify's saves: {}", writes.error()), drained);
        }
        LiveState& live = slots_.live(s.slot, stage);
        live.BeginSaves();
        const std::uint64_t base = live.base(0);
        Status saved;
        bool any = false;
        for (std::uint32_t r = 0; r < s.rows && saved; ++r) {
          for (const model::StateRange& range : writes->rows[r]) {
            saved = live.Save(base + range.offset, range.bytes, r);
            if (!saved) break;
            any = true;
          }
        }
        for (const model::StateRange& range : writes->scratch) {
          if (!saved) break;
          saved = live.Save(base + range.offset, range.bytes, -1);
          any = true;
        }
        if (!saved) {
          live.BeginSaves();
          if (!slots_.FinishPreparations()) return Refused(saved.error(), false);
          return Refused(saved.error(), true, s.slot);
        }
        // Every stage it writes or commits, even one that saves nothing
        // (every tensor it writes there truncates): its rows await their
        // Accept, and a failure undoes it whole (by nothing), never
        // quarantines it.
        saving.push_back({.live = &live, .rows = s.rows, .any = any});
      }
    }
  }

  if (auto begin = BeginRouted(program, runs, *copies, *outputs, *carries, capture); !begin)
    return Refused(begin.error(), slots_.FinishPreparations().has_value());
  struct RoutedCleanup {
    ModelRunner& runner;
    ~RoutedCleanup() { runner.ForgetUnstartedRouted(); }
  } routed_cleanup{*this};
  // What the step writes, recorded before the job (incremental spill).
  for (const pl::Segment& s : shape.segments) {
    if (auto recorded = slots_.RecordWrite(s.slot, s.past); !recorded) {
      if (!slots_.FinishPreparations()) return Refused(recorded.error(), false);
      return Refused(recorded.error(), true, s.slot);
    }
  }
  bool wrote = false, unknown = false;
  // A verify's saves queued: work on the stream (the job never "not
  // started"), though they write no state.
  bool saves_queued = false;
  for (const pl::Segment& s : shape.segments) {
    if (auto prepared = slots_.PrepareOwed(s.slot); !prepared) {
      const bool drained = slots_.FinishPreparations().has_value();
      return Refused(prepared.error(), drained && !slots_.faulted() && slots_.usable(s.slot));
    }
  }
  for (RowRead& read : row_reads) {
    RowTableState& table = *read.table;
    const int fd = config_.weights[table.config.weight.stage].weights->shard_fd(table.shard);
    if (fd < 0)
      return Refused("a row table's shard is not open", slots_.FinishPreparations().has_value());
    if (auto loaded = ReadPleRows(*table.storage, fd, read.plan, table.landing); !loaded) {
      table.stalled = table.storage->in_flight() != 0;
      const bool drained = slots_.FinishPreparations().has_value();
      return Refused(loaded.error(), drained && !table.stalled);
    }
    std::ranges::copy(read.plan.sources, table.sources);
    *table.count = static_cast<std::uint32_t>(read.plan.sources.size());
  }
  if (verify) {
    for (const pl::Segment& s : shape.segments) {
      if (auto checked = slots_.CheckProgressVerify(s.slot, s.rows); !checked)
        return Refused(checked.error(), slots_.FinishPreparations().has_value());
    }
    for (const pl::Segment& s : shape.segments) {
      if (auto begun = slots_.BeginProgressVerify(s.slot, s.rows); !begun) {
        (void)slots_.FinishPreparations();
        for (const pl::Segment& owner : shape.segments) slots_.ProgressVerifyFailed(owner.slot);
        return Refused(begun.error(), false);
      }
    }
  }
  std::string failure;
  auto posted = node_.Job(
      slots_.execution(),
      [&](providers::NativeStream native) {
        // What the owners' states owe (a verify's restore and commit)
        // before this step's own work.
        for (const pl::Segment& s : shape.segments) {
          auto owed = slots_.QueueOwed(s.slot, native);
          wrote = wrote || owed.wrote;
          if (!owed.result) {
            failure = owed.result.error().detail;
            unknown = owed.result.error().unknown;
            return unknown ? sc::JobResult::kUnknown
                   : wrote ? sc::JobResult::kFailed
                           : sc::JobResult::kNotStarted;
          }
        }
        for (Saving& sv : saving) {
          // Nothing to save (what it writes truncates, or a commit rebuilds
          // it): undone whole by nothing.
          if (sv.any) {
            if (auto saved = sv.live->QueueSaves(native); !saved) {
              failure = saved.error().detail;
              unknown = saved.error().unknown;
              return unknown                 ? sc::JobResult::kUnknown
                     : wrote || saves_queued ? sc::JobResult::kFailed
                                             : sc::JobResult::kNotStarted;
            }
            saves_queued = true;
          }
          sv.queued = true;
        }
        // This point precedes the first leading-tap copy or private cache
        // write. Clean planning/admission refusals never enter it.
        for (const auto& segment : shape.segments) {
          const auto* companion = CompanionFor(segment.slot, shape.stage);
          if (companion == nullptr) continue;
          for (const auto& [stage, unit] : companion->units) {
            auto* progress = slots_.progress(segment.slot, stage);
            Status marked = companion->part == CompanionPart::kPre ? progress->PreQueued(unit)
                            : companion->part == CompanionPart::kTarget
                                ? progress->TargetQueued(unit)
                                : Status{};
            if (!marked) {
              failure = marked.error();
              unknown = true;
              return sc::JobResult::kUnknown;
            }
          }
        }
        // Outside any recording: their sources move with the positions.
        for (const auto& [to, from, bytes] : *taps) {
          const auto copied =
              providers::CopyAsync(native, support::Pointer(to), support::Pointer(from), bytes,
                                   providers::CopyKind::kDeviceToDevice);
          if (!copied.ok()) {
            failure = std::format("a chunk tap's carry: {}", copied.text());
            unknown = true;
            return sc::JobResult::kUnknown;
          }
        }
        for (const auto& [to, from, bytes] : *features) {
          if (const providers::DeviceStatus copied =
                  providers::CopyAsync(native, support::Pointer(to), support::Pointer(from), bytes,
                                       providers::CopyKind::kDeviceToDevice);
              !copied.ok()) {
            failure = std::format("a step's features: {}", copied.text());
            unknown = true;
            return sc::JobResult::kUnknown;
          }
        }
        for (const RowRead& read : row_reads) {
          const RowTableState& table = *read.table;
          if (!kernels::paging::GatherPleRows(
                  table.landing, table.sources, table.count,
                  static_cast<std::uint32_t>(read.plan.slots.size()),
                  static_cast<std::uint32_t>(table.table.row_bytes),
                  static_cast<std::byte*>(support::Pointer(table.slots.base)), native.handle)) {
            failure = "a row table's gather failed";
            unknown = true;
            return sc::JobResult::kUnknown;
          }
          saves_queued = true;
        }
        const StepQueued q = program.routed()
                                 ? QueueRoutedPrefix(native)
                                 : programs_.runs().Queue(runs, *copies, {}, program, *outputs,
                                                          capture, graph_stats_, native);
        wrote = wrote || q.before || q.result.has_value();
        report.path = q.path;
        if (!q.result) {
          failure = q.result.error().detail;
          unknown = q.result.error().error == kn::StepError::kUnknown;
          return unknown                 ? sc::JobResult::kUnknown
                 : wrote || saves_queued ? sc::JobResult::kFailed
                                         : sc::JobResult::kNotStarted;
        }
        if (!program.routed())
          for (const auto& [to, from, bytes] : *carries) {
            if (const providers::DeviceStatus copied =
                    providers::CopyAsync(native, support::Pointer(to), support::Pointer(from),
                                         bytes, providers::CopyKind::kDeviceToDevice);
                !copied.ok()) {
              failure = std::format("a step's carried features: {}", copied.text());
              unknown = true;
              return sc::JobResult::kUnknown;
            }
          }
        if (upcoming != nullptr) {
          auto captured = programs_.runs().CaptureAhead(upcoming->runs[0], upcoming_copies,
                                                        *upcoming->planned->program,
                                                        upcoming_outputs, graph_stats_, native);
          if (!captured) {
            // This run's work is queued; the capture's fault is the launch's.
            failure = captured.error().detail;
            unknown = true;
            return sc::JobResult::kUnknown;
          }
          report.captured_ahead = *captured;
        }
        return sc::JobResult::kQueued;
      },
      "a model step", stream_, meanwhile);
  if (program.routed() && posted && failure.empty() && !programs_.launch()->faulted()) {
    if (auto continued = ContinueRouted(report.path); !continued) {
      failure = continued.error();
      unknown = true;  // prefix effects exist; no completion inferred from refusal
      posted = Error(continued.error());
    }
  }
  // The job retired: every optional preparation drained before anything
  // else touches the state.
  const Status preparation = slots_.FinishPreparations();
  // What a drafter step carried, by its slots: nothing until it ran.
  for (const pl::Segment& seg : shape.segments) {
    carry_rows_.at((std::size_t{seg.slot} * slots_.stages()) + shape.stage) = 0;
  }
  if (!posted || !failure.empty() || programs_.launch()->faulted()) {
    slots_.CheckFailedJob();
    for (const pl::Segment& s : shape.segments) {
      (void)slots_.CompleteOwed(s.slot, false,
                                !posted || unknown ? LiveState::CopyRetirement::kUnproven
                                                   : LiveState::CopyRetirement::kProven);
    }
    // What the step's own work may write: its stage, and on the target the
    // drafters' taps (Admit's rule). Another stage changed only by what it
    // owed, which ran whole or stays owed (a partial queueing is unknown).
    const auto written = [&](std::size_t stage) {
      return stage == shape.stage ||
             (shape.stage == 0 &&
              std::ranges::any_of(slots_.layout(stage).items,
                                  [](const auto& item) { return item.contract.target_writes; }));
    };
    for (const pl::Segment& s : shape.segments) {
      for (std::size_t stage = 0; stage < slots_.stages(); ++stage) {
        LiveState& live = slots_.live(s.slot, stage);
        // A verify whose saves were queued is undone whole; anything else
        // the step may have written quarantines.
        const auto sv = std::ranges::find(saving, &live, &Saving::live);
        (void)live.Settle(sv != saving.end() && sv->queued, wrote && written(stage), unknown);
      }
      if (verify) slots_.ProgressVerifyFailed(s.slot);
    }
    if (programs_.launch()->faulted()) slots_.Fault();
    future.Abandon();
    // A faulted cohort (the launch's fault, or one CheckFailedJob found:
    // a scheduler fault, a lost lease) quarantined every owner, written or
    // not: never clean.
    return Refused(!failure.empty() ? failure
                   : !posted        ? posted.error()
                                    : "the launch faulted",
                   !wrote && !unknown && !slots_.faulted() && preparation.has_value());
  }
  for (const pl::Segment& s : shape.segments) {
    if (auto completed = slots_.CompleteOwed(s.slot, true); !completed) {
      future.Abandon();
      return Refused(completed.error(), false);
    }
  }
  // The verify ran: its rows await their Accept in every stage it writes or
  // that commits.
  for (const Saving& sv : saving) sv.live->Verified(sv.rows);
  if (!preparation) {
    // This step's writes completed but no position is published: every
    // owner becomes unusable rather than expose state at an old cursor.
    for (const pl::Segment& s : shape.segments) {
      for (std::size_t stage = 0; stage < slots_.stages(); ++stage) {
        slots_.live(s.slot, stage).Quarantine();
      }
    }
    future.Abandon();
    return Refused(preparation.error(), false);
  }
  const auto installed =
      future.InstallAfterCompletion([&](std::size_t i, std::unique_ptr<StepPlanned> planned,
                                        double seconds, const auto& transfer) {
        return Install(*build[i], std::move(planned), seconds, transfer).has_value();
      });
  for (const bool cached : installed) {
    lookahead_.cached += cached;
    report.cached_ahead += cached;
  }
  for (const kn::OutputSlot& out : program.outputs()) {
    if (out.features && out.segment < shape.segments.size()) {
      const pl::Segment& seg = shape.segments[out.segment];
      carry_rows_.at((std::size_t{seg.slot} * slots_.stages()) + shape.stage) = seg.outputs;
    }
  }
  lookahead_.captured_ahead += report.captured_ahead;
  if (shape.stage == 0) {
    for (const pl::Segment& seg : shape.segments) {
      for (std::size_t stage = 1; stage < slots_.stages(); ++stage) {
        const auto* tap = model::FeatureTap(slots_.layout(stage));
        if (tap == nullptr || tap->contract.cells != model::CellRule::kPerRow) continue;
        const std::size_t at = (std::size_t{seg.slot} * slots_.stages()) + stage;
        const TapRows held = HeldTapRows(seg.slot, stage);
        previous_tap_rows_[at] = verify ? held : TapRows{};
        tap_rows_[at] = {.first = seg.past,
                         .count = seg.rows,
                         .preceding = std::uint64_t{held.first} + held.count == seg.past
                                          ? std::min(held.count, tap->contract.window_or_ratio)
                                          : 0};
      }
    }
  }
  report.captured_first = capture_first && report.path == RunPath::kCaptured;
  lookahead_.captured_first += report.captured_first;
  Count(graph_stats_, report.path);

  for (const auto& segment : shape.segments) {
    const auto* companion = CompanionFor(segment.slot, shape.stage);
    if (companion == nullptr) continue;
    for (const auto& [stage, unit] : companion->units) {
      auto* progress = slots_.progress(segment.slot, stage);
      Status completed = companion->part == CompanionPart::kPre ? progress->PreCompleted(unit)
                         : companion->part == CompanionPart::kTarget
                             ? progress->TargetCompleted(unit)
                             : progress->PostCompleted(unit);
      if (!completed) {
        progress->Quarantine();
        return Refused(completed.error(), false);
      }
    }
  }
  // Each owner's own outputs, then its positions.
  report.outputs.resize(shape.segments.size());
  for (std::size_t i = 0; i < program.outputs().size(); ++i) {
    const kn::OutputSlot& slot = program.outputs()[i];
    const auto* from = support::Pointer((*outputs)[i][0]);
    const auto bytes = std::span(static_cast<const std::byte*>(from), slot.bytes);
    std::vector<std::byte>& to =
        slot.segment == ir::kJoinedSegment ? report.joined : report.outputs.at(slot.segment);
    to.insert(to.end(), bytes.begin(), bytes.end());
  }
  if (request.advance) {
    for (const pl::Segment& s : shape.segments) {
      if (auto set = slots_.SetPositions(s.slot, s.past + s.rows); !set) {
        // Admitted owners always take their positions; should one not, the
        // step's writes are done but not every cursor moved: every owner
        // becomes unusable, and no peer may run the step again (no slot).
        for (const pl::Segment& o : shape.segments) {
          for (std::size_t stage = 0; stage < slots_.stages(); ++stage) {
            slots_.live(o.slot, stage).Quarantine();
          }
        }
        return Refused(set.error(), false);
      }
    }
  }
  return report;
}

// -------------------------------------------------------------------- report

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
