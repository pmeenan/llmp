// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "engine/gemma4_runner.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <format>
#include <limits>
#include <numeric>
#include <string_view>
#include <utility>

#include "artifact/representation.h"
#include "engine/support.h"
#include "kernels/ggml/executor.h"
#include "providers/device_runtime.h"

namespace jitllm::engine {
namespace kg = kernels::ggml;
namespace md = model;
namespace sc = scheduler;
using catalog::ExtentId;
using support::Address;
using support::Error;
using support::Pointer;
using support::Round;
using support::Seconds;

std::expected<std::uint64_t, std::string> Gemma4ExpertPitch(
    std::uint64_t minimum, std::span<const std::string_view> types) {
  if (minimum == 0 || types.empty() || types.size() > 3)
    return Error("Gemma4 expert pitch needs one to three bound arrays");
  std::uint64_t quantum = 256;
  for (const auto type : types) {
    const auto* info = artifact::FindGgmlType(type);
    if (info == nullptr) return Error("Gemma4 expert pitch has unknown type");
    quantum = std::lcm(quantum, std::uint64_t{info->block_bytes});
  }
  if (minimum > UINT64_MAX - (quantum - 1)) return Error("Gemma4 expert pitch overflow");
  return Round(minimum, quantum);
}

std::expected<Gemma4Runner::Slot*, std::string> Gemma4Runner::request_slot(std::uint32_t index) {
  if (released_ || index >= o_.slots || !slots_[index]) return Error("Gemma4 slot unavailable");
  return slots_[index].get();
}
std::array<LiveState*, kMaxRequestSlots> Gemma4Runner::States() {
  std::array<LiveState*, kMaxRequestSlots> states{};
  for (std::size_t i = 0; i < slots_.size(); ++i)
    if (slots_[i] && slots_[i]->provisioned) states[i] = &slots_[i]->live;
  return states;
}
Status Gemma4Runner::CheckActive(const Slot& slot) const {
  if (!bound_ || released_ || !slot.provisioned || slot.spilled)
    return Error("Gemma4 slot is not ready");
  if (auto active = cohort_.Check(node_, stream_, slot.index); !active) return active;
  return slot.live.Usable();
}
Status Gemma4Runner::SelectSlots(std::span<const std::uint32_t> slots) {
  if (!bound_ || released_ || cohort_.faulted()) return Error("Gemma4 cohort unavailable");
  auto mask = cohort_.MaskOf(slots);
  if (!mask) return Error(mask.error());
  cohort_.Select(*mask);
  return RefreshClosures();
}
kg::DeviceChoices Gemma4Runner::Choices(kg::LaunchContext& launch, std::uint32_t rows) const {
  auto choices = kg::DeviceChoicesOf(launch);
  choices.fuse_norms = o_.fuse_norms;
  choices.fuse_norm_rope = o_.fuse_norm_rope;
  choices.fuse_norm_add = o_.fuse_norm_add;
  choices.fuse_gemma_route = o_.fuse_gemma_route;
  choices.fuse_gemma_reduce = o_.fuse_gemma_reduce;
  choices.fuse_rope_store = o_.rope_store;
  // The existing one-row sums contract is bounded to kRowInvariantColumns;
  // wider prefills retain the ordinary primitive product policy.
  choices.row_invariant = o_.row_invariant && rows <= kg::kRowInvariantColumns;
  return choices;
}
Status Gemma4Runner::ReserveWeights() {
  const auto& a = weights_.artifact();
  std::vector<GroupPlace> place(a.groups().size(), GroupPlace::kDevice);
  std::vector<SlabSpec> slabs;
  model_.options.expert_stride.resize(a.expert_arrays().size());
  std::vector<bool> covered(a.groups().size(), false);
  for (std::uint32_t il = 0; il < profile_.layers; ++il) {
    const auto& l = binding_.layers[il];
    std::vector<std::pair<std::uint32_t, std::string_view>> arrays;
    for (const auto* t : {&l.gate_up_exps, &l.gate_exps, &l.up_exps, &l.down_exps})
      if (*t) arrays.emplace_back((**t).index, (**t).type);
    if (arrays.empty()) continue;
    auto slab = ExpertSlab(a, arrays, profile_.experts, 256, il);
    if (!slab) return Error(slab.error());
    // Gemma's bound region bases are 256-aligned. Include that quantum
    // in the expert pitch as well as every GGML block size: a shard change
    // then falls on an alignable page boundary even with a zero/tiny gap.
    std::vector<std::string_view> types;
    for (const auto& [index, type] : arrays) {
      (void)index;
      types.push_back(type);
    }
    auto pitch = Gemma4ExpertPitch(slab->stride, types);
    if (!pitch) return Error(pitch.error());
    pitch_padding_ += (*pitch - slab->stride) * profile_.experts;
    slab->stride = *pitch;
    for (const auto& [index, type] : arrays) model_.options.expert_stride[index] = slab->stride;
    for (std::uint32_t e = 0; e < slab->count; ++e) covered[slab->first_group + e] = true;
    slabs.push_back(*slab);
  }
  for (std::size_t i = 0; i < place.size(); ++i) {
    if (a.groups()[i].kind != artifact::GroupKind::kExpert) continue;
    if (!covered[i]) return Error("Gemma4 expert group is not bound");
    place[i] = GroupPlace::kNone;
  }
  return weights_.Reserve(node_, place, slabs);
}
Status Gemma4Runner::Setup() {
  if (setup_started_ || released_) return Error("Gemma4 setup is not repeatable");
  setup_started_ = true;
  switch (o_.variant) {
    case Gemma4Variant::k26BA4B:
      profile_ = md::Gemma4_26BA4B();
      break;
    case Gemma4Variant::k31B:
      profile_ = md::Gemma4_31B();
      break;
    default:
      return Error("Gemma4 runner variant is not approved");
  }
  if (o_.slots == 0 || o_.slots > kMaxRequestSlots || o_.slots > o_.max_rows ||
      o_.max_rows >= o_.context) {
    return Error("Gemma4 runner needs bounded slots/chunks");
  }
  auto layout = md::Gemma4State(profile_, o_.context, o_.max_rows);
  if (!layout) return Error(layout.error());
  layout_ = std::move(*layout);
  checkpoint_layout_id_ = std::format("gemma{}-f16-kv-scalar-device-v1:{}:{}:{}:{}",
                                      o_.variant == Gemma4Variant::k31B ? 31 : 26, layout_.context,
                                      layout_.max_rows, layout_.global_cells, layout_.local_cells);
  if (auto r = weights_.Open(o_.artifact); !r) return r;
  auto binding = md::BindGemma4(profile_, weights_.artifact());
  if (!binding) return Error(binding.error());
  binding_ = std::move(*binding);
  cohort_.set_slots(o_.slots);
  for (std::uint32_t i = 0; i < o_.slots; ++i) {
    slots_[i] = std::make_unique<Slot>(i);
    if (auto r = slots_[i]->live.AddGrowing(node_, std::format("Gemma4 state slot {}", i),
                                            layout_.bytes, owner_);
        !r)
      return r;
    slots_[i]->provisioned = true;
  }
  model_.profile = &profile_;
  model_.binding = &binding_;
  model_.state = &layout_;
  model_.options.shared_q8 = o_.shared_q8;
  model_.options.rope_store = o_.rope_store;
  model_.options.narrow_final = o_.frontier_head;
  model_.options.device_masks = !o_.reference_masks;
  if (auto r = ReserveWeights(); !r) return r;
  model_.resources.resize(weights_.artifact().resources().size());
  for (std::uint32_t i = 0; i < model_.resources.size(); ++i)
    model_.resources[i] = {weights_.resource_address(i),
                           weights_.artifact().resources()[i].readable.value()};
  model_.arrays.resize(weights_.artifact().expert_arrays().size());
  for (std::uint32_t i = 0; i < model_.arrays.size(); ++i) {
    const auto& a = weights_.artifact().expert_arrays()[i];
    model_.arrays[i] = {weights_.array_address(i),
                        model_.options.expert_stride[i] * (a.count - 1) + a.readable.value()};
  }
  model_.slots.resize(o_.slots);
  for (std::uint32_t i = 0; i < o_.slots; ++i)
    model_.slots[i] = {slots_[i]->live.base(0), layout_.bytes};
  if (auto r = resources_.OpenCublas("Gemma4 cuBLAS workspace"); !r) return r;
  auto measuring = resources_.MeasuringContext();
  if (!measuring) return Error(measuring.error());
  std::uint64_t activation = 0, scratch = 0, staging = 0, host = 0;
  // All slot counts: padding each segment's query tile can exceed a scalar
  // prefill's mask storage. Measure one shared maximum, not one per slot.
  std::vector<std::int32_t> tokens(o_.max_rows, 1);
  for (std::uint32_t count = 1; count <= o_.slots; ++count) {
    for (const auto rows : {1U, o_.max_rows / count, o_.max_rows - count + 1}) {
      for (const auto past : {0U, o_.context - rows}) {
        std::vector<md::Gemma4Segment> segments;
        for (std::uint32_t i = 0; i < count; ++i) {
          const auto segment_rows = rows == o_.max_rows - count + 1 && i != 0 ? 1U : rows;
          const auto segment_past = past == 0 ? 0U : o_.context - segment_rows;
          segments.push_back({i, segment_past, std::span(tokens).first(segment_rows)});
        }
        auto in = md::Gemma4Chunk(profile_, layout_, segments, false);
        if (!in) return Error(in.error());
        auto input_bytes =
            md::Gemma4HostInputBytes(profile_, layout_, segments, o_.reference_masks);
        if (!input_bytes) return Error(input_bytes.error());
        for (const bool all : {false, true}) {
          kg::Gemma4ChunkShape shape;
          for (const auto& s : in->segments)
            shape.segments.push_back({s.slot, s.rows, s.n_past, s.global_n_kv, s.local_n_kv});
          shape.outputs = all ? static_cast<std::uint32_t>(in->tokens.size()) : count;
          auto p = PlanGemma4Chunk(
              model_, shape, Choices(**measuring, static_cast<std::uint32_t>(in->tokens.size())), 0,
              0);
          if (!p) return Error(std::format("measuring Gemma4: {}", p.error()));
          auto needed = kg::PlanScratch(**measuring, (*p)->plan);
          if (!needed) return Error(needed.error().detail);
          activation = std::max(activation, (*p)->placement.extent);
          scratch = std::max(scratch, *needed);
          staging = std::max(staging, (*p)->inputs_bytes);
          auto source_bytes = Gemma4SourceBytes((*p)->graph);
          if (!source_bytes) return Error(source_bytes.error());
          host = std::max(host, *input_bytes + *source_bytes);
          plan_floor_bytes_ = std::max(plan_floor_bytes_, PlannedHostBytes(**p));
        }
      }
    }
  }
  activation_bytes_ = Round(activation + activation / 4, kPagedExtent);
  scratch_bytes_ = Round(scratch + scratch / 4 + (1U << 20U), kPagedExtent);
  host_input_bytes_ = Round(host + (1U << 20U), kPagedExtent);
  const auto staging_bytes = Round(staging + staging / 4 + (1U << 20U), kPagedExtent);
  auto staging_host = resources_.Pinned(staging_bytes);
  auto logits = resources_.Pinned(std::uint64_t{o_.max_rows} * profile_.vocab * sizeof(float));
  auto factors = resources_.Pinned(profile_.global_rope_dims / 2 * sizeof(float));
  if (!staging_host || !logits || !factors) return Error("Gemma4 pinned inputs/outputs/factors");
  runs_.SetStaging(*staging_host, staging_bytes);
  logits_ = *logits;
  factors_ = *factors;
  setup_ = true;
  return {};
}
Status Gemma4Runner::Register() {
  if (!setup_ || registered_ || bound_ || released_) return Error("Gemma4 registration order");
  if (auto r = weights_.Register(node_, owner_); !r) return r;
  std::vector<ExtentId> pinned = weights();
  for (auto& slot : slots_) {
    if (!slot) continue;
    if (auto r = slot->live.RegisterSpill(
            node_,
            o_.spill_place
                ? o_.spill_place(slot->index)
                : LiveState::SpillPlace{.directory = o_.out, .dir = -1, .name = {}, .keep = false});
        !r)
      return r;
    const auto reserved = slot->live.reserved_extents();
    pinned.insert(pinned.end(), reserved.begin(), reserved.end());
  }
  if (auto r = node_.scheduler().PinPlaces(pinned); !r)
    return Error(std::format("Gemma4 pin places: {}", sc::ToString(r.error())));
  registered_ = true;
  return {};
}
Status Gemma4Runner::RefreshClosures(SlotMask protect) {
  auto refreshed = node_.Call(
      [&]() -> Status {
        std::vector<ExtentId> shared = weights();
        for (const auto* mapped : {&node_.activations(), &node_.pool()})
          shared.insert(shared.end(), mapped->extents.begin(), mapped->extents.end());
        const auto own = resources_.extents();
        shared.insert(shared.end(), own.begin(), own.end());
        std::array<const LiveState*, kMaxRequestSlots> states{};
        for (const auto& slot : slots_)
          if (slot && !slot->spilled) states[slot->index] = &slot->live;
        auto c = cohort_.Build(node_.catalog(), shared, states, protect);
        if (!c) return Error(c.error());
        everything_ = std::move(c->everything);
        fence_ = std::move(c->fence);
        execution_ = std::move(c->execution);
        for (auto& slot : slots_)
          if (slot) slot->fence = std::move(c->slot_fences[slot->index]);
        return {};
      },
      "Gemma4 refresh closures");
  auto states = States();
  if (!refreshed) {
    cohort_.Fault(states);
    return refreshed;
  }
  return cohort_.Hold(node_, stream_, execution_, states);
}
Status Gemma4Runner::Bind() {
  if (!registered_ || bound_ || released_) return Error("Gemma4 bind order");
  if (auto r = resources_.BindLaunch(scratch_bytes_); !r) return r;
  runs_.SetLaunch(&resources_.launch());
  account_.Bind(
      [this](std::uint64_t bytes, bool required) { return node_.ChargeHost(bytes, required); },
      [this](std::uint64_t bytes) { node_.UnchargeHost(bytes); });
  plans_.set_account(&account_);
  if (auto r = RefreshClosures(); !r) return r;
  // Only the small factor resource is read into caller-funded pinned host
  // storage. It is validated before any model graph can consume it.
  const auto placed = weights_.artifact().ResourcePlacement(binding_.rope_freqs.index);
  if (!placed) return Error("Gemma4 factor placement");
  const auto group = placed->group;
  const auto factor_extents =
      weights_.ExtentsWhere([group](std::uint32_t g, bool) { return g == group; });
  auto described = node_.Call(
      [&]() -> Status {
        auto c = node_.catalog().ClosureOfExtents(factor_extents);
        if (!c) return Error("Gemma4 factors are not cataloged");
        factor_closure_ = std::move(*c);
        return {};
      },
      "Gemma4 factor closure");
  if (!described) return described;
  bound_ = true;
  return {};
}
Status Gemma4Runner::CheckFactors() {
  if (factors_checked_) return {};
  bool unknown = false;
  if (auto r = node_.Job(
          factor_closure_,
          [&](providers::NativeStream native) {
            unknown =
                !providers::CopyAsync(native, factors_,
                                      Pointer(weights_.resource_address(binding_.rope_freqs.index)),
                                      profile_.global_rope_dims / 2 * sizeof(float),
                                      providers::CopyKind::kDeviceToHost)
                     .ok();
            return unknown ? sc::JobResult::kUnknown : sc::JobResult::kQueued;
          },
          "Gemma4 read frequency factors", stream_);
      !r) {
    auto states = States();
    cohort_.CheckFailedJob(node_, stream_, factor_closure_, states);
    if (unknown) cohort_.Fault(states);
    return r;
  }
  if (auto checked = md::CheckGemma4RopeFactors(
          profile_, std::span(static_cast<const float*>(factors_), profile_.global_rope_dims / 2));
      !checked)
    return Error(checked.error());
  factors_checked_ = true;
  return {};
}
Status Gemma4Runner::CheckPlaces() {
  PlaceCheck check;
  auto r = node_.Call(
      [&]() -> Status {
        weights_.CheckPlaces(node_.scheduler(), check);
        for (const auto& slot : slots_)
          if (slot) slot->live.CheckPlaces(node_.scheduler(), check);
        return {};
      },
      "Gemma4 check pinned places");
  if (!r) return r;
  if (check.moved != 0) {
    DropPlans();
    return Error(std::format("Gemma4 {} places moved: {}", check.moved, check.first));
  }
  return {};
}
std::vector<ExtentId> Gemma4Runner::state() const {
  std::vector<ExtentId> all;
  for (const auto& slot : slots_) {
    if (!slot) continue;
    const auto extents = slot->live.extents();
    all.insert(all.end(), extents.begin(), extents.end());
  }
  return all;
}
std::vector<ExtentId> Gemma4Runner::managed_extents() const {
  auto all = weights();
  const auto live = state();
  all.insert(all.end(), live.begin(), live.end());
  return all;
}
std::expected<std::vector<LiveState::Range>, std::string> Gemma4Runner::CheckpointRanges(
    std::uint32_t positions) const {
  auto needed = md::Gemma4UsedState(profile_, layout_, positions);
  if (!needed) return Error(needed.error());
  std::vector<LiveState::Range> ranges;
  for (const auto& r : *needed) ranges.push_back({0, r.offset, r.bytes});
  return ranges;
}
Status Gemma4Runner::ReserveStateThrough(std::uint32_t index, std::uint32_t positions) {
  auto request = request_slot(index);
  if (!request) return Error(request.error());
  auto& slot = **request;
  slot.state_refused = false;
  if (auto active = CheckActive(slot); !active) return active;
  auto ranges = CheckpointRanges(positions);
  if (!ranges) return Error(ranges.error());
  bool over_budget = false;
  auto used = slot.live.Use(node_, *ranges, &execution_, &over_budget);
  if (used && *used) slot.on_disk = false;
  // Even a partial clean refusal can have initialized pages: protect them.
  if ((!used || *used)) {
    if (auto r = RefreshClosures(); !r) return r;
  }
  if (!used) {
    slot.state_refused = over_budget && !cohort_.faulted() && !slot.live.quarantined();
    return Error(used.error());
  }
  return {};
}
Status Gemma4Runner::Clear(std::uint32_t index) {
  auto request = request_slot(index);
  if (!request) return Error(request.error());
  auto& slot = **request;
  if (!bound_ || cohort_.faulted()) return Error("Gemma4 retirement required");
  if (auto r = cohort_.Check(node_, stream_, index); !r) return r;
  if (auto r = RefreshClosures(cohort_.active() & ~(SlotMask{1} << index)); !r) return r;
  const auto cleared = slot.live.DiscardGrowingState(node_);
  if (cleared) {
    slot.spilled = false;
    slot.positions = 0;
    slot.state_refused = false;
    slot.on_disk = false;
    slot.adopted.clear();
    slot.adopted_bytes = 0;
    slot.restoring.reset();
    slot.restore_needed.clear();
    slot.restored_bytes.clear();
  }
  if (auto r = RefreshClosures(); !r) return r;
  return cleared;
}
Status Gemma4Runner::ClearIdle(std::uint32_t index) {
  auto request = request_slot(index);
  if (!request) return Error(request.error());
  if (!bound_ || cohort_.faulted() || (cohort_.IsActive(index) && node_.InRequest(stream_)))
    return Error("Gemma4 idle clear requires an idle healthy slot");
  auto& slot = **request;
  const auto cleared = slot.live.DiscardGrowingState(node_);
  if (cleared) {
    slot.spilled = false;
    slot.positions = 0;
    slot.state_refused = false;
    slot.on_disk = false;
    slot.adopted.clear();
    slot.adopted_bytes = 0;
    slot.restoring.reset();
    slot.restore_needed.clear();
    slot.restored_bytes.clear();
  }
  if (auto r = RefreshClosures(); !r) return r;
  return cleared;
}
Status Gemma4Runner::Spill(std::uint32_t index) {
  auto request = request_slot(index);
  if (!request) return Error(request.error());
  auto& slot = **request;
  if (!bound_ || cohort_.faulted()) return Error("Gemma4 retirement required");
  if (slot.spilled) return {};
  if (auto r = slot.live.Usable(); !r) return r;
  const auto extents = slot.live.extents();
  if (extents.empty()) return {};
  slot.spilled = true;
  if (auto r = RefreshClosures(); !r) {
    slot.spilled = false;
    return r;
  }
  auto evicted = node_.Evict(extents);
  slot.on_disk = evicted.has_value();
  return evicted;
}
Status Gemma4Runner::Restore(std::uint32_t index) {
  auto request = request_slot(index);
  if (!request) return Error(request.error());
  auto& slot = **request;
  slot.state_refused = false;
  if (!bound_ || cohort_.faulted()) return Error("Gemma4 retirement required");
  if (!slot.spilled) return {};
  if (!slot.adopted.empty()) {
    bool over_budget = false;
    auto used = slot.live.Use(node_, slot.adopted, &execution_, &over_budget);
    if (!used) {
      slot.state_refused = over_budget && !cohort_.faulted() && slot.state_usable();
      return Error(used.error());
    }
    slot.adopted.clear();
    slot.adopted_bytes = 0;
    slot.spilled = false;
    slot.on_disk = true;
    return RefreshClosures();
  }
  catalog::Closure restore;
  const auto extents = slot.live.extents();
  if (auto r = node_.Call(
          [&]() -> Status {
            auto c = node_.catalog().ClosureOfExtents(extents);
            if (!c) return Error("Gemma4 spilled state not cataloged");
            restore = std::move(*c);
            return {};
          },
          "Gemma4 describe restore");
      !r)
    return r;
  sc::AcquireReport report;
  bool over_budget = false;
  if (auto r = node_.Acquire(restore, report, "Gemma4 restore", &over_budget); !r) {
    slot.state_refused = over_budget && !cohort_.faulted();
    return r;
  }
  slot.spilled = false;
  slot.on_disk = true;
  return RefreshClosures();
}
Status Gemma4Runner::CopyState(std::uint32_t index, void* pinned,
                               std::span<const LiveState::Range> ranges, bool to_host,
                               LiveState::CopyRetirement* retirement) {
  auto request = request_slot(index);
  if (!request) return Error(request.error());
  if (auto active = CheckActive(**request); !active) return active;
  if (!to_host) (*request)->on_disk = false;
  LiveState::CopyRetirement completed;
  auto copied =
      (*request)->live.Copy(node_, (*request)->fence, stream_, pinned, ranges, to_host, &completed);
  if (retirement) *retirement = completed;
  if (!copied) {
    auto states = States();
    cohort_.CheckFailedJob(node_, stream_, execution_, states);
    if (completed == LiveState::CopyRetirement::kUnproven) cohort_.Fault(states);
  }
  if (copied && !to_host && (*request)->restoring) {
    auto& slot = **request;
    for (std::size_t i = 0; i < slot.restore_needed.size(); ++i) {
      const auto& needed = slot.restore_needed[i];
      for (const auto& r : ranges) {
        const auto at = needed.offset + slot.restored_bytes[i];
        if (r.region == needed.region && r.offset <= at && r.bytes > at - r.offset)
          slot.restored_bytes[i] = std::min(needed.bytes, r.offset + r.bytes - needed.offset);
      }
    }
  }
  return copied;
}
Status Gemma4Runner::RestoreCheckpoint(std::uint32_t index, std::uint32_t positions, void* pinned,
                                       std::span<const LiveState::Range> ranges,
                                       std::string_view source_layout,
                                       LiveState::CopyRetirement* retirement) {
  if (retirement) *retirement = LiveState::CopyRetirement::kProven;
  if (source_layout.empty() || source_layout != checkpoint_layout_id_)
    return Error("Gemma4 checkpoint source layout differs");
  if (positions != 0 && pinned == nullptr) return Error("Gemma4 checkpoint has no pinned buffer");
  auto request = request_slot(index);
  if (!request) return Error(request.error());
  if (auto r = CheckActive(**request); !r) return r;
  if ((*request)->positions != 0 || (*request)->live.used_bytes() != 0)
    return Error("Gemma4 checkpoint restore requires an empty slot");
  auto expected = CheckpointRanges(positions);
  if (!expected) return Error(expected.error());
  if (ranges.size() != expected->size()) return Error("Gemma4 checkpoint footprint differs");
  for (std::size_t i = 0; i < ranges.size(); ++i)
    if (ranges[i].region != (*expected)[i].region || ranges[i].offset != (*expected)[i].offset ||
        ranges[i].bytes != (*expected)[i].bytes)
      return Error("Gemma4 checkpoint footprint differs");
  if (auto r = ReserveStateThrough(index, positions); !r) return r;
  if (auto r = CopyState(index, pinned, ranges, false, retirement); !r) {
    (*request)->live.Quarantine();
    return r;
  }
  (*request)->positions = positions;
  return {};
}
void Gemma4Runner::StateWrittenBack(bool whole) {
  for (auto& slot : slots_)
    if (slot && !slot->spilled) slot->on_disk = whole && slot->state_usable();
}
std::expected<void, std::string> Gemma4CheckpointFootprint(
    const model::Gemma4Profile& profile, const model::Gemma4StateLayout& layout,
    std::uint32_t positions, std::span<const LiveState::Range> ranges) {
  auto needed = md::Gemma4UsedState(profile, layout, positions);
  if (!needed) return Error(needed.error());
  std::uint64_t previous = 0;
  for (const auto& r : ranges) {
    if (r.region != 0 || r.offset >= layout.bytes || r.offset % kPagedExtent != 0 ||
        r.bytes != std::min(kPagedExtent, layout.bytes - r.offset) || r.offset < previous)
      return Error("Gemma4 restored footprint is not an ordered initialized extent set");
    previous = r.offset + r.bytes;
  }
  for (const auto& r : *needed) {
    const auto first = r.offset / kPagedExtent;
    const auto last = (r.offset + r.bytes - 1) / kPagedExtent;
    for (auto extent = first; extent <= last; ++extent)
      if (std::ranges::none_of(
              ranges, [extent](const auto& got) { return got.offset / kPagedExtent == extent; }))
        return Error("Gemma4 restored footprint does not fund its completed positions");
  }
  if (positions == 0 && !ranges.empty()) return Error("Gemma4 empty restore has backing");
  return {};
}
Status Gemma4Runner::ValidateFootprint(std::uint32_t positions,
                                       std::span<const LiveState::Range> ranges) const {
  return Gemma4CheckpointFootprint(profile_, layout_, positions, ranges);
}
Status Gemma4Runner::PrepareRestore(std::uint32_t index, std::uint32_t positions,
                                    std::span<const LiveState::Range> footprint,
                                    std::string_view source_layout) {
  if (source_layout.empty() || source_layout != checkpoint_layout_id_)
    return Error("Gemma4 checkpoint source layout differs");
  if (auto r = ValidateFootprint(positions, footprint); !r) return r;
  auto request = request_slot(index);
  if (!request) return Error(request.error());
  auto& slot = **request;
  if (auto r = CheckActive(slot); !r) return r;
  if (slot.restoring) return Error("Gemma4 restore is already pending");
  // Retain needs this destination unleased; every selected peer stays held.
  if (auto r = RefreshClosures(cohort_.active() & ~(SlotMask{1} << index)); !r) return r;
  slot.state_refused = false;
  bool over_budget = false;
  auto used = slot.live.Use(node_, footprint, &execution_, &over_budget);
  if (!used || *used) slot.on_disk = false;
  Status prepared;
  if (!used) {
    slot.state_refused = over_budget && !cohort_.faulted() && slot.state_usable();
    prepared = Error(used.error());
    if (!slot.state_refused) slot.live.Quarantine();
  } else {
    prepared = slot.live.Retain(node_, footprint);
    if (!prepared) slot.live.Quarantine();
  }
  if (auto r = RefreshClosures(); !r) return r;
  if (!prepared) return prepared;
  slot.on_disk = false;
  auto needed = CheckpointRanges(positions);
  if (!needed) return Error(needed.error());
  slot.restore_needed = std::move(*needed);
  slot.restored_bytes.assign(slot.restore_needed.size(), 0);
  slot.restoring = positions;
  return {};
}
Status Gemma4Runner::CompleteRestore(std::uint32_t index, std::uint32_t positions) {
  auto request = request_slot(index);
  if (!request) return Error(request.error());
  auto& slot = **request;
  if (auto r = CheckActive(slot); !r) return r;
  if (!slot.restoring || *slot.restoring != positions)
    return Error("Gemma4 restore completion differs from its checked metadata");
  for (std::size_t i = 0; i < slot.restore_needed.size(); ++i)
    if (slot.restored_bytes[i] != slot.restore_needed[i].bytes)
      return Error("Gemma4 restore has not completed every logical checkpoint range");
  slot.positions = positions;
  slot.restoring.reset();
  slot.restore_needed.clear();
  slot.restored_bytes.clear();
  return {};
}
Status Gemma4Runner::Adopt(std::uint32_t index, std::uint32_t positions,
                           std::span<const LiveState::Range> footprint,
                           std::string_view source_layout) {
  if (source_layout.empty() || source_layout != checkpoint_layout_id_)
    return Error("Gemma4 checkpoint source layout differs");
  if (auto r = ValidateFootprint(positions, footprint); !r) return r;
  auto request = request_slot(index);
  if (!request) return Error(request.error());
  auto& slot = **request;
  if (!bound_ || cohort_.faulted() || positions == 0 || slot.positions != 0 || slot.spilled ||
      slot.live.used_bytes() != 0 || Held(index))
    return Error("Gemma4 adopts kept state only into an empty idle healthy slot");
  auto bytes = slot.live.UsedBytesOf(footprint);
  if (!bytes) return Error(bytes.error());
  slot.adopted.assign(footprint.begin(), footprint.end());
  slot.adopted_bytes = *bytes;
  slot.spilled = true;
  slot.positions = positions;
  slot.on_disk = true;
  return RefreshClosures();
}
std::expected<Gemma4Runner::Plans::Entry*, std::string> Gemma4Runner::Planned(
    const kg::Gemma4ChunkShape& shape) {
  if (auto* found = plans_.Find(shape)) return found;
  const auto started = std::chrono::steady_clock::now();
  std::uint32_t rows = 0;
  for (const auto& segment : shape.segments) rows += segment.rows;
  auto p = PlanGemma4Chunk(model_, shape, Choices(resources_.launch(), rows),
                           node_.activations().base, node_.activations().bytes);
  if (!p) return Error(p.error());
  if (auto r = BindPlanned(**p, resources_.launch(), resources_.registry(), "Gemma4 chunk"); !r)
    return Error(r.error());
  policy_ = {.rows = rows, .segments = static_cast<std::uint32_t>(shape.segments.size())};
  for (const auto& step : (*p)->plan.steps) {
    policy_.norm_fused += step.implementation == "ggml.rms_norm_mul.fused";
    policy_.norm_rope += step.implementation == "ggml.rms_norm_mul_rope.fused";
    policy_.norm_add += step.implementation == "ggml.rms_norm_mul_add.fused";
    policy_.gemma_route += step.implementation == "ggml.gemma.route.fused";
    policy_.gemma_reduce += step.implementation == "ggml.gemma.scaled_reduce.fused";
    policy_.rope_store += step.implementation == "ggml.rope_set_rows.fused";
    policy_.shared_vecq += step.implementation == "jitllm.vecq";
    policy_.row_products +=
        step.implementation.ends_with("mmvq_rows") || step.implementation.ends_with("mmvf_rows");
    policy_.lane_steps += step.lane != 0;
  }
  std::vector<const ggml_tensor*> state_tensors;
  for (const auto& segment : (*p)->graph.segments)
    for (const auto& [k, v] : segment.caches) {
      state_tensors.push_back(k);
      state_tensors.push_back(v);
    }
  CheckCoverage(node_, owner_, (*p)->graph.nodes,
                {.state = state_tensors, .inputs = (*p)->graph.inputs}, coverage_);
  if (coverage_.violations != 0)
    return Error(std::format("Gemma4 catalog coverage: {}", coverage_.first_violation));
  const auto bytes = PlannedHostBytes(**p);
  const auto nodes = PlannedNodes(**p);
  return &plans_.Add(shape, std::move(*p), bytes, nodes,
                     Seconds(std::chrono::steady_clock::now() - started));
}
Status Gemma4Runner::Chunk(std::uint32_t n_past, std::span<const std::int32_t> tokens,
                           std::vector<float>& logits, bool all_outputs) {
  const Work work{0, n_past, tokens, &logits};
  return Wave(std::span(&work, 1), all_outputs);
}
Status Gemma4Runner::Wave(std::span<const Work> work, bool all_outputs) {
  const PlanStep step;
  if (!bound_ || released_ || work.empty() || work.size() > o_.slots)
    return Error("Gemma4 wave is unavailable or unbounded");
  std::array<md::Gemma4Segment, kMaxRequestSlots> segments{};
  std::array<bool, kMaxRequestSlots> seen{};
  std::uint32_t rows = 0;
  for (std::size_t i = 0; i < work.size(); ++i) {
    const auto& w = work[i];
    if (w.slot >= o_.slots || seen[w.slot] || w.logits == nullptr || w.tokens.empty() ||
        w.tokens.size() > o_.max_rows - rows || w.n_past != slots_[w.slot]->positions ||
        slots_[w.slot]->restoring)
      return Error("Gemma4 wave needs distinct slots, bounded rows and exact continuations");
    if (auto r = CheckActive(*slots_[w.slot]); !r) return r;
    for (std::size_t j = 0; j < i; ++j)
      if (work[j].logits == w.logits) return Error("Gemma4 output vectors must be independent");
    rows += static_cast<std::uint32_t>(w.tokens.size());
    seen[w.slot] = true;
    segments[i] = {w.slot, w.n_past, w.tokens};
  }
  const auto selected = std::span(segments).first(work.size());
  auto bytes = md::Gemma4HostInputBytes(profile_, layout_, selected, o_.reference_masks);
  if (!bytes) return Error(bytes.error());
  if (*bytes > host_input_bytes_) return Error("Gemma4 host descriptor envelope exceeded");
  if (auto r = CheckPlaces(); !r) return r;
  if (auto r = CheckFactors(); !r) return r;
  // Fund host descriptors and masks before their allocation. The caller's
  // floor covers the measured maximum; optional charging can refuse cleanly.
  struct HostGrant {
    PagedNode& node;
    std::uint64_t bytes;
    ~HostGrant() { node.UnchargeHost(bytes); }
  };
  if (!node_.ChargeHost(host_input_bytes_, false)) return Error("Gemma4 host inputs do not fit");
  const HostGrant grant{node_, host_input_bytes_};
  auto in = md::Gemma4Chunk(profile_, layout_, selected, o_.reference_masks);
  if (!in) return Error(in.error());
  kg::Gemma4ChunkShape shape;
  std::vector<std::int32_t> frontier;
  frontier.reserve(rows);
  for (const auto& s : in->segments) {
    shape.segments.push_back({s.slot, s.rows, s.n_past, s.global_n_kv, s.local_n_kv});
    if (all_outputs) {
      for (std::uint32_t i = 0; i < s.rows; ++i)
        frontier.push_back(static_cast<std::int32_t>(s.first_row + i));
    } else
      frontier.push_back(static_cast<std::int32_t>(s.first_row + s.rows - 1));
    if (auto r = ReserveStateThrough(s.slot, s.n_past + s.rows); !r) return r;
  }
  shape.outputs = static_cast<std::uint32_t>(frontier.size());
  auto entry_of = Planned(shape);
  if (!entry_of) return Error(entry_of.error());
  auto& entry = **entry_of;
  auto& p = *entry.planned;
  auto source_bytes = Gemma4SourceBytes(p.graph);
  if (!source_bytes || *bytes > host_input_bytes_ || *source_bytes > host_input_bytes_ - *bytes)
    return Error("Gemma4 host input envelope exceeded");
  auto host = Gemma4Sources(p.graph, *in, frontier, {}, host_input_bytes_ - *bytes);
  if (!host) return Error(host.error());
  auto copies = runs_.Stage(host->sources, 0);
  if (!copies) return Error(copies.error());
  bool capture = entry.runs[0].CaptureDue(runs_.graphs());
  if (capture && !plans_.ChargeGraph(entry)) capture = false;
  const std::uint64_t output_bytes = std::uint64_t{shape.outputs} * profile_.vocab * sizeof(float);
  const std::array<RunCopy, 1> outputs{
      {{Address(logits_), Address(p.graph.logits->data), output_bytes}}};
  bool wrote = false, unknown = false;
  Status queued;
  RunPath path = RunPath::kEager;
  const auto posted = node_.Job(
      execution_,
      [&](providers::NativeStream native) {
        const auto result = runs_.Queue(entry.runs[0], *copies, {}, *p.bound, outputs, capture,
                                        graph_stats_, native);
        wrote = result.before || result.result.has_value();
        path = result.path;
        if (!result.result) {
          queued = Error(result.result.error().detail);
          unknown = result.result.error().error == kg::KernelError::kUnknown;
          return unknown         ? sc::JobResult::kUnknown
                 : result.before ? sc::JobResult::kFailed
                                 : sc::JobResult::kNotStarted;
        }
        return sc::JobResult::kQueued;
      },
      "Gemma4 chunk/wave", stream_);
  if (!posted || !queued || resources_.launch().faulted()) {
    for (const auto& w : work)
      if (wrote || unknown) slots_[w.slot]->live.Quarantine();
    auto states = States();
    cohort_.CheckFailedJob(node_, stream_, execution_, states);
    if (resources_.launch().faulted()) cohort_.Fault(states);
    if (!queued) return queued;
    return !posted ? posted : Error("Gemma4 launch context faulted");
  }
  Count(graph_stats_, path);
  std::size_t at = 0;
  for (const auto& w : work) {
    const auto count = all_outputs ? w.tokens.size() : 1;
    const auto n = count * profile_.vocab;
    const auto* values = static_cast<const float*>(logits_) + at;
    w.logits->assign(values, values + n);
    at += n;
    slots_[w.slot]->positions += static_cast<std::uint32_t>(w.tokens.size());
    slots_[w.slot]->on_disk = false;
  }
  return {};
}
void Gemma4Runner::DropPlans() { plans_.Clear(); }
void Gemma4Runner::ReclaimCandidates(std::uint32_t owner, bool running,
                                     std::vector<memory::ReclaimCandidate>& out) {
  if (released_ || cohort_.faulted()) return;
  std::array<PlanCacheBase*, 1> caches{&plans_};
  CollectPlans(caches, owner, running, out);
}
std::uint64_t Gemma4Runner::Reclaim(memory::ReclaimKind kind, std::uint64_t id) {
  if (released_ || cohort_.faulted()) return 0;
  std::array<PlanCacheBase*, 1> caches{&plans_};
  return ReclaimPlan(caches, kind, id);
}
Status Gemma4Runner::Release() {
  if (released_) return {};
  released_ = true;
  DropPlans();
  std::vector<std::string> problems;
  resources_.Release(problems);
  for (auto& slot : slots_)
    if (slot) slot->live.Release(node_.memory(), problems);
  if (auto r = weights_.Release(node_.memory()); !r) problems.push_back(r.error());
  return support::Joined(problems);
}
}  // namespace jitllm::engine
