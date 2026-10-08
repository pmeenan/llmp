// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "engine/gemma2_runner.h"

#include <algorithm>
#include <chrono>
#include <format>
#include <string>
#include <utility>

#include "engine/prefill_lookahead.h"
#include "engine/support.h"
#include "kernels/ggml/executor.h"
#include "kernels/ggml/gemma_norm.h"
#include "kernels/ggml/jitllm_ops.h"

namespace jitllm::engine {
namespace kg = kernels::ggml;
namespace md = model;
namespace sc = scheduler;
using catalog::ExtentId;
using support::Address;
using support::Error;
using support::Round;
using support::Seconds;

Gemma2Runner::Gemma2Runner(PagedNode& node, Gemma2Options options, int owner, std::uint32_t stream)
    : node_(node),
      o_(std::move(options)),
      owner_(owner),
      stream_(stream),
      resources_(node, owner, stream),
      runs_(o_.graphs) {}
Gemma2Runner::~Gemma2Runner() = default;
kg::DeviceChoices Gemma2Runner::Choices(kg::LaunchContext& launch) const {
  auto choices = kg::DeviceChoicesOf(launch);
  choices.fuse_norms = o_.fuse_norms;
  choices.fuse_quant_glu = o_.fuse_quant_glu;
  choices.fuse_norm_rope = o_.fuse_norm_rope;
  choices.fuse_norm_add = o_.fuse_norm_add;
  return choices;
}
Status Gemma2Runner::Setup() {
  if (setup_started_ || released_) return Error("Gemma2 setup is not repeatable");
  setup_started_ = true;
  if (o_.prefill_lookahead_capacity < 1 || o_.prefill_lookahead_capacity > 2)
    return Error("Gemma2 lookahead capacity must be one or two");
  if ((o_.slots != 1 && o_.slots != 2) || o_.slots > o_.max_rows || o_.max_rows >= o_.context)
    return Error("Gemma2 runner requires one or two bounded slots/chunk");
  const auto wave_rows = o_.max_wave_rows == 0 ? o_.max_rows : o_.max_wave_rows;
  const auto head_rows = o_.max_head_rows == 0 ? o_.max_rows : o_.max_head_rows;
  if (wave_rows < o_.max_rows || wave_rows > md::kGemma2MaxRows ||
      wave_rows > std::uint64_t{o_.max_rows} * o_.slots)
    return Error("Gemma2 total wave rows exceed their bounded per-slot envelope");
  if (head_rows < o_.slots || head_rows > o_.max_rows)
    return Error("Gemma2 head capacity must cover slots within max_rows");
  auto layout = md::Gemma2State(profile_, o_.context, o_.max_rows);
  if (!layout) return Error(layout.error());
  layout_ = std::move(*layout);
  checkpoint_layout_id_ = std::format("gemma2-2b-f16-kv-device-v1:{}:{}:{}:{}", o_.context,
                                      o_.max_rows, layout_.global_cells, layout_.local_cells);
  if (auto r = weights_.Open(o_.artifact); !r) return r;
  auto binding = md::BindApprovedGemma2(weights_.artifact());
  if (!binding) return Error(binding.error());
  binding_ = std::move(*binding);
  cohort_.set_slots(o_.slots);
  for (std::uint32_t i = 0; i < o_.slots; ++i) {
    slots_[i] = std::make_unique<Slot>(i);
    if (auto r = slots_[i]->live.AddGrowing(node_, std::format("Gemma2 state slot {}", i),
                                            layout_.bytes, owner_);
        !r)
      return r;
    slots_[i]->provisioned = true;
  }
  model_.profile = &profile_;
  model_.binding = &binding_;
  model_.state = &layout_;
  model_.options.narrow_final = o_.frontier_head;
  model_.options.owner_decode = o_.owner_decode;
  model_.options.shared_q8 = o_.shared_q8;
  model_.options.packed_prefill = o_.packed_prefill;
  model_.options.owner_prefill = o_.owner_prefill;
  model_.options.flexible_owner_prefill = o_.flexible_owner_prefill;
  model_.options.bounded_roots = o_.bounded_roots;
  model_.options.device_masks = o_.device_masks;
  model_.options.max_total_rows = wave_rows;
  std::vector<GroupPlace> places(weights_.artifact().groups().size(), GroupPlace::kDevice);
  if (auto r = weights_.Reserve(node_, places, {}); !r) return r;
  model_.resources.resize(weights_.artifact().resources().size());
  for (std::uint32_t i = 0; i < model_.resources.size(); ++i)
    model_.resources[i] = {weights_.resource_address(i),
                           weights_.artifact().resources()[i].readable.value()};
  model_.slots.resize(o_.slots);
  for (std::uint32_t i = 0; i < o_.slots; ++i)
    model_.slots[i] = {slots_[i]->live.base(0), layout_.bytes};
  if (auto r = resources_.OpenCublas("Gemma2 cuBLAS workspace"); !r) return r;
  auto measuring = resources_.MeasuringContext();
  if (!measuring) return Error(measuring.error());
  std::uint64_t activation = 0, scratch = 0, staging = 0, host = 0;
  std::vector<std::int32_t> tokens(o_.max_rows, 1);
  // Scalar chunks and equal/ragged waves at both context endpoints. Head
  // publication has its own immutable capacity; products still share columns.
  for (const auto budget : {wave_rows, head_rows}) {
    for (std::uint32_t count = 1; count <= o_.slots; ++count) {
      for (const auto& row_counts :
           support::ChunkMeasurementRows(o_.max_rows, budget, count, o_.shared_q8)) {
        const auto rows = *std::ranges::max_element(row_counts);
        for (const auto past : {0U, o_.context - rows}) {
          for (const bool mixed : {false, true}) {
            if (mixed && (count != 2 || past != 0 || !o_.owner_decode)) continue;
            std::vector<md::Gemma2Segment> segments;
            for (std::uint32_t i = 0; i < count; ++i) {
              const auto n = row_counts[i];
              segments.push_back({i, (past == 0 && (!mixed || i == 0)) ? 0U : o_.context - n,
                                  std::span(tokens).first(n)});
            }

            auto input =
                md::Gemma2Chunk(profile_, layout_, segments, !o_.device_masks, 256, wave_rows);
            auto input_bytes = md::Gemma2HostInputBytes(profile_, layout_, segments,
                                                        !o_.device_masks, 256, wave_rows);
            if (!input || !input_bytes) return Error("Gemma2 measuring input contract");
            for (const auto output : {0U, 1U, 2U, 3U}) {
              const bool all = output == 1, state_only = output == 2, greedy = output == 3;
              if (all ? budget != head_rows : budget != wave_rows) continue;
              kg::Gemma2ChunkShape shape;
              for (const auto& s : input->segments)
                shape.segments.push_back({s.slot, s.rows, s.n_past, s.global_n_kv, s.local_n_kv});
              shape.output_mode =
                  state_only ? kg::Gemma2OutputMode::kStateOnly : kg::Gemma2OutputMode::kHead;
              shape.greedy = greedy;
              shape.outputs = state_only ? 0U
                              : all      ? static_cast<std::uint32_t>(input->tokens.size())
                                         : count;
              auto p = PlanGemma2Chunk(model_, shape, Choices(**measuring), 0, 0);
              if (!p) return Error(std::format("measuring Gemma2: {}", p.error()));
              auto needed = kg::PlanScratch(**measuring, (*p)->plan);
              if (!needed) return Error(needed.error().detail);
              activation = std::max(activation, (*p)->placement.extent);
              scratch = std::max(scratch, *needed);
              staging = std::max(staging, (*p)->inputs_bytes);
              auto source_bytes = Gemma2SourceBytes((*p)->graph);
              if (!source_bytes) return Error(source_bytes.error());
              host = std::max(host, *input_bytes + *source_bytes);
              plan_floor_bytes_ = std::max(plan_floor_bytes_, PlannedHostBytes(**p));
            }
          }
        }
      }
    }
    if (head_rows == wave_rows) break;
  }
  activation_bytes_ = Round(activation + activation / 4, kPagedExtent);
  scratch_bytes_ = Round(scratch + scratch / 4 + (1U << 20U), kPagedExtent);
  host_input_bytes_ = Round(host + (1U << 20U), kPagedExtent);
  const auto staging_bytes = Round(staging + staging / 4 + (1U << 20U), kPagedExtent);
  auto staging_host = resources_.Pinned(staging_bytes);
  auto logits = resources_.Pinned(std::uint64_t{head_rows} * profile_.vocab * sizeof(float));
  if (!staging_host || !logits) return Error("Gemma2 pinned inputs/outputs");
  runs_.SetStaging(*staging_host, staging_bytes);
  logits_ = *logits;
  setup_ = true;
  return {};
}
Status Gemma2Runner::Register() {
  if (!setup_ || registered_ || bound_ || released_) return Error("Gemma2 registration order");
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
    return Error(std::format("Gemma2 pin places: {}", sc::ToString(r.error())));
  registered_ = true;
  return {};
}
std::array<LiveState*, kMaxRequestSlots> Gemma2Runner::States() {
  std::array<LiveState*, kMaxRequestSlots> states{};
  for (std::size_t i = 0; i < slots_.size(); ++i)
    if (slots_[i] && slots_[i]->provisioned) states[i] = &slots_[i]->live;
  return states;
}
Status Gemma2Runner::RefreshClosures(SlotMask protect) {
  places_clean_.reset();
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
      "Gemma2 refresh closures");
  auto states = States();
  if (!refreshed) {
    cohort_.Fault(states);
    return refreshed;
  }
  return cohort_.Hold(node_, stream_, execution_, states);
}
Status Gemma2Runner::Bind() {
  if (!registered_ || bound_ || released_) return Error("Gemma2 bind order");
  if (auto r = resources_.BindLaunch(scratch_bytes_); !r) return r;
  runs_.SetLaunch(&resources_.launch());
  account_.Bind(
      [this](std::uint64_t bytes, bool required) { return node_.ChargeHost(bytes, required); },
      [this](std::uint64_t bytes) { node_.UnchargeHost(bytes); });
  plans_.set_account(&account_);
  if (auto r = RefreshClosures(); !r) return r;
  bound_ = true;
  return {};
}
std::expected<Gemma2Runner::Slot*, std::string> Gemma2Runner::request_slot(std::uint32_t index) {
  if (released_ || index >= o_.slots || !slots_[index]) return Error("Gemma2 slot unavailable");
  return slots_[index].get();
}
Status Gemma2Runner::CheckActive(const Slot& slot) const {
  if (!bound_ || released_ || !slot.provisioned || slot.spilled)
    return Error("Gemma2 slot is not ready");
  if (auto active = cohort_.Check(node_, stream_, slot.index); !active) return active;
  return slot.live.Usable();
}
Status Gemma2Runner::SelectSlots(std::span<const std::uint32_t> slots) {
  if (!bound_ || released_ || cohort_.faulted()) return Error("Gemma2 cohort unavailable");
  auto mask = cohort_.MaskOf(slots);
  if (!mask) return Error(mask.error());
  for (const auto& slot : slots_)
    if (slot && slot->restoring && cohort_.IsActive(slot->index) &&
        ((*mask >> slot->index) & 1U) == 0)
      return Error("Gemma2 selection cannot remove a pending restore peer");
  if (*mask == cohort_.active() && node_.InRequest(stream_)) return {};
  cohort_.Select(*mask);
  return RefreshClosures();
}
Status Gemma2Runner::CheckPlaces() {
  const auto changes = node_.scheduler().placement_changes();
  if (places_clean_ == changes) return {};
  PlaceCheck check;
  auto r = node_.Call(
      [&]() -> Status {
        weights_.CheckPlaces(node_.scheduler(), check);
        for (const auto& slot : slots_)
          if (slot) slot->live.CheckPlaces(node_.scheduler(), check);
        return {};
      },
      "Gemma2 check pinned places");
  if (!r) return r;
  if (check.moved != 0) {
    DropPlans();
    return Error(std::format("Gemma2 {} places moved: {}", check.moved, check.first));
  }
  places_clean_ = changes;
  return {};
}
std::vector<ExtentId> Gemma2Runner::weights() const { return weights_.extents(); }
std::vector<ExtentId> Gemma2Runner::state() const {
  std::vector<ExtentId> result;
  for (const auto& slot : slots_) {
    if (!slot) continue;
    const auto extents = slot->live.extents();
    result.insert(result.end(), extents.begin(), extents.end());
  }
  return result;
}
std::vector<ExtentId> Gemma2Runner::kept_state() const {
  std::vector<ExtentId> result;
  for (const auto& slot : slots_) {
    if (!slot) continue;
    const auto extents = slot->live.kept_extents();
    result.insert(result.end(), extents.begin(), extents.end());
  }
  return result;
}
std::vector<ExtentId> Gemma2Runner::managed_extents() const {
  auto result = weights();
  for (const auto& extents : {state(), kept_state()})
    result.insert(result.end(), extents.begin(), extents.end());
  return result;
}
std::expected<std::vector<LiveState::Range>, std::string> Gemma2Runner::CheckpointRanges(
    std::uint32_t positions) const {
  auto needed = md::Gemma2UsedState(profile_, layout_, positions);
  if (!needed) return Error(needed.error());
  std::vector<LiveState::Range> result;
  for (const auto& range : *needed) result.push_back({0, range.offset, range.bytes});
  return result;
}
Status Gemma2Runner::ReserveStateThrough(std::uint32_t index, std::uint32_t positions) {
  auto request = request_slot(index);
  if (!request) return Error(request.error());
  auto& slot = **request;
  if (slot.restoring) return Error("Gemma2 restore must complete before state growth");
  slot.state_refused = false;
  if (auto active = CheckActive(slot); !active) return active;
  auto ranges = CheckpointRanges(positions);
  if (!ranges) return Error(ranges.error());
  bool over_budget = false;
  auto used = slot.live.Use(node_, *ranges, &execution_, &over_budget);
  // A clean partial refusal may still have initialized new extents.
  if (!used || *used) {
    slot.on_disk = false;
    if (auto r = RefreshClosures(); !r) return r;
  }
  if (!used) {
    slot.state_refused = over_budget && !cohort_.faulted() && !slot.live.quarantined();
    return Error(used.error());
  }
  return {};
}
Status Gemma2Runner::Clear(std::uint32_t index) {
  auto request = request_slot(index);
  if (!request) return Error(request.error());
  auto& slot = **request;
  if (!bound_ || cohort_.faulted()) return Error("Gemma2 retirement required");
  if (auto r = cohort_.Check(node_, stream_, index); !r) return r;
  bool keep = false;
  if (!slot.spilled) {
    auto zeroed = slot.live.ZeroForReuse(node_, slot.fence, stream_);
    if (!zeroed) {
      auto states = States();
      cohort_.CheckFailedJob(node_, stream_, execution_, states);
      return Error(zeroed.error());
    }
    keep = *zeroed;
  }
  if (auto r = RefreshClosures(cohort_.active() & ~(SlotMask{1} << index)); !r) return r;
  const auto cleared = slot.live.DiscardGrowingState(node_, keep);
  if (cleared) {
    slot.spilled = false;
    slot.on_disk = false;
    slot.adopted.clear();
    slot.adopted_bytes = 0;
    slot.restoring.reset();
    slot.restore_needed.clear();
    slot.restored_bytes.clear();
    slot.positions = 0;
    slot.state_refused = false;
  }
  if (auto r = RefreshClosures(); !r) return r;
  return cleared;
}
Status Gemma2Runner::ClearIdle(std::uint32_t index) {
  auto request = request_slot(index);
  if (!request) return Error(request.error());
  if (!bound_ || cohort_.faulted() || (cohort_.IsActive(index) && node_.InRequest(stream_)))
    return Error("Gemma2 idle clear requires an idle healthy slot");
  auto& slot = **request;
  const auto cleared = slot.live.DiscardGrowingState(node_);
  if (cleared) {
    slot.spilled = false;
    slot.on_disk = false;
    slot.adopted.clear();
    slot.adopted_bytes = 0;
    slot.restoring.reset();
    slot.restore_needed.clear();
    slot.restored_bytes.clear();
    slot.positions = 0;
    slot.state_refused = false;
  }
  if (auto r = RefreshClosures(); !r) return r;
  return cleared;
}
Status Gemma2Runner::Spill(std::uint32_t index) {
  auto request = request_slot(index);
  if (!request) return Error(request.error());
  auto& slot = **request;
  if (!bound_ || cohort_.faulted()) return Error("Gemma2 retirement required");
  if (slot.restoring) return Error("Gemma2 restore must complete before spill");
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
Status Gemma2Runner::Restore(std::uint32_t index) {
  auto request = request_slot(index);
  if (!request) return Error(request.error());
  auto& slot = **request;
  slot.state_refused = false;
  if (!bound_ || cohort_.faulted()) return Error("Gemma2 retirement required");
  if (slot.restoring) return Error("Gemma2 restore must complete before disk restore");
  if (!slot.spilled) return {};
  if (!slot.adopted.empty()) {
    bool over_budget = false;
    auto used = slot.live.Use(node_, slot.adopted, &execution_, &over_budget);
    if (!used) {
      if (auto r = RefreshClosures(); !r) return r;
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
            if (!c) return Error("Gemma2 spilled state not cataloged");
            restore = std::move(*c);
            return {};
          },
          "Gemma2 describe restore");
      !r)
    return r;
  sc::AcquireReport report;
  bool over_budget = false;
  if (auto r = node_.Acquire(restore, report, "Gemma2 restore", &over_budget); !r) {
    slot.state_refused = over_budget && !cohort_.faulted();
    return r;
  }
  slot.spilled = false;
  slot.on_disk = true;
  return RefreshClosures();
}
Status Gemma2Runner::CopyState(std::uint32_t index, void* pinned,
                               std::span<const LiveState::Range> ranges,
                               LiveState::CopyRetirement* retirement) {
  return CopyState(index, pinned, ranges, true, retirement);
}
Status Gemma2Runner::CopyState(std::uint32_t index, void* pinned,
                               std::span<const LiveState::Range> ranges, bool to_host,
                               LiveState::CopyRetirement* retirement) {
  if (retirement) *retirement = LiveState::CopyRetirement::kProven;
  auto request = request_slot(index);
  if (!request) return Error(request.error());
  if (to_host == (*request)->restoring.has_value())
    return Error("Gemma2 checkpoint copy direction differs from restore state");
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
void Gemma2Runner::StateWrittenBack(bool whole) {
  for (auto& slot : slots_)
    if (slot && !slot->spilled) slot->on_disk = whole && slot->state_usable();
}
std::expected<void, std::string> Gemma2CheckpointFootprint(
    const model::Gemma2Profile& profile, const model::Gemma2StateLayout& layout,
    std::uint32_t positions, std::span<const LiveState::Range> ranges) {
  auto needed = md::Gemma2UsedState(profile, layout, positions);
  if (!needed) return Error(needed.error());
  std::uint64_t previous = 0;
  for (const auto& r : ranges) {
    if (r.region != 0 || r.offset >= layout.bytes || r.offset % kPagedExtent != 0 ||
        r.bytes != std::min(kPagedExtent, layout.bytes - r.offset) || r.offset < previous)
      return Error("Gemma2 restored footprint is not an ordered initialized extent set");
    previous = r.offset + r.bytes;
  }
  for (const auto& r : *needed) {
    const auto first = r.offset / kPagedExtent;
    const auto last = (r.offset + r.bytes - 1) / kPagedExtent;
    for (auto extent = first; extent <= last; ++extent)
      if (std::ranges::none_of(
              ranges, [extent](const auto& got) { return got.offset / kPagedExtent == extent; }))
        return Error("Gemma2 restored footprint does not fund its completed positions");
  }
  if (positions == 0 && !ranges.empty()) return Error("Gemma2 empty restore has backing");
  return {};
}
Status Gemma2Runner::ValidateFootprint(std::uint32_t positions,
                                       std::span<const LiveState::Range> ranges) const {
  return Gemma2CheckpointFootprint(profile_, layout_, positions, ranges);
}
Status Gemma2Runner::PrepareRestore(std::uint32_t index, std::uint32_t positions,
                                    std::span<const LiveState::Range> footprint,
                                    std::string_view source_layout) {
  if (source_layout.empty() || source_layout != checkpoint_layout_id_)
    return Error("Gemma2 checkpoint source layout differs");
  if (auto r = ValidateFootprint(positions, footprint); !r) return r;
  auto request = request_slot(index);
  if (!request) return Error(request.error());
  auto& slot = **request;
  if (auto r = CheckActive(slot); !r) return r;
  if (slot.restoring) return Error("Gemma2 restore is already pending");
  // Retain needs this destination unleased; every selected peer stays held.
  if (auto r = RefreshClosures(cohort_.active() & ~(SlotMask{1} << index)); !r) return r;
  slot.state_refused = false;
  bool over_budget = false;
  // Replacement begins here, before any destination footprint adjustment.
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
Status Gemma2Runner::CompleteRestore(std::uint32_t index, std::uint32_t positions) {
  auto request = request_slot(index);
  if (!request) return Error(request.error());
  auto& slot = **request;
  if (auto r = CheckActive(slot); !r) return r;
  if (!slot.restoring || *slot.restoring != positions)
    return Error("Gemma2 restore completion differs from its checked metadata");
  for (std::size_t i = 0; i < slot.restore_needed.size(); ++i)
    if (slot.restored_bytes[i] != slot.restore_needed[i].bytes)
      return Error("Gemma2 restore has not completed every logical checkpoint range");
  slot.positions = positions;
  slot.restoring.reset();
  slot.restore_needed.clear();
  slot.restored_bytes.clear();
  return {};
}
Status Gemma2Runner::Adopt(std::uint32_t index, std::uint32_t positions,
                           std::span<const LiveState::Range> footprint,
                           std::string_view source_layout) {
  if (source_layout.empty() || source_layout != checkpoint_layout_id_)
    return Error("Gemma2 checkpoint source layout differs");
  if (auto r = ValidateFootprint(positions, footprint); !r) return r;
  auto request = request_slot(index);
  if (!request) return Error(request.error());
  auto& slot = **request;
  if (!bound_ || cohort_.faulted() || positions == 0 || slot.positions != 0 || slot.spilled ||
      slot.live.used_bytes() != 0 || slot.restoring || Held(index))
    return Error("Gemma2 adopts kept state only into an empty idle healthy slot");
  auto bytes = slot.live.UsedBytesOf(footprint);
  if (!bytes) return Error(bytes.error());
  slot.adopted.assign(footprint.begin(), footprint.end());
  slot.adopted_bytes = *bytes;
  slot.spilled = true;
  slot.positions = positions;
  slot.on_disk = true;
  return RefreshClosures();
}
std::expected<Gemma2Runner::Plans::Entry*, std::string> Gemma2Runner::Planned(
    const kg::Gemma2ChunkShape& shape) {
  if (auto* found = plans_.Find(shape)) return found;
  const auto started = std::chrono::steady_clock::now();
  auto p = PlanGemma2Chunk(model_, shape, Choices(resources_.launch()), node_.activations().base,
                           node_.activations().bytes);
  if (!p) return Error(p.error());
  return CachePlanned(shape, std::move(*p), Seconds(std::chrono::steady_clock::now() - started));
}
std::expected<Gemma2Runner::Plans::Entry*, std::string> Gemma2Runner::CachePlanned(
    const kg::Gemma2ChunkShape& shape, std::unique_ptr<Gemma2Planned> p, double seconds,
    const std::function<void()>& transfer_charge) {
  const auto started = std::chrono::steady_clock::now();
  if (auto r = BindPlanned(*p, resources_.launch(), resources_.registry(), "Gemma2 chunk"); !r)
    return Error(r.error());
  std::vector<const ggml_tensor*> state_tensors;
  for (const auto& segment : p->graph.segments)
    for (const auto& [k, v] : segment.caches) {
      state_tensors.push_back(k);
      state_tensors.push_back(v);
    }
  Coverage checked;
  CheckCoverage(node_, owner_, p->graph.nodes, {.state = state_tensors, .inputs = p->graph.inputs},
                checked);
  if (checked.violations != 0)
    return Error(std::format("Gemma2 catalog coverage: {}", checked.first_violation));
  coverage_.tensors += checked.tensors;
  ++plan_selections_.plans;
  plan_selections_.steps += p->plan.steps.size();
  for (const auto& selected : p->plan.steps) {
    plan_selections_.device_masks += selected.implementation == kg::kGemma4MaskName;
    plan_selections_.q8_preparations += selected.implementation == kg::kQuantizeQ8Name;
    plan_selections_.prepared_mmvq_products += selected.implementation == kg::kMmvqPreparedName;
    plan_selections_.norm_mul += selected.implementation == kg::kRmsNormMulFused;
    plan_selections_.quant_geglu += selected.implementation == kg::kMulMatGeGluQFused;
    plan_selections_.norm_rope += selected.implementation == kg::kGemmaNormRopeName;
    plan_selections_.norm_add += selected.implementation == kg::kGemmaNormAddName;
    plan_selections_.owner_attention += selected.implementation == kg::kFlashAttnOwnersName;
    for (const auto* node : selected.nodes) {
      plan_selections_.bounded_owner_attention +=
          kg::JitllmOpOf(node) == kg::JitllmOp::kFlashAttnOwners && kg::JitllmOpInt(node, 4) == 1;
      plan_selections_.packed_prefill_attention +=
          selected.implementation == kg::kFlashAttnMmaGqa2Name &&
          std::string_view(ggml_get_name(node)).ends_with("packed_prefill_attention");
      const bool owner_prefill =
          kg::JitllmOpOf(node) == kg::JitllmOp::kFlashAttnOwners &&
          std::string_view(ggml_get_name(node)).ends_with("owner_prefill_attention");
      plan_selections_.owner_prefill_attention += owner_prefill;
      if (owner_prefill) {
        plan_selections_.largest_owner_prefill_rows =
            std::max(plan_selections_.largest_owner_prefill_rows,
                     static_cast<std::uint64_t>(node->src[0]->ne[1]));
        plan_selections_.largest_owner_prefill_kv_cells =
            std::max(plan_selections_.largest_owner_prefill_kv_cells,
                     static_cast<std::uint64_t>(node->src[1]->ne[0]));
      }
      plan_selections_.flexible_owner_prefill_attention +=
          owner_prefill && node->src[0]->ne[1] > 1 && node->src[0]->ne[1] < 128;
    }
  }
  const auto bytes = PlannedHostBytes(*p), nodes = PlannedNodes(*p);
  if (transfer_charge) transfer_charge();
  return &plans_.Add(shape, std::move(p), bytes, nodes,
                     seconds + Seconds(std::chrono::steady_clock::now() - started));
}
Status Gemma2Runner::Chunk(std::uint32_t past, std::span<const std::int32_t> tokens,
                           std::vector<float>& logits, bool all_outputs) {
  const Work work{0, past, tokens, &logits};
  return Wave(std::span(&work, 1), all_outputs);
}
Status Gemma2Runner::Wave(std::span<const Work> work, bool all_outputs) {
  return WaveWithMode(work, all_outputs, kg::Gemma2OutputMode::kHead);
}
Status Gemma2Runner::WavePrefill(std::span<const Work> work, bool want_head,
                                 std::span<const PrefillNext> next,
                                 std::optional<bool> next_want_head,
                                 std::optional<bool> after_want_head) {
  if (o_.packed_prefill && work.size() > 1) {
    if (work.size() != 2 || work[0].tokens.size() < 2 ||
        work[0].tokens.size() != work[1].tokens.size())
      return Error("Gemma2 packed prefill needs two equal multirow chunks");
    const auto read = [&](const Work& unit, std::uint32_t capacity) {
      if (unit.n_past > layout_.context || unit.tokens.size() > layout_.context - unit.n_past)
        return std::uint64_t{0};
      return std::min<std::uint64_t>(
          capacity, (std::uint64_t{unit.n_past} + unit.tokens.size() + 255U) / 256U * 256U);
    };
    for (const auto capacity : {layout_.global_cells, layout_.local_cells})
      if (read(work[0], capacity) == 0 || read(work[0], capacity) != read(work[1], capacity))
        return Error("Gemma2 packed prefill needs equal initialized read widths");
  }
  return WaveWithMode(work, false,
                      want_head ? kg::Gemma2OutputMode::kHead : kg::Gemma2OutputMode::kStateOnly,
                      next, next_want_head, after_want_head);
}
Status Gemma2Runner::WaveWithMode(std::span<const Work> work, bool all_outputs,
                                  kg::Gemma2OutputMode mode, std::span<const PrefillNext> next,
                                  std::optional<bool> next_want_head,
                                  std::optional<bool> after_want_head) {
  const PlanStep step;
  if (!bound_ || released_ || work.empty() || work.size() > o_.slots)
    return Error("Gemma2 wave is unavailable or unbounded");
  std::array<md::Gemma2Segment, kMaxRequestSlots> segments{};
  std::array<bool, kMaxRequestSlots> seen{};
  std::uint32_t rows = 0;
  const auto wave_rows = o_.max_wave_rows == 0 ? o_.max_rows : o_.max_wave_rows;
  const bool greedy = work.front().token != nullptr;
  if (greedy && (mode != kg::Gemma2OutputMode::kHead || all_outputs))
    return Error("Gemma2 greedy waves publish one frontier token per segment");
  for (std::size_t i = 0; i < work.size(); ++i) {
    const auto& w = work[i];
    if ((w.token != nullptr) != greedy || (w.logits == nullptr) == (w.token == nullptr))
      return Error("Gemma2 wave outputs must be all rows or all greedy tokens");
    if (w.slot >= o_.slots || seen[w.slot] || w.tokens.empty() || w.tokens.size() > o_.max_rows ||
        w.tokens.size() > wave_rows - rows || w.n_past != slots_[w.slot]->positions)
      return Error("Gemma2 wave needs distinct slots, bounded rows and exact continuations");
    if (slots_[w.slot]->restoring) return Error("Gemma2 restore must complete before execution");
    if (auto r = CheckActive(*slots_[w.slot]); !r) return r;
    for (std::size_t j = 0; j < i; ++j)
      if (greedy ? work[j].token == w.token : work[j].logits == w.logits)
        return Error("Gemma2 outputs must be independent");
    rows += static_cast<std::uint32_t>(w.tokens.size());
    seen[w.slot] = true;
    segments[i] = {w.slot, w.n_past, w.tokens};
  }
  const auto head_rows = o_.max_head_rows == 0 ? o_.max_rows : o_.max_head_rows;
  const auto outputs = mode == kg::Gemma2OutputMode::kStateOnly ? 0U
                       : all_outputs ? rows
                                     : static_cast<std::uint32_t>(work.size());
  if (outputs > head_rows) return Error("Gemma2 wave exceeds head publication capacity");
  const auto selected = std::span(segments).first(work.size());
  auto bytes =
      md::Gemma2HostInputBytes(profile_, layout_, selected, !o_.device_masks, 256, wave_rows);
  if (!bytes) return Error(bytes.error());
  if (*bytes > host_input_bytes_) return Error("Gemma2 host descriptor envelope exceeded");
  if (auto r = CheckPlaces(); !r) return r;
  // The caller funds host_input_bytes()+plan_floor_bytes() in the node's
  // startup host floor, as for the shared runners. One driver stages a wave.
  auto input = md::Gemma2Chunk(profile_, layout_, selected, !o_.device_masks, 256, wave_rows);
  if (!input) return Error(input.error());
  kg::Gemma2ChunkShape shape;
  shape.output_mode = mode;
  shape.outputs = outputs;
  shape.greedy = greedy;
  std::vector<std::int32_t> frontier;
  frontier.reserve(outputs);
  for (const auto& s : input->segments) {
    shape.segments.push_back({s.slot, s.rows, s.n_past, s.global_n_kv, s.local_n_kv});
    if (mode == kg::Gemma2OutputMode::kHead) {
      if (all_outputs)
        for (std::uint32_t i = 0; i < s.rows; ++i)
          frontier.push_back(static_cast<std::int32_t>(s.first_row + i));
      else
        frontier.push_back(static_cast<std::int32_t>(s.first_row + s.rows - 1));
    }
    if (auto r = ReserveStateThrough(s.slot, s.n_past + s.rows); !r) return r;
  }
  auto entry_of = Planned(shape);
  if (!entry_of) return Error(entry_of.error());
  auto& entry = **entry_of;
  auto& p = *entry.planned;
  const auto output_bytes = std::uint64_t{outputs} * profile_.vocab * sizeof(float);
  if ((mode == kg::Gemma2OutputMode::kHead &&
       (p.graph.logits == nullptr || p.graph.logits->type != GGML_TYPE_F32 ||
        p.graph.logits->ne[0] != profile_.vocab || p.graph.logits->ne[1] != outputs ||
        p.graph.logits->ne[2] != 1 || p.graph.logits->ne[3] != 1 ||
        !ggml_is_contiguous(p.graph.logits) || ggml_nbytes(p.graph.logits) != output_bytes)) ||
      (mode == kg::Gemma2OutputMode::kStateOnly && p.graph.logits != nullptr) ||
      (greedy && (p.graph.greedy == nullptr || p.graph.greedy->type != GGML_TYPE_I32 ||
                  p.graph.greedy->ne[0] != outputs || p.graph.greedy->ne[1] != 1 ||
                  p.graph.greedy->ne[2] != 1 || p.graph.greedy->ne[3] != 1 ||
                  !ggml_is_contiguous(p.graph.greedy) ||
                  ggml_nbytes(p.graph.greedy) != std::uint64_t{outputs} * sizeof(std::int32_t))))
    return Error("Gemma2 planned head publication exceeds its envelope");
  auto source_bytes = Gemma2SourceBytes(p.graph);
  if (!source_bytes || *source_bytes > host_input_bytes_ - *bytes)
    return Error("Gemma2 host input envelope exceeded");
  auto host = Gemma2Sources(p.graph, *input, frontier, {}, host_input_bytes_ - *bytes);
  if (!host) return Error(host.error());
  auto copies = runs_.Stage(host->sources, 0);
  if (!copies) return Error(copies.error());
  // Upcoming shapes, when the caller hints them: the same wave's slots
  // continuing from this wave's end with the hint's rows (ahead[0]), then
  // with the rows after those (ahead[1]). A hint is never work: a wrong one
  // costs only an unused plan or graph.
  std::array<kg::Gemma2ChunkShape, 2> ahead;
  std::array<bool, 2> known{};
  if (!next.empty() && next.size() <= work.size() && !greedy && !all_outputs &&
      (o_.prefill_lookahead || o_.capture_ahead)) {
    for (std::size_t k = 0; k < ahead.size(); ++k) {
      const auto head = k == 0 ? next_want_head : after_want_head;
      if (!head.has_value()) continue;
      bool valid = true;
      std::uint32_t predicted_rows = 0;
      for (std::size_t i = 0; valid && i < next.size(); ++i) {
        const auto& hint = next[i];
        const auto rows_k = k == 0 ? hint.rows : hint.after;
        // A completed owner may disappear from either future cohort.
        if (rows_k == 0) continue;
        const auto from = std::ranges::find(work, hint.slot, &Work::slot);
        const auto current_end =
            from == work.end() ? 0U
                               : from->n_past + static_cast<std::uint32_t>(from->tokens.size());
        // The next stage may be suppressed (mixed head modes), so bound its
        // row descriptor independently before using it for the after position.
        valid = from != work.end() && hint.rows != 0 && hint.rows <= o_.max_rows &&
                current_end <= layout_.context && hint.rows <= layout_.context - current_end;
        if (!valid) break;
        const auto past = current_end + (k == 0 ? 0U : hint.rows);
        valid =
            std::ranges::none_of(ahead[k].segments,
                                 [&](const auto& segment) { return segment.slot == hint.slot; }) &&
            rows_k <= o_.max_rows && rows_k <= wave_rows - predicted_rows &&
            past <= layout_.context && rows_k <= layout_.context - past;
        if (!valid) break;
        predicted_rows += rows_k;
        const auto cells = Round(std::uint64_t{past} + rows_k, 256);
        ahead[k].segments.push_back(
            {hint.slot, rows_k, past,
             static_cast<std::uint32_t>(std::min<std::uint64_t>(cells, layout_.global_cells)),
             static_cast<std::uint32_t>(std::min<std::uint64_t>(cells, layout_.local_cells))});
      }
      if (!valid || ahead[k].segments.empty()) continue;
      // Packed two-owner plans must satisfy the same geometry as an actual
      // WavePrefill. A later stage may have a different owner count/width.
      if (o_.packed_prefill && ahead[k].segments.size() > 1) {
        if (ahead[k].segments.size() != 2) continue;
        const auto& a = ahead[k].segments[0];
        const auto& b = ahead[k].segments[1];
        if (a.rows < 2 || a.rows != b.rows || a.global_n_kv != b.global_n_kv ||
            a.local_n_kv != b.local_n_kv)
          continue;
      }
      ahead[k].output_mode = *head ? kg::Gemma2OutputMode::kHead : kg::Gemma2OutputMode::kStateOnly;
      ahead[k].outputs = *head ? static_cast<std::uint32_t>(ahead[k].segments.size()) : 0U;
      known[k] = true;
    }
  }
  // Protect all cached predictions before any optional capture/plan charge.
  // Full shape equality includes cohort, output mode and padded cache widths;
  // positions themselves remain runtime inputs. Deduplicate before funding.
  std::array<Plans::Entry*, 2> ahead_cached{};
  std::array<bool, 2> distinct{};
  for (std::size_t k = 0; k < ahead.size(); ++k) {
    if (!known[k]) continue;
    if (ahead[k] == shape) {
      ahead_cached[k] = &entry;
      continue;
    }
    bool duplicate = false;
    for (std::size_t prior = 0; prior < k; ++prior)
      if (known[prior] && ahead[k] == ahead[prior]) {
        ahead_cached[k] = ahead_cached[prior];
        duplicate = true;
        break;
      }
    if (duplicate) continue;
    distinct[k] = true;
    ahead_cached[k] = plans_.Find(ahead[k]);
  }
  auto& runs = entry.runs[0];
  // A graph captured ahead holds the layout its plan's inputs predicted;
  // one that differs from what was staged is dropped (Settle returns its
  // charge) and this run goes launch by launch instead. The check covers
  // every graph (any other differing layout, which Queue would refuse,
  // counts here too).
  if (runs_.graphs() && runs.graph.has_value() && *copies != runs.copies) {
    runs.DropGraph();
    ++lookahead_.dropped_ahead;
  }
  // Prefill grows its read width every 256 positions, so with 128-row
  // chunks every shape runs exactly twice: D-090's second-run capture would
  // never replay. A shape the next chunk repeats is captured on this, its
  // first, run instead, unless a graph was captured ahead for it.
  const bool repeats = known[0] && ahead[0] == shape;
  bool capture =
      runs.CaptureDue(runs_.graphs()) || (o_.capture_ahead && repeats && runs_.graphs() &&
                                          !runs.graph.has_value() && !runs.uncapturable);
  if (capture && !plans_.ChargeGraph(entry)) capture = false;
  const bool capture_first = capture && runs.eager_runs == 0;
  // The next chunk's planned shape, when it differs from this one: its
  // graph captured beside this run (which replays, leaving the host idle),
  // so its first run replays too. Its inputs' layout is its plan's.
  Plans::Entry* upcoming = nullptr;
  Copies upcoming_copies;
  std::array<RunCopy, 1> upcoming_output{};
  if (o_.capture_ahead && known[0] && !repeats && runs_.graphs()) {
    upcoming = ahead_cached[0];
    const Gemma2Planned* u = upcoming == nullptr ? nullptr : upcoming->planned.get();
    auto layout =
        u == nullptr ? std::expected<Copies, std::string>{} : runs_.Layout(u->graph.inputs, 0);
    if (u == nullptr || upcoming->runs[0].graph.has_value() || upcoming->runs[0].uncapturable ||
        upcoming->runs[0].ahead_refused || !layout || !plans_.ChargeGraph(*upcoming)) {
      upcoming = nullptr;
    } else {
      upcoming_copies = std::move(*layout);
      if (ahead[0].output_mode == kg::Gemma2OutputMode::kHead)
        upcoming_output[0] = {Address(logits_), Address(u->graph.logits->data),
                              std::uint64_t{ahead[0].outputs} * profile_.vocab * sizeof(float)};
    }
  }
  // Independent optional grants for distinct missing near/far shapes. All
  // cached predictions above are protected throughout these callbacks.
  std::array<const kg::Gemma2ChunkShape*, 2> build{};
  std::size_t build_count = 0;
  if (o_.prefill_lookahead && node_.threaded())
    for (std::size_t k = 0; k < ahead.size() && build_count < o_.prefill_lookahead_capacity; ++k)
      if (distinct[k] && ahead_cached[k] == nullptr) build[build_count++] = &ahead[k];
  PrefillLookaheadGroup<Gemma2Planned, 2> future(node_, plan_floor_bytes_);
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
  const auto choices = funded ? Choices(resources_.launch()) : kg::DeviceChoices{};
  const std::function<void()> meanwhile = funded ? std::function<void()>([&] {
    const auto built = future.BuildAll([&](std::size_t i) {
      return PlanGemma2Chunk(model_, *build[i], choices, node_.activations().base,
                             node_.activations().bytes);
    });
    for (std::size_t i = 0; i < built.size(); ++i) {
      lookahead_.built += built[i];
      lookahead_.build_seconds += future.seconds(i);
    }
    lookahead_.built_pairs += built[0] && built[1];
  })
                                                 : std::function<void()>{};
  std::array<RunCopy, 1> output{};
  if (greedy)
    output[0] = {Address(logits_), Address(p.graph.greedy->data),
                 std::uint64_t{outputs} * sizeof(std::int32_t)};
  else if (mode == kg::Gemma2OutputMode::kHead)
    output[0] = {Address(logits_), Address(p.graph.logits->data), output_bytes};
  const auto output_copies = std::span(output).first(mode == kg::Gemma2OutputMode::kHead ? 1U : 0U);
  bool wrote = false, unknown = false;
  Status queued;
  RunPath path = RunPath::kEager;
  const auto posted = node_.Job(
      execution_,
      [&](providers::NativeStream native) {
        const auto result =
            runs_.Queue(runs, *copies, {}, *p.bound, output_copies, capture, graph_stats_, native);
        wrote = result.before || result.result.has_value();
        path = result.path;
        if (!result.result) {
          queued = Error(result.result.error().detail);
          unknown = result.result.error().error == kg::KernelError::kUnknown;
          return unknown         ? sc::JobResult::kUnknown
                 : result.before ? sc::JobResult::kFailed
                                 : sc::JobResult::kNotStarted;
        }
        if (upcoming != nullptr) {
          const auto ahead_outputs =
              std::span(upcoming_output)
                  .first(ahead[0].output_mode == kg::Gemma2OutputMode::kHead ? 1U : 0U);
          auto captured =
              runs_.CaptureAhead(upcoming->runs[0], upcoming_copies, *upcoming->planned->bound,
                                 ahead_outputs, graph_stats_, native);
          if (!captured) {
            // This run's work is queued; the capture's fault is the context's.
            queued = Error(captured.error().detail);
            unknown = true;
            return sc::JobResult::kUnknown;
          }
          lookahead_.captured_ahead += *captured;
        }
        return sc::JobResult::kQueued;
      },
      "Gemma2 chunk/wave", stream_, meanwhile);
  if (!posted || !queued || resources_.launch().faulted()) {
    auto states = States();
    cohort_.CheckFailedJob(node_, stream_, execution_, states);
    for (const auto& w : work)
      if (wrote || unknown) slots_[w.slot]->live.Quarantine();
    if (resources_.launch().faulted()) cohort_.Fault(states);
    if (!queued) return queued;
    return !posted ? posted : Error("Gemma2 launch context faulted");
  }
  const auto installed = future.InstallAfterCompletion(
      [&](std::size_t i, auto planned, double seconds, const auto& transfer) {
        return CachePlanned(*build[i], std::move(planned), seconds, transfer);
      });
  for (const bool cached : installed) lookahead_.cached += cached;
  lookahead_.cached_pairs += installed[0] && installed[1];
  lookahead_.captured_first += capture_first && path == RunPath::kCaptured;
  Count(graph_stats_, path);
  if (greedy)
    for (std::size_t i = 0; i < work.size(); ++i) {
      const auto token = static_cast<const std::int32_t*>(logits_)[i];
      if (token < 0 || std::cmp_greater_equal(token, profile_.vocab)) {
        for (const auto& w : work) slots_[w.slot]->live.Quarantine();
        return Error("Gemma2 device choice is outside its vocabulary; state quarantined");
      }
    }
  std::size_t at = 0;
  for (std::size_t i = 0; i < work.size(); ++i) {
    const auto& w = work[i];
    if (greedy) {
      *w.token = static_cast<const std::int32_t*>(logits_)[i];
      ++greedy_tokens_;
    } else {
      const auto count = mode == kg::Gemma2OutputMode::kStateOnly ? 0U
                         : all_outputs                            ? w.tokens.size()
                                                                  : 1U;
      const auto n = count * profile_.vocab;
      const auto* values = static_cast<const float*>(logits_) + at;
      if (mode == kg::Gemma2OutputMode::kStateOnly)
        w.logits->clear();
      else
        w.logits->assign(values, values + n);
      at += n;
    }
    slots_[w.slot]->positions += static_cast<std::uint32_t>(w.tokens.size());
    slots_[w.slot]->on_disk = false;
  }
  return {};
}
void Gemma2Runner::DropPlans() { plans_.Clear(); }
void Gemma2Runner::ReclaimCandidates(std::uint32_t owner, bool running,
                                     std::vector<memory::ReclaimCandidate>& out) {
  if (released_ || cohort_.faulted()) return;
  std::array<PlanCacheBase*, 1> caches{&plans_};
  CollectPlans(caches, owner, running, out);
}
std::uint64_t Gemma2Runner::Reclaim(memory::ReclaimKind kind, std::uint64_t id) {
  if (released_ || cohort_.faulted()) return 0;
  std::array<PlanCacheBase*, 1> caches{&plans_};
  return ReclaimPlan(caches, kind, id);
}
Status Gemma2Runner::Release() {
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
