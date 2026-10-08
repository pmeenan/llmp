// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "engine/dsv4_runner.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <expected>
#include <format>
#include <initializer_list>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

#include "engine/checkpoint_file.h"
#include "engine/support.h"
#include "ggml.h"
#include "kernels/ggml/dsv4_outa.h"
#include "kernels/ggml/executor.h"
#include "kernels/ggml/graph_plan.h"
#include "kernels/ggml/llmp_ops.h"
#include "kernels/ggml/ops.h"
#include "kernels/ggml/ops_ext.h"
#include "kernels/paging/paging.h"
#include "providers/device_runtime.h"
#include "scheduler/commands.h"
#include "scheduler/scheduler.h"

namespace llmp::engine {

namespace {

namespace kg = llmp::kernels::ggml;
namespace md = llmp::model;
namespace sc = llmp::scheduler;
using catalog::ExtentId;
using catalog::MemoryClass;
using support::Address;
using support::Error;
using support::Pointer;

std::optional<std::uint32_t> HcaFirstPosition(const Dsv4Model& model,
                                              const kg::Dsv4ChunkShape& shape,
                                              std::uint32_t first) {
  if (Dsv4PrefillHca(model, shape)) {
    return first;
  }
  return std::nullopt;
}

// Whether every layer's output-A weights are the Q8_0 [4,096, 8,192] the
// output-A prefill takes (kernels/ggml/dsv4_outa.h): only then can its
// HCA run, so only then is a plan an HCA plan.
bool OutAWeights(const md::Dsv4Binding& b) {
  return std::ranges::all_of(b.layers, [](const md::Dsv4Layer& l) {
    return l.out_a.type == "Q8_0" && l.out_a.ne == std::vector<std::uint64_t>{4096, 8192};
  });
}
using support::Round;
using support::Seconds;

// A draft block's ids, bounded to the vocabulary in place before anything
// reads them (the verify's tokens and its rows' lookup): the drafter's
// argmax over non-finite logits gives -1. Queued on the job's stream.
bool ClampDrafts(void* drafts, std::uint32_t count, std::uint32_t vocab,
                 providers::NativeStream native) {
  return kernels::paging::ClampTokens(static_cast<std::int32_t*>(drafts), count,
                                      static_cast<std::int32_t>(vocab), native.handle);
}

// The device's token-embedding lookup (the drafts' rows, and
// CheckDeviceEmbedding's): GGML's float gather for an F32, F16 or BF16
// table (the community GGUF's is F16), its quantized gather otherwise. Each
// widens exactly as the host's lookup does (Dsv4EmbeddingRows).
std::expected<void, kg::KernelFailure> LookUpRows(kg::LaunchContext& launch, ggml_tensor* rows) {
  const ggml_type type = rows->src[0]->type;
  if (type == GGML_TYPE_F32 || type == GGML_TYPE_F16 || type == GGML_TYPE_BF16) {
    return kg::GetRows(launch, rows);
  }
  return kg::GetRowsExt(launch, rows);
}

constexpr std::uint64_t kExtent = kPagedExtent;
// The most snapshot ranges a verify saves: at most 8 rows of about 230
// ranges each (a cell a layer, a CSA layer's six ring and compressed rows,
// an HCA layer's three, the drafter's three cells), and the scratch rows.
constexpr std::uint32_t kRangeCapacity = 4096;

}  // namespace

void SetDsv4ServedPrefill(Dsv4Options& options) {
  options.prefill_outa_hca = true;
  options.prefill_outa_hca_partial = true;
}

// ------------------------------------------------------------------ slots

std::array<Dsv4Runner::RequestState*, Dsv4Runner::kRequestSlots> Dsv4Runner::Requests() {
  std::array<RequestState*, kRequestSlots> all{};
  all[0] = &default_request_;
  for (std::size_t i = 1; i < kRequestSlots; ++i) {
    all[i] = &additional_requests_[i - 1];
  }
  return all;
}

std::array<const Dsv4Runner::RequestState*, Dsv4Runner::kRequestSlots> Dsv4Runner::Requests()
    const {
  std::array<const RequestState*, kRequestSlots> all{};
  all[0] = &default_request_;
  for (std::size_t i = 1; i < kRequestSlots; ++i) {
    all[i] = &additional_requests_[i - 1];
  }
  return all;
}

std::array<LiveState*, Dsv4Runner::kRequestSlots> Dsv4Runner::States() {
  std::array<LiveState*, kRequestSlots> states{};
  for (RequestState* request : Requests()) {
    states[request->slot] = request->provisioned ? &request->live : nullptr;
  }
  return states;
}

std::expected<Dsv4Runner::Slot*, std::string> Dsv4Runner::request_slot(std::size_t index) {
  if (released_ || index >= kRequestSlots || !Requests()[index]->provisioned) {
    return Error("a DeepSeek request slot outside the live runner");
  }
  return &request_slots_[index];
}

Status Dsv4Runner::SelectSlots(std::span<Slot* const> active) {
  if (released_ || cohort_.faulted()) {
    return Error("the DeepSeek cohort requires retirement");
  }
  std::array<std::uint32_t, kRequestSlots> indices{};
  if (active.size() > wave_slots_) {
    return Error(std::format("at most {} DeepSeek request slots", wave_slots_));
  }
  for (std::size_t i = 0; i < active.size(); ++i) {
    const Slot* slot = active[i];
    if (slot == nullptr || &slot->owner_ != this || !slot->request_.provisioned) {
      return Error("a DeepSeek slot belongs to another runner or is not provisioned");
    }
    indices[i] = slot->index();
  }
  auto mask = cohort_.MaskOf(std::span(indices).first(active.size()));
  if (!mask) {
    return std::unexpected(mask.error());
  }
  // An unchanged selection within an open request keeps its closure; state
  // growth, clears and restores still refresh it.
  if (*mask == cohort_.active() && node_.InRequest(stream_)) {
    return {};
  }
  cohort_.Select(*mask);
  return RefreshClosures();
}

bool Dsv4Runner::HasRetainedState() const {
  return std::ranges::any_of(Requests(), [](const RequestState* request) {
    return request->provisioned && request->live.used_bytes() != 0;
  });
}

void Dsv4Runner::DropPlans() {
  waves_.Clear();
  dwaves_.Clear();
  for (RequestState* request : Requests()) {
    request->plans.Clear();
    request->dplans.Clear();
  }
  last_planned_ = nullptr;
}

std::size_t Dsv4Runner::plans() const {
  std::size_t count = waves_.size();
  for (const RequestState* request : Requests()) {
    count += request->plans.size();
  }
  return count;
}

std::size_t Dsv4Runner::graphs() const {
  std::size_t count = waves_.graphs() + dwaves_.graphs();
  for (const RequestState* request : Requests()) {
    count += request->plans.graphs() + request->dplans.graphs();
  }
  return count;
}

std::uint64_t Dsv4Runner::cached_graph_bytes() const {
  std::uint64_t bytes = waves_.graph_bytes() + dwaves_.graph_bytes();
  for (const RequestState* request : Requests()) {
    bytes += request->plans.graph_bytes() + request->dplans.graph_bytes();
  }
  return bytes;
}

std::uint64_t Dsv4Runner::graph_measured_bytes() const {
  std::uint64_t bytes = waves_.graph_measured_bytes() + dwaves_.graph_measured_bytes();
  for (const RequestState* request : Requests()) {
    bytes += request->plans.graph_measured_bytes() + request->dplans.graph_measured_bytes();
  }
  return bytes;
}

std::uint64_t Dsv4Runner::cached_plan_bytes() const {
  std::uint64_t bytes = waves_.host_bytes() + dwaves_.host_bytes();
  for (const RequestState* request : Requests()) {
    bytes += request->plans.host_bytes() + request->dplans.host_bytes();
  }
  return bytes;
}

std::uint64_t Dsv4Runner::cached_arena_used(bool capacity) const {
  std::uint64_t bytes = 0;
  const auto add = [&bytes, capacity](const auto& cache) {
    cache.ForEach([&bytes, capacity](const auto& entry) {
      const auto& arena = entry.planned->arena;
      if (arena) {
        bytes += capacity ? arena->bytes() : arena->used();
      }
    });
  };
  add(waves_);
  add(dwaves_);
  for (const RequestState* request : Requests()) {
    add(request->plans);
    add(request->dplans);
  }
  return bytes;
}

std::array<PlanCacheBase*, (2 * Dsv4Runner::kRequestSlots) + 2> Dsv4Runner::PlanCaches() {
  std::array<PlanCacheBase*, (2 * kRequestSlots) + 2> caches{};
  for (RequestState* request : Requests()) {
    caches[std::size_t{2} * request->slot] = &request->plans;
    caches[(std::size_t{2} * request->slot) + 1] = &request->dplans;
  }
  caches[2 * kRequestSlots] = &waves_;
  caches.back() = &dwaves_;
  return caches;
}

std::array<const PlanCacheBase*, (2 * Dsv4Runner::kRequestSlots) + 2> Dsv4Runner::PlanCaches()
    const {
  std::array<const PlanCacheBase*, (2 * kRequestSlots) + 2> caches{};
  for (const RequestState* request : Requests()) {
    caches[std::size_t{2} * request->slot] = &request->plans;
    caches[(std::size_t{2} * request->slot) + 1] = &request->dplans;
  }
  caches[2 * kRequestSlots] = &waves_;
  caches.back() = &dwaves_;
  return caches;
}

void Dsv4Runner::ForgetLastPlanned() {
  if (last_planned_ == nullptr) {
    return;
  }
  bool kept = false;
  for (const RequestState* request : Requests()) {
    request->plans.ForEach(
        [&](const ChunkPlans::Entry& e) { kept = kept || e.planned.get() == last_planned_; });
  }
  if (!kept) {
    last_planned_ = nullptr;  // DumpLast's plan went
  }
}

void Dsv4Runner::ReclaimCandidates(std::uint32_t owner, bool running,
                                   std::vector<memory::ReclaimCandidate>& out) {
  if (released_ || cohort_.faulted()) {
    return;  // a faulted cohort keeps its owners until retirement
  }
  auto caches = PlanCaches();
  CollectPlans(caches, owner, running, out);
}

std::uint64_t Dsv4Runner::Reclaim(memory::ReclaimKind kind, std::uint64_t id) {
  if (released_ || cohort_.faulted()) {
    return 0;
  }
  auto caches = PlanCaches();
  const std::uint64_t freed = ReclaimPlan(caches, kind, id);
  ForgetLastPlanned();
  return freed;
}

std::uint64_t Dsv4Runner::reclaimed_plans() const {
  std::uint64_t n = 0;
  for (const PlanCacheBase* cache : PlanCaches()) {
    n += cache->reclaimed_plans();
  }
  return n;
}

std::uint64_t Dsv4Runner::reclaimed_graphs() const {
  std::uint64_t n = 0;
  for (const PlanCacheBase* cache : PlanCaches()) {
    n += cache->reclaimed_graphs();
  }
  return n;
}

// ------------------------------------------------------------------ setup

std::vector<ExtentId> Dsv4Runner::weights() const {
  std::vector<ExtentId> all = weights_.extents();
  all.insert(all.end(), dweights_.extents().begin(), dweights_.extents().end());
  return all;
}

std::vector<ExtentId> Dsv4Runner::state() const {
  std::vector<ExtentId> all;
  for (const RequestState* request : Requests()) {
    if (request->provisioned) {
      const auto extents = request->live.extents();
      all.insert(all.end(), extents.begin(), extents.end());
    }
  }
  return all;
}

std::vector<ExtentId> Dsv4Runner::unchanged_state() const {
  std::vector<ExtentId> all;
  for (const RequestState* request : Requests()) {
    if (request->provisioned && !request->spilled) {
      std::vector<ExtentId> written;
      SplitForSpill(*request, written, all);
    }
  }
  return all;
}

void Dsv4Runner::StateWrittenBack(bool whole) {
  for (RequestState* request : Requests()) {
    if (request->provisioned && !request->spilled) {
      if (whole) {
        request->track.Saved();
      } else {
        request->track.Lost();
      }
    }
  }
}

std::vector<ExtentId> Dsv4Runner::kept_state() const {
  std::vector<ExtentId> all;
  for (const RequestState* request : Requests()) {
    const std::vector<ExtentId> kept = request->live.kept_extents();
    all.insert(all.end(), kept.begin(), kept.end());
  }
  return all;
}

std::vector<ExtentId> Dsv4Runner::managed_extents() const {
  std::vector<ExtentId> all = weights();
  const std::vector<ExtentId> live = state();
  all.insert(all.end(), live.begin(), live.end());
  const std::vector<ExtentId> kept = kept_state();
  all.insert(all.end(), kept.begin(), kept.end());
  return all;
}

std::span<const std::byte> Dsv4Runner::table() const {
  const std::uint64_t base = weights_.group_address(table_group_);
  return {static_cast<const std::byte*>(Pointer(base)), weights_.host_bytes()};
}

Status Dsv4Runner::Setup() {
  const auto setup_started = std::chrono::steady_clock::now();
  if (o_.context > md::kDsv4FlashContext) {
    return Error(std::format("context {} exceeds DeepSeek V4 Flash's trained ceiling {}",
                             o_.context, md::kDsv4FlashContext));
  }
  if (o_.wave_slots == 0 || o_.wave_slots > kRequestSlots || (o_.wave_slots > 1 && o_.exact)) {
    return Error(std::format("DeepSeek request slots: one to {}, several only in the fast plan",
                             kRequestSlots));
  }
  // A DSpark verify joins a wave with at least two rows a slot.
  if (!o_.drafter.empty() && o_.wave_slots > static_cast<std::uint32_t>(kWaveRows / 2)) {
    return Error(
        std::format("DeepSeek with a drafter takes at most {} request slots (a verify of "
                    "at least two of a wave's {} rows a slot)",
                    kWaveRows / 2, kWaveRows));
  }
  wave_slots_ = o_.wave_slots;
  serial_reason_.clear();
  if (auto r = weights_.Open(o_.artifact); !r) {
    return r;
  }
  auto binding = md::BindDsv4(profile_, weights_.artifact());
  if (!binding) {
    return std::unexpected(binding.error());
  }
  binding_ = std::move(*binding);
  // The fast plan keeps its window as a ring; the reference mode, llama.cpp's
  // full-size window cache (model/dsv4.h Dsv4Window).
  auto layout =
      md::Dsv4State(profile_, o_.context, o_.max_rows,
                    o_.exact || o_.full_window ? md::Dsv4Window::kFull : md::Dsv4Window::kRing);
  if (!layout) {
    return std::unexpected(layout.error());
  }
  layout_ = std::move(*layout);
  // The largest shapes are planned at the context's end and a decode step
  // after a whole chunk (below): checked here, so those positions cannot
  // wrap.
  if (o_.max_rows >= o_.context || (!o_.drafter.empty() && o_.max_verify > o_.context)) {
    return Error(std::format("a context of {} leaves no room for chunks of {} rows", o_.context,
                             o_.max_rows));
  }
  if (!o_.drafter.empty()) {
    if (o_.draft_rows == 0 || o_.draft_rows > dprofile_.block_size || o_.max_verify == 0 ||
        o_.max_verify > o_.draft_rows + 1 || std::cmp_greater(o_.max_verify, kg::kRowsMaxColumns)) {
      return Error(
          std::format("a draft block of 1 to {} rows, a verify of 1 to the drafts + 1 "
                      "(at most {}) rows",
                      dprofile_.block_size, kg::kRowsMaxColumns));
    }
    if (auto r = dweights_.Open(o_.drafter); !r) {
      return r;
    }
    auto dbinding = md::BindDspark(dprofile_, dweights_.artifact(), profile_, binding_);
    if (!dbinding) {
      return Error(std::format("the drafter: {}", dbinding.error()));
    }
    dbinding_ = std::move(*dbinding);
    auto dlayout = md::DsparkState(dprofile_, o_.draft_rows);
    if (!dlayout) {
      return std::unexpected(dlayout.error());
    }
    dlayout_ = std::move(*dlayout);
  }
  // Waves need every layer in the fast plan's fused form. An artifact whose
  // weights cannot take it is served, one request at a time, rather than
  // refused: one request slot, and the reason for the start's log.
  if (wave_slots_ > 1) {
    if (auto support = kg::Dsv4WaveSupport(profile_, binding_); !support) {
      serial_reason_ =
          std::format("no waves of {} requests: {}", wave_slots_, support.error().detail);
      wave_slots_ = 1;
    }
  }
  cohort_.set_slots(wave_slots_);
  // The state first: its extents come before the weights' in a closure, so
  // a swap back restores it before paging the weights in. Every request
  // slot's: the default one's, then the others' in order.
  for (RequestState* request : Requests()) {
    if (request->slot >= wave_slots_) {
      continue;
    }
    const std::string slot = request->slot == 0 ? "" : std::format(" (slot {})", request->slot);
    if (auto r =
            request->live.AddGrowing(node_, "the DeepSeek state" + slot, layout_.bytes, owner_);
        !r) {
      return r;
    }
    if (speculative()) {
      if (auto r =
              request->live.AddGrowing(node_, "the DSpark ring" + slot, dlayout_.bytes, owner_);
          !r) {
        return r;
      }
    }
    request->provisioned = true;
  }
  std::vector<std::uint64_t> stride;
  std::vector<std::uint64_t> dstride;
  if (auto r = ReservePart(weights_, binding_, profile_.layers, profile_.experts, true, stride);
      !r) {
    return r;
  }
  if (speculative()) {
    if (auto r = ReservePart(dweights_, dbinding_.blocks, dprofile_.blocks.layers,
                             dprofile_.blocks.experts, false, dstride);
        !r) {
      return Error(std::format("the drafter: {}", r.error()));
    }
  }
  if (auto r = resources_.OpenCublas("the DeepSeek cuBLAS workspace"); !r) {
    return r;
  }

  // The largest shapes, as the resident harness sizes them: a full chunk
  // at the start and at the end of the context, and a decode step at the
  // end, planned over placeless addresses; beside a drafter, the same
  // chunks with its injection, verifies and its draft block too; with
  // several request slots, the widest waves.
  const auto placeless = [](std::uint32_t) { return std::uint64_t{1} << 44U; };
  model_ = Dsv4Model{.artifact = &weights_.artifact(),
                     .profile = &profile_,
                     .binding = &binding_,
                     .state = &layout_,
                     .places = {.resource = placeless,
                                .array = placeless,
                                .stride = std::move(stride),
                                .state = std::uint64_t{1} << 45U},
                     .rot = kg::HadamardMatrix(profile_.indexer_head_dim),
                     .exact = o_.exact,
                     .prefill_outa_hca = o_.prefill_outa_hca && !o_.exact && !o_.full_window,
                     .prefill_outa_hca_partial = o_.prefill_outa_hca_partial,
                     .wave_lanes = o_.wave_lanes,
                     .device_raw_masks = o_.device_raw_masks};
  if (speculative()) {
    dmodel_ = DsparkModel{.artifact = &dweights_.artifact(),
                          .profile = &dprofile_,
                          .binding = &dbinding_,
                          .state = &dlayout_,
                          .places = {.resource = placeless,
                                     .array = placeless,
                                     .stride = std::move(dstride),
                                     .state = std::uint64_t{1} << 45U},
                          .target_resource = placeless,
                          .exact = o_.exact,
                          .device_masks = o_.device_draft_masks};
  }
  std::uint64_t most_activations = 0;
  std::uint64_t most_scratch = 0;
  std::uint64_t most_lane_scratch = 0;  // a wave lane's (graph_plan.h AssignLanes)
  std::uint64_t most_inputs = 0;
  std::uint64_t draft_inputs = 0;
  std::uint64_t draft_wave_inputs = 0;  // a joined draft's, every slot's
  // The largest plan of each kind (PlannedHostBytes) and the most nodes any
  // plan launches, for plan_floor_bytes().
  std::uint64_t chunk_host = 0;
  std::uint64_t draft_host = 0;
  std::uint64_t draft_wave_host = 0;
  std::uint64_t wave_host = 0;
  std::uint64_t most_nodes = 0;
  {
    auto measure = resources_.MeasuringContext();
    if (!measure) {
      return std::unexpected(measure.error());
    }
    model_.prefill_outa_hca =
        model_.prefill_outa_hca && kg::Dsv4OutASupported(**measure) && OutAWeights(binding_);
    const kg::DeviceChoices choices = kg::DeviceChoicesOf(**measure);
    const auto measurement = [&]() -> std::optional<ActivationMeasurement> {
      if (!o_.startup_activation_threshold) return std::nullopt;
      return ActivationMeasurement{most_activations};
    };
    const auto account = [&](const PlannedBase& planned, std::uint64_t& host) -> Status {
      startup_placement_.Account(planned);
      host = std::max(host, PlannedHostBytes(planned));
      most_nodes = std::max(most_nodes, PlannedNodes(planned));
      most_activations = std::max(most_activations, planned.placement.extent);
      auto scratch = kg::PlanScratch(**measure, planned.plan);
      if (!scratch) {
        return Error(scratch.error().detail);
      }
      most_scratch = std::max(most_scratch, *scratch);
      auto lane_scratch = kg::PlanLaneScratch(**measure, planned.plan);
      if (!lane_scratch) {
        return Error(lane_scratch.error().detail);
      }
      most_lane_scratch = std::max(most_lane_scratch, *lane_scratch);
      most_inputs = std::max(most_inputs, planned.inputs_bytes);
      return {};
    };
    struct Probe {
      std::uint32_t n_past;
      std::uint32_t rows;
      Dsv4ChunkKind kind;
      bool state_only = false;
    };
    std::vector<Probe> probes = {{0, o_.max_rows, Dsv4ChunkKind::kPlain},
                                 {o_.context - o_.max_rows, o_.max_rows, Dsv4ChunkKind::kPlain},
                                 {o_.context - 1, 1, Dsv4ChunkKind::kPlain},
                                 {o_.max_rows, 1, Dsv4ChunkKind::kPlain}};
    // Past the HCA fast path's 256 compressed cells, fund the ordinary
    // 512-cell attention fallback as well as the existing context-end probe.
    if (model_.prefill_outa_hca && o_.max_rows <= kg::kDsv4HcaMaxRows &&
        o_.context >= 32768 + o_.max_rows) {
      probes.push_back({32768, o_.max_rows, Dsv4ChunkKind::kPlain});
    }
    if (speculative()) {
      probes.push_back({0, o_.max_rows, Dsv4ChunkKind::kInject});
      probes.push_back({o_.context - o_.max_rows, o_.max_rows, Dsv4ChunkKind::kInject});
      probes.push_back({0, o_.max_verify, Dsv4ChunkKind::kVerify});
      probes.push_back({o_.context - o_.max_verify, o_.max_verify, Dsv4ChunkKind::kVerify});
    }
    if (o_.state_only_prefill && !model_.exact && dump_.empty()) {
      const auto count = probes.size();
      for (std::size_t i = 0; i < count; ++i) {
        if (probes[i].kind == Dsv4ChunkKind::kVerify) continue;
        auto state = probes[i];
        state.state_only = true;
        probes.push_back(state);
      }
    }
    for (const Probe& probe : probes) {
      auto in = md::Dsv4Chunk(profile_, layout_, probe.n_past, probe.rows, o_.exact,
                              !o_.device_raw_masks);
      if (!in) {
        return std::unexpected(in.error());
      }
      Dsv4Speculation speculation;
      if (probe.kind != Dsv4ChunkKind::kPlain) {
        speculation = {.verify = probe.kind == Dsv4ChunkKind::kVerify,
                       .drafter = &dmodel_,
                       .inject_rows = static_cast<std::int64_t>(
                           md::DsparkInject(dlayout_, probe.n_past, probe.rows).cells.size())};
      }
      speculation.state_only = probe.state_only;
      const std::int64_t requested_outputs = o_.frontier_head && !model_.exact && !o_.full_window &&
                                                     probe.rows > 1 && !speculation.verify &&
                                                     dump_.empty()
                                                 ? 1
                                                 : 0;
      const auto shape = kg::Dsv4ShapeOf(layout_, *in, requested_outputs);
      auto planned = PlanDsv4Chunk(model_, shape, choices, dump_, 0, 0, speculation,
                                   HcaFirstPosition(model_, shape, probe.n_past), measurement());
      if (!planned) {
        return Error(std::format("measuring a chunk of {} at {}: {}", probe.rows, probe.n_past,
                                 planned.error()));
      }
      if (auto r = account(**planned, chunk_host); !r) {
        return r;
      }
      if (!probe.state_only && probe.rows == 1 && probe.kind == Dsv4ChunkKind::kPlain &&
          !speculative() && !model_.exact && dump_.empty()) {
        auto token_shape = shape;
        token_shape.token = true;
        auto token_plan =
            PlanDsv4Chunk(model_, token_shape, choices, {}, 0, 0, {}, std::nullopt, measurement());
        if (!token_plan) return Error(token_plan.error());
        if (auto r = account(**token_plan, chunk_host); !r) return r;
      }
    }
    if (speculative()) {
      auto planned = PlanDsparkDraft(dmodel_, o_.draft_rows, choices, 0, 0, measurement());
      if (!planned) {
        return Error(std::format("measuring the draft block: {}", planned.error()));
      }
      if (auto r = account(**planned, draft_host); !r) {
        return r;
      }
      draft_inputs = (*planned)->inputs_bytes;
    }
    if (wave_slots_ > 1) {
      // The widest waves at the context's end: every slot a decode step
      // (beside a drafter, injected) and every slot an equal share of a
      // verify's rows.
      const auto wave = [&](std::uint32_t rows) -> Status {
        kg::Dsv4WaveShape shape;
        std::vector<std::uint64_t> states(wave_slots_, std::uint64_t{1} << 45U);
        for (std::uint32_t s = 0; s < wave_slots_; ++s) {
          auto in = md::Dsv4Chunk(profile_, layout_, o_.context - rows, rows, false,
                                  !o_.device_raw_masks);
          if (!in) {
            return std::unexpected(in.error());
          }
          shape.slots.push_back(kg::Dsv4ShapeOf(layout_, *in));
          if (speculative()) {
            shape.inject_rows.push_back(static_cast<std::int64_t>(
                md::DsparkInject(dlayout_, o_.context - rows, rows).cells.size()));
          }
        }
        auto planned = PlanDsv4Wave(model_, states, shape, choices, 0, 0,
                                    speculative() ? &dmodel_ : nullptr, states, measurement());
        if (!planned) {
          return Error(
              std::format("measuring a wave of {} slots: {}", wave_slots_, planned.error()));
        }
        if (auto r = account(**planned, wave_host); !r) return r;
        if (rows == 1 && !model_.exact) {
          shape.token = true;
          auto token_plan = PlanDsv4Wave(model_, states, shape, choices, 0, 0,
                                         speculative() ? &dmodel_ : nullptr, states, measurement());
          if (!token_plan) return Error(token_plan.error());
          if (auto r = account(**token_plan, wave_host); !r) return r;
        }
        return {};
      };
      if (auto r = wave(1); !r) {
        return r;
      }
      const auto share = std::min<std::uint32_t>(
          o_.max_verify, static_cast<std::uint32_t>(kg::kDsv4WaveRows) / wave_slots_);
      if (speculative() && share > 0) {
        if (auto r = wave(share); !r) {
          return r;
        }
      }
      // Every slot's draft block as one graph, the widest whose rows fit a
      // wave's.
      const auto joined =
          std::min<std::uint32_t>({wave_slots_, static_cast<std::uint32_t>(kg::kDsv4WaveSlots),
                                   static_cast<std::uint32_t>(kg::kDsv4WaveRows) /
                                       std::max<std::uint32_t>(o_.draft_rows, 1)});
      joined_drafts_ =
          speculative() && o_.joined_drafts && !model_.exact && joined >= 2 && o_.draft_rows >= 2;
      if (joined_drafts_) {
        const std::vector<std::uint64_t> rings(joined, std::uint64_t{1} << 45U);
        auto planned = PlanDsparkWave(dmodel_, rings, o_.draft_rows, choices, 0, 0,
                                      model_.wave_lanes, measurement());
        if (!planned) {
          return Error(
              std::format("measuring a joined draft of {} slots: {}", joined, planned.error()));
        }
        if (auto r = account(**planned, draft_wave_host); !r) {
          return r;
        }
        draft_wave_inputs = (*planned)->inputs_bytes;
      }
    }
  }
  // What one step holds at once at most (plan_floor_bytes): a chunk (or a
  // verify) beside its slot's draft block, or a wave beside every slot's
  // draft block. Every plan and graph past it is charged inside the budget.
  {
    const std::uint64_t slots = wave_slots_;
    const std::uint64_t draft = speculative() ? draft_host : 0;
    const std::uint64_t chunk_step = chunk_host + draft;
    const std::uint64_t wave_step =
        slots > 1 ? wave_host + std::max(slots * draft, draft_wave_host) : 0;
    plan_floor_bytes_ = std::max(chunk_step, wave_step);
    prefill_plan_allowance_ = chunk_host;
    if (o_.prefill_lookahead) {
      if (prefill_plan_allowance_ >
          (std::numeric_limits<std::uint64_t>::max() - plan_floor_bytes_) / 2)
        return Error("DeepSeek prefill plan allowances overflow");
      plan_floor_bytes_ += 2 * prefill_plan_allowance_;
    }
    const auto mib = [](std::uint64_t bytes) { return static_cast<double>(bytes) / (1U << 20U); };
    plan_report_ = std::format(
        "{:.1f} MiB a step at most: chunk plans of {:.1f} MiB, wave plans of {:.1f} MiB, draft "
        "plans of {:.1f} MiB (the most nodes a plan launches: {}, a graph of it {:.0f} MiB)",
        mib(plan_floor_bytes_), mib(chunk_host), mib(wave_host), mib(draft_host), most_nodes,
        mib(most_nodes * kGraphNodeHostBytes));
  }
  // Margins: other shapes of these widths place a little differently.
  activation_bytes_ = Round(most_activations + (most_activations / 4), kExtent);
  scratch_bytes_ = Round(most_scratch + (most_scratch / 4) + (1U << 20U), kExtent);
  // A wave's slots run their attention and state operations on concurrent
  // lanes, each with a scratch pool of its own past the stream's.
  lane_scratch_ = wave_slots_ > 1 && o_.wave_lanes
                      ? Round(most_lane_scratch + (most_lane_scratch / 4) + (1U << 18U), 256)
                      : 0;
  scratch_bytes_ = Round(scratch_bytes_ + (kg::kMaxLanes * lane_scratch_), kExtent);
  const std::uint64_t input_bytes = Round((most_inputs * 2) + (1U << 20U), kExtent);
  // A chunk's host-built inputs (Dsv4ChunkInputs and the embedding rows)
  // are the staged bytes again, on the host.
  host_input_bytes_ = Round(most_inputs + (1U << 20U), kExtent);
  // A verify's (and a wave's) inputs from the staging's second half, which
  // the largest inputs fit; each slot's draft block's below it.
  verify_base_ = Round(input_bytes / 2, 256);
  draft_staging_ = Round(draft_inputs + 256, 4096);
  if (draft_staging_ * wave_slots_ > verify_base_) {
    return Error("the draft blocks' staging does not fit below the verify's");
  }
  // A joined draft's inputs from the staging's start, below the verify's
  // (the draft blocks' places: one or the other runs in a wave).
  joined_drafts_ = joined_drafts_ && draft_wave_inputs + 256 <= verify_base_;

  auto inputs = resources_.Pinned(input_bytes);
  std::uint64_t table_bytes = 0;
  for (const md::Dsv4Layer& l : binding_.layers) {
    if (l.hash) {
      table_bytes += weights_.artifact().resources()[l.tid2eid.index].bytes.value();
    }
  }
  auto tables = resources_.Pinned(std::max<std::uint64_t>(table_bytes, 256));
  if (!inputs || !tables) {
    return Error("pinned staging for DeepSeek");
  }
  runs_.SetStaging(*inputs, input_bytes);
  hash_tables_ = *tables;
  if (speculative()) {
    // A verify's snapshot: every byte it may write, the target's and the
    // ring's (D-068 working state).
    snapshot_bytes_ =
        md::Dsv4VerifySnapshotBytes(profile_, layout_, o_.max_verify) +
        (std::uint64_t{o_.max_verify} * dprofile_.blocks.layers * dprofile_.blocks.head_dim * 2);
  }
  for (RequestState* request : Requests()) {
    if (request->provisioned) {
      if (auto r = SetupSlot(*request, snapshot_bytes_); !r) {
        return r;
      }
    }
  }
  if (wave_slots_ > 1) {
    auto logits = resources_.Pinned(static_cast<std::uint64_t>(kg::kDsv4WaveRows) * profile_.vocab *
                                    sizeof(float));
    if (!logits) {
      return Error("pinned staging for DeepSeek's waves");
    }
    wave_logits_ = static_cast<float*>(*logits);
  }
  setup_seconds_ = Seconds(std::chrono::steady_clock::now() - setup_started);
  return {};
}

Status Dsv4Runner::SetupSlot(RequestState& request, std::uint64_t snapshot_bytes) {
  const std::uint64_t logit_rows = speculative() ? o_.max_verify : 1;
  auto logits = resources_.Pinned(logit_rows * std::uint64_t{profile_.vocab} * sizeof(float));
  if (!logits) {
    return Error("pinned staging for DeepSeek");
  }
  request.logits = *logits;
  if (request.slot != 0) {
    // The default request's places were measured above with it.
    request.model = model_;
    request.dmodel = dmodel_;
  }
  if (!speculative()) {
    return {};
  }
  const std::string name = request.slot == 0
                               ? "the DeepSeek verify snapshot"
                               : std::format("the DeepSeek slot {} verify snapshot", request.slot);
  if (auto r = resources_.Map(request.snapshot, name, snapshot_bytes, MemoryClass::kRuntime); !r) {
    return r;
  }
  request.live.SnapshotAt(request.snapshot.base, request.snapshot.bytes);
  if (auto r = request.live.AllocateSnapshot(resources_, kRangeCapacity); !r) {
    return r;
  }
  auto drafts =
      resources_.Pinned(std::max<std::uint64_t>(o_.draft_rows * sizeof(std::int32_t), 256));
  if (!drafts) {
    return Error("pinned staging for DeepSeek's speculation");
  }
  request.drafts = *drafts;
  return {};
}

// A part's places, one extent per dense chunk, per slab page and per host
// table chunk, cataloged in file order (paged_weights.h); each layer's
// expert stride in `stride`.
Status Dsv4Runner::ReservePart(PagedWeights& part, const md::Dsv4Binding& binding,
                               std::uint32_t layers, std::uint32_t experts, bool table,
                               std::vector<std::uint64_t>& stride) {
  const artifact::Artifact& a = part.artifact();
  const auto groups = a.groups();
  std::vector<GroupPlace> place(groups.size(), GroupPlace::kDevice);
  if (table) {
    table_group_ = a.resources()[binding.token_embd.index].group;
    for (std::uint32_t r = 0; r < a.resources().size(); ++r) {
      if (r != binding.token_embd.index && a.resources()[r].group == table_group_) {
        return Error(std::format("{} shares the token table's group", a.resources()[r].name));
      }
    }
    place[table_group_] = GroupPlace::kHost;
  }
  // Each layer's expert slab, at the resident layout's stride.
  std::vector<SlabSpec> slabs;
  stride.assign(layers, 0);
  for (std::uint32_t il = 0; il < layers; ++il) {
    const md::Dsv4Layer& l = binding.layers[il];
    const std::array<std::pair<std::uint32_t, std::string_view>, 3> arrays = {
        {{l.gate_exps.index, l.gate_exps.type},
         {l.up_exps.index, l.up_exps.type},
         {l.down_exps.index, l.down_exps.type}}};
    auto slab = ExpertSlab(a, arrays, experts, 256, il);
    if (!slab) {
      return std::unexpected(slab.error());
    }
    stride[il] = slab->stride;
    slabs.push_back(*slab);
  }
  // Every other expert group belongs to no bound array; dense groups are
  // placed whole.
  std::vector<bool> in_slab(groups.size(), false);
  for (const SlabSpec& s : slabs) {
    for (std::uint32_t e = 0; e < s.count && s.first_group + e < groups.size(); ++e) {
      in_slab[s.first_group + e] = true;
    }
  }
  for (std::uint32_t g = 0; g < groups.size(); ++g) {
    if (groups[g].kind == artifact::GroupKind::kExpert) {
      if (!in_slab[g]) {
        return Error(std::format("expert group {} belongs to no bound array", g));
      }
      place[g] = GroupPlace::kNone;
    }
  }
  return part.Reserve(node_, place, slabs);
}

Status Dsv4Runner::Register() {
  if (auto r = weights_.Register(node_, owner_); !r) {
    return Error(std::format("a DeepSeek weight: {}", r.error()));
  }
  if (speculative()) {
    if (auto r = dweights_.Register(node_, owner_); !r) {
      return Error(std::format("a DSpark weight: {}", r.error()));
    }
  }
  for (RequestState* request : Requests()) {
    if (request->provisioned) {
      auto r = o_.spill_place ? request->live.RegisterSpill(node_, o_.spill_place(request->slot))
                              : request->live.RegisterSpill(node_, o_.out);
      if (!r) {
        return r;
      }
    }
  }
  // D-090: the places every graph will name stay put for the model's life.
  auto pinned_extents = weights();
  for (const RequestState* request : Requests()) {
    if (request->provisioned) {
      const auto reserved = request->live.reserved_extents();
      pinned_extents.insert(pinned_extents.end(), reserved.begin(), reserved.end());
    }
  }
  if (auto pinned = node_.scheduler().PinPlaces(pinned_extents); !pinned) {
    return Error(std::format("pinning DeepSeek's places: {}", sc::ToString(pinned.error())));
  }
  return {};
}

Status Dsv4Runner::CheckPlaces() {
  // Nothing placed since the last clean check: every place is as it was, so
  // no turn of the scheduler's is needed (each object's own stamp would
  // skip it all).
  const std::uint64_t changes = node_.scheduler().placement_changes();
  if (places_clean_ == changes) return {};
  PlaceCheck check;
  auto checked = node_.Call(
      [&]() -> Status {
        weights_.CheckPlaces(node_.scheduler(), check);
        dweights_.CheckPlaces(node_.scheduler(), check);
        for (const RequestState* request : Requests()) {
          if (request->provisioned) {
            request->live.CheckPlaces(node_.scheduler(), check);
          }
        }
        return {};
      },
      "checking DeepSeek's places");
  if (!checked) {
    return checked;
  }
  if (check.moved != 0) {
    DropPlans();
    return Error(
        std::format("{} extents are no longer pinned at their places (first: {}); every "
                    "graph was dropped",
                    check.moved, check.first));
  }
  places_clean_ = changes;
  return {};
}

Status Dsv4Runner::RefreshClosures(SlotMask protected_mask) {
  places_clean_.reset();  // the states checked may change
  auto refreshed = node_.Call(
      [&]() -> Status {
        auto& catalog = node_.catalog();
        // What every job leases beside the state: the weights, the
        // workspace, the model's own memory and staging.
        std::vector<ExtentId> common;
        for (const Mapped* mapped :
             std::initializer_list<const Mapped*>{&node_.activations(), &node_.pool()}) {
          common.insert(common.end(), mapped->extents.begin(), mapped->extents.end());
        }
        const std::vector<ExtentId> own = resources_.extents();
        common.insert(common.end(), own.begin(), own.end());
        std::vector<ExtentId> shared = weights();
        shared.insert(shared.end(), common.begin(), common.end());
        std::array<const LiveState*, kRequestSlots> states{};
        for (const RequestState* request : Requests()) {
          // A spilled slot's state is in no closure until Restore brings it back.
          states[request->slot] =
              request->provisioned && !request->spilled ? &request->live : nullptr;
        }
        auto closures = cohort_.Build(catalog, shared, states, protected_mask);
        if (!closures) {
          return std::unexpected(closures.error());
        }
        everything_ = std::move(closures->everything);
        fence_ = std::move(closures->fence);
        execution_ = std::move(closures->execution);
        for (RequestState* request : Requests()) {
          request->fence = std::move(closures->slot_fences[request->slot]);
        }
        if (speculative()) {
          // A draft reads the drafter's weights, the target's head (its group)
          // and token table; its job restores a verify's rows first (the
          // protected states and their snapshots).
          const std::uint32_t head = weights_.artifact().resources()[binding_.output.index].group;
          std::vector<ExtentId> draft = dweights_.extents();
          const std::vector<ExtentId> target = weights_.ExtentsWhere(
              [head](std::uint32_t group, bool host) { return host || group == head; });
          draft.insert(draft.end(), target.begin(), target.end());
          draft.insert(draft.end(), common.begin(), common.end());
          for (const RequestState* request : Requests()) {
            if (request->provisioned && !request->spilled &&
                (protected_mask & (SlotMask{1} << request->slot)) != 0) {
              const auto live = request->live.extents();
              draft.insert(draft.end(), live.begin(), live.end());
            }
          }
          auto of = catalog.ClosureOfExtents(draft);
          if (!of) {
            return Error("the DeepSeek draft closure is no longer cataloged");
          }
          draft_closure_ = std::move(*of);
        }
        return {};
      },
      "refreshing DeepSeek's used state closure");
  const auto states = States();
  if (!refreshed) {
    cohort_.Fault(states);
    return refreshed;
  }
  return cohort_.Hold(node_, stream_, execution_, states);
}

void Dsv4Runner::BindSlot(RequestState& request) {
  request.model.places.resource = [this](std::uint32_t resource) {
    return weights_.resource_address(resource);
  };
  request.model.places.array = [this](std::uint32_t array) {
    return weights_.array_address(array);
  };
  request.model.places.state = request.live.base(kTarget);
  if (speculative()) {
    request.dmodel.places.resource = [this](std::uint32_t resource) {
      return dweights_.resource_address(resource);
    };
    request.dmodel.places.array = [this](std::uint32_t array) {
      return dweights_.array_address(array);
    };
    request.dmodel.places.state = request.live.base(kDrafter);
    request.dmodel.target_resource = request.model.places.resource;
  }
}

Status Dsv4Runner::Bind() {
  if (auto refreshed = RefreshClosures(); !refreshed) {
    return refreshed;
  }
  // Every plan and graph charged to the node as it is kept.
  account_.Bind(
      [this](std::uint64_t bytes, bool required) { return node_.ChargeHost(bytes, required); },
      [this](std::uint64_t bytes) { node_.UnchargeHost(bytes); });
  for (RequestState* request : Requests()) {
    request->plans.set_account(&account_);
    request->dplans.set_account(&account_);
  }
  waves_.set_account(&account_);
  dwaves_.set_account(&account_);
  for (RequestState* request : Requests()) {
    if (request->provisioned) {
      BindSlot(*request);
    }
  }
  if (auto r = resources_.BindLaunch(scratch_bytes_); !r) {
    return r;
  }
  if (lane_scratch_ > 0) {
    if (auto r = resources_.launch().ConfigureLanes(kg::kMaxLanes, base::Bytes(lane_scratch_));
        !r) {
      return Error(r.error().detail);
    }
  }
  runs_.SetLaunch(&resources_.launch());
  return {};
}

// ------------------------------------------------------------------ work

Status Dsv4Runner::Clear(RequestState& request) {
  if (auto active = CheckActive(request); !active) {
    return active;
  }
  // Zeroed in place, the discard keeps the resident backing for the next
  // conversation's growth (LiveState::ZeroForReuse).
  bool keep = false;
  if (!request.spilled) {
    auto zeroed = request.live.ZeroForReuse(node_, request.fence, stream_);
    if (!zeroed) {
      request.track.Lost();
      cohort_.CheckFailedJob(node_, stream_, execution_, States());
      return Error(zeroed.error());
    }
    keep = *zeroed;
  }
  // Remove only this slot's state from the held request; the shared
  // extents and every other active slot's state stay protected.
  const SlotMask others = cohort_.active() & ~(SlotMask{1} << request.slot);
  if (auto protected_others = RefreshClosures(others); !protected_others) {
    return protected_others;
  }
  const Status cleared = request.live.DiscardGrowingState(node_, keep);
  request.track.Lost();
  if (cleared) {
    request.spilled = false;  // nothing left in its spill file either
    request.adopted.clear();
    request.adopted_bytes = 0;
  }
  if (auto refreshed = RefreshClosures(); !refreshed) {
    return refreshed;
  }
  return cleared;
}

Status Dsv4Runner::Adopt(RequestState& request, std::span<const LiveState::Range> used) {
  if (released_ || cohort_.faulted()) {
    return Error("the DeepSeek cohort requires retirement");
  }
  if (!request.provisioned || request.spilled || !request.live.extents().empty() || used.empty() ||
      (cohort_.IsActive(request.slot) && node_.InRequest(stream_))) {
    return Error("a DeepSeek slot adopts a kept conversation only while it holds none");
  }
  if (auto bytes = request.live.UsedBytesOf(used); !bytes) {
    return std::unexpected(bytes.error());
  }
  // Spilled, outside every closure, until Restore reads it from the file.
  request.adopted.assign(used.begin(), used.end());
  request.adopted_bytes = used.size() * kPagedExtent;
  request.spilled = true;
  request.track.Lost();
  return RefreshClosures();
}

std::expected<SlotMask, std::string> Dsv4Runner::RecoverInPlace(
    const std::function<void(std::uint32_t slot)>& before_discard) {
  if (released_) {
    return Error("the DeepSeek runner is released");
  }
  if (node_.InRequest(stream_)) {
    return Error("a request is still open on DeepSeek's stream");
  }
  if (resources_.launch().faulted()) {
    return Error(
        "DeepSeek's launch context faulted: a launch of unknown outcome is not proven "
        "retired");
  }
  const bool faulted = cohort_.faulted();
  cohort_.Recover();
  SlotMask discarded = 0;
  for (RequestState* request : Requests()) {
    if (!request->provisioned) {
      continue;
    }
    // On disk since before the failure, and nothing ran on it: kept.
    if (request->spilled && request->live.LiftIfPreserved(node_)) {
      continue;
    }
    if (!faulted && !request->live.quarantined()) {
      continue;  // untouched by the failed work
    }
    if (before_discard) {
      before_discard(request->slot);
    }
    request->state_refused = false;
    const Status cleared = request->live.DiscardGrowingState(node_);
    request->track.Lost();
    if (!cleared) {
      cohort_.Fault(States());
      return Error(std::format("discarding a DeepSeek slot's state: {}", cleared.error()));
    }
    request->spilled = false;
    request->adopted.clear();
    request->adopted_bytes = 0;
    discarded |= SlotMask{1} << request->slot;
  }
  DropPlans();
  if (auto refreshed = RefreshClosures(); !refreshed) {
    return std::unexpected(refreshed.error());
  }
  return discarded;
}

std::string Dsv4Runner::kept_layout() const {
  // Every slot's state has slot 0's layout. Version 1 of the state format:
  // a change to what a region's bytes mean bumps it.
  const LiveState& live = default_request_.live;
  std::string layout =
      std::format("deepseek4-state/1;context={};dspark={};window={};regions=", o_.context,
                  speculative() ? 1 : 0, o_.full_window || o_.exact ? "full" : "ring");
  for (std::size_t i = 0; i < live.regions(); ++i) {
    layout += std::format("{}{}:{}:{}", i == 0 ? "" : ",", live.region_name(i), live.bytes(i),
                          live.mapped_bytes(i));
  }
  return layout;
}

Status Dsv4Runner::Spill(RequestState& request) {
  if (released_ || cohort_.faulted()) {
    return Error("the DeepSeek cohort requires retirement");
  }
  if (!request.provisioned) {
    return Error("spilling a DeepSeek slot that is not provisioned");
  }
  if (request.spilled) {
    return {};
  }
  if (auto usable = request.live.Usable(); !usable) {
    return usable;
  }
  if (auto awaiting = request.live.AwaitingAccept(); !awaiting) {
    return awaiting;
  }
  const std::vector<ExtentId> extents = request.live.extents();
  if (extents.empty()) {
    return {};  // nothing initialized: nothing to spill
  }
  // Out of every closure first, so the held request no longer leases it
  // (a selected slot leaves the lease; its peers stay protected); then
  // every initialized extent written back to its place in the slot's
  // spill file and its backing released (the swap's write-back path). A
  // restore owed by its last verify stays owed: the snapshot it restores
  // from is the slot's own runtime memory, not spilled.
  std::vector<ExtentId> written;
  std::vector<ExtentId> unchanged;
  SplitForSpill(request, written, unchanged);
  request.spilled = true;
  if (auto refreshed = RefreshClosures(); !refreshed) {
    request.spilled = false;
    return refreshed;
  }
  // Only what changed since the spill file last held it is written; the
  // rest is released with the file's copy kept (the catalog checks its
  // generation is the one saved there).
  for (const auto& [extents_of, unchanged_word] :
       {std::pair{&written, false}, std::pair{&unchanged, true}}) {
    if (extents_of->empty()) {
      continue;
    }
    if (auto evicted = node_.Evict(*extents_of, {.unchanged = unchanged_word}); !evicted) {
      // Some extents may be written back and released, others not: still
      // spilled, so Restore materializes whichever are not resident.
      request.track.Lost();
      return evicted;
    }
  }
  request.track.Saved();
  return {};
}

void Dsv4Runner::SplitForSpill(const RequestState& request, std::vector<ExtentId>& written,
                               std::vector<ExtentId>& unchanged) const {
  std::vector<LiveState::Range> changed = request.track.written;
  bool known = request.track.on_disk;
  if (known && request.track.written_from != SpillTrack::kUnwritten) {
    auto writes = StateWrites(request.track.written_from);
    if (writes) {
      changed.insert(changed.end(), writes->begin(), writes->end());
    } else {
      known = false;
    }
  }
  if (!known) {
    written = request.live.extents();
    return;
  }
  request.live.SplitExtents(changed, written, unchanged);
}

std::uint64_t Dsv4Runner::SpillWriteBytes(const RequestState& request) const {
  if (request.spilled) {
    return 0;
  }
  std::vector<ExtentId> written;
  std::vector<ExtentId> unchanged;
  SplitForSpill(request, written, unchanged);
  return written.size() * kPagedExtent;
}

Status Dsv4Runner::Restore(RequestState& request) {
  request.state_refused = false;
  if (!request.spilled) {
    return {};
  }
  if (released_ || cohort_.faulted()) {
    return Error("the DeepSeek cohort requires retirement");
  }
  if (!request.adopted.empty()) {
    // Kept from the process before (D-105): its extents are fresh here, and
    // their first load reads them from the named spill file, which holds
    // them (LiveState::Use), under the budget: a clean refusal leaves it
    // spilled for the runtime to make room.
    bool over_budget = false;
    if (auto used = request.live.Use(node_, request.adopted, nullptr, &over_budget); !used) {
      request.state_refused = over_budget && !cohort_.faulted();
      return std::unexpected(used.error());
    }
    request.adopted.clear();
    request.adopted_bytes = 0;
    request.spilled = false;
    request.track.Saved();  // the file holds the state as it is now
    return RefreshClosures();
  }
  catalog::Closure closure;
  const std::vector<ExtentId> extents = request.live.extents();
  if (auto described = node_.Call(
          [&]() -> Status {
            auto of = node_.catalog().ClosureOfExtents(extents);
            if (!of) {
              return Error("a spilled DeepSeek state is no longer cataloged");
            }
            closure = std::move(*of);
            return {};
          },
          "describing a spilled DeepSeek state");
      !described) {
    return described;
  }
  // Each extent read back from the spill file at the generation it was
  // written at (invariant 4), under the budget: a clean refusal leaves it
  // spilled, for the runtime to make room and try again.
  sc::AcquireReport report;
  bool over_budget = false;
  if (auto acquired =
          node_.Acquire(closure, report, "restoring a spilled DeepSeek state", &over_budget);
      !acquired) {
    request.state_refused = over_budget && !cohort_.faulted();
    return acquired;
  }
  request.spilled = false;
  // Read back whole: the file still holds it as it is, if the spill that
  // wrote it completed (SpillTrack: nothing writes a spilled slot).
  return RefreshClosures();
}

Status Dsv4Runner::ClearIdle(RequestState& request) {
  if (released_ || cohort_.faulted()) {
    return Error("the DeepSeek cohort requires retirement");
  }
  // Leased by the request open on the stream: cleared within its own
  // request (Clear). Selected last but with no request open, it is idle.
  if (!request.provisioned || (cohort_.IsActive(request.slot) && node_.InRequest(stream_))) {
    return Error("an active DeepSeek slot is cleared within its own request");
  }
  request.state_refused = false;
  // Not selected, so outside the stream's lease, which keeps protecting
  // every active state; the catalog refuses the discard if anything still
  // holds or operates on this state's extents (DiscardGrowingState).
  const Status cleared = request.live.DiscardGrowingState(node_);
  request.track.Lost();
  if (cleared) {
    request.spilled = false;
    request.adopted.clear();
    request.adopted_bytes = 0;
  }
  if (auto refreshed = RefreshClosures(); !refreshed) {
    return refreshed;
  }
  return cleared;
}

std::expected<std::vector<LiveState::Range>, std::string> Dsv4Runner::StateRanges(
    std::uint32_t positions) const {
  auto needed = md::Dsv4UsedState(layout_, positions);
  if (!needed) return std::unexpected(needed.error());
  std::vector<LiveState::Range> ranges;
  ranges.reserve(needed->size() + (speculative() ? 1U : 0U));
  for (const auto& range : *needed)
    ranges.push_back({.region = kTarget, .offset = range.offset, .bytes = range.bytes});
  if (speculative()) ranges.push_back({.region = kDrafter, .offset = 0, .bytes = dlayout_.bytes});
  return ranges;
}

LiveState::PreparationStats Dsv4Runner::state_preparation_stats() const {
  LiveState::PreparationStats total;
  for (const auto* request : Requests()) {
    if (request == nullptr) continue;
    const auto& s = request->live.preparation_stats();
    total.attempted += s.attempted;
    total.submitted += s.submitted;
    total.refused += s.refused;
    total.failed += s.failed;
    total.completed_extents += s.completed_extents;
    total.adopted_extents += s.adopted_extents;
    total.cancel_requested += s.cancel_requested;
  }
  return total;
}

Status Dsv4Runner::EnsureState(RequestState& request, std::uint32_t positions) {
  request.state_refused = false;
  if (auto usable = Usable(request); !usable) return usable;
  auto ranges = StateRanges(positions);
  if (!ranges) return std::unexpected(ranges.error());
  bool over_budget = false;
  auto used = request.live.Use(node_, *ranges, &execution_, &over_budget);
  if (!used) {
    if (auto refreshed = RefreshClosures(); !refreshed) {
      request.live.Quarantine();
      return Error(std::format("{}; {}", used.error(), refreshed.error()));
    }
    // Only a clean refusal that left the state usable and the cohort
    // healthy may be retried once leased state is freed (Slot::state_refused).
    request.state_refused = over_budget && !cohort_.faulted() && !request.live.quarantined();
    return std::unexpected(used.error());
  }
  return *used ? RefreshClosures() : Status{};
}

std::expected<std::uint64_t, std::string> Dsv4Runner::StateBytesThrough(
    std::uint32_t positions) const {
  // EnsureState's ranges, as a slot's growth through `positions` takes them.
  auto ranges = StateRanges(std::min(positions, o_.context));
  if (!ranges) return std::unexpected(ranges.error());
  return live_.UsedBytesOf(*ranges);
}

Status Dsv4Runner::CheckHashRouting() {
  std::vector<std::pair<std::uint64_t, std::uint64_t>> tables;  // address, bytes
  for (const md::Dsv4Layer& l : binding_.layers) {
    if (l.hash) {
      tables.emplace_back(model_.places.resource(l.tid2eid.index),
                          weights_.artifact().resources()[l.tid2eid.index].bytes.value());
    }
  }
  void* host = hash_tables_;
  if (auto r = node_.Job(
          execution_,
          [&tables, host](providers::NativeStream stream) {
            std::uint64_t at = 0;
            for (const auto& [address, bytes] : tables) {
              if (!providers::CopyAsync(stream, static_cast<std::byte*>(host) + at,
                                        Pointer(address), bytes, providers::CopyKind::kDeviceToHost)
                       .ok()) {
                return sc::JobResult::kUnknown;
              }
              at += bytes;
            }
            return sc::JobResult::kQueued;
          },
          "reading the hash-routing tables", stream_);
      !r) {
    return r;
  }
  std::uint64_t at = 0;
  std::uint32_t il = 0;
  for (const auto& [address, bytes] : tables) {
    const std::span<const std::int32_t> table(
        reinterpret_cast<const std::int32_t*>(static_cast<const std::byte*>(host) + at),
        bytes / sizeof(std::int32_t));
    if (auto checked = md::CheckDsv4HashRouting(profile_, table); !checked) {
      return Error(std::format("hash table {}: {}", il, checked.error()));
    }
    at += bytes;
    ++il;
  }
  return {};
}

std::expected<Dsv4Runner::ChunkPlans::Entry*, std::string> Dsv4Runner::Planned(
    RequestState& request, const ChunkKey& key, std::uint32_t first) {
  if (ChunkPlans::Entry* found = request.plans.Find(key); found != nullptr) {
    if (key.hca) {
      // The plan's HCA nodes take this chunk's position (no re-plan); it
      // is never captured (Chunk), which this refuses to rely on.
      if (auto r = SetDsv4HcaFirstPosition(request.model, found->planned->graph, first,
                                           found->has_graph());
          !r) {
        return std::unexpected(r.error());
      }
    }
    return found;
  }
  const auto start = std::chrono::steady_clock::now();
  auto planned = BuildPrefillPlan(request, key, first, kg::DeviceChoicesOf(resources_.launch()));
  if (!planned) return std::unexpected(planned.error());
  return CachePrefillPlan(request, key, std::move(*planned),
                          Seconds(std::chrono::steady_clock::now() - start));
}

std::expected<std::unique_ptr<Dsv4Planned>, std::string> Dsv4Runner::BuildPrefillPlan(
    RequestState& request, const ChunkKey& key, std::uint32_t first,
    const kg::DeviceChoices& choices) {
  Dsv4Speculation speculation;
  if (key.kind != Dsv4ChunkKind::kPlain)
    speculation = {.verify = key.kind == Dsv4ChunkKind::kVerify,
                   .drafter = &request.dmodel,
                   .inject_rows = key.inject_rows};
  speculation.state_only = key.state_only;
  return PlanDsv4Chunk(request.model, key.shape, choices, dump_, node_.activations().base,
                       node_.activations().bytes, speculation,
                       key.hca ? std::optional<std::uint32_t>(first) : std::nullopt);
}

std::expected<Dsv4Runner::ChunkPlans::Entry*, std::string> Dsv4Runner::CachePrefillPlan(
    RequestState& request, const ChunkKey& key, std::unique_ptr<Dsv4Planned> planned,
    double seconds, const std::function<void()>& transfer) {
  const auto binding = std::chrono::steady_clock::now();
  if (auto r = BindPlanned(*planned, resources_.launch(), resources_.registry(), "the plan"); !r)
    return std::unexpected(r.error());
  bound_raw_masks_ +=
      static_cast<std::uint64_t>(std::ranges::count_if(planned->plan.steps, [](const auto& step) {
        return step.implementation == kg::kGemma4MaskName;
      }));
  Check(planned->graph);
  seconds += Seconds(std::chrono::steady_clock::now() - binding);
  const auto bytes = PlannedHostBytes(*planned), nodes = PlannedNodes(*planned);
  if (transfer) transfer();
  plan_seconds_ += seconds;
  return &request.plans.Add(key, std::move(planned), bytes, nodes, seconds);
}

std::expected<Dsv4Runner::DraftPlans::Entry*, std::string> Dsv4Runner::PlannedDraft(
    RequestState& request) {
  if (DraftPlans::Entry* found = request.dplans.Find(o_.draft_rows); found != nullptr) {
    return found;
  }
  const auto start = std::chrono::steady_clock::now();
  kg::LaunchContext& launch = resources_.launch();
  auto planned = PlanDsparkDraft(request.dmodel, o_.draft_rows, kg::DeviceChoicesOf(launch),
                                 node_.activations().base, node_.activations().bytes);
  if (!planned) {
    return std::unexpected(planned.error());
  }
  if (auto r = BindPlanned(**planned, launch, resources_.registry(), "the draft"); !r) {
    return std::unexpected(r.error());
  }
  bound_draft_masks_ += static_cast<std::uint64_t>(std::ranges::count_if(
      (*planned)->plan.steps,
      [](const auto& step) { return step.implementation == kg::kGemma4MaskName; }));
  const double seconds = Seconds(std::chrono::steady_clock::now() - start);
  plan_seconds_ += seconds;
  const std::uint64_t bytes = PlannedHostBytes(**planned);
  const std::uint64_t nodes = PlannedNodes(**planned);
  return &request.dplans.Add(o_.draft_rows, std::move(*planned), bytes, nodes, seconds);
}

std::expected<Dsv4Runner::WavePlans::Entry*, std::string> Dsv4Runner::PlannedWave(
    const WaveKey& key) {
  if (WavePlans::Entry* found = waves_.Find(key); found != nullptr) {
    return found;
  }
  const auto start = std::chrono::steady_clock::now();
  std::vector<std::uint64_t> states;
  std::vector<std::uint64_t> rings;
  const auto requests = Requests();
  for (std::size_t i = 0; i < key.shape.slots.size(); ++i) {
    const RequestState& request = *requests[key.slots[i]];
    states.push_back(request.live.base(kTarget));
    rings.push_back(request.live.base(kDrafter));
  }
  kg::LaunchContext& launch = resources_.launch();
  auto planned = PlanDsv4Wave(model_, states, key.shape, kg::DeviceChoicesOf(launch),
                              node_.activations().base, node_.activations().bytes,
                              key.shape.inject_rows.empty() ? nullptr : &dmodel_, rings);
  if (!planned) {
    return std::unexpected(planned.error());
  }
  if (auto r = BindPlanned(**planned, launch, resources_.registry(), "the wave"); !r) {
    return std::unexpected(r.error());
  }
  bound_raw_masks_ += static_cast<std::uint64_t>(std::ranges::count_if(
      (*planned)->plan.steps,
      [](const auto& step) { return step.implementation == kg::kGemma4MaskName; }));
  CheckWave((*planned)->graph);
  const double seconds = Seconds(std::chrono::steady_clock::now() - start);
  plan_seconds_ += seconds;
  const std::uint64_t bytes = PlannedHostBytes(**planned);
  const std::uint64_t nodes = PlannedNodes(**planned);
  return &waves_.Add(key, std::move(*planned), bytes, nodes, seconds);
}

std::expected<Dsv4Runner::DraftWavePlans::Entry*, std::string> Dsv4Runner::PlannedDraftWave(
    const DraftWaveKey& key) {
  if (DraftWavePlans::Entry* found = dwaves_.Find(key); found != nullptr) {
    return found;
  }
  const auto start = std::chrono::steady_clock::now();
  std::vector<std::uint64_t> rings;
  rings.reserve(key.count);
  const auto requests = Requests();
  for (std::uint32_t i = 0; i < key.count; ++i) {
    rings.push_back(requests[key.slots[i]]->live.base(kDrafter));
  }
  kg::LaunchContext& launch = resources_.launch();
  auto planned =
      PlanDsparkWave(dmodel_, rings, o_.draft_rows, kg::DeviceChoicesOf(launch),
                     node_.activations().base, node_.activations().bytes, model_.wave_lanes);
  if (!planned) {
    return std::unexpected(planned.error());
  }
  if (auto r = BindPlanned(**planned, launch, resources_.registry(), "the joined draft"); !r) {
    return std::unexpected(r.error());
  }
  bound_draft_masks_ += static_cast<std::uint64_t>(std::ranges::count_if(
      (*planned)->plan.steps,
      [](const auto& step) { return step.implementation == kg::kGemma4MaskName; }));
  const double seconds = Seconds(std::chrono::steady_clock::now() - start);
  plan_seconds_ += seconds;
  const std::uint64_t bytes = PlannedHostBytes(**planned);
  const std::uint64_t nodes = PlannedNodes(**planned);
  return &dwaves_.Add(key, std::move(*planned), bytes, nodes, seconds);
}

// BP-A1's check (planned.h): the state is live state (the target's and the
// drafter's ring), the rest weights or the activations.
void Dsv4Runner::Check(const kg::Dsv4Graph& graph) {
  std::vector<const ggml_tensor*> state;
  for (const kg::Dsv4LayerTensors& l : graph.layers) {
    for (const ggml_tensor* t :
         {l.raw_k, l.csa_k, l.csa_state_kv, l.csa_state_score, l.lid_k, l.lid_state_kv,
          l.lid_state_score, l.hca_k, l.hca_state_kv, l.hca_state_score}) {
      if (t != nullptr) {
        state.push_back(t);
      }
    }
  }
  if (graph.inject) {
    state.insert(state.end(), graph.inject->ring.begin(), graph.inject->ring.end());
  }
  const auto inputs = graph.inputs();
  CheckCoverage(node_, owner_, graph.nodes, {.state = state, .inputs = inputs}, coverage_);
}

void Dsv4Runner::CheckWave(const kg::Dsv4WaveGraph& graph) {
  std::vector<const ggml_tensor*> state;
  for (const kg::Dsv4Graph& slot : graph.slots) {
    for (const kg::Dsv4LayerTensors& l : slot.layers) {
      for (const ggml_tensor* t :
           {l.raw_k, l.csa_k, l.csa_state_kv, l.csa_state_score, l.lid_k, l.lid_state_kv,
            l.lid_state_score, l.hca_k, l.hca_state_kv, l.hca_state_score}) {
        if (t != nullptr) {
          state.push_back(t);
        }
      }
    }
    if (const std::optional<kg::DsparkInjectTensors>& injected = slot.inject;
        injected.has_value()) {
      state.insert(state.end(), injected->ring.begin(), injected->ring.end());
    }
  }
  const auto inputs = graph.inputs();
  CheckCoverage(node_, owner_, graph.joined.nodes,
                {.state = state, .inputs = inputs, .what = "the wave's "}, coverage_);
}

void Dsv4Runner::Settle(RequestState& request, bool saved, bool wrote, bool unknown) {
  const bool uncertain = unknown || resources_.launch().faulted();
  request.live.Settle(saved, wrote, uncertain);
  if (uncertain && wave_slots_ > 1) {
    // The shared stream's effect is unknown: no slot is proven intact.
    cohort_.Fault(States());
  }
}

Status Dsv4Runner::PlanSnapshot(RequestState& request, const md::Dsv4ChunkInputs& in) {
  const md::Dsv4Writes writes = md::Dsv4ChunkWrites(profile_, layout_, in);
  const std::vector<std::vector<md::StateRange>> ring =
      md::DsparkWrites(dprofile_, dlayout_, in.n_past, in.rows);
  const std::uint64_t state = request.live.base(kTarget);
  const std::uint64_t drafter = request.live.base(kDrafter);
  request.live.BeginSaves();
  for (std::uint32_t i = 0; i < in.rows; ++i) {
    for (const md::StateRange& r : writes.rows[i]) {
      if (auto added = request.live.Save(state + r.offset, r.bytes, i); !added) {
        return added;
      }
    }
    for (const md::StateRange& r : ring[i]) {
      if (auto added = request.live.Save(drafter + r.offset, r.bytes, i); !added) {
        return added;
      }
    }
  }
  for (const md::StateRange& r : writes.scratch) {
    if (auto added = request.live.Save(state + r.offset, r.bytes, -1); !added) {
      return added;
    }
  }
  return {};
}

std::vector<Dsv4Runner::VerifyWrite> Dsv4Runner::last_verify_writes(const RequestState& request) {
  const std::uint64_t ring_base = request.live.base(kDrafter);
  const std::uint64_t ring_bytes = request.live.bytes(kDrafter);
  std::vector<VerifyWrite> out;
  out.reserve(request.live.saved().size());
  for (const LiveState::Saved& s : request.live.saved()) {
    const bool ring = s.address >= ring_base && s.address < ring_base + ring_bytes;
    out.push_back({.ring = ring,
                   .offset = s.address - (ring ? ring_base : request.live.base(kTarget)),
                   .bytes = s.bytes,
                   .row = s.row});
  }
  return out;
}

Status Dsv4Runner::Accept(RequestState& request, std::uint32_t keep) {
  if (auto usable = Usable(request); !usable) {
    return usable;
  }
  return request.live.Accept(keep);
}

Status Dsv4Runner::DiscardVerify(RequestState& request) {
  if (auto usable = Usable(request); !usable) {
    return usable;
  }
  if (request.live.verify_rows() == 0) {
    return Error("discarding a DeepSeek verify that does not await Accept");
  }
  // A completed verify: its saves restore every range it wrote (no row
  // kept), then the restore runs now.
  (void)request.live.Settle(true, true, false);
  return Rollback(request);
}

Status Dsv4Runner::Rollback(RequestState& request) {
  if (auto active = CheckResident(request); !active) {
    return active;
  }
  auto rolled = request.live.Rollback(node_, execution_, stream_, resources_.launch(),
                                      "restoring a verify's rejected rows");
  if (!rolled && wave_slots_ > 1) {
    cohort_.CheckFailedJob(node_, stream_, execution_, States());
  }
  if (resources_.launch().faulted() && wave_slots_ > 1) {
    cohort_.Fault(States());
  }
  return rolled;
}

Status Dsv4Runner::Chunk(RequestState& request, std::uint32_t n_past,
                         std::span<const std::int32_t> tokens, std::vector<float>& logits,
                         const std::function<Status()>& meanwhile, Dsv4ChunkKind kind,
                         std::int32_t* token, PrefillHint next, bool want_head) {
  const PlanStep step;  // the plan this step borrows stays until its job ends
  if (token != nullptr && (tokens.size() != 1 || kind != Dsv4ChunkKind::kPlain || speculative() ||
                           model_.exact || !dump_.empty()))
    return Error("a device token needs one plain non-speculative DeepSeek row");
  request.state_refused = false;
  request.track.Wrote(n_past);
  const auto rows = static_cast<std::uint32_t>(tokens.size());
  if (kind != Dsv4ChunkKind::kPlain && !speculative()) {
    return Error("an injection or a verify needs the drafter");
  }
  if (auto usable = Usable(request); !usable) {
    return usable;
  }
  if (auto waiting = request.live.AwaitingAccept(); !waiting) {
    return waiting;
  }
  const Dsv4Model& model = request.model;
  auto in = md::Dsv4Chunk(profile_, layout_, n_past, rows, model.exact, !model.device_raw_masks);
  if (!in) {
    return std::unexpected(in.error());
  }
  const bool verify = kind == Dsv4ChunkKind::kVerify;
  const bool state_only = o_.state_only_prefill && !want_head && !verify && token == nullptr &&
                          !model.exact && dump_.empty();
  md::DsparkInjection inject;
  if (kind != Dsv4ChunkKind::kPlain) {
    inject = md::DsparkInject(dlayout_, n_past, rows);
  }
  // Production prefill returns only its frontier head. Verify/reference
  // retain every requested row; one-row shapes keep their decode key.
  const std::int64_t requested_outputs =
      o_.frontier_head && !model.exact && !o_.full_window && rows > 1 && !verify && dump_.empty()
          ? 1
          : 0;
  auto shape = kg::Dsv4ShapeOf(layout_, *in, requested_outputs);
  shape.token = token != nullptr;
  // A plain or injected prefill chunk may take HCA; a verify never does
  // (PlanDsv4Chunk refuses it).
  const bool hca = !verify && Dsv4PrefillHca(model, shape);
  const ChunkKey key{.shape = shape,
                     .kind = kind,
                     .inject_rows = static_cast<std::int64_t>(inject.cells.size()),
                     .hca = hca,
                     .state_only = state_only};
  // External RE-029 callbacks may page weights in and retain their legacy
  // Submit/Await semantics. Only the ordinary Job path overlaps pure CPU plans.
  const bool predict = !meanwhile && token == nullptr && !verify && dump_.empty() &&
                       node_.threaded() && (o_.prefill_lookahead || o_.prepare_state);
  const auto ahead =
      PredictPrefill(n_past, rows, o_.context, o_.max_rows, predict ? next : PrefillHint{});
  std::array<ChunkKey, 2> keys{};
  std::array<bool, 2> build{};
  for (std::size_t i = 0; i < ahead.size(); ++i) {
    if (ahead[i].rows == 0) continue;
    const auto outputs =
        o_.frontier_head && !model.exact && !o_.full_window && ahead[i].rows > 1 ? 1 : 0;
    auto predicted = Dsv4PrefillShape(layout_, ahead[i].first, ahead[i].rows, outputs);
    if (!predicted) continue;
    keys[i] = {.shape = *predicted,
               .kind = kind,
               .inject_rows = kind == Dsv4ChunkKind::kPlain
                                  ? 0
                                  : std::min<std::int64_t>(ahead[i].rows, dlayout_.ring),
               .hca = Dsv4PrefillHca(model, *predicted),
               .state_only = o_.state_only_prefill && !ahead[i].want_head && !model.exact};
    // Find only: Planned would repatch HCA first-position parameters while the
    // current graph is in flight. Protect every cached near/far plan BEFORE
    // any current allocation or optional charge can invoke cache reclaim.
    const bool duplicate = keys[i] == key || (i != 0 && ahead[0].rows != 0 && keys[i] == keys[0]);
    if (!duplicate) build[i] = request.plans.Find(keys[i]) == nullptr;
  }
  (void)request.plans.Find(key);
  if (auto used = EnsureState(request, n_past + rows); !used) {
    return used;
  }
  if (verify) {
    if (rows > o_.max_verify || !md::Dsv4SameWidths(layout_, n_past, rows)) {
      return Error(std::format("a verify of {} rows at {}: at most {}, at its steps' mask widths",
                               rows, n_past, o_.max_verify));
    }
    if (auto r = PlanSnapshot(request, *in); !r) {
      return r;
    }
  }
  auto planned = Planned(request, key, n_past);
  if (!planned) {
    return std::unexpected(planned.error());
  }
  ChunkPlans::Entry& entry = **planned;
  PlanRuns& runs = entry.runs[0];
  Dsv4Planned* p = entry.planned.get();
  const kg::Dsv4Graph& g = p->graph;
  const std::uint64_t row_bytes = std::uint64_t{profile_.vocab} * sizeof(float);
  const std::uint32_t out_rows = verify ? rows : 1;
  std::uint64_t token_bytes = 0;
  if (token != nullptr) {
    auto bytes = GreedyOutputBytes(g.token, 1, node_.activations().base, node_.activations().bytes);
    if (!bytes) return Error(bytes.error());
    token_bytes = *bytes;
  }
  // Decode graphs (D-090): replay a shape's graph; capture a one-row
  // shape (or a verify's) that has run once launch by launch; otherwise
  // launch by launch. An HCA plan never: its position changes each run.
  bool capture = !hca && runs.CaptureDue(runs_.graphs()) &&
                 ((rows == 1 && kind == Dsv4ChunkKind::kPlain) || verify);
  if (capture && !request.plans.ChargeGraph(entry)) {
    capture = false;  // no room for its graph even after a reclaim: launch by launch
  }
  // The last row's logits (the next token's), or a verify's every row's.
  Copies outputs;
  if (!state_only) {
    outputs.push_back(
        {Address(request.logits),
         token != nullptr
             ? Address(g.token->data)
             : Address(static_cast<const std::byte*>(g.logits->data) +
                       (static_cast<std::uint64_t>(g.logits->ne[1] - out_rows) * row_bytes)),
         token != nullptr ? token_bytes : out_rows * row_bytes});
  }
  kg::LaunchContext& launch = resources_.launch();
  RunPath path = RunPath::kEager;
  Status ran;
  Dsv4HostInputs host;
  // What the job queued, for a failure's settling (Settle).
  bool saved = false;    // this verify's snapshot
  bool wrote = false;    // anything that may write the state
  bool unknown = false;  // a launch of unknown effect
  LiveState& live = request.live;
  auto job = [&](providers::NativeStream native) -> sc::JobResult {
    const auto started = std::chrono::steady_clock::now();
    // A rejected draft's rows first, then this verify's snapshot.
    const bool restoring = live.owed();
    if (auto r = live.QueueOwed(launch); !r) {
      ran = Error(std::format("chunk at {}: {}", n_past, r.error().detail));
      unknown = true;
      return sc::JobResult::kUnknown;
    }
    bool before = restoring;  // work queued before a failure
    if (verify) {
      if (auto r = live.QueueSaves(launch); !r) {
        ran = Error(std::format("chunk at {}: {}", n_past, r.error().detail));
        unknown = true;
        return sc::JobResult::kUnknown;
      }
      before = true;
      saved = true;
    }
    // The embedding rows and the chunk plan's inputs, staged in order.
    if (auto r = BuildDsv4Inputs(model, g, *in, tokens, table(), host, inject.cells); !r) {
      ran = std::unexpected(r.error());
      return before ? sc::JobResult::kFailed : sc::JobResult::kNotStarted;
    }
    auto copies = runs_.Stage(host.sources, verify ? verify_base_ : 0);
    if (!copies) {
      ran = std::unexpected(copies.error());
      return before ? sc::JobResult::kFailed : sc::JobResult::kNotStarted;
    }
    const Queued queued =
        runs_.Queue(runs, *copies, {}, *p->bound, outputs, capture, graph_stats_, native);
    path = queued.path;
    wrote = queued.result.has_value() || queued.before;  // the chunk's own writes
    last_submit_seconds_ = Seconds(std::chrono::steady_clock::now() - started);
    if (!queued.result) {
      ran = Error(std::format("chunk at {}: {}", n_past, queued.result.error().detail));
      if (queued.result.error().error == kg::KernelError::kUnknown) {
        unknown = true;
        return sc::JobResult::kUnknown;
      }
      return before || queued.before ? sc::JobResult::kFailed : sc::JobResult::kNotStarted;
    }
    return sc::JobResult::kQueued;
  };
  PrefillLookaheadGroup<Dsv4Planned, 2> future(node_, prefill_plan_allowance_);
  bool funded = false;
  for (std::size_t i = 0; i < build.size(); ++i) {
    if (!build[i] || !o_.prefill_lookahead) continue;
    ++prefill_stats_.attempted;
    if (!future.Fund(i)) {
      ++prefill_stats_.refused;
      continue;
    }
    funded = true;
  }
  const auto choices = funded ? kg::DeviceChoicesOf(launch) : kg::DeviceChoices{};
  const std::function<void()> cpu = funded ? std::function<void()>([&] {
    const auto built = future.BuildAll(
        [&](std::size_t i) { return BuildPrefillPlan(request, keys[i], ahead[i].first, choices); });
    for (std::size_t i = 0; i < built.size(); ++i) {
      prefill_stats_.built += built[i];
      prefill_stats_.build_seconds += future.seconds(i);
    }
  })
                                           : std::function<void()>{};
  if (o_.prepare_state && ahead[0].rows != 0) {
    // At most two ranges/tensor: both temporary range vectors fit Setup's
    // existing 1 MiB descriptor slack. Prepare funds its owned ticket itself.
    static_assert(4096 * 2 * (2 * sizeof(md::StateRange) + sizeof(LiveState::Range)) < (1U << 20U));
    if (layout_.tensors.size() <= 4096) {
      auto ranges = StateRanges(ahead[0].end());
      if (!ranges) return Error(ranges.error());
      auto prepared = request.live.Prepare(node_, *ranges);
      if (!prepared) return Error(prepared.error());
    }
  }
  Status posted;
  Status alongside;
  if (meanwhile) {
    // Submitted without waiting; the frame (and `job`'s references) lives
    // until Await has seen the program gone.
    sc::ProgramDone done;
    const std::uint64_t submitted =
        node_.Submit(std::make_unique<sc::RunProgram>(done, execution_, std::move(job), stream_));
    alongside = meanwhile();
    posted = node_.Await(done, "a DeepSeek chunk", submitted);
  } else {
    posted = node_.Job(execution_, std::move(job), "a DeepSeek chunk", stream_, cpu);
  }
  const auto prepared = request.live.FinishPreparation();
  if (!posted || !ran || !alongside || launch.faulted()) {
    if (!posted && wave_slots_ > 1) {
      cohort_.CheckFailedJob(node_, stream_, execution_, States());
    }
    // Never left half-written: a verify is undone, anything else that may
    // have written the state quarantines it.
    Settle(request, saved, wrote, unknown);
    if (launch.faulted()) {
      return Error(std::format("chunk at {}: the launch context faulted", n_past));
    }
    if (!ran) {
      return ran;
    }
    return Error(
        std::format("chunk at {}: {}", n_past, !posted ? posted.error() : alongside.error()));
  }
  if (!prepared) {
    if (wrote || unknown) request.live.Quarantine();
    return Error(prepared.error());
  }
  const auto cached = future.InstallAfterCompletion(
      [&](std::size_t i, auto built, double seconds, const auto& transfer) {
        return CachePrefillPlan(request, keys[i], std::move(built), seconds, transfer).has_value();
      });
  for (bool installed : cached) prefill_stats_.cached += installed;
  last_path_ = path;
  last_planned_ = p;
  Count(graph_stats_, path);
  if (verify) {
    live.Verified(rows);
  }
  if (token != nullptr) {
    const auto value = *static_cast<const std::int32_t*>(request.logits);
    if (auto checked = CheckGreedyTokens(std::span(&value, 1), profile_.vocab); !checked) {
      Settle(request, false, true, false);
      return Error(checked.error());
    }
    *token = value;
    ++device_token_outputs_;
  } else if (state_only) {
    logits.clear();
  } else {
    const auto* values = static_cast<const float*>(request.logits);
    logits.assign(values, values + (std::size_t{out_rows} * profile_.vocab));
  }
  prefill_output_stats_.Completed(state_only, g.state_only_tail_cut, rows, g.nodes.size());
  return {};
}

Status Dsv4Runner::Draft(RequestState& request, std::uint32_t pos0, std::int32_t anchor,
                         std::vector<std::int32_t>& drafts) {
  const PlanStep step;  // the plan this step borrows stays until its job ends
  request.state_refused = false;
  request.track.Wrote(pos0 > 0 ? pos0 - 1 : 0);
  if (!speculative()) {
    return Error("drafting needs the drafter");
  }
  if (auto usable = Usable(request); !usable) {
    return usable;
  }
  if (auto waiting = request.live.AwaitingAccept(); !waiting) {
    return waiting;
  }
  auto in =
      md::DsparkBlock(dprofile_, dlayout_, pos0, anchor, o_.draft_rows, !o_.device_draft_masks);
  if (!in) {
    return std::unexpected(in.error());
  }
  draft_mask_host_bytes_ += in->mask.size() * sizeof(std::uint16_t);
  if (auto used = EnsureState(request, pos0); !used) {
    return used;
  }
  auto planned = PlannedDraft(request);
  if (!planned) {
    return std::unexpected(planned.error());
  }
  DraftPlans::Entry& entry = **planned;
  PlanRuns& runs = entry.runs[0];
  DsparkPlanned* p = entry.planned.get();
  const kg::DsparkGraph& g = p->graph;
  const std::array<RunCopy, 1> outputs = {
      RunCopy{Address(request.drafts), Address(g.drafts->data),
              std::uint64_t{o_.draft_rows} * sizeof(std::int32_t)}};
  bool capture = runs.CaptureDue(runs_.graphs());
  if (capture && !request.dplans.ChargeGraph(entry)) {
    capture = false;  // no room for its graph even after a reclaim: launch by launch
  }
  kg::LaunchContext& launch = resources_.launch();
  RunPath path = RunPath::kEager;
  Status ran;
  Dsv4HostInputs host;
  bool unknown = false;  // a launch of unknown effect
  LiveState& live = request.live;
  const Dsv4Model& model = request.model;
  const std::uint64_t staging_at = DraftStagingAt(request);
  auto job = [&](providers::NativeStream native) -> sc::JobResult {
    const auto started = std::chrono::steady_clock::now();
    if (auto r = live.QueueOwed(launch); !r) {
      ran = Error(std::format("draft at {}: {}", pos0, r.error().detail));
      unknown = true;
      return sc::JobResult::kUnknown;
    }
    if (auto r = BuildDsparkInputs(model, request.dmodel, g, *in, table(), host); !r) {
      ran = std::unexpected(r.error());
      return sc::JobResult::kFailed;
    }
    auto copies = runs_.Stage(host.sources, staging_at);
    if (!copies) {
      ran = std::unexpected(copies.error());
      return sc::JobResult::kFailed;
    }
    const Queued queued =
        runs_.Queue(runs, *copies, {}, *p->bound, outputs, capture, draft_stats_, native);
    path = queued.path;
    last_submit_seconds_ = Seconds(std::chrono::steady_clock::now() - started);
    if (!queued.result) {
      ran = Error(std::format("draft at {}: {}", pos0, queued.result.error().detail));
      // A restore may have been queued before a refusal.
      unknown = queued.result.error().error == kg::KernelError::kUnknown;
      return unknown ? sc::JobResult::kUnknown : sc::JobResult::kFailed;
    }
    return sc::JobResult::kQueued;
  };
  auto posted = node_.Job(draft_closure_, std::move(job), "a DSpark draft", stream_);
  if (!posted || !ran || launch.faulted()) {
    if (!posted && wave_slots_ > 1) {
      cohort_.CheckFailedJob(node_, stream_, execution_, States());
    }
    // A draft writes only its own block's ring cells, past the committed
    // positions; a launch of unknown effect quarantines the state.
    Settle(request, false, false, unknown);
    if (launch.faulted()) {
      return Error(std::format("draft at {}: the launch context faulted", pos0));
    }
    return !ran ? ran : Error(std::format("draft at {}: {}", pos0, posted.error()));
  }
  Count(draft_stats_, path);
  const auto* values = static_cast<const std::int32_t*>(request.drafts);
  drafts.assign(values, values + o_.draft_rows);
  return {};
}

std::expected<ggml_tensor*, std::string> Dsv4Runner::DraftRowsNode(std::uint32_t slot,
                                                                   std::uint32_t n,
                                                                   const void* drafts,
                                                                   std::uint64_t staging_at) {
  if (n == 0 || n > o_.draft_rows || slot >= kRequestSlots) {
    return Error("no drafts' rows to look up");
  }
  if (!rows_arena_) {
    auto arena = kg::TensorArena::Create(64 + (kRequestSlots * 3 * 8));
    if (!arena) {
      return Error(arena.error().detail);
    }
    rows_arena_.emplace(std::move(*arena));
  }
  ggml_context* c = rows_arena_->context();
  if (rows_table_ == nullptr) {
    const md::Dsv4Tensor& embedding = binding_.token_embd;
    auto type = kg::GgmlTypeOf(embedding.type);
    if (!type) {
      return Error(type.error().detail);
    }
    rows_table_ = ggml_new_tensor_2d(c, *type, profile_.width, profile_.vocab);
    kg::TensorArena::Bind(rows_table_, weights_.resource_address(embedding.index));
  }
  std::vector<ggml_tensor*>& nodes = draft_rows_[slot];
  if (nodes.empty()) {
    for (std::uint32_t k = 1; k <= o_.draft_rows; ++k) {
      ggml_tensor* ids = ggml_new_tensor_1d(c, GGML_TYPE_I32, k);
      kg::TensorArena::Bind(ids, Address(drafts));
      ggml_tensor* rows = ggml_get_rows(c, rows_table_, ids);
      kg::TensorArena::Bind(rows, Address(runs_.staging()) + staging_at);
      nodes.push_back(rows);
    }
  }
  // A draft planned again (DropPlans) may place its drafts elsewhere, and a
  // wave stages this slot's rows at its own rows.
  ggml_tensor* rows = nodes[n - 1];
  kg::TensorArena::Bind(rows->src[1], Address(drafts));
  kg::TensorArena::Bind(rows, Address(runs_.staging()) + staging_at);
  return rows;
}

Status Dsv4Runner::DraftVerify(RequestState& request, std::uint32_t pos, std::int32_t anchor,
                               std::uint32_t rows, std::vector<std::int32_t>& drafts,
                               std::vector<float>& logits) {
  const PlanStep step;  // the plans this step borrows stay until its job ends
  request.state_refused = false;
  request.track.Wrote(pos > 0 ? pos - 1 : 0);
  if (!speculative()) {
    return Error("drafting needs the drafter");
  }
  if (auto usable = Usable(request); !usable) {
    return usable;
  }
  if (auto waiting = request.live.AwaitingAccept(); !waiting) {
    return waiting;
  }
  if (rows == 0 || rows > o_.max_verify || rows > o_.draft_rows + 1 ||
      !md::Dsv4SameWidths(layout_, pos, rows)) {
    return Error(
        std::format("a verify of {} rows at {}: at most {} and the drafts, at its steps' "
                    "mask widths",
                    rows, pos, o_.max_verify));
  }
  const Dsv4Model& model = request.model;
  // The draft.
  auto block =
      md::DsparkBlock(dprofile_, dlayout_, pos, anchor, o_.draft_rows, !o_.device_draft_masks);
  if (!block) {
    return std::unexpected(block.error());
  }
  draft_mask_host_bytes_ += block->mask.size() * sizeof(std::uint16_t);
  if (auto used = EnsureState(request, pos + rows); !used) {
    return used;
  }
  auto dplanned = PlannedDraft(request);
  if (!dplanned) {
    return std::unexpected(dplanned.error());
  }
  DraftPlans::Entry& dentry = **dplanned;
  PlanRuns& druns = dentry.runs[0];
  const kg::DsparkGraph& dg = dentry.planned->graph;
  bool dcapture = druns.CaptureDue(runs_.graphs());
  // The verify: its tokens the anchor and placeholders the drafts replace
  // on the device.
  auto in = md::Dsv4Chunk(profile_, layout_, pos, rows, model.exact, !model.device_raw_masks);
  if (!in) {
    return std::unexpected(in.error());
  }
  const md::DsparkInjection inject = md::DsparkInject(dlayout_, pos, rows);
  if (auto r = PlanSnapshot(request, *in); !r) {
    return r;
  }
  auto planned = Planned(request, {.shape = kg::Dsv4ShapeOf(layout_, *in),
                                   .kind = Dsv4ChunkKind::kVerify,
                                   .inject_rows = static_cast<std::int64_t>(inject.cells.size())});
  if (!planned) {
    return std::unexpected(planned.error());
  }
  ChunkPlans::Entry& ventry = **planned;
  PlanRuns& vruns = ventry.runs[0];
  const kg::Dsv4Graph& vg = ventry.planned->graph;
  bool vcapture = vruns.CaptureDue(runs_.graphs());
  // Each graph charged before its capture; one with no room even after a
  // reclaim is not made (that plan runs launch by launch).
  dcapture = dcapture && request.dplans.ChargeGraph(dentry);
  vcapture = vcapture && request.plans.ChargeGraph(ventry);
  ggml_tensor* lookup = nullptr;
  if (rows > 1) {
    // The verify's embedding input is staged first: row 0 the anchor's,
    // the drafts' after it.
    auto node = DraftRowsNode(request.slot, rows - 1, dg.drafts->data,
                              verify_base_ + (std::uint64_t{profile_.width} * sizeof(float)));
    if (!node) {
      return std::unexpected(node.error());
    }
    lookup = *node;
  }
  const std::vector<std::int32_t> placeholders(rows, anchor);
  const std::uint64_t row_bytes = std::uint64_t{profile_.vocab} * sizeof(float);
  const std::array<RunCopy, 1> doutputs = {
      RunCopy{Address(request.drafts), Address(dg.drafts->data),
              std::uint64_t{o_.draft_rows} * sizeof(std::int32_t)}};
  const std::array<RunCopy, 1> voutputs = {
      RunCopy{Address(request.logits), Address(vg.logits->data), rows * row_bytes}};
  kg::LaunchContext& launch = resources_.launch();
  std::byte* const staging = runs_.staging();
  const std::uint64_t draft_at = DraftStagingAt(request);
  Queued dq;
  Queued vq;
  Status ran;
  Dsv4HostInputs dhost;
  Dsv4HostInputs vhost;
  LiveState& live = request.live;
  // What the job queued, for a failure's settling (Settle): the verify's
  // snapshot, and a launch of unknown effect. Before the snapshot only the
  // draft's own ring cells, past the committed positions, are written.
  bool saved = false;
  bool unknown_effect = false;
  auto job = [&](providers::NativeStream native) -> sc::JobResult {
    const auto started = std::chrono::steady_clock::now();
    const auto failed = [&](std::string what, bool unknown) {
      ran = Error(std::format("draft and verify at {}: {}", pos, what));
      unknown_effect = unknown_effect || unknown;
      return unknown ? sc::JobResult::kUnknown : sc::JobResult::kFailed;
    };
    if (auto r = live.QueueOwed(launch); !r) {
      return failed(r.error().detail, true);
    }
    // Both stagings first: the host writes them before anything it queues
    // reads them.
    if (auto r = BuildDsparkInputs(model, request.dmodel, dg, *block, table(), dhost); !r) {
      return failed(r.error(), false);
    }
    auto dcopies = runs_.Stage(dhost.sources, draft_at);
    if (!dcopies) {
      return failed(dcopies.error(), false);
    }
    // The draft's staging must end before the verify's begins.
    for (const auto& copy : *dcopies) {
      if (copy[2] + copy[1] > verify_base_) {
        return failed("the draft's inputs reach the verify's staging", false);
      }
    }
    if (auto r = BuildDsv4Inputs(model, vg, *in, placeholders, table(), vhost, inject.cells); !r) {
      return failed(r.error(), false);
    }
    auto vcopies = runs_.Stage(vhost.sources, verify_base_);
    // Staged first the embedding rows, then the tokens: the drafts' rows
    // and ids are written over them below.
    if (!vcopies || vcopies->size() < 2 || (*vcopies)[0][2] != verify_base_ ||
        vhost.sources[0].second != vhost.embd.data() ||
        vhost.sources[1].second != vhost.tokens.data()) {
      return failed(vcopies ? "the verify's staging" : vcopies.error(), false);
    }
    dq = runs_.Queue(druns, *dcopies, {}, *dentry.planned->bound, doutputs, dcapture, draft_stats_,
                     native);
    if (!dq.result) {
      return failed(dq.result.error().detail, dq.result.error().error == kg::KernelError::kUnknown);
    }
    if (rows > 1) {
      // The drafts into the staged tokens after the anchor, and their
      // embedding rows into the staged rows after the anchor's.
      const std::uint64_t tokens_at = (*vcopies)[1][2];
      if (!ClampDrafts(dg.drafts->data, rows - 1, profile_.vocab, native)) {
        return failed("bounding the drafts' ids", true);
      }
      if (!providers::CopyAsync(native, staging + tokens_at + sizeof(std::int32_t), dg.drafts->data,
                                std::uint64_t{rows - 1} * sizeof(std::int32_t),
                                providers::CopyKind::kDeviceToHost)
               .ok()) {
        return failed("the drafts' copy into the verify's tokens", true);
      }
      if (auto r = LookUpRows(launch, lookup); !r) {
        return failed(r.error().detail, r.error().error == kg::KernelError::kUnknown);
      }
    }
    if (auto r = live.QueueSaves(launch); !r) {
      return failed(r.error().detail, true);
    }
    saved = true;
    vq = runs_.Queue(vruns, *vcopies, {}, *ventry.planned->bound, voutputs, vcapture, graph_stats_,
                     native);
    last_submit_seconds_ = Seconds(std::chrono::steady_clock::now() - started);
    if (!vq.result) {
      return failed(vq.result.error().detail, vq.result.error().error == kg::KernelError::kUnknown);
    }
    return sc::JobResult::kQueued;
  };
  auto posted = node_.Job(execution_, std::move(job), "a DSpark draft and its verify", stream_);
  if (!posted || !ran || launch.faulted()) {
    if (!posted && wave_slots_ > 1) {
      cohort_.CheckFailedJob(node_, stream_, execution_, States());
    }
    // Never left half-written: the verify is undone before the next job's
    // work, or the state quarantined.
    Settle(request, saved, false, unknown_effect);
    if (launch.faulted()) {
      return Error(std::format("draft and verify at {}: the launch context faulted", pos));
    }
    return !ran ? ran : Error(std::format("draft and verify at {}: {}", pos, posted.error()));
  }
  Count(draft_stats_, dq.path);
  Count(graph_stats_, vq.path);
  last_path_ = vq.path;
  live.Verified(rows);
  const auto* values = static_cast<const std::int32_t*>(request.drafts);
  drafts.assign(values, values + o_.draft_rows);
  const auto* rows_out = static_cast<const float*>(request.logits);
  logits.assign(rows_out, rows_out + (std::size_t{rows} * profile_.vocab));
  return {};
}

// ------------------------------------------------------------------ waves

Status Dsv4Runner::DecodeWave(std::span<const WaveWork> work) { return Wave(work, false); }

Status Dsv4Runner::DraftVerifyWave(std::span<const WaveWork> work) { return Wave(work, true); }

Status Dsv4Runner::Wave(std::span<const WaveWork> work, bool spec) {
  const PlanStep step;  // the plans this wave borrows stay until its job ends
  if (!waves_provisioned() || work.empty() || work.size() > wave_slots_ ||
      (spec && !speculative()) || model_.exact) {
    return Error("a DeepSeek wave needs provisioned slots (and a drafter to verify)");
  }
  const bool tokens = work.front().token != nullptr;
  if (tokens && spec) return Error("a verify wave cannot publish plain device tokens");
  for (const WaveWork& w : work) {
    if (w.slot != nullptr && &w.slot->owner_ == this) {
      w.slot->request_.track.Wrote(w.pos > 0 ? w.pos - 1 : 0);
    }
  }
  // Each slot's frame; host sources point into it until the job ends.
  struct Frame {
    RequestState* request = nullptr;
    std::uint32_t rows = 0;
    md::Dsv4ChunkInputs in;
    md::DsparkInjection inject;
    std::optional<md::DsparkBlockInputs> block;
    std::vector<std::int32_t> tokens;
    Dsv4HostInputs dhost;
    DraftPlans::Entry* draft = nullptr;  // its own block's (none in a joined draft)
    bool dcapture = false;
    void* drafts = nullptr;  // where its draft block's drafts are, on the device
    ggml_tensor* lookup = nullptr;
    Queued dq;
    bool saved = false;
  };
  std::array<Frame, kRequestSlots> frames;
  // Every slot's draft block as one graph (JoinedDrafts).
  DraftWavePlans::Entry* jdraft = nullptr;
  bool jcapture = false;
  DsparkWaveHostInputs jhost;
  Queued jq;
  WaveKey key;
  key.verify = spec;
  key.shape.token = tokens;
  std::uint32_t previous = 0;
  std::int64_t total = 0;
  for (std::size_t i = 0; i < work.size(); ++i) {
    const WaveWork& w = work[i];
    if (w.slot == nullptr || &w.slot->owner_ != this || !w.slot->request_.provisioned ||
        (i != 0 && w.slot->index() <= previous) ||
        (tokens ? (w.token == nullptr || w.logits != nullptr)
                : (w.logits == nullptr || w.token != nullptr)) ||
        (spec && w.drafts == nullptr)) {
      return Error("a DeepSeek wave needs this runner's slots, ascending, with their outputs");
    }
    for (std::size_t j = 0; j < i; ++j) {
      if ((tokens ? work[j].token == w.token : work[j].logits == w.logits) ||
          (spec && work[j].drafts == w.drafts)) {
        return Error("DeepSeek wave outputs must have independent owners");
      }
    }
    previous = w.slot->index();
    Frame& f = frames[i];
    f.request = &w.slot->request_;
    if (auto usable = Usable(*f.request); !usable) {
      return usable;
    }
    if (auto waiting = f.request->live.AwaitingAccept(); !waiting) {
      return waiting;
    }
    f.rows = spec ? w.rows : 1;
    if (f.rows == 0 || f.rows > o_.max_verify || (spec && f.rows > o_.draft_rows + 1) ||
        !md::Dsv4SameWidths(layout_, w.pos, f.rows)) {
      return Error(std::format("a wave slot's {} rows at {}: at most {}, at its steps' widths",
                               f.rows, w.pos, o_.max_verify));
    }
    total += f.rows;
    if (total > kg::kDsv4WaveRows) {
      return Error("a DeepSeek wave exceeds its rows");
    }
    if (w.anchor < 0 || std::cmp_greater_equal(w.anchor, profile_.vocab)) {
      return Error("a DeepSeek wave anchor is outside the vocabulary");
    }
    auto in = md::Dsv4Chunk(profile_, layout_, w.pos, f.rows, false, !model_.device_raw_masks);
    if (!in) {
      return std::unexpected(in.error());
    }
    f.in = std::move(*in);
    key.shape.slots.push_back(kg::Dsv4ShapeOf(layout_, f.in));
    key.slots[i] = f.request->slot;
    f.tokens.assign(f.rows, w.anchor);
    if (speculative()) {
      // Every step of a model with a drafter injects its rows' features:
      // the ring holds every committed position.
      f.inject = md::DsparkInject(dlayout_, w.pos, f.rows);
      key.shape.inject_rows.push_back(static_cast<std::int64_t>(f.inject.cells.size()));
    }
    if (spec) {
      auto block = md::DsparkBlock(dprofile_, dlayout_, w.pos, w.anchor, o_.draft_rows,
                                   !o_.device_draft_masks);
      if (!block) {
        return std::unexpected(block.error());
      }
      draft_mask_host_bytes_ += block->mask.size() * sizeof(std::uint16_t);
      f.block = std::move(*block);
    }
  }
  const std::size_t count = work.size();
  // No native work yet. Each slot's state through its rows; a refusal here
  // keeps whatever growth completed, protected, and runs nothing.
  for (std::size_t i = 0; i < count; ++i) {
    if (auto used = EnsureState(*frames[i].request, work[i].pos + frames[i].rows); !used) {
      return used;
    }
  }
  auto planned = PlannedWave(key);
  if (!planned) {
    return std::unexpected(planned.error());
  }
  WavePlans::Entry& entry = **planned;
  PlanRuns& run = entry.runs[0];
  Dsv4WavePlanned* p = entry.planned.get();
  const kg::Dsv4WaveGraph& g = p->graph;
  if (g.slots.size() != count || g.joined.logits == nullptr ||
      std::cmp_not_equal(g.joined.logits->ne[1], total)) {
    return Error("the DeepSeek wave plan is not this wave's");
  }
  std::array<Dsv4WaveSlotInputs, kRequestSlots> slot_inputs{};
  for (std::size_t i = 0; i < count; ++i) {
    slot_inputs[i] = {
        .chunk = &frames[i].in, .tokens = frames[i].tokens, .inject_cells = frames[i].inject.cells};
  }
  Dsv4WaveHostInputs host;
  if (auto built =
          BuildDsv4WaveInputs(model_, g, std::span(slot_inputs).first(count), table(), host);
      !built) {
    return built;
  }
  auto copies = runs_.Stage(host.sources, verify_base_);
  if (!copies || copies->size() < 2 || (*copies)[0][2] != verify_base_ ||
      host.sources[0].second != host.embd.data() || host.sources[1].second != host.tokens.data()) {
    return Error(copies ? "the wave's staging" : copies.error());
  }
  const std::uint64_t embd_at = (*copies)[0][2];
  const std::uint64_t tokens_at = (*copies)[1][2];
  const std::uint64_t width_bytes = std::uint64_t{profile_.width} * sizeof(float);
  const bool joined = spec && JoinedDrafts(count);
  if (spec) {
    if (joined) {
      // Every slot's block in one graph: each slot's drafts equal its own
      // block's (dsv4_graph.h BuildDsparkWaveGraph).
      DraftWaveKey dkey{.count = static_cast<std::uint32_t>(count)};
      for (std::size_t i = 0; i < count; ++i) {
        dkey.slots[i] = frames[i].request->slot;
      }
      auto dplanned = PlannedDraftWave(dkey);
      if (!dplanned) {
        return std::unexpected(dplanned.error());
      }
      jdraft = *dplanned;
      jcapture = jdraft->runs[0].CaptureDue(runs_.graphs());
    }
    for (std::size_t i = 0; i < count; ++i) {
      Frame& f = frames[i];
      if (joined) {
        f.drafts = jdraft->planned->graph.drafts[i]->data;
      } else {
        auto dplanned = PlannedDraft(*f.request);
        if (!dplanned) {
          return std::unexpected(dplanned.error());
        }
        f.draft = *dplanned;
        f.dcapture = f.draft->runs[0].CaptureDue(runs_.graphs());
        f.drafts = f.draft->planned->graph.drafts->data;
      }
      if (f.rows > 1) {
        const auto first = static_cast<std::uint64_t>(g.first[i]) + 1;
        auto node =
            DraftRowsNode(f.request->slot, f.rows - 1, f.drafts, embd_at + (first * width_bytes));
        if (!node) {
          return std::unexpected(node.error());
        }
        f.lookup = *node;
      }
    }
    // Every slot's saves, after its draft: what its verify rows write.
    for (std::size_t i = 0; i < count; ++i) {
      if (auto r = PlanSnapshot(*frames[i].request, frames[i].in); !r) {
        return r;
      }
    }
  }
  // Each graph charged before its capture; one with no room even after a
  // reclaim is not made (that plan runs launch by launch).
  const bool capture = run.CaptureDue(runs_.graphs()) && waves_.ChargeGraph(entry);
  jcapture = jcapture && dwaves_.ChargeGraph(*jdraft);
  for (std::size_t i = 0; i < count && !joined; ++i) {
    frames[i].dcapture =
        frames[i].dcapture && frames[i].request->dplans.ChargeGraph(*frames[i].draft);
  }
  std::uint64_t token_bytes = 0;
  if (tokens) {
    auto bytes = GreedyOutputBytes(g.joined.token, static_cast<std::uint32_t>(count),
                                   node_.activations().base, node_.activations().bytes);
    if (!bytes) return Error(bytes.error());
    token_bytes = *bytes;
  }
  const std::array<RunCopy, 1> outputs = {RunCopy{
      Address(wave_logits_),
      tokens ? Address(g.joined.token->data) : Address(g.joined.logits->data),
      tokens ? token_bytes : static_cast<std::uint64_t>(total) * profile_.vocab * sizeof(float)}};
  kg::LaunchContext& launch = resources_.launch();
  std::byte* const staging = runs_.staging();
  Status ran;
  Queued wq;
  bool unknown_effect = false;
  bool wrote = false;
  auto job = [&](providers::NativeStream native) -> sc::JobResult {
    const auto started = std::chrono::steady_clock::now();
    const auto failed = [&](std::string what, bool unknown) {
      ran = Error(std::format("a DeepSeek wave: {}", what));
      unknown_effect = unknown_effect || unknown;
      return unknown ? sc::JobResult::kUnknown : sc::JobResult::kFailed;
    };
    for (std::size_t i = 0; i < count; ++i) {
      if (auto r = frames[i].request->live.QueueOwed(launch); !r) {
        return failed(r.error().detail, true);
      }
    }
    if (spec) {
      if (joined) {
        const DsparkWavePlanned& dp = *jdraft->planned;
        std::array<const md::DsparkBlockInputs*, kRequestSlots> blocks{};
        std::array<RunCopy, kRequestSlots> doutputs{};
        for (std::size_t i = 0; i < count; ++i) {
          blocks[i] = &*frames[i].block;
          doutputs[i] =
              RunCopy{Address(frames[i].request->drafts), Address(dp.graph.drafts[i]->data),
                      std::uint64_t{o_.draft_rows} * sizeof(std::int32_t)};
        }
        if (auto r =
                BuildDsparkWaveInputs(frames[0].request->model, frames[0].request->dmodel, dp.graph,
                                      std::span(blocks).first(count), table(), jhost);
            !r) {
          return failed(r.error(), false);
        }
        // From the staging's start, below the verify's (the blocks' own
        // places, which no block of this wave uses).
        auto dcopies = runs_.Stage(jhost.sources, 0);
        if (!dcopies) {
          return failed(dcopies.error(), false);
        }
        for (const auto& copy : *dcopies) {
          if (copy[2] + copy[1] > verify_base_) {
            return failed("a joined draft's inputs reach past its staging", false);
          }
        }
        jq = runs_.Queue(jdraft->runs[0], *dcopies, {}, *jdraft->planned->bound,
                         std::span(doutputs).first(count), jcapture, draft_stats_, native);
        if (!jq.result) {
          return failed(jq.result.error().detail,
                        jq.result.error().error == kg::KernelError::kUnknown);
        }
      }
      for (std::size_t i = 0; i < count; ++i) {
        Frame& f = frames[i];
        if (!joined) {
          const DsparkPlanned& dp = *f.draft->planned;
          if (auto r = BuildDsparkInputs(f.request->model, f.request->dmodel, dp.graph, *f.block,
                                         table(), f.dhost);
              !r) {
            return failed(r.error(), false);
          }
          auto dcopies = runs_.Stage(f.dhost.sources, DraftStagingAt(*f.request));
          if (!dcopies) {
            return failed(dcopies.error(), false);
          }
          for (const auto& copy : *dcopies) {
            if (copy[2] + copy[1] > DraftStagingAt(*f.request) + draft_staging_ ||
                copy[2] + copy[1] > verify_base_) {
              return failed("a draft's inputs reach past its staging", false);
            }
          }
          const std::array<RunCopy, 1> doutputs = {
              RunCopy{Address(f.request->drafts), Address(dp.graph.drafts->data),
                      std::uint64_t{o_.draft_rows} * sizeof(std::int32_t)}};
          f.dq = runs_.Queue(f.draft->runs[0], *dcopies, {}, *f.draft->planned->bound, doutputs,
                             f.dcapture, draft_stats_, native);
          if (!f.dq.result) {
            return failed(f.dq.result.error().detail,
                          f.dq.result.error().error == kg::KernelError::kUnknown);
          }
        }
        if (f.rows > 1) {
          // This slot's drafts into its staged tokens after its anchor, and
          // their embedding rows into its staged rows after the anchor's.
          const auto first = static_cast<std::uint64_t>(g.first[i]) + 1;
          if (!ClampDrafts(f.drafts, f.rows - 1, profile_.vocab, native)) {
            return failed("bounding the drafts' ids", true);
          }
          if (!providers::CopyAsync(native, staging + tokens_at + (first * sizeof(std::int32_t)),
                                    f.drafts, std::uint64_t{f.rows - 1} * sizeof(std::int32_t),
                                    providers::CopyKind::kDeviceToHost)
                   .ok()) {
            return failed("the drafts' copy into the wave's tokens", true);
          }
          if (auto r = LookUpRows(launch, f.lookup); !r) {
            return failed(r.error().detail, r.error().error == kg::KernelError::kUnknown);
          }
        }
      }
      for (std::size_t i = 0; i < count; ++i) {
        if (auto r = frames[i].request->live.QueueSaves(launch); !r) {
          return failed(r.error().detail, true);
        }
        frames[i].saved = true;
      }
    }
    wq = runs_.Queue(run, *copies, {}, *p->bound, outputs, capture, wave_stats_, native);
    wrote = wq.before || wq.result.has_value();
    last_submit_seconds_ = Seconds(std::chrono::steady_clock::now() - started);
    if (!wq.result) {
      return failed(wq.result.error().detail, wq.result.error().error == kg::KernelError::kUnknown);
    }
    return sc::JobResult::kQueued;
  };
  auto posted =
      node_.Job(execution_, std::move(job),
                spec ? "a DeepSeek draft and verify wave" : "a DeepSeek decode wave", stream_);
  if (!posted || !ran || launch.faulted()) {
    if (!posted) {
      cohort_.CheckFailedJob(node_, stream_, execution_, States());
    }
    for (std::size_t i = 0; i < count; ++i) {
      // A verify's snapshot undoes it; a decode wave's writes quarantine.
      Settle(*frames[i].request, frames[i].saved, !spec && wrote, unknown_effect);
    }
    if (launch.faulted()) {
      return Error("a DeepSeek wave: the launch context faulted");
    }
    return !ran ? ran : Error(std::format("a DeepSeek wave: {}", posted.error()));
  }
  Count(graph_stats_, wq.path);
  Count(wave_stats_, wq.path);
  last_path_ = wq.path;
  if (joined) {
    Count(draft_stats_, jq.path);
  }
  const auto* token_ids = reinterpret_cast<const std::int32_t*>(wave_logits_);
  if (tokens) {
    if (auto checked = CheckGreedyTokens(std::span(token_ids, count), profile_.vocab); !checked) {
      for (std::size_t j = 0; j < count; ++j) Settle(*frames[j].request, false, true, false);
      return Error(checked.error());
    }
  }
  if (tokens) device_token_outputs_ += count;
  for (std::size_t i = 0; i < count; ++i) {
    Frame& f = frames[i];
    const WaveWork& w = work[i];
    if (tokens) {
      *w.token = token_ids[i];
      continue;
    }
    const float* from = wave_logits_ + (static_cast<std::size_t>(g.first[i]) * profile_.vocab);
    w.logits->assign(from, from + (std::size_t{f.rows} * profile_.vocab));
    if (spec) {
      if (!joined) {
        Count(draft_stats_, f.dq.path);
      }
      f.request->live.Verified(f.rows);
      const auto* values = static_cast<const std::int32_t*>(f.request->drafts);
      w.drafts->assign(values, values + o_.draft_rows);
    }
  }
  return {};
}

// ------------------------------------------------------------------ checks

std::expected<std::uint64_t, std::string> Dsv4Runner::CheckDeviceEmbedding() {
  constexpr std::uint32_t kBatch = 2048;
  const md::Dsv4Tensor& embedding = binding_.token_embd;
  auto type = kg::GgmlTypeOf(embedding.type);
  if (!type) {
    return Error(type.error().detail);
  }
  const std::uint64_t width = profile_.width;
  const std::uint64_t out_bytes = kBatch * width * sizeof(float);
  const std::uint64_t ids_at = Round(out_bytes, 256);
  if (ids_at + (kBatch * sizeof(std::int32_t)) > node_.activations().bytes ||
      kBatch * sizeof(std::int32_t) > runs_.staging_bytes()) {
    return Error("the lookups' batch does not fit the activations or the staging");
  }
  if (live_.total_bytes() < out_bytes) {
    return Error("the state's host copy is shorter than a batch of rows");
  }
  // The state's host copy holds each batch's rows.
  void* const host = live_.HostCopy(node_, out_bytes);
  if (host == nullptr) {
    return Error("pinned host memory for the lookups' rows");
  }
  auto arena = kg::TensorArena::Create(8);
  if (!arena) {
    return Error(arena.error().detail);
  }
  ggml_context* c = arena->context();
  ggml_tensor* table_rows = ggml_new_tensor_2d(c, *type, profile_.width, profile_.vocab);
  kg::TensorArena::Bind(table_rows, weights_.resource_address(embedding.index));
  ggml_tensor* ids = ggml_new_tensor_1d(c, GGML_TYPE_I32, kBatch);
  kg::TensorArena::Bind(ids, node_.activations().base + ids_at);
  ggml_tensor* rows = ggml_get_rows(c, table_rows, ids);
  kg::TensorArena::Bind(rows, node_.activations().base);
  std::byte* const staging = runs_.staging();
  kg::LaunchContext& launch = resources_.launch();
  std::vector<std::int32_t> batch(kBatch);
  std::vector<float> want;
  std::uint64_t checked = 0;
  for (std::uint32_t first = 0; first < profile_.vocab; first += kBatch) {
    for (std::uint32_t i = 0; i < kBatch; ++i) {
      batch[i] = static_cast<std::int32_t>(std::min(first + i, profile_.vocab - 1));
    }
    std::memcpy(staging, batch.data(), kBatch * sizeof(std::int32_t));
    std::string failed;
    auto posted = node_.Job(
        execution_,
        [&](providers::NativeStream native) {
          if (!providers::CopyAsync(native, ids->data, staging, kBatch * sizeof(std::int32_t),
                                    providers::CopyKind::kHostToDevice)
                   .ok()) {
            return sc::JobResult::kUnknown;
          }
          if (auto r = LookUpRows(launch, rows); !r) {
            failed = r.error().detail;
            return sc::JobResult::kFailed;
          }
          return providers::CopyAsync(native, host, rows->data, out_bytes,
                                      providers::CopyKind::kDeviceToHost)
                         .ok()
                     ? sc::JobResult::kQueued
                     : sc::JobResult::kUnknown;
        },
        "looking up embedding rows", stream_);
    if (!posted || !failed.empty()) {
      if (!posted) {
        live_.HostCopyUnproven();
      }
      return Error(failed.empty() ? posted.error() : failed);
    }
    // The host's lookup, as every chunk makes it.
    if (auto r = Dsv4EmbeddingRows(model_, batch, table(), want); !r) {
      return std::unexpected(r.error());
    }
    if (std::memcmp(host, want.data(), out_bytes) != 0) {
      return Error(
          std::format("an embedding row of tokens {} to {} differs between the device's "
                      "lookup and the host's",
                      first, first + kBatch - 1));
    }
    checked += std::min<std::uint64_t>(kBatch, profile_.vocab - first);
  }
  return checked;
}

Status Dsv4Runner::SaveUsedState(RequestState& request, void* host,
                                 std::span<const LiveState::Range> ranges) {
  if (auto active = CheckResident(request); !active) {
    return active;
  }
  LiveState::CopyRetirement retirement = LiveState::CopyRetirement::kProven;
  auto copied = request.live.Copy(node_, request.fence, stream_, host, ranges, true, &retirement);
  if (retirement == LiveState::CopyRetirement::kUnproven && wave_slots_ > 1) {
    cohort_.Fault(States());
  }
  return copied;
}

Status Dsv4Runner::RestoreUsedState(RequestState& request, void* host,
                                    std::span<const LiveState::Range> ranges) {
  if (host == nullptr &&
      std::ranges::any_of(ranges, [](const LiveState::Range& r) { return r.bytes != 0; })) {
    return Error("the conversation snapshot has no source buffer");
  }
  request.track.Lost();  // a diagnostic restore replaces the state whole
  if (auto prepared = PrepareRestoreState(request, ranges); !prepared) {
    return prepared;
  }
  return CopyCheckpointState(request, host, ranges, false);
}

Status Dsv4Runner::PrepareRestoreState(RequestState& request,
                                       std::span<const LiveState::Range> ranges) {
  if (auto active = CheckResident(request); !active) {
    return active;
  }
  // The held request without this slot's state, which Retain needs
  // unleased; every other active slot stays protected.
  const SlotMask others = cohort_.active() & ~(SlotMask{1} << request.slot);
  if (auto protected_others = RefreshClosures(others); !protected_others) {
    return protected_others;
  }
  auto prepared = [&]() -> Status {
    if (auto used = request.live.Use(node_, ranges, &execution_); !used) {
      return std::unexpected(used.error());
    }
    return request.live.Retain(node_, ranges);
  }();
  if (!prepared) {
    request.live.Quarantine();
  }
  // Even refused or partial growth belongs to this slot and joins both the
  // retained-state union and the held request before returning.
  if (auto refreshed = RefreshClosures(); !refreshed) {
    return refreshed;
  }
  return prepared;
}

Status Dsv4Runner::CopyCheckpointState(RequestState& request, void* host,
                                       std::span<const LiveState::Range> ranges, bool to_host) {
  if (auto active = CheckResident(request); !active) {
    return active;
  }
  if (!to_host) {
    request.track.Wrote(ranges);  // a turn checkpoint's pages copied in
  }
  LiveState::CopyRetirement retirement = LiveState::CopyRetirement::kProven;
  auto copied =
      request.live.Copy(node_, request.fence, stream_, host, ranges, to_host, &retirement);
  if (!copied) {
    request.live.Quarantine();
  }
  if (retirement == LiveState::CopyRetirement::kUnproven && wave_slots_ > 1) {
    cohort_.Fault(States());
  }
  return copied;
}

std::expected<std::vector<LiveState::Range>, std::string> Dsv4Runner::CheckpointRanges(
    const RequestState& request, std::uint32_t positions) const {
  if (auto active = CheckActive(request); !active) {
    return std::unexpected(active.error());
  }
  if (auto usable = request.live.Usable(); !usable) {
    return std::unexpected(usable.error());
  }
  if (auto settled = request.live.AwaitingAccept(); !settled) {
    return std::unexpected(settled.error());
  }
  if (request.live.owed()) {
    return Error("checkpoint has an unsettled DeepSeek verify");
  }
  auto writes = StateWrites(positions);
  if (!writes) {
    return std::unexpected(writes.error());
  }
  return CheckpointPages(request.live.used_ranges(), *writes);
}

std::expected<std::vector<LiveState::Range>, std::string> Dsv4Runner::StateWrites(
    std::uint32_t positions) const {
  auto mutable_bytes = md::Dsv4CheckpointWrites(layout_, positions);
  if (!mutable_bytes) {
    return std::unexpected(mutable_bytes.error());
  }
  std::vector<LiveState::Range> writes;
  for (const auto& range : *mutable_bytes) {
    writes.push_back({.region = kTarget, .offset = range.offset, .bytes = range.bytes});
  }
  if (speculative()) {
    writes.push_back({.region = kDrafter, .offset = 0, .bytes = dlayout_.bytes});
  }
  return writes;
}

Status Dsv4Runner::ReadState(RequestState& request, std::vector<std::byte>& target,
                             std::vector<std::byte>& drafter) {
  if (auto usable = Usable(request); !usable) {
    return usable;
  }
  if (request.live.owed() || request.live.verify_rows() != 0) {
    return Error("reading the state with a verify's rollback pending");
  }
  const std::array<std::vector<std::byte>*, 2> out = {&target, &drafter};
  LiveState::CopyRetirement retirement = LiveState::CopyRetirement::kProven;
  auto read = request.live.Read(node_, request.fence, stream_, "reading the DeepSeek state", out,
                                &retirement);
  if (retirement == LiveState::CopyRetirement::kUnproven && wave_slots_ > 1) {
    cohort_.Fault(States());
  }
  return read;
}

Status Dsv4Runner::DumpLast(std::vector<Dumped>& out) {
  out.clear();
  if (last_planned_ == nullptr || dump_.empty()) {
    return Error("no chunk has run with a dump since the plans were dropped");
  }
  struct Read {
    const void* from = nullptr;
    std::uint64_t at = 0;
    std::uint64_t bytes = 0;
  };
  std::vector<Read> reads;
  std::uint64_t total = 0;
  for (const auto& [name, t] : last_planned_->graph.named) {
    if (t->data == nullptr || !ggml_is_contiguous(t)) {
      continue;
    }
    Dumped& d = out.emplace_back();
    d.name = name;
    d.type = t->type;
    d.ne = {t->ne[0], t->ne[1], t->ne[2], t->ne[3]};
    reads.push_back({.from = t->data, .at = total, .bytes = ggml_nbytes(t)});
    total += Round(ggml_nbytes(t), 256);
  }
  auto pinned = providers::AllocatePinned(std::max<std::uint64_t>(total, 256));
  if (!pinned) {
    return Error("pinned host memory for a dump");
  }
  void* const host = *pinned;
  auto posted = node_.Job(
      execution_,
      [&](providers::NativeStream stream) {
        for (const Read& r : reads) {
          if (!providers::CopyAsync(stream, static_cast<std::byte*>(host) + r.at, r.from, r.bytes,
                                    providers::CopyKind::kDeviceToHost)
                   .ok()) {
            return sc::JobResult::kUnknown;
          }
        }
        return sc::JobResult::kQueued;
      },
      "reading a DeepSeek chunk's dump", stream_);
  if (!posted) {
    // A failed job need not prove that DMA retired. This diagnostic's
    // allocation stays pinned until process exit, as unproven state-copy
    // destinations do; never free memory a copy may still write.
    return posted;
  }
  for (std::size_t i = 0; i < reads.size(); ++i) {
    const auto* bytes = static_cast<const std::byte*>(host) + reads[i].at;
    out[i].bytes.assign(bytes, bytes + reads[i].bytes);
  }
  providers::FreePinned(host);
  return posted;
}

std::expected<double, std::string> Dsv4Runner::TimeReplays(std::uint32_t n_past, std::int32_t token,
                                                           std::uint32_t count) {
  const PlanStep step;  // its plan stays while this runs
  default_request_.track.Wrote(n_past);
  // The replays write the state at n_past as a step there does: usable,
  // resident, settled and backed through it, like any step's.
  if (auto usable = Usable(default_request_); !usable) {
    return std::unexpected(usable.error());
  }
  if (auto waiting = default_request_.live.AwaitingAccept(); !waiting) {
    return std::unexpected(waiting.error());
  }
  if (auto settled = Rollback(default_request_); !settled) {  // a restore owed runs first
    return std::unexpected(settled.error());
  }
  auto in = md::Dsv4Chunk(profile_, layout_, n_past, 1, model_.exact, !model_.device_raw_masks);
  if (!in) {
    return std::unexpected(in.error());
  }
  if (auto used = EnsureState(default_request_, n_past + 1); !used) {
    return std::unexpected(used.error());
  }
  auto planned = Planned(default_request_, {.shape = kg::Dsv4ShapeOf(layout_, *in)});
  if (!planned) {
    return std::unexpected(planned.error());
  }
  ChunkPlans::Entry& entry = **planned;
  const PlanRuns& runs = entry.runs[0];
  if (!runs.graph) {
    return Error("the step's shape has no graph yet");
  }
  const kg::CapturedGraph& graph = *runs.graph;
  Dsv4HostInputs host;
  if (auto r =
          BuildDsv4Inputs(model_, entry.planned->graph, *in, std::span(&token, 1), table(), host);
      !r) {
    return std::unexpected(r.error());
  }
  auto copies = runs_.Stage(host.sources, 0);
  if (!copies) {
    return std::unexpected(copies.error());
  }
  if (*copies != runs.copies) {
    return Error("the inputs' staging differs from the captured graph's");
  }
  kg::LaunchContext& launch = resources_.launch();
  std::string failed;
  bool wrote = false;    // a replay was queued: the state at n_past written
  bool unknown = false;  // a launch of unknown effect
  const auto start = std::chrono::steady_clock::now();
  auto posted = node_.Job(
      execution_,
      [&](providers::NativeStream) {
        for (std::uint32_t i = 0; i < count; ++i) {
          if (auto r = launch.Launch(graph); !r) {
            failed = r.error().detail;
            if (r.error().error == kg::KernelError::kUnknown) {
              unknown = true;
              return sc::JobResult::kUnknown;
            }
            return i == 0 ? sc::JobResult::kNotStarted : sc::JobResult::kFailed;
          }
          wrote = true;
        }
        return sc::JobResult::kQueued;
      },
      "back-to-back replays", stream_);
  const double seconds = Seconds(std::chrono::steady_clock::now() - start);
  if (!posted || !failed.empty()) {
    // Replays that ran wrote the state: never left half-written.
    Settle(default_request_, false, wrote, unknown);
    return Error(failed.empty() ? posted.error() : failed);
  }
  return seconds / count;
}

Status Dsv4Runner::Release() {
  if (released_) {
    return {};
  }
  released_ = true;
  std::vector<std::string> problems;
  // The plans and their graphs first: they name the launch context and
  // the memory below (D-090).
  DropPlans();
  resources_.Release(problems);
  auto& memory = node_.memory();
  for (RequestState* request : Requests()) {
    if (request->provisioned) {
      request->live.Release(memory, problems);
    }
  }
  for (PagedWeights* part : {&weights_, &dweights_}) {
    if (auto r = part->Release(memory); !r) {
      problems.push_back(std::format("DeepSeek: {}", r.error()));
    }
  }
  return support::Joined(problems);
}

}  // namespace llmp::engine
