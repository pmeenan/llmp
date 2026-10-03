// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "engine/qwen38_runner.h"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstring>
#include <expected>
#include <format>
#include <initializer_list>
#include <limits>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

#include "artifact/layout.h"
#include "engine/checkpoint_file.h"
#include "engine/support.h"
#include "ggml.h"
#include "kernels/ggml/dsv4_graph.h"
#include "kernels/ggml/executor.h"
#include "kernels/ggml/graph_plan.h"
#include "kernels/paging/paging.h"
#include "providers/device_runtime.h"
#include "scheduler/commands.h"
#include "scheduler/scheduler.h"

namespace jitllm::engine {

namespace {

namespace kg = jitllm::kernels::ggml;
namespace md = jitllm::model;
namespace sc = jitllm::scheduler;
using catalog::ExtentId;
using catalog::MemoryClass;
using support::Address;
using support::Error;
using support::Pointer;
using support::Round;
using support::Seconds;

constexpr std::uint64_t kExtent = kPagedExtent;
constexpr std::size_t kRingDepth = 32;
// The slabs' offset in their first page: the stride's own alignment (16),
// since the 80-byte gap between Qwen3.8's expert groups cannot hold 256.
constexpr std::uint64_t kSlabAlignment = 16;
// The most cell ranges a verify saves: 3 a QSA layer a row, at most 8 rows.
constexpr std::uint32_t kRangeCapacity = 512;
// Where a verify's argmaxes land in the drafts' pinned buffer (I32s).
constexpr std::size_t kArgmaxAt = 64;
// And a draft's probabilities (F32 bits), after its drafts (at most 8).
constexpr std::size_t kProbabilityAt = 32;

// The slab of a layer's expert arrays (paged_weights.h ExpertSlab).
std::expected<SlabSpec, std::string> SlabOf(const artifact::Artifact& artifact,
                                            const md::Qwen38Layer& l, bool cutlass,
                                            std::uint32_t count, std::uint32_t il) {
  std::vector<std::pair<std::uint32_t, std::string_view>> arrays;
  for (const md::Qwen38Tensor* t : l.expert_arrays(cutlass)) {
    arrays.emplace_back(t->index, t->type);
  }
  return ExpertSlab(artifact, arrays, count, kSlabAlignment, il);
}

// The cells a wave's attention reads are aligned more coarsely than a
// lone request's: a wave's graph keys every slot's shape, so with several
// slots at different depths a 256-cell alignment changes some slot's shape
// every few waves and the plans are never replayed. Cells past a row's
// position are masked or hidden from selection by position: no result
// changes (model/qwen38.h Qwen38Chunk). They are still read (the vector
// attention and the block scores read every cell), so a wave's EnsureState
// backs the caches through the same alignment (Qwen38Runner::WaveReadAlign:
// Qwen38Options::wave_read_align, 2,048 cells by default).
constexpr std::uint32_t kLoneReadAlign = 256;

}  // namespace

Qwen38Runner::~Qwen38Runner() = default;

std::expected<Qwen38Runner::Slot*, std::string> Qwen38Runner::request_slot(std::size_t index) {
  if (index >= slot_count_ || released_) {
    return Error("a Qwen3.8 request slot outside the live runner");
  }
  return &request_slots_[index];
}

void Qwen38Runner::FaultCohort() {
  cohort_faulted_ = true;
  for (RequestState* request : Requests()) {
    request->live.Quarantine();
  }
}

void Qwen38Runner::CheckFailedJob() {
  auto healthy = node_.Call(
      [&]() -> Status {
        if (node_.scheduler().fault()) {
          return Error("the Qwen3.8 shared scheduler faulted");
        }
        for (const auto& [id, generation] : execution_.extents) {
          const auto view = node_.catalog().Describe(id);
          if (!view || view->content_generation != generation ||
              view->state == catalog::ExtentState::kQuarantined) {
            return Error("the Qwen3.8 active closure is no longer usable");
          }
        }
        return {};
      },
      "checking Qwen3.8's failed shared job");
  if (!healthy || (std::popcount(active_mask_) > 1 && !node_.InRequest(stream_))) {
    FaultCohort();
  }
}

Status Qwen38Runner::CheckActive(const RequestState& request) const {
  if (released_ || cohort_faulted_) {
    return Error("the Qwen3.8 cohort requires retirement");
  }
  if ((active_mask_ & (SlotMask{1} << request.slot)) == 0) {
    return Error("the Qwen3.8 request slot is not active");
  }
  if (std::popcount(active_mask_) > 1 && !node_.InRequest(stream_)) {
    return Error("active Qwen3.8 slots require one held stream request");
  }
  return {};
}

Status Qwen38Runner::SelectSlots(std::span<Slot* const> active) {
  if (released_ || cohort_faulted_) {
    return Error("the Qwen3.8 cohort requires retirement");
  }
  if (active.size() > slot_count_) {
    return Error(std::format("at most {} Qwen3.8 request slots", slot_count_));
  }
  SlotMask mask = 0;
  for (const Slot* slot : active) {
    if (slot == nullptr || &slot->owner_ != this || slot->index() >= slot_count_) {
      return Error("a Qwen3.8 slot belongs to another runner or is not provisioned");
    }
    const SlotMask bit = SlotMask{1} << slot->index();
    if ((mask & bit) != 0) {
      return Error("a Qwen3.8 request slot occurs twice");
    }
    mask |= bit;
  }
  // Unchanged selection within an open request
  // keeps its closure. State-growth, clear and restore paths still refresh.
  if (mask == active_mask_ && node_.InRequest(stream_)) {
    return {};
  }
  active_mask_ = mask;
  return RefreshClosures();
}

void Qwen38Runner::DropPlans() {
  target_waves_.Clear();
  draft_waves_.Clear();
  for (RequestState* request : Requests()) {
    request->plans.Clear();
    request->mplans.Clear();
  }
}

std::size_t Qwen38Runner::plans() const {
  std::size_t count = target_waves_.size() + draft_waves_.size();
  for (const RequestState* request : Requests()) {
    count += request->plans.size();
  }
  return count;
}

std::size_t Qwen38Runner::graphs() const {
  std::size_t count = target_waves_.graphs() + draft_waves_.graphs();
  for (const RequestState* request : Requests()) {
    count += request->plans.graphs() + request->mplans.graphs();
  }
  return count;
}

std::uint64_t Qwen38Runner::cached_plan_bytes() const {
  std::uint64_t bytes = target_waves_.host_bytes() + draft_waves_.host_bytes();
  for (const RequestState* request : Requests()) {
    bytes += request->plans.host_bytes() + request->mplans.host_bytes();
  }
  return bytes;
}

std::uint64_t Qwen38Runner::cached_graph_bytes() const {
  std::uint64_t bytes = 0;
  for (const PlanCacheBase* cache : PlanCaches()) {
    bytes += cache->graph_bytes();
  }
  return bytes;
}

std::uint64_t Qwen38Runner::graph_measured_bytes() const {
  std::uint64_t bytes = 0;
  for (const PlanCacheBase* cache : PlanCaches()) {
    bytes += cache->graph_measured_bytes();
  }
  return bytes;
}

std::array<PlanCacheBase*, (2 * Qwen38Runner::kRequestSlots) + 2> Qwen38Runner::PlanCaches() {
  std::array<PlanCacheBase*, (2 * kRequestSlots) + 2> caches{};
  for (RequestState* request : requests_) {
    caches[std::size_t{2} * request->slot] = &request->plans;
    caches[(std::size_t{2} * request->slot) + 1] = &request->mplans;
  }
  caches[2 * kRequestSlots] = &target_waves_;
  caches[(2 * kRequestSlots) + 1] = &draft_waves_;
  return caches;
}

std::array<const PlanCacheBase*, (2 * Qwen38Runner::kRequestSlots) + 2> Qwen38Runner::PlanCaches()
    const {
  std::array<const PlanCacheBase*, (2 * kRequestSlots) + 2> caches{};
  for (const RequestState* request : const_requests_) {
    caches[std::size_t{2} * request->slot] = &request->plans;
    caches[(std::size_t{2} * request->slot) + 1] = &request->mplans;
  }
  caches[2 * kRequestSlots] = &target_waves_;
  caches[(2 * kRequestSlots) + 1] = &draft_waves_;
  return caches;
}

void Qwen38Runner::ReclaimCandidates(std::uint32_t owner, bool running,
                                     std::vector<memory::ReclaimCandidate>& out) {
  if (released_ || cohort_faulted_) {
    return;  // a faulted cohort keeps its owners until retirement
  }
  auto caches = PlanCaches();
  CollectPlans(caches, owner, running, out);
}

std::uint64_t Qwen38Runner::Reclaim(memory::ReclaimKind kind, std::uint64_t id) {
  if (released_ || cohort_faulted_) {
    return 0;
  }
  auto caches = PlanCaches();
  return ReclaimPlan(caches, kind, id);
}

std::uint64_t Qwen38Runner::reclaimed_plans() const {
  std::uint64_t n = 0;
  for (const PlanCacheBase* cache : PlanCaches()) {
    n += cache->reclaimed_plans();
  }
  return n;
}

std::uint64_t Qwen38Runner::reclaimed_graphs() const {
  std::uint64_t n = 0;
  for (const PlanCacheBase* cache : PlanCaches()) {
    n += cache->reclaimed_graphs();
  }
  return n;
}

std::vector<ExtentId> Qwen38Runner::state() const {
  std::vector<ExtentId> all;
  for (const RequestState* request : Requests()) {
    const auto extents = request->live.extents();
    all.insert(all.end(), extents.begin(), extents.end());
  }
  return all;
}

std::vector<ExtentId> Qwen38Runner::unchanged_state() const {
  std::vector<ExtentId> all;
  for (const RequestState* request : Requests()) {
    if (!request->spilled) {
      std::vector<ExtentId> written;
      SplitForSpill(*request, written, all);
    }
  }
  return all;
}

void Qwen38Runner::StateWrittenBack(bool whole) {
  for (RequestState* request : Requests()) {
    if (!request->spilled) {
      if (whole) {
        request->track.Saved();
      } else {
        request->track.Lost();
      }
    }
  }
}

bool Qwen38Runner::HasRetainedState() const {
  return std::ranges::any_of(
      Requests(), [](const RequestState* request) { return request->live.used_bytes() != 0; });
}

Qwen38SetupBudget Qwen38Runner::setup_budget() const {
  Qwen38SetupBudget budget = setup_budget_;
  for (const RequestState* request : Requests()) {
    budget.initialized_logical += request->live.used_bytes();
    budget.initialized_extent_bytes += request->live.extents().size() * kExtent;
  }
  return budget;
}

std::vector<ExtentId> Qwen38Runner::weights() const {
  std::vector<ExtentId> all = weights_.extents();
  all.insert(all.end(), dweights_.extents().begin(), dweights_.extents().end());
  return all;
}

std::vector<ExtentId> Qwen38Runner::managed_extents() const {
  std::vector<ExtentId> all = weights();
  const std::vector<ExtentId> live = state();
  all.insert(all.end(), live.begin(), live.end());
  return all;
}

Status Qwen38Runner::Setup() {
  if (o_.wave_slots == 0 || o_.request_slots < o_.wave_slots || o_.request_slots > kRequestSlots) {
    return Error(std::format(
        "Qwen3.8 provisioning needs one to {} request slots, at least its wave's {} (not {})",
        kRequestSlots, o_.wave_slots, o_.request_slots));
  }
  slot_count_ = o_.request_slots;
  if (o_.routed_capture != 0 &&
      (!kg::Qwen38RoutedCaptureFits(o_.routed_capture, profile_.layers) || o_.drafter.empty() ||
       o_.draft_rows == 0 || o_.draft_rows > 3 || o_.context > 131072 || o_.max_rows > 8192)) {
    return Error(
        "routed capture requires <=3 layers, a depth<=3 drafter, context<=131072 and chunk<=8192");
  }
  if (o_.draft_head_capture && (o_.drafter.empty() || o_.context > 131072 || o_.max_rows > 8192)) {
    return Error(
        "draft-head capture needs a drafter, context at most 131072 and chunk at most 8192");
  }
  if (o_.context > md::kQwen38FlashContext) {
    return Error(std::format("context {} exceeds Qwen3.8 Flash Next's trained ceiling {}",
                             o_.context, md::kQwen38FlashContext));
  }
  if (auto r = weights_.Open(o_.artifact); !r) {
    return r;
  }
  const artifact::Artifact& a = weights_.artifact();
  auto binding = md::BindQwen38(profile_, a);
  if (!binding) {
    return std::unexpected(binding.error());
  }
  binding_ = std::move(*binding);
  // The fast graph (the runner's) builds no tensor of every cell by every
  // row: no RE-037 bound on its chunks.
  auto layout = md::Qwen38State(profile_, o_.context, o_.max_rows, false);
  if (!layout) {
    return std::unexpected(layout.error());
  }
  layout_ = std::move(*layout);
  // The largest shapes are planned at the context's end (below): a whole
  // chunk after at least one position (a drafter's prefill pass starts a
  // row before its chunk), and beside a drafter a verify after its drafts.
  // Checked here, so those positions cannot wrap.
  if (o_.max_rows >= o_.context ||
      (!o_.drafter.empty() && (std::uint64_t{o_.draft_rows} * 2) + 1 > o_.context)) {
    return Error(std::format(
        "a context of {} leaves no room for chunks of {} rows{}", o_.context, o_.max_rows,
        o_.drafter.empty() ? std::string() : std::format(" and drafts of {}", o_.draft_rows)));
  }
  if (!o_.drafter.empty()) {
    const std::uint32_t verify = o_.draft_rows + 1;
    if (o_.draft_rows == 0 || verify > kg::kMxfp8VecColumns || verify > o_.max_rows ||
        o_.draft_vocab > profile_.vocab || !binding_.cutlass()) {
      return Error(
          std::format("a draft of 1 to {} rows (its verify the vector products' rows) over at "
                      "most the vocabulary, beside a CUTLASS-layout target",
                      kg::kMxfp8VecColumns - 1));
    }
    if (auto r = dweights_.Open(o_.drafter); !r) {
      return Error(std::format("the drafter: {}", r.error()));
    }
    auto dbinding = md::BindQwen38Mtp(profile_, dweights_.artifact());
    if (!dbinding) {
      return Error(std::format("the drafter: {}", dbinding.error()));
    }
    dbinding_ = std::move(*dbinding);
    if (o_.draft_head_capture) {
      const std::uint64_t available =
          dbinding_.selected_head() ? dbinding_.draft_ids.ne[1] : profile_.vocab;
      const std::uint64_t requested =
          o_.draft_vocab == 0 ? available : std::min<std::uint64_t>(o_.draft_vocab, available);
      if (requested == 0 || requested > 65536) {
        return Error("draft-head capture requires 1 to 65536 head rows");
      }
      capture_head_rows_ = static_cast<std::uint32_t>(requested);
    }
    auto mtp = md::Qwen38MtpStateOf(profile_, layout_);
    auto commit = md::Qwen38Commit(profile_, verify);
    if (!mtp || !commit) {
      return Error(!mtp ? mtp.error() : commit.error());
    }
    mtp_layout_ = *mtp;
    commit_layout_ = std::move(*commit);
  }

  // The n-gram table: its group alone, stored contiguously in one shard.
  const auto groups = a.groups();
  const artifact::Resource& table = a.resources()[binding_.ple_table.index];
  const std::uint32_t table_group = table.group;
  for (std::uint32_t r = 0; r < a.resources().size(); ++r) {
    if (r != binding_.ple_table.index && a.resources()[r].group == table_group) {
      return Error(std::format("{} shares the n-gram table's group", a.resources()[r].name));
    }
  }
  const auto first = artifact::ChunkRangeOf(a.layout(), {.group = table_group, .chunk = 0});
  if (!first) {
    return Error("the n-gram table's file range");
  }
  for (std::uint32_t c = 1; c < groups[table_group].chunks; ++c) {
    const auto range = artifact::ChunkRangeOf(a.layout(), {.group = table_group, .chunk = c});
    if (!range || range->shard != first->shard ||
        range->file_offset.value() != first->file_offset.value() + (std::uint64_t{c} * kExtent)) {
      return Error("the n-gram table is not stored contiguously in one shard");
    }
  }
  if (binding_.ple_table.ne.size() != 2) {
    return Error("the n-gram table is not a table of rows");
  }
  // A row's bytes: the ModelOpt table's plain bytes [row bytes, rows], or a
  // GGUF checkpoint's GGML rows [values, rows] of its type.
  std::uint64_t row_bytes = binding_.ple_table.ne[0];
  if (binding_.gguf()) {
    auto type = kg::GgmlTypeOf(binding_.ple_table.type);
    if (!type) {
      return Error(type.error().detail);
    }
    row_bytes = ggml_row_size(*type, static_cast<std::int64_t>(binding_.ple_table.ne[0]));
  }
  table_ = PleTable{.fd = weights_.shard_fd(first->shard),
                    .file_offset = first->file_offset.value() + table.offset.value(),
                    .rows = binding_.ple_table.ne[1],
                    .row_bytes = row_bytes,
                    .chunk_file_offset = first->file_offset.value(),
                    .file_bytes = first->file_offset.value() + groups[table_group].stored.value()};
  const std::uint64_t row_slots =
      o_.wave_slots > 1 ? std::max<std::uint64_t>(o_.max_rows, std::uint64_t{o_.wave_slots} * 4U)
                        : o_.max_rows;
  slots_ = row_slots * profile_.ple_heads();
  graph_binding_ = binding_;
  graph_binding_.ple_table.ne[1] = slots_;
  graph_binding_.ple_table.readable = 0;  // the slots reserve no over-read

  // The state first: its extents come before the weights' in a closure, so
  // a swap back restores it before paging the weights in.
  if (auto r = live_.AddGrowing(node_, "the Qwen3.8 state", layout_.bytes, owner_); !r) {
    return r;
  }
  if (speculative()) {
    if (auto r =
            live_.AddGrowing(node_, "the Qwen3.8 MTP drafter's state", mtp_layout_.bytes, owner_);
        !r) {
      return r;
    }
  }
  if (auto r = resources_.Map(slot_memory_, "the Qwen3.8 n-gram row slots",
                              slots_ * table_.row_bytes, MemoryClass::kScratch);
      !r) {
    return r;
  }
  std::vector<std::uint64_t> stride;
  std::uint64_t mtp_stride = 0;
  if (auto r = ReserveWeights(stride, mtp_stride); !r) {
    return r;
  }
  if (auto r = resources_.OpenCublas("the Qwen3.8 cuBLAS workspace"); !r) {
    return r;
  }

  // The largest shapes, as the resident harness sizes them (one output
  // row: the runner reads only the last row's logits; a verify every
  // row's), planned over placeless addresses with a stand-in hash (the rows
  // do not shape a chunk); beside a drafter, its prefill pass and its draft
  // too.
  const auto placeless = [](std::uint32_t) { return std::uint64_t{1} << 44U; };
  model_ = Qwen38Model{.artifact = &a,
                       .profile = &profile_,
                       .binding = &graph_binding_,
                       .state = &layout_,
                       .places = {.resource = placeless,
                                  .array = placeless,
                                  .stride = std::move(stride),
                                  .state = std::uint64_t{1} << 45U,
                                  .ple_table = std::uint64_t{1} << 44U,
                                  .mtp_resource = placeless,
                                  .mtp_array = placeless,
                                  .mtp_state = std::uint64_t{1} << 45U,
                                  .commit = std::uint64_t{1} << 45U},
                       .cutlass = binding_.cutlass(),
                       .drafter = speculative() ? &dbinding_ : nullptr,
                       .mtp_state = speculative() ? &mtp_layout_ : nullptr,
                       .mtp_stride = mtp_stride,
                       .commit = speculative() ? &commit_layout_ : nullptr};
  md::Qwen38PleHash stand_in;
  stand_in.multipliers.assign(profile_.ngram, 1);
  stand_in.offsets.assign(profile_.ple_heads(), 0);
  stand_in.vocab.assign(profile_.ple_heads(), 1);
  stand_in.table_rows = 1;
  std::uint64_t most_activations = 0;
  // Each kind's largest plan (PlannedHostBytes) and most launched nodes, for
  // plan_floor_bytes().
  struct PlanKind {
    std::uint64_t host = 0;
    std::uint64_t nodes = 0;
  };
  PlanKind chunk_kind;
  PlanKind mtp_kind;
  std::uint64_t most_scratch = 0;
  std::uint64_t most_inputs = 0;
  // The widest waves' own (below): placed activations, scratch, staged
  // inputs and what their plans hold on the host.
  std::uint64_t wave_activations = 0;
  std::uint64_t wave_scratch = 0;
  std::uint64_t wave_inputs = 0;
  std::uint64_t target_wave_host = 0;
  std::uint64_t draft_wave_host = 0;
  {
    auto measure = resources_.MeasuringContext();
    if (!measure) {
      return std::unexpected(measure.error());
    }
    const kg::DeviceChoices choices = kg::DeviceChoicesOf(**measure);
    struct Probe {
      std::uint32_t n_past;
      std::uint32_t rows;
      Qwen38ChunkKind kind;
    };
    const std::uint32_t verify = o_.draft_rows + 1;
    // (The fast graph's largest host input is the causal mask of the widest
    // chunk that does not select: one whose cells stay within the
    // indexer's budget, [2,048 cells, rows]; the chunks that select read
    // no mask.)
    const std::uint32_t unselected = std::min<std::uint32_t>(
        o_.context, (profile_.indexer_budget + profile_.indexer_ratio - 1) / 256U * 256U);
    const std::uint32_t unselected_rows = std::min(o_.max_rows, unselected);
    std::vector<Probe> probes = {{0, o_.max_rows, {}},
                                 {o_.context - o_.max_rows, o_.max_rows, {}},
                                 {unselected - unselected_rows, unselected_rows, {}},
                                 {o_.context - 1, 1, {}},
                                 {0, 1, {}}};
    if (speculative()) {
      for (const std::uint32_t at : {0U, o_.context - o_.max_rows}) {
        probes.push_back({at, o_.max_rows, {.verify = false, .export_streams = true}});
      }
      probes.push_back({unselected - unselected_rows,
                        unselected_rows,
                        {.verify = false, .export_streams = true}});
      for (const std::uint32_t at : {0U, o_.context - verify}) {
        probes.push_back({at, verify, {.verify = true, .export_streams = true}});
        if (o_.routed_capture != 0) {
          probes.push_back(
              {at,
               verify,
               {.verify = true, .export_streams = true, .capture_routed = o_.routed_capture}});
        }
      }
    }
    const auto account = [&](const PlannedBase& planned, PlanKind& kind) -> Status {
      kind.host = std::max(kind.host, PlannedHostBytes(planned));
      kind.nodes = std::max(kind.nodes, PlannedNodes(planned));
      most_activations = std::max(most_activations, planned.placement.extent);
      auto scratch = kg::PlanScratch(**measure, planned.plan);
      if (!scratch) {
        return Error(scratch.error().detail);
      }
      most_scratch = std::max(most_scratch, *scratch);
      most_inputs = std::max(most_inputs, planned.inputs_bytes);
      return {};
    };
    for (const Probe& probe : probes) {
      std::vector<std::int32_t> history(std::size_t{probe.n_past} + probe.rows, 1000);
      auto in =
          md::Qwen38Chunk(profile_, layout_, stand_in, history, probe.n_past, probe.rows, false);
      if (!in) {
        return std::unexpected(in.error());
      }
      auto planned = PlanQwen38Chunk(
          model_, kg::Qwen38ShapeOf(layout_, *in, probe.kind.verify ? probe.rows : 1), choices, 0,
          0, {}, probe.kind);
      if (!planned) {
        return Error(std::format("measuring a chunk of {} at {}: {}", probe.rows, probe.n_past,
                                 planned.error()));
      }
      if (auto r = account(**planned, chunk_kind); !r) {
        return r;
      }
    }
    if (speculative()) {
      // A prefill pass of a whole chunk at the end, and a draft there; and
      // the widest pass that does not select (its mask).
      for (const auto& [from, rows, passes, head] :
           {std::tuple{o_.context - o_.max_rows - 1, o_.max_rows, 1U, false},
            std::tuple{o_.context - verify - o_.draft_rows, verify, o_.draft_rows, true},
            std::tuple{unselected - unselected_rows, unselected_rows, 1U, false}}) {
        auto shaped =
            MtpInputs(from, rows, passes, head, head ? 1 : 0, head, head && o_.draft_head_capture);
        if (!shaped) {
          return std::unexpected(shaped.error());
        }
        auto planned = PlanQwen38Mtp(model_, shaped->first, choices, 0, 0);
        if (!planned) {
          return Error(std::format("measuring the drafter: {}", planned.error()));
        }
        if (auto r = account(**planned, mtp_kind); !r) {
          return r;
        }
        if (shaped->first.capture_head) {
          shaped->first.capture_head = false;
          auto ordinary = PlanQwen38Mtp(model_, shaped->first, choices, 0, 0);
          if (!ordinary) {
            return Error(std::format("measuring the uncaptured drafter: {}", ordinary.error()));
          }
          if (auto r = account(**ordinary, mtp_kind); !r) {
            return r;
          }
        }
      }
    }
    if (o_.wave_slots > 1) {
      // The widest waves as the runner composes them: every slot at the
      // context's end with its most rows (a verify of every draft, or a
      // decode step; beside a drafter, a draft of every pending row), its
      // caches read at the waves' alignment, each slot's state at a place
      // of its own. Their placed activations, scratch and staged inputs
      // bound what a wave needs of the workspace: a wave runs only these
      // few-row shapes, so a slot adds its share of them, not another
      // prefill chunk's (docs/experiments/request-slots/).
      const std::uint32_t align = WaveReadAlign(o_.wave_slots);
      const std::uint64_t target_span = Round(layout_.bytes, kExtent);
      const std::uint64_t mtp_span = speculative() ? Round(mtp_layout_.bytes, kExtent) : 0;
      const std::uint64_t commit_span = speculative() ? Round(commit_layout_.bytes, kExtent) : 0;
      std::vector<Qwen38Model> models(o_.wave_slots, model_);
      for (std::uint32_t s = 0; s < o_.wave_slots; ++s) {
        auto& places = models[s].places;
        places.state = (std::uint64_t{1} << 45U) + (s * (target_span + mtp_span + commit_span));
        places.mtp_state = places.state + target_span;
        places.commit = places.mtp_state + mtp_span;
      }
      const auto wave_account = [&](const Qwen38WavePlanned& planned,
                                    std::uint64_t& host) -> Status {
        wave_activations = std::max(wave_activations, planned.placement.extent);
        auto scratch = kg::PlanScratch(**measure, planned.plan);
        if (!scratch) {
          return Error(scratch.error().detail);
        }
        wave_scratch = std::max(wave_scratch, *scratch);
        wave_inputs = std::max(wave_inputs, planned.inputs_bytes);
        host = std::max(host, planned.host_bytes());
        return {};
      };
      // (A wave's rows and passes are at most four and three a slot.)
      const std::uint32_t rows = speculative() ? std::min(verify, 4U) : 1;
      std::vector<std::int32_t> history(o_.context, 1000);
      auto in = md::Qwen38Chunk(profile_, layout_, stand_in, history, o_.context - rows, rows,
                                false, align);
      if (!in) {
        return std::unexpected(in.error());
      }
      const Qwen38ChunkKind kind{.verify = speculative(), .export_streams = speculative()};
      std::vector<Qwen38TargetWaveInput> targets;
      targets.reserve(o_.wave_slots);
      for (std::uint32_t s = 0; s < o_.wave_slots; ++s) {
        targets.push_back(
            {s, &models[s], kg::Qwen38ShapeOf(layout_, *in, speculative() ? rows : 1), kind});
      }
      auto target = PlanQwen38TargetWave(targets, choices, {.share_target_head = true});
      if (!target) {
        return Error(
            std::format("measuring a wave of {} slots: {}", o_.wave_slots, target.error()));
      }
      if (auto r = wave_account(**target, target_wave_host); !r) {
        return r;
      }
      if (speculative()) {
        const std::uint32_t passes = std::min(o_.draft_rows, 3U);
        const std::uint32_t anchor = o_.context - passes;
        auto shaped = MtpInputs(anchor - rows, rows, passes, true, 1, true, false, align);
        if (!shaped) {
          return std::unexpected(shaped.error());
        }
        std::vector<Qwen38DraftWaveInput> drafts;
        drafts.reserve(o_.wave_slots);
        for (std::uint32_t s = 0; s < o_.wave_slots; ++s) {
          drafts.push_back({s, &models[s], shaped->first});
        }
        auto draft = PlanQwen38DraftWave(drafts, choices);
        if (!draft) {
          return Error(
              std::format("measuring a draft wave of {} slots: {}", o_.wave_slots, draft.error()));
        }
        if (auto r = wave_account(**draft, draft_wave_host); !r) {
          return r;
        }
      }
    }
  }
  // What one step holds at once at most (plan_floor_bytes): a chunk beside
  // its drafter pass (an injected prefill chunk), or a unit's target wave
  // beside its draft wave (the widest measured above, with a quarter's
  // margin for other shapes). Every plan and graph past it is charged
  // inside the budget.
  {
    const std::uint64_t chunk_step = chunk_kind.host + (speculative() ? mtp_kind.host : 0);
    const std::uint64_t target_wave = target_wave_host + (target_wave_host / 4);
    const std::uint64_t draft_wave = draft_wave_host + (draft_wave_host / 4);
    plan_floor_bytes_ = std::max(chunk_step, target_wave + draft_wave);
    const auto mib = [](std::uint64_t bytes) { return static_cast<double>(bytes) / (1U << 20U); };
    plan_report_ = std::format(
        "{:.1f} MiB a step at most: chunk plans of {:.1f} MiB ({} nodes), drafter plans of {:.1f} "
        "MiB ({} nodes), target waves of {:.1f} MiB, draft waves of {:.1f} MiB",
        mib(plan_floor_bytes_), mib(chunk_kind.host), chunk_kind.nodes, mib(mtp_kind.host),
        mtp_kind.nodes, mib(target_wave), mib(draft_wave));
  }
  const auto limit = std::numeric_limits<std::uint64_t>::max() / 16;
  if (most_activations > limit || wave_activations > limit || most_inputs > limit ||
      wave_inputs > limit || most_scratch > limit || wave_scratch > limit) {
    return Error("Qwen3.8 provisioning exceeds checked wave bounds");
  }
  // The largest chunk's, or the widest wave's (above), with the same
  // margins: other shapes of these widths place a little differently.
  const auto activation_bound = std::max(most_activations, wave_activations);
  activation_bytes_ = Round(activation_bound + (activation_bound / 4), kExtent);
  const auto scratch_bound = std::max(most_scratch, wave_scratch);
  scratch_bytes_ = Round(scratch_bound + (scratch_bound / 4) + (1U << 20U), kExtent);
  // A chunk's inputs and its drafter pass's (from the second half), or a
  // wave's every slot's at once.
  const std::uint64_t input_bytes =
      Round(std::max(most_inputs * 2, wave_inputs) + (1U << 20U), kExtent);
  // A chunk's host-built inputs are the staged bytes again, on the host.
  host_input_bytes_ = Round(std::max(most_inputs, wave_inputs) + (1U << 20U), kExtent);
  setup_budget_.scalar_activations = Round(most_activations + (most_activations / 4), kExtent);
  setup_budget_.scalar_scratch = Round(most_scratch + (most_scratch / 4) + (1U << 20U), kExtent);
  setup_budget_.scalar_staging_inputs = Round((most_inputs * 2) + (1U << 20U), kExtent);
  setup_budget_.scalar_host_inputs = Round(most_inputs + (1U << 20U), kExtent);
  setup_budget_.activations = activation_bytes_;
  setup_budget_.scratch = scratch_bytes_;
  setup_budget_.staging_inputs = input_bytes;
  setup_budget_.host_inputs = host_input_bytes_;
  setup_budget_.ple_mapped = slot_memory_.bytes;
  const std::uint64_t scalar_slots = std::uint64_t{o_.max_rows} * profile_.ple_heads();
  setup_budget_.scalar_ple_mapped = Round(scalar_slots * table_.row_bytes, kExtent);
  // A drafter pass's inputs from the staging's second half, which the
  // largest inputs fit, so one job stages a chunk's and its pass's.
  mtp_base_ = Round(input_bytes / 2, 256);

  landing_bytes_ = PleLandingBound(slots_);
  const std::uint64_t logit_rows = speculative() ? o_.draft_rows + 1 : 1;
  auto inputs = resources_.Pinned(input_bytes);
  auto logits = resources_.Pinned(logit_rows * std::uint64_t{profile_.vocab} * sizeof(float));
  auto hash = resources_.Pinned(
      (std::uint64_t{profile_.ngram} + (2 * std::uint64_t{profile_.ple_heads()})) * 8);
  auto landing = resources_.Pinned(landing_bytes_);
  // The slots' sources, then the count of rows the next gather takes.
  auto sources = resources_.Pinned((slots_ + 1) * sizeof(std::uint32_t));
  unwritten_ = weights_.Unwritten();
  auto scrub = resources_.Pinned(unwritten_.size() * 2 * sizeof(std::uint64_t));
  if (!inputs || !logits || !hash || !landing || !sources || !scrub) {
    return Error("pinned staging for Qwen3.8");
  }
  scrub_ = static_cast<std::uint64_t*>(*scrub);
  runs_.SetStaging(*inputs, input_bytes);
  logits_ = *logits;
  if (o_.wave_slots > 1) {
    // Separate from the legacy default outputs, so existing scalar/proof
    // addresses stay unchanged. Every provisioned slot's indexed slice
    // exists even for a smaller cohort, which may select sparse masks such
    // as 0b1010.
    wave_logit_words_ = std::max<std::uint64_t>(4, logit_rows) * profile_.vocab;
    auto wave_logits = resources_.Pinned(slot_count_ * wave_logit_words_ * sizeof(float));
    auto wave_ids = resources_.Pinned(slot_count_ * 8 * sizeof(std::int32_t));
    auto wave_probabilities = resources_.Pinned(slot_count_ * 8 * sizeof(float));
    if (!wave_logits || !wave_ids || !wave_probabilities) {
      return Error("pinned per-slot Qwen3.8 wave outputs");
    }
    wave_logits_ = static_cast<float*>(*wave_logits);
    wave_ids_ = static_cast<std::int32_t*>(*wave_ids);
    wave_probabilities_ = static_cast<float*>(*wave_probabilities);
    setup_budget_.wave_output_pinned =
        (slot_count_ * wave_logit_words_ * sizeof(float)) +
        (2 * std::max<std::uint64_t>(slot_count_ * 8 * sizeof(float), 256));
  }
  hash_host_ = *hash;
  if (capture_head_rows_ != 0) {
    const std::uint64_t words =
        std::uint64_t{o_.draft_rows} * (std::uint64_t{profile_.width} + capture_head_rows_);
    auto captured = resources_.Pinned(words * sizeof(float));
    if (!captured) {
      return Error("pinned staging for the bounded draft-head capture");
    }
    draft_head_capture_ = static_cast<float*>(*captured);
  }
  if (o_.routed_capture != 0) {
    const std::uint64_t per_row =
        (std::uint64_t{profile_.experts_used} * (profile_.expert_ffn + profile_.width + 2)) +
        (3 * std::uint64_t{profile_.width}) + 1;
    std::uint64_t words = per_row * static_cast<std::uint64_t>(std::popcount(o_.routed_capture));
    for (std::uint32_t layer = 0; layer < profile_.layers; ++layer) {
      if ((o_.routed_capture & (std::uint64_t{1} << layer)) != 0 &&
          !binding_.layers[layer].linear) {
        words +=
            std::uint64_t{profile_.width} + (2 * std::uint64_t{profile_.heads} * profile_.head_dim);
      }
    }
    routed_capture_bytes_ = words * (o_.draft_rows + 1) * 4;
    auto captured = resources_.Pinned(routed_capture_bytes_);
    if (!captured) {
      return Error("pinned staging for the bounded routed capture");
    }
    routed_capture_ = static_cast<std::byte*>(*captured);
  }
  if (dbinding_.selected_head()) {
    auto ids = resources_.Pinned(dbinding_.draft_ids.ne[1] * sizeof(std::int32_t));
    if (!ids) {
      return Error("pinned staging for the draft vocabulary IDs");
    }
    draft_ids_host_ = *ids;
  }
  landing_ = static_cast<std::byte*>(*landing);
  sources_ = static_cast<std::uint32_t*>(*sources);
  ple_count_ = sources_ + slots_;
  *ple_count_ = 0;
  if (Address(landing_) % kPleBlock != 0) {
    return Error("the n-gram rows' landing is not 4 KiB-aligned for direct reads");
  }
  if (speculative()) {
    if (auto r = SetupSnapshot(default_request_); !r) {
      return r;
    }
    // A draft's drafts, their probabilities (from kProbabilityAt), then
    // (from kArgmaxAt) a verify's argmaxes.
    auto drafts = resources_.Pinned(512);
    if (!drafts) {
      return Error("pinned staging for Qwen3.8's speculation");
    }
    drafts_ = *drafts;
  }
  auto ring = providers::OpenStorage(kRingDepth);
  if (!ring) {
    return Error(std::format("the n-gram rows' ring: {}", ring.error().message()));
  }
  ring_ = std::move(*ring);
  // Provision additional slots after every legacy default allocation, so
  // slot zero's state, commit and shared weight/workspace places stay put.
  // Every provisioned slot's virtual ceilings and snapshot exist before
  // Register/Start; growing state acquires physical backing only when a
  // slot uses it.
  for (RequestState* added : Requests().subspan(1)) {
    RequestState& request = *added;
    request.model = model_;
    if (auto r = request.live.AddGrowing(
            node_, std::format("the Qwen3.8 slot {} state", request.slot), layout_.bytes, owner_);
        !r) {
      return r;
    }
    if (speculative()) {
      if (auto r = request.live.AddGrowing(
              node_, std::format("the Qwen3.8 slot {} MTP state", request.slot), mtp_layout_.bytes,
              owner_);
          !r) {
        return r;
      }
      if (auto r = SetupSnapshot(request); !r) {
        return r;
      }
    }
  }
  setup_budget_.pinned = resources_.pinned_bytes();
  // Only input staging, PLE landing/source staging and wave outputs change
  // pinned demand. Capture/selected-head/snapshot descriptors stay identical.
  const auto pinned = [](std::uint64_t bytes) { return std::max<std::uint64_t>(bytes, 256); };
  setup_budget_.scalar_pinned =
      setup_budget_.pinned - pinned(input_bytes) - pinned(landing_bytes_) -
      pinned((slots_ + 1) * sizeof(std::uint32_t)) - setup_budget_.wave_output_pinned +
      pinned(setup_budget_.scalar_staging_inputs) + pinned(PleLandingBound(scalar_slots)) +
      pinned((scalar_slots + 1) * sizeof(std::uint32_t));
  setup_budget_.runner_mapped = resources_.mapped_bytes();
  setup_budget_.target_per_branch = layout_.bytes;
  setup_budget_.drafter_per_branch = speculative() ? mtp_layout_.bytes : 0;
  setup_budget_.virtual_per_branch =
      Round(layout_.bytes, kExtent) + (speculative() ? Round(mtp_layout_.bytes, kExtent) : 0);
  return {};
}

Status Qwen38Runner::SetupSnapshot(RequestState& request) {
  // A verify's saves, then the snapshot of the cells it writes (D-068
  // working state), mapped for the model's life at this slot's own place.
  const std::uint32_t qsa = profile_.layers / 4;
  const std::uint64_t row_cells = (2 * std::uint64_t{profile_.head_dim} * profile_.kv_heads * 2) +
                                  (std::uint64_t{profile_.indexer_head_dim} * 4);
  const std::uint64_t snapshot_offset = Round(commit_layout_.bytes, 256);
  const std::uint64_t block_keys =
      ((std::uint64_t{o_.draft_rows + 1} / profile_.indexer_ratio) + 1) * qsa *
      std::uint64_t{profile_.indexer_head_dim} * 2;
  // Shared verifies can discard a completed result after host judgement
  // fails. Preserve the overwritten pending MTP streams as well as KV cells;
  // the default scalar allocation and verify path remain unchanged.
  const std::uint64_t stream_saves =
      o_.wave_slots > 1
          ? std::uint64_t{o_.draft_rows + 1} * Round(std::uint64_t{profile_.hc_width()} * 4, 256)
          : 0;
  const std::uint64_t commit_bytes =
      snapshot_offset +
      Round((std::uint64_t{o_.draft_rows + 1} * qsa * row_cells) + block_keys + stream_saves + 4096,
            256);
  const std::string name = request.slot == 0
                               ? "the Qwen3.8 verify's saves"
                               : std::format("the Qwen3.8 slot {} verify's saves", request.slot);
  if (auto r = resources_.Map(request.commit, name, commit_bytes, MemoryClass::kRuntime); !r) {
    return r;
  }
  if (request.slot == 0) {
    setup_budget_.snapshot_per_branch = request.commit.bytes;
    setup_budget_.scalar_snapshot_per_branch = Round(
        snapshot_offset +
            Round((std::uint64_t{o_.draft_rows + 1} * qsa * row_cells) + block_keys + 4096, 256),
        kExtent);
  }
  request.live.SnapshotAt(request.commit.base + snapshot_offset,
                          request.commit.bytes - snapshot_offset);
  if (auto r = request.live.AllocateSnapshot(resources_, kRangeCapacity); !r) {
    return r;
  }
  auto carry = resources_.Pinned(4 * sizeof(kg::RangeCopy));
  if (!carry) {
    return Error("pinned staging for Qwen3.8's speculation");
  }
  request.carry = static_cast<kg::RangeCopy*>(*carry);
  return {};
}

// Every dense group but the n-gram table's, and a slab per layer; the
// drafter's dense groups and its slab. Each layer's expert stride in
// `stride`, the drafter's in `mtp_stride`.
Status Qwen38Runner::ReserveWeights(std::vector<std::uint64_t>& stride, std::uint64_t& mtp_stride) {
  const artifact::Artifact& a = weights_.artifact();
  const auto groups = a.groups();
  std::vector<GroupPlace> place(groups.size(), GroupPlace::kNone);
  for (std::size_t g = 0; g < groups.size(); ++g) {
    if (groups[g].kind != artifact::GroupKind::kExpert) {
      place[g] = GroupPlace::kDevice;
    }
  }
  place[a.resources()[binding_.ple_table.index].group] = GroupPlace::kNone;
  std::vector<SlabSpec> slabs;
  stride.assign(profile_.layers, 0);
  for (std::uint32_t il = 0; il < profile_.layers; ++il) {
    auto slab = SlabOf(a, binding_.layers[il], binding_.cutlass(), profile_.experts, il);
    if (!slab) {
      return std::unexpected(slab.error());
    }
    stride[il] = slab->stride;
    slabs.push_back(*slab);
  }
  if (auto r = weights_.Reserve(node_, place, slabs); !r) {
    return r;
  }
  if (!speculative()) {
    return {};
  }
  const artifact::Artifact& d = dweights_.artifact();
  std::vector<GroupPlace> dplace(d.groups().size(), GroupPlace::kNone);
  for (std::size_t g = 0; g < d.groups().size(); ++g) {
    if (d.groups()[g].kind != artifact::GroupKind::kExpert) {
      dplace[g] = GroupPlace::kDevice;
    }
  }
  auto slab = SlabOf(d, dbinding_.layer, true, profile_.experts, 0);
  if (!slab) {
    return std::unexpected(slab.error());
  }
  mtp_stride = slab->stride;
  const std::array<SlabSpec, 1> dslabs = {*slab};
  return dweights_.Reserve(node_, dplace, dslabs);
}

Status Qwen38Runner::Register() {
  if (auto r = weights_.Register(node_, owner_); !r) {
    return r;
  }
  if (speculative()) {
    if (auto r = dweights_.Register(node_, owner_); !r) {
      return r;
    }
  }
  for (RequestState* request : Requests()) {
    auto r = o_.spill_place ? request->live.RegisterSpill(node_, o_.spill_place(request->slot))
                            : request->live.RegisterSpill(node_, o_.out);
    if (!r) {
      return r;
    }
  }
  // D-090, for every model: the places stay put for the model's life
  // (never unpinned; the pins go with the scheduler).
  auto pinned_extents = weights();
  for (const RequestState* request : Requests()) {
    const auto reserved = request->live.reserved_extents();
    pinned_extents.insert(pinned_extents.end(), reserved.begin(), reserved.end());
  }
  if (auto pinned = node_.scheduler().PinPlaces(pinned_extents); !pinned) {
    return Error(std::format("pinning Qwen3.8's places: {}", sc::ToString(pinned.error())));
  }
  return {};
}

Status Qwen38Runner::CheckPlaces() {
  PlaceCheck check;
  auto checked = node_.Call(
      [&]() -> Status {
        weights_.CheckPlaces(node_.scheduler(), check);
        dweights_.CheckPlaces(node_.scheduler(), check);
        for (const RequestState* request : Requests()) {
          request->live.CheckPlaces(node_.scheduler(), check);
        }
        return {};
      },
      "checking Qwen3.8's places");
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
  return {};
}

Status Qwen38Runner::RefreshClosures() { return RefreshClosures(active_mask_); }

Status Qwen38Runner::RefreshClosures(SlotMask protected_mask) {
  auto refreshed = node_.Call(
      [&]() -> Status {
        auto& catalog = node_.catalog();
        std::vector<ExtentId> shared = weights();
        for (const Mapped* mapped :
             std::initializer_list<const Mapped*>{&node_.activations(), &node_.pool()}) {
          shared.insert(shared.end(), mapped->extents.begin(), mapped->extents.end());
        }
        const std::vector<ExtentId> own = resources_.extents();
        shared.insert(shared.end(), own.begin(), own.end());
        std::vector<ExtentId> all = shared;
        std::vector<ExtentId> active = shared;
        std::array<catalog::Closure, kRequestSlots> slot_fences;
        std::vector<ExtentId> resident_state;
        for (RequestState* request : Requests()) {
          if (request->spilled) {
            continue;  // in no closure until Restore brings it back
          }
          const auto live = request->live.extents();
          resident_state.insert(resident_state.end(), live.begin(), live.end());
          all.insert(all.end(), live.begin(), live.end());
          if ((protected_mask & (SlotMask{1} << request->slot)) != 0) {
            active.insert(active.end(), live.begin(), live.end());
          }
          auto fence = catalog.ClosureOfExtents(live);
          if (!fence) {
            return Error("a Qwen3.8 slot's state closure is no longer cataloged");
          }
          slot_fences[request->slot] = std::move(*fence);
        }
        auto everything = catalog.ClosureOfExtents(all);
        auto fence = catalog.ClosureOfExtents(resident_state);
        auto execution = catalog.ClosureOfExtents(active);
        if (!everything || !fence || !execution) {
          return Error("a Qwen3.8 cohort closure is no longer cataloged");
        }
        for (RequestState* request : Requests()) {
          request->fence = std::move(slot_fences[request->slot]);
        }
        everything_ = std::move(*everything);
        fence_ = std::move(*fence);
        execution_ = std::move(*execution);
        return {};
      },
      "refreshing Qwen3.8's used state closure");
  if (!refreshed) {
    FaultCohort();
    return refreshed;
  }
  if (auto held = node_.RefreshRequest(stream_, execution_); !held) {
    // RefreshRequest may have ended the previous lease before refusing its
    // replacement. No slot may dispatch after that loss of protection.
    FaultCohort();
    return held;
  }
  return {};
}

Status Qwen38Runner::Bind() {
  if (auto refreshed = RefreshClosures(); !refreshed) {
    return refreshed;
  }
  for (RequestState* request : Requests()) {
    if (auto bound = BindRequest(*request); !bound) {
      return bound;
    }
  }
  // Every plan and graph charged to the node as it is kept.
  account_.Bind(
      [this](std::uint64_t bytes, bool required) { return node_.ChargeHost(bytes, required); },
      [this](std::uint64_t bytes) { node_.UnchargeHost(bytes); });
  for (RequestState* request : Requests()) {
    request->plans.set_account(&account_);
    request->mplans.set_account(&account_);
  }
  target_waves_.set_account(&account_);
  draft_waves_.set_account(&account_);
  if (auto r = resources_.BindLaunch(scratch_bytes_); !r) {
    return r;
  }
  runs_.SetLaunch(&resources_.launch());
  return {};
}

Status Qwen38Runner::BindRequest(RequestState& request) {
  request.model.places.resource = [this](std::uint32_t resource) {
    return weights_.resource_address(resource);
  };
  request.model.places.array = [this](std::uint32_t array) {
    return weights_.array_address(array);
  };
  request.model.places.state = request.live.base(kTarget);
  request.model.places.ple_table = slot_memory_.base;
  if (speculative()) {
    request.model.places.mtp_resource = [this](std::uint32_t resource) {
      return dweights_.resource_address(resource);
    };
    request.model.places.mtp_array = [this](std::uint32_t array) {
      return dweights_.array_address(array);
    };
    request.model.places.mtp_state = request.live.base(kDrafter);
    request.model.places.commit = request.commit.base;
    // The commit's places: every linear-attention layer's state and saves.
    using K = md::Qwen38StateTensor::Kind;
    const std::uint64_t state = request.live.base(kTarget);
    const auto at = [&](std::uint32_t il, K kind) {
      return state + layout_.tensors[static_cast<std::size_t>(layout_.Find(il, kind))].offset;
    };
    kg::Qwen38CommitArgs& c = request.commit_args;
    c.layers = static_cast<int>(commit_layout_.layers.size());
    c.channels = static_cast<int>(profile_.conv_channels());
    c.qk_heads = static_cast<int>(profile_.lin_k_heads);
    c.v_heads = static_cast<int>(profile_.lin_v_heads);
    c.taps = static_cast<int>(profile_.conv - 1);
    for (std::size_t i = 0; i < commit_layout_.layers.size(); ++i) {
      const std::uint32_t il = commit_layout_.layers[i];
      const std::uint64_t base = request.commit.base;
      c.layer[i] = {.state = static_cast<float*>(Pointer(at(il, K::kRecurrent))),
                    .history = static_cast<float*>(Pointer(at(il, K::kConv))),
                    .conv = static_cast<const float*>(Pointer(base + commit_layout_.conv_out(i))),
                    .qkv = static_cast<const float*>(Pointer(base + commit_layout_.qkv(i))),
                    .gate = static_cast<const float*>(Pointer(base + commit_layout_.gate(i))),
                    .beta = static_cast<const float*>(Pointer(base + commit_layout_.beta(i)))};
    }
    c.ple_history = static_cast<float*>(Pointer(at(profile_.ple_layer, K::kPleConv)));
    c.ple_rows = static_cast<const float*>(Pointer(request.commit.base + commit_layout_.ple()));
    c.ple_width = static_cast<int>(profile_.hc_width());
    c.ple_taps = static_cast<int>(profile_.ple_history());
    // A verify's kept rows committed after the rejected rows' restore.
    request.live.SetCommit([&request](kg::LaunchContext& launch, std::uint32_t keep) {
      kg::Qwen38CommitArgs args = request.commit_args;
      args.keep = static_cast<int>(keep);
      return kg::Qwen38Commit(launch, args);
    });
  }
  return {};
}

// ------------------------------------------------------------------ work

Status Qwen38Runner::ReadPleHash() {
  if (cohort_faulted_ || released_) {
    return Error("the Qwen3.8 cohort requires retirement");
  }
  hash_checked_ = false;
  if (binding_.gguf()) {
    // A GGUF checkpoint's constants are its kept metadata's (no drafter).
    auto hash = md::ReadQwen38GgufHash(profile_, weights_.artifact(), table_.rows);
    if (!hash) {
      return std::unexpected(hash.error());
    }
    hash_ = std::move(*hash);
    hash_checked_ = true;
    return {};
  }
  const md::Qwen38Layer& l = binding_.layers[profile_.ple_layer];
  const std::array<const md::Qwen38Tensor*, 3> parts = {&l.ple_multipliers, &l.ple_head_offsets,
                                                        &l.ple_head_vocab};
  std::vector<std::pair<std::uint64_t, std::uint64_t>> copies;  // address, bytes
  copies.reserve(parts.size());
  for (const md::Qwen38Tensor* t : parts) {
    copies.emplace_back(model_.places.resource(t->index), t->ne[0] * sizeof(std::int64_t));
  }
  void* host = hash_host_;
  if (auto r = node_.Job(
          execution_,
          [&copies, host](providers::NativeStream stream) {
            std::uint64_t at = 0;
            for (const auto& [address, bytes] : copies) {
              if (!providers::CopyAsync(stream, static_cast<std::byte*>(host) + at,
                                        Pointer(address), bytes, providers::CopyKind::kDeviceToHost)
                       .ok()) {
                return sc::JobResult::kUnknown;
              }
              at += bytes;
            }
            return sc::JobResult::kQueued;
          },
          "reading the n-gram hash", stream_);
      !r) {
    CheckFailedJob();
    return r;
  }
  const auto* values = static_cast<const std::int64_t*>(host);
  const std::span<const std::int64_t> m(values, profile_.ngram);
  const std::span<const std::int64_t> o(values + profile_.ngram, profile_.ple_heads());
  const std::span<const std::int64_t> v(values + profile_.ngram + profile_.ple_heads(),
                                        profile_.ple_heads());
  auto hash = md::CheckQwen38PleHash(profile_, m, o, v, table_.rows);
  if (!hash) {
    return std::unexpected(hash.error());
  }
  hash_ = std::move(*hash);
  if (dbinding_.selected_head()) {
    const auto count = static_cast<std::size_t>(dbinding_.draft_ids.ne[1]);
    const auto address = model_.places.mtp_resource(dbinding_.draft_ids.index);
    if (auto r = node_.Job(
            execution_,
            [this, address, count](providers::NativeStream stream) {
              return providers::CopyAsync(stream, draft_ids_host_, Pointer(address),
                                          count * sizeof(std::int32_t),
                                          providers::CopyKind::kDeviceToHost)
                             .ok()
                         ? sc::JobResult::kQueued
                         : sc::JobResult::kUnknown;
            },
            "reading the draft vocabulary", stream_);
        !r) {
      CheckFailedJob();
      return r;
    }
    if (auto checked = md::CheckQwen38DraftIds(
            {static_cast<const std::int32_t*>(draft_ids_host_), count}, profile_.vocab);
        !checked) {
      return checked;
    }
  }
  hash_checked_ = true;
  return {};
}

Status Qwen38Runner::Scrub(std::uint8_t value, bool slabs, bool dense) {
  if (cohort_faulted_ || released_) {
    return Error("the Qwen3.8 cohort requires retirement");
  }
  std::uint32_t count = 0;
  for (const PagedWeights::Range& r : unwritten_) {
    if (r.slab ? slabs : dense) {
      scrub_[2 * std::size_t{count}] = r.address;
      scrub_[(2 * std::size_t{count}) + 1] = r.bytes;
      ++count;
    }
  }
  const std::uint64_t* ranges = scrub_;
  auto scrubbed = node_.Job(
      execution_,
      [ranges, count, value](providers::NativeStream stream) {
        return kernels::paging::FillRanges(ranges, count, value, stream.handle)
                   ? sc::JobResult::kQueued
                   : sc::JobResult::kUnknown;
      },
      "filling the weights' unwritten bytes", stream_);
  if (!scrubbed) {
    CheckFailedJob();
  }
  return scrubbed;
}

Status Qwen38Runner::Clear() { return Clear(default_request_); }

Status Qwen38Runner::Clear(RequestState& request) {
  if (auto active = CheckActive(request); !active) {
    return active;
  }
  request.pending_rows = 0;
  request.verify_restores_streams = false;
  // Remove only this destination's state from the completed stream lease.
  // Shared storage and all other active initialized states remain protected.
  const SlotMask others = active_mask_ & ~(SlotMask{1} << request.slot);
  if (auto protected_others = RefreshClosures(others); !protected_others) {
    return protected_others;
  }
  const Status cleared = request.live.DiscardGrowingState(node_);
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

Status Qwen38Runner::Adopt(RequestState& request, std::span<const LiveState::Range> used) {
  if (released_ || cohort_faulted_) {
    return Error("the Qwen3.8 cohort requires retirement");
  }
  if (request.spilled || !request.live.extents().empty() || used.empty() ||
      ((active_mask_ & (SlotMask{1} << request.slot)) != 0 && node_.InRequest(stream_))) {
    return Error("a Qwen3.8 slot adopts a kept conversation only while it holds none");
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

std::expected<SlotMask, std::string> Qwen38Runner::RecoverInPlace(
    const std::function<void(std::uint32_t slot)>& before_discard) {
  if (released_) {
    return Error("the Qwen3.8 runner is released");
  }
  if (node_.InRequest(stream_)) {
    return Error("a request is still open on Qwen3.8's stream");
  }
  if (resources_.launch().faulted()) {
    return Error(
        "Qwen3.8's launch context faulted: a launch of unknown outcome is not proven "
        "retired");
  }
  const bool faulted = cohort_faulted_;
  cohort_faulted_ = false;
  active_mask_ = 1;
  SlotMask discarded = 0;
  for (RequestState* request : Requests()) {
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
    request->pending_rows = 0;
    request->verify_restores_streams = false;
    const Status cleared = request->live.DiscardGrowingState(node_);
    request->track.Lost();
    if (!cleared) {
      FaultCohort();
      return Error(std::format("discarding a Qwen3.8 slot's state: {}", cleared.error()));
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

std::string Qwen38Runner::kept_layout() const {
  // Every slot's state has slot 0's layout. Version 1 of the state format:
  // a change to what a region's bytes mean bumps it.
  const LiveState& live = default_request_.live;
  std::string layout =
      std::format("qwen38-state/1;context={};mtp={};regions=", o_.context, speculative() ? 1 : 0);
  for (std::size_t i = 0; i < live.regions(); ++i) {
    layout += std::format("{}{}:{}:{}", i == 0 ? "" : ",", live.region_name(i), live.bytes(i),
                          live.mapped_bytes(i));
  }
  return layout;
}

Status Qwen38Runner::ClearIdle(RequestState& request) {
  if (released_ || cohort_faulted_) {
    return Error("the Qwen3.8 cohort requires retirement");
  }
  // Leased by the request open on the stream: cleared within its own
  // request (Clear). Selected last but with no request open, it is idle.
  if ((active_mask_ & (SlotMask{1} << request.slot)) != 0 && node_.InRequest(stream_)) {
    return Error("an active Qwen3.8 slot is cleared within its own request");
  }
  request.pending_rows = 0;
  request.verify_restores_streams = false;
  // Not selected, so outside the stream's lease, which keeps protecting
  // every active state; the catalog refuses the discard if anything still
  // holds or operates on this state's extents (DiscardGrowingState).
  const Status cleared = request.live.DiscardGrowingState(node_);
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

Status Qwen38Runner::CheckResident(const RequestState& request) const {
  if (auto active = CheckActive(request); !active) {
    return active;
  }
  if (request.spilled) {
    return Error("the Qwen3.8 slot's state is spilled: restore it first");
  }
  return {};
}

Status Qwen38Runner::Spill(RequestState& request) {
  if (released_ || cohort_faulted_) {
    return Error("the Qwen3.8 cohort requires retirement");
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
  // (its peers stay protected); then every initialized extent written back
  // to its place in the slot's spill file and its backing released (the
  // swap's write-back path). A commit or restore owed by its last verify
  // stays owed: it works from the slot's own runtime memory, not spilled.
  std::vector<ExtentId> written;
  std::vector<ExtentId> unchanged;
  SplitForSpill(request, written, unchanged);
  request.spilled = true;
  if (auto refreshed = RefreshClosures(); !refreshed) {
    request.spilled = false;
    return refreshed;
  }
  // Only what changed since the spill file last held it is written.
  for (const auto& [extents_of, unchanged_word] :
       {std::pair{&written, false}, std::pair{&unchanged, true}}) {
    if (extents_of->empty()) {
      continue;
    }
    if (auto evicted = node_.Evict(*extents_of, {.unchanged = unchanged_word}); !evicted) {
      request.track.Lost();
      return evicted;
    }
  }
  request.track.Saved();
  return {};
}

void Qwen38Runner::SplitForSpill(const RequestState& request, std::vector<ExtentId>& written,
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

std::uint64_t Qwen38Runner::SpillWriteBytes(const RequestState& request) const {
  if (request.spilled) {
    return 0;
  }
  std::vector<ExtentId> written;
  std::vector<ExtentId> unchanged;
  SplitForSpill(request, written, unchanged);
  return written.size() * kPagedExtent;
}

Status Qwen38Runner::Restore(RequestState& request) {
  request.state_refused = false;
  if (!request.spilled) {
    return {};
  }
  if (released_ || cohort_faulted_) {
    return Error("the Qwen3.8 cohort requires retirement");
  }
  if (!request.adopted.empty()) {
    // Kept from the process before (D-105): its extents are fresh here, and
    // their first load reads them from the named spill file, which holds
    // them (LiveState::Use), under the budget: a clean refusal leaves it
    // spilled for the runtime to make room.
    bool over_budget = false;
    if (auto used = request.live.Use(node_, request.adopted, nullptr, &over_budget); !used) {
      request.state_refused = over_budget && !cohort_faulted_;
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
              return Error("a spilled Qwen3.8 state is no longer cataloged");
            }
            closure = std::move(*of);
            return {};
          },
          "describing a spilled Qwen3.8 state");
      !described) {
    return described;
  }
  // Read back at the generations it was written at, under the budget: a
  // clean refusal leaves it spilled for the runtime to make room.
  sc::AcquireReport report;
  bool over_budget = false;
  if (auto acquired =
          node_.Acquire(closure, report, "restoring a spilled Qwen3.8 state", &over_budget);
      !acquired) {
    request.state_refused = over_budget && !cohort_faulted_;
    return acquired;
  }
  request.spilled = false;
  return RefreshClosures();
}

Status Qwen38Runner::EnsureState(std::uint32_t positions) {
  return EnsureState(default_request_, positions);
}

std::uint32_t Qwen38Runner::DecodeReadAlign() const { return WaveReadAlign(o_.wave_slots); }

std::uint32_t Qwen38Runner::WaveReadAlign(std::size_t slots) const {
  return slots > 1 ? std::max(o_.wave_read_align, kLoneReadAlign) : kLoneReadAlign;
}

std::expected<std::vector<LiveState::Range>, std::string> Qwen38Runner::StateRanges(
    std::uint32_t positions, std::uint32_t read_align) const {
  // A wave's graphs read the caches through a coarser alignment than the
  // positions need (WaveReadAlign): the cells they read must be backed.
  auto needed = md::Qwen38UsedState(profile_, layout_, positions, read_align);
  if (!needed) {
    return std::unexpected(needed.error());
  }
  std::vector<LiveState::Range> ranges;
  for (const auto& range : *needed) {
    ranges.push_back({.region = kTarget, .offset = range.offset, .bytes = range.bytes});
  }
  if (speculative()) {
    const auto cells = std::min<std::uint64_t>(mtp_layout_.cells, Round(positions, read_align));
    const std::uint64_t kv_row = std::uint64_t{profile_.head_dim} * profile_.kv_heads * 2;
    const std::uint64_t indexer_row = std::uint64_t{profile_.indexer_head_dim} * 4;
    const std::uint64_t blocks = (cells + profile_.indexer_ratio - 1) / profile_.indexer_ratio;
    ranges.push_back({.region = kDrafter, .offset = mtp_layout_.k, .bytes = cells * kv_row});
    ranges.push_back({.region = kDrafter, .offset = mtp_layout_.v, .bytes = cells * kv_row});
    ranges.push_back(
        {.region = kDrafter, .offset = mtp_layout_.indexer, .bytes = cells * indexer_row});
    ranges.push_back({.region = kDrafter,
                      .offset = mtp_layout_.blocks,
                      .bytes = blocks * profile_.indexer_head_dim * 2});
    ranges.push_back({.region = kDrafter,
                      .offset = mtp_layout_.hidden,
                      .bytes = mtp_layout_.bytes - mtp_layout_.hidden});
  }
  return ranges;
}

std::expected<std::uint64_t, std::string> Qwen38Runner::StateBytesThrough(
    std::uint32_t positions) const {
  auto ranges = StateRanges(std::min(positions, o_.context), DecodeReadAlign());
  if (!ranges) {
    return std::unexpected(ranges.error());
  }
  return live_.UsedBytesOf(*ranges);
}

Status Qwen38Runner::EnsureState(RequestState& request, std::uint32_t positions,
                                 std::uint32_t read_align) {
  if (auto active = CheckResident(request); !active) {
    return active;
  }
  auto ranges = StateRanges(positions, read_align);
  if (!ranges) {
    return std::unexpected(ranges.error());
  }
  bool over_budget = false;
  auto used = request.live.Use(node_, *ranges, &execution_, &over_budget);
  if (!used) {
    if (auto refreshed = RefreshClosures(); !refreshed) {
      request.live.Quarantine();
      return Error(std::format("{}; {}", used.error(), refreshed.error()));
    }
    // Only a clean refusal that left the state usable and the cohort
    // healthy may be retried once leased state is freed (Slot::state_refused).
    request.state_refused = over_budget && !cohort_faulted_ && !request.live.quarantined();
    return std::unexpected(used.error());
  }
  return *used ? RefreshClosures() : Status{};
}

std::expected<Qwen38Runner::ChunkPlans::Entry*, std::string> Qwen38Runner::Planned(
    const ChunkKey& key) {
  return Planned(default_request_, key);
}

std::expected<Qwen38Runner::ChunkPlans::Entry*, std::string> Qwen38Runner::Planned(
    RequestState& request, const ChunkKey& key) {
  if (ChunkPlans::Entry* found = request.plans.Find(key); found != nullptr) {
    return found;
  }
  const auto start = std::chrono::steady_clock::now();
  kg::LaunchContext& launch = resources_.launch();
  auto planned = PlanQwen38Chunk(request.model, key.shape, kg::DeviceChoicesOf(launch),
                                 node_.activations().base, node_.activations().bytes, {}, key.kind);
  if (!planned) {
    return std::unexpected(planned.error());
  }
  if (auto r = BindPlanned(**planned, launch, resources_.registry(), "the plan"); !r) {
    return std::unexpected(r.error());
  }
  Check((*planned)->graph);
  const double seconds = Seconds(std::chrono::steady_clock::now() - start);
  plan_seconds_ += seconds;
  const std::uint64_t bytes = PlannedHostBytes(**planned);
  const std::uint64_t nodes = PlannedNodes(**planned);
  return &request.plans.Add(key, std::move(*planned), bytes, nodes, seconds);
}

std::expected<Qwen38Runner::MtpPlans::Entry*, std::string> Qwen38Runner::PlannedMtp(
    const kg::Qwen38MtpShape& shape) {
  return PlannedMtp(default_request_, shape);
}

std::expected<Qwen38Runner::MtpPlans::Entry*, std::string> Qwen38Runner::PlannedMtp(
    RequestState& request, const kg::Qwen38MtpShape& shape) {
  if (MtpPlans::Entry* found = request.mplans.Find(shape); found != nullptr) {
    return found;
  }
  const auto start = std::chrono::steady_clock::now();
  kg::LaunchContext& launch = resources_.launch();
  auto planned = PlanQwen38Mtp(request.model, shape, kg::DeviceChoicesOf(launch),
                               node_.activations().base, node_.activations().bytes);
  if (!planned) {
    return std::unexpected(planned.error());
  }
  if (auto r = BindPlanned(**planned, launch, resources_.registry(), "the drafter"); !r) {
    return std::unexpected(r.error());
  }
  CheckMtp((*planned)->graph);
  const double seconds = Seconds(std::chrono::steady_clock::now() - start);
  plan_seconds_ += seconds;
  const std::uint64_t bytes = PlannedHostBytes(**planned);
  const std::uint64_t nodes = PlannedNodes(**planned);
  return &request.mplans.Add(shape, std::move(*planned), bytes, nodes, seconds);
}

// BP-A1's check (planned.h): the state is live state (the target's, the
// drafter's caches and streams), the verify's saves runtime, and the row
// slots and activations scratch.
void Qwen38Runner::Check(const kg::Qwen38Graph& graph) {
  std::vector<const ggml_tensor*> state;
  std::vector<const ggml_tensor*> saves;
  for (const kg::Qwen38LayerTensors& l : graph.layers) {
    for (const ggml_tensor* t : {l.cache_k, l.cache_v, l.cache_idx, l.cache_pool, l.conv_state,
                                 l.recurrent, l.ple_state}) {
      if (t != nullptr) {
        state.push_back(t);
      }
    }
    for (const ggml_tensor* t :
         {l.commit_conv, l.commit_qkv, l.commit_gate, l.commit_beta, l.commit_ple}) {
      if (t != nullptr) {
        saves.push_back(t);
      }
    }
  }
  if (graph.streams != nullptr) {
    state.push_back(graph.streams);
  }
  const std::array<const ggml_tensor*, 1> slots = {graph.ple_table};
  const auto inputs = graph.inputs();
  CheckCoverage(node_, owner_, graph.nodes,
                {.state = state,
                 .runtime = saves,
                 .scratch = slots,
                 .inputs = inputs,
                 .fill_reads_nothing = true},
                coverage_);
}

void Qwen38Runner::CheckMtp(const kg::Qwen38MtpGraph& graph) {
  const std::array<const ggml_tensor*, 5> state = {graph.layer.cache_k, graph.layer.cache_v,
                                                   graph.layer.cache_idx, graph.layer.cache_pool,
                                                   graph.streams};
  const auto inputs = graph.inputs();
  CheckCoverage(
      node_, owner_, graph.nodes,
      {.state = state, .inputs = inputs, .fill_reads_nothing = true, .what = "the drafter's "},
      coverage_);
}

std::expected<std::vector<std::int32_t>, std::string> Qwen38Runner::ReadRows(
    const md::Qwen38ChunkInputs& in) {
  // The chunk's n-gram rows: planned, read and their slots' sources set,
  // while no job of this model holds the landing (the last one's fence
  // has completed: Job returns only after it).
  const auto planning = std::chrono::steady_clock::now();
  auto rows_plan = PlanPleRows(table_, in.ple_rows, landing_bytes_, slots_);
  const auto ple_seconds = Seconds(std::chrono::steady_clock::now() - planning);
  if (!rows_plan) {
    return std::unexpected(rows_plan.error());
  }
  // The gather's grid is the chunk's lookups (rows × heads) and the device
  // reads the count: it must never exceed them, nor the slots.
  if (rows_plan->sources.size() > in.ple_rows.size() || rows_plan->sources.size() > slots_) {
    return std::unexpected(std::format("{} n-gram rows for {} lookups and {} slots",
                                       rows_plan->sources.size(), in.ple_rows.size(), slots_));
  }
  if (auto read = ReadRows(*rows_plan, in.ple_rows.size()); !read) {
    return std::unexpected(read.error());
  }
  ple_.seconds += ple_seconds;
  return std::move(rows_plan->slots);
}

Status Qwen38Runner::ReadRows(const PleRowPlan& planned, std::size_t lookups) {
  const auto reading = std::chrono::steady_clock::now();
  if (planned.sources.size() > lookups || planned.sources.size() > slots_ ||
      planned.sources.size() > std::numeric_limits<std::uint32_t>::max()) {
    return Error("the Qwen3.8 union row gather exceeds its provisioned slots");
  }
  if (auto r = ReadPleRows(*ring_, table_.fd, planned, landing_); !r) {
    rows_stalled_ = ring_->in_flight() != 0;  // reads that may still land
    return std::unexpected(r.error());
  }
  std::ranges::copy(planned.sources, sources_);
  *ple_count_ = static_cast<std::uint32_t>(planned.sources.size());
  ple_.chunks += 1;
  ple_.lookups += lookups;
  ple_.rows += planned.sources.size();
  ple_.reads += planned.reads.size();
  ple_.read_bytes += planned.landing_bytes;
  ple_.useful_bytes += planned.useful_bytes;
  ple_.extent_bytes += planned.extents * kExtent;
  ple_.seconds += Seconds(std::chrono::steady_clock::now() - reading);
  return {};
}

std::function<bool(void* stream)> Qwen38Runner::Gather(std::uint32_t rows) {
  // The gather's grid: the shape's lookups, a slot each at most.
  const auto max_count = static_cast<std::uint32_t>(std::uint64_t{rows} * profile_.ple_heads());
  return [this, max_count](void* stream) {
    return kernels::paging::GatherPleRows(
        landing_, sources_, ple_count_, max_count, static_cast<std::uint32_t>(table_.row_bytes),
        static_cast<std::byte*>(Pointer(slot_memory_.base)), stream);
  };
}

void Qwen38Runner::Settle(bool saved, bool wrote, bool unknown) {
  Settle(default_request_, saved, wrote, unknown);
}

void Qwen38Runner::Settle(RequestState& request, bool saved, bool wrote, bool unknown) {
  // An undone verify wrote no other target state, but its streams rows may
  // have overwritten the ones the next draft would catch up on (rows 1 ..):
  // none is pending until a chunk with the injection or an accepted verify
  // writes them again.
  const bool uncertain = unknown || resources_.launch().faulted();
  request.verify_restores_streams = false;
  if (request.live.Settle(saved, wrote, uncertain)) {
    request.pending_rows = 0;
  }
  if (uncertain) {
    FaultCohort();
  }
}

Status Qwen38Runner::Usable() const { return Usable(default_request_); }

Status Qwen38Runner::Usable(const RequestState& request) const {
  if (auto active = CheckResident(request); !active) {
    return active;
  }
  if (!hash_checked_) {
    return Error("the n-gram hash is not checked since the last load");
  }
  if (rows_stalled_) {
    return Error("the n-gram rows' reads stalled earlier; their landing may still be written");
  }
  return request.live.Usable();
}

std::expected<std::pair<kg::Qwen38MtpShape, std::vector<md::Qwen38ChunkInputs>>, std::string>
Qwen38Runner::MtpInputs(std::uint32_t first, std::uint32_t rows, std::uint32_t passes, bool head,
                        std::int64_t hidden_row, bool confidence, bool capture_head,
                        std::uint32_t read_align) const {
  const std::uint64_t end = std::uint64_t{first} + rows + passes - 1;
  if (rows == 0 || passes == 0 || end > mtp_layout_.context) {
    return Error(std::format("a draft of {} passes after {} rows at {} passes the context", passes,
                             rows, first));
  }
  const auto n_kv = static_cast<std::uint32_t>(
      std::min<std::uint64_t>(Round(end, read_align), mtp_layout_.cells));
  std::vector<md::Qwen38ChunkInputs> ins;
  for (std::uint32_t p = 0; p < passes; ++p) {
    auto in = md::Qwen38Rows(profile_, mtp_layout_.cells, p == 0 ? first : first + rows + p - 1,
                             p == 0 ? rows : 1, n_kv, false);
    if (!in) {
      return std::unexpected(in.error());
    }
    ins.push_back(std::move(*in));
  }
  const kg::Qwen38MtpShape shape{
      .rows = rows,
      .passes = passes,
      .n_kv = n_kv,
      .cells = mtp_layout_.cells,
      .qsa_select = ins.front().qsa_select,
      .qsa_blocks = ins.front().qsa_select ? ins.front().qsa.blocks : 0,
      .head = head,
      .head_rows = static_cast<std::int64_t>(
          dbinding_.selected_head()
              ? std::min<std::uint64_t>(o_.draft_vocab, dbinding_.draft_ids.ne[1])
              : o_.draft_vocab),
      .confidence = confidence,
      .capture_head = capture_head,
      .hidden_row = hidden_row,
      .hidden_rows = mtp_layout_.hidden_rows};
  return std::pair{shape, std::move(ins)};
}

Status Qwen38Runner::CheckWaveSlot(const Slot* slot, std::uint32_t previous) const {
  if (!waves_provisioned() || slot == nullptr || &slot->owner_ != this ||
      slot->index() >= slot_count_ || slot->index() + 1 <= previous) {
    return Error("Qwen3.8 waves need provisioned, owned, strictly ascending slots");
  }
  if (auto usable = Usable(slot->request_); !usable) {
    return usable;
  }
  return slot->request_.live.AwaitingAccept();
}

Status Qwen38Runner::CheckWaveSources(std::span<const std::pair<ggml_tensor*, const void*>> sources,
                                      const Qwen38WavePlanned& planned) const {
  if (sources.size() != planned.inputs().size()) {
    return Error("the Qwen3.8 wave does not stage every planned input");
  }
  std::uint64_t staged = 0;
  for (const auto& [tensor, source] : sources) {
    if (tensor == nullptr || tensor->data == nullptr || source == nullptr ||
        std::ranges::count(planned.inputs(), tensor) != 1 ||
        std::ranges::count_if(sources,
                              [tensor](const auto& entry) { return entry.first == tensor; }) != 1) {
      return Error("the Qwen3.8 wave repeats or omits an input source");
    }
    const std::uint64_t bytes = ggml_nbytes(tensor);
    if (bytes == 0 || bytes > runs_.staging_bytes() ||
        Round(bytes, 256) > runs_.staging_bytes() - staged) {
      return Error("the Qwen3.8 wave inputs exceed their staging cap");
    }
    staged += Round(bytes, 256);
    const auto& region = node_.activations();
    const auto at = Address(tensor->data);
    if (at < region.base || at - region.base > region.bytes ||
        bytes > region.bytes - (at - region.base)) {
      return Error("the Qwen3.8 wave input is outside its placed activation allocation");
    }
  }
  if (staged != planned.inputs_bytes) {
    return Error("the Qwen3.8 wave staging differs from its measured input bytes");
  }
  return {};
}

Status Qwen38Runner::CheckWaveOutput(const ggml_tensor* tensor, ggml_type type,
                                     std::uint64_t bytes) const {
  if (tensor == nullptr || tensor->data == nullptr || tensor->type != type ||
      !ggml_is_contiguous(tensor) || ggml_nbytes(tensor) != bytes) {
    return Error("the Qwen3.8 wave output has an invalid packed shape");
  }
  const auto& region = node_.activations();
  const auto at = Address(tensor->data);
  if (at < region.base || at - region.base > region.bytes ||
      bytes > region.bytes - (at - region.base)) {
    return Error("the Qwen3.8 wave output exceeds its retained placed allocation");
  }
  return {};
}

std::expected<Qwen38Runner::TargetWaves::Entry*, std::string> Qwen38Runner::PlannedWave(
    const TargetWaveKey& key) {
  if (auto* found = target_waves_.Find(key); found != nullptr) {
    return found;
  }
  if (auto places = CheckPlaces(); !places) {
    return std::unexpected(places.error());
  }
  const auto start = std::chrono::steady_clock::now();
  std::vector<Qwen38TargetWaveInput> inputs;
  for (const RequestState* request : Requests()) {
    const auto s = request->slot;
    if ((key.mask & (SlotMask{1} << s)) != 0) {
      inputs.push_back({s, &request->model, key.slots[s].shape, key.slots[s].kind});
    }
  }
  auto& launch = resources_.launch();
  auto planned = PlanQwen38TargetWave(inputs, kg::DeviceChoicesOf(launch),
                                      {.activations = node_.activations().base,
                                       .bytes = node_.activations().bytes,
                                       .paired = key.paired,
                                       .share_target_head = true});
  if (!planned) {
    return std::unexpected(planned.error());
  }
  if (auto bound = BindPlanned(**planned, launch, resources_.registry(), "the target wave");
      !bound) {
    return std::unexpected(bound.error());
  }
  const double seconds = Seconds(std::chrono::steady_clock::now() - start);
  plan_seconds_ += seconds;
  auto owner = std::make_unique<WaveCacheOwner>();
  owner->plan = std::move(*planned);
  const std::uint64_t bytes = owner->plan->host_bytes();
  const std::uint64_t nodes = PlannedNodes(*owner->plan);
  return &target_waves_.Add(key, std::move(owner), bytes, nodes, seconds);
}

std::expected<Qwen38Runner::DraftWaves::Entry*, std::string> Qwen38Runner::PlannedWave(
    const DraftWaveKey& key) {
  if (auto* found = draft_waves_.Find(key); found != nullptr) {
    return found;
  }
  if (auto places = CheckPlaces(); !places) {
    return std::unexpected(places.error());
  }
  const auto start = std::chrono::steady_clock::now();
  std::vector<Qwen38DraftWaveInput> inputs;
  for (const RequestState* request : Requests()) {
    const auto s = request->slot;
    if ((key.mask & (SlotMask{1} << s)) != 0) {
      inputs.push_back({s, &request->model, key.slots[s]});
    }
  }
  auto& launch = resources_.launch();
  auto planned = PlanQwen38DraftWave(inputs, kg::DeviceChoicesOf(launch),
                                     {.activations = node_.activations().base,
                                      .bytes = node_.activations().bytes,
                                      .paired = key.paired});
  if (!planned) {
    return std::unexpected(planned.error());
  }
  if (auto bound = BindPlanned(**planned, launch, resources_.registry(), "the draft wave");
      !bound) {
    return std::unexpected(bound.error());
  }
  const double seconds = Seconds(std::chrono::steady_clock::now() - start);
  plan_seconds_ += seconds;
  auto owner = std::make_unique<WaveCacheOwner>();
  owner->plan = std::move(*planned);
  const std::uint64_t bytes = owner->plan->host_bytes();
  const std::uint64_t nodes = PlannedNodes(*owner->plan);
  return &draft_waves_.Add(key, std::move(owner), bytes, nodes, seconds);
}

Status Qwen38Runner::CheckWave(const Qwen38WavePlanned& planned) {
  std::vector<const ggml_tensor*> state;
  std::vector<const ggml_tensor*> saves;
  std::vector<const ggml_tensor*> scratch;
  for (std::size_t s = 0; s < kRequestSlots; ++s) {
    if (const auto* g = planned.target(s); g != nullptr) {
      for (const auto& l : g->layers) {
        for (const ggml_tensor* t : {l.cache_k, l.cache_v, l.cache_idx, l.cache_pool, l.conv_state,
                                     l.recurrent, l.ple_state}) {
          if (t != nullptr) {
            state.push_back(t);
          }
        }
        for (const ggml_tensor* t :
             {l.commit_conv, l.commit_qkv, l.commit_gate, l.commit_beta, l.commit_ple}) {
          if (t != nullptr) {
            saves.push_back(t);
          }
        }
      }
      if (g->streams != nullptr) {
        state.push_back(g->streams);
      }
      scratch.push_back(g->ple_table);
    }
    if (const auto* g = planned.draft(s); g != nullptr) {
      for (const ggml_tensor* t : {g->layer.cache_k, g->layer.cache_v, g->layer.cache_idx,
                                   g->layer.cache_pool, g->streams}) {
        if (t != nullptr) {
          state.push_back(t);
        }
      }
    }
  }
  const auto before = coverage_.violations;
  CheckCoverage(node_, owner_, planned.nodes(),
                {.state = state,
                 .runtime = saves,
                 .scratch = scratch,
                 .inputs = planned.inputs(),
                 .fill_reads_nothing = true,
                 .what = "the wave's "},
                coverage_);
  if (coverage_.violations != before) {
    return Error(
        std::format("the Qwen3.8 wave failed catalog coverage: {}", coverage_.first_violation));
  }
  return {};
}

Status Qwen38Runner::ChunkWave(std::span<const ChunkWork> work, bool paired) {
  if (work.empty() || work.size() > kRequestSlots) {
    return Error(std::format("a Qwen3.8 chunk wave needs one to {} slots", kRequestSlots));
  }
  std::array<TargetWork, kRequestSlots> target{};
  for (std::size_t i = 0; i < work.size(); ++i) {
    target[i] = {work[i].slot, work[i].history, work[i].n_past, nullptr, work[i].logits};
  }
  return TargetWave(std::span(target).first(work.size()), false, paired);
}

Status Qwen38Runner::VerifyWave(std::span<const VerifyWork> work, bool paired) {
  if (work.empty() || work.size() > kRequestSlots) {
    return Error(std::format("a Qwen3.8 verify wave needs one to {} slots", kRequestSlots));
  }
  std::array<TargetWork, kRequestSlots> target{};
  for (std::size_t i = 0; i < work.size(); ++i) {
    target[i] = {work[i].slot, work[i].history, work[i].n_past, work[i].argmax, work[i].logits};
  }
  return TargetWave(std::span(target).first(work.size()), true, paired);
}

Status Qwen38Runner::TargetWave(std::span<const TargetWork> work, bool verify, bool paired) {
  const PlanStep step;  // the plans this wave borrows stay until its job ends
  if (work.empty() || work.size() > o_.wave_slots || (verify && !speculative())) {
    return Error("the Qwen3.8 target wave exceeds its provisioned cohort or lacks a drafter");
  }
  for (const TargetWork& w : work) {
    if (w.slot != nullptr && &w.slot->owner_ == this) {
      w.slot->request_.track.Wrote(w.n_past > 0 ? w.n_past - 1 : 0);
    }
  }
  struct Save {
    std::uint64_t address, bytes;
    std::int64_t row;
  };
  struct Frame {
    RequestState* request = nullptr;
    md::Qwen38ChunkInputs in;
    Qwen38HostInputs host;
    std::vector<Save> saves;
    bool saved = false;
  };
  // Host-source owners cannot move after Sources takes pointers to their
  // scalar members/vectors. Their lifetime includes Stage and the whole Job.
  std::array<std::unique_ptr<Frame>, kRequestSlots> frames;
  TargetWaveKey key;
  key.paired = paired;
  std::vector<std::int32_t> ple_rows;
  std::uint32_t previous = 0;
  std::uint32_t total_rows = 0;
  for (const auto& w : work) {
    if (auto valid = CheckWaveSlot(w.slot, previous); !valid) {
      return valid;
    }
    const auto s = w.slot->index();
    previous = s + 1;
    if (w.history.size() <= w.n_past || w.history.size() > o_.context ||
        w.history.size() - w.n_past > std::min<std::size_t>(4, o_.max_rows) ||
        (verify &&
         (w.argmax == nullptr || w.history.size() - w.n_past > std::size_t{o_.draft_rows} + 1)) ||
        (!verify && w.logits == nullptr)) {
      return Error("a Qwen3.8 target wave has invalid rows, context or outputs");
    }
    for (const auto& other : work) {
      if (&other != &w && ((w.logits != nullptr && w.logits == other.logits) ||
                           (w.argmax != nullptr && w.argmax == other.argmax))) {
        return Error("Qwen3.8 target wave outputs must have independent owners");
      }
    }
    auto in = md::Qwen38Chunk(profile_, layout_, hash_, w.history, w.n_past,
                              static_cast<std::uint32_t>(w.history.size() - w.n_past), false,
                              WaveReadAlign(work.size()));
    if (!in) {
      return std::unexpected(in.error());
    }
    auto needed =
        md::Qwen38UsedState(profile_, layout_, static_cast<std::uint32_t>(w.history.size()));
    if (!needed) {
      return std::unexpected(needed.error());
    }
    auto& f = frames[s] = std::make_unique<Frame>();
    f->request = &w.slot->request_;
    f->in = std::move(*in);
    const auto rows = f->in.rows;
    total_rows += rows;
    ple_rows.insert(ple_rows.end(), f->in.ple_rows.begin(), f->in.ple_rows.end());
    key.mask |= SlotMask{1} << s;
    if (w.logits != nullptr) {
      key.logits |= SlotMask{1} << s;
    }
    key.slots[s] = {.shape = kg::Qwen38ShapeOf(layout_, f->in, verify ? rows : 1),
                    .kind = {.verify = verify, .export_streams = verify}};
    if (verify) {
      using K = md::Qwen38StateTensor::Kind;
      const auto base = f->request->live.base(kTarget);
      for (std::uint32_t i = 0; i < rows; ++i) {
        const auto cell = std::uint64_t{w.n_past} + i;
        for (const auto& t : layout_.tensors) {
          if (t.kind == K::kK || t.kind == K::kV || t.kind == K::kIndexerK) {
            const auto bytes = t.ne0 * (t.f16 ? 2U : 4U);
            f->saves.push_back({base + t.offset + (cell * bytes), bytes, i});
          }
        }
      }
      const auto ratio = profile_.indexer_ratio;
      for (std::uint64_t b = w.n_past / ratio; b < (std::uint64_t{w.n_past} + rows) / ratio; ++b) {
        const auto row = static_cast<std::int64_t>(((b + 1) * ratio) - 1 - w.n_past);
        for (const auto& t : layout_.tensors) {
          if (t.kind == K::kIndexerBlocks) {
            const auto bytes = t.ne0 * 2;
            f->saves.push_back({base + t.offset + (b * bytes), bytes, row});
          }
        }
      }
      const auto offset = Round(commit_layout_.bytes, 256);
      const auto& commit = f->request->commit;
      const std::uint64_t stream_row = std::uint64_t{profile_.hc_width()} * 4;
      const auto stream_base = f->request->live.base(kDrafter) + mtp_layout_.hidden;
      for (std::uint32_t i = 0; i < rows; ++i) {
        f->saves.push_back({stream_base + (std::uint64_t{i + 1} * stream_row), stream_row, i});
      }
      if (commit.bytes < offset || f->saves.size() > kRangeCapacity) {
        return Error("the Qwen3.8 wave exceeds a slot's snapshot geometry");
      }
      std::uint64_t saved_bytes = 0;
      for (const auto& save : f->saves) {
        const auto within = [&](std::uint64_t at, std::uint64_t bytes) {
          return save.address >= at && save.address - at <= bytes &&
                 save.bytes <= bytes - (save.address - at);
        };
        if ((!within(base, layout_.bytes) &&
             !within(stream_base, std::uint64_t{mtp_layout_.hidden_rows} * stream_row)) ||
            saved_bytes > commit.bytes - offset ||
            Round(save.bytes, 256) > commit.bytes - offset - saved_bytes) {
          return Error("the Qwen3.8 wave exceeds a slot's snapshot capacity");
        }
        saved_bytes += Round(save.bytes, 256);
      }
    }
  }
  // No state growth, snapshot mutation or submission has occurred. One
  // union row plan preserves every request's lookup slice, deduplicating
  // actual file rows once; gathering into shared immutable slots is paid.
  const auto planning = std::chrono::steady_clock::now();
  auto rows_plan = PlanPleRows(table_, ple_rows, landing_bytes_, slots_);
  const auto ple_seconds = Seconds(std::chrono::steady_clock::now() - planning);
  if (!rows_plan) {
    return std::unexpected(rows_plan.error());
  }
  if (rows_plan->slots.size() != ple_rows.size() || rows_plan->sources.size() > ple_rows.size() ||
      rows_plan->sources.size() > slots_) {
    return Error("the Qwen3.8 union row plan omitted a lookup or exceeded its gather capacity");
  }
  auto cached = PlannedWave(key);
  if (!cached) {
    return std::unexpected(cached.error());
  }
  auto& entry = **cached;
  auto& planned = *entry.planned->plan;
  std::vector<std::pair<ggml_tensor*, const void*>> sources;
  Copies outputs;
  std::size_t row_at = 0;
  for (const auto& w : work) {
    const auto s = w.slot->index();
    auto& f = *frames[s];
    const auto& g = *planned.target(s);
    if (f.in.qsa_select && (g.mask != nullptr || g.mask_f32 != nullptr)) {
      auto masked = md::Qwen38Chunk(profile_, layout_, hash_, w.history, w.n_past, f.in.rows, true,
                                    WaveReadAlign(work.size()));
      if (!masked) {
        return std::unexpected(masked.error());
      }
      f.in = std::move(*masked);
    }
    const auto count = f.in.ple_rows.size();
    Qwen38Sources(g, f.in, verify ? f.in.rows : 1,
                  std::span(rows_plan->slots).subspan(row_at, count), f.host, 1);
    row_at += count;
    sources.insert(sources.end(), f.host.sources.begin(), f.host.sources.end());
    const auto words = std::uint64_t{verify ? f.in.rows : 1} * profile_.vocab;
    if (words > wave_logit_words_) {
      return Error("the Qwen3.8 wave exceeds its fixed output slice");
    }
    if (auto valid = CheckWaveOutput(g.logits, GGML_TYPE_F32, words * sizeof(float)); !valid) {
      return valid;
    }
    if (w.logits != nullptr) {
      outputs.push_back({Address(wave_logits_ + (s * wave_logit_words_)), Address(g.logits->data),
                         words * sizeof(float)});
    }
    if (verify) {
      const auto bytes = std::uint64_t{f.in.rows} * sizeof(std::int32_t);
      if (auto valid = CheckWaveOutput(g.argmax, GGML_TYPE_I32, bytes); !valid) {
        return valid;
      }
      outputs.push_back(
          {Address(wave_ids_ + (std::size_t{s} * 8)), Address(g.argmax->data), bytes});
    }
  }
  if (auto valid = CheckWaveSources(sources, planned); !valid) {
    return valid;
  }
  for (const auto& w : work) {
    if (auto used = EnsureState(w.slot->request_, static_cast<std::uint32_t>(w.history.size()),
                                WaveReadAlign(work.size()));
        !used) {
      return used;  // partial growth stays initialized/protected; no graph ran
    }
  }
  if (!entry.planned->covered) {
    if (auto covered = CheckWave(planned); !covered) {
      return covered;
    }
    entry.planned->covered = true;
  }
  if (auto read = ReadRows(*rows_plan, ple_rows.size()); !read) {
    return read;
  }
  ple_.seconds += ple_seconds;
  auto copies = runs_.Stage(sources, 0);
  if (!copies) {
    return std::unexpected(copies.error());
  }
  if (verify) {
    for (const auto& w : work) {
      auto& f = *frames[w.slot->index()];
      f.request->live.BeginSaves();
      f.request->verify_restores_streams = false;
      for (const auto& save : f.saves) {
        if (auto added = f.request->live.Save(save.address, save.bytes, save.row); !added) {
          // Host preflight already proved the exact count/packed capacity;
          // an unexpected refusal invalidates the cohort before submission.
          FaultCohort();
          return added;
        }
      }
    }
  }
  auto& launch = resources_.launch();
  auto& run = entry.runs[0];
  bool capture =
      run.CaptureDue(runs_.graphs()) && (verify || std::ranges::all_of(work, [](const auto& w) {
                                           return w.history.size() - w.n_past == 1;
                                         }));
  if (capture && !target_waves_.ChargeGraph(entry)) {
    capture = false;  // no room for its graph even after a reclaim: launch by launch
  }
  const auto gather = Gather(total_rows);
  Status ran;
  RunPath path = RunPath::kEager;
  bool unknown = false;
  bool wrote = false;
  bool committing = false;
  auto job = [&](providers::NativeStream native) -> sc::JobResult {
    for (const auto& w : work) {
      auto& f = *frames[w.slot->index()];
      committing = committing || f.request->live.owed();
      if (auto owed = f.request->live.QueueOwed(launch); !owed) {
        ran = Error(owed.error().detail);
        unknown = true;
        return sc::JobResult::kUnknown;
      }
    }
    if (verify) {
      for (const auto& w : work) {
        auto& f = *frames[w.slot->index()];
        if (auto saved = f.request->live.QueueSaves(launch); !saved) {
          ran = Error(saved.error().detail);
          unknown = true;
          return sc::JobResult::kUnknown;
        }
        f.saved = true;
      }
    }
    const auto queued =
        runs_.Queue(run, *copies, gather, *planned.bound, outputs, capture, graph_stats_, native);
    path = queued.path;
    wrote = queued.before || queued.result.has_value();
    if (!queued.result) {
      ran = Error(queued.result.error().detail);
      unknown = queued.result.error().error == kg::KernelError::kUnknown;
      if (unknown) {
        return sc::JobResult::kUnknown;
      }
      return committing || verify || wrote ? sc::JobResult::kFailed : sc::JobResult::kNotStarted;
    }
    return sc::JobResult::kQueued;
  };
  const auto posted = node_.Job(execution_, std::move(job), "a Qwen3.8 target wave", stream_);
  if (!posted || !ran || launch.faulted()) {
    if (!posted) {
      CheckFailedJob();
    }
    for (const auto& w : work) {
      auto& f = *frames[w.slot->index()];
      Settle(*f.request, f.saved, !verify && wrote, unknown);
    }
    if (launch.faulted()) {
      return Error("the Qwen3.8 target wave launch context faulted");
    }
    return !ran ? ran : Error(posted.error());
  }
  Count(graph_stats_, path);
  last_wave_ = planned.stats();
  for (const auto& w : work) {
    const auto s = w.slot->index();
    auto& f = *frames[s];
    if (verify) {
      f.request->live.Verified(f.in.rows);
      f.request->verify_restores_streams = true;
      w.argmax->assign(wave_ids_ + (std::size_t{s} * 8),
                       wave_ids_ + (std::size_t{s} * 8) + f.in.rows);
    } else {
      f.request->pending_rows = 0;
    }
    if (w.logits != nullptr) {
      const auto* from = wave_logits_ + (s * wave_logit_words_);
      w.logits->assign(from, from + (std::size_t{verify ? f.in.rows : 1} * profile_.vocab));
    }
  }
  return {};
}

Status Qwen38Runner::DraftWave(std::span<const DraftWork> work, bool paired) {
  const PlanStep step;  // the plans this wave borrows stay until its job ends
  if (work.empty() || work.size() > o_.wave_slots || work.size() > kRequestSlots ||
      !speculative()) {
    return Error("a Qwen3.8 draft wave needs a provisioned cohort and drafter");
  }
  for (const DraftWork& w : work) {
    if (w.slot != nullptr && &w.slot->owner_ == this) {
      w.slot->request_.track.Wrote(
          w.history.size() > 1 ? static_cast<std::uint32_t>(w.history.size() - 2) : 0);
    }
  }
  struct Frame {
    RequestState* request = nullptr;
    std::vector<md::Qwen38ChunkInputs> ins;
    Qwen38MtpHostInputs host;
    std::uint32_t through = 0;
  };
  std::array<std::unique_ptr<Frame>, kRequestSlots> frames;
  DraftWaveKey key;
  key.paired = paired;
  std::uint32_t previous = 0;
  for (const auto& w : work) {
    if (auto valid = CheckWaveSlot(w.slot, previous); !valid) {
      return valid;
    }
    const auto s = w.slot->index();
    previous = s + 1;
    auto& request = w.slot->request_;
    const auto rows = request.pending_rows;
    const auto passes = w.passes == 0 ? o_.draft_rows : w.passes;
    if (w.drafts == nullptr || rows == 0 || rows > 4 || rows > o_.draft_rows + 1 || passes == 0 ||
        passes > 3 || passes > o_.draft_rows || w.history.size() < std::size_t{rows} + 1 ||
        w.history.size() > o_.context) {
      return Error("a Qwen3.8 draft wave has invalid pending rows, passes, context or outputs");
    }
    for (const auto& other : work) {
      if (&other != &w && (w.drafts == other.drafts || (w.probabilities != nullptr &&
                                                        w.probabilities == other.probabilities))) {
        return Error("Qwen3.8 draft wave outputs must have independent owners");
      }
    }
    const auto n = static_cast<std::uint32_t>(w.history.size() - 1);
    for (const auto t : w.history.subspan(n - rows + 1, rows)) {
      if (t < 0 || std::cmp_greater_equal(t, profile_.vocab)) {
        return Error("a Qwen3.8 draft wave token is outside the vocabulary");
      }
    }
    auto shaped = MtpInputs(n - rows, rows, passes, true, 1, w.probabilities != nullptr, false,
                            WaveReadAlign(work.size()));
    if (!shaped) {
      return std::unexpected(shaped.error());
    }
    const auto through = n + passes - 1;
    auto needed = md::Qwen38UsedState(profile_, layout_, through);
    if (!needed) {
      return std::unexpected(needed.error());
    }
    auto& f = frames[s] = std::make_unique<Frame>();
    f->request = &request;
    f->through = through;
    f->ins = std::move(shaped->second);
    key.slots[s] = shaped->first;
    key.mask |= SlotMask{1} << s;
  }
  auto cached = PlannedWave(key);
  if (!cached) {
    return std::unexpected(cached.error());
  }
  auto& entry = **cached;
  auto& planned = *entry.planned->plan;
  std::vector<std::pair<ggml_tensor*, const void*>> sources;
  Copies outputs;
  for (const auto& w : work) {
    const auto s = w.slot->index();
    auto& f = *frames[s];
    const auto& g = *planned.draft(s);
    const auto rows = f.request->pending_rows;
    const auto n = static_cast<std::uint32_t>(w.history.size() - 1);
    Qwen38MtpSources(g, f.ins, w.history.subspan(n - rows + 1, rows), f.host);
    sources.insert(sources.end(), f.host.sources.begin(), f.host.sources.end());
    if (g.drafts.size() != static_cast<std::size_t>(key.slots[s].passes) ||
        g.probabilities.size() != (w.probabilities != nullptr ? g.drafts.size() : 0)) {
      return Error("the Qwen3.8 draft wave omitted a requested pass output");
    }
    for (std::size_t j = 0; j < g.drafts.size(); ++j) {
      if (auto valid = CheckWaveOutput(g.drafts[j], GGML_TYPE_I32, sizeof(std::int32_t)); !valid) {
        return valid;
      }
      outputs.push_back({Address(wave_ids_ + (std::size_t{s} * 8) + j), Address(g.drafts[j]->data),
                         sizeof(std::int32_t)});
    }
    for (std::size_t j = 0; j < g.probabilities.size(); ++j) {
      if (auto valid = CheckWaveOutput(g.probabilities[j], GGML_TYPE_F32, sizeof(float)); !valid) {
        return valid;
      }
      outputs.push_back({Address(wave_probabilities_ + (std::size_t{s} * 8) + j),
                         Address(g.probabilities[j]->data), sizeof(float)});
    }
  }
  if (auto valid = CheckWaveSources(sources, planned); !valid) {
    return valid;
  }
  for (const auto& w : work) {
    auto& f = *frames[w.slot->index()];
    if (auto used = EnsureState(*f.request, f.through, WaveReadAlign(work.size())); !used) {
      return used;
    }
  }
  if (!entry.planned->covered) {
    if (auto covered = CheckWave(planned); !covered) {
      return covered;
    }
    entry.planned->covered = true;
  }
  auto copies = runs_.Stage(sources, 0);
  if (!copies) {
    return std::unexpected(copies.error());
  }
  auto& launch = resources_.launch();
  auto& run = entry.runs[0];
  bool capture = run.CaptureDue(runs_.graphs());
  if (capture && !draft_waves_.ChargeGraph(entry)) {
    capture = false;  // no room for its graph even after a reclaim: launch by launch
  }
  Status ran;
  RunPath path = RunPath::kEager;
  bool unknown = false;
  auto job = [&](providers::NativeStream native) -> sc::JobResult {
    for (const auto& w : work) {
      if (auto owed = w.slot->request_.live.QueueOwed(launch); !owed) {
        ran = Error(owed.error().detail);
        unknown = true;
        return sc::JobResult::kUnknown;
      }
    }
    const auto queued =
        runs_.Queue(run, *copies, {}, *planned.bound, outputs, capture, draft_stats_, native);
    path = queued.path;
    if (!queued.result) {
      ran = Error(queued.result.error().detail);
      unknown = queued.result.error().error == kg::KernelError::kUnknown;
      return unknown ? sc::JobResult::kUnknown : sc::JobResult::kFailed;
    }
    return sc::JobResult::kQueued;
  };
  const auto posted = node_.Job(execution_, std::move(job), "a Qwen3.8 draft wave", stream_);
  if (!posted || !ran || launch.faulted()) {
    if (!posted) {
      CheckFailedJob();
    }
    for (const auto& w : work) {
      // As in scalar Draft: a known fenced failure wrote only MTP cells
      // that the next catch-up rewrites; unknown effects stop all owners.
      Settle(w.slot->request_, false, false, unknown);
    }
    if (launch.faulted()) {
      return Error("the Qwen3.8 draft wave launch context faulted");
    }
    return !ran ? ran : Error(posted.error());
  }
  Count(draft_stats_, path);
  last_wave_ = planned.stats();
  for (const auto& w : work) {
    const auto s = w.slot->index();
    const auto count = static_cast<std::size_t>(key.slots[s].passes);
    w.drafts->assign(wave_ids_ + (std::size_t{s} * 8), wave_ids_ + (std::size_t{s} * 8) + count);
    if (w.probabilities != nullptr) {
      w.probabilities->assign(wave_probabilities_ + (std::size_t{s} * 8),
                              wave_probabilities_ + (std::size_t{s} * 8) + count);
    }
  }
  return {};
}

Status Qwen38Runner::Chunk(std::span<const std::int32_t> history, std::uint32_t n_past,
                           std::vector<float>& logits, bool inject) {
  return Chunk(default_request_, history, n_past, logits, inject);
}

Status Qwen38Runner::Chunk(RequestState& request, std::span<const std::int32_t> history,
                           std::uint32_t n_past, std::vector<float>& logits, bool inject) {
  const PlanStep step;  // the plans this step borrows stay until its job ends
  request.track.Wrote(n_past);
  if (auto usable = Usable(request); !usable) {
    return usable;
  }
  if (auto waiting = request.live.AwaitingAccept(); !waiting) {
    return waiting;
  }
  if (inject && !speculative()) {
    return Error("an injection needs the drafter");
  }
  if (history.size() <= n_past) {
    return Error("an empty chunk");
  }
  const auto rows = static_cast<std::uint32_t>(history.size() - n_past);
  // Without the selection's host masks, which the fast graph makes on the
  // device; built below if the planned graph reads them.
  auto in = md::Qwen38Chunk(profile_, layout_, hash_, history, n_past, rows, false);
  if (!in) {
    return std::unexpected(in.error());
  }
  if (auto used = EnsureState(request, n_past + rows); !used) {
    return used;
  }
  auto slots = ReadRows(*in);
  if (!slots) {
    return std::unexpected(slots.error());
  }
  const Qwen38ChunkKind kind{.verify = false, .export_streams = inject};
  auto planned = Planned(request, {.shape = kg::Qwen38ShapeOf(layout_, *in, 1), .kind = kind});
  if (!planned) {
    return std::unexpected(planned.error());
  }
  ChunkPlans::Entry& entry = **planned;
  PlanRuns& runs = entry.runs[kWithLogits];
  Qwen38Planned* p = entry.planned.get();
  const kg::Qwen38Graph& g = p->graph;
  if (in->qsa_select && (g.mask != nullptr || g.mask_f32 != nullptr)) {
    // This graph selects over the host's masks (GGML's top-k).
    in = md::Qwen38Chunk(profile_, layout_, hash_, history, n_past, rows, true);
    if (!in) {
      return std::unexpected(in.error());
    }
  }
  Qwen38HostInputs host;
  Qwen38Sources(g, *in, 1, *slots, host, 1);
  auto copies = runs_.Stage(host.sources, 0);
  if (!copies) {
    return std::unexpected(copies.error());
  }
  // The drafter's pass over the positions whose next token the chunk
  // holds: the pending row of the chunk before (its streams in row 0) and
  // the chunk's rows but its last (in rows 1 ..), or at the start of the
  // sequence the chunk's rows but its last; then the last row's streams
  // become the pending row (rows 0 and 1).
  MtpPlans::Entry* mentry = nullptr;
  Copies mcopies;
  Qwen38MtpHostInputs mhost;
  std::vector<md::Qwen38ChunkInputs> mins;
  std::uint32_t carries = 0;
  if (inject) {
    const std::uint32_t mrows = n_past > 0 ? rows : rows - 1;
    const std::uint32_t first = n_past > 0 ? n_past - 1 : 0;
    if (mrows > 0) {
      auto shaped = MtpInputs(first, mrows, 1, false, n_past > 0 ? 0 : 1);
      if (!shaped) {
        return std::unexpected(shaped.error());
      }
      mins = std::move(shaped->second);
      auto mplanned = PlannedMtp(request, shaped->first);
      if (!mplanned) {
        return std::unexpected(mplanned.error());
      }
      mentry = *mplanned;
      Qwen38MtpSources(mentry->planned->graph, mins, history.subspan(first + 1, mrows), mhost);
      auto staged = runs_.Stage(mhost.sources, mtp_base_);
      if (!staged) {
        return std::unexpected(staged.error());
      }
      mcopies = std::move(*staged);
    }
    const std::uint64_t row = std::uint64_t{profile_.hc_width()} * sizeof(float);
    const std::uint64_t streams = request.live.base(kDrafter) + mtp_layout_.hidden;
    request.carry[carries++] = {.from = streams + (rows * row), .to = streams, .bytes = row};
    if (rows != 1) {
      request.carry[carries++] = {
          .from = streams + (rows * row), .to = streams + row, .bytes = row};
    }
  }
  const std::uint64_t row_bytes = std::uint64_t{profile_.vocab} * sizeof(float);
  const std::function<bool(void*)> gather = Gather(rows);
  // Decode graphs (D-090): replay a shape's graph; capture a one-row shape
  // that has run once launch by launch; otherwise launch by launch.
  bool capture = runs.CaptureDue(runs_.graphs()) && rows == 1 && !inject;
  if (capture && !request.plans.ChargeGraph(entry, kWithLogits)) {
    capture = false;  // no room for its graph even after a reclaim: launch by launch
  }
  const Copies outputs = {{Address(logits_), Address(g.logits->data), row_bytes}};
  kg::LaunchContext& launch = resources_.launch();
  RunPath path = RunPath::kEager;
  Status ran;
  bool wrote = false;
  bool unknown = false;
  auto job = [&](providers::NativeStream native) -> sc::JobResult {
    // A pending commit or restore queued here is work this job must fence,
    // even if its own run is then refused before anything else.
    const bool committing = request.live.owed();
    if (auto r = request.live.QueueOwed(launch); !r) {
      ran = Error(std::format("chunk at {}: {}", n_past, r.error().detail));
      unknown = true;
      return sc::JobResult::kUnknown;
    }
    const Queued queued =
        runs_.Queue(runs, *copies, gather, *p->bound, outputs, capture, graph_stats_, native);
    path = queued.path;
    wrote = queued.result.has_value() || queued.before;
    if (!queued.result) {
      ran = Error(std::format("chunk at {}: {}", n_past, queued.result.error().detail));
      if (queued.result.error().error == kg::KernelError::kUnknown) {
        unknown = true;
        return sc::JobResult::kUnknown;
      }
      return committing || queued.before ? sc::JobResult::kFailed : sc::JobResult::kNotStarted;
    }
    if (mentry != nullptr) {
      const Queued m = runs_.Queue(mentry->runs[0], mcopies, {}, *mentry->planned->bound, {}, false,
                                   draft_stats_, native);
      if (!m.result) {
        ran = Error(std::format("the drafter's pass at {}: {}", n_past, m.result.error().detail));
        unknown = m.result.error().error == kg::KernelError::kUnknown;
        return unknown ? sc::JobResult::kUnknown : sc::JobResult::kFailed;
      }
      Count(draft_stats_, m.path);
    }
    if (carries != 0) {
      if (auto r = kg::CopyRanges(launch, request.carry, carries); !r) {
        ran = Error(std::format("the drafter's pending row at {}: {}", n_past, r.error().detail));
        unknown = true;
        return sc::JobResult::kUnknown;
      }
    }
    return sc::JobResult::kQueued;
  };
  const Status posted = node_.Job(execution_, std::move(job), "a Qwen3.8 chunk", stream_);
  if (!posted || !ran || launch.faulted()) {
    if (!posted) {
      CheckFailedJob();
    }
    Settle(request, false, wrote, unknown);
    if (launch.faulted()) {
      return Error(std::format("chunk at {}: the launch context faulted", n_past));
    }
    return !ran ? ran : Error(std::format("chunk at {}: {}", n_past, posted.error()));
  }
  Count(graph_stats_, path);
  // With the injection the chunk's last row is the pending one; without,
  // the streams rows no longer precede the anchor, so no draft may read
  // them until a chunk with the injection or an accepted verify.
  request.pending_rows = inject ? 1 : 0;
  const auto* values = static_cast<const float*>(logits_);
  logits.assign(values, values + profile_.vocab);
  return {};
}

Status Qwen38Runner::Draft(std::span<const std::int32_t> history, std::vector<std::int32_t>& drafts,
                           std::vector<float>* probabilities, std::uint32_t passes,
                           Qwen38DraftHeadCapture* head_capture) {
  return Draft(default_request_, history, drafts, probabilities, passes, head_capture);
}

Status Qwen38Runner::Draft(RequestState& request, std::span<const std::int32_t> history,
                           std::vector<std::int32_t>& drafts, std::vector<float>* probabilities,
                           std::uint32_t passes, Qwen38DraftHeadCapture* head_capture) {
  const PlanStep step;  // the plan this step borrows stays until its job ends
  // The drafter catches up the previous target row before advancing.
  request.track.Wrote(history.size() > 1 ? static_cast<std::uint32_t>(history.size() - 2) : 0);
  if (head_capture != nullptr) {
    *head_capture = {};
    if (!o_.draft_head_capture || draft_head_capture_ == nullptr || capture_head_rows_ == 0) {
      return Error("draft-head capture was not provisioned at setup");
    }
  }
  if (!speculative()) {
    return Error("drafting needs the drafter");
  }
  if (auto usable = Usable(request); !usable) {
    return usable;
  }
  if (auto waiting = request.live.AwaitingAccept(); !waiting) {
    return waiting;
  }
  const std::uint32_t rows = request.pending_rows;
  if (rows == 0 || history.size() < std::size_t{rows} + 1) {
    return Error("no streams to draft from (a prefill with the injection, or a verify, first)");
  }
  // The catch-up: the rows the last verify kept (or the prefill left), at
  // the positions before the anchor, each with the token after it. Those
  // tokens index the target's table on the device (the anchor no chunk has
  // checked yet among them).
  const auto n = static_cast<std::uint32_t>(history.size() - 1);
  for (const std::int32_t t : history.subspan(n - rows + 1, rows)) {
    if (t < 0 || std::cmp_greater_equal(t, profile_.vocab)) {
      return Error(std::format("token {} is outside the vocabulary", t));
    }
  }
  passes = passes == 0 ? o_.draft_rows : passes;
  if (passes > o_.draft_rows) {
    return Error("a draft past the configured maximum depth");
  }
  auto shaped =
      MtpInputs(n - rows, rows, passes, true, 1, probabilities != nullptr, head_capture != nullptr);
  if (!shaped) {
    return std::unexpected(shaped.error());
  }
  if (auto used = EnsureState(request, n + passes - 1); !used) {
    return used;
  }
  auto planned = PlannedMtp(request, shaped->first);
  if (!planned) {
    return std::unexpected(planned.error());
  }
  MtpPlans::Entry& entry = **planned;
  PlanRuns& runs = entry.runs[0];
  const kg::Qwen38MtpGraph& g = entry.planned->graph;
  if (head_capture != nullptr) {
    if (g.head_inputs.size() != passes || g.head_logits.size() != passes) {
      return Error("draft-head capture is missing a pass's input or logits");
    }
    const auto held_row = [&](const ggml_tensor* t, std::uint32_t width) {
      if (t == nullptr || t->data == nullptr || t->type != GGML_TYPE_F32 ||
          std::cmp_not_equal(t->ne[0], width) || t->ne[1] != 1 || t->ne[2] != 1 || t->ne[3] != 1 ||
          !ggml_is_contiguous(t)) {
        return false;
      }
      const auto& region = node_.activations();
      const std::uint64_t address = Address(t->data);
      const std::uint64_t bytes = std::uint64_t{width} * sizeof(float);
      return address >= region.base && address - region.base <= region.bytes &&
             bytes <= region.bytes - (address - region.base);
    };
    for (std::size_t j = 0; j < passes; ++j) {
      if (!held_row(g.head_inputs[j], profile_.width) ||
          !held_row(g.head_logits[j], capture_head_rows_)) {
        return Error("draft-head capture source is outside its retained packed activation row");
      }
    }
  }
  Qwen38MtpHostInputs host;
  Qwen38MtpSources(g, shaped->second, history.subspan(n - rows + 1, rows), host);
  auto copies = runs_.Stage(host.sources, 0);
  if (!copies) {
    return std::unexpected(copies.error());
  }
  Copies outputs;
  for (std::size_t j = 0; j < g.drafts.size(); ++j) {
    outputs.push_back({Address(static_cast<std::int32_t*>(drafts_) + j), Address(g.drafts[j]->data),
                       sizeof(std::int32_t)});
  }
  for (std::size_t j = 0; j < g.probabilities.size(); ++j) {
    outputs.push_back({Address(static_cast<std::int32_t*>(drafts_) + kProbabilityAt + j),
                       Address(g.probabilities[j]->data), sizeof(std::int32_t)});
  }
  if (head_capture != nullptr) {
    const std::uint64_t stride = std::uint64_t{profile_.width} + capture_head_rows_;
    for (std::size_t j = 0; j < passes; ++j) {
      float* into = draft_head_capture_ + (j * stride);
      outputs.push_back({Address(into), Address(g.head_inputs[j]->data),
                         std::uint64_t{profile_.width} * sizeof(float)});
      outputs.push_back({Address(into + profile_.width), Address(g.head_logits[j]->data),
                         std::uint64_t{capture_head_rows_} * sizeof(float)});
    }
  }
  bool capture = runs.CaptureDue(runs_.graphs());
  if (capture && !request.mplans.ChargeGraph(entry)) {
    capture = false;  // no room for its graph even after a reclaim: launch by launch
  }
  kg::LaunchContext& launch = resources_.launch();
  RunPath path = RunPath::kEager;
  Status ran;
  bool unknown = false;
  auto job = [&](providers::NativeStream native) -> sc::JobResult {
    if (auto r = request.live.QueueOwed(launch); !r) {
      ran = Error(std::format("draft at {}: {}", n, r.error().detail));
      unknown = true;
      return sc::JobResult::kUnknown;
    }
    const Queued queued = runs_.Queue(runs, *copies, {}, *entry.planned->bound, outputs, capture,
                                      draft_stats_, native);
    path = queued.path;
    if (!queued.result) {
      ran = Error(std::format("draft at {}: {}", n, queued.result.error().detail));
      unknown = queued.result.error().error == kg::KernelError::kUnknown;
      return unknown ? sc::JobResult::kUnknown : sc::JobResult::kFailed;
    }
    return sc::JobResult::kQueued;
  };
  const Status posted = node_.Job(execution_, std::move(job), "a Qwen3.8 MTP draft", stream_);
  if (!posted || !ran || launch.faulted()) {
    if (!posted) {
      CheckFailedJob();
    }
    // A draft writes the drafter's cells alone (the committed ones as the
    // next draft rewrites them); a launch of unknown effect quarantines.
    Settle(request, false, false, unknown);
    if (launch.faulted()) {
      return Error(std::format("draft at {}: the launch context faulted", n));
    }
    return !ran ? ran : Error(std::format("draft at {}: {}", n, posted.error()));
  }
  Count(draft_stats_, path);
  const auto* values = static_cast<const std::int32_t*>(drafts_);
  drafts.assign(values, values + g.drafts.size());
  if (probabilities != nullptr) {
    probabilities->resize(g.probabilities.size());
    std::memcpy(probabilities->data(), values + kProbabilityAt,
                g.probabilities.size() * sizeof(float));
  }
  if (head_capture != nullptr) {
    // The node job has retired successfully. DMA named only node-owned
    // pinned staging; ordinary vectors begin owning copies after completion.
    head_capture->catch_up_rows = rows;
    head_capture->head_rows = capture_head_rows_;
    head_capture->token_ids.resize(capture_head_rows_);
    if (dbinding_.selected_head()) {
      std::memcpy(head_capture->token_ids.data(), draft_ids_host_,
                  std::uint64_t{capture_head_rows_} * sizeof(std::int32_t));
    } else {
      for (std::uint32_t i = 0; i < capture_head_rows_; ++i) {
        head_capture->token_ids[i] = static_cast<std::int32_t>(i);
      }
    }
    const std::uint64_t stride = std::uint64_t{profile_.width} + capture_head_rows_;
    for (std::size_t j = 0; j < passes; ++j) {
      const float* from = draft_head_capture_ + (j * stride);
      head_capture->inputs.insert(head_capture->inputs.end(), from, from + profile_.width);
      head_capture->logits.insert(head_capture->logits.end(), from + profile_.width, from + stride);
    }
  }
  return {};
}

Status Qwen38Runner::Verify(std::span<const std::int32_t> history, std::uint32_t n_past,
                            std::vector<std::int32_t>& argmax, std::vector<float>* logits,
                            Qwen38RoutedCapture* routed_capture) {
  return Verify(default_request_, history, n_past, argmax, logits, routed_capture);
}

Status Qwen38Runner::Verify(RequestState& request, std::span<const std::int32_t> history,
                            std::uint32_t n_past, std::vector<std::int32_t>& argmax,
                            std::vector<float>* logits, Qwen38RoutedCapture* routed_capture) {
  const PlanStep step;  // the plan this step borrows stays until its job ends
  request.track.Wrote(n_past > 0 ? n_past - 1 : 0);
  if (routed_capture != nullptr) {
    *routed_capture = {};
    if (o_.routed_capture == 0 || routed_capture_ == nullptr) {
      return Error("routed capture was not provisioned at setup");
    }
  }
  if (!speculative()) {
    return Error("a verify needs the drafter");
  }
  if (auto usable = Usable(request); !usable) {
    return usable;
  }
  if (auto waiting = request.live.AwaitingAccept(); !waiting) {
    return waiting;
  }
  if (history.size() <= n_past || history.size() - n_past > std::size_t{o_.draft_rows} + 1) {
    return Error(std::format("a verify of 1 to {} rows", o_.draft_rows + 1));
  }
  const auto rows = static_cast<std::uint32_t>(history.size() - n_past);
  auto in = md::Qwen38Chunk(profile_, layout_, hash_, history, n_past, rows, false);
  if (!in) {
    return std::unexpected(in.error());
  }
  if (auto used = EnsureState(request, n_past + rows); !used) {
    return used;
  }
  auto slots = ReadRows(*in);
  if (!slots) {
    return std::unexpected(slots.error());
  }
  const Qwen38ChunkKind kind{.verify = true,
                             .export_streams = true,
                             .capture_routed = routed_capture != nullptr ? o_.routed_capture : 0};
  auto planned = Planned(request, {.shape = kg::Qwen38ShapeOf(layout_, *in, rows), .kind = kind});
  if (!planned) {
    return std::unexpected(planned.error());
  }
  ChunkPlans::Entry& entry = **planned;
  Qwen38Planned* p = entry.planned.get();
  const kg::Qwen38Graph& g = p->graph;
  // The graph keeps these computed operands through the final node. Only
  // runner-owned pinned memory is a DMA destination; caller vectors are
  // populated after node_.Job has proven completion and retirement.
  struct Retained {
    ggml_tensor* tensor;
    std::uint64_t offset;
    std::uint64_t bytes;
  };
  std::vector<Retained> retained;
  std::uint64_t retained_bytes = 0;
  if (routed_capture != nullptr) {
    if (g.routed.size() != static_cast<std::size_t>(std::popcount(o_.routed_capture))) {
      return Error("routed capture is missing a requested layer");
    }
    const auto& region = node_.activations();
    for (const auto& layer : g.routed) {
      if (layer.layer >= profile_.layers ||
          (o_.routed_capture & (std::uint64_t{1} << layer.layer)) == 0) {
        return Error("routed capture selected an unrequested layer");
      }
      const std::int64_t width = profile_.width;
      const std::int64_t ffn = profile_.expert_ffn;
      const std::int64_t used = profile_.experts_used;
      struct Part {
        ggml_tensor* tensor;
        ggml_type type;
        std::int64_t n0, n1, n2;
      };
      std::vector<Part> parts = {Part{layer.input, GGML_TYPE_F32, width, 1, rows},
                                 Part{layer.activation, GGML_TYPE_F32, ffn, used, rows},
                                 Part{layer.down, GGML_TYPE_F32, width, used, rows},
                                 Part{layer.shared, GGML_TYPE_F32, width, rows, 1},
                                 Part{layer.gate, GGML_TYPE_F32, 1, rows, 1},
                                 Part{layer.weights, GGML_TYPE_F32, 1, used, rows},
                                 Part{layer.ids, GGML_TYPE_I32, used, rows, 1},
                                 Part{layer.combined, GGML_TYPE_F32, width, rows, 1}};
      if (!binding_.layers[layer.layer].linear) {
        const std::int64_t projection = 2 * std::int64_t{profile_.heads} * profile_.head_dim;
        parts.push_back({layer.attention_input, GGML_TYPE_F32, width, rows, 1});
        parts.push_back({layer.attention_projection, GGML_TYPE_F32, projection, rows, 1});
      } else if (layer.attention_input != nullptr || layer.attention_projection != nullptr) {
        return Error("a GDN layer cannot supply the QSA projection capture");
      }
      for (const auto part : parts) {
        const ggml_tensor* t = part.tensor;
        const std::uint64_t bytes = static_cast<std::uint64_t>(part.n0 * part.n1 * part.n2) * 4;
        if (t == nullptr || t->data == nullptr || t->type != part.type || t->ne[0] != part.n0 ||
            t->ne[1] != part.n1 || t->ne[2] != part.n2 || t->ne[3] != 1 || !ggml_is_contiguous(t)) {
          return Error("routed capture has an invalid packed operand");
        }
        const std::uint64_t address = Address(t->data);
        if (address < region.base || address - region.base > region.bytes ||
            bytes > region.bytes - (address - region.base) ||
            retained_bytes > routed_capture_bytes_ ||
            bytes > routed_capture_bytes_ - retained_bytes) {
          return Error("routed capture source or destination exceeds its retained allocation");
        }
        retained.push_back({part.tensor, retained_bytes, bytes});
        retained_bytes += bytes;
      }
    }
  }
  if (in->qsa_select && (g.mask != nullptr || g.mask_f32 != nullptr)) {
    in = md::Qwen38Chunk(profile_, layout_, hash_, history, n_past, rows, true);
    if (!in) {
      return std::unexpected(in.error());
    }
  }
  Qwen38HostInputs host;
  Qwen38Sources(g, *in, rows, *slots, host, 1);
  auto copies = runs_.Stage(host.sources, 0);
  if (!copies) {
    return std::unexpected(copies.error());
  }
  // The cells it writes (each QSA layer's K, V and indexer rows), saved.
  using K = md::Qwen38StateTensor::Kind;
  const std::uint64_t state = request.live.base(kTarget);
  request.live.BeginSaves();
  request.verify_restores_streams = false;
  for (std::uint32_t i = 0; i < rows; ++i) {
    const std::uint64_t cell = std::uint64_t{n_past} + i;
    for (const md::Qwen38StateTensor& t : layout_.tensors) {
      if (t.kind != K::kK && t.kind != K::kV && t.kind != K::kIndexerK) {
        continue;
      }
      const std::uint64_t bytes = t.ne0 * (t.f16 ? 2 : 4);
      if (auto added = request.live.Save(state + t.offset + (cell * bytes), bytes, i); !added) {
        return added;
      }
    }
  }
  // And each block key it writes: the row that completes the block writes
  // it, so a rejected row's block is restored with its cells.
  const std::uint32_t ratio = profile_.indexer_ratio;
  for (std::uint64_t b = n_past / ratio; b < (std::uint64_t{n_past} + rows) / ratio; ++b) {
    const auto row = static_cast<std::uint32_t>(((b + 1) * ratio) - 1 - n_past);
    for (const md::Qwen38StateTensor& t : layout_.tensors) {
      if (t.kind != K::kIndexerBlocks) {
        continue;
      }
      const std::uint64_t bytes = t.ne0 * 2;
      if (auto added = request.live.Save(state + t.offset + (b * bytes), bytes, row); !added) {
        return added;
      }
    }
  }
  const std::function<bool(void*)> gather = Gather(rows);
  // The argmaxes always; the logits (their own runs) when asked.
  PlanRuns& runs = entry.runs[logits != nullptr ? kWithLogits : kLean];
  bool capture = runs.CaptureDue(runs_.graphs());
  if (capture && !request.plans.ChargeGraph(entry, logits != nullptr ? kWithLogits : kLean)) {
    capture = false;  // no room for its graph even after a reclaim: launch by launch
  }
  const std::uint64_t row_bytes = std::uint64_t{profile_.vocab} * sizeof(float);
  auto* const argmax_host = static_cast<std::int32_t*>(drafts_) + kArgmaxAt;
  Copies outputs = {{Address(argmax_host), Address(g.argmax->data), rows * sizeof(std::int32_t)}};
  if (logits != nullptr) {
    outputs.push_back({Address(logits_), Address(g.logits->data), rows * row_bytes});
  }
  for (const auto& part : retained) {
    outputs.push_back(
        {Address(routed_capture_ + part.offset), Address(part.tensor->data), part.bytes});
  }
  kg::LaunchContext& launch = resources_.launch();
  RunPath path = RunPath::kEager;
  Status ran;
  bool saved = false;
  bool unknown = false;
  auto job = [&](providers::NativeStream native) -> sc::JobResult {
    if (auto r = request.live.QueueOwed(launch); !r) {
      ran = Error(std::format("verify at {}: {}", n_past, r.error().detail));
      unknown = true;
      return sc::JobResult::kUnknown;
    }
    if (auto r = request.live.QueueSaves(launch); !r) {
      ran = Error(std::format("verify at {}: {}", n_past, r.error().detail));
      unknown = true;
      return sc::JobResult::kUnknown;
    }
    saved = true;
    const Queued queued =
        runs_.Queue(runs, *copies, gather, *p->bound, outputs, capture, graph_stats_, native);
    path = queued.path;
    if (!queued.result) {
      ran = Error(std::format("verify at {}: {}", n_past, queued.result.error().detail));
      unknown = queued.result.error().error == kg::KernelError::kUnknown;
      return unknown ? sc::JobResult::kUnknown : sc::JobResult::kFailed;
    }
    return sc::JobResult::kQueued;
  };
  const Status posted = node_.Job(execution_, std::move(job), "a Qwen3.8 verify", stream_);
  if (!posted || !ran || launch.faulted()) {
    if (!posted) {
      CheckFailedJob();
    }
    // Never left half-written: the verify is undone before the next job's
    // work, or the state quarantined.
    Settle(request, saved, false, unknown);
    if (launch.faulted()) {
      return Error(std::format("verify at {}: the launch context faulted", n_past));
    }
    return !ran ? ran : Error(std::format("verify at {}: {}", n_past, posted.error()));
  }
  Count(graph_stats_, path);
  request.live.Verified(rows);
  argmax.assign(argmax_host, argmax_host + rows);
  if (logits != nullptr) {
    const auto* values = static_cast<const float*>(logits_);
    logits->assign(values, values + (std::size_t{rows} * profile_.vocab));
  }
  if (routed_capture != nullptr) {
    routed_capture->rows = rows;
    std::size_t part = 0;
    for (const auto& layer : g.routed) {
      Qwen38RoutedCapture::Layer values;
      values.layer = layer.layer;
      const auto take = [&](auto& into) {
        const Retained& source = retained[part++];
        using Value = std::remove_reference_t<decltype(into)>::value_type;
        into.resize(source.bytes / sizeof(Value));
        std::memcpy(into.data(), routed_capture_ + source.offset, source.bytes);
      };
      take(values.input);
      take(values.activation);
      take(values.down);
      take(values.shared);
      take(values.gate);
      take(values.weights);
      take(values.ids);
      take(values.combined);
      if (!binding_.layers[layer.layer].linear) {
        take(values.attention_input);
        take(values.attention_projection);
      }
      routed_capture->layers.push_back(std::move(values));
    }
  }
  return {};
}

Status Qwen38Runner::Accept(std::uint32_t keep) { return Accept(default_request_, keep); }

Status Qwen38Runner::Accept(RequestState& request, std::uint32_t keep) {
  if (auto usable = Usable(request); !usable) {
    return usable;
  }
  if (auto accepted = request.live.Accept(keep); !accepted) {
    return accepted;
  }
  request.pending_rows = keep;
  request.verify_restores_streams = false;
  return {};
}

Status Qwen38Runner::Rollback() { return Rollback(default_request_); }

Status Qwen38Runner::DiscardVerify(RequestState& request) {
  if (auto usable = Usable(request); !usable) {
    return usable;
  }
  if (request.live.verify_rows() == 0 || !request.verify_restores_streams) {
    return Error("discarding a Qwen verify that does not await Accept");
  }
  // A successful synchronous Verify completed its saves and writes. Accept
  // has not changed pending_rows. Restore every saved range, with no kept
  // recurrent streams, then drain through the existing failure handling.
  (void)request.live.Settle(true, true, false);
  request.verify_restores_streams = false;
  return Rollback(request);
}

Status Qwen38Runner::Rollback(RequestState& request) {
  if (auto usable = Usable(request); !usable) {
    return usable;
  }
  auto rolled = request.live.Rollback(node_, execution_, stream_, resources_.launch(),
                                      "committing a verify's kept rows");
  if (!rolled) {
    CheckFailedJob();
  }
  if (resources_.launch().faulted() || (!rolled && request.live.quarantined())) {
    FaultCohort();
  }
  return rolled;
}

Status Qwen38Runner::ReadState(std::vector<std::byte>& target, std::vector<std::byte>& drafter) {
  return ReadState(default_request_, target, drafter);
}

Status Qwen38Runner::ReadState(RequestState& request, std::vector<std::byte>& target,
                               std::vector<std::byte>& drafter) {
  if (auto r = Rollback(request); !r) {
    return r;
  }
  if (request.live.verify_rows() != 0) {
    return Error("reading the state with a verify awaiting its Accept");
  }
  const std::array<std::vector<std::byte>*, 2> out = {&target, &drafter};
  LiveState::CopyRetirement retirement = LiveState::CopyRetirement::kProven;
  auto read = request.live.Read(node_, request.fence, stream_, "reading the Qwen3.8 state", out,
                                &retirement);
  if (retirement == LiveState::CopyRetirement::kUnproven) {
    FaultCohort();
  }
  return read;
}

Status Qwen38Runner::Release() {
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
    request->live.Release(memory, problems);
  }
  for (PagedWeights* part : {&weights_, &dweights_}) {
    if (auto r = part->Release(memory); !r) {
      problems.push_back(std::format("Qwen3.8: {}", r.error()));
    }
  }
  // Reads that stalled may still land: the ring and their landing are left
  // to the process's end, the landing kept from the node's frees at Close.
  // Otherwise every read was harvested (ReadPleRows drains) and the ring
  // goes.
  const std::array<void*, 1> landing = {landing_};
  if (node_.RetireRing(std::move(ring_), landing)) {
    problems.emplace_back(
        "n-gram row reads were still in flight; their ring and landing are kept to the "
        "process's end");
  }
  return support::Joined(problems);
}

Status Qwen38Runner::SaveUsedState(void* host, std::span<const LiveState::Range> ranges) {
  return SaveUsedState(default_request_, host, ranges);
}

Status Qwen38Runner::SaveUsedState(RequestState& request, void* host,
                                   std::span<const LiveState::Range> ranges) {
  if (auto active = CheckResident(request); !active) {
    return active;
  }
  LiveState::CopyRetirement retirement = LiveState::CopyRetirement::kProven;
  auto copied = request.live.Copy(node_, request.fence, stream_, host, ranges, true, &retirement);
  if (retirement == LiveState::CopyRetirement::kUnproven) {
    FaultCohort();
  }
  return copied;
}

Status Qwen38Runner::RestoreUsedState(void* host, std::span<const LiveState::Range> ranges) {
  return RestoreUsedState(default_request_, host, ranges);
}

Status Qwen38Runner::RestoreUsedState(RequestState& request, void* host,
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

Status Qwen38Runner::PrepareRestoreState(std::span<const LiveState::Range> ranges) {
  return PrepareRestoreState(default_request_, ranges);
}

Status Qwen38Runner::PrepareRestoreState(RequestState& request,
                                         std::span<const LiveState::Range> ranges) {
  if (auto active = CheckResident(request); !active) {
    return active;
  }
  const SlotMask others = active_mask_ & ~(SlotMask{1} << request.slot);
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
  // Even refused/partial growth belongs to this stable owner and appears
  // in both the retained-state union and the active lease before returning.
  if (auto refreshed = RefreshClosures(); !refreshed) {
    return refreshed;
  }
  if (!prepared) {
    return prepared;
  }
  return {};
}

Status Qwen38Runner::CopyCheckpointState(void* host, std::span<const LiveState::Range> ranges,
                                         bool to_host) {
  return CopyCheckpointState(default_request_, host, ranges, to_host);
}

Status Qwen38Runner::CopyCheckpointState(RequestState& request, void* host,
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
  if (retirement == LiveState::CopyRetirement::kUnproven) {
    FaultCohort();
  }
  return copied;
}

std::expected<std::vector<LiveState::Range>, std::string> Qwen38Runner::CheckpointRanges(
    std::uint32_t positions) const {
  return CheckpointRanges(default_request_, positions);
}

std::expected<std::vector<LiveState::Range>, std::string> Qwen38Runner::CheckpointRanges(
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
    return Error("checkpoint has an unsettled Qwen3.8 verify");
  }
  auto writes = StateWrites(positions);
  if (!writes) {
    return std::unexpected(writes.error());
  }
  return CheckpointPages(request.live.used_ranges(), *writes);
}

std::expected<std::vector<LiveState::Range>, std::string> Qwen38Runner::StateWrites(
    std::uint32_t positions) const {
  auto mutable_bytes = md::Qwen38CheckpointWrites(profile_, layout_, positions);
  if (!mutable_bytes) {
    return std::unexpected(mutable_bytes.error());
  }
  std::vector<LiveState::Range> writes;
  for (const auto& range : *mutable_bytes) {
    writes.push_back({.region = kTarget, .offset = range.offset, .bytes = range.bytes});
  }
  if (speculative()) {
    // The drafter catches up the previous target row before advancing.
    const std::uint64_t first = positions == 0 ? 0 : positions - 1;
    const std::uint64_t kv_row = std::uint64_t{profile_.head_dim} * profile_.kv_heads * 2;
    const std::uint64_t indexer_row = std::uint64_t{profile_.indexer_head_dim} * 4;
    const auto tail = [&](std::uint64_t offset, std::uint64_t row, std::uint64_t begin,
                          std::uint64_t cells) {
      if (begin < cells) {
        writes.push_back(
            {.region = kDrafter, .offset = offset + (begin * row), .bytes = (cells - begin) * row});
      }
    };
    tail(mtp_layout_.k, kv_row, first, mtp_layout_.cells);
    tail(mtp_layout_.v, kv_row, first, mtp_layout_.cells);
    tail(mtp_layout_.indexer, indexer_row, first, mtp_layout_.cells);
    tail(mtp_layout_.blocks, std::uint64_t{profile_.indexer_head_dim} * 2,
         first / profile_.indexer_ratio,
         (std::uint64_t{mtp_layout_.cells} + profile_.indexer_ratio - 1) / profile_.indexer_ratio);
    writes.push_back({.region = kDrafter,
                      .offset = mtp_layout_.hidden,
                      .bytes = mtp_layout_.bytes - mtp_layout_.hidden});
  }
  return writes;
}

}  // namespace jitllm::engine
