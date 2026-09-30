// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "engine/dsv4_runner.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <expected>
#include <format>
#include <initializer_list>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

#include "engine/checkpoint_file.h"
#include "engine/support.h"
#include "ggml.h"
#include "kernels/ggml/executor.h"
#include "kernels/ggml/graph_plan.h"
#include "kernels/ggml/jitllm_ops.h"
#include "kernels/ggml/ops_ext.h"
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
// The most snapshot ranges a verify saves: at most 8 rows of about 230
// ranges each (a cell a layer, a CSA layer's six ring and compressed rows,
// an HCA layer's three, the drafter's three cells), and the scratch rows.
constexpr std::uint32_t kRangeCapacity = 4096;

}  // namespace

// ------------------------------------------------------------------ setup

std::vector<ExtentId> Dsv4Runner::weights() const {
  std::vector<ExtentId> all = weights_.extents();
  all.insert(all.end(), dweights_.extents().begin(), dweights_.extents().end());
  return all;
}

std::vector<ExtentId> Dsv4Runner::managed_extents() const {
  std::vector<ExtentId> all = weights();
  const std::vector<ExtentId> live = state();
  all.insert(all.end(), live.begin(), live.end());
  return all;
}

std::span<const std::byte> Dsv4Runner::table() const {
  const std::uint64_t base = weights_.group_address(table_group_);
  return {static_cast<const std::byte*>(Pointer(base)), weights_.host_bytes()};
}

Status Dsv4Runner::Setup() {
  if (o_.context > md::kDsv4FlashContext) {
    return Error(std::format("context {} exceeds DeepSeek V4 Flash's trained ceiling {}",
                             o_.context, md::kDsv4FlashContext));
  }
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
  // The state first: its extents come before the weights' in a closure, so
  // a swap back restores it before paging the weights in.
  if (auto r = live_.AddGrowing(node_, "the DeepSeek state", layout_.bytes, owner_); !r) {
    return r;
  }
  if (speculative()) {
    if (auto r = live_.AddGrowing(node_, "the DSpark ring", dlayout_.bytes, owner_); !r) {
      return r;
    }
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
  // chunks with its injection, verifies and its draft block too.
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
                     .exact = o_.exact};
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
                          .exact = o_.exact};
  }
  std::uint64_t most_activations = 0;
  std::uint64_t most_scratch = 0;
  std::uint64_t most_inputs = 0;
  {
    auto measure = resources_.MeasuringContext();
    if (!measure) {
      return std::unexpected(measure.error());
    }
    const kg::DeviceChoices choices = kg::DeviceChoicesOf(**measure);
    const auto account = [&](const PlannedBase& planned) -> Status {
      most_activations = std::max(most_activations, planned.placement.extent);
      auto scratch = kg::PlanScratch(**measure, planned.plan);
      if (!scratch) {
        return Error(scratch.error().detail);
      }
      most_scratch = std::max(most_scratch, *scratch);
      most_inputs = std::max(most_inputs, planned.inputs_bytes);
      return {};
    };
    struct Probe {
      std::uint32_t n_past;
      std::uint32_t rows;
      Dsv4ChunkKind kind;
    };
    std::vector<Probe> probes = {{0, o_.max_rows, Dsv4ChunkKind::kPlain},
                                 {o_.context - o_.max_rows, o_.max_rows, Dsv4ChunkKind::kPlain},
                                 {o_.context - 1, 1, Dsv4ChunkKind::kPlain},
                                 {o_.max_rows, 1, Dsv4ChunkKind::kPlain}};
    if (speculative()) {
      probes.push_back({0, o_.max_rows, Dsv4ChunkKind::kInject});
      probes.push_back({o_.context - o_.max_rows, o_.max_rows, Dsv4ChunkKind::kInject});
      probes.push_back({0, o_.max_verify, Dsv4ChunkKind::kVerify});
      probes.push_back({o_.context - o_.max_verify, o_.max_verify, Dsv4ChunkKind::kVerify});
    }
    for (const Probe& probe : probes) {
      auto in = md::Dsv4Chunk(profile_, layout_, probe.n_past, probe.rows, o_.exact);
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
      auto planned =
          PlanDsv4Chunk(model_, kg::Dsv4ShapeOf(layout_, *in), choices, {}, 0, 0, speculation);
      if (!planned) {
        return Error(std::format("measuring a chunk of {} at {}: {}", probe.rows, probe.n_past,
                                 planned.error()));
      }
      if (auto r = account(**planned); !r) {
        return r;
      }
    }
    if (speculative()) {
      auto planned = PlanDsparkDraft(dmodel_, o_.draft_rows, choices, 0, 0);
      if (!planned) {
        return Error(std::format("measuring the draft block: {}", planned.error()));
      }
      if (auto r = account(**planned); !r) {
        return r;
      }
    }
  }
  // Margins: other shapes of these widths place a little differently.
  activation_bytes_ = Round(most_activations + (most_activations / 4), kExtent);
  scratch_bytes_ = Round(most_scratch + (most_scratch / 4) + (1U << 20U), kExtent);
  const std::uint64_t input_bytes = Round((most_inputs * 2) + (1U << 20U), kExtent);
  // A chunk's host-built inputs (Dsv4ChunkInputs and the embedding rows)
  // are the staged bytes again, on the host.
  host_input_bytes_ = Round(most_inputs + (1U << 20U), kExtent);
  // A verify's inputs from the staging's second half, which the largest
  // inputs fit.
  verify_base_ = Round(input_bytes / 2, 256);

  const std::uint64_t logit_rows = speculative() ? o_.max_verify : 1;
  auto inputs = resources_.Pinned(input_bytes);
  auto logits = resources_.Pinned(logit_rows * std::uint64_t{profile_.vocab} * sizeof(float));
  std::uint64_t table_bytes = 0;
  for (const md::Dsv4Layer& l : binding_.layers) {
    if (l.hash) {
      table_bytes += weights_.artifact().resources()[l.tid2eid.index].bytes.value();
    }
  }
  auto tables = resources_.Pinned(std::max<std::uint64_t>(table_bytes, 256));
  if (!inputs || !logits || !tables) {
    return Error("pinned staging for DeepSeek");
  }
  runs_.SetStaging(*inputs, input_bytes);
  logits_ = *logits;
  hash_tables_ = *tables;
  if (speculative()) {
    // A verify's snapshot: every byte it may write, the target's and the
    // ring's (D-068 working state).
    const std::uint64_t snapshot =
        md::Dsv4VerifySnapshotBytes(profile_, layout_, o_.max_verify) +
        (std::uint64_t{o_.max_verify} * dprofile_.blocks.layers * dprofile_.blocks.head_dim * 2);
    if (auto r = resources_.Map(snapshot_, "the DeepSeek verify snapshot", snapshot,
                                MemoryClass::kRuntime);
        !r) {
      return r;
    }
    live_.SnapshotAt(snapshot_.base, snapshot_.bytes);
    if (auto r = live_.AllocateSnapshot(resources_, kRangeCapacity); !r) {
      return r;
    }
    auto drafts =
        resources_.Pinned(std::max<std::uint64_t>(o_.draft_rows * sizeof(std::int32_t), 256));
    if (!drafts) {
      return Error("pinned staging for DeepSeek's speculation");
    }
    drafts_ = *drafts;
  }
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
  if (auto r = live_.RegisterSpill(node_, o_.out); !r) {
    return r;
  }
  // D-090: the places every graph will name stay put for the model's life.
  auto pinned_extents = weights();
  const auto reserved = live_.reserved_extents();
  pinned_extents.insert(pinned_extents.end(), reserved.begin(), reserved.end());
  if (auto pinned = node_.scheduler().PinPlaces(pinned_extents); !pinned) {
    return Error(std::format("pinning DeepSeek's places: {}", sc::ToString(pinned.error())));
  }
  return {};
}

Status Dsv4Runner::CheckPlaces() {
  PlaceCheck check;
  auto checked = node_.Call(
      [&]() -> Status {
        weights_.CheckPlaces(node_.scheduler(), check);
        dweights_.CheckPlaces(node_.scheduler(), check);
        live_.CheckPlaces(node_.scheduler(), check);
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
  return {};
}

Status Dsv4Runner::RefreshClosures() {
  auto refreshed = node_.Call(
      [&]() -> Status {
        auto& catalog = node_.catalog();
        // What every job but a draft's leases beside the weights: the state, the
        // workspace, the model's own memory and staging.
        std::vector<ExtentId> common = live_.extents();
        for (const Mapped* mapped :
             std::initializer_list<const Mapped*>{&node_.activations(), &node_.pool()}) {
          common.insert(common.end(), mapped->extents.begin(), mapped->extents.end());
        }
        const std::vector<ExtentId> own = resources_.extents();
        common.insert(common.end(), own.begin(), own.end());
        std::vector<ExtentId> all = weights();
        all.insert(all.end(), common.begin(), common.end());
        everything_ = catalog.ClosureOfExtents(all).value();
        fence_ = catalog.ClosureOfExtents(state()).value();
        if (speculative()) {
          // A draft reads the drafter's weights, the target's head (its group)
          // and token table; its job restores a verify's rows first (both states
          // and the snapshot).
          const std::uint32_t head = weights_.artifact().resources()[binding_.output.index].group;
          std::vector<ExtentId> draft = dweights_.extents();
          const std::vector<ExtentId> target = weights_.ExtentsWhere(
              [head](std::uint32_t group, bool host) { return host || group == head; });
          draft.insert(draft.end(), target.begin(), target.end());
          draft.insert(draft.end(), common.begin(), common.end());
          draft_closure_ = catalog.ClosureOfExtents(draft).value();
        }
        return {};
      },
      "refreshing DeepSeek's used state closure");
  if (!refreshed) {
    return refreshed;
  }
  return node_.RefreshRequest(stream_, everything_);
}

Status Dsv4Runner::Bind() {
  if (auto refreshed = RefreshClosures(); !refreshed) {
    return refreshed;
  }
  model_.places.resource = [this](std::uint32_t resource) {
    return weights_.resource_address(resource);
  };
  model_.places.array = [this](std::uint32_t array) { return weights_.array_address(array); };
  model_.places.state = live_.base(kTarget);
  if (speculative()) {
    dmodel_.places.resource = [this](std::uint32_t resource) {
      return dweights_.resource_address(resource);
    };
    dmodel_.places.array = [this](std::uint32_t array) { return dweights_.array_address(array); };
    dmodel_.places.state = live_.base(kDrafter);
    dmodel_.target_resource = model_.places.resource;
  }
  if (auto r = resources_.BindLaunch(scratch_bytes_); !r) {
    return r;
  }
  runs_.SetLaunch(&resources_.launch());
  return {};
}

// ------------------------------------------------------------------ work

Status Dsv4Runner::Clear() {
  const bool open = node_.InRequest(stream_);
  if (auto cleared = live_.Clear(node_, fence_, stream_, "clearing the DeepSeek state"); !cleared) {
    return cleared;
  }
  if (auto refreshed = RefreshClosures(); !refreshed) {
    return refreshed;
  }
  return open ? node_.BeginRequest(stream_, everything_, "a cleared DeepSeek conversation")
              : Status{};
}

Status Dsv4Runner::EnsureState(std::uint32_t positions) {
  auto needed = md::Dsv4UsedState(layout_, positions);
  if (!needed) {
    return std::unexpected(needed.error());
  }
  std::vector<LiveState::Range> ranges;
  for (const auto& range : *needed) {
    ranges.push_back({.region = kTarget, .offset = range.offset, .bytes = range.bytes});
  }
  if (speculative()) {
    ranges.push_back({.region = kDrafter, .offset = 0, .bytes = dlayout_.bytes});
  }
  auto used = live_.Use(node_, ranges, &everything_);
  if (!used) {
    if (auto refreshed = RefreshClosures(); !refreshed) {
      live_.Quarantine();
      return Error(std::format("{}; {}", used.error(), refreshed.error()));
    }
    return std::unexpected(used.error());
  }
  return *used ? RefreshClosures() : Status{};
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
          everything_,
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
    const ChunkKey& key) {
  if (ChunkPlans::Entry* found = plans_.Find(key); found != nullptr) {
    return found;
  }
  const auto start = std::chrono::steady_clock::now();
  Dsv4Speculation speculation;
  if (key.kind != Dsv4ChunkKind::kPlain) {
    speculation = {.verify = key.kind == Dsv4ChunkKind::kVerify,
                   .drafter = &dmodel_,
                   .inject_rows = key.inject_rows};
  }
  kg::LaunchContext& launch = resources_.launch();
  auto planned = PlanDsv4Chunk(model_, key.shape, kg::DeviceChoicesOf(launch), dump_,
                               node_.activations().base, node_.activations().bytes, speculation);
  if (!planned) {
    return std::unexpected(planned.error());
  }
  if (auto r = BindPlanned(**planned, launch, resources_.registry(), "the plan"); !r) {
    return std::unexpected(r.error());
  }
  Check((*planned)->graph);
  plan_seconds_ += Seconds(std::chrono::steady_clock::now() - start);
  return &plans_.Add(key, std::move(*planned));
}

std::expected<Dsv4Runner::DraftPlans::Entry*, std::string> Dsv4Runner::PlannedDraft() {
  if (DraftPlans::Entry* found = dplans_.Find(o_.draft_rows); found != nullptr) {
    return found;
  }
  const auto start = std::chrono::steady_clock::now();
  kg::LaunchContext& launch = resources_.launch();
  auto planned = PlanDsparkDraft(dmodel_, o_.draft_rows, kg::DeviceChoicesOf(launch),
                                 node_.activations().base, node_.activations().bytes);
  if (!planned) {
    return std::unexpected(planned.error());
  }
  if (auto r = BindPlanned(**planned, launch, resources_.registry(), "the draft"); !r) {
    return std::unexpected(r.error());
  }
  plan_seconds_ += Seconds(std::chrono::steady_clock::now() - start);
  return &dplans_.Add(o_.draft_rows, std::move(*planned));
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

void Dsv4Runner::Settle(bool saved, bool wrote, bool unknown) {
  live_.Settle(saved, wrote, unknown || resources_.launch().faulted());
}

Status Dsv4Runner::PlanSnapshot(const md::Dsv4ChunkInputs& in) {
  const md::Dsv4Writes writes = md::Dsv4ChunkWrites(profile_, layout_, in);
  const std::vector<std::vector<md::StateRange>> ring =
      md::DsparkWrites(dprofile_, dlayout_, in.n_past, in.rows);
  const std::uint64_t state = live_.base(kTarget);
  const std::uint64_t drafter = live_.base(kDrafter);
  live_.BeginSaves();
  for (std::uint32_t i = 0; i < in.rows; ++i) {
    for (const md::StateRange& r : writes.rows[i]) {
      if (auto added = live_.Save(state + r.offset, r.bytes, i); !added) {
        return added;
      }
    }
    for (const md::StateRange& r : ring[i]) {
      if (auto added = live_.Save(drafter + r.offset, r.bytes, i); !added) {
        return added;
      }
    }
  }
  for (const md::StateRange& r : writes.scratch) {
    if (auto added = live_.Save(state + r.offset, r.bytes, -1); !added) {
      return added;
    }
  }
  return {};
}

std::vector<Dsv4Runner::VerifyWrite> Dsv4Runner::last_verify_writes() const {
  const std::uint64_t ring_base = live_.base(kDrafter);
  const std::uint64_t ring_bytes = live_.bytes(kDrafter);
  std::vector<VerifyWrite> out;
  out.reserve(live_.saved().size());
  for (const LiveState::Saved& s : live_.saved()) {
    const bool ring = s.address >= ring_base && s.address < ring_base + ring_bytes;
    out.push_back({.ring = ring,
                   .offset = s.address - (ring ? ring_base : live_.base(kTarget)),
                   .bytes = s.bytes,
                   .row = s.row});
  }
  return out;
}

Status Dsv4Runner::Rollback() {
  return live_.Rollback(node_, everything_, stream_, resources_.launch(),
                        "restoring a verify's rejected rows");
}

Status Dsv4Runner::Chunk(std::uint32_t n_past, std::span<const std::int32_t> tokens,
                         std::vector<float>& logits, const std::function<Status()>& meanwhile,
                         Dsv4ChunkKind kind) {
  const auto rows = static_cast<std::uint32_t>(tokens.size());
  if (kind != Dsv4ChunkKind::kPlain && !speculative()) {
    return Error("an injection or a verify needs the drafter");
  }
  if (auto usable = live_.Usable(); !usable) {
    return usable;
  }
  if (auto waiting = live_.AwaitingAccept(); !waiting) {
    return waiting;
  }
  auto in = md::Dsv4Chunk(profile_, layout_, n_past, rows, model_.exact);
  if (!in) {
    return std::unexpected(in.error());
  }
  if (auto used = EnsureState(n_past + rows); !used) {
    return used;
  }
  const bool verify = kind == Dsv4ChunkKind::kVerify;
  md::DsparkInjection inject;
  if (kind != Dsv4ChunkKind::kPlain) {
    inject = md::DsparkInject(dlayout_, n_past, rows);
  }
  if (verify) {
    if (rows > o_.max_verify || !md::Dsv4SameWidths(layout_, n_past, rows)) {
      return Error(std::format("a verify of {} rows at {}: at most {}, at its steps' mask widths",
                               rows, n_past, o_.max_verify));
    }
    if (auto r = PlanSnapshot(*in); !r) {
      return r;
    }
  }
  auto planned = Planned({.shape = kg::Dsv4ShapeOf(layout_, *in),
                          .kind = kind,
                          .inject_rows = static_cast<std::int64_t>(inject.cells.size())});
  if (!planned) {
    return std::unexpected(planned.error());
  }
  ChunkPlans::Entry& entry = **planned;
  PlanRuns& runs = entry.runs[0];
  Dsv4Planned* p = entry.planned.get();
  const kg::Dsv4Graph& g = p->graph;
  const std::uint64_t row_bytes = std::uint64_t{profile_.vocab} * sizeof(float);
  const std::uint32_t out_rows = verify ? rows : 1;
  // Decode graphs (D-090): replay a shape's graph; capture a one-row
  // shape (or a verify's) that has run once launch by launch; otherwise
  // launch by launch.
  const bool capture =
      runs.CaptureDue(runs_.graphs()) && ((rows == 1 && kind == Dsv4ChunkKind::kPlain) || verify);
  if (capture) {
    RoomForGraph(kMaxGraphs, graph_stats_, plans_);
  }
  // The last row's logits (the next token's), or a verify's every row's.
  const std::array<RunCopy, 1> outputs = {
      RunCopy{Address(logits_),
              Address(static_cast<const std::byte*>(g.logits->data) +
                      (std::uint64_t{rows - out_rows} * row_bytes)),
              out_rows * row_bytes}};
  kg::LaunchContext& launch = resources_.launch();
  RunPath path = RunPath::kEager;
  Status ran;
  Dsv4HostInputs host;
  // What the job queued, for a failure's settling (Settle).
  bool saved = false;    // this verify's snapshot
  bool wrote = false;    // anything that may write the state
  bool unknown = false;  // a launch of unknown effect
  auto job = [&](providers::NativeStream native) -> sc::JobResult {
    const auto started = std::chrono::steady_clock::now();
    // A rejected draft's rows first, then this verify's snapshot.
    const bool restoring = live_.owed();
    if (auto r = live_.QueueOwed(launch); !r) {
      ran = Error(std::format("chunk at {}: {}", n_past, r.error().detail));
      unknown = true;
      return sc::JobResult::kUnknown;
    }
    bool before = restoring;  // work queued before a failure
    if (verify) {
      if (auto r = live_.QueueSaves(launch); !r) {
        ran = Error(std::format("chunk at {}: {}", n_past, r.error().detail));
        unknown = true;
        return sc::JobResult::kUnknown;
      }
      before = true;
      saved = true;
    }
    // The embedding rows and the chunk plan's inputs, staged in order.
    if (auto r = BuildDsv4Inputs(model_, g, *in, tokens, table(), host, inject.cells); !r) {
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
  Status posted;
  Status alongside;
  if (meanwhile) {
    // Submitted without waiting; the frame (and `job`'s references) lives
    // until Await has seen the program gone.
    sc::ProgramDone done;
    const std::uint64_t request =
        node_.Submit(std::make_unique<sc::RunProgram>(done, everything_, std::move(job), stream_));
    alongside = meanwhile();
    posted = node_.Await(done, "a DeepSeek chunk", request);
  } else {
    posted = node_.Job(everything_, std::move(job), "a DeepSeek chunk", stream_);
  }
  if (!posted || !ran || !alongside || launch.faulted()) {
    // Never left half-written: a verify is undone, anything else that may
    // have written the state quarantines it.
    Settle(saved, wrote, unknown);
    if (launch.faulted()) {
      return Error(std::format("chunk at {}: the launch context faulted", n_past));
    }
    if (!ran) {
      return ran;
    }
    return Error(
        std::format("chunk at {}: {}", n_past, !posted ? posted.error() : alongside.error()));
  }
  last_path_ = path;
  last_planned_ = p;
  Count(graph_stats_, path);
  if (verify) {
    live_.Verified(rows);
  }
  const auto* values = static_cast<const float*>(logits_);
  logits.assign(values, values + (std::size_t{out_rows} * profile_.vocab));
  return {};
}

Status Dsv4Runner::Draft(std::uint32_t pos0, std::int32_t anchor,
                         std::vector<std::int32_t>& drafts) {
  if (!speculative()) {
    return Error("drafting needs the drafter");
  }
  if (auto usable = live_.Usable(); !usable) {
    return usable;
  }
  if (auto waiting = live_.AwaitingAccept(); !waiting) {
    return waiting;
  }
  auto in = md::DsparkBlock(dprofile_, dlayout_, pos0, anchor, o_.draft_rows);
  if (!in) {
    return std::unexpected(in.error());
  }
  if (auto used = EnsureState(pos0); !used) {
    return used;
  }
  auto planned = PlannedDraft();
  if (!planned) {
    return std::unexpected(planned.error());
  }
  DraftPlans::Entry& entry = **planned;
  PlanRuns& runs = entry.runs[0];
  DsparkPlanned* p = entry.planned.get();
  const kg::DsparkGraph& g = p->graph;
  const std::array<RunCopy, 1> outputs = {
      RunCopy{Address(drafts_), Address(g.drafts->data),
              std::uint64_t{o_.draft_rows} * sizeof(std::int32_t)}};
  const bool capture = runs.CaptureDue(runs_.graphs());
  kg::LaunchContext& launch = resources_.launch();
  RunPath path = RunPath::kEager;
  Status ran;
  Dsv4HostInputs host;
  bool unknown = false;  // a launch of unknown effect
  auto job = [&](providers::NativeStream native) -> sc::JobResult {
    const auto started = std::chrono::steady_clock::now();
    if (auto r = live_.QueueOwed(launch); !r) {
      ran = Error(std::format("draft at {}: {}", pos0, r.error().detail));
      unknown = true;
      return sc::JobResult::kUnknown;
    }
    if (auto r = BuildDsparkInputs(model_, g, *in, table(), host); !r) {
      ran = std::unexpected(r.error());
      return sc::JobResult::kFailed;
    }
    auto copies = runs_.Stage(host.sources, 0);
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
    // A draft writes only its own block's ring cells, past the committed
    // positions; a launch of unknown effect quarantines the state.
    Settle(false, false, unknown);
    if (launch.faulted()) {
      return Error(std::format("draft at {}: the launch context faulted", pos0));
    }
    return !ran ? ran : Error(std::format("draft at {}: {}", pos0, posted.error()));
  }
  Count(draft_stats_, path);
  const auto* values = static_cast<const std::int32_t*>(drafts_);
  drafts.assign(values, values + o_.draft_rows);
  return {};
}

std::expected<ggml_tensor*, std::string> Dsv4Runner::DraftRowsNode(std::uint32_t n,
                                                                   const void* drafts) {
  if (n == 0 || n > o_.draft_rows) {
    return Error("no drafts' rows to look up");
  }
  if (!rows_arena_) {
    auto arena = kg::TensorArena::Create(64);
    if (!arena) {
      return Error(arena.error().detail);
    }
    rows_arena_.emplace(std::move(*arena));
  }
  if (draft_rows_.empty()) {
    ggml_context* c = rows_arena_->context();
    const md::Dsv4Tensor& embedding = binding_.token_embd;
    auto type = kg::GgmlTypeOf(embedding.type);
    if (!type) {
      return Error(type.error().detail);
    }
    ggml_tensor* table = ggml_new_tensor_2d(c, *type, profile_.width, profile_.vocab);
    kg::TensorArena::Bind(table, weights_.resource_address(embedding.index));
    for (std::uint32_t k = 1; k <= o_.draft_rows; ++k) {
      ggml_tensor* ids = ggml_new_tensor_1d(c, GGML_TYPE_I32, k);
      kg::TensorArena::Bind(ids, Address(drafts));
      ggml_tensor* rows = ggml_get_rows(c, table, ids);
      // The verify's embedding input is staged first: row 0 the anchor's,
      // the drafts' after it.
      kg::TensorArena::Bind(rows, Address(runs_.staging()) + verify_base_ +
                                      (std::uint64_t{profile_.width} * sizeof(float)));
      draft_rows_.push_back(rows);
    }
  }
  // A draft planned again (DropPlans) may place its drafts elsewhere.
  kg::TensorArena::Bind(draft_rows_[n - 1]->src[1], Address(drafts));
  return draft_rows_[n - 1];
}

Status Dsv4Runner::DraftVerify(std::uint32_t pos, std::int32_t anchor, std::uint32_t rows,
                               std::vector<std::int32_t>& drafts, std::vector<float>& logits) {
  if (!speculative()) {
    return Error("drafting needs the drafter");
  }
  if (auto usable = live_.Usable(); !usable) {
    return usable;
  }
  if (auto waiting = live_.AwaitingAccept(); !waiting) {
    return waiting;
  }
  if (rows == 0 || rows > o_.max_verify || rows > o_.draft_rows + 1 ||
      !md::Dsv4SameWidths(layout_, pos, rows)) {
    return Error(
        std::format("a verify of {} rows at {}: at most {} and the drafts, at its steps' "
                    "mask widths",
                    rows, pos, o_.max_verify));
  }
  // The draft.
  auto block = md::DsparkBlock(dprofile_, dlayout_, pos, anchor, o_.draft_rows);
  if (!block) {
    return std::unexpected(block.error());
  }
  if (auto used = EnsureState(pos + rows); !used) {
    return used;
  }
  auto dplanned = PlannedDraft();
  if (!dplanned) {
    return std::unexpected(dplanned.error());
  }
  DraftPlans::Entry& dentry = **dplanned;
  PlanRuns& druns = dentry.runs[0];
  const kg::DsparkGraph& dg = dentry.planned->graph;
  const bool dcapture = druns.CaptureDue(runs_.graphs());
  // The verify: its tokens the anchor and placeholders the drafts replace
  // on the device.
  auto in = md::Dsv4Chunk(profile_, layout_, pos, rows, model_.exact);
  if (!in) {
    return std::unexpected(in.error());
  }
  const md::DsparkInjection inject = md::DsparkInject(dlayout_, pos, rows);
  if (auto r = PlanSnapshot(*in); !r) {
    return r;
  }
  auto planned = Planned({.shape = kg::Dsv4ShapeOf(layout_, *in),
                          .kind = Dsv4ChunkKind::kVerify,
                          .inject_rows = static_cast<std::int64_t>(inject.cells.size())});
  if (!planned) {
    return std::unexpected(planned.error());
  }
  ChunkPlans::Entry& ventry = **planned;
  PlanRuns& vruns = ventry.runs[0];
  const kg::Dsv4Graph& vg = ventry.planned->graph;
  const bool vcapture = vruns.CaptureDue(runs_.graphs());
  if (vcapture) {
    RoomForGraph(kMaxGraphs, graph_stats_, plans_);
  }
  ggml_tensor* lookup = nullptr;
  if (rows > 1) {
    auto node = DraftRowsNode(rows - 1, dg.drafts->data);
    if (!node) {
      return std::unexpected(node.error());
    }
    lookup = *node;
  }
  const std::vector<std::int32_t> placeholders(rows, anchor);
  const std::uint64_t row_bytes = std::uint64_t{profile_.vocab} * sizeof(float);
  const std::array<RunCopy, 1> doutputs = {
      RunCopy{Address(drafts_), Address(dg.drafts->data),
              std::uint64_t{o_.draft_rows} * sizeof(std::int32_t)}};
  const std::array<RunCopy, 1> voutputs = {
      RunCopy{Address(logits_), Address(vg.logits->data), rows * row_bytes}};
  kg::LaunchContext& launch = resources_.launch();
  std::byte* const staging = runs_.staging();
  Queued dq;
  Queued vq;
  Status ran;
  Dsv4HostInputs dhost;
  Dsv4HostInputs vhost;
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
    if (auto r = live_.QueueOwed(launch); !r) {
      return failed(r.error().detail, true);
    }
    // Both stagings first: the host writes them before anything it queues
    // reads them.
    if (auto r = BuildDsparkInputs(model_, dg, *block, table(), dhost); !r) {
      return failed(r.error(), false);
    }
    auto dcopies = runs_.Stage(dhost.sources, 0);
    if (!dcopies) {
      return failed(dcopies.error(), false);
    }
    // The draft's staging must end before the verify's begins.
    for (const auto& copy : *dcopies) {
      if (copy[2] + copy[1] > verify_base_) {
        return failed("the draft's inputs reach the verify's staging", false);
      }
    }
    if (auto r = BuildDsv4Inputs(model_, vg, *in, placeholders, table(), vhost, inject.cells); !r) {
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
      if (!providers::CopyAsync(native, staging + tokens_at + sizeof(std::int32_t), dg.drafts->data,
                                std::uint64_t{rows - 1} * sizeof(std::int32_t),
                                providers::CopyKind::kDeviceToHost)
               .ok()) {
        return failed("the drafts' copy into the verify's tokens", true);
      }
      if (auto r = kg::GetRowsExt(launch, lookup); !r) {
        return failed(r.error().detail, r.error().error == kg::KernelError::kUnknown);
      }
    }
    if (auto r = live_.QueueSaves(launch); !r) {
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
  auto posted = node_.Job(everything_, std::move(job), "a DSpark draft and its verify", stream_);
  if (!posted || !ran || launch.faulted()) {
    // Never left half-written: the verify is undone before the next job's
    // work, or the state quarantined.
    Settle(saved, false, unknown_effect);
    if (launch.faulted()) {
      return Error(std::format("draft and verify at {}: the launch context faulted", pos));
    }
    return !ran ? ran : Error(std::format("draft and verify at {}: {}", pos, posted.error()));
  }
  Count(draft_stats_, dq.path);
  Count(graph_stats_, vq.path);
  last_path_ = vq.path;
  live_.Verified(rows);
  const auto* values = static_cast<const std::int32_t*>(drafts_);
  drafts.assign(values, values + o_.draft_rows);
  const auto* rows_out = static_cast<const float*>(logits_);
  logits.assign(rows_out, rows_out + (std::size_t{rows} * profile_.vocab));
  return {};
}

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
        everything_,
        [&](providers::NativeStream native) {
          if (!providers::CopyAsync(native, ids->data, staging, kBatch * sizeof(std::int32_t),
                                    providers::CopyKind::kHostToDevice)
                   .ok()) {
            return sc::JobResult::kUnknown;
          }
          if (auto r = kg::GetRowsExt(launch, rows); !r) {
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

Status Dsv4Runner::SaveUsedState(void* host, std::span<const LiveState::Range> ranges) {
  return live_.Copy(node_, fence_, stream_, host, ranges, true);
}

Status Dsv4Runner::RestoreUsedState(void* host, std::span<const LiveState::Range> ranges) {
  if (host == nullptr &&
      std::ranges::any_of(ranges, [](const LiveState::Range& r) { return r.bytes != 0; })) {
    return Error("the conversation snapshot has no source buffer");
  }
  if (auto prepared = PrepareRestoreState(ranges); !prepared) {
    return prepared;
  }
  return CopyCheckpointState(host, ranges, false);
}

Status Dsv4Runner::PrepareRestoreState(std::span<const LiveState::Range> ranges) {
  const bool requested = node_.InRequest(stream_);
  if (requested) {
    if (auto ended = node_.EndRequest(stream_); !ended) {
      return ended;
    }
  }
  auto prepared = [&]() -> Status {
    if (auto used = live_.Use(node_, ranges, &everything_); !used) {
      return std::unexpected(used.error());
    }
    return live_.Retain(node_, ranges);
  }();
  if (auto refreshed = RefreshClosures(); !refreshed) {
    return refreshed;
  }
  if (requested) {
    if (auto opened = node_.BeginRequest(stream_, everything_, "restored conversation"); !opened) {
      return opened;
    }
  }
  if (!prepared) {
    return prepared;
  }
  return {};
}

Status Dsv4Runner::CopyCheckpointState(void* host, std::span<const LiveState::Range> ranges,
                                       bool to_host) {
  auto copied = live_.Copy(node_, fence_, stream_, host, ranges, to_host);
  if (!copied) {
    live_.Quarantine();
  }
  return copied;
}

std::expected<std::vector<LiveState::Range>, std::string> Dsv4Runner::CheckpointRanges(
    std::uint32_t positions) const {
  if (auto usable = live_.Usable(); !usable) {
    return std::unexpected(usable.error());
  }
  if (auto settled = live_.AwaitingAccept(); !settled) {
    return std::unexpected(settled.error());
  }
  if (live_.owed()) {
    return Error("checkpoint has an unsettled DeepSeek verify");
  }
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
  return CheckpointPages(live_.used_ranges(), writes);
}

Status Dsv4Runner::ReadState(std::vector<std::byte>& target, std::vector<std::byte>& drafter) {
  if (auto usable = live_.Usable(); !usable) {
    return usable;
  }
  if (live_.owed() || live_.verify_rows() != 0) {
    return Error("reading the state with a verify's rollback pending");
  }
  const std::array<std::vector<std::byte>*, 2> out = {&target, &drafter};
  return live_.Read(node_, fence_, stream_, "reading the DeepSeek state", out);
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
      everything_,
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
  if (posted) {
    for (std::size_t i = 0; i < reads.size(); ++i) {
      const auto* bytes = static_cast<const std::byte*>(host) + reads[i].at;
      out[i].bytes.assign(bytes, bytes + reads[i].bytes);
    }
  }
  providers::FreePinned(host);
  return posted;
}

std::expected<double, std::string> Dsv4Runner::TimeReplays(std::uint32_t n_past, std::int32_t token,
                                                           std::uint32_t count) {
  auto in = md::Dsv4Chunk(profile_, layout_, n_past, 1, model_.exact);
  if (!in) {
    return std::unexpected(in.error());
  }
  auto planned = Planned({.shape = kg::Dsv4ShapeOf(layout_, *in)});
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
  const auto start = std::chrono::steady_clock::now();
  auto posted = node_.Job(
      everything_,
      [&](providers::NativeStream) {
        for (std::uint32_t i = 0; i < count; ++i) {
          if (auto r = launch.Launch(graph); !r) {
            failed = r.error().detail;
            if (r.error().error == kg::KernelError::kUnknown) {
              return sc::JobResult::kUnknown;
            }
            return i == 0 ? sc::JobResult::kNotStarted : sc::JobResult::kFailed;
          }
        }
        return sc::JobResult::kQueued;
      },
      "back-to-back replays", stream_);
  const double seconds = Seconds(std::chrono::steady_clock::now() - start);
  if (!posted || !failed.empty()) {
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
  live_.Release(memory, problems);
  for (PagedWeights* part : {&weights_, &dweights_}) {
    if (auto r = part->Release(memory); !r) {
      problems.push_back(std::format("DeepSeek: {}", r.error()));
    }
  }
  return support::Joined(problems);
}

}  // namespace jitllm::engine
