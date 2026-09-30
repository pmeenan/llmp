// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "engine/qwen38_runner.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <expected>
#include <format>
#include <initializer_list>
#include <optional>
#include <string>
#include <tuple>
#include <utility>

#include "artifact/layout.h"
#include "engine/checkpoint_file.h"
#include "engine/support.h"
#include "ggml.h"
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

}  // namespace

Qwen38Runner::~Qwen38Runner() = default;

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
  table_ = PleTable{.fd = weights_.shard_fd(first->shard),
                    .file_offset = first->file_offset.value() + table.offset.value(),
                    .rows = binding_.ple_table.ne[1],
                    .row_bytes = binding_.ple_table.ne[0],
                    .chunk_file_offset = first->file_offset.value(),
                    .file_bytes = first->file_offset.value() + groups[table_group].stored.value()};
  slots_ = std::uint64_t{o_.max_rows} * profile_.ple_heads();
  graph_binding_ = binding_;
  graph_binding_.ple_table.ne[1] = slots_;

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
  std::uint64_t most_scratch = 0;
  std::uint64_t most_inputs = 0;
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
      }
    }
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
      if (auto r = account(**planned); !r) {
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
        if (auto r = account(**planned); !r) {
          return r;
        }
        if (shaped->first.capture_head) {
          shaped->first.capture_head = false;
          auto ordinary = PlanQwen38Mtp(model_, shaped->first, choices, 0, 0);
          if (!ordinary) {
            return Error(std::format("measuring the uncaptured drafter: {}", ordinary.error()));
          }
          if (auto r = account(**ordinary); !r) {
            return r;
          }
        }
      }
    }
  }
  activation_bytes_ = Round(most_activations + (most_activations / 4), kExtent);
  scratch_bytes_ = Round(most_scratch + (most_scratch / 4) + (1U << 20U), kExtent);
  const std::uint64_t input_bytes = Round((most_inputs * 2) + (1U << 20U), kExtent);
  // A chunk's host-built inputs are the staged bytes again, on the host.
  host_input_bytes_ = Round(most_inputs + (1U << 20U), kExtent);
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
    // A verify's saves, then the snapshot of the cells it writes (D-068
    // working state), mapped for the model's life.
    const std::uint32_t qsa = profile_.layers / 4;
    const std::uint64_t row_cells = (2 * std::uint64_t{profile_.head_dim} * profile_.kv_heads * 2) +
                                    (std::uint64_t{profile_.indexer_head_dim} * 4);
    const std::uint64_t snapshot_offset = Round(commit_layout_.bytes, 256);
    // (And the block keys a verify completes: at most one a ratio of rows,
    // and one more across a block's boundary.)
    const std::uint64_t block_keys =
        ((std::uint64_t{o_.draft_rows + 1} / profile_.indexer_ratio) + 1) * qsa *
        std::uint64_t{profile_.indexer_head_dim} * 2;
    const std::uint64_t commit_bytes =
        snapshot_offset +
        Round((std::uint64_t{o_.draft_rows + 1} * qsa * row_cells) + block_keys + 4096, 256);
    if (auto r = resources_.Map(commit_, "the Qwen3.8 verify's saves", commit_bytes,
                                MemoryClass::kRuntime);
        !r) {
      return r;
    }
    live_.SnapshotAt(commit_.base + snapshot_offset, commit_.bytes - snapshot_offset);
    if (auto r = live_.AllocateSnapshot(resources_, kRangeCapacity); !r) {
      return r;
    }
    auto carry = resources_.Pinned(4 * sizeof(kg::RangeCopy));
    // A draft's drafts, their probabilities (from kProbabilityAt), then
    // (from kArgmaxAt) a verify's argmaxes.
    auto drafts = resources_.Pinned(512);
    if (!carry || !drafts) {
      return Error("pinned staging for Qwen3.8's speculation");
    }
    carry_ = static_cast<kg::RangeCopy*>(*carry);
    drafts_ = *drafts;
  }
  auto ring = providers::OpenStorage(kRingDepth);
  if (!ring) {
    return Error(std::format("the n-gram rows' ring: {}", ring.error().message()));
  }
  ring_ = std::move(*ring);
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
  if (auto r = live_.RegisterSpill(node_, o_.out); !r) {
    return r;
  }
  // D-090, for every model: the places stay put for the model's life
  // (never unpinned; the pins go with the scheduler).
  auto pinned_extents = weights();
  const auto reserved = live_.reserved_extents();
  pinned_extents.insert(pinned_extents.end(), reserved.begin(), reserved.end());
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
        live_.CheckPlaces(node_.scheduler(), check);
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

Status Qwen38Runner::RefreshClosures() {
  auto refreshed = node_.Call(
      [&]() -> Status {
        auto& catalog = node_.catalog();
        std::vector<ExtentId> all = weights();
        const std::vector<ExtentId> live = live_.extents();
        all.insert(all.end(), live.begin(), live.end());
        for (const Mapped* mapped :
             std::initializer_list<const Mapped*>{&node_.activations(), &node_.pool()}) {
          all.insert(all.end(), mapped->extents.begin(), mapped->extents.end());
        }
        const std::vector<ExtentId> own = resources_.extents();
        all.insert(all.end(), own.begin(), own.end());
        everything_ = catalog.ClosureOfExtents(all).value();
        fence_ = catalog.ClosureOfExtents(state()).value();
        return {};
      },
      "refreshing Qwen3.8's used state closure");
  if (!refreshed) {
    return refreshed;
  }
  return node_.RefreshRequest(stream_, everything_);
}

Status Qwen38Runner::Bind() {
  if (auto refreshed = RefreshClosures(); !refreshed) {
    return refreshed;
  }
  model_.places.resource = [this](std::uint32_t resource) {
    return weights_.resource_address(resource);
  };
  model_.places.array = [this](std::uint32_t array) { return weights_.array_address(array); };
  model_.places.state = live_.base(kTarget);
  model_.places.ple_table = slot_memory_.base;
  if (speculative()) {
    model_.places.mtp_resource = [this](std::uint32_t resource) {
      return dweights_.resource_address(resource);
    };
    model_.places.mtp_array = [this](std::uint32_t array) {
      return dweights_.array_address(array);
    };
    model_.places.mtp_state = live_.base(kDrafter);
    model_.places.commit = commit_.base;
    // The commit's places: every linear-attention layer's state and saves.
    using K = md::Qwen38StateTensor::Kind;
    const std::uint64_t state = live_.base(kTarget);
    const auto at = [&](std::uint32_t il, K kind) {
      return state + layout_.tensors[static_cast<std::size_t>(layout_.Find(il, kind))].offset;
    };
    kg::Qwen38CommitArgs& c = commit_args_;
    c.layers = static_cast<int>(commit_layout_.layers.size());
    c.channels = static_cast<int>(profile_.conv_channels());
    c.qk_heads = static_cast<int>(profile_.lin_k_heads);
    c.v_heads = static_cast<int>(profile_.lin_v_heads);
    c.taps = static_cast<int>(profile_.conv - 1);
    for (std::size_t i = 0; i < commit_layout_.layers.size(); ++i) {
      const std::uint32_t il = commit_layout_.layers[i];
      const std::uint64_t base = commit_.base;
      c.layer[i] = {.state = static_cast<float*>(Pointer(at(il, K::kRecurrent))),
                    .history = static_cast<float*>(Pointer(at(il, K::kConv))),
                    .conv = static_cast<const float*>(Pointer(base + commit_layout_.conv_out(i))),
                    .qkv = static_cast<const float*>(Pointer(base + commit_layout_.qkv(i))),
                    .gate = static_cast<const float*>(Pointer(base + commit_layout_.gate(i))),
                    .beta = static_cast<const float*>(Pointer(base + commit_layout_.beta(i)))};
    }
    c.ple_history = static_cast<float*>(Pointer(at(profile_.ple_layer, K::kPleConv)));
    c.ple_rows = static_cast<const float*>(Pointer(commit_.base + commit_layout_.ple()));
    c.ple_width = static_cast<int>(profile_.hc_width());
    c.ple_taps = static_cast<int>(profile_.ple_history());
    // A verify's kept rows committed after the rejected rows' restore.
    live_.SetCommit([this](kg::LaunchContext& launch, std::uint32_t keep) {
      kg::Qwen38CommitArgs args = commit_args_;
      args.keep = static_cast<int>(keep);
      return kg::Qwen38Commit(launch, args);
    });
  }
  if (auto r = resources_.BindLaunch(scratch_bytes_); !r) {
    return r;
  }
  runs_.SetLaunch(&resources_.launch());
  return {};
}

// ------------------------------------------------------------------ work

Status Qwen38Runner::ReadPleHash() {
  hash_checked_ = false;
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
          everything_,
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
            everything_,
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
  std::uint32_t count = 0;
  for (const PagedWeights::Range& r : unwritten_) {
    if (r.slab ? slabs : dense) {
      scrub_[2 * std::size_t{count}] = r.address;
      scrub_[(2 * std::size_t{count}) + 1] = r.bytes;
      ++count;
    }
  }
  const std::uint64_t* ranges = scrub_;
  return node_.Job(
      everything_,
      [ranges, count, value](providers::NativeStream stream) {
        return kernels::paging::FillRanges(ranges, count, value, stream.handle)
                   ? sc::JobResult::kQueued
                   : sc::JobResult::kUnknown;
      },
      "filling the weights' unwritten bytes", stream_);
}

Status Qwen38Runner::Clear() {
  const bool open = node_.InRequest(stream_);
  pending_rows_ = 0;
  if (auto cleared = live_.Clear(node_, fence_, stream_, "clearing the Qwen3.8 state"); !cleared) {
    return cleared;
  }
  if (auto refreshed = RefreshClosures(); !refreshed) {
    return refreshed;
  }
  return open ? node_.BeginRequest(stream_, everything_, "a cleared Qwen3.8 conversation")
              : Status{};
}

Status Qwen38Runner::EnsureState(std::uint32_t positions) {
  auto needed = md::Qwen38UsedState(profile_, layout_, positions);
  if (!needed) {
    return std::unexpected(needed.error());
  }
  std::vector<LiveState::Range> ranges;
  for (const auto& range : *needed) {
    ranges.push_back({.region = kTarget, .offset = range.offset, .bytes = range.bytes});
  }
  if (speculative()) {
    const auto cells = std::min<std::uint64_t>(mtp_layout_.cells, Round(positions, 256));
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

std::expected<Qwen38Runner::ChunkPlans::Entry*, std::string> Qwen38Runner::Planned(
    const ChunkKey& key) {
  if (ChunkPlans::Entry* found = plans_.Find(key); found != nullptr) {
    return found;
  }
  const auto start = std::chrono::steady_clock::now();
  kg::LaunchContext& launch = resources_.launch();
  auto planned = PlanQwen38Chunk(model_, key.shape, kg::DeviceChoicesOf(launch),
                                 node_.activations().base, node_.activations().bytes, {}, key.kind);
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

std::expected<Qwen38Runner::MtpPlans::Entry*, std::string> Qwen38Runner::PlannedMtp(
    const kg::Qwen38MtpShape& shape) {
  if (MtpPlans::Entry* found = mplans_.Find(shape); found != nullptr) {
    return found;
  }
  const auto start = std::chrono::steady_clock::now();
  kg::LaunchContext& launch = resources_.launch();
  auto planned = PlanQwen38Mtp(model_, shape, kg::DeviceChoicesOf(launch), node_.activations().base,
                               node_.activations().bytes);
  if (!planned) {
    return std::unexpected(planned.error());
  }
  if (auto r = BindPlanned(**planned, launch, resources_.registry(), "the drafter"); !r) {
    return std::unexpected(r.error());
  }
  CheckMtp((*planned)->graph);
  plan_seconds_ += Seconds(std::chrono::steady_clock::now() - start);
  return &mplans_.Add(shape, std::move(*planned));
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
  const auto reading = std::chrono::steady_clock::now();
  auto rows_plan = PlanPleRows(table_, in.ple_rows, landing_bytes_, slots_);
  if (!rows_plan) {
    return std::unexpected(rows_plan.error());
  }
  // The gather's grid is the chunk's lookups (rows × heads) and the device
  // reads the count: it must never exceed them, nor the slots.
  if (rows_plan->sources.size() > in.ple_rows.size() || rows_plan->sources.size() > slots_) {
    return std::unexpected(std::format("{} n-gram rows for {} lookups and {} slots",
                                       rows_plan->sources.size(), in.ple_rows.size(), slots_));
  }
  if (auto r = ReadPleRows(*ring_, table_.fd, *rows_plan, landing_); !r) {
    rows_stalled_ = ring_->in_flight() != 0;  // reads that may still land
    return std::unexpected(r.error());
  }
  std::ranges::copy(rows_plan->sources, sources_);
  *ple_count_ = static_cast<std::uint32_t>(rows_plan->sources.size());
  ple_.chunks += 1;
  ple_.lookups += in.ple_rows.size();
  ple_.rows += rows_plan->sources.size();
  ple_.reads += rows_plan->reads.size();
  ple_.read_bytes += rows_plan->landing_bytes;
  ple_.useful_bytes += rows_plan->useful_bytes;
  ple_.extent_bytes += rows_plan->extents * kExtent;
  ple_.seconds += Seconds(std::chrono::steady_clock::now() - reading);
  return std::move(rows_plan->slots);
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
  // An undone verify wrote no other target state, but its streams rows may
  // have overwritten the ones the next draft would catch up on (rows 1 ..):
  // none is pending until a chunk with the injection or an accepted verify
  // writes them again.
  if (live_.Settle(saved, wrote, unknown || resources_.launch().faulted())) {
    pending_rows_ = 0;
  }
}

Status Qwen38Runner::Usable() const {
  if (!hash_checked_) {
    return Error("the n-gram hash is not checked since the last load");
  }
  if (rows_stalled_) {
    return Error("the n-gram rows' reads stalled earlier; their landing may still be written");
  }
  return live_.Usable();
}

std::expected<std::pair<kg::Qwen38MtpShape, std::vector<md::Qwen38ChunkInputs>>, std::string>
Qwen38Runner::MtpInputs(std::uint32_t first, std::uint32_t rows, std::uint32_t passes, bool head,
                        std::int64_t hidden_row, bool confidence, bool capture_head) const {
  const std::uint64_t end = std::uint64_t{first} + rows + passes - 1;
  if (rows == 0 || passes == 0 || end > mtp_layout_.context) {
    return Error(std::format("a draft of {} passes after {} rows at {} passes the context", passes,
                             rows, first));
  }
  const auto n_kv =
      static_cast<std::uint32_t>(std::min<std::uint64_t>(Round(end, 256), mtp_layout_.cells));
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

Status Qwen38Runner::Chunk(std::span<const std::int32_t> history, std::uint32_t n_past,
                           std::vector<float>& logits, bool inject) {
  if (auto usable = Usable(); !usable) {
    return usable;
  }
  if (auto waiting = live_.AwaitingAccept(); !waiting) {
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
  if (auto used = EnsureState(n_past + rows); !used) {
    return used;
  }
  auto slots = ReadRows(*in);
  if (!slots) {
    return std::unexpected(slots.error());
  }
  const Qwen38ChunkKind kind{.verify = false, .export_streams = inject};
  auto planned = Planned({.shape = kg::Qwen38ShapeOf(layout_, *in, 1), .kind = kind});
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
      auto mplanned = PlannedMtp(shaped->first);
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
    const std::uint64_t streams = live_.base(kDrafter) + mtp_layout_.hidden;
    carry_[carries++] = {.from = streams + (rows * row), .to = streams, .bytes = row};
    if (rows != 1) {
      carry_[carries++] = {.from = streams + (rows * row), .to = streams + row, .bytes = row};
    }
  }
  const std::uint64_t row_bytes = std::uint64_t{profile_.vocab} * sizeof(float);
  const std::function<bool(void*)> gather = Gather(rows);
  // Decode graphs (D-090): replay a shape's graph; capture a one-row shape
  // that has run once launch by launch; otherwise launch by launch.
  const bool capture = runs.CaptureDue(runs_.graphs()) && rows == 1 && !inject;
  if (capture) {
    RoomForGraph(kMaxGraphs, graph_stats_, plans_, mplans_);
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
    const bool committing = live_.owed();
    if (auto r = live_.QueueOwed(launch); !r) {
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
      if (auto r = kg::CopyRanges(launch, carry_, carries); !r) {
        ran = Error(std::format("the drafter's pending row at {}: {}", n_past, r.error().detail));
        unknown = true;
        return sc::JobResult::kUnknown;
      }
    }
    return sc::JobResult::kQueued;
  };
  const Status posted = node_.Job(everything_, std::move(job), "a Qwen3.8 chunk", stream_);
  if (!posted || !ran || launch.faulted()) {
    Settle(false, wrote, unknown);
    if (launch.faulted()) {
      return Error(std::format("chunk at {}: the launch context faulted", n_past));
    }
    return !ran ? ran : Error(std::format("chunk at {}: {}", n_past, posted.error()));
  }
  Count(graph_stats_, path);
  // With the injection the chunk's last row is the pending one; without,
  // the streams rows no longer precede the anchor, so no draft may read
  // them until a chunk with the injection or an accepted verify.
  pending_rows_ = inject ? 1 : 0;
  const auto* values = static_cast<const float*>(logits_);
  logits.assign(values, values + profile_.vocab);
  return {};
}

Status Qwen38Runner::Draft(std::span<const std::int32_t> history, std::vector<std::int32_t>& drafts,
                           std::vector<float>* probabilities, std::uint32_t passes,
                           Qwen38DraftHeadCapture* head_capture) {
  if (head_capture != nullptr) {
    *head_capture = {};
    if (!o_.draft_head_capture || draft_head_capture_ == nullptr || capture_head_rows_ == 0) {
      return Error("draft-head capture was not provisioned at setup");
    }
  }
  if (!speculative()) {
    return Error("drafting needs the drafter");
  }
  if (auto usable = Usable(); !usable) {
    return usable;
  }
  if (auto waiting = live_.AwaitingAccept(); !waiting) {
    return waiting;
  }
  const std::uint32_t rows = pending_rows_;
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
  if (auto used = EnsureState(n + passes - 1); !used) {
    return used;
  }
  auto planned = PlannedMtp(shaped->first);
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
  const bool capture = runs.CaptureDue(runs_.graphs());
  if (capture) {
    RoomForGraph(kMaxGraphs, graph_stats_, plans_, mplans_);
  }
  kg::LaunchContext& launch = resources_.launch();
  RunPath path = RunPath::kEager;
  Status ran;
  bool unknown = false;
  auto job = [&](providers::NativeStream native) -> sc::JobResult {
    if (auto r = live_.QueueOwed(launch); !r) {
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
  const Status posted = node_.Job(everything_, std::move(job), "a Qwen3.8 MTP draft", stream_);
  if (!posted || !ran || launch.faulted()) {
    // A draft writes the drafter's cells alone (the committed ones as the
    // next draft rewrites them); a launch of unknown effect quarantines.
    Settle(false, false, unknown);
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
                            std::vector<std::int32_t>& argmax, std::vector<float>* logits) {
  if (!speculative()) {
    return Error("a verify needs the drafter");
  }
  if (auto usable = Usable(); !usable) {
    return usable;
  }
  if (auto waiting = live_.AwaitingAccept(); !waiting) {
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
  if (auto used = EnsureState(n_past + rows); !used) {
    return used;
  }
  auto slots = ReadRows(*in);
  if (!slots) {
    return std::unexpected(slots.error());
  }
  const Qwen38ChunkKind kind{.verify = true, .export_streams = true};
  auto planned = Planned({.shape = kg::Qwen38ShapeOf(layout_, *in, rows), .kind = kind});
  if (!planned) {
    return std::unexpected(planned.error());
  }
  ChunkPlans::Entry& entry = **planned;
  Qwen38Planned* p = entry.planned.get();
  const kg::Qwen38Graph& g = p->graph;
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
  const std::uint64_t state = live_.base(kTarget);
  live_.BeginSaves();
  for (std::uint32_t i = 0; i < rows; ++i) {
    const std::uint64_t cell = std::uint64_t{n_past} + i;
    for (const md::Qwen38StateTensor& t : layout_.tensors) {
      if (t.kind != K::kK && t.kind != K::kV && t.kind != K::kIndexerK) {
        continue;
      }
      const std::uint64_t bytes = t.ne0 * (t.f16 ? 2 : 4);
      if (auto added = live_.Save(state + t.offset + (cell * bytes), bytes, i); !added) {
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
      if (auto added = live_.Save(state + t.offset + (b * bytes), bytes, row); !added) {
        return added;
      }
    }
  }
  const std::function<bool(void*)> gather = Gather(rows);
  // The argmaxes always; the logits (their own runs) when asked.
  PlanRuns& runs = entry.runs[logits != nullptr ? kWithLogits : kLean];
  const bool capture = runs.CaptureDue(runs_.graphs());
  if (capture) {
    RoomForGraph(kMaxGraphs, graph_stats_, plans_, mplans_);
  }
  const std::uint64_t row_bytes = std::uint64_t{profile_.vocab} * sizeof(float);
  auto* const argmax_host = static_cast<std::int32_t*>(drafts_) + kArgmaxAt;
  Copies outputs = {{Address(argmax_host), Address(g.argmax->data), rows * sizeof(std::int32_t)}};
  if (logits != nullptr) {
    outputs.push_back({Address(logits_), Address(g.logits->data), rows * row_bytes});
  }
  kg::LaunchContext& launch = resources_.launch();
  RunPath path = RunPath::kEager;
  Status ran;
  bool saved = false;
  bool unknown = false;
  auto job = [&](providers::NativeStream native) -> sc::JobResult {
    if (auto r = live_.QueueOwed(launch); !r) {
      ran = Error(std::format("verify at {}: {}", n_past, r.error().detail));
      unknown = true;
      return sc::JobResult::kUnknown;
    }
    if (auto r = live_.QueueSaves(launch); !r) {
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
  const Status posted = node_.Job(everything_, std::move(job), "a Qwen3.8 verify", stream_);
  if (!posted || !ran || launch.faulted()) {
    // Never left half-written: the verify is undone before the next job's
    // work, or the state quarantined.
    Settle(saved, false, unknown);
    if (launch.faulted()) {
      return Error(std::format("verify at {}: the launch context faulted", n_past));
    }
    return !ran ? ran : Error(std::format("verify at {}: {}", n_past, posted.error()));
  }
  Count(graph_stats_, path);
  live_.Verified(rows);
  argmax.assign(argmax_host, argmax_host + rows);
  if (logits != nullptr) {
    const auto* values = static_cast<const float*>(logits_);
    logits->assign(values, values + (std::size_t{rows} * profile_.vocab));
  }
  return {};
}

Status Qwen38Runner::Accept(std::uint32_t keep) {
  if (auto usable = Usable(); !usable) {
    return usable;
  }
  if (auto accepted = live_.Accept(keep); !accepted) {
    return accepted;
  }
  pending_rows_ = keep;
  return {};
}

Status Qwen38Runner::Rollback() {
  if (auto usable = Usable(); !usable) {
    return usable;
  }
  return live_.Rollback(node_, everything_, stream_, resources_.launch(),
                        "committing a verify's kept rows");
}

Status Qwen38Runner::ReadState(std::vector<std::byte>& target, std::vector<std::byte>& drafter) {
  if (auto r = Rollback(); !r) {
    return r;
  }
  if (live_.verify_rows() != 0) {
    return Error("reading the state with a verify awaiting its Accept");
  }
  const std::array<std::vector<std::byte>*, 2> out = {&target, &drafter};
  return live_.Read(node_, fence_, stream_, "reading the Qwen3.8 state", out);
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
  live_.Release(memory, problems);
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
  return live_.Copy(node_, fence_, stream_, host, ranges, true);
}

Status Qwen38Runner::RestoreUsedState(void* host, std::span<const LiveState::Range> ranges) {
  if (host == nullptr &&
      std::ranges::any_of(ranges, [](const LiveState::Range& r) { return r.bytes != 0; })) {
    return Error("the conversation snapshot has no source buffer");
  }
  if (auto prepared = PrepareRestoreState(ranges); !prepared) {
    return prepared;
  }
  return CopyCheckpointState(host, ranges, false);
}

Status Qwen38Runner::PrepareRestoreState(std::span<const LiveState::Range> ranges) {
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

Status Qwen38Runner::CopyCheckpointState(void* host, std::span<const LiveState::Range> ranges,
                                         bool to_host) {
  auto copied = live_.Copy(node_, fence_, stream_, host, ranges, to_host);
  if (!copied) {
    live_.Quarantine();
  }
  return copied;
}

std::expected<std::vector<LiveState::Range>, std::string> Qwen38Runner::CheckpointRanges(
    std::uint32_t positions) const {
  if (auto usable = live_.Usable(); !usable) {
    return std::unexpected(usable.error());
  }
  if (auto settled = live_.AwaitingAccept(); !settled) {
    return std::unexpected(settled.error());
  }
  if (live_.owed()) {
    return Error("checkpoint has an unsettled Qwen3.8 verify");
  }
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
  return CheckpointPages(live_.used_ranges(), writes);
}

}  // namespace jitllm::engine
