// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "qwen38_batch.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <expected>
#include <format>
#include <fstream>
#include <limits>
#include <span>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "base/sha256.h"
#include "engine/support.h"
#include "providers/device_runtime.h"
#include "scheduler/scheduler.h"

namespace llmp::benchmarks::qwen_batch {
namespace {
namespace en = engine;
namespace kg = kernels::ggml;
namespace md = model;
namespace sc = scheduler;
namespace dv = draft_vocab;
using en::support::Address;
using en::support::Error;
using en::support::Pointer;
using en::support::Round;
constexpr std::uint64_t kPage = en::kPagedExtent;
constexpr std::uint64_t kInputs = 8U << 20;
constexpr std::uint32_t kCopies = 2 * dv::ReferencePages::kMaxRanges;

bool SameRanges(std::span<const en::LiveState::Range> a, std::span<const en::LiveState::Range> b) {
  return std::ranges::equal(a, b, [](const auto& x, const auto& y) {
    return x.region == y.region && x.offset == y.offset && x.bytes == y.bytes;
  });
}

std::vector<en::LiveState::Range> Needed(const en::Qwen38Runner& owner, std::uint32_t positions) {
  // Same initialized-range policy as Qwen38Runner::EnsureState. This helper
  // only computes bounded geometry; LiveState::Use remains the allocator.
  const auto& p = md::Qwen38Flash();
  auto target = md::Qwen38UsedState(p, owner.state_layout(), positions);
  if (!target) {
    return {};
  }
  std::vector<en::LiveState::Range> ranges;
  for (const auto& r : *target) {
    ranges.push_back({.region = 0, .offset = r.offset, .bytes = r.bytes});
  }
  const auto& m = owner.mtp_state();
  const std::uint64_t cells = std::min<std::uint64_t>(m.cells, Round(positions, 256));
  const std::uint64_t kv = std::uint64_t{p.head_dim} * p.kv_heads * 2;
  const std::uint64_t idx = std::uint64_t{p.indexer_head_dim} * 4;
  const std::uint64_t blocks = (cells + p.indexer_ratio - 1) / p.indexer_ratio;
  ranges.push_back({.region = 1, .offset = m.k, .bytes = cells * kv});
  ranges.push_back({.region = 1, .offset = m.v, .bytes = cells * kv});
  ranges.push_back({.region = 1, .offset = m.indexer, .bytes = cells * idx});
  ranges.push_back({.region = 1, .offset = m.blocks, .bytes = blocks * p.indexer_head_dim * 2});
  ranges.push_back({.region = 1, .offset = m.hidden, .bytes = m.bytes - m.hidden});
  return ranges;
}

std::vector<dv::PageRange> Pages(std::span<const en::LiveState::Range> ranges) {
  std::vector<dv::PageRange> result;
  for (const auto& r : ranges) {
    result.push_back({.region = r.region, .offset = r.offset, .bytes = r.bytes});
  }
  return result;
}

template <typename T>
bool Exact(const std::vector<T>& a, const std::vector<T>& b) {
  return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0;
}

std::string Sha(std::span<const std::byte> bytes) {
  base::Sha256 sha;
  sha.Update(bytes);
  return base::ToHex(sha.Finish());
}

template <typename T>
en::Status Write(const std::filesystem::path& path, const std::vector<T>& values) {
  std::ofstream file(path, std::ios::binary);
  file.write(reinterpret_cast<const char*>(values.data()),
             static_cast<std::streamsize>(values.size() * sizeof(T)));
  file.close();
  return file ? en::Status{} : Error("C2 diagnostic output flush failed");
}
}  // namespace

Proof::Status Proof::Fail(std::string detail) {
  failed_ = true;
  for (auto& slot : slots_) {
    slot.state.Quarantine();
  }
  // A failure may follow a copy of unknown effect. Node owns all these
  // buffers through process exit rather than a local vector/destructor.
  for (void* p : {static_cast<void*>(live_page_), static_cast<void*>(expected_page_),
                  static_cast<void*>(output_), static_cast<void*>(copies_),
                  static_cast<void*>(runs_.staging())}) {
    if (p != nullptr) {
      owner_.node_.KeepPinned(p);
    }
  }
  return Error(std::move(detail));
}

Requests<en::Qwen38Model> Proof::Models(bool actual) const {
  Requests<en::Qwen38Model> models = {owner_.model_, owner_.model_, owner_.model_, owner_.model_};
  for (std::size_t s = 0; s < requests_; ++s) {
    if (actual) {
      models[s].places.state = slots_[s].state.base(0);
      models[s].places.mtp_state = slots_[s].state.base(1);
      models[s].places.commit = slots_[s].commit.base;
    }
  }
  return models;
}

Requests<std::vector<std::int32_t>> Proof::Histories() const {
  Requests<std::vector<std::int32_t>> histories;
  for (std::size_t s = 0; s < requests_; ++s) {
    histories[s] = slots_[s].initial;
  }
  return histories;
}

Proof::Status Proof::Setup() {
  if ((requests_ != 2 && requests_ != 4) || (requests_ == 4 && !natural_) ||
      owner_.o_.context != kContext || owner_.o_.max_rows != 4096 || owner_.o_.draft_rows != 3 ||
      owner_.o_.draft_vocab != 47172 || !owner_.dbinding_.selected_head() ||
      owner_.dbinding_.draft_ids.ne[1] != 47172 || !owner_.o_.draft_head_capture) {
    return Error("C2 proof requires context33792/chunk4096/fixed3/selected47172/full capture");
  }
  const auto needed = Needed(owner_, End());
  if (needed.empty()) {
    return Error("C2 initialized-state geometry is unavailable");
  }
  // Every physical page Use may materialize, once per region. Checkpoints
  // retain *all* initialized bytes including padding, not just model tensors.
  std::array<std::vector<std::uint64_t>, 2> page_offsets;
  for (const auto& r : needed) {
    const auto capacity = r.region == 0 ? owner_.layout_.bytes : owner_.mtp_layout_.bytes;
    if (r.bytes == 0 || r.offset > capacity || r.bytes > capacity - r.offset) {
      return Error("C2 required state range is outside its virtual region");
    }
    for (std::uint64_t at = r.offset / kPage * kPage; at < r.offset + r.bytes; at += kPage) {
      page_offsets[r.region].push_back(at);
    }
  }
  std::vector<en::LiveState::Range> physical;
  for (std::size_t region = 0; region < 2; ++region) {
    auto& offsets = page_offsets[region];
    std::ranges::sort(offsets);
    offsets.erase(std::ranges::unique(offsets).begin(), offsets.end());
    const auto capacity = region == 0 ? owner_.layout_.bytes : owner_.mtp_layout_.bytes;
    for (const auto offset : offsets) {
      const std::uint64_t bytes = std::min(kPage, capacity - offset);
      physical.push_back({.region = region, .offset = offset, .bytes = bytes});
      checkpoint_bytes_ += bytes;
    }
  }
  if (physical.size() > dv::ReferencePages::kMaxRanges || checkpoint_bytes_ == 0 ||
      checkpoint_bytes_ > dv::ReferencePages::kMaxBytes) {
    return Error("C2 complete state checkpoint exceeds the bounded reference contract");
  }
  for (std::size_t s = 0; s < requests_; ++s) {
    auto& slot = slots_[s];
    slot.ranges = physical;
    for (const auto& [name, bytes] :
         {std::pair{"target", owner_.layout_.bytes}, std::pair{"mtp", owner_.mtp_layout_.bytes}}) {
      if (auto r = slot.state.AddGrowing(owner_.node_, std::format("C2 slot{} {}", s, name), bytes,
                                         owner_.owner_);
          !r) {
        return r;
      }
    }
    if (auto r = owner_.resources_.Map(slot.commit, std::format("C2 slot{} commit", s),
                                       owner_.commit_.bytes, catalog::MemoryClass::kRuntime);
        !r) {
      return r;
    }
    const auto saved_offset = Round(owner_.commit_layout_.bytes, 256);
    slot.state.SnapshotAt(slot.commit.base + saved_offset, slot.commit.bytes - saved_offset);
    if (auto r = slot.state.AllocateSnapshot(owner_.resources_, 512); !r) {
      return r;
    }
    if (auto r = owner_.resources_.Map(slot.baseline, std::format("C2 slot{} baseline", s),
                                       checkpoint_bytes_, catalog::MemoryClass::kRuntime);
        !r) {
      return r;
    }
  }
  output_bytes_ = requests_ * ((4 * std::uint64_t{owner_.profile_.vocab} * 4) +
                               (3 * (std::uint64_t{owner_.profile_.width} + 47172) * 4) + 256);
  auto inputs = owner_.resources_.Pinned(kInputs);
  auto output = owner_.resources_.Pinned(output_bytes_);
  auto live = owner_.resources_.Pinned(kPage);
  auto expected = owner_.resources_.Pinned(kPage);
  auto copies = owner_.resources_.Pinned(kCopies * sizeof(kg::RangeCopy));
  if (!inputs || !output || !live || !expected || !copies) {
    return Error("C2 paid pinned staging allocation failed");
  }
  runs_.SetStaging(*inputs, kInputs);
  output_ = static_cast<std::byte*>(*output);
  live_page_ = static_cast<std::byte*>(*live);
  expected_page_ = static_cast<std::byte*>(*expected);
  copies_ = static_cast<kg::RangeCopy*>(*copies);
  auto measure = owner_.resources_.MeasuringContext();
  if (!measure) {
    return Error(measure.error());
  }
  const auto choices = kg::DeviceChoicesOf(**measure);
  std::uint64_t single_allocations = 0;
  std::uint64_t packed_allocations = 0;
  const auto account = [&](const Plan& p) -> Status {
    auto scratch = kg::PlanScratch(**measure, p.plan);
    if (!scratch) {
      return Error(scratch.error().detail);
    }
    if (p.inputs_bytes > kInputs) {
      return Error("C2 input staging is too small for a measured pair");
    }
    activations_ = std::max(activations_, p.placement.extent);
    pool_ = std::max(pool_, *scratch);
    ++workspace_probes_;
    if (requests_ == 4) {
      const auto add = [](std::uint64_t& sum, std::uint64_t bytes) -> bool {
        if (bytes > std::numeric_limits<std::uint64_t>::max() - 255) {
          return false;
        }
        const auto rounded = Round(bytes, 256);
        if (rounded > std::numeric_limits<std::uint64_t>::max() - sum) {
          return false;
        }
        sum += rounded;
        return true;
      };
      for (std::size_t s = 0; s < requests_; ++s) {
        const en::PlannedBase* original = nullptr;
        if (p.target[s]) {
          original = p.target[s].get();
        } else if (p.mtp[s]) {
          original = p.mtp[s].get();
        }
        if (original == nullptr) {
          continue;
        }
        std::uint64_t bytes = 0;
        for (const auto& [tensor, offset] : original->placement.offsets) {
          (void)offset;
          if (!add(bytes, ggml_nbytes(tensor))) {
            return Error("C4 activation sum overflow");
          }
        }
        single_allocations = std::max(single_allocations, bytes);
      }
      std::uint64_t packed = 0;
      for (const auto& pair : p.products) {
        const auto* both = pair[2];
        const bool mx = kg::LlmpOpOf(both) == kg::LlmpOp::kMxfp8MulMatVec;
        if (!add(packed, ggml_nbytes(both)) || !add(packed, ggml_nbytes(both->src[mx ? 2 : 1])) ||
            (!mx && !add(packed, ggml_nbytes(both->src[2])))) {
          return Error("C4 paid pack/replacement allocation sum overflow");
        }
      }
      packed_allocations = std::max(packed_allocations, packed);
    }
    return {};
  };
  md::Qwen38PleHash stand_in;
  stand_in.multipliers.assign(owner_.profile_.ngram, 1);
  stand_in.offsets.assign(owner_.profile_.ple_heads(), 0);
  stand_in.vocab.assign(owner_.profile_.ple_heads(), 1);
  stand_in.table_rows = 1;
  // Keep the original C2 enumeration. C4 probes all masks at full, short
  // and unequal boundaries, with a conservative no-reuse storage envelope.
  // Every later plan still checks its actual placement/input/scratch against
  // mapped capacity before queue: a missed shape refuses, never overflows.
  std::vector<Requests<std::uint32_t>> probes;
  if (requests_ == 4) {
    constexpr std::array<Requests<std::uint32_t>, 4> patterns = {
        {{1, 1, 1, 1}, {4, 4, 4, 4}, {4, 3, 2, 1}, {3, 3, 1, 2}}};
    for (std::uint32_t active = 1; active < 16; ++active) {
      for (auto widths : patterns) {
        for (std::size_t s = 0; s < requests_; ++s) {
          if ((active & (1U << s)) == 0) {
            widths[s] = 0;
          }
        }
        if (std::ranges::find(probes, widths) == probes.end()) {
          probes.push_back(widths);
        }
      }
    }
  } else {
    for (std::uint32_t tuple = 1; tuple < 25; ++tuple) {
      probes.push_back({tuple % 5, tuple / 5, 0, 0});
    }
  }
  for (bool batch : {false, true}) {
    for (const auto& widths : probes) {
      std::uint32_t active = 0;
      for (std::size_t s = 0; s < requests_; ++s) {
        if (widths[s] != 0) {
          active |= 1U << s;
        }
      }
      if (!natural_ && (active != 3 || widths[1] != 4)) {
        continue;
      }
      workspace_masks_ |= 1U << active;
      for (std::uint32_t first : {8191U, 8192U, End() - 4}) {
        if (!natural_ && first == 8192) {
          continue;
        }
        for (bool diagnostic : {false, true}) {
          if (!natural_ && !diagnostic) {
            continue;
          }
          Requests<kg::Qwen38ChunkShape> target_shapes{};
          Requests<kg::Qwen38MtpShape> draft_shapes{};
          for (std::size_t s = 0; s < requests_; ++s) {
            if ((active & (1U << s)) == 0) {
              continue;
            }
            const std::uint32_t count = widths[s];
            std::vector<std::int32_t> history(std::size_t{first} + count, 1000);
            auto in = md::Qwen38Chunk(owner_.profile_, owner_.layout_, stand_in, history, first,
                                      count, false);
            auto draft = owner_.MtpInputs(first - count, count, 3, true, 1, !natural_, diagnostic);
            if (!in || !draft) {
              return Error("C2 workspace probe has an invalid bounded shape");
            }
            target_shapes[s] = kg::Qwen38ShapeOf(owner_.layout_, *in, count);
            draft_shapes[s] = draft->first;
          }
          auto target = TargetPlan(Models(false), target_shapes, choices, batch, 0, 0, active);
          auto draft = DraftPlan(Models(false), draft_shapes, choices, batch, 0, 0, active);
          if (!target || !draft) {
            return Error(!target ? target.error() : draft.error());
          }
          if (auto r = account(**target); !r) {
            return r;
          }
          if (auto r = account(**draft); !r) {
            return r;
          }
        }
      }
    }
  }
  if (requests_ == 4) {
    if (single_allocations >
        (std::numeric_limits<std::uint64_t>::max() - packed_allocations) / requests_) {
      return Error("C4 conservative activation envelope overflow");
    }
    conservative_activation_bytes_ = (requests_ * single_allocations) + packed_allocations;
    activations_ = std::max(activations_, conservative_activation_bytes_);
  }
  activations_ = Round(activations_ + (activations_ / 4), kPage);
  pool_ = Round(pool_ + (1U << 20), kPage);
  // owner binds its single shared LaunchContext against this same pool.
  owner_.scratch_bytes_ = std::max(owner_.scratch_bytes_, pool_);
  return {};
}

Proof::Status Proof::Register() {
  for (std::size_t s = 0; s < requests_; ++s) {
    if (auto r = slots_[s].state.RegisterSpill(owner_.node_, owner_.o_.out); !r) {
      return r;
    }
  }
  return {};
}

Proof::Status Proof::Refresh() {
  if (auto refreshed = owner_.RefreshClosures(); !refreshed) {
    return refreshed;
  }
  std::vector<catalog::ExtentId> extents;
  for (const auto& [extent, generation] : owner_.everything_.extents) {
    (void)generation;
    extents.push_back(extent);
  }
  for (const auto& slot : slots_) {
    const auto own = slot.state.extents();
    extents.insert(extents.end(), own.begin(), own.end());
  }
  auto r = owner_.node_.Call(
      [&]() -> Status {
        auto made = owner_.node_.catalog().ClosureOfExtents(extents);
        if (!made) {
          return Error("C2 closure construction failed");
        }
        closure_ = std::move(*made);
        return {};
      },
      "refreshing C2 request closure");
  if (!r) {
    return r;
  }
  return owner_.node_.RefreshRequest(stream(), closure_);
}

Proof::Status Proof::Bind() {
  const auto& p = owner_.profile_;
  using K = md::Qwen38StateTensor::Kind;
  for (auto& slot : std::span(slots_).first(requests_)) {
    slot.commit_args = owner_.commit_args_;
    const auto at = [&](std::uint32_t layer, K kind) {
      return slot.state.base(0) +
             owner_.layout_.tensors[static_cast<std::size_t>(owner_.layout_.Find(layer, kind))]
                 .offset;
    };
    for (std::size_t i = 0; i < owner_.commit_layout_.layers.size(); ++i) {
      auto& into = slot.commit_args.layer[i];
      const auto layer = owner_.commit_layout_.layers[i];
      into = {
          .state = static_cast<float*>(Pointer(at(layer, K::kRecurrent))),
          .history = static_cast<float*>(Pointer(at(layer, K::kConv))),
          .conv = static_cast<const float*>(
              Pointer(slot.commit.base + owner_.commit_layout_.conv_out(i))),
          .qkv =
              static_cast<const float*>(Pointer(slot.commit.base + owner_.commit_layout_.qkv(i))),
          .gate =
              static_cast<const float*>(Pointer(slot.commit.base + owner_.commit_layout_.gate(i))),
          .beta =
              static_cast<const float*>(Pointer(slot.commit.base + owner_.commit_layout_.beta(i)))};
    }
    slot.commit_args.ple_history = static_cast<float*>(Pointer(at(p.ple_layer, K::kPleConv)));
    slot.commit_args.ple_rows =
        static_cast<const float*>(Pointer(slot.commit.base + owner_.commit_layout_.ple()));
    Slot* const stable = &slot;
    slot.state.SetCommit([stable](kg::LaunchContext& launch, std::uint32_t keep) {
      auto args = stable->commit_args;
      args.keep = static_cast<int>(keep);
      return kg::Qwen38Commit(launch, args);
    });
  }
  runs_.SetLaunch(&owner_.resources_.launch());
  return Refresh();
}

Proof::Status Proof::Transfer(std::size_t s, bool from_owner, bool to_baseline) {
  auto& slot = slots_[s];
  if (slot.ranges.size() > kCopies / 2 ||
      (from_owner && !SameRanges(slot.ranges, owner_.live_.used_ranges()))) {
    return Error("C2 baseline copy does not cover exactly the initialized source pages");
  }
  std::uint32_t count = 0;
  std::uint64_t at = 0;
  for (const auto& r : slot.ranges) {
    const auto state = slot.state.base(r.region) + r.offset;
    const auto baseline = slot.baseline.base + at;
    const auto source = from_owner ? owner_.live_.base(r.region) + r.offset : baseline;
    if (at > checkpoint_bytes_ || r.bytes > checkpoint_bytes_ - at || source % 16 != 0 ||
        state % 16 != 0 || r.bytes % 16 != 0) {
      return Error("C2 baseline descriptor exceeds its paid aligned allocation");
    }
    copies_[count++] = {.from = source, .to = state, .bytes = r.bytes};
    if (to_baseline) {
      copies_[count++] = {.from = source, .to = baseline, .bytes = r.bytes};
    }
    at += r.bytes;
  }
  if (at != checkpoint_bytes_) {
    return Error("C2 baseline omitted initialized bytes");
  }
  auto r = owner_.node_.Job(
      closure_,
      [&](providers::NativeStream) {
        auto copied = kg::CopyRanges(owner_.resources_.launch(), copies_, count);
        return copied ? sc::JobResult::kQueued : sc::JobResult::kUnknown;
      },
      "copying paid C2 baseline pages", stream());
  if (!r) {
    return Fail(r.error());
  }
  copied_checkpoint_bytes_ += at * (to_baseline ? 2 : 1);
  return {};
}

Proof::Status Proof::Initialize(const Requests<std::vector<std::int32_t>>& prompts) {
  for (std::size_t s = 0; s < requests_; ++s) {
    if (prompts[s].size() != 8192) {
      return Error("C2 fixture requires exactly8192 IDs");
    }
    if (auto r = owner_.Clear(); !r) {
      return r;
    }
    // Distinct anchors straddle the 8192/256-cell boundary in the first pair.
    const std::uint32_t n = !natural_ && s == 0 ? 8191 : 8192;
    slots_[s].initial.assign(prompts[s].begin(), prompts[s].begin() + n);
    std::vector<float> logits;
    for (std::uint32_t first = 0; first < n;) {
      const std::uint32_t end = std::min(n, first + 4096);
      if (auto r = owner_.Chunk(std::span(slots_[s].initial).first(end), first, logits, true); !r) {
        return r;
      }
      first = end;
    }
    if (logits.size() != owner_.profile_.vocab ||
        !std::ranges::all_of(logits, [](float v) { return std::isfinite(v); })) {
      return Error("C2 native prefill did not return a finite full head");
    }
    const auto winner = std::ranges::max_element(logits);
    slots_[s].initial.push_back(static_cast<std::int32_t>(winner - logits.begin()));
    if (owner_.pending_rows_ != 1) {
      return Error("C2 prefill pending cursor differs from1");
    }
    if (auto r = owner_.EnsureState(End()); !r) {
      return r;
    }
    auto used = slots_[s].state.Use(owner_.node_, slots_[s].ranges, &owner_.everything_);
    if (!used) {
      return Error(used.error());
    }
    if (!SameRanges(slots_[s].ranges, slots_[s].state.used_ranges())) {
      return Error("C2 initialized slot range order/extent differs from the baseline");
    }
    if (auto r = Refresh(); !r) {
      return r;
    }
    if (auto r = Transfer(s, true, true); !r) {
      return r;
    }
    slots_[s].pending = 1;
  }
  return {};
}

Proof::Status Proof::Reset() {
  if (failed_) {
    return Error("C2 failed state cannot be reset/reused");
  }
  for (std::size_t s = 0; s < requests_; ++s) {
    auto& slot = slots_[s];
    if (slot.state.owed() || slot.state.verify_rows() != 0 || slot.state.quarantined()) {
      return Error("C2 reset attempted before settled retirement");
    }
    if (auto r = Transfer(s, false, false); !r) {
      return r;
    }
    slot.pending = 1;
  }
  return {};
}

Proof::Status Proof::CheckPlan(Plan& p) {
  if (p.inputs_bytes > kInputs || p.placement.extent > owner_.node_.activations().bytes) {
    return Error("Shared-row runtime plan exceeds mapped input/activation capacity");
  }
  std::vector<const ggml_tensor*> state;
  std::vector<const ggml_tensor*> runtime;
  std::vector<const ggml_tensor*> scratch;
  for (std::size_t s = 0; s < requests_; ++s) {
    if (p.target[s]) {
      const auto& g = p.target[s]->graph;
      for (const auto& l : g.layers) {
        for (const ggml_tensor* t : {l.cache_k, l.cache_v, l.cache_idx, l.cache_pool, l.conv_state,
                                     l.recurrent, l.ple_state}) {
          if (t != nullptr) {
            state.push_back(t);
          }
        }
        for (const ggml_tensor* t :
             {l.commit_conv, l.commit_qkv, l.commit_gate, l.commit_beta, l.commit_ple}) {
          if (t != nullptr) {
            runtime.push_back(t);
          }
        }
      }
      state.push_back(g.streams);
      scratch.push_back(g.ple_table);
    } else if (p.mtp[s]) {
      const auto& g = p.mtp[s]->graph;
      for (const ggml_tensor* t :
           {g.layer.cache_k, g.layer.cache_v, g.layer.cache_idx, g.layer.cache_pool, g.streams}) {
        if (t != nullptr) {
          state.push_back(t);
        }
      }
    }
  }
  en::CheckCoverage(owner_.node_, owner_.owner_, p.nodes,
                    {.state = state,
                     .runtime = runtime,
                     .scratch = scratch,
                     .inputs = p.inputs,
                     .fill_reads_nothing = true},
                    coverage_);
  if (coverage_.violations != 0) {
    return Error(coverage_.first_violation);
  }
  return en::BindPlanned(p, owner_.resources_.launch(), owner_.resources_.registry(), "C2 pair");
}

Proof::Status Proof::Saves(std::size_t s, std::uint32_t first, std::uint32_t rows) {
  auto& slot = slots_[s];
  slot.state.BeginSaves();
  using K = md::Qwen38StateTensor::Kind;
  for (std::uint32_t row = 0; row < rows; ++row) {
    for (const auto& t : owner_.layout_.tensors) {
      if (t.kind != K::kK && t.kind != K::kV && t.kind != K::kIndexerK) {
        continue;
      }
      const auto bytes = t.ne0 * (t.f16 ? 2 : 4);
      if (auto r = slot.state.Save(
              slot.state.base(0) + t.offset + ((std::uint64_t{first} + row) * bytes), bytes, row);
          !r) {
        return r;
      }
    }
  }
  const auto ratio = owner_.profile_.indexer_ratio;
  for (std::uint64_t block = first / ratio; block < (std::uint64_t{first} + rows) / ratio;
       ++block) {
    const auto row = static_cast<std::uint32_t>(((block + 1) * ratio) - 1 - first);
    for (const auto& t : owner_.layout_.tensors) {
      if (t.kind != K::kIndexerBlocks) {
        continue;
      }
      const auto bytes = t.ne0 * 2;
      if (auto r = slot.state.Save(slot.state.base(0) + t.offset + (block * bytes), bytes, row);
          !r) {
        return r;
      }
    }
  }
  return {};
}

Proof::Status Proof::Settle() {
  if (failed_) {
    return Error("C2 stream/state was quarantined");
  }
  auto r = owner_.node_.Job(
      closure_,
      [&](providers::NativeStream) {
        for (auto& slot : std::span(slots_).first(requests_)) {
          if (auto queued = slot.state.QueueOwed(owner_.resources_.launch()); !queued) {
            return sc::JobResult::kUnknown;
          }
        }
        return sc::JobResult::kQueued;
      },
      "settling both independent C2 requests", stream());
  return r ? Status{} : Fail(r.error());
}

Proof::Status Proof::Draft(const Requests<std::vector<std::int32_t>>& histories, bool batch,
                           bool graphs, Output& output, std::uint32_t active, bool diagnostic) {
  if (active == 0 || active >= (1U << requests_) || (!natural_ && (active != 3 || !diagnostic))) {
    return Error("C2 draft active/output mode is outside this control");
  }
  Requests<std::vector<md::Qwen38ChunkInputs>> ins;
  DraftKey key{.shapes = {}, .batch = batch, .active = active};
  Requests<std::uint32_t> n{};
  for (std::size_t s = 0; s < requests_; ++s) {
    if ((active & (1U << s)) == 0) {
      continue;
    }
    auto& slot = slots_[s];
    if (auto r = slot.state.AwaitingAccept(); !r) {
      return r;
    }
    if (slot.pending == 0 || slot.pending > 4 || histories[s].size() <= slot.pending ||
        histories[s].size() > End() - 3) {
      return Error("C2 MTP pending/history window exceeds the frozen proof envelope");
    }
    n[s] = static_cast<std::uint32_t>(histories[s].size() - 1);
    auto shaped =
        owner_.MtpInputs(n[s] - slot.pending, slot.pending, 3, true, 1, !natural_, diagnostic);
    if (!shaped) {
      return Error(shaped.error());
    }
    key.shapes[s] = shaped->first;
    ins[s] = std::move(shaped->second);
  }
  auto* found = draft_plans_.Find(key);
  if (found == nullptr) {
    auto plan =
        DraftPlan(Models(true), key.shapes, kg::DeviceChoicesOf(owner_.resources_.launch()), batch,
                  owner_.node_.activations().base, owner_.node_.activations().bytes, active);
    if (!plan) {
      return Error(plan.error());
    }
    if (auto r = CheckPlan(**plan); !r) {
      return r;
    }
    found = &draft_plans_.Add(key, std::move(*plan));
  }
  Plan& plan = *found->planned;
  Requests<en::Qwen38MtpHostInputs> host;
  std::vector<std::pair<ggml_tensor*, const void*>> sources;
  en::Copies outputs;
  std::uint64_t at = 0;
  Requests<std::uint64_t> starts{};
  for (std::size_t s = 0; s < requests_; ++s) {
    if ((active & (1U << s)) == 0) {
      continue;
    }
    const auto& g = plan.mtp[s]->graph;
    if (g.drafts.size() != 3 || g.probabilities.size() != (natural_ ? 0U : 3U) ||
        g.head_inputs.size() != (diagnostic ? 3U : 0U) ||
        g.head_logits.size() != (diagnostic ? 3U : 0U)) {
      return Error("C2 draft lacks its full three-pass diagnostic closure");
    }
    en::Qwen38MtpSources(
        g, ins[s], std::span(histories[s]).subspan(n[s] - slots_[s].pending + 1, slots_[s].pending),
        host[s]);
    sources.insert(sources.end(), host[s].sources.begin(), host[s].sources.end());
    starts[s] = at;
    for (const auto& list : {g.drafts, g.probabilities, g.head_inputs, g.head_logits}) {
      for (const auto* tensor : list) {
        const auto bytes = static_cast<std::uint64_t>(ggml_nbytes(tensor));
        if (at > output_bytes_ || bytes > output_bytes_ - at) {
          return Error("C2 draft output staging overflow");
        }
        outputs.push_back({Address(output_ + at), Address(tensor->data), bytes});
        at += bytes;
      }
    }
  }
  auto copies = runs_.Stage(sources, 0);
  if (!copies) {
    return Error(copies.error());
  }
  runs_.set_graphs(graphs);
  en::Status queued;
  auto r = owner_.node_.Job(
      closure_,
      [&](providers::NativeStream native) {
        for (std::size_t s = 0; s < requests_; ++s) {
          if ((active & (1U << s)) == 0) {
            continue;
          }
          auto& slot = slots_[s];
          if (auto owed = slot.state.QueueOwed(owner_.resources_.launch()); !owed) {
            queued = Error(owed.error().detail);
            return sc::JobResult::kUnknown;
          }
        }
        auto q = runs_.Queue(found->runs[0], *copies, {}, *plan.bound, outputs,
                             found->runs[0].CaptureDue(graphs), graph_stats_, native);
        if (!q.result) {
          queued = Error(q.result.error().detail);
          return sc::JobResult::kUnknown;
        }
        en::Count(graph_stats_, q.path);
        return sc::JobResult::kQueued;
      },
      "C2 independent-state MTP windows", stream());
  if (!r || !queued || owner_.resources_.launch().faulted()) {
    if (!queued) {
      return Fail(queued.error());
    }
    if (!r) {
      return Fail(r.error());
    }
    return Fail("C2 launch fault");
  }
  for (std::size_t s = 0; s < requests_; ++s) {
    if ((active & (1U << s)) == 0) {
      continue;
    }
    at = starts[s];
    const auto take = [&](auto& into, std::uint64_t count) {
      using Value = std::remove_reference_t<decltype(into)>::value_type;
      into.resize(count);
      if (count != 0) {
        std::memcpy(into.data(), output_ + at, count * sizeof(Value));
      }
      at += count * sizeof(Value);
    };
    take(output.ids[s], 3);
    take(output.values[s], natural_ ? 0U : 3U);
    take(output.head_inputs[s], diagnostic ? 3 * std::uint64_t{owner_.profile_.width} : 0);
    take(output.head_logits[s], diagnostic ? 3 * 47172ULL : 0);
    if (!std::ranges::all_of(
            output.ids[s],
            [&](auto id) { return id >= 0 && std::cmp_less(id, owner_.profile_.vocab); }) ||
        !std::ranges::all_of(output.values[s],
                             [](float v) { return std::isfinite(v) && v >= 0 && v <= 1; }) ||
        !std::ranges::all_of(output.head_inputs[s], [](float v) { return std::isfinite(v); }) ||
        !std::ranges::all_of(output.head_logits[s], [](float v) { return std::isfinite(v); })) {
      return Error("C2 draft returned invalid/nonfinite output");
    }
  }
  mxfp8_pairs_ += plan.mxfp8_pairs;
  routed_pairs_ += plan.routed_pairs;
  packed_bytes_ += plan.packed_bytes;
  return {};
}

Proof::Status Proof::Verify(const Requests<std::vector<std::int32_t>>& histories,
                            const Requests<std::uint32_t>& first, bool batch, bool graphs,
                            Output& output, std::uint32_t active, bool diagnostic) {
  if (active == 0 || active >= (1U << requests_) || (!natural_ && (active != 3 || !diagnostic))) {
    return Error("C2 verify active/output mode is outside this control");
  }
  TargetKey key{.shapes = {}, .batch = batch, .active = active, .diagnostic = diagnostic};
  Requests<md::Qwen38ChunkInputs> ins;
  md::Qwen38ChunkInputs aggregate;
  Requests<std::uint32_t> rows{};
  std::uint32_t total_rows = 0;
  for (std::size_t s = 0; s < requests_; ++s) {
    if ((active & (1U << s)) == 0) {
      continue;
    }
    if (auto r = slots_[s].state.AwaitingAccept(); !r) {
      return r;
    }
    if (histories[s].size() <= first[s] || histories[s].size() - first[s] > 4 ||
        histories[s].size() > End()) {
      return Error("C2 verify requires one to four bounded rows per request");
    }
    rows[s] = static_cast<std::uint32_t>(histories[s].size() - first[s]);
    total_rows += rows[s];
    auto in = md::Qwen38Chunk(owner_.profile_, owner_.layout_, owner_.hash_, histories[s], first[s],
                              rows[s], false);
    if (!in) {
      return Error(in.error());
    }
    ins[s] = std::move(*in);
    key.shapes[s] = kg::Qwen38ShapeOf(owner_.layout_, ins[s], rows[s]);
    aggregate.ple_rows.insert(aggregate.ple_rows.end(), ins[s].ple_rows.begin(),
                              ins[s].ple_rows.end());
  }
  auto indices = owner_.ReadRows(aggregate);
  if (!indices) {
    return Error(indices.error());
  }
  if (indices->size() != aggregate.ple_rows.size()) {
    return Error("C2 PLE slot count differs");
  }
  auto* found = target_plans_.Find(key);
  if (found == nullptr) {
    auto plan =
        TargetPlan(Models(true), key.shapes, kg::DeviceChoicesOf(owner_.resources_.launch()), batch,
                   owner_.node_.activations().base, owner_.node_.activations().bytes, active);
    if (!plan) {
      return Error(plan.error());
    }
    if (auto r = CheckPlan(**plan); !r) {
      return r;
    }
    found = &target_plans_.Add(key, std::move(*plan));
  }
  Plan& plan = *found->planned;
  Requests<en::Qwen38HostInputs> host;
  std::vector<std::pair<ggml_tensor*, const void*>> sources;
  en::Copies outputs;
  Requests<std::uint64_t> starts{};
  std::uint64_t at = 0;
  std::size_t ple_at = 0;
  for (std::size_t s = 0; s < requests_; ++s) {
    if ((active & (1U << s)) == 0) {
      continue;
    }
    const auto& g = plan.target[s]->graph;
    if (ins[s].qsa_select && (g.mask != nullptr || g.mask_f32 != nullptr)) {
      auto masked = md::Qwen38Chunk(owner_.profile_, owner_.layout_, owner_.hash_, histories[s],
                                    first[s], rows[s], true);
      if (!masked) {
        return Error(masked.error());
      }
      ins[s] = std::move(*masked);
    }
    en::Qwen38Sources(g, ins[s], rows[s],
                      std::span(*indices).subspan(ple_at, ins[s].ple_rows.size()), host[s], 1);
    ple_at += ins[s].ple_rows.size();
    sources.insert(sources.end(), host[s].sources.begin(), host[s].sources.end());
    starts[s] = at;
    for (const auto* tensor : {g.argmax, g.logits}) {
      if (!diagnostic && tensor == g.logits) {
        continue;
      }
      const auto bytes = static_cast<std::uint64_t>(ggml_nbytes(tensor));
      if (at > output_bytes_ || bytes > output_bytes_ - at) {
        return Error("C2 target output overflow");
      }
      outputs.push_back({Address(output_ + at), Address(tensor->data), bytes});
      at += bytes;
    }
    if (auto r = Saves(s, first[s], rows[s]); !r) {
      return r;
    }
  }
  auto copies = runs_.Stage(sources, 0);
  if (!copies) {
    return Error(copies.error());
  }
  runs_.set_graphs(graphs);
  en::Status queued;
  auto r = owner_.node_.Job(
      closure_,
      [&](providers::NativeStream native) {
        for (std::size_t s = 0; s < requests_; ++s) {
          if ((active & (1U << s)) == 0) {
            continue;
          }
          auto& slot = slots_[s];
          if (auto owed = slot.state.QueueOwed(owner_.resources_.launch()); !owed) {
            queued = Error(owed.error().detail);
            return sc::JobResult::kUnknown;
          }
          if (auto saved = slot.state.QueueSaves(owner_.resources_.launch()); !saved) {
            queued = Error(saved.error().detail);
            return sc::JobResult::kUnknown;
          }
        }
        auto q = runs_.Queue(found->runs[0], *copies, owner_.Gather(total_rows), *plan.bound,
                             outputs, found->runs[0].CaptureDue(graphs), graph_stats_, native);
        if (!q.result) {
          queued = Error(q.result.error().detail);
          return sc::JobResult::kUnknown;
        }
        en::Count(graph_stats_, q.path);
        return sc::JobResult::kQueued;
      },
      "C2 separate-position target windows", stream());
  if (!r || !queued || owner_.resources_.launch().faulted()) {
    if (!queued) {
      return Fail(queued.error());
    }
    if (!r) {
      return Fail(r.error());
    }
    return Fail("C2 launch fault");
  }
  for (std::size_t s = 0; s < requests_; ++s) {
    if ((active & (1U << s)) == 0) {
      continue;
    }
    slots_[s].state.Verified(rows[s]);
    output.ids[s].resize(rows[s]);
    output.values[s].resize(diagnostic ? std::uint64_t{rows[s]} * owner_.profile_.vocab : 0);
    at = starts[s];
    std::memcpy(output.ids[s].data(), output_ + at, rows[s] * sizeof(std::int32_t));
    at += rows[s] * sizeof(std::int32_t);
    if (!std::ranges::all_of(output.ids[s], [&](auto id) {
          return id >= 0 && std::cmp_less(id, owner_.profile_.vocab);
        })) {
      return Error("C2 target returned an out-of-vocabulary ID");
    }
    if (!diagnostic) {
      continue;
    }
    std::memcpy(output.values[s].data(), output_ + at, output.values[s].size() * sizeof(float));
    if (!std::ranges::all_of(output.values[s], [](float v) { return std::isfinite(v); })) {
      return Error("C2 target returned a nonfinite full head");
    }
    for (std::uint32_t row = 0; row < rows[s]; ++row) {
      const auto values =
          std::span(output.values[s])
              .subspan(std::uint64_t{row} * owner_.profile_.vocab, owner_.profile_.vocab);
      const auto winner = std::ranges::max_element(values);
      if (output.ids[s][row] != winner - values.begin()) {
        return Error("C2 target ID differs from its natural full-head argmax");
      }
    }
  }
  mxfp8_pairs_ += plan.mxfp8_pairs;
  routed_pairs_ += plan.routed_pairs;
  packed_bytes_ += plan.packed_bytes;
  return {};
}

dv::ReferencePages::Copy Proof::Reader(std::size_t s) {
  return [this, s](void* into, const dv::PageRange& r) -> Status {
    if (r.region >= 2 || r.bytes == 0 || r.bytes > kPage ||
        !std::ranges::any_of(slots_[s].ranges, [&](const auto& held) {
          return held.region == r.region && held.offset == r.offset && held.bytes == r.bytes;
        })) {
      return Error("C2 page reader range is not an exact initialized page");
    }
    auto copied = owner_.node_.Job(
        closure_,
        [&](providers::NativeStream native) {
          return providers::CopyAsync(native, into,
                                      Pointer(slots_[s].state.base(r.region) + r.offset), r.bytes,
                                      providers::CopyKind::kDeviceToHost)
                         .ok()
                     ? sc::JobResult::kQueued
                     : sc::JobResult::kUnknown;
        },
        "reading complete C2 request state page", stream());
    return copied ? Status{} : Fail(copied.error());
  };
}

Proof::Status Proof::StateProof(std::size_t arm, std::size_t round, std::string_view phase,
                                const std::filesystem::path& out) {
  std::size_t phase_index = 2;
  if (phase == "reset") {
    phase_index = 0;
  } else if (phase == "draft") {
    phase_index = 1;
  }
  const std::size_t index = (round * 3) + phase_index;
  for (std::size_t s = 0; s < 2; ++s) {
    if (!SameRanges(slots_[s].ranges, slots_[s].state.used_ranges())) {
      return Error("C2 state geometry changed/omitted an initialized page");
    }
    const auto ranges = Pages(slots_[s].ranges);
    if (arm == 0) {
      auto captured = dv::ReferencePages::Capture(out, ranges, slots_[s].pending,
                                                  {live_page_, kPage}, Reader(s), reference_stats_);
      if (!captured) {
        return Error(captured.error().detail);
      }
      references_[index][s] = std::move(*captured);
    } else {
      auto same = references_[index][s].Compare(ranges, slots_[s].pending, {expected_page_, kPage},
                                                {live_page_, kPage}, Reader(s), reference_stats_);
      if (!same) {
        return Error(same.error().detail);
      }
    }
    ++state_controls_;
  }
  return {};
}

Proof::Status Proof::Controls(const Requests<std::vector<std::int32_t>>& prompts,
                              const std::filesystem::path& out) {
  if (requests_ != 2 || natural_) {
    return Error("Forced controls require the unchanged C2 mode");
  }
  if (auto r = Initialize(prompts); !r) {
    return r;
  }
  // Lease the union once, including all page-reader steps. There is no state
  // growth inside this bounded proof; neither request can be evicted midway.
  if (auto r = owner_.node_.BeginRequest(stream(), closure_, "C2 shared-weight proof"); !r) {
    return r;
  }
  const auto checked = [&]() -> Status {
    // Independent accepted lengths, then different catch-up widths. Stop tokens
    // are literal fixture IDs here; no answer/task/concurrency-quality claim.
    constexpr std::array<Requests<std::uint32_t>, 2> rows = {{{4, 4, 0, 0}, {3, 2, 0, 0}}};
    constexpr std::array<Requests<std::uint32_t>, 2> keep = {{{1, 4, 0, 0}, {2, 1, 0, 0}}};
    for (std::size_t arm = 0; arm < 8; ++arm) {
      const bool batch = arm % 4 == 1 || arm % 4 == 2;
      const bool graphs = arm >= 4;
      const auto before = graph_stats_;
      if (auto r = Reset(); !r) {
        return r;
      }
      auto histories = Histories();
      for (std::size_t round = 0; round < 2; ++round) {
        if (auto r = StateProof(arm, round, "reset", out); !r) {
          return r;
        }
        Output draft;
        if (auto r = Draft(histories, batch, graphs, draft); !r) {
          return r;
        }
        if (auto r = StateProof(arm, round, "draft", out); !r) {
          return r;
        }
        Requests<std::uint32_t> first{};
        auto verify_history = histories;
        for (std::size_t s = 0; s < 2; ++s) {
          first[s] = static_cast<std::uint32_t>(histories[s].size() - 1);
          verify_history[s].insert(verify_history[s].end(), draft.ids[s].begin(),
                                   draft.ids[s].begin() + rows[round][s] - 1);
        }
        Output target;
        if (auto r = Verify(verify_history, first, batch, graphs, target); !r) {
          return r;
        }
        // Preserve even a later failed comparison as bounded raw outside Git.
        for (std::size_t s = 0; s < 2; ++s) {
          const auto prefix = std::format("arm{}-round{}-slot{}-", arm, round, s);
          for (const auto& [name, values] : {std::pair{"draft-confidence", &draft.values[s]},
                                             std::pair{"draft-input", &draft.head_inputs[s]},
                                             std::pair{"draft-logits", &draft.head_logits[s]},
                                             std::pair{"target-logits", &target.values[s]}}) {
            if (auto r = Write(out / (prefix + name + ".f32"), *values); !r) {
              return r;
            }
          }
          if (auto r = Write(out / (prefix + "draft-ids.i32"), draft.ids[s]); !r) {
            return r;
          }
          if (auto r = Write(out / (prefix + "target-ids.i32"), target.ids[s]); !r) {
            return r;
          }
        }
        for (std::size_t s = 0; s < 2; ++s) {
          if (arm == 0) {
            expected_outputs_[round][0] = draft;
            expected_outputs_[round][1] = target;
          } else {
            const auto& d = expected_outputs_[round][0];
            const auto& t = expected_outputs_[round][1];
            if (!Exact(draft.ids[s], d.ids[s]) || !Exact(draft.values[s], d.values[s]) ||
                !Exact(draft.head_inputs[s], d.head_inputs[s]) ||
                !Exact(draft.head_logits[s], d.head_logits[s]) || !Exact(target.ids[s], t.ids[s]) ||
                !Exact(target.values[s], t.values[s])) {
              return Error(
                  std::format("C2 serial/batch/graph full-output mismatch arm{} round{} slot{}",
                              arm, round, s));
            }
          }
          ++output_controls_;
          if (auto r = slots_[s].state.Accept(keep[round][s]); !r) {
            return r;
          }
          slots_[s].pending = keep[round][s];
          histories[s] = verify_history[s];
          histories[s].resize(std::size_t{first[s]} + keep[round][s]);
          histories[s].push_back(target.ids[s][keep[round][s] - 1]);
        }
        if (auto r = Settle(); !r) {
          return r;
        }
        if (auto r = StateProof(arm, round, "committed", out); !r) {
          return r;
        }
      }
      auto& actual = arm_graph_stats_[arm];
      actual.eager = graph_stats_.eager - before.eager;
      actual.captured = graph_stats_.captured - before.captured;
      actual.replayed = graph_stats_.replayed - before.replayed;
      actual.refused = graph_stats_.refused - before.refused;
      actual.nodes = graph_stats_.nodes - before.nodes;
      actual.capture_seconds = graph_stats_.capture_seconds - before.capture_seconds;
      actual.instantiate_seconds = graph_stats_.instantiate_seconds - before.instantiate_seconds;
      if (actual.eager != (arm < 4 ? 4ULL : 0ULL) ||
          actual.captured != (arm == 4 || arm == 5 ? 4ULL : 0ULL) ||
          actual.replayed != (arm >= 6 ? 4ULL : 0ULL) || actual.refused != 0 ||
          ((arm == 4 || arm == 5) &&
           (actual.nodes == 0 || !std::isfinite(actual.capture_seconds) ||
            actual.capture_seconds <= 0 || !std::isfinite(actual.instantiate_seconds) ||
            actual.instantiate_seconds <= 0))) {
        return Error(std::format("C2 missing actual eager/capture/replay path in arm{}", arm));
      }
    }
    if (graph_stats_.eager != 16 || graph_stats_.captured != 8 || graph_stats_.replayed != 8 ||
        graph_stats_.refused != 0 || graph_stats_.dropped != 0 || coverage_.violations != 0 ||
        output_controls_ != 32 || state_controls_ != 96 || reference_stats_.captures != 12 ||
        reference_stats_.comparisons != 84) {
      return Error("C2 proof missing graph/eager full state/output controls");
    }
    return {};
  }();
  const auto ended = owner_.node_.EndRequest(stream());
  if (!checked || !ended) {
    std::string error = checked ? "" : checked.error();
    if (!ended) {
      error += (error.empty() ? "" : "; ") + ended.error();
    }
    return Error(std::move(error));
  }
  complete_ = true;
  return {};
}

std::string Proof::receipt() const {
  if (natural_) {
    return GenerationReceipt();
  }
  std::string initial;
  for (const auto& slot : std::span(slots_).first(requests_)) {
    initial += std::format("{}\"{}\"", initial.empty() ? "" : ",",
                           Sha(std::as_bytes(std::span(slot.initial))));
  }
  std::string state_sha;
  for (const auto& phase : references_) {
    for (const auto& slot : std::span(phase).first(requests_)) {
      state_sha += std::format("{}\"{}\"", state_sha.empty() ? "" : ",", slot.sha256());
    }
  }
  std::string arms;
  for (std::size_t arm = 0; arm < arm_graph_stats_.size(); ++arm) {
    const auto& paths = arm_graph_stats_[arm];
    arms +=
        std::format(R"({}{{"arm":{},"batch":{},"graphs":{},"eager":{},"captured":{},"replayed":{},)"
                    R"("refused":{},"nodes":{},"capture_seconds":{},"instantiate_seconds":{}}})",
                    arms.empty() ? "" : ",", arm, arm % 4 == 1 || arm % 4 == 2 ? "true" : "false",
                    arm >= 4 ? "true" : "false", paths.eager, paths.captured, paths.replayed,
                    paths.refused, paths.nodes, paths.capture_seconds, paths.instantiate_seconds);
  }
  return std::format(
      R"({{"schema":"llmp-qwen-c2-shared-rows-graphs-v1","complete":{},"requests":2,)"
      R"("fixed_depth":3,"draft_vocab":47172,"timing_claim":false,"shared_weight_owners":1,)"
      R"("live_request_states":2,"prefill_owner_state":1,"baseline_device_bytes":{},)"
      R"("copied_checkpoint_bytes":{},"initial_history_sha256":[{}],"output_controls":{},)"
      R"("state_controls":{},"state_captures":{},"state_comparisons":{},"state_read_bytes":{},)"
      R"("reference_state_sha256":[{}],"state_capture_seconds":{},"state_comparison_seconds":{},)"
      R"("host_control_allowance_bytes":134217728,"graph_replay_qualified":{},"arms":[{}],)"
      R"("graph_nodes":{},"graph_capture_seconds":{},"graph_instantiate_seconds":{},)"
      R"("graph_free_memory_delta_bytes":{},)"
      R"("mxfp8_pairs":{},"routed_launch_pairs":{},"packed_input_bytes":{},)"
      R"("eager":{},"captured":{},"replayed":{},"capture_refused":{},"coverage_violations":{}}})",
      complete_ && !failed_ ? "true" : "false", 2 * checkpoint_bytes_, copied_checkpoint_bytes_,
      initial, output_controls_, state_controls_, reference_stats_.captures,
      reference_stats_.comparisons,
      reference_stats_.capture_bytes + reference_stats_.comparison_bytes, state_sha,
      reference_stats_.capture_seconds, reference_stats_.comparison_seconds,
      complete_ && !failed_ ? "true" : "false", arms, graph_stats_.nodes,
      graph_stats_.capture_seconds, graph_stats_.instantiate_seconds, graph_stats_.memory_bytes,
      mxfp8_pairs_, routed_pairs_, packed_bytes_, graph_stats_.eager, graph_stats_.captured,
      graph_stats_.replayed, graph_stats_.refused, coverage_.violations);
}

std::vector<catalog::ExtentId> Proof::managed_extents() const {
  std::vector<catalog::ExtentId> result;
  for (const auto& slot : slots_) {
    const auto extents = slot.state.extents();
    result.insert(result.end(), extents.begin(), extents.end());
  }
  return result;
}

Proof::Status Proof::Release() {
  if (released_) {
    return {};
  }
  released_ = true;
  target_plans_.Clear();
  draft_plans_.Clear();
  std::vector<std::string> problems;
  for (auto& slot : slots_) {
    slot.state.Release(owner_.node_.memory(), problems);
  }
  // Commit/baseline mappings are stable members tracked by owner's resources;
  // owner.Release follows this call, before this object's members die.
  return en::support::Joined(problems);
}
}  // namespace llmp::benchmarks::qwen_batch
