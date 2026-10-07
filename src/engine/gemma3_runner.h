// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Native Gemma3 runner over the shared paged engine.
#ifndef JITLLM_ENGINE_GEMMA3_RUNNER_H_
#define JITLLM_ENGINE_GEMMA3_RUNNER_H_

#include <array>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "engine/gemma3_plan.h"
#include "engine/graph_runs.h"
#include "engine/live_state.h"
#include "engine/paged_weights.h"
#include "engine/request_cohort.h"
#include "engine/runner_resources.h"

namespace jitllm::engine {
std::expected<void, std::string> Gemma3CheckpointFootprint(
    const model::Gemma3Profile& profile, const model::Gemma3StateLayout& layout,
    std::uint32_t positions, std::span<const LiveState::Range> ranges);
struct Gemma3Options {
  std::filesystem::path artifact = {}, out = {};
  std::uint32_t context = 4096, max_rows = 128, slots = 1;
  // Zero retains max_rows total. Each slot still accepts at most max_rows.
  std::uint32_t max_wave_rows = 0;
  // Immutable head publication capacity. Zero retains the all-row envelope.
  std::uint32_t max_head_rows = 0;
  bool graphs = true, frontier_head = true;
  bool owner_decode = false, packed_prefill = false;
  // Opt-in copy-free C2 attention; state layout and publication are unchanged.
  bool bounded_roots = false;
  // Internal whole-C12 factor; retain cohort-wide masks/grid with actual roots.
  bool bounded_whole12 = false;
  // Explicit graph-owned causal/ring mask policy; host reference is the default.
  bool device_masks = false;
  // Explicit numerical comparisons only; ordinary plans use primitives.
  bool fuse_norms = false, fuse_quant_glu = false, fuse_norm_rope = false, fuse_norm_add = false;
  // A prefill chunk's hints of its next two chunks (WavePrefill's next): an
  // upcoming shape's descriptors built on the host beside this chunk's
  // device run, and an upcoming shape's graph captured before its first
  // run (or, without that, on its first run when the next chunk repeats it).
  bool prefill_lookahead = true, capture_ahead = true;
  std::function<LiveState::SpillPlace(std::uint32_t)> spill_place = {};
};
class Gemma3Runner final : public PagedModel {
 public:
  using Plans = PlanCache<kernels::ggml::Gemma3ChunkShape, Gemma3Planned>;
  struct Slot {
    explicit Slot(std::uint32_t id) : index(id) {}
    const std::uint32_t index;
    std::uint32_t completed_positions() const { return positions; }
    std::uint64_t used_state_bytes() const { return live.used_bytes(); }
    bool state_usable() const { return !live.quarantined() && !restoring; }
    bool kept_whole() const {
      return on_disk && adopted.empty() && state_usable() && positions != 0;
    }
    std::uint64_t spilled_bytes() const { return spilled ? adopted_bytes + live.used_bytes() : 0; }
    std::uint64_t refused_bytes() const { return live.refused_bytes(); }
    bool is_spilled() const { return spilled; }
    bool refused_state_growth() const { return state_refused; }
    const LiveState& state() const { return live; }

   private:
    friend class Gemma3Runner;
    LiveState live{"Gemma3"};
    catalog::Closure fence;
    std::uint32_t positions = 0;
    bool provisioned = false, spilled = false, state_refused = false, on_disk = false;
    std::vector<LiveState::Range> adopted;
    std::uint64_t adopted_bytes = 0;
    std::optional<std::uint32_t> restoring;
    std::vector<LiveState::Range> restore_needed;
    std::vector<std::uint64_t> restored_bytes;
  };
  struct Work {
    std::uint32_t slot = 0, n_past = 0;
    std::span<const std::int32_t> tokens;
    std::vector<float>* logits = nullptr;
    // Exactly one output: a full row, or one device-chosen frontier token.
    std::int32_t* token = nullptr;
  };
  // Cumulative selections in successfully bound runtime plans, including
  // plans later reclaimed. Setup's envelope probes are not counted.
  struct PlanSelections {
    std::uint64_t plans = 0, steps = 0, norm_mul = 0, quant_geglu = 0, norm_rope = 0, norm_add = 0,
                  owner_attention = 0, packed_prefill_attention = 0, bounded_owner_attention = 0,
                  device_masks = 0;
  };
  Gemma3Runner(PagedNode& node, Gemma3Options options, int owner, std::uint32_t stream);
  ~Gemma3Runner() override;
  Gemma3Runner(const Gemma3Runner&) = delete;
  Gemma3Runner& operator=(const Gemma3Runner&) = delete;
  void SetSpillPlaces(const std::function<LiveState::SpillPlace(std::uint32_t)>& place) {
    o_.spill_place = place;
  }
  Status Setup();
  Status Register();
  Status Bind();
  Status Release() override;
  std::uint32_t stream() const override { return stream_; }
  const catalog::Closure& fence_closure() const override { return fence_; }
  const catalog::Closure& closure() const { return execution_; }
  const catalog::Closure& everything() const { return everything_; }
  std::vector<catalog::ExtentId> managed_extents() const override;
  std::vector<catalog::ExtentId> weights() const;
  std::vector<catalog::ExtentId> state() const;
  std::vector<catalog::ExtentId> kept_state() const;
  std::expected<Slot*, std::string> request_slot(std::uint32_t slot);
  Status SelectSlots(std::span<const std::uint32_t> slots);
  Status Clear(std::uint32_t slot = 0);
  Status ClearIdle(std::uint32_t slot);
  Status Spill(std::uint32_t slot);
  Status Restore(std::uint32_t slot);
  Status CheckPlaces();
  Status ReserveStateThrough(std::uint32_t slot, std::uint32_t positions);
  std::expected<std::vector<LiveState::Range>, std::string> CheckpointRanges(
      std::uint32_t positions) const;
  // Read-only diagnostic copy; caller funds and retains node-owned pinned
  // output until retirement. An unproven failure requires keeping the buffer.
  Status CopyState(std::uint32_t slot, void* pinned, std::span<const LiveState::Range> ranges,
                   LiveState::CopyRetirement* retirement = nullptr);
  Status CopyState(std::uint32_t slot, void* pinned, std::span<const LiveState::Range> ranges,
                   bool to_host, LiveState::CopyRetirement* retirement = nullptr);
  const std::string& CheckpointLayoutId() const { return checkpoint_layout_id_; }
  bool cohort_usable() const { return !cohort_.faulted(); }
  bool Held(std::uint32_t slot) const { return cohort_.IsActive(slot) && node_.InRequest(stream_); }
  void StateWrittenBack(bool whole);
  Status ValidateFootprint(std::uint32_t positions, std::span<const LiveState::Range> ranges) const;
  Status PrepareRestore(std::uint32_t slot, std::uint32_t positions,
                        std::span<const LiveState::Range> footprint,
                        std::string_view source_layout);
  Status CompleteRestore(std::uint32_t slot, std::uint32_t positions);
  Status Adopt(std::uint32_t slot, std::uint32_t positions,
               std::span<const LiveState::Range> footprint, std::string_view source_layout);
  Status Chunk(std::uint32_t past, std::span<const std::int32_t> tokens, std::vector<float>& logits,
               bool all_outputs = false);
  Status Wave(std::span<const Work> work, bool all_outputs = false);
  // A slot's next chunk's rows and, if known, the rows of the one after it.
  struct PrefillNext {
    std::uint32_t slot = 0, rows = 0, after = 0;
  };
  // A bounded shape prediction only: no future tokens, state or work is posted.
  // Next slots must belong to this wave; their past is its completed end.
  Status WavePrefill(std::span<const Work> work, bool want_head = true,
                     std::span<const PrefillNext> next = {}, bool next_want_head = true,
                     bool after_want_head = true);
  struct LookaheadStats {
    std::uint64_t attempted = 0, built = 0, cached = 0, refused = 0;
    std::uint64_t captured_first = 0;  // shapes captured on their first run
    std::uint64_t captured_ahead = 0;  // graphs captured before their plan's first run
    std::uint64_t dropped_ahead = 0;   // of those, dropped as their staging differed
    double build_seconds = 0;
  };
  const LookaheadStats& lookahead_stats() const { return lookahead_; }
  std::uint64_t weight_read_bytes() const { return weights_.read_bytes(); }
  std::uint64_t graph_measured_bytes() const { return plans_.graph_measured_bytes(); }
  std::uint64_t graph_count() const { return plans_.graphs(); }
  std::uint64_t activations_needed() const { return activation_bytes_; }
  std::uint64_t pool_needed() const { return scratch_bytes_; }
  std::uint64_t host_input_bytes() const { return host_input_bytes_; }
  std::uint64_t plan_floor_bytes() const { return plan_floor_bytes_; }
  std::uint64_t plans_bytes() const { return account_.bytes(); }
  const model::Gemma3Profile& profile() const { return profile_; }
  const model::Gemma3StateLayout& layout() const { return layout_; }
  const GraphStats& graph_stats() const { return graph_stats_; }
  const PlanSelections& plan_selections() const { return plan_selections_; }
  const Coverage& coverage() const { return coverage_; }
  std::uint64_t greedy_tokens() const { return greedy_tokens_; }
  void DropPlans();
  void ReclaimCandidates(std::uint32_t owner, bool running,
                         std::vector<memory::ReclaimCandidate>& out);
  std::uint64_t Reclaim(memory::ReclaimKind kind, std::uint64_t id);

 private:
  Status WaveWithMode(std::span<const Work> work, bool all_outputs,
                      kernels::ggml::Gemma3OutputMode mode, std::span<const PrefillNext> next = {},
                      bool next_want_head = true, bool after_want_head = true);
  Status RefreshClosures(SlotMask protect);
  Status RefreshClosures() { return RefreshClosures(cohort_.active()); }
  std::array<LiveState*, kMaxRequestSlots> States();
  Status CheckActive(const Slot& slot) const;
  kernels::ggml::DeviceChoices Choices(kernels::ggml::LaunchContext& launch) const;
  std::expected<Plans::Entry*, std::string> Planned(const kernels::ggml::Gemma3ChunkShape& shape);
  // Binds, checks and caches built descriptors; `transfer_charge` runs just
  // before the cache's own charge (a lookahead's temporary host allowance).
  std::expected<Plans::Entry*, std::string> CachePlanned(
      const kernels::ggml::Gemma3ChunkShape& shape, std::unique_ptr<Gemma3Planned> p,
      double seconds, const std::function<void()>& transfer_charge = {});
  PagedNode& node_;
  Gemma3Options o_;
  int owner_;
  std::uint32_t stream_;
  model::Gemma3Profile profile_ = model::Gemma3_4BQat();
  model::Gemma3Binding binding_;
  std::string checkpoint_layout_id_;
  model::Gemma3StateLayout layout_;
  RunnerResources resources_;
  PagedWeights weights_;
  std::array<std::unique_ptr<Slot>, kMaxRequestSlots> slots_;
  Gemma3Model model_;
  RequestCohort cohort_{"Gemma3"};
  catalog::Closure everything_, fence_, execution_;
  PlanAccount account_;
  Plans plans_;
  GraphRuns runs_;
  GraphStats graph_stats_;
  PlanSelections plan_selections_;
  LookaheadStats lookahead_;
  Coverage coverage_;
  std::optional<std::uint64_t> places_clean_;
  void* logits_ = nullptr;
  std::uint64_t activation_bytes_ = 0, scratch_bytes_ = 0, host_input_bytes_ = 0;
  std::uint64_t plan_floor_bytes_ = 0, greedy_tokens_ = 0;
  bool setup_started_ = false, setup_ = false, registered_ = false, bound_ = false,
       released_ = false;
};
}  // namespace jitllm::engine
#endif  // JITLLM_ENGINE_GEMMA3_RUNNER_H_
