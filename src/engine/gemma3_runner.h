// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Native Gemma3 runner over the shared paged engine. No serving adapter.
#ifndef JITLLM_ENGINE_GEMMA3_RUNNER_H_
#define JITLLM_ENGINE_GEMMA3_RUNNER_H_

#include <array>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "engine/gemma3_plan.h"
#include "engine/graph_runs.h"
#include "engine/live_state.h"
#include "engine/paged_weights.h"
#include "engine/request_cohort.h"
#include "engine/runner_resources.h"

namespace jitllm::engine {
struct Gemma3Options {
  std::filesystem::path artifact = {}, out = {};
  std::uint32_t context = 4096, max_rows = 128, slots = 1;
  // Immutable head publication capacity. Zero retains the all-row envelope.
  std::uint32_t max_head_rows = 0;
  bool graphs = true, frontier_head = true;
  bool owner_decode = false;
  // Explicit numerical comparisons only; ordinary plans use primitives.
  bool fuse_norms = false, fuse_quant_glu = false, fuse_norm_rope = false, fuse_norm_add = false;
};
class Gemma3Runner final : public PagedModel {
 public:
  using Plans = PlanCache<kernels::ggml::Gemma3ChunkShape, Gemma3Planned>;
  struct Slot {
    explicit Slot(std::uint32_t id) : index(id) {}
    const std::uint32_t index;
    std::uint32_t completed_positions() const { return positions; }
    std::uint64_t used_state_bytes() const { return live.used_bytes(); }
    bool state_usable() const { return !live.quarantined(); }
    bool is_spilled() const { return spilled; }
    bool refused_state_growth() const { return state_refused; }
    const LiveState& state() const { return live; }

   private:
    friend class Gemma3Runner;
    LiveState live{"Gemma3"};
    catalog::Closure fence;
    std::uint32_t positions = 0;
    bool provisioned = false, spilled = false, state_refused = false;
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
                  owner_attention = 0;
  };
  Gemma3Runner(PagedNode& node, Gemma3Options options, int owner, std::uint32_t stream);
  ~Gemma3Runner() override;
  Gemma3Runner(const Gemma3Runner&) = delete;
  Gemma3Runner& operator=(const Gemma3Runner&) = delete;
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
  Status Chunk(std::uint32_t past, std::span<const std::int32_t> tokens, std::vector<float>& logits,
               bool all_outputs = false);
  Status Wave(std::span<const Work> work, bool all_outputs = false);
  Status WavePrefill(std::span<const Work> work, bool want_head = true);
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
                      kernels::ggml::Gemma3OutputMode mode);
  Status RefreshClosures(SlotMask protect);
  Status RefreshClosures() { return RefreshClosures(cohort_.active()); }
  std::array<LiveState*, kMaxRequestSlots> States();
  Status CheckActive(const Slot& slot) const;
  kernels::ggml::DeviceChoices Choices(kernels::ggml::LaunchContext& launch) const;
  std::expected<Plans::Entry*, std::string> Planned(const kernels::ggml::Gemma3ChunkShape& shape);
  PagedNode& node_;
  Gemma3Options o_;
  int owner_;
  std::uint32_t stream_;
  model::Gemma3Profile profile_ = model::Gemma3_4BQat();
  model::Gemma3Binding binding_;
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
