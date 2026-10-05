// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Native Gemma engine runner. Serving and model qualification are separate.
#ifndef JITLLM_ENGINE_GEMMA4_RUNNER_H_
#define JITLLM_ENGINE_GEMMA4_RUNNER_H_

#include <array>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "engine/gemma4_plan.h"
#include "engine/graph_runs.h"
#include "engine/live_state.h"
#include "engine/paged_weights.h"
#include "engine/request_cohort.h"
#include "engine/runner_resources.h"

namespace jitllm::engine {
// Slab pitch preserves both 256-byte region alignment and GGML blocks.
std::expected<std::uint64_t, std::string> Gemma4ExpertPitch(
    std::uint64_t minimum, std::span<const std::string_view> types);
struct Gemma4Options {
  std::filesystem::path artifact = {}, out = {};
  std::uint32_t context = 4096, max_rows = 128, slots = 1;
  bool graphs = true, frontier_head = true;
  // Caller-funded host masks remain an explicit numerical diagnostic.
  bool reference_masks = false;
  // Unqualified experiments default off; retain ordinary numerical control.
  bool shared_q8 = false, fuse_norms = false, row_invariant = false;
};
class Gemma4Runner final : public PagedModel {
 public:
  using Plans = PlanCache<kernels::ggml::Gemma4ChunkShape, Gemma4Planned>;
  struct Slot {
    explicit Slot(std::uint32_t id) : index(id) {}
    Slot(const Slot&) = delete;
    Slot& operator=(const Slot&) = delete;
    const std::uint32_t index;
    std::uint32_t completed_positions() const { return positions; }
    bool refused_state_growth() const { return state_refused; }
    std::uint64_t used_state_bytes() const { return live.used_bytes(); }
    bool state_usable() const { return !live.quarantined(); }

   private:
    friend class Gemma4Runner;
    LiveState live{"Gemma4"};
    catalog::Closure fence;
    bool provisioned = false, spilled = false, state_refused = false;
    std::uint32_t positions = 0;
  };
  struct Work {
    std::uint32_t slot = 0, n_past = 0;
    std::span<const std::int32_t> tokens;
    // Either this segment's frontier or every row, according to all_outputs.
    std::vector<float>* logits = nullptr;
  };
  Gemma4Runner(PagedNode& node, Gemma4Options options, int owner, std::uint32_t stream)
      : node_(node),
        o_(std::move(options)),
        owner_(owner),
        stream_(stream),
        resources_(node, owner, stream),
        runs_(o_.graphs) {}
  Gemma4Runner(const Gemma4Runner&) = delete;
  Gemma4Runner& operator=(const Gemma4Runner&) = delete;
  Status Setup();
  Status Register();
  Status Bind();
  std::uint64_t activations_needed() const { return activation_bytes_; }
  std::uint64_t pool_needed() const { return scratch_bytes_; }
  std::uint64_t host_input_bytes() const { return host_input_bytes_; }
  std::uint64_t plan_floor_bytes() const { return plan_floor_bytes_; }
  std::uint64_t plans_bytes() const { return account_.bytes(); }
  std::uint64_t weight_bytes() const { return weights_.bytes(); }
  std::uint64_t slab_padding() const { return weights_.slab_padding(); }
  std::uint64_t pitch_padding() const { return pitch_padding_; }
  const model::Gemma4Profile& profile() const { return profile_; }
  const model::Gemma4StateLayout& layout() const { return layout_; }
  std::expected<Slot*, std::string> request_slot(std::uint32_t slot);
  Status SelectSlots(std::span<const std::uint32_t> slots);
  const catalog::Closure& closure() const { return execution_; }
  std::vector<catalog::ExtentId> weights() const { return weights_.extents(); }
  std::vector<catalog::ExtentId> state() const;
  std::uint32_t stream() const override { return stream_; }
  const catalog::Closure& fence_closure() const override { return fence_; }
  std::vector<catalog::ExtentId> managed_extents() const override;
  Status Release() override;
  Status Clear(std::uint32_t slot = 0);
  Status ClearIdle(std::uint32_t slot);
  Status Spill(std::uint32_t slot);
  Status Restore(std::uint32_t slot);
  Status ReserveStateThrough(std::uint32_t slot, std::uint32_t positions);
  Status CopyState(std::uint32_t slot, void* pinned, std::span<const LiveState::Range> ranges,
                   bool to_host, LiveState::CopyRetirement* retirement = nullptr);
  // Restore into an empty slot with the exact CheckpointRanges footprint.
  Status RestoreCheckpoint(std::uint32_t slot, std::uint32_t positions, void* pinned,
                           std::span<const LiveState::Range> ranges,
                           LiveState::CopyRetirement* retirement = nullptr);
  std::expected<std::vector<LiveState::Range>, std::string> CheckpointRanges(
      std::uint32_t positions) const;
  Status Chunk(std::uint32_t n_past, std::span<const std::int32_t> tokens,
               std::vector<float>& logits, bool all_outputs = false);
  Status Wave(std::span<const Work> work, bool all_outputs = false);
  void DropPlans();
  void ReclaimCandidates(std::uint32_t owner, bool running,
                         std::vector<memory::ReclaimCandidate>& out);
  std::uint64_t Reclaim(memory::ReclaimKind kind, std::uint64_t id);
  const GraphStats& graph_stats() const { return graph_stats_; }
  const Coverage& coverage() const { return coverage_; }
  struct PolicyCounts {
    std::uint32_t rows = 0, segments = 0, norm_fused = 0, rope_store = 0;
    std::uint32_t shared_vecq = 0, row_products = 0, lane_steps = 0;
  };
  const PolicyCounts& last_built_policy() const { return policy_; }

 private:
  Status RefreshClosures(SlotMask protect);
  Status RefreshClosures() { return RefreshClosures(cohort_.active()); }
  std::array<LiveState*, kMaxRequestSlots> States();
  Status CheckActive(const Slot& slot) const;
  Status CheckFactors();
  Status CheckPlaces();
  Status ReserveWeights();
  kernels::ggml::DeviceChoices Choices(kernels::ggml::LaunchContext& launch,
                                       std::uint32_t rows) const;
  std::expected<Plans::Entry*, std::string> Planned(const kernels::ggml::Gemma4ChunkShape& shape);
  PagedNode& node_;
  Gemma4Options o_;
  int owner_;
  std::uint32_t stream_;
  model::Gemma4Profile profile_ = model::Gemma4_26BA4B();
  model::Gemma4Binding binding_;
  model::Gemma4StateLayout layout_;
  RunnerResources resources_;
  PagedWeights weights_;
  std::array<std::unique_ptr<Slot>, kMaxRequestSlots> slots_;
  Gemma4Model model_;
  RequestCohort cohort_{"Gemma4"};
  catalog::Closure everything_, fence_, execution_, factor_closure_;
  PlanAccount account_;
  Plans plans_;
  GraphRuns runs_;
  GraphStats graph_stats_;
  Coverage coverage_;
  PolicyCounts policy_;
  void* logits_ = nullptr;
  void* factors_ = nullptr;
  std::uint64_t activation_bytes_ = 0, scratch_bytes_ = 0, host_input_bytes_ = 0;
  std::uint64_t plan_floor_bytes_ = 0;
  std::uint64_t pitch_padding_ = 0;
  bool setup_started_ = false, registered_ = false, setup_ = false, bound_ = false,
       factors_checked_ = false, released_ = false;
};
}  // namespace jitllm::engine
#endif  // JITLLM_ENGINE_GEMMA4_RUNNER_H_
