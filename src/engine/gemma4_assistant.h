// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Native Q-only component over scoped frozen target operands, not serving
// speculation. Owned by its target; shares the target stream and launch.
#ifndef LLMP_ENGINE_GEMMA4_ASSISTANT_H_
#define LLMP_ENGINE_GEMMA4_ASSISTANT_H_
#include <array>
#include <expected>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

#include "engine/gemma4_assistant_plan.h"
#include "engine/gemma4_greedy.h"
#include "engine/gemma4_runner.h"

namespace llmp::engine {
// Validate complete, completed output batches before publishing any row.
Status CheckGemma4AssistantOutputs(std::uint32_t owners, std::uint32_t target_width,
                                   std::span<const float> heads, std::span<const float> features);
class Gemma4Assistant {
 public:
  Gemma4Assistant(const Gemma4Assistant&) = delete;
  Gemma4Assistant& operator=(const Gemma4Assistant&) = delete;
  // One query per borrowed owner. first copies the original protected final
  // feature; later calls consume this component's separate recurrent feature.
  // First anchors must match each borrow's pending target anchor. Subsequent
  // canonical anchors are caller-owned proposals, with no sampler here.
  // Output vectors are caller-funded. Publication follows actual completion.
  Status Step(std::span<const Gemma4Runner::FrozenBorrow* const> borrows,
              std::span<const std::int32_t> anchors, bool first,
              std::span<std::vector<float>* const> logits,
              std::span<std::vector<float>* const> features = {});
  // Optional C1 transaction, enabled only by the target's explicit verify
  // envelope. A completed target frontier at past supplies the anchor. The
  // held request owns every draft/verify/accept retirement; all borrows end
  // before Verify. Result publication follows successful acceptance only.
  // Caller must fund/reserve workspace and result capacities before entry.
  Status GreedyUnit(std::uint32_t slot, std::uint32_t past, std::uint32_t depth,
                    std::span<const float> completed_target_head, Gemma4GreedyWorkspace& workspace,
                    Gemma4GreedyResult& result);
  std::uint64_t activations_needed() const { return activation_bytes_; }
  std::uint64_t pool_needed() const { return scratch_bytes_; }
  std::uint64_t host_input_bytes() const { return host_bytes_; }
  std::uint64_t plan_floor_bytes() const { return plan_floor_bytes_; }
  std::uint64_t weight_bytes() const { return weights_.bytes(); }
  const GraphStats& graph_stats() const { return stats_; }
  std::uint64_t graph_count() const { return plans_.graphs(); }

 private:
  friend class Gemma4Runner;
  using Plans = PlanCache<kernels::ggml::Gemma4AssistantShape, Gemma4AssistantPlanned>;
  explicit Gemma4Assistant(Gemma4Runner& target);
  Status Setup(const std::filesystem::path& artifact);
  Status Register();
  Status Bind();
  Status Release();
  void DropPlans() { plans_.Clear(); }
  Status Factors();
  std::vector<catalog::ExtentId> Shared() const;
  std::expected<Plans::Entry*, std::string> Planned(
      const kernels::ggml::Gemma4AssistantShape& shape);
  Gemma4Runner& target_;
  model::Gemma4AssistantProfile profile_;
  model::Gemma4AssistantBinding binding_;
  RunnerResources resources_;
  Mapped recurrent_;
  PagedWeights weights_;
  Gemma4AssistantModel model_;
  PlanAccount account_;
  Plans plans_;
  GraphRuns runs_;
  GraphStats stats_;
  Coverage coverage_;
  std::array<std::uint64_t, kMaxRequestSlots> epoch_{};
  std::array<std::uint32_t, kMaxRequestSlots> prefix_{};
  std::uint64_t activation_bytes_ = 0, scratch_bytes_ = 0, host_bytes_ = 0, plan_floor_bytes_ = 0;
  std::uint64_t feature_stride_ = 0;
  void *logits_ = nullptr, *features_ = nullptr, *factors_ = nullptr;
  bool setup_success_ = false;
  bool registered_ = false, bound_ = false, released_ = false, factors_checked_ = false;
};
}  // namespace llmp::engine
#endif  // LLMP_ENGINE_GEMMA4_ASSISTANT_H_
