// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef LLMP_BENCHMARKS_QWEN38_BATCH_H_
#define LLMP_BENCHMARKS_QWEN38_BATCH_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "engine/qwen38_runner.h"
#include "qwen38_batch_graph.h"
#include "qwen38_reference.h"

namespace llmp::benchmarks::qwen_batch {

// Private C2/C4 mechanism proof, no HTTP/default change. One existing
// runner owns weights/PLE/launch; each request owns distinct target+MTP
// state, snapshots, kept-row commits and pending cursors on that same stream.
// The owner and this object both outlive fenced TearDown({this, &owner}).
class Proof final : public engine::PagedModel {
 public:
  using Status = engine::Status;
  static constexpr std::uint32_t kContext = 33792;
  static constexpr std::uint32_t kEnd = 8448;
  static constexpr std::uint32_t kNaturalEnd = 8704;
  explicit Proof(engine::Qwen38Runner& owner, bool natural = false, std::uint32_t requests = 2)
      : owner_(owner), natural_(natural), requests_(requests) {}
  ~Proof() override = default;
  Proof(const Proof&) = delete;
  Proof& operator=(const Proof&) = delete;
  Proof(Proof&&) = delete;
  Proof& operator=(Proof&&) = delete;

  // After owner.Setup; before MapWorkspace/Start. Measures both joint arms
  // and maps paid baseline checkpoints, staging, independent commit buffers.
  Status Setup();
  Status Register();
  Status Bind();
  std::uint64_t activations_needed() const { return activations_; }
  std::uint64_t pool_needed() const { return pool_; }
  Status Controls(const Requests<std::vector<std::int32_t>>& prompts,
                  const std::filesystem::path& out);
  Status GenerationControls(const Requests<std::vector<std::int32_t>>& prompts,
                            const std::filesystem::path& out);
  std::string receipt() const;

  std::uint32_t stream() const override { return owner_.stream(); }
  const catalog::Closure& fence_closure() const override { return closure_; }
  std::vector<catalog::ExtentId> managed_extents() const override;
  Status Release() override;

 private:
  struct Slot {
    engine::LiveState state{"C2 Qwen request"};
    engine::Mapped commit;
    engine::Mapped baseline;
    kernels::ggml::Qwen38CommitArgs commit_args;
    std::uint32_t pending = 0;
    std::vector<engine::LiveState::Range> ranges;
    std::vector<std::int32_t> initial;
  };
  struct TargetKey {
    Requests<kernels::ggml::Qwen38ChunkShape> shapes;
    bool batch = false;
    std::uint32_t active = 3;
    bool diagnostic = true;
    bool operator==(const TargetKey&) const = default;
  };
  struct DraftKey {
    Requests<kernels::ggml::Qwen38MtpShape> shapes;
    bool batch = false;
    std::uint32_t active = 3;
    bool operator==(const DraftKey&) const = default;
  };
  struct Output {
    Requests<std::vector<std::int32_t>> ids;
    Requests<std::vector<float>> values;
    Requests<std::vector<float>> head_inputs;
    Requests<std::vector<float>> head_logits;
  };
  struct GenerationStep {
    std::uint32_t active = 0;
    Requests<std::uint32_t> first{}, rows{}, keep{};
    Requests<std::vector<std::int32_t>> drafts, verdicts;
    bool operator==(const GenerationStep&) const = default;
  };
  struct GenerationRun {
    bool complete = false;
    Requests<std::vector<std::int32_t>> tokens;
    std::vector<GenerationStep> steps;
    Requests<double> completed_seconds{};
    Requests<std::uint32_t> final_pending{};
    double seconds = 0;
    double draft_seconds = 0;
    double verify_seconds = 0;
    double settle_seconds = 0;
    engine::GraphStats graphs;
    std::uint64_t mxfp8_pairs = 0, routed_pairs = 0, packed_bytes = 0;
  };
  Status GenerationShapeControls(const std::filesystem::path& out);
  Status GenerationState(std::size_t arm, std::size_t index, const std::filesystem::path& out);
  Status Generate(bool batch, GenerationRun& run);
  Status Judge(GenerationStep& step, const Output& draft, const Output& target,
               Requests<std::vector<std::int32_t>>& histories,
               Requests<std::vector<std::int32_t>>* tokens);
  std::string GenerationReceipt() const;
  std::uint32_t End() const { return natural_ ? kNaturalEnd : kEnd; }
  Status Refresh();
  Requests<engine::Qwen38Model> Models(bool actual) const;
  Requests<std::vector<std::int32_t>> Histories() const;
  Status Initialize(const Requests<std::vector<std::int32_t>>& prompts);
  Status Reset();
  Status Transfer(std::size_t slot, bool from_owner, bool to_baseline);
  Status CheckPlan(Plan& plan);
  Status Saves(std::size_t slot, std::uint32_t first, std::uint32_t rows);
  Status Draft(const Requests<std::vector<std::int32_t>>& histories, bool batch, bool graphs,
               Output& output, std::uint32_t active = 3, bool diagnostic = true);
  Status Verify(const Requests<std::vector<std::int32_t>>& histories,
                const Requests<std::uint32_t>& first, bool batch, bool graphs, Output& output,
                std::uint32_t active = 3, bool diagnostic = true);
  Status Settle();
  Status StateProof(std::size_t arm, std::size_t round, std::string_view phase,
                    const std::filesystem::path& out);
  draft_vocab::ReferencePages::Copy Reader(std::size_t slot);
  Status Fail(std::string detail);

  engine::Qwen38Runner& owner_;
  Requests<Slot> slots_;
  catalog::Closure closure_;
  engine::GraphRuns runs_{false};
  engine::GraphStats graph_stats_;
  std::array<engine::GraphStats, 8> arm_graph_stats_;
  engine::PlanCache<TargetKey, Plan> target_plans_;  // uncharged: the proof's own shapes
  engine::PlanCache<DraftKey, Plan> draft_plans_;
  engine::Coverage coverage_;
  std::byte* output_ = nullptr;
  std::byte* live_page_ = nullptr;
  std::byte* expected_page_ = nullptr;
  kernels::ggml::RangeCopy* copies_ = nullptr;
  std::array<Requests<draft_vocab::ReferencePages>, 6> references_;
  draft_vocab::ReferenceStats reference_stats_;
  std::array<std::array<Output, 2>, 2> expected_outputs_;
  std::uint64_t checkpoint_bytes_ = 0;
  std::uint64_t activations_ = 0;
  std::uint64_t pool_ = 0;
  std::uint64_t workspace_probes_ = 0;
  std::uint32_t workspace_masks_ = 0;
  std::uint64_t conservative_activation_bytes_ = 0;
  std::uint64_t output_bytes_ = 0;
  std::uint64_t mxfp8_pairs_ = 0;
  std::uint64_t routed_pairs_ = 0;
  std::uint64_t packed_bytes_ = 0;
  std::uint64_t copied_checkpoint_bytes_ = 0;
  std::uint64_t output_controls_ = 0;
  std::uint64_t state_controls_ = 0;
  bool released_ = false;
  bool failed_ = false;
  bool complete_ = false;
  bool natural_ = false;
  std::uint32_t requests_ = 2;
  std::array<GenerationRun, 4> generation_runs_;
  Requests<std::string> generation_state_sha_;
  std::uint64_t new_shape_output_controls_ = 0;
  std::uint64_t new_shape_state_controls_ = 0;
  // The first four arms stay eager; the next four capture then replay the
  // same fixed shapes, comparing every output and initialized state page.
};

}  // namespace llmp::benchmarks::qwen_batch
#endif  // LLMP_BENCHMARKS_QWEN38_BATCH_H_
