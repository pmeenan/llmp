// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Temporary complete configured reference on the native runner skeleton.
#ifndef JITLLM_BENCHMARKS_DS4_COMPLETE_RUNNER_H_
#define JITLLM_BENCHMARKS_DS4_COMPLETE_RUNNER_H_

#include <array>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "ds4_complete/binding.h"
#include "ds4_complete/weight_preparation.h"
#include "engine/dsv4_ds4_paged_weights.h"
#include "engine/live_state.h"
#include "engine/paged_weights.h"
#include "engine/runner_resources.h"

namespace jitllm::benchmarks::ds4_complete {

struct ProfileSample {
  std::uint32_t chunk = 0, layer = 0, ratio = 0;
  std::string_view chain;
  float milliseconds = 0;
  bool active = false;
};
struct Pass {
  double seconds = 0;
  double wall_seconds = 0;
  double result_copy_seconds = 0;
  engine::StepTimes steps{};
  std::vector<float> logits;
  std::array<double, 2> chunks{};
  std::array<ResolvedDispatch, 2> dispatch{};
  // Diagnostic event samples; this pass's wall time is not a speed result.
  std::vector<ProfileSample> profile;
  OutputBConsumer output_b_consumer = OutputBConsumer::kOriginal;
  RoutedFfnTier routed_ffn_tier = RoutedFfnTier::kDirect;
  bool operand_capture = false;
};

class Runner final : public engine::PagedModel {
 public:
  explicit Runner(engine::PagedNode& node, std::uint32_t device_sms)
      : node_(node), resources_(node, 0, 0), device_sms_(device_sms) {}
  // Setup requires a freshly opened node. The caller always tears it down,
  // including partial setup failures, before any Runner owner is destroyed.
  Result Setup(const std::filesystem::path& artifact, const std::filesystem::path& scratch,
               PreparedModelWeights& prepared, bool output_b_study = false,
               bool routed_ffn_study = false);
  Result Load();
  Result Initialize();
  // Create all marks once before warmup. A partial failure still retains
  // every created handle until the caller's fenced TearDown invokes Release.
  Result PrepareProfile();
  std::expected<Pass, std::string> Prefill(std::span<const std::int32_t> tokens,
                                           bool profile = false,
                                           OutputBConsumer output_b = OutputBConsumer::kOriginal,
                                           const std::filesystem::path& capture = {},
                                           RoutedFfnTier routed_ffn = RoutedFfnTier::kDirect);
  std::uint32_t stream() const override { return 0; }
  const catalog::Closure& fence_closure() const override { return everything_; }
  std::vector<catalog::ExtentId> managed_extents() const override;
  Result Release() override;

  const engine::Ds4PreparedWeightSet& prepared_plan() const { return prepared_plan_; }
  const engine::Ds4PagedAlignedWeights& aligned() const { return aligned_; }
  const model::Ds4BaselineStateLayout& state_layout() const { return state_layout_; }
  const ScratchPlan& scratch_plan() const { return scratch_plan_; }
  std::uint64_t budget_bytes() const { return budget_bytes_; }
  std::uint64_t raw_read_bytes() const { return raw_.read_bytes(); }
  std::uint64_t physical_state_bytes() const { return state_.used_bytes(); }
  std::span<const engine::LoadStats> loads() const { return loads_; }
  std::uint32_t profile_mark_count() const { return profile_mark_count_; }
  std::array<std::uint64_t, 2> profile_setup_free_bytes() const {
    return profile_setup_free_bytes_;
  }

 private:
  Result RefreshClosure();
  Result UseState();
  Result ReadHashTables();
  Result CaptureOutputB(const Chunk& chunk, std::uint32_t layer,
                        const std::filesystem::path& directory);
  Result CaptureRoutedFfn(const Chunk& chunk, std::uint32_t layer,
                          const std::filesystem::path& directory);
  const NamedScratch* FindScratch(std::string_view name) const;
  kg::Ds4CacheBuffer State(model::Ds4BaselineStateKind kind, std::uint32_t layer) const;

  engine::PagedNode& node_;
  engine::RunnerResources resources_;
  std::uint32_t device_sms_ = 0;
  engine::PagedWeights raw_;
  engine::Ds4PagedAlignedWeights aligned_;
  engine::Ds4PreparedWeightSet prepared_plan_;
  model::Dsv4Binding native_binding_;
  model::Ds4BaselineStateLayout state_layout_;
  engine::LiveState state_{"temporary ds4 reference"};
  ScratchPlan scratch_plan_;
  std::vector<engine::Mapped> mapped_;
  std::vector<NamedScratch> named_;
  std::vector<HashTable> hash_tables_;
  void* host_tokens_ = nullptr;
  void* host_hashes_ = nullptr;
  void* host_logits_ = nullptr;
  void* host_decode_table_ = nullptr;
  void* host_output_b_capture_ = nullptr;
  bool output_b_study_ = false;
  bool routed_ffn_study_ = false;
  std::uint64_t hash_bytes_ = 0;
  std::uint64_t budget_bytes_ = 0;
  std::uint64_t generation_ = 0;
  bool loaded_ = false;
  bool initialized_ = false;
  ProfileMarks profile_{};
  std::uint32_t profile_mark_count_ = 0;
  std::array<std::uint64_t, 2> profile_setup_free_bytes_{};
  catalog::Closure everything_;
  std::vector<engine::LoadStats> loads_;
};

}  // namespace jitllm::benchmarks::ds4_complete
#endif  // JITLLM_BENCHMARKS_DS4_COMPLETE_RUNNER_H_
