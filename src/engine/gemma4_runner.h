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
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "engine/gemma4_plan.h"
#include "engine/gemma4_wave.h"
#include "engine/graph_runs.h"
#include "engine/live_state.h"
#include "engine/paged_weights.h"
#include "engine/request_cohort.h"
#include "engine/runner_resources.h"
#include "model/gemma4_assistant.h"

namespace jitllm::engine {
class Gemma4Assistant;
// Slab pitch preserves both 256-byte region alignment and GGML blocks.
std::expected<std::uint64_t, std::string> Gemma4ExpertPitch(
    std::uint64_t minimum, std::span<const std::string_view> types);
std::expected<void, std::string> Gemma4CheckpointFootprint(
    const model::Gemma4Profile& profile, const model::Gemma4StateLayout& layout,
    std::uint32_t positions, std::span<const LiveState::Range> ranges);
enum class Gemma4Variant : std::uint8_t { k26BA4B, k31B };
struct Gemma4Options {
  std::filesystem::path artifact = {}, out = {};
  // Only approved architecture contracts; binding still checks every tensor.
  Gemma4Variant variant = Gemma4Variant::k26BA4B;
  std::uint32_t context = 4096, max_rows = 128, slots = 1;
  // Immutable publication capacity; zero retains the manual all-row envelope.
  std::uint32_t max_head_rows = 0;
  // Engine-only, ordinary target verification. Explicit features and a funded
  // head envelope are required; zero keeps verification unavailable.
  std::uint32_t max_verify_rows = 0;
  bool graphs = true, frontier_head = true;
  // Explicit assistant-feature arithmetic/placement policy. Full final rows
  // remain normalized even when only frontier logits are requested.
  bool retain_features = false;
  // Caller-funded host masks remain an explicit numerical diagnostic.
  bool reference_masks = false;
  // Checked plain RMSNorm/Mul preserves the primitive arithmetic in both
  // approved profiles. Other unqualified experiments remain explicit opt-ins.
  bool shared_q8 = false, fuse_norms = true, row_invariant = false, rope_store = false;
  // Experimental original-consumer dense input sharing; old shared_q8 wins
  // if both are set. This never enables routed VecQ or vector-float policies.
  bool dense_shared_q8 = false;
  bool fuse_norm_rope = false, fuse_norm_add = false;
  bool fuse_gemma_route = false, fuse_gemma_reduce = false;
  // Upstream's quantized one-column gate/up/GeGLU MMVQ fusion (stock's
  // decode arithmetic); never in row-invariant plans.
  bool fuse_quant_glu = false;
  bool prefill_lookahead = true;
  // Optional fresh-zero backing preparation beside the current prefill job.
  // It neither advances future cursors nor enables CPU plan lookahead.
  bool prepare_state = true;
  // Independent physical KV copies only; producers and default policy unchanged.
  bool group_kv_stores = false;
  // Explicit frontier-prefill capture policy, including one retained feature
  // per owner. Verification/all-row paths stay excluded; both profiles default off.
  bool capture_ahead = false;
  std::uint32_t prefill_lookahead_capacity = 1;
  // Explicit independent-root attention policy, immutable for this runner.
  bool owner_attention = false;
  // Initially bounded to two one-row owners; wider cohorts keep equal reads.
  bool common_owner_reads = false;
  // Closed C2 common-width attention over each actual initialized cache root.
  bool bounded_owner_roots = false;
  std::function<LiveState::SpillPlace(std::uint32_t)> spill_place = {};
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
    bool state_usable() const { return !live.quarantined() && !restoring && !verify_pending; }
    const LiveState& state() const { return live; }
    bool is_spilled() const { return spilled; }
    bool kept_whole() const {
      return on_disk && adopted.empty() && state_usable() && positions != 0;
    }
    std::uint64_t spilled_bytes() const { return spilled ? adopted_bytes + live.used_bytes() : 0; }
    std::uint64_t refused_bytes() const { return live.refused_bytes(); }

   private:
    friend class Gemma4Runner;
    LiveState live{"Gemma4"};
    catalog::Closure fence;
    bool provisioned = false, spilled = false, state_refused = false, on_disk = false;
    std::vector<LiveState::Range> adopted;
    std::uint64_t adopted_bytes = 0;
    std::optional<std::uint32_t> restoring;
    std::vector<LiveState::Range> restore_needed;
    std::vector<std::uint64_t> restored_bytes;
    std::uint32_t positions = 0;
    std::uint64_t feature_epoch = 1;
    std::uint32_t feature_first = 0, feature_count = 0;
    bool borrowed = false, verify_pending = false;
    std::uint32_t verified_rows = 0;
    Mapped snapshot;
    kernels::ggml::RangeCopy* feature_commit = nullptr;
  };
  // Move-only same-slot guard. The caller keeps its held stream request
  // alive until this guard and any component jobs have actually completed.
  // It dispatches no work itself; component failure quarantines uncertain
  // shared state before the guard can be released.
  class FrozenBorrow {
   public:
    FrozenBorrow() = default;
    FrozenBorrow(const FrozenBorrow&) = delete;
    FrozenBorrow& operator=(const FrozenBorrow&) = delete;
    FrozenBorrow(FrozenBorrow&& other) noexcept;
    FrozenBorrow& operator=(FrozenBorrow&& other) noexcept;
    ~FrozenBorrow();
    std::uint32_t slot() const { return slot_; }
    std::uint32_t prefix() const { return prefix_; }
    std::int32_t anchor() const { return anchor_; }
    std::uint64_t feature_epoch() const { return epoch_; }

   private:
    friend class Gemma4Runner;
    friend class Gemma4Assistant;
    void Reset();
    Gemma4Runner* owner_ = nullptr;
    std::uint32_t slot_ = 0, prefix_ = 0;
    std::int32_t anchor_ = 0;
    std::uint64_t epoch_ = 0, feature_ = 0;
    struct FeatureGeneration {
      catalog::ExtentId extent;
      std::uint64_t backing = 0, content = 0;
    };
    std::array<FeatureGeneration, 2> feature_generations_{};
    std::uint32_t feature_extents_ = 0;
    std::array<std::uint8_t, 32> cache_generations_{};
  };
  std::expected<FrozenBorrow, std::string> BorrowFrozen(std::uint32_t slot, std::int32_t anchor);
  Status CheckBorrow(const FrozenBorrow& borrow) const;
  // Completed host diagnostic copy; caller supplies node-owned, funded pinned
  // memory. Failed submission/retirement retains that buffer to process exit.
  Status CopyFeatures(std::uint32_t slot, std::uint32_t first_position, std::uint32_t rows,
                      void* pinned);
  struct Work {
    std::uint32_t slot = 0, n_past = 0;
    std::span<const std::int32_t> tokens;
    // Either this segment's frontier or every row, according to all_outputs.
    std::vector<float>* logits = nullptr;
    // Instead of `logits` (every segment of a wave alike): the frontier's
    // greedy token, chosen on the device; the row is not published.
    std::int32_t* token = nullptr;
  };
  // Caller keeps this slot's request held until synchronous Accept/Discard
  // retires. These completed rows are judge inputs, not a published prefix.
  // The caller funds both output vectors before calling Verify.
  Status Verify(std::uint32_t slot, std::uint32_t past, std::span<const std::int32_t> tokens,
                std::vector<float>& all_heads, std::vector<float>& all_features);
  Status AcceptVerify(std::uint32_t slot, std::uint32_t keep);
  Status DiscardVerify(std::uint32_t slot);
  Gemma4Runner(PagedNode& node, Gemma4Options options, int owner, std::uint32_t stream);
  ~Gemma4Runner() override;
  // Between Setup and node.Start/Register. The caller supplies native decoded
  // vocabulary views authenticated to the corresponding kept artifact files,
  // with all parser/container bytes funded before allocation.
  std::expected<Gemma4Assistant*, std::string> SetupAssistant(
      const std::filesystem::path& artifact,
      const model::Gemma4AssistantVocabulary& target_vocabulary,
      const model::Gemma4AssistantVocabulary& assistant_vocabulary);
  Gemma4Runner(const Gemma4Runner&) = delete;
  Gemma4Runner& operator=(const Gemma4Runner&) = delete;
  void SetSpillPlaces(const std::function<LiveState::SpillPlace(std::uint32_t)>& place) {
    o_.spill_place = place;
  }
  Status Setup();
  Status Register();
  Status Bind();
  std::uint64_t activations_needed() const { return activation_bytes_; }
  std::uint64_t pool_needed() const { return scratch_bytes_; }
  std::uint64_t host_input_bytes() const { return host_input_bytes_; }
  std::uint64_t plan_floor_bytes() const { return plan_floor_bytes_; }
  std::uint64_t plans_bytes() const { return account_.bytes(); }
  std::uint64_t weight_bytes() const { return weights_.bytes(); }
  std::uint64_t weight_read_bytes() const { return weights_.read_bytes(); }
  std::uint64_t graph_measured_bytes() const { return plans_.graph_measured_bytes(); }
  std::uint64_t graph_count() const { return plans_.graphs(); }
  std::uint64_t slab_padding() const { return weights_.slab_padding(); }
  std::uint64_t pitch_padding() const { return pitch_padding_; }
  const model::Gemma4Profile& profile() const { return profile_; }
  const model::Gemma4StateLayout& layout() const { return layout_; }
  const std::string& CheckpointLayoutId() const { return checkpoint_layout_id_; }
  std::expected<Slot*, std::string> request_slot(std::uint32_t slot);
  Status SelectSlots(std::span<const std::uint32_t> slots);
  const catalog::Closure& closure() const { return execution_; }
  const catalog::Closure& everything() const { return everything_; }
  bool cohort_usable() const { return !cohort_.faulted(); }
  bool Held(std::uint32_t slot) const { return cohort_.IsActive(slot) && node_.InRequest(stream_); }
  void StateWrittenBack(bool whole);
  Status CheckPlaces();
  Status ValidateFootprint(std::uint32_t positions, std::span<const LiveState::Range> ranges) const;
  Status PrepareRestore(std::uint32_t slot, std::uint32_t positions,
                        std::span<const LiveState::Range> footprint,
                        std::string_view source_layout);
  Status CompleteRestore(std::uint32_t slot, std::uint32_t positions);
  Status Adopt(std::uint32_t slot, std::uint32_t positions,
               std::span<const LiveState::Range> footprint, std::string_view source_layout);
  std::vector<catalog::ExtentId> weights() const;
  std::vector<catalog::ExtentId> state() const;
  // Zeroed backing its clears kept out of the state (LiveState::ZeroForReuse).
  std::vector<catalog::ExtentId> kept_state() const;
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
  // Trusted callers retain the source identity with the saved bytes. A tag
  // comparison prevents accidental cross-variant use, not fabricated metadata.
  Status RestoreCheckpoint(std::uint32_t slot, std::uint32_t positions, void* pinned,
                           std::span<const LiveState::Range> ranges, std::string_view source_layout,
                           LiveState::CopyRetirement* retirement = nullptr);
  std::expected<std::vector<LiveState::Range>, std::string> CheckpointRanges(
      std::uint32_t positions) const;
  Status Chunk(std::uint32_t n_past, std::span<const std::int32_t> tokens,
               std::vector<float>& logits, bool all_outputs = false);
  Status Wave(std::span<const Work> work, bool all_outputs = false, bool all_features = false);
  // Intermediate prompt chunks may finish after final KV stores. Requests
  // needing retained features keep the existing complete output path.
  struct PrefillNext {
    std::uint32_t slot = 0, rows = 0, after = 0;
  };
  // A bounded prediction: no future tokens or logical state are published.
  // Optional preparation may initialize eligible backing beside current work.
  // Next slots must belong to this wave; their past is its completed end.
  Status WavePrefill(std::span<const Work> work, bool want_head = true,
                     std::span<const PrefillNext> next = {}, bool next_want_head = true,
                     bool after_want_head = true);
  Status ChunkPrefill(std::uint32_t n_past, std::span<const std::int32_t> tokens,
                      std::vector<float>& logits, bool want_head = true,
                      std::uint32_t next_rows = 0, bool next_want_head = true,
                      std::uint32_t after_rows = 0, bool after_want_head = true);
  struct LookaheadStats {
    std::uint64_t attempted = 0, built = 0, cached = 0, refused = 0;
    std::uint64_t built_pairs = 0, cached_pairs = 0;
    std::uint64_t captured_first = 0, captured_ahead = 0, dropped_ahead = 0;
    double build_seconds = 0;
  };
  const LookaheadStats& lookahead_stats() const { return lookahead_; }
  void DropPlans();
  void ReclaimCandidates(std::uint32_t owner, bool running,
                         std::vector<memory::ReclaimCandidate>& out);
  std::uint64_t Reclaim(memory::ReclaimKind kind, std::uint64_t id);
  const GraphStats& graph_stats() const { return graph_stats_; }
  // Tokens published by greedy waves (Work::token) so far.
  std::uint64_t greedy_tokens() const { return greedy_tokens_; }
  const Coverage& coverage() const { return coverage_; }
  // Manual diagnostic only; off by default, no per-call logging.
  enum class Phase : std::uint8_t {
    kChecks,
    kInputs,
    kState,
    kPlanning,
    kStaging,
    kExecution,
    kPublication,
    kCount
  };
  struct PhaseAccounting {
    std::array<double, static_cast<std::size_t>(Phase::kCount)> seconds{};
    std::uint64_t planned_calls = 0, hits = 0, misses = 0;
    // Actual Clear calls, including refused calls. Separate from the seven
    // wave phases; captures reset/eviction and closure refresh, not just fill.
    double clear_seconds = 0;
    std::uint64_t clear_calls = 0;
    // Nested CachePlanned elapsed intervals: not additive with the enclosing
    // required-planning or post-completion publication phases.
    double bind_seconds = 0, coverage_seconds = 0, cache_seconds = 0;
  };
  void EnablePhaseAccounting() { account_phases_ = true; }
  PhaseAccounting TakePhaseAccounting() {
    auto taken = phases_;
    phases_ = {};
    return taken;
  }
  std::size_t plan_count() const { return plans_.size(); }
  struct PolicyCounts {
    std::uint32_t rows = 0, segments = 0, norm_fused = 0, rope_store = 0;
    std::uint32_t grouped_store_steps = 0, grouped_stores = 0, primitive_store_steps = 0;
    std::uint32_t shared_vecq = 0, row_products = 0, lane_steps = 0;
    std::uint32_t q8_preparations = 0, prepared_mmvq_products = 0;
    std::uint32_t norm_rope = 0, norm_add = 0;
    std::uint32_t gemma_route = 0, gemma_reduce = 0;
    // Selected plan implementations, not executions or per-replay launches.
    std::uint32_t owner_attention_steps = 0, bounded_owner_steps = 0;
    // Requested immutable node geometry; a device plan can fall back to four.
    std::uint32_t requested_cohort8_steps = 0, requested_cohort12_steps = 0;
    std::uint32_t requested_partial_cohort_steps = 0;
  };
  const PolicyCounts& last_built_policy() const { return policy_; }

 private:
  friend class Gemma4Assistant;
  Status WaveWithMode(std::span<const Work> work, bool all_outputs, bool all_features,
                      kernels::ggml::Gemma4OutputMode mode, std::span<const PrefillNext> next = {},
                      bool next_want_head = true, bool verify = false, bool after_want_head = true);
  void InvalidateFeatures(Slot& slot);
  // A verify whose rollback failed or whose state is lost: the slot is
  // quarantined and awaits no Accept, so Clear (or Release) can run.
  static void AbandonVerify(Slot& slot);
  std::expected<std::array<std::uint8_t, 32>, std::string> CacheGenerations(const Slot& slot) const;
  Status RefreshClosures(SlotMask protect);
  Status RefreshClosures() { return RefreshClosures(cohort_.active()); }
  std::array<LiveState*, kMaxRequestSlots> States();
  Status CheckActive(const Slot& slot) const;
  Status CheckFactors();
  Status ReserveWeights();
  kernels::ggml::DeviceChoices Choices(kernels::ggml::LaunchContext& launch,
                                       std::uint32_t rows) const;
  std::expected<Plans::Entry*, std::string> Planned(const kernels::ggml::Gemma4ChunkShape& shape);
  std::expected<Plans::Entry*, std::string> CachePlanned(
      const kernels::ggml::Gemma4ChunkShape& shape, std::unique_ptr<Gemma4Planned> planned,
      double seconds, const std::function<void()>& transfer_charge = {});
  PagedNode& node_;
  Gemma4Options o_;
  int owner_;
  std::uint32_t stream_;
  model::Gemma4Profile profile_ = model::Gemma4_26BA4B();
  model::Gemma4Binding binding_;
  model::Gemma4StateLayout layout_;
  std::string checkpoint_layout_id_;
  RunnerResources resources_;
  Mapped features_, verify_features_;
  void* verify_feature_host_ = nullptr;
  std::uint64_t verify_feature_slot_bytes_ = 0, verify_host_charge_ = 0;
  std::uint64_t feature_slot_bytes_ = 0;
  PagedWeights weights_;
  std::unique_ptr<Gemma4Assistant> assistant_;
  std::array<std::unique_ptr<Slot>, kMaxRequestSlots> slots_;
  Gemma4Model model_;
  RequestCohort cohort_{"Gemma4"};
  catalog::Closure everything_, fence_, execution_, factor_closure_;
  PlanAccount account_;
  Plans plans_;
  GraphRuns runs_;
  GraphStats graph_stats_;
  std::uint64_t greedy_tokens_ = 0;
  // The scheduler's placement changes at the last clean CheckPlaces, until
  // the states it checks change (RefreshClosures).
  std::optional<std::uint64_t> places_clean_;
  Coverage coverage_;
  PolicyCounts policy_;
  PhaseAccounting phases_;
  LookaheadStats lookahead_;
  bool account_phases_ = false;
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
