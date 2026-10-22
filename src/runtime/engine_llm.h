// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The serving adapter of the LLM pipeline kind (D-107; docs/runtime-
// serving.md, "Engine-backed models"; docs/engine.md, "The serving
// adapter"): the one Llm, over the shared runner, of every LLM the runtime
// serves.
//
// It holds a ModelRunner (engine/model_runner.h: the slot lifecycle with
// every stage's state, the plans, the step runs), a WaveComposer over the
// components' batching capability, and two DecodingOrchestrators (plain,
// and speculative with the model's drafters when it has any), and maps each
// of Llm's units of work onto them:
//
// - a branch is the request slot of its index (Llm::BranchIndex);
// - prompt chunks (RunPrefillChunkFor, RunPreparedPrefillWave) are prefill
//   steps through RunWave, a joined wave where the capability joins prompt
//   rows, each owner's head (its last row's logits) or a state-only cut,
//   with the session's next two chunks (PrefillHint) as the step's futures
//   (lookahead, capture ahead, prepare ahead, as the parts' PrefillOptions
//   allow);
// - a decode row (RunChunkFor of one row, RunGreedyChunkFor, a plain
//   generation wave) is the plain mode's unit, its token chosen on the
//   device for a greedy row without a caller of its logits when the parts
//   say the step returns it;
// - a speculative step (SpecStepFor, a speculative generation wave) is the
//   speculative mode's unit: every drafter's passes, then one verify, its
//   verdict the model's (Llm::Keep and Choose: greedy agreement, or
//   speculative sampling seeded by position);
// - settle, clear, spill, restore, checkpoint ranges and restores, copies,
//   adoption, place checks and recovery are the SlotLifecycle's, each
//   checked there (its layout ID, footprint and positions), in place of
//   Llm's unchecked defaults (which only its tests' fakes use);
// - a conversation kept across a restart (D-105) is the record's version 2
//   (kept_record.h): a spill file a stage, the layout model::KeptLayoutId of
//   the spec bound to its artifacts and every stage's layout.
//
// Every setting comes from the model's settings (D-103): its context, prefill
// chunk, request slots and speculation, given to the build that makes its
// parts (layouts, plans) and capped here by what the parts hold.
//
// What a head row is: `logit_width` F32 logits; a greedy device choice: one
// I32, the greedy segment's own output (its one output row's token). The
// step source's lowering (kernels/ggml/lower.cc's output slots) and the
// components agree on it.
//
// Vendor-free, every build profile: the engine core and no kernel library.

#ifndef LLMP_RUNTIME_ENGINE_LLM_H_
#define LLMP_RUNTIME_ENGINE_LLM_H_

#include <algorithm>
#include <array>
#include <cstddef>
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

#include "config/node_config.h"
#include "config/storage_roles.h"
#include "engine/decoding.h"
#include "engine/model_runner.h"
#include "engine/paged_node.h"
#include "engine/prefill.h"
#include "engine/runner_owner.h"
#include "engine/wave_composer.h"
#include "execution/adaptive_wave_mode.h"
#include "model/batching.h"
#include "runtime/serving.h"

namespace llmp::runtime {

// What an engine-backed LLM runs, made from its settings.
struct EngineLlmParts {
  // What the step source and the launch borrow (the spec, the artifact's
  // tensors, a kernel source's backend): kept until the runner is gone.
  std::shared_ptr<void> keep;
  // The runner: the slots (each stage's layout, sized by the settings'
  // context and prefill chunk; the bound spec's digest, D-105's layout
  // identity), the stages' weights, setup shapes, launch, verify rows and
  // commits. EngineLlm sets the slots' model name, count (max_slots), spill
  // directory and places, and chains its own invalidation.
  engine::RunnerConfig config;
  std::unique_ptr<engine::StepSource> source;
  model::Batching caps;  // the components' batching capability
  // The decoding mode, with its drafters when the model has any (they
  // draft only with the speculation setting on); kPlain otherwise.
  engine::DecodingMode mode;
  // Optional existing counted wave-form policy. Unset preserves the source
  // recipe: every speculative session uses its declared drafter schedule.
  std::optional<execution::AdaptiveWaveMode> wave_mode{};
  std::uint32_t logit_width = 0;  // a head row's F32 logits
  // A plain greedy step returns its tokens chosen on the device.
  bool device_greedy = false;
  // A greedy verify may return one native I32 token per output row.
  bool device_verify = false;
  // The sole drafter's adaptive schedule lives with the conversation's
  // existing persisted AdaptiveDepth, rather than the orchestrator instance.
  std::uint32_t branch_depth = 0;  // persisted policy ceiling; 0 uses the ordinary bound
  // Lookahead, capture ahead and prepare ahead of prompt chunks (the chunk
  // rows are the settings').
  engine::PrefillOptions prefill;
  // Its family's name for the report ("gemma2").
  std::string architecture;
  // What of its chat template the chat route serves (the model's qualified
  // slice): thought-channel output off by default and refused on, tool
  // declarations refused, and control tokens that end a chat reply beside
  // the template's own stops (Gemma 4's channel and tool-call openers,
  // which no parser here reads). `default_thinking` is a request's
  // thinking when it names none and thinking is served (DeepSeek V4's on,
  // as llama-server's /apply-template rendered its references' prompts);
  // unset, the template's own default.
  struct ChatSlice {
    bool thinking = true;
    bool tools = true;
    std::vector<std::string> stops;
    std::optional<bool> default_thinking;
  };
  ChatSlice chat;
  std::optional<artifact::Artifact> target_artifact{};
  std::uint64_t startup_host_bytes = 0;
  std::uint64_t startup_cap = 0;
  // Remains live through source sealing; its counted high-water may grow
  // after the parts factory returns and must enter startup loan retirement.
  std::shared_ptr<engine::ExecutionMetadataAllowance> execution_metadata{};
};

// Makes a model's parts from its settings: the component engine's
// composer in serving (the spec, components, planner and lowering), a fake
// in the tests.
using EngineBuild =
    std::function<std::expected<EngineLlmParts, std::string>(const ModelSettings& settings)>;

class EngineLlm final : public Llm {
 public:
  // `entry` and `roles` give the chat assets (none without an artifact: no
  // chat turns, as in the tests) and the spill directory; `index` is the
  // model's registration index (owner and stream).
  EngineLlm(engine::PagedNode& node, const config::ModelEntry& entry, const ModelSettings& settings,
            const config::RuntimeRoles& roles, int index, EngineBuild build);
  EngineLlm(const EngineLlm&) = delete;
  EngineLlm& operator=(const EngineLlm&) = delete;
  EngineLlm(EngineLlm&&) = delete;
  EngineLlm& operator=(EngineLlm&&) = delete;
  ~EngineLlm() override;

  // ---------------------------------------------------------- Served
  engine::PagedModel& paged() override;
  bool has_paged() const override { return static_cast<bool>(runner_); }
  Status Setup() override;
  Status Register() override;
  Status Bind() override;
  std::uint64_t activations_needed() const override;
  std::uint64_t pool_needed() const override;
  std::uint64_t host_input_bytes() const override;
  std::uint64_t plan_floor_bytes() const override;
  std::string plan_report() const override;
  void set_plan_report_sink(kernels::PlanRecordSink sink) override;
  void ReportPlans(const kernels::PlanRecordSink& sink) const override;
  std::uint64_t startup_host_bytes() const override {
    return startup_host_bytes_ +
           (execution_metadata_ ? execution_metadata_->capacity() +
                                      engine::ExecutionMetadataAllowance::OverheadBytes()
                                : 0) +
           (chat_startup_
                ? chat_startup_->capacity() + engine::ExecutionMetadataAllowance::OverheadBytes()
                : 0);
  }
  std::string slots_report() const override;
  void ReclaimCandidates(std::uint32_t owner, bool running,
                         std::vector<memory::ReclaimCandidate>& out) override;
  std::uint64_t Reclaim(memory::ReclaimKind kind, std::uint64_t id) override;
  std::vector<catalog::ExtentId> weights() const override;
  std::vector<catalog::ExtentId> state() const override;
  std::vector<catalog::ExtentId> kept_state() const override;
  std::vector<catalog::ExtentId> unchanged_state() const override;
  void StateWrittenBack(bool whole) override;
  bool HasRetainedState() const override { return AnyBranchHasRetainedState(); }
  const catalog::Closure& everything() const override;
  const catalog::Closure& request_closure() const override;
  Status PrepareDefaultRequest() override;
  std::uint64_t weight_read_bytes() const override;
  Status CheckPlaces() override;
  void DropPlans() override;
  // D-102's rung 2: the lifecycle's recovery (its untouched spilled slots
  // kept, the rest discarded, their records removed first), the discarded
  // conversations forgotten, every plan dropped.
  Status RecoverInPlace() override;
  GraphCounts graphs() const override;
  std::string violations() const override;
  std::string extra() const override;

  // ---------------------------------------------------------- Llm
  std::uint32_t logit_width() const override { return logit_width_; }
  std::uint32_t usable_context() const override {
    return context_ - std::min(context_, context_margin_);
  }
  void Defaults(chat::Conversation& conversation) const override;
  std::span<const std::int32_t> ChatStops() const override { return chat_stops_; }
  Status SelectBranches(std::span<Branch* const> active) override;
  std::expected<std::uint64_t, std::string> StateBytesThrough(
      std::uint32_t positions) const override;
  bool supports_generation_waves() const override { return true; }
  std::size_t generation_wave_capacity() const override;
  std::size_t prefill_wave_capacity() const override;
  bool supports_causal_waves() const override {
    return !speculate_ && caps_.mixed_causal && caps_.join_prefill && caps_.max_segments > 1;
  }
  bool CompatiblePrefill(std::uint32_t past, std::uint32_t rows, std::uint32_t peer_past,
                         std::uint32_t peer_rows, bool want_head = true,
                         bool peer_want_head = true) const override;
  std::string KeptLayout() const override;
  std::expected<std::optional<base::Sha256Digest>, std::string> ExecutionIdentity() const override;
  std::vector<std::uint64_t> KeptRegions() const override;
  std::vector<std::uint64_t> KeptLayouts() const override;

  // The runner, after Setup (the tests' and the report's view).
  const engine::ModelRunner& runner() const { return *runner_; }
  engine::ModelRunner& runner() { return *runner_; }
  // Units run so far: joined prefill steps and their rows, generation
  // waves of more than one owner, speculative units and what they drafted
  // and kept.
  struct Counts {
    std::uint64_t joined_prefill_steps = 0, joined_prefill_rows = 0;
    std::uint64_t joined_causal_steps = 0, joined_causal_rows = 0;
    std::uint64_t joined_waves = 0, joined_units = 0;
    std::uint64_t speculative_units = 0, drafted = 0, accepted = 0, discarded = 0;
    std::uint64_t greedy_tokens = 0;
  };
  const Counts& counts() const { return counts_; }

 protected:
  Status CheckConversation(const chat::Conversation& conversation) const override;
  // -------------------------------------------- the units of work
  Status RunChunkFor(Branch& branch, std::span<const std::int32_t> all, std::uint32_t past,
                     bool inject, std::vector<float>& logits) override;
  std::optional<Status> RunGreedyChunkFor(Branch& branch, std::span<const std::int32_t> all,
                                          std::uint32_t past, std::int32_t& token) override;
  Status RunPrefillChunkFor(Branch& branch, std::span<const std::int32_t> all, std::uint32_t past,
                            bool inject, bool want_head, std::vector<float>& logits,
                            PrefillHint next) override;
  Status PreparePrefillStateFor(Branch& branch, std::uint32_t past, std::uint32_t rows) override;
  std::uint32_t ScoreChunkRowsFor(const Branch& branch) const override;
  Status RunPreparedPrefillWave(std::span<PreparedPrefill> prepared) override;
  Status RunPreparedCausalWave(std::span<PreparedPrefill> prompts,
                               std::span<PreparedGeneration> generations) override;
  Status RunPreparedGenerationWave(std::span<PreparedGeneration> prepared) override;
  Status SpecStepFor(Branch& branch, std::span<const std::int32_t> all, std::uint32_t pos,
                     std::uint32_t left, std::vector<std::int32_t>& kept,
                     std::vector<std::vector<float>>* logits, std::uint64_t& drafted,
                     bool* prefix_kept) override;
  bool GenerationCohortUsable() const override;
  // A speculative step commits a whole verify's rows at most (every
  // drafter's depth and the anchor, which may pass AdaptiveDepth's bound).
  std::uint32_t StepTokenBoundFor(const Branch& branch, std::uint32_t left) const override;
  Status PrepareSamplingScratchFor(Branch& branch, std::size_t rows) override;
  void RetireSamplingScratchFor(Branch& branch) override;

  // -------------------------------------------- the state
  Status SettleFor(Branch& branch) override;
  Status ClearStateFor(Branch& branch) override;
  bool StateUsableFor(const Branch& branch) const override;
  bool StateRefusedFor(const Branch& branch) const override;
  Status ReleaseIdleStateFor(Branch& branch) override;
  Status SpillFor(Branch& branch) override;
  Status RestoreFor(Branch& branch) override;
  bool SpilledFor(const Branch& branch) const override;
  std::uint64_t SpilledBytesFor(const Branch& branch) const override;
  std::uint64_t RetainedSpillBytesFor(const Branch& branch) const override;
  Status ReleaseRetainedSpillFor(Branch& branch) override;
  bool LeasedFor(const Branch& branch) const override;
  std::uint64_t RefusedBytesFor(const Branch& branch) const override;
  std::uint64_t SpillWriteBytesFor(const Branch& branch) const override;
  Status PrepareDecodeStateFor(Branch& branch, std::uint32_t pos, std::uint32_t left) override;
  std::uint64_t TargetStateBaseFor(const Branch& branch) const override;
  std::uint64_t TargetStateBytesFor(const Branch& branch) const override;
  std::uint64_t DrafterStateBaseFor(const Branch& branch) const override;
  std::uint64_t DrafterStateBytesFor(const Branch& branch) const override;
  std::uint64_t UsedStateBytesFor(const Branch& branch) const override;
  std::vector<engine::LiveState::Range> UsedStateRangesFor(const Branch& branch) const override;
  Status SaveUsedStateFor(Branch& branch, void* host,
                          std::span<const engine::LiveState::Range> ranges) override;
  // A restore names its positions: through RestoreSnapshotFor and the
  // position-aware checkpoint restore only.
  Status RestoreUsedStateFor(Branch& branch, void* host,
                             std::span<const engine::LiveState::Range> ranges) override;
  std::expected<std::vector<engine::LiveState::Range>, std::string> CheckpointRangesFor(
      const Branch& branch, std::uint32_t positions) const override;
  Status PrepareRestoreStateFor(Branch& branch,
                                std::span<const engine::LiveState::Range> footprint) override;
  Status CopyCheckpointStateFor(Branch& branch, void* host,
                                std::span<const engine::LiveState::Range> ranges,
                                bool to_host) override;
  // `layout`: the restored state's kept layout ID, which the lifecycle
  // holds against its own (a turn checkpoint's, or its kept record's).
  Status PreparePositionRestoreFor(Branch& branch, std::uint32_t position, std::uint32_t cursor,
                                   std::span<const engine::LiveState::Range> footprint,
                                   std::string_view layout) override;
  Status CheckCheckpointMetadataFor(
      const Branch& branch, std::uint32_t position, std::uint32_t cursor,
      std::span<const engine::LiveState::Range> footprint) const override;
  Status CompletePositionRestoreFor(Branch& branch, std::uint32_t position) override;
  Status RestoreSnapshotFor(Branch& branch, void* host,
                            std::span<const engine::LiveState::Range> ranges, std::uint32_t cursor,
                            std::string_view layout) override;
  // `layout`: the kept record's (identity.layout), held against the
  // lifecycle's own.
  Status AdoptPositionFor(Branch& branch, std::span<const std::int32_t> tokens,
                          std::uint32_t cursor, std::span<const engine::LiveState::Range> ranges,
                          std::string_view layout) override;
  // The cursor is the slot's committed positions (a restore, an adoption or
  // a step publishes them); the drafters' own catch-up cursors are the
  // decoding orchestrator's, reset to them after any of those.
  std::uint32_t CursorFor(const Branch& branch) const override;
  void SetCursorFor(Branch& /*branch*/, std::uint32_t /*value*/) override {}
  void SaveDecodingStateFor(Branch& branch) override { SaveBranchDecoding(branch); }
  void RestoreDecodingStateFor(Branch& branch) override { RestoreBranchDecoding(branch); }
  execution::AdaptiveDepth TurnDecodingStateFor(const Branch& branch) const override {
    return BranchDecoding(branch);
  }
  void RestoreTurnDecodingStateFor(Branch& branch, const execution::AdaptiveDepth& state) override {
    BranchDecoding(branch) = state;
  }

  // -------------------------------------------- kept (D-105)
  std::vector<const engine::LiveState*> KeptLivesFor(const Branch& branch) const override;
  bool KeptWholeFor(const Branch& branch) const override;
  bool OwedFor(const Branch& branch) const override;
  void SetStageSpillPlaces(const StageSpillPlace& place) override;

 private:
  // One prompt chunk a unit, run as one wave (RunWave): each unit's result
  // its own.
  Status RunPrefill(std::span<PreparedPrefill> prepared);
  // One plain decode row a unit (greedy on the device where `greedy` and
  // the parts allow).
  void RunPlain(std::span<PreparedGeneration> units);
  // One speculative unit each.
  void RunSpeculative(std::span<PreparedGeneration> units);
  // A head row's logits from a step's outputs (`row` of them).
  std::expected<std::vector<float>, std::string> LogitsRow(std::span<const std::byte> outputs,
                                                           std::size_t row) const;
  std::expected<std::size_t, std::string> HeadRows(std::span<const std::byte> outputs) const;
  // The decoding orchestrators' drafter cursors follow a slot's positions
  // after anything that changes its state from outside a unit.
  void ResetDrafters(std::uint32_t slot);
  void SyncDrafters(std::span<const engine::DecodeWork> work);
  std::uint32_t Slot(const Branch& branch) const { return BranchIndex(branch); }
  Status Built() const;
  engine::DecodingPolicy Policy();

  engine::PagedNode& node_ref_;
  config::ModelEntry entry_;
  std::filesystem::path store_;
  std::filesystem::path spill_;
  int index_;
  EngineBuild build_;
  StageSpillPlace stage_place_;
  std::string architecture_;

  // Declared before the runner, so it outlives it (EngineLlmParts::keep).
  std::shared_ptr<void> keep_;
  std::uint64_t startup_host_bytes_ = 0;
  std::shared_ptr<engine::ExecutionMetadataAllowance> execution_metadata_{};
  engine::RunnerOwner runner_;
  std::optional<base::Sha256Digest> execution_identity_{};
  std::optional<engine::WaveComposer> composer_;
  std::optional<engine::DecodingOrchestrator> plain_;
  std::optional<engine::DecodingOrchestrator> speculative_;
  std::optional<execution::AdaptiveWaveMode> wave_mode_;
  model::Batching caps_;
  engine::PrefillOptions prefill_;
  std::uint32_t logit_width_ = 0;
  bool device_greedy_ = false;
  bool device_verify_ = false;
  std::uint32_t branch_depth_ = 0;
  std::uint32_t context_margin_ = 0;
  std::vector<engine::DrafterSpec> drafter_specs_;
  EngineLlmParts::ChatSlice chat_;
  std::vector<std::int32_t> chat_stops_;
  std::uint32_t slots_ = 1;
  std::vector<Branch*> active_branches_;
  std::uint32_t verify_rows_ = 1;
  // Per slot: its drafters' cursors to reset before its next speculative
  // unit; and the judge's record of a verify's kept rows' logits, when the
  // unit asked for them.
  std::array<bool, kMaxBranches> reset_{};
  std::array<bool, kMaxBranches> want_rows_{};
  std::array<bool, kMaxBranches> verify_tokens_{};
  std::array<std::vector<std::vector<float>>, kMaxBranches> verified_rows_{};
  Counts counts_;
  kernels::PlanRecordSink plan_report_sink_{};
};

}  // namespace llmp::runtime

#endif  // LLMP_RUNTIME_ENGINE_LLM_H_
