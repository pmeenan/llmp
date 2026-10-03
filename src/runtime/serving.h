// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Serving on the node (D-096; docs/runtime-serving.md): the configured
// models registered on one paged node (engine/paged_node.h), the full swap
// between them, and each model's turns. CUDA builds only; the commands
// (commands.h) drive it from the runtime's main thread, the node's one
// driver.
//
// - Registration (Server::Start): every configured model's artifacts opened
//   (under the store's trust rules first: the runtime's user and root
//   alone), its runner set up on its own stream, its tokenizer and chat
//   renderer found (D-067: a template runs only if a native renderer has
//   its hash), the shared workspace mapped at the largest model's need, and
//   the scheduler started with a physical cap from available memory,
//   leaving room for host-built chunk inputs and a 6 GiB guard. The
//   largest model's weights must fit beside the fixed allocations.
//   Nothing is paged yet: a model's weights come in when it is first
//   activated.
// - One model is resident at a time (M3's full swap). Activating another
//   swaps: the resident model's conversation state is written back through
//   the landing zone if it holds a conversation (0 context: it stays), its
//   weights evicted with their backing handed to the incoming model's
//   loads (D-033), and the incoming model's whole closure paged in (its
//   state restored if it was spilled); then its checks of what its kernels
//   index unchecked, and its places checked still pinned (D-090). Each part
//   is timed (SwapParts).
// - An LLM holds one conversation: `history` is every token its state has
//   seen. A turn's tokens extend it when they start with it; otherwise a
//   matching retained turn checkpoint restores its prefix, or the state
//   clears before fresh prefill. Generation is greedy, speculative by default
//   where the model has a drafter (DSpark for DeepSeek, MTP for Qwen3.8),
//   and each turn is one request: the model's closure leased once, every
//   chunk a step under it (D-093). The prefill's chunk is the model's
//   (runtime/prefill.h), and a caller's `go_on` may stop a prefill between
//   chunks, the history then the chunks that ran.
// - The image pipeline generates the prompt and latents it registered with
//   (this slice's runner fixes them at setup).

#ifndef JITLLM_RUNTIME_SERVING_H_
#define JITLLM_RUNTIME_SERVING_H_

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <expected>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "catalog/catalog.h"
#include "chat/chat.h"
#include "config/node_config.h"
#include "config/storage_roles.h"
#include "engine/checkpoint_file.h"
#include "engine/live_state.h"
#include "engine/paged_node.h"
#include "execution/adaptive_depth.h"
#include "execution/sampling.h"
#include "memory/reclaim.h"
#include "runtime/commands.h"
#include "runtime/prefill.h"
#include "runtime/pressure_trim.h"
#include "runtime/turn_reuse.h"
#include "scheduler/scheduler.h"
#include "tokenizer/tokenizer.h"

namespace jitllm::runtime {

using Status = std::expected<void, std::string>;
using Clock = std::chrono::steady_clock;

// The lowest available memory (platform::AvailableMemoryBytes, Linux's
// MemAvailable) since the last Reset, sampled every 20 ms on a thread of its
// own, and the lowest over its life.
class MemorySampler {
 public:
  MemorySampler();
  MemorySampler(const MemorySampler&) = delete;
  MemorySampler& operator=(const MemorySampler&) = delete;
  MemorySampler(MemorySampler&&) = delete;
  MemorySampler& operator=(MemorySampler&&) = delete;
  ~MemorySampler();
  void Reset();
  std::uint64_t low() const { return low_.load(); }
  std::uint64_t all() const { return all_.load(); }
  std::uint64_t start() const { return start_; }
  // In use at the lowest point since the last Reset, against the start.
  std::uint64_t peak() const { return start_ > low() ? start_ - low() : 0; }

  static std::uint64_t Available();

 private:
  void Loop();
  std::uint64_t start_ = 0;
  std::atomic<std::uint64_t> low_{0};
  std::atomic<std::uint64_t> all_{0};
  std::atomic<bool> stop_{false};
  std::jthread thread_;
};

// When each extent last became resident (the scheduler's page-in
// observer): written on the scheduler's thread, read by the driver once
// the program that paged it in is gone.
class ResidentTimes final : public scheduler::PageInObserver {
 public:
  explicit ResidentTimes(std::size_t extents) : at_(extents) {}
  void Staged(catalog::ExtentId extent, scheduler::PageInEvent event) override;
  Clock::time_point Latest(std::span<const catalog::ExtentId> extents) const;

 private:
  std::vector<Clock::time_point> at_;
};

// How a model's chunks ran: launched, captured as decode graphs and
// replayed (D-090), the target's and a drafter's summed.
struct GraphCounts {
  std::uint64_t eager = 0;
  std::uint64_t captured = 0;
  std::uint64_t replayed = 0;
  std::uint64_t refused = 0;
  std::uint64_t kept = 0;  // captured and not yet destroyed
};

// One swap's parts, in seconds, each from the end of the one before.
struct SwapParts {
  std::string from;
  std::string to;
  bool with_state = false;  // the outgoing model's conversation spilled
  double evict = 0;         // the request to every outgoing extent evicted (or parked)
  double restore = 0;       // then the incoming model's state resident (0 if it had none spilled)
  double page_in = 0;       // then its weights resident
  double setup = 0;         // then its checks (DeepSeek's hash routing, Qwen3.8's n-gram hash)
  double total = 0;         // the request to the end of setup (the first output is the turn's)
  double release = 0;       // after it: backing no load took, released (off the path)
  std::uint64_t evicted = 0;
  std::uint64_t loaded = 0;
  std::uint64_t read_bytes = 0;     // weights and state paged in
  std::uint64_t spilled_bytes = 0;  // the outgoing state written back
  std::uint64_t handed_off = 0;
  std::uint64_t released_unused = 0;
  std::uint64_t dropped_graphs = 0;  // graphs the reclaim order took for the incoming model
  // A failed Activate that left the node as it was (the outgoing model
  // resident and usable, the incoming one not): the swap was refused for
  // room, before it started or with its effects undone. Only the request
  // that needed it fails.
  bool refused = false;
  Clock::time_point requested;
  Clock::time_point ready;  // setup's end
};

// A configured model on the node.
class Served {
 public:
  Served() = default;
  Served(const Served&) = delete;
  Served& operator=(const Served&) = delete;
  Served(Served&&) = delete;
  Served& operator=(Served&&) = delete;
  virtual ~Served() = default;

  const std::string& name() const { return name_; }
  virtual bool llm() const = 0;
  virtual engine::PagedModel& paged() = 0;
  virtual Status Setup() = 0;
  virtual std::uint64_t activations_needed() const = 0;
  virtual std::uint64_t pool_needed() const = 0;
  // What a chunk builds on the host before staging it (its inputs), at
  // most: the node's fixed memory beside what the catalog maps.
  virtual std::uint64_t host_input_bytes() const { return 0; }
  // What one step of the model holds of plans at once at most
  // (engine/planned.h): set apart beside the catalog's budget, as the host
  // inputs are, by the start's memory guard. Every plan and graph past it
  // is charged inside the budget and given back through the node's one
  // reclaim order (Server::Reclaim); a swap keeps them (D-090 as amended).
  virtual std::uint64_t plan_floor_bytes() const { return 0; }
  // What its kept graphs took of the device's free memory at their
  // captures (the count's check, for the teardown's log).
  virtual std::uint64_t graph_measured_bytes() const { return 0; }
  // How plan_floor_bytes() is made up, for the start's log (empty: none).
  virtual std::string plan_report() const { return {}; }
  // Why a model that batches concurrent requests serves this artifact one
  // at a time, for the start's log (empty: it batches as configured, or
  // never batches). A model is never refused only because its artifact
  // cannot batch.
  virtual std::string serial_reason() const { return {}; }
  // Its plans and graphs as candidates for the reclaim order (memory/
  // reclaim.h), `owner` its registration index, `running` if resident;
  // and one's reclaim: the bytes it freed (0: gone or held).
  virtual void ReclaimCandidates(std::uint32_t /*owner*/, bool /*running*/,
                                 std::vector<memory::ReclaimCandidate>& /*out*/) {}
  virtual std::uint64_t Reclaim(memory::ReclaimKind /*kind*/, std::uint64_t /*id*/) { return 0; }
  virtual Status Register() = 0;
  virtual Status Bind() = 0;
  virtual std::vector<catalog::ExtentId> weights() const = 0;
  virtual std::vector<catalog::ExtentId> state() const { return {}; }
  // A swap's incremental write-back: the state extents nothing wrote since
  // their spill files last held them (released without writing,
  // scheduler::EvictOptions::unchanged); and, after the swap's write-back,
  // whether it wrote the rest (`whole`) or failed.
  virtual std::vector<catalog::ExtentId> unchanged_state() const { return {}; }
  virtual void StateWrittenBack(bool /*whole*/) {}
  virtual bool HasRetainedState() const { return false; }
  virtual const catalog::Closure& everything() const = 0;
  // Execution may protect fewer idle retained states than full swaps do.
  virtual const catalog::Closure& request_closure() const { return everything(); }
  virtual Status PrepareDefaultRequest() { return {}; }
  virtual std::uint64_t weight_read_bytes() const = 0;
  // After a full load: the checks of what the kernels index unchecked.
  virtual Status AfterLoad() { return {}; }
  // After a swap in: a model's own check of its places beyond the pins.
  virtual Status CheckPlaces() { return {}; }
  // Forgets every plan and graph: the next chunks plan (and capture) again,
  // as a model first used in the process would.
  virtual void DropPlans() {}
  virtual double plan_seconds() const { return 0; }
  // What a model reads on demand during its chunks (Qwen3.8's n-gram rows).
  virtual double demand_seconds() const { return 0; }
  virtual std::uint64_t demand_bytes() const { return 0; }
  virtual GraphCounts graphs() const { return {}; }
  // Bound tensors outside cataloged extents of their class (BP-A1), or "".
  virtual std::string violations() const { return {}; }
  // The model's own counters, a JSON object.
  virtual std::string extra() const { return "{}"; }
  // Experimental allocation diagnostics, read after successful Setup and
  // before node.Start. Empty for runners without detailed diagnostics.
  virtual std::string allocation_report() const { return {}; }

 protected:
  std::string name_;
};

// A generation's result.
struct Generation {
  std::vector<std::int32_t> tokens;        // generated, the first from the prefill
  std::vector<std::vector<float>> logits;  // each token's row, when asked for
  bool stopped = false;                    // ended at a stop token
  bool cancelled = false;                  // ended by GenerateOptions::on_tokens
  std::uint64_t drafted = 0;
  std::uint64_t accepted = 0;
  std::uint64_t steps = 0;       // decode steps (verifies, when speculating)
  Clock::time_point first_step;  // when the first decode step ended
  double decode_seconds = 0;     // the tokens after the first
};

struct GenerateOptions {
  std::uint32_t max_tokens = 256;
  bool stop = true;          // end at the model's stop tokens
  bool keep_logits = false;  // every token's logits in Generation::logits
  // Seeded sampling (execution/sampling.h), keyed by the seed and each
  // token's position in the conversation; absent, or temperature 0, is
  // greedy. Speculation then verifies drafts by speculative sampling
  // (VerifyDraft), so the tokens are distributed as plain sampling's.
  std::optional<execution::SamplingParams> sampling;
  std::uint64_t seed = 0;
  // Called with each step's new tokens (the first call with the token from
  // the prefill's logits), a stop token left out; returning false ends
  // the generation after that step, the state holding what it accepted.
  std::function<bool(std::span<const std::int32_t>)> on_tokens;
  // One natural target row per generated token, including a stop token.
  // False cancels after this completed step. Rows are borrowed only during
  // the call and never accumulated unless keep_logits was also requested.
  std::function<bool(std::int32_t, std::span<const float>)> on_logits = nullptr;
};

// A model with a conversation: DeepSeek V4 Flash, Qwen3.8 Flash Next.
class Llm : public Served {
 private:
  struct TurnCheckpoint {
    TurnBoundary boundary;
    engine::CheckpointFile file;
    std::vector<engine::LiveState::Range> footprint;
    std::uint32_t cursor = 0;
    execution::AdaptiveDepth decoding;
  };

 public:
  class PromptSession;
  class GenerationSession;
  // Host conversation ownership over this model's immutable metadata. Only
  // this model constructs branches; each names one of its native state slots.
  // A branch never registers or owns another model.
  class Branch {
   public:
    Branch(const Branch&) = delete;
    Branch& operator=(const Branch&) = delete;
    Branch(Branch&&) = delete;
    Branch& operator=(Branch&&) = delete;
    ~Branch();

    const Llm& model() const { return model_; }
    const std::vector<std::int32_t>& history() const { return history_; }
    bool HasRetainedState() const { return !history_.empty(); }
    Status Clear();
    // Clears an idle branch's retained state (its finished conversation's
    // reuse cache) while it is outside the selected cohort. Families
    // without separate native slots refuse.
    Status ReleaseIdleState();
    // Whether its state is spilled (Llm::SpillIdle): resident again only
    // once its next prompt's first unit restores it.
    bool spilled() const;
    void Forget();
    Status Prefill(std::span<const std::int32_t> tokens, std::vector<float>& last,
                   const PrefillGoOn& go_on = {}, PrefillRun* run = nullptr);
    Status ScorePrompt(std::span<const std::int32_t> tokens, std::vector<float>& last,
                       const std::function<bool(std::int32_t, std::span<const float>)>& on_row,
                       const PrefillGoOn& go_on = {}, PrefillRun* run = nullptr);
    Status PreparePrompt(std::span<const std::int32_t> tokens, std::uint32_t stable_boundary,
                         std::vector<float>& last, std::uint32_t& reused,
                         const PrefillGoOn& go_on = {}, PrefillRun* run = nullptr,
                         bool fresh = false);
    // Host-only admission. The session owns the supplied prompt and performs
    // reuse, chunks and checkpointing only through declared completed units.
    // With `resume`, `tokens` may equal the history exactly (a generation
    // set aside for its peers, its state spilled): the session then only
    // restores the state and completes with no logits (ResumeGeneration
    // needs none).
    std::expected<std::unique_ptr<PromptSession>, std::string> BeginPrompt(
        std::span<const std::int32_t> tokens, std::uint32_t stable_boundary = 0, bool fresh = false,
        bool resume = false) &;
    std::size_t turn_checkpoints() const { return turn_checkpoints_.size(); }
    std::uint64_t turn_checkpoint_bytes() const;
    std::expected<std::unique_ptr<GenerationSession>, std::string> BeginGeneration(
        const std::vector<float>& last, const GenerateOptions& options, Generation& out) &;
    std::expected<std::unique_ptr<GenerationSession>, std::string> BeginGeneration(
        const std::vector<float>& last, const GenerateOptions&& options,
        Generation& out) & = delete;
    // Continues a generation `out` whose anchor (its last token, generated
    // and reported but not yet in the state) follows exactly this branch's
    // history: after a preemption spilled the state and a prompt session
    // restored it (or discarded it and rebuilt it by prefill from the history
    // the preempted session published). Nothing is chosen or reported again;
    // `last` is not read (empty after a restore).
    std::expected<std::unique_ptr<GenerationSession>, std::string> ResumeGeneration(
        const std::vector<float>& last, const GenerateOptions& options, Generation& out) &;
    std::expected<std::unique_ptr<GenerationSession>, std::string> ResumeGeneration(
        const std::vector<float>& last, const GenerateOptions&& options,
        Generation& out) & = delete;
    Status Generate(const std::vector<float>& last, const GenerateOptions& options,
                    Generation& out);
    std::uint64_t state_snapshot_bytes() const;
    Status SaveState(void* host);
    Status RestoreState(void* host);
    void InvalidateStateSnapshot() { saved_valid_ = false; }
    // Whether the last serial Prefill, ScorePrompt, PreparePrompt or
    // Generate of this branch ended on a capacity refusal (Llm::
    // set_capacity_reclaim): the state usable at its completed prefix.
    bool capacity_refused() const { return capacity_refused_; }

   private:
    friend class Llm;
    friend class PromptSession;
    friend class GenerationSession;
    explicit Branch(Llm& model, std::uint32_t slot = 0) : model_(model), slot_(slot) {}

    Llm& model_;
    const std::uint32_t slot_;
    std::vector<std::int32_t> history_;
    // A possibly run failed step invalidates this branch until native clear.
    bool needs_clear_ = false;
    std::vector<std::int32_t> saved_history_;
    std::uint32_t saved_cursor_ = 0;
    std::vector<engine::LiveState::Range> saved_ranges_;
    bool saved_valid_ = false;
    std::optional<execution::SamplingParams> sampling_;
    std::uint64_t seed_ = 0;
    std::vector<execution::SamplingCandidate> scratch_;
    std::vector<TurnCheckpoint> turn_checkpoints_;
    // When its history was last extended or reused, with the reclaim
    // order's inflation value then (memory::ReclaimInflation): its
    // retention's clock and its priority in the order.
    struct UseStamp {
      Clock::time_point at = Clock::now();
      memory::ReclaimStamp reclaim = memory::StampUse();
      UseStamp& operator=(Clock::time_point when) {
        at = when;
        reclaim = memory::StampUse();
        return *this;
      }
    };
    UseStamp history_used_;
    bool generation_active_ = false;
    bool capacity_refused_ = false;
    const PromptSession* prompt_session_ = nullptr;
    execution::AdaptiveDepth decoding_{1};
    execution::AdaptiveDepth saved_decoding_{1};
  };

  Llm();
  Branch& default_branch() & { return default_branch_; }
  const Branch& default_branch() const& { return default_branch_; }
  static constexpr std::size_t kMaxBranches = 4;
  std::size_t branches() const { return branch_count_; }
  // A stable model-owned wrapper. Looking it up performs no native work.
  std::expected<Branch*, std::string> branch(std::size_t index) &;
  // Between completed native units. The Qwen implementation protects the
  // complete selected set; selecting performs no model registration.
  virtual Status SelectBranches(std::span<Branch* const> active);
  // Serial state capacity (docs/runtime-serving.md#state-capacity-in-a-
  // cohort). A serial Prefill, ScorePrompt, PreparePrompt or Generate of a
  // branch whose prefill chunk or decode step only the state's capacity
  // refused (StateRefusedFor: before dispatch, the state usable as it was)
  // asks `reclaim` to free capacity and runs the same unit again while it
  // returns true (at most kMaxBranches times a unit). A refusal it cannot
  // relieve ends the call with the refusal, the state holding its completed
  // prefix, and Branch::capacity_refused() true. Every other failure ends it
  // as before. Called on the driver thread between completed units, with
  // the refused branch's session still open; empty clears it.
  using CapacityReclaim = std::function<bool(const Branch& refused)>;
  void set_capacity_reclaim(CapacityReclaim reclaim) { capacity_reclaim_ = std::move(reclaim); }
  // Spill, not clear (docs/runtime-serving.md#state-capacity-in-a-cohort):
  // an idle branch's state (no session open, not leased by a request)
  // written to its slot's spill file and its backing released, between
  // completed units; its history kept, so its next turn restores the state
  // exactly and continues it (the prompt's first unit). A branch whose
  // state cannot be spilled (unknown contents) is cleared instead.
  Status SpillIdle(Branch& branch);
  // The same for a cohort member set aside for its peers (its sessions
  // ended at a completed unit, its state still leased): it leaves the
  // request's lease, and begins again from its restored state.
  Status SpillSetAside(Branch& branch);
  // Whether a branch is idle: no session open, its state not leased by a
  // request open on the model's stream.
  bool BranchIdle(const Branch& branch) const;
  // What a branch's state holds (resident, spilled, or written back by a
  // swap), what of it is resident (0 while spilled), and spilled.
  std::uint64_t StateBytes(const Branch& branch) const;
  std::uint64_t ResidentStateBytes(const Branch& branch) const;
  std::uint64_t SpilledStateBytes(const Branch& branch) const;
  // When a branch was last used (its history extended or reused): its
  // retention's clock and the reclaim order's recency.
  Clock::time_point LastUsed(const Branch& branch) const;
  // The reclaim order's stamp at that last use (GreedyDual, memory/
  // reclaim.h): its idle state's priority beside its cost.
  memory::ReclaimStamp LastStamp(const Branch& branch) const;
  // What spilling its resident state would write: only what changed since
  // the slot's spill file last held it (written back, or restored from it).
  std::uint64_t SpillWriteBytes(const Branch& branch) const;
  // What of `tokens` a prompt on `branch` would reuse, as its first unit
  // decides (host-only): its live history when the prompt continues it,
  // else the latest turn checkpoint inside their common prefix; 0 when
  // neither, or when the branch's state is unusable or past its retention.
  std::size_t ReusablePrefix(const Branch& branch, std::span<const std::int32_t> tokens) const;
  // How long an idle conversation's state, resident or spilled, and its
  // turn checkpoints stay reusable (`[memory] retention_hours`; D-055's
  // idle cap): past it the next turn starts afresh.
  void set_retention(Clock::duration retention) { retention_ = retention; }
  Clock::duration retention() const { return retention_; }
  // Where it reports what the runtime's log should show (a turn
  // checkpoint not captured or not restored); unset, nothing is reported.
  void set_log(std::function<void(std::string_view)> log) { log_ = std::move(log); }
  // After a capacity refusal of `branch` (StateRefusedFor): the bytes its
  // refused growth or restore asked for (0: unknown).
  std::uint64_t RefusedBytes(const Branch& branch) const {
    CheckBranch(branch);
    return RefusedBytesFor(branch);
  }
  // Spills and restores so far, measured: the reclaim order's cost of idle
  // state (Server::Reclaim).
  struct SpillStats {
    std::uint64_t spills = 0;
    std::uint64_t spilled_bytes = 0;
    double spill_seconds = 0;
    std::uint64_t restores = 0;
    std::uint64_t restored_bytes = 0;
    double restore_seconds = 0;
  };
  const SpillStats& spill_stats() const { return spill_stats_; }
  bool llm() const override { return true; }
  // The prefill chunk's rows, and the configuration's prefill_chunk if set
  // (max_rows is at most it, capped by the model at its context).
  std::uint32_t max_rows() const { return max_rows_; }
  std::optional<std::uint32_t> configured_rows() const { return configured_rows_; }
  std::uint32_t context() const { return context_; }
  // What a conversation may use of the context: its speculative steps may
  // need rows past the last token (Qwen3.8's MTP drafts).
  virtual std::uint32_t usable_context() const { return context_; }
  bool speculative() const { return speculate_; }
  // The template's reasoning markers as tokens (Qwen3.8's and DeepSeek's
  // "<think>" and "</think>"), where the vocabulary has them.
  std::optional<std::int32_t> think_start() const { return think_start_; }
  std::optional<std::int32_t> think_end() const { return think_end_; }
  const tokenizer::Tokenizer& tokenizer() const { return *tokenizer_; }

  // Tokens of plain text (no template; BOS first where the vocabulary has
  // one, as the reference runs fed it).
  std::expected<std::vector<std::int32_t>, std::string> EncodeText(std::string_view text) const;
  // A conversation rendered by the model's chat template and tokenized.
  std::expected<std::vector<std::int32_t>, std::string> RenderChat(
      const chat::Conversation& conversation, std::uint32_t* stable_boundary = nullptr) const;
  // Text of generated tokens (control tokens left out).
  std::string Detokenize(std::span<const std::int32_t> tokens) const;
  // Template options this model's reference runs rendered with.
  virtual void Defaults(chat::Conversation& conversation) const = 0;

  // The conversation: every token the state has seen.
  const std::vector<std::int32_t>& history() const { return default_branch_.history(); }
  bool HasRetainedState() const override { return default_branch_.HasRetainedState(); }
  // The state zeroed and the conversation empty.
  Status Clear();
  // The conversation dropped without a job: the state's contents are no
  // longer wanted (a swap out does not spill them), and the next use clears
  // it first.
  void Forget();
  // Runs `tokens` after the history in chunks of max_rows (with the
  // drafter's injection when speculating): the last row's logits. With
  // `go_on`, asked with each chunk's rows before it (runtime/prefill.h):
  // false stops the prefill there, not an error: the history then holds
  // the chunks that ran (a prefix of `tokens` after it, which the state
  // processed and a later prefill continues from), `last` is empty, and
  // `run` (if given) says so.
  Status Prefill(std::span<const std::int32_t> tokens, std::vector<float>& last,
                 const PrefillGoOn& go_on = {}, PrefillRun* run = nullptr);
  // Literal teacher forcing from an empty history. Runs one completed
  // target step per supplied token, reporting token j's row from j-1.
  // The first supplied token has no preceding distribution. This bounded
  // first scorer reuses the existing one-row workspace; it has decode-like
  // throughput and does not exercise full prefill attention tiles.
  Status ScorePrompt(std::span<const std::int32_t> tokens, std::vector<float>& last,
                     const std::function<bool(std::int32_t, std::span<const float>)>& on_row,
                     const PrefillGoOn& go_on = {}, PrefillRun* run = nullptr);
  // Reuse the live prefix or restore its nearest matching turn checkpoint,
  // then prefill the suffix. Capture before the renderer's unstable assistant
  // opening, at stable_boundary; zero means no renderer boundary supplied.
  Status PreparePrompt(std::span<const std::int32_t> tokens, std::uint32_t stable_boundary,
                       std::vector<float>& last, std::uint32_t& reused,
                       const PrefillGoOn& go_on = {}, PrefillRun* run = nullptr,
                       bool fresh = false);
  // A prompt on one branch, advanced separately from peer decode steps.
  // NextUnit is host-only; Advance owns native work through its completed
  // boundary. Finish releases host ownership, not proof of GPU retirement.
  // The model outlives the session; an unfinished session cannot be destroyed.
  class PromptSession {
   public:
    enum class Phase { kReuse, kChunk, kCheckpoint };
    struct Unit {
      Phase phase = Phase::kReuse;
      std::uint32_t rows = 0;
    };
    PromptSession(const PromptSession&) = delete;
    PromptSession& operator=(const PromptSession&) = delete;
    PromptSession(PromptSession&&) = delete;
    PromptSession& operator=(PromptSession&&) = delete;
    ~PromptSession();
    bool done() const;
    std::expected<Unit, std::string> NextUnit() const;
    // With `defer_capacity`, a chunk the model refused only for state
    // capacity (Llm::StateRefusedFor, the state usable) returns its error
    // but leaves the session resumable: refused() is true, nothing was
    // processed, and the next Advance retries the same unit. Otherwise
    // (and by default) every failure ends the session.
    Status Advance(const PrefillGoOn& go_on = {}, bool defer_capacity = false);
    bool refused() const { return refused_; }
    // Only between completed units; keeps exactly the processed prefix.
    void Cancel();
    Status Finish();
    std::uint32_t reused() const { return reused_; }
    // The prompt tokens left to prefill, a scheduling hint
    // (runtime/cohort_schedule.h): before its reuse, past the branch's
    // history's common prefix with the prompt.
    std::uint32_t remaining_rows() const;
    const PrefillRun& run() const { return run_; }
    const std::vector<float>& last() const { return last_; }

   private:
    friend class Llm;
    PromptSession(Llm& model, Branch& branch, std::span<const std::int32_t> tokens,
                  std::uint32_t stable_boundary, bool fresh, bool resume);
    void NextPhase();
    void Stop();
    Status Fail(std::string error);
    Llm& model_;
    Branch& branch_;
    const std::vector<std::int32_t> tokens_;
    const std::uint32_t stable_boundary_;
    const bool fresh_;
    const bool resume_;
    std::vector<float> last_;
    std::uint32_t reused_ = 0;
    PrefillRun run_;
    Phase phase_ = Phase::kReuse;
    Status ran_;
    bool checkpoint_pending_ = false;
    bool complete_ = false;
    bool finished_ = false;
    bool advancing_ = false;
    bool refused_ = false;
  };
  std::size_t turn_checkpoints() const { return default_branch_.turn_checkpoints(); }
  std::uint64_t turn_checkpoint_bytes() const;
  // One resumable generation on the default conversation. The model,
  // options and `out` outlive the session; no other operation may mutate that
  // conversation before Finish. Borrowing preserves stateful callbacks.
  // Finish is explicit: destroying an unfinished session violates ownership.
  class GenerationSession {
   public:
    struct Step {
      std::span<const std::int32_t> all;
      std::uint32_t position = 0;
      std::uint32_t left = 0;
      bool speculative = false;
      bool need_logits = false;
    };
    GenerationSession(const GenerationSession&) = delete;
    GenerationSession& operator=(const GenerationSession&) = delete;
    GenerationSession(GenerationSession&&) = delete;
    GenerationSession& operator=(GenerationSession&&) = delete;
    ~GenerationSession();

    bool done() const;
    // Runs existing context/growth preparation; called within a declared
    // native unit, not from a scheduler's host-only eligibility check.
    // The returned tokens are borrowed until Apply/FailStep, never rebound.
    // With `defer_capacity`, a growth the model refused only for state
    // capacity (Llm::StateRefusedFor, the state usable) returns its error
    // but leaves the session active and unprepared: refused() is true and a
    // later PrepareStep retries. Otherwise every refusal ends the session.
    std::expected<Step, std::string> PrepareStep(bool defer_capacity = false);
    bool refused() const { return !refusal_.empty(); }
    const std::string& refusal() const { return refusal_; }
    // The legacy native path, applying exactly one completed scalar step;
    // `defer_capacity` as PrepareStep's.
    Status RunScalarStep(bool defer_capacity = false);
    // Apply only after the named native step completed. A speculative result
    // already includes per-model selection/Accept; drafted is an increment.
    Status ApplyPlain(std::vector<float> row);
    Status ApplySpeculative(std::vector<std::int32_t> kept, std::vector<std::vector<float>> logits,
                            std::uint64_t drafted);
    // A failed submitted step: prefix_valid requires an independently proven
    // prefix at Step::position. False invalidates history; neither proves retirement.
    Status FailStep(std::string error, bool prefix_valid = false);
    // Called at a completed boundary, with no step awaiting Apply/FailStep.
    void Cancel();
    // Once done or cancelled, settles and publishes history once. An Error
    // retains the runner's completion/lifetime obligations, not retirement proof.
    Status Finish();

   private:
    friend class Llm;
    GenerationSession(Llm& model, Branch& branch, const GenerateOptions& options, Generation& out);
    Status Begin(const std::vector<float>& last, bool resume);
    bool IsStop(std::int32_t token) const;
    bool Report();
    // Ends a deferred capacity refusal as an undeferred one would have.
    void EndRefused();
    Status ApplyTokens(std::vector<std::int32_t> kept, std::vector<std::vector<float>> logits);
    void Close();

    Llm& model_;
    Branch& branch_;
    const GenerateOptions& options_;
    Generation& out_;
    std::vector<std::int32_t> all_;
    std::uint32_t position_ = 0;
    std::uint32_t left_ = 0;
    std::size_t reported_ = 0;
    Clock::time_point start_;
    Status ran_;
    std::string refusal_;  // the last PrepareStep's deferred capacity refusal
    bool failed_prefix_valid_ = false;
    bool prepared_ = false;
    bool finished_ = false;
  };
  std::expected<std::unique_ptr<GenerationSession>, std::string> BeginGeneration(
      const std::vector<float>& last, const GenerateOptions& options, Generation& out) &;
  std::expected<std::unique_ptr<GenerationSession>, std::string> BeginGeneration(
      const std::vector<float>& last, const GenerateOptions&& options, Generation& out) & = delete;
  // One completed native unit over stable sessions belonging to this model.
  // The caller selects/leases their branches before entry. Shared dispatch
  // retains independent judgement, sampling and history publication. A
  // per-session preparation or sampling error is retained by that session:
  // it becomes done(), and Finish reports its error while peers continue.
  // An error return here means invalid cohort input or a shared native error.
  // Finish settles host ownership; neither success, StateUsable nor Finish
  // alone proves native-reference retirement after a shared native failure.
  // With `defer_capacity`, a session whose preparation was refused only for
  // state capacity stays active and unstepped instead (refused(); its peers
  // still run), for the caller to retry or end (GenerationSession::PrepareStep).
  Status RunGenerationWave(std::span<GenerationSession* const> sessions,
                           bool defer_capacity = false);
  virtual bool supports_generation_waves() const { return false; }
  // Immutable live-owner bound after runner setup. Native branch identifiers
  // may be sparse; capacity limits simultaneous owners, not their slot IDs.
  virtual std::size_t generation_wave_capacity() const { return 1; }
  // A dispatch/fence admission guard only; a healthy cohort is not proof
  // that native references have retired. The Server fences explicitly.
  bool generation_cohort_usable() const { return GenerationCohortUsable(); }
  // Drives the same resumable core to completion using ordinary native steps.
  Status Generate(const std::vector<float>& last, const GenerateOptions& options, Generation& out);
  // The whole conversation state (the target's and the drafter's, and the
  // host's speculation cursor) saved to or put back from pinned host
  // memory the caller keeps (state_snapshot_bytes() long), by one job; and
  // the history with it.
  std::uint64_t state_snapshot_bytes() const;
  Status SaveState(void* host);
  Status RestoreState(void* host);
  void InvalidateStateSnapshot() { default_branch_.InvalidateStateSnapshot(); }

  std::vector<catalog::ExtentId> state() const override = 0;

 protected:
  struct PreparedGeneration {
    GenerationSession* session = nullptr;
    Branch* branch = nullptr;
    GenerationSession::Step step;
    std::vector<float> row;
    std::vector<std::int32_t> kept;
    std::vector<std::vector<float>> logits;
    std::uint64_t drafted = 0;
    Status result;
    // Only after independently restoring this step's starting prefix.
    bool failed_prefix_valid = false;
  };
  // Success means the native unit completed. Each independent judgement
  // supplies its own result. A shared error supplies no result to apply.
  virtual Status RunPreparedGenerationWave(std::span<PreparedGeneration> prepared);
  // Execution health only: false prevents dispatch after a shared failure.
  // This is deliberately separate from native-reference retirement proof.
  virtual bool GenerationCohortUsable() const { return true; }
  // Called once, after the native slots exist. Other model families retain
  // one default branch and their existing scalar overrides.
  Status PrepareBranches(std::uint32_t count, std::uint32_t draft_depth);
  std::uint32_t BranchIndex(const Branch& branch) const;
  bool AnyBranchHasRetainedState() const;
  execution::AdaptiveDepth& BranchDecoding(Branch& branch);
  const execution::AdaptiveDepth& BranchDecoding(const Branch& branch) const;
  void SaveBranchDecoding(Branch& branch);
  void RestoreBranchDecoding(Branch& branch);

  // One chunk: all[n_past, end) after n_past (all holds every token from
  // position 0), `inject` with the drafter's pass; the last row's logits.
  virtual Status RunChunk(std::span<const std::int32_t> all, std::uint32_t n_past, bool inject,
                          std::vector<float>& logits) = 0;
  // One speculative step from `pos`, the anchor `all.back()` (at pos, not in
  // the state yet): the verify's kept tokens (the anchor's accepted drafts
  // then the next token) and, with `logits`, each kept row's logits.
  virtual Status SpecStep(std::span<const std::int32_t> all, std::uint32_t pos, std::uint32_t left,
                          std::vector<std::int32_t>& kept, std::vector<std::vector<float>>* logits,
                          std::uint64_t& drafted) = 0;
  // Runs any pending restore or commit (the last verify's).
  virtual Status Settle() = 0;
  virtual Status ClearState() = 0;
  virtual bool StateUsable() const = 0;
  virtual Status PrepareDecodeState(std::uint32_t pos, std::uint32_t left) = 0;
  virtual std::uint64_t target_state_base() const = 0;
  virtual std::uint64_t target_state_bytes() const = 0;
  virtual std::uint64_t drafter_state_base() const = 0;
  virtual std::uint64_t drafter_state_bytes() const = 0;
  virtual std::uint64_t used_state_bytes() const = 0;
  virtual std::vector<engine::LiveState::Range> used_state_ranges() const = 0;
  virtual Status SaveUsedState(void* host, std::span<const engine::LiveState::Range> ranges) = 0;
  virtual Status RestoreUsedState(void* host, std::span<const engine::LiveState::Range> ranges) = 0;
  virtual std::expected<std::vector<engine::LiveState::Range>, std::string> CheckpointRanges(
      std::uint32_t positions) const = 0;
  virtual Status PrepareRestoreState(std::span<const engine::LiveState::Range> footprint) = 0;
  virtual Status CopyCheckpointState(void* host, std::span<const engine::LiveState::Range> ranges,
                                     bool to_host) = 0;
  virtual std::uint32_t cursor() const { return 0; }
  virtual void set_cursor(std::uint32_t /*value*/) {}
  virtual void SaveDecodingState() {}
  virtual void RestoreDecodingState() {}
  virtual execution::AdaptiveDepth TurnDecodingState() const { return execution::AdaptiveDepth(1); }
  virtual void RestoreTurnDecodingState(const execution::AdaptiveDepth& /*state*/) {}

  // Explicit conversation forwarding. Defaults accept branch zero only;
  // a family with independent native slots overrides this set together.
  virtual Status RunChunkFor(Branch& branch, std::span<const std::int32_t> all,
                             std::uint32_t n_past, bool inject, std::vector<float>& logits);
  virtual Status SpecStepFor(Branch& branch, std::span<const std::int32_t> all, std::uint32_t pos,
                             std::uint32_t left, std::vector<std::int32_t>& kept,
                             std::vector<std::vector<float>>* logits, std::uint64_t& drafted);
  virtual Status SettleFor(Branch& branch);
  virtual Status ClearStateFor(Branch& branch);
  virtual bool StateUsableFor(const Branch& branch) const;
  // After a failed RunChunkFor or PrepareDecodeStateFor of this branch: true
  // when only the state's capacity refused it, before any dispatch, leaving
  // the state usable as it was. Waiting for peers' state to be freed may
  // then let it succeed. Families without this distinction never claim it.
  virtual bool StateRefusedFor(const Branch& /*branch*/) const { return false; }
  // Discards an unselected branch's native state (Branch::ReleaseIdleState).
  virtual Status ReleaseIdleStateFor(Branch& /*branch*/) {
    return std::unexpected("this model keeps no idle conversation state apart");
  }
  // A branch's native state spilled to its slot's spill file, and restored
  // (a clean capacity refusal: StateRefusedFor true, still spilled);
  // whether it is spilled, and what its spill holds; whether a request
  // open on the model's stream leases its state now.
  virtual Status SpillFor(Branch& /*branch*/) {
    return std::unexpected("this model keeps no idle conversation state apart");
  }
  virtual Status RestoreFor(Branch& /*branch*/) { return {}; }
  virtual bool SpilledFor(const Branch& /*branch*/) const { return false; }
  virtual std::uint64_t SpilledBytesFor(const Branch& /*branch*/) const { return 0; }
  virtual bool LeasedFor(const Branch& /*branch*/) const { return true; }
  virtual std::uint64_t RefusedBytesFor(const Branch& /*branch*/) const { return 0; }
  // What a spill of its resident state would write (SpillWriteBytes); by
  // default all of it.
  virtual std::uint64_t SpillWriteBytesFor(const Branch& branch) const {
    return UsedStateBytesFor(branch);
  }
  virtual Status PrepareDecodeStateFor(Branch& branch, std::uint32_t pos, std::uint32_t left);
  virtual std::uint64_t TargetStateBaseFor(const Branch& branch) const;
  virtual std::uint64_t TargetStateBytesFor(const Branch& branch) const;
  virtual std::uint64_t DrafterStateBaseFor(const Branch& branch) const;
  virtual std::uint64_t DrafterStateBytesFor(const Branch& branch) const;
  virtual std::uint64_t UsedStateBytesFor(const Branch& branch) const;
  virtual std::vector<engine::LiveState::Range> UsedStateRangesFor(const Branch& branch) const;
  virtual Status SaveUsedStateFor(Branch& branch, void* host,
                                  std::span<const engine::LiveState::Range> ranges);
  virtual Status RestoreUsedStateFor(Branch& branch, void* host,
                                     std::span<const engine::LiveState::Range> ranges);
  virtual std::expected<std::vector<engine::LiveState::Range>, std::string> CheckpointRangesFor(
      const Branch& branch, std::uint32_t positions) const;
  virtual Status PrepareRestoreStateFor(Branch& branch,
                                        std::span<const engine::LiveState::Range> footprint);
  virtual Status CopyCheckpointStateFor(Branch& branch, void* host,
                                        std::span<const engine::LiveState::Range> ranges,
                                        bool to_host);
  virtual std::uint32_t CursorFor(const Branch& branch) const;
  virtual void SetCursorFor(Branch& branch, std::uint32_t value);
  virtual void SaveDecodingStateFor(Branch& branch);
  virtual void RestoreDecodingStateFor(Branch& branch);
  virtual execution::AdaptiveDepth TurnDecodingStateFor(const Branch& branch) const;
  virtual void RestoreTurnDecodingStateFor(Branch& branch, const execution::AdaptiveDepth& state);

  // During Generate: whether it samples, the token for a row's logits at
  // a position in the conversation (greedy: the argmax), and whether the
  // target keeps a draft there (greedy: the argmax equals it); if not,
  // `next` is the token in its place.
  bool sampling() const { return default_branch_.sampling_.has_value(); }
  bool sampling(const Branch& branch) const;
  std::expected<std::int32_t, std::string> Choose(std::span<const float> row,
                                                  std::uint64_t position);
  std::expected<bool, std::string> Keep(std::span<const float> row, std::int32_t draft,
                                        std::uint64_t position, std::int32_t& next);
  std::expected<std::int32_t, std::string> Choose(Branch& branch, std::span<const float> row,
                                                  std::uint64_t position);
  std::expected<bool, std::string> Keep(Branch& branch, std::span<const float> row,
                                        std::int32_t draft, std::uint64_t position,
                                        std::int32_t& next);
  // Finds the reasoning markers in the vocabulary (after the tokenizer).
  void FindThinkTokens();
  // Chooses how the model's chat template renders (chat::ChatTemplate) and
  // its stop tokens (after the tokenizer); an error refuses the model.
  Status UseTemplate(std::string_view text);

  engine::PagedNode* node_ = nullptr;
  std::uint32_t max_rows_ = 0;  // the prefill chunk (runtime/prefill.h), set at construction
  std::optional<std::uint32_t> configured_rows_;  // the configuration's prefill_chunk
  std::uint32_t context_ = config::kDefaultContext;
  bool speculate_ = false;
  bool bos_ = false;  // EncodeText puts BOS first
  std::unique_ptr<tokenizer::Tokenizer> tokenizer_;
  std::optional<chat::ChatTemplate> template_;
  std::vector<std::int32_t> stops_;
  std::optional<std::int32_t> think_start_;
  std::optional<std::int32_t> think_end_;

 private:
  Status Clear(Branch& branch, const PromptSession* prompt = nullptr);
  Status ReleaseIdleState(Branch& branch);
  void Forget(Branch& branch, const PromptSession* prompt = nullptr);
  Status Prefill(Branch& branch, std::span<const std::int32_t> tokens, std::vector<float>& last,
                 const PrefillGoOn& go_on, PrefillRun* run);
  Status ScorePrompt(Branch& branch, std::span<const std::int32_t> tokens, std::vector<float>& last,
                     const std::function<bool(std::int32_t, std::span<const float>)>& on_row,
                     const PrefillGoOn& go_on, PrefillRun* run);
  Status PreparePrompt(Branch& branch, std::span<const std::int32_t> tokens,
                       std::uint32_t stable_boundary, std::vector<float>& last,
                       std::uint32_t& reused, const PrefillGoOn& go_on, PrefillRun* run,
                       bool fresh);
  Status CaptureTurnCheckpoint(Branch& branch, const PrefillGoOn& go_on, bool& stopped,
                               const PromptSession* prompt = nullptr);
  Status ReusePrompt(Branch& branch, std::span<const std::int32_t> tokens, std::uint32_t& reused,
                     bool fresh, bool resume, const PrefillGoOn& go_on, bool& stopped,
                     const PromptSession* prompt = nullptr);
  std::expected<std::unique_ptr<PromptSession>, std::string> BeginPrompt(
      Branch& branch, std::span<const std::int32_t> tokens, std::uint32_t stable_boundary,
      bool fresh, bool resume = false);
  std::expected<std::unique_ptr<GenerationSession>, std::string> BeginGeneration(
      Branch& branch, const std::vector<float>& last, const GenerateOptions& options,
      Generation& out, bool resume = false);
  Status Generate(Branch& branch, const std::vector<float>& last, const GenerateOptions& options,
                  Generation& out);
  Status SaveState(Branch& branch, void* host);
  Status RestoreState(Branch& branch, void* host);
  void CheckBranch(const Branch& branch) const;
  void CheckDefaultBranch(const Branch& branch) const;
  void CheckIdleGeneration(const Branch& branch, const PromptSession* prompt = nullptr) const;
  // After a serial unit of `branch` failed: whether only the state's
  // capacity refused it (the state usable); and then whether reclaiming
  // freed capacity for it to run again (set_capacity_reclaim).
  bool CapacityRefused(const Branch& branch) const;
  bool ReclaimFor(const Branch& branch);
  Branch default_branch_;
  CapacityReclaim capacity_reclaim_;
  std::array<std::unique_ptr<Branch>, kMaxBranches - 1> extra_branches_;
  std::uint32_t branch_count_ = 1;
  bool branches_prepared_ = false;
  SpillStats spill_stats_;
  // Restores the branch's spilled state (a prompt's first unit): a clean
  // capacity refusal sets `refused`, the branch still spilled.
  Status RestoreSpilled(Branch& branch, bool& refused);
  Status Spill(Branch& branch, bool set_aside);

 protected:
  std::filesystem::path checkpoint_directory_;
  Clock::duration retention_ = kTurnCheckpointRetention;
  std::function<void(std::string_view)> log_;
  void Say(std::string_view text) const {
    if (log_) {
      log_(text);
    }
  }
};

// The image pipeline (Qwen-Image-2.1).
class Image : public Served {
 public:
  bool llm() const override { return false; }
  // Encode and the first denoising step (the swap table's endpoint); its
  // noise prediction's SHA-256.
  virtual Status FirstOutput(std::string& sha) = 0;
  // The remaining steps and the decoder; the RGBA pixels' SHA-256.
  virtual Status Finish(std::string& sha) = 0;
};

// The node with the configured models on it.
class Server {
 public:
  Server(const config::NodeConfig& config, const config::RuntimeRoles& roles,
         const ServingOptions& options, std::FILE* log);
  Server(const Server&) = delete;
  Server& operator=(const Server&) = delete;
  Server(Server&&) = delete;
  Server& operator=(Server&&) = delete;
  ~Server();

  // Registers every configured model (above), and with `snapshot` enables
  // a lazy pinned copy of initialized LLM state (snapshot()), for
  // a check that saves a state and puts it back. Once.
  Status Start(bool snapshot);
  void* snapshot() const { return snapshot_; }
  Status SaveSnapshot(Llm& model);
  Status RestoreSnapshot(Llm& model);
  // Every model's memory released and the node torn down. Once; the
  // destructor stops the node's threads if it never ran.
  Status TearDown();

  Served* Find(std::string_view name);
  std::span<const std::unique_ptr<Served>> models() const { return models_; }
  Served* resident() const { return resident_; }

  // Makes `m` the resident model (above); `parts` gets a swap's parts, or
  // with nothing resident the first load's (evict and restore 0).
  // `spill_state` false keeps the outgoing LLM's state resident (0
  // context); by default it is spilled when the LLM holds a conversation.
  Status Activate(Served& m, SwapParts& parts, std::optional<bool> spill_state = {});
  // After the incoming model's first output, off the swap's path: waits
  // until the backing no load took is released, and records that time and
  // the handoff's counts in `parts`.
  Status FinishSwap(SwapParts& parts);
  // `body` as one request of `m` (D-093): its closure leased once, every
  // job of `body` a step under it.
  Status InRequest(Served& m, const std::function<Status()>& body);
  // A cooperative cohort retains one stream lease between completed units.
  // The selected branch set can change only at such a boundary. The caller
  // ends this lease before changing models or freeing the cohort owner.
  Status SelectRequestBranches(Llm& model, std::span<Llm::Branch* const> active);
  Status EndRequestBranches(Llm& model);
  struct ReferenceRetirement {
    Status result;
    bool references_retired = false;
  };
  // After completed prompt/generation settlement, with the complete active
  // union selected. A queued no-op job records and waits for a stream fence.
  // Success of that explicit fence proves reference retirement; an ordinary
  // error from another operation, or cohort health, does not. End the stream
  // request only for the last owner; peers keep their selected state leases.
  ReferenceRetirement RetireRequestBranches(Llm& model, bool last_owner);
  // Every eviction's backing no load took released (after a swap).
  Status WaitReleased();

  // The node's one reclaim order (memory/reclaim.h; D-055 as amended
  // 2026-10-02; docs/retention-policy.md#victims-spill-and-exhaustion):
  // everything that can give memory back, whoever holds it (every model's
  // plans and graphs, and with `states` the resident model's idle
  // conversations' state, spilled), the kinds cheapest to restore a byte
  // first by their measured costs, least recently used within a kind,
  // until `needed` bytes are freed. On the driver's thread between
  // completed units (or from a charge inside a step, which it spares); the
  // freed heap returned to the system. Returns what it freed; `why` names
  // the occasion in its log line (counts and bytes only, D-014).
  // `running`: the model about to run (the resident one by default),
  // whose in-use floor is never taken (memory::ProtectFloor: its most
  // recently used plans up to its plan_floor_bytes, with their graphs) and
  // whose other plans and graphs go last within their kinds. All of it or
  // nothing: when the order cannot free `needed`, nothing is taken (the
  // caller waits or refuses), and no more than `needed` is taken
  // (candidates are whole). With `below_kind` (an optional charge of that
  // kind, a graph's capture), only what costs less to restore than it.
  // `partial` (pressure from outside): whatever part of it the order has.
  std::uint64_t Reclaim(std::uint64_t needed, bool states, std::string_view why,
                        const Served* running = nullptr,
                        std::optional<memory::ReclaimKind> below_kind = std::nullopt,
                        bool partial = false);
  // Between units and when idle (the chat route's driver): spilled
  // conversations past `[memory] retention_hours` deleted, and past
  // `spill_budget_gib` the least recently used deleted first; and when
  // memory from outside presses (PressureTrim: MemAvailable under its low
  // mark, or a full memory stall), one reclaim of what would restore its
  // target headroom, spaced by a growing back-off while it persists.
  void Maintain();
  // What a swap to `m` pages in beyond what it evicts and the budget's free
  // room: reclaimed before the swap (its plans and graphs; the outgoing
  // state is spilled by the swap itself).
  Status MakeRoomForSwap(Served& m, std::span<const catalog::ExtentId> out);
  // After a swap from `out` to `in` failed partway: what of `in` came in
  // (its weights and state) evicted again and `out` loaded back whole, so
  // `out` stays resident and usable; an error when that could not be done
  // either (no model is then resident: the next activation loads one whole).
  Status UndoSwap(Served& out, Served& in);
  // Evicts a model's resident weights and conversation state (state
  // written back), never the shared workspace or its own pinned memory.
  Status EvictPaged(Served& m);
  // Whether the node can go on after a failed activation (its scheduler has
  // not faulted): the failure is then the request's alone (D-102, recovery
  // first); otherwise the service stops for its supervisor's restart.
  bool NodeHealthy();
  // Spilled conversations deleted, the least recently used first, until
  // what is spilled and `extra` more fit `[memory] spill_budget_gib`; false
  // if they still do not.
  bool KeepWithinSpillBudget(std::uint64_t extra);
  // What every model's spilled conversations hold on disk now.
  std::uint64_t SpilledBytes();

  engine::PagedNode& node() { return node_; }
  MemorySampler& memory() { return memory_; }
  bool handoff() const { return handoff_; }
  void set_handoff(bool on) { handoff_ = on; }
  std::uint64_t budget() const { return budget_; }
  std::uint64_t fixed_bytes() const { return fixed_; }
  // The most any model's chunk builds on the host (Served::host_input_bytes;
  // one model runs at a time), which the start's memory guard counts beside
  // its margin.
  std::uint64_t host_input_bytes() const { return host_inputs_; }
  // The most one step of any model holds of plans at once (Served::
  // plan_floor_bytes): what the start's guard set apart beside the budget.
  std::uint64_t plan_floor_bytes() const { return plans_; }
  std::uint64_t workspace_bytes() const { return workspace_; }  // the shared activations and pool

 private:
  Status Make(const config::ModelEntry& entry, int index, std::string_view architecture);
  void Log(std::string_view text);

  const config::NodeConfig& config_;
  const config::RuntimeRoles& roles_;
  const ServingOptions& options_;
  std::FILE* log_;
  MemorySampler memory_;
  ResidentTimes times_;
  engine::PagedNode node_;
  std::vector<std::unique_ptr<Served>> models_;
  Served* resident_ = nullptr;
  bool handoff_ = true;
  bool started_ = false;
  bool torn_down_ = false;
  std::uint64_t budget_ = 0;
  std::uint64_t fixed_ = 0;
  std::uint64_t host_inputs_ = 0;
  std::uint64_t plans_ = 0;  // plan_floor_bytes()
  std::uint64_t workspace_ = 0;
  void* snapshot_ = nullptr;
  std::uint64_t snapshot_capacity_ = 0;
  bool snapshot_enabled_ = false;
  bool snapshot_unproven_ = false;
  Llm* snapshot_model_ = nullptr;
  scheduler::SchedulerStats swap_before_;  // the counters at the last Activate
  // Models whose conversation state was spilled by a swap out (and is
  // restored when they come back).
  std::vector<const Served*> spilled_;
  // The reclaim order's state (Reclaim, Maintain).
  bool reclaiming_ = false;
  std::uint64_t reclaims_short_ = 0;  // reclaims that took nothing: too little to take
  Clock::duration retention_;
  std::uint64_t spill_budget_ = 0;
  Clock::time_point next_maintenance_;
  PressureTrim pressure_;  // when pressure from outside has it trim
};

// The commands (commands.h), on a started server; the process's exit
// status comes from their result.
Status RunChat(Server& server, const ChatOptions& options, const ServingOptions& serving,
               std::FILE* out);
Status RunSwapTable(Server& server, const SwapTableOptions& options, const ServingOptions& serving,
                    std::FILE* out);

// The service with models configured (serve_api.cc; D-097): registers
// them, serves the loopback chat route until SIGTERM or SIGINT (which the
// caller has blocked in every thread), and tears the node down. Returns
// the process's exit status (runtime.h).
int RunService(const config::NodeConfig& config, const config::RuntimeRoles& roles, std::FILE* log);

}  // namespace jitllm::runtime

#endif  // JITLLM_RUNTIME_SERVING_H_
