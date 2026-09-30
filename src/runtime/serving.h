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
#include "runtime/commands.h"
#include "runtime/prefill.h"
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
  virtual Status Register() = 0;
  virtual Status Bind() = 0;
  virtual std::vector<catalog::ExtentId> weights() const = 0;
  virtual std::vector<catalog::ExtentId> state() const { return {}; }
  virtual const catalog::Closure& everything() const = 0;
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
 public:
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
  const std::vector<std::int32_t>& history() const { return history_; }
  // The state zeroed and the conversation empty.
  Status Clear();
  // The conversation dropped without a job: the state's contents are no
  // longer wanted (a swap out does not spill them), and the next use clears
  // it first.
  void Forget() {
    turn_checkpoints_.clear();
    history_.clear();
    needs_clear_ = true;
  }
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
  std::size_t turn_checkpoints() const { return turn_checkpoints_.size(); }
  std::uint64_t turn_checkpoint_bytes() const;
  // Greedy generation from `last` (a whole prefill's logits): the first
  // token is its argmax, the rest from decode steps.
  Status Generate(const std::vector<float>& last, const GenerateOptions& options, Generation& out);
  // The whole conversation state (the target's and the drafter's, and the
  // host's speculation cursor) saved to or put back from pinned host
  // memory the caller keeps (state_snapshot_bytes() long), by one job; and
  // the history with it.
  std::uint64_t state_snapshot_bytes() const;
  Status SaveState(void* host);
  Status RestoreState(void* host);
  void InvalidateStateSnapshot() { saved_valid_ = false; }

  std::vector<catalog::ExtentId> state() const override = 0;

 protected:
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

  // During Generate: whether it samples, the token for a row's logits at
  // a position in the conversation (greedy: the argmax), and whether the
  // target keeps a draft there (greedy: the argmax equals it); if not,
  // `next` is the token in its place.
  bool sampling() const { return sampling_.has_value(); }
  std::expected<std::int32_t, std::string> Choose(std::span<const float> row,
                                                  std::uint64_t position);
  std::expected<bool, std::string> Keep(std::span<const float> row, std::int32_t draft,
                                        std::uint64_t position, std::int32_t& next);
  // Finds the reasoning markers in the vocabulary (after the tokenizer).
  void FindThinkTokens();

  engine::PagedNode* node_ = nullptr;
  std::uint32_t max_rows_ = 0;  // the prefill chunk (runtime/prefill.h), set at construction
  std::optional<std::uint32_t> configured_rows_;  // the configuration's prefill_chunk
  std::uint32_t context_ = config::kDefaultContext;
  bool speculate_ = false;
  bool bos_ = false;  // EncodeText puts BOS first
  std::unique_ptr<tokenizer::Tokenizer> tokenizer_;
  const chat::Template* template_ = nullptr;
  std::vector<std::int32_t> stops_;
  std::vector<std::int32_t> history_;
  // A chunk or step failed after it may have run: the history is unknown,
  // and the state is cleared before the next use.
  bool needs_clear_ = false;
  std::vector<std::int32_t> saved_history_;
  std::uint32_t saved_cursor_ = 0;
  std::vector<engine::LiveState::Range> saved_ranges_;
  bool saved_valid_ = false;
  std::optional<std::int32_t> think_start_;
  std::optional<std::int32_t> think_end_;
  // The running generation's sampling, if it samples.
  std::optional<execution::SamplingParams> sampling_;
  std::uint64_t seed_ = 0;
  std::vector<execution::SamplingCandidate> scratch_;

 private:
  struct TurnCheckpoint {
    TurnBoundary boundary;
    engine::CheckpointFile file;
    std::vector<engine::LiveState::Range> footprint;
    std::uint32_t cursor = 0;
    execution::AdaptiveDepth decoding;
  };
  Status CaptureTurnCheckpoint(const PrefillGoOn& go_on, bool& stopped);
  Status ReusePrompt(std::span<const std::int32_t> tokens, std::uint32_t& reused, bool fresh,
                     const PrefillGoOn& go_on, bool& stopped);
  std::vector<TurnCheckpoint> turn_checkpoints_;
  Clock::time_point history_used_ = Clock::now();

 protected:
  std::filesystem::path checkpoint_directory_;
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
  // Every eviction's backing no load took released (after a swap).
  Status WaitReleased();

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
