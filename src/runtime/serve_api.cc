// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// The service with models configured (D-097; docs/runtime-serving.md#the-chat-route):
// the configured models registered on the node (serving.h), the chat
// route (api_server.h) over them, where [client] bind says (binding.h), and
// the runtime's signals watched as readiness (platform/event_loop.h) on the
// node's driver thread, which runs every request.

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <expected>
#include <filesystem>
#include <format>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include "base/check.h"
#include "base/report.h"
#include "base/work_pulse.h"
#include "chat/chat.h"
#include "config/node_config.h"
#include "platform/event_loop.h"
#include "platform/host_probe.h"
#include "platform/interfaces.h"
#include "platform/job.h"
#include "platform/sd_notify.h"
#include "platform/sockets.h"
#include "runtime/api.h"
#include "runtime/api_server.h"
#include "runtime/binding.h"
#include "runtime/cohort_capacity.h"
#include "runtime/cohort_schedule.h"
#include "runtime/commands.h"
#include "runtime/completion_tokens.h"
#include "runtime/hang_ladder.h"
#include "runtime/intake_limits.h"
#include "runtime/prefill.h"
#include "runtime/runtime.h"
#include "runtime/serving.h"
#include "runtime/watchdog.h"
#include "tokenizer/tokenizer.h"

namespace jitllm::runtime {
namespace {

// The least a capacity refusal has the reclaim order free: one 2 MiB
// extent of state.
constexpr std::uint64_t kExtentBytes = std::uint64_t{2} << 20U;

// D-105 at the process's end. A hang's last resort writes the conversation
// records already queued for at most this long (best effort, bounded:
// recovery comes first). A graceful stop keeps conversations for at most
// kPersistLongest, giving up sooner only when nothing moves (Server::
// Persist), and asks the service manager each second for kStopExtension
// more.
constexpr auto kLastResortKeep = std::chrono::seconds(10);
constexpr auto kPersistLongest = std::chrono::hours(1);
constexpr std::chrono::microseconds kStopExtension = std::chrono::seconds(15);
// A start step's extension of the start timeout: jitllm.service's
// TimeoutStartSec.
constexpr std::chrono::microseconds kStartExtension = std::chrono::minutes(5);
// How often the driver asks the hang ladder whether a cancellation drained
// (NodeBackend::AwaitDrained).
constexpr auto kDrainPoll = std::chrono::milliseconds(20);

void Say(std::FILE* log, std::string_view text) {
  const std::string line = std::format("jitllm-runtime: {}\n", base::Printable(text));
  (void)std::fwrite(line.data(), 1, line.size(), log);
  (void)std::fflush(log);
}

// The node's waits under the hang ladder (D-102): each wait counts as work
// under way while it lasts, feeds the ladder the node's progress as it
// polls, and cancels its request at rung 1 when it began before the hang
// was confirmed (a later wait is new work). Rung 3 is the ladder's last
// resort, run by whichever thread entered it: the wait itself never gives
// up.
class LadderPatience final : public engine::Patience {
 public:
  explicit LadderPatience(HangLadder& ladder) : ladder_(ladder) {}
  void Begin() override { ladder_.BeginWait(); }
  void End() override { ladder_.EndWait(); }
  engine::WaitVerdict Check(const engine::WaitState& wait, std::uint64_t progress) override {
    const auto now = Clock::now();
    ladder_.Activity(progress, now);
    (void)ladder_.Check(now);
    if (ladder_.rung() == HangLadder::Rung::kCancel && !wait.cancelled &&
        wait.began <= ladder_.cancelled_at()) {
      return engine::WaitVerdict::kCancel;
    }
    return engine::WaitVerdict::kWait;
  }

 private:
  HangLadder& ladder_;
};

api::Error Failure(int status, std::string message, std::string code = {}, std::string param = {}) {
  return api::Error{.status = status,
                    .type = status >= 500 ? "server_error" : "invalid_request_error",
                    .message = std::move(message),
                    .param = std::move(param),
                    .code = std::move(code)};
}

std::optional<api::Error> FundHistory(MemoryCharge& charge, RequestMemory& memory,
                                      std::uint64_t held, bool granted) {
  if (held <= charge.bytes()) {
    return std::nullopt;
  }
  const std::uint64_t more = held - charge.bytes();
  if (charge.Add(memory, more, granted)) {
    return std::nullopt;
  }
  // The completed step already grew its vector. Keep it accounted until
  // retirement, and stop before another step. Callbacks cannot reclaim
  // native state in the middle of work.
  charge.Force(memory, more);
  return api::MemoryRefusal(memory, held, "the request token history");
}

std::uint64_t RandomSeed() {
  std::uint64_t seed = 0;
  if (!platform::FillRandom(std::as_writable_bytes(std::span(&seed, 1)))) {
    seed = static_cast<std::uint64_t>(Clock::now().time_since_epoch().count());
  }
  return seed;
}

// A request's sampling: each field it sent, else the model's default (its
// settings: the checkpoint's own, an override, or OpenAI's; D-103). None
// at temperature 0: greedy.
std::optional<execution::SamplingParams> SamplingOf(const api::ChatRequest& request,
                                                    const Llm& model) {
  const api::SamplingDefaults s = request.Sampling(model.sampling_defaults());
  if (s.temperature <= 0) {
    return std::nullopt;
  }
  return execution::SamplingParams{.temperature = static_cast<float>(s.temperature),
                                   .top_k = s.top_k,
                                   .top_p = static_cast<float>(s.top_p),
                                   .min_p = static_cast<float>(s.min_p)};
}

struct ChatPrompt {
  std::shared_ptr<const api::PromptTokens> token_storage;
  const std::vector<std::int32_t>& tokens() const { return token_storage->values(); }
  std::uint32_t stable_boundary = 0;
  std::uint32_t max_tokens = 0;
  bool reasoning = false;
};

// The context's refusal, for a conversation that cannot fit before it is
// rendered or tokenized.
api::Error ContextExceeded(std::uint32_t usable) {
  return Failure(400,
                 std::format("This model's maximum context length is {} tokens. However, your "
                             "messages resulted in more than {} tokens. Please reduce the length "
                             "of the messages.",
                             usable, usable),
                 "context_length_exceeded", "messages");
}

// Renders and counts a chat request on the driver. Its bounds are the
// model's (D-102): a conversation whose text alone is longer than a prompt
// that fits the usable context could render to (Llm::render_bytes) is
// refused before it is copied or rendered; the rendering is bounded by its
// messages' bytes (Llm::RenderChargeFor) and its tokens by the usable
// context. Each step is charged to the request memory (`memory`) before
// it runs: the conversation's copy of the messages, the rendering, the
// tokenization. The exchange is asked as it goes, so a client that leaves,
// the runtime stopping or a configured deadline (set before rendering) end
// a long rendering (it is not moved off the driver: other requests wait
// for it, as for any unit).
std::expected<ChatPrompt, api::Error> PrepareChat(Llm& model, const api::ChatRequest& request,
                                                  api::Exchange& exchange, RequestMemory& memory) {
  const std::uint32_t usable = model.usable_context();
  std::uint64_t text = 0;
  for (const api::Message& message : request.messages) {
    text += message.content.size() + (message.reasoning ? message.reasoning->size() : 0);
  }
  if (text > model.render_bytes()) {
    return std::unexpected(ContextExceeded(usable));
  }
  const RenderCharge render = model.RenderChargeFor(text, request.messages.size());
  const std::uint64_t copied = text + (request.messages.size() * sizeof(chat::Message));
  MemoryCharge charge;
  if (!charge.Add(memory, copied + render.bytes)) {
    return std::unexpected(
        api::MemoryRefusal(memory, copied + render.bytes, "rendering the conversation"));
  }
  chat::Conversation conversation;
  conversation.messages.reserve(request.messages.size());
  for (const api::Message& message : request.messages) {
    chat::Role role = chat::Role::kUser;
    if (message.role == api::Role::kSystem) {
      role = chat::Role::kSystem;
    } else if (message.role == api::Role::kAssistant) {
      role = chat::Role::kAssistant;
    }
    conversation.messages.push_back({.role = role,
                                     .content = message.content,
                                     .reasoning_content = message.reasoning,
                                     .tool_calls = {}});
  }
  model.Defaults(conversation);
  conversation.max_render_bytes = static_cast<std::size_t>(render.output);
  conversation.max_live_bytes = static_cast<std::size_t>(render.live);
  ChatPrompt result;
  // Each check beats the driver's pulse (a long rendering or tokenization
  // is progress, not a hang), and stops it when a confirmed hang's rung 1
  // asks the CPU work under way to stop (D-102).
  base::WorkPulse* pulse = base::ThreadPulse();
  const std::uint64_t cancels = pulse != nullptr ? pulse->cancels() : 0;
  const std::function<bool()> cancelled = [&exchange] {
    return !base::Pulse() || !exchange.Continue();
  };
  ChatRenderFailure failure = ChatRenderFailure::kOther;
  std::uint64_t needed = 0;
  auto rendered = model.RenderChat(
      conversation, &result.stable_boundary,
      {.max_tokens = usable, .cancelled = &cancelled, .memory = &memory, .memory_needed = &needed},
      &failure);
  if (!rendered && failure == ChatRenderFailure::kTooLong) {
    return std::unexpected(ContextExceeded(usable));
  }
  if (!rendered && failure == ChatRenderFailure::kMemory) {
    return std::unexpected(api::MemoryRefusal(memory, needed, "tokenizing the conversation"));
  }
  if (!rendered && failure == ChatRenderFailure::kUnbroken) {
    // Text the tokenizer cannot cut into windows (no line break or word
    // boundary) and cannot hold whole: the prompt's shape, refused as the
    // context is.
    return std::unexpected(Failure(
        400,
        std::format("The messages hold {} bytes of text without a line break or a space between "
                    "words; this model tokenizes at most {} bytes without one. Please break up "
                    "the text.",
                    needed, memory.capacity() / tokenizer::kEncodeBytesPerWindowByte),
        "context_length_exceeded", "messages"));
  }
  if (!rendered && failure == ChatRenderFailure::kCancelled && pulse != nullptr &&
      pulse->cancels() != cancels) {
    // Stopped by the hang ladder (rung 1): the work stops here, so the
    // next request's is not asked to stop too.
    pulse->Clear();
    return std::unexpected(Failure(
        503, "the conversation's rendering made no progress and was cancelled: retry the request",
        "backend_hung"));
  }
  if (!rendered && failure == ChatRenderFailure::kCancelled) {
    // The exchange answers for why (the client gone, the runtime stopping).
    return std::unexpected(
        Failure(503, "the request ended while its conversation was rendered", "request_ended"));
  }
  if (!rendered) {
    return std::unexpected(
        Failure(400, "the conversation cannot be rendered: " + rendered.error(), {}, "messages"));
  }
  auto held = api::PromptTokens::Hold(std::move(*rendered), memory);
  if (!held) {
    return std::unexpected(held.error());
  }
  result.token_storage = std::move(*held);
  const auto prompt = static_cast<std::uint32_t>(result.tokens().size());
  if (prompt >= usable) {
    return std::unexpected(
        Failure(400,
                std::format("This model's maximum context length is {} tokens. However, your "
                            "messages resulted in {} tokens. Please reduce the length of the "
                            "messages.",
                            usable, prompt),
                "context_length_exceeded", "messages"));
  }
  result.max_tokens = request.max_tokens.value_or(usable - prompt);
  if (std::uint64_t{prompt} + result.max_tokens > usable) {
    return std::unexpected(Failure(
        400,
        std::format("This model's maximum context length is {} tokens. However, you requested "
                    "{} tokens ({} in the messages, {} in the completion). Please reduce the "
                    "length of the messages or completion.",
                    usable, std::uint64_t{prompt} + result.max_tokens, prompt, result.max_tokens),
        "context_length_exceeded", "messages"));
  }
  if (model.think_start() && model.think_end()) {
    const auto marker = std::ranges::find_if(
        result.tokens().rbegin(), result.tokens().rend(), [&](std::int32_t token) {
          return token == *model.think_start() || token == *model.think_end();
        });
    result.reasoning = marker != result.tokens().rend() && *marker == *model.think_start();
  }
  return result;
}

// The exact scalar chat decoder and reasoning-boundary rules, with stable
// ownership so an options callback can borrow it across completed units.
class ChatOutput final {
 public:
  ChatOutput(Llm& model, api::Exchange& exchange, bool reasoning)
      : model_(model), exchange_(exchange), decoder_(model.tokenizer(), {}), thinking_(reasoning) {}
  bool Push(std::span<const std::int32_t> fresh) {
    std::string piece;
    bool go = true;
    bool said = false;
    const auto send = [&]() {
      if (!piece.empty()) {
        go = (thinking_ ? exchange_.Reasoning(piece) : exchange_.Content(piece)) && go;
        any_content_ = any_content_ || !thinking_;
        said = true;
        piece.clear();
      }
    };
    for (const std::int32_t token : fresh) {
      if (thinking_ && model_.think_end() && token == *model_.think_end()) {
        decoder_.Finish(piece);
        if (piece.empty()) {
          go = exchange_.Reasoning({}) && go;
        }
        send();
        thinking_ = false;
        continue;
      }
      if (!thinking_ && !any_content_ && piece.empty() && model_.think_start() &&
          token == *model_.think_start()) {
        thinking_ = true;
        continue;
      }
      std::ignore = decoder_.Push(token, piece);
    }
    send();
    return said ? go : exchange_.Continue();
  }
  void Finish() {
    std::string rest;
    decoder_.Finish(rest);
    if (!rest.empty()) {
      (void)(thinking_ ? exchange_.Reasoning(rest) : exchange_.Content(rest));
    }
  }

  // Where the output stands (a partial character held back, inside the
  // reasoning or not): a request that yields its place continues from it.
  struct State {
    tokenizer::StreamDecoder decoder;
    bool thinking = false;
    bool any_content = false;
  };
  State Save() const {
    return {.decoder = decoder_, .thinking = thinking_, .any_content = any_content_};
  }
  void Restore(const State& state) {
    decoder_ = state.decoder;
    thinking_ = state.thinking;
    any_content_ = state.any_content;
  }

 private:
  Llm& model_;
  api::Exchange& exchange_;
  tokenizer::StreamDecoder decoder_;
  bool thinking_ = false;
  bool any_content_ = false;
};

// A chat request that yielded its place (api::Exchange::Yielding): what it
// continues from, on either path. Its state is kept as a finished turn's
// (resident, or spilled when memory needs it), so taking it up again
// reuses it (Llm::ReusablePrefix) and prefills only what it lost.
struct ChatResume final : api::Yielded {
  ChatResume(const ChatResume&) = delete;
  ChatResume& operator=(const ChatResume&) = delete;
  ChatResume(ChatResume&&) = delete;
  ChatResume& operator=(ChatResume&&) = delete;
  ChatResume(ChatPrompt prompt, Generation so_far, const GenerateOptions& options,
             ChatOutput::State out, std::uint32_t cached, MemoryCharge charge,
             Llm::Branch* held = nullptr, std::shared_ptr<api::LiteralOutput> literal = nullptr)
      : rendered(std::move(prompt)),
        generation(std::move(so_far)),
        sampling(options.sampling),
        seed(options.seed),
        output(std::move(out)),
        reused(cached),
        token_charge(std::move(charge)),
        held_branch(held),
        literal(std::move(literal)) {
    if (held_branch != nullptr) {
      held_branch->HoldContinuation();
    }
    // The prompt and every generated token but the last (the anchor:
    // reported, and not yet in the state).
    history.reserve(rendered.tokens().size() +
                    (generation.tokens.empty() ? 0 : generation.tokens.size() - 1));
    history.insert(history.end(), rendered.tokens().begin(), rendered.tokens().end());
    if (!generation.tokens.empty()) {
      history.insert(history.end(), generation.tokens.begin(), generation.tokens.end() - 1);
    }
    generation.cancelled = false;
  }
  ~ChatResume() override {
    if (held_branch != nullptr) {
      held_branch->ReleaseContinuation();
    }
  }
  ChatPrompt rendered;
  Generation generation;  // so far
  std::vector<std::int32_t> history;
  std::optional<execution::SamplingParams> sampling;
  std::uint64_t seed = 0;
  ChatOutput::State output;
  std::uint32_t reused = 0;  // the first turn's cached tokens, for its usage
  MemoryCharge token_charge;
  Llm::Branch* held_branch = nullptr;
  std::shared_ptr<api::LiteralOutput> literal;
};

std::expected<std::shared_ptr<ChatResume>, api::Error> KeepContinuation(
    const ChatPrompt& prompt, const Generation& generation, const GenerateOptions& options,
    ChatOutput::State output, std::uint32_t reused, RequestMemory& memory,
    Llm::Branch* branch = nullptr, std::shared_ptr<api::LiteralOutput> literal = nullptr) {
  // Charge both copies before constructing them. The immutable prompt
  // itself is shared with the continuation and already owns its charge.
  const std::uint64_t ids = prompt.tokens().size() + generation.tokens.size() +
                            (generation.tokens.empty() ? 0 : generation.tokens.size() - 1);
  MemoryCharge charge;
  const std::uint64_t bytes = sizeof(ChatResume) + ids * sizeof(std::int32_t);
  if (!charge.Add(memory, bytes)) {
    return std::unexpected(api::MemoryRefusal(memory, bytes, "the request continuation"));
  }
  return std::make_shared<ChatResume>(prompt, generation, options, std::move(output), reused,
                                      std::move(charge), branch, std::move(literal));
}

// A serial request's state-capacity policy on its model for the request's
// duration (Llm::set_capacity_reclaim); cleared when it ends.
class ScopedReclaim final {
 public:
  ScopedReclaim(Llm& model, Llm::CapacityReclaim reclaim) : model_(model) {
    model_.set_capacity_reclaim(std::move(reclaim));
  }
  ~ScopedReclaim() { model_.set_capacity_reclaim({}); }
  ScopedReclaim(const ScopedReclaim&) = delete;
  ScopedReclaim& operator=(const ScopedReclaim&) = delete;
  ScopedReclaim(ScopedReclaim&&) = delete;
  ScopedReclaim& operator=(ScopedReclaim&&) = delete;

 private:
  Llm& model_;
};

// A serial request's work under its lease, with the refusal that ended it
// when only the state's capacity did (Branch::capacity_refused): the lease
// then ends normally, and the request alone fails (as a cohort member that
// cannot fit alone). Any other failure is returned for the node to stop.
class SerialCapacity final {
 public:
  explicit SerialCapacity(Llm& model) : model_(model) {}
  Status operator()(const Status& status, std::string_view what) {
    if (!status && model_.default_branch().capacity_refused()) {
      refusal_ = std::format("{}{}", what, status.error());
      return {};
    }
    return status;
  }
  const std::optional<std::string>& refusal() const { return refusal_; }

 private:
  Llm& model_;
  std::optional<std::string> refusal_;
};

// The chat route's requests as turns on the node: one model resident, a
// swap when another is asked for, the conversation state reused when the
// rendered request extends it, and each request one lease (D-093).
class NodeBackend final : public api::Backend, public api::CooperativeBackend {
  class ChatWork final : public api::CooperativeBackend::Work {
   public:
    enum class Stage { kPrompt, kGeneration, kDone };
    ChatWork(NodeBackend& owner, Llm& model, Llm::Branch& branch, std::size_t slot,
             const api::ChatRequest& request, api::Exchange& exchange, ChatPrompt rendered,
             Floors floors, std::uint64_t swap_bytes,
             std::shared_ptr<api::LiteralOutput> literal = nullptr)
        : owner(owner),
          model(model),
          branch(branch),
          slot(slot),
          request(request),
          exchange(exchange),
          rendered(std::move(rendered)),
          floors(floors),
          swap_bytes(swap_bytes),
          output(model, exchange, this->rendered.reasoning),
          literal(std::move(literal)) {
      options.max_tokens = this->rendered.max_tokens;
      options.stop = true;
      if (this->literal == nullptr) options.extra_stops = model.ChatStops();
      options.sampling = SamplingOf(request, model);
      if (options.sampling) {
        options.seed = request.seed.value_or(RandomSeed());
      }
      options.on_tokens = [this](std::span<const std::int32_t> fresh) {
        if (!FundTokens(true)) {
          return false;
        }
        if (this->literal == nullptr) {
          return output.Push(fresh);
        }
        bool said = false;
        const bool sent = this->literal->Push(fresh, [this, &said](std::string_view piece) {
          said = true;
          return this->exchange.Content(piece);
        });
        return sent && (said || this->exchange.Continue());
      };
      if (this->literal != nullptr && this->literal->score_generation()) {
        options.on_logits = [this](std::int32_t id, std::span<const float> row) {
          return this->literal->GeneratedRow(id, row) && this->exchange.Continue();
        };
      }
    }
    ~ChatWork() override {
      base::Check(retired, "a cooperative chat owner was freed before retirement");
    }
    ChatWork(const ChatWork&) = delete;
    ChatWork& operator=(const ChatWork&) = delete;
    ChatWork(ChatWork&&) = delete;
    ChatWork& operator=(ChatWork&&) = delete;
    bool terminal() const override {
      return cancelled || yielding || error.has_value() || stage == Stage::kDone ||
             (generation_session != nullptr && generation_session->done());
    }
    void Cancel() override {
      cancelled = true;
      yielding = false;  // cancelled, it ends; it does not continue
    }
    // Yields only a generation under way that is not set aside or waiting
    // for capacity (its continuation is its generation so far).
    bool Yield() override {
      if (generation_session == nullptr || generation_session->done() ||
          generation.tokens.empty() || wait != CapacityWait::kNone || resume || error ||
          cancelled) {
        return false;
      }
      yielding = true;
      return true;
    }
    bool YieldForSwitch() override {
      if (terminal()) {
        return false;
      }
      if (generation.tokens.empty() && prompt_session != nullptr &&
          prompt_session->remaining_rows() == 0) {
        // The final prefill head may already exist while its checkpoint
        // is still owed. Complete that boundary and choose its anchor
        // before pausing; otherwise cancelling would discard the head
        // and an equal-history resume could recompute a different one.
        return false;
      }
      yielding = true;
      model_paused = true;
      return true;
    }

    bool FundTokens(bool granted) {
      const std::uint64_t held =
          (generation.tokens.capacity() + resume_tokens.capacity()) * sizeof(std::int32_t);
      if (auto refused = FundHistory(token_charge, *owner.memory_, held, granted)) {
        error = std::move(refused);
        return false;
      }
      return true;
    }

    NodeBackend& owner;
    Llm& model;
    Llm::Branch& branch;
    const std::size_t slot;
    const api::ChatRequest& request;
    api::Exchange& exchange;
    const ChatPrompt rendered;
    const Floors floors;
    const std::uint64_t swap_bytes;
    ChatOutput output;
    std::shared_ptr<api::LiteralOutput> literal;
    GenerateOptions options;
    Generation generation;
    std::unique_ptr<Llm::PromptSession> prompt_session;
    std::unique_ptr<Llm::GenerationSession> generation_session;
    std::optional<api::Error> error;
    Stage stage = Stage::kPrompt;
    std::uint32_t reused = 0;
    bool cancelled = false;
    bool yielding = false;      // yields its place (Yield): retired with a continuation
    bool model_paused = false;  // prompt or generation paused for another model
    bool native_touched = false;
    bool retired = false;
    // State capacity (cohort_capacity.h): admission order, whether it
    // waits, the refusal it fails with if it cannot fit alone, and after a
    // preemption mid-generation, the tokens its state is rebuilt from (the
    // prompt and every generated token but the unprocessed anchor).
    std::uint64_t admitted = 0;
    // The schedule's counters (cohort_schedule.h): while it prefills, the
    // other prompt units since its own last; while it generates, the prompt
    // units since its last wave.
    std::uint32_t passed = 0;
    std::uint32_t waited = 0;
    CapacityWait wait = CapacityWait::kNone;
    std::string refusal;
    bool resume = false;
    bool continued = false;       // original cached-token usage survives a prompt pause too
    bool switch_restore = false;  // restore generating peers before resuming their wave
    std::vector<std::int32_t> resume_tokens;
    MemoryCharge token_charge;
  };
  static_assert(kCohortSlots == Llm::kMaxBranches);

 public:
  // `memory`: the request memory (intake_limits.h), which the chat route
  // shares; renderings, tokenizations and literal scores are charged to it.
  NodeBackend(Server& server, const config::NodeConfig& config,
              std::shared_ptr<RequestMemory> memory, std::FILE* log, HangLadder* ladder = nullptr)
      : server_(server), config_(config), memory_(std::move(memory)), log_(log), ladder_(ladder) {}

  // Chat and literal requests share funded independent slots. Unsupported
  // models retain their ordinary entry points.
  api::CooperativeBackend* cooperative() override { return this; }

  bool Supports(const api::CooperativeBackend::Request& request) const override {
    Served* model = server_.Find(request.options.model);
    return model != nullptr && model->llm() &&
           static_cast<Llm&>(*model).supports_generation_waves();
  }

  std::expected<std::unique_ptr<api::CooperativeBackend::Work>, api::Error> Start(
      const api::CooperativeBackend::Request& request, api::Exchange& exchange) override {
    if (!Supports(request)) {
      return std::unique_ptr<api::CooperativeBackend::Work>{};
    }
    auto& model = static_cast<Llm&>(*server_.Find(request.options.model));
    if (cohort_model_ != nullptr && cohort_model_ != &model) {
      return std::unique_ptr<api::CooperativeBackend::Work>{};
    }
    const auto claimed =
        std::ranges::count_if(cohort_, [](const ChatWork* frame) { return frame != nullptr; });
    if (static_cast<std::size_t>(claimed) >= model.generation_wave_capacity() ||
        std::ranges::find(cohort_, nullptr) == cohort_.end()) {
      return std::unique_ptr<api::CooperativeBackend::Work>{};
    }
    // Members waiting for state capacity come first (FIFO): a new prompt
    // would compete for the capacity they wait for.
    if (!AdmissionOpen(Snapshot())) {
      return std::unique_ptr<api::CooperativeBackend::Work>{};
    }
    // A request already found no memory for a slot waits for a peer to
    // retire before it is looked at again (below).
    if (memory_wait_.Blocked(static_cast<std::size_t>(claimed))) {
      return std::unique_ptr<api::CooperativeBackend::Work>{};
    }
    // A request that yielded its place continues from its generation so far
    // (ChatResume); any other is rendered now.
    const auto* resume = static_cast<const ChatResume*>(request.resume);
    std::shared_ptr<api::LiteralOutput> literal = resume != nullptr ? resume->literal : nullptr;
    std::expected<ChatPrompt, api::Error> rendered;
    if (resume != nullptr) {
      rendered = resume->rendered;
    } else if (request.literal != nullptr) {
      auto prepared =
          api::PrepareLiteralPrompt(*request.literal, model.tokenizer(), model.usable_context(),
                                    memory_->capacity(), memory_.get());
      if (!prepared) {
        return std::unexpected(prepared.error());
      }
      auto output =
          api::LiteralOutput::Create(*request.literal, *prepared, model.tokenizer(), *memory_);
      if (!output) {
        return std::unexpected(output.error());
      }
      literal = std::move(*output);
      auto held = api::PromptTokens::Hold(std::move(prepared->tokens), *memory_);
      if (!held) {
        return std::unexpected(held.error());
      }
      rendered = ChatPrompt{.token_storage = std::move(*held),
                            .max_tokens = request.options.max_tokens.value_or(16)};
    } else {
      rendered = PrepareChat(model, request.options, exchange, *memory_);
    }
    if (!rendered) {
      return std::unexpected(rendered.error());
    }
    const std::vector<std::int32_t>& wanted =
        resume != nullptr ? resume->history : rendered->tokens();
    // Choose only an unclaimed model-owned branch. Prefix matching is a
    // reuse preference, never conversation identity or retention policy:
    // the branch whose state this prompt can actually reuse the most of
    // (Llm::ReusablePrefix: its live history continued, or a turn
    // checkpoint inside the common prefix); with none, an empty branch,
    // then the least recently used, so a new conversation does not
    // discard an idle one's state while an empty branch is free.
    std::size_t slot = cohort_.size();
    std::size_t best = 0;
    bool best_empty = false;
    Clock::time_point best_used;
    for (std::size_t i = 0; i < model.branches() && i < cohort_.size(); ++i) {
      if (cohort_[i] != nullptr) {
        continue;
      }
      auto branch = model.branch(i);
      if (!branch) {
        return std::unexpected(Failure(500, "the model's conversation branch is unavailable"));
      }
      if (resume != nullptr && resume->held_branch != nullptr && *branch != resume->held_branch) {
        continue;  // logical identity survives the swap; prefix matching is insufficient
      }
      if ((*branch)->HeldByContinuation() &&
          (resume == nullptr || resume->held_branch != *branch)) {
        continue;
      }
      const std::size_t reusable = literal != nullptr ? 0 : model.ReusablePrefix(**branch, wanted);
      const bool empty = !(*branch)->HasRetainedState();
      const Clock::time_point used = model.LastUsed(**branch);
      const bool better = slot == cohort_.size() || reusable > best ||
                          (reusable == best &&
                           ((empty && !best_empty) || (empty == best_empty && used < best_used)));
      if (better) {
        slot = i;
        best = reusable;
        best_empty = empty;
        best_used = used;
      }
    }
    if (slot == cohort_.size()) {
      return std::unique_ptr<api::CooperativeBackend::Work>{};
    }
    auto branch = model.branch(slot);
    if (!branch) {
      return std::unexpected(Failure(500, "the model's conversation branch is unavailable"));
    }
    // Request slots follow memory (docs/runtime-serving.md#request-slots):
    // beside its peers, a request joins only where the budget holds its
    // prompt's state and what its peers' prompts have yet to take, free or
    // freed through the reclaim order (idle conversations spilled, stale
    // plans and graphs dropped). Otherwise it waits first in the queue until
    // a peer retires. A lone request always starts: the capacity policy
    // governs it as before.
    if (claimed != 0 && server_.resident() == &model) {
      std::uint64_t needed = PromptStateToCome(model, **branch, rendered->tokens().size());
      for (const ChatWork* peer : cohort_) {
        if (peer != nullptr && !peer->terminal() && peer->stage == ChatWork::Stage::kPrompt) {
          needed += PromptStateToCome(model, peer->branch, peer->rendered.tokens().size());
        }
      }
      if (needed != 0 && !server_.RoomFor(needed, "a request slot", *branch)) {
        memory_wait_.NoRoom();
        Say(log_, std::format("{}'s request waits for memory for a request slot: {:.1f} MiB of "
                              "prompt state to come for {} tokens beside {} active requests",
                              model.name(), static_cast<double>(needed) / (1U << 20U),
                              rendered->tokens().size(), claimed));
        return std::unique_ptr<api::CooperativeBackend::Work>{};
      }
    }
    Floors floors = model.floors();
    if (literal != nullptr && literal->score_prompt()) {
      floors.prefill = floors.decode;  // teacher forcing is one-row target work
    }
    const auto swap_bytes = SwapBytes(model);
    auto frame =
        std::make_unique<ChatWork>(*this, model, **branch, slot, request.options, exchange,
                                   std::move(*rendered), floors, swap_bytes, std::move(literal));
    if (resume != nullptr) {
      // Its state rebuilt to the history it yielded at (restored, reused or
      // prefilled), then its generation continues from the anchor
      // (BeginChatGeneration's resume): nothing is chosen or sent again.
      frame->resume = !resume->generation.tokens.empty();
      frame->continued = true;
      frame->switch_restore = resume->held_branch != nullptr;
      frame->resume_tokens = resume->history;
      frame->generation = resume->generation;
      frame->options.sampling = resume->sampling;
      frame->options.seed = resume->seed;
      frame->output.Restore(resume->output);
      frame->reused = resume->reused;
      if (!frame->FundTokens(false)) {
        frame->retired = true;  // no session or native work borrowed it
        return std::unexpected(*frame->error);
      }
    }
    auto prompt = WorkPrompt(*frame);
    if (!prompt) {
      frame->retired = true;  // no session or native work retained a reference
      return std::unexpected(Failure(500, "the prompt could not be admitted: " + prompt.error()));
    }
    frame->prompt_session = std::move(*prompt);
    if (!exchange.Admit(
            {.prompt_tokens = static_cast<std::uint32_t>(frame->rendered.tokens().size()),
             .max_tokens = frame->rendered.max_tokens,
             .swap_bytes = swap_bytes,
             .floors = floors})) {
      frame->Cancel();  // retired normally; no native unit has run
    }
    frame->admitted = ++admissions_;
    cohort_model_ = &model;
    cohort_[slot] = frame.get();
    return std::unique_ptr<api::CooperativeBackend::Work>(std::move(frame));
  }

  std::expected<api::CooperativeBackend::Unit, std::string> NextUnit(
      std::span<api::CooperativeBackend::Work* const> work) override {
    if (auto valid = CheckCohort(work); !valid) {
      return std::unexpected(valid.error());
    }
    selected_prompt_ = nullptr;
    selected_prefill_count_ = 0;
    selected_decode_.clear();
    selected_swap_ = server_.resident() != cohort_model_;
    if (selected_swap_) {
      return api::CooperativeBackend::Unit{
          .phase = Phase::kSwap,
          .expected_seconds =
              static_cast<double>(SwapBytes(*cohort_model_)) / kSwapFloorBytesPerSecond};
    }
    // The schedule (cohort_schedule.h): the prompt with the fewest tokens
    // left goes next, a long one passed over for a bounded number of units,
    // and a generating member waits for at most one prompt unit.
    std::vector<ScheduledMember> members;
    std::vector<ChatWork*> frames;
    bool paused = false;
    const bool restoring_generations = std::ranges::any_of(work, [](const auto* base) {
      const auto& frame = static_cast<const ChatWork&>(*base);
      return frame.switch_restore && frame.resume && frame.stage == ChatWork::Stage::kPrompt &&
             frame.wait == CapacityWait::kNone;
    });
    for (auto* base : work) {
      auto& frame = static_cast<ChatWork&>(*base);
      if (frame.wait != CapacityWait::kNone) {
        continue;  // waits for state capacity (cohort_capacity.h)
      }
      if (frame.stage == ChatWork::Stage::kGeneration) {
        if (restoring_generations) {
          continue;  // restore the old wave before any member decodes at a narrower width
        }
        // A stream whose client is behind waits at its completed step
        // (backpressure, D-102); the others go on.
        if (frame.exchange.Paused()) {
          paused = true;
          continue;
        }
        selected_decode_.push_back(&frame);
        members.push_back({.stage = ScheduledMember::Stage::kGeneration,
                           .admitted = frame.admitted,
                           .remaining = 0,
                           .passed = 0,
                           .waited = frame.waited,
                           .finishing = false});
        frames.push_back(&frame);
      } else if (frame.stage == ChatWork::Stage::kPrompt) {
        if (restoring_generations && !(frame.switch_restore && frame.resume)) {
          continue;
        }
        const std::uint32_t remaining = frame.prompt_session->remaining_rows();
        members.push_back(
            {.stage = ScheduledMember::Stage::kPrompt,
             .admitted = frame.admitted,
             .remaining = remaining,
             .passed = frame.passed,
             .waited = 0,
             .finishing = frame.prompt_session->started() && remaining <= kPromptFinishRows});
        frames.push_back(&frame);
      }
    }
    const ScheduleChoice choice = NextCohortUnit(members);
    ChatWork* prompt = choice.prompt ? frames[*choice.prompt] : nullptr;
    if (choice.decode) {
      double expected = 0;
      for (const ChatWork* frame : selected_decode_) {
        expected +=
            ExpectedSeconds(Phase::kDecode, frame->model.speculative() ? 4 : 1, frame->floors);
      }
      return api::CooperativeBackend::Unit{.phase = Phase::kDecode, .expected_seconds = expected};
    }
    selected_decode_.clear();
    if (prompt == nullptr && paused) {
      // Every runnable member waits for its client: the server waits for
      // one to read or leave.
      return api::CooperativeBackend::Unit{.phase = Phase::kPaused, .expected_seconds = 0};
    }
    if (prompt == nullptr) {
      return std::unexpected("an active chat cohort has no next completed unit");
    }
    std::array<Llm::PromptSession*, Llm::kMaxBranches> prompt_sessions{};
    for (std::size_t i = 0; i < frames.size(); ++i) {
      if (frames[i]->stage == ChatWork::Stage::kPrompt)
        prompt_sessions[i] = frames[i]->prompt_session.get();
    }
    const auto ready = Llm::PromptSession::SelectForWave(
        choice.decode, choice.prompt, std::span(prompt_sessions).first(frames.size()));
    base::Check(ready.has_value(), "a selected prompt has no readiness unit");
    prompt = frames[*ready];
    auto unit = prompt->prompt_session->NextUnit();
    if (!unit) {
      return std::unexpected(unit.error());
    }
    selected_prompt_ = prompt;
    selected_prefill_[0] = prompt;
    selected_prefill_count_ = 1;
    double expected = ExpectedSeconds(Phase::kPrefill, unit->rows, prompt->floors);
    // Keep the scheduler's shortest/aged choice first. Only an already
    // runnable compatible chunk may share this same completed prompt unit.
    for (ChatWork* peer : frames) {
      if (selected_prefill_count_ >= cohort_model_->prefill_wave_capacity()) break;
      if (peer == prompt || peer->stage != ChatWork::Stage::kPrompt ||
          !prompt->prompt_session->CanJoin(*peer->prompt_session))
        continue;
      const auto other = peer->prompt_session->NextUnit();
      base::Check(other.has_value(), "a compatible prompt has no declared unit");
      selected_prefill_[selected_prefill_count_++] = peer;
      expected += ExpectedSeconds(Phase::kPrefill, other->rows, peer->floors);
    }
    if (unit->phase != Llm::PromptSession::Phase::kChunk) {
      expected +=
          static_cast<double>(prompt->branch.state_snapshot_bytes()) / kSwapFloorBytesPerSecond;
    }
    return api::CooperativeBackend::Unit{.phase = Phase::kPrefill, .expected_seconds = expected};
  }

  std::expected<void, std::string> Advance(
      std::span<api::CooperativeBackend::Work* const> work) override {
    if (auto valid = CheckCohort(work); !valid) {
      return valid;
    }
    // Whether this unit's waits were cancelled for a hang (rung 1), asked
    // after it.
    (void)server_.node().TakeHangCancelled();
    if (selected_swap_) {
      selected_swap_ = false;
      bool needed = false;
      for (auto* base : work) {
        auto& frame = static_cast<ChatWork&>(*base);
        if (!frame.exchange.Next(Phase::kSwap, frame.swap_bytes)) {
          frame.Cancel();
        } else {
          needed = true;
        }
      }
      if (!needed) {
        return {};
      }
      swapped_ = server_.resident() != cohort_model_;
      if (auto activated = server_.Activate(*cohort_model_, parts_); !activated) {
        swapped_ = false;
        if (server_.node().TakeHangCancelled()) {
          Say(log_,
              "hang recovery: the swap's hung work was cancelled; the swap failed and only "
              "the requests that needed it fail");
          (void)AwaitDrained();  // a read still in the drive: rung 3
        }
        if (server_.NodeHealthy()) {
          // Recovery first (D-102): a swap refused for room, or failed and
          // undone (or not: no model resident, the next activation loads one
          // whole), fails only these requests. The cohort posted no native
          // work, so its retirement has no references to fence.
          for (auto* base : work) {
            auto& frame = static_cast<ChatWork&>(*base);
            frame.error =
                Failure(503, "the model could not be made resident: " + activated.error());
          }
          return {};
        }
        Fail("making the cooperative chat model resident: " + activated.error());
        return activated;
      }
      // Resident: the cohort's native work begins, whose references its
      // retirement fences.
      cohort_native_touched_ = true;
      for (auto* base : work) {
        static_cast<ChatWork&>(*base).native_touched = true;
      }
      return {};
    }
    if (selected_prompt_ == nullptr && selected_decode_.empty()) {
      return std::unexpected("advancing chat without its declared unit");
    }
    // Selection may itself change native ownership before it reports an
    // error. Retirement therefore needs the fence even after refusal here.
    cohort_native_touched_ = true;
    // All active states stay protected even while just one branch prefills.
    if (auto selected = SelectCohort(); !selected) {
      if (server_.node().TakeHangCancelled()) {
        return HangCohort("selecting the native chat cohort");
      }
      Fail("selecting the native chat cohort: " + selected.error());
      return selected;
    }
    if (selected_prompt_ != nullptr && selected_prefill_count_ > 1) return AdvancePrefillWave();
    if (selected_prompt_ != nullptr) {
      selected_prefill_count_ = 0;
      ChatWork& frame = *std::exchange(selected_prompt_, nullptr);
      frame.native_touched = true;
      // The schedule's counters (cohort_schedule.h), as this unit runs; a
      // unit that prefills no rows (a reuse, a checkpoint) ages no one.
      const auto unit = frame.prompt_session->NextUnit();
      const bool ages = unit && unit->rows > 0;
      for (ChatWork* other : cohort_) {
        if (other == nullptr || other == &frame) {
          continue;
        }
        if (other->stage == ChatWork::Stage::kGeneration) {
          ++other->waited;
        } else if (other->stage == ChatWork::Stage::kPrompt && ages) {
          ++other->passed;
        }
      }
      frame.passed = 0;
      const PrefillGoOn go_on = [&frame](std::uint32_t rows) {
        return !frame.cancelled && frame.exchange.Next(Phase::kPrefill, rows);
      };
      if (auto advanced = frame.prompt_session->Advance(go_on, true); !advanced) {
        if (server_.node().TakeHangCancelled()) {
          return HangCohort("a prompt's unit");
        }
        std::string error = "the prompt could not be processed: " + advanced.error();
        if (frame.prompt_session->refused()) {
          frame.refusal = std::move(error);
          const std::array<ChatWork*, 1> refused = {&frame};
          WaitForCapacity(refused);
        } else {
          frame.error = Failure(500, std::move(error));
        }
        Rebalance();
        return {};  // own failure; Retire independently proves references
      }
      if (server_.node().TakeHangCancelled()) {
        return HangCohort("a prompt's unit");  // a step ended after its request was cancelled
      }
      if (frame.literal != nullptr && unit && unit->phase == Llm::PromptSession::Phase::kReuse) {
        frame.literal->PromptStarted();
      }
      if (frame.prompt_session->done()) {
        auto begun = BeginChatGeneration(frame);
        Rebalance();
        return begun;
      }
      return {};
    }
    std::vector<Llm::GenerationSession*> sessions;
    std::vector<ChatWork*> stepped;
    sessions.reserve(selected_decode_.size());
    stepped.reserve(selected_decode_.size());
    for (ChatWork* frame : selected_decode_) {
      if (!frame->exchange.Next(Phase::kDecode, 0)) {
        frame->Cancel();
        continue;
      }
      frame->native_touched = true;
      sessions.push_back(frame->generation_session.get());
      stepped.push_back(frame);
    }
    selected_decode_.clear();
    if (sessions.empty()) {
      return {};
    }
    if (auto advanced = cohort_model_->RunGenerationWave(sessions, true); !advanced) {
      if (server_.node().TakeHangCancelled()) {
        return HangCohort("a generation wave");
      }
      Fail("the native chat generation wave failed: " + advanced.error());
      return advanced;
    }
    if (server_.node().TakeHangCancelled()) {
      return HangCohort("a generation wave");  // a step ended after its request was cancelled
    }
    for (ChatWork* frame : stepped) {
      frame->waited = 0;
    }
    std::vector<ChatWork*> refused;
    for (ChatWork* frame : stepped) {
      if (frame->generation_session->refused()) {
        frame->refusal = "the generation failed: " + frame->generation_session->refusal();
        refused.push_back(frame);
      }
    }
    if (!refused.empty()) {
      WaitForCapacity(refused);
    }
    Rebalance();
    return {};
  }

  api::CooperativeBackend::Retirement Retire(api::CooperativeBackend::Work& work) override {
    auto& frame = static_cast<ChatWork&>(work);
    if (&frame.owner != this || frame.retired || frame.slot >= cohort_.size() ||
        cohort_[frame.slot] != &frame || !frame.terminal()) {
      return {.result = std::unexpected(Failure(500, "retiring an invalid chat owner")),
              .references_retired = false};
    }
    if (frame.prompt_session != nullptr) {
      if (!frame.prompt_session->done()) {
        frame.prompt_session->Cancel();
      }
      if (!frame.resume && !frame.continued) {
        frame.reused = frame.prompt_session->reused();  // a rebuild's reuse is not the turn's
      }
      if (auto finished = frame.prompt_session->Finish(); !finished && !frame.error) {
        frame.error = Failure(500, "the prompt failed: " + finished.error());
      }
    }
    if (frame.generation_session != nullptr) {
      if (!frame.generation_session->done()) {
        frame.generation_session->Cancel();
      }
      if (auto finished = frame.generation_session->Finish(); !finished && !frame.error) {
        frame.error = Failure(500, "the generation failed: " + finished.error());
      }
    }
    const auto peers =
        std::ranges::count_if(cohort_, [](const ChatWork* peer) { return peer != nullptr; });
    if (cohort_native_touched_ && hang_reset_ != nullptr) {
      // After a hang's cancellation (rung 2) the cohort may be faulted: a
      // fence of the model's stream, not the cohort's retirement, proves
      // that nothing queued still borrows this owner's frame. A fence that
      // cannot be made, or a cancellation that never drained, leaves
      // nothing less than a restart (rung 3).
      if (!AwaitDrained()) {
        return {.result = std::unexpected(HungFailure()), .references_retired = false};
      }
      if (auto fenced = server_.FenceModel(frame.model); !fenced) {
        if (ladder_ != nullptr) {
          ladder_->Restart(
              std::format("the {} cohort's stream could not be fenced after its "
                          "hung work was cancelled: {}",
                          frame.model.name(), fenced.error()));
        }
        return {.result = std::unexpected(HungFailure()), .references_retired = false};
      }
    } else if (cohort_native_touched_) {
      // The explicit fence hook, not a session status or StateUsable flag,
      // proves that native copies/jobs no longer borrow this owner's frame.
      const auto retired = server_.RetireRequestBranches(frame.model, peers == 1);
      if (!retired.references_retired) {
        return {.result = std::unexpected(Failure(500, "native chat references did not retire")),
                .references_retired = false};
      }
      if (!retired.result) {
        Fail("ending the native chat cohort: " + retired.result.error());
        if (!frame.error) {
          frame.error = Failure(500, "the native chat cohort failed during retirement");
        }
      }
    }
    // Yielding its place (D-102): its state stays as a finished turn's, and
    // it continues from its generation so far when taken up again, unless
    // the generation ended whole anyway.
    const bool yields = frame.yielding && !frame.error && !frame.generation.stopped &&
                        ((frame.model_paused && frame.generation.tokens.empty()) ||
                         (!frame.generation.tokens.empty() &&
                          frame.generation.tokens.size() < frame.options.max_tokens));
    std::shared_ptr<api::Yielded> yielded;
    if (yields) {
      if (frame.model_paused) {
        Say(log_,
            std::format("{}'s request paused for a model switch: {} of {} prompt tokens "
                        "processed, {} generated",
                        frame.model.name(),
                        std::min(frame.branch.history().size(), frame.rendered.tokens().size()),
                        frame.rendered.tokens().size(), frame.generation.tokens.size()));
      }
      auto kept = KeepContinuation(frame.rendered, frame.generation, frame.options,
                                   frame.output.Save(), frame.reused, *memory_,
                                   frame.model_paused ? &frame.branch : nullptr, frame.literal);
      if (!kept) {
        frame.error = kept.error();
      } else {
        yielded = std::move(*kept);
      }
    } else if (!frame.error && !frame.generation.tokens.empty()) {
      if (frame.literal == nullptr) {
        frame.output.Finish();  // including a generation preempted and not yet resumed
      }
    }
    api::LiteralResult literal_result;
    if (frame.literal != nullptr && !yields) {
      if (!frame.error) {
        (void)frame.literal->Finish(
            [&frame](std::string_view piece) { return frame.exchange.Content(piece); });
        if (frame.literal->error()) {
          frame.error = *frame.literal->error();
        }
      }
      literal_result = frame.literal->TakeResult();
    }
    if (!frame.error && ladder_ != nullptr) {
      ladder_->Served(frame.model.name());  // a later hang may reset it again
    }
    api::Completion result{
        .completion_tokens = static_cast<std::uint32_t>(frame.generation.tokens.size()),
        .cached_tokens = frame.literal != nullptr ? 0 : frame.reused,
        .stopped = frame.rendered.max_tokens == 0 || frame.generation.stopped,
        .literal = std::move(literal_result),
        .yielded = std::move(yielded)};
    cohort_[frame.slot] = nullptr;
    frame.retired = true;
    memory_wait_.MemberRetired();  // its state is idle now, and reclaimable
    frame.generation_session.reset();
    frame.prompt_session.reset();
    if (peers == 1) {
      cohort_model_ = nullptr;
      cohort_native_touched_ = false;
      selected_prompt_ = nullptr;
      selected_prefill_count_ = 0;
      selected_decode_.clear();
      if (Llm* hung = std::exchange(hang_reset_, nullptr); hung != nullptr) {
        // The last member of a cohort whose work hung: the model is reset.
        (void)RecoverFromHang(*hung, std::format("the {} cohort's unit", hung->name()));
      }
    } else {
      // Its state is no longer leased once the cohort is next selected:
      // members waiting for capacity retry (cohort_capacity.h).
      Cohort cohort = Snapshot();
      OnMemberRetired(cohort);
      Apply(cohort);
      Rebalance();
    }
    return {.result = frame.error ? std::expected<api::Completion, api::Error>(
                                        std::unexpected(*frame.error))
                                  : std::expected<api::Completion, api::Error>(std::move(result)),
            .references_retired = true};
  }

  std::vector<api::ModelInfo> Models() const override {
    std::vector<api::ModelInfo> models;
    for (const config::ModelEntry& entry : config_.models) {
      if (entry.composition) {
        models.push_back({.name = entry.name, .chat = false, .context = 0, .render_bytes = 0});
        continue;
      }
      Served* m = server_.Find(entry.name);
      if (m != nullptr && m->llm()) {
        const auto& l = static_cast<Llm&>(*m);
        models.push_back({.name = entry.name,
                          .chat = true,
                          .context = l.usable_context(),
                          .render_bytes = l.render_bytes()});
      }
    }
    return models;
  }

  std::expected<api::Completion, api::Error> Complete(const api::ChatRequest& request,
                                                      api::Exchange& exchange) override {
    Served* m = server_.Find(request.model);
    if (m == nullptr || !m->llm()) {
      return std::unexpected(Failure(404, "The model does not exist", "model_not_found", "model"));
    }
    auto& l = static_cast<Llm&>(*m);
    auto rendered = PrepareChat(l, request, exchange, *memory_);
    if (!rendered) {
      return std::unexpected(rendered.error());
    }
    return SerialChat(l, request, exchange, std::move(*rendered), nullptr);
  }

  std::expected<api::Completion, api::Error> Resume(const api::ChatRequest& request,
                                                    api::Exchange& exchange,
                                                    const api::Yielded& from) override {
    Served* m = server_.Find(request.model);
    if (m == nullptr || !m->llm()) {
      return std::unexpected(Failure(404, "The model does not exist", "model_not_found", "model"));
    }
    // Only this backend yields, so only its continuations come back.
    const auto& resume = static_cast<const ChatResume&>(from);
    return SerialChat(static_cast<Llm&>(*m), request, exchange, resume.rendered, &resume);
  }

  // A chat request on the serial path, from its rendering, or continuing
  // from where it yielded its place (`resume`).
  std::expected<api::Completion, api::Error> SerialChat(Llm& l, const api::ChatRequest& request,
                                                        api::Exchange& exchange,
                                                        ChatPrompt rendered,
                                                        const ChatResume* resume) {
    Served* const m = &l;
    // Continuing: the prompt and everything generated so far (the state
    // holds most of it), and the generation's remaining tokens.
    std::vector<std::int32_t> continued;
    if (resume != nullptr) {
      continued = resume->history;
      if (!resume->generation.tokens.empty()) {
        continued.push_back(resume->generation.tokens.back());
      }
    }
    MemoryCharge token_charge;
    if (auto refused = FundHistory(token_charge, *memory_,
                                   continued.capacity() * sizeof(std::int32_t), false)) {
      return std::unexpected(*refused);
    }
    const std::vector<std::int32_t>& tokens = resume != nullptr ? continued : rendered.tokens();
    const auto prompt = static_cast<std::uint32_t>(rendered.tokens().size());
    const std::uint32_t done =
        resume != nullptr ? static_cast<std::uint32_t>(resume->generation.tokens.size()) : 0;
    const std::uint32_t max_tokens = rendered.max_tokens - done;
    // What making the model resident pages in: its weights and its state,
    // and the resident conversation's state written back (an upper bound;
    // the watchdog allows the swap for it, watchdog.h).
    std::uint64_t swap_bytes = 0;
    if (server_.resident() != m) {
      swap_bytes = m->weight_read_bytes() + l.state_snapshot_bytes();
      if (Served* out = server_.resident(); out != nullptr && out->llm()) {
        swap_bytes += static_cast<Llm&>(*out).state_snapshot_bytes();
      }
    }
    const Floors floors = l.floors();
    // Its usage: what so far it continues from.
    const auto so_far = [&](std::uint32_t cached) {
      return api::Completion{.completion_tokens = done,
                             .cached_tokens = resume != nullptr ? resume->reused : cached,
                             .stopped = false,
                             .literal = {},
                             .yielded = nullptr};
    };
    if (!exchange.Admit({.prompt_tokens = prompt,
                         .max_tokens = max_tokens,
                         .swap_bytes = swap_bytes,
                         .floors = floors})) {
      return so_far(0);
    }

    swapped_ = server_.resident() != m;
    if (swapped_ && !exchange.Next(Phase::kSwap, swap_bytes)) {
      swapped_ = false;
      return so_far(0);
    }
    (void)server_.node().TakeHangCancelled();  // asked after the work (D-102)
    if (auto r = server_.Activate(*m, parts_); !r) {
      if (server_.node().TakeHangCancelled()) {
        Say(log_,
            "hang recovery: the swap's hung work was cancelled; the swap failed and only "
            "the request that needed it fails");
        if (!AwaitDrained()) {  // a read still in the drive: rung 3
          swapped_ = false;
          return std::unexpected(HungFailure());
        }
      }
      if (!server_.NodeHealthy()) {  // otherwise recovered (D-102): this request only
        Fail(std::format("making {} resident: {}", m->name(), r.error()));
      }
      swapped_ = false;
      return std::unexpected(Failure(503, "the model could not be made resident: " + r.error()));
    }
    // A swap is one program (about 10 s on a Spark), watched as one unit
    // at its bytes; whatever ended the request meanwhile (the client gone,
    // the backend stalled, the deadline, the runtime stopping) ends it
    // here, before any of its model work. The exchange answers for it.
    if (!exchange.Continue()) {
      return so_far(0);
    }

    GenerateOptions options{.max_tokens = max_tokens,
                            .extra_stops = l.ChatStops(),
                            .stop = true,
                            .keep_logits = false,
                            .sampling = std::nullopt,
                            .seed = 0,
                            .on_tokens = {}};
    if (resume != nullptr) {
      options.sampling = resume->sampling;
      options.seed = resume->seed;
    } else if (auto sampling = SamplingOf(request, l)) {
      options.sampling = sampling;
      options.seed = request.seed.value_or(RandomSeed());
    }
    ChatOutput output(l, exchange, rendered.reasoning);
    if (resume != nullptr) {
      output.Restore(resume->output);
    }
    Generation generation;
    std::optional<api::Error> token_problem;
    options.on_tokens = [&](std::span<const std::int32_t> fresh) {
      const std::uint64_t held =
          (generation.tokens.capacity() + continued.capacity()) * sizeof(std::int32_t);
      token_problem = FundHistory(token_charge, *memory_, held, true);
      return !token_problem && output.Push(fresh);
    };
    std::uint32_t reused = 0;
    PrefillRun prefill;
    // Each chunk is a unit the watchdog allows for its rows (watchdog.h).
    const PrefillGoOn go_on = [&exchange](std::uint32_t rows) {
      return exchange.Next(Phase::kPrefill, rows);
    };
    // Idle branches' retained state gives way to this request, the largest
    // first; a refusal it still meets fails the request alone.
    const ScopedReclaim reclaim(l, SerialReclaim(l));
    SerialCapacity capacity(l);
    auto ran = server_.InRequest(*m, [&]() -> Status {
      // Between chunks, whatever ends the request (the client gone, the
      // backend stalled, the deadline, the runtime stopping) stops the
      // prefill: an ordinary end, the state holding the chunks that ran
      // (serving.h Llm::Prefill).
      std::vector<float> last;
      if (auto r = l.PreparePrompt(tokens, rendered.stable_boundary, last, reused, go_on, &prefill);
          !r) {
        return capacity(r, "the prompt could not be processed: ");
      }
      if (prefill.stopped || !exchange.Next(Phase::kDecode, 0)) {
        return {};  // the state holds what ran; the exchange answers for why
      }
      return capacity(l.Generate(last, options, generation), "the generation failed: ");
    });
    if (!ran) {
      if (server_.node().TakeHangCancelled()) {
        return std::unexpected(RecoverFromHang(*m, std::format("{}'s request", m->name())));
      }
      Fail(std::format("{}'s request: {}", m->name(), ran.error()));
      return std::unexpected(Failure(500, "the generation failed; the runtime is stopping"));
    }
    if (server_.node().TakeHangCancelled()) {
      // A step ended after its request was cancelled for a hang: the reply
      // so far stands, and the model is reset for the next.
      (void)RecoverFromHang(*m, std::format("{}'s request", m->name()));
    }
    if (capacity.refusal()) {
      RefusedAlone(l);
      return std::unexpected(Failure(500, *capacity.refusal()));
    }
    if (token_problem) {
      return std::unexpected(*token_problem);
    }
    if (prefill.stopped) {
      // Token counts and times only (D-014); the exchange answers for why.
      Say(log_, std::format("{}'s prefill stopped after {} chunks: the state holds {} of the "
                            "prompt's {} tokens",
                            m->name(), prefill.chunks, l.history().size(), tokens.size()));
      return so_far(reused);
    }
    // Yielding its place (D-102): ended at this completed step, unless the
    // generation is whole anyway; it continues from here when taken up.
    if (exchange.Yielding() && !generation.stopped && !generation.tokens.empty() &&
        generation.tokens.size() < max_tokens) {
      MemoryCharge combined;
      const std::uint64_t bytes = (done + generation.tokens.size()) * sizeof(std::int32_t);
      if (!combined.Add(*memory_, 2 * bytes)) {
        return std::unexpected(api::MemoryRefusal(*memory_, 2 * bytes, "the continued generation"));
      }
      Generation all = resume != nullptr ? resume->generation : Generation{};
      all.tokens.insert(all.tokens.end(), generation.tokens.begin(), generation.tokens.end());
      const auto total = static_cast<std::uint32_t>(all.tokens.size());
      auto yielded = KeepContinuation(rendered, all, options, output.Save(),
                                      resume != nullptr ? resume->reused : reused, *memory_);
      if (!yielded) {
        return std::unexpected(yielded.error());
      }
      return api::Completion{.completion_tokens = total,
                             .cached_tokens = (*yielded)->reused,
                             .stopped = false,
                             .literal = {},
                             .yielded = std::move(*yielded)};
    }
    output.Finish();
    if (ladder_ != nullptr) {
      ladder_->Served(m->name());  // a later hang may reset it again
    }
    return api::Completion{
        .completion_tokens = done + static_cast<std::uint32_t>(generation.tokens.size()),
        .cached_tokens = resume != nullptr ? resume->reused : reused,
        .stopped = generation.stopped,
        .literal = {},
        .yielded = nullptr};
  }

  std::expected<api::Completion, api::Error> Complete(const api::CompletionRequest& request,
                                                      api::Exchange& exchange) override {
    const auto& controls = request.options;
    Served* model = server_.Find(controls.model);
    if (model == nullptr || !model->llm()) {
      return std::unexpected(Failure(404, "The model does not exist", "model_not_found", "model"));
    }
    auto& llm = static_cast<Llm&>(*model);
    // A response's most: what the request memory holds (D-102); its score
    // rows are charged to it before any work (below).
    const auto response_bytes = static_cast<std::size_t>(memory_->capacity());
    auto prompt = api::PrepareLiteralPrompt(request, llm.tokenizer(), llm.usable_context(),
                                            response_bytes, memory_.get());
    if (!prompt) {
      return std::unexpected(prompt.error());
    }
    auto prompt_tokens = api::PromptTokens::Hold(std::move(prompt->tokens), *memory_);
    if (!prompt_tokens) {
      return std::unexpected(prompt_tokens.error());
    }
    const auto& tokens = (*prompt_tokens)->values();
    const auto max_tokens = controls.max_tokens.value_or(16);
    const bool score_prompt = request.prompt_logprobs || (request.echo && request.logprobs);
    api::Completion result;
    result.literal.prompt_text = std::move(prompt->text);
    std::size_t budget = 1024 + (6 * result.literal.prompt_text.size());
    if (budget > response_bytes) {
      return std::unexpected(Failure(413, "prompt echo exceeds the response size", {}, "prompt"));
    }
    // The rows it will hold, at most (PrepareLiteralPrompt figured them),
    // and its text: charged while it runs (the completed body is charged
    // in its place when it is made).
    MemoryCharge rows;
    if (const std::uint64_t held = prompt->score_bytes + budget; !rows.Add(*memory_, held)) {
      return std::unexpected(api::MemoryRefusal(*memory_, held, "the completion's scores"));
    }
    std::uint64_t swap_bytes = 0;
    if (server_.resident() != model) {
      swap_bytes = model->weight_read_bytes() + llm.state_snapshot_bytes();
      if (auto* out = server_.resident(); out != nullptr && out->llm()) {
        swap_bytes += static_cast<Llm&>(*out).state_snapshot_bytes();
      }
    }
    Floors floors = llm.floors();
    if (score_prompt) {
      // Pure scoring is one-row target work, not tiled prefill. Its scaled
      // deadline and per-row watchdog allowance must use the decode floor.
      floors.prefill = floors.decode;
    }
    if (!exchange.Admit({.prompt_tokens = static_cast<std::uint32_t>(tokens.size()),
                         .max_tokens = max_tokens,
                         .swap_bytes = swap_bytes,
                         .floors = floors})) {
      return api::Completion{};
    }
    swapped_ = server_.resident() != model;
    if (swapped_ && !exchange.Next(Phase::kSwap, swap_bytes)) {
      swapped_ = false;
      return api::Completion{};
    }
    (void)server_.node().TakeHangCancelled();  // asked after the work (D-102)
    if (auto activated = server_.Activate(*model, parts_); !activated) {
      if (server_.node().TakeHangCancelled()) {
        Say(log_,
            "hang recovery: the swap's hung work was cancelled; the swap failed and only "
            "the request that needed it fails");
        (void)AwaitDrained();  // a read still in the drive: rung 3
      }
      if (!server_.NodeHealthy()) {  // otherwise recovered (D-102): this request only
        Fail("making the literal completion model resident: " + activated.error());
      }
      swapped_ = false;
      return std::unexpected(
          Failure(503, "the model could not be made resident: " + activated.error()));
    }
    if (!exchange.Continue()) {
      return api::Completion{};
    }
    std::optional<api::Error> output_problem;
    api::LiteralRows prompt_rows(llm.tokenizer(), request.return_tokens_as_token_ids, true, 0,
                                 budget, response_bytes);
    api::LiteralRows generated_rows(
        llm.tokenizer(), request.return_tokens_as_token_ids, false,
        request.echo ? api::TextCharacters(result.literal.prompt_text) : 0, budget, response_bytes);
    const std::uint32_t prompt_top = std::max(request.prompt_logprobs.value_or(0),
                                              request.echo ? request.logprobs.value_or(0) : 0);
    const auto add_prompt = [&](std::int32_t id, std::span<const float> row, bool first) {
      auto scored = prompt_rows.Add(id, row, prompt_top, first, first && prompt->added_bos);
      if (!scored) {
        output_problem = scored.error();
        return false;
      }
      if (request.echo && request.logprobs) {
        result.literal.logprobs.push_back(*scored);
      }
      if (request.prompt_logprobs) {
        result.literal.prompt_logprobs.push_back(std::move(*scored));
      }
      return exchange.Continue();
    };
    tokenizer::StreamDecoder decoder(llm.tokenizer(), {});
    GenerateOptions options{.max_tokens = max_tokens,
                            .stop = true,
                            .keep_logits = false,
                            .sampling = std::nullopt,
                            .seed = 0,
                            .on_tokens = {},
                            .on_logits = {}};
    if (auto sampling = SamplingOf(controls, llm)) {
      options.sampling = sampling;
      options.seed = controls.seed.value_or(RandomSeed());
    }
    if (request.logprobs) {
      options.on_logits = [&](std::int32_t id, std::span<const float> row) {
        auto scored = generated_rows.Add(id, row, *request.logprobs);
        if (!scored) {
          output_problem = scored.error();
          return false;
        }
        result.literal.logprobs.push_back(std::move(*scored));
        return exchange.Continue();
      };
    }
    Generation generation;
    MemoryCharge token_charge;
    options.on_tokens = [&](std::span<const std::int32_t> fresh) {
      if (auto refused = FundHistory(token_charge, *memory_,
                                     generation.tokens.capacity() * sizeof(std::int32_t), true)) {
        output_problem = std::move(refused);
        return false;
      }
      std::string piece;
      for (const auto id : fresh) {
        if (!decoder.Push(id, piece)) {
          output_problem =
              Failure(500, "a generated token cannot be decoded", "invalid_model_token");
          return false;
        }
      }
      if (budget > response_bytes || 6 * piece.size() > response_bytes - budget) {
        output_problem =
            Failure(413, "completion text exceeds the response size", "response_too_large");
        return false;
      }
      budget += 6 * piece.size();
      return piece.empty() ? exchange.Continue() : exchange.Content(piece);
    };
    PrefillRun prefill;
    // As chat's serial turn: idle branches' retained state gives way to this
    // request, and a refusal it still meets fails the request alone.
    const ScopedReclaim reclaim(llm, SerialReclaim(llm));
    SerialCapacity capacity(llm);
    auto ran = server_.InRequest(*model, [&]() -> Status {
      if (auto cleared = llm.Clear(); !cleared) {
        return cleared;
      }
      std::vector<float> last;
      const PrefillGoOn go_on = [&](std::uint32_t rows) {
        return exchange.Next(Phase::kPrefill, rows);
      };
      if (score_prompt) {
        if (!add_prompt(tokens.front(), {}, true)) {
          prefill.stopped = true;
          return {};
        }
        if (auto scored = llm.ScorePrompt(
                tokens, last,
                [&](std::int32_t id, std::span<const float> row) {
                  return add_prompt(id, row, false);
                },
                go_on, &prefill);
            !scored) {
          return capacity(scored, "the prompt could not be processed: ");
        }
      } else if (auto filled = llm.Prefill(tokens, last, go_on, &prefill); !filled) {
        return capacity(filled, "the prompt could not be processed: ");
      }
      prompt_rows.Finish();  // same boundary flush as prompt_text, never carried into generation
      if (prefill.stopped || !exchange.Continue() || max_tokens == 0 ||
          !exchange.Next(Phase::kDecode, 0)) {
        return {};
      }
      return capacity(llm.Generate(last, options, generation), "the generation failed: ");
    });
    if (!ran) {
      if (server_.node().TakeHangCancelled()) {
        return std::unexpected(RecoverFromHang(*model, "a literal completion"));
      }
      Fail("the literal completion failed: " + ran.error());
      return std::unexpected(Failure(500, "the completion failed; the runtime is stopping"));
    }
    if (server_.node().TakeHangCancelled()) {
      (void)RecoverFromHang(*model, "a literal completion");
    }
    if (capacity.refusal()) {
      RefusedAlone(llm);
      return std::unexpected(Failure(500, *capacity.refusal()));
    }
    if (output_problem) {
      return std::unexpected(*output_problem);
    }
    std::string rest;
    decoder.Finish(rest);
    generated_rows.Finish();
    if (!rest.empty()) {
      if (budget > response_bytes || 6 * rest.size() > response_bytes - budget) {
        return std::unexpected(Failure(413, "completion text exceeds the response size",
                                       "response_unrepresentable", "logprobs"));
      }
      (void)exchange.Content(rest);
    }
    result.completion_tokens = static_cast<std::uint32_t>(generation.tokens.size());
    result.stopped = max_tokens == 0 || generation.stopped;
    if (ladder_ != nullptr) {
      ladder_->Served(model->name());  // a later hang may reset it again
    }
    return result;
  }

  void AfterResponse() override {
    if (!swapped_) {
      return;
    }
    swapped_ = false;
    if (auto r = server_.FinishSwap(parts_); !r) {
      Fail("finishing the swap: " + r.error());
      return;
    }
    Say(log_,
        std::format(
            "swap {} -> {}: ready in {:.3f} s (evict {:.3f}, restore {:.3f}, "
            "page-in {:.3f}, setup {:.3f}; {} graphs reclaimed for it); backing released {:.3f} s "
            "later",
            parts_.from.empty() ? "(nothing)" : parts_.from, parts_.to, parts_.total, parts_.evict,
            parts_.restore, parts_.page_in, parts_.setup, parts_.dropped_graphs, parts_.release));
  }

  // Between units and while idle: idle conversations past their retention
  // or the spill budget, and memory pressure from outside (Server::Maintain).
  void Maintain() override {
    if (failure_.empty()) {
      // The request memory grown to what the I/O and parse threads wanted
      // (or a denial counted), or given back toward what is used.
      memory_->Settle();
      server_.Maintain();
    }
  }
  bool healthy() const override { return failure_.empty(); }
  std::string failure() const override { return failure_; }

 private:
  Status AdvancePrefillWave() {
    const auto selected = std::span(selected_prefill_).first(selected_prefill_count_);
    selected_prompt_ = nullptr;
    selected_prefill_count_ = 0;
    std::array<Llm::PromptSession*, Llm::kMaxBranches> sessions{};
    std::array<PrefillGoOn, Llm::kMaxBranches> callbacks{};
    std::array<std::uint32_t, Llm::kMaxBranches> before{};
    for (std::size_t i = 0; i < selected.size(); ++i) {
      ChatWork* frame = selected[i];
      frame->native_touched = true;
      sessions[i] = frame->prompt_session.get();
      before[i] = sessions[i]->run().end;
      callbacks[i] = [frame](std::uint32_t rows) {
        return !frame->cancelled && frame->exchange.Next(Phase::kPrefill, rows);
      };
    }
    const auto count = selected.size();
    if (auto advanced = cohort_model_->RunPromptWave(std::span(sessions).first(count),
                                                     std::span(callbacks).first(count), true);
        !advanced) {
      if (server_.node().TakeHangCancelled()) return HangCohort("a prompt wave");
      Fail("the native chat prompt wave failed: " + advanced.error());
      return advanced;
    }
    if (server_.node().TakeHangCancelled()) return HangCohort("a prompt wave");
    // No session is prepared now: native completion and all peer history
    // publication precede cancellation, retirement or BeginChatGeneration.
    bool ages = false;
    std::array<bool, Llm::kMaxBranches> processed{};
    for (std::size_t i = 0; i < count; ++i) {
      if (sessions[i]->run().end > before[i]) {
        selected[i]->passed = 0;
        processed[selected[i]->slot] = true;
        ages = true;
      }
    }
    if (ages) {
      for (ChatWork* other : cohort_) {
        if (!other || processed[other->slot]) continue;
        if (other->stage == ChatWork::Stage::kGeneration)
          ++other->waited;
        else if (other->stage == ChatWork::Stage::kPrompt)
          ++other->passed;
      }
    }
    std::array<ChatWork*, Llm::kMaxBranches> refused{};
    std::size_t refusals = 0;
    for (std::size_t i = 0; i < count; ++i) {
      ChatWork& frame = *selected[i];
      const auto& result = sessions[i]->last_unit_result();
      if (!result) {
        const auto error = "the prompt could not be processed: " + result.error();
        if (sessions[i]->refused()) {
          frame.refusal = error;
          refused[refusals++] = &frame;
        } else {
          frame.error = Failure(500, error);
        }
      } else if (sessions[i]->done()) {
        if (auto begun = BeginChatGeneration(frame); !begun) return begun;
      }
    }
    if (refusals != 0) WaitForCapacity(std::span(refused).first(refusals));
    Rebalance();
    return {};
  }

  std::uint64_t SwapBytes(Llm& model) const {
    if (server_.resident() == &model) {
      return 0;
    }
    const auto used = [](Llm& llm) {
      std::uint64_t bytes = 0;
      // Branch wrappers and metadata are stable. Actual expiry, checkpoint
      // restoration and growth remain in their declared native prompt units.
      for (std::size_t i = 0; i < llm.branches(); ++i) {
        auto branch = llm.branch(i);
        if (branch) {
          bytes += (*branch)->state_snapshot_bytes();
        }
      }
      return bytes;
    };
    auto bytes = model.weight_read_bytes() + used(model);
    if (auto* outgoing = server_.resident(); outgoing != nullptr && outgoing->llm()) {
      bytes += used(static_cast<Llm&>(*outgoing));
    }
    return bytes;
  }

  Status CheckCohort(std::span<api::CooperativeBackend::Work* const> work) const {
    if (cohort_model_ == nullptr || work.empty() || work.size() > cohort_.size()) {
      return std::unexpected("the chat cohort has no active model or exceeds its bound");
    }
    std::array<bool, Llm::kMaxBranches> seen{};
    for (auto* base : work) {
      if (base == nullptr) {
        return std::unexpected("the chat cohort has a null owner");
      }
      // Only Start on this backend constructs the caller-owned descriptors.
      const auto& frame = static_cast<const ChatWork&>(*base);
      if (&frame.owner != this || &frame.model != cohort_model_ || frame.slot >= cohort_.size() ||
          cohort_[frame.slot] != &frame || frame.retired || frame.terminal() || seen[frame.slot]) {
        return std::unexpected("the chat cohort contains an inactive, foreign or duplicate owner");
      }
      seen[frame.slot] = true;
    }
    if (std::ranges::count(seen, true) !=
        std::ranges::count_if(cohort_, [](const ChatWork* frame) { return frame != nullptr; })) {
      return std::unexpected("the chat unit omitted a claimed owner");
    }
    return {};
  }

  Status SelectCohort() {
    std::array<Llm::Branch*, Llm::kMaxBranches> branches{};
    std::size_t count = 0;
    for (ChatWork* frame : cohort_) {
      if (frame != nullptr) {
        branches[count++] = &frame->branch;
      }
    }
    return server_.SelectRequestBranches(*cohort_model_, std::span(branches).first(count));
  }

  static std::expected<std::unique_ptr<Llm::PromptSession>, std::string> WorkPrompt(
      ChatWork& frame) {
    if (frame.literal != nullptr && frame.literal->score_prompt() && !frame.resume) {
      return frame.branch.BeginScoringPrompt(
          frame.rendered.tokens(),
          [&frame](std::int32_t id, std::span<const float> row) {
            return frame.literal->PromptRow(frame.prompt_session->run().end, id, row) &&
                   frame.exchange.Continue();
          },
          frame.literal->prompt_started());
    }
    const bool started = frame.literal != nullptr && frame.literal->prompt_started();
    return frame.branch.BeginPrompt(frame.resume ? frame.resume_tokens : frame.rendered.tokens(),
                                    frame.rendered.stable_boundary,
                                    frame.literal != nullptr && !started, frame.resume || started);
  }

  static Status BeginChatGeneration(ChatWork& frame) {
    if (!frame.resume && !frame.continued) {
      frame.reused = frame.prompt_session->reused();
    }
    const bool stopped = frame.prompt_session->run().stopped;
    if (auto finished = frame.prompt_session->Finish(); !finished) {
      frame.error = Failure(500, "the prompt failed: " + finished.error());
      return {};
    }
    if (frame.literal != nullptr) {
      frame.literal->EndPrompt();
      if (frame.literal->error()) {
        frame.error = *frame.literal->error();
        return {};
      }
    }
    if (frame.rendered.max_tokens == 0) {
      frame.stage = ChatWork::Stage::kDone;
      return {};
    }
    if (stopped || frame.cancelled || !frame.exchange.Next(Phase::kDecode, 0)) {
      frame.stage = ChatWork::Stage::kDone;
      return {};
    }
    // After a preemption the rebuilt state continues the generation from its
    // anchor; nothing is chosen or streamed again.
    auto generation = frame.resume ? frame.branch.ResumeGeneration(frame.prompt_session->last(),
                                                                   frame.options, frame.generation)
                                   : frame.branch.BeginGeneration(frame.prompt_session->last(),
                                                                  frame.options, frame.generation);
    if (!generation) {
      frame.error = Failure(500, "the generation could not start: " + generation.error());
      return {};
    }
    frame.resume = false;
    frame.switch_restore = false;
    frame.resume_tokens = {};
    frame.generation_session = std::move(*generation);
    frame.stage = ChatWork::Stage::kGeneration;
    frame.waited = 0;
    // Last logits are no longer borrowed by generation: Begin already chose
    // the first token. The callback and options remain in this stable frame.
    frame.prompt_session.reset();
    return {};
  }

  // State capacity (cohort_capacity.h; docs/runtime-serving.md#state-
  // capacity-in-a-cohort). The cohort as the policy sees it, and back.
  Cohort Snapshot() const {
    Cohort cohort{};
    for (std::size_t i = 0; i < cohort_.size(); ++i) {
      if (const ChatWork* frame = cohort_[i]; frame != nullptr) {
        cohort[i] = {.present = true,
                     .terminal = frame->terminal(),
                     .admitted = frame->admitted,
                     .wait = frame->wait};
      }
    }
    return cohort;
  }
  void Apply(const Cohort& cohort) {
    for (std::size_t i = 0; i < cohort_.size(); ++i) {
      if (cohort_[i] != nullptr) {
        cohort_[i]->wait = cohort[i].wait;
      }
    }
  }

  // Within Advance, the cohort selected: members whose unit the state's
  // capacity refused (each with its `refusal`, its state usable and its
  // completed prefix kept) wait, fail if they cannot fit alone, or free
  // the youngest waiter's state when no member could otherwise go on.
  void WaitForCapacity(std::span<ChatWork* const> refused) {
    std::vector<std::size_t> slots;
    for (const ChatWork* frame : refused) {
      slots.push_back(frame->slot);
    }
    Cohort cohort = Snapshot();
    // The node's reclaim order first (Server::Reclaim): plans and graphs,
    // idle conversations spilled, in the order's priority, for what each
    // refused member asked: all of it or nothing, member by member, so one
    // member's need that cannot be met does not keep another's from it
    // (those relieved run again; the rest are refused again and wait).
    bool reclaimed = false;
    for (const ChatWork* frame : refused) {
      const std::uint64_t needed =
          std::max<std::uint64_t>(frame->model.RefusedBytes(frame->branch), kExtentBytes);
      reclaimed =
          server_.Reclaim(needed, true, "conversation state refused") >= needed || reclaimed;
    }
    CapacityDecision decision = OnCapacityRefused(cohort, slots, reclaimed);
    Apply(cohort);
    if (decision.reclaim) {
      return;  // the refused members run again
    }
    for (const std::size_t slot : decision.refuse) {
      ChatWork& frame = *cohort_[slot];
      frame.error = Failure(500, frame.refusal);
    }
    // Positions and slots only (D-014).
    for (const ChatWork* frame : refused) {
      // The tokens its state holds: a generation publishes its history only
      // when it ends, and its anchor is not processed yet.
      const std::size_t held =
          frame->generation_session != nullptr
              ? frame->rendered.tokens().size() + frame->generation.tokens.size() - 1
              : frame->branch.history().size();
      if (frame->wait == CapacityWait::kBlocked) {
        Say(log_, std::format("{}'s request in slot {} waits for conversation-state capacity at "
                              "{} tokens",
                              frame->model.name(), frame->slot, held));
      } else if (frame->error) {
        Say(log_, std::format("{}'s request in slot {} does not fit the state's capacity alone at "
                              "{} tokens: refused",
                              frame->model.name(), frame->slot, held));
      }
    }
    if (decision.preempt) {
      Preempt(*cohort_[*decision.preempt]);
    }
  }

  // A serial request runs alone, as a cohort of one: refused for capacity,
  // it has the node's reclaim order free what its unit asked for (plans,
  // graphs, idle conversations spilled) and runs again.
  Llm::CapacityReclaim SerialReclaim(Llm& model) {
    return [this, &model](const Llm::Branch& refused) {
      const std::uint64_t needed =
          std::max<std::uint64_t>(model.RefusedBytes(refused), kExtentBytes);
      return server_.Reclaim(needed, true, "conversation state refused") >= needed;
    };
  }

  // Positions only (D-014).
  void RefusedAlone(const Llm& model) {
    Say(log_, std::format("{}'s request does not fit the state's capacity alone at {} tokens: "
                          "refused",
                          model.name(), model.history().size()));
  }

  // Frees a waiting member's state between its completed units: its
  // sessions end at their completed prefix, the host keeps the tokens to
  // rebuild it from, and its state is discarded (Branch::Clear). It begins
  // again once no other member can run (Rebalance).
  void Preempt(ChatWork& frame) {
    Status finished;
    if (frame.prompt_session != nullptr) {
      frame.prompt_session->Cancel();
      finished = frame.prompt_session->Finish();
      frame.prompt_session.reset();
    } else if (frame.generation_session != nullptr) {
      frame.generation_session->Cancel();
      finished = frame.generation_session->Finish();
      frame.generation_session.reset();
      if (finished) {
        // The prompt and the generated tokens the state holds; the anchor
        // stays in frame.generation, already streamed.
        frame.resume_tokens = frame.branch.history();
        (void)frame.FundTokens(true);
        frame.resume = true;
        frame.stage = ChatWork::Stage::kPrompt;
      }
    }
    if (!finished) {
      frame.error = Failure(500, "the request could not release its state: " + finished.error());
      return;
    }
    const std::size_t held = frame.branch.history().size();
    // Spill, not clear: its state written to its slot's spill file and
    // restored exactly when it begins again, so it continues as if never
    // set aside (Llm::SpillSetAside). Only a state that cannot be spilled
    // is cleared, and then rebuilt by prefill.
    if (auto spilled = frame.model.SpillSetAside(frame.branch); spilled) {
      Say(log_, std::format("{}'s request in slot {} set aside for its peers: its state spilled "
                            "at {} tokens, restored when they finish",
                            frame.model.name(), frame.slot, held));
      return;
    }
    if (auto cleared = frame.branch.Clear(); !cleared) {
      frame.error = Failure(500, "the request could not release its state: " + cleared.error());
      return;
    }
    Say(log_, std::format("{}'s request in slot {} released its state at {} tokens for its "
                          "peers; it is rebuilt when they finish",
                          frame.model.name(), frame.slot, held));
  }

  // After a unit or a retirement: when no member can go on, waiting
  // members retry, or the oldest preempted one begins its prompt again
  // (host-only; its first unit reuses or clears the branch as usual).
  void Rebalance() {
    Cohort cohort = Snapshot();
    const auto restart = NextRestart(cohort);
    Apply(cohort);
    if (!restart) {
      return;
    }
    ChatWork& frame = *cohort_[*restart];
    // A generation set aside resumes from its spilled state: its prompt
    // only restores it (BeginPrompt's `resume`).
    auto prompt = WorkPrompt(frame);
    if (!prompt) {
      frame.error = Failure(500, "the request could not begin again: " + prompt.error());
      return;
    }
    frame.prompt_session = std::move(*prompt);
    frame.stage = ChatWork::Stage::kPrompt;
  }

  // What a prompt of `tokens` on `branch` has yet to take of the budget:
  // its state through its tokens and its first step, less what the branch
  // holds now (continued, or cleared for it).
  static std::uint64_t PromptStateToCome(const Llm& model, const Llm::Branch& branch,
                                         std::size_t tokens) {
    const auto positions =
        static_cast<std::uint32_t>(std::min<std::size_t>(tokens + 1, model.context()));
    return StateToCome(model.StateBytesThrough(positions), model.ResidentStateBytes(branch));
  }

  void Fail(std::string what) {
    Say(log_, what);
    if (failure_.empty()) {
      failure_ = std::move(what);
    }
  }

  // D-102's hang recovery, rung 2: a unit's work hung and its cancellation
  // drained (the node's waits flag it). The requests that needed it fail
  // with a 503 to retry, and the model is reset in place (Server::
  // RecoverModel); when that cannot be done, rung 3 (the ladder's last
  // resort restarts the process). `what` names the work.
  api::Error RecoverFromHang(Served& m, std::string_view what) {
    // Only once the cancellation drained: a read the drive still holds
    // would land in what the reset frees (a stuck read is rung 3's).
    if (!AwaitDrained()) {
      Fail(std::format("{}'s hung work did not drain after its cancellation", m.name()));
      return HungFailure();  // reached only when the last resort returns (a test's)
    }
    if (ladder_ != nullptr) {
      if (auto again = ladder_->Resetting(m.name()); again) {
        ladder_->Restart(*again);
        Fail(*again);  // reached only when the last resort returns (a test's)
        return HungFailure();
      }
    }
    Say(log_, std::format("hang recovery, rung 2 (reset the model in place): {} failed after its "
                          "hung work was cancelled and drained; {} is reset in place",
                          what, m.name()));
    if (auto recovered = server_.RecoverModel(m); !recovered) {
      const std::string why = std::format(
          "rung 2 could not reset model {} in place ({}): nothing less than a restart frees it",
          m.name(), recovered.error());
      if (ladder_ != nullptr) {
        ladder_->Restart(why);
      }
      Fail(why);  // reached only when the last resort returns (a test's): the service stops
    } else {
      Say(log_, std::format("hang recovery, rung 2: model {} was reset in place and evicted; its "
                            "next request loads it whole, and the service goes on",
                            m.name()));
    }
    return HungFailure();
  }
  // After a hang's cancellation (rung 1), the driver waits until it drained
  // (the ladder watches again: the driver moved and nothing the work
  // submitted is still in flight) before it frees or reuses anything the
  // work touched. False when it did not drain within its grace: the ladder
  // entered rung 3 (whose last resort exits; a test's returns).
  bool AwaitDrained() {
    if (ladder_ == nullptr) {
      return true;
    }
    while (ladder_->rung() == HangLadder::Rung::kCancel) {
      const auto now = Clock::now();
      ladder_->Activity(server_.progress(), now);
      (void)ladder_->Check(now);
      if (ladder_->rung() != HangLadder::Rung::kCancel) {
        break;
      }
      std::this_thread::sleep_for(kDrainPoll);
    }
    return ladder_->rung() != HangLadder::Rung::kRestart;
  }
  static api::Error HungFailure() {
    return Failure(503,
                   "the model's work hung and was cancelled, and the model was reset: retry the "
                   "request",
                   "backend_hung");
  }
  // A cooperative unit's hang (rung 2 in a cohort): every member fails with
  // a 503; the model is reset once the last of them retires (Retire), after
  // a fence of its stream proves their native references retired.
  std::expected<void, std::string> HangCohort(std::string_view what) {
    hang_reset_ = cohort_model_;
    std::size_t members = 0;
    for (ChatWork* frame : cohort_) {
      if (frame != nullptr && !frame->retired) {
        frame->error = HungFailure();
        ++members;
      }
    }
    selected_prompt_ = nullptr;
    selected_prefill_count_ = 0;
    selected_decode_.clear();
    Say(log_, std::format("hang recovery, rung 2 (reset the model in place): {} failed after its "
                          "hung work was cancelled; its {} requests fail, and {} is reset once "
                          "they retire and the cancellation drained",
                          what, members, cohort_model_ != nullptr ? cohort_model_->name() : ""));
    return {};
  }

  Server& server_;
  const config::NodeConfig& config_;
  std::shared_ptr<RequestMemory> memory_;  // the request memory (intake_limits.h)
  std::FILE* log_;
  HangLadder* ladder_ = nullptr;
  // The model a cohort's hang left to reset once its members retire.
  Llm* hang_reset_ = nullptr;
  SwapParts parts_;
  bool swapped_ = false;
  std::string failure_;  // a node failure: the service stops
  std::array<ChatWork*, Llm::kMaxBranches> cohort_{};
  Llm* cohort_model_ = nullptr;
  ChatWork* selected_prompt_ = nullptr;
  std::array<ChatWork*, Llm::kMaxBranches> selected_prefill_{};
  std::size_t selected_prefill_count_ = 0;
  std::vector<ChatWork*> selected_decode_;
  std::uint64_t admissions_ = 0;
  bool selected_swap_ = false;
  bool cohort_native_touched_ = false;
  // The queue's head found no memory for a request slot: it waits until a
  // member retires (Start, Retire).
  MemoryWait memory_wait_;
};

// The reverse lookups' time at startup, all of them together: MagicDNS
// answers in milliseconds, and a resolver that does not answer costs no
// more than this (its names are then not accepted; the start log says
// which are).
constexpr auto kLookupBudget = std::chrono::seconds(5);

// `[client] bind` against the node's addresses (binding.h), its notes
// logged (its unauthenticated listeners are, once it listens).
std::expected<api::Listening, std::string> ResolveBind(const config::ClientConfig& client,
                                                       std::FILE* log) {
  auto addresses = platform::ReadInterfaceAddresses();
  if (!addresses) {
    return std::unexpected("the chat route cannot read the node's addresses: " + addresses.error());
  }
  const auto by = std::chrono::steady_clock::now() + kLookupBudget;
  const api::ReverseLookup reverse =
      [by](const platform::InterfaceAddress& a) -> std::optional<std::string> {
    const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(
        by - std::chrono::steady_clock::now());
    if (left.count() <= 0) {
      return std::nullopt;
    }
    return platform::ReverseName(a, left);
  };
  api::Listening listening =
      api::ResolveListening(client, *addresses, platform::HostName(), reverse);
  for (const std::string& note : listening.notes) {
    Say(log, note);
  }
  if (listening.endpoints.empty()) {
    return std::unexpected("[client] bind resolves to no address to listen on");
  }
  return listening;
}

std::string HostNames(const api::HostGuard& hosts) {
  std::string names;
  for (const std::string& name : hosts.names()) {
    names += (names.empty() ? "" : ", ") + name;
  }
  return names.empty() ? "(none)" : names;
}

}  // namespace

int RunService(const config::NodeConfig& config, const config::RuntimeRoles& roles, std::FILE* log,
               bool gemma_joined, bool gemma_row_invariant, bool gemma31_production) {
  // Speculative where there is a drafter; no image prompt; conversations
  // kept across a restart (D-105).
  ServingOptions serving;
  serving.keep_conversations = true;
  serving.gemma_joined = gemma_joined;
  serving.gemma_row_invariant = gemma_row_invariant;
  serving.gemma31_production = gemma31_production;
  int status = kExitOk;
  // Hang recovery (D-102; hang_ladder.h): one ladder for the chat route's
  // watch and the node's waits, outliving both (the teardown's waits are
  // watched too). Its last resort exits for the supervisor; nothing is torn
  // down (the driver may be the thread that hangs).
  const config::ClientConfig& client_limits = config.client;
  const std::chrono::milliseconds hang =
      client_limits.hang_seconds
          ? std::chrono::milliseconds(std::chrono::seconds(*client_limits.hang_seconds))
          : api::DefaultHang(std::chrono::seconds(client_limits.stall_seconds));
  HangLadder ladder(hang, Clock::now());
  ladder.set_log([log](std::string_view line) { Say(log, std::string(line)); });
  // What can be kept is kept first, briefly (D-105): the conversation
  // records already queued are written (their spill files are whole on
  // disk); nothing new is spilled, since the device may be what hangs.
  std::atomic<Server*> running{nullptr};
  ladder.set_last_resort([log, &running](const std::string& /*why*/) {
    std::ignore = platform::NotifyServiceManager("STATUS=exiting after a confirmed hang");
    if (Server* server = running.load(); server != nullptr) {
      const bool kept = server->DrainKept(Clock::now() + kLastResortKeep);
      Say(log, kept ? "the conversation records queued were written"
                    : "some conversation records queued were not written in time");
    }
    Say(log, "exiting for the supervisor to restart the runtime (hang recovery's last resort)");
    std::_Exit(kExitFailure);
  });
  LadderPatience patience(ladder);
  // This thread is the driver: its long CPU work beats the ladder's pulse
  // and stops when rung 1 asks it to (base/work_pulse.h).
  base::SetThreadPulse(&ladder.pulse());
  // The test hook below (JITLLM_TEST_HOLD_READS): with
  // JITLLM_TEST_HOLD_READS_CANCELLABLE=1 a held read completes as cancelled
  // when the lane cancels it (as a read still queued would); otherwise, as
  // a read a drive holds, it does not.
  // NOLINTNEXTLINE(concurrency-mt-unsafe): read before any thread starts
  if (const char* cancellable = std::getenv("JITLLM_TEST_HOLD_READS_CANCELLABLE");
      cancellable != nullptr && std::string_view(cancellable) == "1") {
    serving.hold_cancellable = true;
  }
  {
    Server server(config, roles, serving, log);
    server.SetPatience(&patience);
    // Each step of the start that made progress extends the service
    // manager's start timeout (jitllm.service's TimeoutStartSec).
    server.set_start_progress([] {
      std::ignore = platform::NotifyServiceManager(
          std::format("EXTEND_TIMEOUT_USEC={}", kStartExtension.count()));
    });
    running.store(&server);
    // A test hook (D-102's hang recovery, tested end to end): with
    // JITLLM_TEST_HOLD_READS naming a file, the node's reads are held while
    // that file exists, as a stuck drive's would be. Unset in service.
    std::jthread hold_watch;
    // NOLINTNEXTLINE(concurrency-mt-unsafe): read before any thread starts
    if (const char* path = std::getenv("JITLLM_TEST_HOLD_READS"); path != nullptr && *path != 0) {
      Say(log, std::format("test hook: the node's reads are held while {} exists ({})", path,
                           serving.hold_cancellable ? "cancellable, as queued reads"
                                                    : "not cancellable, as a drive's"));
      hold_watch = std::jthread([&server, file = std::string(path)](const std::stop_token& stop) {
        while (!stop.stop_requested()) {
          std::error_code error;
          server.HoldReads(std::filesystem::exists(file, error));
          std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        server.HoldReads(false);
      });
    }
    auto started = server.Start(false);
    if (started && (gemma_joined || gemma_row_invariant))
      for (const auto& model : server.models())
        Say(log, std::format("Gemma diagnostic {}: {}", model->name(), model->extra()));
    // A cancellation drains only once nothing submitted before it is still
    // in flight (a read a drive holds does not end when cancelled).
    ladder.set_operations([&server] { return server.node().oldest_io(); });
    std::optional<NodeBackend> backend;
    std::optional<api::Server> http;
    if (started) {
      // The limits that follow memory (D-102), or what [client] sets: the
      // request memory the start reserved beside the guard's margin, a
      // body's most within it and the largest configured context's bytes.
      const IntakeLimits intake =
          DeriveIntakeLimits(server.request_memory(), server.request_capacity(),
                             server.context_body_bytes(), config.client);
      auto memory = std::make_shared<RequestMemory>(intake.request_floor, intake.request_capacity);
      // This thread is the driver: its charges grow the request memory at
      // once, through the reclaim order; the I/O and parse threads' wait
      // for Maintain to grow it (RequestMemory::Settle).
      memory->SetDriver(std::this_thread::get_id(),
                        [&server](std::uint64_t to) { return server.SetRequestMemory(to); });
      backend.emplace(server, config, memory, log, &ladder);
      api::ServerOptions options;
      options.intake = intake;
      options.memory = memory;
      std::vector<std::string> unauthenticated;
      if (auto listening = ResolveBind(config.client, log); !listening) {
        started = std::unexpected(listening.error());
      } else {
        options.bind = std::move(listening->endpoints);
        options.hosts = std::move(listening->hosts);
        unauthenticated = std::move(listening->unauthenticated);
      }
      const std::vector<config::ClientEndpoint> endpoints = options.bind;
      const std::string names = HostNames(options.hosts);
      const config::ClientConfig& client = config.client;
      const auto seconds = [](std::optional<std::uint32_t> s) {
        return s ? std::optional<std::chrono::milliseconds>(std::chrono::seconds(*s))
                 : std::nullopt;
      };
      options.max_queued = client.max_queued;
      options.queue_wait = seconds(client.queue_wait_seconds);
      options.stall = std::chrono::seconds(client.stall_seconds);
      options.stall_fails = client.stall_action == config::StallAction::kFail;
      options.deadline_cap = seconds(client.deadline_cap_seconds);
      options.idle_timeout = std::chrono::seconds(client.idle_seconds);
      options.request_inactivity = std::chrono::seconds(client.request_inactivity_seconds);
      options.write_inactivity = seconds(client.write_inactivity_seconds);
      options.model_turn = std::chrono::seconds(client.model_turn_seconds);
      // A genuine hang is recovered (D-102): the I/O thread feeds the ladder
      // the node's progress (any lane's completion, any wait of the
      // driver's ending) and escalates it; the node's waits cancel at its
      // rung 1, and its last resort restarts the process.
      options.ladder = &ladder;
      options.activity = [&server] { return server.progress(); };
      // The server logs each change of the backend's health; the service
      // manager's status line says it too (`systemctl status`).
      const std::size_t served = backend->Models().size();
      const bool fails = options.stall_fails;
      options.on_health = [log, served, fails](const Health& health) {
        std::string status = std::format("STATUS=serving {} models on the chat route", served);
        if (!health.healthy) {
          status = std::format("STATUS=the model backend is not making progress ({}); {}",
                               PhaseName(health.phase),
                               fails ? "refusing requests until it does" : "requests wait for it");
        }
        if (auto notified = platform::NotifyServiceManager(status); !notified) {
          Say(log, notified.error());
        }
      };
      // Each connection is a descriptor; the rest (model files, the
      // node's own) fit in kReservedFiles. Without max_connections the
      // route keeps what the open-file hard limit allows (D-102).
      const std::uint64_t want = client.max_connections
                                     ? std::uint64_t{*client.max_connections} + kReservedFiles
                                     : std::numeric_limits<std::uint64_t>::max();
      const std::uint64_t files = platform::RaiseOpenFileLimit(want);
      const std::uint64_t fits = ConnectionsFor(files);
      if (client.max_connections && fits < *client.max_connections) {
        Say(log, std::format("the open-file limit is {}: the chat route keeps at most {} "
                             "connections, not the {} configured",
                             files, fits, *client.max_connections));
      }
      options.max_connections = static_cast<std::size_t>(
          client.max_connections ? std::min<std::uint64_t>(*client.max_connections, fits) : fits);
      Say(log, std::format("the chat route's limits: request memory {} bytes set apart, growing "
                           "within the budget to {} bytes{} (bodies, parses, renderings, unread "
                           "output, responses); bodies up to {} bytes; a stream pauses past {} "
                           "unread bytes; {} connections; queue {}; stalls reported after {} s{}; "
                           "a hang confirmed after {} s without progress is recovered (cancel, "
                           "reset the model, restart)",
                           intake.request_floor, intake.request_capacity,
                           client.request_memory_bytes ? " ([client] request_memory_bytes)" : "",
                           intake.max_body, intake.stream_buffer, options.max_connections,
                           client.max_queued ? std::format("at most {}", *client.max_queued)
                                             : std::string("unbounded"),
                           client.stall_seconds, fails ? " (and fail requests)" : "",
                           std::chrono::duration_cast<std::chrono::seconds>(hang).count()));
      options.log = log;
      if (started) {
        http.emplace(*backend, std::move(options));
        if (auto ports = http->Listen(); !ports) {
          started = std::unexpected(std::format("the chat route cannot listen: {}", ports.error()));
        } else {
          std::string where;
          for (std::size_t i = 0; i < endpoints.size() && i < ports->size(); ++i) {
            const config::ClientEndpoint& e = endpoints[i];
            where += std::format("{}{}{}{}:{}", where.empty() ? "" : ", ", e.ipv6 ? "[" : "",
                                 e.address, e.ipv6 ? "]" : "", (*ports)[i]);
          }
          Say(log, std::format("the chat route listens on {} (/v1/chat/completions, /v1/models); "
                               "Host names accepted besides loopback: {}",
                               where, names));
          for (const std::string& line : unauthenticated) {
            Say(log, line);
          }
        }
      }
    }
    if (!started) {
      Say(log, "refusing to serve: " + started.error());
      status = kExitFailure;
    } else if (!backend.has_value() || !http.has_value()) {
      Say(log, "refusing to serve: the chat route was not set up");  // not reached
      status = kExitFailure;
    } else {
      constexpr std::array kWatched = {SIGTERM, SIGINT, SIGHUP, SIGCHLD};
      const platform::SignalWatch signals = platform::SignalWatch::Open(kWatched);
      if (!signals.valid()) {
        Say(log, "cannot watch signals");
        status = kExitFailure;
      } else {
        if (auto notified = platform::NotifyServiceManager(std::format(
                "READY=1\nSTATUS=serving {} models on the chat route", backend->Models().size()));
            !notified) {
          Say(log, notified.error());
        }
        Say(log, "ready");
        const auto on_wake = [&]() {
          bool stop = false;
          for (int signo = signals.Take(); signo != 0; signo = signals.Take()) {
            if (signo == SIGCHLD) {
              (void)platform::ReapExited();
            } else if (signo == SIGHUP) {
              Say(log,
                  "the configuration is read only at startup; restart the runtime to apply a "
                  "change");
            } else {
              stop = true;
            }
          }
          return stop;
        };
        auto ran = http->Run(signals.descriptor(), on_wake);
        if (auto notified = platform::NotifyServiceManager("STOPPING=1"); !notified) {
          Say(log, notified.error());
        }
        if (!ran) {
          Say(log, "stopping after a failure: " + ran.error());
          status = kExitFailure;
        } else {
          Say(log, "stopping");
          // A graceful stop keeps the conversations for the next start
          // (D-105): spilled and recorded, telling the service manager it is
          // still working (each extension a few seconds past the last), so
          // its stop timeout never cuts a stop that makes progress.
          const auto extend = [] {
            std::ignore = platform::NotifyServiceManager(
                std::format("EXTEND_TIMEOUT_USEC={}", kStopExtension.count()));
          };
          extend();
          server.Persist(Clock::now() + kPersistLongest, extend);
          extend();  // and the teardown that follows
        }
      }
    }
    http.reset();
    for (const auto& model : server.models())
      if (model->settings().architecture == "gemma3")
        Say(log, std::format("Gemma3 serving final {}: {}", model->name(), model->extra()));
      else if (model->settings().architecture == "gemma2")
        Say(log, std::format("Gemma2 serving final {}: {}", model->name(), model->extra()));
    if (gemma_joined || gemma_row_invariant)
      for (const auto& model : server.models())
        Say(log, std::format("Gemma diagnostic final {}: {}", model->name(), model->extra()));
    if (auto stopped = server.TearDown(); !stopped) {
      Say(log, "stopping: " + stopped.error());
      status = kExitFailure;
    }
    ladder.set_operations({});  // the server goes
    running.store(nullptr);
  }
  base::SetThreadPulse(nullptr);  // the ladder goes
  return status;
}

}  // namespace jitllm::runtime
