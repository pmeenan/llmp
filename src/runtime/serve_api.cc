// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// The service with models configured (D-097; docs/runtime-serving.md#the-chat-route):
// the configured models registered on the node (serving.h), the chat
// route (api_server.h) over them, where [client] bind says (binding.h), and
// the runtime's signals watched as readiness (platform/event_loop.h) on the
// node's driver thread, which runs every request.

#include <algorithm>
#include <array>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <expected>
#include <format>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "base/check.h"
#include "base/report.h"
#include "chat/chat.h"
#include "config/node_config.h"
#include "platform/event_loop.h"
#include "platform/interfaces.h"
#include "platform/job.h"
#include "platform/sd_notify.h"
#include "platform/sockets.h"
#include "runtime/api.h"
#include "runtime/api_server.h"
#include "runtime/binding.h"
#include "runtime/cohort_capacity.h"
#include "runtime/commands.h"
#include "runtime/completion_tokens.h"
#include "runtime/prefill.h"
#include "runtime/runtime.h"
#include "runtime/serving.h"
#include "runtime/watchdog.h"
#include "tokenizer/tokenizer.h"

namespace jitllm::runtime {
namespace {

void Say(std::FILE* log, std::string_view text) {
  const std::string line = std::format("jitllm-runtime: {}\n", base::Printable(text));
  (void)std::fwrite(line.data(), 1, line.size(), log);
  (void)std::fflush(log);
}

api::Error Failure(int status, std::string message, std::string code = {}, std::string param = {}) {
  return api::Error{.status = status,
                    .type = status >= 500 ? "server_error" : "invalid_request_error",
                    .message = std::move(message),
                    .param = std::move(param),
                    .code = std::move(code)};
}

std::uint64_t RandomSeed() {
  std::uint64_t seed = 0;
  if (!platform::FillRandom(std::as_writable_bytes(std::span(&seed, 1)))) {
    seed = static_cast<std::uint64_t>(Clock::now().time_since_epoch().count());
  }
  return seed;
}

struct ChatPrompt {
  std::vector<std::int32_t> tokens;
  std::uint32_t stable_boundary = 0;
  std::uint32_t max_tokens = 0;
  bool reasoning = false;
};

std::expected<ChatPrompt, api::Error> PrepareChat(Llm& model, const api::ChatRequest& request) {
  chat::Conversation conversation;
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
  ChatPrompt result;
  auto rendered = model.RenderChat(conversation, &result.stable_boundary);
  if (!rendered) {
    return std::unexpected(
        Failure(400, "the conversation cannot be rendered: " + rendered.error(), {}, "messages"));
  }
  result.tokens = std::move(*rendered);
  const std::uint32_t usable = model.usable_context();
  const auto prompt = static_cast<std::uint32_t>(result.tokens.size());
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
    const auto marker =
        std::ranges::find_if(result.tokens.rbegin(), result.tokens.rend(), [&](std::int32_t token) {
          return token == *model.think_start() || token == *model.think_end();
        });
    result.reasoning = marker != result.tokens.rend() && *marker == *model.think_start();
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

 private:
  Llm& model_;
  api::Exchange& exchange_;
  tokenizer::StreamDecoder decoder_;
  bool thinking_ = false;
  bool any_content_ = false;
};

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
             Floors floors, std::uint64_t swap_bytes)
        : owner(owner),
          model(model),
          branch(branch),
          slot(slot),
          request(request),
          exchange(exchange),
          rendered(std::move(rendered)),
          floors(floors),
          swap_bytes(swap_bytes),
          output(model, exchange, this->rendered.reasoning) {
      options.max_tokens = this->rendered.max_tokens;
      options.stop = true;
      if (request.temperature > 0) {
        options.sampling =
            execution::SamplingParams{.temperature = static_cast<float>(request.temperature),
                                      .top_k = request.top_k,
                                      .top_p = static_cast<float>(request.top_p),
                                      .min_p = static_cast<float>(request.min_p)};
        options.seed = request.seed.value_or(RandomSeed());
      }
      options.on_tokens = [this](std::span<const std::int32_t> fresh) {
        return output.Push(fresh);
      };
    }
    ~ChatWork() override {
      base::Check(retired, "a cooperative chat owner was freed before retirement");
    }
    ChatWork(const ChatWork&) = delete;
    ChatWork& operator=(const ChatWork&) = delete;
    ChatWork(ChatWork&&) = delete;
    ChatWork& operator=(ChatWork&&) = delete;
    bool terminal() const override {
      return cancelled || error.has_value() || stage == Stage::kDone ||
             (generation_session != nullptr && generation_session->done());
    }
    void Cancel() override { cancelled = true; }

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
    GenerateOptions options;
    Generation generation;
    std::unique_ptr<Llm::PromptSession> prompt_session;
    std::unique_ptr<Llm::GenerationSession> generation_session;
    std::optional<api::Error> error;
    Stage stage = Stage::kPrompt;
    std::uint32_t reused = 0;
    bool cancelled = false;
    bool native_touched = false;
    bool retired = false;
    // State capacity (cohort_capacity.h): admission order, whether it
    // waits, the refusal it fails with if it cannot fit alone, and after a
    // preemption mid-generation, the tokens its state is rebuilt from (the
    // prompt and every generated token but the unprocessed anchor).
    std::uint64_t admitted = 0;
    CapacityWait wait = CapacityWait::kNone;
    std::string refusal;
    bool resume = false;
    std::vector<std::int32_t> resume_tokens;
  };
  static_assert(kCohortSlots == Llm::kMaxBranches);

 public:
  NodeBackend(Server& server, const config::NodeConfig& config, std::FILE* log)
      : server_(server), config_(config), log_(log) {}

  // Only models with funded independent slots opt in. Literal completion,
  // scoring and unsupported models retain their ordinary entry points.
  api::CooperativeBackend* cooperative() override { return this; }

  bool Supports(const api::CooperativeBackend::Request& request) const override {
    if (request.literal != nullptr) {
      return false;
    }
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
    auto rendered = PrepareChat(model, request.options);
    if (!rendered) {
      return std::unexpected(rendered.error());
    }
    // Choose only an unclaimed model-owned branch. Prefix matching is a
    // reuse preference, never conversation identity or retention policy.
    std::size_t slot = cohort_.size();
    std::size_t longest = 0;
    for (std::size_t i = 0; i < model.branches() && i < cohort_.size(); ++i) {
      if (cohort_[i] != nullptr) {
        continue;
      }
      auto branch = model.branch(i);
      if (!branch) {
        return std::unexpected(Failure(500, "the model's conversation branch is unavailable"));
      }
      const auto& history = (*branch)->history();
      const auto end = std::ranges::mismatch(history, rendered->tokens).in1;
      const auto common = static_cast<std::size_t>(end - history.begin());
      if (slot == cohort_.size() || common > longest) {
        slot = i;
        longest = common;
      }
    }
    if (slot == cohort_.size()) {
      return std::unique_ptr<api::CooperativeBackend::Work>{};
    }
    auto branch = model.branch(slot);
    if (!branch) {
      return std::unexpected(Failure(500, "the model's conversation branch is unavailable"));
    }
    const Floors floors = ModelFloors(model);
    const auto swap_bytes = SwapBytes(model);
    auto frame = std::make_unique<ChatWork>(*this, model, **branch, slot, request.options, exchange,
                                            std::move(*rendered), floors, swap_bytes);
    auto prompt =
        frame->branch.BeginPrompt(frame->rendered.tokens, frame->rendered.stable_boundary);
    if (!prompt) {
      frame->retired = true;  // no session or native work retained a reference
      return std::unexpected(Failure(500, "the prompt could not be admitted: " + prompt.error()));
    }
    frame->prompt_session = std::move(*prompt);
    if (!exchange.Admit({.prompt_tokens = static_cast<std::uint32_t>(frame->rendered.tokens.size()),
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
    selected_decode_.clear();
    selected_swap_ = server_.resident() != cohort_model_;
    if (selected_swap_) {
      return api::CooperativeBackend::Unit{
          .phase = Phase::kSwap,
          .expected_seconds =
              static_cast<double>(SwapBytes(*cohort_model_)) / kSwapFloorBytesPerSecond};
    }
    ChatWork* prompt = nullptr;
    std::size_t prompt_distance = cohort_.size();
    for (auto* base : work) {
      auto& frame = static_cast<ChatWork&>(*base);
      if (frame.wait != CapacityWait::kNone) {
        continue;  // waits for state capacity (cohort_capacity.h)
      }
      if (frame.stage == ChatWork::Stage::kGeneration) {
        selected_decode_.push_back(&frame);
      } else if (frame.stage == ChatWork::Stage::kPrompt) {
        const auto distance = (frame.slot + cohort_.size() - next_prompt_slot_) % cohort_.size();
        if (distance < prompt_distance) {
          prompt = &frame;
          prompt_distance = distance;
        }
      }
    }
    if (!selected_decode_.empty() && (prefer_decode_ || prompt == nullptr)) {
      double expected = 0;
      for (const ChatWork* frame : selected_decode_) {
        expected +=
            ExpectedSeconds(Phase::kDecode, frame->model.speculative() ? 4 : 1, frame->floors);
      }
      return api::CooperativeBackend::Unit{.phase = Phase::kDecode, .expected_seconds = expected};
    }
    selected_decode_.clear();
    if (prompt == nullptr) {
      return std::unexpected("an active chat cohort has no next completed unit");
    }
    auto unit = prompt->prompt_session->NextUnit();
    if (!unit) {
      return std::unexpected(unit.error());
    }
    selected_prompt_ = prompt;
    double expected = ExpectedSeconds(Phase::kPrefill, unit->rows, prompt->floors);
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
    if (selected_swap_) {
      selected_swap_ = false;
      bool needed = false;
      for (auto* base : work) {
        auto& frame = static_cast<ChatWork&>(*base);
        if (!frame.exchange.Next(Phase::kSwap, frame.swap_bytes)) {
          frame.Cancel();
        } else {
          needed = true;
          frame.native_touched = true;
        }
      }
      if (!needed) {
        return {};
      }
      cohort_native_touched_ = true;
      swapped_ = server_.resident() != cohort_model_;
      if (auto activated = server_.Activate(*cohort_model_, parts_); !activated) {
        Fail("making the cooperative chat model resident: " + activated.error());
        return activated;
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
      Fail("selecting the native chat cohort: " + selected.error());
      return selected;
    }
    if (selected_prompt_ != nullptr) {
      ChatWork& frame = *std::exchange(selected_prompt_, nullptr);
      frame.native_touched = true;
      next_prompt_slot_ = (frame.slot + 1) % cohort_.size();
      prefer_decode_ = true;
      const PrefillGoOn go_on = [&frame](std::uint32_t rows) {
        return !frame.cancelled && frame.exchange.Next(Phase::kPrefill, rows);
      };
      if (auto advanced = frame.prompt_session->Advance(go_on, true); !advanced) {
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
    prefer_decode_ = false;
    if (sessions.empty()) {
      return {};
    }
    if (auto advanced = cohort_model_->RunGenerationWave(sessions, true); !advanced) {
      Fail("the native chat generation wave failed: " + advanced.error());
      return advanced;
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
      if (!frame.resume) {
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
    if (cohort_native_touched_) {
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
    if (!frame.error && !frame.generation.tokens.empty()) {
      frame.output.Finish();  // including a generation preempted and not yet resumed
    }
    api::Completion result{
        .completion_tokens = static_cast<std::uint32_t>(frame.generation.tokens.size()),
        .cached_tokens = frame.reused,
        .stopped = frame.generation.stopped};
    cohort_[frame.slot] = nullptr;
    frame.retired = true;
    frame.generation_session.reset();
    frame.prompt_session.reset();
    if (peers == 1) {
      cohort_model_ = nullptr;
      cohort_native_touched_ = false;
      selected_prompt_ = nullptr;
      selected_decode_.clear();
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
        models.push_back({.name = entry.name, .chat = false, .context = 0});
        continue;
      }
      Served* m = server_.Find(entry.name);
      if (m != nullptr && m->llm()) {
        models.push_back(
            {.name = entry.name, .chat = true, .context = static_cast<Llm&>(*m).usable_context()});
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
    auto rendered = PrepareChat(l, request);
    if (!rendered) {
      return std::unexpected(rendered.error());
    }
    const std::vector<std::int32_t>& tokens = rendered->tokens;
    const auto prompt = static_cast<std::uint32_t>(tokens.size());
    const std::uint32_t max_tokens = rendered->max_tokens;
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
    Floors floors;
    if (const auto entry =
            std::ranges::find(config_.models, request.model, &config::ModelEntry::name);
        entry != config_.models.end()) {
      floors = {.prefill = entry->prefill_floor_tok_s, .decode = entry->decode_floor_tok_s};
    }
    if (!exchange.Admit({.prompt_tokens = prompt,
                         .max_tokens = max_tokens,
                         .swap_bytes = swap_bytes,
                         .floors = floors})) {
      return api::Completion{};
    }

    swapped_ = server_.resident() != m;
    if (swapped_ && !exchange.Next(Phase::kSwap, swap_bytes)) {
      swapped_ = false;
      return api::Completion{};
    }
    if (auto r = server_.Activate(*m, parts_); !r) {
      Fail(std::format("making {} resident: {}", m->name(), r.error()));
      swapped_ = false;
      return std::unexpected(Failure(503, "the model could not be made resident"));
    }
    // A swap is one program (about 10 s on a Spark), watched as one unit
    // at its bytes; whatever ended the request meanwhile (the client gone,
    // the backend stalled, the deadline, the runtime stopping) ends it
    // here, before any of its model work. The exchange answers for it.
    if (!exchange.Continue()) {
      return api::Completion{};
    }

    GenerateOptions options{.max_tokens = max_tokens,
                            .stop = true,
                            .keep_logits = false,
                            .sampling = std::nullopt,
                            .seed = 0,
                            .on_tokens = {}};
    if (request.temperature > 0) {
      options.sampling =
          execution::SamplingParams{.temperature = static_cast<float>(request.temperature),
                                    .top_k = request.top_k,
                                    .top_p = static_cast<float>(request.top_p),
                                    .min_p = static_cast<float>(request.min_p)};
      options.seed = request.seed.value_or(RandomSeed());
    }
    ChatOutput output(l, exchange, rendered->reasoning);
    options.on_tokens = [&](std::span<const std::int32_t> fresh) { return output.Push(fresh); };

    Generation generation;
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
      if (auto r =
              l.PreparePrompt(tokens, rendered->stable_boundary, last, reused, go_on, &prefill);
          !r) {
        return capacity(r, "the prompt could not be processed: ");
      }
      if (prefill.stopped || !exchange.Next(Phase::kDecode, 0)) {
        return {};  // the state holds what ran; the exchange answers for why
      }
      return capacity(l.Generate(last, options, generation), "the generation failed: ");
    });
    if (!ran) {
      Fail(std::format("{}'s request: {}", m->name(), ran.error()));
      return std::unexpected(Failure(500, "the generation failed; the runtime is stopping"));
    }
    if (capacity.refusal()) {
      RefusedAlone(l);
      return std::unexpected(Failure(500, *capacity.refusal()));
    }
    if (prefill.stopped) {
      // Token counts and times only (D-014); the exchange answers for why.
      Say(log_, std::format("{}'s prefill stopped after {} chunks: the state holds {} of the "
                            "prompt's {} tokens",
                            m->name(), prefill.chunks, l.history().size(), tokens.size()));
      return api::Completion{.completion_tokens = 0, .cached_tokens = reused, .stopped = false};
    }
    output.Finish();
    return api::Completion{
        .completion_tokens = static_cast<std::uint32_t>(generation.tokens.size()),
        .cached_tokens = reused,
        .stopped = generation.stopped};
  }

  std::expected<api::Completion, api::Error> Complete(const api::CompletionRequest& request,
                                                      api::Exchange& exchange) override {
    const auto& controls = request.options;
    Served* model = server_.Find(controls.model);
    if (model == nullptr || !model->llm()) {
      return std::unexpected(Failure(404, "The model does not exist", "model_not_found", "model"));
    }
    auto& llm = static_cast<Llm&>(*model);
    auto prompt = api::PrepareLiteralPrompt(request, llm.tokenizer(), llm.usable_context());
    if (!prompt) {
      return std::unexpected(prompt.error());
    }
    const auto& tokens = prompt->tokens;
    const auto max_tokens = controls.max_tokens.value_or(16);
    const bool score_prompt = request.prompt_logprobs || (request.echo && request.logprobs);
    api::Completion result;
    result.literal.prompt_text = std::move(prompt->text);
    std::size_t budget = 1024 + (6 * result.literal.prompt_text.size());
    if (budget > api::kMaxCompletionResponseBytes) {
      return std::unexpected(Failure(413, "prompt echo exceeds the response size", {}, "prompt"));
    }
    std::uint64_t swap_bytes = 0;
    if (server_.resident() != model) {
      swap_bytes = model->weight_read_bytes() + llm.state_snapshot_bytes();
      if (auto* out = server_.resident(); out != nullptr && out->llm()) {
        swap_bytes += static_cast<Llm&>(*out).state_snapshot_bytes();
      }
    }
    Floors floors;
    if (const auto entry =
            std::ranges::find(config_.models, controls.model, &config::ModelEntry::name);
        entry != config_.models.end()) {
      floors = {.prefill = entry->prefill_floor_tok_s, .decode = entry->decode_floor_tok_s};
    }
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
    if (auto activated = server_.Activate(*model, parts_); !activated) {
      Fail("making the literal completion model resident: " + activated.error());
      return std::unexpected(Failure(503, "the model could not be made resident"));
    }
    if (!exchange.Continue()) {
      return api::Completion{};
    }
    std::optional<api::Error> output_problem;
    api::LiteralRows prompt_rows(llm.tokenizer(), request.return_tokens_as_token_ids, true, 0,
                                 budget);
    api::LiteralRows generated_rows(
        llm.tokenizer(), request.return_tokens_as_token_ids, false,
        request.echo ? api::TextCharacters(result.literal.prompt_text) : 0, budget);
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
    if (controls.temperature > 0) {
      options.sampling =
          execution::SamplingParams{.temperature = static_cast<float>(controls.temperature),
                                    .top_k = controls.top_k,
                                    .top_p = static_cast<float>(controls.top_p),
                                    .min_p = static_cast<float>(controls.min_p)};
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
    options.on_tokens = [&](std::span<const std::int32_t> fresh) {
      std::string piece;
      for (const auto id : fresh) {
        if (!decoder.Push(id, piece)) {
          output_problem =
              Failure(500, "a generated token cannot be decoded", "invalid_model_token");
          return false;
        }
      }
      if (6 * piece.size() > api::kMaxCompletionResponseBytes - budget) {
        output_problem =
            Failure(413, "completion text exceeds the response size", "response_too_large");
        return false;
      }
      budget += 6 * piece.size();
      return piece.empty() ? exchange.Continue() : exchange.Content(piece);
    };
    Generation generation;
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
      Fail("the literal completion failed: " + ran.error());
      return std::unexpected(Failure(500, "the completion failed; the runtime is stopping"));
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
      if (6 * rest.size() > api::kMaxCompletionResponseBytes - budget) {
        return std::unexpected(Failure(413, "completion text exceeds the response size",
                                       "response_unrepresentable", "logprobs"));
      }
      (void)exchange.Content(rest);
    }
    result.completion_tokens = static_cast<std::uint32_t>(generation.tokens.size());
    result.stopped = max_tokens == 0 || generation.stopped;
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
        std::format("swap {} -> {}: ready in {:.3f} s (evict {:.3f}, restore {:.3f}, "
                    "page-in {:.3f}, setup {:.3f}; {} graphs dropped); backing released {:.3f} s "
                    "later",
                    parts_.from.empty() ? "(nothing)" : parts_.from, parts_.to, parts_.total,
                    parts_.evict, parts_.restore, parts_.page_in, parts_.setup,
                    parts_.dropped_graphs, parts_.release));
  }

  bool healthy() const override { return failure_.empty(); }
  std::string failure() const override { return failure_; }

 private:
  Floors ModelFloors(const Llm& model) const {
    const auto entry = std::ranges::find(config_.models, model.name(), &config::ModelEntry::name);
    return entry == config_.models.end()
               ? Floors{}
               : Floors{.prefill = entry->prefill_floor_tok_s, .decode = entry->decode_floor_tok_s};
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

  static Status BeginChatGeneration(ChatWork& frame) {
    if (!frame.resume) {
      frame.reused = frame.prompt_session->reused();
    }
    const bool stopped = frame.prompt_session->run().stopped;
    if (auto finished = frame.prompt_session->Finish(); !finished) {
      frame.error = Failure(500, "the prompt failed: " + finished.error());
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
    frame.resume_tokens = {};
    frame.generation_session = std::move(*generation);
    frame.stage = ChatWork::Stage::kGeneration;
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
    const std::optional<std::size_t> idle = LargestIdleState();
    CapacityDecision decision = OnCapacityRefused(cohort, slots, idle.has_value());
    if (decision.reclaim) {
      if (idle && ReleaseIdle(*cohort_model_, *idle)) {
        Apply(cohort);  // the refused members run again
        return;
      }
      cohort = Snapshot();
      decision = OnCapacityRefused(cohort, slots, false);
    }
    Apply(cohort);
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
              ? frame->rendered.tokens.size() + frame->generation.tokens.size() - 1
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

  // The branch outside the cohort retaining the most state (a finished
  // conversation's reuse cache), if any.
  std::optional<std::size_t> LargestIdleState() const {
    std::optional<std::size_t> idle;
    std::uint64_t most = 0;
    for (std::size_t i = 0; i < cohort_model_->branches() && i < cohort_.size(); ++i) {
      if (cohort_[i] != nullptr) {
        continue;
      }
      auto branch = cohort_model_->branch(i);
      if (branch && (*branch)->state_snapshot_bytes() > most) {
        most = (*branch)->state_snapshot_bytes();
        idle = i;
      }
    }
    return idle;
  }

  // Clears that cache for requests refused for capacity: conversation state
  // is never an eviction victim, so a retired conversation's state would
  // otherwise hold its capacity until its branch is reused. Its next turn
  // then prefills from the start. False if the clear failed (logged).
  bool ReleaseIdle(Llm& model, std::size_t slot) {
    auto branch = model.branch(slot);
    if (!branch) {
      return false;
    }
    const std::size_t tokens = (*branch)->history().size();
    if (auto released = (*branch)->ReleaseIdleState(); !released) {
      Say(log_, std::format("{}'s idle conversation state in slot {} could not be cleared: {}",
                            model.name(), slot, released.error()));
      return false;
    }
    Say(log_, std::format("{}'s idle conversation state in slot {} ({} tokens) cleared for a "
                          "request short of state capacity",
                          model.name(), slot, tokens));
    return true;
  }

  // A serial request runs alone, as a cohort of one: refused for capacity,
  // it clears the largest idle branch's retained state and runs again.
  Llm::CapacityReclaim SerialReclaim(Llm& model) {
    return [this, &model](const Llm::Branch& refused) {
      const std::optional<std::size_t> idle = model.LargestIdleBranch(refused);
      return idle.has_value() && ReleaseIdle(model, *idle);
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
        frame.resume = true;
        frame.stage = ChatWork::Stage::kPrompt;
      }
    }
    if (!finished) {
      frame.error = Failure(500, "the request could not release its state: " + finished.error());
      return;
    }
    const std::size_t held = frame.branch.history().size();
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
    const std::vector<std::int32_t>& tokens =
        frame.resume ? frame.resume_tokens : frame.rendered.tokens;
    auto prompt = frame.branch.BeginPrompt(tokens, frame.rendered.stable_boundary);
    if (!prompt) {
      frame.error = Failure(500, "the request could not begin again: " + prompt.error());
      return;
    }
    frame.prompt_session = std::move(*prompt);
    frame.stage = ChatWork::Stage::kPrompt;
  }

  void Fail(std::string what) {
    Say(log_, what);
    if (failure_.empty()) {
      failure_ = std::move(what);
    }
  }

  Server& server_;
  const config::NodeConfig& config_;
  std::FILE* log_;
  SwapParts parts_;
  bool swapped_ = false;
  std::string failure_;  // a node failure: the service stops
  std::array<ChatWork*, Llm::kMaxBranches> cohort_{};
  Llm* cohort_model_ = nullptr;
  ChatWork* selected_prompt_ = nullptr;
  std::vector<ChatWork*> selected_decode_;
  std::size_t next_prompt_slot_ = 0;
  std::uint64_t admissions_ = 0;
  bool prefer_decode_ = true;
  bool selected_swap_ = false;
  bool cohort_native_touched_ = false;
};

// Descriptors the node needs besides the chat route's connections.
constexpr std::uint64_t kReservedFiles = 256;

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

int RunService(const config::NodeConfig& config, const config::RuntimeRoles& roles,
               std::FILE* log) {
  const ServingOptions serving;  // speculative where there is a drafter; no image prompt
  int status = kExitOk;
  {
    Server server(config, roles, serving, log);
    auto started = server.Start(false);
    std::optional<NodeBackend> backend;
    std::optional<api::Server> http;
    if (started) {
      backend.emplace(server, config, log);
      api::ServerOptions options;
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
      options.max_queued = config.client.max_queued;
      options.max_connections = config.client.max_connections;
      options.stall = std::chrono::seconds(config.client.stall_seconds);
      options.deadline_cap = std::chrono::seconds(config.client.deadline_cap_seconds);
      // The server logs each change of the backend's health; the service
      // manager's status line says it too (`systemctl status`).
      const std::size_t served = backend->Models().size();
      options.on_health = [log, served](const Health& health) {
        const std::string status =
            health.healthy ? std::format("STATUS=serving {} models on the chat route", served)
                           : std::format(
                                 "STATUS=the model backend is not making progress ({}); refusing "
                                 "requests until it does",
                                 PhaseName(health.phase));
        if (auto notified = platform::NotifyServiceManager(status); !notified) {
          Say(log, notified.error());
        }
      };
      // Each connection is a descriptor; the rest (model files, the
      // node's own) fit in kReservedFiles.
      const std::uint64_t files =
          platform::RaiseOpenFileLimit(std::uint64_t{options.max_connections} + kReservedFiles);
      if (files < std::uint64_t{options.max_connections} + kReservedFiles) {
        const std::uint64_t fits = files > kReservedFiles + 16 ? files - kReservedFiles : 16;
        Say(log, std::format("the open-file limit is {}: the chat route keeps at most {} "
                             "connections, not the {} configured",
                             files, fits, options.max_connections));
        options.max_connections = static_cast<std::size_t>(fits);
      }
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
        }
      }
    }
    http.reset();
    if (auto stopped = server.TearDown(); !stopped) {
      Say(log, "stopping: " + stopped.error());
      status = kExitFailure;
    }
  }
  return status;
}

}  // namespace jitllm::runtime
