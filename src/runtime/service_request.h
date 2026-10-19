// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
// Paid request rendering, output framing and continuation ownership shared
// by the serial and cooperative service adapters.
#ifndef LLMP_RUNTIME_SERVICE_REQUEST_H_
#define LLMP_RUNTIME_SERVICE_REQUEST_H_
#include <algorithm>
#include <chrono>
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
#include <utility>
#include <vector>

#include "runtime/api_server.h"
#include "runtime/completion_tokens.h"
#include "runtime/serving.h"
#include "tokenizer/tokenizer.h"

namespace llmp::runtime::service_detail {
inline constexpr std::uint64_t kExtentBytes = std::uint64_t{2} << 20U;
void Say(std::FILE* log, std::string_view text);
api::Error Failure(int status, std::string message, std::string code = {}, std::string param = {});

std::optional<api::Error> FundHistory(MemoryCharge& charge, RequestMemory& memory,
                                      std::uint64_t held, bool granted);

std::uint64_t RandomSeed();

// A request's sampling: each field it sent, else the model's default (its
// settings: the checkpoint's own, an override, or OpenAI's; D-103). None
// at temperature 0: greedy.
std::optional<execution::SamplingParams> SamplingOf(const api::ChatRequest& request,
                                                    const Llm& model);

struct ChatPrompt {
  std::shared_ptr<const api::PromptTokens> token_storage;
  const std::vector<std::int32_t>& tokens() const { return token_storage->values(); }
  std::uint32_t stable_boundary = 0;
  std::uint32_t max_tokens = 0;
  bool reasoning = false;
};

// The context's refusal, for a conversation that cannot fit before it is
// rendered or tokenized.
api::Error ContextExceeded(std::uint32_t usable);

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
                                                  api::Exchange& exchange, RequestMemory& memory);

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
    Llm::Branch* branch = nullptr, std::shared_ptr<api::LiteralOutput> literal = nullptr);

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

}  // namespace llmp::runtime::service_detail
#endif  // LLMP_RUNTIME_SERVICE_REQUEST_H_
