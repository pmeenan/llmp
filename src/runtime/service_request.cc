// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#include "runtime/service_request.h"

#include "base/report.h"
#include "base/work_pulse.h"
#include "platform/sockets.h"

namespace llmp::runtime::service_detail {
void Say(std::FILE* log, std::string_view text) {
  const std::string line = std::format("llmp-runtime: {}\n", base::Printable(text));
  (void)std::fwrite(line.data(), 1, line.size(), log);
  (void)std::fflush(log);
}

api::Error Failure(int status, std::string message, std::string code, std::string param) {
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

api::Error ContextExceeded(std::uint32_t usable) {
  return Failure(400,
                 std::format("This model's maximum context length is {} tokens. However, your "
                             "messages resulted in more than {} tokens. Please reduce the length "
                             "of the messages.",
                             usable, usable),
                 "context_length_exceeded", "messages");
}

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

std::expected<std::shared_ptr<ChatResume>, api::Error> KeepContinuation(
    const ChatPrompt& prompt, const Generation& generation, const GenerateOptions& options,
    ChatOutput::State output, std::uint32_t reused, RequestMemory& memory, Llm::Branch* branch,
    std::shared_ptr<api::LiteralOutput> literal) {
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

}  // namespace llmp::runtime::service_detail
