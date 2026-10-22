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
  std::uint64_t storage = request.messages.size() * sizeof(chat::Message);
  std::uint64_t projection = storage;
  bool structured = false;
  for (const api::Message& message : request.messages) {
    const std::uint64_t reasoning = message.reasoning ? message.reasoning->capacity() : 0;
    storage += message.content.StorageBytes() + reasoning;
    projection += message.content.ProjectionBytes() + reasoning;
    structured = structured || message.content.parts() != nullptr;
    text += message.content.PayloadBytes() + (message.reasoning ? message.reasoning->size() : 0);
  }
  const bool tools =
      request.tools && request.tools->request.choice().kind != chat::ToolChoiceKind::kNone;
  if (tools && !model.tool_format())
    return std::unexpected(Failure(400, "this template has no verified generated tool-call format",
                                   "tools_unsupported", "tools"));
  if (tools) {
    MemoryCharge names;
    const auto scratch = model.tool_format()->arguments == chat::ToolArguments::kObjectNotation
                             ? request.tools->request.NameBytes()
                             : 0;
    if (!names.Add(memory, scratch))
      return std::unexpected(api::MemoryRefusal(memory, scratch, "validating tool names"));
    if (auto valid = request.tools->request.ValidateNames(*model.tool_format()); !valid)
      return std::unexpected(Failure(400, valid.error(), "invalid_tool_name", "tools"));
    const auto bytes = request.tools->request.TemplateBytes();
    if (bytes > model.render_bytes() || text > model.render_bytes() - bytes)
      return std::unexpected(ContextExceeded(usable));
    text += bytes;
    storage += bytes;  // template-tool handles and temporary projection
  }
  if (text > model.render_bytes()) {
    return std::unexpected(ContextExceeded(usable));
  }
  const RenderCharge render = model.RenderChargeFor(text, request.messages.size());
  const std::uint64_t copied = storage + (structured ? projection : 0);
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
    } else if (message.role == api::Role::kTool) {
      role = chat::Role::kTool;
    }
    conversation.messages.push_back({.role = role,
                                     .content = message.content,
                                     .reasoning_content = message.reasoning,
                                     .tool_calls = {}});
  }
  if (tools) conversation.tools = request.tools->request.TemplateTools();
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
  auto rendered = model.RenderChat(conversation, &result.stable_boundary,
                                   {.max_tokens = usable,
                                    .cancelled = &cancelled,
                                    .memory = &memory,
                                    .memory_needed = &needed,
                                    .reasoning = &result.reasoning},
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
    return std::unexpected(Failure(exchange.timed_out() ? 504 : 503,
                                   exchange.timed_out()
                                       ? "the request timed out while its conversation was rendered"
                                       : "the request ended while its conversation was rendered",
                                   exchange.timed_out() ? "request_timeout" : "request_ended"));
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
  if (!tools && model.think_start() && model.think_end()) {
    const auto marker = std::ranges::find_if(
        result.tokens().rbegin(), result.tokens().rend(), [&](std::int32_t token) {
          return token == *model.think_start() || token == *model.think_end();
        });
    result.reasoning = marker != result.tokens().rend() && *marker == *model.think_start();
  }
  // Output batches and decoder carry survive accepted-and-paused callbacks,
  // independently of whether this request declares tools.
  MemoryCharge output_charge;
  const std::uint64_t output_bytes = sizeof(ChatOutput::State) + sizeof(MemoryCharge) + 128 +
                                     std::uint64_t{result.max_tokens} * 32 +
                                     model.tokenizer().longest_token_bytes() * 4;
  if (!output_charge.Add(memory, output_bytes))
    return std::unexpected(
        api::MemoryRefusal(memory, output_bytes, "the generated output continuation"));
  result.output_charge = std::make_shared<MemoryCharge>(std::move(output_charge));
  if (tools) {
    const auto& format = *model.tool_format();
    const auto kind = request.tools->request.choice().kind;
    if ((kind == chat::ToolChoiceKind::kRequired || kind == chat::ToolChoiceKind::kNamed) &&
        format.arguments == chat::ToolArguments::kJsonEnvelope && !format.bare_envelope)
      return std::unexpected(Failure(400, "this format has no verified tool-name constraint",
                                     "tool_choice_unsupported", "tool_choice"));
    std::size_t generated = 0;
    if (__builtin_mul_overflow(std::size_t{result.max_tokens},
                               model.tokenizer().longest_token_bytes(), &generated))
      return std::unexpected(
          Failure(400, "generated tool output exceeds addressable bytes", {}, "max_tokens"));
    const auto bound = chat::ToolOutputBytes(format, generated);
    std::size_t bytes = 0, ids = 0;
    if (!bound || __builtin_mul_overflow(*bound, std::size_t{4}, &bytes) ||
        __builtin_mul_overflow(std::size_t{result.max_tokens}, std::size_t{32}, &ids) ||
        __builtin_add_overflow(bytes, ids, &bytes) ||
        __builtin_add_overflow(bytes, model.tokenizer().size() * sizeof(float), &bytes))
      return std::unexpected(
          Failure(400, "tool output storage exceeds addressable bytes", {}, "max_tokens"));
    const auto planner = model.tokenizer().ForcedSpanBytes(
        model.tokenizer().longest_token_bytes() + request.tools->request.NameBytes() * 6 +
        format.block_begin.size() + format.call_begin.size() + format.name_end.size());
    if (__builtin_add_overflow(bytes, planner, &bytes))
      return std::unexpected(Failure(400, "tool planning exceeds addressable bytes", {}, "tools"));
    MemoryCharge paid;
    if (!paid.Add(memory, bytes + sizeof(ChatTools) + sizeof(ChatOutput::State)))
      return std::unexpected(api::MemoryRefusal(memory, bytes, "the generated tool-call state"));
    auto owner = std::make_shared<ChatTools>();
    owner->charge = std::move(paid);
    owner->options = request.tools;
    owner->format = format;
    owner->limits = {.max_bytes = generated,
                     .max_depth = generated,
                     .max_values = generated,
                     .max_string_bytes = bytes};
    const auto stops = model.StopTokens();
    std::vector<std::int32_t> terminals(stops.begin(), stops.end());
    const auto extra = model.ChatStops();
    terminals.insert(terminals.end(), extra.begin(), extra.end());
    owner->sampling =
        std::make_shared<chat::ToolSampling>(model.tokenizer(), format, request.tools->request,
                                             owner->limits, result.reasoning, terminals);
    owner->sampling->Budget(result.max_tokens, usable - prompt);
    result.tools = std::move(owner);
  }
  return result;
}

bool ChatOutput::Text(std::string_view text) {
  const bool go = state_->thinking ? exchange_.Reasoning(text) : exchange_.Content(text);
  state_->any_content = state_->any_content || !state_->thinking;
  return go;
}

chat::ToolInterpreter::Flow ChatOutput::Send(chat::ToolEvent event) {
  bool go = true;
  if (const auto* text = std::get_if<chat::TextPart>(&event))
    go = Text(text->text);
  else if (const auto* call = std::get_if<chat::ToolUsePart>(&event))
    go = exchange_.ToolCall(*call);
  else {
    state_->thinking = false;
    go = exchange_.Reasoning({});
  }
  // The sink accepted the event even when the resulting buffered output pauses
  // it. This outcome must not poison the parser like a stop/disconnect does.
  if (exchange_.Yielding() || exchange_.Paused()) return chat::ToolInterpreter::Flow::kPause;
  return go ? chat::ToolInterpreter::Flow::kContinue : chat::ToolInterpreter::Flow::kCancel;
}

bool ChatOutput::pending() const {
  return state_->consumed < state_->unread.size() || (state_->parser && state_->parser->paused()) ||
         (state_->finishing && !state_->finished);
}

bool ChatOutput::Push(std::span<const std::int32_t> fresh) {
  if (state_->error || state_->finished) return false;
  // Before another causal unit the adapters must drain any accepted suffix.
  if (pending()) {
    if (!exchange_.Paused() && !Drain() && !exchange_.Yielding() && !exchange_.Paused())
      return false;
    if (exchange_.Yielding()) return false;
    if (pending()) {
      state_->unread.insert(state_->unread.end(), fresh.begin(), fresh.end());
      return exchange_.Paused();
    }
  }
  state_->unread.assign(fresh.begin(), fresh.end());
  state_->consumed = 0;
  const bool drained = Drain();
  return drained || (!exchange_.Yielding() && exchange_.Paused());
}

bool ChatOutput::Drain() {
  if (state_->error || exchange_.Stopped()) return false;
  const auto sink = [this](chat::ToolEvent event) { return Send(std::move(event)); };
  const auto checked = [&](chat::ToolStatus status) {
    if (!status && !exchange_.Stopped() && !exchange_.Yielding() && !exchange_.Paused())
      state_->error =
          Failure(500, "the generated tool call could not be interpreted: " + status.error(),
                  "tool_output_invalid");
    return status.has_value();
  };
  if (state_->parser && state_->parser->paused()) {
    if (!checked(state_->parser->Resume(sink))) return false;
    if (state_->parser->paused()) return false;
    if (state_->parser->ended()) {
      state_->finished = true;
      return !state_->error;
    }
  }
  while (state_->consumed < state_->unread.size()) {
    const auto token = state_->unread[state_->consumed++];
    std::string piece;
    if (state_->tools) {
      const auto& sampling = *state_->tools->sampling;
      if (model_.tokenizer().Kind(token) == tokenizer::TokenKind::kControl &&
          !sampling.Preserves(token))
        continue;
      if (auto decoded = state_->decoder.Push(token, piece); !decoded) {
        state_->error = Failure(500, "generated token could not be decoded", "tool_output_invalid");
        return false;
      }
      if (!checked(state_->parser->Push(piece, sink)) || state_->parser->paused()) return false;
    } else {
      if (state_->thinking && model_.think_end() && token == *model_.think_end()) {
        state_->decoder.Finish(piece);
        if (!Text(piece)) return false;
        state_->thinking = false;
      } else if (!state_->thinking && !state_->any_content && model_.think_start() &&
                 token == *model_.think_start()) {
        state_->thinking = true;
      } else {
        std::ignore = state_->decoder.Push(token, piece);
        if (!piece.empty() && !Text(piece)) return false;
      }
    }
    if (exchange_.Yielding() || exchange_.Paused()) return false;
  }
  state_->unread.clear();
  state_->consumed = 0;
  if (state_->finishing && !state_->finished) return Finish(*state_->finishing);
  return exchange_.Continue();
}

bool ChatOutput::Finish(bool stopped) {
  if (state_->finished) return !state_->error;
  state_->finishing = stopped;
  if (state_->consumed < state_->unread.size() || (state_->parser && state_->parser->paused()))
    return Drain();
  if (exchange_.Stopped()) {
    state_->finished = true;
    return true;
  }
  const auto sink = [this](chat::ToolEvent event) { return Send(std::move(event)); };
  std::string rest;
  if (!state_->decoder_finished) {
    state_->decoder_finished = true;
    state_->decoder.Finish(rest);
  }
  if (state_->parser) {
    auto status = rest.empty() ? chat::ToolStatus{} : state_->parser->Push(rest, sink);
    if (status && !state_->parser->paused())
      status = state_->parser->Finish(stopped ? chat::ToolFinish::kStop : chat::ToolFinish::kLength,
                                      sink);
    if (!status && !exchange_.Stopped())
      state_->error =
          Failure(500, "the generated tool call could not be interpreted: " + status.error(),
                  "tool_output_invalid");
    if (state_->parser->paused()) return false;
  } else if (!rest.empty() && !Text(rest))
    return false;
  state_->finished = true;
  return !state_->error;
}

std::expected<std::shared_ptr<ChatResume>, api::Error> KeepContinuation(
    const ChatPrompt& prompt, const Generation& generation, const GenerateOptions& options,
    std::shared_ptr<ChatOutput::State> output, std::uint32_t reused, RequestMemory& memory,
    Llm::Branch* branch, std::shared_ptr<api::LiteralOutput> literal) {
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
