// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "runtime/completion_tokens.h"

#include <algorithm>
#include <format>
#include <utility>

#include "base/check.h"
#include "execution/sampling.h"
#include "tokenizer/unicode.h"

namespace jitllm::runtime::api {
namespace {

Error BadPrompt(std::string message, std::string code = {}) {
  return {.status = 400,
          .type = "invalid_request_error",
          .message = std::move(message),
          .param = "prompt",
          .code = std::move(code)};
}

std::string TokenText(const tokenizer::Tokenizer& tokenizer, std::int32_t id, bool as_id) {
  if (as_id) {
    return std::format("token_id:{}", id);
  }
  std::string bytes;
  if (!tokenizer.Decode(std::span(&id, 1), {.control_tokens = true}, bytes)) {
    return std::format("token_id:{}", id);
  }
  return tokenizer::unicode::ReplaceInvalidUtf8(bytes);
}

}  // namespace

std::size_t TextCharacters(std::string_view text) {
  return static_cast<std::size_t>(
      std::ranges::count_if(text, [](unsigned char c) { return (c & 0xC0U) != 0x80U; }));
}

std::expected<LiteralPrompt, Error> PrepareLiteralPrompt(const CompletionRequest& request,
                                                         const tokenizer::Tokenizer& tokenizer,
                                                         std::uint32_t context,
                                                         std::size_t response_bytes,
                                                         RequestMemory* memory) {
  // Top scores up to the model's vocabulary (D-102); the response's bytes
  // bound how many rows may carry them (below).
  for (const auto& [name, count] : {std::pair{"logprobs", request.logprobs},
                                    std::pair{"prompt_logprobs", request.prompt_logprobs}}) {
    if (count && std::cmp_greater(*count, tokenizer.size())) {
      return std::unexpected(
          Error{.status = 400,
                .type = "invalid_request_error",
                .message = std::format("{} must be at most the model's vocabulary size, {}", name,
                                       tokenizer.size()),
                .param = name,
                .code = {}});
    }
  }
  LiteralPrompt prompt{.tokens = request.token_ids, .text = {}, .added_bos = false};
  if (request.prompt) {
    // The text's bytes are the body's; its tokens the context's (D-102),
    // and its tokenization's working set the request memory's.
    MemoryCharge charge;
    if (memory != nullptr) {
      const std::uint64_t need = tokenizer::Tokenizer::EncodeWorkingBytes(*request.prompt) +
                                 (std::uint64_t{2} * sizeof(std::int32_t) *
                                  std::min<std::uint64_t>(request.prompt->size() + 1, context));
      if (!charge.Add(*memory, need)) {
        return std::unexpected(MemoryRefusal(*memory, need, "tokenizing the prompt"));
      }
    }
    auto encoded = tokenizer.Encode(*request.prompt,
                                    {.special = tokenizer::SpecialTokens::kParse,
                                     .add_bos_eos = false,
                                     .max_bytes = request.prompt->size(),
                                     .max_tokens = context},
                                    prompt.tokens);
    if (!encoded && encoded.error().rule == tokenizer::Rule::kOutputTooLarge) {
      return std::unexpected(BadPrompt(
          std::format("the prompt is longer than the model's context of {} tokens", context),
          "context_length_exceeded"));
    }
    if (!encoded) {
      return std::unexpected(BadPrompt("the literal prompt cannot be tokenized"));
    }
    const auto bos = tokenizer.bos();
    if (request.add_special_tokens && tokenizer.adds_bos() && bos &&
        (prompt.tokens.empty() || prompt.tokens.front() != *bos)) {
      prompt.tokens.insert(prompt.tokens.begin(), *bos);
      prompt.added_bos = true;
    }
  }
  for (const auto id : prompt.tokens) {
    if (id < 0 || std::cmp_greater_equal(id, tokenizer.size()) ||
        tokenizer.Kind(id) == tokenizer::TokenKind::kUnused) {
      return std::unexpected(
          BadPrompt("a prompt token ID is outside the model vocabulary", "invalid_token_id"));
    }
  }
  const auto max_tokens = request.options.max_tokens.value_or(16);
  if (prompt.tokens.empty() || prompt.tokens.size() + std::uint64_t{max_tokens} > context) {
    return std::unexpected(
        BadPrompt("prompt and completion must fit the model context; an empty prompt needs an "
                  "enabled model BOS",
                  "context_length_exceeded"));
  }
  // Score rows: the context bounds their number (prompt plus output, as
  // checked above); the response's bytes, with their top scores, bound
  // what they may hold. Each row is charged as LiteralRows::Add charges
  // it, at least: twice its own bytes and its top scores' (the supplied
  // token's among them).
  const bool score_prompt = request.prompt_logprobs || (request.echo && request.logprobs);
  const std::uint64_t prompt_top = std::max(request.prompt_logprobs.value_or(0),
                                            request.echo ? request.logprobs.value_or(0) : 0);
  const auto row_bytes = [](std::uint64_t top) {
    return 2 * (kScoreRowBytes + ((top + 1) * kTopScoreBytes));
  };
  const std::uint64_t charged =
      (score_prompt ? prompt.tokens.size() * row_bytes(prompt_top) : 0) +
      (request.logprobs ? std::uint64_t{max_tokens} * row_bytes(*request.logprobs) : 0);
  if (charged > response_bytes) {
    return std::unexpected(BadPrompt(
        std::format("the score rows would pass the response's {} bytes (the request memory, "
                    "[client] request_memory_bytes); use explicit rolling windows or fewer top "
                    "scores",
                    response_bytes),
        "score_limit_exceeded"));
  }
  prompt.score_bytes = charged;
  std::string decoded;
  tokenizer::StreamDecoder decoder(tokenizer, {.control_tokens = true});
  for (const auto id : std::span(prompt.tokens).subspan(prompt.added_bos ? 1 : 0)) {
    if (!decoder.Push(id, decoded)) {
      return std::unexpected(BadPrompt("the prompt cannot be decoded"));
    }
    if (decoded.size() > response_bytes / 6) {
      return std::unexpected(
          BadPrompt("decoded prompt exceeds the bounded response size", "response_too_large"));
    }
  }
  decoder.Finish(decoded);
  if (decoded.size() > response_bytes / 6) {
    return std::unexpected(
        BadPrompt("decoded prompt exceeds the bounded response size", "response_too_large"));
  }
  if (score_prompt && request.prompt && decoded != *request.prompt) {
    return std::unexpected(
        BadPrompt("scored text must round-trip through the model tokenizer; use exact token IDs "
                  "for normalized text",
                  "tokenization_changes_text"));
  }
  prompt.text = request.prompt.value_or(decoded);
  return prompt;
}

LiteralRows::LiteralRows(const tokenizer::Tokenizer& tokenizer, bool as_id, bool controls,
                         std::size_t offset, std::size_t& budget, std::size_t limit)
    : tokenizer_(tokenizer),
      decoder_(tokenizer, {.control_tokens = controls}),
      as_id_(as_id),
      offset_(offset),
      budget_(budget),
      limit_(limit) {}

std::expected<TokenLogprob, Error> LiteralRows::Add(std::int32_t id, std::span<const float> logits,
                                                    std::uint32_t top, bool first, bool hidden) {
  if (id < 0 || std::cmp_greater_equal(id, tokenizer_.size()) ||
      tokenizer_.Kind(id) == tokenizer::TokenKind::kUnused ||
      std::cmp_greater(top, tokenizer_.size())) {
    return std::unexpected(Error{.status = 500,
                                 .type = "server_error",
                                 .message = "a score row has an invalid token or top-score count",
                                 .param = "logprobs",
                                 .code = "invalid_model_token"});
  }
  TokenLogprob row{.id = id,
                   .token = TokenText(tokenizer_, id, as_id_),
                   .logprob = std::nullopt,
                   .top = {},
                   .text_offset = offset_};
  if (!first) {
    auto scored = execution::ScoreToken(logits, id, top);
    if (!scored) {
      return std::unexpected(Error{.status = 500,
                                   .type = "server_error",
                                   .message = "a target row is not a finite scoreable distribution",
                                   .param = "logprobs",
                                   .code = "invalid_model_logits"});
    }
    row.logprob = scored->logprob;
    for (const auto& value : scored->top) {
      row.top.push_back({.id = value.id,
                         .token = TokenText(tokenizer_, value.id, as_id_),
                         .logprob = value.logprob,
                         .rank = value.rank});
    }
  }
  std::size_t bytes = kScoreRowBytes + (6 * row.token.size());
  for (const auto& value : row.top) {
    bytes += kTopScoreBytes + (6 * value.token.size());
  }
  bytes *= 2;  // prompt metadata can appear in both response forms
  if (budget_ > limit_ || bytes > limit_ - budget_) {
    return std::unexpected(Error{.status = 413,
                                 .type = "invalid_request_error",
                                 .message = "completion scores exceed the bounded response size",
                                 .param = "logprobs",
                                 .code = "response_too_large"});
  }
  budget_ += bytes;
  if (!hidden) {
    std::string piece;
    if (!decoder_.Push(id, piece)) {
      return std::unexpected(Error{.status = 500,
                                   .type = "server_error",
                                   .message = "a token cannot be decoded",
                                   .param = "logprobs",
                                   .code = "invalid_model_token"});
    }
    offset_ += TextCharacters(piece);
  }
  return row;
}

void LiteralRows::Finish() {
  std::string rest;
  decoder_.Finish(rest);
  offset_ += TextCharacters(rest);
}

LiteralOutput::LiteralOutput(const CompletionRequest& request, LiteralPrompt& prompt,
                             const tokenizer::Tokenizer& tokenizer, std::size_t limit)
    : result_{.prompt_text = std::move(prompt.text), .logprobs = {}, .prompt_logprobs = {}},
      budget_(1024 + (6 * result_.prompt_text.size())),
      limit_(limit),
      echo_(request.echo),
      logprobs_(request.logprobs),
      prompt_logprobs_(request.prompt_logprobs),
      prompt_top_(std::max(request.prompt_logprobs.value_or(0),
                           request.echo ? request.logprobs.value_or(0) : 0)),
      score_prompt_(prompt_logprobs_ || (echo_ && logprobs_)),
      prompt_rows_(tokenizer, request.return_tokens_as_token_ids, true, 0, budget_, limit_),
      generated_rows_(tokenizer, request.return_tokens_as_token_ids, false,
                      echo_ ? TextCharacters(result_.prompt_text) : 0, budget_, limit_),
      decoder_(tokenizer, {}) {}

std::expected<std::shared_ptr<LiteralOutput>, Error> LiteralOutput::Create(
    const CompletionRequest& request, LiteralPrompt& prompt, const tokenizer::Tokenizer& tokenizer,
    RequestMemory& memory) {
  if (prompt.tokens.empty()) {
    return std::unexpected(BadPrompt("literal response needs a nonempty prompt"));
  }
  const auto limit = static_cast<std::size_t>(memory.capacity());
  if (limit < 1024 || prompt.text.size() > (limit - 1024) / 6) {
    return std::unexpected(Error{.status = 413,
                                 .type = "invalid_request_error",
                                 .message = "prompt echo exceeds the response size",
                                 .param = "prompt",
                                 .code = "response_too_large"});
  }
  auto output =
      std::shared_ptr<LiteralOutput>(new LiteralOutput(request, prompt, tokenizer, limit));
  const std::uint64_t held = prompt.score_bytes + output->budget_;
  if (!output->charge_.Add(memory, held)) {
    return std::unexpected(MemoryRefusal(memory, held, "the completion's scores"));
  }
  if (output->score_prompt_ &&
      !output->PromptRow(0, prompt.tokens.front(), {}, true, prompt.added_bos)) {
    if (!output->error_) {
      base::Fatal("a failed initial literal row has no error");
    }
    return std::unexpected(*output->error_);
  }
  return output;
}

bool LiteralOutput::PromptRow(std::size_t position, std::int32_t id, std::span<const float> row,
                              bool first, bool hidden) {
  if (error_) {
    return false;
  }
  if (position < prompt_rows_seen_) {
    return true;  // a capacity rebuild, not another row in the response
  }
  if (position != prompt_rows_seen_ || prompt_ended_) {
    error_ = Error{.status = 500,
                   .type = "server_error",
                   .message = "literal prompt rows arrived out of order",
                   .param = "prompt_logprobs",
                   .code = "invalid_score_row"};
    return false;
  }
  auto scored = prompt_rows_.Add(id, row, prompt_top_, first, hidden);
  if (!scored) {
    error_ = scored.error();
    return false;
  }
  if (echo_ && logprobs_) {
    result_.logprobs.push_back(*scored);
  }
  if (prompt_logprobs_) {
    result_.prompt_logprobs.push_back(std::move(*scored));
  }
  ++prompt_rows_seen_;
  return true;
}

void LiteralOutput::EndPrompt() {
  if (!prompt_ended_) {
    prompt_rows_.Finish();
    prompt_ended_ = true;
  }
}

bool LiteralOutput::GeneratedRow(std::int32_t id, std::span<const float> row) {
  if (error_) {
    return false;
  }
  if (!logprobs_) {
    return true;
  }
  auto scored = generated_rows_.Add(id, row, *logprobs_);
  if (!scored) {
    error_ = scored.error();
    return false;
  }
  result_.logprobs.push_back(std::move(*scored));
  return true;
}

bool LiteralOutput::Text(std::string_view piece, const Send& send) {
  if (budget_ > limit_ || piece.size() > (limit_ - budget_) / 6) {
    error_ = Error{.status = 413,
                   .type = "invalid_request_error",
                   .message = "completion text exceeds the response size",
                   .param = "prompt",
                   .code = "response_too_large"};
    return false;
  }
  budget_ += 6 * piece.size();
  return piece.empty() || send(piece);
}

bool LiteralOutput::Push(std::span<const std::int32_t> fresh, const Send& send) {
  if (error_) {
    return false;
  }
  std::string piece;
  for (const auto id : fresh) {
    if (!decoder_.Push(id, piece)) {
      error_ = Error{.status = 500,
                     .type = "server_error",
                     .message = "a generated token cannot be decoded",
                     .param = "logprobs",
                     .code = "invalid_model_token"};
      return false;
    }
  }
  return Text(piece, send);
}

bool LiteralOutput::Finish(const Send& send) {
  if (error_) {
    return false;
  }
  EndPrompt();
  generated_rows_.Finish();
  std::string rest;
  decoder_.Finish(rest);
  return Text(rest, send);
}

LiteralResult LiteralOutput::TakeResult() {
  charge_.Reset();
  return std::move(result_);
}

}  // namespace jitllm::runtime::api
