// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "runtime/completion_tokens.h"

#include <algorithm>
#include <format>
#include <utility>

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
                                                         std::uint32_t context) {
  LiteralPrompt prompt{.tokens = request.token_ids, .text = {}, .added_bos = false};
  if (request.prompt) {
    if (!tokenizer.Encode(*request.prompt,
                          {.special = tokenizer::SpecialTokens::kParse,
                           .add_bos_eos = false,
                           .max_bytes = kMaxMessageBytes,
                           .max_tokens = context},
                          prompt.tokens)) {
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
  const bool score_prompt = request.prompt_logprobs || (request.echo && request.logprobs);
  const std::uint64_t rows = (score_prompt ? prompt.tokens.size() : 0) +
                             (request.logprobs ? std::uint64_t{max_tokens} : 0);
  if (rows > kMaxCompletionScoreRows) {
    return std::unexpected(
        BadPrompt("too many score rows; use explicit rolling windows", "score_limit_exceeded"));
  }
  std::string decoded;
  tokenizer::StreamDecoder decoder(tokenizer, {.control_tokens = true});
  for (const auto id : std::span(prompt.tokens).subspan(prompt.added_bos ? 1 : 0)) {
    if (!decoder.Push(id, decoded)) {
      return std::unexpected(BadPrompt("the prompt cannot be decoded"));
    }
    if (decoded.size() > kMaxCompletionResponseBytes / 6) {
      return std::unexpected(
          BadPrompt("decoded prompt exceeds the bounded response size", "response_too_large"));
    }
  }
  decoder.Finish(decoded);
  if (decoded.size() > kMaxCompletionResponseBytes / 6) {
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
                         std::size_t offset, std::size_t& budget)
    : tokenizer_(tokenizer),
      decoder_(tokenizer, {.control_tokens = controls}),
      as_id_(as_id),
      offset_(offset),
      budget_(budget) {}

std::expected<TokenLogprob, Error> LiteralRows::Add(std::int32_t id, std::span<const float> logits,
                                                    std::uint32_t top, bool first, bool hidden) {
  if (id < 0 || std::cmp_greater_equal(id, tokenizer_.size()) ||
      tokenizer_.Kind(id) == tokenizer::TokenKind::kUnused || top > kMaxCompletionTopLogprobs) {
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
  std::size_t bytes = 256 + (6 * row.token.size());
  for (const auto& value : row.top) {
    bytes += 128 + (6 * value.token.size());
  }
  bytes *= 2;  // prompt metadata can appear in both response forms
  if (budget_ > kMaxCompletionResponseBytes || bytes > kMaxCompletionResponseBytes - budget_) {
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

}  // namespace jitllm::runtime::api
