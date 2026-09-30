// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Bounded literal prompt admission and compact score metadata, independent
// of the device/model runner so CPU controls exercise BOS and byte tokens.

#ifndef JITLLM_RUNTIME_COMPLETION_TOKENS_H_
#define JITLLM_RUNTIME_COMPLETION_TOKENS_H_

#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "runtime/api.h"
#include "tokenizer/tokenizer.h"

namespace jitllm::runtime::api {

struct LiteralPrompt {
  std::vector<std::int32_t> tokens;
  std::string text;
  bool added_bos = false;
};
std::expected<LiteralPrompt, Error> PrepareLiteralPrompt(const CompletionRequest& request,
                                                         const tokenizer::Tokenizer& tokenizer,
                                                         std::uint32_t context);
std::size_t TextCharacters(std::string_view text);

// No borrowed vocabulary row survives Add. Offset tracks decoded Unicode
// characters, including shared offsets for tokens within a UTF-8 sequence.
// Finish flushes a trailing partial sequence at the prompt/output boundary.
class LiteralRows {
 public:
  LiteralRows(const tokenizer::Tokenizer& tokenizer, bool as_id, bool controls, std::size_t offset,
              std::size_t& budget);
  std::expected<TokenLogprob, Error> Add(std::int32_t id, std::span<const float> logits,
                                         std::uint32_t top, bool first = false,
                                         bool hidden = false);
  void Finish();
  std::size_t offset() const { return offset_; }

 private:
  const tokenizer::Tokenizer& tokenizer_;
  tokenizer::StreamDecoder decoder_;
  bool as_id_;
  std::size_t offset_;
  std::size_t& budget_;
};

}  // namespace jitllm::runtime::api

#endif  // JITLLM_RUNTIME_COMPLETION_TOKENS_H_
