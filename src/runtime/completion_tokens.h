// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Bounded literal prompt admission and compact score metadata, independent
// of the device/model runner so CPU controls exercise BOS and byte tokens.
// The bounds are the model's (its context and vocabulary) and the
// response's bytes (`response_bytes`: the request memory's, D-102): score
// rows and their top scores are figured against it before any work, and
// charged as they are made.

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
  std::uint64_t score_bytes = 0;  // what its score rows are charged, at most
};
// With `memory`, a text prompt's tokenization is charged to it while it
// runs (Tokenizer::EncodeWorkingBytes and the tokens), and refused when it
// does not fit.
std::expected<LiteralPrompt, Error> PrepareLiteralPrompt(const CompletionRequest& request,
                                                         const tokenizer::Tokenizer& tokenizer,
                                                         std::uint32_t context,
                                                         std::size_t response_bytes,
                                                         RequestMemory* memory = nullptr);
std::size_t TextCharacters(std::string_view text);

// No borrowed vocabulary row survives Add. Offset tracks decoded Unicode
// characters, including shared offsets for tokens within a UTF-8 sequence.
// Finish flushes a trailing partial sequence at the prompt/output boundary.
class LiteralRows {
 public:
  // `budget` is the response bytes charged so far, shared by the request's
  // rows, which may reach `limit`.
  LiteralRows(const tokenizer::Tokenizer& tokenizer, bool as_id, bool controls, std::size_t offset,
              std::size_t& budget, std::size_t limit);
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
  std::size_t limit_;
};

}  // namespace jitllm::runtime::api

#endif  // JITLLM_RUNTIME_COMPLETION_TOKENS_H_
