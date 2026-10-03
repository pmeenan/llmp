// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Why the tokenizer refused its data or an input (D-066): the violated
// rule, a static reason, and the item it concerns (a byte offset, token ID,
// merge rank or other index). Nothing from the untrusted text is copied into
// an error.

#ifndef JITLLM_TOKENIZER_ERROR_H_
#define JITLLM_TOKENIZER_ERROR_H_

#include <cstdint>
#include <limits>
#include <string>
#include <string_view>

namespace jitllm::tokenizer {

enum class Rule : std::uint8_t {
  // Inputs.
  kInvalidUtf8,     // text is not well-formed UTF-8; item: byte offset
  kInputTooLarge,   // text over the encoder's byte bound; item: the bound
  kOutputTooLarge,  // encoding would pass the token bound; item: the bound
  kInvalidToken,    // a token ID outside the vocabulary; item: its position
  kSpecialToken,    // a marked special span is not a special token's text; item: span index
  // Tokenizer data.
  kFormat,       // not a well-formed file of its kind (GGUF, tokenizer.json)
  kUnsupported,  // well formed, but a feature or variant jitLLM does not implement
  kVocabulary,   // inconsistent: duplicate texts, merges of unknown tokens, missing bytes
  kBounds,       // over a cap on counts or sizes
  // Work.
  kCancelled,  // the calling thread's pulse asked it to stop (base/work_pulse.h, D-102)
};

// The rule's name ("invalid-utf8").
std::string_view RuleName(Rule rule);

inline constexpr std::uint64_t kNoItem = std::numeric_limits<std::uint64_t>::max();

struct Error {
  Rule rule = Rule::kFormat;
  std::string_view reason;  // static text
  std::uint64_t item = kNoItem;

  // "vocabulary: merge names an unknown token (item 17)".
  std::string ToString() const;
};

}  // namespace jitllm::tokenizer

#endif  // JITLLM_TOKENIZER_ERROR_H_
