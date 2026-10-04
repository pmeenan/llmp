// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// A GGUF file's tokenizer, read from the start of the file (its header and
// key-value metadata), which is untrusted: every length and count is
// checked against the bytes given before anything is allocated, the keys
// the reader uses also against the caps below (other keys are skipped
// unread), keys may not repeat, and the tokenizer keys must have their
// documented types.
//
// Supported: `tokenizer.ggml.model` "gpt2" (byte-level BPE) with the
// pre-tokenizers `qwen2`, `qwen35`, `deepseek-v3` and `joyai-llm`, as
// llama.cpp reads them; token types normal, control, user-defined and unused.
// Also "gemma4": raw UTF-8 BPE, optional `gemma4` pre-tokenizer, and
// <0xNN> byte fallback tokens. Anything else is refused as unsupported.

#ifndef JITLLM_TOKENIZER_GGUF_H_
#define JITLLM_TOKENIZER_GGUF_H_

#include <cstddef>
#include <expected>
#include <span>
#include <string>

#include "tokenizer/error.h"
#include "tokenizer/tokenizer.h"

namespace jitllm::tokenizer {

inline constexpr std::size_t kMaxGgufKeys = std::size_t{1} << 16U;
inline constexpr std::size_t kMaxGgufString = std::size_t{16} << 20U;
inline constexpr std::size_t kMaxGgufArray = std::size_t{1} << 24U;

struct GgufTokenizer {
  TokenizerSpec spec;
  std::string architecture;   // general.architecture
  std::string pre;            // tokenizer.ggml.pre
  std::string chat_template;  // tokenizer.chat_template, or empty
  bool has_chat_template = false;
};

// Reads the tokenizer from `bytes`, a prefix of a GGUF file long enough to
// hold its metadata (kFormat "truncated" otherwise).
std::expected<GgufTokenizer, Error> ReadGgufTokenizer(std::span<const std::byte> bytes);

}  // namespace jitllm::tokenizer

#endif  // JITLLM_TOKENIZER_GGUF_H_
