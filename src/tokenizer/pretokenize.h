// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Pre-tokenizers: split normalized text into the pieces BPE merges within,
// natively, with no regex engine. Each one implements the expressions that
// llama.cpp (pinned b29c606e, src/llama-vocab.cpp and src/unicode.cpp) and
// the models' tokenizer.json files name, with the same results:
//
// - kQwen2 (GGUF `qwen2`; Qwen2Tokenizer's Split):
//     (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}
//     | ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
// - kQwen35 (GGUF `qwen35`; Qwen3.5 and Qwen3.8): as kQwen2, with letter
//   runs taking marks, [\p{L}\p{M}]+, and marks out of the symbol run.
// - kDeepSeekV3 (GGUF `deepseek-v3` and `joyai-llm`; DeepSeek V3 and V4):
//   three Split stages in turn, each splitting every piece of the last:
//     \p{N}{1,3}
//     [一-龥぀-ゟ゠-ヿ]+
//     [!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+
//     | ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
// - kGemma4 (GGUF `gemma4`): [^\n]+|[\n]+ over raw UTF-8 with
//   spaces replaced by ▁ before splitting.
// - kSentencePiece (GGUF model `llama`): one whole nonempty fragment;
//   token scores select merges, with optional dummy prefix and ▁ spaces.
//   Text a stage does not match stays one piece between its matches (the
//   Split's "Isolated" behavior; llama.cpp's gaps).
//
// Matching is backtracking-regex semantics (the first alternative that
// matches at the leftmost position), which both references use: \s is
// White_Space, \p{..} the general category, both from UCD 15.1.0. The
// contractions' (?i:...) matches ASCII letters of either case only: llama.cpp
// lowercases with UnicodeData's simple mapping and Hugging Face tokenizers
// uses Oniguruma, and both leave 'ſ (long s) and every other non-ASCII
// letter unmatched (the corpus checks it).

#ifndef JITLLM_TOKENIZER_PRETOKENIZE_H_
#define JITLLM_TOKENIZER_PRETOKENIZE_H_

#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace jitllm::tokenizer {

enum class PreTokenizer : std::uint8_t { kQwen2, kQwen35, kDeepSeekV3, kGemma4, kSentencePiece };

std::string_view PreTokenizerName(PreTokenizer p);

// Splits text into pieces, appending each piece's length in code points;
// the lengths sum to text.size() and none is zero.
void PreTokenize(PreTokenizer p, std::span<const char32_t> text,
                 std::vector<std::uint32_t>& lengths);

}  // namespace jitllm::tokenizer

#endif  // JITLLM_TOKENIZER_PRETOKENIZE_H_
