// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// A Hugging Face tokenizer.json, read into a TokenizerSpec. The file is
// untrusted JSON (base/json.h's limits apply) and only this shape is
// accepted, each field checked:
//
// - model: BPE, with no dropout, unknown token, subword prefix or suffix,
//   byte fallback or fused unknowns; ignore_merges either way; merges as
//   "a b" strings or [a, b] pairs;
// - normalizer: none, NFC, or a Sequence of none or NFC alone;
// - pre_tokenizer: nested Sequence and isolated regex Split stages, ending
//   in ByteLevel without prefix spaces; use_regex adds GPT-2 splitting;
// - decoder ByteLevel; post_processor ByteLevel, TemplateProcessing with
//   compact multi-ID definitions/references and exactly one A, or nested Sequence; no padding or
//   truncation. Count/validate precedes exact recipe allocation; expanded IDs
//   are streamed only at encoding. Unsupported layouts/flags are refused;
// - added_tokens without lstrip, rstrip or single_word, and "normalized"
//   only where no normalizer could change them. A special one is a control
//   token and any other a user-defined one; an added token may restate a
//   vocabulary entry's ID and text but not change either.
//
// Raw UTF-8 BPE also accepts the declared space->▁ Replace, no-op Split
// on replaced spaces, inverse Replace/ByteFallback/Fuse decoder and a
// the shared single-input postprocessor (the Gemma 2/3 HF pipelines).
// Model membership remains independent of added-token matching metadata.
//
// IDs must run densely from 0.

#ifndef LLMP_TOKENIZER_HF_H_
#define LLMP_TOKENIZER_HF_H_

#include <expected>
#include <string_view>

#include "tokenizer/error.h"
#include "tokenizer/tokenizer.h"

namespace llmp::tokenizer {

std::expected<TokenizerSpec, Error> ReadHfTokenizer(std::string_view text);

// The kept Transformers tokenizer_config applies special-token identities,
// added-token metadata and automatic BOS/EOS insertion to either native
// reader. Matching options that cannot be represented are refused.
std::expected<void, Error> ApplyHfTokenizerConfig(std::string_view text, TokenizerSpec& spec);

}  // namespace llmp::tokenizer

#endif  // LLMP_TOKENIZER_HF_H_
