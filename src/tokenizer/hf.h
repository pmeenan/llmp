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
// - pre_tokenizer: a Sequence of Split(Regex, Isolated) stages whose
//   expressions are exactly one of the pre-tokenizers pretokenize.h
//   implements, then ByteLevel without a prefix space or its own regex;
// - decoder ByteLevel; post_processor ByteLevel or none, which add nothing;
//   no truncation or padding;
// - added_tokens without lstrip, rstrip or single_word, and "normalized"
//   only where no normalizer could change them. A special one is a control
//   token and any other a user-defined one; an added token may restate a
//   vocabulary entry's ID and text but not change either.
//
// Raw UTF-8 BPE also accepts the declared space->▁ Replace, no-op Split
// on replaced spaces, inverse Replace/ByteFallback/Fuse decoder and a
// single-input BOS/EOS TemplateProcessing (the Gemma 2/3 HF pipelines).
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
