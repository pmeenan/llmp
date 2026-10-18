// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// A standalone SentencePiece ModelProto. Reads protobuf wire bytes directly;
// no generated runtime, checkpoint code, or external process is used.
#ifndef LLMP_TOKENIZER_SENTENCEPIECE_H_
#define LLMP_TOKENIZER_SENTENCEPIECE_H_

#include <expected>
#include <string_view>

#include "tokenizer/error.h"
#include "tokenizer/tokenizer.h"

namespace llmp::tokenizer {

// Supports byte-fallback BPE with an identity normalization map and the
// whitespace options represented by TokenizerSpec. Other algorithms/maps
// are refused explicitly. Lengths, scalar types and piece kinds are checked.
std::expected<TokenizerSpec, Error> ReadSentencePieceTokenizer(std::string_view bytes);

}  // namespace llmp::tokenizer
#endif  // LLMP_TOKENIZER_SENTENCEPIECE_H_
