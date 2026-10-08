// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Unicode for the tokenizer: code point properties from the Unicode
// Character Database 15.1.0 (unicode_data.h, D-088), strict UTF-8, and
// Normalization Form C; and the full case mappings and case properties
// Python's str operations use, for the chat templates (src/chat/pycase.h).
//
// UCD 15.1.0 is the version llama.cpp's tables at the pinned revision were
// generated from (docs/licensing.md), so the pre-tokenizers classify code
// points exactly as the reference does.

#ifndef LLMP_TOKENIZER_UNICODE_H_
#define LLMP_TOKENIZER_UNICODE_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "tokenizer/error.h"

namespace llmp::tokenizer::unicode {

// The general category (UAX #44), in unicode_data.h's order.
enum class Category : std::uint8_t {
  kCn,
  kLu,
  kLl,
  kLt,
  kLm,
  kLo,
  kMn,
  kMc,
  kMe,
  kNd,
  kNl,
  kNo,
  kPc,
  kPd,
  kPs,
  kPe,
  kPi,
  kPf,
  kPo,
  kSm,
  kSc,
  kSk,
  kSo,
  kZs,
  kZl,
  kZp,
  kCc,
  kCf,
  kCs,
  kCo,
};

// The UCD version of the tables ("15.1.0").
std::string_view Version();

// Code points above U+10FFFF read as unassigned (Cn), with no properties.
Category GetCategory(char32_t cp);
bool IsLetter(char32_t cp);       // L
bool IsMark(char32_t cp);         // M
bool IsNumber(char32_t cp);       // N
bool IsPunctuation(char32_t cp);  // P
bool IsSymbol(char32_t cp);       // S
bool IsWhiteSpace(char32_t cp);   // the White_Space property
std::uint8_t CombiningClass(char32_t cp);

// Full case mappings as Python's str operations use them (CPython's
// _PyUnicode_ToLowerFull and the like): SpecialCasing.txt's unconditional
// mappings, else the simple ones; 1 to 3 code points. Final_Sigma, the one
// conditional mapping Python applies, is the caller's (src/chat/pycase.h).
struct CaseMapping {
  std::array<char32_t, 3> code_points{};
  std::size_t length = 0;
  std::span<const char32_t> view() const { return {code_points.data(), length}; }
};
CaseMapping ToLowerFull(char32_t cp);
CaseMapping ToUpperFull(char32_t cp);
CaseMapping ToTitleFull(char32_t cp);
// The derived core properties Python's case operations and tests read.
bool IsLowercase(char32_t cp);      // Lowercase
bool IsUppercase(char32_t cp);      // Uppercase
bool IsCased(char32_t cp);          // Cased
bool IsCaseIgnorable(char32_t cp);  // Case_Ignorable

// Strict UTF-8 (Unicode table 3-7): decodes all of `text`, appending its
// code points, or refuses with kInvalidUtf8 at the first ill-formed
// sequence's offset.
std::expected<void, Error> DecodeUtf8(std::string_view text, std::vector<char32_t>& out);
// The same check without decoding.
std::expected<void, Error> ValidateUtf8(std::string_view text);
void AppendUtf8(char32_t cp, std::string& out);
// Replaces each maximal ill-formed subpart with U+FFFD (the Unicode
// Standard's recommended practice, also Python's "replace" and WHATWG's
// decoder), keeping everything else.
std::string ReplaceInvalidUtf8(std::string_view bytes);
// The length of the longest prefix of `bytes` that ends on a character
// boundary of a well-formed sequence, not counting a trailing incomplete but
// still possible sequence; used by streaming decoders.
std::size_t CompleteUtf8Prefix(std::string_view bytes);

// Normalization Form C (UAX #15): replaces `text` by its NFC, which is
// usually itself (a quick check skips the rest).
void ToNfc(std::vector<char32_t>& text);

}  // namespace llmp::tokenizer::unicode

#endif  // LLMP_TOKENIZER_UNICODE_H_
