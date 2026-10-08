// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "tokenizer/pretokenize.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

#include "tokenizer/unicode.h"

namespace llmp::tokenizer {
namespace {

using unicode::IsLetter;
using unicode::IsMark;
using unicode::IsNumber;
using unicode::IsPunctuation;
using unicode::IsSymbol;
using unicode::IsWhiteSpace;

using Text = std::span<const char32_t>;

bool IsCrLf(char32_t c) { return c == U'\r' || c == U'\n'; }

// An ASCII letter in lower case; anything else as 0.
char LowerAscii(char32_t c) {
  if (c >= U'a' && c <= U'z') {
    return static_cast<char>(c);
  }
  if (c >= U'A' && c <= U'Z') {
    return static_cast<char>(c - U'A' + U'a');
  }
  return '\0';
}

// (?i:'s|'t|'re|'ve|'m|'ll|'d) at p, within [.., e): its length or 0.
std::size_t Contraction(Text t, std::size_t p, std::size_t e) {
  if (t[p] != U'\'' || p + 1 >= e) {
    return 0;
  }
  const char a = LowerAscii(t[p + 1]);
  if (a == 's' || a == 't' || a == 'm' || a == 'd') {
    return 2;
  }
  if (p + 2 < e) {
    const char b = LowerAscii(t[p + 2]);
    if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) {
      return 3;
    }
  }
  return 0;
}

// The whitespace alternatives shared by all three expressions,
// \s*[\r\n]+|\s+(?!\S)|\s+ : their match length at p, or 0.
std::size_t Whitespace(Text t, std::size_t p, std::size_t e) {
  std::size_t w = p;
  std::size_t crlf_end = 0;
  while (w < e && IsWhiteSpace(t[w])) {
    if (IsCrLf(t[w])) {
      crlf_end = w + 1;
    }
    ++w;
  }
  if (crlf_end != 0) {
    return crlf_end - p;  // \s*[\r\n]+ backtracks to the last CR or LF
  }
  const std::size_t n = w - p;
  if (n > 1 && w < e) {
    return n - 1;  // \s+(?!\S) leaves the space before a non-space
  }
  return n;  // \s+(?!\S) at the end, or \s+
}

// The Qwen2 and Qwen3.5 expression at p; every code point matches one of
// its alternatives, so the result is at least 1.
std::size_t MatchQwen(Text t, std::size_t p, std::size_t e, bool marks) {
  if (const std::size_t n = Contraction(t, p, e); n != 0) {
    return n;
  }
  const char32_t c = t[p];
  auto word = [&](char32_t x) { return IsLetter(x) || (marks && IsMark(x)); };
  // [^\r\n\p{L}\p{N}]?\p{L}+  (Qwen3.5: [\p{L}\p{M}]+)
  if (!IsCrLf(c) && !IsNumber(c) && (word(c) || (p + 1 < e && word(t[p + 1])))) {
    std::size_t q = p + 1;
    while (q < e && word(t[q])) {
      ++q;
    }
    return q - p;
  }
  // \p{N}
  if (IsNumber(c)) {
    return 1;
  }
  //  ?[^\s\p{L}\p{N}]+[\r\n]*  (Qwen3.5: also not \p{M})
  auto other = [&](char32_t x) { return !IsWhiteSpace(x) && !IsNumber(x) && !word(x); };
  std::size_t q = c == U' ' ? p + 1 : p;
  if (q < e && other(t[q])) {
    while (q < e && other(t[q])) {
      ++q;
    }
    while (q < e && IsCrLf(t[q])) {
      ++q;
    }
    return q - p;
  }
  if (const std::size_t n = Whitespace(t, p, e); n != 0) {
    return n;
  }
  return 1;  // unreachable: every code point is a letter, number, space or other
}

bool IsAsciiLetter(char32_t c) { return (c >= U'A' && c <= U'Z') || (c >= U'a' && c <= U'z'); }

bool IsAsciiPunctuationOrSymbol(char32_t c) {
  return (c >= 0x21 && c <= 0x2F) || (c >= 0x3A && c <= 0x40) || (c >= 0x5B && c <= 0x60) ||
         (c >= 0x7B && c <= 0x7E);
}

// DeepSeek V3's first stage, \p{N}{1,3}.
std::size_t MatchDigits(Text t, std::size_t p, std::size_t e) {
  std::size_t q = p;
  while (q < e && q - p < 3 && IsNumber(t[q])) {
    ++q;
  }
  return q - p;
}

// The second, [一-龥぀-ゟ゠-ヿ]+.
std::size_t MatchCjk(Text t, std::size_t p, std::size_t e) {
  auto cjk = [](char32_t c) {
    return (c >= 0x4E00 && c <= 0x9FA5) || (c >= 0x3040 && c <= 0x309F) ||
           (c >= 0x30A0 && c <= 0x30FF);
  };
  std::size_t q = p;
  while (q < e && cjk(t[q])) {
    ++q;
  }
  return q - p;
}

// The third expression at p, or 0 when no alternative matches there.
std::size_t MatchDeepSeek(Text t, std::size_t p, std::size_t e) {
  const char32_t c = t[p];
  // [!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+
  if (IsAsciiPunctuationOrSymbol(c) && p + 1 < e && IsAsciiLetter(t[p + 1])) {
    std::size_t q = p + 1;
    while (q < e && IsAsciiLetter(t[q])) {
      ++q;
    }
    return q - p;
  }
  // [^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+
  auto word = [](char32_t x) { return IsLetter(x) || IsMark(x); };
  const bool prefix = !IsCrLf(c) && !IsLetter(c) && !IsPunctuation(c) && !IsSymbol(c);
  if ((prefix && p + 1 < e && word(t[p + 1])) || word(c)) {
    std::size_t q = p + 1;
    while (q < e && word(t[q])) {
      ++q;
    }
    return q - p;
  }
  //  ?[\p{P}\p{S}]+[\r\n]*
  auto ps = [](char32_t x) { return IsPunctuation(x) || IsSymbol(x); };
  std::size_t q = p;
  if (c == U' ' && p + 1 < e && ps(t[p + 1])) {
    q = p + 1;
  }
  if (ps(t[q])) {
    while (q < e && ps(t[q])) {
      ++q;
    }
    while (q < e && IsCrLf(t[q])) {
      ++q;
    }
    return q - p;
  }
  return Whitespace(t, p, e);
}

// One Split stage with "Isolated" behavior: each piece in `in` is split
// into its matches and the unmatched runs between them.
template <typename Match>
void Stage(Text t, const std::vector<std::uint32_t>& in, std::vector<std::uint32_t>& out,
           Match match) {
  std::size_t start = 0;
  for (const std::uint32_t length : in) {
    const std::size_t e = start + length;
    std::size_t gap = start;
    std::size_t p = start;
    while (p < e) {
      const std::size_t n = match(t, p, e);
      if (n == 0) {
        ++p;
        continue;
      }
      if (p > gap) {
        out.push_back(static_cast<std::uint32_t>(p - gap));
      }
      out.push_back(static_cast<std::uint32_t>(n));
      p += n;
      gap = p;
    }
    if (e > gap) {
      out.push_back(static_cast<std::uint32_t>(e - gap));
    }
    start = e;
  }
}

}  // namespace

std::string_view PreTokenizerName(PreTokenizer p) {
  switch (p) {
    case PreTokenizer::kQwen2:
      return "qwen2";
    case PreTokenizer::kQwen35:
      return "qwen35";
    case PreTokenizer::kDeepSeekV3:
      return "deepseek-v3";
    case PreTokenizer::kGemma4:
      return "gemma4";
    case PreTokenizer::kSentencePiece:
      return "sentencepiece";
  }
  return "unknown";
}

void PreTokenize(PreTokenizer p, Text text, std::vector<std::uint32_t>& lengths) {
  if (text.empty()) {
    return;
  }
  const std::vector<std::uint32_t> whole = {static_cast<std::uint32_t>(text.size())};
  switch (p) {
    case PreTokenizer::kQwen2:
    case PreTokenizer::kQwen35: {
      const bool marks = p == PreTokenizer::kQwen35;
      Stage(text, whole, lengths,
            [&](Text t, std::size_t at, std::size_t e) { return MatchQwen(t, at, e, marks); });
      return;
    }
    case PreTokenizer::kSentencePiece:
      if (!text.empty()) {
        lengths.push_back(static_cast<std::uint32_t>(text.size()));
      }
      return;
    case PreTokenizer::kGemma4: {
      for (std::size_t at = 0; at < text.size();) {
        std::size_t end = at + 1;
        while (end < text.size() && (text[end] == U'\n') == (text[at] == U'\n')) {
          ++end;
        }
        lengths.push_back(static_cast<std::uint32_t>(end - at));
        at = end;
      }
      return;
    }
    case PreTokenizer::kDeepSeekV3: {
      std::vector<std::uint32_t> digits;
      std::vector<std::uint32_t> cjk;
      Stage(text, whole, digits, MatchDigits);
      Stage(text, digits, cjk, MatchCjk);
      Stage(text, cjk, lengths, MatchDeepSeek);
      return;
    }
  }
}

}  // namespace llmp::tokenizer
