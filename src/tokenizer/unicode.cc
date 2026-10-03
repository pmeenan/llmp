// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "tokenizer/unicode.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "tokenizer/error.h"
#include "tokenizer/unicode_data.h"

namespace jitllm::tokenizer::unicode {
namespace {

namespace data = unicode_data;

std::uint8_t Properties(char32_t cp) {
  if (cp > data::kMaxCodePoint) {
    return 0;
  }
  const std::uint16_t block = data::kPropertyIndex[cp >> data::kBlockBits];
  return data::kPropertyBlocks[block][cp & 0xFFU];
}

bool InRange(Category c, Category first, Category last) {
  return static_cast<std::uint8_t>(c) >= static_cast<std::uint8_t>(first) &&
         static_cast<std::uint8_t>(c) <= static_cast<std::uint8_t>(last);
}

// Hangul syllables (Unicode chapter 3.12).
constexpr char32_t kSBase = 0xAC00;
constexpr char32_t kLBase = 0x1100;
constexpr char32_t kVBase = 0x1161;
constexpr char32_t kTBase = 0x11A7;
constexpr char32_t kLCount = 19;
constexpr char32_t kVCount = 21;
constexpr char32_t kTCount = 28;
constexpr char32_t kNCount = kVCount * kTCount;
constexpr char32_t kSCount = kLCount * kNCount;

void Decompose(char32_t cp, std::vector<char32_t>& out) {
  if (cp >= kSBase && cp < kSBase + kSCount) {
    const char32_t s = cp - kSBase;
    out.push_back(kLBase + (s / kNCount));
    out.push_back(kVBase + ((s % kNCount) / kTCount));
    if (s % kTCount != 0) {
      out.push_back(kTBase + (s % kTCount));
    }
    return;
  }
  const auto entries = data::kDecompositions;
  const auto it = std::ranges::lower_bound(entries, cp, {}, &data::Decomposition::code_point);
  if (it == entries.end() || it->code_point != cp) {
    out.push_back(cp);
    return;
  }
  const auto parts = data::kDecompositionParts.subspan(it->offset, it->length);
  out.insert(out.end(), parts.begin(), parts.end());
}

// The primary composite of a + b, or 0.
char32_t Compose(char32_t a, char32_t b) {
  if (a >= kLBase && a < kLBase + kLCount && b >= kVBase && b < kVBase + kVCount) {
    return kSBase + ((((a - kLBase) * kVCount) + (b - kVBase)) * kTCount);
  }
  if (a >= kSBase && a < kSBase + kSCount && (a - kSBase) % kTCount == 0 && b > kTBase &&
      b < kTBase + kTCount) {
    return a + (b - kTBase);
  }
  const auto entries = data::kCompositions;
  const auto it = std::ranges::lower_bound(
      entries, std::pair{a, b}, {},
      [](const data::Composition& c) { return std::pair{c.first, c.second}; });
  return it != entries.end() && it->first == a && it->second == b ? it->composite : 0;
}

// Whether text is certainly NFC already (UAX #15's quick check giving YES).
bool QuickCheckNfc(std::span<const char32_t> text) {
  std::uint8_t last = 0;
  for (const char32_t cp : text) {
    if (cp < 0x300) {  // no code point below U+0300 has NFC_QC other than Yes, or a nonzero class
      last = 0;
      continue;
    }
    const std::uint8_t ccc = CombiningClass(cp);
    if (ccc != 0 && last > ccc) {
      return false;
    }
    if ((Properties(cp) & (data::kNfcNoBit | data::kNfcMaybeBit)) != 0) {
      return false;
    }
    last = ccc;
  }
  return true;
}

}  // namespace

std::string_view Version() { return data::kVersion; }

Category GetCategory(char32_t cp) {
  return static_cast<Category>(Properties(cp) & data::kCategoryMask);
}

bool IsLetter(char32_t cp) { return InRange(GetCategory(cp), Category::kLu, Category::kLo); }
bool IsMark(char32_t cp) { return InRange(GetCategory(cp), Category::kMn, Category::kMe); }
bool IsNumber(char32_t cp) { return InRange(GetCategory(cp), Category::kNd, Category::kNo); }
bool IsPunctuation(char32_t cp) { return InRange(GetCategory(cp), Category::kPc, Category::kPo); }
bool IsSymbol(char32_t cp) { return InRange(GetCategory(cp), Category::kSm, Category::kSo); }
bool IsWhiteSpace(char32_t cp) { return (Properties(cp) & data::kWhiteSpaceBit) != 0; }

std::uint8_t CombiningClass(char32_t cp) {
  if (cp > data::kMaxCodePoint) {
    return 0;
  }
  const std::uint16_t block = data::kCombiningClassIndex[cp >> data::kBlockBits];
  return data::kCombiningClassBlocks[block][cp & 0xFFU];
}

namespace {

CaseMapping MapCase(std::span<const data::CaseMapping> table, char32_t cp) {
  CaseMapping m;
  const auto it = std::ranges::lower_bound(table, cp, {}, &data::CaseMapping::code_point);
  if (it == table.end() || it->code_point != cp) {
    m.code_points[0] = cp;
    m.length = 1;
    return m;
  }
  m.length = it->length;
  std::ranges::copy(data::kCaseParts.subspan(it->offset, it->length), m.code_points.begin());
  return m;
}

bool InRanges(std::span<const data::CodePointRange> ranges, char32_t cp) {
  const auto it = std::ranges::lower_bound(ranges, cp, {}, &data::CodePointRange::last);
  return it != ranges.end() && it->first <= cp;
}

}  // namespace

CaseMapping ToLowerFull(char32_t cp) { return MapCase(data::kLowerFull, cp); }
CaseMapping ToUpperFull(char32_t cp) { return MapCase(data::kUpperFull, cp); }
CaseMapping ToTitleFull(char32_t cp) { return MapCase(data::kTitleFull, cp); }
bool IsLowercase(char32_t cp) { return InRanges(data::kLowercase, cp); }
bool IsUppercase(char32_t cp) { return InRanges(data::kUppercase, cp); }
bool IsCased(char32_t cp) { return InRanges(data::kCased, cp); }
bool IsCaseIgnorable(char32_t cp) { return InRanges(data::kCaseIgnorable, cp); }

namespace {

// The length of the well-formed sequence at p[0] (with n bytes available),
// or, when it is ill formed, 0 and the length of its maximal subpart in
// *subpart (at least 1). A truncated but so-far valid sequence at the end
// reports its available length as the subpart.
std::size_t SequenceAt(const unsigned char* p, std::size_t n, std::size_t* subpart) {
  const unsigned c = p[0];
  if (c < 0x80) {
    return 1;
  }
  std::size_t length = 0;
  unsigned low = 0x80;
  unsigned high = 0xBF;
  if (c >= 0xC2 && c <= 0xDF) {
    length = 2;
  } else if (c >= 0xE0 && c <= 0xEF) {
    length = 3;
    low = c == 0xE0 ? 0xA0 : 0x80;
    high = c == 0xED ? 0x9F : 0xBF;
  } else if (c >= 0xF0 && c <= 0xF4) {
    length = 4;
    low = c == 0xF0 ? 0x90 : 0x80;
    high = c == 0xF4 ? 0x8F : 0xBF;
  } else {
    *subpart = 1;
    return 0;
  }
  std::size_t good = 1;
  for (std::size_t k = 1; k < length; ++k) {
    if (k >= n) {
      *subpart = good;
      return 0;
    }
    const unsigned b = p[k];
    const unsigned lo = k == 1 ? low : 0x80;
    const unsigned hi = k == 1 ? high : 0xBF;
    if (b < lo || b > hi) {
      *subpart = good;
      return 0;
    }
    ++good;
  }
  return length;
}

}  // namespace

std::expected<void, Error> DecodeUtf8(std::string_view text, std::vector<char32_t>& out) {
  const auto* p = reinterpret_cast<const unsigned char*>(text.data());
  const std::size_t n = text.size();
  out.reserve(out.size() + n);
  std::size_t i = 0;
  while (i < n) {
    const unsigned c = p[i];
    if (c < 0x80) {
      out.push_back(c);
      ++i;
      continue;
    }
    std::size_t subpart = 0;
    const std::size_t length = SequenceAt(p + i, n - i, &subpart);
    if (length == 0) {
      return std::unexpected(Error{Rule::kInvalidUtf8, "ill-formed UTF-8", i});
    }
    char32_t cp = 0;
    if (length == 2) {
      cp = ((c & 0x1FU) << 6U) | (p[i + 1] & 0x3FU);
    } else if (length == 3) {
      cp = ((c & 0x0FU) << 12U) | ((p[i + 1] & 0x3FU) << 6U) | (p[i + 2] & 0x3FU);
    } else {
      cp = ((c & 0x07U) << 18U) | ((p[i + 1] & 0x3FU) << 12U) | ((p[i + 2] & 0x3FU) << 6U) |
           (p[i + 3] & 0x3FU);
    }
    out.push_back(cp);
    i += length;
  }
  return {};
}

std::expected<void, Error> ValidateUtf8(std::string_view text) {
  const auto* p = reinterpret_cast<const unsigned char*>(text.data());
  const std::size_t n = text.size();
  std::size_t i = 0;
  while (i < n) {
    if (p[i] < 0x80) {
      ++i;
      continue;
    }
    std::size_t subpart = 0;
    const std::size_t length = SequenceAt(p + i, n - i, &subpart);
    if (length == 0) {
      return std::unexpected(Error{Rule::kInvalidUtf8, "ill-formed UTF-8", i});
    }
    i += length;
  }
  return {};
}

void AppendUtf8(char32_t cp, std::string& out) {
  if (cp < 0x80) {
    out.push_back(static_cast<char>(cp));
  } else if (cp < 0x800) {
    out.push_back(static_cast<char>(0xC0U | (cp >> 6U)));
    out.push_back(static_cast<char>(0x80U | (cp & 0x3FU)));
  } else if (cp < 0x10000) {
    out.push_back(static_cast<char>(0xE0U | (cp >> 12U)));
    out.push_back(static_cast<char>(0x80U | ((cp >> 6U) & 0x3FU)));
    out.push_back(static_cast<char>(0x80U | (cp & 0x3FU)));
  } else {
    out.push_back(static_cast<char>(0xF0U | (cp >> 18U)));
    out.push_back(static_cast<char>(0x80U | ((cp >> 12U) & 0x3FU)));
    out.push_back(static_cast<char>(0x80U | ((cp >> 6U) & 0x3FU)));
    out.push_back(static_cast<char>(0x80U | (cp & 0x3FU)));
  }
}

std::string ReplaceInvalidUtf8(std::string_view bytes) {
  std::string out;
  out.reserve(bytes.size());
  const auto* p = reinterpret_cast<const unsigned char*>(bytes.data());
  const std::size_t n = bytes.size();
  std::size_t i = 0;
  while (i < n) {
    std::size_t subpart = 0;
    const std::size_t length = SequenceAt(p + i, n - i, &subpart);
    if (length == 0) {
      out += "\xEF\xBF\xBD";
      i += subpart;
      continue;
    }
    out.append(bytes.substr(i, length));
    i += length;
  }
  return out;
}

std::size_t CompleteUtf8Prefix(std::string_view bytes) {
  const auto* p = reinterpret_cast<const unsigned char*>(bytes.data());
  const std::size_t n = bytes.size();
  // Look back at most three bytes for the start of the last sequence.
  std::size_t start = n;
  for (std::size_t back = 1; back <= 4 && back <= n; ++back) {
    if ((p[n - back] & 0xC0U) != 0x80U) {
      start = n - back;
      break;
    }
  }
  if (start == n) {
    return n;  // only continuation bytes at the end: nothing can complete them
  }
  std::size_t subpart = 0;
  const std::size_t length = SequenceAt(p + start, n - start, &subpart);
  if (length != 0) {
    return n;  // the last sequence is complete
  }
  // Incomplete only if the subpart runs to the end (more bytes could finish it).
  return start + subpart == n && subpart < 4 && p[start] >= 0xC2 && p[start] <= 0xF4 ? start : n;
}

void ToNfc(std::vector<char32_t>& text) {
  if (QuickCheckNfc(text)) {
    return;
  }
  // Canonical decomposition.
  std::vector<char32_t> decomposed;
  decomposed.reserve(text.size() + (text.size() / 4));
  for (const char32_t cp : text) {
    Decompose(cp, decomposed);
  }
  // Canonical ordering: stable-sort each run of nonzero combining classes.
  for (std::size_t i = 0; i < decomposed.size();) {
    if (CombiningClass(decomposed[i]) == 0) {
      ++i;
      continue;
    }
    std::size_t j = i;
    while (j < decomposed.size() && CombiningClass(decomposed[j]) != 0) {
      ++j;
    }
    if (j - i > 1) {
      std::stable_sort(
          decomposed.begin() + static_cast<std::ptrdiff_t>(i),
          decomposed.begin() + static_cast<std::ptrdiff_t>(j),
          [](char32_t a, char32_t b) { return CombiningClass(a) < CombiningClass(b); });
    }
    i = j;
  }
  // Canonical composition.
  text.clear();
  std::size_t starter = 0;
  bool have_starter = false;
  std::uint8_t last_ccc = 0;
  for (const char32_t cp : decomposed) {
    const std::uint8_t ccc = CombiningClass(cp);
    if (have_starter) {
      const bool adjacent = text.size() - 1 == starter;
      if (adjacent || (last_ccc != 0 && last_ccc < ccc)) {
        if (const char32_t composite = Compose(text[starter], cp); composite != 0) {
          text[starter] = composite;
          continue;
        }
      }
    }
    if (ccc == 0) {
      starter = text.size();
      have_starter = true;
    }
    last_ccc = ccc;
    text.push_back(cp);
  }
}

}  // namespace jitllm::tokenizer::unicode
