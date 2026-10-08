// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The Unicode Character Database tables behind unicode.h, generated into
// unicode_data.cc by tools/gen-unicode-tables from pinned UCD 15.1.0 files
// (D-088). Only unicode.cc reads them. Every table is constant-initialized.

#ifndef LLMP_TOKENIZER_UNICODE_DATA_H_
#define LLMP_TOKENIZER_UNICODE_DATA_H_

#include <array>
#include <cstdint>
#include <span>
#include <string_view>

namespace llmp::tokenizer::unicode_data {

inline constexpr char32_t kMaxCodePoint = 0x10FFFF;
inline constexpr unsigned kBlockBits = 8;  // 256 code points per block
inline constexpr unsigned kBlocks = (kMaxCodePoint + 1) >> kBlockBits;

// A property byte: the general category in bits 0-4 (unicode.h's Category
// order), White_Space, NFC_QC=No and NFC_QC=Maybe.
inline constexpr std::uint8_t kCategoryMask = 0x1F;
inline constexpr std::uint8_t kWhiteSpaceBit = 0x20;
inline constexpr std::uint8_t kNfcNoBit = 0x40;
inline constexpr std::uint8_t kNfcMaybeBit = 0x80;

using Block = std::array<std::uint8_t, 256>;

struct Decomposition {
  char32_t code_point;
  std::uint16_t offset;  // into kDecompositionParts
  std::uint8_t length;
};

// A primary composite: first + second compose to composite.
struct Composition {
  char32_t first;
  char32_t second;
  char32_t composite;
};

// A code point's full case mapping: `length` code points at `offset` in
// kCaseParts.
struct CaseMapping {
  char32_t code_point;
  std::uint16_t offset;
  std::uint8_t length;
};

// Code points first..last, inclusive.
struct CodePointRange {
  char32_t first;
  char32_t last;
};

extern const std::string_view kVersion;
// Two-stage tables: index[cp >> 8] names the block holding cp's byte.
extern const std::span<const std::uint16_t, kBlocks> kPropertyIndex;
extern const std::span<const Block> kPropertyBlocks;
extern const std::span<const std::uint16_t, kBlocks> kCombiningClassIndex;
extern const std::span<const Block> kCombiningClassBlocks;
// Sorted by code point.
extern const std::span<const Decomposition> kDecompositions;
extern const std::span<const char32_t> kDecompositionParts;
// Sorted by (first, second).
extern const std::span<const Composition> kCompositions;
// Python's full case mappings, where they are not the code point itself;
// sorted by code point.
extern const std::span<const CaseMapping> kLowerFull;
extern const std::span<const CaseMapping> kUpperFull;
extern const std::span<const CaseMapping> kTitleFull;
extern const std::span<const char32_t> kCaseParts;
// The Lowercase, Uppercase, Cased and Case_Ignorable properties: sorted,
// disjoint, not adjacent.
extern const std::span<const CodePointRange> kLowercase;
extern const std::span<const CodePointRange> kUppercase;
extern const std::span<const CodePointRange> kCased;
extern const std::span<const CodePointRange> kCaseIgnorable;

}  // namespace llmp::tokenizer::unicode_data

#endif  // LLMP_TOKENIZER_UNICODE_DATA_H_
