// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef LLMP_TOKENIZER_PRETOKENIZE_H_
#define LLMP_TOKENIZER_PRETOKENIZE_H_

#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "tokenizer/regex.h"

namespace llmp::tokenizer {

// Source algorithms remain distinct; byte-alphabet BPE splits come from data.
enum class PreTokenizer : std::uint8_t { kByteBpe, kRawBpe, kSentencePiece };
std::string_view PreTokenizerName(PreTokenizer p);

struct PreTokenizerPreset {
  std::vector<std::string> expressions;
  bool ignore_merges = false;
  bool add_bos = false;
  RegexDialect dialect = RegexDialect::kGguf;
};
std::expected<PreTokenizerPreset, Error> ReadPreTokenizerPreset(std::string_view name);
std::vector<std::string> DefaultByteSplits();

class SplitPlan {
 public:
  static std::expected<SplitPlan, Error> Compile(std::span<const std::string> expressions,
                                                 RegexDialect dialect = RegexDialect::kHuggingFace);
  std::uint64_t ProgramSize() const;
  RegexBudget Budget(std::size_t input_bytes) const;
  std::uint64_t WorkingBytes(std::size_t input_bytes) const;
  std::expected<void, Error> Split(std::span<const char32_t> text,
                                   std::vector<std::uint32_t>& lengths, RegexBudget& budget,
                                   std::size_t* stable_prefix = nullptr) const;

 private:
  std::vector<Regex> stages_;
};

}  // namespace llmp::tokenizer
#endif  // LLMP_TOKENIZER_PRETOKENIZE_H_
