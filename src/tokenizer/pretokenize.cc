// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "tokenizer/pretokenize.h"

#include <algorithm>
#include <limits>
#include <utility>

#include "base/work_pulse.h"

namespace llmp::tokenizer {

std::string_view PreTokenizerName(PreTokenizer p) {
  switch (p) {
    case PreTokenizer::kByteBpe:
      return "byte-bpe";
    case PreTokenizer::kRawBpe:
      return "raw-bpe";
    case PreTokenizer::kSentencePiece:
      return "sentencepiece";
  }
  return "unknown";
}

std::vector<std::string> DefaultByteSplits() {
  return ReadPreTokenizerPreset("qwen2")->expressions;
}

std::expected<SplitPlan, Error> SplitPlan::Compile(std::span<const std::string> expressions,
                                                   RegexDialect dialect) {
  SplitPlan out;
  if (expressions.size() > out.stages_.max_size())
    return std::unexpected(
        Error{Rule::kBounds, "regex stage storage overflows", expressions.size()});
  out.stages_.reserve(expressions.size());
  for (const auto& expression : expressions) {
    auto regex = Regex::Compile(expression, dialect);
    if (!regex) return std::unexpected(regex.error());
    out.stages_.push_back(std::move(*regex));
  }
  return out;
}

std::uint64_t SplitPlan::ProgramSize() const {
  std::uint64_t size = 1;
  for (const auto& stage : stages_) {
    std::uint64_t next = 0;
    if (__builtin_add_overflow(static_cast<std::uint64_t>(stage.instructions()), std::uint64_t{1},
                               &next) ||
        __builtin_add_overflow(size, next, &size))
      return std::numeric_limits<std::uint64_t>::max();
  }
  return size;
}

RegexBudget SplitPlan::Budget(std::size_t n) const {
  std::uint64_t work = 0;
  if (__builtin_add_overflow(static_cast<std::uint64_t>(n), std::uint64_t{1}, &work) ||
      __builtin_mul_overflow(work, ProgramSize(), &work) ||
      __builtin_mul_overflow(work, kRegexWorkFactor, &work))
    work = std::numeric_limits<std::uint64_t>::max();
  return {work, 0};
}

std::uint64_t SplitPlan::WorkingBytes(std::size_t n) const {
  std::uint64_t peak = 0;
  for (const auto& stage : stages_) peak = std::max(peak, stage.WorkingBytes(n));
  // Two stage vectors plus the caller's output, reserved to exactly n.
  const auto limit = std::numeric_limits<std::uint64_t>::max();
  if (n > (limit - peak) / 12) return limit;
  return peak + 12 * n;
}

std::expected<void, Error> SplitPlan::Split(std::span<const char32_t> text,
                                            std::vector<std::uint32_t>& lengths,
                                            RegexBudget& budget) const {
  if (text.empty()) return {};
  if (text.size() >= std::numeric_limits<std::uint32_t>::max())
    return std::unexpected(Error{Rule::kBounds, "split input code point limit"});
  std::vector<std::uint32_t> current, next;
  current.reserve(text.size());
  next.reserve(text.size());
  current.push_back(static_cast<std::uint32_t>(text.size()));
  for (const auto& stage : stages_) {
    next.clear();
    std::size_t offset = 0;
    for (const auto length : current) {
      if (auto paid = budget.Pay(); !paid) return paid;
      if (auto split = stage.Split(text.subspan(offset, length), next, budget); !split)
        return split;
      offset += length;
    }
    current.swap(next);
  }
  if (auto paid = budget.Pay(current.size()); !paid) return paid;
  lengths.insert(lengths.end(), current.begin(), current.end());
  return {};
}
}  // namespace llmp::tokenizer
