// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef LLMP_TOKENIZER_REGEX_H_
#define LLMP_TOKENIZER_REGEX_H_

#include <cstddef>
#include <cstdint>
#include <expected>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

#include "tokenizer/error.h"

namespace llmp::tokenizer {

// Ordered, bounded backtracking over immutable bytecode. No host recursion,
// elapsed-time deadline, or silently approximate result. The same allowance
// travels through every search position, stage and fragment of an encoding.
inline constexpr std::uint64_t kRegexWorkFactor = 16;
// Source semantics within the supported grammar. HF uses Oniguruma line
// anchors; GGUF presets use llama.cpp's non-multiline ECMAScript fallback.
enum class RegexDialect : std::uint8_t { kHuggingFace, kGguf };

struct RegexBudget {
  std::uint64_t remaining = 0;
  std::uint64_t spent = 0;
  std::expected<void, Error> Pay(std::uint64_t amount = 1);
};

class Regex {
 public:
  static std::expected<Regex, Error> Compile(std::string_view expression,
                                             RegexDialect dialect = RegexDialect::kHuggingFace);
  // Includes compiler peak storage as well as the retained program. Checked
  // before Compile: expression size bounds every parser/program allocation.
  static std::uint64_t ConstructionBytes(std::size_t expression_bytes);
  std::size_t instructions() const {
    return code_.size() + classes_.size() + terms_.size() + literals_.size();
  }
  std::uint64_t WorkingBytes(std::size_t code_points) const;
  // Isolated splitting, retaining unmatched gaps. Empty matches establish a
  // boundary, emit no empty piece, then advance the search one code point.
  // Optional stable_prefix: complete split pieces whose decisions did not
  // consult EOF on any attempted ordered path. Appending input cannot change
  // those pieces; a failed earlier alternative is included in this proof.
  // An unmatched gap is open until a match ends it.
  std::expected<void, Error> Split(std::span<const char32_t> text,
                                   std::vector<std::uint32_t>& lengths, RegexBudget& budget,
                                   std::size_t* stable_prefix = nullptr) const;

 private:
  enum class Op : std::uint8_t {
    kClass,
    kLiteral,
    kAheadClass,
    kScan,
    kSplit,
    kJump,
    kEnter,
    kRepeat,
    kNext,
    kExit,
    kBegin,
    kEnd,
    kNotNonSpace,
    kAccept
  };
  struct Instruction {
    Op op = Op::kAccept;
    std::uint32_t a = 0, b = 0, c = 0, d = 0;
  };
  struct Term {
    char32_t first = 0, last = 0;
    std::uint32_t categories = 0;
    bool whitespace = false, inverse = false;
  };
  enum class ClassOp : std::uint8_t { kTerms, kUnion, kIntersection };
  struct Class {
    // Terms: [a,a+b). Algebra: earlier child class indexes a,b.
    std::uint32_t a = 0, b = 0;
    ClassOp op = ClassOp::kTerms;
    bool inverse = false, caseless = false;
  };
  struct ClassFrame {
    std::uint32_t index = 0;
    std::uint8_t phase = 0;
    bool left = false;
  };
  std::vector<Instruction> code_;
  std::vector<Class> classes_;
  std::vector<Term> terms_;
  std::vector<char32_t> literals_;
  std::uint32_t repeats_ = 0;
  std::expected<bool, Error> TestClass(std::uint32_t index, char32_t cp, RegexBudget& budget,
                                       std::vector<ClassFrame>& stack) const;
  std::expected<std::size_t, Error> Match(std::span<const char32_t> text, std::size_t start,
                                          RegexBudget& budget, bool* consulted_end = nullptr) const;
};

}  // namespace llmp::tokenizer
#endif  // LLMP_TOKENIZER_REGEX_H_
