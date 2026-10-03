// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Python's str case operations, exactly as CPython 3.12 performs them on
// Unicode 15.0 (whose case data UCD 15.1.0 repeats unchanged): full
// mappings with SpecialCasing.txt's unconditional entries (ß to SS, İ to
// i̇, ligatures) from tokenizer/unicode.h, and the Final_Sigma rule CPython
// applies when lower-casing Σ. Jinja2's filters and the native renderers
// use them so that case mapping never differs from transformers':
// `upper`, `lower`, `capitalize`, `title` (the filter's own rule and
// str.title's), dictsort's and sort's case folding (str.lower), and the
// `lower` and `upper` tests (str.islower, str.isupper).
//
// Text is well-formed UTF-8 (the interpreter's and JSON's strings are).
// Each operation is linear in the text; a mapping is at most three code
// points, so the output is at most three times the input.

#ifndef JITLLM_CHAT_PYCASE_H_
#define JITLLM_CHAT_PYCASE_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace jitllm::chat {

enum class CaseOp : std::uint8_t {
  kUpper,       // str.upper()
  kLower,       // str.lower()
  kCapitalize,  // str.capitalize(): the first character title-cased, the rest lower-cased
  kTitle,       // str.title(): title case after an uncased character, else lower case
  // Jinja2's `title` filter: the text split around runs of `-`, white space,
  // `(`, `{`, `[` and `<`, each piece's first character upper-cased and the
  // rest of the piece lower-cased on its own.
  kJinjaTitle,
};

// Receives the mapped text in order: source bytes [source_offset,
// source_offset + source_length) map to `mapped`. A piece is either one
// code point, or a run of ASCII that maps byte for byte (mapped.size() ==
// source_length, each output byte from the source byte at its offset).
class CaseSink {
 public:
  virtual void Emit(std::size_t source_offset, std::size_t source_length,
                    std::string_view mapped) = 0;

 protected:
  CaseSink() = default;
  CaseSink(const CaseSink&) = default;
  CaseSink& operator=(const CaseSink&) = default;
  CaseSink(CaseSink&&) = default;
  CaseSink& operator=(CaseSink&&) = default;
  ~CaseSink() = default;
};

void MapPythonCase(std::string_view text, CaseOp op, CaseSink& sink);
std::string PythonCase(std::string_view text, CaseOp op);

// str.islower() and str.isupper(): some cased character, and none of the
// other case (nor title case).
bool PythonIsLower(std::string_view text);
bool PythonIsUpper(std::string_view text);

}  // namespace jitllm::chat

#endif  // JITLLM_CHAT_PYCASE_H_
