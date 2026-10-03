// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "chat/pycase.h"

#include <cstddef>
#include <string>
#include <string_view>

#include "chat/pyjson.h"
#include "tokenizer/unicode.h"

namespace jitllm::chat {
namespace {

namespace unicode = tokenizer::unicode;

constexpr char32_t kCapitalSigma = 0x03A3;
constexpr char32_t kSmallSigma = 0x03C3;
constexpr char32_t kFinalSigma = 0x03C2;

// str.isspace() of one code point, which the `\s` of Jinja2's title split
// matches: bidirectional class WS, B or S, or category Zs.
bool PythonSpace(char32_t cp) {
  return (cp >= 0x09 && cp <= 0x0D) || (cp >= 0x1C && cp <= 0x20) || cp == 0x85 || cp == 0xA0 ||
         cp == 0x1680 || (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028 || cp == 0x2029 ||
         cp == 0x202F || cp == 0x205F || cp == 0x3000;
}

// What Jinja2's `title` filter splits words at: [-\s({\[<].
bool TitleSeparator(char32_t cp) {
  return cp == '-' || cp == '(' || cp == '{' || cp == '[' || cp == '<' || PythonSpace(cp);
}

// ASCII's case data, which the tables hold too: letters are the cased
// characters (none title-case), and ' . : ^ ` are the case-ignorable ones.
bool AsciiLetter(unsigned char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); }
bool AsciiCaseIgnorable(unsigned char c) {
  return c == '\'' || c == '.' || c == ':' || c == '^' || c == '`';
}
char AsciiUpper(unsigned char c) {
  return c >= 'a' && c <= 'z' ? static_cast<char>(c - 'a' + 'A') : static_cast<char>(c);
}
char AsciiLower(unsigned char c) {
  return c >= 'A' && c <= 'Z' ? static_cast<char>(c - 'A' + 'a') : static_cast<char>(c);
}

// Whether a capital sigma ending before `next` is word-final by the text
// after it: no cased character follows, past case-ignorable ones (CPython's
// handle_capital_sigma; the text before it is the caller's).
bool NothingCasedFollows(std::string_view text, std::size_t next) {
  for (std::size_t at = next; at < text.size();) {
    const auto b0 = static_cast<unsigned char>(text[at]);
    if (b0 < 0x80) {
      if (!AsciiCaseIgnorable(b0)) {
        return !AsciiLetter(b0);
      }
      ++at;
      continue;
    }
    std::size_t length = 0;
    const char32_t cp = DecodeUtf8At(text, at, length);
    if (!unicode::IsCaseIgnorable(cp)) {
      return !unicode::IsCased(cp);
    }
    at += length;
  }
  return true;
}

void AppendMapping(const unicode::CaseMapping& m, std::string& out) {
  for (const char32_t cp : m.view()) {
    unicode::AppendUtf8(cp, out);
  }
}

class StringSink final : public CaseSink {
 public:
  explicit StringSink(std::string& out) : out_(&out) {}
  void Emit(std::size_t /*source_offset*/, std::size_t /*source_length*/,
            std::string_view mapped) override {
    *out_ += mapped;
  }

 private:
  std::string* out_;
};

// One operation's walk over the text, with the context it carries from one
// code point to the next.
class Mapper {
 public:
  Mapper(std::string_view text, CaseOp op, CaseSink& sink) : text_(text), op_(op), sink_(sink) {}

  void Run() {
    while (at_ < text_.size()) {
      if (static_cast<unsigned char>(text_[at_]) < 0x80) {
        AsciiRun();
      } else {
        CodePoint();
      }
    }
  }

 private:
  // A run of ASCII, byte for byte, by the same rules as CodePoint.
  void AsciiRun() {
    const std::size_t start = at_;
    run_.clear();
    for (; at_ < text_.size(); ++at_) {
      const auto c = static_cast<unsigned char>(text_[at_]);
      if (c >= 0x80) {
        break;
      }
      char out = static_cast<char>(c);
      bool context_starts_after = false;
      switch (op_) {
        case CaseOp::kUpper:
          out = AsciiUpper(c);
          break;
        case CaseOp::kLower:
          out = AsciiLower(c);
          break;
        case CaseOp::kCapitalize:
          out = at_ == 0 ? AsciiUpper(c) : AsciiLower(c);  // ASCII title case is upper case
          break;
        case CaseOp::kTitle:
          out = previous_cased_ ? AsciiLower(c) : AsciiUpper(c);
          previous_cased_ = AsciiLetter(c);
          break;
        case CaseOp::kJinjaTitle:
          if (TitleSeparator(c)) {
            word_start_ = true;
            context_starts_after = true;
          } else if (word_start_) {
            out = AsciiUpper(c);
            word_start_ = false;
            context_starts_after = true;
          } else {
            out = AsciiLower(c);
          }
          break;
      }
      if (context_starts_after) {
        cased_before_ = false;
      } else if (!AsciiCaseIgnorable(c)) {
        cased_before_ = AsciiLetter(c);
      }
      run_.push_back(out);
    }
    sink_.Emit(start, at_ - start, run_);
  }

  void CodePoint() {
    std::size_t length = 0;
    const char32_t cp = DecodeUtf8At(text_, at_, length);
    const auto lower = [&] {
      if (cp == kCapitalSigma) {
        const bool final = cased_before_ && NothingCasedFollows(text_, at_ + length);
        unicode::CaseMapping m;
        m.code_points[0] = final ? kFinalSigma : kSmallSigma;
        m.length = 1;
        return m;
      }
      return unicode::ToLowerFull(cp);
    };
    unicode::CaseMapping m;
    bool context_starts_after = false;  // this code point is outside the lower-cased text
    switch (op_) {
      case CaseOp::kUpper:
        m = unicode::ToUpperFull(cp);
        break;
      case CaseOp::kLower:
        m = lower();
        break;
      case CaseOp::kCapitalize:
        m = at_ == 0 ? unicode::ToTitleFull(cp) : lower();
        break;
      case CaseOp::kTitle:
        m = previous_cased_ ? lower() : unicode::ToTitleFull(cp);
        previous_cased_ = unicode::IsCased(cp);
        break;
      case CaseOp::kJinjaTitle:
        if (TitleSeparator(cp)) {
          m = unicode::ToUpperFull(cp);  // a separator piece: its first is upper-cased, as itself
          word_start_ = true;
          context_starts_after = true;
        } else if (word_start_) {
          m = unicode::ToUpperFull(cp);
          word_start_ = false;
          context_starts_after = true;  // item[1:].lower() starts after it
        } else {
          m = lower();
        }
        break;
    }
    if (context_starts_after) {
      cased_before_ = false;
    } else if (!unicode::IsCaseIgnorable(cp)) {
      cased_before_ = unicode::IsCased(cp);
    }
    if (m.length == 1 && m.code_points[0] == cp) {
      sink_.Emit(at_, length, text_.substr(at_, length));
    } else {
      mapped_.clear();
      AppendMapping(m, mapped_);
      sink_.Emit(at_, length, mapped_);
    }
    at_ += length;
  }

  std::string_view text_;
  CaseOp op_;
  CaseSink& sink_;
  std::size_t at_ = 0;
  // Final_Sigma's backward condition: the last code point before this one
  // that is not case-ignorable (within the lower-cased text) is cased.
  bool cased_before_ = false;
  bool previous_cased_ = false;  // str.title(): the previous code point is cased
  bool word_start_ = true;       // the Jinja2 title filter: a piece starts here
  std::string run_;
  std::string mapped_;
};

}  // namespace

void MapPythonCase(std::string_view text, CaseOp op, CaseSink& sink) {
  Mapper(text, op, sink).Run();
}

std::string PythonCase(std::string_view text, CaseOp op) {
  std::string out;
  out.reserve(text.size());
  StringSink sink(out);
  MapPythonCase(text, op, sink);
  return out;
}

namespace {

// str.islower() (lower) or str.isupper() (!lower).
bool PythonIsCase(std::string_view text, bool lower) {
  bool cased = false;
  for (std::size_t at = 0; at < text.size();) {
    const auto c = static_cast<unsigned char>(text[at]);
    if (c < 0x80) {  // ASCII: letters only are cased, none title-case
      const bool upper = c >= 'A' && c <= 'Z';
      const bool lower_letter = c >= 'a' && c <= 'z';
      if (lower ? upper : lower_letter) {
        return false;
      }
      cased = cased || (lower ? lower_letter : upper);
      ++at;
      continue;
    }
    std::size_t length = 0;
    const char32_t cp = DecodeUtf8At(text, at, length);
    const bool other = lower ? unicode::IsUppercase(cp) : unicode::IsLowercase(cp);
    if (other || unicode::GetCategory(cp) == unicode::Category::kLt) {
      return false;
    }
    cased = cased || (lower ? unicode::IsLowercase(cp) : unicode::IsUppercase(cp));
    at += length;
  }
  return cased;
}

}  // namespace

bool PythonIsLower(std::string_view text) { return PythonIsCase(text, true); }
bool PythonIsUpper(std::string_view text) { return PythonIsCase(text, false); }

}  // namespace jitllm::chat
