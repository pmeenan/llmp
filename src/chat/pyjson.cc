// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "chat/pyjson.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <string_view>
#include <system_error>

#include "base/json.h"
#include "tokenizer/unicode.h"

namespace jitllm::chat {

void AppendPythonFloat(double value, std::string& out) {
  if (std::isinf(value)) {
    out += value < 0 ? "-Infinity" : "Infinity";
    return;
  }
  if (std::isnan(value)) {
    out += "NaN";
    return;
  }
  std::array<char, 64> buffer{};
  const auto [end, ec] = std::to_chars(buffer.data(), buffer.data() + buffer.size(), value,
                                       std::chars_format::scientific);
  if (ec != std::errc()) {
    out += "NaN";  // unreachable: 64 bytes hold any double
    return;
  }
  std::string_view s(buffer.data(), static_cast<std::size_t>(end - buffer.data()));
  if (s.starts_with('-')) {
    out.push_back('-');
    s.remove_prefix(1);
  }
  const std::size_t e = s.find('e');
  std::string digits;
  for (const char c : s.substr(0, e)) {
    if (c != '.') {
      digits.push_back(c);
    }
  }
  int exponent = 0;
  const std::string_view exp_text = s.substr(e + 1);
  std::from_chars(exp_text.data() + (exp_text.starts_with('+') ? 1 : 0),
                  exp_text.data() + exp_text.size(), exponent);
  const int decpt = exponent + 1;  // digits d1 d2 ... are 0.d1d2... x 10^decpt
  const auto n = static_cast<int>(digits.size());
  if (decpt <= -4 || decpt > 16) {
    out.push_back(digits[0]);
    if (n > 1) {
      out.push_back('.');
      out.append(digits, 1);
    }
    const int x = decpt - 1;
    out.push_back('e');
    out.push_back(x < 0 ? '-' : '+');
    const int magnitude = std::abs(x);
    if (magnitude < 10) {
      out.push_back('0');
    }
    out += std::to_string(magnitude);
    return;
  }
  if (decpt <= 0) {
    out += "0.";
    out.append(static_cast<std::size_t>(-decpt), '0');
    out += digits;
  } else if (decpt < n) {
    out.append(digits, 0, static_cast<std::size_t>(decpt));
    out.push_back('.');
    out.append(digits, static_cast<std::size_t>(decpt));
  } else {
    out += digits;
    out.append(static_cast<std::size_t>(decpt - n), '0');
    out += ".0";
  }
}

namespace {

void AppendString(std::string_view text, std::string& out) {
  // Python's json escapes exactly what RFC 8259's minimal escaping does,
  // with lower-case hex, and leaves DEL and everything non-ASCII alone.
  base::json::AppendQuoted(text, out);
}

void AppendHex(std::uint32_t value, int digits, std::string& out) {
  constexpr std::string_view kHex = "0123456789abcdef";
  for (int i = digits - 1; i >= 0; --i) {
    out.push_back(kHex[(value >> (static_cast<unsigned>(i) * 4U)) & 0xFU]);
  }
}

// Python's str() of a number json.loads read: an int as its digits (-0 is
// 0), a float as its repr.
void AppendPythonNumber(base::json::Value value, std::string& out) {
  if (value.is_integer()) {
    if (const auto i = value.int64()) {
      out += std::to_string(*i);
    } else {
      out += value.number();  // beyond 64 bits: Python's int prints its digits
    }
    return;
  }
  const double x = value.float64().value_or(HUGE_VAL);
  if (std::isinf(x)) {
    out += x < 0 ? "-inf" : "inf";
  } else {
    AppendPythonFloat(x, out);
  }
}

// Python's repr() of a value json.loads read.
void AppendPythonRepr(base::json::Value value, std::string& out) {
  using base::json::Kind;
  switch (value.kind()) {
    case Kind::kNull:
      out += "None";
      return;
    case Kind::kTrue:
      out += "True";
      return;
    case Kind::kFalse:
      out += "False";
      return;
    case Kind::kNumber:
      AppendPythonNumber(value, out);
      return;
    case Kind::kString:
      AppendPythonStringRepr(value.string(), out);
      return;
    case Kind::kArray:
      out.push_back('[');
      for (std::size_t i = 0; i < value.size(); ++i) {
        if (i != 0) {
          out += ", ";
        }
        AppendPythonRepr(value.at(i), out);
      }
      out.push_back(']');
      return;
    case Kind::kObject:
      out.push_back('{');
      for (std::size_t i = 0; i < value.size(); ++i) {
        if (i != 0) {
          out += ", ";
        }
        AppendPythonStringRepr(value.key(i), out);
        out += ": ";
        AppendPythonRepr(value.member(i), out);
      }
      out.push_back('}');
      return;
  }
}

}  // namespace

char32_t DecodeUtf8At(std::string_view s, std::size_t at, std::size_t& length) {
  const auto b0 = static_cast<unsigned char>(s[at]);
  if (b0 < 0x80) {
    length = 1;
    return b0;
  }
  std::size_t n = 0;
  char32_t cp = 0;
  if ((b0 & 0xE0U) == 0xC0U) {
    n = 2;
    cp = b0 & 0x1FU;
  } else if ((b0 & 0xF0U) == 0xE0U) {
    n = 3;
    cp = b0 & 0x0FU;
  } else {
    n = 4;
    cp = b0 & 0x07U;
  }
  n = std::min(n, s.size() - at);
  for (std::size_t i = 1; i < n; ++i) {
    cp = (cp << 6U) | (static_cast<unsigned char>(s[at + i]) & 0x3FU);
  }
  length = n;
  return cp;
}

bool PythonPrintable(char32_t cp) {
  using tokenizer::unicode::Category;
  if (cp == U' ') {
    return true;
  }
  switch (tokenizer::unicode::GetCategory(cp)) {
    case Category::kCc:
    case Category::kCf:
    case Category::kCs:
    case Category::kCo:
    case Category::kCn:
    case Category::kZl:
    case Category::kZp:
    case Category::kZs:
      return false;
    default:
      return true;
  }
}

bool AppendPythonStringRepr(std::string_view s, std::string& out, std::size_t limit) {
  const bool single = !s.contains('\'') || s.contains('"');
  const char quote = single ? '\'' : '"';
  out.push_back(quote);
  for (std::size_t at = 0; at < s.size();) {
    if (out.size() > limit) {
      return false;
    }
    std::size_t length = 0;
    const char32_t cp = DecodeUtf8At(s, at, length);
    if (cp == static_cast<char32_t>(quote) || cp == U'\\') {
      out.push_back('\\');
      out.push_back(static_cast<char>(cp));
    } else if (cp == U'\t') {
      out += "\\t";
    } else if (cp == U'\n') {
      out += "\\n";
    } else if (cp == U'\r') {
      out += "\\r";
    } else if (cp < 0x20 || cp == 0x7F) {
      out += "\\x";
      AppendHex(cp, 2, out);
    } else if (cp >= 0x80 && !PythonPrintable(cp)) {
      if (cp < 0x100) {
        out += "\\x";
        AppendHex(cp, 2, out);
      } else if (cp < 0x10000) {
        out += "\\u";
        AppendHex(cp, 4, out);
      } else {
        out += "\\U";
        AppendHex(cp, 8, out);
      }
    } else {
      out.append(s.substr(at, length));
    }
    at += length;
  }
  out.push_back(quote);
  return out.size() <= limit;
}

namespace {

// The length of a string as AppendString writes it.
std::size_t QuotedSize(std::string_view text) {
  std::size_t n = 2;
  for (const char ch : text) {
    const auto c = static_cast<unsigned char>(ch);
    if (c == '"' || c == '\\' || c == '\n' || c == '\r' || c == '\t' || c == '\b' || c == '\f') {
      n += 2;
    } else {
      n += c < 0x20 ? 6 : 1;
    }
  }
  return n;
}

bool AppendWithin(std::string_view text, std::string& out, std::size_t limit) {
  if (out.size() + text.size() > limit) {
    return false;
  }
  out += text;
  return true;
}

}  // namespace

bool AppendPythonJson(base::json::Value value, std::string& out, std::size_t limit) {
  using base::json::Kind;
  switch (value.kind()) {
    case Kind::kString:
      if (out.size() + QuotedSize(value.string()) > limit) {
        return false;
      }
      AppendString(value.string(), out);
      return true;
    case Kind::kArray:
      if (!AppendWithin("[", out, limit)) {
        return false;
      }
      for (std::size_t i = 0; i < value.size(); ++i) {
        if ((i != 0 && !AppendWithin(", ", out, limit)) ||
            !AppendPythonJson(value.at(i), out, limit)) {
          return false;
        }
      }
      return AppendWithin("]", out, limit);
    case Kind::kObject:
      if (!AppendWithin("{", out, limit)) {
        return false;
      }
      for (std::size_t i = 0; i < value.size(); ++i) {
        if (i != 0 && !AppendWithin(", ", out, limit)) {
          return false;
        }
        if (out.size() + QuotedSize(value.key(i)) > limit) {
          return false;
        }
        AppendString(value.key(i), out);
        if (!AppendWithin(": ", out, limit) || !AppendPythonJson(value.member(i), out, limit)) {
          return false;
        }
      }
      return AppendWithin("}", out, limit);
    default: {
      std::string scalar;  // a number or a literal: a few dozen bytes
      AppendPythonJson(value, scalar);
      return AppendWithin(scalar, out, limit);
    }
  }
}

void AppendPythonStr(base::json::Value value, std::string& out) {
  if (value.is_string()) {
    out += value.string();
  } else {
    AppendPythonRepr(value, out);
  }
}

void AppendPythonJson(base::json::Value value, std::string& out) {
  using base::json::Kind;
  switch (value.kind()) {
    case Kind::kNull:
      out += "null";
      return;
    case Kind::kTrue:
      out += "true";
      return;
    case Kind::kFalse:
      out += "false";
      return;
    case Kind::kString:
      AppendString(value.string(), out);
      return;
    case Kind::kNumber:
      if (value.is_integer()) {
        const std::string_view t = value.number();
        out += t.find_first_not_of("-0") == std::string_view::npos ? "0" : t;
      } else if (const auto x = value.float64()) {
        AppendPythonFloat(*x, out);
      }
      return;
    case Kind::kArray:
      out.push_back('[');
      for (std::size_t i = 0; i < value.size(); ++i) {
        if (i != 0) {
          out += ", ";
        }
        AppendPythonJson(value.at(i), out);
      }
      out.push_back(']');
      return;
    case Kind::kObject:
      out.push_back('{');
      for (std::size_t i = 0; i < value.size(); ++i) {
        if (i != 0) {
          out += ", ";
        }
        AppendString(value.key(i), out);
        out += ": ";
        AppendPythonJson(value.member(i), out);
      }
      out.push_back('}');
      return;
  }
}

}  // namespace jitllm::chat
