// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "base/json.h"

#include <algorithm>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace llmp::base::json {

std::string Error::ToString() const { return std::format("json: {} (byte {})", reason, offset); }

std::optional<std::size_t> FirstInvalidUtf8(std::string_view bytes) {
  const auto* p = reinterpret_cast<const unsigned char*>(bytes.data());
  const std::size_t n = bytes.size();
  std::size_t i = 0;
  while (i < n) {
    const unsigned c = p[i];
    if (c < 0x80) {
      ++i;
      continue;
    }
    std::size_t length = 0;
    unsigned low = 0x80;
    unsigned high = 0xBF;
    if (c >= 0xC2 && c <= 0xDF) {
      length = 2;
    } else if (c >= 0xE0 && c <= 0xEF) {
      length = 3;
      if (c == 0xE0) {
        low = 0xA0;
      } else if (c == 0xED) {
        high = 0x9F;
      }
    } else if (c >= 0xF0 && c <= 0xF4) {
      length = 4;
      if (c == 0xF0) {
        low = 0x90;
      } else if (c == 0xF4) {
        high = 0x8F;
      }
    } else {
      return i;
    }
    if (n - i < length) {
      return i;
    }
    if (p[i + 1] < low || p[i + 1] > high) {
      return i;
    }
    for (std::size_t k = 2; k < length; ++k) {
      if (p[i + k] < 0x80 || p[i + k] > 0xBF) {
        return i;
      }
    }
    i += length;
  }
  return std::nullopt;
}

void AppendQuoted(std::string_view text, std::string& out) {
  static constexpr std::string_view kHex = "0123456789abcdef";
  out.push_back('"');
  for (const char ch : text) {
    const auto c = static_cast<unsigned char>(ch);
    switch (c) {
      case '"':
        out += "\\\"";
        break;
      case '\\':
        out += "\\\\";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
      case '\b':
        out += "\\b";
        break;
      case '\f':
        out += "\\f";
        break;
      default:
        if (c < 0x20) {
          out += "\\u00";
          out.push_back(kHex[c >> 4U]);
          out.push_back(kHex[c & 0xFU]);
        } else {
          out.push_back(ch);
        }
    }
  }
  out.push_back('"');
}

// A Document::Node's bytes (ParseWorkingBytes; checked in Parser).
constexpr std::size_t kNodeBytes = 12;

class Parser {
 public:
  Parser(std::string_view text, const Limits& limits, Document& doc)
      : text_(text), limits_(limits), doc_(doc) {}

  // Exactly reserved, so no table doubles past what ParseWorkingBytes
  // charged: `values` nodes, children and pending children, and the
  // strings' bytes.
  void Reserve(std::size_t values, std::size_t string_bytes) {
    doc_.nodes_.reserve(values);
    doc_.kids_.reserve(values);
    doc_.text_.reserve(string_bytes);
    pending_.reserve(values);
  }
  static_assert(sizeof(Document::Node) == kNodeBytes, "ParseWorkingBytes counts a node's bytes");

  std::expected<void, Error> Run() {
    doc_.nodes_.emplace_back();  // the root is node 0
    SkipSpace();
    if (auto v = ParseValue(0, 0); !v) {
      return v;
    }
    SkipSpace();
    if (pos_ != text_.size()) {
      return Fail("text after the value");
    }
    return {};
  }

 private:
  std::unexpected<Error> Fail(std::string_view reason) const {
    return std::unexpected(Error{reason, pos_});
  }

  void SkipSpace() {
    while (pos_ < text_.size()) {
      const char c = text_[pos_];
      if (c != ' ' && c != '\t' && c != '\n' && c != '\r') {
        return;
      }
      ++pos_;
    }
  }

  bool Consume(std::string_view literal) {
    if (text_.substr(pos_, literal.size()) != literal) {
      return false;
    }
    pos_ += literal.size();
    return true;
  }

  std::expected<std::uint32_t, Error> NewNode() {
    if (doc_.nodes_.size() >= limits_.max_values ||
        doc_.nodes_.size() >= std::numeric_limits<std::uint32_t>::max()) {
      return Fail("too many values");
    }
    doc_.nodes_.emplace_back();
    return static_cast<std::uint32_t>(doc_.nodes_.size() - 1);
  }

  std::expected<void, Error> AppendText(std::string_view bytes) {
    if (limits_.max_string_bytes - std::min(limits_.max_string_bytes, doc_.text_.size()) <
        bytes.size()) {
      return Fail("too many string bytes");
    }
    doc_.text_.append(bytes);
    return {};
  }

  // Parses the value at pos_ into node `index`.
  std::expected<void, Error> ParseValue(std::uint32_t index, std::size_t depth) {
    if (pos_ >= text_.size()) {
      return Fail("value expected");
    }
    const char c = text_[pos_];
    Document::Node node;
    switch (c) {
      case '{':
      case '[':
        return ParseContainer(index, depth, c == '{');
      case '"': {
        auto s = ParseString();
        if (!s) {
          return std::unexpected(s.error());
        }
        doc_.nodes_[index] = *s;
        return {};
      }
      case 't':
        if (!Consume("true")) {
          return Fail("invalid literal");
        }
        node.kind = Kind::kTrue;
        break;
      case 'f':
        if (!Consume("false")) {
          return Fail("invalid literal");
        }
        node.kind = Kind::kFalse;
        break;
      case 'n':
        if (!Consume("null")) {
          return Fail("invalid literal");
        }
        node.kind = Kind::kNull;
        break;
      default: {
        auto number = ParseNumber();
        if (!number) {
          return std::unexpected(number.error());
        }
        node = *number;
      }
    }
    doc_.nodes_[index] = node;
    return {};
  }

  std::expected<Document::Node, Error> ParseNumber() {
    const std::size_t start = pos_;
    auto digits = [&] {
      const std::size_t from = pos_;
      while (pos_ < text_.size() && text_[pos_] >= '0' && text_[pos_] <= '9') {
        ++pos_;
      }
      return pos_ - from;
    };
    if (pos_ < text_.size() && text_[pos_] == '-') {
      ++pos_;
    }
    if (pos_ < text_.size() && text_[pos_] == '0') {
      ++pos_;
    } else if (digits() == 0) {
      return Fail("invalid number");
    }
    bool integer = true;
    if (pos_ < text_.size() && text_[pos_] == '.') {
      ++pos_;
      integer = false;
      if (digits() == 0) {
        return Fail("invalid number");
      }
    }
    if (pos_ < text_.size() && (text_[pos_] == 'e' || text_[pos_] == 'E')) {
      ++pos_;
      integer = false;
      if (pos_ < text_.size() && (text_[pos_] == '+' || text_[pos_] == '-')) {
        ++pos_;
      }
      if (digits() == 0) {
        return Fail("invalid number");
      }
    }
    Document::Node node{.kind = Kind::kNumber,
                        .integer = integer,
                        .size = static_cast<std::uint32_t>(pos_ - start),
                        .first = static_cast<std::uint32_t>(doc_.text_.size())};
    if (auto t = AppendText(text_.substr(start, pos_ - start)); !t) {
      return std::unexpected(t.error());
    }
    return node;
  }

  static int HexValue(char c) {
    if (c >= '0' && c <= '9') {
      return c - '0';
    }
    if (c >= 'a' && c <= 'f') {
      return c - 'a' + 10;
    }
    if (c >= 'A' && c <= 'F') {
      return c - 'A' + 10;
    }
    return -1;
  }

  std::optional<std::uint32_t> Hex4() {
    if (text_.size() - pos_ < 4) {
      return std::nullopt;
    }
    std::uint32_t value = 0;
    for (std::size_t i = 0; i < 4; ++i) {
      const int h = HexValue(text_[pos_ + i]);
      if (h < 0) {
        return std::nullopt;
      }
      value = (value << 4U) | static_cast<std::uint32_t>(h);
    }
    pos_ += 4;
    return value;
  }

  static void AppendUtf8(std::uint32_t cp, std::string& out) {
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

  // Parses the string at pos_ (at its opening quote) into a string node.
  std::expected<Document::Node, Error> ParseString() {
    ++pos_;  // the opening quote
    std::string scratch;
    while (true) {
      // Copy a run without escapes in one piece.
      const std::size_t run = pos_;
      while (pos_ < text_.size() && text_[pos_] != '"' && text_[pos_] != '\\' &&
             static_cast<unsigned char>(text_[pos_]) >= 0x20) {
        ++pos_;
      }
      scratch.append(text_.substr(run, pos_ - run));
      if (pos_ >= text_.size()) {
        return Fail("unterminated string");
      }
      const char c = text_[pos_];
      if (c == '"') {
        ++pos_;
        break;
      }
      if (c != '\\') {
        return Fail("control character in a string");
      }
      ++pos_;
      if (pos_ >= text_.size()) {
        return Fail("unterminated string");
      }
      const char e = text_[pos_++];
      switch (e) {
        case '"':
        case '\\':
        case '/':
          scratch.push_back(e);
          break;
        case 'b':
          scratch.push_back('\b');
          break;
        case 'f':
          scratch.push_back('\f');
          break;
        case 'n':
          scratch.push_back('\n');
          break;
        case 'r':
          scratch.push_back('\r');
          break;
        case 't':
          scratch.push_back('\t');
          break;
        case 'u': {
          auto unit = Hex4();
          if (!unit) {
            return Fail("invalid \\u escape");
          }
          std::uint32_t cp = *unit;
          if (cp >= 0xDC00 && cp <= 0xDFFF) {
            return Fail("unpaired surrogate escape");
          }
          if (cp >= 0xD800 && cp <= 0xDBFF) {
            if (!Consume("\\u")) {
              return Fail("unpaired surrogate escape");
            }
            auto low = Hex4();
            if (!low || *low < 0xDC00 || *low > 0xDFFF) {
              return Fail("unpaired surrogate escape");
            }
            cp = 0x10000 + ((cp - 0xD800) << 10U) + (*low - 0xDC00);
          }
          AppendUtf8(cp, scratch);
          break;
        }
        default:
          return Fail("invalid escape");
      }
    }
    Document::Node node{.kind = Kind::kString,
                        .integer = false,
                        .size = 0,
                        .first = static_cast<std::uint32_t>(doc_.text_.size())};
    if (scratch.size() > std::numeric_limits<std::uint32_t>::max()) {
      return Fail("string too long");
    }
    node.size = static_cast<std::uint32_t>(scratch.size());
    if (auto t = AppendText(scratch); !t) {
      return std::unexpected(t.error());
    }
    return node;
  }

  std::expected<void, Error> ParseContainer(std::uint32_t index, std::size_t depth, bool object) {
    if (depth + 1 > limits_.max_depth) {
      return Fail("nesting too deep");
    }
    ++pos_;  // the bracket
    const std::size_t base = pending_.size();
    SkipSpace();
    const char close = object ? '}' : ']';
    if (pos_ < text_.size() && text_[pos_] == close) {
      ++pos_;
    } else {
      while (true) {
        if (object) {
          if (pos_ >= text_.size() || text_[pos_] != '"') {
            return Fail("member name expected");
          }
          auto key = NewNode();
          if (!key) {
            return std::unexpected(key.error());
          }
          auto s = ParseString();
          if (!s) {
            return std::unexpected(s.error());
          }
          doc_.nodes_[*key] = *s;
          pending_.push_back(*key);
          SkipSpace();
          if (pos_ >= text_.size() || text_[pos_] != ':') {
            return Fail("':' expected");
          }
          ++pos_;
          SkipSpace();
        }
        auto child = NewNode();
        if (!child) {
          return std::unexpected(child.error());
        }
        pending_.push_back(*child);
        if (auto v = ParseValue(*child, depth + 1); !v) {
          return v;
        }
        SkipSpace();
        if (pos_ < text_.size() && text_[pos_] == ',') {
          ++pos_;
          SkipSpace();
          continue;
        }
        if (pos_ < text_.size() && text_[pos_] == close) {
          ++pos_;
          break;
        }
        return Fail(object ? "',' or '}' expected" : "',' or ']' expected");
      }
    }
    const std::size_t count = pending_.size() - base;
    Document::Node node{.kind = object ? Kind::kObject : Kind::kArray,
                        .integer = false,
                        .size = static_cast<std::uint32_t>(object ? count / 2 : count),
                        .first = static_cast<std::uint32_t>(doc_.kids_.size())};
    doc_.kids_.insert(doc_.kids_.end(), pending_.begin() + static_cast<std::ptrdiff_t>(base),
                      pending_.end());
    pending_.resize(base);
    doc_.nodes_[index] = node;
    if (object && node.size > 1) {
      std::vector<std::string_view> keys;
      keys.reserve(node.size);
      for (std::uint32_t i = 0; i < node.size; ++i) {
        const auto& k = doc_.nodes_[doc_.kids_[node.first + (2 * i)]];
        keys.emplace_back(doc_.text_.data() + k.first, k.size);
      }
      std::ranges::sort(keys);
      if (std::ranges::adjacent_find(keys) != keys.end()) {
        return Fail("duplicate member name");
      }
    }
    return {};
  }

  std::string_view text_;
  const Limits& limits_;
  Document& doc_;
  std::size_t pos_ = 0;
  std::vector<std::uint32_t> pending_;  // children of the containers being parsed
};

std::size_t MaxValues(std::string_view text) {
  std::size_t values = 1;
  bool in_string = false;
  bool escaped = false;
  for (const char c : text) {
    if (in_string) {
      if (escaped) {
        escaped = false;
      } else if (c == '\\') {
        escaped = true;
      } else if (c == '"') {
        in_string = false;
      }
      continue;
    }
    if (c == '"') {
      in_string = true;
    } else if (c == '[' || c == '{' || c == ',' || c == ':') {
      ++values;
    }
  }
  return values;
}

std::size_t ParseWorkingBytes(std::string_view text) {
  // A value's node, its place among its parent's children and while its
  // parent is parsed; and the unescaped strings and number texts, at most
  // the text's bytes.
  return (MaxValues(text) * (kNodeBytes + (2 * sizeof(std::uint32_t)))) + text.size();
}

std::expected<Document, Error> Parse(std::string_view text, const Limits& limits) {
  if (text.size() > limits.max_bytes) {
    return std::unexpected(Error{"document too large", 0});
  }
  if (auto bad = FirstInvalidUtf8(text)) {
    return std::unexpected(Error{"invalid UTF-8", *bad});
  }
  Document doc;
  Parser parser(text, limits, doc);
  // At most max_values, which also bounds the scan's count.
  parser.Reserve(std::min(MaxValues(text), limits.max_values),
                 std::min(text.size(), limits.max_string_bytes));
  if (auto r = parser.Run(); !r) {
    return std::unexpected(r.error());
  }
  return doc;
}

Kind Value::kind() const { return document_->nodes_[node_].kind; }

std::string_view Value::string() const {
  const auto& n = document_->nodes_[node_];
  if (n.kind != Kind::kString) {
    return {};
  }
  return {document_->text_.data() + n.first, n.size};
}

std::string_view Value::number() const {
  const auto& n = document_->nodes_[node_];
  if (n.kind != Kind::kNumber) {
    return {};
  }
  return {document_->text_.data() + n.first, n.size};
}

bool Value::is_integer() const {
  const auto& n = document_->nodes_[node_];
  return n.kind == Kind::kNumber && n.integer;
}

std::optional<std::int64_t> Value::int64() const {
  if (!is_integer()) {
    return std::nullopt;
  }
  const std::string_view t = number();
  std::int64_t value = 0;
  const auto [end, ec] = std::from_chars(t.data(), t.data() + t.size(), value);
  if (ec != std::errc() || end != t.data() + t.size()) {
    return std::nullopt;
  }
  return value;
}

std::optional<double> Value::float64() const {
  if (!is_number()) {
    return std::nullopt;
  }
  const std::string_view t = number();
  double value = 0;
  const auto [end, ec] = std::from_chars(t.data(), t.data() + t.size(), value);
  if (ec == std::errc() && end == t.data() + t.size()) {
    return value;
  }
  if (ec != std::errc::result_out_of_range) {
    return std::nullopt;
  }
  // Out of range: infinity when the magnitude is huge, zero when tiny, as
  // a correctly rounding reader gives. The magnitude is the power of ten of
  // the first nonzero digit plus the exponent.
  const bool negative = t.starts_with('-');
  std::string_view mantissa = negative ? t.substr(1) : t;
  std::int64_t exponent = 0;
  if (const auto e = mantissa.find_first_of("eE"); e != std::string_view::npos) {
    std::string_view digits = mantissa.substr(e + 1);
    mantissa = mantissa.substr(0, e);
    if (digits.starts_with('+')) {
      digits.remove_prefix(1);
    }
    const auto [ep, eec] = std::from_chars(digits.data(), digits.data() + digits.size(), exponent);
    if (eec == std::errc::result_out_of_range) {
      exponent = digits.starts_with('-') ? std::numeric_limits<std::int64_t>::min() / 4
                                         : std::numeric_limits<std::int64_t>::max() / 4;
    }
  }
  const std::size_t dot = std::min(mantissa.find('.'), mantissa.size());
  const std::size_t first = mantissa.find_first_of("123456789");
  std::int64_t magnitude = 0;
  if (first < dot) {
    magnitude = static_cast<std::int64_t>(dot - first - 1);
  } else if (first != std::string_view::npos) {
    magnitude = -static_cast<std::int64_t>(first - dot);
  }
  const bool huge = magnitude + exponent > 0;
  const double inf = std::numeric_limits<double>::infinity();
  if (huge) {
    return negative ? -inf : inf;
  }
  return negative ? -0.0 : 0.0;
}

std::size_t Value::size() const {
  const auto& n = document_->nodes_[node_];
  return n.kind == Kind::kArray || n.kind == Kind::kObject ? n.size : 0;
}

Value Value::at(std::size_t i) const {
  const auto& n = document_->nodes_[node_];
  return {document_, document_->kids_[n.first + i]};
}

std::string_view Value::key(std::size_t i) const {
  const auto& n = document_->nodes_[node_];
  return Value(document_, document_->kids_[n.first + (2 * i)]).string();
}

Value Value::member(std::size_t i) const {
  const auto& n = document_->nodes_[node_];
  return {document_, document_->kids_[n.first + (2 * i) + 1]};
}

std::optional<Value> Value::find(std::string_view name) const {
  if (!is_object()) {
    return std::nullopt;
  }
  for (std::size_t i = 0; i < size(); ++i) {
    if (key(i) == name) {
      return member(i);
    }
  }
  return std::nullopt;
}

}  // namespace llmp::base::json
