// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "artifact/json.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "artifact/error.h"
#include "base/check.h"

namespace llmp::artifact::json {

Kind Value::kind() const { return document_->nodes_[node_].kind; }

std::optional<std::int64_t> Value::integer() const {
  const Document::Node& node = document_->nodes_[node_];
  if (node.kind != Kind::kInteger || !node.fits) {
    return std::nullopt;
  }
  return node.integer;
}

std::string_view Value::raw_integer() const {
  const Document::Node& node = document_->nodes_[node_];
  if (node.kind != Kind::kInteger) {
    return {};
  }
  return document_->text_.substr(node.first, node.size);
}

std::string_view Value::string() const {
  const Document::Node& node = document_->nodes_[node_];
  if (node.kind != Kind::kString) {
    return {};
  }
  return document_->text_.substr(node.first, node.size);
}

std::size_t Value::size() const {
  const Document::Node& node = document_->nodes_[node_];
  return (node.kind == Kind::kArray || node.kind == Kind::kObject) ? node.size : 0;
}

Value Value::at(std::size_t i) const {
  const Document::Node& node = document_->nodes_[node_];
  base::Check(node.kind == Kind::kArray && i < node.size, "json array index in range");
  return {document_, document_->kids_[node.first + i]};
}

std::string_view Value::key(std::size_t i) const {
  const Document::Node& node = document_->nodes_[node_];
  base::Check(node.kind == Kind::kObject && i < node.size, "json member index in range");
  return Value(document_, document_->kids_[node.first + (2 * i)]).string();
}

Value Value::member(std::size_t i) const {
  const Document::Node& node = document_->nodes_[node_];
  base::Check(node.kind == Kind::kObject && i < node.size, "json member index in range");
  return {document_, document_->kids_[node.first + (2 * i) + 1]};
}

std::optional<Value> Value::find(std::string_view name) const {
  if (kind() != Kind::kObject) {
    return std::nullopt;
  }
  for (std::size_t i = 0; i < size(); ++i) {
    if (key(i) == name) {
      return member(i);
    }
  }
  return std::nullopt;
}

class Parser {
 public:
  Parser(std::string_view text, Document& document) : text_(text), document_(document) {}

  std::expected<void, Error> Run() {
    for (std::size_t i = 0; i < text_.size(); ++i) {
      if (static_cast<unsigned char>(text_[i]) >= 0x80) {
        return Fail("non-ASCII byte", i);
      }
    }
    auto root = ParseValue(0);
    if (!root) {
      return std::unexpected(root.error());
    }
    document_.root_ = *root;
    // Only whitespace may follow; the canonical encoding ends in exactly
    // one newline, and has no whitespace anywhere else.
    const std::size_t end = pos_;
    while (pos_ < text_.size() && IsSpace(text_[pos_])) {
      ++pos_;
    }
    if (pos_ != text_.size()) {
      return Fail("data after the document", pos_);
    }
    if (text_.substr(end) != "\n") {
      document_.canonical_ = false;
    }
    return {};
  }

 private:
  static bool IsSpace(char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; }
  static bool IsDigit(char c) { return c >= '0' && c <= '9'; }

  static std::unexpected<Error> Fail(std::string_view reason, std::size_t at,
                                     Rule rule = Rule::kJson) {
    return std::unexpected(Error{.rule = rule, .reason = reason, .item = at});
  }

  void SkipSpace() {
    const std::size_t start = pos_;
    while (pos_ < text_.size() && IsSpace(text_[pos_])) {
      ++pos_;
    }
    if (pos_ != start) {
      document_.canonical_ = false;
    }
  }

  std::uint32_t AddNode(const Document::Node& node) {
    document_.nodes_.push_back(node);
    return static_cast<std::uint32_t>(document_.nodes_.size() - 1);
  }

  std::expected<void, Error> CountSeparator() {
    if (++separators_ > kMaxSeparators) {
      return Fail("more separators than the cap", pos_);
    }
    return {};
  }

  std::expected<std::uint32_t, Error> ParseValue(std::size_t depth) {
    SkipSpace();
    if (pos_ >= text_.size()) {
      return Fail("unexpected end", pos_);
    }
    const char c = text_[pos_];
    if (c == '{' || c == '[') {
      if (depth + 1 > kMaxDepth) {
        return Fail("nesting deeper than the cap", pos_);
      }
      if (++containers_ > kMaxContainers) {
        return Fail("more arrays and objects than the cap", pos_);
      }
      return c == '{' ? ParseObject(depth + 1) : ParseArray(depth + 1);
    }
    if (c == '"') {
      return ParseString();
    }
    if (c == '-' || IsDigit(c)) {
      return ParseInteger();
    }
    for (const auto& [word, kind] : {std::pair{std::string_view("true"), Kind::kTrue},
                                     std::pair{std::string_view("false"), Kind::kFalse},
                                     std::pair{std::string_view("null"), Kind::kNull}}) {
      if (text_.substr(pos_, word.size()) == word) {
        pos_ += word.size();
        return AddNode({.kind = kind});
      }
    }
    return Fail("not a JSON value", pos_);
  }

  std::expected<std::uint32_t, Error> ParseString() {
    const std::size_t start = ++pos_;  // past the quote
    while (true) {
      if (pos_ >= text_.size()) {
        return Fail("unterminated string", start - 1);
      }
      const char c = text_[pos_];
      if (c == '"') {
        break;
      }
      if (c == '\\') {
        return Fail("a string escape (canonical strings need none)", pos_, Rule::kCanonical);
      }
      if (static_cast<unsigned char>(c) < 0x20) {
        return Fail("a control character in a string", pos_);
      }
      if (c == 0x7f) {
        document_.canonical_ = false;  // the canonical encoder escapes DEL
      }
      ++pos_;
    }
    if (++strings_ > kMaxStrings) {
      return Fail("more strings than the cap", start - 1);
    }
    const std::size_t length = pos_ - start;
    ++pos_;  // the closing quote
    return AddNode({.kind = Kind::kString,
                    .size = static_cast<std::uint32_t>(length),
                    .first = static_cast<std::uint32_t>(start)});
  }

  std::expected<std::uint32_t, Error> ParseInteger() {
    const std::size_t start = pos_;
    const bool negative = text_[pos_] == '-';
    if (negative) {
      ++pos_;
    }
    if (pos_ >= text_.size() || !IsDigit(text_[pos_])) {
      return Fail("a malformed number", start);
    }
    const std::size_t digits = pos_;
    if (text_[pos_] == '0') {
      ++pos_;
    } else {
      while (pos_ < text_.size() && IsDigit(text_[pos_])) {
        ++pos_;
      }
    }
    if (pos_ < text_.size() &&
        (text_[pos_] == '.' || text_[pos_] == 'e' || text_[pos_] == 'E' || IsDigit(text_[pos_]))) {
      return Fail("a number that is not a canonical integer", start);
    }
    if (pos_ - start > kMaxIntegerChars) {
      return Fail("an integer longer than the cap", start);
    }
    std::uint64_t magnitude = 0;
    bool overflow = false;
    for (std::size_t i = digits; i < pos_; ++i) {
      overflow =
          overflow || __builtin_mul_overflow(magnitude, 10U, &magnitude) ||
          __builtin_add_overflow(magnitude, static_cast<unsigned>(text_[i] - '0'), &magnitude);
    }
    constexpr auto kMax = static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
    Document::Node node{.kind = Kind::kInteger,
                        .size = static_cast<std::uint32_t>(pos_ - start),
                        .first = static_cast<std::uint32_t>(start)};
    if (!overflow && !negative && magnitude <= kMax) {
      node.fits = true;
      node.integer = static_cast<std::int64_t>(magnitude);
    } else if (!overflow && negative && magnitude <= kMax + 1) {
      node.fits = true;
      node.integer = magnitude == kMax + 1 ? std::numeric_limits<std::int64_t>::min()
                                           : -static_cast<std::int64_t>(magnitude);
      if (magnitude == 0) {
        document_.canonical_ = false;  // "-0" is written "0"
      }
    }
    return AddNode(node);
  }

  // Appends the children collected since `mark` to the kid table.
  std::uint32_t Collect(std::size_t mark) {
    const auto first = static_cast<std::uint32_t>(document_.kids_.size());
    document_.kids_.insert(document_.kids_.end(),
                           stack_.begin() + static_cast<std::ptrdiff_t>(mark), stack_.end());
    stack_.resize(mark);
    return first;
  }

  std::expected<std::uint32_t, Error> ParseArray(std::size_t depth) {
    ++pos_;
    const std::size_t mark = stack_.size();
    SkipSpace();
    if (pos_ < text_.size() && text_[pos_] == ']') {
      ++pos_;
    } else {
      while (true) {
        auto element = ParseValue(depth);
        if (!element) {
          return element;
        }
        stack_.push_back(*element);
        SkipSpace();
        if (pos_ < text_.size() && text_[pos_] == ',') {
          if (auto counted = CountSeparator(); !counted) {
            return std::unexpected(counted.error());
          }
          ++pos_;
          continue;
        }
        if (pos_ < text_.size() && text_[pos_] == ']') {
          ++pos_;
          break;
        }
        return Fail("expected ',' or ']'", pos_);
      }
    }
    const auto count = static_cast<std::uint32_t>(stack_.size() - mark);
    const std::uint32_t first = Collect(mark);
    return AddNode({.kind = Kind::kArray, .size = count, .first = first});
  }

  std::expected<std::uint32_t, Error> ParseObject(std::size_t depth) {
    const std::size_t open = pos_++;
    const std::size_t mark = stack_.size();
    SkipSpace();
    if (pos_ < text_.size() && text_[pos_] == '}') {
      ++pos_;
    } else {
      while (true) {
        SkipSpace();
        if (pos_ >= text_.size() || text_[pos_] != '"') {
          return Fail("expected a key", pos_);
        }
        auto key = ParseString();
        if (!key) {
          return key;
        }
        SkipSpace();
        if (pos_ >= text_.size() || text_[pos_] != ':') {
          return Fail("expected ':'", pos_);
        }
        if (auto counted = CountSeparator(); !counted) {
          return std::unexpected(counted.error());
        }
        ++pos_;
        auto value = ParseValue(depth);
        if (!value) {
          return value;
        }
        stack_.push_back(*key);
        stack_.push_back(*value);
        SkipSpace();
        if (pos_ < text_.size() && text_[pos_] == ',') {
          if (auto counted = CountSeparator(); !counted) {
            return std::unexpected(counted.error());
          }
          ++pos_;
          continue;
        }
        if (pos_ < text_.size() && text_[pos_] == '}') {
          ++pos_;
          break;
        }
        return Fail("expected ',' or '}'", pos_);
      }
    }
    const std::size_t count = (stack_.size() - mark) / 2;
    // Canonical keys are strictly ascending; equal keys are duplicates.
    bool ascending = true;
    for (std::size_t i = 1; i < count && ascending; ++i) {
      ascending = KeyAt(mark, i - 1) < KeyAt(mark, i);
    }
    if (!ascending) {
      keys_.clear();
      for (std::size_t i = 0; i < count; ++i) {
        keys_.push_back(KeyAt(mark, i));
      }
      std::ranges::sort(keys_);
      if (std::ranges::adjacent_find(keys_) != keys_.end()) {
        return Fail("a duplicate key", open);
      }
      document_.canonical_ = false;
    }
    const std::uint32_t first = Collect(mark);
    return AddNode(
        {.kind = Kind::kObject, .size = static_cast<std::uint32_t>(count), .first = first});
  }

  std::string_view KeyAt(std::size_t mark, std::size_t i) const {
    const Document::Node& node = document_.nodes_[stack_[mark + (2 * i)]];
    return text_.substr(node.first, node.size);
  }

  std::string_view text_;
  Document& document_;
  std::size_t pos_ = 0;
  std::size_t containers_ = 0;
  std::size_t separators_ = 0;
  std::size_t strings_ = 0;
  std::vector<std::uint32_t> stack_;
  std::vector<std::string_view> keys_;
};

std::expected<Document, Error> Parse(std::string_view text) {
  if (text.size() > std::numeric_limits<std::uint32_t>::max()) {
    return std::unexpected(Error{.rule = Rule::kJson, .reason = "document too large"});
  }
  Document document;
  document.text_ = text;
  Parser parser(text, document);
  if (auto parsed = parser.Run(); !parsed) {
    return std::unexpected(parsed.error());
  }
  return document;
}

void Serialize(Value v, std::string& out) {
  switch (v.kind()) {
    case Kind::kNull:
      out += "null";
      return;
    case Kind::kFalse:
      out += "false";
      return;
    case Kind::kTrue:
      out += "true";
      return;
    case Kind::kInteger: {
      const std::optional<std::int64_t> value = v.integer();
      if (value) {
        out += std::to_string(*value);
      } else {
        // Outside 64 bits: the digits as written, which cannot be "-0".
        out += v.raw_integer();
      }
      return;
    }
    case Kind::kString:
      out += '"';
      out += v.string();
      out += '"';
      return;
    case Kind::kArray:
      out += '[';
      for (std::size_t i = 0; i < v.size(); ++i) {
        if (i > 0) {
          out += ',';
        }
        Serialize(v.at(i), out);
      }
      out += ']';
      return;
    case Kind::kObject: {
      std::vector<std::size_t> order(v.size());
      for (std::size_t i = 0; i < order.size(); ++i) {
        order[i] = i;
      }
      std::ranges::sort(order, [&](std::size_t a, std::size_t b) { return v.key(a) < v.key(b); });
      out += '{';
      for (std::size_t i = 0; i < order.size(); ++i) {
        if (i > 0) {
          out += ',';
        }
        out += '"';
        out += v.key(order[i]);
        out += "\":";
        Serialize(v.member(order[i]), out);
      }
      out += '}';
      return;
    }
  }
}

}  // namespace llmp::artifact::json
