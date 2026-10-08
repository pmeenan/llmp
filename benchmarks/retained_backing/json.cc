// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "retained_backing/json.h"

#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <limits>
#include <string>
#include <string_view>
#include <utility>

namespace llmp::rb {

class JsonParser {
 public:
  explicit JsonParser(std::string_view text) : text_(text) {}

  std::expected<Json, std::string> Document() {
    Json value;
    if (!Value(value, 0)) {
      return std::unexpected(std::format("JSON: {} at byte {}", error_, at_));
    }
    SkipSpace();
    if (at_ != text_.size()) {
      return std::unexpected(std::format("JSON: trailing bytes at byte {}", at_));
    }
    return value;
  }

 private:
  static constexpr int kMaxDepth = 16;

  bool Fail(const char* what) {
    error_ = what;
    return false;
  }

  void SkipSpace() {
    while (at_ < text_.size() &&
           (text_[at_] == ' ' || text_[at_] == '\n' || text_[at_] == '\r' || text_[at_] == '\t')) {
      ++at_;
    }
  }

  bool Literal(std::string_view word) {
    if (text_.substr(at_, word.size()) != word) {
      return Fail("unknown literal");
    }
    at_ += word.size();
    return true;
  }

  bool Value(Json& out, int depth) {
    if (depth > kMaxDepth) {
      return Fail("nesting too deep");
    }
    SkipSpace();
    if (at_ >= text_.size()) {
      return Fail("unexpected end");
    }
    const char c = text_[at_];
    if (c == '{') {
      out.kind_ = Json::Kind::kObject;
      return Object(out, depth);
    }
    if (c == '[') {
      out.kind_ = Json::Kind::kArray;
      return Array(out, depth);
    }
    if (c == '"') {
      out.kind_ = Json::Kind::kString;
      return String(out.string_);
    }
    if (c == 't' || c == 'f') {
      out.kind_ = Json::Kind::kBool;
      out.boolean_ = c == 't';
      return Literal(c == 't' ? "true" : "false");
    }
    if (c == 'n') {
      out.kind_ = Json::Kind::kNull;
      return Literal("null");
    }
    out.kind_ = Json::Kind::kInt;
    return Integer(out.integer_);
  }

  bool Integer(std::int64_t& out) {
    const bool negative = text_[at_] == '-';
    if (negative) {
      ++at_;
    }
    const std::size_t start = at_;
    std::uint64_t magnitude = 0;
    while (at_ < text_.size() && text_[at_] >= '0' && text_[at_] <= '9') {
      const auto digit = static_cast<std::uint64_t>(text_[at_] - '0');
      if (magnitude > (std::numeric_limits<std::uint64_t>::max() - digit) / 10) {
        return Fail("integer out of range");
      }
      magnitude = (magnitude * 10) + digit;
      ++at_;
    }
    if (at_ == start || (at_ - start > 1 && text_[start] == '0')) {
      return Fail("not an integer");
    }
    if (at_ < text_.size() && (text_[at_] == '.' || text_[at_] == 'e' || text_[at_] == 'E')) {
      return Fail("only integers are accepted");
    }
    const auto limit = static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
    if (magnitude > limit + (negative ? 1U : 0U)) {
      return Fail("integer out of range");
    }
    if (negative) {
      out = magnitude == limit + 1U ? std::numeric_limits<std::int64_t>::min()
                                    : -static_cast<std::int64_t>(magnitude);
    } else {
      out = static_cast<std::int64_t>(magnitude);
    }
    return true;
  }

  bool String(std::string& out) {
    ++at_;  // the opening quote
    while (at_ < text_.size()) {
      const char c = text_[at_++];
      if (c == '"') {
        return true;
      }
      if (static_cast<unsigned char>(c) < 0x20 || static_cast<unsigned char>(c) > 0x7e) {
        return Fail("strings are printable ASCII");
      }
      if (c != '\\') {
        out.push_back(c);
        continue;
      }
      if (at_ >= text_.size()) {
        break;
      }
      const char escaped = text_[at_++];
      switch (escaped) {
        case '"':
        case '\\':
        case '/':
          out.push_back(escaped);
          break;
        case 'n':
          out.push_back('\n');
          break;
        case 't':
          out.push_back('\t');
          break;
        default:
          return Fail("unsupported escape");
      }
    }
    return Fail("unterminated string");
  }

  bool Array(Json& out, int depth) {
    ++at_;
    SkipSpace();
    if (at_ < text_.size() && text_[at_] == ']') {
      ++at_;
      return true;
    }
    while (true) {
      Json item;
      if (!Value(item, depth + 1)) {
        return false;
      }
      out.items_.push_back(std::move(item));
      SkipSpace();
      if (at_ >= text_.size()) {
        return Fail("unterminated array");
      }
      const char c = text_[at_++];
      if (c == ']') {
        return true;
      }
      if (c != ',') {
        return Fail("expected , or ]");
      }
    }
  }

  bool Object(Json& out, int depth) {
    ++at_;
    SkipSpace();
    if (at_ < text_.size() && text_[at_] == '}') {
      ++at_;
      return true;
    }
    while (true) {
      SkipSpace();
      if (at_ >= text_.size() || text_[at_] != '"') {
        return Fail("expected a key");
      }
      std::string key;
      if (!String(key)) {
        return false;
      }
      if (out.Find(key) != nullptr) {
        return Fail("duplicate key");
      }
      SkipSpace();
      if (at_ >= text_.size() || text_[at_] != ':') {
        return Fail("expected :");
      }
      ++at_;
      Json value;
      if (!Value(value, depth + 1)) {
        return false;
      }
      out.members_.emplace_back(std::move(key), std::move(value));
      SkipSpace();
      if (at_ >= text_.size()) {
        return Fail("unterminated object");
      }
      const char c = text_[at_++];
      if (c == '}') {
        return true;
      }
      if (c != ',') {
        return Fail("expected , or }");
      }
    }
  }

  std::string_view text_;
  std::size_t at_ = 0;
  const char* error_ = "";
};

std::expected<Json, std::string> Json::Parse(std::string_view text) {
  return JsonParser(text).Document();
}

const Json* Json::Find(std::string_view key) const {
  for (const auto& [name, value] : members_) {
    if (name == key) {
      return &value;
    }
  }
  return nullptr;
}

}  // namespace llmp::rb
