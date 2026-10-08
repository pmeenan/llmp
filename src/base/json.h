// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// General JSON (RFC 8259), parsed from untrusted bytes without exceptions:
// tokenizer data (tokenizer.json), chat fixtures and, later, request bodies.
// The v0 artifact's canonical subset has its own stricter parser
// (artifact/json.h).
//
// Accepted: exactly RFC 8259. The input must be well-formed UTF-8; strings
// are unescaped into UTF-8 and a \u escape of a lone surrogate is refused;
// numbers keep their text (Value::number) and convert on request; duplicate
// object keys are refused, since readers of the same text disagree on which
// one wins. Leading and trailing whitespace is allowed; anything else after
// the value is not.
//
// Limits bound what a hostile document can make the parser build: the
// input size, nesting depth, value count and unescaped string bytes, each
// checked while parsing. The document owns everything it returns, so it
// does not borrow the input.

#ifndef LLMP_BASE_JSON_H_
#define LLMP_BASE_JSON_H_

#include <cstddef>
#include <cstdint>
#include <expected>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace llmp::base::json {

enum class Kind : std::uint8_t { kNull, kFalse, kTrue, kNumber, kString, kArray, kObject };

struct Limits {
  std::size_t max_bytes = std::size_t{64} << 20U;
  std::size_t max_depth = 64;
  std::size_t max_values = std::size_t{1} << 24U;
  std::size_t max_string_bytes = std::size_t{256} << 20U;  // all strings, keys and numbers
};

struct Error {
  std::string_view reason;   // static text; nothing from the document
  std::uint64_t offset = 0;  // byte offset where parsing stopped

  // "json: <reason> (byte <offset>)".
  std::string ToString() const;
};

class Document;

// A value in a document: a cheap handle, valid while its document lives at
// the same address.
class Value {
 public:
  Kind kind() const;
  bool is_null() const { return kind() == Kind::kNull; }
  bool is_bool() const { return kind() == Kind::kTrue || kind() == Kind::kFalse; }
  bool is_number() const { return kind() == Kind::kNumber; }
  bool is_string() const { return kind() == Kind::kString; }
  bool is_array() const { return kind() == Kind::kArray; }
  bool is_object() const { return kind() == Kind::kObject; }

  // true for `true`; false for anything else.
  bool boolean() const { return kind() == Kind::kTrue; }
  // A string's unescaped UTF-8; empty for any other value.
  std::string_view string() const;
  // A number's text as written; empty for any other value.
  std::string_view number() const;
  // Whether a number is written without a fraction or exponent.
  bool is_integer() const;
  // An integer (is_integer()) that fits in 64 signed bits.
  std::optional<std::int64_t> int64() const;
  // Any number, correctly rounded; ±infinity when out of range.
  std::optional<double> float64() const;

  // Array elements or object members.
  std::size_t size() const;
  // Array element i (i < size()).
  Value at(std::size_t i) const;
  // Object member i (i < size()), in document order.
  std::string_view key(std::size_t i) const;
  Value member(std::size_t i) const;
  // The member named `name`, if the object has one.
  std::optional<Value> find(std::string_view name) const;

 private:
  friend class Document;
  Value(const Document* document, std::uint32_t node) : document_(document), node_(node) {}

  const Document* document_;
  std::uint32_t node_;
};

class Document {
 public:
  Value root() const { return {this, 0}; }

 private:
  friend class Value;
  friend class Parser;

  struct Node {
    Kind kind = Kind::kNull;
    bool integer = false;     // numbers: no fraction or exponent
    std::uint32_t size = 0;   // text length, or element / member count
    std::uint32_t first = 0;  // text offset in text_, or first index in kids_
  };

  std::string text_;  // unescaped strings, keys and number texts
  std::vector<Node> nodes_;
  // Arrays: element nodes; objects: key node, value node per member.
  std::vector<std::uint32_t> kids_;
};

std::expected<Document, Error> Parse(std::string_view text, const Limits& limits = {});

// The most a document parsed from `text` can have values (a scan that
// counts the structural characters outside strings: every value but the
// root follows a '[', '{', ',' or ':'). Parse reserves its tables for this
// many, so they never grow by doubling.
std::size_t MaxValues(std::string_view text);

// The bytes Parse allocates for `text` at most besides the text itself:
// its value, child and pending tables for MaxValues and its strings'
// bytes. A caller charges this before parsing untrusted input.
std::size_t ParseWorkingBytes(std::string_view text);

// Whether bytes are well-formed UTF-8 (Unicode 15.1, table 3-7); on failure
// the offset of the first byte of the first ill-formed sequence.
std::optional<std::size_t> FirstInvalidUtf8(std::string_view bytes);

// Appends RFC 8259's minimal escaping of a UTF-8 string, with its quotes.
void AppendQuoted(std::string_view text, std::string& out);

}  // namespace llmp::base::json

#endif  // LLMP_BASE_JSON_H_
