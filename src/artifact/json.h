// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The strict JSON of the v0 manifest and index (docs/artifact-format.md,
// D-056), parsed from untrusted bytes without exceptions.
//
// Accepted: RFC 8259 JSON restricted to what the canonical encoding can
// hold. Numbers are integers only (a fraction, an exponent, NaN or
// Infinity is refused); strings are printable ASCII without escapes (the
// format's strings exclude `"` and `\`, so a canonical document never
// needs one); the whole document is ASCII; duplicate keys are refused.
// Whitespace, unsorted keys, "-0" and a raw DEL parse but make the document
// non-canonical: `Document::canonical()` says whether the bytes are exactly
// what the canonical encoder writes (sorted keys, no whitespace, one
// trailing newline). The caller decides when that matters, as the
// prototype verifier does after its version checks.
//
// Caps, checked while parsing so that a small hostile document cannot make
// the parser build a large tree: nesting depth 8; 2^19 arrays and objects;
// 2^21 separators (`,` and `:`); 2^21 + 2^19 + 1 strings; integers of at
// most 20 characters. Strings are views into the caller's bytes, which must
// outlive the document.

#ifndef LLMP_ARTIFACT_JSON_H_
#define LLMP_ARTIFACT_JSON_H_

#include <cstddef>
#include <cstdint>
#include <expected>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "artifact/error.h"

namespace llmp::artifact::json {

inline constexpr std::size_t kMaxDepth = 8;
inline constexpr std::size_t kMaxContainers = std::size_t{1} << 19U;
inline constexpr std::size_t kMaxSeparators = std::size_t{1} << 21U;
inline constexpr std::size_t kMaxStrings = kMaxSeparators + kMaxContainers + 1;
inline constexpr std::size_t kMaxIntegerChars = 20;

enum class Kind : std::uint8_t { kNull, kFalse, kTrue, kInteger, kString, kArray, kObject };

class Document;

// A value in a document: a cheap handle, valid while its document lives.
class Value {
 public:
  Kind kind() const;
  bool is_object() const { return kind() == Kind::kObject; }
  bool is_array() const { return kind() == Kind::kArray; }
  bool is_string() const { return kind() == Kind::kString; }
  bool is_integer() const { return kind() == Kind::kInteger; }

  // An integer that fits in 64 signed bits; nothing for any other value,
  // including integers outside that range (which every field refuses).
  std::optional<std::int64_t> integer() const;
  // An integer's digits as written; empty for any other value.
  std::string_view raw_integer() const;
  // A string's contents; empty for any other value.
  std::string_view string() const;
  // The number of array elements or object members.
  std::size_t size() const;
  // Array element i (i < size()).
  Value at(std::size_t i) const;
  // Object member i's key and value (i < size()), in document order.
  std::string_view key(std::size_t i) const;
  Value member(std::size_t i) const;
  // The object member named `name`.
  std::optional<Value> find(std::string_view name) const;

 private:
  friend class Document;
  Value(const Document* document, std::uint32_t node) : document_(document), node_(node) {}

  const Document* document_;
  std::uint32_t node_;
};

class Document {
 public:
  Value root() const { return {this, root_}; }
  // Whether the bytes are exactly the canonical encoding of root().
  bool canonical() const { return canonical_; }

 private:
  friend class Value;
  friend std::expected<Document, Error> Parse(std::string_view text);
  friend class Parser;

  struct Node {
    Kind kind = Kind::kNull;
    bool fits = false;        // integers: whether `integer` holds the value
    std::uint32_t size = 0;   // string length, or element / member count
    std::uint32_t first = 0;  // string offset in text, or first index in kids
    std::int64_t integer = 0;
  };

  std::string_view text_;
  std::vector<Node> nodes_;
  // Arrays: element nodes; objects: key node, value node, per member.
  std::vector<std::uint32_t> kids_;
  std::uint32_t root_ = 0;
  bool canonical_ = true;
};

// Parses text. Errors are Rule::kJson, or Rule::kCanonical for a string
// escape; `item` is the byte offset where parsing stopped.
std::expected<Document, Error> Parse(std::string_view text);

// Appends v's canonical encoding (sorted keys, no whitespace, no trailing
// newline) to out.
void Serialize(Value v, std::string& out);

}  // namespace llmp::artifact::json

#endif  // LLMP_ARTIFACT_JSON_H_
