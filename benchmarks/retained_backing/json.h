// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// A small strict JSON reader for the retained-backing replay's inputs: the
// swap trace's JSON Lines and its manifest (docs/experiments/retained-backing/).
// Both are identified by SHA-256 before use, so this is not a parser of
// untrusted input; it still rejects anything outside the subset they use:
// objects, arrays, ASCII strings, integers that fit in 64 bits, true, false
// and null. Duplicate keys, fractions, exponents and trailing bytes are
// errors.

#ifndef LLMP_BENCHMARKS_RETAINED_BACKING_JSON_H_
#define LLMP_BENCHMARKS_RETAINED_BACKING_JSON_H_

#include <cstdint>
#include <expected>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace llmp::rb {

class Json {
 public:
  enum class Kind : std::uint8_t { kNull, kBool, kInt, kString, kArray, kObject };

  static std::expected<Json, std::string> Parse(std::string_view text);

  Kind kind() const { return kind_; }
  bool is_null() const { return kind_ == Kind::kNull; }
  bool is_int() const { return kind_ == Kind::kInt; }
  bool is_string() const { return kind_ == Kind::kString; }
  bool is_array() const { return kind_ == Kind::kArray; }
  bool is_object() const { return kind_ == Kind::kObject; }

  bool boolean() const { return boolean_; }
  std::int64_t integer() const { return integer_; }
  const std::string& string() const { return string_; }
  const std::vector<Json>& items() const { return items_; }
  const std::vector<std::pair<std::string, Json>>& members() const { return members_; }

  // The member named `key` of an object, or nullptr.
  const Json* Find(std::string_view key) const;

 private:
  friend class JsonParser;

  Kind kind_ = Kind::kNull;
  bool boolean_ = false;
  std::int64_t integer_ = 0;
  std::string string_;
  std::vector<Json> items_;
  std::vector<std::pair<std::string, Json>> members_;
};

}  // namespace llmp::rb

#endif  // LLMP_BENCHMARKS_RETAINED_BACKING_JSON_H_
