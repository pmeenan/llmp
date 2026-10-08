// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The module's typed checks on parsed JSON, and the string patterns of the
// v0 documents (docs/artifact-format.md). Each check fails with the rule
// the prototype verifier names for it; types are exact (`false` is not 0).
// Internal to src/artifact.

#ifndef LLMP_ARTIFACT_SCHEMA_H_
#define LLMP_ARTIFACT_SCHEMA_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <initializer_list>
#include <limits>
#include <optional>
#include <string_view>

#include "artifact/error.h"
#include "artifact/json.h"
#include "base/check.h"
#include "base/sha256.h"

namespace llmp::artifact::schema {

inline constexpr std::int64_t kMaxInt = std::numeric_limits<std::int64_t>::max();

inline std::unexpected<Error> Fail(Rule rule, std::string_view reason,
                                   std::uint64_t item = kNoItem) {
  return std::unexpected(Error{.rule = rule, .reason = reason, .item = item});
}

// An object whose keys are exactly `required`, plus any of `optional`.
inline std::expected<void, Error> Object(json::Value v,
                                         std::initializer_list<std::string_view> required,
                                         std::initializer_list<std::string_view> optional,
                                         std::string_view reason, std::uint64_t item = kNoItem) {
  if (!v.is_object()) {
    return Fail(Rule::kSchema, reason, item);
  }
  for (std::size_t i = 0; i < v.size(); ++i) {
    const std::string_view key = v.key(i);
    bool known = false;
    for (const std::string_view k : required) {
      known = known || k == key;
    }
    for (const std::string_view k : optional) {
      known = known || k == key;
    }
    if (!known) {
      return Fail(Rule::kSchema, reason, item);
    }
  }
  for (const std::string_view k : required) {
    if (!v.find(k)) {
      return Fail(Rule::kSchema, reason, item);
    }
  }
  return {};
}

// A member the caller has already required.
inline json::Value Get(json::Value object, std::string_view key) {
  const auto found = object.find(key);
  if (!found) {
    base::Fatal("a required member is present");
  }
  return *found;
}

// An integer in [lo, hi].
inline std::expected<std::uint64_t, Error> Int(json::Value v, std::int64_t lo, std::int64_t hi,
                                               std::string_view reason,
                                               std::uint64_t item = kNoItem) {
  const auto value = v.integer();
  if (!value || *value < lo || *value > hi) {
    return Fail(Rule::kSchema, reason, item);
  }
  return static_cast<std::uint64_t>(*value);
}

// Whether v is exactly the integer `want` (not a boolean, not a float).
inline bool IsInt(json::Value v, std::int64_t want) {
  const auto value = v.integer();
  return value && *value == want;
}

inline bool IsString(json::Value v, std::string_view want) {
  return v.is_string() && v.string() == want;
}

// A list of at least `min` elements.
inline std::expected<void, Error> List(json::Value v, std::size_t min, std::string_view reason,
                                       std::uint64_t item = kNoItem) {
  if (!v.is_array() || v.size() < min) {
    return Fail(Rule::kSchema, reason, item);
  }
  return {};
}

// A string matching `pattern`.
template <typename Pattern>
std::expected<std::string_view, Error> Str(json::Value v, Pattern pattern, std::string_view reason,
                                           std::uint64_t item = kNoItem) {
  if (!v.is_string() || !pattern(v.string())) {
    return Fail(Rule::kSchema, reason, item);
  }
  return v.string();
}

inline bool AnyString(std::string_view /*s*/) { return true; }

constexpr bool IsDigit(char c) { return c >= '0' && c <= '9'; }

constexpr bool IsAlnumUnderscore(char c) {
  return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_';
}

// [A-Za-z0-9_][A-Za-z0-9_.-]{0,199}: resource, role, tensor, source and
// architecture names.
constexpr bool IsName(std::string_view s) {
  if (s.empty() || s.size() > 200 || !IsAlnumUnderscore(s[0])) {
    return false;
  }
  return std::ranges::all_of(s.substr(1),
                             [](char c) { return IsAlnumUnderscore(c) || c == '.' || c == '-'; });
}

// 0|[1-9][0-9]{0,max_digits-1}
constexpr bool IsCanonicalNumber(std::string_view s, std::size_t max_digits) {
  if (s.empty() || s.size() > max_digits || (s.size() > 1 && s[0] == '0')) {
    return false;
  }
  return std::ranges::all_of(s, IsDigit);
}

// <name>#<expert>: an expert slice's container entry name.
constexpr bool IsSliceName(std::string_view s) {
  const std::size_t hash = s.rfind('#');
  return hash != std::string_view::npos && IsName(s.substr(0, hash)) &&
         IsCanonicalNumber(s.substr(hash + 1), 7);
}

// ~pad.<n>: a container pad entry's name.
constexpr bool IsPadName(std::string_view s) {
  return s.starts_with("~pad.") && IsCanonicalNumber(s.substr(5), 10);
}

// 64 lowercase hex digits.
constexpr bool IsHex(std::string_view s) {
  return s.size() == 64 &&
         std::ranges::all_of(s, [](char c) { return IsDigit(c) || (c >= 'a' && c <= 'f'); });
}

// The digest 64 lowercase hex digits spell, or nothing.
inline std::optional<base::Sha256Digest> DigestFromHex(std::string_view hex) {
  if (!IsHex(hex)) {
    return std::nullopt;
  }
  base::Sha256Digest out{};
  for (std::size_t i = 0; i < hex.size(); ++i) {
    const char c = hex[i];
    const auto nibble = static_cast<unsigned>(IsDigit(c) ? c - '0' : c - 'a' + 10);
    out[i / 2] = static_cast<std::uint8_t>(out[i / 2] | (nibble << ((i % 2 == 0) ? 4U : 0U)));
  }
  return out;
}

// data/NNNNN.safetensors
constexpr bool IsShardPath(std::string_view s) {
  return s.size() == 22 && s.starts_with("data/") && s.ends_with(".safetensors") &&
         std::ranges::all_of(s.substr(5, 5), IsDigit);
}

// meta/<name>
constexpr bool IsMetaPath(std::string_view s) {
  return s.starts_with("meta/") && IsName(s.substr(5));
}

// Printable ASCII without `"` or `\`, 1-200 characters: converter fields.
constexpr bool IsText(std::string_view s) {
  if (s.empty() || s.size() > 200) {
    return false;
  }
  return std::ranges::all_of(s,
                             [](char c) { return c >= ' ' && c <= '~' && c != '"' && c != '\\'; });
}

}  // namespace llmp::artifact::schema

#endif  // LLMP_ARTIFACT_SCHEMA_H_
