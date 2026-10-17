// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Shorthand for writing mapping tables (spec/mapping.h) as constant data,
// and for the source values the mapping data carries (a drafter import's
// target, the native MTP drafter's block). Internal to src/spec/mappings
// (and the tests' family fixtures).

#ifndef LLMP_SPEC_MAPPINGS_RULES_H_
#define LLMP_SPEC_MAPPINGS_RULES_H_

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "spec/mapping.h"
#include "spec/spec.h"

namespace llmp::spec::mappings {

// A parameter from a source key; `optional`: left out when the key is
// absent and no later rule gives it.
constexpr ParamRule FromKey(std::string_view slot, std::string_view param, std::string_view key,
                            ValueType type, Transform transform = Transform::kNone,
                            std::string_view layer_types = {}, bool optional = false) {
  return {.slot = slot,
          .param = param,
          .layer_types = layer_types,
          .key = key,
          .type = type,
          .transform = transform,
          .optional = optional};
}

// A parameter from a source key, or if absent, a ratio of two keys;
// `optional`: left out when both the key and the ratio are absent and no later rule gives it.
constexpr ParamRule FromOrRatio(std::string_view slot, std::string_view param, std::string_view key,
                                std::string_view numerator_key, std::string_view divisor_key,
                                ValueType type, Transform transform = Transform::kNone,
                                std::string_view layer_types = {}, bool optional = false) {
  return {.slot = slot,
          .param = param,
          .layer_types = layer_types,
          .key = key,
          .numerator_key = numerator_key,
          .divisor_key = divisor_key,
          .type = type,
          .transform = transform,
          .optional = optional};
}

// A parameter from the ratio of two keys; `optional`: left out when absent and no later rule gives
// it.
constexpr ParamRule FromRatio(std::string_view slot, std::string_view param,
                              std::string_view numerator_key, std::string_view divisor_key,
                              ValueType type, Transform transform = Transform::kNone,
                              std::string_view layer_types = {}, bool optional = false) {
  return {.slot = slot,
          .param = param,
          .layer_types = layer_types,
          .numerator_key = numerator_key,
          .divisor_key = divisor_key,
          .type = type,
          .transform = transform,
          .optional = optional};
}

// A parameter from a key listing a value a layer (or one for all).
constexpr ParamRule PerLayer(std::string_view slot, std::string_view param, std::string_view key,
                             ValueType type, std::string_view layer_types = {}) {
  return {.slot = slot,
          .param = param,
          .layer_types = layer_types,
          .key = key,
          .type = type,
          .per_layer = true};
}

// Constant parameters.
constexpr ParamRule Int(std::string_view slot, std::string_view param, std::int64_t value,
                        std::string_view layer_types = {}) {
  return {.slot = slot,
          .param = param,
          .layer_types = layer_types,
          .type = ValueType::kInt,
          .value = {.integer = value}};
}
constexpr ParamRule Real(std::string_view slot, std::string_view param, double value,
                         std::string_view layer_types = {}) {
  return {.slot = slot,
          .param = param,
          .layer_types = layer_types,
          .type = ValueType::kReal,
          .value = {.real = value}};
}
constexpr ParamRule Flag(std::string_view slot, std::string_view param, bool value,
                         std::string_view layer_types = {}) {
  return {.slot = slot,
          .param = param,
          .layer_types = layer_types,
          .type = ValueType::kBool,
          .value = {.flag = value}};
}
constexpr ParamRule Name(std::string_view slot, std::string_view param, std::string_view value,
                         std::string_view layer_types = {}) {
  return {.slot = slot,
          .param = param,
          .layer_types = layer_types,
          .type = ValueType::kString,
          .value = {.text = value}};
}

// A requirement: the key's value is `value`, or (optional) the key is
// absent.
constexpr ParamRule Require(std::string_view key, std::string_view value, bool optional) {
  return {.key = key, .type = ValueType::kString, .value = {.text = value}, .optional = optional};
}
constexpr ParamRule RequireInt(std::string_view key, std::int64_t value, bool optional) {
  return {.key = key, .type = ValueType::kInt, .value = {.integer = value}, .optional = optional};
}

// A profile's value under a source key: integers as integers, floats as the
// reals they are.
inline SourceValue Value(std::string_view key, std::uint32_t v) {
  return {.key = std::string(key), .value = ParamValue(static_cast<std::int64_t>(v))};
}
inline SourceValue Value(std::string_view key, float v) {
  return {.key = std::string(key), .value = ParamValue(static_cast<double>(v))};
}
inline SourceValue Value(std::string_view key, std::string_view v) {
  return {.key = std::string(key), .value = ParamValue(std::string(v))};
}
// A list of integers (GGUF's arrays of integers or flags).
inline SourceValue Value(std::string_view key, std::vector<std::int64_t> v) {
  return {.key = std::string(key), .value = ParamValue(std::move(v))};
}
// A flag (a source's fact, an approval's gate).
inline SourceValue FlagValue(std::string_view key, bool v) {
  return {.key = std::string(key), .value = ParamValue(v)};
}

}  // namespace llmp::spec::mappings

#endif  // LLMP_SPEC_MAPPINGS_RULES_H_
