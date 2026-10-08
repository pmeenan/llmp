// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// A typed byte count with checked arithmetic (AGENTS.md rule 6; D-050's
// "checked arithmetic"): a sum that would overflow or a difference that
// would go negative is an empty optional, never a wrapped value.

#ifndef LLMP_BASE_BYTES_H_
#define LLMP_BASE_BYTES_H_

#include <compare>
#include <cstdint>
#include <optional>

#include "base/check.h"

namespace llmp::base {

class Bytes {
 public:
  constexpr Bytes() = default;
  constexpr explicit Bytes(std::uint64_t value) : value_(value) {}

  constexpr std::uint64_t value() const { return value_; }
  constexpr auto operator<=>(const Bytes&) const = default;

  constexpr std::optional<Bytes> Plus(Bytes other) const {
    std::uint64_t sum = 0;
    if (__builtin_add_overflow(value_, other.value_, &sum)) {
      return std::nullopt;
    }
    return Bytes(sum);
  }
  constexpr std::optional<Bytes> Minus(Bytes other) const {
    if (other.value_ > value_) {
      return std::nullopt;
    }
    return Bytes(value_ - other.value_);
  }

 private:
  std::uint64_t value_ = 0;
};

consteval Bytes operator""_B(unsigned long long value) { return Bytes(value); }
consteval Bytes operator""_KiB(unsigned long long value) {
  if (value > (UINT64_MAX >> 10U)) {
    Fatal("KiB literal exceeds the byte count range");
  }
  return Bytes(value << 10U);
}
consteval Bytes operator""_MiB(unsigned long long value) {
  if (value > (UINT64_MAX >> 20U)) {
    Fatal("MiB literal exceeds the byte count range");
  }
  return Bytes(value << 20U);
}

}  // namespace llmp::base

#endif  // LLMP_BASE_BYTES_H_
