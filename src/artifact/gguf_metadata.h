// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The GGUF metadata an artifact keeps (docs/artifact-format.md, import rule
// 7: a GGUF source's header through its key/value section, with n_tensors
// set to 0), read natively as untrusted input. The file must be exactly a
// complete zero-tensor GGUF header, as the prototype verifier's
// _check_kv_gguf requires: the magic, version 3, no tensors, at most
// kMaxGgufKeys keys, no byte after the last value; keys non-empty and
// unique; every value of a known type, no array of arrays, an array's
// element type valid even when it is empty; general.alignment a u32 power
// of two, general.architecture a string of at most 200 bytes, and every
// `*.expert_count` a u32 (the rules of pinned gguf.cpp). Every length and
// count is checked against the bytes left before anything is read or
// allocated.
//
// Only the values of the keys the caller names are returned; every other
// value is skipped unread (its bytes still bounded).

#ifndef LLMP_ARTIFACT_GGUF_METADATA_H_
#define LLMP_ARTIFACT_GGUF_METADATA_H_

#include <cstddef>
#include <cstdint>
#include <expected>
#include <functional>
#include <map>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "artifact/error.h"

namespace llmp::artifact {

inline constexpr std::uint64_t kMaxGgufKeys = std::uint64_t{1} << 16U;
// The longest wanted array the reader materializes (a vocabulary's worth).
inline constexpr std::uint64_t kMaxGgufWantedArray = std::uint64_t{1} << 20U;
// The longest wanted string.
inline constexpr std::uint64_t kMaxGgufWantedString = std::uint64_t{16} << 20U;

// A wanted key's value. Integers (any GGUF integer type, or bool) as
// int64; an unsigned 64-bit value above INT64_MAX is refused, not wrapped.
// Floats (F32 or F64) as double. Arrays of integers or of floats likewise;
// an array of strings, or of another kind, is kOther.
struct GgufValue {
  enum class Kind : std::uint8_t { kInteger, kFloat, kString, kIntegers, kFloats, kOther };
  Kind kind = Kind::kOther;
  std::uint32_t type = 0;  // the GGUF value type (an array's: 9)
  std::int64_t integer = 0;
  double real = 0.0;
  std::string text;
  std::vector<std::int64_t> integers;
  std::vector<double> reals;
};

using GgufMetadata = std::map<std::string, GgufValue, std::less<>>;

// The wanted keys' values, those present; refused (kFormat for the
// container, kBounds for a cap, kSchema for a key rule) as above.
std::expected<GgufMetadata, Error> ReadGgufMetadata(std::span<const std::byte> bytes,
                                                    std::span<const std::string_view> wanted);

}  // namespace llmp::artifact

#endif  // LLMP_ARTIFACT_GGUF_METADATA_H_
