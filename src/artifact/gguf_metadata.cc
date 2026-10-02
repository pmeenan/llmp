// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "artifact/gguf_metadata.h"

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>
#include <limits>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <utility>

#include "artifact/error.h"

namespace jitllm::artifact {
namespace {

// GGUF value types (gguf.h).
enum : std::uint32_t {
  kU8 = 0,
  kI8 = 1,
  kU16 = 2,
  kI16 = 3,
  kU32 = 4,
  kI32 = 5,
  kF32 = 6,
  kBool = 7,
  kString = 8,
  kArray = 9,
  kU64 = 10,
  kI64 = 11,
  kF64 = 12,
};

// The bytes of a scalar of `type`; 0 for a string, an array or an unknown
// type.
std::size_t ScalarBytes(std::uint32_t type) {
  switch (type) {
    case kU8:
    case kI8:
    case kBool:
      return 1;
    case kU16:
    case kI16:
      return 2;
    case kU32:
    case kI32:
    case kF32:
      return 4;
    case kU64:
    case kI64:
    case kF64:
      return 8;
    default:
      return 0;
  }
}

bool IsInteger(std::uint32_t type) {
  return type != kF32 && type != kF64 && ScalarBytes(type) != 0;
}

bool IsFloat(std::uint32_t type) { return type == kF32 || type == kF64; }

bool KnownType(std::uint32_t type) { return type <= kF64; }

std::unexpected<Error> Fail(Rule rule, std::string_view reason, std::uint64_t item = kNoItem) {
  return std::unexpected(Error{.rule = rule, .reason = reason, .item = item});
}

class Reader {
 public:
  explicit Reader(std::span<const std::byte> bytes) : bytes_(bytes) {}

  std::uint64_t offset() const { return pos_; }
  std::uint64_t remaining() const { return bytes_.size() - pos_; }

  // A little-endian unsigned integer of `size` bytes (at most 8).
  std::expected<std::uint64_t, Error> Unsigned(std::size_t size) {
    if (remaining() < size) {
      return Fail(Rule::kFormat, "GGUF metadata truncated", pos_);
    }
    std::uint64_t value = 0;
    for (std::size_t i = 0; i < size; ++i) {
      value |= static_cast<std::uint64_t>(std::to_integer<std::uint8_t>(bytes_[pos_ + i]))
               << (8U * i);
    }
    pos_ += size;
    return value;
  }

  // A string's bytes, its length checked against the bytes left.
  std::expected<std::string_view, Error> String() {
    auto length = Unsigned(8);
    if (!length) {
      return std::unexpected(length.error());
    }
    if (*length > remaining()) {
      return Fail(Rule::kFormat, "GGUF metadata truncated", pos_);
    }
    const std::string_view s(reinterpret_cast<const char*>(bytes_.data() + pos_),
                             static_cast<std::size_t>(*length));
    pos_ += *length;
    return s;
  }

  std::expected<void, Error> Skip(std::uint64_t size) {
    if (size > remaining()) {
      return Fail(Rule::kFormat, "GGUF metadata truncated", pos_);
    }
    pos_ += size;
    return {};
  }

 private:
  std::span<const std::byte> bytes_;
  std::uint64_t pos_ = 0;
};

// A scalar's bits as a signed integer, or refused when an unsigned 64-bit
// value does not fit.
std::expected<std::int64_t, Error> AsInteger(std::uint32_t type, std::uint64_t bits,
                                             std::uint64_t at) {
  switch (type) {
    case kI8:
      return static_cast<std::int8_t>(static_cast<std::uint8_t>(bits));
    case kI16:
      return static_cast<std::int16_t>(static_cast<std::uint16_t>(bits));
    case kI32:
      return static_cast<std::int32_t>(static_cast<std::uint32_t>(bits));
    case kI64:
      return std::bit_cast<std::int64_t>(bits);
    case kU64:
      if (bits > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
        return Fail(Rule::kBounds, "a GGUF u64 value beyond int64", at);
      }
      return static_cast<std::int64_t>(bits);
    default:  // u8, u16, u32, bool
      return static_cast<std::int64_t>(bits);
  }
}

double AsFloat(std::uint32_t type, std::uint64_t bits) {
  if (type == kF32) {
    return static_cast<double>(std::bit_cast<float>(static_cast<std::uint32_t>(bits)));
  }
  return std::bit_cast<double>(bits);
}

// One value of `type` (not an array's element: arrays are read by the
// caller). With `keep`, its value into `out`.
std::expected<void, Error> Value(Reader& r, std::uint32_t type, bool keep, GgufValue& out) {
  out.type = type;
  const std::uint64_t at = r.offset();
  if (const std::size_t size = ScalarBytes(type); size != 0) {
    auto bits = r.Unsigned(size);
    if (!bits) {
      return std::unexpected(bits.error());
    }
    if (!keep) {
      return {};
    }
    if (IsFloat(type)) {
      out.kind = GgufValue::Kind::kFloat;
      out.real = AsFloat(type, *bits);
      return {};
    }
    auto integer = AsInteger(type, *bits, at);
    if (!integer) {
      return std::unexpected(integer.error());
    }
    out.kind = GgufValue::Kind::kInteger;
    out.integer = *integer;
    return {};
  }
  if (type == kString) {
    auto s = r.String();
    if (!s) {
      return std::unexpected(s.error());
    }
    if (keep) {
      if (s->size() > kMaxGgufWantedString) {
        return Fail(Rule::kBounds, "a wanted GGUF string beyond the cap", at);
      }
      out.kind = GgufValue::Kind::kString;
      out.text.assign(*s);
    }
    return {};
  }
  if (type != kArray) {
    return Fail(Rule::kFormat, "an unknown GGUF value type", at);
  }
  auto element = r.Unsigned(4);
  auto count = element ? r.Unsigned(8) : element;
  if (!count) {
    return std::unexpected(count.error());
  }
  const auto e = static_cast<std::uint32_t>(*element);
  if (e == kArray || !KnownType(e)) {
    return Fail(Rule::kFormat, "a GGUF array of arrays or of an unknown type", at);
  }
  // Every element takes at least a byte (a string its 8-byte length), so
  // the bytes left bound the count before anything loops or allocates.
  const std::size_t size = ScalarBytes(e);
  const std::uint64_t least = size != 0 ? size : 8;
  if (*count > r.remaining() / least) {
    return Fail(Rule::kFormat, "GGUF metadata truncated", r.offset());
  }
  const bool numbers = keep && (IsInteger(e) || IsFloat(e));
  if (numbers && *count > kMaxGgufWantedArray) {
    return Fail(Rule::kBounds, "a wanted GGUF array beyond the cap", at);
  }
  if (!numbers) {
    if (size != 0) {
      return r.Skip(*count * size);
    }
    for (std::uint64_t i = 0; i < *count; ++i) {
      auto s = r.String();
      if (!s) {
        return std::unexpected(s.error());
      }
    }
    return {};
  }
  out.kind = IsFloat(e) ? GgufValue::Kind::kFloats : GgufValue::Kind::kIntegers;
  for (std::uint64_t i = 0; i < *count; ++i) {
    const std::uint64_t item = r.offset();
    auto bits = r.Unsigned(size);
    if (!bits) {
      return std::unexpected(bits.error());
    }
    if (IsFloat(e)) {
      out.reals.push_back(AsFloat(e, *bits));
      continue;
    }
    auto integer = AsInteger(e, *bits, item);
    if (!integer) {
      return std::unexpected(integer.error());
    }
    out.integers.push_back(*integer);
  }
  return {};
}

constexpr std::uint64_t kMaxArchitecture = 200;  // layout.py MAX_ARCH

}  // namespace

std::expected<GgufMetadata, Error> ReadGgufMetadata(std::span<const std::byte> bytes,
                                                    std::span<const std::string_view> wanted) {
  Reader r(bytes);
  auto magic = r.Unsigned(4);
  if (!magic || *magic != 0x46554747U) {  // "GGUF", little endian
    return Fail(Rule::kFormat, "not GGUF metadata");
  }
  auto version = r.Unsigned(4);
  auto tensors = version ? r.Unsigned(8) : version;
  auto keys = tensors ? r.Unsigned(8) : tensors;
  if (!keys) {
    return std::unexpected(keys.error());
  }
  if (*version != 3 || *tensors != 0) {
    return Fail(Rule::kFormat, "not a version 3, zero-tensor GGUF header");
  }
  if (*keys > kMaxGgufKeys) {
    return Fail(Rule::kBounds, "more GGUF keys than the cap");
  }
  GgufMetadata out;
  std::set<std::string_view> seen;
  for (std::uint64_t k = 0; k < *keys; ++k) {
    const std::uint64_t at = r.offset();
    auto key = r.String();
    if (!key) {
      return std::unexpected(key.error());
    }
    if (key->empty() || !seen.insert(*key).second) {
      return Fail(Rule::kSchema, "an empty or repeated GGUF key", at);
    }
    auto type = r.Unsigned(4);
    if (!type) {
      return std::unexpected(type.error());
    }
    const auto t = static_cast<std::uint32_t>(*type);
    if (!KnownType(t)) {
      return Fail(Rule::kFormat, "an unknown GGUF value type", at);
    }
    // The key rules, by type before any value is read.
    const bool alignment = *key == "general.alignment";
    const bool architecture = *key == "general.architecture";
    const bool expert_count = key->ends_with(".expert_count");
    if ((alignment || expert_count) && t != kU32) {
      return Fail(Rule::kSchema, "general.alignment and *.expert_count must be u32", at);
    }
    if (architecture && t != kString) {
      return Fail(Rule::kSchema, "general.architecture must be a string", at);
    }
    const bool keep = std::ranges::find(wanted, *key) != wanted.end();
    GgufValue value;
    if (auto read = Value(r, t, keep || alignment || architecture, value); !read) {
      return std::unexpected(read.error());
    }
    if (alignment &&
        (value.integer <= 0 || !std::has_single_bit(static_cast<std::uint64_t>(value.integer)))) {
      return Fail(Rule::kSchema, "general.alignment must be a power of two", at);
    }
    if (architecture && value.text.size() > kMaxArchitecture) {
      return Fail(Rule::kBounds, "general.architecture is longer than 200 bytes", at);
    }
    if (keep) {
      out.emplace(std::string(*key), std::move(value));
    }
  }
  if (r.remaining() != 0) {
    return Fail(Rule::kFormat, "bytes after the GGUF metadata", r.offset());
  }
  return out;
}

}  // namespace jitllm::artifact
