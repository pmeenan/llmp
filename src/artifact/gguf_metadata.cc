// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "artifact/gguf_metadata.h"

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>

#include "artifact/error.h"
#include "base/gguf.h"

namespace llmp::artifact {
namespace {

namespace gg = base::gguf;

std::unexpected<Error> Fail(Rule rule, std::string_view reason, std::uint64_t item = kNoItem) {
  return std::unexpected(Error{.rule = rule, .reason = reason, .item = item});
}

std::uint64_t LittleEndian(std::span<const std::byte> bytes) {
  std::uint64_t value = 0;
  for (std::size_t i = 0; i < bytes.size(); ++i) {
    value |= static_cast<std::uint64_t>(std::to_integer<std::uint8_t>(bytes[i])) << (8U * i);
  }
  return value;
}

// The parser's fault as this reader's rule: the container's structure is
// kFormat, a key's emptiness or repetition kSchema, a cap kBounds.
Error FromHeader(const gg::Error& e) {
  Rule rule = Rule::kFormat;
  if (e.fault == gg::Fault::kName || e.fault == gg::Fault::kDuplicate) {
    rule = Rule::kSchema;
  } else if (e.fault == gg::Fault::kBounds) {
    rule = Rule::kBounds;
  }
  return {.rule = rule, .reason = e.reason, .item = e.offset};
}

bool Numeric(gg::Type type) { return gg::IsInteger(type) || type == gg::Type::kBool; }

// A wanted key's value.
std::expected<GgufValue, Error> ValueOf(const gg::KeyValue& kv) {
  GgufValue out;
  out.type = static_cast<std::uint32_t>(kv.type);
  if (kv.type == gg::Type::kString) {
    if (kv.payload.size() > kMaxGgufWantedString) {
      return Fail(Rule::kBounds, "a wanted GGUF string beyond the cap", kv.offset);
    }
    out.kind = GgufValue::Kind::kString;
    out.text.assign(gg::String(kv));
    return out;
  }
  if (gg::IsFloat(kv.type)) {
    out.kind = GgufValue::Kind::kFloat;
    out.real = gg::FloatValue(kv.type, gg::ScalarBits(kv));
    return out;
  }
  if (kv.type != gg::Type::kArray) {
    const auto v = gg::SignedValue(kv.type, gg::ScalarBits(kv));
    if (!v) {
      return Fail(Rule::kBounds, "a GGUF u64 value beyond int64", kv.offset);
    }
    out.kind = GgufValue::Kind::kInteger;
    out.integer = *v;
    return out;
  }
  const bool floats = gg::IsFloat(kv.element);
  if (!floats && !Numeric(kv.element)) {
    return out;  // an array of strings: kOther
  }
  if (kv.count > kMaxGgufWantedArray) {
    return Fail(Rule::kBounds, "a wanted GGUF array beyond the cap", kv.offset);
  }
  out.kind = floats ? GgufValue::Kind::kFloats : GgufValue::Kind::kIntegers;
  for (std::uint64_t i = 0; i < kv.count; ++i) {
    const std::uint64_t bits = gg::ElementBits(kv, i);
    if (floats) {
      out.reals.push_back(gg::FloatValue(kv.element, bits));
      continue;
    }
    const auto v = gg::SignedValue(kv.element, bits);
    if (!v) {
      return Fail(Rule::kBounds, "a GGUF u64 value beyond int64", kv.offset);
    }
    out.integers.push_back(*v);
  }
  return out;
}

constexpr std::uint64_t kMaxArchitecture = 200;  // layout.py MAX_ARCH

}  // namespace

std::expected<GgufMetadata, Error> ReadGgufMetadata(std::span<const std::byte> bytes,
                                                    std::span<const std::string_view> wanted) {
  // The fixed header first, for this reader's own refusals: only a
  // version 3 header with no tensors is kept metadata.
  if (bytes.size() < 4 || LittleEndian(bytes.first(4)) != gg::kMagic) {
    return Fail(Rule::kFormat, "not GGUF metadata");
  }
  if (bytes.size() < 24) {
    return Fail(Rule::kFormat, "GGUF metadata truncated", bytes.size());
  }
  if (LittleEndian(bytes.subspan(4, 4)) != 3 || LittleEndian(bytes.subspan(8, 8)) != 0) {
    return Fail(Rule::kFormat, "not a version 3, zero-tensor GGUF header");
  }
  if (LittleEndian(bytes.subspan(16, 8)) > kMaxGgufKeys) {
    return Fail(Rule::kBounds, "more GGUF keys than the cap");
  }
  // Strings and arrays nothing reads need no cap beyond the bytes they
  // span; the wanted ones are capped below.
  constexpr auto kAny = std::numeric_limits<std::uint64_t>::max();
  const auto header = gg::Parse(bytes, {.max_keys = kMaxGgufKeys,
                                        .max_tensors = 0,
                                        .max_key_bytes = kAny,
                                        .max_string_bytes = kAny,
                                        .max_array_elements = kAny});
  if (!header) {
    return std::unexpected(FromHeader(header.error()));
  }
  if (header->end != bytes.size()) {
    return Fail(Rule::kFormat, "bytes after the GGUF metadata", header->end);
  }
  GgufMetadata out;
  for (const gg::KeyValue& kv : header->metadata) {
    // The key rules (the prototype verifier's _check_kv_gguf).
    const bool alignment = kv.key == "general.alignment";
    const bool architecture = kv.key == "general.architecture";
    if ((alignment || kv.key.ends_with(".expert_count")) && kv.type != gg::Type::kU32) {
      return Fail(Rule::kSchema, "general.alignment and *.expert_count must be u32", kv.offset);
    }
    if (architecture && kv.type != gg::Type::kString) {
      return Fail(Rule::kSchema, "general.architecture must be a string", kv.offset);
    }
    if (alignment && !std::has_single_bit(gg::ScalarBits(kv))) {
      return Fail(Rule::kSchema, "general.alignment must be a power of two", kv.offset);
    }
    if (architecture && kv.payload.size() > kMaxArchitecture) {
      return Fail(Rule::kBounds, "general.architecture is longer than 200 bytes", kv.offset);
    }
    if (std::ranges::find(wanted, kv.key) == wanted.end()) {
      continue;
    }
    auto value = ValueOf(kv);
    if (!value) {
      return std::unexpected(value.error());
    }
    out.emplace(std::string(kv.key), std::move(*value));
  }
  return out;
}

}  // namespace llmp::artifact
