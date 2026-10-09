// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "base/gguf.h"

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <functional>
#include <limits>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <string_view>

#include "base/json.h"

namespace llmp::base::gguf {
namespace {

// The least bytes a key/value takes (a key's 8-byte length and one byte,
// its type, a one-byte value), and a tensor info besides its dimensions
// (a name's length and one byte, its dimension count, type and offset).
constexpr std::uint64_t kLeastKeyValue = 8 + 1 + 4 + 1;
constexpr std::uint64_t kLeastTensorInfo = 8 + 1 + 4 + 4 + 8;

std::unexpected<Error> Fail(Fault fault, std::string_view reason, std::uint64_t offset) {
  return std::unexpected(Error{.fault = fault, .reason = reason, .offset = offset});
}

bool Known(std::uint32_t type) { return type <= static_cast<std::uint32_t>(Type::kF64); }

std::uint64_t LittleEndian(std::span<const std::byte> bytes) {
  std::uint64_t value = 0;
  for (std::size_t i = 0; i < bytes.size(); ++i) {
    value |= static_cast<std::uint64_t>(std::to_integer<std::uint8_t>(bytes[i])) << (8U * i);
  }
  return value;
}

class Reader {
 public:
  Reader(std::span<const std::byte> bytes, const Limits& limits) : bytes_(bytes), limits_(limits) {}

  std::uint64_t offset() const { return pos_; }
  std::uint64_t remaining() const { return bytes_.size() - pos_; }

  std::expected<std::uint64_t, Error> Unsigned(std::size_t size) {
    if (remaining() < size) {
      return Fail(Fault::kTruncated, "the header is truncated", pos_);
    }
    const std::uint64_t value = LittleEndian(bytes_.subspan(pos_, size));
    pos_ += size;
    return value;
  }

  std::expected<std::span<const std::byte>, Error> Take(std::uint64_t size) {
    if (remaining() < size) {
      return Fail(Fault::kTruncated, "the header is truncated", pos_);
    }
    const auto taken = bytes_.subspan(pos_, static_cast<std::size_t>(size));
    pos_ += size;
    return taken;
  }

  // A string's bytes: its length checked against `limit`, then against
  // the bytes left (a length past the limit is refused whatever follows).
  std::expected<std::string_view, Error> String(std::uint64_t limit) {
    const std::uint64_t at = pos_;
    auto length = Unsigned(8);
    if (!length) {
      return std::unexpected(length.error());
    }
    if (*length > limit) {
      return Fail(Fault::kBounds, "a string longer than the limit", at);
    }
    if (*length > remaining()) {
      return Fail(Fault::kTruncated, "the header is truncated", at);
    }
    auto bytes = Take(*length);
    if (!bytes) {
      return std::unexpected(bytes.error());
    }
    return std::string_view(reinterpret_cast<const char*>(bytes->data()), bytes->size());
  }

  // A key or tensor name: non-empty, within `limit`, well-formed UTF-8.
  std::expected<std::string_view, Error> Name(std::uint64_t limit) {
    const std::uint64_t at = pos_;
    auto name = String(limit);
    if (!name) {
      if (name.error().fault == Fault::kBounds) {
        return Fail(Fault::kName, "a key or tensor name longer than the limit", at);
      }
      return name;
    }
    if (name->empty()) {
      return Fail(Fault::kName, "an empty key or tensor name", at);
    }
    if (json::FirstInvalidUtf8(*name)) {
      return Fail(Fault::kName, "a key or tensor name that is not UTF-8", at);
    }
    return name;
  }

  // A value of `type` after its key and type: its payload, and for an
  // array its element type and count, into kv.
  std::expected<void, Error> Value(KeyValue& kv) {
    const std::uint64_t at = pos_;
    if (const std::size_t size = ScalarBytes(kv.type); size != 0) {
      auto payload = Take(size);
      if (!payload) {
        return std::unexpected(payload.error());
      }
      if (kv.type == Type::kBool && LittleEndian(*payload) > 1) {
        return Fail(Fault::kMalformed, "a bool that is not 0 or 1", at);
      }
      kv.payload = *payload;
      return {};
    }
    if (kv.type == Type::kString) {
      auto s = String(limits_.max_string_bytes);
      if (!s) {
        return std::unexpected(s.error());
      }
      kv.payload = std::as_bytes(std::span(s->data(), s->size()));
      return {};
    }
    // An array (Parse admits only known types).
    auto element = Unsigned(4);
    auto count = element ? Unsigned(8) : element;
    if (!count) {
      return std::unexpected(count.error());
    }
    if (!Known(static_cast<std::uint32_t>(*element)) ||
        static_cast<Type>(*element) == Type::kArray) {
      return Fail(Fault::kMalformed, "an array of arrays or of an unknown type", at);
    }
    kv.element = static_cast<Type>(*element);
    // The limit, then the bytes left: every element takes at least a byte
    // (a string its 8-byte length), so they bound the count before
    // anything loops on it.
    if (*count > limits_.max_array_elements) {
      return Fail(Fault::kBounds, "an array longer than the limit", at);
    }
    const std::size_t size = ScalarBytes(kv.element);
    const std::uint64_t least = size != 0 ? size : 8;
    if (*count > remaining() / least) {
      return Fail(Fault::kTruncated, "the header is truncated", at);
    }
    kv.count = *count;
    const std::uint64_t start = pos_;
    if (size != 0) {
      auto payload = Take(*count * size);
      if (!payload) {
        return std::unexpected(payload.error());
      }
      if (kv.element == Type::kBool &&
          std::ranges::any_of(*payload, [](std::byte b) { return std::to_integer<int>(b) > 1; })) {
        return Fail(Fault::kMalformed, "a bool that is not 0 or 1", at);
      }
      kv.payload = *payload;
      return {};
    }
    for (std::uint64_t i = 0; i < *count; ++i) {
      if (auto s = String(limits_.max_string_bytes); !s) {
        return std::unexpected(s.error());
      }
    }
    kv.payload = bytes_.subspan(start, static_cast<std::size_t>(pos_ - start));
    return {};
  }

 private:
  std::span<const std::byte> bytes_;
  const Limits& limits_;
  std::uint64_t pos_ = 0;
};

}  // namespace

std::size_t ScalarBytes(Type type) {
  switch (type) {
    case Type::kU8:
    case Type::kI8:
    case Type::kBool:
      return 1;
    case Type::kU16:
    case Type::kI16:
      return 2;
    case Type::kU32:
    case Type::kI32:
    case Type::kF32:
      return 4;
    case Type::kU64:
    case Type::kI64:
    case Type::kF64:
      return 8;
    case Type::kString:
    case Type::kArray:
      return 0;
  }
  return 0;
}

bool IsInteger(Type type) {
  return type != Type::kBool && !IsFloat(type) && ScalarBytes(type) != 0;
}

bool IsFloat(Type type) { return type == Type::kF32 || type == Type::kF64; }

std::string_view TypeName(Type type) {
  switch (type) {
    case Type::kU8:
      return "u8";
    case Type::kI8:
      return "i8";
    case Type::kU16:
      return "u16";
    case Type::kI16:
      return "i16";
    case Type::kU32:
      return "u32";
    case Type::kI32:
      return "i32";
    case Type::kF32:
      return "f32";
    case Type::kBool:
      return "bool";
    case Type::kString:
      return "string";
    case Type::kArray:
      return "array";
    case Type::kU64:
      return "u64";
    case Type::kI64:
      return "i64";
    case Type::kF64:
      return "f64";
  }
  return "?";
}

std::string Error::ToString() const { return std::format("gguf: {} (byte {})", reason, offset); }

const KeyValue* Header::Find(std::string_view key) const {
  const auto found = std::ranges::find(metadata, key, &KeyValue::key);
  return found == metadata.end() ? nullptr : &*found;
}

namespace {

std::expected<Header, Error> ParseHeader(std::span<const std::byte> bytes, const Limits& limits,
                                         bool with_tensors) {
  Reader r(bytes, limits);
  auto magic = r.Unsigned(4);
  if (!magic) {
    return std::unexpected(magic.error());
  }
  if (*magic != kMagic) {
    return Fail(Fault::kMalformed, "not a GGUF file", 0);
  }
  auto version = r.Unsigned(4);
  if (!version) {
    return std::unexpected(version.error());
  }
  // A byte-swapped file's version reads as a multiple of 2^16.
  if (*version != 2 && *version != 3) {
    return Fail(Fault::kMalformed, "a GGUF version other than 2 or 3", 4);
  }
  auto tensors = r.Unsigned(8);
  auto keys = tensors ? r.Unsigned(8) : tensors;
  if (!keys) {
    return std::unexpected(keys.error());
  }
  if (*keys > limits.max_keys) {
    return Fail(Fault::kBounds, "more keys than the limit", 16);
  }
  if (*tensors > limits.max_tensors) {
    return Fail(Fault::kBounds, "more tensors than the limit", 8);
  }

  Header header;
  header.version = static_cast<std::uint32_t>(*version);
  header.metadata.reserve(
      static_cast<std::size_t>(std::min(*keys, r.remaining() / kLeastKeyValue)));
  std::set<std::string_view> seen;
  for (std::uint64_t k = 0; k < *keys; ++k) {
    KeyValue kv{.offset = r.offset()};
    auto key = r.Name(limits.max_key_bytes);
    if (!key) {
      return std::unexpected(key.error());
    }
    if (!seen.insert(*key).second) {
      return Fail(Fault::kDuplicate, "a key twice", kv.offset);
    }
    kv.key = *key;
    const std::uint64_t at = r.offset();
    auto type = r.Unsigned(4);
    if (!type) {
      return std::unexpected(type.error());
    }
    if (!Known(static_cast<std::uint32_t>(*type))) {
      return Fail(Fault::kMalformed, "an unknown value type", at);
    }
    kv.type = static_cast<Type>(*type);
    if (auto v = r.Value(kv); !v) {
      return std::unexpected(v.error());
    }
    header.metadata.push_back(kv);
  }
  header.metadata_end = r.offset();
  header.tensor_count = *tensors;
  if (!with_tensors) {
    header.end = header.metadata_end;
    return header;
  }

  header.tensors.reserve(
      static_cast<std::size_t>(std::min(*tensors, r.remaining() / kLeastTensorInfo)));
  std::set<std::string_view> names;
  for (std::uint64_t t = 0; t < *tensors; ++t) {
    TensorInfo info{.at = r.offset()};
    auto name = r.Name(limits.max_name_bytes);
    if (!name) {
      return std::unexpected(name.error());
    }
    if (!names.insert(*name).second) {
      return Fail(Fault::kDuplicate, "a tensor name twice", info.at);
    }
    info.name = *name;
    const std::uint64_t dims_at = r.offset();
    auto dims = r.Unsigned(4);
    if (!dims) {
      return std::unexpected(dims.error());
    }
    if (*dims > kMaxDims) {
      return Fail(Fault::kMalformed, "a tensor of more than four dimensions", dims_at);
    }
    info.dims = static_cast<std::uint32_t>(*dims);
    for (std::uint32_t d = 0; d < info.dims; ++d) {
      const std::uint64_t at = r.offset();
      auto ne = r.Unsigned(8);
      if (!ne) {
        return std::unexpected(ne.error());
      }
      if (*ne > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
        return Fail(Fault::kMalformed, "a negative tensor dimension", at);
      }
      info.ne.at(d) = *ne;
    }
    // gguf.cpp's overflow rule: the element count fits in int64.
    std::uint64_t elements = 1;
    for (const std::uint64_t ne : info.ne) {
      if (ne != 0 &&
          elements > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) / ne) {
        return Fail(Fault::kMalformed, "a tensor of more than 2^63 elements", dims_at);
      }
      elements *= ne;
    }
    auto type = r.Unsigned(4);
    auto offset = type ? r.Unsigned(8) : type;
    if (!offset) {
      return std::unexpected(offset.error());
    }
    info.type = static_cast<std::uint32_t>(*type);
    info.offset = *offset;
    header.tensors.push_back(info);
  }
  header.end = r.offset();
  return header;
}

}  // namespace

std::expected<Header, Error> Parse(std::span<const std::byte> bytes, const Limits& limits) {
  return ParseHeader(bytes, limits, true);
}

std::expected<Header, Error> ParseMetadata(std::span<const std::byte> bytes, const Limits& limits) {
  return ParseHeader(bytes, limits, false);
}

std::uint64_t ScalarBits(const KeyValue& kv) { return LittleEndian(kv.payload); }

std::uint64_t ElementBits(const KeyValue& kv, std::uint64_t i) {
  const std::size_t size = ScalarBytes(kv.element);
  return LittleEndian(kv.payload.subspan(static_cast<std::size_t>(i) * size, size));
}

std::string_view String(const KeyValue& kv) {
  return {reinterpret_cast<const char*>(kv.payload.data()), kv.payload.size()};
}

void ForEachString(const KeyValue& kv, const std::function<void(std::string_view)>& fn) {
  // Parse checked every length against the payload, which is exactly the
  // elements' bytes.
  std::size_t pos = 0;
  for (std::uint64_t i = 0; i < kv.count && kv.payload.size() - pos >= 8; ++i) {
    const std::uint64_t length = LittleEndian(kv.payload.subspan(pos, 8));
    pos += 8;
    const std::size_t take =
        static_cast<std::size_t>(std::min<std::uint64_t>(length, kv.payload.size() - pos));
    fn({reinterpret_cast<const char*>(kv.payload.data() + pos), take});
    pos += take;
  }
}

std::optional<std::int64_t> SignedValue(Type type, std::uint64_t bits) {
  switch (type) {
    case Type::kI8:
      return static_cast<std::int8_t>(static_cast<std::uint8_t>(bits));
    case Type::kI16:
      return static_cast<std::int16_t>(static_cast<std::uint16_t>(bits));
    case Type::kI32:
      return static_cast<std::int32_t>(static_cast<std::uint32_t>(bits));
    case Type::kI64:
      return std::bit_cast<std::int64_t>(bits);
    case Type::kU64:
      if (bits > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
        return std::nullopt;
      }
      return static_cast<std::int64_t>(bits);
    default:  // u8, u16, u32, bool
      return static_cast<std::int64_t>(bits);
  }
}

double FloatValue(Type type, std::uint64_t bits) {
  if (type == Type::kF32) {
    return static_cast<double>(std::bit_cast<float>(static_cast<std::uint32_t>(bits)));
  }
  return std::bit_cast<double>(bits);
}

}  // namespace llmp::base::gguf
