// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// General GGUF: a file's header (its version, metadata key/values and
// tensor infos) parsed from untrusted bytes without exceptions, for the
// readers that interpret it: the importer's container reader
// (import/readers/gguf.h), the artifact's kept metadata
// (artifact/gguf_metadata.h) and the tokenizer (tokenizer/gguf.h).
//
// Accepted: GGUF versions 2 and 3 in little-endian byte order (llama.cpp's
// gguf.cpp at the pin reads both; version 1 and byte-swapped files are
// refused), every value type of the format, and arrays of any type but
// arrays (an array of arrays is invalid in gguf.cpp, which every GGUF must
// load in). Keys and tensor names are non-empty, well-formed UTF-8 and
// unique; a tensor info has at most four dimensions, each a non-negative
// 64-bit signed count whose product stays below 2^63 (gguf.cpp's rules).
// Bools are 0 or 1. What a type, an offset or an alignment means is the
// caller's to judge; Parse checks only the container's structure.
//
// Every count and length is checked against its limit (kBounds), then
// against the bytes left (kTruncated), before anything loops on it or
// allocates: a key/value takes at least 14 bytes, a tensor info 25 plus 8
// a dimension, an array element at least its scalar size (a string its
// 8-byte length). So a small hostile header cannot make the parser build a
// large table, and a caller reading growing prefixes of a file reads more
// only on kTruncated. The header holds views into the bytes, which must
// outlive it; Parse copies nothing.

#ifndef LLMP_BASE_GGUF_H_
#define LLMP_BASE_GGUF_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace llmp::base::gguf {

inline constexpr std::uint32_t kMagic = 0x46554747U;    // "GGUF", little endian
inline constexpr std::size_t kMaxDims = 4;              // GGML_MAX_DIMS
inline constexpr std::uint32_t kDefaultAlignment = 32;  // GGUF_DEFAULT_ALIGNMENT

// GGUF value types (gguf.h).
enum class Type : std::uint32_t {
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

// The bytes of a scalar of `type`; 0 for a string, an array or a value
// outside the enum.
std::size_t ScalarBytes(Type type);
bool IsInteger(Type type);             // the integer types, not bool
bool IsFloat(Type type);               // F32 and F64
std::string_view TypeName(Type type);  // "u32", "string", "array"; "?" outside the enum

enum class Fault : std::uint8_t {
  kTruncated,  // the bytes end inside the header
  kBounds,     // a count or length beyond the limits
  kMalformed,  // magic, version, an unknown or nested type, a bool not 0 or 1, a bad dimension
  kName,       // an empty key or tensor name, ill-formed UTF-8, or longer than its limit
  kDuplicate,  // a key or tensor name twice
};

struct Error {
  Fault fault = Fault::kMalformed;
  std::string_view reason{};  // static text; nothing from the file
  std::uint64_t offset = 0;   // the byte where the fault was found

  // "gguf: <reason> (byte <offset>)".
  std::string ToString() const;
};

struct Limits {
  std::uint64_t max_keys = std::uint64_t{1} << 16U;
  std::uint64_t max_tensors = std::uint64_t{1} << 20U;
  std::uint64_t max_key_bytes = std::uint64_t{1} << 16U;
  // GGML_MAX_NAME - 1: gguf.cpp refuses longer names.
  std::uint64_t max_name_bytes = 63;
  // Any string value or array element, and any array's element count.
  std::uint64_t max_string_bytes = UINT64_MAX;
  std::uint64_t max_array_elements = UINT64_MAX;
};

// A metadata key/value: views into the parsed bytes.
struct KeyValue {
  std::string_view key{};
  Type type = Type::kU8;
  // Arrays: the element type and count; scalars and strings: kU8 and 1.
  Type element = Type::kU8;
  std::uint64_t count = 1;
  // The value's bytes after its type: a scalar's own, a string's
  // contents, an array's elements (after its element type and count).
  std::span<const std::byte> payload{};
  std::uint64_t offset = 0;  // of the key's length
};

struct TensorInfo {
  std::string_view name{};
  std::uint32_t dims = 0;
  std::array<std::uint64_t, kMaxDims> ne{1, 1, 1, 1};  // innermost first; 1 past `dims`
  std::uint32_t type = 0;                              // ggml_type's value, not judged here
  std::uint64_t offset = 0;                            // into the data section, not judged here
  std::uint64_t at = 0;                                // where its info starts in the file
};

struct Header {
  std::uint32_t version = 0;
  std::vector<KeyValue> metadata{};   // in file order
  std::vector<TensorInfo> tensors{};  // in file order
  std::uint64_t tensor_count = 0;     // as the header declares it
  std::uint64_t metadata_end = 0;     // the byte after the last key/value
  std::uint64_t end = 0;              // the byte after the last tensor info

  // The key's value, if the header has it.
  const KeyValue* Find(std::string_view key) const;
};

std::expected<Header, Error> Parse(std::span<const std::byte> bytes, const Limits& limits = {});
// The header through its metadata only, from a prefix of a file that
// holds at least that: no tensor info is read (tensor_count is still
// bounded by the limits), and `end` is metadata_end.
std::expected<Header, Error> ParseMetadata(std::span<const std::byte> bytes,
                                           const Limits& limits = {});

// A scalar's little-endian bits (kv.type a scalar type).
std::uint64_t ScalarBits(const KeyValue& kv);
// Element i of a numeric or bool array (i < kv.count).
std::uint64_t ElementBits(const KeyValue& kv, std::uint64_t i);
// A string value's bytes.
std::string_view String(const KeyValue& kv);
// Calls fn with each element of an array of strings, in order.
void ForEachString(const KeyValue& kv, const std::function<void(std::string_view)>& fn);

// A scalar integer of `type` (an integer type or bool) from its bits as a
// signed 64-bit value; none for a u64 above INT64_MAX.
std::optional<std::int64_t> SignedValue(Type type, std::uint64_t bits);
// A float of `type` (F32 or F64) from its bits.
double FloatValue(Type type, std::uint64_t bits);

}  // namespace llmp::base::gguf

#endif  // LLMP_BASE_GGUF_H_
