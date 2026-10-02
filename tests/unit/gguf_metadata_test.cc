// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// The kept GGUF metadata reader (artifact/gguf_metadata.h) on GGUF headers
// built here: the wanted keys' values of every type, and a refusal for each
// rule the prototype verifier's _check_kv_gguf applies (layout.py) and for
// each length and count that would run past the bytes.

#include "artifact/gguf_metadata.h"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "artifact/error.h"

namespace {

namespace ja = jitllm::artifact;

// A GGUF v3 header and key/value section, built value by value.
class Gguf {
 public:
  explicit Gguf(std::uint64_t keys, std::uint64_t tensors = 0, std::uint32_t version = 3) {
    Bytes("GGUF");
    U32(version);
    U64(tensors);
    U64(keys);
  }
  Gguf& Key(std::string_view key, std::uint32_t type) {
    Str(key);
    U32(type);
    return *this;
  }
  Gguf& U32(std::uint32_t v) { return Raw(&v, 4); }
  Gguf& U64(std::uint64_t v) { return Raw(&v, 8); }
  Gguf& I64(std::int64_t v) { return Raw(&v, 8); }
  Gguf& F32(float v) { return Raw(&v, 4); }
  Gguf& Str(std::string_view s) {
    U64(s.size());
    return Bytes(s);
  }
  Gguf& Bytes(std::string_view s) { return Raw(s.data(), s.size()); }
  Gguf& Raw(const void* p, std::size_t n) {
    const auto* b = static_cast<const std::byte*>(p);
    bytes_.insert(bytes_.end(), b, b + n);
    return *this;
  }
  std::span<const std::byte> span() const { return bytes_; }
  std::vector<std::byte>& bytes() { return bytes_; }

 private:
  std::vector<std::byte> bytes_;
};

constexpr std::uint32_t kU32 = 4;
constexpr std::uint32_t kI32 = 5;
constexpr std::uint32_t kF32 = 6;
constexpr std::uint32_t kString = 8;
constexpr std::uint32_t kArray = 9;
constexpr std::uint32_t kU64 = 10;
constexpr std::uint32_t kI64 = 11;

const std::array<std::string_view, 6> kWanted = {
    "general.architecture", "m.count", "m.eps", "m.list", "m.big", "m.floats"};

// Every kind of value: wanted scalars, arrays and a string, and unwanted
// ones (a long string, an array of strings) skipped.
Gguf Typical() {
  Gguf g(8);
  g.Key("general.architecture", kString).Str("qwen4exp");
  g.Key("general.alignment", kU32).U32(32);
  g.Key("m.count", kU32).U32(48);
  g.Key("m.eps", kF32).F32(1e-6f);
  g.Key("m.list", kArray).U32(kI64).U64(3).I64(-5).I64(23703573157769).I64(7);
  g.Key("m.big", kU64).U64(std::uint64_t{1} << 40U);
  g.Key("tokenizer.ggml.tokens", kArray).U32(kString).U64(2).Str("a").Str("bc");
  g.Key("tokenizer.chat_template", kString).Str(std::string(5000, 'x'));
  return g;
}

ja::Rule RuleOf(const std::expected<ja::GgufMetadata, ja::Error>& r) {
  EXPECT_FALSE(r.has_value());
  return r ? ja::Rule::kIo : r.error().rule;
}

TEST(GgufMetadataTest, ReadsTheWantedKeysAndSkipsTheRest) {
  const Gguf g = Typical();
  auto read = ja::ReadGgufMetadata(g.span(), kWanted);
  ASSERT_TRUE(read.has_value()) << read.error().ToString();
  using Kind = ja::GgufValue::Kind;
  EXPECT_EQ(read->size(), 5U);  // m.floats is absent
  EXPECT_EQ(read->at("general.architecture").kind, Kind::kString);
  EXPECT_EQ(read->at("general.architecture").text, "qwen4exp");
  EXPECT_EQ(read->at("m.count").kind, Kind::kInteger);
  EXPECT_EQ(read->at("m.count").integer, 48);
  EXPECT_EQ(read->at("m.eps").kind, Kind::kFloat);
  EXPECT_EQ(static_cast<float>(read->at("m.eps").real), 1e-6f);
  EXPECT_EQ(read->at("m.list").kind, Kind::kIntegers);
  EXPECT_EQ(read->at("m.list").integers, (std::vector<std::int64_t>{-5, 23703573157769, 7}));
  EXPECT_EQ(read->at("m.big").integer, std::int64_t{1} << 40U);
  EXPECT_FALSE(read->contains("tokenizer.chat_template"));
}

TEST(GgufMetadataTest, RefusesAnythingButACompleteZeroTensorV3Header) {
  EXPECT_EQ(RuleOf(ja::ReadGgufMetadata({}, kWanted)), ja::Rule::kFormat);
  Gguf bad_magic = Typical();
  bad_magic.bytes()[0] = std::byte{'X'};
  EXPECT_EQ(RuleOf(ja::ReadGgufMetadata(bad_magic.span(), kWanted)), ja::Rule::kFormat);
  Gguf v2(0, 0, 2);
  EXPECT_EQ(RuleOf(ja::ReadGgufMetadata(v2.span(), kWanted)), ja::Rule::kFormat);
  Gguf tensors(0, 1);
  EXPECT_EQ(RuleOf(ja::ReadGgufMetadata(tensors.span(), kWanted)), ja::Rule::kFormat);
  Gguf many(ja::kMaxGgufKeys + 1);
  EXPECT_EQ(RuleOf(ja::ReadGgufMetadata(many.span(), kWanted)), ja::Rule::kBounds);
  Gguf trailing = Typical();
  trailing.U32(0);
  EXPECT_EQ(RuleOf(ja::ReadGgufMetadata(trailing.span(), kWanted)), ja::Rule::kFormat);
  // Every prefix of a valid header is truncated, never read past.
  const Gguf g = Typical();
  for (std::size_t n = 0; n < g.span().size(); ++n) {
    auto r = ja::ReadGgufMetadata(g.span().first(n), kWanted);
    EXPECT_FALSE(r.has_value()) << n;
  }
}

TEST(GgufMetadataTest, AppliesGgufCppsKeyRules) {
  Gguf empty_key(1);
  empty_key.Key("", kU32).U32(1);
  EXPECT_EQ(RuleOf(ja::ReadGgufMetadata(empty_key.span(), kWanted)), ja::Rule::kSchema);
  Gguf repeated(2);
  repeated.Key("m.count", kU32).U32(1).Key("m.count", kU32).U32(2);
  EXPECT_EQ(RuleOf(ja::ReadGgufMetadata(repeated.span(), kWanted)), ja::Rule::kSchema);
  Gguf alignment(1);
  alignment.Key("general.alignment", kU32).U32(24);
  EXPECT_EQ(RuleOf(ja::ReadGgufMetadata(alignment.span(), kWanted)), ja::Rule::kSchema);
  Gguf alignment_type(1);
  alignment_type.Key("general.alignment", kU64).U64(32);
  EXPECT_EQ(RuleOf(ja::ReadGgufMetadata(alignment_type.span(), kWanted)), ja::Rule::kSchema);
  Gguf arch_type(1);
  arch_type.Key("general.architecture", kU32).U32(1);
  EXPECT_EQ(RuleOf(ja::ReadGgufMetadata(arch_type.span(), kWanted)), ja::Rule::kSchema);
  Gguf arch_long(1);
  arch_long.Key("general.architecture", kString).Str(std::string(201, 'a'));
  EXPECT_EQ(RuleOf(ja::ReadGgufMetadata(arch_long.span(), kWanted)), ja::Rule::kBounds);
  Gguf experts(1);
  experts.Key("qwen4exp.expert_count", kI32).U32(512);
  EXPECT_EQ(RuleOf(ja::ReadGgufMetadata(experts.span(), kWanted)), ja::Rule::kSchema);
  Gguf unknown(1);
  unknown.Key("m.count", 13).U32(1);
  EXPECT_EQ(RuleOf(ja::ReadGgufMetadata(unknown.span(), kWanted)), ja::Rule::kFormat);
}

TEST(GgufMetadataTest, BoundsArraysAndValues) {
  // No array of arrays, nor of an unknown type, even empty.
  Gguf nested(1);
  nested.Key("m.list", kArray).U32(kArray).U64(0);
  EXPECT_EQ(RuleOf(ja::ReadGgufMetadata(nested.span(), kWanted)), ja::Rule::kFormat);
  Gguf unknown(1);
  unknown.Key("m.other", kArray).U32(99).U64(0);
  EXPECT_EQ(RuleOf(ja::ReadGgufMetadata(unknown.span(), kWanted)), ja::Rule::kFormat);
  // A count past the bytes is refused before anything is read, wanted or
  // not.
  for (const std::string_view key : {"m.list", "m.other"}) {
    Gguf huge(1);
    huge.Key(key, kArray).U32(kI64).U64(std::numeric_limits<std::uint64_t>::max() / 8).I64(1);
    EXPECT_EQ(RuleOf(ja::ReadGgufMetadata(huge.span(), kWanted)), ja::Rule::kFormat) << key;
    Gguf strings(1);
    strings.Key(key, kArray).U32(kString).U64(std::uint64_t{1} << 60U);
    EXPECT_EQ(RuleOf(ja::ReadGgufMetadata(strings.span(), kWanted)), ja::Rule::kFormat) << key;
  }
  Gguf long_string(1);
  long_string.Key("m.other", kString).U64(std::uint64_t{1} << 62U);
  EXPECT_EQ(RuleOf(ja::ReadGgufMetadata(long_string.span(), kWanted)), ja::Rule::kFormat);
  // An unsigned 64-bit value beyond int64 is refused, not wrapped, where it
  // is wanted; skipped where it is not.
  Gguf big(1);
  big.Key("m.big", kU64).U64(std::numeric_limits<std::uint64_t>::max());
  EXPECT_EQ(RuleOf(ja::ReadGgufMetadata(big.span(), kWanted)), ja::Rule::kBounds);
  Gguf big_list(1);
  big_list.Key("m.list", kArray).U32(kU64).U64(1).U64(std::numeric_limits<std::uint64_t>::max());
  EXPECT_EQ(RuleOf(ja::ReadGgufMetadata(big_list.span(), kWanted)), ja::Rule::kBounds);
  Gguf skipped(1);
  skipped.Key("m.unwanted", kU64).U64(std::numeric_limits<std::uint64_t>::max());
  EXPECT_TRUE(ja::ReadGgufMetadata(skipped.span(), kWanted).has_value());
  // F32 arrays are read as floats.
  Gguf floats(1);
  floats.Key("m.floats", kArray).U32(kF32).U64(2).F32(0.5f).F32(-2.0f);
  auto read = ja::ReadGgufMetadata(floats.span(), kWanted);
  ASSERT_TRUE(read.has_value()) << read.error().ToString();
  EXPECT_EQ(read->at("m.floats").kind, ja::GgufValue::Kind::kFloats);
  EXPECT_EQ(read->at("m.floats").reals, (std::vector<double>{0.5, -2.0}));
}

}  // namespace
