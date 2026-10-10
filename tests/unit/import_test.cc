// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The general GGUF header parser (base/gguf.h) and the importer's container
// readers (import/) on checkpoints built here: every value type, the
// tensor formats, extents and metadata each reader gives, the mapping
// source a Gemma 2 checkpoint makes, and a refusal for each rule: every
// truncation, counts and lengths past the bytes or the caps, offsets that
// overlap, leave gaps, run past the file or break the alignment, unknown
// types and dtypes, duplicate and ill-formed names, nesting, split and
// shard sets that disagree, the quantizations, and paths others could
// change.

#include <fcntl.h>
#include <gtest/gtest.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <expected>
#include <filesystem>
#include <format>
#include <fstream>
#include <initializer_list>
#include <limits>
#include <map>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <variant>
#include <vector>

#include "base/gguf.h"
#include "expected_error.h"
#include "format/encoding.h"
#include "format/layout.h"
#include "format/tensor_format.h"
#include "import/checkpoint.h"
#include "import/error.h"
#include "import/files.h"
#include "import/reader.h"
#include "import/readers/gguf.h"
#include "import/readers/hf_quant.h"
#include "import/readers/safetensors.h"
#include "import_fixtures.h"
#include "spec/mapping.h"
#include "spec/spec.h"

namespace llmp::import {
namespace {

namespace gg = base::gguf;
using format::Encoding;
using format::Layout;
using format::PartRole;
using test_support::Failed;
using namespace fixtures;  // NOLINT(google-build-using-namespace): the tests' builders

// GGML type numbers (ggml.h at the pin).
constexpr std::uint32_t kF32 = 0;
constexpr std::uint32_t kF16 = 1;
constexpr std::uint32_t kQ8_0 = 8;
constexpr std::uint32_t kQ4_K = 12;
constexpr std::uint32_t kBF16 = 30;
constexpr std::uint32_t kNvfp4 = 40;

std::vector<std::byte> StringBytes(std::string_view s) {
  std::vector<std::byte> out;
  Append(out, s.size(), 8);
  AppendText(out, s);
  return out;
}

// A tensor's bytes by the size rule (GGML's nbytes).
std::uint64_t GgmlBytes(std::uint32_t type, const std::vector<std::uint64_t>& ne) {
  const auto shape = format::ShapeOf(ne);
  for (const format::EncodingInfo& info : format::Encodings()) {
    if (shape && info.ggml && info.ggml->type == type) {
      format::TensorFormat f{.encoding = info.id, .shape = *shape};
      if (info.quantized) {
        f.layout = info.id == Encoding::kNvfp4 ? Layout::kGgmlNvfp4Blocks : Layout::kGgmlBlocks;
      }
      const auto parts = format::PartsOf(f);
      return parts ? (*parts)[0].bytes : 0;
    }
  }
  return 0;
}

// A GGUF file built key by key and tensor by tensor; its tensors' data
// packed as gguf.cpp writes it, each padded to the alignment.
class Gguf {
 public:
  Gguf() { Str("general.architecture", "test"); }

  // A key/value of any type number, the format's or not.
  Gguf& KvRaw(std::string_view key, std::uint32_t type, const std::vector<std::byte>& payload) {
    std::vector<std::byte> kv = StringBytes(key);
    Append(kv, type, 4);
    kv.insert(kv.end(), payload.begin(), payload.end());
    kvs_.push_back(std::move(kv));
    return *this;
  }
  Gguf& Kv(std::string_view key, gg::Type type, const std::vector<std::byte>& payload) {
    return KvRaw(key, static_cast<std::uint32_t>(type), payload);
  }
  Gguf& U8(std::string_view key, std::uint8_t v) { return Kv(key, gg::Type::kU8, Le(v, 1)); }
  Gguf& U16(std::string_view key, std::uint16_t v) { return Kv(key, gg::Type::kU16, Le(v, 2)); }
  Gguf& U32(std::string_view key, std::uint32_t v) { return Kv(key, gg::Type::kU32, Le(v, 4)); }
  Gguf& I32(std::string_view key, std::int32_t v) {
    return Kv(key, gg::Type::kI32, Le(static_cast<std::uint32_t>(v), 4));
  }
  Gguf& U64(std::string_view key, std::uint64_t v) { return Kv(key, gg::Type::kU64, Le(v, 8)); }
  Gguf& F32(std::string_view key, float v) {
    return Kv(key, gg::Type::kF32, Le(std::bit_cast<std::uint32_t>(v), 4));
  }
  Gguf& Bool(std::string_view key, std::uint8_t v) { return Kv(key, gg::Type::kBool, Le(v, 1)); }
  Gguf& Str(std::string_view key, std::string_view v) {
    return Kv(key, gg::Type::kString, StringBytes(v));
  }
  Gguf& ArrayRaw(std::string_view key, std::uint32_t element, std::uint64_t count,
                 const std::vector<std::byte>& elements) {
    std::vector<std::byte> payload = Le(element, 4);
    Append(payload, count, 8);
    payload.insert(payload.end(), elements.begin(), elements.end());
    return Kv(key, gg::Type::kArray, payload);
  }
  Gguf& Array(std::string_view key, gg::Type element, std::uint64_t count,
              const std::vector<std::byte>& elements) {
    return ArrayRaw(key, static_cast<std::uint32_t>(element), count, elements);
  }
  Gguf& Strings(std::string_view key, const std::vector<std::string>& values) {
    std::vector<std::byte> elements;
    for (const std::string& v : values) {
      const auto s = StringBytes(v);
      elements.insert(elements.end(), s.begin(), s.end());
    }
    return Array(key, gg::Type::kString, values.size(), elements);
  }
  Gguf& Remove(std::string_view key) {
    std::erase_if(kvs_, [&](const std::vector<std::byte>& kv) {
      const auto s = StringBytes(key);
      return kv.size() >= s.size() && std::equal(s.begin(), s.end(), kv.begin());
    });
    return *this;
  }

  Gguf& Tensor(std::string name, std::uint32_t type, std::vector<std::uint64_t> ne,
               std::optional<std::uint64_t> offset = std::nullopt) {
    tensors_.push_back(
        {.name = std::move(name), .type = type, .ne = std::move(ne), .offset = offset});
    return *this;
  }

  Gguf& Version(std::uint32_t v) {
    version_ = v;
    return *this;
  }
  Gguf& Counts(std::optional<std::uint64_t> keys, std::optional<std::uint64_t> tensors) {
    keys_ = keys;
    count_ = tensors;
    return *this;
  }
  Gguf& Alignment(std::uint64_t a) {
    alignment_ = a;
    return *this;
  }

  // The file; `final_pad`: the last tensor's padding written; `extra`
  // bytes appended.
  std::vector<std::byte> Build(bool final_pad = true, std::size_t extra = 0) const {
    std::vector<std::byte> out;
    AppendText(out, "GGUF");
    Append(out, version_, 4);
    Append(out, count_.value_or(tensors_.size()), 8);
    Append(out, keys_.value_or(kvs_.size()), 8);
    for (const auto& kv : kvs_) {
      out.insert(out.end(), kv.begin(), kv.end());
    }
    std::uint64_t next = 0;
    std::vector<std::uint64_t> sizes;
    for (const TensorSpec& t : tensors_) {
      const auto name = StringBytes(t.name);
      out.insert(out.end(), name.begin(), name.end());
      Append(out, t.ne.size(), 4);
      for (const std::uint64_t n : t.ne) {
        Append(out, n, 8);
      }
      Append(out, t.type, 4);
      Append(out, t.offset.value_or(next), 8);
      const std::uint64_t bytes = GgmlBytes(t.type, t.ne);
      sizes.push_back(bytes);
      next += Padded(bytes);
    }
    out.resize(Padded(out.size()));
    for (std::size_t i = 0; i < sizes.size(); ++i) {
      // A tensor larger than any test writes (one past the file's end) is
      // left short.
      for (std::uint64_t b = 0; b < std::min<std::uint64_t>(sizes[i], kMaxData); ++b) {
        out.push_back(static_cast<std::byte>(((i * 31) + b) & 0xFFU));
      }
      if (final_pad || i + 1 < sizes.size()) {
        out.resize(Padded(out.size()));
      }
    }
    out.resize(out.size() + extra, std::byte{0x5A});
    return out;
  }

 private:
  static constexpr std::uint64_t kMaxData = std::uint64_t{1} << 20U;
  struct TensorSpec {
    std::string name;
    std::uint32_t type = 0;
    std::vector<std::uint64_t> ne;
    std::optional<std::uint64_t> offset;
  };
  std::uint64_t Padded(std::uint64_t n) const {
    return (n + alignment_ - 1) / alignment_ * alignment_;
  }

  std::vector<std::vector<std::byte>> kvs_;
  std::vector<TensorSpec> tensors_;
  std::uint32_t version_ = 3;
  std::optional<std::uint64_t> keys_;
  std::optional<std::uint64_t> count_;
  std::uint64_t alignment_ = 32;
};

std::optional<gg::Fault> FaultOf(const std::expected<gg::Header, gg::Error>& r) {
  return Failed(r, &gg::Error::fault);
}

// ---------------------------------------------------------------- base/gguf

TEST(BaseGgufTest, ParsesEveryValueTypeAndTensorInfo) {
  std::vector<std::byte> u16s = Le(7, 2);
  Append(u16s, 65535, 2);
  std::vector<std::byte> f64s = Le(std::bit_cast<std::uint64_t>(-2.5), 8);
  const auto file = Gguf()
                        .U8("u8", 200)
                        .Kv("i8", gg::Type::kI8, Le(0xFF, 1))
                        .U16("u16", 65535)
                        .Kv("i16", gg::Type::kI16, Le(0x8000, 2))
                        .U32("u32", 4000000000U)
                        .I32("i32", -5)
                        .F32("f32", 0.5F)
                        .Bool("bool", 1)
                        .U64("u64", std::numeric_limits<std::uint64_t>::max())
                        .Kv("i64", gg::Type::kI64, Le(static_cast<std::uint64_t>(-9), 8))
                        .Kv("f64", gg::Type::kF64, Le(std::bit_cast<std::uint64_t>(1e300), 8))
                        .Array("u16s", gg::Type::kU16, 2, u16s)
                        .Array("f64s", gg::Type::kF64, 1, f64s)
                        .Array("bools", gg::Type::kBool, 2, {std::byte{0}, std::byte{1}})
                        .Array("empty", gg::Type::kString, 0, {})
                        .Strings("words", {"a", "", "\xC3\xA9"})
                        .Tensor("t", kF32, {3, 2})
                        .Build();
  const auto header = gg::Parse(file);
  ASSERT_TRUE(header.has_value()) << header.error().ToString();
  EXPECT_EQ(header->version, 3U);
  ASSERT_EQ(header->metadata.size(), 17U);
  const auto value = [&](std::string_view key) {
    const gg::KeyValue* kv = header->Find(key);
    EXPECT_NE(kv, nullptr) << key;
    return kv;
  };
  EXPECT_EQ(gg::SignedValue(gg::Type::kU8, gg::ScalarBits(*value("u8"))), 200);
  EXPECT_EQ(gg::SignedValue(gg::Type::kI8, gg::ScalarBits(*value("i8"))), -1);
  EXPECT_EQ(gg::SignedValue(gg::Type::kU16, gg::ScalarBits(*value("u16"))), 65535);
  EXPECT_EQ(gg::SignedValue(gg::Type::kI16, gg::ScalarBits(*value("i16"))), -32768);
  EXPECT_EQ(gg::SignedValue(gg::Type::kU32, gg::ScalarBits(*value("u32"))), 4000000000LL);
  EXPECT_EQ(gg::SignedValue(gg::Type::kI32, gg::ScalarBits(*value("i32"))), -5);
  EXPECT_EQ(gg::FloatValue(gg::Type::kF32, gg::ScalarBits(*value("f32"))), 0.5);
  EXPECT_EQ(gg::ScalarBits(*value("bool")), 1U);
  EXPECT_EQ(gg::SignedValue(gg::Type::kU64, gg::ScalarBits(*value("u64"))), std::nullopt);
  EXPECT_EQ(gg::SignedValue(gg::Type::kI64, gg::ScalarBits(*value("i64"))), -9);
  EXPECT_EQ(gg::FloatValue(gg::Type::kF64, gg::ScalarBits(*value("f64"))), 1e300);
  const gg::KeyValue& u16_list = *value("u16s");
  EXPECT_EQ(u16_list.element, gg::Type::kU16);
  EXPECT_EQ(u16_list.count, 2U);
  EXPECT_EQ(gg::ElementBits(u16_list, 1), 65535U);
  EXPECT_EQ(gg::FloatValue(gg::Type::kF64, gg::ElementBits(*value("f64s"), 0)), -2.5);
  std::vector<std::string> words;
  gg::ForEachString(*value("words"), [&](std::string_view s) { words.emplace_back(s); });
  EXPECT_EQ(words, (std::vector<std::string>{"a", "", "\xC3\xA9"}));
  EXPECT_EQ(value("empty")->count, 0U);
  ASSERT_EQ(header->tensors.size(), 1U);
  EXPECT_EQ(header->tensors[0].name, "t");
  EXPECT_EQ(header->tensors[0].dims, 2U);
  EXPECT_EQ(header->tensors[0].ne, (std::array<std::uint64_t, 4>{3, 2, 1, 1}));
  EXPECT_EQ(header->tensors[0].type, kF32);
  EXPECT_EQ(header->tensors[0].offset, 0U);
  EXPECT_LE(header->end, file.size());
  // Version 2 has version 3's layout.
  EXPECT_TRUE(gg::Parse(Gguf().Version(2).Build()).has_value());
}

TEST(BaseGgufTest, EveryTruncationIsTruncated) {
  std::vector<std::byte> u32s = Le(1, 4);
  Append(u32s, 2, 4);
  const auto file = Gguf()
                        .U32("general.alignment", 32)
                        .Strings("tokens", {"x", "yy"})
                        .Array("ids", gg::Type::kU32, 2, u32s)
                        .Tensor("a", kQ8_0, {32, 2})
                        .Tensor("b", kF32, {4})
                        .Build();
  const auto header = gg::Parse(file);
  ASSERT_TRUE(header.has_value());
  for (std::size_t n = 0; n < header->end; ++n) {
    EXPECT_EQ(FaultOf(gg::Parse(View(file).first(n))), gg::Fault::kTruncated) << n;
  }
}

TEST(BaseGgufTest, RefusesEachFault) {
  using enum gg::Fault;
  auto bad_magic = Gguf().Build();
  bad_magic[0] = std::byte{'X'};
  EXPECT_EQ(FaultOf(gg::Parse(bad_magic)), kMalformed);
  EXPECT_EQ(FaultOf(gg::Parse(Gguf().Version(1).Build())), kMalformed);
  EXPECT_EQ(FaultOf(gg::Parse(Gguf().Version(4).Build())), kMalformed);
  EXPECT_EQ(FaultOf(gg::Parse(Gguf().Version(0x03000000).Build())), kMalformed);  // big endian
  // Counts past the limits, before anything is read or reserved.
  EXPECT_EQ(FaultOf(gg::Parse(Gguf().Counts((std::uint64_t{1} << 16U) + 1, 0).Build())), kBounds);
  EXPECT_EQ(FaultOf(gg::Parse(Gguf().Counts(1, (std::uint64_t{1} << 20U) + 1).Build())), kBounds);
  EXPECT_EQ(FaultOf(gg::Parse(Gguf().Counts(std::uint64_t{1} << 40U, 0).Build())), kBounds);
  // Counts the bytes cannot hold.
  EXPECT_EQ(FaultOf(gg::Parse(Gguf().Alignment(1).Counts(60000, 0).Build())), kTruncated);
  EXPECT_EQ(FaultOf(gg::Parse(Gguf().Alignment(1).Counts(1, 1000000).Build())), kTruncated);
  // Keys.
  EXPECT_EQ(FaultOf(gg::Parse(Gguf().U32("", 1).Build())), kName);
  EXPECT_EQ(FaultOf(gg::Parse(Gguf().U32("a\xFF", 1).Build())), kName);
  EXPECT_EQ(FaultOf(gg::Parse(Gguf().U32("\xC3", 1).Build())), kName);  // a cut sequence
  EXPECT_EQ(FaultOf(gg::Parse(Gguf().U32(std::string(70000, 'k'), 1).Build())), kName);
  EXPECT_EQ(FaultOf(gg::Parse(Gguf().U32("k", 1).U32("k", 2).Build())), kDuplicate);
  // Values.
  EXPECT_EQ(FaultOf(gg::Parse(Gguf().KvRaw("k", 13, Le(0, 8)).Build())), kMalformed);
  EXPECT_EQ(FaultOf(gg::Parse(Gguf().Bool("k", 2).Build())), kMalformed);
  EXPECT_EQ(FaultOf(gg::Parse(Gguf().Array("k", gg::Type::kBool, 1, {std::byte{3}}).Build())),
            kMalformed);
  EXPECT_EQ(FaultOf(gg::Parse(Gguf().Array("k", gg::Type::kArray, 0, {}).Build())), kMalformed);
  EXPECT_EQ(FaultOf(gg::Parse(Gguf().ArrayRaw("k", 99, 0, {}).Build())), kMalformed);
  EXPECT_EQ(
      FaultOf(gg::Parse(Gguf().Array("k", gg::Type::kU64, std::uint64_t{1} << 60U, {}).Build())),
      kTruncated);
  EXPECT_EQ(
      FaultOf(gg::Parse(Gguf().Array("k", gg::Type::kString, std::uint64_t{1} << 60U, {}).Build())),
      kTruncated);
  EXPECT_EQ(FaultOf(gg::Parse(Gguf().Kv("k", gg::Type::kString, Le(~std::uint64_t{0}, 8)).Build())),
            kTruncated);
  // Per-string and per-array limits.
  const gg::Limits tight{.max_string_bytes = 4, .max_array_elements = 2};
  EXPECT_TRUE(gg::Parse(Gguf().Str("k", "abcd").Strings("l", {"a", "b"}).Build(), tight));
  EXPECT_EQ(FaultOf(gg::Parse(Gguf().Str("k", "abcde").Build(), tight)), kBounds);
  EXPECT_EQ(FaultOf(gg::Parse(Gguf().Strings("k", {"a", "b", "c"}).Build(), tight)), kBounds);
  EXPECT_EQ(FaultOf(gg::Parse(Gguf().Strings("k", {"abcde"}).Build(), tight)), kBounds);
  // Tensor infos.
  EXPECT_EQ(FaultOf(gg::Parse(Gguf().Tensor("t", kF32, {1, 1, 1, 1, 1}).Build())), kMalformed);
  EXPECT_EQ(FaultOf(gg::Parse(Gguf().Tensor("t", kF32, {std::uint64_t{1} << 63U}).Build())),
            kMalformed);
  EXPECT_EQ(
      FaultOf(gg::Parse(
          Gguf().Tensor("t", kF32, {std::uint64_t{1} << 32U, std::uint64_t{1} << 32U}).Build())),
      kMalformed);
  EXPECT_EQ(FaultOf(gg::Parse(Gguf().Tensor("", kF32, {1}).Build())), kName);
  EXPECT_EQ(FaultOf(gg::Parse(Gguf().Tensor("t\x80", kF32, {1}).Build())), kName);
  EXPECT_EQ(FaultOf(gg::Parse(Gguf().Tensor(std::string(64, 'n'), kF32, {1}).Build())), kName);
  EXPECT_TRUE(gg::Parse(Gguf().Tensor(std::string(63, 'n'), kF32, {1}).Build()).has_value());
  EXPECT_EQ(FaultOf(gg::Parse(Gguf().Tensor("t", kF32, {1}).Tensor("t", kF32, {1}).Build())),
            kDuplicate);
}

// ---------------------------------------------------------------- GGUF reader

template <typename T>
const T* Get(const Checkpoint& c, std::string_view key) {
  const MetaValue* v = c.FindValue(key);
  return v == nullptr ? nullptr : std::get_if<T>(v);
}

TEST(ImportGgufTest, ReadsTensorsAndMetadata) {
  std::vector<std::byte> u64s = Le(1, 8);
  Append(u64s, std::numeric_limits<std::uint64_t>::max(), 8);
  std::vector<std::byte> i32s = Le(3, 4);
  Append(i32s, static_cast<std::uint32_t>(-4), 4);
  const auto file = Gguf()
                        .U32("test.block_count", 2)
                        .F32("test.eps", 1e-6F)
                        .Bool("test.flag", 1)
                        .Str("general.name", "a name")
                        .Array("test.list", gg::Type::kI32, 2, i32s)
                        .Array("test.big", gg::Type::kU64, 2, u64s)
                        .Array("test.bools", gg::Type::kBool, 2, {std::byte{1}, std::byte{0}})
                        .Strings("tokenizer.ggml.tokens", {"a", "b"})
                        .U32("key with space", 1)
                        .Tensor("token_embd.weight", kQ8_0, {64, 2})
                        .Tensor("norm.weight", kF32, {8})
                        .Tensor("k.weight", kQ4_K, {256, 1})
                        .Tensor("fp4.weight", kNvfp4, {64, 2})
                        .Tensor("bf.weight", kBF16, {4, 4, 2})
                        .Tensor("blk.0.ffn_gate_exps.weight", kQ8_0, {32, 4, 3})
                        .Tensor("blk.0.ffn_down_exps.bias", kF16, {4, 3})
                        .Build();
  const auto c = ParseGguf(file);
  ASSERT_TRUE(c.has_value()) << c.error().ToString();
  EXPECT_EQ(c->container, "gguf");
  EXPECT_EQ(c->architecture, "test");
  ASSERT_EQ(c->tensors.size(), 7U);
  ASSERT_EQ(c->files.size(), 1U);
  EXPECT_EQ(c->files[0].bytes, file.size());

  const Tensor& q8 = c->tensors[0];
  EXPECT_EQ(q8.name, "token_embd.weight");
  EXPECT_EQ(q8.format.encoding, Encoding::kQ8_0);
  EXPECT_EQ(q8.format.layout, Layout::kGgmlBlocks);
  EXPECT_EQ(q8.format.shape, (format::Shape{.ne = {64, 2, 1, 1}, .rank = 2}));
  EXPECT_EQ(q8.part, PartRole::kData);
  EXPECT_EQ(q8.dtype, "q8_0");
  EXPECT_EQ(q8.dims, (std::vector<std::uint64_t>{64, 2}));
  EXPECT_EQ(q8.extent.bytes, 136U);
  EXPECT_EQ(q8.extent.offset % 32, 0U);
  EXPECT_EQ(c->tensors[1].format.layout, Layout::kDense);
  EXPECT_EQ(c->tensors[1].extent.offset, q8.extent.offset + 160);  // 136 padded to 32
  EXPECT_EQ(c->tensors[2].format.encoding, Encoding::kQ4_K);
  EXPECT_EQ(c->tensors[3].format.layout, Layout::kGgmlNvfp4Blocks);
  EXPECT_EQ(c->tensors[3].extent.bytes, 72U);
  EXPECT_EQ(c->tensors[4].format.shape.rank, 3U);
  EXPECT_FALSE(c->tensors[4].experts);
  EXPECT_TRUE(c->tensors[5].experts);
  EXPECT_FALSE(c->tensors[6].experts);  // two dimensions
  // The bytes are where the extents say.
  for (const Tensor& t : c->tensors) {
    const auto i = static_cast<std::size_t>(&t - c->tensors.data());
    EXPECT_EQ(file[t.extent.offset], static_cast<std::byte>((i * 31) & 0xFFU)) << t.name;
  }

  // Metadata, sorted, typed.
  EXPECT_TRUE(std::ranges::is_sorted(c->metadata, {}, &MetaEntry::key));
  EXPECT_EQ(*Get<std::int64_t>(*c, "test.block_count"), 2);
  EXPECT_EQ(*Get<double>(*c, "test.eps"), static_cast<double>(1e-6F));
  EXPECT_EQ(*Get<bool>(*c, "test.flag"), true);
  EXPECT_EQ(*Get<std::string>(*c, "general.name"), "a name");
  EXPECT_EQ(*Get<std::vector<std::int64_t>>(*c, "test.list"), (std::vector<std::int64_t>{3, -4}));
  EXPECT_TRUE(std::holds_alternative<std::monostate>(*c->FindValue("test.big")));
  EXPECT_EQ(*Get<std::vector<std::int64_t>>(*c, "test.bools"), (std::vector<std::int64_t>{1, 0}));
  EXPECT_EQ(*Get<std::vector<std::string>>(*c, "tokenizer.ggml.tokens"),
            (std::vector<std::string>{"a", "b"}));

  // What a mapping reads: no string lists, no none, no key it cannot name.
  std::map<std::string, spec::ParamValue> values;
  for (spec::SourceValue& v : ValuesOf(*c)) {
    values.emplace(v.key, std::move(v.value));
  }
  EXPECT_EQ(values.count("test.block_count"), 1U);
  EXPECT_EQ(values.count("test.list"), 1U);
  EXPECT_EQ(values.count("general.name"), 1U);
  EXPECT_EQ(values.count("test.big"), 0U);
  EXPECT_EQ(values.count("tokenizer.ggml.tokens"), 0U);
  EXPECT_EQ(values.count("key with space"), 0U);
  const spec::Source source = SourceOf(*c);
  EXPECT_EQ(source.architecture, "test");
  EXPECT_EQ(source.container, "gguf");
  ASSERT_EQ(source.tensors.size(), 7U);
  EXPECT_EQ(source.tensors[5].format, c->tensors[5].format);
  EXPECT_TRUE(source.tensors[5].experts);
}

TEST(ImportGgufTest, AlignmentAndPacking) {
  // An alignment of 64, and a file that stops at its last tensor's end.
  auto aligned = Gguf().U32("general.alignment", 64).Alignment(64);
  aligned.Tensor("a", kF32, {3}).Tensor("b", kF32, {5});
  const auto c = ParseGguf(aligned.Build(false));
  ASSERT_TRUE(c.has_value()) << c.error().ToString();
  EXPECT_EQ(c->tensors[1].extent.offset - c->tensors[0].extent.offset, 64U);
  EXPECT_EQ(c->tensors[0].extent.offset % 64, 0U);
  for (const std::uint32_t a : {0U, 3U, 48U}) {
    EXPECT_EQ(RuleOf(ParseGguf(Gguf().U32("general.alignment", a).Build())), Rule::kContainer) << a;
  }
  EXPECT_EQ(RuleOf(ParseGguf(Gguf().U64("general.alignment", std::uint64_t{1} << 32U).Build())),
            Rule::kContainer);
  EXPECT_EQ(RuleOf(ParseGguf(Gguf().Str("general.alignment", "32").Build())), Rule::kContainer);
  // Any integer type names the alignment.
  EXPECT_TRUE(ParseGguf(Gguf().U64("general.alignment", 32).Build()).has_value());
  // Offsets: packed as gguf.cpp writes them; a gap, a misaligned start, a
  // first not at 0 and another order are as unambiguous; an overlap is
  // refused.
  const auto offsets = [](std::uint64_t a, std::uint64_t b) {
    return ParseGguf(Gguf().Tensor("a", kF32, {3}, a).Tensor("b", kF32, {3}, b).Build(true, 64));
  };
  EXPECT_TRUE(offsets(0, 32).has_value());
  for (const auto& [a, b] :
       {std::pair<std::uint64_t, std::uint64_t>{0, 64}, {0, 12}, {32, 64}, {64, 0}}) {
    const auto placed = offsets(a, b);
    ASSERT_TRUE(placed.has_value()) << a << " " << b << ": " << placed.error().ToString();
    EXPECT_EQ(placed->tensors[1].extent.offset - placed->tensors[0].extent.offset, b - a);
  }
  EXPECT_EQ(RuleOf(offsets(0, 0)), Rule::kContainer);
  EXPECT_EQ(RuleOf(offsets(0, 8)), Rule::kContainer);
  EXPECT_EQ(RuleOf(offsets(8, 0)), Rule::kContainer);
  // Data past the end; bytes after the data section, which are ignored.
  const auto two = Gguf().Tensor("a", kF32, {3}).Tensor("b", kF32, {3});
  auto short_file = two.Build(false);
  short_file.pop_back();
  EXPECT_EQ(RuleOf(ParseGguf(short_file)), Rule::kContainer);
  EXPECT_TRUE(ParseGguf(two.Build(true)).has_value());
  EXPECT_TRUE(ParseGguf(two.Build(true, 1)).has_value());
  EXPECT_TRUE(ParseGguf(Gguf().Build()).has_value());  // no tensors
  EXPECT_TRUE(ParseGguf(Gguf().Build(true, 33)).has_value());
}

TEST(ImportGgufTest, EveryPrefixIsRefused) {
  std::vector<std::byte> f32s = Le(std::bit_cast<std::uint32_t>(1.0F), 4);
  const auto file = Gguf()
                        .U32("general.alignment", 32)
                        .Array("test.f", gg::Type::kF32, 1, f32s)
                        .Strings("test.s", {"one", "two"})
                        .Tensor("a", kQ8_0, {32, 2})
                        .Tensor("b", kF32, {3})
                        .Build(false);
  ASSERT_TRUE(ParseGguf(file).has_value());
  for (std::size_t n = 0; n < file.size(); ++n) {
    const auto rule = RuleOf(ParseGguf(View(file).first(n)));
    EXPECT_EQ(rule, Rule::kContainer) << n;
  }
}

TEST(ImportGgufTest, RefusesTypesShapesAndNames) {
  for (const std::uint32_t type : {4U, 5U, 31U, 36U, 38U, 43U, 1000U}) {
    EXPECT_EQ(RuleOf(ParseGguf(Gguf().Tensor("t", type, {32}).Build())), Rule::kUnsupported)
        << type;
  }
  EXPECT_EQ(RuleOf(ParseGguf(Gguf().Tensor("t", kQ4_K, {128}).Build())), Rule::kContainer);
  EXPECT_EQ(RuleOf(ParseGguf(Gguf().Tensor("t", kQ8_0, {33}).Build())), Rule::kContainer);
  EXPECT_EQ(RuleOf(ParseGguf(Gguf().Tensor("t", kF32, {4, 0}).Build())), Rule::kUnsupported);
  // 2^62 F32 elements: 2^64 bytes.
  EXPECT_EQ(
      RuleOf(ParseGguf(
          Gguf().Tensor("t", kF32, {std::uint64_t{1} << 40U, std::uint64_t{1} << 22U}).Build())),
      Rule::kBounds);
  // Past the end of the file (2^40 bytes).
  EXPECT_EQ(RuleOf(ParseGguf(Gguf().Tensor("t", kF32, {std::uint64_t{1} << 38U}).Build())),
            Rule::kContainer);
  EXPECT_EQ(RuleOf(ParseGguf(Gguf().Tensor("t", kF32, {1}).Tensor("t", kF32, {1}).Build())),
            Rule::kContainer);
  EXPECT_EQ(RuleOf(ParseGguf(Gguf().U32("\xFF", 1).Build())), Rule::kContainer);
  EXPECT_EQ(RuleOf(ParseGguf(Gguf().Array("k", gg::Type::kArray, 0, {}).Build())),
            Rule::kContainer);
  EXPECT_EQ(RuleOf(ParseGguf(Gguf().Counts(std::uint64_t{1} << 17U, 0).Build())), Rule::kBounds);
  EXPECT_EQ(RuleOf(ParseGguf(Gguf().Counts(1, std::uint64_t{1} << 21U).Build())), Rule::kBounds);
  // The architecture: absent, not a string, too long.
  EXPECT_EQ(RuleOf(ParseGguf(Gguf().Remove("general.architecture").Build())), Rule::kContainer);
  EXPECT_EQ(RuleOf(ParseGguf(
                Gguf().Remove("general.architecture").U32("general.architecture", 1).Build())),
            Rule::kContainer);
  EXPECT_EQ(RuleOf(ParseGguf(Gguf()
                                 .Remove("general.architecture")
                                 .Str("general.architecture", std::string(201, 'a'))
                                 .Build())),
            Rule::kContainer);
  for (const std::string_view arch : {"a\nb", "\xFF", "x\x7F"}) {
    EXPECT_EQ(RuleOf(ParseGguf(
                  Gguf().Remove("general.architecture").Str("general.architecture", arch).Build())),
              Rule::kContainer);
  }
  EXPECT_EQ(RuleOf(ParseGguf(Gguf().Tensor("a\tb", kF32, {1}).Build())), Rule::kContainer);
  // Names past gguf.cpp's 63 bytes, up to the reader's cap.
  EXPECT_TRUE(
      ParseGguf(Gguf().Tensor(std::string(kMaxNameBytes, 'n'), kF32, {1}).Build()).has_value());
  EXPECT_EQ(
      RuleOf(ParseGguf(Gguf().Tensor(std::string(kMaxNameBytes + 1, 'n'), kF32, {1}).Build())),
      Rule::kContainer);
  // Strings and arrays past the reader's caps.
  EXPECT_EQ(RuleOf(ParseGguf(Gguf().Str("k", std::string((16U << 20U) + 1, 'x')).Build())),
            Rule::kBounds);
}

TEST(ImportGgufTest, RandomCorruptionsNeverCrash) {
  std::vector<std::byte> u32s = Le(1, 4);
  Append(u32s, 2, 4);
  const auto file = Gguf()
                        .U32("general.alignment", 32)
                        .Strings("tokens", {"x", "yy"})
                        .Array("ids", gg::Type::kU32, 2, u32s)
                        .U16("split.count", 1)
                        .Tensor("a", kQ8_0, {32, 2})
                        .Tensor("blk.0.ffn_up_exps.weight", kF32, {2, 2, 2})
                        .Tensor("b", kNvfp4, {64})
                        .Build();
  Corrupt(file, [](std::span<const std::byte> b) { return ParseGguf(b); });
}

TEST(ImportGgufTest, SplitKeysInOneFile) {
  const auto split = [](std::uint16_t no, std::uint16_t count, std::int32_t tensors) {
    return Gguf()
        .U16("split.no", no)
        .U16("split.count", count)
        .I32("split.tensors.count", tensors)
        .Tensor("a", kF32, {2})
        .Build();
  };
  EXPECT_TRUE(ParseGguf(split(0, 1, 1)).has_value());
  EXPECT_EQ(RuleOf(ParseGguf(split(0, 2, 1))), Rule::kShards);
  EXPECT_EQ(RuleOf(ParseGguf(split(0, 1, 2))), Rule::kShards);
  EXPECT_EQ(RuleOf(ParseGguf(split(1, 1, 1))), Rule::kShards);
  EXPECT_EQ(RuleOf(ParseGguf(split(0, 0, 1))), Rule::kShards);
  EXPECT_EQ(RuleOf(ParseGguf(split(0, 1, -1))), Rule::kShards);
  // Any integer type is a count; a string is not.
  EXPECT_TRUE(ParseGguf(Gguf().U32("split.count", 1).Build()).has_value());
  EXPECT_EQ(RuleOf(ParseGguf(Gguf().Str("split.count", "1").Build())), Rule::kShards);
  // llama.cpp loads a split set without split.tensors.count.
  EXPECT_TRUE(
      ParseGguf(Gguf().U16("split.no", 0).U16("split.count", 1).Tensor("a", kF32, {2}).Build())
          .has_value());
}

// The Gemma 2 mapping applied to a small checkpoint of its tensors and keys.
TEST(ImportGgufTest, MakesTheSourceAMappingReads) {
  Gguf g;
  g.Remove("general.architecture")
      .Str("general.architecture", "gemma2")
      .U32("gemma2.block_count", 2)
      .U32("gemma2.context_length", 128)
      .U32("gemma2.embedding_length", 8)
      .U32("gemma2.feed_forward_length", 16)
      .U32("gemma2.attention.head_count", 2)
      .U32("gemma2.attention.head_count_kv", 1)
      .U32("gemma2.attention.key_length", 4)
      .U32("gemma2.attention.value_length", 4)
      .F32("gemma2.attention.layer_norm_rms_epsilon", 1e-6F)
      .F32("gemma2.attn_logit_softcapping", 50.0F)
      .F32("gemma2.final_logit_softcapping", 30.0F)
      .U32("gemma2.attention.sliding_window", 64)
      .Strings("tokenizer.ggml.tokens", {"a", "b"});
  g.Tensor("token_embd.weight", kF32, {8, 10}).Tensor("output_norm.weight", kF32, {8});
  for (int il = 0; il < 2; ++il) {
    const auto name = [&](std::string_view t) { return std::format("blk.{}.{}.weight", il, t); };
    g.Tensor(name("attn_norm"), kF32, {8})
        .Tensor(name("attn_q"), kF32, {8, 8})
        .Tensor(name("attn_k"), kF32, {8, 4})
        .Tensor(name("attn_v"), kF32, {8, 4})
        .Tensor(name("attn_output"), kF32, {8, 8})
        .Tensor(name("post_attention_norm"), kF32, {8})
        .Tensor(name("ffn_norm"), kF32, {8})
        .Tensor(name("ffn_gate"), kF32, {8, 16})
        .Tensor(name("ffn_up"), kF32, {8, 16})
        .Tensor(name("ffn_down"), kF32, {16, 8})
        .Tensor(name("post_ffw_norm"), kF32, {8});
  }
  const auto c = ParseGguf(g.Build());
  ASSERT_TRUE(c.has_value()) << c.error().ToString();
  const spec::Mapping* mapping = spec::FindMapping(c->architecture);
  ASSERT_NE(mapping, nullptr);
  const auto s = spec::Synthesize(*mapping, SourceOf(*c), ValuesOf(*c));
  ASSERT_TRUE(s.has_value()) << s.error();
  ASSERT_EQ(s->stages.size(), 1U);
  EXPECT_EQ(s->stages[0].layers.size(), 2U);
  EXPECT_EQ(s->stages[0].context, 128U);
  EXPECT_EQ(s->stages[0].folds, std::vector<std::string>{"norm.plus_one"});
  ASSERT_TRUE(s->stages[0].heads.output.has_value());
  if (const auto& output = s->stages[0].heads.output) {
    EXPECT_EQ(output->weights.at(0).tensor, "token_embd.weight");
  }
}

// ---------------------------------------------------------------- files

using ImportGgufFilesTest = ScratchTest;

// Three split files as gguf-split writes them: the first with the
// metadata, each with its split keys.
std::vector<std::vector<std::byte>> Splits(std::uint16_t count = 3) {
  std::vector<std::vector<std::byte>> out;
  for (std::uint16_t i = 0; i < count; ++i) {
    Gguf g;
    if (i != 0) {
      g.Remove("general.architecture");
    }
    g.U16("split.no", i).U16("split.count", count).I32("split.tensors.count", 2 * count);
    if (i == 0) {
      g.U32("test.block_count", 1);
    }
    g.Tensor(std::format("t{}a", i), kF32, {3}).Tensor(std::format("t{}b", i), kQ8_0, {32});
    out.push_back(g.Build(false));
  }
  return out;
}

std::string SplitFile(int no, int count) {
  return std::format("model-{:05}-of-{:05}.gguf", no, count);
}

TEST_F(ImportGgufFilesTest, ReadsSplitFilesAndTheirBytes) {
  const auto splits = Splits();
  for (int i = 0; i < 3; ++i) {
    Write(dir_ / SplitFile(i + 1, 3), splits[static_cast<std::size_t>(i)]);
  }
  const auto c = ReadCheckpoint(dir_ / SplitFile(1, 3));
  ASSERT_TRUE(c.has_value()) << c.error().ToString();
  ASSERT_EQ(c->files.size(), 3U);
  EXPECT_EQ(c->files[2].name, SplitFile(3, 3));
  EXPECT_EQ(c->files[2].path, dir_ / SplitFile(3, 3));
  ASSERT_EQ(c->tensors.size(), 6U);
  EXPECT_EQ(c->tensors[4].name, "t2a");
  EXPECT_EQ(c->tensors[4].extent.file, 2U);
  EXPECT_EQ(*Get<std::int64_t>(*c, "split.no"), 0);
  EXPECT_EQ(*Get<std::int64_t>(*c, "test.block_count"), 1);
  // The tensors' bytes, each read from its file opened again.
  for (const Tensor& t : c->tensors) {
    std::vector<std::byte> bytes(t.extent.bytes);
    ASSERT_TRUE(c->Read(t.extent, 0, bytes).has_value()) << t.name;
    const auto& file = splits[t.extent.file];
    EXPECT_TRUE(std::equal(bytes.begin(), bytes.end(),
                           file.begin() + static_cast<std::ptrdiff_t>(t.extent.offset)))
        << t.name;
  }
  std::vector<std::byte> past(c->tensors[0].extent.bytes + 1);
  EXPECT_EQ(Failed(c->Read(c->tensors[0].extent, 0, past), &Error::rule), Rule::kBounds);
  std::vector<std::byte> one(1);
  EXPECT_EQ(Failed(c->Read(c->tensors[0].extent, c->tensors[0].extent.bytes, one), &Error::rule),
            Rule::kBounds);
}

TEST_F(ImportGgufFilesTest, RefusesSplitSetsThatDisagree) {
  const auto write = [&](const std::vector<std::vector<std::byte>>& files) {
    std::filesystem::remove_all(dir_);
    std::filesystem::create_directories(dir_);
    for (std::size_t i = 0; i < files.size(); ++i) {
      Write(dir_ / SplitFile(static_cast<int>(i) + 1, 3), files[i]);
    }
    return ReadGguf(dir_ / SplitFile(1, 3));
  };
  auto files = Splits();
  // A missing split.
  EXPECT_EQ(RuleOf(write({files[0], files[1]})), Rule::kFileType);
  // A split's number or count not its place's.
  auto swapped = files;
  std::swap(swapped[1], swapped[2]);
  EXPECT_EQ(RuleOf(write(swapped)), Rule::kShards);
  auto other_count = files;
  other_count[2] = Splits(4)[2];
  EXPECT_EQ(RuleOf(write(other_count)), Rule::kShards);
  // A key the first file has, with another value.
  auto disagree = files;
  disagree[1] = Gguf()
                    .Remove("general.architecture")
                    .U16("split.no", 1)
                    .U16("split.count", 3)
                    .I32("split.tensors.count", 6)
                    .Str("general.architecture", "other")
                    .Tensor("t1a", kF32, {3})
                    .Tensor("t1b", kQ8_0, {32})
                    .Build(false);
  EXPECT_EQ(RuleOf(write(disagree)), Rule::kShards);
  // A tensor in two files.
  auto twice = files;
  twice[2] = Gguf()
                 .Remove("general.architecture")
                 .U16("split.no", 2)
                 .U16("split.count", 3)
                 .I32("split.tensors.count", 6)
                 .Tensor("t0a", kF32, {3})
                 .Tensor("t2b", kQ8_0, {32})
                 .Build(false);
  EXPECT_EQ(RuleOf(write(twice)), Rule::kShards);
  // A tensor count other than split.tensors.count.
  auto fewer = files;
  fewer[2] = Gguf()
                 .Remove("general.architecture")
                 .U16("split.no", 2)
                 .U16("split.count", 3)
                 .I32("split.tensors.count", 6)
                 .Tensor("t2a", kF32, {3})
                 .Build(false);
  EXPECT_EQ(RuleOf(write(fewer)), Rule::kShards);
  // Read from a later split, or a first file not named as splits are.
  ASSERT_TRUE(write(files).has_value());
  EXPECT_EQ(RuleOf(ReadGguf(dir_ / SplitFile(2, 3))), Rule::kShards);
  Write(dir_ / "renamed.gguf", files[0]);
  EXPECT_EQ(RuleOf(ReadGguf(dir_ / "renamed.gguf")), Rule::kShards);
}

// Files are opened as any file is, links followed (D-073 as amended on
// 2026-10-09): who could change them is not the reader's business, but
// only a regular file is opened.
TEST_F(ImportGgufFilesTest, OpensRegularFiles) {
  const auto file = Gguf().Tensor("a", kF32, {2}).Build();
  Write(dir_ / "m.gguf", file);
  ASSERT_TRUE(ReadGguf(dir_ / "m.gguf").has_value());
  // Through a link, as a Hugging Face cache's snapshots link into blobs.
  std::filesystem::create_directories(dir_ / "blobs");
  Write(dir_ / "blobs" / "abc", file);
  std::filesystem::create_symlink("blobs/abc", dir_ / "linked.gguf");
  EXPECT_TRUE(ReadCheckpoint(dir_ / "linked.gguf").has_value());
  // A file others can write, in a directory others can change.
  ::chmod((dir_ / "m.gguf").c_str(), 0666);
  EXPECT_TRUE(ReadGguf(dir_ / "m.gguf").has_value());
  std::filesystem::create_directories(dir_ / "open");
  Write(dir_ / "open" / "m.gguf", file);
  ::chmod((dir_ / "open").c_str(), 0777);
  EXPECT_TRUE(ReadGguf(dir_ / "open" / "m.gguf").has_value());
  ::chmod((dir_ / "open").c_str(), 0755);
  // Not a regular file; absent.
  std::filesystem::create_directories(dir_ / "dir.gguf");
  EXPECT_EQ(RuleOf(ReadGguf(dir_ / "dir.gguf")), Rule::kFileType);
  ASSERT_EQ(::mkfifo((dir_ / "fifo.gguf").c_str(), 0600), 0);
  EXPECT_EQ(RuleOf(ReadGguf(dir_ / "fifo.gguf")), Rule::kFileType);
  EXPECT_EQ(RuleOf(ReadGguf(dir_ / "absent.gguf")), Rule::kFileType);
  std::filesystem::create_symlink("fifo.gguf", dir_ / "to-fifo.gguf");
  EXPECT_EQ(RuleOf(ReadGguf(dir_ / "to-fifo.gguf")), Rule::kFileType);
  // Links that loop, and a link to nothing.
  std::filesystem::create_symlink("loop-b.gguf", dir_ / "loop-a.gguf");
  std::filesystem::create_symlink("loop-a.gguf", dir_ / "loop-b.gguf");
  EXPECT_EQ(RuleOf(ReadGguf(dir_ / "loop-a.gguf")), Rule::kFileType);
  std::filesystem::create_symlink("blobs/missing", dir_ / "dangling.gguf");
  const auto dangling = ReadGguf(dir_ / "dangling.gguf");
  EXPECT_EQ(RuleOf(dangling), Rule::kFileType);
  if (!dangling) {
    EXPECT_EQ(dangling.error().reason, "a checkpoint file is a link to nothing");
  }
  // A path no reader claims.
  Write(dir_ / "m.bin", file);
  EXPECT_EQ(RuleOf(ReadCheckpoint(dir_ / "m.bin")), Rule::kFileType);
}

// A tensor's bytes come from the file its reader read: one renamed over
// it or shrunk under it fails the read.
TEST_F(ImportGgufFilesTest, ReadsOnlyTheFilesItRead) {
  const auto file = Gguf().Tensor("a", kF32, {64}).Build();
  Write(dir_ / "m.gguf", file);
  const auto c = ReadGguf(dir_ / "m.gguf");
  ASSERT_TRUE(c.has_value()) << c.error().ToString();
  const Extent extent = c->tensors[0].extent;
  std::vector<std::byte> bytes(extent.bytes);
  ASSERT_TRUE(c->Read(extent, 0, bytes).has_value());
  EXPECT_TRUE(std::equal(bytes.begin(), bytes.end(),
                         file.begin() + static_cast<std::ptrdiff_t>(extent.offset)));
  // Another file renamed over the one read.
  Write(dir_ / "new.gguf", file);
  std::filesystem::rename(dir_ / "new.gguf", dir_ / "m.gguf");
  EXPECT_EQ(Failed(c->Read(extent, 0, bytes), &Error::rule), Rule::kIo);
  // The file read shrunk under the checkpoint.
  const auto again = ReadGguf(dir_ / "m.gguf");
  ASSERT_TRUE(again.has_value()) << again.error().ToString();
  std::filesystem::resize_file(dir_ / "m.gguf", extent.offset + 8);
  EXPECT_EQ(Failed(again->Read(extent, 0, bytes), &Error::rule), Rule::kIo);
  // Gone.
  std::filesystem::remove(dir_ / "m.gguf");
  EXPECT_EQ(Failed(again->Read(extent, 0, bytes), &Error::rule), Rule::kFileType);
  // A checkpoint parsed from memory has no file to read.
  const auto memory = ParseGguf(file);
  ASSERT_TRUE(memory.has_value());
  EXPECT_EQ(Failed(memory->Read(extent, 0, bytes), &Error::rule), Rule::kIo);
  // A link to a device is refused before it is opened.
  std::filesystem::create_symlink("/dev/zero", dir_ / "zero.gguf");
  EXPECT_EQ(RuleOf(ReadGguf(dir_ / "zero.gguf")), Rule::kFileType);
}

// The kept-file rule (files.h): a file kept verbatim must resolve inside
// the checkpoint, or inside a Hugging Face cache repository's blobs/
// when the checkpoint is one of its snapshots; a link a repository
// planted to a file elsewhere does not.
TEST_F(ImportGgufFilesTest, KeptFilesResolveInsideTheCheckpoint) {
  namespace fs = std::filesystem;
  // A plain checkpoint directory.
  const fs::path repo = dir_ / "repo";
  Write(repo / "tokenizer.json", "{}");
  Write(repo / "sub" / "scheduler_config.json", "{}");
  fs::create_symlink("sub/scheduler_config.json", repo / "inner_link.json");
  Write(dir_ / "secret", "the user's own");
  fs::create_symlink(dir_ / "secret", repo / "planted.json");
  fs::create_symlink("../secret", repo / "relative.json");
  EXPECT_TRUE(ResolvesInside(repo, repo / "tokenizer.json"));
  EXPECT_TRUE(ResolvesInside(repo, repo / "sub" / "scheduler_config.json"));
  EXPECT_TRUE(ResolvesInside(repo, repo / "inner_link.json"));
  EXPECT_FALSE(ResolvesInside(repo, repo / "planted.json"));
  EXPECT_FALSE(ResolvesInside(repo, repo / "relative.json"));
  EXPECT_FALSE(ResolvesInside(repo, repo / "absent.json"));
  EXPECT_FALSE(ResolvesInside(repo, repo));  // the tree itself is not a file in it
  // A Hugging Face cache: snapshot files link into the repository's blobs.
  const fs::path cache = dir_ / "hub" / "models--org--name";
  const fs::path snapshot = cache / "snapshots" / "0123abcd";
  Write(cache / "blobs" / "f00d", "{}");
  fs::create_directories(snapshot / "scheduler");
  fs::create_symlink("../../blobs/f00d", snapshot / "tokenizer.json");
  fs::create_symlink("../../../blobs/f00d", snapshot / "scheduler" / "scheduler_config.json");
  fs::create_symlink(dir_ / "secret", snapshot / "planted.json");
  Write(cache / "refs" / "main", "0123abcd");
  fs::create_symlink("../../refs/main", snapshot / "outside_blobs.json");
  EXPECT_TRUE(ResolvesInside(snapshot, snapshot / "tokenizer.json"));
  EXPECT_TRUE(ResolvesInside(snapshot, snapshot / "scheduler" / "scheduler_config.json"));
  // A component's directory as the tree: still the snapshot's repository.
  EXPECT_TRUE(
      ResolvesInside(snapshot / "scheduler", snapshot / "scheduler" / "scheduler_config.json"));
  EXPECT_FALSE(ResolvesInside(snapshot, snapshot / "planted.json"));
  EXPECT_FALSE(ResolvesInside(snapshot, snapshot / "outside_blobs.json"));
  // A blobs/ that is itself a link elsewhere is not the repository's.
  const fs::path linked = dir_ / "hub" / "models--org--other";
  Write(dir_ / "elsewhere" / "f00d", "{}");
  fs::create_directories(linked / "snapshots" / "rev");
  fs::create_directory_symlink(dir_ / "elsewhere", linked / "blobs");
  fs::create_symlink("../../blobs/f00d", linked / "snapshots" / "rev" / "tokenizer.json");
  EXPECT_FALSE(ResolvesInside(linked / "snapshots" / "rev",
                              linked / "snapshots" / "rev" / "tokenizer.json"));
}

TEST_F(ImportGgufFilesTest, ReadsLargeHeadersInGrowingPrefixes) {
  // A header past the first read: a vocabulary of 300,000 tokens.
  std::vector<std::string> tokens;
  tokens.reserve(300000);
  for (int i = 0; i < 300000; ++i) {
    tokens.push_back(std::format("token{}", i));
  }
  const auto file = Gguf().Strings("tokenizer.ggml.tokens", tokens).Tensor("a", kF32, {2}).Build();
  ASSERT_GT(file.size(), kGgufFirstRead * 4);
  Write(dir_ / "big.gguf", file);
  const auto c = ReadGguf(dir_ / "big.gguf");
  ASSERT_TRUE(c.has_value()) << c.error().ToString();
  EXPECT_EQ(Get<std::vector<std::string>>(*c, "tokenizer.ggml.tokens")->size(), 300000U);
  // Truncated inside the header, after the first read.
  Write(dir_ / "cut.gguf", View(file).first(kGgufFirstRead + 100));
  EXPECT_EQ(RuleOf(ReadGguf(dir_ / "cut.gguf")), Rule::kContainer);
}

// ---------------------------------------------------------------- safetensors

constexpr std::string_view kConfig = R"({"model_type":"test","num_layers":2})";

TEST(ImportSafetensorsTest, ReadsTensorsAndConfig) {
  const std::string config = R"({
    "model_type": "test_arch",
    "architectures": ["TestForCausalLM"],
    "num_hidden_layers": 2,
    "rms_norm_eps": 1e-06,
    "rope_theta": 10000.0,
    "huge": 123456789012345678901234567890,
    "flag": true,
    "nothing": null,
    "ints": [1, -2, 3],
    "reals": [1, 2.5],
    "mixed": [1, "a"],
    "nested": [[1]],
    "empty": [],
    "text_config": {"hidden_size": 8, "rope": {"type": "yarn", "factor": 4.0}},
    "quantization_config": null
  })";
  const auto shard =
      Shard({E("b.weight", "BF16", {8, 4}), E("a.bias", "F32", {8}), E("ids", "I64", {3}),
             E("deep", "U8", {2, 3, 4, 5, 6}), E("scales", "F8_E8M0", {2}), E("scalar", "F32", {}),
             E("model.layers.0.mlp.experts.gate_up_proj", "BF16", {2, 3, 4})},
            R"({"format":"pt"})");
  const auto c = ParseSafetensors(shard, config);
  ASSERT_TRUE(c.has_value()) << c.error().ToString();
  EXPECT_EQ(c->container, "safetensors");
  EXPECT_EQ(c->architecture, "test_arch");
  EXPECT_TRUE(c->quantization.empty());
  ASSERT_EQ(c->tensors.size(), 7U);
  const Tensor& b = c->tensors[0];
  EXPECT_EQ(b.name, "b.weight");
  EXPECT_EQ(b.dtype, "BF16");
  EXPECT_EQ(b.dims, (std::vector<std::uint64_t>{8, 4}));
  EXPECT_EQ(b.format.encoding, Encoding::kBF16);
  EXPECT_EQ(b.format.layout, Layout::kDense);
  EXPECT_EQ(b.format.shape, (format::Shape{.ne = {4, 8, 1, 1}, .rank = 2}));  // GGML's order
  const std::uint64_t data = shard.size() - (64 + 32 + 24 + 720 + 2 + 4 + 48);
  EXPECT_EQ(b.extent.offset, data);
  EXPECT_EQ(b.extent.bytes, 64U);
  EXPECT_EQ(c->tensors[1].extent.offset, data + 64);
  EXPECT_EQ(c->tensors[3].format.shape, (format::Shape{.ne = {6, 5, 4, 6}, .rank = 4}));
  EXPECT_EQ(c->tensors[4].format.encoding, Encoding::kE8M0);
  EXPECT_EQ(c->tensors[5].format.shape.rank, 0U);
  EXPECT_TRUE(c->tensors[6].experts);
  EXPECT_FALSE(c->tensors[0].experts);

  EXPECT_EQ(*Get<std::int64_t>(*c, "num_hidden_layers"), 2);
  EXPECT_EQ(*Get<double>(*c, "rms_norm_eps"), 1e-6);
  EXPECT_EQ(*Get<double>(*c, "rope_theta"), 10000.0);
  EXPECT_EQ(*Get<double>(*c, "huge"), 123456789012345678901234567890.0);
  EXPECT_EQ(*Get<bool>(*c, "flag"), true);
  EXPECT_TRUE(std::holds_alternative<std::monostate>(*c->FindValue("nothing")));
  EXPECT_EQ(*Get<std::vector<std::int64_t>>(*c, "ints"), (std::vector<std::int64_t>{1, -2, 3}));
  EXPECT_EQ(*Get<std::vector<double>>(*c, "reals"), (std::vector<double>{1.0, 2.5}));
  EXPECT_TRUE(std::holds_alternative<std::monostate>(*c->FindValue("mixed")));
  EXPECT_TRUE(std::holds_alternative<std::monostate>(*c->FindValue("nested")));
  EXPECT_EQ(*Get<std::vector<std::int64_t>>(*c, "empty"), std::vector<std::int64_t>{});
  EXPECT_EQ(*Get<std::vector<std::string>>(*c, "architectures"),
            std::vector<std::string>{"TestForCausalLM"});
  EXPECT_EQ(*Get<std::int64_t>(*c, "text_config.hidden_size"), 8);
  EXPECT_EQ(*Get<std::string>(*c, "text_config.rope.type"), "yarn");
  EXPECT_EQ(c->FindValue("quantization_config"), nullptr);
  EXPECT_TRUE(std::ranges::is_sorted(c->metadata, {}, &MetaEntry::key));
  // The data tiles in offset order whatever the header's order.
  const auto reordered =
      Shard({E("x", "F32", {2}), E("y", "F32", {1})}, {}, {{0, "[4,12]"}, {1, "[0,4]"}});
  const auto r = ParseSafetensors(reordered, kConfig);
  ASSERT_TRUE(r.has_value()) << r.error().ToString();
  EXPECT_EQ(r->tensors[0].name, "y");
  EXPECT_EQ(r->tensors[1].name, "x");
  // Padding spaces after the header are safetensors' own alignment.
  EXPECT_TRUE(ParseSafetensors(Shard({E("x", "F32", {2})}, {}, {}, "   "), kConfig).has_value());
}

TEST(ImportSafetensorsTest, EveryPrefixIsRefused) {
  const auto shard = Shard({E("a", "F32", {3}), E("b", "BF16", {2, 2})}, R"({"k":"v"})");
  ASSERT_TRUE(ParseSafetensors(shard, kConfig).has_value());
  for (std::size_t n = 0; n < shard.size(); ++n) {
    EXPECT_EQ(RuleOf(ParseSafetensors(View(shard).first(n), kConfig)), Rule::kContainer) << n;
  }
}

TEST(ImportSafetensorsTest, RandomCorruptionsNeverCrash) {
  const auto shard = Shard({E("a.weight", "U8", {4, 16}), E("a.weight_scale", "F8_E4M3", {4, 2}),
                            E("a.weight_scale_2", "F32", {}), E("b", "BF16", {2, 3, 4})},
                           R"({"format":"pt"})");
  const std::string config =
      R"({"model_type":"t","quantization_config":{"quant_method":"modelopt","quant_algo":"NVFP4","group_size":16}})";
  ASSERT_TRUE(ParseSafetensors(shard, config).has_value());
  Corrupt(shard, [&](std::span<const std::byte> b) { return ParseSafetensors(b, config); });
  const auto config_bytes = std::as_bytes(std::span(config.data(), config.size()));
  Corrupt(std::vector<std::byte>(config_bytes.begin(), config_bytes.end()),
          [&](std::span<const std::byte> b) {
            return ParseSafetensors(
                shard, std::string_view(reinterpret_cast<const char*>(b.data()), b.size()));
          });
}

TEST(ImportSafetensorsTest, RefusesHeaders) {
  const auto rule = [](const std::vector<std::byte>& shard) {
    return RuleOf(ParseSafetensors(shard, kConfig));
  };
  // The length: past the cap, past the file.
  EXPECT_EQ(rule(Le(kMaxHeaderBytes + 1, 8)), Rule::kBounds);
  EXPECT_EQ(rule(RawShard("{}", 0)), std::nullopt);
  auto long_length = RawShard("{}", 0);
  long_length[0] = std::byte{3};
  EXPECT_EQ(rule(long_length), Rule::kContainer);
  // Leading spaces are JSON's (safetensors' own reader wants '{' first).
  EXPECT_EQ(rule(RawShard(" {}", 8)), std::nullopt);
  // Not an object, not JSON, not UTF-8, a dimension not a number,
  // duplicated.
  for (
      const std::string_view header :
      {"[]", "", "{", "{}x", R"({"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]})",
       "{\"\xFF\":1}", R"({"a":{"dtype":"F32","shape":[[1]],"data_offsets":[0,4]}})",
       R"({"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"a":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}})"}) {
    EXPECT_EQ(rule(RawShard(header, 8)), Rule::kContainer) << header;
  }
  // Entries: members beside the three are ignored; one missing, mistyped,
  // out of range.
  EXPECT_EQ(
      rule(RawShard(R"({"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4],"x":[[1]]}})", 4)),
      std::nullopt);
  for (const std::string_view entry :
       {R"({"dtype":"F32","shape":[1]})", R"({"dtype":1,"shape":[1],"data_offsets":[0,4]})",
        R"({"dtype":"F32","shape":1,"data_offsets":[0,4]})",
        R"({"dtype":"F32","shape":[-1],"data_offsets":[0,4]})",
        R"({"dtype":"F32","shape":[1.5],"data_offsets":[0,4]})",
        R"({"dtype":"F32","shape":[1],"data_offsets":[0,4,8]})",
        R"({"dtype":"F32","shape":[1],"data_offsets":[4,0]})",
        R"({"dtype":"F32","shape":[1],"data_offsets":[0,8]})",
        R"({"dtype":"F32","shape":[1],"data_offsets":[-4,0]})", R"([1])"}) {
    EXPECT_EQ(rule(RawShard(std::format(R"({{"a":{}}})", entry), 4)), Rule::kContainer) << entry;
  }
  for (const std::string_view dtype : {"F4", "F6_E2M3", "C64", "Q4_0", "f32"}) {
    EXPECT_EQ(
        rule(RawShard(
            std::format(R"({{"a":{{"dtype":"{}","shape":[2],"data_offsets":[0,1]}}}})", dtype), 1)),
        Rule::kUnsupported)
        << dtype;
  }
  EXPECT_EQ(rule(Shard({E("a", "F32", {1, 1, 1, 1, 1, 1, 1, 1, 1})})), Rule::kBounds);
  EXPECT_TRUE(
      ParseSafetensors(Shard({E("a", "F32", {1, 1, 1, 1, 1, 1, 1, 1})}), kConfig).has_value());
  EXPECT_EQ(rule(Shard({E("a", "F32", {2, 0})})), Rule::kUnsupported);
  EXPECT_EQ(
      rule(RawShard(
          R"({"a":{"dtype":"F32","shape":[4294967296,4294967296,2],"data_offsets":[0,4]}})", 4)),
      Rule::kBounds);
  // Names: control characters, too long.
  EXPECT_EQ(rule(Shard({E("a\\u0001", "F32", {1})})), Rule::kContainer);
  // C1's controls too (U+009B, which a terminal may take as CSI); other
  // non-ASCII text is a name.
  EXPECT_EQ(rule(Shard({E("a\\u009b", "F32", {1})})), Rule::kContainer);
  EXPECT_EQ(rule(Shard({E("a\u00a0\u00e9", "F32", {1})})), std::nullopt);
  EXPECT_EQ(rule(Shard({E(std::string(kMaxNameBytes + 1, 'n'), "F32", {1})})), Rule::kContainer);
  EXPECT_TRUE(ParseSafetensors(Shard({E(std::string(kMaxNameBytes, 'n'), "F32", {1})}), kConfig)
                  .has_value());
  // __metadata__ of any kind is not read.
  EXPECT_EQ(rule(Shard({E("a", "F32", {1})}, R"({"k":1})")), std::nullopt);
  EXPECT_EQ(rule(Shard({E("a", "F32", {1})}, R"([])")), std::nullopt);
  // Placement: gaps, a start past 0 and bytes left over describe every
  // tensor; an overlap or data past the end does not.
  const auto two = [&](std::string_view a, std::string_view b) {
    return ParseSafetensors(
        RawShard(std::format(R"({{"a":{{"dtype":"F32","shape":[2],"data_offsets":{}}},)"
                             R"("b":{{"dtype":"F32","shape":[2],"data_offsets":{}}}}})",
                             a, b),
                 24),
        kConfig);
  };
  struct Placed {
    std::string_view a;
    std::string_view b;
    std::int64_t gap;  // b's offset less a's
  };
  for (const auto& [a, b, gap] :
       {Placed{"[0,8]", "[8,16]", 8}, Placed{"[0,8]", "[12,20]", 12},
        Placed{"[4,12]", "[12,20]", 8}, Placed{"[16,24]", "[0,8]", -16}}) {
    const auto c = two(a, b);
    ASSERT_TRUE(c.has_value()) << a << " " << b << ": " << c.error().ToString();
    ASSERT_TRUE(c->FindTensor("a") != nullptr && c->FindTensor("b") != nullptr);
    EXPECT_EQ(static_cast<std::int64_t>(c->FindTensor("b")->extent.offset) -
                  static_cast<std::int64_t>(c->FindTensor("a")->extent.offset),
              gap)
        << a << " " << b;
  }
  EXPECT_EQ(RuleOf(two("[0,8]", "[4,12]")), Rule::kContainer);
  EXPECT_EQ(RuleOf(two("[0,8]", "[20,28]")), Rule::kContainer);
  auto extra = Shard({E("a", "F32", {2})});
  extra.push_back(std::byte{0});
  EXPECT_EQ(rule(extra), std::nullopt);
  auto missing = Shard({E("a", "F32", {2})});
  missing.pop_back();
  EXPECT_EQ(rule(missing), Rule::kContainer);
}

TEST(ImportSafetensorsTest, RefusesConfigs) {
  const auto shard = Shard({E("a", "F32", {1})});
  const auto rule = [&](std::string_view config) {
    return RuleOf(ParseSafetensors(shard, config));
  };
  EXPECT_EQ(rule("{"), Rule::kConfig);
  EXPECT_EQ(rule("[]"), Rule::kConfig);
  EXPECT_EQ(rule("{}"), Rule::kConfig);
  EXPECT_EQ(rule(R"({"model_type":1})"), Rule::kConfig);
  EXPECT_EQ(rule(R"({"model_type":""})"), Rule::kConfig);
  // A diffusers component's _class_name when there is no model_type.
  EXPECT_EQ(rule(R"({"_class_name":"AutoencoderKL"})"), std::nullopt);
  EXPECT_EQ(ParseSafetensors(shard, R"({"_class_name":"V","model_type":"m"})")->architecture, "m");
  EXPECT_EQ(rule(R"({"_class_name":1})"), Rule::kConfig);
  EXPECT_EQ(rule(R"({"model_type":1,"_class_name":"V"})"), Rule::kConfig);
  EXPECT_EQ(rule(R"({"model_type":"a\u0001"})"), Rule::kConfig);
  EXPECT_EQ(rule(std::format(R"({{"model_type":"{}"}})", std::string(201, 'a'))), Rule::kConfig);
  EXPECT_EQ(rule(R"({"model_type":"t","a":1,"a":2})"), Rule::kConfig);
  EXPECT_EQ(rule("{\"model_type\":\"t\",\"x\":\"\xFF\"}"), Rule::kConfig);
  // Two members flattening to one key.
  EXPECT_EQ(rule(R"({"model_type":"t","a.b":1,"a":{"b":2}})"), Rule::kConfig);
  // Nesting at most 32.
  const auto nested = [](std::size_t inner) {  // the root and `inner` objects in it
    std::string s = R"({"model_type":"t","x":)";
    for (std::size_t i = 1; i < inner; ++i) {
      s += R"({"x":)";
    }
    return s + "{}" + std::string(inner, '}');
  };
  EXPECT_EQ(rule(nested(kMaxConfigDepth)), Rule::kConfig);
  EXPECT_EQ(rule(nested(kMaxConfigDepth - 1)), std::nullopt);
  // Keys past their cap; a document past its cap.
  EXPECT_EQ(rule(std::format(R"({{"model_type":"t","{}":1}})", std::string(kMaxKeyBytes + 1, 'k'))),
            Rule::kBounds);
  EXPECT_EQ(
      rule(std::format(R"({{"model_type":"t","s":"{}"}})", std::string(kMaxConfigBytes, 's'))),
      Rule::kBounds);
}

// ---------------------------------------------------------------- quantization

// A ModelOpt NVFP4 linear of N rows and K inputs: codes, block scales, a
// tensor scale and an input scale.
std::vector<Entry> Nvfp4(std::string_view module, std::uint64_t n, std::uint64_t k,
                         bool tensor_scale = true, bool input_scale = true) {
  std::vector<Entry> out = {E(std::format("{}.weight", module), "U8", {n, k / 2}),
                            E(std::format("{}.weight_scale", module), "F8_E4M3", {n, k / 16})};
  if (tensor_scale) {
    out.push_back(E(std::format("{}.weight_scale_2", module), "F32", {}));
  }
  if (input_scale) {
    out.push_back(E(std::format("{}.input_scale", module), "F32", {}));
  }
  return out;
}

std::vector<Entry> Mxfp8(std::string_view module, std::uint64_t n, std::uint64_t k,
                         std::string scale_dtype = "U8") {
  return {E(std::format("{}.weight", module), "F8_E4M3", {n, k}),
          E(std::format("{}.weight_scale", module), std::move(scale_dtype), {n, k / 32})};
}

std::vector<Entry> Join(std::initializer_list<std::vector<Entry>> parts) {
  std::vector<Entry> out;
  for (const auto& p : parts) {
    out.insert(out.end(), p.begin(), p.end());
  }
  return out;
}

const Tensor& Named(const Checkpoint& c, std::string_view name) {
  const Tensor* t = c.FindTensor(name);
  EXPECT_NE(t, nullptr) << name;
  return t != nullptr ? *t : c.tensors.front();
}

constexpr std::string_view kNvfp4Hf =
    R"({"producer":{"name":"modelopt","version":"0.1"},"quantization":{"quant_algo":"NVFP4","group_size":16,"kv_cache_quant_algo":null,"exclude_modules":["lm_head","model.layers.*.mlp.gate"]}})";

TEST(ImportQuantizationTest, PairsModelOptNvfp4) {
  const auto shard = Shard(Join({Nvfp4("model.layers.0.mlp.down_proj", 4, 64),
                                 {E("lm_head.weight", "BF16", {8, 64}),
                                  E("model.layers.0.mlp.gate.weight", "BF16", {4, 64})}}));
  const auto c = ParseSafetensors(shard, kConfig, kNvfp4Hf);
  ASSERT_TRUE(c.has_value()) << c.error().ToString();
  EXPECT_EQ(c->quantization, "modelopt NVFP4");
  const format::TensorFormat want{.encoding = Encoding::kNvfp4,
                                  .params = {.tensor_scale = true},
                                  .layout = Layout::kSplit,
                                  .shape = {.ne = {64, 4, 1, 1}, .rank = 2}};
  const Tensor& codes = Named(*c, "model.layers.0.mlp.down_proj.weight");
  EXPECT_EQ(codes.format, want);
  EXPECT_EQ(codes.part, PartRole::kCodes);
  EXPECT_EQ(Named(*c, "model.layers.0.mlp.down_proj.weight_scale").format, want);
  EXPECT_EQ(Named(*c, "model.layers.0.mlp.down_proj.weight_scale").part, PartRole::kBlockScales);
  EXPECT_EQ(Named(*c, "model.layers.0.mlp.down_proj.weight_scale_2").part, PartRole::kTensorScale);
  EXPECT_EQ(Named(*c, "model.layers.0.mlp.down_proj.input_scale").format.encoding, Encoding::kF32);
  EXPECT_EQ(Named(*c, "lm_head.weight").format.encoding, Encoding::kBF16);
  // As the mapping sees them.
  const spec::Source s = SourceOf(*c);
  const auto found = std::ranges::find(s.tensors, "model.layers.0.mlp.down_proj.weight_scale",
                                       &spec::SourceTensor::name);
  ASSERT_NE(found, s.tensors.end());
  EXPECT_EQ(found->part, PartRole::kBlockScales);
  EXPECT_EQ(found->format, want);
  // W4A16 without a tensor scale.
  const auto w4a16 =
      ParseSafetensors(Shard(Nvfp4("m", 4, 64, false, false)), kConfig,
                       R"({"quantization":{"quant_algo":"W4A16_NVFP4","group_size":16}})");
  ASSERT_TRUE(w4a16.has_value()) << w4a16.error().ToString();
  EXPECT_FALSE(Named(*w4a16, "m.weight").format.params.tensor_scale);
}

TEST(ImportQuantizationTest, PairsMxfp8AndMixedPrecision) {
  const std::string config =
      R"({"model_type":"t","quantization_config":{"quant_method":"modelopt","quant_algo":"MXFP8","group_size":32}})";
  const auto mx = ParseSafetensors(Shard(Mxfp8("q_proj", 4, 64)), config);
  ASSERT_TRUE(mx.has_value()) << mx.error().ToString();
  EXPECT_EQ(Named(*mx, "q_proj.weight").format,
            (format::TensorFormat{.encoding = Encoding::kMxfp8,
                                  .layout = Layout::kSplit,
                                  .shape = {.ne = {64, 4, 1, 1}, .rank = 2}}));
  EXPECT_EQ(Named(*mx, "q_proj.weight_scale").part, PartRole::kBlockScales);
  EXPECT_TRUE(ParseSafetensors(Shard(Mxfp8("q_proj", 4, 64, "F8_E8M0")), config).has_value());

  // Per module, as Qwen3.8's checkpoint: the experts by their ancestor, a
  // linear by its own name, n-gram-like row shards by their dtypes, with
  // their table's one tensor scale; a norm beside the linear stays dense.
  const std::string mixed = R"({"quant_method":"modelopt","quant_algo":"MIXED_PRECISION",
    "quantized_layers":{"layers.0.mlp.experts":{"quant_algo":"NVFP4","group_size":16},
                        "layers.0.attn.q_proj":{"quant_algo":"MXFP8","group_size":32}},
    "config_groups":{},"ignore":[]})";
  const auto shard = Shard(Join(
      {Nvfp4("layers.0.mlp.experts.0.up_proj", 4, 32),
       Nvfp4("layers.0.mlp.experts.1.up_proj", 4, 32),
       Mxfp8("layers.0.attn.q_proj", 8, 32),
       Nvfp4("ngram.shard_0", 6, 160, false, false),
       Nvfp4("ngram.shard_1", 6, 160, false, false),
       {E("ngram.weight_scale_2", "F32", {1}), E("layers.0.attn.q_norm.weight", "BF16", {8})}}));
  const auto c = ParseSafetensors(
      shard, std::format(R"({{"model_type":"t","quantization_config":{}}})", mixed), mixed);
  ASSERT_TRUE(c.has_value()) << c.error().ToString();
  EXPECT_EQ(c->quantization, "modelopt MIXED_PRECISION");
  EXPECT_EQ(Named(*c, "layers.0.mlp.experts.1.up_proj.weight").format.encoding, Encoding::kNvfp4);
  EXPECT_EQ(Named(*c, "layers.0.attn.q_proj.weight").format.encoding, Encoding::kMxfp8);
  const Tensor& ngram = Named(*c, "ngram.shard_0.weight");
  EXPECT_EQ(ngram.format.encoding, Encoding::kNvfp4);
  EXPECT_EQ(ngram.format.shape.ne[0], 160U);
  EXPECT_TRUE(ngram.format.params.tensor_scale);
  EXPECT_EQ(Named(*c, "ngram.shard_1.weight_scale").format, ngram.format);
  EXPECT_EQ(Named(*c, "ngram.weight_scale_2").format, ngram.format);
  EXPECT_EQ(Named(*c, "ngram.weight_scale_2").part, PartRole::kTensorScale);
  EXPECT_EQ(Named(*c, "layers.0.attn.q_norm.weight").format.encoding, Encoding::kBF16);
  // A tensor scale stays the module's own: one beside block scales is not
  // inherited; one shared by weights of different formats is refused.
  const auto own = ParseSafetensors(
      Shard(Join({Nvfp4("t", 4, 32, true, false), Nvfp4("t.shard_0", 4, 32, false, false)})),
      kConfig, R"({"quantization":{"quant_algo":"NVFP4","group_size":16}})");
  ASSERT_TRUE(own.has_value()) << own.error().ToString();
  EXPECT_FALSE(Named(*own, "t.shard_0.weight").format.params.tensor_scale);
  EXPECT_EQ(Named(*own, "t.weight_scale_2").format, Named(*own, "t.weight").format);
  EXPECT_EQ(RuleOf(ParseSafetensors(Shard(Join({Nvfp4("u.shard_0", 4, 32, false, false),
                                                Nvfp4("u.shard_1", 6, 32, false, false),
                                                {E("u.weight_scale_2", "F32", {})}})),
                                    kConfig,
                                    R"({"quantization":{"quant_algo":"NVFP4","group_size":16}})")),
            Rule::kQuantization);
}

TEST(ImportQuantizationTest, RefusesPartsThatDoNotPair) {
  const auto rule = [](const std::vector<Entry>& entries, std::string_view hf = kNvfp4Hf) {
    return RuleOf(ParseSafetensors(Shard(entries), kConfig, hf));
  };
  // Scales without their weight are a tensor of their own.
  const auto orphan =
      ParseSafetensors(Shard({E("m.weight_scale", "F8_E4M3", {4, 4})}), kConfig, kNvfp4Hf);
  ASSERT_TRUE(orphan.has_value()) << orphan.error().ToString();
  EXPECT_EQ(Named(*orphan, "m.weight_scale").format.encoding, Encoding::kF8E4M3);
  EXPECT_EQ(Named(*orphan, "m.weight_scale").part, PartRole::kData);
  // Scales of the wrong shape, rows or dtype; a tensor scale of two values.
  EXPECT_EQ(rule({E("m.weight", "U8", {4, 32}), E("m.weight_scale", "F8_E4M3", {4, 5})}),
            Rule::kQuantization);
  EXPECT_EQ(rule({E("m.weight", "U8", {4, 32}), E("m.weight_scale", "F8_E4M3", {3, 4})}),
            Rule::kQuantization);
  EXPECT_EQ(rule({E("m.weight", "U8", {4, 32}), E("m.weight_scale", "F16", {4, 4})}),
            Rule::kQuantization);
  EXPECT_EQ(rule({E("m.weight", "BF16", {4, 64}), E("m.weight_scale", "F8_E4M3", {4, 4})}),
            Rule::kQuantization);
  EXPECT_EQ(rule({E("m.weight", "U8", {4, 32}), E("m.weight_scale", "F8_E4M3", {4, 4}),
                  E("m.weight_scale_2", "F32", {2})}),
            Rule::kQuantization);
  // Rows not whole blocks (K = 18), and three dimensions.
  EXPECT_EQ(rule({E("m.weight", "U8", {4, 9}), E("m.weight_scale", "F8_E4M3", {4, 1})}),
            Rule::kQuantization);
  EXPECT_EQ(rule({E("m.weight", "U8", {2, 4, 32}), E("m.weight_scale", "F8_E4M3", {2, 4, 4})}),
            Rule::kQuantization);
  // A scaled weight in an excluded module is described by its dtypes; one
  // whose dtypes match no convention cannot be.
  const auto excluded = ParseSafetensors(Shard(Nvfp4("lm_head", 4, 64)), kConfig, kNvfp4Hf);
  ASSERT_TRUE(excluded.has_value()) << excluded.error().ToString();
  EXPECT_EQ(Named(*excluded, "lm_head.weight").format.encoding, Encoding::kNvfp4);
  EXPECT_TRUE(Named(*excluded, "lm_head.weight").format.params.tensor_scale);
  EXPECT_EQ(rule({E("model.layers.3.mlp.gate.weight", "BF16", {4, 64}),
                  E("model.layers.3.mlp.gate.weight_scale", "F32", {4, 1})}),
            Rule::kQuantization);
  // MXFP8 with a tensor scale.
  const std::string mxfp8 = R"({"quantization":{"quant_algo":"MXFP8","group_size":32}})";
  auto with_s2 = Mxfp8("m", 4, 64);
  with_s2.push_back(E("m.weight_scale_2", "F32", {}));
  EXPECT_EQ(rule(with_s2, mxfp8), Rule::kQuantization);
  // Per module: a configured weight without scales is an unquantized
  // tensor; a pair matching no convention outside the configured modules,
  // and a configured pair of another scheme's dtypes, are refused.
  const std::string mixed =
      R"({"quant_method":"modelopt","quant_algo":"MIXED_PRECISION","quantized_layers":{"a":{"quant_algo":"NVFP4","group_size":16}}})";
  const auto plain = ParseSafetensors(
      Shard({E("a.weight", "BF16", {4, 64}), E("a.b.weight", "BF16", {4, 64})}), kConfig, mixed);
  ASSERT_TRUE(plain.has_value()) << plain.error().ToString();
  EXPECT_EQ(Named(*plain, "a.weight").format.encoding, Encoding::kBF16);
  EXPECT_EQ(Named(*plain, "a.b.weight").format.layout, Layout::kDense);
  EXPECT_EQ(rule(Mxfp8("a", 4, 64), mixed), Rule::kQuantization);
  EXPECT_EQ(rule({E("b.weight", "BF16", {4, 64}), E("b.weight_scale", "F32", {4, 1})}, mixed),
            Rule::kQuantization);
  EXPECT_EQ(rule({E("b.weight", "BF16", {4, 64})}, mixed), std::nullopt);
  // Exclusion patterns that backtrack on every scaled weight: matching
  // stops at the reader's budget rather than running for hours.
  std::string patterns;
  for (int i = 0; i < 256; ++i) {
    patterns += std::format(R"({}"*{}{}b")", i == 0 ? "" : ",", std::string(500, 'a'), i);
  }
  std::string literals;
  for (int i = 0; i < 3800; ++i) {
    literals += std::format(R"(,"model.layers.{}.mlp.gate")", i);
  }
  const std::string hostile = std::format(
      R"({{"quantization":{{"quant_algo":"NVFP4","group_size":16,"exclude_modules":[{}]}}}})",
      patterns);
  std::vector<Entry> pairs;
  for (int i = 0; i < 400; ++i) {
    for (Entry& e : Nvfp4(std::format("{}{}", std::string(480, 'a'), i), 1, 16, false, false)) {
      pairs.push_back(std::move(e));
    }
  }
  EXPECT_EQ(rule(pairs, hostile), Rule::kBounds);
  // Literal patterns are compared, not matched: thousands of them (as a
  // config listing every excluded module) stay within the budget.
  EXPECT_EQ(
      rule(pairs,
           std::format(
               R"({{"quantization":{{"quant_algo":"NVFP4","exclude_modules":["lm_head"{}]}}}})",
               literals)),
      std::nullopt);
  // A pattern longer than a key.
  EXPECT_EQ(
      rule(Nvfp4("m", 4, 64),
           std::format(R"({{"quantization":{{"quant_algo":"NVFP4","exclude_modules":["{}"]}}}})",
                       std::string(1025, 'x'))),
      Rule::kConfig);
}

TEST(ImportQuantizationTest, RecognizesAndRefusesMethods) {
  const auto shard = Shard({E("a.weight", "BF16", {4, 4})});
  const auto config_rule = [&](std::string_view qc) {
    return RuleOf(ParseSafetensors(
        shard, std::format(R"({{"model_type":"t","quantization_config":{}}})", qc)));
  };
  EXPECT_EQ(config_rule(R"({"quant_method":"compressed-tensors","format":"dense"})"), std::nullopt);
  for (
      const std::string_view qc : {
          R"({"quant_method":"compressed-tensors","format":"nvfp4-pack-quantized"})",
          R"({"quant_method":"compressed-tensors","format":"float-quantized"})",
          R"({"quant_method":"awq","bits":4,"group_size":128})",
          R"({"quant_method":"gptq","bits":4,"group_size":128})",
          R"({"quant_method":"fp8","weight_block_size":[128,128]})",
          R"({"quant_method":"bitsandbytes"})",
          R"({"quant_method":"modelopt","quant_algo":"FP8"})",
          R"({"quant_method":"modelopt","quant_algo":"MIXED_PRECISION","quantized_layers":{"a":{"quant_algo":"INT4_AWQ"}}})",
          R"({"group_size":64,"bits":4})",
      }) {
    EXPECT_EQ(config_rule(qc), Rule::kUnsupported) << qc;
  }
  for (
      const std::string_view qc : {
          R"({"quant_method":1})",
          R"({"quant_method":"modelopt"})",
          R"({"quant_method":"modelopt","quant_algo":"NVFP4","group_size":32})",
          R"({"quant_method":"modelopt","quant_algo":"MXFP8","group_size":16})",
          R"({"quant_method":"modelopt","quant_algo":"MIXED_PRECISION"})",
          R"({"quant_method":"modelopt","quant_algo":"MIXED_PRECISION","quantized_layers":{"a":{}}})",
          R"({"quant_method":"modelopt","quant_algo":"NVFP4","ignore":["a[0]"]})",
          R"({"quant_method":"modelopt","quant_algo":"NVFP4","ignore":"a"})",
          R"([])",
      }) {
    EXPECT_EQ(config_rule(qc), Rule::kConfig) << qc;
  }
  // MLX's own key, an object; a null one names nothing.
  EXPECT_EQ(RuleOf(ParseSafetensors(
                shard, R"({"model_type":"t","quantization":{"group_size":64,"bits":4}})")),
            Rule::kUnsupported);
  EXPECT_EQ(RuleOf(ParseSafetensors(shard, R"({"model_type":"t","quantization":null})")),
            std::nullopt);
  // config.json and hf_quant_config.json must agree.
  const std::string nvfp4 = R"({"quant_method":"modelopt","quant_algo":"NVFP4","group_size":16})";
  const std::string with = std::format(R"({{"model_type":"t","quantization_config":{}}})", nvfp4);
  EXPECT_TRUE(ParseSafetensors(shard, with, nvfp4).has_value());
  EXPECT_EQ(
      RuleOf(ParseSafetensors(
          shard, with, R"({"quant_method":"modelopt","quant_algo":"MXFP8","group_size":32})")),
      Rule::kConfig);
  EXPECT_EQ(RuleOf(ParseSafetensors(shard, with, "{")), Rule::kConfig);
}

// ---------------------------------------------------------------- directories

using ImportSafetensorsFilesTest = ScratchTest;

TEST_F(ImportSafetensorsFilesTest, ReadsShardedAndSingleCheckpoints) {
  Write(dir_ / "config.json", kConfig);
  const auto a = Shard({E("a", "F32", {2}), E("b", "F16", {3})});
  const auto b = Shard({E("c", "BF16", {4})});
  Write(dir_ / "model-00002.safetensors", b);
  Write(dir_ / "model-00001.safetensors", a);
  Write(dir_ / "model.safetensors.index.json", Index({{"a", "model-00001.safetensors"},
                                                      {"b", "model-00001.safetensors"},
                                                      {"c", "model-00002.safetensors"}}));
  Write(dir_ / "amax.safetensors", std::string_view("not read"));
  const auto c = ReadCheckpoint(dir_);
  ASSERT_TRUE(c.has_value()) << c.error().ToString();
  EXPECT_EQ(c->architecture, "test");
  ASSERT_EQ(c->files.size(), 4U);  // two shards, config.json, the index
  EXPECT_EQ(c->files[0].name, "model-00001.safetensors");
  EXPECT_EQ(c->files[1].name, "model-00002.safetensors");
  EXPECT_EQ(c->files[2].name, "config.json");
  EXPECT_EQ(c->files[3].name, "model.safetensors.index.json");
  ASSERT_EQ(c->tensors.size(), 3U);
  EXPECT_EQ(c->tensors[2].name, "c");
  EXPECT_EQ(c->tensors[2].extent.file, 1U);
  std::vector<std::byte> bytes(c->tensors[2].extent.bytes);
  ASSERT_TRUE(c->Read(c->tensors[2].extent, 0, bytes).has_value());
  EXPECT_TRUE(std::equal(bytes.begin(), bytes.end(),
                         b.begin() + static_cast<std::ptrdiff_t>(c->tensors[2].extent.offset)));

  // Both an index and model.safetensors, which the index does not name;
  // and an index that names it as a shard.
  Write(dir_ / "model.safetensors", a);
  EXPECT_EQ(RuleOf(ReadSafetensors(dir_)), Rule::kShards);
  Write(dir_ / "model.safetensors.index.json", Index({{"a", "model.safetensors"},
                                                      {"b", "model.safetensors"},
                                                      {"c", "model-00002.safetensors"}}));
  const auto named = ReadSafetensors(dir_);
  ASSERT_TRUE(named.has_value()) << named.error().ToString();
  ASSERT_EQ(named->files.size(), 4U);
  EXPECT_EQ(named->files[1].name, "model.safetensors");
  EXPECT_EQ(named->tensors.size(), 3U);
  // model.safetensors alone.
  std::filesystem::remove(dir_ / "model.safetensors.index.json");
  const auto single = ReadSafetensors(dir_);
  ASSERT_TRUE(single.has_value()) << single.error().ToString();
  EXPECT_EQ(single->files[0].name, "model.safetensors");
  EXPECT_EQ(single->tensors.size(), 2U);
  // Neither.
  std::filesystem::remove(dir_ / "model.safetensors");
  EXPECT_EQ(RuleOf(ReadSafetensors(dir_)), Rule::kFileType);
  // No config.json.
  Write(dir_ / "model.safetensors", a);
  std::filesystem::remove(dir_ / "config.json");
  EXPECT_EQ(RuleOf(ReadSafetensors(dir_)), Rule::kConfig);
}

TEST_F(ImportSafetensorsFilesTest, RefusesIndexesThatDisagree) {
  Write(dir_ / "config.json", kConfig);
  Write(dir_ / "s1.safetensors", Shard({E("a", "F32", {2}), E("b", "F16", {3})}));
  Write(dir_ / "s2.safetensors", Shard({E("c", "BF16", {4})}));
  const auto with = [&](const std::string& index) {
    Write(dir_ / "model.safetensors.index.json", index);
    return RuleOf(ReadSafetensors(dir_));
  };
  EXPECT_EQ(
      with(Index({{"a", "s1.safetensors"}, {"b", "s1.safetensors"}, {"c", "s2.safetensors"}})),
      std::nullopt);
  // A shard holding a tensor the index leaves out or places elsewhere.
  EXPECT_EQ(with(Index({{"a", "s1.safetensors"}, {"c", "s2.safetensors"}})), Rule::kShards);
  EXPECT_EQ(
      with(Index({{"a", "s1.safetensors"}, {"b", "s2.safetensors"}, {"c", "s2.safetensors"}})),
      Rule::kShards);
  // A tensor no shard holds; a shard that does not exist.
  EXPECT_EQ(with(Index({{"a", "s1.safetensors"},
                        {"b", "s1.safetensors"},
                        {"c", "s2.safetensors"},
                        {"d", "s2.safetensors"}})),
            Rule::kShards);
  EXPECT_EQ(with(Index({{"a", "s1.safetensors"},
                        {"b", "s1.safetensors"},
                        {"c", "s2.safetensors"},
                        {"d", "s3.safetensors"}})),
            Rule::kFileType);
  // Shard names that are not plain .safetensors file names.
  for (const std::string_view shard :
       {"../s1.safetensors", "sub/s1.safetensors", "..", "s1.bin", ".safetensors"}) {
    EXPECT_EQ(with(Index({{"a", std::string(shard)}})),
              shard == ".safetensors" ? Rule::kFileType : Rule::kConfig)
        << shard;
  }
  // Members other than weight_map are not read; not JSON, no weight_map.
  EXPECT_EQ(with(Index({{"a", "s1.safetensors"}, {"b", "s1.safetensors"}, {"c", "s2.safetensors"}},
                       R"(,"other":{"x":[1]})")),
            std::nullopt);
  EXPECT_EQ(
      with(
          R"({"metadata":1,"weight_map":{"a":"s1.safetensors","b":"s1.safetensors","c":"s2.safetensors"}})"),
      std::nullopt);
  EXPECT_EQ(with(R"({"metadata":{}})"), Rule::kConfig);
  EXPECT_EQ(with("{"), Rule::kConfig);
  EXPECT_EQ(with(R"({"weight_map":[]})"), Rule::kConfig);
}

TEST_F(ImportSafetensorsFilesTest, OpensDirectoriesThroughLinks) {
  Write(dir_ / "ckpt" / "config.json", kConfig);
  Write(dir_ / "blobs" / "weights", Shard({E("a", "F32", {2})}));
  std::filesystem::create_symlink("../blobs/weights", dir_ / "ckpt" / "model.safetensors");
  ASSERT_TRUE(ReadSafetensors(dir_ / "ckpt").has_value());
  // Modes others could use are no concern of the reader's.
  ::chmod((dir_ / "ckpt").c_str(), 0777);
  ::chmod((dir_ / "blobs" / "weights").c_str(), 0666);
  ASSERT_TRUE(ReadSafetensors(dir_ / "ckpt").has_value());
  ::chmod((dir_ / "ckpt").c_str(), 0755);
  ::chmod((dir_ / "blobs" / "weights").c_str(), 0644);
  // The checkpoint itself through a link.
  std::filesystem::create_directory_symlink(dir_ / "ckpt", dir_ / "alias");
  const auto c = ReadSafetensors(dir_ / "alias");
  ASSERT_TRUE(c.has_value()) << c.error().ToString();
  EXPECT_EQ(c->files[0].path, std::filesystem::canonical(dir_ / "ckpt") / "model.safetensors");
  EXPECT_EQ(RuleOf(ReadSafetensors(dir_ / "ckpt" / "config.json")), Rule::kFileType);
  EXPECT_EQ(RuleOf(ReadSafetensors(dir_ / "absent")), Rule::kFileType);
  // An optional document linking to nothing (a blob a cache never
  // fetched) is a broken checkpoint, not one without that document.
  std::filesystem::create_symlink("../blobs/never-fetched", dir_ / "ckpt" / "hf_quant_config.json");
  EXPECT_EQ(RuleOf(ReadSafetensors(dir_ / "ckpt")), Rule::kFileType);
}

}  // namespace
}  // namespace llmp::import
