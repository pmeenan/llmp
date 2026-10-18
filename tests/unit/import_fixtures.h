// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// What the importer's unit tests (import_test.cc, import_readers_test.cc)
// build checkpoints with: little-endian bytes, safetensors shards and
// indexes, a private scratch directory per test, and the checks every
// accepted checkpoint passes (consistent extents; random corruptions
// refused or consistent).

#ifndef LLMP_TESTS_UNIT_IMPORT_FIXTURES_H_
#define LLMP_TESTS_UNIT_IMPORT_FIXTURES_H_

#include <gtest/gtest.h>
#include <sys/stat.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <expected>
#include <filesystem>
#include <format>
#include <fstream>
#include <map>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include "expected_error.h"
#include "format/tensor_format.h"
#include "import/checkpoint.h"
#include "import/error.h"

namespace llmp::import::fixtures {

inline void Append(std::vector<std::byte>& out, std::uint64_t v, std::size_t size) {
  for (std::size_t i = 0; i < size; ++i) {
    out.push_back(static_cast<std::byte>((v >> (8U * i)) & 0xFFU));
  }
}

inline void AppendText(std::vector<std::byte>& out, std::string_view s) {
  for (const char c : s) {
    out.push_back(static_cast<std::byte>(c));
  }
}

inline std::vector<std::byte> Le(std::uint64_t v, std::size_t size) {
  std::vector<std::byte> out;
  Append(out, v, size);
  return out;
}

inline std::optional<Rule> RuleOf(const std::expected<Checkpoint, Error>& r) {
  return test_support::Failed(r, &Error::rule);
}

inline std::span<const std::byte> View(const std::vector<std::byte>& b) { return b; }

// What a reader accepts is consistent: every extent inside its file, the
// size rule's for its format and part, and overlapping no other.
inline void ExpectConsistent(const Checkpoint& c) {
  std::vector<Extent> extents;
  for (const Tensor& t : c.tensors) {
    extents.push_back(t.extent);
  }
  std::ranges::sort(extents, {}, [](const Extent& e) { return std::pair(e.file, e.offset); });
  for (std::size_t i = 1; i < extents.size(); ++i) {
    if (extents[i].file == extents[i - 1].file) {
      EXPECT_GE(extents[i].offset, extents[i - 1].offset + extents[i - 1].bytes);
    }
  }
  for (const Tensor& t : c.tensors) {
    ASSERT_LT(t.extent.file, c.files.size()) << t.name;
    const std::uint64_t size = c.files[t.extent.file].bytes;
    EXPECT_LE(t.extent.offset, size);
    EXPECT_LE(t.extent.bytes, size - std::min(size, t.extent.offset));
    const auto parts = format::PartsOf(t.format);
    ASSERT_TRUE(parts.has_value()) << t.name;
    const auto part = parts->Find(t.part);
    ASSERT_TRUE(part.has_value()) << t.name;
    EXPECT_EQ(part.value_or(format::PartExtent{}).bytes, t.extent.bytes) << t.name;
  }
}

// Bytes changed at random, deterministically: refused or consistent,
// never a crash, a read past the bytes or an assertion.
template <typename Read>
void Corrupt(const std::vector<std::byte>& valid, Read read) {
  std::mt19937_64 rng(20261009);  // NOLINT(bugprone-random-generator-seed): reproducible
  std::size_t accepted = 0;
  for (int round = 0; round < 20000; ++round) {
    std::vector<std::byte> bytes = valid;
    const auto changes = 1 + (rng() % 4);
    for (std::uint64_t i = 0; i < changes; ++i) {
      bytes[rng() % bytes.size()] = static_cast<std::byte>(rng() & 0xFFU);
    }
    if (rng() % 8 == 0) {
      bytes.resize(rng() % bytes.size());
    }
    const auto c = read(bytes);
    if (c) {
      ++accepted;
      ExpectConsistent(*c);
    }
  }
  testing::Test::RecordProperty("accepted", static_cast<int>(accepted));
}

class ScratchTest : public testing::Test {
 protected:
  void SetUp() override {
    // NOLINTNEXTLINE(concurrency-mt-unsafe): tests read, never set, the environment
    const char* scratch = std::getenv("LLMP_TEST_SCRATCH");
    std::filesystem::path base = scratch != nullptr ? std::filesystem::path(scratch)
                                                    : std::filesystem::temp_directory_path();
    const auto* info = testing::UnitTest::GetInstance()->current_test_info();
    dir_ = base / "import" / std::format("{}.{}", info->test_suite_name(), info->name());
    std::error_code error;
    std::filesystem::permissions(dir_,
                                 std::filesystem::perms::owner_all |
                                     std::filesystem::perms::group_read |
                                     std::filesystem::perms::group_exec,
                                 error);
    std::filesystem::remove_all(dir_, error);
    std::filesystem::create_directories(dir_, error);
    ASSERT_FALSE(error) << error.message();
    std::filesystem::permissions(dir_,
                                 std::filesystem::perms::owner_all |
                                     std::filesystem::perms::group_read |
                                     std::filesystem::perms::group_exec,
                                 error);
  }

  static void Write(const std::filesystem::path& path, std::span<const std::byte> bytes) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char*>(bytes.data()),
              static_cast<std::streamsize>(bytes.size()));
    ASSERT_TRUE(out.good()) << path;
    out.close();
    ::chmod(path.c_str(), 0644);
  }
  static void Write(const std::filesystem::path& path, std::string_view text) {
    Write(path, std::as_bytes(std::span(text.data(), text.size())));
  }

  std::filesystem::path dir_;
};

// ---------------------------------------------------------------- safetensors

struct Entry {
  std::string name;
  std::string dtype;
  std::vector<std::uint64_t> shape;
  std::uint64_t bytes = 0;  // the data's
};

inline std::uint64_t DtypeBytes(std::string_view dtype) {
  if (dtype == "F64" || dtype == "I64" || dtype == "U64") {
    return 8;
  }
  if (dtype == "F32" || dtype == "I32" || dtype == "U32") {
    return 4;
  }
  if (dtype == "F16" || dtype == "BF16" || dtype == "I16" || dtype == "U16") {
    return 2;
  }
  return 1;
}

inline Entry E(std::string name, std::string dtype, std::vector<std::uint64_t> shape) {
  std::uint64_t n = DtypeBytes(dtype);
  for (const std::uint64_t d : shape) {
    n *= d;
  }
  return {
      .name = std::move(name), .dtype = std::move(dtype), .shape = std::move(shape), .bytes = n};
}

// A complete small HF Qwen2 checkpoint for import/publication controls.
inline std::vector<Entry> TinyQwen2HfEntries() {
  return {E("model.embed_tokens.weight", "F16", {10, 8}),
          E("model.norm.weight", "F16", {8}),
          E("lm_head.weight", "F16", {10, 8}),
          E("model.layers.0.input_layernorm.weight", "F16", {8}),
          E("model.layers.0.post_attention_layernorm.weight", "F16", {8}),
          E("model.layers.0.self_attn.q_proj.weight", "F16", {8, 8}),
          E("model.layers.0.self_attn.k_proj.weight", "F16", {4, 8}),
          E("model.layers.0.self_attn.v_proj.weight", "F16", {4, 8}),
          E("model.layers.0.self_attn.o_proj.weight", "F16", {8, 8}),
          E("model.layers.0.self_attn.q_proj.bias", "F16", {8}),
          E("model.layers.0.self_attn.k_proj.bias", "F16", {4}),
          E("model.layers.0.self_attn.v_proj.bias", "F16", {4}),
          E("model.layers.0.mlp.gate_proj.weight", "F16", {16, 8}),
          E("model.layers.0.mlp.up_proj.weight", "F16", {16, 8}),
          E("model.layers.0.mlp.down_proj.weight", "F16", {8, 16}),
          E("unused.empty", "F16", {0})};
}
inline std::string TinyQwen2HfConfig(std::string_view dtype = "dtype") {
  return std::string(R"({"architectures":["Qwen2ForCausalLM"],"model_type":"qwen2",
      "hidden_size":8,"intermediate_size":16,"num_hidden_layers":1,
      "num_attention_heads":2,"num_key_value_heads":1,"head_dim":4,
      "max_position_embeddings":128,"rms_norm_eps":0.000001,"rope_theta":10000,
      "hidden_act":"silu","use_sliding_window":false,"sliding_window":null,"rope_scaling":null,
      "attention_dropout":0.0,"bos_token_id":1,"eos_token_id":2,"vocab_size":10,
      "tie_word_embeddings":false,"use_cache":true,"transformers_version":"5.17.0",
      "new_training_note":{"anything":true},")") +
         std::string(dtype) + R"(":"float16"})";
}

inline std::string ShapeText(const std::vector<std::uint64_t>& shape) {
  std::string s = "[";
  for (std::size_t i = 0; i < shape.size(); ++i) {
    s += std::format("{}{}", i == 0 ? "" : ",", shape[i]);
  }
  return s + "]";
}

// A shard: its header lists the entries in the given order with their data
// back to back in that order (unless `offsets` overrides an entry's).
inline std::vector<std::byte> Shard(const std::vector<Entry>& entries,
                                    std::string_view metadata = {},
                                    std::map<std::size_t, std::string> offsets = {},
                                    std::string_view padding = {}) {
  std::string header = "{";
  if (!metadata.empty()) {
    header += std::format(R"("__metadata__":{},)", metadata);
  }
  std::uint64_t at = 0;
  for (std::size_t i = 0; i < entries.size(); ++i) {
    const Entry& e = entries[i];
    const std::string range =
        offsets.contains(i) ? offsets[i] : std::format("[{},{}]", at, at + e.bytes);
    header += std::format(R"({}"{}":{{"dtype":"{}","shape":{},"data_offsets":{}}})",
                          i == 0 ? "" : ",", e.name, e.dtype, ShapeText(e.shape), range);
    at += e.bytes;
  }
  header += "}";
  header += padding;
  std::vector<std::byte> out = Le(header.size(), 8);
  AppendText(out, header);
  for (std::uint64_t b = 0; b < at; ++b) {
    out.push_back(static_cast<std::byte>(b & 0xFFU));
  }
  return out;
}

// A shard with the header text given whole, and `data` bytes after it.
inline std::vector<std::byte> RawShard(std::string_view header, std::uint64_t data) {
  std::vector<std::byte> out = Le(header.size(), 8);
  AppendText(out, header);
  out.resize(out.size() + data);
  return out;
}

inline std::string Index(const std::map<std::string, std::string>& map,
                         std::string_view extra = {}) {
  std::string text = R"({"metadata":{"total_size":0},"weight_map":{)";
  bool first = true;
  for (const auto& [tensor, shard] : map) {
    text += std::format(R"({}"{}":"{}")", first ? "" : ",", tensor, shard);
    first = false;
  }
  return text + "}" + std::string(extra) + "}";
}

}  // namespace llmp::import::fixtures

#endif  // LLMP_TESTS_UNIT_IMPORT_FIXTURES_H_
