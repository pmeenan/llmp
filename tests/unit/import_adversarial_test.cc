// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <map>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "artifact/artifact.h"
#include "base/check.h"
#include "base/gguf.h"
#include "base/json.h"
#include "base/sha256.h"
#include "format/encoding.h"
#include "format/layout.h"
#include "format/tensor_format.h"
#include "import/archive_input.h"
#include "import/checkpoint.h"
#include "import/error.h"
#include "import/files.h"
#include "import/indexed_parts.h"
#include "import/normalize.h"
#include "import/reader.h"
#include "import/readers/diffusers.h"
#include "import/readers/exl3.h"
#include "import/readers/gguf.h"
#include "import/readers/mlx.h"
#include "import/readers/safetensors.h"
#include "import/stream.h"
#include "import/writer.h"
#include "import_fixtures.h"
#include "spec/stored.h"

namespace llmp::import {
namespace {

namespace gg = base::gguf;
using fixtures::Append;
using fixtures::AppendText;
using fixtures::Le;
using fixtures::RuleOf;
using format::Encoding;
using format::Layout;

std::vector<std::byte> StringBytes(std::string_view s) {
  std::vector<std::byte> out;
  Append(out, s.size(), 8);
  AppendText(out, s);
  return out;
}

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

class Gguf {
 public:
  Gguf() { Str("general.architecture", "test"); }

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

constexpr NormalizationLimits kFunded{std::uint64_t{2} << 30U, std::uint64_t{1} << 45U};

void Ordered(Checkpoint& s) { std::ranges::sort(s.metadata, {}, &MetaEntry::key); }

void Value(Checkpoint& s, std::string key, MetaValue value) {
  const auto old = std::ranges::find(s.metadata, key, &MetaEntry::key);
  if (old == s.metadata.end()) {
    s.metadata.push_back({std::move(key), std::move(value)});
  } else {
    old->value = std::move(value);
  }
  Ordered(s);
}

void AddTensor(Checkpoint& s, std::string name, std::initializer_list<std::uint64_t> ne,
               Encoding encoding = Encoding::kF32) {
  const auto shape = format::ShapeOf(std::span(ne.begin(), ne.size()));
  base::Check(shape.has_value(), "test shape");
  const auto& info = format::Info(encoding);
  format::TensorFormat f{
      .encoding = encoding,
      .layout = info.quantized ? format::Layout::kGgmlBlocks : format::Layout::kDense,
      .shape = *shape};
  const auto parts = format::PartsOf(f);
  base::Check(parts.has_value(), "test format");
  const auto offset = (s.files[0].bytes + 31) / 32 * 32;
  s.tensors.push_back({.name = std::move(name),
                       .format = f,
                       .extent = {.offset = offset, .bytes = (*parts)[0].bytes},
                       .dtype = info.name,
                       .dims = std::vector<std::uint64_t>(ne)});
  s.files[0].bytes = offset + (*parts)[0].bytes;
}

Checkpoint TinyGemma2() {
  Checkpoint s{
      .container = "gguf", .architecture = "gemma2", .files = {{.name = "memory", .bytes = 0}}};
  Value(s, "general.architecture", std::string("gemma2"));
  for (const auto& [key, value] :
       std::array<std::pair<std::string_view, std::int64_t>, 10>{{{"block_count", 2},
                                                                  {"context_length", 128},
                                                                  {"embedding_length", 8},
                                                                  {"feed_forward_length", 16},
                                                                  {"attention.head_count", 2},
                                                                  {"attention.head_count_kv", 1},
                                                                  {"attention.key_length", 4},
                                                                  {"attention.value_length", 4},
                                                                  {"attention.sliding_window", 64},
                                                                  {"rope.dimension_count", 4}}}) {
    Value(s, "gemma2." + std::string(key), value);
  }
  Value(s, "gemma2.attention.layer_norm_rms_epsilon", 1e-6);
  Value(s, "gemma2.attn_logit_softcapping", 50.0);
  Value(s, "gemma2.final_logit_softcapping", 30.0);

  AddTensor(s, "token_embd.weight", {8, 10});
  AddTensor(s, "output_norm.weight", {8});
  for (int layer = 0; layer < 2; ++layer) {
    const auto name = [&](std::string_view key) {
      return std::format("blk.{}.{}.weight", layer, key);
    };
    for (const auto key : {"attn_norm", "post_attention_norm", "ffn_norm", "post_ffw_norm"}) {
      AddTensor(s, name(key), {8});
    }
    AddTensor(s, name("attn_q"), {8, 8});
    AddTensor(s, name("attn_k"), {8, 4});
    AddTensor(s, name("attn_v"), {8, 4});
    AddTensor(s, name("attn_output"), {8, 8});
    AddTensor(s, name("ffn_gate"), {8, 16});
    AddTensor(s, name("ffn_up"), {8, 16});
    AddTensor(s, name("ffn_down"), {16, 8});
  }
  return s;
}

std::vector<std::byte> SerializeGguf(const Checkpoint& s) {
  std::vector<std::byte> file;
  const auto add = [&](std::uint64_t n, std::size_t bytes) { fixtures::Append(file, n, bytes); };
  const auto string = [&](std::string_view text) {
    add(text.size(), 8);
    fixtures::AppendText(file, text);
  };
  add(0x46554747, 4);
  add(3, 4);
  add(s.tensors.size(), 8);
  add(s.metadata.size(), 8);
  for (const auto& e : s.metadata) {
    string(e.key);
    if (const auto* integer = std::get_if<std::int64_t>(&e.value)) {
      add(11, 4);
      add(std::bit_cast<std::uint64_t>(*integer), 8);
    } else if (const auto* real = std::get_if<double>(&e.value)) {
      add(12, 4);
      add(std::bit_cast<std::uint64_t>(*real), 8);
    } else if (const auto* list = std::get_if<std::vector<std::int64_t>>(&e.value)) {
      add(9, 4);
      add(11, 4);
      add(list->size(), 8);
      for (const auto item : *list) add(std::bit_cast<std::uint64_t>(item), 8);
    } else {
      add(8, 4);
      string(std::get<std::string>(e.value));
    }
  }
  for (const auto& t : s.tensors) {
    string(t.name);
    add(t.dims.size(), 4);
    for (auto d : t.dims) add(d, 8);
    add(format::Info(t.format.encoding).ggml->type, 4);
    add(t.extent.offset, 8);
  }
  file.resize((file.size() + 31) / 32 * 32);
  const auto begin = file.size();
  base::Check(s.files[0].bytes < (1U << 20U), "test payload ceiling");
  file.resize(begin + static_cast<std::size_t>(s.files[0].bytes));
  for (std::size_t i = begin; i < file.size(); ++i) file[i] = static_cast<std::byte>(i % 251);
  return file;
}

class ImportAdversarialTest : public fixtures::ScratchTest {
 protected:
  std::expected<NormalizedPlan, Error> MakePlan() {
    Write(dir_ / "source/model.gguf", SerializeGguf(TinyGemma2()));
    auto input = ReadGguf(dir_ / "source/model.gguf");
    if (!input) return std::unexpected(input.error());
    return Normalize(std::move(*input), kFunded);
  }

  WriteOptions Options(std::string_view store = "store") {
    return {.store = dir_ / store, .build = "test", .metadata_bytes = std::uint64_t{1} << 30U};
  }
};

// ============================================================================
// 1. Arithmetic Overflow Bounds
// ============================================================================

TEST_F(ImportAdversarialTest, SafetensorsShapeProduct64BitOverflowRefused) {
  // Dimension product: 2^40 * 2^25 overflows uint64_t.
  const std::string header =
      R"({"tensor":{"dtype":"F32","shape":[1099511627776,33554432],"data_offsets":[0,16]}})";
  auto shard = fixtures::RawShard(header, 16);
  auto parsed = ParseSafetensors(shard, R"({"model_type":"gemma2"})");
  ASSERT_FALSE(parsed.has_value());
  EXPECT_EQ(parsed.error().rule, Rule::kBounds);
}

TEST_F(ImportAdversarialTest, SafetensorsInvertedAndMismatchOffsetsRefused) {
  // Inverted offsets: end < begin.
  {
    const std::string header = R"({"t":{"dtype":"F32","shape":[1],"data_offsets":[100,50]}})";
    auto shard = fixtures::RawShard(header, 100);
    auto parsed = ParseSafetensors(shard, R"({"model_type":"gemma2"})");
    ASSERT_FALSE(parsed.has_value());
    EXPECT_EQ(parsed.error().rule, Rule::kContainer);
  }
  // Span size does not match shape byte requirement.
  {
    const std::string header =
        R"({"t":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}})";  // requires 8 bytes
    auto shard = fixtures::RawShard(header, 8);
    auto parsed = ParseSafetensors(shard, R"({"model_type":"gemma2"})");
    ASSERT_FALSE(parsed.has_value());
    EXPECT_EQ(parsed.error().rule, Rule::kContainer);
  }
}

TEST_F(ImportAdversarialTest, GgufTensorElementsOverflowRefused) {
  // 2^40 * 2^25 elements > 2^63 - 1 limit.
  auto file =
      Gguf().Tensor("overflow", 0, {std::uint64_t{1} << 40U, std::uint64_t{1} << 25U}).Build();
  auto parsed = ParseGguf(file);
  ASSERT_FALSE(parsed.has_value());
  EXPECT_EQ(parsed.error().rule, Rule::kContainer);
}

TEST_F(ImportAdversarialTest, GgufTensorOffsets64BitOverflowRefused) {
  // Offset + size overflows 64-bit bounds.
  auto file = Gguf()
                  .Tensor("overflow_offset", 0, {10}, std::numeric_limits<std::uint64_t>::max())
                  .Build(false);
  auto parsed = ParseGguf(file);
  ASSERT_FALSE(parsed.has_value());
  EXPECT_EQ(parsed.error().rule, Rule::kBounds);
}

TEST_F(ImportAdversarialTest, NormalizationUnderfundedLimitsRefused) {
  Write(dir_ / "source/model.gguf", SerializeGguf(TinyGemma2()));
  auto input = ReadGguf(dir_ / "source/model.gguf");
  ASSERT_TRUE(input.has_value());

  auto bound = NormalizationBound(*input);
  ASSERT_TRUE(bound.has_value());

  // Underfunded metadata bytes:
  {
    NormalizationLimits underfunded{bound->metadata_bytes - 1, bound->work_units};
    auto plan = Normalize(Checkpoint(*input), underfunded);
    ASSERT_FALSE(plan.has_value());
    EXPECT_EQ(plan.error().rule, Rule::kBounds);
  }
  // Underfunded work units:
  {
    NormalizationLimits underfunded{bound->metadata_bytes, bound->work_units - 1};
    auto plan = Normalize(Checkpoint(*input), underfunded);
    ASSERT_FALSE(plan.has_value());
    EXPECT_EQ(plan.error().rule, Rule::kBounds);
  }
}

TEST_F(ImportAdversarialTest, IndexedPartPlanPrepaymentDeficitRefused) {
  Checkpoint source{.container = "safetensors",
                    .architecture = "test",
                    .files = {{.name = "shard0", .bytes = 100}}};
  source.tensors.push_back({.name = "w", .format = {.shape = {.ne = {10, 1, 1, 1}, .rank = 1}}});

  TensorAssembly assembly{.name = "out", .parts = {{.sources = {0}}}};
  const std::array<TensorAssembly, 1> outputs{assembly};

  auto bound = IndexedPartPlan::Bound(source, outputs);
  ASSERT_TRUE(bound.has_value());

  // Underpaid metadata:
  auto underpaid = *bound;
  underpaid.metadata_bytes -= 1;
  const std::array<SourceDisposition, 1> disps{
      SourceDisposition{.kind = SourceDispositionKind::kRouted, .rule = 0}};
  auto plan = IndexedPartPlan::Make(Checkpoint(source), outputs, disps, underpaid);
  ASSERT_FALSE(plan.has_value());
  EXPECT_EQ(plan.error().rule, Rule::kBounds);
}

// ============================================================================
// 2. Path Traversal & Hostile File Types
// ============================================================================

TEST_F(ImportAdversarialTest, SafetensorsIndexDirectoryTraversalRefused) {
  Write(dir_ / "config.json", R"({"model_type":"gemma2"})");

  for (const std::string_view hostile :
       {"../escaped.safetensors", "/etc/passwd.safetensors", "sub/shard.safetensors",
        "shard\\back.safetensors", "shard\x01.safetensors"}) {
    Write(dir_ / "model.safetensors.index.json",
          std::format(R"({{"weight_map":{{"tensor":"{}"}}}})", hostile));
    auto result = ReadSafetensors(dir_);
    ASSERT_FALSE(result.has_value()) << hostile;
    EXPECT_EQ(result.error().rule, Rule::kConfig) << hostile;
  }
}

TEST_F(ImportAdversarialTest, ArchiveManifestDirectoryTraversalRefused) {
  const auto root = dir_ / "archive";
  const auto revision = root / "hf/owner/model@aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
  Write(root / "llmp-archive-root.json", R"({"format":"llmp-archive-root","format_version":1})");

  for (const std::string_view hostile :
       {"../escaped", "/etc/shadow", "foo/../../escaped", "dir\x01/file", "bad:name", "trail."}) {
    std::string manifest = std::format(
        R"({{"format":"llmp-archive-checkpoint","format_version":1,"origin":"hf","repo":"owner/model","commit":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","hub":{{}},"files":[{{"path":"{}","size":10,"git_oid":null,"lfs":true,"sha256":"0000000000000000000000000000000000000000000000000000000000000000","present":true}}],"variants":[{{"id":"model","files":["{}"]}}]}})",
        hostile, hostile);
    Write(revision / "llmp-archive.json", manifest);
    auto found = FindArchiveInput(revision);
    ASSERT_FALSE(found.has_value()) << hostile;
    EXPECT_EQ(found.error().rule, Rule::kConfig) << hostile;
  }
}

TEST_F(ImportAdversarialTest, NonRegularFifoFileRefusedImmediately) {
  const auto fifo = dir_ / "named_pipe.gguf";
  std::filesystem::remove(fifo);
  ASSERT_EQ(::mkfifo(fifo.c_str(), 0600), 0);

  auto opened = OpenCheckpointFile(fifo, 0);
  ASSERT_FALSE(opened.has_value());
  EXPECT_EQ(opened.error().rule, Rule::kFileType);

  auto read_result = ReadGguf(fifo);
  ASSERT_FALSE(read_result.has_value());
  EXPECT_EQ(read_result.error().rule, Rule::kFileType);

  std::filesystem::remove(fifo);
}

TEST_F(ImportAdversarialTest, StagingDirectorySymlinkRefused) {
  auto plan = MakePlan();
  ASSERT_TRUE(plan.has_value());

  const auto store = dir_ / "store";
  std::filesystem::create_directories(store);
  const auto target = dir_ / "symlink_target";
  std::filesystem::create_directories(target);

  std::filesystem::create_directory_symlink(target, store / ".staging");

  auto written = WriteArtifact(*plan, Options("store"));
  ASSERT_FALSE(written.has_value());
  EXPECT_EQ(written.error().reason, "artifact staging must not be a symlink");
}

// ============================================================================
// 3. TOCTOU & Mid-Stream Mutation
// ============================================================================

TEST_F(ImportAdversarialTest, HeaderPrefixMutatedBeforeStreamingRefused) {
  auto plan = MakePlan();
  ASSERT_TRUE(plan.has_value());

  const auto& file = plan->source().files.front();
  auto bytes = SerializeGguf(TinyGemma2());
  bytes[16] ^= std::byte{0xFF};
  Write(file.path, bytes);

  auto result = WriteArtifact(*plan, Options());
  ASSERT_FALSE(result.has_value());
  EXPECT_EQ(result.error().rule, Rule::kIo);
  EXPECT_EQ(result.error().reason, "parsed source bytes changed before streaming");

  // Verify staging cleaned up and no artifact published.
  EXPECT_FALSE(std::filesystem::exists(Options().store / "manifest.json"));
  for (const auto& entry : std::filesystem::directory_iterator(Options().store)) {
    EXPECT_EQ(entry.path().filename(), ".staging");
  }
}

TEST_F(ImportAdversarialTest, FileSizeChangedBeforeStreamingRefused) {
  // Truncated file before streaming:
  {
    auto plan = MakePlan();
    ASSERT_TRUE(plan.has_value());
    const auto& file = plan->source().files.front();
    std::filesystem::resize_file(file.path, file.bytes / 2);

    auto result = WriteArtifact(*plan, Options("store_trunc"));
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().rule, Rule::kIo);
    EXPECT_EQ(result.error().reason, "a source changed identity or size after parsing");

    // Verify zero published artifacts.
    for (const auto& entry : std::filesystem::directory_iterator(Options("store_trunc").store)) {
      EXPECT_EQ(entry.path().filename(), ".staging");
    }
  }
  // Grown file before streaming:
  {
    auto plan = MakePlan();
    ASSERT_TRUE(plan.has_value());
    const auto& file = plan->source().files.front();
    std::filesystem::resize_file(file.path, file.bytes + 1024);

    auto result = WriteArtifact(*plan, Options("store_grow"));
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().rule, Rule::kIo);
    EXPECT_EQ(result.error().reason, "a source changed identity or size after parsing");

    for (const auto& entry : std::filesystem::directory_iterator(Options("store_grow").store)) {
      EXPECT_EQ(entry.path().filename(), ".staging");
    }
  }
}

TEST_F(ImportAdversarialTest, FileReplacedWithNewInodeBeforeStreamingRefused) {
  auto plan = MakePlan();
  ASSERT_TRUE(plan.has_value());

  const auto& file = plan->source().files.front();
  const auto content = SerializeGguf(TinyGemma2());
  const auto old_path = file.path.string() + ".replaced";
  std::filesystem::rename(file.path, old_path);
  Write(file.path, content);  // New file creation gives a new inode while old file holds old inode.
  std::filesystem::remove(old_path);

  auto result = WriteArtifact(*plan, Options());
  ASSERT_FALSE(result.has_value());
  EXPECT_EQ(result.error().rule, Rule::kIo);
  EXPECT_EQ(result.error().reason, "a source changed identity or size after parsing");
}

TEST_F(ImportAdversarialTest, ConcurrentMutationDuringStreamingDetected) {
  const std::string payload((2U << 20U) + 64, 'z');
  const auto path = dir_ / "stream_payload.bin";
  Write(path, payload);

  auto opened = OpenCheckpointFile(path, 0);
  ASSERT_TRUE(opened.has_value());

  SourceFile source{
      .name = "stream_payload.bin",
      .path = opened->path,
      .bytes = opened->bytes,
      .device = opened->device,
      .inode = opened->inode,
      .parsed = {16, base::Sha256().Update(std::string_view(payload).substr(0, 16)).Finish()}};

  // Sink appends a byte to the file during streaming.
  auto digest = StreamSource(source, 0, [&](auto at, auto) -> std::expected<void, Error> {
    if (at == 0) {
      std::ofstream mutate(source.path, std::ios::binary | std::ios::app);
      mutate << '!';
    }
    return {};
  });

  ASSERT_FALSE(digest.has_value());
  EXPECT_EQ(digest.error().rule, Rule::kIo);
  EXPECT_EQ(digest.error().reason, "a source changed during streaming");
}

// ============================================================================
// 4. Overlapping Extents, Duplicate Names/Keys, Unmapped Parts
// ============================================================================

TEST_F(ImportAdversarialTest, SafetensorsOverlappingExtentsRefused) {
  // Tensor a spans [0, 64], tensor b spans [32, 96] -> overlap!
  const std::string header =
      R"({"a":{"dtype":"U8","shape":[64],"data_offsets":[0,64]},"b":{"dtype":"U8","shape":[64],"data_offsets":[32,96]}})";
  auto shard = fixtures::RawShard(header, 96);
  auto parsed = ParseSafetensors(shard, R"({"model_type":"gemma2"})");
  ASSERT_FALSE(parsed.has_value());
  EXPECT_EQ(parsed.error().rule, Rule::kContainer);
  EXPECT_EQ(parsed.error().reason, "two tensors' data overlap");
}

TEST_F(ImportAdversarialTest, GgufOverlappingExtentsRefused) {
  // GGUF tensor a and b placed at the same data offset.
  auto file = Gguf().Tensor("t1", 0, {8}, 0).Tensor("t2", 0, {8}, 0).Build(true, 64);
  auto parsed = ParseGguf(file);
  ASSERT_FALSE(parsed.has_value());
  EXPECT_EQ(parsed.error().rule, Rule::kContainer);
  EXPECT_EQ(parsed.error().reason, "two tensors' data overlap");
}

TEST_F(ImportAdversarialTest, SafetensorsDuplicateTensorNamesRefused) {
  const std::string header =
      R"({"t":{"dtype":"U8","shape":[16],"data_offsets":[0,16]},"t":{"dtype":"U8","shape":[16],"data_offsets":[16,32]}})";
  auto shard = fixtures::RawShard(header, 32);
  auto parsed = ParseSafetensors(shard, R"({"model_type":"gemma2"})");
  ASSERT_FALSE(parsed.has_value());
  EXPECT_EQ(parsed.error().rule, Rule::kContainer);
}

TEST_F(ImportAdversarialTest, GgufDuplicateTensorNamesRefused) {
  auto file = Gguf().Tensor("dup", 0, {8}).Tensor("dup", 0, {8}).Build(true);
  auto parsed = ParseGguf(file);
  ASSERT_FALSE(parsed.has_value());
  EXPECT_EQ(parsed.error().rule, Rule::kContainer);
}

TEST_F(ImportAdversarialTest, GgufDuplicateMetadataKeysRefused) {
  auto file = Gguf().U32("custom.key", 1).U32("custom.key", 2).Build(true);
  auto parsed = ParseGguf(file);
  ASSERT_FALSE(parsed.has_value());
  EXPECT_EQ(parsed.error().rule, Rule::kContainer);
}

TEST_F(ImportAdversarialTest, SafetensorsConfigDuplicateKeysRefused) {
  const std::string config = R"({"model_type":"gemma2","model_type":"gemma2"})";
  Write(dir_ / "config.json", config);
  Write(dir_ / "model.safetensors", fixtures::Shard({fixtures::E("w", "F32", {1})}));
  auto result = ReadSafetensors(dir_);
  ASSERT_FALSE(result.has_value());
  EXPECT_EQ(result.error().rule, Rule::kConfig);
}

TEST_F(ImportAdversarialTest, IndexedPartsUnmappedRoutedSourceRefused) {
  Checkpoint source{.container = "safetensors",
                    .architecture = "test",
                    .files = {{.name = "shard0", .bytes = 100}}};
  source.tensors.push_back({.name = "w1",
                            .format = {.shape = {.ne = {10, 1, 1, 1}, .rank = 1}},
                            .extent = {.file = 0, .offset = 0, .bytes = 40}});
  source.tensors.push_back({.name = "w2",
                            .format = {.shape = {.ne = {10, 1, 1, 1}, .rank = 1}},
                            .extent = {.file = 0, .offset = 40, .bytes = 40}});

  // Only source 0 is assembled into output; source 1 is left unmapped.
  TensorAssembly assembly{.name = "out",
                          .format = {.shape = {.ne = {10, 1, 1, 1}, .rank = 1}},
                          .parts = {{.sources = {0}}}};
  const std::array<TensorAssembly, 1> outputs{assembly};

  auto bound = IndexedPartPlan::Bound(source, outputs);
  ASSERT_TRUE(bound.has_value());

  // Both marked kRouted, but w2 was never used.
  const std::array<SourceDisposition, 2> disps{
      SourceDisposition{.kind = SourceDispositionKind::kRouted, .rule = 0},
      SourceDisposition{.kind = SourceDispositionKind::kRouted, .rule = 0}};

  auto plan = IndexedPartPlan::Make(std::move(source), outputs, disps, *bound);
  ASSERT_FALSE(plan.has_value());
  EXPECT_EQ(plan.error().rule, Rule::kContainer);
  EXPECT_EQ(plan.error().reason, "indexed source disposition is incomplete");
}

TEST_F(ImportAdversarialTest, ControlCharactersInNamesRefused) {
  for (const std::string_view bad_name :
       {"tensor\nname", "tensor\tname", "tensor\x7F", "tensor\x1B"}) {
    auto file = Gguf().Tensor(std::string(bad_name), 0, {4}).Build(true);
    auto parsed = ParseGguf(file);
    ASSERT_FALSE(parsed.has_value()) << bad_name;
    EXPECT_EQ(parsed.error().rule, Rule::kContainer) << bad_name;
  }
}

// ============================================================================
// 5. Damaged Existing Artifact Refused & Intact Reused
// ============================================================================

TEST_F(ImportAdversarialTest, DamagedExistingArtifactMissingManifestRefused) {
  auto plan = MakePlan();
  ASSERT_TRUE(plan.has_value());

  auto result = WriteArtifact(*plan, Options());
  ASSERT_TRUE(result.has_value());

  // Corrupt by removing manifest.json.
  std::filesystem::remove(result->path / "manifest.json");

  auto again = WriteArtifact(*plan, Options());
  ASSERT_FALSE(again.has_value());
  EXPECT_EQ(again.error().reason, "the existing artifact is damaged");

  // Verify staging cleaned up.
  for (const auto& entry : std::filesystem::directory_iterator(Options().store / ".staging")) {
    EXPECT_EQ(entry.path().filename(), "native-import.lock");
  }
}

TEST_F(ImportAdversarialTest, DamagedExistingArtifactCorruptedManifestRefused) {
  auto plan = MakePlan();
  ASSERT_TRUE(plan.has_value());

  auto result = WriteArtifact(*plan, Options());
  ASSERT_TRUE(result.has_value());

  // Corrupt manifest.json with invalid JSON.
  Write(result->path / "manifest.json", "NOT_JSON");

  auto again = WriteArtifact(*plan, Options());
  ASSERT_FALSE(again.has_value());
  EXPECT_EQ(again.error().reason, "the existing artifact is damaged");
}

TEST_F(ImportAdversarialTest, DamagedExistingArtifactMismatchedIdRefused) {
  auto plan = MakePlan();
  ASSERT_TRUE(plan.has_value());

  auto result = WriteArtifact(*plan, Options());
  ASSERT_TRUE(result.has_value());

  // Overwrite manifest with a different valid manifest (which changes its ID).
  const std::string alt_manifest =
      R"({"converter":{"build":"test","name":"llmp-import","options_sha256":"0000000000000000000000000000000000000000000000000000000000000000","version":1},"files":[],"groups":[],"layout":{"chunk_bytes":2097152,"format":"llmp-artifact","format_version":1,"member_alignment":4096,"profile":"spark-v0","profile_version":0},"model":{"architecture":"other","container":"gguf","expert_count":0},"source":[],"transformations":[]})" +
      std::string("\n");
  Write(result->path / "manifest.json", alt_manifest);

  auto again = WriteArtifact(*plan, Options());
  ASSERT_FALSE(again.has_value());
  EXPECT_EQ(again.error().reason, "the existing artifact is damaged");
}

TEST_F(ImportAdversarialTest, DamagedExistingArtifactCorruptedShardRefused) {
  auto plan = MakePlan();
  ASSERT_TRUE(plan.has_value());

  auto result = WriteArtifact(*plan, Options());
  ASSERT_TRUE(result.has_value());

  // Corrupt 1 byte in shard data.
  Write(result->path / "data/00000.safetensors", "corrupted_shard_data");

  auto again = WriteArtifact(*plan, Options());
  ASSERT_FALSE(again.has_value());
  EXPECT_EQ(again.error().reason, "the existing artifact is damaged");
}

TEST_F(ImportAdversarialTest, DamagedExistingArtifactMissingSpecRefused) {
  auto plan = MakePlan();
  ASSERT_TRUE(plan.has_value());

  auto result = WriteArtifact(*plan, Options());
  ASSERT_TRUE(result.has_value());

  // Remove spec.json.
  std::filesystem::remove(result->path / "spec.json");

  auto again = WriteArtifact(*plan, Options());
  ASSERT_FALSE(again.has_value());
  EXPECT_EQ(again.error().reason, "the existing artifact is damaged");
}

TEST_F(ImportAdversarialTest, IntactExistingArtifactReusedSuccessfully) {
  auto plan = MakePlan();
  ASSERT_TRUE(plan.has_value());

  auto first = WriteArtifact(*plan, Options());
  ASSERT_TRUE(first.has_value());
  EXPECT_FALSE(first->reused);

  auto again = WriteArtifact(*plan, Options());
  ASSERT_TRUE(again.has_value());
  EXPECT_TRUE(again->reused);
  EXPECT_EQ(first->id, again->id);
}

// ============================================================================
// 6. Archive Manifest Hash Mismatches
// ============================================================================

TEST_F(ImportAdversarialTest, ArchiveManifestHashMismatchRefused) {
  auto plan = MakePlan();
  ASSERT_TRUE(plan.has_value());

  // Provide an expected hash that differs from the actual file hash.
  base::Sha256Digest wrong_hash{};
  wrong_hash[0] = 0xAA;
  std::array<std::optional<base::Sha256Digest>, 1> expected{wrong_hash};

  auto options = Options();
  options.expected = expected;

  auto result = WriteArtifact(*plan, options);
  ASSERT_FALSE(result.has_value());
  EXPECT_EQ(result.error().rule, Rule::kIo);
  EXPECT_EQ(result.error().reason, "a source hash differs from its archive manifest");

  // Zero published artifacts.
  for (const auto& entry : std::filesystem::directory_iterator(options.store)) {
    EXPECT_EQ(entry.path().filename(), ".staging");
  }
}

// ============================================================================
// 7. Ambiguous or inconsistent checkpoints the readers once accepted
// ============================================================================

using fixtures::E;
using fixtures::Entry;
using fixtures::Shard;

std::vector<Entry> Joined(std::vector<Entry> a, const std::vector<Entry>& b) {
  a.insert(a.end(), b.begin(), b.end());
  return a;
}

constexpr std::string_view kExl3NoCodebook =
    R"({"model_type":"qwen2","quantization_config":{"quant_method":"exl3","bits":4.0}})";

// An EXL3 linear of 32 inputs and 16 outputs at 4 bits.
std::vector<Entry> Exl3Linear() {
  return {E("w.trellis", "I16", {2, 1, 64}), E("w.suh", "F16", {32}), E("w.svh", "F16", {16})};
}

TEST_F(ImportAdversarialTest, Exl3EmptyPartIsNotReadAsAbsent) {
  // Without a marker the linear is 3inst; an empty marker must not make it so.
  ASSERT_TRUE(ParseExl3(Shard(Exl3Linear()), kExl3NoCodebook));
  for (const auto* marker : {"w.mcg", "w.mul1"}) {
    EXPECT_EQ(
        RuleOf(ParseExl3(Shard(Joined(Exl3Linear(), {E(marker, "I32", {0})})), kExl3NoCodebook)),
        Rule::kQuantization)
        << marker;
  }
  // An empty sign vector is no vector at all.
  auto signs = Exl3Linear();
  signs[1] = E("w.suh", "F16", {0, 32});
  EXPECT_EQ(RuleOf(ParseExl3(Shard(signs), kExl3NoCodebook)), Rule::kQuantization);
}

TEST_F(ImportAdversarialTest, Exl3StatedCodebookMustMatchEveryMarker) {
  const auto config = [](std::string_view codebook) {
    return std::format(
        R"({{"model_type":"qwen2","quantization_config":{{"quant_method":"exl3","codebook":"{}"}}}})",
        codebook);
  };
  const auto mcg = Joined(Exl3Linear(), {E("w.mcg", "I32", {})});
  const auto mul1 = Joined(Exl3Linear(), {E("w.mul1", "I32", {})});
  EXPECT_EQ(RuleOf(ParseExl3(Shard(mcg), config("mcg"))), std::nullopt);
  EXPECT_EQ(RuleOf(ParseExl3(Shard(mul1), config("mul1"))), std::nullopt);
  EXPECT_EQ(RuleOf(ParseExl3(Shard(Exl3Linear()), config("3inst"))), std::nullopt);
  // A marker the stated codebook contradicts, or a missing one.
  EXPECT_EQ(RuleOf(ParseExl3(Shard(mul1), config("mcg"))), Rule::kQuantization);
  EXPECT_EQ(RuleOf(ParseExl3(Shard(Exl3Linear()), config("mcg"))), Rule::kQuantization);
  EXPECT_EQ(RuleOf(ParseExl3(Shard(mcg), config("3inst"))), Rule::kQuantization);
  EXPECT_EQ(RuleOf(ParseExl3(Shard(mcg), config("trellis9"))), Rule::kUnsupported);
  // ModelOpt's description beside EXL3's.
  Write(dir_ / "exl3" / "config.json", config("mcg"));
  Write(dir_ / "exl3" / "model.safetensors", Shard(mcg));
  ASSERT_TRUE(ReadExl3(dir_ / "exl3"));
  Write(dir_ / "exl3" / "hf_quant_config.json",
        std::string_view(R"({"quantization":{"quant_algo":"NVFP4"}})"));
  EXPECT_EQ(RuleOf(ReadExl3(dir_ / "exl3")), Rule::kConfig);
}

constexpr std::string_view kNvfp4Config =
    R"({"model_type":"t","quantization_config":{"quant_method":"modelopt","quant_algo":"NVFP4"}})";

// An NVFP4 weight of 32 values a row, 16 rows.
std::vector<Entry> Nvfp4(const std::string& m) {
  return {E(m + ".weight", "U8", {16, 16}), E(m + ".weight_scale", "F8_E4M3", {16, 2})};
}

TEST_F(ImportAdversarialTest, HfQuantEmptyScalesAreNotReadAsAbsent) {
  const auto scale2 = [](const std::string& m, std::vector<std::uint64_t> shape) {
    return E(m + ".weight_scale_2", "F32", std::move(shape));
  };
  const auto parse = [](const std::vector<Entry>& e) {
    return ParseSafetensors(Shard(e), kNvfp4Config);
  };
  ASSERT_TRUE(parse(Joined(Nvfp4("w"), {scale2("w", {})})));
  // An empty tensor scale would silently drop the weight's tensor scale.
  EXPECT_EQ(RuleOf(parse(Joined(Nvfp4("w"), {scale2("w", {0})}))), Rule::kQuantization);
  // ... or take an ancestor's in its place.
  const auto shared = Joined(Nvfp4("a.b"), {scale2("a", {})});
  const auto inherited = parse(shared);
  ASSERT_TRUE(inherited);
  EXPECT_EQ(inherited->FindTensor("a.weight_scale_2")->part, format::PartRole::kTensorScale);
  EXPECT_EQ(RuleOf(parse(Joined(shared, {scale2("a.b", {0})}))), Rule::kQuantization);
  // An empty block scale would leave the codes dense; empty codes beside
  // their scales likewise.
  EXPECT_EQ(RuleOf(parse({E("w.weight", "U8", {16, 16}), E("w.weight_scale", "F8_E4M3", {16, 0})})),
            Rule::kQuantization);
  EXPECT_EQ(RuleOf(parse({E("w.weight", "U8", {0, 16}), E("w.weight_scale", "F8_E4M3", {16, 2})})),
            Rule::kQuantization);
  // Unquantized, an empty tensor is only that.
  EXPECT_TRUE(ParseSafetensors(Shard({E("w.weight_scale_2", "F32", {0}), E("x", "F32", {1})}),
                               R"({"model_type":"t"})"));
}

TEST_F(ImportAdversarialTest, HfQuantLiteralExclusionsChargeTheirComparisons) {
  // Literal exclusions of a name's length, each compared in full against
  // every scaled weight: about 2 million steps a weight, so a thousand-odd
  // weights spend the budget that charging one step a pattern never did.
  const auto module = [](std::size_t i) { return std::string(484, 'm') + std::format("{:06}", i); };
  std::string excluded;
  for (std::size_t j = 0; j < 4096; ++j)
    excluded += std::format(R"({}"{}x")", j == 0 ? "" : ",",
                            std::string(484, 'm') + std::format("{:05}", j));
  const std::string config = std::format(
      R"({{"model_type":"t","quantization_config":{{"quant_method":"modelopt","quant_algo":"NVFP4","exclude_modules":[{}]}}}})",
      excluded);
  const auto weights = [&](std::size_t n) {
    std::vector<Entry> out;
    for (std::size_t i = 0; i < n; ++i) {
      out.push_back(E(module(i) + ".weight", "U8", {1, 8}));
      out.push_back(E(module(i) + ".weight_scale", "F8_E4M3", {1, 1}));
    }
    return out;
  };
  EXPECT_EQ(RuleOf(ParseSafetensors(Shard(weights(4)), config)), std::nullopt);
  EXPECT_EQ(RuleOf(ParseSafetensors(Shard(weights(1100)), config)), Rule::kBounds);
}

constexpr std::string_view kMlxAffine =
    R"({"model_type":"qwen3","quantization":{"group_size":32,"bits":4}})";

// An affine 4-bit module of 64 values a row, group 32, 4 rows.
std::vector<Entry> MlxAffine() {
  return {E("m.weight", "U32", {4, 8}), E("m.scales", "BF16", {4, 2}),
          E("m.biases", "BF16", {4, 2})};
}

TEST_F(ImportAdversarialTest, MlxEmptyPartsAndDisagreeingConfigsAreRefused) {
  const auto parse = [](const std::vector<Entry>& e, std::string_view config = kMlxAffine) {
    return ParseMlx(Shard(e), config);
  };
  ASSERT_TRUE(parse(MlxAffine()));
  for (std::size_t i = 0; i < 3; ++i) {
    auto empty = MlxAffine();
    empty[i].shape[0] = 0;
    empty[i].bytes = 0;
    EXPECT_EQ(RuleOf(parse(empty)), Rule::kQuantization) << empty[i].name;
  }
  // mlx-lm's copy in quantization_config must say the same.
  EXPECT_TRUE(parse(MlxAffine(), R"({"model_type":"qwen3","quantization":{"group_size":32,"bits":4},
                                    "quantization_config":{"bits":4,"group_size":32}})"));
  EXPECT_EQ(RuleOf(parse(MlxAffine(), R"({"model_type":"qwen3",
      "quantization":{"group_size":32,"bits":4},"quantization_config":{"group_size":64,"bits":4}})")),
            Rule::kConfig);
  EXPECT_EQ(RuleOf(parse(MlxAffine(), R"({"model_type":"qwen3",
      "quantization":{"group_size":32,"bits":4},"quantization_config":{"quant_method":"fp8"}})")),
            Rule::kConfig);
  // MLX-written shards with a quantization_config but no `quantization`.
  const auto dense = Shard({E("x", "BF16", {4})}, R"({"format":"mlx"})");
  EXPECT_TRUE(ParseSafetensors(dense, R"({"model_type":"qwen3"})"));
  EXPECT_EQ(
      RuleOf(ParseSafetensors(
          dense, R"({"model_type":"qwen3","quantization_config":{"group_size":32,"bits":4}})")),
      Rule::kConfig);
  // ModelOpt's description beside MLX's.
  auto dir = ParseHfShard(Shard(MlxAffine()), kMlxAffine, R"({"quant_algo":"NVFP4"})");
  ASSERT_TRUE(dir);
  EXPECT_EQ(RuleOf(MlxCheckpoint(std::move(*dir))), Rule::kConfig);
}

TEST_F(ImportAdversarialTest, ConfigMemberNamesCannotForgeNestedKeys) {
  const auto shard = Shard({E("w", "F32", {1})});
  // A root member named "" once made its members look top-level.
  EXPECT_EQ(RuleOf(ParseSafetensors(shard, R"({"model_type":"t","":{"num_hidden_layers":5}})")),
            Rule::kConfig);
  EXPECT_EQ(RuleOf(ParseSafetensors(shard, R"({"model_type":"t","a":{"":1}})")), Rule::kConfig);
  // A dotted name is not a nested member: transformers reads it only as
  // itself, so it is left out, whatever it spells.
  auto dotted = ParseSafetensors(shard, R"({"model_type":"t","text_config.num_hidden_layers":99,
      "text_config":{"num_hidden_layers":4,"base_model_ep_plan":{"layers.*.mlp.experts":"ep"}},
      "rope_parameters.rope_theta":1})");
  ASSERT_TRUE(dotted) << dotted.error().ToString();
  const auto* layers = dotted->FindValue("text_config.num_hidden_layers");
  ASSERT_NE(layers, nullptr);
  EXPECT_EQ(std::get<std::int64_t>(*layers), 4);
  EXPECT_EQ(dotted->FindValue("rope_parameters.rope_theta"), nullptr);
  EXPECT_EQ(dotted->FindValue("text_config.base_model_ep_plan.layers.*.mlp.experts"), nullptr);
}

TEST_F(ImportAdversarialTest, UninspectablePipelineFileIsRefusedNotDropped) {
  const auto p = dir_ / "pipeline";
  Write(p / "model_index.json", R"({"_class_name":"P","vae":["d","V"],"scheduler":["d","S"]})");
  Write(p / "vae" / "config.json", R"({"_class_name":"V"})");
  Write(p / "vae" / "diffusion_pytorch_model.safetensors", Shard({E("y", "F32", {3})}));
  Write(p / "scheduler" / "scheduler_config.json", std::string_view("{}"));
  ASSERT_TRUE(ReadDiffusers(p));
  // A link to itself: stat fails with ELOOP, not ENOENT.
  std::filesystem::create_symlink("loop.json", p / "scheduler" / "loop.json");
  EXPECT_EQ(RuleOf(ReadDiffusers(p)), Rule::kIo);
}

}  // namespace
}  // namespace llmp::import
