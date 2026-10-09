// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The container readers (import/) on the real checkpoints of the current
// models: the Gemma 2 and Gemma 3 source GGUFs give, through their mappings
// and their own metadata, the spec the installed v0 artifacts give with
// today's profiles (spec/mapping.h SynthesizeV0), tensor for tensor in the
// same formats; the split and single GGUFs of DeepSeek V4, Qwen3.8 and
// Gemma 4 read whole; Qwen3.8's ModelOpt checkpoint pairs every NVFP4 and
// MXFP8 weight; and the readers refuse what is another reader's (a
// diffusers component, an MLX checkpoint). Needs the model files (label
// `models`); each case is skipped without its files.

#include <gtest/gtest.h>
#include <unistd.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <initializer_list>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <variant>
#include <vector>

#include "artifact/artifact.h"
#include "artifact/format_map.h"
#include "artifact/layout.h"
#include "format/encoding.h"
#include "format/layout.h"
#include "format/tensor_format.h"
#include "import/checkpoint.h"
#include "import/error.h"
#include "import/reader.h"
#include "import/readers/gguf.h"
#include "import/readers/safetensors.h"
#include "model/gemma2.h"
#include "model/gemma3.h"
#include "spec/mapping.h"
#include "spec/mappings/gemma2.h"
#include "spec/mappings/gemma3.h"
#include "spec/spec.h"

namespace llmp::import {
namespace {

std::filesystem::path Models() {
  // NOLINTNEXTLINE(concurrency-mt-unsafe): tests read, never set, the environment
  if (const char* dir = std::getenv("LLMP_TEST_MODELS"); dir != nullptr) {
    return dir;
  }
  const char* home = std::getenv("HOME");  // NOLINT(concurrency-mt-unsafe)
  return home != nullptr ? std::filesystem::path(home) / ".local/share/llmp" : "";
}

bool Exists(const std::filesystem::path& path) {
  std::error_code error;
  return std::filesystem::exists(path, error);
}

// Every tensor inside its file, its format the size rule's for its bytes.
void ExpectWellFormed(const Checkpoint& c) {
  for (const Tensor& t : c.tensors) {
    ASSERT_LT(t.extent.file, c.files.size()) << t.name;
    EXPECT_LE(t.extent.offset + t.extent.bytes, c.files[t.extent.file].bytes) << t.name;
    const auto parts = format::PartsOf(t.format);
    ASSERT_TRUE(parts.has_value()) << t.name;
    const auto part = parts->Find(t.part);
    ASSERT_TRUE(part.has_value()) << t.name;
    EXPECT_EQ(part.value_or(format::PartExtent{}).bytes, t.extent.bytes) << t.name;
  }
}

struct GemmaCase {
  std::string_view architecture;
  std::string_view gguf;
  std::string_view artifact;
  std::size_t tensors;
};

class ImportGemmaTest : public testing::TestWithParam<GemmaCase> {};

TEST_P(ImportGemmaTest, SourceGgufGivesTheV0ArtifactsSpec) {
  const GemmaCase& c = GetParam();
  const std::filesystem::path gguf = Models() / c.gguf;
  const std::filesystem::path path = Models() / c.artifact;
  if (!Exists(gguf) || !Exists(path)) {
    GTEST_SKIP() << "no " << gguf << " or " << path;
  }
  const auto checkpoint = ReadCheckpoint(gguf, ::geteuid());
  ASSERT_TRUE(checkpoint.has_value()) << checkpoint.error().ToString();
  EXPECT_EQ(checkpoint->container, "gguf");
  EXPECT_EQ(checkpoint->architecture, c.architecture);
  EXPECT_EQ(checkpoint->tensors.size(), c.tensors);
  ASSERT_EQ(checkpoint->files.size(), 1U);
  ExpectWellFormed(*checkpoint);

  auto artifact = artifact::Artifact::Open(path, {.expected_id = {}, .trusted_owner = ::geteuid()});
  ASSERT_TRUE(artifact.has_value()) << artifact.error().ToString();
  // The artifact was imported from this file.
  ASSERT_EQ(artifact->sources().size(), 1U);
  EXPECT_EQ(artifact->sources()[0].name, gguf.filename().string());
  EXPECT_EQ(artifact->sources()[0].bytes.value(), checkpoint->files[0].bytes);

  // Tensor for tensor, the v0 artifact's formats.
  const auto v0 = spec::SourceOfV0(*artifact);
  ASSERT_TRUE(v0.has_value()) << v0.error();
  const spec::Source source = SourceOf(*checkpoint);
  EXPECT_EQ(source.container, v0->container);
  ASSERT_EQ(source.tensors.size(), v0->tensors.size());
  std::map<std::string, const spec::SourceTensor*> by_name;
  for (const spec::SourceTensor& t : source.tensors) {
    by_name.emplace(t.name, &t);
  }
  for (const spec::SourceTensor& t : v0->tensors) {
    const auto found = by_name.find(t.name);
    ASSERT_NE(found, by_name.end()) << t.name;
    EXPECT_EQ(found->second->format, t.format) << t.name;
    EXPECT_EQ(found->second->part, t.part) << t.name;
    EXPECT_EQ(found->second->experts, t.experts) << t.name;
  }

  // The spec from the checkpoint's own metadata is the profile's.
  const spec::Mapping* mapping = spec::FindMapping(checkpoint->architecture);
  ASSERT_NE(mapping, nullptr);
  const auto profile = c.architecture == "gemma2"
                           ? spec::mappings::Gemma2V0Values(model::Gemma2_2B())
                           : spec::mappings::Gemma3V0Values(model::Gemma3_4BQat());
  ASSERT_TRUE(profile.has_value()) << profile.error();
  const auto from_v0 = spec::SynthesizeV0(*artifact, *mapping, *profile);
  ASSERT_TRUE(from_v0.has_value()) << from_v0.error();
  const auto from_gguf = spec::Synthesize(*mapping, source, ValuesOf(*checkpoint));
  ASSERT_TRUE(from_gguf.has_value()) << from_gguf.error();
  EXPECT_EQ(*from_gguf, *from_v0);
  EXPECT_EQ(spec::ToJson(*from_gguf), spec::ToJson(*from_v0));
  RecordProperty("metadata_keys", static_cast<int>(checkpoint->metadata.size()));
}

INSTANTIATE_TEST_SUITE_P(
    Gemma, ImportGemmaTest,
    testing::Values(
        GemmaCase{.architecture = "gemma2",
                  .gguf = "models/bartowski/gemma-2-2b-it-GGUF@855f67ca/gemma-2-2b-it-Q8_0.gguf",
                  .artifact = "gemma2-import-20261007/artifacts/"
                              "eb18d30d0a7de3a95c7b6994b65a12a057ffbf42866add6f128873de8b7aa870",
                  .tensors = 288},
        GemmaCase{.architecture = "gemma3",
                  .gguf = "models/ggml-org/gemma-3-4b-it-qat-GGUF@bbcac0d0/"
                          "gemma-3-4b-it-qat-Q4_0.gguf",
                  .artifact = "gemma3-import-20261007/artifacts/"
                              "8c7103418a6608022e5eda50a0dcc4b7688a0d59ef239813c9de0984161397fb",
                  .tensors = 444}),
    [](const testing::TestParamInfo<GemmaCase>& info) {
      return std::string(info.param.architecture);
    });

struct GgufCase {
  std::string_view name;
  std::string_view path;  // the file, or a split's first file
  std::size_t files;
};

class ImportGgufTest : public testing::TestWithParam<GgufCase> {};

TEST_P(ImportGgufTest, ReadsWhole) {
  const GgufCase& c = GetParam();
  const std::filesystem::path path = Models() / c.path;
  if (!Exists(path)) {
    GTEST_SKIP() << "no " << path;
  }
  const auto checkpoint = ReadGguf(path, ::geteuid());
  ASSERT_TRUE(checkpoint.has_value()) << checkpoint.error().ToString();
  EXPECT_EQ(checkpoint->files.size(), c.files);
  EXPECT_EQ(checkpoint->open.size(), c.files);
  EXPECT_FALSE(checkpoint->architecture.empty());
  EXPECT_FALSE(checkpoint->tensors.empty());
  ExpectWellFormed(*checkpoint);
  if (c.files > 1) {
    const MetaValue* count = checkpoint->FindValue("split.tensors.count");
    ASSERT_NE(count, nullptr);
    EXPECT_EQ(std::get<std::int64_t>(*count),
              static_cast<std::int64_t>(checkpoint->tensors.size()));
  }
  std::size_t experts = 0;
  for (const Tensor& t : checkpoint->tensors) {
    experts += t.experts ? 1 : 0;
  }
  RecordProperty("architecture", checkpoint->architecture);
  RecordProperty("tensors", static_cast<int>(checkpoint->tensors.size()));
  RecordProperty("expert_arrays", static_cast<int>(experts));
  RecordProperty("metadata_keys", static_cast<int>(checkpoint->metadata.size()));
}

INSTANTIATE_TEST_SUITE_P(
    Models, ImportGgufTest,
    testing::Values(
        GgufCase{.name = "deepseek_v4_ud_q2",
                 .path = "models/unsloth/DeepSeek-V4-Flash-0731-GGUF@fbbb5b93/UD-Q2_K_XL/"
                         "DeepSeek-V4-Flash-0731-UD-Q2_K_XL-00001-of-00003.gguf",
                 .files = 3},
        GgufCase{.name = "deepseek_v4_antirez",
                 .path = "models/antirez/deepseek-v4-gguf@f71f23d5/"
                         "DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-"
                         "0731.gguf",
                 .files = 1},
        GgufCase{.name = "dspark_q8",
                 .path = "models/unsloth/DeepSeek-V4-Flash-0731-GGUF@fbbb5b93/"
                         "dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf",
                 .files = 1},
        GgufCase{.name = "qwen38_ud_iq3",
                 .path = "reference-models/Q/UD-IQ3_XXS/"
                         "Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf",
                 .files = 3},
        GgufCase{.name = "gemma4_26b",
                 .path = "reference-models/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf",
                 .files = 1},
        GgufCase{.name = "gemma4_31b",
                 .path = "reference-models/gemma-4-31B-it-UD-Q4_K_XL.gguf",
                 .files = 1},
        GgufCase{.name = "gemma4_26b_mtp",
                 .path = "reference-models/mtp-gemma-4-26B-A4B-it.gguf",
                 .files = 1}),
    [](const testing::TestParamInfo<GgufCase>& info) { return std::string(info.param.name); });

struct V0Case {
  std::string_view name;
  std::string_view artifact;
  std::string_view gguf;  // its source, or the first of its splits
};

class ImportV0FormatsTest : public testing::TestWithParam<V0Case> {};

// Every tensor of a v0 artifact imported from GGUF has the format the
// reader gives the source's tensor of that name (v0's expert arrays as the
// GGUF's three-dimensional expert tensors).
TEST_P(ImportV0FormatsTest, SourceFormatsAreTheArtifacts) {
  const V0Case& c = GetParam();
  const std::filesystem::path gguf = Models() / c.gguf;
  const std::filesystem::path path = Models() / c.artifact;
  if (!Exists(gguf) || !Exists(path)) {
    GTEST_SKIP() << "no " << gguf << " or " << path;
  }
  const auto checkpoint = ReadGguf(gguf, ::geteuid());
  ASSERT_TRUE(checkpoint.has_value()) << checkpoint.error().ToString();
  auto artifact = artifact::Artifact::Open(path, {.expected_id = {}, .trusted_owner = ::geteuid()});
  ASSERT_TRUE(artifact.has_value()) << artifact.error().ToString();
  ASSERT_EQ(artifact->sources().size(), checkpoint->files.size());
  for (std::size_t i = 0; i < checkpoint->files.size(); ++i) {
    EXPECT_EQ(artifact->sources()[i].name, checkpoint->files[i].name);
    EXPECT_EQ(artifact->sources()[i].bytes.value(), checkpoint->files[i].bytes);
  }
  EXPECT_EQ(artifact->model().architecture, checkpoint->architecture);
  const auto v0 = spec::SourceOfV0(*artifact);
  ASSERT_TRUE(v0.has_value()) << v0.error();
  std::map<std::string, const Tensor*> by_name;
  for (const Tensor& t : checkpoint->tensors) {
    by_name.emplace(t.name, &t);
  }
  for (const spec::SourceTensor& t : v0->tensors) {
    const auto found = by_name.find(t.name);
    ASSERT_NE(found, by_name.end()) << t.name;
    EXPECT_EQ(found->second->format, t.format) << t.name;
    EXPECT_EQ(found->second->experts, t.experts) << t.name;
  }
  RecordProperty("v0_tensors", static_cast<int>(v0->tensors.size()));
  RecordProperty("source_tensors", static_cast<int>(checkpoint->tensors.size()));
}

INSTANTIATE_TEST_SUITE_P(
    M3, ImportV0FormatsTest,
    testing::Values(
        V0Case{.name = "gemma4_26b",
               .artifact = "m3-artifacts/"
                           "4ddb360c9ce08f1e984ab304b6af918be44246d52346734066b06443f7c249d3",
               .gguf = "reference-models/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"},
        V0Case{.name = "gemma4_31b",
               .artifact = "m3-artifacts/"
                           "32c92e077a6816b54aa988e2dee61a3639c958fd510ea99e25f3621f10b2aa08",
               .gguf = "reference-models/gemma-4-31B-it-UD-Q4_K_XL.gguf"},
        V0Case{.name = "gemma4_26b_assistant",
               .artifact = "m3-artifacts/"
                           "1040a0299a459e00ad0a77efd77bd319ac593986ba2c9ef29eb03d07ce97db42",
               .gguf = "reference-models/mtp-gemma-4-26B-A4B-it.gguf"},
        V0Case{.name = "gemma4_31b_assistant",
               .artifact = "m3-artifacts/"
                           "447a5c20a0a25632bf35e118d5dde1867a182cd209b9ccd3afe93866c3696120",
               .gguf = "reference-models/mtp-gemma-4-31B-it.gguf"},
        V0Case{.name = "deepseek_v4_ud_q2",
               .artifact = "m3-artifacts/"
                           "8a355bfb27c90e1150fbd7fa62ea6e63f6bf34fcca33934e52d22773f1508234",
               .gguf = "models/unsloth/DeepSeek-V4-Flash-0731-GGUF@fbbb5b93/UD-Q2_K_XL/"
                       "DeepSeek-V4-Flash-0731-UD-Q2_K_XL-00001-of-00003.gguf"},
        V0Case{.name = "deepseek_v4_antirez",
               .artifact = "m3-artifacts/"
                           "cd39d504dc2dbfe911a4a521fa8efc8053dc3e80e99738a9b25fa6b70c97a1ac",
               .gguf = "models/antirez/deepseek-v4-gguf@f71f23d5/"
                       "DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-"
                       "0731.gguf"},
        V0Case{.name = "dspark_q8",
               .artifact = "m3-artifacts/"
                           "dd2d3f9c66f070fb231d27d5a11f38ff22c78dc8f089cecedbb67721e9b4bec5",
               .gguf = "models/unsloth/DeepSeek-V4-Flash-0731-GGUF@fbbb5b93/"
                       "dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf"},
        V0Case{.name = "qwen38_ud_iq3",
               .artifact = "qgguf-artifacts/"
                           "5356b5b05fd93d06419cd842c4946e0df9d57ec816916109af5c724cadf25b78",
               .gguf = "reference-models/Q/UD-IQ3_XXS/"
                       "Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf"}),
    [](const testing::TestParamInfo<V0Case>& info) { return std::string(info.param.name); });

constexpr std::string_view kQwen38 = "models/Mia-AiLab/Qwen3.8-Flash-Next-NVFP4@925d7be6";

TEST(ImportSafetensorsModelsTest, Qwen38ModelOptPairsEveryWeight) {
  const std::filesystem::path path = Models() / kQwen38;
  if (!Exists(path)) {
    GTEST_SKIP() << "no " << path;
  }
  const auto checkpoint = ReadCheckpoint(path, ::geteuid());
  ASSERT_TRUE(checkpoint.has_value()) << checkpoint.error().ToString();
  EXPECT_EQ(checkpoint->container, "safetensors");
  EXPECT_EQ(checkpoint->architecture, "qwen3_8_flash_next");
  EXPECT_EQ(checkpoint->quantization, "modelopt MIXED_PRECISION");
  ExpectWellFormed(*checkpoint);
  // The shards the index names (34 and the input scales'), then
  // config.json, hf_quant_config.json and the index.
  EXPECT_EQ(checkpoint->files.size(), 35U + 3U);
  std::map<std::pair<format::Encoding, format::PartRole>, std::size_t> parts;
  std::size_t tensor_scaled = 0;
  for (const Tensor& t : checkpoint->tensors) {
    ++parts[{t.format.encoding, t.part}];
    tensor_scaled += t.part == format::PartRole::kCodes && t.format.params.tensor_scale ? 1 : 0;
  }
  using E = format::Encoding;
  using P = format::PartRole;
  // The checkpoint's U8 weights with E4M3 scales (the routed experts, MTP
  // and vision fc2 under the config; the n-gram table's 128 shards by
  // their dtypes), and its E4M3 weights with E8M0 scales.
  EXPECT_EQ((parts[{E::kNvfp4, P::kCodes}]), 75419U);
  EXPECT_EQ((parts[{E::kNvfp4, P::kBlockScales}]), 75419U);
  // Each its own tensor scale but the n-gram shards, which share their
  // table's.
  EXPECT_EQ((parts[{E::kNvfp4, P::kTensorScale}]), 75292U);
  EXPECT_EQ(tensor_scaled, 75419U);
  EXPECT_EQ((parts[{E::kMxfp8, P::kCodes}]), 467U);
  EXPECT_EQ((parts[{E::kMxfp8, P::kBlockScales}]), 467U);
  const Tensor* ngram = checkpoint->FindTensor(
      "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight");
  ASSERT_NE(ngram, nullptr);
  EXPECT_EQ(ngram->format.encoding, E::kNvfp4);
  EXPECT_EQ(ngram->format.layout, format::Layout::kSplit);
  EXPECT_EQ(ngram->format.shape.ne[0], 160U);
  EXPECT_TRUE(ngram->format.params.tensor_scale);
  const Tensor* table_scale = checkpoint->FindTensor(
      "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.weight_scale_2");
  ASSERT_NE(table_scale, nullptr);
  EXPECT_EQ(table_scale->part, P::kTensorScale);
  EXPECT_EQ(table_scale->format, ngram->format);
  const MetaValue* layers = checkpoint->FindValue("text_config.num_hidden_layers");
  ASSERT_NE(layers, nullptr);
  EXPECT_TRUE(std::holds_alternative<std::int64_t>(*layers));
  RecordProperty("tensors", static_cast<int>(checkpoint->tensors.size()));
  RecordProperty("metadata_keys", static_cast<int>(checkpoint->metadata.size()));
}

// A v0 artifact's bytes: `bytes` at `offset` in group `group`'s data.
std::vector<std::byte> ReadV0(const std::filesystem::path& root, const artifact::Artifact& a,
                              std::uint32_t group, std::uint64_t offset, std::uint64_t bytes) {
  const artifact::Group& g = a.groups()[group];
  const artifact::Shard& shard = a.shards()[g.shard];
  std::ifstream in(root / shard.path, std::ios::binary);
  in.seekg(static_cast<std::streamoff>(shard.data_offset.value() + g.offset.value() + offset));
  std::vector<std::byte> out(bytes);
  in.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(out.size()));
  EXPECT_TRUE(in.good());
  return out;
}

std::vector<std::byte> ReadV0Resource(const std::filesystem::path& root,
                                      const artifact::Artifact& a, std::string_view role,
                                      std::uint64_t offset = 0,
                                      std::optional<std::uint64_t> bytes = std::nullopt) {
  const auto r = a.FindResource(role);
  EXPECT_TRUE(r.has_value()) << role;
  if (!r) {
    return {};
  }
  const auto p = a.ResourcePlacement(*r);
  EXPECT_TRUE(p.has_value()) << role;
  if (!p) {
    return {};
  }
  return ReadV0(root, a, p->group, p->offset.value() + offset,
                bytes.value_or(p->bytes.value() - offset));
}

std::vector<std::byte> ReadSource(const Checkpoint& c, std::string_view name,
                                  std::uint64_t offset = 0,
                                  std::optional<std::uint64_t> bytes = std::nullopt) {
  const Tensor* t = c.FindTensor(name);
  EXPECT_NE(t, nullptr) << name;
  if (t == nullptr) {
    return {};
  }
  std::vector<std::byte> out(bytes.value_or(t->extent.bytes - offset));
  const auto ok = c.Read(t->extent, offset, out);
  EXPECT_TRUE(ok.has_value()) << name << ": " << (ok ? "" : ok.error().ToString());
  return out;
}

// Equal bytes, or where they first differ (never the bytes themselves).
testing::AssertionResult Same(const std::vector<std::byte>& a, const std::vector<std::byte>& b) {
  if (a.size() != b.size()) {
    return testing::AssertionFailure() << "sizes " << a.size() << " and " << b.size();
  }
  const auto [x, y] = std::ranges::mismatch(a, b);
  if (x != a.end()) {
    return testing::AssertionFailure() << "first difference at byte " << (x - a.begin());
  }
  return testing::AssertionSuccess();
}

// The same rows in any order, `row` bytes each.
testing::AssertionResult SameRows(const std::vector<std::byte>& a, const std::vector<std::byte>& b,
                                  std::size_t row) {
  if (a.size() != b.size() || row == 0 || a.size() % row != 0) {
    return testing::AssertionFailure() << "sizes " << a.size() << " and " << b.size();
  }
  const auto rows = [row](const std::vector<std::byte>& v) {
    std::vector<std::string_view> out;
    for (std::size_t i = 0; i < v.size(); i += row) {
      out.emplace_back(reinterpret_cast<const char*>(v.data() + i), row);
    }
    std::ranges::sort(out);
    return out;
  };
  if (rows(a) != rows(b)) {
    return testing::AssertionFailure() << "different rows";
  }
  return testing::AssertionSuccess();
}

// The ModelOpt checkpoint as the reader describes it against the v0
// import the Python importer made of the same files (the CUTLASS-layout
// artifact): layer 1's MXFP8 linears have the format v0's codes and scales
// assemble to, and their bytes: the shared expert's the same, the linear
// attention's input projections the same rows (v0 reordered their heads
// as llama.cpp's converter does, which a mapping's fold will do, and its
// output projection's columns, compared by format only); every routed
// expert's NVFP4 tensor scale is the value v0 keeps for it, per
// projection (gate and up the halves v0 stacks); and the n-gram table's
// 128 shards are v0's row-interleaved table, under the one tensor scale v0
// keeps for it.
TEST(ImportSafetensorsModelsTest, Qwen38ModelOptMatchesTheV0Import) {
  const std::filesystem::path path = Models() / kQwen38;
  const std::filesystem::path v0 =
      Models() / "m3-artifacts/c4fb47a911207c11f935f932d05196dc1701aa0d886eac1b5e91934e554b5a93";
  if (!Exists(path) || !Exists(v0)) {
    GTEST_SKIP() << "no " << path << " or " << v0;
  }
  const auto c = ReadCheckpoint(path, ::geteuid());
  ASSERT_TRUE(c.has_value()) << c.error().ToString();
  auto a = artifact::Artifact::Open(v0, {.expected_id = {}, .trusted_owner = ::geteuid()});
  ASSERT_TRUE(a.has_value()) << a.error().ToString();
  const std::string l1 = "model.language_model.layers.1.";
  enum class Bytes : std::uint8_t { kSame, kSameRows, kFormatOnly };
  struct Linear {
    std::string_view v0;
    std::string source;
    Bytes bytes;
  };
  for (const auto& [v0_name, source, bytes] : std::initializer_list<Linear>{
           {"blk.1.attn_qkv", l1 + "linear_attn.in_proj_qkv", Bytes::kSameRows},
           {"blk.1.attn_gate", l1 + "linear_attn.in_proj_z", Bytes::kSameRows},
           {"blk.1.ssm_beta", l1 + "linear_attn.in_proj_b", Bytes::kSameRows},
           {"blk.1.ssm_alpha", l1 + "linear_attn.in_proj_a", Bytes::kSameRows},
           {"blk.1.ssm_out", l1 + "linear_attn.out_proj", Bytes::kFormatOnly},
           {"blk.1.ffn_gate_shexp", l1 + "mlp.shared_expert.gate_proj", Bytes::kSame},
           {"blk.1.ffn_up_shexp", l1 + "mlp.shared_expert.up_proj", Bytes::kSame},
           {"blk.1.ffn_down_shexp", l1 + "mlp.shared_expert.down_proj", Bytes::kSame}}) {
    const std::string codes = std::string(v0_name) + ".weight";
    const std::string scales = codes + "_scale";
    const auto ci = a->FindResource(codes);
    const auto si = a->FindResource(scales);
    ASSERT_TRUE(ci && si) << v0_name;
    const std::vector<artifact::V0Part> v0_parts = {
        {format::PartRole::kCodes, &a->resources()[*ci].repr},
        {format::PartRole::kBlockScales, &a->resources()[*si].repr}};
    const auto v0_format =
        artifact::Assemble(format::Encoding::kMxfp8, format::Layout::kSplit, v0_parts);
    ASSERT_TRUE(v0_format.has_value()) << v0_name;
    const Tensor* w = c->FindTensor(source + ".weight");
    const Tensor* s = c->FindTensor(source + ".weight_scale");
    ASSERT_TRUE(w != nullptr && s != nullptr) << source;
    EXPECT_EQ(w->format, *v0_format) << source;
    EXPECT_EQ(s->format, *v0_format) << source;
    const std::uint64_t k = w->format.shape.ne[0];
    if (bytes == Bytes::kSame) {
      EXPECT_TRUE(Same(ReadSource(*c, w->name), ReadV0Resource(v0, *a, codes))) << source;
      EXPECT_TRUE(Same(ReadSource(*c, s->name), ReadV0Resource(v0, *a, scales))) << source;
    } else if (bytes == Bytes::kSameRows) {
      EXPECT_TRUE(SameRows(ReadSource(*c, w->name), ReadV0Resource(v0, *a, codes), k)) << source;
      EXPECT_TRUE(SameRows(ReadSource(*c, s->name), ReadV0Resource(v0, *a, scales), k / 32))
          << source;
    }
  }
  // The routed experts' tensor scales, F32 [512] a projection in v0.
  for (const std::string_view proj : {"gate", "up", "down"}) {
    const auto v0_scales =
        ReadV0Resource(v0, *a, std::format("blk.1.ffn_{}_exps.weight_scale_2", proj));
    ASSERT_EQ(v0_scales.size(), 512U * 4U) << proj;
    for (std::uint64_t e = 0; e < 512; ++e) {
      const std::string module = std::format("{}mlp.experts.{}.{}_proj", l1, e, proj);
      const Tensor* w = c->FindTensor(module + ".weight");
      ASSERT_NE(w, nullptr) << module;
      EXPECT_TRUE(w->format.params.tensor_scale) << module;
      EXPECT_EQ(w->format.encoding, format::Encoding::kNvfp4) << module;
      const auto source = ReadSource(*c, module + ".weight_scale_2");
      EXPECT_TRUE(std::equal(source.begin(), source.end(),
                             v0_scales.begin() + static_cast<std::ptrdiff_t>(e * 4)))
          << module;
    }
  }
  // The n-gram table: v0's row r is shard r / rows's row r % rows, its 80
  // code bytes then its 10 scale bytes.
  const std::string table = l1 + "ple.ple_embedding.ngram_embedding.";
  const Tensor* shard0 = c->FindTensor(table + "shard_0.weight");
  ASSERT_NE(shard0, nullptr);
  const std::uint64_t rows = shard0->format.shape.ne[1];
  const std::uint64_t code_row = shard0->format.shape.ne[0] / 2;
  const std::uint64_t scale_row = shard0->format.shape.ne[0] / 16;
  for (const std::uint64_t r :
       {std::uint64_t{0}, rows - 1, rows, std::uint64_t{160000000}, (128 * rows) - 1}) {
    const std::string shard = std::format("{}shard_{}.", table, r / rows);
    const std::uint64_t row = r % rows;
    auto want = ReadSource(*c, shard + "weight", row * code_row, code_row);
    const auto scales = ReadSource(*c, shard + "weight_scale", row * scale_row, scale_row);
    want.insert(want.end(), scales.begin(), scales.end());
    EXPECT_TRUE(Same(ReadV0Resource(v0, *a, "per_layer_token_embd.weight",
                                    r * (code_row + scale_row), code_row + scale_row),
                     want))
        << r;
  }
  EXPECT_TRUE(Same(ReadSource(*c, table + "weight_scale_2"),
                   ReadV0Resource(v0, *a, "per_layer_token_embd.weight_scale_2")));
}

TEST(ImportSafetensorsModelsTest, ReadsQwenImagesTextEncoder) {
  const std::filesystem::path path = Models() / "models/Qwen/Qwen-Image-2.1@790c9263/text_encoder";
  if (!Exists(path)) {
    GTEST_SKIP() << "no " << path;
  }
  const auto checkpoint = ReadSafetensors(path, ::geteuid());
  ASSERT_TRUE(checkpoint.has_value()) << checkpoint.error().ToString();
  EXPECT_EQ(checkpoint->architecture, "qwen3_vl");
  EXPECT_TRUE(checkpoint->quantization.empty());
  ExpectWellFormed(*checkpoint);
  RecordProperty("tensors", static_cast<int>(checkpoint->tensors.size()));
}

TEST(ImportSafetensorsModelsTest, LeavesOtherReadersCheckpointsToThem) {
  // A diffusers component has no model_type; an MLX checkpoint's
  // quantization is the MLX reader's.
  const std::filesystem::path transformer =
      Models() / "models/Qwen/Qwen-Image-2.1@790c9263/transformer";
  if (Exists(transformer)) {
    const auto c = ReadSafetensors(transformer, ::geteuid());
    ASSERT_FALSE(c.has_value());
    EXPECT_EQ(c.error().rule, Rule::kConfig) << c.error().ToString();
  }
  const std::filesystem::path mlx =
      Models() / "models/Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP@dadefa80";
  if (Exists(mlx)) {
    const auto c = ReadSafetensors(mlx, ::geteuid());
    ASSERT_FALSE(c.has_value());
    EXPECT_EQ(c.error().rule, Rule::kUnsupported) << c.error().ToString();
  }
  if (!Exists(transformer) && !Exists(mlx)) {
    GTEST_SKIP() << "no diffusers or MLX checkpoint";
  }
}

}  // namespace
}  // namespace llmp::import
