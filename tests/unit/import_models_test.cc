// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The container readers (import/) on the real checkpoints of the current
// models: the Gemma 2 and Gemma 3 source GGUFs give, through their mappings
// and their own metadata, the spec the installed v0 artifacts give with
// today's profiles (spec/mapping.h SynthesizeV0), tensor for tensor in the
// same formats; the split and single GGUFs of DeepSeek V4, Qwen3.8 and
// Gemma 4 read whole; Qwen3.8's ModelOpt checkpoint pairs every NVFP4 and
// MXFP8 weight; Qwen-Image-2.1's pipeline is its v0 composition's
// components tensor for tensor and byte for byte; the Qwen2.5 EXL3
// fixtures are their v0 imports; Qwen3.8's MLX checkpoint pairs every
// quantized module and decodes within its quantization of the native
// checkpoint's weights; and the safetensors reader leaves an MLX
// checkpoint to its reader. Needs the model files (label `models`); each
// legacy case is skipped without its files; the explicitly scheduled closed
// Qwen header-normalization case refuses missing input.

#include <fcntl.h>
#include <gtest/gtest.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
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
#include "artifact/composition.h"
#include "artifact/format_map.h"
#include "artifact/layout.h"
#include "base/sha256.h"
#include "expected_error.h"
#include "format/encoding.h"
#include "format/layout.h"
#include "format/tensor_format.h"
#include "import/checkpoint.h"
#include "import/error.h"
#include "import/normalize.h"
#include "import/reader.h"
#include "import/readers/gguf.h"
#include "import/readers/safetensors.h"
#include "kernels/reference/decode.h"
#include "model/gemma2.h"
#include "model/gemma3.h"
#include "model/gemma4.h"
#include "qwen38_gguf_normalization_fixture.h"
#include "spec/mapping.h"
#include "spec/mappings/gemma2.h"
#include "spec/mappings/gemma3.h"
#include "spec/mappings/gemma4.h"
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
  auto checkpoint = ReadCheckpoint(gguf);
  ASSERT_TRUE(checkpoint.has_value()) << checkpoint.error().ToString();
  EXPECT_EQ(checkpoint->container, "gguf");
  EXPECT_EQ(checkpoint->architecture, c.architecture);
  EXPECT_EQ(checkpoint->tensors.size(), c.tensors);
  ASSERT_EQ(checkpoint->files.size(), 1U);
  ExpectWellFormed(*checkpoint);

  auto artifact = artifact::Artifact::Open(path, {.expected_id = {}});
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
  const auto bound = NormalizationBound(*checkpoint);
  ASSERT_TRUE(bound) << test_support::Failed(bound, &Error::ToString).value_or("");
  auto normalized = Normalize(std::move(*checkpoint), {bound->metadata_bytes, bound->work_units});
  ASSERT_TRUE(normalized) << test_support::Failed(normalized, &Error::ToString).value_or("");
  EXPECT_EQ(normalized->spec(), *from_v0);
  ASSERT_EQ(normalized->canonical().tensors.size(), source.tensors.size());
  ASSERT_EQ(normalized->routes().size(), source.tensors.size());
  ASSERT_EQ(normalized->metadata().size(), normalized->source().metadata.size());
  std::uint64_t payload = 0;
  for (std::size_t i = 0; i < source.tensors.size(); ++i) {
    const auto& actual = normalized->canonical().tensors[i];
    EXPECT_EQ(actual.name, source.tensors[i].name);
    EXPECT_EQ(actual.format, source.tensors[i].format);
    EXPECT_EQ(actual.part, source.tensors[i].part);
    const auto& route = normalized->routes()[i];
    EXPECT_EQ(route.source_tensor, i);
    EXPECT_EQ(route.extent, normalized->source().tensors[i].extent);
    payload += route.extent.bytes;
    auto cursor = normalized->CursorFor(i);
    ASSERT_TRUE(cursor);
    const auto copy = cursor->Next(route.extent.bytes);
    ASSERT_TRUE(copy && *copy);
    EXPECT_EQ((**copy).file, route.extent.file);
    EXPECT_EQ((**copy).source_offset, route.extent.offset);
    EXPECT_EQ((**copy).output_offset, 0U);
    EXPECT_EQ((**copy).bytes, route.extent.bytes);
    const auto end = cursor->Next(1);
    EXPECT_TRUE(end && !*end);
  }
  EXPECT_EQ(payload, normalized->cost().payload_bytes);
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
  const auto checkpoint = ReadGguf(path);
  ASSERT_TRUE(checkpoint.has_value()) << checkpoint.error().ToString();
  EXPECT_EQ(checkpoint->files.size(), c.files);
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
  const auto checkpoint = ReadGguf(gguf);
  ASSERT_TRUE(checkpoint.has_value()) << checkpoint.error().ToString();
  auto artifact = artifact::Artifact::Open(path, {.expected_id = {}});
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
  if (checkpoint->architecture == "gemma4") {
    const auto& profile = c.name == "gemma4_26b" ? model::Gemma4_26BA4B() : model::Gemma4_31B();
    const auto values = spec::mappings::Gemma4V0Values(profile);
    ASSERT_TRUE(values) << test_support::Failed(values).value_or("");
    const auto expected = spec::SynthesizeV0(*artifact, spec::mappings::Gemma4Mapping(), *values);
    ASSERT_TRUE(expected) << test_support::Failed(expected).value_or("");
    const auto bound = NormalizationBound(*checkpoint);
    ASSERT_TRUE(bound) << test_support::Failed(bound, &Error::ToString).value_or("");
    auto normalized = Normalize(*checkpoint, {bound->metadata_bytes, bound->work_units});
    ASSERT_TRUE(normalized) << test_support::Failed(normalized, &Error::ToString).value_or("");
    EXPECT_EQ(normalized->spec(), *expected);
    EXPECT_TRUE(normalized->applied_folds().empty());
    ASSERT_EQ(normalized->routes().size(), checkpoint->tensors.size());
    std::uint64_t payload = 0;
    for (std::size_t i = 0; i < checkpoint->tensors.size(); ++i) {
      const auto& source = checkpoint->tensors[i];
      const auto& canonical = normalized->canonical().tensors[i];
      EXPECT_EQ(canonical.name, source.name);
      EXPECT_EQ(canonical.format, source.format);
      EXPECT_EQ(canonical.experts, source.experts);
      EXPECT_EQ(normalized->routes()[i].extent, source.extent);
      EXPECT_EQ(normalized->routes()[i].source_tensor, i);
      payload += source.extent.bytes;
    }
    EXPECT_EQ(normalized->cost().payload_bytes, payload);
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
constexpr std::string_view kMlx = "models/Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP@dadefa80";

TEST(ImportSafetensorsModelsTest, Qwen38ModelOptPairsEveryWeight) {
  const std::filesystem::path path = Models() / kQwen38;
  if (!Exists(path)) {
    GTEST_SKIP() << "no " << path;
  }
  const auto checkpoint = ReadCheckpoint(path);
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
  const auto c = ReadCheckpoint(path);
  ASSERT_TRUE(c.has_value()) << c.error().ToString();
  auto a = artifact::Artifact::Open(v0, {.expected_id = {}});
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
  const auto checkpoint = ReadSafetensors(path);
  ASSERT_TRUE(checkpoint.has_value()) << checkpoint.error().ToString();
  EXPECT_EQ(checkpoint->architecture, "qwen3_vl");
  EXPECT_TRUE(checkpoint->quantization.empty());
  ExpectWellFormed(*checkpoint);
  RecordProperty("tensors", static_cast<int>(checkpoint->tensors.size()));
}

TEST(ImportSafetensorsModelsTest, LeavesOtherReadersCheckpointsToThem) {
  // A diffusers component alone reads by its _class_name; an MLX
  // checkpoint (its shards in MLX's format) is the MLX reader's, which
  // the safetensors reader hands it to.
  const std::filesystem::path transformer =
      Models() / "models/Qwen/Qwen-Image-2.1@790c9263/transformer";
  if (Exists(transformer)) {
    const auto c = ReadSafetensors(transformer);
    ASSERT_TRUE(c.has_value()) << c.error().ToString();
    EXPECT_EQ(c->architecture, "QwenImage21Transformer2DModel");
  }
  const std::filesystem::path mlx = Models() / kMlx;
  if (Exists(mlx)) {
    const auto c = ReadSafetensors(mlx);
    ASSERT_TRUE(c.has_value()) << c.error().ToString();
    EXPECT_EQ(c->container, "mlx");
    EXPECT_EQ(c->quantization, "mlx affine");
  }
  if (!Exists(transformer) && !Exists(mlx)) {
    GTEST_SKIP() << "no diffusers or MLX checkpoint";
  }
}

// ---------------------------------------------------------------- byte checks

// A v0 resource's bytes and a checkpoint tensor's are equal, read in
// chunks (never the bytes themselves in a failure). Each chunk read is
// dropped from the page cache after it is compared: on the GB10 the cache
// shares the GPU's memory, and tens of GB of it would starve the GPU
// tests ctest runs beside this one (their startup guards count free
// device memory).
testing::AssertionResult SameResourceBytes(const std::filesystem::path& root,
                                           const artifact::Artifact& a, std::uint32_t resource,
                                           const Checkpoint& c, const Tensor& t) {
  const auto p = a.ResourcePlacement(resource);
  if (!p) {
    return testing::AssertionFailure() << "no placement";
  }
  if (p->bytes.value() != t.extent.bytes) {
    return testing::AssertionFailure() << "sizes " << p->bytes.value() << " and " << t.extent.bytes;
  }
  const artifact::Group& g = a.groups()[p->group];
  const artifact::Shard& shard = a.shards()[g.shard];
  const artifact::FileDescriptor v0(::open((root / shard.path).c_str(), O_RDONLY | O_CLOEXEC));
  if (!v0.valid()) {
    return testing::AssertionFailure() << "cannot open a v0 shard";
  }
  const std::uint64_t base = shard.data_offset.value() + g.offset.value() + p->offset.value();
  // The source file, for dropping what was read from the page cache.
  const artifact::FileDescriptor source(
      ::open(c.files.at(t.extent.file).path.c_str(), O_RDONLY | O_CLOEXEC));
  constexpr std::uint64_t kChunk = std::uint64_t{64} << 20U;
  std::vector<std::byte> want;
  std::vector<std::byte> got;
  for (std::uint64_t at = 0; at < t.extent.bytes; at += kChunk) {
    const std::uint64_t n = std::min(kChunk, t.extent.bytes - at);
    want.resize(n);
    got.resize(n);
    if (::pread(v0.get(), want.data(), n, static_cast<off_t>(base + at)) !=
        static_cast<ssize_t>(n)) {
      return testing::AssertionFailure() << "short v0 read";
    }
    if (auto ok = c.Read(t.extent, at, got); !ok) {
      return testing::AssertionFailure() << ok.error().ToString();
    }
    ::posix_fadvise(v0.get(), static_cast<off_t>(base + at), static_cast<off_t>(n),
                    POSIX_FADV_DONTNEED);
    ::posix_fadvise(source.get(), static_cast<off_t>(t.extent.offset + at), static_cast<off_t>(n),
                    POSIX_FADV_DONTNEED);
    const auto [x, y] = std::ranges::mismatch(want, got);
    if (x != want.end()) {
      return testing::AssertionFailure()
             << "first difference at byte " << (at + static_cast<std::uint64_t>(x - want.begin()));
    }
  }
  return testing::AssertionSuccess();
}

// Every resource of a v0 artifact is the checkpoint tensor of each of its
// roles, byte for byte; returns the bytes compared.
std::uint64_t ExpectResourcesAreTheCheckpoints(const std::filesystem::path& root,
                                               const artifact::Artifact& a, const Checkpoint& c) {
  std::uint64_t bytes = 0;
  for (std::uint32_t r = 0; r < a.resources().size(); ++r) {
    for (const std::string& role : a.resources()[r].roles) {
      const Tensor* t = c.FindTensor(role);
      EXPECT_NE(t, nullptr) << role;
      if (t != nullptr) {
        EXPECT_TRUE(SameResourceBytes(root, a, r, c, *t)) << role;
        bytes += t->extent.bytes;
      }
    }
  }
  return bytes;
}

// ---------------------------------------------------------------- diffusers

// Qwen-Image-2.1's pipeline as the diffusers reader gives it against the
// v0 composition the Python importer made of the same directory (D-089):
// the same components and architectures, the kept files the composition
// keeps, and in each component every tensor the v0 artifact holds, in
// v0's format and with its bytes (all ~33 GB compared).
TEST(ImportPipelineModelsTest, QwenImageComponentsAreTheV0Imports) {
  const std::filesystem::path path = Models() / "models/Qwen/Qwen-Image-2.1@790c9263";
  const std::filesystem::path store = Models() / "m3-artifacts";
  const std::filesystem::path composition =
      store / "eca21baad38229e471a44cb2479d392ffcf745fb812e8a41668f336139fa1acd";
  if (!Exists(path) || !Exists(composition)) {
    GTEST_SKIP() << "no " << path << " or " << composition;
  }
  const auto c = ReadCheckpoint(path);
  ASSERT_TRUE(c.has_value()) << c.error().ToString();
  EXPECT_EQ(c->container, "diffusers");
  EXPECT_TRUE(c->tensors.empty());
  const auto v0 = artifact::OpenComposition(composition);
  ASSERT_TRUE(v0.has_value()) << v0.error().ToString();
  EXPECT_EQ(c->architecture, v0->architecture());
  // The kept files: model_index.json, the scheduler's and the processor's,
  // by name and size.
  std::map<std::string, std::uint64_t> kept;
  for (const SourceFile& f : c->files) {
    kept.emplace(std::filesystem::path(f.name).filename().string(), f.bytes);
  }
  std::map<std::string, std::uint64_t> v0_kept;
  for (const artifact::Source& s : v0->sources()) {
    v0_kept.emplace(s.name, s.bytes.value());
  }
  EXPECT_EQ(kept, v0_kept);
  ASSERT_EQ(c->components.size(), v0->components().size());
  std::uint64_t compared = 0;
  for (std::size_t i = 0; i < c->components.size(); ++i) {
    const Checkpoint& component = c->components[i];
    const artifact::CompositionComponent& want = v0->components()[i];
    SCOPED_TRACE(component.role);
    EXPECT_EQ(component.role, want.role);
    EXPECT_EQ(component.architecture, want.architecture);
    ExpectWellFormed(component);
    const std::filesystem::path root = store / want.artifact;
    auto a = artifact::Artifact::Open(root, {.expected_id = {}});
    ASSERT_TRUE(a.has_value()) << a.error().ToString();
    // Every shard the reader opened is a source v0 recorded, and every
    // file both have is of one size (v0 also kept files the reader does
    // not open, a generation config).
    for (const SourceFile& f : component.files) {
      const auto s = std::ranges::find(a->sources(), f.name, &artifact::Source::name);
      if (s != a->sources().end()) {
        EXPECT_EQ(f.bytes, s->bytes.value()) << f.name;
      } else {
        EXPECT_FALSE(f.name.ends_with(".safetensors")) << f.name;
      }
    }
    const auto source = spec::SourceOfV0(*a);
    ASSERT_TRUE(source.has_value()) << source.error();
    EXPECT_EQ(source->tensors.size(), component.tensors.size());
    for (const spec::SourceTensor& t : source->tensors) {
      const Tensor* got = component.FindTensor(t.name);
      ASSERT_NE(got, nullptr) << t.name;
      EXPECT_EQ(got->format, t.format) << t.name;
      EXPECT_EQ(got->part, t.part) << t.name;
    }
    compared += ExpectResourcesAreTheCheckpoints(root, *a, component);
  }
  RecordProperty("bytes_compared", std::to_string(compared));
}

// ---------------------------------------------------------------- EXL3

struct Exl3Case {
  std::string_view name;
  std::string_view source;
  std::string_view artifact;
};

class ImportExl3ModelsTest : public testing::TestWithParam<Exl3Case> {};

// The Qwen2.5 EXL3 fixtures' source checkpoints as the EXL3 reader gives
// them against their v0 imports: every linear (169) a trellis whose parts
// have the format v0's resources assemble to, every other tensor v0's
// format, and every resource's bytes.
TEST_P(ImportExl3ModelsTest, FixturesAreTheirV0Imports) {
  const Exl3Case& e = GetParam();
  const std::filesystem::path path = Models() / e.source;
  const std::filesystem::path root = Models() / e.artifact;
  if (!Exists(path) || !Exists(root)) {
    GTEST_SKIP() << "no " << path << " or " << root;
  }
  const auto c = ReadCheckpoint(path);
  ASSERT_TRUE(c.has_value()) << c.error().ToString();
  EXPECT_EQ(c->container, "exl3");
  EXPECT_EQ(c->architecture, "qwen2");
  ExpectWellFormed(*c);
  auto a = artifact::Artifact::Open(root, {.expected_id = {}});
  ASSERT_TRUE(a.has_value()) << a.error().ToString();
  for (const SourceFile& f : c->files) {
    const auto s = std::ranges::find(a->sources(), f.name, &artifact::Source::name);
    ASSERT_NE(s, a->sources().end()) << f.name;
    EXPECT_EQ(s->bytes.value(), f.bytes) << f.name;
  }
  std::size_t trellises = 0;
  for (const artifact::Resource& r : a->resources()) {
    const Tensor* t = c->FindTensor(r.name);
    ASSERT_NE(t, nullptr) << r.name;
    EXPECT_EQ(t->part, artifact::V0PartRole(r.repr)) << r.name;
    if (r.repr.family != artifact::Family::kExl3) {
      EXPECT_EQ(t->format, artifact::FormatOf(r.repr).value()) << r.name;
      continue;
    }
    if (r.repr.role != "trellis") {
      continue;  // checked with its trellis
    }
    ++trellises;
    const std::string module =
        r.name.substr(0, r.name.size() - std::string_view(".trellis").size());
    std::vector<artifact::V0Part> parts = {{format::PartRole::kCodes, &r.repr}};
    for (const auto& [suffix, role] : {std::pair{".suh", format::PartRole::kInputSigns},
                                       std::pair{".svh", format::PartRole::kOutputSigns},
                                       std::pair{".mcg", format::PartRole::kCodebookMarker}}) {
      const auto i = a->FindResource(module + suffix);
      ASSERT_TRUE(i.has_value()) << module << suffix;
      parts.push_back({role, &a->resources()[*i].repr});
    }
    const auto v0 =
        artifact::Assemble(format::Encoding::kExl3Trellis, format::Layout::kExl3Tiles, parts);
    ASSERT_TRUE(v0.has_value()) << module;
    for (const std::string_view suffix : {".trellis", ".suh", ".svh", ".mcg"}) {
      const Tensor* part = c->FindTensor(module + std::string(suffix));
      ASSERT_NE(part, nullptr) << module << suffix;
      EXPECT_EQ(part->format, *v0) << module << suffix;
    }
  }
  EXPECT_EQ(trellises, 169U);
  EXPECT_EQ(a->resources().size(), c->tensors.size());
  RecordProperty("bytes_compared", std::to_string(ExpectResourcesAreTheCheckpoints(root, *a, *c)));
}

INSTANTIATE_TEST_SUITE_P(
    Fixtures, ImportExl3ModelsTest,
    testing::Values(
        Exl3Case{.name = "exl3_4_0bpw",
                 .source = "exl3-reference-20260922/models/4.0bpw",
                 .artifact = "artifact-layout-20260922/installed/"
                             "6e96e499ea9b859ea726327d4f82fb1739cb116ce4e9d3541d1ba67506efb4b4"},
        Exl3Case{.name = "exl3_4_5bpw",
                 .source = "exl3-reference-20260922/models/4.5bpw",
                 .artifact = "artifact-layout-20260922/installed/"
                             "00d77caf056c45b6fa05ebdbeff8b38da79d0556a2fd4f1b661e7e21ef9ed83a"}),
    [](const testing::TestParamInfo<Exl3Case>& info) { return std::string(info.param.name); });

// ---------------------------------------------------------------- MLX

// Every MLX module with scales is paired: codes, scales and biases in
// one affine format (4 bits, group 32, BF16 scales), the switch_mlp ones
// as expert arrays.
TEST(ImportMlxModelsTest, Qwen38MlxPairsEveryWeight) {
  const std::filesystem::path path = Models() / kMlx;
  if (!Exists(path)) {
    GTEST_SKIP() << "no " << path;
  }
  const auto c = ReadCheckpoint(path);
  ASSERT_TRUE(c.has_value()) << c.error().ToString();
  EXPECT_EQ(c->container, "mlx");
  EXPECT_EQ(c->architecture, "qwen4_exp");
  EXPECT_EQ(c->quantization, "mlx affine");
  ExpectWellFormed(*c);
  std::size_t scales = 0;
  std::size_t experts = 0;
  std::map<format::PartRole, std::size_t> parts;
  for (const Tensor& t : c->tensors) {
    if (t.name.ends_with(".scales") &&
        c->FindTensor(t.name.substr(0, t.name.size() - 7) + ".weight") != nullptr) {
      ++scales;
    }
    if (t.format.encoding != format::Encoding::kAffine) {
      continue;
    }
    ++parts[t.part];
    experts += t.experts && t.part == format::PartRole::kCodes ? 1 : 0;
    EXPECT_EQ(t.format.params.bits, 4) << t.name;
    EXPECT_EQ(t.format.params.group, 32U) << t.name;
    EXPECT_EQ(t.format.params.scale, format::AffineScale::kBF16) << t.name;
  }
  EXPECT_EQ(parts[format::PartRole::kCodes], scales);
  EXPECT_EQ(parts[format::PartRole::kBlockScales], scales);
  EXPECT_EQ(parts[format::PartRole::kBiases], scales);
  // gate, up and down of each of 48 layers and the MTP layer.
  EXPECT_EQ(experts, 49U * 3U);
  const Tensor* down =
      c->FindTensor("language_model.model.layers.1.mlp.switch_mlp.down_proj.weight");
  ASSERT_NE(down, nullptr);
  EXPECT_EQ(down->format.shape, (format::Shape{.ne = {640, 2560, 512, 1}, .rank = 3}));
  RecordProperty("tensors", static_cast<int>(c->tensors.size()));
  RecordProperty("quantized", static_cast<int>(scales));
}

// Rows [first, first + count) of matrix `matrix` of the logical tensor
// whose parts are `names`, decoded by the reference from just those rows'
// bytes.
std::vector<float> DecodeSlice(const Checkpoint& c, std::initializer_list<std::string> names,
                               std::uint64_t matrix, std::uint64_t first, std::uint64_t count) {
  const Tensor* any = c.FindTensor(*names.begin());
  EXPECT_NE(any, nullptr) << *names.begin();
  if (any == nullptr) {
    return {};
  }
  const format::TensorFormat whole = any->format;
  format::TensorFormat slice = whole;
  slice.shape.ne[1] = count;
  slice.shape.ne[2] = 1;
  slice.shape.ne[3] = 1;
  slice.shape.rank = 2;
  std::vector<std::vector<std::byte>> bytes;
  std::vector<kernels::reference::PartView> views;
  bytes.reserve(names.size());
  for (const std::string& name : names) {
    const Tensor* t = c.FindTensor(name);
    EXPECT_NE(t, nullptr) << name;
    if (t == nullptr) {
      return {};
    }
    EXPECT_EQ(t->format, whole) << name;
    std::uint64_t offset = matrix * 4;
    std::uint64_t n = 4;
    if (t->part != format::PartRole::kTensorScale) {
      const std::uint64_t row = format::RowBytes(whole, t->part).value();
      offset = (matrix * (t->extent.bytes / format::Matrices(whole))) + (first * row);
      n = count * row;
    }
    bytes.emplace_back(n);
    const auto ok = c.Read(t->extent, offset, bytes.back());
    EXPECT_TRUE(ok.has_value()) << name;
    views.push_back({.role = t->part, .bytes = bytes.back()});
  }
  std::vector<float> out(count * whole.shape.ne[0]);
  const auto ok = kernels::reference::DecodeRows(slice, views, 0, 0, count, out);
  EXPECT_TRUE(ok.has_value()) << *names.begin();
  return out;
}

double RelativeRms(const std::vector<float>& got, const std::vector<float>& want) {
  double diff = 0;
  double norm = 0;
  for (std::size_t i = 0; i < got.size() && i < want.size(); ++i) {
    diff += (static_cast<double>(got[i]) - want[i]) * (static_cast<double>(got[i]) - want[i]);
    norm += static_cast<double>(want[i]) * want[i];
  }
  return norm > 0 ? std::sqrt(diff / norm) : 0.0;
}

// The MLX checkpoint's weights decoded by the reference against the same
// weights of Qwen3.8's native checkpoint (both quantized from one set of
// BF16 weights). Where the native one is BF16 (the head, the token
// table), every MLX value is within one step s of it, plus the BF16
// rounding of the stored scale and bias (15 · |s| · 2^-8 + |b| · 2^-8):
// mx.quantize keeps zero exact (scale = edge / round(edge / scale),
// mlx/backend/cpu/quantized.cpp), which can leave a group's far end up to
// one step short and clamp it there, so its documented s / 2 is s. Where
// the native one is MXFP8 or NVFP4 too, the two agree within 0.15
// relative RMS, both quantizations' noise (MLX's own against BF16 is
// about 0.09 here).
TEST(ImportMlxModelsTest, DecodesLikeTheNativeCheckpoint) {
  const std::filesystem::path mlx_path = Models() / kMlx;
  const std::filesystem::path native_path = Models() / kQwen38;
  if (!Exists(mlx_path) || !Exists(native_path)) {
    GTEST_SKIP() << "no " << mlx_path << " or " << native_path;
  }
  const auto mlx = ReadCheckpoint(mlx_path);
  ASSERT_TRUE(mlx.has_value()) << mlx.error().ToString();
  const auto native = ReadCheckpoint(native_path);
  ASSERT_TRUE(native.has_value()) << native.error().ToString();
  const std::string m = "language_model.";
  const std::string n = "model.language_model.";
  // BF16 natives: the bound per element.
  struct Dense {
    std::string mlx, native;
    std::uint64_t first, count;
  };
  for (const Dense& d : {Dense{m + "lm_head", "lm_head.weight", 0, 256},
                         Dense{m + "lm_head", "lm_head.weight", 248320 - 64, 64},
                         Dense{m + "model.embed_tokens", n + "embed_tokens.weight", 1000, 256}}) {
    SCOPED_TRACE(d.mlx);
    const std::vector<float> got = DecodeSlice(
        *mlx, {d.mlx + ".weight", d.mlx + ".scales", d.mlx + ".biases"}, 0, d.first, d.count);
    const std::vector<float> want = DecodeSlice(*native, {d.native}, 0, d.first, d.count);
    // The stored scales and biases of these rows.
    const Tensor* s = mlx->FindTensor(d.mlx + ".scales");
    const Tensor* b = mlx->FindTensor(d.mlx + ".biases");
    ASSERT_TRUE(s != nullptr && b != nullptr);
    const std::uint64_t row = format::RowBytes(s->format, format::PartRole::kBlockScales).value();
    std::vector<std::byte> sb(d.count * row);
    std::vector<std::byte> bb(d.count * row);
    ASSERT_TRUE(mlx->Read(s->extent, d.first * row, sb).has_value());
    ASSERT_TRUE(mlx->Read(b->extent, d.first * row, bb).has_value());
    const std::uint64_t k = s->format.shape.ne[0];
    ASSERT_EQ(got.size(), want.size());
    double worst = 0;
    for (std::size_t i = 0; i < got.size(); ++i) {
      const std::size_t g = (i / k * (k / 32)) + ((i % k) / 32);
      std::uint16_t sv = 0;
      std::uint16_t bv = 0;
      std::memcpy(&sv, sb.data() + (2 * g), 2);
      std::memcpy(&bv, bb.data() + (2 * g), 2);
      const double scale = std::abs(kernels::reference::Bf16ToF32(sv));
      const double bias = std::abs(kernels::reference::Bf16ToF32(bv));
      const double bound = scale + (15 * scale / 256) + (bias / 256) + 1e-12;
      worst = std::max(worst, std::abs(static_cast<double>(got[i]) - want[i]) / bound);
    }
    EXPECT_LE(worst, 1.0);
    RecordProperty(std::format("{}_{}_worst_over_bound", d.mlx, d.first), std::to_string(worst));
    RecordProperty(std::format("{}_{}_relative_rms", d.mlx, d.first),
                   std::to_string(RelativeRms(got, want)));
  }
  // Quantized natives: MXFP8 (the shared expert, a full-attention q_proj)
  // and NVFP4 with a tensor scale (routed expert 7).
  const std::string se = "layers.1.mlp.shared_expert.gate_proj";
  const std::vector<float> se_got = DecodeSlice(
      *mlx,
      {m + "model." + se + ".weight", m + "model." + se + ".scales", m + "model." + se + ".biases"},
      0, 0, 640);
  const std::vector<float> se_want =
      DecodeSlice(*native, {n + se + ".weight", n + se + ".weight_scale"}, 0, 0, 640);
  const std::string q = "layers.3.self_attn.q_proj";
  const std::vector<float> q_got = DecodeSlice(
      *mlx,
      {m + "model." + q + ".weight", m + "model." + q + ".scales", m + "model." + q + ".biases"}, 0,
      0, 512);
  const std::vector<float> q_want =
      DecodeSlice(*native, {n + q + ".weight", n + q + ".weight_scale"}, 0, 0, 512);
  const std::string sw = m + "model.layers.1.mlp.switch_mlp.down_proj";
  const std::string ex = n + "layers.1.mlp.experts.7.down_proj";
  const std::vector<float> ex_got =
      DecodeSlice(*mlx, {sw + ".weight", sw + ".scales", sw + ".biases"}, 7, 0, 256);
  const std::vector<float> ex_want = DecodeSlice(
      *native, {ex + ".weight", ex + ".weight_scale", ex + ".weight_scale_2"}, 0, 0, 256);
  const double se_rms = RelativeRms(se_got, se_want);
  const double q_rms = RelativeRms(q_got, q_want);
  const double ex_rms = RelativeRms(ex_got, ex_want);
  RecordProperty("shared_expert_relative_rms", std::to_string(se_rms));
  RecordProperty("q_proj_relative_rms", std::to_string(q_rms));
  RecordProperty("expert7_relative_rms", std::to_string(ex_rms));
  EXPECT_LT(se_rms, 0.15);
  EXPECT_LT(q_rms, 0.15);
  EXPECT_LT(ex_rms, 0.15);
}

TEST(ImportGgufModelsTest, NormalizesClosedQwen38TargetHeadersWithoutPayloadReads) {
  // An explicitly scheduled source case must refuse missing input, not skip.
  // NOLINTNEXTLINE(concurrency-mt-unsafe): immutable launcher input
  const char* selected = std::getenv("LLMP_TEST_QWEN38_GGUF");
  const auto path =
      selected != nullptr
          ? std::filesystem::path(selected)
          : Models() /
                "reference-models/Q/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf";
  ASSERT_TRUE(Exists(path));
  auto source = ReadGguf(path);
  ASSERT_TRUE(source) << test_support::Failed(source, &Error::ToString).value_or("");
  ASSERT_EQ(source->files.size(), 3U);
  EXPECT_EQ(source->files[0].bytes, 10946624U);
  EXPECT_EQ(source->files[1].bytes, 49567921344ULL);
  EXPECT_EQ(source->files[2].bytes, 32382955968ULL);
  constexpr std::array<std::uint64_t, 3> header_bytes{10946618, 29604, 48297};
  constexpr std::array<std::string_view, 3> header_hashes{
      "7c1460a95eb2e95377c32b14deb92cf74fe504d4feb5a62a40d48f78e9e0dcb8",
      "7744187256a32b79476b0c443b912a887160258793dddf86782608874b31de64",
      "77ceba7e90d26ca4e93b4e39b96573e52d69022c32369c0cd0ed441f3eac4c0f"};
  for (std::size_t i = 0; i < source->files.size(); ++i) {
    EXPECT_EQ(source->files[i].parsed.bytes, header_bytes[i]);
    EXPECT_EQ(base::ToHex(source->files[i].parsed.sha256), header_hashes[i]);
  }
  ASSERT_EQ(source->tensors.size(), 1224U);
  const auto fixture = test_support::Qwen38GgufCheckpoint();
  for (const auto& tensor : source->tensors) {
    const auto* expected = fixture.FindTensor(tensor.name);
    ASSERT_NE(expected, nullptr) << tensor.name;
    EXPECT_EQ(tensor.format, expected->format) << tensor.name;
    EXPECT_EQ(tensor.dims, expected->dims) << tensor.name;
    EXPECT_EQ(tensor.experts, expected->experts) << tensor.name;
  }
  auto cost = NormalizationBound(*source);
  ASSERT_TRUE(cost) << test_support::Failed(cost, &Error::ToString).value_or("");
  ASSERT_LT(cost->metadata_bytes, std::uint64_t{512} << 20U);
  auto normalized = Normalize(std::move(*source), {cost->metadata_bytes, cost->work_units});
  ASSERT_TRUE(normalized) << test_support::Failed(normalized, &Error::ToString).value_or("");
  ASSERT_EQ(normalized->routes().size(), 1224U);
  for (std::size_t i = 0; i < normalized->routes().size(); ++i) {
    EXPECT_EQ(normalized->routes()[i].source_tensor, i);
    EXPECT_EQ(normalized->routes()[i].extent, normalized->source().tensors[i].extent);
  }
  std::size_t selected_keys = 0;
  for (const auto& entry : normalized->metadata())
    selected_keys += entry.classification == MetadataClass::kSelected;
  EXPECT_EQ(normalized->spec().stages[0].side.size(), 1U);
  EXPECT_EQ(normalized->applied_folds().size(), 3U);
  RecordProperty("normalized_tensors", "1224");
  RecordProperty("source_files", "3");
  RecordProperty("selected_keys", std::to_string(selected_keys));
  RecordProperty("normalization_metadata_bound", std::to_string(cost->metadata_bytes));
  RecordProperty("source_payload_bytes", std::to_string(cost->payload_bytes));
  RecordProperty("scope", "actual headers only; no payload hash, publication or execution");
}

}  // namespace
}  // namespace llmp::import
