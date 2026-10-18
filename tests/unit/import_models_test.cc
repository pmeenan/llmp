// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The container readers (import/) on the real checkpoints of the current
// models: the Gemma 2 and Gemma 3 source GGUFs give, through their mappings
// and their own metadata, the spec their published v1 artifacts store,
// tensor for tensor in the same formats, and the GGUFs of Gemma 4, its
// assistants, DeepSeek V4, DSpark and Qwen3.8 the formats of theirs; the
// split and single GGUFs read whole; Qwen3.8's ModelOpt checkpoint pairs
// every NVFP4 and MXFP8 weight; Qwen-Image-2.1's pipeline is its v0
// composition's components tensor for tensor and byte for byte; the Qwen2.5
// EXL3 fixtures are their v1 imports; Qwen3.8's MLX checkpoint pairs every
// quantized module and decodes within its quantization of the native
// checkpoint's weights; and the safetensors reader leaves an MLX
// checkpoint to its reader. Needs the model files (label `models`); each
// case is skipped without its files; the explicitly scheduled closed
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
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <initializer_list>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <variant>
#include <vector>

#include "artifact/artifact.h"
#include "artifact/composition.h"
#include "artifact/format_json.h"
#include "artifact/format_map.h"
#include "artifact/layout.h"
#include "base/json.h"
#include "base/sha256.h"
#include "expected_error.h"
#include "fixture_source.h"
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
#include "qwen38_gguf_normalization_fixture.h"
#include "spec/mapping.h"
#include "spec/spec.h"
#include "spec/stored.h"
#include "v1_frozen.h"

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

std::string InventoryDigest(const spec::Source& source) {
  std::map<std::string, const spec::SourceTensor*, std::less<>> ordered;
  for (const auto& tensor : source.tensors) ordered.emplace(tensor.name, &tensor);
  base::Sha256 hash;
  for (const auto& [name, tensor] : ordered) {
    std::string format;
    artifact::AppendFormatJson(tensor->format, format);
    hash.Update(name).Update("\n").Update(format);
    hash.Update(tensor->experts ? "experts\n" : "dense\n");
    hash.Update(format::PartRoleName(tensor->part)).Update("\n");
  }
  return base::ToHex(hash.Finish());
}

void ReportNormalized(std::string_view name, const NormalizedPlan& plan) {
  const auto spec = base::ToHex(plan.spec().Digest());
  const auto inventory = InventoryDigest(plan.canonical());
  std::printf("normalized %.*s spec=%s inventory=%s\n", static_cast<int>(name.size()), name.data(),
              spec.c_str(), inventory.c_str());
}

// Uses pinned real config/index bytes, but generates tensor descriptors only.
// It never opens or invents the absent 27B weight payloads.
TEST(ImportGemmaMetadataTest, Official27BConfigsAndIndexNamesProduceTheDeclaredTextSpecs) {
  // NOLINTNEXTLINE(concurrency-mt-unsafe): tests read, never set, the environment
  const char* metadata_root = std::getenv("LLMP_TEST_GEMMA27_METADATA");
  const auto root =
      std::filesystem::path(metadata_root ? metadata_root : "/mnt/llm/checkpoints/hf/google");
  for (const bool gemma3 : {false, true}) {
    const auto directory =
        root / (gemma3 ? "gemma-3-27b-it@005ad3404e59d6023443cb575daa05336842228a"
                       : "gemma-2-27b-it@aaf20e6b9f4c0fcf043f6fb2a2068419086d77b0");
    if (!Exists(directory / "config.json") || !Exists(directory / "model.safetensors.index.json")) {
      // NOLINTNEXTLINE(concurrency-mt-unsafe): tests read, never set, the environment
      ASSERT_EQ(std::getenv("LLMP_REQUIRE_GEMMA27_METADATA"), nullptr) << directory;
      GTEST_SKIP() << "pinned Gemma 27B config/index metadata is absent: " << directory;
    }
    const auto read = [&](std::string_view name) {
      auto file = OpenCheckpointFile(directory / name, 0);
      EXPECT_TRUE(file);
      return file ? ReadDocument(*file, 2ULL << 20U, 0)
                  : std::expected<std::string, Error>(std::unexpected(file.error()));
    };
    auto config_text = read("config.json");
    auto index_text = read("model.safetensors.index.json");
    ASSERT_TRUE(config_text);
    ASSERT_TRUE(index_text);
    EXPECT_EQ(base::ToHex(base::Sha256().Update(*config_text).Finish()),
              gemma3 ? "cabd884f5e0d4f01a5bd7fe14bd4bacd0bd83f3725a02acbdc4e72dc001835fa"
                     : "8e560335c5f3156a7340edd8087e9a23339b71072338016f46f3f5ef89b8eb05");
    EXPECT_EQ(base::ToHex(base::Sha256().Update(*index_text).Finish()),
              gemma3 ? "71567cc4dcf16476da69af42108fb6b30b24f6cf49a4f71f9917da57715f74c5"
                     : "0912fde150e64f7ea677c29e73dd59f7aaadbfac8cbb338745ce93321572ce43");
    auto config = base::json::Parse(*config_text);
    auto index = base::json::Parse(*index_text);
    ASSERT_TRUE(config);
    ASSERT_TRUE(index);
    const auto model_config_root = config->root();
    const auto text = gemma3 ? *model_config_root.find("text_config") : model_config_root;
    const auto integer = [&](std::string_view key) {
      return static_cast<std::uint64_t>(*text.find(key)->int64());
    };
    const auto width = integer("hidden_size"), hidden = integer("intermediate_size");
    const auto heads = integer("num_attention_heads"), kv = integer("num_key_value_heads");
    const auto dim = integer("head_dim"), layers = integer("num_hidden_layers");
    // The official Gemma3 config omits this architecture-default vocabulary.
    const auto vocabulary = gemma3 ? 262208ULL : integer("vocab_size");
    Checkpoint source{.container = "safetensors",
                      .architecture = gemma3 ? "gemma3" : "gemma2",
                      .files = {{.name = "descriptor-only"}}};
    auto metadata = FlattenMetadata(model_config_root);
    ASSERT_TRUE(metadata);
    source.metadata = std::move(*metadata);
    const auto* table = FindSourceMapping(source.container, source.architecture);
    ASSERT_NE(table, nullptr);
    const auto weights = index->root().find("weight_map");
    ASSERT_TRUE(weights && weights->is_object());
    for (std::size_t i = 0; i < weights->size(); ++i) {
      const auto name = weights->key(i);
      const auto alias = std::ranges::find_if(table->tensors, [&](const auto& candidate) {
        return normalization_detail::Match(candidate.source, name,
                                           static_cast<std::uint32_t>(layers))
            .has_value();
      });
      std::vector<std::uint64_t> shape{1};
      if (alias == table->tensors.end()) {
        ASSERT_TRUE(std::ranges::any_of(table->ignored_tensor_prefixes, [&](auto prefix) {
          return name.starts_with(prefix);
        })) << name;
      } else {
        const auto role = std::ranges::find(table->architecture->tensors, alias->canonical,
                                            &spec::TensorRule::pattern);
        ASSERT_NE(role, table->architecture->tensors.end());
        if (role->slot == "embedding" || role->slot == "output")
          shape = {width, vocabulary};
        else if (role->slot == "final_norm")
          shape = {width};
        else if (role->role == "norm" || role->role == "post_norm")
          shape = {width};
        else if (role->role == "q_norm" || role->role == "k_norm")
          shape = {dim};
        else if (role->slot == "ffn")
          shape = role->role == "down" ? std::vector<std::uint64_t>{hidden, width}
                                       : std::vector<std::uint64_t>{width, hidden};
        else if (role->role == "q")
          shape = {width, heads * dim};
        else if (role->role == "k" || role->role == "v")
          shape = {width, kv * dim};
        else if (role->role == "output")
          shape = {heads * dim, width};
        else
          FAIL() << name;
      }
      format::TensorFormat descriptor{.encoding = format::Encoding::kBF16,
                                      .layout = format::Layout::kDense,
                                      .shape = *format::ShapeOf(shape)};
      const auto bytes = (*format::PartsOf(descriptor))[0].bytes;
      const auto offset = source.files.front().bytes;
      std::ranges::reverse(shape);
      source.tensors.push_back({.name = std::string(name),
                                .format = descriptor,
                                .extent = {.offset = offset, .bytes = bytes},
                                .dtype = "BF16",
                                .dims = std::move(shape)});
      source.files.front().bytes += bytes;
    }
    auto plan = Normalize(std::move(source), {2ULL << 30U, 1ULL << 45U});
    ASSERT_TRUE(plan) << test_support::Failed(plan, &Error::ToString).value_or("");
    const auto& stage = plan->spec().stages.front();
    EXPECT_EQ(stage.layers.size(), layers);
    EXPECT_EQ(stage.context, gemma3 ? 131072U : 8192U);
    EXPECT_EQ(stage.stream.width, width);
    EXPECT_EQ(plan->canonical().tensors.size(), gemma3 ? 808U : 508U);
    const auto& attention = stage.layers.front().components.front();
    EXPECT_DOUBLE_EQ(std::get<double>(attention.Find("scale")->value),
                     1.0 / std::sqrt(gemma3 ? 168.0 : 144.0));
    EXPECT_DOUBLE_EQ(std::get<double>(attention.Find("eps")->value), 1e-6);
    if (gemma3) {
      const auto& global = stage.layers[5].components.front();
      EXPECT_EQ(std::get<double>(global.Find("rope_base")->value), 1000000.0);
      EXPECT_EQ(std::get<double>(global.Find("rope_scale")->value), 0.125);
    }
  }
}

struct GemmaCase {
  std::string_view architecture;
  std::string_view gguf;
  std::string_view artifact;  // the store's v1-ids.json name
  std::size_t tensors;
};

class ImportGemmaTest : public testing::TestWithParam<GemmaCase> {};

TEST_P(ImportGemmaTest, SourceGgufGivesTheV1ArtifactsSpec) {
  const GemmaCase& c = GetParam();
  const std::filesystem::path gguf = Models() / c.gguf;
  const auto id = test_support::V1IdNamed(c.artifact);
  if (!Exists(gguf) || !id) {
    GTEST_SKIP() << "no " << gguf << " or published v1 " << c.artifact << " in "
                 << test_support::V1Store();
  }
  const std::filesystem::path path = std::filesystem::path(test_support::V1Store()) / *id;
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

  // Tensor for tensor, the v1 artifact's formats.
  const auto v1 = spec::SourceOf(*artifact);
  ASSERT_TRUE(v1.has_value()) << v1.error();
  const spec::Source source = SourceOf(*checkpoint);
  EXPECT_EQ(source.container, v1->container);
  ASSERT_EQ(source.tensors.size(), v1->tensors.size());
  std::map<std::string, const spec::SourceTensor*> by_name;
  for (const spec::SourceTensor& t : source.tensors) {
    by_name.emplace(t.name, &t);
  }
  for (const spec::SourceTensor& t : v1->tensors) {
    const auto found = by_name.find(t.name);
    ASSERT_NE(found, by_name.end()) << t.name;
    EXPECT_EQ(found->second->format, t.format) << t.name;
    EXPECT_EQ(found->second->part, t.part) << t.name;
    EXPECT_EQ(found->second->experts, t.experts) << t.name;
  }

  // The spec from the checkpoint's own metadata is the one the v1 artifact
  // stores.
  const spec::Mapping* mapping = spec::FindMapping(checkpoint->architecture);
  ASSERT_NE(mapping, nullptr);
  const auto stored = spec::StoredSpec(*artifact);
  ASSERT_TRUE(stored.has_value()) << stored.error();
  const auto from_gguf = spec::Synthesize(*mapping, source, ValuesOf(*checkpoint));
  ASSERT_TRUE(from_gguf.has_value()) << from_gguf.error();
  EXPECT_EQ(*from_gguf, *stored);
  EXPECT_EQ(spec::ToJson(*from_gguf), spec::ToJson(*stored));
  RecordProperty("metadata_keys", static_cast<int>(checkpoint->metadata.size()));
  const auto bound = NormalizationBound(*checkpoint);
  ASSERT_TRUE(bound) << test_support::Failed(bound, &Error::ToString).value_or("");
  auto normalized = Normalize(std::move(*checkpoint), {bound->metadata_bytes, bound->work_units});
  ASSERT_TRUE(normalized) << test_support::Failed(normalized, &Error::ToString).value_or("");
  EXPECT_EQ(normalized->spec(), *stored);
  EXPECT_EQ(InventoryDigest(normalized->canonical()), InventoryDigest(*v1));
  ReportNormalized(c.artifact, *normalized);
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
                  .artifact = "gemma2",
                  .tensors = 288},
        GemmaCase{.architecture = "gemma3",
                  .gguf = "models/ggml-org/gemma-3-4b-it-qat-GGUF@bbcac0d0/"
                          "gemma-3-4b-it-qat-Q4_0.gguf",
                  .artifact = "gemma3",
                  .tensors = 444}),
    [](const testing::TestParamInfo<GemmaCase>& info) {
      return std::string(info.param.architecture);
    });

INSTANTIATE_TEST_SUITE_P(Qwen2, ImportGemmaTest,
                         testing::Values(GemmaCase{
                             .architecture = "qwen2",
                             .gguf = "models/Qwen/Qwen2.5-0.5B-Instruct-GGUF@9217f5db/"
                                     "qwen2.5-0.5b-instruct-fp16.gguf",
                             .artifact = "qwen2-fp16",
                             .tensors = 291}),
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

struct V1Case {
  std::string_view name;
  std::string_view artifact;  // the store's v1-ids.json name
  std::string_view gguf;      // its source, or the first of its splits
};

class ImportV1FormatsTest : public testing::TestWithParam<V1Case> {};

// Every tensor of a v1 artifact imported from GGUF has the format the
// reader gives the source's tensor of that name (expert arrays as the
// GGUF's three-dimensional expert tensors), but those its import folded or
// relaid out, which it records by name.
TEST_P(ImportV1FormatsTest, SourceFormatsAreTheArtifacts) {
  const V1Case& c = GetParam();
  const std::filesystem::path gguf = Models() / c.gguf;
  const auto id = test_support::V1IdNamed(c.artifact);
  if (!Exists(gguf) || !id) {
    GTEST_SKIP() << "no " << gguf << " or published v1 " << c.artifact << " in "
                 << test_support::V1Store();
  }
  const std::filesystem::path path = std::filesystem::path(test_support::V1Store()) / *id;
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
  const auto v1 = spec::SourceOf(*artifact);
  ASSERT_TRUE(v1.has_value()) << v1.error();
  std::set<std::string> transformed;
  for (const artifact::AppliedFold& f : artifact->folds()) {
    transformed.insert(f.tensors.begin(), f.tensors.end());
  }
  for (const artifact::Relayout& r : artifact->relayouts()) {
    transformed.insert(r.tensors.begin(), r.tensors.end());
  }
  std::map<std::string, const Tensor*> by_name;
  for (const Tensor& t : checkpoint->tensors) {
    by_name.emplace(t.name, &t);
  }
  std::size_t compared = 0;
  for (const spec::SourceTensor& t : v1->tensors) {
    const auto found = by_name.find(t.name);
    ASSERT_NE(found, by_name.end()) << t.name;
    EXPECT_EQ(found->second->experts, t.experts) << t.name;
    if (transformed.contains(t.name)) {
      continue;
    }
    EXPECT_EQ(found->second->format, t.format) << t.name;
    ++compared;
  }
  if (const auto* table = FindSourceMapping(checkpoint->container, checkpoint->architecture)) {
    const auto bound = NormalizationBound(*checkpoint);
    ASSERT_TRUE(bound) << test_support::Failed(bound, &Error::ToString).value_or("");
    std::optional<NormalizationTarget> target;
    if (table->architecture->role == spec::StageRole::kDrafter) {
      const std::string_view target_name = c.artifact == "gemma26-assistant"   ? "gemma26"
                                           : c.artifact == "gemma31-assistant" ? "gemma31"
                                                                               : "deepseek";
      const auto target_id = test_support::V1IdNamed(target_name);
      ASSERT_TRUE(target_id) << target_name;
      auto opened_target =
          artifact::Artifact::Open(std::filesystem::path(test_support::V1Store()) / *target_id);
      ASSERT_TRUE(opened_target);
      auto target_model = spec::StoredSpec(*opened_target);
      auto target_source = spec::SourceOf(*opened_target);
      ASSERT_TRUE(target_model);
      ASSERT_TRUE(target_source);
      const auto* mapping =
          spec::FindMapping(target_source->architecture, target_source->container);
      target.emplace(
          NormalizationTarget{mapping, std::move(*target_model), std::move(*target_source)});
    }
    auto normalized = Normalize(*checkpoint, {bound->metadata_bytes, bound->work_units},
                                target ? &*target : nullptr);
    ASSERT_TRUE(normalized) << test_support::Failed(normalized, &Error::ToString).value_or("");
    if (table->architecture->role == spec::StageRole::kTarget) {
      const auto expected = spec::StoredSpec(*artifact);
      ASSERT_TRUE(expected) << test_support::Failed(expected).value_or("");
      EXPECT_EQ(normalized->spec(), *expected);
    } else {
      ASSERT_EQ(normalized->spec().stages.size(), 2U);
      EXPECT_EQ(normalized->spec().stages[1].role, spec::StageRole::kDrafter);
    }
    ReportNormalized(c.name, *normalized);
    EXPECT_EQ(InventoryDigest(normalized->canonical()), InventoryDigest(*v1));
    RecordProperty("normalized_spec_digest", base::ToHex(normalized->spec().Digest()));
    RecordProperty("normalized_inventory_digest", InventoryDigest(normalized->canonical()));
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
  RecordProperty("v1_tensors", static_cast<int>(v1->tensors.size()));
  RecordProperty("v1_tensors_compared", static_cast<int>(compared));
  RecordProperty("source_tensors", static_cast<int>(checkpoint->tensors.size()));
}

INSTANTIATE_TEST_SUITE_P(
    M3, ImportV1FormatsTest,
    testing::Values(V1Case{.name = "gemma4_26b",
                           .artifact = "gemma26",
                           .gguf = "reference-models/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"},
                    V1Case{.name = "gemma4_31b",
                           .artifact = "gemma31",
                           .gguf = "reference-models/gemma-4-31B-it-UD-Q4_K_XL.gguf"},
                    V1Case{.name = "gemma4_26b_assistant",
                           .artifact = "gemma26-assistant",
                           .gguf = "reference-models/mtp-gemma-4-26B-A4B-it.gguf"},
                    V1Case{.name = "gemma4_31b_assistant",
                           .artifact = "gemma31-assistant",
                           .gguf = "reference-models/mtp-gemma-4-31B-it.gguf"},
                    V1Case{.name = "deepseek_v4_ud_q2",
                           .artifact = "deepseek",
                           .gguf = "models/unsloth/DeepSeek-V4-Flash-0731-GGUF@fbbb5b93/UD-Q2_K_XL/"
                                   "DeepSeek-V4-Flash-0731-UD-Q2_K_XL-00001-of-00003.gguf"},
                    V1Case{.name = "dspark_q8",
                           .artifact = "dspark",
                           .gguf = "models/unsloth/DeepSeek-V4-Flash-0731-GGUF@fbbb5b93/"
                                   "dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf"},
                    V1Case{.name = "qwen38_ud_iq3",
                           .artifact = "qwen38-gguf",
                           .gguf = "reference-models/Q/UD-IQ3_XXS/"
                                   "Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf"}),
    [](const testing::TestParamInfo<V1Case>& info) { return std::string(info.param.name); });

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
  // config.json, hf_quant_config.json, the index and four kept tokenizer documents.
  EXPECT_EQ(checkpoint->files.size(), 35U + 3U + 4U);
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

// An artifact resource's bytes and a checkpoint tensor's are equal, read in
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
  const artifact::FileDescriptor stored(::open((root / shard.path).c_str(), O_RDONLY | O_CLOEXEC));
  if (!stored.valid()) {
    return testing::AssertionFailure() << "cannot open an artifact shard";
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
    if (::pread(stored.get(), want.data(), n, static_cast<off_t>(base + at)) !=
        static_cast<ssize_t>(n)) {
      return testing::AssertionFailure() << "short artifact read";
    }
    if (auto ok = c.Read(t.extent, at, got); !ok) {
      return testing::AssertionFailure() << ok.error().ToString();
    }
    ::posix_fadvise(stored.get(), static_cast<off_t>(base + at), static_cast<off_t>(n),
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

// Every resource of an artifact is the checkpoint tensor of each of its
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
    const auto source = test_support::V0ArtifactSource(*a);
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
  std::string_view artifact;  // the store's v1-ids.json name
};

class ImportExl3ModelsTest : public testing::TestWithParam<Exl3Case> {};

// The Qwen2.5 EXL3 fixtures' source checkpoints as the EXL3 reader gives
// them against their published v1 imports: every resource the checkpoint
// tensor of its name, in the format and part v1 records (every linear (169)
// a trellis of four parts), and with its bytes.
TEST_P(ImportExl3ModelsTest, FixturesAreTheirV1Imports) {
  const Exl3Case& e = GetParam();
  const std::filesystem::path path = Models() / e.source;
  const auto id = test_support::V1IdNamed(e.artifact);
  if (!Exists(path) || !id) {
    GTEST_SKIP() << "no " << path << " or published v1 " << e.artifact << " in "
                 << test_support::V1Store();
  }
  const std::filesystem::path root = std::filesystem::path(test_support::V1Store()) / *id;
  const auto c = ReadCheckpoint(path);
  ASSERT_TRUE(c.has_value()) << c.error().ToString();
  EXPECT_EQ(c->container, "exl3");
  EXPECT_EQ(c->architecture, "qwen2");
  ExpectWellFormed(*c);
  auto a = artifact::Artifact::Open(root, {.expected_id = {}});
  ASSERT_TRUE(a.has_value()) << a.error().ToString();
  for (const SourceFile& f : c->files) {
    const auto s = std::ranges::find(a->sources(), f.name, &artifact::Source::name);
    if (s == a->sources().end()) {
      // Old v1 imports predate preservation of optional tokenizer documents.
      constexpr std::array<std::string_view, 7> documents{
          "tokenizer.json",         "tokenizer_config.json", "special_tokens_map.json",
          "generation_config.json", "chat_template.jinja",   "chat_template.json",
          "tokenizer.model"};
      EXPECT_TRUE(std::ranges::contains(documents, f.name)) << f.name;
      EXPECT_EQ(f.parsed.bytes, f.bytes) << f.name;
      continue;
    }
    EXPECT_EQ(s->bytes.value(), f.bytes) << f.name;
  }
  std::size_t trellises = 0;
  for (const artifact::Resource& r : a->resources()) {
    const Tensor* t = c->FindTensor(r.name);
    ASSERT_NE(t, nullptr) << r.name;
    ASSERT_TRUE(r.format.has_value()) << r.name;
    EXPECT_EQ(t->part, r.part) << r.name;
    EXPECT_EQ(t->format, *r.format) << r.name;
    if (r.format->encoding == format::Encoding::kExl3Trellis &&
        r.part == format::PartRole::kCodes) {
      ++trellises;
    }
  }
  EXPECT_EQ(trellises, 169U);
  EXPECT_EQ(a->resources().size(), c->tensors.size());
  const auto bound = NormalizationBound(*c);
  ASSERT_TRUE(bound) << test_support::Failed(bound, &Error::ToString).value_or("");
  auto normalized = Normalize(*c, {bound->metadata_bytes, bound->work_units});
  ASSERT_TRUE(normalized) << test_support::Failed(normalized, &Error::ToString).value_or("");
  const auto stored = spec::StoredSpec(*a);
  ASSERT_TRUE(stored);
  EXPECT_EQ(normalized->spec(), *stored);
  const auto stored_source = spec::SourceOf(*a);
  ASSERT_TRUE(stored_source);
  EXPECT_EQ(InventoryDigest(normalized->canonical()), InventoryDigest(*stored_source));
  RecordProperty("normalized_spec_digest", base::ToHex(normalized->spec().Digest()));
  RecordProperty("normalized_inventory_digest", InventoryDigest(normalized->canonical()));
  ReportNormalized(e.name, *normalized);
  RecordProperty("bytes_compared", std::to_string(ExpectResourcesAreTheCheckpoints(root, *a, *c)));
}

INSTANTIATE_TEST_SUITE_P(Fixtures, ImportExl3ModelsTest,
                         testing::Values(Exl3Case{.name = "exl3_4_0bpw",
                                                  .source = "exl3-reference-20260922/models/4.0bpw",
                                                  .artifact = "qwen2-exl3-4.0"},
                                         Exl3Case{.name = "exl3_4_5bpw",
                                                  .source = "exl3-reference-20260922/models/4.5bpw",
                                                  .artifact = "qwen2-exl3-4.5"}),
                         [](const testing::TestParamInfo<Exl3Case>& info) {
                           return std::string(info.param.name);
                         });

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
  if (selected == nullptr && !Exists(path)) GTEST_SKIP() << "no staged Qwen3.8 GGUF: " << path;
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
