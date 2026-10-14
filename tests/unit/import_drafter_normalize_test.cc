// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <charconv>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "artifact/artifact.h"
#include "artifact_writer.h"
#include "base/check.h"
#include "base/json.h"
#include "components/step.h"
#include "format/encoding.h"
#include "format/layout.h"
#include "format/tensor_format.h"
#include "gemma4_fixture.h"
#include "import/checkpoint.h"
#include "import/normalize.h"
#include "import/reader.h"
#include "import/readers/gguf.h"
#include "import/writer.h"
#include "runtime/component_admission.h"
#include "spec/spec.h"
#include "spec/stored.h"
#include "tokenizer_fixtures.h"

namespace llmp::import {
namespace {
namespace rt = llmp::runtime;
namespace fm = llmp::format;

artifact::Source Origin(std::string_view name, std::uint64_t bytes, std::string_view hex) {
  base::Sha256Digest digest{};
  for (std::size_t i = 0; i < digest.size(); ++i) {
    unsigned byte = 0;
    const auto read = std::from_chars(hex.data() + 2 * i, hex.data() + 2 * i + 2, byte, 16);
    EXPECT_EQ(read.ec, std::errc{});
    digest[i] = static_cast<std::uint8_t>(byte);
  }
  return artifact::Source{std::string(name), base::Bytes(bytes), digest};
}

std::filesystem::path ModelsRoot() { return std::filesystem::path(test_support::ModelsDir()); }

std::filesystem::path TestDataDir() {
  const char* dir = std::getenv("LLMP_TEST_DATA");
  if (dir && *dir) return dir;
  return "tests/unit/data";
}

void Ordered(Checkpoint& s) { std::ranges::sort(s.metadata, {}, &MetaEntry::key); }

void Value(Checkpoint& s, std::string key, MetaValue value) {
  const auto old = std::ranges::find(s.metadata, key, &MetaEntry::key);
  if (old == s.metadata.end())
    s.metadata.push_back({std::move(key), std::move(value)});
  else
    old->value = std::move(value);
  Ordered(s);
}

void AddTensor(Checkpoint& s, std::string name, std::span<const std::uint64_t> dims,
               fm::Encoding encoding = fm::Encoding::kF32) {
  const auto shape = fm::ShapeOf(dims);
  base::Check(shape.has_value(), "test shape");
  const auto& info = fm::Info(encoding);
  fm::TensorFormat f{.encoding = encoding,
                     .layout = info.quantized ? fm::Layout::kGgmlBlocks : fm::Layout::kDense,
                     .shape = *shape};
  const auto parts = fm::PartsOf(f);
  base::Check(parts.has_value(), "test format");
  const auto offset = (s.files[0].bytes + 31) / 32 * 32;
  s.tensors.push_back({.name = std::move(name),
                       .format = f,
                       .extent = {.offset = offset, .bytes = (*parts)[0].bytes},
                       .dtype = info.name,
                       .dims = std::vector<std::uint64_t>(dims.begin(), dims.end())});
  s.files[0].bytes = offset + (*parts)[0].bytes;
}

std::optional<Checkpoint> SyntheticAssistantCheckpoint(std::uint32_t size) {
  const auto file_path =
      TestDataDir() / ("gemma4-assistant/assistant" + std::to_string(size) + ".json");
  std::error_code ec;
  if (!std::filesystem::exists(file_path, ec)) return std::nullopt;
  std::ifstream file(file_path);
  std::string text(std::istreambuf_iterator<char>{file}, {});
  auto doc = base::json::Parse(text);
  if (!doc.has_value()) return std::nullopt;
  Checkpoint s{.container = "gguf",
               .architecture = "gemma4-assistant",
               .files = {{.name = "mtp-gemma-4-" + std::to_string(size) +
                                  (size == 26 ? "B-A4B-it.gguf" : "B-it.gguf"),
                          .bytes = 0}}};
  const auto meta = test_support::gemma4::Get(doc->root(), "metadata");
  for (std::size_t i = 0; i < meta.size(); ++i) {
    const auto key = meta.key(i);
    if (key != "general.architecture" && !key.starts_with("gemma4-assistant.")) continue;
    const auto val = meta.member(i);
    if (val.is_bool()) {
      Value(s, std::string(key), val.boolean());
    } else if (val.is_number()) {
      if (val.is_integer()) {
        Value(s, std::string(key), *val.int64());
      } else {
        Value(s, std::string(key), *val.float64());
      }
    } else if (val.is_string()) {
      Value(s, std::string(key), std::string(val.string()));
    } else if (val.is_array()) {
      if (val.size() > 0 && val.at(0).is_bool()) {
        std::vector<std::int64_t> list;
        for (std::size_t j = 0; j < val.size(); ++j) list.push_back(val.at(j).boolean() ? 1 : 0);
        Value(s, std::string(key), list);
      } else if (val.size() > 0 && val.at(0).is_number() && val.at(0).is_integer()) {
        std::vector<std::int64_t> list;
        for (std::size_t j = 0; j < val.size(); ++j) list.push_back(*val.at(j).int64());
        Value(s, std::string(key), list);
      }
    }
  }
  Ordered(s);
  const auto tensors = test_support::gemma4::Get(doc->root(), "tensors");
  for (std::size_t i = 0; i < tensors.size(); ++i) {
    const auto t = tensors.at(i);
    const std::string name = std::string(test_support::gemma4::Get(t, "name").string());
    const std::string type_name = std::string(test_support::gemma4::Get(t, "type").string());
    const auto ne = test_support::gemma4::Get(t, "ne");
    std::vector<std::uint64_t> dims;
    for (std::size_t j = 0; j < ne.size(); ++j)
      dims.push_back(static_cast<std::uint64_t>(*ne.at(j).int64()));
    fm::Encoding enc = fm::Encoding::kF32;
    if (type_name == "Q8_0") enc = fm::Encoding::kQ8_0;
    AddTensor(s, name, dims, enc);
  }
  return s;
}

std::expected<artifact::Artifact, artifact::Error> MakeTargetV1(
    const std::filesystem::path& target_scratch, std::string arch, std::string source_name,
    std::uint64_t bytes, std::string sha256) {
  llmp::test::V1Plan p;
  p.architecture = std::move(arch);
  p.groups = {
      llmp::test::V1Group{.kind = llmp::artifact::GroupKind::kTable,
                          .resources = {{.name = "token_embd.weight",
                                         .format = {.encoding = fm::Encoding::kBF16,
                                                    .layout = fm::Layout::kDense,
                                                    .shape = {.ne = {64, 64, 1, 1}, .rank = 2}}}}}};
  auto rendered = llmp::test::Render(p);
  llmp::test::Json::Array source_list;
  source_list.emplace_back(llmp::test::Json::Object{{"bytes", static_cast<std::int64_t>(bytes)},
                                                    {"name", std::move(source_name)},
                                                    {"sha256", std::move(sha256)}});
  rendered.manifest["source"] = llmp::test::Json(std::move(source_list));
  return llmp::artifact::Artifact::Open(llmp::test::Publish(target_scratch, rendered));
}

std::expected<artifact::Artifact, artifact::Error> MakeDrafterV1(
    const std::filesystem::path& drafter_scratch, std::string arch,
    std::span<const spec::SourceTensor> tensors, std::string source_name, std::uint64_t bytes,
    std::string sha256) {
  llmp::test::V1Plan p;
  p.architecture = std::move(arch);
  p.container = "gguf";
  p.sources.clear();
  llmp::test::V1Group g;
  g.kind = artifact::GroupKind::kGlobal;
  for (const auto& t : tensors) {
    g.resources.push_back({
        .name = t.name,
        .format = t.format,
        .part = t.part,
    });
  }
  p.groups.push_back(std::move(g));
  auto rendered = llmp::test::Render(p);
  llmp::test::Json::Array source_list;
  source_list.emplace_back(llmp::test::Json::Object{{"bytes", static_cast<std::int64_t>(bytes)},
                                                    {"name", std::move(source_name)},
                                                    {"sha256", std::move(sha256)}});
  rendered.manifest["source"] = llmp::test::Json(std::move(source_list));
  return llmp::artifact::Artifact::Open(llmp::test::Publish(drafter_scratch, rendered));
}

TEST(ImportDrafterNormalizeTest,
     SyntheticGemma4AssistantNormalizesPublishesAndAdmitsWithTargetComposition) {
  auto cp = SyntheticAssistantCheckpoint(26);
  if (!cp.has_value()) {
    GTEST_SKIP() << "no synthetic assistant fixture in " << TestDataDir();
  }

  auto cost = NormalizationBound(*cp);
  ASSERT_TRUE(cost) << cost.error().ToString();
  NormalizationLimits limits{cost->metadata_bytes, cost->work_units};
  auto plan = Normalize(std::move(*cp), limits);
  ASSERT_TRUE(plan) << plan.error().ToString();

  ASSERT_EQ(plan->spec().stages.size(), 2U);
  EXPECT_EQ(plan->spec().stages[0].role, spec::StageRole::kTarget);
  EXPECT_EQ(plan->spec().stages[1].role, spec::StageRole::kDrafter);
  EXPECT_EQ(plan->spec().stages[1].artifact, 1U);
  EXPECT_EQ(plan->spec().stages[0].stream.width, 2816U);
  EXPECT_EQ(plan->spec().stages[1].stream.width, 1024U);
  EXPECT_EQ(plan->routes().size(), 49U);

  const std::filesystem::path scratch =
      std::filesystem::path(testing::TempDir()) / "import-drafter-synthetic-gemma4-asst";
  std::error_code ec;
  std::filesystem::remove_all(scratch, ec);
  std::filesystem::create_directories(scratch, ec);

  auto drafter_art =
      MakeDrafterV1(scratch / "drafter", "gemma4-assistant", plan->canonical().tensors,
                    "mtp-gemma-4-26B-A4B-it.gguf", 461766816ULL,
                    "6326fb9f5e487aa8dcdd313a091e3c67724cb2a666ec3b7d2895b5b26d93ed1b");
  ASSERT_TRUE(drafter_art) << (drafter_art ? "" : drafter_art.error().ToString());
  EXPECT_EQ(drafter_art->model().architecture, "gemma4-assistant");

  // Drafter artifact omits standalone spec.json
  EXPECT_FALSE(drafter_art->spec_json().has_value());

  auto target_art = MakeTargetV1(
      scratch / "target", "gemma4", "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf", 16947541728ULL,
      "f2c28b3dc4776931ac6f879e11f203dec637ea0f14267a86ec8f6165f63f293f");
  ASSERT_TRUE(target_art) << (target_art ? "" : target_art.error().ToString());

  auto drafter_source = spec::SourceOf(*drafter_art);
  ASSERT_TRUE(drafter_source) << drafter_source.error();
  // Drafter admission check with matching pinned origin
  std::array<artifact::Source, 1> origins{
      Origin("mtp-gemma-4-26B-A4B-it.gguf", 461766816ULL,
             "6326fb9f5e487aa8dcdd313a091e3c67724cb2a666ec3b7d2895b5b26d93ed1b")};
  auto admission = rt::CheckComponentDrafterAdmission(*drafter_source, origins);
  ASSERT_TRUE(admission) << (admission ? "" : admission.error());

  // Target composition values
  auto values = rt::ComponentDrafterValues(*target_art, *drafter_art);
  ASSERT_TRUE(values) << (values ? "" : values.error());
  EXPECT_FALSE(values->empty());

  std::filesystem::remove_all(scratch, ec);
}

TEST(ImportDrafterNormalizeModelsTest,
     Gemma4AssistantRealCheckpointsNormalizePublishAndAdmitWithTargetComposition) {
  for (const std::uint32_t size : {26U, 31U}) {
    const char* env_var =
        size == 26 ? "LLMP_GEMMA4_26B_ASSISTANT_GGUF" : "LLMP_GEMMA4_31B_ASSISTANT_GGUF";
    const char* configured = std::getenv(env_var);
    const std::filesystem::path path =
        configured != nullptr
            ? std::filesystem::path(configured)
            : ModelsRoot() / (size == 26 ? "reference-models/mtp-gemma-4-26B-A4B-it.gguf"
                                         : "reference-models/mtp-gemma-4-31B-it.gguf");
    std::error_code ec;
    if (!std::filesystem::exists(path, ec)) {
      GTEST_SKIP() << "no Gemma 4 " << size << "B assistant GGUF: " << path;
    }

    auto checkpoint = ReadGguf(path);
    ASSERT_TRUE(checkpoint) << checkpoint.error().ToString();
    EXPECT_EQ(checkpoint->container, "gguf");
    EXPECT_EQ(checkpoint->architecture, "gemma4-assistant");
    EXPECT_EQ(checkpoint->tensors.size(), 49U);

    auto bound = NormalizationBound(*checkpoint);
    ASSERT_TRUE(bound) << bound.error().ToString();

    NormalizationLimits limits{bound->metadata_bytes, bound->work_units};
    auto plan = Normalize(std::move(*checkpoint), limits);
    ASSERT_TRUE(plan) << plan.error().ToString();

    // Composed spec has stage 0 target, stage 1 drafter
    ASSERT_EQ(plan->spec().stages.size(), 2U);
    EXPECT_EQ(plan->spec().stages[0].role, spec::StageRole::kTarget);
    EXPECT_EQ(plan->spec().stages[0].artifact, 0U);
    EXPECT_EQ(plan->spec().stages[1].role, spec::StageRole::kDrafter);
    EXPECT_EQ(plan->spec().stages[1].artifact, 1U);
    const std::uint32_t expected_target_width = size == 26 ? 2816 : 5376;
    EXPECT_EQ(plan->spec().stages[0].stream.width, expected_target_width);
    EXPECT_EQ(plan->spec().stages[1].stream.width, 1024U);
    EXPECT_EQ(plan->routes().size(), 49U);

    const char* store_root = std::getenv("LLMP_TEST_PUBLICATION_STORE");
    if (store_root != nullptr && *store_root != '\0') {
      const std::filesystem::path scratch =
          std::filesystem::path(store_root) /
          ("import-gemma4-assistant-" + std::to_string(size) + "-models-test");
      std::filesystem::remove_all(scratch, ec);
      std::filesystem::create_directories(scratch, ec);

      WriteOptions options{
          .store = scratch / ("gemma4-asst-" + std::to_string(size) + "-v1"),
          .build = "test",
          .metadata_bytes = std::max(bound->metadata_bytes, std::uint64_t{1} << 30U),
      };
      auto written = WriteArtifact(*plan, options);
      ASSERT_TRUE(written) << written.error().ToString();
      EXPECT_FALSE(written->reused);

      // Opening the published drafter artifact
      auto drafter_art = artifact::Artifact::Open(written->path, {.expected_id = written->id});
      ASSERT_TRUE(drafter_art) << drafter_art.error().ToString();
      EXPECT_EQ(drafter_art->model().architecture, "gemma4-assistant");

      // Drafter artifact omits standalone spec.json
      EXPECT_FALSE(std::filesystem::exists(written->path / "spec.json"));

      // Verify drafter admission
      auto drafter_source = spec::SourceOf(*drafter_art);
      ASSERT_TRUE(drafter_source) << drafter_source.error();
      auto admission = rt::CheckComponentDrafterAdmission(*drafter_source, drafter_art->sources());
      ASSERT_TRUE(admission) << (admission ? "" : admission.error());

      // Verify target composition
      const auto target_art = MakeTargetV1(
          scratch / "target", "gemma4",
          size == 26 ? "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf" : "gemma-4-31B-it-UD-Q4_K_XL.gguf",
          size == 26 ? 16947541728ULL : 18822970304ULL,
          size == 26 ? "f2c28b3dc4776931ac6f879e11f203dec637ea0f14267a86ec8f6165f63f293f"
                     : "9e92cb6236044c6a9870af406029c74a76e0571c157a6f95df724dcc8c7a1575");
      ASSERT_TRUE(target_art) << (target_art ? "" : target_art.error().ToString());

      auto vals = rt::ComponentDrafterValues(*target_art, *drafter_art);
      ASSERT_TRUE(vals) << (vals ? "" : vals.error());

      // Refusal: target width mismatch (e.g. 26B target with 31B assistant)
      const auto bad_target_art = MakeTargetV1(
          scratch / "bad_target", "gemma4",
          size == 26 ? "gemma-4-31B-it-UD-Q4_K_XL.gguf" : "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf",
          size == 26 ? 18822970304ULL : 16947541728ULL,
          size == 26 ? "9e92cb6236044c6a9870af406029c74a76e0571c157a6f95df724dcc8c7a1575"
                     : "f2c28b3dc4776931ac6f879e11f203dec637ea0f14267a86ec8f6165f63f293f");
      ASSERT_TRUE(bad_target_art);
      auto mismatch = rt::ComponentDrafterValues(*bad_target_art, *drafter_art);
      EXPECT_FALSE(mismatch.has_value());

      RecordProperty("drafter_id", written->id);
      RecordProperty("drafter_path", written->path.string());
      RecordProperty("routes", "49");

      std::filesystem::remove_all(scratch, ec);
    }
  }
}

TEST(ImportDrafterNormalizeModelsTest,
     DsparkRealCheckpointNormalizesPublishesAndAdmitsWithTargetComposition) {
  const char* configured = std::getenv("LLMP_DSPARK_GGUF");
  const std::filesystem::path path =
      configured != nullptr ? std::filesystem::path(configured)
                            : ModelsRoot() /
                                  "models/unsloth/DeepSeek-V4-Flash-0731-GGUF@fbbb5b93/"
                                  "dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf";
  std::error_code ec;
  if (!std::filesystem::exists(path, ec)) {
    GTEST_SKIP() << "no DSpark GGUF: " << path;
  }

  auto checkpoint = ReadGguf(path);
  ASSERT_TRUE(checkpoint) << checkpoint.error().ToString();
  EXPECT_EQ(checkpoint->container, "gguf");
  EXPECT_EQ(checkpoint->architecture, "dflash");

  auto bound = NormalizationBound(*checkpoint);
  ASSERT_TRUE(bound) << bound.error().ToString();

  NormalizationLimits limits{bound->metadata_bytes, bound->work_units};
  auto plan = Normalize(std::move(*checkpoint), limits);
  ASSERT_TRUE(plan) << plan.error().ToString();

  ASSERT_EQ(plan->spec().stages.size(), 2U);
  EXPECT_EQ(plan->spec().stages[0].role, spec::StageRole::kTarget);
  EXPECT_EQ(plan->spec().stages[0].artifact, 0U);
  EXPECT_EQ(plan->spec().stages[1].role, spec::StageRole::kDrafter);
  EXPECT_EQ(plan->spec().stages[1].artifact, 1U);

  const char* store_root = std::getenv("LLMP_TEST_PUBLICATION_STORE");
  if (store_root != nullptr && *store_root != '\0') {
    const std::filesystem::path scratch =
        std::filesystem::path(store_root) / "import-dspark-models-test";
    std::filesystem::remove_all(scratch, ec);
    std::filesystem::create_directories(scratch, ec);

    WriteOptions options{
        .store = scratch / "dspark-v1",
        .build = "test",
        .metadata_bytes = std::max(bound->metadata_bytes, std::uint64_t{1} << 30U),
    };
    auto written = WriteArtifact(*plan, options);
    ASSERT_TRUE(written) << written.error().ToString();
    EXPECT_FALSE(written->reused);

    auto drafter_art = artifact::Artifact::Open(written->path, {.expected_id = written->id});
    ASSERT_TRUE(drafter_art) << drafter_art.error().ToString();
    EXPECT_EQ(drafter_art->model().architecture, "dflash");

    // Drafter artifact omits standalone spec.json
    EXPECT_FALSE(std::filesystem::exists(written->path / "spec.json"));

    // Verify drafter admission
    auto drafter_source = spec::SourceOf(*drafter_art);
    ASSERT_TRUE(drafter_source) << drafter_source.error();
    auto admission = rt::CheckComponentDrafterAdmission(*drafter_source, drafter_art->sources());
    ASSERT_TRUE(admission) << (admission ? "" : admission.error());

    // Verify target composition with Dsv4 target
    llmp::test::V1Plan p;
    p.architecture = "deepseek4";
    p.groups = {
        llmp::test::V1Group{.kind = llmp::artifact::GroupKind::kTable,
                            .resources = {{.name = "token_embd.weight",
                                           .format = {.encoding = fm::Encoding::kBF16,
                                                      .layout = fm::Layout::kDense,
                                                      .shape = {.ne = {64, 64, 1, 1}, .rank = 2}}}}}};
    auto rendered = llmp::test::Render(p);
    llmp::test::Json::Array source_list;
    source_list.emplace_back(llmp::test::Json::Object{
        {"bytes", static_cast<std::int64_t>(5257664ULL)},
        {"name", std::string("DeepSeek-V4-Flash-0731-UD-Q2_K_XL-00001-of-00003.gguf")},
        {"sha256", std::string("de65c5bb660817b95cc281bf54935f9d1bcc3b22535f2eae63b35a14c7c5724b")}});
    source_list.emplace_back(llmp::test::Json::Object{
        {"bytes", static_cast<std::int64_t>(49437013568ULL)},
        {"name", std::string("DeepSeek-V4-Flash-0731-UD-Q2_K_XL-00002-of-00003.gguf")},
        {"sha256", std::string("c8728a298862cd8736782e2e4056e5e3e03c415ba552a38173264494e41ef327")}});
    source_list.emplace_back(llmp::test::Json::Object{
        {"bytes", static_cast<std::int64_t>(47390237120ULL)},
        {"name", std::string("DeepSeek-V4-Flash-0731-UD-Q2_K_XL-00003-of-00003.gguf")},
        {"sha256", std::string("a4bae0788c025c1b2388754592c98fefa1372981748ac9875039bf6a30e6eeb4")}});
    rendered.manifest["source"] = llmp::test::Json(std::move(source_list));
    auto target_art =
        llmp::artifact::Artifact::Open(llmp::test::Publish(scratch / "target_dsv4", rendered));
    ASSERT_TRUE(target_art) << (target_art ? "" : target_art.error().ToString());

    auto vals = rt::ComponentDrafterValues(*target_art, *drafter_art);
    ASSERT_TRUE(vals) << (vals ? "" : vals.error());

    // Refusal: architecture mismatch (gemma4 target with dflash drafter)
    auto gemma_target_art = MakeTargetV1(
        scratch / "target_gemma", "gemma4", "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf", 16947541728ULL,
        "f2c28b3dc4776931ac6f879e11f203dec637ea0f14267a86ec8f6165f63f293f");
    ASSERT_TRUE(gemma_target_art);
    auto arch_mismatch = rt::ComponentDrafterValues(*gemma_target_art, *drafter_art);
    EXPECT_FALSE(arch_mismatch.has_value());

    RecordProperty("drafter_id", written->id);
    RecordProperty("drafter_path", written->path.string());

    std::filesystem::remove_all(scratch, ec);
  }
}

}  // namespace
}  // namespace llmp::import
