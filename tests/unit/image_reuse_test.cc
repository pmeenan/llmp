// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#include "import/image_reuse.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "cli/cli.h"
#include "cli/import.h"
#include "expected_error.h"
#include "import/image_pipeline.h"
#include "import/readers/safetensors.h"
#include "import_fixtures.h"

namespace llmp::import {
namespace {
using test_support::Failed;
constexpr std::uint64_t kMetadata = std::uint64_t{1} << 30U;
class ImageReuseTest : public fixtures::ScratchTest {
 protected:
  void SetUp() override {
    fixtures::ScratchTest::SetUp();
    Write(dir_ / "source/config.json", R"({"_class_name":"ToyImageDecoder"})");
    Write(dir_ / "source/model.safetensors", fixtures::Shard({fixtures::E("weight", "F32", {2})}));
  }
  std::expected<ComponentPlan, Error> Plan(const Checkpoint& checkpoint) {
    spec::Stage stage{.name = "decoder", .role = spec::StageRole::kVae};
    stage.stream.width = 2;
    stage.layers = {{{{.kind = "norm.rms",
                       .params = {{"eps", 1e-6}},
                       .weights = {{.role = "weight", .tensor = "weight"}}}}}};
    stage.links = {{0, {}}};
    stage.exports = {1};
    spec::ModelSpec model{.pipeline = spec::PipelineKind::kImage, .stages = {stage}};
    spec::Canonicalize(model);
    const std::array<model::ImageTensor, 1> tensors{{{"weight", "F32", {2}}}};
    auto plan = image_pipeline_detail::PlanComponent(checkpoint, tensors, {}, model, {});
    if (plan) {
      plan->mapping_architecture = "ToyImagePipeline";
      plan->mapping_version = 3;
    }
    return plan;
  }
  std::vector<std::optional<base::Sha256Digest>> Identities(const Checkpoint& checkpoint) {
    std::vector<std::optional<base::Sha256Digest>> identities;
    for (const auto& file : checkpoint.files) {
      std::ifstream in(file.path, std::ios::binary);
      const std::string bytes{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
      EXPECT_EQ(bytes.size(), file.bytes);
      identities.push_back(base::Sha256().Update(bytes).Finish());
    }
    return identities;
  }
  std::pair<int, std::string> Command(const std::vector<std::string>& words) {
    std::vector<std::string_view> args(words.begin(), words.end());
    std::FILE* output = std::tmpfile();
    std::FILE* errors = std::tmpfile();
    EXPECT_NE(output, nullptr);
    EXPECT_NE(errors, nullptr);
    if (!output || !errors) return {-1, "cannot open capture"};
    const int result = cli::Import(args, output, errors);
    std::rewind(errors);
    std::string text;
    std::array<char, 1024> buffer{};
    while (const auto read = std::fread(buffer.data(), 1, buffer.size(), errors))
      text.append(buffer.data(), read);
    std::fclose(output);
    std::fclose(errors);
    return {result, text};
  }
};
TEST_F(ImageReuseTest, ExistingWeightsComposeWithoutReadingSourcePayloadAgain) {
  auto checkpoint = ReadSafetensors(dir_ / "source");
  ASSERT_TRUE(checkpoint) << Failed(checkpoint, &Error::ToString).value_or("");
  checkpoint->role = "vae";
  auto plan = Plan(*checkpoint);
  ASSERT_TRUE(plan) << Failed(plan, &Error::ToString).value_or("");
  auto identities = Identities(*checkpoint);
  auto written = WriteArtifact(*plan, {.store = dir_ / "store",
                                       .build = "test",
                                       .expected = identities,
                                       .metadata_bytes = kMetadata});
  ASSERT_TRUE(written) << Failed(written, &Error::ToString).value_or("");
  auto opened = artifact::Artifact::Open(written->path);
  ASSERT_TRUE(opened) << Failed(opened, &artifact::Error::ToString).value_or("");
  // Remove all source files after parsing/planning: any attempt to stream the
  // unchanged component would fail. The authenticated prepared artifact remains.
  std::filesystem::remove_all(dir_ / "source");
  ASSERT_TRUE(ValidateImageComponentReuse(*plan, *opened, identities, kMetadata));
  std::ranges::reverse(checkpoint->files);
  std::ranges::reverse(identities);
  EXPECT_TRUE(ValidateImageComponentReuse(*plan, *opened, identities, kMetadata));
  auto composition =
      WriteComposition({.store = dir_ / "store",
                        .build = "test",
                        .architecture = "ToyPipeline",
                        .components = {{"vae", written->id, checkpoint->architecture}}});
  ASSERT_TRUE(composition) << Failed(composition, &Error::ToString).value_or("");
  auto selected = artifact::OpenComposition(composition->path);
  ASSERT_TRUE(selected);
  ASSERT_EQ(selected->components().size(), 1U);
  EXPECT_EQ(selected->components()[0].artifact, written->id);
  EXPECT_TRUE(artifact::Artifact::Open(written->path));
}
TEST_F(ImageReuseTest, RefusesChangedSourceConfigurationInventoryAndUnfundedMetadata) {
  auto checkpoint = ReadSafetensors(dir_ / "source");
  ASSERT_TRUE(checkpoint);
  checkpoint->role = "vae";
  auto plan = Plan(*checkpoint);
  ASSERT_TRUE(plan) << Failed(plan, &Error::ToString).value_or("");
  const auto identities = Identities(*checkpoint);
  auto written = WriteArtifact(*plan, {.store = dir_ / "store",
                                       .build = "test",
                                       .expected = identities,
                                       .metadata_bytes = kMetadata});
  ASSERT_TRUE(written) << Failed(written, &Error::ToString).value_or("");
  auto opened = artifact::Artifact::Open(written->path);
  ASSERT_TRUE(opened);
  const auto valid = [&](const auto& candidate, const auto& expected) {
    return ValidateImageComponentReuse(candidate, *opened, expected, kMetadata).has_value();
  };
  ASSERT_TRUE(valid(*plan, identities));
  EXPECT_FALSE(ValidateImageComponentReuse(*plan, *opened, identities, 1));
  auto missing = identities;
  missing.front().reset();
  EXPECT_FALSE(valid(*plan, missing));
  auto wrong_hash = identities;
  (*wrong_hash.front())[0] ^= 1;
  EXPECT_FALSE(valid(*plan, wrong_hash));
  auto changed = *plan;
  changed.spec.stages[0].role = spec::StageRole::kDenoiser;
  EXPECT_FALSE(valid(changed, identities));
  changed.spec = plan->spec;
  changed.spec.stages[0].layers[0].components[0].params[0].value = 1e-4;
  EXPECT_FALSE(valid(changed, identities));
  auto wrong_format = *plan;
  wrong_format.canonical.tensors[0].format.encoding = format::Encoding::kBF16;
  EXPECT_FALSE(valid(wrong_format, identities));
  wrong_format.canonical = plan->canonical;
  wrong_format.canonical.tensors[0].name = "other";
  EXPECT_FALSE(valid(wrong_format, identities));
  // Another mapping or mapping version (a route, permutation or relayout
  // fix keeps names, formats and the spec), or other options.
  auto remapped = *plan;
  remapped.mapping_version = 4;
  EXPECT_FALSE(valid(remapped, identities));
  remapped.mapping_version = plan->mapping_version;
  remapped.mapping_architecture = "OtherImagePipeline";
  EXPECT_FALSE(valid(remapped, identities));
  remapped.mapping_architecture.clear();
  remapped.mapping_version = 0;
  EXPECT_FALSE(valid(remapped, identities));
  EXPECT_FALSE(ValidateImageComponentReuse(*plan, *opened, identities, kMetadata, "{\"x\":1}"));
  EXPECT_EQ(opened->converter().mapping_architecture, "ToyImagePipeline");
  EXPECT_EQ(opened->converter().mapping_version, 3U);
  checkpoint->files[0].bytes += 1;
  EXPECT_FALSE(valid(*plan, identities));
  checkpoint->files[0].bytes -= 1;
  ASSERT_GE(checkpoint->files.size(), 2U);
  checkpoint->files[1].name = checkpoint->files[0].name;
  EXPECT_FALSE(valid(*plan, identities));
}
TEST_F(ImageReuseTest, CliRefusesDuplicateMalformedAndUnknownSelectionsBeforePublication) {
  const auto store = (dir_ / "store").string();
  const std::string id(64, 'a');
  for (const std::string& selection :
       std::array<std::string, 3>{"vae", "vae=short", "bad.role=" + id}) {
    const auto result =
        Command({"/missing/source", "--store", store, "--reuse-component", selection});
    EXPECT_EQ(result.first, cli::kExitUsage) << result.second;
  }
  const auto duplicate = Command({"/missing/source", "--store", store, "--reuse-component",
                                  "vae=" + id, "--reuse-component", "vae=" + id});
  EXPECT_EQ(duplicate.first, cli::kExitUsage) << duplicate.second;
  // A readable pipeline with no such role must refuse before planning or writing.
  Write(dir_ / "pipeline/model_index.json",
        R"({"_class_name":"QwenImage21Pipeline","vae":["diffusers","ToyImageDecoder"]})");
  Write(dir_ / "pipeline/vae/config.json", R"({"_class_name":"ToyImageDecoder"})");
  Write(dir_ / "pipeline/vae/model.safetensors",
        fixtures::Shard({fixtures::E("weight", "F32", {2})}));
  const auto unknown = Command(
      {(dir_ / "pipeline").string(), "--store", store, "--reuse-component", "unknown=" + id});
  EXPECT_EQ(unknown.first, cli::kExitFailure);
  EXPECT_NE(unknown.second.find("unknown image component role"), std::string::npos)
      << unknown.second;
  EXPECT_FALSE(std::filesystem::exists(dir_ / "store"));
}
TEST_F(ImageReuseTest, HandlesEscapedQuotesInsideConfigJsonStrings) {
  Write(dir_ / "source/config.json",
        "{\n  \"_class_name\": \"ToyImageDecoder\",\n  \"note\": \"a \\\"quoted value\\\" with "
        "spaces\"\n}\n");
  auto checkpoint = ReadSafetensors(dir_ / "source");
  ASSERT_TRUE(checkpoint) << Failed(checkpoint, &Error::ToString).value_or("");
  checkpoint->role = "vae";
  auto plan = Plan(*checkpoint);
  ASSERT_TRUE(plan) << Failed(plan, &Error::ToString).value_or("");
  const auto identities = Identities(*checkpoint);
  auto written = WriteArtifact(*plan, {.store = dir_ / "store",
                                       .build = "test",
                                       .expected = identities,
                                       .metadata_bytes = kMetadata});
  ASSERT_TRUE(written) << Failed(written, &Error::ToString).value_or("");
  auto opened = artifact::Artifact::Open(written->path);
  ASSERT_TRUE(opened) << Failed(opened, &artifact::Error::ToString).value_or("");
  EXPECT_TRUE(ValidateImageComponentReuse(*plan, *opened, identities, kMetadata));
}
}  // namespace
}  // namespace llmp::import
