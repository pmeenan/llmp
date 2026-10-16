// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "runtime/component_admission.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <cstdint>
#include <filesystem>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "artifact_writer.h"
#include "base/bytes.h"
#include "base/sha256.h"
#include "dsv4_gguf_normalization_fixture.h"
#include "import/normalize.h"
#include "model/gemma2.h"
#include "model/gemma3.h"
#include "model/gemma4.h"
#include "qwen2_exl3_normalization_fixture.h"
#include "qwen2_gguf_normalization_fixture.h"
#include "qwen38_fixture.h"
#include "qwen38_gguf_normalization_fixture.h"
#include "runtime/calibration.h"
#include "spec/mappings/gemma2.h"
#include "spec/mappings/gemma3.h"
#include "spec/mappings/gemma4.h"
#include "spec/mappings/qwen38.h"
#include "spec_fixtures.h"
#include "tokenizer_fixtures.h"

namespace {
namespace rt = llmp::runtime;
namespace sp = llmp::spec;
namespace fm = llmp::format;

std::array<llmp::artifact::Source, 1> Origins(
    std::string_view name = "gemma-2-2b-it-Q8_0.gguf", std::uint64_t bytes = 2784495456ULL,
    std::string_view hex = "2d448a9aab894b8e8e18168cf3f490cb9f65632222f29f93514ac9ecc754debe") {
  llmp::base::Sha256Digest digest{};
  for (std::size_t i = 0; i < digest.size(); ++i) {
    unsigned byte = 0;
    const auto read = std::from_chars(hex.data() + 2 * i, hex.data() + 2 * i + 2, byte, 16);
    EXPECT_EQ(read.ec, std::errc{});
    digest[i] = static_cast<std::uint8_t>(byte);
  }
  return {{{std::string(name), llmp::base::Bytes(bytes), digest}}};
}

class ComponentAdmissionTest : public testing::Test {
 protected:
  sp::Source source = llmp::test_support::Gemma2Source();
  sp::ModelSpec model;
  std::array<llmp::artifact::Source, 1> origins = Origins();
  void SetUp() override {
    const auto values = sp::mappings::Gemma2V0Values(llmp::model::Gemma2_2B());
    ASSERT_TRUE(values);
    auto made = sp::Synthesize(sp::mappings::Gemma2Mapping(), source, *values);
    ASSERT_TRUE(made) << (made ? "" : made.error());
    model = std::move(*made);
  }
};

TEST_F(ComponentAdmissionTest, ActualSpecChangesInvalidateCalibrationForTheSameArtifact) {
  rt::CalibrationKey key{.artifact = std::string(64, 'a'),
                         .drafter = {},
                         .device = "test device",
                         .build = "test build",
                         .settings = {}};
  const std::array<std::string_view, 1> artifacts{key.artifact};
  auto digest = sp::BoundDigest(model, artifacts);
  ASSERT_TRUE(digest) << (digest ? "" : digest.error());
  rt::ModelSettings settings;
  settings.bound_spec = *digest;
  key.settings = rt::MeasuredWith(settings);
  const auto record = rt::FormatCalibration(rt::Calibration{}, key);
  EXPECT_TRUE(rt::ParseCalibration(record, key).calibration);
  // Same artifact/build/settings, different real spec semantics and unchanged
  // layout sizes: calibration must still become stale.
  ASSERT_TRUE(model.stages.front().heads.softcap);
  model.stages.front().heads.softcap = *model.stages.front().heads.softcap + 1.0;
  digest = sp::BoundDigest(model, artifacts);
  ASSERT_TRUE(digest) << (digest ? "" : digest.error());
  settings.bound_spec = *digest;
  key.settings = rt::MeasuredWith(settings);
  EXPECT_FALSE(rt::ParseCalibration(record, key).calibration);
}

TEST_F(ComponentAdmissionTest, ApprovesCompleteNativeFormatsAndPreservesTheStoredSpec) {
  const auto before = model.Digest();
  const auto admitted = rt::CheckComponentAdmission(model, source, origins);
  ASSERT_TRUE(admitted) << (admitted ? "" : admitted.error());
  EXPECT_EQ(model.Digest(), before);
  EXPECT_EQ(model.stages.front().context, 8192U);
  EXPECT_EQ(model.stages.front().folds, std::vector<std::string>{"norm.plus_one"});
  EXPECT_EQ(model.stages.front().embedding->weights.front().tensor,
            model.stages.front().heads.output->weights.front().tensor);
}

TEST_F(ComponentAdmissionTest, UnknownOriginOrFamilyHasNoV1Admission) {
  origins[0].sha256[0] ^= 1;
  EXPECT_FALSE(rt::CheckComponentAdmission(model, source, origins));
  origins = Origins();
  origins[0].bytes = llmp::base::Bytes(origins[0].bytes.value() - 1);
  EXPECT_FALSE(rt::CheckComponentAdmission(model, source, origins));
  origins = Origins();
  source.architecture = "gemma3";
  EXPECT_FALSE(rt::CheckComponentAdmission(model, source, origins));
}

TEST_F(ComponentAdmissionTest, RejectsContextArithmeticFoldsAndUntiedHeadChanges) {
  const auto original = model;
  ++model.stages.front().context;
  EXPECT_FALSE(rt::CheckComponentAdmission(model, source, origins));
  model = original;
  model.stages.front().heads.softcap = 29.0;
  EXPECT_FALSE(rt::CheckComponentAdmission(model, source, origins));
  model = original;
  model.stages.front().folds.clear();
  EXPECT_FALSE(rt::CheckComponentAdmission(model, source, origins));
  model = original;
  model.stages.front().heads.output->weights.front().tensor = "output.weight";
  EXPECT_FALSE(rt::CheckComponentAdmission(model, source, origins));
  model = original;
  source.folds = {"norm.plus_one"};
  EXPECT_FALSE(rt::CheckComponentAdmission(model, source, origins));
}

TEST_F(ComponentAdmissionTest, RejectsNativeFormatGeometryAndRoleChanges) {
  const auto original = source;
  auto& format = source.tensors.front().format;
  format.encoding = fm::Encoding::kQ4_0;
  EXPECT_FALSE(rt::CheckComponentAdmission(model, source, origins));
  source = original;
  source.tensors.front().format.shape.ne[0] += 32;
  EXPECT_FALSE(rt::CheckComponentAdmission(model, source, origins));
  source = original;
  source.tensors.front().part = fm::PartRole::kCodes;
  EXPECT_FALSE(rt::CheckComponentAdmission(model, source, origins));
  source = original;
  source.tensors.front().name = "unbound.weight";
  EXPECT_FALSE(rt::CheckComponentAdmission(model, source, origins));
}

TEST_F(ComponentAdmissionTest, PolicyRechecksTheApprovedSpecBeforeSelectingItsRecipe) {
  auto mapping = sp::mappings::Gemma2Mapping();
  auto rule = rt::ComponentAdmissionPolicy(model, source, origins, {.context = 8192, .slots = 2});
  ASSERT_TRUE(rule) << (rule ? "" : rule.error());
  EXPECT_EQ(rule->text, mapping.policies.front().text);
  EXPECT_FALSE(rt::ComponentAdmissionPolicy(model, source, origins, {.context = 8193, .slots = 2}));
  EXPECT_FALSE(rt::ComponentAdmissionPolicy(model, source, origins, {.context = 8192, .slots = 3}));
  model.stages.front().context = 8193;
  EXPECT_FALSE(rt::ComponentAdmissionPolicy(model, source, origins, {.context = 8192, .slots = 2}));
}

TEST(ComponentAdmissionQwenTest, FullNormalizedTargetKeepsItsUntiedHeadAndSourceHash) {
  auto checkpoint = llmp::test_support::Qwen38GgufCheckpoint();
  auto bound = llmp::import::NormalizationBound(checkpoint);
  ASSERT_TRUE(bound);
  auto normalized =
      llmp::import::Normalize(std::move(checkpoint), {bound->metadata_bytes, bound->work_units});
  ASSERT_TRUE(normalized) << (normalized ? "" : normalized.error().ToString());
  std::array<llmp::artifact::Source, 3> origins{
      Origins("Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf", 10946624,
              "268f81fdedf3149a538f252308927a4d5d1f6e062c178568a51e3b519744f8a8")[0],
      Origins("Qwen3.8-Flash-Next-UD-IQ3_XXS-00002-of-00003.gguf", 49567921344ULL,
              "cfe600b236b88c7fad1613a5ca5e83b9f2beb63cbd44c32b2be50a44747c695f")[0],
      Origins("Qwen3.8-Flash-Next-UD-IQ3_XXS-00003-of-00003.gguf", 32382955968ULL,
              "f1912ba34c79427d2295a58dcb2b732b5931af5bef7a373c60557a57d9ee7250")[0]};
  auto model = normalized->spec();
  auto source = normalized->canonical();
  const auto admitted = rt::CheckComponentAdmission(model, source, origins);
  ASSERT_TRUE(admitted) << (admitted ? "" : admitted.error());
  EXPECT_TRUE(rt::OnComponentEngine("qwen4exp"));
  EXPECT_NE(model.stages[0].embedding->weights[0].tensor,
            model.stages[0].heads.output->weights[0].tensor);
  auto policy = rt::ComponentAdmissionPolicy(model, source, origins, {.context = 8192, .slots = 4});
  ASSERT_TRUE(policy) << (policy ? "" : policy.error());
  EXPECT_TRUE(
      rt::ComponentAdmissionPolicy(model, source, origins, {.context = 262144, .slots = 8}));
  EXPECT_FALSE(
      rt::ComponentAdmissionPolicy(model, source, origins, {.context = 262145, .slots = 1}));
  for (int mutation = 0; mutation != 7; ++mutation) {
    auto changed_model = model;
    auto changed_source = source;
    auto changed_origins = origins;
    switch (mutation) {
      case 0:
        changed_origins[2].sha256[0] ^= 1;
        break;
      case 1:
        changed_origins[2] = changed_origins[1];
        break;
      case 2:
        ++changed_model.stages[0].context;
        break;
      case 3:
        changed_model.stages[0].folds.clear();
        break;
      case 4:
        changed_model.stages[0].heads.output->weights[0].tensor =
            changed_model.stages[0].embedding->weights[0].tensor;
        break;
      case 5:
        changed_source.tensors[0].format.encoding = fm::Encoding::kF16;
        break;
      case 6:
        ASSERT_FALSE(changed_model.stages[0].side.empty());
        for (auto& param : changed_model.stages[0].side[0].source.params)
          if (param.name == "multipliers") std::get<std::vector<std::int64_t>>(param.value)[0] += 2;
        break;
    }
    EXPECT_FALSE(rt::CheckComponentAdmission(changed_model, changed_source, changed_origins))
        << mutation;
  }
}

class NativeGemmaAdmissionTest : public testing::TestWithParam<std::uint32_t> {
 protected:
  sp::Source source;
  sp::ModelSpec model;
  std::array<llmp::artifact::Source, 1> origins;
  void SetUp() override {
    std::vector<sp::SourceValue> values;
    if (GetParam() == 3) {
      source = llmp::test_support::Gemma3Source();
      const auto approved = sp::mappings::Gemma3V0Values(llmp::model::Gemma3_4BQat());
      ASSERT_TRUE(approved);
      values = *approved;
      origins = Origins("gemma-3-4b-it-qat-Q4_0.gguf", 2526080992ULL,
                        "ee91c3e7a4ab95d8c95672f9fcb58bf236b257e9f217966bcf53a5a6df4ab49a");
    } else {
      source = llmp::test_support::Gemma4Source(GetParam());
      // Independent captured GGUF metadata, not the compiled profile values
      // that admission uses as its comparison oracle.
      values = llmp::test_support::Gemma4FixtureValues(GetParam());
      const auto doc = llmp::test_support::gemma4::Fixture(GetParam());
      const auto origin = llmp::test_support::gemma4::Get(doc.root(), "source");
      const auto get = [&](std::string_view key) {
        return llmp::test_support::gemma4::Get(origin, key);
      };
      origins =
          Origins(get("file").string(), static_cast<std::uint64_t>(*get("full_bytes").int64()),
                  get("full_sha256").string());
    }
    auto made = sp::Synthesize(*sp::FindMapping(source.architecture), source, values);
    ASSERT_TRUE(made) << (made ? "" : made.error());
    model = std::move(*made);
  }
};
TEST_P(NativeGemmaAdmissionTest, CapturedDirectoryAndSpecAreAcceptedWithoutMutation) {
  const auto before = model.Digest();
  const auto admitted = rt::CheckComponentAdmission(model, source, origins);
  ASSERT_TRUE(admitted) << (admitted ? "" : admitted.error());
  EXPECT_EQ(before, model.Digest());
  std::ranges::reverse(source.tensors);
  EXPECT_TRUE(rt::CheckComponentAdmission(model, source, origins));
  EXPECT_EQ(model.stages.front().folds.empty(), GetParam() != 3);
}
TEST_P(NativeGemmaAdmissionTest, RefusesChangedFormatPlacementExpertIdentityAndArithmetic) {
  const auto saved_source = source;
  auto matrix =
      std::ranges::find_if(source.tensors, [](const auto& t) { return t.format.shape.rank == 2; });
  ASSERT_NE(matrix, source.tensors.end());
  matrix->format.encoding =
      matrix->format.encoding == fm::Encoding::kQ8_0 ? fm::Encoding::kQ4_0 : fm::Encoding::kQ8_0;
  EXPECT_FALSE(rt::CheckComponentAdmission(model, source, origins));
  source = saved_source;
  source.tensors.front().format.params.tensor_scale = true;
  EXPECT_FALSE(rt::CheckComponentAdmission(model, source, origins));
  source = saved_source;
  source.tensors.front().experts = !source.tensors.front().experts;
  EXPECT_FALSE(rt::CheckComponentAdmission(model, source, origins));
  source = saved_source;
  const auto saved_model = model;
  model.stages.front().heads.output->weights.front().tensor = "output.weight";
  EXPECT_FALSE(rt::CheckComponentAdmission(model, source, origins));
  model = saved_model;
  ++model.stages.front().context;
  EXPECT_FALSE(rt::CheckComponentAdmission(model, source, origins));
  model = saved_model;
  model.stages.front().role = sp::StageRole::kDrafter;
  EXPECT_FALSE(rt::CheckComponentAdmission(model, source, origins));
  model = saved_model;
  origins.front().sha256.front() ^= 1;
  EXPECT_FALSE(rt::CheckComponentAdmission(model, source, origins));
}
TEST_P(NativeGemmaAdmissionTest, ChoosesExactProfileAndEnvelopeRecipeWithoutCheckpointKeys) {
  const bool gemma3 = GetParam() == 3;
  const std::uint32_t context = gemma3 ? 131072 : 8192, slots = gemma3 ? 1 : 4;
  const auto rule =
      rt::ComponentAdmissionPolicy(model, source, origins, {.context = context, .slots = slots});
  ASSERT_TRUE(rule) << (rule ? "" : rule.error());
  EXPECT_EQ(rule->prefill_chunk, gemma3 ? 0U : GetParam() == 26 ? 1024U : 256U);
  EXPECT_EQ(rule->prompts_alone, !gemma3);
  EXPECT_EQ(rule->text.contains("disable mul_mat_glu"), GetParam() == 26);
  if (gemma3) {
    EXPECT_TRUE(
        rt::ComponentAdmissionPolicy(model, source, origins, {.context = 4096, .slots = 2}));
    EXPECT_FALSE(
        rt::ComponentAdmissionPolicy(model, source, origins, {.context = 4097, .slots = 2}));
  }
  if (!gemma3) {
    EXPECT_TRUE(rule->text.contains("disable owner_bounded_roots"));
    const auto small =
        rt::ComponentAdmissionPolicy(model, source, origins, {.context = 4096, .slots = 2});
    ASSERT_TRUE(small);
    EXPECT_FALSE(small->text.contains("disable owner_bounded_roots"));
  }
  EXPECT_FALSE(rt::ComponentAdmissionPolicy(model, source, origins,
                                            {.context = context + 1, .slots = slots}));
  EXPECT_FALSE(rt::ComponentAdmissionPolicy(model, source, origins,
                                            {.context = context, .slots = slots + 1}));
  EXPECT_FALSE(
      rt::ComponentAdmissionPolicy(model, source, origins, {.context = 0, .slots = slots}));
  model.stages.front().layers.pop_back();
  EXPECT_FALSE(rt::ComponentAdmissionPolicy(model, source, origins, {.context = 4096, .slots = 2}));
}
INSTANTIATE_TEST_SUITE_P(ApprovedTargets, NativeGemmaAdmissionTest, testing::Values(3U, 26U, 31U));

TEST(ComponentAdmissionDsv4Test, FullStoredTargetPreservesProfilePolicyAndClosedSourceAuthority) {
  auto checkpoint = llmp::test_support::Dsv4GgufCheckpoint();
  auto bound = llmp::import::NormalizationBound(checkpoint);
  ASSERT_TRUE(bound) << (bound ? "" : bound.error().ToString());
  auto normalized =
      llmp::import::Normalize(std::move(checkpoint), {bound->metadata_bytes, bound->work_units});
  ASSERT_TRUE(normalized) << (normalized ? "" : normalized.error().ToString());
  std::array<llmp::artifact::Source, 3> origins{
      Origins("DeepSeek-V4-Flash-0731-UD-Q2_K_XL-00001-of-00003.gguf", 5257664,
              "de65c5bb660817b95cc281bf54935f9d1bcc3b22535f2eae63b35a14c7c5724b")[0],
      Origins("DeepSeek-V4-Flash-0731-UD-Q2_K_XL-00002-of-00003.gguf", 49437013568ULL,
              "c8728a298862cd8736782e2e4056e5e3e03c415ba552a38173264494e41ef327")[0],
      Origins("DeepSeek-V4-Flash-0731-UD-Q2_K_XL-00003-of-00003.gguf", 47390237120ULL,
              "a4bae0788c025c1b2388754592c98fefa1372981748ac9875039bf6a30e6eeb4")[0]};
  const auto& model = normalized->spec();
  const auto& source = normalized->canonical();
  const auto admitted = rt::CheckComponentAdmission(model, source, origins);
  ASSERT_TRUE(admitted) << (admitted ? "" : admitted.error());
  // Served on the component engine since c96deafd (K1).
  EXPECT_TRUE(rt::OnComponentEngine("deepseek4"));
  auto policy =
      rt::ComponentAdmissionPolicy(model, source, origins, {.context = 1048576, .slots = 8});
  ASSERT_TRUE(policy) << (policy ? "" : policy.error());
  EXPECT_TRUE(policy->text.empty());
  EXPECT_FALSE(
      rt::ComponentAdmissionPolicy(model, source, origins, {.context = 1048577, .slots = 1}));
  for (unsigned mutation = 0; mutation != 6; ++mutation) {
    auto changed_model = model;
    auto changed_source = source;
    auto changed_origins = origins;
    if (mutation == 0) changed_origins[2].sha256[0] ^= 1;
    if (mutation == 1) changed_origins[2] = changed_origins[1];
    if (mutation == 2) changed_source.tensors[0].format.encoding = fm::Encoding::kF16;
    if (mutation == 3) changed_model.stages[0].side.pop_back();
    if (mutation == 4)
      changed_model.stages[0].heads.output->weights[0].tensor =
          changed_model.stages[0].embedding->weights[0].tensor;
    if (mutation == 5) changed_model.stages[0].folds.push_back("norm.plus_one");
    EXPECT_FALSE(rt::CheckComponentAdmission(changed_model, changed_source, changed_origins))
        << mutation;
  }
}

TEST(ComponentAdmissionQwen2Test, FullStoredTargetPreservesProfilePolicyAndClosedSourceAuthority) {
  auto checkpoint = llmp::test_support::Qwen2GgufCheckpoint();
  auto bound = llmp::import::NormalizationBound(checkpoint);
  ASSERT_TRUE(bound) << (bound ? "" : bound.error().ToString());
  auto normalized =
      llmp::import::Normalize(std::move(checkpoint), {bound->metadata_bytes, bound->work_units});
  ASSERT_TRUE(normalized) << (normalized ? "" : normalized.error().ToString());
  const auto origins = Origins("qwen2.5-0.5b-instruct-fp16.gguf", 1266425696ULL,
                               "8e0ae26000627ed62de0e78e41860af70094558b9d2913385c842a6aa06cf3fc");
  const auto& model = normalized->spec();
  const auto& source = normalized->canonical();
  const auto admitted = rt::CheckComponentAdmission(model, source, origins);
  ASSERT_TRUE(admitted) << (admitted ? "" : admitted.error());
  EXPECT_FALSE(rt::OnComponentEngine("qwen2"));
  auto policy = rt::ComponentAdmissionPolicy(model, source, origins, {.context = 8192, .slots = 1});
  ASSERT_TRUE(policy) << (policy ? "" : policy.error());
  EXPECT_TRUE(policy->text.contains("max_decode_rows 1"));
  EXPECT_FALSE(rt::ComponentAdmissionPolicy(model, source, origins, {.context = 8193, .slots = 1}));
  EXPECT_FALSE(rt::ComponentAdmissionPolicy(model, source, origins, {.context = 8192, .slots = 2}));
  for (unsigned mutation = 0; mutation != 5; ++mutation) {
    auto changed_model = model;
    auto changed_source = source;
    auto changed_origins = origins;
    if (mutation == 0) changed_origins[0].sha256[0] ^= 1;
    if (mutation == 1) changed_origins[0].bytes = llmp::base::Bytes(1266425695ULL);
    if (mutation == 2) changed_source.tensors[0].format.encoding = fm::Encoding::kBF16;
    if (mutation == 3)
      changed_model.stages[0].heads.output->weights[0].tensor =
          changed_model.stages[0].embedding->weights[0].tensor;
    if (mutation == 4) changed_model.stages[0].folds.push_back("norm.plus_one");
    EXPECT_FALSE(rt::CheckComponentAdmission(changed_model, changed_source, changed_origins))
        << mutation;
  }
}

TEST(ComponentAdmissionQwen2Exl3Test,
     Approved40And45TargetsPreserveMultipartInventoryAndClosedSourceAuthority) {
  for (const bool rate_4_5 : {false, true}) {
    auto checkpoint = llmp::test_support::Qwen2Exl3Checkpoint(rate_4_5);
    auto bound = llmp::import::NormalizationBound(checkpoint);
    ASSERT_TRUE(bound) << (bound ? "" : bound.error().ToString());
    auto normalized =
        llmp::import::Normalize(std::move(checkpoint), {bound->metadata_bytes, bound->work_units});
    ASSERT_TRUE(normalized) << (normalized ? "" : normalized.error().ToString());
    std::array<llmp::artifact::Source, 2> origins{
        Origins("config.json", rate_4_5 ? 923ULL : 996ULL,
                rate_4_5 ? "dcd945f38ba6e7835c8c281a45c03d9a20759d03afe5761d09def8c62aa11f3c"
                         : "966657b7e8c331ea16127a868eb6d6ff16ad0b16a5ee39d26279e12de0a3205c")[0],
        Origins("model.safetensors", rate_4_5 ? 611257914ULL : 588951098ULL,
                rate_4_5 ? "94c45f9838a56fbf23f43d0a04c47e187d840dd97f17b4a027a69a20339762bc"
                         : "66b61bd1ca3ebb0fc1cea29148eda8bdaca45d232c6a7e319c6d2835b65c066b")[0]};
    const auto& model = normalized->spec();
    const auto& source = normalized->canonical();
    const auto admitted = rt::CheckComponentAdmission(model, source, origins);
    ASSERT_TRUE(admitted) << (admitted ? "" : admitted.error());
    auto policy =
        rt::ComponentAdmissionPolicy(model, source, origins, {.context = 8192, .slots = 1});
    ASSERT_TRUE(policy) << (policy ? "" : policy.error());
    EXPECT_TRUE(policy->text.contains("max_decode_rows 1"));
    for (unsigned mutation = 0; mutation != 6; ++mutation) {
      auto changed_model = model;
      auto changed_source = source;
      auto changed_origins = origins;
      if (mutation == 0) changed_origins[0].sha256[0] ^= 1;
      if (mutation == 1) changed_origins[1] = changed_origins[0];
      if (mutation == 2) changed_source.container = "gguf";
      if (mutation == 3) changed_source.tensors[2].format.params.k_bits = 6;
      if (mutation == 4) changed_source.tensors[2].part = fm::PartRole::kData;
      if (mutation == 5) changed_model.stages[0].folds.push_back("norm.plus_one");
      EXPECT_FALSE(rt::CheckComponentAdmission(changed_model, changed_source, changed_origins))
          << mutation;
    }
  }
}

TEST(ComponentAdmissionDrafterTest, ComponentDrafterV0ValuesRejectsMismatchedAndInvalidArtifacts) {
  const std::filesystem::path scratch =
      std::filesystem::path(testing::TempDir()) / "component-admission-drafters";
  std::error_code ec;
  std::filesystem::remove_all(scratch, ec);
  std::filesystem::create_directories(scratch, ec);

  const auto make_plan = [](std::string arch) {
    llmp::test::V1Plan p;
    p.architecture = std::move(arch);
    p.groups = {llmp::test::V1Group{
        .kind = llmp::artifact::GroupKind::kTable,
        .resources = {{.name = "token_embd.weight",
                       .format = {.encoding = fm::Encoding::kBF16,
                                  .layout = fm::Layout::kDense,
                                  .shape = {.ne = {64, 64, 1, 1}, .rank = 2}}}}}};
    return p;
  };

  const auto gemma_target = llmp::artifact::Artifact::Open(
      llmp::test::Publish(scratch / "target", llmp::test::Render(make_plan("gemma4"))));
  ASSERT_TRUE(gemma_target.has_value()) << (gemma_target ? "" : gemma_target.error().ToString());

  const auto gemma_drafter = llmp::artifact::Artifact::Open(
      llmp::test::Publish(scratch / "asst", llmp::test::Render(make_plan("gemma4-assistant"))));
  ASSERT_TRUE(gemma_drafter.has_value()) << (gemma_drafter ? "" : gemma_drafter.error().ToString());

  const auto dsv4_target = llmp::artifact::Artifact::Open(
      llmp::test::Publish(scratch / "dsv4", llmp::test::Render(make_plan("deepseek4"))));
  ASSERT_TRUE(dsv4_target.has_value()) << (dsv4_target ? "" : dsv4_target.error().ToString());

  const auto dflash_drafter = llmp::artifact::Artifact::Open(
      llmp::test::Publish(scratch / "dflash", llmp::test::Render(make_plan("dflash"))));
  ASSERT_TRUE(dflash_drafter.has_value())
      << (dflash_drafter ? "" : dflash_drafter.error().ToString());

  // Mismatched pairings are rejected with descriptive errors.
  auto mismatched1 = rt::ComponentDrafterV0Values(*gemma_target, *dflash_drafter);
  EXPECT_FALSE(mismatched1.has_value());
  EXPECT_NE(mismatched1.error().find("no drafter dflash for architecture gemma4"),
            std::string::npos);

  auto mismatched2 = rt::ComponentDrafterV0Values(*dsv4_target, *gemma_drafter);
  EXPECT_FALSE(mismatched2.has_value());
  EXPECT_NE(mismatched2.error().find("no drafter gemma4-assistant for architecture deepseek4"),
            std::string::npos);

  // Matched architectures with invalid (v1 synthetic) artifacts fail through the binders.
  auto gemma_pair = rt::ComponentDrafterV0Values(*gemma_target, *gemma_drafter);
  EXPECT_FALSE(gemma_pair.has_value());

  auto dsv4_pair = rt::ComponentDrafterV0Values(*dsv4_target, *dflash_drafter);
  EXPECT_FALSE(dsv4_pair.has_value());
  EXPECT_NE(dsv4_pair.error().find("version 0"), std::string::npos);

  // If actual v0 artifacts are installed (e.g. in full models environment), verify success.
  const std::filesystem::path models = llmp::test_support::ModelsDir();
  const auto g4_target_path = models / "m3-artifacts/gemma-4-26B-A4B-it-UD-Q4_K_M";
  const auto g4_asst_path = models / "m3-artifacts/gemma-4-26B-A4B-it-assistant";
  if (std::filesystem::exists(g4_target_path, ec) && std::filesystem::exists(g4_asst_path, ec)) {
    auto t = llmp::artifact::Artifact::Open(g4_target_path, {.expected_id = {}});
    auto a = llmp::artifact::Artifact::Open(g4_asst_path, {.expected_id = {}});
    if (t && a) {
      auto vals = rt::ComponentDrafterV0Values(*t, *a);
      EXPECT_TRUE(vals.has_value()) << (vals ? "" : vals.error());
    }
  }
}

TEST(ComponentAdmissionDrafterTest, CheckComponentDrafterAdmissionApprovesGemma4Assistants) {
  for (std::uint32_t size : {26U, 31U}) {
    auto source = llmp::test_support::Gemma4AssistantSource(size);
    auto origins =
        size == 26 ? Origins("mtp-gemma-4-26B-A4B-it.gguf", 461766816ULL,
                             "6326fb9f5e487aa8dcdd313a091e3c67724cb2a666ec3b7d2895b5b26d93ed1b")
                   : Origins("mtp-gemma-4-31B-it.gguf", 514687104ULL,
                             "5ae8b0117bed601e8924c6305bd5b0585de361d51f0e77091bcb4252cf1f27de");
    auto check = rt::CheckComponentDrafterAdmission(source, origins);
    EXPECT_TRUE(check.has_value()) << (check ? "" : check.error());

    // Mutations must fail admission
    auto bad_source = source;
    bad_source.container = "safetensors";
    EXPECT_FALSE(rt::CheckComponentDrafterAdmission(bad_source, origins));

    bad_source = source;
    bad_source.folds = {"some_fold"};
    EXPECT_FALSE(rt::CheckComponentDrafterAdmission(bad_source, origins));

    auto bad_origins = origins;
    bad_origins[0].sha256[0] ^= 1;
    EXPECT_FALSE(rt::CheckComponentDrafterAdmission(source, bad_origins));
  }
}

TEST(ComponentAdmissionDrafterTest, ComponentDrafterValuesApprovesV1DraftersAndRejectsMismatches) {
  const std::filesystem::path scratch =
      std::filesystem::path(testing::TempDir()) / "component-drafter-values-v1";
  std::error_code ec;
  std::filesystem::remove_all(scratch, ec);
  std::filesystem::create_directories(scratch, ec);

  const auto make_v1 = [&](const std::filesystem::path& path, std::string arch,
                           std::string source_name, std::uint64_t bytes, std::string sha256) {
    llmp::test::V1Plan p;
    p.architecture = std::move(arch);
    p.groups = {llmp::test::V1Group{
        .kind = llmp::artifact::GroupKind::kTable,
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
    return llmp::artifact::Artifact::Open(llmp::test::Publish(path, rendered));
  };

  const auto g26_target =
      make_v1(scratch / "g26", "gemma4", "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf", 16947541728ULL,
              "f2c28b3dc4776931ac6f879e11f203dec637ea0f14267a86ec8f6165f63f293f");
  ASSERT_TRUE(g26_target.has_value()) << (g26_target ? "" : g26_target.error().ToString());

  const auto g31_target =
      make_v1(scratch / "g31", "gemma4", "gemma-4-31B-it-UD-Q4_K_XL.gguf", 18822970304ULL,
              "9e92cb6236044c6a9870af406029c74a76e0571c157a6f95df724dcc8c7a1575");
  ASSERT_TRUE(g31_target.has_value()) << (g31_target ? "" : g31_target.error().ToString());

  const auto g26_asst =
      make_v1(scratch / "a26", "gemma4-assistant", "mtp-gemma-4-26B-A4B-it.gguf", 461766816ULL,
              "6326fb9f5e487aa8dcdd313a091e3c67724cb2a666ec3b7d2895b5b26d93ed1b");
  ASSERT_TRUE(g26_asst.has_value()) << (g26_asst ? "" : g26_asst.error().ToString());

  const auto g31_asst =
      make_v1(scratch / "a31", "gemma4-assistant", "mtp-gemma-4-31B-it.gguf", 514687104ULL,
              "5ae8b0117bed601e8924c6305bd5b0585de361d51f0e77091bcb4252cf1f27de");
  ASSERT_TRUE(g31_asst.has_value()) << (g31_asst ? "" : g31_asst.error().ToString());

  const auto dflash =
      make_v1(scratch / "dflash", "dflash", "dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf",
              10896057440ULL, "2c7ac54b0b64a99df1f139a9f1371a00198265e1d6a614b77597d20a655a4249");
  ASSERT_TRUE(dflash.has_value()) << (dflash ? "" : dflash.error().ToString());

  // Matched approved pairs pass their pairing; their values are the stored
  // drafter's kept metadata, which these synthetic artifacts lack.
  auto g26_pair = rt::ComponentDrafterValues(*g26_target, *g26_asst);
  ASSERT_FALSE(g26_pair.has_value());
  EXPECT_NE(g26_pair.error().find("keeps no GGUF metadata"), std::string::npos) << g26_pair.error();

  auto g31_pair = rt::ComponentDrafterValues(*g31_target, *g31_asst);
  ASSERT_FALSE(g31_pair.has_value());
  EXPECT_NE(g31_pair.error().find("keeps no GGUF metadata"), std::string::npos) << g31_pair.error();

  // Size mismatches are refused: each assistant's approval names its target.
  auto size_mismatch1 = rt::ComponentDrafterValues(*g26_target, *g31_asst);
  ASSERT_FALSE(size_mismatch1.has_value());
  EXPECT_NE(size_mismatch1.error().find("is gemma-4-31b-it-ud-q4_k_xl's, not "
                                        "gemma-4-26b-a4b-it-ud-q4_k_m's"),
            std::string::npos)
      << size_mismatch1.error();

  auto size_mismatch2 = rt::ComponentDrafterValues(*g31_target, *g26_asst);
  ASSERT_FALSE(size_mismatch2.has_value());
  EXPECT_NE(size_mismatch2.error().find("is gemma-4-26b-a4b-it-ud-q4_k_m's, not "
                                        "gemma-4-31b-it-ud-q4_k_xl's"),
            std::string::npos)
      << size_mismatch2.error();

  // Architecture mismatch is refused.
  auto arch_mismatch = rt::ComponentDrafterValues(*g26_target, *dflash);
  ASSERT_FALSE(arch_mismatch.has_value());
  EXPECT_NE(arch_mismatch.error().find("the approved drafter dspark-deepseek-v4-flash-0731-q8_0"),
            std::string::npos)
      << arch_mismatch.error();

  // ComponentDrafterV0Values refuses v1 artifacts.
  auto v0_attempt = rt::ComponentDrafterV0Values(*g26_target, *g26_asst);
  EXPECT_FALSE(v0_attempt.has_value());
}

TEST(ComponentAdmissionModelOptTest,
     FullNormalizedTargetPreservesNativeProfilePolicyAndClosedSourceAuthority) {
  auto fixture = llmp::test_support::qwen38::NativeModelOpt();
  auto origins = llmp::test_support::qwen38::ModelOptOrigins();
  const auto& model = fixture.model;
  const auto& source = fixture.source;
  const auto admitted = rt::CheckComponentAdmission(model, source, origins);
  ASSERT_TRUE(admitted) << (admitted ? "" : admitted.error());

  auto policy =
      rt::ComponentAdmissionPolicy(model, source, origins, {.context = 262144, .slots = 8});
  ASSERT_TRUE(policy) << (policy ? "" : policy.error());
  // The native mapping's fast form.
  EXPECT_EQ(policy->text, llmp::spec::mappings::Qwen38NativeMapping().policies.front().text);
  EXPECT_FALSE(
      rt::ComponentAdmissionPolicy(model, source, origins, {.context = 262145, .slots = 1}));
  EXPECT_FALSE(rt::ComponentAdmissionPolicy(model, source, origins, {.context = 0, .slots = 1}));
  EXPECT_FALSE(
      rt::ComponentAdmissionPolicy(model, source, origins, {.context = 262144, .slots = 0}));

  for (unsigned mutation = 0; mutation != 8; ++mutation) {
    auto changed_model = model;
    auto changed_source = source;
    auto changed_origins = origins;
    if (mutation == 0) changed_origins[0].sha256[0] ^= 1;
    if (mutation == 1)
      changed_origins[0].bytes = llmp::base::Bytes(changed_origins[0].bytes.value() + 1);
    if (mutation == 2) changed_source.container = "gguf";
    if (mutation == 3) changed_source.folds.clear();
    if (mutation == 4) changed_model.stages[0].folds.clear();
    if (mutation == 5) changed_source.tensors[0].format.encoding = fm::Encoding::kF16;
    if (mutation == 6) {
      auto it = std::ranges::find_if(changed_source.tensors, [](const auto& t) {
        return t.format.encoding == fm::Encoding::kNvfp4 && t.format.params.tensor_scale;
      });
      ASSERT_NE(it, changed_source.tensors.end());
      it->format.params.tensor_scale = false;
    }
    if (mutation == 7) {
      changed_model.stages[0].heads.output->weights[0].tensor =
          changed_model.stages[0].embedding->weights[0].tensor;
    }
    EXPECT_FALSE(rt::CheckComponentAdmission(changed_model, changed_source, changed_origins))
        << mutation;
  }

  const std::filesystem::path scratch =
      std::filesystem::path(testing::TempDir()) / "component-admission-modelopt";
  std::error_code ec;
  std::filesystem::remove_all(scratch, ec);
  std::filesystem::create_directories(scratch, ec);

  llmp::test::V1Plan plan;
  plan.architecture = "qwen4exp";
  plan.groups = {
      llmp::test::V1Group{.kind = llmp::artifact::GroupKind::kTable,
                          .resources = {{.name = "token_embd.weight",
                                         .format = {.encoding = fm::Encoding::kBF16,
                                                    .layout = fm::Layout::kDense,
                                                    .shape = {.ne = {64, 64, 1, 1}, .rank = 2}}}}}};
  auto rendered = llmp::test::Render(plan);
  llmp::test::Json::Array source_list;
  for (const auto& o : origins) {
    source_list.emplace_back(
        llmp::test::Json::Object{{"bytes", static_cast<std::int64_t>(o.bytes.value())},
                                 {"name", o.name},
                                 {"sha256", llmp::base::ToHex(o.sha256)}});
  }
  rendered.manifest["source"] = llmp::test::Json(std::move(source_list));
  auto art = llmp::artifact::Artifact::Open(llmp::test::Publish(scratch / "artifact", rendered));
  ASSERT_TRUE(art.has_value()) << (art ? "" : art.error().ToString());
  EXPECT_FALSE(rt::CheckComponentArtifactAdmission(*art));
  EXPECT_FALSE(rt::ReadComponentAdmission(*art));
}

TEST(ComponentAdmissionMlxTest,
     FullNormalizedTargetPreservesNativeProfilePolicyAndClosedSourceAuthority) {
  auto fixture = llmp::test_support::qwen38::NativeMlx();
  auto origins = llmp::test_support::qwen38::MlxOrigins();
  const auto& model = fixture.model;
  const auto& source = fixture.source;
  const auto admitted = rt::CheckComponentAdmission(model, source, origins);
  ASSERT_TRUE(admitted) << (admitted ? "" : admitted.error());

  auto policy =
      rt::ComponentAdmissionPolicy(model, source, origins, {.context = 262144, .slots = 8});
  ASSERT_TRUE(policy) << (policy ? "" : policy.error());
  EXPECT_TRUE(policy->text.empty());
  EXPECT_FALSE(
      rt::ComponentAdmissionPolicy(model, source, origins, {.context = 262145, .slots = 1}));
  EXPECT_FALSE(rt::ComponentAdmissionPolicy(model, source, origins, {.context = 0, .slots = 1}));
  EXPECT_FALSE(
      rt::ComponentAdmissionPolicy(model, source, origins, {.context = 262144, .slots = 0}));

  for (unsigned mutation = 0; mutation != 9; ++mutation) {
    auto changed_model = model;
    auto changed_source = source;
    auto changed_origins = origins;
    if (mutation == 0) changed_origins[0].sha256[0] ^= 1;
    if (mutation == 1)
      changed_origins[0].bytes = llmp::base::Bytes(changed_origins[0].bytes.value() + 1);
    if (mutation == 2) changed_source.container = "safetensors";
    if (mutation == 3) changed_source.folds.clear();
    if (mutation == 4) changed_model.stages[0].folds.clear();
    if (mutation == 5) changed_source.tensors[0].format.encoding = fm::Encoding::kF16;
    if (mutation == 6) {
      auto it = std::ranges::find_if(changed_source.tensors, [](const auto& t) {
        return t.format.encoding == fm::Encoding::kAffine;
      });
      ASSERT_NE(it, changed_source.tensors.end());
      it->format.params.bits = 8;
    }
    if (mutation == 7) {
      changed_model.stages[0].heads.output->weights[0].tensor =
          changed_model.stages[0].embedding->weights[0].tensor;
    }
    if (mutation == 8) {
      auto it = std::ranges::find_if(changed_source.tensors, [](const auto& t) {
        return t.format.encoding == fm::Encoding::kAffine;
      });
      ASSERT_NE(it, changed_source.tensors.end());
      it->format.params.zero_point = true;
    }
    EXPECT_FALSE(rt::CheckComponentAdmission(changed_model, changed_source, changed_origins))
        << mutation;
  }

  const std::filesystem::path scratch =
      std::filesystem::path(testing::TempDir()) / "component-admission-mlx";
  std::error_code ec;
  std::filesystem::remove_all(scratch, ec);
  std::filesystem::create_directories(scratch, ec);

  llmp::test::V1Plan plan;
  plan.architecture = "qwen4exp";
  plan.groups = {
      llmp::test::V1Group{.kind = llmp::artifact::GroupKind::kTable,
                          .resources = {{.name = "token_embd.weight",
                                         .format = {.encoding = fm::Encoding::kBF16,
                                                    .layout = fm::Layout::kDense,
                                                    .shape = {.ne = {64, 64, 1, 1}, .rank = 2}}}}}};
  auto rendered = llmp::test::Render(plan);
  llmp::test::Json::Array source_list;
  for (const auto& o : origins) {
    source_list.emplace_back(
        llmp::test::Json::Object{{"bytes", static_cast<std::int64_t>(o.bytes.value())},
                                 {"name", o.name},
                                 {"sha256", llmp::base::ToHex(o.sha256)}});
  }
  rendered.manifest["source"] = llmp::test::Json(std::move(source_list));
  auto art = llmp::artifact::Artifact::Open(llmp::test::Publish(scratch / "artifact", rendered));
  ASSERT_TRUE(art.has_value()) << (art ? "" : art.error().ToString());
  EXPECT_FALSE(rt::CheckComponentArtifactAdmission(*art));
  EXPECT_FALSE(rt::ReadComponentAdmission(*art));
}

}  // namespace

// The scale-atom experts' slot (the block-scaled grouped GEMM's fixed
// offsets): the stacked gate-up table's codes and scales, then down's, back
// to back from each expert group's start.
TEST(ComponentAdmissionScaleAtomTest, SlotsHoldTheStackedTableFirstBackToBack) {
  const auto atoms = [](std::uint64_t k, std::uint64_t n, std::uint8_t stacked) {
    return fm::TensorFormat{.encoding = fm::Encoding::kNvfp4,
                            .layout = fm::Layout::kScaleAtoms,
                            .shape = {.ne = {k, n, 1, 1}, .rank = 2},
                            .stacked = stacked};
  };
  const std::vector<llmp::test::V1Array> gate_up{{.name = "blk.0.ffn_gate_up_exps.weight",
                                                  .format = atoms(64, 256, 2),
                                                  .part = fm::PartRole::kCodes},
                                                 {.name = "blk.0.ffn_gate_up_exps.weight_scale",
                                                  .format = atoms(64, 256, 2),
                                                  .part = fm::PartRole::kBlockScales}};
  const std::vector<llmp::test::V1Array> down{{.name = "blk.0.ffn_down_exps.weight",
                                               .format = atoms(128, 128, 1),
                                               .part = fm::PartRole::kCodes},
                                              {.name = "blk.0.ffn_down_exps.weight_scale",
                                               .format = atoms(128, 128, 1),
                                               .part = fm::PartRole::kBlockScales}};
  const std::filesystem::path scratch =
      std::filesystem::path(testing::TempDir()) / "component-admission-scale-atoms";
  std::error_code ec;
  std::filesystem::remove_all(scratch, ec);
  const auto open = [&](std::string_view name, std::vector<llmp::test::V1Array> arrays) {
    llmp::test::V1Plan plan;
    plan.architecture = "qwen4exp";
    plan.expert_count = 2;
    plan.groups = {llmp::test::V1Group{
        .kind = llmp::artifact::GroupKind::kExpert, .layer = 0, .arrays = std::move(arrays)}};
    return llmp::artifact::Artifact::Open(
        llmp::test::Publish(scratch / name, llmp::test::Render(plan)));
  };
  auto first = gate_up;
  first.insert(first.end(), down.begin(), down.end());
  auto good = open("good", first);
  ASSERT_TRUE(good) << good.error().ToString();
  EXPECT_TRUE(rt::CheckScaleAtomSlots(*good));
  auto last = down;
  last.insert(last.end(), gate_up.begin(), gate_up.end());
  auto swapped = open("swapped", last);
  ASSERT_TRUE(swapped) << swapped.error().ToString();
  EXPECT_FALSE(rt::CheckScaleAtomSlots(*swapped));
  auto gap = first;
  gap[2].offset = 8192 + 1024 + 256;  // down's codes past a gap
  gap[3].offset = 8192 + 1024 + 256 + 8192;
  auto spaced = open("spaced", gap);
  ASSERT_TRUE(spaced) << spaced.error().ToString();
  EXPECT_FALSE(rt::CheckScaleAtomSlots(*spaced));
}
