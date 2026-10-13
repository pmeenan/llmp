// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "runtime/component_admission.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <cstdint>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "base/bytes.h"
#include "base/sha256.h"
#include "dsv4_gguf_normalization_fixture.h"
#include "import/normalize.h"
#include "model/gemma2.h"
#include "model/gemma3.h"
#include "model/gemma4.h"
#include "qwen38_gguf_normalization_fixture.h"
#include "runtime/calibration.h"
#include "spec/mappings/gemma2.h"
#include "spec/mappings/gemma3.h"
#include "spec/mappings/gemma4.h"
#include "spec_fixtures.h"

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
  EXPECT_FALSE(rt::OnComponentEngine("qwen4exp"));  // v0 default is unchanged
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
  EXPECT_FALSE(rt::OnComponentEngine("deepseek4"));
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

}  // namespace
