// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "runtime/image_stage.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <utility>
#include <variant>

#include "expected_error.h"
#include "kernels/reference/exec.h"
#include "spec/image_spatial.h"
#include "spec/mappings/qwen_image.h"

namespace llmp::runtime {
namespace {
namespace co = components;
namespace ref = kernels::reference;
using test_support::Failed;

// Two residual graphs sharing the same components and weights. Only the
// second residual's saved value differs: 2x+x versus 2x+2x. This topology is
// supplied as serialized spec data to the same binder used by serving.
spec::ModelSpec Decoder(std::uint32_t factor, bool doubled) {
  spec::Stage stage{.name = "alternate", .role = spec::StageRole::kVae};
  stage.stream.width = 1;
  stage.layers = {{{
      {.kind = "reshape.spatial",
       .params = {{"height", std::int64_t{1}}, {"width", std::int64_t{1}}}},
      {.kind = "residual.spatial"},
      {.kind = "residual.spatial"},
      {.kind = "upsample.nearest", .params = {{"factor", std::int64_t{2}}}},
  }}};
  stage.links = {{0, {}}, {1, 1}, {2, doubled ? 2 : 1}, {3, {}}};
  stage.exports = {4};
  if (factor == 4) {
    stage.layers[0].components.push_back(
        {.kind = "upsample.nearest", .params = {{"factor", std::int64_t{2}}}});
    stage.links.push_back({4, {}});
    stage.exports = {5};
  }
  stage.scalar_bindings = {{1, "height", "latent_height"}, {1, "width", "latent_width"}};
  spec::ModelSpec result{.pipeline = spec::PipelineKind::kImage, .stages = {std::move(stage)}};
  spec::Canonicalize(result);
  return result;
}
model::ImageProfile Profile(std::uint32_t factor) {
  model::ImageProfile profile;
  profile.vae.spatial_factor = factor;
  profile.vae.z_dim = 1;
  profile.vae.out_channels = 1;
  return profile;
}
struct Budget {
  std::uint64_t used = 0;
  co::SpatialPayment Payment() {
    return {[this](std::uint64_t bytes) {
              used += bytes;
              return true;
            },
            [this](std::uint64_t bytes) {
              EXPECT_LE(bytes, used);
              used -= bytes;
            }};
  }
};
constexpr std::array<float, 6> kInput{1, -2, 3, -4, 5, -6};
constexpr std::array<co::SpatialOwner, 1> kOwners{{{0, {1, 6, 1, 1}}}};
const std::array<std::span<const co::BoundTensor>, 1> kTables{};

TEST(ImageStage, SerializedAlternateWiringDrivesRuntimeBindingAndArithmetic) {
  for (const auto factor : {2U, 4U}) {
    const auto profile = Profile(factor);
    for (const auto shape : {std::array{3U, 2U}, std::array{2U, 3U}}) {
      const ImageStageGeometry geometry{shape[0] * factor, shape[1] * factor, 5, 2};
      for (const bool doubled : {false, true}) {
        const auto source = Decoder(factor, doubled);
        const auto bytes = spec::ToJson(source);
        auto stored = spec::FromJson(bytes);
        ASSERT_TRUE(stored) << Failed(stored).value_or("");
        const auto digest = stored->Digest();
        auto bound = BindImageStage(*stored, profile, geometry);
        ASSERT_TRUE(bound) << Failed(bound).value_or("");
        EXPECT_EQ(spec::ToJson(*stored), bytes);
        EXPECT_EQ(stored->Digest(), digest);
        EXPECT_EQ(bound->stages[0].links, stored->stages[0].links);
        EXPECT_EQ(bound->stages[0].exports, stored->stages[0].exports);
        EXPECT_TRUE(bound->stages[0].scalar_bindings.empty());
        Budget budget;
        {
          auto graph = co::BuildSpatialStep(*bound, kTables, kOwners, budget.Payment());
          ASSERT_TRUE(graph) << Failed(graph).value_or("");
          ASSERT_TRUE(ValidateImageStageOutputs(spec::StageRole::kVae, graph->outputs(), profile,
                                                geometry));
          auto executor = ref::Executor::Create(graph->graph());
          ASSERT_TRUE(executor) << Failed(executor, &ref::ExecFailure::detail).value_or("");
          ASSERT_TRUE(executor->BindValues(graph->graph().inputs()[0], kInput));
          ASSERT_TRUE(executor->Run());
          auto values = executor->Values(graph->graph().outputs()[0]);
          ASSERT_TRUE(values);
          ASSERT_EQ(values->size(), geometry.width * geometry.height);
          for (std::uint32_t y = 0; y < geometry.height; ++y)
            for (std::uint32_t x = 0; x < geometry.width; ++x) {
              const auto expected =
                  kInput[(y / factor) * shape[0] + x / factor] * (doubled ? 4.0F : 3.0F);
              EXPECT_FLOAT_EQ((*values)[y * geometry.width + x], expected);
            }
        }
        EXPECT_EQ(budget.used, 0U);
      }
    }
  }
}

TEST(ImageStage, RefusesUnresolvedUnknownAndInconsistentGeometryWithoutRegeneration) {
  const auto profile = Profile(2);
  const auto source = Decoder(2, false);
  Budget budget;
  EXPECT_FALSE(co::BuildSpatialStep(source, kTables, kOwners, budget.Payment()));
  EXPECT_EQ(budget.used, 0U);
  auto unknown = source;
  unknown.stages[0].scalar_bindings[0].source = "unknown_dimension";
  EXPECT_FALSE(BindImageStage(unknown, profile, {6, 4, 5, 2}));
  EXPECT_FALSE(BindImageStage(source, profile, {7, 4, 5, 2}));
  EXPECT_FALSE(BindImageStage(source, profile, {6, 4, 2, 2}));
  EXPECT_FALSE(BindImageStage(source, Profile(1), {UINT32_MAX, UINT32_MAX, 1, 0}));
  EXPECT_FALSE(BindImageStage(source, Profile(0), {6, 4, 5, 2}));

  // A fixed graph is valid at its own geometry; it is not regenerated to fit
  // a different rectangle with the same area or a conflicting family scale.
  auto fixed = BindImageStage(source, profile, {6, 4, 5, 2});
  ASSERT_TRUE(fixed);
  auto same = BindImageStage(*fixed, profile, {6, 4, 5, 2});
  ASSERT_TRUE(same);
  auto graph = co::BuildSpatialStep(*same, kTables, kOwners, budget.Payment());
  ASSERT_TRUE(graph);
  EXPECT_TRUE(
      ValidateImageStageOutputs(spec::StageRole::kVae, graph->outputs(), profile, {6, 4, 5, 2}));
  EXPECT_FALSE(
      ValidateImageStageOutputs(spec::StageRole::kVae, graph->outputs(), profile, {4, 6, 5, 2}));
  EXPECT_FALSE(ValidateImageStageOutputs(spec::StageRole::kVae, graph->outputs(), Profile(4),
                                         {12, 8, 5, 2}));
}

TEST(ImageStage, BindingMetadataIsValidatedAndChangesStoredIdentity) {
  const auto source = Decoder(2, false);
  EXPECT_NE(source.Digest(), Decoder(2, true).Digest());
  auto renamed = source;
  renamed.stages[0].scalar_bindings[0].source = "another_height";
  EXPECT_NE(source.Digest(), renamed.Digest());
  for (const auto id : {0U, 5U}) {
    auto invalid = source;
    invalid.stages[0].scalar_bindings[0].component = static_cast<std::uint16_t>(id);
    spec::Canonicalize(invalid);
    EXPECT_FALSE(spec::Validate(invalid));
    EXPECT_FALSE(spec::FromJson(spec::ToJson(invalid)));
  }
  auto duplicate = source;
  duplicate.stages[0].scalar_bindings.push_back(duplicate.stages[0].scalar_bindings[0]);
  spec::Canonicalize(duplicate);
  EXPECT_FALSE(spec::Validate(duplicate));
  auto forged = source;
  forged.stages[0].scalar_bindings[0].parameter = "nonexistent";
  spec::Canonicalize(forged);
  EXPECT_FALSE(spec::Validate(forged));
  auto wrong_type = source;
  wrong_type.stages[0].layers[0].components[0].params[0].value = true;
  EXPECT_FALSE(spec::Validate(wrong_type));
  auto bad_name = source;
  bad_name.stages[0].scalar_bindings[0].source = "BadName";
  EXPECT_FALSE(spec::Validate(bad_name));
  const std::array<spec::StageScalar, 1> missing{{{"latent_height", 2}}};
  EXPECT_FALSE(spec::BindStageScalars(source, missing));
  const std::array<spec::StageScalar, 3> repeated{
      {{"latent_height", 2}, {"latent_height", 2}, {"latent_width", 3}}};
  EXPECT_FALSE(spec::BindStageScalars(source, repeated));
  const std::array<spec::StageScalar, 2> invalid_value{
      {{"latent_height", -1}, {"latent_width", 3}}};
  auto bound = spec::BindStageScalars(source, invalid_value);
  ASSERT_TRUE(bound);  // The component owns its parameter range.
  Budget budget;
  EXPECT_FALSE(co::BuildSpatialStep(*bound, kTables, kOwners, budget.Payment()));
}

TEST(ImageStage, ImportedTemplatesPreserveSourceGraphsAcrossRequestGeometries) {
  auto profile = Profile(2);
  profile.vae.base = 2;
  profile.vae.dim_mult = {1, 2};
  profile.vae.res_blocks = 0;
  profile.vae.temporal_upsample = {false};
  profile.denoiser = {.blocks = 1,
                      .width = 6,
                      .heads = 1,
                      .head_dim = 6,
                      .mlp = 12,
                      .in_channels = 1,
                      .out_channels = 1,
                      .context = 6,
                      .timestep_dim = 4,
                      .rope_axes = {2, 2, 2},
                      .rope_theta = 10000,
                      .eps = 1e-6F};
  const auto& names = spec::mappings::QwenImage21Mapping().names;
  const auto vae_tensors = model::VaeDecoderTensors(profile.vae, names.vae);
  const auto flow_tensors = model::DenoiserTensors(profile.denoiser, names.denoiser);
  auto vae = spec::ImageDecodeSpec(profile.vae, 1, 1, vae_tensors);
  auto flow = spec::ImageFlowSpec(profile.denoiser, 1, 1, flow_tensors);
  ASSERT_TRUE(vae) << Failed(vae).value_or("");
  ASSERT_TRUE(flow) << Failed(flow).value_or("");
  vae->spec.stages[0].scalar_bindings = {{1, "height", "latent_height"},
                                         {1, "width", "latent_width"}};
  const auto output_slice = static_cast<std::uint16_t>(15 + 6 * profile.denoiser.blocks);
  flow->spec.stages[0].scalar_bindings = {
      {8, "input_rows", "latent_rows"},           {8, "rows", "latent_rows"},
      {11, "prefix_rows", "text_rows"},           {12, "prefix_rows", "text_rows"},
      {13, "prefix_rows", "text_rows"},           {14, "prefix_rows", "text_rows"},
      {output_slice, "input_rows", "joint_rows"}, {output_slice, "rows", "latent_rows"},
      {output_slice, "start", "text_rows"}};
  for (const auto geometry : {ImageStageGeometry{6, 4, 5, 2}, ImageStageGeometry{4, 8, 7, 1}}) {
    auto dimensions = ImageStageDimensionsOf(profile, geometry);
    ASSERT_TRUE(dimensions);
    auto decoded = BindImageStage(vae->spec, profile, geometry);
    auto denoised = BindImageStage(flow->spec, profile, geometry);
    auto old_decode =
        spec::ImageDecodeSpec(profile.vae, dimensions->height, dimensions->width, vae_tensors);
    auto old_flow = spec::ImageFlowSpec(profile.denoiser, dimensions->text_rows, dimensions->rows,
                                        flow_tensors);
    ASSERT_TRUE(decoded) << Failed(decoded).value_or("");
    ASSERT_TRUE(denoised) << Failed(denoised).value_or("");
    ASSERT_TRUE(old_decode) << Failed(old_decode).value_or("");
    ASSERT_TRUE(old_flow) << Failed(old_flow).value_or("");
    EXPECT_EQ(*decoded, old_decode->spec);
    EXPECT_EQ(*denoised, old_flow->spec);
  }
}
}  // namespace
}  // namespace llmp::runtime
