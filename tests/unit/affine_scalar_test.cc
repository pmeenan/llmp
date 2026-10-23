// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <vector>

#include "components/spatial.h"
#include "components/step.h"
#include "expected_error.h"
#include "kernels/reference/exec.h"
#include "spec/spec.h"

namespace llmp::components {
namespace {
namespace ref = kernels::reference;
using test_support::Failed;

TEST(AffineScalarTest, BuildSpatialStepPreservesShapeAndMatchesFloatScaleBias) {
  spec::Stage stage{.name = "vae", .role = spec::StageRole::kVae, .artifact = 0};
  stage.stream.width = 3;
  stage.layers.push_back({.components = {{.kind = "affine.scalar",
                                          .version = 1,
                                          .params = {{"bias", 0.375}, {"scale", -1.25}}}}});
  spec::ModelSpec model{.pipeline = spec::PipelineKind::kImage, .stages = {stage}};
  spec::Canonicalize(model);
  const std::array<std::span<const BoundTensor>, 1> tables{std::span<const BoundTensor>{}};
  const std::array owners{SpatialOwner{0, {2, 1, 3, 1}}};
  SpatialPayment payment{[](std::uint64_t) { return true; }, [](std::uint64_t) {}};
  auto graph = BuildSpatialStep(model, tables, owners, payment);
  ASSERT_TRUE(graph) << Failed(graph).value_or("");
  ASSERT_EQ(graph->outputs().size(), 1U);
  EXPECT_EQ(graph->outputs()[0].shape, owners[0].shape);

  auto executor = ref::Executor::Create(graph->graph());
  ASSERT_TRUE(executor) << Failed(executor, &ref::ExecFailure::detail).value_or("");
  const std::array<float, 6> input{0.0F, 1.0F, -2.0F, 3.25F, 1.0e-7F, 8.0F};
  ASSERT_TRUE(executor->BindValues(graph->graph().inputs()[0], input));
  ASSERT_TRUE(executor->Run());
  auto actual = executor->Values(graph->graph().outputs()[0]);
  ASSERT_TRUE(actual);
  ASSERT_EQ(actual->size(), input.size());
  for (std::size_t i = 0; i < input.size(); ++i) {
    const auto expected = static_cast<float>(input[i] * -1.25F + 0.375F);
    EXPECT_EQ((*actual)[i], expected) << i;
  }
}

TEST(AffineScalarTest, RejectsParametersThatCannotBeRepresentedAsFiniteFloat) {
  const auto* def = Find("affine.scalar", 1);
  ASSERT_NE(def, nullptr);
  for (const auto bad_scale :
       {std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN(),
        static_cast<double>(std::numeric_limits<float>::max()) * 2.0}) {
    spec::ComponentRef ref{
        .kind = "affine.scalar", .version = 1, .params = {{"bias", 0.0}, {"scale", bad_scale}}};
    auto params = Params::Of(def->params, ref);
    EXPECT_FALSE(params);
  }
  for (const auto bad_bias :
       {std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN(),
        static_cast<double>(std::numeric_limits<float>::max()) * 2.0}) {
    spec::ComponentRef ref{
        .kind = "affine.scalar", .version = 1, .params = {{"bias", bad_bias}, {"scale", 1.0}}};
    auto params = Params::Of(def->params, ref);
    EXPECT_FALSE(params);
  }
}

TEST(AffineScalarTest, SpatialReferenceAndStoredSpecRoundTripRetainScalarParameters) {
  const auto* def = Find("affine.scalar", 1);
  ASSERT_NE(def, nullptr);
  spec::ComponentRef ref{
      .kind = "affine.scalar", .version = 1, .params = {{"bias", -0.25}, {"scale", 0.0}}};
  auto params = Params::Of(def->params, ref);
  ASSERT_TRUE(params);
  Instance instance{.def = def, .ref = &ref, .params = std::move(*params), .stage = {.width = 2}};
  const std::array<double, 4> input{-1.0, 0.0, 2.0, 8.0};
  SpatialRefIO reference{.instance = instance, .shape = {2, 1, 2, 1}, .input = input};
  ASSERT_TRUE(def->spatial->reference(reference));
  EXPECT_EQ(reference.output, (std::vector<double>{-0.25, -0.25, -0.25, -0.25}));
  auto inferred = def->spatial->infer(instance, {2, 1, 2, 1});
  ASSERT_TRUE(inferred);
  EXPECT_EQ(*inferred, (SpatialShape{2, 1, 2, 1}));
  EXPECT_FALSE(def->spatial->infer(instance, {0, 1, 2, 1}));
  EXPECT_FALSE(def->spatial->infer(instance, {2, 1, 2, 1}, SpatialShape{1, 1, 2, 1}));

  spec::Stage stage{.name = "vae", .role = spec::StageRole::kVae, .artifact = 0};
  stage.stream.width = 2;
  stage.layers.push_back({.components = {ref}});
  spec::ModelSpec model{.pipeline = spec::PipelineKind::kImage, .stages = {stage}};
  spec::Canonicalize(model);
  auto decoded = spec::FromJson(spec::ToJson(model));
  ASSERT_TRUE(decoded);
  ASSERT_EQ(decoded->stages[0].layers[0].components.size(), 1U);
  const auto& stored = decoded->stages[0].layers[0].components[0];
  ASSERT_NE(stored.Find("scale"), nullptr);
  ASSERT_NE(stored.Find("bias"), nullptr);
  EXPECT_EQ(std::get<double>(stored.Find("scale")->value), 0.0);
  EXPECT_EQ(std::get<double>(stored.Find("bias")->value), -0.25);
}

TEST(ActivationElementwiseTest, BuildSpatialStepMatchesIndependentSiluReference) {
  spec::Stage stage{.name = "denoiser", .role = spec::StageRole::kDenoiser, .artifact = 0};
  stage.stream.width = 3;
  stage.layers.push_back({.components = {{.kind = "activation.elementwise",
                                          .version = 1,
                                          .params = {{"activation", std::string("silu")}}}}});
  spec::ModelSpec model{.pipeline = spec::PipelineKind::kImage, .stages = {stage}};
  spec::Canonicalize(model);
  const std::array<std::span<const BoundTensor>, 1> tables{std::span<const BoundTensor>{}};
  const std::array owners{SpatialOwner{0, {1, 2, 3, 1}}};
  SpatialPayment payment{[](std::uint64_t) { return true; }, [](std::uint64_t) {}};
  auto graph = BuildSpatialStep(model, tables, owners, payment);
  ASSERT_TRUE(graph) << Failed(graph).value_or("");
  EXPECT_EQ(graph->outputs()[0].shape, owners[0].shape);
  auto executor = ref::Executor::Create(graph->graph());
  ASSERT_TRUE(executor) << Failed(executor, &ref::ExecFailure::detail).value_or("");
  const std::array<float, 6> input{-4.0F, -0.5F, 0.0F, 0.25F, 2.0F, 8.0F};
  ASSERT_TRUE(executor->BindValues(graph->graph().inputs()[0], input));
  ASSERT_TRUE(executor->Run());
  auto output = executor->Values(graph->graph().outputs()[0]);
  ASSERT_TRUE(output);
  ASSERT_EQ(output->size(), input.size());
  for (std::size_t i = 0; i < input.size(); ++i) {
    const float expected = input[i] / (1.0F + std::exp(-input[i]));
    EXPECT_NEAR((*output)[i], expected, 2e-7F) << i;
  }
}

TEST(ActivationElementwiseTest, RejectsUnknownActivation) {
  const auto* def = Find("activation.elementwise", 1);
  ASSERT_NE(def, nullptr);
  spec::ComponentRef ref{.kind = "activation.elementwise",
                         .version = 1,
                         .params = {{"activation", std::string("gelu")}}};
  EXPECT_FALSE(Params::Of(def->params, ref));
}

TEST(ActivationElementwiseTest, HardtanhBuildMatchesScalarReferenceAndClampsToUnitRange) {
  spec::Stage stage{.name = "vae", .role = spec::StageRole::kVae, .artifact = 0};
  stage.stream.width = 3;
  stage.layers.push_back({.components = {{.kind = "activation.elementwise",
                                          .version = 1,
                                          .params = {{"activation", std::string("hardtanh")}}}}});
  spec::ModelSpec model{.pipeline = spec::PipelineKind::kImage, .stages = {stage}};
  spec::Canonicalize(model);
  const std::array<std::span<const BoundTensor>, 1> tables{std::span<const BoundTensor>{}};
  const std::array owners{SpatialOwner{0, {2, 1, 3, 1}}};
  SpatialPayment payment{[](std::uint64_t) { return true; }, [](std::uint64_t) {}};
  auto graph = BuildSpatialStep(model, tables, owners, payment);
  ASSERT_TRUE(graph) << Failed(graph).value_or("");
  auto executor = ref::Executor::Create(graph->graph());
  ASSERT_TRUE(executor) << Failed(executor, &ref::ExecFailure::detail).value_or("");
  const std::array<float, 6> input{-3.0F, -1.0F, -0.25F, 0.5F, 1.0F, 2.0F};
  ASSERT_TRUE(executor->BindValues(graph->graph().inputs()[0], input));
  ASSERT_TRUE(executor->Run());
  auto output = executor->Values(graph->graph().outputs()[0]);
  ASSERT_TRUE(output);
  ASSERT_EQ(output->size(), input.size());
  for (std::size_t i = 0; i < input.size(); ++i)
    EXPECT_EQ((*output)[i], std::clamp(input[i], -1.0F, 1.0F)) << i;
}

TEST(ActivationElementwiseTest, SpatialReferenceAndStoredSpecRoundTripRetainActivation) {
  const auto* def = Find("activation.elementwise", 1);
  ASSERT_NE(def, nullptr);
  spec::ComponentRef ref{.kind = "activation.elementwise",
                         .version = 1,
                         .params = {{"activation", std::string("silu")}}};
  auto params = Params::Of(def->params, ref);
  ASSERT_TRUE(params);
  Instance instance{.def = def, .ref = &ref, .params = std::move(*params), .stage = {.width = 2}};
  const std::array<double, 4> input{-2.0, -0.25, 0.5, 3.0};
  SpatialRefIO reference{.instance = instance, .shape = {2, 1, 2, 1}, .input = input};
  ASSERT_TRUE(def->spatial->reference(reference));
  for (std::size_t i = 0; i < input.size(); ++i) {
    const auto x = static_cast<float>(input[i]);
    EXPECT_EQ(reference.output[i], static_cast<float>(x / (1.0F + std::exp(-x))));
  }
  EXPECT_FALSE(def->spatial->infer(instance, {2, 1, 2, 1}, SpatialShape{1, 1, 2, 1}));
  EXPECT_FALSE(def->spatial->infer(instance, {2, 1, 0, 1}));

  spec::Stage stage{.name = "denoiser", .role = spec::StageRole::kDenoiser, .artifact = 0};
  stage.stream.width = 2;
  stage.layers.push_back({.components = {ref}});
  spec::ModelSpec model{.pipeline = spec::PipelineKind::kImage, .stages = {stage}};
  spec::Canonicalize(model);
  auto decoded = spec::FromJson(spec::ToJson(model));
  ASSERT_TRUE(decoded);
  ASSERT_EQ(decoded->stages[0].layers[0].components.size(), 1U);
  const auto* activation = decoded->stages[0].layers[0].components[0].Find("activation");
  ASSERT_NE(activation, nullptr);
  EXPECT_EQ(std::get<std::string>(activation->value), "silu");
}

}  // namespace
}  // namespace llmp::components
