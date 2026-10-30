// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "import/stage_recipe.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <set>
#include <string>
#include <string_view>
#include <vector>

#include "format/tensor_format.h"
#include "import/image_pipeline.h"
#include "import/normalize.h"

namespace llmp::import {
namespace {

constexpr spec::RecipeNode kResidual{
    .name = "step.{i}",
    .component_kind = "residual.spatial",
    .input = {.kind = spec::RecipeRefKind::kNamedValue, .name = "in.{i}"},
    .auxiliary = {.kind = spec::RecipeRefKind::kNamedValue, .name = "in.{i}"},
    .has_auxiliary = true,
};
constexpr spec::RecipeStep kBodyStep{.kind = spec::RecipeStep::Kind::kNode, .node = &kResidual};
constexpr std::array kBody{kBodyStep};
constexpr spec::RecipeRepeat kRepeat{
    .index = "i",
    .count = {.kind = spec::RecipeValueKind::kLiteral, .literal = std::int64_t{2}},
    .body = kBody,
    .input = {.kind = spec::RecipeRefKind::kHostPort, .port = 7},
    .input_name = "in.{i}",
    .body_yield = "step.{i}",
    .yield = "final",
};
constexpr spec::RecipeStep kRepeatStep{.kind = spec::RecipeStep::Kind::kRepeat, .repeat = &kRepeat};
constexpr std::array kSteps{kRepeatStep};
constexpr std::array<spec::RecipeRef, 1> kExports{{
    {.kind = spec::RecipeRefKind::kNamedValue, .name = "final"},
}};
constexpr spec::RecipeSpatialShape kUnitShape{.width = {.literal = std::int64_t{1}},
                                              .height = {.literal = std::int64_t{1}},
                                              .channels = {.literal = std::int64_t{1}},
                                              .images = {.literal = std::int64_t{1}}};
constexpr std::array<spec::RecipeSpatialOperand, 8> kHostPorts{
    spec::RecipeSpatialOperand{.shape = kUnitShape},
    spec::RecipeSpatialOperand{.shape = kUnitShape},
    spec::RecipeSpatialOperand{.shape = kUnitShape},
    spec::RecipeSpatialOperand{.shape = kUnitShape},
    spec::RecipeSpatialOperand{.shape = kUnitShape},
    spec::RecipeSpatialOperand{.shape = kUnitShape},
    spec::RecipeSpatialOperand{.shape = kUnitShape},
    spec::RecipeSpatialOperand{.shape = kUnitShape}};
constexpr std::array<spec::RecipeSpatialShape, 1> kExpectedOutput{kUnitShape};
constexpr spec::StageRecipe kRecipe{
    .name = "recipe",
    .role = spec::StageRole::kVae,
    .stream_width = {.kind = spec::RecipeValueKind::kLiteral, .literal = std::int64_t{1}},
    .primary_input = {.shape = kUnitShape},
    .host_ports = kHostPorts,
    .steps = kSteps,
    .exports = kExports,
    .expected_outputs = kExpectedOutput,
};

TEST(StageRecipe, ExpandsFiniteRepeatAndCarriesHostPortOnlyToFirstIteration) {
  const spec::ModelSpec family{.pipeline = spec::PipelineKind::kImage};
  auto expanded = ExpandStageRecipe(kRecipe, family, 1U << 20U);
  ASSERT_TRUE(expanded) << expanded.error();
  ASSERT_EQ(expanded->stage.layers.size(), 1U);
  ASSERT_EQ(expanded->stage.layers[0].components.size(), 2U);
  ASSERT_EQ(expanded->stage.links.size(), 2U);
  EXPECT_EQ(expanded->stage.links[0].input, 0U);
  EXPECT_EQ(expanded->stage.links[0].input_port, 7U);
  EXPECT_EQ(expanded->stage.links[1].input, 1U);
  EXPECT_FALSE(expanded->stage.links[1].input_port);
  EXPECT_EQ(expanded->stage.exports, (std::vector<std::uint16_t>{2}));
}

TEST(StageRecipe, TensorIndexWorkGrowsWithLookupDepth) {
  static constexpr auto width = spec::ex::TensorDim("weight.{i}", 0);
  static constexpr std::array params{spec::RecipeParam{.name = "in_channels", .value = width},
                                     spec::RecipeParam{.name = "out_channels", .value = width}};
  static constexpr std::array weights{spec::RecipeWeight{.role = "weight", .tensor = "weight.{i}"}};
  static constexpr spec::RecipeNode node{
      .name = "step.{i}",
      .component_kind = "projection.spatial",
      .params = params,
      .weights = weights,
      .input = {.kind = spec::RecipeRefKind::kNamedValue, .name = "in.{i}"}};
  static constexpr std::array body{
      spec::RecipeStep{.kind = spec::RecipeStep::Kind::kNode, .node = &node}};
  auto repeat = kRepeat;
  repeat.body = body;
  const std::array steps{
      spec::RecipeStep{.kind = spec::RecipeStep::Kind::kRepeat, .repeat = &repeat}};
  auto recipe = kRecipe;
  recipe.steps = steps;
  std::uint64_t previous = 0;
  for (const auto count : {256U, 512U, 1024U}) {
    std::vector<Tensor> source;
    for (unsigned i = 0; i < count; ++i)
      source.push_back({.name = "weight." + std::to_string(i), .dims = {1, 1}});
    repeat.count = spec::ex::I(count);
    const auto cost = PreflightStageRecipe(recipe, {}, 1U << 24U, {}, source);
    ASSERT_TRUE(cost) << cost.error();
    RecordProperty("lookup_work_" + std::to_string(count), std::to_string(cost->expansion_work));
    if (previous != 0) EXPECT_LT(cost->expansion_work, 3 * previous);
    previous = cost->expansion_work;
    const auto expanded = ExpandStageRecipe(recipe, {}, cost->metadata_bytes, {}, source);
    ASSERT_TRUE(expanded) << expanded.error();
    EXPECT_EQ(expanded->tensors.size(), count);
  }
}

TEST(StageRecipe, WeightFreeScalarLookupsAndLazyBranchesAreFunded) {
  static constexpr auto zero = spec::ex::I(0);
  static constexpr auto one = spec::ex::I(1);
  static constexpr auto two = spec::ex::I(2);
  static constexpr auto yes = spec::ex::B(true);
  static constexpr auto tensor = spec::ex::TensorDim("header", 0);
  static constexpr auto value = spec::ex::V("rows");
  static constexpr std::string_view unchosen_name = "a.much.longer.absent.header";
  static constexpr auto missing = spec::ex::TensorDim(unchosen_name, 0);
  static constexpr spec::RecipeValue lazy{
      .kind = spec::ExprKind::kSelect, .left = &yes, .right = &value, .alternate = &missing};
  static constexpr auto templated = spec::ex::TensorDim("header.{i}", 0);
  static constexpr auto unbound = spec::ex::TensorDim("header.{unused}", 0);
  static constexpr auto malformed = spec::ex::TensorDim("header.{unused", 0);
  static constexpr spec::RecipeValue lazy_unbound{
      .kind = spec::ExprKind::kSelect, .left = &yes, .right = &two, .alternate = &unbound};
  static constexpr spec::RecipeValue lazy_malformed{
      .kind = spec::ExprKind::kSelect, .left = &yes, .right = &two, .alternate = &malformed};
  constexpr unsigned kCount = 32;
  const std::array config{MetaEntry{.key = "rows", .value = std::int64_t{2}}};
  const std::array expressions{tensor, value, lazy, templated, lazy_unbound, lazy_malformed};
  for (std::size_t test = 0; test < expressions.size(); ++test) {
    SCOPED_TRACE(test);
    const std::array params{spec::RecipeParam{.name = "start", .value = zero},
                            spec::RecipeParam{.name = "rows", .value = expressions[test]},
                            spec::RecipeParam{.name = "input_rows", .value = two}};
    const spec::RecipeNode node{
        .name = "step.{i}",
        .component_kind = "slice.spatial",
        .params = params,
        .input = {.kind = spec::RecipeRefKind::kNamedValue, .name = "in.{i}"}};
    const std::array body{spec::RecipeStep{.kind = spec::RecipeStep::Kind::kNode, .node = &node}};
    auto repeat = kRepeat;
    repeat.count = spec::ex::I(kCount);
    repeat.body = body;
    repeat.input = {.kind = spec::RecipeRefKind::kStageInput};
    const std::array steps{
        spec::RecipeStep{.kind = spec::RecipeStep::Kind::kRepeat, .repeat = &repeat}};
    const spec::RecipeSpatialShape shape{
        .width = one, .height = two, .channels = one, .images = one};
    const std::array outputs{shape};
    auto recipe = kRecipe;
    recipe.primary_input.shape = shape;
    recipe.expected_outputs = outputs;
    recipe.steps = steps;
    std::vector<Tensor> source{{.name = "header", .dims = {2}}};
    if (test == 3) {
      source.clear();
      for (unsigned i = 0; i < kCount; ++i)
        source.push_back({.name = "header." + std::to_string(i), .dims = {2}});
    }
    const auto cost = PreflightStageRecipe(recipe, {}, 1U << 20U, config, source);
    ASSERT_TRUE(cost) << cost.error();
    EXPECT_EQ(cost->tensor_bindings, 0U);
    // Value-only parameters were read by PayloadBound without incrementing
    // EvalOptions::lookups. Tensor-only parameters were never read in Count.
    EXPECT_EQ(cost->source_value_lookups, test == 1 || test == 2 ? 8U * kCount : 0U);
    if (test != 1) {
      const auto max_name = test == 2   ? unchosen_name.size()
                            : test == 3 ? 9U
                            : test == 4 ? 7U + 20U
                            : test == 5 ? malformed.name.size()
                                        : 6U;
      const auto comparisons = static_cast<std::uint64_t>(std::bit_width(source.size())) + 3;
      EXPECT_GE(cost->expansion_work, 8U * kCount * comparisons * (max_name + 1));
    }
    const auto expanded = ExpandStageRecipe(recipe, {}, cost->metadata_bytes, config, source);
    ASSERT_TRUE(expanded) << expanded.error();
    EXPECT_EQ(expanded->stage.layers[0].components.size(), kCount);
    // Pricing a lazy branch must not resolve it, while selecting that same
    // unbound/malformed template still reaches the evaluator's refusal.
    if (test >= 4) {
      auto active_params = params;
      active_params[1].value = test == 4 ? unbound : malformed;
      auto active_node = node;
      active_node.params = active_params;
      const std::array active_body{
          spec::RecipeStep{.kind = spec::RecipeStep::Kind::kNode, .node = &active_node}};
      repeat.body = active_body;
      EXPECT_FALSE(ExpandStageRecipe(recipe, {}, cost->metadata_bytes, config, source));
    }
  }
}

TEST(StageRecipe, SourceReadsInDimensionsGroupsAndControlAreFunded) {
  static constexpr auto one = spec::ex::I(1);
  static constexpr auto two = spec::ex::I(2);
  static constexpr auto dim = spec::ex::TensorDim("header", 0);
  static constexpr spec::RecipeValue present{.kind = spec::ExprKind::kTensorPresent,
                                             .name = "header"};
  static constexpr std::array params{spec::RecipeParam{.name = "in_channels", .value = two},
                                     spec::RecipeParam{.name = "out_channels", .value = two}};
  static constexpr std::array axes{dim, dim};
  static constexpr std::array weights{spec::RecipeWeight{
      .role = "weight", .tensor = "weight.{i}", .source_dims = axes, .source_dims_when = &present}};
  static constexpr spec::RecipeNode node{
      .name = "step.{i}",
      .component_kind = "projection.spatial",
      .params = params,
      .weights = weights,
      .input = {.kind = spec::RecipeRefKind::kNamedValue, .name = "in.{i}"},
      .group = spec::RecipeGroupKind::kLayer,
      .group_index = {.kind = spec::ExprKind::kSubtract, .left = &dim, .right = &two}};
  static constexpr std::array body{
      spec::RecipeStep{.kind = spec::RecipeStep::Kind::kNode, .node = &node}};
  auto repeat = kRepeat;
  repeat.count = dim;
  repeat.body = body;
  repeat.input = {.kind = spec::RecipeRefKind::kStageInput};
  const std::array steps{
      spec::RecipeStep{.kind = spec::RecipeStep::Kind::kRepeat, .repeat = &repeat}};
  const spec::RecipeSpatialShape shape{.width = one, .height = one, .channels = two, .images = one};
  const std::array outputs{shape};
  auto recipe = kRecipe;
  recipe.stream_width = two;
  recipe.primary_input.shape = shape;
  recipe.expected_outputs = outputs;
  recipe.steps = steps;
  const std::array source{Tensor{.name = "header", .dims = {2}}};
  const auto cost = PreflightStageRecipe(recipe, {}, 1U << 20U, {}, source);
  ASSERT_TRUE(cost) << cost.error();
  // One repeat-control read, then four leaves in each of two nodes:
  // two raw axes, their condition, and the group index.
  EXPECT_GE(cost->expansion_work, 8U * (1U + 2U * 4U) * 4U * 7U);
  const auto expanded = ExpandStageRecipe(recipe, {}, cost->metadata_bytes, {}, source);
  ASSERT_TRUE(expanded) << expanded.error();
  ASSERT_EQ(expanded->tensors.size(), 2U);
  EXPECT_EQ(expanded->tensors[0].source_dims, (std::vector<std::uint64_t>{2, 2}));
  EXPECT_EQ(expanded->tensors[0].layer, 0U);
}

TEST(StageRecipe, ExpandsTypedStandaloneSourceScalarAndListValues) {
  static constexpr spec::RecipeValue dim = spec::ex::V("in_channels");
  static constexpr spec::RecipeValue widths = spec::ex::V("widths");
  static constexpr spec::RecipeValue output_dim = spec::ex::At(widths, 1);
  static constexpr spec::RecipeValue width = spec::ex::At(widths, 0);
  static constexpr std::array projection_params{
      spec::RecipeParam{.name = "in_channels", .value = dim},
      spec::RecipeParam{.name = "out_channels", .value = output_dim}};
  static constexpr std::array projection_weights{
      spec::RecipeWeight{.role = "weight", .tensor = "proj.weight"}};
  static constexpr spec::RecipeNode projection{
      .name = "projection",
      .component_kind = "projection.spatial",
      .params = projection_params,
      .weights = projection_weights,
      .input = {.kind = spec::RecipeRefKind::kStageInput},
  };
  static constexpr spec::RecipeStep step{.kind = spec::RecipeStep::Kind::kNode,
                                         .node = &projection};
  static constexpr std::array steps{step};
  static constexpr spec::RecipeSpatialShape input_shape{.width = {.literal = std::int64_t{1}},
                                                        .height = {.literal = std::int64_t{1}},
                                                        .channels = dim,
                                                        .images = {.literal = std::int64_t{1}}};
  static constexpr spec::RecipeSpatialShape expected_shape{.width = {.literal = std::int64_t{1}},
                                                           .height = {.literal = std::int64_t{1}},
                                                           .channels = output_dim,
                                                           .images = {.literal = std::int64_t{1}}};
  static constexpr std::array expected{expected_shape};
  static constexpr spec::StageRecipe recipe{.name = "transformer",
                                            .role = spec::StageRole::kDenoiser,
                                            .stream_width = width,
                                            .primary_input = {.shape = input_shape},
                                            .steps = steps,
                                            .expected_outputs = expected};
  const std::array source_values{
      MetaEntry{.key = "in_channels", .value = std::int64_t{2}},
      MetaEntry{.key = "widths", .value = std::vector<std::int64_t>{4, 3}}};
  auto expanded =
      ExpandStageRecipe(recipe, {.pipeline = spec::PipelineKind::kImage}, 1U << 20U, source_values);
  ASSERT_TRUE(expanded) << expanded.error();
  EXPECT_EQ(expanded->stage.stream.width, 4U);
  ASSERT_EQ(expanded->stage.layers[0].components.size(), 1U);
  const auto& params = expanded->stage.layers[0].components[0].params;
  ASSERT_EQ(params.size(), 2U);
  EXPECT_EQ(params[0].value, (spec::ParamValue{std::int64_t{2}}));
  EXPECT_EQ(params[1].value, (spec::ParamValue{std::int64_t{3}}));
  ASSERT_EQ(expanded->tensors.size(), 1U);
  EXPECT_EQ(expanded->tensors[0].shape, (std::vector<std::uint64_t>{3, 2}));
}

TEST(StageRecipe, SharedRawDimensionsMergeAndConflictsAreOrderIndependent) {
  static constexpr spec::RecipeValue one{.literal = std::int64_t{1}};
  static constexpr spec::RecipeValue two{.literal = std::int64_t{2}};
  static constexpr spec::RecipeValue three{.literal = std::int64_t{3}};
  static constexpr std::array params{
      spec::RecipeParam{.name = "in_channels", .value = two},
      spec::RecipeParam{.name = "out_channels", .value = three},
      spec::RecipeParam{.name = "gelu", .value = {.literal = false}}};
  static constexpr std::array constrained_dims{two, three};
  static constexpr std::array conflicting_dims{three, two};
  static constexpr std::array constrained_weight{spec::RecipeWeight{
      .role = "weight", .tensor = "shared.weight", .source_dims = constrained_dims}};
  static constexpr std::array empty_weight{
      spec::RecipeWeight{.role = "weight", .tensor = "shared.weight"}};
  static constexpr std::array conflicting_weight{spec::RecipeWeight{
      .role = "weight", .tensor = "shared.weight", .source_dims = conflicting_dims}};
  static constexpr spec::RecipeNode first{.name = "first",
                                          .component_kind = "projection.spatial",
                                          .params = params,
                                          .weights = constrained_weight,
                                          .input = {.kind = spec::RecipeRefKind::kStageInput}};
  static constexpr spec::RecipeNode second{.name = "second",
                                           .component_kind = "projection.spatial",
                                           .params = params,
                                           .weights = empty_weight,
                                           .input = {.kind = spec::RecipeRefKind::kStageInput}};
  static constexpr spec::RecipeNode conflicting{
      .name = "second",
      .component_kind = "projection.spatial",
      .params = params,
      .weights = conflicting_weight,
      .input = {.kind = spec::RecipeRefKind::kStageInput}};
  static constexpr spec::RecipeStep first_step{.kind = spec::RecipeStep::Kind::kNode,
                                               .node = &first};
  static constexpr spec::RecipeStep second_step{.kind = spec::RecipeStep::Kind::kNode,
                                                .node = &second};
  static constexpr spec::RecipeStep conflict_step{.kind = spec::RecipeStep::Kind::kNode,
                                                  .node = &conflicting};
  static constexpr spec::RecipeSpatialShape input{
      .width = one, .height = one, .channels = two, .images = one};
  static constexpr spec::RecipeSpatialShape output{
      .width = one, .height = one, .channels = three, .images = one};
  static constexpr std::array outputs{output};
  static constexpr std::array first_export{
      spec::RecipeRef{.kind = spec::RecipeRefKind::kNamedValue, .name = "first"}};
  static constexpr std::array second_export{
      spec::RecipeRef{.kind = spec::RecipeRefKind::kNamedValue, .name = "second"}};
  const spec::ModelSpec family{.pipeline = spec::PipelineKind::kImage};
  static constexpr spec::StageRecipe base{.name = "shared_raw_dimensions",
                                          .role = spec::StageRole::kDenoiser,
                                          .stream_width = three,
                                          .primary_input = {.shape = input},
                                          .expected_outputs = outputs};

  static constexpr std::array constrained_then_empty{first_step, second_step};
  auto merged_recipe = base;
  merged_recipe.steps = constrained_then_empty;
  merged_recipe.exports = second_export;
  auto merged = ExpandStageRecipe(merged_recipe, family, 1U << 20U);
  ASSERT_TRUE(merged) << merged.error();
  ASSERT_EQ(merged->tensors.size(), 1U);
  EXPECT_EQ(merged->tensors[0].source_dims, (std::vector<std::uint64_t>{2, 3}));

  static constexpr std::array empty_then_constrained{second_step, first_step};
  merged_recipe.steps = empty_then_constrained;
  merged_recipe.exports = first_export;
  merged = ExpandStageRecipe(merged_recipe, family, 1U << 20U);
  ASSERT_TRUE(merged) << merged.error();
  ASSERT_EQ(merged->tensors.size(), 1U);
  EXPECT_EQ(merged->tensors[0].source_dims, (std::vector<std::uint64_t>{2, 3}));

  static constexpr std::array constrained_then_conflicting{first_step, conflict_step};
  merged_recipe.steps = constrained_then_conflicting;
  merged_recipe.exports = second_export;
  const auto first_order = ExpandStageRecipe(merged_recipe, family, 1U << 20U);
  ASSERT_FALSE(first_order);
  EXPECT_NE(first_order.error().find("conflicting raw source dimensions"), std::string::npos);

  static constexpr std::array conflicting_then_constrained{conflict_step, first_step};
  merged_recipe.steps = conflicting_then_constrained;
  const auto reverse_order = ExpandStageRecipe(merged_recipe, family, 1U << 20U);
  ASSERT_FALSE(reverse_order);
  EXPECT_NE(reverse_order.error().find("conflicting raw source dimensions"), std::string::npos);
}

TEST(StageRecipe, RawDimensionsUseRepeatIndexAndRejectInvalidAxes) {
  static constexpr spec::RecipeValue zero{.literal = std::int64_t{0}};
  static constexpr spec::RecipeValue one{.literal = std::int64_t{1}};
  static constexpr spec::RecipeValue count{.literal = std::int64_t{2}};
  static constexpr spec::RecipeValue index = spec::ex::Idx("i");
  static constexpr spec::RecipeValue first_axis{
      .kind = spec::RecipeValueKind::kAdd, .left = &index, .right = &one};
  static constexpr std::array indexed_dims{first_axis, one};
  static constexpr std::array zero_dims{zero, one};
  static constexpr std::array params{spec::RecipeParam{.name = "in_channels", .value = one},
                                     spec::RecipeParam{.name = "out_channels", .value = one}};
  static constexpr std::array indexed_weight{spec::RecipeWeight{
      .role = "weight", .tensor = "projection.{i}.weight", .source_dims = indexed_dims}};
  static constexpr std::array zero_weight{spec::RecipeWeight{
      .role = "weight", .tensor = "projection.zero.weight", .source_dims = zero_dims}};
  static constexpr spec::RecipeNode node{.name = "projection.{i}",
                                         .component_kind = "projection.spatial",
                                         .params = params,
                                         .weights = indexed_weight,
                                         .input = {.kind = spec::RecipeRefKind::kStageInput}};
  static constexpr spec::RecipeNode invalid_node{
      .name = "projection.zero",
      .component_kind = "projection.spatial",
      .params = params,
      .weights = zero_weight,
      .input = {.kind = spec::RecipeRefKind::kStageInput}};
  static constexpr spec::RecipeStep body_step{.kind = spec::RecipeStep::Kind::kNode, .node = &node};
  static constexpr std::array body{body_step};
  static constexpr spec::RecipeRepeat repeat{.index = "i",
                                             .count = count,
                                             .body = body,
                                             .input = {.kind = spec::RecipeRefKind::kStageInput},
                                             .input_name = "in.{i}",
                                             .body_yield = "projection.{i}",
                                             .yield = "last"};
  static constexpr spec::RecipeStep repeat_step{.kind = spec::RecipeStep::Kind::kRepeat,
                                                .repeat = &repeat};
  static constexpr spec::RecipeStep invalid_step{.kind = spec::RecipeStep::Kind::kNode,
                                                 .node = &invalid_node};
  static constexpr std::array steps{repeat_step};
  static constexpr std::array invalid_steps{invalid_step};
  static constexpr spec::RecipeSpatialShape unit{
      .width = one, .height = one, .channels = one, .images = one};
  static constexpr std::array outputs{unit};
  auto recipe = kRecipe;
  recipe.steps = steps;
  recipe.exports = {};
  recipe.expected_outputs = outputs;
  const auto expanded =
      ExpandStageRecipe(recipe, {.pipeline = spec::PipelineKind::kImage}, 1U << 20U);
  ASSERT_TRUE(expanded) << expanded.error();
  ASSERT_EQ(expanded->tensors.size(), 2U);
  EXPECT_EQ(expanded->tensors[0].source_dims, (std::vector<std::uint64_t>{1, 1}));
  EXPECT_EQ(expanded->tensors[1].source_dims, (std::vector<std::uint64_t>{2, 1}));

  recipe.steps = invalid_steps;
  const auto invalid =
      ExpandStageRecipe(recipe, {.pipeline = spec::PipelineKind::kImage}, 1U << 20U);
  ASSERT_FALSE(invalid);
  EXPECT_NE(invalid.error().find("positive integers"), std::string::npos);
}

TEST(StageRecipe, ExactIntegerDivisionChecksDomainAndRemainder) {
  static constexpr spec::RecipeValue channels{.literal = std::int64_t{384}};
  static constexpr spec::RecipeValue two{.literal = std::int64_t{2}};
  static constexpr spec::RecipeValue negative_channels{.literal = std::int64_t{-384}};
  static constexpr spec::RecipeValue negative_one{.literal = std::int64_t{-1}};
  static constexpr spec::RecipeValue zero_value{.literal = std::int64_t{0}};
  static constexpr spec::RecipeValue five{.literal = std::int64_t{5}};
  static constexpr spec::RecipeValue minimum{.literal = INT64_MIN};
  static constexpr spec::RecipeValue zero{.literal = std::int64_t{0}};
  static constexpr spec::RecipeValue truth{.literal = true};
  static constexpr spec::RecipeValue real{.literal = 4.0};
  static constexpr spec::RecipeValue condition{.literal = true};
  static constexpr spec::RecipeValue one{.literal = std::int64_t{1}};
  static constexpr spec::RecipeValue negative_exact{
      .kind = spec::RecipeValueKind::kDivideExact, .left = &negative_channels, .right = &two};
  static constexpr spec::RecipeValue positive_from_negative{
      .kind = spec::RecipeValueKind::kMultiply, .left = &negative_exact, .right = &negative_one};
  static constexpr spec::RecipeValue zero_exact{
      .kind = spec::RecipeValueKind::kDivideExact, .left = &zero_value, .right = &two};
  static constexpr spec::RecipeValue positive_from_zero{
      .kind = spec::RecipeValueKind::kAdd, .left = &zero_exact, .right = &one};
  static constexpr spec::RecipeValue exact{
      .kind = spec::RecipeValueKind::kDivideExact, .left = &channels, .right = &two};
  static constexpr spec::RecipeValue non_integral{
      .kind = spec::RecipeValueKind::kDivideExact, .left = &channels, .right = &five};
  static constexpr spec::RecipeValue zero_divisor{
      .kind = spec::RecipeValueKind::kDivideExact, .left = &channels, .right = &zero};
  static constexpr spec::RecipeValue signed_overflow{
      .kind = spec::RecipeValueKind::kDivideExact, .left = &minimum, .right = &negative_one};
  static constexpr spec::RecipeValue bool_operand{
      .kind = spec::RecipeValueKind::kDivideExact, .left = &truth, .right = &two};
  static constexpr spec::RecipeValue real_operand{
      .kind = spec::RecipeValueKind::kDivideExact, .left = &real, .right = &two};
  static constexpr spec::RecipeValue lazy_invalid{
      .kind = spec::RecipeValueKind::kDivideExact, .left = &channels, .right = &zero};
  static constexpr spec::RecipeValue lazy_select{.kind = spec::RecipeValueKind::kSelect,
                                                 .left = &condition,
                                                 .right = &one,
                                                 .alternate = &lazy_invalid};
  auto recipe = kRecipe;
  recipe.stream_width = exact;
  const auto accepted =
      PreflightStageRecipe(recipe, {.pipeline = spec::PipelineKind::kImage}, 1U << 20U);
  ASSERT_TRUE(accepted) << accepted.error();
  EXPECT_EQ(accepted->stream_width, 192U);
  recipe.stream_width = positive_from_negative;
  const auto accepted_negative =
      PreflightStageRecipe(recipe, {.pipeline = spec::PipelineKind::kImage}, 1U << 20U);
  ASSERT_TRUE(accepted_negative) << accepted_negative.error();
  EXPECT_EQ(accepted_negative->stream_width, 192U);
  recipe.stream_width = positive_from_zero;
  const auto accepted_zero =
      PreflightStageRecipe(recipe, {.pipeline = spec::PipelineKind::kImage}, 1U << 20U);
  ASSERT_TRUE(accepted_zero) << accepted_zero.error();
  EXPECT_EQ(accepted_zero->stream_width, 1U);
  recipe.stream_width = lazy_select;
  const auto accepted_lazy =
      PreflightStageRecipe(recipe, {.pipeline = spec::PipelineKind::kImage}, 1U << 20U);
  ASSERT_TRUE(accepted_lazy) << accepted_lazy.error();
  EXPECT_EQ(accepted_lazy->stream_width, 1U);
  recipe.stream_width = non_integral;
  const auto rejected_remainder =
      PreflightStageRecipe(recipe, {.pipeline = spec::PipelineKind::kImage}, 1U << 20U);
  ASSERT_FALSE(rejected_remainder);
  EXPECT_NE(rejected_remainder.error().find("non-integral"), std::string::npos);
  recipe.stream_width = zero_divisor;
  const auto rejected_zero =
      PreflightStageRecipe(recipe, {.pipeline = spec::PipelineKind::kImage}, 1U << 20U);
  ASSERT_FALSE(rejected_zero);
  EXPECT_NE(rejected_zero.error().find("division is invalid"), std::string::npos);
  recipe.stream_width = signed_overflow;
  const auto rejected_overflow =
      PreflightStageRecipe(recipe, {.pipeline = spec::PipelineKind::kImage}, 1U << 20U);
  ASSERT_FALSE(rejected_overflow);
  EXPECT_NE(rejected_overflow.error().find("division is invalid"), std::string::npos);
  recipe.stream_width = bool_operand;
  const auto rejected_bool =
      PreflightStageRecipe(recipe, {.pipeline = spec::PipelineKind::kImage}, 1U << 20U);
  ASSERT_FALSE(rejected_bool);
  EXPECT_NE(rejected_bool.error().find("operands must be integers"), std::string::npos);
  recipe.stream_width = real_operand;
  const auto rejected_real =
      PreflightStageRecipe(recipe, {.pipeline = spec::PipelineKind::kImage}, 1U << 20U);
  ASSERT_FALSE(rejected_real);
  EXPECT_NE(rejected_real.error().find("operands must be integers"), std::string::npos);
}

TEST(StageRecipe, FloorIntegerDivisionUsesMathematicalFloorAndChecksDomain) {
  static constexpr spec::RecipeValue pos_odd{.literal = std::int64_t{285}};
  static constexpr spec::RecipeValue neg_odd{.literal = std::int64_t{-285}};
  static constexpr spec::RecipeValue neg_even{.literal = std::int64_t{-286}};
  static constexpr spec::RecipeValue pos_even{.literal = std::int64_t{286}};
  static constexpr spec::RecipeValue two{.literal = std::int64_t{2}};
  static constexpr spec::RecipeValue neg_two{.literal = std::int64_t{-2}};
  static constexpr spec::RecipeValue neg_one{.literal = std::int64_t{-1}};
  static constexpr spec::RecipeValue zero{.literal = std::int64_t{0}};
  static constexpr spec::RecipeValue minimum{.literal = INT64_MIN};
  static constexpr spec::RecipeValue truth{.literal = true};
  static constexpr spec::RecipeValue real{.literal = 4.0};
  static constexpr spec::RecipeValue condition{.literal = true};
  static constexpr spec::RecipeValue pos_remainder{
      .kind = spec::RecipeValueKind::kDivideFloor, .left = &pos_odd, .right = &two};
  static constexpr spec::RecipeValue neg_numerator_remainder{
      .kind = spec::RecipeValueKind::kDivideFloor, .left = &neg_odd, .right = &two};
  static constexpr spec::RecipeValue neg_denominator_remainder{
      .kind = spec::RecipeValueKind::kDivideFloor, .left = &pos_odd, .right = &neg_two};
  static constexpr spec::RecipeValue neg_exact{
      .kind = spec::RecipeValueKind::kDivideFloor, .left = &neg_even, .right = &two};
  static constexpr spec::RecipeValue negative_one{.literal = std::int64_t{-1}};
  static constexpr spec::RecipeValue neg_numerator_positive_result{
      .kind = spec::RecipeValueKind::kMultiply,
      .left = &neg_numerator_remainder,
      .right = &negative_one};
  static constexpr spec::RecipeValue neg_denominator_positive_result{
      .kind = spec::RecipeValueKind::kMultiply,
      .left = &neg_denominator_remainder,
      .right = &negative_one};
  static constexpr spec::RecipeValue neg_exact_positive_result{
      .kind = spec::RecipeValueKind::kMultiply, .left = &neg_exact, .right = &negative_one};
  static constexpr spec::RecipeValue positive_exact{
      .kind = spec::RecipeValueKind::kDivideFloor, .left = &pos_even, .right = &two};
  static constexpr spec::RecipeValue zero_divisor{
      .kind = spec::RecipeValueKind::kDivideFloor, .left = &pos_odd, .right = &zero};
  static constexpr spec::RecipeValue signed_overflow{
      .kind = spec::RecipeValueKind::kDivideFloor, .left = &minimum, .right = &neg_one};
  static constexpr spec::RecipeValue bool_operand{
      .kind = spec::RecipeValueKind::kDivideFloor, .left = &truth, .right = &two};
  static constexpr spec::RecipeValue real_operand{
      .kind = spec::RecipeValueKind::kDivideFloor, .left = &real, .right = &two};
  static constexpr spec::RecipeValue lazy_select{.kind = spec::RecipeValueKind::kSelect,
                                                 .left = &condition,
                                                 .right = &pos_remainder,
                                                 .alternate = &zero_divisor};
  auto recipe = kRecipe;
  const auto accepts = [&](const spec::RecipeValue& value, std::int64_t expected) {
    recipe.stream_width = value;
    const auto result =
        PreflightStageRecipe(recipe, {.pipeline = spec::PipelineKind::kImage}, 1U << 20U);
    ASSERT_TRUE(result) << result.error();
    EXPECT_EQ(result->stream_width, static_cast<std::uint64_t>(expected));
  };
  accepts(pos_remainder, 142);
  accepts(positive_exact, 143);
  accepts(neg_numerator_positive_result, 143);    // floor(-285/2) == -143
  accepts(neg_denominator_positive_result, 143);  // floor(285/-2) == -143
  accepts(neg_exact_positive_result, 143);        // floor(-286/2) == -143
  accepts(lazy_select, 142);

  const auto rejects = [&](const spec::RecipeValue& value, std::string_view reason) {
    recipe.stream_width = value;
    const auto result =
        PreflightStageRecipe(recipe, {.pipeline = spec::PipelineKind::kImage}, 1U << 20U);
    ASSERT_FALSE(result);
    EXPECT_NE(result.error().find(reason), std::string::npos);
  };
  rejects(zero_divisor, "floor integer recipe division is invalid");
  rejects(signed_overflow, "floor integer recipe division is invalid");
  rejects(bool_operand, "operands must be integers");
  rejects(real_operand, "operands must be integers");
}

TEST(StageRecipe, StandaloneMappingUsesTheNormalizationPlan) {
  static constexpr spec::RecipeValue width{.literal = std::int64_t{2}};
  static constexpr spec::RecipeValue one{.literal = std::int64_t{1}};
  static constexpr spec::RecipeValue channels{.literal = std::int64_t{2}};
  static constexpr std::array params{
      spec::RecipeParam{.name = "in_channels", .value = channels},
      spec::RecipeParam{.name = "out_channels", .value = channels},
      spec::RecipeParam{.name = "gelu", .value = {.literal = false}}};
  static constexpr std::array weights{
      spec::RecipeWeight{.role = "weight", .tensor = "proj.weight"}};
  static constexpr spec::RecipeNode projection{.name = "projection",
                                               .component_kind = "projection.spatial",
                                               .params = params,
                                               .weights = weights,
                                               .input = {.kind = spec::RecipeRefKind::kStageInput}};
  static constexpr spec::RecipeStep step{.kind = spec::RecipeStep::Kind::kNode,
                                         .node = &projection};
  static constexpr std::array steps{step};
  static constexpr spec::RecipeSpatialShape input{
      .width = one, .height = one, .channels = channels, .images = one};
  static constexpr std::array outputs{input};
  static constexpr spec::StageRecipe recipe{.name = "transformer",
                                            .role = spec::StageRole::kDenoiser,
                                            .stream_width = width,
                                            .primary_input = {.shape = input},
                                            .steps = steps,
                                            .expected_outputs = outputs};
  static constexpr std::array containers{std::string_view("safetensors")};
  const spec::Mapping mapping{.architecture = "tiny_spatial",
                              .pipeline = spec::PipelineKind::kImage,
                              .role = spec::StageRole::kDenoiser,
                              .stage = "transformer",
                              .stage_recipe = &recipe,
                              .containers = containers};

  static constexpr std::array<std::uint64_t, 2> source_dims{2, 2};
  const auto shape = format::ShapeOf(source_dims);
  ASSERT_TRUE(shape);
  const format::TensorFormat tensor_format{
      .encoding = format::Encoding::kF32, .layout = format::Layout::kDense, .shape = *shape};
  const auto parts = format::PartsOf(tensor_format);
  ASSERT_TRUE(parts);
  Checkpoint source{.container = "safetensors",
                    .architecture = "tiny_spatial",
                    .files = {{.name = "model.safetensors",
                               .path = "/tmp/model.safetensors",
                               .bytes = (*parts)[0].bytes,
                               .device = 1,
                               .inode = 1}},
                    .tensors = {{.name = "proj.weight",
                                 .format = tensor_format,
                                 .extent = {.file = 0, .offset = 0, .bytes = (*parts)[0].bytes},
                                 .dtype = "F32",
                                 .dims = {2, 2}}}};
  auto cost = normalization_detail::Bound(source, mapping, {.component = "transformer"});
  ASSERT_TRUE(cost) << cost.error().reason;
  EXPECT_FALSE(normalization_detail::Build(Checkpoint(source), mapping,
                                           {cost->metadata_bytes - 1, cost->work_units}, nullptr,
                                           {.component = "transformer"}));
  EXPECT_FALSE(normalization_detail::Build(Checkpoint(source), mapping,
                                           {cost->metadata_bytes, cost->work_units - 1}, nullptr,
                                           {.component = "transformer"}));
  auto plan = normalization_detail::Build(std::move(source), mapping,
                                          {cost->metadata_bytes, cost->work_units}, nullptr,
                                          {.component = "transformer"});
  ASSERT_TRUE(plan) << plan.error().reason;
  EXPECT_EQ(plan->mapping_architecture(), "tiny_spatial");
  ASSERT_EQ(plan->spec().stages.size(), 1U);
  ASSERT_EQ(plan->spec().stages[0].layers[0].components.size(), 1U);
  EXPECT_EQ(plan->canonical().tensors.size(), 1U);
  EXPECT_EQ(plan->routes().size(), 1U);
  EXPECT_EQ(plan->groups().size(), 1U);
}

// A causal 3-D kernel's first frame (conv.spatial's temporal_kernel): the raw
// rank-five header must factor into the folded role shape as the recipe says.
TEST(StageRecipe, TemporalKernelChecksRawRankFiveChannelFactorization) {
  static constexpr spec::RecipeValue one{.literal = std::int64_t{1}};
  static constexpr spec::RecipeValue in_channels{.literal = std::int64_t{16}};
  static constexpr spec::RecipeValue out_channels{.literal = std::int64_t{4}};
  static constexpr spec::RecipeValue kernel{.literal = std::int64_t{3}};
  static constexpr std::array kernel_dims{out_channels, in_channels, kernel, kernel, kernel};
  static constexpr std::array bias_dims{out_channels};
  static constexpr std::array params{
      spec::RecipeParam{.name = "in_channels", .value = in_channels},
      spec::RecipeParam{.name = "out_channels", .value = out_channels},
      spec::RecipeParam{.name = "kernel", .value = kernel},
      spec::RecipeParam{.name = "temporal_kernel", .value = kernel},
      spec::RecipeParam{.name = "padding", .value = one}};
  static constexpr std::array weights{
      spec::RecipeWeight{
          .role = "kernel", .tensor = "decoder.conv.weight", .source_dims = kernel_dims},
      spec::RecipeWeight{.role = "bias", .tensor = "decoder.conv.bias", .source_dims = bias_dims}};
  static constexpr spec::RecipeNode conv{.name = "conv",
                                         .component_kind = "conv.spatial",
                                         .params = params,
                                         .weights = weights,
                                         .input = {.kind = spec::RecipeRefKind::kStageInput}};
  static constexpr spec::RecipeStep step{.kind = spec::RecipeStep::Kind::kNode, .node = &conv};
  static constexpr std::array steps{step};
  static constexpr spec::RecipeSpatialShape input{
      .width = one, .height = one, .channels = in_channels, .images = one};
  static constexpr spec::RecipeSpatialShape output{
      .width = one, .height = one, .channels = out_channels, .images = one};
  static constexpr std::array outputs{output};
  static constexpr spec::StageRecipe recipe{.name = "vae",
                                            .role = spec::StageRole::kVae,
                                            .stream_width = in_channels,
                                            .primary_input = {.shape = input},
                                            .steps = steps,
                                            .expected_outputs = outputs};
  static constexpr std::array containers{std::string_view("safetensors")};
  static constexpr spec::Mapping mapping{.architecture = "tiny_causal_vae",
                                         .pipeline = spec::PipelineKind::kImage,
                                         .role = spec::StageRole::kVae,
                                         .stage = "vae",
                                         .stage_recipe = &recipe,
                                         .containers = containers};

  static constexpr std::array<std::uint64_t, 4> folded_dims{3, 3, 3, 64};
  const auto kernel_shape = format::ShapeOf(folded_dims);
  const auto bias_shape = format::ShapeOf(std::array<std::uint64_t, 1>{4});
  ASSERT_TRUE(kernel_shape);
  ASSERT_TRUE(bias_shape);
  const format::TensorFormat kernel_format{.encoding = format::Encoding::kBF16,
                                           .layout = format::Layout::kDense,
                                           .shape = *kernel_shape};
  const format::TensorFormat bias_format{
      .encoding = format::Encoding::kBF16, .layout = format::Layout::kDense, .shape = *bias_shape};
  const auto kernel_parts = format::PartsOf(kernel_format);
  const auto bias_parts = format::PartsOf(bias_format);
  ASSERT_TRUE(kernel_parts);
  ASSERT_TRUE(bias_parts);
  const std::uint64_t bias_offset = (*kernel_parts)[0].bytes;
  Checkpoint source{
      .container = "safetensors",
      .architecture = "tiny_causal_vae",
      .files = {{.name = "model.safetensors",
                 .path = "/tmp/model.safetensors",
                 .bytes = bias_offset + (*bias_parts)[0].bytes,
                 .device = 1,
                 .inode = 1}},
      .tensors = {{.name = "decoder.conv.weight",
                   .format = kernel_format,
                   .extent = {.file = 0, .offset = 0, .bytes = (*kernel_parts)[0].bytes},
                   .dtype = "BF16",
                   .dims = {4, 16, 3, 3, 3}},
                  {.name = "decoder.conv.bias",
                   .format = bias_format,
                   .extent = {.file = 0, .offset = bias_offset, .bytes = (*bias_parts)[0].bytes},
                   .dtype = "BF16",
                   .dims = {4}}}};
  const auto preflight = PreflightStageRecipe(recipe, {}, std::uint64_t{1} << 20U);
  ASSERT_TRUE(preflight) << preflight.error();
  EXPECT_EQ(preflight->raw_dimension_axes, 6U);
  const auto preflight_expand = ExpandStageRecipe(recipe, {}, preflight->metadata_bytes);
  ASSERT_TRUE(preflight_expand) << preflight_expand.error();
  ASSERT_GT(preflight->metadata_bytes, 0U);
  EXPECT_FALSE(ExpandStageRecipe(recipe, {}, preflight->metadata_bytes - 1));
  auto plan = PlanSpatialComponent(source, mapping);
  ASSERT_TRUE(plan) << plan.error().reason;
  EXPECT_EQ(plan->payload_bytes, (*kernel_parts)[0].bytes + (*bias_parts)[0].bytes);
  EXPECT_EQ(plan->canonical.tensors[0].format.encoding, format::Encoding::kBF16);
  const auto kernel_route = std::ranges::find(plan->routes, 0U, &CopyRoute::source_tensor);
  ASSERT_NE(kernel_route, plan->routes.end());
  EXPECT_EQ(kernel_route->extent.bytes, (*kernel_parts)[0].bytes);

  source.tensors[0].dims = {2, 32, 3, 3, 3};
  const auto wrong_factorization = PlanSpatialComponent(source, mapping);
  ASSERT_FALSE(wrong_factorization);
  EXPECT_NE(wrong_factorization.error().reason.find("raw header dimensions"), std::string::npos);
}

TEST(StageRecipe, ManyGeneratedTensorNamesAreChargedBeforeBuild) {
  static constexpr spec::RecipeValue one{.literal = std::int64_t{1}};
  static constexpr spec::RecipeValue count{.literal = std::int64_t{32}};
  static constexpr std::array params{spec::RecipeParam{.name = "in_channels", .value = one},
                                     spec::RecipeParam{.name = "out_channels", .value = one}};
  static constexpr std::array weights{
      spec::RecipeWeight{.role = "weight", .tensor = "blk.{i}.weight"}};
  static constexpr spec::RecipeNode node{
      .name = "projection.{i}",
      .component_kind = "projection.spatial",
      .params = params,
      .weights = weights,
      .input = {.kind = spec::RecipeRefKind::kNamedValue, .name = "in.{i}"}};
  static constexpr spec::RecipeStep body_step{.kind = spec::RecipeStep::Kind::kNode, .node = &node};
  static constexpr std::array body{body_step};
  static constexpr spec::RecipeRepeat repeat{.index = "i",
                                             .count = count,
                                             .body = body,
                                             .input = {.kind = spec::RecipeRefKind::kStageInput},
                                             .input_name = "in.{i}",
                                             .body_yield = "projection.{i}",
                                             .yield = "final"};
  static constexpr spec::RecipeStep step{.kind = spec::RecipeStep::Kind::kRepeat,
                                         .repeat = &repeat};
  static constexpr std::array steps{step};
  static constexpr spec::RecipeSpatialShape shape{
      .width = one, .height = one, .channels = one, .images = one};
  static constexpr std::array outputs{shape};
  static constexpr spec::StageRecipe recipe{.name = "transformer",
                                            .role = spec::StageRole::kDenoiser,
                                            .stream_width = one,
                                            .primary_input = {.shape = shape},
                                            .steps = steps,
                                            .expected_outputs = outputs};
  static constexpr std::array containers{std::string_view("safetensors")};
  static constexpr spec::Mapping mapping{.architecture = "tiny_spatial",
                                         .pipeline = spec::PipelineKind::kImage,
                                         .role = spec::StageRole::kDenoiser,
                                         .stage = "transformer",
                                         .stage_recipe = &recipe,
                                         .containers = containers};
  static constexpr std::array<std::uint64_t, 2> dims{1, 1};
  const auto tensor_shape = format::ShapeOf(dims);
  ASSERT_TRUE(tensor_shape);
  const format::TensorFormat tensor_format{
      .encoding = format::Encoding::kF32, .layout = format::Layout::kDense, .shape = *tensor_shape};
  Checkpoint incomplete{.container = "safetensors",
                        .architecture = "tiny_spatial",
                        .files = {{.name = "model.safetensors",
                                   .path = "/tmp/model.safetensors",
                                   .bytes = 4,
                                   .device = 1,
                                   .inode = 1}},
                        .tensors = {{.name = "unrelated.weight",
                                     .format = tensor_format,
                                     .extent = {.file = 0, .offset = 0, .bytes = 4},
                                     .dtype = "F32",
                                     .dims = {1, 1}}}};
  const auto cost = normalization_detail::Bound(incomplete, mapping, {.component = "transformer"});
  ASSERT_TRUE(cost) << cost.error().reason;
  EXPECT_FALSE(normalization_detail::Build(Checkpoint(incomplete), mapping,
                                           {cost->metadata_bytes, cost->work_units - 1}, nullptr,
                                           {.component = "transformer"}));
  EXPECT_FALSE(normalization_detail::Build(
      std::move(incomplete), mapping, {cost->metadata_bytes, cost->work_units}, nullptr,
      {.component =
           "transformer"}));  // accepted reservation reaches the intended missing-source refusal
  EXPECT_GE(cost->work_units, 32U * 32U);
}

TEST(StageRecipe, LongOmittedTensorNamesAreChargedByComparedBytes) {
  static constexpr spec::RecipeValue one{.literal = std::int64_t{1}};
  static constexpr std::array params{spec::RecipeParam{.name = "in_channels", .value = one},
                                     spec::RecipeParam{.name = "out_channels", .value = one}};
  static constexpr std::array weights{
      spec::RecipeWeight{.role = "weight", .tensor = "proj.weight"}};
  static constexpr spec::RecipeNode node{.name = "projection",
                                         .component_kind = "projection.spatial",
                                         .params = params,
                                         .weights = weights,
                                         .input = {.kind = spec::RecipeRefKind::kStageInput}};
  static constexpr spec::RecipeStep step{.kind = spec::RecipeStep::Kind::kNode, .node = &node};
  static constexpr std::array steps{step};
  static constexpr spec::RecipeSpatialShape shape{
      .width = one, .height = one, .channels = one, .images = one};
  static constexpr std::array outputs{shape};
  static constexpr spec::StageRecipe recipe{.name = "transformer",
                                            .role = spec::StageRole::kDenoiser,
                                            .stream_width = one,
                                            .primary_input = {.shape = shape},
                                            .steps = steps,
                                            .expected_outputs = outputs};
  const std::string prefix(200, 'x');
  const std::array<std::string_view, 1> ignored_prefixes{prefix};
  static constexpr std::array containers{std::string_view("safetensors")};
  const spec::Mapping mapping{.architecture = "tiny_spatial",
                              .pipeline = spec::PipelineKind::kImage,
                              .role = spec::StageRole::kDenoiser,
                              .stage = "transformer",
                              .stage_recipe = &recipe,
                              .containers = containers,
                              .ignored_tensor_prefixes = ignored_prefixes};
  static constexpr std::array<std::uint64_t, 2> dims{1, 1};
  const auto tensor_shape = format::ShapeOf(dims);
  ASSERT_TRUE(tensor_shape);
  const format::TensorFormat tensor_format{
      .encoding = format::Encoding::kF32, .layout = format::Layout::kDense, .shape = *tensor_shape};
  static constexpr std::size_t omitted_count = 1024;
  Checkpoint source{.container = "safetensors",
                    .architecture = "tiny_spatial",
                    .files = {{.name = "model.safetensors",
                               .path = "/tmp/model.safetensors",
                               .bytes = (omitted_count + 1) * 4,
                               .device = 1,
                               .inode = 1}}};
  source.tensors.reserve(omitted_count + 1);
  source.tensors.push_back({.name = "proj.weight",
                            .format = tensor_format,
                            .extent = {.file = 0, .offset = 0, .bytes = 4},
                            .dtype = "F32",
                            .dims = {1, 1}});
  std::uint64_t compared_name_bytes = source.tensors.front().name.size();
  std::uint64_t maximum_name_bytes = source.tensors.front().name.size();
  for (std::size_t i = 0; i < omitted_count; ++i) {
    auto name = prefix + std::to_string(i);
    compared_name_bytes += name.size();
    maximum_name_bytes = std::max(maximum_name_bytes, static_cast<std::uint64_t>(name.size()));
    source.tensors.push_back({.name = std::move(name),
                              .format = tensor_format,
                              .extent = {.file = 0, .offset = (i + 1) * 4, .bytes = 4},
                              .dtype = "F32",
                              .dims = {1, 1}});
  }
  const auto cost = normalization_detail::Bound(source, mapping, {.component = "transformer"});
  ASSERT_TRUE(cost) << cost.error().reason;
  EXPECT_GE(cost->work_units, compared_name_bytes * ignored_prefixes.size());
  const auto name_tree_floor =
      std::uint64_t{4} *
      (static_cast<std::uint64_t>(
           std::bit_width(static_cast<std::uint64_t>(source.tensors.size()))) +
       1) *
      static_cast<std::uint64_t>(source.tensors.size()) * maximum_name_bytes;
  EXPECT_GE(cost->work_units, name_tree_floor);
  EXPECT_FALSE(normalization_detail::Build(Checkpoint(source), mapping,
                                           {cost->metadata_bytes, cost->work_units - 1}, nullptr,
                                           {.component = "transformer"}));
  auto plan = normalization_detail::Build(std::move(source), mapping,
                                          {cost->metadata_bytes, cost->work_units}, nullptr,
                                          {.component = "transformer"});
  ASSERT_TRUE(plan) << plan.error().reason;
}

TEST(StageRecipe, RejectsExpansionThatExceedsPaidMetadata) {
  auto expanded = ExpandStageRecipe(kRecipe, {.pipeline = spec::PipelineKind::kImage}, 1U);
  EXPECT_FALSE(expanded);
}

TEST(StageRecipe, PreflightsExpressionCyclesAndLexicalTemplates) {
  const spec::ModelSpec family{.pipeline = spec::PipelineKind::kImage};
  spec::RecipeValue one{.kind = spec::RecipeValueKind::kLiteral, .literal = std::int64_t{1}};
  spec::RecipeValue cycle{.kind = spec::RecipeValueKind::kAdd, .right = &one};
  cycle.left = &cycle;
  auto cyclic = kRecipe;
  cyclic.stream_width = cycle;
  EXPECT_FALSE(ExpandStageRecipe(cyclic, family, 1U << 20U));

  const spec::RecipeNode bad_node{
      .name = "step.{unbound}",
      .component_kind = "residual.spatial",
      .input = {.kind = spec::RecipeRefKind::kStageInput},
  };
  const spec::RecipeStep bad_step{.kind = spec::RecipeStep::Kind::kNode, .node = &bad_node};
  const std::array bad_steps{bad_step};
  auto bad_template = kRecipe;
  bad_template.steps = bad_steps;
  EXPECT_FALSE(ExpandStageRecipe(bad_template, family, 1U << 20U));
}

TEST(StageRecipe, PreflightsControlAliasesBeforeAllocatingThem) {
  const std::string huge_name(1U << 20U, 'x');
  const std::string huge_body_yield(1U << 20U, 'z');
  const spec::RecipeNode node{.name = "seed",
                              .component_kind = "residual.spatial",
                              .input = {.kind = spec::RecipeRefKind::kStageInput}};
  const spec::RecipeStep node_step{.kind = spec::RecipeStep::Kind::kNode, .node = &node};
  const std::array<spec::RecipeStep, 0> empty{};
  const spec::RecipeIf branch{
      .condition = {.kind = spec::RecipeValueKind::kLiteral, .literal = true},
      .then_steps = empty,
      .else_steps = empty,
      .input = {.kind = spec::RecipeRefKind::kNamedValue, .name = "seed"},
      .then_yield = huge_body_yield,
      .else_yield = "unselected",
      .yield = "out",
      .input_name = huge_name,
  };
  const spec::RecipeStep if_step{.kind = spec::RecipeStep::Kind::kIf, .branch = &branch};
  const std::array steps{node_step, if_step};
  const std::array exports{
      spec::RecipeRef{.kind = spec::RecipeRefKind::kNamedValue, .name = "seed"}};
  const std::array expected{kUnitShape};
  const spec::StageRecipe recipe{
      .name = "alias-budget",
      .role = spec::StageRole::kVae,
      .stream_width = {.kind = spec::RecipeValueKind::kLiteral, .literal = std::int64_t{1}},
      .primary_input = {.shape = kUnitShape},
      .steps = steps,
      .exports = exports,
      .expected_outputs = expected,
  };

  auto expanded = ExpandStageRecipe(recipe, {.pipeline = spec::PipelineKind::kImage}, 3'000'000);
  ASSERT_FALSE(expanded);
  EXPECT_NE(expanded.error().find("prepaid metadata bound"), std::string::npos);
}

TEST(StageRecipe, PreflightsLargeRepeatIndexBeforeExpansion) {
  const std::string huge_index(1U << 20U, 'i');
  const spec::RecipeNode node{.name = "inside", .component_kind = "residual.spatial"};
  const spec::RecipeStep node_step{.kind = spec::RecipeStep::Kind::kNode, .node = &node};
  const std::array body{node_step};
  const spec::RecipeRepeat repeat{
      .index = huge_index,
      .count = {.kind = spec::RecipeValueKind::kLiteral, .literal = std::int64_t{1}},
      .body = body,
      .input = {.kind = spec::RecipeRefKind::kStageInput},
      .input_name = "in",
      .body_yield = "inside",
      .yield = "inside",
  };
  const spec::RecipeStep repeat_step{.kind = spec::RecipeStep::Kind::kRepeat, .repeat = &repeat};
  const std::array steps{repeat_step};
  const std::array expected{kUnitShape};
  auto recipe = kRecipe;
  recipe.steps = steps;
  recipe.expected_outputs = expected;

  auto expanded = ExpandStageRecipe(recipe, {.pipeline = spec::PipelineKind::kImage}, 8'192);
  ASSERT_FALSE(expanded);
  EXPECT_NE(expanded.error().find("prepaid metadata bound"), std::string::npos);
}

TEST(StageRecipe, RejectsPayloadBearingExpectedShapeWithoutCopyingIt) {
  const std::string large_payload(1U << 20U, 'p');
  const spec::RecipeSpatialShape invalid_shape{
      .width = {.kind = spec::RecipeValueKind::kLiteral, .literal = large_payload},
      .height = {.literal = std::int64_t{1}},
      .channels = {.literal = std::int64_t{1}},
      .images = {.literal = std::int64_t{1}},
  };
  const std::array expected{invalid_shape};
  auto recipe = kRecipe;
  recipe.expected_outputs = expected;

  auto expanded = ExpandStageRecipe(recipe, {.pipeline = spec::PipelineKind::kImage}, 8'192);
  ASSERT_FALSE(expanded);
  EXPECT_NE(expanded.error().find("dimensions must be scalar"), std::string::npos);
}

TEST(StageRecipe, RejectsEmptyGraphBeforeImplicitOutputGeometryLookup) {
  const std::array<spec::RecipeStep, 0> empty{};
  auto recipe = kRecipe;
  recipe.steps = empty;
  recipe.exports = {};
  const auto expanded =
      ExpandStageRecipe(recipe, {.pipeline = spec::PipelineKind::kImage}, 1U << 20U);
  ASSERT_FALSE(expanded);
  EXPECT_NE(expanded.error().find("produced no components"), std::string::npos);
}

TEST(StageRecipe, ValidatesTypedSpatialEdgesAndAllowsProjectedContextWidthDifference) {
  static constexpr spec::RecipeValue one{.literal = std::int64_t{1}};
  static constexpr spec::RecipeValue context{.literal = std::int64_t{32}};
  static constexpr spec::RecipeValue width{.literal = std::int64_t{64}};
  static constexpr spec::RecipeValue wrong_context{.literal = std::int64_t{31}};
  static constexpr spec::RecipeValue output_channels{.literal = std::int64_t{64}};
  static constexpr std::array projection_params{
      spec::RecipeParam{.name = "in_channels", .value = context},
      spec::RecipeParam{.name = "out_channels", .value = width},
      spec::RecipeParam{.name = "gelu", .value = {.literal = false}}};
  static constexpr std::array projection_weights{
      spec::RecipeWeight{.role = "weight", .tensor = "projection.weight"}};
  static constexpr spec::RecipeNode projection{.name = "noise",
                                               .component_kind = "projection.spatial",
                                               .params = projection_params,
                                               .weights = projection_weights,
                                               .input = {.kind = spec::RecipeRefKind::kStageInput}};
  static constexpr spec::RecipeStep projection_step{.kind = spec::RecipeStep::Kind::kNode,
                                                    .node = &projection};
  static constexpr std::array projection_steps{projection_step};
  static constexpr std::array noise_export{
      spec::RecipeRef{.kind = spec::RecipeRefKind::kNamedValue, .name = "noise"}};
  static constexpr std::array projected_shape{spec::RecipeSpatialShape{
      .width = one, .height = one, .channels = output_channels, .images = one}};
  spec::StageRecipe projected{
      .name = "projected",
      .role = spec::StageRole::kDenoiser,
      .stream_width = width,
      .primary_input = {.shape = {.width = one, .height = one, .channels = context, .images = one}},
      .steps = projection_steps,
      .exports = noise_export,
      .expected_outputs = projected_shape};
  const spec::ModelSpec family{.pipeline = spec::PipelineKind::kImage};
  auto mapped = ExpandStageRecipe(projected, family, 1U << 20U);
  ASSERT_TRUE(mapped) << mapped.error();
  EXPECT_EQ(mapped->stage.context, 0U);
  EXPECT_EQ(mapped->stage.stream.width, 64U);

  projected.primary_input.shape.channels = wrong_context;
  EXPECT_FALSE(ExpandStageRecipe(projected, family, 1U << 20U));
  projected.primary_input.shape.channels = context;

  static constexpr std::array scale_port{spec::RecipeSpatialOperand{
      .shape = {.width = one, .height = one, .channels = one, .images = one}}};
  static constexpr spec::RecipeNode euler{
      .name = "euler",
      .component_kind = "residual.scaled",
      .input = {.kind = spec::RecipeRefKind::kNamedValue, .name = "noise"},
      .auxiliary = {.kind = spec::RecipeRefKind::kHostPort, .port = 0},
      .carry = {.kind = spec::RecipeRefKind::kStageInput},
      .has_auxiliary = true,
      .has_carry = true};
  static constexpr spec::RecipeStep euler_step{.kind = spec::RecipeStep::Kind::kNode,
                                               .node = &euler};
  static constexpr std::array euler_steps{projection_step, euler_step};
  static constexpr std::array euler_export{
      spec::RecipeRef{.kind = spec::RecipeRefKind::kNamedValue, .name = "euler"}};
  auto invalid_euler = projected;
  invalid_euler.steps = euler_steps;
  invalid_euler.exports = euler_export;
  invalid_euler.host_ports = scale_port;
  invalid_euler.expected_outputs = projected_shape;
  auto mismatch = ExpandStageRecipe(invalid_euler, family, 1U << 20U);
  ASSERT_FALSE(mismatch);
  EXPECT_NE(mismatch.error().find("carry"), std::string::npos);
}

TEST(StageRecipeTest, RejectsInvalidRequestScalarNameAndChargesScalarBindings) {
  static constexpr spec::RecipeValue one{.literal = std::int64_t{1}};
  static constexpr spec::RecipeValue width{.literal = std::int64_t{64}};
  static constexpr std::array projection_weights{
      spec::RecipeWeight{.role = "weight", .tensor = "projection.weight"}};
  static constexpr std::array noise_export{
      spec::RecipeRef{.kind = spec::RecipeRefKind::kNamedValue, .name = "noise"}};
  static constexpr std::array projected_shape{
      spec::RecipeSpatialShape{.width = one, .height = one, .channels = width, .images = one}};
  const spec::ModelSpec family{.pipeline = spec::PipelineKind::kImage};
  // A malformed name, and well-formed names outside the closed vocabulary of
  // request scalars the runtime supplies (spec::kStageScalarNames).
  for (const std::string_view scalar : {"Invalid-Scalar", "image_rows", "latent_depth"}) {
    const std::array bad_params{
        spec::RecipeParam{.name = "in_channels", .value = width},
        spec::RecipeParam{.name = "out_channels", .value = width},
        spec::RecipeParam{.name = "gelu", .value = {.literal = false}, .request_scalar = scalar}};
    const spec::RecipeNode bad_node{.name = "noise",
                                    .component_kind = "projection.spatial",
                                    .params = bad_params,
                                    .weights = projection_weights,
                                    .input = {.kind = spec::RecipeRefKind::kStageInput}};
    const spec::RecipeStep bad_step{.kind = spec::RecipeStep::Kind::kNode, .node = &bad_node};
    const std::array bad_steps{bad_step};
    const spec::StageRecipe bad_recipe{
        .name = "projected",
        .role = spec::StageRole::kDenoiser,
        .stream_width = width,
        .primary_input = {.shape = {.width = one, .height = one, .channels = width, .images = one}},
        .steps = bad_steps,
        .exports = noise_export,
        .expected_outputs = projected_shape};
    const auto rejected = PreflightStageRecipe(bad_recipe, family, 1U << 20U);
    ASSERT_FALSE(rejected) << scalar;
    EXPECT_NE(rejected.error().find("request scalar"), std::string::npos) << rejected.error();
  }
}

// A parameter or a raw-dimension contract only some sources have (its
// condition), a value from the family where one is given else the source's
// (kFirst), an expected output geometry the source does not state (not
// checked), and the names a recipe reads, collected by the same walk.
TEST(StageRecipe, ConditionsAndOneEnvironmentServeEverySource) {
  static constexpr spec::RecipeValue family_width = spec::ex::V("fixture.kind.width");
  static constexpr spec::RecipeValue source_width = spec::ex::V("width");
  static constexpr spec::RecipeValue width = spec::ex::First(family_width, source_width);
  static constexpr spec::RecipeValue flag_key = spec::ex::V("extra");
  static constexpr spec::RecipeValue no = spec::ex::B(false);
  static constexpr spec::RecipeValue extra = spec::ex::First(flag_key, no);
  static constexpr spec::RecipeValue rank = spec::ex::TensorRank("proj.weight");
  static constexpr spec::RecipeValue three = spec::ex::I(3);
  static constexpr spec::RecipeValue ranked = spec::ex::E(spec::ExprKind::kEqual, rank, three);
  static constexpr spec::RecipeValue one = spec::ex::I(1);
  static constexpr spec::RecipeValue unstated = spec::ex::V("factor");
  static constexpr std::array params{
      spec::RecipeParam{.name = "in_channels", .value = width},
      spec::RecipeParam{.name = "out_channels", .value = width},
      spec::RecipeParam{.name = "silu", .value = spec::ex::B(true), .when = &extra}};
  static constexpr std::array dims{width, width, one};
  static constexpr std::array weights{spec::RecipeWeight{
      .role = "weight", .tensor = "proj.weight", .source_dims = dims, .source_dims_when = &ranked}};
  static constexpr spec::RecipeNode node{.name = "proj",
                                         .component_kind = "projection.spatial",
                                         .params = params,
                                         .weights = weights,
                                         .input = {.kind = spec::RecipeRefKind::kStageInput}};
  static constexpr std::array steps{
      spec::RecipeStep{.kind = spec::RecipeStep::Kind::kNode, .node = &node}};
  static constexpr spec::RecipeSpatialShape shape{
      .width = one, .height = one, .channels = width, .images = one};
  static constexpr spec::RecipeSpatialShape output{
      .width = unstated, .height = one, .channels = width, .images = one};
  static constexpr std::array outputs{output};
  static constexpr spec::StageRecipe recipe{.name = "fixture",
                                            .role = spec::StageRole::kDenoiser,
                                            .stream_width = width,
                                            .primary_input = {.shape = shape},
                                            .steps = steps,
                                            .expected_outputs = outputs};
  const std::array config{MetaEntry{.key = "width", .value = std::int64_t{4}}};
  const std::array two_d{Tensor{.name = "proj.weight", .dims = {4, 4}}};
  auto plain = ExpandStageRecipe(recipe, {}, 1U << 20U, config, two_d);
  ASSERT_TRUE(plain) << plain.error();
  const auto& component = plain->stage.layers[0].components[0];
  EXPECT_EQ(component.Find("silu"), nullptr);
  EXPECT_EQ(component.Find("in_channels")->value, (spec::ParamValue{std::int64_t{4}}));
  EXPECT_TRUE(plain->tensors[0].source_dims.empty());
  // Indexing preserves name/part ambiguity rules, including a non-data
  // companion with the same logical name and an unrelated duplicate name.
  std::vector<Tensor> indexed(two_d.begin(), two_d.end());
  indexed.push_back({.name = "proj.weight", .part = format::PartRole::kTensorScale, .dims = {1}});
  indexed.push_back({.name = "unused", .dims = {1}});
  indexed.push_back({.name = "unused", .dims = {2}});
  EXPECT_TRUE(ExpandStageRecipe(recipe, {}, 1U << 20U, config, indexed));
  indexed.push_back(two_d[0]);
  const auto ambiguous = ExpandStageRecipe(recipe, {}, 1U << 20U, config, indexed);
  ASSERT_FALSE(ambiguous);
  EXPECT_NE(ambiguous.error().find("ambiguous"), std::string::npos);
  const auto unfunded_index =
      PreflightStageRecipe(recipe, {}, indexed.size() * sizeof(const Tensor*) - 1, config, indexed);
  ASSERT_FALSE(unfunded_index);
  EXPECT_NE(unfunded_index.error().find("source tensor index"), std::string::npos);

  const std::array flagged{MetaEntry{.key = "extra", .value = true},
                           MetaEntry{.key = "width", .value = std::int64_t{4}}};
  const std::array three_d{Tensor{.name = "proj.weight", .dims = {4, 4, 1}}};
  auto extended = ExpandStageRecipe(recipe, {}, 1U << 20U, flagged, three_d);
  ASSERT_TRUE(extended) << extended.error();
  EXPECT_NE(extended->stage.layers[0].components[0].Find("silu"), nullptr);
  EXPECT_EQ(extended->tensors[0].source_dims, (std::vector<std::uint64_t>{4, 4, 1}));
  // A family's value is the one read, the source's key then unread.
  spec::ModelSpec family{.pipeline = spec::PipelineKind::kImage};
  family.stages.push_back({.layers = {{.components = {{.kind = "fixture.kind",
                                                       .params = {{"width", std::int64_t{8}}}}}}}});
  auto from_family = ExpandStageRecipe(recipe, family, 1U << 20U, config, two_d);
  ASSERT_TRUE(from_family) << from_family.error();
  EXPECT_EQ(from_family->stage.stream.width, 8U);
  // Without the kernel's header its condition cannot be read.
  EXPECT_FALSE(ExpandStageRecipe(recipe, {}, 1U << 20U, config));
  // A stated geometry is checked.
  const std::array stated{MetaEntry{.key = "factor", .value = std::int64_t{2}},
                          MetaEntry{.key = "width", .value = std::int64_t{4}}};
  EXPECT_FALSE(ExpandStageRecipe(recipe, {}, 1U << 20U, stated, two_d));

  const auto names = RecipeValueNames(recipe);
  ASSERT_TRUE(names) << names.error();
  EXPECT_EQ(*names, (std::set<std::string_view>{"extra", "factor", "fixture.kind.width", "width"}));
  // Counting is the expansion's own accounting: exactly what it charges.
  const auto counted = PreflightStageRecipe(recipe, {}, 1U << 20U, flagged, three_d);
  ASSERT_TRUE(counted) << counted.error();
  EXPECT_EQ(counted->tensor_bindings, 1U);
  EXPECT_EQ(counted->raw_dimension_axes, 3U);
  EXPECT_TRUE(ExpandStageRecipe(recipe, {}, counted->metadata_bytes, flagged, three_d));
  EXPECT_FALSE(ExpandStageRecipe(recipe, {}, counted->metadata_bytes - 1, flagged, three_d));
}

}  // namespace
}  // namespace llmp::import
