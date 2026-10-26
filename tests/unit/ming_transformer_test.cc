// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "spec/mappings/ming_transformer.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <map>
#include <ranges>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#include "base/sha256.h"
#include "import/stage_recipe.h"

namespace llmp::import {
namespace {

constexpr std::uint32_t kWidth = 3840;
constexpr std::uint32_t kHeads = 30;
constexpr std::uint32_t kHeadDim = 128;
constexpr std::uint32_t kHidden = 10240;
constexpr std::uint32_t kChannels = 15360;
constexpr double kEps = 1.0e-5;

std::vector<MetaEntry> PinnedSource() {
  return {{.key = "dim", .value = std::int64_t{kWidth}},
          {.key = "n_heads", .value = std::int64_t{kHeads}},
          {.key = "cap_feat_dim", .value = std::int64_t{2560}},
          {.key = "in_channels", .value = std::int64_t{16}},
          {.key = "all_patch_size", .value = std::vector<std::int64_t>{2}},
          {.key = "all_f_patch_size", .value = std::vector<std::int64_t>{1}},
          {.key = "n_refiner_layers", .value = std::int64_t{2}},
          {.key = "n_layers", .value = std::int64_t{30}},
          {.key = "norm_eps", .value = kEps},
          {.key = "rope_theta", .value = 256.0},
          {.key = "qk_norm", .value = true},
          {.key = "axes_dims", .value = std::vector<std::int64_t>{32, 48, 48}}};
}

const spec::Param* Parameter(const spec::ComponentRef& component, std::string_view name) {
  return component.Find(name);
}

struct ExpectedTensor {
  std::vector<std::uint64_t> shape;
  spec::RecipeGroupKind group = spec::RecipeGroupKind::kWhole;
  std::uint32_t layer = 0;
  bool operator==(const ExpectedTensor&) const = default;
};

std::map<std::string, ExpectedTensor> PinnedTensorInventory() {
  std::map<std::string, ExpectedTensor> expected;
  const auto whole = [&](std::string name, std::vector<std::uint64_t> shape) {
    expected.emplace(std::move(name), ExpectedTensor{.shape = std::move(shape)});
  };
  const auto layer = [&](std::string name, std::vector<std::uint64_t> shape, std::uint32_t index) {
    expected.emplace(std::move(name), ExpectedTensor{.shape = std::move(shape),
                                                     .group = spec::RecipeGroupKind::kLayer,
                                                     .layer = index});
  };
  whole("t_embedder.mlp.0.weight", {1024, 256});
  whole("t_embedder.mlp.0.bias", {1024});
  whole("t_embedder.mlp.2.weight", {256, 1024});
  whole("t_embedder.mlp.2.bias", {256});
  whole("cap_embedder.0.weight", {2560});
  whole("cap_embedder.1.weight", {kWidth, 2560});
  whole("cap_embedder.1.bias", {kWidth});
  whole("all_x_embedder.2-1.weight", {kWidth, 64});
  whole("all_x_embedder.2-1.bias", {kWidth});

  const auto add_block = [&](std::string_view prefix, std::uint32_t layer_index, bool modulation) {
    const std::string p(prefix);
    if (modulation) {
      layer(p + ".adaLN_modulation.0.weight", {kChannels, 256}, layer_index);
      layer(p + ".adaLN_modulation.0.bias", {kChannels}, layer_index);
    }
    layer(p + ".attention_norm1.weight", {kWidth}, layer_index);
    for (const auto* name : {"to_k", "to_q", "to_v", "to_out.0"})
      layer(p + ".attention." + name + ".weight", {kWidth, kWidth}, layer_index);
    layer(p + ".attention.norm_k.weight", {kHeadDim}, layer_index);
    layer(p + ".attention.norm_q.weight", {kHeadDim}, layer_index);
    layer(p + ".attention_norm2.weight", {kWidth}, layer_index);
    layer(p + ".ffn_norm1.weight", {kWidth}, layer_index);
    layer(p + ".feed_forward.w1.weight", {kHidden, kWidth}, layer_index);
    layer(p + ".feed_forward.w3.weight", {kHidden, kWidth}, layer_index);
    layer(p + ".feed_forward.w2.weight", {kWidth, kHidden}, layer_index);
    layer(p + ".ffn_norm2.weight", {kWidth}, layer_index);
  };
  for (std::uint32_t i = 0; i < 2; ++i) {
    add_block("noise_refiner." + std::to_string(i), i, true);
    add_block("context_refiner." + std::to_string(i), i + 2, false);
  }
  for (std::uint32_t i = 0; i < 30; ++i) add_block("layers." + std::to_string(i), i + 4, true);
  whole("all_final_layer.2-1.adaLN_modulation.1.weight", {kWidth, 256});
  whole("all_final_layer.2-1.adaLN_modulation.1.bias", {kWidth});
  whole("all_final_layer.2-1.linear.weight", {64, kWidth});
  whole("all_final_layer.2-1.linear.bias", {64});
  return expected;
}

void ExpectIntegerParam(const spec::ComponentRef& component, std::string_view name,
                        std::int64_t expected) {
  const auto* param = Parameter(component, name);
  ASSERT_NE(param, nullptr) << name;
  const auto* value = std::get_if<std::int64_t>(&param->value);
  ASSERT_NE(value, nullptr) << name;
  EXPECT_EQ(*value, expected) << name;
}

void ExpectRealParam(const spec::ComponentRef& component, std::string_view name, double expected) {
  const auto* param = Parameter(component, name);
  ASSERT_NE(param, nullptr) << name;
  const auto* value = std::get_if<double>(&param->value);
  ASSERT_NE(value, nullptr) << name;
  EXPECT_DOUBLE_EQ(*value, expected) << name;
}

void ExpectStringParam(const spec::ComponentRef& component, std::string_view name,
                       std::string_view expected) {
  const auto* param = Parameter(component, name);
  ASSERT_NE(param, nullptr) << name;
  const auto* value = std::get_if<std::string>(&param->value);
  ASSERT_NE(value, nullptr) << name;
  EXPECT_EQ(*value, expected) << name;
}

void ExpectFlagParam(const spec::ComponentRef& component, std::string_view name, bool expected) {
  const auto* param = Parameter(component, name);
  ASSERT_NE(param, nullptr) << name;
  const auto* value = std::get_if<bool>(&param->value);
  ASSERT_NE(value, nullptr) << name;
  EXPECT_EQ(*value, expected) << name;
}

TEST(MingTransformer, ExpandsPinnedLearnedCoreToCompleteTopologyAndInventory) {
  const auto& mapping = spec::mappings::MingTransformerMapping();
  ASSERT_TRUE(spec::ValidateMapping(mapping));
  auto source = PinnedSource();
  auto expanded = ExpandStageRecipe(*mapping.stage_recipe, {}, 16U << 20U, source);
  ASSERT_TRUE(expanded) << expanded.error();
  ASSERT_EQ(expanded->stage.layers.size(), 1U);
  const auto& stage = expanded->stage;
  const auto& components = stage.layers[0].components;
  ASSERT_EQ(components.size(), 508U);
  ASSERT_EQ(stage.links.size(), components.size());
  ASSERT_EQ(expanded->tensors.size(), 519U);
  ASSERT_EQ(stage.exports.size(), 1U);
  EXPECT_EQ(stage.context, 0U);
  EXPECT_EQ(stage.stream.width, kWidth);
  spec::ModelSpec canonical;
  canonical.pipeline = spec::PipelineKind::kImage;
  canonical.stages.push_back(stage);
  spec::Canonicalize(canonical);
  // Re-recorded 2026-10-25: the output slice binds latent_rows, a request
  // scalar the runtime supplies (spec::kStageScalarNames), not image_rows.
  EXPECT_EQ(base::ToHex(canonical.Digest()),
            "b0368f3d04eaaeeb9840bfa5238366577c96146e410b0089248a8b5b292a27dc");

  const auto count = [&](std::string_view kind) {
    return std::ranges::count(components, kind, &spec::ComponentRef::kind);
  };
  EXPECT_EQ(count("attention.dense"), 34);
  EXPECT_EQ(count("ffn.glu"), 34);
  EXPECT_EQ(count("residual.gated"), 64);
  EXPECT_EQ(count("residual.spatial"), 4);
  EXPECT_EQ(count("concat.spatial"), 1);
  EXPECT_EQ(count("slice.spatial"), 1);

  // Compare every physical source binding against an independently enumerated
  // inventory derived from the pinned checkpoint config and tensor contract.
  const auto expected = PinnedTensorInventory();
  std::map<std::string, ExpectedTensor> actual;
  for (const auto& tensor : expanded->tensors) {
    const auto [unused, inserted] = actual.emplace(
        tensor.name,
        ExpectedTensor{.shape = tensor.shape, .group = tensor.group, .layer = tensor.layer});
    (void)unused;
    EXPECT_TRUE(inserted) << "duplicate physical tensor " << tensor.name;
    EXPECT_TRUE(tensor.source_dims.empty()) << tensor.name;
  }
  EXPECT_EQ(actual, expected);

  // The 15-node modulated path is checked for every noise and joint block:
  // source-derived quarter parameters, pre/post norms, attention host roots,
  // two gated residuals, and the saved residual carried into the FFN branch.
  const auto check_modulated = [&](std::size_t first, std::uint32_t block, std::uint32_t group,
                                   std::uint16_t stream_value, std::uint16_t positions_port,
                                   std::uint16_t mask_port, std::string_view prefix) {
    const auto& modulation = components[first];
    ExpectIntegerParam(modulation, "out_channels", kChannels);
    const auto& norm1 = components[first + 5];
    const auto& attention = components[first + 7];
    const auto& post_attention = components[first + 8];
    const auto& norm2 = components[first + 10];
    const auto& ffn = components[first + 12];
    const auto& post_ffn = components[first + 13];
    EXPECT_EQ(stage.links[first].input, 2);
    for (const auto [offset, quarter] :
         {std::pair<std::size_t, std::int64_t>{1, 0}, {2, 1}, {3, 2}, {4, 3}})
      ExpectIntegerParam(components[first + offset], "quarter", quarter);
    for (std::size_t offset = 1; offset <= 4; ++offset) {
      EXPECT_EQ(stage.links[first + offset].input, stream_value);
      ASSERT_TRUE(stage.links[first + offset].auxiliary);
      EXPECT_EQ(*stage.links[first + offset].auxiliary, first + 1);
    }
    for (const auto* norm : {&norm1, &post_attention, &norm2, &post_ffn})
      ExpectRealParam(*norm, "eps", kEps);
    ExpectFlagParam(attention, "qk_norm", true);
    ExpectIntegerParam(attention, "heads", kHeads);
    ExpectIntegerParam(attention, "key_dim", kHeadDim);
    ExpectIntegerParam(attention, "rope_axis0", 32);
    ExpectIntegerParam(attention, "rope_axis1", 48);
    ExpectIntegerParam(attention, "rope_axis2", 48);
    ExpectRealParam(attention, "rope_base", 256.0);
    ExpectStringParam(ffn, "activation", "silu");
    ExpectIntegerParam(ffn, "hidden", kHidden);
    EXPECT_EQ(stage.links[first + 5].input, stream_value);
    EXPECT_EQ(stage.links[first + 6].input, first + 6);
    ASSERT_TRUE(stage.links[first + 6].auxiliary);
    EXPECT_EQ(*stage.links[first + 6].auxiliary, first + 2);
    EXPECT_EQ(stage.links[first + 7].input, first + 7);
    EXPECT_EQ(stage.links[first + 7].auxiliary_port, positions_port);
    EXPECT_EQ(stage.links[first + 7].carry_port, mask_port);
    EXPECT_EQ(stage.links[first + 8].input, first + 8);
    EXPECT_EQ(stage.links[first + 9].input, first + 9);
    ASSERT_TRUE(stage.links[first + 9].auxiliary);
    EXPECT_EQ(*stage.links[first + 9].auxiliary, first + 3);
    ASSERT_TRUE(stage.links[first + 9].carry);
    EXPECT_EQ(*stage.links[first + 9].carry, stream_value);
    EXPECT_EQ(stage.links[first + 10].input, first + 10);
    EXPECT_EQ(stage.links[first + 11].input, first + 11);
    ASSERT_TRUE(stage.links[first + 11].auxiliary);
    EXPECT_EQ(*stage.links[first + 11].auxiliary, first + 4);
    EXPECT_EQ(stage.links[first + 12].input, first + 12);
    EXPECT_EQ(stage.links[first + 13].input, first + 13);
    EXPECT_EQ(stage.links[first + 14].input, first + 14);
    ASSERT_TRUE(stage.links[first + 14].auxiliary);
    EXPECT_EQ(*stage.links[first + 14].auxiliary, first + 5);
    ASSERT_TRUE(stage.links[first + 14].carry);
    EXPECT_EQ(*stage.links[first + 14].carry, first + 10);
    for (const auto [offset, role, suffix] :
         {std::tuple<std::size_t, std::string_view, std::string_view>{12, "gate", "w1"},
          {12, "up", "w3"},
          {12, "down", "w2"}}) {
      const auto& component = components[first + offset];
      const auto found = std::ranges::find_if(
          component.weights,
          [&](const spec::WeightBinding& weight) { return weight.role == role; });
      ASSERT_NE(found, component.weights.end());
      EXPECT_EQ(found->tensor,
                std::string(prefix) + ".feed_forward." + std::string(suffix) + ".weight");
    }
    const std::string q_name = std::string(prefix) + ".attention.to_q.weight";
    const auto q_tensor =
        std::ranges::find(expanded->tensors, q_name, &spec::RecipeExpansion::Tensor::name);
    ASSERT_NE(q_tensor, expanded->tensors.end());
    EXPECT_EQ(q_tensor->layer, group);
    (void)block;
  };
  for (std::uint32_t block = 0; block < 30; ++block) {
    if (block < 2) {
      const std::size_t noise = 5U + 15U * block;
      check_modulated(noise, block, block, block == 0 ? 5 : static_cast<std::uint16_t>(noise), 2, 3,
                      "noise_refiner." + std::to_string(block));
    }
    const std::size_t joint = 52U + 15U * block;
    check_modulated(joint, block, block + 4, block == 0 ? 52 : static_cast<std::uint16_t>(joint), 6,
                    7, "layers." + std::to_string(block));
  }

  // Two unmodulated context refiners still normalize after each branch before
  // residual addition and feed the next layer from the first residual.
  for (std::uint32_t block = 0; block < 2; ++block) {
    const std::size_t first = 35U + 8U * block;
    const std::uint16_t stream = block == 0 ? 4 : static_cast<std::uint16_t>(first);
    const auto& attention = components[first + 1];
    ExpectFlagParam(attention, "qk_norm", true);
    ExpectIntegerParam(attention, "rope_axis0", 32);
    ExpectIntegerParam(attention, "rope_axis1", 48);
    ExpectIntegerParam(attention, "rope_axis2", 48);
    ExpectRealParam(components[first + 2], "eps", kEps);
    EXPECT_EQ(stage.links[first + 1].auxiliary_port, 4);
    EXPECT_EQ(stage.links[first + 1].carry_port, 5);
    EXPECT_EQ(stage.links[first + 3].input, first + 3);
    ASSERT_TRUE(stage.links[first + 3].auxiliary);
    EXPECT_EQ(*stage.links[first + 3].auxiliary, stream);
    EXPECT_EQ(stage.links[first + 4].input, first + 4);
    EXPECT_EQ(stage.links[first + 5].input, first + 5);
    EXPECT_EQ(stage.links[first + 6].input, first + 6);
    EXPECT_EQ(stage.links[first + 7].input, first + 7);
    ASSERT_TRUE(stage.links[first + 7].auxiliary);
    EXPECT_EQ(*stage.links[first + 7].auxiliary, first + 4);
    (void)attention;
  }

  // Prefix construction is image-first then caption; the head applies SiLU
  // before its biased modulation, then adaptive norm, output projection, slice.
  const std::size_t concat = 51;
  EXPECT_EQ(components[concat].kind, "concat.spatial");
  EXPECT_EQ(stage.links[concat].input, 35);
  ASSERT_TRUE(stage.links[concat].auxiliary);
  EXPECT_EQ(*stage.links[concat].auxiliary, 51);
  ASSERT_EQ(components.size(), 508U);
  EXPECT_EQ(components[502].kind, "activation.elementwise");
  ExpectStringParam(components[502], "activation", "silu");
  EXPECT_EQ(stage.links[502].input, 2);
  EXPECT_EQ(components[503].kind, "projection.spatial");
  EXPECT_EQ(stage.links[503].input, 503);
  ExpectFlagParam(components[503], "bias", true);
  EXPECT_EQ(components[504].kind, "condition.rows");
  EXPECT_EQ(stage.links[504].input, 502);
  EXPECT_EQ(stage.links[504].auxiliary, 504);
  EXPECT_EQ(components[505].kind, "norm.adaptive");
  ExpectRealParam(components[505], "eps", 1.0e-6);
  EXPECT_EQ(stage.links[505].input, 502);
  EXPECT_EQ(stage.links[505].auxiliary, 505);
  EXPECT_EQ(components[506].kind, "projection.spatial");
  EXPECT_EQ(stage.links[506].input, 506);
  ExpectFlagParam(components[506], "bias", true);
  EXPECT_EQ(components[507].kind, "slice.spatial");
  EXPECT_EQ(stage.links[507].input, 507);
  EXPECT_EQ(stage.exports[0], 508);

  EXPECT_EQ(stage.links[0].input_port, 1);
  EXPECT_EQ(stage.links[1].input, 1);
  ExpectFlagParam(components[0], "bias", true);
  ExpectFlagParam(components[0], "silu", true);
  ExpectFlagParam(components[1], "bias", true);
  ExpectFlagParam(components[1], "silu", false);
  EXPECT_EQ(stage.links[2].input, 0);
  EXPECT_EQ(stage.links[3].input, 3);
  EXPECT_EQ(stage.links[4].input_port, 0);

  // This source value controls the optional learned Q/K normalization. The
  // mapping supports the pinned true topology and refuses unsupported forms.
  for (const std::string_view invalid : {"false", "missing", "wrong-type"}) {
    auto bad_source = source;
    const auto qk = std::ranges::find(bad_source, "qk_norm", &MetaEntry::key);
    ASSERT_NE(qk, bad_source.end());
    if (invalid == "missing")
      bad_source.erase(qk);
    else if (invalid == "false")
      qk->value = false;
    else
      qk->value = 1.0;
    auto rejected = ExpandStageRecipe(*mapping.stage_recipe, {}, 16U << 20U, bad_source);
    EXPECT_FALSE(rejected) << invalid;
  }

  auto bad = source;
  const auto heads = std::ranges::find(bad, "n_heads", &MetaEntry::key);
  ASSERT_NE(heads, bad.end());
  heads->value = std::int64_t{32};
  auto rejected = ExpandStageRecipe(*mapping.stage_recipe, {}, 16U << 20U, bad);
  EXPECT_FALSE(rejected);
}

TEST(MingTransformer, FloorsHeadAndMlpDimensionsForOddModelWidth) {
  const auto& mapping = spec::mappings::MingTransformerMapping();
  auto source = PinnedSource();
  const auto dim = std::ranges::find(source, "dim", &MetaEntry::key);
  ASSERT_NE(dim, source.end());
  dim->value = std::int64_t{257};
  const auto heads = std::ranges::find(source, "n_heads", &MetaEntry::key);
  ASSERT_NE(heads, source.end());
  heads->value = std::int64_t{2};
  const auto expanded = ExpandStageRecipe(*mapping.stage_recipe, {}, 16U << 20U, source);
  ASSERT_TRUE(expanded) << expanded.error();
  const auto& components = expanded->stage.layers[0].components;
  const auto attention =
      std::ranges::find(components, "attention.dense", &spec::ComponentRef::kind);
  ASSERT_NE(attention, components.end());
  ExpectIntegerParam(*attention, "key_dim", 128);
  ExpectIntegerParam(*attention, "rope_axis0", 32);
  ExpectIntegerParam(*attention, "rope_axis1", 48);
  ExpectIntegerParam(*attention, "rope_axis2", 48);
  const auto ffn = std::ranges::find(components, "ffn.glu", &spec::ComponentRef::kind);
  ASSERT_NE(ffn, components.end());
  ExpectIntegerParam(*ffn, "hidden", 685);
  const auto q_norm =
      std::ranges::find(expanded->tensors, "noise_refiner.0.attention.norm_q.weight",
                        &spec::RecipeExpansion::Tensor::name);
  ASSERT_NE(q_norm, expanded->tensors.end());
  EXPECT_EQ(q_norm->shape, (std::vector<std::uint64_t>{128}));
}

}  // namespace
}  // namespace llmp::import
