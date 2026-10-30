// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <expected>
#include <format>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "components/spatial.h"
#include "components/spatial_stream.h"
#include "components/step.h"
#include "format/tensor_format.h"
#include "import/normalize.h"
#include "import/source_mapping.h"
#include "import/stage_recipe.h"
#include "spec/mappings/qwen_image_vae.h"

namespace llmp::import {
namespace {

std::vector<MetaEntry> PinnedConfig() {
  return {{.key = "base_dim", .value = std::int64_t{96}},
          {.key = "z_dim", .value = std::int64_t{16}},
          {.key = "dim_mult", .value = std::vector<std::int64_t>{1, 2, 4, 4}},
          {.key = "num_res_blocks", .value = std::int64_t{2}},
          {.key = "input_channels", .value = std::int64_t{4}},
          {.key = "scaling_factor", .value = 8.0064},
          {.key = "shift_factor", .value = 0.0}};
}

// Ming's causal 3-D kernels, as its checkpoint's header states its first
// convolution's: what selects the mapping's 3-D dialect.
std::vector<Tensor> PinnedKernels(const std::vector<MetaEntry>& config) {
  const auto value = [&](std::string_view key) -> std::int64_t {
    const auto found = std::ranges::find(config, key, &MetaEntry::key);
    return found == config.end() ? 0 : std::get<std::int64_t>(found->value);
  };
  const auto mult = std::ranges::find(config, "dim_mult", &MetaEntry::key);
  const auto& list = std::get<std::vector<std::int64_t>>(mult->value);
  const auto mid = static_cast<std::uint64_t>(value("base_dim") * list.back());
  return {Tensor{.name = "decoder.conv_in.weight",
                 .dims = {mid, static_cast<std::uint64_t>(value("z_dim")), 3, 3, 3}}};
}

std::expected<spec::RecipeExpansion, std::string> Expand(const std::vector<MetaEntry>& config) {
  const auto kernels = PinnedKernels(config);
  return ExpandStageRecipe(*spec::mappings::QwenImageVaeMapping().stage_recipe, {}, 32U << 20U,
                           config, kernels, "safetensors");
}

const spec::Param* Param(const spec::ComponentRef& component, std::string_view name) {
  const auto found = std::ranges::find(component.params, name, &spec::Param::name);
  return found == component.params.end() ? nullptr : &*found;
}

std::expected<components::SpatialShape, std::string> InferFinalShape(
    const spec::Stage& stage, components::SpatialShape input_shape) {
  if (stage.layers.size() != 1) return std::unexpected("test recipe did not expand to one layer");
  const auto& refs = stage.layers[0].components;
  if (stage.links.size() != refs.size())
    return std::unexpected("expanded spatial links differ from component count");
  const components::StageFacts facts{.width = stage.stream.width,
                                     .context = stage.context,
                                     .softcap = stage.heads.softcap,
                                     .streams = stage.stream.streams,
                                     .folds = stage.folds};
  std::vector<components::SpatialOperand> values{
      components::SpatialOperand{.shape = input_shape, .encoding = format::Encoding::kF32}};
  values.reserve(refs.size() + 1);
  for (std::size_t i = 0; i < refs.size(); ++i) {
    auto instance = components::MakeInstance(refs[i], facts, components::Site::kLayer, 0,
                                             static_cast<std::uint16_t>(i));
    if (!instance) return std::unexpected(instance.error());
    const auto& link = stage.links[i];
    if (link.input >= values.size() || (link.auxiliary && *link.auxiliary >= values.size()) ||
        link.input_port || link.auxiliary_port || link.carry_port)
      return std::unexpected("test mapping contains an unsupported typed edge");
    const auto auxiliary = link.auxiliary ? std::optional(values[*link.auxiliary]) : std::nullopt;
    const auto carry = link.carry ? std::optional(values[*link.carry]) : std::nullopt;
    if (carry &&
        (carry->encoding != format::Encoding::kF32 ||
         carry->shape != components::CarrySpatialShape(*instance->def, values[link.input].shape)))
      return std::unexpected("test mapping has an invalid carry geometry");
    auto output = components::InferSpatialOutput(*instance, values[link.input], auxiliary, carry);
    if (!output) return std::unexpected(output.error());
    values.push_back(*output);
  }
  return values.back().shape;
}

// One mapping for the source class, whichever variant: Ming's VAE and
// Qwen-Image-2.1's (a second configuration of the same architecture).
TEST(MingVae, RegistryAndPinnedConfigUseDecoderInputChannels) {
  const auto& direct = spec::mappings::QwenImageVaeMapping();
  const auto* registered = FindImportMapping("safetensors", "AutoencoderKLQwenImage", "vae");
  ASSERT_NE(registered, nullptr);
  EXPECT_EQ(registered, &direct);
  EXPECT_EQ(FindImportMapping("safetensors", "AutoencoderKLQwenImage21", "vae"), &direct);
  EXPECT_TRUE(spec::ValidateMapping(direct));
  const auto config = PinnedConfig();
  EXPECT_EQ(std::ranges::find(config, "out_channels", &MetaEntry::key), config.end());
  auto expanded = Expand(config);
  ASSERT_TRUE(expanded) << expanded.error();
  const auto& components = expanded->stage.layers[0].components;
  // Its causal 3-D kernels' first frame is conv.spatial's temporal view.
  EXPECT_EQ(std::ranges::count(components, "conv.causal_first_frame", &spec::ComponentRef::kind),
            0);
  const auto output =
      std::ranges::find_if(components.rbegin(), components.rend(), [](const auto& ref) {
        return ref.kind == "conv.spatial" && ref.Find("temporal_kernel") != nullptr;
      });
  ASSERT_NE(output, components.rend());
  const auto* channels = Param(*output, "out_channels");
  ASSERT_NE(channels, nullptr);
  ASSERT_TRUE(std::holds_alternative<std::int64_t>(channels->value));
  EXPECT_EQ(std::get<std::int64_t>(channels->value), 4);
  // The halving decoder's output is clamped, its norms round to BF16, its
  // latent affine is the config's scalar one.
  EXPECT_EQ(components.back().kind, "activation.elementwise");
  EXPECT_EQ(std::ranges::count(components, "affine.scalar", &spec::ComponentRef::kind), 1);
  EXPECT_EQ(std::ranges::count(components, "affine.channel", &spec::ComponentRef::kind), 0);
  for (const auto& component : components)
    if (component.kind == "norm.channel") EXPECT_NE(component.Find("round_bf16"), nullptr);
  // Without its kernels' rank (no tensor inventory) the dialect is unknown.
  EXPECT_FALSE(ExpandStageRecipe(*registered->stage_recipe, {}, 32U << 20U, config));
}

TEST(MingVae, HalvesCompletePreviousChannelCount) {
  auto config = PinnedConfig();
  const auto mult = std::ranges::find(config, "dim_mult", &MetaEntry::key);
  ASSERT_NE(mult, config.end());
  mult->value = std::vector<std::int64_t>{1, 2, 3, 3};
  auto expanded = Expand(config);
  ASSERT_TRUE(expanded) << expanded.error();
  const auto weight =
      std::ranges::find(expanded->tensors, "decoder.up_blocks.1.resnets.0.conv1.weight",
                        &spec::RecipeExpansion::Tensor::name);
  ASSERT_NE(weight, expanded->tensors.end());
  EXPECT_EQ(weight->source_dims, (std::vector<std::uint64_t>{288, 144, 3, 3, 3}));
}

TEST(MingVae, FloorsOddCompleteChannelTotals) {
  auto config = PinnedConfig();
  const auto base = std::ranges::find(config, "base_dim", &MetaEntry::key);
  ASSERT_NE(base, config.end());
  base->value = std::int64_t{95};
  const auto mult = std::ranges::find(config, "dim_mult", &MetaEntry::key);
  ASSERT_NE(mult, config.end());
  mult->value = std::vector<std::int64_t>{1, 2, 3, 3};
  const auto expanded = Expand(config);
  ASSERT_TRUE(expanded) << expanded.error();
  std::vector<std::int64_t> output_channels;
  for (const auto& component : expanded->stage.layers[0].components) {
    if (component.kind != "conv.spatial") continue;
    if (std::ranges::none_of(component.weights, [](const auto& weight) {
          return weight.tensor.find(".upsamplers.0.resample.1.weight") != std::string::npos;
        }))
      continue;
    const auto* output = Param(component, "out_channels");
    ASSERT_NE(output, nullptr);
    ASSERT_TRUE(std::holds_alternative<std::int64_t>(output->value));
    output_channels.push_back(std::get<std::int64_t>(output->value));
  }
  EXPECT_EQ(output_channels, (std::vector<std::int64_t>{142, 142, 95}));
}

TEST(MingVae, ThreeLevelRecipeInfersFourByFourOutput) {
  // Ming's config states no spatial factor: its decoded geometry is
  // inferred, not checked against one.
  auto config = PinnedConfig();
  EXPECT_EQ(std::ranges::find(config, "scale_factor_spatial", &MetaEntry::key), config.end());
  const auto mult = std::ranges::find(config, "dim_mult", &MetaEntry::key);
  ASSERT_NE(mult, config.end());
  mult->value = std::vector<std::int64_t>{1, 2, 4};
  auto expanded = Expand(config);
  ASSERT_TRUE(expanded) << expanded.error();
  EXPECT_EQ(std::ranges::count(expanded->stage.layers[0].components, "upsample.nearest",
                               &spec::ComponentRef::kind),
            2);
  const auto output = InferFinalShape(expanded->stage, {.width = 1, .height = 1, .channels = 16});
  ASSERT_TRUE(output) << output.error();
  EXPECT_EQ(output->width, 4U);
  EXPECT_EQ(output->height, 4U);
  EXPECT_EQ(output->channels, 4U);
}

struct SyntheticContracts {
  std::vector<std::string> tensor_names;
  std::vector<std::string> node_names;
  std::vector<std::array<spec::RecipeWeight, 1>> node_weights;
  std::vector<spec::RecipeNode> nodes;
  std::vector<spec::RecipeStep> steps;
  spec::StageRecipe recipe;
  spec::Mapping mapping;
  Checkpoint source;
};

std::unique_ptr<SyntheticContracts> MakeSyntheticContracts(std::size_t count, bool contracted) {
  auto out = std::make_unique<SyntheticContracts>();
  out->tensor_names.reserve(count);
  out->node_names.reserve(count);
  out->node_weights.reserve(count);
  out->nodes.reserve(count);
  out->steps.reserve(count);
  for (std::size_t i = 0; i < count; ++i) {
    out->tensor_names.push_back(std::string(196, 'p') + std::format("{:04x}", i));
    out->node_names.push_back(std::format("projection.{}", i));
  }
  static constexpr spec::RecipeValue kOne{.kind = spec::RecipeValueKind::kLiteral,
                                          .literal = std::int64_t{1}};
  static constexpr std::array kDims{kOne, kOne};
  static constexpr std::array kParams{spec::RecipeParam{.name = "in_channels", .value = kOne},
                                      spec::RecipeParam{.name = "out_channels", .value = kOne}};
  static constexpr spec::RecipeSpatialShape kShape{
      .width = kOne, .height = kOne, .channels = kOne, .images = kOne};
  for (std::size_t i = 0; i < count; ++i) {
    out->node_weights.push_back({spec::RecipeWeight{
        .role = "weight",
        .tensor = out->tensor_names[i],
        .source_dims = contracted ? std::span(kDims) : std::span<const spec::RecipeValue>{}}});
    out->nodes.push_back({.name = out->node_names[i],
                          .component_kind = "projection.spatial",
                          .params = kParams,
                          .weights = out->node_weights.back(),
                          .input = {.kind = spec::RecipeRefKind::kStageInput}});
    out->steps.push_back({.kind = spec::RecipeStep::Kind::kNode, .node = &out->nodes.back()});
  }
  out->recipe = {.name = "vae",
                 .role = spec::StageRole::kVae,
                 .stream_width = kOne,
                 .primary_input = {.shape = kShape},
                 .steps = out->steps};
  out->mapping = spec::mappings::QwenImageVaeMapping();
  out->mapping.stage_recipe = &out->recipe;

  const std::array<std::uint64_t, 2> dims{1, 1};
  const auto shape = format::ShapeOf(dims);
  if (!shape) return nullptr;
  const format::TensorFormat tensor_format{
      .encoding = format::Encoding::kBF16, .layout = format::Layout::kDense, .shape = *shape};
  const auto parts = format::PartsOf(tensor_format);
  if (!parts || parts->size() != 1) return nullptr;
  const auto bytes = (*parts)[0].bytes;
  out->source.container = "safetensors";
  out->source.architecture = "AutoencoderKLQwenImage";
  out->source.files.push_back({.name = "model.safetensors",
                               .path = "/tmp/model.safetensors",
                               .bytes = count * bytes,
                               .device = 1,
                               .inode = 1});
  out->source.tensors.reserve(count);
  for (std::size_t i = 0; i < count; ++i)
    out->source.tensors.push_back({.name = out->tensor_names[i],
                                   .format = tensor_format,
                                   .extent = {.file = 0, .offset = i * bytes, .bytes = bytes},
                                   .dtype = "BF16",
                                   .dims = {1, 1}});
  return out;
}

TEST(MingVae, RawContractIndexesAndEvaluatedWeightLookupsArePaid) {
  constexpr std::uint64_t kBindings = 1024;
  auto synthetic = MakeSyntheticContracts(kBindings, true);
  ASSERT_NE(synthetic, nullptr);
  const auto cost = normalization_detail::Bound(synthetic->source, synthetic->mapping);
  ASSERT_TRUE(cost) << cost.error().reason;
  constexpr std::uint64_t kNameBytes = 200;
  const auto tree_comparisons = 4U * (std::bit_width(kBindings) + 1U) * kBindings;
  // Expected bindings, source tensors, and exact-contract identities each
  // use an ordered name index. The funded work includes those logarithmic
  // comparisons, not the old cross-product of both inventories.
  const auto index_work = 3 * tree_comparisons * kNameBytes;
  EXPECT_GE(cost->work_units, index_work);
  RecordProperty("contract_work_1024", std::to_string(cost->work_units));
  auto doubled = MakeSyntheticContracts(2 * kBindings, true);
  ASSERT_NE(doubled, nullptr);
  const auto doubled_cost = normalization_detail::Bound(doubled->source, doubled->mapping);
  ASSERT_TRUE(doubled_cost) << doubled_cost.error().reason;
  RecordProperty("contract_work_2048", std::to_string(doubled_cost->work_units));
  EXPECT_LT(doubled_cost->work_units, 3 * cost->work_units);

  const auto one_short = normalization_detail::Build(
      Checkpoint(synthetic->source), synthetic->mapping,
      {.metadata_bytes = cost->metadata_bytes, .work_units = cost->work_units - 1});
  ASSERT_FALSE(one_short);
  EXPECT_EQ(one_short.error().reason, "normalization reservation is insufficient");
  auto exact = normalization_detail::Build(
      Checkpoint(synthetic->source), synthetic->mapping,
      {.metadata_bytes = cost->metadata_bytes, .work_units = cost->work_units});
  ASSERT_TRUE(exact) << exact.error().reason;
  EXPECT_EQ(exact->cost().work_units, cost->work_units);

  auto no_contracts = MakeSyntheticContracts(16, false);
  ASSERT_NE(no_contracts, nullptr);
  const auto no_contract_cost =
      normalization_detail::Bound(no_contracts->source, no_contracts->mapping);
  ASSERT_TRUE(no_contract_cost) << no_contract_cost.error().reason;
  constexpr std::uint64_t kSmallBindings = 16;
  const auto empty_contract_lookup_floor = kSmallBindings * (kNameBytes + 1);
  EXPECT_GE(no_contract_cost->work_units, empty_contract_lookup_floor);
}

}  // namespace
}  // namespace llmp::import
