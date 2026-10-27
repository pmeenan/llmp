// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Qwen-Image's stage graphs for tests, from its mappings' production recipes
// (spec/mappings/qwen_image.h, qwen_image_vae.h) expanded through the import
// route (import/stage_recipe.h ExpandStageRecipe) over a family spec: the
// stages the importer stores, bound at a request's geometry as serving binds
// them (spec::BindStageScalars), and selections of their components as
// stages of their own. No test carries a second implementation of the
// topology; the published digests the expansions keep are pinned by
// image_family_test (RecipesExpandToTheApprovedComponentSpecs).

#ifndef LLMP_TESTS_SUPPORT_IMAGE_RECIPES_H_
#define LLMP_TESTS_SUPPORT_IMAGE_RECIPES_H_

#include <algorithm>
#include <array>
#include <cstdint>
#include <expected>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "import/checkpoint.h"
#include "import/stage_recipe.h"
#include "model/image_pipeline.h"
#include "spec/image_family.h"
#include "spec/image_spatial.h"
#include "spec/mappings/qwen_image.h"
#include "spec/spec.h"

namespace llmp::test_support {

// The family spec a pipeline with `p`'s values synthesizes to (its kinds and
// their schema's parameters, spec/image_family.h): what the recipes read.
inline spec::ModelSpec QwenImageFamilyOf(const model::ImageProfile& p) {
  const auto ints = [](const auto& list) {
    std::vector<std::int64_t> out;
    for (const auto v : list) out.push_back(static_cast<std::int64_t>(v));
    return out;
  };
  const auto reals = [](const auto& list) {
    std::vector<double> out;
    for (const auto v : list) out.push_back(static_cast<double>(v));
    return out;
  };
  const auto& t = p.text;
  const auto& d = p.denoiser;
  const auto& s = p.scheduler;
  const auto& v = p.vae;
  spec::ComponentRef text{.kind = std::string(spec::kImageTextKind),
                          .params = {{"ffn", std::int64_t{t.ffn}},
                                     {"head_dim", std::int64_t{t.head_dim}},
                                     {"heads", std::int64_t{t.heads}},
                                     {"kv_heads", std::int64_t{t.kv_heads}},
                                     {"layers", std::int64_t{t.layers}},
                                     {"rms_eps", double{t.rms_eps}},
                                     {"rope_theta", double{t.rope_theta}},
                                     {"vocab", std::int64_t{t.vocab}},
                                     {"width", std::int64_t{t.width}}}};
  spec::ComponentRef flow{
      .kind = std::string(spec::kImageFlowKind),
      .params = {
          {"blocks", std::int64_t{d.blocks}},
          {"context", std::int64_t{d.context}},
          {"eps", double{d.eps}},
          {"head_dim", std::int64_t{d.head_dim}},
          {"heads", std::int64_t{d.heads}},
          {"in_channels", std::int64_t{d.in_channels}},
          {"mlp_ratio", std::int64_t{d.width != 0 && d.mlp >= d.width ? d.mlp / d.width : 1}},
          {"out_channels", std::int64_t{d.out_channels}},
          {"rope_axes", ints(d.rope_axes)},
          {"rope_theta", double{d.rope_theta}},
          {"timestep_dim", std::int64_t{d.timestep_dim}},
          {"timestep_divisor", double{d.timestep_divisor}},
          {"timestep_period", double{d.timestep_period}},
          {"timestep_scale", double{d.timestep_scale}}}};
  spec::ComponentRef schedule{.kind = std::string(spec::kImageScheduleKind),
                              .params = {{"base_image_seq_len", std::int64_t{s.base_image_seq_len}},
                                         {"base_shift", s.base_shift},
                                         {"max_image_seq_len", std::int64_t{s.max_image_seq_len}},
                                         {"max_shift", s.max_shift},
                                         {"shift_terminal", s.shift_terminal},
                                         {"train_timesteps", std::int64_t{s.train_timesteps}}}};
  spec::ComponentRef vae{.kind = std::string(spec::kImageVaeKind),
                         .params = {{"base", std::int64_t{v.base}},
                                    {"dim_mult", ints(v.dim_mult)},
                                    {"latents_mean", reals(v.latents_mean)},
                                    {"latents_std", reals(v.latents_std)},
                                    {"out_channels", std::int64_t{v.out_channels}},
                                    {"res_blocks", std::int64_t{v.res_blocks}},
                                    {"spatial_factor", std::int64_t{v.spatial_factor}},
                                    {"temporal_upsample", ints(v.temporal_upsample)},
                                    {"z_dim", std::int64_t{v.z_dim}}}};
  spec::ModelSpec family{.pipeline = spec::PipelineKind::kImage};
  family.stages.push_back({.name = "text_encoder",
                           .role = spec::StageRole::kTextEncoder,
                           .artifact = 0,
                           .layers = {{.components = {std::move(text)}}}});
  family.stages.push_back({.name = "denoiser",
                           .role = spec::StageRole::kDenoiser,
                           .artifact = 1,
                           .layers = {{.components = {std::move(flow), std::move(schedule)}}}});
  family.stages.push_back({.name = "vae",
                           .role = spec::StageRole::kVae,
                           .artifact = 2,
                           .layers = {{.components = {std::move(vae)}}}});
  return family;
}

// What the VAE recipe reads of its source besides the family: Qwen-Image-
// 2.1's config (a residual decoder) and its 2-D kernels' rank.
struct QwenImageVaeSource {
  std::vector<import::MetaEntry> config{};
  std::vector<import::Tensor> tensors{};
};
inline QwenImageVaeSource QwenImageVaeSourceOf(const model::ImageVaeProfile& v) {
  const std::uint64_t mid = std::uint64_t{v.base} * (v.dim_mult.empty() ? 0 : v.dim_mult.back());
  return {.config = {{.key = "is_residual", .value = true}},
          .tensors = {{.name = "decoder.conv_in.weight", .dims = {mid, v.z_dim, 3, 3}}}};
}

// A component's stage ("transformer" or "vae"), as the importer stores it.
inline std::expected<spec::RecipeExpansion, std::string> QwenImageStage(
    const spec::ModelSpec& family, const model::ImageProfile& profile, std::string_view role) {
  const auto& mapping = spec::mappings::QwenImage21Mapping();
  const auto rule = std::ranges::find(mapping.components, role, &spec::ImageComponentRule::role);
  if (rule == mapping.components.end()) return std::unexpected("no such Qwen-Image component");
  const auto* recipe = spec::ImportOf(*rule).recipe;
  if (recipe == nullptr) return std::unexpected("the component has no stage recipe");
  const auto source = role == "vae" ? QwenImageVaeSourceOf(profile.vae) : QwenImageVaeSource{};
  auto expanded = import::ExpandStageRecipe(*recipe, family, std::uint64_t{512} << 20U,
                                            source.config, source.tensors, "safetensors");
  // A reduced test geometry the family cannot state (a hidden width that is
  // no multiple of the stream's): the recipe's graph, its MLPs that wide.
  const auto& d = profile.denoiser;
  if (expanded && role == "transformer" && d.width != 0 && d.mlp % d.width != 0)
    for (auto& component : expanded->stage.layers[0].components)
      if (component.kind == "ffn.glu")
        for (auto& param : component.params)
          if (param.name == "hidden") param.value = std::int64_t{d.mlp};
  return expanded;
}

// A stage spec bound at a request's geometry, as serving binds it: its
// composition (links and exports), the stage's own.
inline std::expected<spec::SpatialComposition, std::string> BoundStage(
    spec::RecipeExpansion expanded, std::uint32_t width, std::uint32_t height,
    std::uint32_t text_rows, std::uint32_t rows) {
  spec::ModelSpec stored{.pipeline = spec::PipelineKind::kImage,
                         .stages = {std::move(expanded.stage)}};
  spec::Canonicalize(stored);
  const std::array<spec::StageScalar, 5> scalars{{{"joint_rows", std::int64_t{rows} + text_rows},
                                                  {"latent_height", height},
                                                  {"latent_rows", rows},
                                                  {"latent_width", width},
                                                  {"text_rows", text_rows}}};
  auto bound = spec::BindStageScalars(stored, scalars);
  if (!bound) return std::unexpected(bound.error());
  auto links = bound->stages[0].links;
  auto exports = bound->stages[0].exports;
  return spec::SpatialComposition{
      .spec = std::move(*bound), .links = std::move(links), .exports = std::move(exports)};
}

// The flow transformer's stage for `text_rows` prompt rows and `image_rows`
// latent rows.
inline std::expected<spec::SpatialComposition, std::string> QwenImageFlowGraph(
    const model::ImageProfile& profile, std::uint32_t text_rows, std::uint32_t image_rows) {
  auto expanded = QwenImageStage(QwenImageFamilyOf(profile), profile, "transformer");
  if (!expanded) return std::unexpected(expanded.error());
  return BoundStage(std::move(*expanded), 1, 1, text_rows, image_rows);
}

// The VAE decoder's stage for a latent grid.
inline std::expected<spec::SpatialComposition, std::string> QwenImageVaeGraph(
    const model::ImageProfile& profile, std::uint32_t height, std::uint32_t width) {
  auto expanded = QwenImageStage(QwenImageFamilyOf(profile), profile, "vae");
  if (!expanded) return std::unexpected(expanded.error());
  return BoundStage(std::move(*expanded), width, height, 1, width * height);
}

// A selection of a stage's components as a stage of its own: `values` the
// components kept (value IDs, ascending); a link to `input` (an original
// value, or with `input_port` an original host port) reads the new stage's
// input, a host port is renumbered by `ports` (original, new), and `exports`
// names the outputs by their original value IDs. A link to anything else
// outside the selection is refused.
struct StageSelection {
  std::vector<std::uint16_t> values{};
  std::uint16_t input = 0;
  std::optional<std::uint16_t> input_port{};
  std::vector<std::pair<std::uint16_t, std::uint16_t>> ports{};
  std::vector<std::uint16_t> exports{};
};
inline std::expected<spec::SpatialComposition, std::string> SelectStage(
    const spec::SpatialComposition& whole, const StageSelection& selection) {
  const auto& stage = whole.spec.stages.at(0);
  const auto& components = stage.layers.at(0).components;
  spec::Stage out{.name = stage.name, .role = stage.role, .artifact = stage.artifact};
  out.stream = stage.stream;
  out.layers.emplace_back();
  std::vector<spec::GraphLink> links;
  const auto value_of = [&](std::uint16_t original) -> std::optional<std::uint16_t> {
    if (original == selection.input && !selection.input_port) return std::uint16_t{0};
    const auto at = std::ranges::find(selection.values, original);
    if (at == selection.values.end()) return std::nullopt;
    return static_cast<std::uint16_t>(at - selection.values.begin() + 1);
  };
  const auto port_of = [&](std::uint16_t original) -> std::optional<std::uint16_t> {
    for (const auto& [from, to] : selection.ports)
      if (from == original) return to;
    return std::nullopt;
  };
  for (const auto value : selection.values) {
    if (value == 0 || value > components.size()) return std::unexpected("selection outside stage");
    out.layers[0].components.push_back(components[value - 1]);
    const auto& link = stage.links.at(value - 1);
    spec::GraphLink mapped;
    if (link.input_port && selection.input_port && *link.input_port == *selection.input_port) {
      mapped.input = 0;
    } else if (link.input_port) {
      const auto port = port_of(*link.input_port);
      if (!port) return std::unexpected("selection reads an unmapped host port");
      mapped.input = 0;
      mapped.input_port = *port;
    } else {
      const auto input = value_of(link.input);
      if (!input) return std::unexpected("selection reads a value outside it");
      mapped.input = *input;
    }
    const auto side =
        [&](const std::optional<std::uint16_t>& id, const std::optional<std::uint16_t>& port,
            std::optional<std::uint16_t>& to_id, std::optional<std::uint16_t>& to_port) -> bool {
      if (port) {
        to_port = port_of(*port);
        return to_port.has_value();
      }
      if (id) {
        to_id = value_of(*id);
        return to_id.has_value();
      }
      return true;
    };
    if (!side(link.auxiliary, link.auxiliary_port, mapped.auxiliary, mapped.auxiliary_port) ||
        !side(link.carry, link.carry_port, mapped.carry, mapped.carry_port))
      return std::unexpected("selection reads a side value outside it");
    links.push_back(mapped);
  }
  std::vector<std::uint16_t> exports;
  for (const auto value : selection.exports) {
    const auto mapped = value_of(value);
    if (!mapped || *mapped == 0) return std::unexpected("selection exports a value outside it");
    exports.push_back(*mapped);
  }
  out.links = links;
  out.exports = exports;
  spec::ModelSpec spec{.pipeline = spec::PipelineKind::kImage, .stages = {std::move(out)}};
  spec::Canonicalize(spec);
  if (auto valid = spec::Validate(spec); !valid) return std::unexpected(valid.error());
  return spec::SpatialComposition{
      .spec = std::move(spec), .links = std::move(links), .exports = std::move(exports)};
}

// The value ID (one-based) of a stage's first component of `kind` after
// `after` whose weights bind `tensor` (empty: any).
inline std::optional<std::uint16_t> FindComponent(const spec::SpatialComposition& whole,
                                                  std::string_view kind,
                                                  std::string_view tensor = {},
                                                  std::uint16_t after = 0) {
  const auto& components = whole.spec.stages.at(0).layers.at(0).components;
  for (std::size_t i = after; i < components.size(); ++i) {
    if (components[i].kind != kind) continue;
    if (!tensor.empty() &&
        std::ranges::none_of(components[i].weights,
                             [&](const spec::WeightBinding& w) { return w.tensor == tensor; }))
      continue;
    return static_cast<std::uint16_t>(i + 1);
  }
  return std::nullopt;
}

// Parts of those stages, as stages of their own:
//
// The VAE's decoder (after its latent reshape and affine): its input the
// latent grid.
inline std::expected<spec::SpatialComposition, std::string> QwenImageDecoderGraph(
    const model::ImageProfile& profile, std::uint32_t height, std::uint32_t width) {
  auto whole = QwenImageVaeGraph(profile, height, width);
  if (!whole) return std::unexpected(whole.error());
  const auto count = whole->spec.stages[0].layers[0].components.size();
  StageSelection selection{.input = 2};
  for (std::size_t v = 3; v <= count; ++v)
    selection.values.push_back(static_cast<std::uint16_t>(v));
  return SelectStage(*whole, selection);
}

// The flow's timestep projections: its input the sinusoid; exports the
// modulation and the final normalization's scale.
inline std::expected<spec::SpatialComposition, std::string> QwenImageTimestepGraph(
    model::ImageProfile profile) {
  profile.denoiser.blocks = 0;  // the projections alone: no block expanded
  auto whole = QwenImageFlowGraph(profile, 1, 1);
  if (!whole) return std::unexpected(whole.error());
  return SelectStage(*whole,
                     {.values = {1, 2, 3, 4}, .input_port = 1, .ports = {}, .exports = {3, 4}});
}

// The flow's conditioner and its first block over a joint stream of
// `text_rows` prompt rows: its input the projected joint stream; ports 0 the
// sinusoid, 1 the positions, 2 the mask; exports the block's stream and the
// final normalization's scale.
inline std::expected<spec::SpatialComposition, std::string> QwenImageBlockGraph(
    const model::ImageProfile& profile, std::uint32_t text_rows) {
  auto whole = QwenImageFlowGraph(profile, text_rows, 1);
  if (!whole) return std::unexpected(whole.error());
  const auto joint = FindComponent(*whole, "concat.spatial");
  if (!joint) return std::unexpected("the flow has no joint stream");
  StageSelection selection{.input = *joint, .ports = {{1, 0}, {2, 1}, {3, 2}}};
  for (std::uint16_t v = 1; v <= 4; ++v) selection.values.push_back(v);
  for (std::uint16_t v = 1; v <= 4; ++v) selection.values.push_back(*joint + v);
  for (std::uint16_t v = 5; v <= 10; ++v) selection.values.push_back(*joint + v);
  selection.exports = {static_cast<std::uint16_t>(*joint + 10), 4};
  return SelectStage(*whole, selection);
}

}  // namespace llmp::test_support

#endif  // LLMP_TESTS_SUPPORT_IMAGE_RECIPES_H_
