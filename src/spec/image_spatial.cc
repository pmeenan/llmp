// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "spec/image_spatial.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <ranges>
#include <string_view>
#include <utility>

namespace llmp::spec {
namespace {
// The text encoder's tensors by structural role, in its list's order
// (model/image_pipeline.h TextEncoderTensors): the names are its mapping's,
// never matched here.
constexpr std::size_t kTextTensors = static_cast<std::size_t>(model::TextTensor::kCount);
const model::ImageTensor& TextOf(std::span<const model::ImageTensor> tensors, std::uint32_t layer,
                                 model::TextTensor role) {
  return tensors[static_cast<std::size_t>(model::TextGlobal::kCount) +
                 (std::size_t{layer} * kTextTensors) + static_cast<std::size_t>(role)];
}
const model::ImageTensor& TextGlobalOf(std::span<const model::ImageTensor> tensors,
                                       model::TextGlobal role) {
  return tensors[static_cast<std::size_t>(role)];
}
}  // namespace

std::expected<std::uint64_t, std::string> ImageTextSpecBound(
    const model::ImageTextProfile& p, std::span<const model::ImageTensor> tensors) {
  const auto components = std::uint64_t{p.layers} * 2 + 2;
  if (!p.layers || components > kMaxComponents || tensors.size() > 4096 ||
      std::uint64_t{p.layers} * 11 + 1 > tensors.size())
    return std::unexpected("text mapping exceeds component/source representation");
  for (const auto& tensor : tensors)
    if (tensor.name.empty() || tensor.name.size() > 256 || tensor.shape.empty() ||
        tensor.shape.size() > 4)
      return std::unexpected("text mapping source descriptor is unbounded");
  return 4096 * components + 1024 * tensors.size() + 65536;
}
std::expected<SpatialComposition, std::string> ImageTextSpec(
    const model::ImageTextProfile& p, std::uint32_t rows, std::uint32_t drop,
    std::span<const model::ImageTensor> tensors) {
  auto bound = ImageTextSpecBound(p, tensors);
  if (!bound) return std::unexpected(bound.error());
  if (!rows || drop >= rows || !p.width || p.width > 65536 || !p.heads || !p.kv_heads ||
      p.heads % p.kv_heads || !p.head_dim || p.head_dim > 512 || p.head_dim % 2 ||
      std::uint64_t{p.heads} * p.head_dim != p.width || !p.ffn || p.ffn > 65536 || !p.vocab ||
      !std::isfinite(p.rms_eps) || p.rms_eps <= 0 || p.rms_eps > 1 ||
      !std::isfinite(p.rope_theta) || p.rope_theta < 1 || p.rope_theta > 1e10)
    return std::unexpected("text source profile or prompt geometry differs");
  // Every required source is validated before any generated component
  // copies: the list's roles in order, their encodings and shapes.
  const auto required = model::TextEncoderTensors(p, {});
  if (required.size() != std::uint64_t{p.layers} * kTextTensors + 1 ||
      tensors.size() != required.size())
    return std::unexpected("text source role inventory differs");
  for (std::size_t i = 0; i < required.size(); ++i)
    if (tensors[i].dtype != "BF16" || tensors[i].shape != required[i].shape)
      return std::unexpected("text source role encoding or geometry differs");
  Stage stage{.name = "text", .role = StageRole::kTextEncoder, .artifact = 0};
  stage.stream.width = p.width;
  stage.layers.emplace_back();
  SpatialComposition out{.spec = {.pipeline = PipelineKind::kImage, .stages = {std::move(stage)}},
                         .links = {},
                         .exports = {}};
  auto& components = out.spec.stages[0].layers[0].components;
  const auto add = [&](ComponentRef component, GraphLink link) {
    components.push_back(std::move(component));
    out.links.push_back(link);
    return static_cast<std::uint16_t>(components.size());
  };
  auto current =
      add({.kind = "embedding.token",
           .version = 1,
           .weights = {{.role = "table",
                        .tensor = TextGlobalOf(tensors, model::TextGlobal::kEmbedding).name}}},
          {0, {}});
  for (std::uint32_t layer = 0; layer < p.layers; ++layer) {
    using enum model::TextTensor;
    const auto name = [&](model::TextTensor role) { return TextOf(tensors, layer, role).name; };
    current = add({.kind = "attention.dense",
                   .version = 1,
                   .params = {{"eps", double{p.rms_eps}},
                              {"heads", std::int64_t{p.heads}},
                              {"key_dim", std::int64_t{p.head_dim}},
                              {"kv_heads", std::int64_t{p.kv_heads}},
                              {"placement", std::string("pre")},
                              {"qk_norm", true},
                              {"rope_base", double{p.rope_theta}},
                              {"rope_dims", std::int64_t{p.head_dim}},
                              {"rope_scale", 1.0},
                              {"rotation", std::string("neox")},
                              {"scale", 1.0 / std::sqrt(static_cast<double>(p.head_dim))},
                              {"scale_scores", true},
                              {"stateless", true},
                              {"value_dim", std::int64_t{p.head_dim}},
                              {"window", std::int64_t{0}}},
                   .weights = {{.role = "k", .tensor = name(kK)},
                               {.role = "k_norm", .tensor = name(kKNorm)},
                               {.role = "norm", .tensor = name(kInputNorm)},
                               {.role = "output", .tensor = name(kO)},
                               {.role = "q", .tensor = name(kQ)},
                               {.role = "q_norm", .tensor = name(kQNorm)},
                               {.role = "v", .tensor = name(kV)}}},
                  {.input = current, .auxiliary_port = 0, .carry_port = 1});
    current = add({.kind = "ffn.glu",
                   .version = 1,
                   .params = {{"activation", std::string("silu")},
                              {"eps", double{p.rms_eps}},
                              {"hidden", std::int64_t{p.ffn}},
                              {"placement", std::string("pre")}},
                   .weights = {{.role = "down", .tensor = name(kDown)},
                               {.role = "gate", .tensor = name(kGate)},
                               {.role = "norm", .tensor = name(kPostNorm)},
                               {.role = "up", .tensor = name(kUp)}}},
                  {current, {}});
  }
  current = add({.kind = "slice.spatial",
                 .version = 1,
                 .params = {{"start", std::int64_t{drop}},
                            {"rows", std::int64_t{rows - drop}},
                            {"input_rows", std::int64_t{rows}}}},
                {current, {}});
  out.exports = {current};
  out.spec.stages[0].links = out.links;
  out.spec.stages[0].exports = out.exports;
  Canonicalize(out.spec);
  if (auto checked = Validate(out.spec); !checked) return std::unexpected(checked.error());
  return out;
}

}  // namespace llmp::spec
