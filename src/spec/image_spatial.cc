// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "spec/image_spatial.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <utility>

namespace llmp::spec {
std::expected<ModelSpec, std::string> VaeSpatialSpec(
    std::span<const model::VaeStep> steps, std::span<const model::QwenImageTensor> tensors) {
  static_assert(sizeof(ComponentRef) <= 512);
  using Kind = model::VaeStep::Kind;
  if (steps.empty() || steps.size() > 64 || tensors.size() > 4096)
    return std::unexpected("a VAE spatial span needs a bounded source inventory");
  // Validate the complete source trajectory and borrowed names before copies.
  std::uint64_t width = steps.front().width, height = steps.front().height;
  std::uint32_t channels = steps.front().in_channels;
  std::uint8_t buffer = steps.front().in;
  for (const auto& step : steps) {
    if (step.in >= model::kVaeBuffers || step.out >= model::kVaeBuffers || step.in != buffer ||
        step.in_channels != channels || step.width != width || step.height != height ||
        width == 0 || height == 0 || channels == 0 || channels > 65536 || step.out_channels == 0 ||
        step.out_channels > 65536 || step.factor_t != 1)
      return std::unexpected("a VAE spatial span is not a complete serial source trajectory");
    if (step.kind == Kind::kUpsample) {
      if (step.out_channels != channels || step.weight != -1 || step.bias != -1 ||
          width > UINT32_MAX / 2 || height > UINT32_MAX / 2)
        return std::unexpected("a VAE upsample changes channels, owns weights or overflows");
      width *= 2;
      height *= 2;
    } else if (step.kind == Kind::kConv1x1 || step.kind == Kind::kConv3x3) {
      if (step.weight < 0 || step.bias < 0 ||
          static_cast<std::size_t>(step.weight) >= tensors.size() ||
          static_cast<std::size_t>(step.bias) >= tensors.size())
        return std::unexpected("a VAE convolution is missing its actual source tensors");
      const auto& w = tensors[static_cast<std::size_t>(step.weight)];
      const auto& b = tensors[static_cast<std::size_t>(step.bias)];
      const std::uint64_t k = step.kind == Kind::kConv3x3 ? 3 : 1;
      const std::array<std::uint64_t, 4> shape{step.out_channels, channels, k, k};
      if (w.dtype != "F32" || b.dtype != "F32" || w.shape.size() != 4 ||
          !std::equal(w.shape.begin(), w.shape.end(), shape.begin()) || b.shape.size() != 1 ||
          b.shape[0] != step.out_channels || w.name.empty() || b.name.empty() ||
          w.name.size() > 256 || b.name.size() > 256 || w.name == b.name)
        return std::unexpected("a VAE convolution differs from its F32 source shape or names");
    } else {
      return std::unexpected("this VAE span contains a branch or unsupported component");
    }
    buffer = step.out;
    channels = step.out_channels;
  }
  Stage stage{.name = "vae", .role = StageRole::kVae, .artifact = 0};
  stage.stream.width = steps.front().in_channels;
  Layer layer;
  layer.components.reserve(steps.size());
  for (const auto& step : steps) {
    if (step.kind == Kind::kUpsample) {
      layer.components.push_back(
          {.kind = "upsample.nearest", .version = 1, .params = {{"factor", std::int64_t{2}}}});
    } else {
      const std::int64_t k = step.kind == Kind::kConv3x3 ? 3 : 1;
      layer.components.push_back(
          {.kind = "conv.spatial",
           .version = 1,
           .params = {{"in_channels", std::int64_t{step.in_channels}},
                      {"out_channels", std::int64_t{step.out_channels}},
                      {"kernel", k},
                      {"padding", k / 2}},
           .weights = {
               {.role = "kernel", .tensor = tensors[static_cast<std::size_t>(step.weight)].name},
               {.role = "bias", .tensor = tensors[static_cast<std::size_t>(step.bias)].name}}});
    }
  }
  stage.layers.push_back(std::move(layer));
  ModelSpec spec{.pipeline = PipelineKind::kImage, .stages = {std::move(stage)}};
  Canonicalize(spec);
  if (auto valid = Validate(spec); !valid) return std::unexpected(valid.error());
  return spec;
}

std::expected<SpatialComposition, std::string> VaeDecoderSpec(
    std::span<const model::VaeStep> steps, std::span<const model::QwenImageTensor> tensors) {
  using Kind = model::VaeStep::Kind;
  if (steps.empty() || steps.size() > 256 || tensors.size() > 4096 ||
      steps.back().kind == Kind::kCopy)
    return std::unexpected("a VAE decoder needs a bounded complete source trajectory");
  for (const auto& tensor : tensors)
    if (tensor.name.empty() || tensor.name.size() > 256 || tensor.shape.empty() ||
        tensor.shape.size() > 4 || tensor.dtype != "F32")
      return std::unexpected("a VAE decoder tensor inventory differs from its F32 source");
  struct Value {
    std::uint16_t id = 0;
    std::uint64_t w = 0, h = 0, c = 0;
    bool valid = false;
  };
  std::array<Value, model::kVaeBuffers> current{};
  if (steps.front().in >= current.size()) return std::unexpected("invalid decoder input buffer");
  current[steps.front().in] = {0, steps.front().width, steps.front().height,
                               steps.front().in_channels, true};
  Stage stage{.name = "vae", .role = StageRole::kVae, .artifact = 0};
  stage.stream.width = steps.front().in_channels;
  Layer layer;
  layer.components.reserve(steps.size());
  std::vector<GraphLink> links;
  links.reserve(steps.size());
  const auto shape = [](const Value& v, std::uint64_t w, std::uint64_t h, std::uint64_t c) {
    return v.valid && v.w == w && v.h == h && v.c == c;
  };
  for (const auto& step : steps) {
    if (step.in >= current.size() || step.out >= current.size() || step.aux >= current.size() ||
        !step.width || !step.height || !step.in_channels || !step.out_channels ||
        step.width > 4096 || step.height > 4096 || step.in_channels > 65536 ||
        step.out_channels > 65536 || (step.kind != Kind::kAddDupUp && step.factor_t != 1))
      return std::unexpected("decoder source step dimensions or buffers differ");
    const auto input = current[step.in];
    Value output{0, step.width, step.height, step.out_channels, true};
    GraphLink link{input.id, {}};
    ComponentRef component;
    // DupUp describes the shortcut geometry; attention describes C, while its
    // actual source contains Q,K,V concatenated over3C. All other inputs match.
    if (step.kind != Kind::kAddDupUp &&
        !shape(input, step.width, step.height,
               step.kind == Kind::kAttention ? 3ULL * step.in_channels : step.in_channels))
      return std::unexpected("decoder step reads a missing or incompatible source value");
    if (step.kind == Kind::kCopy) {
      if (step.in_channels != step.out_channels || step.weight != -1 || step.bias != -1)
        return std::unexpected("decoder saved alias changes shape or owns weights");
      current[step.out] = input;
      continue;
    }
    const auto weight = [&](std::int32_t index) -> const model::QwenImageTensor* {
      return index >= 0 && static_cast<std::size_t>(index) < tensors.size()
                 ? &tensors[static_cast<std::size_t>(index)]
                 : nullptr;
    };
    if (step.kind == Kind::kConv1x1 || step.kind == Kind::kConv3x3) {
      const auto* w = weight(step.weight);
      const auto* b = weight(step.bias);
      const std::uint64_t k = step.kind == Kind::kConv3x3 ? 3 : 1;
      const std::array<std::uint64_t, 4> wanted{step.out_channels, step.in_channels, k, k};
      if (!w || !b || w->shape.size() != 4 ||
          !std::equal(w->shape.begin(), w->shape.end(), wanted.begin()) || b->shape.size() != 1 ||
          b->shape[0] != step.out_channels || w->name == b->name)
        return std::unexpected("decoder convolution source parts differ");
      component = {
          .kind = "conv.spatial",
          .version = 1,
          .params = {{"in_channels", std::int64_t{step.in_channels}},
                     {"out_channels", std::int64_t{step.out_channels}},
                     {"kernel", static_cast<std::int64_t>(k)},
                     {"padding", static_cast<std::int64_t>(k / 2)}},
          .weights = {{.role = "kernel", .tensor = w->name}, {.role = "bias", .tensor = b->name}}};
    } else if (step.kind == Kind::kNorm || step.kind == Kind::kNormSilu) {
      const auto* gamma = weight(step.weight);
      if (!gamma || gamma->shape[0] != step.in_channels || step.in_channels != step.out_channels ||
          step.bias != -1 ||
          !std::all_of(gamma->shape.begin() + 1, gamma->shape.end(), [](auto n) { return n == 1; }))
        return std::unexpected("decoder norm gamma shape or source differs");
      component = {.kind = "norm.channel",
                   .version = 1,
                   .params = {{"channels", std::int64_t{step.in_channels}},
                              {"weight_rank", static_cast<std::int64_t>(gamma->shape.size())},
                              {"weight_axis", static_cast<std::int64_t>(gamma->shape.size() - 1)},
                              {"silu", step.kind == Kind::kNormSilu}},
                   .weights = {{.role = "weight", .tensor = gamma->name}}};
    } else {
      if (step.weight != -1 || step.bias != -1)
        return std::unexpected("unweighted decoder operation names a source weight");
      switch (step.kind) {
        case Kind::kUpsample:
          if (step.in_channels != step.out_channels)
            return std::unexpected("decoder upsample changes channels");
          output.w *= 2;
          output.h *= 2;
          component = {
              .kind = "upsample.nearest", .version = 1, .params = {{"factor", std::int64_t{2}}}};
          break;
        case Kind::kAdd:
          if (step.in_channels != step.out_channels ||
              !shape(current[step.aux], output.w, output.h, output.c))
            return std::unexpected("decoder residual source geometry differs");
          link.auxiliary = current[step.aux].id;
          component = {.kind = "residual.spatial", .version = 1};
          break;
        case Kind::kAddDupUp:
          output.w *= 2;
          output.h *= 2;
          if (!shape(input, output.w, output.h, output.c) ||
              !shape(current[step.aux], step.width, step.height, step.in_channels) ||
              (step.factor_t != 1 && step.factor_t != 2) ||
              (output.c * step.factor_t * 4) % step.in_channels)
            return std::unexpected("decoder duplicate-up shortcut geometry differs");
          link.auxiliary = current[step.aux].id;
          component = {.kind = "residual.upsample",
                       .version = 1,
                       .params = {{"factor_t", std::int64_t{step.factor_t}}}};
          break;
        case Kind::kAttention:
          if (step.in_channels != step.out_channels)
            return std::unexpected("decoder attention changes its head width");
          component = {.kind = "attention.spatial", .version = 1};
          break;
        default:
          return std::unexpected("unsupported decoder source operation");
      }
    }
    layer.components.push_back(std::move(component));
    links.push_back(link);
    output.id = static_cast<std::uint16_t>(layer.components.size());
    current[step.out] = output;
  }
  if (layer.components.empty()) return std::unexpected("decoder has no computed output");
  stage.layers.push_back(std::move(layer));
  ModelSpec spec{.pipeline = PipelineKind::kImage, .stages = {std::move(stage)}};
  Canonicalize(spec);
  if (auto valid = Validate(spec); !valid) return std::unexpected(valid.error());
  return SpatialComposition{std::move(spec), std::move(links)};
}
}  // namespace llmp::spec
