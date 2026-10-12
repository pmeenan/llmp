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
}  // namespace llmp::spec
