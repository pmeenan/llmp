// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "import/qwen_image.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "components/step.h"
#include "format/layout.h"
#include "format/tensor_format.h"
#include "ir/graph.h"
#include "model/qwen_image.h"
#include "spec/image_spatial.h"

namespace llmp::import {
namespace {

std::unexpected<Error> Bad(std::string_view reason, Rule rule = Rule::kConfig) {
  return std::unexpected(Error{.rule = rule, .reason = reason});
}

const Checkpoint* FindComponent(const Checkpoint& source, std::string_view role) {
  for (const auto& comp : source.components) {
    if (comp.role == role) return &comp;
  }
  return nullptr;
}

std::expected<ComponentPlan, Error> PlanComponent(
    const Checkpoint& comp, std::string_view expected_arch,
    std::span<const model::QwenImageTensor> expected_tensors, const spec::ModelSpec& spec) {
  if (comp.container != "safetensors") {
    return Bad("component container must be safetensors", Rule::kContainer);
  }
  if (comp.architecture != expected_arch) {
    if (!(expected_arch == "AutoencoderKLQwenImage21" &&
          comp.architecture == "AutoencoderKLQwenImage")) {
      return Bad("component architecture differs", Rule::kConfig);
    }
  }

  spec::Source canonical{.architecture = comp.architecture, .container = comp.container};
  canonical.tensors.reserve(expected_tensors.size());
  std::vector<CopyRoute> routes;
  routes.reserve(expected_tensors.size());
  std::uint64_t payload = 0;

  for (const auto& need : expected_tensors) {
    const Tensor* found = comp.FindTensor(need.name);
    if (!found) {
      return Bad("component lacks required tensor: " + need.name, Rule::kConfig);
    }
    if (found->dtype != need.dtype || found->dims != need.shape) {
      return Bad("component tensor format or shape differs: " + need.name, Rule::kConfig);
    }

    format::Shape shape;
    shape.rank = static_cast<std::uint8_t>(need.shape.size());
    for (std::size_t d = 0; d < need.shape.size(); ++d) {
      shape.ne[d] = need.shape[need.shape.size() - 1 - d];
    }
    const auto encoding = need.dtype == "BF16" ? format::Encoding::kBF16 : format::Encoding::kF32;
    canonical.tensors.push_back({
        .name = need.name,
        .format = ir::Dense(encoding, shape),
        .part = format::PartRole::kData,
        .experts = false,
    });

    const auto si = static_cast<std::uint32_t>(found - comp.tensors.data());
    routes.push_back({
        .source_tensor = si,
        .extent = found->extent,
        .readable = 0,
    });
    payload += found->extent.bytes;
  }

  if (comp.tensors.size() != expected_tensors.size()) {
    return Bad("component has extraneous tensors", Rule::kConfig);
  }

  if (auto valid = components::ValidateSpec(spec, canonical.tensors); !valid) {
    return Bad("component spec validation failed: " + valid.error(), Rule::kConfig);
  }

  return ComponentPlan{
      .checkpoint = comp,
      .canonical = std::move(canonical),
      .spec = spec,
      .routes = std::move(routes),
      .payload_bytes = payload,
  };
}

}  // namespace

std::expected<NormalizationCost, Error> QwenImagePipelineBound(const Checkpoint& source) {
  if (source.container != "diffusers" || source.architecture != "QwenImage21Pipeline") {
    return Bad("not a QwenImage21Pipeline diffusers checkpoint", Rule::kContainer);
  }
  std::uint64_t total_payload = 0;
  std::uint64_t total_tensors = 0;
  for (const auto& comp : source.components) {
    for (const auto& t : comp.tensors) {
      total_payload += t.extent.bytes;
      ++total_tensors;
    }
  }
  return NormalizationCost{
      .metadata_bytes = std::uint64_t{512} << 20U,
      .work_units = total_tensors,
      .payload_bytes = total_payload,
      .layers = 36 + 32,
  };
}

std::expected<QwenImagePipelinePlan, Error> PlanQwenImagePipeline(const Checkpoint& source) {
  if (source.container != "diffusers" || source.architecture != "QwenImage21Pipeline") {
    return Bad("not a QwenImage21Pipeline diffusers checkpoint", Rule::kContainer);
  }

  const auto* text_comp = FindComponent(source, "text_encoder");
  const auto* denoiser_comp = FindComponent(source, "transformer");
  const auto* vae_comp = FindComponent(source, "vae");
  if (!text_comp || !denoiser_comp || !vae_comp) {
    return Bad("missing required Qwen-Image-2.1 component (text_encoder, transformer, or vae)",
               Rule::kConfig);
  }

  const auto& p = model::QwenImage21();

  const auto text_tensors = model::TextEncoderTensors(p.text);
  auto text_spec = spec::ImageTextSpec(p.text, 1, 0, text_tensors);
  if (!text_spec) return Bad("cannot build text encoder spec: " + text_spec.error(), Rule::kConfig);

  const auto denoiser_tensors = model::DenoiserTensors(p.denoiser);
  auto denoiser_spec = spec::ImageFlowSpec(p.denoiser, 1, 1, denoiser_tensors);
  if (!denoiser_spec) {
    return Bad("cannot build denoiser spec: " + denoiser_spec.error(), Rule::kConfig);
  }

  const auto vae_tensors = model::VaeDecoderTensors(p.vae);
  auto vae_spec = spec::ImageDecodeSpec(p.vae, 64, 64, vae_tensors);
  if (!vae_spec) return Bad("cannot build vae spec: " + vae_spec.error(), Rule::kConfig);

  auto text_plan = PlanComponent(*text_comp, p.text_architecture, text_tensors, text_spec->spec);
  if (!text_plan) return std::unexpected(text_plan.error());

  auto denoiser_plan =
      PlanComponent(*denoiser_comp, p.denoiser_architecture, denoiser_tensors, denoiser_spec->spec);
  if (!denoiser_plan) return std::unexpected(denoiser_plan.error());

  auto vae_plan = PlanComponent(*vae_comp, p.vae_architecture, vae_tensors, vae_spec->spec);
  if (!vae_plan) return std::unexpected(vae_plan.error());

  const std::uint64_t total_payload =
      text_plan->payload_bytes + denoiser_plan->payload_bytes + vae_plan->payload_bytes;

  return QwenImagePipelinePlan{
      .components = {std::move(*text_plan), std::move(*denoiser_plan), std::move(*vae_plan)},
      .payload_bytes = total_payload,
  };
}

}  // namespace llmp::import
