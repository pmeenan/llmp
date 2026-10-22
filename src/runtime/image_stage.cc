// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#include "runtime/image_stage.h"

#include <algorithm>

namespace llmp::runtime {
std::array<spec::StageScalar, 5> ImageStageDimensions::Scalars() const {
  return {{{"joint_rows", std::int64_t{rows} + text_rows},
           {"latent_height", height},
           {"latent_rows", rows},
           {"latent_width", width},
           {"text_rows", text_rows}}};
}
std::expected<ImageStageDimensions, std::string> ImageStageDimensionsOf(
    const model::ImageProfile& profile, ImageStageGeometry geometry) {
  const auto factor = profile.vae.spatial_factor;
  if (!geometry.width || !geometry.height || !factor || geometry.width % factor ||
      geometry.height % factor || !geometry.prompt_rows || geometry.prompt_rows > INT32_MAX ||
      geometry.drop >= geometry.prompt_rows)
    return std::unexpected("image stage geometry or prompt/drop is invalid");
  ImageStageDimensions result{geometry.width / factor, geometry.height / factor, 0,
                              geometry.prompt_rows - geometry.drop};
  const auto rows = std::uint64_t{result.width} * result.height;
  if (rows > UINT32_MAX - result.text_rows || result.width > INT32_MAX || result.height > INT32_MAX)
    return std::unexpected("image stage geometry exceeds row/position representation");
  result.rows = static_cast<std::uint32_t>(rows);
  return result;
}

std::expected<spec::ModelSpec, std::string> BindImageStage(const spec::ModelSpec& stored,
                                                           const model::ImageProfile& profile,
                                                           ImageStageGeometry geometry) {
  if (stored.pipeline != spec::PipelineKind::kImage || stored.stages.size() != 1 ||
      (stored.stages[0].role != spec::StageRole::kDenoiser &&
       stored.stages[0].role != spec::StageRole::kVae))
    return std::unexpected("stored image topology must describe one denoiser or VAE stage");
  auto checked = ImageStageDimensionsOf(profile, geometry);
  if (!checked) return std::unexpected(checked.error());
  const auto scalars = checked->Scalars();
  return spec::BindStageScalars(stored, scalars);
}

std::expected<void, std::string> ValidateImageStageOutputs(
    spec::StageRole role, std::span<const components::SpatialOwner> outputs,
    const model::ImageProfile& profile, ImageStageGeometry geometry) {
  auto dimensions = ImageStageDimensionsOf(profile, geometry);
  if (!dimensions) return std::unexpected(dimensions.error());
  if (role != spec::StageRole::kDenoiser && role != spec::StageRole::kVae)
    return std::unexpected("image stage output contract requires a denoiser or VAE");
  const auto expected =
      role == spec::StageRole::kDenoiser
          ? components::SpatialShape{1, dimensions->rows, profile.vae.z_dim, 1}
          : components::SpatialShape{geometry.width, geometry.height, profile.vae.out_channels, 1};
  const auto ports = role == spec::StageRole::kDenoiser ? 2U : 1U;
  if (outputs.size() != ports || std::ranges::any_of(outputs, [&](const auto& output) {
        return output.owner != 0 || output.shape != expected;
      }))
    return std::unexpected(
        "stored image topology output differs from the requested stage geometry");
  return {};
}

}  // namespace llmp::runtime
