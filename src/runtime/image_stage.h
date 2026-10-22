// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#ifndef LLMP_RUNTIME_IMAGE_STAGE_H_
#define LLMP_RUNTIME_IMAGE_STAGE_H_

#include <array>
#include <cstdint>
#include <expected>
#include <span>
#include <string>

#include "components/spatial.h"
#include "model/image_pipeline.h"
#include "spec/spec.h"

namespace llmp::runtime {
struct ImageStageGeometry {
  std::uint32_t width = 0, height = 0, prompt_rows = 0, drop = 0;
};
struct ImageStageDimensions {
  std::uint32_t width = 0, height = 0, rows = 0, text_rows = 0;
  std::array<spec::StageScalar, 5> Scalars() const;
};
// Geometry facts derive from the stored profile and request, before allocation.
std::expected<ImageStageDimensions, std::string> ImageStageDimensionsOf(
    const model::ImageProfile& profile, ImageStageGeometry geometry);
// Instantiate an already-paid stored graph. No topology is regenerated;
// fixed graphs are valid too, subject to compiler/request shape checks.
std::expected<spec::ModelSpec, std::string> BindImageStage(const spec::ModelSpec& stored,
                                                           const model::ImageProfile& profile,
                                                           ImageStageGeometry geometry);
// Check actual inferred ports, including same-area rectangular reshapes and
// stored topology whose upsampling differs from its family profile.
std::expected<void, std::string> ValidateImageStageOutputs(
    spec::StageRole role, std::span<const components::SpatialOwner> outputs,
    const model::ImageProfile& profile, ImageStageGeometry geometry);
}  // namespace llmp::runtime
#endif  // LLMP_RUNTIME_IMAGE_STAGE_H_
