// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#ifndef LLMP_SPEC_IMAGE_SPATIAL_H_
#define LLMP_SPEC_IMAGE_SPATIAL_H_

#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "model/image_pipeline.h"
#include "spec/graph_links.h"
#include "spec/spec.h"

namespace llmp::spec {
// The Qwen-Image text encoder's stage (its pipeline's one component without
// a stage recipe yet: spec/image_family.h ImageComponentRule::recipe), built
// from its family profile. Every other image stage is its mapping's recipe
// (spec/stage_recipe.h), expanded at import.
struct SpatialComposition {
  ModelSpec spec;
  std::vector<GraphLink> links;
  std::vector<std::uint16_t> exports;
};
// Source text encoder as a stateless typed graph: primary I32 IDs, extra
// I32 positions and explicit causal F32 mask. Exports pre-final-norm features
// after the source prompt prefix; no generation head or token cache is implied.
std::expected<std::uint64_t, std::string> ImageTextSpecBound(
    const model::ImageTextProfile& profile, std::span<const model::ImageTensor> tensors);
std::expected<SpatialComposition, std::string> ImageTextSpec(
    const model::ImageTextProfile& profile, std::uint32_t rows, std::uint32_t drop,
    std::span<const model::ImageTensor> tensors);
}  // namespace llmp::spec
#endif  // LLMP_SPEC_IMAGE_SPATIAL_H_
