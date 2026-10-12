// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#ifndef LLMP_SPEC_IMAGE_SPATIAL_H_
#define LLMP_SPEC_IMAGE_SPATIAL_H_

#include <cstdint>
#include <expected>
#include <span>
#include <string>

#include "model/qwen_image.h"
#include "spec/spec.h"

namespace llmp::spec {
// Mapping only, over an already-owned, paid VaeDecoderPlan/tensor inventory.
// Consumes a closed serial conv/nearest span, preserving source tensor names
// and F32 shapes. Graph inference owns the actual spatial input geometry;
// this mapping neither dispatches work nor imposes legacy intermediate casts.
// Caller prepays at least this bound for the returned spec and temporary names.
inline constexpr std::uint64_t kVaeSpatialSpecBytes = 1ULL << 20U;
std::expected<ModelSpec, std::string> VaeSpatialSpec(
    std::span<const model::VaeStep> steps, std::span<const model::QwenImageTensor> tensors);
}  // namespace llmp::spec
#endif  // LLMP_SPEC_IMAGE_SPATIAL_H_
