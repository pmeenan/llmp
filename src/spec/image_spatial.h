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

#include "model/qwen_image.h"
#include "spec/graph_links.h"
#include "spec/spec.h"

namespace llmp::spec {
// The Qwen-Image pipeline's planning policy (planner/policy.h canonical text), every
// stage's: the text encoder's and the flow's QK norms take the opt-in
// sectioned norm and rotation (planner/pattern.h rms_norm_mul_rope_sections).
inline constexpr std::string_view kImagePlanningPolicy =
    "max_decode_rows 16\nenable rms_norm_mul_rope_sections\n";
// Mapping only, over an already-owned, paid VaeDecoderPlan/tensor inventory.
// Consumes a closed serial conv/nearest span, preserving source tensor names
// and F32 shapes. Graph inference owns the actual spatial input geometry;
// this mapping neither dispatches work nor imposes legacy intermediate casts.
// Caller prepays at least this bound for the returned spec and temporary names.
inline constexpr std::uint64_t kVaeSpatialSpecBytes = 1ULL << 20U;
std::expected<ModelSpec, std::string> VaeSpatialSpec(
    std::span<const model::VaeStep> steps, std::span<const model::QwenImageTensor> tensors);
struct SpatialComposition {
  ModelSpec spec;
  std::vector<GraphLink> links;
  std::vector<std::uint16_t> exports;
};
// Complete first-frame source plan, including residual and duplicate-up edges.
// Source buffers become SSA values; Copy saves an alias, not a payload copy.
// Same prepaid 1MiB source-mapping allowance, at most256 steps/4096 tensor names.
std::expected<SpatialComposition, std::string> VaeDecoderSpec(
    std::span<const model::VaeStep> steps, std::span<const model::QwenImageTensor> tensors);
// Decoder including typed latent reshape and source per-channel affine input.
// Primary [1,width*height,z_dim,1], extra [1,2,z_dim,1] carries F32 std/mean.
// Caller pays the same bounded VAE mapping inventory before construction.
std::expected<SpatialComposition, std::string> ImageDecodeSpec(
    const model::QwenImageVaeProfile& profile, std::uint32_t height, std::uint32_t width,
    std::span<const model::QwenImageTensor> tensors);
inline std::expected<SpatialComposition, std::string> ImageDecodeSpec(
    const model::QwenImageVaeProfile& profile, std::uint32_t grid,
    std::span<const model::QwenImageTensor> tensors) {
  return ImageDecodeSpec(profile, grid, grid, tensors);
}
// Four BF16 source projections; exports modulation and final normalization scale.
// Caller prepays the existing 1MiB source-mapping allowance before this call.
// This maps only source roles and branches; input sinusoid policy belongs to the caller.
std::expected<SpatialComposition, std::string> ImageTimestepSpec(
    const model::QwenImageDenoiserProfile& profile,
    std::span<const model::QwenImageTensor> tensors);
// Complete conditioner and one source transformer block. Primary input is the
// already projected text/image stream. Extra owner-local ports are: 0 sinusoid
// [1,2,timestep_dim,1] F32, 1 positions [rows,3,1,1] I32, and 2 explicit attention
// mask [rows,rows,1,1] F32. Text-prefix policy/positions belong to the source caller.
// Exports updated stream and final normalization scale. Same prepaid 1MiB mapper.
std::expected<SpatialComposition, std::string> ImageTransformerBlockSpec(
    const model::QwenImageDenoiserProfile& profile, std::uint32_t block, std::uint32_t text_rows,
    std::span<const model::QwenImageTensor> tensors);
// Source text encoder as a stateless typed graph: primary I32 IDs, extra
// I32 positions and explicit causal F32 mask. Exports pre-final-norm features
// after the source prompt prefix; no generation head or token cache is implied.
std::expected<std::uint64_t, std::string> ImageTextSpecBound(
    const model::QwenImageTextProfile& profile, std::span<const model::QwenImageTensor> tensors);
std::expected<SpatialComposition, std::string> ImageTextSpec(
    const model::QwenImageTextProfile& profile, std::uint32_t rows, std::uint32_t drop,
    std::span<const model::QwenImageTensor> tensors);
// Complete denoiser plus explicit Euler residual. Caller prepays the returned
// mapping bound before ImageFlowSpec; includes all copies, no source payloads.
std::expected<std::uint64_t, std::string> ImageFlowSpecBound(
    const model::QwenImageDenoiserProfile& profile,
    std::span<const model::QwenImageTensor> tensors);
// Primary text embedding rows; extra ports:0 latent,1 sinusoid,2 I32 positions,
// 3 explicit mask,4 F32 dt scalar. Exports full noise and updated latent.
std::expected<SpatialComposition, std::string> ImageFlowSpec(
    const model::QwenImageDenoiserProfile& profile, std::uint32_t text_rows,
    std::uint32_t image_rows, std::span<const model::QwenImageTensor> tensors);
}  // namespace llmp::spec
#endif  // LLMP_SPEC_IMAGE_SPATIAL_H_
