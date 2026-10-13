// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef LLMP_KERNELS_GGML_CONV2D_TILED_H_
#define LLMP_KERNELS_GGML_CONV2D_TILED_H_

#include <cstdint>
#include <expected>
#include <string_view>

#include "ggml.h"
#include "kernels/ggml/tensors.h"

namespace llmp::kernels::ggml {
class LaunchContext;
inline constexpr std::string_view kConv2dTiledF32 = "ggml.conv2d.tiled_f32";
// An implementation scratch target, not a limit on model/image geometry.
inline constexpr std::uint64_t kConv2dTileBudget = 64ULL << 20U;
struct Conv2dTilePlan {
  std::uint64_t rows = 0;
  std::uint64_t columns_bytes = 0;
  std::uint64_t product_bytes = 0;
  std::uint64_t scratch = 0;
  std::uint64_t stripes = 0;
  std::uint64_t launches = 0;
};
// Host-only descriptor and integer-domain checks. The same partition is used
// by physical scratch measurement and launch. A smaller supplied budget is
// useful for boundary controls; the registered implementation uses the target.
std::expected<Conv2dTilePlan, KernelFailure> CheckConv2dTiledF32(
    const ggml_tensor* node, std::uint64_t budget = kConv2dTileBudget);
std::expected<std::uint64_t, KernelFailure> PlanConv2dTiledF32(const LaunchContext& launch,
                                                               const ggml_tensor* node);
std::expected<void, KernelFailure> Conv2dTiledF32(LaunchContext& launch, ggml_tensor* node);
}  // namespace llmp::kernels::ggml
#endif  // LLMP_KERNELS_GGML_CONV2D_TILED_H_
