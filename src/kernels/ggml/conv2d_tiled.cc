// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "kernels/ggml/conv2d_tiled.h"

#include <algorithm>
#include <cstdint>
#include <limits>

#include "kernels/ggml/validate_ext.h"
#include "kernels/ggml/validate_util.h"

namespace llmp::kernels::ggml {
std::expected<Conv2dTilePlan, KernelFailure> CheckConv2dTiledF32(const ggml_tensor* node,
                                                                 std::uint64_t budget) {
  if (auto valid = CheckConv2dF32(node); !valid) return std::unexpected(valid.error());
  const auto* w = node->src[0];
  if (w->ne[0] != 3 || w->ne[1] != 3 || node->op_params[2] != 1 || node->op_params[3] != 1)
    return detail::Rejected("tiled F32 convolution requires 3x3, stride/dilation1, padding1");
  // Its bias, where lowered with one (a third source): an F32 [Co].
  if (const ggml_tensor* bias = node->src[2];
      bias != nullptr && (bias->type != GGML_TYPE_F32 || !ggml_is_contiguous(bias) ||
                          ggml_nelements(bias) != node->ne[2] || !detail::AllSane({bias}) ||
                          !detail::Disjoint(node, bias, false)))
    return detail::Rejected("tiled convolution bias is not an F32 [Co] apart from its output");
  // Pinned im2col casts IC*KH*KW to int before choosing its block width.
  // cuBLAS also takes K/M/N as int. The common spatial check bounds all
  // original byte spans before these derived products are formed.
  const auto k = static_cast<std::uint64_t>(w->ne[2]) * 9;
  const auto width = static_cast<std::uint64_t>(node->ne[0]);
  const auto height = static_cast<std::uint64_t>(node->ne[1]);
  const auto channels = static_cast<std::uint64_t>(node->ne[2]);
  if (k > static_cast<std::uint64_t>(INT32_MAX) || channels > INT32_MAX || height > INT32_MAX ||
      width > INT32_MAX)
    return detail::Rejected("tiled convolution exceeds im2col/cuBLAS integer operands");
  const auto round = [](std::uint64_t n) { return (n + 255) & ~std::uint64_t{255}; };
  const auto bytes = [&](std::uint64_t rows) {
    // All operands are bounded by the validated spatial spans; even the
    // expanded nine-tap columns fit uint64_t. Rounding is included per draw.
    return round(4 * width * rows * k) + 2 * round(4 * width * rows * channels);
  };
  budget = std::min(budget, kConv2dTileBudget);
  if (bytes(1) > budget) return detail::Rejected("one convolution row exceeds paid tile workspace");
  std::uint64_t lo = 1, hi = std::min(height, static_cast<std::uint64_t>(INT32_MAX) / width);
  while (lo < hi) {
    const auto mid = lo + (hi - lo + 1) / 2;
    if (bytes(mid) <= budget)
      lo = mid;
    else
      hi = mid - 1;
  }
  const auto stripes = ((height + lo - 1) / lo) * static_cast<std::uint64_t>(node->ne[3]);
  // Every stripe queues im2col, one product, a packed transpose and a
  // pitched copy; the bias, where it has one, in the scatter.
  if (stripes > UINT64_MAX / 4) return detail::Rejected("tiled convolution launch count overflow");
  return Conv2dTilePlan{.rows = lo,
                        .columns_bytes = round(4 * width * lo * k),
                        .product_bytes = round(4 * width * lo * channels),
                        .scratch = bytes(lo),
                        .stripes = stripes,
                        .launches = stripes * 4};
}
}  // namespace llmp::kernels::ggml
