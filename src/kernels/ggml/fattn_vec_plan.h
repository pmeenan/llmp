// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: MIT AND Apache-2.0

// The vector flash-attention kernel's launch arithmetic after its occupancy
// query, shared by every vector instance llmpalooza compiles (fattn.cu's F16
// ones, fattn_vec_quant.cu's quantized ones): a recorded copy of
// launch_fattn's host arithmetic (fattn-common.cuh at the pin, stream-k
// off, not sparse), so that the pool scratch a launch draws is known and
// checked before it is queued. The kernel's own type does not enter it
// beyond its occupancy, which the caller queries on the specialization it
// will launch.

#ifndef LLMP_KERNELS_GGML_FATTN_VEC_PLAN_H_
#define LLMP_KERNELS_GGML_FATTN_VEC_PLAN_H_

#include <algorithm>
#include <climits>
#include <cstdint>
#include <expected>

#include "ggml.h"
#include "kernels/ggml/ops.h"
#include "kernels/ggml/tensors.h"

namespace llmp::kernels::ggml::detail {

// `node` a checked flash_attn_ext node of head size `head`, `per_sm` the
// specialization's blocks per SM, `nsm` the device's SMs. Fills everything
// but columns_per_block, which the caller chose for the query.
inline std::expected<void, KernelFailure> PlanVectorGrid(const ggml_tensor* node, std::int64_t head,
                                                         int per_sm, int nsm, FlashAttnPlan& plan) {
  constexpr int kKqStride = 256;         // FATTN_KQ_STRIDE
  constexpr std::uint64_t kBlock = 256;  // the pool's block boundary (launch.h)
  const auto round = [](std::uint64_t bytes) { return (bytes + kBlock - 1) / kBlock * kBlock; };
  const ggml_tensor* q = node->src[0];
  const ggml_tensor* k = node->src[1];
  // launch_fattn<D, ncols1 = columns per block, ncols2 = 1>.
  const std::int64_t ntiles_x = (q->ne[1] + plan.columns_per_block - 1) / plan.columns_per_block;
  const std::int64_t gqa = q->ne[2] / k->ne[2];
  const std::int64_t ntiles_dst = ntiles_x * gqa * k->ne[2] * q->ne[3];
  const std::int64_t ntiles_kv = (k->ne[1] + head - 1) / head;
  // launch_fattn's vector grid uses x=query tiles, z=query heads and
  // an int tail-efficiency product. Reject shapes before those narrowings.
  if (ntiles_x > INT32_MAX || q->ne[2] > 65535 || ntiles_kv > (INT32_MAX / 100) / ntiles_dst) {
    return std::unexpected(KernelFailure{
        .error = KernelError::kRejected,
        .detail = "vector attention exceeds the pinned grid or 32-bit tile arithmetic"});
  }
  std::int64_t parallel = std::min<std::int64_t>(per_sm, ntiles_kv);
  const std::int64_t per_wave = static_cast<std::int64_t>(nsm) * per_sm;
  std::int64_t best_waves = 0;
  std::int64_t best_efficiency = 0;
  for (std::int64_t test = parallel; test <= ntiles_kv; ++test) {
    const std::int64_t total = ntiles_dst * test;
    const std::int64_t waves = (total + per_wave - 1) / per_wave;
    const std::int64_t efficiency = 100 * total / (waves * per_wave);
    if (best_efficiency >= 95 && waves > best_waves) {
      break;
    }
    if (efficiency > best_efficiency) {
      best_waves = waves;
      best_efficiency = efficiency;
      parallel = test;
    }
  }
  plan.parallel_blocks = static_cast<int>(parallel);
  plan.mask_prepass = k->ne[1] % kKqStride == 0 && (q->ne[1] >= 1024 || q->ne[3] > 1);
  if (plan.mask_prepass) {
    plan.scratch += round(static_cast<std::uint64_t>(ntiles_x * q->ne[3]) * sizeof(int));
  }
  if (plan.parallel_blocks > 1) {
    plan.scratch += round(static_cast<std::uint64_t>(parallel) *
                          static_cast<std::uint64_t>(ggml_nelements(node)) * sizeof(float));
    plan.scratch +=
        round(static_cast<std::uint64_t>(parallel) * static_cast<std::uint64_t>(ggml_nrows(node)) *
              2 * sizeof(float));  // float2
  }
  return {};
}

}  // namespace llmp::kernels::ggml::detail

#endif  // LLMP_KERNELS_GGML_FATTN_VEC_PLAN_H_
