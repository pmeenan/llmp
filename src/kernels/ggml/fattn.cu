// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: MIT AND Apache-2.0

// GGML's vector flash-attention kernel for the EXL3 plan (ops.h
// FlashAttnVec; docs/backend-proof.md, "Native EXL3 operation plan"). The
// kernel, launch_fattn and the combine and mask pre-pass kernels are
// GGML's own, from fattn-vec.cuh and fattn-common.cuh at llama.cpp
// b29c606e2: this unit instantiates the D64 EXL3 and D256 Gemma-local
// ggml_cuda_flash_attn_ext_vec_case<D, F16, F16> cases, as GGML's
// template-instances/fattn-vec-instance-f16-f16.cu does, and is built with
// GGML's device flags (CMakeLists.txt), so that its SASS is the bridge's.
//
// PlanFlashAttnVec is a recorded copy of launch_fattn's host arithmetic
// (fattn-common.cuh:1106-1197, stream-k off, not sparse), so the pool
// scratch the launch draws is known and checked before it is queued. This
// case never compacts the mask (use_sparse is false); the function
// launch_fattn names for that is fattn_mma.cu's.

#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstring>
#include <expected>
#include <string>
#include <utility>

#include "base/bytes.h"
#include "fattn-vec.cuh"
#include "kernels/ggml/ops.h"
#include "kernels/ggml/validate.h"

DECL_FATTN_VEC_CASE(64, GGML_TYPE_F16, GGML_TYPE_F16);
DECL_FATTN_VEC_CASE(256, GGML_TYPE_F16, GGML_TYPE_F16);

namespace llmp::kernels::ggml {
namespace {

std::unexpected<KernelFailure> Rejected(std::string detail) {
  return std::unexpected(
      KernelFailure{.error = KernelError::kRejected, .detail = std::move(detail)});
}

constexpr int kKqStride = 256;         // FATTN_KQ_STRIDE
constexpr std::uint64_t kBlock = 256;  // the pool's block boundary (launch.h)

std::uint64_t Round(std::uint64_t bytes) { return (bytes + kBlock - 1) / kBlock * kBlock; }

}  // namespace

template <int kHead>
static std::expected<FlashAttnPlan, KernelFailure> PlanFlashAttnVecHead(const LaunchContext& launch,
                                                                        const ggml_tensor* node) {
  if (auto checked = kHead == 64 ? CheckFlashAttnVec(node) : CheckFlashAttnVec256(node); !checked) {
    return std::unexpected(checked.error());
  }
  const ggml_tensor* q = node->src[0];
  const ggml_tensor* k = node->src[1];
  const auto& device = ggml_cuda_info().devices[launch.device()];
  FlashAttnPlan plan;
  plan.columns_per_block = q->ne[1] == 1 ? 1 : 2;
  // ggml_cuda_flash_attn_ext_vec_case_impl: 128 threads, no dynamic shared
  // memory (fattn-vec.cuh:532-541).
  const int threads = ggml_cuda_fattn_vec_get_nthreads_host(device.cc);
  int per_sm = 0;
  // Query the same specialization the upstream case will launch.
  const auto query = [&]<bool kSoftcap>() {
    return plan.columns_per_block == 1
               ? cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                     &per_sm, flash_attn_ext_vec<kHead, 1, GGML_TYPE_F16, GGML_TYPE_F16, kSoftcap>,
                     threads, 0)
               : cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                     &per_sm, flash_attn_ext_vec<kHead, 2, GGML_TYPE_F16, GGML_TYPE_F16, kSoftcap>,
                     threads, 0);
  };
  cudaError_t occupancy;
  if constexpr (kHead == 256) {
    float softcap = 0;
    std::memcpy(&softcap, node->op_params + 2, sizeof(softcap));
    occupancy =
        softcap != 0.0f ? query.template operator()<true>() : query.template operator()<false>();
  } else {
    occupancy = query.template operator()<false>();
  }
  if (occupancy != cudaSuccess || per_sm <= 0) {
    return std::unexpected(
        KernelFailure{.error = KernelError::kUnknown,
                      .detail = std::string("the vector kernel's occupancy query failed: ") +
                                cudaGetErrorString(occupancy)});
  }
  // launch_fattn<D, ncols1 = columns per block, ncols2 = 1>.
  const std::int64_t ntiles_x = (q->ne[1] + plan.columns_per_block - 1) / plan.columns_per_block;
  const std::int64_t gqa = q->ne[2] / k->ne[2];
  const std::int64_t ntiles_dst = ntiles_x * gqa * k->ne[2] * q->ne[3];
  const std::int64_t ntiles_kv = (k->ne[1] + kHead - 1) / kHead;
  // launch_fattn's vector grid uses x=query tiles, z=query heads and
  // an int tail-efficiency product. Reject shapes before those narrowings.
  if (ntiles_x > INT32_MAX || q->ne[2] > 65535 || ntiles_kv > (INT32_MAX / 100) / ntiles_dst) {
    return Rejected("vector attention exceeds the pinned grid or 32-bit tile arithmetic");
  }
  std::int64_t parallel = std::min<std::int64_t>(per_sm, ntiles_kv);
  const std::int64_t per_wave = static_cast<std::int64_t>(device.nsm) * per_sm;
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
    plan.scratch += Round(static_cast<std::uint64_t>(ntiles_x * q->ne[3]) * sizeof(int));
  }
  if (plan.parallel_blocks > 1) {
    plan.scratch += Round(static_cast<std::uint64_t>(parallel) *
                          static_cast<std::uint64_t>(ggml_nelements(node)) * sizeof(float));
    plan.scratch += Round(static_cast<std::uint64_t>(parallel) *
                          static_cast<std::uint64_t>(ggml_nrows(node)) * sizeof(float2));
  }
  return plan;
}

std::expected<FlashAttnPlan, KernelFailure> PlanFlashAttnVec(const LaunchContext& launch,
                                                             const ggml_tensor* node) {
  return PlanFlashAttnVecHead<64>(launch, node);
}
std::expected<FlashAttnPlan, KernelFailure> PlanFlashAttnVec256(const LaunchContext& launch,
                                                                const ggml_tensor* node) {
  return PlanFlashAttnVecHead<256>(launch, node);
}

std::expected<void, KernelFailure> FlashAttnVec(LaunchContext& launch, ggml_tensor* node) {
  auto plan = PlanFlashAttnVec(launch, node);
  if (!plan) {
    return std::unexpected(plan.error());
  }
  if (plan->parallel_blocks < 1) {
    return Rejected("no parallel blocks");
  }
  return launch.Run(base::Bytes(plan->scratch), [node](ggml_backend_cuda_context& context) {
    ggml_cuda_flash_attn_ext_vec_case<64, GGML_TYPE_F16, GGML_TYPE_F16>(context, node);
  });
}

std::expected<void, KernelFailure> FlashAttnVec256(LaunchContext& launch, ggml_tensor* node) {
  auto plan = PlanFlashAttnVec256(launch, node);
  if (!plan) return std::unexpected(plan.error());
  return launch.Run(base::Bytes(plan->scratch), [node](ggml_backend_cuda_context& context) {
    ggml_cuda_flash_attn_ext_vec_case<256, GGML_TYPE_F16, GGML_TYPE_F16>(context, node);
  });
}

// Pinned overall selector, restricted to the new Gemma-local contract.
// Other models retain their existing identities and device policies.
bool FlashAttnVec256Selected(const LaunchContext& launch, const ggml_tensor* node) {
  if (!CheckFlashAttnVec256(node)) return false;
  const int cc = ggml_cuda_info().devices[launch.device()].cc;
  return GGML_CUDA_CC_IS_NVIDIA(cc) && turing_mma_available(cc) &&
         cc >= GGML_CUDA_CC_ADA_LOVELACE && node->src[0]->ne[1] == 1;
}

}  // namespace llmp::kernels::ggml
