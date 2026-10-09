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
// (fattn_vec_plan.h, stream-k off, not sparse), so the pool scratch the
// launch draws is known and checked before it is queued. This
// case never compacts the mask (use_sparse is false); the function
// launch_fattn names for that is fattn_mma.cu's.

#include <cstdint>
#include <cstring>
#include <expected>
#include <string>
#include <utility>

#include "base/bytes.h"
#include "fattn-vec.cuh"
#include "kernels/ggml/fattn_vec_plan.h"
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

}  // namespace

template <int kHead>
static std::expected<FlashAttnPlan, KernelFailure> PlanFlashAttnVecHead(const LaunchContext& launch,
                                                                        const ggml_tensor* node) {
  if (auto checked = kHead == 64 ? CheckFlashAttnVec(node) : CheckFlashAttnVec256(node); !checked) {
    return std::unexpected(checked.error());
  }
  const ggml_tensor* q = node->src[0];
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
  if (auto grid = detail::PlanVectorGrid(node, kHead, per_sm, device.nsm, plan); !grid) {
    return std::unexpected(grid.error());
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
