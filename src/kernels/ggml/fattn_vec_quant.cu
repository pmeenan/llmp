// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: MIT AND Apache-2.0

// GGML's vector flash-attention kernel over Q8_0 or Q4_0 K and V
// (kv_quant.h FlashAttnVecQuant). The kernel, launch_fattn and the combine
// and mask pre-pass kernels are GGML's own, from fattn-vec.cuh and
// fattn-common.cuh at the pin: this unit instantiates exactly the cases
// llmpalooza's models need, ggml_cuda_flash_attn_ext_vec_case<D, T, T> for
// D 64 (Qwen2), 256 (Gemma 2, 3 and 4's local layers, Qwen3.8), and 512
// (Gemma 4's global layers) and T Q8_0 (and Q4_0 at 64 and 256), as GGML's
// template-instances/fattn-vec-instance-q8_0-q8_0.cu and -q4_0-q4_0.cu do for
// D 64, 128 and 256, with GGML's device flags (CMakeLists.txt), as fattn.cu
// builds its F16 cases. No model's K or V has head size 128 today.

#include <cstdint>
#include <cstring>
#include <expected>
#include <string>

#include "base/bytes.h"
#include "fattn-vec.cuh"
#include "kernels/ggml/fattn_vec_plan.h"
#include "kernels/ggml/kv_quant.h"
#include "kernels/ggml/ops.h"

DECL_FATTN_VEC_CASE(64, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0);
DECL_FATTN_VEC_CASE(256, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0);
DECL_FATTN_VEC_CASE(512, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0);
DECL_FATTN_VEC_CASE(64, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0);
DECL_FATTN_VEC_CASE(256, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0);

namespace llmp::kernels::ggml {
namespace {

// The occupancy of the specialization the upstream case will launch:
// ggml_cuda_flash_attn_ext_vec_case_impl's 128 threads, no dynamic shared
// memory (fattn-vec.cuh).
template <int kHead, ggml_type kType>
cudaError_t Occupancy(int threads, int columns, bool softcap, int* per_sm) {
  const auto query = [&]<int kColumns, bool kSoftcap>() {
    return cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        per_sm, flash_attn_ext_vec<kHead, kColumns, kType, kType, kSoftcap>, threads, 0);
  };
  if (columns == 1) {
    return softcap ? query.template operator()<1, true>() : query.template operator()<1, false>();
  }
  return softcap ? query.template operator()<2, true>() : query.template operator()<2, false>();
}

template <int kHead, ggml_type kType>
void RunCase(ggml_backend_cuda_context& context, ggml_tensor* node) {
  ggml_cuda_flash_attn_ext_vec_case<kHead, kType, kType>(context, node);
}

}  // namespace

std::expected<FlashAttnPlan, KernelFailure> PlanFlashAttnVecQuant(const LaunchContext& launch,
                                                                  const ggml_tensor* node) {
  if (auto checked = CheckFlashAttnVecQuant(node); !checked) {
    return std::unexpected(checked.error());
  }
  const ggml_tensor* q = node->src[0];
  const auto& device = ggml_cuda_info().devices[launch.device()];
  FlashAttnPlan plan;
  plan.columns_per_block = q->ne[1] == 1 ? 1 : 2;
  float softcap = 0;
  std::memcpy(&softcap, node->op_params + 2, sizeof(softcap));
  const int threads = ggml_cuda_fattn_vec_get_nthreads_host(device.cc);
  const bool q8 = node->src[1]->type == GGML_TYPE_Q8_0;
  int per_sm = 0;
  const bool capped = softcap != 0.0f;
  const std::int64_t head = q->ne[0];
  const cudaError_t occupancy =
      head == 64
          ? (q8 ? Occupancy<64, GGML_TYPE_Q8_0>(threads, plan.columns_per_block, capped, &per_sm)
                : Occupancy<64, GGML_TYPE_Q4_0>(threads, plan.columns_per_block, capped, &per_sm))
          : (head == 256 ? (q8 ? Occupancy<256, GGML_TYPE_Q8_0>(threads, plan.columns_per_block,
                                                                capped, &per_sm)
                               : Occupancy<256, GGML_TYPE_Q4_0>(threads, plan.columns_per_block,
                                                                capped, &per_sm))
                         : Occupancy<512, GGML_TYPE_Q8_0>(threads, plan.columns_per_block, capped,
                                                          &per_sm));
  if (occupancy != cudaSuccess || per_sm <= 0) {
    return std::unexpected(
        KernelFailure{.error = KernelError::kUnknown,
                      .detail = std::string("the vector kernel's occupancy query failed: ") +
                                cudaGetErrorString(occupancy)});
  }
  if (auto grid = detail::PlanVectorGrid(node, q->ne[0], per_sm, device.nsm, plan); !grid) {
    return std::unexpected(grid.error());
  }
  return plan;
}

std::expected<void, KernelFailure> FlashAttnVecQuant(LaunchContext& launch, ggml_tensor* node) {
  auto plan = PlanFlashAttnVecQuant(launch, node);
  if (!plan) {
    return std::unexpected(plan.error());
  }
  const bool q8 = node->src[1]->type == GGML_TYPE_Q8_0;
  const std::int64_t head = node->src[0]->ne[0];
  void (*run)(ggml_backend_cuda_context&, ggml_tensor*) =
      head == 64
          ? (q8 ? &RunCase<64, GGML_TYPE_Q8_0> : &RunCase<64, GGML_TYPE_Q4_0>)
          : (head == 256 ? (q8 ? &RunCase<256, GGML_TYPE_Q8_0> : &RunCase<256, GGML_TYPE_Q4_0>)
                         : &RunCase<512, GGML_TYPE_Q8_0>);
  return launch.Run(base::Bytes(plan->scratch),
                    [node, run](ggml_backend_cuda_context& context) { run(context, node); });
}

}  // namespace llmp::kernels::ggml
