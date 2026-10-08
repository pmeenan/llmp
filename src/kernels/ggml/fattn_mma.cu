// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: MIT AND Apache-2.0

// GGML's tensor-core flash attention for head dimensions 256 and 512 under
// llmpalooza's dispatch (ops_ext.h FlashAttnMma), and at 128 without head
// grouping or a mask (FlashAttnMma128). Llmpalooza does not compile
// GGML's fattn.cu, whose dispatcher names every head size's and K/V type's
// instance; this unit has, from fattn.cu at llama.cpp b29c606e2, with
// the query-tile sparse union backported from dc9879cf (PR 29298):
//
// - flash_attn_mask_to_sparse_indices, ggml_cuda_flash_attn_ext_compact_mask
//   and ggml_cuda_flash_attn_ext_mma_f16_shall_use_sparse (fattn.cu:8-128),
//   with formatting and one llmpalooza condition (a node its graph
//   marks, llmp_ops.h SetFlashAttnSparseAny, gathers below 4,096 cells
//   too): a sparse gather shared by up to eight queries at D=256/512;
// - the column choice of ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1 for
//   groups of 8 query heads (fattn.cu:131-164).
//
// PlanFlashAttnMma is a recorded copy of launch_fattn's host arithmetic with
// stream-k on (fattn-common.cuh:1085-1180), so the pool scratch the launch
// draws is known before it is queued.

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <expected>
#include <string>
#include <utility>

#include "base/bytes.h"
#include "common.cuh"
#include "fattn-common.cuh"
#include "kernels/ggml/fattn_mma.h"
#include "kernels/ggml/llmp_ops.h"
#include "kernels/ggml/ops_ext.h"
#include "kernels/ggml/validate_ext.h"
#include "kernels/ggml/validate_util.h"

// ---- From GGML's fattn.cu (MIT) ----

// one list per group of ncols1 queries: a column is selected if any query of the group can see it
template <int ncols1, bool oob>
__launch_bounds__(256, 1) static __global__
    void flash_attn_mask_to_sparse_indices(const half* mask_ptr, int32_t* indices_ptr,
                                           int32_t* counts_ptr, const int ne30, const int n_queries,
                                           const int n_kv_max, const int64_t s31,
                                           const int64_t s33) {
  ggml_cuda_pdl_sync();

  constexpr int values_per_lane = 8;
  const int tid = threadIdx.x;
  const int warp = tid / WARP_SIZE;
  const int lane = tid % WARP_SIZE;
  const int sequence = blockIdx.y;
  const int group = blockIdx.x;

  const int q0 = group * ncols1;
  const int q1 = min(q0 + ncols1, n_queries);

  const half* mask = mask_ptr + sequence * s33 + q0 * s31;
  int32_t* indices = indices_ptr + (int64_t(sequence) * gridDim.x + group) * n_kv_max;

  __shared__ int warp_offsets[256 / WARP_SIZE];
  __shared__ int row_count;
  __shared__ int chunk_count;

  if (tid == 0) {
    row_count = 0;
  }
  __syncthreads();

  for (int i0 = 0; i0 < ne30; i0 += blockDim.x * values_per_lane) {
    uint32_t selected_warp[values_per_lane];
    int warp_count = 0;
#pragma unroll
    for (int item = 0; item < values_per_lane; ++item) {
      const int i = i0 + (warp * values_per_lane + item) * WARP_SIZE + lane;
      bool selected = false;
      if (i < ne30) {
#pragma unroll
        for (int q = 0; q < ncols1; ++q) {
          selected |= (!oob || q < q1 - q0) && isfinite(__half2float(mask[q * s31 + i]));
        }
      }
      selected_warp[item] = __ballot_sync(0xFFFFFFFF, selected);
      warp_count += __popc(selected_warp[item]);
    }

    if (lane == 0) {
      warp_offsets[warp] = warp_count;
    }
    __syncthreads();

    if (tid == 0) {
      int offset = 0;
#pragma unroll
      for (int iw = 0; iw < 256 / WARP_SIZE; ++iw) {
        const int count = warp_offsets[iw];
        warp_offsets[iw] = offset;
        offset += count;
      }
      chunk_count = offset;
    }
    __syncthreads();

    const uint32_t lane_mask = lane == 0 ? 0 : (1u << lane) - 1;
    int warp_item_offset = 0;
#pragma unroll
    for (int item = 0; item < values_per_lane; ++item) {
      const int i = i0 + (warp * values_per_lane + item) * WARP_SIZE + lane;
      const int dst = row_count + warp_offsets[warp] + warp_item_offset +
                      __popc(selected_warp[item] & lane_mask);
      if ((selected_warp[item] & (uint32_t(1) << lane)) && dst < n_kv_max) {
        indices[dst] = i;
      }
      warp_item_offset += __popc(selected_warp[item]);
    }
    __syncthreads();

    if (tid == 0) {
      row_count += chunk_count;
    }
    __syncthreads();
  }

  const int count = min(row_count, n_kv_max);
  for (int i = count + tid; i < n_kv_max; i += blockDim.x) {
    indices[i] = -1;
  }
  if (tid == 0) {
    // One column retains the original fixed loop bound, including its
    // trailing -1 cells. Live-count elision is only a wide-tile choice.
    counts_ptr[int64_t(sequence) * gridDim.x + group] = ncols1 == 1 ? n_kv_max : count;
  }
  __syncthreads();

  // the dependent grid reads indices, signal once the row is complete
  ggml_cuda_pdl_lc();
}

void ggml_cuda_flash_attn_ext_compact_mask(const ggml_tensor* mask, int32_t* indices,
                                           int32_t* counts, int32_t n_queries, int32_t ncols1,
                                           int32_t n_kv_max, cudaStream_t stream) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
  GGML_UNUSED_VARS(mask, indices, counts, n_queries, ncols1, n_kv_max, stream);
  GGML_ABORT("sparse flash attention is only supported on NVIDIA CUDA");
#else
  const auto s31 = static_cast<int64_t>(mask->nb[1] / sizeof(half));
  const auto s33 = static_cast<int64_t>(mask->nb[3] / sizeof(half));
  const dim3 blocks_num(static_cast<unsigned>((n_queries + ncols1 - 1) / ncols1),
                        static_cast<unsigned>(mask->ne[3]), 1);
  const dim3 block_dim(256, 1, 1);
  const ggml_cuda_kernel_launch_params launch_params(blocks_num, block_dim, 0, stream);
  // the last group of queries is partial only if ncols1 does not divide n_queries
  GGML_ASSERT(ncols1 == 1 || ncols1 == 8);
  const auto kernel = ncols1 == 1          ? flash_attn_mask_to_sparse_indices<1, false>
                      : n_queries % 8 != 0 ? flash_attn_mask_to_sparse_indices<8, true>
                                           : flash_attn_mask_to_sparse_indices<8, false>;
  ggml_cuda_kernel_launch(kernel, launch_params, (const half*)mask->data, indices, counts,
                          int(mask->ne[0]), n_queries, n_kv_max, s31, s33);
  CUDA_CHECK(cudaGetLastError());
#endif  // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
}

bool ggml_cuda_flash_attn_ext_mma_f16_shall_use_sparse(const int cc, const ggml_tensor* dst,
                                                       const int ncols1, const int ncols2) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
  GGML_UNUSED_VARS(cc, dst, ncols1, ncols2);
  return false;
#else
  const ggml_tensor* Q = dst->src[0];
  const ggml_tensor* K = dst->src[1];
  const ggml_tensor* mask = dst->src[3];

  float max_bias = 0.0f;
  float logit_softcap = 0.0f;
  memcpy(&max_bias, (const float*)dst->op_params + 1, sizeof(float));
  memcpy(&logit_softcap, (const float*)dst->op_params + 2, sizeof(float));

  const int32_t n_kv_max = ggml_get_op_params_i32(dst, 4);

  GGML_UNUSED_VARS(ncols2);
  const bool any = ggml_get_op_params_i32(dst, llmp::kernels::ggml::kFlashAttnSparseParam) == 1;
  const bool wide =
      ggml_get_op_params_i32(dst, llmp::kernels::ggml::kFlashAttnWideSparseParam) == 1;

  return GGML_CUDA_CC_IS_NVIDIA(cc) && turing_mma_available(cc) && mask != nullptr &&
         (wide || (Q->ne[0] == 512 && ncols1 == 1)) && n_kv_max > 0 && max_bias == 0.0f &&
         logit_softcap == 0.0f && mask->ne[0] == K->ne[1] && mask->ne[1] >= Q->ne[1] &&
         mask->ne[2] == 1 && K->ne[1] >= std::max<int64_t>(any ? 0 : 4096, 2LL * n_kv_max);
#endif  // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
}

// ---- llmpalooza ----

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

static std::expected<FlashAttnMmaPlan, KernelFailure> PlanFlashAttnMmaGroup(
    const LaunchContext& launch, const ggml_tensor* node, bool wide_sparse, bool group2) {
  if (auto checked = group2 ? CheckFlashAttnMmaGqa2(node) : CheckFlashAttnMma(node); !checked) {
    return std::unexpected(checked.error());
  }
  const ggml_tensor* q = node->src[0];
  const ggml_tensor* k = node->src[1];
  const ggml_tensor* mask = node->src[3];
  const auto& device = ggml_cuda_info().devices[launch.device()];
  const int cc = device.cc;
  if (!GGML_CUDA_CC_IS_NVIDIA(cc) || !turing_mma_available(cc)) {
    return Rejected("the MMA flash-attention kernels need Turing or later");
  }
  const int kGroup = group2 ? 2 : 8;
  FlashAttnMmaPlan plan;
  plan.group = kGroup;
  plan.head = static_cast<int>(q->ne[0]);
  // The optional sparse kernel shares a query tile's union. The original
  // path uses one sparse column at D512 and dense attention at D256.
  const std::int32_t n_kv_max = node->op_params[4];
  const bool any = node->op_params[kFlashAttnSparseParam] == 1;
  const bool sparse_eligible = n_kv_max > 0 && mask->ne[0] == k->ne[1] && mask->ne[1] >= q->ne[1] &&
                               k->ne[1] >= std::max<std::int64_t>(any ? 0 : 4096, 2LL * n_kv_max);
  if (group2) {
    plan.columns = q->ne[1] <= 4   ? 4
                   : q->ne[1] <= 8 ? 8
                   : (q->ne[1] <= 16 || ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_TURING)
                       ? 16
                       : 32;
  } else if (sparse_eligible && (plan.head == 512 || wide_sparse)) {
    plan.columns = wide_sparse && q->ne[1] > 4 ? 8 : 1;
    plan.sparse = true;
  } else if (q->ne[1] <= 1) {
    plan.columns = 1;
  } else if (q->ne[1] <= 2) {
    plan.columns = 2;
  } else if (q->ne[1] <= 4 || ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_TURING) {
    plan.columns = 4;
  } else {
    plan.columns = 8;
  }
  auto shape = group2 ? detail::FlashAttnMmaShapeGqa2(plan.columns, launch.device(),
                                                      detail::ParamF32(node, 2) != 0.0f)
               : plan.head == 512
                   ? detail::FlashAttnMmaShape512(plan.columns, plan.sparse, launch.device())
                   : detail::FlashAttnMmaShape256(plan.columns, plan.sparse, launch.device());
  if (!shape) {
    return std::unexpected(KernelFailure{.error = KernelError::kUnknown, .detail = shape.error()});
  }
  // launch_fattn<DV, columns, group> with stream-k (fattn-common.cuh:1085-1180).
  const std::int64_t ncols = static_cast<std::int64_t>(plan.columns) * kGroup;
  const std::int64_t ntiles_x = (q->ne[1] + plan.columns - 1) / plan.columns;
  const std::int64_t gqa = q->ne[2] / k->ne[2];
  const std::int64_t ntiles_z = (gqa + kGroup - 1) / kGroup;
  const std::int64_t ntiles_dst = ntiles_x * ntiles_z * k->ne[2] * q->ne[3];
  if (ntiles_dst > INT32_MAX) {
    return Rejected("flash attention beyond the launcher's tile count");
  }
  const auto gathered =
      plan.sparse ? std::min<std::int64_t>(k->ne[1], std::int64_t{plan.columns} * n_kv_max)
                  : k->ne[1];
  if (plan.sparse) {
    // One bounded union and one live count per query tile and mask sequence.
    const auto lists = static_cast<std::uint64_t>(ntiles_x * mask->ne[3]);
    plan.scratch = (static_cast<std::uint64_t>(gathered) + 1) * lists * sizeof(std::int32_t);
    if (q->ne[1] > 2147483647 || mask->ne[3] > 65535) {
      return Rejected("the sparse gather's grid beyond its limits");
    }
  } else if (k->ne[1] % kKqStride == 0 && (q->ne[1] >= 1024 || q->ne[3] > 1)) {
    // The mask pre-pass reads `columns` mask rows per tile, the last tile's
    // included, and one mask per sequence without broadcasting
    // (fattn-common.cuh:664-718), so the mask's rows and sequences must
    // reach them.
    plan.mask_prepass = true;
    if (mask->ne[1] < ntiles_x * plan.columns || mask->ne[3] != q->ne[3] || q->ne[3] > 65535 ||
        ntiles_x > INT32_MAX) {
      return Rejected("the mask pre-pass reads whole column tiles past the mask's rows");
    }
    plan.scratch = static_cast<std::uint64_t>(ntiles_x * q->ne[3]) * sizeof(int);
  }
  const std::int64_t kv = gathered;
  if (kv > INT32_MAX - (shape->kv_batch - 1)) {
    return Rejected("flash attention beyond the pinned KV ceil-division arithmetic");
  }
  const std::int64_t ntiles_kv = (kv + shape->kv_batch - 1) / shape->kv_batch;
  // The pinned int loop advances kbc by iter_k before rounding, and forms
  // kb0_start + kbc_stop before subtracting kbc. Fund both intermediate
  // sums, not just the final total iteration count.
  if (ntiles_dst > INT32_MAX / 100 ||
      ntiles_kv > (static_cast<std::int64_t>(INT32_MAX) + 1) / (ntiles_dst + 1)) {
    return Rejected(
        "flash attention beyond the pinned 32-bit iteration/advance/efficiency arithmetic");
  }
  // Match the pinned launch_fattn whole-tile preference on GB10, including
  // its actual two-stage nonsparse kernel and mask scan.
  const std::int64_t max_blocks = static_cast<std::int64_t>(shape->blocks_per_sm) * device.nsm;
  const std::int64_t waves = (ntiles_dst + max_blocks - 1) / max_blocks;
  const auto efficiency = 100 * ntiles_dst / (max_blocks * waves);
  const bool prefer_whole_tiles =
      cc == GGML_CUDA_CC_DGX_SPARK && shape->async_kv_preload && plan.mask_prepass;
  const bool stream_k =
      !(prefer_whole_tiles && efficiency >= 75) &&
      ((GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_ADA_LOVELACE) || efficiency < 75);
  std::int64_t blocks = ntiles_dst;
  if (stream_k) {
    const std::int64_t raw = std::min(max_blocks, ntiles_kv * ntiles_dst);
    const std::int64_t rounded = raw / ntiles_dst * ntiles_dst;
    const std::int64_t loss = rounded > 0 ? 100 * (raw - rounded) / raw : 100;
    blocks = loss <= 5 ? rounded : raw;
  }
  if (blocks <= 0 || blocks > INT32_MAX) {
    return Rejected("flash attention beyond the launcher's grid");
  }
  plan.blocks = static_cast<int>(blocks);
  if (ntiles_dst % blocks != 0) {
    // The stream-k fixup's partial results (fattn-common.cuh:1134-1136).
    const std::uint64_t meta = static_cast<std::uint64_t>(blocks) *
                               static_cast<std::uint64_t>(ncols) *
                               static_cast<std::uint64_t>(2 + (plan.head / 2)) * sizeof(float2);
    plan.scratch = (plan.scratch > 0 ? Round(plan.scratch) : 0) + meta;
  }
  return plan;
}

std::expected<FlashAttnMmaPlan, KernelFailure> PlanFlashAttnMma(const LaunchContext& launch,
                                                                const ggml_tensor* node,
                                                                bool wide_sparse) {
  return PlanFlashAttnMmaGroup(launch, node, wide_sparse, false);
}
std::expected<FlashAttnMmaPlan, KernelFailure> PlanFlashAttnMmaGqa2(const LaunchContext& launch,
                                                                    const ggml_tensor* node) {
  return PlanFlashAttnMmaGroup(launch, node, false, true);
}
std::expected<void, KernelFailure> FlashAttnMmaGqa2(LaunchContext& launch, ggml_tensor* node) {
  auto plan = PlanFlashAttnMmaGqa2(launch, node);
  if (!plan) return std::unexpected(plan.error());
  const auto run = detail::FlashAttnMmaCaseGqa2(plan->columns);
  if (run == nullptr) return Rejected("no group2 MMA query tile");
  return launch.Run(base::Bytes(plan->scratch),
                    [node, run](ggml_backend_cuda_context& context) { run(context, node); });
}

std::expected<FlashAttnMmaPlan, KernelFailure> PlanFlashAttnMma128(const LaunchContext& launch,
                                                                   const ggml_tensor* node) {
  if (auto checked = CheckFlashAttnMma128(node); !checked) {
    return std::unexpected(checked.error());
  }
  const ggml_tensor* q = node->src[0];
  const ggml_tensor* k = node->src[1];
  const auto& device = ggml_cuda_info().devices[launch.device()];
  const int cc = device.cc;
  if (!GGML_CUDA_CC_IS_NVIDIA(cc) || !turing_mma_available(cc)) {
    return Rejected("the MMA flash-attention kernels need Turing or later");
  }
  FlashAttnMmaPlan plan;
  plan.head = 128;
  plan.group = 1;
  // switch_ncols1 for ncols2 = 1 (fattn.cu:146-166).
  if (q->ne[1] <= 8) {
    plan.columns = 8;
  } else if (q->ne[1] <= 16) {
    plan.columns = 16;
  } else if (q->ne[1] <= 32 || ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_TURING) {
    plan.columns = 32;
  } else {
    plan.columns = 64;
  }
  auto shape = detail::FlashAttnMmaShape128(plan.columns, launch.device());
  if (!shape) {
    return std::unexpected(KernelFailure{.error = KernelError::kUnknown, .detail = shape.error()});
  }
  // launch_fattn<128, columns, 1> with stream-k (fattn-common.cuh:1085-1180);
  // no mask, so no pre-pass.
  const std::int64_t ncols = plan.columns;
  const std::int64_t ntiles_x = (q->ne[1] + plan.columns - 1) / plan.columns;
  const std::int64_t ntiles_dst = ntiles_x * q->ne[2] * q->ne[3];
  if (ntiles_dst > INT32_MAX) {
    return Rejected("flash attention beyond the launcher's tile count");
  }
  if (k->ne[1] > INT32_MAX - (shape->kv_batch - 1)) {
    return Rejected("flash attention beyond the pinned KV ceil-division arithmetic");
  }
  const std::int64_t ntiles_kv = (k->ne[1] + shape->kv_batch - 1) / shape->kv_batch;
  // The pinned int loop advances kbc by iter_k before rounding, and forms
  // kb0_start + kbc_stop before subtracting kbc. Fund both intermediate
  // sums, not just the final total iteration count.
  if (ntiles_dst > INT32_MAX / 100 ||
      ntiles_kv > (static_cast<std::int64_t>(INT32_MAX) + 1) / (ntiles_dst + 1)) {
    return Rejected(
        "flash attention beyond the pinned 32-bit iteration/advance/efficiency arithmetic");
  }
  const std::int64_t max_blocks = static_cast<std::int64_t>(shape->blocks_per_sm) * device.nsm;
  const std::int64_t waves = (ntiles_dst + max_blocks - 1) / max_blocks;
  const auto efficiency = 100 * ntiles_dst / (max_blocks * waves);
  const bool prefer_whole_tiles =
      cc == GGML_CUDA_CC_DGX_SPARK && shape->async_kv_preload && plan.mask_prepass;
  const bool stream_k =
      !(prefer_whole_tiles && efficiency >= 75) &&
      ((GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_ADA_LOVELACE) || efficiency < 75);
  std::int64_t blocks = ntiles_dst;
  if (stream_k) {
    const std::int64_t raw = std::min(max_blocks, ntiles_kv * ntiles_dst);
    const std::int64_t rounded = raw / ntiles_dst * ntiles_dst;
    const std::int64_t loss = rounded > 0 ? 100 * (raw - rounded) / raw : 100;
    blocks = loss <= 5 ? rounded : raw;
  }
  if (blocks <= 0 || blocks > INT32_MAX) {
    return Rejected("flash attention beyond the launcher's grid");
  }
  plan.blocks = static_cast<int>(blocks);
  if (ntiles_dst % blocks != 0) {
    plan.scratch = static_cast<std::uint64_t>(blocks) * static_cast<std::uint64_t>(ncols) *
                   static_cast<std::uint64_t>(2 + (plan.head / 2)) * sizeof(float2);
  }
  return plan;
}

std::expected<void, KernelFailure> FlashAttnMma128(LaunchContext& launch, ggml_tensor* node) {
  auto plan = PlanFlashAttnMma128(launch, node);
  if (!plan) {
    return std::unexpected(plan.error());
  }
  const detail::MmaCase run = detail::FlashAttnMmaCase128(plan->columns);
  if (run == nullptr) {
    return Rejected("no MMA case for this column count");
  }
  return launch.Run(base::Bytes(plan->scratch),
                    [node, run](ggml_backend_cuda_context& context) { run(context, node); });
}

std::expected<void, KernelFailure> FlashAttnMma(LaunchContext& launch, ggml_tensor* node,
                                                bool wide_sparse) {
  auto plan = PlanFlashAttnMma(launch, node, wide_sparse);
  if (!plan) {
    return std::unexpected(plan.error());
  }
  const bool q16 = node->src[0]->type == GGML_TYPE_F16;
  const detail::MmaCase run = plan->head == 512
                                  ? (q16 ? detail::FlashAttnMmaCase512Q16(plan->columns)
                                         : detail::FlashAttnMmaCase512(plan->columns))
                                  : detail::FlashAttnMmaCase256(plan->columns);
  if (run == nullptr) {
    return Rejected("no MMA case for this head size and column count");
  }
  ggml_tensor dispatch = *node;
  dispatch.op_params[kFlashAttnWideSparseParam] = wide_sparse ? 1 : 0;
  // Run consumes this descriptor synchronously while queueing/capturing;
  // device parameters contain tensor addresses, never this host pointer.
  return launch.Run(
      base::Bytes(plan->scratch),
      [&dispatch, run](ggml_backend_cuda_context& context) { run(context, &dispatch); });
}

}  // namespace llmp::kernels::ggml
