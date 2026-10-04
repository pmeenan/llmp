// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: MIT AND Apache-2.0

// Row-invariant quantized vector products (ops_ext.h MulMatVecQRows,
// D-092): GGML's MMVQ kernel (mmvq.cu at llama.cpp b29c606e2, MIT), with
// every output column computed by the arithmetic of GGML's one-column
// launch. Upstream's kernel picks its warps, rows per block and K-loop
// stride from the column count (calc_nwarps, calc_rows_per_block): on the
// GB10 a one-column product of Q8_0, Q4_K, Q5_K or Q6_K weights runs twice
// the warps of a two-to-four-column one, so a column's sums are added in a
// different order and a speculative verify's rows would differ in their
// last bits from the decode steps they stand for. Here the warps, rows per
// block, small-K and halved-iteration choices are always the one-column
// launch's; a block still reads its weight rows once and applies them to
// every column (the dense products), so a k+1-row verify streams the
// weights once. For mul_mat_id every token is its own one-column launch
// over grid z, as upstream's one-token path runs it.
//
// The kernel body, the parameter tables and the host arithmetic are
// upstream's (mmvq.cu:1-146, 452-770, 983-1130, 1404-1535), cut to the
// unfused path; only the column count's part in the launch configuration
// changes. The unit builds with GGML's device flags (-use_fast_math,
// -extended-lambda), as mmvq.cu does, so the same inline vec_dot code
// compiles to the same arithmetic.

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <string>
#include <type_traits>
#include <utility>

#include "base/bytes.h"
#include "common.cuh"
#include "kernels/ggml/ops_ext.h"
#include "kernels/ggml/validate.h"
#include "kernels/ggml/validate_ext.h"
#include "mmvf.cuh"
#include "mmvq.cuh"
#include "quantize.cuh"
#include "vecdotq.cuh"

namespace jitllm::kernels::ggml {
namespace {

// ---- From GGML's mmvq.cu (MIT) ----

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == GGML_CUDA_CC_DGX_SPARK
constexpr __host__ __device__ bool RowsShouldPrefetch(ggml_type type) {
  switch (type) {
    case GGML_TYPE_Q4_0:
    case GGML_TYPE_Q5_0:
    case GGML_TYPE_Q8_0:
    case GGML_TYPE_MXFP4:
    case GGML_TYPE_Q3_K:
    case GGML_TYPE_Q4_K:
    case GGML_TYPE_Q5_K:
    case GGML_TYPE_Q6_K:
    case GGML_TYPE_IQ1_M:
    case GGML_TYPE_IQ4_NL:
    case GGML_TYPE_IQ4_XS:
      return true;
    default:
      return false;
  }
}

__device__ __forceinline__ void RowsPrefetchL2(const void* p) {
  asm volatile("prefetch.global.L2 [%0];" ::"l"(p));
}
#endif

using VecDot = float (*)(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                         const int& kbx, const int& iqs);

// The weight types jitLLM's models bring to vector products.
constexpr __device__ VecDot RowsVecDot(ggml_type type) {
  switch (type) {
    case GGML_TYPE_Q4_0:
      return vec_dot_q4_0_q8_1;
    case GGML_TYPE_Q4_1:
      return vec_dot_q4_1_q8_1;
    case GGML_TYPE_Q5_0:
      return vec_dot_q5_0_q8_1;
    case GGML_TYPE_Q5_1:
      return vec_dot_q5_1_q8_1;
    case GGML_TYPE_IQ4_NL:
      return vec_dot_iq4_nl_q8_1;
    case GGML_TYPE_Q8_0:
      return vec_dot_q8_0_q8_1;
    case GGML_TYPE_Q2_K:
      return vec_dot_q2_K_q8_1;
    case GGML_TYPE_IQ2_XXS:
      return vec_dot_iq2_xxs_q8_1;
    case GGML_TYPE_MXFP4:
      return vec_dot_mxfp4_q8_1;
    case GGML_TYPE_Q4_K:
      return vec_dot_q4_K_q8_1;
    case GGML_TYPE_Q5_K:
      return vec_dot_q5_K_q8_1;
    case GGML_TYPE_Q6_K:
      return vec_dot_q6_K_q8_1;
    case GGML_TYPE_IQ2_XS:
      return vec_dot_iq2_xs_q8_1;
    case GGML_TYPE_IQ3_XXS:
      return vec_dot_iq3_xxs_q8_1;
    default:
      return nullptr;
  }
}

constexpr __host__ __device__ int RowsVdr(ggml_type type) {
  switch (type) {
    case GGML_TYPE_Q4_0:
      return VDR_Q4_0_Q8_1_MMVQ;
    case GGML_TYPE_Q4_1:
      return VDR_Q4_1_Q8_1_MMVQ;
    case GGML_TYPE_Q5_0:
      return VDR_Q5_0_Q8_1_MMVQ;
    case GGML_TYPE_Q5_1:
      return VDR_Q5_1_Q8_1_MMVQ;
    case GGML_TYPE_IQ4_NL:
      return VDR_IQ4_NL_Q8_1_MMVQ;
    case GGML_TYPE_Q2_0:
      return VDR_Q2_0_Q8_1_MMVQ;
    case GGML_TYPE_Q3_K:
      return VDR_Q3_K_Q8_1_MMVQ;
    case GGML_TYPE_IQ2_S:
      return VDR_IQ2_S_Q8_1_MMVQ;
    case GGML_TYPE_IQ3_S:
      return VDR_IQ3_S_Q8_1_MMVQ;
    case GGML_TYPE_IQ4_XS:
      return VDR_IQ4_XS_Q8_1_MMVQ;
    case GGML_TYPE_NVFP4:
      return VDR_NVFP4_Q8_1_MMVQ;
    case GGML_TYPE_IQ1_S:
      return 1;  // pinned get_vdr_mmvq default
    case GGML_TYPE_Q8_0:
      return VDR_Q8_0_Q8_1_MMVQ;
    case GGML_TYPE_Q2_K:
      return VDR_Q2_K_Q8_1_MMVQ;
    case GGML_TYPE_IQ2_XXS:
      return VDR_IQ2_XXS_Q8_1_MMVQ;
    case GGML_TYPE_MXFP4:
      return VDR_MXFP4_Q8_1_MMVQ;
    case GGML_TYPE_Q4_K:
      return VDR_Q4_K_Q8_1_MMVQ;
    case GGML_TYPE_Q5_K:
      return VDR_Q5_K_Q8_1_MMVQ;
    case GGML_TYPE_Q6_K:
      return VDR_Q6_K_Q8_1_MMVQ;
    case GGML_TYPE_IQ2_XS:
      return VDR_IQ2_XS_Q8_1_MMVQ;
    case GGML_TYPE_IQ3_XXS:
      return VDR_IQ3_XXS_Q8_1_MMVQ;
    default:
      return 1;
  }
}

enum RowsTable : std::uint8_t {
  kRowsGeneric = 0,
  kRowsTuring,
  kRowsGcn,
  kRowsRdna2,
  kRowsRdna3,
  kRowsRdna4,
  kRowsGb10
};

constexpr __device__ RowsTable RowsDeviceTable() {
#if defined(RDNA4)
  return kRowsRdna4;
#elif defined(RDNA3_0)
  return kRowsRdna3;
#elif defined(RDNA2) || defined(RDNA3_5)
  return kRowsRdna2;
#elif defined(GCN) || defined(CDNA)
  return kRowsGcn;
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_TURING && \
    __CUDA_ARCH__ < GGML_CUDA_CC_AMPERE
  return kRowsTuring;
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ == GGML_CUDA_CC_DGX_SPARK
  return kRowsGb10;
#else
  return kRowsGeneric;
#endif
}

RowsTable RowsHostTable(int cc) {
  if (GGML_CUDA_CC_IS_RDNA4(cc)) {
    return kRowsRdna4;
  }
  if (GGML_CUDA_CC_IS_RDNA3_0(cc)) {
    return kRowsRdna3;
  }
  if (GGML_CUDA_CC_IS_RDNA2(cc) || GGML_CUDA_CC_IS_RDNA3_5(cc)) {
    return kRowsRdna2;
  }
  if (GGML_CUDA_CC_IS_GCN(cc) || GGML_CUDA_CC_IS_CDNA(cc)) {
    return kRowsGcn;
  }
  if (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_TURING &&
      ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_AMPERE) {
    return kRowsTuring;
  }
  if (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_DGX_SPARK) {
    return kRowsGb10;
  }
  return kRowsGeneric;
}

// calc_nwarps for one column (mmvq.cu:452-560 with ncols_dst = 1).
constexpr __host__ __device__ int RowsWarps(ggml_type type, RowsTable table, bool small_k,
                                            bool halve_iters) {
  switch (table) {
    case kRowsGeneric:
      return 4;
    case kRowsGcn:
      return 2;
    case kRowsRdna4:
      switch (type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ4_XS:
          return 8;
        default:
          return 1;
      }
    case kRowsRdna3:
      switch (type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
          return 8;
        case GGML_TYPE_Q6_K:
          return 2;
        case GGML_TYPE_IQ4_NL:
          return 8;
        default:
          return 1;
      }
    case kRowsTuring:
      switch (type) {
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
          return 2;
        default:
          return 4;
      }
    case kRowsGb10:
      if (!small_k && halve_iters) {
        switch (type) {
          case GGML_TYPE_Q4_0:
          case GGML_TYPE_Q4_1:
          case GGML_TYPE_Q5_0:
          case GGML_TYPE_Q5_1:
          case GGML_TYPE_Q8_0:
          case GGML_TYPE_Q4_K:
          case GGML_TYPE_Q5_K:
          case GGML_TYPE_Q6_K:
          case GGML_TYPE_IQ4_NL:
            return 8;
          default:
            break;
        }
      }
      return 4;
    case kRowsRdna2:
      break;
  }
  return 1;
}

// calc_rows_per_block for one column.
constexpr __host__ __device__ int RowsPerBlock(RowsTable table, bool small_k, int nwarps) {
  if (table == kRowsGeneric || table == kRowsGcn || table == kRowsTuring || table == kRowsGb10) {
    return small_k ? nwarps : 1;
  }
  return 1;
}

// mul_mat_vec_q<type, 1, false, small_k, halve_iters> (mmvq.cu:583-770),
// generalized to `ncols` columns that share each weight read, or with
// `per_token` to one token of a mul_mat_id per grid z. Every column's
// partial sums, their shared-memory reduction and the warp reduction are
// the one-column kernel's.
template <ggml_type type, int ncols, bool small_k, bool halve_iters, bool per_token>
__launch_bounds__(RowsWarps(type, RowsDeviceTable(), small_k, halve_iters) *
                      ggml_cuda_get_physical_warp_size(),
                  1) __global__
    void MulMatVecQRowsKernel(const void* vx_ptr, const void* vy_ptr, const int32_t* ids_ptr,
                              float* dst_ptr, const uint32_t ncols_x, const uint32_t nrows_x,
                              const uint3 nchannels_y, const uint32_t stride_row_x,
                              const uint32_t stride_col_y, const uint32_t stride_col_dst,
                              const uint3 channel_ratio, const uint32_t stride_channel_x,
                              const uint32_t stride_channel_y, const uint32_t stride_channel_dst,
                              const uint3 sample_ratio, const uint32_t stride_sample_x,
                              const uint32_t stride_sample_y, const uint32_t stride_sample_dst,
                              const uint32_t ids_stride) {
  const void* GGML_CUDA_RESTRICT vx = vx_ptr;
  const void* GGML_CUDA_RESTRICT vy = vy_ptr;
  const int32_t* GGML_CUDA_RESTRICT ids = ids_ptr;
  float* GGML_CUDA_RESTRICT dst = dst_ptr;

  constexpr int qk = ggml_cuda_type_traits<type>::qk;
  constexpr int qi = ggml_cuda_type_traits<type>::qi;
  constexpr int vdr = RowsVdr(type);
  constexpr RowsTable table_id = RowsDeviceTable();
  constexpr int nwarps = RowsWarps(type, table_id, small_k, halve_iters);
  constexpr int rows_per_cuda_block = RowsPerBlock(table_id, small_k, nwarps);
  constexpr int warp_size = ggml_cuda_get_physical_warp_size();

  constexpr VecDot vec_dot_q_cuda = RowsVecDot(type);

  const int tid = warp_size * threadIdx.y + threadIdx.x;
  const int row0 = rows_per_cuda_block * blockIdx.x;
  const int blocks_per_row_x = ncols_x / qk;
  constexpr int blocks_per_iter = vdr * nwarps * warp_size / qi;

  const uint32_t channel_dst = blockIdx.y;

  uint32_t channel_x;
  uint32_t channel_y;
  uint32_t sample_dst;

  ggml_cuda_pdl_sync();
  if constexpr (per_token) {
    // One token's one-column launch: its ids, activations and output.
    const uint32_t token = blockIdx.z;
    ids += token * ids_stride;
    vy = static_cast<const block_q8_1*>(vy) + token * stride_col_y;
    dst += token * stride_col_dst;
    channel_x = ids[channel_dst];
    channel_y = fastmodulo(channel_dst, nchannels_y);
    sample_dst = 0;
  } else {
    channel_x = fastdiv(channel_dst, channel_ratio);
    channel_y = channel_dst;
    sample_dst = blockIdx.z;
  }

  const uint32_t sample_x = fastdiv(sample_dst, sample_ratio);
  const uint32_t sample_y = sample_dst;

  // partial sum for each thread
  float tmp[ncols][rows_per_cuda_block] = {{0.0f}};

  const block_q8_1* y =
      ((const block_q8_1*)vy) + sample_y * stride_sample_y + channel_y * stride_channel_y;
  const int kbx_offset =
      sample_x * stride_sample_x + channel_x * stride_channel_x + row0 * stride_row_x;

  for (int kbx = tid / (qi / vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
    const int kby = kbx * (qk / QK8_1);  // y block index that aligns with kbx

    // x block quant index when casting the quants to int
    const int kqs = vdr * (tid % (qi / vdr));

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == GGML_CUDA_CC_DGX_SPARK
    // start the next iterations' weight loads early
    if constexpr (RowsShouldPrefetch(type)) {
      constexpr int pf_dist = 2;  // loop iterations, not blocks
      const int kbx_pf = kbx + pf_dist * blocks_per_iter;
      if (kbx_pf < blocks_per_row_x) {
#pragma unroll
        for (int i = 0; i < rows_per_cuda_block; ++i) {
          if (uint32_t(row0 + i) < nrows_x) {
            const size_t off =
                (size_t)(kbx_offset + i * stride_row_x + kbx_pf) * ggml_cuda_type_traits<type>::bs;
            RowsPrefetchL2((const char*)vx + off);
          }
        }
      }
    }
#endif

#pragma unroll
    for (int j = 0; j < ncols; ++j) {
#pragma unroll
      for (int i = 0; i < rows_per_cuda_block; ++i) {
        if (uint32_t(row0 + i) < nrows_x) {
          tmp[j][i] += vec_dot_q_cuda(vx, &y[j * stride_col_y + kby],
                                      kbx_offset + i * stride_row_x + kbx, kqs);
        }
      }
    }
  }

  __shared__ float tmp_shared[nwarps - 1 > 0 ? nwarps - 1 : 1][ncols][rows_per_cuda_block]
                             [warp_size];

  if (threadIdx.y > 0) {
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
#pragma unroll
      for (int i = 0; i < rows_per_cuda_block; ++i) {
        tmp_shared[threadIdx.y - 1][j][i][threadIdx.x] = tmp[j][i];
      }
    }
  }
  __syncthreads();
  if (threadIdx.y > 0) {
    return;
  }

  dst += sample_dst * stride_sample_dst + channel_dst * stride_channel_dst + row0;

  // sum up partial sums and write back result
#pragma unroll
  for (int j = 0; j < ncols; ++j) {
#pragma unroll
    for (int i = 0; i < rows_per_cuda_block; ++i) {
#pragma unroll
      for (int l = 0; l < nwarps - 1; ++l) {
        tmp[j][i] += tmp_shared[l][j][i][threadIdx.x];
      }
      tmp[j][i] = warp_reduce_sum<warp_size>(tmp[j][i]);

      if (threadIdx.x == i && uint32_t(row0 + i) < nrows_x) {
        dst[j * stride_col_dst + i] = tmp[j][i];
      }
    }
  }
}

// ---- jitLLM ----

constexpr std::uint64_t kBlock = 256;  // the pool's block boundary (launch.h)

std::unexpected<KernelFailure> Rejected(std::string detail) {
  return std::unexpected(
      KernelFailure{.error = KernelError::kRejected, .detail = std::move(detail)});
}

bool RowsType(ggml_type type) {
  switch (type) {
    case GGML_TYPE_Q4_0:
    case GGML_TYPE_Q4_1:
    case GGML_TYPE_Q5_0:
    case GGML_TYPE_Q5_1:
    case GGML_TYPE_IQ4_NL:
    case GGML_TYPE_Q2_K:
    case GGML_TYPE_IQ2_XXS:
    case GGML_TYPE_Q8_0:
    case GGML_TYPE_MXFP4:
    case GGML_TYPE_Q4_K:
    case GGML_TYPE_Q5_K:
    case GGML_TYPE_Q6_K:
    case GGML_TYPE_IQ2_XS:
    case GGML_TYPE_IQ3_XXS:
      return true;
    default:
      return false;
  }
}

// What the launch passes the kernel, as ggml_cuda_mul_mat_vec_q computes it
// (mmvq.cu:1484-1535).
struct RowsArgs {
  const void* vx = nullptr;
  const void* vy = nullptr;
  const int32_t* ids = nullptr;
  float* dst = nullptr;
  int ncols_x = 0;
  int nrows_x = 0;
  int columns = 0;  // dense: activation columns; per token: tokens
  int stride_row_x = 0;
  int stride_col_y = 0;
  int stride_col_dst = 0;
  int nchannels_x = 0;
  int nchannels_y = 0;
  int nchannels_dst = 0;
  int stride_channel_x = 0;
  int stride_channel_y = 0;
  int stride_channel_dst = 0;
  int nsamples_x = 0;
  int nsamples_dst = 0;
  int stride_sample_x = 0;
  int stride_sample_y = 0;
  int stride_sample_dst = 0;
  int ids_stride = 0;
};

template <ggml_type type, int ncols, bool small_k, bool halve_iters, bool per_token>
void LaunchRows(const RowsArgs& a, RowsTable table, int warp_size, cudaStream_t stream) {
  const int nwarps = RowsWarps(type, table, small_k, halve_iters);
  const int rpb = RowsPerBlock(table, small_k, nwarps);
  const int64_t nblocks = (a.nrows_x + rpb - 1) / rpb;
  const dim3 block_nums(static_cast<unsigned>(nblocks), static_cast<unsigned>(a.nchannels_dst),
                        static_cast<unsigned>(per_token ? a.columns : a.nsamples_dst));
  const dim3 block_dims(static_cast<unsigned>(warp_size), static_cast<unsigned>(nwarps), 1);
  const uint3 nchannels_y_fd =
      per_token ? init_fastdiv_values(static_cast<uint64_t>(a.nchannels_y)) : make_uint3(0, 0, 0);
  const uint3 channel_ratio_fd =
      per_token ? make_uint3(0, 0, 0)
                : init_fastdiv_values(static_cast<uint64_t>(a.nchannels_dst / a.nchannels_x));
  const uint3 sample_ratio_fd =
      init_fastdiv_values(static_cast<uint64_t>(a.nsamples_dst / a.nsamples_x));
  const ggml_cuda_kernel_launch_params params(block_nums, block_dims, 0, stream);
  ggml_cuda_kernel_launch(
      MulMatVecQRowsKernel<type, ncols, small_k, halve_iters, per_token>, params, a.vx, a.vy, a.ids,
      a.dst, static_cast<uint32_t>(a.ncols_x), static_cast<uint32_t>(a.nrows_x), nchannels_y_fd,
      static_cast<uint32_t>(a.stride_row_x), static_cast<uint32_t>(a.stride_col_y),
      static_cast<uint32_t>(a.stride_col_dst), channel_ratio_fd,
      static_cast<uint32_t>(a.stride_channel_x), static_cast<uint32_t>(a.stride_channel_y),
      static_cast<uint32_t>(a.stride_channel_dst), sample_ratio_fd,
      static_cast<uint32_t>(a.stride_sample_x), static_cast<uint32_t>(a.stride_sample_y),
      static_cast<uint32_t>(a.stride_sample_dst), static_cast<uint32_t>(a.ids_stride));
}

template <ggml_type type, bool small_k, bool halve_iters>
void SwitchColumns(const RowsArgs& a, bool per_token, RowsTable table, int warp_size,
                   cudaStream_t stream) {
  if (per_token) {
    LaunchRows<type, 1, small_k, halve_iters, true>(a, table, warp_size, stream);
    return;
  }
  switch (a.columns) {
    case 1:
      LaunchRows<type, 1, small_k, halve_iters, false>(a, table, warp_size, stream);
      break;
    case 2:
      LaunchRows<type, 2, small_k, halve_iters, false>(a, table, warp_size, stream);
      break;
    case 3:
      LaunchRows<type, 3, small_k, halve_iters, false>(a, table, warp_size, stream);
      break;
    case 4:
      LaunchRows<type, 4, small_k, halve_iters, false>(a, table, warp_size, stream);
      break;
    case 5:
      LaunchRows<type, 5, small_k, halve_iters, false>(a, table, warp_size, stream);
      break;
    case 6:
      LaunchRows<type, 6, small_k, halve_iters, false>(a, table, warp_size, stream);
      break;
    case 7:
      LaunchRows<type, 7, small_k, halve_iters, false>(a, table, warp_size, stream);
      break;
    default:
      LaunchRows<type, 8, small_k, halve_iters, false>(a, table, warp_size, stream);
      break;
  }
}

// Pinned should_use_small_k(1); shared by the original MMVQ read-footprint
// plan and our guarded row-invariant launch, so eligibility cannot drift.
bool SmallK(ggml_type type, RowsTable table, int cc, int blocks_per_row_x,
            int blocks_per_iter_1warp) {
  bool small_k = false;
  {
    const int nwarps = RowsWarps(type, table, false, false);
    small_k = nwarps > 1 && blocks_per_row_x < nwarps * blocks_per_iter_1warp;
    const bool is_nvidia_turing_plus = GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_TURING;
    const bool is_nvidia_pascal_older = GGML_CUDA_CC_IS_NVIDIA(cc) && cc < GGML_CUDA_CC_VOLTA;
    const bool iq_slow_turing = type == GGML_TYPE_IQ3_XXS || type == GGML_TYPE_IQ3_S;
    const bool iq_slow_other = type == GGML_TYPE_IQ1_S || type == GGML_TYPE_IQ1_M ||
                               type == GGML_TYPE_IQ2_XXS || type == GGML_TYPE_IQ2_XS ||
                               type == GGML_TYPE_IQ2_S || type == GGML_TYPE_IQ3_XXS ||
                               type == GGML_TYPE_IQ3_S || type == GGML_TYPE_IQ4_XS;
    const bool slow_pascal =
        type == GGML_TYPE_IQ3_S || type == GGML_TYPE_Q2_K || type == GGML_TYPE_Q3_K;
    if (is_nvidia_turing_plus) {
      if (iq_slow_turing) {
        small_k = false;
      }
    } else if (iq_slow_other || (is_nvidia_pascal_older && slow_pascal) ||
               GGML_CUDA_CC_IS_RDNA(cc)) {
      small_k = false;
    }
  }
  return small_k;
}

// mul_mat_vec_q_switch_ncols_dst's one-column case (mmvq.cu:1068-1130):
// the small-K and halved-iteration choices as ncols_dst = 1 makes them.
template <ggml_type type>
void LaunchType(const RowsArgs& a, bool per_token, int cc, int warp_size, cudaStream_t stream) {
  const RowsTable table = RowsHostTable(cc);
  constexpr int qk = ggml_cuda_type_traits<type>::qk;
  constexpr int qi = ggml_cuda_type_traits<type>::qi;
  constexpr int vdr = RowsVdr(type);
  const int blocks_per_row_x = a.ncols_x / qk;
  const int blocks_per_iter_1warp = vdr * warp_size / qi;

  const bool small_k = SmallK(type, table, cc, blocks_per_row_x, blocks_per_iter_1warp);
  // should_halve_iters: the GB10's wider block, never with ids.
  bool halve_iters = false;
  if (table == kRowsGb10 && !per_token) {
    const int blocks_per_iter = RowsWarps(type, table, false, false) * blocks_per_iter_1warp;
    const int iters = (blocks_per_row_x + blocks_per_iter - 1) / blocks_per_iter;
    const int iters_wide = (blocks_per_row_x + (blocks_per_iter * 2) - 1) / (blocks_per_iter * 2);
    const int idle = (iters_wide * 2) - iters;
    halve_iters = idle * 8 <= iters_wide * 2;
  }
  // Types the table does not promote would compile a second, identical
  // kernel (c_promoted).
  constexpr bool promoted =
      RowsWarps(type, kRowsGb10, false, true) != RowsWarps(type, kRowsGb10, false, false);
  if (small_k) {
    SwitchColumns<type, true, false>(a, per_token, table, warp_size, stream);
  } else if (halve_iters && promoted) {
    SwitchColumns<type, false, promoted>(a, per_token, table, warp_size, stream);
  } else {
    SwitchColumns<type, false, false>(a, per_token, table, warp_size, stream);
  }
}

void LaunchSwitchType(ggml_type type, const RowsArgs& a, bool per_token, int cc, int warp_size,
                      cudaStream_t stream) {
  switch (type) {
    case GGML_TYPE_Q4_0:
      LaunchType<GGML_TYPE_Q4_0>(a, per_token, cc, warp_size, stream);
      break;
    case GGML_TYPE_Q4_1:
      LaunchType<GGML_TYPE_Q4_1>(a, per_token, cc, warp_size, stream);
      break;
    case GGML_TYPE_Q5_0:
      LaunchType<GGML_TYPE_Q5_0>(a, per_token, cc, warp_size, stream);
      break;
    case GGML_TYPE_Q5_1:
      LaunchType<GGML_TYPE_Q5_1>(a, per_token, cc, warp_size, stream);
      break;
    case GGML_TYPE_IQ4_NL:
      LaunchType<GGML_TYPE_IQ4_NL>(a, per_token, cc, warp_size, stream);
      break;
    case GGML_TYPE_Q8_0:
      LaunchType<GGML_TYPE_Q8_0>(a, per_token, cc, warp_size, stream);
      break;
    case GGML_TYPE_Q2_K:
      LaunchType<GGML_TYPE_Q2_K>(a, per_token, cc, warp_size, stream);
      break;
    case GGML_TYPE_IQ2_XXS:
      LaunchType<GGML_TYPE_IQ2_XXS>(a, per_token, cc, warp_size, stream);
      break;
    case GGML_TYPE_MXFP4:
      LaunchType<GGML_TYPE_MXFP4>(a, per_token, cc, warp_size, stream);
      break;
    case GGML_TYPE_Q4_K:
      LaunchType<GGML_TYPE_Q4_K>(a, per_token, cc, warp_size, stream);
      break;
    case GGML_TYPE_Q5_K:
      LaunchType<GGML_TYPE_Q5_K>(a, per_token, cc, warp_size, stream);
      break;
    case GGML_TYPE_Q6_K:
      LaunchType<GGML_TYPE_Q6_K>(a, per_token, cc, warp_size, stream);
      break;
    case GGML_TYPE_IQ2_XS:
      LaunchType<GGML_TYPE_IQ2_XS>(a, per_token, cc, warp_size, stream);
      break;
    case GGML_TYPE_IQ3_XXS:
      LaunchType<GGML_TYPE_IQ3_XXS>(a, per_token, cc, warp_size, stream);
      break;
    default:
      break;  // refused by the plan
  }
}

template <ggml_type type>
int OriginalRowBlock(int k, int cc, int warp_size) {
  const RowsTable table = RowsHostTable(cc);
  const int blocks = k / ggml_cuda_type_traits<type>::qk;
  const int per_warp = RowsVdr(type) * warp_size / ggml_cuda_type_traits<type>::qi;
  const bool small = SmallK(type, table, cc, blocks, per_warp);
  // A halved-iteration GB10 launch is considered only when !small_k,
  // whose row block is one regardless of the promoted warp count.
  return RowsPerBlock(table, small, RowsWarps(type, table, small, false));
}

}  // namespace

int MmvqRowsPerBlock(const LaunchContext& launch, const ggml_tensor* node) {
  const auto& device = ggml_cuda_info().devices[launch.device()];
  const auto* w = node->src[0];
  const bool routed = node->op == GGML_OP_MUL_MAT_ID;
  const auto columns = routed ? node->ne[2] : node->src[1]->ne[1];
  if (columns > 1) {
    // The pinned dedicated MoE multi-token kernel uses two rows; dense
    // ncols2..8 uses two for generic/Turing/GB10 tables, one for RDNA.
    const auto table = RowsHostTable(device.cc);
    return routed || table == kRowsGeneric || table == kRowsGcn || table == kRowsTuring ||
                   table == kRowsGb10
               ? 2
               : 1;
  }
  switch (w->type) {
    case GGML_TYPE_Q4_0:
      return OriginalRowBlock<GGML_TYPE_Q4_0>(static_cast<int>(w->ne[0]), device.cc,
                                              device.warp_size);
    case GGML_TYPE_Q4_1:
      return OriginalRowBlock<GGML_TYPE_Q4_1>(static_cast<int>(w->ne[0]), device.cc,
                                              device.warp_size);
    case GGML_TYPE_Q5_0:
      return OriginalRowBlock<GGML_TYPE_Q5_0>(static_cast<int>(w->ne[0]), device.cc,
                                              device.warp_size);
    case GGML_TYPE_Q5_1:
      return OriginalRowBlock<GGML_TYPE_Q5_1>(static_cast<int>(w->ne[0]), device.cc,
                                              device.warp_size);
    case GGML_TYPE_IQ4_NL:
      return OriginalRowBlock<GGML_TYPE_IQ4_NL>(static_cast<int>(w->ne[0]), device.cc,
                                                device.warp_size);
    case GGML_TYPE_Q8_0:
      return OriginalRowBlock<GGML_TYPE_Q8_0>(static_cast<int>(w->ne[0]), device.cc,
                                              device.warp_size);
    case GGML_TYPE_Q2_K:
      return OriginalRowBlock<GGML_TYPE_Q2_K>(static_cast<int>(w->ne[0]), device.cc,
                                              device.warp_size);
    case GGML_TYPE_IQ2_XXS:
      return OriginalRowBlock<GGML_TYPE_IQ2_XXS>(static_cast<int>(w->ne[0]), device.cc,
                                                 device.warp_size);
    case GGML_TYPE_MXFP4:
      return OriginalRowBlock<GGML_TYPE_MXFP4>(static_cast<int>(w->ne[0]), device.cc,
                                               device.warp_size);
    case GGML_TYPE_Q4_K:
      return OriginalRowBlock<GGML_TYPE_Q4_K>(static_cast<int>(w->ne[0]), device.cc,
                                              device.warp_size);
    case GGML_TYPE_Q5_K:
      return OriginalRowBlock<GGML_TYPE_Q5_K>(static_cast<int>(w->ne[0]), device.cc,
                                              device.warp_size);
    case GGML_TYPE_Q6_K:
      return OriginalRowBlock<GGML_TYPE_Q6_K>(static_cast<int>(w->ne[0]), device.cc,
                                              device.warp_size);
    case GGML_TYPE_IQ2_XS:
      return OriginalRowBlock<GGML_TYPE_IQ2_XS>(static_cast<int>(w->ne[0]), device.cc,
                                                device.warp_size);
    case GGML_TYPE_IQ3_XXS:
      return OriginalRowBlock<GGML_TYPE_IQ3_XXS>(static_cast<int>(w->ne[0]), device.cc,
                                                 device.warp_size);
    case GGML_TYPE_Q2_0:
      return OriginalRowBlock<GGML_TYPE_Q2_0>(static_cast<int>(w->ne[0]), device.cc,
                                              device.warp_size);
    case GGML_TYPE_Q3_K:
      return OriginalRowBlock<GGML_TYPE_Q3_K>(static_cast<int>(w->ne[0]), device.cc,
                                              device.warp_size);
    case GGML_TYPE_IQ1_S:
      return OriginalRowBlock<GGML_TYPE_IQ1_S>(static_cast<int>(w->ne[0]), device.cc,
                                               device.warp_size);
    case GGML_TYPE_IQ2_S:
      return OriginalRowBlock<GGML_TYPE_IQ2_S>(static_cast<int>(w->ne[0]), device.cc,
                                               device.warp_size);
    case GGML_TYPE_IQ3_S:
      return OriginalRowBlock<GGML_TYPE_IQ3_S>(static_cast<int>(w->ne[0]), device.cc,
                                               device.warp_size);
    case GGML_TYPE_IQ4_XS:
      return OriginalRowBlock<GGML_TYPE_IQ4_XS>(static_cast<int>(w->ne[0]), device.cc,
                                                device.warp_size);
    case GGML_TYPE_NVFP4:
      return OriginalRowBlock<GGML_TYPE_NVFP4>(static_cast<int>(w->ne[0]), device.cc,
                                               device.warp_size);
    default:
      return 0;  // the ordinary plan rejects unknown geometry
  }
}

std::expected<std::uint64_t, KernelFailure> PlanMulMatVecQRows(const LaunchContext& launch,
                                                               const ggml_tensor* node) {
  (void)launch;
  if (node == nullptr || (node->op != GGML_OP_MUL_MAT && node->op != GGML_OP_MUL_MAT_ID)) {
    return Rejected("not a quantized product");
  }
  if (auto checked = node->op == GGML_OP_MUL_MAT_ID ? CheckMulMatIdQ(node) : CheckMulMatQ(node);
      !checked) {
    return std::unexpected(checked.error());
  }
  const ggml_tensor* weights = node->src[0];
  const ggml_tensor* input = node->src[1];
  if (!RowsType(weights->type)) {
    return Rejected(std::format("no row-invariant vector product for {} weights",
                                ggml_type_name(weights->type)));
  }
  if (node->op == GGML_OP_MUL_MAT_ID) {
    // Tokens over grid z, one sample; the selected experts over grid y.
    if (node->ne[2] < 1 || node->ne[2] > kRowsMaxColumns || node->ne[3] != 1 || input->ne[3] != 1 ||
        node->ne[1] > 65535) {
      return Rejected("a row-invariant expert product of 1 to 8 tokens, one sample");
    }
  } else if (input->ne[1] < 1 || input->ne[1] > kRowsMaxColumns || node->ne[2] > 65535 ||
             node->ne[3] > 65535) {
    return Rejected("a row-invariant vector product of 1 to 8 columns");
  }
  if (input->ne[2] * input->ne[3] > 65535) {
    return Rejected("the activations' quantization beyond its grid");
  }
  // The Q8_1 activations, as ggml_cuda_mul_mat_vec_q draws them.
  const std::int64_t padded = GGML_PAD(input->ne[0], MATRIX_ROW_PADDING);
  const auto bytes =
      static_cast<std::uint64_t>(input->ne[3] * input->ne[2] * input->ne[1] * padded) *
      sizeof(block_q8_1) / QK8_1;
  return (bytes + kBlock - 1) / kBlock * kBlock;
}

std::expected<void, KernelFailure> MulMatVecQRows(LaunchContext& launch, ggml_tensor* node) {
  auto scratch = PlanMulMatVecQRows(launch, node);
  if (!scratch) {
    return std::unexpected(scratch.error());
  }
  const int device = launch.device();
  return launch.Run(base::Bytes(*scratch), [node, device](ggml_backend_cuda_context& context) {
    const ggml_tensor* src0 = node->src[0];
    const ggml_tensor* src1 = node->src[1];
    const ggml_tensor* ids = node->op == GGML_OP_MUL_MAT_ID ? node->src[2] : nullptr;
    const ggml_tensor* dst = node;
    cudaStream_t stream = context.stream();
    const auto elements = [](std::size_t stride, const ggml_tensor* t) {
      return static_cast<std::int64_t>(stride / ggml_type_size(t->type));
    };
    const std::int64_t ne10 = src1->ne[0];
    const std::int64_t ne11 = src1->ne[1];
    const std::int64_t ne12 = src1->ne[2];
    const std::int64_t ne13 = src1->ne[3];
    const std::int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    ggml_cuda_pool_alloc<char> src1_q8_1(
        context.pool(),
        static_cast<std::size_t>(ne13 * ne12 * ne11 * ne10_padded) * sizeof(block_q8_1) / QK8_1);
    quantize_row_q8_1_cuda(static_cast<const float*>(src1->data), nullptr, src1_q8_1.get(),
                           src0->type, ne10, elements(src1->nb[1], src1),
                           elements(src1->nb[2], src1), elements(src1->nb[3], src1), ne10_padded,
                           ne11, ne12, ne13, stream);
    const std::int64_t s01 = elements(src0->nb[1], src0);
    const std::int64_t s11 = ne10_padded / QK8_1;
    const std::int64_t s1 = elements(dst->nb[1], dst);
    const std::int64_t s02 = elements(src0->nb[2], src0);
    const std::int64_t s2 = elements(dst->nb[2], dst);
    const std::int64_t s03 = elements(src0->nb[3], src0);
    const std::int64_t s3 = elements(dst->nb[3], dst);
    const std::int64_t s12 = ne11 * s11;
    const std::int64_t s13 = ne12 * s12;
    const bool has_ids = ids != nullptr;

    RowsArgs a;
    a.vx = src0->data;
    a.vy = src1_q8_1.get();
    a.ids = has_ids ? static_cast<const int32_t*>(ids->data) : nullptr;
    a.dst = static_cast<float*>(node->data);
    a.ncols_x = static_cast<int>(src0->ne[0]);
    a.nrows_x = static_cast<int>(src0->ne[1]);
    a.columns = static_cast<int>(has_ids ? dst->ne[2] : dst->ne[1]);
    a.stride_row_x = static_cast<int>(s01);
    a.stride_col_y = static_cast<int>(has_ids ? s12 : s11);
    a.stride_col_dst = static_cast<int>(has_ids ? s2 : s1);
    a.nchannels_x = static_cast<int>(src0->ne[2]);
    a.nchannels_y = static_cast<int>(has_ids ? ne11 : ne12);
    a.nchannels_dst = static_cast<int>(has_ids ? dst->ne[1] : dst->ne[2]);
    a.stride_channel_x = static_cast<int>(s02);
    a.stride_channel_y = static_cast<int>(has_ids ? s11 : s12);
    a.stride_channel_dst = static_cast<int>(has_ids ? s1 : s2);
    a.nsamples_x = static_cast<int>(src0->ne[3]);
    a.nsamples_dst = static_cast<int>(dst->ne[3]);
    a.stride_sample_x = static_cast<int>(s03);
    a.stride_sample_y = static_cast<int>(s13);
    a.stride_sample_dst = static_cast<int>(s3);
    a.ids_stride = has_ids ? static_cast<int>(elements(ids->nb[1], ids)) : 0;
    const auto& info = ggml_cuda_info().devices[device];
    LaunchSwitchType(src0->type, a, has_ids, info.cc, info.warp_size, stream);
  });
}

std::expected<void, KernelFailure> MulMatVecFRows(LaunchContext& launch, ggml_tensor* node) {
  if (auto checked = CheckMulMat(node); !checked) {
    return checked;
  }
  const ggml_tensor* weights = node->src[0];
  const ggml_tensor* input = node->src[1];
  if (weights->type != GGML_TYPE_F32 && weights->type != GGML_TYPE_F16 &&
      weights->type != GGML_TYPE_BF16) {
    return Rejected("the float vector kernel takes F32, F16 or BF16 weights");
  }
  if (input->ne[1] < 1 || input->ne[1] > kRowsMaxColumns) {
    return Rejected("the float vector kernel takes 1 to 8 columns");
  }
  // ggml_cuda_should_use_mmvf's layout conditions (mmvf.cu:792-807), which
  // its launcher asserts, without its column thresholds.
  const std::size_t ts = ggml_type_size(weights->type);
  bool aligned = weights->ne[0] % 2 == 0 && weights->nb[0] == ts;
  for (std::size_t i = 1; i < GGML_MAX_DIMS; ++i) {
    aligned = aligned && weights->nb[i] % (2 * ts) == 0;
  }
  if (!aligned) {
    return Rejected("weights the float vector kernel cannot read in pairs");
  }
  (void)launch;
  return launch.Run(base::Bytes(0), [node](ggml_backend_cuda_context& context) {
    ggml_cuda_mul_mat_vec_f(context, node->src[0], node->src[1], nullptr, node);
  });
}

}  // namespace jitllm::kernels::ggml
