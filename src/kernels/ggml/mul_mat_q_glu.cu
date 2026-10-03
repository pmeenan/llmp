// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: MIT AND Apache-2.0

// A copy of ggml mmq.cuh's compact-expert branch of mul_mat_q and of
// mul_mat_q_process_tile, specialized to the occupancy-two J64 IQ2_XXS and
// IQ2_XS pair products (patch 0005), with one change: the MMA write-back applies
// unary.cuh's ggml_cuda_op_swiglu_clamp_single to the separately written
// gate value and this product, and stores the activation instead of the up
// product. Products, accumulation order and the activation's arithmetic are
// the separate kernels'; built with the same -use_fast_math as GGML.

#include <cuda_runtime.h>

#include <cstdint>

#include "common.cuh"
#include "kernels/ggml/mul_mat_q_glu.cuh"
#include "kernels/ggml/validate_ext.h"
#include "mmq.cuh"
#include "unary.cuh"

namespace jitllm::kernels::ggml {
namespace {

constexpr bool kFallback = false;

// The launch the pair's gate product takes: for IQ2_XXS (the community
// GGUF's gate/up experts) patch 0005's occupancy-two J64 product, for
// IQ2_XS (UD-Q2_K_XL's) GGML's ordinary J128 compact product.
template <ggml_type kType>
constexpr int kJOf = kType == GGML_TYPE_IQ2_XXS ? 64 : 128;
template <ggml_type kType>
constexpr int kOccupancyOf =
    kType == GGML_TYPE_IQ2_XXS ? 2 : ggml_cuda_mmq_get_occupancy(kType, kJOf<kType>, kFallback);

template <ggml_type kType, bool kQ8, int kJ = kJOf<kType>>
__launch_bounds__(ggml_cuda_mmq_get_nthreads(kType, kJ, kFallback), kOccupancyOf<kType>) __global__
    void Iq2PairGluUpKernel(const char* __restrict__ x, const int* __restrict__ y,
                            const int32_t* __restrict__ ids_dst,
                            const int32_t* __restrict__ expert_bounds, float* __restrict__ dst,
                            const float* __restrict__ gate, const float limit,
                            const uint3 blocks_per_ne00, const int nrows_x, const int stride_row_x,
                            const int ncols_y, const int stride_col_dst, const uint3 channel_ratio,
                            const int stride_channel_x, const int2* __restrict__ expert_tiles,
                            void* __restrict__ q8, const int q8_ncols) {
#if defined(TURING_MMA_AVAILABLE)
  constexpr int warp_size = ggml_cuda_get_physical_warp_size();
  constexpr int nwarps = ggml_cuda_mmq_get_nthreads(kType, kJ, kFallback) / warp_size;
  constexpr int qk = ggml_cuda_type_traits<kType>::qk;
  constexpr int I = ggml_cuda_mmq_get_I(kType, kJ, kFallback);
  constexpr ggml_cuda_mmq_load_tiles_t load_tiles =
      ggml_cuda_mmq_get_load_tiles<kType, kJ, kFallback>();
  constexpr ggml_cuda_mmq_vec_dot_t vec_dot = ggml_cuda_mmq_get_vec_dot<kType, kJ, kFallback>();

  extern __shared__ int ids_dst_shared[];
#pragma unroll
  for (int j0 = 0; j0 < kJ; j0 += nwarps * warp_size) {
    const int j = j0 + threadIdx.y * warp_size + threadIdx.x;
    if (j0 + nwarps * warp_size > kJ && j >= kJ) {
      break;
    }
    ids_dst_shared[j] = j;
  }
  __syncthreads();

  const int2 tile = expert_tiles[blockIdx.y];
  if (tile.x < 0) {
    return;
  }
  const int zt = tile.x;
  const int jt = tile.y;
  const int it = blockIdx.x;

  const int col_low = expert_bounds[zt + 0];
  const int col_high = expert_bounds[zt + 1];
  const int col_diff = col_high - col_low;
  if (jt * kJ >= col_diff) {
    return;
  }
#pragma unroll
  for (int j0 = 0; j0 < kJ; j0 += nwarps * warp_size) {
    const int j = j0 + threadIdx.y * warp_size + threadIdx.x;
    if (j0 + nwarps * warp_size > kJ && j >= kJ) {
      break;
    }
    ids_dst_shared[j] = ids_dst[col_low + jt * kJ + j];
  }
  __syncthreads();

  const int offset_y = (col_low + jt * kJ) * (sizeof(block_q8_1_mmq) / sizeof(int));
  const int offset_dst = it * I;
  const int tile_y_max_j = col_diff - jt * kJ - 1;
  const int offset_x = fastdiv(zt, channel_ratio) * stride_channel_x + it * I * stride_row_x;
  const int tile_x_max_i = nrows_x - it * I - 1;
  y += offset_y;
  dst += offset_dst;
  gate += offset_dst;

  // mul_mat_q_process_tile<kType, kJ, false, false>, kb0 0..blocks_per_ne00.
  int* tile_y = ids_dst_shared + kJ;
  int* tile_x = tile_y + GGML_PAD(kJ * MMQ_TILE_Y_K, nwarps * warp_size);
  constexpr int ne_block = QK8_1_MMQ;
  constexpr int ITER_K = ggml_cuda_mmq_get_K_vram(kType, kJ, kFallback);
  constexpr int blocks_per_iter = ITER_K / qk;
  float sum[kJ * I / (nwarps * warp_size)] = {0.0f};
  constexpr int sz = sizeof(block_q8_1_mmq) / sizeof(int);
  const int kb0_stop = static_cast<int>(blocks_per_ne00.z);
  for (int kb0 = 0; kb0 < kb0_stop; kb0 += blocks_per_iter) {
    load_tiles(x, tile_x, offset_x + kb0, tile_x_max_i, stride_row_x);
    {
      const int* by0 = y + ncols_y * (kb0 * qk / ne_block) * sz;
#pragma unroll
      for (int l0 = 0; l0 < kJ * MMQ_TILE_Y_K; l0 += nwarps * warp_size) {
        const int l = l0 + threadIdx.y * warp_size + threadIdx.x;
        tile_y[l] = by0[l];
      }
    }
    __syncthreads();
    vec_dot(tile_x, tile_y, sum, 0);
    __syncthreads();
    {
      const int* by0 = y + ncols_y * ((kb0 * qk / ne_block) * sz + sz);
#pragma unroll
      for (int l0 = 0; l0 < kJ * MMQ_TILE_Y_K; l0 += nwarps * warp_size) {
        const int l = l0 + threadIdx.y * warp_size + threadIdx.x;
        tile_y[l] = by0[l];
      }
    }
    __syncthreads();
    vec_dot(tile_x, tile_y, sum, MMQ_TILE_NE_K);
    __syncthreads();
  }

  // ggml_cuda_mmq_write_back_mma's layout, storing the activation: to the
  // destination, or (kQ8) to shared memory, column-major, for quantization.
  using tile_C = ggml_cuda_mma::tile<16, 8, int>;
  constexpr int rows_per_warp = ggml_cuda_mmq_get_rows_per_warp(kType, kJ, kFallback);
  constexpr int ntx = rows_per_warp / tile_C::I;
  constexpr int ld = I + 4;  // the staged column stride, float4-aligned
  float* stage = reinterpret_cast<float*>(tile_y);
  static_assert(!kQ8 || (I % 64 == 0 && kJ % 2 == 0));
  const int i0 = (threadIdx.y / ntx) * (ntx * tile_C::I);
#pragma unroll
  for (int j0 = 0; j0 < kJ; j0 += ntx * tile_C::J) {
#pragma unroll
    for (int n = 0; n < ntx; ++n) {
#pragma unroll
      for (int l = 0; l < tile_C::ne; ++l) {
        const int j = j0 + (threadIdx.y % ntx) * tile_C::J + tile_C::get_j(l);
        if (j > tile_y_max_j) {
          continue;
        }
        const int i = i0 + n * tile_C::I + tile_C::get_i(l);
        const int k = ids_dst_shared[j] * stride_col_dst + i;
        const float a = ggml_cuda_op_swiglu_clamp_single(
            gate[k], sum[(j0 / tile_C::J + n) * tile_C::ne + l], limit);
        if constexpr (kQ8) {
          stage[(j * ld) + i] = a;
        } else {
          dst[k] = a;
        }
      }
    }
  }
  if constexpr (kQ8) {
    // quantize.cu's quantize_mmq_q8_1<D2S6, false> of each activation
    // column, written at its sorted (compact) column: each half warp takes
    // 64 consecutive values, four per lane, as the quantizer's lanes do.
    __syncthreads();
    block_q8_1_mmq* q8_blocks = static_cast<block_q8_1_mmq*>(q8);
    const int lane = static_cast<int>(threadIdx.x);
    const int half = lane / 16;
    constexpr int chunks = I / 64;
    // Every pass covers whole half warps of real (column, chunk) units: no
    // pass reaches past the tile's kJ staged columns.
    static_assert((kJ * chunks) % (nwarps * 2) == 0);
    for (int base = 0; base < kJ * chunks; base += nwarps * 2) {
      const int u = base + (static_cast<int>(threadIdx.y) * 2) + half;
      const int j = u / chunks;
      const int i = ((u % chunks) * 64) + ((lane % 16) * 4);
      const bool active = j <= tile_y_max_j;
      const float4 xi = active ? *reinterpret_cast<const float4*>(stage + (j * ld) + i)
                               : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      float amax = fabsf(xi.x);
      amax = fmaxf(amax, fabsf(xi.y));
      amax = fmaxf(amax, fabsf(xi.z));
      amax = fmaxf(amax, fabsf(xi.w));
#pragma unroll
      for (int offset = 64 / 8; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset, WARP_SIZE));
      }
      float s = xi.x + xi.y + xi.z + xi.w;
#pragma unroll
      for (int offset = 16 / 8; offset > 0; offset >>= 1) {
        s += __shfl_xor_sync(0xFFFFFFFF, s, offset, WARP_SIZE);
      }
      const float d_inv = 127.0f / amax;
      char4 q;
      q.x = roundf(xi.x * d_inv);
      q.y = roundf(xi.y * d_inv);
      q.z = roundf(xi.z * d_inv);
      q.w = roundf(xi.w * d_inv);
      const float d = 1.0f / d_inv;
      if (active) {
        const int feature = (it * I) + i;
        const int k_block = feature / QK8_1_MMQ;
        const int iqs = feature % QK8_1_MMQ;
        const int64_t ib = (static_cast<int64_t>(k_block) * q8_ncols) + col_low + (jt * kJ) + j;
        char4* yqs4 = reinterpret_cast<char4*>(q8_blocks[ib].qs);
        yqs4[iqs / 4] = q;
        if (iqs % 16 == 0 && iqs < 96) {
          q8_blocks[ib].d2s6[2 + (iqs / 16)] = s;
          if (iqs % 64 == 0) {
            q8_blocks[ib].d2s6[iqs / 64] = d;
          }
        }
      }
    }
  }
  GGML_UNUSED(tile_x_max_i);

#else
  GGML_UNUSED_VARS(x, y, ids_dst, expert_bounds, dst, gate, limit, blocks_per_ne00, nrows_x,
                   stride_row_x, ncols_y, stride_col_dst, channel_ratio, stride_channel_x,
                   expert_tiles, q8, q8_ncols);
  NO_DEVICE_CODE;
#endif  // defined(TURING_MMA_AVAILABLE)
}

template <ggml_type kType, bool kQ8>
cudaError_t Launch(ggml_backend_cuda_context& ctx, const mmq_args& args, const float* gate,
                   float limit, void* q8, cudaStream_t stream) {
  constexpr int kJ = kJOf<kType>;
  const int device = ggml_cuda_get_device();
  const int cc = ggml_cuda_info().devices[device].cc;
  // DeepSeek V4's IQ2 gate/up experts over a prefill chunk of
  // kDsv4StagePairMinRows to kDsv4StageMaxRows tokens, six experts a token.
  if (cc != 1210 || args.type_x != kType || args.ncols_x != 4096 || args.nrows_x != 2048 ||
      args.ncols_max < kDsv4StagePairMinRows || args.ncols_max > kDsv4StageMaxRows ||
      args.ncols_dst != 6 * args.ncols_max || args.ncols_y != args.ncols_dst ||
      args.nrows_dst != 2048 || args.nchannels_x != 256 || args.nchannels_y != 256 ||
      args.nsamples_x != 1 || args.nsamples_y != 1 || args.ids_dst == nullptr ||
      args.expert_bounds == nullptr || !mmq_use_compact_experts(args, kJ)) {
    return cudaErrorInvalidValue;
  }
  const int warp_size = ggml_cuda_info().devices[device].warp_size;
  const ggml_cuda_mmq_config config = ggml_cuda_mmq_get_config(kType, kJ, kFallback, cc);
  const int nwarps = config.nthreads / warp_size;
  const size_t nbytes_shared = mmq_get_nbytes_shared(config, cc);
  const dim3 block_dims(static_cast<unsigned>(warp_size), static_cast<unsigned>(nwarps), 1);
  // The quantizing write-back stages kJ columns of I + 4 floats over the
  // tiles, after the column ids: they must fit the kernel's shared memory.
  const size_t staged =
      static_cast<size_t>(kJ) * (sizeof(int) + (static_cast<size_t>(config.I + 4) * sizeof(float)));
  if (kQ8 != (q8 != nullptr) || (kQ8 && staged > nbytes_shared)) {
    return cudaErrorInvalidValue;
  }
  CUDA_SET_SHARED_MEMORY_LIMIT((Iq2PairGluUpKernel<kType, kQ8>), static_cast<int>(nbytes_shared));
  const auto nty = static_cast<unsigned>((args.nrows_x + config.I - 1) / config.I);
  const uint3 blocks_per_ne00_fd =
      init_fastdiv_values(static_cast<uint64_t>(args.ncols_x / ggml_cuda_type_traits<kType>::qk));
  const uint3 channel_ratio_fd =
      init_fastdiv_values(static_cast<uint64_t>(args.nchannels_y / args.nchannels_x));
  const auto capacity = static_cast<int>(mmq_compact_expert_capacity(
      kType, args.nrows_x, args.ncols_max, args.ncols_dst, args.nchannels_x, config.J));
  ggml_cuda_pool_alloc<int2> tiles(ctx.pool(device), static_cast<size_t>(capacity));
  mmq_expert_tiles<kJ><<<1, 256, static_cast<size_t>(args.nchannels_x + 1) * sizeof(int), stream>>>(
      args.expert_bounds, tiles.ptr, static_cast<int>(args.nchannels_x), capacity);
  const dim3 grid(nty, static_cast<unsigned>(capacity), 1);
  Iq2PairGluUpKernel<kType, kQ8><<<grid, block_dims, nbytes_shared, stream>>>(
      args.x, args.y, args.ids_dst, args.expert_bounds, args.dst, gate, limit, blocks_per_ne00_fd,
      static_cast<int>(args.nrows_x), static_cast<int>(args.stride_row_x),
      static_cast<int>(args.ncols_y), static_cast<int>(args.nrows_dst), channel_ratio_fd,
      static_cast<int>(args.stride_channel_x), tiles.ptr, q8, static_cast<int>(args.ncols_y));
  return cudaGetLastError();
}

}  // namespace

cudaError_t LaunchIq2PairGluUp(ggml_backend_cuda_context& ctx, const mmq_args& args,
                               const float* gate, float limit, void* q8, cudaStream_t stream) {
  // The quantizing write-back is the IQ2_XXS pair's alone (validate_ext.h
  // MulMatIdQCompactPrequantFits).
  switch (args.type_x) {
    case GGML_TYPE_IQ2_XXS:
      return q8 != nullptr ? Launch<GGML_TYPE_IQ2_XXS, true>(ctx, args, gate, limit, q8, stream)
                           : Launch<GGML_TYPE_IQ2_XXS, false>(ctx, args, gate, limit, q8, stream);
    case GGML_TYPE_IQ2_XS:
      return q8 != nullptr ? cudaErrorInvalidValue
                           : Launch<GGML_TYPE_IQ2_XS, false>(ctx, args, gate, limit, q8, stream);
    default:
      return cudaErrorInvalidValue;
  }
}

}  // namespace jitllm::kernels::ggml
