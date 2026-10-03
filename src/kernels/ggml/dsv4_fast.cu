// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: MIT AND Apache-2.0

// DeepSeek V4's fast decode plan (jitllm_ops.h, "DeepSeek V4's fast plan"):
// jitLLM's kernels for the quantized vector products, the routing, the
// routed experts' combination and the hyper-connections' pre-mix. None
// reproduces GGML's arithmetic bit for bit (the owner's policy, 2026-09-28:
// speed first, correctness judged coarsely against llama.cpp); each is
// deterministic (no atomics, fixed summation orders), so a model's own runs,
// swaps and restores still repeat bit for bit.
//
// jitllm.vecq's body is GGML's MMVQ loop (mmvq.cu at llama.cpp b29c606e2,
// MIT): the same per-block vec_dot_*_q8_1 functions over Q8_1 activations,
// each thread striding the row's blocks, a shared-memory reduction over
// the warps and a warp reduction. What differs is what a block reads once:
// all of a chunk's tokens for a dense row, and for routed experts one
// block per distinct (expert, row block) of the chunk, which computes every
// token that selected the expert. A token's partial sums are the same
// whatever other tokens share the read.

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>
#include <string>
#include <utility>

#include "base/bytes.h"
#include "common.cuh"
#include "kernels/ggml/dsv4_fast.h"
#include "kernels/ggml/jitllm_ops.h"
#include "kernels/ggml/launch.h"
#include "quantize.cuh"
#include "vecdotq.cuh"

namespace jitllm::kernels::ggml {
namespace {

// ---------------------------------------------------------------- vecq

using VecDot = float (*)(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                         const int& kbx, const int& iqs);

constexpr __device__ VecDot DotOf(ggml_type type) {
  switch (type) {
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
    // Qwen3.8's GGUF quantizations' (the generic per-token path).
    case GGML_TYPE_Q3_K:
      return vec_dot_q3_K_q8_1;
    case GGML_TYPE_IQ2_S:
      return vec_dot_iq2_s_q8_1;
    case GGML_TYPE_IQ3_S:
      return vec_dot_iq3_s_q8_1;
    case GGML_TYPE_IQ4_NL:
      return vec_dot_iq4_nl_q8_1;
    case GGML_TYPE_IQ4_XS:
      return vec_dot_iq4_xs_q8_1;
    default:
      return nullptr;
  }
}

constexpr __host__ __device__ int VdrOf(ggml_type type) {
  switch (type) {
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
    case GGML_TYPE_Q3_K:
      return VDR_Q3_K_Q8_1_MMVQ;
    case GGML_TYPE_IQ2_S:
      return VDR_IQ2_S_Q8_1_MMVQ;
    case GGML_TYPE_IQ3_S:
      return VDR_IQ3_S_Q8_1_MMVQ;
    case GGML_TYPE_IQ4_NL:
      return VDR_IQ4_NL_Q8_1_MMVQ;
    case GGML_TYPE_IQ4_XS:
      return VDR_IQ4_XS_Q8_1_MMVQ;
    default:
      return 1;
  }
}

constexpr int kMaxPairs = 128;  // CheckVecQ: used · tokens <= 128

// The types GGML's MMVQ prefetches into L2 on the GB10 (mmvq.cu
// mmvq_should_prefetch, of the types here).
constexpr __host__ __device__ bool Prefetches(ggml_type type) {
  switch (type) {
    case GGML_TYPE_Q8_0:
    case GGML_TYPE_MXFP4:
    case GGML_TYPE_Q4_K:
    case GGML_TYPE_Q5_K:
    case GGML_TYPE_Q6_K:
      return true;
    default:
      return false;
  }
}

__device__ __forceinline__ float Silu(float x) { return x / (1.0f + expf(-x)); }

__device__ __forceinline__ float Glu(int glu, float gate, float up, float limit) {
  if (glu == static_cast<int>(VecQGlu::kSwigluClamp)) {
    gate = fminf(gate, limit);
    up = fmaxf(fminf(up, limit), -limit);
  }
  return Silu(gate) * up;
}

// The live tokens' (m of P) dot products over one weight block, row i of
// `acc`: GGML's vec_dot_*_q8_1 of (kbx, kqs) for each token, the weights
// loaded and decoded once for all of them where the type is DeepSeek's
// (IQ2_XS, IQ3_XXS, Q8_0, MXFP4); each token's arithmetic is the
// single-token function's.
template <ggml_type type, int P, int R>
__device__ __forceinline__ void DotMulti(const void* vx, const block_q8_1* const (&y)[P], int kby,
                                         int kbx, int kqs, int m, float (&acc)[P][R], int i) {
  if constexpr (type == GGML_TYPE_IQ2_XS) {
    const block_iq2_xs* bq2 = static_cast<const block_iq2_xs*>(vx) + kbx;
    const int2 q2_packed = make_int2(get_int_b2(bq2->qs, kqs + 0), get_int_b2(bq2->qs, kqs + 1));
    const auto* q2 = reinterpret_cast<const std::uint16_t*>(&q2_packed);
    const int ls0 = bq2->scales[kqs / 2] & 0x0F;
    const int ls1 = bq2->scales[kqs / 2] >> 4;
    int gl[4];
    int gh[4];
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
      const uint2 grid_pos = (reinterpret_cast<const uint2*>(iq2xs_grid))[q2[l0 / 2] & 0x1FF];
      const std::uint32_t signs = unpack_ksigns(q2[l0 / 2] >> 9);
      const int signs0 = __vcmpne4(signs & 0x08040201, 0);
      gl[l0 / 2] = __vsub4(grid_pos.x ^ signs0, signs0);
      const int signs1 = __vcmpne4(signs & 0x80402010, 0);
      gh[l0 / 2] = __vsub4(grid_pos.y ^ signs1, signs1);
    }
    const float dw = __half2float(bq2->d);
#pragma unroll
    for (int j = 0; j < P; ++j) {
      if (j < m) {
        const block_q8_1* b8 = &y[j][kby] + (kqs / 2);
        int sumi0 = 0;
        int sumi1 = 0;
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
          const int u0 = get_int_b4(b8->qs, l0 + 0);
          const int u1 = get_int_b4(b8->qs, l0 + 1);
          if (l0 < 4) {
            sumi0 = ggml_cuda_dp4a(gl[l0 / 2], u0, sumi0);
            sumi0 = ggml_cuda_dp4a(gh[l0 / 2], u1, sumi0);
          } else {
            sumi1 = ggml_cuda_dp4a(gl[l0 / 2], u0, sumi1);
            sumi1 = ggml_cuda_dp4a(gh[l0 / 2], u1, sumi1);
          }
        }
        const int sumi = ((sumi0 * ls0) + (sumi1 * ls1) + ((sumi0 + sumi1) / 2)) / 4;
        const float d = dw * __low2float(b8->ds);
        acc[j][i] += d * static_cast<float>(sumi);
      }
    }
  } else if constexpr (type == GGML_TYPE_IQ3_XXS) {
    const block_iq3_xxs* bq3 = static_cast<const block_iq3_xxs*>(vx) + kbx;
    const int2 q3_packed = make_int2(get_int_b2(bq3->qs, kqs), get_int_b2(bq3->qs, kqs + 1));
    const auto* q3 = reinterpret_cast<const std::uint8_t*>(&q3_packed);
    const std::uint32_t aux32 = get_int_b2(bq3->qs, (QK_K / 16) + (kqs / 2));
    int gl[4];
    int gh[4];
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
      const int2 grid_pos = make_int2(iq3xxs_grid[q3[l0 + 0]], iq3xxs_grid[q3[l0 + 1]]);
      const std::uint32_t signs = unpack_ksigns(aux32 >> (7 * l0 / 2));
      const int signs0 = __vcmpne4(signs & 0x08040201, 0);
      gl[l0 / 2] = __vsub4(grid_pos.x ^ signs0, signs0);
      const int signs1 = __vcmpne4(signs & 0x80402010, 0);
      gh[l0 / 2] = __vsub4(grid_pos.y ^ signs1, signs1);
    }
    const int ls = static_cast<int>(aux32 >> 28);
    const float dw = __half2float(bq3->d);
#pragma unroll
    for (int j = 0; j < P; ++j) {
      if (j < m) {
        const block_q8_1* b8 = &y[j][kby] + (kqs / 2);
        int sumi = 0;
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
          sumi = ggml_cuda_dp4a(gl[l0 / 2], get_int_b4(b8->qs, l0 + 0), sumi);
          sumi = ggml_cuda_dp4a(gh[l0 / 2], get_int_b4(b8->qs, l0 + 1), sumi);
        }
        sumi = ((ls * sumi) + (sumi / 2)) / 2;
        const float d = dw * __low2float(b8->ds);
        acc[j][i] += d * static_cast<float>(sumi);
      }
    }
  } else if constexpr (type == GGML_TYPE_Q8_0) {
    const block_q8_0* bq8_0 = static_cast<const block_q8_0*>(vx) + kbx;
    int v[VDR_Q8_0_Q8_1_MMVQ];
#pragma unroll
    for (int l = 0; l < VDR_Q8_0_Q8_1_MMVQ; ++l) {
      v[l] = get_int_b2(bq8_0->qs, kqs + l);
    }
    const float d8_0 = bq8_0->d;
#pragma unroll
    for (int j = 0; j < P; ++j) {
      if (j < m) {
        const block_q8_1* b8 = &y[j][kby];
        int sumi = 0;
#pragma unroll
        for (int l = 0; l < VDR_Q8_0_Q8_1_MMVQ; ++l) {
          sumi = ggml_cuda_dp4a(v[l], get_int_b4(b8->qs, kqs + l), sumi);
        }
        const float d8_1 = __low2half(b8->ds);
        acc[j][i] += d8_0 * d8_1 * static_cast<float>(sumi);
      }
    }
  } else if constexpr (type == GGML_TYPE_MXFP4) {
    const block_mxfp4* bq4 = static_cast<const block_mxfp4*>(vx) + kbx;
    int2 v[VDR_MXFP4_Q8_1_MMVQ];
#pragma unroll
    for (int l = 0; l < VDR_MXFP4_Q8_1_MMVQ; ++l) {
      v[l] = get_int_from_table_16(get_int_b1(bq4->qs, kqs + l), kvalues_mxfp4);
    }
    const float e = ggml_cuda_e8m0_to_fp32(bq4->e) * 0.5f;
#pragma unroll
    for (int j = 0; j < P; ++j) {
      if (j < m) {
        const block_q8_1* b8 = &y[j][kby];
        const int* q8 = reinterpret_cast<const int*>(b8->qs) + kqs;
        int sumi = 0;
#pragma unroll
        for (int l = 0; l < VDR_MXFP4_Q8_1_MMVQ; ++l) {
          sumi = ggml_cuda_dp4a(v[l].x, q8[l + 0], sumi);
          sumi = ggml_cuda_dp4a(v[l].y, q8[l + 4], sumi);
        }
        acc[j][i] += e * __low2float(b8->ds) * static_cast<float>(sumi);
      }
    }
  } else {
    constexpr VecDot dot = DotOf(type);
#pragma unroll
    for (int j = 0; j < P; ++j) {
      if (j < m) {
        acc[j][i] += dot(vx, &y[j][kby], kbx, kqs);
      }
    }
  }
}

// Rows R, warps W, tokens a pass P. kWarpRows: each warp computes its own R
// rows (a block covers W · R rows), its lanes striding the row's blocks;
// else the block's W warps stride R rows' blocks together and reduce
// through shared memory, as GGML's MMVQ does. Each (token, row) sum takes
// the same arithmetic whatever else the block computes.
template <ggml_type type, int R, int W, int P, bool kGlu, bool kWarpRows>
__launch_bounds__(W * 32, 1) __global__ void VecQKernel(const VecQDesc a) {
  constexpr int qk = ggml_cuda_type_traits<type>::qk;
  constexpr int qi = ggml_cuda_type_traits<type>::qi;
  constexpr int vdr = VdrOf(type);
  constexpr int threads = kWarpRows ? 32 : W * 32;  // striding a row's blocks
  constexpr int blocks_per_iter = vdr * threads / qi;
  constexpr int kShared = kWarpRows || W == 1 ? 1 : W - 1;

  __shared__ int pairs[kMaxPairs];  // token << 8 | slot
  __shared__ int count;
  __shared__ float partial[kShared][kGlu ? 2 : 1][kWarpRows ? 1 : P][kWarpRows ? 1 : R][32];

  const int tid = (32 * static_cast<int>(threadIdx.y)) + static_cast<int>(threadIdx.x);
  const int stride_id = kWarpRows ? static_cast<int>(threadIdx.x) : tid;
  const int row0 = kWarpRows
                       ? R * ((static_cast<int>(blockIdx.x) * W) + static_cast<int>(threadIdx.y))
                       : R * static_cast<int>(blockIdx.x);
  const auto* wbase = static_cast<const char*>(a.w);
  const auto* gbase = static_cast<const char*>(a.g);
  const auto* ybase = static_cast<const block_q8_1*>(a.y);
  const int nrows = a.nrows;
  float* const dst_base = a.dst;
  const int dst_token = a.dst_token;
  // Launched as a programmatic dependent (PDL, as GGML's kernels are): a
  // dense product's weights depend on no earlier kernel, so the thread's
  // first blocks are fetched into L2 while the kernel before drains; then
  // wait for it before reading anything it wrote.
  // Dense and grouped products name their matrix by grid row y (a
  // group's, at the group stride).
  const int dense_expert = static_cast<int>(blockIdx.y);
  if (a.ids == nullptr && (!kWarpRows || row0 < nrows)) {
    const int kbx0 = stride_id / (qi / vdr);
    if (kbx0 < a.ncols_x / qk) {
      const int base = (dense_expert * a.stride_expert) + (row0 * a.stride_row);
#pragma unroll
      for (int i = 0; i < R; ++i) {
        const std::size_t at = static_cast<std::size_t>(base + (i * a.stride_row) + kbx0) *
                               ggml_cuda_type_traits<type>::bs;
        asm volatile("prefetch.global.L2 [%0];" ::"l"(wbase + at));
        if constexpr (kGlu) {
          asm volatile("prefetch.global.L2 [%0];" ::"l"(gbase + at));
        }
      }
    }
  }
  ggml_cuda_pdl_sync();
  int expert = 0;
  if (a.ids == nullptr) {
    // Dense (one slot), or grouped: slot g's weights are matrix g.
    expert = dense_expert;
    if (tid < a.tokens) {
      pairs[tid] = (tid << 8) | expert;
    }
    if (tid == 0) {
      count = a.tokens;
    }
  } else {
    // One block per distinct expert of the chunk (and row block): the
    // first (token, slot) pair naming it computes every pair that does.
    const int s = static_cast<int>(blockIdx.y);
    expert = a.ids[((s / a.used) * a.ids_stride) + (s % a.used)];
    if (tid == 0) {
      int n = 0;
      bool owner = true;
      for (int s2 = 0; s2 < a.tokens * a.used; ++s2) {
        const int t = s2 / a.used;
        const int k = s2 % a.used;
        if (a.ids[(t * a.ids_stride) + k] == expert) {
          if (s2 < s) {
            owner = false;
            break;
          }
          pairs[n++] = (t << 8) | k;
        }
      }
      count = owner ? n : 0;
    }
  }
  __syncthreads();
  const int n = count;
  if (n == 0 || (kWarpRows && row0 >= nrows)) {
    return;  // no later barrier in the warp form
  }
  const int blocks_per_row = a.ncols_x / qk;
  const int kbx_offset = (expert * a.stride_expert) + (row0 * a.stride_row);

  for (int p = 0; p < n; p += P) {
    const int m = min(P, n - p);
    const block_q8_1* y[P];
#pragma unroll
    for (int j = 0; j < P; ++j) {
      const int pr = pairs[p + min(j, m - 1)];
      y[j] = ybase + ((pr >> 8) * a.y_token) + ((pr & 0xff) * a.y_slot);
    }
    float tmp[P][R] = {};
    float tmpg[kGlu ? P : 1][R] = {};
    for (int kbx = stride_id / (qi / vdr); kbx < blocks_per_row; kbx += blocks_per_iter) {
      const int kby = kbx * (qk / QK8_1);
      const int kqs = vdr * (stride_id % (qi / vdr));
      if constexpr (Prefetches(type)) {
        // GGML's MMVQ on the GB10: start the weight loads two iterations
        // ahead into L2.
        const int ahead = kbx + (2 * blocks_per_iter);
        if (ahead < blocks_per_row) {
#pragma unroll
          for (int i = 0; i < R; ++i) {
            const std::size_t at =
                static_cast<std::size_t>(kbx_offset + (i * a.stride_row) + ahead) *
                ggml_cuda_type_traits<type>::bs;
            asm volatile("prefetch.global.L2 [%0];" ::"l"(wbase + at));
            if constexpr (kGlu) {
              asm volatile("prefetch.global.L2 [%0];" ::"l"(gbase + at));
            }
          }
        }
      }
#pragma unroll
      for (int i = 0; i < R; ++i) {
        DotMulti<type, P, R>(wbase, y, kby, kbx_offset + (i * a.stride_row) + kbx, kqs, m, tmp, i);
        if constexpr (kGlu) {
          DotMulti<type, P, R>(gbase, y, kby, kbx_offset + (i * a.stride_row) + kbx, kqs, m, tmpg,
                               i);
        }
      }
    }
    // The weights are read: the next kernel may launch (it waits for this
    // one's writes at its own ggml_cuda_pdl_sync).
    ggml_cuda_pdl_lc();
    if constexpr (kWarpRows) {
#pragma unroll
      for (int j = 0; j < P; ++j) {
        if (j >= m) {
          continue;
        }
        const int pr = pairs[p + j];
        float* dst = dst_base + ((pr >> 8) * dst_token) + ((pr & 0xff) * a.dst_slot) + row0;
#pragma unroll
        for (int i = 0; i < R; ++i) {
          const float up = warp_reduce_sum<32>(tmp[j][i]);
          float value = up;
          if constexpr (kGlu) {
            value = Glu(a.glu, warp_reduce_sum<32>(tmpg[j][i]), up, a.limit);
          }
          if (static_cast<int>(threadIdx.x) == i && row0 + i < nrows) {
            dst[i] = value;
          }
        }
      }
    } else {
      if (threadIdx.y > 0) {
#pragma unroll
        for (int j = 0; j < P; ++j) {
#pragma unroll
          for (int i = 0; i < R; ++i) {
            partial[threadIdx.y - 1][0][j][i][threadIdx.x] = tmp[j][i];
            if constexpr (kGlu) {
              partial[threadIdx.y - 1][1][j][i][threadIdx.x] = tmpg[j][i];
            }
          }
        }
      }
      __syncthreads();
      if (threadIdx.y == 0) {
#pragma unroll
        for (int j = 0; j < P; ++j) {
          if (j >= m) {
            continue;
          }
          const int pr = pairs[p + j];
          float* dst = dst_base + ((pr >> 8) * dst_token) + ((pr & 0xff) * a.dst_slot) + row0;
#pragma unroll
          for (int i = 0; i < R; ++i) {
#pragma unroll
            for (int l = 0; l < W - 1; ++l) {
              tmp[j][i] += partial[l][0][j][i][threadIdx.x];
              if constexpr (kGlu) {
                tmpg[j][i] += partial[l][1][j][i][threadIdx.x];
              }
            }
            tmp[j][i] = warp_reduce_sum<32>(tmp[j][i]);
            if constexpr (kGlu) {
              tmpg[j][i] = warp_reduce_sum<32>(tmpg[j][i]);
            }
            if (static_cast<int>(threadIdx.x) == i && row0 + i < nrows) {
              if constexpr (kGlu) {
                dst[i] = Glu(a.glu, tmpg[j][i], tmp[j][i], a.limit);
              } else {
                dst[i] = tmp[j][i];
              }
            }
          }
        }
      }
      __syncthreads();
    }
  }
}

struct Variant {
  const char* name;
  int rows;
  int warps;
  int pass;
  bool warp_rows;
};

// The configurations built. The default picks among them by shape
// (DefaultVariant), from measurements on the GB10 (benchmarks/vecq_bench.cc).
constexpr Variant kVariants[] = {
    {"block r1 w4 p1", 1, 4, 1, false}, {"block r1 w8 p1", 1, 8, 1, false},
    {"block r2 w4 p1", 2, 4, 1, false}, {"warp r1 w4 p1", 1, 4, 1, true},
    {"warp r2 w4 p1", 2, 4, 1, true},   {"warp r4 w2 p1", 4, 2, 1, true},
    {"block r1 w4 p4", 1, 4, 4, false}, {"block r2 w4 p4", 2, 4, 4, false},
    {"warp r2 w4 p4", 2, 4, 4, true},   {"warp r1 w4 p4", 1, 4, 4, true},
    {"block r4 w4 p1", 4, 4, 1, false},
};
constexpr int kVariantCount = static_cast<int>(sizeof(kVariants) / sizeof(kVariants[0]));

template <ggml_type type, int R, int W, int P, bool kWarpRows>
void Launch(const VecQDesc& d, cudaStream_t stream) {
  // Routed: a block row per (token, slot) pair, most of which find their
  // expert owned by another; dense: one; grouped: one per group.
  const int slots = d.ids != nullptr ? d.used * d.tokens : d.used;
  const int rows_per_block = kWarpRows ? R * W : R;
  const dim3 grid(static_cast<unsigned>((d.nrows + rows_per_block - 1) / rows_per_block),
                  static_cast<unsigned>(slots));
  const dim3 block(32, W);
  const ggml_cuda_kernel_launch_params params(grid, block, 0, stream);
  if (d.g != nullptr) {
    ggml_cuda_kernel_launch(VecQKernel<type, R, W, P, true, kWarpRows>, params, d);
  } else {
    ggml_cuda_kernel_launch(VecQKernel<type, R, W, P, false, kWarpRows>, params, d);
  }
}

template <ggml_type type>
void LaunchVariant(const VecQDesc& d, int variant, cudaStream_t stream) {
  switch (variant) {
    case 0:
      Launch<type, 1, 4, 1, false>(d, stream);
      break;
    case 1:
      Launch<type, 1, 8, 1, false>(d, stream);
      break;
    case 2:
      Launch<type, 2, 4, 1, false>(d, stream);
      break;
    case 3:
      Launch<type, 1, 4, 1, true>(d, stream);
      break;
    case 4:
      Launch<type, 2, 4, 1, true>(d, stream);
      break;
    case 5:
      Launch<type, 4, 2, 1, true>(d, stream);
      break;
    case 6:
      Launch<type, 1, 4, 4, false>(d, stream);
      break;
    case 7:
      Launch<type, 2, 4, 4, false>(d, stream);
      break;
    case 8:
      Launch<type, 2, 4, 4, true>(d, stream);
      break;
    case 9:
      Launch<type, 1, 4, 4, true>(d, stream);
      break;
    default:
      Launch<type, 4, 4, 1, false>(d, stream);
      break;
  }
}

template <ggml_type type>
int LanesPerRow(int k) {
  constexpr int qk = ggml_cuda_type_traits<type>::qk;
  constexpr int qi = ggml_cuda_type_traits<type>::qi;
  return (k / qk) * (qi / VdrOf(type));
}

int LanesPerRowOf(ggml_type type, int k) {
  switch (type) {
    case GGML_TYPE_Q8_0:
      return LanesPerRow<GGML_TYPE_Q8_0>(k);
    case GGML_TYPE_Q2_K:
      return LanesPerRow<GGML_TYPE_Q2_K>(k);
    case GGML_TYPE_IQ2_XXS:
      return LanesPerRow<GGML_TYPE_IQ2_XXS>(k);
    case GGML_TYPE_MXFP4:
      return LanesPerRow<GGML_TYPE_MXFP4>(k);
    case GGML_TYPE_Q4_K:
      return LanesPerRow<GGML_TYPE_Q4_K>(k);
    case GGML_TYPE_Q5_K:
      return LanesPerRow<GGML_TYPE_Q5_K>(k);
    case GGML_TYPE_Q6_K:
      return LanesPerRow<GGML_TYPE_Q6_K>(k);
    case GGML_TYPE_IQ2_XS:
      return LanesPerRow<GGML_TYPE_IQ2_XS>(k);
    case GGML_TYPE_IQ3_XXS:
      return LanesPerRow<GGML_TYPE_IQ3_XXS>(k);
    case GGML_TYPE_Q3_K:
      return LanesPerRow<GGML_TYPE_Q3_K>(k);
    case GGML_TYPE_IQ2_S:
      return LanesPerRow<GGML_TYPE_IQ2_S>(k);
    case GGML_TYPE_IQ3_S:
      return LanesPerRow<GGML_TYPE_IQ3_S>(k);
    case GGML_TYPE_IQ4_NL:
      return LanesPerRow<GGML_TYPE_IQ4_NL>(k);
    case GGML_TYPE_IQ4_XS:
      return LanesPerRow<GGML_TYPE_IQ4_XS>(k);
    default:
      return 0;
  }
}

// The default for a shape, from benchmarks/vecq_bench.cc on the GB10
// (docs/experiments/dsv4-decode/). Routed products (short rows of many
// small matrices): one token, two rows a block striding with four warps;
// several, a warp's two rows (its blocks computing more tokens each).
// Dense products: see below.
int DefaultVariant(ggml_type type, const VecQDesc& d) {
  const bool even = d.nrows % 2 == 0;
  // Qwen3.8's GGUF shapes at decode (vecq_bench's "qwen" cases on the GB10,
  // docs/experiments/qwen38-gguf/): its routed experts' types, gate and up
  // one row a block (r1 w4), down two rows a warp; and dense rows shorter
  // than one 128-lane pass and not whole 256-value steps (its mixers' 320-
  // and shared expert's 640-value rows; no DeepSeek product's) four rows a
  // block.
  if (d.tokens == 1) {
    const bool gguf_type = type == GGML_TYPE_IQ2_S || type == GGML_TYPE_IQ3_S ||
                           type == GGML_TYPE_IQ4_NL || type == GGML_TYPE_IQ4_XS ||
                           type == GGML_TYPE_Q3_K;
    if (d.ids != nullptr && gguf_type) {
      return d.g != nullptr || !even ? 0 : 4;
    }
    if (d.ids == nullptr && d.ncols_x % 256 != 0 && LanesPerRowOf(type, d.ncols_x) < 128 &&
        d.nrows % 4 == 0) {
      return 10;
    }
  }
  if (d.ids != nullptr) {
    if (!even) {
      return 0;
    }
    return d.tokens == 1 || type == GGML_TYPE_MXFP4 ? 2 : 4;
  }
  const int passes = (LanesPerRowOf(type, d.ncols_x) + 127) / 128;
  const bool k_quant = type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K || type == GGML_TYPE_Q6_K;
  if (d.tokens == 1) {
    // Eight warps a row where four would take many passes (the K-quants'
    // rows from two).
    return passes >= 8 || (k_quant && passes >= 2) ? 1 : 0;
  }
  // Several tokens: eight warps for the long Q8_0 rows; the head (Q4_K) a
  // warp's row for all its tokens; the K-quant GLU four warps for all; the
  // short Q8_0 rows two rows a block, the tokens in one pass.
  if (passes >= 8) {
    return 1;
  }
  if (type == GGML_TYPE_Q4_K) {
    return 9;
  }
  if (k_quant && d.g != nullptr) {
    return 6;
  }
  if (type == GGML_TYPE_Q8_0 && passes == 1 && even) {
    return 7;
  }
  return even ? 2 : 0;
}

}  // namespace

int VecQVariants() { return kVariantCount; }

std::string VecQVariantName(int variant) {
  return variant >= 0 && variant < kVariantCount ? kVariants[variant].name : "default";
}

bool LaunchVecQ(ggml_type type, const VecQDesc& d, int variant, cudaStream_t stream) {
  if (variant < 0) {
    variant = DefaultVariant(type, d);
  }
  if (variant >= kVariantCount) {
    return false;
  }
  const Variant& v = kVariants[variant];
  // A block reads whole rows: R rows from row0 must exist.
  if (d.nrows % v.rows != 0) {
    return false;
  }
  switch (type) {
    case GGML_TYPE_Q8_0:
      LaunchVariant<GGML_TYPE_Q8_0>(d, variant, stream);
      break;
    case GGML_TYPE_Q2_K:
      LaunchVariant<GGML_TYPE_Q2_K>(d, variant, stream);
      break;
    case GGML_TYPE_IQ2_XXS:
      LaunchVariant<GGML_TYPE_IQ2_XXS>(d, variant, stream);
      break;
    case GGML_TYPE_MXFP4:
      LaunchVariant<GGML_TYPE_MXFP4>(d, variant, stream);
      break;
    case GGML_TYPE_Q4_K:
      LaunchVariant<GGML_TYPE_Q4_K>(d, variant, stream);
      break;
    case GGML_TYPE_Q5_K:
      LaunchVariant<GGML_TYPE_Q5_K>(d, variant, stream);
      break;
    case GGML_TYPE_Q6_K:
      LaunchVariant<GGML_TYPE_Q6_K>(d, variant, stream);
      break;
    case GGML_TYPE_IQ2_XS:
      LaunchVariant<GGML_TYPE_IQ2_XS>(d, variant, stream);
      break;
    case GGML_TYPE_IQ3_XXS:
      LaunchVariant<GGML_TYPE_IQ3_XXS>(d, variant, stream);
      break;
    case GGML_TYPE_Q3_K:
      LaunchVariant<GGML_TYPE_Q3_K>(d, variant, stream);
      break;
    case GGML_TYPE_IQ2_S:
      LaunchVariant<GGML_TYPE_IQ2_S>(d, variant, stream);
      break;
    case GGML_TYPE_IQ3_S:
      LaunchVariant<GGML_TYPE_IQ3_S>(d, variant, stream);
      break;
    case GGML_TYPE_IQ4_NL:
      LaunchVariant<GGML_TYPE_IQ4_NL>(d, variant, stream);
      break;
    case GGML_TYPE_IQ4_XS:
      LaunchVariant<GGML_TYPE_IQ4_XS>(d, variant, stream);
      break;
    default:
      return false;
  }
  return true;
}

namespace {

// ---------------------------------------------------------------- routing

constexpr int kExperts = 256;
constexpr int kPerLane = kExperts / 32;
constexpr int kRouteWarps = 4;

struct RouteArgs {
  const float* logits = nullptr;
  int logits_stride = 0;  // floats between tokens
  const float* bias = nullptr;
  const std::int32_t* table = nullptr;  // the hash layers' [used, vocab]
  const std::int32_t* tokens = nullptr;
  std::int32_t* out = nullptr;
  int n_tokens = 0;
  int used = 0;
  int norm = 0;
  float clamp = 0.0f;
  float scale = 1.0f;
};

// One warp a token: each lane holds 8 experts' scores.
__global__ void __launch_bounds__(32 * kRouteWarps) RouteKernel(const RouteArgs a) {
  ggml_cuda_pdl_sync();
  const int t = (static_cast<int>(blockIdx.x) * kRouteWarps) + static_cast<int>(threadIdx.y);
  if (t >= a.n_tokens) {
    return;
  }
  const int lane = static_cast<int>(threadIdx.x);
  float prob[kPerLane];
  float select[kPerLane];
#pragma unroll
  for (int i = 0; i < kPerLane; ++i) {
    const int e = lane + (32 * i);
    const float v = a.logits[(t * a.logits_stride) + e];
    prob[i] = sqrtf(v > 20.0f ? v : logf(1.0f + expf(v)));
    select[i] = a.bias != nullptr ? prob[i] + a.bias[e] : prob[i];
  }
  // Lane k keeps slot k's expert and weight.
  int my_id = 0;
  float my_weight = 0.0f;
  float sum = 0.0f;
  for (int k = 0; k < a.used; ++k) {
    int best = 0;
    float weight = 0.0f;
    if (a.table != nullptr) {
      best = a.table[(a.tokens[t] * a.used) + k];
      const int owner = best & 31;
      const int slot = (best >> 5) & (kPerLane - 1);
      float mine = 0.0f;
#pragma unroll
      for (int i = 0; i < kPerLane; ++i) {
        mine = i == slot ? prob[i] : mine;
      }
      weight = __shfl_sync(0xffffffffU, mine, owner);
    } else {
      float best_s = -INFINITY;
      float best_p = 0.0f;
      best = lane;
#pragma unroll
      for (int i = 0; i < kPerLane; ++i) {
        if (select[i] > best_s) {
          best_s = select[i];
          best_p = prob[i];
          best = lane + (32 * i);
        }
      }
#pragma unroll
      for (int mask = 16; mask > 0; mask >>= 1) {
        const float os = __shfl_xor_sync(0xffffffffU, best_s, mask);
        const float op = __shfl_xor_sync(0xffffffffU, best_p, mask);
        const int oe = __shfl_xor_sync(0xffffffffU, best, mask);
        if (os > best_s || (os == best_s && oe < best)) {
          best_s = os;
          best_p = op;
          best = oe;
        }
      }
#pragma unroll
      for (int i = 0; i < kPerLane; ++i) {
        if ((best & 31) == lane && (best >> 5) == i) {
          select[i] = -INFINITY;
        }
      }
      weight = best_p;
    }
    sum += weight;
    if (lane == k) {
      my_id = best;
      my_weight = weight;
    }
  }
  if (a.norm != 0) {
    my_weight = my_weight / fmaxf(sum, a.clamp);
  }
  my_weight *= a.scale;
  if (lane < a.used) {
    std::int32_t* row = a.out + (static_cast<std::ptrdiff_t>(t) * 2 * a.used);
    row[lane] = my_id;
    row[a.used + lane] = __float_as_int(my_weight);
  }
}

// ---------------------------------------------------------------- combine

__global__ void CombineKernel(const float4* down, const std::int32_t* route, const float4* shared,
                              float4* out, int n4, int used) {
  ggml_cuda_pdl_sync();
  const int c = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
  const int t = static_cast<int>(blockIdx.y);
  if (c >= n4) {
    return;
  }
  const std::int32_t* weights = route + (static_cast<std::ptrdiff_t>(t) * 2 * used) + used;
  float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  for (int k = 0; k < used; ++k) {
    const float w = __int_as_float(weights[k]);
    const float4 d = down[(static_cast<std::ptrdiff_t>((t * used) + k) * n4) + c];
    sum.x += d.x * w;
    sum.y += d.y * w;
    sum.z += d.z * w;
    sum.w += d.w * w;
  }
  const float4 s = shared[(static_cast<std::ptrdiff_t>(t) * n4) + c];
  out[(static_cast<std::ptrdiff_t>(t) * n4) + c] =
      make_float4(sum.x + s.x, sum.y + s.y, sum.z + s.z, sum.w + s.w);
}

// ---------------------------------------------------------------- hyper-connections

constexpr int kHc = 4;
constexpr int kMixes = (2 + kHc) * kHc;
constexpr int kMixThreads = static_cast<int>(kDsv4HcChunkThreads);
constexpr int kPreThreads = 1024;
constexpr int kPreMaxPerThread = 8;  // widths up to 8,192

// Each block: one chunk of the flattened values of up to kMixTokens tokens,
// their dot products with the kMixes weight rows and their sums of squares.
// A token's sums are the same whatever tokens share the block.
template <int kMixTokens>
__global__ void __launch_bounds__(kMixThreads)
    HcMixKernel(const float* x, const float* fn, float* partials, int flat, int tokens) {
  ggml_cuda_pdl_sync();
  const int c = static_cast<int>(blockIdx.x);
  const int t0 = static_cast<int>(blockIdx.y) * kMixTokens;
  const int m = min(kMixTokens, tokens - t0);
  const int chunk = flat / static_cast<int>(kDsv4HcChunks);
  float acc[kMixTokens][kMixes + 1] = {};
  for (int k = (c * chunk) + static_cast<int>(threadIdx.x); k < (c + 1) * chunk; k += kMixThreads) {
    float w[kMixes];
#pragma unroll
    for (int j = 0; j < kMixes; ++j) {
      w[j] = fn[(static_cast<std::ptrdiff_t>(j) * flat) + k];
    }
#pragma unroll
    for (int i = 0; i < kMixTokens; ++i) {
      if (i < m) {
        const float xv = x[(static_cast<std::ptrdiff_t>(t0 + i) * flat) + k];
#pragma unroll
        for (int j = 0; j < kMixes; ++j) {
          acc[i][j] += w[j] * xv;
        }
        acc[i][kMixes] += xv * xv;
      }
    }
  }
  __shared__ float warps[kMixThreads / 32][kMixTokens][kMixes + 1];
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
#pragma unroll
  for (int i = 0; i < kMixTokens; ++i) {
    if (i < m) {
#pragma unroll
      for (int j = 0; j <= kMixes; ++j) {
        const float v = warp_reduce_sum<32>(acc[i][j]);
        if (lane == 0) {
          warps[warp][i][j] = v;
        }
      }
    }
  }
  __syncthreads();
  for (int e = static_cast<int>(threadIdx.x); e < m * (kMixes + 1); e += kMixThreads) {
    const int i = e / (kMixes + 1);
    const int j = e % (kMixes + 1);
    float s = 0.0f;
#pragma unroll
    for (int w = 0; w < kMixThreads / 32; ++w) {
      s += warps[w][i][j];
    }
    partials[(((static_cast<std::ptrdiff_t>(t0 + i) * kDsv4HcChunks) + c) * (kMixes + 1)) + j] = s;
  }
}

__device__ __forceinline__ float Sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

// ---------------------------------------------------------------- compressors

struct CompressArgs {
  const float* state_kv = nullptr;
  const float* state_score = nullptr;
  const float* kv = nullptr;
  const float* score = nullptr;
  int state_stride = 0;  // floats between the state's rows (both)
  int state_score_stride = 0;
  int kv_stride = 0;  // between the chunk's rows
  int score_stride = 0;
  int state_rows = 0;
  int tokens = 0;
  const std::int32_t* read = nullptr;
  float* out = nullptr;
  int head = 0;
  int ratio = 0;
  int overlap = 0;
};

// One block a compressed block's 64 channels, a thread a channel: the
// online softmax of the block's rows' scores per channel and the weighted
// sum of their values.
__global__ void __launch_bounds__(256) CompressKernel(const CompressArgs a) {
  ggml_cuda_pdl_sync();
  const int b = static_cast<int>(blockIdx.x);
  const int per_block = a.overlap != 0 ? 2 * a.ratio : a.ratio;
  const int n_read = a.ratio * static_cast<int>(gridDim.x);
  const int zero = a.state_rows + a.tokens;  // the zero row (overlap only)
  const int c = static_cast<int>((blockIdx.y * blockDim.x) + threadIdx.x);
  if (c < a.head) {
    float mx = -INFINITY;
    float sum = 0.0f;
    float acc = 0.0f;
    for (int j = 0; j < per_block; ++j) {
      // Overlap: the first `ratio` rows' first half (the previous block),
      // then the next `ratio` rows' second half (this one).
      const bool second = a.overlap != 0 && j >= a.ratio;
      const int index = second ? n_read + (b * a.ratio) + (j - a.ratio) : (b * a.ratio) + j;
      const int row = a.read[index];
      const int ch = second ? a.head + c : c;
      float s = -INFINITY;
      float v = 0.0f;
      if (row >= 0 && row < a.state_rows) {
        s = a.state_score[(row * a.state_score_stride) + ch];
        v = a.state_kv[(row * a.state_stride) + ch];
      } else if (row >= a.state_rows && row < zero) {
        s = a.score[((row - a.state_rows) * a.score_stride) + ch];
        v = a.kv[((row - a.state_rows) * a.kv_stride) + ch];
      }
      if (s == -INFINITY) {
        continue;
      }
      if (s > mx) {
        const float scale = expf(mx - s);
        sum *= scale;
        acc *= scale;
        mx = s;
      }
      const float e = expf(s - mx);
      sum += e;
      acc += e * v;
    }
    a.out[(static_cast<std::ptrdiff_t>(b) * a.head) + c] = sum > 0.0f ? acc / sum : 0.0f;
  }
}

struct HcPreArgs {
  const float* partials = nullptr;
  const float* x = nullptr;
  const float* scale = nullptr;
  const float* base = nullptr;
  const float* norm = nullptr;
  float* out = nullptr;
  int width = 0;
  float rms_eps = 0.0f;
  float hc_eps = 0.0f;
  int iterations = 0;
};

// One block a token: the mixes, pre, post and comb (dsv4_hc_comb's
// Sinkhorn), then the streams' weighted sum, its RMSNorm and the norm
// weight.
__global__ void __launch_bounds__(kPreThreads) HcPreKernel(const HcPreArgs a) {
  ggml_cuda_pdl_sync();
  const int t = static_cast<int>(blockIdx.x);
  const int tid = static_cast<int>(threadIdx.x);
  const int flat = a.width * kHc;
  // The normed rows first, packed; then each token's post and comb.
  float* out = a.out + (static_cast<std::ptrdiff_t>(t) * a.width);
  float* tail = a.out + (static_cast<std::ptrdiff_t>(gridDim.x) * a.width) +
                (static_cast<std::ptrdiff_t>(t) * kDsv4HcTail);
  __shared__ float sums[kMixes + 1];
  __shared__ float pre[kHc];
  __shared__ float warps[kPreThreads / 32];
  const int warp = tid >> 5;
  const int lane = tid & 31;
  // Warp j sums mix j over the chunks (the last: the sum of squares).
  if (warp <= kMixes) {
    const float* p = a.partials + (static_cast<std::ptrdiff_t>(t) * kDsv4HcChunks * (kMixes + 1));
    float s = 0.0f;
    for (int c = lane; c < kDsv4HcChunks; c += 32) {
      s += p[(c * (kMixes + 1)) + warp];
    }
    s = warp_reduce_sum<32>(s);
    if (lane == 0) {
      sums[warp] = s;
    }
  }
  __syncthreads();
  if (warp == 0) {
    const float rms = rsqrtf((sums[kMixes] / static_cast<float>(flat)) + a.rms_eps);
    // Lanes 0-3: pre; 4-7: post; the comb (dsv4_hc_comb_f32's Sinkhorn)
    // on lanes 0-15, lane = idst + 4 · isrc, row and column sums by
    // shuffles (lanes 16-31 shadow them).
    if (lane < kHc) {
      pre[lane] = Sigmoid((sums[lane] * rms * a.scale[0]) + a.base[lane]) + a.hc_eps;
    } else if (lane < 2 * kHc) {
      tail[lane - kHc] = 2.0f * Sigmoid((sums[lane] * rms * a.scale[1]) + a.base[lane]);
    }
    const int idx = lane & 15;
    const float v = (sums[(2 * kHc) + idx] * rms * a.scale[2]) + a.base[(2 * kHc) + idx];
    float mx = fmaxf(v, __shfl_xor_sync(0xffffffffU, v, 1));
    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffU, mx, 2));
    const float e = expf(v - mx);
    float sum = e + __shfl_xor_sync(0xffffffffU, e, 1);
    sum += __shfl_xor_sync(0xffffffffU, sum, 2);
    float c = (e * (1.0f / sum)) + a.hc_eps;
    const auto cols = [&] {  // over isrc: lanes 4 and 8 apart
      float s = c + __shfl_xor_sync(0xffffffffU, c, 4);
      s += __shfl_xor_sync(0xffffffffU, s, 8);
      c *= 1.0f / (a.hc_eps + s);
    };
    const auto rows = [&] {  // over idst: lanes 1 and 2 apart
      float s = c + __shfl_xor_sync(0xffffffffU, c, 1);
      s += __shfl_xor_sync(0xffffffffU, s, 2);
      c *= 1.0f / (a.hc_eps + s);
    };
    cols();
    for (int i = 1; i < a.iterations; ++i) {
      rows();
      cols();
    }
    if (lane < kHc * kHc) {
      tail[kHc + lane] = c;
    }
  }
  __syncthreads();
  const float* xt = a.x + (static_cast<std::ptrdiff_t>(t) * flat);
  float y[kPreMaxPerThread];
  float ss = 0.0f;
#pragma unroll
  for (int r = 0; r < kPreMaxPerThread; ++r) {
    const int i = tid + (r * kPreThreads);
    y[r] = 0.0f;
    if (i < a.width) {
      float v = xt[i] * pre[0];
#pragma unroll
      for (int h = 1; h < kHc; ++h) {
        v += xt[(h * a.width) + i] * pre[h];
      }
      y[r] = v;
      ss += v * v;
    }
  }
  ss = warp_reduce_sum<32>(ss);
  if ((tid & 31) == 0) {
    warps[tid >> 5] = ss;
  }
  __syncthreads();
  if (tid < 32) {
    float v = warps[tid];
    v = warp_reduce_sum<32>(v);
    if (tid == 0) {
      warps[0] = v;
    }
  }
  __syncthreads();
  const float scale = rsqrtf((warps[0] / static_cast<float>(a.width)) + a.rms_eps);
#pragma unroll
  for (int r = 0; r < kPreMaxPerThread; ++r) {
    const int i = tid + (r * kPreThreads);
    if (i < a.width) {
      out[i] = y[r] * scale * a.norm[i];
    }
  }
}

}  // namespace

std::expected<void, KernelFailure> RunQuantizeQ8(LaunchContext& launch, ggml_tensor* node) {
  if (auto checked = CheckQuantizeQ8(node); !checked) {
    return checked;
  }
  return launch.Run(base::Bytes(0), [node](ggml_backend_cuda_context& context) {
    const ggml_tensor* x = node->src[0];
    const std::int64_t k = x->ne[0];
    const std::int64_t padded =
        (k + MATRIX_ROW_PADDING - 1) / MATRIX_ROW_PADDING * MATRIX_ROW_PADDING;
    const auto f = [](std::size_t bytes) {
      return static_cast<std::int64_t>(bytes / sizeof(float));
    };
    quantize_row_q8_1_cuda(static_cast<const float*>(x->data), nullptr, node->data, GGML_TYPE_Q8_0,
                           k, f(x->nb[1]), f(x->nb[2]), f(x->nb[2]) * x->ne[2], padded, x->ne[1],
                           x->ne[2], 1, context.stream());
  });
}

std::expected<void, KernelFailure> RunVecQ(LaunchContext& launch, ggml_tensor* node) {
  if (auto checked = CheckVecQ(node); !checked) {
    return checked;
  }
  return launch.Run(base::Bytes(0), [node](ggml_backend_cuda_context& context) {
    const ggml_tensor* w = node->src[0];
    const ggml_tensor* q8 = node->src[1];
    const bool routed = node->src[2] != nullptr && node->src[2]->type == GGML_TYPE_I32;
    const ggml_tensor* ids = routed ? node->src[2] : nullptr;
    const ggml_tensor* gate = routed ? node->src[3] : node->src[2];
    const int tokens = JitllmOpInt(node, 0);
    const bool per_slot = JitllmOpInt(node, 1) != 0;
    const auto type_size = static_cast<std::size_t>(ggml_type_size(w->type));
    const std::int64_t k = w->ne[0];
    const std::int64_t y_row = (k + MATRIX_ROW_PADDING - 1) / MATRIX_ROW_PADDING *
                               MATRIX_ROW_PADDING / QK8_1;  // Q8_1 blocks a row
    const bool grouped = !routed && w->ne[2] > 1;
    const int used = routed ? static_cast<int>(ids->ne[0]) : static_cast<int>(w->ne[2]);
    VecQDesc a;
    a.w = w->data;
    a.g = gate != nullptr ? gate->data : nullptr;
    a.y = q8->data;
    a.ids = routed ? static_cast<const std::int32_t*>(ids->data) : nullptr;
    a.dst = static_cast<float*>(node->data);
    a.ncols_x = static_cast<int>(k);
    a.nrows = static_cast<int>(w->ne[1]);
    a.stride_row = static_cast<int>(w->nb[1] / type_size);
    a.stride_expert = static_cast<int>(w->nb[2] / type_size);
    a.y_token = static_cast<int>(per_slot ? y_row * used : y_row);
    a.y_slot = static_cast<int>(per_slot ? y_row : 0);
    a.ids_stride = routed ? static_cast<int>(ids->nb[1] / sizeof(std::int32_t)) : 0;
    a.used = used;
    a.tokens = tokens;
    const bool slotted = routed || grouped;
    a.dst_token =
        static_cast<int>(slotted ? node->nb[2] / sizeof(float) : node->nb[1] / sizeof(float));
    a.dst_slot = static_cast<int>(slotted ? node->nb[1] / sizeof(float) : 0);
    a.glu = JitllmOpInt(node, 2);
    a.limit = JitllmOpFloat(node, 3);
    int variant = -1;
    if (VecQOneToken(node)) {
      // The launch a one-token product of this shape takes (SetVecQOneToken).
      VecQDesc one = a;
      one.tokens = 1;
      variant = DefaultVariant(w->type, one);
    }
    if (!LaunchVecQ(w->type, a, variant, context.stream())) {
      // CheckVecQ admits only what the default launches.
      ggml_cuda_error("LaunchVecQ", __func__, __FILE__, __LINE__,
                      "jitllm.vecq refused a node its check admitted");
    }
  });
}

std::expected<void, KernelFailure> RunDsv4Route(LaunchContext& launch, ggml_tensor* node) {
  if (auto checked = CheckDsv4Route(node); !checked) {
    return checked;
  }
  return launch.Run(base::Bytes(0), [node](ggml_backend_cuda_context& context) {
    const ggml_tensor* logits = node->src[0];
    const bool hashed = node->src[2] != nullptr;
    RouteArgs a;
    a.logits = static_cast<const float*>(logits->data);
    a.logits_stride = static_cast<int>(logits->nb[1] / sizeof(float));
    a.bias = hashed ? nullptr : static_cast<const float*>(node->src[1]->data);
    a.table = hashed ? static_cast<const std::int32_t*>(node->src[1]->data) : nullptr;
    a.tokens = hashed ? static_cast<const std::int32_t*>(node->src[2]->data) : nullptr;
    a.out = static_cast<std::int32_t*>(node->data);
    a.n_tokens = static_cast<int>(logits->ne[1]);
    a.used = JitllmOpInt(node, 0);
    a.norm = JitllmOpInt(node, 1);
    a.clamp = JitllmOpFloat(node, 2);
    a.scale = JitllmOpFloat(node, 3);
    const dim3 grid(static_cast<unsigned>((a.n_tokens + kRouteWarps - 1) / kRouteWarps));
    ggml_cuda_kernel_launch(
        RouteKernel,
        ggml_cuda_kernel_launch_params(grid, dim3(32, kRouteWarps), 0, context.stream()), a);
  });
}

std::expected<void, KernelFailure> RunDsv4Combine(LaunchContext& launch, ggml_tensor* node) {
  if (auto checked = CheckDsv4Combine(node); !checked) {
    return checked;
  }
  return launch.Run(base::Bytes(0), [node](ggml_backend_cuda_context& context) {
    const ggml_tensor* down = node->src[0];
    const int n4 = static_cast<int>(down->ne[0] / 4);
    const dim3 grid(static_cast<unsigned>((n4 + 255) / 256), static_cast<unsigned>(down->ne[2]));
    ggml_cuda_kernel_launch(CombineKernel,
                            ggml_cuda_kernel_launch_params(grid, dim3(256), 0, context.stream()),
                            static_cast<const float4*>(down->data),
                            static_cast<const std::int32_t*>(node->src[1]->data),
                            static_cast<const float4*>(node->src[2]->data),
                            static_cast<float4*>(node->data), n4, static_cast<int>(down->ne[1]));
  });
}

std::expected<void, KernelFailure> RunDsv4HcMix(LaunchContext& launch, ggml_tensor* node) {
  if (auto checked = CheckDsv4HcMix(node); !checked) {
    return checked;
  }
  return launch.Run(base::Bytes(0), [node](ggml_backend_cuda_context& context) {
    const ggml_tensor* x = node->src[0];
    const int flat = static_cast<int>(x->ne[0] * x->ne[1]);
    const int tokens = static_cast<int>(x->ne[2]);
    const auto* xs = static_cast<const float*>(x->data);
    const auto* fn = static_cast<const float*>(node->src[1]->data);
    auto* out = static_cast<float*>(node->data);
    // CheckDsv4HcMix: flat is a multiple of kDsv4HcChunks · kMixThreads.
    if (tokens == 1) {
      ggml_cuda_kernel_launch(HcMixKernel<1>,
                              ggml_cuda_kernel_launch_params(
                                  dim3(kDsv4HcChunks, 1), dim3(kMixThreads), 0, context.stream()),
                              xs, fn, out, flat, tokens);
    } else {
      constexpr int kTokens = 4;  // a block's tokens, reading the weights once
      const dim3 grid(static_cast<unsigned>(kDsv4HcChunks),
                      static_cast<unsigned>((tokens + kTokens - 1) / kTokens));
      ggml_cuda_kernel_launch(
          HcMixKernel<kTokens>,
          ggml_cuda_kernel_launch_params(grid, dim3(kMixThreads), 0, context.stream()), xs, fn, out,
          flat, tokens);
    }
  });
}

std::expected<void, KernelFailure> RunDsv4Compress(LaunchContext& launch, ggml_tensor* node) {
  if (auto checked = CheckDsv4Compress(node); !checked) {
    return checked;
  }
  return launch.Run(base::Bytes(0), [node](ggml_backend_cuda_context& context) {
    const auto f = [](std::size_t bytes) { return static_cast<int>(bytes / sizeof(float)); };
    const ggml_tensor* skv = node->src[0];
    const ggml_tensor* ssc = node->src[1];
    const ggml_tensor* kv = node->src[2];
    const ggml_tensor* score = node->src[3];
    CompressArgs a;
    a.state_kv = static_cast<const float*>(skv->data);
    a.state_score = static_cast<const float*>(ssc->data);
    a.kv = static_cast<const float*>(kv->data);
    a.score = static_cast<const float*>(score->data);
    a.state_stride = f(skv->nb[1]);
    a.state_score_stride = f(ssc->nb[1]);
    a.kv_stride = f(kv->nb[1]);
    a.score_stride = f(score->nb[1]);
    a.state_rows = static_cast<int>(skv->ne[1]);
    a.tokens = static_cast<int>(kv->ne[1]);
    a.read = static_cast<const std::int32_t*>(node->src[4]->data);
    a.out = static_cast<float*>(node->data);
    a.head = static_cast<int>(node->ne[0]);
    a.ratio = JitllmOpInt(node, 0);
    a.overlap = JitllmOpInt(node, 1);
    // A block a compressed block and 64 channels: decode's one or two
    // blocks still spread over the SMs.
    constexpr int kChannels = 64;
    const dim3 grid(static_cast<unsigned>(node->ne[2]),
                    static_cast<unsigned>((a.head + kChannels - 1) / kChannels));
    ggml_cuda_kernel_launch(
        CompressKernel, ggml_cuda_kernel_launch_params(grid, dim3(kChannels), 0, context.stream()),
        a);
  });
}

std::expected<void, KernelFailure> RunDsv4HcPre(LaunchContext& launch, ggml_tensor* node) {
  if (auto checked = CheckDsv4HcPre(node); !checked) {
    return checked;
  }
  return launch.Run(base::Bytes(0), [node](ggml_backend_cuda_context& context) {
    HcPreArgs a;
    a.partials = static_cast<const float*>(node->src[0]->data);
    a.x = static_cast<const float*>(node->src[1]->data);
    a.scale = static_cast<const float*>(node->src[2]->data);
    a.base = static_cast<const float*>(node->src[3]->data);
    a.norm = static_cast<const float*>(node->src[4]->data);
    a.out = static_cast<float*>(node->data);
    a.width = static_cast<int>(node->src[1]->ne[0]);
    a.rms_eps = JitllmOpFloat(node, 0);
    a.hc_eps = JitllmOpFloat(node, 1);
    a.iterations = JitllmOpInt(node, 2);
    ggml_cuda_kernel_launch(
        HcPreKernel,
        ggml_cuda_kernel_launch_params(dim3(static_cast<unsigned>(node->src[1]->ne[2])),
                                       dim3(kPreThreads), 0, context.stream()),
        a);
  });
}

}  // namespace jitllm::kernels::ggml
