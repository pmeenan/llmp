// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: MIT AND Apache-2.0

// llmp.vecq (llmp_ops.h): the quantized vector products over Q8_1
// activations that every GGML-encoded family's decode uses (dense rows,
// routed experts, grouped matrices, with or without a GLU), and llmp.q8_1,
// the activations' quantization they read. None reproduces GGML's
// arithmetic bit for bit (the owner's policy, 2026-09-28: speed first,
// correctness judged coarsely against llama.cpp); each is deterministic (no
// atomics, fixed summation orders), so a model's own runs, swaps and
// restores still repeat bit for bit.
//
// llmp.vecq's body is GGML's MMVQ loop (mmvq.cu at llama.cpp b29c606e2,
// MIT): the same per-block vec_dot_*_q8_1 functions over Q8_1 activations,
// each thread striding the row's blocks, a shared-memory reduction over
// the warps and a warp reduction. What differs is what a block reads once:
// all of a chunk's tokens for a dense row, and for routed experts one
// block per distinct (expert, row block) of the chunk, which computes every
// token that selected the expert. A token's partial sums are the same
// whatever other tokens share the read.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>
#include <initializer_list>
#include <optional>
#include <string>
#include <utility>

#include "base/bytes.h"
#include "common.cuh"
#include "kernels/ggml/launch.h"
#include "kernels/ggml/llmp_ops.h"
#include "kernels/ggml/vecq.h"
#include "quantize.cuh"
#include "unary.cuh"
#include "vecdotq.cuh"

namespace llmp::kernels::ggml {
namespace {

// ---------------------------------------------------------------- vecq

using VecDot = float (*)(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                         const int& kbx, const int& iqs);

constexpr __device__ VecDot DotOf(ggml_type type) {
  switch (type) {
    case GGML_TYPE_Q4_0:
      return vec_dot_q4_0_q8_1;
    case GGML_TYPE_Q4_1:
      return vec_dot_q4_1_q8_1;
    case GGML_TYPE_Q5_0:
      return vec_dot_q5_0_q8_1;
    case GGML_TYPE_Q5_1:
      return vec_dot_q5_1_q8_1;
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
    // The community GGUFs' low-bit types (the generic per-token path).
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
    case GGML_TYPE_Q4_0:
      return VDR_Q4_0_Q8_1_MMVQ;
    case GGML_TYPE_Q4_1:
      return VDR_Q4_1_Q8_1_MMVQ;
    case GGML_TYPE_Q5_0:
      return VDR_Q5_0_Q8_1_MMVQ;
    case GGML_TYPE_Q5_1:
      return VDR_Q5_1_Q8_1_MMVQ;
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

// The types GGML's MMVQ prefetches into L2 on the GB10 (the pinned mmvq.cu's
// mmvq_should_prefetch, of the types here: all but IQ1_M, which llmp.vecq
// does not compute). Upstream prefetches only when built for the DGX Spark
// (its measured gain on every type listed; on higher-bandwidth parts the
// extra requests cost more than they save): PrefetchesOnDevice.
[[maybe_unused]] constexpr __host__ __device__ bool Prefetches(ggml_type type) {
  switch (type) {
    case GGML_TYPE_Q4_0:
    case GGML_TYPE_Q5_0:
    case GGML_TYPE_Q8_0:
    case GGML_TYPE_MXFP4:
    case GGML_TYPE_Q3_K:
    case GGML_TYPE_Q4_K:
    case GGML_TYPE_Q5_K:
    case GGML_TYPE_Q6_K:
    case GGML_TYPE_IQ4_NL:
    case GGML_TYPE_IQ4_XS:
      return true;
    default:
      return false;
  }
}
constexpr __device__ bool PrefetchesOnDevice(ggml_type type) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == GGML_CUDA_CC_DGX_SPARK
  return Prefetches(type);
#else
  (void)type;
  return false;
#endif
}
// The weights' first blocks fetched into L2 before the dependency wait:
// only where programmatic dependent launch exists (upstream's Hopper-and-
// newer condition), since without it the kernel before has already drained.
constexpr __device__ bool PrefetchesBeforeWait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_HOPPER
  return true;
#else
  return false;
#endif
}

__device__ __forceinline__ float Silu(float x) { return x / (1.0f + expf(-x)); }

__device__ __forceinline__ float Glu(int glu, float gate, float up, float limit) {
  if (glu == static_cast<int>(VecQGlu::kGeGlu)) {
    return ggml_cuda_op_gelu_single(gate) * up;
  }
  if (glu == static_cast<int>(VecQGlu::kSwigluClamp)) {
    gate = fminf(gate, limit);
    up = fmaxf(fminf(up, limit), -limit);
  }
  return Silu(gate) * up;
}

// The live tokens' (m of P) dot products over one weight block, row i of
// `acc`: GGML's vec_dot_*_q8_1 of (kbx, kqs) for each token, the weights
// loaded and decoded once for IQ2_XS, IQ3_XXS, Q8_0, MXFP4 and the legacy
// Q4_0/Q4_1/Q5_0/Q5_1/IQ4_NL formats; each token's arithmetic is the
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
  } else if constexpr (type == GGML_TYPE_Q4_0) {
    // GGML's original helper arithmetic; only weight loads move outside
    // the token loop. Each column keeps its own activation sums/scales.
    const auto* b = static_cast<const block_q4_0*>(vx) + kbx;
    int v[VDR_Q4_0_Q8_1_MMVQ];
#pragma unroll
    for (int l = 0; l < VDR_Q4_0_Q8_1_MMVQ; ++l) {
      v[l] = get_int_b2(b->qs, kqs + l);
    }
    const auto scale = b->d;
#pragma unroll
    for (int j = 0; j < P; ++j) {
      if (j < m) {
        const auto* b8 = &y[j][kby];
        int u[2 * VDR_Q4_0_Q8_1_MMVQ];
#pragma unroll
        for (int l = 0; l < VDR_Q4_0_Q8_1_MMVQ; ++l) {
          u[2 * l] = get_int_b4(b8->qs, kqs + l);
          u[2 * l + 1] = get_int_b4(b8->qs, kqs + l + QI4_0);
        }
        acc[j][i] += vec_dot_q4_0_q8_1_impl<VDR_Q4_0_Q8_1_MMVQ>(v, u, scale, b8->ds);
      }
    }
  } else if constexpr (type == GGML_TYPE_Q4_1) {
    // GGML's original helper arithmetic; only weight loads move outside
    // the token loop. Each column keeps its own activation sums/scales.
    const auto* b = static_cast<const block_q4_1*>(vx) + kbx;
    int v[VDR_Q4_1_Q8_1_MMVQ];
#pragma unroll
    for (int l = 0; l < VDR_Q4_1_Q8_1_MMVQ; ++l) {
      v[l] = get_int_b4(b->qs, kqs + l);
    }
    const auto scale = b->dm;
#pragma unroll
    for (int j = 0; j < P; ++j) {
      if (j < m) {
        const auto* b8 = &y[j][kby];
        int u[2 * VDR_Q4_1_Q8_1_MMVQ];
#pragma unroll
        for (int l = 0; l < VDR_Q4_1_Q8_1_MMVQ; ++l) {
          u[2 * l] = get_int_b4(b8->qs, kqs + l);
          u[2 * l + 1] = get_int_b4(b8->qs, kqs + l + QI4_1);
        }
        acc[j][i] += vec_dot_q4_1_q8_1_impl<VDR_Q4_1_Q8_1_MMVQ>(v, u, scale, b8->ds);
      }
    }
  } else if constexpr (type == GGML_TYPE_Q5_0) {
    // GGML's original helper arithmetic; only weight loads move outside
    // the token loop. Each column keeps its own activation sums/scales.
    const auto* b = static_cast<const block_q5_0*>(vx) + kbx;
    int vl[VDR_Q5_0_Q8_1_MMVQ];
    int vh[VDR_Q5_0_Q8_1_MMVQ];
#pragma unroll
    for (int l = 0; l < VDR_Q5_0_Q8_1_MMVQ; ++l) {
      vl[l] = get_int_b2(b->qs, kqs + l);
      vh[l] = get_int_b2(b->qh, 0) >> (4 * (kqs + l));
    }
    const auto scale = b->d;
#pragma unroll
    for (int j = 0; j < P; ++j) {
      if (j < m) {
        const auto* b8 = &y[j][kby];
        int u[2 * VDR_Q5_0_Q8_1_MMVQ];
#pragma unroll
        for (int l = 0; l < VDR_Q5_0_Q8_1_MMVQ; ++l) {
          u[2 * l] = get_int_b4(b8->qs, kqs + l);
          u[2 * l + 1] = get_int_b4(b8->qs, kqs + l + QI5_0);
        }
        acc[j][i] += vec_dot_q5_0_q8_1_impl<VDR_Q5_0_Q8_1_MMVQ>(vl, vh, u, scale, b8->ds);
      }
    }
  } else if constexpr (type == GGML_TYPE_Q5_1) {
    // GGML's original helper arithmetic; only weight loads move outside
    // the token loop. Each column keeps its own activation sums/scales.
    const auto* b = static_cast<const block_q5_1*>(vx) + kbx;
    int vl[VDR_Q5_1_Q8_1_MMVQ];
    int vh[VDR_Q5_1_Q8_1_MMVQ];
#pragma unroll
    for (int l = 0; l < VDR_Q5_1_Q8_1_MMVQ; ++l) {
      vl[l] = get_int_b4(b->qs, kqs + l);
      vh[l] = get_int_b4(b->qh, 0) >> (4 * (kqs + l));
    }
    const auto scale = b->dm;
#pragma unroll
    for (int j = 0; j < P; ++j) {
      if (j < m) {
        const auto* b8 = &y[j][kby];
        int u[2 * VDR_Q5_1_Q8_1_MMVQ];
#pragma unroll
        for (int l = 0; l < VDR_Q5_1_Q8_1_MMVQ; ++l) {
          u[2 * l] = get_int_b4(b8->qs, kqs + l);
          u[2 * l + 1] = get_int_b4(b8->qs, kqs + l + QI5_1);
        }
        acc[j][i] += vec_dot_q5_1_q8_1_impl<VDR_Q5_1_Q8_1_MMVQ>(vl, vh, u, scale, b8->ds);
      }
    }
  } else if constexpr (type == GGML_TYPE_IQ4_NL) {
    const auto* b = static_cast<const block_iq4_nl*>(vx) + kbx;
    int2 v[VDR_IQ4_NL_Q8_1_MMVQ];
#pragma unroll
    for (int l = 0; l < VDR_IQ4_NL_Q8_1_MMVQ; ++l) {
      v[l] = get_int_from_table_16(get_int_b2(b->qs, kqs + l), kvalues_iq4nl);
    }
    const float dw = __half2float(b->d);
#pragma unroll
    for (int j = 0; j < P; ++j) {
      if (j < m) {
        const auto* b8 = &y[j][kby];
        const int* q8 = reinterpret_cast<const int*>(b8->qs) + kqs;
        int sumi = 0;
#pragma unroll
        for (int l = 0; l < VDR_IQ4_NL_Q8_1_MMVQ; ++l) {
          sumi = ggml_cuda_dp4a(v[l].x, q8[l], sumi);
          sumi = ggml_cuda_dp4a(v[l].y, q8[l + 4], sumi);
        }
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
  if (PrefetchesBeforeWait() && a.ids == nullptr && (!kWarpRows || row0 < nrows)) {
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
    // The first warp scans the pairs 32 at a time (a wave's 16 rows name
    // up to 128; one thread's serial scan of them cost more than the
    // short rows' products), keeping them in pair order.
    const int s = static_cast<int>(blockIdx.y);
    expert = a.ids[((s / a.used) * a.ids_stride) + (s % a.used)];
    if (threadIdx.y == 0) {
      const int lane = static_cast<int>(threadIdx.x);
      const int total = a.tokens * a.used;
      int n = 0;
      bool owner = true;
      for (int base = 0; base < total && owner; base += 32) {
        const int s2 = base + lane;
        const int t = s2 / a.used;
        const int k = s2 % a.used;
        const bool match = s2 < total && a.ids[(t * a.ids_stride) + k] == expert;
        const unsigned ballot = __ballot_sync(0xffffffffU, match);
        if (n == 0 && ballot != 0) {
          // The first pair naming the expert owns it.
          owner = base + __ffs(static_cast<int>(ballot)) - 1 == s;
        }
        if (owner && match) {
          pairs[n + __popc(ballot & ((1U << lane) - 1U))] = (t << 8) | k;
        }
        n += __popc(ballot);
      }
      if (lane == 0) {
        count = owner ? n : 0;
      }
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
      if constexpr (PrefetchesOnDevice(type)) {
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
    {"block r4 w4 p1", 4, 4, 1, false}, {"block r1 w8 p4", 1, 8, 4, false},
    {"warp r4 w2 p4", 4, 2, 4, true},   {"block r4 w4 p4", 4, 4, 4, false},
    {"block r2 w2 p4", 2, 2, 4, false}, {"block r1 w2 p4", 1, 2, 4, false},
};

// The four-token sibling of each one-token configuration: the same rows,
// warps and form, so every token's sums take the one-token launch's
// arithmetic bit for bit (each (token, row) sum is independent of the
// others a pass computes), while a pass reads each weight block once for
// four tokens. A wave of one-row steps (SetVecQOneToken) takes it for its
// dense products without a GLU (RunVecQ).
int FourTokenSibling(int variant) {
  switch (variant) {
    case 0:
      return 6;
    case 1:
      return 11;
    case 2:
      return 7;
    case 3:
      return 9;
    case 4:
      return 8;
    case 5:
      return 12;
    case 10:
      return 13;
    default:
      return variant;  // already four tokens a pass
  }
}
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
    case 10:
      Launch<type, 4, 4, 1, false>(d, stream);
      break;
    case 11:
      Launch<type, 1, 8, 4, false>(d, stream);
      break;
    case 12:
      Launch<type, 4, 2, 4, true>(d, stream);
      break;
    case 13:
      Launch<type, 4, 4, 4, false>(d, stream);
      break;
    case 14:
      Launch<type, 2, 2, 4, false>(d, stream);
      break;
    case 15:
      Launch<type, 1, 2, 4, false>(d, stream);
      break;
    default:
      return;
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
    case GGML_TYPE_Q4_0:
      return LanesPerRow<GGML_TYPE_Q4_0>(k);
    case GGML_TYPE_Q4_1:
      return LanesPerRow<GGML_TYPE_Q4_1>(k);
    case GGML_TYPE_Q5_0:
      return LanesPerRow<GGML_TYPE_Q5_0>(k);
    case GGML_TYPE_Q5_1:
      return LanesPerRow<GGML_TYPE_Q5_1>(k);
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

// Incumbent GB10 scalar MMVQ schedule (pinned mmvq.cu d812350):
// calc_nwarps, should_use_small_k and should_halve_iters. LanesPerRowOf
// uses that source's qk/qi/VDR traits; these are encoding rules, not model
// or measured-shape eligibility. The caller restricts this to GB10/no IDs.
int ScalarMmvqVariant(ggml_type type, const VecQDesc& d) {
  const int lanes = LanesPerRowOf(type, d.ncols_x);
  // Upstream excludes these two IQ types from small-K on Turing and newer.
  if (lanes < 128 && type != GGML_TYPE_IQ3_XXS && type != GGML_TYPE_IQ3_S) {
    // R batches independent rows, without changing a row's partial sums.
    // For an output tail use R=1, avoiding upstream's padded weight reads.
    return d.nrows % 4 == 0 ? 10 : 0;
  }
  // The GB10 table only doubles these encodings' baseline four warps.
  switch (type) {
    case GGML_TYPE_Q4_0:
    case GGML_TYPE_Q4_1:
    case GGML_TYPE_Q5_0:
    case GGML_TYPE_Q5_1:
    case GGML_TYPE_Q8_0:
    case GGML_TYPE_Q4_K:
    case GGML_TYPE_Q5_K:
    case GGML_TYPE_Q6_K:
    case GGML_TYPE_IQ4_NL: {
      const int trips = (lanes + 127) / 128;
      const int wide_trips = (trips + 1) / 2;
      const int idle = 2 * wide_trips - trips;
      return idle * 8 <= 2 * wide_trips ? 1 : 0;
    }
    default:
      return 0;
  }
}

// The default schedule for a shape: the first row of kSchedules that the
// launch matches. The rows are measured data (benchmarks/vecq_bench.cc on
// the GB10; docs/experiments/dsv4-decode/ and qwen38-gguf/), not model
// names: a device class, the product's form, its token count, its weight
// encodings, whether it has a GLU, its rows' length in 128-lane passes (and
// whether they are whole 256-value steps) and the output rows it needs.
enum class SchedDevice : std::uint8_t { kAny, kGb10 };
enum class SchedForm : std::uint8_t {
  kDense,    // no ids (one matrix, or grouped matrices)
  kOneSlot,  // dense with one matrix
  kRouted,   // ids
};
enum class SchedGlu : std::uint8_t { kAny, kWith, kWithout };
// Weight encodings, as a bit set by ggml_type.
constexpr std::uint64_t Types(std::initializer_list<ggml_type> types) {
  std::uint64_t bits = 0;
  for (const ggml_type t : types) bits |= std::uint64_t{1} << static_cast<unsigned>(t);
  return bits;
}
constexpr std::uint64_t kAnyType = ~std::uint64_t{0};
constexpr std::uint64_t kKQuants = Types({GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K});
// The low-bit types of the community GGUFs' routed experts.
constexpr std::uint64_t kGgufExperts =
    Types({GGML_TYPE_IQ2_S, GGML_TYPE_IQ3_S, GGML_TYPE_IQ4_NL, GGML_TYPE_IQ4_XS, GGML_TYPE_Q3_K});
constexpr int kScalarMmvq = -2;  // ScalarMmvqVariant: upstream's one-column schedule

struct Schedule {
  SchedDevice device = SchedDevice::kAny;
  SchedForm form = SchedForm::kDense;
  int tokens_min = 1, tokens_max = INT32_MAX;
  std::uint64_t types = kAnyType;
  SchedGlu glu = SchedGlu::kAny;
  int passes_min = 0, passes_max = INT32_MAX;  // the row's 128-lane passes
  bool unaligned = false;                      // rows not whole 256-value steps only
  int rows_multiple = 1;                       // output rows a multiple of this
  int variant = 0;
};

// In order; the last row of each form takes the rest.
constexpr Schedule kSchedules[] = {
    // The GB10's one-column dense products: upstream MMVQ's schedule.
    {.device = SchedDevice::kGb10, .tokens_max = 1, .variant = kScalarMmvq},
    // Its multi-column schedule: four warps reduce columns 2..4, two warps
    // columns 5..8 (four a pass: one pass of eight measured slower, the
    // second pass's weights coming from L2; docs/experiments/vecq-columns),
    // two output rows a block (odd row counts the one-row tail-safe form).
    // Grouped and routed work keep their own.
    {.device = SchedDevice::kGb10,
     .form = SchedForm::kOneSlot,
     .tokens_min = 2,
     .tokens_max = 4,
     .rows_multiple = 2,
     .variant = 7},
    {.device = SchedDevice::kGb10,
     .form = SchedForm::kOneSlot,
     .tokens_min = 2,
     .tokens_max = 4,
     .variant = 6},
    {.device = SchedDevice::kGb10,
     .form = SchedForm::kOneSlot,
     .tokens_min = 5,
     .tokens_max = 8,
     .rows_multiple = 2,
     .variant = 14},
    {.device = SchedDevice::kGb10,
     .form = SchedForm::kOneSlot,
     .tokens_min = 5,
     .tokens_max = 8,
     .variant = 15},
    // Routed GGUF experts at one token: gate and up one row a block (r1
    // w4), down two rows a warp.
    {.form = SchedForm::kRouted,
     .tokens_max = 1,
     .types = kGgufExperts,
     .glu = SchedGlu::kWithout,
     .rows_multiple = 2,
     .variant = 4},
    {.form = SchedForm::kRouted, .tokens_max = 1, .types = kGgufExperts, .variant = 0},
    // Dense rows shorter than one pass and not whole 256-value steps (320-
    // and 640-value rows): four rows a block.
    {.tokens_max = 1, .passes_max = 1, .unaligned = true, .rows_multiple = 4, .variant = 10},
    // Routed products: two rows a block striding with four warps at one
    // token (and for MXFP4), a warp's two rows for several.
    {.form = SchedForm::kRouted, .tokens_max = 1, .rows_multiple = 2, .variant = 2},
    {.form = SchedForm::kRouted,
     .types = Types({GGML_TYPE_MXFP4}),
     .rows_multiple = 2,
     .variant = 2},
    {.form = SchedForm::kRouted, .rows_multiple = 2, .variant = 4},
    {.form = SchedForm::kRouted, .variant = 0},
    // Dense, one token: eight warps a row where four would take many passes
    // (the K-quants' rows from two).
    {.tokens_max = 1, .passes_min = 8, .variant = 1},
    {.tokens_max = 1, .types = kKQuants, .passes_min = 2, .variant = 1},
    {.tokens_max = 1, .variant = 0},
    // Dense, several tokens: eight warps for the long rows; the Q4_K head a
    // warp's row for all its tokens; the K-quant GLU four warps for all; the
    // short Q8_0 rows two rows a block, the tokens in one pass.
    {.passes_min = 8, .variant = 1},
    {.types = Types({GGML_TYPE_Q4_K}), .variant = 9},
    {.types = kKQuants, .glu = SchedGlu::kWith, .variant = 6},
    {.types = Types({GGML_TYPE_Q8_0}),
     .passes_min = 1,
     .passes_max = 1,
     .rows_multiple = 2,
     .variant = 7},
    {.rows_multiple = 2, .variant = 2},
    {.variant = 0},
};

bool OnGb10() {
  const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
  return GGML_CUDA_CC_IS_NVIDIA(cc) &&
         ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_DGX_SPARK;
}

int DefaultVariant(ggml_type type, const VecQDesc& d) {
  const bool routed = d.ids != nullptr;
  const int lanes = LanesPerRowOf(type, d.ncols_x);
  const int passes = (lanes + 127) / 128;
  const std::uint64_t bit = std::uint64_t{1} << static_cast<unsigned>(type);
  // The device is asked only when a row needs it (dense, up to 8 tokens).
  std::optional<bool> gb10;
  for (const Schedule& r : kSchedules) {
    const bool form = r.form == SchedForm::kRouted    ? routed
                      : r.form == SchedForm::kOneSlot ? !routed && d.used == 1
                                                      : !routed;
    if (!form || d.tokens < r.tokens_min || d.tokens > r.tokens_max || (r.types & bit) == 0 ||
        (r.glu == SchedGlu::kWith && d.g == nullptr) ||
        (r.glu == SchedGlu::kWithout && d.g != nullptr) || passes < r.passes_min ||
        passes > r.passes_max || (r.unaligned && d.ncols_x % 256 == 0) ||
        d.nrows % r.rows_multiple != 0) {
      continue;
    }
    if (r.device == SchedDevice::kGb10) {
      if (!gb10) gb10 = OnGb10();
      if (!*gb10) continue;
    }
    return r.variant == kScalarMmvq ? ScalarMmvqVariant(type, d) : r.variant;
  }
  return 0;
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
    case GGML_TYPE_Q4_0:
      LaunchVariant<GGML_TYPE_Q4_0>(d, variant, stream);
      break;
    case GGML_TYPE_Q4_1:
      LaunchVariant<GGML_TYPE_Q4_1>(d, variant, stream);
      break;
    case GGML_TYPE_Q5_0:
      LaunchVariant<GGML_TYPE_Q5_0>(d, variant, stream);
      break;
    case GGML_TYPE_Q5_1:
      LaunchVariant<GGML_TYPE_Q5_1>(d, variant, stream);
      break;
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
    const int tokens = LlmpOpInt(node, 0);
    const bool per_slot = LlmpOpInt(node, 1) != 0;
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
    a.glu = LlmpOpInt(node, 2);
    a.limit = LlmpOpFloat(node, 3);
    int variant = -1;
    if (VecQOneToken(node)) {
      // The configuration a one-token product of this shape takes
      // (SetVecQOneToken): each token's sums are its one-token product's
      // bit for bit. A dense product without a GLU takes it four tokens a
      // pass, each weight block read once for four tokens; routed and GLU
      // products keep one token a pass, which measured faster on the GB10
      // (vecq_bench, four tokens: routed IQ2_XS gate+up 517 against 677
      // us, Q5_K shared gate+up 62 against 84; dense Q8_0 1024x32768 179
      // against 152, the Q4_K head 2,059 against 1,845).
      VecQDesc one = a;
      one.tokens = 1;
      variant = DefaultVariant(w->type, one);
      if (!routed && gate == nullptr) {
        variant = FourTokenSibling(variant);
      }
    }
    if (!LaunchVecQ(w->type, a, variant, context.stream())) {
      // CheckVecQ admits only what the default launches.
      ggml_cuda_error("LaunchVecQ", __func__, __FILE__, __LINE__,
                      "llmp.vecq refused a node its check admitted");
    }
  });
}

}  // namespace llmp::kernels::ggml
