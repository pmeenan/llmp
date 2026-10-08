// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Llmpalooza's fusions of Qwen3.8's hyper-connection and MoE-output elementwise
// work, the BF16 conversion and the BF16 product (llmp_ops.h). Each
// kernel repeats the arithmetic of the GGML nodes it replaces, operation by
// operation in their order, so that its result is theirs bit for bit: this
// file builds with GGML's device flags (-use_fast_math, as its own units,
// third_party/patches/ggml/0002), writes GGML's expressions for the
// functions (op_sigmoid, ggml_cuda_op_silu_single, scale_f32's
// scale · x + bias), reduces each norm as rms_norm_f32<1024> does, and
// writes a product that GGML computes in a node of its own and then adds
// with __fmul_rn, which is never contracted into a multiply-add.

#include <cooperative_groups.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_pipeline.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <string>
#include <type_traits>
#include <utility>

#include "base/bytes.h"
#include "common.cuh"
#include "kernels/ggml/cublas.h"
#include "kernels/ggml/launch.h"
#include "kernels/ggml/llmp_ops.h"
#include "kernels/ggml/moe_cutlass.h"
#include "kernels/ggml/mxfp8_cutlass.h"
#include "kernels/ggml/mxfp8_quant.cuh"
#include "kernels/ggml/qwen38_commit.h"
#include "kernels/ggml/validate.h"

namespace llmp::kernels::ggml {
namespace {

std::unexpected<KernelFailure> Refused(std::string detail) {
  return std::unexpected(
      KernelFailure{.error = KernelError::kRejected, .detail = std::move(detail)});
}

constexpr int kThreads = 256;
constexpr int kGdnColumns = 4;      // value columns a warp of the recurrence keeps
constexpr int kNormThreads = 1024;  // rms_norm_f32's block for rows of 1,024 or more
constexpr int kMaxStreams = 8;      // the checks' bound on hc

// GGML's op_sigmoid and ggml_cuda_op_silu_single.
__device__ __forceinline__ float Sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
__device__ __forceinline__ float Silu(float x) { return x / (1.0f + expf(-x)); }
// scale_f32: scale · x + bias, the bias a kernel argument as there (ggml_scale's
// 0), so that it is not folded away.
__device__ __forceinline__ float Scale(float x, float scale, float bias) {
  return scale * x + bias;
}

// GGML's warp_reduce_sum over 32 lanes.
__device__ __forceinline__ float WarpSum(float x) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    x += __shfl_xor_sync(0xffffffffu, x, offset, 32);
  }
  return x;
}

// rms_norm_f32<1024>'s scale of a row of n >= 1,024 floats: each thread sums
// the squares of its columns tid, tid + 1,024, ... in order, then GGML's
// block_reduce (a warp sum, each warp's sum through shared memory, a warp
// sum of those), then rsqrtf(mean + eps). GGML's `tmp / ncols + eps`
// compiles (fast math) to one multiply-add of the sum, the approximate
// reciprocal of ncols and eps; it is written so here, since the compiler
// would not contract it where only one thread uses the result. Every thread
// returns it; `shared` is free again when it returns.
__device__ float RowScale(const float* __restrict__ x, int n, float eps, float* shared) {
  const int tid = static_cast<int>(threadIdx.x);
  float tmp = 0.0f;
  for (int col = tid; col < n; col += kNormThreads) {
    const float xi = x[col];
    tmp += xi * xi;
  }
  tmp = WarpSum(tmp);
  if (tid % 32 == 0) {
    shared[tid / 32] = tmp;
  }
  __syncthreads();
  tmp = WarpSum(shared[tid % 32]);
  __syncthreads();
  const float inverse = 1.0f / static_cast<float>(n);
  return rsqrtf(fmaf(tmp, inverse, eps));
}

// res + out · (2 · sigmoid(inject / hc)), four columns a thread.
__global__ void HcCombineKernel(const float4* __restrict__ res, const float4* __restrict__ out,
                                const float* __restrict__ inject, float4* __restrict__ dst,
                                int width4, int hc, std::int64_t n4, float inv_hc, float bias) {
  const std::int64_t i = (static_cast<std::int64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  if (i >= n4) {
    return;
  }
  const std::int64_t row = i / width4;  // t · hc + k
  const std::int64_t c4 = i % width4;
  const std::int64_t t = row / hc;
  const float w = Scale(Sigmoid(Scale(inject[row], inv_hc, bias)), 2.0f, bias);
  const float4 b = out[(t * width4) + c4];
  const float4 r = res[i];
  dst[i] = make_float4(r.x + __fmul_rn(b.x, w), r.y + __fmul_rn(b.y, w), r.z + __fmul_rn(b.z, w),
                       r.w + __fmul_rn(b.w, w));
}

template <typename T>
__device__ __forceinline__ T Store(float v);
template <>
__device__ __forceinline__ float Store<float>(float v) {
  return v;
}
template <>
__device__ __forceinline__ nv_bfloat16 Store<nv_bfloat16>(float v) {
  return __float2bfloat16(v);
}

// One stream of one token a block: rms_norm, then times the weight.
template <typename T>
__global__ void __launch_bounds__(kNormThreads)
    HcNormKernel(const float* __restrict__ x, const float* __restrict__ weight, T* __restrict__ dst,
                 int width, int hc, float eps) {
  __shared__ float shared[32];
  const std::int64_t row = (static_cast<std::int64_t>(blockIdx.x) * hc) + blockIdx.y;
  const float* xr = x + (row * width);
  const float* wk = weight + (static_cast<std::int64_t>(blockIdx.y) * width);
  T* out = dst + (row * width);
  const float scale = RowScale(xr, width, eps, shared);
  for (int c = static_cast<int>(threadIdx.x); c < width; c += kNormThreads) {
    out[c] = Store<T>(scale * xr[c] * wk[c]);
  }
}

// One token a block: each stream's norm scale, then the gated streams
// folded in order and scaled by 1 / hc.
__global__ void __launch_bounds__(kNormThreads)
    HcMixKernel(const float* __restrict__ x, const float* __restrict__ weight,
                const float* __restrict__ gate, float* __restrict__ dst, int width, int hc,
                float eps, float inv_hc, float bias) {
  __shared__ float shared[32];
  __shared__ float scales[kMaxStreams];
  const std::int64_t t = blockIdx.x;
  const std::int64_t stride = static_cast<std::int64_t>(hc) * width;
  const float* xt = x + (t * stride);
  const float* gt = gate + (t * stride);
  for (int k = 0; k < hc; ++k) {
    const float s = RowScale(xt + (static_cast<std::int64_t>(k) * width), width, eps, shared);
    if (threadIdx.x == 0) {
      scales[k] = s;
    }
  }
  __syncthreads();
  for (int c = static_cast<int>(threadIdx.x); c < width; c += kNormThreads) {
    float acc = 0.0f;
    for (int k = 0; k < hc; ++k) {
      const std::int64_t at = (static_cast<std::int64_t>(k) * width) + c;
      const float xn = scales[k] * xt[at] * weight[at];
      const float gated = __fmul_rn(xn, Sigmoid(gt[at]));
      acc = k == 0 ? gated : acc + gated;
    }
    dst[(t * width) + c] = Scale(acc, inv_hc, bias);
  }
}

__device__ __forceinline__ float ExpertScale(const float* __restrict__ scales, std::int32_t id,
                                             int experts) {
  return id >= 0 && id < experts ? scales[id] : __uint_as_float(0x7fc00000u);
}

// silu(gate · s_gate[e]) · (up · s_up[e]), four columns a thread.
__global__ void MoeGluKernel(const float4* __restrict__ gate, const float4* __restrict__ up,
                             const std::int32_t* __restrict__ ids,
                             const float* __restrict__ gate_scale,
                             const float* __restrict__ up_scale, float4* __restrict__ dst, int n4,
                             int used, int ids_stride, int experts, std::int64_t total4) {
  const std::int64_t i = (static_cast<std::int64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  if (i >= total4) {
    return;
  }
  const std::int64_t row = i / n4;  // t · used + j
  const std::int64_t t = row / used;
  const std::int64_t j = row % used;
  const std::int32_t id = ids[(t * ids_stride) + j];
  const float gs = ExpertScale(gate_scale, id, experts);
  const float us = ExpertScale(up_scale, id, experts);
  const float4 g = gate[i];
  const float4 u = up[i];
  dst[i] = make_float4(Silu(g.x * gs) * (u.x * us), Silu(g.y * gs) * (u.y * us),
                       Silu(g.z * gs) * (u.z * us), Silu(g.w * gs) * (u.w * us));
}

// Σ_j (down_j · s_down[e_j]) · w_j in expert order, plus shared ·
// sigmoid(shared gate), four columns a thread. Without scales (null: a GGUF
// checkpoint's experts) Σ_j down_j · w_j, GGML's unfused nodes' arithmetic
// (a product by the weight, the slots added in order).
__global__ void MoeCombineKernel(const float4* __restrict__ down,
                                 const std::int32_t* __restrict__ ids,
                                 const float* __restrict__ down_scale,
                                 const float* __restrict__ weights,
                                 const float4* __restrict__ shared,
                                 const float* __restrict__ shared_gate, float4* __restrict__ dst,
                                 int width4, int used, int ids_stride, int experts,
                                 std::int64_t total4) {
  const std::int64_t i = (static_cast<std::int64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  if (i >= total4) {
    return;
  }
  const std::int64_t t = i / width4;
  const std::int64_t c4 = i % width4;
  float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  for (int j = 0; j < used; ++j) {
    const float w = weights[(t * used) + j];
    const float4 d = down[(((t * used) + j) * width4) + c4];
    float4 e;
    if (down_scale == nullptr) {
      e = make_float4(__fmul_rn(d.x, w), __fmul_rn(d.y, w), __fmul_rn(d.z, w), __fmul_rn(d.w, w));
    } else {
      const float ds = ExpertScale(down_scale, ids[(t * ids_stride) + j], experts);
      e = make_float4(__fmul_rn(d.x * ds, w), __fmul_rn(d.y * ds, w), __fmul_rn(d.z * ds, w),
                      __fmul_rn(d.w * ds, w));
    }
    if (j == 0) {
      acc = e;
    } else {
      acc = make_float4(acc.x + e.x, acc.y + e.y, acc.z + e.z, acc.w + e.w);
    }
  }
  const float g = Sigmoid(shared_gate[t]);
  const float4 s = shared[i];
  dst[i] = make_float4(acc.x + __fmul_rn(s.x, g), acc.y + __fmul_rn(s.y, g),
                       acc.z + __fmul_rn(s.z, g), acc.w + __fmul_rn(s.w, g));
}

// gated_delta_net_cuda<128, false, false>'s recurrence, each column's
// arithmetic as its SASS computes it (sm_121a, fast math): kv = k · S's
// column by a multiply-add chain over each lane's rows, a warp sum,
// delta = beta · (v - g · kv), S = g · S + k · delta, attn = q · S by the same
// chain and sum, times the scale. Upstream gives each warp one column, so
// 6,144 warps stream the whole sequence in several waves; here a warp keeps
// kColumns columns in registers and reads each token's k and q once for
// them, so one wave covers the heads. The state pointers are not
// __restrict__: llmp.gdn.step passes the same state for both (each warp
// reads its columns into registers before any is written, and writes only
// its own).
template <int kColumns>
__global__ void __launch_bounds__(128)
    GdnColumnsKernel(const float* __restrict__ q, const float* __restrict__ k,
                     const float* __restrict__ v, const float* __restrict__ g,
                     const float* __restrict__ beta, const float* state_in, float* __restrict__ dst,
                     float* state_out, int heads, int qk_heads, int tokens, std::int64_t sq1,
                     std::int64_t sq2, std::int64_t sv1, std::int64_t sv2, std::int64_t sb1,
                     std::int64_t sb2, float scale) {
  // A programmatic dependent after it (the next product) may launch now and
  // fetch its weights; it waits for this kernel before reading its output.
  ggml_cuda_pdl_lc();
  constexpr int kS = 128;
  constexpr int kRows = kS / 32;
  const int lane = static_cast<int>(threadIdx.x);
  const int h = static_cast<int>(blockIdx.x);
  const int col0 =
      ((static_cast<int>(blockIdx.y) * blockDim.y) + static_cast<int>(threadIdx.y)) * kColumns;
  const int hq = h % qk_heads;
  float s[kColumns][kRows];
  const float* s0 = state_in + (static_cast<std::int64_t>(h) * kS * kS);
#pragma unroll
  for (int c = 0; c < kColumns; ++c) {
#pragma unroll
    for (int r = 0; r < kRows; ++r) {
      s[c][r] = s0[(static_cast<std::int64_t>(col0 + c) * kS) + (r * 32) + lane];
    }
  }
  float* attn = dst + (static_cast<std::int64_t>(h) * kS);
  // Each token's inputs are loaded a token ahead, so that the loads' latency
  // overlaps the previous token's arithmetic.
  float kn[kRows];
  float qn[kRows];
  float vn[kColumns];
  float beta_n = 0.0f;
  float g_n = 0.0f;
  const auto load = [&](int t) {
    const float* qt = q + (t * sq2) + (hq * sq1);
    const float* kt = k + (t * sq2) + (hq * sq1);
    const float* vt = v + (t * sv2) + (h * sv1);
    const std::int64_t gb = (t * sb2) + (h * sb1);
    beta_n = beta[gb];
    g_n = g[gb];
#pragma unroll
    for (int r = 0; r < kRows; ++r) {
      kn[r] = kt[(r * 32) + lane];
      qn[r] = qt[(r * 32) + lane];
    }
#pragma unroll
    for (int c = 0; c < kColumns; ++c) {
      vn[c] = vt[col0 + c];
    }
  };
  if (tokens > 0) {
    load(0);
  }
  for (int t = 0; t < tokens; ++t) {
    float kr[kRows];
    float qr[kRows];
    float vr[kColumns];
#pragma unroll
    for (int r = 0; r < kRows; ++r) {
      kr[r] = kn[r];
      qr[r] = qn[r];
    }
#pragma unroll
    for (int c = 0; c < kColumns; ++c) {
      vr[c] = vn[c];
    }
    const float beta_t = beta_n;
    const float g_t = expf(g_n);
    if (t + 1 < tokens) {
      load(t + 1);
    }
    float kv[kColumns];
#pragma unroll
    for (int c = 0; c < kColumns; ++c) {
      kv[c] = fmaf(kr[0], s[c][0], 0.0f);
#pragma unroll
      for (int r = 1; r < kRows; ++r) {
        kv[c] = fmaf(kr[r], s[c][r], kv[c]);
      }
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
#pragma unroll
      for (int c = 0; c < kColumns; ++c) {
        kv[c] = __fadd_rn(kv[c], __shfl_xor_sync(0xffffffffu, kv[c], offset, 32));
      }
    }
    float at[kColumns];
#pragma unroll
    for (int c = 0; c < kColumns; ++c) {
      const float delta = __fmul_rn(beta_t, fmaf(-g_t, kv[c], vr[c]));
#pragma unroll
      for (int r = 0; r < kRows; ++r) {
        s[c][r] = fmaf(g_t, s[c][r], __fmul_rn(kr[r], delta));
      }
      at[c] = fmaf(qr[0], s[c][0], 0.0f);
#pragma unroll
      for (int r = 1; r < kRows; ++r) {
        at[c] = fmaf(qr[r], s[c][r], at[c]);
      }
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
#pragma unroll
      for (int c = 0; c < kColumns; ++c) {
        at[c] = __fadd_rn(at[c], __shfl_xor_sync(0xffffffffu, at[c], offset, 32));
      }
    }
    if (lane == 0) {
#pragma unroll
      for (int c = 0; c < kColumns; ++c) {
        attn[col0 + c] = __fmul_rn(at[c], scale);
      }
    }
    attn += static_cast<std::int64_t>(kS) * heads;
  }
  if (state_out == nullptr) {
    return;  // a verify's step: its state is the commit's to write
  }
  float* s1 = state_out + (static_cast<std::int64_t>(h) * kS * kS);
#pragma unroll
  for (int c = 0; c < kColumns; ++c) {
#pragma unroll
    for (int r = 0; r < kRows; ++r) {
      s1[(static_cast<std::int64_t>(col0 + c) * kS) + (r * 32) + lane] = s[c][r];
    }
  }
}

// A verify's commit (qwen38_commit.h): GdnColumnsKernel's state update, the
// same operations in the same order, over the kept rows of every layer
// (blockIdx.z), in place; no attention output.
__global__ void __launch_bounds__(128) GdnCommitKernel(Qwen38CommitArgs a) {
  constexpr int kS = 128;
  constexpr int kRows = kS / 32;
  const Qwen38CommitLayer& layer = a.layer[blockIdx.z];
  const int lane = static_cast<int>(threadIdx.x);
  const int h = static_cast<int>(blockIdx.x);
  const int col0 =
      ((static_cast<int>(blockIdx.y) * blockDim.y) + static_cast<int>(threadIdx.y)) * kGdnColumns;
  const int hq = h % a.qk_heads;
  float s[kGdnColumns][kRows];
  float* s0 = layer.state + (static_cast<std::int64_t>(h) * kS * kS);
#pragma unroll
  for (int c = 0; c < kGdnColumns; ++c) {
#pragma unroll
    for (int r = 0; r < kRows; ++r) {
      s[c][r] = s0[(static_cast<std::int64_t>(col0 + c) * kS) + (r * 32) + lane];
    }
  }
  const int k_offset = a.qk_heads * kS;      // q heads, then k heads,
  const int v_offset = 2 * a.qk_heads * kS;  // then v heads
  for (int t = 0; t < a.keep; ++t) {
    const float* row = layer.conv + (static_cast<std::int64_t>(t) * a.channels);
    const float* kt = row + k_offset + (hq * kS);
    const float* vt = row + v_offset + (h * kS);
    float kr[kRows];
    float vr[kGdnColumns];
#pragma unroll
    for (int r = 0; r < kRows; ++r) {
      kr[r] = kt[(r * 32) + lane];
    }
#pragma unroll
    for (int c = 0; c < kGdnColumns; ++c) {
      vr[c] = vt[col0 + c];
    }
    const std::int64_t gb = (static_cast<std::int64_t>(t) * a.v_heads) + h;
    const float beta_t = layer.beta[gb];
    const float g_t = expf(layer.gate[gb]);
    float kv[kGdnColumns];
#pragma unroll
    for (int c = 0; c < kGdnColumns; ++c) {
      kv[c] = fmaf(kr[0], s[c][0], 0.0f);
#pragma unroll
      for (int r = 1; r < kRows; ++r) {
        kv[c] = fmaf(kr[r], s[c][r], kv[c]);
      }
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
#pragma unroll
      for (int c = 0; c < kGdnColumns; ++c) {
        kv[c] = __fadd_rn(kv[c], __shfl_xor_sync(0xffffffffu, kv[c], offset, 32));
      }
    }
#pragma unroll
    for (int c = 0; c < kGdnColumns; ++c) {
      const float delta = __fmul_rn(beta_t, fmaf(-g_t, kv[c], vr[c]));
#pragma unroll
      for (int r = 0; r < kRows; ++r) {
        s[c][r] = fmaf(g_t, s[c][r], __fmul_rn(kr[r], delta));
      }
    }
  }
#pragma unroll
  for (int c = 0; c < kGdnColumns; ++c) {
#pragma unroll
    for (int r = 0; r < kRows; ++r) {
      s0[(static_cast<std::int64_t>(col0 + c) * kS) + (r * 32) + lane] = s[c][r];
    }
  }
}

// A convolution history after `keep` rows: tap j of channel c becomes the
// (keep + j)-th of the old taps followed by the rows' inputs. A thread a
// channel, blockIdx.y a layer (the last, past `layers`, the n-gram layer's),
// in place: every old tap is read before any is written.
__global__ void HistoryCommitKernel(Qwen38CommitArgs a) {
  const int l = static_cast<int>(blockIdx.y);
  const bool ple = l == a.layers;
  float* history = ple ? a.ple_history : a.layer[l].history;
  const float* rows = ple ? a.ple_rows : a.layer[l].qkv;
  const int channels = ple ? a.ple_width : a.channels;
  const int taps = ple ? a.ple_taps : a.taps;
  const int c =
      (static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x)) + static_cast<int>(threadIdx.x);
  if (c >= channels) {
    return;
  }
  float old[16];
  for (int j = 0; j < taps; ++j) {
    old[j] = history[(static_cast<std::int64_t>(c) * taps) + j];
  }
  for (int j = 0; j < taps; ++j) {
    const int at = a.keep + j;  // in the old taps followed by the rows
    history[(static_cast<std::int64_t>(c) * taps) + j] =
        at < taps ? old[at] : rows[(static_cast<std::int64_t>(at - taps) * channels) + c];
  }
}

// The same recurrence with each value column split over 8 lanes of 16
// rows (4 columns a warp, 16 warps a block: 64 columns of one head; lane
// part p holds rows 32j + 4p .. 32j + 4p + 3, so that the 8 parts' float4
// reads of a staged row are 128 contiguous bytes, free of bank conflicts). The
// recurrence is sequential in tokens, and each token's k, q and v come from
// memory: read one token at a time, every step waits a memory latency. So
// the block stages chunks of kGdnChunk tokens' k, q, v, gate and beta in
// shared memory with asynchronous copies, the next chunk's while it runs
// this one's. The column sums take three shuffle steps. The F32 sums run in
// another order than upstream's, so the results differ from its in the
// last bits (tests/unit/qwen38_fused_test.cc bounds them).
constexpr int kGdnChunk = 16;
constexpr int kGdnBlockColumns = 64;
struct GdnStage {
  float k[kGdnChunk][128];
  float q[kGdnChunk][128];
  float v[kGdnChunk][kGdnBlockColumns];
  float g[kGdnChunk];
  float beta[kGdnChunk];
};

__global__ void __launch_bounds__(512, 2)
    GdnLanesKernel(const float* __restrict__ q, const float* __restrict__ k,
                   const float* __restrict__ v, const float* __restrict__ g,
                   const float* __restrict__ beta, const float* __restrict__ state_in,
                   float* __restrict__ dst, float* __restrict__ state_out, int heads, int qk_heads,
                   int tokens, std::int64_t sq1, std::int64_t sq2, std::int64_t sv1,
                   std::int64_t sv2, std::int64_t sb1, std::int64_t sb2, float scale) {
  constexpr int kS = 128;
  constexpr int kRows = 16;  // rows a lane holds
  __shared__ __align__(16) GdnStage stage[2];
  const int lane = static_cast<int>(threadIdx.x);
  const int tid = (static_cast<int>(threadIdx.y) * 32) + lane;
  const int part = lane % 8;  // which 16 rows
  const int h = static_cast<int>(blockIdx.x);
  const int col0 = static_cast<int>(blockIdx.y) * kGdnBlockColumns;
  const int local = (static_cast<int>(threadIdx.y) * 4) + (lane / 8);  // column in the block
  const int col = col0 + local;
  const int hq = h % qk_heads;
  const auto load = [&](int chunk, GdnStage& to) {
    const int t0 = chunk * kGdnChunk;
    const int n = min(kGdnChunk, tokens - t0);
    for (int i = tid; i < n * 32; i += 512) {
      const int tok = i / 32;
      const int ch = i % 32;
      const std::int64_t at = ((t0 + tok) * sq2) + (hq * sq1) + (ch * 4);
      __pipeline_memcpy_async(&to.k[tok][ch * 4], k + at, 16);
      __pipeline_memcpy_async(&to.q[tok][ch * 4], q + at, 16);
    }
    for (int i = tid; i < n * (kGdnBlockColumns / 4); i += 512) {
      const int tok = i / (kGdnBlockColumns / 4);
      const int ch = i % (kGdnBlockColumns / 4);
      __pipeline_memcpy_async(&to.v[tok][ch * 4],
                              v + ((t0 + tok) * sv2) + (h * sv1) + col0 + (ch * 4), 16);
    }
    if (tid < n) {
      const std::int64_t gb = ((t0 + tid) * sb2) + (h * sb1);
      __pipeline_memcpy_async(&to.g[tid], g + gb, 4);
      __pipeline_memcpy_async(&to.beta[tid], beta + gb, 4);
    }
    __pipeline_commit();
  };
  // Element i of a lane's rows is row Row(i).
  const auto row = [part](int i) { return (32 * (i / 4)) + (4 * part) + (i % 4); };
  float s[kRows];
  const float* s0 =
      state_in + (static_cast<std::int64_t>(h) * kS * kS) + (static_cast<std::int64_t>(col) * kS);
#pragma unroll
  for (int i = 0; i < kRows; i += 4) {
    const float4 f = *reinterpret_cast<const float4*>(s0 + row(i));
    s[i] = f.x;
    s[i + 1] = f.y;
    s[i + 2] = f.z;
    s[i + 3] = f.w;
  }
  float* attn = dst + (static_cast<std::int64_t>(h) * kS) + col;
  const int chunks = (tokens + kGdnChunk - 1) / kGdnChunk;
  if (chunks > 0) {
    load(0, stage[0]);
  }
  for (int c = 0; c < chunks; ++c) {
    if (c + 1 < chunks) {
      load(c + 1, stage[(c + 1) % 2]);
      __pipeline_wait_prior(1);
    } else {
      __pipeline_wait_prior(0);
    }
    __syncthreads();
    const GdnStage& st = stage[c % 2];
    const int n = min(kGdnChunk, tokens - (c * kGdnChunk));
    for (int tok = 0; tok < n; ++tok) {
      const float beta_t = st.beta[tok];
      const float g_t = expf(st.g[tok]);
      const float v_t = st.v[tok][local];
      float kr[kRows];
#pragma unroll
      for (int i = 0; i < kRows; i += 4) {
        const float4 kf = *reinterpret_cast<const float4*>(&st.k[tok][row(i)]);
        kr[i] = kf.x;
        kr[i + 1] = kf.y;
        kr[i + 2] = kf.z;
        kr[i + 3] = kf.w;
      }
      float kv = 0.0f;
#pragma unroll
      for (int i = 0; i < kRows; ++i) {
        kv = fmaf(kr[i], s[i], kv);
      }
#pragma unroll
      for (int offset = 4; offset > 0; offset >>= 1) {
        kv += __shfl_xor_sync(0xffffffffu, kv, offset, 32);
      }
      const float delta = beta_t * fmaf(-g_t, kv, v_t);
      float at = 0.0f;
#pragma unroll
      for (int i = 0; i < kRows; i += 4) {
        const float4 qf = *reinterpret_cast<const float4*>(&st.q[tok][row(i)]);
        const float qr[4] = {qf.x, qf.y, qf.z, qf.w};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          s[i + j] = fmaf(g_t, s[i + j], kr[i + j] * delta);
          at = fmaf(qr[j], s[i + j], at);
        }
      }
#pragma unroll
      for (int offset = 4; offset > 0; offset >>= 1) {
        at += __shfl_xor_sync(0xffffffffu, at, offset, 32);
      }
      if (part == 0) {
        *attn = at * scale;
      }
      attn += static_cast<std::int64_t>(kS) * heads;
    }
    // The buffer is loaded again two chunks on.
    __syncthreads();
  }
  float* s1 =
      state_out + (static_cast<std::int64_t>(h) * kS * kS) + (static_cast<std::int64_t>(col) * kS);
#pragma unroll
  for (int i = 0; i < kRows; i += 4) {
    *reinterpret_cast<float4*>(s1 + row(i)) = make_float4(s[i], s[i + 1], s[i + 2], s[i + 3]);
  }
}

// rms_norm_f32<256>'s scale of a 128-value row held one value a thread by
// the 128 threads of a block: its four warps' sums, then the sum over the
// eight warps GGML's 256-thread block has (the last four hold zeros).
__device__ float RowScale128(float v, float eps, float* shared) {
  const int tid = static_cast<int>(threadIdx.x);
  float tmp = WarpSum(fmaf(v, v, 0.0f));
  if (tid % 32 == 0) {
    shared[tid / 32] = tmp;
  }
  __syncthreads();
  const int lane = tid % 32;
  tmp = WarpSum(lane < 4 ? shared[lane] : 0.0f);
  __syncthreads();
  return rsqrtf(fmaf(tmp, 1.0f / 128.0f, eps));
}

// One head of one token a block, a channel a thread: the 4-tap causal
// convolution over the history then the rows (ssm_conv's multiply-add chain
// and its zero bias), silu, and for the query and key heads the L2 norm
// (rms_norm with eps, then scale_f32's scale).
__device__ __forceinline__ float Load(const float* p) { return *p; }
__device__ __forceinline__ float Load(const nv_bfloat16* p) { return __bfloat162float(*p); }

template <typename T>
__global__ void __launch_bounds__(128)
    GdnConvKernel(const T* __restrict__ x, const float* __restrict__ history,
                  const float* __restrict__ weight, float* __restrict__ out, int channels,
                  int qk_channels, float eps, float scale, float bias) {
  __shared__ float shared[4];
  const int c = (static_cast<int>(blockIdx.x) * 128) + static_cast<int>(threadIdx.x);
  const int t = static_cast<int>(blockIdx.y);
  float in[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int tau = t + j;  // time in the history-then-rows sequence
    in[j] = tau < 3 ? history[(c * 3) + tau]
                    : Load(x + (static_cast<std::int64_t>(tau - 3) * channels) + c);
  }
  const float* w = weight + (c * 4);
  float sum = fmaf(in[0], w[0], 0.0f);
  sum = fmaf(in[1], w[1], sum);
  sum = fmaf(in[2], w[2], sum);
  sum = fmaf(in[3], w[3], sum);
  sum = sum + bias;
  const float v = sum / (1.0f + expf(-sum));  // ggml_cuda_op_silu_single
  float y = v;
  if (static_cast<int>(blockIdx.x) * 128 < qk_channels) {  // a whole head: uniform
    const float s = RowScale128(v, eps, shared);
    y = Scale(s * v, scale, bias);
  }
  out[(static_cast<std::int64_t>(t) * channels) + c] = y;
}

// One 128-value head a warp: rms_norm_f32<256>'s scale (each warp-sized
// quarter summed as one of its warps would, then the eight warps' sums),
// times the weight, times sigmoid(z).
template <typename T, typename Z>
__global__ void __launch_bounds__(256)
    GdnNormGateKernel(const float* __restrict__ o, const float* __restrict__ weight,
                      const Z* __restrict__ z, T* __restrict__ out, int rows, float eps) {
  ggml_cuda_pdl_lc();  // the output product after it fetches its weights meanwhile
  const int lane = static_cast<int>(threadIdx.x) % 32;
  const int row = (static_cast<int>(blockIdx.x) * 8) + (static_cast<int>(threadIdx.x) / 32);
  if (row >= rows) {
    return;
  }
  const float* x = o + (static_cast<std::int64_t>(row) * 128);
  const Z* g = z + (static_cast<std::int64_t>(row) * 128);
  float v[4];
  float quarter[4];
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    v[k] = x[(32 * k) + lane];
    quarter[k] = WarpSum(fmaf(v[k], v[k], 0.0f));
  }
  const float mine = lane == 0   ? quarter[0]
                     : lane == 1 ? quarter[1]
                     : lane == 2 ? quarter[2]
                     : lane == 3 ? quarter[3]
                                 : 0.0f;
  const float tmp = WarpSum(mine);
  const float s = rsqrtf(fmaf(tmp, 1.0f / 128.0f, eps));
  T* dst = out + (static_cast<std::int64_t>(row) * 128);
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const int d = (32 * k) + lane;
    dst[d] = Store<T>(s * v[k] * weight[d] * Sigmoid(Load(g + d)));
  }
}

// The same, quantized to MXFP8 for the output projection (the fast graph's):
// each 32-value block is a quarter of the head, one value a lane, so its
// scale is a warp's max. Row r is head r % heads of token r / heads, so its
// codes are the rows' codes from r · 128.
template <typename Z>
__global__ void __launch_bounds__(256)
    GdnNormGateMxfp8Kernel(const float* __restrict__ o, const float* __restrict__ weight,
                           const Z* __restrict__ z, std::uint8_t* __restrict__ codes,
                           std::uint8_t* __restrict__ scales, int rows, int heads, float eps) {
  ggml_cuda_pdl_lc();
  const int lane = static_cast<int>(threadIdx.x) % 32;
  const int row = (static_cast<int>(blockIdx.x) * 8) + (static_cast<int>(threadIdx.x) / 32);
  if (row >= rows) {
    return;
  }
  const float* x = o + (static_cast<std::int64_t>(row) * 128);
  const Z* g = z + (static_cast<std::int64_t>(row) * 128);
  float v[4];
  float sum = 0.0f;
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    v[k] = x[(32 * k) + lane];
    sum = fmaf(v[k], v[k], sum);
  }
  const float s = rsqrtf(fmaf(WarpSum(sum), 1.0f / 128.0f, eps));
  const auto t = static_cast<std::uint64_t>(row / heads);
  const auto h = static_cast<std::uint64_t>(row % heads);
  const auto blocks = static_cast<std::uint64_t>(heads) * 4;
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const int d = (32 * k) + lane;
    const float y = s * v[k] * weight[d] * Sigmoid(Load(g + d));
    float amax = fabsf(y);
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, offset, 32));
    }
    const std::uint32_t e = mxfp8::ScaleCode(amax);
    codes[(static_cast<std::int64_t>(row) * 128) + d] = static_cast<std::uint8_t>(
        __nv_cvt_float_to_fp8(y * mxfp8::InverseScale(e), __NV_SATFINITE, __NV_E4M3));
    if (lane == 0) {
      scales[moe::SfOffset(t, (h * 4) + static_cast<std::uint64_t>(k), blocks)] =
          static_cast<std::uint8_t>(e);
    }
  }
}

// A head of a token a warp (dimension lane + 32 j in the lane's slot j): its
// rms_norm times the weight, then the NEOX rotation of dimensions (lane,
// lane + 32), both in the lane, by pos · theta_scale^lane (GGML's rope_multi
// with every section at the token's position).
constexpr int kQsaSlots = 16;  // d / 32, at most
__global__ void __launch_bounds__(256)
    QsaPrepKernel(const float* __restrict__ x, const float* __restrict__ weight,
                  const std::int32_t* __restrict__ positions, float* __restrict__ out, int d,
                  int heads, int stride, std::int64_t row, int items, float eps,
                  float theta_scale) {
  const int lane = static_cast<int>(threadIdx.x) % 32;
  const int g = (static_cast<int>(blockIdx.x) * 8) + (static_cast<int>(threadIdx.x) / 32);
  if (g >= items) {
    return;
  }
  const int t = g / heads;
  const int h = g % heads;
  const int slots = d / 32;
  const float* src =
      x + (static_cast<std::int64_t>(t) * row) + (static_cast<std::int64_t>(h) * stride);
  float v[kQsaSlots];
  float sum = 0.0f;
#pragma unroll
  for (int j = 0; j < kQsaSlots; ++j) {
    v[j] = j < slots ? src[(32 * j) + lane] : 0.0f;
    sum = fmaf(v[j], v[j], sum);
  }
  const float s = rsqrtf(fmaf(WarpSum(sum), 1.0f / static_cast<float>(d), eps));
#pragma unroll
  for (int j = 0; j < kQsaSlots; ++j) {
    if (j < slots) {
      v[j] = (v[j] * s) * weight[(32 * j) + lane];
    }
  }
  const float theta =
      static_cast<float>(positions[t]) * powf(theta_scale, static_cast<float>(lane));
  const float c = cosf(theta);
  const float sn = sinf(theta);
  const float x0 = v[0];
  const float x1 = v[1];
  v[0] = (x0 * c) - (x1 * sn);
  v[1] = (x0 * sn) + (x1 * c);
  float* dst = out + (static_cast<std::int64_t>(g) * d);
#pragma unroll
  for (int j = 0; j < kQsaSlots; ++j) {
    if (j < slots) {
      dst[(32 * j) + lane] = v[j];
    }
  }
}

// A 32-value block of a token a thread: the attention's output times the
// sigmoid of its gate, quantized to MXFP8 (rows past t, up to the scale
// atoms' padding, write a zero scale).
__global__ void QsaGateQuantizeKernel(const float* __restrict__ attn,
                                      const float* __restrict__ q_full,
                                      std::uint8_t* __restrict__ codes,
                                      std::uint8_t* __restrict__ scales, int d, int k, int rows,
                                      std::int64_t items) {
  const std::int64_t i = (static_cast<std::int64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  if (i >= items) {
    return;
  }
  const int blocks = k / 32;
  const auto r = static_cast<int>(i / blocks);
  const auto b = static_cast<int>(i % blocks);
  const std::uint64_t at =
      moe::SfOffset(static_cast<std::uint64_t>(r), static_cast<std::uint64_t>(b),
                    static_cast<std::uint64_t>(blocks));
  if (r >= rows) {
    scales[at] = 0;
    return;
  }
  const int h = (b * 32) / d;
  const int o = (b * 32) % d;
  const auto* a =
      reinterpret_cast<const float4*>(attn + (static_cast<std::int64_t>(r) * k) + (b * 32));
  const auto* g = reinterpret_cast<const float4*>(q_full + (static_cast<std::int64_t>(r) * 2 * k) +
                                                  (static_cast<std::int64_t>(h) * 2 * d) + d + o);
  float y[32];
  float amax = 0.0f;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const float4 av = a[j];
    const float4 gv = g[j];
    y[(4 * j) + 0] = av.x * Sigmoid(gv.x);
    y[(4 * j) + 1] = av.y * Sigmoid(gv.y);
    y[(4 * j) + 2] = av.z * Sigmoid(gv.z);
    y[(4 * j) + 3] = av.w * Sigmoid(gv.w);
#pragma unroll
    for (int m = 0; m < 4; ++m) {
      amax = fmaxf(amax, fabsf(y[(4 * j) + m]));
    }
  }
  const std::uint32_t e = mxfp8::ScaleCode(amax);
  const float inverse = mxfp8::InverseScale(e);
  std::uint32_t packed[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    packed[j] =
        mxfp8::Pack4(y[(4 * j) + 0], y[(4 * j) + 1], y[(4 * j) + 2], y[(4 * j) + 3], inverse);
  }
  auto* dst = reinterpret_cast<uint4*>(codes + (static_cast<std::int64_t>(r) * k) + (b * 32));
  dst[0] = make_uint4(packed[0], packed[1], packed[2], packed[3]);
  dst[1] = make_uint4(packed[4], packed[5], packed[6], packed[7]);
  scales[at] = static_cast<std::uint8_t>(e);
}

// The last `taps` rows of x transposed into the conv state (tap j of channel
// c at c · taps + j), a channel a thread; with `history` (the old state, for
// fewer rows than taps) the last `taps` of the old taps followed by the rows.
template <typename T>
__global__ void GdnHistoryKernel(const T* __restrict__ x, const float* __restrict__ history,
                                 float* __restrict__ out, int channels, int t, int taps) {
  const int c = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
  if (c >= channels) {
    return;
  }
  for (int j = 0; j < taps; ++j) {
    const int i = t + j;  // in the old taps then the rows
    out[(static_cast<std::int64_t>(c) * taps) + j] =
        i < taps ? history[(static_cast<std::int64_t>(c) * taps) + i]
                 : Load(x + (static_cast<std::int64_t>(i - taps) * channels) + c);
  }
}

// The fast path's hyper-connections (llmp_ops.h llmp.hc.prep, .lo and
// .mix_bf16): not GGML's order of operations, so not bit for bit.

// Four BF16 values (8 bytes) as floats, and back.
__device__ __forceinline__ float4 Bf16x4(uint2 raw) {
  return make_float4(__uint_as_float(raw.x << 16U), __uint_as_float(raw.x & 0xffff0000U),
                     __uint_as_float(raw.y << 16U), __uint_as_float(raw.y & 0xffff0000U));
}
__device__ __forceinline__ uint2 ToBf16x4(float4 v) {
  const __nv_bfloat162 lo = __floats2bfloat162_rn(v.x, v.y);
  const __nv_bfloat162 hi = __floats2bfloat162_rn(v.z, v.w);
  return make_uint2(*reinterpret_cast<const std::uint32_t*>(&lo),
                    *reinterpret_cast<const std::uint32_t*>(&hi));
}
__device__ __forceinline__ float Dot4(float4 a, float4 b) {
  return fmaf(a.x, b.x, fmaf(a.y, b.y, fmaf(a.z, b.z, a.w * b.w)));
}

// Each of `hc` per-thread values summed over the block (warps of 32, at
// most 32 of them); every thread gets the sums. `shared` holds 32 · 8.
template <int kStreams>
__device__ __forceinline__ void BlockSums(float (&v)[kStreams], int hc, float* shared) {
  const int lane = static_cast<int>(threadIdx.x) % 32;
  const int warp = static_cast<int>(threadIdx.x) / 32;
  const int warps = static_cast<int>((blockDim.x + 31) / 32);
#pragma unroll
  for (int k = 0; k < kStreams; ++k) {
    if (k < hc) {
      const float s = WarpSum(v[k]);
      if (lane == 0) {
        shared[(warp * kStreams) + k] = s;
      }
    }
  }
  __syncthreads();
#pragma unroll
  for (int k = 0; k < kStreams; ++k) {
    if (k < hc) {
      v[k] = WarpSum(lane < warps ? shared[(lane * kStreams) + k] : 0.0f);
    }
  }
  __syncthreads();
}

// A token a block, a float4 column of every stream a thread: the combine
// (res + out · 2 sigmoid(logit / hc)) if asked, each stream's RMS scale, the
// normalized streams in BF16, and the next combine's logits from them.
template <bool kCombine, bool kInject>
__global__ void __launch_bounds__(1024)
    HcPrepKernel(const float4* __restrict__ x, const float4* __restrict__ norm,
                 const uint2* __restrict__ inject_w, const float4* __restrict__ out,
                 const float* __restrict__ logits_in, float4* __restrict__ streams,
                 uint2* __restrict__ normed, float* __restrict__ logits, int width, int hc,
                 float eps, float inv_hc) {
  __shared__ float shared[32 * kMaxStreams];
  __shared__ float gains[kMaxStreams];
  const std::int64_t t = blockIdx.x;
  const int c4 = static_cast<int>(threadIdx.x);
  const int w4 = width / 4;
  const bool live = c4 < w4;
  const std::int64_t row = t * hc * w4;  // token t's first float4
  if constexpr (kCombine) {
    if (threadIdx.x < static_cast<unsigned>(hc)) {
      gains[threadIdx.x] = 2.0f * Sigmoid(logits_in[(t * hc) + threadIdx.x] * inv_hc);
    }
    __syncthreads();
  }
  float4 v[kMaxStreams];
  float sums[kMaxStreams];
  float4 o = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  if (kCombine && live) {
    o = out[(t * w4) + c4];
  }
#pragma unroll
  for (int k = 0; k < kMaxStreams; ++k) {
    sums[k] = 0.0f;
    v[k] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (k < hc && live) {
      float4 r = x[row + (static_cast<std::int64_t>(k) * w4) + c4];
      if constexpr (kCombine) {
        const float g = gains[k];
        r = make_float4(fmaf(o.x, g, r.x), fmaf(o.y, g, r.y), fmaf(o.z, g, r.z), fmaf(o.w, g, r.w));
        streams[row + (static_cast<std::int64_t>(k) * w4) + c4] = r;
      }
      v[k] = r;
      sums[k] = Dot4(r, r);
    }
  }
  BlockSums(sums, hc, shared);
  float dots[kMaxStreams];
#pragma unroll
  for (int m = 0; m < kMaxStreams; ++m) {
    dots[m] = 0.0f;
  }
  const float inverse = 1.0f / static_cast<float>(width);
#pragma unroll
  for (int k = 0; k < kMaxStreams; ++k) {
    if (k < hc && live) {
      const float s = rsqrtf(fmaf(sums[k], inverse, eps));
      const float4 w = norm[(static_cast<std::int64_t>(k) * w4) + c4];
      const float4 xn =
          make_float4(s * v[k].x * w.x, s * v[k].y * w.y, s * v[k].z * w.z, s * v[k].w * w.w);
      normed[row + (static_cast<std::int64_t>(k) * w4) + c4] = ToBf16x4(xn);
      if constexpr (kInject) {
#pragma unroll
        for (int m = 0; m < kMaxStreams; ++m) {
          if (m < hc) {
            const std::int64_t at =
                (static_cast<std::int64_t>(m) * hc * w4) + (static_cast<std::int64_t>(k) * w4) + c4;
            dots[m] += Dot4(xn, Bf16x4(inject_w[at]));
          }
        }
      }
    }
  }
  if constexpr (kInject) {
    BlockSums(dots, hc, shared);
    if (threadIdx.x < static_cast<unsigned>(hc)) {
#pragma unroll
      for (int m = 0; m < kMaxStreams; ++m) {
        if (m == static_cast<int>(threadIdx.x)) {
          logits[(t * hc) + m] = dots[m];
        }
      }
    }
  }
}

// The decode form of HcPrepKernel (up to kHcPrepClusterTokens tokens): one
// block a token leaves all but one multiprocessor idle for 13-18 us a mix
// (nsys, spark-b, 2026-09-28), so a token takes a cluster of
// kHcPrepCluster blocks, each a slice of the float4 columns. Each block's
// partial sums (the streams' squares, then the inject dot products) go to
// its shared memory, and every block adds all the blocks' partials in block
// order through distributed shared memory, so each holds the same totals and
// the result repeats run to run. The arithmetic is HcPrepKernel's apart from
// that order of the sums.
constexpr int kHcPrepCluster = 8;
constexpr std::int64_t kHcPrepClusterTokens = 8;
template <bool kCombine, bool kInject>
__global__ void __launch_bounds__(256)
    HcPrepClusterKernel(const float4* __restrict__ x, const float4* __restrict__ norm,
                        const uint2* __restrict__ inject_w, const float4* __restrict__ out,
                        const float* __restrict__ logits_in, float4* __restrict__ streams,
                        uint2* __restrict__ normed, float* __restrict__ logits, int width, int hc,
                        float eps, float inv_hc) {
  // Thread-block clusters are sm_90 and later: the discrete sm_86 build
  // (D-082) compiles an empty body, which RunHcPrep never launches there.
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 900
  namespace cg = cooperative_groups;
  cg::cluster_group cluster = cg::this_cluster();
  __shared__ float shared[32 * kMaxStreams];
  __shared__ float gains[kMaxStreams];
  __shared__ float squares[kMaxStreams];  // this block's partial sums
  __shared__ float products[kMaxStreams];
  ggml_cuda_pdl_lc();  // the down product after it fetches its weights meanwhile
  const int rank = static_cast<int>(cluster.block_rank());
  const std::int64_t t = blockIdx.y;
  const int w4 = width / 4;
  const int per = (w4 + kHcPrepCluster - 1) / kHcPrepCluster;
  const int c4 = (rank * per) + static_cast<int>(threadIdx.x);
  const bool live = static_cast<int>(threadIdx.x) < per && c4 < w4;
  const std::int64_t row = t * hc * w4;
  // The norm and inject weights the second half reads depend on nothing
  // this kernel computes: into L2 before the first exchange.
  if (live) {
    for (int k = 0; k < hc; ++k) {
      asm volatile(
          "prefetch.global.L2 [%0];" ::"l"(norm + (static_cast<std::int64_t>(k) * w4) + c4));
      if constexpr (kInject) {
        for (int m = 0; m < hc; ++m) {
          asm volatile("prefetch.global.L2 [%0];" ::"l"(inject_w +
                                                        (static_cast<std::int64_t>(m) * hc * w4) +
                                                        (static_cast<std::int64_t>(k) * w4) + c4));
        }
      }
    }
  }
  if constexpr (kCombine) {
    if (threadIdx.x < static_cast<unsigned>(hc)) {
      gains[threadIdx.x] = 2.0f * Sigmoid(logits_in[(t * hc) + threadIdx.x] * inv_hc);
    }
    __syncthreads();
  }
  float4 v[kMaxStreams];
  float sums[kMaxStreams];
  float4 o = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  if (kCombine && live) {
    o = out[(t * w4) + c4];
  }
#pragma unroll
  for (int k = 0; k < kMaxStreams; ++k) {
    sums[k] = 0.0f;
    v[k] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (k < hc && live) {
      float4 r = x[row + (static_cast<std::int64_t>(k) * w4) + c4];
      if constexpr (kCombine) {
        const float g = gains[k];
        r = make_float4(fmaf(o.x, g, r.x), fmaf(o.y, g, r.y), fmaf(o.z, g, r.z), fmaf(o.w, g, r.w));
        streams[row + (static_cast<std::int64_t>(k) * w4) + c4] = r;
      }
      v[k] = r;
      sums[k] = Dot4(r, r);
    }
  }
  BlockSums(sums, hc, shared);
  if (threadIdx.x == 0) {
#pragma unroll
    for (int k = 0; k < kMaxStreams; ++k) {
      squares[k] = sums[k];
    }
  }
  cluster.sync();
#pragma unroll
  for (int k = 0; k < kMaxStreams; ++k) {
    sums[k] = 0.0f;
  }
  for (int b = 0; b < kHcPrepCluster; ++b) {
    const float* theirs = cluster.map_shared_rank(squares, b);
#pragma unroll
    for (int k = 0; k < kMaxStreams; ++k) {
      if (k < hc) {
        sums[k] += theirs[k];
      }
    }
  }
  float dots[kMaxStreams];
#pragma unroll
  for (int m = 0; m < kMaxStreams; ++m) {
    dots[m] = 0.0f;
  }
  const float inverse = 1.0f / static_cast<float>(width);
#pragma unroll
  for (int k = 0; k < kMaxStreams; ++k) {
    if (k < hc && live) {
      const float s = rsqrtf(fmaf(sums[k], inverse, eps));
      const float4 w = norm[(static_cast<std::int64_t>(k) * w4) + c4];
      const float4 xn =
          make_float4(s * v[k].x * w.x, s * v[k].y * w.y, s * v[k].z * w.z, s * v[k].w * w.w);
      normed[row + (static_cast<std::int64_t>(k) * w4) + c4] = ToBf16x4(xn);
      if constexpr (kInject) {
#pragma unroll
        for (int m = 0; m < kMaxStreams; ++m) {
          if (m < hc) {
            const std::int64_t at =
                (static_cast<std::int64_t>(m) * hc * w4) + (static_cast<std::int64_t>(k) * w4) + c4;
            dots[m] += Dot4(xn, Bf16x4(inject_w[at]));
          }
        }
      }
    }
  }
  if constexpr (kInject) {
    BlockSums(dots, hc, shared);
    if (threadIdx.x == 0) {
#pragma unroll
      for (int m = 0; m < kMaxStreams; ++m) {
        products[m] = dots[m];
      }
    }
    cluster.sync();
    if (rank == 0 && threadIdx.x < static_cast<unsigned>(hc)) {
      float total = 0.0f;
      for (int b = 0; b < kHcPrepCluster; ++b) {
        total += cluster.map_shared_rank(products, b)[threadIdx.x];
      }
      logits[(t * hc) + threadIdx.x] = total;
    }
  }
  // No block leaves while another may still read its shared memory.
  cluster.sync();
#endif
}

// llmp.gemm.bf16's vector form (GemvBf16, up to kGemvBf16Columns
// columns): y[n, t] = W[k, n] · x[k, t], BF16 weights and activations, F32
// sums, F32 or BF16 out. Every 16-byte vector of a thread's share of its
// row is loaded before any arithmetic, so each thread keeps VPL loads in
// flight (TensorFold's lesson for these shapes: enough bytes in flight, not
// tensor cores). Two shapes (scratch microbenchmark, spark-b, 2026-09-28,
// cold weights; cuBLAS's gemv 36.0 and 31.6 us):
//   GemvBf16Rows   a warp a row, for short rows (the up product, k 320):
//                  29.2 us;
//   GemvBf16Split  a block of 256 two rows, 128 threads a row and a fixed
//                  reduction order, for long rows (the down product, k
//                  10,240): 31.3 us.
__device__ __forceinline__ float DotBf16x8(const uint4 a, const uint4 b, float sum) {
  const auto* pa = reinterpret_cast<const __nv_bfloat162*>(&a);
  const auto* pb = reinterpret_cast<const __nv_bfloat162*>(&b);
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const float2 fa = __bfloat1622float2(pa[j]);
    const float2 fb = __bfloat1622float2(pb[j]);
    sum = fmaf(fa.x, fb.x, sum);
    sum = fmaf(fa.y, fb.y, sum);
  }
  return sum;
}

template <typename Out>
__device__ __forceinline__ void StoreOut(Out* y, std::int64_t at, float v) {
  if constexpr (std::is_same_v<Out, float>) {
    y[at] = v;
  } else {
    y[at] = __float2bfloat16(v);
  }
}

// Both are programmatic dependents (PDL): the weights' loads are issued
// before the wait for the kernel before, which then only gates x.
template <int kVpl, typename Out>
__global__ void __launch_bounds__(256)
    GemvBf16Rows(const uint4* __restrict__ w, const uint4* x, Out* y, int n, int k, int t) {
  const int lane = static_cast<int>(threadIdx.x) % 32;
  const int row = (static_cast<int>(blockIdx.x) * 8) + (static_cast<int>(threadIdx.x) / 32);
  if (row >= n) {
    return;
  }
  const int vectors = k / 8;
  const uint4* wr = w + (static_cast<std::int64_t>(row) * vectors);
  uint4 q[kVpl];
#pragma unroll
  for (int i = 0; i < kVpl; ++i) {
    const int v = lane + (i * 32);
    if (v < vectors) {
      q[i] = __ldg(wr + v);
    }
  }
  ggml_cuda_pdl_sync();
  ggml_cuda_pdl_lc();
  float sum[kGemvBf16Columns];
#pragma unroll
  for (int c = 0; c < kGemvBf16Columns; ++c) {
    sum[c] = 0.0f;
    if (c < t) {
#pragma unroll
      for (int i = 0; i < kVpl; ++i) {
        const int v = lane + (i * 32);
        if (v < vectors) {
          sum[c] = DotBf16x8(q[i], __ldg(x + (static_cast<std::int64_t>(c) * vectors) + v), sum[c]);
        }
      }
      sum[c] = WarpSum(sum[c]);
      if (lane == 0) {
        StoreOut(y, (static_cast<std::int64_t>(c) * n) + row, sum[c]);
      }
    }
  }
}

constexpr int kGemvSplitRows = 2;
template <int kVpl, typename Out>
__global__ void __launch_bounds__(256)
    GemvBf16Split(const uint4* __restrict__ w, const uint4* x, Out* y, int n, int k, int t) {
  constexpr int kPer = 256 / kGemvSplitRows;  // threads a row
  constexpr int kWarpsPer = kPer / 32;
  __shared__ float part[kGemvBf16Columns][256 / 32];
  const int sub = static_cast<int>(threadIdx.x) % kPer;
  const int local = static_cast<int>(threadIdx.x) / kPer;
  const int row = (static_cast<int>(blockIdx.x) * kGemvSplitRows) + local;
  const bool live = row < n;
  const int vectors = k / 8;
  const uint4* wr = w + (static_cast<std::int64_t>(live ? row : n - 1) * vectors);
  uint4 q[kVpl];
#pragma unroll
  for (int i = 0; i < kVpl; ++i) {
    const int v = sub + (i * kPer);
    if (v < vectors) {
      q[i] = __ldg(wr + v);
    }
  }
  ggml_cuda_pdl_sync();
  ggml_cuda_pdl_lc();
  const int warp = static_cast<int>(threadIdx.x) / 32;
#pragma unroll
  for (int c = 0; c < kGemvBf16Columns; ++c) {
    if (c < t) {
      float sum = 0.0f;
#pragma unroll
      for (int i = 0; i < kVpl; ++i) {
        const int v = sub + (i * kPer);
        if (v < vectors) {
          sum = DotBf16x8(q[i], __ldg(x + (static_cast<std::int64_t>(c) * vectors) + v), sum);
        }
      }
      sum = WarpSum(sum);
      if (threadIdx.x % 32 == 0) {
        part[c][warp] = sum;
      }
    }
  }
  __syncthreads();
  if (live && sub < t) {
    float total = 0.0f;
    for (int i = 0; i < kWarpsPer; ++i) {
      total += part[sub][(local * kWarpsPer) + i];
    }
    StoreOut(y, (static_cast<std::int64_t>(sub) * n) + row, total);
  }
}

template <typename Out>
void LaunchGemvBf16As(const ggml_tensor* node, cudaStream_t s) {
  const auto* w = static_cast<const uint4*>(node->src[0]->data);
  const auto* x = static_cast<const uint4*>(node->src[1]->data);
  auto* y = static_cast<Out*>(node->data);
  const int k = static_cast<int>(node->src[0]->ne[0]);
  const int n = static_cast<int>(node->src[0]->ne[1]);
  const int t = static_cast<int>(node->src[1]->ne[1]);
  const int vectors = k / 8;
  const auto rows_grid = static_cast<unsigned>((n + 7) / 8);
  const auto split_grid = static_cast<unsigned>((n + kGemvSplitRows - 1) / kGemvSplitRows);
  const auto go = [&](auto kernel, unsigned grid) {
    ggml_cuda_kernel_launch(kernel, ggml_cuda_kernel_launch_params(dim3(grid), dim3(256), 0, s), w,
                            x, y, n, k, t);
  };
  if (vectors <= 32 * 2) {
    go(GemvBf16Rows<2, Out>, rows_grid);
  } else if (vectors <= 32 * 12) {
    go(GemvBf16Rows<12, Out>, rows_grid);
  } else if (vectors <= 128 * 10) {
    go(GemvBf16Split<10, Out>, split_grid);
  } else {
    go(GemvBf16Split<20, Out>, split_grid);
  }
}

void LaunchGemvBf16(const ggml_tensor* node, cudaStream_t s) {
  if (node->type == GGML_TYPE_BF16) {
    LaunchGemvBf16As<__nv_bfloat16>(node, s);
  } else {
    LaunchGemvBf16As<float>(node, s);
  }
}

// silu(lo / hc) in BF16.
__global__ void HcLoKernel(const float* __restrict__ lo, nv_bfloat16* __restrict__ dst,
                           std::int64_t n, float inv_hc) {
  // The up product after it (a programmatic dependent) may launch and fetch
  // its weights now; it waits for this kernel before reading its output.
  ggml_cuda_pdl_lc();
  const std::int64_t i = (static_cast<std::int64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  if (i < n) {
    dst[i] = __float2bfloat16(Silu(lo[i] * inv_hc));
  }
}

// Eight columns of a token a thread: (1 / hc) Σ_k xn_k · sigmoid(g_k); with
// kQuantize also its MXFP8 codes (four threads a 32-value block, whose
// scale their shuffled max gives) and, with kRound, its BF16.
template <bool kQuantize, bool kRound>
__global__ void HcMixBf16Kernel(const uint4* __restrict__ normed, const uint4* __restrict__ gate,
                                float4* __restrict__ dst, std::uint8_t* __restrict__ codes,
                                std::uint8_t* __restrict__ scales, uint4* __restrict__ rounded,
                                int width8, int hc, std::int64_t items, float inv_hc) {
  ggml_cuda_pdl_lc();  // the products after it fetch their weights meanwhile
  const std::int64_t i = (static_cast<std::int64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  if (i >= items) {
    return;
  }
  const std::int64_t t = i / width8;
  const std::int64_t c8 = i % width8;
  float4 lo = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  float4 hi = lo;
  for (int k = 0; k < hc; ++k) {
    const std::int64_t at = (((t * hc) + k) * width8) + c8;
    const uint4 n = normed[at];
    const uint4 g = gate[at];
    const float4 n0 = Bf16x4(make_uint2(n.x, n.y));
    const float4 n1 = Bf16x4(make_uint2(n.z, n.w));
    const float4 g0 = Bf16x4(make_uint2(g.x, g.y));
    const float4 g1 = Bf16x4(make_uint2(g.z, g.w));
    lo = make_float4(fmaf(n0.x, Sigmoid(g0.x), lo.x), fmaf(n0.y, Sigmoid(g0.y), lo.y),
                     fmaf(n0.z, Sigmoid(g0.z), lo.z), fmaf(n0.w, Sigmoid(g0.w), lo.w));
    hi = make_float4(fmaf(n1.x, Sigmoid(g1.x), hi.x), fmaf(n1.y, Sigmoid(g1.y), hi.y),
                     fmaf(n1.z, Sigmoid(g1.z), hi.z), fmaf(n1.w, Sigmoid(g1.w), hi.w));
  }
  lo = make_float4(lo.x * inv_hc, lo.y * inv_hc, lo.z * inv_hc, lo.w * inv_hc);
  hi = make_float4(hi.x * inv_hc, hi.y * inv_hc, hi.z * inv_hc, hi.w * inv_hc);
  dst[2 * i] = lo;
  dst[(2 * i) + 1] = hi;
  if constexpr (kQuantize) {
    // Rows are whole blocks of four threads, and a block's threads share a
    // warp, so the four are all here.
    const unsigned lane = threadIdx.x % 32U;
    const unsigned group = 0xfU << (lane & ~3U);
    float amax = fmaxf(fmaxf(fmaxf(fabsf(lo.x), fabsf(lo.y)), fmaxf(fabsf(lo.z), fabsf(lo.w))),
                       fmaxf(fmaxf(fabsf(hi.x), fabsf(hi.y)), fmaxf(fabsf(hi.z), fabsf(hi.w))));
    amax = fmaxf(amax, __shfl_xor_sync(group, amax, 1, 32));
    amax = fmaxf(amax, __shfl_xor_sync(group, amax, 2, 32));
    const std::uint32_t e = mxfp8::ScaleCode(amax);
    const float inverse = mxfp8::InverseScale(e);
    *reinterpret_cast<uint2*>(codes + (i * 8)) =
        make_uint2(mxfp8::Pack4(lo.x, lo.y, lo.z, lo.w, inverse),
                   mxfp8::Pack4(hi.x, hi.y, hi.z, hi.w, inverse));
    if (c8 % 4 == 0) {
      scales[moe::SfOffset(static_cast<std::uint64_t>(t), static_cast<std::uint64_t>(c8 / 4),
                           static_cast<std::uint64_t>(width8 / 4))] = static_cast<std::uint8_t>(e);
    }
    if constexpr (kRound) {
      const uint2 a = ToBf16x4(lo);
      const uint2 b = ToBf16x4(hi);
      rounded[i] = make_uint4(a.x, a.y, b.x, b.y);
    }
  }
}

// A token a warp: the softmax of its router logits (expert lane + 32 i in
// lane `lane`'s slot i), the `used` most probable experts one warp argmax at
// a time (the lower index first among equals), their probabilities over
// their clamped sum, and the shared expert's gate logit.
constexpr int kRouterWarps = 8;
constexpr int kRouterSlots = 32;  // experts / 32, at most
// The gate row's four values at `c4`: BF16 (uint2) or F32 (float4).
__device__ __forceinline__ float4 GateRow4(const uint2* __restrict__ row, int c4) {
  return Bf16x4(row[c4]);
}
__device__ __forceinline__ float4 GateRow4(const float4* __restrict__ row, int c4) {
  return row[c4];
}
template <typename GateRow>
__global__ void __launch_bounds__(kRouterWarps * 32)
    MoeRouterKernel(const float* __restrict__ logits, const float4* __restrict__ x,
                    const GateRow* __restrict__ gate_row, std::int32_t* __restrict__ ids,
                    float* __restrict__ weights, float* __restrict__ gate, int tokens, int slots,
                    int used, int width4) {
  ggml_cuda_pdl_lc();  // the routed products after it may launch meanwhile
  const int lane = static_cast<int>(threadIdx.x) % 32;
  const std::int64_t t =
      (static_cast<std::int64_t>(blockIdx.x) * kRouterWarps) + (threadIdx.x / 32);
  if (t >= tokens) {
    return;
  }
  const float* l = logits + (t * slots * 32);
  float p[kRouterSlots];
  float most = -INFINITY;
#pragma unroll
  for (int i = 0; i < kRouterSlots; ++i) {
    p[i] = i < slots ? l[(32 * i) + lane] : -INFINITY;
    most = fmaxf(most, p[i]);
  }
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    most = fmaxf(most, __shfl_xor_sync(0xffffffffu, most, offset, 32));
  }
  float sum = 0.0f;
#pragma unroll
  for (int i = 0; i < kRouterSlots; ++i) {
    p[i] = i < slots ? expf(p[i] - most) : -1.0f;
    sum += i < slots ? p[i] : 0.0f;
  }
  const float inverse = 1.0f / WarpSum(sum);
  // A NaN probability (from NaN or all -inf logits) counts as 0, so every
  // pick is a distinct expert within range, never the argmax's sentinel.
#pragma unroll
  for (int i = 0; i < kRouterSlots; ++i) {
    const float q = p[i] * inverse;
    p[i] = i < slots ? (q >= 0.0f ? q : 0.0f) : -1.0f;
  }
  float mine = 0.0f;  // lane j: the j-th pick's probability
  float picked = 0.0f;
  for (int j = 0; j < used; ++j) {
    float best = -1.0f;
    int index = 0x7fffffff;
#pragma unroll
    for (int i = 0; i < kRouterSlots; ++i) {
      if (p[i] > best) {
        best = p[i];
        index = (32 * i) + lane;
      }
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      const float b = __shfl_xor_sync(0xffffffffu, best, offset, 32);
      const int bi = __shfl_xor_sync(0xffffffffu, index, offset, 32);
      if (b > best || (b == best && bi < index)) {
        best = b;
        index = bi;
      }
    }
    if ((index % 32) == lane) {
#pragma unroll
      for (int i = 0; i < kRouterSlots; ++i) {
        if (i == index / 32) {
          p[i] = -1.0f;
        }
      }
    }
    if (lane == j) {
      mine = best;
    }
    if (lane == 0) {
      ids[(t * used) + j] = index;
    }
    picked += best;
  }
  if (lane < used) {
    weights[(t * used) + lane] = mine / fmaxf(picked, 6.103515625e-5f);
  }
  float dot = 0.0f;
  for (int c4 = lane; c4 < width4; c4 += 32) {
    dot += Dot4(x[(t * width4) + c4], GateRow4(gate_row, c4));
  }
  dot = WarpSum(dot);
  if (lane == 0) {
    gate[t] = dot;
  }
}

// Each intermediate rounds as its own GGML F32 node. This file has
// GGML's fast-math flags; expf/logf and the 20.0 threshold are unary.cu's
// expressions, including their FTZ behavior. No recurrence or state write.
__global__ void GdnGatesKernel(const float* __restrict__ alpha, const float* __restrict__ beta,
                               const float* __restrict__ dt_bias, const float* __restrict__ ssm_a,
                               float* __restrict__ dst, int n) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= n) {
    return;
  }
  const int h = i % 48;
  const float x = __fadd_rn(alpha[i], dt_bias[h]);
  const float sp = x > 20.0f ? x : logf(1.0f + expf(x));
  dst[i] = __fmul_rn(sp, ssm_a[h]);
  dst[n + i] = Sigmoid(beta[i]);
}

__global__ void Bf16Kernel(const float* __restrict__ x, nv_bfloat16* __restrict__ dst,
                           std::int64_t n) {
  const std::int64_t i = (static_cast<std::int64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  if (i < n) {
    dst[i] = __float2bfloat16(x[i]);
  }
}

unsigned Blocks(std::int64_t items, int threads) {
  return static_cast<unsigned>((items + threads - 1) / threads);
}

}  // namespace

std::expected<void, KernelFailure> RunHcCombine(LaunchContext& launch, ggml_tensor* node) {
  if (auto checked = CheckHcCombine(node); !checked) {
    return checked;
  }
  return launch.Run(base::Bytes(0), [node](ggml_backend_cuda_context& context) {
    const ggml_tensor* res = node->src[0];
    const std::int64_t n4 = ggml_nelements(node) / 4;
    const int hc = static_cast<int>(res->ne[1]);
    HcCombineKernel<<<Blocks(n4, kThreads), kThreads, 0, context.stream()>>>(
        static_cast<const float4*>(res->data), static_cast<const float4*>(node->src[1]->data),
        static_cast<const float*>(node->src[2]->data), static_cast<float4*>(node->data),
        static_cast<int>(res->ne[0] / 4), hc, n4, 1.0f / static_cast<float>(hc), 0.0f);
  });
}

std::expected<void, KernelFailure> RunHcNorm(LaunchContext& launch, ggml_tensor* node) {
  if (auto checked = CheckHcNorm(node); !checked) {
    return checked;
  }
  return launch.Run(base::Bytes(0), [node](ggml_backend_cuda_context& context) {
    const ggml_tensor* x = node->src[0];
    const int width = static_cast<int>(x->ne[0]);
    const int hc = static_cast<int>(x->ne[1]);
    const dim3 grid(static_cast<unsigned>(x->ne[2]), static_cast<unsigned>(hc));
    const auto* xs = static_cast<const float*>(x->data);
    const auto* w = static_cast<const float*>(node->src[1]->data);
    const float eps = LlmpOpEps(node);
    if (node->type == GGML_TYPE_BF16) {
      HcNormKernel<nv_bfloat16><<<grid, kNormThreads, 0, context.stream()>>>(
          xs, w, static_cast<nv_bfloat16*>(node->data), width, hc, eps);
    } else {
      HcNormKernel<float><<<grid, kNormThreads, 0, context.stream()>>>(
          xs, w, static_cast<float*>(node->data), width, hc, eps);
    }
  });
}

std::expected<void, KernelFailure> RunHcMix(LaunchContext& launch, ggml_tensor* node) {
  if (auto checked = CheckHcMix(node); !checked) {
    return checked;
  }
  return launch.Run(base::Bytes(0), [node](ggml_backend_cuda_context& context) {
    const ggml_tensor* x = node->src[0];
    const int hc = static_cast<int>(x->ne[1]);
    HcMixKernel<<<static_cast<unsigned>(x->ne[2]), kNormThreads, 0, context.stream()>>>(
        static_cast<const float*>(x->data), static_cast<const float*>(node->src[1]->data),
        static_cast<const float*>(node->src[2]->data), static_cast<float*>(node->data),
        static_cast<int>(x->ne[0]), hc, LlmpOpEps(node), 1.0f / static_cast<float>(hc), 0.0f);
  });
}

std::expected<void, KernelFailure> RunMoeGlu(LaunchContext& launch, ggml_tensor* node) {
  if (auto checked = CheckMoeGlu(node); !checked) {
    return checked;
  }
  return launch.Run(base::Bytes(0), [node](ggml_backend_cuda_context& context) {
    const ggml_tensor* ids = node->src[2];
    const std::int64_t total4 = ggml_nelements(node) / 4;
    MoeGluKernel<<<Blocks(total4, kThreads), kThreads, 0, context.stream()>>>(
        static_cast<const float4*>(node->src[0]->data),
        static_cast<const float4*>(node->src[1]->data), static_cast<const std::int32_t*>(ids->data),
        static_cast<const float*>(node->src[3]->data),
        static_cast<const float*>(node->src[4]->data), static_cast<float4*>(node->data),
        static_cast<int>(node->ne[0] / 4), static_cast<int>(node->ne[1]),
        static_cast<int>(ids->nb[1] / sizeof(std::int32_t)), static_cast<int>(node->src[3]->ne[0]),
        total4);
  });
}

std::expected<void, KernelFailure> RunMoeCombine(LaunchContext& launch, ggml_tensor* node) {
  if (auto checked = CheckMoeCombine(node); !checked) {
    return checked;
  }
  return launch.Run(base::Bytes(0), [node](ggml_backend_cuda_context& context) {
    const ggml_tensor* down = node->src[0];
    const ggml_tensor* ids = node->src[1];
    // Unscaled (LlmpOpInt 0): the scale's slot left out.
    const bool unscaled = LlmpOpInt(node, 0) == 1;
    const int at = unscaled ? 2 : 3;
    const ggml_tensor* scale = unscaled ? nullptr : node->src[2];
    const std::int64_t total4 = ggml_nelements(node) / 4;
    MoeCombineKernel<<<Blocks(total4, kThreads), kThreads, 0, context.stream()>>>(
        static_cast<const float4*>(down->data), static_cast<const std::int32_t*>(ids->data),
        scale != nullptr ? static_cast<const float*>(scale->data) : nullptr,
        static_cast<const float*>(node->src[at]->data),
        static_cast<const float4*>(node->src[at + 1]->data),
        static_cast<const float*>(node->src[at + 2]->data), static_cast<float4*>(node->data),
        static_cast<int>(down->ne[0] / 4), static_cast<int>(down->ne[1]),
        static_cast<int>(ids->nb[1] / sizeof(std::int32_t)),
        scale != nullptr ? static_cast<int>(scale->ne[0]) : 0, total4);
  });
}

std::expected<void, KernelFailure> RunGatedDeltaNetColumns(LaunchContext& launch,
                                                           ggml_tensor* node) {
  if (auto checked = CheckGatedDeltaNetColumns(node); !checked) {
    return checked;
  }
  return launch.Run(base::Bytes(0), [node](ggml_backend_cuda_context& context) {
    const ggml_tensor* q = node->src[0];
    const ggml_tensor* v = node->src[2];
    const ggml_tensor* beta = node->src[4];
    constexpr int kS = 128;
    const int heads = static_cast<int>(v->ne[1]);
    const int tokens = static_cast<int>(v->ne[2]);
    const auto f = [](std::size_t bytes) {
      return static_cast<std::int64_t>(bytes / sizeof(float));
    };
    auto* dst = static_cast<float*>(node->data);
    const dim3 block(32, 4);
    const dim3 grid(static_cast<unsigned>(heads), kS / (4 * kGdnColumns));
    GdnColumnsKernel<kGdnColumns><<<grid, block, 0, context.stream()>>>(
        static_cast<const float*>(q->data), static_cast<const float*>(node->src[1]->data),
        static_cast<const float*>(v->data), static_cast<const float*>(node->src[3]->data),
        static_cast<const float*>(beta->data), static_cast<const float*>(node->src[5]->data), dst,
        dst + (static_cast<std::int64_t>(kS) * heads * tokens), heads, static_cast<int>(q->ne[1]),
        tokens, f(q->nb[1]), f(q->nb[2]), f(v->nb[1]), f(v->nb[2]), f(beta->nb[1]), f(beta->nb[2]),
        1.0f / sqrtf(static_cast<float>(kS)));
  });
}

std::expected<void, KernelFailure> RunGdnGates(LaunchContext& launch, ggml_tensor* node) {
  if (auto checked = CheckGdnGates(node); !checked) {
    return checked;
  }
  return launch.Run(base::Bytes(0), [node](ggml_backend_cuda_context& context) {
    const int n = 48 * static_cast<int>(node->ne[1]);
    GdnGatesKernel<<<Blocks(n, kThreads), kThreads, 0, context.stream()>>>(
        static_cast<const float*>(node->src[0]->data),
        static_cast<const float*>(node->src[1]->data),
        static_cast<const float*>(node->src[2]->data),
        static_cast<const float*>(node->src[3]->data), static_cast<float*>(node->data), n);
  });
}

std::expected<void, KernelFailure> RunGdnStep(LaunchContext& launch, ggml_tensor* node) {
  if (auto checked = CheckGdnStep(node); !checked) {
    return checked;
  }
  return launch.Run(base::Bytes(0), [node](ggml_backend_cuda_context& context) {
    const ggml_tensor* q = node->src[0];
    const ggml_tensor* v = node->src[2];
    const ggml_tensor* beta = node->src[4];
    constexpr int kS = 128;
    const int heads = static_cast<int>(v->ne[1]);
    const int tokens = static_cast<int>(v->ne[2]);
    const auto f = [](std::size_t bytes) {
      return static_cast<std::int64_t>(bytes / sizeof(float));
    };
    auto* state = static_cast<float*>(node->src[5]->data);
    // A verify's step (LlmpOpInt 0 == 1) reads the state and writes none.
    float* state_out = LlmpOpInt(node, 0) == 1 ? nullptr : state;
    const dim3 block(32, 4);
    const dim3 grid(static_cast<unsigned>(heads), kS / (4 * kGdnColumns));
    GdnColumnsKernel<kGdnColumns><<<grid, block, 0, context.stream()>>>(
        static_cast<const float*>(q->data), static_cast<const float*>(node->src[1]->data),
        static_cast<const float*>(v->data), static_cast<const float*>(node->src[3]->data),
        static_cast<const float*>(beta->data), state, static_cast<float*>(node->data), state_out,
        heads, static_cast<int>(q->ne[1]), tokens, f(q->nb[1]), f(q->nb[2]), f(v->nb[1]),
        f(v->nb[2]), f(beta->nb[1]), f(beta->nb[2]), 1.0f / sqrtf(static_cast<float>(kS)));
  });
}

std::expected<void, KernelFailure> RunGatedDeltaNetLanes(LaunchContext& launch, ggml_tensor* node) {
  if (auto checked = CheckGatedDeltaNetLanes(node); !checked) {
    return checked;
  }
  return launch.Run(base::Bytes(0), [node](ggml_backend_cuda_context& context) {
    const ggml_tensor* q = node->src[0];
    const ggml_tensor* v = node->src[2];
    const ggml_tensor* beta = node->src[4];
    constexpr int kS = 128;
    const int heads = static_cast<int>(v->ne[1]);
    const int tokens = static_cast<int>(v->ne[2]);
    const auto f = [](std::size_t bytes) {
      return static_cast<std::int64_t>(bytes / sizeof(float));
    };
    auto* dst = static_cast<float*>(node->data);
    const dim3 block(32, kGdnBlockColumns / 4);
    const dim3 grid(static_cast<unsigned>(heads), kS / kGdnBlockColumns);
    GdnLanesKernel<<<grid, block, 0, context.stream()>>>(
        static_cast<const float*>(q->data), static_cast<const float*>(node->src[1]->data),
        static_cast<const float*>(v->data), static_cast<const float*>(node->src[3]->data),
        static_cast<const float*>(beta->data), static_cast<const float*>(node->src[5]->data), dst,
        dst + (static_cast<std::int64_t>(kS) * heads * tokens), heads, static_cast<int>(q->ne[1]),
        tokens, f(q->nb[1]), f(q->nb[2]), f(v->nb[1]), f(v->nb[2]), f(beta->nb[1]), f(beta->nb[2]),
        1.0f / sqrtf(static_cast<float>(kS)));
  });
}

std::expected<void, KernelFailure> RunGdnConv(LaunchContext& launch, ggml_tensor* node) {
  if (auto checked = CheckGdnConv(node); !checked) {
    return checked;
  }
  return launch.Run(base::Bytes(0), [node](ggml_backend_cuda_context& context) {
    const ggml_tensor* x = node->src[0];
    const dim3 grid(static_cast<unsigned>(x->ne[0] / 128), static_cast<unsigned>(x->ne[1]));
    const auto launch_with = [&](const auto* rows) {
      GdnConvKernel<<<grid, 128, 0, context.stream()>>>(
          rows, static_cast<const float*>(node->src[1]->data),
          static_cast<const float*>(node->src[2]->data), static_cast<float*>(node->data),
          static_cast<int>(x->ne[0]), LlmpOpInt(node, 0), LlmpOpFloat(node, 2),
          LlmpOpFloat(node, 3), 0.0f);
    };
    if (x->type == GGML_TYPE_BF16) {
      launch_with(static_cast<const nv_bfloat16*>(x->data));
    } else {
      launch_with(static_cast<const float*>(x->data));
    }
  });
}

std::expected<void, KernelFailure> RunQsaPrep(LaunchContext& launch, ggml_tensor* node) {
  if (auto checked = CheckQsaPrep(node); !checked) {
    return checked;
  }
  return launch.Run(base::Bytes(0), [node](ggml_backend_cuda_context& context) {
    const ggml_tensor* x = node->src[0];
    const int d = LlmpOpInt(node, 0);
    const int heads = LlmpOpInt(node, 1);
    const int items = heads * LlmpOpInt(node, 3);
    QsaPrepKernel<<<static_cast<unsigned>((items + 7) / 8), 256, 0, context.stream()>>>(
        static_cast<const float*>(x->data), static_cast<const float*>(node->src[1]->data),
        static_cast<const std::int32_t*>(node->src[2]->data), static_cast<float*>(node->data), d,
        heads, LlmpOpInt(node, 2), static_cast<std::int64_t>(x->nb[1] / sizeof(float)), items,
        LlmpOpFloat(node, 4), LlmpOpFloat(node, 5));
  });
}

std::expected<void, KernelFailure> RunQsaGateQuantize(LaunchContext& launch, ggml_tensor* node) {
  if (auto checked = CheckQsaGateQuantize(node); !checked) {
    return checked;
  }
  return launch.Run(base::Bytes(0), [node](ggml_backend_cuda_context& context) {
    const int d = LlmpOpInt(node, 0);
    const int k = d * LlmpOpInt(node, 1);
    const int rows = LlmpOpInt(node, 2);
    const mxfp8::RowsLayout layout{.k = static_cast<std::uint64_t>(k),
                                   .rows = static_cast<std::uint64_t>(rows)};
    auto* base = static_cast<std::uint8_t*>(node->data);
    const std::int64_t items = static_cast<std::int64_t>(mxfp8::PaddedRows(layout.rows)) * (k / 32);
    QsaGateQuantizeKernel<<<Blocks(items, kThreads), kThreads, 0, context.stream()>>>(
        static_cast<const float*>(node->src[0]->data),
        static_cast<const float*>(node->src[1]->data), base + layout.codes(),
        base + layout.scales(), d, k, rows, items);
  });
}

std::expected<void, KernelFailure> RunGdnHistory(LaunchContext& launch, ggml_tensor* node) {
  if (auto checked = CheckGdnHistory(node); !checked) {
    return checked;
  }
  return launch.Run(base::Bytes(0), [node](ggml_backend_cuda_context& context) {
    const ggml_tensor* x = node->src[0];
    const int channels = static_cast<int>(x->ne[0]);
    const int t = static_cast<int>(x->ne[1]);
    const int taps = LlmpOpInt(node, 0);
    auto* out = static_cast<float*>(node->data);
    const auto* history =
        node->src[1] != nullptr ? static_cast<const float*>(node->src[1]->data) : nullptr;
    if (x->type == GGML_TYPE_BF16) {
      GdnHistoryKernel<<<Blocks(channels, kThreads), kThreads, 0, context.stream()>>>(
          static_cast<const nv_bfloat16*>(x->data), history, out, channels, t, taps);
    } else {
      GdnHistoryKernel<<<Blocks(channels, kThreads), kThreads, 0, context.stream()>>>(
          static_cast<const float*>(x->data), history, out, channels, t, taps);
    }
  });
}

std::expected<void, KernelFailure> RunGdnNormGate(LaunchContext& launch, ggml_tensor* node) {
  if (auto checked = CheckGdnNormGate(node); !checked) {
    return checked;
  }
  return launch.Run(base::Bytes(0), [node](ggml_backend_cuda_context& context) {
    const ggml_tensor* o = node->src[0];
    const int rows = static_cast<int>(o->ne[1] * o->ne[2]);
    const auto* x = static_cast<const float*>(o->data);
    const auto* w = static_cast<const float*>(node->src[1]->data);
    const float eps = LlmpOpEps(node);
    const unsigned blocks = static_cast<unsigned>((rows + 7) / 8);
    cudaStream_t stream = context.stream();
    const auto launch_with = [&](const auto* z) {
      if (node->type == GGML_TYPE_I8) {
        const mxfp8::RowsLayout layout{.k = static_cast<std::uint64_t>(o->ne[0] * o->ne[1]),
                                       .rows = static_cast<std::uint64_t>(o->ne[2])};
        auto* base = static_cast<std::uint8_t*>(node->data);
        // The padding rows' scales are zero, never stale bytes.
        (void)cudaMemsetAsync(base + layout.scales(), 0, layout.bytes() - layout.scales(), stream);
        GdnNormGateMxfp8Kernel<<<blocks, 256, 0, stream>>>(x, w, z, base + layout.codes(),
                                                           base + layout.scales(), rows,
                                                           static_cast<int>(o->ne[1]), eps);
      } else if (node->type == GGML_TYPE_BF16) {
        GdnNormGateKernel<<<blocks, 256, 0, stream>>>(
            x, w, z, static_cast<nv_bfloat16*>(node->data), rows, eps);
      } else {
        GdnNormGateKernel<<<blocks, 256, 0, stream>>>(x, w, z, static_cast<float*>(node->data),
                                                      rows, eps);
      }
    };
    if (node->src[2]->type == GGML_TYPE_BF16) {
      launch_with(static_cast<const nv_bfloat16*>(node->src[2]->data));
    } else {
      launch_with(static_cast<const float*>(node->src[2]->data));
    }
  });
}

std::expected<void, KernelFailure> RunBf16(LaunchContext& launch, ggml_tensor* node) {
  if (auto checked = CheckBf16(node); !checked) {
    return checked;
  }
  return launch.Run(base::Bytes(0), [node](ggml_backend_cuda_context& context) {
    const std::int64_t n = ggml_nelements(node);
    Bf16Kernel<<<Blocks(n, kThreads), kThreads, 0, context.stream()>>>(
        static_cast<const float*>(node->src[0]->data), static_cast<nv_bfloat16*>(node->data), n);
  });
}

std::expected<void, KernelFailure> RunHcPrep(LaunchContext& launch, ggml_tensor* node) {
  if (auto checked = CheckHcPrep(node); !checked) {
    return checked;
  }
  // The cluster form needs thread-block clusters (compute capability 9.0 and
  // later): not the discrete sm_86 (D-082), which keeps a block a token.
  const int cc = ggml_cuda_info().devices[launch.device()].cc;
  const bool clusters = GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_HOPPER;
  return launch.Run(base::Bytes(0), [node, clusters](ggml_backend_cuda_context& context) {
    const HcPrepLayout l{.width = LlmpOpInt(node, 0),
                         .hc = LlmpOpInt(node, 1),
                         .t = LlmpOpInt(node, 2),
                         .combine = LlmpOpInt(node, 3) == 1,
                         .inject = LlmpOpInt(node, 4) == 1};
    auto* blob = static_cast<std::uint8_t*>(node->data);
    const auto* x = static_cast<const float4*>(node->src[0]->data);
    const auto* norm = static_cast<const float4*>(node->src[1]->data);
    const int next = l.inject ? 3 : 2;
    const auto* inject = l.inject ? static_cast<const uint2*>(node->src[2]->data)
                                  : static_cast<const uint2*>(nullptr);
    const auto* out = l.combine ? static_cast<const float4*>(node->src[next]->data)
                                : static_cast<const float4*>(nullptr);
    const auto* logits_in = l.combine ? static_cast<const float*>(node->src[next + 1]->data)
                                      : static_cast<const float*>(nullptr);
    auto* streams = reinterpret_cast<float4*>(blob + HcPrepLayout::streams());
    auto* normed = reinterpret_cast<uint2*>(blob + l.normed());
    auto* logits = reinterpret_cast<float*>(blob + l.logits());
    const int width = static_cast<int>(l.width);
    const int hc = static_cast<int>(l.hc);
    const float eps = LlmpOpFloat(node, 5);
    const float inv_hc = 1.0f / static_cast<float>(hc);
    cudaStream_t s = context.stream();
    if (l.t <= kHcPrepClusterTokens && clusters) {
      const int per = ((width / 4) + kHcPrepCluster - 1) / kHcPrepCluster;
      cudaLaunchConfig_t config{};
      config.gridDim = dim3(kHcPrepCluster, static_cast<unsigned>(l.t));
      config.blockDim = dim3(static_cast<unsigned>((per + 31) / 32 * 32));
      config.stream = s;
      cudaLaunchAttribute attribute{};
      attribute.id = cudaLaunchAttributeClusterDimension;
      attribute.val.clusterDim.x = kHcPrepCluster;
      attribute.val.clusterDim.y = 1;
      attribute.val.clusterDim.z = 1;
      config.attrs = &attribute;
      config.numAttrs = 1;
      const auto go = [&](auto kernel) {
        CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, x, norm, inject, out, logits_in, streams,
                                      normed, logits, width, hc, eps, inv_hc));
      };
      if (l.combine && l.inject) {
        go(HcPrepClusterKernel<true, true>);
      } else if (l.combine) {
        go(HcPrepClusterKernel<true, false>);
      } else if (l.inject) {
        go(HcPrepClusterKernel<false, true>);
      } else {
        go(HcPrepClusterKernel<false, false>);
      }
      return;
    }
    const auto threads = static_cast<unsigned>(((width / 4) + 31) / 32 * 32);
    const auto grid = static_cast<unsigned>(l.t);
    if (l.combine && l.inject) {
      HcPrepKernel<true, true><<<grid, threads, 0, s>>>(x, norm, inject, out, logits_in, streams,
                                                        normed, logits, width, hc, eps, inv_hc);
    } else if (l.combine) {
      HcPrepKernel<true, false><<<grid, threads, 0, s>>>(x, norm, inject, out, logits_in, streams,
                                                         normed, logits, width, hc, eps, inv_hc);
    } else if (l.inject) {
      HcPrepKernel<false, true><<<grid, threads, 0, s>>>(x, norm, inject, out, logits_in, streams,
                                                         normed, logits, width, hc, eps, inv_hc);
    } else {
      HcPrepKernel<false, false><<<grid, threads, 0, s>>>(x, norm, inject, out, logits_in, streams,
                                                          normed, logits, width, hc, eps, inv_hc);
    }
  });
}

std::expected<void, KernelFailure> RunHcLo(LaunchContext& launch, ggml_tensor* node) {
  if (auto checked = CheckHcLo(node); !checked) {
    return checked;
  }
  return launch.Run(base::Bytes(0), [node](ggml_backend_cuda_context& context) {
    const std::int64_t n = ggml_nelements(node);
    HcLoKernel<<<Blocks(n, kThreads), kThreads, 0, context.stream()>>>(
        static_cast<const float*>(node->src[0]->data), static_cast<nv_bfloat16*>(node->data), n,
        1.0f / static_cast<float>(LlmpOpInt(node, 0)));
  });
}

std::expected<void, KernelFailure> RunHcMixBf16(LaunchContext& launch, ggml_tensor* node) {
  if (auto checked = CheckHcMixBf16(node); !checked) {
    return checked;
  }
  return launch.Run(base::Bytes(0), [node](ggml_backend_cuda_context& context) {
    const int hc = LlmpOpInt(node, 0);
    const HcMixLayout layout{
        .width = LlmpOpInt(node, 1), .t = LlmpOpInt(node, 2), .bf16 = LlmpOpInt(node, 4) == 1};
    const bool quantize = LlmpOpInt(node, 3) == 1;
    const std::int64_t items = layout.width * layout.t / 8;
    auto* blob = static_cast<std::uint8_t*>(node->data);
    const mxfp8::RowsLayout rows{.k = static_cast<std::uint64_t>(layout.width),
                                 .rows = static_cast<std::uint64_t>(layout.t)};
    std::uint8_t* codes = blob + layout.quantized() + rows.codes();
    std::uint8_t* scales = blob + layout.quantized() + rows.scales();
    auto* rounded = reinterpret_cast<uint4*>(blob + layout.rounded());
    const auto* normed = static_cast<const uint4*>(node->src[0]->data);
    const auto* gate = static_cast<const uint4*>(node->src[1]->data);
    auto* dst = reinterpret_cast<float4*>(blob + HcMixLayout::mixed());
    const auto width8 = static_cast<int>(layout.width / 8);
    const float inv_hc = 1.0f / static_cast<float>(hc);
    cudaStream_t s = context.stream();
    if (quantize) {
      // The padding rows' scales are zero, never stale bytes.
      (void)cudaMemsetAsync(scales, 0, rows.bytes() - rows.scales(), s);
    }
    if (quantize && layout.bf16) {
      HcMixBf16Kernel<true, true><<<Blocks(items, kThreads), kThreads, 0, s>>>(
          normed, gate, dst, codes, scales, rounded, width8, hc, items, inv_hc);
    } else if (quantize) {
      HcMixBf16Kernel<true, false><<<Blocks(items, kThreads), kThreads, 0, s>>>(
          normed, gate, dst, codes, scales, rounded, width8, hc, items, inv_hc);
    } else {
      HcMixBf16Kernel<false, false><<<Blocks(items, kThreads), kThreads, 0, s>>>(
          normed, gate, dst, codes, scales, rounded, width8, hc, items, inv_hc);
    }
  });
}

std::expected<void, KernelFailure> RunMoeRouter(LaunchContext& launch, ggml_tensor* node) {
  if (auto checked = CheckMoeRouter(node); !checked) {
    return checked;
  }
  return launch.Run(base::Bytes(0), [node](ggml_backend_cuda_context& context) {
    const int experts = LlmpOpInt(node, 0);
    const int used = LlmpOpInt(node, 1);
    const int tokens = LlmpOpInt(node, 2);
    const MoeRouterLayout layout{.used = used, .t = tokens};
    auto* blob = static_cast<std::uint8_t*>(node->data);
    const auto launch_with = [&](const auto* gate_row) {
      MoeRouterKernel<<<static_cast<unsigned>((tokens + kRouterWarps - 1) / kRouterWarps),
                        kRouterWarps * 32, 0, context.stream()>>>(
          static_cast<const float*>(node->src[0]->data),
          static_cast<const float4*>(node->src[1]->data), gate_row,
          reinterpret_cast<std::int32_t*>(blob + MoeRouterLayout::ids()),
          reinterpret_cast<float*>(blob + layout.weights()),
          reinterpret_cast<float*>(blob + layout.gate()), tokens, experts / 32, used,
          LlmpOpInt(node, 3) / 4);
    };
    if (node->src[2]->type == GGML_TYPE_BF16) {
      launch_with(static_cast<const uint2*>(node->src[2]->data));
    } else {
      launch_with(static_cast<const float4*>(node->src[2]->data));
    }
  });
}

std::expected<void, KernelFailure> RunGemmBf16(LaunchContext& launch, ggml_tensor* node) {
  if (auto checked = CheckGemmBf16(node); !checked) {
    return checked;
  }
  if (IsGemvBf16(node) && node->src[1]->ne[1] <= kGemvBf16FastColumns &&
      node->src[0]->ne[0] % 8 == 0 && node->src[0]->ne[0] <= kGemvBf16MaxK) {
    return launch.Run(base::Bytes(0), [node](ggml_backend_cuda_context& context) {
      LaunchGemvBf16(node, context.stream());
    });
  }
  if (launch.cublas() == nullptr) {
    return Refused("the launch context lends no cuBLAS handle");
  }
  // cuBLAS writes its workspace while it reads the operands.
  if (auto clear = CheckClearOf(node, launch.cublas()->workspace().base,
                                launch.cublas()->workspace().size.value());
      !clear) {
    return clear;
  }
  return launch.Run(base::Bytes(0), [node](ggml_backend_cuda_context& context) {
    const ggml_tensor* weights = node->src[0];
    const ggml_tensor* x = node->src[1];
    const int k = static_cast<int>(weights->ne[0]);
    const int n = static_cast<int>(weights->ne[1]);
    const int t = static_cast<int>(x->ne[1]);
    const float alpha = 1.0f;
    const float beta = 0.0f;
    // ggml_cuda_mul_mat_cublas_impl's call for BF16 weights with F32 output
    // (its lda and ldb the packed rows, ldc the output's), or BF16 output.
    const cudaDataType_t out = node->type == GGML_TYPE_BF16 ? CUDA_R_16BF : CUDA_R_32F;
    CUBLAS_CHECK(cublasGemmEx(context.cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N, n, t, k, &alpha,
                              weights->data, CUDA_R_16BF, k, x->data, CUDA_R_16BF, k, &beta,
                              node->data, out, n, CUBLAS_COMPUTE_32F,
                              CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  });
}

std::expected<void, KernelFailure> Qwen38Commit(LaunchContext& launch,
                                                const Qwen38CommitArgs& args) {
  constexpr int kS = 128;
  const auto aligned = [](const void* p) {
    return p != nullptr && reinterpret_cast<std::uintptr_t>(p) % 16 == 0;
  };
  bool ok = args.layers >= 0 && args.layers <= kQwen38CommitLayers && args.keep >= 1 &&
            args.keep <= 8 && args.taps >= 1 && args.taps <= 16 && args.qk_heads > 0 &&
            args.v_heads > 0 && args.v_heads % args.qk_heads == 0 &&
            args.channels == (2 * args.qk_heads * kS) + (args.v_heads * kS);
  for (int l = 0; ok && l < args.layers; ++l) {
    const Qwen38CommitLayer& layer = args.layer[l];
    ok = aligned(layer.state) && aligned(layer.history) && aligned(layer.conv) &&
         aligned(layer.qkv) && aligned(layer.gate) && aligned(layer.beta);
  }
  const bool ple = args.ple_history != nullptr;
  if (ple) {
    ok = ok && aligned(args.ple_history) && aligned(args.ple_rows) && args.ple_width > 0 &&
         args.ple_taps >= 1 && args.ple_taps <= 16;
  }
  if (!ok) {
    return Refused("a commit's extents are not its kernels' (128-wide heads, 1 to 8 rows)");
  }
  return launch.Run(base::Bytes(0), [&args, ple](ggml_backend_cuda_context& context) {
    if (args.layers > 0) {
      const dim3 block(32, 4);
      const dim3 grid(static_cast<unsigned>(args.v_heads), kS / (4 * kGdnColumns),
                      static_cast<unsigned>(args.layers));
      GdnCommitKernel<<<grid, block, 0, context.stream()>>>(args);
    }
    const int widest = ple ? std::max(args.channels, args.ple_width) : args.channels;
    const int histories = args.layers + (ple ? 1 : 0);
    if (histories > 0) {
      HistoryCommitKernel<<<dim3(Blocks(widest, kThreads), static_cast<unsigned>(histories)),
                            kThreads, 0, context.stream()>>>(args);
    }
  });
}

}  // namespace llmp::kernels::ggml
