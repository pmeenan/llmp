// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: MIT AND Apache-2.0

// The routed experts over the CUTLASS layout (jitllm_ops.h, moe_layout.h):
// routing, NVFP4 activation quantization, the gate/up SwiGLU and its
// quantization, the sorted rows' weighted sum, decode's vector products,
// and the layout conversions. The activation quantization repeats GGML's
// quantize_mmq_nvfp4 (ggml-cuda/quantize.cu at llama.cpp b29c606e2, MIT): a
// row scale of amax / (6 · 448), each 16-value block's E4M3 scale searched
// over five codes around amax / 6 for the least squared error
// (nvfp4_native_scale_error, copied below), E2M1 codes by CUDA's
// conversion; so the codes and scales are those GGML's MMQ multiplies, only
// laid out for CUTLASS. The file builds with GGML's device flags
// (-use_fast_math), as its functions do.

#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <string>
#include <utility>

#include "base/bytes.h"
#include "common.cuh"
#include "kernels/ggml/jitllm_ops.h"
#include "kernels/ggml/launch.h"
#include "kernels/ggml/moe_cutlass.h"
#include "kernels/ggml/moe_layout.h"
#include "vecdotq.cuh"

namespace jitllm::kernels::ggml {
namespace {

std::unexpected<KernelFailure> Refused(std::string detail) {
  return std::unexpected(
      KernelFailure{.error = KernelError::kRejected, .detail = std::move(detail)});
}

constexpr int kSub = 16;  // values an E4M3 block scale covers

struct Route {
  const std::int32_t* offsets;
  const std::int32_t* scale_rows;
  const std::int32_t* position;
  const std::int32_t* token;
  const std::int32_t* expert;
  std::int32_t* counts;
  int experts;
  int used;
  int tokens;
  int chunks;
};

Route RouteOf(const ggml_tensor* route) {
  const moe::RouteLayout layout{
      .experts = JitllmOpInt(route, 0),
      .slots = static_cast<std::int64_t>(JitllmOpInt(route, 1)) * JitllmOpInt(route, 2),
      .chunks = (JitllmOpInt(route, 2) + moe::kRouteChunk - 1) / moe::kRouteChunk};
  auto* base = static_cast<std::int32_t*>(route->data);
  return {.offsets = base + layout.offsets(),
          .scale_rows = base + layout.scale_rows(),
          .position = base + layout.position(),
          .token = base + layout.token(),
          .expert = base + layout.expert(),
          .counts = base + layout.counts(),
          .experts = static_cast<int>(layout.experts),
          .used = JitllmOpInt(route, 1),
          .tokens = JitllmOpInt(route, 2),
          .chunks = static_cast<int>(layout.chunks)};
}

// ------------------------------------------------------------------ routing

// Each block counts its chunk's tokens' ids per expert.
__global__ void RouteCount(const std::int32_t* __restrict__ ids, int ids_stride, Route r) {
  extern __shared__ int hist[];
  for (int e = static_cast<int>(threadIdx.x); e < r.experts; e += blockDim.x) {
    hist[e] = 0;
  }
  __syncthreads();
  const int c = static_cast<int>(blockIdx.x);
  const int t0 = c * static_cast<int>(moe::kRouteChunk);
  const int t1 = min(t0 + static_cast<int>(moe::kRouteChunk), r.tokens);
  const int n = (t1 - t0) * r.used;
  for (int i = static_cast<int>(threadIdx.x); i < n; i += blockDim.x) {
    const int t = t0 + (i / r.used);
    const int e = ids[(static_cast<std::int64_t>(t) * ids_stride) + (i % r.used)];
    if (e >= 0 && e < r.experts) {
      atomicAdd(&hist[e], 1);
    }
  }
  __syncthreads();
  for (int e = static_cast<int>(threadIdx.x); e < r.experts; e += blockDim.x) {
    r.counts[(static_cast<std::int64_t>(c) * r.experts) + e] = hist[e];
  }
}

// One block: each expert's chunk counts become the chunks' bases, then the
// experts' offsets and scale rows.
__global__ void RouteScan(Route r) {
  extern __shared__ int total[];
  for (int e = static_cast<int>(threadIdx.x); e < r.experts; e += blockDim.x) {
    int running = 0;
    for (int c = 0; c < r.chunks; ++c) {
      std::int32_t* at = r.counts + (static_cast<std::int64_t>(c) * r.experts) + e;
      const int v = *at;
      *at = running;
      running += v;
    }
    total[e] = running;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    auto* offsets = const_cast<std::int32_t*>(r.offsets);
    auto* scale_rows = const_cast<std::int32_t*>(r.scale_rows);
    offsets[0] = 0;
    scale_rows[0] = 0;
    const int rows = static_cast<int>(moe::kScaleRows);
    for (int e = 0; e < r.experts; ++e) {
      offsets[e + 1] = offsets[e] + total[e];
      scale_rows[e + 1] = scale_rows[e] + ((total[e] + rows - 1) / rows * rows);
    }
  }
}

// Each block's first thread places its chunk's slots in token order.
__global__ void RouteAssign(const std::int32_t* __restrict__ ids, int ids_stride, Route r) {
  extern __shared__ int placed[];
  for (int e = static_cast<int>(threadIdx.x); e < r.experts; e += blockDim.x) {
    placed[e] = 0;
  }
  __syncthreads();
  if (threadIdx.x != 0) {
    return;
  }
  const int c = static_cast<int>(blockIdx.x);
  const int t0 = c * static_cast<int>(moe::kRouteChunk);
  const int t1 = min(t0 + static_cast<int>(moe::kRouteChunk), r.tokens);
  auto* position = const_cast<std::int32_t*>(r.position);
  auto* token = const_cast<std::int32_t*>(r.token);
  auto* expert = const_cast<std::int32_t*>(r.expert);
  for (int t = t0; t < t1; ++t) {
    for (int j = 0; j < r.used; ++j) {
      const int slot = (t * r.used) + j;
      const int e = ids[(static_cast<std::int64_t>(t) * ids_stride) + j];
      if (e < 0 || e >= r.experts) {
        position[slot] = -1;
        continue;
      }
      const int row =
          r.offsets[e] + r.counts[(static_cast<std::int64_t>(c) * r.experts) + e] + placed[e]++;
      position[slot] = row;
      token[row] = t;
      expert[row] = e;
    }
  }
}

// ------------------------------------------------------------------ quantization

#if defined(BLACKWELL_MMA_AVAILABLE)
// quantize.cu's nvfp4_native_scale_error (GGML, MIT), unchanged: the
// squared error of a block quantized at `scale`.
__device__ __forceinline__ float Nvfp4ScaleError(const float vals[kSub], const float inv_col_scale,
                                                 const float inv_scale, const float scale) {
  const float scale_dequant = 2.0f * scale;
  float err = 0.0f;
#pragma unroll
  for (int k = 0; k < kSub; k += 4) {
    const float v0 = vals[k + 0] * inv_col_scale;
    const float v1 = vals[k + 1] * inv_col_scale;
    const float v2 = vals[k + 2] * inv_col_scale;
    const float v3 = vals[k + 3] * inv_col_scale;
    const __nv_fp4x4_e2m1 q(
        make_float4(v0 * inv_scale, v1 * inv_scale, v2 * inv_scale, v3 * inv_scale));
    const __nv_fp4x4_storage_t q_storage = q.__x;
    const auto q_lo = static_cast<__nv_fp4x2_storage_t>(q_storage);
    const auto q_hi = static_cast<__nv_fp4x2_storage_t>(q_storage >> 8U);
    const __half2_raw hraw2_lo = __nv_cvt_fp4x2_to_halfraw2(q_lo, __NV_E2M1);
    const __half2_raw hraw2_hi = __nv_cvt_fp4x2_to_halfraw2(q_hi, __NV_E2M1);
    const float2 dq_lo = __half22float2(static_cast<__half2>(hraw2_lo));
    const float2 dq_hi = __half22float2(static_cast<__half2>(hraw2_hi));
    const float err0 = fabsf(v0) - fabsf(dq_lo.x) * scale_dequant;
    const float err1 = fabsf(v1) - fabsf(dq_lo.y) * scale_dequant;
    const float err2 = fabsf(v2) - fabsf(dq_hi.x) * scale_dequant;
    const float err3 = fabsf(v3) - fabsf(dq_hi.y) * scale_dequant;
    err = fmaf(err0, err0, err);
    err = fmaf(err1, err1, err);
    err = fmaf(err2, err2, err);
    err = fmaf(err3, err3, err);
  }
  return err;
}

// quantize_mmq_nvfp4's block step: the E4M3 scale code and the 16 E2M1
// codes (element 2i in the low nibble of byte i) of `vals` under the row's
// inverse scale.
__device__ __forceinline__ std::uint8_t QuantizeBlock(const float vals[kSub], float inv_col_scale,
                                                      uint2& codes) {
  float amax_sub = 0.0f;
#pragma unroll
  for (int k = 0; k < kSub; ++k) {
    amax_sub = fmaxf(amax_sub, fabsf(vals[k] * inv_col_scale));
  }
  constexpr int kOffsets[5] = {0, -1, 1, -2, 2};
  const int first = static_cast<int>(ggml_cuda_fp32_to_ue4m3(amax_sub / 6.0f));
  auto code = static_cast<std::uint8_t>(first);
  float scale = ggml_cuda_ue4m3_to_fp32(code);
  float best = Nvfp4ScaleError(vals, inv_col_scale, scale > 0.0f ? 0.5f / scale : 0.0f, scale);
#pragma unroll
  for (int i = 1; i < 5; ++i) {
    const int test = first + kOffsets[i];
    if (test < 0 || test > 0x7e) {
      continue;
    }
    const float test_scale = ggml_cuda_ue4m3_to_fp32(static_cast<std::uint8_t>(test));
    const float err = Nvfp4ScaleError(vals, inv_col_scale,
                                      test_scale > 0.0f ? 0.5f / test_scale : 0.0f, test_scale);
    if (err < best) {
      best = err;
      code = static_cast<std::uint8_t>(test);
      scale = test_scale;
    }
  }
  const float inv_scale = scale > 0.0f ? 0.5f / scale : 0.0f;
  const float s = inv_col_scale * inv_scale;
  std::uint32_t words[2];
#pragma unroll
  for (int w = 0; w < 2; ++w) {
    const __nv_fp4x4_e2m1 lo(make_float4(vals[(8 * w) + 0] * s, vals[(8 * w) + 1] * s,
                                         vals[(8 * w) + 2] * s, vals[(8 * w) + 3] * s));
    const __nv_fp4x4_e2m1 hi(make_float4(vals[(8 * w) + 4] * s, vals[(8 * w) + 5] * s,
                                         vals[(8 * w) + 6] * s, vals[(8 * w) + 7] * s));
    words[w] = static_cast<std::uint32_t>(lo.__x) | (static_cast<std::uint32_t>(hi.__x) << 16U);
  }
  codes = make_uint2(words[0], words[1]);
  return code;
}

// A row's amax over its k values in `row` (shared or global), reduced over
// the block as quantize_mmq_nvfp4 reduces it (a max is order-free).
__device__ float BlockAmax(const float* row, int k, float* warp_amax) {
  float amax = 0.0f;
  for (int i = static_cast<int>(threadIdx.x); i < k; i += blockDim.x) {
    amax = fmaxf(amax, fabsf(row[i]));
  }
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, offset, 32));
  }
  const int lane = static_cast<int>(threadIdx.x) % 32;
  const int warp = static_cast<int>(threadIdx.x) / 32;
  if (lane == 0) {
    warp_amax[warp] = amax;
  }
  __syncthreads();
  if (warp == 0) {
    amax = lane < static_cast<int>(blockDim.x / 32) ? warp_amax[lane] : 0.0f;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, offset, 32));
    }
    if (lane == 0) {
      warp_amax[0] = amax / (6.0f * 448.0f);
    }
  }
  __syncthreads();
  return warp_amax[0];
}
#endif  // defined(BLACKWELL_MMA_AVAILABLE)

struct Quantized {
  std::uint8_t* codes;
  std::uint8_t* scales;
  float* row_scales;
  int k;
};

Quantized QuantizedOf(void* base, const moe::QuantLayout& q) {
  auto* b = static_cast<std::uint8_t*>(base);
  return {.codes = b + q.codes(),
          .scales = b + q.scales(),
          .row_scales = reinterpret_cast<float*>(b + q.row_scales()),
          .k = static_cast<int>(q.k)};
}

#if defined(BLACKWELL_MMA_AVAILABLE)
// Writes one quantized row's block (codes and scale) into sorted row `row`
// of expert e.
__device__ __forceinline__ void PutBlock(const Quantized& out, const Route& r, int row, int e,
                                         int block, uint2 codes, std::uint8_t scale) {
  *reinterpret_cast<uint2*>(out.codes + (static_cast<std::int64_t>(row) * (out.k / 2)) +
                            (block * 8)) = codes;
  const auto scale_row = static_cast<std::uint64_t>(r.scale_rows[e] + (row - r.offsets[e]));
  out.scales[moe::SfOffset(scale_row, static_cast<std::uint64_t>(block),
                           static_cast<std::uint64_t>(out.k / kSub))] = scale;
}
#endif  // defined(BLACKWELL_MMA_AVAILABLE)

// One token a block: its row scale, then each block quantized once and
// written to the token's sorted rows.
__global__ void QuantizeRows(const float* __restrict__ x, Route r, Quantized out) {
#if defined(BLACKWELL_MMA_AVAILABLE)
  __shared__ float warp_amax[32];
  const int t = static_cast<int>(blockIdx.x);
  const float* row = x + (static_cast<std::int64_t>(t) * out.k);
  const float row_scale = BlockAmax(row, out.k, warp_amax);
  const float inv = row_scale > 0.0f ? 1.0f / row_scale : 0.0f;
  if (threadIdx.x < static_cast<unsigned>(r.used)) {
    const int p = r.position[(t * r.used) + static_cast<int>(threadIdx.x)];
    if (p >= 0) {
      out.row_scales[p] = row_scale;
    }
  }
  for (int b = static_cast<int>(threadIdx.x); b < out.k / kSub; b += blockDim.x) {
    float vals[kSub];
#pragma unroll
    for (int i = 0; i < kSub; i += 4) {
      const float4 v = *reinterpret_cast<const float4*>(row + (b * kSub) + i);
      vals[i] = v.x;
      vals[i + 1] = v.y;
      vals[i + 2] = v.z;
      vals[i + 3] = v.w;
    }
    uint2 codes;
    const std::uint8_t scale = QuantizeBlock(vals, inv, codes);
    for (int j = 0; j < r.used; ++j) {
      const int p = r.position[(t * r.used) + j];
      if (p >= 0) {
        PutBlock(out, r, p, r.expert[p], b, codes, scale);
      }
    }
  }
#endif
}

__device__ __forceinline__ float Silu(float x) { return x / (1.0f + expf(-x)); }
__device__ __forceinline__ float Sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

// One sorted row a block: silu(gate · g · s_gate[e]) · (up · g · s_up[e]),
// then quantized as its row.
__global__ void GluQuantizeRows(const __nv_bfloat16* __restrict__ d, const float* __restrict__ g_in,
                                const float* __restrict__ gate_scale,
                                const float* __restrict__ up_scale, Route r, Quantized out) {
#if defined(BLACKWELL_MMA_AVAILABLE)
  extern __shared__ float act[];
  __shared__ float warp_amax[32];
  const int row = static_cast<int>(blockIdx.x);
  if (row >= r.offsets[r.experts]) {
    return;
  }
  const int f = out.k;
  const int e = r.expert[row];
  const float g = g_in[row];
  const float gs = gate_scale[e];
  const float us = up_scale[e];
  const __nv_bfloat16* dr = d + (static_cast<std::int64_t>(row) * 2 * f);
  for (int i = static_cast<int>(threadIdx.x); i < f; i += blockDim.x) {
    const float gate = (__bfloat162float(dr[i]) * g) * gs;
    const float up = (__bfloat162float(dr[f + i]) * g) * us;
    act[i] = Silu(gate) * up;
  }
  __syncthreads();
  const float row_scale = BlockAmax(act, f, warp_amax);
  const float inv = row_scale > 0.0f ? 1.0f / row_scale : 0.0f;
  if (threadIdx.x == 0) {
    out.row_scales[row] = row_scale;
  }
  for (int b = static_cast<int>(threadIdx.x); b < f / kSub; b += blockDim.x) {
    float vals[kSub];
#pragma unroll
    for (int i = 0; i < kSub; ++i) {
      vals[i] = act[(b * kSub) + i];
    }
    uint2 codes;
    const std::uint8_t scale = QuantizeBlock(vals, inv, codes);
    PutBlock(out, r, row, e, b, codes, scale);
  }
#endif
}

// ------------------------------------------------------------------ combination

// Σ_j ((d_j · g_j) · s_down[e_j]) · w_j in expert order, plus shared ·
// sigmoid(shared gate): jitllm.moe.combine's arithmetic over the sorted
// rows, four columns a thread.
__global__ void CombineSorted(const __nv_bfloat16* __restrict__ d, const float* __restrict__ g,
                              const float* __restrict__ down_scale,
                              const float* __restrict__ weights, const float4* __restrict__ shared,
                              const float* __restrict__ shared_gate, float4* __restrict__ dst,
                              Route r, int width4, std::int64_t total4) {
  const std::int64_t i = (static_cast<std::int64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  if (i >= total4) {
    return;
  }
  const std::int64_t t = i / width4;
  const std::int64_t c4 = i % width4;
  float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  for (int j = 0; j < r.used; ++j) {
    const int p = r.position[(t * r.used) + j];
    const float nan = __uint_as_float(0x7fc00000u);
    const float ds = p >= 0 ? down_scale[r.expert[p]] : nan;
    const float gj = p >= 0 ? g[p] : nan;
    const float w = weights[(t * r.used) + j];
    float4 v = make_float4(nan, nan, nan, nan);
    if (p >= 0) {
      const uint2 raw = *reinterpret_cast<const uint2*>(
          d + (static_cast<std::int64_t>(p) * width4 * 4) + (c4 * 4));
      const __nv_bfloat162 lo = *reinterpret_cast<const __nv_bfloat162*>(&raw.x);
      const __nv_bfloat162 hi = *reinterpret_cast<const __nv_bfloat162*>(&raw.y);
      v = make_float4(__low2float(lo), __high2float(lo), __low2float(hi), __high2float(hi));
    }
    const float4 e = make_float4(__fmul_rn((v.x * gj) * ds, w), __fmul_rn((v.y * gj) * ds, w),
                                 __fmul_rn((v.z * gj) * ds, w), __fmul_rn((v.w * gj) * ds, w));
    acc = j == 0 ? e : make_float4(acc.x + e.x, acc.y + e.y, acc.z + e.z, acc.w + e.w);
  }
  const float sg = Sigmoid(shared_gate[t]);
  const float4 s = shared[i];
  dst[i] = make_float4(acc.x + __fmul_rn(s.x, sg), acc.y + __fmul_rn(s.y, sg),
                       acc.z + __fmul_rn(s.z, sg), acc.w + __fmul_rn(s.w, sg));
}

// ------------------------------------------------------------------ decode

struct GemvArgs {
  const std::uint8_t* weights;
  std::uint64_t stride;
  int experts;
  std::uint64_t codes;
  std::uint64_t scales;
  int row0;
  int n;
  int k;
  const float* x;
  std::int64_t x_slot;   // floats between slots' inputs (0: one input a token)
  std::int64_t x_token;  // floats between tokens' inputs
  const std::int32_t* ids;
  int ids_stride;
  int used;
  const float* gate_scale;  // the SwiGLU form's per-expert scales
  const float* up_scale;
  float* y;
};

constexpr int kGemvRows = 8;  // weight rows a warp computes, reusing its x

// The weight rows' outputs a warp writes: all of them, or in the SwiGLU
// form half (each gate row with its up row, n rows further).
template <bool kGlu>
constexpr int kGemvOutputs = kGlu ? kGemvRows / 2 : kGemvRows;

// One warp kGemvRows weight rows of one slot, as GGML's MMVQ computes an
// NVFP4 product (vec_dot_nvfp4_q8_1): each lane its 16-value blocks of x,
// quantized to 8 bits once for the rows (a scale a block), the codes
// through GGML's value table, 8-bit dot products, times the block's two
// scales; then the warp sums. The SwiGLU form takes gate rows n0.. and up
// rows n + n0.. and writes silu(gate · s_gate[e]) · (up · s_up[e]), as
// jitllm.moe.glu does.
// Launched as a programmatic dependent (PDL): the expert ids are the
// routing's output, so it waits before anything, but its blocks are placed
// while the kernel before drains; it lets the next launch once its weights
// are read.
template <bool kGlu>
__global__ void __launch_bounds__(256) Gemv(GemvArgs a) {
  constexpr int kOutputs = kGemvOutputs<kGlu>;
  const int lane = static_cast<int>(threadIdx.x) % 32;
  const int n0 =
      ((static_cast<int>(blockIdx.x) * 8) + (static_cast<int>(threadIdx.x) / 32)) * kOutputs;
  const int slot = static_cast<int>(blockIdx.y);  // t · used + j
  if (n0 >= a.n) {
    return;
  }
  ggml_cuda_pdl_sync();
  const int t = slot / a.used;
  const int j = slot % a.used;
  const int e = a.ids[(static_cast<std::int64_t>(t) * a.ids_stride) + j];
  float* out = a.y + (static_cast<std::int64_t>(slot) * a.n) + n0;
  const int rows = min(kOutputs, a.n - n0);
  if (e < 0 || e >= a.experts) {
    if (lane < rows) {
      out[lane] = __uint_as_float(0x7fc00000u);
    }
    return;
  }
  const std::uint8_t* w = a.weights + (static_cast<std::uint64_t>(e) * a.stride);
  const auto row0 = static_cast<std::uint64_t>(a.row0 + n0);
  const std::uint8_t* codes = w + a.codes + (row0 * static_cast<std::uint64_t>(a.k / 2));
  const std::uint8_t* scales = w + a.scales;
  const float* x = a.x + (t * a.x_token) + (j * a.x_slot);
  const int blocks = a.k / kSub;
  float sum[kGemvRows] = {};
  for (int b = lane; b < blocks; b += 32) {
    // Every row's codes and scale first, then x: all the loads in flight
    // before the arithmetic.
    uint2 qs[kGemvRows];
    std::uint8_t ss[kGemvRows];
#pragma unroll
    for (int r = 0; r < kGemvRows; ++r) {
      // A short last group rereads its last row; the up rows are n further.
      const int rr = min(r % kOutputs, rows - 1) + (kGlu && r >= kOutputs ? a.n : 0);
      qs[r] = *reinterpret_cast<const uint2*>(codes + (static_cast<std::int64_t>(rr) * (a.k / 2)) +
                                              (b * 8));
      ss[r] =
          scales[moe::SfOffset(row0 + static_cast<std::uint64_t>(rr), static_cast<std::uint64_t>(b),
                               static_cast<std::uint64_t>(blocks))];
    }
    // The block of x quantized to 8 bits (scale amax / 127, rounded as
    // quantize_q8_1 rounds), its even and odd elements packed apart, as the
    // code table returns them.
    const float4* xv = reinterpret_cast<const float4*>(x + (b * kSub));
    const float4 xs[4] = {xv[0], xv[1], xv[2], xv[3]};
    const float vals[kSub] = {xs[0].x, xs[0].y, xs[0].z, xs[0].w, xs[1].x, xs[1].y,
                              xs[1].z, xs[1].w, xs[2].x, xs[2].y, xs[2].z, xs[2].w,
                              xs[3].x, xs[3].y, xs[3].z, xs[3].w};
    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < kSub; ++i) {
      amax = fmaxf(amax, fabsf(vals[i]));
    }
    const float d = amax / 127.0f;
    int xq[4] = {0, 0, 0, 0};  // even 0-7, odd 0-7, even 8-15, odd 8-15
    if (amax > 0.0f) {
#pragma unroll
      for (int i = 0; i < kSub; ++i) {
        const auto q8 = static_cast<std::uint32_t>(
            static_cast<std::uint8_t>(static_cast<std::int8_t>(roundf(vals[i] / d))));
        const int word = ((i / 8) * 2) + (i % 2);
        xq[word] |= static_cast<int>(q8 << (8U * static_cast<std::uint32_t>((i % 8) / 2)));
      }
    }
#pragma unroll
    for (int r = 0; r < kGemvRows; ++r) {
      // GGML's NVFP4 table holds twice E2M1's values, its E4M3 decoding
      // half the scale's: their product is the weight's.
      const int2 lo = get_int_from_table_16(static_cast<int>(qs[r].x), kvalues_mxfp4);
      const int2 hi = get_int_from_table_16(static_cast<int>(qs[r].y), kvalues_mxfp4);
      int dot = ggml_cuda_dp4a(lo.x, xq[0], 0);
      dot = ggml_cuda_dp4a(lo.y, xq[1], dot);
      dot = ggml_cuda_dp4a(hi.x, xq[2], dot);
      dot = ggml_cuda_dp4a(hi.y, xq[3], dot);
      sum[r] = fmaf(static_cast<float>(dot), ggml_cuda_ue4m3_to_fp32(ss[r]) * d, sum[r]);
    }
  }
  ggml_cuda_pdl_lc();
#pragma unroll
  for (int r = 0; r < kGemvRows; ++r) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      sum[r] += __shfl_xor_sync(0xffffffffu, sum[r], offset, 32);
    }
  }
  if (lane != 0) {
    return;
  }
#pragma unroll
  for (int r = 0; r < kOutputs; ++r) {
    if (r < rows) {
      if constexpr (kGlu) {
        out[r] = Silu(sum[r] * a.gate_scale[e]) * (sum[r + kOutputs] * a.up_scale[e]);
      } else {
        out[r] = sum[r];
      }
    }
  }
}

// The expert-major form of Gemv, for several tokens: each distinct expert's
// weight rows are read once for every slot that chose it. A block (n-tile,
// slot s) leads when s is a 0th, kMatched-th, ... slot choosing its expert,
// in slot order; it computes those slots (up to kMatched) and the others
// return. Each slot's output is Gemv's to the bit: the same lanes' blocks,
// the same 8-bit quantization of x, dot products, scales and warp sums.
constexpr int kMatched = 8;

// Whether `tokens` take the expert-major form: more than one request's
// verify rows (a wave's shared product). Within one request's rows L2
// already serves the repeated experts, and the per-slot form is as fast
// (C2 screen on GB10: +7.0% shared-wave throughput, −0.6% solo, outputs
// identical).
constexpr bool MoeGemvExpertMajor(std::int64_t tokens) { return tokens > 4; }

template <bool kGlu, int kTiles>
__global__ void __launch_bounds__(256) GemvExperts(GemvArgs a) {
  constexpr int kOutputs = kGemvOutputs<kGlu>;
  const int lane = static_cast<int>(threadIdx.x) % 32;
  const int warp = static_cast<int>(threadIdx.x) / 32;
  const int tile0 = static_cast<int>(blockIdx.x) * kTiles;
  const int slot = static_cast<int>(blockIdx.y);  // t · used + j
  const int slots = static_cast<int>(gridDim.y);
  __shared__ int matched[kMatched];
  __shared__ int count;
  __shared__ int lead;
  ggml_cuda_pdl_sync();
  const auto expert_of = [&](int s) {
    return a.ids[(static_cast<std::int64_t>(s / a.used) * a.ids_stride) + (s % a.used)];
  };
  const int e = expert_of(slot);
  if (warp == 0) {
    int before = 0;
    int listed = 0;
    for (int base = 0; base < slots; base += 32) {
      const int i = base + lane;
      const bool same = i < slots && expert_of(i) == e;
      const unsigned ballot = __ballot_sync(0xffffffffu, same);
      // The lanes of slots before `slot`.
      const int k = slot - base;
      const unsigned earlier = k <= 0 ? 0u : (k >= 32 ? 0xffffffffu : ((1u << k) - 1u));
      before += __popc(ballot & earlier);
      const unsigned after = ballot & ~earlier;
      if (same && i >= slot) {
        const int at = listed + __popc(after & ((1u << lane) - 1u));
        if (at < kMatched) {
          matched[at] = i;
        }
      }
      listed += __popc(after);
    }
    if (lane == 0) {
      lead = (e < 0 || e >= a.experts) ? 1 : (before % kMatched == 0);
      count = (e < 0 || e >= a.experts) ? 1 : min(listed, kMatched);
      if (e < 0 || e >= a.experts) {
        matched[0] = slot;
      }
    }
  }
  __syncthreads();
  if (!lead) {
    return;
  }
  const int m_count = count;
  // kTiles consecutive output tiles an expert: one leader scan for all.
  for (int tile = tile0; tile < tile0 + kTiles; ++tile) {
    const int n0 = ((tile * 8) + warp) * kOutputs;
    if (n0 >= a.n) {
      break;
    }
    const int rows = min(kOutputs, a.n - n0);
    if (e < 0 || e >= a.experts) {
      if (lane < rows) {
        a.y[(static_cast<std::int64_t>(slot) * a.n) + n0 + lane] = __uint_as_float(0x7fc00000u);
      }
      continue;
    }
    const std::uint8_t* w = a.weights + (static_cast<std::uint64_t>(e) * a.stride);
    const auto row0 = static_cast<std::uint64_t>(a.row0 + n0);
    const std::uint8_t* codes = w + a.codes + (row0 * static_cast<std::uint64_t>(a.k / 2));
    const std::uint8_t* scales = w + a.scales;
    const int blocks = a.k / kSub;
    float sum[kMatched][kGemvRows];
#pragma unroll
    for (int m = 0; m < kMatched; ++m) {
#pragma unroll
      for (int r = 0; r < kGemvRows; ++r) {
        sum[m][r] = 0.0f;
      }
    }
    for (int b = lane; b < blocks; b += 32) {
      uint2 qs[kGemvRows];
      std::uint8_t ss[kGemvRows];
#pragma unroll
      for (int r = 0; r < kGemvRows; ++r) {
        const int rr = min(r % kOutputs, rows - 1) + (kGlu && r >= kOutputs ? a.n : 0);
        qs[r] = *reinterpret_cast<const uint2*>(
            codes + (static_cast<std::int64_t>(rr) * (a.k / 2)) + (b * 8));
        ss[r] = scales[moe::SfOffset(row0 + static_cast<std::uint64_t>(rr),
                                     static_cast<std::uint64_t>(b),
                                     static_cast<std::uint64_t>(blocks))];
      }
      int2 lo[kGemvRows];
      int2 hi[kGemvRows];
      float sc[kGemvRows];
#pragma unroll
      for (int r = 0; r < kGemvRows; ++r) {
        lo[r] = get_int_from_table_16(static_cast<int>(qs[r].x), kvalues_mxfp4);
        hi[r] = get_int_from_table_16(static_cast<int>(qs[r].y), kvalues_mxfp4);
        sc[r] = ggml_cuda_ue4m3_to_fp32(ss[r]);
      }
#pragma unroll
      for (int m = 0; m < kMatched; ++m) {
        if (m < m_count) {
          const int s = matched[m];
          const int t = s / a.used;
          const int j = s % a.used;
          const float* x = a.x + (t * a.x_token) + (j * a.x_slot);
          const float4* xv = reinterpret_cast<const float4*>(x + (b * kSub));
          const float4 xs[4] = {xv[0], xv[1], xv[2], xv[3]};
          const float vals[kSub] = {xs[0].x, xs[0].y, xs[0].z, xs[0].w, xs[1].x, xs[1].y,
                                    xs[1].z, xs[1].w, xs[2].x, xs[2].y, xs[2].z, xs[2].w,
                                    xs[3].x, xs[3].y, xs[3].z, xs[3].w};
          float amax = 0.0f;
#pragma unroll
          for (int i = 0; i < kSub; ++i) {
            amax = fmaxf(amax, fabsf(vals[i]));
          }
          const float d = amax / 127.0f;
          int xq[4] = {0, 0, 0, 0};
          if (amax > 0.0f) {
#pragma unroll
            for (int i = 0; i < kSub; ++i) {
              const auto q8 = static_cast<std::uint32_t>(
                  static_cast<std::uint8_t>(static_cast<std::int8_t>(roundf(vals[i] / d))));
              const int word = ((i / 8) * 2) + (i % 2);
              xq[word] |= static_cast<int>(q8 << (8U * static_cast<std::uint32_t>((i % 8) / 2)));
            }
          }
#pragma unroll
          for (int r = 0; r < kGemvRows; ++r) {
            int dot = ggml_cuda_dp4a(lo[r].x, xq[0], 0);
            dot = ggml_cuda_dp4a(lo[r].y, xq[1], dot);
            dot = ggml_cuda_dp4a(hi[r].x, xq[2], dot);
            dot = ggml_cuda_dp4a(hi[r].y, xq[3], dot);
            sum[m][r] = fmaf(static_cast<float>(dot), sc[r] * d, sum[m][r]);
          }
        }
      }
    }
    ggml_cuda_pdl_lc();
#pragma unroll
    for (int m = 0; m < kMatched; ++m) {
      if (m < m_count) {
#pragma unroll
        for (int r = 0; r < kGemvRows; ++r) {
#pragma unroll
          for (int offset = 16; offset > 0; offset >>= 1) {
            sum[m][r] += __shfl_xor_sync(0xffffffffu, sum[m][r], offset, 32);
          }
        }
        if (lane == 0) {
          float* out = a.y + (static_cast<std::int64_t>(matched[m]) * a.n) + n0;
#pragma unroll
          for (int r = 0; r < kOutputs; ++r) {
            if (r < rows) {
              if constexpr (kGlu) {
                out[r] = Silu(sum[m][r] * a.gate_scale[e]) * (sum[m][r + kOutputs] * a.up_scale[e]);
              } else {
                out[r] = sum[m][r];
              }
            }
          }
        }
      }
    }
  }
}

// ------------------------------------------------------------------ layout conversion

// One projection of the experts' slots: its GGML rows at `ggml` (each k / 64
// blocks of 36 bytes: 4 E4M3 scales, then 32 bytes in which byte 8s + j
// holds elements 16s + j and 16s + j + 8) and its CUTLASS rows row0 .. of a
// block of `rows` rows at `codes` and `scales`.
struct Projection {
  std::uint64_t ggml;
  std::uint64_t codes;
  std::uint64_t scales;
  int rows_here;
  int row0;
  int k;
};

// One 64-value block a thread: GGML to CUTLASS (`to_cutlass`) or back.
__global__ void ConvertBlocks(const std::uint8_t* __restrict__ src, std::uint8_t* __restrict__ dst,
                              std::uint64_t src_stride, std::uint64_t dst_stride, int experts,
                              Projection p, bool to_cutlass) {
  const int per_row = p.k / 64;
  const std::int64_t i = (static_cast<std::int64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  const std::int64_t total = static_cast<std::int64_t>(experts) * p.rows_here * per_row;
  if (i >= total) {
    return;
  }
  const auto e = static_cast<std::uint64_t>(i / (static_cast<std::int64_t>(p.rows_here) * per_row));
  const int r = static_cast<int>((i / per_row) % p.rows_here);
  const int blk = static_cast<int>(i % per_row);
  const auto row = static_cast<std::uint64_t>(p.row0 + r);
  const auto blocks16 = static_cast<std::uint64_t>(p.k / kSub);
  const std::uint64_t g_off =
      p.ggml + (static_cast<std::uint64_t>(r) * static_cast<std::uint64_t>(per_row) * 36) +
      (static_cast<std::uint64_t>(blk) * 36);
  const std::uint64_t c_off = p.codes + (row * static_cast<std::uint64_t>(p.k / 2)) +
                              (static_cast<std::uint64_t>(blk) * 32);
  if (to_cutlass) {
    const std::uint8_t* g = src + (e * src_stride) + g_off;
    std::uint8_t* c = dst + (e * dst_stride);
    std::uint8_t v[64];
#pragma unroll
    for (int s = 0; s < 4; ++s) {
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const std::uint8_t b = g[4 + (8 * s) + j];
        v[(16 * s) + j] = b & 15U;
        v[(16 * s) + j + 8] = b >> 4U;
      }
      c[p.scales + moe::SfOffset(row, static_cast<std::uint64_t>((blk * 4) + s), blocks16)] = g[s];
    }
#pragma unroll
    for (int b = 0; b < 32; ++b) {
      c[c_off + b] = static_cast<std::uint8_t>(v[2 * b] | (v[(2 * b) + 1] << 4U));
    }
  } else {
    const std::uint8_t* c = src + (e * src_stride);
    std::uint8_t* g = dst + (e * dst_stride) + g_off;
    std::uint8_t v[64];
#pragma unroll
    for (int b = 0; b < 32; ++b) {
      const std::uint8_t byte = c[c_off + b];
      v[2 * b] = byte & 15U;
      v[(2 * b) + 1] = byte >> 4U;
    }
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      g[s] = c[p.scales + moe::SfOffset(row, static_cast<std::uint64_t>((blk * 4) + s), blocks16)];
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        g[4 + (8 * s) + j] =
            static_cast<std::uint8_t>(v[(16 * s) + j] | (v[(16 * s) + j + 8] << 4U));
      }
    }
  }
}

unsigned Blocks(std::int64_t items, int threads) {
  return static_cast<unsigned>((items + threads - 1) / threads);
}

bool SlabFits(const moe::ExpertSlab& s) {
  const moe::ExpertLayout& l = s.layout;
  const std::uint64_t gate_up = l.ffn * (l.width / 64) * 36;
  const std::uint64_t down = l.width * (l.ffn / 64) * 36;
  return s.base != nullptr && s.experts > 0 && l.ffn % 64 == 0 && l.width % 128 == 0 &&
         l.ffn % 64 == 0 && s.stride % 16 == 0 && l.bytes() <= s.stride &&
         s.gate + gate_up <= s.stride && s.up + gate_up <= s.stride && s.down + down <= s.stride;
}

void ConvertAll(const moe::ExpertSlab& s, const std::uint8_t* src, std::uint8_t* dst,
                std::int64_t experts, bool to_cutlass, cudaStream_t stream) {
  const moe::ExpertLayout& l = s.layout;
  const auto f = static_cast<int>(l.ffn);
  const auto w = static_cast<int>(l.width);
  const Projection parts[3] = {{.ggml = s.gate,
                                .codes = l.gate_up_codes(),
                                .scales = l.gate_up_scales(),
                                .rows_here = f,
                                .row0 = 0,
                                .k = w},
                               {.ggml = s.up,
                                .codes = l.gate_up_codes(),
                                .scales = l.gate_up_scales(),
                                .rows_here = f,
                                .row0 = f,
                                .k = w},
                               {.ggml = s.down,
                                .codes = l.down_codes(),
                                .scales = l.down_scales(),
                                .rows_here = w,
                                .row0 = 0,
                                .k = f}};
  for (const Projection& p : parts) {
    const std::int64_t items = experts * p.rows_here * (p.k / 64);
    ConvertBlocks<<<Blocks(items, 256), 256, 0, stream>>>(src, dst, s.stride, s.stride,
                                                          static_cast<int>(experts), p, to_cutlass);
  }
}

}  // namespace

namespace moe {

bool ToCutlassLayout(const ExpertSlab& slab, void* temp, std::int64_t batch, void* stream) {
  if (!SlabFits(slab) || temp == nullptr || batch <= 0) {
    return false;
  }
  auto* s = static_cast<cudaStream_t>(stream);
  auto* base = static_cast<std::uint8_t*>(slab.base);
  auto* t = static_cast<std::uint8_t*>(temp);
  for (std::int64_t e0 = 0; e0 < slab.experts; e0 += batch) {
    const std::int64_t n = std::min(batch, slab.experts - e0);
    const std::uint64_t bytes = static_cast<std::uint64_t>(n) * slab.stride;
    std::uint8_t* src = base + (static_cast<std::uint64_t>(e0) * slab.stride);
    (void)cudaMemsetAsync(t, 0, bytes, s);
    ConvertAll(slab, src, t, n, true, s);
    (void)cudaMemcpyAsync(src, t, bytes, cudaMemcpyDeviceToDevice, s);
  }
  return true;
}

bool ToGgmlLayout(const ExpertSlab& slab, void* out, void* stream) {
  if (!SlabFits(slab) || out == nullptr) {
    return false;
  }
  ConvertAll(slab, static_cast<const std::uint8_t*>(slab.base), static_cast<std::uint8_t*>(out),
             slab.experts, false, static_cast<cudaStream_t>(stream));
  return true;
}

}  // namespace moe

// ------------------------------------------------------------------ launchers

std::expected<void, KernelFailure> RunMoeRoute(LaunchContext& launch, ggml_tensor* node) {
  if (auto checked = CheckMoeRoute(node); !checked) {
    return checked;
  }
  return launch.Run(base::Bytes(0), [node](ggml_backend_cuda_context& context) {
    const Route r = RouteOf(node);
    const ggml_tensor* ids = node->src[0];
    const auto* id = static_cast<const std::int32_t*>(ids->data);
    const int stride = static_cast<int>(ids->nb[1] / sizeof(std::int32_t));
    const std::size_t shared = static_cast<std::size_t>(r.experts) * sizeof(int);
    // Rows no slot reaches (ids outside the experts) keep token and expert
    // -1.
    (void)cudaMemsetAsync(const_cast<std::int32_t*>(r.token), 0xff,
                          2 * sizeof(std::int32_t) * static_cast<std::size_t>(r.used) *
                              static_cast<std::size_t>(r.tokens),
                          context.stream());
    RouteCount<<<static_cast<unsigned>(r.chunks), 256, shared, context.stream()>>>(id, stride, r);
    RouteScan<<<1, 512, shared, context.stream()>>>(r);
    RouteAssign<<<static_cast<unsigned>(r.chunks), 128, shared, context.stream()>>>(id, stride, r);
  });
}

std::expected<void, KernelFailure> RunMoeQuantize(LaunchContext& launch, ggml_tensor* node) {
  if (auto checked = CheckMoeQuantize(node); !checked) {
    return checked;
  }
  return launch.Run(base::Bytes(0), [node](ggml_backend_cuda_context& context) {
    const Route r = RouteOf(node->src[1]);
    const moe::QuantLayout q{
        .k = static_cast<std::uint64_t>(JitllmOpInt(node, 3)),
        .slots = static_cast<std::uint64_t>(r.used) * static_cast<std::uint64_t>(r.tokens),
        .experts = static_cast<std::uint64_t>(r.experts)};
    const Quantized out = QuantizedOf(node->data, q);
    // The padding rows' scales are zero, never stale bytes.
    (void)cudaMemsetAsync(out.scales, 0, q.row_scales() - q.scales(), context.stream());
    QuantizeRows<<<static_cast<unsigned>(r.tokens), 128, 0, context.stream()>>>(
        static_cast<const float*>(node->src[0]->data), r, out);
  });
}

std::expected<std::uint64_t, KernelFailure> PlanMoeGemm(const LaunchContext& launch,
                                                        const ggml_tensor* node) {
  if (auto checked = CheckMoeGemm(node); !checked) {
    return std::unexpected(checked.error());
  }
  const auto& device = ggml_cuda_info().devices[launch.device()];
  if (device.cc != 1210 || !moe::GroupedGemmAvailable()) {
    return Refused("the NVFP4 grouped GEMM runs on a compute capability 12.1 device only");
  }
  return static_cast<std::uint64_t>(moe::GroupedGemmScratch(JitllmOpInt(node, 0), device.nsm));
}

std::expected<void, KernelFailure> RunMoeGemm(LaunchContext& launch, ggml_tensor* node) {
  auto scratch = PlanMoeGemm(launch, node);
  if (!scratch) {
    return std::unexpected(scratch.error());
  }
  const int sms = ggml_cuda_info().devices[launch.device()].nsm;
  int status = 0;
  auto ran = launch.Run(base::Bytes(*scratch), [&](ggml_backend_cuda_context& context) {
    ggml_cuda_pool_alloc<std::uint8_t> pool(context.pool(), *scratch);
    const Route r = RouteOf(node->src[1]);
    const moe::QuantLayout q{
        .k = static_cast<std::uint64_t>(JitllmOpInt(node, 3)),
        .slots = static_cast<std::uint64_t>(r.used) * static_cast<std::uint64_t>(r.tokens),
        .experts = static_cast<std::uint64_t>(r.experts)};
    const auto* a = static_cast<const std::uint8_t*>(node->src[0]->data);
    const ggml_tensor* weights = node->src[2];
    const moe::GroupedGemm gemm{.groups = r.experts,
                                .n = JitllmOpInt(node, 4),
                                .k = JitllmOpInt(node, 3),
                                .offsets = r.offsets,
                                .sf_rows = r.scale_rows,
                                .a = a + q.codes(),
                                .a_scales = a + q.scales(),
                                .weights = weights->data,
                                .expert_stride = weights->nb[1],
                                .codes_offset = static_cast<std::uint64_t>(JitllmOpInt(node, 5)),
                                .scales_offset = static_cast<std::uint64_t>(JitllmOpInt(node, 6)),
                                .d = node->data,
                                .scratch = pool.get()};
    status = moe::RunGroupedGemm(gemm, sms, context.stream());
  });
  if (!ran) {
    return ran;
  }
  if (status != 0) {
    return Refused(std::format("CUTLASS refused the grouped GEMM (status {})", status - 1));
  }
  return {};
}

std::expected<void, KernelFailure> RunMoeGluQuantize(LaunchContext& launch, ggml_tensor* node) {
  if (auto checked = CheckMoeGluQuantize(node); !checked) {
    return checked;
  }
  return launch.Run(base::Bytes(0), [node](ggml_backend_cuda_context& context) {
    const Route r = RouteOf(node->src[2]);
    const auto slots = static_cast<std::uint64_t>(r.used) * static_cast<std::uint64_t>(r.tokens);
    const auto experts = static_cast<std::uint64_t>(r.experts);
    const moe::QuantLayout q{
        .k = static_cast<std::uint64_t>(JitllmOpInt(node, 3)), .slots = slots, .experts = experts};
    const moe::QuantLayout in{.k = static_cast<std::uint64_t>(JitllmOpInt(node->src[1], 3)),
                              .slots = slots,
                              .experts = experts};
    const Quantized out = QuantizedOf(node->data, q);
    const Quantized a = QuantizedOf(node->src[1]->data, in);
    (void)cudaMemsetAsync(out.scales, 0, q.row_scales() - q.scales(), context.stream());
    GluQuantizeRows<<<static_cast<unsigned>(slots), 128, q.k * sizeof(float), context.stream()>>>(
        static_cast<const __nv_bfloat16*>(node->src[0]->data), a.row_scales,
        static_cast<const float*>(node->src[3]->data),
        static_cast<const float*>(node->src[4]->data), r, out);
  });
}

std::expected<void, KernelFailure> RunMoeCombineSorted(LaunchContext& launch, ggml_tensor* node) {
  if (auto checked = CheckMoeCombineSorted(node); !checked) {
    return checked;
  }
  return launch.Run(base::Bytes(0), [node](ggml_backend_cuda_context& context) {
    const Route r = RouteOf(node->src[2]);
    const moe::QuantLayout in{
        .k = static_cast<std::uint64_t>(JitllmOpInt(node->src[1], 3)),
        .slots = static_cast<std::uint64_t>(r.used) * static_cast<std::uint64_t>(r.tokens),
        .experts = static_cast<std::uint64_t>(r.experts)};
    const Quantized a = QuantizedOf(node->src[1]->data, in);
    const std::int64_t total4 = ggml_nelements(node) / 4;
    CombineSorted<<<Blocks(total4, 256), 256, 0, context.stream()>>>(
        static_cast<const __nv_bfloat16*>(node->src[0]->data), a.row_scales,
        static_cast<const float*>(node->src[3]->data),
        static_cast<const float*>(node->src[4]->data),
        static_cast<const float4*>(node->src[5]->data),
        static_cast<const float*>(node->src[6]->data), static_cast<float4*>(node->data), r,
        static_cast<int>(node->ne[0] / 4), total4);
  });
}

std::expected<void, KernelFailure> RunMoeGemv(LaunchContext& launch, ggml_tensor* node) {
  if (auto checked = CheckMoeGemv(node); !checked) {
    return checked;
  }
  return launch.Run(base::Bytes(0), [node](ggml_backend_cuda_context& context) {
    const ggml_tensor* w = node->src[0];
    const ggml_tensor* x = node->src[1];
    const ggml_tensor* ids = node->src[2];
    const bool glu = IsMoeGemvSwiglu(node);
    const auto f = [](std::size_t bytes) { return static_cast<std::int64_t>(bytes / 4); };
    const GemvArgs a{.weights = static_cast<const std::uint8_t*>(w->data),
                     .stride = w->nb[1],
                     .experts = static_cast<int>(w->ne[1]),
                     .codes = static_cast<std::uint64_t>(JitllmOpInt(node, 2)),
                     .scales = static_cast<std::uint64_t>(JitllmOpInt(node, 3)),
                     .row0 = JitllmOpInt(node, 0),
                     .n = static_cast<int>(node->ne[0]),
                     .k = static_cast<int>(x->ne[0]),
                     .x = static_cast<const float*>(x->data),
                     .x_slot = x->ne[1] == 1 ? 0 : f(x->nb[1]),
                     .x_token = f(x->nb[2]),
                     .ids = static_cast<const std::int32_t*>(ids->data),
                     .ids_stride = static_cast<int>(ids->nb[1] / sizeof(std::int32_t)),
                     .used = static_cast<int>(ids->ne[0]),
                     .gate_scale = glu ? static_cast<const float*>(node->src[3]->data) : nullptr,
                     .up_scale = glu ? static_cast<const float*>(node->src[4]->data) : nullptr,
                     .y = static_cast<float*>(node->data)};
    const int outputs = glu ? kGemvOutputs<true> : kGemvOutputs<false>;
    const std::int64_t per_block = 8LL * outputs;
    const dim3 grid(static_cast<unsigned>((node->ne[0] + per_block - 1) / per_block),
                    static_cast<unsigned>(ids->ne[0] * ids->ne[1]));
    const ggml_cuda_kernel_launch_params params(grid, dim3(256), 0, context.stream());
    if (MoeGemvExpertMajor(ids->ne[1])) {
      // Several output tiles a block, one leader scan for them all: the
      // SwiGLU (gate/up) form 4, the down projection 8 (C4 HTTP profiles on
      // GB10, a layer: 588 -> 544 and 458 -> 312 us).
      const unsigned tiles = glu ? 4 : 8;
      const dim3 tiled((grid.x + tiles - 1) / tiles, grid.y);
      const ggml_cuda_kernel_launch_params p(tiled, dim3(256), 0, context.stream());
      if (glu) {
        ggml_cuda_kernel_launch(GemvExperts<true, 4>, p, a);
      } else {
        ggml_cuda_kernel_launch(GemvExperts<false, 8>, p, a);
      }
      return;
    }
    if (glu) {
      ggml_cuda_kernel_launch(Gemv<true>, params, a);
    } else {
      ggml_cuda_kernel_launch(Gemv<false>, params, a);
    }
  });
}

}  // namespace jitllm::kernels::ggml
