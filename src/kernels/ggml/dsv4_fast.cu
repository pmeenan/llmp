// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: MIT AND Apache-2.0

// DeepSeek V4's fast decode plan (llmp_ops.h, "DeepSeek V4's fast plan"):
// llmpalooza's kernels for the routing, the routed experts' combination,
// the compressor and the hyper-connections' pre-mix (the quantized vector
// products these plans use are llmp.vecq, vecq.cu). None reproduces GGML's
// arithmetic bit for bit (the owner's policy, 2026-09-28: speed first,
// correctness judged coarsely against llama.cpp); each is deterministic (no
// atomics, fixed summation orders), so a model's own runs, swaps and
// restores still repeat bit for bit.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
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
#include "kernels/ggml/launch.h"
#include "kernels/ggml/llmp_ops.h"
#include "quantize.cuh"
#include "unary.cuh"
#include "vecdotq.cuh"

namespace llmp::kernels::ggml {
namespace {

// ---------------------------------------------------------------- routing

constexpr int kExperts = 256;
constexpr int kPerLane = kExperts / 32;
constexpr int kRouteWarps = 4;

struct RouteArgs {
  const float* logits = nullptr;
  int logits_stride = 0;  // floats between tokens
  int experts = kExperts;
  bool selected_ids = false;
  bool planar_output = false;
  const float* bias = nullptr;
  const std::int32_t* table = nullptr;  // the hash layers' [used, vocab]
  const std::int32_t* tokens = nullptr;
  int vocab = 0;  // the table's rows, into which a token ID is clamped
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
    const float v = e < a.experts ? a.logits[(t * a.logits_stride) + e] : 0.0f;
    prob[i] = e < a.experts ? sqrtf(v > 20.0f ? v : logf(1.0f + expf(v))) : 0.0f;
    select[i] = e >= a.experts ? -INFINITY : a.bias != nullptr ? prob[i] + a.bias[e] : prob[i];
  }
  // Lane k keeps slot k's expert and weight.
  int my_id = 0;
  float my_weight = 0.0f;
  float sum = 0.0f;
  for (int k = 0; k < a.used; ++k) {
    int best = 0;
    float weight = 0.0f;
    if (a.table != nullptr) {
      // A token ID outside the table is clamped into it rather than read
      // out of bounds (the hash layers' IDs are device data).
      const int row = a.selected_ids ? t : min(max(a.tokens[t], 0), a.vocab - 1);
      best = a.table[(row * a.used) + k];
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
    if (a.planar_output) {
      const auto at = static_cast<std::ptrdiff_t>(t) * a.used + lane;
      a.out[at] = my_id;
      a.out[static_cast<std::ptrdiff_t>(a.n_tokens) * a.used + at] = __float_as_int(my_weight);
    } else {
      std::int32_t* row = a.out + (static_cast<std::ptrdiff_t>(t) * 2 * a.used);
      row[lane] = my_id;
      row[a.used + lane] = __float_as_int(my_weight);
    }
  }
}

// ---------------------------------------------------------------- combine

__global__ void CombineKernel(const float4* down, const std::int32_t* route, const float4* shared,
                              float4* out, int n4, int used, const float* selected_weights) {
  ggml_cuda_pdl_sync();
  const int c = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
  const int t = static_cast<int>(blockIdx.y);
  if (c >= n4) {
    return;
  }
  const std::int32_t* weights =
      selected_weights ? nullptr : route + (static_cast<std::ptrdiff_t>(t) * 2 * used) + used;
  float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  for (int k = 0; k < used; ++k) {
    const float w = selected_weights ? selected_weights[static_cast<std::ptrdiff_t>(t) * used + k]
                                     : __int_as_float(weights[k]);
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

// A mixing weight as F32: exact for F16 and BF16.
__device__ __forceinline__ float HcWeight(float w) { return w; }
__device__ __forceinline__ float HcWeight(half w) { return __half2float(w); }
__device__ __forceinline__ float HcWeight(nv_bfloat16 w) { return __bfloat162float(w); }

// Each block: one chunk of the flattened values of up to kMixTokens tokens,
// their dot products with the kMixes weight rows and their sums of squares.
// A token's sums are the same whatever tokens share the block. The weights
// are F32, F16 or BF16 (W), each widened to F32 exactly, so the sums' order
// and arithmetic are the same for every weight type.
template <int kMixTokens, typename W>
__global__ void __launch_bounds__(kMixThreads)
    HcMixKernel(const float* x, const W* fn, float* partials, int flat, int tokens) {
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
      w[j] = HcWeight(fn[(static_cast<std::ptrdiff_t>(j) * flat) + k]);
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

template <typename W>
void LaunchHcMix(const float* x, const W* fn, float* partials, int flat, int tokens,
                 cudaStream_t stream) {
  if (tokens == 1) {
    ggml_cuda_kernel_launch(
        HcMixKernel<1, W>,
        ggml_cuda_kernel_launch_params(dim3(kDsv4HcChunks, 1), dim3(kMixThreads), 0, stream), x, fn,
        partials, flat, tokens);
    return;
  }
  constexpr int kTokens = 4;  // a block's tokens, reading the weights once
  const dim3 grid(static_cast<unsigned>(kDsv4HcChunks),
                  static_cast<unsigned>((tokens + kTokens - 1) / kTokens));
  ggml_cuda_kernel_launch(HcMixKernel<kTokens, W>,
                          ggml_cuda_kernel_launch_params(grid, dim3(kMixThreads), 0, stream), x, fn,
                          partials, flat, tokens);
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
    bool nonfinite_value = false;
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
      // Pool's primitive definition multiplies every selected value by
      // its probability, including zero masked probabilities. Preserve
      // 0 * NaN/Inf without changing the finite-value arithmetic.
      if (!isfinite(v)) nonfinite_value = true;
      if (s == -INFINITY) {
        if (!isfinite(v)) acc += 0.0f * v;
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
    a.out[(static_cast<std::ptrdiff_t>(b) * a.head) + c] =
        (sum > 0.0f || nonfinite_value) ? acc / sum : 0.0f;
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
std::expected<void, KernelFailure> RunDsv4Route(LaunchContext& launch, ggml_tensor* node) {
  if (auto checked = CheckDsv4Route(node); !checked) {
    return checked;
  }
  return launch.Run(base::Bytes(0), [node](ggml_backend_cuda_context& context) {
    const ggml_tensor* logits = node->src[0];
    const bool selected = LlmpOpInt(node, 4) == 1;
    const bool hashed = selected || node->src[2] != nullptr;
    RouteArgs a;
    a.selected_ids = selected;
    a.planar_output = LlmpOpInt(node, 4) != 0;
    a.experts = static_cast<int>(logits->ne[0]);
    a.logits = static_cast<const float*>(logits->data);
    a.logits_stride = static_cast<int>(logits->nb[1] / sizeof(float));
    a.bias = hashed ? nullptr : static_cast<const float*>(node->src[1]->data);
    a.table = hashed ? static_cast<const std::int32_t*>(node->src[1]->data) : nullptr;
    a.tokens = hashed && !selected ? static_cast<const std::int32_t*>(node->src[2]->data) : nullptr;
    a.vocab = hashed && !selected ? static_cast<int>(node->src[1]->ne[1]) : 0;
    a.out = static_cast<std::int32_t*>(node->data);
    a.n_tokens = static_cast<int>(logits->ne[1]);
    a.used = LlmpOpInt(node, 0);
    a.norm = LlmpOpInt(node, 1);
    a.clamp = LlmpOpFloat(node, 2);
    a.scale = LlmpOpFloat(node, 3);
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
    ggml_cuda_kernel_launch(
        CombineKernel, ggml_cuda_kernel_launch_params(grid, dim3(256), 0, context.stream()),
        static_cast<const float4*>(down->data),
        static_cast<const std::int32_t*>(node->src[1]->data),
        static_cast<const float4*>(node->src[2]->data), static_cast<float4*>(node->data), n4,
        static_cast<int>(down->ne[1]),
        LlmpOpInt(node, 0) == 1 ? static_cast<const float*>(node->src[1]->data) : nullptr);
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
    auto* out = static_cast<float*>(node->data);
    // CheckDsv4HcMix: flat is a multiple of kDsv4HcChunks · kMixThreads, and
    // the weights F32, F16 or BF16.
    const ggml_tensor* fn = node->src[1];
    const cudaStream_t stream = context.stream();
    switch (fn->type) {
      case GGML_TYPE_F16:
        LaunchHcMix(xs, static_cast<const half*>(fn->data), out, flat, tokens, stream);
        break;
      case GGML_TYPE_BF16:
        LaunchHcMix(xs, static_cast<const nv_bfloat16*>(fn->data), out, flat, tokens, stream);
        break;
      default:
        LaunchHcMix(xs, static_cast<const float*>(fn->data), out, flat, tokens, stream);
        break;
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
    a.ratio = LlmpOpInt(node, 0);
    a.overlap = LlmpOpInt(node, 1);
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
    a.rms_eps = LlmpOpFloat(node, 0);
    a.hc_eps = LlmpOpFloat(node, 1);
    a.iterations = LlmpOpInt(node, 2);
    ggml_cuda_kernel_launch(
        HcPreKernel,
        ggml_cuda_kernel_launch_params(dim3(static_cast<unsigned>(node->src[1]->ne[2])),
                                       dim3(kPreThreads), 0, context.stream()),
        a);
  });
}

}  // namespace llmp::kernels::ggml
