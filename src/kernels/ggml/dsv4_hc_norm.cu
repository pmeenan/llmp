// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: MIT AND Apache-2.0

// The locked GGML arithmetic of dsv4-hc.cu's HC post and norm.cu's
// rms_norm_f32<1024> over the flattened 16384 columns, followed by the F16
// rounding of convert.cu's F32-to-F16 conversion, which the cuBLAS product
// otherwise applies to the F32 normalized tensor (or, for F32 mixing
// weights, rms_norm's F32 output as it is).

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <expected>
#include <type_traits>

#include "base/bytes.h"
#include "common.cuh"
#include "kernels/ggml/dsv4_hc_norm.h"
#include "kernels/ggml/jitllm_ops.h"
#include "kernels/ggml/launch.h"

namespace jitllm::kernels::ggml {
namespace {

constexpr int kThreads = 1024;
constexpr int kPerThread = 16;  // 16384 columns

// Native rms_norm's reduction (ascending tid + j * 1024 local FMA sums,
// block_reduce<SUM, 1024>, mean, rsqrt) and F32 scale * x: stored as is
// (T float, rms_norm's own output), or then RN F16 (T half).
template <typename T>
__device__ __forceinline__ void NormalizeTo(const float (&value)[kPerThread], float sum,
                                            float epsilon, T* out, int tid) {
  extern __shared__ float sums[];
  sum = block_reduce<block_reduce_method::SUM, kThreads>(sum, sums);
  const float scale = rsqrtf(__fadd_rn(__fmul_rn(sum, 1.0f / 16384.0f), epsilon));
#pragma unroll
  for (int j = 0; j < kPerThread; ++j) {
    const float y = __fmul_rn(scale, value[j]);
    if constexpr (std::is_same_v<T, half>) {
      out[tid + (j * kThreads)] = __float2half_rn(y);
    } else {
      out[tid + (j * kThreads)] = y;
    }
  }
}

template <typename T>
__global__ __launch_bounds__(kThreads) void HcNormF16Kernel(const float* x, T* normalized,
                                                            float epsilon) {
  const int tid = static_cast<int>(threadIdx.x);
  const auto row = static_cast<std::size_t>(blockIdx.x);
  x += row * 16384;
  normalized += row * 16384;
  float value[kPerThread];
  float sum = 0.0f;
#pragma unroll
  for (int j = 0; j < kPerThread; ++j) {
    const float v = x[tid + (j * kThreads)];
    value[j] = v;
    sum = __fmaf_rn(v, v, sum);
  }
  NormalizeTo(value, sum, epsilon, normalized, tid);
}

// With kExperts, x is formed here: the ordered six-slot reduction of
// dsv4_weighted_reduce.cu (one multiply, five ascending adds) plus the
// shared expert (the add), as their separate kernels compute it.
template <bool kExperts, typename T>
__global__ __launch_bounds__(kThreads) void HcPostNormF16Kernel(
    const float* __restrict__ x, const float* __restrict__ down, const float* __restrict__ route,
    const float* __restrict__ shared, const float* __restrict__ residual,
    const float* __restrict__ post, const float* __restrict__ comb, float* __restrict__ expanded,
    T* __restrict__ normalized, float epsilon) {
  const int tid = static_cast<int>(threadIdx.x);
  const auto row = static_cast<std::size_t>(blockIdx.x);
  residual += row * 16384;
  post += row * 4;
  comb += row * 16;
  expanded += row * 16384;
  normalized += row * 16384;
  float xs[4];
#pragma unroll
  for (int m = 0; m < 4; ++m) {
    const int feature = tid + (m * kThreads);
    if constexpr (kExperts) {
      const float* d = down + (row * 6 * 4096) + feature;
      const float* w = route + (row * 6);
      float s = __fmul_rn(d[0], w[0]);
#pragma unroll
      for (int slot = 1; slot < 6; ++slot) {
        s = __fadd_rn(s, __fmul_rn(d[slot * 4096], w[slot]));
      }
      xs[m] = __fadd_rn(s, shared[(row * 4096) + feature]);
    } else {
      xs[m] = x[(row * 4096) + feature];
    }
  }
  // Every operand is loaded once: the row's four post weights and sixteen
  // comb weights, and each of this thread's four features' four streams.
  float pw[4];
  float cw[16];
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    pw[k] = post[k];
  }
#pragma unroll
  for (int k = 0; k < 16; ++k) {
    cw[k] = comb[k];
  }
  float res[4][4];
#pragma unroll
  for (int m = 0; m < 4; ++m) {
#pragma unroll
    for (int source = 0; source < 4; ++source) {
      res[m][source] = residual[tid + (m * kThreads) + (source * 4096)];
    }
  }
  float value[kPerThread];
  float sum = 0.0f;
#pragma unroll
  for (int j = 0; j < kPerThread; ++j) {
    const int col = tid + (j * kThreads);
    const int m = j % 4;
    const int destination = j / 4;
    // The separate post producer's F32 value: its product, then the four
    // source-ordered FMAs (the screened HC-post/RMS fusion's order).
    float v = __fmul_rn(xs[m], pw[destination]);
#pragma unroll
    for (int source = 0; source < 4; ++source) {
      v = __fmaf_rn(res[m][source], cw[destination + (source * 4)], v);
    }
    value[j] = v;
    expanded[col] = v;
    sum = __fmaf_rn(v, v, sum);
  }
  NormalizeTo(value, sum, epsilon, normalized, tid);
}

// The norm's rows as the kernels take them: half for F16, float for F32.
template <typename F>
void WithRows(ggml_tensor* norm, F&& launch) {
  if (norm->type == GGML_TYPE_F32) {
    launch(static_cast<float*>(norm->data));
  } else {
    launch(static_cast<half*>(norm->data));
  }
}

}  // namespace

std::expected<void, KernelFailure> RunDsv4HcNormF16(LaunchContext& launch, ggml_tensor* norm) {
  if (auto checked = CheckDsv4HcNormF16(norm); !checked) {
    return checked;
  }
  const auto* x = static_cast<const float*>(norm->src[0]->data);
  const auto rows = static_cast<unsigned>(norm->ne[1]);
  const float epsilon = JitllmOpEps(norm);
  return launch.Run(base::Bytes(0), [=](auto& context) {
    WithRows(norm, [&](auto* normalized) {
      HcNormF16Kernel<<<rows, kThreads, 32 * sizeof(float), context.stream()>>>(x, normalized,
                                                                                epsilon);
    });
    CUDA_CHECK(cudaGetLastError());
  });
}

std::expected<void, KernelFailure> RunDsv4HcPostNormF16(LaunchContext& launch, ggml_tensor* post,
                                                        ggml_tensor* norm) {
  if (auto checked = CheckDsv4HcPostNormF16(post, norm); !checked) {
    return checked;
  }
  const auto* x = static_cast<const float*>(post->src[0]->data);
  const auto* residual = static_cast<const float*>(post->src[1]->data);
  const auto* weights = static_cast<const float*>(post->src[2]->data);
  const auto* comb = static_cast<const float*>(post->src[3]->data);
  auto* expanded = static_cast<float*>(post->data);
  const auto rows = static_cast<unsigned>(post->ne[2]);
  const float epsilon = JitllmOpEps(norm);
  return launch.Run(base::Bytes(0), [=](auto& context) {
    WithRows(norm, [&](auto* normalized) {
      HcPostNormF16Kernel<false><<<rows, kThreads, 32 * sizeof(float), context.stream()>>>(
          x, nullptr, nullptr, nullptr, residual, weights, comb, expanded, normalized, epsilon);
    });
    CUDA_CHECK(cudaGetLastError());
  });
}

std::expected<void, KernelFailure> RunDsv4HcPostExpertsNormF16(LaunchContext& launch,
                                                               ggml_tensor* reduce,
                                                               ggml_tensor* add, ggml_tensor* post,
                                                               ggml_tensor* norm) {
  if (auto checked = CheckDsv4HcPostExpertsNormF16(reduce, add, post, norm); !checked) {
    return checked;
  }
  const ggml_tensor* shared_tensor = add->src[0] == reduce ? add->src[1] : add->src[0];
  const auto* down = static_cast<const float*>(reduce->src[0]->data);
  const auto* route = static_cast<const float*>(reduce->src[1]->data);
  const auto* shared = static_cast<const float*>(shared_tensor->data);
  const auto* residual = static_cast<const float*>(post->src[1]->data);
  const auto* weights = static_cast<const float*>(post->src[2]->data);
  const auto* comb = static_cast<const float*>(post->src[3]->data);
  auto* expanded = static_cast<float*>(post->data);
  const auto rows = static_cast<unsigned>(post->ne[2]);
  const float epsilon = JitllmOpEps(norm);
  return launch.Run(base::Bytes(0), [=](auto& context) {
    WithRows(norm, [&](auto* normalized) {
      HcPostNormF16Kernel<true><<<rows, kThreads, 32 * sizeof(float), context.stream()>>>(
          nullptr, down, route, shared, residual, weights, comb, expanded, normalized, epsilon);
    });
    CUDA_CHECK(cudaGetLastError());
  });
}

namespace {

__global__ void F16CopyKernel(const float4* __restrict__ x, half2* __restrict__ y, unsigned n4) {
  const unsigned i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i >= n4) {
    return;
  }
  const float4 v = x[i];
  // convert.cu's F32-to-F16 conversion, element by element (RN).
  y[(2 * i) + 0] = make_half2(__float2half(v.x), __float2half(v.y));
  y[(2 * i) + 1] = make_half2(__float2half(v.z), __float2half(v.w));
}

}  // namespace

std::expected<void, KernelFailure> RunDsv4F16Copy(LaunchContext& launch, ggml_tensor* copy) {
  if (auto checked = CheckDsv4F16Copy(copy); !checked) {
    return checked;
  }
  const auto* x = static_cast<const float4*>(copy->src[0]->data);
  auto* y = static_cast<half2*>(copy->data);
  const auto n4 = static_cast<unsigned>(ggml_nelements(copy) / 4);
  return launch.Run(base::Bytes(0), [=](auto& context) {
    F16CopyKernel<<<(n4 + 255) / 256, 256, 0, context.stream()>>>(x, y, n4);
    CUDA_CHECK(cudaGetLastError());
  });
}

}  // namespace jitllm::kernels::ggml
