// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <format>
#include <initializer_list>
#include <limits>
#include <string>

#include "kernels/image/ops.h"

namespace llmp::kernels::image {
namespace {

constexpr std::int64_t kMaxGrid = std::int64_t{1} << 30;

// ---- BF16 as PyTorch rounds it: round to nearest even, NaN kept quiet.

__device__ __forceinline__ float B2F(std::uint16_t b) {
  return __uint_as_float(static_cast<std::uint32_t>(b) << 16U);
}

__device__ __forceinline__ std::uint16_t F2B(float f) {
  std::uint32_t u = __float_as_uint(f);
  if ((u & 0x7fffffffU) > 0x7f800000U) {
    return static_cast<std::uint16_t>((u >> 16U) | 0x40U);
  }
  u += 0x7fffU + ((u >> 16U) & 1U);
  return static_cast<std::uint16_t>(u >> 16U);
}

// The value a BF16 tensor would hold.
__device__ __forceinline__ float R(float f) { return B2F(F2B(f)); }

__device__ __forceinline__ std::uint16_t F2H(float f) {
  return __half_as_ushort(__float2half_rn(f));
}

__device__ __forceinline__ float SiluF(float x) { return x / (1.0f + expf(-x)); }

// Sum over a block of `threads` (a multiple of 32, at most 1,024) threads;
// every thread gets the total. `shared` holds 32 floats.
__device__ __forceinline__ float BlockSum(float v, float* shared) {
  for (int o = 16; o > 0; o >>= 1) {
    v += __shfl_xor_sync(0xffffffffU, v, o);
  }
  const int warp = static_cast<int>(threadIdx.x) / 32;
  const int lane = static_cast<int>(threadIdx.x) % 32;
  __syncthreads();
  if (lane == 0) {
    shared[warp] = v;
  }
  __syncthreads();
  const int warps = static_cast<int>(blockDim.x) / 32;
  float total = 0.0f;
  for (int w = 0; w < warps; ++w) {
    total += shared[w];
  }
  return total;
}

__device__ __forceinline__ float BlockMax(float v, float* shared) {
  for (int o = 16; o > 0; o >>= 1) {
    v = fmaxf(v, __shfl_xor_sync(0xffffffffU, v, o));
  }
  const int warp = static_cast<int>(threadIdx.x) / 32;
  const int lane = static_cast<int>(threadIdx.x) % 32;
  __syncthreads();
  if (lane == 0) {
    shared[warp] = v;
  }
  __syncthreads();
  const int warps = static_cast<int>(blockDim.x) / 32;
  float m = -INFINITY;
  for (int w = 0; w < warps; ++w) {
    m = fmaxf(m, shared[w]);
  }
  return m;
}

__device__ __forceinline__ float WarpSum(float v) {
  for (int o = 16; o > 0; o >>= 1) {
    v += __shfl_xor_sync(0xffffffffU, v, o);
  }
  return v;
}

Status Launched(const char* what) {
  const cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    return std::unexpected(std::format("{}: {}", what, cudaGetErrorString(error)));
  }
  return {};
}

Status Refuse(const char* what) { return std::unexpected(std::string(what)); }

cudaStream_t S(Stream stream) { return static_cast<cudaStream_t>(stream); }

unsigned Blocks(std::int64_t n, int threads) {
  const std::int64_t b = (n + threads - 1) / threads;
  return static_cast<unsigned>(b < 1 ? 1 : (b > 65535 * 64 ? 65535 * 64 : b));
}

// ---- elementwise

__global__ void AddKernel(const std::uint16_t* a, const std::uint16_t* b, std::uint16_t* out,
                          std::int64_t n) {
  for (std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
    out[i] = F2B(B2F(a[i]) + B2F(b[i]));
  }
}

__global__ void SiluKernel(const std::uint16_t* x, std::uint16_t* out, std::int64_t n) {
  for (std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
    out[i] = F2B(SiluF(B2F(x[i])));
  }
}

__global__ void GeluTanhKernel(const std::uint16_t* x, std::uint16_t* out, std::int64_t n) {
  // PyTorch's GeluCUDAKernelImpl, tanh approximation, in F32.
  constexpr float kBeta = static_cast<float>(M_SQRT2 * M_2_SQRTPI * 0.5);
  constexpr float kKappa = 0.044715f;
  for (std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
    const float v = B2F(x[i]);
    const float cube = v * v * v;
    const float inner = kBeta * (v + kKappa * cube);
    out[i] = F2B(0.5f * v * (1.0f + tanhf(inner)));
  }
}

__global__ void SwiGluKernel(const std::uint16_t* gate, std::int64_t gate_stride,
                             const std::uint16_t* up, std::int64_t up_stride, std::uint16_t* out,
                             std::int64_t rows, std::int64_t width) {
  const std::int64_t n = rows * width;
  for (std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
    const std::int64_t r = i / width;
    const std::int64_t c = i % width;
    const float g = R(SiluF(B2F(gate[r * gate_stride + c])));
    out[i] = F2B(g * B2F(up[r * up_stride + c]));
  }
}

// The same, 8 elements per thread (widths and strides multiples of 8,
// 16-byte aligned rows); `chunks` is width / 8.
__global__ void SwiGlu8Kernel(const std::uint16_t* gate, std::int64_t gate_stride,
                              const std::uint16_t* up, std::int64_t up_stride, std::uint16_t* out,
                              std::int64_t rows, std::int64_t chunks) {
  const std::int64_t n = rows * chunks;
  for (std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
    const std::int64_t r = i / chunks;
    const std::int64_t c = (i - (r * chunks)) * 8;
    const uint4 gp = *reinterpret_cast<const uint4*>(gate + r * gate_stride + c);
    const uint4 up8 = *reinterpret_cast<const uint4*>(up + r * up_stride + c);
    const auto* g = reinterpret_cast<const std::uint16_t*>(&gp);
    const auto* u = reinterpret_cast<const std::uint16_t*>(&up8);
    uint4 result;
    auto* o = reinterpret_cast<std::uint16_t*>(&result);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      o[j] = F2B(R(SiluF(B2F(g[j]))) * B2F(u[j]));
    }
    *reinterpret_cast<uint4*>(out + (r * chunks * 8) + c) = result;
  }
}

__global__ void Bf16ToF16Kernel(const std::uint16_t* x, std::uint16_t* out, std::int64_t n) {
  for (std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
    out[i] = F2H(B2F(x[i]));
  }
}

__global__ void F32ToBf16Kernel(const float* x, std::uint16_t* out, std::int64_t n) {
  for (std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
    out[i] = F2B(x[i]);
  }
}

__global__ void Bf16ToF32Kernel(const std::uint16_t* x, float* out, std::int64_t n) {
  for (std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
    out[i] = B2F(x[i]);
  }
}

__global__ void Bf16RowsToF16Kernel(const std::uint16_t* x, std::int64_t x_stride,
                                    std::uint16_t* out, std::int64_t rows, std::int64_t width) {
  const std::int64_t n = rows * width;
  for (std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
    out[i] = F2H(B2F(x[(i / width) * x_stride + (i % width)]));
  }
}

__global__ void EmbedRowsKernel(const std::uint16_t* table, std::int64_t table_rows,
                                const std::int32_t* ids, std::int64_t width, std::uint16_t* out,
                                std::int32_t* bad) {
  const std::int64_t r = blockIdx.x;
  const std::int32_t id = ids[r];
  const bool ok = id >= 0 && id < table_rows;
  if (!ok && threadIdx.x == 0) {
    *bad = 1;
  }
  for (std::int64_t c = threadIdx.x; c < width; c += blockDim.x) {
    out[r * width + c] = ok ? table[static_cast<std::int64_t>(id) * width + c] : 0;
  }
}

// ---- row norms (one block per row)

__global__ void RmsNormKernel(const std::uint16_t* x, const std::uint16_t* w, std::uint16_t* out,
                              std::int64_t width, float eps) {
  __shared__ float shared[32];
  const std::uint16_t* row = x + blockIdx.x * width;
  float sum = 0.0f;
  for (std::int64_t c = threadIdx.x; c < width; c += blockDim.x) {
    const float v = B2F(row[c]);
    sum += v * v;
  }
  const float rrms = rsqrtf(BlockSum(sum, shared) / static_cast<float>(width) + eps);
  for (std::int64_t c = threadIdx.x; c < width; c += blockDim.x) {
    out[blockIdx.x * width + c] = F2B(B2F(w[c]) * R(B2F(row[c]) * rrms));
  }
}

__global__ void ZeroCenterRmsNormKernel(const std::uint16_t* x, const std::uint16_t* w,
                                        std::uint16_t* out, std::int64_t width, float eps) {
  __shared__ float shared[32];
  const std::uint16_t* row = x + blockIdx.x * width;
  float sum = 0.0f;
  for (std::int64_t c = threadIdx.x; c < width; c += blockDim.x) {
    const float v = B2F(row[c]);
    sum += v * v;
  }
  const float rrms = rsqrtf(BlockSum(sum, shared) / static_cast<float>(width) + eps);
  for (std::int64_t c = threadIdx.x; c < width; c += blockDim.x) {
    out[blockIdx.x * width + c] = F2B(B2F(row[c]) * rrms * (B2F(w[c]) + 1.0f));
  }
}

// Width a multiple of 8 and at most 8 x 1,024 (checked by the caller); each
// thread keeps its 8-element chunks in registers between the passes.
constexpr int kLnChunks = 4;  // up to 4 chunks of 8 per thread
constexpr int kLnThreads = 256;

// The gated residual's arithmetic on one 8-element chunk (GatedResidual):
// x = bf16(x + bf16(bf16(tanh(gate)) * y)), in place in `xs`.
__device__ __forceinline__ void GatedChunk(std::uint16_t* xs, const std::uint16_t* ys,
                                           const std::uint16_t* gs) {
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const float t = R(R(tanhf(B2F(gs[j]))) * B2F(ys[j]));
    xs[j] = F2B(B2F(xs[j]) + t);
  }
}

// One row per block of kLnThreads threads, thread t taking the chunks at
// t * 8 + i * kLnThreads * 8. kChunks > 0: exactly that many chunks per
// thread (width = kChunks * kLnThreads * 8), kept in registers; 0: any width
// up to kLnChunks chunks (the loop's bound at run time). kResidual: the
// gated residual first, written back to x, then the norm of the new x
// (GatedResidualNorm); the sums take the same elements in the same order
// either way, so the norm is LayerNormModulate's to the bit.
template <int kChunks, bool kResidual>
__global__ void __launch_bounds__(kLnThreads)
    LayerNormModulateKernel(std::uint16_t* x, std::uint16_t* out, std::int64_t width, float eps,
                            const std::uint16_t* mod, std::int64_t mod_stride,
                            std::int64_t first_target, const std::uint16_t* y,
                            std::int64_t y_stride, const std::uint16_t* gate,
                            std::int64_t gate_stride) {
  __shared__ float shared[32];
  constexpr int kMax = kChunks > 0 ? kChunks : kLnChunks;
  const std::int64_t r = blockIdx.x;
  std::uint16_t* row = x + r * width;
  const std::uint16_t* scale = mod + (r >= first_target ? 0 : mod_stride);
  const int chunks =
      kChunks > 0
          ? kChunks
          : static_cast<int>((width - threadIdx.x * 8 + kLnThreads * 8 - 1) / (kLnThreads * 8));
  float v[kMax][8];
  float sum = 0.0f;
#pragma unroll
  for (int k = 0; k < kMax; ++k) {
    if (kChunks == 0 && k >= chunks) {
      break;
    }
    const std::int64_t c = (threadIdx.x * 8) + (static_cast<std::int64_t>(k) * kLnThreads * 8);
    uint4 packed = *reinterpret_cast<const uint4*>(row + c);
    auto* h = reinterpret_cast<std::uint16_t*>(&packed);
    if constexpr (kResidual) {
      const uint4 yp = *reinterpret_cast<const uint4*>(y + r * y_stride + c);
      const std::uint16_t* g = gate + (r >= first_target ? 0 : gate_stride) + c;
      const uint4 gp = *reinterpret_cast<const uint4*>(g);
      GatedChunk(h, reinterpret_cast<const std::uint16_t*>(&yp),
                 reinterpret_cast<const std::uint16_t*>(&gp));
      *reinterpret_cast<uint4*>(row + c) = packed;
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      v[k][j] = B2F(h[j]);
      sum += v[k][j];
    }
  }
  const float mean = BlockSum(sum, shared) / static_cast<float>(width);
  float sq = 0.0f;
#pragma unroll
  for (int k = 0; k < kMax; ++k) {
    if (kChunks == 0 && k >= chunks) {
      break;
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float d = v[k][j] - mean;
      sq += d * d;
    }
  }
  const float rstd = rsqrtf(BlockSum(sq, shared) / static_cast<float>(width) + eps);
#pragma unroll
  for (int k = 0; k < kMax; ++k) {
    if (kChunks == 0 && k >= chunks) {
      break;
    }
    const std::int64_t c = (threadIdx.x * 8) + (static_cast<std::int64_t>(k) * kLnThreads * 8);
    const uint4 packed = *reinterpret_cast<const uint4*>(scale + c);
    const auto* s = reinterpret_cast<const std::uint16_t*>(&packed);
    uint4 result;
    auto* o = reinterpret_cast<std::uint16_t*>(&result);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      o[j] = F2B(R((v[k][j] - mean) * rstd) * R(1.0f + B2F(s[j])));
    }
    *reinterpret_cast<uint4*>(out + r * width + c) = result;
  }
}

template <bool kResidual>
void LaunchLayerNorm(std::uint16_t* x, std::uint16_t* out, std::int64_t rows, std::int64_t width,
                     float eps, const std::uint16_t* mod, std::int64_t mod_stride,
                     std::int64_t first_target, const std::uint16_t* y, std::int64_t y_stride,
                     const std::uint16_t* gate, std::int64_t gate_stride, cudaStream_t s) {
  const auto blocks = static_cast<unsigned>(rows);
  switch (width / (kLnThreads * 8) * (width % (kLnThreads * 8) == 0 ? 1 : 0)) {
    case 1:
      LayerNormModulateKernel<1, kResidual><<<blocks, kLnThreads, 0, s>>>(
          x, out, width, eps, mod, mod_stride, first_target, y, y_stride, gate, gate_stride);
      break;
    case 2:
      LayerNormModulateKernel<2, kResidual><<<blocks, kLnThreads, 0, s>>>(
          x, out, width, eps, mod, mod_stride, first_target, y, y_stride, gate, gate_stride);
      break;
    case 3:
      LayerNormModulateKernel<3, kResidual><<<blocks, kLnThreads, 0, s>>>(
          x, out, width, eps, mod, mod_stride, first_target, y, y_stride, gate, gate_stride);
      break;
    case 4:
      LayerNormModulateKernel<4, kResidual><<<blocks, kLnThreads, 0, s>>>(
          x, out, width, eps, mod, mod_stride, first_target, y, y_stride, gate, gate_stride);
      break;
    default:
      LayerNormModulateKernel<0, kResidual><<<blocks, kLnThreads, 0, s>>>(
          x, out, width, eps, mod, mod_stride, first_target, y, y_stride, gate, gate_stride);
      break;
  }
}

// ---- attention's q and k (one warp per row and head, head_dim 128)

__global__ void HeadNormRopeNeoxKernel(std::uint16_t* x, std::int64_t stride, std::int64_t rows,
                                       std::int64_t heads, const std::uint16_t* w,
                                       const std::uint16_t* cos, const std::uint16_t* sin,
                                       float eps) {
  const std::int64_t item = (blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x) / 32;
  const int lane = static_cast<int>(threadIdx.x % 32);
  if (item >= rows * heads) {
    return;
  }
  const std::int64_t r = item / heads;
  std::uint16_t* h = x + r * stride + (item % heads) * 128;
  float v[4];
  float sum = 0.0f;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    v[j] = B2F(h[lane + 32 * j]);
    sum += v[j] * v[j];
  }
  const float rrms = rsqrtf(WarpSum(sum) / 128.0f + eps);
  float n[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    n[j] = R(B2F(w[lane + 32 * j]) * R(v[j] * rrms));
  }
  // rotate_half: element i < 64 takes -n[i + 64], i >= 64 takes n[i - 64].
  const float rot[4] = {-n[2], -n[3], n[0], n[1]};
  const std::uint16_t* c = cos + r * 128;
  const std::uint16_t* s = sin + r * 128;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int i = lane + 32 * j;
    h[i] = F2B(R(n[j] * B2F(c[i])) + R(rot[j] * B2F(s[i])));
  }
}

// kOut: 0 F32, 1 F16, 2 BF16 (in place: each lane reads its four values
// before it writes them).
template <int kOut>
__global__ void HeadNormRopeComplexKernel(const std::uint16_t* x, std::int64_t x_stride,
                                          std::int64_t rows, std::int64_t heads,
                                          const std::uint16_t* w, const float* freqs, float eps,
                                          void* out) {
  const std::int64_t item = (blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x) / 32;
  const int lane = static_cast<int>(threadIdx.x % 32);
  if (item >= rows * heads) {
    return;
  }
  const std::int64_t r = item / heads;
  const std::uint16_t* h = x + r * x_stride + (item % heads) * 128 + lane * 4;
  const uint2 packed = *reinterpret_cast<const uint2*>(h);
  const auto* e = reinterpret_cast<const std::uint16_t*>(&packed);
  float v[4];
  // The contractions spelled out (as the compiler made them before they
  // were: flash_attention.cu's query norm reproduces them bit for bit).
  float sum = 0.0f;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    v[j] = B2F(e[j]);
    sum = __fmaf_rn(v[j], v[j], sum);
  }
  const float rrms = rsqrtf(__fmaf_rn(WarpSum(sum), 0.0078125f, eps));
  float n[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    n[j] = R(R(__fmul_rn(v[j], rrms)) * B2F(w[lane * 4 + j]));
  }
  const float4 f = *reinterpret_cast<const float4*>(freqs + r * 128 + lane * 4);
  // (a + ib)(c + is), as c10::complex<float>'s operator*.
  const float o0 = __fmaf_rn(n[0], f.x, -__fmul_rn(n[1], f.y));
  const float o1 = __fmaf_rn(n[0], f.y, __fmul_rn(n[1], f.x));
  const float o2 = __fmaf_rn(n[2], f.z, -__fmul_rn(n[3], f.w));
  const float o3 = __fmaf_rn(n[2], f.w, __fmul_rn(n[3], f.z));
  const std::int64_t at = item * 128 + lane * 4;
  if constexpr (kOut == 2) {
    uint2 packed_out;
    auto* o = reinterpret_cast<std::uint16_t*>(&packed_out);
    o[0] = F2B(o0);
    o[1] = F2B(o1);
    o[2] = F2B(o2);
    o[3] = F2B(o3);
    *reinterpret_cast<uint2*>(const_cast<std::uint16_t*>(h)) = packed_out;
  } else if constexpr (kOut == 1) {
    auto* o = static_cast<std::uint16_t*>(out) + at;
    o[0] = F2H(R(o0));
    o[1] = F2H(R(o1));
    o[2] = F2H(R(o2));
    o[3] = F2H(R(o3));
  } else {
    *reinterpret_cast<float4*>(static_cast<float*>(out) + at) =
        make_float4(R(o0), R(o1), R(o2), R(o3));
  }
}

// ---- residuals and the scheduler

__global__ void GatedResidualKernel(std::uint16_t* x, const std::uint16_t* y, std::int64_t y_stride,
                                    std::int64_t rows, std::int64_t width, const std::uint16_t* mod,
                                    std::int64_t mod_stride, std::int64_t first_target) {
  const std::int64_t n = rows * width / 8;
  for (std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
    const std::int64_t r = (i * 8) / width;
    const std::int64_t c = (i * 8) % width;
    const std::uint16_t* gate = mod + (r >= first_target ? 0 : mod_stride) + c;
    uint4 xp = *reinterpret_cast<const uint4*>(x + r * width + c);
    const uint4 yp = *reinterpret_cast<const uint4*>(y + r * y_stride + c);
    const uint4 gp = *reinterpret_cast<const uint4*>(gate);
    auto* xs = reinterpret_cast<std::uint16_t*>(&xp);
    const auto* ys = reinterpret_cast<const std::uint16_t*>(&yp);
    const auto* gs = reinterpret_cast<const std::uint16_t*>(&gp);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float t = R(R(tanhf(B2F(gs[j]))) * B2F(ys[j]));
      xs[j] = F2B(B2F(xs[j]) + t);
    }
    *reinterpret_cast<uint4*>(x + r * width + c) = xp;
  }
}

__global__ void EulerStepKernel(const std::uint16_t* sample, const std::uint16_t* noise,
                                std::uint16_t* out, float dt, std::int64_t n) {
  for (std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
    out[i] = F2B(B2F(sample[i]) + R(dt * B2F(noise[i])));
  }
}

__global__ void EulerStepAtKernel(const std::uint16_t* sample, const std::uint16_t* noise,
                                  std::uint16_t* out, const float* dt, std::int64_t n) {
  const float step = *dt;
  for (std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
    out[i] = F2B(B2F(sample[i]) + R(step * B2F(noise[i])));
  }
}

// ---- short attention (one block of 128 threads per query row and head)

constexpr int kSmallMaxRows = 1024;

__device__ __forceinline__ float Load(const std::uint16_t* p, bool f16) {
  return f16 ? __half2float(__ushort_as_half(*p)) : B2F(*p);
}
__device__ __forceinline__ float Load(const float* p, bool /*f16*/) { return *p; }

template <typename Q, bool kKvF16>
__global__ void SmallAttentionKernel(const Q* q, std::int64_t q_stride, const std::uint16_t* k,
                                     std::int64_t k_stride, const std::uint16_t* v,
                                     std::int64_t v_stride, std::uint16_t* out,
                                     std::int64_t out_stride, std::int64_t rows, std::int64_t group,
                                     bool causal, float scale) {
  __shared__ float qs[128];
  __shared__ float p[kSmallMaxRows];
  __shared__ float shared[32];
  const std::int64_t i = blockIdx.x;
  const std::int64_t h = blockIdx.y;
  const std::int64_t kvh = h / group;
  const int d = static_cast<int>(threadIdx.x);
  qs[d] = Load(q + i * q_stride + h * 128 + d, false);
  __syncthreads();
  const std::int64_t keys = causal ? i + 1 : rows;
  float local_max = -INFINITY;
  for (std::int64_t j = d; j < keys; j += 128) {
    const std::uint16_t* kr = k + j * k_stride + kvh * 128;
    float dot = 0.0f;
    for (int e = 0; e < 128; ++e) {
      dot += qs[e] * Load(kr + e, kKvF16);
    }
    p[j] = dot * scale;
    local_max = fmaxf(local_max, p[j]);
  }
  const float m = BlockMax(local_max, shared);
  float local_sum = 0.0f;
  for (std::int64_t j = d; j < keys; j += 128) {
    p[j] = expf(p[j] - m);
    local_sum += p[j];
  }
  const float total = BlockSum(local_sum, shared);
  __syncthreads();
  float acc = 0.0f;
  for (std::int64_t j = 0; j < keys; ++j) {
    acc += p[j] * Load(v + j * v_stride + kvh * 128 + d, kKvF16);
  }
  out[i * out_stride + h * 128 + d] = F2B(acc / total);
}

// ---- the VAE

__global__ void TransposeKernel(const std::uint16_t* x, std::uint16_t* out, std::int64_t rows,
                                std::int64_t cols) {
  const std::int64_t n = rows * cols;
  for (std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
    const std::int64_t c = i / rows;
    const std::int64_t r = i % rows;
    out[i] = x[r * cols + c];
  }
}

__global__ void ScaleShiftChannelsKernel(const std::uint16_t* z, const std::uint16_t* std_,
                                         const std::uint16_t* mean, std::uint16_t* out,
                                         std::int64_t channels, std::int64_t pixels) {
  const std::int64_t n = channels * pixels;
  for (std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
    const std::int64_t c = i / pixels;
    out[i] = F2B(R(B2F(z[i]) * B2F(std_[c])) + B2F(mean[c]));
  }
}

__global__ void ChannelRmsNormKernel(const std::uint16_t* x, const std::uint16_t* gamma,
                                     std::uint16_t* out, std::int64_t channels, std::int64_t pixels,
                                     float scale, bool silu) {
  const std::int64_t p = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
  if (p >= pixels) {
    return;
  }
  float sum = 0.0f;
  for (std::int64_t c = 0; c < channels; ++c) {
    const float v = B2F(x[c * pixels + p]);
    sum += v * v;
  }
  const float denom = fmaxf(sqrtf(sum), 1e-12f);
  for (std::int64_t c = 0; c < channels; ++c) {
    const float n = R(B2F(x[c * pixels + p]) / denom);
    float y = R(R(n * scale) * B2F(gamma[c]));
    if (silu) {
      y = SiluF(y);
    }
    out[c * pixels + p] = F2B(y);
  }
}

__global__ void Im2Col3x3Kernel(const std::uint16_t* x, std::int64_t channels, std::int64_t height,
                                std::int64_t width, std::int64_t pixel0, std::int64_t count,
                                std::uint16_t* col) {
  const std::int64_t n = channels * 9 * count;
  for (std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
    const std::int64_t j = i % count;
    const std::int64_t kk = i / count;
    const std::int64_t c = kk / 9;
    const std::int64_t ky = (kk % 9) / 3;
    const std::int64_t kx = kk % 3;
    const std::int64_t pixel = pixel0 + j;
    const std::int64_t y = pixel / width + ky - 1;
    const std::int64_t xx = pixel % width + kx - 1;
    col[i] = (y >= 0 && y < height && xx >= 0 && xx < width) ? x[(c * height + y) * width + xx]
                                                             : static_cast<std::uint16_t>(0);
  }
}

__global__ void FillBiasKernel(const std::uint16_t* bias, std::uint16_t* out, std::int64_t channels,
                               std::int64_t pixels) {
  const std::int64_t n = channels * pixels;
  for (std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
    out[i] = bias[i / pixels];
  }
}

__global__ void Upsample2xKernel(const std::uint16_t* x, std::uint16_t* out, std::int64_t channels,
                                 std::int64_t height, std::int64_t width) {
  const std::int64_t ow = width * 2;
  const std::int64_t oh = height * 2;
  const std::int64_t n = channels * oh * ow;
  for (std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
    const std::int64_t xx = i % ow;
    const std::int64_t y = (i / ow) % oh;
    const std::int64_t c = i / (ow * oh);
    out[i] = x[(c * height + y / 2) * width + xx / 2];
  }
}

__global__ void AddDupUpKernel(std::uint16_t* x, const std::uint16_t* shortcut,
                               std::int64_t out_channels, std::int64_t factor_t,
                               std::int64_t repeats, std::int64_t height, std::int64_t width) {
  const std::int64_t ow = width * 2;
  const std::int64_t oh = height * 2;
  const std::int64_t n = out_channels * oh * ow;
  for (std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
    const std::int64_t X = i % ow;
    const std::int64_t Y = (i / ow) % oh;
    const std::int64_t o = i / (ow * oh);
    const std::int64_t b = Y % 2;
    const std::int64_t d = X % 2;
    const std::int64_t rep = ((o * factor_t + factor_t - 1) * 2 + b) * 2 + d;
    const std::int64_t ci = rep / repeats;
    const float s = B2F(shortcut[(ci * height + Y / 2) * width + X / 2]);
    x[i] = F2B(B2F(x[i]) + s);
  }
}

__global__ void SoftmaxRowsKernel(const float* scores, std::uint16_t* probs, std::int64_t cols,
                                  float scale) {
  __shared__ float shared[32];
  const float* row = scores + blockIdx.x * cols;
  float m = -INFINITY;
  for (std::int64_t c = threadIdx.x; c < cols; c += blockDim.x) {
    m = fmaxf(m, row[c] * scale);
  }
  m = BlockMax(m, shared);
  float sum = 0.0f;
  for (std::int64_t c = threadIdx.x; c < cols; c += blockDim.x) {
    sum += expf(row[c] * scale - m);
  }
  sum = BlockSum(sum, shared);
  for (std::int64_t c = threadIdx.x; c < cols; c += blockDim.x) {
    probs[blockIdx.x * cols + c] = F2B(expf(row[c] * scale - m) / sum);
  }
}

bool Positive(std::int64_t v) { return v > 0 && v < kMaxGrid * 1024; }

// Every factor and their product Positive: the element counts the kernels
// index, refused before they could overflow.
bool Product(std::initializer_list<std::int64_t> factors) {
  std::int64_t n = 1;
  for (const std::int64_t f : factors) {
    if (!Positive(f) || __builtin_mul_overflow(n, f, &n)) {
      return false;
    }
  }
  return Positive(n);
}

// The vector loads' alignment (uint2, uint4, float4).
bool Aligned(const void* p, std::uintptr_t bytes) {
  return reinterpret_cast<std::uintptr_t>(p) % bytes == 0;
}

}  // namespace

Status Add(const Bf16* a, const Bf16* b, Bf16* out, std::int64_t n, Stream stream) {
  if (!Positive(n)) {
    return Refuse("Add: size");
  }
  AddKernel<<<Blocks(n, 256), 256, 0, S(stream)>>>(a, b, out, n);
  return Launched("Add");
}

Status EmbedRows(const Bf16* table, std::int64_t table_rows, const std::int32_t* ids,
                 std::int64_t rows, std::int64_t width, Bf16* out, std::int32_t* bad,
                 Stream stream) {
  if (!Product({rows, width}) || rows > 65535 * 32 || !Product({table_rows, width})) {
    return Refuse("EmbedRows: size");
  }
  EmbedRowsKernel<<<static_cast<unsigned>(rows), 256, 0, S(stream)>>>(table, table_rows, ids, width,
                                                                      out, bad);
  return Launched("EmbedRows");
}

Status Silu(const Bf16* x, Bf16* out, std::int64_t n, Stream stream) {
  if (!Positive(n)) {
    return Refuse("Silu: size");
  }
  SiluKernel<<<Blocks(n, 256), 256, 0, S(stream)>>>(x, out, n);
  return Launched("Silu");
}

Status GeluTanh(const Bf16* x, Bf16* out, std::int64_t n, Stream stream) {
  if (!Positive(n)) {
    return Refuse("GeluTanh: size");
  }
  GeluTanhKernel<<<Blocks(n, 256), 256, 0, S(stream)>>>(x, out, n);
  return Launched("GeluTanh");
}

Status SwiGlu(const Bf16* gate, std::int64_t gate_stride, const Bf16* up, std::int64_t up_stride,
              Bf16* out, std::int64_t rows, std::int64_t width, Stream stream) {
  if (!Product({rows, width}) || gate_stride < width || up_stride < width) {
    return Refuse("SwiGlu: size");
  }
  if (width % 8 == 0 && gate_stride % 8 == 0 && up_stride % 8 == 0 && Aligned(gate, 16) &&
      Aligned(up, 16) && Aligned(out, 16)) {
    SwiGlu8Kernel<<<Blocks(rows * width / 8, 256), 256, 0, S(stream)>>>(
        gate, gate_stride, up, up_stride, out, rows, width / 8);
    return Launched("SwiGlu");
  }
  SwiGluKernel<<<Blocks(rows * width, 256), 256, 0, S(stream)>>>(gate, gate_stride, up, up_stride,
                                                                 out, rows, width);
  return Launched("SwiGlu");
}

Status Bf16ToF16(const Bf16* x, F16* out, std::int64_t n, Stream stream) {
  if (!Positive(n)) {
    return Refuse("Bf16ToF16: size");
  }
  Bf16ToF16Kernel<<<Blocks(n, 256), 256, 0, S(stream)>>>(x, out, n);
  return Launched("Bf16ToF16");
}

Status F32ToBf16(const float* x, Bf16* out, std::int64_t n, Stream stream) {
  if (!Positive(n)) {
    return Refuse("F32ToBf16: size");
  }
  F32ToBf16Kernel<<<Blocks(n, 256), 256, 0, S(stream)>>>(x, out, n);
  return Launched("F32ToBf16");
}

Status Bf16ToF32(const Bf16* x, float* out, std::int64_t n, Stream stream) {
  if (!Positive(n)) {
    return Refuse("Bf16ToF32: size");
  }
  Bf16ToF32Kernel<<<Blocks(n, 256), 256, 0, S(stream)>>>(x, out, n);
  return Launched("Bf16ToF32");
}

Status RmsNorm(const Bf16* x, const Bf16* w, Bf16* out, std::int64_t rows, std::int64_t width,
               float eps, Stream stream) {
  if (!Product({rows, width}) || rows > kMaxGrid) {
    return Refuse("RmsNorm: size");
  }
  RmsNormKernel<<<static_cast<unsigned>(rows), 512, 0, S(stream)>>>(x, w, out, width, eps);
  return Launched("RmsNorm");
}

Status HeadNormRopeNeox(Bf16* x, std::int64_t stride, std::int64_t rows, std::int64_t heads,
                        const Bf16* w, const Bf16* cos, const Bf16* sin, float eps, Stream stream) {
  if (!Product({rows, heads, 32}) || stride < heads * 128) {
    return Refuse("HeadNormRopeNeox: size");
  }
  const std::int64_t threads = rows * heads * 32;
  HeadNormRopeNeoxKernel<<<Blocks(threads, 256), 256, 0, S(stream)>>>(x, stride, rows, heads, w,
                                                                      cos, sin, eps);
  return Launched("HeadNormRopeNeox");
}

Status SmallAttention(const Bf16* q, std::int64_t q_stride, const Bf16* k, std::int64_t k_stride,
                      const Bf16* v, std::int64_t v_stride, Bf16* out, std::int64_t out_stride,
                      std::int64_t rows, std::int64_t q_heads, std::int64_t kv_heads, bool causal,
                      float scale, Stream stream) {
  if (!Positive(rows) || rows > kSmallMaxRows || !Positive(q_heads) || q_heads > 65535 ||
      !Positive(kv_heads) || q_heads % kv_heads != 0 || q_stride < q_heads * 128 ||
      k_stride < kv_heads * 128 || v_stride < kv_heads * 128 || out_stride < q_heads * 128) {
    return Refuse("SmallAttention: size");
  }
  const dim3 grid(static_cast<unsigned>(rows), static_cast<unsigned>(q_heads));
  SmallAttentionKernel<std::uint16_t, false>
      <<<grid, 128, 0, S(stream)>>>(q, q_stride, k, k_stride, v, v_stride, out, out_stride, rows,
                                    q_heads / kv_heads, causal, scale);
  return Launched("SmallAttention");
}

Status SmallAttentionF32F16(const float* q, std::int64_t q_stride, const F16* k,
                            std::int64_t k_stride, const F16* v, std::int64_t v_stride, Bf16* out,
                            std::int64_t out_stride, std::int64_t rows, std::int64_t q_heads,
                            std::int64_t kv_heads, bool causal, float scale, Stream stream) {
  if (!Positive(rows) || rows > kSmallMaxRows || !Positive(q_heads) || q_heads > 65535 ||
      !Positive(kv_heads) || q_heads % kv_heads != 0 || q_stride < q_heads * 128 ||
      k_stride < kv_heads * 128 || v_stride < kv_heads * 128 || out_stride < q_heads * 128) {
    return Refuse("SmallAttentionF32F16: size");
  }
  const dim3 grid(static_cast<unsigned>(rows), static_cast<unsigned>(q_heads));
  SmallAttentionKernel<float, true><<<grid, 128, 0, S(stream)>>>(q, q_stride, k, k_stride, v,
                                                                 v_stride, out, out_stride, rows,
                                                                 q_heads / kv_heads, causal, scale);
  return Launched("SmallAttentionF32F16");
}

Status ZeroCenterRmsNorm(const Bf16* x, const Bf16* w, Bf16* out, std::int64_t rows,
                         std::int64_t width, float eps, Stream stream) {
  if (!Product({rows, width}) || rows > kMaxGrid) {
    return Refuse("ZeroCenterRmsNorm: size");
  }
  ZeroCenterRmsNormKernel<<<static_cast<unsigned>(rows), 512, 0, S(stream)>>>(x, w, out, width,
                                                                              eps);
  return Launched("ZeroCenterRmsNorm");
}

Status LayerNormModulate(const Bf16* x, Bf16* out, std::int64_t rows, std::int64_t width, float eps,
                         const Bf16* mod, std::int64_t mod_stride, std::int64_t first_target,
                         Stream stream) {
  if (!Product({rows, width}) || rows > kMaxGrid || width % 8 != 0 ||
      width > std::int64_t{kLnThreads} * 8 * kLnChunks || mod_stride < width ||
      mod_stride % 8 != 0 || !Aligned(x, 16) || !Aligned(out, 16) || !Aligned(mod, 16)) {
    return Refuse("LayerNormModulate: width a multiple of 8 up to 8,192, 16-byte aligned rows");
  }
  // x is only read without the residual.
  LaunchLayerNorm<false>(const_cast<Bf16*>(x), out, rows, width, eps, mod, mod_stride, first_target,
                         nullptr, 0, nullptr, 0, S(stream));
  return Launched("LayerNormModulate");
}

Status GatedResidualNorm(Bf16* x, const Bf16* y, std::int64_t y_stride, std::int64_t rows,
                         std::int64_t width, const Bf16* gate_mod, std::int64_t gate_stride,
                         std::int64_t first_target, Bf16* out, float eps, const Bf16* scale_mod,
                         std::int64_t scale_stride, Stream stream) {
  if (!Product({rows, width}) || rows > kMaxGrid || width % 8 != 0 ||
      width > std::int64_t{kLnThreads} * 8 * kLnChunks || y_stride < width || y_stride % 8 != 0 ||
      gate_stride < width || gate_stride % 8 != 0 || scale_stride < width ||
      scale_stride % 8 != 0 || !Aligned(x, 16) || !Aligned(y, 16) || !Aligned(out, 16) ||
      !Aligned(gate_mod, 16) || !Aligned(scale_mod, 16)) {
    return Refuse("GatedResidualNorm: width a multiple of 8 up to 8,192, 16-byte aligned rows");
  }
  LaunchLayerNorm<true>(x, out, rows, width, eps, scale_mod, scale_stride, first_target, y,
                        y_stride, gate_mod, gate_stride, S(stream));
  return Launched("GatedResidualNorm");
}

Status HeadNormRopeComplexF32(const Bf16* x, std::int64_t x_stride, std::int64_t rows,
                              std::int64_t heads, const Bf16* w, const float* freqs, float eps,
                              float* out, Stream stream) {
  if (!Product({rows, heads, 128}) || x_stride < heads * 128 || x_stride % 4 != 0 ||
      !Aligned(x, 8) || !Aligned(freqs, 16) || !Aligned(out, 16)) {
    return Refuse("HeadNormRopeComplex: sizes, strides or alignment");
  }
  HeadNormRopeComplexKernel<0><<<Blocks(rows * heads * 32, 256), 256, 0, S(stream)>>>(
      x, x_stride, rows, heads, w, freqs, eps, out);
  return Launched("HeadNormRopeComplexF32");
}

Status HeadNormRopeComplex(Bf16* x, std::int64_t x_stride, std::int64_t rows, std::int64_t heads,
                           const Bf16* w, const float* freqs, float eps, Stream stream) {
  if (!Product({rows, heads, 128}) || x_stride < heads * 128 || x_stride % 4 != 0 ||
      !Aligned(x, 8) || !Aligned(freqs, 16)) {
    return Refuse("HeadNormRopeComplex: sizes, strides or alignment");
  }
  HeadNormRopeComplexKernel<2><<<Blocks(rows * heads * 32, 256), 256, 0, S(stream)>>>(
      x, x_stride, rows, heads, w, freqs, eps, nullptr);
  return Launched("HeadNormRopeComplex");
}

Status HeadNormRopeComplexF16(const Bf16* x, std::int64_t x_stride, std::int64_t rows,
                              std::int64_t heads, const Bf16* w, const float* freqs, float eps,
                              F16* out, Stream stream) {
  if (!Product({rows, heads, 128}) || x_stride < heads * 128 || x_stride % 4 != 0 ||
      !Aligned(x, 8) || !Aligned(freqs, 16)) {
    return Refuse("HeadNormRopeComplex: sizes, strides or alignment");
  }
  HeadNormRopeComplexKernel<1><<<Blocks(rows * heads * 32, 256), 256, 0, S(stream)>>>(
      x, x_stride, rows, heads, w, freqs, eps, out);
  return Launched("HeadNormRopeComplexF16");
}

Status Bf16RowsToF16(const Bf16* x, std::int64_t x_stride, F16* out, std::int64_t rows,
                     std::int64_t width, Stream stream) {
  if (!Product({rows, width}) || x_stride < width) {
    return Refuse("Bf16RowsToF16: size");
  }
  Bf16RowsToF16Kernel<<<Blocks(rows * width, 256), 256, 0, S(stream)>>>(x, x_stride, out, rows,
                                                                        width);
  return Launched("Bf16RowsToF16");
}

Status GatedResidual(Bf16* x, const Bf16* y, std::int64_t y_stride, std::int64_t rows,
                     std::int64_t width, const Bf16* mod, std::int64_t mod_stride,
                     std::int64_t first_target, Stream stream) {
  if (!Product({rows, width}) || width % 8 != 0 || y_stride % 8 != 0 || y_stride < width ||
      mod_stride < width || mod_stride % 8 != 0 || !Aligned(x, 16) || !Aligned(y, 16) ||
      !Aligned(mod, 16)) {
    return Refuse("GatedResidual: sizes, strides or alignment");
  }
  GatedResidualKernel<<<Blocks(rows * width / 8, 256), 256, 0, S(stream)>>>(
      x, y, y_stride, rows, width, mod, mod_stride, first_target);
  return Launched("GatedResidual");
}

float EulerStepDt(float dt, bool dt_bf16) {
  float step = dt;
  if (dt_bf16) {
    // Host rounding, as F2B does.
    std::uint32_t u = 0;
    std::memcpy(&u, &step, sizeof u);
    u += 0x7fffU + ((u >> 16U) & 1U);
    u &= 0xffff0000U;
    std::memcpy(&step, &u, sizeof u);
  }
  return step;
}

Status EulerStep(const Bf16* sample, const Bf16* noise, Bf16* out, float dt, bool dt_bf16,
                 std::int64_t n, Stream stream) {
  if (!Positive(n)) {
    return Refuse("EulerStep: size");
  }
  EulerStepKernel<<<Blocks(n, 256), 256, 0, S(stream)>>>(sample, noise, out,
                                                         EulerStepDt(dt, dt_bf16), n);
  return Launched("EulerStep");
}

Status EulerStepAt(const Bf16* sample, const Bf16* noise, Bf16* out, const float* dt,
                   std::int64_t n, Stream stream) {
  if (!Positive(n) || !Aligned(dt, 4)) {
    return Refuse("EulerStepAt: size");
  }
  EulerStepAtKernel<<<Blocks(n, 256), 256, 0, S(stream)>>>(sample, noise, out, dt, n);
  return Launched("EulerStepAt");
}

Status Transpose(const Bf16* x, Bf16* out, std::int64_t rows, std::int64_t cols, Stream stream) {
  if (!Product({rows, cols})) {
    return Refuse("Transpose: size");
  }
  TransposeKernel<<<Blocks(rows * cols, 256), 256, 0, S(stream)>>>(x, out, rows, cols);
  return Launched("Transpose");
}

Status ScaleShiftChannels(const Bf16* z, const Bf16* std, const Bf16* mean, Bf16* out,
                          std::int64_t channels, std::int64_t pixels, Stream stream) {
  if (!Product({channels, pixels})) {
    return Refuse("ScaleShiftChannels: size");
  }
  ScaleShiftChannelsKernel<<<Blocks(channels * pixels, 256), 256, 0, S(stream)>>>(z, std, mean, out,
                                                                                  channels, pixels);
  return Launched("ScaleShiftChannels");
}

Status ChannelRmsNorm(const Bf16* x, const Bf16* gamma, Bf16* out, std::int64_t channels,
                      std::int64_t pixels, bool silu, Stream stream) {
  if (!Product({channels, pixels})) {
    return Refuse("ChannelRmsNorm: size");
  }
  const auto scale = static_cast<float>(std::sqrt(static_cast<double>(channels)));
  ChannelRmsNormKernel<<<Blocks(pixels, 256), 256, 0, S(stream)>>>(x, gamma, out, channels, pixels,
                                                                   scale, silu);
  return Launched("ChannelRmsNorm");
}

Status Im2Col3x3(const Bf16* x, std::int64_t channels, std::int64_t height, std::int64_t width,
                 std::int64_t pixel0, std::int64_t count, Bf16* col, Stream stream) {
  if (!Product({channels, height, width}) || !Product({channels, 9, count}) || pixel0 < 0 ||
      pixel0 + count > height * width) {
    return Refuse("Im2Col3x3: size");
  }
  Im2Col3x3Kernel<<<Blocks(channels * 9 * count, 256), 256, 0, S(stream)>>>(
      x, channels, height, width, pixel0, count, col);
  return Launched("Im2Col3x3");
}

Status FillBias(const Bf16* bias, Bf16* out, std::int64_t channels, std::int64_t pixels,
                Stream stream) {
  if (!Product({channels, pixels})) {
    return Refuse("FillBias: size");
  }
  FillBiasKernel<<<Blocks(channels * pixels, 256), 256, 0, S(stream)>>>(bias, out, channels,
                                                                        pixels);
  return Launched("FillBias");
}

Status Upsample2x(const Bf16* x, Bf16* out, std::int64_t channels, std::int64_t height,
                  std::int64_t width, Stream stream) {
  if (!Product({channels, height, width, 4})) {
    return Refuse("Upsample2x: size");
  }
  Upsample2xKernel<<<Blocks(channels * height * width * 4, 256), 256, 0, S(stream)>>>(
      x, out, channels, height, width);
  return Launched("Upsample2x");
}

Status AddDupUp(Bf16* x, const Bf16* shortcut, std::int64_t in_channels, std::int64_t out_channels,
                std::int64_t factor_t, std::int64_t height, std::int64_t width, Stream stream) {
  if (!Product({in_channels, height, width}) || !Product({out_channels, height, width, 8}) ||
      (factor_t != 1 && factor_t != 2) || (out_channels * factor_t * 4) % in_channels != 0) {
    return Refuse("AddDupUp: size");
  }
  const std::int64_t repeats = out_channels * factor_t * 4 / in_channels;
  AddDupUpKernel<<<Blocks(out_channels * height * width * 4, 256), 256, 0, S(stream)>>>(
      x, shortcut, out_channels, factor_t, repeats, height, width);
  return Launched("AddDupUp");
}

Status SoftmaxRowsToBf16(const float* scores, Bf16* probs, std::int64_t rows, std::int64_t cols,
                         float scale, Stream stream) {
  if (!Product({rows, cols}) || rows > kMaxGrid) {
    return Refuse("SoftmaxRowsToBf16: size");
  }
  SoftmaxRowsKernel<<<static_cast<unsigned>(rows), 512, 0, S(stream)>>>(scores, probs, cols, scale);
  return Launched("SoftmaxRowsToBf16");
}

}  // namespace llmp::kernels::image
