// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Owned MXFP8 GEMV arithmetic from src/kernels/ggml/llmp_ops.cu. Only
// scheduling changes: rows/warp, warps/block and the input/weight load order.

#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <cstdint>

#include "common.cuh"
#include "qwen38_mxfp8_tune_kernels.h"

namespace llmp::diag {
namespace {

__device__ __forceinline__ float E8m0(std::uint8_t e) {
  if (e == 0) return __uint_as_float(0x00400000u);
  return e == 255 ? __uint_as_float(0x7fc00000u) : __uint_as_float(std::uint32_t{e} << 23);
}
__device__ __forceinline__ float2 E4m3x2(std::uint16_t pair) {
  const __half2_raw raw = __nv_cvt_fp8x2_to_halfraw2(pair, __NV_E4M3);
  return __half22float2(__half2(raw));
}
__device__ __forceinline__ void Decode16(const uint4 q, float out[16]) {
  const std::uint32_t words[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
  for (int w = 0; w < 4; ++w) {
    const float2 lo = E4m3x2(static_cast<std::uint16_t>(words[w] & 0xffffu));
    const float2 hi = E4m3x2(static_cast<std::uint16_t>(words[w] >> 16));
    out[4 * w + 0] = lo.x;
    out[4 * w + 1] = lo.y;
    out[4 * w + 2] = hi.x;
    out[4 * w + 3] = hi.y;
  }
}
__device__ __forceinline__ void Load16(const float* input, float values[16]) {
  const auto* vectors = reinterpret_cast<const float4*>(input);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float4 a = vectors[i];
    values[4 * i + 0] = a.x;
    values[4 * i + 1] = a.y;
    values[4 * i + 2] = a.z;
    values[4 * i + 3] = a.w;
  }
}
__device__ __forceinline__ float Dot16(const float weights[16], const float input[16]) {
  float dot = 0.0f;
#pragma unroll
  for (int i = 0; i < 16; ++i) dot = fmaf(weights[i], input[i], dot);
  return dot;
}

template <int kColumns, int kRows, int kWarps, bool kColumnAtATime>
__global__ void __launch_bounds__(kWarps * 32)
    Mxfp8Gemv(const std::uint8_t* __restrict__ codes, const std::uint8_t* __restrict__ scales,
              const float* x, float* y, int n, int k, int x_stride) {
  const int lane = static_cast<int>(threadIdx.x) % 32;
  const int row0 =
      ((static_cast<int>(blockIdx.x) * kWarps) + (static_cast<int>(threadIdx.x) / 32)) * kRows;
  if (row0 >= n) return;
  const int rows = min(kRows, n - row0);
#pragma unroll
  for (int r = 0; r < kRows; ++r) {
    const std::uint8_t* w = codes + (static_cast<std::int64_t>(row0 + min(r, rows - 1)) * k);
    for (int v = lane; v < min(k / 16, 64); v += 32) {
      asm volatile("prefetch.global.L2 [%0];" ::"l"(w + (static_cast<std::int64_t>(v) * 16)));
    }
  }
  ggml_cuda_pdl_sync();
  float sum[kRows][kColumns];
#pragma unroll
  for (int r = 0; r < kRows; ++r) {
#pragma unroll
    for (int c = 0; c < kColumns; ++c) sum[r][c] = 0.0f;
  }
  for (int v = lane; v < k / 16; v += 32) {
    if constexpr (kColumnAtATime) {
      float wf[kRows][16];
      float scale[kRows];
#pragma unroll
      for (int r = 0; r < kRows; ++r) {
        const int row = row0 + min(r, rows - 1);
        const std::uint8_t* w = codes + static_cast<std::int64_t>(row) * k;
        const std::uint8_t* s = scales + static_cast<std::int64_t>(row) * (k / 32);
        Decode16(*reinterpret_cast<const uint4*>(w + static_cast<std::int64_t>(v) * 16), wf[r]);
        scale[r] = E8m0(s[v / 2]);
      }
#pragma unroll
      for (int c = 0; c < kColumns; ++c) {
        float xs[16];
        Load16(x + static_cast<std::int64_t>(c) * x_stride + v * 16, xs);
#pragma unroll
        for (int r = 0; r < kRows; ++r) {
          sum[r][c] = fmaf(Dot16(wf[r], xs), scale[r], sum[r][c]);
        }
      }
    } else {
      float xs[kColumns][16];
#pragma unroll
      for (int c = 0; c < kColumns; ++c) {
        Load16(x + static_cast<std::int64_t>(c) * x_stride + v * 16, xs[c]);
      }
#pragma unroll
      for (int r = 0; r < kRows; ++r) {
        const int row = row0 + min(r, rows - 1);
        const std::uint8_t* w = codes + static_cast<std::int64_t>(row) * k;
        const std::uint8_t* s = scales + static_cast<std::int64_t>(row) * (k / 32);
        float wf[16];
        Decode16(*reinterpret_cast<const uint4*>(w + static_cast<std::int64_t>(v) * 16), wf);
        const float scale = E8m0(s[v / 2]);
#pragma unroll
        for (int c = 0; c < kColumns; ++c) {
          sum[r][c] = fmaf(Dot16(wf, xs[c]), scale, sum[r][c]);
        }
      }
    }
  }
  ggml_cuda_pdl_lc();
#pragma unroll
  for (int r = 0; r < kRows; ++r) {
#pragma unroll
    for (int c = 0; c < kColumns; ++c) {
#pragma unroll
      for (int offset = 16; offset > 0; offset /= 2) {
        sum[r][c] += __shfl_xor_sync(0xffffffffu, sum[r][c], offset);
      }
      if (lane == 0 && r < rows) y[static_cast<std::int64_t>(c) * n + row0 + r] = sum[r][c];
    }
  }
}

template <int kColumns, int kRows, int kWarps>
void Launch(cudaStream_t stream, const std::uint8_t* codes, const std::uint8_t* scales,
            const float* x, float* y, int n, int k, int stride, bool column_at_a_time) {
  const auto grid = dim3(static_cast<unsigned>((n + kWarps * kRows - 1) / (kWarps * kRows)));
  const auto parameters = ggml_cuda_kernel_launch_params(grid, dim3(kWarps * 32), 0, stream);
  if (column_at_a_time) {
    ggml_cuda_kernel_launch(Mxfp8Gemv<kColumns, kRows, kWarps, true>, parameters, codes, scales, x,
                            y, n, k, stride);
  } else {
    ggml_cuda_kernel_launch(Mxfp8Gemv<kColumns, kRows, kWarps, false>, parameters, codes, scales, x,
                            y, n, k, stride);
  }
}
template <int kColumns, int kRows>
void Warps(cudaStream_t stream, const std::uint8_t* codes, const std::uint8_t* scales,
           const float* x, float* y, int n, int k, int stride, Mxfp8Schedule schedule) {
  if (schedule.warps == 4) {
    Launch<kColumns, kRows, 4>(stream, codes, scales, x, y, n, k, stride,
                               schedule.column_at_a_time);
  } else {
    Launch<kColumns, kRows, 8>(stream, codes, scales, x, y, n, k, stride,
                               schedule.column_at_a_time);
  }
}
template <int kColumns>
void Rows(cudaStream_t stream, const std::uint8_t* codes, const std::uint8_t* scales,
          const float* x, float* y, int n, int k, int stride, Mxfp8Schedule schedule) {
  switch (schedule.rows) {
    case 1:
      Warps<kColumns, 1>(stream, codes, scales, x, y, n, k, stride, schedule);
      break;
    case 2:
      Warps<kColumns, 2>(stream, codes, scales, x, y, n, k, stride, schedule);
      break;
    default:
      Warps<kColumns, 4>(stream, codes, scales, x, y, n, k, stride, schedule);
      break;
  }
}

}  // namespace

cudaError_t Mxfp8Tune(cudaStream_t stream, const std::uint8_t* codes, const std::uint8_t* scales,
                      const float* input, float* output, int columns, int outputs, int inputs,
                      int input_stride, Mxfp8Schedule schedule) {
  if (codes == nullptr || scales == nullptr || input == nullptr || output == nullptr ||
      columns < 1 || columns > 8 || outputs < 1 || outputs > 262144 || inputs < 32 ||
      inputs > 20480 || inputs % 32 != 0 || input_stride < inputs || input_stride > inputs + 4096 ||
      input_stride % 4 != 0 || reinterpret_cast<std::uintptr_t>(codes) % 16 != 0 ||
      reinterpret_cast<std::uintptr_t>(input) % 16 != 0 ||
      (schedule.rows != 1 && schedule.rows != 2 && schedule.rows != 4) ||
      (schedule.warps != 4 && schedule.warps != 8))
    return cudaErrorInvalidValue;
  switch (columns) {
    case 1:
      Rows<1>(stream, codes, scales, input, output, outputs, inputs, input_stride, schedule);
      break;
    case 2:
      Rows<2>(stream, codes, scales, input, output, outputs, inputs, input_stride, schedule);
      break;
    case 3:
      Rows<3>(stream, codes, scales, input, output, outputs, inputs, input_stride, schedule);
      break;
    case 4:
      Rows<4>(stream, codes, scales, input, output, outputs, inputs, input_stride, schedule);
      break;
    case 5:
      Rows<5>(stream, codes, scales, input, output, outputs, inputs, input_stride, schedule);
      break;
    case 6:
      Rows<6>(stream, codes, scales, input, output, outputs, inputs, input_stride, schedule);
      break;
    case 7:
      Rows<7>(stream, codes, scales, input, output, outputs, inputs, input_stride, schedule);
      break;
    default:
      Rows<8>(stream, codes, scales, input, output, outputs, inputs, input_stride, schedule);
      break;
  }
  return cudaGetLastError();
}

}  // namespace llmp::diag
