// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "vmm_diag_kernels.h"

namespace llmp::diag {
namespace {

constexpr int kThreads = 256;

__device__ std::uint32_t Mix(std::uint64_t index, std::uint32_t seed) {
  auto x = static_cast<std::uint32_t>(index) ^ static_cast<std::uint32_t>(index >> 32U) ^
           (seed * 0x9e3779b9U);
  x ^= x >> 16U;
  x *= 0x7feb352dU;
  x ^= x >> 15U;
  x *= 0x846ca68bU;
  return x ^ (x >> 16U);
}

__device__ std::uint64_t Thread() {
  return (static_cast<std::uint64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
}

__device__ std::uint64_t Stride() { return static_cast<std::uint64_t>(gridDim.x) * blockDim.x; }

__global__ void FillKernel(std::uint32_t* data, std::uint64_t words, std::uint32_t seed) {
  for (std::uint64_t i = Thread(); i < words; i += Stride()) {
    data[i] = Mix(i, seed);
  }
}

__global__ void FillHalfKernel(__half* data, std::uint64_t count, std::uint32_t seed) {
  for (std::uint64_t i = Thread(); i < count; i += Stride()) {
    const float unit = static_cast<float>(Mix(i, seed) >> 8U) / 16777216.0f;
    data[i] = __float2half((unit * 0.2f) - 0.1f);
  }
}

__global__ void FillFloatKernel(float* data, std::uint64_t count, std::uint32_t seed) {
  for (std::uint64_t i = Thread(); i < count; i += Stride()) {
    const float unit = static_cast<float>(Mix(i, seed) >> 8U) / 16777216.0f;
    data[i] = (unit * 2.0f) - 1.0f;
  }
}

// The I/O experiment's scan, unchanged but for names.
__global__ void ScanWordsKernel(const std::uint32_t* data, std::uint64_t words,
                                std::uint32_t* out) {
  std::uint32_t sum = 0;
  for (std::uint64_t i = Thread(); i < words; i += Stride()) {
    sum += data[i];
  }
  out[Thread()] = sum;
}

__global__ void ScanVectorKernel(const uint4* data, std::uint64_t count, std::uint32_t* out) {
  std::uint32_t sum = 0;
  for (std::uint64_t i = Thread(); i < count; i += Stride()) {
    const uint4 v = data[i];
    sum += v.x ^ v.y ^ v.z ^ v.w;
  }
  out[Thread()] = sum;
}

__global__ void WriteVectorKernel(uint4* data, std::uint64_t count) {
  for (std::uint64_t i = Thread(); i < count; i += Stride()) {
    const auto x = static_cast<std::uint32_t>(i);
    data[i] = make_uint4(x, x + 1U, x + 2U, x + 3U);
  }
}

__global__ void WriteSparseKernel(std::uint32_t* data, std::uint64_t count,
                                  std::uint64_t stride_words) {
  for (std::uint64_t i = Thread(); i < count; i += Stride()) {
    data[i * stride_words] = static_cast<std::uint32_t>(i);
  }
}

__global__ void WritePerBlockKernel(std::uint32_t* data) {
  if (threadIdx.x == 0) {
    data[blockIdx.x] = blockIdx.x;
  }
}

__global__ void RereadKernel(const uint4* data, std::uint64_t count, int passes,
                             std::uint32_t* out) {
  std::uint32_t sum = 0;
  const std::uint64_t stride = Stride();
  for (int pass = 0; pass < passes; ++pass) {
    // Shift each pass by a prime number of blocks' worth of elements.
    const std::uint64_t shift = (static_cast<std::uint64_t>(pass) * 7919U * kThreads) % count;
    for (std::uint64_t i = Thread(); i < count; i += stride) {
      std::uint64_t at = i + shift;
      if (at >= count) {
        at -= count;
      }
      const uint4 v = __ldcg(data + at);
      sum += v.x ^ v.y ^ v.z ^ v.w;
    }
  }
  out[Thread()] = sum;
}

__global__ void RandomLinesKernel(const std::uint32_t* data, std::uint64_t lines,
                                  std::uint64_t window_lines, int run, int per_warp,
                                  std::uint32_t* out) {
  const std::uint64_t warp = Thread() / 32U;
  const std::uint32_t lane = threadIdx.x % 32U;
  std::uint32_t sum = 0;
  std::uint64_t region = 0;
  for (int j = 0; j < per_warp; ++j) {
    std::uint64_t line = 0;
    const std::uint64_t r =
        (static_cast<std::uint64_t>(Mix((warp << 20U) + static_cast<std::uint64_t>(j), 17U))
         << 32U) |
        Mix((warp << 20U) + static_cast<std::uint64_t>(j), 91U);
    if (window_lines == 0) {
      line = r % lines;
    } else {
      if (j % run == 0) {
        region = (r >> 17U) % (lines / window_lines);
      }
      line = (region * window_lines) + (r % window_lines);
    }
    sum += data[(line * 32U) + lane];
  }
  out[Thread()] = sum;
}

__global__ void CopyVectorKernel(uint4* to, const uint4* from, std::uint64_t count) {
  for (std::uint64_t i = Thread(); i < count; i += Stride()) {
    to[i] = from[i];
  }
}

template <typename T>
T* At(std::uint64_t address) {
  return reinterpret_cast<T*>(address);  // NOLINT(performance-no-int-to-ptr)
}

dim3 Grid(int blocks) { return {static_cast<unsigned>(blocks), 1U, 1U}; }

}  // namespace

cudaError_t Fill(cudaStream_t stream, std::uint64_t data, std::uint64_t words, std::uint32_t seed) {
  FillKernel<<<Grid(1024), kThreads, 0, stream>>>(At<std::uint32_t>(data), words, seed);
  return cudaGetLastError();
}

cudaError_t FillHalf(cudaStream_t stream, std::uint64_t data, std::uint64_t count,
                     std::uint32_t seed) {
  FillHalfKernel<<<Grid(1024), kThreads, 0, stream>>>(At<__half>(data), count, seed);
  return cudaGetLastError();
}

cudaError_t FillFloat(cudaStream_t stream, std::uint64_t data, std::uint64_t count,
                      std::uint32_t seed) {
  FillFloatKernel<<<Grid(1024), kThreads, 0, stream>>>(At<float>(data), count, seed);
  return cudaGetLastError();
}

cudaError_t ScanWords(cudaStream_t stream, std::uint64_t data, std::uint64_t bytes,
                      std::uint64_t out) {
  ScanWordsKernel<<<Grid(256), kThreads, 0, stream>>>(At<const std::uint32_t>(data), bytes / 4U,
                                                      At<std::uint32_t>(out));
  return cudaGetLastError();
}

cudaError_t ScanVector(cudaStream_t stream, std::uint64_t data, std::uint64_t bytes,
                       std::uint64_t out, int blocks) {
  ScanVectorKernel<<<Grid(blocks), kThreads, 0, stream>>>(At<const uint4>(data), bytes / 16U,
                                                          At<std::uint32_t>(out));
  return cudaGetLastError();
}

cudaError_t WriteVector(cudaStream_t stream, std::uint64_t data, std::uint64_t bytes, int blocks) {
  WriteVectorKernel<<<Grid(blocks), kThreads, 0, stream>>>(At<uint4>(data), bytes / 16U);
  return cudaGetLastError();
}

cudaError_t WriteSparse(cudaStream_t stream, std::uint64_t data, std::uint64_t count,
                        std::uint64_t stride, int blocks) {
  WriteSparseKernel<<<Grid(blocks), kThreads, 0, stream>>>(At<std::uint32_t>(data), count,
                                                           stride / 4U);
  return cudaGetLastError();
}

cudaError_t WritePerBlock(cudaStream_t stream, std::uint64_t data, std::uint64_t count) {
  WritePerBlockKernel<<<Grid(static_cast<int>(count)), kThreads, 0, stream>>>(
      At<std::uint32_t>(data));
  return cudaGetLastError();
}

cudaError_t Reread(cudaStream_t stream, std::uint64_t data, std::uint64_t bytes, int passes,
                   std::uint64_t out, int blocks) {
  RereadKernel<<<Grid(blocks), kThreads, 0, stream>>>(At<const uint4>(data), bytes / 16U, passes,
                                                      At<std::uint32_t>(out));
  return cudaGetLastError();
}

cudaError_t RandomLines(cudaStream_t stream, std::uint64_t data, std::uint64_t bytes,
                        std::uint64_t window, int run, int lines_per_warp, std::uint64_t out,
                        int blocks) {
  RandomLinesKernel<<<Grid(blocks), kThreads, 0, stream>>>(At<const std::uint32_t>(data),
                                                           bytes / 128U, window / 128U, run,
                                                           lines_per_warp, At<std::uint32_t>(out));
  return cudaGetLastError();
}

cudaError_t CopyVector(cudaStream_t stream, std::uint64_t to, std::uint64_t from,
                       std::uint64_t bytes, int blocks) {
  CopyVectorKernel<<<Grid(blocks), kThreads, 0, stream>>>(At<uint4>(to), At<const uint4>(from),
                                                          bytes / 16U);
  return cudaGetLastError();
}

}  // namespace llmp::diag
