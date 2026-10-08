// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#include <cuda_runtime.h>
__device__ unsigned pattern(unsigned long long index) {
  unsigned x = static_cast<unsigned>(index) ^ static_cast<unsigned>(index >> 32) ^ 0x9e3779b9U;
  x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; return x ^ (x >> 16);
}
extern "C" __global__ void verify(const unsigned* data, unsigned long long words,
                                 unsigned long long first, unsigned* errors) {
  unsigned bad = 0;
  for (auto i = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < words; i += static_cast<unsigned long long>(gridDim.x) * blockDim.x)
    bad += data[i] != pattern(first + i);
  if (bad) atomicAdd(errors, bad);
}
extern "C" __global__ void fill(unsigned* data, unsigned long long words,
                               unsigned long long first) {
  for (auto i = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < words; i += static_cast<unsigned long long>(gridDim.x) * blockDim.x) data[i] = pattern(first + i);
}
// Fixed work, not duration-controlled: its rate can expose compute interference.
extern "C" __global__ void compute(float* output, int rounds) {
  float a = 1.0f + threadIdx.x, b = 2.0f + threadIdx.x;
  for (int i = 0; i < rounds; ++i) {
    a = fmaf(a, 0.99999f, 0.00001f); b = fmaf(b, 0.99998f, 0.00002f);
  }
  output[blockIdx.x * blockDim.x + threadIdx.x] = a + b;
}
extern "C" __global__ void scan(const unsigned* data, unsigned long long words,
                               unsigned* output) {
  unsigned sum = 0;
  for (auto i = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < words; i += static_cast<unsigned long long>(gridDim.x) * blockDim.x) sum += data[i];
  output[blockIdx.x * blockDim.x + threadIdx.x] = sum;
}
