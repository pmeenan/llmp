// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#include <cuda_runtime.h>

__device__ unsigned long long timer_ns() {
  unsigned long long value;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
  return value;
}
// Bounded independent compute, with a system-scope start handshake.
extern "C" __global__ void busy(int* started, float* output,
                                unsigned long long duration_ns) {
  const auto start = timer_ns();
  if (threadIdx.x == 0) atomicAdd_system(started, 1);
  float value = 1.0f + threadIdx.x;
  do {
    for (int i = 0; i < 256; ++i) value = fmaf(value, 0.9999f, 0.0001f);
  } while (timer_ns() - start < duration_ns);
  output[blockIdx.x * blockDim.x + threadIdx.x] = value;
}
extern "C" __global__ void verify(const unsigned* data, unsigned long long count,
                                  unsigned expected, unsigned* errors) {
  for (auto i = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += static_cast<unsigned long long>(gridDim.x) * blockDim.x) {
    if (data[i] != expected) atomicAdd(errors, 1U);
  }
}
