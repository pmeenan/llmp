// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include <cuda_runtime.h>

#include <cstdint>

#include "wake_bench_kernels.h"

namespace llmp::wake {
namespace {

__device__ std::uint64_t Now() {
  std::uint64_t t = 0;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

__global__ void StepKernel(std::uint64_t ns, std::uint64_t* stamps) {
  const std::uint64_t start = Now();
  std::uint64_t now = start;
  while (now - start < ns) {
    now = Now();
  }
  stamps[0] = start;
  stamps[1] = now;
  __threadfence_system();
}

__global__ void TickKernel(std::uint64_t ns, std::uint64_t* out) {
  const std::uint64_t start = Now();
  std::uint64_t now = start;
  while (now - start < ns) {
    now = Now();
    *reinterpret_cast<volatile std::uint64_t*>(out) = now;
    __threadfence_system();
  }
}

}  // namespace

cudaError_t Step(cudaStream_t stream, std::uint64_t ns, std::uint64_t* stamps) {
  StepKernel<<<1, 1, 0, stream>>>(ns, stamps);
  return cudaGetLastError();
}

cudaError_t Tick(cudaStream_t stream, std::uint64_t ns, std::uint64_t* out) {
  TickKernel<<<1, 1, 0, stream>>>(ns, out);
  return cudaGetLastError();
}

}  // namespace llmp::wake
