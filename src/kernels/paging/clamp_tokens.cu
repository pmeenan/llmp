// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// ClampTokens (paging.h): one thread a token.

#include <cuda_runtime.h>

#include <cstdint>

#include "kernels/paging/paging.h"

namespace llmp::kernels::paging {

namespace {

__global__ void ClampKernel(std::int32_t* tokens, std::uint32_t count, std::int32_t limit) {
  const std::uint32_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i < count && (tokens[i] < 0 || tokens[i] >= limit)) {
    tokens[i] = 0;
  }
}

}  // namespace

bool ClampTokens(std::int32_t* tokens, std::uint32_t count, std::int32_t limit, void* stream) {
  if (count == 0) {
    return true;
  }
  constexpr std::uint32_t kThreads = 128;
  ClampKernel<<<(count + kThreads - 1) / kThreads, kThreads, 0,
                static_cast<cudaStream_t>(stream)>>>(tokens, count, limit);
  return cudaGetLastError() == cudaSuccess;
}

}  // namespace llmp::kernels::paging
