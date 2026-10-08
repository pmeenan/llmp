// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// FillRanges (paging.h): one block per range, its threads striding over the
// range's bytes.

#include <cuda_runtime.h>

#include <cstdint>

#include "kernels/paging/paging.h"

namespace llmp::kernels::paging {

namespace {

__global__ void FillKernel(const std::uint64_t* ranges, unsigned char value) {
  const std::uint64_t address = ranges[2 * static_cast<std::uint64_t>(blockIdx.x)];
  const std::uint64_t bytes = ranges[(2 * static_cast<std::uint64_t>(blockIdx.x)) + 1];
  auto* to = reinterpret_cast<unsigned char*>(address);  // NOLINT(performance-no-int-to-ptr)
  for (std::uint64_t i = threadIdx.x; i < bytes; i += blockDim.x) {
    to[i] = value;
  }
}

}  // namespace

bool FillRanges(const std::uint64_t* ranges, std::uint32_t count, std::uint8_t value,
                void* stream) {
  if (count == 0) {
    return true;
  }
  FillKernel<<<count, 256, 0, static_cast<cudaStream_t>(stream)>>>(ranges, value);
  return cudaGetLastError() == cudaSuccess;
}

}  // namespace llmp::kernels::paging
