// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The n-gram rows' gather (paging.h): one block per row slot, each thread
// a byte of the row, from the pinned landing into device memory. The grid
// is the chunk shape's bound on slots, and the slots gathered are the count
// the device reads, so a captured decode graph (D-090) replays it with the
// next step's rows.

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "kernels/paging/paging.h"

namespace llmp::kernels::paging {

namespace {

__global__ void GatherKernel(const unsigned char* landing, const std::uint32_t* sources,
                             const std::uint32_t* count, std::uint32_t row_bytes,
                             unsigned char* slots) {
  const std::uint32_t slot = blockIdx.x;
  if (slot >= *count) {
    return;
  }
  const unsigned char* from = landing + sources[slot];
  unsigned char* to = slots + (static_cast<std::uint64_t>(slot) * row_bytes);
  for (std::uint32_t i = threadIdx.x; i < row_bytes; i += blockDim.x) {
    to[i] = from[i];
  }
}

}  // namespace

bool GatherPleRows(const std::byte* landing, const std::uint32_t* sources,
                   const std::uint32_t* count, std::uint32_t max_count, std::uint32_t row_bytes,
                   std::byte* slots, void* stream) {
  if (max_count == 0) {
    return true;
  }
  GatherKernel<<<max_count, 128, 0, static_cast<cudaStream_t>(stream)>>>(
      reinterpret_cast<const unsigned char*>(landing), sources, count, row_bytes,
      reinterpret_cast<unsigned char*>(slots));
  return cudaGetLastError() == cudaSuccess;
}

}  // namespace llmp::kernels::paging
