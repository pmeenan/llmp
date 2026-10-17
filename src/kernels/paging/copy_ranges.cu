// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// CopyRanges (paging.h): one block per range, 16 bytes a thread per step.
// The engine's live state queues a verify's snapshot and restore with it
// (engine/live_state.cc).

#include <cuda_runtime.h>

#include <cstdint>

#include "kernels/paging/paging.h"

namespace llmp::kernels::paging {

namespace {

__global__ void CopyKernel(const RangeCopy* __restrict__ ranges) {
  const RangeCopy r = ranges[blockIdx.x];
  const std::uint64_t vectors = r.bytes / 16;
  const auto* from = reinterpret_cast<const uint4*>(r.from);  // NOLINT(performance-no-int-to-ptr)
  auto* to = reinterpret_cast<uint4*>(r.to);                  // NOLINT(performance-no-int-to-ptr)
  for (std::uint64_t v = threadIdx.x; v < vectors; v += blockDim.x) {
    to[v] = from[v];
  }
}

}  // namespace

bool CopyRanges(const RangeCopy* ranges, std::uint32_t count, void* stream) {
  if (count == 0) {
    return true;
  }
  if (!RangeCopiesValid(ranges, count)) {
    return false;
  }
  CopyKernel<<<count, 256, 0, static_cast<cudaStream_t>(stream)>>>(ranges);
  return cudaGetLastError() == cudaSuccess;
}

}  // namespace llmp::kernels::paging
