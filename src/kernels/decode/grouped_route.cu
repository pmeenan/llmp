// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include <cuda_runtime.h>

#include <format>

#include "kernels/decode/grouped_route.h"

namespace llmp::kernels::decode {
namespace {

constexpr unsigned kThreads = 512;

__device__ const std::uint8_t* Moved(const std::uint8_t* p, std::uint64_t pitch, std::uint32_t e) {
  return p == nullptr ? nullptr : p + (static_cast<std::uint64_t>(e) * pitch);
}

__global__ void __launch_bounds__(kThreads) RouteKernel(const ResidentRoute r) {
  __shared__ std::int32_t count[kMaxResidentExperts];
  __shared__ std::int32_t cursor[kMaxResidentExperts];
  // A group's index fits 16 bits (kMaxResidentExperts), which keeps the
  // block within 48 KiB of static shared memory.
  __shared__ std::int16_t slot[kMaxResidentExperts];
  static_assert(kMaxResidentExperts <= 32768);
  __shared__ std::int32_t active[kThreads];
  __shared__ std::int32_t runs[kThreads];
  const std::uint32_t pairs = r.used * r.tokens;
  for (std::uint32_t e = threadIdx.x; e < r.experts; e += blockDim.x) count[e] = 0;
  __syncthreads();
  for (std::uint32_t p = threadIdx.x; p < pairs; p += blockDim.x) {
    const std::uint32_t t = p / r.used, j = p % r.used;
    const std::int32_t e = r.ids[(static_cast<std::uint64_t>(t) * r.ids_stride) + j];
    if (e >= 0 && static_cast<std::uint32_t>(e) < r.experts) atomicAdd(&count[e], 1);
  }
  __syncthreads();
  // Each thread's run of experts: its active experts and their pairs, then
  // an inclusive scan of both over the block, in expert order.
  const std::uint32_t per = (r.experts + kThreads - 1) / kThreads;
  const std::uint32_t first = min(r.experts, threadIdx.x * per);
  const std::uint32_t last = min(r.experts, first + per);
  std::int32_t mine = 0, mine_pairs = 0;
  for (std::uint32_t e = first; e < last; ++e)
    if (count[e] > 0) {
      ++mine;
      mine_pairs += count[e];
    }
  active[threadIdx.x] = mine;
  runs[threadIdx.x] = mine_pairs;
  __syncthreads();
  for (unsigned offset = 1; offset < kThreads; offset <<= 1U) {
    const std::int32_t a = threadIdx.x >= offset ? active[threadIdx.x - offset] : 0;
    const std::int32_t n = threadIdx.x >= offset ? runs[threadIdx.x - offset] : 0;
    __syncthreads();
    active[threadIdx.x] += a;
    runs[threadIdx.x] += n;
    __syncthreads();
  }
  std::uint32_t c = static_cast<std::uint32_t>(active[threadIdx.x] - mine);
  std::int32_t at = runs[threadIdx.x] - mine_pairs;
  for (std::uint32_t e = first; e < last; ++e) {
    slot[e] = -1;
    if (count[e] == 0 || c >= r.groups) continue;
    slot[e] = static_cast<std::int16_t>(c);
    cursor[e] = at;
    r.offsets[c] = at;
    Operand d = r.first;
    d.data = Moved(d.data, r.data_pitch, e);
    d.scales = Moved(d.scales, r.scales_pitch, e);
    d.tensor_scale = reinterpret_cast<const float*>(
        Moved(reinterpret_cast<const std::uint8_t*>(d.tensor_scale), r.tensor_scale_pitch, e));
    d.biases = Moved(d.biases, r.biases_pitch, e);
    r.descriptors[c] = d;
    at += count[e];
    ++c;
  }
  const auto used_groups = min(static_cast<std::uint32_t>(active[kThreads - 1]), r.groups);
  const std::int32_t total = runs[kThreads - 1];
  for (std::uint32_t g = used_groups + threadIdx.x; g < r.groups; g += blockDim.x) {
    r.offsets[g] = total;
    r.descriptors[g] = Operand{};
  }
  if (threadIdx.x == 0) r.offsets[r.groups] = total;
  __syncthreads();
  for (std::uint32_t p = threadIdx.x; p < pairs; p += blockDim.x) {
    const std::uint32_t t = p / r.used, j = p % r.used;
    const std::int32_t e = r.ids[(static_cast<std::uint64_t>(t) * r.ids_stride) + j];
    if (e < 0 || static_cast<std::uint32_t>(e) >= r.experts || slot[e] < 0) {
      // A dropped pair's output row reads as zeros, not stale values.
      if (r.y != nullptr)
        for (std::uint32_t i = 0; i < r.y_outputs; ++i)
          r.y[(static_cast<std::uint64_t>(p) * r.y_stride) + i] = 0.0F;
      continue;
    }
    const std::int32_t at_pair = atomicAdd(&cursor[e], 1);
    r.pair_rows[at_pair] = static_cast<std::int32_t>(r.per_slot_input ? p : t);
    r.pair_out[at_pair] = static_cast<std::int32_t>(p);
  }
}

}  // namespace

Status PrepareResidentRoute(const ResidentRoute& r, Stream stream) {
  const std::uint64_t pairs = static_cast<std::uint64_t>(r.used) * r.tokens;
  if (r.experts == 0 || r.experts > kMaxResidentExperts || r.used == 0 || r.tokens == 0 ||
      pairs > static_cast<std::uint64_t>(INT32_MAX) || r.groups == 0 ||
      r.groups > (pairs < r.experts ? pairs : r.experts) || r.ids == nullptr ||
      r.ids_stride < r.used || r.descriptors == nullptr || r.offsets == nullptr ||
      r.pair_rows == nullptr || r.pair_out == nullptr ||
      (r.y != nullptr && r.y_stride < r.y_outputs))
    return std::unexpected(std::format("grouped route: {} experts, {} used of {} tokens, {} groups",
                                       r.experts, r.used, r.tokens, r.groups));
  RouteKernel<<<1, kThreads, 0, static_cast<cudaStream_t>(stream)>>>(r);
  const cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess)
    return std::unexpected(std::string("grouped route: ") + cudaGetErrorString(error));
  return {};
}

}  // namespace llmp::kernels::decode
