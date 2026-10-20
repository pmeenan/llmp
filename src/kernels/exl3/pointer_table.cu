// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include <cuda_runtime.h>

#include <cstdint>
#include <span>

#include "kernels/exl3/pointer_table.h"

namespace llmp::kernels::exl3 {
namespace {

struct PointerTablePayload {
  std::uint64_t words[8];
  std::uint32_t count;
};

__global__ void WritePointerTableKernel(std::uint64_t* dst, PointerTablePayload payload) {
  const std::uint32_t i = threadIdx.x;
  if (i < payload.count) {
    dst[i] = payload.words[i];
  }
}

}  // namespace

std::expected<void, KernelFailure> LaunchWritePointerTable(
    void* stream, std::uint64_t dst_device_ptr, std::span<const std::uint64_t> words) noexcept {
  if (words.empty() || words.size() > 8 || dst_device_ptr == 0 ||
      (dst_device_ptr % alignof(std::uint64_t)) != 0) {
    return std::unexpected(KernelFailure{.error = KernelError::kRejected,
                                         .detail = "invalid pointer table arguments"});
  }
  PointerTablePayload payload{};
  payload.count = static_cast<std::uint32_t>(words.size());
  for (std::size_t i = 0; i < words.size(); ++i) {
    payload.words[i] = words[i];
  }
  auto* dst = reinterpret_cast<std::uint64_t*>(dst_device_ptr);
  auto cuda_stream = static_cast<cudaStream_t>(stream);
  WritePointerTableKernel<<<1, static_cast<unsigned>(words.size()), 0, cuda_stream>>>(dst, payload);
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return std::unexpected(
        KernelFailure{.error = KernelError::kUnknown, .detail = cudaGetErrorString(err)});
  }
  return {};
}

}  // namespace llmp::kernels::exl3
