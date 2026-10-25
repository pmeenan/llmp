// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include <cuda_runtime.h>

#include <cstdint>
#include <limits>

#include "convert.cuh"
#include "kernels/exl3/input_conversion.h"

#ifndef LLMP_EXL3_CONVERSION_GGML_TREE
#error "EXL3 input conversion must name its actual pinned GGML source"
#endif

namespace llmp::kernels::exl3 {
std::string_view InputConversionSourceIdentity() {
  return "ggml-f32-to-f16-rn-v1;" LLMP_EXL3_CONVERSION_GGML_TREE;
}
std::expected<void, KernelFailure> LaunchInputConversion(void* stream, std::uint64_t source,
                                                         std::uint64_t destination,
                                                         std::uint64_t elements) noexcept {
  if (source == 0 || destination == 0 || source % 4 != 0 || destination % 2 != 0 || elements == 0 ||
      elements > static_cast<std::uint64_t>(INT64_MAX) / 4 || source > UINT64_MAX - elements * 4 ||
      destination > UINT64_MAX - elements * 2 ||
      (source < destination ? destination - source < elements * 4
                            : source - destination < elements * 2)) {
    return std::unexpected(KernelFailure{.error = KernelError::kRejected,
                                         .detail = "invalid EXL3 F32 input conversion ranges"});
  }
  const auto convert = ggml_get_to_fp16_cuda(GGML_TYPE_F32);
  if (!convert)
    return std::unexpected(KernelFailure{.error = KernelError::kRejected,
                                         .detail = "the pinned F32 input converter is absent"});
  convert(reinterpret_cast<const void*>(source), reinterpret_cast<half*>(destination),
          static_cast<std::int64_t>(elements), static_cast<cudaStream_t>(stream));
  const auto error = cudaGetLastError();
  if (error != cudaSuccess)
    return std::unexpected(
        KernelFailure{.error = KernelError::kUnknown, .detail = cudaGetErrorString(error)});
  return {};
}
}  // namespace llmp::kernels::exl3
