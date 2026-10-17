// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: MIT AND Apache-2.0

// The D=512 cases of llmpalooza's F16-Q copy of GGML's MMA flash attention
// (fattn_mma_q16.cuh), with the same column counts as fattn_mma_d512.cu,
// and their occupancy: the copy's kernels, not GGML's, are what the cases
// launch, and the plan's grid and scratch must be the launcher's.

#include <algorithm>
#include <cstddef>
#include <expected>
#include <string>

#include "kernels/ggml/fattn_mma.h"
#include "kernels/ggml/fattn_mma_q16.cuh"

namespace llmp_fattn_q16 {
// MmaShape over this namespace's configuration and kernels.
#include "kernels/ggml/fattn_mma_shape_body.cuh"
}  // namespace llmp_fattn_q16

template void llmp_fattn_q16::ggml_cuda_flash_attn_ext_mma_f16_case<512, 512, 1, 8>(
    ggml_backend_cuda_context& ctx, ggml_tensor* dst);
template void llmp_fattn_q16::ggml_cuda_flash_attn_ext_mma_f16_case<512, 512, 2, 8>(
    ggml_backend_cuda_context& ctx, ggml_tensor* dst);
template void llmp_fattn_q16::ggml_cuda_flash_attn_ext_mma_f16_case<512, 512, 4, 8>(
    ggml_backend_cuda_context& ctx, ggml_tensor* dst);
template void llmp_fattn_q16::ggml_cuda_flash_attn_ext_mma_f16_case<512, 512, 8, 8>(
    ggml_backend_cuda_context& ctx, ggml_tensor* dst);

namespace llmp::kernels::ggml::detail {

std::expected<MmaKernelShape, std::string> FlashAttnMmaShape512Q16(int columns, bool sparse,
                                                                   int device) {
  if (sparse) {
    if (columns == 1) {
      return llmp_fattn_q16::MmaShape<512, 1, true>(device);
    }
    if (columns == 8) {
      return llmp_fattn_q16::MmaShape<512, 8, true>(device);
    }
    return std::unexpected(std::string("the sparse D=512 MMA case has one or eight columns"));
  }
  switch (columns) {
    case 1:
      return llmp_fattn_q16::MmaShape<512, 1, false>(device);
    case 2:
      return llmp_fattn_q16::MmaShape<512, 2, false>(device);
    case 4:
      return llmp_fattn_q16::MmaShape<512, 4, false>(device);
    case 8:
      return llmp_fattn_q16::MmaShape<512, 8, false>(device);
    default:
      return std::unexpected(std::string("no D=512 MMA case for this column count"));
  }
}

MmaCase FlashAttnMmaCase512Q16(int columns) {
  switch (columns) {
    case 1:
      return &llmp_fattn_q16::ggml_cuda_flash_attn_ext_mma_f16_case<512, 512, 1, 8>;
    case 2:
      return &llmp_fattn_q16::ggml_cuda_flash_attn_ext_mma_f16_case<512, 512, 2, 8>;
    case 4:
      return &llmp_fattn_q16::ggml_cuda_flash_attn_ext_mma_f16_case<512, 512, 4, 8>;
    case 8:
      return &llmp_fattn_q16::ggml_cuda_flash_attn_ext_mma_f16_case<512, 512, 8, 8>;
    default:
      return nullptr;
  }
}

}  // namespace llmp::kernels::ggml::detail
