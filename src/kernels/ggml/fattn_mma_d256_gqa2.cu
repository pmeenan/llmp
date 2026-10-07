// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: MIT AND Apache-2.0

// Pinned D256/group2 MMA cases for Gemma local attention; no copied kernel.
#include <expected>
#include <string>

#include "fattn-mma-f16.cuh"
#include "kernels/ggml/fattn_mma.h"
#include "kernels/ggml/fattn_mma_shape.cuh"

DECL_FATTN_MMA_F16_CASE(256, 256, 4, 2);
DECL_FATTN_MMA_F16_CASE(256, 256, 8, 2);
DECL_FATTN_MMA_F16_CASE(256, 256, 16, 2);
DECL_FATTN_MMA_F16_CASE(256, 256, 32, 2);

namespace jitllm::kernels::ggml::detail {
template <bool kSoftcap>
static std::expected<MmaKernelShape, std::string> Gqa2Shape(int columns, int device) {
  switch (columns) {
    case 4:
      return MmaShape<256, 4, false, 2, kSoftcap>(device);
    case 8:
      return MmaShape<256, 8, false, 2, kSoftcap>(device);
    case 16:
      return MmaShape<256, 16, false, 2, kSoftcap>(device);
    case 32:
      return MmaShape<256, 32, false, 2, kSoftcap>(device);
    default:
      return std::unexpected(std::string("no D256/group2 MMA query tile"));
  }
}
std::expected<MmaKernelShape, std::string> FlashAttnMmaShapeGqa2(int columns, int device,
                                                                 bool softcap) {
  return softcap ? Gqa2Shape<true>(columns, device) : Gqa2Shape<false>(columns, device);
}
MmaCase FlashAttnMmaCaseGqa2(int columns) {
  switch (columns) {
    case 4:
      return &ggml_cuda_flash_attn_ext_mma_f16_case<256, 256, 4, 2>;
    case 8:
      return &ggml_cuda_flash_attn_ext_mma_f16_case<256, 256, 8, 2>;
    case 16:
      return &ggml_cuda_flash_attn_ext_mma_f16_case<256, 256, 16, 2>;
    case 32:
      return &ggml_cuda_flash_attn_ext_mma_f16_case<256, 256, 32, 2>;
    default:
      return nullptr;
  }
}
}  // namespace jitllm::kernels::ggml::detail
