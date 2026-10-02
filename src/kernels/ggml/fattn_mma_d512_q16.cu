// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: MIT AND Apache-2.0

// The D=512 cases of jitLLM's F16-Q copy of GGML's MMA flash attention
// (fattn_mma_q16.cuh), with the same column counts as fattn_mma_d512.cu.

#include "kernels/ggml/fattn_mma.h"
#include "kernels/ggml/fattn_mma_q16.cuh"

template void jitllm_fattn_q16::ggml_cuda_flash_attn_ext_mma_f16_case<512, 512, 1, 8>(
    ggml_backend_cuda_context& ctx, ggml_tensor* dst);
template void jitllm_fattn_q16::ggml_cuda_flash_attn_ext_mma_f16_case<512, 512, 2, 8>(
    ggml_backend_cuda_context& ctx, ggml_tensor* dst);
template void jitllm_fattn_q16::ggml_cuda_flash_attn_ext_mma_f16_case<512, 512, 4, 8>(
    ggml_backend_cuda_context& ctx, ggml_tensor* dst);
template void jitllm_fattn_q16::ggml_cuda_flash_attn_ext_mma_f16_case<512, 512, 8, 8>(
    ggml_backend_cuda_context& ctx, ggml_tensor* dst);

namespace jitllm::kernels::ggml::detail {

MmaCase FlashAttnMmaCase512Q16(int columns) {
  switch (columns) {
    case 1:
      return &jitllm_fattn_q16::ggml_cuda_flash_attn_ext_mma_f16_case<512, 512, 1, 8>;
    case 2:
      return &jitllm_fattn_q16::ggml_cuda_flash_attn_ext_mma_f16_case<512, 512, 2, 8>;
    case 4:
      return &jitllm_fattn_q16::ggml_cuda_flash_attn_ext_mma_f16_case<512, 512, 4, 8>;
    case 8:
      return &jitllm_fattn_q16::ggml_cuda_flash_attn_ext_mma_f16_case<512, 512, 8, 8>;
    default:
      return nullptr;
  }
}

}  // namespace jitllm::kernels::ggml::detail
