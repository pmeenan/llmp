// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: MIT AND Apache-2.0

// GGML's tensor-core flash attention at head dimension 256 (Qwen3.8's QSA),
// grouping 8 query heads per KV head: the four cases the dispatcher
// (fattn_mma.cu) launches, instantiated as GGML's
// template-instances/fattn-mma-f16-instance-ncols1_*-ncols2_8.cu do at
// llama.cpp b29c606e2, and built with GGML's device flags (CMakeLists.txt).

#include <expected>
#include <string>

#include "fattn-mma-f16.cuh"
#include "kernels/ggml/fattn_mma.h"
#include "kernels/ggml/fattn_mma_shape.cuh"

DECL_FATTN_MMA_F16_CASE(256, 256, 1, 8);
DECL_FATTN_MMA_F16_CASE(256, 256, 2, 8);
DECL_FATTN_MMA_F16_CASE(256, 256, 4, 8);
DECL_FATTN_MMA_F16_CASE(256, 256, 8, 8);

namespace jitllm::kernels::ggml::detail {

std::expected<MmaKernelShape, std::string> FlashAttnMmaShape256(int columns, bool sparse,
                                                                int device) {
  if (sparse) {
    if (columns == 1) {
      return MmaShape<256, 1, true>(device);
    }
    if (columns == 8) {
      return MmaShape<256, 8, true>(device);
    }
    return std::unexpected(std::string("the sparse D=256 MMA case has one or eight columns"));
  }
  switch (columns) {
    case 1:
      return MmaShape<256, 1, false>(device);
    case 2:
      return MmaShape<256, 2, false>(device);
    case 4:
      return MmaShape<256, 4, false>(device);
    case 8:
      return MmaShape<256, 8, false>(device);
    default:
      return std::unexpected(std::string("no D=256 MMA case for this column count"));
  }
}

MmaCase FlashAttnMmaCase256(int columns) {
  switch (columns) {
    case 1:
      return &ggml_cuda_flash_attn_ext_mma_f16_case<256, 256, 1, 8>;
    case 2:
      return &ggml_cuda_flash_attn_ext_mma_f16_case<256, 256, 2, 8>;
    case 4:
      return &ggml_cuda_flash_attn_ext_mma_f16_case<256, 256, 4, 8>;
    case 8:
      return &ggml_cuda_flash_attn_ext_mma_f16_case<256, 256, 8, 8>;
    default:
      return nullptr;
  }
}

}  // namespace jitllm::kernels::ggml::detail
