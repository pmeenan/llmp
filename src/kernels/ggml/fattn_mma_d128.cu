// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: MIT AND Apache-2.0

// GGML's tensor-core flash attention at head dimension 128 without query
// head grouping (ncols2 1), for multi-head attention over long sequences
// (Qwen-Image-2.1's denoiser: 32 heads, one per KV head): the four cases
// upstream's switch_ncols1 picks from for one query head per tile
// (fattn.cu:133-167), instantiated as GGML's
// template-instances/fattn-mma-f16-instance-ncols1_*-ncols2_1.cu do at
// llama.cpp b29c606e2, and built with GGML's device flags (CMakeLists.txt).

#include <expected>
#include <string>

#include "fattn-mma-f16.cuh"
#include "kernels/ggml/fattn_mma.h"
#include "kernels/ggml/fattn_mma_shape.cuh"

DECL_FATTN_MMA_F16_CASE(128, 128, 8, 1);
DECL_FATTN_MMA_F16_CASE(128, 128, 16, 1);
DECL_FATTN_MMA_F16_CASE(128, 128, 32, 1);
DECL_FATTN_MMA_F16_CASE(128, 128, 64, 1);

namespace llmp::kernels::ggml::detail {

std::expected<MmaKernelShape, std::string> FlashAttnMmaShape128(int columns, int device) {
  switch (columns) {
    case 8:
      return MmaShape<128, 8, false, 1>(device);
    case 16:
      return MmaShape<128, 16, false, 1>(device);
    case 32:
      return MmaShape<128, 32, false, 1>(device);
    case 64:
      return MmaShape<128, 64, false, 1>(device);
    default:
      return std::unexpected(std::string("no D=128 MMA case for this column count"));
  }
}

MmaCase FlashAttnMmaCase128(int columns) {
  switch (columns) {
    case 8:
      return &ggml_cuda_flash_attn_ext_mma_f16_case<128, 128, 8, 1>;
    case 16:
      return &ggml_cuda_flash_attn_ext_mma_f16_case<128, 128, 16, 1>;
    case 32:
      return &ggml_cuda_flash_attn_ext_mma_f16_case<128, 128, 32, 1>;
    case 64:
      return &ggml_cuda_flash_attn_ext_mma_f16_case<128, 128, 64, 1>;
    default:
      return nullptr;
  }
}

}  // namespace llmp::kernels::ggml::detail
