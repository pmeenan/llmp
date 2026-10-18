// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: MIT AND Apache-2.0

// Upstream group-one MMA supports arbitrary GQA ratios and tail KV cells.
// Two query tiles per added head size bound template build cost. D128 uses
// its existing instance unit and retains the image path's launch geometry.
#include "fattn-mma-f16.cuh"
#include "kernels/ggml/fattn_mma_shape.cuh"

DECL_FATTN_MMA_F16_CASE(64, 64, 8, 1);
DECL_FATTN_MMA_F16_CASE(64, 64, 32, 1);
DECL_FATTN_MMA_F16_CASE(80, 80, 8, 1);
DECL_FATTN_MMA_F16_CASE(80, 80, 32, 1);
DECL_FATTN_MMA_F16_CASE(96, 96, 8, 1);
DECL_FATTN_MMA_F16_CASE(96, 96, 32, 1);
DECL_FATTN_MMA_F16_CASE(112, 112, 8, 1);
DECL_FATTN_MMA_F16_CASE(112, 112, 32, 1);
DECL_FATTN_MMA_F16_CASE(192, 128, 8, 1);
DECL_FATTN_MMA_F16_CASE(192, 128, 32, 1);
DECL_FATTN_MMA_F16_CASE(192, 192, 8, 1);
DECL_FATTN_MMA_F16_CASE(192, 192, 32, 1);

namespace llmp::kernels::ggml::detail {
std::expected<MmaKernelShape, std::string> FlashAttnMmaShapeCommon(int head, int value, int columns,
                                                                   int device) {
  switch (head) {
    case 64:
      if (columns == 8) return MmaShape<64, 8, false, 1>(device);
      if (columns == 32) return MmaShape<64, 32, false, 1>(device);
      break;
    case 80:
      if (columns == 8) return MmaShape<80, 8, false, 1>(device);
      if (columns == 32) return MmaShape<80, 32, false, 1>(device);
      break;
    case 96:
      if (columns == 8) return MmaShape<96, 8, false, 1>(device);
      if (columns == 32) return MmaShape<96, 32, false, 1>(device);
      break;
    case 112:
      if (columns == 8) return MmaShape<112, 8, false, 1>(device);
      if (columns == 32) return MmaShape<112, 32, false, 1>(device);
      break;
    case 192:
      if (value == 192 && columns == 8) return MmaShape<192, 8, false, 1>(device);
      if (value == 192 && columns == 32) return MmaShape<192, 32, false, 1>(device);
      if (columns == 8) return MmaShape<192, 8, false, 1, false, 128>(device);
      if (columns == 32) return MmaShape<192, 32, false, 1, false, 128>(device);
      break;
    case 128:
      return FlashAttnMmaShape128(columns, device);
  }
  return std::unexpected(std::string("no common-head MMA specialization"));
}
MmaCase FlashAttnMmaCaseCommon(int head, int value, int columns) {
  switch (head) {
    case 64:
      if (columns == 8) return &ggml_cuda_flash_attn_ext_mma_f16_case<64, 64, 8, 1>;
      if (columns == 32) return &ggml_cuda_flash_attn_ext_mma_f16_case<64, 64, 32, 1>;
      break;
    case 80:
      if (columns == 8) return &ggml_cuda_flash_attn_ext_mma_f16_case<80, 80, 8, 1>;
      if (columns == 32) return &ggml_cuda_flash_attn_ext_mma_f16_case<80, 80, 32, 1>;
      break;
    case 96:
      if (columns == 8) return &ggml_cuda_flash_attn_ext_mma_f16_case<96, 96, 8, 1>;
      if (columns == 32) return &ggml_cuda_flash_attn_ext_mma_f16_case<96, 96, 32, 1>;
      break;
    case 112:
      if (columns == 8) return &ggml_cuda_flash_attn_ext_mma_f16_case<112, 112, 8, 1>;
      if (columns == 32) return &ggml_cuda_flash_attn_ext_mma_f16_case<112, 112, 32, 1>;
      break;
    case 192:
      if (value == 192 && columns == 8)
        return &ggml_cuda_flash_attn_ext_mma_f16_case<192, 192, 8, 1>;
      if (value == 192 && columns == 32)
        return &ggml_cuda_flash_attn_ext_mma_f16_case<192, 192, 32, 1>;
      if (columns == 8) return &ggml_cuda_flash_attn_ext_mma_f16_case<192, 128, 8, 1>;
      if (columns == 32) return &ggml_cuda_flash_attn_ext_mma_f16_case<192, 128, 32, 1>;
      break;
    case 128:
      return FlashAttnMmaCase128(columns);
  }
  return nullptr;
}
}  // namespace llmp::kernels::ggml::detail
