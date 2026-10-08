// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Internal to the GGML module: what the tensor-core flash-attention
// instance units (fattn_mma_d128.cu, fattn_mma_d256.cu, fattn_mma_d512.cu) tell the dispatcher
// (fattn_mma.cu) about the kernels they compile. Each unit instantiates
// ggml_cuda_flash_attn_ext_mma_f16_case<D, D, columns, 8> for columns 1, 2,
// 4 and 8, as GGML's template-instances/fattn-mma-f16-instance-*.cu do for
// every head size, and can query its kernels' occupancy, which only the
// unit that instantiates a kernel can name.

#ifndef LLMP_KERNELS_GGML_FATTN_MMA_H_
#define LLMP_KERNELS_GGML_FATTN_MMA_H_

#include <expected>
#include <string>

struct ggml_backend_cuda_context;
struct ggml_tensor;

namespace llmp::kernels::ggml::detail {

// What launch_fattn's grid arithmetic needs of one MMA kernel on a device
// (fattn-mma-f16.cuh:1966-2067): its KV batch and how many of its blocks
// fit a multiprocessor, after raising its dynamic shared memory limit as
// the case function does before launching it.
struct MmaKernelShape {
  int kv_batch = 0;               // nbatch_fa
  int blocks_per_sm = 0;          // occupancy at the case's threads and shared memory
  bool async_kv_preload = false;  // the upstream two-stage, nonsparse kernel
};

// For columns 1, 2, 4 or 8; sparse cases have one or eight columns.
std::expected<MmaKernelShape, std::string> FlashAttnMmaShape256(int columns, bool sparse,
                                                                int device);
std::expected<MmaKernelShape, std::string> FlashAttnMmaShape512(int columns, bool sparse,
                                                                int device);

// The instantiated case for columns 1, 2, 4 or 8, or null.
using MmaCase = void (*)(ggml_backend_cuda_context& context, ggml_tensor* node);
MmaCase FlashAttnMmaCase256(int columns);
// Exact GQA2 D256 cases, query tiles4/8/16/32. Query the specialization
// selected by the node; omitted softcap preserves existing zero-only callers.
std::expected<MmaKernelShape, std::string> FlashAttnMmaShapeGqa2(int columns, int device,
                                                                 bool softcap = false);
MmaCase FlashAttnMmaCaseGqa2(int columns);
MmaCase FlashAttnMmaCase512(int columns);
// Experimental: the same cases reading F16 Q (fattn_mma_q16.cuh).
MmaCase FlashAttnMmaCase512Q16(int columns);

// Head dimension 128 without grouping (ncols2 1: multi-head attention, the
// Qwen-Image denoiser's), for columns 8, 16, 32 or 64
// (fattn_mma_d128.cu).
std::expected<MmaKernelShape, std::string> FlashAttnMmaShape128(int columns, int device);
MmaCase FlashAttnMmaCase128(int columns);

}  // namespace llmp::kernels::ggml::detail

#endif  // LLMP_KERNELS_GGML_FATTN_MMA_H_
