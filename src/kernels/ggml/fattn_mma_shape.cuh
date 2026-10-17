// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: MIT AND Apache-2.0

// Internal to the GGML module, for the tensor-core flash-attention instance
// units only (fattn_mma.h): the shape of one MMA kernel on a device, by a
// recorded copy of the host arithmetic of ggml_cuda_flash_attn_ext_mma_f16_case
// at llama.cpp d81235049384534c167caea52b85a694f6103d14 and of
// launch_fattn's occupancy query, for the kernel the case
// launches with the selected logit-softcap specialization.

#ifndef LLMP_KERNELS_GGML_FATTN_MMA_SHAPE_CUH_
#define LLMP_KERNELS_GGML_FATTN_MMA_SHAPE_CUH_

#include <algorithm>
#include <cstddef>
#include <expected>
#include <string>

// fattn-mma-f16.cuh has no include guard: the includer includes it once,
// before this header.
#include "kernels/ggml/fattn_mma.h"

namespace llmp::kernels::ggml::detail {

#include "kernels/ggml/fattn_mma_shape_body.cuh"

}  // namespace llmp::kernels::ggml::detail

#endif  // LLMP_KERNELS_GGML_FATTN_MMA_SHAPE_CUH_
