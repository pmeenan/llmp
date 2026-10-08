// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// llmp.vecq's raw launch (dsv4_fast.cu; llmp_ops.h, "DeepSeek V4's fast
// plan"), below the node checks, for the launch-configuration measurements
// (benchmarks/vecq_bench.cc) and RunVecQ. CUDA builds only.

#ifndef LLMP_KERNELS_GGML_DSV4_FAST_H_
#define LLMP_KERNELS_GGML_DSV4_FAST_H_

#include <cuda_runtime.h>

#include <cstdint>
#include <string>

#include "ggml.h"

namespace llmp::kernels::ggml {

// What a llmp.vecq launch reads and writes. Weight offsets count the
// weight type's blocks; activation offsets count Q8_1 blocks (36 bytes).
struct VecQDesc {
  const void* w = nullptr;            // weights (the up projection with a GLU)
  const void* g = nullptr;            // gate weights, or null
  const void* y = nullptr;            // Q8_1 activations
  const std::int32_t* ids = nullptr;  // null: dense
  float* dst = nullptr;
  int ncols_x = 0;        // k
  int nrows = 0;          // n
  int stride_row = 0;     // weight blocks between rows
  int stride_expert = 0;  // weight blocks between experts
  int y_token = 0;        // Q8_1 blocks between tokens' rows
  int y_slot = 0;         // between slots' rows (per-slot activations), else 0
  int ids_stride = 0;     // ids between tokens
  int used = 1;
  int tokens = 1;
  int dst_token = 0;  // floats
  int dst_slot = 0;
  int glu = 0;  // VecQGlu
  float limit = 0.0f;
};

// The launch configurations built (for measurement): -1 is the default
// choice for the shape. Returns false for a variant the shape cannot take.
int VecQVariants();
std::string VecQVariantName(int variant);
bool LaunchVecQ(ggml_type type, const VecQDesc& d, int variant, cudaStream_t stream);

}  // namespace llmp::kernels::ggml

#endif  // LLMP_KERNELS_GGML_DSV4_FAST_H_
