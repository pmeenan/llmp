// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// GGML's ordinary MMVQ and SwiGLU for vecq_bench (vecq_upstream.cu). CUDA
// builds only.
#ifndef LLMP_BENCHMARKS_VECQ_UPSTREAM_H_
#define LLMP_BENCHMARKS_VECQ_UPSTREAM_H_

#include <cuda_runtime.h>

#include "ggml.h"

namespace llmp::benchmarks::vecq {
// W [k, n] of `type`, `q8` the columns' Q8_1 rows (each padded to 512
// values), dst F32 [n, columns]; 1..8 columns. False if refused.
bool UpstreamMmvq(ggml_type type, const void* w, const void* q8, float* dst, int k, int n,
                  int columns, cudaStream_t stream);
// GGML's SwiGLU of two F32 [n, columns] products into dst (the kernel a
// GLU's decomposition adds after its two products). False if refused.
bool UpstreamSwiGlu(const float* gate, const float* up, float* dst, int n, int columns,
                    cudaStream_t stream);
}  // namespace llmp::benchmarks::vecq

#endif  // LLMP_BENCHMARKS_VECQ_UPSTREAM_H_
