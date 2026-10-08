// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Benchmark-only scheduling variants of llmp_ops.cu's MXFP8 GEMV.
// No product dispatch or quantization changes. The stream owns every launch.

#ifndef LLMP_BENCHMARKS_QWEN38_MXFP8_TUNE_KERNELS_H_
#define LLMP_BENCHMARKS_QWEN38_MXFP8_TUNE_KERNELS_H_

#include <cuda_runtime_api.h>

#include <cstdint>

namespace llmp::diag {

struct Mxfp8Schedule {
  int rows = 1;
  int warps = 8;
  bool column_at_a_time = false;
};

// Inputs have the production kernel's layout and alignment. Refuses invalid
// schedules/shapes before launching. Outputs retain the original FMA order.
cudaError_t Mxfp8Tune(cudaStream_t stream, const std::uint8_t* codes, const std::uint8_t* scales,
                      const float* input, float* output, int columns, int outputs, int inputs,
                      int input_stride, Mxfp8Schedule schedule);

}  // namespace llmp::diag

#endif  // LLMP_BENCHMARKS_QWEN38_MXFP8_TUNE_KERNELS_H_
