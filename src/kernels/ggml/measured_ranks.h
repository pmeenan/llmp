// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#ifndef LLMP_KERNELS_GGML_MEASURED_RANKS_H_
#define LLMP_KERNELS_GGML_MEASURED_RANKS_H_

#include "execution/rank_data.h"

namespace llmp::kernels::ggml {
std::vector<execution::FusionComparison> MeasuredFusionComparisons();

// The backend owns these measurements; unrelated registries do not inherit them.
inline execution::RankData MeasuredRanks() {
  // Filled only from the frozen protocol in docs/experiments/decode-ladder-r6.md.
  auto data = execution::RankData{{
      {.device_class = "sm_121",
       .device_model = "NVIDIA GB10",
       .arch = 121,
       .multiprocessors = 48,
       .op = ir::Primitive(ir::Op::kMatMul),
       .encoding = format::Encoding::kMxfp8,
       .layout = format::Layout::kSplit,
       .rows_min = 1,
       .rows_max = 1,
       .k = 2560,
       .n = 10240,
       .implementation = "llmp.mxfp8.mul_mat_vec",
       .rank = 0,
       .params = {}},
      {.device_class = "sm_121",
       .device_model = "NVIDIA GB10",
       .arch = 121,
       .multiprocessors = 48,
       .op = ir::Primitive(ir::Op::kGetRows),
       .encoding = format::Encoding::kNvfp4,
       .layout = format::Layout::kRowInterleaved,
       .rows_min = 16,
       .rows_max = 16,
       .k = 160,
       .n = 1048576,
       .implementation = "llmp.nvfp4.get_rows",
       .rank = 0,
       .params = {.tensor_scale = true}},
  }};
  data.fusions = MeasuredFusionComparisons();
  return data;
}
}  // namespace llmp::kernels::ggml
#endif  // LLMP_KERNELS_GGML_MEASURED_RANKS_H_
