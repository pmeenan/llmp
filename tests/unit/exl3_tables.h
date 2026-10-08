// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The tables of EXL3 GEMM kernels that the locked ExLlamaV3 compilation
// units define (comp_units/exl3_comp_unit_<K>_cb1.cu), for the tests.

#ifndef LLMP_TESTS_UNIT_EXL3_TABLES_H_
#define LLMP_TESTS_UNIT_EXL3_TABLES_H_

#include <cuda_fp16.h>

#include <array>
#include <cstdint>
#include <set>

#include "kernels/exl3/launch_contract.h"
#include "quant/exl3_kernel_map.cuh"

// The tables' declarations for codebook 1 (mcg), the only one the build
// compiles. comp_units/exl3_comp_unit_<K>.cuh declares all three codebooks'.
EXL3_KERNEL_EXTERNS_CB(4, 1)
EXL3_KERNEL_EXTERNS_CB(5, 1)
EXL3_KERNEL_EXTERNS_CB(6, 1)
EXL3_KERNEL_EXTERNS_CB(8, 1)

namespace llmp::tests::exl3 {

// One rate's tables for the mcg codebook (cb 1), indexed by tile shape;
// index 0 is upstream's unused slot.
struct RateTables {
  int bits;
  fp_exl3_gemm_kernel* gemm_fp16;
  fp_exl3_gemm_kernel* gemm_fp32;
  fp_exl3_mgemm_kernel* mgemm_fp16;
  fp_exl3_mgemm_kernel* mgemm_fp32;
};

inline const std::array<RateTables, 4> kRates{{
    {4, tfp_exl3_gemm_kernel_fp16_b4_cb1, tfp_exl3_gemm_kernel_fp32_b4_cb1,
     tfp_exl3_mgemm_kernel_fp16_b4_cb1, tfp_exl3_mgemm_kernel_fp32_b4_cb1},
    {5, tfp_exl3_gemm_kernel_fp16_b5_cb1, tfp_exl3_gemm_kernel_fp32_b5_cb1,
     tfp_exl3_mgemm_kernel_fp16_b5_cb1, tfp_exl3_mgemm_kernel_fp32_b5_cb1},
    {6, tfp_exl3_gemm_kernel_fp16_b6_cb1, tfp_exl3_gemm_kernel_fp32_b6_cb1,
     tfp_exl3_mgemm_kernel_fp16_b6_cb1, tfp_exl3_mgemm_kernel_fp32_b6_cb1},
    {8, tfp_exl3_gemm_kernel_fp16_b8_cb1, tfp_exl3_gemm_kernel_fp32_b8_cb1,
     tfp_exl3_mgemm_kernel_fp16_b8_cb1, tfp_exl3_mgemm_kernel_fp32_b8_cb1},
}};

constexpr int kShapes = EXL3_GEMM_NUM_SHAPES;

// Every kernel in the tables, as the runtime identifies it: its host stub.
inline std::set<const void*> AllKernels() {
  std::set<const void*> kernels;
  for (const RateTables& rate : kRates) {
    for (int shape = 1; shape <= kShapes; ++shape) {
      kernels.insert(reinterpret_cast<const void*>(rate.gemm_fp16[shape]));
      kernels.insert(reinterpret_cast<const void*>(rate.gemm_fp32[shape]));
      kernels.insert(reinterpret_cast<const void*>(rate.mgemm_fp16[shape]));
      kernels.insert(reinterpret_cast<const void*>(rate.mgemm_fp32[shape]));
    }
  }
  return kernels;
}

}  // namespace llmp::tests::exl3

#endif  // LLMP_TESTS_UNIT_EXL3_TABLES_H_
