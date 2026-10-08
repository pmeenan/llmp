// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The locked ExLlamaV3 kernels link into llmpalooza's build (every CUDA
// profile, no GPU; docs/backend-proof.md, P1 and P3): upstream's
// compilation units for the mcg codebook at K = 4, 5, 6 and 8 each define
// their GEMM and multi-GEMM tables, for FP16 and FP32 outputs, with a
// kernel for each of tile shapes 1 to 4; llmpalooza's instance unit
// (llmp_exl3_kernels.h) finds them and defines the GEMV, reconstruction,
// Hadamard and bias-add instances the native linear launches, and nothing
// else; and the host checks' constants (validate.h) are the headers'.
// exl3_kernels_test.cc loads them on a GB10.

#include "exl3_tables.h"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <set>
#include <string>

#include "kernels/exl3/validate.h"
#include "llmp_exl3_kernels.h"
#include "quant/exl3_devctx.cuh"

namespace {

using llmp::tests::exl3::AllKernels;
using llmp::tests::exl3::kRates;
using llmp::tests::exl3::kShapes;
using llmp::tests::exl3::RateTables;

TEST(Exl3KernelTablesTest, EveryRateHasAKernelPerShape) {
  static_assert(kShapes == 4);
  for (const RateTables& rate : kRates) {
    SCOPED_TRACE("K=" + std::to_string(rate.bits));
    EXPECT_EQ(rate.gemm_fp16[0], nullptr);
    EXPECT_EQ(rate.gemm_fp32[0], nullptr);
    EXPECT_EQ(rate.mgemm_fp16[0], nullptr);
    EXPECT_EQ(rate.mgemm_fp32[0], nullptr);
    for (int shape = 1; shape <= kShapes; ++shape) {
      SCOPED_TRACE("shape " + std::to_string(shape));
      EXPECT_NE(rate.gemm_fp16[shape], nullptr);
      EXPECT_NE(rate.gemm_fp32[shape], nullptr);
      EXPECT_NE(rate.mgemm_fp16[shape], nullptr);
      EXPECT_NE(rate.mgemm_fp32[shape], nullptr);
    }
  }
  // 4 rates x 4 shapes x {GEMM, multi-GEMM} x {FP16, FP32}, all distinct.
  EXPECT_EQ(AllKernels().size(), 64U);
}

TEST(Exl3KernelTablesTest, TheLookupsFindTheTablesAndTheInstances) {
  namespace k = llmp_exl3;
  for (const RateTables& rate : kRates) {
    for (int shape = 1; shape <= kShapes; ++shape) {
      EXPECT_EQ(k::GemmKernel(rate.bits, shape, false),
                reinterpret_cast<const void*>(rate.gemm_fp16[shape]));
      EXPECT_EQ(k::GemmKernel(rate.bits, shape, true),
                reinterpret_cast<const void*>(rate.gemm_fp32[shape]));
      EXPECT_EQ(k::MultiGemmKernel(rate.bits, shape, false),
                reinterpret_cast<const void*>(rate.mgemm_fp16[shape]));
      EXPECT_EQ(k::MultiGemmKernel(rate.bits, shape, true),
                reinterpret_cast<const void*>(rate.mgemm_fp32[shape]));
    }
    EXPECT_EQ(k::GemmKernel(rate.bits, 0, false), nullptr);
    EXPECT_EQ(k::GemmKernel(rate.bits, kShapes + 1, false), nullptr);
    EXPECT_NE(k::ReconstructKernel(rate.bits), nullptr);
    EXPECT_NE(k::ReconstructHadKernel(rate.bits), nullptr);
    EXPECT_NE(k::ReconstructKernel(rate.bits), k::ReconstructHadKernel(rate.bits));
  }
  for (const int bits : {1, 2, 3, 7, 9}) {
    EXPECT_EQ(k::GemmKernel(bits, 2, false), nullptr);
    EXPECT_EQ(k::ReconstructKernel(bits), nullptr);
    EXPECT_EQ(k::ReconstructHadKernel(bits), nullptr);
  }
  // The GEMV: K = 4 only, both outputs, both row modes, both configurations.
  std::set<const void*> gemv;
  for (const bool fp32 : {false, true}) {
    for (int mmode = 0; mmode < 2; ++mmode) {
      for (int cfg = 0; cfg < 2; ++cfg) {
        gemv.insert(k::GemvKernel(4, fp32, mmode, cfg));
        EXPECT_EQ(k::GemvKernel(5, fp32, mmode, cfg), nullptr);
      }
    }
  }
  EXPECT_EQ(gemv.size(), 8U);
  EXPECT_FALSE(gemv.contains(nullptr));
  EXPECT_EQ(k::GemvKernel(4, false, 2, 0), nullptr);
  // The three Hadamard variants the linear launches, and no other.
  EXPECT_NE(k::HadamardKernel(false, true, false), nullptr);
  EXPECT_NE(k::HadamardKernel(false, false, true), nullptr);
  EXPECT_NE(k::HadamardKernel(true, false, true), nullptr);
  EXPECT_EQ(k::HadamardKernel(true, true, false), nullptr);
  EXPECT_EQ(k::HadamardKernel(false, false, false), nullptr);
  EXPECT_EQ(k::HadamardKernel(false, true, true), nullptr);
  EXPECT_NE(k::AddKernelHhh(), nullptr);
}

TEST(Exl3KernelTablesTest, TheHostChecksUseTheHeadersConstants) {
  namespace e = llmp::kernels::exl3;
  EXPECT_EQ(e::kShapes, EXL3_GEMM_NUM_SHAPES);
  constexpr std::array<int, 5> kTileK{EXL3_GEMM_TILESIZE_K};
  constexpr std::array<int, 5> kTileN{EXL3_GEMM_TILESIZE_N};
  constexpr std::array<int, 5> kBlockDim{EXL3_GEMM_BLOCKDIM};
  EXPECT_EQ(e::kTileK, kTileK);
  EXPECT_EQ(e::kTileN, kTileN);
  EXPECT_EQ(e::kBlockDim, kBlockDim);
  EXPECT_EQ(e::kGemmSharedMemory, llmp_exl3::GemmSharedMemory());
  EXPECT_EQ(e::kGemvMaxRows, llmp_exl3::GemvMaxRows());
  EXPECT_EQ(e::kLockBytes, (MAX_TILES_C + (MAX_BARRIERS * 2) + MOE_SCHED_INTS) * sizeof(int));
}

}  // namespace
