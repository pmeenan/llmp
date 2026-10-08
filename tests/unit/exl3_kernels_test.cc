// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The locked ExLlamaV3 GEMM kernels on a GB10 (label `gpu`;
// docs/backend-proof.md, P1): every kernel of the tables
// (exl3_tables_test.cc) loads from SASS built for the device (sm_121 on a
// GB10; on a discrete GPU the build targets, label `gpu-discrete` too,
// its own, D-082), and each kernel the P0 launch record names uses the
// registers per thread recorded there (GB10 only). No kernel is launched:
// llmpalooza's launchers come with the first native EXL3 linear.

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <string>

#include "exl3_tables.h"

namespace {

using llmp::tests::exl3::AllKernels;
using llmp::tests::exl3::kRates;
using llmp::tests::exl3::RateTables;

// A GEMM kernel of the P0 launch record
// (docs/experiments/backend-proof-p0/exl3-launch.json): its rate, output
// type and shape, and its registers per thread there. The record names
// exl3_gemm_kernel<bits, false, fp32, 1, SHAPE>, e.g.
// <4, false, false, 1, 16, 32, 128, 4, 3> is K=4, FP16, shape 2.
struct RecordedKernel {
  int bits;
  bool fp32;
  int shape;
  int registers;
};

constexpr std::array<RecordedKernel, 14> kRecorded{{
    {4, false, 1, 128},
    {4, false, 2, 124},
    {4, false, 3, 128},
    {4, false, 4, 170},
    {4, true, 2, 126},
    {4, true, 3, 128},
    {5, false, 3, 128},
    {5, false, 4, 172},
    {5, true, 2, 128},
    {6, false, 2, 128},
    {6, false, 3, 128},
    {8, false, 1, 128},
    {8, false, 2, 125},
    {8, false, 3, 128},
}};

// The threads per block each shape launches with (EXL3_GEMM_BLOCKDIM).
constexpr std::array<int, 5> kBlockDim{EXL3_GEMM_BLOCKDIM};

bool HaveDevice() {
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

// Device 0's compute capability as the SASS number, such as 121 for 12.1.
int DeviceArchitecture() {
  cudaDeviceProp prop{};
  return cudaGetDeviceProperties(&prop, 0) == cudaSuccess ? (prop.major * 10) + prop.minor : 0;
}

TEST(Exl3KernelsGpuTest, EveryKernelLoadsFromTheDevicesSass) {
  ASSERT_TRUE(HaveDevice());
  const int architecture = DeviceArchitecture();
  ASSERT_GT(architecture, 0);
  for (const void* kernel : AllKernels()) {
    cudaFuncAttributes attributes{};
    ASSERT_EQ(cudaFuncGetAttributes(&attributes, kernel), cudaSuccess);
    EXPECT_EQ(attributes.binaryVersion, architecture);
    EXPECT_EQ(attributes.ptxVersion, architecture);
    EXPECT_GE(attributes.maxThreadsPerBlock, 256);
  }
}

TEST(Exl3KernelsGpuTest, RecordedKernelsUseTheRecordedRegisters) {
  ASSERT_TRUE(HaveDevice());
  for (const RecordedKernel& recorded : kRecorded) {
    SCOPED_TRACE("K=" + std::to_string(recorded.bits) + (recorded.fp32 ? " FP32" : " FP16") +
                 " shape " + std::to_string(recorded.shape));
    const RateTables* rate = nullptr;
    for (const RateTables& candidate : kRates) {
      if (candidate.bits == recorded.bits) rate = &candidate;
    }
    ASSERT_NE(rate, nullptr);
    const fp_exl3_gemm_kernel* table = recorded.fp32 ? rate->gemm_fp32 : rate->gemm_fp16;
    const void* kernel = reinterpret_cast<const void*>(table[recorded.shape]);
    cudaFuncAttributes attributes{};
    ASSERT_EQ(cudaFuncGetAttributes(&attributes, kernel), cudaSuccess);
    EXPECT_EQ(attributes.numRegs, recorded.registers);
    EXPECT_GE(attributes.maxThreadsPerBlock, kBlockDim[static_cast<std::size_t>(recorded.shape)]);
  }
}

}  // namespace
