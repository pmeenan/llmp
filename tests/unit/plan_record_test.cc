// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The executed-plan recording's JSON lines (tests/support/plan_record.h),
// in every profile:
// - the writer produces plan_record_sample.txt for its events, the file
//   docs/experiments/backend-proof-p2/plan_compare.py's tests read, so the
//   two sides of the format are held to one sample;
// - copies, memsets and cuBLAS calls are written in the fields the
//   converter reads, and names are escaped;
// - NVCC's per-file hashes are masked in kernel names, and nothing else.

#include "plan_record.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace {

using llmp::test_support::Chunk;
using llmp::test_support::Event;
using llmp::test_support::EventKind;

// A kernel launch of the sample: the bridge's decode step, on stream 0x10.
Event Launch(std::string_view name, std::array<unsigned, 3> grid, std::array<unsigned, 3> block,
             std::size_t shared, int registers) {
  Event event;
  event.kind = EventKind::kKernel;
  event.name = name;
  event.api = "cudaLaunchKernelExC";
  event.grid = grid;
  event.block = block;
  event.shared = shared;
  event.registers = registers;
  event.stream = reinterpret_cast<const void*>(0x10);  // NOLINT(performance-no-int-to-ptr)
  return event;
}

TEST(PlanRecordTest, TheWriterProducesTheSample) {
  constexpr std::string_view kNorm =
      "_Z12rms_norm_f32ILi256ELb1ELb0ELb0EEvPKfPfilllfS1_lll5uint3S3_S3_S3_S1_lllS3_S3_S3_S3_f";
  constexpr std::string_view kMmvf =
      "_Z13mul_mat_vec_fI6__halfS0_Li1ELi224ELb1ELb0EEvPKT_PKfPKi31ggml_cuda_mm_fusion_args_"
      "devicePfi5uint3iiiSA_iiiSA_iiii";
  constexpr std::string_view kRope =
      "_Z9rope_neoxILb1ELb0EffEvPKT1_PT2_iiiiiiiiiiiPKifff14rope_corr_dimsfPKfPKlib";
  constexpr std::string_view kRopeHalf =
      "_Z9rope_neoxILb1ELb0Ef6__halfEvPKT1_PT2_iiiiiiiiiiiPKifff14rope_corr_dimsfPKfPKlib";
  constexpr std::string_view kSetRows =
      "_Z10k_set_rowsIfl6__halfEvPKT_PKT0_PT1_llllllllllllll5uint3S9_S9_S9_S9_";
  const std::vector<Event> events = {
      Launch(kNorm, {1, 1, 1}, {256, 1, 1}, 128, 40),
      Launch(kMmvf, {896, 1, 1}, {224, 1, 1}, 256, 42),
      Launch(kRope, {14, 1, 1}, {1, 256, 1}, 0, 18),
      Launch(kMmvf, {128, 1, 1}, {224, 1, 1}, 256, 42),
      Launch(kMmvf, {128, 1, 1}, {224, 1, 1}, 256, 42),
      Launch(kRopeHalf, {2, 1, 1}, {1, 256, 1}, 0, 18),
      Launch(kSetRows, {1, 1, 1}, {256, 1, 1}, 0, 18),
  };
  std::string written = llmp::test_support::HeaderLine(
      "sample: the first seven launches of control-fused's decode step", {});
  written +=
      llmp::test_support::ChunkLine(Chunk{.evaluation = 1, .chunk = 1, .rows = 1, .n_past = 32});
  for (const Event& event : events) {
    written += llmp::test_support::EventLine(event);
  }
  written += llmp::test_support::EndChunkLine();
  EXPECT_EQ(written, llmp::test_support::PlanRecordSample());
}

TEST(PlanRecordTest, CopiesMemsetsAndCublasCallsCarryTheirFields) {
  Event copy;
  copy.kind = EventKind::kCopy;
  copy.api = "driver";
  copy.bytes = 3584;
  EXPECT_EQ(llmp::test_support::EventLine(copy),
            "{\"type\":\"memcpy\",\"kind\":\"driver\",\"bytes\":3584,\"stream\":\"0x0\"}\n");
  Event memset;
  memset.kind = EventKind::kMemset;
  memset.bytes = 608;
  memset.value = 0;
  EXPECT_EQ(llmp::test_support::EventLine(memset),
            "{\"type\":\"memset\",\"bytes\":608,\"value\":0,\"stream\":\"0x0\"}\n");
  Event gemm;
  gemm.kind = EventKind::kCublas;
  gemm.name = "cublasGemmEx";
  gemm.shape = {896, 32, 896, 0};
  EXPECT_EQ(llmp::test_support::EventLine(gemm),
            "{\"type\":\"cublas\",\"function\":\"cublasGemmEx\",\"m\":896,\"n\":32,\"k\":896}\n");
  gemm.name = "cublasGemmBatchedEx";
  gemm.shape = {256, 32, 64, 14};
  EXPECT_EQ(llmp::test_support::EventLine(gemm),
            "{\"type\":\"cublas\",\"function\":\"cublasGemmBatchedEx\",\"m\":256,\"n\":32,\"k\":64,"
            "\"batch\":14}\n");
  const std::string header =
      llmp::test_support::HeaderLine("a \"quoted\"\\source\n", {{"libcublas.so.13", "/a/b"}});
  EXPECT_EQ(header,
            "{\"type\":\"header\",\"format\":\"llmp-plan-record/1\",\"source\":\"a "
            "\\\"quoted\\\"\\\\source\\u000a\",\"loaded\":{\"libcublas.so.13\":\"/a/b\"}}\n");
}

TEST(PlanRecordTest, OnlyNvccsPerFileHashesAreMasked) {
  EXPECT_EQ(llmp::test_support::NormalizedKernelName(
                "_Z11k_bin_bcastIXadL_ZN42_INTERNAL_d5c41c42_11_binbcast_cu_6840010b6op_addEffEEv"),
            "_Z11k_bin_bcastIXadL_ZN42_INTERNAL_xxxxxxxx_11_binbcast_cu_xxxxxxxx6op_addEffEEv");
  constexpr std::string_view kPlain = "_Z16k_get_rows_floatIffEvPKT_PKiPT0_ll5uint3mmmmmmmmm";
  EXPECT_EQ(llmp::test_support::NormalizedKernelName(kPlain), kPlain);
  // A truncated token is left as it is past the point it ends.
  EXPECT_EQ(llmp::test_support::NormalizedKernelName("_INTERNAL_d5c4"), "_INTERNAL_d5c4");
}

}  // namespace
