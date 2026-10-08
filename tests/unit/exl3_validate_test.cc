// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The EXL3 launchers' host checks (src/kernels/exl3/validate.h) and
// upstream's GEMV choice (upstream_gemv.h), in every profile: what each
// launch refuses before anything is queued, the co-resident bound on
// cooperative grids (launch_contract.h), upstream's path thresholds and
// slices, and exl3_gemv_cfg's heuristic on the fixtures' shapes. No device
// is touched; addresses are never dereferenced.

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <tuple>
#include <vector>

#include "expected_error.h"
#include "kernels/exl3/upstream_gemv.h"
#include "kernels/exl3/validate.h"

namespace {

namespace exl3 = llmp::kernels::exl3;
using exl3::Output;
using llmp::test_support::FailedCode;

constexpr std::uint64_t kMiB = 1ULL << 20;

// A q_proj-shaped linear at 4 bits, its tensors 1 MiB apart.
exl3::Weights Q() {
  return {.trellis = 1 * kMiB, .suh = 2 * kMiB, .svh = 3 * kMiB, .k = 896, .n = 896, .bits = 4};
}

exl3::LinearOperands Packed(int m, Output output = Output::kF16) {
  return {
      .weights = Q(), .x = 4 * kMiB, .a_had = 5 * kMiB, .y = 6 * kMiB, .output = output, .m = m};
}

constexpr std::uint64_t kLocks = 16 * kMiB;
constexpr int kCoresident = 48;

TEST(Exl3ValidateTest, WeightsNeedACompiledRateAndAlignedTensors) {
  EXPECT_TRUE(exl3::CheckWeights(Q()).has_value());
  EXPECT_EQ(exl3::TrellisBytes(Q()), 896ULL / 16 * 896 / 16 * 16 * 4 * 2);
  for (const int bits : {4, 5, 6, 8}) {
    EXPECT_TRUE(exl3::RateCompiled(bits));
  }
  for (const int bits : {0, 1, 2, 3, 7, 9}) {
    exl3::Weights w = Q();
    w.bits = bits;
    EXPECT_FALSE(exl3::CheckWeights(w).has_value()) << bits;
  }
  exl3::Weights w = Q();
  w.k = 896 + 16;  // a multiple of 16, not of 128
  EXPECT_FALSE(exl3::CheckWeights(w).has_value());
  w = Q();
  w.n = 0;
  EXPECT_FALSE(exl3::CheckWeights(w).has_value());
  w = Q();
  w.n = exl3::kMaxDimension + 128;
  EXPECT_FALSE(exl3::CheckWeights(w).has_value());
  w = Q();
  w.trellis += 8;  // 8-byte aligned: the kernels copy 16-byte vectors
  EXPECT_FALSE(exl3::CheckWeights(w).has_value());
  w = Q();
  w.suh += 4;
  EXPECT_FALSE(exl3::CheckWeights(w).has_value());
  w = Q();
  w.svh = 0;
  EXPECT_FALSE(exl3::CheckWeights(w).has_value());
  w = Q();
  w.svh += 8;  // 8 bytes is the side vectors' minimum
  EXPECT_TRUE(exl3::CheckWeights(w).has_value());
}

TEST(Exl3ValidateTest, GemmPlansMustFitTheShapeAndBeCoresident) {
  const exl3::LinearOperands o = Packed(16);
  EXPECT_TRUE(exl3::CheckGemm(o, {.shape = 2, .blocks = 14}, kCoresident, kLocks).has_value());
  // Shape 3's 256-column tiles do not divide 896 columns.
  EXPECT_FALSE(exl3::CheckGemm(o, {.shape = 3, .blocks = 14}, kCoresident, kLocks).has_value());
  EXPECT_FALSE(exl3::CheckGemm(o, {.shape = 0, .blocks = 14}, kCoresident, kLocks).has_value());
  EXPECT_FALSE(exl3::CheckGemm(o, {.shape = 5, .blocks = 14}, kCoresident, kLocks).has_value());
  EXPECT_FALSE(exl3::CheckGemm(o, {.shape = 2, .blocks = 0}, kCoresident, kLocks).has_value());
  // A grid that does not fit on the device at once would hang or trap.
  EXPECT_FALSE(exl3::CheckGemm(o, {.shape = 2, .blocks = 49}, kCoresident, kLocks).has_value());
  EXPECT_TRUE(exl3::CheckGemm(o, {.shape = 2, .blocks = 48}, kCoresident, kLocks).has_value());
  // More blocks than split-K slices: k/v at shape 2 has 896/32 * 128/128.
  exl3::LinearOperands kv = o;
  kv.weights.n = 128;
  EXPECT_TRUE(exl3::CheckGemm(kv, {.shape = 2, .blocks = 28}, kCoresident, kLocks).has_value());
  EXPECT_FALSE(exl3::CheckGemm(kv, {.shape = 2, .blocks = 29}, kCoresident, kLocks).has_value());
}

TEST(Exl3ValidateTest, GemmOperandsMustBeAlignedAndDisjoint) {
  const exl3::GemmPlan plan{.shape = 2, .blocks = 14};
  exl3::LinearOperands o = Packed(16, Output::kF32);
  EXPECT_TRUE(exl3::CheckGemm(o, plan, kCoresident, kLocks).has_value());
  o.y += 8;  // an F32 output is transformed in 16-byte vectors
  EXPECT_FALSE(exl3::CheckGemm(o, plan, kCoresident, kLocks).has_value());
  o = Packed(16);
  o.y += 8;  // an F16 output needs 8
  EXPECT_TRUE(exl3::CheckGemm(o, plan, kCoresident, kLocks).has_value());
  o = Packed(16);
  o.x += 4;
  EXPECT_FALSE(exl3::CheckGemm(o, plan, kCoresident, kLocks).has_value());
  o = Packed(16);
  o.a_had += 8;
  EXPECT_FALSE(exl3::CheckGemm(o, plan, kCoresident, kLocks).has_value());
  // The scratch over the input, the output over the weights, either over the
  // lock slots.
  o = Packed(16);
  o.a_had = o.x + 64;
  EXPECT_FALSE(exl3::CheckGemm(o, plan, kCoresident, kLocks).has_value());
  o = Packed(16);
  o.y = o.weights.trellis;
  EXPECT_FALSE(exl3::CheckGemm(o, plan, kCoresident, kLocks).has_value());
  o = Packed(16);
  o.y = kLocks + 1024;
  EXPECT_FALSE(exl3::CheckGemm(o, plan, kCoresident, kLocks).has_value());
  o = Packed(16);
  EXPECT_FALSE(exl3::CheckGemm(o, plan, kCoresident, 0).has_value());
  o = Packed(0);
  EXPECT_FALSE(exl3::CheckGemm(o, plan, kCoresident, kLocks).has_value());
  o = Packed(exl3::kMaxRows + 1);
  EXPECT_FALSE(exl3::CheckGemm(o, plan, kCoresident, kLocks).has_value());
  EXPECT_EQ(exl3::ScratchBytes(Packed(16)), 16ULL * 896 * 2);
}

TEST(Exl3ValidateTest, TheGemvTakesFourBitsUpToEightRows) {
  const exl3::GemvPlan plan{.config = 0, .blocks = 28};
  EXPECT_TRUE(exl3::CheckGemv(Packed(1), plan, 96, kLocks).has_value());
  EXPECT_TRUE(exl3::CheckGemv(Packed(8), plan, 96, kLocks).has_value());
  EXPECT_FALSE(exl3::CheckGemv(Packed(9), plan, 96, kLocks).has_value());
  exl3::LinearOperands o = Packed(1);
  o.weights.bits = 5;
  EXPECT_FALSE(exl3::CheckGemv(o, plan, 96, kLocks).has_value());
  EXPECT_FALSE(exl3::CheckGemv(Packed(1), {.config = 2, .blocks = 28}, 96, kLocks).has_value());
  // One block per 32 columns (narrow) or 64 (wide), at most.
  EXPECT_FALSE(exl3::CheckGemv(Packed(1), {.config = 0, .blocks = 29}, 96, kLocks).has_value());
  EXPECT_TRUE(exl3::CheckGemv(Packed(1), {.config = 1, .blocks = 14}, 96, kLocks).has_value());
  EXPECT_FALSE(exl3::CheckGemv(Packed(1), {.config = 1, .blocks = 15}, 96, kLocks).has_value());
  EXPECT_FALSE(exl3::CheckGemv(Packed(1), plan, 27, kLocks).has_value());
  EXPECT_EQ(exl3::GemvThreads(0), 512);
  EXPECT_EQ(exl3::GemvThreads(1), 256);
}

exl3::MultiLinearOperands GateUp(int m) {
  exl3::Weights gate{
      .trellis = 1 * kMiB, .suh = 7 * kMiB, .svh = 8 * kMiB, .k = 896, .n = 4864, .bits = 4};
  exl3::Weights up = gate;
  up.trellis = 9 * kMiB;
  up.suh = 10 * kMiB;
  up.svh = (10 * kMiB) + 4096;
  return {.first = gate,
          .second = up,
          .trellis_table = 11 * kMiB,
          .suh_table = (11 * kMiB) + 16,
          .svh_table = (11 * kMiB) + 32,
          .written = exl3::MultiGemmTables(gate, up),
          .x = 12 * kMiB,
          .a_had = 13 * kMiB,
          .y = 14 * kMiB,
          .output = Output::kF32,
          .m = m};
}

TEST(Exl3ValidateTest, MultiGemmPlansBoundTheWholeGrid) {
  const exl3::MultiGemmPlan plan{.shape = 3, .blocks = 24, .concurrency = 2};
  EXPECT_TRUE(exl3::CheckMultiGemm(GateUp(1), plan, kCoresident, kLocks).has_value());
  // 24 × 2 blocks must all be co-resident.
  EXPECT_FALSE(exl3::CheckMultiGemm(GateUp(1), plan, 47, kLocks).has_value());
  EXPECT_FALSE(exl3::CheckMultiGemm(GateUp(1), {.shape = 3, .blocks = 24, .concurrency = 3},
                                    kCoresident, kLocks)
                   .has_value());
  EXPECT_FALSE(exl3::CheckMultiGemm(GateUp(1), {.shape = 3, .blocks = 24, .concurrency = 0},
                                    kCoresident, kLocks)
                   .has_value());
  exl3::MultiLinearOperands o = GateUp(1);
  o.second.bits = 5;
  EXPECT_FALSE(exl3::CheckMultiGemm(o, plan, kCoresident, kLocks).has_value());
  o = GateUp(1);
  o.second.n = 896;
  EXPECT_FALSE(exl3::CheckMultiGemm(o, plan, kCoresident, kLocks).has_value());
  o = GateUp(1);
  o.suh_table += 4;
  EXPECT_FALSE(exl3::CheckMultiGemm(o, plan, kCoresident, kLocks).has_value());
  o = GateUp(1);
  o.y = o.trellis_table;
  EXPECT_FALSE(exl3::CheckMultiGemm(o, plan, kCoresident, kLocks).has_value());
  // Two slabs of transformed input.
  EXPECT_EQ(exl3::ScratchBytes(GateUp(8)), 2ULL * 8 * 896 * 2);
}

// BP-P5: the kernel follows the addresses in its device tables, which no
// host check reads. A linear that moves after its tables were written, with
// its operands updated and its tables not, is refused before launch.
TEST(Exl3ValidateTest, MultiGemmRefusesTablesWrittenForOtherTensors) {
  const exl3::MultiGemmPlan plan{.shape = 3, .blocks = 24, .concurrency = 2};
  const exl3::MultiLinearOperands o = GateUp(1);
  EXPECT_EQ(o.written, (std::array<std::uint64_t, 6>{o.first.trellis, o.second.trellis, o.first.suh,
                                                     o.second.suh, o.first.svh, o.second.svh}));
  ASSERT_TRUE(exl3::CheckMultiGemm(o, plan, kCoresident, kLocks).has_value());
  for (std::size_t moved = 0; moved < 6; ++moved) {
    exl3::MultiLinearOperands relocated = o;
    exl3::Weights& w = moved % 2 == 0 ? relocated.first : relocated.second;
    std::uint64_t& address = [&]() -> std::uint64_t& {
      if (moved < 2) {
        return w.trellis;
      }
      return moved < 4 ? w.suh : w.svh;
    }();
    address += 20 * kMiB;
    EXPECT_EQ(FailedCode(exl3::CheckMultiGemm(relocated, plan, kCoresident, kLocks)),
              exl3::KernelError::kRejected)
        << moved;
    relocated.written = exl3::MultiGemmTables(relocated.first, relocated.second);
    EXPECT_TRUE(exl3::CheckMultiGemm(relocated, plan, kCoresident, kLocks).has_value()) << moved;
  }
  // Nor may the record name the linears in the other order.
  exl3::MultiLinearOperands swapped = o;
  swapped.written = exl3::MultiGemmTables(o.second, o.first);
  EXPECT_FALSE(exl3::CheckMultiGemm(swapped, plan, kCoresident, kLocks).has_value());
}

TEST(Exl3ValidateTest, ReconstructionTakesWholeColumnBlocks) {
  const exl3::ReconstructOperands whole{
      .weights = Q(), .w = 20 * kMiB, .column = 0, .columns = 896, .fused = false};
  EXPECT_TRUE(exl3::CheckReconstruct(whole).has_value());
  exl3::ReconstructOperands o = whole;
  o.column = 128;
  o.columns = 768;
  EXPECT_TRUE(exl3::CheckReconstruct(o).has_value());
  o.columns = 896;  // past the matrix
  EXPECT_FALSE(exl3::CheckReconstruct(o).has_value());
  o = whole;
  o.column = 64;
  o.columns = 128;
  EXPECT_FALSE(exl3::CheckReconstruct(o).has_value());
  o = whole;
  o.columns = 0;
  EXPECT_FALSE(exl3::CheckReconstruct(o).has_value());
  o = whole;
  o.w += 8;
  o.fused = true;
  EXPECT_FALSE(exl3::CheckReconstruct(o).has_value());
  o = whole;
  o.w = Q().trellis;
  EXPECT_FALSE(exl3::CheckReconstruct(o).has_value());
}

TEST(Exl3ValidateTest, HadamardsRunInPlaceOrDisjoint) {
  exl3::HadamardOperands o{.x = 20 * kMiB,
                           .y = 20 * kMiB,
                           .scale = 1 * kMiB,
                           .rows = 145,
                           .columns = 896,
                           .type = Output::kF32,
                           .input_scale = false};
  EXPECT_TRUE(exl3::CheckHadamard(o).has_value());
  o.y = o.x + 256;  // a partial overlap
  EXPECT_FALSE(exl3::CheckHadamard(o).has_value());
  o.y = 30 * kMiB;
  EXPECT_TRUE(exl3::CheckHadamard(o).has_value());
  o.y += 8;  // F32 rows go in 16-byte vectors
  EXPECT_FALSE(exl3::CheckHadamard(o).has_value());
  o.y = 30 * kMiB;
  o.input_scale = true;  // compiled for F16 only
  EXPECT_FALSE(exl3::CheckHadamard(o).has_value());
  o.type = Output::kF16;
  EXPECT_TRUE(exl3::CheckHadamard(o).has_value());
  o.columns = 900;
  EXPECT_FALSE(exl3::CheckHadamard(o).has_value());
}

TEST(Exl3ValidateTest, BiasAddsInPlaceOrDisjoint) {
  exl3::BiasOperands o{.x = 20 * kMiB, .bias = 1 * kMiB, .y = 20 * kMiB, .rows = 8, .columns = 128};
  EXPECT_TRUE(exl3::CheckBias(o).has_value());
  o.y = o.x + 2;
  EXPECT_FALSE(exl3::CheckBias(o).has_value());
  o.y = o.x;
  o.bias = o.x + 64;
  EXPECT_FALSE(exl3::CheckBias(o).has_value());
  o.bias = kMiB + 1;
  EXPECT_FALSE(exl3::CheckBias(o).has_value());
  // The packed path's bias must survive its product: apart from a_had and
  // the lock area, which the product writes.
  const exl3::LinearOperands packed = Packed(16);
  EXPECT_TRUE(exl3::CheckBiasApart(packed, 7 * kMiB, kLocks).has_value());
  EXPECT_FALSE(exl3::CheckBiasApart(packed, packed.a_had + 64, kLocks).has_value());
  EXPECT_FALSE(exl3::CheckBiasApart(packed, kLocks + 1024, kLocks).has_value());
}

TEST(Exl3ValidateTest, ReconstructionGemmsWriteColumnsOfAWiderOutput) {
  // lm_head's second slice: 32,768 columns of 151,936.
  exl3::ReconGemmOperands o{.w = 100 * kMiB,
                            .x = 200 * kMiB,
                            .y = (300 * kMiB) + (32768ULL * 2),
                            .m = 145,
                            .k = 896,
                            .n = 32768,
                            .ldc = 151936,
                            .output = Output::kF16};
  EXPECT_TRUE(exl3::CheckReconGemm(o).has_value());
  o.ldc = 32767;
  EXPECT_FALSE(exl3::CheckReconGemm(o).has_value());
  o.ldc = 151936;
  o.x += 8;  // the pinned algorithms assume 16-byte operands
  EXPECT_FALSE(exl3::CheckReconGemm(o).has_value());
  o.x = 200 * kMiB;
  o.w = o.y + 1024;
  EXPECT_FALSE(exl3::CheckReconGemm(o).has_value());
  // Leading dimensions that leave later rows off 16 bytes.
  o.w = 100 * kMiB;
  o.k = 900;
  EXPECT_FALSE(exl3::CheckReconGemm(o).has_value());
  o.k = 896;
  o.ldc = 151940;
  EXPECT_FALSE(exl3::CheckReconGemm(o).has_value());
  o.ldc = 151936;
  EXPECT_TRUE(exl3::CheckReconGemm(o).has_value());
}

TEST(Exl3ValidateTest, ReconstructionScratchOverlapsNothingItFeeds) {
  const exl3::ReconstructedOperands o{.weights = Q(),
                                      .x = 20 * kMiB,
                                      .xh = 30 * kMiB,
                                      .w = 40 * kMiB,
                                      .y = 50 * kMiB,
                                      .output = Output::kF32,
                                      .m = 145,
                                      .bias = 0};
  EXPECT_TRUE(exl3::CheckReconstructedScratch(o, false).has_value());
  exl3::ReconstructedOperands bad = o;
  bad.w = bad.xh + 1024;  // the weights would overwrite the transformed input
  EXPECT_FALSE(exl3::CheckReconstructedScratch(bad, false).has_value());
  bad = o;
  bad.w = bad.x;  // or, fused, the input the GEMM reads
  EXPECT_FALSE(exl3::CheckReconstructedScratch(bad, true).has_value());
  bad = o;
  bad.xh = bad.y;
  EXPECT_FALSE(exl3::CheckReconstructedScratch(bad, false).has_value());
  // Fused, xh is unused and may be anything.
  bad = o;
  bad.xh = bad.x;
  EXPECT_FALSE(exl3::CheckReconstructedScratch(bad, false).has_value());
  EXPECT_TRUE(exl3::CheckReconstructedScratch(bad, true).has_value());
  // Nothing written may hold the weights, which every slice reads, or the
  // bias, read last.
  bad = o;
  bad.y = bad.weights.trellis;
  EXPECT_FALSE(exl3::CheckReconstructedScratch(bad, false).has_value());
  bad = o;
  bad.xh = bad.weights.svh;
  EXPECT_FALSE(exl3::CheckReconstructedScratch(bad, false).has_value());
  bad = o;
  bad.bias = bad.w + 256;
  EXPECT_FALSE(exl3::CheckReconstructedScratch(bad, true).has_value());
  bad = o;
  bad.bias = 60 * kMiB;
  EXPECT_TRUE(exl3::CheckReconstructedScratch(bad, false).has_value());
}

TEST(Exl3ValidateTest, UpstreamPathsAndSlices) {
  EXPECT_EQ(exl3::UpstreamPath(1), exl3::LinearPath::kPacked);
  EXPECT_EQ(exl3::UpstreamPath(144), exl3::LinearPath::kPacked);
  EXPECT_EQ(exl3::UpstreamPath(145), exl3::LinearPath::kReconstruct);
  EXPECT_EQ(exl3::UpstreamPath(1023), exl3::LinearPath::kReconstruct);
  EXPECT_EQ(exl3::UpstreamPath(1024), exl3::LinearPath::kReconstructFused);
  EXPECT_EQ(exl3::ReconstructSlices(896), (std::vector<int>{896}));
  EXPECT_EQ(exl3::ReconstructSlices(32768), (std::vector<int>{32768}));
  EXPECT_EQ(exl3::ReconstructSlices(151936), (std::vector<int>{32768, 32768, 32768, 32768, 20864}));
  exl3::Weights head = Q();
  head.n = 151936;
  EXPECT_EQ(exl3::ReconstructScratchBytes(head), 896ULL * 32768 * 2);
  EXPECT_EQ(exl3::ReconstructScratchBytes(Q()), 896ULL * 896 * 2);
}

TEST(Exl3ValidateTest, UpstreamGemvChoiceOnTheFixturesShapes) {
  EXPECT_EQ(exl3::UpstreamCcClass(12, 1), exl3::CcClass::kBlackwell);
  EXPECT_EQ(exl3::UpstreamCcClass(9, 0), exl3::CcClass::kHopper);
  EXPECT_EQ(exl3::UpstreamCcClass(8, 9), exl3::CcClass::kAda);
  EXPECT_EQ(exl3::UpstreamCcClass(8, 6), exl3::CcClass::kAmpere);
  EXPECT_EQ(exl3::UpstreamCcClass(7, 5), exl3::CcClass::kOld);
  const auto bw = exl3::CcClass::kBlackwell;
  constexpr int kMcg = exl3::kCodebookMcg;
  // Qwen2.5-0.5B at 4 bits on 48 SMs with two narrow blocks per SM: every
  // projection fits one wave of narrow blocks.
  for (const auto& [k, n, blocks] : {std::tuple{896, 896, 28}, std::tuple{896, 128, 4},
                                     std::tuple{4864, 896, 28}, std::tuple{896, 4864, 96}}) {
    const auto plan = exl3::UpstreamGemvPlan(bw, 1, k, n, 4, kMcg, 96, 48);
    EXPECT_TRUE(plan.has_value()) << k << " × " << n;
    const exl3::GemvPlan chosen = plan.value_or(exl3::GemvPlan{.config = -1, .blocks = 0});
    EXPECT_EQ(chosen.config, 0);
    EXPECT_EQ(chosen.blocks, blocks);
  }
  // Other rates, more than eight rows, or no codebook at 4 bits: the GEMM.
  EXPECT_FALSE(exl3::UpstreamGemvPlan(bw, 1, 896, 896, 5, kMcg, 96, 48).has_value());
  EXPECT_FALSE(exl3::UpstreamGemvPlan(bw, 1, 896, 151936, 8, kMcg, 96, 48).has_value());
  EXPECT_FALSE(exl3::UpstreamGemvPlan(bw, 9, 896, 896, 4, kMcg, 96, 48).has_value());
  EXPECT_TRUE(exl3::UpstreamGemvPlan(bw, 8, 896, 896, 4, 0, 96, 48).has_value());
  EXPECT_FALSE(exl3::UpstreamGemvPlan(bw, 8, 896, 896, 3, 0, 96, 48).has_value());
  // exl3_gemv_cfg's other branches: past one wave, small k stays narrow;
  // large n with k up to 4,096 goes wide; big k and n keep the GEMM.
  EXPECT_EQ(exl3::UpstreamGemvConfig(bw, 1, 2048, 8192, 4, kMcg, 16), 0);
  EXPECT_EQ(exl3::UpstreamGemvConfig(bw, 1, 4096, 16384, 4, kMcg, 16), 1);
  EXPECT_EQ(exl3::UpstreamGemvConfig(bw, 1, 8192, 16384, 4, kMcg, 16), -1);
  EXPECT_EQ(exl3::UpstreamGemvConfig(bw, 1, 8192, 16384, 2, kMcg, 16), 1);
  EXPECT_EQ(exl3::UpstreamGemvConfig(bw, 1, 8192, 16384, 3, kMcg, 16), -1);
  EXPECT_EQ(exl3::UpstreamGemvConfig(exl3::CcClass::kAda, 1, 8192, 16384, 3, kMcg, 16), 1);
  EXPECT_EQ(exl3::UpstreamGemvConfig(exl3::CcClass::kAmpere, 1, 5120, 10240, 4, kMcg, 16), 1);
  EXPECT_EQ(exl3::UpstreamGemvConfig(bw, 1, 5120, 10240, 4, kMcg, 16), -1);
  // The wide grid is bounded by the wide kernel's co-resident blocks.
  const exl3::GemvPlan wide = exl3::UpstreamGemvPlan(bw, 1, 4096, 16384, 4, kMcg, 16, 40)
                                  .value_or(exl3::GemvPlan{.config = -1, .blocks = 0});
  EXPECT_EQ(wide.config, 1);
  EXPECT_EQ(wide.blocks, 40);
}

}  // namespace
