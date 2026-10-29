// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#ifndef LLMP_KERNELS_GGML_MEASURED_RANKS_H_
#define LLMP_KERNELS_GGML_MEASURED_RANKS_H_

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "execution/rank_data.h"

namespace llmp::kernels::ggml {

// Coarse fusion rows (execution/rank_data.h FusionRank), each naming the
// report of its measurement (docs/experiments/fusion-ranks/README.md, the
// protocol). A row names its implementation's kernel version;
// tests/unit/fusion_ranks_test.cc fails when a row names an implementation
// or version this backend no longer declares.
//
// llmp.vecq's fused GLU on the GB10 against its decomposition as the
// planner plans it: two products, each by the device rows below (GGML's
// MMVQ in its band, llmp.vecq otherwise), then GGML's SwiGLU. From
// benchmarks/vecq_bench.cc --sweep (docs/experiments/vecq-columns/
// README.md: eight runs, each arm's fastest, every product over prepared
// Q8_1 rows, at K x N 4096 x 4096 and 2560 x 10240). The faster is chosen
// wherever the gap is beyond the sweep's repeat noise: a row holds the
// column counts at which the decomposition was more than 5% faster
// (execution::ClassifyFusionLatency's band) on both shapes, and its times
// are the band's narrowest gap's. Every other encoding and column count
// keeps the fusion (faster, or within 5% on a shape).
inline execution::FusionRank VecQGluRow(format::Encoding encoding, std::uint64_t rows_min,
                                        std::uint64_t rows_max, std::uint64_t fused_ns,
                                        std::uint64_t decomposition_ns) {
  return {.device_class = "sm_121",
          .op = ir::Fused(ir::FusedKind::kMatMulGlu),
          .implementation = "llmp.vecq.glu",
          .version = 2,
          .encoding = encoding,
          .layout = format::Layout::kGgmlBlocks,
          .rows_min = rows_min,
          .rows_max = rows_max,
          .fused_ns = fused_ns,
          .decomposition_ns = decomposition_ns,
          .samples = 8,
          .provenance = "docs/experiments/vecq-columns/README.md"};
}
inline std::vector<execution::FusionRank> MeasuredFusionRanks() {
  using format::Encoding;
  return {
      VecQGluRow(Encoding::kQ4_0, 6, 8, 95120, 89980),
      VecQGluRow(Encoding::kQ4_1, 7, 8, 101830, 92560),
      VecQGluRow(Encoding::kQ5_0, 6, 8, 115500, 105280),
      VecQGluRow(Encoding::kQ5_1, 7, 8, 123010, 109050),
      VecQGluRow(Encoding::kQ8_0, 5, 8, 170610, 162000),
      VecQGluRow(Encoding::kQ4_K, 5, 8, 113860, 91090),
      VecQGluRow(Encoding::kQ5_K, 5, 8, 135790, 109810),
      VecQGluRow(Encoding::kQ6_K, 4, 8, 150990, 133610),
      VecQGluRow(Encoding::kIQ2_XXS, 7, 8, 306360, 289170),
      VecQGluRow(Encoding::kIQ3_XXS, 6, 8, 109840, 104140),
      VecQGluRow(Encoding::kIQ4_NL, 6, 8, 97640, 87860),
  };
}

// GGML's MMVQ ahead of llmp.vecq for the GB10's dense products of these
// encodings at these column counts (both its forms: over the F32 input, and
// over Q8_1 rows another product shares): from benchmarks/vecq_bench.cc
// --sweep (docs/experiments/vecq-columns/README.md), each band from the
// first column count at which MMVQ was more than 10% faster on both shapes
// through 8. (Q4_K's single column, MMVQ 20% faster alone, measured slower
// end to end: Gemma 4 31B's C1 decode; docs/experiments/vecq-columns.) Encodings llmp.mmvq.prepared
// does not take (the IQ2/IQ3 types, Q2_K, Q3_K, IQ4_XS, MXFP4) keep llmp.vecq, whose shared
// preparation they would otherwise lose.
inline void AddMmvqRows(std::vector<execution::DeviceMeasuredRank>& out, format::Encoding encoding,
                        std::uint64_t rows_min, std::uint64_t rows_max) {
  for (const std::string_view name : {"ggml.mul_mat.mmvq", "llmp.mmvq.prepared"}) {
    out.push_back({.device_class = "sm_121",
                   .device_model = "NVIDIA GB10",
                   .arch = 121,
                   .multiprocessors = 48,
                   .op = ir::Primitive(ir::Op::kMatMul),
                   .encoding = encoding,
                   .layout = format::Layout::kGgmlBlocks,
                   .rows_min = rows_min,
                   .rows_max = rows_max,
                   .implementation = std::string(name),
                   .rank = 0,
                   .params = {}});
  }
}
inline std::vector<execution::DeviceMeasuredRank> MeasuredMmvqRanks() {
  std::vector<execution::DeviceMeasuredRank> out;
  AddMmvqRows(out, format::Encoding::kQ4_0, 4, 8);
  AddMmvqRows(out, format::Encoding::kQ4_1, 6, 8);
  AddMmvqRows(out, format::Encoding::kQ5_0, 5, 8);
  AddMmvqRows(out, format::Encoding::kQ5_1, 5, 8);
  AddMmvqRows(out, format::Encoding::kQ8_0, 5, 8);
  AddMmvqRows(out, format::Encoding::kQ4_K, 3, 8);
  AddMmvqRows(out, format::Encoding::kQ5_K, 4, 8);
  AddMmvqRows(out, format::Encoding::kQ6_K, 2, 8);
  AddMmvqRows(out, format::Encoding::kIQ4_NL, 4, 8);
  return out;
}

// The backend owns these measurements; unrelated registries do not inherit them.
inline execution::RankData MeasuredRanks() {
  // The two exact rows from the frozen protocol in
  // docs/experiments/decode-ladder-r6.md; then MeasuredMmvqRanks' bands.
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
  for (auto& row : MeasuredMmvqRanks()) data.records.push_back(std::move(row));
  data.fusions = MeasuredFusionRanks();
  return data;
}
}  // namespace llmp::kernels::ggml
#endif  // LLMP_KERNELS_GGML_MEASURED_RANKS_H_
