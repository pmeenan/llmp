// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: MIT AND Apache-2.0

// Shared attention IR emission and scalar reference core for src/components:
// per-head RMS norm and RoPE, KV cache store and view, per-segment query
// slicing, masked attention (fused or explicit scores, with optional softcap,
// sinks, and sparse bounds), sigmoid output gating, and scalar reference
// evaluation.

#ifndef LLMP_COMPONENTS_ATTENTION_CORE_H_
#define LLMP_COMPONENTS_ATTENTION_CORE_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "components/component.h"
#include "components/step.h"
#include "format/encoding.h"
#include "ir/builder.h"
#include "ir/op.h"

namespace llmp::components::defs {

struct YarnParams {
  float factor = 1.0F;
  std::uint32_t context = 0;
  float beta_fast = 32.0F;
  float beta_slow = 1.0F;
  bool inverse_attn_scale = false;
};

ir::RopeParams MakeRopeParams(ir::RopeMode mode, std::uint32_t dims, std::uint32_t offset,
                              std::uint32_t default_context, float base, float freq_scale,
                              const YarnParams& yarn, std::array<std::uint32_t, 4> sections = {},
                              bool inverse = false, float default_beta_fast = 32.0F,
                              float default_beta_slow = 1.0F);

ir::TensorId EmitHeadNorm(ir::Builder& b, ir::TensorId heads, float eps,
                          ir::TensorId norm_weight = {});

ir::TensorId EmitHeadNormRope(ir::Builder& b, ir::TensorId heads, float eps,
                              ir::TensorId norm_weight, ir::TensorId positions, ir::TensorId freqs,
                              const ir::RopeParams& rope, bool apply_norm = true);

ir::TensorId EmitInterleavedHeadSlice(ir::Builder& b, ir::TensorId full, std::uint64_t dim,
                                      std::uint64_t stride_dim, std::uint64_t heads,
                                      std::uint64_t rows, std::uint64_t offset_elems);

ir::TensorId EmitSigmoidOutputGate(ir::Builder& b, ir::TensorId attended, ir::TensorId q_full,
                                   std::uint64_t kd, std::uint64_t vd, std::uint64_t heads,
                                   std::uint64_t rows);

ir::TensorId EmitKvCacheStore(ir::Builder& b, ir::TensorId state, ir::TensorId values,
                              ir::TensorId store_cells, format::Encoding kv_encoding);

ir::TensorId EmitKvCacheView(ir::Builder& b, ir::TensorId cache, std::uint64_t dim,
                             std::uint64_t kv_heads, std::uint64_t read_cells,
                             bool direct_3d_f16 = false);

ir::TensorId EmitSegmentQueries(ir::Builder& b, ir::TensorId q, std::uint64_t kd,
                                std::uint64_t heads, const SegmentRows& seg,
                                bool elide_single = false, bool rank4_view = false);

ir::TensorId EmitSegmentAttention(ir::Builder& b, ir::TensorId queries, ir::TensorId keys,
                                  ir::TensorId values, ir::TensorId mask, ir::TensorId sinks,
                                  const ir::AttnParams& params, std::uint64_t vd,
                                  std::uint64_t heads, std::uint64_t seg_rows,
                                  bool explicit_scores = false);

ir::TensorId JoinSegmentOutputs(ir::Builder& b, std::span<const ir::TensorId> outputs);

// Scalar reference helpers.
using RowCodecFn = std::expected<void, std::string> (*)(format::Encoding, std::span<const double>,
                                                        std::span<double>);

std::expected<void, std::string> StoreKvCacheRow(std::vector<double>& cache, std::uint64_t cell,
                                                 std::span<const double> values, std::uint64_t row,
                                                 std::uint64_t kv_heads, std::uint64_t dim,
                                                 format::Encoding kv_encoding,
                                                 RowCodecFn store_row);

void AttendCacheRow(std::span<const double> q_row, std::span<const double> kc,
                    std::span<const double> vc, std::uint64_t read_cells,
                    std::span<const std::uint8_t> visible, std::uint64_t heads,
                    std::uint64_t kv_heads, std::uint64_t kd, std::uint64_t vd, double q_scale,
                    double score_scale, double softcap, std::span<const float> sinks,
                    std::span<double> out_row);

void AttendGatheredCellsRow(
    std::span<const double> q_row,
    std::span<const std::pair<const std::vector<double>*, std::uint64_t>> cells,
    std::uint64_t heads, std::uint64_t dim, double score_scale, std::span<const float> sinks,
    std::span<double> out_row);

void ApplySigmoidOutputGateRef(std::span<double> attended, std::span<const double> qg,
                               std::uint64_t rows, std::uint64_t heads, std::uint64_t kd,
                               std::uint64_t vd);

}  // namespace llmp::components::defs

#endif  // LLMP_COMPONENTS_ATTENTION_CORE_H_
