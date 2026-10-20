// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: MIT AND Apache-2.0

#include "components/attention_core.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <limits>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "components/defs.h"
#include "format/encoding.h"
#include "format/tensor_format.h"
#include "ir/builder.h"
#include "ir/graph.h"
#include "ir/op.h"

namespace llmp::components::defs {

using format::Encoding;

ir::RopeParams MakeRopeParams(ir::RopeMode mode, std::uint32_t dims, std::uint32_t offset,
                              std::uint32_t default_context, float base, float freq_scale,
                              const YarnParams& yarn, std::array<std::uint32_t, 4> sections,
                              bool inverse, float default_beta_fast, float default_beta_slow) {
  if (!(yarn.factor > 1.0F)) {
    return {.mode = mode,
            .inverse = inverse,
            .dims = dims,
            .offset = offset,
            .original_context = default_context,
            .base = base,
            .scale = freq_scale,
            .ext_factor = 0.0F,
            .attn_factor = 1.0F,
            .beta_fast = default_beta_fast,
            .beta_slow = default_beta_slow,
            .sections = sections};
  }
  const float scale = 1.0F / yarn.factor;
  const float attn =
      yarn.inverse_attn_scale ? 1.0F / (1.0F + (0.1F * std::log(1.0F / scale))) : 1.0F;
  return {.mode = mode,
          .inverse = inverse,
          .dims = dims,
          .offset = offset,
          .original_context = yarn.context,
          .base = base,
          .scale = scale,
          .ext_factor = 1.0F,
          .attn_factor = attn,
          .beta_fast = yarn.beta_fast,
          .beta_slow = yarn.beta_slow,
          .sections = sections};
}

ir::TensorId EmitHeadNorm(ir::Builder& b, ir::TensorId heads, float eps, ir::TensorId norm_weight) {
  const ir::TensorId normed = b.RmsNorm(heads, eps);
  return norm_weight.valid() ? b.Mul(normed, norm_weight) : normed;
}

ir::TensorId EmitHeadNormRope(ir::Builder& b, ir::TensorId heads, float eps,
                              ir::TensorId norm_weight, ir::TensorId positions, ir::TensorId freqs,
                              const ir::RopeParams& rope, bool apply_norm) {
  const ir::TensorId in = apply_norm ? EmitHeadNorm(b, heads, eps, norm_weight) : heads;
  return b.Rope(in, positions, freqs, rope);
}

ir::TensorId EmitInterleavedHeadSlice(ir::Builder& b, ir::TensorId full, std::uint64_t dim,
                                      std::uint64_t stride_dim, std::uint64_t heads,
                                      std::uint64_t rows, std::uint64_t offset_elems) {
  const format::Shape shape = Shape({dim, heads, rows});
  return b.Cont(b.View(full, shape,
                       {4, stride_dim * 4, stride_dim * heads * 4, stride_dim * heads * 4 * rows},
                       offset_elems * 4));
}

ir::TensorId EmitSigmoidOutputGate(ir::Builder& b, ir::TensorId attended, ir::TensorId q_full,
                                   std::uint64_t kd, std::uint64_t vd, std::uint64_t heads,
                                   std::uint64_t rows) {
  const std::uint64_t stride = kd + vd;
  const format::Shape gate_shape = Shape({vd, heads, rows});
  const ir::TensorId gate =
      b.Cont(b.View(q_full, gate_shape,
                    {4, stride * 4, stride * heads * 4, stride * heads * 4 * rows}, kd * 4),
             Shape({vd * heads, rows}));
  return b.Mul(attended, b.Unary(gate, ir::UnaryFn::kSigmoid));
}

ir::TensorId EmitKvCacheStore(ir::Builder& b, ir::TensorId state, ir::TensorId values,
                              ir::TensorId store_cells, Encoding kv_encoding) {
  if (kv_encoding == Encoding::kQ8_0 && b.tensor(values).format.encoding != Encoding::kF32) {
    values = b.Convert(values, Encoding::kF32);
  }
  return b.CacheStore(state, values, store_cells);
}

ir::TensorId EmitKvCacheView(ir::Builder& b, ir::TensorId cache, std::uint64_t dim,
                             std::uint64_t kv_heads, std::uint64_t read_cells, bool direct_3d_f16) {
  if (direct_3d_f16 && b.tensor(cache).format.encoding != Encoding::kQ8_0) {
    const std::uint64_t el = 2;
    const std::uint64_t kv = dim * kv_heads;
    return b.View(cache, Shape({dim, read_cells, kv_heads}),
                  {el, kv * el, dim * el, dim * el * kv_heads}, 0);
  }
  const format::Shape shape = Shape({dim, kv_heads, read_cells});
  auto format = b.tensor(cache).format;
  format.shape = shape;
  return b.Permute(b.View(cache, shape, ir::NaturalStrides(format), 0), {0, 2, 1, 3});
}

ir::TensorId EmitSegmentQueries(ir::Builder& b, ir::TensorId q, std::uint64_t kd,
                                std::uint64_t heads, const SegmentRows& seg, bool elide_single,
                                bool rank4_view) {
  if (elide_single) {
    return b.Permute(q, {0, 2, 1, 3});
  }
  const ir::TensorDesc qd = Desc(b, q);
  if (rank4_view) {
    const ir::TensorId view = b.View(q, Shape({kd, heads, seg.rows, 1}),
                                     {qd.nb[0], qd.nb[1], qd.nb[2], qd.nb[2] * seg.rows},
                                     std::uint64_t{seg.first_row} * qd.nb[2]);
    return b.Permute(view, {0, 2, 1, 3});
  }
  const format::Shape q_shape = Shape({kd, heads, seg.rows});
  const ir::TensorId view = b.View(q, q_shape, ViewStrides(qd.nb[0], qd.nb[1], qd.nb[2], q_shape),
                                   std::uint64_t{seg.first_row} * qd.nb[2]);
  return b.Permute(view, {0, 2, 1, 3});
}

ir::TensorId EmitSegmentAttention(ir::Builder& b, ir::TensorId queries, ir::TensorId keys,
                                  ir::TensorId values, ir::TensorId mask, ir::TensorId sinks,
                                  const ir::AttnParams& params, std::uint64_t vd,
                                  std::uint64_t heads, std::uint64_t seg_rows,
                                  bool explicit_scores) {
  ir::TensorId attention{};
  if (explicit_scores) {
    const auto scores = b.MatMul(keys, queries, {.accumulate = ir::Accumulate::kF32});
    const auto probabilities =
        b.SoftMax(scores, b.Convert(mask, Encoding::kF32), sinks, {.scale = params.scale});
    const auto transposed = b.Cont(b.Transpose(values));
    const auto product = b.MatMul(transposed, probabilities);
    attention = b.Cont(b.Permute(product, {0, 2, 1, 3}));
  } else {
    attention = b.Attention(queries, keys, values, mask, sinks, params);
  }
  return b.Reshape(attention, Shape({vd * heads, seg_rows}));
}

ir::TensorId JoinSegmentOutputs(ir::Builder& b, std::span<const ir::TensorId> outputs) {
  ir::TensorId joined = outputs.front();
  for (std::size_t s = 1; s < outputs.size(); ++s) {
    joined = b.Concat(joined, outputs[s], 1);
  }
  return joined;
}

std::expected<void, std::string> StoreKvCacheRow(std::vector<double>& cache, std::uint64_t cell,
                                                 std::span<const double> values, std::uint64_t row,
                                                 std::uint64_t kv_heads, std::uint64_t dim,
                                                 Encoding kv_encoding, RowCodecFn store_row) {
  const auto row_width = kv_heads * dim;
  for (std::uint64_t head = 0; head < kv_heads; ++head) {
    auto dst = std::span(cache).subspan((cell * row_width) + (head * dim), dim);
    const auto src = values.subspan((row * row_width) + (head * dim), dim);
    if (kv_encoding == Encoding::kQ8_0) {
      if (auto stored = store_row(kv_encoding, src, dst); !stored) return stored;
    } else {
      for (std::size_t j = 0; j < dst.size(); ++j) dst[j] = Stored(kv_encoding, src[j]);
    }
  }
  return {};
}

void AttendCacheRow(std::span<const double> q_row, std::span<const double> kc,
                    std::span<const double> vc, std::uint64_t read_cells,
                    std::span<const std::uint8_t> visible, std::uint64_t heads,
                    std::uint64_t kv_heads, std::uint64_t kd, std::uint64_t vd, double q_scale,
                    double score_scale, double softcap, std::span<const float> sinks,
                    std::span<double> out_row) {
  const std::uint64_t group = heads / kv_heads;
  std::vector<double> score(read_cells);
  for (std::uint64_t h = 0; h < heads; ++h) {
    const std::uint64_t kh = h / group;
    const double* qh = &q_row[h * kd];
    double top =
        sinks.empty() ? -std::numeric_limits<double>::infinity() : static_cast<double>(sinks[h]);
    for (std::uint64_t c = 0; c < read_cells; ++c) {
      if (!visible[c]) {
        score[c] = -std::numeric_limits<double>::infinity();
        continue;
      }
      double dot = 0.0;
      for (std::uint64_t j = 0; j < kd; ++j) {
        dot += (qh[j] * q_scale) * kc[(c * kv_heads * kd) + (kh * kd) + j];
      }
      dot *= score_scale;
      if (softcap > 0.0) dot = softcap * std::tanh(dot / softcap);
      score[c] = dot;
      top = std::max(top, dot);
    }
    double sum = sinks.empty() ? 0.0 : std::exp(static_cast<double>(sinks[h]) - top);
    for (double& sc : score) {
      sc = std::isinf(sc) ? 0.0 : std::exp(sc - top);
      sum += sc;
    }
    double* out = &out_row[h * vd];
    if (sum == 0.0) continue;
    for (std::uint64_t c = 0; c < read_cells; ++c) {
      if (score[c] == 0.0) continue;
      const double pc = score[c] / sum;
      for (std::uint64_t j = 0; j < vd; ++j) {
        out[j] += pc * vc[(c * kv_heads * vd) + (kh * vd) + j];
      }
    }
  }
}

void AttendGatheredCellsRow(
    std::span<const double> q_row,
    std::span<const std::pair<const std::vector<double>*, std::uint64_t>> cells,
    std::uint64_t heads, std::uint64_t dim, double score_scale, std::span<const float> sinks,
    std::span<double> out_row) {
  for (std::uint64_t h = 0; h < heads; ++h) {
    const double* qh = &q_row[h * dim];
    const double sink =
        sinks.empty() ? -std::numeric_limits<double>::infinity() : static_cast<double>(sinks[h]);
    std::vector<double> scores;
    scores.reserve(cells.size());
    double top = sink;
    for (const auto& [cache, c] : cells) {
      double dot = 0.0;
      for (std::uint64_t j = 0; j < dim; ++j) dot += qh[j] * (*cache)[(c * dim) + j];
      scores.push_back(dot * score_scale);
      top = std::max(top, scores.back());
    }
    double sum = sinks.empty() ? 0.0 : std::exp(sink - top);
    for (double& sc : scores) {
      sc = std::exp(sc - top);
      sum += sc;
    }
    double* out = &out_row[h * dim];
    if (sum == 0.0) continue;
    for (std::size_t i = 0; i < cells.size(); ++i) {
      const auto& [cache, c] = cells[i];
      for (std::uint64_t j = 0; j < dim; ++j) {
        out[j] += (scores[i] / sum) * (*cache)[(c * dim) + j];
      }
    }
  }
}

void ApplySigmoidOutputGateRef(std::span<double> attended, std::span<const double> qg,
                               std::uint64_t rows, std::uint64_t heads, std::uint64_t kd,
                               std::uint64_t vd) {
  const std::uint64_t stride = kd + vd;
  for (std::uint64_t r = 0; r < rows; ++r) {
    for (std::uint64_t h = 0; h < heads; ++h) {
      for (std::uint64_t i = 0; i < vd; ++i) {
        const double g = qg[(r * heads * stride) + (h * stride) + kd + i];
        attended[(((r * heads) + h) * vd) + i] *= 1.0 / (1.0 + std::exp(-g));
      }
    }
  }
}

}  // namespace llmp::components::defs
