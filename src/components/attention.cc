// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// attention.dense, version 1: dense grouped-query attention over each
// request's own cache, with its norms. Pre- or sandwich-normed (Gemma's
// post-attention norm before the residual add); Q, K and V products; an
// optional per-head RMS norm of Q and K (Gemma 3); NEOX or normal pair
// rotation at the instance's base and frequency scale; queries scaled by
// `scale` after the rotation and attention run at scale 1 (as today's
// builders scale them); an optional logit softcap; a global layer's cache
// holds a cell per position, a sliding layer's (window > 0) a ring.
// K and V are F16 state; attention accumulates in F32 or wider.
//
// The fragment emits what kernels/ggml/gemma2_graph.cc and gemma3_graph.cc
// record, in their order (GraphOrder over their outputs): the input norm;
// Q's product, reshape, norm, rotation and scale; V's product and reshape;
// K's product, reshape, norm and rotation; then per segment the K store
// (the rows reshape first), the V store, and attention over views of the
// segment's caches; then the segments' outputs joined, the output
// product, the post norm and the residual add.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <expected>
#include <format>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "base/check.h"
#include "components/component.h"
#include "components/defs.h"
#include "components/step.h"
#include "format/encoding.h"
#include "ir/graph.h"
#include "ir/op.h"
#include "model/state_contract.h"

namespace llmp::components::defs {
namespace {

using format::Encoding;

constexpr std::array<std::string_view, 2> kPlacements{"sandwich", "pre"};
constexpr std::array<std::string_view, 2> kRotations{"neox", "normal"};

constexpr std::array kParams{
    ParamSpec{.name = "placement", .type = ParamType::kName, .choices = kPlacements},
    ParamSpec{.name = "eps", .type = ParamType::kReal, .least = 0.0, .most = 1.0},
    ParamSpec{.name = "heads", .type = ParamType::kInt, .least = 1, .most = 1024},
    ParamSpec{.name = "kv_heads", .type = ParamType::kInt, .least = 1, .most = 1024},
    ParamSpec{.name = "key_dim", .type = ParamType::kInt, .least = 1, .most = 1024},
    ParamSpec{.name = "value_dim", .type = ParamType::kInt, .least = 1, .most = 1024},
    // Queries' scale, applied after the rotation.
    ParamSpec{.name = "scale", .type = ParamType::kReal, .least = 1e-30, .most = 1e6},
    // A logit softcap: cap · tanh(score / cap); 0: none.
    ParamSpec{.name = "softcap",
              .type = ParamType::kReal,
              .required = false,
              .least = 0.0,
              .most = 1e6,
              .fallback = 0.0},
    // The positions a query sees, its own included; 0: all (a global layer).
    ParamSpec{.name = "window", .type = ParamType::kInt, .least = 0, .most = 1 << 30},
    ParamSpec{.name = "rotation", .type = ParamType::kName, .choices = kRotations},
    ParamSpec{.name = "rope_dims", .type = ParamType::kInt, .least = 2, .most = 1024},
    ParamSpec{.name = "rope_base", .type = ParamType::kReal, .least = 1e-30, .most = 1e12},
    // The frequency scale (1 / a linear scaling's factor).
    ParamSpec{.name = "rope_scale", .type = ParamType::kReal, .least = 1e-30, .most = 1e6},
    ParamSpec{.name = "qk_norm",
              .type = ParamType::kBool,
              .required = false,
              .least = 0,
              .most = 1,
              .fallback = 0.0},
};

enum : std::uint16_t { kNorm, kQ, kK, kV, kOutput, kPostNorm, kQNorm, kKNorm };
constexpr std::array kRoles{
    WeightRole{.name = "norm", .encodings = kF32Only},
    WeightRole{.name = "q"},
    WeightRole{.name = "k"},
    WeightRole{.name = "v"},
    WeightRole{.name = "output"},
    WeightRole{.name = "post_norm", .encodings = kF32Only},
    WeightRole{.name = "q_norm", .encodings = kF32Only},
    WeightRole{.name = "k_norm", .encodings = kF32Only},
};

// The state's encoding (D-108: F16 by default; a lossy one is a later
// per-alias setting).
constexpr Encoding kKvEncoding = Encoding::kF16;

struct Dims {
  std::uint64_t width, heads, kv_heads, kd, vd, rope_dims;
  std::uint32_t window;
  bool sandwich, qk_norm, neox;
  float eps, scale, softcap, base, freq_scale;
};
Dims DimsOf(const Instance& in) {
  const Params& p = in.params;
  return {.width = in.stage.width,
          .heads = static_cast<std::uint64_t>(p.Int("heads")),
          .kv_heads = static_cast<std::uint64_t>(p.Int("kv_heads")),
          .kd = static_cast<std::uint64_t>(p.Int("key_dim")),
          .vd = static_cast<std::uint64_t>(p.Int("value_dim")),
          .rope_dims = static_cast<std::uint64_t>(p.Int("rope_dims")),
          .window = static_cast<std::uint32_t>(p.Int("window")),
          .sandwich = p.Name("placement") == "sandwich",
          .qk_norm = p.Flag("qk_norm"),
          .neox = p.Name("rotation") == "neox",
          .eps = F32(p.Real("eps")),
          .scale = F32(p.Real("scale")),
          .softcap = F32(p.Real("softcap")),
          .base = F32(p.Real("rope_base")),
          .freq_scale = F32(p.Real("rope_scale"))};
}

std::expected<std::vector<RoleShape>, std::string> Shapes(const Instance& in) {
  const Dims d = DimsOf(in);
  if (d.heads % d.kv_heads != 0) return std::unexpected("heads are not a multiple of kv_heads");
  if (d.rope_dims > d.kd || d.rope_dims % 2 != 0) {
    return std::unexpected("rope_dims is odd or past key_dim");
  }
  std::vector<RoleShape> out{Role(kNorm, {d.width}), Role(kQ, {d.width, d.heads * d.kd}),
                             Role(kK, {d.width, d.kv_heads * d.kd}),
                             Role(kV, {d.width, d.kv_heads * d.vd}),
                             Role(kOutput, {d.heads * d.vd, d.width})};
  if (d.sandwich) out.push_back(Role(kPostNorm, {d.width}));
  if (d.qk_norm) {
    out.push_back(Role(kQNorm, {d.kd}));
    out.push_back(Role(kKNorm, {d.kd}));
  }
  return out;
}

model::StateTensorContract Kv(std::string name, std::uint64_t width, std::uint32_t window) {
  const bool ring = window != 0;
  // The format layer's cell size (model::CellBytes): a lossy encoding's
  // whole blocks once an alias opts in (D-108); F16's elements today.
  const auto cell = model::CellBytes(kKvEncoding, width);
  base::Check(cell.has_value(), "an F16 cell of any width");
  return {.name = std::move(name),
          .kind = ring ? model::StateKind::kRing : model::StateKind::kKv,
          .encoding = kKvEncoding,
          .cell_bytes = *cell,
          .cells = ring ? model::CellRule::kRing : model::CellRule::kPerPosition,
          .window_or_ratio = window,
          // A ring's rows are overwritten in place: no verify may write it.
          .rollback = ring ? model::Rollback::kNone : model::Rollback::kTruncate};
}

model::StateContract State(const Instance& in) {
  const Dims d = DimsOf(in);
  return {.tensors = {Kv("k", d.kv_heads * d.kd, d.window), Kv("v", d.kv_heads * d.vd, d.window)}};
}

std::vector<std::pair<std::uint16_t, std::uint32_t>> Masks(const Instance& in) {
  return {{0, DimsOf(in).window}};
}

constexpr std::array kBatchClasses{
    BatchClass{.op = ir::Op::kRmsNorm, .batch = ir::Batch::kJoined},
    BatchClass{.op = ir::Op::kMatMul, .batch = ir::Batch::kJoined},
    BatchClass{.op = ir::Op::kRope, .batch = ir::Batch::kJoined},
    BatchClass{.op = ir::Op::kMaskBuild, .batch = ir::Batch::kPerOwner},
    BatchClass{.op = ir::Op::kCacheStore, .batch = ir::Batch::kPerOwner},
    BatchClass{.op = ir::Op::kAttention, .batch = ir::Batch::kPerOwner},
};

// By heads: Q, K and V by their output rows (whole heads, a query head's
// group with its KV head), the K and V caches by their cells' heads, the
// output product by its input columns; its partial sums reduce before the
// post norm.
constexpr std::array kHeadRoles{RoleSplit{.role = kQ, .dim = 1}, RoleSplit{.role = kK, .dim = 1},
                                RoleSplit{.role = kV, .dim = 1},
                                RoleSplit{.role = kOutput, .dim = 0}};
constexpr std::array kHeadState{StateSplit{.tensor = 0, .dim = 0},
                                StateSplit{.tensor = 1, .dim = 0}};
constexpr std::array kSplits{SplitOption{
    .kind = SplitKind::kHeads,
    .collective = ir::Op::kAllReduce,
    // Whole KV heads per device.
    .units =
        [](const Instance& in) { return static_cast<std::uint32_t>(in.params.Int("kv_heads")); },
    .bytes_per_step = [](const SplitStep& s) -> std::uint64_t {
      return s.instance->stage.width * s.rows * 4;
    },
    .weights = kHeadRoles,
    .state = kHeadState}};

constexpr std::array<std::string_view, 7> kPatterns{
    "rms_norm_mul",    "rms_norm_mul_rope", "rope_store",      "grouped_stores",
    "owner_attention", "prepared_products", "rms_norm_mul_add"};

ir::RopeParams Rotation(const Instance& in, const Dims& d) {
  return {.mode = d.neox ? ir::RopeMode::kNeox : ir::RopeMode::kNormal,
          .dims = static_cast<std::uint32_t>(d.rope_dims),
          .original_context = in.stage.context,
          .base = d.base,
          .scale = d.freq_scale,
          .ext_factor = 0.0F,
          .attn_factor = 1.0F,
          .beta_fast = 32.0F,
          .beta_slow = 1.0F};
}

// The cell rule of a state tensor of this instance.
CellKey CellsOf(const Fragment& f, std::size_t tensor) {
  return {.capacity = f.b.tensor(f.State(0, tensor)).ne(1), .ring = DimsOf(f.instance).window != 0};
}

std::expected<FragmentOut, std::string> Emit(Fragment& f) {
  ir::Builder& b = f.b;
  const Dims d = DimsOf(f.instance);
  const auto w = [&](std::uint16_t role) { return f.Weight(role); };
  const std::uint64_t rows = f.step.rows();
  const ir::TensorId x = f.stream;
  ir::TensorId normalized = b.Mul(b.RmsNorm(x, d.eps), w(kNorm));
  const ir::RopeParams rope = Rotation(f.instance, d);
  const ir::TensorId positions = f.step.Positions();
  const auto head_norm = [&](ir::TensorId t, std::uint16_t role) {
    return d.qk_norm ? b.Mul(b.RmsNorm(t, d.eps), w(role)) : t;
  };
  ir::TensorId q{};
  if (f.stream_needed) {
    q = b.Reshape(b.MatMul(w(kQ), normalized), Shape({d.kd, d.heads, rows}));
    q = b.Rope(head_norm(q, kQNorm), positions, {}, rope);
    q = b.Scale(q, d.scale);
  }
  const ir::TensorId v = b.Reshape(b.MatMul(w(kV), normalized), Shape({d.vd, d.kv_heads, rows}));
  ir::TensorId k = b.Reshape(b.MatMul(w(kK), normalized), Shape({d.kd, d.kv_heads, rows}));
  k = b.Rope(head_norm(k, kKNorm), positions, {}, rope);

  const std::uint64_t k_width = d.kd * d.kv_heads;
  const std::uint64_t v_width = d.vd * d.kv_heads;
  ir::TensorId k_rows{};
  ir::TensorId v_rows{};
  const CellKey cells = CellsOf(f, 0);
  const std::uint64_t el = format::Info(kKvEncoding).element_bits / 8;
  std::vector<ir::TensorId> outputs;
  for (std::size_t s = 0; s < f.step.segments().size(); ++s) {
    const SegmentRows& seg = f.step.segments()[s];
    const auto slice = [&](ir::TensorId t, std::uint64_t width) {
      const auto nb = b.tensor(t).nb;
      const format::Shape shape = Shape({width, seg.rows});
      return b.View(t, shape, ViewStrides(nb[0], nb[1], 0, shape),
                    std::uint64_t{seg.first_row} * nb[1]);
    };
    const ir::TensorId store = f.step.StoreCells(s, cells);
    if (!k_rows.valid()) k_rows = b.Reshape(k, Shape({k_width, rows}));
    b.CacheStore(f.State(s, 0), slice(k_rows, k_width), store);
    if (!v_rows.valid()) v_rows = b.Reshape(v, Shape({v_width, rows}));
    b.CacheStore(f.State(s, 1), slice(v_rows, v_width), store);
    if (!f.stream_needed) continue;
    const std::uint64_t read = f.step.ReadCells(s, cells);
    auto mask = f.step.Mask(s, cells, d.window);
    if (!mask) return std::unexpected(mask.error());
    b.SetOwner(static_cast<std::uint8_t>(s));
    const auto qn = b.tensor(q).nb;
    const format::Shape q_shape = Shape({d.kd, d.heads, seg.rows});
    const ir::TensorId queries =
        b.Permute(b.View(q, q_shape, ViewStrides(qn[0], qn[1], qn[2], q_shape),
                         std::uint64_t{seg.first_row} * qn[2]),
                  {0, 2, 1, 3});
    const auto cache_view = [&](ir::TensorId cache, std::uint64_t dim, std::uint64_t width) {
      const format::Shape shape = Shape({dim, d.kv_heads, read});
      return b.Permute(b.View(cache, shape, ViewStrides(el, dim * el, width * el, shape), 0),
                       {0, 2, 1, 3});
    };
    const ir::TensorId keys = cache_view(f.State(s, 0), d.kd, k_width);
    const ir::TensorId values = cache_view(f.State(s, 1), d.vd, v_width);
    const ir::TensorId attention =
        b.Attention(queries, keys, values, *mask, {},
                    {.scale = 1.0F, .softcap = d.softcap, .accumulate = ir::Accumulate::kF32});
    outputs.push_back(b.Reshape(attention, Shape({d.vd * d.heads, seg.rows})));
    b.SetJoined();
  }
  if (!f.stream_needed) return FragmentOut{.stream = x};
  ir::TensorId joined = outputs.front();
  for (std::size_t s = 1; s < outputs.size(); ++s) joined = b.Concat(joined, outputs[s], 1);
  ir::TensorId projected = b.MatMul(w(kOutput), joined);
  // Narrowed (the head's rows only follow): the projection's and the
  // residual's output rows, gathered after the product (the builders'
  // narrow_final: the last layer's FFN and the head at the outputs' width).
  const auto narrowed = [&](ir::TensorId t) {
    return f.narrow ? b.GetRows(t, f.step.OutputRows()) : t;
  };
  projected = narrowed(projected);
  if (!d.sandwich) return FragmentOut{.stream = b.Add(projected, narrowed(x))};
  const ir::TensorId normed = b.Mul(b.RmsNorm(projected, d.eps), w(kPostNorm));
  return FragmentOut{.stream = b.Add(normed, narrowed(x))};
}

// ------------------------------------------------------------- reference

// Rotates `heads` heads of `dim` values at `position` in place.
void Rotate(std::span<double> row, std::uint64_t heads, const Dims& d, std::int64_t position) {
  const std::uint64_t pairs = d.rope_dims / 2;
  for (std::uint64_t h = 0; h < heads; ++h) {
    double* x = &row[h * d.kd];
    for (std::uint64_t i = 0; i < pairs; ++i) {
      const std::uint64_t a = d.neox ? i : 2 * i;
      const std::uint64_t c = d.neox ? i + pairs : (2 * i) + 1;
      const double theta =
          RopeAngle(d.base, static_cast<std::uint32_t>(d.rope_dims), d.freq_scale, i, position);
      const double x0 = x[a];
      const double x1 = x[c];
      x[a] = (x0 * std::cos(theta)) - (x1 * std::sin(theta));
      x[c] = (x0 * std::sin(theta)) + (x1 * std::cos(theta));
    }
  }
}

std::expected<void, std::string> Reference(RefIO& io) {
  const Instance& in = *io.instance;
  const Dims d = DimsOf(in);
  const std::uint64_t width = d.width;
  const std::uint64_t rows = io.rows();
  if (io.stream.size() != width * rows || io.state.size() != io.segments.size() ||
      io.state_cells.size() != 2) {
    return std::unexpected("attention reference: stream or state of another shape");
  }
  const auto weight = [&](std::uint16_t role) -> std::span<const float> {
    return io.weights.at(role);
  };
  std::vector<double> normalized = io.stream;
  RmsRows(normalized, width, d.eps, weight(kNorm));
  std::vector<double> q = Product(weight(kQ), width, d.heads * d.kd, normalized);
  std::vector<double> k = Product(weight(kK), width, d.kv_heads * d.kd, normalized);
  const std::vector<double> v = Product(weight(kV), width, d.kv_heads * d.vd, normalized);
  if (d.qk_norm) {
    RmsRows(q, d.kd, d.eps, weight(kQNorm));
    RmsRows(k, d.kd, d.eps, weight(kKNorm));
  }
  const std::uint64_t group = d.heads / d.kv_heads;
  std::vector<double> attended(rows * d.heads * d.vd, 0.0);
  std::uint64_t first = 0;
  for (std::size_t s = 0; s < io.segments.size(); ++s) {
    const RefSegment& seg = io.segments[s];
    auto& kc = io.state[s].at(0);
    auto& vc = io.state[s].at(1);
    const std::uint64_t capacity = io.state_cells[0];
    const bool ring = d.window != 0;
    const std::uint64_t end = std::uint64_t{seg.past} + seg.rows;
    // The model sees every position of a row's window; a ring that would
    // overwrite one before the row attends cannot hold the step.
    if (ring && end > capacity && seg.rows + std::uint64_t{d.window} > capacity + 1) {
      return std::unexpected("attention reference: a ring too small for the rows and window");
    }
    // Rotate, then store every row before any row attends (the step's
    // stores precede its attention).
    for (std::uint64_t i = 0; i < seg.rows; ++i) {
      const std::uint64_t r = first + i;
      const auto pos = static_cast<std::int64_t>(seg.past + i);
      Rotate(std::span(q).subspan(r * d.heads * d.kd, d.heads * d.kd), d.heads, d, pos);
      Rotate(std::span(k).subspan(r * d.kv_heads * d.kd, d.kv_heads * d.kd), d.kv_heads, d, pos);
      const std::uint64_t cell = ring ? (seg.past + i) % capacity : seg.past + i;
      if (cell >= capacity) return std::unexpected("attention reference: a row past the cache");
      for (std::uint64_t j = 0; j < d.kv_heads * d.kd; ++j) {
        kc[(cell * d.kv_heads * d.kd) + j] = Stored(kKvEncoding, k[(r * d.kv_heads * d.kd) + j]);
      }
      for (std::uint64_t j = 0; j < d.kv_heads * d.vd; ++j) {
        vc[(cell * d.kv_heads * d.vd) + j] = Stored(kKvEncoding, v[(r * d.kv_heads * d.vd) + j]);
      }
    }
    const std::uint64_t read = std::min<std::uint64_t>(capacity, seg.read_cells);
    std::vector<double> score(read);
    for (std::uint64_t i = 0; i < seg.rows; ++i) {
      const std::uint64_t r = first + i;
      const auto p = static_cast<std::int64_t>(seg.past + i);
      for (std::uint64_t h = 0; h < d.heads; ++h) {
        const std::uint64_t kh = h / group;
        const double* qh = &q[(r * d.heads * d.kd) + (h * d.kd)];
        double top = -std::numeric_limits<double>::infinity();
        for (std::uint64_t c = 0; c < read; ++c) {
          // The position the cell holds once the step's rows are stored.
          std::int64_t held = -1;
          if (c < end) {
            held = ring ? static_cast<std::int64_t>(c + (((end - 1 - c) / capacity) * capacity))
                        : static_cast<std::int64_t>(c);
          }
          const bool seen = held >= 0 && held <= p && (d.window == 0 || p - held < d.window);
          if (!seen) {
            score[c] = -std::numeric_limits<double>::infinity();
            continue;
          }
          double dot = 0.0;
          for (std::uint64_t j = 0; j < d.kd; ++j) {
            dot += qh[j] * static_cast<double>(d.scale) *
                   kc[(c * d.kv_heads * d.kd) + (kh * d.kd) + j];
          }
          if (d.softcap > 0.0F) dot = d.softcap * std::tanh(dot / d.softcap);
          score[c] = dot;
          top = std::max(top, dot);
        }
        double sum = 0.0;
        for (double& sc : score) {
          sc = std::isinf(sc) ? 0.0 : std::exp(sc - top);
          sum += sc;
        }
        double* out = &attended[(r * d.heads * d.vd) + (h * d.vd)];
        if (sum == 0.0) continue;  // sees no cell: zero (op.h)
        for (std::uint64_t c = 0; c < read; ++c) {
          if (score[c] == 0.0) continue;
          const double pc = score[c] / sum;
          for (std::uint64_t j = 0; j < d.vd; ++j) {
            out[j] += pc * vc[(c * d.kv_heads * d.vd) + (kh * d.vd) + j];
          }
        }
      }
    }
    first += seg.rows;
  }
  std::vector<double> projected = Product(weight(kOutput), d.heads * d.vd, width, attended);
  if (d.sandwich) RmsRows(projected, width, d.eps, weight(kPostNorm));
  for (std::size_t i = 0; i < projected.size(); ++i) io.stream[i] += projected[i];
  return {};
}

}  // namespace

const ComponentDef& AttentionDense() {
  static const ComponentDef def{
      .kind = "attention.dense",
      .version = 1,
      .narrows = true,
      .params = kParams,
      .roles = kRoles,
      .shapes = Shapes,
      .state = State,
      .batch_classes = kBatchClasses,
      .batching = nullptr,
      .splits = kSplits,
      .patterns = kPatterns,
      .masks = Masks,
      .emit = Emit,
      .reference = Reference,
  };
  return def;
}

}  // namespace llmp::components::defs
