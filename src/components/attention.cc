// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: MIT AND Apache-2.0

// attention.dense, version 1: dense grouped-query attention over each
// request's own cache, with its norms. Pre- or sandwich-normed (Gemma's
// post-attention norm before the residual add); Q, K and V products; an
// optional per-head RMS norm of Q and K (Gemma 3, Gemma 4); NEOX or normal
// pair rotation at the instance's base and frequency scale, optionally
// divided by per-pair frequency factors (`rope_freqs`: Gemma 4's
// proportional rotation of its global layers, a table the layers share);
// queries scaled by `scale` after the rotation (no node when it is 1:
// Gemma 4's) and attention run at scale 1 (as today's builders scale
// them); an optional logit softcap; a global layer's cache holds a cell per
// position, a sliding layer's (window > 0) a ring. Gemma 4's global layers
// tie V to K (`kv_tie`: V is K's product, before K's norm and rotation; the
// layer has no V weight) and normalize V by an unweighted RMS norm
// (`v_norm`, every Gemma 4 layer). K and V are F16 state; attention
// accumulates in F32 or wider.
//
// attention.borrowed, version 1: the same attention for queries only, over
// another stage's K and V (a drafter reading its target's cache: Gemma 4's
// assistant). Its state contract names the target's tensors ("k" and "v"
// of target layer `target_layer`, borrowed_from_target): it owns no bytes,
// stores nothing, and reads the cells the target's committed positions
// hold (a frozen read: the step's own rows are never stored, so its masks
// see only the positions before its rows).
//
// The fragment emits what the Gemma 2, Gemma 3 and Gemma 4 graph builders
// (retired with their runners) recorded, in their order (GraphOrder over
// their outputs): the input norm; Q's product, reshape,
// norm, rotation and scale; V's product (K's when tied), reshape and norm;
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

constexpr ParamSpec Flag(std::string_view name) {
  return {.name = name,
          .type = ParamType::kBool,
          .required = false,
          .least = 0,
          .most = 1,
          .fallback = 0.0};
}

// The parameters both kinds take, in order; the dense kind adds its V's
// form, the borrowed one the target layer it reads.
constexpr std::size_t kCommon = 19;
constexpr std::array<ParamSpec, kCommon> kCommonParams{
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
    Flag("qk_norm"),
    // Per-pair frequency factors divide the angles (role rope_freqs).
    Flag("rope_freqs"),
    // Biases after Q/K/V products, before their head reshapes and rotation.
    Flag("qkv_bias"),
    // Apply scale to the scores rather than Q (different F32 rounding).
    Flag("scale_scores"),
    // Materialize scores and probabilities instead of a fused attention op.
    Flag("explicit_scores"),
    // F16 normalized inputs, Q/K/V products and biases, and attended rows;
    // the residual stream, rotations and output product remain F32.
    Flag("f16_projections"),
};

template <std::size_t N>
constexpr std::array<ParamSpec, kCommon + N> WithCommon(const std::array<ParamSpec, N>& more) {
  std::array<ParamSpec, kCommon + N> out{};
  for (std::size_t i = 0; i < kCommon; ++i) out.at(i) = kCommonParams.at(i);
  for (std::size_t i = 0; i < N; ++i) out.at(kCommon + i) = more.at(i);
  return out;
}

constexpr auto kParams = WithCommon(std::array{
    // V is K's product (no V weight), before K's norm and rotation.
    Flag("kv_tie"),
    // V normalized by an unweighted RMS norm of each head.
    Flag("v_norm"),
});

constexpr auto kBorrowedParams = WithCommon(std::array{
    // The target layer whose K and V it reads.
    ParamSpec{.name = "target_layer", .type = ParamType::kInt, .least = 0, .most = 4095},
});

enum : std::uint16_t {
  kNorm,
  kQ,
  kK,
  kV,
  kOutput,
  kPostNorm,
  kQNorm,
  kKNorm,
  kRopeFreqs,
  kQBias,
  kKBias,
  kVBias
};
constexpr std::array kRoles{
    WeightRole{.name = "norm", .encodings = kNormEncodings},
    WeightRole{.name = "q"},
    WeightRole{.name = "k"},
    WeightRole{.name = "v"},
    WeightRole{.name = "output"},
    WeightRole{.name = "post_norm", .encodings = kF32Only},
    WeightRole{.name = "q_norm", .encodings = kF32Only},
    WeightRole{.name = "k_norm", .encodings = kF32Only},
    WeightRole{.name = "rope_freqs", .encodings = kF32Only},
    WeightRole{.name = "q_bias", .encodings = kBiasEncodings},
    WeightRole{.name = "k_bias", .encodings = kBiasEncodings},
    WeightRole{.name = "v_bias", .encodings = kBiasEncodings},
};

// The state's encoding (D-108: F16 by default; a lossy one is a later
// per-alias setting).
constexpr Encoding kKvEncoding = Encoding::kF16;

struct Dims {
  std::uint64_t width, heads, kv_heads, kd, vd, rope_dims;
  std::uint32_t window;
  bool sandwich, qk_norm, neox, freqs, kv_tie, v_norm, borrowed;
  bool qkv_bias, scale_scores, explicit_scores, f16;
  std::uint32_t target_layer;
  float eps, scale, softcap, base, freq_scale;
};
Dims DimsOf(const Instance& in) {
  const Params& p = in.params;
  const bool borrowed = in.def->kind == "attention.borrowed";
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
          .freqs = p.Flag("rope_freqs"),
          .kv_tie = !borrowed && p.Flag("kv_tie"),
          .v_norm = !borrowed && p.Flag("v_norm"),
          .borrowed = borrowed,
          .qkv_bias = p.Flag("qkv_bias"),
          .scale_scores = p.Flag("scale_scores"),
          .explicit_scores = p.Flag("explicit_scores"),
          .f16 = p.Flag("f16_projections"),
          .target_layer = borrowed ? static_cast<std::uint32_t>(p.Int("target_layer")) : 0U,
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
  if (d.kv_tie && d.kd != d.vd) return std::unexpected("V tied to K of another width");
  if (d.qkv_bias && (d.borrowed || d.kv_tie)) {
    return std::unexpected("Q/K/V biases require separate owned Q/K/V products");
  }
  if (d.explicit_scores && d.softcap != 0.0F) {
    return std::unexpected("explicit scores with a softcap are not implemented");
  }
  std::vector<RoleShape> out{Role(kNorm, {d.width}), Role(kQ, {d.width, d.heads * d.kd}),
                             Role(kOutput, {d.heads * d.vd, d.width})};
  if (!d.borrowed) {
    out.push_back(Role(kK, {d.width, d.kv_heads * d.kd}));
    if (!d.kv_tie) out.push_back(Role(kV, {d.width, d.kv_heads * d.vd}));
  }
  if (d.sandwich) out.push_back(Role(kPostNorm, {d.width}));
  if (d.qk_norm) {
    out.push_back(Role(kQNorm, {d.kd}));
    if (!d.borrowed) out.push_back(Role(kKNorm, {d.kd}));
  }
  if (d.freqs) out.push_back(Role(kRopeFreqs, {d.rope_dims / 2}));
  if (d.qkv_bias) {
    out.push_back(Role(kQBias, {d.heads * d.kd}));
    out.push_back(Role(kKBias, {d.kv_heads * d.kd}));
    out.push_back(Role(kVBias, {d.kv_heads * d.vd}));
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
          // A verify's rejected rows are never read: past the kept prefix
          // (a cell a position), or in a ring, which holds a chunk's rows
          // beside its window (a verify's rows fit the chunk bound), so a
          // later query's mask puts a rejected row's cell (and the older
          // position it overwrote) outside its window.
          .rollback = model::Rollback::kTruncate};
}

model::StateContract State(const Instance& in) {
  const Dims d = DimsOf(in);
  model::StateContract c{
      .tensors = {Kv("k", d.kv_heads * d.kd, d.window), Kv("v", d.kv_heads * d.vd, d.window)}};
  if (d.borrowed) {
    // The target's tensors of these names in its layer: read, never written.
    for (auto& t : c.tensors) {
      t.borrowed_from_target = true;
      t.target_layer = d.target_layer;
    }
  }
  return c;
}

std::vector<MaskUse> Masks(const Instance& in) {
  return {{.tensor = 0, .window = DimsOf(in).window}};
}

constexpr std::array kBatchClasses{
    BatchClass{.op = ir::Op::kRmsNorm, .batch = ir::Batch::kJoined},
    BatchClass{.op = ir::Op::kMatMul, .batch = ir::Batch::kJoined},
    BatchClass{.op = ir::Op::kRope, .batch = ir::Batch::kJoined},
    BatchClass{.op = ir::Op::kMaskBuild, .batch = ir::Batch::kPerOwner},
    BatchClass{.op = ir::Op::kAttention, .batch = ir::Batch::kPerOwner},
    BatchClass{.op = ir::Op::kCacheStore, .batch = ir::Batch::kPerOwner},
};

// By heads: Q, K and V by their output rows (whole heads, a query head's
// group with its KV head), the K and V caches by their cells' heads, the
// output product by its input columns; its partial sums reduce before the
// post norm. A tied layer has no V weight (its V is K's split).
constexpr std::array kHeadRoles{
    RoleSplit{.role = kQ, .dim = 1},     RoleSplit{.role = kK, .dim = 1},
    RoleSplit{.role = kV, .dim = 1},     RoleSplit{.role = kOutput, .dim = 0},
    RoleSplit{.role = kQBias, .dim = 0}, RoleSplit{.role = kKBias, .dim = 0},
    RoleSplit{.role = kVBias, .dim = 0}};
constexpr std::array kHeadState{StateSplit{.tensor = 0, .dim = 0},
                                StateSplit{.tensor = 1, .dim = 0}};
std::uint32_t KvHeadUnits(const Instance& in) {
  return static_cast<std::uint32_t>(in.params.Int("kv_heads"));
}
std::uint64_t StreamBytes(const SplitStep& s) { return s.instance->stage.width * s.rows * 4; }
constexpr std::array kSplits{SplitOption{.kind = SplitKind::kHeads,
                                         .collective = ir::Op::kAllReduce,
                                         // Whole KV heads per device.
                                         .units = KvHeadUnits,
                                         .bytes_per_step = StreamBytes,
                                         .weights = kHeadRoles,
                                         .state = kHeadState}};
// Queries only: Q and the output product by heads, the borrowed caches by
// their heads (the target's own split must agree; the partitioner places
// both stages).
constexpr std::array kBorrowedRoles{RoleSplit{.role = kQ, .dim = 1},
                                    RoleSplit{.role = kOutput, .dim = 0}};
constexpr std::array kBorrowedSplits{SplitOption{.kind = SplitKind::kHeads,
                                                 .collective = ir::Op::kAllReduce,
                                                 .units = KvHeadUnits,
                                                 .bytes_per_step = StreamBytes,
                                                 .weights = kBorrowedRoles,
                                                 .state = kHeadState}};

constexpr std::array<std::string_view, 7> kPatterns{
    "rms_norm_mul",    "rms_norm_mul_rope", "rope_store",      "grouped_stores",
    "owner_attention", "prepared_products", "rms_norm_mul_add"};
constexpr std::array<std::string_view, 4> kBorrowedPatterns{"rms_norm_mul", "rms_norm_mul_rope",
                                                            "owner_attention", "rms_norm_mul_add"};

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
  const Dims d = DimsOf(f.instance);
  return {.capacity = f.b.tensor(f.State(0, tensor)).ne(1),
          .ring = d.window != 0,
          .frozen = d.borrowed};
}

std::expected<FragmentOut, std::string> Emit(Fragment& f) {
  ir::Builder& b = f.b;
  const Dims d = DimsOf(f.instance);
  const auto w = [&](std::uint16_t role) { return f.Weight(role); };
  const std::uint64_t rows = f.step.rows();
  const ir::TensorId x = f.stream;
  // A query-only fragment emits nothing for a state-only step.
  if (d.borrowed && !f.stream_needed) return FragmentOut{.stream = x};
  ir::TensorId normalized = b.Mul(b.RmsNorm(x, d.eps), NormScale(b, w(kNorm)));
  if (d.f16) normalized = b.Convert(normalized, Encoding::kF16);
  const auto projection = [&](std::uint16_t role, std::uint16_t bias) {
    ir::TensorId product =
        MatMulOutput(b, w(role), normalized, d.f16 ? Encoding::kF16 : Encoding::kF32);
    if (d.qkv_bias) product = b.Add(product, w(bias));
    return product;
  };
  const auto rotation_input = [&](ir::TensorId t) {
    return d.f16 ? b.Convert(t, Encoding::kF32) : t;
  };
  const ir::RopeParams rope = Rotation(f.instance, d);
  const ir::TensorId positions = f.step.Positions();
  const ir::TensorId freqs = d.freqs ? w(kRopeFreqs) : ir::TensorId{};
  const auto head_norm = [&](ir::TensorId t, std::uint16_t role) {
    return d.qk_norm ? b.Mul(b.RmsNorm(t, d.eps), w(role)) : t;
  };
  ir::TensorId q{};
  if (f.stream_needed) {
    q = b.Reshape(rotation_input(projection(kQ, kQBias)), Shape({d.kd, d.heads, rows}));
    q = b.Rope(head_norm(q, kQNorm), positions, freqs, rope);
    if (!d.scale_scores && d.scale != 1.0F) q = b.Scale(q, d.scale);
  }
  ir::TensorId k{};
  ir::TensorId v{};
  if (!d.borrowed) {
    // V before K, as the builders' graphs order them; a tied V is K's
    // product, which K then reshapes again.
    const ir::TensorId v_product = projection(d.kv_tie ? kK : kV, kVBias);
    v = b.Reshape(v_product, Shape({d.vd, d.kv_heads, rows}));
    if (d.v_norm) v = b.RmsNorm(v, d.eps);
    const ir::TensorId k_product = d.kv_tie ? v_product : projection(kK, kKBias);
    k = b.Reshape(rotation_input(k_product), Shape({d.kd, d.kv_heads, rows}));
    k = b.Rope(head_norm(k, kKNorm), positions, freqs, rope);
  }

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
    if (!d.borrowed) {
      const ir::TensorId store = f.step.StoreCells(s, cells);
      if (!k_rows.valid()) k_rows = b.Reshape(k, Shape({k_width, rows}));
      b.CacheStore(f.State(s, 0), slice(k_rows, k_width), store);
      if (!v_rows.valid()) v_rows = b.Reshape(v, Shape({v_width, rows}));
      b.CacheStore(f.State(s, 1), slice(v_rows, v_width), store);
    }
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
    const float scale = d.scale_scores ? d.scale : 1.0F;
    ir::TensorId attention;
    if (d.explicit_scores) {
      const auto scores = b.MatMul(keys, queries, {.accumulate = ir::Accumulate::kF32});
      const auto probabilities =
          b.SoftMax(scores, b.Convert(*mask, Encoding::kF32), {}, {.scale = scale});
      // V remains cell-major state; only the attended view is transposed
      // for a product, so the shared lifecycle still spills whole cells.
      const auto transposed = b.Cont(b.Transpose(values));
      const auto product = b.MatMul(transposed, probabilities);
      attention = b.Cont(b.Permute(product, {0, 2, 1, 3}));
    } else {
      attention =
          b.Attention(queries, keys, values, *mask, {},
                      {.scale = scale, .softcap = d.softcap, .accumulate = ir::Accumulate::kF32});
    }
    outputs.push_back(b.Reshape(attention, Shape({d.vd * d.heads, seg.rows})));
    b.SetJoined();
  }
  if (!f.stream_needed) return FragmentOut{.stream = x};
  ir::TensorId joined = outputs.front();
  for (std::size_t s = 1; s < outputs.size(); ++s) joined = b.Concat(joined, outputs[s], 1);
  if (d.f16) joined = b.Convert(joined, Encoding::kF16);
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
void Rotate(std::span<double> row, std::uint64_t heads, const Dims& d, std::span<const float> freqs,
            std::int64_t position) {
  const std::uint64_t pairs = d.rope_dims / 2;
  for (std::uint64_t h = 0; h < heads; ++h) {
    double* x = &row[h * d.kd];
    for (std::uint64_t i = 0; i < pairs; ++i) {
      const std::uint64_t a = d.neox ? i : 2 * i;
      const std::uint64_t c = d.neox ? i + pairs : (2 * i) + 1;
      const double theta = RopeAngle(d.base, static_cast<std::uint32_t>(d.rope_dims), d.freq_scale,
                                     i, position, freqs.empty() ? 1.0F : freqs[i]);
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
  const std::span<const float> freqs = d.freqs ? weight(kRopeFreqs) : std::span<const float>{};
  std::vector<double> normalized = io.stream;
  RmsRows(normalized, width, d.eps, weight(kNorm));
  if (d.f16)
    for (double& value : normalized) value = RoundF16(value);
  std::vector<double> q = Product(weight(kQ), width, d.heads * d.kd, normalized);
  std::vector<double> k;
  std::vector<double> v;
  if (!d.borrowed) {
    k = Product(weight(kK), width, d.kv_heads * d.kd, normalized);
    v = d.kv_tie ? k : Product(weight(kV), width, d.kv_heads * d.vd, normalized);
  }
  const auto bias = [&](std::vector<double>& values, std::uint16_t role) {
    const auto bs = d.qkv_bias ? weight(role) : std::span<const float>{};
    for (std::size_t i = 0; i < values.size(); ++i) {
      if (d.f16) values[i] = RoundF16(values[i]);
      if (!bs.empty()) {
        values[i] += bs[i % bs.size()];
        if (d.f16) values[i] = RoundF16(values[i]);
      }
    }
  };
  bias(q, kQBias);
  if (!d.borrowed) {
    bias(k, kKBias);
    bias(v, kVBias);
    if (d.v_norm) RmsRows(v, d.vd, d.eps, {});
  }
  if (d.qk_norm) {
    RmsRows(q, d.kd, d.eps, weight(kQNorm));
    if (!d.borrowed) RmsRows(k, d.kd, d.eps, weight(kKNorm));
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
    // The positions the cells hold through: the step's rows stored, or
    // (borrowed) the positions before them.
    const std::uint64_t end = std::uint64_t{seg.past} + (d.borrowed ? 0 : seg.rows);
    // The model sees every position of a row's window; a ring that would
    // overwrite one before the row attends cannot hold the step.
    if (!d.borrowed && ring && end > capacity &&
        seg.rows + std::uint64_t{d.window} > capacity + 1) {
      return std::unexpected("attention reference: a ring too small for the rows and window");
    }
    // Rotate, then store every row before any row attends (the step's
    // stores precede its attention).
    for (std::uint64_t i = 0; i < seg.rows; ++i) {
      const std::uint64_t r = first + i;
      const auto pos = static_cast<std::int64_t>(seg.past + i);
      Rotate(std::span(q).subspan(r * d.heads * d.kd, d.heads * d.kd), d.heads, d, freqs, pos);
      if (d.borrowed) continue;
      Rotate(std::span(k).subspan(r * d.kv_heads * d.kd, d.kv_heads * d.kd), d.kv_heads, d, freqs,
             pos);
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
          // The position the cell holds.
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
  if (d.f16)
    for (double& value : attended) value = RoundF16(value);
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

const ComponentDef& AttentionBorrowed() {
  static const ComponentDef def{
      .kind = "attention.borrowed",
      .version = 1,
      .narrows = true,
      .params = kBorrowedParams,
      .roles = kRoles,
      .shapes = Shapes,
      .state = State,
      .batch_classes = std::span(kBatchClasses).first(5),  // no stores
      .batching = nullptr,
      .splits = kBorrowedSplits,
      .patterns = kBorrowedPatterns,
      .masks = Masks,
      .emit = Emit,
      .reference = Reference,
  };
  return def;
}

}  // namespace llmp::components::defs
