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
#include "components/attention_core.h"
#include "components/component.h"
#include "components/defs.h"
#include "components/rotation_reference.h"
#include "components/spatial_stream.h"
#include "components/step.h"
#include "format/encoding.h"
#include "ir/graph.h"
#include "ir/op.h"
#include "model/state_contract.h"

namespace llmp::components::defs {
namespace {

using format::Encoding;

constexpr std::array<std::string_view, 3> kPlacements{"sandwich", "pre", "bare"};
constexpr std::array<std::string_view, 2> kRotations{"neox", "normal"};

constexpr ParamSpec Flag(std::string_view name) {
  return {.name = name,
          .type = ParamType::kBool,
          .required = false,
          .least = 0,
          .most = 1,
          .fallback = 0.0};
}

constexpr ParamSpec OptionalInt(std::string_view name, double most) {
  return {.name = name,
          .type = ParamType::kInt,
          .required = false,
          .least = 0.0,
          .most = most,
          .fallback = 0.0};
}

// The parameters both kinds take, in order; the dense kind adds its V's
// form, the borrowed one the target layer it reads.
constexpr std::size_t kCommon = 31;
constexpr std::array<ParamSpec, kCommon> kCommonParams{
    kNormZeroCentered,
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
    ParamSpec{.name = "rope_partial_factor",
              .type = ParamType::kReal,
              .required = false,
              .least = 0.0,
              .most = 1.0,
              .fallback = 1.0},
    ParamSpec{.name = "rope_offset",
              .type = ParamType::kInt,
              .required = false,
              .least = 0,
              .most = 1024,
              .fallback = 0},
    ParamSpec{.name = "rope_base", .type = ParamType::kReal, .least = 1e-30, .most = 1e12},
    // The frequency scale (1 / a linear scaling's factor).
    ParamSpec{.name = "rope_scale", .type = ParamType::kReal, .least = 1e-30, .most = 1e6},
    Flag("yarn"),
    ParamSpec{.name = "rope_factor",
              .type = ParamType::kReal,
              .required = false,
              .least = 1.0,
              .most = 1e6,
              .fallback = 1.0},
    ParamSpec{.name = "yarn_context",
              .type = ParamType::kInt,
              .required = false,
              .least = 0,
              .most = 1 << 30,
              .fallback = 0},
    ParamSpec{.name = "beta_fast",
              .type = ParamType::kReal,
              .required = false,
              .least = 0.0,
              .most = 1e6,
              .fallback = 32.0},
    ParamSpec{.name = "beta_slow",
              .type = ParamType::kReal,
              .required = false,
              .least = 0.0,
              .most = 1e6,
              .fallback = 1.0},
    Flag("llama3_rope"),
    ParamSpec{.name = "low_freq_factor",
              .type = ParamType::kReal,
              .required = false,
              .least = 0.0001,
              .most = 1000.0,
              .fallback = 1.0},
    ParamSpec{.name = "high_freq_factor",
              .type = ParamType::kReal,
              .required = false,
              .least = 0.0001,
              .most = 1000.0,
              .fallback = 4.0},
    Flag("qk_norm"),
    // Per-pair frequency factors divide the angles (role rope_freqs).
    Flag("rope_freqs"),
    // Biases after Q/K/V products, before their head reshapes and rotation.
    Flag("qkv_bias"),
    // Apply scale to the scores rather than Q (different F32 rounding).
    Flag("scale_scores"),
    // Materialize scores and probabilities instead of a fused attention op.
    Flag("explicit_scores"),
    // Per-query-head attention sinks in the softmax denominator (role sinks).
    Flag("sinks"),
    // Interleaved per-head output gate in the Q projection, applied via sigmoid.
    Flag("output_gate"),
};

template <std::size_t N>
constexpr std::array<ParamSpec, kCommon + N> WithCommon(const std::array<ParamSpec, N>& more) {
  std::array<ParamSpec, kCommon + N> out{};
  for (std::size_t i = 0; i < kCommon; ++i) out.at(i) = kCommonParams.at(i);
  for (std::size_t i = 0; i < N; ++i) out.at(kCommon + i) = more.at(i);
  return out;
}

constexpr std::array<std::string_view, 2> kKvEncodings{"f16", "q8_0"};
constexpr auto kParams = WithCommon(std::array{
    ParamSpec{.name = "kv_encoding",
              .type = ParamType::kName,
              .required = false,
              .choices = kKvEncodings},
    // V is K's product (no V weight), before K's norm and rotation.
    Flag("kv_tie"),
    // V normalized by an unweighted RMS norm of each head.
    Flag("v_norm"),
    // Stateless self-attention over the step's own K and V rows under an explicit mask.
    Flag("stateless"),
    OptionalInt("rope_axis0", 512),
    OptionalInt("rope_axis1", 512),
    OptionalInt("rope_axis2", 512),
    ParamSpec{.name = "target_layer",
              .type = ParamType::kInt,
              .required = false,
              .least = -1,
              .most = 4095,
              .fallback = -1},
});

constexpr auto kBorrowedParams = WithCommon(std::array{
    ParamSpec{.name = "kv_encoding",
              .type = ParamType::kName,
              .required = false,
              .choices = kKvEncodings},
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
  kVBias,
  kSinks
};
constexpr std::array kRoles{
    WeightRole{.name = "norm", .encodings = kNormEncodings, .derived = kZeroCenteredNorm},
    WeightRole{.name = "q"},
    WeightRole{.name = "k"},
    WeightRole{.name = "v"},
    WeightRole{.name = "output"},
    WeightRole{.name = "post_norm", .encodings = kNormEncodings, .derived = kZeroCenteredNorm},
    WeightRole{.name = "q_norm", .encodings = kNormEncodings, .derived = kZeroCenteredNorm},
    WeightRole{.name = "k_norm", .encodings = kNormEncodings, .derived = kZeroCenteredNorm},
    WeightRole{.name = "rope_freqs", .encodings = kF32Only},
    WeightRole{.name = "q_bias", .encodings = kBiasEncodings},
    WeightRole{.name = "k_bias", .encodings = kBiasEncodings},
    WeightRole{.name = "v_bias", .encodings = kBiasEncodings},
    WeightRole{.name = "sinks", .encodings = kF32Only},
};

struct Dims {
  Encoding kv_encoding;
  std::uint64_t width, heads, kv_heads, kd, vd, rope_dims, rope_offset;
  std::uint32_t window, rotated_dims;
  bool sandwich, bare, qk_norm, neox, freqs, kv_tie, v_norm, borrowed, stateless;
  bool qkv_bias, scale_scores, explicit_scores, sinks, output_gate, yarn_enabled, llama3_rope;
  std::array<std::uint64_t, 3> rope_axes{};
  std::uint32_t target_layer;
  float eps, scale, softcap, base, freq_scale, low_freq_factor, high_freq_factor;
  YarnParams yarn;
};
Dims DimsOf(const Instance& in) {
  const Params& p = in.params;
  const bool borrowed = in.def->kind == "attention.borrowed" ||
                        (p.Given("target_layer") && p.Int("target_layer") >= 0);
  const auto rope_dims = static_cast<std::uint64_t>(p.Int("rope_dims"));
  const auto a0 = borrowed ? 0ULL : static_cast<std::uint64_t>(p.Int("rope_axis0"));
  const auto a1 = borrowed ? 0ULL : static_cast<std::uint64_t>(p.Int("rope_axis1"));
  const auto a2 = borrowed ? 0ULL : static_cast<std::uint64_t>(p.Int("rope_axis2"));
  const std::array<std::uint64_t, 3> rope_axes = (a0 == 0 && a1 == 0 && a2 == 0)
                                                     ? std::array<std::uint64_t, 3>{rope_dims, 0, 0}
                                                     : std::array<std::uint64_t, 3>{a0, a1, a2};
  const bool llama3_rope = p.Flag("llama3_rope");
  const bool yarn_enabled = !llama3_rope && (p.Flag("yarn") || p.Real("rope_factor") > 1.0);
  const float yarn_factor = p.Given("rope_factor")
                                ? F32(p.Real("rope_factor"))
                                : ((p.Flag("yarn") || llama3_rope) && p.Real("rope_scale") < 1.0
                                       ? F32(1.0 / p.Real("rope_scale"))
                                       : F32(p.Real("rope_factor")));
  const std::uint32_t yarn_ctx = p.Given("yarn_context")
                                     ? static_cast<std::uint32_t>(p.Int("yarn_context"))
                                     : in.stage.context;
  const auto enc_name = p.Name("kv_encoding");
  const Encoding kv_enc = enc_name == "q8_0" ? Encoding::kQ8_0 : Encoding::kF16;
  return {.kv_encoding = kv_enc,
          .width = in.stage.width,
          .heads = static_cast<std::uint64_t>(p.Int("heads")),
          .kv_heads = static_cast<std::uint64_t>(p.Int("kv_heads")),
          .kd = static_cast<std::uint64_t>(p.Int("key_dim")),
          .vd = static_cast<std::uint64_t>(p.Int("value_dim")),
          .rope_dims = rope_dims,
          .rope_offset = static_cast<std::uint64_t>(p.Int("rope_offset")),
          .window = static_cast<std::uint32_t>(p.Int("window")),
          .rotated_dims = p.Real("rope_partial_factor") == 1.0
                              ? UINT32_MAX
                              : static_cast<std::uint32_t>(
                                    2.0 * std::floor(p.Real("rope_partial_factor") *
                                                     static_cast<double>(rope_dims) / 2.0)),
          .sandwich = p.Name("placement") == "sandwich",
          .bare = p.Name("placement") == "bare",
          .qk_norm = p.Flag("qk_norm"),
          .neox = p.Name("rotation") == "neox",
          .freqs = p.Flag("rope_freqs"),
          .kv_tie = !borrowed && p.Flag("kv_tie"),
          .v_norm = !borrowed && p.Flag("v_norm"),
          .borrowed = borrowed,
          .stateless = !borrowed && p.Flag("stateless"),
          .qkv_bias = p.Flag("qkv_bias"),
          .scale_scores = p.Flag("scale_scores"),
          .explicit_scores = p.Flag("explicit_scores"),
          .sinks = p.Flag("sinks"),
          .output_gate = p.Flag("output_gate"),
          .yarn_enabled = yarn_enabled,
          .llama3_rope = llama3_rope,
          .rope_axes = rope_axes,
          .target_layer = borrowed ? static_cast<std::uint32_t>(p.Int("target_layer")) : 0U,
          .eps = F32(p.Real("eps")),
          .scale = F32(p.Real("scale")),
          .softcap = F32(p.Real("softcap")),
          .base = F32(p.Real("rope_base")),
          .freq_scale = F32(p.Real("rope_scale")),
          .low_freq_factor = F32(p.Real("low_freq_factor")),
          .high_freq_factor = F32(p.Real("high_freq_factor")),
          .yarn = {.factor = (yarn_enabled || llama3_rope) ? yarn_factor : 1.0F,
                   .context = yarn_ctx,
                   .beta_fast = F32(p.Real("beta_fast")),
                   .beta_slow = F32(p.Real("beta_slow")),
                   .inverse_attn_scale = false}};
}

std::expected<std::vector<RoleShape>, std::string> Shapes(const Instance& in) {
  if (NormOffsetContradictsFold(in))
    return std::unexpected("zero-centered norms contradict the norm.plus_one fold");
  const Dims d = DimsOf(in);
  if (in.def->kind == "attention.dense" && d.borrowed &&
      (in.params.Flag("stateless") || in.params.Flag("kv_tie") || in.params.Flag("v_norm") ||
       in.params.Int("rope_axis0") != 0 || in.params.Int("rope_axis1") != 0 ||
       in.params.Int("rope_axis2") != 0)) {
    return std::unexpected("borrowed attention does not admit dense-only state or rope options");
  }
  if (d.bare && !d.stateless) return std::unexpected("bare placement requires stateless attention");
  if ((d.rope_axes[0] % 2) != 0 || (d.rope_axes[1] % 2) != 0 || (d.rope_axes[2] % 2) != 0 ||
      d.rope_axes[0] + d.rope_axes[1] + d.rope_axes[2] != d.rope_dims) {
    return std::unexpected("rope_axes must be even and sum to rope_dims");
  }
  if ((d.rope_axes[1] != 0 || d.rope_axes[2] != 0) && (!d.stateless || d.freqs)) {
    return std::unexpected("multi-axis rope requires stateless attention without rope_freqs");
  }
  if (d.stateless && d.kv_encoding == Encoding::kQ8_0) {
    return std::unexpected("stateless attention does not use quantized KV state");
  }
  if (d.kv_encoding == Encoding::kQ8_0 &&
      (d.kd != d.vd || (d.kd != 64 && d.kd != 256 && d.kd != 512) || d.explicit_scores ||
       (d.kd != 256 && d.softcap != 0.0F)))
    return std::unexpected(
        "quantized KV requires owned D64/D256/D512 attention, equal K/V, "
        "no explicit scores and softcap only at D256");
  if (d.heads % d.kv_heads != 0) return std::unexpected("heads are not a multiple of kv_heads");
  if (d.rope_dims % 2 != 0 || d.rope_offset + d.rope_dims > d.kd) {
    return std::unexpected("rope_dims is odd or past key_dim");
  }
  if (d.rotated_dims != UINT32_MAX &&
      (d.yarn_enabled || d.llama3_rope || d.rope_axes[1] != 0 || d.rope_axes[2] != 0)) {
    return std::unexpected("proportional RoPE requires one-dimensional unscaled frequency pairs");
  }
  if (d.llama3_rope && in.params.Flag("yarn")) {
    return std::unexpected("Llama 3 RoPE and YaRN cannot both be enabled");
  }
  if (d.llama3_rope && (d.rope_axes[1] != 0 || d.rope_axes[2] != 0 || !(d.yarn.factor > 1.0F) ||
                        d.yarn.context == 0 || !(d.low_freq_factor > 0.0F) ||
                        !(d.high_freq_factor > d.low_freq_factor))) {
    return std::unexpected(
        "Llama 3 RoPE requires 1D rotation, rope_factor > 1, a positive original context, and "
        "0 < low_freq_factor < high_freq_factor");
  }
  if (d.yarn_enabled && (!(d.yarn.factor > 1.0F) || d.yarn.context == 0)) {
    return std::unexpected("YaRN requires rope_factor > 1 and a positive original context");
  }
  if (d.kv_tie && d.kd != d.vd) return std::unexpected("V tied to K of another width");
  if (d.qkv_bias && (d.borrowed || d.kv_tie)) {
    return std::unexpected("Q/K/V biases require separate owned Q/K/V products");
  }
  if (d.explicit_scores && d.softcap != 0.0F) {
    return std::unexpected("explicit scores with a softcap are not implemented");
  }
  const std::uint64_t q_stride = d.output_gate ? (d.kd + d.vd) : d.kd;
  std::vector<RoleShape> out;
  if (!d.bare) out.push_back(Role(kNorm, {d.width}));
  out.push_back(Role(kQ, {d.width, d.heads * q_stride}));
  out.push_back(Role(kOutput, {d.heads * d.vd, d.width}));
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
    out.push_back(Role(kQBias, {d.heads * q_stride}));
    out.push_back(Role(kKBias, {d.kv_heads * d.kd}));
    out.push_back(Role(kVBias, {d.kv_heads * d.vd}));
  }
  if (d.sinks) out.push_back(Role(kSinks, {d.heads}));
  return out;
}

model::StateTensorContract Kv(std::string name, std::uint64_t width, std::uint32_t window,
                              Encoding encoding) {
  const bool ring = window != 0;
  // The format layer's cell size (model::CellBytes): a lossy encoding's
  // whole blocks once an alias opts in (D-108); F16's elements today.
  const auto cell = model::CellBytes(encoding, width);
  base::Check(cell.has_value(), "validated dense attention cell geometry");
  return {.name = std::move(name),
          .kind = ring ? model::StateKind::kRing : model::StateKind::kKv,
          .encoding = encoding,
          .cell_bytes = *cell,
          .cells = ring ? model::CellRule::kRing : model::CellRule::kPerPosition,
          .window = window,
          // A verify's rejected rows are never read: past the kept prefix
          // (a cell a position), or in a ring, which holds a chunk's rows
          // beside its window (a verify's rows fit the chunk bound), so a
          // later query's mask puts a rejected row's cell (and the older
          // position it overwrote) outside its window.
          .rollback = model::Rollback::kTruncate};
}

model::StateContract State(const Instance& in) {
  const Dims d = DimsOf(in);
  if (d.stateless) return {};
  model::StateContract c{.tensors = {Kv("k", d.kv_heads * d.kd, d.window, d.kv_encoding),
                                     Kv("v", d.kv_heads * d.vd, d.window, d.kv_encoding)}};
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
  const Dims d = DimsOf(in);
  if (d.stateless) return {};
  return {{.tensor = 0, .window = d.window}};
}

// By heads: Q, K and V by their output rows (whole heads, a query head's
// group with its KV head), the K and V caches by their cells' heads, the
// output product by its input columns; its partial sums reduce before the
// post norm. A tied layer has no V weight (its V is K's split).
constexpr std::array kHeadRoles{
    RoleSplit{.role = kQ, .dim = 1},     RoleSplit{.role = kK, .dim = 1},
    RoleSplit{.role = kV, .dim = 1},     RoleSplit{.role = kOutput, .dim = 0},
    RoleSplit{.role = kQBias, .dim = 0}, RoleSplit{.role = kKBias, .dim = 0},
    RoleSplit{.role = kVBias, .dim = 0}, RoleSplit{.role = kSinks, .dim = 0}};
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

ir::RopeParams Rotation(const Instance& in, const Dims& d) {
  if (d.llama3_rope) {
    return {.mode = d.neox ? ir::RopeMode::kNeox : ir::RopeMode::kNormal,
            .llama3 = true,
            .dims = static_cast<std::uint32_t>(d.rope_dims),
            .offset = static_cast<std::uint32_t>(d.rope_offset),
            .original_context = d.yarn.context,
            .base = d.base,
            .scale = 1.0F / d.yarn.factor,
            .ext_factor = 0.0F,
            .attn_factor = 1.0F,
            .beta_fast = d.high_freq_factor,
            .beta_slow = d.low_freq_factor};
  }
  auto rotation = MakeRopeParams(
      d.neox ? ir::RopeMode::kNeox : ir::RopeMode::kNormal, static_cast<std::uint32_t>(d.rope_dims),
      static_cast<std::uint32_t>(d.rope_offset), in.stage.context, d.base, d.freq_scale, d.yarn);
  rotation.rotated_dims = d.rotated_dims;
  return rotation;
}

// The cell rule of a state tensor of this instance.
CellKey CellsOf(const Fragment& f, std::size_t tensor) {
  const Dims d = DimsOf(f.instance);
  if (d.stateless) return {};
  return {.capacity = f.b.tensor(f.State(0, tensor)).ne(1),
          .ring = d.window != 0,
          .frozen = d.borrowed};
}

std::expected<FragmentOut, std::string> Emit(Fragment& f) {
  ir::Builder& b = f.b;
  const Dims d = DimsOf(f.instance);
  F32Parameters<kRoles.size()> parameters(f);
  const auto w = [&](std::uint16_t role) {
    return role == kNorm || role == kPostNorm || role == kQNorm || role == kKNorm
               ? parameters.Norm(role)
               : f.Weight(role);
  };
  const std::uint64_t rows = f.step.rows();
  const ir::TensorId x = f.stream;
  // A query-only or stateless fragment emits nothing for a state-only step.
  if ((d.borrowed || d.stateless) && !f.stream_needed) return FragmentOut{.stream = x};
  if (d.stateless &&
      (f.Side("mask").valid() || f.Side("positions").valid() || d.bare || d.rope_axes[1] != 0 ||
       d.rope_axes[2] != 0 || f.instance.stage.context == 0)) {
    const ir::TensorId mask = f.Side("mask");
    const std::uint8_t axes = (d.rope_axes[1] != 0 || d.rope_axes[2] != 0) ? 3U : 1U;
    const ir::TensorId pos_src =
        f.Side("positions").valid() ? f.Side("positions") : f.step.Positions(axes);
    if (!mask.valid() || !pos_src.valid() || b.tensor(mask).format.encoding != Encoding::kF32 ||
        b.tensor(mask).ne(0) != rows || b.tensor(mask).ne(1) != rows ||
        b.tensor(pos_src).format.encoding != Encoding::kI32 ||
        b.tensor(pos_src).ne(0) * b.tensor(pos_src).ne(1) != axes * rows) {
      return std::unexpected("stateless attention requires positions and a square F32 mask");
    }
    ir::TensorId normalized = x;
    if (!d.bare) {
      const ir::TensorId nw = w(kNorm);
      normalized = b.Mul(b.RmsNorm(x, d.eps), nw);
    }
    const auto projection = [&](std::uint16_t role, std::uint16_t bias) {
      ir::TensorId product = b.MatMul(w(role), normalized);
      if (d.qkv_bias) product = b.Add(product, w(bias));
      return product;
    };
    const ir::TensorId q_product = projection(kQ, kQBias);
    const ir::TensorId k_product = projection(kK, kKBias);
    const ir::TensorId v_product = d.kv_tie ? k_product : projection(kV, kVBias);
    const ir::TensorId freqs = d.freqs ? w(kRopeFreqs) : ir::TensorId{};
    const auto rotate = [&](ir::TensorId product, std::uint64_t head_count,
                            std::uint16_t norm_role) {
      ir::TensorId t = b.Reshape(product, Shape({d.kd, head_count, rows}));
      if (d.qk_norm) {
        const ir::TensorId qw = w(norm_role);
        t = b.Mul(b.RmsNorm(t, d.eps), qw);
      }
      std::uint32_t offset = 0;
      for (std::size_t axis = 0; axis < d.rope_axes.size(); ++axis) {
        const auto dims = static_cast<std::uint32_t>(d.rope_axes[axis]);
        if (dims == 0) continue;
        const ir::TensorId pos = b.Cont(
            b.View(pos_src, Shape({rows}), {4, rows * 4, rows * 4, rows * 4}, axis * rows * 4));
        ir::RopeParams rp = Rotation(f.instance, d);
        rp.dims = dims;
        rp.offset = offset;
        t = b.Rope(t, pos, freqs, rp);
        offset += dims;
      }
      return t;
    };
    const ir::TensorId q_in =
        d.output_gate ? EmitInterleavedHeadSlice(b, q_product, d.kd, d.kd + d.vd, d.heads, rows, 0)
                      : q_product;
    ir::TensorId q = rotate(q_in, d.heads, kQNorm);
    if (!d.scale_scores && d.scale != 1.0F) q = b.Scale(q, d.scale);
    ir::TensorId v = b.Reshape(v_product, Shape({d.vd, d.kv_heads, rows}));
    if (d.v_norm) v = b.RmsNorm(v, d.eps);
    ir::TensorId k = rotate(k_product, d.kv_heads, kKNorm);
    const ir::TensorId queries = b.Cont(b.Permute(q, {0, 2, 1, 3}));
    const ir::TensorId keys = b.Cont(b.Permute(k, {0, 2, 1, 3}));
    const ir::TensorId values = b.Permute(v, {0, 2, 1, 3});
    const float scale = d.scale_scores ? d.scale : 1.0F;
    const ir::TensorId sinks = d.sinks ? w(kSinks) : ir::TensorId{};
    const ir::TensorId attention =
        b.Attention(queries, keys, values, mask, sinks, {.scale = scale, .softcap = d.softcap});
    ir::TensorId att_rows = b.Reshape(attention, Shape({d.vd * d.heads, rows}));
    if (d.output_gate)
      att_rows = EmitSigmoidOutputGate(b, att_rows, q_product, d.kd, d.vd, d.heads, rows);
    const ir::TensorId projected = b.MatMul(w(kOutput), att_rows);
    if (d.bare) return FragmentOut{.stream = projected};
    if (!d.sandwich) return FragmentOut{.stream = b.Add(projected, x)};
    const ir::TensorId pw = w(kPostNorm);
    const ir::TensorId normed = b.Mul(b.RmsNorm(projected, d.eps), pw);
    return FragmentOut{.stream = b.Add(normed, x)};
  }
  const ir::TensorId normalized = b.Mul(b.RmsNorm(x, d.eps), NormScale(b, w(kNorm)));
  const auto projection = [&](std::uint16_t role, std::uint16_t bias) {
    ir::TensorId product = b.MatMul(w(role), normalized);
    if (d.qkv_bias) product = b.Add(product, w(bias));
    return product;
  };
  const ir::RopeParams rope = Rotation(f.instance, d);
  const ir::TensorId positions = f.step.Positions();
  const ir::TensorId freqs = d.freqs ? w(kRopeFreqs) : ir::TensorId{};
  ir::TensorId q_full{};
  ir::TensorId q{};
  if (f.stream_needed) {
    q_full = projection(kQ, kQBias);
    q = d.output_gate ? EmitInterleavedHeadSlice(b, q_full, d.kd, d.kd + d.vd, d.heads, rows, 0)
                      : b.Reshape(q_full, Shape({d.kd, d.heads, rows}));
    q = EmitHeadNormRope(b, q, d.eps, d.qk_norm ? w(kQNorm) : ir::TensorId{}, positions, freqs,
                         rope, d.qk_norm);
    if (!d.scale_scores && d.scale != 1.0F) q = b.Scale(q, d.scale);
  }
  ir::TensorId k{};
  ir::TensorId v{};
  if (!d.borrowed) {
    // V before K, as the builders' graphs order them; a tied V is K's
    // product, which K then reshapes again.
    const ir::TensorId v_product = projection(d.kv_tie ? kK : kV, kVBias);
    v = b.Reshape(v_product, Shape({d.vd, d.kv_heads, rows}));
    if (d.v_norm) v = EmitHeadNorm(b, v, d.eps);
    const ir::TensorId k_product = d.kv_tie ? v_product : projection(kK, kKBias);
    k = b.Reshape(k_product, Shape({d.kd, d.kv_heads, rows}));
    k = EmitHeadNormRope(b, k, d.eps, d.qk_norm ? w(kKNorm) : ir::TensorId{}, positions, freqs,
                         rope, d.qk_norm);
  }

  const std::uint64_t k_width = d.kd * d.kv_heads;
  const std::uint64_t v_width = d.vd * d.kv_heads;
  ir::TensorId k_rows{};
  ir::TensorId v_rows{};
  const CellKey cells = CellsOf(f, 0);
  const ir::TensorId sinks = d.sinks && f.stream_needed ? w(kSinks) : ir::TensorId{};
  std::vector<ir::TensorId> outputs;
  for (std::size_t s = 0; s < f.step.segments().size(); ++s) {
    const SegmentRows& seg = f.step.segments()[s];
    const auto slice = [&](ir::TensorId t, std::uint64_t width) {
      const auto nb = b.tensor(t).nb;
      const format::Shape shape = Shape({width, seg.rows});
      return b.View(t, shape, ViewStrides(nb[0], nb[1], 0, shape),
                    std::uint64_t{seg.first_row} * nb[1]);
    };
    if (!d.borrowed && !d.stateless) {
      const ir::TensorId store = f.step.StoreCells(s, cells);
      if (!k_rows.valid()) k_rows = b.Reshape(k, Shape({k_width, rows}));
      EmitKvCacheStore(b, f.State(s, 0), slice(k_rows, k_width), store, d.kv_encoding);
      if (!v_rows.valid()) v_rows = b.Reshape(v, Shape({v_width, rows}));
      EmitKvCacheStore(b, f.State(s, 1), slice(v_rows, v_width), store, d.kv_encoding);
    }
    if (!f.stream_needed) continue;
    const float scale = d.scale_scores ? d.scale : 1.0F;
    const ir::AttnParams attn_params{
        .scale = scale, .softcap = d.softcap, .sinks = d.sinks, .accumulate = ir::Accumulate::kF32};
    if (d.stateless) {
      const ir::TensorId seg_pos = f.step.SegmentPositions(s);
      b.SetOwner(static_cast<std::uint8_t>(s));
      const ir::TensorId queries = EmitSegmentQueries(b, q, d.kd, d.heads, seg);
      const ir::TensorId keys = EmitSegmentQueries(b, k, d.kd, d.kv_heads, seg);
      const ir::TensorId values = EmitSegmentQueries(b, v, d.vd, d.kv_heads, seg);
      const ir::TensorId mask =
          b.MaskBuild(seg_pos, seg_pos, seg.rows,
                      {.kind = d.window != 0 ? ir::MaskKind::kWindow : ir::MaskKind::kCausal,
                       .exact_rows = true,
                       .window = d.window},
                      Encoding::kF16);
      outputs.push_back(EmitSegmentAttention(b, queries, keys, values, mask, sinks, attn_params,
                                             d.vd, d.heads, seg.rows, d.explicit_scores));
      b.SetJoined();
      continue;
    }
    const std::uint64_t read = f.step.ReadCells(s, cells);
    auto mask = f.step.Mask(s, cells, d.window);
    if (!mask) return std::unexpected(mask.error());
    b.SetOwner(static_cast<std::uint8_t>(s));
    const ir::TensorId queries = EmitSegmentQueries(b, q, d.kd, d.heads, seg);
    const ir::TensorId keys = EmitKvCacheView(b, f.State(s, 0), d.kd, d.kv_heads, read);
    const ir::TensorId values = EmitKvCacheView(b, f.State(s, 1), d.vd, d.kv_heads, read);
    outputs.push_back(EmitSegmentAttention(b, queries, keys, values, *mask, sinks, attn_params,
                                           d.vd, d.heads, seg.rows, d.explicit_scores));
    b.SetJoined();
  }
  if (!f.stream_needed) return FragmentOut{.stream = x};
  ir::TensorId joined = JoinSegmentOutputs(b, outputs);
  if (d.output_gate) joined = EmitSigmoidOutputGate(b, joined, q_full, d.kd, d.vd, d.heads, rows);
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

std::expected<void, std::string> Reference(RefIO& io) {
  const Instance& in = *io.instance;
  const Dims d = DimsOf(in);
  if (!d.stateless && d.kv_encoding == Encoding::kQ8_0 && io.store_row == nullptr)
    return std::unexpected("Q8_0 reference state requires the block row codec");
  const std::uint64_t width = d.width;
  const std::uint64_t rows = io.rows();
  const std::uint64_t axes = (d.rope_axes[1] != 0 || d.rope_axes[2] != 0) ? 3U : 1U;
  const bool explicit_stateless =
      d.stateless && (!io.carry.empty() || !io.side.empty() || !io.side_ids.empty() || d.bare ||
                      d.rope_axes[1] != 0 || d.rope_axes[2] != 0 || in.stage.context == 0);
  if (explicit_stateless) {
    if (rows == 0 || io.stream.size() != width * rows || io.carry.size() != rows * rows ||
        (io.side.size() != axes * rows && io.side_ids.size() != axes * rows)) {
      return std::unexpected(
          "stateless attention reference: stream, positions or mask of another shape");
    }
    if (!io.side.empty() && !std::ranges::all_of(io.side, [](double v) {
          return std::isfinite(v) && v == std::trunc(v) && v >= INT32_MIN && v <= INT32_MAX;
        })) {
      return std::unexpected("stateless attention reference: positions are not exact int32 values");
    }
  } else if (io.stream.size() != width * rows || io.state.size() != io.segments.size() ||
             io.state_cells.size() != (d.stateless ? 0U : 2U)) {
    return std::unexpected("attention reference: stream or state of another shape");
  }
  std::array<std::vector<float>, kRoles.size()> adjusted;
  if (in.params.Flag("norm_zero_centered")) {
    for (const auto role : {kNorm, kPostNorm, kQNorm, kKNorm}) {
      if (d.bare && role == kNorm) continue;
      if (role < io.weights.size() && !io.weights.at(role).empty()) {
        adjusted[role] = io.weights.at(role);
        for (float& value : adjusted[role]) value += 1.0F;
      }
    }
  }
  const auto weight = [&](std::uint16_t role) -> std::span<const float> {
    if (role >= io.weights.size() || io.weights.at(role).empty()) return {};
    return adjusted[role].empty() ? io.weights.at(role) : adjusted[role];
  };
  const std::span<const float> freqs = d.freqs ? weight(kRopeFreqs) : std::span<const float>{};
  const std::span<const float> sinks = d.sinks ? weight(kSinks) : std::span<const float>{};
  std::vector<double> normalized = io.stream;
  if (!d.bare) RmsRows(normalized, width, d.eps, weight(kNorm));
  const std::uint64_t q_stride = d.output_gate ? (d.kd + d.vd) : d.kd;
  std::vector<double> qg = Product(weight(kQ), width, d.heads * q_stride, normalized);
  std::vector<double> k;
  std::vector<double> v;
  if (!d.borrowed) {
    k = Product(weight(kK), width, d.kv_heads * d.kd, normalized);
    v = d.kv_tie ? k : Product(weight(kV), width, d.kv_heads * d.vd, normalized);
  }
  const auto bias = [&](std::vector<double>& values, std::uint16_t role) {
    const auto bs = d.qkv_bias ? weight(role) : std::span<const float>{};
    if (bs.empty()) return;
    for (std::size_t i = 0; i < values.size(); ++i) {
      values[i] += bs[i % bs.size()];
    }
  };
  bias(qg, kQBias);
  std::vector<double> q;
  if (d.output_gate) {
    q.resize(rows * d.heads * d.kd);
    for (std::uint64_t r = 0; r < rows; ++r) {
      for (std::uint64_t h = 0; h < d.heads; ++h) {
        for (std::uint64_t i = 0; i < d.kd; ++i) {
          q[(((r * d.heads) + h) * d.kd) + i] = qg[(r * d.heads * q_stride) + (h * q_stride) + i];
        }
      }
    }
  } else {
    q = qg;
  }
  if (!d.borrowed) {
    bias(k, kKBias);
    bias(v, kVBias);
    if (d.v_norm) RmsRows(v, d.vd, d.eps, {});
  }
  if (d.qk_norm) {
    RmsRows(q, d.kd, d.eps, weight(kQNorm));
    if (!d.borrowed) RmsRows(k, d.kd, d.eps, weight(kKNorm));
  }
  const ir::RopeParams rope = Rotation(in, d);
  std::vector<double> attended(rows * d.heads * d.vd, 0.0);
  if (explicit_stateless) {
    const std::uint64_t group = d.heads / d.kv_heads;
    const auto pos_at = [&](std::uint64_t index) -> std::int64_t {
      return !io.side.empty() ? static_cast<std::int64_t>(io.side[index]) : io.side_ids[index];
    };
    const auto rotate_stateless = [&](std::vector<double>& values, std::uint64_t head_count) {
      for (std::uint64_t r = 0; r < rows; ++r) {
        std::uint64_t offset = 0;
        for (std::size_t axis = 0; axis < d.rope_axes.size(); ++axis) {
          const std::uint64_t axis_dims = d.rope_axes[axis];
          if (axis_dims == 0) continue;
          const std::int64_t pos = pos_at(r + rows * axis);
          ir::RopeParams rp = rope;
          rp.dims = static_cast<std::uint32_t>(axis_dims);
          rp.offset = static_cast<std::uint32_t>(offset);
          const auto axis_freqs =
              freqs.empty() ? std::span<const float>{} : freqs.subspan(offset / 2, axis_dims / 2);
          RotateHeads(std::span(values).subspan(r * head_count * d.kd, head_count * d.kd),
                      head_count, d.kd, rp, pos, axis_freqs);
          offset += axis_dims;
        }
      }
    };
    rotate_stateless(q, d.heads);
    rotate_stateless(k, d.kv_heads);
    if (!d.scale_scores && d.scale != 1.0F) {
      for (double& val : q) val *= static_cast<double>(d.scale);
    }
    const double score_mult = d.scale_scores ? static_cast<double>(d.scale) : 1.0;
    std::vector<double> score(rows);
    for (std::uint64_t r = 0; r < rows; ++r) {
      for (std::uint64_t h = 0; h < d.heads; ++h) {
        const std::uint64_t kh = h / group;
        const double* qh = &q[(r * d.heads + h) * d.kd];
        const double sink_logit = (d.sinks && !sinks.empty())
                                      ? static_cast<double>(sinks[h])
                                      : -std::numeric_limits<double>::infinity();
        double top = sink_logit;
        for (std::uint64_t c = 0; c < rows; ++c) {
          const double mask = io.carry[(r * rows) + c];
          if (std::isinf(mask) && mask < 0.0) {
            score[c] = -std::numeric_limits<double>::infinity();
            continue;
          }
          double dot = 0.0;
          for (std::uint64_t j = 0; j < d.kd; ++j) {
            dot += qh[j] * score_mult * k[(c * d.kv_heads + kh) * d.kd + j];
          }
          if (d.softcap > 0.0F) dot = d.softcap * std::tanh(dot / d.softcap);
          dot += mask;
          score[c] = dot;
          top = std::max(top, dot);
        }
        double sum = (d.sinks && !sinks.empty()) ? std::exp(sink_logit - top) : 0.0;
        for (double& sc : score) {
          sc = std::isinf(sc) ? 0.0 : std::exp(sc - top);
          sum += sc;
        }
        double* out = &attended[(r * d.heads + h) * d.vd];
        if (sum == 0.0) continue;
        for (std::uint64_t c = 0; c < rows; ++c) {
          if (score[c] == 0.0) continue;
          const double pc = score[c] / sum;
          for (std::uint64_t j = 0; j < d.vd; ++j) {
            out[j] += pc * v[(c * d.kv_heads + kh) * d.vd + j];
          }
        }
      }
    }
    if (d.output_gate) ApplySigmoidOutputGateRef(attended, qg, rows, d.heads, d.kd, d.vd);
    std::vector<double> projected = Product(weight(kOutput), d.heads * d.vd, width, attended);
    if (d.bare) {
      io.stream = std::move(projected);
      return {};
    }
    if (d.sandwich) RmsRows(projected, width, d.eps, weight(kPostNorm));
    for (std::size_t i = 0; i < projected.size(); ++i) io.stream[i] += projected[i];
    return {};
  }
  std::uint64_t first = 0;
  for (std::size_t s = 0; s < io.segments.size(); ++s) {
    const RefSegment& seg = io.segments[s];
    if (d.stateless) {
      for (std::uint64_t i = 0; i < seg.rows; ++i) {
        const std::uint64_t r = first + i;
        const auto pos = static_cast<std::int64_t>(seg.past + i);
        RotateHeads(std::span(q).subspan(r * d.heads * d.kd, d.heads * d.kd), d.heads, d.kd, rope,
                    pos, freqs);
        RotateHeads(std::span(k).subspan(r * d.kv_heads * d.kd, d.kv_heads * d.kd), d.kv_heads,
                    d.kd, rope, pos, freqs);
      }
      const auto seg_k = std::span<const double>(k).subspan(first * d.kv_heads * d.kd,
                                                            seg.rows * d.kv_heads * d.kd);
      const auto seg_v = std::span<const double>(v).subspan(first * d.kv_heads * d.vd,
                                                            seg.rows * d.kv_heads * d.vd);
      std::vector<std::uint8_t> visible(seg.rows);
      for (std::uint64_t i = 0; i < seg.rows; ++i) {
        const std::uint64_t r = first + i;
        for (std::uint64_t c = 0; c < seg.rows; ++c) {
          visible[c] = c <= i && (d.window == 0 || (i - c) < d.window) ? 1U : 0U;
        }
        AttendCacheRow(std::span(q).subspan(r * d.heads * d.kd, d.heads * d.kd), seg_k, seg_v,
                       seg.rows, visible, d.heads, d.kv_heads, d.kd, d.vd,
                       !d.scale_scores ? static_cast<double>(d.scale) : 1.0,
                       d.scale_scores ? static_cast<double>(d.scale) : 1.0,
                       static_cast<double>(d.softcap), sinks,
                       std::span(attended).subspan(r * d.heads * d.vd, d.heads * d.vd));
      }
      first += seg.rows;
      continue;
    }
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
      RotateHeads(std::span(q).subspan(r * d.heads * d.kd, d.heads * d.kd), d.heads, d.kd, rope,
                  pos, freqs);
      if (d.borrowed) continue;
      RotateHeads(std::span(k).subspan(r * d.kv_heads * d.kd, d.kv_heads * d.kd), d.kv_heads, d.kd,
                  rope, pos, freqs);
      const std::uint64_t cell = ring ? (seg.past + i) % capacity : seg.past + i;
      if (cell >= capacity) return std::unexpected("attention reference: a row past the cache");
      if (auto stored =
              StoreKvCacheRow(kc, cell, k, r, d.kv_heads, d.kd, d.kv_encoding, io.store_row);
          !stored) {
        return stored;
      }
      if (auto stored =
              StoreKvCacheRow(vc, cell, v, r, d.kv_heads, d.vd, d.kv_encoding, io.store_row);
          !stored) {
        return stored;
      }
    }
    const std::uint64_t read = std::min<std::uint64_t>(capacity, seg.read_cells);
    std::vector<std::uint8_t> visible(read);
    for (std::uint64_t i = 0; i < seg.rows; ++i) {
      const std::uint64_t r = first + i;
      const auto p = static_cast<std::int64_t>(seg.past + i);
      for (std::uint64_t c = 0; c < read; ++c) {
        std::int64_t held = -1;
        if (c < end) {
          held = ring ? static_cast<std::int64_t>(c + (((end - 1 - c) / capacity) * capacity))
                      : static_cast<std::int64_t>(c);
        }
        visible[c] = held >= 0 && held <= p && (d.window == 0 || p - held < d.window) ? 1U : 0U;
      }
      AttendCacheRow(
          std::span(q).subspan(r * d.heads * d.kd, d.heads * d.kd), kc, vc, read, visible, d.heads,
          d.kv_heads, d.kd, d.vd, !d.scale_scores ? static_cast<double>(d.scale) : 1.0,
          d.scale_scores ? static_cast<double>(d.scale) : 1.0, static_cast<double>(d.softcap),
          sinks, std::span(attended).subspan(r * d.heads * d.vd, d.heads * d.vd));
    }
    first += seg.rows;
  }
  if (d.output_gate) ApplySigmoidOutputGateRef(attended, qg, rows, d.heads, d.kd, d.vd);
  std::vector<double> projected = Product(weight(kOutput), d.heads * d.vd, width, attended);
  if (d.sandwich) RmsRows(projected, width, d.eps, weight(kPostNorm));
  for (std::size_t i = 0; i < projected.size(); ++i) io.stream[i] += projected[i];
  return {};
}

std::expected<SpatialShape, std::string> DenseSpatialShape(const Instance& instance,
                                                           SpatialShape shape,
                                                           std::optional<SpatialShape> auxiliary) {
  if (!SpatialElements(shape))
    return std::unexpected("spatial dimensions exceed representable storage");
  const auto axes =
      (instance.params.Int("rope_axis1") != 0 || instance.params.Int("rope_axis2") != 0) ? 3U : 1U;
  if (!instance.params.Flag("stateless") || !instance.def->shapes(instance) || shape.width != 1 ||
      shape.images != 1 || shape.channels != instance.stage.width || !auxiliary ||
      *auxiliary != SpatialShape{shape.height, axes, 1, 1} ||
      !SpatialElements({shape.height, shape.height,
                        static_cast<std::uint64_t>(instance.params.Int("heads")), 1}))
    return std::unexpected("spatial attention input or position geometry differs");
  return shape;
}

std::uint64_t DenseSpatialTemporaries(const Instance& instance, SpatialShape shape) {
  return shape.height * shape.height * static_cast<std::uint64_t>(instance.params.Int("heads")) +
         6ULL * *SpatialElements(shape);
}
SpatialShape DenseSpatialCarry(SpatialShape shape) { return {shape.height, shape.height, 1, 1}; }
bool DenseSpatialStateless(const spec::ComponentRef& ref) {
  return std::ranges::any_of(ref.params, [](const spec::Param& p) {
    return p.name == "stateless" && std::holds_alternative<bool>(p.value) &&
           std::get<bool>(p.value);
  });
}
const SpatialOps kAttentionDenseSpatial{
    .infer_shape = DenseSpatialShape,
    .emit = EmitSpatialStream<SpatialStreamLayout::kChannelRows>,
    .reference = RunSpatialStreamReference<>,
    .max_nodes = 64,
    .temporary_values = DenseSpatialTemporaries,
    .auxiliary_encoding = format::Encoding::kI32,
    .needs_carry = true,
    .carry_shape = DenseSpatialCarry,
    .stateless_ref = DenseSpatialStateless};

}  // namespace

const ComponentDef& AttentionDense() {
  static constexpr std::array kNumericalProductRoles{std::string_view{"q"}, std::string_view{"k"},
                                                     std::string_view{"v"},
                                                     std::string_view{"output"}};
  static constexpr std::array kStatelessNumerical{
      NumericalCondition{.parameter = "stateless", .flag = true}};
  static constexpr std::array kNumericalSites{
      NumericalSite{.semantic = "product.block32_q8",
                    .change = ir::NumericalChange::kQ8VectorReduction,
                    .anchors = numerics::kProductAnchors,
                    .roles = kNumericalProductRoles,
                    .encodings = numerics::kBlockWeightEncodings},
      NumericalSite{.semantic = "product.mxfp8_input",
                    .change = ir::NumericalChange::kMxfp8ProductInput,
                    .anchors = numerics::kProductAnchors,
                    .roles = kNumericalProductRoles,
                    .encodings = numerics::kPreparedWeightEncodings},
      NumericalSite{.semantic = "product.bf16_input_and_result",
                    .change = ir::NumericalChange::kBf16ProductInputAndResult,
                    .anchors = numerics::kProductAnchors,
                    .roles = kNumericalProductRoles,
                    .encodings = numerics::kFloatEncodings,
                    .when = kStatelessNumerical},
      NumericalSite{.semantic = "attention.stateless_bf16_intermediates",
                    .change = ir::NumericalChange::kBf16AttentionIntermediates,
                    .anchors = numerics::kAttentionAnchors,
                    .encodings = numerics::kFloatEncodings,
                    .when = kStatelessNumerical},
  };
  static const ComponentDef def{
      .kind = "attention.dense",
      .version = 1,
      .narrows = true,
      .params = kParams,
      .roles = kRoles,
      .numerical = kNumericalSites,
      .shapes = Shapes,
      .state = State,
      .max_state_tensors = 2,

      .batching = nullptr,
      .splits = kSplits,

      .masks = Masks,
      .spatial = &kAttentionDenseSpatial,
      .emit = Emit,
      .reference = Reference,
  };
  return def;
}

const ComponentDef& AttentionBorrowed() {
  static constexpr std::array kNumericalProductRoles{std::string_view{"q"}, std::string_view{"k"},
                                                     std::string_view{"v"},
                                                     std::string_view{"output"}};
  static constexpr std::array kNumericalSites{
      NumericalSite{.semantic = "product.block32_q8",
                    .change = ir::NumericalChange::kQ8VectorReduction,
                    .anchors = numerics::kProductAnchors,
                    .roles = kNumericalProductRoles,
                    .encodings = numerics::kBlockWeightEncodings},
      NumericalSite{.semantic = "product.mxfp8_input",
                    .change = ir::NumericalChange::kMxfp8ProductInput,
                    .anchors = numerics::kProductAnchors,
                    .roles = kNumericalProductRoles,
                    .encodings = numerics::kPreparedWeightEncodings},
  };
  static const ComponentDef def{
      .kind = "attention.borrowed",
      .version = 1,
      .narrows = true,
      .params = kBorrowedParams,
      .roles = kRoles,
      .numerical = kNumericalSites,
      .shapes = Shapes,
      .state = State,
      .max_state_tensors = 2,

      .batching = nullptr,
      .splits = kSplits,

      .masks = Masks,
      .emit = Emit,
      .reference = Reference,
  };
  return def;
}

}  // namespace llmp::components::defs
