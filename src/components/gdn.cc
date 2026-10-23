// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: MIT AND Apache-2.0

// attention.gdn, version 1: Gated DeltaNet linear attention (Qwen3.8's
// linear layers; llama.cpp's qwen4exp build_layer_attn_linear with the
// gated delta rule, kernels/ggml/qwen38_graph.cc's LinearAttention) inside
// hyper-connections: no norm or residual of its own (its mix and combine
// are hc.mix and hc.combine). From the stream x:
//
//   - the products: q, k and v heads' channels together (qkv), the output
//     gate z, beta and alpha (a value a value head);
//   - a causal depthwise convolution of the qkv rows over the last
//     `conv - 1` rows before them (the convolution state) and SiLU; the
//     convolution state becomes the last `conv - 1` rows;
//   - q and k L2-normalized per head (llama.cpp's build_gdn_l2_norm:
//     rms_norm(x, eps / d) / sqrt(d), which is x / sqrt(Σ x² + eps));
//   - the gated delta rule (ir::kGatedDeltaNet) over the recurrent state,
//     beta = sigmoid(beta), the decay's log g = softplus(alpha + dt_bias) ·
//     A: value head h reads key head h % k_heads (the value heads tiled:
//     the gdn.value_heads_tiled fold) and A is stored as -exp(A_log) (the
//     ssm.a_neg_exp fold). Explicit a_log instead computes -exp(A_log) in
//     transient F32 storage; the value-head fold remains mandatory;
//   - each head's output RMS-normed, times the norm weight and the declared
//     sigmoid(z) or SiLU(z),
//     and the output product.
//
// Its state is fixed: the convolution's history F32 [(conv - 1) ·
// channels] and the recurrent state F32 [d · d · v_heads], one cell a
// request, rewritten whole by every step. A speculative verify never
// writes them (Rollback::kCommit): in a verify step its fragment reads the
// state and saves each row's convolution input (the qkv rows), the
// recurrence's q, k and v (its convolution's output, q and k L2-normed),
// gate and beta instead (StepContext::Save), from which
// the commit of the kept rows (BuildCommit) advances both.
//
// The fragment emits what the unfused form records, in GGML's graph order:
// the convolution's history first (its store's), the qkv product, the
// convolution and its store, then the recurrence's inputs and the
// recurrence with its state's store, then the gated output; the
// recurrence's attention and state are the two outputs of one node where
// GGML views one result. Its fusions (llmp.gdn.conv, .history, .step,
// .norm_gate, the gates) are the planner's (G3b). Joined owners each run
// their own convolution and recurrence over their rows.

#include <array>
#include <cmath>
#include <cstdint>
#include <expected>
#include <format>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "components/component.h"
#include "components/defs.h"
#include "components/step.h"
#include "ir/graph.h"
#include "ir/op.h"
#include "model/state_contract.h"

namespace llmp::components::defs {
namespace {

constexpr std::array<std::string_view, 3> kPlacements{"none", "pre", "sandwich"};

constexpr std::array<std::string_view, 2> kOutputGates{"sigmoid", "silu"};

constexpr std::array kParams{
    ParamSpec{.name = "output_gate",
              .type = ParamType::kName,
              .required = false,
              .choices = kOutputGates},
    kNormZeroCentered,
    ParamSpec{
        .name = "placement", .type = ParamType::kName, .required = false, .choices = kPlacements},
    ParamSpec{.name = "eps", .type = ParamType::kReal, .least = 0.0, .most = 1.0},
    ParamSpec{.name = "k_heads", .type = ParamType::kInt, .least = 1, .most = 1024},
    ParamSpec{.name = "v_heads", .type = ParamType::kInt, .least = 1, .most = 1024},
    // A head's key and value width.
    ParamSpec{.name = "head_dim", .type = ParamType::kInt, .least = 1, .most = 1024},
    // Source logarithmic A: compute -exp(A_log) in transient graph storage.
    ParamSpec{.name = "a_log",
              .type = ParamType::kBool,
              .required = false,
              .least = 0,
              .most = 1,
              .fallback = 0.0},
    // The convolution's kernel.
    ParamSpec{.name = "conv", .type = ParamType::kInt, .least = 2, .most = 64},
};

enum : std::uint16_t {
  kQkv,
  kGate,
  kBeta,
  kAlpha,
  kDtBias,
  kA,
  kConv,
  kNorm,
  kOutput,
  kInputNorm,
  kPostNorm,
};
constexpr std::array kRoles{
    WeightRole{.name = "qkv", .ordering_fold = "gdn.value_heads_tiled"},
    WeightRole{.name = "gate", .ordering_fold = "gdn.value_heads_tiled"},
    WeightRole{.name = "beta", .ordering_fold = "gdn.value_heads_tiled"},
    WeightRole{.name = "alpha", .ordering_fold = "gdn.value_heads_tiled"},
    WeightRole{
        .name = "dt_bias", .encodings = kNormEncodings, .ordering_fold = "gdn.value_heads_tiled"},
    WeightRole{.name = "a",
               .encodings = kNormEncodings,
               .derived = kLogDecay,
               .ordering_fold = "gdn.value_heads_tiled"},
    WeightRole{
        .name = "conv", .encodings = kNormEncodings, .ordering_fold = "gdn.value_heads_tiled"},
    WeightRole{.name = "norm", .encodings = kNormEncodings},
    WeightRole{.name = "output", .ordering_fold = "gdn.value_heads_tiled"},
    WeightRole{.name = "input_norm", .encodings = kNormEncodings, .derived = kZeroCenteredNorm},
    WeightRole{.name = "post_norm", .encodings = kNormEncodings, .derived = kZeroCenteredNorm},
};

struct Dims {
  std::uint64_t width, hk, hv, d, conv;
  float eps;
  std::string_view placement;
  std::uint64_t channels() const { return (2 * hk * d) + (hv * d); }
};
Dims DimsOf(const Instance& in) {
  const Params& p = in.params;
  return {.width = in.stage.width,
          .hk = static_cast<std::uint64_t>(p.Int("k_heads")),
          .hv = static_cast<std::uint64_t>(p.Int("v_heads")),
          .d = static_cast<std::uint64_t>(p.Int("head_dim")),
          .conv = static_cast<std::uint64_t>(p.Int("conv")),
          .eps = F32(p.Real("eps")),
          .placement = p.Name("placement")};
}

bool Folded(const Instance& in, std::string_view fold) {
  return std::ranges::find(in.stage.folds, fold) != in.stage.folds.end();
}

std::expected<std::vector<RoleShape>, std::string> Shapes(const Instance& in) {
  const Dims d = DimsOf(in);
  if (d.hv % d.hk != 0) return std::unexpected("v_heads are not a multiple of k_heads");
  if (d.hv != d.hk && !Folded(in, "gdn.value_heads_tiled"))
    return std::unexpected("a gated delta rule needs the gdn.value_heads_tiled fold");
  if (in.params.Flag("a_log") == Folded(in, "ssm.a_neg_exp"))
    return std::unexpected("logarithmic A and the ssm.a_neg_exp fold must be exclusive");
  if (d.placement == "none") {
    if (in.params.Flag("norm_zero_centered")) {
      return std::unexpected(
          "attention.gdn does not support norm_zero_centered when placement is none");
    }
  } else {
    if (NormOffsetContradictsFold(in)) {
      return std::unexpected("zero-centered norms contradict the norm.plus_one fold");
    }
    if (d.eps <= 0.0F) {
      return std::unexpected("placement requires eps > 0");
    }
  }
  const std::uint64_t w = d.width;
  std::vector<RoleShape> out{Role(kQkv, {w, d.channels()}),
                             Role(kGate, {w, d.hv * d.d}),
                             Role(kBeta, {w, d.hv}),
                             Role(kAlpha, {w, d.hv}),
                             Role(kDtBias, {d.hv}),
                             Role(kA, {d.hv}),
                             Role(kConv, {d.conv, d.channels()}),
                             Role(kNorm, {d.d}),
                             Role(kOutput, {d.hv * d.d, w})};
  if (d.placement == "pre" || d.placement == "sandwich") {
    out.push_back(Role(kInputNorm, {w}));
  }
  if (d.placement == "sandwich") {
    out.push_back(Role(kPostNorm, {w}));
  }
  return out;
}

model::StateTensorContract Fixed(std::string name, model::StateKind kind, std::uint64_t bytes) {
  return {.name = std::move(name),
          .kind = kind,
          .encoding = format::Encoding::kF32,
          .cell_bytes = bytes,
          .cells = model::CellRule::kFixed,
          .fixed_cells = 1,
          .rollback = model::Rollback::kCommit};
}

model::StateContract State(const Instance& in) {
  const Dims d = DimsOf(in);
  return {
      .tensors = {Fixed("conv", model::StateKind::kConvolution, (d.conv - 1) * d.channels() * 4),
                  Fixed("recurrent", model::StateKind::kRecurrent, d.d * d.d * d.hv * 4)}};
}

// By heads: the value heads' rows of every product with their key heads
// (whole key-head groups), the output product by its columns, the
// recurrent state by its value heads; the partial sums reduce after the
// output product. The convolution's channels (q, k and v together) do not
// split by contiguous rows, so no split is declared until a component
// takes its channels apart (the partitioner then runs it whole).
constexpr std::array<SplitOption, 0> kSplits{};

std::expected<FragmentOut, std::string> Emit(Fragment& f) {
  ir::Builder& b = f.b;
  const Instance& in = f.instance;
  const Dims d = DimsOf(in);
  F32Parameters<kRoles.size()> parameters(f);
  const auto w = [&](std::uint16_t role) {
    if (role == kInputNorm || role == kPostNorm) return parameters.Norm(role);
    return role >= kDtBias && role <= kNorm
               ? parameters.Get(role, role == kA && in.params.Flag("a_log"))
               : f.Weight(role);
  };
  const ir::TensorId x = f.stream;
  const ir::TensorId in_stream = (d.placement == "pre" || d.placement == "sandwich")
                                     ? b.Mul(b.RmsNorm(x, d.eps), parameters.Norm(kInputNorm))
                                     : x;
  const std::uint64_t rows = f.step.rows();
  const std::uint64_t ch = d.channels();
  const std::uint64_t hist = d.conv - 1;
  const auto segments = f.step.segments();
  const bool verify = f.step.phase() == planner::Phase::kVerify;
  const bool one = segments.size() == 1;
  // A segment's rows of a [.., rows, ..] tensor along `dim` (1 or 2): the
  // tensor itself for a lone segment.
  const auto rows_of = [&](ir::TensorId t, std::size_t s, std::size_t dim) {
    if (one) return t;
    const ir::TensorDesc td = Desc(b, t);
    format::Shape shape = td.shape();
    shape.ne.at(dim) = segments[s].rows;
    // The rows' own strides past `dim`: a packed view of a packed tensor.
    auto nb = td.nb;
    for (std::size_t d = dim + 1; d < nb.size(); ++d) nb.at(d) = nb.at(d - 1) * shape.ne.at(d - 1);
    return b.View(t, shape, nb, segments[s].first_row * td.nb.at(dim));
  };
  // The histories first (the convolution's stores read them first).
  std::vector<ir::TensorId> history(segments.size());
  for (std::size_t s = 0; s < segments.size(); ++s) {
    b.SetOwner(static_cast<std::uint8_t>(s));
    history[s] = b.Reshape(f.State(s, 0), Shape({hist, ch, 1}));
    b.SetJoined();
  }
  const ir::TensorId qkv = b.Reshape(b.MatMul(w(kQkv), in_stream), Shape({ch, rows, 1}));
  // Each owner's convolution input: its history, then its rows transposed
  // into packed rows; the new history its last `hist` columns.
  std::vector<ir::TensorId> inputs(segments.size());
  for (std::size_t s = 0; s < segments.size(); ++s) {
    b.SetOwner(static_cast<std::uint8_t>(s));
    if (verify) {
      f.step.Save(in, "qkv", b.Reshape(rows_of(qkv, s, 1), Shape({ch, segments[s].rows})),
                  static_cast<std::uint8_t>(s));
    }
    const ir::TensorId cols = b.Cont(b.Transpose(rows_of(qkv, s, 1)));
    inputs[s] = b.Concat(history[s], cols, 0);
    if (!verify) {
      const ir::TensorDesc it = Desc(b, inputs[s]);
      const ir::TensorId tail =
          b.View(inputs[s], Shape({hist, ch, 1}), {it.nb[0], it.nb[1], it.nb[2], it.nb[2]},
                 (it.ne(0) - hist) * it.nb[0]);
      const ir::TensorId kept = b.Reshape(b.Cont(tail), Shape({hist * ch, 1}));
      b.CacheStore(f.State(s, 0), kept, f.step.StateRow());
    }
    b.SetJoined();
  }
  // The recurrence's inputs, its run and its state's store, per owner; the
  // gates' products are joined (made at the first owner's).
  ir::TensorId gate{};
  ir::TensorId beta{};
  std::vector<ir::TensorId> outputs;
  for (std::size_t s = 0; s < segments.size(); ++s) {
    b.SetOwner(static_cast<std::uint8_t>(s));
    const ir::TensorId conv = b.Unary(b.SsmConv(inputs[s], w(kConv)), ir::UnaryFn::kSilu);
    const std::uint64_t n = segments[s].rows;
    const std::uint64_t el = 4;
    const auto head = [&](std::uint64_t heads, std::uint64_t first) {
      const format::Shape shape = Shape({d.d, heads, n, 1});
      return b.View(conv, shape, {el, d.d * el, ch * el, ch * n * el}, first * el);
    };
    const auto l2 = [&](ir::TensorId t) {
      const auto dims = static_cast<float>(d.d);
      return b.Scale(b.RmsNorm(b.Cont(t), d.eps / dims), 1.0F / std::sqrt(dims));
    };
    const ir::TensorId q = l2(head(d.hk, 0));
    const ir::TensorId k = l2(head(d.hk, d.hk * d.d));
    const ir::TensorId v = head(d.hv, 2 * d.hk * d.d);
    if (verify) {
      // The recurrence's inputs, q and k normed, as one row a token [ch, n]
      // (the convolution's channels): what the commit replays.
      const std::uint64_t qk = d.hk * d.d;
      const ir::TensorId q_rows = b.Reshape(q, Shape({qk, n}));
      const ir::TensorId k_rows = b.Reshape(k, Shape({qk, n}));
      const ir::TensorId both = b.Concat(q_rows, k_rows, 0);
      const format::Shape v_shape = Shape({d.hv * d.d, n});
      const ir::TensorId v_view =
          b.View(conv, v_shape, ViewStrides(el, ch * el, 0, v_shape), 2 * qk * el);
      const ir::TensorId v_rows = b.Cont(v_view);
      f.step.Save(in, "conv", b.Concat(both, v_rows, 0), static_cast<std::uint8_t>(s));
    }
    b.SetJoined();
    if (!gate.valid()) {
      const ir::TensorId alpha = b.Reshape(b.MatMul(w(kAlpha), in_stream), Shape({d.hv, rows, 1}));
      gate = b.Mul(b.Unary(b.Add(alpha, w(kDtBias)), ir::UnaryFn::kSoftplus), w(kA));
      gate = b.Reshape(gate, Shape({1, d.hv, rows, 1}));
      beta = b.Unary(b.Reshape(b.MatMul(w(kBeta), in_stream), Shape({1, d.hv, rows, 1})),
                     ir::UnaryFn::kSigmoid);
    }
    b.SetOwner(static_cast<std::uint8_t>(s));
    if (verify) {
      const std::uint64_t kept = segments[s].rows;
      f.step.Save(in, "gate", b.Reshape(rows_of(gate, s, 2), Shape({d.hv, kept})),
                  static_cast<std::uint8_t>(s));
      f.step.Save(in, "beta", b.Reshape(rows_of(beta, s, 2), Shape({d.hv, kept})),
                  static_cast<std::uint8_t>(s));
    }
    const ir::TensorId state = b.Reshape(f.State(s, 1), Shape({d.d, d.d, d.hv, 1}));
    const ir::Outputs ran =
        b.GatedDeltaNet(q, k, v, rows_of(gate, s, 2), rows_of(beta, s, 2), state);
    if (!verify) {
      const ir::TensorId recurrent = b.Reshape(ran[1], Shape({d.d * d.d * d.hv, 1}));
      b.CacheStore(f.State(s, 1), recurrent, f.step.StateRow());
    }
    outputs.push_back(ran[0]);
    b.SetJoined();
  }
  if (!f.stream_needed) return FragmentOut{.stream = x};
  ir::TensorId out = outputs.front();
  for (std::size_t s = 1; s < outputs.size(); ++s) out = b.Concat(out, outputs[s], 2);
  // The gated norm: each head's RMS norm times the weight and its declared gate.
  const ir::TensorId normed = b.Mul(b.RmsNorm(out, d.eps), w(kNorm));
  const ir::TensorId z = b.Reshape(b.MatMul(w(kGate), in_stream), Shape({d.d, d.hv, rows, 1}));
  const ir::TensorId gated =
      b.Mul(normed, b.Unary(z, in.params.Name("output_gate") == "silu" ? ir::UnaryFn::kSilu
                                                                       : ir::UnaryFn::kSigmoid));
  ir::TensorId projected = b.MatMul(w(kOutput), b.Reshape(gated, Shape({d.d * d.hv, rows})));
  if (d.placement == "sandwich") {
    projected = b.Mul(b.RmsNorm(projected, d.eps), parameters.Norm(kPostNorm));
  }
  if (d.placement == "pre" || d.placement == "sandwich") {
    projected = b.Add(projected, x);
  }
  return FragmentOut{.stream = projected};
}

// The commit of a verify's kept rows: from each owner's saves of them, its
// convolution history (the old history, then the kept rows' qkv) and its
// recurrence over the kept rows' saved q, k and v, gate and beta.
std::expected<void, std::string> Commit(Fragment& f) {
  ir::Builder& b = f.b;
  const Instance& in = f.instance;
  const Dims d = DimsOf(in);
  const std::uint64_t ch = d.channels();
  const std::uint64_t hist = d.conv - 1;
  const auto segments = f.step.segments();
  for (std::size_t s = 0; s < segments.size(); ++s) {
    const std::uint64_t n = segments[s].rows;
    b.SetOwner(static_cast<std::uint8_t>(s));
    const ir::TensorId history = b.Reshape(f.State(s, 0), Shape({hist, ch, 1}));
    const ir::TensorId qkv = b.Reshape(f.step.Saved(in, "qkv", s, ch), Shape({ch, n, 1}));
    const ir::TensorId input = b.Concat(history, b.Cont(b.Transpose(qkv)), 0);
    const ir::TensorDesc it = Desc(b, input);
    const ir::TensorId tail =
        b.View(input, Shape({hist, ch, 1}), {it.nb[0], it.nb[1], it.nb[2], it.nb[2]},
               (it.ne(0) - hist) * it.nb[0]);
    const ir::TensorId kept = b.Reshape(b.Cont(tail), Shape({hist * ch, 1}));
    b.CacheStore(f.State(s, 0), kept, f.step.StateRow());
    // The recurrence's inputs as the verify saved them (q and k normed).
    const ir::TensorId conv = f.step.Saved(in, "conv", s, ch);
    const auto head = [&](std::uint64_t heads, std::uint64_t first) {
      return b.View(conv, Shape({d.d, heads, n, 1}), {4, d.d * 4, ch * 4, ch * n * 4}, first * 4);
    };
    const ir::TensorId q = head(d.hk, 0);
    const ir::TensorId k = head(d.hk, d.hk * d.d);
    const ir::TensorId v = head(d.hv, 2 * d.hk * d.d);
    const ir::TensorId gate = b.Reshape(f.step.Saved(in, "gate", s, d.hv), Shape({1, d.hv, n, 1}));
    const ir::TensorId beta = b.Reshape(f.step.Saved(in, "beta", s, d.hv), Shape({1, d.hv, n, 1}));
    const ir::TensorId state = b.Reshape(f.State(s, 1), Shape({d.d, d.d, d.hv, 1}));
    const ir::Outputs ran = b.GatedDeltaNet(q, k, v, gate, beta, state);
    const ir::TensorId recurrent = b.Reshape(ran[1], Shape({d.d * d.d * d.hv, 1}));
    b.CacheStore(f.State(s, 1), recurrent, f.step.StateRow());
    b.SetJoined();
  }
  return {};
}

// ------------------------------------------------------------- reference

double Softplus(double x) { return x > 20.0 ? x : std::log1p(std::exp(x)); }
double Sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

std::expected<void, std::string> Reference(RefIO& io) {
  const Instance& in = *io.instance;
  const Dims d = DimsOf(in);
  const std::uint64_t w = d.width;
  const std::uint64_t ch = d.channels();
  const std::uint64_t hist = d.conv - 1;
  const std::uint64_t rows = io.rows();
  if (io.stream.size() != w * rows || io.state.size() != io.segments.size()) {
    return std::unexpected("gdn reference: a stream or state of another shape");
  }
  std::array<std::vector<float>, kRoles.size()> adjusted;
  if (in.params.Flag("norm_zero_centered")) {
    for (const auto role : {kInputNorm, kPostNorm}) {
      if (role < io.weights.size() && !io.weights.at(role).empty()) {
        adjusted[role] = io.weights.at(role);
        for (float& value : adjusted[role]) value += 1.0F;
      }
    }
  }
  const auto weight = [&](std::uint16_t role) {
    return std::span<const float>(adjusted[role].empty() ? io.weights.at(role) : adjusted[role]);
  };
  const std::vector<double> residual = io.stream;
  std::vector<double> in_stream = io.stream;
  if (d.placement == "pre" || d.placement == "sandwich") {
    RmsRows(in_stream, w, d.eps, weight(kInputNorm));
  }
  const std::vector<double> qkv = Product(weight(kQkv), w, ch, in_stream);
  const std::vector<double> z = Product(weight(kGate), w, d.hv * d.d, in_stream);
  const std::vector<double> braw = Product(weight(kBeta), w, d.hv, in_stream);
  const std::vector<double> araw = Product(weight(kAlpha), w, d.hv, in_stream);
  const auto conv_w = weight(kConv);  // [conv, ch]: tap j of channel c at c · conv + j
  std::vector<double> out(rows * d.hv * d.d, 0.0);
  std::uint64_t first = 0;
  for (std::size_t si = 0; si < io.segments.size(); ++si) {
    const RefSegment& seg = io.segments[si];
    auto& history = io.state[si].at(0);    // [hist, ch]: time fastest
    auto& recurrent = io.state[si].at(1);  // [d (key), d (value), hv]
    // The convolution input: the history, then the rows, time fastest.
    std::vector<double> sx((hist + seg.rows) * ch);
    for (std::uint64_t c = 0; c < ch; ++c) {
      for (std::uint64_t j = 0; j < hist; ++j)
        sx[(c * (hist + seg.rows)) + j] = history[(c * hist) + j];
      for (std::uint64_t t = 0; t < seg.rows; ++t) {
        sx[(c * (hist + seg.rows)) + hist + t] = qkv[((first + t) * ch) + c];
      }
    }
    for (std::uint64_t c = 0; c < ch; ++c) {
      for (std::uint64_t j = 0; j < hist; ++j) {
        history[(c * hist) + j] =
            Stored(format::Encoding::kF32, sx[(c * (hist + seg.rows)) + seg.rows + j]);
      }
    }
    for (std::uint64_t t = 0; t < seg.rows; ++t) {
      const std::uint64_t r = first + t;
      std::vector<double> conv(ch);
      for (std::uint64_t c = 0; c < ch; ++c) {
        double s = 0.0;
        for (std::uint64_t j = 0; j < d.conv; ++j) {
          s += sx[(c * (hist + seg.rows)) + t + j] * static_cast<double>(conv_w[(c * d.conv) + j]);
        }
        conv[c] = Silu(s);
      }
      const auto l2 = [&](std::uint64_t at) {
        double ss = 0.0;
        for (std::uint64_t i = 0; i < d.d; ++i) ss += conv[at + i] * conv[at + i];
        const double inv = 1.0 / std::sqrt(ss + static_cast<double>(d.eps));
        for (std::uint64_t i = 0; i < d.d; ++i) conv[at + i] *= inv;
      };
      for (std::uint64_t h = 0; h < 2 * d.hk; ++h) l2(h * d.d);
      for (std::uint64_t h = 0; h < d.hv; ++h) {
        const std::uint64_t kh = h % d.hk;  // tiled
        const double* q = &conv[kh * d.d];
        const double* k = &conv[(d.hk + kh) * d.d];
        const double* v = &conv[(2 * d.hk * d.d) + (h * d.d)];
        const double beta = Sigmoid(braw[(r * d.hv) + h]);
        const double g = Softplus(araw[(r * d.hv) + h] + static_cast<double>(weight(kDtBias)[h])) *
                         (in.params.Flag("a_log") ? -static_cast<double>(std::exp(weight(kA)[h]))
                                                  : static_cast<double>(weight(kA)[h]));
        const double decay = std::exp(g);
        double* s = &recurrent[h * d.d * d.d];  // s[i + d · c]
        for (std::uint64_t i = 0; i < d.d * d.d; ++i) s[i] *= decay;
        for (std::uint64_t c = 0; c < d.d; ++c) {
          double kv = 0.0;
          for (std::uint64_t i = 0; i < d.d; ++i) kv += s[i + (d.d * c)] * k[i];
          const double delta = beta * (v[c] - kv);
          for (std::uint64_t i = 0; i < d.d; ++i) s[i + (d.d * c)] += k[i] * delta;
        }
        double* o = &out[(r * d.hv * d.d) + (h * d.d)];
        const double scale = 1.0 / std::sqrt(static_cast<double>(d.d));
        for (std::uint64_t c = 0; c < d.d; ++c) {
          double a = 0.0;
          for (std::uint64_t i = 0; i < d.d; ++i) a += s[i + (d.d * c)] * q[i];
          o[c] = a * scale;
        }
      }
    }
    // The state as stored (F32).
    for (double& v : recurrent) v = Stored(format::Encoding::kF32, v);
    first += seg.rows;
  }
  RmsRows(out, d.d, d.eps, weight(kNorm));
  for (std::size_t i = 0; i < out.size(); ++i)
    out[i] *= in.params.Name("output_gate") == "silu" ? Silu(z[i]) : Sigmoid(z[i]);
  std::vector<double> projected = Product(weight(kOutput), d.hv * d.d, w, out);
  if (d.placement == "sandwich") {
    RmsRows(projected, w, d.eps, weight(kPostNorm));
  }
  if (d.placement == "pre" || d.placement == "sandwich") {
    for (std::size_t i = 0; i < projected.size(); ++i) {
      projected[i] += residual[i];
    }
  }
  io.stream = std::move(projected);
  return {};
}

}  // namespace

const ComponentDef& AttentionGdn() {
  // Existing numerical evidence covers the sigmoid contract. A SiLU component
  // needs separately applicable evidence before those arithmetic rewrites.
  static constexpr std::array kSigmoidGate{
      NumericalCondition{.parameter = "output_gate", .choice = "sigmoid"}};
  static constexpr std::array kNumericalProductRoles{
      std::string_view{"qkv"}, std::string_view{"gate"}, std::string_view{"beta"},
      std::string_view{"alpha"}, std::string_view{"output"}};
  // These product results feed the convolution/history and output gate.
  // The stored matrices are MXFP8; BF16 names their rounded results.
  static constexpr std::array kBf16ResultRoles{std::string_view{"qkv"}, std::string_view{"gate"}};
  static constexpr std::array kBf16ResultWeights{format::Encoding::kMxfp8};
  static constexpr std::array kNumericalSites{
      NumericalSite{.semantic = "product.block32_q8",
                    .change = ir::NumericalChange::kQ8VectorReduction,
                    .anchors = numerics::kProductAnchors,
                    .roles = kNumericalProductRoles,
                    .encodings = numerics::kBlockWeightEncodings,
                    .when = kSigmoidGate},
      NumericalSite{.semantic = "product.mxfp8_input",
                    .change = ir::NumericalChange::kMxfp8ProductInput,
                    .anchors = numerics::kProductAnchors,
                    .roles = kNumericalProductRoles,
                    .encodings = numerics::kPreparedWeightEncodings,
                    .when = kSigmoidGate},
      NumericalSite{.semantic = "gdn.product_bf16_intermediates",
                    .change = ir::NumericalChange::kBf16ProductInputAndResult,
                    .anchors = numerics::kProductAnchors,
                    .roles = kBf16ResultRoles,
                    .encodings = kBf16ResultWeights,
                    .when = kSigmoidGate},
  };
  static const ComponentDef def{
      .kind = "attention.gdn",
      .version = 1,
      .params = kParams,
      .roles = kRoles,
      .numerical = kNumericalSites,
      .shapes = Shapes,
      .state = State,
      .max_state_tensors = 2,

      .splits = kSplits,

      .emit = Emit,
      .reference = Reference,
      .commit = Commit,
      .max_commit_saves = 4,
  };
  return def;
}

}  // namespace llmp::components::defs
