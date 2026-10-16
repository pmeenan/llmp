// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: MIT AND Apache-2.0

// Hyper-connections (Qwen3.8's sigmoid mix, llama.cpp's qwen4exp
// build_hc_mix and build_hc_combine, kernels/ggml/qwen38_graph.cc; DeepSeek
// V4's Sinkhorn mix, deepseek4.cpp's build_hc_pre, build_hc_post and
// build_hc_head, and the retired reference-form builder
// kernels/ggml/dsv4_graph.cc): the residual is `streams`
// streams of the stage's width, F32 [width, streams, rows], and each block
// reads one stream mixed from them and adds its output back to all of
// them, weighted.
//
// residual.hc, version 1: a stage's stream mixing (spec.h ResidualStream:
// `mixing`): the embedding's rows repeated into every stream.
//
// hc.mix, version 1: the mix before a block. Each stream's RMS norm, the
// streams as one row [streams · width] times the learned norm weight (xn);
// a low-rank gate sigmoid(up · silu(down · xn / streams)) over that row; the
// gated streams summed and divided by the streams: the block's input
// stream. Its carry is xn (Fragment::carry), from which the combine after
// the block makes its weights. As a stage's final norm (`inject` false) it
// is the head's mix: it gathers the output rows' streams first (the head
// reads only them) and, with `features`, returns those streams as the
// drafter's next feature (an MTP drafter's: its streams before the mix).
//
// hc.combine, version 1: the combine after a block: the block's output
// added to every stream, stream s weighted by 2 · sigmoid(inject · xn / streams)[s].
//
// The Sinkhorn mix (`mix` "sinkhorn"): hc.mix normalizes the streams as one
// row (an unweighted RMS norm over streams · width) and takes their
// product with `fn`, (2 + streams) · streams mixing values a row; the
// first streams of them (times scale[0], plus their base) make each
// stream's weight sigmoid(·) + hc_eps, by which the streams are summed
// (ir::kHcPre), and the block's input is that sum RMS-normed times
// `post_norm` (the block's own input norm: attn_norm, ffn_norm); the next
// streams (times scale[1], plus their base) make the post weights 2 ·
// sigmoid(·) (the carry), and the last streams² the combination
// (ir::kHcMix's Sinkhorn: softmax over destinations, then normalized
// over sources and destinations alternately, `iterations` times: carry2).
// hc.combine then writes stream d as the block's output times post[d]
// plus Σ_s stream s times comb[d, s] (ir::kHcPost). As a final norm (the
// head's mix) the product with `fn` makes `streams` weights (times
// scale[0], plus base, sigmoid, plus hc_eps), the streams' weighted sum is
// RMS-normed times `post_norm` (the final norm), the output rows' streams
// gathered first unless `gather` is off (a drafter's head of every row).
// The fragments emit what the retired dsv4_graph.cc's reference form recorded, in
// GGML's graph order but for the carry's: the post weights and the
// combination right after the mixing product (the order its FFN mix
// records; its attention mix's are recorded before their combine, which
// reads them). Their fusions (llmp.dsv4.hc_mix, hc_pre, hc_post and their
// norms) are the planner's patterns (G4b).
//
// The fragments emit what qwen38_graph.cc's unfused form records, in GGML's
// graph order (its HcMix and HcCombine): the mix's norm, products, gate and
// the folds over the streams; the combine's block repeat, then its inject
// product and weights. Their fusions (llmp.hc.norm, .mix, .combine, and the
// fast form's llmp.hc.prep, which combines the last block's output while it
// normalizes the next mix's input) are the planner's patterns (G3b).

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <expected>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "components/component.h"
#include "components/defs.h"
#include "components/step.h"
#include "ir/graph.h"
#include "ir/op.h"

namespace llmp::components::defs {
namespace {

constexpr std::array kJoined{
    BatchClass{.op = ir::Op::kRmsNorm, .batch = ir::Batch::kJoined},
    BatchClass{.op = ir::Op::kMatMul, .batch = ir::Batch::kJoined},
    BatchClass{.op = ir::Op::kMul, .batch = ir::Batch::kJoined},
    BatchClass{.op = ir::Op::kAdd, .batch = ir::Batch::kJoined},
    BatchClass{.op = ir::Op::kRepeat, .batch = ir::Batch::kJoined},
    BatchClass{.op = ir::Op::kGetRows, .batch = ir::Batch::kJoined},
};

std::expected<void, std::string> Streams(const Instance& in) {
  if (in.stage.streams < 2) return std::unexpected("hyper-connections in a stage of one stream");
  return {};
}

// ---------------------------------------------------------------- residual.hc

constexpr std::array<std::string_view, 2> kMixes{"sigmoid", "sinkhorn"};

bool Sinkhorn(const Instance& in) { return in.params.Name("mix") == "sinkhorn"; }
constexpr std::array kResidualParams{
    // The mix the stage's blocks use (hc.mix's).
    ParamSpec{.name = "mix", .type = ParamType::kName, .choices = kMixes},
};

std::expected<std::vector<RoleShape>, std::string> ResidualShapes(const Instance& in) {
  if (auto ok = Streams(in); !ok) return std::unexpected(ok.error());
  return std::vector<RoleShape>{};
}

std::expected<FragmentOut, std::string> ResidualEmit(Fragment& f) {
  ir::Builder& b = f.b;
  const std::uint64_t w = f.instance.stage.width;
  const std::uint64_t rows = Desc(b, f.stream).ne(1);
  const ir::TensorId one = b.Reshape(f.stream, Shape({w, 1, rows}));
  return FragmentOut{.stream = f.stream,
                     .residual = b.Repeat(one, Shape({w, f.instance.stage.streams, rows}))};
}

std::expected<void, std::string> ResidualReference(RefIO& io) {
  const Instance& in = *io.instance;
  const std::uint64_t w = in.stage.width;
  const std::uint64_t hc = in.stage.streams;
  const std::uint64_t rows = io.rows();
  if (io.stream.size() != w * rows) return std::unexpected("residual.hc reference: a stream");
  io.residual.assign(w * hc * rows, 0.0);
  for (std::uint64_t t = 0; t < rows; ++t) {
    for (std::uint64_t s = 0; s < hc; ++s) {
      for (std::uint64_t i = 0; i < w; ++i)
        io.residual[(((t * hc) + s) * w) + i] = io.stream[(t * w) + i];
    }
  }
  return {};
}

// ------------------------------------------------------------------ hc.mix

constexpr ParamSpec MixFlag(std::string_view name, double fallback) {
  return {.name = name,
          .type = ParamType::kBool,
          .required = false,
          .least = 0,
          .most = 1,
          .fallback = fallback};
}
constexpr std::array kMixParams{
    kNormZeroCentered,
    ParamSpec{.name = "eps", .type = ParamType::kReal, .least = 0.0, .most = 1.0},
    // The sigmoid mix's gate's low rank.
    ParamSpec{
        .name = "rank", .type = ParamType::kInt, .required = false, .least = 0, .most = 1 << 16},
    // Its carry for a combine (a block's mix); false: a head's mix.
    MixFlag("inject", 1.0),
    // A head's mix returns the gathered streams as the next feature.
    MixFlag("features", 0.0),
    // The mix: "sigmoid" (Qwen3.8's) or "sinkhorn" (DeepSeek V4's).
    ParamSpec{.name = "mix", .type = ParamType::kName, .required = false, .choices = kMixes},
    // The Sinkhorn mix's weights' epsilon and its normalizing iterations.
    ParamSpec{.name = "hc_eps",
              .type = ParamType::kReal,
              .required = false,
              .least = 0.0,
              .most = 1.0,
              .fallback = 0.0},
    ParamSpec{
        .name = "iterations", .type = ParamType::kInt, .required = false, .least = 0, .most = 1024},
    // A head's mix gathers the output rows' streams first; false: every row
    // is an output (a drafter's head).
    MixFlag("gather", 1.0),
};
enum : std::uint16_t { kMixNorm, kMixDown, kMixUp, kMixFn, kMixBase, kMixScale, kMixPostNorm };
constexpr std::array kMixRoles{
    WeightRole{.name = "norm", .encodings = kNormEncodings, .derived = kZeroCenteredNorm},
    WeightRole{.name = "down"},
    WeightRole{.name = "up"},
    WeightRole{.name = "fn"},
    WeightRole{.name = "base", .encodings = kF32Only},
    WeightRole{.name = "scale", .encodings = kF32Only},
    WeightRole{.name = "post_norm", .encodings = kF32Only},
};
constexpr std::array<std::string_view, 4> kMixPatterns{"hc_mix", "hc_prep", "dsv4_hc_pre",
                                                       "rms_norm_mul"};

std::expected<std::vector<RoleShape>, std::string> MixShapes(const Instance& in) {
  if (NormOffsetContradictsFold(in))
    return std::unexpected("zero-centered norms contradict the norm.plus_one fold");
  if (Sinkhorn(in) && in.params.Flag("norm_zero_centered"))
    return std::unexpected("zero-centered norms require the sigmoid mix");
  if (auto ok = Streams(in); !ok) return std::unexpected(ok.error());
  const std::uint64_t hc = in.stage.streams;
  const std::uint64_t hw = in.stage.width * hc;
  const bool head = in.site == Site::kFinalNorm;
  if (in.params.Flag("inject") == head) {
    return std::unexpected("a block's mix injects, a head's does not");
  }
  if (in.params.Flag("features") && !head) {
    return std::unexpected("features from a mix that is no head's");
  }
  if (!in.params.Flag("gather") && !head) return std::unexpected("a gather off a head's mix");
  if (Sinkhorn(in)) {
    if (in.params.Given("rank") || in.params.Flag("features")) {
      return std::unexpected("a Sinkhorn mix has no gate rank or features");
    }
    if (!head && in.params.Int("iterations") < 1) {
      return std::unexpected("a Sinkhorn mix's iterations");
    }
    // A block's: pre, post and the combination; a head's: its weights.
    const std::uint64_t mixes = head ? hc : (2 + hc) * hc;
    return std::vector<RoleShape>{Role(kMixFn, {hw, mixes}), Role(kMixBase, {mixes}),
                                  Role(kMixScale, {head ? 1U : 3U}),
                                  Role(kMixPostNorm, {in.stage.width})};
  }
  if (!in.params.Given("rank") || in.params.Int("rank") < 1) {
    return std::unexpected("a sigmoid mix's gate rank");
  }
  const auto rank = static_cast<std::uint64_t>(in.params.Int("rank"));
  return std::vector<RoleShape>{Role(kMixNorm, {hw}), Role(kMixDown, {hw, rank}),
                                Role(kMixUp, {rank, hw})};
}

// One of a mixing product's runs of `hc` values a row (from value `first`):
// a view [hc, rows], as GGML's view_2d of the product.
ir::TensorId MixRun(ir::Builder& b, ir::TensorId mixes, std::uint64_t hc, std::uint64_t first) {
  const ir::TensorDesc d = Desc(b, mixes);
  const format::Shape shape = Shape({hc, d.ne(1)});
  return b.View(mixes, shape, ViewStrides(d.nb[0], d.nb[1], 0, shape), first * d.nb[0]);
}

// `count` values of a vector weight from value `first`: GGML's view_1d.
ir::TensorId Values(ir::Builder& b, ir::TensorId w, std::uint64_t count, std::uint64_t first) {
  const ir::TensorDesc d = Desc(b, w);
  const format::Shape shape = Shape({count});
  return b.View(w, shape, ViewStrides(d.nb[0], d.nb[0] * count, 0, shape), first * d.nb[0]);
}

std::expected<FragmentOut, std::string> SinkhornEmit(Fragment& f) {
  ir::Builder& b = f.b;
  const Instance& in = f.instance;
  const std::uint64_t w = in.stage.width;
  const std::uint64_t hc = in.stage.streams;
  const float eps = F32(in.params.Real("eps"));
  const float hc_eps = F32(in.params.Real("hc_eps"));
  ir::TensorId x = f.residual;
  if (in.site == Site::kFinalNorm) {
    // The head's: the output rows' streams (unless every row is one), the
    // weighted streams' sum of views, as build_hc_head records it.
    if (in.params.Flag("gather") && !f.narrowed) {
      const std::uint64_t rows = Desc(b, x).ne(2);
      ir::TensorId flat = b.Reshape(x, Shape({w * hc, rows}));
      flat = b.GetRows(flat, f.step.OutputRows());
      x = b.Reshape(flat, Shape({w, hc, f.step.outputs()}));
    }
    const ir::TensorDesc xd = Desc(b, x);
    const std::uint64_t rows = xd.ne(2);
    const format::Shape one = Shape({w, rows});
    const auto stream = [&](std::uint64_t h) {
      return b.View(x, one, ViewStrides(xd.nb[0], xd.nb[2], 0, one), h * xd.nb[1]);
    };
    // Stream 0's view, then the weights, then each stream's product.
    const ir::TensorId first = stream(0);
    const ir::TensorId flat = b.Reshape(x, Shape({w * hc, rows}));
    const ir::TensorId mixes = b.MatMul(f.Weight(kMixFn), b.RmsNorm(flat, eps));
    ir::TensorId pre = b.Add(b.Mul(mixes, f.Weight(kMixScale)), f.Weight(kMixBase));
    pre = b.Scale(b.Unary(pre, ir::UnaryFn::kSigmoid), 1.0F, hc_eps);
    const ir::TensorDesc pd = Desc(b, pre);
    const format::Shape weight = Shape({1, rows});
    const auto weight_of = [&](std::uint64_t h) {
      return b.View(pre, weight, ViewStrides(pd.nb[0], pd.nb[1], 0, weight), h * pd.nb[0]);
    };
    ir::TensorId sum = b.Mul(first, weight_of(0));
    for (std::uint64_t h = 1; h < hc; ++h) {
      const ir::TensorId xh = stream(h);
      const ir::TensorId wh = weight_of(h);
      sum = b.Add(sum, b.Mul(xh, wh));
    }
    return FragmentOut{.stream = b.Mul(b.RmsNorm(sum, eps), f.Weight(kMixPostNorm)),
                       .narrowed = in.params.Flag("gather")};
  }
  const std::uint64_t rows = Desc(b, x).ne(2);
  const ir::TensorId flat = b.Reshape(x, Shape({w * hc, rows}));
  const ir::TensorId mixes = b.MatMul(f.Weight(kMixFn), b.RmsNorm(flat, eps));
  const ir::TensorId scale = f.Weight(kMixScale);
  const ir::TensorId base = f.Weight(kMixBase);
  // The post weights and the combination (the carry, where its combine
  // runs), then the pre weights.
  ir::TensorId post{};
  ir::TensorId comb{};
  if (f.carry_needed) {
    const ir::TensorId post_mix = MixRun(b, mixes, hc, hc);
    const ir::TensorId post_scale = Values(b, scale, 1, 1);
    post = b.Mul(post_mix, post_scale);
    const ir::TensorId post_base = Values(b, base, hc, hc);
    post = b.Scale(b.Unary(b.Add(post, post_base), ir::UnaryFn::kSigmoid), 2.0F);
    comb = b.HcMix(mixes, scale, base,
                   {.mix = ir::HcMixKind::kSinkhorn,
                    .streams = static_cast<std::uint32_t>(hc),
                    .iterations = static_cast<std::uint32_t>(in.params.Int("iterations")),
                    .eps = hc_eps});
  }
  const ir::TensorId pre_mix = MixRun(b, mixes, hc, 0);
  const ir::TensorId pre_scale = Values(b, scale, 1, 0);
  ir::TensorId pre = b.Mul(pre_mix, pre_scale);
  const ir::TensorId pre_base = Values(b, base, hc, 0);
  pre = b.Scale(b.Unary(b.Add(pre, pre_base), ir::UnaryFn::kSigmoid), 1.0F, hc_eps);
  const ir::TensorId folded =
      b.HcPre(x, pre, {.fold = ir::HcFold::kWeights, .streams = static_cast<std::uint32_t>(hc)});
  return FragmentOut{.stream = b.Mul(b.RmsNorm(folded, eps), f.Weight(kMixPostNorm)),
                     .carry = post,
                     .carry2 = comb};
}

std::expected<FragmentOut, std::string> MixEmit(Fragment& f) {
  if (!f.stream_needed) return FragmentOut{};
  ir::Builder& b = f.b;
  const Instance& in = f.instance;
  if (!f.residual.valid()) return std::unexpected("a mix without streams");
  if (Sinkhorn(in)) return SinkhornEmit(f);
  F32Parameters<kMixRoles.size()> parameters(f);
  const std::uint64_t w = in.stage.width;
  const std::uint64_t hc = in.stage.streams;
  const float eps = F32(in.params.Real("eps"));
  const auto inv = 1.0F / static_cast<float>(hc);
  FragmentOut out;
  ir::TensorId x = f.residual;
  // A head's mix reads only the output rows' streams: gathered first.
  if (in.site == Site::kFinalNorm && !f.narrowed) {
    const std::uint64_t rows = Desc(b, x).ne(2);
    ir::TensorId flat = b.Reshape(x, Shape({w * hc, rows}));
    const auto segments = f.step.segments();
    if (f.step.phase() == planner::Phase::kDraft && segments.size() == 1) {
      // A draft pass's output rows are its last: a view of them (as the
      // drafter's fast form reads them), no gather.
      const std::uint64_t outputs = f.step.outputs();
      const format::Shape shape = Shape({w * hc, outputs});
      const ir::TensorDesc fd = Desc(b, flat);
      flat = b.View(flat, shape, ViewStrides(fd.nb[0], fd.nb[1], 0, shape),
                    (rows - outputs) * fd.nb[1]);
    } else {
      flat = b.GetRows(flat, f.step.OutputRows());
    }
    if (in.params.Flag("features")) out.features = flat;
    x = b.Reshape(flat, Shape({w, hc, f.step.outputs()}));
    out.narrowed = true;
  } else if (in.params.Flag("features")) {
    out.features = b.Reshape(x, Shape({w * hc, Desc(b, x).ne(2)}));
  }
  const std::uint64_t rows = Desc(b, x).ne(2);
  ir::TensorId xn = b.RmsNorm(x, eps);
  xn = b.Reshape(xn, Shape({w * hc, rows}));
  xn = b.Mul(xn, parameters.Norm(kMixNorm));
  ir::TensorId lo = b.MatMul(f.Weight(kMixDown), xn);
  lo = b.Unary(b.Scale(lo, inv), ir::UnaryFn::kSilu);
  const ir::TensorId gate = b.Unary(b.MatMul(f.Weight(kMixUp), lo), ir::UnaryFn::kSigmoid);
  const ir::TensorId gated = b.Reshape(b.Mul(xn, gate), Shape({w, hc, rows}));
  // The streams' sum: the first stream's view made packed, then the others
  // added in order.
  const auto nb = Desc(b, gated).nb;
  const format::Shape one = Shape({w, rows});
  const auto strides = ViewStrides(nb[0], nb[2], 0, one);
  ir::TensorId mixed = b.Cont(b.View(gated, one, strides, 0));
  for (std::uint64_t s = 1; s < hc; ++s)
    mixed = b.Add(mixed, b.View(gated, one, strides, s * nb[1]));
  out.stream = b.Scale(mixed, inv);
  if (in.params.Flag("inject")) out.carry = xn;
  return out;
}

// xn of `x` [w, hc, rows] (each stream's RMS norm times the weight).
std::vector<double> MixNormed(std::span<const double> x, std::uint64_t w, double eps,
                              std::span<const float> norm) {
  std::vector<double> xn(x.begin(), x.end());
  RmsRows(xn, w, eps, {});
  for (std::size_t i = 0; i < xn.size(); ++i) xn[i] *= static_cast<double>(norm[i % norm.size()]);
  return xn;
}

double Sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

// The Sinkhorn combination of one row's last hc² mixing values (DeepSeek
// V4's): c[dst + hc · src] softmaxed over dst for each src, plus eps; then
// each dst's values over the sources normalized, and alternately each
// src's over the destinations and each dst's again, `iterations` times in
// all (each sum plus eps).
std::vector<double> SinkhornOf(std::span<const double> m, std::span<const float> scale,
                               std::span<const float> base, std::uint64_t hc, double eps,
                               std::int64_t iterations) {
  std::vector<double> c(hc * hc);
  for (std::uint64_t src = 0; src < hc; ++src) {
    double top = -std::numeric_limits<double>::infinity();
    for (std::uint64_t dst = 0; dst < hc; ++dst) {
      const std::uint64_t at = (2 * hc) + dst + (hc * src);
      c[dst + (hc * src)] = (m[at] * static_cast<double>(scale[2])) + static_cast<double>(base[at]);
      top = std::max(top, c[dst + (hc * src)]);
    }
    double sum = 0.0;
    for (std::uint64_t dst = 0; dst < hc; ++dst) {
      c[dst + (hc * src)] = std::exp(c[dst + (hc * src)] - top);
      sum += c[dst + (hc * src)];
    }
    for (std::uint64_t dst = 0; dst < hc; ++dst)
      c[dst + (hc * src)] = (c[dst + (hc * src)] / sum) + eps;
  }
  const auto over_sources = [&] {
    for (std::uint64_t dst = 0; dst < hc; ++dst) {
      double sum = eps;
      for (std::uint64_t src = 0; src < hc; ++src) sum += c[dst + (hc * src)];
      for (std::uint64_t src = 0; src < hc; ++src) c[dst + (hc * src)] /= sum;
    }
  };
  const auto over_destinations = [&] {
    for (std::uint64_t src = 0; src < hc; ++src) {
      double sum = eps;
      for (std::uint64_t dst = 0; dst < hc; ++dst) sum += c[dst + (hc * src)];
      for (std::uint64_t dst = 0; dst < hc; ++dst) c[dst + (hc * src)] /= sum;
    }
  };
  over_sources();
  for (std::int64_t i = 1; i < iterations; ++i) {
    over_destinations();
    over_sources();
  }
  return c;
}

std::expected<void, std::string> SinkhornReference(RefIO& io) {
  const Instance& in = *io.instance;
  const std::uint64_t w = in.stage.width;
  const std::uint64_t hc = in.stage.streams;
  const std::uint64_t hw = w * hc;
  const double eps = F32(in.params.Real("eps"));
  const double hc_eps = F32(in.params.Real("hc_eps"));
  if (io.residual.size() != hw * io.rows()) return std::unexpected("hc.mix reference: streams");
  const std::span<const float> scale = io.weights.at(kMixScale);
  const std::span<const float> base = io.weights.at(kMixBase);
  const bool head = in.site == Site::kFinalNorm;
  std::vector<double> x;
  if (head && in.params.Flag("gather")) {
    std::uint64_t first = 0;
    for (const RefSegment& s : io.segments) {
      for (std::uint64_t i = s.rows - s.outputs; i < s.rows; ++i) {
        const auto row = std::span(io.residual).subspan((first + i) * hw, hw);
        x.insert(x.end(), row.begin(), row.end());
      }
      first += s.rows;
    }
  } else {
    x = io.residual;
  }
  const std::uint64_t rows = x.size() / hw;
  const std::uint64_t mixes = head ? hc : (2 + hc) * hc;
  std::vector<double> normed = x;
  RmsRows(normed, hw, eps, {});
  const std::vector<double> m = Product(io.weights.at(kMixFn), hw, mixes, normed);
  io.stream.assign(w * rows, 0.0);
  if (!head) {
    io.carry.assign(hc * rows, 0.0);
    io.carry2.assign(hc * hc * rows, 0.0);
  }
  for (std::uint64_t t = 0; t < rows; ++t) {
    const auto row = std::span(m).subspan(t * mixes, mixes);
    for (std::uint64_t h = 0; h < hc; ++h) {
      const double pre =
          Sigmoid((row[h] * static_cast<double>(scale[0])) + static_cast<double>(base[h])) + hc_eps;
      for (std::uint64_t i = 0; i < w; ++i)
        io.stream[(t * w) + i] += x[(t * hw) + (h * w) + i] * pre;
      if (!head) {
        io.carry[(t * hc) + h] = 2.0 * Sigmoid((row[hc + h] * static_cast<double>(scale[1])) +
                                               static_cast<double>(base[hc + h]));
      }
    }
    if (!head) {
      const std::vector<double> c =
          SinkhornOf(row, scale, base, hc, hc_eps, in.params.Int("iterations"));
      std::ranges::copy(c, io.carry2.begin() + static_cast<std::ptrdiff_t>(t * hc * hc));
    }
  }
  RmsRows(io.stream, w, eps, io.weights.at(kMixPostNorm));
  return {};
}

std::expected<void, std::string> MixReference(RefIO& io) {
  const Instance& in = *io.instance;
  if (Sinkhorn(in)) return SinkhornReference(io);
  const std::uint64_t w = in.stage.width;
  const std::uint64_t hc = in.stage.streams;
  const std::uint64_t hw = w * hc;
  const auto rank = static_cast<std::uint64_t>(in.params.Int("rank"));
  if (io.residual.size() != hw * io.rows()) return std::unexpected("hc.mix reference: streams");
  std::vector<double> x;
  if (in.site == Site::kFinalNorm) {
    // The output rows' streams.
    std::uint64_t first = 0;
    for (const RefSegment& s : io.segments) {
      for (std::uint64_t i = s.rows - s.outputs; i < s.rows; ++i) {
        const auto row = std::span(io.residual).subspan((first + i) * hw, hw);
        x.insert(x.end(), row.begin(), row.end());
      }
      first += s.rows;
    }
    if (in.params.Flag("features")) io.features_out = x;
  } else {
    x = io.residual;
  }
  const std::uint64_t rows = x.size() / hw;
  std::vector<float> norm = io.weights.at(kMixNorm);
  if (in.params.Flag("norm_zero_centered"))
    for (float& value : norm) value += 1.0F;
  const std::vector<double> xn = MixNormed(x, w, in.params.Real("eps"), norm);
  std::vector<double> lo = Product(io.weights.at(kMixDown), hw, rank, xn);
  for (double& v : lo) v = Silu(v / static_cast<double>(hc));
  const std::vector<double> gate = Product(io.weights.at(kMixUp), rank, hw, lo);
  io.stream.assign(w * rows, 0.0);
  for (std::uint64_t t = 0; t < rows; ++t) {
    for (std::uint64_t s = 0; s < hc; ++s) {
      for (std::uint64_t i = 0; i < w; ++i) {
        const std::size_t at = (t * hw) + (s * w) + i;
        io.stream[(t * w) + i] += xn[at] / (1.0 + std::exp(-gate[at]));
      }
    }
  }
  for (double& v : io.stream) v /= static_cast<double>(hc);
  if (in.params.Flag("inject")) io.carry = xn;
  return {};
}

// -------------------------------------------------------------- hc.combine

enum : std::uint16_t { kInject };
constexpr std::array kCombineParams{
    // The mix it combines after: "sigmoid" (its inject weights) or
    // "sinkhorn" (its mix's post weights and combination).
    ParamSpec{.name = "mix", .type = ParamType::kName, .required = false, .choices = kMixes},
};
constexpr std::array kCombineRoles{WeightRole{.name = "inject"}};
constexpr std::array<std::string_view, 3> kCombinePatterns{"hc_combine", "hc_prep", "dsv4_hc_post"};

std::expected<std::vector<RoleShape>, std::string> CombineShapes(const Instance& in) {
  if (auto ok = Streams(in); !ok) return std::unexpected(ok.error());
  if (Sinkhorn(in)) return std::vector<RoleShape>{};
  const std::uint64_t hc = in.stage.streams;
  return std::vector<RoleShape>{Role(kInject, {in.stage.width * hc, hc})};
}

std::expected<FragmentOut, std::string> CombineEmit(Fragment& f) {
  if (!f.stream_needed) return FragmentOut{};
  ir::Builder& b = f.b;
  const Instance& in = f.instance;
  if (Sinkhorn(in)) {
    if (!f.residual.valid() || !f.carry.valid() || !f.carry2.valid()) {
      return std::unexpected("a combine without streams or its Sinkhorn mix's carry");
    }
    return FragmentOut{
        .stream = f.stream,
        .residual = b.HcPost(f.stream, f.residual, f.carry, f.carry2,
                             {.streams = static_cast<std::uint32_t>(in.stage.streams)})};
  }
  if (!f.residual.valid() || !f.carry.valid()) {
    return std::unexpected("a combine without streams or its mix's carry");
  }
  const std::uint64_t w = in.stage.width;
  const std::uint64_t hc = in.stage.streams;
  const std::uint64_t rows = Desc(b, f.residual).ne(2);
  const ir::TensorId block =
      b.Repeat(b.Reshape(f.stream, Shape({w, 1, rows})), Shape({w, hc, rows}));
  ir::TensorId weights = b.MatMul(f.Weight(kInject), f.carry);
  weights = b.Unary(b.Scale(weights, 1.0F / static_cast<float>(hc)), ir::UnaryFn::kSigmoid);
  weights = b.Reshape(b.Scale(weights, 2.0F), Shape({1, hc, rows}));
  return FragmentOut{.stream = f.stream, .residual = b.Add(f.residual, b.Mul(block, weights))};
}

std::expected<void, std::string> CombineReference(RefIO& io) {
  const Instance& in = *io.instance;
  const std::uint64_t w = in.stage.width;
  const std::uint64_t hc = in.stage.streams;
  const std::uint64_t rows = io.rows();
  if (Sinkhorn(in)) {
    if (io.residual.size() != w * hc * rows || io.stream.size() != w * rows ||
        io.carry.size() != hc * rows || io.carry2.size() != hc * hc * rows) {
      return std::unexpected("hc.combine reference: streams, stream or carries");
    }
    const std::vector<double> x = io.residual;
    for (std::uint64_t t = 0; t < rows; ++t) {
      for (std::uint64_t d = 0; d < hc; ++d) {
        for (std::uint64_t i = 0; i < w; ++i) {
          double v = io.stream[(t * w) + i] * io.carry[(t * hc) + d];
          for (std::uint64_t src = 0; src < hc; ++src) {
            v += x[(((t * hc) + src) * w) + i] * io.carry2[(t * hc * hc) + d + (hc * src)];
          }
          io.residual[(((t * hc) + d) * w) + i] = v;
        }
      }
    }
    return {};
  }
  if (io.residual.size() != w * hc * rows || io.stream.size() != w * rows ||
      io.carry.size() != w * hc * rows) {
    return std::unexpected("hc.combine reference: streams, stream or carry");
  }
  const std::vector<double> inject = Product(io.weights.at(kInject), w * hc, hc, io.carry);
  for (std::uint64_t t = 0; t < rows; ++t) {
    for (std::uint64_t s = 0; s < hc; ++s) {
      const double weight = 2.0 / (1.0 + std::exp(-inject[(t * hc) + s] / static_cast<double>(hc)));
      for (std::uint64_t i = 0; i < w; ++i) {
        io.residual[(((t * hc) + s) * w) + i] += io.stream[(t * w) + i] * weight;
      }
    }
  }
  return {};
}

}  // namespace

const ComponentDef& ResidualHc() {
  static const ComponentDef def{
      .kind = "residual.hc",
      .version = 1,
      .row_local = true,
      .mixing = true,
      .params = kResidualParams,
      .shapes = ResidualShapes,
      .batch_classes = std::span(kJoined).subspan(4, 1),
      .emit = ResidualEmit,
      .reference = ResidualReference,
  };
  return def;
}

const ComponentDef& HcMix() {
  static const ComponentDef def{
      .kind = "hc.mix",
      .version = 1,
      .row_local = true,
      .params = kMixParams,
      .roles = kMixRoles,
      .shapes = MixShapes,
      .batch_classes = kJoined,
      .patterns = kMixPatterns,
      .emit = MixEmit,
      .reference = MixReference,
  };
  return def;
}

const ComponentDef& HcCombine() {
  static const ComponentDef def{
      .kind = "hc.combine",
      .version = 1,
      .row_local = true,
      .params = kCombineParams,
      .roles = kCombineRoles,
      .shapes = CombineShapes,
      .batch_classes = kJoined,
      .patterns = kCombinePatterns,
      .emit = CombineEmit,
      .reference = CombineReference,
  };
  return def;
}

}  // namespace llmp::components::defs
