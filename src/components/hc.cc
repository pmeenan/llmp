// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: MIT AND Apache-2.0

// Hyper-connections with a sigmoid mix (Qwen3.8; llama.cpp's qwen4exp
// build_hc_mix and build_hc_combine, kernels/ggml/qwen38_graph.cc): the
// residual is `streams` streams of the stage's width, F32 [width, streams,
// rows], and each block reads one stream mixed from them and adds its
// output back to all of them, weighted.
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
// The fragments emit what qwen38_graph.cc's unfused form records, in GGML's
// graph order (its HcMix and HcCombine): the mix's norm, products, gate and
// the folds over the streams; the combine's block repeat, then its inject
// product and weights. Their fusions (llmp.hc.norm, .mix, .combine, and the
// fast form's llmp.hc.prep, which combines the last block's output while it
// normalizes the next mix's input) are the planner's patterns (G3b).

#include <array>
#include <cmath>
#include <cstdint>
#include <expected>
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

constexpr std::array<std::string_view, 1> kMixes{"sigmoid"};
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
    ParamSpec{.name = "eps", .type = ParamType::kReal, .least = 0.0, .most = 1.0},
    // The gate's low rank.
    ParamSpec{.name = "rank", .type = ParamType::kInt, .least = 1, .most = 1 << 16},
    // Its carry for a combine (a block's mix); false: a head's mix.
    MixFlag("inject", 1.0),
    // A head's mix returns the gathered streams as the next feature.
    MixFlag("features", 0.0),
};
enum : std::uint16_t { kMixNorm, kMixDown, kMixUp };
constexpr std::array kMixRoles{
    WeightRole{.name = "norm", .encodings = kF32Only},
    WeightRole{.name = "down"},
    WeightRole{.name = "up"},
};
constexpr std::array<std::string_view, 2> kMixPatterns{"hc_mix", "hc_prep"};

std::expected<std::vector<RoleShape>, std::string> MixShapes(const Instance& in) {
  if (auto ok = Streams(in); !ok) return std::unexpected(ok.error());
  const std::uint64_t hw = in.stage.width * in.stage.streams;
  const auto rank = static_cast<std::uint64_t>(in.params.Int("rank"));
  if (in.params.Flag("inject") == (in.site == Site::kFinalNorm)) {
    return std::unexpected("a block's mix injects, a head's does not");
  }
  if (in.params.Flag("features") && in.site != Site::kFinalNorm) {
    return std::unexpected("features from a mix that is no head's");
  }
  return std::vector<RoleShape>{Role(kMixNorm, {hw}), Role(kMixDown, {hw, rank}),
                                Role(kMixUp, {rank, hw})};
}

std::expected<FragmentOut, std::string> MixEmit(Fragment& f) {
  if (!f.stream_needed) return FragmentOut{};
  ir::Builder& b = f.b;
  const Instance& in = f.instance;
  if (!f.residual.valid()) return std::unexpected("a mix without streams");
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
    flat = b.GetRows(flat, f.step.OutputRows());
    if (in.params.Flag("features")) out.features = flat;
    x = b.Reshape(flat, Shape({w, hc, f.step.outputs()}));
    out.narrowed = true;
  } else if (in.params.Flag("features")) {
    out.features = b.Reshape(x, Shape({w * hc, Desc(b, x).ne(2)}));
  }
  const std::uint64_t rows = Desc(b, x).ne(2);
  ir::TensorId xn = b.RmsNorm(x, eps);
  xn = b.Mul(b.Reshape(xn, Shape({w * hc, rows})), f.Weight(kMixNorm));
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

std::expected<void, std::string> MixReference(RefIO& io) {
  const Instance& in = *io.instance;
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
  const std::vector<double> xn = MixNormed(x, w, in.params.Real("eps"), io.weights.at(kMixNorm));
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
constexpr std::array kCombineRoles{WeightRole{.name = "inject"}};
constexpr std::array<std::string_view, 2> kCombinePatterns{"hc_combine", "hc_prep"};

std::expected<std::vector<RoleShape>, std::string> CombineShapes(const Instance& in) {
  if (auto ok = Streams(in); !ok) return std::unexpected(ok.error());
  const std::uint64_t hc = in.stage.streams;
  return std::vector<RoleShape>{Role(kInject, {in.stage.width * hc, hc})};
}

std::expected<FragmentOut, std::string> CombineEmit(Fragment& f) {
  if (!f.stream_needed) return FragmentOut{};
  ir::Builder& b = f.b;
  const Instance& in = f.instance;
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
