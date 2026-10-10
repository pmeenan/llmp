// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: MIT AND Apache-2.0

// The feed-forward pieces (moe.h), and ffn.moe, version 1: a routed-expert
// feed-forward block with a dense GLU beside it (Gemma 4 26B-A4B's layer,
// llama.cpp's gemma4.cpp). Placement "parallel": the dense GLU of the
// RMS-normed stream, post-normed; the routed experts of the stream normed
// by their own norm, routed from the stream's unweighted, scaled norm
// (moe.h's router), reduced in slot order and post-normed; their sum
// post-normed and added to the stream. Placement "none" (Qwen3.8's layer,
// llama.cpp's qwen4exp.cpp, inside hyper-connections: no norm of its own):
// the routed experts of the stream, routed from the stream itself
// (router_input "stream"), each projection by its per-expert tensor scale
// where `projection_scales` (down's in the reduce), and the shared expert's
// GLU of the stream behind its sigmoid gate (`shared_gate`: the stream's
// product with the gate's row), added to them. The route and reduce shapes
// are its parameters (experts, used, the expert width), not constants.
//
// It splits by experts (expert parallelism): each device holds its own
// experts' matrices whole, the routing and the dense GLU whole on every
// device; the reduce's partial sums join in an all-reduce
// (planner/partition.h).

#include "components/moe.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <expected>
#include <numeric>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "components/component.h"
#include "components/defs.h"
#include "components/step.h"
#include "ir/graph.h"
#include "ir/op.h"

namespace llmp::components {
namespace moe {

ir::TensorId EmitGlu(ir::Builder& b, ir::TensorId h, ir::TensorId gate, ir::TensorId up,
                     ir::TensorId down, ir::GluKind kind) {
  const ir::TensorId g = b.MatMul(gate, h);
  const ir::TensorId u = b.MatMul(up, h);
  return b.MatMul(down, b.Glu(g, u, {.kind = kind}));
}

namespace {
double Act(ir::GluKind kind, double gate) {
  return kind == ir::GluKind::kSwiGlu ? defs::Silu(gate) : defs::GeluTanh(gate);
}
}  // namespace

std::vector<double> GluReference(std::span<const double> h, std::span<const float> gate,
                                 std::span<const float> up, std::span<const float> down,
                                 std::uint64_t width, std::uint64_t hidden, ir::GluKind kind) {
  const std::vector<double> g = defs::Product(gate, width, hidden, h);
  std::vector<double> u = defs::Product(up, width, hidden, h);
  for (std::size_t i = 0; i < u.size(); ++i) u[i] *= Act(kind, g[i]);
  return defs::Product(down, hidden, width, u);
}

namespace {
// The router's input scale (Gemma 4's 1 / sqrt(width), formed in F32 as
// the builders form it).
float InputScale(std::uint64_t width) { return 1.0F / std::sqrt(static_cast<float>(width)); }
}  // namespace

Routed EmitRouter(ir::Builder& b, ir::TensorId x, ir::TensorId router, ir::TensorId scale,
                  const RouterParams& p) {
  const std::uint64_t width = b.tensor(x).ne(0);
  const ir::TensorId in =
      p.rms_input ? b.Mul(b.Scale(b.RmsNorm(x, p.eps), InputScale(width)), scale) : x;
  const ir::Outputs out = b.Route(b.MatMul(router, in), {}, {}, {},
                                  {.kind = ir::RouteKind::kSoftmaxTopK,
                                   .normalize = p.normalize,
                                   .experts = p.experts,
                                   .used = p.used,
                                   .scale = 1.0F});
  return {.ids = out[0], .weights = out[1]};
}

RoutedRef RouterReference(std::span<const double> x, std::uint64_t width,
                          std::span<const float> router, std::span<const float> scale,
                          const RouterParams& p) {
  std::vector<double> in(x.begin(), x.end());
  if (p.rms_input) {
    defs::RmsRows(in, width, p.eps, {});
    const auto s = static_cast<double>(InputScale(width));
    for (std::size_t i = 0; i < in.size(); ++i) {
      in[i] *= s * static_cast<double>(scale[i % width]);
    }
  }
  const std::vector<double> logits = defs::Product(router, width, p.experts, in);
  RoutedRef out;
  const std::size_t rows = in.size() / width;
  std::vector<double> probs(p.experts);
  std::vector<std::uint32_t> order(p.experts);
  for (std::size_t t = 0; t < rows; ++t) {
    const auto row = std::span(logits).subspan(t * p.experts, p.experts);
    const double top = *std::ranges::max_element(row);
    double sum = 0.0;
    for (std::uint32_t e = 0; e < p.experts; ++e) {
      probs[e] = std::exp(row[e] - top);
      sum += probs[e];
    }
    for (double& v : probs) v /= sum;
    std::iota(order.begin(), order.end(), 0U);
    // The highest first; equals in index order.
    std::ranges::stable_sort(order,
                             [&](std::uint32_t a, std::uint32_t b) { return probs[a] > probs[b]; });
    double chosen = 0.0;
    for (std::uint32_t u = 0; u < p.used; ++u) chosen += probs[order[u]];
    for (std::uint32_t u = 0; u < p.used; ++u) {
      out.ids.push_back(order[u]);
      double w = probs[order[u]];
      if (p.normalize) w /= std::max(chosen, 0x1p-14);
      out.weights.push_back(w);
    }
  }
  return out;
}

ir::TensorId EmitExperts(ir::Builder& b, ir::TensorId h, const Routed& routed, ir::TensorId gate,
                         ir::TensorId up, ir::TensorId down, ir::TensorId scale,
                         const ExpertParams& p, ir::TensorId gate_scale, ir::TensorId up_scale) {
  const std::uint64_t rows = b.tensor(h).ne(1);
  const std::uint64_t used = b.tensor(routed.ids).ne(0);
  // Each slot's expert scale, gathered as the builders gather it (the
  // scales repeated a row each, then each row's selected ones): [1, used,
  // rows].
  const auto gathered = [&](ir::TensorId s) {
    const std::uint64_t experts = b.tensor(s).ne(0);
    const ir::TensorId table =
        b.Repeat(b.Reshape(s, defs::Shape({1, experts, 1})), defs::Shape({1, experts, rows}));
    return b.GetRows(table, routed.ids);
  };
  const auto scaled = [&](ir::TensorId t, ir::TensorId s) {
    return s.valid() ? b.Mul(t, gathered(s)) : t;
  };
  const ir::TensorId in = b.Reshape(h, defs::Shape({p.width, 1, rows}));
  ir::TensorId g{};
  ir::TensorId u{};
  if (p.stacked) {
    // The pair's halves: the gate's rows, then the up's.
    const ir::TensorId pair = b.MatMulId(gate, in, routed.ids);
    const auto nb = b.tensor(pair).nb;
    const format::Shape half = defs::Shape({p.hidden, used, rows});
    const auto strides = defs::ViewStrides(nb[0], nb[1], nb[2], half);
    g = scaled(b.View(pair, half, strides, 0), gate_scale);
    u = scaled(b.View(pair, half, strides, p.hidden * nb[0]), up_scale);
  } else {
    g = scaled(b.MatMulId(gate, in, routed.ids), gate_scale);
    u = scaled(b.MatMulId(up, in, routed.ids), up_scale);
  }
  const ir::TensorId act = b.Glu(g, u, {.kind = p.kind});
  const ir::TensorId out = b.MatMulId(down, act, routed.ids);
  if (!scale.valid()) return b.ExpertReduce(out, routed.weights, routed.ids, {});
  // The reduce reads each slot's scale.
  return b.ExpertReduce(out, routed.weights, routed.ids, gathered(scale));
}

std::vector<double> ExpertsReference(std::span<const double> h, const RoutedRef& routed,
                                     std::span<const float> gate, std::span<const float> up,
                                     std::span<const float> down, std::span<const float> scale,
                                     const ExpertParams& p, std::span<const float> gate_scale,
                                     std::span<const float> up_scale) {
  const std::uint64_t w = p.width;
  const std::uint64_t f = p.hidden;
  const std::size_t rows = h.size() / w;
  const std::size_t used = rows == 0 ? 0 : routed.ids.size() / rows;
  std::vector<double> out(w * rows, 0.0);
  for (std::size_t t = 0; t < rows; ++t) {
    const auto x = h.subspan(t * w, w);
    for (std::size_t s = 0; s < used; ++s) {
      const std::uint32_t e = routed.ids[(t * used) + s];
      std::vector<double> gv(f);
      std::vector<double> uv(f);
      for (std::uint64_t j = 0; j < f; ++j) {
        // Stacked: expert e's matrix has 2f rows, the gate's first.
        const std::size_t grow = p.stacked ? (e * 2 * f) + j : (e * f) + j;
        const std::size_t urow = p.stacked ? (e * 2 * f) + f + j : (e * f) + j;
        const std::span<const float> ug = p.stacked ? gate : up;
        double sg = 0.0;
        double su = 0.0;
        for (std::uint64_t k = 0; k < w; ++k) {
          sg += static_cast<double>(gate[(grow * w) + k]) * x[k];
          su += static_cast<double>(ug[(urow * w) + k]) * x[k];
        }
        if (!gate_scale.empty()) sg *= static_cast<double>(gate_scale[e]);
        if (!up_scale.empty()) su *= static_cast<double>(up_scale[e]);
        gv[j] = sg;
        uv[j] = su * Act(p.kind, sg);
      }
      const double factor = scale.empty() ? 1.0 : static_cast<double>(scale[e]);
      const double weight = routed.weights[(t * used) + s];
      for (std::uint64_t i = 0; i < w; ++i) {
        double d = 0.0;
        for (std::uint64_t j = 0; j < f; ++j) {
          d += static_cast<double>(down[(((e * w) + i) * f) + j]) * uv[j];
        }
        out[(t * w) + i] += (d * factor) * weight;
      }
    }
  }
  return out;
}

}  // namespace moe

namespace defs {
namespace {

constexpr std::array<std::string_view, 2> kPlacements{"parallel", "none"};
constexpr std::array<std::string_view, 2> kActivations{"gelu_tanh", "silu"};
constexpr std::array<std::string_view, 1> kRouters{"softmax_topk"};
constexpr std::array<std::string_view, 2> kRouterInputs{"rms_scaled", "stream"};

constexpr ParamSpec Flag(std::string_view name) {
  return {.name = name,
          .type = ParamType::kBool,
          .required = false,
          .least = 0,
          .most = 1,
          .fallback = 0.0};
}

constexpr std::array kParams{
    ParamSpec{.name = "placement", .type = ParamType::kName, .choices = kPlacements},
    ParamSpec{.name = "eps", .type = ParamType::kReal, .least = 0.0, .most = 1.0},
    ParamSpec{.name = "activation", .type = ParamType::kName, .choices = kActivations},
    // The dense GLU's width (the shared expert's, placement "none").
    ParamSpec{.name = "hidden", .type = ParamType::kInt, .least = 1, .most = 1 << 20},
    ParamSpec{.name = "experts", .type = ParamType::kInt, .least = 1, .most = 4096},
    ParamSpec{.name = "used", .type = ParamType::kInt, .least = 1, .most = 64},
    // An expert's GLU width.
    ParamSpec{.name = "expert_hidden", .type = ParamType::kInt, .least = 1, .most = 1 << 20},
    ParamSpec{.name = "router", .type = ParamType::kName, .choices = kRouters},
    ParamSpec{.name = "router_input", .type = ParamType::kName, .choices = kRouterInputs},
    // The selected weights renormalized to sum to 1.
    Flag("normalize"),
    // A learned scale per expert on its down product.
    Flag("expert_scales"),
    // One gate-up array (the gate's rows, then the up's) for the experts.
    Flag("stacked"),
    // A scale per expert on each of its products (NVFP4's tensor scales
    // kept beside the experts): roles gate_scale, up_scale, down_scale.
    Flag("projection_scales"),
    // The dense GLU (the shared expert) times sigmoid(x · shared_gate).
    Flag("shared_gate"),
};

enum : std::uint16_t {
  kNorm,
  kGate,
  kUp,
  kDown,
  kDensePostNorm,
  kRouter,
  kRouterScale,
  kRoutedNorm,
  kGateUpExperts,
  kGateExperts,
  kUpExperts,
  kDownExperts,
  kExpertScale,
  kRoutedPostNorm,
  kPostNorm,
  kGateScale,
  kUpScale,
  kDownScale,
  kSharedGate,
};
constexpr std::array kRoles{
    WeightRole{.name = "norm", .encodings = kF32Only},
    WeightRole{.name = "gate"},
    WeightRole{.name = "up"},
    WeightRole{.name = "down"},
    WeightRole{.name = "dense_post_norm", .encodings = kF32Only},
    WeightRole{.name = "router"},
    WeightRole{.name = "router_scale", .encodings = kF32Only},
    WeightRole{.name = "routed_norm", .encodings = kF32Only},
    WeightRole{.name = "gate_up_experts"},
    WeightRole{.name = "gate_experts"},
    WeightRole{.name = "up_experts"},
    WeightRole{.name = "down_experts"},
    WeightRole{.name = "expert_scale", .encodings = kF32Only},
    WeightRole{.name = "routed_post_norm", .encodings = kF32Only},
    WeightRole{.name = "post_norm", .encodings = kF32Only},
    WeightRole{.name = "gate_scale", .encodings = kF32Only},
    WeightRole{.name = "up_scale", .encodings = kF32Only},
    WeightRole{.name = "down_scale", .encodings = kF32Only},
    WeightRole{.name = "shared_gate"},
};

struct Dims {
  std::uint64_t width, hidden, experts, used, expert_hidden;
  bool parallel, rms_router, normalize, expert_scales, stacked, projection_scales, shared_gate;
  ir::GluKind kind;
  float eps;
};
Dims DimsOf(const Instance& in) {
  const Params& p = in.params;
  return {.width = in.stage.width,
          .hidden = static_cast<std::uint64_t>(p.Int("hidden")),
          .experts = static_cast<std::uint64_t>(p.Int("experts")),
          .used = static_cast<std::uint64_t>(p.Int("used")),
          .expert_hidden = static_cast<std::uint64_t>(p.Int("expert_hidden")),
          .parallel = p.Name("placement") == "parallel",
          .rms_router = p.Name("router_input") == "rms_scaled",
          .normalize = p.Flag("normalize"),
          .expert_scales = p.Flag("expert_scales"),
          .stacked = p.Flag("stacked"),
          .projection_scales = p.Flag("projection_scales"),
          .shared_gate = p.Flag("shared_gate"),
          .kind = p.Name("activation") == "silu" ? ir::GluKind::kSwiGlu : ir::GluKind::kGeGluTanh,
          .eps = F32(p.Real("eps"))};
}

std::expected<std::vector<RoleShape>, std::string> Shapes(const Instance& in) {
  const Dims d = DimsOf(in);
  if (d.used > d.experts) return std::unexpected("more experts used than there are");
  if (d.expert_scales && d.projection_scales) {
    return std::unexpected("a learned expert scale and the products' scales both");
  }
  const std::uint64_t w = d.width;
  const std::uint64_t e = d.experts;
  const std::uint64_t f = d.expert_hidden;
  std::vector<RoleShape> out{Role(kGate, {w, d.hidden}), Role(kUp, {w, d.hidden}),
                             Role(kDown, {d.hidden, w}), Role(kRouter, {w, e}),
                             Role(kDownExperts, {f, w, e})};
  if (d.parallel) {
    // Gemma 4's norms around both branches and their sum.
    for (const std::uint16_t r : {kNorm, kDensePostNorm, kRoutedNorm, kRoutedPostNorm, kPostNorm}) {
      out.push_back(Role(r, {w}));
    }
  }
  if (d.rms_router) out.push_back(Role(kRouterScale, {w}));
  if (d.stacked) {
    out.push_back(Role(kGateUpExperts, {w, 2 * f, e}));
  } else {
    out.push_back(Role(kGateExperts, {w, f, e}));
    out.push_back(Role(kUpExperts, {w, f, e}));
  }
  if (d.expert_scales) out.push_back(Role(kExpertScale, {e}));
  if (d.projection_scales) {
    for (const std::uint16_t r : {kGateScale, kUpScale, kDownScale}) out.push_back(Role(r, {e}));
  }
  if (d.shared_gate) out.push_back(Role(kSharedGate, {w}));
  return out;
}

constexpr std::array kBatchClasses{
    BatchClass{.op = ir::Op::kRmsNorm, .batch = ir::Batch::kJoined},
    BatchClass{.op = ir::Op::kMatMul, .batch = ir::Batch::kJoined},
    BatchClass{.op = ir::Op::kRoute, .batch = ir::Batch::kJoined},
    BatchClass{.op = ir::Op::kMatMulId, .batch = ir::Batch::kJoined},
    BatchClass{.op = ir::Op::kGlu, .batch = ir::Batch::kJoined},
    BatchClass{.op = ir::Op::kExpertReduce, .batch = ir::Batch::kJoined},
};

// Expert parallelism: the expert arrays by their experts (each device its
// experts whole, with their products' scales); the routing and the dense
// GLU whole on every device.
constexpr std::array kExpertRoles{
    RoleSplit{.role = kGateUpExperts, .dim = 2}, RoleSplit{.role = kGateExperts, .dim = 2},
    RoleSplit{.role = kUpExperts, .dim = 2},     RoleSplit{.role = kDownExperts, .dim = 2},
    RoleSplit{.role = kGateScale, .dim = 0},     RoleSplit{.role = kUpScale, .dim = 0},
    RoleSplit{.role = kDownScale, .dim = 0}};
constexpr std::array kSplits{SplitOption{
    .kind = SplitKind::kExperts,
    .collective = ir::Op::kAllReduce,
    // Whole experts per device.
    .units =
        [](const Instance& in) { return static_cast<std::uint32_t>(in.params.Int("experts")); },
    .bytes_per_step = [](const SplitStep& s) -> std::uint64_t {
      return s.instance->stage.width * s.rows * 4;
    },
    .weights = kExpertRoles,
    .state = {}}};

constexpr std::array<std::string_view, 7> kPatterns{
    "rms_norm_mul", "rms_norm_mul_add", "mul_mat_glu", "prepared_products",
    "route_reduce", "moe_router",       "moe_combine"};

std::expected<FragmentOut, std::string> Emit(Fragment& f) {
  if (!f.stream_needed) return FragmentOut{.stream = f.stream};
  ir::Builder& b = f.b;
  const Dims d = DimsOf(f.instance);
  const auto w = [&](std::uint16_t role) { return f.Weight(role); };
  const ir::TensorId x = f.stream;
  const auto norm = [&](ir::TensorId t, std::uint16_t role) {
    return b.Mul(b.RmsNorm(t, d.eps), w(role));
  };
  // The builders' order: the routing, the routed experts and their reduce,
  // then the dense GLU, then the sums.
  const moe::Routed routed =
      moe::EmitRouter(b, x, w(kRouter), d.rms_router ? w(kRouterScale) : ir::TensorId{},
                      {.experts = static_cast<std::uint32_t>(d.experts),
                       .used = static_cast<std::uint32_t>(d.used),
                       .normalize = d.normalize,
                       .eps = d.eps,
                       .rms_input = d.rms_router});
  const ir::TensorId reduced = moe::EmitExperts(
      b, d.parallel ? norm(x, kRoutedNorm) : x, routed,
      w(d.stacked ? kGateUpExperts : kGateExperts), d.stacked ? ir::TensorId{} : w(kUpExperts),
      w(kDownExperts),
      d.expert_scales       ? w(kExpertScale)
      : d.projection_scales ? w(kDownScale)
                            : ir::TensorId{},
      {.width = d.width, .hidden = d.expert_hidden, .stacked = d.stacked, .kind = d.kind},
      d.projection_scales ? w(kGateScale) : ir::TensorId{},
      d.projection_scales ? w(kUpScale) : ir::TensorId{});
  if (!d.parallel) {
    // Inside hyper-connections (Qwen3.8's): the experts' sum and the shared
    // expert, behind its gate; no norm or residual of its own.
    ir::TensorId shared = moe::EmitGlu(b, x, w(kGate), w(kUp), w(kDown), d.kind);
    if (d.shared_gate) {
      const ir::TensorId logit = b.SumRows(b.Mul(x, w(kSharedGate)));
      shared = b.Mul(shared, b.Unary(logit, ir::UnaryFn::kSigmoid));
    }
    return FragmentOut{.stream = b.Add(reduced, shared)};
  }
  const ir::TensorId dense =
      norm(moe::EmitGlu(b, norm(x, kNorm), w(kGate), w(kUp), w(kDown), d.kind), kDensePostNorm);
  ir::TensorId shared = dense;
  if (d.shared_gate) {
    const ir::TensorId logit = b.SumRows(b.Mul(x, w(kSharedGate)));
    shared = b.Mul(shared, b.Unary(logit, ir::UnaryFn::kSigmoid));
  }
  const ir::TensorId sum = b.Add(shared, norm(reduced, kRoutedPostNorm));
  return FragmentOut{.stream = b.Add(norm(sum, kPostNorm), x)};
}

std::expected<void, std::string> Reference(RefIO& io) {
  const Instance& in = *io.instance;
  const Dims d = DimsOf(in);
  if (io.stream.size() != d.width * io.rows()) {
    return std::unexpected("moe reference: a stream of another shape");
  }
  const auto w = [&](std::uint16_t role) -> std::span<const float> { return io.weights.at(role); };
  const auto normed = [&](std::span<const double> t, std::uint16_t role) {
    std::vector<double> out(t.begin(), t.end());
    RmsRows(out, d.width, d.eps, w(role));
    return out;
  };
  const moe::RoutedRef routed = moe::RouterReference(
      io.stream, d.width, w(kRouter), d.rms_router ? w(kRouterScale) : std::span<const float>{},
      {.experts = static_cast<std::uint32_t>(d.experts),
       .used = static_cast<std::uint32_t>(d.used),
       .normalize = d.normalize,
       .eps = d.eps,
       .rms_input = d.rms_router});
  std::vector<double> reduced = moe::ExpertsReference(
      d.parallel ? normed(io.stream, kRoutedNorm) : io.stream, routed,
      w(d.stacked ? kGateUpExperts : kGateExperts),
      d.stacked ? std::span<const float>{} : w(kUpExperts), w(kDownExperts),
      d.expert_scales       ? w(kExpertScale)
      : d.projection_scales ? w(kDownScale)
                            : std::span<const float>{},
      {.width = d.width, .hidden = d.expert_hidden, .stacked = d.stacked, .kind = d.kind},
      d.projection_scales ? w(kGateScale) : std::span<const float>{},
      d.projection_scales ? w(kUpScale) : std::span<const float>{});
  const std::vector<double> x = d.parallel ? normed(io.stream, kNorm) : io.stream;
  std::vector<double> shared =
      moe::GluReference(x, w(kGate), w(kUp), w(kDown), d.width, d.hidden, d.kind);
  if (d.parallel) shared = normed(shared, kDensePostNorm);
  if (d.shared_gate) {
    const std::span<const float> g = w(kSharedGate);
    for (std::size_t t = 0; t < io.rows(); ++t) {
      double logit = 0.0;
      for (std::uint64_t i = 0; i < d.width; ++i) {
        logit += io.stream[(t * d.width) + i] * static_cast<double>(g[i]);
      }
      const double gate = 1.0 / (1.0 + std::exp(-logit));
      for (std::uint64_t i = 0; i < d.width; ++i) shared[(t * d.width) + i] *= gate;
    }
  }
  if (!d.parallel) {
    for (std::size_t i = 0; i < reduced.size(); ++i) reduced[i] += shared[i];
    io.stream = std::move(reduced);
    return {};
  }
  reduced = normed(reduced, kRoutedPostNorm);
  std::vector<double> sum(shared.size());
  for (std::size_t i = 0; i < sum.size(); ++i) sum[i] = shared[i] + reduced[i];
  const std::vector<double> out = normed(sum, kPostNorm);
  for (std::size_t i = 0; i < out.size(); ++i) io.stream[i] += out[i];
  return {};
}

}  // namespace

const ComponentDef& FfnMoe() {
  static const ComponentDef def{
      .kind = "ffn.moe",
      .version = 1,
      .row_local = true,
      .params = kParams,
      .roles = kRoles,
      .shapes = Shapes,
      .state = nullptr,
      .batch_classes = kBatchClasses,
      .batching = nullptr,
      .splits = kSplits,
      .patterns = kPatterns,
      .masks = nullptr,
      .emit = Emit,
      .reference = Reference,
  };
  return def;
}

}  // namespace defs
}  // namespace llmp::components
