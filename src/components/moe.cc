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
// where their individual scale flags are set (down's in the reduce), and the shared expert's
// GLU of the stream behind its sigmoid gate (`shared_gate`: the stream's
// product with the gate's row), added to them. The route and reduce shapes
// are its parameters (experts, used, the expert width), not constants.
// DeepSeek V4's layer (deepseek4.cpp's build_moe_ffn call): placement
// "none", the router over the stream summed in F32 (`router_f32`),
// sqrt(softplus) probabilities chosen by the highest plus a learned bias
// per expert (router "sqrt_softplus"), or in its first layers by a
// token-hash side input (router "hash", its side input `hash`), the
// selected weights renormalized and scaled (`route_scale`); the routed
// experts' and the ungated shared expert's clamped SwiGLU (activation
// "silu_clamp", `limit` and `shared_limit`).
//
// side.hash, version 1: a token-hash side input (DeepSeek V4's
// ffn_gate_tid2eid): each row's `used` experts, the row's token's row of a
// table I32 [used, vocabulary] that the engine reads by rows on demand (a
// host table: the engine checks every id it reads against `experts`
// before the step (runtime/component_admission.h's stored hash-table check), since the
// expert products index memory with them), as ids I32 [used, rows] that
// its layer's hash router reads.
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
#include <cstring>
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
                     ir::TensorId down, ir::GluKind kind, float limit) {
  const ir::TensorId g = b.MatMul(gate, h);
  const ir::TensorId u = b.MatMul(up, h);
  return b.MatMul(down, b.Glu(g, u, {.kind = kind, .limit = limit}));
}

double GluOf(ir::GluKind kind, double gate, double up, float limit) {
  switch (kind) {
    case ir::GluKind::kSwiGlu:
      return defs::Silu(gate) * up;
    case ir::GluKind::kSwiGluClamp: {
      const auto l = static_cast<double>(limit);
      return defs::Silu(std::min(gate, l)) * std::clamp(up, -l, l);
    }
    default:
      return defs::GeluTanh(gate) * up;
  }
}

std::vector<double> GluReference(std::span<const double> h, std::span<const float> gate,
                                 std::span<const float> up, std::span<const float> down,
                                 std::uint64_t width, std::uint64_t hidden, ir::GluKind kind,
                                 float limit) {
  const std::vector<double> g = defs::Product(gate, width, hidden, h);
  std::vector<double> u = defs::Product(up, width, hidden, h);
  for (std::size_t i = 0; i < u.size(); ++i) u[i] = GluOf(kind, g[i], u[i], limit);
  return defs::Product(down, hidden, width, u);
}

namespace {
// The router's input scale (Gemma 4's 1 / sqrt(width), formed in F32 as
// the builders form it).
float InputScale(std::uint64_t width) { return 1.0F / std::sqrt(static_cast<float>(width)); }
}  // namespace

Routed EmitRouter(ir::Builder& b, ir::TensorId x, ir::TensorId router, ir::TensorId scale,
                  const RouterParams& p, ir::TensorId bias, ir::TensorId ids) {
  const std::uint64_t width = b.tensor(x).ne(0);
  const ir::TensorId in =
      p.rms_input ? b.Mul(b.Scale(b.RmsNorm(x, p.eps), InputScale(width)), scale) : x;
  const ir::TensorId logits =
      b.MatMul(router, in, {.accumulate = p.f32 ? ir::Accumulate::kF32 : ir::Accumulate::kDefault});
  const ir::Outputs out = b.Route(logits, bias, ids, {},
                                  {.kind = p.kind,
                                   .normalize = p.normalize,
                                   .experts = p.experts,
                                   .used = p.used,
                                   .scale = p.route_scale});
  return {.ids = out[0], .weights = out[1]};
}

RoutedRef RouterReference(std::span<const double> x, std::uint64_t width,
                          std::span<const float> router, std::span<const float> scale,
                          const RouterParams& p, std::span<const float> bias,
                          std::span<const std::int64_t> ids) {
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
  std::vector<double> selection(p.experts);
  for (std::size_t t = 0; t < rows; ++t) {
    const auto row = std::span(logits).subspan(t * p.experts, p.experts);
    if (p.kind == ir::RouteKind::kSoftmaxTopK) {
      const double top = *std::ranges::max_element(row);
      double sum = 0.0;
      for (std::uint32_t e = 0; e < p.experts; ++e) {
        probs[e] = std::exp(row[e] - top);
        sum += probs[e];
      }
      for (double& v : probs) v /= sum;
    } else if (p.kind == ir::RouteKind::kSigmoidTopK) {
      for (std::uint32_t e = 0; e < p.experts; ++e) {
        probs[e] = 1.0 / (1.0 + std::exp(-row[e]));
      }
    } else {
      // sqrt(softplus(x)) (DeepSeek V4's).
      for (std::uint32_t e = 0; e < p.experts; ++e) {
        const double softplus = row[e] > 20.0 ? row[e] : std::log1p(std::exp(row[e]));
        probs[e] = std::sqrt(softplus);
      }
    }
    for (std::uint32_t e = 0; e < p.experts; ++e) {
      selection[e] = probs[e] + (bias.empty() ? 0.0 : static_cast<double>(bias[e]));
    }
    std::iota(order.begin(), order.end(), 0U);
    if (p.kind == ir::RouteKind::kHash) {
      for (std::uint32_t u = 0; u < p.used; ++u) {
        order[u] = static_cast<std::uint32_t>(ids[(t * p.used) + u]);
      }
    } else {
      // The highest first; equals in index order.
      std::ranges::stable_sort(
          order, [&](std::uint32_t a, std::uint32_t b) { return selection[a] > selection[b]; });
    }
    double chosen = 0.0;
    for (std::uint32_t u = 0; u < p.used; ++u) chosen += probs[order[u]];
    for (std::uint32_t u = 0; u < p.used; ++u) {
      out.ids.push_back(order[u]);
      double w = probs[order[u]];
      if (p.normalize) w /= std::max(chosen, 0x1p-14);
      out.weights.push_back(w * static_cast<double>(p.route_scale));
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
  const ir::TensorId act = b.Glu(g, u, {.kind = p.kind, .limit = p.limit});
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
        uv[j] = GluOf(p.kind, sg, su, p.limit);
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

constexpr std::array<std::string_view, 3> kPlacements{"parallel", "none", "pre"};
// "silu_clamp": SwiGLU with the gate clamped above and up both ways
// (DeepSeek V4's, at `limit` for the routed experts and `shared_limit` for
// the dense GLU).
constexpr std::array<std::string_view, 3> kActivations{"gelu_tanh", "silu", "silu_clamp"};
// "sqrt_softplus": sqrt(softplus) probabilities, the highest plus a bias
// per expert chosen (role router_bias); "hash": each row's experts a
// token-hash side input names (`hash`); both DeepSeek V4's.
constexpr std::array<std::string_view, 4> kRouters{"softmax_topk", "sigmoid_topk", "sqrt_softplus",
                                                   "hash"};
constexpr std::array<std::string_view, 2> kRouterInputs{"rms_scaled", "stream"};

constexpr ParamSpec Real(std::string_view name, double fallback) {
  return {.name = name,
          .type = ParamType::kReal,
          .required = false,
          .least = 0.0,
          .most = 1e6,
          .fallback = fallback};
}

constexpr ParamSpec Flag(std::string_view name) {
  return {.name = name,
          .type = ParamType::kBool,
          .required = false,
          .least = 0,
          .most = 1,
          .fallback = 0.0};
}

constexpr std::array kParams{
    defs::kNormZeroCentered,
    ParamSpec{.name = "placement", .type = ParamType::kName, .choices = kPlacements},
    ParamSpec{.name = "eps", .type = ParamType::kReal, .least = 0.0, .most = 1.0},
    ParamSpec{.name = "activation", .type = ParamType::kName, .choices = kActivations},
    // The dense GLU's width (the shared expert's, placement "none").
    ParamSpec{.name = "hidden",
              .type = ParamType::kInt,
              .required = false,
              .least = 0,
              .most = 1 << 20,
              .fallback = 0},
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
    // Individual projections may have different source encodings and scales.
    Flag("gate_scale"),
    Flag("up_scale"),
    Flag("down_scale"),
    // The dense GLU (the shared expert) times sigmoid(x · shared_gate).
    Flag("shared_gate"),
    // The selected weights' scale (DeepSeek V4's expert_weights_scale).
    Real("route_scale", 1.0),
    // The router's product summed in F32.
    Flag("router_f32"),
    // The clamped SwiGLU's limits: the routed experts', the dense GLU's.
    Real("limit", 0.0),
    Real("shared_limit", 0.0),
    // A hash router's side input (spec.h SideInput: its ids).
    ParamSpec{.name = "hash", .type = ParamType::kName, .required = false},
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
  kRouterBias,
};
constexpr std::array kRoles{
    WeightRole{
        .name = "norm", .encodings = defs::kNormEncodings, .derived = defs::kZeroCenteredNorm},
    WeightRole{.name = "gate"},
    WeightRole{.name = "up"},
    WeightRole{.name = "down"},
    WeightRole{.name = "dense_post_norm",
               .encodings = defs::kNormEncodings,
               .derived = defs::kZeroCenteredNorm},
    WeightRole{.name = "router"},
    WeightRole{.name = "router_scale", .encodings = defs::kNormEncodings},
    WeightRole{.name = "routed_norm",
               .encodings = defs::kNormEncodings,
               .derived = defs::kZeroCenteredNorm},
    WeightRole{.name = "gate_up_experts"},
    WeightRole{
        .name = "gate_experts",
        .import_transform = {.expert = ExpertProjection::kGate, .stacked = "gate_up_experts"}},
    WeightRole{.name = "up_experts", .import_transform = {.expert = ExpertProjection::kUp}},
    WeightRole{.name = "down_experts", .import_transform = {.expert = ExpertProjection::kDown}},
    WeightRole{.name = "expert_scale", .encodings = defs::kNormEncodings},
    WeightRole{.name = "routed_post_norm",
               .encodings = defs::kNormEncodings,
               .derived = defs::kZeroCenteredNorm},
    WeightRole{
        .name = "post_norm", .encodings = defs::kNormEncodings, .derived = defs::kZeroCenteredNorm},
    WeightRole{.name = "gate_scale", .encodings = kF32Only},
    WeightRole{.name = "up_scale", .encodings = kF32Only},
    WeightRole{.name = "down_scale", .encodings = kF32Only},
    // A converter's [1, width] gate matrix: the unit axis dropped.
    WeightRole{.name = "shared_gate",
               .import_transform = {.squeeze_rank = 2, .squeeze_axes = 0b010}},
    WeightRole{.name = "router_bias", .encodings = defs::kNormEncodings},
};

struct Dims {
  std::uint64_t width, hidden, experts, used, expert_hidden;
  bool parallel, pre, rms_router, normalize, expert_scales, stacked, gate_scale, up_scale,
      down_scale, shared_gate;
  ir::GluKind kind;
  float eps;
  ir::RouteKind route;
  float route_scale, limit, shared_limit;
  bool router_f32;
};
ir::GluKind GluKindOf(std::string_view name) {
  if (name == "silu") return ir::GluKind::kSwiGlu;
  if (name == "silu_clamp") return ir::GluKind::kSwiGluClamp;
  return ir::GluKind::kGeGluTanh;
}
ir::RouteKind RouteKindOf(std::string_view name) {
  if (name == "sigmoid_topk") return ir::RouteKind::kSigmoidTopK;
  if (name == "sqrt_softplus") return ir::RouteKind::kSqrtSoftplusBias;
  if (name == "hash") return ir::RouteKind::kHash;
  return ir::RouteKind::kSoftmaxTopK;
}
Dims DimsOf(const Instance& in) {
  const Params& p = in.params;
  const std::string_view placement = p.Name("placement");
  return {.width = in.stage.width,
          .hidden = static_cast<std::uint64_t>(p.Int("hidden")),
          .experts = static_cast<std::uint64_t>(p.Int("experts")),
          .used = static_cast<std::uint64_t>(p.Int("used")),
          .expert_hidden = static_cast<std::uint64_t>(p.Int("expert_hidden")),
          .parallel = placement == "parallel",
          .pre = placement == "pre",
          .rms_router = p.Name("router_input") == "rms_scaled",
          .normalize = p.Flag("normalize"),
          .expert_scales = p.Flag("expert_scales"),
          .stacked = p.Flag("stacked"),
          .gate_scale = p.Flag("projection_scales") || p.Flag("gate_scale"),
          .up_scale = p.Flag("projection_scales") || p.Flag("up_scale"),
          .down_scale = p.Flag("projection_scales") || p.Flag("down_scale"),
          .shared_gate = p.Flag("shared_gate"),
          .kind = GluKindOf(p.Name("activation")),
          .eps = F32(p.Real("eps")),
          .route = RouteKindOf(p.Name("router")),
          .route_scale = F32(p.Real("route_scale")),
          .limit = F32(p.Real("limit")),
          .shared_limit = F32(p.Real("shared_limit")),
          .router_f32 = p.Flag("router_f32")};
}

std::string_view HashInput(const Instance& in) {
  return in.params.Name("router") == "hash" ? in.params.Name("hash") : std::string_view{};
}

std::expected<std::vector<RoleShape>, std::string> Shapes(const Instance& in) {
  if (defs::NormOffsetContradictsFold(in)) {
    return std::unexpected("zero-centered norms contradict the norm.plus_one fold");
  }
  const Dims d = DimsOf(in);
  if (d.parallel && d.hidden == 0) {
    return std::unexpected("parallel MoE requires a dense GLU (hidden > 0)");
  }
  if (d.hidden == 0 && (d.shared_gate || in.params.Given("shared_limit"))) {
    return std::unexpected("shared_gate and shared_limit require a shared expert (hidden > 0)");
  }
  if (d.used > d.experts) return std::unexpected("more experts used than there are");
  const bool valid_limits =
      d.hidden > 0 ? (d.limit > 0.0F && d.shared_limit > 0.0F) : (d.limit > 0.0F);
  if ((d.kind == ir::GluKind::kSwiGluClamp) != valid_limits ||
      (d.kind != ir::GluKind::kSwiGluClamp &&
       (in.params.Given("limit") || in.params.Given("shared_limit")))) {
    return std::unexpected("a clamped SwiGLU's limits, both above zero, and only its");
  }
  if ((d.route == ir::RouteKind::kHash) != in.params.Given("hash")) {
    return std::unexpected("a hash router's side input, and only its");
  }
  if ((d.route == ir::RouteKind::kSqrtSoftplusBias || d.route == ir::RouteKind::kHash) &&
      d.rms_router) {
    return std::unexpected("a DeepSeek router reads the stream itself");
  }
  if (!(d.route_scale > 0.0F)) return std::unexpected("a route scale above zero");
  const std::uint64_t w = d.width;
  const std::uint64_t e = d.experts;
  const std::uint64_t f = d.expert_hidden;
  std::vector<RoleShape> out{Role(kRouter, {w, e}), Role(kDownExperts, {f, w, e})};
  if (d.hidden > 0) {
    out.push_back(Role(kGate, {w, d.hidden}));
    out.push_back(Role(kUp, {w, d.hidden}));
    out.push_back(Role(kDown, {d.hidden, w}));
  }
  if (d.parallel) {
    // Gemma 4's norms around both branches and their sum.
    for (const std::uint16_t r : {kNorm, kDensePostNorm, kRoutedNorm, kRoutedPostNorm, kPostNorm}) {
      out.push_back(Role(r, {w}));
    }
  } else if (d.pre) {
    out.push_back(Role(kNorm, {w}));
  }
  if (d.rms_router) out.push_back(Role(kRouterScale, {w}));
  if (d.stacked) {
    out.push_back(Role(kGateUpExperts, {w, 2 * f, e}));
  } else {
    out.push_back(Role(kGateExperts, {w, f, e}));
    out.push_back(Role(kUpExperts, {w, f, e}));
  }
  if (d.expert_scales) out.push_back(Role(kExpertScale, {e}));
  if (d.gate_scale) out.push_back(Role(kGateScale, {e}));
  if (d.up_scale) out.push_back(Role(kUpScale, {e}));
  if (d.down_scale) out.push_back(Role(kDownScale, {e}));
  if (d.shared_gate) out.push_back(Role(kSharedGate, {w}));
  if (d.route == ir::RouteKind::kSqrtSoftplusBias) out.push_back(Role(kRouterBias, {e}));
  return out;
}

// Expert parallelism: the expert arrays by their experts (each device its
// experts whole, with their products' scales); the routing and the dense
// GLU whole on every device.
constexpr std::array kExpertRoles{
    RoleSplit{.role = kGateUpExperts, .dim = 2}, RoleSplit{.role = kGateExperts, .dim = 2},
    RoleSplit{.role = kUpExperts, .dim = 2},     RoleSplit{.role = kDownExperts, .dim = 2},
    RoleSplit{.role = kExpertScale, .dim = 0},   RoleSplit{.role = kGateScale, .dim = 0},
    RoleSplit{.role = kUpScale, .dim = 0},       RoleSplit{.role = kDownScale, .dim = 0}};
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

std::expected<FragmentOut, std::string> Emit(Fragment& f) {
  if (!f.stream_needed) return FragmentOut{.stream = f.stream};
  ir::Builder& b = f.b;
  const Dims d = DimsOf(f.instance);
  defs::F32Parameters<kRoles.size()> parameters(f);
  const auto w = [&](std::uint16_t role) {
    if (role == kNorm || role == kDensePostNorm || role == kRoutedNorm || role == kRoutedPostNorm ||
        role == kPostNorm) {
      return parameters.Norm(role);
    }
    if (role == kRouterScale || role == kExpertScale || role == kRouterBias ||
        role == kSharedGate) {
      return parameters.Get(role);
    }
    return f.Weight(role);
  };
  const ir::TensorId x = f.stream;
  const auto norm = [&](ir::TensorId t, std::uint16_t role) {
    return b.Mul(b.RmsNorm(t, d.eps), w(role));
  };
  // The builders' order: the routing, the routed experts and their reduce,
  // then the dense GLU, then the sums.
  ir::TensorId hashed{};
  if (d.route == ir::RouteKind::kHash) {
    hashed = f.Side(HashInput(f.instance));
    if (!hashed.valid()) return std::unexpected("a hash router without its side input");
  }
  const ir::TensorId in = d.pre ? norm(x, kNorm) : x;
  // The router fusion is anchored before the shared expert. Materialize its
  // immutable gate parameter here so that the replacement never reads a
  // later Convert, including for source-native F16/BF16 gate rows.
  const ir::TensorId shared_gate = d.shared_gate ? w(kSharedGate) : ir::TensorId{};
  const moe::Routed routed = moe::EmitRouter(
      b, in, w(kRouter), d.rms_router ? w(kRouterScale) : ir::TensorId{},
      {.experts = static_cast<std::uint32_t>(d.experts),
       .used = static_cast<std::uint32_t>(d.used),
       .normalize = d.normalize,
       .eps = d.eps,
       .rms_input = d.rms_router,
       .kind = d.route,
       .route_scale = d.route_scale,
       .f32 = d.router_f32},
      d.route == ir::RouteKind::kSqrtSoftplusBias ? w(kRouterBias) : ir::TensorId{}, hashed);
  const ir::TensorId reduced = moe::EmitExperts(
      b, d.parallel ? norm(x, kRoutedNorm) : in, routed,
      w(d.stacked ? kGateUpExperts : kGateExperts), d.stacked ? ir::TensorId{} : w(kUpExperts),
      w(kDownExperts),
      d.expert_scales && d.down_scale ? b.Mul(w(kExpertScale), w(kDownScale))
      : d.expert_scales               ? w(kExpertScale)
      : d.down_scale                  ? w(kDownScale)
                                      : ir::TensorId{},
      {.width = d.width,
       .hidden = d.expert_hidden,
       .stacked = d.stacked,
       .kind = d.kind,
       .limit = d.limit},
      d.gate_scale ? w(kGateScale) : ir::TensorId{}, d.up_scale ? w(kUpScale) : ir::TensorId{});
  if (!d.parallel) {
    // Inside hyper-connections (Qwen3.8's, DeepSeek V4's) or pre-norm:
    // the experts' sum and optional shared expert.
    ir::TensorId out = reduced;
    if (d.hidden > 0) {
      ir::TensorId shared = moe::EmitGlu(b, in, w(kGate), w(kUp), w(kDown), d.kind, d.shared_limit);
      if (d.shared_gate) {
        const ir::TensorId logit = b.SumRows(b.Mul(in, shared_gate));
        shared = b.Mul(shared, b.Unary(logit, ir::UnaryFn::kSigmoid));
      }
      out = b.Add(reduced, shared);
    }
    if (d.pre) {
      out = b.Add(out, x);
    }
    return FragmentOut{.stream = out};
  }
  const ir::TensorId dense =
      norm(moe::EmitGlu(b, norm(x, kNorm), w(kGate), w(kUp), w(kDown), d.kind, d.shared_limit),
           kDensePostNorm);
  ir::TensorId shared = dense;
  if (d.shared_gate) {
    const ir::TensorId logit = b.SumRows(b.Mul(x, shared_gate));
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
  std::array<std::vector<float>, kRoles.size()> adjusted;
  if (in.params.Flag("norm_zero_centered")) {
    for (const auto role : {kNorm, kDensePostNorm, kRoutedNorm, kRoutedPostNorm, kPostNorm}) {
      if (role < io.weights.size() && !io.weights.at(role).empty()) {
        adjusted[role] = io.weights.at(role);
        for (float& value : adjusted[role]) value += 1.0F;
      }
    }
  }
  const auto w = [&](std::uint16_t role) -> std::span<const float> {
    return adjusted[role].empty() ? io.weights.at(role) : adjusted[role];
  };
  const auto normed = [&](std::span<const double> t, std::uint16_t role) {
    std::vector<double> out(t.begin(), t.end());
    RmsRows(out, d.width, d.eps, w(role));
    return out;
  };
  if (d.route == ir::RouteKind::kHash && (io.side_ids.size() != d.used * io.rows() ||
                                          !std::ranges::all_of(io.side_ids, [&](std::int64_t id) {
                                            return id >= 0 &&
                                                   static_cast<std::uint64_t>(id) < d.experts;
                                          }))) {
    return std::unexpected("moe reference: a hash router's ids, each below the experts");
  }
  const std::vector<double> in_normed = d.pre ? normed(io.stream, kNorm) : std::vector<double>{};
  const std::span<const double> in_ref =
      d.pre ? std::span<const double>(in_normed) : std::span<const double>(io.stream);

  const moe::RoutedRef routed = moe::RouterReference(
      in_ref, d.width, w(kRouter), d.rms_router ? w(kRouterScale) : std::span<const float>{},
      {.experts = static_cast<std::uint32_t>(d.experts),
       .used = static_cast<std::uint32_t>(d.used),
       .normalize = d.normalize,
       .eps = d.eps,
       .rms_input = d.rms_router,
       .kind = d.route,
       .route_scale = d.route_scale,
       .f32 = d.router_f32},
      d.route == ir::RouteKind::kSqrtSoftplusBias ? w(kRouterBias) : std::span<const float>{},
      io.side_ids);
  std::vector<float> combined_scale;
  if (d.expert_scales && d.down_scale) {
    const auto es = w(kExpertScale);
    const auto ds = w(kDownScale);
    combined_scale.resize(std::min(es.size(), ds.size()));
    for (std::size_t i = 0; i < combined_scale.size(); ++i) combined_scale[i] = es[i] * ds[i];
  }
  std::vector<double> reduced =
      moe::ExpertsReference(d.parallel ? normed(io.stream, kRoutedNorm) : in_ref, routed,
                            w(d.stacked ? kGateUpExperts : kGateExperts),
                            d.stacked ? std::span<const float>{} : w(kUpExperts), w(kDownExperts),
                            d.expert_scales && d.down_scale ? std::span<const float>(combined_scale)
                            : d.expert_scales               ? w(kExpertScale)
                            : d.down_scale                  ? w(kDownScale)
                                                            : std::span<const float>{},
                            {.width = d.width,
                             .hidden = d.expert_hidden,
                             .stacked = d.stacked,
                             .kind = d.kind,
                             .limit = d.limit},
                            d.gate_scale ? w(kGateScale) : std::span<const float>{},
                            d.up_scale ? w(kUpScale) : std::span<const float>{});
  std::vector<double> shared{};
  if (d.hidden > 0) {
    shared = moe::GluReference(d.parallel ? normed(io.stream, kNorm) : in_ref, w(kGate), w(kUp),
                               w(kDown), d.width, d.hidden, d.kind, d.shared_limit);
    if (d.parallel) shared = normed(shared, kDensePostNorm);
    if (d.shared_gate) {
      const std::span<const float> g = w(kSharedGate);
      for (std::size_t t = 0; t < io.rows(); ++t) {
        double logit = 0.0;
        for (std::uint64_t i = 0; i < d.width; ++i) {
          logit += in_ref[(t * d.width) + i] * static_cast<double>(g[i]);
        }
        const double gate = 1.0 / (1.0 + std::exp(-logit));
        for (std::uint64_t i = 0; i < d.width; ++i) shared[(t * d.width) + i] *= gate;
      }
    }
  }
  if (!d.parallel) {
    std::vector<double> out = std::move(reduced);
    if (d.hidden > 0) {
      for (std::size_t i = 0; i < out.size(); ++i) out[i] += shared[i];
    }
    if (d.pre) {
      for (std::size_t i = 0; i < out.size(); ++i) out[i] += io.stream[i];
    }
    io.stream = std::move(out);
    return {};
  }
  reduced = normed(reduced, kRoutedPostNorm);
  std::vector<double> sum(shared.size());
  for (std::size_t i = 0; i < sum.size(); ++i) sum[i] = shared[i] + reduced[i];
  const std::vector<double> out = normed(sum, kPostNorm);
  for (std::size_t i = 0; i < out.size(); ++i) io.stream[i] += out[i];
  return {};
}

// ---------------------------------------------------------------- side.hash

constexpr std::array kHashParams{
    ParamSpec{.name = "experts", .type = ParamType::kInt, .least = 1, .most = 4096},
    ParamSpec{.name = "used", .type = ParamType::kInt, .least = 1, .most = 64},
};
constexpr std::array<format::Encoding, 1> kI32{format::Encoding::kI32};
constexpr std::array kHashRoles{WeightRole{.name = "table",
                                           .encodings = kI32,
                                           .rows_on_demand = true,
                                           .import_transform = {.row_table = true}}};

std::expected<std::vector<RoleShape>, std::string> HashShapes(const Instance& in) {
  if (in.params.Int("used") > in.params.Int("experts")) {
    return std::unexpected("more experts used than there are");
  }
  return std::vector<RoleShape>{Role(0, {static_cast<std::uint64_t>(in.params.Int("used")), 0})};
}

std::expected<FragmentOut, std::string> HashEmit(Fragment& f) {
  // Each row's token's row of the table: its experts' ids.
  return FragmentOut{.stream = f.b.GetRows(f.Weight(0), f.step.SideRows(f.instance.index, 1))};
}

// A row's lookup: its token.
std::expected<void, std::string> HashRows(const Instance&, std::span<const std::int32_t> history,
                                          std::uint32_t position, std::span<std::int32_t> out) {
  if (position >= history.size() || out.size() != 1) {
    return std::unexpected("a token-hash row past the history, or of another count");
  }
  out[0] = history[position];
  return {};
}

std::uint64_t HashCount(const Instance&) { return 1; }

std::expected<void, std::string> HashReference(RefIO& io) {
  const Instance& in = *io.instance;
  const auto used = static_cast<std::uint64_t>(in.params.Int("used"));
  const auto experts = static_cast<std::int64_t>(in.params.Int("experts"));
  const std::span<const float> table = io.weights.at(0);
  if (io.side_rows.size() != io.rows()) return std::unexpected("hash reference: its rows");
  io.side_ids.clear();
  for (const std::int64_t token : io.side_rows) {
    if (token < 0 || static_cast<std::uint64_t>(token + 1) * used > table.size()) {
      return std::unexpected("hash reference: a token past the table");
    }
    for (std::uint64_t u = 0; u < used; ++u) {
      const auto id =
          static_cast<std::int64_t>(table[(static_cast<std::uint64_t>(token) * used) + u]);
      if (id < 0 || id >= experts) return std::unexpected("hash reference: an id past the experts");
      io.side_ids.push_back(id);
    }
  }
  return {};
}

// Its stored table, before any routed work: I32 [used, tokens] of
// consistent, aligned bytes (read as I32: a table of any other encoding is
// never reinterpreted) whose every entry names one of its experts, read
// through admission's bounded windows (no whole-table copy).
std::expected<void, std::string> HashCheckStored(const Instance& in, const StoredWeight& weight) {
  const auto experts = in.params.Int("experts");
  const auto used = static_cast<std::uint64_t>(in.params.Int("used"));
  const format::TensorFormat& format = *weight.format;
  if (format.encoding != format::Encoding::kI32 || format.layout != format::Layout::kDense)
    return std::unexpected("stored hash-routing table is not dense I32");
  if (format.shape.rank != 2 || format.shape.ne[0] != used || format.shape.ne[1] == 0 ||
      format.shape.ne[1] > (std::uint64_t{1} << 32U))
    return std::unexpected("stored hash-routing table has another shape");
  if (weight.bytes != format.shape.ne[0] * format.shape.ne[1] * sizeof(std::int32_t))
    return std::unexpected("stored hash-routing table has an inconsistent byte count");
  // Aligned: each window holds whole entries (4,096 is a multiple of 4).
  if (weight.file_offset % sizeof(std::int32_t) != 0)
    return std::unexpected("stored hash-routing table is unaligned");
  bool invalid = false;
  auto read = weight.read(
      [&](std::uint64_t, std::span<const std::byte> bytes) -> std::expected<void, std::string> {
        for (std::size_t at = 0; at + sizeof(std::int32_t) <= bytes.size();
             at += sizeof(std::int32_t)) {
          std::int32_t expert = 0;
          std::memcpy(&expert, bytes.data() + at, sizeof(expert));
          if (expert < 0 || expert >= experts) {
            invalid = true;
            return std::unexpected("invalid");
          }
        }
        return {};
      });
  if (invalid) return std::unexpected("stored hash-routing table names an invalid expert");
  if (!read) return std::unexpected("reading stored hash-routing table: " + read.error());
  return {};
}

}  // namespace

const ComponentDef& SideHash() {
  static const ComponentDef def{
      .kind = "side.hash",
      .version = 1,
      .row_local = true,
      .side_source = true,
      .params = kHashParams,
      .roles = kHashRoles,
      .shapes = HashShapes,

      .emit = HashEmit,
      .reference = HashReference,
      .side_rows = HashRows,
      .side_count = HashCount,
      .check_stored = HashCheckStored,
  };
  return def;
}

const ComponentDef& FfnMoe() {
  static constexpr std::array kNumericalProductRoles{
      std::string_view{"gate"}, std::string_view{"up"}, std::string_view{"down"},
      std::string_view{"router"}, std::string_view{"shared_gate"}};
  static constexpr std::array kNumericalExpertRoles{
      std::string_view{"gate_up_experts"}, std::string_view{"gate_experts"},
      std::string_view{"up_experts"}, std::string_view{"down_experts"}};
  static constexpr std::array kNumericalReduceRoles{std::string_view{"down_experts"}};
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
      NumericalSite{.semantic = "product.selected_block32_q8",
                    .change = ir::NumericalChange::kQ8VectorReduction,
                    .anchors = numerics::kExpertAnchors,
                    .roles = kNumericalExpertRoles,
                    .encodings = numerics::kBlockWeightEncodings},
      NumericalSite{.semantic = "product.selected_mxfp8_input",
                    .change = ir::NumericalChange::kMxfp8ProductInput,
                    .anchors = numerics::kExpertAnchors,
                    .roles = kNumericalExpertRoles,
                    .encodings = numerics::kPreparedWeightEncodings},
      NumericalSite{.semantic = "experts.nvfp4_intermediates",
                    .change = ir::NumericalChange::kNvfp4ExpertIntermediates,
                    .anchors = numerics::kReduceAnchors,
                    .roles = kNumericalReduceRoles,
                    .encodings = numerics::kNvfp4Encoding},
  };
  static const ComponentDef def{
      .kind = "ffn.moe",
      .version = 1,
      .row_local = true,
      .side_input = HashInput,
      .params = kParams,
      .roles = kRoles,
      .numerical = kNumericalSites,
      .shapes = Shapes,
      .state = nullptr,

      .batching = nullptr,
      .splits = kSplits,

      .masks = nullptr,
      .emit = Emit,
      .reference = Reference,
  };
  return def;
}

}  // namespace defs
}  // namespace llmp::components
