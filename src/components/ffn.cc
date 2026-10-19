// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// ffn.glu, version 1: a gated feed-forward block with its norms. Pre- or
// sandwich-normed; gate and up products of the normalized stream; the GLU
// (GELU-tanh: Gemma's GeGLU; or SiLU); the down product; the post norm and
// the residual add. Emitted in today's builders' order: the gate product
// before the up product (the GLU's first operand is visited first).

#include <array>
#include <cstdint>
#include <expected>
#include <string>
#include <string_view>
#include <vector>

#include "components/component.h"
#include "components/defs.h"
#include "components/moe.h"
#include "components/step.h"
#include "ir/graph.h"
#include "ir/op.h"

namespace llmp::components::defs {
namespace {

constexpr std::array<std::string_view, 2> kPlacements{"sandwich", "pre"};
constexpr std::array<std::string_view, 2> kActivations{"gelu_tanh", "silu"};

constexpr std::array kParams{
    kNormZeroCentered,
    ParamSpec{.name = "placement", .type = ParamType::kName, .choices = kPlacements},
    ParamSpec{.name = "eps", .type = ParamType::kReal, .least = 0.0, .most = 1.0},
    ParamSpec{.name = "activation", .type = ParamType::kName, .choices = kActivations},
    ParamSpec{.name = "hidden", .type = ParamType::kInt, .least = 1, .most = 1 << 20},
    ParamSpec{.name = "f16_inputs",
              .type = ParamType::kBool,
              .required = false,
              .least = 0,
              .most = 1,
              .fallback = 0.0},
};

enum : std::uint16_t { kNorm, kGate, kUp, kDown, kPostNorm };
constexpr std::array kRoles{
    WeightRole{.name = "norm", .encodings = kNormEncodings, .derived = kZeroCenteredNorm},
    WeightRole{.name = "gate"},
    WeightRole{.name = "up"},
    WeightRole{.name = "down"},
    WeightRole{.name = "post_norm", .encodings = kNormEncodings, .derived = kZeroCenteredNorm},
};

bool Sandwich(const Instance& in) { return in.params.Name("placement") == "sandwich"; }
std::uint64_t Hidden(const Instance& in) {
  return static_cast<std::uint64_t>(in.params.Int("hidden"));
}

std::expected<std::vector<RoleShape>, std::string> Shapes(const Instance& in) {
  if (NormOffsetContradictsFold(in))
    return std::unexpected("zero-centered norms contradict the norm.plus_one fold");
  const std::uint64_t w = in.stage.width;
  const std::uint64_t h = Hidden(in);
  std::vector<RoleShape> out{Role(kNorm, {w}), Role(kGate, {w, h}), Role(kUp, {w, h}),
                             Role(kDown, {h, w})};
  if (Sandwich(in)) out.push_back(Role(kPostNorm, {w}));
  return out;
}

// By the hidden columns: the gate and up products by their output rows, the
// down product by its input columns (whole blocks of its encoding, which the
// partitioner checks); the down product's partial sums reduce.
constexpr std::array kColumnRoles{RoleSplit{.role = kGate, .dim = 1},
                                  RoleSplit{.role = kUp, .dim = 1},
                                  RoleSplit{.role = kDown, .dim = 0}};
constexpr std::array kSplits{SplitOption{.kind = SplitKind::kColumns,
                                         .collective = ir::Op::kAllReduce,
                                         .units = nullptr,
                                         .bytes_per_step = [](const SplitStep& s) -> std::uint64_t {
                                           return s.instance->stage.width * s.rows * 4;
                                         },
                                         .weights = kColumnRoles,
                                         .state = {}}};

ir::GluKind Kind(const Instance& in) {
  return in.params.Name("activation") == "silu" ? ir::GluKind::kSwiGlu : ir::GluKind::kGeGluTanh;
}

std::expected<FragmentOut, std::string> Emit(Fragment& f) {
  if (!f.stream_needed) return FragmentOut{.stream = f.stream};
  ir::Builder& b = f.b;
  const float eps = F32(f.instance.params.Real("eps"));
  F32Parameters<kRoles.size()> parameters(f);
  const auto w = [&](std::uint16_t role) {
    return role == kNorm || role == kPostNorm ? parameters.Norm(role) : f.Weight(role);
  };
  const ir::TensorId x = f.stream;
  ir::TensorId h = b.Mul(b.RmsNorm(x, eps), NormScale(b, w(kNorm)));
  ir::TensorId down;
  if (f.instance.params.Flag("f16_inputs")) {
    h = b.Convert(h, format::Encoding::kF16);
    const auto gate = b.MatMul(w(kGate), h);
    const auto up = b.MatMul(w(kUp), h);
    const auto glu = b.Glu(gate, up, {.kind = Kind(f.instance)});
    down = b.MatMul(w(kDown), b.Convert(glu, format::Encoding::kF16));
  } else {
    down = moe::EmitGlu(b, h, w(kGate), w(kUp), w(kDown), Kind(f.instance));
  }
  if (!Sandwich(f.instance)) return FragmentOut{.stream = b.Add(down, x)};
  return FragmentOut{.stream = b.Add(b.Mul(b.RmsNorm(down, eps), w(kPostNorm)), x)};
}

std::expected<void, std::string> Reference(RefIO& io) {
  const Instance& in = *io.instance;
  const std::uint64_t width = in.stage.width;
  const std::uint64_t hidden = Hidden(in);
  if (io.stream.size() != width * io.rows()) {
    return std::unexpected("ffn reference: a stream of another shape");
  }
  std::array<std::vector<float>, kRoles.size()> adjusted;
  if (in.params.Flag("norm_zero_centered")) {
    for (const auto role : {kNorm, kPostNorm}) {
      adjusted[role] = io.weights.at(role);
      for (float& value : adjusted[role]) value += 1.0F;
    }
  }
  const auto weight = [&](std::uint16_t role) -> std::span<const float> {
    return adjusted[role].empty() ? io.weights.at(role) : adjusted[role];
  };
  const double eps = F32(in.params.Real("eps"));
  std::vector<double> h = io.stream;
  RmsRows(h, width, eps, weight(kNorm));
  std::vector<double> down;
  if (in.params.Flag("f16_inputs")) {
    for (double& value : h) value = RoundF16(value);
    const auto gate = Product(io.weights.at(kGate), width, hidden, h);
    auto up = Product(io.weights.at(kUp), width, hidden, h);
    for (std::size_t i = 0; i < up.size(); ++i) {
      up[i] = RoundF16(moe::GluOf(Kind(in), gate[i], up[i], 0.0F));
    }
    down = Product(io.weights.at(kDown), hidden, width, up);
  } else {
    down = moe::GluReference(h, io.weights.at(kGate), io.weights.at(kUp), io.weights.at(kDown),
                             width, hidden, Kind(in));
  }
  if (Sandwich(in)) RmsRows(down, width, eps, weight(kPostNorm));
  for (std::size_t i = 0; i < down.size(); ++i) io.stream[i] += down[i];
  return {};
}

}  // namespace

const ComponentDef& FfnGlu() {
  static const ComponentDef def{
      .kind = "ffn.glu",
      .version = 1,
      .row_local = true,
      .params = kParams,
      .roles = kRoles,
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

}  // namespace llmp::components::defs
