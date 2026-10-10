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
    ParamSpec{.name = "placement", .type = ParamType::kName, .choices = kPlacements},
    ParamSpec{.name = "eps", .type = ParamType::kReal, .least = 0.0, .most = 1.0},
    ParamSpec{.name = "activation", .type = ParamType::kName, .choices = kActivations},
    ParamSpec{.name = "hidden", .type = ParamType::kInt, .least = 1, .most = 1 << 20},
};

enum : std::uint16_t { kNorm, kGate, kUp, kDown, kPostNorm };
constexpr std::array kRoles{
    WeightRole{.name = "norm", .encodings = kF32Only},
    WeightRole{.name = "gate"},
    WeightRole{.name = "up"},
    WeightRole{.name = "down"},
    WeightRole{.name = "post_norm", .encodings = kF32Only},
};

bool Sandwich(const Instance& in) { return in.params.Name("placement") == "sandwich"; }
std::uint64_t Hidden(const Instance& in) {
  return static_cast<std::uint64_t>(in.params.Int("hidden"));
}

std::expected<std::vector<RoleShape>, std::string> Shapes(const Instance& in) {
  const std::uint64_t w = in.stage.width;
  const std::uint64_t h = Hidden(in);
  std::vector<RoleShape> out{Role(kNorm, {w}), Role(kGate, {w, h}), Role(kUp, {w, h}),
                             Role(kDown, {h, w})};
  if (Sandwich(in)) out.push_back(Role(kPostNorm, {w}));
  return out;
}

constexpr std::array kBatchClasses{
    BatchClass{.op = ir::Op::kRmsNorm, .batch = ir::Batch::kJoined},
    BatchClass{.op = ir::Op::kMatMul, .batch = ir::Batch::kJoined},
    BatchClass{.op = ir::Op::kGlu, .batch = ir::Batch::kJoined},
};

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

constexpr std::array<std::string_view, 4> kPatterns{"rms_norm_mul", "rms_norm_mul_add",
                                                    "mul_mat_glu", "prepared_products"};

ir::GluKind Kind(const Instance& in) {
  return in.params.Name("activation") == "silu" ? ir::GluKind::kSwiGlu : ir::GluKind::kGeGluTanh;
}

std::expected<FragmentOut, std::string> Emit(Fragment& f) {
  if (!f.stream_needed) return FragmentOut{.stream = f.stream};
  ir::Builder& b = f.b;
  const float eps = F32(f.instance.params.Real("eps"));
  const auto w = [&](std::uint16_t role) { return f.Weight(role); };
  const ir::TensorId x = f.stream;
  const ir::TensorId h = b.Mul(b.RmsNorm(x, eps), w(kNorm));
  const ir::TensorId down = moe::EmitGlu(b, h, w(kGate), w(kUp), w(kDown), Kind(f.instance));
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
  const double eps = F32(in.params.Real("eps"));
  std::vector<double> h = io.stream;
  RmsRows(h, width, eps, io.weights.at(kNorm));
  std::vector<double> down = moe::GluReference(h, io.weights.at(kGate), io.weights.at(kUp),
                                               io.weights.at(kDown), width, hidden, Kind(in));
  if (Sandwich(in)) RmsRows(down, width, eps, io.weights.at(kPostNorm));
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

}  // namespace llmp::components::defs
