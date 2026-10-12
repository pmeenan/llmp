// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Stateless image building blocks over the shared channel-first stream.
// norm.channel uses the VAE's L2 denominator and learned scale. This F32
// component does not impose the existing image kernel's BF16 intermediates.
#include <algorithm>
#include <array>
#include <cmath>
#include <expected>
#include <string>
#include <vector>

#include "components/component.h"
#include "components/defs.h"
#include "components/step.h"
#include "ir/op.h"

namespace llmp::components::defs {
namespace {

constexpr std::array kParams{
    ParamSpec{.name = "eps",
              .type = ParamType::kReal,
              .required = false,
              .least = 1e-30,
              .most = 1.0,
              .fallback = 1e-12},
    ParamSpec{.name = "silu",
              .type = ParamType::kBool,
              .required = false,
              .least = 0,
              .most = 1,
              .fallback = 0},
};
constexpr std::array kRoles{WeightRole{.name = "weight", .encodings = kNormEncodings}};
constexpr std::array kBatch{
    BatchClass{.op = ir::Op::kL2Norm, .batch = ir::Batch::kJoined},
    BatchClass{.op = ir::Op::kScale, .batch = ir::Batch::kJoined},
    BatchClass{.op = ir::Op::kMul, .batch = ir::Batch::kJoined},
    BatchClass{.op = ir::Op::kUnary, .batch = ir::Batch::kJoined},
};

std::expected<std::vector<RoleShape>, std::string> Shapes(const Instance& in) {
  if (in.stage.width == 0 || in.stage.width > kMaxStreamWidth || in.stage.streams != 1)
    return std::unexpected("channel norm requires one bounded channel stream");
  return std::vector<RoleShape>{Role(0, {in.stage.width})};
}

std::expected<FragmentOut, std::string> Emit(Fragment& f) {
  if (!f.stream_needed) return FragmentOut{.stream = f.stream};
  // Channels are ne[0] and pixels/rows ne[1] in the common stream. L2Norm's
  // denominator is max(sqrt(sum(x*x)), eps), not sqrt(mean(x*x) + eps).
  auto out = f.b.L2Norm(f.stream, F32(f.instance.params.Real("eps")));
  out = f.b.Scale(out, F32(std::sqrt(static_cast<double>(f.instance.stage.width))));
  out = f.b.Mul(out, NormScale(f.b, f.Weight(0)));
  if (f.instance.params.Flag("silu")) out = f.b.Unary(out, ir::UnaryFn::kSilu);
  return FragmentOut{.stream = out};
}

std::expected<void, std::string> Reference(RefIO& io) {
  const auto width = io.instance->stage.width;
  if (width == 0 || width > kMaxStreamWidth || io.stream.size() % width != 0 ||
      io.stream.size() / width != io.rows() || io.weights.size() != 1 ||
      io.weights[0].size() != width)
    return std::unexpected("channel norm reference: stream or scale shape differs");
  const double eps = F32(io.instance->params.Real("eps"));
  const double scale = F32(std::sqrt(static_cast<double>(width)));
  for (std::size_t row = 0; row < io.stream.size(); row += width) {
    double squares = 0.0;
    for (std::size_t c = 0; c < width; ++c) squares += io.stream[row + c] * io.stream[row + c];
    const double denominator = std::max(std::sqrt(squares), eps);
    for (std::size_t c = 0; c < width; ++c) {
      double value = (io.stream[row + c] / denominator) * scale * io.weights[0][c];
      if (io.instance->params.Flag("silu")) value /= 1.0 + std::exp(-value);
      io.stream[row + c] = value;
    }
  }
  return {};
}

}  // namespace

const ComponentDef& NormChannel() {
  static const ComponentDef def{
      .kind = "norm.channel",
      .version = 1,
      .row_local = true,
      .params = kParams,
      .roles = kRoles,
      .shapes = Shapes,
      .batch_classes = kBatch,
      .emit = Emit,
      .reference = Reference,
  };
  return def;
}

}  // namespace llmp::components::defs
