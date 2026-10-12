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
#include <string_view>
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

// Qwen-Image's DiT uses scale-only adaptive LayerNorm (no shift) and a
// tanh gate. Conditioning is a dynamic operand, never a learned constant.
// The caller supplies rows in the same owner order as the stream. Its
// eventual producer may broadcast per-owner modulation on device instead
// of expanding it on the CPU; this fragment does not prescribe transport.
constexpr std::array kAdaptiveParams{
    ParamSpec{.name = "side", .type = ParamType::kName},
    ParamSpec{.name = "eps",
              .type = ParamType::kReal,
              .required = false,
              .least = 1e-30,
              .most = 1.0,
              .fallback = 1e-6},
};
constexpr std::array kGatedParams{ParamSpec{.name = "side", .type = ParamType::kName}};
constexpr std::array kAdaptiveBatch{
    BatchClass{.op = ir::Op::kLayerNorm, .batch = ir::Batch::kJoined},
    BatchClass{.op = ir::Op::kScale, .batch = ir::Batch::kJoined},
    BatchClass{.op = ir::Op::kMul, .batch = ir::Batch::kJoined},
};
constexpr std::array kGatedBatch{
    BatchClass{.op = ir::Op::kUnary, .batch = ir::Batch::kJoined},
    BatchClass{.op = ir::Op::kMul, .batch = ir::Batch::kJoined},
    BatchClass{.op = ir::Op::kAdd, .batch = ir::Batch::kJoined},
};

std::string_view Conditioning(const Instance& in) { return in.params.Name("side"); }

std::expected<std::vector<RoleShape>, std::string> AdaptiveShapes(const Instance& in) {
  if (in.site != Site::kLayer || in.stage.width == 0 || in.stage.width > kMaxStreamWidth ||
      in.stage.streams != 1 || Conditioning(in).empty())
    return std::unexpected(
        "adaptive blocks require one bounded layer stream and a named side input");
  return std::vector<RoleShape>{};
}

bool JoinedRows(const Fragment& f, ir::TensorId id) {
  if (!id.valid()) return false;
  const auto t = f.b.tensor(id);
  return t.format.encoding == format::Encoding::kF32 && t.shape().rank == 2 &&
         t.ne(0) == f.instance.stage.width && t.ne(1) == f.step.rows() && t.ne(2) == 1 &&
         t.ne(3) == 1 && t.segment == ir::kJoinedSegment;
}

std::expected<FragmentOut, std::string> AdaptiveEmit(Fragment& f) {
  if (!f.stream_needed) return FragmentOut{.stream = f.stream};
  const auto side = f.Side(Conditioning(f.instance));
  if (!JoinedRows(f, f.stream) || !JoinedRows(f, side))
    return std::unexpected(
        "adaptive norm requires matching F32 stream and conditioning owner rows");
  const auto norm = f.b.LayerNorm(f.stream, F32(f.instance.params.Real("eps")));
  const auto out = f.b.Mul(norm, f.b.Scale(side, 1.0F, 1.0F));
  // Retain the original stream, before normalization and before the branch.
  return FragmentOut{.stream = out, .carry = f.stream};
}

std::expected<FragmentOut, std::string> GatedEmit(Fragment& f) {
  if (!f.stream_needed) return FragmentOut{.stream = f.stream};
  const auto gate = f.Side(Conditioning(f.instance));
  if (!JoinedRows(f, f.stream) || !JoinedRows(f, gate) || !JoinedRows(f, f.carry))
    return std::unexpected(
        "gated residual requires matching F32 branch, gate and saved owner rows");
  const auto gated = f.b.Mul(f.stream, f.b.Unary(gate, ir::UnaryFn::kTanh));
  return FragmentOut{.stream = f.b.Add(f.carry, gated)};
}

bool RefRows(const RefIO& io) {
  const auto width = io.instance->stage.width;
  return width > 0 && width <= kMaxStreamWidth && io.instance->stage.streams == 1 &&
         io.stream.size() % width == 0 && io.stream.size() / width == io.rows() &&
         io.side.size() == io.stream.size() && io.weights.empty();
}

std::expected<void, std::string> AdaptiveReference(RefIO& io) {
  if (!RefRows(io)) return std::unexpected("adaptive norm reference: stream or conditioning shape");
  const auto width = io.instance->stage.width;
  const double eps = F32(io.instance->params.Real("eps"));
  io.carry = io.stream;
  for (std::size_t row = 0; row < io.stream.size(); row += width) {
    double mean = 0.0;
    for (std::size_t c = 0; c < width; ++c) mean += io.stream[row + c];
    mean /= static_cast<double>(width);
    double variance = 0.0;
    for (std::size_t c = 0; c < width; ++c) {
      const double d = io.stream[row + c] - mean;
      variance += d * d;
    }
    const double denominator = std::sqrt(variance / static_cast<double>(width) + eps);
    for (std::size_t c = 0; c < width; ++c)
      io.stream[row + c] = (io.stream[row + c] - mean) / denominator * (1.0 + io.side[row + c]);
  }
  return {};
}

std::expected<void, std::string> GatedReference(RefIO& io) {
  if (!RefRows(io) || io.carry.size() != io.stream.size())
    return std::unexpected("gated residual reference: branch, gate or saved shape");
  for (std::size_t i = 0; i < io.stream.size(); ++i)
    io.stream[i] = io.carry[i] + std::tanh(io.side[i]) * io.stream[i];
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

const ComponentDef& NormAdaptive() {
  static const ComponentDef def{
      .kind = "norm.adaptive",
      .version = 1,
      .row_local = true,
      .side_input = Conditioning,
      .params = kAdaptiveParams,
      .shapes = AdaptiveShapes,
      .batch_classes = kAdaptiveBatch,
      .emit = AdaptiveEmit,
      .reference = AdaptiveReference,
  };
  return def;
}

const ComponentDef& ResidualGated() {
  static const ComponentDef def{
      .kind = "residual.gated",
      .version = 1,
      .row_local = true,
      .side_input = Conditioning,
      .params = kGatedParams,
      .shapes = AdaptiveShapes,
      .batch_classes = kGatedBatch,
      .emit = GatedEmit,
      .reference = GatedReference,
  };
  return def;
}

}  // namespace llmp::components::defs
