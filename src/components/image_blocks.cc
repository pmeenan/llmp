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
#include "components/spatial.h"
#include "components/step.h"
#include "ir/op.h"

namespace llmp::components::defs {
namespace {

constexpr std::array kParams{
    ParamSpec{.name = "channels",
              .type = ParamType::kInt,
              .required = false,
              .least = 0,
              .most = 65536,
              .fallback = 0},
    ParamSpec{.name = "weight_axis",
              .type = ParamType::kInt,
              .required = false,
              .least = 0,
              .most = 3,
              .fallback = 0},
    ParamSpec{.name = "weight_rank",
              .type = ParamType::kInt,
              .required = false,
              .least = 1,
              .most = 4,
              .fallback = 1},
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

std::uint64_t Channels(const Instance& in) {
  const auto explicit_width = in.params.Int("channels");
  return explicit_width == 0 ? in.stage.width : static_cast<std::uint64_t>(explicit_width);
}
std::expected<std::vector<RoleShape>, std::string> Shapes(const Instance& in) {
  if (Channels(in) == 0 || Channels(in) > kMaxStreamWidth || in.stage.streams != 1)
    return std::unexpected("channel norm requires one bounded channel stream");
  const auto axis = static_cast<std::size_t>(in.params.Int("weight_axis"));
  auto role = Role(0, {1});
  role.rank = static_cast<std::uint8_t>(in.params.Int("weight_rank"));
  if (axis >= role.rank) return std::unexpected("channel gamma axis exceeds source rank");
  role.ne[axis] = Channels(in);
  return std::vector{role};
}
ir::TensorId Normalized(ir::Builder& builder, const Instance& in, ir::TensorId input,
                        ir::TensorId weight) {
  auto out = builder.L2Norm(input, F32(in.params.Real("eps")));
  out = builder.Scale(out, F32(std::sqrt(static_cast<double>(Channels(in)))));
  auto scale = NormScale(builder, weight);
  if (builder.tensor(scale).shape().rank != 1)
    scale = builder.Reshape(scale, ir::ShapeOf({Channels(in)}));
  out = builder.Mul(out, scale);
  if (in.params.Flag("silu")) out = builder.Unary(out, ir::UnaryFn::kSilu);
  return out;
}
std::expected<FragmentOut, std::string> Emit(Fragment& f) {
  if (!f.stream_needed) return FragmentOut{.stream = f.stream};
  if (Channels(f.instance) != f.instance.stage.width)
    return std::unexpected("token channel norm must retain its stage width");
  return FragmentOut{.stream = Normalized(f.b, f.instance, f.stream, f.Weight(0))};
}
std::expected<SpatialShape, std::string> SpatialNormShape(const Instance& in, SpatialShape shape,
                                                          std::optional<SpatialShape> auxiliary) {
  if (!SpatialElements(shape))
    return std::unexpected("spatial channel norm dimensions exceed representable storage");
  if (auxiliary || shape.channels != Channels(in))
    return std::unexpected("spatial channel norm input or auxiliary differs");
  return shape;
}
std::expected<SpatialValue, std::string> SpatialNorm(SpatialFragment& f) {
  auto shape = SpatialNormShape(f.instance, f.shape,
                                f.auxiliary ? std::optional(f.auxiliary->shape) : std::nullopt);
  if (!shape || f.weights.size() != 1)
    return std::unexpected("spatial channel norm needs one actual gamma source");
  // Reuse the same channel-first fragment. No numerical cast at the layout boundary.
  auto rows = f.builder.Cont(f.builder.Permute(f.input, {1, 2, 0, 3}));
  auto out = Normalized(f.builder, f.instance, rows, f.weights[0]);
  return SpatialValue{f.builder.Cont(f.builder.Permute(out, {2, 0, 1, 3})), *shape};
}
std::expected<void, std::string> SpatialNormReference(SpatialRefIO& io) {
  if (!SpatialNormShape(io.instance, io.shape, io.auxiliary_shape) || !io.auxiliary.empty() ||
      !io.bias.empty() || io.shape.channels != Channels(io.instance) ||
      io.kernel.size() != io.shape.channels ||
      io.input.size() != io.shape.width * io.shape.height * io.shape.channels * io.shape.images)
    return std::unexpected("spatial channel norm reference sizes differ");
  const auto pixels = io.shape.width * io.shape.height;
  const auto channels = io.shape.channels;
  const double eps = F32(io.instance.params.Real("eps"));
  const double factor = F32(std::sqrt(static_cast<double>(channels)));
  std::vector<double> out(io.input.size());
  for (std::uint64_t image = 0; image < io.shape.images; ++image)
    for (std::uint64_t pixel = 0; pixel < pixels; ++pixel) {
      double squares = 0;
      for (std::uint64_t c = 0; c < channels; ++c) {
        const double value = io.input[pixel + pixels * (c + channels * image)];
        squares += value * value;
      }
      const double denominator = std::max(std::sqrt(squares), eps);
      for (std::uint64_t c = 0; c < channels; ++c) {
        const auto index = pixel + pixels * (c + channels * image);
        double value = io.input[index] / denominator * factor * io.kernel[c];
        if (io.instance.params.Flag("silu")) value /= 1 + std::exp(-value);
        out[index] = value;
      }
    }
  io.output = std::move(out);
  return {};
}
const SpatialOps kSpatialNorm{SpatialNormShape, SpatialNorm, SpatialNormReference, 12};

std::expected<void, std::string> Reference(RefIO& io) {
  const auto width = io.instance->stage.width;
  if (Channels(*io.instance) != width || width == 0 || width > kMaxStreamWidth ||
      io.stream.size() % width != 0 || io.stream.size() / width != io.rows() ||
      io.weights.size() != 1 || io.weights[0].size() != width)
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

ir::TensorId AdaptiveMath(ir::Builder& b, ir::TensorId x, ir::TensorId side, float eps) {
  return b.Mul(b.LayerNorm(x, eps), b.Scale(side, 1.0F, 1.0F));
}
ir::TensorId GatedMath(ir::Builder& b, ir::TensorId x, ir::TensorId gate, ir::TensorId carry) {
  return b.Add(carry, b.Mul(x, b.Unary(gate, ir::UnaryFn::kTanh)));
}
std::expected<FragmentOut, std::string> AdaptiveEmit(Fragment& f) {
  if (!f.stream_needed) return FragmentOut{.stream = f.stream};
  const auto side = f.Side(Conditioning(f.instance));
  if (!JoinedRows(f, f.stream) || !JoinedRows(f, side))
    return std::unexpected(
        "adaptive norm requires matching F32 stream and conditioning owner rows");
  const auto out = AdaptiveMath(f.b, f.stream, side, F32(f.instance.params.Real("eps")));
  // Retain the original stream, before normalization and before the branch.
  return FragmentOut{.stream = out, .carry = f.stream};
}

std::expected<FragmentOut, std::string> GatedEmit(Fragment& f) {
  if (!f.stream_needed) return FragmentOut{.stream = f.stream};
  const auto gate = f.Side(Conditioning(f.instance));
  if (!JoinedRows(f, f.stream) || !JoinedRows(f, gate) || !JoinedRows(f, f.carry))
    return std::unexpected(
        "gated residual requires matching F32 branch, gate and saved owner rows");
  return FragmentOut{.stream = GatedMath(f.b, f.stream, gate, f.carry)};
}

bool RefRows(const RefIO& io) {
  const auto width = io.instance->stage.width;
  return width > 0 && width <= kMaxStreamWidth && io.instance->stage.streams == 1 &&
         io.stream.size() % width == 0 && io.stream.size() / width == io.rows() &&
         io.side.size() == io.stream.size() && io.weights.empty();
}

void AdaptiveScalar(std::vector<double>& output, std::span<const double> side, std::uint64_t width,
                    double eps) {
  for (std::size_t row = 0; row < output.size(); row += width) {
    double mean = 0.0;
    for (std::size_t c = 0; c < width; ++c) mean += output[row + c];
    mean /= static_cast<double>(width);
    double variance = 0.0;
    for (std::size_t c = 0; c < width; ++c) {
      const double d = output[row + c] - mean;
      variance += d * d;
    }
    const double denominator = std::sqrt(variance / static_cast<double>(width) + eps);
    for (std::size_t c = 0; c < width; ++c)
      output[row + c] = (output[row + c] - mean) / denominator * (1.0 + side[row + c]);
  }
}
std::expected<void, std::string> AdaptiveReference(RefIO& io) {
  if (!RefRows(io)) return std::unexpected("adaptive norm reference: stream or conditioning shape");
  io.carry = io.stream;
  AdaptiveScalar(io.stream, io.side, io.instance->stage.width,
                 F32(io.instance->params.Real("eps")));
  return {};
}

std::expected<void, std::string> GatedReference(RefIO& io) {
  if (!RefRows(io) || io.carry.size() != io.stream.size())
    return std::unexpected("gated residual reference: branch, gate or saved shape");
  for (std::size_t i = 0; i < io.stream.size(); ++i)
    io.stream[i] = io.carry[i] + std::tanh(io.side[i]) * io.stream[i];
  return {};
}

std::optional<std::uint64_t> Count(SpatialShape shape) { return SpatialElements(shape); }
std::expected<SpatialShape, std::string> AdaptiveSpatialShape(
    const Instance& in, SpatialShape shape, std::optional<SpatialShape> auxiliary) {
  if (!Count(shape) || shape.channels != in.stage.width || !auxiliary || *auxiliary != shape)
    return std::unexpected("adaptive spatial operands must have identical owner geometry");
  return shape;
}
ir::TensorId ChannelRows(ir::Builder& b, ir::TensorId x, SpatialShape shape) {
  return b.Reshape(b.Cont(b.Permute(x, {1, 2, 0, 3})),
                   ir::ShapeOf({shape.channels, shape.width * shape.height * shape.images}));
}
ir::TensorId FromChannelRows(ir::Builder& b, ir::TensorId x, SpatialShape shape) {
  return b.Cont(b.Permute(
      b.Reshape(x, ir::ShapeOf({shape.channels, shape.width, shape.height, shape.images})),
      {2, 0, 1, 3}));
}
std::expected<SpatialValue, std::string> AdaptiveSpatialEmit(SpatialFragment& f) {
  auto shape = AdaptiveSpatialShape(f.instance, f.shape,
                                    f.auxiliary ? std::optional(f.auxiliary->shape) : std::nullopt);
  if (!shape || !f.weights.empty() || f.carry) return std::unexpected("adaptive spatial operands");
  auto out = AdaptiveMath(f.builder, ChannelRows(f.builder, f.input, f.shape),
                          ChannelRows(f.builder, f.auxiliary->tensor, f.shape),
                          F32(f.instance.params.Real("eps")));
  return SpatialValue{FromChannelRows(f.builder, out, f.shape), f.shape};
}
std::expected<SpatialValue, std::string> GatedSpatialEmit(SpatialFragment& f) {
  auto shape = AdaptiveSpatialShape(f.instance, f.shape,
                                    f.auxiliary ? std::optional(f.auxiliary->shape) : std::nullopt);
  if (!shape || !f.weights.empty() || !f.carry || f.carry->shape != f.shape)
    return std::unexpected("gated spatial original carry or operands differ");
  return SpatialValue{GatedMath(f.builder, f.input, f.auxiliary->tensor, f.carry->tensor), f.shape};
}
std::expected<void, std::string> AdaptiveSpatialReference(SpatialRefIO& io) {
  if (!AdaptiveSpatialShape(io.instance, io.shape, io.auxiliary_shape) ||
      io.input.size() != *Count(io.shape) || io.auxiliary.size() != io.input.size() ||
      !io.kernel.empty() || !io.bias.empty() || !io.carry.empty())
    return std::unexpected("adaptive spatial reference extents differ");
  const auto rows = io.shape.width * io.shape.height * io.shape.images, width = io.shape.channels;
  std::vector<double> channels(io.input.size()), side(io.input.size());
  for (std::uint64_t n = 0; n < io.shape.images; ++n)
    for (std::uint64_t row = 0; row < io.shape.width * io.shape.height; ++row)
      for (std::uint64_t c = 0; c < width; ++c) {
        const auto dst = c + width * (row + n * io.shape.width * io.shape.height);
        const auto src = row + io.shape.width * io.shape.height * (c + width * n);
        channels[dst] = io.input[src];
        side[dst] = io.auxiliary[src];
      }
  AdaptiveScalar(channels, side, width, F32(io.instance.params.Real("eps")));
  io.output.resize(rows * width);
  for (std::uint64_t n = 0; n < io.shape.images; ++n)
    for (std::uint64_t row = 0; row < io.shape.width * io.shape.height; ++row)
      for (std::uint64_t c = 0; c < width; ++c)
        io.output[row + io.shape.width * io.shape.height * (c + width * n)] =
            channels[c + width * (row + n * io.shape.width * io.shape.height)];
  return {};
}
std::expected<void, std::string> GatedSpatialReference(SpatialRefIO& io) {
  if (!AdaptiveSpatialShape(io.instance, io.shape, io.auxiliary_shape) ||
      io.input.size() != *Count(io.shape) || io.auxiliary.size() != io.input.size() ||
      !io.carry_shape || *io.carry_shape != io.shape || io.carry.size() != io.input.size() ||
      !io.kernel.empty() || !io.bias.empty())
    return std::unexpected("gated spatial reference original carry or extents differ");
  io.output.resize(io.input.size());
  for (std::size_t i = 0; i < io.output.size(); ++i)
    io.output[i] = io.carry[i] + std::tanh(io.auxiliary[i]) * io.input[i];
  return {};
}
const SpatialOps kSpatialAdaptive{AdaptiveSpatialShape, AdaptiveSpatialEmit,
                                  AdaptiveSpatialReference, 16};
const SpatialOps kSpatialGated{.infer_shape = AdaptiveSpatialShape,
                               .emit = GatedSpatialEmit,
                               .reference = GatedSpatialReference,
                               .max_nodes = 3,
                               .needs_carry = true};

}  // namespace

const ComponentDef& NormChannel() {
  static const ComponentDef def{
      .kind = "norm.channel",
      .version = 1,
      .row_local = true,
      .params = kParams,
      .roles = kRoles,
      .shapes = Shapes,

      .spatial = &kSpatialNorm,
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

      .spatial = &kSpatialAdaptive,
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

      .spatial = &kSpatialGated,
      .emit = GatedEmit,
      .reference = GatedReference,
  };
  return def;
}

}  // namespace llmp::components::defs
