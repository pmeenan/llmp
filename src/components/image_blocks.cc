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
#include "components/spatial_stream.h"
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
    ParamSpec{.name = "round_bf16",
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
  if (in.params.Flag("round_bf16")) {
    out = builder.Convert(out, format::Encoding::kBF16);
    out = builder.Convert(out, format::Encoding::kF32);
  }
  out = builder.Scale(out, F32(std::sqrt(static_cast<double>(Channels(in)))));
  if (in.params.Flag("round_bf16")) {
    out = builder.Convert(out, format::Encoding::kBF16);
    out = builder.Convert(out, format::Encoding::kF32);
  }
  auto scale = NormScale(builder, weight);
  if (builder.tensor(scale).shape().rank != 1)
    scale = builder.Reshape(scale, ir::ShapeOf({Channels(in)}));
  out = builder.Mul(out, scale);
  if (in.params.Flag("round_bf16")) {
    out = builder.Convert(out, format::Encoding::kBF16);
    out = builder.Convert(out, format::Encoding::kF32);
  }
  if (in.params.Flag("silu")) {
    out = builder.Unary(out, ir::UnaryFn::kSilu);
    if (in.params.Flag("round_bf16")) {
      out = builder.Convert(out, format::Encoding::kBF16);
      out = builder.Convert(out, format::Encoding::kF32);
    }
  }
  return out;
}
std::expected<FragmentOut, std::string> Emit(Fragment& f) {
  if (!f.stream_needed) return FragmentOut{.stream = f.stream};
  if (f.b.tensor(f.stream).ne(0) != Channels(f.instance))
    return std::unexpected("token channel norm must retain its configured channel width");
  return FragmentOut{.stream = Normalized(f.b, f.instance, f.stream, f.Weight(0))};
}

std::expected<void, std::string> Reference(RefIO& io) {
  const auto width = Channels(*io.instance);
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
      const bool round_bf16 = io.instance->params.Flag("round_bf16");
      const auto round = [](double x) {
        float f = static_cast<float>(x);
        std::uint32_t bits = 0;
        std::memcpy(&bits, &f, sizeof(bits));
        bits += 0x7FFFU + ((bits >> 16U) & 1U);
        bits &= 0xFFFF0000U;
        std::memcpy(&f, &bits, sizeof(f));
        return static_cast<double>(f);
      };
      double value = io.stream[row + c] / denominator;
      if (round_bf16) value = round(value);
      value *= scale;
      if (round_bf16) value = round(value);
      value *= io.weights[0][c];
      if (round_bf16) value = round(value);
      if (io.instance->params.Flag("silu")) {
        value /= 1.0 + std::exp(-value);
        if (round_bf16) value = round(value);
      }
      io.stream[row + c] = value;
    }
  }
  return {};
}

// Generic scale-only modulation, x * (1 + scale). The scale is an explicit
// dynamic side input so it can be produced per request or per owner.
constexpr std::array kModulateParams{ParamSpec{.name = "side", .type = ParamType::kName}};
std::string_view Conditioning(const Instance& in);
bool JoinedRows(const Fragment& f, ir::TensorId id);
bool RefRows(const RefIO& io);
std::expected<std::vector<RoleShape>, std::string> ModulateShapes(const Instance& in) {
  if (in.site != Site::kLayer || in.stage.width == 0 || in.stage.width > kMaxStreamWidth ||
      in.stage.streams != 1 || in.params.Name("side").empty())
    return std::unexpected(
        "scale modulation requires one bounded layer stream and named side input");
  return std::vector<RoleShape>{};
}
std::expected<FragmentOut, std::string> ModulateEmit(Fragment& f) {
  if (!f.stream_needed) return FragmentOut{.stream = f.stream};
  const auto side = f.Side(Conditioning(f.instance));
  if (!JoinedRows(f, f.stream) || !JoinedRows(f, side))
    return std::unexpected("scale modulation requires matching F32 stream and conditioning rows");
  return FragmentOut{.stream = f.b.Mul(f.stream, f.b.Scale(side, 1.0F, 1.0F))};
}
std::expected<void, std::string> ModulateReference(RefIO& io) {
  if (!RefRows(io))
    return std::unexpected("scale modulation reference: stream or conditioning shape");
  for (std::size_t i = 0; i < io.stream.size(); ++i) io.stream[i] *= 1.0 + io.side[i];
  return {};
}
std::expected<SpatialShape, std::string> ModulateSpatialShape(
    const Instance& in, SpatialShape shape, std::optional<SpatialShape> auxiliary) {
  if (!SpatialElements(shape))
    return std::unexpected("spatial dimensions exceed representable storage");
  if (shape.channels != in.stage.width || !auxiliary || *auxiliary != shape)
    return std::unexpected("scale modulation requires identical stream and condition geometry");
  return shape;
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
  return t.format.encoding == format::Encoding::kF32 && t.ne(0) == f.instance.stage.width &&
         t.ne(1) == f.step.rows() && t.ne(2) == 1 && t.ne(3) == 1 &&
         t.segment == f.b.tensor(f.stream).segment;
}

bool MatchingElementwise(const Fragment& f, ir::TensorId a, ir::TensorId b) {
  if (!a.valid() || !b.valid()) return false;
  const auto ta = f.b.tensor(a);
  const auto tb = f.b.tensor(b);
  const bool spatial = f.spatial_layout || f.b.tensor(f.stream).shape().rank > 2;
  return ta.format.encoding == format::Encoding::kF32 && ta.format == tb.format &&
         ta.segment == tb.segment &&
         (spatial ? ta.ne(2) == f.instance.stage.width : JoinedRows(f, a));
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
  if (!MatchingElementwise(f, f.stream, gate) || !MatchingElementwise(f, f.stream, f.carry))
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

std::expected<SpatialShape, std::string> ChannelSpatialShape(
    const Instance& instance, SpatialShape shape, std::optional<SpatialShape> auxiliary) {
  if (!SpatialElements(shape))
    return std::unexpected("spatial dimensions exceed representable storage");
  const auto explicit_width = instance.params.Int("channels");
  const auto width =
      explicit_width == 0 ? instance.stage.width : static_cast<std::uint64_t>(explicit_width);
  if (auxiliary || shape.channels != width || !instance.def->shapes(instance))
    return std::unexpected("spatial norm input or auxiliary differs");
  return shape;
}

std::expected<SpatialShape, std::string> AdaptiveSpatialShape(
    const Instance& instance, SpatialShape shape, std::optional<SpatialShape> auxiliary) {
  if (!SpatialElements(shape))
    return std::unexpected("spatial dimensions exceed representable storage");
  if (shape.channels != instance.stage.width || !auxiliary || *auxiliary != shape)
    return std::unexpected("adaptive spatial operands must have identical owner geometry");
  return shape;
}

std::expected<SpatialShape, std::string> GatedSpatialShape(const Instance& instance,
                                                           SpatialShape shape,
                                                           std::optional<SpatialShape> auxiliary) {
  if (!SpatialElements(shape))
    return std::unexpected("spatial dimensions exceed representable storage");
  if (shape.channels != instance.stage.width || !auxiliary || *auxiliary != shape)
    return std::unexpected("adaptive spatial operands must have identical owner geometry");
  return shape;
}

const SpatialOps kNormChannelSpatial{.infer_shape = ChannelSpatialShape,
                                     .emit = EmitSpatialStream<SpatialStreamLayout::kChannelFirst>,
                                     .reference = RunSpatialStreamReference<>,
                                     .max_nodes = 18};
const SpatialOps kModulateSpatial{
    .infer_shape = ModulateSpatialShape,
    .emit = EmitSpatialStream<SpatialStreamLayout::kChannelRows, SpatialStreamLayout::kChannelRows,
                              true>,
    .reference = RunSpatialStreamReference<true, true>,
    .max_nodes = 11};
const SpatialOps kNormAdaptiveSpatial{
    .infer_shape = AdaptiveSpatialShape,
    .emit = EmitSpatialStream<SpatialStreamLayout::kChannelRows, SpatialStreamLayout::kChannelRows,
                              true>,
    .reference = RunSpatialStreamReference<true, true>,
    .max_nodes = 16};
const SpatialOps kResidualGatedSpatial{.infer_shape = GatedSpatialShape,
                                       .emit = EmitSpatialStream<>,
                                       .reference = RunSpatialStreamReference<false>,
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
      .spatial = &kNormChannelSpatial,
      .emit = Emit,
      .reference = Reference,
  };
  return def;
}

const ComponentDef& NormAdaptive() {
  static constexpr std::array kNumericalSites{
      NumericalSite{.semantic = "residual.bf16_carry",
                    .change = ir::NumericalChange::kBf16ResidualCarry,
                    .anchors = numerics::kCarryAnchors,
                    .encodings = numerics::kFloatEncodings,
                    .origins = numerics::kMultiplyAnchors},
  };

  static const ComponentDef def{
      .kind = "norm.adaptive",
      .version = 1,
      .row_local = true,
      .side_input = Conditioning,
      .params = kAdaptiveParams,
      .numerical = kNumericalSites,
      .shapes = AdaptiveShapes,
      .spatial = &kNormAdaptiveSpatial,
      .emit = AdaptiveEmit,
      .reference = AdaptiveReference,
  };
  return def;
}

const ComponentDef& ModulateScale() {
  static const ComponentDef def{.kind = "modulate.scale",
                                .version = 1,
                                .row_local = true,
                                .side_input = Conditioning,
                                .params = kModulateParams,
                                .shapes = ModulateShapes,
                                .spatial = &kModulateSpatial,
                                .emit = ModulateEmit,
                                .reference = ModulateReference};
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
      .spatial = &kResidualGatedSpatial,
      .emit = GatedEmit,
      .reference = GatedReference,
  };
  return def;
}

}  // namespace llmp::components::defs
