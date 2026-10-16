// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

#include "components/defs.h"
#include "components/spatial.h"

namespace llmp::components::defs {
namespace {
std::optional<std::uint64_t> Count(SpatialShape s) { return SpatialElements(s); }
std::expected<std::vector<RoleShape>, std::string> NoRoles(const Instance& in) {
  if (in.site != Site::kLayer) return std::unexpected("conditioning fragments require a layer");
  return std::vector<RoleShape>{};
}
std::expected<FragmentOut, std::string> NoToken(Fragment&) {
  return std::unexpected("conditioning fragment requires typed spatial ports");
}
std::expected<void, std::string> NoTokenRef(RefIO&) {
  return std::unexpected("conditioning reference requires explicit spatial operands");
}
constexpr std::array kSelectParams{
    ParamSpec{.name = "quarter", .type = ParamType::kInt, .least = 0, .most = 3},
    ParamSpec{.name = "prefix_rows", .type = ParamType::kInt, .least = 0, .most = 4294967295.0},
    ParamSpec{.name = "quarters",
              .type = ParamType::kInt,
              .required = false,
              .least = 1,
              .most = 4,
              .fallback = 4}};
std::expected<SpatialShape, std::string> SelectShape(const Instance& in, SpatialShape x,
                                                     std::optional<SpatialShape> aux) {
  if (!Count(x) || x.width != 1 || x.images != 1 || !aux || aux->width != 1 || aux->height != 2 ||
      aux->channels != static_cast<std::uint64_t>(in.params.Int("quarters")) * x.channels ||
      aux->images != 1 || static_cast<std::uint64_t>(in.params.Int("prefix_rows")) > x.height ||
      in.params.Int("quarter") >= in.params.Int("quarters"))
    return std::unexpected("conditioning row selection geometry differs");
  return x;
}
std::expected<SpatialValue, std::string> SelectEmit(SpatialFragment& f) {
  auto out = SelectShape(f.instance, f.shape,
                         f.auxiliary ? std::optional(f.auxiliary->shape) : std::nullopt);
  if (!out || !f.weights.empty() || f.carry) return std::unexpected("conditioning row operands");
  const auto c = out->channels, rows = out->height;
  const auto prefix = static_cast<std::uint64_t>(f.instance.params.Int("prefix_rows"));
  const auto quarter = static_cast<std::uint64_t>(f.instance.params.Int("quarter"));
  const std::array<std::uint64_t, 4> stride{4, 8, 8 * c, 8 * c};
  const auto select = [&](std::uint64_t row, std::uint64_t count) {
    auto v = f.builder.View(f.auxiliary->tensor, ir::ShapeOf({1, c}), stride,
                            (2 * quarter * c + row) * 4);
    return f.builder.Repeat(f.builder.Cont(v), ir::ShapeOf({count, c}));
  };
  const auto values = prefix == 0 ? select(0, rows)
                      : prefix == rows
                          ? select(1, rows)
                          : f.builder.Concat(select(1, prefix), select(0, rows - prefix), 0);
  return SpatialValue{f.builder.Reshape(values, ir::ShapeOf({1, rows, c, 1})), *out};
}
std::expected<void, std::string> SelectReference(SpatialRefIO& io) {
  auto out = SelectShape(io.instance, io.shape, io.auxiliary_shape);
  if (!out || io.input.size() != *Count(io.shape) ||
      io.auxiliary.size() != *Count(*io.auxiliary_shape) || !io.kernel.empty() ||
      !io.bias.empty() || !io.carry.empty())
    return std::unexpected("conditioning row reference extents");
  const auto c = out->channels, rows = out->height;
  const auto prefix = static_cast<std::uint64_t>(io.instance.params.Int("prefix_rows"));
  const auto quarter = static_cast<std::uint64_t>(io.instance.params.Int("quarter"));
  io.output.resize(rows * c);
  for (std::uint64_t channel = 0; channel < c; ++channel)
    for (std::uint64_t row = 0; row < rows; ++row)
      io.output[row + rows * channel] =
          io.auxiliary[2 * (quarter * c + channel) + (row < prefix ? 1U : 0U)];
  return {};
}
constexpr std::array kConcatParams{ParamSpec{.name = "axis",
                                             .type = ParamType::kInt,
                                             .required = false,
                                             .least = 1,
                                             .most = 2,
                                             .fallback = 2}};
std::expected<SpatialShape, std::string> ConcatShape(const Instance& in, SpatialShape x,
                                                     std::optional<SpatialShape> aux) {
  if (!Count(x) || !aux || !Count(*aux) || x.width != aux->width || x.images != aux->images)
    return std::unexpected("spatial concatenation geometry differs");
  if (in.params.Int("axis") == 1) {
    if (x.channels != aux->channels || __builtin_add_overflow(x.height, aux->height, &x.height))
      return std::unexpected("row concatenation geometry differs");
  } else if (x.height != aux->height ||
             __builtin_add_overflow(x.channels, aux->channels, &x.channels)) {
    return std::unexpected("channel concatenation geometry differs");
  }
  if (!Count(x)) return std::unexpected("concatenation output exceeds the value bound");
  return x;
}
std::expected<SpatialValue, std::string> ConcatEmit(SpatialFragment& f) {
  auto out = ConcatShape(f.instance, f.shape,
                         f.auxiliary ? std::optional(f.auxiliary->shape) : std::nullopt);
  if (!out || !f.weights.empty() || f.carry)
    return std::unexpected("channel concatenation operands");
  return SpatialValue{f.builder.Concat(f.input, f.auxiliary->tensor,
                                       static_cast<std::uint8_t>(f.instance.params.Int("axis"))),
                      *out};
}
std::expected<void, std::string> ConcatReference(SpatialRefIO& io) {
  auto out = ConcatShape(io.instance, io.shape, io.auxiliary_shape);
  if (!out || io.input.size() != *Count(io.shape) ||
      io.auxiliary.size() != *Count(*io.auxiliary_shape) || !io.kernel.empty() ||
      !io.bias.empty() || !io.carry.empty())
    return std::unexpected("channel concatenation reference extents");
  io.output.resize(*Count(*out));
  const auto batches =
      io.shape.images * (io.instance.params.Int("axis") == 1 ? io.shape.channels : 1);
  const auto a = io.input.size() / batches, b = io.auxiliary.size() / batches;
  for (std::size_t n = 0; n < batches; ++n) {
    std::ranges::copy(io.input.subspan(n * a, a),
                      io.output.begin() + static_cast<std::ptrdiff_t>(n * (a + b)));
    std::ranges::copy(io.auxiliary.subspan(n * b, b),
                      io.output.begin() + static_cast<std::ptrdiff_t>(n * (a + b) + a));
  }
  return {};
}
std::expected<SpatialShape, std::string> GluShape(const Instance&, SpatialShape x,
                                                  std::optional<SpatialShape> aux) {
  if (!Count(x) || !aux || *aux != x) return std::unexpected("spatial GLU operands differ");
  return x;
}
std::expected<SpatialValue, std::string> GluEmit(SpatialFragment& f) {
  auto out =
      GluShape(f.instance, f.shape, f.auxiliary ? std::optional(f.auxiliary->shape) : std::nullopt);
  if (!out || !f.weights.empty() || f.carry) return std::unexpected("spatial GLU operands");
  return SpatialValue{f.builder.Glu(f.input, f.auxiliary->tensor, {.kind = ir::GluKind::kSwiGlu}),
                      *out};
}
std::expected<void, std::string> GluReference(SpatialRefIO& io) {
  if (!GluShape(io.instance, io.shape, io.auxiliary_shape) || io.input.size() != *Count(io.shape) ||
      io.auxiliary.size() != io.input.size() || !io.kernel.empty() || !io.bias.empty() ||
      !io.carry.empty())
    return std::unexpected("spatial GLU reference extents");
  io.output.resize(io.input.size());
  for (std::size_t i = 0; i < io.output.size(); ++i)
    io.output[i] = io.input[i] / (1 + std::exp(-io.input[i])) * io.auxiliary[i];
  return {};
}
constexpr std::array kRopeParams{
    ParamSpec{.name = "heads", .type = ParamType::kInt, .least = 1, .most = 65536},
    ParamSpec{.name = "axis0", .type = ParamType::kInt, .least = 2, .most = 512},
    ParamSpec{.name = "axis1", .type = ParamType::kInt, .least = 0, .most = 512},
    ParamSpec{.name = "axis2", .type = ParamType::kInt, .least = 0, .most = 512},
    ParamSpec{.name = "base", .type = ParamType::kReal, .least = 1, .most = 1e10},
    ParamSpec{.name = "eps", .type = ParamType::kReal, .least = 1e-30, .most = 1},
    ParamSpec{.name = "neox",
              .type = ParamType::kBool,
              .required = false,
              .least = 0,
              .most = 1,
              .fallback = 0}};
constexpr std::array kRopeRoles{WeightRole{.name = "weight", .encodings = kNormEncodings}};
std::array<std::uint64_t, 3> Axes(const Instance& in) {
  return {static_cast<std::uint64_t>(in.params.Int("axis0")),
          static_cast<std::uint64_t>(in.params.Int("axis1")),
          static_cast<std::uint64_t>(in.params.Int("axis2"))};
}
std::expected<std::vector<RoleShape>, std::string> RopeRoles(const Instance& in) {
  const auto axes = Axes(in);
  const auto heads = static_cast<std::uint64_t>(in.params.Int("heads"));
  const auto dim = axes[0] + axes[1] + axes[2];
  if (in.site != Site::kLayer || dim > 512 || dim * heads > 65536 ||
      (!in.params.Flag("neox") && (dim * heads != in.stage.width || axes[1] < 2 || axes[2] < 2)) ||
      (in.params.Flag("neox") && (axes[1] != 0 || axes[2] != 0)) ||
      std::ranges::any_of(axes, [](auto v) { return v % 2 != 0; }))
    return std::unexpected("three-axis head normalization geometry differs");
  return std::vector{Role(0, {dim})};
}
std::expected<SpatialShape, std::string> RopeShape(const Instance& in, SpatialShape x,
                                                   std::optional<SpatialShape> aux) {
  if (!RopeRoles(in) || !Count(x) || x.width != 1 || x.images != 1 ||
      x.channels != (Axes(in)[0] + Axes(in)[1] + Axes(in)[2]) *
                        static_cast<std::uint64_t>(in.params.Int("heads")) ||
      !aux || *aux != SpatialShape{x.height, in.params.Flag("neox") ? 1U : 3U, 1, 1})
    return std::unexpected("three-axis rotation input or position geometry differs");
  return x;
}
std::expected<SpatialValue, std::string> RopeEmit(SpatialFragment& f) {
  auto out = RopeShape(f.instance, f.shape,
                       f.auxiliary ? std::optional(f.auxiliary->shape) : std::nullopt);
  if (!out || f.weights.size() != 1 || f.carry)
    return std::unexpected("three-axis rotation operands");
  const auto axes = Axes(f.instance);
  const auto rows = out->height;
  const auto dim = axes[0] + axes[1] + axes[2], heads = out->channels / dim;
  auto x = f.builder.Cont(f.builder.Permute(f.input, {1, 2, 0, 3}));
  x = f.builder.Reshape(x, ir::ShapeOf({dim, heads, rows}));
  x = f.builder.Mul(f.builder.RmsNorm(x, F32(f.instance.params.Real("eps"))),
                    NormScale(f.builder, f.weights[0]));
  std::uint32_t offset = 0;
  for (std::size_t axis = 0; axis < axes.size(); ++axis) {
    if (axes[axis] == 0) continue;
    auto pos = f.builder.Cont(f.builder.View(f.auxiliary->tensor, ir::ShapeOf({rows}),
                                             {4, rows * 4, rows * 4, rows * 4}, axis * rows * 4));
    x = f.builder.Rope(
        x, pos, {},
        {.mode = f.instance.params.Flag("neox") ? ir::RopeMode::kNeox : ir::RopeMode::kNormal,
         .dims = static_cast<std::uint32_t>(axes[axis]),
         .offset = offset,
         .base = F32(f.instance.params.Real("base"))});
    offset += static_cast<std::uint32_t>(axes[axis]);
  }
  x = f.builder.Reshape(x, ir::ShapeOf({out->channels, 1, rows, 1}));
  return SpatialValue{f.builder.Cont(f.builder.Permute(x, {2, 0, 1, 3})), *out};
}
std::expected<void, std::string> RopeReference(SpatialRefIO& io) {
  auto shape = RopeShape(io.instance, io.shape, io.auxiliary_shape);
  if (!shape || io.input.size() != *Count(io.shape) ||
      io.auxiliary.size() != (io.instance.params.Flag("neox") ? 1U : 3U) * shape->height ||
      !io.bias.empty() || !io.carry.empty())
    return std::unexpected("three-axis reference extents");
  const auto axes = Axes(io.instance);
  const auto rows = shape->height;
  const auto dim = axes[0] + axes[1] + axes[2], heads = shape->channels / dim;
  if (io.kernel.size() != dim || !std::ranges::all_of(io.auxiliary, [](double v) {
        return std::isfinite(v) && v == std::trunc(v) && v >= INT32_MIN && v <= INT32_MAX;
      }))
    return std::unexpected("three-axis reference gamma or actual positions differ");
  io.output.resize(io.input.size());
  for (std::uint64_t row = 0; row < rows; ++row)
    for (std::uint64_t head = 0; head < heads; ++head) {
      double square = 0;
      for (std::uint64_t c = 0; c < dim; ++c) {
        const auto v = io.input[row + rows * (head * dim + c)];
        square += v * v;
      }
      const double norm =
          std::sqrt(square / static_cast<double>(dim) + F32(io.instance.params.Real("eps")));
      std::uint64_t offset = 0;
      for (std::size_t axis = 0; axis < axes.size(); ++axis) {
        for (std::uint64_t c = 0; c < axes[axis]; c += 2) {
          const double angle = io.auxiliary[row + rows * axis] *
                               std::pow(F32(io.instance.params.Real("base")),
                                        -static_cast<double>(c) / static_cast<double>(axes[axis]));
          const auto c0 = offset + (io.instance.params.Flag("neox") ? c / 2 : c);
          const auto c1 = c0 + (io.instance.params.Flag("neox") ? axes[axis] / 2 : 1);
          const auto i0 = row + rows * (head * dim + c0), i1 = row + rows * (head * dim + c1);
          const double a = io.input[i0] / norm * io.kernel[c0],
                       b = io.input[i1] / norm * io.kernel[c1];
          io.output[i0] = a * std::cos(angle) - b * std::sin(angle);
          io.output[i1] = a * std::sin(angle) + b * std::cos(angle);
        }
        offset += axes[axis];
      }
    }
  return {};
}
constexpr std::array kSliceParams{
    ParamSpec{.name = "start", .type = ParamType::kInt, .least = 0, .most = 4294967295.0},
    ParamSpec{.name = "rows", .type = ParamType::kInt, .least = 1, .most = 4294967295.0},
    ParamSpec{.name = "input_rows",
              .type = ParamType::kInt,
              .required = false,
              .least = 0,
              .most = 4294967295.0,
              .fallback = 0}};
std::expected<SpatialShape, std::string> SliceShape(const Instance& in, SpatialShape x,
                                                    std::optional<SpatialShape> aux) {
  const auto start = static_cast<std::uint64_t>(in.params.Int("start"));
  const auto rows = static_cast<std::uint64_t>(in.params.Int("rows"));
  if (!Count(x) || aux || start >= x.height || rows > x.height - start ||
      (in.params.Int("input_rows") != 0 &&
       static_cast<std::uint64_t>(in.params.Int("input_rows")) != x.height))
    return std::unexpected("spatial row slice lies outside its input");
  x.height = rows;
  return x;
}
std::expected<SpatialValue, std::string> SliceEmit(SpatialFragment& f) {
  auto shape = SliceShape(f.instance, f.shape,
                          f.auxiliary ? std::optional(f.auxiliary->shape) : std::nullopt);
  if (!shape || !f.weights.empty() || f.carry) return std::unexpected("spatial row slice operands");
  const auto x = f.shape;
  const auto start = static_cast<std::uint64_t>(f.instance.params.Int("start"));
  const auto view =
      f.builder.View(f.input, ir::ShapeOf({x.width, shape->height, x.channels, x.images}),
                     {4, 4 * x.width, 4 * x.width * x.height, 4 * x.width * x.height * x.channels},
                     start * x.width * 4);
  return SpatialValue{f.builder.Cont(view), *shape};
}
std::expected<void, std::string> SliceReference(SpatialRefIO& io) {
  auto shape = SliceShape(io.instance, io.shape, io.auxiliary_shape);
  if (!shape || io.input.size() != *Count(io.shape) || !io.kernel.empty() || !io.bias.empty() ||
      !io.auxiliary.empty() || !io.carry.empty())
    return std::unexpected("spatial row slice reference extents");
  const auto start = static_cast<std::uint64_t>(io.instance.params.Int("start"));
  const auto rows = shape->height, width = shape->width;
  io.output.resize(*Count(*shape));
  for (std::uint64_t outer = 0; outer < shape->channels * shape->images; ++outer)
    for (std::uint64_t v = 0; v < width * rows; ++v)
      io.output[outer * width * rows + v] = io.input[(outer * io.shape.height + start) * width + v];
  return {};
}
constexpr std::array kSpatialRmsParams{
    ParamSpec{.name = "channels", .type = ParamType::kInt, .least = 1, .most = 65536},
    ParamSpec{.name = "eps", .type = ParamType::kReal, .least = 1e-30, .most = 1},
    kNormZeroCentered};
constexpr std::array kSpatialRmsRoles{
    WeightRole{.name = "weight", .encodings = kNormEncodings, .derived = kZeroCenteredNorm}};
std::expected<std::vector<RoleShape>, std::string> SpatialRmsRoles(const Instance& in) {
  if (in.site != Site::kLayer || NormOffsetContradictsFold(in))
    return std::unexpected("spatial RMS role or source gamma convention differs");
  return std::vector{Role(0, {static_cast<std::uint64_t>(in.params.Int("channels"))})};
}
std::expected<SpatialShape, std::string> SpatialRmsShape(const Instance& in, SpatialShape x,
                                                         std::optional<SpatialShape> aux) {
  if (!Count(x) || aux || x.channels != static_cast<std::uint64_t>(in.params.Int("channels")) ||
      !SpatialRmsRoles(in))
    return std::unexpected("spatial RMS channel geometry differs");
  return x;
}
std::expected<SpatialValue, std::string> SpatialRmsEmit(SpatialFragment& f) {
  auto shape = SpatialRmsShape(f.instance, f.shape,
                               f.auxiliary ? std::optional(f.auxiliary->shape) : std::nullopt);
  if (!shape || f.weights.size() != 1 || f.carry) return std::unexpected("spatial RMS operands");
  auto x = f.builder.Cont(f.builder.Permute(f.input, {1, 2, 0, 3}));
  x = f.builder.Reshape(
      x, ir::ShapeOf({shape->channels, shape->width * shape->height * shape->images}));
  auto gamma = NormScale(f.builder, f.weights[0]);
  if (f.instance.params.Flag("norm_zero_centered")) gamma = f.builder.Scale(gamma, 1.F, 1.F);
  x = f.builder.Mul(f.builder.RmsNorm(x, F32(f.instance.params.Real("eps"))), gamma);
  x = f.builder.Reshape(x,
                        ir::ShapeOf({shape->channels, shape->width, shape->height, shape->images}));
  return SpatialValue{f.builder.Cont(f.builder.Permute(x, {2, 0, 1, 3})), *shape};
}
std::expected<void, std::string> SpatialRmsReference(SpatialRefIO& io) {
  auto shape = SpatialRmsShape(io.instance, io.shape, io.auxiliary_shape);
  if (!shape || io.input.size() != *Count(io.shape) || io.kernel.size() != shape->channels ||
      !io.bias.empty() || !io.auxiliary.empty() || !io.carry.empty())
    return std::unexpected("spatial RMS reference extents");
  const auto pixels = shape->width * shape->height, channels = shape->channels;
  io.output.resize(io.input.size());
  for (std::uint64_t n = 0; n < shape->images; ++n)
    for (std::uint64_t row = 0; row < pixels; ++row) {
      double square = 0;
      for (std::uint64_t c = 0; c < channels; ++c) {
        const auto value = io.input[row + pixels * (c + channels * n)];
        square += value * value;
      }
      const double denominator =
          std::sqrt(square / static_cast<double>(channels) + F32(io.instance.params.Real("eps")));
      for (std::uint64_t c = 0; c < channels; ++c) {
        const auto at = row + pixels * (c + channels * n);
        io.output[at] = io.input[at] / denominator *
                        (static_cast<double>(io.kernel[c]) +
                         (io.instance.params.Flag("norm_zero_centered") ? 1.0 : 0.0));
      }
    }
  return {};
}
std::expected<SpatialShape, std::string> ScaledShape(const Instance&, SpatialShape x,
                                                     std::optional<SpatialShape> aux) {
  if (!Count(x) || !aux || *aux != SpatialShape{1, 1, 1, 1})
    return std::unexpected("scaled residual requires an explicit scalar");
  return x;
}
std::expected<SpatialValue, std::string> ScaledEmit(SpatialFragment& f) {
  auto shape = ScaledShape(f.instance, f.shape,
                           f.auxiliary ? std::optional(f.auxiliary->shape) : std::nullopt);
  if (!shape || !f.weights.empty() || !f.carry || f.carry->shape != *shape)
    return std::unexpected("scaled residual requires its complete original carry");
  return SpatialValue{f.builder.Add(f.carry->tensor, f.builder.Mul(f.input, f.auxiliary->tensor)),
                      *shape};
}
std::expected<void, std::string> ScaledReference(SpatialRefIO& io) {
  auto shape = ScaledShape(io.instance, io.shape, io.auxiliary_shape);
  if (!shape || io.input.size() != *Count(*shape) || io.auxiliary.size() != 1 || !io.carry_shape ||
      *io.carry_shape != *shape || io.carry.size() != io.input.size() || !io.kernel.empty() ||
      !io.bias.empty())
    return std::unexpected("scaled residual reference operands differ");
  io.output.resize(io.input.size());
  for (std::size_t i = 0; i < io.output.size(); ++i)
    io.output[i] = io.carry[i] + io.auxiliary[0] * io.input[i];
  return {};
}
const SpatialOps kSlice{SliceShape, SliceEmit, SliceReference, 2};
const SpatialOps kSpatialRms{SpatialRmsShape, SpatialRmsEmit, SpatialRmsReference, 10};
const SpatialOps kScaled{.infer_shape = ScaledShape,
                         .emit = ScaledEmit,
                         .reference = ScaledReference,
                         .max_nodes = 2,
                         .needs_carry = true};

const SpatialOps kSelect{SelectShape, SelectEmit, SelectReference, 8};
const SpatialOps kConcat{ConcatShape, ConcatEmit, ConcatReference, 1};
const SpatialOps kGlu{GluShape, GluEmit, GluReference, 1};
const SpatialOps kRope{.infer_shape = RopeShape,
                       .emit = RopeEmit,
                       .reference = RopeReference,
                       .max_nodes = 24,
                       .auxiliary_encoding = format::Encoding::kI32};
constexpr std::array kReshapeParams{
    ParamSpec{.name = "width", .type = ParamType::kInt, .least = 1, .most = 4294967295.0},
    ParamSpec{.name = "height", .type = ParamType::kInt, .least = 1, .most = 4294967295.0}};
std::expected<SpatialShape, std::string> ReshapeShape(const Instance& in, SpatialShape x,
                                                      std::optional<SpatialShape> auxiliary) {
  const SpatialShape out{static_cast<std::uint64_t>(in.params.Int("width")),
                         static_cast<std::uint64_t>(in.params.Int("height")), x.channels, x.images};
  if (auxiliary || !Count(x) || !Count(out) || *Count(x) != *Count(out))
    return std::unexpected("spatial reshape must retain complete typed storage");
  return out;
}
std::expected<SpatialValue, std::string> ReshapeEmit(SpatialFragment& f) {
  auto out = ReshapeShape(f.instance, f.shape,
                          f.auxiliary ? std::optional(f.auxiliary->shape) : std::nullopt);
  if (!out || !f.weights.empty() || f.carry)
    return std::unexpected("spatial reshape operands differ");
  return SpatialValue{f.builder.Reshape(f.input, ir::ShapeOf({out->width, out->height,
                                                              out->channels, out->images})),
                      *out};
}
std::expected<void, std::string> ReshapeReference(SpatialRefIO& io) {
  auto out = ReshapeShape(io.instance, io.shape, io.auxiliary_shape);
  if (!out || io.input.size() != *Count(*out) || !io.kernel.empty() || !io.bias.empty() ||
      !io.auxiliary.empty() || !io.carry.empty())
    return std::unexpected("spatial reshape reference operands differ");
  io.output.assign(io.input.begin(), io.input.end());
  return {};
}
std::expected<SpatialShape, std::string> AffineShape(const Instance&, SpatialShape x,
                                                     std::optional<SpatialShape> auxiliary) {
  if (!Count(x) || auxiliary != SpatialShape{1, 2, x.channels, 1})
    return std::unexpected("channel affine requires one scale and bias per channel");
  return x;
}
std::expected<SpatialValue, std::string> AffineEmit(SpatialFragment& f) {
  auto out = AffineShape(f.instance, f.shape,
                         f.auxiliary ? std::optional(f.auxiliary->shape) : std::nullopt);
  if (!out || !f.weights.empty() || f.carry)
    return std::unexpected("channel affine operands differ");
  const auto shape = ir::ShapeOf({1, 1, out->channels, 1});
  const std::array<std::uint64_t, 4> strides{4, 4, 8, 8 * out->channels};
  auto scale = f.builder.View(f.auxiliary->tensor, shape, strides, 0);
  auto bias = f.builder.View(f.auxiliary->tensor, shape, strides, 4);
  return SpatialValue{f.builder.Add(f.builder.Mul(f.input, scale), bias), *out};
}
std::expected<void, std::string> AffineReference(SpatialRefIO& io) {
  auto out = AffineShape(io.instance, io.shape, io.auxiliary_shape);
  if (!out || io.input.size() != *Count(*out) || io.auxiliary.size() != 2 * out->channels ||
      !io.kernel.empty() || !io.bias.empty() || !io.carry.empty())
    return std::unexpected("channel affine reference operands differ");
  io.output.resize(io.input.size());
  const auto pixels = out->width * out->height;
  for (std::size_t i = 0; i < io.input.size(); ++i) {
    const auto channel = (i / pixels) % out->channels;
    io.output[i] = io.input[i] * io.auxiliary[2 * channel] + io.auxiliary[2 * channel + 1];
  }
  return {};
}
const SpatialOps kReshape{ReshapeShape, ReshapeEmit, ReshapeReference, 1};
const SpatialOps kAffine{AffineShape, AffineEmit, AffineReference, 4};

}  // namespace
const ComponentDef& ConditionRows() {
  static const ComponentDef d{.kind = "condition.rows",
                              .params = kSelectParams,
                              .shapes = NoRoles,
                              .spatial = &kSelect,
                              .emit = NoToken,
                              .reference = NoTokenRef};
  return d;
}
const ComponentDef& ConcatSpatial() {
  static const ComponentDef d{.kind = "concat.spatial",
                              .params = kConcatParams,
                              .shapes = NoRoles,
                              .spatial = &kConcat,
                              .emit = NoToken,
                              .reference = NoTokenRef};
  return d;
}
const ComponentDef& GluSpatial() {
  static const ComponentDef d{.kind = "glu.spatial",
                              .shapes = NoRoles,
                              .spatial = &kGlu,
                              .emit = NoToken,
                              .reference = NoTokenRef};
  return d;
}
const ComponentDef& NormRopeAxes() {
  static const ComponentDef d{.kind = "norm.rope_axes",
                              .params = kRopeParams,
                              .roles = kRopeRoles,
                              .shapes = RopeRoles,
                              .spatial = &kRope,
                              .emit = NoToken,
                              .reference = NoTokenRef};
  return d;
}
const ComponentDef& SliceSpatial() {
  static const ComponentDef d{.kind = "slice.spatial",
                              .params = kSliceParams,
                              .shapes = NoRoles,
                              .spatial = &kSlice,
                              .emit = NoToken,
                              .reference = NoTokenRef};
  return d;
}
const ComponentDef& NormRmsSpatial() {
  static const ComponentDef d{.kind = "norm.rms_spatial",
                              .params = kSpatialRmsParams,
                              .roles = kSpatialRmsRoles,
                              .shapes = SpatialRmsRoles,
                              .spatial = &kSpatialRms,
                              .emit = NoToken,
                              .reference = NoTokenRef};
  return d;
}
const ComponentDef& ResidualScaled() {
  static const ComponentDef d{.kind = "residual.scaled",
                              .shapes = NoRoles,
                              .spatial = &kScaled,
                              .emit = NoToken,
                              .reference = NoTokenRef};
  return d;
}
const ComponentDef& ReshapeSpatial() {
  static const ComponentDef d{.kind = "reshape.spatial",
                              .params = kReshapeParams,
                              .shapes = NoRoles,
                              .spatial = &kReshape,
                              .emit = NoToken,
                              .reference = NoTokenRef};
  return d;
}
const ComponentDef& AffineChannel() {
  static const ComponentDef d{.kind = "affine.channel",
                              .shapes = NoRoles,
                              .spatial = &kAffine,
                              .emit = NoToken,
                              .reference = NoTokenRef};
  return d;
}

}  // namespace llmp::components::defs
