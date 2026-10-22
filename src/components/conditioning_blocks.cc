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
#include "components/spatial_stream.h"

namespace llmp::components::defs {
namespace {
std::optional<std::uint64_t> Count(SpatialShape s) { return SpatialElements(s); }
std::expected<std::vector<RoleShape>, std::string> NoRoles(const Instance& in) {
  if (in.site != Site::kLayer) return std::unexpected("conditioning fragments require a layer");
  return std::vector<RoleShape>{};
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
std::expected<FragmentOut, std::string> SelectTokenEmit(Fragment& f) {
  if (!f.stream_needed) return FragmentOut{.stream = f.stream};
  const auto aux = f.carry.valid() ? f.carry : f.Side("conditioning");
  if (!f.stream.valid() || !aux.valid() ||
      f.b.tensor(f.stream).format.encoding != format::Encoding::kF32 ||
      f.b.tensor(aux).format.encoding != format::Encoding::kF32 ||
      f.b.tensor(f.stream).segment != f.b.tensor(aux).segment)
    return std::unexpected("conditioning row operands");
  const auto& x = f.b.tensor(f.stream);
  const auto& a = f.b.tensor(aux);
  const bool spatial = f.spatial_layout || f.b.tensor(f.stream).shape().rank > 2;
  const auto c = spatial ? x.ne(2) : x.ne(0);
  const auto rows = x.ne(1);
  const auto quarters = static_cast<std::uint64_t>(f.instance.params.Int("quarters"));
  const auto prefix = static_cast<std::uint64_t>(f.instance.params.Int("prefix_rows"));
  const auto quarter = static_cast<std::uint64_t>(f.instance.params.Int("quarter"));
  if ((spatial && (x.ne(0) != 1 || x.ne(3) != 1 || a.ne(0) != 1 || a.ne(1) != 2 ||
                   a.ne(2) != quarters * c || a.ne(3) != 1)) ||
      (!spatial && (x.shape().rank > 2 || a.ne(0) * a.ne(1) < 2 * quarters * c)) || prefix > rows ||
      quarter >= quarters)
    return std::unexpected("conditioning row selection geometry differs");
  const std::array<std::uint64_t, 4> stride{4, 8, 8 * c, 8 * c};
  const auto select = [&](std::uint64_t row, std::uint64_t count) {
    auto v = f.b.View(aux, ir::ShapeOf({1, c}), stride, (2 * quarter * c + row) * 4);
    return f.b.Repeat(f.b.Cont(v), ir::ShapeOf({count, c}));
  };
  const auto values = prefix == 0      ? select(0, rows)
                      : prefix == rows ? select(1, rows)
                                       : f.b.Concat(select(1, prefix), select(0, rows - prefix), 0);
  return FragmentOut{.stream = spatial ? f.b.Reshape(values, ir::ShapeOf({1, rows, c, 1}))
                                       : f.b.Cont(f.b.Transpose(values))};
}
std::expected<void, std::string> SelectTokenReference(RefIO& io) {
  const auto& in = *io.instance;
  const auto& aux = !io.carry.empty() ? io.carry : io.side;
  const auto rows = static_cast<std::uint64_t>(io.rows());
  const auto quarters = static_cast<std::uint64_t>(in.params.Int("quarters"));
  const auto prefix = static_cast<std::uint64_t>(in.params.Int("prefix_rows"));
  const auto quarter = static_cast<std::uint64_t>(in.params.Int("quarter"));
  if (rows == 0 || io.stream.empty() || io.stream.size() % rows != 0 || !io.weights.empty() ||
      prefix > rows || quarter >= quarters)
    return std::unexpected("conditioning row reference extents");
  const auto c = io.stream.size() / rows;
  if (aux.size() != 2 * quarters * c) return std::unexpected("conditioning row reference extents");
  for (std::uint64_t row = 0; row < rows; ++row)
    for (std::uint64_t channel = 0; channel < c; ++channel)
      io.stream[row * c + channel] = aux[2 * (quarter * c + channel) + (row < prefix ? 1U : 0U)];
  return {};
}
constexpr std::array kConcatParams{ParamSpec{.name = "axis",
                                             .type = ParamType::kInt,
                                             .required = false,
                                             .least = 1,
                                             .most = 2,
                                             .fallback = 2}};
std::expected<FragmentOut, std::string> ConcatTokenEmit(Fragment& f) {
  if (!f.stream_needed) return FragmentOut{.stream = f.stream};
  const auto aux = f.carry.valid() ? f.carry : f.Side("other");
  const bool spatial = f.spatial_layout || f.b.tensor(f.stream).shape().rank > 2;
  if (!f.stream.valid() || !aux.valid() ||
      f.b.tensor(f.stream).format.encoding != format::Encoding::kF32 ||
      f.b.tensor(aux).format.encoding != format::Encoding::kF32 ||
      f.b.tensor(f.stream).segment != f.b.tensor(aux).segment)
    return std::unexpected("channel concatenation operands");
  const auto& x = f.b.tensor(f.stream);
  const auto& a = f.b.tensor(aux);
  const auto axis = f.instance.params.Int("axis");
  if (spatial) {
    if (x.ne(0) != a.ne(0) || x.ne(3) != a.ne(3) || (axis == 1 && x.ne(2) != a.ne(2)) ||
        (axis == 2 && x.ne(1) != a.ne(1)))
      return std::unexpected("spatial concatenation geometry differs");
    return FragmentOut{.stream = f.b.Concat(f.stream, aux, static_cast<std::uint8_t>(axis))};
  }
  if (x.ne(2) != 1 || x.ne(3) != 1 || a.ne(2) != 1 || a.ne(3) != 1)
    return std::unexpected("token concatenation geometry differs");
  const std::uint8_t token_axis = axis == 1 ? 1U : 0U;
  if ((token_axis == 0 && x.ne(1) != a.ne(1)) || (token_axis == 1 && x.ne(0) != a.ne(0)))
    return std::unexpected("token concatenation geometry differs");
  return FragmentOut{.stream = f.b.Concat(f.stream, aux, token_axis)};
}
std::expected<void, std::string> ConcatTokenReference(RefIO& io) {
  const auto& aux = !io.carry.empty() ? io.carry : io.side;
  const auto rows = static_cast<std::size_t>(io.rows());
  if (rows == 0 || io.stream.empty() || aux.empty() || !io.weights.empty())
    return std::unexpected("channel concatenation reference extents");
  if (io.instance->params.Int("axis") == 1) {
    if (io.stream.size() % rows != 0) return std::unexpected("row concatenation reference extents");
    const auto width = io.stream.size() / rows;
    if (aux.size() % width != 0) return std::unexpected("row concatenation reference extents");
    io.stream.insert(io.stream.end(), aux.begin(), aux.end());
    return {};
  }
  if (io.stream.size() % rows != 0 || aux.size() % rows != 0)
    return std::unexpected("channel concatenation reference extents");
  const auto a = io.stream.size() / rows, b = aux.size() / rows;
  std::vector<double> out(io.stream.size() + aux.size());
  for (std::size_t r = 0; r < rows; ++r) {
    std::ranges::copy(std::span(io.stream).subspan(r * a, a),
                      out.begin() + static_cast<std::ptrdiff_t>(r * (a + b)));
    std::ranges::copy(std::span(aux).subspan(r * b, b),
                      out.begin() + static_cast<std::ptrdiff_t>(r * (a + b) + a));
  }
  io.stream = std::move(out);
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
std::expected<FragmentOut, std::string> SliceTokenEmit(Fragment& f) {
  if (!f.stream_needed) return FragmentOut{.stream = f.stream};
  if (!f.stream.valid()) return std::unexpected("spatial row slice operands");
  const bool spatial = f.spatial_layout || f.b.tensor(f.stream).shape().rank > 2;
  const auto& x = f.b.tensor(f.stream);
  const auto start = static_cast<std::uint64_t>(f.instance.params.Int("start"));
  const auto rows = static_cast<std::uint64_t>(f.instance.params.Int("rows"));
  const auto in_rows = x.ne(1);
  if (start >= in_rows || rows > in_rows - start ||
      (f.instance.params.Int("input_rows") != 0 &&
       static_cast<std::uint64_t>(f.instance.params.Int("input_rows")) != in_rows))
    return std::unexpected("spatial row slice lies outside its input");
  if (spatial) {
    const auto view =
        f.b.View(f.stream, ir::ShapeOf({x.ne(0), rows, x.ne(2), x.ne(3)}),
                 {4, 4 * x.ne(0), 4 * x.ne(0) * x.ne(1), 4 * x.ne(0) * x.ne(1) * x.ne(2)},
                 start * x.ne(0) * 4);
    return FragmentOut{.stream = f.b.Cont(view)};
  }
  const auto view =
      f.b.View(f.stream, ir::ShapeOf({x.ne(0), rows}),
               {4, 4 * x.ne(0), 4 * x.ne(0) * in_rows, 4 * x.ne(0) * in_rows}, start * x.ne(0) * 4);
  return FragmentOut{.stream = f.b.Cont(view)};
}
std::expected<void, std::string> SliceTokenReference(RefIO& io) {
  const auto& in = *io.instance;
  const auto in_rows = static_cast<std::uint64_t>(io.rows());
  const auto start = static_cast<std::uint64_t>(in.params.Int("start"));
  const auto rows = static_cast<std::uint64_t>(in.params.Int("rows"));
  if (in_rows == 0 || io.stream.empty() || io.stream.size() % in_rows != 0 || !io.weights.empty() ||
      start >= in_rows || rows > in_rows - start ||
      (in.params.Int("input_rows") != 0 &&
       static_cast<std::uint64_t>(in.params.Int("input_rows")) != in_rows))
    return std::unexpected("spatial row slice reference extents");
  const auto width = io.stream.size() / in_rows;
  std::vector<double> out(static_cast<std::size_t>(width * rows));
  std::ranges::copy(std::span(io.stream).subspan(static_cast<std::size_t>(start * width),
                                                 static_cast<std::size_t>(rows * width)),
                    out.begin());
  io.stream = std::move(out);
  return {};
}
std::expected<FragmentOut, std::string> ScaledTokenEmit(Fragment& f) {
  if (!f.stream_needed) return FragmentOut{.stream = f.stream};
  const auto scale = f.Side("scale").valid() ? f.Side("scale") : f.carry2;
  if (!f.stream.valid() || !f.carry.valid() || !scale.valid() ||
      f.b.tensor(f.stream).format != f.b.tensor(f.carry).format ||
      f.b.tensor(f.stream).segment != f.b.tensor(f.carry).segment ||
      f.b.tensor(scale).format.encoding != format::Encoding::kF32 ||
      f.b.tensor(scale).segment != f.b.tensor(f.stream).segment)
    return std::unexpected("scaled residual requires its complete original carry");
  return FragmentOut{.stream = f.b.Add(f.carry, f.b.Mul(f.stream, scale))};
}
std::expected<void, std::string> ScaledTokenReference(RefIO& io) {
  const auto& scale = !io.side.empty() ? io.side : io.carry2;
  if (io.stream.empty() || io.carry.size() != io.stream.size() ||
      (scale.size() != 1 && scale.size() != io.rows()) || !io.weights.empty())
    return std::unexpected("scaled residual reference operands differ");
  const auto width = io.stream.size() / std::max<std::size_t>(1, io.rows());
  for (std::size_t i = 0; i < io.stream.size(); ++i)
    io.stream[i] = io.carry[i] + (scale.size() == 1 ? scale[0] : scale[i / width]) * io.stream[i];
  return {};
}
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

std::expected<SpatialShape, std::string> ConditionSpatialShape(
    const Instance& instance, SpatialShape shape, std::optional<SpatialShape> auxiliary) {
  if (!SpatialElements(shape))
    return std::unexpected("spatial dimensions exceed representable storage");
  if (shape.width != 1 || shape.images != 1 || !auxiliary || auxiliary->width != 1 ||
      auxiliary->height != 2 ||
      auxiliary->channels !=
          static_cast<std::uint64_t>(instance.params.Int("quarters")) * shape.channels ||
      auxiliary->images != 1 ||
      static_cast<std::uint64_t>(instance.params.Int("prefix_rows")) > shape.height ||
      instance.params.Int("quarter") >= instance.params.Int("quarters"))
    return std::unexpected("conditioning row selection geometry differs");
  return shape;
}

std::expected<SpatialShape, std::string> ConcatSpatialShape(const Instance& instance,
                                                            SpatialShape shape,
                                                            std::optional<SpatialShape> auxiliary) {
  if (!SpatialElements(shape))
    return std::unexpected("spatial dimensions exceed representable storage");
  if (!auxiliary || !SpatialElements(*auxiliary) || shape.width != auxiliary->width ||
      shape.images != auxiliary->images)
    return std::unexpected("spatial concatenation geometry differs");
  if (instance.params.Int("axis") == 1) {
    if (shape.channels != auxiliary->channels ||
        __builtin_add_overflow(shape.height, auxiliary->height, &shape.height))
      return std::unexpected("row concatenation geometry differs");
  } else if (shape.height != auxiliary->height ||
             __builtin_add_overflow(shape.channels, auxiliary->channels, &shape.channels)) {
    return std::unexpected("channel concatenation geometry differs");
  }
  if (!SpatialElements(shape))
    return std::unexpected("concatenation output exceeds the value bound");
  return shape;
}

std::expected<SpatialShape, std::string> SliceSpatialShape(const Instance& instance,
                                                           SpatialShape shape,
                                                           std::optional<SpatialShape> auxiliary) {
  if (!SpatialElements(shape))
    return std::unexpected("spatial dimensions exceed representable storage");
  const auto start = static_cast<std::uint64_t>(instance.params.Int("start"));
  const auto rows = static_cast<std::uint64_t>(instance.params.Int("rows"));
  if (auxiliary || start >= shape.height || rows > shape.height - start ||
      (instance.params.Int("input_rows") != 0 &&
       static_cast<std::uint64_t>(instance.params.Int("input_rows")) != shape.height))
    return std::unexpected("spatial row slice lies outside its input");
  shape.height = rows;
  return shape;
}

std::expected<SpatialShape, std::string> ScaledSpatialShape(const Instance&, SpatialShape shape,
                                                            std::optional<SpatialShape> auxiliary) {
  if (!SpatialElements(shape))
    return std::unexpected("spatial dimensions exceed representable storage");
  if (!auxiliary || *auxiliary != SpatialShape{1, 1, 1, 1})
    return std::unexpected("scaled residual requires an explicit scalar");
  return shape;
}

std::expected<void, std::string> SliceSpatialReference(SpatialRefIO& io) {
  auto out = ValidateSpatialReference(io);
  if (!out) return std::unexpected(out.error());
  RefIO ref{.instance = &io.instance};

  const SpatialShape in_planes{1, io.shape.height,
                               io.shape.width * io.shape.channels * io.shape.images, 1};
  const SpatialShape out_planes{1, out->height, out->width * out->channels * out->images, 1};
  ref.segments = {{.rows = static_cast<std::uint32_t>(io.shape.height)}};
  ref.stream = ToChannelRows(io.input, in_planes);
  if (auto r = io.instance.def->reference(ref); !r) return r;
  io.output = FromChannelRows(ref.stream, out_planes);
  return {};
}
std::expected<void, std::string> ConcatSpatialReference(SpatialRefIO& io) {
  if (io.instance.params.Int("axis") != 1) return RunSpatialStreamReference<true, true>(io);
  auto out = ValidateSpatialReference(io);
  if (!out) return std::unexpected(out.error());
  RefIO ref{.instance = &io.instance};

  const SpatialShape in_planes{io.shape.width, io.shape.height, io.shape.channels * io.shape.images,
                               1};
  const SpatialShape aux_planes{io.auxiliary_shape->width, io.auxiliary_shape->height,
                                io.auxiliary_shape->channels * io.auxiliary_shape->images, 1};
  const SpatialShape out_planes{out->width, out->height, out->channels * out->images, 1};
  ref.segments = {{.rows = static_cast<std::uint32_t>(io.shape.width * io.shape.height)}};
  ref.stream = ToChannelRows(io.input, in_planes);
  ref.side = ToChannelRows(io.auxiliary, aux_planes);
  if (auto r = io.instance.def->reference(ref); !r) return r;
  io.output = FromChannelRows(ref.stream, out_planes);
  return {};
}
const SpatialOps kConditionRowsSpatial{.infer_shape = ConditionSpatialShape,
                                       .emit = EmitSpatialStream<>,
                                       .reference = RunSpatialStreamReference<>,
                                       .max_nodes = 8};
const SpatialOps kConcatSpatial{.infer_shape = ConcatSpatialShape,
                                .emit = EmitSpatialStream<>,
                                .reference = ConcatSpatialReference,
                                .max_nodes = 1};
const SpatialOps kSliceSpatial{.infer_shape = SliceSpatialShape,
                               .emit = EmitSpatialStream<>,
                               .reference = SliceSpatialReference,
                               .max_nodes = 2};
const SpatialOps kResidualScaledSpatial{.infer_shape = ScaledSpatialShape,
                                        .emit = EmitSpatialStream<>,
                                        .reference = RunSpatialStreamReference<false>,
                                        .max_nodes = 2,
                                        .needs_carry = true};

}  // namespace
const ComponentDef& ConditionRows() {
  static const ComponentDef d{.kind = "condition.rows",
                              .row_local = true,
                              .params = kSelectParams,
                              .shapes = NoRoles,
                              .spatial = &kConditionRowsSpatial,
                              .emit = SelectTokenEmit,
                              .reference = SelectTokenReference};
  return d;
}
const ComponentDef& ConcatSpatial() {
  static const ComponentDef d{.kind = "concat.spatial",
                              .params = kConcatParams,
                              .shapes = NoRoles,
                              .spatial = &kConcatSpatial,
                              .emit = ConcatTokenEmit,
                              .reference = ConcatTokenReference};
  return d;
}
const ComponentDef& SliceSpatial() {
  static const ComponentDef d{.kind = "slice.spatial",
                              .params = kSliceParams,
                              .shapes = NoRoles,
                              .spatial = &kSliceSpatial,
                              .emit = SliceTokenEmit,
                              .reference = SliceTokenReference};
  return d;
}
const ComponentDef& ResidualScaled() {
  static const ComponentDef d{.kind = "residual.scaled",
                              .row_local = true,
                              .shapes = NoRoles,
                              .spatial = &kResidualScaledSpatial,
                              .emit = ScaledTokenEmit,
                              .reference = ScaledTokenReference};
  return d;
}
const ComponentDef& ReshapeSpatial() {
  static const ComponentDef d{
      .kind = "reshape.spatial", .params = kReshapeParams, .shapes = NoRoles, .spatial = &kReshape};
  return d;
}
const ComponentDef& AffineChannel() {
  static const ComponentDef d{.kind = "affine.channel", .shapes = NoRoles, .spatial = &kAffine};
  return d;
}

}  // namespace llmp::components::defs
