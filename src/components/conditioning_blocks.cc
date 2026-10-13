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
std::optional<std::uint64_t> Count(SpatialShape s) {
  std::uint64_t n = 1;
  for (auto v : {s.width, s.height, s.channels, s.images})
    if (!v || __builtin_mul_overflow(n, v, &n) || n > (1ULL << 24U)) return {};
  return n;
}
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
    ParamSpec{.name = "prefix_rows", .type = ParamType::kInt, .least = 0, .most = 4096}};
std::expected<SpatialShape, std::string> SelectShape(const Instance& in, SpatialShape x,
                                                     std::optional<SpatialShape> aux) {
  if (!Count(x) || x.width != 1 || x.images != 1 || x.height > 4096 || !aux || aux->width != 1 ||
      aux->height != 2 || aux->channels != 4 * x.channels || aux->images != 1 ||
      static_cast<std::uint64_t>(in.params.Int("prefix_rows")) > x.height)
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
std::expected<SpatialShape, std::string> ConcatShape(const Instance&, SpatialShape x,
                                                     std::optional<SpatialShape> aux) {
  if (!Count(x) || !aux || !Count(*aux) || x.width != aux->width || x.height != aux->height ||
      x.images != aux->images || __builtin_add_overflow(x.channels, aux->channels, &x.channels) ||
      !Count(x))
    return std::unexpected("channel concatenation geometry differs");
  return x;
}
std::expected<SpatialValue, std::string> ConcatEmit(SpatialFragment& f) {
  auto out = ConcatShape(f.instance, f.shape,
                         f.auxiliary ? std::optional(f.auxiliary->shape) : std::nullopt);
  if (!out || !f.weights.empty() || f.carry)
    return std::unexpected("channel concatenation operands");
  return SpatialValue{f.builder.Concat(f.input, f.auxiliary->tensor, 2), *out};
}
std::expected<void, std::string> ConcatReference(SpatialRefIO& io) {
  auto out = ConcatShape(io.instance, io.shape, io.auxiliary_shape);
  if (!out || io.input.size() != *Count(io.shape) ||
      io.auxiliary.size() != *Count(*io.auxiliary_shape) || !io.kernel.empty() ||
      !io.bias.empty() || !io.carry.empty())
    return std::unexpected("channel concatenation reference extents");
  io.output.resize(*Count(*out));
  const auto a = io.input.size() / io.shape.images, b = io.auxiliary.size() / io.shape.images;
  for (std::size_t n = 0; n < io.shape.images; ++n) {
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
    ParamSpec{.name = "axis1", .type = ParamType::kInt, .least = 2, .most = 512},
    ParamSpec{.name = "axis2", .type = ParamType::kInt, .least = 2, .most = 512},
    ParamSpec{.name = "base", .type = ParamType::kReal, .least = 1, .most = 1e10},
    ParamSpec{.name = "eps", .type = ParamType::kReal, .least = 1e-30, .most = 1}};
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
  if (in.site != Site::kLayer || dim > 512 || dim * heads != in.stage.width ||
      std::ranges::any_of(axes, [](auto v) { return v % 2 != 0; }))
    return std::unexpected("three-axis head normalization geometry differs");
  return std::vector{Role(0, {dim})};
}
std::expected<SpatialShape, std::string> RopeShape(const Instance& in, SpatialShape x,
                                                   std::optional<SpatialShape> aux) {
  if (!RopeRoles(in) || !Count(x) || x.width != 1 || x.images != 1 ||
      x.channels != in.stage.width || !aux || *aux != SpatialShape{x.height, 3, 1, 1})
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
    auto pos = f.builder.Cont(f.builder.View(f.auxiliary->tensor, ir::ShapeOf({rows}),
                                             {4, rows * 4, rows * 4, rows * 4}, axis * rows * 4));
    x = f.builder.Rope(x, pos, {},
                       {.mode = ir::RopeMode::kNormal,
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
  if (!shape || io.input.size() != *Count(io.shape) || io.auxiliary.size() != 3 * shape->height ||
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
          const auto c0 = offset + c, c1 = c0 + 1;
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
const SpatialOps kSelect{SelectShape, SelectEmit, SelectReference, 8};
const SpatialOps kConcat{ConcatShape, ConcatEmit, ConcatReference, 1};
const SpatialOps kGlu{GluShape, GluEmit, GluReference, 1};
const SpatialOps kRope{.infer_shape = RopeShape,
                       .emit = RopeEmit,
                       .reference = RopeReference,
                       .max_nodes = 24,
                       .auxiliary_encoding = format::Encoding::kI32};
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
}  // namespace llmp::components::defs
