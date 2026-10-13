// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "components/defs.h"
#include "components/spatial.h"

namespace llmp::components::defs {
namespace {
std::optional<std::uint64_t> Count(SpatialShape s) {
  std::uint64_t count = 1;
  for (auto n : {s.width, s.height, s.channels, s.images})
    if (!n || __builtin_mul_overflow(count, n, &count) || count > (1ULL << 24U)) return {};
  return count;
}
std::expected<std::vector<RoleShape>, std::string> NoRoles(const Instance& in) {
  if (in.site != Site::kLayer) return std::unexpected("spatial blocks require a layer");
  return std::vector<RoleShape>{};
}
std::expected<FragmentOut, std::string> NoToken(Fragment&) {
  return std::unexpected("spatial blocks require typed graph inputs");
}
std::expected<void, std::string> NoTokenRef(RefIO&) {
  return std::unexpected("spatial blocks require explicit geometry");
}
std::expected<SpatialShape, std::string> AddShape(const Instance&, SpatialShape x,
                                                  std::optional<SpatialShape> auxiliary) {
  if (!Count(x) || !auxiliary || x != *auxiliary)
    return std::unexpected("spatial residual requires equal owner-local operand geometry");
  return x;
}
std::expected<SpatialValue, std::string> AddEmit(SpatialFragment& f) {
  auto shape =
      AddShape(f.instance, f.shape, f.auxiliary ? std::optional(f.auxiliary->shape) : std::nullopt);
  if (!shape || !f.weights.empty()) return std::unexpected("spatial residual operands differ");
  return SpatialValue{f.builder.Add(f.input, f.auxiliary->tensor), *shape};
}
std::expected<void, std::string> AddReference(SpatialRefIO& io) {
  auto shape = AddShape(io.instance, io.shape, io.auxiliary_shape);
  if (!shape || io.input.size() != *Count(*shape) || io.auxiliary.size() != io.input.size() ||
      !io.kernel.empty() || !io.bias.empty())
    return std::unexpected("spatial residual reference sizes differ");
  std::vector<double> out(io.input.size());
  for (std::size_t i = 0; i < out.size(); ++i) out[i] = io.input[i] + io.auxiliary[i];
  io.output = std::move(out);
  return {};
}
constexpr std::array kDupParams{
    ParamSpec{.name = "factor_t", .type = ParamType::kInt, .least = 1, .most = 2}};
std::expected<SpatialShape, std::string> DupShape(const Instance& in, SpatialShape x,
                                                  std::optional<SpatialShape> auxiliary) {
  if (!Count(x) || !auxiliary || !Count(*auxiliary) || x.images != 1 || auxiliary->images != 1 ||
      x.width != 2 * auxiliary->width || x.height != 2 * auxiliary->height ||
      (x.channels * 4 * static_cast<std::uint64_t>(in.params.Int("factor_t"))) %
          auxiliary->channels)
    return std::unexpected("duplicate-up residual requires complete last-frame source geometry");
  return x;
}
std::expected<SpatialValue, std::string> DupEmit(SpatialFragment& f) {
  auto shape =
      DupShape(f.instance, f.shape, f.auxiliary ? std::optional(f.auxiliary->shape) : std::nullopt);
  if (!shape || !f.weights.empty()) return std::unexpected("duplicate-up residual operands differ");
  const std::array inputs{f.input, f.auxiliary->tensor};
  const ir::DupUpParams params{.factor_t =
                                   static_cast<std::uint32_t>(f.instance.params.Int("factor_t"))};
  return SpatialValue{f.builder.Fused(ir::FusedKind::kDupUpAdd, inputs, params).ids[0], *shape};
}
std::expected<void, std::string> DupReference(SpatialRefIO& io) {
  auto shape = DupShape(io.instance, io.shape, io.auxiliary_shape);
  if (!shape || io.input.size() != *Count(*shape) ||
      io.auxiliary.size() != *Count(*io.auxiliary_shape) || !io.kernel.empty() || !io.bias.empty())
    return std::unexpected("duplicate-up residual reference sizes differ");
  const auto sc = *io.auxiliary_shape;
  const auto ft = static_cast<std::uint64_t>(io.instance.params.Int("factor_t"));
  const auto repeats = shape->channels * ft * 4 / sc.channels;
  std::vector<double> out(io.input.size());
  for (std::uint64_t c = 0; c < shape->channels; ++c)
    for (std::uint64_t y = 0; y < shape->height; ++y)
      for (std::uint64_t x = 0; x < shape->width; ++x) {
        const auto index = x + shape->width * (y + shape->height * c);
        const auto source_channel = (((c * ft + ft - 1) * 2 + y % 2) * 2 + x % 2) / repeats;
        out[index] =
            io.input[index] + io.auxiliary[x / 2 + sc.width * (y / 2 + sc.height * source_channel)];
      }
  io.output = std::move(out);
  return {};
}
constexpr std::array kAttentionParams{ParamSpec{.name = "heads",
                                                .type = ParamType::kInt,
                                                .required = false,
                                                .least = 1,
                                                .most = 65536,
                                                .fallback = 1}};
std::expected<SpatialShape, std::string> AttentionShape(const Instance& in, SpatialShape x,
                                                        std::optional<SpatialShape> auxiliary) {
  const auto pixels = x.width * x.height;
  const auto heads = static_cast<std::uint64_t>(in.params.Int("heads"));
  if (!Count(x) || x.images != 1 || x.channels % 3 || pixels > 4096 || (x.channels / 3) % heads ||
      heads * pixels * pixels > (1ULL << 24U) ||
      (auxiliary && *auxiliary != SpatialShape{pixels, pixels, 1, 1}))
    return std::unexpected("spatial attention QKV, heads or mask geometry differs");
  x.channels /= 3;
  return x;
}
std::expected<SpatialValue, std::string> AttentionEmit(SpatialFragment& f) {
  auto shape = AttentionShape(f.instance, f.shape,
                              f.auxiliary ? std::optional(f.auxiliary->shape) : std::nullopt);
  if (!shape || !f.weights.empty() || f.carry)
    return std::unexpected("spatial attention operands differ");
  const auto pixels = shape->width * shape->height, channels = shape->channels;
  const auto heads = static_cast<std::uint64_t>(f.instance.params.Int("heads")),
             dim = channels / heads;
  const auto src = f.builder.Reshape(f.input, ir::ShapeOf({pixels, channels * 3}));
  const std::array<std::uint64_t, 4> strides{4, pixels * 4, pixels * dim * 4,
                                             pixels * channels * 4};
  const auto plane = ir::ShapeOf({pixels, dim, heads});
  const auto q = f.builder.Cont(f.builder.Transpose(f.builder.View(src, plane, strides, 0)));
  const auto k = f.builder.Cont(
      f.builder.Transpose(f.builder.View(src, plane, strides, pixels * channels * 4)));
  const auto v = f.builder.Cont(f.builder.View(src, plane, strides, pixels * channels * 8));
  const auto scores = f.builder.MatMul(k, q);
  const ir::SoftMaxParams soft{.scale =
                                   static_cast<float>(1.0 / std::sqrt(static_cast<double>(dim)))};
  const auto probs =
      f.builder.SoftMax(scores, f.auxiliary ? f.auxiliary->tensor : ir::TensorId{}, {}, soft);
  const auto result = f.builder.MatMul(v, probs);
  return SpatialValue{f.builder.Cont(f.builder.Transpose(result),
                                     ir::ShapeOf({shape->width, shape->height, channels, 1})),
                      *shape};
}
std::expected<void, std::string> AttentionReference(SpatialRefIO& io) {
  auto shape = AttentionShape(io.instance, io.shape, io.auxiliary_shape);
  if (!shape || io.input.size() != *Count(io.shape) ||
      io.auxiliary.size() != (io.auxiliary_shape ? *Count(*io.auxiliary_shape) : 0) ||
      !io.kernel.empty() || !io.bias.empty() || !io.carry.empty())
    return std::unexpected("spatial attention reference sizes differ");
  const auto pixels = shape->width * shape->height, channels = shape->channels;
  const auto heads = static_cast<std::uint64_t>(io.instance.params.Int("heads")),
             dim = channels / heads;
  const double scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(dim)));
  std::vector<double> out(static_cast<std::size_t>(pixels * channels)),
      row(static_cast<std::size_t>(pixels));
  for (std::uint64_t head = 0; head < heads; ++head)
    for (std::uint64_t query = 0; query < pixels; ++query) {
      for (std::uint64_t key = 0; key < pixels; ++key) {
        double dot = 0;
        for (std::uint64_t channel = 0; channel < dim; ++channel)
          dot += io.input[query + pixels * (head * dim + channel)] *
                 io.input[key + pixels * (channels + head * dim + channel)];
        row[key] = dot * scale + (io.auxiliary.empty() ? 0 : io.auxiliary[key + pixels * query]);
      }
      const double max = *std::max_element(row.begin(), row.end());
      if (!std::isfinite(max))
        return std::unexpected("attention reference has no finite visible key");
      double sum = 0;
      for (auto& value : row) {
        value = std::exp(value - max);
        sum += value;
      }
      for (std::uint64_t channel = 0; channel < dim; ++channel) {
        double value = 0;
        for (std::uint64_t key = 0; key < pixels; ++key)
          value +=
              (row[key] / sum) * io.input[key + pixels * (2 * channels + head * dim + channel)];
        out[query + pixels * (head * dim + channel)] = value;
      }
    }
  io.output = std::move(out);
  return {};
}
// A pointwise source-typed projection; channels contract, WHN are rows.
constexpr std::array kProjectionParams{
    ParamSpec{.name = "in_channels", .type = ParamType::kInt, .least = 1, .most = 65536},
    ParamSpec{.name = "out_channels", .type = ParamType::kInt, .least = 1, .most = 65536},
    ParamSpec{.name = "silu",
              .type = ParamType::kBool,
              .required = false,
              .least = 0,
              .most = 1,
              .fallback = 0},
    ParamSpec{.name = "gelu",
              .type = ParamType::kBool,
              .required = false,
              .least = 0,
              .most = 1,
              .fallback = 0},
};
constexpr std::array kProjectionRoles{WeightRole{.name = "weight", .encodings = kNormEncodings}};
std::expected<std::vector<RoleShape>, std::string> ProjectionRoles(const Instance& in) {
  if (in.site != Site::kLayer) return std::unexpected("spatial projection requires a layer");
  return std::vector{Role(0, {static_cast<std::uint64_t>(in.params.Int("in_channels")),
                              static_cast<std::uint64_t>(in.params.Int("out_channels"))})};
}
std::expected<SpatialShape, std::string> ProjectionShape(const Instance& in, SpatialShape shape,
                                                         std::optional<SpatialShape> auxiliary) {
  if (auxiliary || !Count(shape) || (in.params.Flag("silu") && in.params.Flag("gelu")) ||
      shape.channels != static_cast<std::uint64_t>(in.params.Int("in_channels")))
    return std::unexpected("spatial projection input channels or auxiliary differ");
  shape.channels = static_cast<std::uint64_t>(in.params.Int("out_channels"));
  if (!Count(shape)) return std::unexpected("spatial projection output exceeds value bound");
  return shape;
}
std::expected<SpatialValue, std::string> ProjectionEmit(SpatialFragment& f) {
  auto shape = ProjectionShape(f.instance, f.shape,
                               f.auxiliary ? std::optional(f.auxiliary->shape) : std::nullopt);
  if (!shape || f.weights.size() != 1)
    return std::unexpected("spatial projection needs its validated source weight");
  auto rows = f.builder.Cont(f.builder.Permute(f.input, {1, 2, 0, 3}));
  rows = f.builder.Reshape(
      rows, ir::ShapeOf({f.shape.channels, f.shape.width * f.shape.height * f.shape.images}));
  auto output = f.builder.MatMul(f.weights[0], rows);
  if (f.instance.params.Flag("silu")) output = f.builder.Unary(output, ir::UnaryFn::kSilu);
  if (f.instance.params.Flag("gelu")) output = f.builder.Unary(output, ir::UnaryFn::kGeluTanh);
  output = f.builder.Reshape(
      output, ir::ShapeOf({shape->channels, shape->width, shape->height, shape->images}));
  return SpatialValue{f.builder.Cont(f.builder.Permute(output, {2, 0, 1, 3})), *shape};
}
std::expected<void, std::string> ProjectionReference(SpatialRefIO& io) {
  auto shape = ProjectionShape(io.instance, io.shape, io.auxiliary_shape);
  if (!shape || !io.auxiliary.empty() || !io.bias.empty() || io.input.size() != *Count(io.shape) ||
      io.kernel.size() != io.shape.channels * shape->channels)
    return std::unexpected("spatial projection reference source extents differ");
  std::vector<double> output(static_cast<std::size_t>(*Count(*shape)));
  const auto pixels = io.shape.width * io.shape.height;
  for (std::uint64_t n = 0; n < shape->images; ++n)
    for (std::uint64_t c = 0; c < shape->channels; ++c)
      for (std::uint64_t p = 0; p < pixels; ++p) {
        double value = 0;
        for (std::uint64_t k = 0; k < io.shape.channels; ++k)
          value += io.input[p + pixels * (k + io.shape.channels * n)] *
                   io.kernel[k + io.shape.channels * c];
        if (io.instance.params.Flag("silu")) value /= 1 + std::exp(-value);
        if (io.instance.params.Flag("gelu"))
          value *= 0.5 *
                   (1 + std::tanh(0.7978845608028654 * (value + 0.044715 * value * value * value)));
        output[p + pixels * (c + shape->channels * n)] = value;
      }
  io.output = std::move(output);
  return {};
}
const SpatialOps kProjection{ProjectionShape, ProjectionEmit, ProjectionReference, 8};
const SpatialOps kAdd{AddShape, AddEmit, AddReference};
const SpatialOps kDup{DupShape, DupEmit, DupReference, 24};
std::uint64_t AttentionTemporaries(const Instance& in, SpatialShape shape) {
  const auto pixels = shape.width * shape.height;
  return pixels * pixels * static_cast<std::uint64_t>(in.params.Int("heads"));
}
const SpatialOps kAttention{AttentionShape, AttentionEmit, AttentionReference, 24,
                            AttentionTemporaries};
}  // namespace

const ComponentDef& ProjectionSpatial() {
  static const ComponentDef def{.kind = "projection.spatial",
                                .params = kProjectionParams,
                                .roles = kProjectionRoles,
                                .shapes = ProjectionRoles,
                                .spatial = &kProjection,
                                .emit = NoToken,
                                .reference = NoTokenRef};
  return def;
}
const ComponentDef& ResidualSpatial() {
  static const ComponentDef def{.kind = "residual.spatial",
                                .shapes = NoRoles,
                                .spatial = &kAdd,
                                .emit = NoToken,
                                .reference = NoTokenRef};
  return def;
}
const ComponentDef& ResidualUpsample() {
  static const ComponentDef def{.kind = "residual.upsample",
                                .params = kDupParams,
                                .shapes = NoRoles,
                                .spatial = &kDup,
                                .emit = NoToken,
                                .reference = NoTokenRef};
  return def;
}
const ComponentDef& AttentionSpatial() {
  static const ComponentDef def{.kind = "attention.spatial",
                                .params = kAttentionParams,
                                .shapes = NoRoles,
                                .spatial = &kAttention,
                                .emit = NoToken,
                                .reference = NoTokenRef};
  return def;
}
}  // namespace llmp::components::defs
