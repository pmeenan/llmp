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
std::optional<std::uint64_t> Count(SpatialShape s) { return SpatialElements(s); }
std::expected<std::vector<RoleShape>, std::string> NoRoles(const Instance& in) {
  if (in.site != Site::kLayer) return std::unexpected("spatial blocks require a layer");
  return std::vector<RoleShape>{};
}
std::expected<FragmentOut, std::string> AddTokenEmit(Fragment& f) {
  if (!f.stream_needed) return FragmentOut{.stream = f.stream};
  const auto other = f.carry.valid() ? f.carry : f.Side("residual");
  if (!f.stream.valid() || !other.valid() ||
      f.b.tensor(f.stream).format != f.b.tensor(other).format ||
      f.b.tensor(f.stream).segment != f.b.tensor(other).segment)
    return std::unexpected("spatial residual requires equal owner-local operand geometry");
  return FragmentOut{.stream = f.b.Add(f.stream, other)};
}
std::expected<void, std::string> AddTokenReference(RefIO& io) {
  const auto& other = !io.carry.empty() ? io.carry : io.side;
  if (io.stream.empty() || other.size() != io.stream.size() || !io.weights.empty())
    return std::unexpected("spatial residual reference sizes differ");
  for (std::size_t i = 0; i < io.stream.size(); ++i) io.stream[i] += other[i];
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
                                                .fallback = 1},
                                      ParamSpec{.name = "kv_heads",
                                                .type = ParamType::kInt,
                                                .required = false,
                                                .least = 0,
                                                .most = 65536,
                                                .fallback = 0}};
std::uint64_t KvHeads(const Instance& in) {
  const auto kv = in.params.Int("kv_heads");
  return static_cast<std::uint64_t>(kv == 0 ? in.params.Int("heads") : kv);
}
std::expected<SpatialShape, std::string> AttentionShape(const Instance& in, SpatialShape x,
                                                        std::optional<SpatialShape> auxiliary) {
  const auto pixels = x.width * x.height;
  const auto heads = static_cast<std::uint64_t>(in.params.Int("heads"));
  const auto kv = KvHeads(in);
  const auto total_heads = heads + 2 * kv;
  if (!Count(x) || x.images != 1 || heads % kv || x.channels % total_heads ||
      !SpatialElements({pixels, pixels, heads, 1}) ||
      (auxiliary && *auxiliary != SpatialShape{pixels, pixels, 1, 1}))
    return std::unexpected("spatial attention QKV, heads or mask geometry differs");
  x.channels = x.channels / total_heads * heads;
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
  const auto kv = KvHeads(f.instance), kv_channels = kv * dim;
  const auto src = f.builder.Reshape(f.input, ir::ShapeOf({pixels, channels + 2 * kv_channels}));
  const std::array<std::uint64_t, 4> strides{4, pixels * 4, pixels * dim * 4,
                                             pixels * (channels + 2 * kv_channels) * 4};
  const auto q_plane = ir::ShapeOf({pixels, dim, heads});
  const auto kv_plane = ir::ShapeOf({pixels, dim, kv});
  const auto q = f.builder.Cont(f.builder.Transpose(f.builder.View(src, q_plane, strides, 0)));
  const auto k = f.builder.Cont(
      f.builder.Transpose(f.builder.View(src, kv_plane, strides, pixels * channels * 4)));
  const auto v = f.builder.Transpose(
      f.builder.View(src, kv_plane, strides, pixels * (channels + kv_channels) * 4));
  const ir::AttnParams params{.scale =
                                  static_cast<float>(1.0 / std::sqrt(static_cast<double>(dim)))};
  // Keep original F32 inputs and masks in the shared math contract. A selected
  // implementation owns any transient operand conversion; absent one, the
  // primitive decomposition retains the ordinary products and softmax.
  const auto result =
      f.builder.Attention(q, k, v, f.auxiliary ? f.auxiliary->tensor : ir::TensorId{}, {}, params);
  const auto channels_rows = f.builder.Reshape(result, ir::ShapeOf({channels, pixels}));
  return SpatialValue{f.builder.Cont(f.builder.Transpose(channels_rows),
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
  const auto kv = KvHeads(io.instance), kv_channels = kv * dim;
  const double scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(dim)));
  std::vector<double> out(static_cast<std::size_t>(pixels * channels)),
      row(static_cast<std::size_t>(pixels));
  for (std::uint64_t head = 0; head < heads; ++head)
    for (std::uint64_t query = 0; query < pixels; ++query) {
      for (std::uint64_t key = 0; key < pixels; ++key) {
        double dot = 0;
        for (std::uint64_t channel = 0; channel < dim; ++channel)
          dot += io.input[query + pixels * (head * dim + channel)] *
                 io.input[key + pixels * (channels + (head / (heads / kv)) * dim + channel)];
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
              (row[key] / sum) * io.input[key + pixels * (channels + kv_channels +
                                                          (head / (heads / kv)) * dim + channel)];
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
  if (in.site != Site::kLayer || (in.params.Flag("silu") && in.params.Flag("gelu")))
    return std::unexpected("spatial projection requires a layer and at most one activation");
  return std::vector{Role(0, {static_cast<std::uint64_t>(in.params.Int("in_channels")),
                              static_cast<std::uint64_t>(in.params.Int("out_channels"))})};
}
std::expected<FragmentOut, std::string> ProjectionTokenEmit(Fragment& f) {
  if (!f.stream_needed) return FragmentOut{.stream = f.stream};
  const auto in_channels = static_cast<std::uint64_t>(f.instance.params.Int("in_channels"));
  if (!f.stream.valid() || (f.instance.params.Flag("silu") && f.instance.params.Flag("gelu")) ||
      f.b.tensor(f.stream).ne(0) != in_channels)
    return std::unexpected("spatial projection input channels or activation flags differ");
  auto output = f.b.MatMul(f.Weight(0), f.stream);
  if (f.instance.params.Flag("silu")) output = f.b.Unary(output, ir::UnaryFn::kSilu);
  if (f.instance.params.Flag("gelu")) output = f.b.Unary(output, ir::UnaryFn::kGeluTanh);
  return FragmentOut{.stream = output};
}
std::expected<void, std::string> ProjectionTokenReference(RefIO& io) {
  const auto& in = *io.instance;
  const auto in_c = static_cast<std::uint64_t>(in.params.Int("in_channels"));
  const auto out_c = static_cast<std::uint64_t>(in.params.Int("out_channels"));
  if ((in.params.Flag("silu") && in.params.Flag("gelu")) || io.rows() == 0 ||
      io.stream.size() != in_c * io.rows() || io.weights.size() != 1 ||
      io.weights[0].size() != in_c * out_c)
    return std::unexpected("spatial projection reference source extents differ");
  std::vector<double> output(static_cast<std::size_t>(out_c * io.rows()));
  for (std::uint64_t r = 0; r < io.rows(); ++r)
    for (std::uint64_t c = 0; c < out_c; ++c) {
      double value = 0;
      for (std::uint64_t k = 0; k < in_c; ++k)
        value += io.stream[r * in_c + k] * io.weights[0][k + in_c * c];
      if (in.params.Flag("silu")) value /= 1 + std::exp(-value);
      if (in.params.Flag("gelu"))
        value *=
            0.5 * (1 + std::tanh(0.7978845608028654 * (value + 0.044715 * value * value * value)));
      output[r * out_c + c] = value;
    }
  io.stream = std::move(output);
  return {};
}
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
                                .row_local = true,
                                .params = kProjectionParams,
                                .roles = kProjectionRoles,
                                .shapes = ProjectionRoles,
                                .emit = ProjectionTokenEmit,
                                .reference = ProjectionTokenReference};
  return def;
}
const ComponentDef& ResidualSpatial() {
  static const ComponentDef def{.kind = "residual.spatial",
                                .row_local = true,
                                .shapes = NoRoles,
                                .emit = AddTokenEmit,
                                .reference = AddTokenReference};
  return def;
}
const ComponentDef& ResidualUpsample() {
  static const ComponentDef def{
      .kind = "residual.upsample", .params = kDupParams, .shapes = NoRoles, .spatial = &kDup};
  return def;
}
const ComponentDef& AttentionSpatial() {
  static const ComponentDef def{.kind = "attention.spatial",
                                .params = kAttentionParams,
                                .shapes = NoRoles,
                                .spatial = &kAttention};
  return def;
}
}  // namespace llmp::components::defs
