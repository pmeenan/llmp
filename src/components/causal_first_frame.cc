// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <limits>
#include <utility>

#include "components/defs.h"
#include "components/spatial.h"

namespace llmp::components::defs {
namespace {

constexpr std::array kParams{
    ParamSpec{.name = "in_channels", .type = ParamType::kInt, .least = 1, .most = 65536},
    ParamSpec{.name = "out_channels", .type = ParamType::kInt, .least = 1, .most = 65536},
    ParamSpec{.name = "kernel_width", .type = ParamType::kInt, .least = 1, .most = 16},
    ParamSpec{.name = "kernel_height", .type = ParamType::kInt, .least = 1, .most = 16},
    ParamSpec{.name = "kernel_depth", .type = ParamType::kInt, .least = 1, .most = 16},
    ParamSpec{.name = "stride",
              .type = ParamType::kInt,
              .required = false,
              .least = 1,
              .most = 16,
              .fallback = 1},
    ParamSpec{.name = "padding",
              .type = ParamType::kInt,
              .required = false,
              .least = 0,
              .most = 16,
              .fallback = 0},
};
constexpr std::array kRoles{
    WeightRole{.name = "kernel", .encodings = kSpatialEncodings},
    WeightRole{.name = "bias", .encodings = kSpatialEncodings},
};

bool Product(std::initializer_list<std::uint64_t> values, std::uint64_t& out) {
  out = 1;
  for (const auto value : values)
    if (value == 0 || __builtin_mul_overflow(out, value, &out)) return false;
  return out <=
         static_cast<std::uint64_t>(std::numeric_limits<std::ptrdiff_t>::max()) / sizeof(double);
}

std::expected<std::vector<RoleShape>, std::string> Shapes(const Instance& in) {
  if (in.site != Site::kLayer) return std::unexpected("first-frame convolution requires a layer");
  const auto kw = static_cast<std::uint64_t>(in.params.Int("kernel_width"));
  const auto kh = static_cast<std::uint64_t>(in.params.Int("kernel_height"));
  const auto kt = static_cast<std::uint64_t>(in.params.Int("kernel_depth"));
  const auto ci = static_cast<std::uint64_t>(in.params.Int("in_channels"));
  const auto co = static_cast<std::uint64_t>(in.params.Int("out_channels"));
  std::uint64_t io = 0, count = 0;
  if (!Product({ci, co}, io) || !Product({kw, kh, kt, io}, count))
    return std::unexpected("first-frame convolution kernel exceeds its value bound");
  return std::vector{Role(0, {kw, kh, kt, io}), Role(1, {co})};
}

std::expected<SpatialShape, std::string> Shape(const Instance& in, SpatialShape s,
                                               std::optional<SpatialShape> auxiliary) {
  if (auxiliary || !SpatialElements(s) ||
      s.channels != static_cast<std::uint64_t>(in.params.Int("in_channels")))
    return std::unexpected("first-frame convolution input geometry differs");
  const auto kw = static_cast<std::uint64_t>(in.params.Int("kernel_width"));
  const auto kh = static_cast<std::uint64_t>(in.params.Int("kernel_height"));
  const auto pad = static_cast<std::uint64_t>(in.params.Int("padding"));
  const auto stride = static_cast<std::uint64_t>(in.params.Int("stride"));
  std::uint64_t pw = 0, ph = 0;
  if (__builtin_mul_overflow(pad, std::uint64_t{2}, &pw) ||
      __builtin_mul_overflow(pad, std::uint64_t{2}, &ph) ||
      __builtin_add_overflow(s.width, pw, &pw) || __builtin_add_overflow(s.height, ph, &ph) ||
      pw < kw || ph < kh)
    return std::unexpected("first-frame convolution spatial kernel or padding is invalid");
  SpatialShape out{(pw - kw) / stride + 1, (ph - kh) / stride + 1,
                   static_cast<std::uint64_t>(in.params.Int("out_channels")), s.images};
  if (!SpatialElements(out)) return std::unexpected("first-frame convolution output exceeds bound");
  std::uint64_t max_x = 0, max_y = 0;
  if (__builtin_mul_overflow(out.width - 1, stride, &max_x) ||
      __builtin_mul_overflow(out.height - 1, stride, &max_y) ||
      __builtin_add_overflow(max_x, kw, &max_x) || __builtin_add_overflow(max_y, kh, &max_y) ||
      max_x > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) ||
      max_y > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
    return std::unexpected("first-frame convolution coordinates overflow");
  return out;
}

std::expected<SpatialValue, std::string> Emit(SpatialFragment& f) {
  auto out =
      Shape(f.instance, f.shape, f.auxiliary ? std::optional{f.auxiliary->shape} : std::nullopt);
  if (!out || f.weights.size() != 2) return std::unexpected("first-frame convolution roles differ");
  const auto& source = f.builder.tensor(f.weights[0]);
  const auto encoding = source.format.encoding;
  if (encoding != format::Encoding::kF32 && encoding != format::Encoding::kBF16)
    return std::unexpected("first-frame convolution kernel encoding differs");
  const std::uint64_t elem = encoding == format::Encoding::kBF16 ? 2 : 4;
  const auto kw = static_cast<std::uint64_t>(f.instance.params.Int("kernel_width"));
  const auto kh = static_cast<std::uint64_t>(f.instance.params.Int("kernel_height"));
  const auto kt = static_cast<std::uint64_t>(f.instance.params.Int("kernel_depth"));
  const auto ci = static_cast<std::uint64_t>(f.instance.params.Int("in_channels"));
  const auto co = static_cast<std::uint64_t>(f.instance.params.Int("out_channels"));
  std::uint64_t kwh = 0, kwhkt = 0, offset = 0, stride_y = 0, stride_t = 0, stride_o = 0;
  if (__builtin_mul_overflow(kw, kh, &kwh) || __builtin_mul_overflow(kwh, kt, &kwhkt) ||
      __builtin_mul_overflow(kwhkt, elem, &stride_t) ||
      __builtin_mul_overflow(stride_t, ci, &stride_o) ||
      __builtin_mul_overflow(kwh, kt - 1, &offset) ||
      __builtin_mul_overflow(offset, elem, &offset) || __builtin_mul_overflow(kw, elem, &stride_y))
    return std::unexpected("first-frame convolution view strides overflow");
  const std::array<std::uint64_t, format::kMaxDims> strides{elem, stride_y, stride_t, stride_o};
  // The source is physically [Kw, Kh, Kt, In*Out]. Select its last temporal
  // plane without copying or widening any source bytes.
  const auto view = f.builder.View(f.weights[0], ir::ShapeOf({kw, kh, ci, co}), strides, offset);
  const ir::Conv2dParams params{
      .stride = static_cast<std::uint32_t>(f.instance.params.Int("stride")),
      .padding = static_cast<std::uint32_t>(f.instance.params.Int("padding")),
      .dilation = 1};
  return SpatialValue{f.builder.Conv2d(view, f.input, f.weights[1], params), *out};
}

std::expected<void, std::string> Reference(SpatialRefIO& io) {
  auto out = Shape(io.instance, io.shape, io.auxiliary_shape);
  std::uint64_t kernel_count = 0;
  const auto kw = static_cast<std::uint64_t>(io.instance.params.Int("kernel_width"));
  const auto kh = static_cast<std::uint64_t>(io.instance.params.Int("kernel_height"));
  const auto kt = static_cast<std::uint64_t>(io.instance.params.Int("kernel_depth"));
  const auto ci = io.shape.channels;
  const auto co = static_cast<std::uint64_t>(io.instance.params.Int("out_channels"));
  if (!out || !Product({kw, kh, kt, ci, co}, kernel_count) ||
      io.input.size() != *SpatialElements(io.shape) || io.kernel.size() != kernel_count ||
      io.bias.size() != co || !io.auxiliary.empty() || !io.carry.empty() || !io.weights.empty())
    return std::unexpected("first-frame convolution reference extents differ");
  std::vector<double> result(static_cast<std::size_t>(*SpatialElements(*out)));
  const auto pad = static_cast<std::int64_t>(io.instance.params.Int("padding"));
  const auto stride = static_cast<std::uint64_t>(io.instance.params.Int("stride"));
  for (std::uint64_t n = 0; n < out->images; ++n)
    for (std::uint64_t oc = 0; oc < co; ++oc)
      for (std::uint64_t y = 0; y < out->height; ++y)
        for (std::uint64_t x = 0; x < out->width; ++x) {
          double value = io.bias[oc];
          for (std::uint64_t ic = 0; ic < ci; ++ic)
            for (std::uint64_t ky = 0; ky < kh; ++ky)
              for (std::uint64_t kx = 0; kx < kw; ++kx) {
                const auto ix = static_cast<std::int64_t>(x * stride + kx) - pad;
                const auto iy = static_cast<std::int64_t>(y * stride + ky) - pad;
                if (ix < 0 || iy < 0 || static_cast<std::uint64_t>(ix) >= io.shape.width ||
                    static_cast<std::uint64_t>(iy) >= io.shape.height)
                  continue;
                // Safetensors order [Out, In, Kt, Kh, Kw], with only Kt-1
                // contributing to the first frame's empty-history result.
                const auto wi = ((oc * ci + ic) * kt + (kt - 1)) * kh * kw + ky * kw + kx;
                const auto xi = static_cast<std::uint64_t>(ix) +
                                io.shape.width * (static_cast<std::uint64_t>(iy) +
                                                  io.shape.height * (ic + ci * n));
                value += io.input[xi] * io.kernel[wi];
              }
          const auto oi = x + out->width * (y + out->height * (oc + co * n));
          result[oi] = value;
        }
  io.output = std::move(result);
  return {};
}

const SpatialOps kSpatial{Shape, Emit, Reference, 2};
}  // namespace

const ComponentDef& ConvCausalFirstFrame() {
  static const ComponentDef def{.kind = "conv.causal_first_frame",
                                .params = kParams,
                                .roles = kRoles,
                                .shapes = Shapes,
                                .spatial = &kSpatial};
  return def;
}
}  // namespace llmp::components::defs
