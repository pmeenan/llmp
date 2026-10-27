// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "components/spatial.h"

#include <algorithm>
#include <array>
#include <limits>
#include <utility>

#include "components/defs.h"
#include "components/spatial_stream.h"
#include "ir/layout.h"

namespace llmp::components {
std::optional<std::uint64_t> SpatialElements(SpatialShape shape) {
  std::uint64_t count = 1;
  // The scalar reference uses doubles and pointer differences; this also
  // bounds every dense F32 activation and all signed backend dimensions.
  constexpr auto limit =
      static_cast<std::uint64_t>(std::numeric_limits<std::ptrdiff_t>::max()) / sizeof(double);
  for (const auto dimension : {shape.width, shape.height, shape.channels, shape.images})
    if (dimension == 0 || __builtin_mul_overflow(count, dimension, &count) || count > limit)
      return {};
  return count;
}
namespace {

constexpr std::size_t kComponents = spec::kMaxComponents, kOwners = 16, kTensors = 1024;
constexpr std::size_t kParameterChars = 64;
static_assert(kComponents <= std::numeric_limits<std::uint16_t>::max());

std::expected<std::uint64_t, std::string> Values(SpatialShape s) {
  const auto count = SpatialElements(s);
  if (!count) return std::unexpected("spatial dimensions exceed representable storage");
  return *count;
}
format::TensorFormat Format(SpatialShape s, format::Encoding encoding = format::Encoding::kF32) {
  return {.encoding = encoding,
          .layout = format::Layout::kDense,
          .shape = ir::ShapeOf({s.width, s.height, s.channels, s.images})};
}
std::uint64_t Index(SpatialShape s, std::uint64_t x, std::uint64_t y, std::uint64_t c,
                    std::uint64_t n) {
  return x + s.width * (y + s.height * (c + s.channels * n));
}

// A 2-D convolution, or (temporal_kernel above 0) the first frame of a
// causal 3-D one: its source kernel is [out, in, temporal_kernel, k, k], held
// as [k, k, temporal_kernel, in*out], and an empty history leaves only its
// last temporal tap, read as a strided view without copying or widening any
// source bytes.
constexpr std::array kConvParams{
    ParamSpec{.name = "in_channels", .type = ParamType::kInt, .least = 1, .most = 65536},
    ParamSpec{.name = "out_channels", .type = ParamType::kInt, .least = 1, .most = 65536},
    ParamSpec{.name = "kernel", .type = ParamType::kInt, .least = 1, .most = 16},
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
    ParamSpec{.name = "temporal_kernel",
              .type = ParamType::kInt,
              .required = false,
              .least = 0,
              .most = 16,
              .fallback = 0},
};
constexpr std::array kConvRoles{WeightRole{.name = "kernel", .encodings = defs::kSpatialEncodings},
                                WeightRole{.name = "bias", .encodings = defs::kSpatialEncodings}};
constexpr std::array kUpParams{
    ParamSpec{.name = "factor", .type = ParamType::kInt, .least = 2, .most = 2}};

std::expected<std::vector<RoleShape>, std::string> ConvRoles(const Instance& in) {
  const auto k = static_cast<std::uint64_t>(in.params.Int("kernel"));
  if (in.site != Site::kLayer || k < 1 || k > 16)
    return std::unexpected("spatial convolution requires a layer and a kernel in [1, 16]");
  const auto ci = static_cast<std::uint64_t>(in.params.Int("in_channels"));
  const auto co = static_cast<std::uint64_t>(in.params.Int("out_channels"));
  const auto taps = static_cast<std::uint64_t>(in.params.Int("temporal_kernel"));
  if (!Values({k, k, ci, co}) || (taps != 0 && !Values({k, k, taps, ci * co})))
    return std::unexpected("spatial convolution kernel exceeds its value bound");
  if (taps != 0) return std::vector{defs::Role(0, {k, k, taps, ci * co}), defs::Role(1, {co})};
  return std::vector{defs::Role(0, {k, k, ci, co}), defs::Role(1, {co})};
}
std::expected<std::vector<RoleShape>, std::string> UpRoles(const Instance& in) {
  if (in.site != Site::kLayer) return std::unexpected("spatial upsampling requires a layer");
  return std::vector<RoleShape>{};
}
std::expected<SpatialShape, std::string> ConvShape(const Instance& in, SpatialShape s,
                                                   std::optional<SpatialShape> auxiliary = {}) {
  if (auxiliary) return std::unexpected("spatial convolution has no auxiliary input");
  if (!Values(s) || s.channels != static_cast<std::uint64_t>(in.params.Int("in_channels")))
    return std::unexpected("spatial convolution input channels or dimensions differ");
  const auto k = static_cast<std::uint64_t>(in.params.Int("kernel"));
  const auto pad = static_cast<std::uint64_t>(in.params.Int("padding"));
  const auto stride = static_cast<std::uint64_t>(in.params.Int("stride"));
  if (stride == 0) return std::unexpected("spatial convolution stride must be positive");
  std::uint64_t w = 0, h = 0;
  if (__builtin_add_overflow(s.width, 2 * pad, &w) ||
      __builtin_add_overflow(s.height, 2 * pad, &h) || w < k || h < k)
    return std::unexpected("spatial convolution padding or kernel is invalid");
  const std::uint64_t out_w = ((w - k) / stride) + 1;
  const std::uint64_t out_h = ((h - k) / stride) + 1;
  SpatialShape out{out_w, out_h, static_cast<std::uint64_t>(in.params.Int("out_channels")),
                   s.images};
  if (!Values(out)) return std::unexpected("spatial convolution output exceeds its value bound");
  return out;
}
std::expected<SpatialShape, std::string> UpShape(const Instance& in, SpatialShape s,
                                                 std::optional<SpatialShape> auxiliary = {}) {
  if (auxiliary) return std::unexpected("spatial upsampling has no auxiliary input");
  if (!Values(s)) return std::unexpected("spatial upsampling input exceeds its value bound");
  const auto f = static_cast<std::uint64_t>(in.params.Int("factor"));
  if (__builtin_mul_overflow(s.width, f, &s.width) ||
      __builtin_mul_overflow(s.height, f, &s.height) || !Values(s))
    return std::unexpected("spatial upsampling output exceeds its value bound");
  return s;
}
std::expected<SpatialValue, std::string> ConvEmit(SpatialFragment& f) {
  auto out = ConvShape(f.instance, f.shape);
  if (!out || f.weights.size() != 2)
    return std::unexpected("spatial convolution needs its two validated roles");
  // A 1x1 kernel too: its implementations (llmp.image.conv1x1) read the
  // pixel-contiguous layout directly.
  const ir::Conv2dParams params{
      .stride = static_cast<std::uint32_t>(f.instance.params.Int("stride")),
      .padding = static_cast<std::uint32_t>(f.instance.params.Int("padding")),
      .dilation = 1};
  ir::TensorId kernel = f.weights[0];
  if (const auto taps = static_cast<std::uint64_t>(f.instance.params.Int("temporal_kernel"));
      taps != 0) {
    // The source is physically [K, K, T, In*Out]: its last temporal plane.
    const auto encoding = f.builder.tensor(f.weights[0]).format.encoding;
    if (encoding != format::Encoding::kF32 && encoding != format::Encoding::kBF16)
      return std::unexpected("a temporal convolution kernel's encoding differs");
    const std::uint64_t elem = encoding == format::Encoding::kBF16 ? 2 : 4;
    const auto k = static_cast<std::uint64_t>(f.instance.params.Int("kernel"));
    const auto ci = static_cast<std::uint64_t>(f.instance.params.Int("in_channels"));
    const auto co = static_cast<std::uint64_t>(f.instance.params.Int("out_channels"));
    std::uint64_t plane = 0, frame = 0, kernel_stride = 0, offset = 0;
    if (__builtin_mul_overflow(k, k, &plane) || __builtin_mul_overflow(plane, taps, &frame) ||
        __builtin_mul_overflow(frame, elem, &frame) ||
        __builtin_mul_overflow(frame, ci, &kernel_stride) ||
        __builtin_mul_overflow(plane, taps - 1, &offset) ||
        __builtin_mul_overflow(offset, elem, &offset))
      return std::unexpected("a temporal convolution kernel's view strides overflow");
    const std::array<std::uint64_t, format::kMaxDims> strides{elem, k * elem, frame, kernel_stride};
    kernel = f.builder.View(f.weights[0], ir::ShapeOf({k, k, ci, co}), strides, offset);
  }
  return SpatialValue{f.builder.Conv2d(kernel, f.input, f.weights[1], params), *out};
}
std::expected<SpatialValue, std::string> UpEmit(SpatialFragment& f) {
  auto out = UpShape(f.instance, f.shape);
  if (!out || !f.weights.empty()) return std::unexpected("spatial upsampling shape or roles");
  return SpatialValue{
      f.builder.Upsample(f.input, static_cast<std::uint32_t>(f.instance.params.Int("factor"))),
      *out};
}
std::expected<void, std::string> ConvReference(SpatialRefIO& io) {
  auto out = ConvShape(io.instance, io.shape);
  const auto count = Values(io.shape);
  const auto k = static_cast<std::uint64_t>(io.instance.params.Int("kernel"));
  const auto co = static_cast<std::uint64_t>(io.instance.params.Int("out_channels"));
  // A temporal kernel's first frame reads its last tap (source order
  // [Out, In, T, K, K]).
  const auto taps = std::max<std::uint64_t>(
      1, static_cast<std::uint64_t>(io.instance.params.Int("temporal_kernel")));
  if (!out || !count || io.input.size() != *count ||
      io.kernel.size() != k * k * taps * io.shape.channels * co || io.bias.size() != co)
    return std::unexpected("spatial convolution reference input/weight sizes differ");
  std::vector<double> result(static_cast<std::size_t>(*Values(*out)));
  const auto pad = io.instance.params.Int("padding");
  const auto stride = static_cast<std::uint64_t>(io.instance.params.Int("stride"));
  // Independent scalar cross-correlation in WHCN order; no forced legacy casts.
  for (std::uint64_t n = 0; n < out->images; ++n)
    for (std::uint64_t c = 0; c < co; ++c)
      for (std::uint64_t y = 0; y < out->height; ++y)
        for (std::uint64_t x = 0; x < out->width; ++x) {
          double value = io.bias[c];
          for (std::uint64_t ci = 0; ci < io.shape.channels; ++ci)
            for (std::uint64_t ky = 0; ky < k; ++ky)
              for (std::uint64_t kx = 0; kx < k; ++kx) {
                const auto ix = static_cast<std::int64_t>(x * stride + kx) - pad;
                const auto iy = static_cast<std::int64_t>(y * stride + ky) - pad;
                if (ix < 0 || iy < 0 || static_cast<std::uint64_t>(ix) >= io.shape.width ||
                    static_cast<std::uint64_t>(iy) >= io.shape.height)
                  continue;
                const auto weight =
                    kx + k * (ky + k * ((taps - 1) + taps * (ci + io.shape.channels * c)));
                value += io.input[Index(io.shape, static_cast<std::uint64_t>(ix),
                                        static_cast<std::uint64_t>(iy), ci, n)] *
                         io.kernel[weight];
              }
          result[Index(*out, x, y, c, n)] = value;
        }
  io.output = std::move(result);
  return {};
}
std::expected<void, std::string> UpReference(SpatialRefIO& io) {
  auto out = UpShape(io.instance, io.shape);
  auto count = Values(io.shape);
  if (!out || !count || io.input.size() != *count || !io.kernel.empty() || !io.bias.empty())
    return std::unexpected("spatial upsampling reference input sizes differ");
  std::vector<double> result(static_cast<std::size_t>(*Values(*out)));
  const auto factor = static_cast<std::uint64_t>(io.instance.params.Int("factor"));
  for (std::uint64_t n = 0; n < out->images; ++n)
    for (std::uint64_t c = 0; c < out->channels; ++c)
      for (std::uint64_t y = 0; y < out->height; ++y)
        for (std::uint64_t x = 0; x < out->width; ++x)
          result[Index(*out, x, y, c, n)] = io.input[Index(io.shape, x / factor, y / factor, c, n)];
  io.output = std::move(result);
  return {};
}
const SpatialOps kConv{ConvShape, ConvEmit, ConvReference, 9};
const SpatialOps kUp{UpShape, UpEmit, UpReference};

bool StatelessRef(const ComponentDef& def, const spec::ComponentRef& ref) {
  return def.state == nullptr ||
         (def.spatial && def.spatial->stateless_ref && def.spatial->stateless_ref(ref));
}
format::Encoding InputSpatialEncoding(const ComponentDef& def) {
  return def.spatial->input_encoding;
}
format::Encoding AuxiliarySpatialEncoding(const ComponentDef& def) {
  return def.spatial->auxiliary_encoding;
}
bool NeedsSpatialCarry(const ComponentDef& def) { return def.spatial->needs_carry; }
}  // namespace

SpatialShape CarrySpatialShape(const ComponentDef& def, SpatialShape source_shape) {
  return def.spatial->carry_shape ? def.spatial->carry_shape(source_shape) : source_shape;
}
namespace {
ir::TensorId ChannelRows(ir::Builder& b, ir::TensorId x, SpatialShape shape) {
  return b.Reshape(b.Cont(b.Permute(x, {1, 2, 0, 3})),
                   ir::ShapeOf({shape.channels, shape.width * shape.height * shape.images}));
}
ir::TensorId FromChannelRows(ir::Builder& b, ir::TensorId x, SpatialShape shape) {
  return b.Cont(b.Permute(
      b.Reshape(x, ir::ShapeOf({shape.channels, shape.width, shape.height, shape.images})),
      {2, 0, 1, 3}));
}
}  // namespace

std::vector<double> ToChannelRows(std::span<const double> whcn, SpatialShape s) {
  std::vector<double> out(whcn.size());
  const auto pixels = s.width * s.height;
  for (std::uint64_t n = 0; n < s.images; ++n)
    for (std::uint64_t p = 0; p < pixels; ++p)
      for (std::uint64_t c = 0; c < s.channels; ++c)
        out[c + s.channels * (p + pixels * n)] = whcn[p + pixels * (c + s.channels * n)];
  return out;
}
std::vector<double> FromChannelRows(std::span<const double> rows, SpatialShape s) {
  std::vector<double> out(rows.size());
  const auto pixels = s.width * s.height;
  for (std::uint64_t n = 0; n < s.images; ++n)
    for (std::uint64_t p = 0; p < pixels; ++p)
      for (std::uint64_t c = 0; c < s.channels; ++c)
        out[p + pixels * (c + s.channels * n)] = rows[c + s.channels * (p + pixels * n)];
  return out;
}

std::expected<SpatialValue, std::string> EmitSpatialStream(SpatialFragment& f,
                                                           SpatialStreamLayout input_layout,
                                                           SpatialStreamLayout output_layout,
                                                           bool auxiliary_channel_rows) {
  auto out = InferSpatialShape(f.instance, f.shape,
                               f.auxiliary ? std::optional(f.auxiliary->shape) : std::nullopt);
  if (!out || f.weights.size() != f.instance.def->roles.size() ||
      f.carry.has_value() != NeedsSpatialCarry(*f.instance.def) ||
      (f.carry && f.carry->shape != CarrySpatialShape(*f.instance.def, f.shape)))
    return std::unexpected("spatial fragment operands do not match component contract");
  const auto rows = static_cast<std::uint32_t>(f.shape.width * f.shape.height * f.shape.images);
  StepContext ctx(f.builder, {SegmentRows{.rows = rows, .outputs = rows}}, false);
  ir::TensorId stream = f.input;
  if (input_layout == SpatialStreamLayout::kChannelRows) {
    stream = ChannelRows(f.builder, f.input, f.shape);
  } else if (input_layout == SpatialStreamLayout::kChannelFirst) {
    stream = f.builder.Cont(f.builder.Permute(f.input, {1, 2, 0, 3}));
  }
  ir::TensorId side{};
  if (f.auxiliary) {
    side = auxiliary_channel_rows ? ChannelRows(f.builder, f.auxiliary->tensor, f.auxiliary->shape)
                                  : f.auxiliary->tensor;
  }
  const ir::TensorId mask_carry = f.carry ? f.carry->tensor : ir::TensorId{};
  const ir::TensorId carry =
      f.carry ? f.carry->tensor : (f.auxiliary ? f.auxiliary->tensor : ir::TensorId{});
  Fragment frag{.b = f.builder,
                .instance = f.instance,
                .step = ctx,
                .stream = stream,
                .weight = [&](std::uint16_t role) { return f.weights[role]; },
                .spatial_layout = true,
                .carry = carry,
                .side = [side, mask_carry](
                            std::string_view name) { return name == "mask" ? mask_carry : side; }};
  auto emitted = f.instance.def->emit(frag);
  if (!emitted) return std::unexpected(emitted.error());
  if (output_layout == SpatialStreamLayout::kChannelRows) {
    return SpatialValue{FromChannelRows(f.builder, emitted->stream, *out), *out};
  }
  if (output_layout == SpatialStreamLayout::kChannelFirst) {
    return SpatialValue{f.builder.Cont(f.builder.Permute(emitted->stream, {2, 0, 1, 3})), *out};
  }
  return SpatialValue{emitted->stream, *out};
}

namespace {

// Count and bound borrowed metadata before any copy, validation allocation or
// visitor. State is refused through definition-owned preflight and instance contracts.
std::expected<std::uint64_t, std::string> MetadataBound(
    const spec::ModelSpec& spec, std::span<const std::span<const BoundTensor>> tensors,
    std::span<const SpatialOwner> owners, const SpatialBuildOptions& options) {
  if (spec.pipeline != spec::PipelineKind::kImage || spec.stages.size() != 1 ||
      tensors.size() > 16 || owners.empty() || owners.size() > kOwners)
    return std::unexpected("spatial compilation needs one image stage and 1..16 owners");
  const auto& stage = spec.stages.front();
  if (!stage.scalar_bindings.empty())
    return std::unexpected("spatial compilation requires resolved stage scalar bindings");
  if (stage.name.size() > 256 || stage.layers.empty() || stage.layers.size() > kComponents ||
      stage.embedding || !stage.side.empty() || stage.stream.mixing || stage.stream.streams != 1 ||
      stage.heads.output || stage.heads.final_norm || !stage.heads.mtp.empty() ||
      stage.heads.softcap || !stage.reads.empty() || !stage.taps.empty() || !stage.folds.empty() ||
      stage.context != 0)
    return std::unexpected("spatial compilation requires a bounded stateless component graph");
  std::size_t components = 0, source_count = 0, primitive_count = 0;
  std::uint64_t parameter_bytes = 0;
  for (const auto& layer : stage.layers) {
    if (layer.components.empty() || layer.components.size() > kComponents - components)
      return std::unexpected("spatial component count is outside its bound");
    components += layer.components.size();
    for (const auto& ref : layer.components) {
      if (ref.kind.size() > 64 || ref.params.size() > spec::kMaxParams || ref.weights.size() > 12)
        return std::unexpected("spatial component metadata exceeds its bound");
      const auto* def = Find(ref.kind, ref.version);
      const auto max_nodes = def != nullptr ? MaxSpatialNodes(*def) : 0U;
      if (def == nullptr || !StatelessRef(*def, ref) || def->roles.size() > 16 || max_nodes == 0 ||
          max_nodes > 64)
        return std::unexpected("a component has no bounded stateless spatial fragment");
      primitive_count += max_nodes;
      // Params owns every schema entry, including defaults, plus the names
      // explicitly given. The main instance set coexists with validation or
      // the weight visitor's set. For each set, three slots per entry cover
      // old and grown vector storage during reallocation; charging their
      // string payloads too covers the temporary typed/default value.
      for (const auto& param : def->params) {
        std::uint64_t name_bytes = 0, names = 0, value_bytes = kParameterChars + 1;
        if (__builtin_add_overflow(std::uint64_t{param.name.size()}, std::uint64_t{1},
                                   &name_bytes) ||
            __builtin_mul_overflow(name_bytes, std::uint64_t{2}, &names))
          return std::unexpected("spatial parameter metadata overflows");
        if (param.type == ParamType::kName && !param.choices.empty()) {
          std::uint64_t fallback_bytes = 0;
          if (__builtin_add_overflow(std::uint64_t{param.choices.front().size()}, std::uint64_t{1},
                                     &fallback_bytes))
            return std::unexpected("spatial parameter metadata overflows");
          value_bytes = std::max(value_bytes, fallback_bytes);
        }
        std::uint64_t entry = sizeof(spec::Param) + sizeof(std::string);
        if (__builtin_add_overflow(entry, names, &entry) ||
            __builtin_add_overflow(entry, value_bytes, &entry) ||
            __builtin_mul_overflow(entry, std::uint64_t{2 * 3}, &entry) ||
            __builtin_add_overflow(parameter_bytes, entry, &parameter_bytes))
          return std::unexpected("spatial parameter metadata overflows");
      }
      for (const auto& param : ref.params)
        if (param.name.size() > kParameterChars ||
            (!std::holds_alternative<std::int64_t>(param.value) &&
             !std::holds_alternative<double>(param.value) &&
             !std::holds_alternative<bool>(param.value) &&
             !(std::holds_alternative<std::string>(param.value) &&
               std::get<std::string>(param.value).size() <= kParameterChars)))
          return std::unexpected("spatial parameters are bounded scalar numbers, flags or names");
      for (const auto& weight : ref.weights)
        if (weight.role.size() > 64 || weight.tensor.size() > 256 || !weight.stage.empty())
          return std::unexpected("spatial weight binding exceeds the single-stage domain");
    }
  }
  if (!options.links.empty() && options.links.size() != components)
    return std::unexpected("spatial links must cover every component");
  for (std::size_t i = 0; i < options.links.size(); ++i)
    if (options.links[i].input > i ||
        (options.links[i].auxiliary && *options.links[i].auxiliary > i) ||
        (options.links[i].carry && *options.links[i].carry > i) ||
        (options.links[i].input_port && options.links[i].input != 0) ||
        (options.links[i].auxiliary_port && options.links[i].auxiliary) ||
        (options.links[i].carry_port && options.links[i].carry))
      return std::unexpected("spatial links must reference an earlier owner-local value");
  if (options.exports.size() > components)
    return std::unexpected("spatial exports exceed produced values");
  for (std::size_t i = 0; i < options.exports.size(); ++i) {
    if (options.exports[i] == 0 || options.exports[i] > components)
      return std::unexpected("spatial export must name a produced value");
    for (std::size_t j = 0; j < i; ++j)
      if (options.exports[i] == options.exports[j])
        return std::unexpected("spatial exports must be distinct");
  }
  for (const auto table : tensors) {
    if (table.size() > kTensors - source_count)
      return std::unexpected("spatial bound tensor inventory is too large");
    source_count += table.size();
    for (const auto& tensor : table)
      if (tensor.name.size() > 256)
        return std::unexpected("spatial bound tensor name is too large");
  }
  for (std::size_t i = 0; i < owners.size(); ++i) {
    if (!Values(owners[i].shape) ||
        (options.input_encoding != format::Encoding::kF32 &&
         options.input_encoding != format::Encoding::kI32) ||
        owners[i].owner >= ir::kJoinedSegment || (i && owners[i - 1].owner >= owners[i].owner))
      return std::unexpected("spatial owner order or initial dimensions differ");
  }
  if (options.inputs.size() > kTensors)
    return std::unexpected("spatial additional root inventory exceeds its bound");
  for (std::size_t i = 0; i < options.inputs.size(); ++i) {
    const auto& port = options.inputs[i];
    if (!Values(port.shape) ||
        (port.encoding != format::Encoding::kF32 && port.encoding != format::Encoding::kI32) ||
        std::ranges::find(owners, port.owner, &SpatialOwner::owner) == owners.end() ||
        (i && options.inputs[i - 1].owner > port.owner) ||
        port.slot !=
            (i && options.inputs[i - 1].owner == port.owner ? options.inputs[i - 1].slot + 1U : 0U))
      return std::unexpected("spatial additional input owner, slot, encoding or shape differs");
  }
  for (const auto& owner : owners) {
    const auto count = std::ranges::count(options.inputs, owner.owner, &SpatialInput::owner);
    for (const auto& link : options.links)
      if ((link.input_port && *link.input_port >= count) ||
          (link.auxiliary_port && *link.auxiliary_port >= count) ||
          (link.carry_port && *link.carry_port >= count))
        return std::unexpected("spatial edge references an absent owner-local input port");
  }
  // Bounded roles/component, one input/owner and the declared primitive limit/component/owner.
  // 4KiB per node/tensor covers graph+clone+rewrite/placement metadata, and
  // 4KiB per source/component covers bounded names and visitor copies;
  // normalized parameter ownership is charged from the schema separately.
  static_assert(sizeof(ir::Node) <= 256 && sizeof(ir::TensorDesc) <= 256);
  const auto nodes = owners.size() * (primitive_count + 1) + options.inputs.size() + 2 * components;
  // Explicit outputs add owned port/shape lists and graph output retention.
  // Layout sinking makes the graph again: at most a node's own copy, a
  // view, a permute and a Cont at each of its places.
  std::uint64_t metadata =
      4096ULL * ((options.sink_layouts ? 6 : 2) * nodes + source_count + 4 * components + 16 +
                 owners.size() * options.exports.size() + 2 * options.inputs.size());
  if (__builtin_add_overflow(metadata, parameter_bytes, &metadata))
    return std::unexpected("spatial compiler metadata overflows");
  return metadata;
}
struct PaymentOwner {
  SpatialPayment calls;
  std::uint64_t bytes = 0;
  PaymentOwner(SpatialPayment value, std::uint64_t paid) : calls(std::move(value)), bytes(paid) {}
  PaymentOwner(const PaymentOwner&) = delete;
  PaymentOwner& operator=(const PaymentOwner&) = delete;
  ~PaymentOwner() { calls.release(bytes); }
  bool Add(std::uint64_t n) {
    if (n > UINT64_MAX - bytes || !calls.charge(n)) return false;
    bytes += n;
    return true;
  }
};
struct RoleUse {
  std::uint32_t component = 0;
  std::uint16_t role = 0;
  ir::WeightRef ref;
  format::TensorFormat format;
  std::vector<std::pair<std::string, format::PartRole>> parts;
  ir::TensorId tensor;
};

}  // namespace

namespace defs {
const ComponentDef& Conv2d() {
  static const ComponentDef def{.kind = "conv.spatial",
                                .params = kConvParams,
                                .roles = kConvRoles,
                                .shapes = ConvRoles,
                                .spatial = &kConv};
  return def;
}
const ComponentDef& UpsampleNearest() {
  static const ComponentDef def{
      .kind = "upsample.nearest", .params = kUpParams, .shapes = UpRoles, .spatial = &kUp};
  return def;
}
}  // namespace defs

std::uint16_t MaxSpatialNodes(const ComponentDef& def) {
  return def.spatial && def.spatial->infer_shape && def.spatial->emit && def.spatial->reference
             ? def.spatial->max_nodes
             : 0;
}

std::expected<SpatialShape, std::string> InferSpatialShape(const Instance& instance,
                                                           SpatialShape shape,
                                                           std::optional<SpatialShape> auxiliary) {
  if (!instance.def || MaxSpatialNodes(*instance.def) == 0)
    return std::unexpected("component has no spatial shape rule");
  return instance.def->spatial->infer(instance, shape, auxiliary);
}

std::expected<SpatialOperand, std::string> InferSpatialOutput(
    const Instance& instance, SpatialOperand input, std::optional<SpatialOperand> auxiliary,
    std::optional<SpatialOperand> carry) {
  if (!instance.def || MaxSpatialNodes(*instance.def) == 0)
    return std::unexpected("component has no spatial shape rule");
  if (!Values(input.shape) || input.encoding != InputSpatialEncoding(*instance.def))
    return std::unexpected("spatial input shape or encoding differs from component contract");
  if (auxiliary &&
      (!Values(auxiliary->shape) || auxiliary->encoding != AuxiliarySpatialEncoding(*instance.def)))
    return std::unexpected("spatial auxiliary shape or encoding differs from component contract");
  if (carry.has_value() != NeedsSpatialCarry(*instance.def) ||
      (carry && (!Values(carry->shape) || carry->encoding != format::Encoding::kF32 ||
                 carry->shape != CarrySpatialShape(*instance.def, input.shape))))
    return std::unexpected("spatial carry shape or encoding differs from component contract");
  auto output = InferSpatialShape(instance, input.shape,
                                  auxiliary ? std::optional(auxiliary->shape) : std::nullopt);
  if (!output) return std::unexpected(output.error());
  if (!Values(*output)) return std::unexpected("spatial output exceeds representable storage");
  return SpatialOperand{.shape = *output, .encoding = format::Encoding::kF32};
}

std::expected<SpatialValue, std::string> EmitSpatialFragment(SpatialFragment& fragment) {
  if (!fragment.instance.def || MaxSpatialNodes(*fragment.instance.def) == 0)
    return std::unexpected("component has no spatial fragment");
  return fragment.instance.def->spatial->emit(fragment);
}

std::expected<void, std::string> RunSpatialReference(SpatialRefIO& io) {
  if (!io.instance.def || MaxSpatialNodes(*io.instance.def) == 0)
    return std::unexpected("component has no spatial reference");
  return io.instance.def->spatial->reference(io);
}

std::expected<SpatialShape, std::string> ValidateSpatialReference(SpatialRefIO& io) {
  auto out = InferSpatialShape(io.instance, io.shape, io.auxiliary_shape);
  if (!out || io.input.size() != *Values(io.shape) || !io.bias.empty() ||
      (!io.weights.empty() && !io.kernel.empty()) ||
      io.auxiliary.size() != (io.auxiliary_shape ? *Values(*io.auxiliary_shape) : 0U) ||
      io.carry_shape.has_value() != NeedsSpatialCarry(*io.instance.def) ||
      (io.carry_shape && (*io.carry_shape != CarrySpatialShape(*io.instance.def, io.shape) ||
                          io.carry.size() != *Values(*io.carry_shape))) ||
      (!io.carry_shape && !io.carry.empty()))
    return std::unexpected("spatial reference extents differ");
  return out;
}

std::expected<RefIO, std::string> PrepareSpatialReference(SpatialRefIO& io) {
  auto validated = ValidateSpatialReference(io);
  if (!validated) return std::unexpected(validated.error());
  RefIO ref{.instance = &io.instance};
  const auto rows = io.shape.width * io.shape.height * io.shape.images;
  ref.segments = {{.rows = static_cast<std::uint32_t>(rows)}};
  if (!io.weights.empty()) {
    if (io.instance.ref != nullptr) {
      if (io.weights.size() != io.instance.ref->weights.size())
        return std::unexpected("spatial reference weight roles differ");
      ref.weights.resize(io.instance.def->roles.size());
      for (std::size_t i = 0; i < io.weights.size(); ++i) {
        const auto& role_name = io.instance.ref->weights[i].role;
        const auto it = std::ranges::find(io.instance.def->roles, role_name, &WeightRole::name);
        if (it == io.instance.def->roles.end())
          return std::unexpected("spatial reference weight role is unknown");
        const auto idx = static_cast<std::size_t>(it - io.instance.def->roles.begin());
        ref.weights[idx].assign(io.weights[i].begin(), io.weights[i].end());
      }
    } else {
      if (io.weights.size() != io.instance.def->roles.size())
        return std::unexpected("spatial reference weight roles differ");
      ref.weights.reserve(io.weights.size());
      for (const auto w : io.weights) ref.weights.emplace_back(w.begin(), w.end());
    }
  } else if (!io.instance.def->roles.empty()) {
    ref.weights = {{io.kernel.begin(), io.kernel.end()}};
  }
  return ref;
}

std::expected<void, std::string> RunSpatialStreamReference(SpatialRefIO& io, bool channel_rows,
                                                           bool auxiliary_channel_rows) {
  auto prepared = PrepareSpatialReference(io);
  if (!prepared) return std::unexpected(prepared.error());
  auto& ref = *prepared;
  const auto out = InferSpatialShape(io.instance, io.shape, io.auxiliary_shape);
  ref.stream = channel_rows ? ToChannelRows(io.input, io.shape)
                            : std::vector<double>(io.input.begin(), io.input.end());
  ref.carry.assign(io.carry.begin(), io.carry.end());
  if (io.auxiliary_shape)
    ref.side = auxiliary_channel_rows
                   ? ToChannelRows(io.auxiliary, *io.auxiliary_shape)
                   : std::vector<double>(io.auxiliary.begin(), io.auxiliary.end());
  if (auto result = io.instance.def->reference(ref); !result) return result;
  io.output = channel_rows ? FromChannelRows(ref.stream, *out) : std::move(ref.stream);
  return {};
}

std::expected<SpatialGraph, std::string> BuildSpatialStep(
    const spec::ModelSpec& spec, std::span<const std::span<const BoundTensor>> tensors,
    std::span<const SpatialOwner> owners, SpatialPayment payment, SpatialBuildOptions options) {
  if (!spec.stages.empty()) {
    if (options.links.empty()) options.links = spec.stages.front().links;
    if (options.exports.empty()) options.exports = spec.stages.front().exports;
  }
  auto metadata = MetadataBound(spec, tensors, owners, options);
  if (!metadata) return std::unexpected(metadata.error());
  if (!payment.charge || !payment.release || !payment.charge(*metadata))
    return std::unexpected("spatial compiler metadata was not funded");
  auto hold = std::make_shared<PaymentOwner>(std::move(payment), *metadata);
  auto instances = InstancesOf(spec, 0);
  if (!instances) return std::unexpected(instances.error());
  std::vector<const Instance*> chain;
  for (const auto& layer : instances->layers)
    for (const auto& instance : layer) {
      if (MaxSpatialNodes(*instance.def) == 0 ||
          (instance.def->state && !instance.def->state(instance).tensors.empty()))
        return std::unexpected("a component has no stateless spatial fragment");
      chain.push_back(&instance);
    }
  // Whole-graph inference before builder creation; every edge is owner-local.
  std::uint64_t values = 0;
  std::vector<SpatialOwner> ends;
  for (const auto& owner : owners) {
    auto shape = owner.shape;
    if (__builtin_add_overflow(values, *Values(shape), &values))
      return std::unexpected("spatial reference inventory overflows");
    std::vector<SpatialShape> inferred{shape};
    std::vector<const SpatialInput*> ports;
    for (const auto& port : options.inputs)
      if (port.owner == owner.owner) {
        ports.push_back(&port);
        if (__builtin_add_overflow(values, *Values(port.shape), &values))
          return std::unexpected("spatial reference input inventory overflows");
      }
    for (std::size_t j = 0; j < chain.size(); ++j) {
      const auto link =
          options.links.empty() ? SpatialLink{static_cast<std::uint16_t>(j), {}} : options.links[j];
      const auto* instance = chain[j];
      const auto& def = *instance->def;
      const auto source_shape =
          link.input_port ? ports[*link.input_port]->shape : inferred[link.input];
      const auto source_encoding =
          link.input_port ? ports[*link.input_port]->encoding
                          : (link.input == 0 ? options.input_encoding : format::Encoding::kF32);
      const auto auxiliary =
          link.auxiliary_port
              ? std::optional(ports[*link.auxiliary_port]->shape)
              : (link.auxiliary ? std::optional(inferred[*link.auxiliary]) : std::nullopt);
      const auto auxiliary_encoding =
          link.auxiliary_port ? ports[*link.auxiliary_port]->encoding
                              : (link.auxiliary && *link.auxiliary == 0 ? options.input_encoding
                                                                        : format::Encoding::kF32);
      const auto carry = link.carry_port
                             ? std::optional(ports[*link.carry_port]->shape)
                             : (link.carry ? std::optional(inferred[*link.carry]) : std::nullopt);
      const auto carry_encoding =
          link.carry_port
              ? ports[*link.carry_port]->encoding
              : (link.carry && *link.carry == 0 ? options.input_encoding : format::Encoding::kF32);
      auto typed = InferSpatialOutput(
          *instance, {.shape = source_shape, .encoding = source_encoding},
          auxiliary
              ? std::optional(SpatialOperand{.shape = *auxiliary, .encoding = auxiliary_encoding})
              : std::nullopt,
          carry ? std::optional(SpatialOperand{.shape = *carry, .encoding = carry_encoding})
                : std::nullopt);
      if (!typed) return std::unexpected(typed.error());
      shape = typed->shape;
      inferred.push_back(shape);
      if (__builtin_add_overflow(values, *Values(shape), &values))
        return std::unexpected("spatial reference inventory overflows");
      const std::uint64_t temp = def.spatial->temporary_values
                                     ? def.spatial->temporary_values(*instance, source_shape)
                                     : 0ULL;
      if (__builtin_add_overflow(values, temp, &values))
        return std::unexpected("spatial reference temporary inventory overflows");
    }
    if (options.exports.empty())
      ends.push_back({owner.owner, shape});
    else
      for (const auto value : options.exports) ends.push_back({owner.owner, inferred[value]});
  }
  std::vector<RoleUse> roles;
  auto visited = VisitWeightBindings(
      spec, tensors, [&](const WeightBindingView& role) -> std::expected<void, std::string> {
        const auto parts = format::PartsOf(role.format);
        if (role.use_stage != 0 || role.rows_per_token || role.head_rows || !parts ||
            parts->size() != 1 || role.parts.size() != 1 ||
            (*parts)[0].role != format::PartRole::kData ||
            role.parts[0].second != format::PartRole::kData)
          return std::unexpected("spatial roles need one complete encoded source data part");
        std::uint64_t count = 1;
        for (auto n : role.format.shape.ne)
          if (__builtin_mul_overflow(count, n, &count))
            return std::unexpected("spatial source weight value count overflows");
        // Resident source descriptors do not allocate decoded payloads. Keep the
        // activation cap separate; reference copies require checked payment below.
        if (__builtin_add_overflow(values, count, &values))
          return std::unexpected("spatial source/reference inventory overflows");
        auto ref = role.ref;
        ref.stage = options.stage;
        roles.push_back({role.use_component,
                         role.use_role,
                         ref,
                         role.format,
                         {role.parts.begin(), role.parts.end()},
                         {}});
        return {};
      });
  if (!visited) return std::unexpected(visited.error());
  // Bound encoded roots, complete reference copies and decoded double work
  // across primitive execution and the independent chain oracle (not GPU VMM).
  std::uint64_t reference_bytes = 0;
  if (options.reference_buffers &&
      (__builtin_mul_overflow(values, std::uint64_t{64}, &reference_bytes) ||
       !hold->Add(reference_bytes)))
    return std::unexpected("spatial graph/reference buffers were not funded");
  std::size_t primitive_count = 0;
  for (const auto* instance : chain) primitive_count += MaxSpatialNodes(*instance->def);
  const auto max_nodes =
      owners.size() * (primitive_count + 1) + options.inputs.size() + roles.size();
  ir::Builder builder(max_nodes);
  std::vector<StepWeight> weights;
  std::vector<std::uint64_t> parameters{0x5350415449414cULL, 2, owners.size(), chain.size()};
  for (std::size_t j = 0; j < chain.size(); ++j) {
    const auto link =
        options.links.empty() ? SpatialLink{static_cast<std::uint16_t>(j), {}} : options.links[j];
    parameters.push_back(link.input);
    parameters.push_back(link.auxiliary ? *link.auxiliary + 1U : 0U);
  }
  if (!options.exports.empty()) {
    parameters[1] = 3;
    parameters.push_back(options.exports.size());
    for (const auto value : options.exports) parameters.push_back(value);
  }
  const bool extended =
      !options.inputs.empty() || std::ranges::any_of(options.links, [](const auto& link) {
        return link.carry.has_value() || link.carry_port.has_value();
      });
  if (extended) {
    parameters[1] = 4;
    parameters.push_back(options.inputs.size());
    for (const auto& port : options.inputs) {
      parameters.push_back(port.owner);
      parameters.push_back(port.slot);
      parameters.push_back(static_cast<std::uint64_t>(port.encoding));
      for (auto n : {port.shape.width, port.shape.height, port.shape.channels, port.shape.images})
        parameters.push_back(n);
    }
    for (const auto& link : options.links) {
      parameters.push_back(link.input_port ? *link.input_port + 1U : 0U);
      parameters.push_back(link.auxiliary_port ? *link.auxiliary_port + 1U : 0U);
      parameters.push_back(link.carry_port
                               ? ((static_cast<std::uint64_t>(*link.carry_port) + 1U) << 32U)
                               : (link.carry ? *link.carry + 1U : 0U));
    }
  }
  std::vector<SpatialOwner> input_ports;
  for (auto& role : roles) {
    const auto prior =
        std::ranges::find_if(weights, [&](const auto& w) { return w.ref == role.ref; });
    if (prior != weights.end()) {
      if (builder.tensor(prior->tensor).format != role.format || prior->parts != role.parts)
        return std::unexpected("spatial canonical weight identity has conflicting source parts");
      role.tensor = prior->tensor;
    } else {
      role.tensor = builder.Weight(role.format, role.ref);
      weights.push_back({role.tensor, role.ref, role.parts});
    }
  }
  for (std::size_t i = 0; i < owners.size(); ++i) {
    const auto& owner = owners[i];
    builder.SetOwner(owner.owner);
    auto shape = owner.shape;
    auto input = builder.Input(Format(shape, options.input_encoding));
    std::vector<SpatialValue> produced{{input, shape}}, ports;
    input_ports.push_back(owner);
    for (const auto& port : options.inputs)
      if (port.owner == owner.owner) {
        ports.push_back({builder.Input(Format(port.shape, port.encoding)), port.shape});
        input_ports.push_back({owner.owner, port.shape});
      }
    for (std::size_t j = 0; j < chain.size(); ++j) {
      const auto* instance = chain[j];
      const auto link =
          options.links.empty() ? SpatialLink{static_cast<std::uint16_t>(j), {}} : options.links[j];
      builder.SetComponent(instance->index);
      std::array<ir::TensorId, 16> selected{};
      for (const auto& role : roles)
        if (role.component == instance->index) selected.at(role.role) = role.tensor;
      const auto& source = link.input_port ? ports[*link.input_port] : produced[link.input];
      SpatialFragment fragment{
          builder,
          *instance,
          source.tensor,
          source.shape,
          std::span(selected).first(instance->def->roles.size()),
          link.auxiliary_port
              ? std::optional(ports[*link.auxiliary_port])
              : (link.auxiliary ? std::optional(produced[*link.auxiliary]) : std::nullopt),
          link.carry_port ? std::optional(ports[*link.carry_port])
                          : (link.carry ? std::optional(produced[*link.carry]) : std::nullopt)};
      auto emitted = EmitSpatialFragment(fragment);
      if (!emitted) return std::unexpected(emitted.error());
      input = emitted->tensor;
      shape = emitted->shape;
      produced.push_back(*emitted);
    }
    if (options.exports.empty()) {
      builder.Output(input);
      parameters.push_back(owner.owner);
      for (auto s : {owner.shape, ends[i].shape})
        for (auto n : {s.width, s.height, s.channels, s.images}) parameters.push_back(n);
    } else {
      parameters.push_back(owner.owner);
      for (auto n :
           {owner.shape.width, owner.shape.height, owner.shape.channels, owner.shape.images})
        parameters.push_back(n);
      for (const auto value : options.exports) {
        const auto& exported = produced[value];
        builder.Output(exported.tensor);
        for (auto n : {exported.shape.width, exported.shape.height, exported.shape.channels,
                       exported.shape.images})
          parameters.push_back(n);
      }
    }
  }
  auto graph = std::move(builder).Finish();
  if (!graph) return std::unexpected(graph.error().detail);
  if (options.sink_layouts) {
    auto sunk = ir::SinkLayouts(*graph);
    if (!sunk) return std::unexpected("layout sinking: " + sunk.error().detail);
    for (auto& weight : weights) {
      weight.tensor = sunk->tensors.at(weight.tensor.v);
      if (!weight.tensor.valid()) return std::unexpected("layout sinking lost a weight");
    }
    *graph = std::move(sunk->graph);
  }
  return SpatialGraph(std::move(hold), std::move(*graph), std::move(weights),
                      std::move(input_ports), std::move(ends), std::move(parameters), *metadata,
                      reference_bytes);
}

}  // namespace llmp::components
