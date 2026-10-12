// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "components/spatial.h"

#include <algorithm>
#include <array>
#include <limits>
#include <utility>

#include "components/defs.h"

namespace llmp::components {
namespace {

constexpr std::uint64_t kValues = 1ULL << 24U;
constexpr std::size_t kComponents = 64, kOwners = 16, kTensors = 1024;

std::expected<std::uint64_t, std::string> Values(SpatialShape s) {
  std::uint64_t count = 1;
  for (auto n : {s.width, s.height, s.channels, s.images})
    if (n == 0 || __builtin_mul_overflow(count, n, &count) || count > kValues)
      return std::unexpected("spatial dimensions exceed the bounded value domain");
  return count;
}
format::TensorFormat Format(SpatialShape s) {
  return {.encoding = format::Encoding::kF32,
          .layout = format::Layout::kDense,
          .shape = ir::ShapeOf({s.width, s.height, s.channels, s.images})};
}
std::uint64_t Index(SpatialShape s, std::uint64_t x, std::uint64_t y, std::uint64_t c,
                    std::uint64_t n) {
  return x + s.width * (y + s.height * (c + s.channels * n));
}

constexpr std::array kConvParams{
    ParamSpec{.name = "in_channels", .type = ParamType::kInt, .least = 1, .most = 65536},
    ParamSpec{.name = "out_channels", .type = ParamType::kInt, .least = 1, .most = 65536},
    ParamSpec{.name = "kernel", .type = ParamType::kInt, .least = 1, .most = 3},
    ParamSpec{.name = "padding", .type = ParamType::kInt, .least = 0, .most = 1},
};
constexpr std::array kConvRoles{WeightRole{.name = "kernel", .encodings = defs::kNormEncodings},
                                WeightRole{.name = "bias", .encodings = defs::kNormEncodings}};
constexpr std::array kUpParams{
    ParamSpec{.name = "factor", .type = ParamType::kInt, .least = 2, .most = 2}};

std::expected<std::vector<RoleShape>, std::string> ConvRoles(const Instance& in) {
  const auto k = static_cast<std::uint64_t>(in.params.Int("kernel"));
  if (in.site != Site::kLayer || (k != 1 && k != 3))
    return std::unexpected("spatial convolution requires a layer and a 1x1 or 3x3 kernel");
  const auto ci = static_cast<std::uint64_t>(in.params.Int("in_channels"));
  const auto co = static_cast<std::uint64_t>(in.params.Int("out_channels"));
  if (!Values({k, k, ci, co}))
    return std::unexpected("spatial convolution kernel exceeds its value bound");
  return std::vector{defs::Role(0, {k, k, ci, co}), defs::Role(1, {co})};
}
std::expected<std::vector<RoleShape>, std::string> UpRoles(const Instance& in) {
  if (in.site != Site::kLayer) return std::unexpected("spatial upsampling requires a layer");
  return std::vector<RoleShape>{};
}
std::expected<SpatialShape, std::string> ConvShape(const Instance& in, SpatialShape s) {
  if (!Values(s) || s.channels != static_cast<std::uint64_t>(in.params.Int("in_channels")))
    return std::unexpected("spatial convolution input channels or dimensions differ");
  const auto k = static_cast<std::uint64_t>(in.params.Int("kernel"));
  const auto pad = static_cast<std::uint64_t>(in.params.Int("padding"));
  std::uint64_t w = 0, h = 0;
  if (__builtin_add_overflow(s.width, 2 * pad, &w) ||
      __builtin_add_overflow(s.height, 2 * pad, &h) || w < k || h < k)
    return std::unexpected("spatial convolution padding or kernel is invalid");
  SpatialShape out{w - k + 1, h - k + 1, static_cast<std::uint64_t>(in.params.Int("out_channels")),
                   s.images};
  if (!Values(out)) return std::unexpected("spatial convolution output exceeds its value bound");
  return out;
}
std::expected<SpatialShape, std::string> UpShape(const Instance& in, SpatialShape s) {
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
  const ir::Conv2dParams params{
      .stride = 1,
      .padding = static_cast<std::uint32_t>(f.instance.params.Int("padding")),
      .dilation = 1};
  return SpatialValue{f.builder.Conv2d(f.weights[0], f.input, f.weights[1], params), *out};
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
  if (!out || !count || io.input.size() != *count ||
      io.kernel.size() != k * k * io.shape.channels * co || io.bias.size() != co)
    return std::unexpected("spatial convolution reference input/weight sizes differ");
  std::vector<double> result(static_cast<std::size_t>(*Values(*out)));
  const auto pad = io.instance.params.Int("padding");
  // Independent scalar cross-correlation in WHCN order; no forced legacy casts.
  for (std::uint64_t n = 0; n < out->images; ++n)
    for (std::uint64_t c = 0; c < co; ++c)
      for (std::uint64_t y = 0; y < out->height; ++y)
        for (std::uint64_t x = 0; x < out->width; ++x) {
          double value = io.bias[c];
          for (std::uint64_t ci = 0; ci < io.shape.channels; ++ci)
            for (std::uint64_t ky = 0; ky < k; ++ky)
              for (std::uint64_t kx = 0; kx < k; ++kx) {
                const auto ix = static_cast<std::int64_t>(x + kx) - pad;
                const auto iy = static_cast<std::int64_t>(y + ky) - pad;
                if (ix < 0 || iy < 0 || static_cast<std::uint64_t>(ix) >= io.shape.width ||
                    static_cast<std::uint64_t>(iy) >= io.shape.height)
                  continue;
                const auto weight = kx + k * (ky + k * (ci + io.shape.channels * c));
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
std::expected<FragmentOut, std::string> RefuseToken(Fragment&) {
  return std::unexpected("spatial components require typed spatial graph inputs");
}
std::expected<void, std::string> RefuseTokenReference(RefIO&) {
  return std::unexpected("spatial reference requires explicit WHCN geometry");
}
const SpatialOps kConv{ConvShape, ConvEmit, ConvReference};
const SpatialOps kUp{UpShape, UpEmit, UpReference};

// Count and bound borrowed metadata before any copy, validation allocation or
// visitor. This first spatial domain has no side/state/embedding/temporal path.
std::expected<std::uint64_t, std::string> MetadataBound(
    const spec::ModelSpec& spec, std::span<const std::span<const BoundTensor>> tensors,
    std::span<const SpatialOwner> owners) {
  if (spec.pipeline != spec::PipelineKind::kImage || spec.stages.size() != 1 ||
      tensors.size() > 16 || owners.empty() || owners.size() > kOwners)
    return std::unexpected("spatial compilation needs one image stage and 1..16 owners");
  const auto& stage = spec.stages.front();
  if (stage.name.size() > 256 || stage.layers.empty() || stage.layers.size() > kComponents ||
      stage.embedding || !stage.side.empty() || stage.stream.mixing || stage.stream.streams != 1 ||
      stage.heads.output || stage.heads.final_norm || !stage.heads.mtp.empty() ||
      stage.heads.softcap || !stage.reads.empty() || !stage.taps.empty() || !stage.folds.empty() ||
      stage.context != 0)
    return std::unexpected("spatial compilation requires a bounded stateless component chain");
  std::size_t components = 0, source_count = 0;
  for (const auto& layer : stage.layers) {
    if (layer.components.empty() || layer.components.size() > kComponents - components)
      return std::unexpected("spatial component count is outside its bound");
    components += layer.components.size();
    for (const auto& ref : layer.components) {
      if (ref.kind.size() > 64 || ref.params.size() > 8 || ref.weights.size() > 2)
        return std::unexpected("spatial component metadata exceeds its bound");
      const auto* def = Find(ref.kind, ref.version);
      if (def == nullptr || def->spatial == nullptr || def->state != nullptr ||
          def->params.size() > 8 || def->roles.size() > 2)
        return std::unexpected("a component has no bounded stateless spatial fragment");
      for (const auto& param : ref.params)
        if (param.name.size() > 64 || !std::holds_alternative<std::int64_t>(param.value))
          return std::unexpected("spatial parameters are bounded scalar integers");
      for (const auto& weight : ref.weights)
        if (weight.role.size() > 64 || weight.tensor.size() > 256 || !weight.stage.empty())
          return std::unexpected("spatial weight binding exceeds the single-stage domain");
    }
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
    if (!Values(owners[i].shape) || owners[i].shape.channels != stage.stream.width ||
        owners[i].owner >= ir::kJoinedSegment || (i && owners[i - 1].owner >= owners[i].owner))
      return std::unexpected("spatial owner order or initial channels/dimensions differ");
  }
  // <=2 roles/component, one input/owner and one primitive/component/owner.
  // 4KiB per node/tensor covers graph+clone+rewrite/placement metadata, and
  // 4KiB per source/component covers bounded names/Params/visitor copies.
  static_assert(sizeof(ir::Node) <= 256 && sizeof(ir::TensorDesc) <= 256);
  const auto nodes = owners.size() * (components + 1) + 2 * components;
  return 4096ULL * (2 * nodes + source_count + 4 * components + 16);
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
                                .spatial = &kConv,
                                .emit = RefuseToken,
                                .reference = RefuseTokenReference};
  return def;
}
const ComponentDef& UpsampleNearest() {
  static const ComponentDef def{.kind = "upsample.nearest",
                                .params = kUpParams,
                                .shapes = UpRoles,
                                .spatial = &kUp,
                                .emit = RefuseToken,
                                .reference = RefuseTokenReference};
  return def;
}
}  // namespace defs

std::expected<SpatialGraph, std::string> BuildSpatialStep(
    const spec::ModelSpec& spec, std::span<const std::span<const BoundTensor>> tensors,
    std::span<const SpatialOwner> owners, SpatialPayment payment) {
  auto metadata = MetadataBound(spec, tensors, owners);
  if (!metadata) return std::unexpected(metadata.error());
  if (!payment.charge || !payment.release || !payment.charge(*metadata))
    return std::unexpected("spatial compiler metadata was not funded");
  auto hold = std::make_shared<PaymentOwner>(std::move(payment), *metadata);
  auto instances = InstancesOf(spec, 0);
  if (!instances) return std::unexpected(instances.error());
  std::vector<const Instance*> chain;
  for (const auto& layer : instances->layers)
    for (const auto& instance : layer) {
      if (!instance.def->spatial || instance.def->state)
        return std::unexpected("a component has no stateless spatial fragment");
      chain.push_back(&instance);
    }
  // Whole-chain inference before builder creation; input + every live result.
  std::uint64_t values = 0;
  std::vector<SpatialOwner> ends;
  for (const auto& owner : owners) {
    auto shape = owner.shape;
    values += *Values(shape);
    for (const auto* instance : chain) {
      auto next = instance->def->spatial->infer(*instance, shape);
      if (!next) return std::unexpected(next.error());
      shape = *next;
      values += *Values(shape);
    }
    ends.push_back({owner.owner, shape});
  }
  std::vector<RoleUse> roles;
  auto visited = VisitWeightBindings(
      spec, tensors, [&](const WeightBindingView& role) -> std::expected<void, std::string> {
        const auto parts = format::PartsOf(role.format);
        if (role.use_stage != 0 || role.rows_per_token || role.head_rows || !parts ||
            parts->size() != 1 || role.parts.size() != 1 ||
            role.format.layout != format::Layout::kDense ||
            (role.format.encoding != format::Encoding::kF32 &&
             role.format.encoding != format::Encoding::kBF16))
          return std::unexpected("spatial roles need complete dense F32/BF16 source parts");
        std::uint64_t count = 1;
        for (auto n : role.format.shape.ne)
          if (__builtin_mul_overflow(count, n, &count) || count > kValues)
            return std::unexpected("spatial weight value count exceeds its bound");
        values += count;
        roles.push_back({role.use_component,
                         role.use_role,
                         role.ref,
                         role.format,
                         {role.parts.begin(), role.parts.end()},
                         {}});
        return {};
      });
  if (!visited) return std::unexpected(visited.error());
  // Bound encoded roots, complete reference copies and decoded double work
  // across primitive execution and the independent chain oracle (not GPU VMM).
  std::uint64_t reference_bytes = 0;
  if (__builtin_mul_overflow(values, std::uint64_t{64}, &reference_bytes) ||
      !hold->Add(reference_bytes))
    return std::unexpected("spatial graph/reference buffers were not funded");
  const auto max_nodes = owners.size() * (chain.size() + 1) + roles.size();
  ir::Builder builder(max_nodes);
  std::vector<StepWeight> weights;
  std::vector<std::uint64_t> parameters{0x5350415449414cULL, 1, owners.size()};
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
    auto input = builder.Input(Format(shape));
    for (const auto* instance : chain) {
      builder.SetComponent(instance->index);
      std::array<ir::TensorId, 2> selected{};
      for (const auto& role : roles)
        if (role.component == instance->index) selected.at(role.role) = role.tensor;
      SpatialFragment fragment{builder, *instance, input, shape,
                               std::span(selected).first(instance->def->roles.size())};
      auto emitted = instance->def->spatial->emit(fragment);
      if (!emitted) return std::unexpected(emitted.error());
      input = emitted->tensor;
      shape = emitted->shape;
    }
    builder.Output(input);
    parameters.push_back(owner.owner);
    for (auto s : {owner.shape, ends[i].shape})
      for (auto n : {s.width, s.height, s.channels, s.images}) parameters.push_back(n);
  }
  auto graph = std::move(builder).Finish();
  if (!graph) return std::unexpected(graph.error().detail);
  return SpatialGraph(std::move(hold), std::move(*graph), std::move(weights),
                      std::vector<SpatialOwner>(owners.begin(), owners.end()), std::move(ends),
                      std::move(parameters), *metadata, reference_bytes);
}

}  // namespace llmp::components
