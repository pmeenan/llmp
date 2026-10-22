// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "spec/image_spatial.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <ranges>
#include <string_view>
#include <utility>

namespace llmp::spec {
namespace {
// A component's tensors by structural role, in its list's order
// (model/image_pipeline.h TextEncoderTensors, DenoiserTensors): the names are
// its mapping's, never matched here.
constexpr std::size_t kDenoiserGlobals = static_cast<std::size_t>(model::DenoiserGlobal::kCount);
constexpr std::size_t kBlockTensors = static_cast<std::size_t>(model::BlockTensor::kCount);
constexpr std::size_t kTextTensors = static_cast<std::size_t>(model::TextTensor::kCount);
const model::ImageTensor* Global(std::span<const model::ImageTensor> tensors,
                                 model::DenoiserGlobal role) {
  const auto at = static_cast<std::size_t>(role);
  return at < tensors.size() ? &tensors[at] : nullptr;
}
const model::ImageTensor* BlockOf(std::span<const model::ImageTensor> tensors, std::uint32_t block,
                                  std::size_t role) {
  const auto at = kDenoiserGlobals + (std::size_t{block} * kBlockTensors) + role;
  return at < tensors.size() ? &tensors[at] : nullptr;
}
const model::ImageTensor& TextOf(std::span<const model::ImageTensor> tensors, std::uint32_t layer,
                                 model::TextTensor role) {
  return tensors[static_cast<std::size_t>(model::TextGlobal::kCount) +
                 (std::size_t{layer} * kTextTensors) + static_cast<std::size_t>(role)];
}
const model::ImageTensor& TextGlobalOf(std::span<const model::ImageTensor> tensors,
                                       model::TextGlobal role) {
  return tensors[static_cast<std::size_t>(role)];
}
}  // namespace
std::expected<ModelSpec, std::string> VaeSpatialSpec(std::span<const model::VaeStep> steps,
                                                     std::span<const model::ImageTensor> tensors) {
  static_assert(sizeof(ComponentRef) <= 512);
  using Kind = model::VaeStep::Kind;
  if (steps.empty() || steps.size() > 64 || tensors.size() > 4096)
    return std::unexpected("a VAE spatial span needs a bounded source inventory");
  // Validate the complete source trajectory and borrowed names before copies.
  std::uint64_t width = steps.front().width, height = steps.front().height;
  std::uint32_t channels = steps.front().in_channels;
  std::uint8_t buffer = steps.front().in;
  for (const auto& step : steps) {
    if (step.in >= model::kVaeBuffers || step.out >= model::kVaeBuffers || step.in != buffer ||
        step.in_channels != channels || step.width != width || step.height != height ||
        width == 0 || height == 0 || channels == 0 || channels > 65536 || step.out_channels == 0 ||
        step.out_channels > 65536 || step.factor_t != 1)
      return std::unexpected("a VAE spatial span is not a complete serial source trajectory");
    if (step.kind == Kind::kUpsample) {
      if (step.out_channels != channels || step.weight != -1 || step.bias != -1 ||
          width > UINT32_MAX / 2 || height > UINT32_MAX / 2)
        return std::unexpected("a VAE upsample changes channels, owns weights or overflows");
      width *= 2;
      height *= 2;
    } else if (step.kind == Kind::kConv1x1 || step.kind == Kind::kConv3x3) {
      if (step.weight < 0 || step.bias < 0 ||
          static_cast<std::size_t>(step.weight) >= tensors.size() ||
          static_cast<std::size_t>(step.bias) >= tensors.size())
        return std::unexpected("a VAE convolution is missing its actual source tensors");
      const auto& w = tensors[static_cast<std::size_t>(step.weight)];
      const auto& b = tensors[static_cast<std::size_t>(step.bias)];
      const std::uint64_t k = step.kind == Kind::kConv3x3 ? 3 : 1;
      const std::array<std::uint64_t, 4> shape{step.out_channels, channels, k, k};
      if (w.dtype != "F32" || b.dtype != "F32" || w.shape.size() != 4 ||
          !std::equal(w.shape.begin(), w.shape.end(), shape.begin()) || b.shape.size() != 1 ||
          b.shape[0] != step.out_channels || w.name.empty() || b.name.empty() ||
          w.name.size() > 256 || b.name.size() > 256 || w.name == b.name)
        return std::unexpected("a VAE convolution differs from its F32 source shape or names");
    } else {
      return std::unexpected("this VAE span contains a branch or unsupported component");
    }
    buffer = step.out;
    channels = step.out_channels;
  }
  Stage stage{.name = "vae", .role = StageRole::kVae, .artifact = 0};
  stage.stream.width = steps.front().in_channels;
  Layer layer;
  layer.components.reserve(steps.size());
  for (const auto& step : steps) {
    if (step.kind == Kind::kUpsample) {
      layer.components.push_back(
          {.kind = "upsample.nearest", .version = 1, .params = {{"factor", std::int64_t{2}}}});
    } else {
      const std::int64_t k = step.kind == Kind::kConv3x3 ? 3 : 1;
      layer.components.push_back(
          {.kind = "conv.spatial",
           .version = 1,
           .params = {{"in_channels", std::int64_t{step.in_channels}},
                      {"out_channels", std::int64_t{step.out_channels}},
                      {"kernel", k},
                      {"padding", k / 2}},
           .weights = {
               {.role = "kernel", .tensor = tensors[static_cast<std::size_t>(step.weight)].name},
               {.role = "bias", .tensor = tensors[static_cast<std::size_t>(step.bias)].name}}});
    }
  }
  stage.layers.push_back(std::move(layer));
  ModelSpec spec{.pipeline = PipelineKind::kImage, .stages = {std::move(stage)}};
  Canonicalize(spec);
  if (auto valid = Validate(spec); !valid) return std::unexpected(valid.error());
  return spec;
}

std::expected<SpatialComposition, std::string> VaeDecoderSpec(
    std::span<const model::VaeStep> steps, std::span<const model::ImageTensor> tensors) {
  using Kind = model::VaeStep::Kind;
  if (steps.empty() || steps.size() > 256 || tensors.size() > 4096 ||
      steps.back().kind == Kind::kCopy)
    return std::unexpected("a VAE decoder needs a bounded complete source trajectory");
  for (const auto& tensor : tensors)
    if (tensor.name.empty() || tensor.name.size() > 256 || tensor.shape.empty() ||
        tensor.shape.size() > 4 || tensor.dtype != "F32")
      return std::unexpected("a VAE decoder tensor inventory differs from its F32 source");
  struct Value {
    std::uint16_t id = 0;
    std::uint64_t w = 0, h = 0, c = 0;
    bool valid = false;
  };
  std::array<Value, model::kVaeBuffers> current{};
  if (steps.front().in >= current.size()) return std::unexpected("invalid decoder input buffer");
  current[steps.front().in] = {0, steps.front().width, steps.front().height,
                               steps.front().in_channels, true};
  Stage stage{.name = "vae", .role = StageRole::kVae, .artifact = 0};
  stage.stream.width = steps.front().in_channels;
  Layer layer;
  layer.components.reserve(steps.size());
  std::vector<GraphLink> links;
  links.reserve(steps.size());
  const auto shape = [](const Value& v, std::uint64_t w, std::uint64_t h, std::uint64_t c) {
    return v.valid && v.w == w && v.h == h && v.c == c;
  };
  for (const auto& step : steps) {
    if (step.in >= current.size() || step.out >= current.size() || step.aux >= current.size() ||
        !step.width || !step.height || !step.in_channels || !step.out_channels ||
        step.in_channels > 65536 || step.out_channels > 65536 ||
        (step.kind != Kind::kAddDupUp && step.factor_t != 1))
      return std::unexpected("decoder source step dimensions or buffers differ");
    const auto input = current[step.in];
    Value output{0, step.width, step.height, step.out_channels, true};
    GraphLink link{input.id, {}};
    ComponentRef component;
    // DupUp describes the shortcut geometry; attention describes C, while its
    // actual source contains Q,K,V concatenated over3C. All other inputs match.
    if (step.kind != Kind::kAddDupUp &&
        !shape(input, step.width, step.height,
               step.kind == Kind::kAttention ? 3ULL * step.in_channels : step.in_channels))
      return std::unexpected("decoder step reads a missing or incompatible source value");
    if (step.kind == Kind::kCopy) {
      if (step.in_channels != step.out_channels || step.weight != -1 || step.bias != -1)
        return std::unexpected("decoder saved alias changes shape or owns weights");
      current[step.out] = input;
      continue;
    }
    const auto weight = [&](std::int32_t index) -> const model::ImageTensor* {
      return index >= 0 && static_cast<std::size_t>(index) < tensors.size()
                 ? &tensors[static_cast<std::size_t>(index)]
                 : nullptr;
    };
    if (step.kind == Kind::kConv1x1 || step.kind == Kind::kConv3x3) {
      const auto* w = weight(step.weight);
      const auto* b = weight(step.bias);
      const std::uint64_t k = step.kind == Kind::kConv3x3 ? 3 : 1;
      const std::array<std::uint64_t, 4> wanted{step.out_channels, step.in_channels, k, k};
      if (!w || !b || w->shape.size() != 4 ||
          !std::equal(w->shape.begin(), w->shape.end(), wanted.begin()) || b->shape.size() != 1 ||
          b->shape[0] != step.out_channels || w->name == b->name)
        return std::unexpected("decoder convolution source parts differ");
      component = {
          .kind = "conv.spatial",
          .version = 1,
          .params = {{"in_channels", std::int64_t{step.in_channels}},
                     {"out_channels", std::int64_t{step.out_channels}},
                     {"kernel", static_cast<std::int64_t>(k)},
                     {"padding", static_cast<std::int64_t>(k / 2)}},
          .weights = {{.role = "kernel", .tensor = w->name}, {.role = "bias", .tensor = b->name}}};
    } else if (step.kind == Kind::kNorm || step.kind == Kind::kNormSilu) {
      const auto* gamma = weight(step.weight);
      if (!gamma || gamma->shape[0] != step.in_channels || step.in_channels != step.out_channels ||
          step.bias != -1 ||
          !std::all_of(gamma->shape.begin() + 1, gamma->shape.end(), [](auto n) { return n == 1; }))
        return std::unexpected("decoder norm gamma shape or source differs");
      component = {.kind = "norm.channel",
                   .version = 1,
                   .params = {{"channels", std::int64_t{step.in_channels}},
                              {"weight_rank", static_cast<std::int64_t>(gamma->shape.size())},
                              {"weight_axis", static_cast<std::int64_t>(gamma->shape.size() - 1)},
                              {"silu", step.kind == Kind::kNormSilu}},
                   .weights = {{.role = "weight", .tensor = gamma->name}}};
    } else {
      if (step.weight != -1 || step.bias != -1)
        return std::unexpected("unweighted decoder operation names a source weight");
      switch (step.kind) {
        case Kind::kUpsample:
          if (step.in_channels != step.out_channels)
            return std::unexpected("decoder upsample changes channels");
          output.w *= 2;
          output.h *= 2;
          component = {
              .kind = "upsample.nearest", .version = 1, .params = {{"factor", std::int64_t{2}}}};
          break;
        case Kind::kAdd:
          if (step.in_channels != step.out_channels ||
              !shape(current[step.aux], output.w, output.h, output.c))
            return std::unexpected("decoder residual source geometry differs");
          link.auxiliary = current[step.aux].id;
          component = {.kind = "residual.spatial", .version = 1};
          break;
        case Kind::kAddDupUp:
          output.w *= 2;
          output.h *= 2;
          if (!shape(input, output.w, output.h, output.c) ||
              !shape(current[step.aux], step.width, step.height, step.in_channels) ||
              (step.factor_t != 1 && step.factor_t != 2) ||
              (output.c * step.factor_t * 4) % step.in_channels)
            return std::unexpected("decoder duplicate-up shortcut geometry differs");
          link.auxiliary = current[step.aux].id;
          component = {.kind = "residual.upsample",
                       .version = 1,
                       .params = {{"factor_t", std::int64_t{step.factor_t}}}};
          break;
        case Kind::kAttention:
          if (step.in_channels != step.out_channels)
            return std::unexpected("decoder attention changes its head width");
          component = {.kind = "attention.spatial", .version = 1};
          break;
        default:
          return std::unexpected("unsupported decoder source operation");
      }
    }
    layer.components.push_back(std::move(component));
    links.push_back(link);
    output.id = static_cast<std::uint16_t>(layer.components.size());
    current[step.out] = output;
  }
  if (layer.components.empty()) return std::unexpected("decoder has no computed output");
  stage.layers.push_back(std::move(layer));
  stage.links = links;
  ModelSpec spec{.pipeline = PipelineKind::kImage, .stages = {std::move(stage)}};
  Canonicalize(spec);
  if (auto valid = Validate(spec); !valid) return std::unexpected(valid.error());
  return SpatialComposition{std::move(spec), std::move(links), {}};
}

namespace {
std::expected<SpatialComposition, std::string> ImageDecodeSpecImpl(
    const model::ImageVaeProfile& profile, std::uint32_t height, std::uint32_t width,
    std::span<const model::ImageTensor> tensors, bool geometry_template) {
  if (!width || !height || width > UINT32_MAX / height || !profile.z_dim || profile.z_dim > 65536 ||
      !profile.spatial_factor || width > UINT32_MAX / profile.spatial_factor ||
      height > UINT32_MAX / profile.spatial_factor)
    return std::unexpected("latent decode geometry exceeds source representation");
  // Bound the source emitter before it creates vectors or indexes the profile.
  const auto levels = profile.dim_mult.size();
  if (levels < 2 || levels > 16 || profile.temporal_upsample.size() < levels - 1 ||
      profile.spatial_factor != (1U << (levels - 1)) || !profile.out_channels)
    return std::unexpected("latent decode level geometry differs from its source trajectory");
  std::uint64_t transitions = 0;
  auto previous = profile.dim_mult.back();
  for (const auto multiplier : std::views::reverse(profile.dim_mult)) {
    if (!multiplier || !profile.base || std::uint64_t{profile.base} * multiplier > UINT32_MAX / 3)
      return std::unexpected("latent decode channel geometry overflows");
    transitions += previous != multiplier;
    previous = multiplier;
  }
  // Boundaries and mid-block: 19 steps/24 roles. Each upsample adds four
  // steps/two roles; each level adds five steps/six roles per residual,
  // with one step/two roles for a channel-changing first residual.
  const auto residuals = std::uint64_t{profile.res_blocks} + 1;
  const auto step_count = 19 + 4 * (levels - 1) + 5 * levels * residuals + transitions;
  const auto role_count = 24 + 2 * (levels - 1) + 6 * levels * residuals + 2 * transitions;
  if (step_count > 256 || role_count > tensors.size() || tensors.size() > 4096)
    return std::unexpected("latent decode emitter exceeds its source trajectory allowance");
  const auto steps = model::VaeDecoderPlan(profile, height, width);
  auto out = VaeDecoderSpec(steps, tensors);
  if (!out) return std::unexpected(out.error());
  auto& components = out->spec.stages[0].layers[0].components;
  if (components.size() > kMaxComponents - 2)
    return std::unexpected("latent handoff exceeds component representation");
  for (auto& link : out->links) {
    link.input = static_cast<std::uint16_t>(link.input + 2);
    if (link.auxiliary) *link.auxiliary = static_cast<std::uint16_t>(*link.auxiliary + 2);
    if (link.carry) *link.carry = static_cast<std::uint16_t>(*link.carry + 2);
  }
  for (auto& value : out->exports) value = static_cast<std::uint16_t>(value + 2);
  components.insert(components.begin(), ComponentRef{.kind = "reshape.spatial",
                                                     .version = 1,
                                                     .params = {{"width", std::int64_t{width}},
                                                                {"height", std::int64_t{height}}}});
  components.insert(components.begin() + 1, ComponentRef{.kind = "affine.channel", .version = 1});
  out->links.insert(out->links.begin(), GraphLink{0, {}});
  out->links.insert(out->links.begin() + 1, GraphLink{.input = 1, .auxiliary_port = 0});
  out->spec.stages[0].links = out->links;
  out->spec.stages[0].exports = out->exports;
  if (geometry_template)
    out->spec.stages[0].scalar_bindings = {{1, "height", "latent_height"},
                                           {1, "width", "latent_width"}};
  Canonicalize(out->spec);
  if (auto valid = Validate(out->spec); !valid) return std::unexpected(valid.error());
  return out;
}

}  // namespace

std::expected<SpatialComposition, std::string> ImageDecodeSpec(
    const model::ImageVaeProfile& profile, std::uint32_t height, std::uint32_t width,
    std::span<const model::ImageTensor> tensors) {
  return ImageDecodeSpecImpl(profile, height, width, tensors, false);
}
std::expected<ModelSpec, std::string> ImageDecodeTemplate(
    const model::ImageVaeProfile& profile, std::span<const model::ImageTensor> tensors) {
  auto built = ImageDecodeSpecImpl(profile, 1, 1, tensors, true);
  if (!built) return std::unexpected(built.error());
  return std::move(built->spec);
}

std::expected<SpatialComposition, std::string> ImageTimestepSpec(
    const model::ImageDenoiserProfile& profile, std::span<const model::ImageTensor> tensors) {
  using enum model::DenoiserGlobal;
  constexpr std::array roles{kTimeLinear1Weight, kTimeLinear2Weight, kModulation, kNormOutWeight};
  const std::uint64_t width = profile.width, input = profile.timestep_dim;
  if (!width || width > 65536 / 4 || !input || input > 65536 || tensors.size() > 4096)
    return std::unexpected("image timestep source dimensions or inventory exceed component bounds");
  const std::array<std::uint64_t, 4> ins{input, width, width, width};
  const std::array<std::uint64_t, 4> outs{width, width, 4 * width, width};
  std::array<const model::ImageTensor*, 4> sources{};
  for (const auto& tensor : tensors)
    if (tensor.name.empty() || tensor.name.size() > 256 || tensor.shape.size() > 4)
      return std::unexpected("image timestep source metadata exceeds mapping allowance");
  for (std::size_t i = 0; i < roles.size(); ++i) {
    sources[i] = Global(tensors, roles[i]);
    if (sources[i] == nullptr) return std::unexpected("image timestep source role is missing");
    const auto& tensor = *sources[i];
    if (tensor.dtype != "BF16" || tensor.shape.size() != 2 || tensor.shape[0] != outs[i] ||
        tensor.shape[1] != ins[i] ||
        std::ranges::count(tensors, tensor.name, &model::ImageTensor::name) != 1)
      return std::unexpected("image timestep source role encoding or shape differs");
  }
  Stage stage{.name = "conditioning", .role = StageRole::kDenoiser, .artifact = 0};
  stage.stream.width = static_cast<std::uint32_t>(input);
  Layer layer;
  layer.components.reserve(4);
  for (std::size_t i = 0; i < roles.size(); ++i)
    layer.components.push_back(
        {.kind = "projection.spatial",
         .version = 1,
         .params = {{"in_channels", static_cast<std::int64_t>(ins[i])},
                    {"out_channels", static_cast<std::int64_t>(outs[i])},
                    {"silu", roles[i] == kTimeLinear1Weight || roles[i] == kTimeLinear2Weight}},
         .weights = {{.role = "weight", .tensor = sources[i]->name}}});
  stage.layers.push_back(std::move(layer));
  std::vector<GraphLink> links{{0, {}}, {1, {}}, {2, {}}, {2, {}}};
  std::vector<std::uint16_t> exports{3, 4};
  stage.links = links;
  stage.exports = exports;
  ModelSpec spec{.pipeline = PipelineKind::kImage, .stages = {std::move(stage)}};
  Canonicalize(spec);
  if (auto valid = Validate(spec); !valid) return std::unexpected(valid.error());
  return SpatialComposition{std::move(spec), std::move(links), std::move(exports)};
}
namespace {
std::expected<std::uint16_t, std::string> AppendImageBlock(
    SpatialComposition& out, const model::ImageDenoiserProfile& p, std::uint32_t block,
    std::span<const model::ImageTensor> tensors, std::uint16_t stream,
    const std::array<std::uint16_t, 4>& condition, std::uint16_t position_port,
    std::uint16_t mask_port) {
  const std::uint64_t width = p.width, dim = p.head_dim;
  if (block >= p.blocks || !p.heads || !dim || dim > 512 || width != std::uint64_t{p.heads} * dim ||
      !p.mlp || p.mlp > 65536 || !std::isfinite(p.eps) || p.eps <= 0 || p.eps > 1 ||
      !std::isfinite(p.rope_theta) || p.rope_theta < 1 || p.rope_theta > 1e10 ||
      std::uint64_t{p.rope_axes[0]} + p.rope_axes[1] + p.rope_axes[2] != dim ||
      std::ranges::any_of(p.rope_axes, [](auto n) { return n < 2 || n % 2; }))
    return std::unexpected("image block profile exceeds the shared mathematical domain");
  // Validate all borrowed roles before copying the conditioning mapping.
  if (tensors.size() > 4096) return std::unexpected("image block source inventory is unbounded");
  for (const auto& t : tensors)
    if (t.name.empty() || t.name.size() > 256 || t.shape.empty() || t.shape.size() > 4)
      return std::unexpected("image block source descriptor exceeds mapping allowance");
  using enum model::BlockTensor;
  std::array<const model::ImageTensor*, kBlockTensors> sources{};
  for (std::size_t i = 0; i < sources.size(); ++i) {
    sources[i] = BlockOf(tensors, block, i);
    if (sources[i] == nullptr) return std::unexpected("image block source role is absent");
    const auto& t = *sources[i];
    const bool norm =
        i == static_cast<std::size_t>(kQNorm) || i == static_cast<std::size_t>(kKNorm);
    const auto input = i == static_cast<std::size_t>(kMlpOut) ? std::uint64_t{p.mlp} : width;
    const auto output =
        (i == static_cast<std::size_t>(kGate) || i == static_cast<std::size_t>(kProj))
            ? std::uint64_t{p.mlp}
            : width;
    if (t.dtype != "BF16" ||
        (norm ? t.shape.size() != 1 || t.shape[0] != dim
              : t.shape.size() != 2 || t.shape[0] != output || t.shape[1] != input))
      return std::unexpected("image block source encoding or role geometry differs");
  }
  auto& components = out.spec.stages[0].layers[0].components;
  if (components.size() > kMaxComponents - 6)
    return std::unexpected("image blocks exceed value-ID capacity");
  const auto add = [&](ComponentRef c, GraphLink link) {
    components.push_back(std::move(c));
    out.links.push_back(link);
    return static_cast<std::uint16_t>(components.size());
  };
  const auto name = [&](model::BlockTensor role) {
    return sources[static_cast<std::size_t>(role)]->name;
  };
  const auto adaptive = [&](std::uint16_t input, std::uint16_t side) {
    return add({.kind = "norm.adaptive",
                .version = 1,
                .params = {{"side", std::string("scale")}, {"eps", double{p.eps}}}},
               {input, side});
  };
  const auto gated = [&](std::uint16_t input, std::uint16_t side, std::uint16_t carry) {
    return add({.kind = "residual.gated", .version = 1, .params = {{"side", std::string("gate")}}},
               {.input = input, .auxiliary = side, .carry = carry});
  };
  const auto normalized = adaptive(stream, condition[0]);
  const auto attention =
      add({.kind = "attention.dense",
           .version = 1,
           .params = {{"eps", double{p.eps}},
                      {"heads", std::int64_t{p.heads}},
                      {"key_dim", std::int64_t{p.head_dim}},
                      {"kv_heads", std::int64_t{p.heads}},
                      {"placement", std::string("bare")},
                      {"qk_norm", true},
                      {"rope_axis0", std::int64_t{p.rope_axes[0]}},
                      {"rope_axis1", std::int64_t{p.rope_axes[1]}},
                      {"rope_axis2", std::int64_t{p.rope_axes[2]}},
                      {"rope_base", double{p.rope_theta}},
                      {"rope_dims", std::int64_t{p.head_dim}},
                      {"rope_scale", 1.0},
                      {"rotation", std::string("normal")},
                      {"scale", 1.0 / std::sqrt(static_cast<double>(dim))},
                      {"scale_scores", true},
                      {"stateless", true},
                      {"value_dim", std::int64_t{p.head_dim}},
                      {"window", std::int64_t{0}}},
           .weights = {{.role = "k", .tensor = name(kK)},
                       {.role = "k_norm", .tensor = name(kKNorm)},
                       {.role = "output", .tensor = name(kOut)},
                       {.role = "q", .tensor = name(kQ)},
                       {.role = "q_norm", .tensor = name(kQNorm)},
                       {.role = "v", .tensor = name(kV)}}},
          {.input = normalized, .auxiliary_port = position_port, .carry_port = mask_port});
  const auto residual = gated(attention, condition[1], stream);
  const auto mlp_input = adaptive(residual, condition[2]);
  const auto mlp = add({.kind = "ffn.glu",
                        .version = 1,
                        .params = {{"activation", std::string("silu")},
                                   {"eps", double{p.eps}},
                                   {"hidden", std::int64_t{p.mlp}},
                                   {"placement", std::string("bare")}},
                        .weights = {{.role = "down", .tensor = name(kMlpOut)},
                                    {.role = "gate", .tensor = name(kGate)},
                                    {.role = "up", .tensor = name(kProj)}}},
                       {mlp_input, {}});
  const auto result = gated(mlp, condition[3], residual);
  return result;
}
}  // namespace
std::expected<SpatialComposition, std::string> ImageTransformerBlockSpec(
    const model::ImageDenoiserProfile& p, std::uint32_t block, std::uint32_t text_rows,
    std::span<const model::ImageTensor> tensors) {
  auto out = ImageTimestepSpec(p, tensors);
  if (!out) return std::unexpected(out.error());
  auto& stage = out->spec.stages[0];
  stage.name = "conditioned";
  stage.stream.width = p.width;
  auto& components = stage.layers[0].components;
  components.reserve(14);
  out->links.reserve(14);
  out->links[0].input_port = 0;
  const auto add = [&](ComponentRef c, GraphLink link) {
    components.push_back(std::move(c));
    out->links.push_back(link);
    return static_cast<std::uint16_t>(components.size());
  };
  std::array<std::uint16_t, 4> condition{};
  for (std::size_t i = 0; i < condition.size(); ++i)
    condition[i] = add({.kind = "condition.rows",
                        .version = 1,
                        .params = {{"quarter", static_cast<std::int64_t>(i)},
                                   {"prefix_rows", std::int64_t{text_rows}}}},
                       {0, 3});
  auto result = AppendImageBlock(*out, p, block, tensors, 0, condition, 1, 2);
  if (!result) return std::unexpected(result.error());
  out->exports = {*result, 4};
  stage.links = out->links;
  stage.exports = out->exports;
  Canonicalize(out->spec);
  if (auto valid = Validate(out->spec); !valid) return std::unexpected(valid.error());
  return out;
}

std::expected<std::uint64_t, std::string> ImageTextSpecBound(
    const model::ImageTextProfile& p, std::span<const model::ImageTensor> tensors) {
  const auto components = std::uint64_t{p.layers} * 2 + 2;
  if (!p.layers || components > kMaxComponents || tensors.size() > 4096 ||
      std::uint64_t{p.layers} * 11 + 1 > tensors.size())
    return std::unexpected("text mapping exceeds component/source representation");
  for (const auto& tensor : tensors)
    if (tensor.name.empty() || tensor.name.size() > 256 || tensor.shape.empty() ||
        tensor.shape.size() > 4)
      return std::unexpected("text mapping source descriptor is unbounded");
  return 4096 * components + 1024 * tensors.size() + 65536;
}
std::expected<SpatialComposition, std::string> ImageTextSpec(
    const model::ImageTextProfile& p, std::uint32_t rows, std::uint32_t drop,
    std::span<const model::ImageTensor> tensors) {
  auto bound = ImageTextSpecBound(p, tensors);
  if (!bound) return std::unexpected(bound.error());
  if (!rows || drop >= rows || !p.width || p.width > 65536 || !p.heads || !p.kv_heads ||
      p.heads % p.kv_heads || !p.head_dim || p.head_dim > 512 || p.head_dim % 2 ||
      std::uint64_t{p.heads} * p.head_dim != p.width || !p.ffn || p.ffn > 65536 || !p.vocab ||
      !std::isfinite(p.rms_eps) || p.rms_eps <= 0 || p.rms_eps > 1 ||
      !std::isfinite(p.rope_theta) || p.rope_theta < 1 || p.rope_theta > 1e10)
    return std::unexpected("text source profile or prompt geometry differs");
  // Every required source is validated before any generated component
  // copies: the list's roles in order, their encodings and shapes.
  const auto required = model::TextEncoderTensors(p, {});
  if (required.size() != std::uint64_t{p.layers} * kTextTensors + 1 ||
      tensors.size() != required.size())
    return std::unexpected("text source role inventory differs");
  for (std::size_t i = 0; i < required.size(); ++i)
    if (tensors[i].dtype != "BF16" || tensors[i].shape != required[i].shape)
      return std::unexpected("text source role encoding or geometry differs");
  Stage stage{.name = "text", .role = StageRole::kTextEncoder, .artifact = 0};
  stage.stream.width = p.width;
  stage.layers.emplace_back();
  SpatialComposition out{.spec = {.pipeline = PipelineKind::kImage, .stages = {std::move(stage)}},
                         .links = {},
                         .exports = {}};
  auto& components = out.spec.stages[0].layers[0].components;
  const auto add = [&](ComponentRef component, GraphLink link) {
    components.push_back(std::move(component));
    out.links.push_back(link);
    return static_cast<std::uint16_t>(components.size());
  };
  auto current =
      add({.kind = "embedding.token",
           .version = 1,
           .weights = {{.role = "table",
                        .tensor = TextGlobalOf(tensors, model::TextGlobal::kEmbedding).name}}},
          {0, {}});
  for (std::uint32_t layer = 0; layer < p.layers; ++layer) {
    using enum model::TextTensor;
    const auto name = [&](model::TextTensor role) { return TextOf(tensors, layer, role).name; };
    current = add({.kind = "attention.dense",
                   .version = 1,
                   .params = {{"eps", double{p.rms_eps}},
                              {"heads", std::int64_t{p.heads}},
                              {"key_dim", std::int64_t{p.head_dim}},
                              {"kv_heads", std::int64_t{p.kv_heads}},
                              {"placement", std::string("pre")},
                              {"qk_norm", true},
                              {"rope_base", double{p.rope_theta}},
                              {"rope_dims", std::int64_t{p.head_dim}},
                              {"rope_scale", 1.0},
                              {"rotation", std::string("neox")},
                              {"scale", 1.0 / std::sqrt(static_cast<double>(p.head_dim))},
                              {"scale_scores", true},
                              {"stateless", true},
                              {"value_dim", std::int64_t{p.head_dim}},
                              {"window", std::int64_t{0}}},
                   .weights = {{.role = "k", .tensor = name(kK)},
                               {.role = "k_norm", .tensor = name(kKNorm)},
                               {.role = "norm", .tensor = name(kInputNorm)},
                               {.role = "output", .tensor = name(kO)},
                               {.role = "q", .tensor = name(kQ)},
                               {.role = "q_norm", .tensor = name(kQNorm)},
                               {.role = "v", .tensor = name(kV)}}},
                  {.input = current, .auxiliary_port = 0, .carry_port = 1});
    current = add({.kind = "ffn.glu",
                   .version = 1,
                   .params = {{"activation", std::string("silu")},
                              {"eps", double{p.rms_eps}},
                              {"hidden", std::int64_t{p.ffn}},
                              {"placement", std::string("pre")}},
                   .weights = {{.role = "down", .tensor = name(kDown)},
                               {.role = "gate", .tensor = name(kGate)},
                               {.role = "norm", .tensor = name(kPostNorm)},
                               {.role = "up", .tensor = name(kUp)}}},
                  {current, {}});
  }
  current = add({.kind = "slice.spatial",
                 .version = 1,
                 .params = {{"start", std::int64_t{drop}},
                            {"rows", std::int64_t{rows - drop}},
                            {"input_rows", std::int64_t{rows}}}},
                {current, {}});
  out.exports = {current};
  out.spec.stages[0].links = out.links;
  out.spec.stages[0].exports = out.exports;
  Canonicalize(out.spec);
  if (auto checked = Validate(out.spec); !checked) return std::unexpected(checked.error());
  return out;
}

std::expected<std::uint64_t, std::string> ImageFlowSpecBound(
    const model::ImageDenoiserProfile& p, std::span<const model::ImageTensor> tensors) {
  constexpr std::uint64_t fixed_components = 19, per_block = 6;
  const std::uint64_t components = fixed_components + per_block * std::uint64_t{p.blocks};
  if (!p.blocks || components > kMaxComponents || tensors.size() > 4096)
    return std::unexpected("image flow mapping exceeds spec value-ID or source inventory capacity");
  for (const auto& t : tensors)
    if (t.name.empty() || t.name.size() > 256 || t.shape.empty() || t.shape.size() > 4)
      return std::unexpected("image flow source metadata exceeds mapping bounds");
  // Complete returned component/param/weight/link copies and one append scratch
  // inventory coexist; source payloads are borrowed, never copied here.
  return 4096 * components + 1024 * tensors.size() + 65536;
}
namespace {
std::expected<SpatialComposition, std::string> ImageFlowSpecImpl(
    const model::ImageDenoiserProfile& p, std::uint32_t text_rows, std::uint32_t image_rows,
    std::span<const model::ImageTensor> tensors, bool geometry_template) {
  auto bound = ImageFlowSpecBound(p, tensors);
  if (!bound) return std::unexpected(bound.error());
  // The whole denoiser's list, by role (DenoiserTensors' order).
  if (tensors.size() != kDenoiserGlobals + (std::size_t{p.blocks} * kBlockTensors))
    return std::unexpected("image flow source inventory is not the denoiser's tensor list");
  if (!text_rows || !image_rows ||
      image_rows > std::numeric_limits<std::uint32_t>::max() - text_rows || p.context != p.width ||
      !p.in_channels || p.in_channels != p.out_channels || p.in_channels > 65536)
    return std::unexpected(
        "image flow needs matching source context/stream and Euler channels with bounded rows");
  using enum model::DenoiserGlobal;
  constexpr std::array roles{kTxtNorm, kTxtIn, kTxtOut, kImgIn, kProjOut};
  std::array<const model::ImageTensor*, roles.size()> sources{};
  for (std::size_t i = 0; i < roles.size(); ++i) {
    sources[i] = Global(tensors, roles[i]);
    if (sources[i] == nullptr) return std::unexpected("image flow boundary source role is missing");
    const auto& t = *sources[i];
    const std::uint64_t input = roles[i] == kImgIn ? p.in_channels : p.width;
    const std::uint64_t output = roles[i] == kProjOut ? p.out_channels : p.width;
    if (t.dtype != "BF16" ||
        (roles[i] == kTxtNorm ? t.shape.size() != 1 || t.shape[0] != p.context
                              : t.shape.size() != 2 || t.shape[0] != output || t.shape[1] != input))
      return std::unexpected("image flow boundary source role/encoding/geometry differs");
  }
  auto out = ImageTimestepSpec(p, tensors);
  if (!out) return std::unexpected(out.error());
  auto& stage = out->spec.stages[0];
  stage.name = "flow";
  stage.stream.width = p.context;
  auto& components = stage.layers[0].components;
  components.reserve(19 + 6 * std::size_t{p.blocks});
  out->links.reserve(components.capacity());
  out->links[0].input_port = 1;
  const auto add = [&](ComponentRef c, GraphLink link) {
    components.push_back(std::move(c));
    out->links.push_back(link);
    return static_cast<std::uint16_t>(components.size());
  };
  const auto projection = [&](model::DenoiserGlobal role, std::uint16_t input, bool gelu = false) {
    const auto source = Global(tensors, role);
    return add({.kind = "projection.spatial",
                .version = 1,
                .params = {{"in_channels", static_cast<std::int64_t>(source->shape[1])},
                           {"out_channels", static_cast<std::int64_t>(source->shape[0])},
                           {"gelu", gelu}},
                .weights = {{.role = "weight", .tensor = source->name}}},
               {input, {}});
  };
  const auto text_norm =
      add({.kind = "norm.rms",
           .version = 1,
           .params = {{"channels", std::int64_t{p.context}},
                      {"eps", double{p.eps}},
                      {"norm_zero_centered", true}},
           .weights = {{.role = "weight", .tensor = Global(tensors, kTxtNorm)->name}}},
          {0, {}});
  const auto text = projection(kTxtOut, projection(kTxtIn, text_norm, true));
  const auto latent = add({.kind = "slice.spatial",
                           .version = 1,
                           .params = {{"start", std::int64_t{0}},
                                      {"rows", std::int64_t{image_rows}},
                                      {"input_rows", std::int64_t{image_rows}}}},
                          {.input_port = 0});
  if (geometry_template) {
    stage.scalar_bindings.push_back({latent, "input_rows", "latent_rows"});
    stage.scalar_bindings.push_back({latent, "rows", "latent_rows"});
  }
  const auto image = projection(kImgIn, latent);
  auto stream = add({.kind = "concat.spatial", .version = 1, .params = {{"axis", std::int64_t{1}}}},
                    {text, image});
  std::array<std::uint16_t, 4> condition{};
  for (std::size_t i = 0; i < condition.size(); ++i) {
    condition[i] = add({.kind = "condition.rows",
                        .version = 1,
                        .params = {{"quarter", static_cast<std::int64_t>(i)},
                                   {"prefix_rows", std::int64_t{text_rows}}}},
                       {stream, 3});
    if (geometry_template)
      stage.scalar_bindings.push_back({condition[i], "prefix_rows", "text_rows"});
  }
  for (std::uint32_t block = 0; block < p.blocks; ++block) {
    auto result = AppendImageBlock(*out, p, block, tensors, stream, condition, 2, 3);
    if (!result) return std::unexpected(result.error());
    stream = *result;
  }
  const auto image_out = add({.kind = "slice.spatial",
                              .version = 1,
                              .params = {{"start", std::int64_t{text_rows}},
                                         {"rows", std::int64_t{image_rows}},
                                         {"input_rows", std::int64_t{text_rows + image_rows}}}},
                             {stream, {}});
  if (geometry_template) {
    stage.scalar_bindings.push_back({image_out, "input_rows", "joint_rows"});
    stage.scalar_bindings.push_back({image_out, "rows", "latent_rows"});
    stage.scalar_bindings.push_back({image_out, "start", "text_rows"});
  }
  const auto scale = add({.kind = "condition.rows",
                          .version = 1,
                          .params = {{"quarter", std::int64_t{0}},
                                     {"quarters", std::int64_t{1}},
                                     {"prefix_rows", std::int64_t{0}}}},
                         {image_out, 4});
  const auto normalized = add({.kind = "norm.adaptive",
                               .version = 1,
                               .params = {{"side", std::string("scale")}, {"eps", double{p.eps}}}},
                              {image_out, scale});
  const auto noise = projection(kProjOut, normalized);
  const auto next = add({.kind = "residual.scaled", .version = 1},
                        {.input = noise, .auxiliary_port = 4, .carry = latent});
  out->exports = {noise, next};
  stage.links = out->links;
  stage.exports = out->exports;
  Canonicalize(out->spec);
  if (auto valid = Validate(out->spec); !valid) return std::unexpected(valid.error());
  return out;
}

}  // namespace

std::expected<SpatialComposition, std::string> ImageFlowSpec(
    const model::ImageDenoiserProfile& profile, std::uint32_t text_rows, std::uint32_t image_rows,
    std::span<const model::ImageTensor> tensors) {
  return ImageFlowSpecImpl(profile, text_rows, image_rows, tensors, false);
}
std::expected<ModelSpec, std::string> ImageFlowTemplate(
    const model::ImageDenoiserProfile& profile, std::span<const model::ImageTensor> tensors) {
  auto built = ImageFlowSpecImpl(profile, 1, 1, tensors, true);
  if (!built) return std::unexpected(built.error());
  return std::move(built->spec);
}

}  // namespace llmp::spec
