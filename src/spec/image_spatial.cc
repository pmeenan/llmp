// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "spec/image_spatial.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string_view>
#include <utility>

namespace llmp::spec {
std::expected<ModelSpec, std::string> VaeSpatialSpec(
    std::span<const model::VaeStep> steps, std::span<const model::QwenImageTensor> tensors) {
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
    std::span<const model::VaeStep> steps, std::span<const model::QwenImageTensor> tensors) {
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
        step.width > 4096 || step.height > 4096 || step.in_channels > 65536 ||
        step.out_channels > 65536 || (step.kind != Kind::kAddDupUp && step.factor_t != 1))
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
    const auto weight = [&](std::int32_t index) -> const model::QwenImageTensor* {
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
  ModelSpec spec{.pipeline = PipelineKind::kImage, .stages = {std::move(stage)}};
  Canonicalize(spec);
  if (auto valid = Validate(spec); !valid) return std::unexpected(valid.error());
  return SpatialComposition{std::move(spec), std::move(links), {}};
}
std::expected<SpatialComposition, std::string> ImageTimestepSpec(
    const model::QwenImageDenoiserProfile& profile,
    std::span<const model::QwenImageTensor> tensors) {
  constexpr std::array<std::string_view, 4> names{
      "time_text_embed.timestep_embedder.linear_1.weight",
      "time_text_embed.timestep_embedder.linear_2.weight", "modulation.1.weight",
      "norm_out.linear.weight"};
  const std::uint64_t width = profile.width, input = profile.timestep_dim;
  if (!width || width > 65536 / 4 || !input || input > 65536 || tensors.size() > 4096)
    return std::unexpected("image timestep source dimensions or inventory exceed component bounds");
  const std::array<std::uint64_t, 4> ins{input, width, width, width};
  const std::array<std::uint64_t, 4> outs{width, width, 4 * width, width};
  std::array<const model::QwenImageTensor*, 4> sources{};
  for (const auto& tensor : tensors) {
    if (tensor.name.empty() || tensor.name.size() > 256 || tensor.shape.size() > 4)
      return std::unexpected("image timestep source metadata exceeds mapping allowance");
    for (std::size_t i = 0; i < names.size(); ++i) {
      if (tensor.name != names[i]) continue;
      if (sources[i] || tensor.dtype != "BF16" || tensor.shape.size() != 2 ||
          tensor.shape[0] != outs[i] || tensor.shape[1] != ins[i])
        return std::unexpected("image timestep source role encoding or shape differs");
      sources[i] = &tensor;
    }
  }
  if (std::ranges::find(sources, nullptr) != sources.end())
    return std::unexpected("image timestep source role is missing");
  Stage stage{.name = "conditioning", .role = StageRole::kDenoiser, .artifact = 0};
  stage.stream.width = static_cast<std::uint32_t>(input);
  Layer layer;
  layer.components.reserve(4);
  for (std::size_t i = 0; i < names.size(); ++i)
    layer.components.push_back({.kind = "projection.spatial",
                                .version = 1,
                                .params = {{"in_channels", static_cast<std::int64_t>(ins[i])},
                                           {"out_channels", static_cast<std::int64_t>(outs[i])},
                                           {"silu", i < 2}},
                                .weights = {{.role = "weight", .tensor = sources[i]->name}}});
  stage.layers.push_back(std::move(layer));
  ModelSpec spec{.pipeline = PipelineKind::kImage, .stages = {std::move(stage)}};
  Canonicalize(spec);
  if (auto valid = Validate(spec); !valid) return std::unexpected(valid.error());
  return SpatialComposition{std::move(spec), {{0, {}}, {1, {}}, {2, {}}, {2, {}}}, {3, 4}};
}
namespace {
std::expected<std::uint16_t, std::string> AppendImageBlock(
    SpatialComposition& out, const model::QwenImageDenoiserProfile& p, std::uint32_t block,
    std::uint32_t text_rows, std::span<const model::QwenImageTensor> tensors, std::uint16_t stream,
    const std::array<std::uint16_t, 4>& condition, std::uint16_t position_port,
    std::uint16_t mask_port) {
  const std::uint64_t width = p.width, dim = p.head_dim;
  if (block >= p.blocks || !p.heads || !dim || dim > 512 || width != std::uint64_t{p.heads} * dim ||
      !p.mlp || p.mlp > 65536 || text_rows > 4096 || !std::isfinite(p.eps) || p.eps <= 0 ||
      p.eps > 1 || !std::isfinite(p.rope_theta) || p.rope_theta < 1 || p.rope_theta > 1e10 ||
      std::uint64_t{p.rope_axes[0]} + p.rope_axes[1] + p.rope_axes[2] != dim ||
      std::ranges::any_of(p.rope_axes, [](auto n) { return n < 2 || n % 2; }))
    return std::unexpected("image block profile exceeds the shared mathematical domain");
  // Validate all borrowed names/roles before copying the conditioning mapping.
  const auto prefix = "transformer_blocks." + std::to_string(block) + ".";
  constexpr std::array<std::string_view, 9> suffixes{
      "attn.to_q.weight",          "attn.to_k.weight",    "attn.to_v.weight",
      "attn.to_out.0.weight",      "attn.norm_q.weight",  "attn.norm_k.weight",
      "img_mlp.gate_layer.weight", "img_mlp.proj.weight", "img_mlp.out.weight"};
  if (tensors.size() > 4096) return std::unexpected("image block source inventory is unbounded");
  std::array<const model::QwenImageTensor*, 9> sources{};
  for (const auto& t : tensors) {
    if (t.name.empty() || t.name.size() > 256 || t.shape.empty() || t.shape.size() > 4)
      return std::unexpected("image block source descriptor exceeds mapping allowance");
    if (!t.name.starts_with(prefix)) continue;
    const auto suffix = std::string_view(t.name).substr(prefix.size());
    for (std::size_t i = 0; i < suffixes.size(); ++i) {
      if (suffix != suffixes[i]) continue;
      const bool norm = i == 4 || i == 5;
      const auto input = i == 8 ? std::uint64_t{p.mlp} : width;
      const auto output = i == 6 || i == 7 ? std::uint64_t{p.mlp} : width;
      if (sources[i] || t.dtype != "BF16" ||
          (norm ? t.shape.size() != 1 || t.shape[0] != dim
                : t.shape.size() != 2 || t.shape[0] != output || t.shape[1] != input))
        return std::unexpected("image block source encoding or role geometry differs");
      sources[i] = &t;
    }
  }
  if (std::ranges::find(sources, nullptr) != sources.end())
    return std::unexpected("image block source role is absent");
  auto& components = out.spec.stages[0].layers[0].components;
  if (components.size() > kMaxComponents - 17)
    return std::unexpected("image blocks exceed value-ID capacity");
  const auto add = [&](ComponentRef c, GraphLink link) {
    components.push_back(std::move(c));
    out.links.push_back(link);
    return static_cast<std::uint16_t>(components.size());
  };
  const auto projection = [&](std::size_t role, std::uint16_t input) {
    return add({.kind = "projection.spatial",
                .version = 1,
                .params = {{"in_channels", static_cast<std::int64_t>(sources[role]->shape[1])},
                           {"out_channels", static_cast<std::int64_t>(sources[role]->shape[0])}},
                .weights = {{.role = "weight", .tensor = sources[role]->name}}},
               {input, {}});
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
  const auto q = projection(0, normalized), k = projection(1, normalized),
             v = projection(2, normalized);
  const auto rotated = [&](std::uint16_t input, std::size_t role) {
    return add({.kind = "norm.rope_axes",
                .version = 1,
                .params = {{"heads", std::int64_t{p.heads}},
                           {"axis0", std::int64_t{p.rope_axes[0]}},
                           {"axis1", std::int64_t{p.rope_axes[1]}},
                           {"axis2", std::int64_t{p.rope_axes[2]}},
                           {"base", double{p.rope_theta}},
                           {"eps", double{p.eps}}},
                .weights = {{.role = "weight", .tensor = sources[role]->name}}},
               {.input = input, .auxiliary_port = position_port});
  };
  const auto rq = rotated(q, 4), rk = rotated(k, 5);
  const auto qk = add({.kind = "concat.spatial", .version = 1}, {rq, rk});
  const auto qkv = add({.kind = "concat.spatial", .version = 1}, {qk, v});
  const auto attention =
      add({.kind = "attention.spatial", .version = 1, .params = {{"heads", std::int64_t{p.heads}}}},
          {.input = qkv, .auxiliary_port = mask_port});
  const auto residual = gated(projection(3, attention), condition[1], stream);
  const auto mlp_input = adaptive(residual, condition[2]);
  const auto gate = projection(6, mlp_input), up = projection(7, mlp_input);
  const auto glu = add({.kind = "glu.spatial", .version = 1}, {gate, up});
  const auto result = gated(projection(8, glu), condition[3], residual);
  return result;
}
}  // namespace
std::expected<SpatialComposition, std::string> ImageTransformerBlockSpec(
    const model::QwenImageDenoiserProfile& p, std::uint32_t block, std::uint32_t text_rows,
    std::span<const model::QwenImageTensor> tensors) {
  auto out = ImageTimestepSpec(p, tensors);
  if (!out) return std::unexpected(out.error());
  auto& stage = out->spec.stages[0];
  stage.name = "conditioned";
  stage.stream.width = p.width;
  auto& components = stage.layers[0].components;
  components.reserve(28);
  out->links.reserve(28);
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
  auto result = AppendImageBlock(*out, p, block, text_rows, tensors, 0, condition, 1, 2);
  if (!result) return std::unexpected(result.error());
  out->exports = {*result, 4};
  Canonicalize(out->spec);
  if (auto valid = Validate(out->spec); !valid) return std::unexpected(valid.error());
  return out;
}

std::expected<std::uint64_t, std::string> ImageFlowSpecBound(
    const model::QwenImageDenoiserProfile& p, std::span<const model::QwenImageTensor> tensors) {
  constexpr std::uint64_t fixed_components = 19, per_block = 17;
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
std::expected<SpatialComposition, std::string> ImageFlowSpec(
    const model::QwenImageDenoiserProfile& p, std::uint32_t text_rows, std::uint32_t image_rows,
    std::span<const model::QwenImageTensor> tensors) {
  auto bound = ImageFlowSpecBound(p, tensors);
  if (!bound) return std::unexpected(bound.error());
  if (!text_rows || !image_rows || text_rows > 4096 || image_rows > 4096 - text_rows ||
      p.context != p.width || !p.in_channels || p.in_channels != p.out_channels ||
      p.in_channels > 65536)
    return std::unexpected(
        "image flow needs matching source context/stream and Euler channels with bounded rows");
  constexpr std::array<std::string_view, 5> names{
      "txt_in.text_norm.weight", "txt_in.in_layer.weight", "txt_in.out_layer.weight",
      "img_in.weight", "proj_out.weight"};
  std::array<const model::QwenImageTensor*, 5> sources{};
  for (const auto& t : tensors)
    for (std::size_t i = 0; i < names.size(); ++i)
      if (t.name == names[i]) {
        const std::uint64_t input = i == 3 ? p.in_channels : p.width;
        const std::uint64_t output = i == 4 ? p.out_channels : p.width;
        if (sources[i] || t.dtype != "BF16" ||
            (i == 0 ? t.shape.size() != 1 || t.shape[0] != p.context
                    : t.shape.size() != 2 || t.shape[0] != output || t.shape[1] != input))
          return std::unexpected("image flow boundary source role/encoding/geometry differs");
        sources[i] = &t;
      }
  if (std::ranges::find(sources, nullptr) != sources.end())
    return std::unexpected("image flow boundary source role is missing");
  auto out = ImageTimestepSpec(p, tensors);
  if (!out) return std::unexpected(out.error());
  auto& stage = out->spec.stages[0];
  stage.name = "flow";
  stage.stream.width = p.context;
  auto& components = stage.layers[0].components;
  components.reserve(19 + 17 * std::size_t{p.blocks});
  out->links.reserve(components.capacity());
  out->links[0].input_port = 1;
  const auto add = [&](ComponentRef c, GraphLink link) {
    components.push_back(std::move(c));
    out->links.push_back(link);
    return static_cast<std::uint16_t>(components.size());
  };
  const auto projection = [&](std::size_t role, std::uint16_t input, bool gelu = false) {
    return add({.kind = "projection.spatial",
                .version = 1,
                .params = {{"in_channels", static_cast<std::int64_t>(sources[role]->shape[1])},
                           {"out_channels", static_cast<std::int64_t>(sources[role]->shape[0])},
                           {"gelu", gelu}},
                .weights = {{.role = "weight", .tensor = sources[role]->name}}},
               {input, {}});
  };
  const auto text_norm = add({.kind = "norm.rms_spatial",
                              .version = 1,
                              .params = {{"channels", std::int64_t{p.context}},
                                         {"eps", double{p.eps}},
                                         {"norm_zero_centered", true}},
                              .weights = {{.role = "weight", .tensor = sources[0]->name}}},
                             {0, {}});
  const auto text = projection(2, projection(1, text_norm, true));
  const auto latent = add({.kind = "slice.spatial",
                           .version = 1,
                           .params = {{"start", std::int64_t{0}},
                                      {"rows", std::int64_t{image_rows}},
                                      {"input_rows", std::int64_t{image_rows}}}},
                          {.input_port = 0});
  const auto image = projection(3, latent);
  auto stream = add({.kind = "concat.spatial", .version = 1, .params = {{"axis", std::int64_t{1}}}},
                    {text, image});
  std::array<std::uint16_t, 4> condition{};
  for (std::size_t i = 0; i < condition.size(); ++i)
    condition[i] = add({.kind = "condition.rows",
                        .version = 1,
                        .params = {{"quarter", static_cast<std::int64_t>(i)},
                                   {"prefix_rows", std::int64_t{text_rows}}}},
                       {stream, 3});
  for (std::uint32_t block = 0; block < p.blocks; ++block) {
    auto result = AppendImageBlock(*out, p, block, text_rows, tensors, stream, condition, 2, 3);
    if (!result) return std::unexpected(result.error());
    stream = *result;
  }
  const auto image_out = add({.kind = "slice.spatial",
                              .version = 1,
                              .params = {{"start", std::int64_t{text_rows}},
                                         {"rows", std::int64_t{image_rows}},
                                         {"input_rows", std::int64_t{text_rows + image_rows}}}},
                             {stream, {}});
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
  const auto noise = projection(4, normalized);
  const auto next = add({.kind = "residual.scaled", .version = 1},
                        {.input = noise, .auxiliary_port = 4, .carry = latent});
  out->exports = {noise, next};
  Canonicalize(out->spec);
  if (auto valid = Validate(out->spec); !valid) return std::unexpected(valid.error());
  return out;
}

}  // namespace llmp::spec
