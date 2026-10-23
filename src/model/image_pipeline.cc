// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "model/image_pipeline.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <optional>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "artifact/artifact.h"

namespace llmp::model {
namespace {

std::unexpected<std::string> Refused(std::string why) { return std::unexpected(std::move(why)); }

std::vector<std::uint64_t> Shape(std::initializer_list<std::uint64_t> dims) { return dims; }

// A pattern with its "{layer}" (once, if any) replaced by `index`.
std::string Named(std::string_view pattern, std::uint32_t index) {
  constexpr std::string_view kField = "{layer}";
  const auto at = pattern.find(kField);
  if (at == std::string_view::npos) return std::string(pattern);
  return std::string(pattern.substr(0, at)) + std::to_string(index) +
         std::string(pattern.substr(at + kField.size()));
}
std::string Joined(std::string_view prefix, std::string_view suffix) {
  return std::string(prefix) + std::string(suffix);
}

// The decoder's tensors and steps, built together.
class VaeBuilder {
 public:
  VaeBuilder(const ImageVaeProfile& p, const ImageVaeNames& n) : p_(p), n_(n) {}

  std::int32_t Tensor(std::string name, std::vector<std::uint64_t> shape) {
    tensors_.push_back({.name = std::move(name), .dtype = "F32", .shape = std::move(shape)});
    return static_cast<std::int32_t>(tensors_.size() - 1);
  }
  void Step(VaeStep step) { steps_.push_back(step); }

  void Conv(std::string_view prefix, std::uint8_t in, std::uint8_t out, std::uint32_t cin,
            std::uint32_t cout, std::uint32_t kernel, std::uint32_t h, std::uint32_t w) {
    const std::int32_t weight =
        Tensor(Joined(prefix, n_.weight), Shape({cout, cin, kernel, kernel}));
    const std::int32_t bias = Tensor(Joined(prefix, n_.bias), Shape({cout}));
    Step({.kind = kernel == 3 ? VaeStep::Kind::kConv3x3 : VaeStep::Kind::kConv1x1,
          .in = in,
          .out = out,
          .in_channels = cin,
          .out_channels = cout,
          .height = h,
          .width = w,
          .weight = weight,
          .bias = bias});
  }

  void NormSilu(std::string_view gamma, std::uint8_t in, std::uint8_t out, std::uint32_t c,
                std::uint32_t h, std::uint32_t w) {
    const std::int32_t g = Tensor(std::string(gamma), Shape({c, 1, 1, 1}));
    Step({.kind = VaeStep::Kind::kNormSilu,
          .in = in,
          .out = out,
          .in_channels = c,
          .out_channels = c,
          .height = h,
          .width = w,
          .weight = g});
  }

  // QwenImage21ResidualBlock on x, in place.
  void Residual(std::string_view prefix, std::uint32_t cin, std::uint32_t cout, std::uint32_t h,
                std::uint32_t w) {
    std::uint8_t shortcut = kVaeX;
    if (cin != cout) {
      Conv(Joined(prefix, n_.shortcut), kVaeX, kVaeS, cin, cout, 1, h, w);
      shortcut = kVaeS;
    }
    NormSilu(Joined(prefix, n_.norm1), kVaeX, kVaeH, cin, h, w);
    Conv(Joined(prefix, n_.conv1), kVaeH, kVaeT, cin, cout, 3, h, w);
    NormSilu(Joined(prefix, n_.norm2), kVaeT, kVaeH, cout, h, w);
    // NOLINTNEXTLINE(readability-suspicious-call-argument): buffers, then channels
    Conv(Joined(prefix, n_.conv2), kVaeH, kVaeT, cout, cout, 3, h, w);
    Step({.kind = VaeStep::Kind::kAdd,
          .in = kVaeT,
          .out = kVaeX,
          .aux = shortcut,
          .in_channels = cout,
          .out_channels = cout,
          .height = h,
          .width = w});
  }

  // QwenImage21AttentionBlock on x, in place.
  void Attention(std::string_view prefix, std::uint32_t c, std::uint32_t h, std::uint32_t w) {
    const std::int32_t g = Tensor(Joined(prefix, n_.attention_norm), Shape({c, 1, 1}));
    Step({.kind = VaeStep::Kind::kNorm,
          .in = kVaeX,
          .out = kVaeH,
          .in_channels = c,
          .out_channels = c,
          .height = h,
          .width = w,
          .weight = g});
    Conv(Joined(prefix, n_.attention_qkv), kVaeH, kVaeQ, c, 3 * c, 1, h, w);
    Step({.kind = VaeStep::Kind::kAttention,
          .in = kVaeQ,
          .out = kVaeH,
          .in_channels = c,
          .out_channels = c,
          .height = h,
          .width = w});
    Conv(Joined(prefix, n_.attention_proj), kVaeH, kVaeT, c, c, 1, h, w);
    Step({.kind = VaeStep::Kind::kAdd,
          .in = kVaeT,
          .out = kVaeX,
          .aux = kVaeX,
          .in_channels = c,
          .out_channels = c,
          .height = h,
          .width = w});
  }

  void Build(std::uint32_t height, std::uint32_t width) {
    const std::uint32_t z = p_.z_dim;
    Conv(n_.post_quant_conv, kVaeX, kVaeH, z, z, 1, height, width);
    // dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
    std::vector<std::uint32_t> dims = {p_.base * p_.dim_mult.back()};
    for (unsigned int it : std::views::reverse(p_.dim_mult)) {
      dims.push_back(p_.base * it);
    }
    std::uint32_t h = height;
    std::uint32_t w = width;
    Conv(n_.conv_in, kVaeH, kVaeX, z, dims[0], 3, h, w);
    Residual(Named(n_.mid_resnet, 0), dims[0], dims[0], h, w);
    Attention(n_.mid_attention, dims[0], h, w);
    Residual(Named(n_.mid_resnet, 1), dims[0], dims[0], h, w);
    for (std::uint32_t i = 0; i + 1 < dims.size(); ++i) {
      const std::uint32_t cin = dims[i];
      const std::uint32_t cout = dims[i + 1];
      const bool up = i != p_.dim_mult.size() - 1;
      const std::string prefix = Named(n_.up_block, i);
      if (up) {
        Step({.kind = VaeStep::Kind::kCopy,
              .in = kVaeX,
              .out = kVaeB,
              .in_channels = cin,
              .out_channels = cin,
              .height = h,
              .width = w});
      }
      std::uint32_t c = cin;
      for (std::uint32_t r = 0; r < p_.res_blocks + 1; ++r) {
        Residual(Joined(prefix, Named(n_.up_resnet, r)), c, cout, h, w);
        c = cout;
      }
      if (up) {
        // upsample2d or upsample3d: on the first chunk both are a nearest
        // 2x upsample and a 3x3 convolution (the temporal one is skipped).
        Step({.kind = VaeStep::Kind::kUpsample,
              .in = kVaeX,
              .out = kVaeH,
              .in_channels = cout,
              .out_channels = cout,
              .height = h,
              .width = w});
        // NOLINTNEXTLINE(readability-suspicious-call-argument): buffers, then channels
        Conv(Joined(prefix, n_.upsampler), kVaeH, kVaeX, cout, cout, 3, h * 2, w * 2);
        Step({.kind = VaeStep::Kind::kAddDupUp,
              .in = kVaeX,
              .out = kVaeX,
              .aux = kVaeB,
              .in_channels = cin,
              .out_channels = cout,
              .height = h,
              .width = w,
              .factor_t = p_.temporal_upsample[i] ? 2U : 1U});
        h *= 2;
        w *= 2;
      }
    }
    NormSilu(n_.norm_out, kVaeX, kVaeH, dims.back(), h, w);
    Conv(n_.conv_out, kVaeH, kVaeX, dims.back(), p_.out_channels, 3, h, w);
  }

  std::vector<ImageTensor> tensors_;
  std::vector<VaeStep> steps_;

 private:
  const ImageVaeProfile& p_;
  const ImageVaeNames& n_;
};

float Round32(double v) { return static_cast<float>(v); }

}  // namespace

std::vector<ImageTensor> TextEncoderTensors(const ImageTextProfile& p,
                                            const ImageTextNames& names) {
  std::vector<ImageTensor> out;
  const std::uint64_t w = p.width;
  const std::uint64_t q = std::uint64_t{p.heads} * p.head_dim;
  const std::uint64_t kv = std::uint64_t{p.kv_heads} * p.head_dim;
  out.push_back({std::string(names.embedding), "BF16", Shape({p.vocab, w})});
  // In TextTensor order.
  const std::array<std::vector<std::uint64_t>, 11> shapes{
      Shape({w}),        Shape({q, w}),       Shape({kv, w}),      Shape({kv, w}),
      Shape({w, q}),     Shape({p.head_dim}), Shape({p.head_dim}), Shape({w}),
      Shape({p.ffn, w}), Shape({p.ffn, w}),   Shape({w, p.ffn})};
  for (std::uint32_t l = 0; l < p.layers; ++l)
    for (std::size_t t = 0; t < shapes.size(); ++t)
      out.push_back({Named(names.layer[t], l), "BF16", shapes[t]});
  return out;
}

std::vector<ImageTensor> DenoiserTensors(const ImageDenoiserProfile& p,
                                         const ImageDenoiserNames& names) {
  const std::uint64_t w = p.width;
  // In DenoiserGlobal order.
  const std::array<std::vector<std::uint64_t>, 9> globals{Shape({w, p.in_channels}),
                                                          Shape({4 * w, w}),
                                                          Shape({w, p.timestep_dim}),
                                                          Shape({w, w}),
                                                          Shape({p.context}),
                                                          Shape({w, p.context}),
                                                          Shape({w, w}),
                                                          Shape({w, w}),
                                                          Shape({p.out_channels, w})};
  // In BlockTensor order.
  const std::array<std::vector<std::uint64_t>, 9> block{
      Shape({w, w}),     Shape({w, w}),       Shape({w, w}),
      Shape({w, w}),     Shape({p.head_dim}), Shape({p.head_dim}),
      Shape({p.mlp, w}), Shape({p.mlp, w}),   Shape({w, p.mlp})};
  std::vector<ImageTensor> out;
  for (std::size_t t = 0; t < globals.size(); ++t)
    out.push_back({std::string(names.globals[t]), "BF16", globals[t]});
  for (std::uint32_t b = 0; b < p.blocks; ++b)
    for (std::size_t t = 0; t < block.size(); ++t)
      out.push_back({Named(names.block[t], b), "BF16", block[t]});
  return out;
}

std::vector<ImageTensor> VaeDecoderTensors(const ImageVaeProfile& p, const ImageVaeNames& names) {
  VaeBuilder b(p, names);
  b.Build(1, 1);
  return std::move(b.tensors_);
}

std::vector<VaeStep> VaeDecoderPlan(const ImageVaeProfile& p, std::uint32_t height,
                                    std::uint32_t width) {
  // The steps name tensors by index; their names are not needed.
  const ImageVaeNames names{};
  VaeBuilder b(p, names);
  b.Build(height, width);
  return std::move(b.steps_);
}

std::expected<std::vector<std::uint32_t>, std::string> BindImageComponent(
    std::span<const ImageTensor> tensors, std::string_view want, std::string_view architecture,
    std::span<const ImageResource> resources) {
  if (architecture != want) {
    return Refused(std::format("the artifact's architecture is {}, not {}", architecture, want));
  }
  std::vector<std::uint32_t> out;
  out.reserve(tensors.size());
  for (const ImageTensor& t : tensors) {
    std::optional<std::uint32_t> found;
    for (std::uint32_t r = 0; r < resources.size() && !found; ++r) {
      if (std::ranges::find(resources[r].roles, t.name) != resources[r].roles.end()) {
        found = r;
      }
    }
    if (!found) {
      return Refused(std::format("{}: no resource of that role", t.name));
    }
    const ImageResource& r = resources[*found];
    const bool actual_supported =
        r.actual_format && r.part == t.part && r.part == format::PartRole::kData &&
        (r.actual_format->layout == format::Layout::kDense ||
         r.actual_format->layout == format::Layout::kGgmlBlocks) &&
        r.actual_format->stacked == 1 && format::Validate(*r.actual_format).has_value();
    if (r.actual_format) {
      if (!actual_supported) {
        return Refused(std::format("{}: source format or shape is not admitted", t.name));
      }
      std::vector<std::uint64_t> shape;
      shape.reserve(r.actual_format->shape.rank);
      for (std::size_t d = 0; d < r.actual_format->shape.rank; ++d)
        shape.push_back(r.actual_format->shape.ne[r.actual_format->shape.rank - 1 - d]);
      if (shape != t.shape) {
        return Refused(std::format("{}: source format or shape is not admitted", t.name));
      }
    } else if (r.family != "plain" || r.dtype != t.dtype || r.shape != t.shape ||
               r.part != t.part) {
      return Refused(
          std::format("{}: not a plain {} tensor of the profile's shape", t.name, t.dtype));
    }
    out.push_back(*found);
  }
  return out;
}

std::expected<std::vector<std::uint32_t>, std::string> BindImageComponent(
    std::span<const ImageTensor> tensors, std::string_view want,
    const artifact::Artifact& artifact) {
  if (artifact.format_version() != 1) {
    return Refused(std::format(
        "an image component reads version 1 prepared artifacts; artifact {} is format version {} "
        "(D-118: re-import its source pipeline with `llmp import`)",
        artifact.id(), artifact.format_version()));
  }
  if (!artifact.expert_arrays().empty() || !artifact.folds().empty() ||
      !artifact.relayouts().empty()) {
    return Refused("an image component cannot have expert arrays, folds, or relayouts");
  }
  std::vector<ImageResource> resources;
  resources.reserve(artifact.resources().size());
  for (const artifact::Resource& r : artifact.resources()) {
    const bool is_plain = r.format && r.part == format::PartRole::kData &&
                          r.format->layout == format::Layout::kDense &&
                          r.format->params == format::EncodingParams{} && r.format->stacked == 1;
    std::string_view dtype;
    if (r.format) {
      if (r.format->encoding == format::Encoding::kBF16) {
        dtype = "BF16";
      } else if (r.format->encoding == format::Encoding::kF16) {
        dtype = "F16";
      } else if (r.format->encoding == format::Encoding::kF32) {
        dtype = "F32";
      }
    }
    std::vector<std::uint64_t> shape;
    if (r.format && r.format->shape.rank > 0 && r.format->shape.rank <= r.format->shape.ne.size() &&
        format::Validate(*r.format)) {
      shape.reserve(r.format->shape.rank);
      for (std::size_t d = 0; d < r.format->shape.rank; ++d) {
        shape.push_back(r.format->shape.ne[r.format->shape.rank - 1 - d]);
      }
    }
    resources.push_back({.roles = r.roles,
                         .family = is_plain ? "plain" : "other",
                         .dtype = dtype,
                         .shape = std::move(shape),
                         .actual_format = r.format,
                         .part = r.part});
  }
  return BindImageComponent(tensors, want, artifact.model().architecture, resources);
}

// ---------------------------------------------------------------- host arithmetic

std::uint16_t ToBf16(float value) {
  auto u = std::bit_cast<std::uint32_t>(value);
  if ((u & 0x7fffffffU) > 0x7f800000U) {
    return static_cast<std::uint16_t>((u >> 16U) | 0x40U);
  }
  u += 0x7fffU + ((u >> 16U) & 1U);
  return static_cast<std::uint16_t>(u >> 16U);
}

float FromBf16(std::uint16_t bits) {
  return std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16U);
}

std::expected<ImageSchedule, std::string> ImageSchedulerSigmas(const ImageSchedulerProfile& p,
                                                               std::uint32_t steps,
                                                               std::uint32_t image_tokens) {
  if (!steps || !image_tokens || !std::isfinite(p.base_shift) || !std::isfinite(p.max_shift) ||
      p.max_image_seq_len <= p.base_image_seq_len || !std::isfinite(p.shift_terminal) ||
      p.shift_terminal <= 0 || p.shift_terminal >= 1 || !p.train_timesteps ||
      std::uint64_t{steps} + 1 > std::vector<float>().max_size())
    return Refused("scheduler geometry or source parameters are not representable");
  ImageSchedule s;
  // calculate_shift, in Python floats.
  const double m = (p.max_shift - p.base_shift) /
                   (static_cast<double>(p.max_image_seq_len) - p.base_image_seq_len);
  const double b = p.base_shift - (m * p.base_image_seq_len);
  s.mu = (image_tokens * m) + b;
  const float e = Round32(std::exp(s.mu));
  if (!std::isfinite(e) || e <= 0)
    return Refused("scheduler source time shift is not finite at this geometry");
  // A single Euler step uses the source start sigma and terminal zero. The
  // multi-step terminal stretch below has zero denominator for one sample;
  // it does not define a different one-step starting noise level.
  if (steps == 1) {
    s.sigmas = {1.F, 0.F};
    s.timesteps = {static_cast<float>(p.train_timesteps)};
    return s;
  }
  // np.linspace(1.0, 1 / steps, steps) in F64, then .astype(float32).
  const double start = 1.0;
  const double stop = 1.0 / steps;
  const double step = (stop - start) / (steps - 1);
  std::vector<float> t(steps);
  for (std::uint32_t i = 0; i < steps; ++i) {
    t[i] = Round32(i + 1 == steps ? stop : (i * step) + start);
  }
  // _time_shift_exponential with NumPy's weak Python scalars: every
  // operation in F32, exp(mu) rounded to F32 first.
  for (float& v : t) {
    const float inv = 1.0f / v;
    const float x = inv - 1.0f;
    v = e / (e + x);  // (1 / t - 1) ** 1.0 is x
  }
  // stretch_shift_to_terminal, in F32.
  const float scale = (1.0f - t.back()) / Round32(1.0 - p.shift_terminal);
  if (!std::isfinite(scale) || scale <= 0)
    return Refused("scheduler source terminal stretch is not finite");
  for (float& v : t) {
    v = 1.0f - ((1.0f - v) / scale);
  }
  s.sigmas = t;
  s.sigmas.push_back(0.0f);
  s.timesteps.resize(steps);
  for (std::uint32_t i = 0; i < steps; ++i) {
    s.timesteps[i] = t[i] * static_cast<float>(p.train_timesteps);
  }
  return s;
}

std::vector<std::uint16_t> ImageTimestepSinusoid(const ImageDenoiserProfile& p, float t) {
  // t.to(bf16), then / the divisor in BF16, then the [t, 0] rows.
  const float scaled = FromBf16(ToBf16(FromBf16(ToBf16(t)) / p.timestep_divisor));
  const std::uint32_t half = p.timestep_dim / 2;
  std::vector<std::uint16_t> out(2 * std::size_t{p.timestep_dim});
  const auto log_period = static_cast<float>(-std::log(p.timestep_period));
  for (std::uint32_t row = 0; row < 2; ++row) {
    const float ts = p.timestep_scale * (row == 0 ? scaled : 0.0f);
    for (std::uint32_t i = 0; i < half; ++i) {
      const float freq = Round32(std::exp(
          static_cast<double>((log_period * static_cast<float>(i)) / static_cast<float>(half))));
      const float arg = ts * freq;
      out[(row * p.timestep_dim) + i] = ToBf16(Round32(std::cos(static_cast<double>(arg))));
      out[(row * p.timestep_dim) + half + i] = ToBf16(Round32(std::sin(static_cast<double>(arg))));
    }
  }
  return out;
}

RotaryTables ImageTextRotary(const ImageTextProfile& p, std::uint32_t rows) {
  const std::uint32_t half = p.head_dim / 2;
  std::vector<float> inv(half);
  for (std::uint32_t i = 0; i < half; ++i) {
    const float exponent = static_cast<float>(2 * i) / static_cast<float>(p.head_dim);
    inv[i] =
        1.0f / Round32(std::pow(static_cast<double>(p.rope_theta), static_cast<double>(exponent)));
  }
  RotaryTables t;
  t.cos.resize(std::size_t{rows} * p.head_dim);
  t.sin.resize(std::size_t{rows} * p.head_dim);
  for (std::uint32_t r = 0; r < rows; ++r) {
    for (std::uint32_t i = 0; i < half; ++i) {
      const float f = static_cast<float>(r) * inv[i];
      const std::uint16_t c = ToBf16(Round32(std::cos(static_cast<double>(f))));
      const std::uint16_t s = ToBf16(Round32(std::sin(static_cast<double>(f))));
      t.cos[(std::size_t{r} * p.head_dim) + i] = c;
      t.cos[(std::size_t{r} * p.head_dim) + half + i] = c;
      t.sin[(std::size_t{r} * p.head_dim) + i] = s;
      t.sin[(std::size_t{r} * p.head_dim) + half + i] = s;
    }
  }
  return t;
}

std::array<std::int32_t, 3> ImageDenoiserPosition(std::uint32_t text_rows, std::uint32_t height,
                                                  std::uint32_t width, std::uint32_t row) {
  if (row < text_rows) {
    const auto position = static_cast<std::int32_t>(row);
    return {position, position, position};
  }
  const auto image = row - text_rows;
  return {static_cast<std::int32_t>(text_rows),
          static_cast<std::int32_t>(image / width) - static_cast<std::int32_t>(height - height / 2),
          static_cast<std::int32_t>(image % width) - static_cast<std::int32_t>(width - width / 2)};
}

std::vector<float> ImageDenoiserRotary(const ImageDenoiserProfile& p, std::uint32_t text_rows,
                                       std::uint32_t grid) {
  // rope_params per axis: 1 / theta ** (arange(0, dim, 2) / dim), F32.
  std::array<std::vector<float>, 3> inv;
  for (std::size_t a = 0; a < 3; ++a) {
    const std::uint32_t dim = p.rope_axes[a];
    for (std::uint32_t i = 0; i < dim; i += 2) {
      const float exponent = static_cast<float>(i) / static_cast<float>(dim);
      inv[a].push_back(1.0f / Round32(std::pow(static_cast<double>(p.rope_theta),
                                               static_cast<double>(exponent))));
    }
  }
  const std::uint32_t pairs = p.head_dim / 2;
  const std::size_t rows = text_rows + (std::size_t{grid} * grid);
  std::vector<float> out(rows * pairs * 2);
  const auto put = [&](std::size_t row, std::array<std::int64_t, 3> pos) {
    std::size_t k = 0;
    for (std::size_t a = 0; a < 3; ++a) {
      for (const float f : inv[a]) {
        const float angle = static_cast<float>(pos[a]) * f;
        out[((row * pairs) + k) * 2] = Round32(std::cos(static_cast<double>(angle)));
        out[(((row * pairs) + k) * 2) + 1] = Round32(std::sin(static_cast<double>(angle)));
        ++k;
      }
    }
  };
  for (std::uint32_t j = 0; j < text_rows; ++j) {
    put(j, {j, j, j});
  }
  // The image block: frame frozen at the text's end, a grid centred on 0.
  const auto lo = -static_cast<std::int64_t>(grid - (grid / 2));
  for (std::uint32_t y = 0; y < grid; ++y) {
    for (std::uint32_t x = 0; x < grid; ++x) {
      put(text_rows + (std::size_t{y} * grid) + x,
          {static_cast<std::int64_t>(text_rows), lo + y, lo + x});
    }
  }
  return out;
}

std::vector<std::uint8_t> ImagePixels(std::span<const std::uint16_t> decoded,
                                      std::uint32_t channels, std::uint32_t height,
                                      std::uint32_t width) {
  const std::size_t pixels = std::size_t{height} * width;
  std::vector<std::uint8_t> out(pixels * channels);
  for (std::uint32_t c = 0; c < channels; ++c) {
    for (std::size_t i = 0; i < pixels; ++i) {
      float v = FromBf16(decoded[(c * pixels) + i]);
      v = std::clamp(v, -1.0f, 1.0f);                           // the decoder's clamp
      v = FromBf16(ToBf16(FromBf16(ToBf16(v * 0.5f)) + 0.5f));  // x * 0.5 + 0.5 in BF16
      v = std::clamp(v, 0.0f, 1.0f);
      const float scaled = v * 255.0f;
      out[(i * channels) + c] = static_cast<std::uint8_t>(std::nearbyint(scaled));
    }
  }
  return out;
}

}  // namespace llmp::model
