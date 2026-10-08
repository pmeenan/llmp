// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "model/qwen_image.h"

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
#include "artifact/representation.h"

namespace llmp::model {
namespace {

std::unexpected<std::string> Refused(std::string why) { return std::unexpected(std::move(why)); }

QwenImageProfile MakeQwenImage21() {
  QwenImageProfile p;
  p.name = "Qwen/Qwen-Image-2.1@790c9263";
  p.pipeline_architecture = "QwenImage21Pipeline";
  p.text_architecture = "qwen3_vl";
  p.denoiser_architecture = "QwenImage21Transformer2DModel";
  p.vae_architecture = "AutoencoderKLQwenImage21";
  // vae/config.json's latents_mean and latents_std.
  p.vae.latents_mean = {
      0.5126f,  0.7721f,  -0.0631f, 1.3506f,  -0.7855f, -2.1025f, -0.3458f, 1.3722f,
      1.8873f,  -1.7177f, -0.651f,  0.2732f,  0.7562f,  -0.6163f, -1.0277f, 3.8363f,
      2.021f,   0.0472f,  0.932f,   2.0087f,  2.4954f,  -0.1391f, -1.4249f, 1.8464f,
      -0.5236f, 1.2826f,  3.7046f,  -1.3035f, 2.7286f,  -1.4518f, -1.9036f, -1.9955f,
      -0.0342f, -1.0265f, -0.7636f, 3.0555f,  0.0746f,  -3.0751f, -0.1076f, 1.7376f,
      -1.0914f, -1.9435f, -0.2784f, -1.368f,  0.4809f,  -0.4433f, 0.3764f,  0.5729f,
      -2.0595f, 1.096f,   -1.326f,  -2.0211f, -5.0179f, 0.5275f,  4.0162f,  1.8505f,
      0.3026f,  1.9373f,  1.4937f,  0.2632f,  0.5547f,  -1.7121f, -0.1562f, 0.0304f};
  p.vae.latents_std = {3.2001f, 3.2936f, 3.4321f, 3.0091f, 3.1061f, 4.0379f, 4.0705f, 3.791f,
                       3.0785f, 3.65f,   3.9308f, 3.0904f, 2.8778f, 3.7675f, 3.732f,  5.0756f,
                       3.2864f, 4.0397f, 3.1317f, 4.0443f, 2.9249f, 3.9454f, 3.0988f, 4.2489f,
                       3.4896f, 3.8513f, 3.9323f, 3.4719f, 3.7498f, 4.283f,  3.5694f, 4.2467f,
                       3.9037f, 3.2947f, 5.077f,  3.5075f, 3.27f,   3.4767f, 2.8063f, 5.1125f,
                       3.5327f, 4.7833f, 3.1286f, 4.1819f, 3.8527f, 3.8312f, 3.5605f, 4.3875f,
                       3.9624f, 4.0168f, 3.5643f, 4.055f,  5.5614f, 4.2963f, 4.408f,  3.4959f,
                       3.8747f, 3.7608f, 3.5735f, 3.149f,  3.7662f, 3.6746f, 3.4563f, 3.8161f};
  return p;
}

std::vector<std::uint64_t> Shape(std::initializer_list<std::uint64_t> dims) { return dims; }

// The decoder's tensors and steps, built together.
class VaeBuilder {
 public:
  explicit VaeBuilder(const QwenImageVaeProfile& p) : p_(p) {}

  std::int32_t Tensor(std::string name, std::vector<std::uint64_t> shape) {
    tensors_.push_back({.name = std::move(name), .dtype = "F32", .shape = std::move(shape)});
    return static_cast<std::int32_t>(tensors_.size() - 1);
  }
  void Step(VaeStep step) { steps_.push_back(step); }

  void Conv(std::string_view prefix, std::uint8_t in, std::uint8_t out, std::uint32_t cin,
            std::uint32_t cout, std::uint32_t kernel, std::uint32_t h, std::uint32_t w) {
    const std::int32_t weight =
        Tensor(std::format("{}.weight", prefix), Shape({cout, cin, kernel, kernel}));
    const std::int32_t bias = Tensor(std::format("{}.bias", prefix), Shape({cout}));
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
      Conv(std::format("{}.conv_shortcut", prefix), kVaeX, kVaeS, cin, cout, 1, h, w);
      shortcut = kVaeS;
    }
    NormSilu(std::format("{}.norm1.gamma", prefix), kVaeX, kVaeH, cin, h, w);
    Conv(std::format("{}.conv1", prefix), kVaeH, kVaeT, cin, cout, 3, h, w);
    NormSilu(std::format("{}.norm2.gamma", prefix), kVaeT, kVaeH, cout, h, w);
    // NOLINTNEXTLINE(readability-suspicious-call-argument): buffers, then channels
    Conv(std::format("{}.conv2", prefix), kVaeH, kVaeT, cout, cout, 3, h, w);
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
    const std::int32_t g = Tensor(std::format("{}.norm.gamma", prefix), Shape({c, 1, 1}));
    Step({.kind = VaeStep::Kind::kNorm,
          .in = kVaeX,
          .out = kVaeH,
          .in_channels = c,
          .out_channels = c,
          .height = h,
          .width = w,
          .weight = g});
    Conv(std::format("{}.to_qkv", prefix), kVaeH, kVaeQ, c, 3 * c, 1, h, w);
    Step({.kind = VaeStep::Kind::kAttention,
          .in = kVaeQ,
          .out = kVaeH,
          .in_channels = c,
          .out_channels = c,
          .height = h,
          .width = w});
    Conv(std::format("{}.proj", prefix), kVaeH, kVaeT, c, c, 1, h, w);
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
    Conv("post_quant_conv", kVaeX, kVaeH, z, z, 1, height, width);
    // dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
    std::vector<std::uint32_t> dims = {p_.base * p_.dim_mult.back()};
    for (unsigned int it : std::views::reverse(p_.dim_mult)) {
      dims.push_back(p_.base * it);
    }
    std::uint32_t h = height;
    std::uint32_t w = width;
    Conv("decoder.conv_in", kVaeH, kVaeX, z, dims[0], 3, h, w);
    Residual("decoder.mid_block.resnets.0", dims[0], dims[0], h, w);
    Attention("decoder.mid_block.attentions.0", dims[0], h, w);
    Residual("decoder.mid_block.resnets.1", dims[0], dims[0], h, w);
    for (std::uint32_t i = 0; i + 1 < dims.size(); ++i) {
      const std::uint32_t cin = dims[i];
      const std::uint32_t cout = dims[i + 1];
      const bool up = i != p_.dim_mult.size() - 1;
      const std::string prefix = std::format("decoder.up_blocks.{}", i);
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
        Residual(std::format("{}.resnets.{}", prefix, r), c, cout, h, w);
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
        Conv(std::format("{}.upsampler.resample.1", prefix), kVaeH, kVaeX, cout, cout, 3, h * 2,
             w * 2);
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
    NormSilu("decoder.norm_out.gamma", kVaeX, kVaeH, dims.back(), h, w);
    Conv("decoder.conv_out", kVaeH, kVaeX, dims.back(), p_.out_channels, 3, h, w);
  }

  std::vector<QwenImageTensor> tensors_;
  std::vector<VaeStep> steps_;

 private:
  const QwenImageVaeProfile& p_;
};

float Round32(double v) { return static_cast<float>(v); }

}  // namespace

const QwenImageProfile& QwenImage21() {
  static const QwenImageProfile profile = MakeQwenImage21();
  return profile;
}

std::vector<QwenImageTensor> TextEncoderTensors(const QwenImageTextProfile& p) {
  std::vector<QwenImageTensor> out;
  const std::uint64_t w = p.width;
  const std::uint64_t q = std::uint64_t{p.heads} * p.head_dim;
  const std::uint64_t kv = std::uint64_t{p.kv_heads} * p.head_dim;
  out.push_back({"model.language_model.embed_tokens.weight", "BF16", Shape({p.vocab, w})});
  for (std::uint32_t l = 0; l < p.layers; ++l) {
    const std::string pre = std::format("model.language_model.layers.{}.", l);
    out.push_back({pre + "input_layernorm.weight", "BF16", Shape({w})});
    out.push_back({pre + "self_attn.q_proj.weight", "BF16", Shape({q, w})});
    out.push_back({pre + "self_attn.k_proj.weight", "BF16", Shape({kv, w})});
    out.push_back({pre + "self_attn.v_proj.weight", "BF16", Shape({kv, w})});
    out.push_back({pre + "self_attn.o_proj.weight", "BF16", Shape({w, q})});
    out.push_back({pre + "self_attn.q_norm.weight", "BF16", Shape({p.head_dim})});
    out.push_back({pre + "self_attn.k_norm.weight", "BF16", Shape({p.head_dim})});
    out.push_back({pre + "post_attention_layernorm.weight", "BF16", Shape({w})});
    out.push_back({pre + "mlp.gate_proj.weight", "BF16", Shape({p.ffn, w})});
    out.push_back({pre + "mlp.up_proj.weight", "BF16", Shape({p.ffn, w})});
    out.push_back({pre + "mlp.down_proj.weight", "BF16", Shape({w, p.ffn})});
  }
  return out;
}

std::vector<QwenImageTensor> DenoiserTensors(const QwenImageDenoiserProfile& p) {
  const std::uint64_t w = p.width;
  std::vector<QwenImageTensor> out = {
      {"img_in.weight", "BF16", Shape({w, p.in_channels})},
      {"modulation.1.weight", "BF16", Shape({4 * w, w})},
      {"time_text_embed.timestep_embedder.linear_1.weight", "BF16", Shape({w, p.timestep_dim})},
      {"time_text_embed.timestep_embedder.linear_2.weight", "BF16", Shape({w, w})},
      {"txt_in.text_norm.weight", "BF16", Shape({p.context})},
      {"txt_in.in_layer.weight", "BF16", Shape({w, p.context})},
      {"txt_in.out_layer.weight", "BF16", Shape({w, w})},
      {"norm_out.linear.weight", "BF16", Shape({w, w})},
      {"proj_out.weight", "BF16", Shape({p.out_channels, w})},
  };
  for (std::uint32_t b = 0; b < p.blocks; ++b) {
    const std::string pre = std::format("transformer_blocks.{}.", b);
    out.push_back({pre + "attn.to_q.weight", "BF16", Shape({w, w})});
    out.push_back({pre + "attn.to_k.weight", "BF16", Shape({w, w})});
    out.push_back({pre + "attn.to_v.weight", "BF16", Shape({w, w})});
    out.push_back({pre + "attn.to_out.0.weight", "BF16", Shape({w, w})});
    out.push_back({pre + "attn.norm_q.weight", "BF16", Shape({p.head_dim})});
    out.push_back({pre + "attn.norm_k.weight", "BF16", Shape({p.head_dim})});
    out.push_back({pre + "img_mlp.gate_layer.weight", "BF16", Shape({p.mlp, w})});
    out.push_back({pre + "img_mlp.proj.weight", "BF16", Shape({p.mlp, w})});
    out.push_back({pre + "img_mlp.out.weight", "BF16", Shape({w, p.mlp})});
  }
  return out;
}

std::vector<QwenImageTensor> VaeDecoderTensors(const QwenImageVaeProfile& p) {
  VaeBuilder b(p);
  b.Build(1, 1);
  return std::move(b.tensors_);
}

std::vector<VaeStep> VaeDecoderPlan(const QwenImageVaeProfile& p, std::uint32_t height,
                                    std::uint32_t width) {
  VaeBuilder b(p);
  b.Build(height, width);
  return std::move(b.steps_);
}

std::expected<std::vector<std::uint32_t>, std::string> BindQwenImageComponent(
    std::span<const QwenImageTensor> tensors, std::string_view want, std::string_view architecture,
    std::span<const QwenImageResource> resources) {
  if (architecture != want) {
    return Refused(std::format("the artifact's architecture is {}, not {}", architecture, want));
  }
  std::vector<std::uint32_t> out;
  out.reserve(tensors.size());
  for (const QwenImageTensor& t : tensors) {
    std::optional<std::uint32_t> found;
    for (std::uint32_t r = 0; r < resources.size() && !found; ++r) {
      if (std::ranges::find(resources[r].roles, t.name) != resources[r].roles.end()) {
        found = r;
      }
    }
    if (!found) {
      return Refused(std::format("{}: no resource of that role", t.name));
    }
    const QwenImageResource& r = resources[*found];
    if (r.family != "plain" || r.dtype != t.dtype || r.shape != t.shape) {
      return Refused(
          std::format("{}: not a plain {} tensor of the profile's shape", t.name, t.dtype));
    }
    out.push_back(*found);
  }
  return out;
}

std::expected<std::vector<std::uint32_t>, std::string> BindQwenImageComponent(
    std::span<const QwenImageTensor> tensors, std::string_view want,
    const artifact::Artifact& artifact) {
  std::vector<QwenImageResource> resources;
  resources.reserve(artifact.resources().size());
  for (const artifact::Resource& r : artifact.resources()) {
    resources.push_back({.roles = r.roles,
                         .family = artifact::FamilyName(r.repr.family),
                         .dtype = r.repr.type,
                         .shape = r.repr.dims});
  }
  return BindQwenImageComponent(tensors, want, artifact.model().architecture, resources);
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

std::expected<QwenImageSchedule, std::string> QwenImageSchedulerSigmas(
    const QwenImageSchedulerProfile& p, std::uint32_t steps, std::uint32_t image_tokens) {
  if (steps < 2 || steps > 1000 || image_tokens == 0) {
    return Refused("2 to 1,000 steps over at least one token");
  }
  QwenImageSchedule s;
  // calculate_shift, in Python floats.
  const double m = (p.max_shift - p.base_shift) /
                   (static_cast<double>(p.max_image_seq_len) - p.base_image_seq_len);
  const double b = p.base_shift - (m * p.base_image_seq_len);
  s.mu = (image_tokens * m) + b;
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
  const float e = Round32(std::exp(s.mu));
  for (float& v : t) {
    const float inv = 1.0f / v;
    const float x = inv - 1.0f;
    v = e / (e + x);  // (1 / t - 1) ** 1.0 is x
  }
  // stretch_shift_to_terminal, in F32.
  const float scale = (1.0f - t.back()) / Round32(1.0 - p.shift_terminal);
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

std::vector<std::uint16_t> QwenImageTimestepSinusoid(const QwenImageDenoiserProfile& p, float t) {
  // t.to(bf16), then / 1000 in BF16, then the [t, 0] rows.
  const float scaled = FromBf16(ToBf16(FromBf16(ToBf16(t)) / 1000.0f));
  const std::uint32_t half = p.timestep_dim / 2;
  std::vector<std::uint16_t> out(2 * std::size_t{p.timestep_dim});
  const auto log_period = static_cast<float>(-std::log(10000.0));
  for (std::uint32_t row = 0; row < 2; ++row) {
    const float ts = 1000.0f * (row == 0 ? scaled : 0.0f);
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

RotaryTables QwenImageTextRotary(const QwenImageTextProfile& p, std::uint32_t rows) {
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

std::vector<float> QwenImageDenoiserRotary(const QwenImageDenoiserProfile& p,
                                           std::uint32_t text_rows, std::uint32_t grid) {
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

std::vector<std::uint8_t> QwenImagePixels(std::span<const std::uint16_t> decoded,
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
