// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The Qwen-Image-2.1 pipeline adapter (M3; docs/plan.md, "Model graphs and
// state"; docs/experiments/qwen-image-native): the three components'
// hyperparameters, compiled in as profiles; the binding of each component's
// v0 artifact (one per component, joined by a composition, D-089) to the
// tensors it reads; and the host-side arithmetic the pipeline does outside
// its kernels, each written to reproduce the pinned diffusers 8b3c707e and
// transformers 5.17 code: the flow-matching schedule, the rotary tables of
// both transformers, the timestep's sinusoid, the latents' packing and the
// image's postprocessing.
//
// The model layer holds no vendor or kernel-module types; the kernels are
// kernels/image's and GGML's, and the phases run in the harness
// (benchmarks/qwen_image_exec.cc).

#ifndef LLMP_MODEL_QWEN_IMAGE_H_
#define LLMP_MODEL_QWEN_IMAGE_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace llmp::artifact {
class Artifact;
}

namespace llmp::model {

// ---------------------------------------------------------------- profiles

// Qwen3-VL-8B's text path (text_encoder/config.json's text_config).
struct QwenImageTextProfile {
  std::uint32_t layers = 36;
  std::uint32_t width = 4096;
  std::uint32_t heads = 32;
  std::uint32_t kv_heads = 8;
  std::uint32_t head_dim = 128;
  std::uint32_t ffn = 12288;
  std::uint32_t vocab = 151936;
  float rms_eps = 1e-6f;
  float rope_theta = 5000000.0f;
};

// QwenImage21Transformer2DModel (transformer/config.json).
struct QwenImageDenoiserProfile {
  std::uint32_t blocks = 32;
  std::uint32_t width = 4096;  // heads x head_dim
  std::uint32_t heads = 32;
  std::uint32_t head_dim = 128;
  std::uint32_t mlp = 12288;  // mlp_ratio 3
  std::uint32_t in_channels = 64;
  std::uint32_t out_channels = 64;
  std::uint32_t context = 4096;  // context_in_dim
  std::uint32_t timestep_dim = 256;
  std::array<std::uint32_t, 3> rope_axes = {16, 56, 56};
  float rope_theta = 10000.0f;
  float eps = 1e-6f;
};

// AutoencoderKLQwenImage21's decoder (vae/config.json).
struct QwenImageVaeProfile {
  std::uint32_t z_dim = 64;
  std::uint32_t base = 144;  // decoder_base_dim
  std::array<std::uint32_t, 5> dim_mult = {1, 2, 4, 8, 8};
  std::uint32_t res_blocks = 2;
  // temperal_downsample reversed: the decoder's up blocks.
  std::array<bool, 4> temporal_upsample = {true, true, true, false};
  std::uint32_t out_channels = 4;
  std::uint32_t spatial_factor = 16;
  std::array<float, 64> latents_mean{};
  std::array<float, 64> latents_std{};
};

// The flow-matching scheduler (scheduler/scheduler_config.json).
struct QwenImageSchedulerProfile {
  std::uint32_t train_timesteps = 1000;
  std::uint32_t base_image_seq_len = 256;
  std::uint32_t max_image_seq_len = 8192;
  // Python floats in the pipeline, so F64 here.
  double base_shift = 0.5;
  double max_shift = 0.9;
  double shift_terminal = 0.02;
};

struct QwenImageProfile {
  std::string_view name;
  QwenImageTextProfile text;
  QwenImageDenoiserProfile denoiser;
  QwenImageVaeProfile vae;
  QwenImageSchedulerProfile scheduler;
  // The composition's and its components' architectures.
  std::string_view pipeline_architecture;
  std::string_view text_architecture;
  std::string_view denoiser_architecture;
  std::string_view vae_architecture;
};

// Qwen/Qwen-Image-2.1 at 790c9263.
const QwenImageProfile& QwenImage21();

// ---------------------------------------------------------------- binding

// A tensor a component reads, as its artifact must hold it: the resource of
// that name (roles are the checkpoint's tensor names), plain family, this
// dtype and shape.
struct QwenImageTensor {
  std::string name;
  std::string_view dtype;  // "BF16" or "F32"
  std::vector<std::uint64_t> shape;
};

// Per text-encoder layer, in this order.
enum class TextTensor : std::uint8_t {
  kInputNorm,
  kQ,
  kK,
  kV,
  kO,
  kQNorm,
  kKNorm,
  kPostNorm,
  kGate,
  kUp,
  kDown,
  kCount
};
// Per denoiser block, in this order.
enum class BlockTensor : std::uint8_t {
  kQ,
  kK,
  kV,
  kOut,
  kQNorm,
  kKNorm,
  kGate,
  kProj,
  kMlpOut,
  kCount
};
// The denoiser's tensors outside its blocks, in this order.
enum class DenoiserGlobal : std::uint8_t {
  kImgIn,
  kModulation,
  kTime1,
  kTime2,
  kTxtNorm,
  kTxtIn,
  kTxtOut,
  kNormOut,
  kProjOut,
  kCount
};

// The component's tensor list: the text encoder's embedding table, then
// each layer's TextTensor list; the denoiser's globals, then each block's
// BlockTensor list; the VAE decoder's tensors in VaeDecoderPlan's order.
std::vector<QwenImageTensor> TextEncoderTensors(const QwenImageTextProfile& p);
std::vector<QwenImageTensor> DenoiserTensors(const QwenImageDenoiserProfile& p);
std::vector<QwenImageTensor> VaeDecoderTensors(const QwenImageVaeProfile& p);

// A resource as the binder sees it.
struct QwenImageResource {
  std::vector<std::string> roles;
  std::string_view family;  // "plain", "ggml" or "exl3"
  std::string_view dtype;
  std::vector<std::uint64_t> shape;
};

// The resource index for each tensor, in the list's order; refused, naming
// the tensor, if one is missing or not plain of its dtype and shape, or if
// `architecture` is not `want`.
std::expected<std::vector<std::uint32_t>, std::string> BindQwenImageComponent(
    std::span<const QwenImageTensor> tensors, std::string_view want, std::string_view architecture,
    std::span<const QwenImageResource> resources);
std::expected<std::vector<std::uint32_t>, std::string> BindQwenImageComponent(
    std::span<const QwenImageTensor> tensors, std::string_view want,
    const artifact::Artifact& artifact);

// ---------------------------------------------------------------- the VAE decoder

// One step of the decoder, over channels-first [C, H, W] activations. The
// decoder's structure (QwenImage21Decoder3d with is_residual, at one frame:
// every causal convolution a 2-D one, the temporal convolution of each
// upsample3d skipped on the first chunk, DupUp3D keeping its last frame).
struct VaeStep {
  enum class Kind : std::uint8_t {
    kConv3x3,    // out = conv3x3(in) + bias; tensors: weight, bias
    kConv1x1,    // out = conv1x1(in) + bias
    kNormSilu,   // out = silu(rms_norm(in)); tensor (weight): gamma
    kNorm,       // out = rms_norm(in) (the attention block's)
    kAdd,        // out = in + aux, out_channels channels
    kCopy,       // out = in (an up block's input, kept for its shortcut)
    kUpsample,   // out = nearest 2x (in)
    kAttention,  // out = single-head attention; in holds q, k, v ([3C, pixels])
    kAddDupUp,   // out += DupUp3D(aux) at the last frame; aux is [in_channels, H, W]
  };
  Kind kind = Kind::kConv3x3;
  std::uint8_t in = 0;
  std::uint8_t out = 0;
  std::uint8_t aux = 0;
  std::uint32_t in_channels = 0;
  std::uint32_t out_channels = 0;
  std::uint32_t height = 0;  // the input's (kAddDupUp: aux's)
  std::uint32_t width = 0;
  std::uint32_t factor_t = 1;  // kAddDupUp
  std::int32_t weight = -1;    // index into VaeDecoderTensors
  std::int32_t bias = -1;
};
// The plan's buffers, each [channels, pixels] up to the decoder's largest
// activation: x (the running activation, which holds the latent first and
// the decoded image last), h, t (the residual branch), s (a 1x1 shortcut),
// b (an up block's input) and q (the attention's q, k and v).
inline constexpr std::uint8_t kVaeX = 0;
inline constexpr std::uint8_t kVaeH = 1;
inline constexpr std::uint8_t kVaeT = 2;
inline constexpr std::uint8_t kVaeS = 3;
inline constexpr std::uint8_t kVaeB = 4;
inline constexpr std::uint8_t kVaeQ = 5;
inline constexpr std::uint8_t kVaeBuffers = 6;

// The decoder's steps for a latent of `height` x `width` (the image's /16),
// after post_quant_conv (the first step) and before the clamp; tensor
// indices into VaeDecoderTensors(p).
std::vector<VaeStep> VaeDecoderPlan(const QwenImageVaeProfile& p, std::uint32_t height,
                                    std::uint32_t width);

// ---------------------------------------------------------------- host arithmetic

// The scheduler's sigmas (steps + 1, the terminal 0 appended) and
// timesteps (steps), as QwenImage21Pipeline sets them for `image_tokens`
// latent tokens: sigmas = linspace(1, 1/steps, steps) as NumPy computes it,
// mu = calculate_shift(image_tokens, ...), the exponential time shift and
// the stretch to shift_terminal in F32, timesteps = sigmas * 1000.
struct QwenImageSchedule {
  double mu = 0;
  std::vector<float> sigmas;
  std::vector<float> timesteps;
};
std::expected<QwenImageSchedule, std::string> QwenImageSchedulerSigmas(
    const QwenImageSchedulerProfile& p, std::uint32_t steps, std::uint32_t image_tokens);

// The step's timestep as the denoiser sees it: t rounded to BF16
// (t.to(latents.dtype)), divided by 1000 in BF16, then its sinusoid for the
// two modulation rows [t, 0]: row-major BF16 bits [2, timestep_dim], cos
// then sin, as QwenImage21TemporalTimesteps computes them in F32.
std::vector<std::uint16_t> QwenImageTimestepSinusoid(const QwenImageDenoiserProfile& p, float t);

// The text encoder's rotary tables for positions 0..rows-1 (all three
// M-RoPE sections equal for text): BF16 bits [rows, head_dim] of cos and of
// sin, cat(freqs, freqs) of pos * inv_freq, as Qwen3VLTextRotaryEmbedding.
struct RotaryTables {
  std::vector<std::uint16_t> cos;
  std::vector<std::uint16_t> sin;
};
RotaryTables QwenImageTextRotary(const QwenImageTextProfile& p, std::uint32_t rows);

// The denoiser's complex rotary frequencies for a joint sequence of
// `text_rows` text tokens then a grid x grid target image, as
// QwenImage21Rope computes them: F32 [rows, head_dim / 2, 2] (cos, sin),
// the frame axis' 8 pairs, then height's 28, then width's 28.
std::vector<float> QwenImageDenoiserRotary(const QwenImageDenoiserProfile& p,
                                           std::uint32_t text_rows, std::uint32_t grid);

// BF16 bits of float values (round to nearest even), and back.
std::uint16_t ToBf16(float value);
float FromBf16(std::uint16_t bits);

// The VAE's decoded tensor (BF16 [channels, H, W], channels 3 or 4) to the
// image's interleaved 8-bit pixels, as VaeImageProcessor.postprocess and
// numpy_to_pil do: clamp(-1, 1) (the decoder's), (x * 0.5 + 0.5).clamp(0, 1)
// in BF16, then F32 * 255, rounded half to even.
std::vector<std::uint8_t> QwenImagePixels(std::span<const std::uint16_t> decoded,
                                          std::uint32_t channels, std::uint32_t height,
                                          std::uint32_t width);

}  // namespace llmp::model

#endif  // LLMP_MODEL_QWEN_IMAGE_H_
