// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The image pipeline adapter (M3; docs/plan.md, "Model graphs and
// state"; docs/experiments/qwen-image-native): the three components'
// hyperparameters and tensor names as the model's data gives them (no
// family is compiled in); the binding of each component's artifact (one per
// component, joined by a composition, D-089) to the tensors it reads; and
// the host-side arithmetic the pipeline does outside
// its kernels, each written to reproduce the pinned diffusers 8b3c707e and
// transformers 5.17 code: the flow-matching schedule, the rotary tables of
// both transformers, the timestep's sinusoid, the latents' packing and the
// image's postprocessing.
//
// The model layer holds no vendor or kernel-module types; the operations run
// on the shared engine.

#ifndef LLMP_MODEL_IMAGE_PIPELINE_H_
#define LLMP_MODEL_IMAGE_PIPELINE_H_

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

// A family's hyperparameters, read from the data its model carries (the
// composition's stored spec, spec/image_family.h: its components' configs
// as its mapping reads them, spec/mappings/qwen_image.h). No profile is
// compiled in: a default-constructed one is all zeros.

// The text encoder's transformer (a decoder-only text path).
struct ImageTextProfile {
  std::uint32_t layers = 0;
  std::uint32_t width = 0;
  std::uint32_t heads = 0;
  std::uint32_t kv_heads = 0;
  std::uint32_t head_dim = 0;
  std::uint32_t ffn = 0;
  std::uint32_t vocab = 0;
  float rms_eps = 0.0f;
  float rope_theta = 0.0f;
};

// The flow transformer (the denoiser).
struct ImageDenoiserProfile {
  std::uint32_t blocks = 0;
  std::uint32_t width = 0;  // heads x head_dim
  std::uint32_t heads = 0;
  std::uint32_t head_dim = 0;
  std::uint32_t mlp = 0;
  std::uint32_t in_channels = 0;
  std::uint32_t out_channels = 0;
  std::uint32_t context = 0;  // the text features' width
  std::uint32_t timestep_dim = 0;
  std::array<std::uint32_t, 3> rope_axes{};
  float rope_theta = 0.0f;
  float eps = 0.0f;
  // The timestep's sinusoid: the pipeline divides the scheduler's timestep
  // by `timestep_divisor` (in BF16), the embedding scales it back by
  // `timestep_scale` over frequencies of period up to `timestep_period`.
  float timestep_divisor = 0.0f;
  float timestep_scale = 0.0f;
  double timestep_period = 0.0;
};

// The VAE's decoder.
struct ImageVaeProfile {
  std::uint32_t z_dim = 0;
  std::uint32_t base = 0;  // the decoder's base width
  std::vector<std::uint32_t> dim_mult{};
  std::uint32_t res_blocks = 0;
  // The encoder's temporal downsampling reversed: the decoder's up blocks'.
  std::vector<bool> temporal_upsample{};
  std::uint32_t out_channels = 0;
  std::uint32_t spatial_factor = 0;
  std::vector<float> latents_mean{};  // z_dim each
  std::vector<float> latents_std{};
};

// The flow-matching scheduler.
struct ImageSchedulerProfile {
  std::uint32_t train_timesteps = 0;
  std::uint32_t base_image_seq_len = 0;
  std::uint32_t max_image_seq_len = 0;
  // Python floats in the pipeline, so F64 here.
  double base_shift = 0.0;
  double max_shift = 0.0;
  double shift_terminal = 0.0;
};

struct ImageProfile {
  ImageTextProfile text;
  ImageDenoiserProfile denoiser;
  ImageVaeProfile vae;
  ImageSchedulerProfile scheduler;

  static ImageProfile Default() { return ImageProfile{}; }
};

// The checkpoint's tensor names, by structural role: its mapping's data
// (spec/mappings/qwen_image.h). A layer's or a block's pattern holds
// "{layer}" once, its index; a VAE name is a prefix (a block's, a
// convolution's) and a suffix.
struct ImageTextNames {
  std::string_view embedding{};
  std::array<std::string_view, 11> layer{};  // in TextTensor order
};
struct ImageDenoiserNames {
  std::array<std::string_view, 9> globals{};  // in DenoiserGlobal order
  std::array<std::string_view, 9> block{};    // in BlockTensor order, with {layer}
};
struct ImageVaeNames {
  std::string_view post_quant_conv{};
  std::string_view conv_in{};
  std::string_view mid_resnet{};     // with {layer}: 0, 1
  std::string_view mid_attention{};  // the mid block's attention
  std::string_view up_block{};       // with {layer}
  std::string_view up_resnet{};      // after an up block's prefix, with {layer}
  std::string_view upsampler{};      // after an up block's prefix: the 2-D resample convolution
  std::string_view norm_out{};       // a gamma's full name
  std::string_view conv_out{};
  // Within a residual block.
  std::string_view shortcut{};
  std::string_view norm1{};
  std::string_view conv1{};
  std::string_view norm2{};
  std::string_view conv2{};
  // Within the attention block.
  std::string_view attention_norm{};
  std::string_view attention_qkv{};
  std::string_view attention_proj{};
  // A convolution's parts.
  std::string_view weight{};
  std::string_view bias{};
};
struct ImageNames {
  ImageTextNames text;
  ImageDenoiserNames denoiser;
  ImageVaeNames vae;

  static ImageNames Default() { return ImageNames{}; }
};

// ---------------------------------------------------------------- binding

// A tensor a component reads, as its artifact must hold it: the resource of
// that name (roles are the checkpoint's tensor names), plain family, this
// dtype and shape.
struct ImageTensor {
  std::string name;
  std::string_view dtype;  // "BF16" or "F32"
  std::vector<std::uint64_t> shape;
};

// The components that compose an image pipeline.
enum class ImageComponent : std::uint8_t { kText = 0, kDenoiser = 1, kVae = 2, kCount = 3 };

// The text encoder's tensors outside its layers.
enum class TextGlobal : std::uint8_t { kEmbedding = 0, kCount = 1 };

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
  kImgIn = 0,
  kModulation = 1,
  kTime1 = 2,
  kTime2 = 3,
  kTxtNorm = 4,
  kTxtIn = 5,
  kTxtOut = 6,
  kNormOut = 7,
  kProjOut = 8,
  kCount = 9,

  // Semantic role aliases
  kImgInWeight = kImgIn,
  kImgInBias = kImgIn,
  kTimeLinear1Weight = kTime1,
  kTimeLinear1Bias = kTime1,
  kTimeLinear2Weight = kTime2,
  kTimeLinear2Bias = kTime2,
  kTxtNormWeight = kTxtNorm,
  kTxtInWeight = kTxtIn,
  kTxtInBias = kTxtIn,
  kNormOutWeight = kNormOut,
  kNormOutBias = kNormOut,
  kProjOutWeight = kProjOut,
  kProjOutBias = kProjOut,
};

// The component's tensor list: the text encoder's embedding table, then
// each layer's TextTensor list; the denoiser's globals, then each block's
// BlockTensor list; the VAE decoder's tensors in VaeDecoderPlan's order.
// Each by its names (its mapping's).
std::vector<ImageTensor> TextEncoderTensors(const ImageTextProfile& p, const ImageTextNames& names);
std::vector<ImageTensor> DenoiserTensors(const ImageDenoiserProfile& p,
                                         const ImageDenoiserNames& names);
std::vector<ImageTensor> VaeDecoderTensors(const ImageVaeProfile& p, const ImageVaeNames& names);

// A resource as the binder sees it.
struct ImageResource {
  std::vector<std::string> roles;
  std::string_view family;  // "plain", "ggml" or "exl3"
  std::string_view dtype;
  std::vector<std::uint64_t> shape;
};

// The resource index for each tensor, in the list's order; refused, naming
// the tensor, if one is missing or not plain of its dtype and shape, or if
// `architecture` is not `want`.
std::expected<std::vector<std::uint32_t>, std::string> BindImageComponent(
    std::span<const ImageTensor> tensors, std::string_view want, std::string_view architecture,
    std::span<const ImageResource> resources);
std::expected<std::vector<std::uint32_t>, std::string> BindImageComponent(
    std::span<const ImageTensor> tensors, std::string_view want,
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
std::vector<VaeStep> VaeDecoderPlan(const ImageVaeProfile& p, std::uint32_t height,
                                    std::uint32_t width);

struct VaePlan {
  static std::vector<VaeStep> Build(const ImageVaeProfile& p, std::uint32_t height,
                                    std::uint32_t width) {
    return VaeDecoderPlan(p, height, width);
  }
};

struct ImageLatentGrid {
  std::uint32_t height = 0;
  std::uint32_t width = 0;
};

// ---------------------------------------------------------------- host arithmetic

// The scheduler's sigmas (steps + 1, the terminal 0 appended) and
// timesteps (steps), as QwenImage21Pipeline sets them for `image_tokens`
// latent tokens: sigmas = linspace(1, 1/steps, steps) as NumPy computes it,
// mu = calculate_shift(image_tokens, ...), the exponential time shift and
// the stretch to shift_terminal in F32, timesteps = sigmas * train_timesteps.
// One step is the explicit Euler interval 1 -> 0; the terminal stretch requires
// multiple samples. The caller prepays vector storage for its requested count.
struct ImageSchedule {
  double mu = 0;
  std::vector<float> sigmas;
  std::vector<float> timesteps;
};
std::expected<ImageSchedule, std::string> ImageSchedulerSigmas(const ImageSchedulerProfile& p,
                                                               std::uint32_t steps,
                                                               std::uint32_t image_tokens);

// The step's timestep as the denoiser sees it: t rounded to BF16
// (t.to(latents.dtype)), divided by 1000 in BF16, then its sinusoid for the
// two modulation rows [t, 0]: row-major BF16 bits [2, timestep_dim], cos
// then sin, as QwenImage21TemporalTimesteps computes them in F32.
std::vector<std::uint16_t> ImageTimestepSinusoid(const ImageDenoiserProfile& p, float t);

// The text encoder's rotary tables for positions 0..rows-1 (all three
// M-RoPE sections equal for text): BF16 bits [rows, head_dim] of cos and of
// sin, cat(freqs, freqs) of pos * inv_freq, as Qwen3VLTextRotaryEmbedding.
struct RotaryTables {
  std::vector<std::uint16_t> cos;
  std::vector<std::uint16_t> sin;
};
RotaryTables ImageTextRotary(const ImageTextProfile& p, std::uint32_t rows);

// Source coordinates for a previously checked joint row. Callers validate
// nonzero width/height, row < text_rows + width*height, and signed32 coordinates.
// Odd axes use -ceil(extent/2), independently for height and width.
std::array<std::int32_t, 3> ImageDenoiserPosition(std::uint32_t text_rows, std::uint32_t height,
                                                  std::uint32_t width, std::uint32_t row);

// The denoiser's complex rotary frequencies for a joint sequence of
// `text_rows` text tokens then a grid x grid target image, as
// QwenImage21Rope computes them: F32 [rows, head_dim / 2, 2] (cos, sin),
// the frame axis' 8 pairs, then height's 28, then width's 28.
std::vector<float> ImageDenoiserRotary(const ImageDenoiserProfile& p, std::uint32_t text_rows,
                                       std::uint32_t grid);

// BF16 bits of float values (round to nearest even), and back.
std::uint16_t ToBf16(float value);
float FromBf16(std::uint16_t bits);

// The VAE's decoded tensor (BF16 [channels, H, W], channels 3 or 4) to the
// image's interleaved 8-bit pixels, as VaeImageProcessor.postprocess and
// numpy_to_pil do: clamp(-1, 1) (the decoder's), (x * 0.5 + 0.5).clamp(0, 1)
// in BF16, then F32 * 255, rounded half to even.
std::vector<std::uint8_t> ImagePixels(std::span<const std::uint16_t> decoded,
                                      std::uint32_t channels, std::uint32_t height,
                                      std::uint32_t width);

// ------------------------------------------------ backwards-compatible aliases

using QwenImageTextProfile = ImageTextProfile;
using QwenImageDenoiserProfile = ImageDenoiserProfile;
using QwenImageVaeProfile = ImageVaeProfile;
using QwenImageSchedulerProfile = ImageSchedulerProfile;
using QwenImageProfile = ImageProfile;
using QwenImageTextNames = ImageTextNames;
using QwenImageDenoiserNames = ImageDenoiserNames;
using QwenImageVaeNames = ImageVaeNames;
using QwenImageNames = ImageNames;
using QwenImageTensor = ImageTensor;
using QwenImageResource = ImageResource;
using QwenImageSchedule = ImageSchedule;

inline std::expected<std::vector<std::uint32_t>, std::string> BindQwenImageComponent(
    std::span<const ImageTensor> tensors, std::string_view want, std::string_view architecture,
    std::span<const ImageResource> resources) {
  return BindImageComponent(tensors, want, architecture, resources);
}

inline std::expected<std::vector<std::uint32_t>, std::string> BindQwenImageComponent(
    std::span<const ImageTensor> tensors, std::string_view want,
    const artifact::Artifact& artifact) {
  return BindImageComponent(tensors, want, artifact);
}

inline std::expected<ImageSchedule, std::string> QwenImageSchedulerSigmas(
    const ImageSchedulerProfile& p, std::uint32_t steps, std::uint32_t image_tokens) {
  return ImageSchedulerSigmas(p, steps, image_tokens);
}

inline std::vector<std::uint16_t> QwenImageTimestepSinusoid(const ImageDenoiserProfile& p,
                                                            float t) {
  return ImageTimestepSinusoid(p, t);
}

inline RotaryTables QwenImageTextRotary(const ImageTextProfile& p, std::uint32_t rows) {
  return ImageTextRotary(p, rows);
}

inline std::array<std::int32_t, 3> QwenImageDenoiserPosition(std::uint32_t text_rows,
                                                             std::uint32_t height,
                                                             std::uint32_t width,
                                                             std::uint32_t row) {
  return ImageDenoiserPosition(text_rows, height, width, row);
}

inline std::vector<float> QwenImageDenoiserRotary(const ImageDenoiserProfile& p,
                                                  std::uint32_t text_rows, std::uint32_t grid) {
  return ImageDenoiserRotary(p, text_rows, grid);
}

inline std::vector<std::uint8_t> QwenImagePixels(std::span<const std::uint16_t> decoded,
                                                 std::uint32_t channels, std::uint32_t height,
                                                 std::uint32_t width) {
  return ImagePixels(decoded, channels, height, width);
}

}  // namespace llmp::model

#endif  // LLMP_MODEL_IMAGE_PIPELINE_H_
