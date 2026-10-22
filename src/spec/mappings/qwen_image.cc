// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "spec/mappings/qwen_image.h"

#include <array>
#include <cstdint>
#include <string_view>

namespace llmp::spec::mappings {
namespace {

constexpr std::string_view kText = "text_encoder";
constexpr std::string_view kFlow = "transformer";
constexpr std::string_view kVae = "vae";
constexpr std::string_view kScheduler = "scheduler/scheduler_config.json";

constexpr std::array<std::string_view, 1> kTextKinds{kImageTextKind};
constexpr std::array<std::string_view, 2> kFlowKinds{kImageFlowKind, kImageScheduleKind};
constexpr std::array<std::string_view, 1> kVaeKinds{kImageVaeKind};
constexpr std::array<std::string_view, 1> kTextArchitectures{"qwen3_vl"};
constexpr std::array<std::string_view, 1> kFlowArchitectures{"QwenImage21Transformer2DModel"};
// The VAE's config names either class (diffusers renamed it).
constexpr std::array<std::string_view, 2> kVaeArchitectures{"AutoencoderKLQwenImage21",
                                                            "AutoencoderKLQwenImage"};

// What text-to-image generation does not run (each stage reads the
// tensors model/image_pipeline.h's lists give it, and no others):
// Qwen3-VL's vision tower, output head and final norm; the VAE's encoder,
// and its up blocks' temporal convolutions, which a one-frame decode never
// applies (the first frame skips them).
constexpr std::array<std::string_view, 3> kTextIgnored{"model.visual.", "lm_head.weight",
                                                       "model.language_model.norm.weight"};
constexpr std::array<std::string_view, 3> kVaeIgnored{
    "encoder.", "quant_conv.", "decoder.up_blocks.{layer}.upsampler.time_conv."};

constexpr std::array kComponents{
    ImageComponentRule{.role = kText,
                       .stage = "text_encoder",
                       .stage_role = StageRole::kTextEncoder,
                       .kinds = kTextKinds,
                       .architectures = kTextArchitectures,
                       .ignored_tensors = kTextIgnored},
    ImageComponentRule{.role = kFlow,
                       .stage = "denoiser",
                       .stage_role = StageRole::kDenoiser,
                       .kinds = kFlowKinds,
                       .architectures = kFlowArchitectures},
    ImageComponentRule{.role = kVae,
                       .stage = "vae",
                       .stage_role = StageRole::kVae,
                       .kinds = kVaeKinds,
                       .architectures = kVaeArchitectures,
                       .ignored_tensors = kVaeIgnored},
};

constexpr ImageParamRule Text(std::string_view param, std::string_view key) {
  return {.kind = kImageTextKind, .param = param, .role = kText, .key = key};
}
constexpr ImageParamRule Flow(std::string_view param, std::string_view key) {
  return {.kind = kImageFlowKind, .param = param, .role = kFlow, .key = key};
}
// What diffusers' QwenImage21 classes fix in code rather than config.json.
constexpr ImageParamRule FlowConstant(std::string_view param, double value) {
  return {.kind = kImageFlowKind, .param = param, .value = {.real = value}};
}
constexpr ImageParamRule Schedule(std::string_view param, std::string_view key) {
  return {.kind = kImageScheduleKind, .param = param, .file = kScheduler, .key = key};
}
constexpr ImageParamRule Vae(std::string_view param, std::string_view key, bool reverse = false) {
  return {.kind = kImageVaeKind, .param = param, .role = kVae, .key = key, .reverse = reverse};
}

constexpr std::array kParams{
    // text_encoder/config.json: Qwen3-VL's text path.
    Text("layers", "text_config.num_hidden_layers"),
    Text("width", "text_config.hidden_size"),
    Text("heads", "text_config.num_attention_heads"),
    Text("kv_heads", "text_config.num_key_value_heads"),
    Text("head_dim", "text_config.head_dim"),
    Text("ffn", "text_config.intermediate_size"),
    Text("vocab", "text_config.vocab_size"),
    Text("rms_eps", "text_config.rms_norm_eps"),
    Text("rope_theta", "text_config.rope_theta"),
    // transformer/config.json.
    Flow("blocks", "num_layers"),
    Flow("heads", "num_attention_heads"),
    Flow("head_dim", "attention_head_dim"),
    Flow("mlp_ratio", "mlp_ratio"),
    Flow("in_channels", "in_channels"),
    Flow("out_channels", "out_channels"),
    Flow("context", "context_in_dim"),
    Flow("rope_axes", "axes_dims_rope"),
    Flow("eps", "eps"),
    // QwenImage21TemporalTimesteps' channels and get_timestep_embedding's
    // max_period and scale; the pipeline's timestep / 1000; QwenImage21Rope's
    // theta.
    ImageParamRule{.kind = kImageFlowKind, .param = "timestep_dim", .value = {.integer = 256}},
    FlowConstant("timestep_period", 10000.0),
    FlowConstant("timestep_scale", 1000.0),
    FlowConstant("timestep_divisor", 1000.0),
    FlowConstant("rope_theta", 10000.0),
    // scheduler/scheduler_config.json.
    Schedule("train_timesteps", "num_train_timesteps"),
    Schedule("base_image_seq_len", "base_image_seq_len"),
    Schedule("max_image_seq_len", "max_image_seq_len"),
    Schedule("base_shift", "base_shift"),
    Schedule("max_shift", "max_shift"),
    Schedule("shift_terminal", "shift_terminal"),
    // vae/config.json: the decoder (the encoder's temporal downsampling,
    // reversed, its up blocks' temporal steps).
    Vae("z_dim", "z_dim"),
    Vae("base", "decoder_base_dim"),
    Vae("dim_mult", "dim_mult"),
    Vae("res_blocks", "num_res_blocks"),
    Vae("temporal_upsample", "temperal_downsample", true),
    Vae("out_channels", "out_channels"),
    Vae("spatial_factor", "scale_factor_spatial"),
    Vae("latents_mean", "latents_mean"),
    Vae("latents_std", "latents_std"),
};

constexpr std::array kRequirements{
    ImageRequirement{.role = kText, .key = "text_config.hidden_act", .value = {.text = "silu"}},
    ImageRequirement{.role = kText,
                     .key = "text_config.attention_bias",
                     .type = ValueType::kBool,
                     .value = {.flag = false}},
    ImageRequirement{
        .role = kFlow, .key = "patch_size", .type = ValueType::kInt, .value = {.integer = 1}},
    ImageRequirement{.role = kFlow,
                     .key = "causal_condition",
                     .type = ValueType::kBool,
                     .value = {.flag = true}},
    ImageRequirement{
        .role = kVae, .key = "is_residual", .type = ValueType::kBool, .value = {.flag = true}},
    ImageRequirement{
        .file = kScheduler, .key = "time_shift_type", .value = {.text = "exponential"}},
    ImageRequirement{.file = kScheduler,
                     .key = "use_dynamic_shifting",
                     .type = ValueType::kBool,
                     .value = {.flag = true}},
    ImageRequirement{.file = kScheduler,
                     .key = "invert_sigmas",
                     .type = ValueType::kBool,
                     .value = {.flag = false}},
    ImageRequirement{.file = kScheduler,
                     .key = "stochastic_sampling",
                     .type = ValueType::kBool,
                     .value = {.flag = false}},
};

// The measured concrete product preference. Structural graph rewrites use
// shared defaults; the admitted exact-geometry numerical evidence separately
// authorizes the existing BF16 intermediate and residual recipes.
constexpr PolicyRule kPolicy{.text =
                                 "max_decode_rows 16\n"
                                 "rank sm_121 llmp.gemm.bf16_lt 0\n"};

// Serving: the settings' names, and the policy above as its serving rule.
constexpr ServingRecipe kServing{
    .family = "Qwen-Image", .name = "QwenImage21", .serving_rule = &kPolicy};

constexpr std::array kHostInputs{
    ImageHostInput{0, 0, ImageInputValue::kPromptIds},
    ImageHostInput{0, 1, ImageInputValue::kTextPositions},
    ImageHostInput{0, 2, ImageInputValue::kTextCausalMask},
    ImageHostInput{1, 2, ImageInputValue::kTimestepSinusoid},
    ImageHostInput{1, 3, ImageInputValue::kJointPositions},
    ImageHostInput{1, 4, ImageInputValue::kJointMask},
    ImageHostInput{1, 5, ImageInputValue::kEulerDelta},
    ImageHostInput{2, 1, ImageInputValue::kLatentAffine},
};

}  // namespace

const ImageMapping& QwenImage21Mapping() {
  static constexpr ImageMapping kMapping{
      .architecture = "QwenImage21Pipeline",
      .version = 1,
      .components = kComponents,
      .params = kParams,
      .requirements = kRequirements,
      .names =
          {.text = {.embedding = "model.language_model.embed_tokens.weight",
                    .layer = {"model.language_model.layers.{layer}.input_layernorm.weight",
                              "model.language_model.layers.{layer}.self_attn.q_proj.weight",
                              "model.language_model.layers.{layer}.self_attn.k_proj.weight",
                              "model.language_model.layers.{layer}.self_attn.v_proj.weight",
                              "model.language_model.layers.{layer}.self_attn.o_proj.weight",
                              "model.language_model.layers.{layer}.self_attn.q_norm.weight",
                              "model.language_model.layers.{layer}.self_attn.k_norm.weight",
                              "model.language_model.layers.{layer}.post_attention_layernorm.weight",
                              "model.language_model.layers.{layer}.mlp.gate_proj.weight",
                              "model.language_model.layers.{layer}.mlp.up_proj.weight",
                              "model.language_model.layers.{layer}.mlp.down_proj.weight"}},
           .denoiser = {.globals = {"img_in.weight", "modulation.1.weight",
                                    "time_text_embed.timestep_embedder.linear_1.weight",
                                    "time_text_embed.timestep_embedder.linear_2.weight",
                                    "txt_in.text_norm.weight", "txt_in.in_layer.weight",
                                    "txt_in.out_layer.weight", "norm_out.linear.weight",
                                    "proj_out.weight"},
                        .block = {"transformer_blocks.{layer}.attn.to_q.weight",
                                  "transformer_blocks.{layer}.attn.to_k.weight",
                                  "transformer_blocks.{layer}.attn.to_v.weight",
                                  "transformer_blocks.{layer}.attn.to_out.0.weight",
                                  "transformer_blocks.{layer}.attn.norm_q.weight",
                                  "transformer_blocks.{layer}.attn.norm_k.weight",
                                  "transformer_blocks.{layer}.img_mlp.gate_layer.weight",
                                  "transformer_blocks.{layer}.img_mlp.proj.weight",
                                  "transformer_blocks.{layer}.img_mlp.out.weight"}},
           .vae = {.post_quant_conv = "post_quant_conv",
                   .conv_in = "decoder.conv_in",
                   .mid_resnet = "decoder.mid_block.resnets.{layer}",
                   .mid_attention = "decoder.mid_block.attentions.0",
                   .up_block = "decoder.up_blocks.{layer}",
                   .up_resnet = ".resnets.{layer}",
                   .upsampler = ".upsampler.resample.1",
                   .norm_out = "decoder.norm_out.gamma",
                   .conv_out = "decoder.conv_out",
                   .shortcut = ".conv_shortcut",
                   .norm1 = ".norm1.gamma",
                   .conv1 = ".conv1",
                   .norm2 = ".norm2.gamma",
                   .conv2 = ".conv2",
                   .attention_norm = ".norm.gamma",
                   .attention_qkv = ".to_qkv",
                   .attention_proj = ".proj",
                   .weight = ".weight",
                   .bias = ".bias"}},
      .serving = &kServing,
      .tokenizer = "processor/tokenizer.json",
      .prompt = "qwen_image_2_1",
      .host_inputs = kHostInputs};
  return kMapping;
}

}  // namespace llmp::spec::mappings
