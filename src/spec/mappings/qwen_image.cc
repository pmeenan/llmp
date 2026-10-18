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

constexpr std::array kComponents{
    ImageComponentRule{.role = kText,
                       .stage = "text_encoder",
                       .stage_role = StageRole::kTextEncoder,
                       .kinds = kTextKinds,
                       .architectures = kTextArchitectures},
    ImageComponentRule{.role = kFlow,
                       .stage = "denoiser",
                       .stage_role = StageRole::kDenoiser,
                       .kinds = kFlowKinds,
                       .architectures = kFlowArchitectures},
    ImageComponentRule{.role = kVae,
                       .stage = "vae",
                       .stage_role = StageRole::kVae,
                       .kinds = kVaeKinds,
                       .architectures = kVaeArchitectures},
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

// The pipeline's planning policy (planner/policy.h canonical text), every
// stage's, its serving recipe's rule: the graphs' layout round trips removed (layout_sinking); the
// QK norms and rotations in one pass (rms_norm_mul_rope_sections); the MLP in BF16 (bf16_glu); the
// attention over BF16 heads with a mask and rotation table made once a step (qk_attention); the
// modulations and gated residuals over their conditioning vectors (split_modulate,
// split_gated_residual), each residual joined with the modulation of its
// sum (split_gated_residual_modulate); the products a residual adds in
// BF16 (bf16_branch) and a sum only later residuals carry kept in BF16
// (bf16_residual): the BF16 activations the image path has always used, at
// the same quality against the reference (docs/experiments/
// m36-runtime-fusion); each product input converted to BF16 once
// (bf16_inputs) and the GB10's pinned cuBLASLt products first.
constexpr PolicyRule kPolicy{
    .text =
        "max_decode_rows 16\nenable bf16_branch\nenable bf16_glu\n"
        "enable bf16_inputs\nenable bf16_residual\nenable layout_sinking\n"
        "enable qk_attention\nenable rms_norm_mul_rope_sections\n"
        "enable split_gated_residual\nenable split_gated_residual_modulate\n"
        "enable split_modulate\nrank sm_121 llmp.gemm.bf16_lt 0\n"};

// Serving: the settings' names, and the policy above as its serving rule.
constexpr ServingRecipe kServing{
    .family = "Qwen-Image", .name = "QwenImage21", .serving_rule = &kPolicy};

// The qualified request's joint rows (a 25-token prompt and 1,024²), and
// its image rows.
constexpr std::array<std::uint64_t, 1> kAttentionRows{4121};
constexpr std::array<std::uint64_t, 1> kAttentionImageRows{4096};

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
      .attention_rows = kAttentionRows,
      .attention_image_rows = kAttentionImageRows,
      .tokenizer = "tokenizer/tokenizer.json",
      .prompt = "qwen_image_2_1"};
  return kMapping;
}

}  // namespace llmp::spec::mappings
