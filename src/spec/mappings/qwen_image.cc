// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "spec/mappings/qwen_image.h"

#include <array>
#include <cstdint>
#include <string_view>

#include "spec/expr.h"
#include "spec/mappings/qwen_image_vae.h"
#include "spec/stage_recipe.h"

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
constexpr std::array<std::string_view, 1> kSafetensorsContainer{"safetensors"};
constexpr std::array<std::string_view, 2> kFlowContainers{"safetensors", "gguf"};
// The VAE's config names either class (diffusers renamed it).
constexpr std::array<std::string_view, 2> kVaeArchitectures{"AutoencoderKLQwenImage21",
                                                            "AutoencoderKLQwenImage"};

using ex::B;
using ex::E;
using ex::I;
using ex::R;
using ex::S;
using ex::U;
using K = ExprKind;
// A flow-transformer family parameter, read from the composition's family.
constexpr RecipeValue F(std::string_view name) { return ex::V(name); }
constexpr RecipeParam P(std::string_view name, const RecipeValue& value,
                        std::string_view scalar = {}) {
  return {.name = name, .value = value, .request_scalar = scalar};
}
constexpr RecipeWeight W(std::string_view role, std::string_view tensor) {
  return {.role = role, .tensor = tensor};
}
constexpr RecipeRef Link(std::string_view name) {
  return {.kind = RecipeRefKind::kNamedValue, .name = name};
}
constexpr RecipeRef H(std::uint16_t port) {
  return {.kind = RecipeRefKind::kHostPort, .port = port};
}
constexpr RecipeRef StageInput() { return {.kind = RecipeRefKind::kStageInput}; }

// Mapping-owned Qwen flow topology, its values the composition family's
// image.flow_transformer parameters (spec/expr.h names them "<kind>.<param>").
constexpr auto kFlowContext = F("image.flow_transformer.context");
constexpr auto kFlowHeads = F("image.flow_transformer.heads");
constexpr auto kFlowHeadDim = F("image.flow_transformer.head_dim");
constexpr auto kFlowWidth = E(K::kMultiply, kFlowHeads, kFlowHeadDim);
constexpr auto kFlowMlpRatio = F("image.flow_transformer.mlp_ratio");
constexpr auto kFlowMlp = E(K::kMultiply, kFlowWidth, kFlowMlpRatio);
constexpr auto kFlowEpsReal = F("image.flow_transformer.eps");
constexpr auto kFlowEps = U(K::kFloat32, kFlowEpsReal);
constexpr auto kFlowRopeTheta = F("image.flow_transformer.rope_theta");
constexpr auto kFlowRopeAxes = F("image.flow_transformer.rope_axes");
constexpr auto kFlowTimestepDim = F("image.flow_transformer.timestep_dim");
constexpr auto kFlowInChannels = F("image.flow_transformer.in_channels");
constexpr auto kFlowOutChannels = F("image.flow_transformer.out_channels");
constexpr auto kFlowBlocks = F("image.flow_transformer.blocks");
constexpr auto kFour = I(4);
constexpr auto kScaleOne = R(1.0);
constexpr auto kScaleRoot = U(K::kSqrt, kFlowHeadDim);
constexpr auto kAttentionScale = E(K::kRealDivide, kScaleOne, kScaleRoot);

constexpr auto kFourWidth = E(K::kMultiply, kFour, kFlowWidth);
const std::array kT0P{P("in_channels", kFlowTimestepDim), P("out_channels", kFlowWidth),
                      P("silu", B(true))};
const std::array kT1P{P("in_channels", kFlowWidth), P("out_channels", kFlowWidth),
                      P("silu", B(true))};
const std::array kT2P{P("in_channels", kFlowWidth), P("out_channels", kFourWidth),
                      P("silu", B(false))};
const std::array kT3P{P("in_channels", kFlowWidth), P("out_channels", kFlowWidth),
                      P("silu", B(false))};
const std::array kT0W{W("weight", "time_text_embed.timestep_embedder.linear_1.weight")};
const std::array kT1W{W("weight", "time_text_embed.timestep_embedder.linear_2.weight")};
const std::array kT2W{W("weight", "modulation.1.weight")};
const std::array kT3W{W("weight", "norm_out.linear.weight")};
const RecipeNode kT0{.name = "time.0",
                     .component_kind = "projection.spatial",
                     .params = kT0P,
                     .weights = kT0W,
                     .input = H(1)};
const RecipeNode kT1{.name = "time.1",
                     .component_kind = "projection.spatial",
                     .params = kT1P,
                     .weights = kT1W,
                     .input = Link("time.0")};
const RecipeNode kT2{.name = "time.2",
                     .component_kind = "projection.spatial",
                     .params = kT2P,
                     .weights = kT2W,
                     .input = Link("time.1")};
const RecipeNode kT3{.name = "time.3",
                     .component_kind = "projection.spatial",
                     .params = kT3P,
                     .weights = kT3W,
                     .input = Link("time.1")};

const std::array kTextNormP{P("channels", kFlowContext), P("eps", kFlowEps),
                            P("norm_zero_centered", B(true))};
const std::array kTextNormW{W("weight", "txt_in.text_norm.weight")};
const RecipeNode kTextNorm{.name = "text.norm",
                           .component_kind = "norm.rms",
                           .params = kTextNormP,
                           .weights = kTextNormW,
                           .input = StageInput()};
const std::array kTextInP{P("in_channels", kFlowContext), P("out_channels", kFlowWidth),
                          P("gelu", B(true))};
const std::array kTextInW{W("weight", "txt_in.in_layer.weight")};
const RecipeNode kTextIn{.name = "text.in",
                         .component_kind = "projection.spatial",
                         .params = kTextInP,
                         .weights = kTextInW,
                         .input = Link("text.norm")};
const std::array kTextOutP{P("in_channels", kFlowWidth), P("out_channels", kFlowWidth),
                           P("gelu", B(false))};
const std::array kTextOutW{W("weight", "txt_in.out_layer.weight")};
const RecipeNode kTextOut{.name = "text.out",
                          .component_kind = "projection.spatial",
                          .params = kTextOutP,
                          .weights = kTextOutW,
                          .input = Link("text.in")};
const std::array kLatentP{P("start", I(0)), P("rows", I(1), "latent_rows"),
                          P("input_rows", I(1), "latent_rows")};
const RecipeNode kLatent{
    .name = "latent", .component_kind = "slice.spatial", .params = kLatentP, .input = H(0)};
const std::array kImageInP{P("in_channels", kFlowInChannels), P("out_channels", kFlowWidth),
                           P("gelu", B(false))};
const std::array kImageInW{W("weight", "img_in.weight")};
const RecipeNode kImageIn{.name = "image.in",
                          .component_kind = "projection.spatial",
                          .params = kImageInP,
                          .weights = kImageInW,
                          .input = Link("latent")};
const std::array kConcatP{P("axis", I(1))};
const RecipeNode kConcat{.name = "joint",
                         .component_kind = "concat.spatial",
                         .params = kConcatP,
                         .input = Link("text.out"),
                         .auxiliary = Link("image.in"),
                         .has_auxiliary = true};
const std::array kConditionP{P("quarter", I(0)), P("prefix_rows", I(1), "text_rows")};
const std::array kCondition1P{P("quarter", I(1)), P("prefix_rows", I(1), "text_rows")};
const std::array kCondition2P{P("quarter", I(2)), P("prefix_rows", I(1), "text_rows")};
const std::array kCondition3P{P("quarter", I(3)), P("prefix_rows", I(1), "text_rows")};
const RecipeNode kCondition0{.name = "condition.0",
                             .component_kind = "condition.rows",
                             .params = kConditionP,
                             .input = Link("joint"),
                             .auxiliary = Link("time.2"),
                             .has_auxiliary = true};
const RecipeNode kCondition1{.name = "condition.1",
                             .component_kind = "condition.rows",
                             .params = kCondition1P,
                             .input = Link("joint"),
                             .auxiliary = Link("time.2"),
                             .has_auxiliary = true};
const RecipeNode kCondition2{.name = "condition.2",
                             .component_kind = "condition.rows",
                             .params = kCondition2P,
                             .input = Link("joint"),
                             .auxiliary = Link("time.2"),
                             .has_auxiliary = true};
const RecipeNode kCondition3{.name = "condition.3",
                             .component_kind = "condition.rows",
                             .params = kCondition3P,
                             .input = Link("joint"),
                             .auxiliary = Link("time.2"),
                             .has_auxiliary = true};

constexpr auto kRopeAxis0 = ex::At(kFlowRopeAxes, 0);
constexpr auto kRopeAxis1 = ex::At(kFlowRopeAxes, 1);
constexpr auto kRopeAxis2 = ex::At(kFlowRopeAxes, 2);
const std::array kAdaptiveP{P("side", S("scale")), P("eps", kFlowEps)};
const std::array kAttentionP{P("eps", kFlowEps),           P("heads", kFlowHeads),
                             P("key_dim", kFlowHeadDim),   P("kv_heads", kFlowHeads),
                             P("placement", S("bare")),    P("qk_norm", B(true)),
                             P("rope_axis0", kRopeAxis0),  P("rope_axis1", kRopeAxis1),
                             P("rope_axis2", kRopeAxis2),  P("rope_base", kFlowRopeTheta),
                             P("rope_dims", kFlowHeadDim), P("rope_scale", R(1.0)),
                             P("rotation", S("normal")),   P("scale", kAttentionScale),
                             P("scale_scores", B(true)),   P("stateless", B(true)),
                             P("value_dim", kFlowHeadDim), P("window", I(0))};
const std::array kAttentionW{W("k", "transformer_blocks.{block}.attn.to_k.weight"),
                             W("k_norm", "transformer_blocks.{block}.attn.norm_k.weight"),
                             W("output", "transformer_blocks.{block}.attn.to_out.0.weight"),
                             W("q", "transformer_blocks.{block}.attn.to_q.weight"),
                             W("q_norm", "transformer_blocks.{block}.attn.norm_q.weight"),
                             W("v", "transformer_blocks.{block}.attn.to_v.weight")};
const std::array kGatedP{P("side", S("gate"))};
const std::array kMlpP{P("activation", S("silu")), P("eps", kFlowEps), P("hidden", kFlowMlp),
                       P("placement", S("bare"))};
const std::array kMlpW{W("down", "transformer_blocks.{block}.img_mlp.out.weight"),
                       W("gate", "transformer_blocks.{block}.img_mlp.gate_layer.weight"),
                       W("up", "transformer_blocks.{block}.img_mlp.proj.weight")};
const RecipeNode kBlock0{.name = "adaptive.0.{block}",
                         .component_kind = "norm.adaptive",
                         .params = kAdaptiveP,
                         .input = Link("stream.{block}"),
                         .auxiliary = Link("condition.0"),
                         .has_auxiliary = true};
const RecipeNode kBlock1{.name = "attention.{block}",
                         .component_kind = "attention.dense",
                         .params = kAttentionP,
                         .weights = kAttentionW,
                         .input = Link("adaptive.0.{block}"),
                         .auxiliary = H(2),
                         .carry = H(3),
                         .has_auxiliary = true,
                         .has_carry = true,
                         .group = RecipeGroupKind::kLayer,
                         .group_index = ex::Idx("block")};
const RecipeNode kBlock2{.name = "residual.0.{block}",
                         .component_kind = "residual.gated",
                         .params = kGatedP,
                         .input = Link("attention.{block}"),
                         .auxiliary = Link("condition.1"),
                         .carry = Link("stream.{block}"),
                         .has_auxiliary = true,
                         .has_carry = true};
const RecipeNode kBlock3{.name = "adaptive.1.{block}",
                         .component_kind = "norm.adaptive",
                         .params = kAdaptiveP,
                         .input = Link("residual.0.{block}"),
                         .auxiliary = Link("condition.2"),
                         .has_auxiliary = true};
const RecipeNode kBlock4{.name = "mlp.{block}",
                         .component_kind = "ffn.glu",
                         .params = kMlpP,
                         .weights = kMlpW,
                         .input = Link("adaptive.1.{block}"),
                         .group = RecipeGroupKind::kLayer,
                         .group_index = ex::Idx("block")};
const RecipeNode kBlock5{.name = "stream.next.{block}",
                         .component_kind = "residual.gated",
                         .params = kGatedP,
                         .input = Link("mlp.{block}"),
                         .auxiliary = Link("condition.3"),
                         .carry = Link("residual.0.{block}"),
                         .has_auxiliary = true,
                         .has_carry = true};
const std::array kBlockSteps{RecipeStep{.kind = RecipeStep::Kind::kNode, .node = &kBlock0},
                             RecipeStep{.kind = RecipeStep::Kind::kNode, .node = &kBlock1},
                             RecipeStep{.kind = RecipeStep::Kind::kNode, .node = &kBlock2},
                             RecipeStep{.kind = RecipeStep::Kind::kNode, .node = &kBlock3},
                             RecipeStep{.kind = RecipeStep::Kind::kNode, .node = &kBlock4},
                             RecipeStep{.kind = RecipeStep::Kind::kNode, .node = &kBlock5}};
const RecipeRepeat kFlowRepeat{.index = "block",
                               .count = kFlowBlocks,
                               .body = kBlockSteps,
                               .input = Link("joint"),
                               .input_name = "stream.{block}",
                               .body_yield = "stream.next.{block}",
                               .yield = "stream.blocks"};

const std::array kSliceOutP{P("start", I(1), "text_rows"), P("rows", I(1), "latent_rows"),
                            P("input_rows", I(2), "joint_rows")};
const RecipeNode kSliceOut{.name = "image.out",
                           .component_kind = "slice.spatial",
                           .params = kSliceOutP,
                           .input = Link("stream.blocks")};
const std::array kScaleP{P("quarter", I(0)), P("quarters", I(1)), P("prefix_rows", I(0))};
const RecipeNode kScale{.name = "image.scale",
                        .component_kind = "condition.rows",
                        .params = kScaleP,
                        .input = Link("image.out"),
                        .auxiliary = Link("time.3"),
                        .has_auxiliary = true};
const RecipeNode kNormOut{.name = "image.norm",
                          .component_kind = "norm.adaptive",
                          .params = kAdaptiveP,
                          .input = Link("image.out"),
                          .auxiliary = Link("image.scale"),
                          .has_auxiliary = true};
const std::array kProjectOutP{P("in_channels", kFlowWidth), P("out_channels", kFlowOutChannels),
                              P("gelu", B(false))};
const std::array kProjectOutW{W("weight", "proj_out.weight")};
const RecipeNode kProjectOut{.name = "noise",
                             .component_kind = "projection.spatial",
                             .params = kProjectOutP,
                             .weights = kProjectOutW,
                             .input = Link("image.norm")};
const RecipeNode kEuler{.name = "euler",
                        .component_kind = "residual.scaled",
                        .input = Link("noise"),
                        .auxiliary = H(4),
                        .carry = Link("latent"),
                        .has_auxiliary = true,
                        .has_carry = true};
const std::array kFlowSteps{RecipeStep{.kind = RecipeStep::Kind::kNode, .node = &kT0},
                            RecipeStep{.kind = RecipeStep::Kind::kNode, .node = &kT1},
                            RecipeStep{.kind = RecipeStep::Kind::kNode, .node = &kT2},
                            RecipeStep{.kind = RecipeStep::Kind::kNode, .node = &kT3},
                            RecipeStep{.kind = RecipeStep::Kind::kNode, .node = &kTextNorm},
                            RecipeStep{.kind = RecipeStep::Kind::kNode, .node = &kTextIn},
                            RecipeStep{.kind = RecipeStep::Kind::kNode, .node = &kTextOut},
                            RecipeStep{.kind = RecipeStep::Kind::kNode, .node = &kLatent},
                            RecipeStep{.kind = RecipeStep::Kind::kNode, .node = &kImageIn},
                            RecipeStep{.kind = RecipeStep::Kind::kNode, .node = &kConcat},
                            RecipeStep{.kind = RecipeStep::Kind::kNode, .node = &kCondition0},
                            RecipeStep{.kind = RecipeStep::Kind::kNode, .node = &kCondition1},
                            RecipeStep{.kind = RecipeStep::Kind::kNode, .node = &kCondition2},
                            RecipeStep{.kind = RecipeStep::Kind::kNode, .node = &kCondition3},
                            RecipeStep{.kind = RecipeStep::Kind::kRepeat, .repeat = &kFlowRepeat},
                            RecipeStep{.kind = RecipeStep::Kind::kNode, .node = &kSliceOut},
                            RecipeStep{.kind = RecipeStep::Kind::kNode, .node = &kScale},
                            RecipeStep{.kind = RecipeStep::Kind::kNode, .node = &kNormOut},
                            RecipeStep{.kind = RecipeStep::Kind::kNode, .node = &kProjectOut},
                            RecipeStep{.kind = RecipeStep::Kind::kNode, .node = &kEuler}};
const std::array kFlowExports{Link("noise"), Link("euler")};
const RecipeSpatialOperand kFlowPrimary{
    .shape = {.width = I(1), .height = I(1), .channels = kFlowContext, .images = I(1)}};
const std::array kFlowHostPorts{
    RecipeSpatialOperand{
        .shape = {.width = I(1), .height = I(1), .channels = kFlowInChannels, .images = I(1)}},
    RecipeSpatialOperand{
        .shape = {.width = I(1), .height = I(2), .channels = kFlowTimestepDim, .images = I(1)}},
    RecipeSpatialOperand{.shape = {.width = I(2), .height = I(3), .channels = I(1), .images = I(1)},
                         .encoding = format::Encoding::kI32},
    RecipeSpatialOperand{
        .shape = {.width = I(2), .height = I(2), .channels = I(1), .images = I(1)}},
    RecipeSpatialOperand{
        .shape = {.width = I(1), .height = I(1), .channels = I(1), .images = I(1)}}};
const RecipeSpatialShape kFlowOutput{
    .width = I(1), .height = I(1), .channels = kFlowInChannels, .images = I(1)};
const std::array kFlowOutputShapes{kFlowOutput, kFlowOutput};
const StageRecipe kFlowRecipe{.name = "flow",
                              .role = StageRole::kDenoiser,
                              .stream_width = kFlowWidth,
                              .primary_input = kFlowPrimary,
                              .host_ports = kFlowHostPorts,
                              .steps = kFlowSteps,
                              .exports = kFlowExports,
                              .expected_outputs = kFlowOutputShapes};

// What text-to-image generation does not run (each stage reads the
// tensors model/image_pipeline.h's lists give it, and no others):
// Qwen3-VL's vision tower, output head and final norm. (The VAE's, its
// configured mapping's: spec/mappings/qwen_image_vae.h.)
constexpr std::array<std::string_view, 2> kTextIgnored{"lm_head.weight",
                                                       "model.language_model.norm.weight"};
constexpr std::array<std::string_view, 1> kTextIgnoredPrefixes{"model.visual."};

// The components' config keys the stages do not read, and why.
constexpr std::array kTextAuxiliary{
    AuxiliaryNamespace{"vision_config.", "vision tower, left out (its tensors are ignored)"},
    AuxiliaryNamespace{"tie_word_embeddings", "output head tying; the encoder has no output head"},
    AuxiliaryNamespace{"text_config.max_position_embeddings",
                       "trained context; the encoder's plain rotary reads prompt positions"},
    AuxiliaryNamespace{"text_config.rope_scaling.mrope_interleaved",
                       "multimodal rotary sections: a text prompt's positions are equal in "
                       "every section, which is plain rotary"},
    AuxiliaryNamespace{"text_config.rope_scaling.mrope_section",
                       "multimodal rotary sections: a text prompt's positions are equal in "
                       "every section, which is plain rotary"},
};
constexpr std::array kComponents{
    ImageComponentRule{.role = kText,
                       .stage = "text_encoder",
                       .stage_role = StageRole::kTextEncoder,
                       .kinds = kTextKinds,
                       .architectures = kTextArchitectures,
                       .containers = kSafetensorsContainer,
                       .ignored_tensors = kTextIgnored,
                       .ignored_tensor_prefixes = kTextIgnoredPrefixes,
                       .auxiliary = kTextAuxiliary},
    ImageComponentRule{.role = kFlow,
                       .stage = "denoiser",
                       .stage_role = StageRole::kDenoiser,
                       .kinds = kFlowKinds,
                       .architectures = kFlowArchitectures,
                       .containers = kFlowContainers,
                       .recipe = &kFlowRecipe},
    ImageComponentRule{.role = kVae,
                       .stage = "vae",
                       .stage_role = StageRole::kVae,
                       .kinds = kVaeKinds,
                       .architectures = kVaeArchitectures,
                       .containers = kSafetensorsContainer,
                       .mapping = &QwenImageVaeMapping},
};

constexpr ImageParamRule Text(std::string_view param, std::string_view key) {
  return {.kind = kImageTextKind, .param = param, .role = kText, .value = ex::V(key)};
}
constexpr ImageParamRule Flow(std::string_view param, std::string_view key) {
  return {.kind = kImageFlowKind, .param = param, .role = kFlow, .value = ex::V(key)};
}
// What diffusers' QwenImage21 classes fix in code rather than config.json.
constexpr ImageParamRule FlowConstant(std::string_view param, const RecipeValue& value) {
  return {.kind = kImageFlowKind, .param = param, .value = value};
}
constexpr ImageParamRule Schedule(std::string_view param, std::string_view key) {
  return {.kind = kImageScheduleKind, .param = param, .file = kScheduler, .value = ex::V(key)};
}
constexpr ImageParamRule Vae(std::string_view param, const RecipeValue& value) {
  return {.kind = kImageVaeKind, .param = param, .role = kVae, .value = value};
}
// The encoder's temporal downsampling, whose reverse is the decoder's.
constexpr auto kTemporalDownsample = ex::V("temperal_downsample");

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
    FlowConstant("timestep_dim", I(256)),
    FlowConstant("timestep_period", R(10000.0)),
    FlowConstant("timestep_scale", R(1000.0)),
    FlowConstant("timestep_divisor", R(1000.0)),
    FlowConstant("rope_theta", R(10000.0)),
    // scheduler/scheduler_config.json.
    Schedule("train_timesteps", "num_train_timesteps"),
    Schedule("base_image_seq_len", "base_image_seq_len"),
    Schedule("max_image_seq_len", "max_image_seq_len"),
    Schedule("base_shift", "base_shift"),
    Schedule("max_shift", "max_shift"),
    Schedule("shift_terminal", "shift_terminal"),
    // vae/config.json: the decoder (the encoder's temporal downsampling,
    // reversed, its up blocks' temporal steps).
    Vae("z_dim", ex::V("z_dim")),
    Vae("base", ex::V("decoder_base_dim")),
    Vae("dim_mult", ex::V("dim_mult")),
    Vae("res_blocks", ex::V("num_res_blocks")),
    Vae("temporal_upsample", ex::Reverse(kTemporalDownsample)),
    Vae("out_channels", ex::V("out_channels")),
    Vae("spatial_factor", ex::V("scale_factor_spatial")),
    Vae("latents_mean", ex::V("latents_mean")),
    Vae("latents_std", ex::V("latents_std")),
};

constexpr std::array kRequirements{
    ImageRequirement{.role = kText, .key = "text_config.hidden_act", .value = {.text = "silu"}},
    ImageRequirement{.role = kText,
                     .key = "text_config.attention_bias",
                     .type = ValueType::kBool,
                     .value = {.flag = false}},
    ImageRequirement{.role = kText,
                     .key = "text_config.rope_scaling.rope_type",
                     .value = {.text = "default"},
                     .optional = true},
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
