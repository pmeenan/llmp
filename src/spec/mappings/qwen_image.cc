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
constexpr std::array<std::string_view, 1> kSafetensorsContainer{"safetensors"};
constexpr std::array<std::string_view, 2> kFlowContainers{"safetensors", "gguf"};
// The VAE's config names either class (diffusers renamed it).
constexpr std::array<std::string_view, 2> kVaeArchitectures{"AutoencoderKLQwenImage21",
                                                            "AutoencoderKLQwenImage"};

constexpr RecipeValue I(std::int64_t value) { return {.literal = value}; }
constexpr RecipeValue R(double value) { return {.literal = value}; }
constexpr RecipeValue B(bool value) { return {.literal = value}; }
constexpr RecipeValue S(std::string_view value) { return {.literal = std::string(value)}; }
constexpr RecipeValue F(std::string_view kind, std::string_view param) {
  return {.kind = RecipeValueKind::kFamilyValue, .family_kind = kind, .family_param = param};
}
constexpr RecipeValue L(std::string_view kind, std::string_view param, std::int64_t index) {
  return {.kind = RecipeValueKind::kListElement,
          .family_kind = kind,
          .family_param = param,
          .fixed_index = index};
}
constexpr RecipeValue LI(std::string_view kind, std::string_view param, std::string_view index) {
  return {.kind = RecipeValueKind::kListElement,
          .family_kind = kind,
          .family_param = param,
          .index_name = index};
}
constexpr RecipeValue LR(std::string_view kind, std::string_view param, std::string_view index,
                         std::int64_t offset = 0) {
  return {.kind = RecipeValueKind::kListElementReverse,
          .family_kind = kind,
          .family_param = param,
          .index_name = index,
          .fixed_index = offset};
}
constexpr RecipeValue ListSize(std::string_view kind, std::string_view param) {
  return {.kind = RecipeValueKind::kListSize, .family_kind = kind, .family_param = param};
}
constexpr RecipeValue Idx(std::string_view name) {
  return {.kind = RecipeValueKind::kIndex, .index_name = name};
}
constexpr RecipeValue E(RecipeValueKind op, const RecipeValue& left, const RecipeValue& right) {
  return {.kind = op, .left = &left, .right = &right};
}
constexpr RecipeValue U(RecipeValueKind op, const RecipeValue& value) {
  return {.kind = op, .left = &value};
}
constexpr RecipeValue Select(const RecipeValue& condition, const RecipeValue& yes,
                             const RecipeValue& no) {
  return {.kind = RecipeValueKind::kSelect, .left = &condition, .right = &yes, .alternate = &no};
}
constexpr RecipeParam P(std::string_view name, RecipeValue value, std::string_view scalar = {}) {
  return {.name = name, .value = value, .request_scalar = scalar};
}
constexpr RecipeWeight W(std::string_view role, std::string_view tensor) {
  return {.role = role, .tensor = tensor};
}
constexpr RecipeRef V(std::string_view name) {
  return {.kind = RecipeRefKind::kNamedValue, .name = name};
}
constexpr RecipeRef H(std::uint16_t port) {
  return {.kind = RecipeRefKind::kHostPort, .port = port};
}
constexpr RecipeRef StageInput() { return {.kind = RecipeRefKind::kStageInput}; }

// Mapping-owned Qwen flow topology. The old spatial emitter remains a
// canonical-equivalence oracle until the recipe is wired into import planning.
const auto kFlowContext = F(kImageFlowKind, "context");
const auto kFlowHeads = F(kImageFlowKind, "heads");
const auto kFlowHeadDim = F(kImageFlowKind, "head_dim");
const auto kFlowWidth = E(RecipeValueKind::kMultiply, kFlowHeads, kFlowHeadDim);
const auto kFlowMlpRatio = F(kImageFlowKind, "mlp_ratio");
const auto kFlowMlp = E(RecipeValueKind::kMultiply, kFlowWidth, kFlowMlpRatio);
const auto kFlowEps = RecipeValue{
    .kind = RecipeValueKind::kFamilyF32Value, .family_kind = kImageFlowKind, .family_param = "eps"};
const auto kFlowRopeTheta = F(kImageFlowKind, "rope_theta");
const auto kFour = I(4);
const auto kScaleOne = R(1.0);
const auto kScaleRoot = U(RecipeValueKind::kSqrt, kFlowHeadDim);
const auto kAttentionScale = E(RecipeValueKind::kRealDivide, kScaleOne, kScaleRoot);

const std::array kT0P{P("in_channels", F(kImageFlowKind, "timestep_dim")),
                      P("out_channels", kFlowWidth), P("silu", B(true))};
const std::array kT1P{P("in_channels", kFlowWidth), P("out_channels", kFlowWidth),
                      P("silu", B(true))};
const std::array kT2P{P("in_channels", kFlowWidth),
                      P("out_channels", E(RecipeValueKind::kMultiply, kFour, kFlowWidth)),
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
                     .input = V("time.0")};
const RecipeNode kT2{.name = "time.2",
                     .component_kind = "projection.spatial",
                     .params = kT2P,
                     .weights = kT2W,
                     .input = V("time.1")};
const RecipeNode kT3{.name = "time.3",
                     .component_kind = "projection.spatial",
                     .params = kT3P,
                     .weights = kT3W,
                     .input = V("time.1")};

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
                         .input = V("text.norm")};
const std::array kTextOutP{P("in_channels", kFlowWidth), P("out_channels", kFlowWidth),
                           P("gelu", B(false))};
const std::array kTextOutW{W("weight", "txt_in.out_layer.weight")};
const RecipeNode kTextOut{.name = "text.out",
                          .component_kind = "projection.spatial",
                          .params = kTextOutP,
                          .weights = kTextOutW,
                          .input = V("text.in")};
const std::array kLatentP{P("start", I(0)), P("rows", I(1), "latent_rows"),
                          P("input_rows", I(1), "latent_rows")};
const RecipeNode kLatent{
    .name = "latent", .component_kind = "slice.spatial", .params = kLatentP, .input = H(0)};
const std::array kImageInP{P("in_channels", F(kImageFlowKind, "in_channels")),
                           P("out_channels", kFlowWidth), P("gelu", B(false))};
const std::array kImageInW{W("weight", "img_in.weight")};
const RecipeNode kImageIn{.name = "image.in",
                          .component_kind = "projection.spatial",
                          .params = kImageInP,
                          .weights = kImageInW,
                          .input = V("latent")};
const std::array kConcatP{P("axis", I(1))};
const RecipeNode kConcat{.name = "joint",
                         .component_kind = "concat.spatial",
                         .params = kConcatP,
                         .input = V("text.out"),
                         .auxiliary = V("image.in"),
                         .has_auxiliary = true};
const std::array kConditionP{P("quarter", I(0)), P("prefix_rows", I(1), "text_rows")};
const std::array kCondition1P{P("quarter", I(1)), P("prefix_rows", I(1), "text_rows")};
const std::array kCondition2P{P("quarter", I(2)), P("prefix_rows", I(1), "text_rows")};
const std::array kCondition3P{P("quarter", I(3)), P("prefix_rows", I(1), "text_rows")};
const RecipeNode kCondition0{.name = "condition.0",
                             .component_kind = "condition.rows",
                             .params = kConditionP,
                             .input = V("joint"),
                             .auxiliary = V("time.2"),
                             .has_auxiliary = true};
const RecipeNode kCondition1{.name = "condition.1",
                             .component_kind = "condition.rows",
                             .params = kCondition1P,
                             .input = V("joint"),
                             .auxiliary = V("time.2"),
                             .has_auxiliary = true};
const RecipeNode kCondition2{.name = "condition.2",
                             .component_kind = "condition.rows",
                             .params = kCondition2P,
                             .input = V("joint"),
                             .auxiliary = V("time.2"),
                             .has_auxiliary = true};
const RecipeNode kCondition3{.name = "condition.3",
                             .component_kind = "condition.rows",
                             .params = kCondition3P,
                             .input = V("joint"),
                             .auxiliary = V("time.2"),
                             .has_auxiliary = true};

const std::array kAdaptiveP{P("side", S("scale")), P("eps", kFlowEps)};
const std::array kAttentionP{P("eps", kFlowEps),
                             P("heads", kFlowHeads),
                             P("key_dim", kFlowHeadDim),
                             P("kv_heads", kFlowHeads),
                             P("placement", S("bare")),
                             P("qk_norm", B(true)),
                             P("rope_axis0", L(kImageFlowKind, "rope_axes", 0)),
                             P("rope_axis1", L(kImageFlowKind, "rope_axes", 1)),
                             P("rope_axis2", L(kImageFlowKind, "rope_axes", 2)),
                             P("rope_base", kFlowRopeTheta),
                             P("rope_dims", kFlowHeadDim),
                             P("rope_scale", R(1.0)),
                             P("rotation", S("normal")),
                             P("scale", kAttentionScale),
                             P("scale_scores", B(true)),
                             P("stateless", B(true)),
                             P("value_dim", kFlowHeadDim),
                             P("window", I(0))};
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
                         .input = V("stream.{block}"),
                         .auxiliary = V("condition.0"),
                         .has_auxiliary = true};
const RecipeNode kBlock1{.name = "attention.{block}",
                         .component_kind = "attention.dense",
                         .params = kAttentionP,
                         .weights = kAttentionW,
                         .input = V("adaptive.0.{block}"),
                         .auxiliary = H(2),
                         .carry = H(3),
                         .has_auxiliary = true,
                         .has_carry = true,
                         .group = RecipeGroupKind::kLayer,
                         .group_index = {.kind = RecipeValueKind::kIndex, .index_name = "block"}};
const RecipeNode kBlock2{.name = "residual.0.{block}",
                         .component_kind = "residual.gated",
                         .params = kGatedP,
                         .input = V("attention.{block}"),
                         .auxiliary = V("condition.1"),
                         .carry = V("stream.{block}"),
                         .has_auxiliary = true,
                         .has_carry = true};
const RecipeNode kBlock3{.name = "adaptive.1.{block}",
                         .component_kind = "norm.adaptive",
                         .params = kAdaptiveP,
                         .input = V("residual.0.{block}"),
                         .auxiliary = V("condition.2"),
                         .has_auxiliary = true};
const RecipeNode kBlock4{.name = "mlp.{block}",
                         .component_kind = "ffn.glu",
                         .params = kMlpP,
                         .weights = kMlpW,
                         .input = V("adaptive.1.{block}"),
                         .group = RecipeGroupKind::kLayer,
                         .group_index = {.kind = RecipeValueKind::kIndex, .index_name = "block"}};
const RecipeNode kBlock5{.name = "stream.next.{block}",
                         .component_kind = "residual.gated",
                         .params = kGatedP,
                         .input = V("mlp.{block}"),
                         .auxiliary = V("condition.3"),
                         .carry = V("residual.0.{block}"),
                         .has_auxiliary = true,
                         .has_carry = true};
const std::array kBlockSteps{RecipeStep{.kind = RecipeStep::Kind::kNode, .node = &kBlock0},
                             RecipeStep{.kind = RecipeStep::Kind::kNode, .node = &kBlock1},
                             RecipeStep{.kind = RecipeStep::Kind::kNode, .node = &kBlock2},
                             RecipeStep{.kind = RecipeStep::Kind::kNode, .node = &kBlock3},
                             RecipeStep{.kind = RecipeStep::Kind::kNode, .node = &kBlock4},
                             RecipeStep{.kind = RecipeStep::Kind::kNode, .node = &kBlock5}};
const RecipeRepeat kFlowRepeat{.index = "block",
                               .count = F(kImageFlowKind, "blocks"),
                               .body = kBlockSteps,
                               .input = V("joint"),
                               .input_name = "stream.{block}",
                               .body_yield = "stream.next.{block}",
                               .yield = "stream.blocks"};

const std::array kSliceOutP{P("start", I(1), "text_rows"), P("rows", I(1), "latent_rows"),
                            P("input_rows", I(2), "joint_rows")};
const RecipeNode kSliceOut{.name = "image.out",
                           .component_kind = "slice.spatial",
                           .params = kSliceOutP,
                           .input = V("stream.blocks")};
const std::array kScaleP{P("quarter", I(0)), P("quarters", I(1)), P("prefix_rows", I(0))};
const RecipeNode kScale{.name = "image.scale",
                        .component_kind = "condition.rows",
                        .params = kScaleP,
                        .input = V("image.out"),
                        .auxiliary = V("time.3"),
                        .has_auxiliary = true};
const RecipeNode kNormOut{.name = "image.norm",
                          .component_kind = "norm.adaptive",
                          .params = kAdaptiveP,
                          .input = V("image.out"),
                          .auxiliary = V("image.scale"),
                          .has_auxiliary = true};
const std::array kProjectOutP{P("in_channels", kFlowWidth),
                              P("out_channels", F(kImageFlowKind, "out_channels")),
                              P("gelu", B(false))};
const std::array kProjectOutW{W("weight", "proj_out.weight")};
const RecipeNode kProjectOut{.name = "noise",
                             .component_kind = "projection.spatial",
                             .params = kProjectOutP,
                             .weights = kProjectOutW,
                             .input = V("image.norm")};
const RecipeNode kEuler{.name = "euler",
                        .component_kind = "residual.scaled",
                        .input = V("noise"),
                        .auxiliary = H(4),
                        .carry = V("latent"),
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
const std::array kFlowExports{V("noise"), V("euler")};
const RecipeSpatialOperand kFlowPrimary{
    .shape = {.width = I(1), .height = I(1), .channels = kFlowContext, .images = I(1)}};
const std::array kFlowHostPorts{
    RecipeSpatialOperand{.shape = {.width = I(1),
                                   .height = I(1),
                                   .channels = F(kImageFlowKind, "in_channels"),
                                   .images = I(1)}},
    RecipeSpatialOperand{.shape = {.width = I(1),
                                   .height = I(2),
                                   .channels = F(kImageFlowKind, "timestep_dim"),
                                   .images = I(1)}},
    RecipeSpatialOperand{.shape = {.width = I(2), .height = I(3), .channels = I(1), .images = I(1)},
                         .encoding = format::Encoding::kI32},
    RecipeSpatialOperand{
        .shape = {.width = I(2), .height = I(2), .channels = I(1), .images = I(1)}},
    RecipeSpatialOperand{
        .shape = {.width = I(1), .height = I(1), .channels = I(1), .images = I(1)}}};
const RecipeSpatialShape kFlowOutput{
    .width = I(1), .height = I(1), .channels = F(kImageFlowKind, "in_channels"), .images = I(1)};
const std::array kFlowOutputShapes{kFlowOutput, kFlowOutput};
const StageRecipe kFlowRecipe{.name = "flow",
                              .role = StageRole::kDenoiser,
                              .stream_width = kFlowWidth,
                              .primary_input = kFlowPrimary,
                              .host_ports = kFlowHostPorts,
                              .steps = kFlowSteps,
                              .exports = kFlowExports,
                              .expected_outputs = kFlowOutputShapes};

// The VAE decoder is also mapping-owned topology. Indices are lexical to
// finite repeats; tensor names and layer groups are derived from the same
// substitutions as the component graph.
const auto kVaeLevels = ListSize(kImageVaeKind, "dim_mult");
const auto kVaeBase = F(kImageVaeKind, "base");
const auto kVaeZ = F(kImageVaeKind, "z_dim");
const auto kVaeOut = F(kImageVaeKind, "out_channels");
const auto kVaeSpatialFactor = F(kImageVaeKind, "spatial_factor");
const auto kVaeOutMultiplier = L(kImageVaeKind, "dim_mult", 0);
const auto kVaeOutChannels = E(RecipeValueKind::kMultiply, kVaeBase, kVaeOutMultiplier);
const auto kVaeMidMultiplier = LR(kImageVaeKind, "dim_mult", "");
const auto kVaeMidChannels = E(RecipeValueKind::kMultiply, kVaeBase, kVaeMidMultiplier);
const auto kVaeLevel = Idx("level");
const auto kVaeResidual = Idx("residual");
const auto kRecipeOne = I(1);
const auto kRecipeZero = I(0);
const auto kVaeLastLevel = E(RecipeValueKind::kSubtract, kVaeLevels, kRecipeOne);
const auto kVaeIsUp = E(RecipeValueKind::kLess, kVaeLevel, kVaeLastLevel);
const auto kVaeZeroLevel = E(RecipeValueKind::kEqual, kVaeLevel, kRecipeZero);
const auto kVaePreviousLevelMultiplier = LR(kImageVaeKind, "dim_mult", "level", -1);
const auto kVaeCinMultiplier =
    Select(kVaeZeroLevel, kVaeMidMultiplier, kVaePreviousLevelMultiplier);
const auto kVaeCin = E(RecipeValueKind::kMultiply, kVaeBase, kVaeCinMultiplier);
const auto kVaeCoutUpMultiplier = LR(kImageVaeKind, "dim_mult", "level");
const auto kVaeCoutMultiplier = Select(kVaeIsUp, kVaeCoutUpMultiplier, kVaeOutMultiplier);
const auto kVaeCout = E(RecipeValueKind::kMultiply, kVaeBase, kVaeCoutMultiplier);
const auto kVaeFirstResidual = E(RecipeValueKind::kEqual, kVaeResidual, kRecipeZero);
const auto kVaeResidualChannels = Select(kVaeFirstResidual, kVaeCin, kVaeCout);
const auto kVaeCinCoutEqual = E(RecipeValueKind::kEqual, kVaeCin, kVaeCout);
const auto kVaeNeedsShortcutNotEqual = U(RecipeValueKind::kNot, kVaeCinCoutEqual);
const auto kVaeNeedsShortcut =
    E(RecipeValueKind::kAnd, kVaeFirstResidual, kVaeNeedsShortcutNotEqual);
const auto kVaeResBlocks = F(kImageVaeKind, "res_blocks");
const auto kVaeResCount = E(RecipeValueKind::kAdd, kVaeResBlocks, kRecipeOne);

const std::array kVaeReshapeP{P("width", I(1), "latent_width"), P("height", I(1), "latent_height")};
const RecipeNode kVaeReshape{.name = "vae.reshape",
                             .component_kind = "reshape.spatial",
                             .params = kVaeReshapeP,
                             .input = StageInput()};
const RecipeNode kVaeAffine{.name = "vae.affine",
                            .component_kind = "affine.channel",
                            .input = V("vae.reshape"),
                            .auxiliary = H(0),
                            .has_auxiliary = true};
const std::array kVaePostQuantP{P("in_channels", kVaeZ), P("out_channels", kVaeZ),
                                P("kernel", I(1)), P("padding", I(0))};
const std::array kVaePostQuantW{W("kernel", "post_quant_conv.weight"),
                                W("bias", "post_quant_conv.bias")};
const RecipeNode kVaePostQuant{.name = "vae.post_quant",
                               .component_kind = "conv.spatial",
                               .params = kVaePostQuantP,
                               .weights = kVaePostQuantW,
                               .input = V("vae.affine")};
const std::array kVaeConvInP{P("in_channels", kVaeZ), P("out_channels", kVaeMidChannels),
                             P("kernel", I(3)), P("padding", I(1))};
const std::array kVaeConvInW{W("kernel", "decoder.conv_in.weight"),
                             W("bias", "decoder.conv_in.bias")};
const RecipeNode kVaeConvIn{.name = "vae.conv_in",
                            .component_kind = "conv.spatial",
                            .params = kVaeConvInP,
                            .weights = kVaeConvInW,
                            .input = V("vae.post_quant")};

const std::array kVaeMidRes1Norm1P{P("channels", kVaeMidChannels), P("weight_rank", I(4)),
                                   P("weight_axis", I(3)), P("silu", B(true))};
const std::array kVaeMidRes1Norm1W{W("weight", "decoder.mid_block.resnets.0.norm1.gamma")};
const RecipeNode kVaeMidRes0Norm1{.name = "vae.mid.0.norm1",
                                  .component_kind = "norm.channel",
                                  .params = kVaeMidRes1Norm1P,
                                  .weights = kVaeMidRes1Norm1W,
                                  .input = V("vae.conv_in")};
const std::array kVaeMidResConvP{P("in_channels", kVaeMidChannels),
                                 P("out_channels", kVaeMidChannels), P("kernel", I(3)),
                                 P("padding", I(1))};
const std::array kVaeMidResConv1W{W("kernel", "decoder.mid_block.resnets.0.conv1.weight"),
                                  W("bias", "decoder.mid_block.resnets.0.conv1.bias")};
const RecipeNode kVaeMidRes0Conv1{.name = "vae.mid.0.conv1",
                                  .component_kind = "conv.spatial",
                                  .params = kVaeMidResConvP,
                                  .weights = kVaeMidResConv1W,
                                  .input = V("vae.mid.0.norm1")};
const std::array kVaeMidResNorm2W{W("weight", "decoder.mid_block.resnets.0.norm2.gamma")};
const RecipeNode kVaeMidRes0Norm2{.name = "vae.mid.0.norm2",
                                  .component_kind = "norm.channel",
                                  .params = kVaeMidRes1Norm1P,
                                  .weights = kVaeMidResNorm2W,
                                  .input = V("vae.mid.0.conv1")};
const std::array kVaeMidResConv2W{W("kernel", "decoder.mid_block.resnets.0.conv2.weight"),
                                  W("bias", "decoder.mid_block.resnets.0.conv2.bias")};
const RecipeNode kVaeMidRes0Conv2{.name = "vae.mid.0.conv2",
                                  .component_kind = "conv.spatial",
                                  .params = kVaeMidResConvP,
                                  .weights = kVaeMidResConv2W,
                                  .input = V("vae.mid.0.norm2")};
const RecipeNode kVaeMidRes0Add{.name = "vae.mid.0.out",
                                .component_kind = "residual.spatial",
                                .input = V("vae.mid.0.conv2"),
                                .auxiliary = V("vae.conv_in"),
                                .has_auxiliary = true};

const std::array kVaeMidAttentionNormP{P("channels", kVaeMidChannels), P("weight_rank", I(3)),
                                       P("weight_axis", I(2)), P("silu", B(false))};
const std::array kVaeMidAttentionNormW{W("weight", "decoder.mid_block.attentions.0.norm.gamma")};
const RecipeNode kVaeMidAttentionNorm{.name = "vae.mid.attn.norm",
                                      .component_kind = "norm.channel",
                                      .params = kVaeMidAttentionNormP,
                                      .weights = kVaeMidAttentionNormW,
                                      .input = V("vae.mid.0.out")};
const auto kRecipeThree = I(3);
const auto kVaeTripleChannels = E(RecipeValueKind::kMultiply, kVaeMidChannels, kRecipeThree);
const std::array kVaeQkvP{P("in_channels", kVaeMidChannels), P("out_channels", kVaeTripleChannels),
                          P("kernel", I(1)), P("padding", I(0))};
const std::array kVaeQkvW{W("kernel", "decoder.mid_block.attentions.0.to_qkv.weight"),
                          W("bias", "decoder.mid_block.attentions.0.to_qkv.bias")};
const RecipeNode kVaeQkv{.name = "vae.mid.attn.qkv",
                         .component_kind = "conv.spatial",
                         .params = kVaeQkvP,
                         .weights = kVaeQkvW,
                         .input = V("vae.mid.attn.norm")};
const RecipeNode kVaeAttention{
    .name = "vae.mid.attn", .component_kind = "attention.spatial", .input = V("vae.mid.attn.qkv")};
const std::array kVaeProjW{W("kernel", "decoder.mid_block.attentions.0.proj.weight"),
                           W("bias", "decoder.mid_block.attentions.0.proj.bias")};
const std::array kVaeProjP{P("in_channels", kVaeMidChannels), P("out_channels", kVaeMidChannels),
                           P("kernel", I(1)), P("padding", I(0))};
const RecipeNode kVaeProj{.name = "vae.mid.attn.proj",
                          .component_kind = "conv.spatial",
                          .params = kVaeProjP,
                          .weights = kVaeProjW,
                          .input = V("vae.mid.attn")};
const RecipeNode kVaeAttentionAdd{.name = "vae.mid.attn.out",
                                  .component_kind = "residual.spatial",
                                  .input = V("vae.mid.attn.proj"),
                                  .auxiliary = V("vae.mid.0.out"),
                                  .has_auxiliary = true};
// The second middle residual uses the same component shapes with its own
// source names; keeping it static makes the mapping easy to audit.
const std::array kVaeMid2Norm1W{W("weight", "decoder.mid_block.resnets.1.norm1.gamma")};
const RecipeNode kVaeMidRes1Norm1{.name = "vae.mid.1.norm1",
                                  .component_kind = "norm.channel",
                                  .params = kVaeMidRes1Norm1P,
                                  .weights = kVaeMid2Norm1W,
                                  .input = V("vae.mid.attn.out")};
const std::array kVaeMid2Conv1W{W("kernel", "decoder.mid_block.resnets.1.conv1.weight"),
                                W("bias", "decoder.mid_block.resnets.1.conv1.bias")};
const RecipeNode kVaeMidRes1Conv1{.name = "vae.mid.1.conv1",
                                  .component_kind = "conv.spatial",
                                  .params = kVaeMidResConvP,
                                  .weights = kVaeMid2Conv1W,
                                  .input = V("vae.mid.1.norm1")};
const std::array kVaeMid2Norm2W{W("weight", "decoder.mid_block.resnets.1.norm2.gamma")};
const RecipeNode kVaeMidRes1Norm2{.name = "vae.mid.1.norm2",
                                  .component_kind = "norm.channel",
                                  .params = kVaeMidRes1Norm1P,
                                  .weights = kVaeMid2Norm2W,
                                  .input = V("vae.mid.1.conv1")};
const std::array kVaeMid2Conv2W{W("kernel", "decoder.mid_block.resnets.1.conv2.weight"),
                                W("bias", "decoder.mid_block.resnets.1.conv2.bias")};
const RecipeNode kVaeMidRes1Conv2{.name = "vae.mid.1.conv2",
                                  .component_kind = "conv.spatial",
                                  .params = kVaeMidResConvP,
                                  .weights = kVaeMid2Conv2W,
                                  .input = V("vae.mid.1.norm2")};
const RecipeNode kVaeMidRes1Add{.name = "vae.mid.1.out",
                                .component_kind = "residual.spatial",
                                .input = V("vae.mid.1.conv2"),
                                .auxiliary = V("vae.mid.attn.out"),
                                .has_auxiliary = true};

const auto kVaeCoutLevel = kVaeCout;
const std::array kVaeShortcutP{P("in_channels", kVaeCin), P("out_channels", kVaeCoutLevel),
                               P("kernel", I(1)), P("padding", I(0))};
const std::array kVaeShortcutW{
    W("kernel", "decoder.up_blocks.{level}.resnets.{residual}.conv_shortcut.weight"),
    W("bias", "decoder.up_blocks.{level}.resnets.{residual}.conv_shortcut.bias")};
const RecipeNode kVaeShortcut{.name = "vae.up.{level}.{residual}.shortcut",
                              .component_kind = "conv.spatial",
                              .params = kVaeShortcutP,
                              .weights = kVaeShortcutW,
                              .input = V("vae.up.{level}.{residual}.source"),
                              .group = RecipeGroupKind::kLayer,
                              .group_index = kVaeLevel};
const std::array kVaeShortcutSteps{
    RecipeStep{.kind = RecipeStep::Kind::kNode, .node = &kVaeShortcut}};
const RecipeIf kVaeShortcutIf{.condition = kVaeNeedsShortcut,
                              .then_steps = kVaeShortcutSteps,
                              .else_steps = {},
                              .input = V("vae.up.{level}.{residual}.stream"),
                              .then_yield = "vae.up.{level}.{residual}.shortcut",
                              .else_yield = "vae.up.{level}.{residual}.source",
                              .yield = "vae.up.{level}.{residual}.skip",
                              .input_name = "vae.up.{level}.{residual}.source"};
const RecipeStep kVaeShortcutStep{.kind = RecipeStep::Kind::kIf, .branch = &kVaeShortcutIf};

const std::array kVaeNorm1P{P("channels", kVaeResidualChannels), P("weight_rank", I(4)),
                            P("weight_axis", I(3)), P("silu", B(true))};
const std::array kVaeNorm1W{
    W("weight", "decoder.up_blocks.{level}.resnets.{residual}.norm1.gamma")};
const RecipeNode kVaeNorm1{.name = "vae.up.{level}.{residual}.norm1",
                           .component_kind = "norm.channel",
                           .params = kVaeNorm1P,
                           .weights = kVaeNorm1W,
                           .input = V("vae.up.{level}.{residual}.stream"),
                           .group = RecipeGroupKind::kLayer,
                           .group_index = kVaeLevel};
const std::array kVaeConv1P{P("in_channels", kVaeResidualChannels),
                            P("out_channels", kVaeCoutLevel), P("kernel", I(3)),
                            P("padding", I(1))};
const std::array kVaeConv1W{
    W("kernel", "decoder.up_blocks.{level}.resnets.{residual}.conv1.weight"),
    W("bias", "decoder.up_blocks.{level}.resnets.{residual}.conv1.bias")};
const RecipeNode kVaeConv1{.name = "vae.up.{level}.{residual}.conv1",
                           .component_kind = "conv.spatial",
                           .params = kVaeConv1P,
                           .weights = kVaeConv1W,
                           .input = V("vae.up.{level}.{residual}.norm1"),
                           .group = RecipeGroupKind::kLayer,
                           .group_index = kVaeLevel};
const std::array kVaeNorm2P{P("channels", kVaeCoutLevel), P("weight_rank", I(4)),
                            P("weight_axis", I(3)), P("silu", B(true))};
const std::array kVaeNorm2W{
    W("weight", "decoder.up_blocks.{level}.resnets.{residual}.norm2.gamma")};
const RecipeNode kVaeNorm2{.name = "vae.up.{level}.{residual}.norm2",
                           .component_kind = "norm.channel",
                           .params = kVaeNorm2P,
                           .weights = kVaeNorm2W,
                           .input = V("vae.up.{level}.{residual}.conv1"),
                           .group = RecipeGroupKind::kLayer,
                           .group_index = kVaeLevel};
const std::array kVaeConv2W{
    W("kernel", "decoder.up_blocks.{level}.resnets.{residual}.conv2.weight"),
    W("bias", "decoder.up_blocks.{level}.resnets.{residual}.conv2.bias")};
const std::array kVaeConv2P{P("in_channels", kVaeCoutLevel), P("out_channels", kVaeCoutLevel),
                            P("kernel", I(3)), P("padding", I(1))};
const RecipeNode kVaeConv2{.name = "vae.up.{level}.{residual}.conv2",
                           .component_kind = "conv.spatial",
                           .params = kVaeConv2P,
                           .weights = kVaeConv2W,
                           .input = V("vae.up.{level}.{residual}.norm2"),
                           .group = RecipeGroupKind::kLayer,
                           .group_index = kVaeLevel};
const RecipeNode kVaeResAdd{.name = "vae.up.{level}.{residual}.out",
                            .component_kind = "residual.spatial",
                            .input = V("vae.up.{level}.{residual}.conv2"),
                            .auxiliary = V("vae.up.{level}.{residual}.skip"),
                            .has_auxiliary = true};
const std::array kVaeResidualSteps{
    kVaeShortcutStep,
    RecipeStep{.kind = RecipeStep::Kind::kNode, .node = &kVaeNorm1},
    RecipeStep{.kind = RecipeStep::Kind::kNode, .node = &kVaeConv1},
    RecipeStep{.kind = RecipeStep::Kind::kNode, .node = &kVaeNorm2},
    RecipeStep{.kind = RecipeStep::Kind::kNode, .node = &kVaeConv2},
    RecipeStep{.kind = RecipeStep::Kind::kNode, .node = &kVaeResAdd}};
const RecipeRepeat kVaeResidualRepeat{.index = "residual",
                                      .count = kVaeResCount,
                                      .body = kVaeResidualSteps,
                                      .input = V("vae.up.entry.{level}"),
                                      .input_name = "vae.up.{level}.{residual}.stream",
                                      .body_yield = "vae.up.{level}.{residual}.out",
                                      .yield = "vae.up.result.{level}"};

const RecipeIf kVaeSaveInput{.condition = B(true),
                             .input = V("vae.up.entry.{level}"),
                             .then_yield = "vae.up.save.{level}",
                             .else_yield = "vae.up.save.{level}",
                             .yield = "vae.up.saved.{level}",
                             .input_name = "vae.up.save.{level}"};
const RecipeStep kVaeSaveInputStep{.kind = RecipeStep::Kind::kIf, .branch = &kVaeSaveInput};
const std::array kVaeUpP{P("factor", I(2))};
const RecipeNode kVaeNearest{.name = "vae.up.{level}.nearest",
                             .component_kind = "upsample.nearest",
                             .params = kVaeUpP,
                             .input = V("vae.up.result.{level}")};
const std::array kVaeUpConvP{P("in_channels", kVaeCoutLevel), P("out_channels", kVaeCoutLevel),
                             P("kernel", I(3)), P("padding", I(1))};
const std::array kVaeUpConvW{W("kernel", "decoder.up_blocks.{level}.upsampler.resample.1.weight"),
                             W("bias", "decoder.up_blocks.{level}.upsampler.resample.1.bias")};
const RecipeNode kVaeUpConv{.name = "vae.up.{level}.conv",
                            .component_kind = "conv.spatial",
                            .params = kVaeUpConvP,
                            .weights = kVaeUpConvW,
                            .input = V("vae.up.{level}.nearest"),
                            .group = RecipeGroupKind::kLayer,
                            .group_index = kVaeLevel};
const std::array kVaeDup1P{P("factor_t", kRecipeOne)};
const RecipeNode kVaeDup1{.name = "vae.up.{level}.dup1",
                          .component_kind = "residual.upsample",
                          .params = kVaeDup1P,
                          .input = V("vae.up.{level}.conv"),
                          .auxiliary = V("vae.up.saved.{level}"),
                          .has_auxiliary = true};
const std::array kVaeDup2P{P("factor_t", I(2))};
const RecipeNode kVaeDup2{.name = "vae.up.{level}.dup2",
                          .component_kind = "residual.upsample",
                          .params = kVaeDup2P,
                          .input = V("vae.up.{level}.conv"),
                          .auxiliary = V("vae.up.saved.{level}"),
                          .has_auxiliary = true};
const std::array kVaeDup1Steps{RecipeStep{.kind = RecipeStep::Kind::kNode, .node = &kVaeDup1}};
const std::array kVaeDup2Steps{RecipeStep{.kind = RecipeStep::Kind::kNode, .node = &kVaeDup2}};
const auto kVaeTemporal = LI(kImageVaeKind, "temporal_upsample", "level");
const auto kVaeTemporalIsUpsample = E(RecipeValueKind::kNotEqual, kVaeTemporal, kRecipeZero);
const RecipeIf kVaeTemporalIf{.condition = kVaeTemporalIsUpsample,
                              .then_steps = kVaeDup2Steps,
                              .else_steps = kVaeDup1Steps,
                              .input = V("vae.up.{level}.conv"),
                              .then_yield = "vae.up.{level}.dup2",
                              .else_yield = "vae.up.{level}.dup1",
                              .yield = "vae.up.{level}.done",
                              .input_name = "vae.up.{level}.upstream"};
const RecipeStep kVaeTemporalIfStep{.kind = RecipeStep::Kind::kIf, .branch = &kVaeTemporalIf};
const std::array kVaeUpsampleSteps{
    kVaeSaveInputStep, RecipeStep{.kind = RecipeStep::Kind::kRepeat, .repeat = &kVaeResidualRepeat},
    RecipeStep{.kind = RecipeStep::Kind::kNode, .node = &kVaeNearest},
    RecipeStep{.kind = RecipeStep::Kind::kNode, .node = &kVaeUpConv}, kVaeTemporalIfStep};
const std::array kVaeFinalLevelSteps{
    RecipeStep{.kind = RecipeStep::Kind::kRepeat, .repeat = &kVaeResidualRepeat}};
const RecipeIf kVaeUpsampleIf{.condition = kVaeIsUp,
                              .then_steps = kVaeUpsampleSteps,
                              .else_steps = kVaeFinalLevelSteps,
                              .input = V("vae.up.current.{level}"),
                              .then_yield = "vae.up.{level}.done",
                              .else_yield = "vae.up.result.{level}",
                              .yield = "vae.up.next.{level}",
                              .input_name = "vae.up.entry.{level}"};
const RecipeStep kVaeUpsampleIfStep{.kind = RecipeStep::Kind::kIf, .branch = &kVaeUpsampleIf};
const std::array kVaeLevelSteps{kVaeUpsampleIfStep};
const RecipeRepeat kVaeLevelRepeat{.index = "level",
                                   .count = kVaeLevels,
                                   .body = kVaeLevelSteps,
                                   .input = V("vae.mid.1.out"),
                                   .input_name = "vae.up.current.{level}",
                                   .body_yield = "vae.up.next.{level}",
                                   .yield = "vae.up.final"};
const RecipeStep kVaeLevelRepeatStep{.kind = RecipeStep::Kind::kRepeat, .repeat = &kVaeLevelRepeat};

const std::array kVaeNormOutP{P("channels", kVaeOutChannels), P("weight_rank", I(4)),
                              P("weight_axis", I(3)), P("silu", B(true))};
const std::array kVaeNormOutW{W("weight", "decoder.norm_out.gamma")};
const RecipeNode kVaeNormOut{.name = "vae.norm_out",
                             .component_kind = "norm.channel",
                             .params = kVaeNormOutP,
                             .weights = kVaeNormOutW,
                             .input = V("vae.up.final")};
const std::array kVaeConvOutP{P("in_channels", kVaeOutChannels), P("out_channels", kVaeOut),
                              P("kernel", I(3)), P("padding", I(1))};
const std::array kVaeConvOutW{W("kernel", "decoder.conv_out.weight"),
                              W("bias", "decoder.conv_out.bias")};
const RecipeNode kVaeConvOut{.name = "vae.output",
                             .component_kind = "conv.spatial",
                             .params = kVaeConvOutP,
                             .weights = kVaeConvOutW,
                             .input = V("vae.norm_out")};
const std::array kVaeSteps{
    RecipeStep{.kind = RecipeStep::Kind::kNode, .node = &kVaeReshape},
    RecipeStep{.kind = RecipeStep::Kind::kNode, .node = &kVaeAffine},
    RecipeStep{.kind = RecipeStep::Kind::kNode, .node = &kVaePostQuant},
    RecipeStep{.kind = RecipeStep::Kind::kNode, .node = &kVaeConvIn},
    RecipeStep{.kind = RecipeStep::Kind::kNode, .node = &kVaeMidRes0Norm1},
    RecipeStep{.kind = RecipeStep::Kind::kNode, .node = &kVaeMidRes0Conv1},
    RecipeStep{.kind = RecipeStep::Kind::kNode, .node = &kVaeMidRes0Norm2},
    RecipeStep{.kind = RecipeStep::Kind::kNode, .node = &kVaeMidRes0Conv2},
    RecipeStep{.kind = RecipeStep::Kind::kNode, .node = &kVaeMidRes0Add},
    RecipeStep{.kind = RecipeStep::Kind::kNode, .node = &kVaeMidAttentionNorm},
    RecipeStep{.kind = RecipeStep::Kind::kNode, .node = &kVaeQkv},
    RecipeStep{.kind = RecipeStep::Kind::kNode, .node = &kVaeAttention},
    RecipeStep{.kind = RecipeStep::Kind::kNode, .node = &kVaeProj},
    RecipeStep{.kind = RecipeStep::Kind::kNode, .node = &kVaeAttentionAdd},
    RecipeStep{.kind = RecipeStep::Kind::kNode, .node = &kVaeMidRes1Norm1},
    RecipeStep{.kind = RecipeStep::Kind::kNode, .node = &kVaeMidRes1Conv1},
    RecipeStep{.kind = RecipeStep::Kind::kNode, .node = &kVaeMidRes1Norm2},
    RecipeStep{.kind = RecipeStep::Kind::kNode, .node = &kVaeMidRes1Conv2},
    RecipeStep{.kind = RecipeStep::Kind::kNode, .node = &kVaeMidRes1Add},
    kVaeLevelRepeatStep,
    RecipeStep{.kind = RecipeStep::Kind::kNode, .node = &kVaeNormOut},
    RecipeStep{.kind = RecipeStep::Kind::kNode, .node = &kVaeConvOut}};
const std::array kVaeHostPorts{RecipeSpatialOperand{
    .shape = {.width = I(1), .height = I(2), .channels = kVaeZ, .images = I(1)}}};
const std::array kVaeOutputShapes{RecipeSpatialShape{
    .width = kVaeSpatialFactor, .height = kVaeSpatialFactor, .channels = kVaeOut, .images = I(1)}};
const StageRecipe kVaeRecipe{
    .name = "vae",
    .role = StageRole::kVae,
    .stream_width = kVaeZ,
    .primary_input = {.shape = {.width = I(1), .height = I(1), .channels = kVaeZ, .images = I(1)}},
    .host_ports = kVaeHostPorts,
    .steps = kVaeSteps,
    .expected_outputs = kVaeOutputShapes};

// What text-to-image generation does not run (each stage reads the
// tensors model/image_pipeline.h's lists give it, and no others):
// Qwen3-VL's vision tower, output head and final norm; the VAE's encoder,
// and its up blocks' temporal convolutions, which a one-frame decode never
// applies (the first frame skips them).
constexpr std::array<std::string_view, 3> kTextIgnored{"model.visual.", "lm_head.weight",
                                                       "model.language_model.norm.weight"};
constexpr std::array<std::string_view, 3> kVaeIgnored{
    "encoder.", "quant_conv.", "decoder.up_blocks.{layer}.upsampler.time_conv."};

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
constexpr std::array kVaeAuxiliary{
    AuxiliaryNamespace{"base_dim", "the encoder's width; the encoder is left out"},
    AuxiliaryNamespace{"in_channels", "the encoder's input channels; the encoder is left out"},
    AuxiliaryNamespace{"attn_scales",
                       "the encoder's attention scales; an attention block they add elsewhere "
                       "is a tensor the stage does not read, refused"},
    AuxiliaryNamespace{"scale_factor_temporal",
                       "frames a latent frame decodes to; one image is "
                       "one frame (the time convolutions are left out)"},
};
constexpr std::array<std::string_view, 1> kVaeNullKeys{"patch_size"};
constexpr std::array kComponents{
    ImageComponentRule{.role = kText,
                       .stage = "text_encoder",
                       .stage_role = StageRole::kTextEncoder,
                       .kinds = kTextKinds,
                       .architectures = kTextArchitectures,
                       .containers = kSafetensorsContainer,
                       .ignored_tensors = kTextIgnored,
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
                       .recipe = &kVaeRecipe,
                       .ignored_tensors = kVaeIgnored,
                       .auxiliary = kVaeAuxiliary,
                       .null_keys = kVaeNullKeys},
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
