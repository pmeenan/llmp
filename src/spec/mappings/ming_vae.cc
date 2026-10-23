// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "spec/mappings/ming_vae.h"

#include <array>
#include <cstdint>
#include <string_view>

#include "spec/stage_recipe.h"

namespace llmp::spec::mappings {
namespace {

constexpr std::array<std::string_view, 1> kSourceArchitectures{"AutoencoderKLQwenImage"};
constexpr std::array<std::string_view, 1> kContainers{"safetensors"};
constexpr RecipeValue I(std::int64_t value) { return {.literal = value}; }
constexpr RecipeValue R(double value) { return {.literal = value}; }
constexpr RecipeValue B(bool value) { return {.literal = value}; }
constexpr RecipeValue S(std::string_view key) {
  return {.kind = RecipeValueKind::kSourceValue, .source_key = key};
}
constexpr RecipeValue SL(std::string_view key, std::int64_t index) {
  return {.kind = RecipeValueKind::kSourceListElement, .source_key = key, .fixed_index = index};
}
constexpr RecipeValue SLR(std::string_view key, std::string_view index, std::int64_t offset = 0) {
  return {.kind = RecipeValueKind::kSourceListElementReverse,
          .source_key = key,
          .index_name = index,
          .fixed_index = offset};
}
constexpr RecipeValue SListSize(std::string_view key) {
  return {.kind = RecipeValueKind::kSourceListSize, .source_key = key};
}
constexpr RecipeValue Idx(std::string_view name) {
  return {.kind = RecipeValueKind::kIndex, .index_name = name};
}
consteval RecipeValue E(RecipeValueKind kind, const RecipeValue& left, const RecipeValue& right) {
  return {.kind = kind, .left = &left, .right = &right};
}
consteval RecipeValue U(RecipeValueKind kind, const RecipeValue& value) {
  return {.kind = kind, .left = &value};
}
consteval RecipeValue Select(const RecipeValue& condition, const RecipeValue& yes,
                             const RecipeValue& no) {
  return {.kind = RecipeValueKind::kSelect, .left = &condition, .right = &yes, .alternate = &no};
}
constexpr RecipeParam P(std::string_view name, RecipeValue value,
                        std::string_view request_scalar = {}) {
  return {.name = name, .value = value, .request_scalar = request_scalar};
}
constexpr RecipeWeight W(std::string_view role, std::string_view tensor,
                         std::span<const RecipeValue> source_dims = {}) {
  return {.role = role, .tensor = tensor, .source_dims = source_dims};
}
constexpr RecipeRef V(std::string_view name) {
  return {.kind = RecipeRefKind::kNamedValue, .name = name};
}
constexpr RecipeRef StageInput() { return {.kind = RecipeRefKind::kStageInput}; }
constexpr RecipeValue Name(std::string_view value) { return {.literal = std::string(value)}; }

// Checkpoint evidence: inclusionAI/Ming-Image-0.1-Design@
// 208087ada1486931692c1896f38d4cd16ff3df82 (MIT weights). Decoder control
// flow is from the distinct inclusionAI/Ming-Image@
// 3515cd7b1e3049fc517eed7d35020251cd06519a companion revision; its
// diffusion/autoencoder_kl_qwenimage.py SHA-256 is
// 4013c3e09e3684d2ca8bbf1f98144a8ebab2a9408641831b1661d89b4a0aa106. The config
// is the source of every channel and affine scalar. Channel geometry is
// checked again against the rank-five safetensors header before a route is
// published. The source class has four decoder levels and three spatial
// upsamplers; the two temporal upsamplers are inactive only for the first
// frame after the source decoder's empty-history reset.
constexpr auto kLevels = SListSize("dim_mult");
constexpr auto kBase = S("base_dim");
constexpr auto kZ = S("z_dim");
constexpr auto kOut = S("input_channels");
constexpr auto kResBlocks = S("num_res_blocks");
constexpr auto kZero = I(0);
constexpr auto kOne = I(1);
constexpr auto kTwo = I(2);
constexpr auto kThree = I(3);
constexpr auto kRealOne = R(1.0);
constexpr auto kResCount = E(RecipeValueKind::kAdd, kResBlocks, kOne);
constexpr auto kMidMultiplier = SLR("dim_mult", "");
constexpr auto kMidChannels = E(RecipeValueKind::kMultiply, kBase, kMidMultiplier);
constexpr auto kOutMultiplier = SL("dim_mult", 0);
constexpr auto kOutChannels = E(RecipeValueKind::kMultiply, kBase, kOutMultiplier);
constexpr auto kLevel = Idx("level");
constexpr auto kResidual = Idx("residual");
constexpr auto kLastLevel = E(RecipeValueKind::kSubtract, kLevels, kOne);
constexpr auto kIsUpsample = E(RecipeValueKind::kLess, kLevel, kLastLevel);
constexpr auto kLevelZero = E(RecipeValueKind::kEqual, kLevel, kZero);
constexpr auto kLevelOutputMultiplier = SLR("dim_mult", "level");
constexpr auto kPreviousMultiplier = SLR("dim_mult", "level", -1);
constexpr auto kPreviousChannels = E(RecipeValueKind::kMultiply, kBase, kPreviousMultiplier);
constexpr auto kPreviousInputChannels = E(RecipeValueKind::kDivideFloor, kPreviousChannels, kTwo);
constexpr auto kCin = Select(kLevelZero, kMidChannels, kPreviousInputChannels);
constexpr auto kCout = E(RecipeValueKind::kMultiply, kBase, kLevelOutputMultiplier);
constexpr auto kFirstResidual = E(RecipeValueKind::kEqual, kResidual, kZero);
constexpr auto kResidualCin = Select(kFirstResidual, kCin, kCout);
constexpr auto kCinEqualsCout = E(RecipeValueKind::kEqual, kCin, kCout);
constexpr auto kCinDiffers = U(RecipeValueKind::kNot, kCinEqualsCout);
constexpr auto kNeedsShortcut = E(RecipeValueKind::kAnd, kFirstResidual, kCinDiffers);
constexpr auto kNextChannels = E(RecipeValueKind::kDivideFloor, kCout, kTwo);
const std::array kPostQuantDims{kZ, kZ, kOne, kOne, kOne};
const std::array kConvInDims{kMidChannels, kZ, kThree, kThree, kThree};
const std::array kMidConvDims{kMidChannels, kMidChannels, kThree, kThree, kThree};
const std::array kShortcutDims{kCout, kCin, kOne, kOne, kOne};
const std::array kUpConv1Dims{kCout, kResidualCin, kThree, kThree, kThree};
const std::array kUpConv2Dims{kCout, kCout, kThree, kThree, kThree};
const std::array kConvOutDims{kOut, kOutChannels, kThree, kThree, kThree};

const std::array kReshapeParams{P("width", I(1), "latent_width"),
                                P("height", I(1), "latent_height")};
const RecipeNode kReshape{.name = "vae.reshape",
                          .component_kind = "reshape.spatial",
                          .params = kReshapeParams,
                          .input = StageInput()};
constexpr auto kScalingFactor = S("scaling_factor");
constexpr auto kScale = E(RecipeValueKind::kRealDivide, kRealOne, kScalingFactor);
const std::array kAffineParams{P("scale", kScale), P("bias", S("shift_factor"))};
const RecipeNode kAffine{.name = "vae.affine",
                         .component_kind = "affine.scalar",
                         .params = kAffineParams,
                         .input = V("vae.reshape")};

const std::array kPostQuantParams{P("in_channels", kZ),    P("out_channels", kZ),
                                  P("kernel_width", kOne), P("kernel_height", kOne),
                                  P("kernel_depth", kOne), P("padding", kZero)};
const std::array kPostQuantWeights{W("kernel", "post_quant_conv.weight", kPostQuantDims),
                                   W("bias", "post_quant_conv.bias")};
const RecipeNode kPostQuant{.name = "vae.post_quant",
                            .component_kind = "conv.causal_first_frame",
                            .params = kPostQuantParams,
                            .weights = kPostQuantWeights,
                            .input = V("vae.affine")};

const std::array kConvInParams{P("in_channels", kZ),      P("out_channels", kMidChannels),
                               P("kernel_width", kThree), P("kernel_height", kThree),
                               P("kernel_depth", kThree), P("padding", I(1))};
const std::array kConvInWeights{W("kernel", "decoder.conv_in.weight", kConvInDims),
                                W("bias", "decoder.conv_in.bias")};
const RecipeNode kConvIn{.name = "vae.conv_in",
                         .component_kind = "conv.causal_first_frame",
                         .params = kConvInParams,
                         .weights = kConvInWeights,
                         .input = V("vae.post_quant")};

const std::array kMidNormParams{P("channels", kMidChannels), P("weight_rank", I(4)),
                                P("weight_axis", I(3)), P("silu", B(true)),
                                P("round_bf16", B(true))};
const std::array kMidNorm0Weights{W("weight", "decoder.mid_block.resnets.0.norm1.gamma")};
const RecipeNode kMidNorm0{.name = "vae.mid.0.norm1",
                           .component_kind = "norm.channel",
                           .params = kMidNormParams,
                           .weights = kMidNorm0Weights,
                           .input = V("vae.conv_in")};
const std::array kMidConvParams{P("in_channels", kMidChannels), P("out_channels", kMidChannels),
                                P("kernel_width", kThree),      P("kernel_height", kThree),
                                P("kernel_depth", kThree),      P("padding", I(1))};
const std::array kMidConv1Weights{
    W("kernel", "decoder.mid_block.resnets.0.conv1.weight", kMidConvDims),
    W("bias", "decoder.mid_block.resnets.0.conv1.bias")};
const RecipeNode kMidConv1{.name = "vae.mid.0.conv1",
                           .component_kind = "conv.causal_first_frame",
                           .params = kMidConvParams,
                           .weights = kMidConv1Weights,
                           .input = V("vae.mid.0.norm1")};
const std::array kMidNorm2Weights{W("weight", "decoder.mid_block.resnets.0.norm2.gamma")};
const RecipeNode kMidNorm2{.name = "vae.mid.0.norm2",
                           .component_kind = "norm.channel",
                           .params = kMidNormParams,
                           .weights = kMidNorm2Weights,
                           .input = V("vae.mid.0.conv1")};
const std::array kMidConv2Weights{
    W("kernel", "decoder.mid_block.resnets.0.conv2.weight", kMidConvDims),
    W("bias", "decoder.mid_block.resnets.0.conv2.bias")};
const RecipeNode kMidConv2{.name = "vae.mid.0.conv2",
                           .component_kind = "conv.causal_first_frame",
                           .params = kMidConvParams,
                           .weights = kMidConv2Weights,
                           .input = V("vae.mid.0.norm2")};
const RecipeNode kMidResidual0{.name = "vae.mid.0.out",
                               .component_kind = "residual.spatial",
                               .input = V("vae.mid.0.conv2"),
                               .auxiliary = V("vae.conv_in"),
                               .has_auxiliary = true};

const std::array kAttentionNormParams{P("channels", kMidChannels), P("weight_rank", I(3)),
                                      P("weight_axis", I(2)), P("silu", B(false)),
                                      P("round_bf16", B(true))};
const std::array kAttentionNormWeights{W("weight", "decoder.mid_block.attentions.0.norm.gamma")};
const RecipeNode kAttentionNorm{.name = "vae.mid.attn.norm",
                                .component_kind = "norm.channel",
                                .params = kAttentionNormParams,
                                .weights = kAttentionNormWeights,
                                .input = V("vae.mid.0.out")};
constexpr auto kTripleMid = E(RecipeValueKind::kMultiply, kMidChannels, kThree);
const std::array kQkvParams{P("in_channels", kMidChannels), P("out_channels", kTripleMid),
                            P("kernel", kOne), P("padding", kZero)};
const std::array kQkvWeights{W("kernel", "decoder.mid_block.attentions.0.to_qkv.weight"),
                             W("bias", "decoder.mid_block.attentions.0.to_qkv.bias")};
const RecipeNode kQkv{.name = "vae.mid.attn.qkv",
                      .component_kind = "conv.spatial",
                      .params = kQkvParams,
                      .weights = kQkvWeights,
                      .input = V("vae.mid.attn.norm")};
const RecipeNode kAttention{
    .name = "vae.mid.attn", .component_kind = "attention.spatial", .input = V("vae.mid.attn.qkv")};
const std::array kAttentionProjParams{P("in_channels", kMidChannels),
                                      P("out_channels", kMidChannels), P("kernel", kOne),
                                      P("padding", kZero)};
const std::array kAttentionProjWeights{W("kernel", "decoder.mid_block.attentions.0.proj.weight"),
                                       W("bias", "decoder.mid_block.attentions.0.proj.bias")};
const RecipeNode kAttentionProj{.name = "vae.mid.attn.proj",
                                .component_kind = "conv.spatial",
                                .params = kAttentionProjParams,
                                .weights = kAttentionProjWeights,
                                .input = V("vae.mid.attn")};
const RecipeNode kAttentionResidual{.name = "vae.mid.attn.out",
                                    .component_kind = "residual.spatial",
                                    .input = V("vae.mid.attn.proj"),
                                    .auxiliary = V("vae.mid.0.out"),
                                    .has_auxiliary = true};

const std::array kMid1Norm1Weights{W("weight", "decoder.mid_block.resnets.1.norm1.gamma")};
const RecipeNode kMid1Norm1{.name = "vae.mid.1.norm1",
                            .component_kind = "norm.channel",
                            .params = kMidNormParams,
                            .weights = kMid1Norm1Weights,
                            .input = V("vae.mid.attn.out")};
const std::array kMid1Conv1Weights{
    W("kernel", "decoder.mid_block.resnets.1.conv1.weight", kMidConvDims),
    W("bias", "decoder.mid_block.resnets.1.conv1.bias")};
const RecipeNode kMid1Conv1{.name = "vae.mid.1.conv1",
                            .component_kind = "conv.causal_first_frame",
                            .params = kMidConvParams,
                            .weights = kMid1Conv1Weights,
                            .input = V("vae.mid.1.norm1")};
const std::array kMid1Norm2Weights{W("weight", "decoder.mid_block.resnets.1.norm2.gamma")};
const RecipeNode kMid1Norm2{.name = "vae.mid.1.norm2",
                            .component_kind = "norm.channel",
                            .params = kMidNormParams,
                            .weights = kMid1Norm2Weights,
                            .input = V("vae.mid.1.conv1")};
const std::array kMid1Conv2Weights{
    W("kernel", "decoder.mid_block.resnets.1.conv2.weight", kMidConvDims),
    W("bias", "decoder.mid_block.resnets.1.conv2.bias")};
const RecipeNode kMid1Conv2{.name = "vae.mid.1.conv2",
                            .component_kind = "conv.causal_first_frame",
                            .params = kMidConvParams,
                            .weights = kMid1Conv2Weights,
                            .input = V("vae.mid.1.norm2")};
const RecipeNode kMidResidual1{.name = "vae.mid.1.out",
                               .component_kind = "residual.spatial",
                               .input = V("vae.mid.1.conv2"),
                               .auxiliary = V("vae.mid.attn.out"),
                               .has_auxiliary = true};

const std::array kShortcutParams{P("in_channels", kCin),  P("out_channels", kCout),
                                 P("kernel_width", kOne), P("kernel_height", kOne),
                                 P("kernel_depth", kOne), P("padding", kZero)};
const std::array kShortcutWeights{
    W("kernel", "decoder.up_blocks.{level}.resnets.{residual}.conv_shortcut.weight", kShortcutDims),
    W("bias", "decoder.up_blocks.{level}.resnets.{residual}.conv_shortcut.bias")};
const RecipeNode kShortcut{.name = "vae.up.{level}.{residual}.shortcut",
                           .component_kind = "conv.causal_first_frame",
                           .params = kShortcutParams,
                           .weights = kShortcutWeights,
                           .input = V("vae.up.{level}.{residual}.source"),
                           .group = RecipeGroupKind::kLayer,
                           .group_index = kLevel};
const std::array kShortcutSteps{RecipeStep{.kind = RecipeStep::Kind::kNode, .node = &kShortcut}};
const RecipeIf kShortcutIf{.condition = kNeedsShortcut,
                           .then_steps = kShortcutSteps,
                           .input = V("vae.up.{level}.{residual}.stream"),
                           .then_yield = "vae.up.{level}.{residual}.shortcut",
                           .else_yield = "vae.up.{level}.{residual}.source",
                           .yield = "vae.up.{level}.{residual}.skip",
                           .input_name = "vae.up.{level}.{residual}.source"};
const RecipeStep kShortcutStep{.kind = RecipeStep::Kind::kIf, .branch = &kShortcutIf};

constexpr auto kResidualNormChannels = kResidualCin;
const std::array kUpNorm1Params{P("channels", kResidualNormChannels), P("weight_rank", I(4)),
                                P("weight_axis", I(3)), P("silu", B(true)),
                                P("round_bf16", B(true))};
const std::array kUpNorm1Weights{
    W("weight", "decoder.up_blocks.{level}.resnets.{residual}.norm1.gamma")};
const RecipeNode kUpNorm1{.name = "vae.up.{level}.{residual}.norm1",
                          .component_kind = "norm.channel",
                          .params = kUpNorm1Params,
                          .weights = kUpNorm1Weights,
                          .input = V("vae.up.{level}.{residual}.stream"),
                          .group = RecipeGroupKind::kLayer,
                          .group_index = kLevel};
const std::array kUpConv1Params{P("in_channels", kResidualNormChannels),
                                P("out_channels", kCout),
                                P("kernel_width", kThree),
                                P("kernel_height", kThree),
                                P("kernel_depth", kThree),
                                P("padding", I(1))};
const std::array kUpConv1Weights{
    W("kernel", "decoder.up_blocks.{level}.resnets.{residual}.conv1.weight", kUpConv1Dims),
    W("bias", "decoder.up_blocks.{level}.resnets.{residual}.conv1.bias")};
const RecipeNode kUpConv1{.name = "vae.up.{level}.{residual}.conv1",
                          .component_kind = "conv.causal_first_frame",
                          .params = kUpConv1Params,
                          .weights = kUpConv1Weights,
                          .input = V("vae.up.{level}.{residual}.norm1"),
                          .group = RecipeGroupKind::kLayer,
                          .group_index = kLevel};
const std::array kUpNorm2Params{P("channels", kCout), P("weight_rank", I(4)),
                                P("weight_axis", I(3)), P("silu", B(true)),
                                P("round_bf16", B(true))};
const std::array kUpNorm2Weights{
    W("weight", "decoder.up_blocks.{level}.resnets.{residual}.norm2.gamma")};
const RecipeNode kUpNorm2{.name = "vae.up.{level}.{residual}.norm2",
                          .component_kind = "norm.channel",
                          .params = kUpNorm2Params,
                          .weights = kUpNorm2Weights,
                          .input = V("vae.up.{level}.{residual}.conv1"),
                          .group = RecipeGroupKind::kLayer,
                          .group_index = kLevel};
const std::array kUpConv2Params{P("in_channels", kCout),   P("out_channels", kCout),
                                P("kernel_width", kThree), P("kernel_height", kThree),
                                P("kernel_depth", kThree), P("padding", I(1))};
const std::array kUpConv2Weights{
    W("kernel", "decoder.up_blocks.{level}.resnets.{residual}.conv2.weight", kUpConv2Dims),
    W("bias", "decoder.up_blocks.{level}.resnets.{residual}.conv2.bias")};
const RecipeNode kUpConv2{.name = "vae.up.{level}.{residual}.conv2",
                          .component_kind = "conv.causal_first_frame",
                          .params = kUpConv2Params,
                          .weights = kUpConv2Weights,
                          .input = V("vae.up.{level}.{residual}.norm2"),
                          .group = RecipeGroupKind::kLayer,
                          .group_index = kLevel};
const RecipeNode kUpAdd{.name = "vae.up.{level}.{residual}.out",
                        .component_kind = "residual.spatial",
                        .input = V("vae.up.{level}.{residual}.conv2"),
                        .auxiliary = V("vae.up.{level}.{residual}.skip"),
                        .has_auxiliary = true};
const std::array kResidualSteps{kShortcutStep,
                                RecipeStep{.kind = RecipeStep::Kind::kNode, .node = &kUpNorm1},
                                RecipeStep{.kind = RecipeStep::Kind::kNode, .node = &kUpConv1},
                                RecipeStep{.kind = RecipeStep::Kind::kNode, .node = &kUpNorm2},
                                RecipeStep{.kind = RecipeStep::Kind::kNode, .node = &kUpConv2},
                                RecipeStep{.kind = RecipeStep::Kind::kNode, .node = &kUpAdd}};
const RecipeRepeat kResidualRepeat{.index = "residual",
                                   .count = kResCount,
                                   .body = kResidualSteps,
                                   .input = V("vae.up.entry.{level}"),
                                   .input_name = "vae.up.{level}.{residual}.stream",
                                   .body_yield = "vae.up.{level}.{residual}.out",
                                   .yield = "vae.up.result.{level}"};

const std::array kNearestParams{P("factor", I(2))};
const RecipeNode kNearest{.name = "vae.up.{level}.nearest",
                          .component_kind = "upsample.nearest",
                          .params = kNearestParams,
                          .input = V("vae.up.result.{level}")};
const std::array kUpSampleConvParams{P("in_channels", kCout), P("out_channels", kNextChannels),
                                     P("kernel", kThree), P("padding", I(1))};
const std::array kUpSampleConvWeights{
    W("kernel", "decoder.up_blocks.{level}.upsamplers.0.resample.1.weight"),
    W("bias", "decoder.up_blocks.{level}.upsamplers.0.resample.1.bias")};
const RecipeNode kUpSampleConv{.name = "vae.up.{level}.conv",
                               .component_kind = "conv.spatial",
                               .params = kUpSampleConvParams,
                               .weights = kUpSampleConvWeights,
                               .input = V("vae.up.{level}.nearest"),
                               .group = RecipeGroupKind::kLayer,
                               .group_index = kLevel};
const std::array kUpsampleSteps{
    RecipeStep{.kind = RecipeStep::Kind::kRepeat, .repeat = &kResidualRepeat},
    RecipeStep{.kind = RecipeStep::Kind::kNode, .node = &kNearest},
    RecipeStep{.kind = RecipeStep::Kind::kNode, .node = &kUpSampleConv}};
const std::array kLastLevelSteps{
    RecipeStep{.kind = RecipeStep::Kind::kRepeat, .repeat = &kResidualRepeat}};
const RecipeIf kLevelIf{.condition = kIsUpsample,
                        .then_steps = kUpsampleSteps,
                        .else_steps = kLastLevelSteps,
                        .input = V("vae.up.current.{level}"),
                        .then_yield = "vae.up.{level}.conv",
                        .else_yield = "vae.up.result.{level}",
                        .yield = "vae.up.next.{level}",
                        .input_name = "vae.up.entry.{level}"};
const RecipeStep kLevelIfStep{.kind = RecipeStep::Kind::kIf, .branch = &kLevelIf};
const std::array kLevelSteps{kLevelIfStep};
const RecipeRepeat kLevelRepeat{.index = "level",
                                .count = kLevels,
                                .body = kLevelSteps,
                                .input = V("vae.mid.1.out"),
                                .input_name = "vae.up.current.{level}",
                                .body_yield = "vae.up.next.{level}",
                                .yield = "vae.up.final"};
const RecipeStep kLevelRepeatStep{.kind = RecipeStep::Kind::kRepeat, .repeat = &kLevelRepeat};

const std::array kNormOutParams{P("channels", kOutChannels), P("weight_rank", I(4)),
                                P("weight_axis", I(3)), P("silu", B(true)),
                                P("round_bf16", B(true))};
const std::array kNormOutWeights{W("weight", "decoder.norm_out.gamma")};
const RecipeNode kNormOut{.name = "vae.norm_out",
                          .component_kind = "norm.channel",
                          .params = kNormOutParams,
                          .weights = kNormOutWeights,
                          .input = V("vae.up.final")};
const std::array kConvOutParams{P("in_channels", kOutChannels), P("out_channels", kOut),
                                P("kernel_width", kThree),      P("kernel_height", kThree),
                                P("kernel_depth", kThree),      P("padding", I(1))};
const std::array kConvOutWeights{W("kernel", "decoder.conv_out.weight", kConvOutDims),
                                 W("bias", "decoder.conv_out.bias")};
const RecipeNode kConvOut{.name = "vae.output",
                          .component_kind = "conv.causal_first_frame",
                          .params = kConvOutParams,
                          .weights = kConvOutWeights,
                          .input = V("vae.norm_out")};
const auto kHardTanh = Name("hardtanh");
const std::array kClampParams{P("activation", kHardTanh)};
const RecipeNode kClamp{.name = "vae.output.clamp",
                        .component_kind = "activation.elementwise",
                        .params = kClampParams,
                        .input = V("vae.output")};

const std::array kSteps{RecipeStep{.kind = RecipeStep::Kind::kNode, .node = &kReshape},
                        RecipeStep{.kind = RecipeStep::Kind::kNode, .node = &kAffine},
                        RecipeStep{.kind = RecipeStep::Kind::kNode, .node = &kPostQuant},
                        RecipeStep{.kind = RecipeStep::Kind::kNode, .node = &kConvIn},
                        RecipeStep{.kind = RecipeStep::Kind::kNode, .node = &kMidNorm0},
                        RecipeStep{.kind = RecipeStep::Kind::kNode, .node = &kMidConv1},
                        RecipeStep{.kind = RecipeStep::Kind::kNode, .node = &kMidNorm2},
                        RecipeStep{.kind = RecipeStep::Kind::kNode, .node = &kMidConv2},
                        RecipeStep{.kind = RecipeStep::Kind::kNode, .node = &kMidResidual0},
                        RecipeStep{.kind = RecipeStep::Kind::kNode, .node = &kAttentionNorm},
                        RecipeStep{.kind = RecipeStep::Kind::kNode, .node = &kQkv},
                        RecipeStep{.kind = RecipeStep::Kind::kNode, .node = &kAttention},
                        RecipeStep{.kind = RecipeStep::Kind::kNode, .node = &kAttentionProj},
                        RecipeStep{.kind = RecipeStep::Kind::kNode, .node = &kAttentionResidual},
                        RecipeStep{.kind = RecipeStep::Kind::kNode, .node = &kMid1Norm1},
                        RecipeStep{.kind = RecipeStep::Kind::kNode, .node = &kMid1Conv1},
                        RecipeStep{.kind = RecipeStep::Kind::kNode, .node = &kMid1Norm2},
                        RecipeStep{.kind = RecipeStep::Kind::kNode, .node = &kMid1Conv2},
                        RecipeStep{.kind = RecipeStep::Kind::kNode, .node = &kMidResidual1},
                        kLevelRepeatStep,
                        RecipeStep{.kind = RecipeStep::Kind::kNode, .node = &kNormOut},
                        RecipeStep{.kind = RecipeStep::Kind::kNode, .node = &kConvOut},
                        RecipeStep{.kind = RecipeStep::Kind::kNode, .node = &kClamp}};
const StageRecipe kVaeRecipe{
    .name = "vae",
    .role = StageRole::kVae,
    .artifact = 0,
    .stream_width = kZ,
    .primary_input = {.shape = {.width = I(1), .height = I(1), .channels = kZ, .images = I(1)}},
    .steps = kSteps};

constexpr std::array<std::string_view, 4> kIgnoredTensors{
    "encoder.", "quant_conv.", "decoder.up_blocks.{layer}.upsamplers.0.time_conv.weight",
    "decoder.up_blocks.{layer}.upsamplers.0.time_conv.bias"};

const Mapping kMapping{.architecture = "qwen_image_vae",
                       .source_architectures = kSourceArchitectures,
                       .version = 1,
                       .pipeline = PipelineKind::kImage,
                       .role = StageRole::kVae,
                       .stage = "vae",
                       .stage_recipe = &kVaeRecipe,
                       .containers = kContainers,
                       .ignored_tensors = kIgnoredTensors};

}  // namespace

const Mapping& MingVaeMapping() { return kMapping; }

}  // namespace llmp::spec::mappings
