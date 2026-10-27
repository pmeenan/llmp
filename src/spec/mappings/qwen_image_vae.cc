// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "spec/mappings/qwen_image_vae.h"

#include <array>
#include <cstdint>
#include <span>
#include <string_view>

#include "spec/expr.h"
#include "spec/stage_recipe.h"

namespace llmp::spec::mappings {
namespace {

using ex::AtBack;
using ex::B;
using ex::E;
using ex::First;
using ex::I;
using ex::Idx;
using ex::Present;
using ex::R;
using ex::Reverse;
using ex::S;
using ex::Select;
using ex::TensorRank;
using ex::U;
using ex::V;
using K = ExprKind;

constexpr RecipeParam P(std::string_view name, const RecipeValue& value,
                        const RecipeValue* when = nullptr, std::string_view scalar = {}) {
  return {.name = name, .value = value, .request_scalar = scalar, .when = when};
}
constexpr RecipeWeight W(std::string_view role, std::string_view tensor,
                         std::span<const RecipeValue> source_dims = {},
                         const RecipeValue* source_dims_when = nullptr) {
  return {.role = role,
          .tensor = tensor,
          .source_dims = source_dims,
          .source_dims_when = source_dims_when};
}
constexpr RecipeRef Link(std::string_view name) {
  return {.kind = RecipeRefKind::kNamedValue, .name = name};
}
constexpr RecipeRef Port(std::uint16_t port) {
  return {.kind = RecipeRefKind::kHostPort, .port = port};
}
constexpr RecipeRef Input() { return {.kind = RecipeRefKind::kStageInput}; }
constexpr RecipeStep Node(const RecipeNode& node) {
  return {.kind = RecipeStep::Kind::kNode, .node = &node};
}

// Checkpoint evidence. Qwen-Image-2.1's VAE (Qwen/Qwen-Image-2.1@
// 790c92633540aa0cb11d9abf19eb46d861714758, AutoencoderKLQwenImage21): 2-D
// kernels, is_residual, decoder_base_dim. Ming-Image's
// (inclusionAI/Ming-Image-0.1-Design@208087ada1486931692c1896f38d4cd16ff3df82,
// MIT weights, AutoencoderKLQwenImage): causal 3-D kernels, halving
// upsamplers, base_dim and a scalar latent scaling; its decoder control flow
// is from inclusionAI/Ming-Image@3515cd7b1e3049fc517eed7d35020251cd06519a's
// diffusion/autoencoder_kl_qwenimage.py (SHA-256
// 4013c3e09e3684d2ca8bbf1f98144a8ebab2a9408641831b1661d89b4a0aa106), whose
// two temporal upsamplers are inactive for the first frame after the
// decoder's empty-history reset. Channel geometry is checked again against
// the safetensors header before a route is published.

// ------------------------------------------------------------- the values
// Each the pipeline family's parameter where the import has one, else the
// config's key (the first of its spellings it states).
constexpr auto kFamilyZ = V("image.vae_decoder.z_dim");
constexpr auto kConfigZ = V("z_dim");
constexpr auto kZ = First(kFamilyZ, kConfigZ);
constexpr auto kFamilyBase = V("image.vae_decoder.base");
constexpr auto kDecoderBaseKey = V("decoder_base_dim");
constexpr auto kBaseKey = V("base_dim");
constexpr auto kConfigBase = First(kDecoderBaseKey, kBaseKey);
constexpr auto kBase = First(kFamilyBase, kConfigBase);
constexpr auto kFamilyMult = V("image.vae_decoder.dim_mult");
constexpr auto kConfigMult = V("dim_mult");
constexpr auto kMult = First(kFamilyMult, kConfigMult);
constexpr auto kFamilyResBlocks = V("image.vae_decoder.res_blocks");
constexpr auto kConfigResBlocks = V("num_res_blocks");
constexpr auto kResBlocks = First(kFamilyResBlocks, kConfigResBlocks);
constexpr auto kFamilyOut = V("image.vae_decoder.out_channels");
constexpr auto kOutKey = V("out_channels");
constexpr auto kInputChannelsKey = V("input_channels");  // Ming's: the image's channels
constexpr auto kConfigOut = First(kOutKey, kInputChannelsKey);
constexpr auto kOut = First(kFamilyOut, kConfigOut);
constexpr auto kFamilySpatial = V("image.vae_decoder.spatial_factor");
constexpr auto kConfigSpatial = V("scale_factor_spatial");
constexpr auto kSpatialFactor = First(kFamilySpatial, kConfigSpatial);
// The up blocks' temporal steps: the encoder's temporal downsampling reversed.
constexpr auto kFamilyTemporal = V("image.vae_decoder.temporal_upsample");
constexpr auto kTemporalDownKey = V("temperal_downsample");
constexpr auto kConfigTemporal = Reverse(kTemporalDownKey);
constexpr auto kTemporal = First(kFamilyTemporal, kConfigTemporal);

// The dialects, as the source says them. A per-channel latent affine (the
// family's latent statistics, a host input) where the source has them, else
// the config's scalar scaling.
constexpr auto kFamilyMean = V("image.vae_decoder.latents_mean");
constexpr auto kConfigMean = V("latents_mean");
constexpr auto kMean = First(kFamilyMean, kConfigMean);
constexpr auto kFamilyStd = V("image.vae_decoder.latents_std");
constexpr auto kConfigStd = V("latents_std");
constexpr auto kStd = First(kFamilyStd, kConfigStd);
constexpr auto kHasMean = Present(kMean);
constexpr auto kHasStd = Present(kStd);
constexpr auto kPerChannel = E(K::kAnd, kHasMean, kHasStd);
// A residual decoder's up path (Wan 2.2's is_residual; absent: false). The
// other's upsamplers halve their channels, its norms round to BF16 and its
// output is clamped, as its source decoder computes them.
constexpr auto kResidualKey = V("is_residual");
constexpr auto kFalse = B(false);
constexpr auto kResidual = First(kResidualKey, kFalse);
constexpr auto kHalving = U(K::kNot, kResidual);
// Causal 3-D kernels (rank five) or 2-D ones.
constexpr auto kKernelRank = TensorRank("decoder.conv_in.weight");
constexpr auto kFive = I(5);
constexpr auto kCausal = E(K::kEqual, kKernelRank, kFive);

constexpr auto kZero = I(0);
constexpr auto kOne = I(1);
constexpr auto kTwo = I(2);
constexpr auto kThree = I(3);
constexpr auto kLevels = ex::Size(kMult);
constexpr auto kOutMultiplier = ex::At(kMult, 0);
constexpr auto kOutChannels = E(K::kMultiply, kBase, kOutMultiplier);
constexpr auto kMidMultiplier = AtBack(kMult, 0);
constexpr auto kMidChannels = E(K::kMultiply, kBase, kMidMultiplier);
constexpr auto kTripleMid = E(K::kMultiply, kMidChannels, kThree);
constexpr auto kLevel = Idx("level");
constexpr auto kResidualIndex = Idx("residual");
constexpr auto kLastLevel = E(K::kSubtract, kLevels, kOne);
constexpr auto kIsUp = E(K::kLess, kLevel, kLastLevel);
constexpr auto kLevelZero = E(K::kEqual, kLevel, kZero);
constexpr auto kPreviousMultiplier = AtBack(kMult, "level", -1);
constexpr auto kPreviousChannels = E(K::kMultiply, kBase, kPreviousMultiplier);
constexpr auto kPreviousHalf = E(K::kDivideFloor, kPreviousChannels, kTwo);
constexpr auto kPreviousInput = Select(kResidual, kPreviousChannels, kPreviousHalf);
constexpr auto kCin = Select(kLevelZero, kMidChannels, kPreviousInput);
constexpr auto kLevelMultiplier = AtBack(kMult, "level");
constexpr auto kCout = E(K::kMultiply, kBase, kLevelMultiplier);
constexpr auto kHalfCout = E(K::kDivideFloor, kCout, kTwo);
constexpr auto kFirstResidual = E(K::kEqual, kResidualIndex, kZero);
constexpr auto kResidualCin = Select(kFirstResidual, kCin, kCout);
constexpr auto kCinEqualsCout = E(K::kEqual, kCin, kCout);
constexpr auto kCinDiffers = U(K::kNot, kCinEqualsCout);
constexpr auto kNeedsShortcut = E(K::kAnd, kFirstResidual, kCinDiffers);
constexpr auto kResCount = E(K::kAdd, kResBlocks, kOne);
constexpr auto kRealOne = R(1.0);
constexpr auto kScalingKey = V("scaling_factor");
constexpr auto kScale = E(K::kRealDivide, kRealOne, kScalingKey);
constexpr auto kShiftKey = V("shift_factor");
constexpr auto kTrue = B(true);

// A convolution's parameters: 2-D, or (causal) its kernel's temporal taps,
// of which a first frame reads the last.
#define VAE_CONV_PARAMS(name, in, out, kernel, padding)                                       \
  constexpr std::array name {                                                                 \
    P("in_channels", in), P("out_channels", out), P("kernel", kernel), P("padding", padding), \
        P("temporal_kernel", kernel, &kCausal)                                                \
  }
// A causal kernel's header dimensions [out, in, t, k, k].
#define VAE_CONV_DIMS(name, in, out, kernel) \
  constexpr std::array name { out, in, kernel, kernel, kernel }

// ------------------------------------------------------- the decoder head
const std::array kReshapeParams{P("width", kOne, nullptr, "latent_width"),
                                P("height", kOne, nullptr, "latent_height")};
const RecipeNode kReshape{.name = "vae.reshape",
                          .component_kind = "reshape.spatial",
                          .params = kReshapeParams,
                          .input = Input()};
const RecipeNode kChannelAffine{.name = "vae.affine.channel",
                                .component_kind = "affine.channel",
                                .input = Link("vae.affine.input"),
                                .auxiliary = Port(0),
                                .has_auxiliary = true};
const std::array kScalarAffineParams{P("scale", kScale), P("bias", kShiftKey)};
const RecipeNode kScalarAffine{.name = "vae.affine.scalar",
                               .component_kind = "affine.scalar",
                               .params = kScalarAffineParams,
                               .input = Link("vae.affine.input")};
const std::array kChannelAffineSteps{Node(kChannelAffine)};
const std::array kScalarAffineSteps{Node(kScalarAffine)};
const RecipeIf kAffineIf{.condition = kPerChannel,
                         .then_steps = kChannelAffineSteps,
                         .else_steps = kScalarAffineSteps,
                         .input = Link("vae.reshape"),
                         .then_yield = "vae.affine.channel",
                         .else_yield = "vae.affine.scalar",
                         .yield = "vae.affine",
                         .input_name = "vae.affine.input"};

VAE_CONV_PARAMS(kPostQuantParams, kZ, kZ, kOne, kZero);
VAE_CONV_DIMS(kPostQuantDims, kZ, kZ, kOne);
const std::array kPostQuantWeights{W("kernel", "post_quant_conv.weight", kPostQuantDims, &kCausal),
                                   W("bias", "post_quant_conv.bias")};
const RecipeNode kPostQuant{.name = "vae.post_quant",
                            .component_kind = "conv.spatial",
                            .params = kPostQuantParams,
                            .weights = kPostQuantWeights,
                            .input = Link("vae.affine")};
VAE_CONV_PARAMS(kConvInParams, kZ, kMidChannels, kThree, kOne);
VAE_CONV_DIMS(kConvInDims, kZ, kMidChannels, kThree);
const std::array kConvInWeights{W("kernel", "decoder.conv_in.weight", kConvInDims, &kCausal),
                                W("bias", "decoder.conv_in.bias")};
const RecipeNode kConvIn{.name = "vae.conv_in",
                         .component_kind = "conv.spatial",
                         .params = kConvInParams,
                         .weights = kConvInWeights,
                         .input = Link("vae.post_quant")};

// ------------------------------------------------------------- the middle
const std::array kMidNormParams{P("channels", kMidChannels), P("weight_rank", I(4)),
                                P("weight_axis", kThree), P("silu", kTrue),
                                P("round_bf16", kTrue, &kHalving)};
VAE_CONV_PARAMS(kMidConvParams, kMidChannels, kMidChannels, kThree, kOne);
VAE_CONV_DIMS(kMidConvDims, kMidChannels, kMidChannels, kThree);

const std::array kMid0Norm1Weights{W("weight", "decoder.mid_block.resnets.0.norm1.gamma")};
const RecipeNode kMid0Norm1{.name = "vae.mid.0.norm1",
                            .component_kind = "norm.channel",
                            .params = kMidNormParams,
                            .weights = kMid0Norm1Weights,
                            .input = Link("vae.conv_in")};
const std::array kMid0Conv1Weights{
    W("kernel", "decoder.mid_block.resnets.0.conv1.weight", kMidConvDims, &kCausal),
    W("bias", "decoder.mid_block.resnets.0.conv1.bias")};
const RecipeNode kMid0Conv1{.name = "vae.mid.0.conv1",
                            .component_kind = "conv.spatial",
                            .params = kMidConvParams,
                            .weights = kMid0Conv1Weights,
                            .input = Link("vae.mid.0.norm1")};
const std::array kMid0Norm2Weights{W("weight", "decoder.mid_block.resnets.0.norm2.gamma")};
const RecipeNode kMid0Norm2{.name = "vae.mid.0.norm2",
                            .component_kind = "norm.channel",
                            .params = kMidNormParams,
                            .weights = kMid0Norm2Weights,
                            .input = Link("vae.mid.0.conv1")};
const std::array kMid0Conv2Weights{
    W("kernel", "decoder.mid_block.resnets.0.conv2.weight", kMidConvDims, &kCausal),
    W("bias", "decoder.mid_block.resnets.0.conv2.bias")};
const RecipeNode kMid0Conv2{.name = "vae.mid.0.conv2",
                            .component_kind = "conv.spatial",
                            .params = kMidConvParams,
                            .weights = kMid0Conv2Weights,
                            .input = Link("vae.mid.0.norm2")};
const RecipeNode kMid0Add{.name = "vae.mid.0.out",
                          .component_kind = "residual.spatial",
                          .input = Link("vae.mid.0.conv2"),
                          .auxiliary = Link("vae.conv_in"),
                          .has_auxiliary = true};

const std::array kAttentionNormParams{P("channels", kMidChannels), P("weight_rank", kThree),
                                      P("weight_axis", kTwo), P("silu", kFalse),
                                      P("round_bf16", kTrue, &kHalving)};
const std::array kAttentionNormWeights{W("weight", "decoder.mid_block.attentions.0.norm.gamma")};
const RecipeNode kAttentionNorm{.name = "vae.mid.attn.norm",
                                .component_kind = "norm.channel",
                                .params = kAttentionNormParams,
                                .weights = kAttentionNormWeights,
                                .input = Link("vae.mid.0.out")};
const std::array kQkvParams{P("in_channels", kMidChannels), P("out_channels", kTripleMid),
                            P("kernel", kOne), P("padding", kZero)};
const std::array kQkvWeights{W("kernel", "decoder.mid_block.attentions.0.to_qkv.weight"),
                             W("bias", "decoder.mid_block.attentions.0.to_qkv.bias")};
const RecipeNode kQkv{.name = "vae.mid.attn.qkv",
                      .component_kind = "conv.spatial",
                      .params = kQkvParams,
                      .weights = kQkvWeights,
                      .input = Link("vae.mid.attn.norm")};
const RecipeNode kAttention{.name = "vae.mid.attn",
                            .component_kind = "attention.spatial",
                            .input = Link("vae.mid.attn.qkv")};
const std::array kProjParams{P("in_channels", kMidChannels), P("out_channels", kMidChannels),
                             P("kernel", kOne), P("padding", kZero)};
const std::array kProjWeights{W("kernel", "decoder.mid_block.attentions.0.proj.weight"),
                              W("bias", "decoder.mid_block.attentions.0.proj.bias")};
const RecipeNode kProj{.name = "vae.mid.attn.proj",
                       .component_kind = "conv.spatial",
                       .params = kProjParams,
                       .weights = kProjWeights,
                       .input = Link("vae.mid.attn")};
const RecipeNode kAttentionAdd{.name = "vae.mid.attn.out",
                               .component_kind = "residual.spatial",
                               .input = Link("vae.mid.attn.proj"),
                               .auxiliary = Link("vae.mid.0.out"),
                               .has_auxiliary = true};

const std::array kMid1Norm1Weights{W("weight", "decoder.mid_block.resnets.1.norm1.gamma")};
const RecipeNode kMid1Norm1{.name = "vae.mid.1.norm1",
                            .component_kind = "norm.channel",
                            .params = kMidNormParams,
                            .weights = kMid1Norm1Weights,
                            .input = Link("vae.mid.attn.out")};
const std::array kMid1Conv1Weights{
    W("kernel", "decoder.mid_block.resnets.1.conv1.weight", kMidConvDims, &kCausal),
    W("bias", "decoder.mid_block.resnets.1.conv1.bias")};
const RecipeNode kMid1Conv1{.name = "vae.mid.1.conv1",
                            .component_kind = "conv.spatial",
                            .params = kMidConvParams,
                            .weights = kMid1Conv1Weights,
                            .input = Link("vae.mid.1.norm1")};
const std::array kMid1Norm2Weights{W("weight", "decoder.mid_block.resnets.1.norm2.gamma")};
const RecipeNode kMid1Norm2{.name = "vae.mid.1.norm2",
                            .component_kind = "norm.channel",
                            .params = kMidNormParams,
                            .weights = kMid1Norm2Weights,
                            .input = Link("vae.mid.1.conv1")};
const std::array kMid1Conv2Weights{
    W("kernel", "decoder.mid_block.resnets.1.conv2.weight", kMidConvDims, &kCausal),
    W("bias", "decoder.mid_block.resnets.1.conv2.bias")};
const RecipeNode kMid1Conv2{.name = "vae.mid.1.conv2",
                            .component_kind = "conv.spatial",
                            .params = kMidConvParams,
                            .weights = kMid1Conv2Weights,
                            .input = Link("vae.mid.1.norm2")};
const RecipeNode kMid1Add{.name = "vae.mid.1.out",
                          .component_kind = "residual.spatial",
                          .input = Link("vae.mid.1.conv2"),
                          .auxiliary = Link("vae.mid.attn.out"),
                          .has_auxiliary = true};

// -------------------------------------------------------- the up blocks
VAE_CONV_PARAMS(kShortcutParams, kCin, kCout, kOne, kZero);
VAE_CONV_DIMS(kShortcutDims, kCin, kCout, kOne);
const std::array kShortcutWeights{
    W("kernel", "decoder.up_blocks.{level}.resnets.{residual}.conv_shortcut.weight", kShortcutDims,
      &kCausal),
    W("bias", "decoder.up_blocks.{level}.resnets.{residual}.conv_shortcut.bias")};
const RecipeNode kShortcut{.name = "vae.up.{level}.{residual}.shortcut",
                           .component_kind = "conv.spatial",
                           .params = kShortcutParams,
                           .weights = kShortcutWeights,
                           .input = Link("vae.up.{level}.{residual}.source"),
                           .group = RecipeGroupKind::kLayer,
                           .group_index = kLevel};
const std::array kShortcutSteps{Node(kShortcut)};
const RecipeIf kShortcutIf{.condition = kNeedsShortcut,
                           .then_steps = kShortcutSteps,
                           .input = Link("vae.up.{level}.{residual}.stream"),
                           .then_yield = "vae.up.{level}.{residual}.shortcut",
                           .else_yield = "vae.up.{level}.{residual}.source",
                           .yield = "vae.up.{level}.{residual}.skip",
                           .input_name = "vae.up.{level}.{residual}.source"};

const std::array kUpNorm1Params{P("channels", kResidualCin), P("weight_rank", I(4)),
                                P("weight_axis", kThree), P("silu", kTrue),
                                P("round_bf16", kTrue, &kHalving)};
const std::array kUpNorm1Weights{
    W("weight", "decoder.up_blocks.{level}.resnets.{residual}.norm1.gamma")};
const RecipeNode kUpNorm1{.name = "vae.up.{level}.{residual}.norm1",
                          .component_kind = "norm.channel",
                          .params = kUpNorm1Params,
                          .weights = kUpNorm1Weights,
                          .input = Link("vae.up.{level}.{residual}.stream"),
                          .group = RecipeGroupKind::kLayer,
                          .group_index = kLevel};
VAE_CONV_PARAMS(kUpConv1Params, kResidualCin, kCout, kThree, kOne);
VAE_CONV_DIMS(kUpConv1Dims, kResidualCin, kCout, kThree);
const std::array kUpConv1Weights{
    W("kernel", "decoder.up_blocks.{level}.resnets.{residual}.conv1.weight", kUpConv1Dims,
      &kCausal),
    W("bias", "decoder.up_blocks.{level}.resnets.{residual}.conv1.bias")};
const RecipeNode kUpConv1{.name = "vae.up.{level}.{residual}.conv1",
                          .component_kind = "conv.spatial",
                          .params = kUpConv1Params,
                          .weights = kUpConv1Weights,
                          .input = Link("vae.up.{level}.{residual}.norm1"),
                          .group = RecipeGroupKind::kLayer,
                          .group_index = kLevel};
const std::array kUpNorm2Params{P("channels", kCout), P("weight_rank", I(4)),
                                P("weight_axis", kThree), P("silu", kTrue),
                                P("round_bf16", kTrue, &kHalving)};
const std::array kUpNorm2Weights{
    W("weight", "decoder.up_blocks.{level}.resnets.{residual}.norm2.gamma")};
const RecipeNode kUpNorm2{.name = "vae.up.{level}.{residual}.norm2",
                          .component_kind = "norm.channel",
                          .params = kUpNorm2Params,
                          .weights = kUpNorm2Weights,
                          .input = Link("vae.up.{level}.{residual}.conv1"),
                          .group = RecipeGroupKind::kLayer,
                          .group_index = kLevel};
VAE_CONV_PARAMS(kUpConv2Params, kCout, kCout, kThree, kOne);
VAE_CONV_DIMS(kUpConv2Dims, kCout, kCout, kThree);
const std::array kUpConv2Weights{
    W("kernel", "decoder.up_blocks.{level}.resnets.{residual}.conv2.weight", kUpConv2Dims,
      &kCausal),
    W("bias", "decoder.up_blocks.{level}.resnets.{residual}.conv2.bias")};
const RecipeNode kUpConv2{.name = "vae.up.{level}.{residual}.conv2",
                          .component_kind = "conv.spatial",
                          .params = kUpConv2Params,
                          .weights = kUpConv2Weights,
                          .input = Link("vae.up.{level}.{residual}.norm2"),
                          .group = RecipeGroupKind::kLayer,
                          .group_index = kLevel};
const RecipeNode kUpAdd{.name = "vae.up.{level}.{residual}.out",
                        .component_kind = "residual.spatial",
                        .input = Link("vae.up.{level}.{residual}.conv2"),
                        .auxiliary = Link("vae.up.{level}.{residual}.skip"),
                        .has_auxiliary = true};
const std::array kResidualSteps{RecipeStep{.kind = RecipeStep::Kind::kIf, .branch = &kShortcutIf},
                                Node(kUpNorm1),
                                Node(kUpConv1),
                                Node(kUpNorm2),
                                Node(kUpConv2),
                                Node(kUpAdd)};
const RecipeRepeat kResidualRepeat{.index = "residual",
                                   .count = kResCount,
                                   .body = kResidualSteps,
                                   .input = Link("vae.up.entry.{level}"),
                                   .input_name = "vae.up.{level}.{residual}.stream",
                                   .body_yield = "vae.up.{level}.{residual}.out",
                                   .yield = "vae.up.result.{level}"};

// An up block's input, saved for a residual decoder's duplicating shortcut.
const RecipeIf kSaveInput{.condition = kTrue,
                          .input = Link("vae.up.entry.{level}"),
                          .then_yield = "vae.up.save.{level}",
                          .else_yield = "vae.up.save.{level}",
                          .yield = "vae.up.saved.{level}",
                          .input_name = "vae.up.save.{level}"};
const std::array kNearestParams{P("factor", kTwo)};
const RecipeNode kNearest{.name = "vae.up.{level}.nearest",
                          .component_kind = "upsample.nearest",
                          .params = kNearestParams,
                          .input = Link("vae.up.result.{level}")};
// The upsampler's convolution: a residual decoder's keeps its channels (its
// `upsampler`), the other's halves them (its `upsamplers.0`).
const std::array kKeepingConvParams{P("in_channels", kCout), P("out_channels", kCout),
                                    P("kernel", kThree), P("padding", kOne)};
const std::array kKeepingConvWeights{
    W("kernel", "decoder.up_blocks.{level}.upsampler.resample.1.weight"),
    W("bias", "decoder.up_blocks.{level}.upsampler.resample.1.bias")};
const RecipeNode kKeepingConv{.name = "vae.up.{level}.conv.keeping",
                              .component_kind = "conv.spatial",
                              .params = kKeepingConvParams,
                              .weights = kKeepingConvWeights,
                              .input = Link("vae.up.{level}.upconv"),
                              .group = RecipeGroupKind::kLayer,
                              .group_index = kLevel};
const std::array kHalvingConvParams{P("in_channels", kCout), P("out_channels", kHalfCout),
                                    P("kernel", kThree), P("padding", kOne)};
const std::array kHalvingConvWeights{
    W("kernel", "decoder.up_blocks.{level}.upsamplers.0.resample.1.weight"),
    W("bias", "decoder.up_blocks.{level}.upsamplers.0.resample.1.bias")};
const RecipeNode kHalvingConv{.name = "vae.up.{level}.conv.halving",
                              .component_kind = "conv.spatial",
                              .params = kHalvingConvParams,
                              .weights = kHalvingConvWeights,
                              .input = Link("vae.up.{level}.upconv"),
                              .group = RecipeGroupKind::kLayer,
                              .group_index = kLevel};
const std::array kKeepingConvSteps{Node(kKeepingConv)};
const std::array kHalvingConvSteps{Node(kHalvingConv)};
const RecipeIf kUpConvIf{.condition = kResidual,
                         .then_steps = kKeepingConvSteps,
                         .else_steps = kHalvingConvSteps,
                         .input = Link("vae.up.{level}.nearest"),
                         .then_yield = "vae.up.{level}.conv.keeping",
                         .else_yield = "vae.up.{level}.conv.halving",
                         .yield = "vae.up.{level}.conv",
                         .input_name = "vae.up.{level}.upconv"};
// A residual decoder's duplicating shortcut from the saved input: in time
// where the block upsamples in time, else in space.
const std::array kDup1Params{P("factor_t", kOne)};
const RecipeNode kDup1{.name = "vae.up.{level}.dup1",
                       .component_kind = "residual.upsample",
                       .params = kDup1Params,
                       .input = Link("vae.up.{level}.conv"),
                       .auxiliary = Link("vae.up.saved.{level}"),
                       .has_auxiliary = true};
const std::array kDup2Params{P("factor_t", kTwo)};
const RecipeNode kDup2{.name = "vae.up.{level}.dup2",
                       .component_kind = "residual.upsample",
                       .params = kDup2Params,
                       .input = Link("vae.up.{level}.conv"),
                       .auxiliary = Link("vae.up.saved.{level}"),
                       .has_auxiliary = true};
const std::array kDup1Steps{Node(kDup1)};
const std::array kDup2Steps{Node(kDup2)};
constexpr auto kTemporalStep = ex::At(kTemporal, "level");
constexpr auto kUpsamplesInTime = E(K::kNotEqual, kTemporalStep, kZero);
const RecipeIf kTemporalIf{.condition = kUpsamplesInTime,
                           .then_steps = kDup2Steps,
                           .else_steps = kDup1Steps,
                           .input = Link("vae.up.{level}.conv"),
                           .then_yield = "vae.up.{level}.dup2",
                           .else_yield = "vae.up.{level}.dup1",
                           .yield = "vae.up.{level}.done",
                           .input_name = "vae.up.{level}.upstream"};
const std::array kDuplicateSteps{RecipeStep{.kind = RecipeStep::Kind::kIf, .branch = &kTemporalIf}};
const RecipeIf kDuplicateIf{.condition = kResidual,
                            .then_steps = kDuplicateSteps,
                            .input = Link("vae.up.{level}.conv"),
                            .then_yield = "vae.up.{level}.done",
                            .else_yield = "vae.up.{level}.tail",
                            .yield = "vae.up.{level}.out",
                            .input_name = "vae.up.{level}.tail"};
const std::array kUpsampleSteps{
    RecipeStep{.kind = RecipeStep::Kind::kIf, .branch = &kSaveInput},
    RecipeStep{.kind = RecipeStep::Kind::kRepeat, .repeat = &kResidualRepeat}, Node(kNearest),
    RecipeStep{.kind = RecipeStep::Kind::kIf, .branch = &kUpConvIf},
    RecipeStep{.kind = RecipeStep::Kind::kIf, .branch = &kDuplicateIf}};
const std::array kFinalLevelSteps{
    RecipeStep{.kind = RecipeStep::Kind::kRepeat, .repeat = &kResidualRepeat}};
const RecipeIf kLevelIf{.condition = kIsUp,
                        .then_steps = kUpsampleSteps,
                        .else_steps = kFinalLevelSteps,
                        .input = Link("vae.up.current.{level}"),
                        .then_yield = "vae.up.{level}.out",
                        .else_yield = "vae.up.result.{level}",
                        .yield = "vae.up.next.{level}",
                        .input_name = "vae.up.entry.{level}"};
const std::array kLevelSteps{RecipeStep{.kind = RecipeStep::Kind::kIf, .branch = &kLevelIf}};
const RecipeRepeat kLevelRepeat{.index = "level",
                                .count = kLevels,
                                .body = kLevelSteps,
                                .input = Link("vae.mid.1.out"),
                                .input_name = "vae.up.current.{level}",
                                .body_yield = "vae.up.next.{level}",
                                .yield = "vae.up.final"};

// ---------------------------------------------------------- the output
const std::array kNormOutParams{P("channels", kOutChannels), P("weight_rank", I(4)),
                                P("weight_axis", kThree), P("silu", kTrue),
                                P("round_bf16", kTrue, &kHalving)};
const std::array kNormOutWeights{W("weight", "decoder.norm_out.gamma")};
const RecipeNode kNormOut{.name = "vae.norm_out",
                          .component_kind = "norm.channel",
                          .params = kNormOutParams,
                          .weights = kNormOutWeights,
                          .input = Link("vae.up.final")};
VAE_CONV_PARAMS(kConvOutParams, kOutChannels, kOut, kThree, kOne);
VAE_CONV_DIMS(kConvOutDims, kOutChannels, kOut, kThree);
const std::array kConvOutWeights{W("kernel", "decoder.conv_out.weight", kConvOutDims, &kCausal),
                                 W("bias", "decoder.conv_out.bias")};
const RecipeNode kConvOut{.name = "vae.output",
                          .component_kind = "conv.spatial",
                          .params = kConvOutParams,
                          .weights = kConvOutWeights,
                          .input = Link("vae.norm_out")};
constexpr auto kHardTanh = S("hardtanh");
const std::array kClampParams{P("activation", kHardTanh)};
const RecipeNode kClamp{.name = "vae.output.clamp",
                        .component_kind = "activation.elementwise",
                        .params = kClampParams,
                        .input = Link("vae.output")};
const std::array kClampSteps{Node(kClamp)};
const RecipeIf kClampIf{.condition = kHalving,
                        .then_steps = kClampSteps,
                        .input = Link("vae.output"),
                        .then_yield = "vae.output.clamp",
                        .else_yield = "vae.output.unclamped",
                        .yield = "vae.decoded",
                        .input_name = "vae.output.unclamped"};

#undef VAE_CONV_PARAMS
#undef VAE_CONV_DIMS

const std::array kSteps{Node(kReshape),
                        RecipeStep{.kind = RecipeStep::Kind::kIf, .branch = &kAffineIf},
                        Node(kPostQuant),
                        Node(kConvIn),
                        Node(kMid0Norm1),
                        Node(kMid0Conv1),
                        Node(kMid0Norm2),
                        Node(kMid0Conv2),
                        Node(kMid0Add),
                        Node(kAttentionNorm),
                        Node(kQkv),
                        Node(kAttention),
                        Node(kProj),
                        Node(kAttentionAdd),
                        Node(kMid1Norm1),
                        Node(kMid1Conv1),
                        Node(kMid1Norm2),
                        Node(kMid1Conv2),
                        Node(kMid1Add),
                        RecipeStep{.kind = RecipeStep::Kind::kRepeat, .repeat = &kLevelRepeat},
                        Node(kNormOut),
                        Node(kConvOut),
                        RecipeStep{.kind = RecipeStep::Kind::kIf, .branch = &kClampIf}};
constexpr auto kImages = I(1);
const std::array kHostPorts{RecipeSpatialOperand{
    .shape = {.width = kOne, .height = kTwo, .channels = kZ, .images = kImages}}};
// The decoded image of a one-latent input: the source's spatial factor
// (unchecked where the source does not state it).
const std::array kOutputShapes{RecipeSpatialShape{
    .width = kSpatialFactor, .height = kSpatialFactor, .channels = kOut, .images = kImages}};
const StageRecipe kRecipe{
    .name = "vae",
    .role = StageRole::kVae,
    .artifact = 0,
    .stream_width = kZ,
    .primary_input = {.shape = {.width = kOne, .height = kOne, .channels = kZ, .images = kImages}},
    .host_ports = kHostPorts,
    .steps = kSteps,
    .expected_outputs = kOutputShapes};

constexpr std::array kSourceArchitectures{std::string_view("AutoencoderKLQwenImage"),
                                          std::string_view("AutoencoderKLQwenImage21")};
constexpr std::array kContainers{std::string_view("safetensors")};
// What decoding one image does not run: the encoder and its quantizing
// convolution, and the up blocks' temporal convolutions, which a first
// frame never applies.
constexpr std::array kIgnoredTensors{
    std::string_view("decoder.up_blocks.{layer}.upsampler.time_conv.weight"),
    std::string_view("decoder.up_blocks.{layer}.upsampler.time_conv.bias"),
    std::string_view("decoder.up_blocks.{layer}.upsamplers.0.time_conv.weight"),
    std::string_view("decoder.up_blocks.{layer}.upsamplers.0.time_conv.bias")};
constexpr std::array kIgnoredPrefixes{std::string_view("encoder."),
                                      std::string_view("quant_conv.")};
// The config keys the recipe does not read, and why.
constexpr std::array kAuxiliary{
    AuxiliaryNamespace{"attn_scales",
                       "the encoder's attention scales; an attention block they add elsewhere "
                       "is a tensor the stage does not read, refused"},
    AuxiliaryNamespace{"in_channels", "the encoder's input channels; the encoder is left out"},
    AuxiliaryNamespace{"scale_factor_temporal",
                       "frames a latent frame decodes to; one image is "
                       "one frame (the time convolutions are left out)"},
};
constexpr std::array kNullKeys{std::string_view("patch_size")};

const Mapping kMapping{.architecture = "qwen_image_vae",
                       .source_architectures = kSourceArchitectures,
                       .version = 1,
                       .pipeline = PipelineKind::kImage,
                       .role = StageRole::kVae,
                       .stage = "vae",
                       .stage_recipe = &kRecipe,
                       .containers = kContainers,
                       .auxiliary = kAuxiliary,
                       .ignored_tensors = kIgnoredTensors,
                       .null_semantics = kNullKeys,
                       .ignored_tensor_prefixes = kIgnoredPrefixes};

}  // namespace

const Mapping& QwenImageVaeMapping() { return kMapping; }

}  // namespace llmp::spec::mappings
