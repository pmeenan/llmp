// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "spec/mappings/ming_transformer.h"

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>

#include "spec/expr.h"
#include "spec/stage_recipe.h"

namespace llmp::spec::mappings {
namespace {

constexpr std::array<std::string_view, 1> kSourceArchitectures{"DiffusionTransformer"};
constexpr std::array<std::string_view, 1> kContainers{"safetensors"};

using ex::B;
using ex::E;
using ex::I;
using ex::R;
using ex::S;
using ex::U;
using K = ExprKind;
constexpr RecipeValue Source(std::string_view key) { return ex::V(key); }
constexpr RecipeValue Index(std::string_view name) { return ex::Idx(name); }
constexpr RecipeParam P(std::string_view name, const RecipeValue& value,
                        std::string_view request_scalar = {}) {
  return {.name = name, .value = value, .request_scalar = request_scalar};
}
constexpr RecipeWeight W(std::string_view role, std::string_view tensor) {
  return {.role = role, .tensor = tensor};
}
using WeightSpans = std::array<std::span<const RecipeWeight>, 15>;
constexpr RecipeRef Link(std::string_view name) {
  return {.kind = RecipeRefKind::kNamedValue, .name = name};
}
constexpr RecipeRef H(std::uint16_t port) {
  return {.kind = RecipeRefKind::kHostPort, .port = port};
}
constexpr RecipeRef Input() { return {.kind = RecipeRefKind::kStageInput}; }

// All expressions are named static objects because an Expr stores child
// pointers. The checked value language evaluates them only during import.
constexpr auto kDim = Source("dim");
constexpr auto kHeads = Source("n_heads");
constexpr auto kHeadDim = E(K::kDivideFloor, kDim, kHeads);
constexpr auto kEight = I(8);
constexpr auto kThree = I(3);
constexpr auto kMlpNumerator = E(K::kMultiply, kDim, kEight);
constexpr auto kMlp = E(K::kDivideFloor, kMlpNumerator, kThree);
constexpr auto kCaptionDim = Source("cap_feat_dim");
constexpr auto kLatentChannels = Source("in_channels");
constexpr auto kPatchSizes = Source("all_patch_size");
constexpr auto kFramePatchSizes = Source("all_f_patch_size");
constexpr auto kPatchHeight = ex::At(kPatchSizes, 0);
constexpr auto kPatchWidth = ex::At(kPatchSizes, 0);
constexpr auto kPatchFrames = ex::At(kFramePatchSizes, 0);
constexpr auto kPatchPixels = E(K::kMultiply, kPatchHeight, kPatchWidth);
constexpr auto kPatchVolume = E(K::kMultiply, kPatchPixels, kPatchFrames);
constexpr auto kPatchFeatures = E(K::kMultiply, kPatchVolume, kLatentChannels);
constexpr auto kRefinerCount = Source("n_refiner_layers");
constexpr auto kLayerCount = Source("n_layers");
constexpr auto kNormEps = Source("norm_eps");
constexpr auto kQkNorm = Source("qk_norm");
constexpr auto kRopeTheta = Source("rope_theta");
constexpr auto kAxes = Source("axes_dims");
constexpr auto kAxis0 = ex::At(kAxes, 0);
constexpr auto kAxis1 = ex::At(kAxes, 1);
constexpr auto kAxis2 = ex::At(kAxes, 2);
constexpr auto kScaleOne = R(1.0);
constexpr auto kHeadRoot = U(K::kSqrt, kHeadDim);
constexpr auto kAttentionScale = E(K::kRealDivide, kScaleOne, kHeadRoot);
constexpr auto kFour = I(4);
constexpr auto kFourDim = E(K::kMultiply, kFour, kDim);
constexpr auto kTwo = I(2);
constexpr auto kTwoRefiners = E(K::kMultiply, kTwo, kRefinerCount);
constexpr auto kBlockIndex = Index("block");
constexpr auto kContextGroup = E(K::kAdd, kBlockIndex, kRefinerCount);
constexpr auto kJointGroup = E(K::kAdd, kBlockIndex, kTwoRefiners);
// The timestep embedder's widths, which the source library fixes in code
// (its frequency embedding and hidden widths, and the modulation width
// min(dim, ADALN_EMBED_DIM)), as the checkpoint's headers state them: its
// first linear's [hidden, frequencies], its second's outputs.
constexpr auto kTimeFrequencies = ex::TensorDim("t_embedder.mlp.0.weight", 1);
constexpr auto kTimeHidden = ex::TensorDim("t_embedder.mlp.0.weight", 0);
constexpr auto kTimeWidth = ex::TensorDim("t_embedder.mlp.2.weight", 0);
// The import-time check geometry: a caption and an image of a few rows
// (any fit; the stored graph's only row counts are its output slice's, and
// those are request-bound).
constexpr auto kCheckCaptionRows = I(32);
constexpr auto kCheckImageRows = I(32);
constexpr auto kCheckJointRows = E(K::kAdd, kCheckCaptionRows, kCheckImageRows);

const std::array kProjectionModParams{P("in_channels", kTimeWidth), P("out_channels", kFourDim),
                                      P("bias", B(true)), P("silu", B(false)), P("gelu", B(false))};
const std::array kCondition0Params{P("quarter", I(0)), P("quarters", kFour),
                                   P("prefix_rows", I(0))};
const std::array kCondition1Params{P("quarter", I(1)), P("quarters", kFour),
                                   P("prefix_rows", I(0))};
const std::array kCondition2Params{P("quarter", I(2)), P("quarters", kFour),
                                   P("prefix_rows", I(0))};
const std::array kCondition3Params{P("quarter", I(3)), P("quarters", kFour),
                                   P("prefix_rows", I(0))};
const std::array kRmsParams{P("channels", kDim), P("eps", kNormEps),
                            P("norm_zero_centered", B(false))};
const std::array kScaleParams{P("side", S("scale"))};
const std::array kGateParams{P("side", S("gate"))};
const std::array kAttentionParams{
    P("eps", R(1.0e-5)),        P("heads", kHeads),          P("key_dim", kHeadDim),
    P("kv_heads", kHeads),      P("placement", S("bare")),   P("qk_norm", kQkNorm),
    P("rope_axis0", kAxis0),    P("rope_axis1", kAxis1),     P("rope_axis2", kAxis2),
    P("rope_base", kRopeTheta), P("rope_dims", kHeadDim),    P("rope_scale", R(1.0)),
    P("rotation", S("normal")), P("scale", kAttentionScale), P("scale_scores", B(true)),
    P("softcap", I(0)),         P("stateless", B(true)),     P("value_dim", kHeadDim),
    P("window", I(0))};
const std::array kGluParams{P("activation", S("silu")), P("eps", kNormEps), P("hidden", kMlp),
                            P("norm_zero_centered", B(false)), P("placement", S("bare"))};

const std::array kNoiseModProjectionWeights{
    W("weight", "noise_refiner.{block}.adaLN_modulation.0.weight"),
    W("bias", "noise_refiner.{block}.adaLN_modulation.0.bias")};
const std::array kNoiseAttentionNormWeights{
    W("weight", "noise_refiner.{block}.attention_norm1.weight")};
const std::array kNoiseAttentionWeights{
    W("k", "noise_refiner.{block}.attention.to_k.weight"),
    W("k_norm", "noise_refiner.{block}.attention.norm_k.weight"),
    W("output", "noise_refiner.{block}.attention.to_out.0.weight"),
    W("q", "noise_refiner.{block}.attention.to_q.weight"),
    W("q_norm", "noise_refiner.{block}.attention.norm_q.weight"),
    W("v", "noise_refiner.{block}.attention.to_v.weight")};
const std::array kNoiseAttentionPostNormWeights{
    W("weight", "noise_refiner.{block}.attention_norm2.weight")};
const std::array kNoiseFfnNormWeights{W("weight", "noise_refiner.{block}.ffn_norm1.weight")};
const std::array kNoiseFfnWeights{W("gate", "noise_refiner.{block}.feed_forward.w1.weight"),
                                  W("up", "noise_refiner.{block}.feed_forward.w3.weight"),
                                  W("down", "noise_refiner.{block}.feed_forward.w2.weight")};
const std::array kNoiseFfnPostNormWeights{W("weight", "noise_refiner.{block}.ffn_norm2.weight")};
const WeightSpans kNoiseModWeights{std::span<const RecipeWeight>{kNoiseModProjectionWeights},
                                   {},
                                   std::span<const RecipeWeight>{},
                                   {},
                                   {},
                                   std::span<const RecipeWeight>{kNoiseAttentionNormWeights},
                                   {},
                                   std::span<const RecipeWeight>{kNoiseAttentionWeights},
                                   std::span<const RecipeWeight>{kNoiseAttentionPostNormWeights},
                                   {},
                                   std::span<const RecipeWeight>{kNoiseFfnNormWeights},
                                   {},
                                   std::span<const RecipeWeight>{kNoiseFfnWeights},
                                   std::span<const RecipeWeight>{kNoiseFfnPostNormWeights},
                                   {}};

const std::array kJointModProjectionWeights{W("weight", "layers.{block}.adaLN_modulation.0.weight"),
                                            W("bias", "layers.{block}.adaLN_modulation.0.bias")};
const std::array kJointAttentionNormWeights{W("weight", "layers.{block}.attention_norm1.weight")};
const std::array kJointAttentionWeights{W("k", "layers.{block}.attention.to_k.weight"),
                                        W("k_norm", "layers.{block}.attention.norm_k.weight"),
                                        W("output", "layers.{block}.attention.to_out.0.weight"),
                                        W("q", "layers.{block}.attention.to_q.weight"),
                                        W("q_norm", "layers.{block}.attention.norm_q.weight"),
                                        W("v", "layers.{block}.attention.to_v.weight")};
const std::array kJointAttentionPostNormWeights{
    W("weight", "layers.{block}.attention_norm2.weight")};
const std::array kJointFfnNormWeights{W("weight", "layers.{block}.ffn_norm1.weight")};
const std::array kJointFfnWeights{W("gate", "layers.{block}.feed_forward.w1.weight"),
                                  W("up", "layers.{block}.feed_forward.w3.weight"),
                                  W("down", "layers.{block}.feed_forward.w2.weight")};
const std::array kJointFfnPostNormWeights{W("weight", "layers.{block}.ffn_norm2.weight")};
const WeightSpans kJointModWeights{std::span<const RecipeWeight>{kJointModProjectionWeights},
                                   {},
                                   {},
                                   {},
                                   {},
                                   std::span<const RecipeWeight>{kJointAttentionNormWeights},
                                   {},
                                   std::span<const RecipeWeight>{kJointAttentionWeights},
                                   std::span<const RecipeWeight>{kJointAttentionPostNormWeights},
                                   {},
                                   std::span<const RecipeWeight>{kJointFfnNormWeights},
                                   {},
                                   std::span<const RecipeWeight>{kJointFfnWeights},
                                   std::span<const RecipeWeight>{kJointFfnPostNormWeights},
                                   {}};

const std::array kContextAttentionNormWeights{
    W("weight", "context_refiner.{block}.attention_norm1.weight")};
const std::array kContextAttentionWeights{
    W("k", "context_refiner.{block}.attention.to_k.weight"),
    W("k_norm", "context_refiner.{block}.attention.norm_k.weight"),
    W("output", "context_refiner.{block}.attention.to_out.0.weight"),
    W("q", "context_refiner.{block}.attention.to_q.weight"),
    W("q_norm", "context_refiner.{block}.attention.norm_q.weight"),
    W("v", "context_refiner.{block}.attention.to_v.weight")};
const std::array kContextAttentionPostNormWeights{
    W("weight", "context_refiner.{block}.attention_norm2.weight")};
const std::array kContextFfnNormWeights{W("weight", "context_refiner.{block}.ffn_norm1.weight")};
const std::array kContextFfnWeights{W("gate", "context_refiner.{block}.feed_forward.w1.weight"),
                                    W("up", "context_refiner.{block}.feed_forward.w3.weight"),
                                    W("down", "context_refiner.{block}.feed_forward.w2.weight")};
const std::array kContextFfnPostNormWeights{
    W("weight", "context_refiner.{block}.ffn_norm2.weight")};

const std::array<std::string_view, 15> kModKinds{
    "projection.spatial", "condition.rows", "condition.rows",  "condition.rows", "condition.rows",
    "norm.rms",           "modulate.scale", "attention.dense", "norm.rms",       "residual.gated",
    "norm.rms",           "modulate.scale", "ffn.glu",         "norm.rms",       "residual.gated"};
const std::array<std::span<const RecipeParam>, 15> kModParams{
    kProjectionModParams, kCondition0Params, kCondition1Params, kCondition2Params,
    kCondition3Params,    kRmsParams,        kScaleParams,      kAttentionParams,
    kRmsParams,           kGateParams,       kRmsParams,        kScaleParams,
    kGluParams,           kRmsParams,        kGateParams};
const std::array<bool, 15> kModHasAuxiliary{false, true, true,  true, true,  false, true, true,
                                            false, true, false, true, false, false, true};
const std::array<bool, 15> kModHasCarry{false, false, false, false, false, false, false, true,
                                        false, true,  false, false, false, false, true};

constexpr std::array<RecipeNode, 15> MakeModNodes(const std::array<std::string_view, 15>& names,
                                                  const std::array<RecipeRef, 15>& inputs,
                                                  const std::array<RecipeRef, 15>& auxiliaries,
                                                  const std::array<RecipeRef, 15>& carries,
                                                  const WeightSpans& weights,
                                                  const RecipeValue& group_index) {
  std::array<RecipeNode, 15> nodes{};
  for (std::size_t i = 0; i < nodes.size(); ++i) {
    nodes[i] = {.name = names[i],
                .component_kind = kModKinds[i],
                .params = kModParams[i],
                .weights = weights[i],
                .input = inputs[i],
                .auxiliary = auxiliaries[i],
                .carry = carries[i],
                .has_auxiliary = kModHasAuxiliary[i],
                .has_carry = kModHasCarry[i],
                .group = weights[i].empty() ? RecipeGroupKind::kWhole : RecipeGroupKind::kLayer,
                .group_index = group_index};
  }
  return nodes;
}

const std::array<std::string_view, 15> kNoiseNames{
    "noise.mod.{block}",           "noise.condition0.{block}", "noise.condition1.{block}",
    "noise.condition2.{block}",    "noise.condition3.{block}", "noise.attn.norm1.{block}",
    "noise.attn.scale.{block}",    "noise.attn.{block}",       "noise.attn.norm2.{block}",
    "noise.attn.residual.{block}", "noise.ffn.norm1.{block}",  "noise.ffn.scale.{block}",
    "noise.ffn.{block}",           "noise.ffn.norm2.{block}",  "noise.ffn.residual.{block}"};
const std::array<RecipeRef, 15> kNoiseInputs{Link("time.raw"),
                                             Link("noise.stream.{block}"),
                                             Link("noise.stream.{block}"),
                                             Link("noise.stream.{block}"),
                                             Link("noise.stream.{block}"),
                                             Link("noise.stream.{block}"),
                                             Link("noise.attn.norm1.{block}"),
                                             Link("noise.attn.scale.{block}"),
                                             Link("noise.attn.{block}"),
                                             Link("noise.attn.norm2.{block}"),
                                             Link("noise.attn.residual.{block}"),
                                             Link("noise.ffn.norm1.{block}"),
                                             Link("noise.ffn.scale.{block}"),
                                             Link("noise.ffn.{block}"),
                                             Link("noise.ffn.norm2.{block}")};
const std::array<RecipeRef, 15> kNoiseAux{RecipeRef{},
                                          Link("noise.mod.{block}"),
                                          Link("noise.mod.{block}"),
                                          Link("noise.mod.{block}"),
                                          Link("noise.mod.{block}"),
                                          {},
                                          Link("noise.condition0.{block}"),
                                          H(2),
                                          {},
                                          Link("noise.condition1.{block}"),
                                          {},
                                          Link("noise.condition2.{block}"),
                                          {},
                                          {},
                                          Link("noise.condition3.{block}")};
const std::array<RecipeRef, 15> kNoiseCarry{
    RecipeRef{}, RecipeRef{}, RecipeRef{}, RecipeRef{}, RecipeRef{},
    {},          {},          H(3),        {},          Link("noise.stream.{block}"),
    {},          {},          {},          {},          Link("noise.attn.residual.{block}")};
const auto kNoiseNodes =
    MakeModNodes(kNoiseNames, kNoiseInputs, kNoiseAux, kNoiseCarry, kNoiseModWeights, kBlockIndex);

const std::array<std::string_view, 15> kJointNames{
    "joint.mod.{block}",           "joint.condition0.{block}", "joint.condition1.{block}",
    "joint.condition2.{block}",    "joint.condition3.{block}", "joint.attn.norm1.{block}",
    "joint.attn.scale.{block}",    "joint.attn.{block}",       "joint.attn.norm2.{block}",
    "joint.attn.residual.{block}", "joint.ffn.norm1.{block}",  "joint.ffn.scale.{block}",
    "joint.ffn.{block}",           "joint.ffn.norm2.{block}",  "joint.ffn.residual.{block}"};
const std::array<RecipeRef, 15> kJointInputs{Link("time.raw"),
                                             Link("joint.stream.{block}"),
                                             Link("joint.stream.{block}"),
                                             Link("joint.stream.{block}"),
                                             Link("joint.stream.{block}"),
                                             Link("joint.stream.{block}"),
                                             Link("joint.attn.norm1.{block}"),
                                             Link("joint.attn.scale.{block}"),
                                             Link("joint.attn.{block}"),
                                             Link("joint.attn.norm2.{block}"),
                                             Link("joint.attn.residual.{block}"),
                                             Link("joint.ffn.norm1.{block}"),
                                             Link("joint.ffn.scale.{block}"),
                                             Link("joint.ffn.{block}"),
                                             Link("joint.ffn.norm2.{block}")};
const std::array<RecipeRef, 15> kJointAux{RecipeRef{},
                                          Link("joint.mod.{block}"),
                                          Link("joint.mod.{block}"),
                                          Link("joint.mod.{block}"),
                                          Link("joint.mod.{block}"),
                                          {},
                                          Link("joint.condition0.{block}"),
                                          H(6),
                                          {},
                                          Link("joint.condition1.{block}"),
                                          {},
                                          Link("joint.condition2.{block}"),
                                          {},
                                          {},
                                          Link("joint.condition3.{block}")};
const std::array<RecipeRef, 15> kJointCarry{
    RecipeRef{}, RecipeRef{}, RecipeRef{}, RecipeRef{}, RecipeRef{},
    {},          {},          H(7),        {},          Link("joint.stream.{block}"),
    {},          {},          {},          {},          Link("joint.attn.residual.{block}")};
const auto kJointNodes =
    MakeModNodes(kJointNames, kJointInputs, kJointAux, kJointCarry, kJointModWeights, kJointGroup);

std::array<RecipeStep, 15> MakeSteps(const std::array<RecipeNode, 15>& nodes) {
  std::array<RecipeStep, 15> steps{};
  for (std::size_t i = 0; i < steps.size(); ++i)
    steps[i] = {.kind = RecipeStep::Kind::kNode, .node = &nodes[i]};
  return steps;
}
const auto kNoiseSteps = MakeSteps(kNoiseNodes);
const auto kJointSteps = MakeSteps(kJointNodes);

const std::array kContextNodes{RecipeNode{.name = "context.attn.norm1.{block}",
                                          .component_kind = "norm.rms",
                                          .params = kRmsParams,
                                          .weights = kContextAttentionNormWeights,
                                          .input = Link("context.stream.{block}"),
                                          .group = RecipeGroupKind::kLayer,
                                          .group_index = kContextGroup},
                               RecipeNode{.name = "context.attn.{block}",
                                          .component_kind = "attention.dense",
                                          .params = kAttentionParams,
                                          .weights = kContextAttentionWeights,
                                          .input = Link("context.attn.norm1.{block}"),
                                          .auxiliary = H(4),
                                          .carry = H(5),
                                          .has_auxiliary = true,
                                          .has_carry = true,
                                          .group = RecipeGroupKind::kLayer,
                                          .group_index = kContextGroup},
                               RecipeNode{.name = "context.attn.norm2.{block}",
                                          .component_kind = "norm.rms",
                                          .params = kRmsParams,
                                          .weights = kContextAttentionPostNormWeights,
                                          .input = Link("context.attn.{block}"),
                                          .group = RecipeGroupKind::kLayer,
                                          .group_index = kContextGroup},
                               RecipeNode{.name = "context.attn.residual.{block}",
                                          .component_kind = "residual.spatial",
                                          .input = Link("context.attn.norm2.{block}"),
                                          .auxiliary = Link("context.stream.{block}"),
                                          .has_auxiliary = true},
                               RecipeNode{.name = "context.ffn.norm1.{block}",
                                          .component_kind = "norm.rms",
                                          .params = kRmsParams,
                                          .weights = kContextFfnNormWeights,
                                          .input = Link("context.attn.residual.{block}"),
                                          .group = RecipeGroupKind::kLayer,
                                          .group_index = kContextGroup},
                               RecipeNode{.name = "context.ffn.{block}",
                                          .component_kind = "ffn.glu",
                                          .params = kGluParams,
                                          .weights = kContextFfnWeights,
                                          .input = Link("context.ffn.norm1.{block}"),
                                          .group = RecipeGroupKind::kLayer,
                                          .group_index = kContextGroup},
                               RecipeNode{.name = "context.ffn.norm2.{block}",
                                          .component_kind = "norm.rms",
                                          .params = kRmsParams,
                                          .weights = kContextFfnPostNormWeights,
                                          .input = Link("context.ffn.{block}"),
                                          .group = RecipeGroupKind::kLayer,
                                          .group_index = kContextGroup},
                               RecipeNode{.name = "context.ffn.residual.{block}",
                                          .component_kind = "residual.spatial",
                                          .input = Link("context.ffn.norm2.{block}"),
                                          .auxiliary = Link("context.attn.residual.{block}"),
                                          .has_auxiliary = true}};
const std::array<RecipeStep, 8> kContextSteps{
    RecipeStep{.kind = RecipeStep::Kind::kNode, .node = &kContextNodes[0]},
    RecipeStep{.kind = RecipeStep::Kind::kNode, .node = &kContextNodes[1]},
    RecipeStep{.kind = RecipeStep::Kind::kNode, .node = &kContextNodes[2]},
    RecipeStep{.kind = RecipeStep::Kind::kNode, .node = &kContextNodes[3]},
    RecipeStep{.kind = RecipeStep::Kind::kNode, .node = &kContextNodes[4]},
    RecipeStep{.kind = RecipeStep::Kind::kNode, .node = &kContextNodes[5]},
    RecipeStep{.kind = RecipeStep::Kind::kNode, .node = &kContextNodes[6]},
    RecipeStep{.kind = RecipeStep::Kind::kNode, .node = &kContextNodes[7]}};

const std::array kTime0Params{P("in_channels", kTimeFrequencies), P("out_channels", kTimeHidden),
                              P("bias", B(true)), P("silu", B(true)), P("gelu", B(false))};
const std::array kTime0Weights{W("weight", "t_embedder.mlp.0.weight"),
                               W("bias", "t_embedder.mlp.0.bias")};
const std::array kTime1Params{P("in_channels", kTimeHidden), P("out_channels", kTimeWidth),
                              P("bias", B(true)), P("silu", B(false)), P("gelu", B(false))};
const std::array kTime1Weights{W("weight", "t_embedder.mlp.2.weight"),
                               W("bias", "t_embedder.mlp.2.bias")};
const RecipeNode kTime0{.name = "time.hidden",
                        .component_kind = "projection.spatial",
                        .params = kTime0Params,
                        .weights = kTime0Weights,
                        .input = H(1)};
const RecipeNode kTime1{.name = "time.raw",
                        .component_kind = "projection.spatial",
                        .params = kTime1Params,
                        .weights = kTime1Weights,
                        .input = Link("time.hidden")};
const std::array kCaptionNormParams{P("channels", kCaptionDim), P("eps", kNormEps),
                                    P("norm_zero_centered", B(false))};
const std::array kCaptionNormWeights{W("weight", "cap_embedder.0.weight")};
const RecipeNode kCaptionNorm{.name = "caption.norm",
                              .component_kind = "norm.rms",
                              .params = kCaptionNormParams,
                              .weights = kCaptionNormWeights,
                              .input = Input()};
const std::array kCaptionProjectionParams{P("in_channels", kCaptionDim), P("out_channels", kDim),
                                          P("bias", B(true)), P("silu", B(false)),
                                          P("gelu", B(false))};
const std::array kCaptionProjectionWeights{W("weight", "cap_embedder.1.weight"),
                                           W("bias", "cap_embedder.1.bias")};
const RecipeNode kCaptionProjection{.name = "caption.embedded",
                                    .component_kind = "projection.spatial",
                                    .params = kCaptionProjectionParams,
                                    .weights = kCaptionProjectionWeights,
                                    .input = Link("caption.norm")};
const std::array kImageProjectionParams{P("in_channels", kPatchFeatures), P("out_channels", kDim),
                                        P("bias", B(true)), P("silu", B(false)),
                                        P("gelu", B(false))};
const std::array kImageProjectionWeights{W("weight", "all_x_embedder.2-1.weight"),
                                         W("bias", "all_x_embedder.2-1.bias")};
const RecipeNode kImageProjection{.name = "image.embedded",
                                  .component_kind = "projection.spatial",
                                  .params = kImageProjectionParams,
                                  .weights = kImageProjectionWeights,
                                  .input = H(0)};

const RecipeRepeat kNoiseRepeat{.index = "block",
                                .count = kRefinerCount,
                                .body = kNoiseSteps,
                                .input = Link("image.embedded"),
                                .input_name = "noise.stream.{block}",
                                .body_yield = "noise.ffn.residual.{block}",
                                .yield = "noise.final"};
const RecipeRepeat kContextRepeat{.index = "block",
                                  .count = kRefinerCount,
                                  .body = kContextSteps,
                                  .input = Link("caption.embedded"),
                                  .input_name = "context.stream.{block}",
                                  .body_yield = "context.ffn.residual.{block}",
                                  .yield = "context.final"};
const std::array kConcatParams{P("axis", I(1))};
const RecipeNode kConcat{.name = "joint.input",
                         .component_kind = "concat.spatial",
                         .params = kConcatParams,
                         .input = Link("noise.final"),
                         .auxiliary = Link("context.final"),
                         .has_auxiliary = true};
const RecipeRepeat kJointRepeat{.index = "block",
                                .count = kLayerCount,
                                .body = kJointSteps,
                                .input = Link("joint.input"),
                                .input_name = "joint.stream.{block}",
                                .body_yield = "joint.ffn.residual.{block}",
                                .yield = "joint.final"};

const std::array kHeadSiluParams{P("activation", S("silu"))};
const RecipeNode kHeadSilu{.name = "head.time.silu",
                           .component_kind = "activation.elementwise",
                           .params = kHeadSiluParams,
                           .input = Link("time.raw")};
const std::array kHeadModParams{P("in_channels", kTimeWidth), P("out_channels", kDim),
                                P("bias", B(true)), P("silu", B(false)), P("gelu", B(false))};
const std::array kHeadModWeights{W("weight", "all_final_layer.2-1.adaLN_modulation.1.weight"),
                                 W("bias", "all_final_layer.2-1.adaLN_modulation.1.bias")};
const RecipeNode kHeadMod{.name = "head.modulation",
                          .component_kind = "projection.spatial",
                          .params = kHeadModParams,
                          .weights = kHeadModWeights,
                          .input = Link("head.time.silu")};
const std::array kHeadConditionParams{P("quarter", I(0)), P("quarters", I(1)),
                                      P("prefix_rows", I(0))};
const RecipeNode kHeadCondition{.name = "head.scale",
                                .component_kind = "condition.rows",
                                .params = kHeadConditionParams,
                                .input = Link("joint.final"),
                                .auxiliary = Link("head.modulation"),
                                .has_auxiliary = true};
const std::array kHeadNormParams{P("side", S("scale")), P("eps", R(1.0e-6))};
const RecipeNode kHeadNorm{.name = "head.norm",
                           .component_kind = "norm.adaptive",
                           .params = kHeadNormParams,
                           .input = Link("joint.final"),
                           .auxiliary = Link("head.scale"),
                           .has_auxiliary = true};
const std::array kHeadProjectionParams{P("in_channels", kDim), P("out_channels", kPatchFeatures),
                                       P("bias", B(true)), P("silu", B(false)),
                                       P("gelu", B(false))};
const std::array kHeadProjectionWeights{W("weight", "all_final_layer.2-1.linear.weight"),
                                        W("bias", "all_final_layer.2-1.linear.bias")};
const RecipeNode kHeadProjection{.name = "head.full",
                                 .component_kind = "projection.spatial",
                                 .params = kHeadProjectionParams,
                                 .weights = kHeadProjectionWeights,
                                 .input = Link("head.norm")};
const std::array kSliceParams{P("start", I(0)), P("rows", kCheckImageRows, "latent_rows"),
                              P("input_rows", kCheckJointRows, "joint_rows")};
const RecipeNode kPrediction{.name = "prediction",
                             .component_kind = "slice.spatial",
                             .params = kSliceParams,
                             .input = Link("head.full")};

const std::array kRecipeSteps{
    RecipeStep{.kind = RecipeStep::Kind::kNode, .node = &kTime0},
    RecipeStep{.kind = RecipeStep::Kind::kNode, .node = &kTime1},
    RecipeStep{.kind = RecipeStep::Kind::kNode, .node = &kCaptionNorm},
    RecipeStep{.kind = RecipeStep::Kind::kNode, .node = &kCaptionProjection},
    RecipeStep{.kind = RecipeStep::Kind::kNode, .node = &kImageProjection},
    RecipeStep{.kind = RecipeStep::Kind::kRepeat, .repeat = &kNoiseRepeat},
    RecipeStep{.kind = RecipeStep::Kind::kRepeat, .repeat = &kContextRepeat},
    RecipeStep{.kind = RecipeStep::Kind::kNode, .node = &kConcat},
    RecipeStep{.kind = RecipeStep::Kind::kRepeat, .repeat = &kJointRepeat},
    RecipeStep{.kind = RecipeStep::Kind::kNode, .node = &kHeadSilu},
    RecipeStep{.kind = RecipeStep::Kind::kNode, .node = &kHeadMod},
    RecipeStep{.kind = RecipeStep::Kind::kNode, .node = &kHeadCondition},
    RecipeStep{.kind = RecipeStep::Kind::kNode, .node = &kHeadNorm},
    RecipeStep{.kind = RecipeStep::Kind::kNode, .node = &kHeadProjection},
    RecipeStep{.kind = RecipeStep::Kind::kNode, .node = &kPrediction}};

constexpr auto kOne = I(1);
const RecipeSpatialOperand kPrimaryInput{
    .shape = {.width = kOne, .height = kCheckCaptionRows, .channels = kCaptionDim, .images = kOne}};
// Image patches, the timestep's sinusoid, then each attention's positions
// and mask: the noise refiner's (image rows), the context refiner's
// (caption rows) and the joint blocks' (both).
const std::array kHostPorts{
    RecipeSpatialOperand{.shape = {.width = kOne,
                                   .height = kCheckImageRows,
                                   .channels = kPatchFeatures,
                                   .images = kOne}},
    RecipeSpatialOperand{
        .shape = {.width = kOne, .height = kOne, .channels = kTimeFrequencies, .images = kOne}},
    RecipeSpatialOperand{
        .shape = {.width = kCheckImageRows, .height = kThree, .channels = kOne, .images = kOne},
        .encoding = format::Encoding::kI32},
    RecipeSpatialOperand{.shape = {.width = kCheckImageRows,
                                   .height = kCheckImageRows,
                                   .channels = kOne,
                                   .images = kOne}},
    RecipeSpatialOperand{
        .shape = {.width = kCheckCaptionRows, .height = kThree, .channels = kOne, .images = kOne},
        .encoding = format::Encoding::kI32},
    RecipeSpatialOperand{.shape = {.width = kCheckCaptionRows,
                                   .height = kCheckCaptionRows,
                                   .channels = kOne,
                                   .images = kOne}},
    RecipeSpatialOperand{
        .shape = {.width = kCheckJointRows, .height = kThree, .channels = kOne, .images = kOne},
        .encoding = format::Encoding::kI32},
    RecipeSpatialOperand{.shape = {.width = kCheckJointRows,
                                   .height = kCheckJointRows,
                                   .channels = kOne,
                                   .images = kOne}}};
const RecipeSpatialShape kOutputShape{
    .width = kOne, .height = kCheckImageRows, .channels = kPatchFeatures, .images = kOne};
const std::array kOutputShapes{kOutputShape};
const std::array kExports{Link("prediction")};
const StageRecipe kRecipe{.name = "transformer",
                          .role = StageRole::kDenoiser,
                          .artifact = 0,
                          .stream_width = kDim,
                          .primary_input = kPrimaryInput,
                          .host_ports = kHostPorts,
                          .steps = kRecipeSteps,
                          .exports = kExports,
                          .expected_outputs = kOutputShapes};

// The config keys the recipe does not read, and why. The timestep's scale
// and the caption padding and frame options are the pipeline's host-side
// work (its timestep and caption inputs, its output); no serving adapter
// reads them yet, which it must when one does.
constexpr std::array kAuxiliary{
    AuxiliaryNamespace{"axes_lens", "rotary table lengths, a precomputation bound"},
    AuxiliaryNamespace{"n_kv_heads",
                       "key and value heads: other than n_heads, the k and v projections' "
                       "shapes differ from the recipe's and are refused"},
    AuxiliaryNamespace{"t_scale", "the host-computed timestep input's scale"},
    AuxiliaryNamespace{"alignment_padding_mode", "the host-built caption padding and its mask"},
    AuxiliaryNamespace{"multi_frame_output", "the pipeline's output framing"},
};
constexpr std::array<std::string_view, 1> kNullKeys{"siglip_feat_dim"};

const Mapping kMapping{.architecture = "diffusion_transformer",
                       .source_architectures = kSourceArchitectures,
                       .version = 1,
                       .pipeline = PipelineKind::kImage,
                       .role = StageRole::kDenoiser,
                       .stage = "transformer",
                       .stage_recipe = &kRecipe,
                       .containers = kContainers,
                       .auxiliary = kAuxiliary,
                       .null_semantics = kNullKeys};

}  // namespace

const Mapping& MingTransformerMapping() { return kMapping; }

}  // namespace llmp::spec::mappings
