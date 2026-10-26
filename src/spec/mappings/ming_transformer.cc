// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "spec/mappings/ming_transformer.h"

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>

#include "spec/stage_recipe.h"

namespace llmp::spec::mappings {
namespace {

constexpr std::array<std::string_view, 1> kSourceArchitectures{"DiffusionTransformer"};
constexpr std::array<std::string_view, 1> kContainers{"safetensors"};

constexpr RecipeValue I(std::int64_t value) { return {.literal = value}; }
constexpr RecipeValue R(double value) { return {.literal = value}; }
constexpr RecipeValue B(bool value) { return {.literal = value}; }
constexpr RecipeValue S(std::string_view value) { return {.literal = std::string(value)}; }
constexpr RecipeValue Source(std::string_view key) {
  return {.kind = RecipeValueKind::kSourceValue, .source_key = key};
}
constexpr RecipeValue SourceList(std::string_view key, std::int64_t index) {
  return {.kind = RecipeValueKind::kSourceListElement, .source_key = key, .fixed_index = index};
}
constexpr RecipeValue Index(std::string_view name) {
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
constexpr RecipeWeight W(std::string_view role, std::string_view tensor) {
  return {.role = role, .tensor = tensor};
}
using WeightSpans = std::array<std::span<const RecipeWeight>, 15>;
constexpr RecipeRef V(std::string_view name) {
  return {.kind = RecipeRefKind::kNamedValue, .name = name};
}
constexpr RecipeRef H(std::uint16_t port) {
  return {.kind = RecipeRefKind::kHostPort, .port = port};
}
constexpr RecipeRef Input() { return {.kind = RecipeRefKind::kStageInput}; }

// All expressions are named static objects because RecipeValue stores child
// pointers. The checked recipe vocabulary evaluates them only during import.
constexpr auto kDim = Source("dim");
constexpr auto kHeads = Source("n_heads");
constexpr auto kHeadDim = E(RecipeValueKind::kDivideFloor, kDim, kHeads);
constexpr auto kEight = I(8);
constexpr auto kThree = I(3);
constexpr auto kMlpNumerator = E(RecipeValueKind::kMultiply, kDim, kEight);
constexpr auto kMlp = E(RecipeValueKind::kDivideFloor, kMlpNumerator, kThree);
constexpr auto kCaptionDim = Source("cap_feat_dim");
constexpr auto kLatentChannels = Source("in_channels");
constexpr auto kPatchHeight = SourceList("all_patch_size", 0);
constexpr auto kPatchWidth = SourceList("all_patch_size", 0);
constexpr auto kPatchFrames = SourceList("all_f_patch_size", 0);
constexpr auto kPatchPixels = E(RecipeValueKind::kMultiply, kPatchHeight, kPatchWidth);
constexpr auto kPatchVolume = E(RecipeValueKind::kMultiply, kPatchPixels, kPatchFrames);
constexpr auto kPatchFeatures = E(RecipeValueKind::kMultiply, kPatchVolume, kLatentChannels);
constexpr auto kRefinerCount = Source("n_refiner_layers");
constexpr auto kLayerCount = Source("n_layers");
constexpr auto kNormEps = Source("norm_eps");
constexpr auto kQkNorm = Source("qk_norm");
constexpr auto kRopeTheta = Source("rope_theta");
constexpr auto kAxis0 = SourceList("axes_dims", 0);
constexpr auto kAxis1 = SourceList("axes_dims", 1);
constexpr auto kAxis2 = SourceList("axes_dims", 2);
constexpr auto kScaleOne = R(1.0);
constexpr auto kHeadRoot = U(RecipeValueKind::kSqrt, kHeadDim);
constexpr auto kAttentionScale = E(RecipeValueKind::kRealDivide, kScaleOne, kHeadRoot);
constexpr auto kFour = I(4);
constexpr auto kFourDim = E(RecipeValueKind::kMultiply, kFour, kDim);
constexpr auto kTwo = I(2);
constexpr auto k256 = I(256);
constexpr auto kTwoRefiners = E(RecipeValueKind::kMultiply, kTwo, kRefinerCount);
constexpr auto kBlockIndex = Index("block");
constexpr auto kContextGroup = E(RecipeValueKind::kAdd, kBlockIndex, kRefinerCount);
constexpr auto kJointGroup = E(RecipeValueKind::kAdd, kBlockIndex, kTwoRefiners);
constexpr auto kDimBelowTime = E(RecipeValueKind::kLess, kDim, k256);
constexpr auto kTimeWidth = Select(kDimBelowTime, kDim, k256);

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
const std::array<RecipeRef, 15> kNoiseInputs{V("time.raw"),
                                             V("noise.stream.{block}"),
                                             V("noise.stream.{block}"),
                                             V("noise.stream.{block}"),
                                             V("noise.stream.{block}"),
                                             V("noise.stream.{block}"),
                                             V("noise.attn.norm1.{block}"),
                                             V("noise.attn.scale.{block}"),
                                             V("noise.attn.{block}"),
                                             V("noise.attn.norm2.{block}"),
                                             V("noise.attn.residual.{block}"),
                                             V("noise.ffn.norm1.{block}"),
                                             V("noise.ffn.scale.{block}"),
                                             V("noise.ffn.{block}"),
                                             V("noise.ffn.norm2.{block}")};
const std::array<RecipeRef, 15> kNoiseAux{RecipeRef{},
                                          V("noise.mod.{block}"),
                                          V("noise.mod.{block}"),
                                          V("noise.mod.{block}"),
                                          V("noise.mod.{block}"),
                                          {},
                                          V("noise.condition0.{block}"),
                                          H(2),
                                          {},
                                          V("noise.condition1.{block}"),
                                          {},
                                          V("noise.condition2.{block}"),
                                          {},
                                          {},
                                          V("noise.condition3.{block}")};
const std::array<RecipeRef, 15> kNoiseCarry{
    RecipeRef{}, RecipeRef{}, RecipeRef{}, RecipeRef{}, RecipeRef{},
    {},          {},          H(3),        {},          V("noise.stream.{block}"),
    {},          {},          {},          {},          V("noise.attn.residual.{block}")};
const auto kNoiseNodes =
    MakeModNodes(kNoiseNames, kNoiseInputs, kNoiseAux, kNoiseCarry, kNoiseModWeights, kBlockIndex);

const std::array<std::string_view, 15> kJointNames{
    "joint.mod.{block}",           "joint.condition0.{block}", "joint.condition1.{block}",
    "joint.condition2.{block}",    "joint.condition3.{block}", "joint.attn.norm1.{block}",
    "joint.attn.scale.{block}",    "joint.attn.{block}",       "joint.attn.norm2.{block}",
    "joint.attn.residual.{block}", "joint.ffn.norm1.{block}",  "joint.ffn.scale.{block}",
    "joint.ffn.{block}",           "joint.ffn.norm2.{block}",  "joint.ffn.residual.{block}"};
const std::array<RecipeRef, 15> kJointInputs{V("time.raw"),
                                             V("joint.stream.{block}"),
                                             V("joint.stream.{block}"),
                                             V("joint.stream.{block}"),
                                             V("joint.stream.{block}"),
                                             V("joint.stream.{block}"),
                                             V("joint.attn.norm1.{block}"),
                                             V("joint.attn.scale.{block}"),
                                             V("joint.attn.{block}"),
                                             V("joint.attn.norm2.{block}"),
                                             V("joint.attn.residual.{block}"),
                                             V("joint.ffn.norm1.{block}"),
                                             V("joint.ffn.scale.{block}"),
                                             V("joint.ffn.{block}"),
                                             V("joint.ffn.norm2.{block}")};
const std::array<RecipeRef, 15> kJointAux{RecipeRef{},
                                          V("joint.mod.{block}"),
                                          V("joint.mod.{block}"),
                                          V("joint.mod.{block}"),
                                          V("joint.mod.{block}"),
                                          {},
                                          V("joint.condition0.{block}"),
                                          H(6),
                                          {},
                                          V("joint.condition1.{block}"),
                                          {},
                                          V("joint.condition2.{block}"),
                                          {},
                                          {},
                                          V("joint.condition3.{block}")};
const std::array<RecipeRef, 15> kJointCarry{
    RecipeRef{}, RecipeRef{}, RecipeRef{}, RecipeRef{}, RecipeRef{},
    {},          {},          H(7),        {},          V("joint.stream.{block}"),
    {},          {},          {},          {},          V("joint.attn.residual.{block}")};
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
                                          .input = V("context.stream.{block}"),
                                          .group = RecipeGroupKind::kLayer,
                                          .group_index = kContextGroup},
                               RecipeNode{.name = "context.attn.{block}",
                                          .component_kind = "attention.dense",
                                          .params = kAttentionParams,
                                          .weights = kContextAttentionWeights,
                                          .input = V("context.attn.norm1.{block}"),
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
                                          .input = V("context.attn.{block}"),
                                          .group = RecipeGroupKind::kLayer,
                                          .group_index = kContextGroup},
                               RecipeNode{.name = "context.attn.residual.{block}",
                                          .component_kind = "residual.spatial",
                                          .input = V("context.attn.norm2.{block}"),
                                          .auxiliary = V("context.stream.{block}"),
                                          .has_auxiliary = true},
                               RecipeNode{.name = "context.ffn.norm1.{block}",
                                          .component_kind = "norm.rms",
                                          .params = kRmsParams,
                                          .weights = kContextFfnNormWeights,
                                          .input = V("context.attn.residual.{block}"),
                                          .group = RecipeGroupKind::kLayer,
                                          .group_index = kContextGroup},
                               RecipeNode{.name = "context.ffn.{block}",
                                          .component_kind = "ffn.glu",
                                          .params = kGluParams,
                                          .weights = kContextFfnWeights,
                                          .input = V("context.ffn.norm1.{block}"),
                                          .group = RecipeGroupKind::kLayer,
                                          .group_index = kContextGroup},
                               RecipeNode{.name = "context.ffn.norm2.{block}",
                                          .component_kind = "norm.rms",
                                          .params = kRmsParams,
                                          .weights = kContextFfnPostNormWeights,
                                          .input = V("context.ffn.{block}"),
                                          .group = RecipeGroupKind::kLayer,
                                          .group_index = kContextGroup},
                               RecipeNode{.name = "context.ffn.residual.{block}",
                                          .component_kind = "residual.spatial",
                                          .input = V("context.ffn.norm2.{block}"),
                                          .auxiliary = V("context.attn.residual.{block}"),
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

const std::array kTime0Params{P("in_channels", I(256)), P("out_channels", I(1024)),
                              P("bias", B(true)), P("silu", B(true)), P("gelu", B(false))};
const std::array kTime0Weights{W("weight", "t_embedder.mlp.0.weight"),
                               W("bias", "t_embedder.mlp.0.bias")};
const std::array kTime1Params{P("in_channels", I(1024)), P("out_channels", kTimeWidth),
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
                        .input = V("time.hidden")};
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
                                    .input = V("caption.norm")};
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
                                .input = V("image.embedded"),
                                .input_name = "noise.stream.{block}",
                                .body_yield = "noise.ffn.residual.{block}",
                                .yield = "noise.final"};
const RecipeRepeat kContextRepeat{.index = "block",
                                  .count = kRefinerCount,
                                  .body = kContextSteps,
                                  .input = V("caption.embedded"),
                                  .input_name = "context.stream.{block}",
                                  .body_yield = "context.ffn.residual.{block}",
                                  .yield = "context.final"};
const std::array kConcatParams{P("axis", I(1))};
const RecipeNode kConcat{.name = "joint.input",
                         .component_kind = "concat.spatial",
                         .params = kConcatParams,
                         .input = V("noise.final"),
                         .auxiliary = V("context.final"),
                         .has_auxiliary = true};
const RecipeRepeat kJointRepeat{.index = "block",
                                .count = kLayerCount,
                                .body = kJointSteps,
                                .input = V("joint.input"),
                                .input_name = "joint.stream.{block}",
                                .body_yield = "joint.ffn.residual.{block}",
                                .yield = "joint.final"};

const std::array kHeadSiluParams{P("activation", S("silu"))};
const RecipeNode kHeadSilu{.name = "head.time.silu",
                           .component_kind = "activation.elementwise",
                           .params = kHeadSiluParams,
                           .input = V("time.raw")};
const std::array kHeadModParams{P("in_channels", kTimeWidth), P("out_channels", kDim),
                                P("bias", B(true)), P("silu", B(false)), P("gelu", B(false))};
const std::array kHeadModWeights{W("weight", "all_final_layer.2-1.adaLN_modulation.1.weight"),
                                 W("bias", "all_final_layer.2-1.adaLN_modulation.1.bias")};
const RecipeNode kHeadMod{.name = "head.modulation",
                          .component_kind = "projection.spatial",
                          .params = kHeadModParams,
                          .weights = kHeadModWeights,
                          .input = V("head.time.silu")};
const std::array kHeadConditionParams{P("quarter", I(0)), P("quarters", I(1)),
                                      P("prefix_rows", I(0))};
const RecipeNode kHeadCondition{.name = "head.scale",
                                .component_kind = "condition.rows",
                                .params = kHeadConditionParams,
                                .input = V("joint.final"),
                                .auxiliary = V("head.modulation"),
                                .has_auxiliary = true};
const std::array kHeadNormParams{P("side", S("scale")), P("eps", R(1.0e-6))};
const RecipeNode kHeadNorm{.name = "head.norm",
                           .component_kind = "norm.adaptive",
                           .params = kHeadNormParams,
                           .input = V("joint.final"),
                           .auxiliary = V("head.scale"),
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
                                 .input = V("head.norm")};
const std::array kSliceParams{P("start", I(0)), P("rows", I(32), "latent_rows"),
                              P("input_rows", I(64), "joint_rows")};
const RecipeNode kPrediction{.name = "prediction",
                             .component_kind = "slice.spatial",
                             .params = kSliceParams,
                             .input = V("head.full")};

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

const RecipeSpatialOperand kPrimaryInput{
    .shape = {.width = I(1), .height = I(32), .channels = kCaptionDim, .images = I(1)}};
const std::array kHostPorts{
    RecipeSpatialOperand{
        .shape = {.width = I(1), .height = I(32), .channels = kPatchFeatures, .images = I(1)}},
    RecipeSpatialOperand{
        .shape = {.width = I(1), .height = I(1), .channels = I(256), .images = I(1)}},
    RecipeSpatialOperand{
        .shape = {.width = I(32), .height = I(3), .channels = I(1), .images = I(1)},
        .encoding = format::Encoding::kI32},
    RecipeSpatialOperand{
        .shape = {.width = I(32), .height = I(32), .channels = I(1), .images = I(1)}},
    RecipeSpatialOperand{
        .shape = {.width = I(32), .height = I(3), .channels = I(1), .images = I(1)},
        .encoding = format::Encoding::kI32},
    RecipeSpatialOperand{
        .shape = {.width = I(32), .height = I(32), .channels = I(1), .images = I(1)}},
    RecipeSpatialOperand{
        .shape = {.width = I(64), .height = I(3), .channels = I(1), .images = I(1)},
        .encoding = format::Encoding::kI32},
    RecipeSpatialOperand{
        .shape = {.width = I(64), .height = I(64), .channels = I(1), .images = I(1)}}};
const RecipeSpatialShape kOutputShape{
    .width = I(1), .height = I(32), .channels = kPatchFeatures, .images = I(1)};
const std::array kOutputShapes{kOutputShape};
const std::array kExports{V("prediction")};
const StageRecipe kRecipe{.name = "transformer",
                          .role = StageRole::kDenoiser,
                          .artifact = 0,
                          .stream_width = kDim,
                          .primary_input = kPrimaryInput,
                          .host_ports = kHostPorts,
                          .steps = kRecipeSteps,
                          .exports = kExports,
                          .expected_outputs = kOutputShapes};

const Mapping kMapping{.architecture = "diffusion_transformer",
                       .source_architectures = kSourceArchitectures,
                       .version = 1,
                       .pipeline = PipelineKind::kImage,
                       .role = StageRole::kDenoiser,
                       .stage = "transformer",
                       .stage_recipe = &kRecipe,
                       .containers = kContainers};

}  // namespace

const Mapping& MingTransformerMapping() { return kMapping; }

}  // namespace llmp::spec::mappings
