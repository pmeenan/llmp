// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "spec/mappings/dense.h"

#include <array>
#include <span>
#include <string_view>

#include "spec/mappings/names.h"
#include "spec/mappings/rules.h"

namespace llmp::spec::mappings {
namespace {
using enum ValueType;
constexpr std::array kSlots = {
    SlotRule{.slot = "embedding", .site = Site::kEmbedding, .kind = "embedding.token"},
    SlotRule{.slot = "attention", .site = Site::kLayer, .kind = "attention.dense"},
    SlotRule{.slot = "ffn", .site = Site::kLayer, .kind = "ffn.glu"},
    SlotRule{.slot = "final_norm", .site = Site::kFinalNorm, .kind = "norm.rms"},
    SlotRule{.slot = "output", .site = Site::kOutput, .kind = "head.linear"},
};

// The canonical tensors, as llama.cpp's GGUF names them (names.h): a
// safetensors checkpoint's are aliases of these, an EXL3 one's its own
// rules below. Qwen2's QKV biases are required; the other families' are
// optional (kOptionalQkvBias).
constexpr std::array kGgufTensors = {
    TensorRule{.pattern = names::kTokenEmbd.gguf, .slot = "embedding", .role = "table"},
    TensorRule{.pattern = names::kAttnNorm.gguf, .slot = "attention", .role = "norm"},
    TensorRule{.pattern = names::kAttnQ.gguf, .slot = "attention", .role = "q"},
    TensorRule{.pattern = names::kAttnK.gguf, .slot = "attention", .role = "k"},
    TensorRule{.pattern = names::kAttnV.gguf, .slot = "attention", .role = "v"},
    TensorRule{.pattern = names::kAttnQBias.gguf, .slot = "attention", .role = "q_bias"},
    TensorRule{.pattern = names::kAttnKBias.gguf, .slot = "attention", .role = "k_bias"},
    TensorRule{.pattern = names::kAttnVBias.gguf, .slot = "attention", .role = "v_bias"},
    TensorRule{.pattern = names::kRopeFreqs.gguf,
               .slot = "attention",
               .role = "rope_freqs",
               .optional = true},
    TensorRule{.pattern = names::kAttnOutput.gguf, .slot = "attention", .role = "output"},
    TensorRule{.pattern = names::kFfnNorm.gguf, .slot = "ffn", .role = "norm"},
    TensorRule{.pattern = names::kFfnGate.gguf, .slot = "ffn", .role = "gate"},
    TensorRule{.pattern = names::kFfnUp.gguf, .slot = "ffn", .role = "up"},
    TensorRule{.pattern = names::kFfnDown.gguf, .slot = "ffn", .role = "down"},
    TensorRule{.pattern = names::kOutputNorm.gguf, .slot = "final_norm", .role = "weight"},
    TensorRule{.pattern = names::kOutput.gguf,
               .slot = "output",
               .role = "weight",
               .fallback = names::kTokenEmbd.gguf},
};
// A safetensors checkpoint's names for them.
constexpr auto kHfTensors = Aliases(
    names::kTokenEmbd, names::kAttnNorm, names::kAttnQ, names::kAttnK, names::kAttnV,
    names::kAttnQBias, names::kAttnKBias, names::kAttnVBias, names::kAttnOutput, names::kFfnNorm,
    names::kFfnGate, names::kFfnUp, names::kFfnDown, names::kOutputNorm, names::kOutput);

// An EXL3 checkpoint's tensors, under their transformers names: its vectors
// whole, its matrices as their trellis parts (names.h kTrellisParts).
constexpr std::array kTrellisMatrices{
    TrellisMatrix{names::kAttnQ, "attention", "q"},
    TrellisMatrix{names::kAttnK, "attention", "k"},
    TrellisMatrix{names::kAttnV, "attention", "v"},
    TrellisMatrix{names::kAttnOutput, "attention", "output"},
    TrellisMatrix{names::kFfnGate, "ffn", "gate"},
    TrellisMatrix{names::kFfnUp, "ffn", "up"},
    TrellisMatrix{names::kFfnDown, "ffn", "down"},
    TrellisMatrix{names::kOutput, "output", "weight"},
};
constexpr auto kTrellisNames = TrellisNames(kTrellisMatrices);
constexpr std::array kTrellisVectors = {
    TensorRule{.pattern = names::kTokenEmbd.hf, .slot = "embedding", .role = "table"},
    TensorRule{.pattern = names::kAttnNorm.hf, .slot = "attention", .role = "norm"},
    TensorRule{.pattern = names::kAttnQBias.hf, .slot = "attention", .role = "q_bias"},
    TensorRule{.pattern = names::kAttnKBias.hf, .slot = "attention", .role = "k_bias"},
    TensorRule{.pattern = names::kAttnVBias.hf, .slot = "attention", .role = "v_bias"},
    TensorRule{.pattern = names::kFfnNorm.hf, .slot = "ffn", .role = "norm"},
    TensorRule{.pattern = names::kOutputNorm.hf, .slot = "final_norm", .role = "weight"},
};
constexpr auto kTrellisTensors =
    Concat(kTrellisVectors, TrellisRules(kTrellisMatrices, kTrellisNames));
constexpr auto kTrellisMarkers = TrellisMarkerAliases(kTrellisNames);
// An untied head's codes are present.
constexpr std::array kTrellisHeadPresence{TensorPresenceRequirement{
    .flag = "tie_word_embeddings",
    .tensor = TrellisPartName(kTrellisNames, kTrellisMatrices.size() - 1, 0)}};

// The QKV biases of the families that may have them.
constexpr std::array kOptionalQkvBias{
    TensorEdit{.slot = "attention", .role = "q_bias", .optional = true},
    TensorEdit{.slot = "attention", .role = "k_bias", .optional = true},
    TensorEdit{.slot = "attention", .role = "v_bias", .optional = true},
};
constexpr auto kDenseGgufTensors = Overlay<kGgufTensors, kOptionalQkvBias, kNoTensorRules>();
constexpr auto kDenseTrellisTensors = Overlay<kTrellisTensors, kOptionalQkvBias, kNoTensorRules>();

// The transformers config keys (names.h): a decoder's, then each family's.
constexpr std::array kDecoderKeys{
    keys::kHeadDim, keys::kHiddenAct,  keys::kWidth,   keys::kFfnWidth, keys::kContext,
    keys::kHeads,   keys::kLayerCount, keys::kKvHeads, keys::kRmsEps,   keys::kRopeTheta};
constexpr auto kQwen2HfKeys = Concat(
    kDecoderKeys,
    std::array{keys::kRopeParametersTheta, KeyAlias{"use_sliding_window", "use_sliding_window"}});
constexpr auto kDenseHfKeys = Concat(
    kDecoderKeys,
    std::array{keys::kSlidingWindow, keys::kAttentionBias,
               KeyAlias{"rope_scaling.rope_type", "rope.scaling.type"},
               KeyAlias{"rope_scaling.type", "rope.scaling.type"},
               KeyAlias{"rope_scaling.factor", "rope.scaling.factor"},
               KeyAlias{"rope_scaling.original_max_position_embeddings",
                        "rope.scaling.original_context_length"},
               KeyAlias{"rope_scaling.beta_fast", "rope.scaling.yarn_beta_fast"},
               KeyAlias{"rope_scaling.beta_slow", "rope.scaling.yarn_beta_slow"},
               KeyAlias{"rope_scaling.low_freq_factor", "rope.scaling.low_freq_factor"},
               KeyAlias{"rope_scaling.high_freq_factor", "rope.scaling.high_freq_factor"},
               keys::kRopeParametersTheta, keys::kRopeParametersType,
               KeyAlias{"rope_parameters.factor", "rope.scaling.factor"},
               KeyAlias{"rope_parameters.original_max_position_embeddings",
                        "rope.scaling.original_context_length"},
               KeyAlias{"rope_parameters.low_freq_factor", "rope.scaling.low_freq_factor"},
               KeyAlias{"rope_parameters.high_freq_factor", "rope.scaling.high_freq_factor"}});

// The GGUF metadata a family's checkpoint carries beside the keys it reads.
constexpr std::array kLlamaAuxiliary{
    kGgufAuxiliary[0], kGgufAuxiliary[1], kGgufAuxiliary[2],
    AuxiliaryNamespace{"llama.vocab_size", "the vocabulary the token table states",
                       names::kTokenEmbd.gguf, 1}};
constexpr std::array kMistralAuxiliary{
    kGgufAuxiliary[0], kGgufAuxiliary[1], kGgufAuxiliary[2],
    AuxiliaryNamespace{"mistral.vocab_size", "the vocabulary the token table states",
                       names::kTokenEmbd.gguf, 1}};
// Tensors a safetensors checkpoint holds that no rule binds: transformers'
// rotation caches and the per-tensor activation and KV scales of an FP8
// export (unread: the matrices' own encodings are bound).
constexpr std::array<std::string_view, 11> kDenseHfIgnored{
    "model.rotary_emb.inv_freq",
    "model.layers.{layer}.self_attn.rotary_emb.inv_freq",
    "model.layers.{layer}.mlp.down_proj.input_scale",
    "model.layers.{layer}.mlp.gate_proj.input_scale",
    "model.layers.{layer}.mlp.up_proj.input_scale",
    "model.layers.{layer}.self_attn.q_proj.input_scale",
    "model.layers.{layer}.self_attn.k_proj.input_scale",
    "model.layers.{layer}.self_attn.v_proj.input_scale",
    "model.layers.{layer}.self_attn.o_proj.input_scale",
    "model.layers.{layer}.self_attn.k_proj.k_scale",
    "model.layers.{layer}.self_attn.v_proj.v_scale",
};
constexpr std::array<std::string_view, 2> kQwen2HfNulls{"rope_scaling", "rope_parameters"};
constexpr std::array<std::string_view, 3> kDenseHfNulls{"rope_scaling", "rope_parameters",
                                                        "sliding_window"};

// Shared pre-norm dense decoder parameter skeleton parameterized by RMS-norm
// epsilon transform and container RoPE rotation convention (`"normal"` for
// Llama/Mistral GGUF, `"neox"` for Qwen2 and HuggingFace/EXL3 checkpoints).
#define LLMP_DENSE_DECODER_CORE_PARAMS(EPS_TRANSFORM, ROTATION)                                 \
  FromKey(kStageSlot, "context", "context_length", kInt),                                       \
      FromKey(kStreamSlot, "width", "embedding_length", kInt),                                  \
      Name("attention", "placement", "pre"),                                                    \
      FromKey("attention", "eps", "attention.layer_norm_rms_epsilon", kReal, EPS_TRANSFORM),    \
      FromKey("attention", "heads", "attention.head_count", kInt),                              \
      FromKey("attention", "kv_heads", "attention.head_count_kv", kInt),                        \
      FromOrRatio("attention", "key_dim", "attention.key_length", "embedding_length",           \
                  "attention.head_count", kInt),                                                \
      FromKey("attention", "value_dim", "attention.value_length", kInt),                        \
      FromOrRatio("attention", "value_dim", "attention.key_length", "embedding_length",         \
                  "attention.head_count", kInt),                                                \
      FromOrRatio("attention", "scale", "attention.key_length", "embedding_length",             \
                  "attention.head_count", kReal, Transform::kRsqrt),                            \
      FromKey("attention", "window", "attention.sliding_window", kInt),                         \
      Int("attention", "window", 0), Name("attention", "rotation", ROTATION),                   \
      FromKey("attention", "rope_dims", "rope.dimension_count", kInt),                          \
      FromOrRatio("attention", "rope_dims", "attention.key_length", "embedding_length",         \
                  "attention.head_count", kInt),                                                \
      FromKey("attention", "rope_base", "rope.freq_base", kReal),                               \
      Real("attention", "rope_base", 10000.0),                                                  \
      FromTensor("attention", "rope_freqs", names::kRopeFreqs.gguf),                            \
      Flag("attention", "scale_scores", true), Flag("attention", "explicit_scores", false),     \
      Name("ffn", "placement", "pre"),                                                          \
      FromKey("ffn", "eps", "attention.layer_norm_rms_epsilon", kReal, EPS_TRANSFORM),          \
      Name("ffn", "activation", "silu"), FromKey("ffn", "hidden", "feed_forward_length", kInt), \
      FromKey("final_norm", "eps", "attention.layer_norm_rms_epsilon", kReal, EPS_TRANSFORM)

// The dense decoders' rope.scaling.type: absent, none or default leave the
// frequencies unscaled (a factor other than 1 beside them is ambiguous: YaRN
// in the attention component's heuristic, linear in llama.cpp's reading of a
// GGUF, ignored by transformers' default; refused); yarn and llama3 select
// their component flags and read the factor, original context and their own
// shape keys; linear is a frequency scale of 1 / factor, never YaRN. Any
// other type (dynamic, longrope, ...) is refused by the flags' string match,
// never run as YaRN. It holds the rope_scale rules (linear's, then the
// unscaled constant); the core parameters leave rope_scale to their users.
#define LLMP_DENSE_DECODER_ROPE_SCALING_PARAMS                                                    \
  KeyEquals("attention", "yarn", "rope.scaling.type", "yarn"),                                    \
      KeyEquals("attention", "llama3_rope", "rope.scaling.type", "llama3"),                       \
      When(FromKey("attention", "rope_scale", "rope.scaling.factor", kReal,                       \
                   Transform::kReciprocal, {}, true),                                             \
           "rope.scaling.type", "linear"),                                                        \
      Real("attention", "rope_scale", 1.0),                                                       \
      LLMP_DENSE_DECODER_SCALED_KEY("rope_factor", "rope.scaling.factor", kReal, "yarn"),         \
      LLMP_DENSE_DECODER_SCALED_KEY("rope_factor", "rope.scaling.factor", kReal, "llama3"),       \
      LLMP_DENSE_DECODER_SCALED_KEY("yarn_context", "rope.scaling.original_context_length", kInt, \
                                    "yarn"),                                                      \
      LLMP_DENSE_DECODER_SCALED_KEY("yarn_context", "rope.scaling.original_context_length", kInt, \
                                    "llama3"),                                                    \
      LLMP_DENSE_DECODER_SCALED_KEY("beta_fast", "rope.scaling.yarn_beta_fast", kReal, "yarn"),   \
      LLMP_DENSE_DECODER_SCALED_KEY("beta_slow", "rope.scaling.yarn_beta_slow", kReal, "yarn"),   \
      LLMP_DENSE_DECODER_SCALED_KEY("low_freq_factor", "rope.scaling.low_freq_factor", kReal,     \
                                    "llama3"),                                                    \
      LLMP_DENSE_DECODER_SCALED_KEY("high_freq_factor", "rope.scaling.high_freq_factor", kReal,   \
                                    "llama3"),                                                    \
      When(RequireReal("rope.scaling.factor", 1.0, true), "rope.scaling.type", {}),               \
      When(RequireReal("rope.scaling.factor", 1.0, true), "rope.scaling.type", "none"),           \
      When(RequireReal("rope.scaling.factor", 1.0, true), "rope.scaling.type", "default")
#define LLMP_DENSE_DECODER_SCALED_KEY(PARAM, KEY, TYPE, SCALING)                                \
  When(FromKey("attention", PARAM, KEY, TYPE, Transform::kNone, {}, true), "rope.scaling.type", \
       SCALING)

#define LLMP_DENSE_DECODER_OPTIONAL_QKV_BIAS                                       \
  FromKey("attention", "qkv_bias", "attention_bias", kBool),                       \
      FromTensor("attention", "qkv_bias", "blk.0.attn_q.bias"),                    \
      FromTensor("attention", "qkv_bias", "model.layers.0.self_attn.q_proj.bias"), \
      Flag("attention", "qkv_bias", false)

// Qwen2's single-slot policy preserves the frozen v1 digests of the approved
// Qwen2.5-0.5B GGUF and EXL3 fixtures (no capture/row-invariance claim is made
// for EXL3).
constexpr std::array kQwen2Policies = {
    PolicyRule{.text = "max_decode_rows 1\n", .owners_alone = true, .capture_ahead = false},
};
constexpr std::array kQwen2Params{
    LLMP_DENSE_DECODER_CORE_PARAMS(Transform::kNone, "neox"),
    Real("attention", "rope_scale", 1.0),
    Flag("attention", "qkv_bias", true),
};
constexpr std::array kQwen2GgufExtras{Require("rope.scaling.type", "none", true)};
constexpr std::array kQwen2HfExtras{
    Require("hidden_act", "silu", false),
    ParamRule{.key = "use_sliding_window", .type = kBool, .value = {.flag = false}},
};
// EXL3 checkpoints' norms run at their float32 epsilon (exllamav3's).
constexpr std::array kFloat32Eps{
    ParamOverride{.slot = "attention",
                  .param = "eps",
                  .key = "attention.layer_norm_rms_epsilon",
                  .rule = FromKey("attention", "eps", "attention.layer_norm_rms_epsilon", kReal,
                                  Transform::kFloat32)},
    ParamOverride{.slot = "ffn",
                  .param = "eps",
                  .key = "attention.layer_norm_rms_epsilon",
                  .rule = FromKey("ffn", "eps", "attention.layer_norm_rms_epsilon", kReal,
                                  Transform::kFloat32)},
    ParamOverride{.slot = "final_norm",
                  .param = "eps",
                  .key = "attention.layer_norm_rms_epsilon",
                  .rule = FromKey("final_norm", "eps", "attention.layer_norm_rms_epsilon", kReal,
                                  Transform::kFloat32)},
};
// Source spellings and container conventions for pre-norm dense decoders:
// one canonical table, each container's dialect its spellings (names.h),
// its matrices' parts and its source keys.
constexpr std::array<std::string_view, 3> kContainers{"gguf", "safetensors", "exl3"};
constexpr std::array kQwen2Dialects{
    SourceDialect{.architecture = "qwen2",
                  .container = "gguf",
                  .params = kQwen2GgufExtras,
                  .auxiliary = kGgufAuxiliary},
    SourceDialect{.architecture = "qwen2",
                  .container = "safetensors",
                  .params = kQwen2HfExtras,
                  .parts = kHfMatrixParts,
                  .tensor_aliases = kHfTensors,
                  .key_aliases = kQwen2HfKeys,
                  .auxiliary = kSafetensorsAuxiliary,
                  .semantic = kHfSemantic,
                  .ignored_tensors = kHfIgnored,
                  .null_semantics = kQwen2HfNulls,
                  .tensor_presence = kHfHeadPresence},
    SourceDialect{.architecture = "qwen2",
                  .container = "exl3",
                  .overrides = kFloat32Eps,
                  .params = kQwen2HfExtras,
                  .tensors = kTrellisTensors,
                  .tensor_aliases = kTrellisMarkers,
                  .key_aliases = kQwen2HfKeys,
                  .auxiliary = kSafetensorsAuxiliary,
                  .semantic = kHfSemantic,
                  .ignored_tensors = kHfIgnored,
                  .null_semantics = kQwen2HfNulls,
                  .tensor_presence = kTrellisHeadPresence},
};
constexpr Mapping kQwen2Mapping{
    .architecture = "qwen2",
    .version = 1,
    .pipeline = PipelineKind::kLlm,
    .layer_count_key = "block_count",
    .layer_types = "g",
    .slots = kSlots,
    .tensors = kGgufTensors,
    .params = kQwen2Params,
    .policies = kQwen2Policies,
    .containers = kContainers,
    .dialects = kQwen2Dialects,
};

constexpr std::array kDenseCanonicalParams{
    LLMP_DENSE_DECODER_CORE_PARAMS(Transform::kNone, "normal"),
    LLMP_DENSE_DECODER_ROPE_SCALING_PARAMS,
    LLMP_DENSE_DECODER_OPTIONAL_QKV_BIAS,
};
constexpr std::array kDenseHfExtras{Require("hidden_act", "silu", true)};
constexpr std::array kDenseHfOverrides{
    ParamOverride{.slot = "attention",
                  .param = "rotation",
                  .key = std::string_view{},
                  .rule = Name("attention", "rotation", "neox")},
};
constexpr auto kDenseTrellisOverrides = Concat(kFloat32Eps, kDenseHfOverrides);

// Llama and Mistral differ only in their architecture names and their GGUF
// vocabulary keys.
constexpr std::array<SourceDialect, 3> DenseDialects(std::string_view architecture,
                                                     std::span<const AuxiliaryNamespace> gguf) {
  return {SourceDialect{.architecture = architecture, .container = "gguf", .auxiliary = gguf},
          SourceDialect{.architecture = architecture,
                        .container = "safetensors",
                        .overrides = kDenseHfOverrides,
                        .params = kDenseHfExtras,
                        .parts = kHfMatrixParts,
                        .tensor_aliases = kHfTensors,
                        .key_aliases = kDenseHfKeys,
                        .auxiliary = kGemmaHfAuxiliary,
                        .semantic = kHfSemantic,
                        .ignored_tensors = kDenseHfIgnored,
                        .null_semantics = kDenseHfNulls,
                        .tensor_presence = kHfHeadPresence},
          SourceDialect{.architecture = architecture,
                        .container = "exl3",
                        .overrides = kDenseTrellisOverrides,
                        .params = kDenseHfExtras,
                        .tensors = kDenseTrellisTensors,
                        .tensor_aliases = kTrellisMarkers,
                        .key_aliases = kDenseHfKeys,
                        .auxiliary = kGemmaHfAuxiliary,
                        .semantic = kHfSemantic,
                        .ignored_tensors = kHfIgnored,
                        .null_semantics = kDenseHfNulls,
                        .tensor_presence = kTrellisHeadPresence}};
}
constexpr Mapping DenseMapping(std::string_view architecture,
                               std::span<const SourceDialect> dialects) {
  return {.architecture = architecture,
          .version = 1,
          .pipeline = PipelineKind::kLlm,
          .layer_count_key = "block_count",
          .layer_types = "g",
          .slots = kSlots,
          .tensors = kDenseGgufTensors,
          .params = kDenseCanonicalParams,
          .policies = kDensePolicies,
          .containers = kContainers,
          .dialects = dialects};
}
constexpr auto kLlamaDialects = DenseDialects("llama", kLlamaAuxiliary);
constexpr auto kMistralDialects = DenseDialects("mistral", kMistralAuxiliary);
constexpr Mapping kLlamaMapping = DenseMapping("llama", kLlamaDialects);
constexpr Mapping kMistralMapping = DenseMapping("mistral", kMistralDialects);
}  // namespace

const Mapping& Qwen2Mapping() { return kQwen2Mapping; }
const Mapping& LlamaMapping() { return kLlamaMapping; }
const Mapping& MistralMapping() { return kMistralMapping; }
}  // namespace llmp::spec::mappings
