// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "spec/mappings/qwen2.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <expected>
#include <ranges>
#include <string>
#include <vector>

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

constexpr std::array kGgufTensors = {
    TensorRule{.pattern = "token_embd.weight", .slot = "embedding", .role = "table"},
    TensorRule{.pattern = "blk.{layer}.attn_norm.weight", .slot = "attention", .role = "norm"},
    TensorRule{.pattern = "blk.{layer}.attn_q.weight", .slot = "attention", .role = "q"},
    TensorRule{.pattern = "blk.{layer}.attn_k.weight", .slot = "attention", .role = "k"},
    TensorRule{.pattern = "blk.{layer}.attn_v.weight", .slot = "attention", .role = "v"},
    TensorRule{.pattern = "blk.{layer}.attn_q.bias", .slot = "attention", .role = "q_bias"},
    TensorRule{.pattern = "blk.{layer}.attn_k.bias", .slot = "attention", .role = "k_bias"},
    TensorRule{.pattern = "blk.{layer}.attn_v.bias", .slot = "attention", .role = "v_bias"},
    TensorRule{.pattern = "rope_freqs.weight",
               .slot = "attention",
               .role = "rope_freqs",
               .optional = true},
    TensorRule{.pattern = "blk.{layer}.attn_output.weight", .slot = "attention", .role = "output"},
    TensorRule{.pattern = "blk.{layer}.ffn_norm.weight", .slot = "ffn", .role = "norm"},
    TensorRule{.pattern = "blk.{layer}.ffn_gate.weight", .slot = "ffn", .role = "gate"},
    TensorRule{.pattern = "blk.{layer}.ffn_up.weight", .slot = "ffn", .role = "up"},
    TensorRule{.pattern = "blk.{layer}.ffn_down.weight", .slot = "ffn", .role = "down"},
    TensorRule{.pattern = "output_norm.weight", .slot = "final_norm", .role = "weight"},
    TensorRule{.pattern = "output.weight",
               .slot = "output",
               .role = "weight",
               .fallback = "token_embd.weight"},
};

constexpr std::array kExl3TensorRules = {
    TensorRule{.pattern = "model.embed_tokens.weight", .slot = "embedding", .role = "table"},
    TensorRule{.pattern = "model.layers.{layer}.input_layernorm.weight",
               .slot = "attention",
               .role = "norm"},
    TensorRule{.pattern = "model.layers.{layer}.self_attn.q_proj.bias",
               .slot = "attention",
               .role = "q_bias"},
    TensorRule{.pattern = "model.layers.{layer}.self_attn.k_proj.bias",
               .slot = "attention",
               .role = "k_bias"},
    TensorRule{.pattern = "model.layers.{layer}.self_attn.v_proj.bias",
               .slot = "attention",
               .role = "v_bias"},
    TensorRule{.pattern = "model.layers.{layer}.post_attention_layernorm.weight",
               .slot = "ffn",
               .role = "norm"},
    TensorRule{.pattern = "model.norm.weight", .slot = "final_norm", .role = "weight"},
    TensorRule{.pattern = "model.layers.{layer}.self_attn.q_proj.trellis",
               .slot = "attention",
               .role = "q",
               .part = format::PartRole::kCodes},
    TensorRule{.pattern = "model.layers.{layer}.self_attn.q_proj.suh",
               .slot = "attention",
               .role = "q",
               .part = format::PartRole::kInputSigns},
    TensorRule{.pattern = "model.layers.{layer}.self_attn.q_proj.svh",
               .slot = "attention",
               .role = "q",
               .part = format::PartRole::kOutputSigns},
    TensorRule{.pattern = "model.layers.{layer}.self_attn.q_proj.mcg",
               .slot = "attention",
               .role = "q",
               .part = format::PartRole::kCodebookMarker,
               .optional = true},
    TensorRule{.pattern = "model.layers.{layer}.self_attn.k_proj.trellis",
               .slot = "attention",
               .role = "k",
               .part = format::PartRole::kCodes},
    TensorRule{.pattern = "model.layers.{layer}.self_attn.k_proj.suh",
               .slot = "attention",
               .role = "k",
               .part = format::PartRole::kInputSigns},
    TensorRule{.pattern = "model.layers.{layer}.self_attn.k_proj.svh",
               .slot = "attention",
               .role = "k",
               .part = format::PartRole::kOutputSigns},
    TensorRule{.pattern = "model.layers.{layer}.self_attn.k_proj.mcg",
               .slot = "attention",
               .role = "k",
               .part = format::PartRole::kCodebookMarker,
               .optional = true},
    TensorRule{.pattern = "model.layers.{layer}.self_attn.v_proj.trellis",
               .slot = "attention",
               .role = "v",
               .part = format::PartRole::kCodes},
    TensorRule{.pattern = "model.layers.{layer}.self_attn.v_proj.suh",
               .slot = "attention",
               .role = "v",
               .part = format::PartRole::kInputSigns},
    TensorRule{.pattern = "model.layers.{layer}.self_attn.v_proj.svh",
               .slot = "attention",
               .role = "v",
               .part = format::PartRole::kOutputSigns},
    TensorRule{.pattern = "model.layers.{layer}.self_attn.v_proj.mcg",
               .slot = "attention",
               .role = "v",
               .part = format::PartRole::kCodebookMarker,
               .optional = true},
    TensorRule{.pattern = "model.layers.{layer}.self_attn.o_proj.trellis",
               .slot = "attention",
               .role = "output",
               .part = format::PartRole::kCodes},
    TensorRule{.pattern = "model.layers.{layer}.self_attn.o_proj.suh",
               .slot = "attention",
               .role = "output",
               .part = format::PartRole::kInputSigns},
    TensorRule{.pattern = "model.layers.{layer}.self_attn.o_proj.svh",
               .slot = "attention",
               .role = "output",
               .part = format::PartRole::kOutputSigns},
    TensorRule{.pattern = "model.layers.{layer}.self_attn.o_proj.mcg",
               .slot = "attention",
               .role = "output",
               .part = format::PartRole::kCodebookMarker,
               .optional = true},
    TensorRule{.pattern = "model.layers.{layer}.mlp.gate_proj.trellis",
               .slot = "ffn",
               .role = "gate",
               .part = format::PartRole::kCodes},
    TensorRule{.pattern = "model.layers.{layer}.mlp.gate_proj.suh",
               .slot = "ffn",
               .role = "gate",
               .part = format::PartRole::kInputSigns},
    TensorRule{.pattern = "model.layers.{layer}.mlp.gate_proj.svh",
               .slot = "ffn",
               .role = "gate",
               .part = format::PartRole::kOutputSigns},
    TensorRule{.pattern = "model.layers.{layer}.mlp.gate_proj.mcg",
               .slot = "ffn",
               .role = "gate",
               .part = format::PartRole::kCodebookMarker,
               .optional = true},
    TensorRule{.pattern = "model.layers.{layer}.mlp.up_proj.trellis",
               .slot = "ffn",
               .role = "up",
               .part = format::PartRole::kCodes},
    TensorRule{.pattern = "model.layers.{layer}.mlp.up_proj.suh",
               .slot = "ffn",
               .role = "up",
               .part = format::PartRole::kInputSigns},
    TensorRule{.pattern = "model.layers.{layer}.mlp.up_proj.svh",
               .slot = "ffn",
               .role = "up",
               .part = format::PartRole::kOutputSigns},
    TensorRule{.pattern = "model.layers.{layer}.mlp.up_proj.mcg",
               .slot = "ffn",
               .role = "up",
               .part = format::PartRole::kCodebookMarker,
               .optional = true},
    TensorRule{.pattern = "model.layers.{layer}.mlp.down_proj.trellis",
               .slot = "ffn",
               .role = "down",
               .part = format::PartRole::kCodes},
    TensorRule{.pattern = "model.layers.{layer}.mlp.down_proj.suh",
               .slot = "ffn",
               .role = "down",
               .part = format::PartRole::kInputSigns},
    TensorRule{.pattern = "model.layers.{layer}.mlp.down_proj.svh",
               .slot = "ffn",
               .role = "down",
               .part = format::PartRole::kOutputSigns},
    TensorRule{.pattern = "model.layers.{layer}.mlp.down_proj.mcg",
               .slot = "ffn",
               .role = "down",
               .part = format::PartRole::kCodebookMarker,
               .optional = true},
    TensorRule{.pattern = "lm_head.trellis",
               .slot = "output",
               .role = "weight",
               .part = format::PartRole::kCodes},
    TensorRule{.pattern = "lm_head.suh",
               .slot = "output",
               .role = "weight",
               .part = format::PartRole::kInputSigns},
    TensorRule{.pattern = "lm_head.svh",
               .slot = "output",
               .role = "weight",
               .part = format::PartRole::kOutputSigns},
    TensorRule{.pattern = "lm_head.mcg",
               .slot = "output",
               .role = "weight",
               .part = format::PartRole::kCodebookMarker,
               .optional = true},
};

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
      Real("attention", "rope_base", 10000.0), Real("attention", "rope_scale", 1.0),            \
      FromTensor("attention", "rope_freqs", "rope_freqs.weight"),                               \
      Flag("attention", "scale_scores", true), Flag("attention", "explicit_scores", false),     \
      Name("ffn", "placement", "pre"),                                                          \
      FromKey("ffn", "eps", "attention.layer_norm_rms_epsilon", kReal, EPS_TRANSFORM),          \
      Name("ffn", "activation", "silu"), FromKey("ffn", "hidden", "feed_forward_length", kInt), \
      FromKey("final_norm", "eps", "attention.layer_norm_rms_epsilon", kReal, EPS_TRANSFORM)

#define LLMP_DENSE_DECODER_YARN_PARAMS                                                          \
  KeyEquals("attention", "yarn", "rope.scaling.type", "yarn"),                                  \
      KeyEquals("attention", "llama3_rope", "rope.scaling.type", "llama3"),                     \
      FromKey("attention", "rope_factor", "rope.scaling.factor", kReal, Transform::kNone, {},   \
              true),                                                                            \
      FromKey("attention", "yarn_context", "rope.scaling.original_context_length", kInt,        \
              Transform::kNone, {}, true),                                                      \
      FromKey("attention", "beta_fast", "rope.scaling.yarn_beta_fast", kReal, Transform::kNone, \
              {}, true),                                                                        \
      FromKey("attention", "beta_slow", "rope.scaling.yarn_beta_slow", kReal, Transform::kNone, \
              {}, true),                                                                        \
      FromKey("attention", "low_freq_factor", "rope.scaling.low_freq_factor", kReal,            \
              Transform::kNone, {}, true),                                                      \
      FromKey("attention", "high_freq_factor", "rope.scaling.high_freq_factor", kReal,          \
              Transform::kNone, {}, true)

#define LLMP_DENSE_DECODER_OPTIONAL_QKV_BIAS                                       \
  FromKey("attention", "qkv_bias", "attention_bias", kBool),                       \
      FromTensor("attention", "qkv_bias", "blk.0.attn_q.bias"),                    \
      FromTensor("attention", "qkv_bias", "model.layers.0.self_attn.q_proj.bias"), \
      Flag("attention", "qkv_bias", false)

// Source spellings and container conventions for pre-norm dense decoders.
// Each dialect selects its own wire table while keeping canonical slot/role
// identities shared by the architecture.
constexpr std::array<std::string_view, 3> kContainers{"gguf", "safetensors", "exl3"};
constexpr auto kQwen2SafetensorsTensors = [] {
  auto out = kGgufTensors;
  for (auto& t : out)
    if (t.role == "table" || t.role == "q" || t.role == "k" || t.role == "v" ||
        t.role == "output" || t.role == "gate" || t.role == "up" || t.role == "down" ||
        (t.slot == "output" && t.role == "weight"))
      t.any_part = true;
  return out;
}();
constexpr auto kDenseGgufTensors = [] {
  auto out = kGgufTensors;
  for (auto& t : out)
    if (t.role == "q_bias" || t.role == "k_bias" || t.role == "v_bias") t.optional = true;
  return out;
}();
constexpr auto kDenseSafetensorsTensors = [] {
  auto out = kDenseGgufTensors;
  for (auto& t : out)
    if (t.role == "table" || t.role == "q" || t.role == "k" || t.role == "v" ||
        t.role == "output" || t.role == "gate" || t.role == "up" || t.role == "down" ||
        (t.slot == "output" && t.role == "weight"))
      t.any_part = true;
  return out;
}();
constexpr auto kDenseExl3Tensors = [] {
  auto out = kExl3TensorRules;
  for (auto& t : out)
    if (t.role == "q_bias" || t.role == "k_bias" || t.role == "v_bias") t.optional = true;
  return out;
}();
// Qwen2's single-slot policy preserves the frozen v1 digests of the approved
// Qwen2.5-0.5B GGUF and EXL3 fixtures (no capture/row-invariance claim is made
// for EXL3).
constexpr std::array kQwen2Policies = {
    PolicyRule{.text = "max_decode_rows 1\n", .owners_alone = true, .capture_ahead = false},
};
constexpr std::array kQwen2Params{
    LLMP_DENSE_DECODER_CORE_PARAMS(Transform::kNone, "neox"),
    Flag("attention", "qkv_bias", true),
};
constexpr std::array kQwen2GgufExtras{Require("rope.scaling.type", "none", true)};
constexpr std::array kQwen2HfExtras{
    Require("hidden_act", "silu", false),
    ParamRule{.key = "use_sliding_window", .type = kBool, .value = {.flag = false}},
};
constexpr std::array kQwen2FloatOverrides{
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
constexpr std::array kQwen2Dialects{
    SourceDialect{.architecture = "qwen2",
                  .container = "gguf",
                  .params = kQwen2GgufExtras,
                  .auxiliary = kGgufAuxiliary},
    SourceDialect{.architecture = "qwen2",
                  .container = "safetensors",
                  .params = kQwen2HfExtras,
                  .tensors = kQwen2SafetensorsTensors,
                  .tensor_aliases = kDenseHfTensors,
                  .key_aliases = kQwen2SafetensorsKeys,
                  .auxiliary = kSafetensorsAuxiliary,
                  .semantic = kHfSemantic,
                  .ignored_tensors = kHfIgnored,
                  .null_semantics = kHfNullSemantics,
                  .tensor_presence = kHfHeadPresence},
    SourceDialect{.architecture = "qwen2",
                  .container = "exl3",
                  .overrides = kQwen2FloatOverrides,
                  .params = kQwen2HfExtras,
                  .tensors = kExl3TensorRules,
                  .tensor_aliases = kExl3Tensors,
                  .key_aliases = kQwen2SafetensorsKeys,
                  .auxiliary = kSafetensorsAuxiliary,
                  .semantic = kHfSemantic,
                  .ignored_tensors = kHfIgnored,
                  .null_semantics = kHfNullSemantics,
                  .tensor_presence = kExl3HeadPresence},
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

// Generic dense decoder policy: batched decode up to 16 rows with the shared
// runner's default lookahead and graph capture, paired MMQ dense products, and
// solo prompt chunks.
constexpr std::array kDensePolicies = {
    PolicyRule{.text = "max_decode_rows 16\nvector_rows 16\nproduct dense_pair\n",
               .prompts_alone = true},
};
constexpr std::array kDenseCanonicalParams{
    LLMP_DENSE_DECODER_CORE_PARAMS(Transform::kNone, "normal"),
    LLMP_DENSE_DECODER_YARN_PARAMS,
    LLMP_DENSE_DECODER_OPTIONAL_QKV_BIAS,
};
constexpr std::array kDenseHfExtras{Require("hidden_act", "silu", true)};
constexpr std::array kDenseHfOverrides{
    ParamOverride{.slot = "attention",
                  .param = "rotation",
                  .key = std::string_view{},
                  .rule = Name("attention", "rotation", "neox")},
};
constexpr std::array kDenseExl3Overrides{
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
    ParamOverride{.slot = "attention",
                  .param = "rotation",
                  .key = std::string_view{},
                  .rule = Name("attention", "rotation", "neox")},
};

#define LLMP_DEFINE_DENSE_FAMILY_MAPPING(NAME, ARCH, GGUF_AUX)                                    \
  constexpr auto k##NAME##Dialects =                                                              \
      std::array{SourceDialect{.architecture = ARCH, .container = "gguf", .auxiliary = GGUF_AUX}, \
                 SourceDialect{.architecture = ARCH,                                              \
                               .container = "safetensors",                                        \
                               .overrides = kDenseHfOverrides,                                    \
                               .params = kDenseHfExtras,                                          \
                               .tensors = kDenseSafetensorsTensors,                               \
                               .tensor_aliases = kDenseHfTensors,                                 \
                               .key_aliases = kHfDenseKeys,                                       \
                               .auxiliary = kGemmaHfAuxiliary,                                    \
                               .semantic = kHfSemantic,                                           \
                               .ignored_tensors = kDenseSafetensorsIgnored,                       \
                               .null_semantics = kDenseHfNulls,                                   \
                               .tensor_presence = kHfHeadPresence},                               \
                 SourceDialect{.architecture = ARCH,                                              \
                               .container = "exl3",                                               \
                               .overrides = kDenseExl3Overrides,                                  \
                               .params = kDenseHfExtras,                                          \
                               .tensors = kDenseExl3Tensors,                                      \
                               .tensor_aliases = kExl3Tensors,                                    \
                               .key_aliases = kHfDenseKeys,                                       \
                               .auxiliary = kGemmaHfAuxiliary,                                    \
                               .semantic = kHfSemantic,                                           \
                               .ignored_tensors = kHfIgnored,                                     \
                               .null_semantics = kDenseHfNulls,                                   \
                               .tensor_presence = kExl3HeadPresence}};                            \
  constexpr Mapping k##NAME##Mapping{.architecture = ARCH,                                        \
                                     .version = 1,                                                \
                                     .pipeline = PipelineKind::kLlm,                              \
                                     .layer_count_key = "block_count",                            \
                                     .layer_types = "g",                                          \
                                     .slots = kSlots,                                             \
                                     .tensors = kDenseGgufTensors,                                \
                                     .params = kDenseCanonicalParams,                             \
                                     .policies = kDensePolicies,                                  \
                                     .containers = kContainers,                                   \
                                     .dialects = k##NAME##Dialects};

LLMP_DEFINE_DENSE_FAMILY_MAPPING(Llama, "llama", kLlamaAuxiliary)
LLMP_DEFINE_DENSE_FAMILY_MAPPING(Mistral, "mistral", kMistralAuxiliary)
#undef LLMP_DEFINE_DENSE_FAMILY_MAPPING
// ------------------------------------------------------------------ qwen35

constexpr std::array<std::string_view, 2> kQwen35SourceArchitectures{"qwen3_5", "qwen3_5_text"};
constexpr std::array<std::string_view, 3> kQwen35Containers{"gguf", "safetensors", "mlx"};
constexpr std::array kQwen35LayerTypeCodes{
    LayerTypeCode{.value = 0, .type = 'l', .label = "linear_attention"},
    LayerTypeCode{.value = 1, .type = 'f', .label = "full_attention"},
};
constexpr std::array kQwen35Slots = {
    SlotRule{.slot = "embedding", .site = Site::kEmbedding, .kind = "embedding.token"},
    SlotRule{.slot = "linear", .site = Site::kLayer, .kind = "attention.gdn", .layer_types = "l"},
    SlotRule{
        .slot = "attention", .site = Site::kLayer, .kind = "attention.dense", .layer_types = "f"},
    SlotRule{.slot = "moe", .site = Site::kLayer, .kind = "ffn.moe", .gate = "expert_count"},
    SlotRule{.slot = "ffn",
             .site = Site::kLayer,
             .kind = "ffn.glu",
             .gate = "expert_count",
             .gate_absent = true},
    SlotRule{.slot = "final_norm", .site = Site::kFinalNorm, .kind = "norm.rms"},
    SlotRule{.slot = "output", .site = Site::kOutput, .kind = "head.linear"},
};
constexpr std::array kQwen35Folds = {
    FoldRule{.fold = "norm.plus_one", .container = "gguf"},
    FoldRule{.fold = "ssm.a_neg_exp", .container = "gguf"},
    FoldRule{.fold = "gdn.value_heads_tiled", .container = "gguf"},
    FoldRule{.fold = "norm.plus_one", .container = "safetensors", .imported = true},
    FoldRule{.fold = "ssm.a_neg_exp", .container = "safetensors", .imported = true},
    FoldRule{.fold = "gdn.value_heads_tiled", .container = "safetensors", .imported = true},
    // MLX conversion stores the RMS scale itself; HF stores gamma - 1.
    FoldRule{.fold = "norm.plus_one", .container = "mlx"},
    FoldRule{.fold = "ssm.a_neg_exp", .container = "mlx", .imported = true},
    FoldRule{.fold = "gdn.value_heads_tiled", .container = "mlx", .imported = true},
};
constexpr std::array kQwen35Tensors = {
    TensorRule{
        .pattern = "token_embd.weight", .slot = "embedding", .role = "table", .any_part = true},
    TensorRule{.pattern = "blk.{layer}.attn_norm.weight",
               .slot = "linear",
               .role = "input_norm",
               .layer_types = "l"},
    TensorRule{.pattern = "blk.{layer}.attn_qkv.weight",
               .slot = "linear",
               .role = "qkv",
               .layer_types = "l",
               .any_part = true},
    TensorRule{.pattern = "blk.{layer}.attn_gate.weight",
               .slot = "linear",
               .role = "gate",
               .layer_types = "l",
               .any_part = true},
    TensorRule{.pattern = "blk.{layer}.ssm_beta.weight",
               .slot = "linear",
               .role = "beta",
               .layer_types = "l",
               .any_part = true},
    TensorRule{.pattern = "blk.{layer}.ssm_alpha.weight",
               .slot = "linear",
               .role = "alpha",
               .layer_types = "l",
               .any_part = true},
    TensorRule{.pattern = "blk.{layer}.ssm_dt.bias",
               .slot = "linear",
               .role = "dt_bias",
               .layer_types = "l"},
    TensorRule{.pattern = "blk.{layer}.ssm_a", .slot = "linear", .role = "a", .layer_types = "l"},
    TensorRule{.pattern = "blk.{layer}.ssm_conv1d.weight",
               .slot = "linear",
               .role = "conv",
               .layer_types = "l"},
    TensorRule{.pattern = "blk.{layer}.ssm_norm.weight",
               .slot = "linear",
               .role = "norm",
               .layer_types = "l"},
    TensorRule{.pattern = "blk.{layer}.ssm_out.weight",
               .slot = "linear",
               .role = "output",
               .layer_types = "l",
               .any_part = true},
    TensorRule{.pattern = "blk.{layer}.attn_norm.weight",
               .slot = "attention",
               .role = "norm",
               .layer_types = "f"},
    TensorRule{.pattern = "blk.{layer}.attn_q.weight",
               .slot = "attention",
               .role = "q",
               .layer_types = "f",
               .any_part = true},
    TensorRule{.pattern = "blk.{layer}.attn_k.weight",
               .slot = "attention",
               .role = "k",
               .layer_types = "f",
               .any_part = true},
    TensorRule{.pattern = "blk.{layer}.attn_v.weight",
               .slot = "attention",
               .role = "v",
               .layer_types = "f",
               .any_part = true},
    TensorRule{.pattern = "blk.{layer}.attn_q_norm.weight",
               .slot = "attention",
               .role = "q_norm",
               .layer_types = "f"},
    TensorRule{.pattern = "blk.{layer}.attn_k_norm.weight",
               .slot = "attention",
               .role = "k_norm",
               .layer_types = "f"},
    TensorRule{.pattern = "blk.{layer}.attn_output.weight",
               .slot = "attention",
               .role = "output",
               .layer_types = "f",
               .any_part = true},
    TensorRule{.pattern = "blk.{layer}.ffn_norm.weight", .slot = "moe", .role = "norm"},
    TensorRule{.pattern = "blk.{layer}.ffn_gate_inp.weight",
               .slot = "moe",
               .role = "router",
               .any_part = true},
    TensorRule{.pattern = "blk.{layer}.ffn_gate_inp_shexp.weight",
               .slot = "moe",
               .role = "shared_gate",
               .optional = true,
               .any_part = true},
    TensorRule{.pattern = "blk.{layer}.ffn_gate_shexp.weight",
               .slot = "moe",
               .role = "gate",
               .optional = true,
               .any_part = true},
    TensorRule{.pattern = "blk.{layer}.ffn_up_shexp.weight",
               .slot = "moe",
               .role = "up",
               .optional = true,
               .any_part = true},
    TensorRule{.pattern = "blk.{layer}.ffn_down_shexp.weight",
               .slot = "moe",
               .role = "down",
               .optional = true,
               .any_part = true},
    TensorRule{.pattern = "blk.{layer}.ffn_gate_up_exps.weight",
               .slot = "moe",
               .role = "gate_up_experts",
               .experts = true,
               .optional = true,
               .any_part = true},
    TensorRule{.pattern = "blk.{layer}.ffn_gate_exps.weight",
               .slot = "moe",
               .role = "gate_experts",
               .experts = true,
               .optional = true,
               .any_part = true},
    TensorRule{.pattern = "blk.{layer}.ffn_up_exps.weight",
               .slot = "moe",
               .role = "up_experts",
               .experts = true,
               .optional = true,
               .any_part = true},
    TensorRule{.pattern = "blk.{layer}.ffn_down_exps.weight",
               .slot = "moe",
               .role = "down_experts",
               .experts = true,
               .optional = true,
               .any_part = true},
    TensorRule{.pattern = "blk.{layer}.ffn_gate_exps.weight_scale_2",
               .slot = "moe",
               .role = "gate_scale",
               .optional = true},
    TensorRule{.pattern = "blk.{layer}.ffn_up_exps.weight_scale_2",
               .slot = "moe",
               .role = "up_scale",
               .optional = true},
    TensorRule{.pattern = "blk.{layer}.ffn_down_exps.weight_scale_2",
               .slot = "moe",
               .role = "down_scale",
               .optional = true},
    TensorRule{.pattern = "blk.{layer}.ffn_norm.weight", .slot = "ffn", .role = "norm"},
    TensorRule{
        .pattern = "blk.{layer}.ffn_gate.weight", .slot = "ffn", .role = "gate", .any_part = true},
    TensorRule{
        .pattern = "blk.{layer}.ffn_up.weight", .slot = "ffn", .role = "up", .any_part = true},
    TensorRule{
        .pattern = "blk.{layer}.ffn_down.weight", .slot = "ffn", .role = "down", .any_part = true},
    TensorRule{.pattern = "output_norm.weight", .slot = "final_norm", .role = "weight"},
    TensorRule{.pattern = "output.weight",
               .slot = "output",
               .role = "weight",
               .fallback = "token_embd.weight",
               .any_part = true},
};
constexpr std::array kQwen35Params = {
    FromKey(kStageSlot, "context", "context_length", kInt),
    Int(kStageSlot, "context", 262144),
    FromKey(kStreamSlot, "width", "embedding_length", kInt),
    Name("linear", "placement", "pre"),
    Name("linear", "output_gate", "silu"),
    ZeroCenteredNorm("linear"),
    FromKey("linear", "eps", "attention.layer_norm_rms_epsilon", kReal),
    FromKey("linear", "k_heads", "ssm.group_count", kInt),
    FromKey("linear", "v_heads", "ssm.time_step_rank", kInt),
    FromKey("linear", "head_dim", "ssm.state_size", kInt),
    FromKey("linear", "conv", "ssm.conv_kernel", kInt),
    ParamRule{.slot = "linear",
              .param = "a_log",
              .type = kBool,
              .value = {.flag = true},
              .unless_fold = "ssm.a_neg_exp"},
    Name("attention", "placement", "pre"),
    ZeroCenteredNorm("attention"),
    FromKey("attention", "eps", "attention.layer_norm_rms_epsilon", kReal),
    FromKey("attention", "heads", "attention.head_count", kInt),
    FromKey("attention", "kv_heads", "attention.head_count_kv", kInt),
    FromOrRatio("attention", "key_dim", "attention.key_length", "embedding_length",
                "attention.head_count", kInt),
    FromKey("attention", "value_dim", "attention.value_length", kInt),
    FromOrRatio("attention", "value_dim", "attention.key_length", "embedding_length",
                "attention.head_count", kInt),
    FromOrRatio("attention", "scale", "attention.key_length", "embedding_length",
                "attention.head_count", kReal, Transform::kRsqrt),
    Int("attention", "window", 0),
    Name("attention", "rotation", "neox"),
    FromKey("attention", "rope_dims", "rope.dimension_count", kInt),
    FromProduct("attention", "rope_dims", "attention.key_length", "rope.partial_rotary_factor",
                kInt),
    FromOrRatio("attention", "rope_dims", "attention.key_length", "embedding_length",
                "attention.head_count", kInt),
    FromKey("attention", "rope_base", "rope.freq_base", kReal),
    Real("attention", "rope_base", 10000000.0),
    Real("attention", "rope_scale", 1.0),
    Flag("attention", "qk_norm", true),
    Flag("attention", "output_gate", true),
    Flag("attention", "scale_scores", true),
    Flag("attention", "explicit_scores", false),
    Name("moe", "placement", "pre"),
    ZeroCenteredNorm("moe"),
    FromKey("moe", "eps", "attention.layer_norm_rms_epsilon", kReal),
    Name("moe", "activation", "silu"),
    FromKey("moe", "hidden", "expert_shared_feed_forward_length", kInt, Transform::kNone, {}, true),
    FromKey("moe", "experts", "expert_count", kInt),
    FromKey("moe", "used", "expert_used_count", kInt),
    FromKey("moe", "expert_hidden", "expert_feed_forward_length", kInt),
    Name("moe", "router", "softmax_topk"),
    Name("moe", "router_input", "stream"),
    Flag("moe", "normalize", true),
    FromTensor("moe", "shared_gate", "blk.{layer}.ffn_gate_inp_shexp.weight"),
    Flag("moe", "shared_gate", false),
    FromTensor("moe", "stacked", "blk.{layer}.ffn_gate_up_exps.weight"),
    Flag("moe", "stacked", false),
    Flag("moe", "projection_scales", false),
    FromTensor("moe", "gate_scale", "blk.{layer}.ffn_gate_exps.weight_scale_2"),
    Flag("moe", "gate_scale", false),
    FromTensor("moe", "up_scale", "blk.{layer}.ffn_up_exps.weight_scale_2"),
    Flag("moe", "up_scale", false),
    FromTensor("moe", "down_scale", "blk.{layer}.ffn_down_exps.weight_scale_2"),
    Flag("moe", "down_scale", false),
    Name("ffn", "placement", "pre"),
    ZeroCenteredNorm("ffn"),
    FromKey("ffn", "eps", "attention.layer_norm_rms_epsilon", kReal),
    Name("ffn", "activation", "silu"),
    FromKey("ffn", "hidden", "feed_forward_length", kInt),
    ZeroCenteredNorm("final_norm"),
    FromKey("final_norm", "eps", "attention.layer_norm_rms_epsilon", kReal),
    Require("rope.scaling.type", "default", true),
    Require("hidden_act", "silu", true),
    Require("mamba_ssm_dtype", "float32", true),
    ParamRule{.key = "attention_bias", .type = kBool, .value = {.flag = false}, .optional = true},
    ParamRule{
        .key = "is_encoder_decoder", .type = kBool, .value = {.flag = false}, .optional = true},
    Require("output_gate_type", "swish", true),
    ParamRule{.key = "attn_output_gate", .type = kBool, .value = {.flag = true}, .optional = true},

};
constexpr std::array kQwen35KeyAliases = {
    KeyAlias{"text_config.output_gate_type", "output_gate_type"},
    KeyAlias{"text_config.attn_output_gate", "attn_output_gate"},
    KeyAlias{"text_config.linear_value_head_dim", "ssm.value_head_dim"},
    KeyAlias{"linear_value_head_dim", "ssm.value_head_dim"},
    KeyAlias{"text_config.rope_parameters.rope_type", "rope.scaling.type"},
    KeyAlias{"rope_parameters.rope_type", "rope.scaling.type"},
    KeyAlias{"text_config.rope_parameters.mrope_interleaved", "rope.mrope_interleaved"},
    KeyAlias{"rope_parameters.mrope_interleaved", "rope.mrope_interleaved"},
    KeyAlias{"text_config.rope_parameters.mrope_section", "rope.dimension_sections"},
    KeyAlias{"rope_parameters.mrope_section", "rope.dimension_sections"},
    KeyAlias{"text_config.hidden_act", "hidden_act"},
    KeyAlias{"text_config.mamba_ssm_dtype", "mamba_ssm_dtype"},
    KeyAlias{"text_config.attention_bias", "attention_bias"},
    KeyAlias{"text_config.is_encoder_decoder", "is_encoder_decoder"},
    KeyAlias{"text_config.num_hidden_layers", "block_count"},
    KeyAlias{"num_hidden_layers", "block_count"},
    KeyAlias{"text_config.max_position_embeddings", "context_length"},
    KeyAlias{"max_position_embeddings", "context_length"},
    KeyAlias{"text_config.hidden_size", "embedding_length"},
    KeyAlias{"hidden_size", "embedding_length"},
    KeyAlias{"text_config.intermediate_size", "feed_forward_length"},
    KeyAlias{"intermediate_size", "feed_forward_length"},
    KeyAlias{"text_config.rms_norm_eps", "attention.layer_norm_rms_epsilon"},
    KeyAlias{"rms_norm_eps", "attention.layer_norm_rms_epsilon"},
    KeyAlias{"text_config.num_attention_heads", "attention.head_count"},
    KeyAlias{"num_attention_heads", "attention.head_count"},
    KeyAlias{"text_config.num_key_value_heads", "attention.head_count_kv"},
    KeyAlias{"num_key_value_heads", "attention.head_count_kv"},
    KeyAlias{"text_config.head_dim", "attention.key_length"},
    KeyAlias{"head_dim", "attention.key_length"},
    KeyAlias{"text_config.rope_parameters.rope_theta", "rope.freq_base"},
    KeyAlias{"rope_parameters.rope_theta", "rope.freq_base"},
    KeyAlias{"text_config.rope_theta", "rope.freq_base"},
    KeyAlias{"rope_theta", "rope.freq_base"},
    KeyAlias{"text_config.rope_parameters.partial_rotary_factor", "rope.partial_rotary_factor"},
    KeyAlias{"rope_parameters.partial_rotary_factor", "rope.partial_rotary_factor"},
    KeyAlias{"text_config.partial_rotary_factor", "rope.partial_rotary_factor"},
    KeyAlias{"partial_rotary_factor", "rope.partial_rotary_factor"},
    KeyAlias{"text_config.linear_conv_kernel_dim", "ssm.conv_kernel"},
    KeyAlias{"linear_conv_kernel_dim", "ssm.conv_kernel"},
    KeyAlias{"text_config.linear_num_key_heads", "ssm.group_count"},
    KeyAlias{"linear_num_key_heads", "ssm.group_count"},
    KeyAlias{"text_config.linear_num_value_heads", "ssm.time_step_rank"},
    KeyAlias{"linear_num_value_heads", "ssm.time_step_rank"},
    KeyAlias{"text_config.linear_key_head_dim", "ssm.state_size"},
    KeyAlias{"linear_key_head_dim", "ssm.state_size"},
    KeyAlias{"text_config.num_experts", "expert_count"},
    KeyAlias{"num_experts", "expert_count"},
    KeyAlias{"text_config.num_experts_per_tok", "expert_used_count"},
    KeyAlias{"num_experts_per_tok", "expert_used_count"},
    KeyAlias{"text_config.moe_intermediate_size", "expert_feed_forward_length"},
    KeyAlias{"moe_intermediate_size", "expert_feed_forward_length"},
    KeyAlias{"text_config.shared_expert_intermediate_size", "expert_shared_feed_forward_length"},
    KeyAlias{"shared_expert_intermediate_size", "expert_shared_feed_forward_length"},
    KeyAlias{"text_config.layer_types", "layer_types"},
    KeyAlias{"layer_types", "layer_types"},
};
constexpr std::array kQwen35TensorAliases = {
    TensorAlias{"model.language_model.embed_tokens.weight", "token_embd.weight"},
    TensorAlias{"model.language_model.norm.weight", "output_norm.weight"},
    TensorAlias{"model.language_model.lm_head.weight", "output.weight"},
    TensorAlias{"model.language_model.layers.{layer}.input_layernorm.weight",
                "blk.{layer}.attn_norm.weight"},
    TensorAlias{"model.language_model.layers.{layer}.post_attention_layernorm.weight",
                "blk.{layer}.ffn_norm.weight"},
    TensorAlias{"model.language_model.layers.{layer}.linear_attn.in_proj_qkv.weight",
                "blk.{layer}.attn_qkv.weight"},
    TensorAlias{"model.language_model.layers.{layer}.linear_attn.in_proj_z.weight",
                "blk.{layer}.attn_gate.weight"},
    TensorAlias{"model.language_model.layers.{layer}.linear_attn.in_proj_b.weight",
                "blk.{layer}.ssm_beta.weight"},
    TensorAlias{"model.language_model.layers.{layer}.linear_attn.in_proj_a.weight",
                "blk.{layer}.ssm_alpha.weight"},
    TensorAlias{"model.language_model.layers.{layer}.linear_attn.dt_bias",
                "blk.{layer}.ssm_dt.bias"},
    TensorAlias{"model.language_model.layers.{layer}.linear_attn.A_log", "blk.{layer}.ssm_a"},
    TensorAlias{"model.language_model.layers.{layer}.linear_attn.conv1d.weight",
                "blk.{layer}.ssm_conv1d.weight"},
    TensorAlias{"model.language_model.layers.{layer}.linear_attn.norm.weight",
                "blk.{layer}.ssm_norm.weight"},
    TensorAlias{"model.language_model.layers.{layer}.linear_attn.out_proj.weight",
                "blk.{layer}.ssm_out.weight"},
    TensorAlias{"model.language_model.layers.{layer}.self_attn.q_proj.weight",
                "blk.{layer}.attn_q.weight"},
    TensorAlias{"model.language_model.layers.{layer}.self_attn.k_proj.weight",
                "blk.{layer}.attn_k.weight"},
    TensorAlias{"model.language_model.layers.{layer}.self_attn.v_proj.weight",
                "blk.{layer}.attn_v.weight"},
    TensorAlias{"model.language_model.layers.{layer}.self_attn.q_norm.weight",
                "blk.{layer}.attn_q_norm.weight"},
    TensorAlias{"model.language_model.layers.{layer}.self_attn.k_norm.weight",
                "blk.{layer}.attn_k_norm.weight"},
    TensorAlias{"model.language_model.layers.{layer}.self_attn.o_proj.weight",
                "blk.{layer}.attn_output.weight"},
    TensorAlias{"model.language_model.layers.{layer}.mlp.gate_proj.weight",
                "blk.{layer}.ffn_gate.weight"},
    TensorAlias{"model.language_model.layers.{layer}.mlp.up_proj.weight",
                "blk.{layer}.ffn_up.weight"},
    TensorAlias{"model.language_model.layers.{layer}.mlp.down_proj.weight",
                "blk.{layer}.ffn_down.weight"},
    TensorAlias{"model.language_model.layers.{layer}.mlp.gate.weight",
                "blk.{layer}.ffn_gate_inp.weight"},
    TensorAlias{"model.language_model.layers.{layer}.mlp.shared_expert_gate.weight",
                "blk.{layer}.ffn_gate_inp_shexp.weight"},
    TensorAlias{"model.language_model.layers.{layer}.mlp.shared_expert.gate_proj.weight",
                "blk.{layer}.ffn_gate_shexp.weight"},
    TensorAlias{"model.language_model.layers.{layer}.mlp.shared_expert.up_proj.weight",
                "blk.{layer}.ffn_up_shexp.weight"},
    TensorAlias{"model.language_model.layers.{layer}.mlp.shared_expert.down_proj.weight",
                "blk.{layer}.ffn_down_shexp.weight"},
    TensorAlias{"model.language_model.layers.{layer}.mlp.experts.gate_up_proj",
                "blk.{layer}.ffn_gate_up_exps.weight"},
    TensorAlias{"model.language_model.layers.{layer}.mlp.experts.gate_up_proj.weight",
                "blk.{layer}.ffn_gate_up_exps.weight"},
    TensorAlias{"model.language_model.layers.{layer}.mlp.experts.down_proj",
                "blk.{layer}.ffn_down_exps.weight"},
    TensorAlias{"model.language_model.layers.{layer}.mlp.experts.down_proj.weight",
                "blk.{layer}.ffn_down_exps.weight"},
    TensorAlias{"model.language_model.layers.{layer}.mlp.experts.{expert}.gate_proj.weight",
                "blk.{layer}.ffn_gate_exps.weight"},
    TensorAlias{"model.language_model.layers.{layer}.mlp.experts.{expert}.up_proj.weight",
                "blk.{layer}.ffn_up_exps.weight"},
    TensorAlias{"model.language_model.layers.{layer}.mlp.experts.{expert}.down_proj.weight",
                "blk.{layer}.ffn_down_exps.weight"},
    TensorAlias{"model.embed_tokens.weight", "token_embd.weight"},
    TensorAlias{"model.norm.weight", "output_norm.weight"},
    TensorAlias{"lm_head.weight", "output.weight"},
    TensorAlias{"model.layers.{layer}.input_layernorm.weight", "blk.{layer}.attn_norm.weight"},
    TensorAlias{"model.layers.{layer}.post_attention_layernorm.weight",
                "blk.{layer}.ffn_norm.weight"},
    TensorAlias{"model.layers.{layer}.linear_attn.in_proj_qkv.weight",
                "blk.{layer}.attn_qkv.weight"},
    TensorAlias{"model.layers.{layer}.linear_attn.in_proj_z.weight",
                "blk.{layer}.attn_gate.weight"},
    TensorAlias{"model.layers.{layer}.linear_attn.in_proj_b.weight", "blk.{layer}.ssm_beta.weight"},
    TensorAlias{"model.layers.{layer}.linear_attn.in_proj_a.weight",
                "blk.{layer}.ssm_alpha.weight"},
    TensorAlias{"model.layers.{layer}.linear_attn.dt_bias", "blk.{layer}.ssm_dt.bias"},
    TensorAlias{"model.layers.{layer}.linear_attn.A_log", "blk.{layer}.ssm_a"},
    TensorAlias{"model.layers.{layer}.linear_attn.conv1d.weight", "blk.{layer}.ssm_conv1d.weight"},
    TensorAlias{"model.layers.{layer}.linear_attn.norm.weight", "blk.{layer}.ssm_norm.weight"},
    TensorAlias{"model.layers.{layer}.linear_attn.out_proj.weight", "blk.{layer}.ssm_out.weight"},
    TensorAlias{"model.layers.{layer}.self_attn.q_proj.weight", "blk.{layer}.attn_q.weight"},
    TensorAlias{"model.layers.{layer}.self_attn.k_proj.weight", "blk.{layer}.attn_k.weight"},
    TensorAlias{"model.layers.{layer}.self_attn.v_proj.weight", "blk.{layer}.attn_v.weight"},
    TensorAlias{"model.layers.{layer}.self_attn.q_norm.weight", "blk.{layer}.attn_q_norm.weight"},
    TensorAlias{"model.layers.{layer}.self_attn.k_norm.weight", "blk.{layer}.attn_k_norm.weight"},
    TensorAlias{"model.layers.{layer}.self_attn.o_proj.weight", "blk.{layer}.attn_output.weight"},
    TensorAlias{"model.layers.{layer}.mlp.gate_proj.weight", "blk.{layer}.ffn_gate.weight"},
    TensorAlias{"model.layers.{layer}.mlp.up_proj.weight", "blk.{layer}.ffn_up.weight"},
    TensorAlias{"model.layers.{layer}.mlp.down_proj.weight", "blk.{layer}.ffn_down.weight"},
    TensorAlias{"model.layers.{layer}.mlp.gate.weight", "blk.{layer}.ffn_gate_inp.weight"},
    TensorAlias{"model.layers.{layer}.mlp.shared_expert_gate.weight",
                "blk.{layer}.ffn_gate_inp_shexp.weight"},
    TensorAlias{"model.layers.{layer}.mlp.shared_expert.gate_proj.weight",
                "blk.{layer}.ffn_gate_shexp.weight"},
    TensorAlias{"model.layers.{layer}.mlp.shared_expert.up_proj.weight",
                "blk.{layer}.ffn_up_shexp.weight"},
    TensorAlias{"model.layers.{layer}.mlp.shared_expert.down_proj.weight",
                "blk.{layer}.ffn_down_shexp.weight"},
    TensorAlias{"model.layers.{layer}.mlp.experts.gate_up_proj",
                "blk.{layer}.ffn_gate_up_exps.weight"},
    TensorAlias{"model.layers.{layer}.mlp.experts.gate_up_proj.weight",
                "blk.{layer}.ffn_gate_up_exps.weight"},
    TensorAlias{"model.layers.{layer}.mlp.experts.down_proj", "blk.{layer}.ffn_down_exps.weight"},
    TensorAlias{"model.layers.{layer}.mlp.experts.down_proj.weight",
                "blk.{layer}.ffn_down_exps.weight"},
    TensorAlias{"model.layers.{layer}.mlp.experts.{expert}.gate_proj.weight",
                "blk.{layer}.ffn_gate_exps.weight"},
    TensorAlias{"model.layers.{layer}.mlp.experts.{expert}.up_proj.weight",
                "blk.{layer}.ffn_up_exps.weight"},
    TensorAlias{"model.layers.{layer}.mlp.experts.{expert}.down_proj.weight",
                "blk.{layer}.ffn_down_exps.weight"},
    TensorAlias{"language_model.model.embed_tokens.weight", "token_embd.weight"},
    TensorAlias{"language_model.model.norm.weight", "output_norm.weight"},
    TensorAlias{"language_model.lm_head.weight", "output.weight"},
    TensorAlias{"language_model.model.layers.{layer}.input_layernorm.weight",
                "blk.{layer}.attn_norm.weight"},
    TensorAlias{"language_model.model.layers.{layer}.post_attention_layernorm.weight",
                "blk.{layer}.ffn_norm.weight"},
    TensorAlias{"language_model.model.layers.{layer}.linear_attn.in_proj_qkv.weight",
                "blk.{layer}.attn_qkv.weight"},
    TensorAlias{"language_model.model.layers.{layer}.linear_attn.in_proj_z.weight",
                "blk.{layer}.attn_gate.weight"},
    TensorAlias{"language_model.model.layers.{layer}.linear_attn.in_proj_b.weight",
                "blk.{layer}.ssm_beta.weight"},
    TensorAlias{"language_model.model.layers.{layer}.linear_attn.in_proj_a.weight",
                "blk.{layer}.ssm_alpha.weight"},
    TensorAlias{"language_model.model.layers.{layer}.linear_attn.dt_bias",
                "blk.{layer}.ssm_dt.bias"},
    TensorAlias{"language_model.model.layers.{layer}.linear_attn.A_log", "blk.{layer}.ssm_a"},
    TensorAlias{"language_model.model.layers.{layer}.linear_attn.conv1d.weight",
                "blk.{layer}.ssm_conv1d.weight"},
    TensorAlias{"language_model.model.layers.{layer}.linear_attn.norm.weight",
                "blk.{layer}.ssm_norm.weight"},
    TensorAlias{"language_model.model.layers.{layer}.linear_attn.out_proj.weight",
                "blk.{layer}.ssm_out.weight"},
    TensorAlias{"language_model.model.layers.{layer}.self_attn.q_proj.weight",
                "blk.{layer}.attn_q.weight"},
    TensorAlias{"language_model.model.layers.{layer}.self_attn.k_proj.weight",
                "blk.{layer}.attn_k.weight"},
    TensorAlias{"language_model.model.layers.{layer}.self_attn.v_proj.weight",
                "blk.{layer}.attn_v.weight"},
    TensorAlias{"language_model.model.layers.{layer}.self_attn.q_norm.weight",
                "blk.{layer}.attn_q_norm.weight"},
    TensorAlias{"language_model.model.layers.{layer}.self_attn.k_norm.weight",
                "blk.{layer}.attn_k_norm.weight"},
    TensorAlias{"language_model.model.layers.{layer}.self_attn.o_proj.weight",
                "blk.{layer}.attn_output.weight"},
    TensorAlias{"language_model.model.layers.{layer}.mlp.gate_proj.weight",
                "blk.{layer}.ffn_gate.weight"},
    TensorAlias{"language_model.model.layers.{layer}.mlp.up_proj.weight",
                "blk.{layer}.ffn_up.weight"},
    TensorAlias{"language_model.model.layers.{layer}.mlp.down_proj.weight",
                "blk.{layer}.ffn_down.weight"},
    TensorAlias{"language_model.model.layers.{layer}.mlp.gate.weight",
                "blk.{layer}.ffn_gate_inp.weight"},
    TensorAlias{"language_model.model.layers.{layer}.mlp.shared_expert_gate.weight",
                "blk.{layer}.ffn_gate_inp_shexp.weight"},
    TensorAlias{"language_model.model.layers.{layer}.mlp.shared_expert.gate_proj.weight",
                "blk.{layer}.ffn_gate_shexp.weight"},
    TensorAlias{"language_model.model.layers.{layer}.mlp.shared_expert.up_proj.weight",
                "blk.{layer}.ffn_up_shexp.weight"},
    TensorAlias{"language_model.model.layers.{layer}.mlp.shared_expert.down_proj.weight",
                "blk.{layer}.ffn_down_shexp.weight"},
    TensorAlias{"language_model.model.layers.{layer}.mlp.experts.gate_up_proj",
                "blk.{layer}.ffn_gate_up_exps.weight"},
    TensorAlias{"language_model.model.layers.{layer}.mlp.experts.gate_up_proj.weight",
                "blk.{layer}.ffn_gate_up_exps.weight"},
    TensorAlias{"language_model.model.layers.{layer}.mlp.experts.down_proj",
                "blk.{layer}.ffn_down_exps.weight"},
    TensorAlias{"language_model.model.layers.{layer}.mlp.experts.down_proj.weight",
                "blk.{layer}.ffn_down_exps.weight"},
    TensorAlias{"language_model.model.layers.{layer}.mlp.experts.{expert}.gate_proj.weight",
                "blk.{layer}.ffn_gate_exps.weight"},
    TensorAlias{"language_model.model.layers.{layer}.mlp.experts.{expert}.up_proj.weight",
                "blk.{layer}.ffn_up_exps.weight"},
    TensorAlias{"language_model.model.layers.{layer}.mlp.experts.{expert}.down_proj.weight",
                "blk.{layer}.ffn_down_exps.weight"},
};
constexpr std::array<std::string_view, 5> kQwen35IgnoredPrefixes{
    "model.vision_tower.", "model.visual.", "vision_tower.", "visual.", "multi_modal_projector."};
constexpr std::array kQwen35Requirements = {
    ScalarRequirement{.key = "rope.mrope_interleaved", .optional = true, .boolean = true},
    ScalarRequirement{
        .key = "ssm.value_head_dim", .equals_key = "ssm.state_size", .optional = true},
};
constexpr auto kQwen35Semantic = [] {
  std::array<std::string_view, kHfSemantic.size() + 4> out{};
  std::ranges::copy(kHfSemantic, out.begin());
  out[kHfSemantic.size()] = "linear_*";
  out[kHfSemantic.size() + 1] = "mamba_*";
  out[kHfSemantic.size() + 2] = "output_gate_type";
  out[kHfSemantic.size() + 3] = "indexer_*";
  return out;
}();
constexpr std::array kQwen35Lists = {
    ListRequirement{.key = "rope.dimension_sections",
                    .size = 4,
                    .min_size = 3,
                    .optional = true,
                    .sections_of = "attention.key_length",
                    .sections_multiplier_key = "rope.partial_rotary_factor"},
};
constexpr Mapping kQwen35Mapping{
    .architecture = "qwen35",
    .source_architectures = kQwen35SourceArchitectures,
    .version = 1,
    .pipeline = PipelineKind::kLlm,
    .source_key_prefix = "text_config.",
    .layer_count_key = "block_count",
    .context_key = "context_length",
    .layer_types = "lllf",
    .layer_type_key = "layer_types",
    .layer_type_codes = kQwen35LayerTypeCodes,
    .slots = kQwen35Slots,
    .tensors = kQwen35Tensors,
    .params = kQwen35Params,
    .folds = kQwen35Folds,
    .policies = kDensePolicies,
    .containers = kQwen35Containers,
    .tensor_aliases = kQwen35TensorAliases,
    .key_aliases = kQwen35KeyAliases,
    .auxiliary = kNativeAuxiliary,
    .requirements = kQwen35Requirements,
    .lists = kQwen35Lists,
    .extended_components = true,
    .semantic = kQwen35Semantic,
    .ignored_tensor_prefixes = kQwen35IgnoredPrefixes,
};
}  // namespace

const Mapping& Qwen2Mapping() { return kQwen2Mapping; }
const Mapping& Qwen2GgufMapping() { return kQwen2Mapping; }
const Mapping& Qwen2Exl3Mapping() { return kQwen2Mapping; }
const Mapping& Qwen2SafetensorsMapping() { return kQwen2Mapping; }
const Mapping& Qwen35Mapping() { return kQwen35Mapping; }

const Mapping& LlamaMapping() { return kLlamaMapping; }
const Mapping& LlamaGgufMapping() { return kLlamaMapping; }
const Mapping& LlamaExl3Mapping() { return kLlamaMapping; }
const Mapping& LlamaSafetensorsMapping() { return kLlamaMapping; }

const Mapping& MistralMapping() { return kMistralMapping; }
const Mapping& MistralGgufMapping() { return kMistralMapping; }
const Mapping& MistralExl3Mapping() { return kMistralMapping; }
const Mapping& MistralSafetensorsMapping() { return kMistralMapping; }
}  // namespace llmp::spec::mappings
