// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Shorthand for writing mapping tables (spec/mapping.h) as constant data,
// and for the source values the mapping data carries (a drafter import's
// target, the native MTP drafter's block). Internal to src/spec/mappings
// (and the tests' family fixtures).

#ifndef LLMP_SPEC_MAPPINGS_RULES_H_
#define LLMP_SPEC_MAPPINGS_RULES_H_

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "spec/mapping.h"
#include "spec/spec.h"

namespace llmp::spec::mappings {
inline constexpr std::uint32_t kLayerLimit = 1024;
inline constexpr std::array<std::string_view, 1> kGgufContainers{"gguf"};
inline constexpr std::array<std::string_view, 1> kSafetensorsContainers{"safetensors"};
inline constexpr std::array<std::string_view, 1> kExl3Containers{"exl3"};
inline constexpr std::array<std::string_view, 2> kNativeContainers{"safetensors", "mlx"};

inline constexpr std::array kGgufAuxiliary{
    AuxiliaryNamespace{"general.", "container provenance"},
    AuxiliaryNamespace{"tokenizer.", "tokenizer metadata retained for the writer"},
    AuxiliaryNamespace{"quantize.imatrix.", "source quantization calibration provenance"}};

inline constexpr std::array kSafetensorsAuxiliary{
    AuxiliaryNamespace{"architectures", "transformers class provenance"},
    AuxiliaryNamespace{"transformers_version", "transformers writer version provenance"},
    AuxiliaryNamespace{"attention_dropout", "training dropout configuration"},
    AuxiliaryNamespace{"bos_token_id", "tokenizer bos token id"},
    AuxiliaryNamespace{"eos_token_id", "tokenizer eos token id"},
    AuxiliaryNamespace{"initializer_range", "weight initializer scale"},
    AuxiliaryNamespace{"max_window_layers", "sliding window layer count"},
    AuxiliaryNamespace{"sliding_window", "sliding window length"},
    AuxiliaryNamespace{"torch_dtype", "checkpoint nominal dtype"},
    AuxiliaryNamespace{"dtype", "checkpoint nominal dtype"},
    AuxiliaryNamespace{"use_cache", "transformers generation cache default"},
    AuxiliaryNamespace{"vocab_size", "vocabulary size stated by embedding table",
                       "model.embed_tokens.weight", 1},
    AuxiliaryNamespace{"layer_types", "transformer block layer types"},
    AuxiliaryNamespace{"text_config.layer_types", "text transformer block layer types"},
};

// Gemma owns its mathematical keys rather than inheriting Qwen's auxiliary windows.
inline constexpr auto kGemmaHfAuxiliary = [] {
  std::array<AuxiliaryNamespace, kSafetensorsAuxiliary.size() - 2> out{};
  std::size_t next = 0;
  for (const auto& rule : kSafetensorsAuxiliary)
    if (rule.prefix != "sliding_window" && rule.prefix != "max_window_layers") out[next++] = rule;
  return out;
}();

inline constexpr std::array kNativeAuxiliary{
    AuxiliaryNamespace{"architectures", "transformers class provenance"},
    AuxiliaryNamespace{"transformers_version", "transformers writer version provenance"},
    AuxiliaryNamespace{"attention_dropout", "training dropout configuration"},
    AuxiliaryNamespace{"bos_token_id", "tokenizer bos token id"},
    AuxiliaryNamespace{"eos_token_id", "tokenizer eos token id"},
    AuxiliaryNamespace{"text_config.eos_token_id", "tokenizer eos token id"},
    AuxiliaryNamespace{"initializer_range", "weight initializer scale"},
    AuxiliaryNamespace{"torch_dtype", "checkpoint nominal dtype"},
    AuxiliaryNamespace{"dtype", "checkpoint nominal dtype"},
    AuxiliaryNamespace{"use_cache", "transformers generation cache default"},
    AuxiliaryNamespace{"tie_word_embeddings", "weight tying configuration"},
    AuxiliaryNamespace{"text_config.tie_word_embeddings", "weight tying configuration"},
    AuxiliaryNamespace{"vocab_size", "vocabulary size stated by embedding table"},
    AuxiliaryNamespace{"text_config.vocab_size", "vocabulary size stated by embedding table"},
    AuxiliaryNamespace{"vision_config.", "vision component outside this text-only import"},
    AuxiliaryNamespace{"audio_config.", "audio component outside this text-only import"},
    AuxiliaryNamespace{"quantization.", "checkpoint quantization metadata"},
    AuxiliaryNamespace{"quantization_config.", "checkpoint quantization metadata"},
};

inline constexpr std::array kGemmaHfRequirements{
    ScalarRequirement{.key = "query_pre_attn_scalar", .positive = true}};
inline constexpr std::array kGemma2HfRequirements{
    kGemmaHfRequirements[0],
    ScalarRequirement{
        .key = "sliding_window_size", .equals_key = "sliding_window", .optional = true}};

inline constexpr std::array kLlamaAuxiliary{
    kGgufAuxiliary[0], kGgufAuxiliary[1], kGgufAuxiliary[2],
    AuxiliaryNamespace{"llama.vocab_size", "the vocabulary the token table states",
                       "token_embd.weight", 1}};
inline constexpr std::array kMistralAuxiliary{
    kGgufAuxiliary[0], kGgufAuxiliary[1], kGgufAuxiliary[2],
    AuxiliaryNamespace{"mistral.vocab_size", "the vocabulary the token table states",
                       "token_embd.weight", 1}};

inline constexpr std::array kGemmaHfTensors{
    TensorAlias{"model.embed_tokens.weight", "token_embd.weight"},
    TensorAlias{"model.layers.{layer}.input_layernorm.weight", "blk.{layer}.attn_norm.weight"},
    TensorAlias{"model.layers.{layer}.self_attn.q_proj.weight", "blk.{layer}.attn_q.weight"},
    TensorAlias{"model.layers.{layer}.self_attn.k_proj.weight", "blk.{layer}.attn_k.weight"},
    TensorAlias{"model.layers.{layer}.self_attn.v_proj.weight", "blk.{layer}.attn_v.weight"},
    TensorAlias{"model.layers.{layer}.self_attn.o_proj.weight", "blk.{layer}.attn_output.weight"},
    TensorAlias{"model.layers.{layer}.post_attention_layernorm.weight",
                "blk.{layer}.post_attention_norm.weight"},
    TensorAlias{"model.layers.{layer}.pre_feedforward_layernorm.weight",
                "blk.{layer}.ffn_norm.weight"},
    TensorAlias{"model.layers.{layer}.post_feedforward_layernorm.weight",
                "blk.{layer}.post_ffw_norm.weight"},
    TensorAlias{"model.layers.{layer}.mlp.gate_proj.weight", "blk.{layer}.ffn_gate.weight"},
    TensorAlias{"model.layers.{layer}.mlp.up_proj.weight", "blk.{layer}.ffn_up.weight"},
    TensorAlias{"model.layers.{layer}.mlp.down_proj.weight", "blk.{layer}.ffn_down.weight"},
    TensorAlias{"model.norm.weight", "output_norm.weight"},
    TensorAlias{"lm_head.weight", "output.weight"},
};

inline constexpr auto kGemma3HfTensors = [] {
  std::array<TensorAlias, kGemmaHfTensors.size() + 2> aliases{};
  std::ranges::copy(kGemmaHfTensors, aliases.begin());
  aliases[kGemmaHfTensors.size()] = {"model.layers.{layer}.self_attn.q_norm.weight",
                                     "blk.{layer}.attn_q_norm.weight"};
  aliases[kGemmaHfTensors.size() + 1] = {"model.layers.{layer}.self_attn.k_norm.weight",
                                         "blk.{layer}.attn_k_norm.weight"};
  return aliases;
}();

inline constexpr std::array kDenseHfTensors{
    TensorAlias{"model.embed_tokens.weight", "token_embd.weight"},
    TensorAlias{"model.layers.{layer}.input_layernorm.weight", "blk.{layer}.attn_norm.weight"},
    TensorAlias{"model.layers.{layer}.self_attn.q_proj.weight", "blk.{layer}.attn_q.weight"},
    TensorAlias{"model.layers.{layer}.self_attn.k_proj.weight", "blk.{layer}.attn_k.weight"},
    TensorAlias{"model.layers.{layer}.self_attn.v_proj.weight", "blk.{layer}.attn_v.weight"},
    TensorAlias{"model.layers.{layer}.self_attn.q_proj.bias", "blk.{layer}.attn_q.bias"},
    TensorAlias{"model.layers.{layer}.self_attn.k_proj.bias", "blk.{layer}.attn_k.bias"},
    TensorAlias{"model.layers.{layer}.self_attn.v_proj.bias", "blk.{layer}.attn_v.bias"},
    TensorAlias{"model.layers.{layer}.self_attn.o_proj.weight", "blk.{layer}.attn_output.weight"},
    TensorAlias{"model.layers.{layer}.post_attention_layernorm.weight",
                "blk.{layer}.ffn_norm.weight"},
    TensorAlias{"model.layers.{layer}.mlp.gate_proj.weight", "blk.{layer}.ffn_gate.weight"},
    TensorAlias{"model.layers.{layer}.mlp.up_proj.weight", "blk.{layer}.ffn_up.weight"},
    TensorAlias{"model.layers.{layer}.mlp.down_proj.weight", "blk.{layer}.ffn_down.weight"},
    TensorAlias{"model.norm.weight", "output_norm.weight"},
    TensorAlias{"lm_head.weight", "output.weight"},
};

inline constexpr std::array kExl3Tensors{
    TensorAlias{"model.layers.{layer}.self_attn.q_proj.mul1",
                "model.layers.{layer}.self_attn.q_proj.mcg"},
    TensorAlias{"model.layers.{layer}.self_attn.k_proj.mul1",
                "model.layers.{layer}.self_attn.k_proj.mcg"},
    TensorAlias{"model.layers.{layer}.self_attn.v_proj.mul1",
                "model.layers.{layer}.self_attn.v_proj.mcg"},
    TensorAlias{"model.layers.{layer}.self_attn.o_proj.mul1",
                "model.layers.{layer}.self_attn.o_proj.mcg"},
    TensorAlias{"model.layers.{layer}.mlp.gate_proj.mul1",
                "model.layers.{layer}.mlp.gate_proj.mcg"},
    TensorAlias{"model.layers.{layer}.mlp.up_proj.mul1", "model.layers.{layer}.mlp.up_proj.mcg"},
    TensorAlias{"model.layers.{layer}.mlp.down_proj.mul1",
                "model.layers.{layer}.mlp.down_proj.mcg"},
    TensorAlias{"lm_head.mul1", "lm_head.mcg"},
};

inline constexpr std::array kQwen2SafetensorsKeys{
    KeyAlias{"head_dim", "attention.key_length"},
    KeyAlias{"hidden_act", "hidden_act"},
    KeyAlias{"hidden_size", "embedding_length"},
    KeyAlias{"intermediate_size", "feed_forward_length"},
    KeyAlias{"max_position_embeddings", "context_length"},
    KeyAlias{"num_attention_heads", "attention.head_count"},
    KeyAlias{"num_hidden_layers", "block_count"},
    KeyAlias{"num_key_value_heads", "attention.head_count_kv"},
    KeyAlias{"rms_norm_eps", "attention.layer_norm_rms_epsilon"},
    KeyAlias{"rope_theta", "rope.freq_base"},
    KeyAlias{"rope_parameters.rope_theta", "rope.freq_base"},
    KeyAlias{"use_sliding_window", "use_sliding_window"},
};

inline constexpr std::array kHfDenseKeys{
    KeyAlias{"head_dim", "attention.key_length"},
    KeyAlias{"hidden_act", "hidden_act"},
    KeyAlias{"hidden_size", "embedding_length"},
    KeyAlias{"intermediate_size", "feed_forward_length"},
    KeyAlias{"max_position_embeddings", "context_length"},
    KeyAlias{"num_attention_heads", "attention.head_count"},
    KeyAlias{"num_hidden_layers", "block_count"},
    KeyAlias{"num_key_value_heads", "attention.head_count_kv"},
    KeyAlias{"rms_norm_eps", "attention.layer_norm_rms_epsilon"},
    KeyAlias{"rope_theta", "rope.freq_base"},
    KeyAlias{"sliding_window", "attention.sliding_window"},
    KeyAlias{"attention_bias", "attention_bias"},
    KeyAlias{"rope_scaling.rope_type", "rope.scaling.type"},
    KeyAlias{"rope_scaling.type", "rope.scaling.type"},
    KeyAlias{"rope_scaling.factor", "rope.scaling.factor"},
    KeyAlias{"rope_scaling.original_max_position_embeddings",
             "rope.scaling.original_context_length"},
    KeyAlias{"rope_scaling.beta_fast", "rope.scaling.yarn_beta_fast"},
    KeyAlias{"rope_scaling.beta_slow", "rope.scaling.yarn_beta_slow"},
    KeyAlias{"rope_scaling.low_freq_factor", "rope.scaling.low_freq_factor"},
    KeyAlias{"rope_scaling.high_freq_factor", "rope.scaling.high_freq_factor"},
    KeyAlias{"rope_parameters.rope_theta", "rope.freq_base"},
    KeyAlias{"rope_parameters.rope_type", "rope.scaling.type"},
    KeyAlias{"rope_parameters.factor", "rope.scaling.factor"},
    KeyAlias{"rope_parameters.original_max_position_embeddings",
             "rope.scaling.original_context_length"},
    KeyAlias{"rope_parameters.low_freq_factor", "rope.scaling.low_freq_factor"},
    KeyAlias{"rope_parameters.high_freq_factor", "rope.scaling.high_freq_factor"},
};

inline constexpr std::array kGemmaSafetensorsKeys{
    KeyAlias{"rope_parameters.rope_theta", "rope_theta"},
    KeyAlias{"text_config.rope_parameters.rope_theta", "rope_theta"},
};

inline constexpr std::array kHfHeadPresence{
    TensorPresenceRequirement{.flag = "tie_word_embeddings", .tensor = "output.weight"}};
inline constexpr std::array kExl3HeadPresence{
    TensorPresenceRequirement{.flag = "tie_word_embeddings", .tensor = "lm_head.trellis"}};
inline constexpr std::array<std::string_view, 2> kHfIgnored{
    "model.rotary_emb.inv_freq", "model.layers.{layer}.self_attn.rotary_emb.inv_freq"};
inline constexpr std::array<std::string_view, 11> kDenseSafetensorsIgnored{
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
inline constexpr std::array<std::string_view, 2> kHfNullSemantics{"rope_scaling",
                                                                  "rope_parameters"};
inline constexpr std::array<std::string_view, 3> kDenseHfNulls{"rope_scaling", "rope_parameters",
                                                               "sliding_window"};
inline constexpr std::array<std::string_view, 5> kGemmaHfNulls{
    "rope_scaling", "rope_parameters", "final_logit_softcapping", "attn_logit_softcapping",
    "use_bidirectional_attention"};
inline constexpr std::array<std::string_view, 19> kHfSemantic{"qwen2.",
                                                              "rope_*",
                                                              "rotary_*",
                                                              "quantization*",
                                                              "sliding_window*",
                                                              "attention_*",
                                                              "attn_*",
                                                              "partial_rotary_factor",
                                                              "softcap*",
                                                              "hidden_*",
                                                              "rms_norm_*",
                                                              "norm_*",
                                                              "num_*",
                                                              "activation*",
                                                              "query_*",
                                                              "head_dim",
                                                              "use_bidirectional_attention",
                                                              "attn_logit_softcapping",
                                                              "final_logit_softcapping"};

// A parameter from a source key; `optional`: left out when the key is
// absent and no later rule gives it.
constexpr ParamRule FromKey(std::string_view slot, std::string_view param, std::string_view key,
                            ValueType type, Transform transform = Transform::kNone,
                            std::string_view layer_types = {}, bool optional = false) {
  return {.slot = slot,
          .param = param,
          .layer_types = layer_types,
          .key = key,
          .type = type,
          .transform = transform,
          .optional = optional};
}

// A parameter from a source key, or if absent, a ratio of two keys;
// `optional`: left out when both the key and the ratio are absent and no later rule gives it.
constexpr ParamRule FromOrRatio(std::string_view slot, std::string_view param, std::string_view key,
                                std::string_view numerator_key, std::string_view divisor_key,
                                ValueType type, Transform transform = Transform::kNone,
                                std::string_view layer_types = {}, bool optional = false) {
  return {.slot = slot,
          .param = param,
          .layer_types = layer_types,
          .key = key,
          .numerator_key = numerator_key,
          .divisor_key = divisor_key,
          .type = type,
          .transform = transform,
          .optional = optional};
}

// A parameter from the ratio of two keys; `optional`: left out when absent and no later rule gives
// it.
constexpr ParamRule FromRatio(std::string_view slot, std::string_view param,
                              std::string_view numerator_key, std::string_view divisor_key,
                              ValueType type, Transform transform = Transform::kNone,
                              std::string_view layer_types = {}, bool optional = false) {
  return {.slot = slot,
          .param = param,
          .layer_types = layer_types,
          .numerator_key = numerator_key,
          .divisor_key = divisor_key,
          .type = type,
          .transform = transform,
          .optional = optional};
}

// A parameter from the product of two keys.
constexpr ParamRule FromProduct(std::string_view slot, std::string_view param,
                                std::string_view numerator_key, std::string_view divisor_key,
                                ValueType type, std::string_view layer_types = {},
                                bool optional = false) {
  return {.slot = slot,
          .param = param,
          .layer_types = layer_types,
          .numerator_key = numerator_key,
          .divisor_key = divisor_key,
          .type = type,
          .transform = Transform::kProduct,
          .optional = optional};
}

constexpr ParamRule ZeroCenteredNorm(std::string_view slot) {
  return {.slot = slot,
          .param = "norm_zero_centered",
          .type = ValueType::kBool,
          .value = {.flag = true},
          .unless_fold = "norm.plus_one"};
}

// A parameter from a key listing a value a layer (or one for all).
constexpr ParamRule PerLayer(std::string_view slot, std::string_view param, std::string_view key,
                             ValueType type, std::string_view layer_types = {}) {
  return {.slot = slot,
          .param = param,
          .layer_types = layer_types,
          .key = key,
          .type = type,
          .per_layer = true};
}

// Constant parameters.
constexpr ParamRule Int(std::string_view slot, std::string_view param, std::int64_t value,
                        std::string_view layer_types = {}) {
  return {.slot = slot,
          .param = param,
          .layer_types = layer_types,
          .type = ValueType::kInt,
          .value = {.integer = value}};
}
constexpr ParamRule Real(std::string_view slot, std::string_view param, double value,
                         std::string_view layer_types = {}) {
  return {.slot = slot,
          .param = param,
          .layer_types = layer_types,
          .type = ValueType::kReal,
          .value = {.real = value}};
}
constexpr ParamRule Flag(std::string_view slot, std::string_view param, bool value,
                         std::string_view layer_types = {}) {
  return {.slot = slot,
          .param = param,
          .layer_types = layer_types,
          .type = ValueType::kBool,
          .value = {.flag = value}};
}
constexpr ParamRule Name(std::string_view slot, std::string_view param, std::string_view value,
                         std::string_view layer_types = {}) {
  return {.slot = slot,
          .param = param,
          .layer_types = layer_types,
          .type = ValueType::kString,
          .value = {.text = value}};
}

// A requirement: the key's value is `value`, or (optional) the key is
// absent.
constexpr ParamRule Require(std::string_view key, std::string_view value, bool optional) {
  return {.key = key, .type = ValueType::kString, .value = {.text = value}, .optional = optional};
}
constexpr ParamRule RequireInt(std::string_view key, std::int64_t value, bool optional) {
  return {.key = key, .type = ValueType::kInt, .value = {.integer = value}, .optional = optional};
}

// A boolean parameter set true when `tensor` is present in the source;
// absent when the tensor is absent, so a subsequent rule or optional omission
// applies.
constexpr ParamRule FromTensor(std::string_view slot, std::string_view param,
                               std::string_view tensor, std::string_view layer_types = {}) {
  return {.slot = slot,
          .param = param,
          .layer_types = layer_types,
          .tensor = tensor,
          .type = ValueType::kBool,
          .optional = true};
}

// An optional boolean parameter set true when `key` equals `value` (and left
// out when `key` is absent or `"none"`, or refused for any other string).
constexpr ParamRule KeyEquals(std::string_view slot, std::string_view param, std::string_view key,
                              std::string_view value, std::string_view layer_types = {}) {
  return {.slot = slot,
          .param = param,
          .layer_types = layer_types,
          .key = key,
          .type = ValueType::kBool,
          .value = {.flag = true, .text = value},
          .optional = true};
}

template <std::size_t N>
constexpr auto GemmaHfParams(const std::array<ParamRule, N>& gguf_params,
                             std::int64_t default_context, std::string_view swa_rope_key,
                             double global_rope_base = 0.0) {
  std::array<ParamRule, N + 12> params{};
  std::ranges::copy(gguf_params, params.begin());
  for (auto& p : params) {
    if (global_rope_base != 0.0 && p.slot == "attention" && p.param == "rope_base" &&
        p.key.empty() && p.layer_types == "g")
      p.value.real = global_rope_base;
    if (p.param == "scale" && p.slot == "attention") {
      p.key = "query_pre_attn_scalar";
      p.transform = Transform::kRsqrt;
    } else if (p.key == "context_length")
      p.key = "max_position_embeddings";
    else if (p.key == "embedding_length")
      p.key = "hidden_size";
    else if (p.key == "attention.layer_norm_rms_epsilon")
      p.key = "rms_norm_eps";
    else if (p.key == "attention.head_count")
      p.key = "num_attention_heads";
    else if (p.key == "attention.head_count_kv")
      p.key = "num_key_value_heads";
    else if (p.key == "attention.key_length" || p.key == "attention.value_length" ||
             p.key == "rope.dimension_count")
      p.key = "head_dim";
    else if (p.key == "attn_logit_softcapping")
      p.key = "attn_logit_softcapping";
    else if (p.key == "final_logit_softcapping")
      p.key = "final_logit_softcapping";
    else if (p.key == "attention.sliding_window")
      p.key = "sliding_window";
    else if (p.key == "feed_forward_length")
      p.key = "intermediate_size";
    else if (p.key == "rope.freq_base")
      p.key = "rope_theta";
    else if (p.key == "rope.freq_base_swa")
      p.key = swa_rope_key;
    else if (p.key == "rope.scaling.factor")
      p.key = "rope_scaling.factor";
    else if (p.key == "rope.scaling.type")
      p.key = "rope_scaling.rope_type";
    else if (p.key == "attention.sliding_window_pattern")
      p.key = "sliding_window_pattern";
  }
  std::size_t i = N;
  for (std::string_view slot : {"attention", "ffn", "final_norm"}) {
    params[i++] = ZeroCenteredNorm(slot);
  }
  params[i++] = Require("hidden_activation", "gelu_pytorch_tanh", true);
  params[i++] = ParamRule{.key = "attention_bias",
                          .type = ValueType::kBool,
                          .value = {.flag = false},
                          .optional = true};
  params[i++] = Require("hidden_act", "gelu_pytorch_tanh", true);
  params[i++] = ParamRule{.key = "use_bidirectional_attention",
                          .type = ValueType::kBool,
                          .value = {.flag = false},
                          .optional = true};
  params[i++] = Int(kStageSlot, "context", default_context);
  params[i++] = Real("attention", "eps", 1e-6);
  params[i++] = Real("ffn", "eps", 1e-6);
  params[i++] = Real("final_norm", "eps", 1e-6);
  params[i] = Require("text_config.model_type", "gemma3_text", true);
  return params;
}

// A profile's value under a source key: integers as integers, floats as the
// reals they are.
inline SourceValue Value(std::string_view key, std::uint32_t v) {
  return {.key = std::string(key), .value = ParamValue(static_cast<std::int64_t>(v))};
}
inline SourceValue Value(std::string_view key, float v) {
  return {.key = std::string(key), .value = ParamValue(static_cast<double>(v))};
}
inline SourceValue Value(std::string_view key, std::string_view v) {
  return {.key = std::string(key), .value = ParamValue(std::string(v))};
}
// A list of integers (GGUF's arrays of integers or flags).
inline SourceValue Value(std::string_view key, std::vector<std::int64_t> v) {
  return {.key = std::string(key), .value = ParamValue(std::move(v))};
}
// A flag (a source's fact, an approval's gate).
inline SourceValue FlagValue(std::string_view key, bool v) {
  return {.key = std::string(key), .value = ParamValue(v)};
}

const Mapping& Gemma4SafetensorsMapping();
const Mapping& Qwen35Mapping();

}  // namespace llmp::spec::mappings

#endif  // LLMP_SPEC_MAPPINGS_RULES_H_
