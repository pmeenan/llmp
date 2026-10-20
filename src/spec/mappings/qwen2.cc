// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "spec/mappings/qwen2.h"

#include <array>
#include <cstdint>
#include <expected>
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

constexpr std::array kExl3Tensors = {
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

// Shared pre-norm dense decoder parameter skeleton parameterized by key prefix
// (`"qwen2."`, `"llama."`, `"mistral."`), RMS-norm epsilon transform, and
// container RoPE rotation convention (`"normal"` for Llama/Mistral GGUF,
// `"neox"` for Qwen2 and HuggingFace/EXL3 checkpoints).
#define LLMP_DENSE_DECODER_CORE_PARAMS(P, EPS_TRANSFORM, ROTATION)                                \
  FromKey(kStageSlot, "context", P "context_length", kInt),                                       \
      FromKey(kStreamSlot, "width", P "embedding_length", kInt),                                  \
      Name("attention", "placement", "pre"),                                                      \
      FromKey("attention", "eps", P "attention.layer_norm_rms_epsilon", kReal, EPS_TRANSFORM),    \
      FromKey("attention", "heads", P "attention.head_count", kInt),                              \
      FromKey("attention", "kv_heads", P "attention.head_count_kv", kInt),                        \
      FromOrRatio("attention", "key_dim", P "attention.key_length", P "embedding_length",         \
                  P "attention.head_count", kInt),                                                \
      FromKey("attention", "value_dim", P "attention.value_length", kInt),                        \
      FromOrRatio("attention", "value_dim", P "attention.key_length", P "embedding_length",       \
                  P "attention.head_count", kInt),                                                \
      FromOrRatio("attention", "scale", P "attention.key_length", P "embedding_length",           \
                  P "attention.head_count", kReal, Transform::kRsqrt),                            \
      FromKey("attention", "window", P "attention.sliding_window", kInt),                         \
      Int("attention", "window", 0), Name("attention", "rotation", ROTATION),                     \
      FromKey("attention", "rope_dims", P "rope.dimension_count", kInt),                          \
      FromOrRatio("attention", "rope_dims", P "attention.key_length", P "embedding_length",       \
                  P "attention.head_count", kInt),                                                \
      FromKey("attention", "rope_base", P "rope.freq_base", kReal),                               \
      Real("attention", "rope_base", 10000.0), Real("attention", "rope_scale", 1.0),              \
      FromTensor("attention", "rope_freqs", "rope_freqs.weight"),                                 \
      Flag("attention", "scale_scores", true), Flag("attention", "explicit_scores", false),       \
      Name("ffn", "placement", "pre"),                                                            \
      FromKey("ffn", "eps", P "attention.layer_norm_rms_epsilon", kReal, EPS_TRANSFORM),          \
      Name("ffn", "activation", "silu"), FromKey("ffn", "hidden", P "feed_forward_length", kInt), \
      FromKey("final_norm", "eps", P "attention.layer_norm_rms_epsilon", kReal, EPS_TRANSFORM)

#define LLMP_DENSE_DECODER_YARN_PARAMS(P)                                                         \
  KeyEquals("attention", "yarn", P "rope.scaling.type", "yarn"),                                  \
      FromKey("attention", "rope_factor", P "rope.scaling.factor", kReal, Transform::kNone, {},   \
              true),                                                                              \
      FromKey("attention", "yarn_context", P "rope.scaling.original_context_length", kInt,        \
              Transform::kNone, {}, true),                                                        \
      FromKey("attention", "beta_fast", P "rope.scaling.yarn_beta_fast", kReal, Transform::kNone, \
              {}, true),                                                                          \
      FromKey("attention", "beta_slow", P "rope.scaling.yarn_beta_slow", kReal, Transform::kNone, \
              {}, true)

#define LLMP_DENSE_DECODER_OPTIONAL_QKV_BIAS(P)                                    \
  FromKey("attention", "qkv_bias", P "attention_bias", kBool),                     \
      FromTensor("attention", "qkv_bias", "blk.0.attn_q.bias"),                    \
      FromTensor("attention", "qkv_bias", "model.layers.0.self_attn.q_proj.bias"), \
      Flag("attention", "qkv_bias", false)

constexpr std::array kQwen2Params = {
    LLMP_DENSE_DECODER_CORE_PARAMS("qwen2.", Transform::kNone, "neox"),
    Flag("attention", "qkv_bias", true),
};

// Qwen2's single-slot policy preserves the frozen v1 digests of the approved
// Qwen2.5-0.5B GGUF and EXL3 fixtures (no capture/row-invariance claim is made
// for EXL3).
constexpr std::array kQwen2Policies = {
    PolicyRule{.text = "max_decode_rows 1\n", .owners_alone = true, .capture_ahead = false},
};
constexpr Mapping kQwen2Mapping{
    .architecture = "qwen2",
    .version = 1,
    .pipeline = PipelineKind::kLlm,
    .layer_count_key = "qwen2.block_count",
    .layer_types = "g",
    .slots = kSlots,
    .tensors = kGgufTensors,
    .params = kQwen2Params,
    .policies = kQwen2Policies,
};

// The GGUF omits qwen2.attention.key_length when it equals width / heads
// (896 / 14 = 64 in the approved Qwen2.5-0.5B-Instruct FP16 fixture).
constexpr std::array kQwen2GgufParams = {
    LLMP_DENSE_DECODER_CORE_PARAMS("qwen2.", Transform::kNone, "neox"),
    Flag("attention", "qkv_bias", true),
    Require("qwen2.rope.scaling.type", "none", true),
};

constexpr Mapping kQwen2GgufMapping{
    .architecture = "qwen2",
    .version = 1,
    .pipeline = PipelineKind::kLlm,
    .layer_count_key = "qwen2.block_count",
    .layer_types = "g",
    .slots = kSlots,
    .tensors = kGgufTensors,
    .params = kQwen2GgufParams,
    .policies = kQwen2Policies,
    .containers = kGgufContainers,
};

// The EXL3 config.json omits head_dim when it equals hidden_size /
// num_attention_heads (896 / 14 = 64 in the approved 0.5B EXL3 fixtures).
// Emit the approved float-rounded rms_eps in component parameters while
// checking config.json's 1e-6 and all remaining profile keys.
constexpr std::array kQwen2Exl3Params = {
    LLMP_DENSE_DECODER_CORE_PARAMS("qwen2.", Transform::kFloat32, "neox"),
    Flag("attention", "qkv_bias", true),
    Require("qwen2.hidden_act", "silu", false),
    ParamRule{.key = "qwen2.use_sliding_window", .type = kBool, .value = {.flag = false}},
};

constexpr Mapping kQwen2Exl3Mapping{
    .architecture = "qwen2",
    .version = 1,
    .pipeline = PipelineKind::kLlm,
    .layer_count_key = "qwen2.block_count",
    .layer_types = "g",
    .slots = kSlots,
    .tensors = kExl3Tensors,
    .params = kQwen2Exl3Params,
    .policies = kQwen2Policies,
    .containers = kExl3Containers,
};
constexpr std::array kQwen2SafetensorsParams = {
    LLMP_DENSE_DECODER_CORE_PARAMS("qwen2.", Transform::kNone, "neox"),
    Flag("attention", "qkv_bias", true),
    Require("qwen2.hidden_act", "silu", false),
    ParamRule{.key = "qwen2.use_sliding_window", .type = kBool, .value = {.flag = false}},
};

constexpr Mapping kQwen2SafetensorsMapping{
    .architecture = "qwen2",
    .version = 1,
    .pipeline = PipelineKind::kLlm,
    .layer_count_key = "qwen2.block_count",
    .layer_types = "g",
    .slots = kSlots,
    .tensors = kGgufTensors,
    .params = kQwen2SafetensorsParams,
    .policies = kQwen2Policies,
    .containers = kSafetensorsContainers,
};

// Generic dense decoder policy: batched decode up to 16 rows with the shared
// runner's default lookahead and graph capture, paired MMQ dense products, and
// solo prompt chunks.
constexpr std::array kDensePolicies = {
    PolicyRule{.text = "max_decode_rows 16\nvector_rows 16\nproduct dense_pair\n",
               .prompts_alone = true},
};

constexpr auto kDenseGgufTensors = [] {
  auto out = kGgufTensors;
  for (auto& t : out)
    if (t.role == "q_bias" || t.role == "k_bias" || t.role == "v_bias") t.optional = true;
  return out;
}();

constexpr auto kDenseExl3Tensors = [] {
  auto out = kExl3Tensors;
  for (auto& t : out)
    if (t.role == "q_bias" || t.role == "k_bias" || t.role == "v_bias") t.optional = true;
  return out;
}();

#define LLMP_DEFINE_DENSE_FAMILY_MAPPINGS(NAME, ARCH)                        \
  constexpr std::array k##NAME##GgufParams = {                               \
      LLMP_DENSE_DECODER_CORE_PARAMS(ARCH ".", Transform::kNone, "normal"),  \
      LLMP_DENSE_DECODER_YARN_PARAMS(ARCH "."),                              \
      LLMP_DENSE_DECODER_OPTIONAL_QKV_BIAS(ARCH "."),                        \
  };                                                                         \
  constexpr std::array k##NAME##SafetensorsParams = {                        \
      LLMP_DENSE_DECODER_CORE_PARAMS(ARCH ".", Transform::kNone, "neox"),    \
      LLMP_DENSE_DECODER_YARN_PARAMS(ARCH "."),                              \
      LLMP_DENSE_DECODER_OPTIONAL_QKV_BIAS(ARCH "."),                        \
      Require(ARCH ".hidden_act", "silu", true),                             \
  };                                                                         \
  constexpr std::array k##NAME##Exl3Params = {                               \
      LLMP_DENSE_DECODER_CORE_PARAMS(ARCH ".", Transform::kFloat32, "neox"), \
      LLMP_DENSE_DECODER_YARN_PARAMS(ARCH "."),                              \
      LLMP_DENSE_DECODER_OPTIONAL_QKV_BIAS(ARCH "."),                        \
      Require(ARCH ".hidden_act", "silu", true),                             \
  };                                                                         \
  constexpr Mapping k##NAME##Mapping{                                        \
      .architecture = ARCH,                                                  \
      .version = 1,                                                          \
      .pipeline = PipelineKind::kLlm,                                        \
      .layer_count_key = ARCH ".block_count",                                \
      .layer_types = "g",                                                    \
      .slots = kSlots,                                                       \
      .tensors = kDenseGgufTensors,                                          \
      .params = k##NAME##GgufParams,                                         \
      .policies = kDensePolicies,                                            \
  };                                                                         \
  constexpr Mapping k##NAME##GgufMapping{                                    \
      .architecture = ARCH,                                                  \
      .version = 1,                                                          \
      .pipeline = PipelineKind::kLlm,                                        \
      .layer_count_key = ARCH ".block_count",                                \
      .layer_types = "g",                                                    \
      .slots = kSlots,                                                       \
      .tensors = kDenseGgufTensors,                                          \
      .params = k##NAME##GgufParams,                                         \
      .policies = kDensePolicies,                                            \
      .containers = kGgufContainers,                                         \
  };                                                                         \
  constexpr Mapping k##NAME##SafetensorsMapping{                             \
      .architecture = ARCH,                                                  \
      .version = 1,                                                          \
      .pipeline = PipelineKind::kLlm,                                        \
      .layer_count_key = ARCH ".block_count",                                \
      .layer_types = "g",                                                    \
      .slots = kSlots,                                                       \
      .tensors = kDenseGgufTensors,                                          \
      .params = k##NAME##SafetensorsParams,                                  \
      .policies = kDensePolicies,                                            \
      .containers = kSafetensorsContainers,                                  \
  };                                                                         \
  constexpr Mapping k##NAME##Exl3Mapping{                                    \
      .architecture = ARCH,                                                  \
      .version = 1,                                                          \
      .pipeline = PipelineKind::kLlm,                                        \
      .layer_count_key = ARCH ".block_count",                                \
      .layer_types = "g",                                                    \
      .slots = kSlots,                                                       \
      .tensors = kDenseExl3Tensors,                                          \
      .params = k##NAME##Exl3Params,                                         \
      .policies = kDensePolicies,                                            \
      .containers = kExl3Containers,                                         \
  };

LLMP_DEFINE_DENSE_FAMILY_MAPPINGS(Llama, "llama")
LLMP_DEFINE_DENSE_FAMILY_MAPPINGS(Mistral, "mistral")

#undef LLMP_DEFINE_DENSE_FAMILY_MAPPINGS
#undef LLMP_DENSE_DECODER_OPTIONAL_QKV_BIAS
#undef LLMP_DENSE_DECODER_YARN_PARAMS
#undef LLMP_DENSE_DECODER_CORE_PARAMS
}  // namespace

const Mapping& Qwen2Mapping() { return kQwen2Mapping; }
const Mapping& Qwen2GgufMapping() { return kQwen2GgufMapping; }
const Mapping& Qwen2Exl3Mapping() { return kQwen2Exl3Mapping; }
const Mapping& Qwen2SafetensorsMapping() { return kQwen2SafetensorsMapping; }

const Mapping& LlamaMapping() { return kLlamaMapping; }
const Mapping& LlamaGgufMapping() { return kLlamaGgufMapping; }
const Mapping& LlamaExl3Mapping() { return kLlamaExl3Mapping; }
const Mapping& LlamaSafetensorsMapping() { return kLlamaSafetensorsMapping; }

const Mapping& MistralMapping() { return kMistralMapping; }
const Mapping& MistralGgufMapping() { return kMistralGgufMapping; }
const Mapping& MistralExl3Mapping() { return kMistralExl3Mapping; }
const Mapping& MistralSafetensorsMapping() { return kMistralSafetensorsMapping; }
}  // namespace llmp::spec::mappings
