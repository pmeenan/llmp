// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "spec/mappings/gemma3.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <expected>
#include <string>
#include <vector>

#include "spec/mapping.h"
#include "spec/mappings/rules.h"
#include "spec/spec.h"

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

constexpr std::array kTensors = {
    TensorRule{.pattern = "token_embd.weight", .slot = "embedding", .role = "table"},
    TensorRule{.pattern = "blk.{layer}.attn_norm.weight", .slot = "attention", .role = "norm"},
    TensorRule{.pattern = "blk.{layer}.attn_q.weight", .slot = "attention", .role = "q"},
    TensorRule{.pattern = "blk.{layer}.attn_k.weight", .slot = "attention", .role = "k"},
    TensorRule{.pattern = "blk.{layer}.attn_v.weight", .slot = "attention", .role = "v"},
    TensorRule{.pattern = "blk.{layer}.attn_q_norm.weight", .slot = "attention", .role = "q_norm"},
    TensorRule{.pattern = "blk.{layer}.attn_k_norm.weight", .slot = "attention", .role = "k_norm"},
    TensorRule{.pattern = "blk.{layer}.attn_output.weight", .slot = "attention", .role = "output"},
    TensorRule{.pattern = "blk.{layer}.post_attention_norm.weight",
               .slot = "attention",
               .role = "post_norm"},
    TensorRule{.pattern = "blk.{layer}.ffn_norm.weight", .slot = "ffn", .role = "norm"},
    TensorRule{.pattern = "blk.{layer}.ffn_gate.weight", .slot = "ffn", .role = "gate"},
    TensorRule{.pattern = "blk.{layer}.ffn_up.weight", .slot = "ffn", .role = "up"},
    TensorRule{.pattern = "blk.{layer}.ffn_down.weight", .slot = "ffn", .role = "down"},
    TensorRule{.pattern = "blk.{layer}.post_ffw_norm.weight", .slot = "ffn", .role = "post_norm"},
    TensorRule{.pattern = "output_norm.weight", .slot = "final_norm", .role = "weight"},
    // The checkpoint ties its output to the embedding (no output.weight).
    TensorRule{.pattern = "output.weight",
               .slot = "output",
               .role = "weight",
               .fallback = "token_embd.weight"},
};

constexpr std::string_view kEps = "gemma3.attention.layer_norm_rms_epsilon";
constexpr std::string_view kKeyLength = "gemma3.attention.key_length";

constexpr std::array kParams = {
    FromKey(kStageSlot, "context", "gemma3.context_length", kInt),
    FromKey(kStreamSlot, "width", "gemma3.embedding_length", kInt),
    FromKey("embedding", "scale", "gemma3.embedding_length", kReal, Transform::kSqrt),
    Name("attention", "placement", "sandwich"),
    FromKey("attention", "eps", kEps, kReal),
    FromKey("attention", "heads", "gemma3.attention.head_count", kInt),
    FromKey("attention", "kv_heads", "gemma3.attention.head_count_kv", kInt),
    FromKey("attention", "key_dim", kKeyLength, kInt),
    FromKey("attention", "value_dim", "gemma3.attention.value_length", kInt),
    // Explicit exporter semantics override the legacy key-length default.
    FromKey("attention", "scale", "gemma3.attention.scale", kReal),
    FromKey("attention", "scale", kKeyLength, kReal, Transform::kRsqrt),
    Real("attention", "softcap", 0.0),
    FromKey("attention", "window", "gemma3.attention.sliding_window", kInt, Transform::kNone, "l"),
    Int("attention", "window", 0, "g"),
    Name("attention", "rotation", "neox"),
    FromKey("attention", "rope_dims", "gemma3.rope.dimension_count", kInt),
    FromKey("attention", "rope_dims", kKeyLength, kInt),
    // Local layers rotate at the checkpoint's sliding-window base, else
    // llama.cpp's built-in one, and unscaled; global ones at the checkpoint's base with its linear
    // factor (a scale of 1 / factor).
    FromKey("attention", "rope_base", "gemma3.rope.freq_base_swa", kReal, Transform::kNone, "l"),
    Real("attention", "rope_base", 10000.0, "l"),
    FromKey("attention", "rope_base", "gemma3.rope.freq_base", kReal, Transform::kNone, "g"),
    Real("attention", "rope_base", 10000.0, "g"),
    Real("attention", "rope_scale", 1.0, "l"),
    FromKey("attention", "rope_scale", "gemma3.rope.scaling.factor", kReal, Transform::kReciprocal,
            "g"),
    Real("attention", "rope_scale", 1.0, "g"),
    Flag("attention", "qk_norm", true),
    Name("ffn", "placement", "sandwich"),
    FromKey("ffn", "eps", kEps, kReal),
    Name("ffn", "activation", "gelu_tanh"),
    FromKey("ffn", "hidden", "gemma3.feed_forward_length", kInt),
    FromKey("final_norm", "eps", kEps, kReal),
    // A final logit softcap when the checkpoint has one (llama.cpp applies
    // it when not 0; a 0 here is refused, as a spec's softcap is above 0).
    FromKey(kHeadsSlot, "softcap", "gemma3.final_logit_softcapping", kReal, Transform::kNone, {},
            true),
    // The components scale rotations linearly, only.
    Require("gemma3.rope.scaling.type", "linear", true),
    // The layer types are "lllllg": a checkpoint's own pattern may only
    // agree (llama.cpp's load_swa_pattern reads it; an array form is refused).
    RequireInt("gemma3.attention.sliding_window_pattern", 6, true),
};

constexpr std::array kFolds = {FoldRule{.fold = "norm.plus_one", .container = "gguf"}};

// The serving recipe (runtime/serving.cc's Gemma3): every pattern but the
// RoPE store (the fused norm-mul, norm-rope and norm-add, the quantized
// GeGLU, shared Q8 preparation, grouped KV stores, owner attention with
// bounded roots and the four-root carriers). One rule: every checkpoint.
constexpr std::array kPolicies = {
    PolicyRule{.text = "max_decode_rows 16\ndisable rope_store\n"},
};

// Its bounded serving route: 4,096 tokens by default (a wider context is an
// explicit opt-in for exactly one owner), at most two owners, no drafter.
constexpr ServingRecipe kServing{.family = "Gemma",
                                 .name = "Gemma3",
                                 .bounded = true,
                                 .prefill_rows = 128,
                                 .default_context = 4096,
                                 .max_owners = 2,
                                 .wide_context = 4096,
                                 .drafters = false};

constexpr Mapping kMapping{
    .architecture = "gemma3",
    .version = 1,
    .pipeline = PipelineKind::kLlm,
    .layer_count_key = "gemma3.block_count",
    .layer_types = "lllllg",
    .slots = kSlots,
    .tensors = kTensors,
    .params = kParams,
    .folds = kFolds,
    .policies = kPolicies,
    .serving = &kServing,
    .containers = kGgufContainers,
};

// Native HF metadata is authoritative; unlike GGUF, its norm vectors are zero-centered.
constexpr auto kNativeParams = [] {
  std::array<ParamRule, kParams.size() + 11> params{};
  std::ranges::copy(kParams, params.begin());
  for (auto& p : params) {
    if (p.slot == "attention" && p.param == "rope_base" && p.key.empty() && p.layer_types == "g")
      p.value.real = 1000000.0;
    if (p.param == "scale" && p.slot == "attention") {
      p.key = "query_pre_attn_scalar";
      p.transform = Transform::kRsqrt;
    } else if (p.key == "gemma3.context_length")
      p.key = "max_position_embeddings";
    else if (p.key == "gemma3.embedding_length")
      p.key = "hidden_size";
    else if (p.key == "gemma3.attention.layer_norm_rms_epsilon")
      p.key = "rms_norm_eps";
    else if (p.key == "gemma3.attention.head_count")
      p.key = "num_attention_heads";
    else if (p.key == "gemma3.attention.head_count_kv")
      p.key = "num_key_value_heads";
    else if (p.key == "gemma3.attention.key_length" || p.key == "gemma3.attention.value_length" ||
             p.key == "gemma3.rope.dimension_count")
      p.key = "head_dim";
    else if (p.key == "gemma3.attn_logit_softcapping")
      p.key = "attn_logit_softcapping";
    else if (p.key == "gemma3.final_logit_softcapping")
      p.key = "final_logit_softcapping";
    else if (p.key == "gemma3.attention.sliding_window")
      p.key = "sliding_window";
    else if (p.key == "gemma3.feed_forward_length")
      p.key = "intermediate_size";
    else if (p.key == "gemma3.rope.freq_base")
      p.key = "rope_theta";
    else if (p.key == "gemma3.rope.freq_base_swa")
      p.key = "rope_local_base_freq";
    else if (p.key == "gemma3.rope.scaling.factor")
      p.key = "rope_scaling.factor";
    else if (p.key == "gemma3.rope.scaling.type")
      p.key = "rope_scaling.rope_type";
    else if (p.key == "gemma3.attention.sliding_window_pattern")
      p.key = "sliding_window_pattern";
  }
  std::size_t i = kParams.size();
  for (std::string_view slot : {"attention", "ffn", "final_norm"}) {
    params[i] = Flag(slot, "norm_zero_centered", true);
    params[i++].unless_fold = "norm.plus_one";
  }
  params[i++] = Require("hidden_activation", "gelu_pytorch_tanh", true);
  params[i++] =
      ParamRule{.key = "attention_bias", .type = kBool, .value = {.flag = false}, .optional = true};
  params[i++] = Require("hidden_act", "gelu_pytorch_tanh", true);
  params[i++] = ParamRule{.key = "use_bidirectional_attention",
                          .type = kBool,
                          .value = {.flag = false},
                          .optional = true};
  // Defaults of the native HF config schema, not checkpoint-size inference.
  params[i++] = Int(kStageSlot, "context", 131072);
  params[i++] = Real("attention", "eps", 1e-6);
  params[i++] = Real("ffn", "eps", 1e-6);
  params[i] = Real("final_norm", "eps", 1e-6);
  return params;
}();
constexpr Mapping kNativeMapping = [] {
  auto mapping = kMapping;
  mapping.architecture = "gemma3_text";
  mapping.layer_count_key = "num_hidden_layers";
  mapping.params = kNativeParams;
  mapping.containers = kSafetensorsContainers;
  return mapping;
}();

constexpr auto kConditionalParams = [] {
  std::array<ParamRule, kNativeParams.size() + 1> params{};
  std::ranges::copy(kNativeParams, params.begin());
  for (auto& p : params) {
    if (p.key == "query_pre_attn_scalar") p.key = "text_config.query_pre_attn_scalar";
    if (p.key == "max_position_embeddings") p.key = "text_config.max_position_embeddings";
    if (p.key == "hidden_size") p.key = "text_config.hidden_size";
    if (p.key == "rms_norm_eps") p.key = "text_config.rms_norm_eps";
    if (p.key == "num_attention_heads") p.key = "text_config.num_attention_heads";
    if (p.key == "num_key_value_heads") p.key = "text_config.num_key_value_heads";
    if (p.key == "head_dim") p.key = "text_config.head_dim";
    if (p.key == "final_logit_softcapping") p.key = "text_config.final_logit_softcapping";
    if (p.key == "sliding_window") p.key = "text_config.sliding_window";
    if (p.key == "intermediate_size") p.key = "text_config.intermediate_size";
    if (p.key == "rope_theta") p.key = "text_config.rope_theta";
    if (p.key == "rope_local_base_freq") p.key = "text_config.rope_local_base_freq";
    if (p.key == "rope_scaling.factor") p.key = "text_config.rope_scaling.factor";
    if (p.key == "rope_scaling.rope_type") p.key = "text_config.rope_scaling.rope_type";
    if (p.key == "sliding_window_pattern") p.key = "text_config.sliding_window_pattern";
    if (p.key == "hidden_activation") p.key = "text_config.hidden_activation";
    if (p.key == "hidden_act") p.key = "text_config.hidden_act";
    if (p.key == "attention_bias") p.key = "text_config.attention_bias";
    if (p.key == "use_bidirectional_attention") p.key = "text_config.use_bidirectional_attention";
  }
  params.back() = Require("text_config.model_type", "gemma3_text", false);
  return params;
}();
constexpr Mapping kConditionalMapping = [] {
  auto mapping = kNativeMapping;
  mapping.architecture = "gemma3";
  mapping.layer_count_key = "text_config.num_hidden_layers";
  mapping.params = kConditionalParams;
  return mapping;
}();

}  // namespace

const Mapping& Gemma3Mapping() { return kMapping; }
const Mapping& Gemma3SafetensorsMapping() { return kNativeMapping; }
const Mapping& Gemma3ConditionalSafetensorsMapping() { return kConditionalMapping; }

}  // namespace llmp::spec::mappings
