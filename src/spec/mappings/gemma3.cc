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

constexpr std::string_view kEps = "attention.layer_norm_rms_epsilon";
constexpr std::string_view kKeyLength = "attention.key_length";

constexpr std::array kParams = {
    FromKey(kStageSlot, "context", "context_length", kInt),
    FromKey(kStreamSlot, "width", "embedding_length", kInt),
    FromKey("embedding", "scale", "embedding_length", kReal, Transform::kSqrt),
    Name("attention", "placement", "sandwich"),
    FromKey("attention", "eps", kEps, kReal),
    FromKey("attention", "heads", "attention.head_count", kInt),
    FromKey("attention", "kv_heads", "attention.head_count_kv", kInt),
    FromKey("attention", "key_dim", kKeyLength, kInt),
    FromKey("attention", "value_dim", "attention.value_length", kInt),
    // Explicit exporter semantics override the legacy key-length default.
    FromKey("attention", "scale", "attention.scale", kReal),
    FromKey("attention", "scale", kKeyLength, kReal, Transform::kRsqrt),
    Real("attention", "softcap", 0.0),
    FromKey("attention", "window", "attention.sliding_window", kInt, Transform::kNone, "l"),
    Int("attention", "window", 0, "g"),
    Name("attention", "rotation", "neox"),
    FromKey("attention", "rope_dims", "rope.dimension_count", kInt),
    FromKey("attention", "rope_dims", kKeyLength, kInt),
    // Local layers rotate at the checkpoint's sliding-window base, else
    // llama.cpp's built-in one, and unscaled; global ones at the checkpoint's base with its linear
    // factor (a scale of 1 / factor).
    FromKey("attention", "rope_base", "rope.freq_base_swa", kReal, Transform::kNone, "l"),
    Real("attention", "rope_base", 10000.0, "l"),
    FromKey("attention", "rope_base", "rope.freq_base", kReal, Transform::kNone, "g"),
    Real("attention", "rope_base", 10000.0, "g"),
    Real("attention", "rope_scale", 1.0, "l"),
    FromKey("attention", "rope_scale", "rope.scaling.factor", kReal, Transform::kReciprocal, "g"),
    Real("attention", "rope_scale", 1.0, "g"),
    Flag("attention", "qk_norm", true),
    Name("ffn", "placement", "sandwich"),
    FromKey("ffn", "eps", kEps, kReal),
    Name("ffn", "activation", "gelu_tanh"),
    FromKey("ffn", "hidden", "feed_forward_length", kInt),
    FromKey("final_norm", "eps", kEps, kReal),
    // A final logit softcap when the checkpoint has one (llama.cpp applies
    // it when not 0; a 0 here is refused, as a spec's softcap is above 0).
    FromKey(kHeadsSlot, "softcap", "final_logit_softcapping", kReal, Transform::kNone, {}, true),
    // The components scale rotations linearly, only.
    Require("rope.scaling.type", "linear", true),
    // The layer types are "lllllg": a checkpoint's own pattern may only
    // agree (llama.cpp's load_swa_pattern reads it; an array form is refused).
    RequireInt("attention.sliding_window_pattern", 6, true),
};

constexpr std::array kFolds = {
    FoldRule{.fold = "norm.plus_one", .container = "gguf"},
    FoldRule{.fold = "norm.plus_one", .container = "safetensors", .imported = true},
    FoldRule{.fold = "norm.plus_one", .container = "mlx", .imported = true},
};

constexpr auto kNativeTensors = [] {
  auto out = kTensors;
  for (auto& r : out) {
    if (r.role != "norm" && r.role != "post_norm" && r.role != "q_norm" && r.role != "k_norm" &&
        !(r.slot == "final_norm" && r.role == "weight")) {
      r.any_part = true;
    }
  }
  return out;
}();

// The serving recipe (runtime/serving.cc's Gemma3): every pattern but the
// RoPE store (the fused norm-mul, norm-rope and norm-add, the quantized
// GeGLU, shared Q8 preparation, grouped KV stores, owner attention with
// bounded roots and the four-root carriers). One rule: every checkpoint.
constexpr std::array kPolicies = {
    PolicyRule{.text = "max_decode_rows 16\n"},
};

// Its bounded serving route: 4,096 tokens by default (a wider context is an
// explicit opt-in); actual component capabilities bound owners. No drafter.
constexpr ServingRecipe kServing{.family = "Gemma",
                                 .name = "Gemma3",
                                 .bounded = true,
                                 .prefill_rows = 128,
                                 .default_context = 4096,
                                 .drafters = false,
                                 .wide_context = 4096};

constexpr std::array kGemma3Requirements{ScalarRequirement{
    .key = "block_count", .reject_equal = 62, .unless_present = "attention.scale"}};

constexpr Mapping kMapping{
    .architecture = "gemma3",
    .version = 1,
    .pipeline = PipelineKind::kLlm,
    .layer_count_key = "block_count",
    .layer_types = "lllllg",
    .slots = kSlots,
    .tensors = kTensors,
    .params = kParams,
    .folds = kFolds,
    .policies = kPolicies,
    .serving = &kServing,
    .containers = kGgufContainers,
    .auxiliary = kGgufAuxiliary,
    .requirements = kGemma3Requirements,
};

// Native HF metadata is authoritative; unlike GGUF, its norm vectors are zero-centered.
constexpr auto kNativeParams = GemmaHfParams(kParams, 131072, "rope_local_base_freq", 1000000.0);
constexpr auto kGemma3ConditionalAuxiliary = [] {
  std::array<AuxiliaryNamespace, kGemmaHfAuxiliary.size() + 7> out{};
  std::ranges::copy(kGemmaHfAuxiliary, out.begin());
  // Root tokenizer/provenance fields are kept by the common declarations above.
  auto i = kGemmaHfAuxiliary.size();
  out[i++] = {"vision_config.", "vision component outside this text-only import"};
  out[i++] = {"text_config.initializer_range", "weight initializer scale"};
  out[i++] = {"text_config.attention_dropout", "training dropout configuration"};
  out[i++] = {"text_config.cache_implementation", "transformers cache implementation"};
  out[i++] = {"text_config.torch_dtype", "checkpoint nominal dtype"};
  out[i++] = {"text_config.use_cache", "transformers generation cache default"};
  out[i] = {"text_config.vocab_size", "vocabulary stated by embedding table",
            "language_model.model.embed_tokens.weight", 1};
  return out;
}();
constexpr std::array<std::string_view, 1> kGemma3ConditionalSemantic{"text_config."};
constexpr std::array<std::string_view, 5> kGemma3ConditionalNulls{
    "text_config.rope_scaling", "text_config.rope_parameters",
    "text_config.final_logit_softcapping", "text_config.attn_logit_softcapping",
    "text_config.use_bidirectional_attention"};
constexpr std::array<std::string_view, 2> kGemma3Excluded{"vision_tower.",
                                                          "multi_modal_projector."};
constexpr std::array kGemmaConditionalHeadPresence{
    TensorPresenceRequirement{.flag = "text_config.tie_word_embeddings", .tensor = "output.weight"},
    TensorPresenceRequirement{.flag = "tie_word_embeddings", .tensor = "output.weight"}};

constexpr Mapping kNativeMapping = [] {
  auto mapping = kMapping;
  mapping.architecture = "gemma3_text";
  mapping.unprefixed_keys = true;
  mapping.source_key_prefix = "text_config.";
  mapping.layer_count_key = "num_hidden_layers";
  mapping.tensors = kNativeTensors;
  mapping.params = kNativeParams;
  mapping.containers = kSafetensorsContainers;
  mapping.tensor_aliases = kGemma3HfTensors;
  mapping.key_aliases = kGemmaSafetensorsKeys;
  mapping.auxiliary = kGemmaHfAuxiliary;
  mapping.requirements = kGemmaHfRequirements;
  mapping.semantic = kHfSemantic;
  mapping.ignored_tensors = kHfIgnored;
  mapping.null_semantics = kGemmaHfNulls;
  mapping.tensor_presence = kHfHeadPresence;
  return mapping;
}();

constexpr Mapping kConditionalMapping = [] {
  auto mapping = kNativeMapping;
  mapping.architecture = "gemma3";
  mapping.key_prefix = "text_config";
  mapping.unprefixed_keys = false;
  mapping.tensor_prefix = "language_model.";
  mapping.layer_count_key = "num_hidden_layers";
  mapping.params = kNativeParams;
  mapping.tensor_aliases = kGemma3HfTensors;
  mapping.auxiliary = kGemma3ConditionalAuxiliary;
  mapping.requirements = kGemmaHfRequirements;
  mapping.semantic = kGemma3ConditionalSemantic;
  mapping.ignored_tensors = {};
  mapping.ignored_tensor_prefixes = kGemma3Excluded;
  mapping.null_semantics = kGemma3ConditionalNulls;
  mapping.tensor_presence = kGemmaConditionalHeadPresence;
  return mapping;
}();

}  // namespace

const Mapping& Gemma3Mapping() { return kMapping; }
const Mapping& Gemma3SafetensorsMapping() { return kNativeMapping; }
const Mapping& Gemma3ConditionalSafetensorsMapping() { return kConditionalMapping; }

}  // namespace llmp::spec::mappings
