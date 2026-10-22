// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "spec/mappings/gemma2.h"

#include <algorithm>
#include <array>
#include <cmath>
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
    FromKey("attention", "softcap", "attn_logit_softcapping", kReal),
    FromKey("attention", "window", "attention.sliding_window", kInt, Transform::kNone, "l"),
    Int("attention", "window", 0, "g"),
    Name("attention", "rotation", "neox"),
    FromKey("attention", "rope_dims", "rope.dimension_count", kInt),
    FromKey("attention", "rope_dims", kKeyLength, kInt),
    // Local layers rotate at the checkpoint's sliding-window base when it
    // has one (llama.cpp reads it), else at the global base.
    FromKey("attention", "rope_base", "rope.freq_base_swa", kReal, Transform::kNone, "l"),
    FromKey("attention", "rope_base", "rope.freq_base", kReal),
    Real("attention", "rope_base", 10000.0),
    Real("attention", "rope_scale", 1.0),
    Flag("attention", "qk_norm", false),
    Name("ffn", "placement", "sandwich"),
    FromKey("ffn", "eps", kEps, kReal),
    Name("ffn", "activation", "gelu_tanh"),
    FromKey("ffn", "hidden", "feed_forward_length", kInt),
    FromKey("final_norm", "eps", kEps, kReal),
    FromKey(kHeadsSlot, "softcap", "final_logit_softcapping", kReal),
    // The components rotate without scaling.
    Require("rope.scaling.type", "none", true),
    // The layer types are "lg": a checkpoint's own pattern may only agree
    // (llama.cpp's load_swa_pattern reads it; an array form is refused).
    RequireInt("attention.sliding_window_pattern", 2, true),
};

constexpr std::array kFolds = {
    FoldRule{.fold = "norm.plus_one", .container = "gguf"},
    FoldRule{.fold = "norm.plus_one", .container = "safetensors", .imported = true},
    FoldRule{.fold = "norm.plus_one", .container = "mlx", .imported = true},
};

constexpr auto kNativeTensors = [] {
  auto out = kTensors;
  for (auto& r : out) {
    if (r.role != "norm" && r.role != "post_norm" &&
        !(r.slot == "final_norm" && r.role == "weight")) {
      r.any_part = true;
    }
  }
  return out;
}();

// The serving recipe (runtime/serving.cc's Gemma2): the fused norm-mul,
// norm-add and quantized GeGLU, shared Q8 preparation, owner attention
// (owner decode with bounded roots, owner and flexible owner prefill); no
// fused norm-rope (Gemma 2 has no QK norm), grouped KV stores or RoPE
// store. One rule: every checkpoint.
constexpr std::array kPolicies = {
    PolicyRule{.text = "max_decode_rows 16\n"},
};

// Its serving defaults; actual component capabilities bound owners. No drafter.
constexpr ServingRecipe kServing{
    .family = "Gemma", .name = "Gemma2", .bounded = true, .prefill_rows = 128, .drafters = false};

constexpr std::array kGemma2Requirements{ScalarRequirement{
    .key = "block_count", .reject_equal = 46, .unless_present = "attention.scale"}};

constexpr Mapping kMapping{
    .architecture = "gemma2",
    .version = 1,
    .pipeline = PipelineKind::kLlm,
    .layer_count_key = "block_count",
    .layer_types = "lg",
    .slots = kSlots,
    .tensors = kTensors,
    .params = kParams,
    .folds = kFolds,
    .policies = kPolicies,
    .serving = &kServing,
    .containers = kGgufContainers,
    .auxiliary = kGgufAuxiliary,
    .requirements = kGemma2Requirements,
};

// Native HF metadata is authoritative; unlike GGUF, its norm vectors are zero-centered.
constexpr auto kNativeParams = GemmaHfParams(kParams, 8192, "rope_theta");
constexpr Mapping kNativeMapping = [] {
  auto mapping = kMapping;
  mapping.architecture = "gemma2";
  mapping.unprefixed_keys = true;
  mapping.source_key_prefix = "text_config.";
  mapping.layer_count_key = "num_hidden_layers";
  mapping.tensors = kNativeTensors;
  mapping.params = kNativeParams;
  mapping.containers = kSafetensorsContainers;
  mapping.tensor_aliases = kGemmaHfTensors;
  mapping.key_aliases = kGemmaSafetensorsKeys;
  mapping.auxiliary = kGemmaHfAuxiliary;
  mapping.requirements = kGemma2HfRequirements;
  mapping.semantic = kHfSemantic;
  mapping.ignored_tensors = kHfIgnored;
  mapping.null_semantics = kGemmaHfNulls;
  mapping.tensor_presence = kHfHeadPresence;
  return mapping;
}();

}  // namespace

const Mapping& Gemma2Mapping() { return kMapping; }
const Mapping& Gemma2SafetensorsMapping() { return kNativeMapping; }

}  // namespace llmp::spec::mappings
