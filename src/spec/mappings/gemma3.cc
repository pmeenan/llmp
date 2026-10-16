// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "spec/mappings/gemma3.h"

#include <array>
#include <cstdint>
#include <expected>
#include <string>
#include <vector>

#include "model/gemma3.h"
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
    // Queries scaled by 1 / sqrt(the key length) (llama.cpp's Gemma 3).
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
};

}  // namespace

const Mapping& Gemma3Mapping() { return kMapping; }

std::expected<std::vector<SourceValue>, std::string> Gemma3V0Values(const model::Gemma3Profile& p) {
  if (p.rope_scaling != "linear") {
    return std::unexpected("the Gemma 3 profile's rotation scaling is not the mapping's");
  }
  for (std::uint32_t il = 0; il < p.layers; ++il) {
    if (p.local(il) != (kMapping.layer_types.at(il % kMapping.layer_types.size()) == 'l')) {
      return std::unexpected("the Gemma 3 profile's local layers are not the mapping's");
    }
  }
  return std::vector<SourceValue>{
      Value("gemma3.context_length", p.context),
      Value("gemma3.embedding_length", p.width),
      Value("gemma3.block_count", p.layers),
      Value("gemma3.feed_forward_length", p.ffn),
      Value("gemma3.attention.head_count", p.heads),
      Value("gemma3.attention.head_count_kv", p.kv_heads),
      Value(kKeyLength, p.key_dim),
      Value("gemma3.attention.value_length", p.value_dim),
      Value(kEps, p.rms_eps),
      Value("gemma3.attention.sliding_window", p.window),
      Value("gemma3.rope.freq_base", p.rope_base),
      Value("gemma3.rope.scaling.type", p.rope_scaling),
      Value("gemma3.rope.scaling.factor", p.rope_scale),
  };
}

}  // namespace llmp::spec::mappings
