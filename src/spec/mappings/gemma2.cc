// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "spec/mappings/gemma2.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <expected>
#include <string>
#include <vector>

#include "model/gemma2.h"
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

constexpr std::string_view kEps = "gemma2.attention.layer_norm_rms_epsilon";
constexpr std::string_view kKeyLength = "gemma2.attention.key_length";

constexpr std::array kParams = {
    FromKey(kStageSlot, "context", "gemma2.context_length", kInt),
    FromKey(kStreamSlot, "width", "gemma2.embedding_length", kInt),
    FromKey("embedding", "scale", "gemma2.embedding_length", kReal, Transform::kSqrt),
    Name("attention", "placement", "sandwich"),
    FromKey("attention", "eps", kEps, kReal),
    FromKey("attention", "heads", "gemma2.attention.head_count", kInt),
    FromKey("attention", "kv_heads", "gemma2.attention.head_count_kv", kInt),
    FromKey("attention", "key_dim", kKeyLength, kInt),
    FromKey("attention", "value_dim", "gemma2.attention.value_length", kInt),
    // llama.cpp's Gemma 2 below 27B: 1 / sqrt(the key length).
    FromKey("attention", "scale", kKeyLength, kReal, Transform::kRsqrt),
    FromKey("attention", "softcap", "gemma2.attn_logit_softcapping", kReal),
    FromKey("attention", "window", "gemma2.attention.sliding_window", kInt, Transform::kNone, "l"),
    Int("attention", "window", 0, "g"),
    Name("attention", "rotation", "neox"),
    FromKey("attention", "rope_dims", "gemma2.rope.dimension_count", kInt),
    FromKey("attention", "rope_dims", kKeyLength, kInt),
    // Local layers rotate at the checkpoint's sliding-window base when it
    // has one (llama.cpp reads it), else at the global base.
    FromKey("attention", "rope_base", "gemma2.rope.freq_base_swa", kReal, Transform::kNone, "l"),
    FromKey("attention", "rope_base", "gemma2.rope.freq_base", kReal),
    Real("attention", "rope_base", 10000.0),
    Real("attention", "rope_scale", 1.0),
    Flag("attention", "qk_norm", false),
    Name("ffn", "placement", "sandwich"),
    FromKey("ffn", "eps", kEps, kReal),
    Name("ffn", "activation", "gelu_tanh"),
    FromKey("ffn", "hidden", "gemma2.feed_forward_length", kInt),
    FromKey("final_norm", "eps", kEps, kReal),
    FromKey(kHeadsSlot, "softcap", "gemma2.final_logit_softcapping", kReal),
    // The components rotate without scaling.
    Require("gemma2.rope.scaling.type", "none", true),
    // The layer types are "lg": a checkpoint's own pattern may only agree
    // (llama.cpp's load_swa_pattern reads it; an array form is refused).
    RequireInt("gemma2.attention.sliding_window_pattern", 2, true),
};

constexpr std::array kFolds = {FoldRule{.fold = "norm.plus_one", .container = "gguf"}};

// The serving recipe (runtime/serving.cc's Gemma2): the fused norm-mul,
// norm-add and quantized GeGLU, shared Q8 preparation, owner attention
// (owner decode with bounded roots, owner and flexible owner prefill); no
// fused norm-rope (Gemma 2 has no QK norm), grouped KV stores or RoPE
// store. One rule: every checkpoint.
constexpr std::array kPolicies = {
    PolicyRule{.text = "max_decode_rows 16\ndisable grouped_stores\ndisable rms_norm_mul_rope\n"
                       "disable rope_store\n"},
};

// Its bounded serving route: at most two owners, no drafter.
constexpr ServingRecipe kServing{.family = "Gemma",
                                 .name = "Gemma2",
                                 .bounded = true,
                                 .prefill_rows = 128,
                                 .max_owners = 2,
                                 .drafters = false};

constexpr Mapping kMapping{
    .architecture = "gemma2",
    .version = 1,
    .pipeline = PipelineKind::kLlm,
    .layer_count_key = "gemma2.block_count",
    .layer_types = "lg",
    .slots = kSlots,
    .tensors = kTensors,
    .params = kParams,
    .folds = kFolds,
    .policies = kPolicies,
    .serving = &kServing,
};

}  // namespace

const Mapping& Gemma2Mapping() { return kMapping; }

std::expected<std::vector<SourceValue>, std::string> Gemma2V0Values(const model::Gemma2Profile& p) {
  if (p.key_dim == 0 ||
      p.attention_scale != static_cast<float>(1.0 / std::sqrt(static_cast<double>(p.key_dim))) ||
      p.rope_scale != 1.0F) {
    return std::unexpected(
        "the Gemma 2 profile's attention scale or rotation scaling is not the mapping's");
  }
  for (std::uint32_t il = 0; il < p.layers; ++il) {
    if (p.local(il) != (kMapping.layer_types.at(il % kMapping.layer_types.size()) == 'l')) {
      return std::unexpected("the Gemma 2 profile's local layers are not the mapping's");
    }
  }
  return std::vector<SourceValue>{
      Value("gemma2.context_length", p.context),
      Value("gemma2.embedding_length", p.width),
      Value("gemma2.block_count", p.layers),
      Value("gemma2.feed_forward_length", p.ffn),
      Value("gemma2.attention.head_count", p.heads),
      Value("gemma2.attention.head_count_kv", p.kv_heads),
      Value(kKeyLength, p.key_dim),
      Value("gemma2.attention.value_length", p.value_dim),
      Value(kEps, p.rms_eps),
      Value("gemma2.attn_logit_softcapping", p.attention_softcap),
      Value("gemma2.final_logit_softcapping", p.final_softcap),
      Value("gemma2.attention.sliding_window", p.window),
      Value("gemma2.rope.freq_base", p.rope_base),
  };
}

}  // namespace llmp::spec::mappings
