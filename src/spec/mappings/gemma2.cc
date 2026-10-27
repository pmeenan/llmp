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
#include "spec/mappings/names.h"
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
    TensorRule{.pattern = names::kTokenEmbd.gguf, .slot = "embedding", .role = "table"},
    TensorRule{.pattern = names::kAttnNorm.gguf, .slot = "attention", .role = "norm"},
    TensorRule{.pattern = names::kAttnQ.gguf, .slot = "attention", .role = "q"},
    TensorRule{.pattern = names::kAttnK.gguf, .slot = "attention", .role = "k"},
    TensorRule{.pattern = names::kAttnV.gguf, .slot = "attention", .role = "v"},
    TensorRule{.pattern = names::kAttnOutput.gguf, .slot = "attention", .role = "output"},
    TensorRule{.pattern = names::kAttnPostNorm.gguf, .slot = "attention", .role = "post_norm"},
    TensorRule{.pattern = names::kFfnPreNorm.gguf, .slot = "ffn", .role = "norm"},
    TensorRule{.pattern = names::kFfnGate.gguf, .slot = "ffn", .role = "gate"},
    TensorRule{.pattern = names::kFfnUp.gguf, .slot = "ffn", .role = "up"},
    TensorRule{.pattern = names::kFfnDown.gguf, .slot = "ffn", .role = "down"},
    TensorRule{.pattern = names::kFfnPostNorm.gguf, .slot = "ffn", .role = "post_norm"},
    TensorRule{.pattern = names::kOutputNorm.gguf, .slot = "final_norm", .role = "weight"},
    // The checkpoint ties its output to the embedding (no output.weight).
    TensorRule{.pattern = names::kOutput.gguf,
               .slot = "output",
               .role = "weight",
               .fallback = names::kTokenEmbd.gguf},
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

constexpr auto kHfKeys = [] {
  std::array<SourceKey, kGemmaHfKeys.size() + 1> out{};
  std::ranges::copy(kGemmaHfKeys, out.begin());
  out[kGemmaHfKeys.size()] = {"rope.freq_base_swa", "rope_theta"};
  return out;
}();
constexpr std::array kHfOverrides{kGemmaHfQueryScale};
constexpr std::array kHfDefaults{Int(kStageSlot, "context", 8192), Real("attention", "eps", 1e-6),
                                 Real("ffn", "eps", 1e-6), Real("final_norm", "eps", 1e-6)};
constexpr std::array kDialects{
    SourceDialect{.architecture = "gemma2",
                  .container = "gguf",
                  .folds = kGemmaGgufFolds,
                  .auxiliary = kGgufAuxiliary,
                  .requirements = kGemma2Requirements},
    SourceDialect{.architecture = "gemma2",
                  .container = "safetensors",
                  .unprefixed_keys = true,
                  .source_key_prefix = "text_config.",
                  .keys = kHfKeys,
                  .overrides = kHfOverrides,
                  .params = kGemmaHfParams,
                  .defaults = kHfDefaults,
                  .parts = kHfMatrixParts,
                  .folds = kGemmaHfFolds,
                  .tensor_aliases = kGemmaHfTensors,
                  .key_aliases = kGemmaSafetensorsKeys,
                  .auxiliary = kGemmaHfAuxiliary,
                  .requirements = kGemma2HfRequirements,
                  .semantic = kHfSemantic,
                  .ignored_tensors = kHfIgnored,
                  .null_semantics = kGemmaHfNulls,
                  .tensor_presence = kHfHeadPresence},
};

constexpr Mapping kMapping{
    .architecture = "gemma2",
    .version = 1,
    .pipeline = PipelineKind::kLlm,
    .layer_count_key = "block_count",
    .layer_types = "lg",
    .slots = kSlots,
    .tensors = kTensors,
    .params = kParams,
    .policies = kPolicies,
    .serving = &kServing,
    .containers = kGemmaContainers,
    .dialects = kDialects,
};

}  // namespace

const Mapping& Gemma2Mapping() { return kMapping; }

}  // namespace llmp::spec::mappings
