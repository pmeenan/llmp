// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "spec/mappings/gemma4.h"

#include <array>
#include <cstdint>
#include <expected>
#include <format>
#include <string>
#include <string_view>
#include <vector>

#include "format/encoding.h"
#include "format/tensor_format.h"
#include "spec/mapping.h"
#include "spec/mappings/rules.h"
#include "spec/spec.h"

namespace llmp::spec::mappings {
namespace {

using enum ValueType;

// ------------------------------------------------------------------ gemma4

constexpr std::string_view kExperts = "gemma4.expert_count";

constexpr std::array kSlots = {
    SlotRule{.slot = "embedding", .site = Site::kEmbedding, .kind = "embedding.token"},
    SlotRule{.slot = "attention", .site = Site::kLayer, .kind = "attention.dense"},
    // A checkpoint with experts has routed-expert layers, one without dense
    // ones.
    SlotRule{.slot = "moe", .site = Site::kLayer, .kind = "ffn.moe", .gate = kExperts},
    SlotRule{.slot = "ffn",
             .site = Site::kLayer,
             .kind = "ffn.glu",
             .gate = kExperts,
             .gate_absent = true},
    SlotRule{.slot = "output_scale", .site = Site::kLayer, .kind = "scale.learned"},
    SlotRule{.slot = "final_norm", .site = Site::kFinalNorm, .kind = "norm.rms"},
    SlotRule{.slot = "output", .site = Site::kOutput, .kind = "head.linear"},
};

constexpr std::array kTensors = {
    TensorRule{.pattern = "token_embd.weight", .slot = "embedding", .role = "table"},
    TensorRule{.pattern = "blk.{layer}.attn_norm.weight", .slot = "attention", .role = "norm"},
    TensorRule{.pattern = "blk.{layer}.attn_q.weight", .slot = "attention", .role = "q"},
    TensorRule{.pattern = "blk.{layer}.attn_k.weight", .slot = "attention", .role = "k"},
    // The global layers have none: their V is K's.
    TensorRule{.pattern = "blk.{layer}.attn_v.weight",
               .slot = "attention",
               .role = "v",
               .layer_types = "l"},
    TensorRule{.pattern = "blk.{layer}.attn_q_norm.weight", .slot = "attention", .role = "q_norm"},
    TensorRule{.pattern = "blk.{layer}.attn_k_norm.weight", .slot = "attention", .role = "k_norm"},
    TensorRule{.pattern = "blk.{layer}.attn_output.weight", .slot = "attention", .role = "output"},
    TensorRule{.pattern = "blk.{layer}.post_attention_norm.weight",
               .slot = "attention",
               .role = "post_norm"},
    // The global layers' proportional rotation factors, one table.
    TensorRule{.pattern = "rope_freqs.weight",
               .slot = "attention",
               .role = "rope_freqs",
               .layer_types = "g"},
    // Routed-expert layers: the dense GLU beside the experts.
    TensorRule{.pattern = "blk.{layer}.ffn_norm.weight", .slot = "moe", .role = "norm"},
    TensorRule{.pattern = "blk.{layer}.ffn_gate.weight", .slot = "moe", .role = "gate"},
    TensorRule{.pattern = "blk.{layer}.ffn_up.weight", .slot = "moe", .role = "up"},
    TensorRule{.pattern = "blk.{layer}.ffn_down.weight", .slot = "moe", .role = "down"},
    TensorRule{
        .pattern = "blk.{layer}.post_ffw_norm_1.weight", .slot = "moe", .role = "dense_post_norm"},
    TensorRule{.pattern = "blk.{layer}.ffn_gate_inp.weight", .slot = "moe", .role = "router"},
    TensorRule{.pattern = "blk.{layer}.ffn_gate_inp.scale", .slot = "moe", .role = "router_scale"},
    TensorRule{
        .pattern = "blk.{layer}.pre_ffw_norm_2.weight", .slot = "moe", .role = "routed_norm"},
    TensorRule{.pattern = "blk.{layer}.ffn_gate_up_exps.weight",
               .slot = "moe",
               .role = "gate_up_experts",
               .experts = true},
    TensorRule{.pattern = "blk.{layer}.ffn_down_exps.weight",
               .slot = "moe",
               .role = "down_experts",
               .experts = true},
    TensorRule{.pattern = "blk.{layer}.ffn_down_exps.scale", .slot = "moe", .role = "expert_scale"},
    TensorRule{
        .pattern = "blk.{layer}.post_ffw_norm_2.weight", .slot = "moe", .role = "routed_post_norm"},
    TensorRule{.pattern = "blk.{layer}.post_ffw_norm.weight", .slot = "moe", .role = "post_norm"},
    // Dense layers.
    TensorRule{.pattern = "blk.{layer}.ffn_norm.weight", .slot = "ffn", .role = "norm"},
    TensorRule{.pattern = "blk.{layer}.ffn_gate.weight", .slot = "ffn", .role = "gate"},
    TensorRule{.pattern = "blk.{layer}.ffn_up.weight", .slot = "ffn", .role = "up"},
    TensorRule{.pattern = "blk.{layer}.ffn_down.weight", .slot = "ffn", .role = "down"},
    TensorRule{.pattern = "blk.{layer}.post_ffw_norm.weight", .slot = "ffn", .role = "post_norm"},
    TensorRule{.pattern = "blk.{layer}.layer_output_scale.weight",
               .slot = "output_scale",
               .role = "weight"},
    TensorRule{.pattern = "output_norm.weight", .slot = "final_norm", .role = "weight"},
    // The checkpoint ties its output to the embedding (no output.weight).
    TensorRule{.pattern = "output.weight",
               .slot = "output",
               .role = "weight",
               .fallback = "token_embd.weight"},
};

constexpr std::string_view kEps = "gemma4.attention.layer_norm_rms_epsilon";
constexpr std::string_view kFfn = "gemma4.feed_forward_length";

constexpr std::array kParams = {
    FromKey(kStageSlot, "context", "gemma4.context_length", kInt),
    FromKey(kStreamSlot, "width", "gemma4.embedding_length", kInt),
    FromKey("embedding", "scale", "gemma4.embedding_length", kReal, Transform::kSqrt),
    Name("attention", "placement", "sandwich"),
    FromKey("attention", "eps", kEps, kReal),
    PerLayer("attention", "heads", "gemma4.attention.head_count", kInt),
    PerLayer("attention", "kv_heads", "gemma4.attention.head_count_kv", kInt),
    FromKey("attention", "key_dim", "gemma4.attention.key_length_swa", kInt, Transform::kNone, "l"),
    FromKey("attention", "key_dim", "gemma4.attention.key_length", kInt, Transform::kNone, "g"),
    FromKey("attention", "value_dim", "gemma4.attention.value_length_swa", kInt, Transform::kNone,
            "l"),
    FromKey("attention", "value_dim", "gemma4.attention.value_length", kInt, Transform::kNone, "g"),
    // Queries unscaled (llama.cpp's f_attention_scale), no attention softcap.
    Real("attention", "scale", 1.0),
    Real("attention", "softcap", 0.0),
    FromKey("attention", "window", "gemma4.attention.sliding_window", kInt, Transform::kNone, "l"),
    Int("attention", "window", 0, "g"),
    Name("attention", "rotation", "neox"),
    FromKey("attention", "rope_dims", "gemma4.rope.dimension_count_swa", kInt, Transform::kNone,
            "l"),
    FromKey("attention", "rope_dims", "gemma4.rope.dimension_count", kInt, Transform::kNone, "g"),
    FromKey("attention", "rope_base", "gemma4.rope.freq_base_swa", kReal, Transform::kNone, "l"),
    FromKey("attention", "rope_base", "gemma4.rope.freq_base", kReal, Transform::kNone, "g"),
    Real("attention", "rope_scale", 1.0),
    Flag("attention", "qk_norm", true),
    Flag("attention", "v_norm", true),
    Flag("attention", "kv_tie", false, "l"),
    Flag("attention", "kv_tie", true, "g"),
    Flag("attention", "rope_freqs", false, "l"),
    Flag("attention", "rope_freqs", true, "g"),
    Name("moe", "placement", "parallel"),
    FromKey("moe", "eps", kEps, kReal),
    Name("moe", "activation", "gelu_tanh"),
    PerLayer("moe", "hidden", kFfn, kInt),
    FromKey("moe", "experts", kExperts, kInt),
    FromKey("moe", "used", "gemma4.expert_used_count", kInt),
    PerLayer("moe", "expert_hidden", "gemma4.expert_feed_forward_length", kInt),
    Name("moe", "router", "softmax_topk"),
    Name("moe", "router_input", "rms_scaled"),
    Flag("moe", "normalize", true),
    Flag("moe", "expert_scales", true),
    Flag("moe", "stacked", true),
    Name("ffn", "placement", "sandwich"),
    FromKey("ffn", "eps", kEps, kReal),
    Name("ffn", "activation", "gelu_tanh"),
    PerLayer("ffn", "hidden", kFfn, kInt),
    FromKey("final_norm", "eps", kEps, kReal),
    FromKey(kHeadsSlot, "softcap", "gemma4.final_logit_softcapping", kReal, Transform::kNone, {},
            true),
    // The components implement neither KV shared between layers nor
    // per-layer input embeddings (Gemma 4's smaller checkpoints).
    RequireInt("gemma4.attention.shared_kv_layers", 0, true),
    RequireInt("gemma4.embedding_length_per_layer_input", 0, true),
};

// The serving policies: structural eligibility is shared across envelopes,
// with `owner_padded_reads` disabled because padding narrower owners' caches
// in unequal-width 4-owner global attention waves regresses Gemma 4 26B-A4B
// C4 decode by +4.715% (102.342 -> 97.734 tok/s, +0.951 ms/wave GPU span from
// 20 `concat_cont` KV-padding launches adding +3.038 ms/wave across the five
// global layers, outweighing the -1.897 ms/wave attention + grouped-stores
// gain; consistent with docs/experiments/gemma4-bounded-owner-roots/README.md).
// The bounded rules also name the prompt chunk the recipes were qualified at
// (26B-A4B at stock's matched 1,024 rows, where its MoE arithmetic reproduces
// stock's tokens, docs/experiments/gemma26-production; 31B at 256), which
// serving takes unless a setting overrides it.
constexpr std::array kPolicies = {
    PolicyRule{.gate = kExperts,
               .max_context = 4096,
               .max_slots = 2,
               .text = "max_decode_rows 16\n"
                       "disable owner_padded_reads\n",
               .prompts_alone = true,
               .prefill_lookahead = 1,
               .capture_ahead = false,
               .prefill_chunk = 1024},
    PolicyRule{.gate = kExperts,
               .max_context = 8192,
               .max_slots = 4,
               .text = "max_decode_rows 16\n"
                       "disable owner_padded_reads\n",
               .prompts_alone = true,
               .prefill_lookahead = 1,
               .capture_ahead = false,
               .prefill_chunk = 1024},
    PolicyRule{.gate = kExperts,
               .text = "max_decode_rows 16\n"
                       "disable owner_padded_reads\n",
               .prompts_alone = true,
               .owners_alone = true,
               .prefill_lookahead = 1,
               .capture_ahead = false},
    PolicyRule{.gate = kExperts,
               .gate_absent = true,
               .max_context = 4096,
               .max_slots = 2,
               .text = "max_decode_rows 16\n"
                       "disable owner_padded_reads\n",
               .prompts_alone = true,
               .prefill_lookahead = 1,
               .capture_ahead = false,
               .prefill_chunk = 256},
    PolicyRule{.gate = kExperts,
               .gate_absent = true,
               .max_context = 8192,
               .max_slots = 4,
               .text = "max_decode_rows 16\n"
                       "disable owner_padded_reads\n",
               .prompts_alone = true,
               .prefill_lookahead = 1,
               .capture_ahead = false,
               .prefill_chunk = 256},
    PolicyRule{.gate = kExperts,
               .gate_absent = true,
               .text = "max_decode_rows 16\n"
                       "disable owner_padded_reads\n",
               .prompts_alone = true,
               .owners_alone = true,
               .prefill_lookahead = 1,
               .capture_ahead = false},
};

// Its bounded serving route; the prompt chunk its envelope's policy rule
// names (kPolicies) where a setting names none. Its chat route serves the
// slice its retired M3 class did: thinking off by default and refused on,
// tool declarations refused, its channel and tool-call openers ending a
// reply (no parser here reads them).
constexpr auto kChatStops = std::to_array<std::string_view>({"<|channel>", "<|tool_call>"});
constexpr ServingRecipe kServing{.family = "Gemma",
                                 .name = "Gemma4",
                                 .bounded = true,
                                 .prefill_rows = 128,
                                 .chat_thinking = false,
                                 .chat_tools = false,
                                 .chat_stops = kChatStops};

constexpr std::array kGates = {GateRule{.key = "gemma4.expert_count", .kind = "ffn.moe"}};

constexpr Mapping kMapping{
    .architecture = "gemma4",
    .version = 1,
    .pipeline = PipelineKind::kLlm,
    .layer_count_key = "gemma4.block_count",
    .layer_types = "lllllg",
    .slots = kSlots,
    .tensors = kTensors,
    .params = kParams,
    .folds = {},
    .layer_flags_key = "gemma4.attention.sliding_window_pattern",
    .layer_flag_type = 'l',
    .policies = kPolicies,
    .serving = &kServing,
    .containers = kGgufContainers,
    .gates = kGates,
};

// -------------------------------------------------------- gemma4-assistant

constexpr std::array kAssistantSlots = {
    SlotRule{.slot = "embedding", .site = Site::kEmbedding, .kind = "embedding.projected"},
    SlotRule{.slot = "attention", .site = Site::kLayer, .kind = "attention.borrowed"},
    SlotRule{.slot = "ffn", .site = Site::kLayer, .kind = "ffn.glu"},
    SlotRule{.slot = "output_scale", .site = Site::kLayer, .kind = "scale.learned"},
    SlotRule{.slot = "final_norm", .site = Site::kFinalNorm, .kind = "norm.rms"},
    SlotRule{.slot = "output", .site = Site::kOutput, .kind = "head.linear"},
};

constexpr std::array kAssistantTensors = {
    // Its input rows are its target's token table's.
    TensorRule{
        .pattern = "token_embd.weight", .slot = "embedding", .role = "table", .target = true},
    TensorRule{.pattern = "nextn.pre_projection.weight", .slot = "embedding", .role = "projection"},
    TensorRule{.pattern = "blk.{layer}.attn_norm.weight", .slot = "attention", .role = "norm"},
    TensorRule{.pattern = "blk.{layer}.attn_q.weight", .slot = "attention", .role = "q"},
    TensorRule{.pattern = "blk.{layer}.attn_q_norm.weight", .slot = "attention", .role = "q_norm"},
    TensorRule{.pattern = "blk.{layer}.attn_output.weight", .slot = "attention", .role = "output"},
    TensorRule{.pattern = "blk.{layer}.post_attention_norm.weight",
               .slot = "attention",
               .role = "post_norm"},
    TensorRule{.pattern = "rope_freqs.weight",
               .slot = "attention",
               .role = "rope_freqs",
               .layer_types = "g"},
    TensorRule{.pattern = "blk.{layer}.ffn_norm.weight", .slot = "ffn", .role = "norm"},
    TensorRule{.pattern = "blk.{layer}.ffn_gate.weight", .slot = "ffn", .role = "gate"},
    TensorRule{.pattern = "blk.{layer}.ffn_up.weight", .slot = "ffn", .role = "up"},
    TensorRule{.pattern = "blk.{layer}.ffn_down.weight", .slot = "ffn", .role = "down"},
    TensorRule{.pattern = "blk.{layer}.post_ffw_norm.weight", .slot = "ffn", .role = "post_norm"},
    TensorRule{.pattern = "blk.{layer}.layer_output_scale.weight",
               .slot = "output_scale",
               .role = "weight"},
    TensorRule{.pattern = "output_norm.weight", .slot = "final_norm", .role = "weight"},
    // Its own token table is its head.
    TensorRule{.pattern = "token_embd.weight", .slot = "output", .role = "weight"},
    TensorRule{.pattern = "nextn.post_projection.weight", .slot = "output", .role = "projection"},
};

constexpr std::string_view kAEps = "gemma4-assistant.attention.layer_norm_rms_epsilon";
constexpr std::string_view kAOut = "gemma4-assistant.embedding_length_out";

constexpr std::array kAssistantParams = {
    FromKey(kStageSlot, "context", "gemma4-assistant.context_length", kInt),
    FromKey(kStreamSlot, "width", "gemma4-assistant.embedding_length", kInt),
    // The target's rows scaled by sqrt(the target's width), its feature beside them.
    FromKey("embedding", "scale", kAOut, kReal, Transform::kSqrt),
    FromKey("embedding", "feature", kAOut, kInt),
    Name("embedding", "tap", "features"),
    Name("attention", "placement", "sandwich"),
    FromKey("attention", "eps", kAEps, kReal),
    PerLayer("attention", "heads", "gemma4-assistant.attention.head_count", kInt),
    PerLayer("attention", "kv_heads", "gemma4-assistant.attention.head_count_kv", kInt),
    FromKey("attention", "key_dim", "gemma4-assistant.attention.key_length_swa", kInt,
            Transform::kNone, "l"),
    FromKey("attention", "key_dim", "gemma4-assistant.attention.key_length", kInt, Transform::kNone,
            "g"),
    FromKey("attention", "value_dim", "gemma4-assistant.attention.value_length_swa", kInt,
            Transform::kNone, "l"),
    FromKey("attention", "value_dim", "gemma4-assistant.attention.value_length", kInt,
            Transform::kNone, "g"),
    Real("attention", "scale", 1.0),
    Real("attention", "softcap", 0.0),
    FromKey("attention", "window", "gemma4-assistant.attention.sliding_window", kInt,
            Transform::kNone, "l"),
    Int("attention", "window", 0, "g"),
    Name("attention", "rotation", "neox"),
    FromKey("attention", "rope_dims", "gemma4-assistant.rope.dimension_count_swa", kInt,
            Transform::kNone, "l"),
    FromKey("attention", "rope_dims", "gemma4-assistant.rope.dimension_count", kInt,
            Transform::kNone, "g"),
    FromKey("attention", "rope_base", "gemma4-assistant.rope.freq_base_swa", kReal,
            Transform::kNone, "l"),
    FromKey("attention", "rope_base", "gemma4-assistant.rope.freq_base", kReal, Transform::kNone,
            "g"),
    Real("attention", "rope_scale", 1.0),
    Flag("attention", "qk_norm", true),
    Flag("attention", "rope_freqs", false, "l"),
    Flag("attention", "rope_freqs", true, "g"),
    // Each layer reads its target's last layer of its type.
    FromKey("attention", "target_layer", "llmp.target.last_layer.l", kInt, Transform::kNone, "l"),
    FromKey("attention", "target_layer", "llmp.target.last_layer.g", kInt, Transform::kNone, "g"),
    Name("ffn", "placement", "sandwich"),
    FromKey("ffn", "eps", kAEps, kReal),
    Name("ffn", "activation", "gelu_tanh"),
    PerLayer("ffn", "hidden", "gemma4-assistant.feed_forward_length", kInt),
    FromKey("final_norm", "eps", kAEps, kReal),
    // Every row an output (a drafter's), projected back to the target's width.
    Flag("output", "gather", false),
    FromKey("output", "features", kAOut, kInt),
    RequireInt("gemma4-assistant.attention.shared_kv_layers", 4, true),
    RequireInt("gemma4-assistant.embedding_length_per_layer_input", 0, true),
    RequireInt("gemma4-assistant.nextn_predict_layers", 4, true),
};

constexpr std::array kAssistantReads = {
    ReadRule{.kind = ReadKind::kState, .name = "k"},
    ReadRule{.kind = ReadKind::kState, .name = "v"},
    ReadRule{.kind = ReadKind::kTap, .name = "features", .point = TapPoint::kPostFinalNorm},
};

// The assistant's plans are plain primitives today (engine/
// gemma4_assistant.cc: no target fusion is inherited).
constexpr std::array kAssistantPolicies = {
    PolicyRule{.text = "max_decode_rows 16\n"
                       ""},
};

// Anchored on its target's tapped features, borrowing its target's state:
// three drafts a unit at most, its qualified transaction's
// (docs/experiments/gemma-assistant-throughput).
constexpr DraftRecipe kAssistantDraft{
    .shape = DraftRecipe::Shape::kAnchored, .depth = 3, .target_architecture = "gemma4"};

constexpr Mapping kAssistantMapping{
    .architecture = "gemma4-assistant",
    .version = 1,
    .pipeline = PipelineKind::kLlm,
    .layer_count_key = "gemma4-assistant.block_count",
    .layer_types = "lllg",
    .slots = kAssistantSlots,
    .tensors = kAssistantTensors,
    .params = kAssistantParams,
    .folds = {},
    .layer_flags_key = "gemma4-assistant.attention.sliding_window_pattern",
    .layer_flag_type = 'l',
    .role = StageRole::kDrafter,
    .stage = "assistant",
    .reads = kAssistantReads,
    .policies = kAssistantPolicies,
    .draft = &kAssistantDraft,
    .containers = kGgufContainers,
};

}  // namespace

const Mapping& Gemma4Mapping() { return kMapping; }
const Mapping& Gemma4AssistantMapping() { return kAssistantMapping; }

}  // namespace llmp::spec::mappings
