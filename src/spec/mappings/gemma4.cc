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

// The serving recipes (the M3 class's, retired with it, are these rules'
// origin). Within the qualified envelope (a context of at most 8,192 and
// four owners): fused norms, norm-rope and norm-add, the
// routing and reduce fused on 26B-A4B and the quantized GLU only on 31B;
// owner attention over joined one-row owners of equal reads, its masks
// joined in pairs, and at most 4,096 cells and two owners each owner's
// cache at its own width (bounded roots, common reads); every row to the
// head (no frontier_head); no rope-store, grouped stores or shared Q8
// inputs. Outside it, the frontier head's narrowed final layer and the
// fused norm-mul alone. The bounded rules also name the prompt chunk the
// recipes were qualified at (26B-A4B at stock's matched 1,024 rows, where
// its MoE arithmetic reproduces stock's tokens, docs/experiments/
// gemma26-production; 31B at 256), which serving takes unless a setting
// overrides it.
constexpr std::array kPolicies = {
    PolicyRule{.gate = kExperts,
               .max_context = 4096,
               .max_slots = 2,
               .text = "max_decode_rows 16\ndisable grouped_stores\ndisable mul_mat_glu\n"
                       "disable narrow_final\ndisable owner_chained_masks\n"
                       "disable owner_padded_reads\ndisable prepared_inputs\ndisable rope_store\n",
               .prompts_alone = true,
               .prefill_lookahead = 1,
               .capture_ahead = false,
               .prefill_chunk = 1024},
    PolicyRule{.gate = kExperts,
               .max_context = 8192,
               .max_slots = 4,
               .text = "max_decode_rows 16\ndisable grouped_stores\ndisable mul_mat_glu\n"
                       "disable narrow_final\ndisable owner_bounded_roots\n"
                       "disable owner_chained_masks\ndisable owner_padded_reads\n"
                       "disable prepared_inputs\ndisable rope_store\n",
               .prompts_alone = true,
               .prefill_lookahead = 1,
               .capture_ahead = false,
               .prefill_chunk = 1024},
    PolicyRule{.gate = kExperts,
               .text = "max_decode_rows 16\ndisable grouped_stores\ndisable mul_mat_glu\n"
                       "disable owner_attention\ndisable prepared_inputs\n"
                       "disable rms_norm_mul_add\ndisable rms_norm_mul_rope\ndisable rope_store\n"
                       "disable route_reduce\n",
               .prompts_alone = true,
               .owners_alone = true,
               .prefill_lookahead = 1,
               .capture_ahead = false},
    PolicyRule{.gate = kExperts,
               .gate_absent = true,
               .max_context = 4096,
               .max_slots = 2,
               .text = "max_decode_rows 16\ndisable grouped_stores\ndisable narrow_final\n"
                       "disable owner_chained_masks\ndisable owner_padded_reads\n"
                       "disable prepared_inputs\ndisable rope_store\n",
               .prompts_alone = true,
               .prefill_lookahead = 1,
               .capture_ahead = false,
               .prefill_chunk = 256},
    PolicyRule{.gate = kExperts,
               .gate_absent = true,
               .max_context = 8192,
               .max_slots = 4,
               .text = "max_decode_rows 16\ndisable grouped_stores\ndisable narrow_final\n"
                       "disable owner_bounded_roots\ndisable owner_chained_masks\n"
                       "disable owner_padded_reads\ndisable prepared_inputs\ndisable rope_store\n",
               .prompts_alone = true,
               .prefill_lookahead = 1,
               .capture_ahead = false,
               .prefill_chunk = 256},
    PolicyRule{.gate = kExperts,
               .gate_absent = true,
               .text = "max_decode_rows 16\ndisable grouped_stores\ndisable mul_mat_glu\n"
                       "disable owner_attention\ndisable prepared_inputs\n"
                       "disable rms_norm_mul_add\ndisable rms_norm_mul_rope\ndisable rope_store\n",
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
    PolicyRule{.text = "max_decode_rows 16\ndisable grouped_stores\ndisable mul_mat_glu\n"
                       "disable owner_attention\ndisable prepared_inputs\ndisable rms_norm_mul\n"
                       "disable rms_norm_mul_add\ndisable rms_norm_mul_rope\ndisable rope_store\n"},
};

std::vector<std::int64_t> PerLayerOf(std::uint32_t layers, auto value) {
  std::vector<std::int64_t> out;
  for (std::uint32_t il = 0; il < layers; ++il) out.push_back(static_cast<std::int64_t>(value(il)));
  return out;
}

// The approved checkpoints an assistant pairs with, as the importer
// composes an assistant's stage with its target (it keeps none of its
// target's): their GGUF metadata's values and their tensors' names and
// shapes, by the target width the assistant's metadata names
// (gemma4-assistant.embedding_length_out). Five local layers to a global
// one, no inter-layer KV sharing, no per-layer input embedding.
struct Target {
  std::uint32_t layers = 0, width = 0, heads = 0;
  std::uint32_t local_kv_heads = 0, global_kv_heads = 0;
  std::uint32_t local_head_dim = 0, global_head_dim = 0;
  std::uint32_t local_rope_dims = 0, global_rope_dims = 0;
  std::uint32_t ffn = 0, experts = 0, experts_used = 0, expert_ffn = 0;
  std::uint32_t vocab = 0, context = 0, window = 0;
  float local_rope_base = 0, global_rope_base = 0, rms_eps = 0, final_softcap = 0;
  bool local(std::uint32_t layer) const { return (layer + 1) % 6 != 0; }
  std::uint32_t head_dim(std::uint32_t layer) const {
    return local(layer) ? local_head_dim : global_head_dim;
  }
  std::uint32_t kv_heads(std::uint32_t layer) const {
    return local(layer) ? local_kv_heads : global_kv_heads;
  }
};
// gemma-4-26B-A4B-it and gemma-4-31B-it.
constexpr Target k26B{.layers = 30,
                      .width = 2816,
                      .heads = 16,
                      .local_kv_heads = 8,
                      .global_kv_heads = 2,
                      .local_head_dim = 256,
                      .global_head_dim = 512,
                      .local_rope_dims = 256,
                      .global_rope_dims = 512,
                      .ffn = 2112,
                      .experts = 128,
                      .experts_used = 8,
                      .expert_ffn = 704,
                      .vocab = 262144,
                      .context = 262144,
                      .window = 1024,
                      .local_rope_base = 10000,
                      .global_rope_base = 1000000,
                      .rms_eps = 1e-6F,
                      .final_softcap = 30};
constexpr Target k31B{.layers = 60,
                      .width = 5376,
                      .heads = 32,
                      .local_kv_heads = 16,
                      .global_kv_heads = 4,
                      .local_head_dim = 256,
                      .global_head_dim = 512,
                      .local_rope_dims = 256,
                      .global_rope_dims = 512,
                      .ffn = 21504,
                      .vocab = 262144,
                      .context = 262144,
                      .window = 1024,
                      .local_rope_base = 10000,
                      .global_rope_base = 1000000,
                      .rms_eps = 1e-6F,
                      .final_softcap = 30};

std::expected<std::vector<SourceValue>, std::string> TargetValues(const Target& p) {
  std::vector<SourceValue> out{
      Value("gemma4.context_length", p.context),
      Value("gemma4.embedding_length", p.width),
      Value("gemma4.block_count", p.layers),
      Value(kFfn, p.ffn),
      Value("gemma4.attention.head_count", p.heads),
      Value("gemma4.attention.head_count_kv",
            PerLayerOf(p.layers, [&](std::uint32_t il) { return p.kv_heads(il); })),
      Value("gemma4.attention.key_length", p.global_head_dim),
      Value("gemma4.attention.key_length_swa", p.local_head_dim),
      Value("gemma4.attention.value_length", p.global_head_dim),
      Value("gemma4.attention.value_length_swa", p.local_head_dim),
      Value(kEps, p.rms_eps),
      Value("gemma4.attention.sliding_window", p.window),
      Value("gemma4.attention.sliding_window_pattern",
            PerLayerOf(p.layers, [&](std::uint32_t il) { return p.local(il) ? 1 : 0; })),
      Value("gemma4.rope.freq_base", p.global_rope_base),
      Value("gemma4.rope.freq_base_swa", p.local_rope_base),
      Value("gemma4.rope.dimension_count", p.global_rope_dims),
      Value("gemma4.rope.dimension_count_swa", p.local_rope_dims),
      Value("gemma4.final_logit_softcapping", p.final_softcap),
  };
  if (p.experts != 0) {
    out.push_back(Value(kExperts, p.experts));
    out.push_back(Value("gemma4.expert_used_count", p.experts_used));
    out.push_back(Value("gemma4.expert_feed_forward_length", p.expert_ffn));
  }
  return out;
}

// Its tensors as its GGUF names them, BF16 matrices and F32 vectors.
Source TargetSource(const Target& p) {
  Source s{.architecture = "gemma4", .container = "gguf"};
  const auto add = [&](std::string name, std::vector<std::uint64_t> ne, bool experts = false) {
    const auto encoding = ne.size() == 1 ? format::Encoding::kF32 : format::Encoding::kBF16;
    format::TensorFormat f{
        .encoding = encoding,
        .layout = format::Layout::kDense,
        .shape = {.ne = {1, 1, 1, 1}, .rank = static_cast<std::uint8_t>(ne.size())},
    };
    for (std::size_t i = 0; i < ne.size(); ++i) f.shape.ne[i] = ne[i];
    s.tensors.push_back({.name = std::move(name),
                         .format = f,
                         .part = format::PartRole::kData,
                         .experts = experts});
  };
  add("token_embd.weight", {p.width, p.vocab});
  add("output_norm.weight", {p.width});
  add("output.weight", {p.width, p.vocab});
  add("rope_freqs.weight", {p.global_head_dim / 2});
  for (std::uint32_t il = 0; il < p.layers; ++il) {
    const std::string prefix = std::format("blk.{}.", il);
    const std::uint64_t d = p.head_dim(il), kv = p.kv_heads(il) * d, q = p.heads * d;
    add(prefix + "attn_norm.weight", {p.width});
    add(prefix + "attn_q.weight", {p.width, q});
    add(prefix + "attn_k.weight", {p.width, kv});
    if (p.local(il)) {
      add(prefix + "attn_v.weight", {p.width, kv});
    }
    add(prefix + "attn_q_norm.weight", {d});
    add(prefix + "attn_k_norm.weight", {d});
    add(prefix + "attn_output.weight", {q, p.width});
    add(prefix + "post_attention_norm.weight", {p.width});
    add(prefix + "layer_output_scale.weight", {1});
    add(prefix + "ffn_norm.weight", {p.width});
    add(prefix + "ffn_gate.weight", {p.width, p.ffn});
    add(prefix + "ffn_up.weight", {p.width, p.ffn});
    add(prefix + "ffn_down.weight", {p.ffn, p.width});
    if (p.experts != 0) {
      add(prefix + "post_ffw_norm_1.weight", {p.width});
      add(prefix + "ffn_gate_inp.weight", {p.width, p.experts});
      add(prefix + "ffn_gate_inp.scale", {p.width});
      add(prefix + "pre_ffw_norm_2.weight", {p.width});
      add(prefix + "ffn_gate_up_exps.weight", {p.width, 2U * p.expert_ffn, p.experts}, true);
      add(prefix + "ffn_down_exps.weight", {p.expert_ffn, p.width, p.experts}, true);
      add(prefix + "ffn_down_exps.scale", {p.experts});
      add(prefix + "post_ffw_norm_2.weight", {p.width});
    }
    add(prefix + "post_ffw_norm.weight", {p.width});
  }
  return s;
}

constexpr std::array<std::string_view, 2> kParentWidth{"gemma4-assistant.embedding_length_out",
                                                       "gemma4-assistant.Parent.embedding_length"};
constexpr std::array kAssistantTargets = {
    ImportTarget{.keys = kParentWidth,
                 .value = k26B.width,
                 .mapping = &Gemma4Mapping,
                 .values = +[] { return TargetValues(k26B); },
                 .source = +[] { return TargetSource(k26B); }},
    ImportTarget{.keys = kParentWidth,
                 .value = k31B.width,
                 .mapping = &Gemma4Mapping,
                 .values = +[] { return TargetValues(k31B); },
                 .source = +[] { return TargetSource(k31B); }},
};

// Anchored on its target's tapped features, borrowing its target's state:
// three drafts a unit at most, its qualified transaction's
// (docs/experiments/gemma-assistant-throughput).
constexpr DraftRecipe kAssistantDraft{
    .shape = DraftRecipe::Shape::kAnchored, .depth = 3, .import_targets = kAssistantTargets};

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
};

}  // namespace

const Mapping& Gemma4Mapping() { return kMapping; }
const Mapping& Gemma4AssistantMapping() { return kAssistantMapping; }

}  // namespace llmp::spec::mappings
