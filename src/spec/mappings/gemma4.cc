// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "spec/mappings/gemma4.h"

#include <array>
#include <cstdint>
#include <expected>
#include <string>
#include <string_view>
#include <vector>

#include "model/gemma4.h"
#include "model/gemma4_assistant.h"
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

// Today's production recipe within its envelope (gemma26_production,
// gemma31_production): fused norms, norm-rope and norm-add; the quantized
// GLU only on 31B; no rope-store, grouped stores or shared Q8 inputs.
constexpr std::array kPolicies = {
    PolicyRule{.gate = kExperts,
               .text = "max_decode_rows 16\ndisable grouped_stores\ndisable mul_mat_glu\n"
                       "disable prepared_inputs\ndisable rope_store\n"},
    PolicyRule{.gate = kExperts,
               .gate_absent = true,
               .text = "max_decode_rows 16\ndisable grouped_stores\ndisable prepared_inputs\n"
                       "disable rope_store\n"},
};

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
};

// -------------------------------------------------------- gemma4-assistant

constexpr std::string_view kA = "gemma4-assistant.";

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
    RequireInt("gemma4-assistant.embedding_length_per_layer_input", 0, true),
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
                       "disable prepared_inputs\ndisable rms_norm_mul\ndisable rms_norm_mul_add\n"
                       "disable rms_norm_mul_rope\ndisable rope_store\n"},
};

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
};

std::vector<std::int64_t> PerLayerOf(std::uint32_t layers, auto value) {
  std::vector<std::int64_t> out;
  for (std::uint32_t il = 0; il < layers; ++il) out.push_back(static_cast<std::int64_t>(value(il)));
  return out;
}

}  // namespace

const Mapping& Gemma4Mapping() { return kMapping; }
const Mapping& Gemma4AssistantMapping() { return kAssistantMapping; }

std::expected<std::vector<SourceValue>, std::string> Gemma4V0Values(const model::Gemma4Profile& p) {
  for (std::uint32_t il = 0; il < p.layers; ++il) {
    if (p.local(il) != (kMapping.layer_types.at(il % kMapping.layer_types.size()) == 'l')) {
      return std::unexpected("the Gemma 4 profile's local layers are not the mapping's");
    }
  }
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

std::expected<std::vector<SourceValue>, std::string> Gemma4AssistantV0Values(
    const model::Gemma4AssistantProfile& a, const model::Gemma4Profile& t) {
  // The composition reads each local layer from the target's last local
  // layer and the global one from its last global layer.
  std::uint32_t last_local = 0;
  std::uint32_t last_global = 0;
  for (std::uint32_t il = 0; il < t.layers; ++il) {
    (t.local(il) ? last_local : last_global) = il;
  }
  if (a.target_width != t.width || a.local_target_layer != last_local ||
      a.global_target_layer != last_global || a.local_kv_heads != t.local_kv_heads ||
      a.global_kv_heads != t.global_kv_heads) {
    return std::unexpected("the assistant's profile is not its target's");
  }
  const auto layers = model::kGemma4AssistantLayers;
  const auto local = [&](std::uint32_t il) {
    return kAssistantMapping.layer_types.at(il % kAssistantMapping.layer_types.size()) == 'l';
  };
  return std::vector<SourceValue>{
      Value(std::string(kA) + "context_length", t.context),
      Value(std::string(kA) + "embedding_length", model::kGemma4AssistantWidth),
      Value(std::string(kA) + "block_count", layers),
      Value(std::string(kA) + "feed_forward_length", model::kGemma4AssistantFfn),
      Value(std::string(kA) + "attention.head_count", a.heads),
      Value(std::string(kA) + "attention.head_count_kv",
            PerLayerOf(layers,
                       [&](std::uint32_t il) {
                         return local(il) ? a.local_kv_heads : a.global_kv_heads;
                       })),
      Value(std::string(kA) + "attention.key_length", t.global_head_dim),
      Value(std::string(kA) + "attention.key_length_swa", t.local_head_dim),
      Value(std::string(kA) + "attention.value_length", t.global_head_dim),
      Value(std::string(kA) + "attention.value_length_swa", t.local_head_dim),
      Value(kAEps, t.rms_eps),
      Value(std::string(kA) + "attention.sliding_window", t.window),
      Value(std::string(kA) + "attention.sliding_window_pattern",
            PerLayerOf(layers, [&](std::uint32_t il) { return local(il) ? 1 : 0; })),
      Value(std::string(kA) + "rope.freq_base", t.global_rope_base),
      Value(std::string(kA) + "rope.freq_base_swa", t.local_rope_base),
      Value(std::string(kA) + "rope.dimension_count", t.global_rope_dims),
      Value(std::string(kA) + "rope.dimension_count_swa", t.local_rope_dims),
      Value(kAOut, a.target_width),
  };
}

}  // namespace llmp::spec::mappings
