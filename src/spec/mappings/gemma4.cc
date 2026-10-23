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

constexpr std::string_view kExperts = "expert_count";
constexpr std::string_view kPolicyExperts = "gemma4.expert_count";

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

constexpr std::string_view kEps = "attention.layer_norm_rms_epsilon";
constexpr std::string_view kFfn = "feed_forward_length";

constexpr std::array kParams = {
    FromKey(kStageSlot, "context", "context_length", kInt),
    FromKey(kStreamSlot, "width", "embedding_length", kInt),
    FromKey("embedding", "scale", "embedding_length", kReal, Transform::kSqrt),
    Name("attention", "placement", "sandwich"),
    FromKey("attention", "eps", kEps, kReal),
    PerLayer("attention", "heads", "attention.head_count", kInt),
    PerLayer("attention", "kv_heads", "attention.head_count_kv", kInt),
    FromKey("attention", "key_dim", "attention.key_length_swa", kInt, Transform::kNone, "l"),
    FromKey("attention", "key_dim", "attention.key_length", kInt, Transform::kNone, "g"),
    FromKey("attention", "value_dim", "attention.value_length_swa", kInt, Transform::kNone, "l"),
    FromKey("attention", "value_dim", "attention.value_length", kInt, Transform::kNone, "g"),
    // Queries unscaled (llama.cpp's f_attention_scale), no attention softcap.
    Real("attention", "scale", 1.0),
    Real("attention", "softcap", 0.0),
    FromKey("attention", "window", "attention.sliding_window", kInt, Transform::kNone, "l"),
    Int("attention", "window", 0, "g"),
    Name("attention", "rotation", "neox"),
    FromKey("attention", "rope_dims", "rope.dimension_count_swa", kInt, Transform::kNone, "l"),
    FromKey("attention", "rope_dims", "rope.dimension_count", kInt, Transform::kNone, "g"),
    FromKey("attention", "rope_base", "rope.freq_base_swa", kReal, Transform::kNone, "l"),
    FromKey("attention", "rope_base", "rope.freq_base", kReal, Transform::kNone, "g"),
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
    FromKey("moe", "used", "expert_used_count", kInt),
    PerLayer("moe", "expert_hidden", "expert_feed_forward_length", kInt),
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
    FromKey(kHeadsSlot, "softcap", "final_logit_softcapping", kReal, Transform::kNone, {}, true),
    // The components implement neither KV shared between layers nor
    // per-layer input embeddings (Gemma 4's smaller checkpoints).
    RequireInt("attention.shared_kv_layers", 0, true),
    RequireInt("embedding_length_per_layer_input", 0, true),
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
    PolicyRule{.gate = kPolicyExperts,
               .max_context = 4096,
               .max_slots = 2,
               .text = "max_decode_rows 16\n"
                       "disable owner_padded_reads\n",
               .prompts_alone = true,
               .prefill_lookahead = 1,
               .capture_ahead = false,
               .prefill_chunk = 1024},
    PolicyRule{.gate = kPolicyExperts,
               .max_context = 8192,
               .max_slots = 4,
               .text = "max_decode_rows 16\n"
                       "disable owner_padded_reads\n",
               .prompts_alone = true,
               .prefill_lookahead = 1,
               .capture_ahead = false,
               .prefill_chunk = 1024},
    PolicyRule{.gate = kPolicyExperts,
               .text = "max_decode_rows 16\n"
                       "disable owner_padded_reads\n",
               .prompts_alone = true,
               .owners_alone = true,
               .prefill_lookahead = 1,
               .capture_ahead = false},
    PolicyRule{.gate = kPolicyExperts,
               .gate_absent = true,
               .max_context = 4096,
               .max_slots = 2,
               .text = "max_decode_rows 16\n"
                       "disable owner_padded_reads\n",
               .prompts_alone = true,
               .prefill_lookahead = 1,
               .capture_ahead = false,
               .prefill_chunk = 256},
    PolicyRule{.gate = kPolicyExperts,
               .gate_absent = true,
               .max_context = 8192,
               .max_slots = 4,
               .text = "max_decode_rows 16\n"
                       "disable owner_padded_reads\n",
               .prompts_alone = true,
               .prefill_lookahead = 1,
               .capture_ahead = false,
               .prefill_chunk = 256},
    PolicyRule{.gate = kPolicyExperts,
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

constexpr std::array kGates = {GateRule{.key = kPolicyExperts, .kind = "ffn.moe"}};

constexpr Mapping kMapping{
    .architecture = "gemma4",
    .version = 1,
    .pipeline = PipelineKind::kLlm,
    .layer_count_key = "block_count",
    .layer_types = "lllllg",
    .slots = kSlots,
    .tensors = kTensors,
    .params = kParams,
    .folds = {},
    .layer_flags_key = "attention.sliding_window_pattern",
    .layer_flag_type = 'l',
    .policies = kPolicies,
    .serving = &kServing,
    .containers = kGgufContainers,
    .gates = kGates,
    .auxiliary = kGgufAuxiliary,
};

constexpr std::array kGemma4LayerTypeCodes{
    LayerTypeCode{.value = 1, .type = 'l', .label = "sliding_attention"},
    LayerTypeCode{.value = 0, .type = 'g', .label = "full_attention"},
};
// Gemma 4 stores direct RMS gamma in HF and MLX, as in GGUF. Unlike
// Gemma 2/3, neither a zero-centered parameter nor a plus-one fold applies.
constexpr auto kGemma4NativeTensors = [] {
  std::array<TensorRule, kTensors.size() + 5> out{};
  std::size_t n = 0;
  for (auto r : kTensors) {
    if (r.pattern == "rope_freqs.weight") {
      r.optional = true;
    } else if (r.pattern == "blk.{layer}.ffn_gate_up_exps.weight") {
      r.optional = true;
      r.any_part = true;
    } else if (r.pattern == "blk.{layer}.ffn_down_exps.weight") {
      r.optional = true;
      r.any_part = true;
    } else if (r.role == "table" || r.role == "q" || r.role == "k" || r.role == "v" ||
               r.role == "output" || r.role == "gate" || r.role == "up" || r.role == "down" ||
               r.role == "router" || (r.slot == "output" && r.role == "weight")) {
      r.any_part = true;
    }
    out[n++] = r;
  }
  out[n++] = TensorRule{.pattern = "blk.{layer}.ffn_gate_exps.weight",
                        .slot = "moe",
                        .role = "gate_experts",
                        .experts = true,
                        .optional = true,
                        .any_part = true};
  out[n++] = TensorRule{.pattern = "blk.{layer}.ffn_up_exps.weight",
                        .slot = "moe",
                        .role = "up_experts",
                        .experts = true,
                        .optional = true,
                        .any_part = true};
  out[n++] = TensorRule{.pattern = "blk.{layer}.ffn_gate_exps.weight_scale_2",
                        .slot = "moe",
                        .role = "gate_scale",
                        .optional = true};
  out[n++] = TensorRule{.pattern = "blk.{layer}.ffn_up_exps.weight_scale_2",
                        .slot = "moe",
                        .role = "up_scale",
                        .optional = true};
  out[n++] = TensorRule{.pattern = "blk.{layer}.ffn_down_exps.weight_scale_2",
                        .slot = "moe",
                        .role = "down_scale",
                        .optional = true};
  return out;
}();
constexpr std::array kGemma4NativeParams = {
    FromKey(kStageSlot, "context", "context_length", kInt),
    Int(kStageSlot, "context", 131072),
    FromKey(kStreamSlot, "width", "embedding_length", kInt),
    FromKey("embedding", "scale", "embedding_length", kReal, Transform::kSqrt),
    Name("attention", "placement", "sandwich"),
    FromKey("attention", "eps", kEps, kReal),
    PerLayer("attention", "heads", "attention.head_count", kInt),
    PerLayer("attention", "kv_heads", "attention.head_count_kv", kInt, "l"),
    FromKey("attention", "kv_heads", "attention.head_count_kv_global", kInt, Transform::kNone, "g"),
    PerLayer("attention", "kv_heads", "attention.head_count_kv", kInt, "g"),
    FromKey("attention", "key_dim", "attention.key_length_swa", kInt, Transform::kNone, "l"),
    FromKey("attention", "key_dim", "attention.key_length", kInt, Transform::kNone, "g"),
    FromKey("attention", "key_dim", "attention.key_length_swa", kInt, Transform::kNone, "g"),
    FromKey("attention", "value_dim", "attention.value_length_swa", kInt, Transform::kNone, "l"),
    FromKey("attention", "value_dim", "attention.key_length_swa", kInt, Transform::kNone, "l"),
    FromKey("attention", "value_dim", "attention.value_length", kInt, Transform::kNone, "g"),
    FromKey("attention", "value_dim", "attention.key_length", kInt, Transform::kNone, "g"),
    FromKey("attention", "value_dim", "attention.key_length_swa", kInt, Transform::kNone, "g"),
    Real("attention", "scale", 1.0),
    Real("attention", "softcap", 0.0),
    FromKey("attention", "window", "attention.sliding_window", kInt, Transform::kNone, "l"),
    Int("attention", "window", 0, "g"),
    Name("attention", "rotation", "neox"),
    FromKey("attention", "rope_dims", "rope.dimension_count_swa", kInt, Transform::kNone, "l"),
    FromKey("attention", "rope_dims", "attention.key_length_swa", kInt, Transform::kNone, "l"),
    FromKey("attention", "rope_dims", "rope.dimension_count", kInt, Transform::kNone, "g"),
    FromKey("attention", "rope_partial_factor", "rope.partial_rotary_factor", kReal,
            Transform::kNone, "g", true),
    FromKey("attention", "rope_dims", "attention.key_length", kInt, Transform::kNone, "g"),
    FromKey("attention", "rope_base", "rope.freq_base_swa", kReal, Transform::kNone, "l"),
    Real("attention", "rope_base", 10000.0, "l"),
    FromKey("attention", "rope_base", "rope.freq_base", kReal, Transform::kNone, "g"),
    Real("attention", "rope_base", 1000000.0, "g"),
    Real("attention", "rope_scale", 1.0),
    Flag("attention", "qk_norm", true),
    Flag("attention", "v_norm", true),
    Flag("attention", "kv_tie", false, "l"),
    Flag("attention", "kv_tie", true, "g"),
    Flag("attention", "rope_freqs", false, "l"),
    FromTensor("attention", "rope_freqs", "rope_freqs.weight", "g"),
    Flag("attention", "rope_freqs", false, "g"),
    Name("moe", "placement", "parallel"),
    FromKey("moe", "eps", kEps, kReal),
    Name("moe", "activation", "gelu_tanh"),
    PerLayer("moe", "hidden", kFfn, kInt),
    FromKey("moe", "experts", kExperts, kInt),
    FromKey("moe", "used", "expert_used_count", kInt),
    PerLayer("moe", "expert_hidden", "expert_feed_forward_length", kInt),
    Name("moe", "router", "softmax_topk"),
    Name("moe", "router_input", "rms_scaled"),
    Flag("moe", "normalize", true),
    Flag("moe", "expert_scales", true),
    Flag("moe", "projection_scales", false),
    FromTensor("moe", "gate_scale", "blk.{layer}.ffn_gate_exps.weight_scale_2"),
    Flag("moe", "gate_scale", false),
    FromTensor("moe", "up_scale", "blk.{layer}.ffn_up_exps.weight_scale_2"),
    Flag("moe", "up_scale", false),
    FromTensor("moe", "down_scale", "blk.{layer}.ffn_down_exps.weight_scale_2"),
    Flag("moe", "down_scale", false),
    FromTensor("moe", "stacked", "blk.{layer}.ffn_gate_up_exps.weight"),
    Flag("moe", "stacked", false),
    Name("ffn", "placement", "sandwich"),
    FromKey("ffn", "eps", kEps, kReal),
    Name("ffn", "activation", "gelu_tanh"),
    PerLayer("ffn", "hidden", kFfn, kInt),
    FromKey("final_norm", "eps", kEps, kReal),
    FromKey(kHeadsSlot, "softcap", "final_logit_softcapping", kReal, Transform::kNone, {}, true),
    RequireInt("attention.shared_kv_layers", 0, true),
    RequireInt("embedding_length_per_layer_input", 0, true),
    Require("hidden_activation", "gelu_pytorch_tanh", true),
    Require("hidden_act", "gelu_pytorch_tanh", true),
    Require("rope.full.type", "proportional", true),
    Require("rope.local.type", "default", true),
    Require("text_model_type", "gemma4_text", true),
    ParamRule{.key = "attention_bias", .type = kBool, .value = {.flag = false}, .optional = true},
    ParamRule{
        .key = "is_encoder_decoder", .type = kBool, .value = {.flag = false}, .optional = true},
    ParamRule{.key = "attention_k_eq_v", .type = kBool, .value = {.flag = true}, .optional = true},
    ParamRule{
        .key = "use_double_wide_mlp", .type = kBool, .value = {.flag = false}, .optional = true},
    ParamRule{.key = "use_bidirectional_attention",
              .type = kBool,
              .value = {.flag = false},
              .flag_text = "vision",
              .optional = true},
};
constexpr std::array kGemma4KeyAliases = {
    KeyAlias{"text_config.enable_moe_block", "enable_moe_block"},
    KeyAlias{"text_config.hidden_activation", "hidden_activation"},
    KeyAlias{"text_config.hidden_act", "hidden_act"},
    KeyAlias{"text_config.attention_bias", "attention_bias"},
    KeyAlias{"text_config.is_encoder_decoder", "is_encoder_decoder"},
    KeyAlias{"text_config.attention_k_eq_v", "attention_k_eq_v"},
    KeyAlias{"text_config.use_double_wide_mlp", "use_double_wide_mlp"},
    KeyAlias{"text_config.use_bidirectional_attention", "use_bidirectional_attention"},
    KeyAlias{"text_config.model_type", "text_model_type"},
    KeyAlias{"text_config.rope_parameters.full_attention.rope_type", "rope.full.type"},
    KeyAlias{"text_config.rope_parameters.sliding_attention.rope_type", "rope.local.type"},
    KeyAlias{"rope_parameters.full_attention.rope_type", "rope.full.type"},
    KeyAlias{"rope_parameters.sliding_attention.rope_type", "rope.local.type"},
    KeyAlias{"text_config.num_hidden_layers", "block_count"},
    KeyAlias{"num_hidden_layers", "block_count"},
    KeyAlias{"text_config.max_position_embeddings", "context_length"},
    KeyAlias{"max_position_embeddings", "context_length"},
    KeyAlias{"text_config.hidden_size", "embedding_length"},
    KeyAlias{"hidden_size", "embedding_length"},
    KeyAlias{"text_config.intermediate_size", kFfn},
    KeyAlias{"intermediate_size", kFfn},
    KeyAlias{"text_config.rms_norm_eps", kEps},
    KeyAlias{"rms_norm_eps", kEps},
    KeyAlias{"text_config.num_attention_heads", "attention.head_count"},
    KeyAlias{"num_attention_heads", "attention.head_count"},
    KeyAlias{"text_config.num_key_value_heads", "attention.head_count_kv"},
    KeyAlias{"num_key_value_heads", "attention.head_count_kv"},
    KeyAlias{"text_config.num_global_key_value_heads", "attention.head_count_kv_global"},
    KeyAlias{"num_global_key_value_heads", "attention.head_count_kv_global"},
    KeyAlias{"text_config.head_dim", "attention.key_length_swa"},
    KeyAlias{"head_dim", "attention.key_length_swa"},
    KeyAlias{"text_config.global_head_dim", "attention.key_length"},
    KeyAlias{"global_head_dim", "attention.key_length"},
    KeyAlias{"text_config.sliding_window", "attention.sliding_window"},
    KeyAlias{"sliding_window", "attention.sliding_window"},
    KeyAlias{"text_config.rope_local_base_freq", "rope.freq_base_swa"},
    KeyAlias{"rope_local_base_freq", "rope.freq_base_swa"},
    KeyAlias{"text_config.rope_parameters.sliding_attention.rope_theta", "rope.freq_base_swa"},
    KeyAlias{"rope_parameters.sliding_attention.rope_theta", "rope.freq_base_swa"},
    KeyAlias{"text_config.rope_theta", "rope.freq_base"},
    KeyAlias{"rope_theta", "rope.freq_base"},
    KeyAlias{"text_config.rope_parameters.full_attention.rope_theta", "rope.freq_base"},
    KeyAlias{"rope_parameters.full_attention.rope_theta", "rope.freq_base"},
    KeyAlias{"text_config.rope_parameters.full_attention.partial_rotary_factor",
             "rope.partial_rotary_factor"},
    KeyAlias{"rope_parameters.full_attention.partial_rotary_factor", "rope.partial_rotary_factor"},
    KeyAlias{"text_config.num_experts", kExperts},
    KeyAlias{"num_experts", kExperts},
    KeyAlias{"text_config.top_k_experts", "expert_used_count"},
    KeyAlias{"top_k_experts", "expert_used_count"},
    KeyAlias{"text_config.num_experts_per_tok", "expert_used_count"},
    KeyAlias{"num_experts_per_tok", "expert_used_count"},
    KeyAlias{"text_config.moe_intermediate_size", "expert_feed_forward_length"},
    KeyAlias{"moe_intermediate_size", "expert_feed_forward_length"},
    KeyAlias{"text_config.num_kv_shared_layers", "attention.shared_kv_layers"},
    KeyAlias{"num_kv_shared_layers", "attention.shared_kv_layers"},
    KeyAlias{"text_config.hidden_size_per_layer_input", "embedding_length_per_layer_input"},
    KeyAlias{"hidden_size_per_layer_input", "embedding_length_per_layer_input"},
    KeyAlias{"text_config.layer_types", "layer_types"},
    KeyAlias{"layer_types", "layer_types"},
    KeyAlias{"text_config.final_logit_softcapping", "final_logit_softcapping"},
    KeyAlias{"final_logit_softcapping", "final_logit_softcapping"},
};
constexpr std::array kGemma4TensorAliases = {
    TensorAlias{"model.language_model.embed_tokens.weight", "token_embd.weight"},
    TensorAlias{"model.language_model.norm.weight", "output_norm.weight"},
    TensorAlias{"model.language_model.lm_head.weight", "output.weight"},
    TensorAlias{"model.language_model.layers.{layer}.input_layernorm.weight",
                "blk.{layer}.attn_norm.weight"},
    TensorAlias{"model.language_model.layers.{layer}.self_attn.q_proj.weight",
                "blk.{layer}.attn_q.weight"},
    TensorAlias{"model.language_model.layers.{layer}.self_attn.k_proj.weight",
                "blk.{layer}.attn_k.weight"},
    TensorAlias{"model.language_model.layers.{layer}.self_attn.v_proj.weight",
                "blk.{layer}.attn_v.weight"},
    TensorAlias{"model.language_model.layers.{layer}.self_attn.q_norm.weight",
                "blk.{layer}.attn_q_norm.weight"},
    TensorAlias{"model.language_model.layers.{layer}.self_attn.k_norm.weight",
                "blk.{layer}.attn_k_norm.weight"},
    TensorAlias{"model.language_model.layers.{layer}.self_attn.o_proj.weight",
                "blk.{layer}.attn_output.weight"},
    TensorAlias{"model.language_model.layers.{layer}.post_attention_layernorm.weight",
                "blk.{layer}.post_attention_norm.weight"},
    TensorAlias{"model.language_model.layers.{layer}.pre_feedforward_layernorm.weight",
                "blk.{layer}.ffn_norm.weight"},
    TensorAlias{"model.language_model.layers.{layer}.mlp.gate_proj.weight",
                "blk.{layer}.ffn_gate.weight"},
    TensorAlias{"model.language_model.layers.{layer}.mlp.up_proj.weight",
                "blk.{layer}.ffn_up.weight"},
    TensorAlias{"model.language_model.layers.{layer}.mlp.down_proj.weight",
                "blk.{layer}.ffn_down.weight"},
    TensorAlias{"model.language_model.layers.{layer}.post_feedforward_layernorm.weight",
                "blk.{layer}.post_ffw_norm.weight"},
    TensorAlias{"model.language_model.layers.{layer}.post_feedforward_layernorm_1.weight",
                "blk.{layer}.post_ffw_norm_1.weight"},
    TensorAlias{"model.language_model.layers.{layer}.pre_feedforward_layernorm_2.weight",
                "blk.{layer}.pre_ffw_norm_2.weight"},
    TensorAlias{"model.language_model.layers.{layer}.post_feedforward_layernorm_2.weight",
                "blk.{layer}.post_ffw_norm_2.weight"},
    TensorAlias{"model.language_model.layers.{layer}.mlp.router.proj.weight",
                "blk.{layer}.ffn_gate_inp.weight"},
    TensorAlias{"model.language_model.layers.{layer}.mlp.router.scale",
                "blk.{layer}.ffn_gate_inp.scale"},
    TensorAlias{"model.language_model.layers.{layer}.router.proj.weight",
                "blk.{layer}.ffn_gate_inp.weight"},
    TensorAlias{"model.language_model.layers.{layer}.router.scale",
                "blk.{layer}.ffn_gate_inp.scale"},
    TensorAlias{"model.language_model.layers.{layer}.router.per_expert_scale",
                "blk.{layer}.ffn_down_exps.scale"},
    TensorAlias{"model.language_model.layers.{layer}.mlp.experts.gate_up_proj",
                "blk.{layer}.ffn_gate_up_exps.weight"},
    TensorAlias{"model.language_model.layers.{layer}.mlp.experts.gate_up_proj.weight",
                "blk.{layer}.ffn_gate_up_exps.weight"},
    TensorAlias{"model.language_model.layers.{layer}.mlp.experts.down_proj",
                "blk.{layer}.ffn_down_exps.weight"},
    TensorAlias{"model.language_model.layers.{layer}.mlp.experts.down_proj.weight",
                "blk.{layer}.ffn_down_exps.weight"},
    TensorAlias{"model.language_model.layers.{layer}.mlp.experts.per_expert_scale",
                "blk.{layer}.ffn_down_exps.scale"},
    TensorAlias{"model.language_model.layers.{layer}.mlp.experts.{expert}.gate_proj.weight",
                "blk.{layer}.ffn_gate_exps.weight"},
    TensorAlias{"model.language_model.layers.{layer}.mlp.experts.{expert}.up_proj.weight",
                "blk.{layer}.ffn_up_exps.weight"},
    TensorAlias{"model.language_model.layers.{layer}.mlp.experts.{expert}.down_proj.weight",
                "blk.{layer}.ffn_down_exps.weight"},
    TensorAlias{"model.language_model.layers.{layer}.experts.{expert}.gate_proj.weight",
                "blk.{layer}.ffn_gate_exps.weight"},
    TensorAlias{"model.language_model.layers.{layer}.experts.{expert}.up_proj.weight",
                "blk.{layer}.ffn_up_exps.weight"},
    TensorAlias{"model.language_model.layers.{layer}.experts.{expert}.down_proj.weight",
                "blk.{layer}.ffn_down_exps.weight"},
    TensorAlias{"model.language_model.layers.{layer}.layer_scalar",
                "blk.{layer}.layer_output_scale.weight"},
    TensorAlias{"model.language_model.layers.{layer}.layer_scalar.weight",
                "blk.{layer}.layer_output_scale.weight"},
    TensorAlias{"model.embed_tokens.weight", "token_embd.weight"},
    TensorAlias{"model.norm.weight", "output_norm.weight"},
    TensorAlias{"lm_head.weight", "output.weight"},
    TensorAlias{"model.layers.{layer}.input_layernorm.weight", "blk.{layer}.attn_norm.weight"},
    TensorAlias{"model.layers.{layer}.self_attn.q_proj.weight", "blk.{layer}.attn_q.weight"},
    TensorAlias{"model.layers.{layer}.self_attn.k_proj.weight", "blk.{layer}.attn_k.weight"},
    TensorAlias{"model.layers.{layer}.self_attn.v_proj.weight", "blk.{layer}.attn_v.weight"},
    TensorAlias{"model.layers.{layer}.self_attn.q_norm.weight", "blk.{layer}.attn_q_norm.weight"},
    TensorAlias{"model.layers.{layer}.self_attn.k_norm.weight", "blk.{layer}.attn_k_norm.weight"},
    TensorAlias{"model.layers.{layer}.self_attn.o_proj.weight", "blk.{layer}.attn_output.weight"},
    TensorAlias{"model.layers.{layer}.post_attention_layernorm.weight",
                "blk.{layer}.post_attention_norm.weight"},
    TensorAlias{"model.layers.{layer}.pre_feedforward_layernorm.weight",
                "blk.{layer}.ffn_norm.weight"},
    TensorAlias{"model.layers.{layer}.mlp.gate_proj.weight", "blk.{layer}.ffn_gate.weight"},
    TensorAlias{"model.layers.{layer}.mlp.up_proj.weight", "blk.{layer}.ffn_up.weight"},
    TensorAlias{"model.layers.{layer}.mlp.down_proj.weight", "blk.{layer}.ffn_down.weight"},
    TensorAlias{"model.layers.{layer}.post_feedforward_layernorm.weight",
                "blk.{layer}.post_ffw_norm.weight"},
    TensorAlias{"model.layers.{layer}.post_feedforward_layernorm_1.weight",
                "blk.{layer}.post_ffw_norm_1.weight"},
    TensorAlias{"model.layers.{layer}.pre_feedforward_layernorm_2.weight",
                "blk.{layer}.pre_ffw_norm_2.weight"},
    TensorAlias{"model.layers.{layer}.post_feedforward_layernorm_2.weight",
                "blk.{layer}.post_ffw_norm_2.weight"},
    TensorAlias{"model.layers.{layer}.mlp.router.proj.weight", "blk.{layer}.ffn_gate_inp.weight"},
    TensorAlias{"model.layers.{layer}.mlp.router.scale", "blk.{layer}.ffn_gate_inp.scale"},
    TensorAlias{"model.layers.{layer}.router.proj.weight", "blk.{layer}.ffn_gate_inp.weight"},
    TensorAlias{"model.layers.{layer}.router.scale", "blk.{layer}.ffn_gate_inp.scale"},
    TensorAlias{"model.layers.{layer}.router.per_expert_scale", "blk.{layer}.ffn_down_exps.scale"},
    TensorAlias{"model.layers.{layer}.mlp.experts.gate_up_proj",
                "blk.{layer}.ffn_gate_up_exps.weight"},
    TensorAlias{"model.layers.{layer}.mlp.experts.gate_up_proj.weight",
                "blk.{layer}.ffn_gate_up_exps.weight"},
    TensorAlias{"model.layers.{layer}.mlp.experts.down_proj", "blk.{layer}.ffn_down_exps.weight"},
    TensorAlias{"model.layers.{layer}.mlp.experts.down_proj.weight",
                "blk.{layer}.ffn_down_exps.weight"},
    TensorAlias{"model.layers.{layer}.mlp.experts.per_expert_scale",
                "blk.{layer}.ffn_down_exps.scale"},
    TensorAlias{"model.layers.{layer}.mlp.experts.{expert}.gate_proj.weight",
                "blk.{layer}.ffn_gate_exps.weight"},
    TensorAlias{"model.layers.{layer}.mlp.experts.{expert}.up_proj.weight",
                "blk.{layer}.ffn_up_exps.weight"},
    TensorAlias{"model.layers.{layer}.mlp.experts.{expert}.down_proj.weight",
                "blk.{layer}.ffn_down_exps.weight"},
    TensorAlias{"model.layers.{layer}.experts.{expert}.gate_proj.weight",
                "blk.{layer}.ffn_gate_exps.weight"},
    TensorAlias{"model.layers.{layer}.experts.{expert}.up_proj.weight",
                "blk.{layer}.ffn_up_exps.weight"},
    TensorAlias{"model.layers.{layer}.experts.{expert}.down_proj.weight",
                "blk.{layer}.ffn_down_exps.weight"},
    TensorAlias{"model.layers.{layer}.layer_scalar", "blk.{layer}.layer_output_scale.weight"},
    TensorAlias{"model.layers.{layer}.layer_scalar.weight",
                "blk.{layer}.layer_output_scale.weight"},
    TensorAlias{"language_model.model.embed_tokens.weight", "token_embd.weight"},
    TensorAlias{"language_model.model.norm.weight", "output_norm.weight"},
    TensorAlias{"language_model.lm_head.weight", "output.weight"},
    TensorAlias{"language_model.model.layers.{layer}.input_layernorm.weight",
                "blk.{layer}.attn_norm.weight"},
    TensorAlias{"language_model.model.layers.{layer}.self_attn.q_proj.weight",
                "blk.{layer}.attn_q.weight"},
    TensorAlias{"language_model.model.layers.{layer}.self_attn.k_proj.weight",
                "blk.{layer}.attn_k.weight"},
    TensorAlias{"language_model.model.layers.{layer}.self_attn.v_proj.weight",
                "blk.{layer}.attn_v.weight"},
    TensorAlias{"language_model.model.layers.{layer}.self_attn.q_norm.weight",
                "blk.{layer}.attn_q_norm.weight"},
    TensorAlias{"language_model.model.layers.{layer}.self_attn.k_norm.weight",
                "blk.{layer}.attn_k_norm.weight"},
    TensorAlias{"language_model.model.layers.{layer}.self_attn.o_proj.weight",
                "blk.{layer}.attn_output.weight"},
    TensorAlias{"language_model.model.layers.{layer}.post_attention_layernorm.weight",
                "blk.{layer}.post_attention_norm.weight"},
    TensorAlias{"language_model.model.layers.{layer}.pre_feedforward_layernorm.weight",
                "blk.{layer}.ffn_norm.weight"},
    TensorAlias{"language_model.model.layers.{layer}.mlp.gate_proj.weight",
                "blk.{layer}.ffn_gate.weight"},
    TensorAlias{"language_model.model.layers.{layer}.mlp.up_proj.weight",
                "blk.{layer}.ffn_up.weight"},
    TensorAlias{"language_model.model.layers.{layer}.mlp.down_proj.weight",
                "blk.{layer}.ffn_down.weight"},
    TensorAlias{"language_model.model.layers.{layer}.post_feedforward_layernorm.weight",
                "blk.{layer}.post_ffw_norm.weight"},
    TensorAlias{"language_model.model.layers.{layer}.post_feedforward_layernorm_1.weight",
                "blk.{layer}.post_ffw_norm_1.weight"},
    TensorAlias{"language_model.model.layers.{layer}.pre_feedforward_layernorm_2.weight",
                "blk.{layer}.pre_ffw_norm_2.weight"},
    TensorAlias{"language_model.model.layers.{layer}.post_feedforward_layernorm_2.weight",
                "blk.{layer}.post_ffw_norm_2.weight"},
    TensorAlias{"language_model.model.layers.{layer}.mlp.router.proj.weight",
                "blk.{layer}.ffn_gate_inp.weight"},
    TensorAlias{"language_model.model.layers.{layer}.mlp.router.scale",
                "blk.{layer}.ffn_gate_inp.scale"},
    TensorAlias{"language_model.model.layers.{layer}.router.proj.weight",
                "blk.{layer}.ffn_gate_inp.weight"},
    TensorAlias{"language_model.model.layers.{layer}.router.scale",
                "blk.{layer}.ffn_gate_inp.scale"},
    TensorAlias{"language_model.model.layers.{layer}.router.per_expert_scale",
                "blk.{layer}.ffn_down_exps.scale"},
    TensorAlias{"language_model.model.layers.{layer}.mlp.experts.gate_up_proj",
                "blk.{layer}.ffn_gate_up_exps.weight"},
    TensorAlias{"language_model.model.layers.{layer}.mlp.experts.gate_up_proj.weight",
                "blk.{layer}.ffn_gate_up_exps.weight"},
    TensorAlias{"language_model.model.layers.{layer}.mlp.experts.down_proj",
                "blk.{layer}.ffn_down_exps.weight"},
    TensorAlias{"language_model.model.layers.{layer}.mlp.experts.down_proj.weight",
                "blk.{layer}.ffn_down_exps.weight"},
    TensorAlias{"language_model.model.layers.{layer}.mlp.experts.per_expert_scale",
                "blk.{layer}.ffn_down_exps.scale"},
    TensorAlias{"language_model.model.layers.{layer}.mlp.experts.{expert}.gate_proj.weight",
                "blk.{layer}.ffn_gate_exps.weight"},
    TensorAlias{"language_model.model.layers.{layer}.mlp.experts.{expert}.up_proj.weight",
                "blk.{layer}.ffn_up_exps.weight"},
    TensorAlias{"language_model.model.layers.{layer}.mlp.experts.{expert}.down_proj.weight",
                "blk.{layer}.ffn_down_exps.weight"},
    TensorAlias{"language_model.model.layers.{layer}.experts.{expert}.gate_proj.weight",
                "blk.{layer}.ffn_gate_exps.weight"},
    TensorAlias{"language_model.model.layers.{layer}.experts.{expert}.up_proj.weight",
                "blk.{layer}.ffn_up_exps.weight"},
    TensorAlias{"language_model.model.layers.{layer}.experts.{expert}.down_proj.weight",
                "blk.{layer}.ffn_down_exps.weight"},
    TensorAlias{"language_model.model.layers.{layer}.layer_scalar",
                "blk.{layer}.layer_output_scale.weight"},
    TensorAlias{"language_model.model.layers.{layer}.layer_scalar.weight",
                "blk.{layer}.layer_output_scale.weight"},
};
constexpr std::array<std::string_view, 18> kGemma4IgnoredTensors{
    "model.language_model.layers.{layer}.mlp.experts.{expert}.gate_proj.input_scale",
    "model.language_model.layers.{layer}.mlp.experts.{expert}.up_proj.input_scale",
    "model.language_model.layers.{layer}.mlp.experts.{expert}.down_proj.input_scale",
    "model.language_model.layers.{layer}.experts.{expert}.gate_proj.input_scale",
    "model.language_model.layers.{layer}.experts.{expert}.up_proj.input_scale",
    "model.language_model.layers.{layer}.experts.{expert}.down_proj.input_scale",
    "model.layers.{layer}.mlp.experts.{expert}.gate_proj.input_scale",
    "model.layers.{layer}.mlp.experts.{expert}.up_proj.input_scale",
    "model.layers.{layer}.mlp.experts.{expert}.down_proj.input_scale",
    "model.layers.{layer}.experts.{expert}.gate_proj.input_scale",
    "model.layers.{layer}.experts.{expert}.up_proj.input_scale",
    "model.layers.{layer}.experts.{expert}.down_proj.input_scale",
    "language_model.model.layers.{layer}.mlp.experts.{expert}.gate_proj.input_scale",
    "language_model.model.layers.{layer}.mlp.experts.{expert}.up_proj.input_scale",
    "language_model.model.layers.{layer}.mlp.experts.{expert}.down_proj.input_scale",
    "language_model.model.layers.{layer}.experts.{expert}.gate_proj.input_scale",
    "language_model.model.layers.{layer}.experts.{expert}.up_proj.input_scale",
    "language_model.model.layers.{layer}.experts.{expert}.down_proj.input_scale",
};
constexpr std::array<std::string_view, 7> kGemma4IgnoredPrefixes{
    "model.vision_tower.", "model.embed_vision.",    "model.audio_tower.", "model.embed_audio.",
    "vision_tower.",       "multi_modal_projector.", "audio_tower."};
// These settings are math-bearing even in nested text configs. Training,
// tokenizer and excluded media metadata remain source provenance.
constexpr auto kGemma4NativeAuxiliary = kNativeAuxiliary;
constexpr std::array kGemma4NativeSemantic{std::string_view("rope_*"),
                                           std::string_view("rotary_*"),
                                           std::string_view("attention_*"),
                                           std::string_view("attn_*"),
                                           std::string_view("hidden_*"),
                                           std::string_view("num_*"),
                                           std::string_view("rms_norm_*"),
                                           std::string_view("norm_*"),
                                           std::string_view("head_dim"),
                                           std::string_view("global_head_dim"),
                                           std::string_view("partial_rotary_factor"),
                                           std::string_view("final_logit_softcapping"),
                                           std::string_view("use_bidirectional_attention"),
                                           std::string_view("use_double_wide_mlp"),
                                           std::string_view("is_encoder_decoder"),
                                           std::string_view("enable_moe_block"),
                                           std::string_view("moe_*"),
                                           std::string_view("top_k_experts"),
                                           std::string_view("zz_*")};
constexpr std::array kGemma4NativeNulls{
    std::string_view("rope_scaling"),
    std::string_view("rope_parameters"),
    std::string_view("final_logit_softcapping"),
    std::string_view("use_bidirectional_attention"),
    std::string_view("text_config.rope_scaling"),
    std::string_view("text_config.rope_parameters"),
    std::string_view("text_config.final_logit_softcapping"),
    std::string_view("text_config.use_bidirectional_attention")};
constexpr std::array kGemma4NativeRequirements{
    ScalarRequirement{.key = "enable_moe_block",
                      .optional = true,
                      .boolean = true,
                      .flag_of_positive = "expert_count"}};
constexpr std::array<std::string_view, 3> kGemma4Containers{"gguf", "safetensors", "mlx"};
constexpr std::array<std::string_view, 1> kGemma4NativeSourceAliases{"gemma4_text"};
constexpr std::array kGemma4AllTensors = [] {
  constexpr auto new_roles = std::ranges::count_if(kGemma4NativeTensors, [](const auto& native) {
    return !std::ranges::any_of(kTensors, [&](const auto& base) {
      return base.slot == native.slot && base.role == native.role;
    });
  });
  std::array<TensorRule, kTensors.size() + new_roles> out{};
  std::ranges::copy(kTensors, out.begin());
  std::size_t next = kTensors.size();
  for (const auto& native : kGemma4NativeTensors) {
    if (std::ranges::any_of(kTensors, [&](const auto& base) {
          return base.slot == native.slot && base.role == native.role;
        }))
      continue;
    out[next++] = native;
  }
  return out;
}();
constexpr std::array kGemma4AllParams = [] {
  constexpr auto new_roles = std::ranges::count_if(kGemma4NativeParams, [](const auto& native) {
    return !std::ranges::any_of(kParams, [&](const auto& base) {
      return base.slot == native.slot && base.param == native.param;
    });
  });
  std::array<ParamRule, kParams.size() + new_roles> out{};
  std::ranges::copy(kParams, out.begin());
  std::size_t next = kParams.size();
  for (const auto& native : kGemma4NativeParams) {
    if (std::ranges::any_of(kParams, [&](const auto& base) {
          return base.slot == native.slot && base.param == native.param;
        }))
      continue;
    out[next++] = native;
  }
  return out;
}();
constexpr std::array<SourceDialect, 5> kGemma4Dialects = {
    SourceDialect{.architecture = "gemma4",
                  .container = "gguf",
                  .execution_architecture = "gemma4",
                  .source_architectures = {},
                  .key_prefix = {},
                  .layer_count_key = "block_count",
                  .context_key = {},
                  .layer_types = "lllllg",
                  .layer_flags_key = "attention.sliding_window_pattern",
                  .layer_flag_type = 'l',
                  .explicit_layer_rules = true,
                  .complete_params = true,
                  .params = kParams,
                  .tensors = kTensors,
                  .folds = {},
                  .auxiliary = kGgufAuxiliary,
                  .policies = kPolicies,
                  .gates = kGates,
                  .extended_components = false,
                  .explicit_source_semantics = true},
    SourceDialect{.architecture = "gemma4_text",
                  .container = "safetensors",
                  .execution_architecture = "gemma4",
                  .source_architectures = kGemma4NativeSourceAliases,
                  .source_key_prefix = "text_config.",
                  .layer_count_key = "block_count",
                  .context_key = "context_length",
                  .softcap_key = "final_logit_softcapping",
                  .layer_types = "lllllg",
                  .layer_type_key = "layer_types",
                  .layer_flags_key = "attention.sliding_window_pattern",
                  .layer_flag_type = 'l',
                  .layer_type_codes = kGemma4LayerTypeCodes,
                  .explicit_layer_rules = true,
                  .complete_params = true,
                  .params = kGemma4NativeParams,
                  .tensors = kGemma4NativeTensors,
                  .tensor_aliases = kGemma4TensorAliases,
                  .key_aliases = kGemma4KeyAliases,
                  .auxiliary = kGemma4NativeAuxiliary,
                  .requirements = kGemma4NativeRequirements,
                  .semantic = kGemma4NativeSemantic,
                  .ignored_tensors = kGemma4IgnoredTensors,
                  .null_semantics = kGemma4NativeNulls,
                  .ignored_tensor_prefixes = kGemma4IgnoredPrefixes,
                  .policies = kPolicies,
                  .gates = kGates,
                  .extended_components = true,
                  .explicit_source_semantics = true},
    SourceDialect{.architecture = "gemma4_text",
                  .container = "mlx",
                  .execution_architecture = "gemma4",
                  .source_architectures = kGemma4NativeSourceAliases,
                  .source_key_prefix = "text_config.",
                  .layer_count_key = "block_count",
                  .context_key = "context_length",
                  .softcap_key = "final_logit_softcapping",
                  .layer_types = "lllllg",
                  .layer_type_key = "layer_types",
                  .layer_flags_key = "attention.sliding_window_pattern",
                  .layer_flag_type = 'l',
                  .layer_type_codes = kGemma4LayerTypeCodes,
                  .explicit_layer_rules = true,
                  .complete_params = true,
                  .params = kGemma4NativeParams,
                  .tensors = kGemma4NativeTensors,
                  .tensor_aliases = kGemma4TensorAliases,
                  .key_aliases = kGemma4KeyAliases,
                  .auxiliary = kGemma4NativeAuxiliary,
                  .requirements = kGemma4NativeRequirements,
                  .semantic = kGemma4NativeSemantic,
                  .ignored_tensors = kGemma4IgnoredTensors,
                  .null_semantics = kGemma4NativeNulls,
                  .ignored_tensor_prefixes = kGemma4IgnoredPrefixes,
                  .policies = kPolicies,
                  .gates = kGates,
                  .extended_components = true,
                  .explicit_source_semantics = true},
    SourceDialect{.architecture = "gemma4",
                  .container = "safetensors",
                  .execution_architecture = "gemma4",
                  .source_architectures = kGemma4NativeSourceAliases,
                  .source_key_prefix = "text_config.",
                  .layer_count_key = "block_count",
                  .context_key = "context_length",
                  .softcap_key = "final_logit_softcapping",
                  .layer_types = "lllllg",
                  .layer_type_key = "layer_types",
                  .layer_flags_key = "attention.sliding_window_pattern",
                  .layer_flag_type = 'l',
                  .layer_type_codes = kGemma4LayerTypeCodes,
                  .explicit_layer_rules = true,
                  .complete_params = true,
                  .params = kGemma4NativeParams,
                  .tensors = kGemma4NativeTensors,
                  .tensor_aliases = kGemma4TensorAliases,
                  .key_aliases = kGemma4KeyAliases,
                  .auxiliary = kGemma4NativeAuxiliary,
                  .requirements = kGemma4NativeRequirements,
                  .semantic = kGemma4NativeSemantic,
                  .ignored_tensors = kGemma4IgnoredTensors,
                  .null_semantics = kGemma4NativeNulls,
                  .ignored_tensor_prefixes = kGemma4IgnoredPrefixes,
                  .policies = kPolicies,
                  .gates = kGates,
                  .extended_components = true,
                  .explicit_source_semantics = true},
    SourceDialect{.architecture = "gemma4",
                  .container = "mlx",
                  .execution_architecture = "gemma4",
                  .source_architectures = kGemma4NativeSourceAliases,
                  .source_key_prefix = "text_config.",
                  .layer_count_key = "block_count",
                  .context_key = "context_length",
                  .softcap_key = "final_logit_softcapping",
                  .layer_types = "lllllg",
                  .layer_type_key = "layer_types",
                  .layer_flags_key = "attention.sliding_window_pattern",
                  .layer_flag_type = 'l',
                  .layer_type_codes = kGemma4LayerTypeCodes,
                  .explicit_layer_rules = true,
                  .complete_params = true,
                  .params = kGemma4NativeParams,
                  .tensors = kGemma4NativeTensors,
                  .tensor_aliases = kGemma4TensorAliases,
                  .key_aliases = kGemma4KeyAliases,
                  .auxiliary = kGemma4NativeAuxiliary,
                  .requirements = kGemma4NativeRequirements,
                  .semantic = kGemma4NativeSemantic,
                  .ignored_tensors = kGemma4IgnoredTensors,
                  .null_semantics = kGemma4NativeNulls,
                  .ignored_tensor_prefixes = kGemma4IgnoredPrefixes,
                  .policies = kPolicies,
                  .gates = kGates,
                  .extended_components = true,
                  .explicit_source_semantics = true}};
constexpr Mapping kCanonicalMapping = [] {
  auto m = kMapping;
  m.source_architectures = kGemma4NativeSourceAliases;
  m.tensors = kGemma4AllTensors;
  m.params = kGemma4AllParams;
  m.containers = kGemma4Containers;
  m.dialects = kGemma4Dialects;
  return m;
}();

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

constexpr std::string_view kAEps = "attention.layer_norm_rms_epsilon";
constexpr std::string_view kAOut = "embedding_length_out";

constexpr std::array kAssistantParams = {
    FromKey(kStageSlot, "context", "context_length", kInt),
    FromKey(kStreamSlot, "width", "embedding_length", kInt),
    // The target's rows scaled by sqrt(the target's width), its feature beside them.
    FromKey("embedding", "scale", kAOut, kReal, Transform::kSqrt),
    FromKey("embedding", "feature", kAOut, kInt),
    Name("embedding", "tap", "features"),
    Name("attention", "placement", "sandwich"),
    FromKey("attention", "eps", kAEps, kReal),
    PerLayer("attention", "heads", "attention.head_count", kInt),
    PerLayer("attention", "kv_heads", "attention.head_count_kv", kInt),
    FromKey("attention", "key_dim", "attention.key_length_swa", kInt, Transform::kNone, "l"),
    FromKey("attention", "key_dim", "attention.key_length", kInt, Transform::kNone, "g"),
    FromKey("attention", "value_dim", "attention.value_length_swa", kInt, Transform::kNone, "l"),
    FromKey("attention", "value_dim", "attention.value_length", kInt, Transform::kNone, "g"),
    Real("attention", "scale", 1.0),
    Real("attention", "softcap", 0.0),
    FromKey("attention", "window", "attention.sliding_window", kInt, Transform::kNone, "l"),
    Int("attention", "window", 0, "g"),
    Name("attention", "rotation", "neox"),
    FromKey("attention", "rope_dims", "rope.dimension_count_swa", kInt, Transform::kNone, "l"),
    FromKey("attention", "rope_dims", "rope.dimension_count", kInt, Transform::kNone, "g"),
    FromKey("attention", "rope_base", "rope.freq_base_swa", kReal, Transform::kNone, "l"),
    FromKey("attention", "rope_base", "rope.freq_base", kReal, Transform::kNone, "g"),
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
    PerLayer("ffn", "hidden", "feed_forward_length", kInt),
    FromKey("final_norm", "eps", kAEps, kReal),
    // Every row an output (a drafter's), projected back to the target's width.
    Flag("output", "gather", false),
    FromKey("output", "features", kAOut, kInt),
    RequireInt("attention.shared_kv_layers", 4, true),
    RequireInt("embedding_length_per_layer_input", 0, true),
    RequireInt("nextn_predict_layers", 4, true),
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
// three drafts is its measured default, not a structural maximum
// (docs/experiments/gemma-assistant-throughput).
constexpr DraftRecipe kAssistantDraft{
    .shape = DraftRecipe::Shape::kAnchored, .depth = 3, .target_architecture = "gemma4"};

constexpr Mapping kAssistantMapping{
    .architecture = "gemma4-assistant",
    .version = 1,
    .pipeline = PipelineKind::kLlm,
    .key_prefix = "gemma4-assistant",
    .layer_count_key = "block_count",
    .layer_types = "lllg",
    .slots = kAssistantSlots,
    .tensors = kAssistantTensors,
    .params = kAssistantParams,
    .folds = {},
    .layer_flags_key = "attention.sliding_window_pattern",
    .layer_flag_type = 'l',
    .role = StageRole::kDrafter,
    .stage = "assistant",
    .reads = kAssistantReads,
    .policies = kAssistantPolicies,
    .draft = &kAssistantDraft,
    .containers = kGgufContainers,
    .auxiliary = kGgufAuxiliary,
};

}  // namespace

const Mapping& Gemma4Mapping() { return kCanonicalMapping; }
const Mapping& Gemma4SafetensorsMapping() { return kCanonicalMapping; }
const Mapping& Gemma4AssistantMapping() { return kAssistantMapping; }

}  // namespace llmp::spec::mappings
