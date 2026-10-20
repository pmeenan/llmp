// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "spec/mappings/qwen38.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <expected>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

#include "format/encoding.h"
#include "format/layout.h"
#include "spec/mapping.h"
#include "spec/mappings/rules.h"
#include "spec/spec.h"

namespace llmp::spec::mappings {
namespace {

using enum ValueType;
using format::PartRole;

// The binding's facts (absent: a GGUF checkpoint's form).
constexpr std::string_view kIndexer = "llmp.qwen4exp.indexer";
constexpr std::string_view kStacked = "llmp.qwen4exp.experts_stacked";
constexpr std::string_view kScaled = "llmp.qwen4exp.expert_scales";
constexpr std::string_view kPleLayers = "qwen4exp.ple.layers";
// The native importer's fact: the checkpoint quantized its routed experts'
// activations too (their input_scale tensors are present: W4A4).
constexpr std::string_view kActivationScales = "llmp.qwen4exp.expert_activation_scales";

// A matrix one import keeps whole and the other as MXFP8 codes beside
// their scales: the tensor with whatever part it holds, and its scales.
constexpr TensorRule Matrix(std::string_view pattern, std::string_view slot,
                            std::string_view role) {
  return {.pattern = pattern, .slot = slot, .role = role, .any_part = true};
}
constexpr TensorRule Scales(std::string_view pattern, std::string_view slot,
                            std::string_view role) {
  return {.pattern = pattern,
          .slot = slot,
          .role = role,
          .part = PartRole::kBlockScales,
          .optional = true};
}
constexpr TensorRule Biases(std::string_view pattern, std::string_view slot,
                            std::string_view role) {
  return {
      .pattern = pattern, .slot = slot, .role = role, .part = PartRole::kBiases, .optional = true};
}
constexpr TensorRule Plain(std::string_view pattern, std::string_view slot, std::string_view role) {
  return {.pattern = pattern, .slot = slot, .role = role};
}

// The folds the GGUF converter applies (the native importer records its
// own: spec/stored.h).
constexpr std::array kFolds = {
    FoldRule{.fold = "norm.plus_one", .container = "gguf"},
    FoldRule{.fold = "ssm.a_neg_exp", .container = "gguf"},
    FoldRule{.fold = "gdn.value_heads_tiled", .container = "gguf"},
};

// ---------------------------------------------------------------- qwen4exp

constexpr std::array kSlots = {
    SlotRule{.slot = "embedding", .site = Site::kEmbedding, .kind = "embedding.token"},
    SlotRule{.slot = "ngram", .site = Site::kSide, .kind = "side.ngram", .layers_key = kPleLayers},
    SlotRule{.slot = "streams", .site = Site::kMixing, .kind = "residual.hc"},
    // A layer: the n-gram block (its layers only), then each block between
    // its mix and its combine.
    SlotRule{.slot = "ple", .site = Site::kLayer, .kind = "residual.ple", .layers_key = kPleLayers},
    SlotRule{.slot = "attn_mix", .site = Site::kLayer, .kind = "hc.mix"},
    SlotRule{.slot = "linear", .site = Site::kLayer, .kind = "attention.gdn", .layer_types = "l"},
    SlotRule{
        .slot = "attention", .site = Site::kLayer, .kind = "attention.qsa", .layer_types = "f"},
    SlotRule{.slot = "attn_combine", .site = Site::kLayer, .kind = "hc.combine"},
    SlotRule{.slot = "ffn_mix", .site = Site::kLayer, .kind = "hc.mix"},
    SlotRule{.slot = "moe", .site = Site::kLayer, .kind = "ffn.moe"},
    SlotRule{.slot = "ffn_combine", .site = Site::kLayer, .kind = "hc.combine"},
    SlotRule{.slot = "output_mix", .site = Site::kFinalNorm, .kind = "hc.mix"},
    SlotRule{.slot = "output", .site = Site::kOutput, .kind = "head.linear"},
};

constexpr std::array kTensors = {
    Plain("token_embd.weight", "embedding", "table"),
    // The n-gram table (a ModelOpt import's with its tensor scale); its
    // hash's constants, a ModelOpt import's tensors too, are parameters.
    Plain("per_layer_token_embd.weight", "ngram", "table"),
    TensorRule{.pattern = "per_layer_token_embd.weight_scale_2",
               .slot = "ngram",
               .role = "table",
               .part = PartRole::kTensorScale,
               .optional = true},
    TensorRule{
        .pattern = "blk.{layer}.ple_multipliers", .slot = "ple", .optional = true, .unbound = true},
    TensorRule{.pattern = "blk.{layer}.ple_head_offsets",
               .slot = "ple",
               .optional = true,
               .unbound = true},
    TensorRule{
        .pattern = "blk.{layer}.ple_head_vocab", .slot = "ple", .optional = true, .unbound = true},
    Matrix("blk.{layer}.ple_key.weight", "ple", "key"),
    Scales("blk.{layer}.ple_key.weight_scale", "ple", "key"),
    Biases("blk.{layer}.ple_key.weight_bias", "ple", "key"),
    Matrix("blk.{layer}.ple_value.weight", "ple", "value"),
    Scales("blk.{layer}.ple_value.weight_scale", "ple", "value"),
    Biases("blk.{layer}.ple_value.weight_bias", "ple", "value"),
    Plain("blk.{layer}.ple_norm_key.weight", "ple", "norm_key"),
    Plain("blk.{layer}.ple_norm_query.weight", "ple", "norm_query"),
    Plain("blk.{layer}.ple_norm_conv.weight", "ple", "norm_conv"),
    Plain("blk.{layer}.ple_conv1d.weight", "ple", "conv"),
    Plain("blk.{layer}.hc_attn_norm.weight", "attn_mix", "norm"),
    Matrix("blk.{layer}.hc_attn_down.weight", "attn_mix", "down"),
    Scales("blk.{layer}.hc_attn_down.weight_scale", "attn_mix", "down"),
    Biases("blk.{layer}.hc_attn_down.weight_bias", "attn_mix", "down"),
    Matrix("blk.{layer}.hc_attn_up.weight", "attn_mix", "up"),
    Scales("blk.{layer}.hc_attn_up.weight_scale", "attn_mix", "up"),
    Biases("blk.{layer}.hc_attn_up.weight_bias", "attn_mix", "up"),
    Matrix("blk.{layer}.hc_attn_inject.weight", "attn_combine", "inject"),
    Scales("blk.{layer}.hc_attn_inject.weight_scale", "attn_combine", "inject"),
    Biases("blk.{layer}.hc_attn_inject.weight_bias", "attn_combine", "inject"),
    Plain("blk.{layer}.hc_ffn_norm.weight", "ffn_mix", "norm"),
    Matrix("blk.{layer}.hc_ffn_down.weight", "ffn_mix", "down"),
    Scales("blk.{layer}.hc_ffn_down.weight_scale", "ffn_mix", "down"),
    Biases("blk.{layer}.hc_ffn_down.weight_bias", "ffn_mix", "down"),
    Matrix("blk.{layer}.hc_ffn_up.weight", "ffn_mix", "up"),
    Scales("blk.{layer}.hc_ffn_up.weight_scale", "ffn_mix", "up"),
    Biases("blk.{layer}.hc_ffn_up.weight_bias", "ffn_mix", "up"),
    Matrix("blk.{layer}.hc_ffn_inject.weight", "ffn_combine", "inject"),
    Scales("blk.{layer}.hc_ffn_inject.weight_scale", "ffn_combine", "inject"),
    Biases("blk.{layer}.hc_ffn_inject.weight_bias", "ffn_combine", "inject"),
    // Gated DeltaNet.
    Matrix("blk.{layer}.attn_qkv.weight", "linear", "qkv"),
    Scales("blk.{layer}.attn_qkv.weight_scale", "linear", "qkv"),
    Biases("blk.{layer}.attn_qkv.weight_bias", "linear", "qkv"),
    Matrix("blk.{layer}.attn_gate.weight", "linear", "gate"),
    Scales("blk.{layer}.attn_gate.weight_scale", "linear", "gate"),
    Biases("blk.{layer}.attn_gate.weight_bias", "linear", "gate"),
    Matrix("blk.{layer}.ssm_beta.weight", "linear", "beta"),
    Scales("blk.{layer}.ssm_beta.weight_scale", "linear", "beta"),
    Biases("blk.{layer}.ssm_beta.weight_bias", "linear", "beta"),
    Matrix("blk.{layer}.ssm_alpha.weight", "linear", "alpha"),
    Scales("blk.{layer}.ssm_alpha.weight_scale", "linear", "alpha"),
    Biases("blk.{layer}.ssm_alpha.weight_bias", "linear", "alpha"),
    Plain("blk.{layer}.ssm_dt.bias", "linear", "dt_bias"),
    Plain("blk.{layer}.ssm_a", "linear", "a"),
    Plain("blk.{layer}.ssm_conv1d.weight", "linear", "conv"),
    Plain("blk.{layer}.ssm_norm.weight", "linear", "norm"),
    Matrix("blk.{layer}.ssm_out.weight", "linear", "output"),
    Scales("blk.{layer}.ssm_out.weight_scale", "linear", "output"),
    Biases("blk.{layer}.ssm_out.weight_bias", "linear", "output"),
    // QSA: the indexer's projection fused (ModelOpt) or split (GGUF).
    Matrix("blk.{layer}.attn_q.weight", "attention", "q"),
    Scales("blk.{layer}.attn_q.weight_scale", "attention", "q"),
    Biases("blk.{layer}.attn_q.weight_bias", "attention", "q"),
    Matrix("blk.{layer}.attn_k.weight", "attention", "k"),
    Scales("blk.{layer}.attn_k.weight_scale", "attention", "k"),
    Biases("blk.{layer}.attn_k.weight_bias", "attention", "k"),
    Matrix("blk.{layer}.attn_v.weight", "attention", "v"),
    Scales("blk.{layer}.attn_v.weight_scale", "attention", "v"),
    Biases("blk.{layer}.attn_v.weight_bias", "attention", "v"),
    Matrix("blk.{layer}.attn_output.weight", "attention", "output"),
    Scales("blk.{layer}.attn_output.weight_scale", "attention", "output"),
    Biases("blk.{layer}.attn_output.weight_bias", "attention", "output"),
    Plain("blk.{layer}.attn_q_norm.weight", "attention", "q_norm"),
    Plain("blk.{layer}.attn_k_norm.weight", "attention", "k_norm"),
    TensorRule{.pattern = "blk.{layer}.indexer.qk_proj.weight",
               .slot = "attention",
               .role = "indexer_qk",
               .optional = true,
               .any_part = true},
    Scales("blk.{layer}.indexer.qk_proj.weight_scale", "attention", "indexer_qk"),
    Biases("blk.{layer}.indexer.qk_proj.weight_bias", "attention", "indexer_qk"),
    TensorRule{.pattern = "blk.{layer}.indexer.q_proj.weight",
               .slot = "attention",
               .role = "indexer_q",
               .optional = true,
               .any_part = true},
    TensorRule{.pattern = "blk.{layer}.indexer.k_proj.weight",
               .slot = "attention",
               .role = "indexer_k",
               .optional = true,
               .any_part = true},
    Plain("blk.{layer}.indexer.q_norm.weight", "attention", "indexer_q_norm"),
    Plain("blk.{layer}.indexer.k_norm.weight", "attention", "indexer_k_norm"),
    // The MoE: router, shared expert and its gate, routed experts (GGML's
    // layout, or the CUTLASS layout's stacked gate-up), their scales.
    Matrix("blk.{layer}.ffn_gate_inp.weight", "moe", "router"),
    Matrix("blk.{layer}.ffn_gate_inp_shexp.weight", "moe", "shared_gate"),
    Scales("blk.{layer}.ffn_gate_inp_shexp.weight_scale", "moe", "shared_gate"),
    Biases("blk.{layer}.ffn_gate_inp_shexp.weight_bias", "moe", "shared_gate"),
    Matrix("blk.{layer}.ffn_gate_shexp.weight", "moe", "gate"),
    Scales("blk.{layer}.ffn_gate_shexp.weight_scale", "moe", "gate"),
    Biases("blk.{layer}.ffn_gate_shexp.weight_bias", "moe", "gate"),
    Matrix("blk.{layer}.ffn_up_shexp.weight", "moe", "up"),
    Scales("blk.{layer}.ffn_up_shexp.weight_scale", "moe", "up"),
    Biases("blk.{layer}.ffn_up_shexp.weight_bias", "moe", "up"),
    Matrix("blk.{layer}.ffn_down_shexp.weight", "moe", "down"),
    Scales("blk.{layer}.ffn_down_shexp.weight_scale", "moe", "down"),
    Biases("blk.{layer}.ffn_down_shexp.weight_bias", "moe", "down"),
    TensorRule{.pattern = "blk.{layer}.ffn_gate_exps.weight",
               .slot = "moe",
               .role = "gate_experts",
               .experts = true,
               .optional = true},
    TensorRule{.pattern = "blk.{layer}.ffn_up_exps.weight",
               .slot = "moe",
               .role = "up_experts",
               .experts = true,
               .optional = true},
    TensorRule{.pattern = "blk.{layer}.ffn_down_exps.weight",
               .slot = "moe",
               .role = "down_experts",
               .experts = true,
               .optional = true},
    TensorRule{.pattern = "blk.{layer}.ffn_gate_up_exps.codes",
               .slot = "moe",
               .role = "gate_up_experts",
               .part = PartRole::kCodes,
               .experts = true,
               .optional = true},
    TensorRule{.pattern = "blk.{layer}.ffn_gate_up_exps.scales",
               .slot = "moe",
               .role = "gate_up_experts",
               .part = PartRole::kBlockScales,
               .experts = true,
               .optional = true},
    TensorRule{.pattern = "blk.{layer}.ffn_down_exps.codes",
               .slot = "moe",
               .role = "down_experts",
               .part = PartRole::kCodes,
               .experts = true,
               .optional = true},
    TensorRule{.pattern = "blk.{layer}.ffn_down_exps.scales",
               .slot = "moe",
               .role = "down_experts",
               .part = PartRole::kBlockScales,
               .experts = true,
               .optional = true},
    TensorRule{.pattern = "blk.{layer}.ffn_gate_exps.weight_scale_2",
               .slot = "moe",
               .role = "gate_scale",
               .optional = true},
    TensorRule{.pattern = "blk.{layer}.ffn_up_exps.weight_scale_2",
               .slot = "moe",
               .role = "up_scale",
               .optional = true},
    TensorRule{.pattern = "blk.{layer}.ffn_down_exps.weight_scale_2",
               .slot = "moe",
               .role = "down_scale",
               .optional = true},
    Plain("output_hc_norm.weight", "output_mix", "norm"),
    Matrix("output_hc_down.weight", "output_mix", "down"),
    Scales("output_hc_down.weight_scale", "output_mix", "down"),
    Biases("output_hc_down.weight_bias", "output_mix", "down"),
    Matrix("output_hc_up.weight", "output_mix", "up"),
    Scales("output_hc_up.weight_scale", "output_mix", "up"),
    Biases("output_hc_up.weight_bias", "output_mix", "up"),
    Matrix("output.weight", "output", "weight"),
};

constexpr std::string_view kEps = "qwen4exp.attention.layer_norm_rms_epsilon";

// The QSA, MoE and mix parameters both mappings read, under a prefix's keys.
#define LLMP_QWEN38_BLOCK_PARAMS(P)                                                             \
  Name("streams", "mix", "sigmoid"),                                                            \
      FromKey("attn_mix", "eps", P "attention.layer_norm_rms_epsilon", kReal),                  \
      FromKey("attn_mix", "rank", P "hyper_connection.low_rank", kInt),                         \
      Flag("attn_mix", "inject", true),                                                         \
      FromKey("ffn_mix", "eps", P "attention.layer_norm_rms_epsilon", kReal),                   \
      FromKey("ffn_mix", "rank", P "hyper_connection.low_rank", kInt),                          \
      Flag("ffn_mix", "inject", true),                                                          \
      FromKey("output_mix", "eps", P "attention.layer_norm_rms_epsilon", kReal),                \
      FromKey("output_mix", "rank", P "hyper_connection.low_rank", kInt),                       \
      Flag("output_mix", "inject", false),                                                      \
      FromKey("attention", "eps", P "attention.layer_norm_rms_epsilon", kReal),                 \
      FromKey("attention", "heads", P "attention.head_count", kInt),                            \
      FromKey("attention", "kv_heads", P "attention.head_count_kv", kInt),                      \
      FromKey("attention", "head_dim", P "attention.key_length", kInt),                         \
      FromKey("attention", "rope_dims", P "rope.dimension_count", kInt),                        \
      FromKey("attention", "rope_base", P "rope.freq_base", kReal),                             \
      FromKey("attention", "rope_sections", P "rope.dimension_sections", kInts),                \
      FromKey("attention", "indexer_heads", P "attention.indexer.head_count", kInt),            \
      FromKey("attention", "indexer_dim", P "attention.indexer.key_length", kInt),              \
      PerLayer("attention", "ratio", P "attention.compress_ratios", kInt),                      \
      FromKey("attention", "budget", P "attention.indexer.top_k", kInt),                        \
      FromKey("attention", "indexer", kIndexer, kString, Transform::kNone, {}, true),           \
      Name("attention", "indexer", "split"), Name("moe", "placement", "none"),                  \
      FromKey("moe", "eps", P "attention.layer_norm_rms_epsilon", kReal),                       \
      Name("moe", "activation", "silu"),                                                        \
      FromKey("moe", "hidden", P "expert_shared_feed_forward_length", kInt),                    \
      FromKey("moe", "experts", P "expert_count", kInt),                                        \
      FromKey("moe", "used", P "expert_used_count", kInt),                                      \
      FromKey("moe", "expert_hidden", P "expert_feed_forward_length", kInt),                    \
      Name("moe", "router", "softmax_topk"), Name("moe", "router_input", "stream"),             \
      Flag("moe", "normalize", true), Flag("moe", "shared_gate", true),                         \
      ParamRule{                                                                                \
          .slot = "moe", .param = "stacked", .key = kStacked, .type = kBool, .optional = true}, \
      Flag("moe", "stacked", false),                                                            \
      ParamRule{.slot = "moe",                                                                  \
                .param = "projection_scales",                                                   \
                .key = kScaled,                                                                 \
                .type = kBool,                                                                  \
                .optional = true},                                                              \
      Flag("moe", "projection_scales", false),                                                  \
      FromKey(kStageSlot, "context", P "context_length", kInt),                                 \
      FromKey(kStreamSlot, "width", P "embedding_length", kInt),                                \
      FromKey(kStreamSlot, "streams", P "hyper_connection.count", kInt)

constexpr std::array kParams = {
    LLMP_QWEN38_BLOCK_PARAMS("qwen4exp."),
    Real("embedding", "scale", 1.0),
    // A verify's tokens as GGML's argmax picks them (the fast form's).
    Flag("output", "native_argmax", true),
    FromKey("ngram", "ngram", "qwen4exp.ple.ngram_size", kInt),
    FromKey("ngram", "heads_per_ngram", "qwen4exp.ple.heads_per_ngram", kInt),
    FromKey("ngram", "row", "qwen4exp.embedding_length_per_layer_input", kInt),
    FromKey("ngram", "eos", "qwen4exp.ple.eos_token_id", kInt),
    FromKey("ngram", "multipliers", "qwen4exp.ple.layer_multipliers", kInts),
    FromKey("ngram", "offsets", "qwen4exp.ple.head_offsets", kInts),
    FromKey("ngram", "vocab", "qwen4exp.ple.head_vocab_sizes", kInts),
    FromKey("ple", "eps", kEps, kReal),
    Name("ple", "input", "ngram"),
    FromKey("ple", "conv", "qwen4exp.ple.conv_kernel", kInt),
    FromKey("ple", "dilation", "qwen4exp.ple.ngram_size", kInt),
    FromKey("linear", "eps", kEps, kReal),
    FromKey("linear", "k_heads", "qwen4exp.ssm.group_count", kInt),
    FromKey("linear", "v_heads", "qwen4exp.ssm.time_step_rank", kInt),
    FromKey("linear", "head_dim", "qwen4exp.ssm.state_size", kInt),
    FromKey("linear", "conv", "qwen4exp.ssm.conv_kernel", kInt),
    // The components' layer pattern: a QSA layer every fourth.
    RequireInt("qwen4exp.full_attention_interval", 4, false),
};

// The caller's owner-row arithmetic requirement. Graph patterns default by
// semantic/capability proof; changed rounding needs admitted numerical facts.
// This policy is shared by the stacked, scaled and ordinary representations.
constexpr std::array kPolicies = {
    PolicyRule{.gate = kStacked,
               .text = "max_decode_rows 16\n"
                       "enable owner_row_invariant\n",
               .common_read = true},
    PolicyRule{.gate = kScaled,
               .text = "max_decode_rows 16\n"
                       "enable owner_row_invariant\n",
               .common_read = true},
    PolicyRule{.text = "max_decode_rows 16\n"
                       "enable owner_row_invariant\n",
               .common_read = true},
};

// Its serving: the derived shared route, qualified by its plain or joined
// envelopes; its QSA selection's F32 [read cells, rows] tensors bound a
// prompt chunk's rows (RE-037: their plane strides in 32 bits); the MTP
// keys, with their measured defaults: drafts a step at most (depths 2-3
// chosen by acceptance, AdaptiveDepth), the draft head's rows (the fastest
// of 32,768 rows to the whole vocabulary, docs/experiments/qwen38-mtp),
// depth 3's cost over depth 2's (execution/adaptive_depth.h), a wave's
// drafts a request (C2/C4 HTTP screens: +8%/+15% over adaptive depth), how
// many requests draft in one joined wave (past two each drafts alone: 3.6%
// faster at C4), and the state a wave reads, rounded up (the plans' reuse
// across depths); a conversation-owned adaptive depth of three.
constexpr ServingRecipe kServing{.family = "Qwen3.8",
                                 .name = "Qwen3.8",
                                 .prefill_rows = 4096,
                                 .draft_rows = true,
                                 .mtp_waves = true,
                                 .default_draft_rows = 3,
                                 .default_draft_vocab = 65536,
                                 .default_depth_cost_ratio = 1.16,
                                 .default_shared_wave_depth = 2,
                                 .default_draft_wave_max = 2,
                                 .default_wave_read_align = 2048,

                                 .frozen_host_masks = false};

constexpr std::array kGates = {
    GateRule{.key = "qwen4exp.expert_count", .kind = "ffn.moe"},
    GateRule{.key = kStacked, .kind = "ffn.moe", .param = "stacked"},
    GateRule{.key = kActivationScales,
             .kind = "ffn.moe",
             .param = "stacked",
             .container = "safetensors"},
    GateRule{.key = kScaled, .kind = "ffn.moe", .param = "projection_scales"},
};
constexpr std::array kNativeFolds = {
    FoldRule{.fold = "norm.plus_one", .container = "safetensors", .imported = true},
    FoldRule{.fold = "ssm.a_neg_exp", .container = "safetensors", .imported = true},
    FoldRule{.fold = "gdn.value_heads_tiled", .container = "safetensors", .imported = true},
    FoldRule{.fold = "norm.plus_one", .container = "mlx", .imported = true},
    FoldRule{.fold = "ssm.a_neg_exp", .container = "mlx", .imported = true},
    FoldRule{.fold = "gdn.value_heads_tiled", .container = "mlx", .imported = true},
};

constexpr Mapping kMapping{
    .architecture = "qwen4exp",
    .version = 1,
    .pipeline = PipelineKind::kLlm,
    .layer_count_key = "qwen4exp.block_count",
    .layer_types = "lllf",
    .slots = kSlots,
    .tensors = kTensors,
    .params = kParams,
    .folds = kFolds,
    .layer_flags_key = "qwen4exp.attention.compress_ratios",
    .layer_flag_type = 'f',
    .policies = kPolicies,
    .serving = &kServing,
    .gates = kGates,
};

// Closed GGUF target projection: whole encoded data parts, split indexer,
// source-owned PLE constants, and the existing GGUF planning semantics.
constexpr bool GgufTensor(const TensorRule& r) {
  return r.part == PartRole::kData && !r.unbound && !r.pattern.contains("weight_scale") &&
         !r.pattern.ends_with(".weight_bias") && !r.pattern.contains("ffn_gate_up_exps") &&
         !r.pattern.contains("indexer.qk_proj");
}
constexpr auto kGgufTensors = [] {
  std::array<TensorRule, std::ranges::count_if(kTensors, GgufTensor)> out{};
  std::size_t n = 0;
  for (auto rule : kTensors) {
    if (!GgufTensor(rule)) continue;
    rule.any_part = false;
    rule.optional = false;
    out[n++] = rule;
  }
  return out;
}();
static_assert(kGgufTensors.size() == 47);
constexpr auto kGgufParams = [] {
  constexpr auto count =
      std::ranges::count_if(kParams, [](const auto& p) { return !p.key.starts_with("llmp."); });
  std::array<ParamRule, count> out{};
  std::size_t n = 0;
  for (const auto& p : kParams)
    if (!p.key.starts_with("llmp.")) out[n++] = p;
  return out;
}();
constexpr auto kGgufMapping = [] {
  auto m = kMapping;
  m.containers = kGgufContainers;
  m.tensors = kGgufTensors;
  m.params = kGgufParams;
  m.folds = kFolds;
  m.policies = std::span(kPolicies).last(1);
  return m;
}();

// Native normalization binds a checkpoint's split parts to the same shared
// roles (MLX's affine experts and tables), and the importer's relaid-out
// NVFP4 experts (import/expert_relayout.h): the scale-atom gate-up and
// down tables, `.weight` and `.weight_scale`, beside their tensor scales'
// F32 [experts] vectors (the converted form's roles). A rule binds one role
// and part, so a down table's split and scale-atom forms share a pattern.
constexpr auto kNativeTensors = [] {
  std::array<TensorRule, kTensors.size() + 10> rules{};
  std::size_t next = 0;
  for (auto rule : kTensors) {
    if (rule.pattern == "blk.{layer}.ffn_gate_up_exps.codes" ||
        rule.pattern == "blk.{layer}.ffn_gate_up_exps.scales" ||
        rule.pattern == "blk.{layer}.ffn_down_exps.codes" ||
        rule.pattern == "blk.{layer}.ffn_down_exps.scales")
      continue;
    if (rule.pattern == "blk.{layer}.ffn_gate_exps.weight" ||
        rule.pattern == "blk.{layer}.ffn_up_exps.weight" ||
        rule.pattern == "blk.{layer}.ffn_down_exps.weight") {
      rule.part = PartRole::kCodes;
      rule.any_part = false;
    }
    // A table's codes (split) or its rows of codes and scales (interleaved).
    if (rule.pattern == "token_embd.weight" || rule.pattern == "per_layer_token_embd.weight") {
      rule.any_part = true;
    }
    rules[next++] = rule;
  }
  for (const auto& [codes, scales, role] : std::array<std::array<std::string_view, 3>, 1>{
           {{"blk.{layer}.ffn_gate_up_exps.weight", "blk.{layer}.ffn_gate_up_exps.weight_scale",
             "gate_up_experts"}}}) {
    rules[next++] = TensorRule{.pattern = codes,
                               .slot = "moe",
                               .role = role,
                               .part = PartRole::kCodes,
                               .experts = true,
                               .optional = true};
    rules[next++] = TensorRule{.pattern = scales,
                               .slot = "moe",
                               .role = role,
                               .part = PartRole::kBlockScales,
                               .experts = true,
                               .optional = true};
  }
  rules[next++] = Scales("token_embd.weight_scale", "embedding", "table");
  rules[next++] = Biases("token_embd.weight_bias", "embedding", "table");
  rules[next++] = Scales("output.weight_scale", "output", "weight");
  rules[next++] = Biases("output.weight_bias", "output", "weight");
  rules[next++] = Scales("per_layer_token_embd.weight_scale", "ngram", "table");
  rules[next++] = Biases("per_layer_token_embd.weight_bias", "ngram", "table");
  for (const auto& [pattern, role] : std::array<std::pair<std::string_view, std::string_view>, 3>{
           {{"blk.{layer}.ffn_gate_exps.weight_scale", "gate_experts"},
            {"blk.{layer}.ffn_up_exps.weight_scale", "up_experts"},
            {"blk.{layer}.ffn_down_exps.weight_scale", "down_experts"}}}) {
    auto rule = Scales(pattern, "moe", role);
    rule.experts = true;
    rules[next++] = rule;
  }
  for (const auto& [pattern, role] : std::array<std::pair<std::string_view, std::string_view>, 3>{
           {{"blk.{layer}.ffn_gate_exps.weight_bias", "gate_experts"},
            {"blk.{layer}.ffn_up_exps.weight_bias", "up_experts"},
            {"blk.{layer}.ffn_down_exps.weight_bias", "down_experts"}}}) {
    auto rule = Biases(pattern, "moe", role);
    rule.experts = true;
    rules[next++] = rule;
  }
  return rules;
}();
static_assert(kNativeTensors.size() == kTensors.size() + 10);
constexpr auto kNativeParams = [] {
  std::array<ParamRule, kParams.size() + 6> params{};
  std::ranges::copy(kParams, params.begin());
  std::size_t i = kParams.size();
  // The checkpoint's convention, unless the importer folded it (D-108 as
  // amended: the ModelOpt import stores 1 + w and -exp(A) in F32).
  for (std::string_view slot : {"attn_mix", "ffn_mix", "output_mix", "attention", "ple"}) {
    params[i] = Flag(slot, "norm_zero_centered", true);
    params[i++].unless_fold = "norm.plus_one";
  }
  params[i] = Flag("linear", "a_log", true);
  params[i].unless_fold = "ssm.a_neg_exp";
  return params;
}();
// The native import's fast forms. Routed experts whose checkpoint
// quantized their activations too (kActivationScales: W4A4) take the
// converted form's (kPolicies[0]): its moe_experts runs a block-scaled
// grouped GEMM where the experts are in the scale-atom layout and the
// device has FP4 tensor cores (kernels/ggml/capabilities.cc). Otherwise
// (W4A16, or split experts) the GGML-expert form's (kPolicies[1]) but its
// experts' fused GLU, which reads GGML's blocks: the format ladder's grouped
// products decode the experts; zero-centered norms and A that an import
// did not fold reach the fused patterns as derived parameters.
constexpr std::array kNativePolicies = {
    PolicyRule{.gate = kActivationScales, .text = kPolicies[0].text},
    PolicyRule{.text = "max_decode_rows 16\n"
                       "enable owner_row_invariant\n"},
};
constexpr auto kNativeMapping = [] {
  auto mapping = kMapping;
  mapping.containers = kNativeContainers;
  mapping.tensors = kNativeTensors;
  mapping.params = kNativeParams;
  mapping.folds = kNativeFolds;
  mapping.policies = kNativePolicies;
  return mapping;
}();

// ------------------------------------------------------------ qwen4exp-mtp

constexpr std::string_view kDraftRows = "qwen4exp-mtp.draft_head_rows";
// The rows of the draft head a pass reads (Qwen38Options::draft_vocab's
// default: 65,536, or a selected head's rows when fewer).
constexpr std::string_view kDraftVocab = "qwen4exp-mtp.draft_vocab";

constexpr std::array kMtpSlots = {
    SlotRule{.slot = "embedding", .site = Site::kEmbedding, .kind = "embedding.mtp"},
    SlotRule{.slot = "streams", .site = Site::kMixing, .kind = "residual.hc"},
    SlotRule{.slot = "attn_mix", .site = Site::kLayer, .kind = "hc.mix"},
    SlotRule{.slot = "attention", .site = Site::kLayer, .kind = "attention.qsa"},
    SlotRule{.slot = "attn_combine", .site = Site::kLayer, .kind = "hc.combine"},
    SlotRule{.slot = "ffn_mix", .site = Site::kLayer, .kind = "hc.mix"},
    SlotRule{.slot = "moe", .site = Site::kLayer, .kind = "ffn.moe"},
    SlotRule{.slot = "ffn_combine", .site = Site::kLayer, .kind = "hc.combine"},
    SlotRule{.slot = "output_mix", .site = Site::kFinalNorm, .kind = "hc.mix"},
    // Its own selected head where it has one, else its target's.
    SlotRule{.slot = "selected", .site = Site::kOutput, .kind = "head.linear", .gate = kDraftRows},
    SlotRule{.slot = "output",
             .site = Site::kOutput,
             .kind = "head.linear",
             .gate = kDraftRows,
             .gate_absent = true},
};

constexpr std::array kMtpTensors = {
    TensorRule{
        .pattern = "token_embd.weight", .slot = "embedding", .role = "table", .target = true},
    Plain("norm_embd.weight", "embedding", "norm_embd"),
    Matrix("fc_embd.weight", "embedding", "fc_embd"),
    Plain("norm_hidden.weight", "embedding", "norm_hidden"),
    Matrix("fc_hidden.weight", "embedding", "fc_hidden"),
    Plain("blk.{layer}.hc_attn_norm.weight", "attn_mix", "norm"),
    Matrix("blk.{layer}.hc_attn_down.weight", "attn_mix", "down"),
    Matrix("blk.{layer}.hc_attn_up.weight", "attn_mix", "up"),
    Matrix("blk.{layer}.hc_attn_inject.weight", "attn_combine", "inject"),
    Plain("blk.{layer}.hc_ffn_norm.weight", "ffn_mix", "norm"),
    Matrix("blk.{layer}.hc_ffn_down.weight", "ffn_mix", "down"),
    Matrix("blk.{layer}.hc_ffn_up.weight", "ffn_mix", "up"),
    Matrix("blk.{layer}.hc_ffn_inject.weight", "ffn_combine", "inject"),
    Matrix("blk.{layer}.attn_q.weight", "attention", "q"),
    Matrix("blk.{layer}.attn_k.weight", "attention", "k"),
    Matrix("blk.{layer}.attn_v.weight", "attention", "v"),
    Matrix("blk.{layer}.attn_output.weight", "attention", "output"),
    Plain("blk.{layer}.attn_q_norm.weight", "attention", "q_norm"),
    Plain("blk.{layer}.attn_k_norm.weight", "attention", "k_norm"),
    Matrix("blk.{layer}.indexer.qk_proj.weight", "attention", "indexer_qk"),
    Plain("blk.{layer}.indexer.q_norm.weight", "attention", "indexer_q_norm"),
    Plain("blk.{layer}.indexer.k_norm.weight", "attention", "indexer_k_norm"),
    Matrix("blk.{layer}.ffn_gate_inp.weight", "moe", "router"),
    Matrix("blk.{layer}.ffn_gate_inp_shexp.weight", "moe", "shared_gate"),
    Matrix("blk.{layer}.ffn_gate_shexp.weight", "moe", "gate"),
    Matrix("blk.{layer}.ffn_up_shexp.weight", "moe", "up"),
    Matrix("blk.{layer}.ffn_down_shexp.weight", "moe", "down"),
    TensorRule{.pattern = "blk.{layer}.ffn_gate_up_exps.codes",
               .slot = "moe",
               .role = "gate_up_experts",
               .part = PartRole::kCodes,
               .experts = true},
    TensorRule{.pattern = "blk.{layer}.ffn_gate_up_exps.scales",
               .slot = "moe",
               .role = "gate_up_experts",
               .part = PartRole::kBlockScales,
               .experts = true},
    TensorRule{.pattern = "blk.{layer}.ffn_down_exps.codes",
               .slot = "moe",
               .role = "down_experts",
               .part = PartRole::kCodes,
               .experts = true},
    TensorRule{.pattern = "blk.{layer}.ffn_down_exps.scales",
               .slot = "moe",
               .role = "down_experts",
               .part = PartRole::kBlockScales,
               .experts = true},
    Plain("blk.{layer}.ffn_gate_exps.weight_scale_2", "moe", "gate_scale"),
    Plain("blk.{layer}.ffn_up_exps.weight_scale_2", "moe", "up_scale"),
    Plain("blk.{layer}.ffn_down_exps.weight_scale_2", "moe", "down_scale"),
    Plain("output_hc_norm.weight", "output_mix", "norm"),
    Matrix("output_hc_down.weight", "output_mix", "down"),
    Matrix("output_hc_up.weight", "output_mix", "up"),
    Matrix("draft_output.weight", "selected", "weight"),
    Plain("draft_output.ids", "selected", "ids"),
    TensorRule{.pattern = "output.weight", .slot = "output", .role = "weight", .target = true},
};

constexpr std::array kMtpParams = {
    LLMP_QWEN38_BLOCK_PARAMS("qwen4exp-mtp."),
    FromKey("embedding", "eps", "qwen4exp-mtp.attention.layer_norm_rms_epsilon", kReal),
    Name("embedding", "tap", "streams"),
    // Its own caches, rewritten by its catch-up: truncated.
    Name("attention", "rollback", "truncate"),
    Flag("output_mix", "features", true),
    Flag("selected", "selected", true),
    // The draft head reads the first rows of its weight (the lowest token
    // IDs, or a selected head's first): the draft vocabulary.
    FromKey("selected", "rows", kDraftVocab, kInt, Transform::kNone, {}, true),
    FromKey("output", "rows", kDraftVocab, kInt, Transform::kNone, {}, true),
    // Its drafts as GGML's argmax picks them (the fast form's).
    Flag("selected", "native_argmax", true),
    Flag("output", "native_argmax", true),
};

// The drafter's: its BF16 linears' inputs rounded once past 16 rows, its
// QSA layer's and MoE's fast forms (its experts in the CUTLASS layout).
constexpr std::array kMtpPolicies = {
    PolicyRule{.text = "max_decode_rows 16\n"
                       "enable owner_row_invariant\n"},
};

constexpr std::array kMtpReads = {
    ReadRule{.kind = ReadKind::kTap, .name = "streams", .point = TapPoint::kStreams},
};

// Autoregressive MTP passes, composed in their replay form; a selected
// head's token map and the draft vocabulary's value.
constexpr std::string_view kHeadIds = "draft_output.ids";
constexpr DraftRecipe kMtpDraft{.shape = DraftRecipe::Shape::kMtp,
                                .composed_version = 2,
                                .target_architecture = "qwen4exp",
                                .head_ids = kHeadIds,
                                .vocab_key = kDraftVocab};

constexpr Mapping kMtpMapping{
    .architecture = "qwen4exp-mtp",
    .version = 1,
    .pipeline = PipelineKind::kLlm,
    .layer_count_key = "qwen4exp-mtp.block_count",
    .layer_types = "f",
    .slots = kMtpSlots,
    .tensors = kMtpTensors,
    .params = kMtpParams,
    .folds = kFolds,
    .role = StageRole::kDrafter,
    .stage = "mtp",
    .reads = kMtpReads,
    .policies = kMtpPolicies,
    .draft = &kMtpDraft,
};

constexpr auto kMtpReplaySlots = [] {
  auto slots = kMtpSlots;
  slots[0].version = 2;
  return slots;
}();
constexpr auto kMtpReplayParams = [] {
  std::array<ParamRule, kMtpParams.size() + 2> params{};
  std::ranges::copy(kMtpParams, params.begin());
  params[kMtpParams.size()] = Int("embedding", "token_offset", 1);
  params[kMtpParams.size() + 1] = Flag("attention", "replayable", true);
  return params;
}();
constexpr auto kMtpReplayMapping = [] {
  auto mapping = kMtpMapping;
  mapping.version = 2;
  mapping.slots = kMtpReplaySlots;
  mapping.params = kMtpReplayParams;
  return mapping;
}();

// The native importer's drafter (import/modelopt_qwen38.h, D-109): the
// checkpoint's MTP tensors in their source encodings (its BF16 linears)
// but its norms' 1 + w folded in F32 (norm.plus_one, chosen by role), its
// routed experts relaid out as the native target's (scale-atom gate-up and
// down tables, `.weight` and `.weight_scale`, and their tensor scales' F32
// vectors), and the optional selected head the importer gathered from the
// target's head rows. The mapping names no fold of its own: the source
// records the importer's (Source::folds).
constexpr auto kMtpNativeTensors = [] {
  auto rules = kMtpTensors;
  for (auto& rule : rules) {
    if (rule.pattern == "blk.{layer}.ffn_gate_up_exps.codes")
      rule.pattern = "blk.{layer}.ffn_gate_up_exps.weight";
    if (rule.pattern == "blk.{layer}.ffn_gate_up_exps.scales")
      rule.pattern = "blk.{layer}.ffn_gate_up_exps.weight_scale";
    if (rule.pattern == "blk.{layer}.ffn_down_exps.codes")
      rule.pattern = "blk.{layer}.ffn_down_exps.weight";
    if (rule.pattern == "blk.{layer}.ffn_down_exps.scales")
      rule.pattern = "blk.{layer}.ffn_down_exps.weight_scale";
  }
  return rules;
}();
// Its norms zero-centered, as the checkpoint holds them, unless the
// importer stored 1 + w in F32 (norm.plus_one, as the converted drafter's).
template <std::size_t N>
constexpr auto ZeroCentered(const std::array<ParamRule, N>& base) {
  std::array<ParamRule, N + 5> params{};
  std::ranges::copy(base, params.begin());
  std::size_t i = N;
  for (std::string_view slot : {"embedding", "attn_mix", "ffn_mix", "output_mix", "attention"}) {
    params[i] = Flag(slot, "norm_zero_centered", true);
    params[i++].unless_fold = "norm.plus_one";
  }
  return params;
}
constexpr auto kMtpNativeParams = ZeroCentered(kMtpParams);
constexpr auto kMtpNativeReplayParams = ZeroCentered(kMtpReplayParams);
// Missing kept metadata in published native MTP artifacts is reconstructed
// from the actual target spec and this artifact's own selected-head tensor.
constexpr std::array kMtpValues = {
    ValueRule{.key = "qwen4exp-mtp.context_length", .origin = ValueOrigin::kContext},
    ValueRule{.key = "qwen4exp-mtp.embedding_length", .origin = ValueOrigin::kWidth},
    ValueRule{.key = "qwen4exp-mtp.hyper_connection.count", .origin = ValueOrigin::kStreams},
    ValueRule{.key = "qwen4exp-mtp.hyper_connection.low_rank",
              .origin = ValueOrigin::kComponent,
              .kind = "hc.mix",
              .param = "rank"},
    ValueRule{.key = "qwen4exp-mtp.attention.layer_norm_rms_epsilon",
              .origin = ValueOrigin::kComponent,
              .kind = "attention.qsa",
              .param = "eps"},
    ValueRule{.key = "qwen4exp-mtp.attention.head_count",
              .origin = ValueOrigin::kComponent,
              .kind = "attention.qsa",
              .param = "heads"},
    ValueRule{.key = "qwen4exp-mtp.attention.head_count_kv",
              .origin = ValueOrigin::kComponent,
              .kind = "attention.qsa",
              .param = "kv_heads"},
    ValueRule{.key = "qwen4exp-mtp.attention.key_length",
              .origin = ValueOrigin::kComponent,
              .kind = "attention.qsa",
              .param = "head_dim"},
    ValueRule{.key = "qwen4exp-mtp.attention.value_length",
              .origin = ValueOrigin::kComponent,
              .kind = "attention.qsa",
              .param = "head_dim"},
    ValueRule{.key = "qwen4exp-mtp.rope.dimension_count",
              .origin = ValueOrigin::kComponent,
              .kind = "attention.qsa",
              .param = "rope_dims"},
    ValueRule{.key = "qwen4exp-mtp.rope.freq_base",
              .origin = ValueOrigin::kComponent,
              .kind = "attention.qsa",
              .param = "rope_base"},
    ValueRule{.key = "qwen4exp-mtp.rope.dimension_sections",
              .origin = ValueOrigin::kComponent,
              .kind = "attention.qsa",
              .param = "rope_sections"},
    ValueRule{.key = "qwen4exp-mtp.attention.indexer.head_count",
              .origin = ValueOrigin::kComponent,
              .kind = "attention.qsa",
              .param = "indexer_heads"},
    ValueRule{.key = "qwen4exp-mtp.attention.indexer.key_length",
              .origin = ValueOrigin::kComponent,
              .kind = "attention.qsa",
              .param = "indexer_dim"},
    ValueRule{.key = "qwen4exp-mtp.attention.indexer.top_k",
              .origin = ValueOrigin::kComponent,
              .kind = "attention.qsa",
              .param = "budget"},
    ValueRule{.key = "qwen4exp-mtp.attention.compress_ratios",
              .origin = ValueOrigin::kComponent,
              .kind = "attention.qsa",
              .param = "ratio",
              .list = true},
    ValueRule{.key = "qwen4exp-mtp.expert_count",
              .origin = ValueOrigin::kComponent,
              .kind = "ffn.moe",
              .param = "experts"},
    ValueRule{.key = "qwen4exp-mtp.expert_used_count",
              .origin = ValueOrigin::kComponent,
              .kind = "ffn.moe",
              .param = "used"},
    ValueRule{.key = "qwen4exp-mtp.expert_feed_forward_length",
              .origin = ValueOrigin::kComponent,
              .kind = "ffn.moe",
              .param = "expert_hidden"},
    ValueRule{.key = "qwen4exp-mtp.expert_shared_feed_forward_length",
              .origin = ValueOrigin::kComponent,
              .kind = "ffn.moe",
              .param = "hidden"},
    ValueRule{.key = "qwen4exp-mtp.block_count",
              .origin = ValueOrigin::kLayerCount,
              .tensor = "blk.{layer}.attn_q.weight"},
    ValueRule{.key = kIndexer, .value = {.text = "fused"}, .type = kString},
    ValueRule{.key = kStacked, .value = {.flag = true}, .type = kBool},
    ValueRule{.key = kScaled, .value = {.flag = true}, .type = kBool},
    ValueRule{.key = kDraftRows,
              .origin = ValueOrigin::kTensor,
              .tensor = "draft_output.weight",
              .dimension = 1,
              .optional = true},
    ValueRule{.key = kDraftVocab,
              .origin = ValueOrigin::kTensor,
              .tensor = "draft_output.weight",
              .dimension = 1,
              .optional = true,
              .at_most = kServing.default_draft_vocab},
    ValueRule{.key = kDraftVocab,
              .origin = ValueOrigin::kVocabulary,
              .at_most = kServing.default_draft_vocab},
};
constexpr DraftRecipe kMtpNativeDraft{.shape = DraftRecipe::Shape::kMtp,
                                      .composed_version = 2,
                                      .target_architecture = "qwen4exp",
                                      .values = kMtpValues,
                                      .head_ids = kHeadIds,
                                      .vocab_key = kDraftVocab};
// A speculative stage's routed experts take the W4A4 grouped GEMM (its
// moe_experts: activations quantized to FP4 with a per-step dynamic scale,
// row amax / (6 · 448), as v0's did) whatever its checkpoint's activation
// quantization: the target's verification decides every token, so the
// drafter's numerics change only acceptance and speed (AGENTS.md rule 11,
// speed over bit-exactness at the same quality), and on spark-b it measured
// 1.2 to 2.0% faster decode than W4A16 at equal acceptance
// (experiments/m36-numerical-audit/qwen-modelopt-v1.md). The converted
// drafter's policy (kMtpMapping's, which the mapping below keeps); a device
// without FP4 falls back to the format ladder.
constexpr auto kMtpNativeMapping = [] {
  auto mapping = kMtpMapping;
  mapping.containers = kNativeContainers;
  mapping.draft = &kMtpNativeDraft;
  mapping.tensors = kMtpNativeTensors;
  mapping.params = kMtpNativeParams;
  mapping.folds = kNativeFolds;
  return mapping;
}();
constexpr auto kMtpNativeReplayMapping = [] {
  auto mapping = kMtpReplayMapping;
  mapping.containers = kNativeContainers;
  mapping.draft = &kMtpNativeDraft;
  mapping.tensors = kMtpNativeTensors;
  mapping.params = kMtpNativeReplayParams;
  mapping.folds = kNativeFolds;
  return mapping;
}();

#undef LLMP_QWEN38_BLOCK_PARAMS

}  // namespace

const Mapping& Qwen38Mapping() { return kMapping; }
const Mapping& Qwen38GgufMapping() { return kGgufMapping; }
const Mapping& Qwen38NativeMapping() { return kNativeMapping; }
const Mapping& Qwen38MtpMapping() { return kMtpMapping; }
const Mapping& Qwen38MtpReplayMapping() { return kMtpReplayMapping; }
const Mapping& Qwen38MtpNativeMapping() { return kMtpNativeMapping; }
const Mapping& Qwen38MtpNativeReplayMapping() { return kMtpNativeReplayMapping; }

}  // namespace llmp::spec::mappings
