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
constexpr std::string_view kPleLayers = "ple.layers";
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

constexpr std::string_view kEps = "attention.layer_norm_rms_epsilon";

// The QSA, MoE and mix parameters both mappings read.
#define LLMP_QWEN38_BLOCK_PARAMS                                                                \
  Name("streams", "mix", "sigmoid"),                                                            \
      FromKey("attn_mix", "eps", "attention.layer_norm_rms_epsilon", kReal),                    \
      FromKey("attn_mix", "rank", "hyper_connection.low_rank", kInt),                           \
      Flag("attn_mix", "inject", true),                                                         \
      FromKey("ffn_mix", "eps", "attention.layer_norm_rms_epsilon", kReal),                     \
      FromKey("ffn_mix", "rank", "hyper_connection.low_rank", kInt),                            \
      Flag("ffn_mix", "inject", true),                                                          \
      FromKey("output_mix", "eps", "attention.layer_norm_rms_epsilon", kReal),                  \
      FromKey("output_mix", "rank", "hyper_connection.low_rank", kInt),                         \
      Flag("output_mix", "inject", false),                                                      \
      FromKey("attention", "eps", "attention.layer_norm_rms_epsilon", kReal),                   \
      FromKey("attention", "heads", "attention.head_count", kInt),                              \
      FromKey("attention", "kv_heads", "attention.head_count_kv", kInt),                        \
      FromKey("attention", "head_dim", "attention.key_length", kInt),                           \
      FromKey("attention", "rope_dims", "rope.dimension_count", kInt),                          \
      FromKey("attention", "rope_base", "rope.freq_base", kReal),                               \
      FromKey("attention", "rope_sections", "rope.dimension_sections", kInts),                  \
      FromKey("attention", "indexer_heads", "attention.indexer.head_count", kInt),              \
      FromKey("attention", "indexer_dim", "attention.indexer.key_length", kInt),                \
      PerLayer("attention", "ratio", "attention.compress_ratios", kInt),                        \
      FromKey("attention", "budget", "attention.indexer.top_k", kInt),                          \
      FromKey("attention", "indexer", kIndexer, kString, Transform::kNone, {}, true),           \
      Name("attention", "indexer", "split"), Name("moe", "placement", "none"),                  \
      FromKey("moe", "eps", "attention.layer_norm_rms_epsilon", kReal),                         \
      Name("moe", "activation", "silu"),                                                        \
      FromKey("moe", "hidden", "expert_shared_feed_forward_length", kInt),                      \
      FromKey("moe", "experts", "expert_count", kInt),                                          \
      FromKey("moe", "used", "expert_used_count", kInt),                                        \
      FromKey("moe", "expert_hidden", "expert_feed_forward_length", kInt),                      \
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
      FromKey(kStageSlot, "context", "context_length", kInt),                                   \
      FromKey(kStreamSlot, "width", "embedding_length", kInt),                                  \
      FromKey(kStreamSlot, "streams", "hyper_connection.count", kInt)

constexpr std::array kParams = {
    LLMP_QWEN38_BLOCK_PARAMS,
    Real("embedding", "scale", 1.0),
    // A verify's tokens as GGML's argmax picks them (the fast form's).
    Flag("output", "native_argmax", true),
    FromKey("ngram", "ngram", "ple.ngram_size", kInt),
    FromKey("ngram", "heads_per_ngram", "ple.heads_per_ngram", kInt),
    FromKey("ngram", "row", "embedding_length_per_layer_input", kInt),
    FromKey("ngram", "eos", "ple.eos_token_id", kInt),
    FromKey("ngram", "multipliers", "ple.layer_multipliers", kInts),
    FromKey("ngram", "offsets", "ple.head_offsets", kInts),
    FromKey("ngram", "vocab", "ple.head_vocab_sizes", kInts),
    FromKey("ple", "eps", kEps, kReal),
    Name("ple", "input", "ngram"),
    FromKey("ple", "conv", "ple.conv_kernel", kInt),
    FromKey("ple", "dilation", "ple.ngram_size", kInt),
    FromKey("linear", "eps", kEps, kReal),
    FromKey("linear", "k_heads", "ssm.group_count", kInt),
    FromKey("linear", "v_heads", "ssm.time_step_rank", kInt),
    FromKey("linear", "head_dim", "ssm.state_size", kInt),
    FromKey("linear", "conv", "ssm.conv_kernel", kInt),
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
    .layer_count_key = "block_count",
    .layer_types = "lllf",
    .layer_type_interval_key = "full_attention_interval",
    .slots = kSlots,
    .tensors = kTensors,
    .params = kParams,
    .folds = kFolds,
    .layer_flags_key = "attention.compress_ratios",
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
constexpr std::array kQwenRequirements{
    ScalarRequirement{.key = "attention.value_length", .equals_key = "attention.key_length"}};
constexpr std::array kQwenAuxiliary{
    kGgufAuxiliary[0], kGgufAuxiliary[1], kGgufAuxiliary[2],
    AuxiliaryNamespace{"qwen4exp.ssm.inner_size",
                       "linear value width stated by its output projection", "blk.0.ssm_out.weight",
                       0},
    // The text-only PLE hashes use eos_token_id. This marks an image input
    // placeholder in the checkpoint; no image input path is admitted here.
    AuxiliaryNamespace{"qwen4exp.ple.image_token_id",
                       "image input placeholder outside the text pipeline"}};
constexpr std::array kQwenLists{
    ListRequirement{.key = "ple.layers", .size = kLayerLimit, .min_size = 1},
    ListRequirement{
        .key = "rope.dimension_sections", .size = 4, .sections_of = "rope.dimension_count"},
    ListRequirement{
        .key = "attention.compress_ratios", .size = kLayerLimit, .min_size = 1, .per_layer = true},
    ListRequirement{.key = "ple.layer_multipliers", .size = kLayerLimit, .min_size = 1},
    ListRequirement{.key = "ple.head_offsets", .size = kLayerLimit, .min_size = 1},
    ListRequirement{.key = "ple.head_vocab_sizes", .size = kLayerLimit, .min_size = 1}};
constexpr std::array kQwenRows{
    RowIndexRequirement{"ple.head_offsets", "ple.head_vocab_sizes", "per_layer_token_embd.weight"}};
constexpr auto kGgufMapping = [] {
  auto m = kMapping;
  m.containers = kGgufContainers;
  m.tensors = kGgufTensors;
  m.params = kGgufParams;
  m.folds = kFolds;
  m.policies = std::span(kPolicies).last(1);
  m.auxiliary = kQwenAuxiliary;
  m.requirements = kQwenRequirements;
  m.lists = kQwenLists;
  m.extended_components = true;
  m.row_indices = kQwenRows;
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
      rule.any_part = true;
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
                               .optional = true,
                               .any_part = true};
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
  std::array<ParamRule, kParams.size() + 23> params{};
  std::size_t i = 0;
  for (auto rule : kParams) {
    if (rule.slot == "moe" && rule.param == "projection_scales") continue;
    if (rule.slot == "moe" && !rule.key.empty() &&
        (rule.param == "stacked" || rule.param == "projection_scales")) {
      rule.tensor = rule.param == "stacked" ? "blk.{layer}.ffn_gate_up_exps.weight"
                                            : "blk.{layer}.ffn_gate_exps.weight_scale_2";
      rule.key = {};
    }
    if (rule.param == "eps") {
      rule.transform = Transform::kFloat32;
    } else if (rule.slot == "attention" && rule.param == "rope_dims") {
      rule.numerator_key = "attention.key_length";
      rule.divisor_key = "rope.partial_rotary_factor";
      rule.transform = Transform::kProduct;
    } else if (rule.slot == "attention" && rule.param == "indexer" && rule.key.empty()) {
      rule.value.text = "fused";
    } else if (rule.slot == "ngram" && rule.param == "row") {
      rule.optional = true;
      params[i++] = rule;
      params[i++] = ParamRule{
          .slot = "ngram", .param = "row", .tensor = "per_layer_token_embd.weight", .type = kInt};
      continue;
    } else if (rule.slot.empty() && rule.key == "full_attention_interval") {
      continue;
    }
    params[i++] = rule;
  }
  params[i++] = Flag("moe", "projection_scales", false);
  params[i++] = FromTensor("moe", "gate_scale", "blk.{layer}.ffn_gate_exps.weight_scale_2");
  params[i++] = Flag("moe", "gate_scale", false);
  params[i++] = FromTensor("moe", "up_scale", "blk.{layer}.ffn_up_exps.weight_scale_2");
  params[i++] = Flag("moe", "up_scale", false);
  params[i++] = FromTensor("moe", "down_scale", "blk.{layer}.ffn_down_exps.weight_scale_2");
  params[i++] = Flag("moe", "down_scale", false);
  // The checkpoint's convention, unless the importer folded it (D-108 as
  // amended: the ModelOpt import stores 1 + w and -exp(A) in F32).
  for (std::string_view slot : {"attn_mix", "ffn_mix", "output_mix", "attention", "ple"}) {
    params[i] = Flag(slot, "norm_zero_centered", true);
    params[i++].unless_fold = "norm.plus_one";
  }
  params[i] = Flag("linear", "a_log", true);
  params[i++].unless_fold = "ssm.a_neg_exp";
  params[i++] = Require("mamba_ssm_dtype", "float32", true);
  params[i++] = Require("dtype", "bfloat16", true);
  params[i++] = Require("hidden_act", "silu", true);
  params[i++] = Require("output_gate_type", "sigmoid", true);
  params[i++] = Require("rope.scaling.type", "default", true);
  params[i++] = ParamRule{
      .key = "is_encoder_decoder", .type = kBool, .value = {.flag = false}, .optional = true};
  params[i++] =
      ParamRule{.key = "attention_bias", .type = kBool, .value = {.flag = false}, .optional = true};
  params[i++] =
      FromKey("ngram", "declared_width", "ple.embed_dim", kInt, Transform::kNone, {}, true);
  params[i] = FromKey("ple", "input_width", "ple.embed_dim", kInt, Transform::kNone, {}, true);
  params[++i] = RequireInt("attention.indexer.head_count_kv", 1, true);
  params[++i] = ParamRule{.key = "use_bidirectional_attention",
                          .type = kBool,
                          .value = {.flag = false},
                          .optional = true};
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
constexpr std::array<std::string_view, 2> kNativeSourceArchitectures = {
    "qwen3_8_flash_next",
    "qwen4_exp",
};
constexpr std::array kNativeLayerTypeCodes = {
    LayerTypeCode{0, 'l', "linear_attention"},
    LayerTypeCode{1, 'f', "full_attention"},
};
constexpr std::array kNativeSemanticTensors = {
    SemanticTensorRule{
        .tensor = "model.language_model.layers.{layer}.ple.ple_embedding.layer_multipliers",
        .key = "ple.layer_multipliers",
        .size = 0},
    SemanticTensorRule{
        .tensor = "language_model.model.layers.{layer}.ple.ple_embedding.layer_multipliers",
        .key = "ple.layer_multipliers",
        .size = 0},
    SemanticTensorRule{.tensor = "model.layers.{layer}.self_attn.ple_embed.layer_multipliers",
                       .key = "ple.layer_multipliers",
                       .size = 0},
    SemanticTensorRule{
        .tensor = "model.language_model.layers.{layer}.ple.ple_embedding.ngram_heads_offsets",
        .key = "ple.head_offsets",
        .size = 0},
    SemanticTensorRule{
        .tensor = "language_model.model.layers.{layer}.ple.ple_embedding.ngram_heads_offsets",
        .key = "ple.head_offsets",
        .size = 0},
    SemanticTensorRule{.tensor = "model.layers.{layer}.self_attn.ple_embed.head_offsets",
                       .key = "ple.head_offsets",
                       .size = 0},
    SemanticTensorRule{
        .tensor = "model.language_model.layers.{layer}.ple.ple_embedding.ngram_heads_vocab_sizes",
        .key = "ple.head_vocab_sizes",
        .size = 0},
    SemanticTensorRule{
        .tensor = "language_model.model.layers.{layer}.ple.ple_embedding.ngram_heads_vocab_sizes",
        .key = "ple.head_vocab_sizes",
        .size = 0},
    SemanticTensorRule{.tensor = "model.layers.{layer}.self_attn.ple_embed.head_vocab_sizes",
                       .key = "ple.head_vocab_sizes",
                       .size = 0},
};
constexpr std::array<std::string_view, 5> kNativeIgnoredPrefixes = {
    "model.visual.", "vision_tower.", "visual.", "mtp.", "language_model.mtp.",
};
constexpr std::array<std::string_view, 3> kNativeIgnoredTensors = {
    "model.language_model.layers.{layer}.mlp.experts.{expert}.down_proj.input_scale",
    "model.language_model.layers.{layer}.mlp.experts.{expert}.gate_proj.input_scale",
    "model.language_model.layers.{layer}.mlp.experts.{expert}.up_proj.input_scale",
};
constexpr std::array kNativeKeyAliases = {
    KeyAlias{"text_config.linear_value_head_dim", "ssm.value_head_dim"},
    KeyAlias{"linear_value_head_dim", "ssm.value_head_dim"},
    KeyAlias{"text_config.indexer_kv_heads", "attention.indexer.head_count_kv"},
    KeyAlias{"indexer_kv_heads", "attention.indexer.head_count_kv"},
    KeyAlias{"text_config.rope_parameters.mrope_interleaved", "rope.mrope_interleaved"},
    KeyAlias{"rope_parameters.mrope_interleaved", "rope.mrope_interleaved"},
    KeyAlias{"text_config.use_bidirectional_attention", "use_bidirectional_attention"},
    KeyAlias{"text_config.num_hidden_layers", "block_count"},
    KeyAlias{"num_hidden_layers", "block_count"},
    KeyAlias{"text_config.max_position_embeddings", "context_length"},
    KeyAlias{"max_position_embeddings", "context_length"},
    KeyAlias{"text_config.hidden_size", "embedding_length"},
    KeyAlias{"hidden_size", "embedding_length"},
    KeyAlias{"text_config.hc_count", "hyper_connection.count"},
    KeyAlias{"hc_count", "hyper_connection.count"},
    KeyAlias{"text_config.hc_lowrank", "hyper_connection.low_rank"},
    KeyAlias{"hc_lowrank", "hyper_connection.low_rank"},
    KeyAlias{"text_config.rms_norm_eps", "attention.layer_norm_rms_epsilon"},
    KeyAlias{"rms_norm_eps", "attention.layer_norm_rms_epsilon"},
    KeyAlias{"text_config.num_attention_heads", "attention.head_count"},
    KeyAlias{"num_attention_heads", "attention.head_count"},
    KeyAlias{"text_config.num_key_value_heads", "attention.head_count_kv"},
    KeyAlias{"num_key_value_heads", "attention.head_count_kv"},
    KeyAlias{"text_config.head_dim", "attention.key_length"},
    KeyAlias{"head_dim", "attention.key_length"},
    KeyAlias{"text_config.rope_parameters.rope_theta", "rope.freq_base"},
    KeyAlias{"rope_parameters.rope_theta", "rope.freq_base"},
    KeyAlias{"rope_theta", "rope.freq_base"},
    KeyAlias{"text_config.rope_parameters.mrope_section", "rope.dimension_sections"},
    KeyAlias{"rope_parameters.mrope_section", "rope.dimension_sections"},
    KeyAlias{"mrope_section", "rope.dimension_sections"},
    KeyAlias{"text_config.rope_parameters.partial_rotary_factor", "rope.partial_rotary_factor"},
    KeyAlias{"rope_parameters.partial_rotary_factor", "rope.partial_rotary_factor"},
    KeyAlias{"text_config.partial_rotary_factor", "rope.partial_rotary_factor"},
    KeyAlias{"partial_rotary_factor", "rope.partial_rotary_factor"},
    KeyAlias{"text_config.indexer_n_heads", "attention.indexer.head_count"},
    KeyAlias{"indexer_n_heads", "attention.indexer.head_count"},
    KeyAlias{"text_config.indexer_head_dim", "attention.indexer.key_length"},
    KeyAlias{"indexer_head_dim", "attention.indexer.key_length"},
    KeyAlias{"text_config.indexer_budget", "attention.indexer.top_k"},
    KeyAlias{"indexer_budget", "attention.indexer.top_k"},
    KeyAlias{"text_config.indexer_compress_ratio", "attention.compress_ratios"},
    KeyAlias{"indexer_compress_ratio", "attention.compress_ratios"},
    KeyAlias{"text_config.num_experts", "expert_count"},
    KeyAlias{"num_experts", "expert_count"},
    KeyAlias{"text_config.num_experts_per_tok", "expert_used_count"},
    KeyAlias{"num_experts_per_tok", "expert_used_count"},
    KeyAlias{"text_config.moe_intermediate_size", "expert_feed_forward_length"},
    KeyAlias{"moe_intermediate_size", "expert_feed_forward_length"},
    KeyAlias{"text_config.shared_expert_intermediate_size", "expert_shared_feed_forward_length"},
    KeyAlias{"shared_expert_intermediate_size", "expert_shared_feed_forward_length"},
    KeyAlias{"text_config.linear_conv_kernel_dim", "ssm.conv_kernel"},
    KeyAlias{"linear_conv_kernel_dim", "ssm.conv_kernel"},
    KeyAlias{"text_config.linear_num_key_heads", "ssm.group_count"},
    KeyAlias{"linear_num_key_heads", "ssm.group_count"},
    KeyAlias{"text_config.linear_num_value_heads", "ssm.time_step_rank"},
    KeyAlias{"linear_num_value_heads", "ssm.time_step_rank"},
    KeyAlias{"text_config.linear_key_head_dim", "ssm.state_size"},
    KeyAlias{"linear_key_head_dim", "ssm.state_size"},
    KeyAlias{"text_config.ngram_size", "ple.ngram_size"},
    KeyAlias{"ngram_size", "ple.ngram_size"},
    KeyAlias{"text_config.heads_per_ngram", "ple.heads_per_ngram"},
    KeyAlias{"heads_per_ngram", "ple.heads_per_ngram"},
    KeyAlias{"text_config.ple_conv_kernel_size", "ple.conv_kernel"},
    KeyAlias{"ple_conv_kernel_size", "ple.conv_kernel"},
    KeyAlias{"text_config.ple_embed_dim", "ple.embed_dim"},
    KeyAlias{"ple_embed_dim", "ple.embed_dim"},
    KeyAlias{"text_config.eos_token_id", "ple.eos_token_id"},
    KeyAlias{"eos_token_id", "ple.eos_token_id", "text_config.eos_token_id"},
    KeyAlias{"text_config.ple_layer_ids", "ple.layer_ids"},
    KeyAlias{"ple_layer_ids", "ple.layer_ids"},
    KeyAlias{"text_config.layer_types", "layer_types"},
    KeyAlias{"layer_types", "layer_types"},
    KeyAlias{"text_config.full_attention_interval", "full_attention_interval"},
    KeyAlias{"full_attention_interval", "full_attention_interval"},
    KeyAlias{"text_config.mamba_ssm_dtype", "mamba_ssm_dtype"},
    KeyAlias{"mamba_ssm_dtype", "mamba_ssm_dtype"},
    KeyAlias{"text_config.dtype", "dtype"},
    KeyAlias{"dtype", "dtype"},
    KeyAlias{"text_config.is_encoder_decoder", "is_encoder_decoder"},
    KeyAlias{"is_encoder_decoder", "is_encoder_decoder"},
    KeyAlias{"text_config.hidden_act", "hidden_act"},
    KeyAlias{"hidden_act", "hidden_act"},
    KeyAlias{"text_config.attention_bias", "attention_bias"},
    KeyAlias{"attention_bias", "attention_bias"},
    KeyAlias{"text_config.output_gate_type", "output_gate_type"},
    KeyAlias{"output_gate_type", "output_gate_type"},
    KeyAlias{"text_config.rope_parameters.rope_type", "rope.scaling.type"},
    KeyAlias{"rope_parameters.rope_type", "rope.scaling.type"},
    KeyAlias{"text_config.rope_parameters.type", "rope.scaling.type"},
    KeyAlias{"rope_parameters.type", "rope.scaling.type"},
};
constexpr std::array kNativeTensorAliases = {
    TensorAlias{"lm_head.weight", "output.weight"},
    TensorAlias{"model.language_model.embed_tokens.weight", "token_embd.weight"},
    TensorAlias{"model.language_model.hyper_connection_mixer.hc_norm.weight",
                "output_hc_norm.weight"},
    TensorAlias{"model.language_model.hyper_connection_mixer.input_mix_weight_down.weight",
                "output_hc_down.weight"},
    TensorAlias{"model.language_model.hyper_connection_mixer.input_mix_weight_up.weight",
                "output_hc_up.weight"},
    TensorAlias{
        "model.language_model.layers.{layer}.attn_hyper_connection.block_inject_weight.weight",
        "blk.{layer}.hc_attn_inject.weight"},
    TensorAlias{"model.language_model.layers.{layer}.attn_hyper_connection.hc_norm.weight",
                "blk.{layer}.hc_attn_norm.weight"},
    TensorAlias{
        "model.language_model.layers.{layer}.attn_hyper_connection.input_mix_weight_down.weight",
        "blk.{layer}.hc_attn_down.weight"},
    TensorAlias{
        "model.language_model.layers.{layer}.attn_hyper_connection.input_mix_weight_up.weight",
        "blk.{layer}.hc_attn_up.weight"},
    TensorAlias{"model.language_model.layers.{layer}.linear_attn.A_log", "blk.{layer}.ssm_a"},
    TensorAlias{"model.language_model.layers.{layer}.linear_attn.conv1d.weight",
                "blk.{layer}.ssm_conv1d.weight"},
    TensorAlias{"model.language_model.layers.{layer}.linear_attn.dt_bias",
                "blk.{layer}.ssm_dt.bias"},
    TensorAlias{"model.language_model.layers.{layer}.linear_attn.in_proj_a.weight",
                "blk.{layer}.ssm_alpha.weight"},
    TensorAlias{"model.language_model.layers.{layer}.linear_attn.in_proj_a.weight_scale",
                "blk.{layer}.ssm_alpha.weight_scale"},
    TensorAlias{"model.language_model.layers.{layer}.linear_attn.in_proj_b.weight",
                "blk.{layer}.ssm_beta.weight"},
    TensorAlias{"model.language_model.layers.{layer}.linear_attn.in_proj_b.weight_scale",
                "blk.{layer}.ssm_beta.weight_scale"},
    TensorAlias{"model.language_model.layers.{layer}.linear_attn.in_proj_qkv.weight",
                "blk.{layer}.attn_qkv.weight"},
    TensorAlias{"model.language_model.layers.{layer}.linear_attn.in_proj_qkv.weight_scale",
                "blk.{layer}.attn_qkv.weight_scale"},
    TensorAlias{"model.language_model.layers.{layer}.linear_attn.in_proj_z.weight",
                "blk.{layer}.attn_gate.weight"},
    TensorAlias{"model.language_model.layers.{layer}.linear_attn.in_proj_z.weight_scale",
                "blk.{layer}.attn_gate.weight_scale"},
    TensorAlias{"model.language_model.layers.{layer}.linear_attn.norm.weight",
                "blk.{layer}.ssm_norm.weight"},
    TensorAlias{"model.language_model.layers.{layer}.linear_attn.out_proj.weight",
                "blk.{layer}.ssm_out.weight"},
    TensorAlias{"model.language_model.layers.{layer}.linear_attn.out_proj.weight_scale",
                "blk.{layer}.ssm_out.weight_scale"},
    TensorAlias{"model.language_model.layers.{layer}.mlp.experts.{expert}.down_proj.weight",
                "blk.{layer}.ffn_down_exps.weight"},
    TensorAlias{"model.language_model.layers.{layer}.mlp.experts.{expert}.down_proj.weight_scale",
                "blk.{layer}.ffn_down_exps.weight_scale"},
    TensorAlias{"model.language_model.layers.{layer}.mlp.experts.{expert}.down_proj.weight_scale_2",
                "blk.{layer}.ffn_down_exps.weight_scale_2"},
    TensorAlias{"model.language_model.layers.{layer}.mlp.experts.{expert}.gate_proj.weight",
                "blk.{layer}.ffn_gate_exps.weight"},
    TensorAlias{"model.language_model.layers.{layer}.mlp.experts.{expert}.gate_proj.weight_scale",
                "blk.{layer}.ffn_gate_exps.weight_scale"},
    TensorAlias{"model.language_model.layers.{layer}.mlp.experts.{expert}.gate_proj.weight_scale_2",
                "blk.{layer}.ffn_gate_exps.weight_scale_2"},
    TensorAlias{"model.language_model.layers.{layer}.mlp.experts.{expert}.up_proj.weight",
                "blk.{layer}.ffn_up_exps.weight"},
    TensorAlias{"model.language_model.layers.{layer}.mlp.experts.{expert}.up_proj.weight_scale",
                "blk.{layer}.ffn_up_exps.weight_scale"},
    TensorAlias{"model.language_model.layers.{layer}.mlp.experts.{expert}.up_proj.weight_scale_2",
                "blk.{layer}.ffn_up_exps.weight_scale_2"},
    TensorAlias{"model.language_model.layers.{layer}.mlp.gate.weight",
                "blk.{layer}.ffn_gate_inp.weight"},
    TensorAlias{"model.language_model.layers.{layer}.mlp.shared_expert.down_proj.weight",
                "blk.{layer}.ffn_down_shexp.weight"},
    TensorAlias{"model.language_model.layers.{layer}.mlp.shared_expert.down_proj.weight_scale",
                "blk.{layer}.ffn_down_shexp.weight_scale"},
    TensorAlias{"model.language_model.layers.{layer}.mlp.shared_expert.gate_proj.weight",
                "blk.{layer}.ffn_gate_shexp.weight"},
    TensorAlias{"model.language_model.layers.{layer}.mlp.shared_expert.gate_proj.weight_scale",
                "blk.{layer}.ffn_gate_shexp.weight_scale"},
    TensorAlias{"model.language_model.layers.{layer}.mlp.shared_expert.up_proj.weight",
                "blk.{layer}.ffn_up_shexp.weight"},
    TensorAlias{"model.language_model.layers.{layer}.mlp.shared_expert.up_proj.weight_scale",
                "blk.{layer}.ffn_up_shexp.weight_scale"},
    TensorAlias{"model.language_model.layers.{layer}.mlp.shared_expert_gate.weight",
                "blk.{layer}.ffn_gate_inp_shexp.weight"},
    TensorAlias{
        "model.language_model.layers.{layer}.mlp_hyper_connection.block_inject_weight.weight",
        "blk.{layer}.hc_ffn_inject.weight"},
    TensorAlias{"model.language_model.layers.{layer}.mlp_hyper_connection.hc_norm.weight",
                "blk.{layer}.hc_ffn_norm.weight"},
    TensorAlias{
        "model.language_model.layers.{layer}.mlp_hyper_connection.input_mix_weight_down.weight",
        "blk.{layer}.hc_ffn_down.weight"},
    TensorAlias{
        "model.language_model.layers.{layer}.mlp_hyper_connection.input_mix_weight_up.weight",
        "blk.{layer}.hc_ffn_up.weight"},
    TensorAlias{"model.language_model.layers.{layer}.ple.conv1d.weight",
                "blk.{layer}.ple_conv1d.weight"},
    TensorAlias{"model.language_model.layers.{layer}.ple.key_proj.weight",
                "blk.{layer}.ple_key.weight"},
    TensorAlias{"model.language_model.layers.{layer}.ple.norm_conv.weight",
                "blk.{layer}.ple_norm_conv.weight"},
    TensorAlias{"model.language_model.layers.{layer}.ple.norm_key.weight",
                "blk.{layer}.ple_norm_key.weight"},
    TensorAlias{"model.language_model.layers.{layer}.ple.norm_query.weight",
                "blk.{layer}.ple_norm_query.weight"},
    TensorAlias{
        "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_{shard}.weight",
        "per_layer_token_embd.weight"},
    TensorAlias{"model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_{shard}."
                "weight_scale",
                "per_layer_token_embd.weight_scale"},
    TensorAlias{"model.language_model.layers.1.ple.ple_embedding.ngram_embedding.weight_scale_2",
                "per_layer_token_embd.weight_scale_2"},
    TensorAlias{"model.language_model.layers.{layer}.ple.value_proj.weight",
                "blk.{layer}.ple_value.weight"},
    TensorAlias{"model.language_model.layers.{layer}.self_attn.indexer.index_qk_proj.weight",
                "blk.{layer}.indexer.qk_proj.weight"},
    TensorAlias{"model.language_model.layers.{layer}.self_attn.indexer.index_qk_proj.weight_scale",
                "blk.{layer}.indexer.qk_proj.weight_scale"},
    TensorAlias{"model.language_model.layers.{layer}.self_attn.indexer.k_layernorm.weight",
                "blk.{layer}.indexer.k_norm.weight"},
    TensorAlias{"model.language_model.layers.{layer}.self_attn.indexer.q_layernorm.weight",
                "blk.{layer}.indexer.q_norm.weight"},
    TensorAlias{"model.language_model.layers.{layer}.self_attn.k_norm.weight",
                "blk.{layer}.attn_k_norm.weight"},
    TensorAlias{"model.language_model.layers.{layer}.self_attn.k_proj.weight",
                "blk.{layer}.attn_k.weight"},
    TensorAlias{"model.language_model.layers.{layer}.self_attn.k_proj.weight_scale",
                "blk.{layer}.attn_k.weight_scale"},
    TensorAlias{"model.language_model.layers.{layer}.self_attn.o_proj.weight",
                "blk.{layer}.attn_output.weight"},
    TensorAlias{"model.language_model.layers.{layer}.self_attn.o_proj.weight_scale",
                "blk.{layer}.attn_output.weight_scale"},
    TensorAlias{"model.language_model.layers.{layer}.self_attn.q_norm.weight",
                "blk.{layer}.attn_q_norm.weight"},
    TensorAlias{"model.language_model.layers.{layer}.self_attn.q_proj.weight",
                "blk.{layer}.attn_q.weight"},
    TensorAlias{"model.language_model.layers.{layer}.self_attn.q_proj.weight_scale",
                "blk.{layer}.attn_q.weight_scale"},
    TensorAlias{"model.language_model.layers.{layer}.self_attn.v_proj.weight",
                "blk.{layer}.attn_v.weight"},
    TensorAlias{"model.language_model.layers.{layer}.self_attn.v_proj.weight_scale",
                "blk.{layer}.attn_v.weight_scale"},
    TensorAlias{"language_model.lm_head.weight", "output.weight"},
    TensorAlias{"language_model.lm_head.scales", "output.weight_scale"},
    TensorAlias{"language_model.lm_head.biases", "output.weight_bias"},
    TensorAlias{"language_model.model.embed_tokens.weight", "token_embd.weight"},
    TensorAlias{"language_model.model.embed_tokens.scales", "token_embd.weight_scale"},
    TensorAlias{"language_model.model.embed_tokens.biases", "token_embd.weight_bias"},
    TensorAlias{"language_model.model.hyper_connection_mixer.hc_norm.weight",
                "output_hc_norm.weight"},
    TensorAlias{"language_model.model.hyper_connection_mixer.input_mix_weight_down.weight",
                "output_hc_down.weight"},
    TensorAlias{"language_model.model.hyper_connection_mixer.input_mix_weight_down.scales",
                "output_hc_down.weight_scale"},
    TensorAlias{"language_model.model.hyper_connection_mixer.input_mix_weight_down.biases",
                "output_hc_down.weight_bias"},
    TensorAlias{"language_model.model.hyper_connection_mixer.input_mix_weight_up.weight",
                "output_hc_up.weight"},
    TensorAlias{"language_model.model.hyper_connection_mixer.input_mix_weight_up.scales",
                "output_hc_up.weight_scale"},
    TensorAlias{"language_model.model.hyper_connection_mixer.input_mix_weight_up.biases",
                "output_hc_up.weight_bias"},
    TensorAlias{"language_model.model.layers.{layer}.ple.conv1d.weight",
                "blk.{layer}.ple_conv1d.weight"},
    TensorAlias{"language_model.model.layers.{layer}.ple.key_proj.weight",
                "blk.{layer}.ple_key.weight"},
    TensorAlias{"language_model.model.layers.{layer}.ple.key_proj.scales",
                "blk.{layer}.ple_key.weight_scale"},
    TensorAlias{"language_model.model.layers.{layer}.ple.key_proj.biases",
                "blk.{layer}.ple_key.weight_bias"},
    TensorAlias{"language_model.model.layers.{layer}.ple.norm_conv.weight",
                "blk.{layer}.ple_norm_conv.weight"},
    TensorAlias{"language_model.model.layers.{layer}.ple.norm_key.weight",
                "blk.{layer}.ple_norm_key.weight"},
    TensorAlias{"language_model.model.layers.{layer}.ple.norm_query.weight",
                "blk.{layer}.ple_norm_query.weight"},
    TensorAlias{"language_model.model.layers.{layer}.ple.value_proj.weight",
                "blk.{layer}.ple_value.weight"},
    TensorAlias{"language_model.model.layers.{layer}.ple.value_proj.scales",
                "blk.{layer}.ple_value.weight_scale"},
    TensorAlias{"language_model.model.layers.{layer}.ple.value_proj.biases",
                "blk.{layer}.ple_value.weight_bias"},
    TensorAlias{
        "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shard_{shard}.weight",
        "per_layer_token_embd.weight"},
    TensorAlias{
        "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shard_{shard}.scales",
        "per_layer_token_embd.weight_scale"},
    TensorAlias{
        "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shard_{shard}.biases",
        "per_layer_token_embd.weight_bias"},
    TensorAlias{
        "language_model.model.layers.{layer}.attn_hyper_connection.block_inject_weight.weight",
        "blk.{layer}.hc_attn_inject.weight"},
    TensorAlias{
        "language_model.model.layers.{layer}.attn_hyper_connection.block_inject_weight.scales",
        "blk.{layer}.hc_attn_inject.weight_scale"},
    TensorAlias{
        "language_model.model.layers.{layer}.attn_hyper_connection.block_inject_weight.biases",
        "blk.{layer}.hc_attn_inject.weight_bias"},
    TensorAlias{"language_model.model.layers.{layer}.attn_hyper_connection.hc_norm.weight",
                "blk.{layer}.hc_attn_norm.weight"},
    TensorAlias{
        "language_model.model.layers.{layer}.attn_hyper_connection.input_mix_weight_down.weight",
        "blk.{layer}.hc_attn_down.weight"},
    TensorAlias{
        "language_model.model.layers.{layer}.attn_hyper_connection.input_mix_weight_down.scales",
        "blk.{layer}.hc_attn_down.weight_scale"},
    TensorAlias{
        "language_model.model.layers.{layer}.attn_hyper_connection.input_mix_weight_down.biases",
        "blk.{layer}.hc_attn_down.weight_bias"},
    TensorAlias{
        "language_model.model.layers.{layer}.attn_hyper_connection.input_mix_weight_up.weight",
        "blk.{layer}.hc_attn_up.weight"},
    TensorAlias{
        "language_model.model.layers.{layer}.attn_hyper_connection.input_mix_weight_up.scales",
        "blk.{layer}.hc_attn_up.weight_scale"},
    TensorAlias{
        "language_model.model.layers.{layer}.attn_hyper_connection.input_mix_weight_up.biases",
        "blk.{layer}.hc_attn_up.weight_bias"},
    TensorAlias{
        "language_model.model.layers.{layer}.mlp_hyper_connection.block_inject_weight.weight",
        "blk.{layer}.hc_ffn_inject.weight"},
    TensorAlias{
        "language_model.model.layers.{layer}.mlp_hyper_connection.block_inject_weight.scales",
        "blk.{layer}.hc_ffn_inject.weight_scale"},
    TensorAlias{
        "language_model.model.layers.{layer}.mlp_hyper_connection.block_inject_weight.biases",
        "blk.{layer}.hc_ffn_inject.weight_bias"},
    TensorAlias{"language_model.model.layers.{layer}.mlp_hyper_connection.hc_norm.weight",
                "blk.{layer}.hc_ffn_norm.weight"},
    TensorAlias{
        "language_model.model.layers.{layer}.mlp_hyper_connection.input_mix_weight_down.weight",
        "blk.{layer}.hc_ffn_down.weight"},
    TensorAlias{
        "language_model.model.layers.{layer}.mlp_hyper_connection.input_mix_weight_down.scales",
        "blk.{layer}.hc_ffn_down.weight_scale"},
    TensorAlias{
        "language_model.model.layers.{layer}.mlp_hyper_connection.input_mix_weight_down.biases",
        "blk.{layer}.hc_ffn_down.weight_bias"},
    TensorAlias{
        "language_model.model.layers.{layer}.mlp_hyper_connection.input_mix_weight_up.weight",
        "blk.{layer}.hc_ffn_up.weight"},
    TensorAlias{
        "language_model.model.layers.{layer}.mlp_hyper_connection.input_mix_weight_up.scales",
        "blk.{layer}.hc_ffn_up.weight_scale"},
    TensorAlias{
        "language_model.model.layers.{layer}.mlp_hyper_connection.input_mix_weight_up.biases",
        "blk.{layer}.hc_ffn_up.weight_bias"},
    TensorAlias{"language_model.model.layers.{layer}.mlp.gate.weight",
                "blk.{layer}.ffn_gate_inp.weight"},
    TensorAlias{"language_model.model.layers.{layer}.mlp.shared_expert_gate.weight",
                "blk.{layer}.ffn_gate_inp_shexp.weight"},
    TensorAlias{"language_model.model.layers.{layer}.mlp.shared_expert_gate.scales",
                "blk.{layer}.ffn_gate_inp_shexp.weight_scale"},
    TensorAlias{"language_model.model.layers.{layer}.mlp.shared_expert_gate.biases",
                "blk.{layer}.ffn_gate_inp_shexp.weight_bias"},
    TensorAlias{"language_model.model.layers.{layer}.mlp.shared_expert.gate_proj.weight",
                "blk.{layer}.ffn_gate_shexp.weight"},
    TensorAlias{"language_model.model.layers.{layer}.mlp.shared_expert.gate_proj.scales",
                "blk.{layer}.ffn_gate_shexp.weight_scale"},
    TensorAlias{"language_model.model.layers.{layer}.mlp.shared_expert.gate_proj.biases",
                "blk.{layer}.ffn_gate_shexp.weight_bias"},
    TensorAlias{"language_model.model.layers.{layer}.mlp.shared_expert.up_proj.weight",
                "blk.{layer}.ffn_up_shexp.weight"},
    TensorAlias{"language_model.model.layers.{layer}.mlp.shared_expert.up_proj.scales",
                "blk.{layer}.ffn_up_shexp.weight_scale"},
    TensorAlias{"language_model.model.layers.{layer}.mlp.shared_expert.up_proj.biases",
                "blk.{layer}.ffn_up_shexp.weight_bias"},
    TensorAlias{"language_model.model.layers.{layer}.mlp.shared_expert.down_proj.weight",
                "blk.{layer}.ffn_down_shexp.weight"},
    TensorAlias{"language_model.model.layers.{layer}.mlp.shared_expert.down_proj.scales",
                "blk.{layer}.ffn_down_shexp.weight_scale"},
    TensorAlias{"language_model.model.layers.{layer}.mlp.shared_expert.down_proj.biases",
                "blk.{layer}.ffn_down_shexp.weight_bias"},
    TensorAlias{"language_model.model.layers.{layer}.mlp.switch_mlp.gate_proj.weight",
                "blk.{layer}.ffn_gate_exps.weight"},
    TensorAlias{"language_model.model.layers.{layer}.mlp.switch_mlp.gate_proj.scales",
                "blk.{layer}.ffn_gate_exps.weight_scale"},
    TensorAlias{"language_model.model.layers.{layer}.mlp.switch_mlp.gate_proj.biases",
                "blk.{layer}.ffn_gate_exps.weight_bias"},
    TensorAlias{"language_model.model.layers.{layer}.mlp.switch_mlp.up_proj.weight",
                "blk.{layer}.ffn_up_exps.weight"},
    TensorAlias{"language_model.model.layers.{layer}.mlp.switch_mlp.up_proj.scales",
                "blk.{layer}.ffn_up_exps.weight_scale"},
    TensorAlias{"language_model.model.layers.{layer}.mlp.switch_mlp.up_proj.biases",
                "blk.{layer}.ffn_up_exps.weight_bias"},
    TensorAlias{"language_model.model.layers.{layer}.mlp.switch_mlp.down_proj.weight",
                "blk.{layer}.ffn_down_exps.weight"},
    TensorAlias{"language_model.model.layers.{layer}.mlp.switch_mlp.down_proj.scales",
                "blk.{layer}.ffn_down_exps.weight_scale"},
    TensorAlias{"language_model.model.layers.{layer}.mlp.switch_mlp.down_proj.biases",
                "blk.{layer}.ffn_down_exps.weight_bias"},
    TensorAlias{"language_model.model.layers.{layer}.linear_attn.in_proj_qkv.weight",
                "blk.{layer}.attn_qkv.weight"},
    TensorAlias{"language_model.model.layers.{layer}.linear_attn.in_proj_qkv.scales",
                "blk.{layer}.attn_qkv.weight_scale"},
    TensorAlias{"language_model.model.layers.{layer}.linear_attn.in_proj_qkv.biases",
                "blk.{layer}.attn_qkv.weight_bias"},
    TensorAlias{"language_model.model.layers.{layer}.linear_attn.in_proj_z.weight",
                "blk.{layer}.attn_gate.weight"},
    TensorAlias{"language_model.model.layers.{layer}.linear_attn.in_proj_z.scales",
                "blk.{layer}.attn_gate.weight_scale"},
    TensorAlias{"language_model.model.layers.{layer}.linear_attn.in_proj_z.biases",
                "blk.{layer}.attn_gate.weight_bias"},
    TensorAlias{"language_model.model.layers.{layer}.linear_attn.in_proj_b.weight",
                "blk.{layer}.ssm_beta.weight"},
    TensorAlias{"language_model.model.layers.{layer}.linear_attn.in_proj_b.scales",
                "blk.{layer}.ssm_beta.weight_scale"},
    TensorAlias{"language_model.model.layers.{layer}.linear_attn.in_proj_b.biases",
                "blk.{layer}.ssm_beta.weight_bias"},
    TensorAlias{"language_model.model.layers.{layer}.linear_attn.in_proj_a.weight",
                "blk.{layer}.ssm_alpha.weight"},
    TensorAlias{"language_model.model.layers.{layer}.linear_attn.in_proj_a.scales",
                "blk.{layer}.ssm_alpha.weight_scale"},
    TensorAlias{"language_model.model.layers.{layer}.linear_attn.in_proj_a.biases",
                "blk.{layer}.ssm_alpha.weight_bias"},
    TensorAlias{"language_model.model.layers.{layer}.linear_attn.out_proj.weight",
                "blk.{layer}.ssm_out.weight"},
    TensorAlias{"language_model.model.layers.{layer}.linear_attn.out_proj.scales",
                "blk.{layer}.ssm_out.weight_scale"},
    TensorAlias{"language_model.model.layers.{layer}.linear_attn.out_proj.biases",
                "blk.{layer}.ssm_out.weight_bias"},
    TensorAlias{"language_model.model.layers.{layer}.linear_attn.conv1d.weight",
                "blk.{layer}.ssm_conv1d.weight"},
    TensorAlias{"language_model.model.layers.{layer}.linear_attn.dt_bias",
                "blk.{layer}.ssm_dt.bias"},
    TensorAlias{"language_model.model.layers.{layer}.linear_attn.A_log", "blk.{layer}.ssm_a"},
    TensorAlias{"language_model.model.layers.{layer}.linear_attn.norm.weight",
                "blk.{layer}.ssm_norm.weight"},
    TensorAlias{"language_model.model.layers.{layer}.self_attn.q_proj.weight",
                "blk.{layer}.attn_q.weight"},
    TensorAlias{"language_model.model.layers.{layer}.self_attn.q_proj.scales",
                "blk.{layer}.attn_q.weight_scale"},
    TensorAlias{"language_model.model.layers.{layer}.self_attn.q_proj.biases",
                "blk.{layer}.attn_q.weight_bias"},
    TensorAlias{"language_model.model.layers.{layer}.self_attn.k_proj.weight",
                "blk.{layer}.attn_k.weight"},
    TensorAlias{"language_model.model.layers.{layer}.self_attn.k_proj.scales",
                "blk.{layer}.attn_k.weight_scale"},
    TensorAlias{"language_model.model.layers.{layer}.self_attn.k_proj.biases",
                "blk.{layer}.attn_k.weight_bias"},
    TensorAlias{"language_model.model.layers.{layer}.self_attn.v_proj.weight",
                "blk.{layer}.attn_v.weight"},
    TensorAlias{"language_model.model.layers.{layer}.self_attn.v_proj.scales",
                "blk.{layer}.attn_v.weight_scale"},
    TensorAlias{"language_model.model.layers.{layer}.self_attn.v_proj.biases",
                "blk.{layer}.attn_v.weight_bias"},
    TensorAlias{"language_model.model.layers.{layer}.self_attn.o_proj.weight",
                "blk.{layer}.attn_output.weight"},
    TensorAlias{"language_model.model.layers.{layer}.self_attn.o_proj.scales",
                "blk.{layer}.attn_output.weight_scale"},
    TensorAlias{"language_model.model.layers.{layer}.self_attn.o_proj.biases",
                "blk.{layer}.attn_output.weight_bias"},
    TensorAlias{"language_model.model.layers.{layer}.self_attn.indexer.index_qk_proj.weight",
                "blk.{layer}.indexer.qk_proj.weight"},
    TensorAlias{"language_model.model.layers.{layer}.self_attn.indexer.index_qk_proj.scales",
                "blk.{layer}.indexer.qk_proj.weight_scale"},
    TensorAlias{"language_model.model.layers.{layer}.self_attn.indexer.index_qk_proj.biases",
                "blk.{layer}.indexer.qk_proj.weight_bias"},
    TensorAlias{"language_model.model.layers.{layer}.self_attn.q_norm.weight",
                "blk.{layer}.attn_q_norm.weight"},
    TensorAlias{"language_model.model.layers.{layer}.self_attn.k_norm.weight",
                "blk.{layer}.attn_k_norm.weight"},
    TensorAlias{"language_model.model.layers.{layer}.self_attn.indexer.q_layernorm.weight",
                "blk.{layer}.indexer.q_norm.weight"},
    TensorAlias{"language_model.model.layers.{layer}.self_attn.indexer.k_layernorm.weight",
                "blk.{layer}.indexer.k_norm.weight"},
    TensorAlias{"model.embed_tokens.weight", "token_embd.weight"},
    TensorAlias{"model.hyper_connection_mixer.hc_norm.weight", "output_hc_norm.weight"},
    TensorAlias{"model.hyper_connection_mixer.input_mix_weight_down.weight",
                "output_hc_down.weight"},
    TensorAlias{"model.hyper_connection_mixer.input_mix_weight_up.weight", "output_hc_up.weight"},
    TensorAlias{"model.layers.{layer}.attn_hyper_connection.block_inject_weight.weight",
                "blk.{layer}.hc_attn_inject.weight"},
    TensorAlias{"model.layers.{layer}.attn_hyper_connection.hc_norm.weight",
                "blk.{layer}.hc_attn_norm.weight"},
    TensorAlias{"model.layers.{layer}.attn_hyper_connection.input_mix_weight_down.weight",
                "blk.{layer}.hc_attn_down.weight"},
    TensorAlias{"model.layers.{layer}.attn_hyper_connection.input_mix_weight_up.weight",
                "blk.{layer}.hc_attn_up.weight"},
    TensorAlias{"model.layers.{layer}.linear_attn.A_log", "blk.{layer}.ssm_a"},
    TensorAlias{"model.layers.{layer}.linear_attn.conv1d.weight", "blk.{layer}.ssm_conv1d.weight"},
    TensorAlias{"model.layers.{layer}.linear_attn.dt_bias", "blk.{layer}.ssm_dt.bias"},
    TensorAlias{"model.layers.{layer}.linear_attn.in_proj_a.weight",
                "blk.{layer}.ssm_alpha.weight"},
    TensorAlias{"model.layers.{layer}.linear_attn.in_proj_a.weight_scale",
                "blk.{layer}.ssm_alpha.weight_scale"},
    TensorAlias{"model.layers.{layer}.linear_attn.in_proj_b.weight", "blk.{layer}.ssm_beta.weight"},
    TensorAlias{"model.layers.{layer}.linear_attn.in_proj_b.weight_scale",
                "blk.{layer}.ssm_beta.weight_scale"},
    TensorAlias{"model.layers.{layer}.linear_attn.in_proj_qkv.weight",
                "blk.{layer}.attn_qkv.weight"},
    TensorAlias{"model.layers.{layer}.linear_attn.in_proj_qkv.weight_scale",
                "blk.{layer}.attn_qkv.weight_scale"},
    TensorAlias{"model.layers.{layer}.linear_attn.in_proj_z.weight",
                "blk.{layer}.attn_gate.weight"},
    TensorAlias{"model.layers.{layer}.linear_attn.in_proj_z.weight_scale",
                "blk.{layer}.attn_gate.weight_scale"},
    TensorAlias{"model.layers.{layer}.linear_attn.norm.weight", "blk.{layer}.ssm_norm.weight"},
    TensorAlias{"model.layers.{layer}.linear_attn.out_proj.weight", "blk.{layer}.ssm_out.weight"},
    TensorAlias{"model.layers.{layer}.linear_attn.out_proj.weight_scale",
                "blk.{layer}.ssm_out.weight_scale"},
    TensorAlias{"model.layers.{layer}.mlp.experts.{expert}.down_proj.weight",
                "blk.{layer}.ffn_down_exps.weight"},
    TensorAlias{"model.layers.{layer}.mlp.experts.{expert}.down_proj.weight_scale",
                "blk.{layer}.ffn_down_exps.weight_scale"},
    TensorAlias{"model.layers.{layer}.mlp.experts.{expert}.down_proj.weight_scale_2",
                "blk.{layer}.ffn_down_exps.weight_scale_2"},
    TensorAlias{"model.layers.{layer}.mlp.experts.{expert}.gate_proj.weight",
                "blk.{layer}.ffn_gate_exps.weight"},
    TensorAlias{"model.layers.{layer}.mlp.experts.{expert}.gate_proj.weight_scale",
                "blk.{layer}.ffn_gate_exps.weight_scale"},
    TensorAlias{"model.layers.{layer}.mlp.experts.{expert}.gate_proj.weight_scale_2",
                "blk.{layer}.ffn_gate_exps.weight_scale_2"},
    TensorAlias{"model.layers.{layer}.mlp.experts.{expert}.up_proj.weight",
                "blk.{layer}.ffn_up_exps.weight"},
    TensorAlias{"model.layers.{layer}.mlp.experts.{expert}.up_proj.weight_scale",
                "blk.{layer}.ffn_up_exps.weight_scale"},
    TensorAlias{"model.layers.{layer}.mlp.experts.{expert}.up_proj.weight_scale_2",
                "blk.{layer}.ffn_up_exps.weight_scale_2"},
    TensorAlias{"model.layers.{layer}.mlp.gate.weight", "blk.{layer}.ffn_gate_inp.weight"},
    TensorAlias{"model.layers.{layer}.mlp.shared_expert.down_proj.weight",
                "blk.{layer}.ffn_down_shexp.weight"},
    TensorAlias{"model.layers.{layer}.mlp.shared_expert.down_proj.weight_scale",
                "blk.{layer}.ffn_down_shexp.weight_scale"},
    TensorAlias{"model.layers.{layer}.mlp.shared_expert.gate_proj.weight",
                "blk.{layer}.ffn_gate_shexp.weight"},
    TensorAlias{"model.layers.{layer}.mlp.shared_expert.gate_proj.weight_scale",
                "blk.{layer}.ffn_gate_shexp.weight_scale"},
    TensorAlias{"model.layers.{layer}.mlp.shared_expert.up_proj.weight",
                "blk.{layer}.ffn_up_shexp.weight"},
    TensorAlias{"model.layers.{layer}.mlp.shared_expert.up_proj.weight_scale",
                "blk.{layer}.ffn_up_shexp.weight_scale"},
    TensorAlias{"model.layers.{layer}.mlp.shared_expert_gate.weight",
                "blk.{layer}.ffn_gate_inp_shexp.weight"},
    TensorAlias{"model.layers.{layer}.mlp_hyper_connection.block_inject_weight.weight",
                "blk.{layer}.hc_ffn_inject.weight"},
    TensorAlias{"model.layers.{layer}.mlp_hyper_connection.hc_norm.weight",
                "blk.{layer}.hc_ffn_norm.weight"},
    TensorAlias{"model.layers.{layer}.mlp_hyper_connection.input_mix_weight_down.weight",
                "blk.{layer}.hc_ffn_down.weight"},
    TensorAlias{"model.layers.{layer}.mlp_hyper_connection.input_mix_weight_up.weight",
                "blk.{layer}.hc_ffn_up.weight"},
    TensorAlias{"model.layers.{layer}.ple.conv1d.weight", "blk.{layer}.ple_conv1d.weight"},
    TensorAlias{"model.layers.{layer}.ple.key_proj.weight", "blk.{layer}.ple_key.weight"},
    TensorAlias{"model.layers.{layer}.ple.norm_conv.weight", "blk.{layer}.ple_norm_conv.weight"},
    TensorAlias{"model.layers.{layer}.ple.norm_key.weight", "blk.{layer}.ple_norm_key.weight"},
    TensorAlias{"model.layers.{layer}.ple.norm_query.weight", "blk.{layer}.ple_norm_query.weight"},
    TensorAlias{"model.layers.1.ple.ple_embedding.ngram_embedding.shard_{shard}.weight",
                "per_layer_token_embd.weight"},
    TensorAlias{"model.layers.1.ple.ple_embedding.ngram_embedding.shard_{shard}.weight_scale",
                "per_layer_token_embd.weight_scale"},
    TensorAlias{"model.layers.1.ple.ple_embedding.ngram_embedding.weight_scale_2",
                "per_layer_token_embd.weight_scale_2"},
    TensorAlias{"model.layers.{layer}.ple.value_proj.weight", "blk.{layer}.ple_value.weight"},
    TensorAlias{"model.layers.{layer}.self_attn.indexer.index_qk_proj.weight",
                "blk.{layer}.indexer.qk_proj.weight"},
    TensorAlias{"model.layers.{layer}.self_attn.indexer.index_qk_proj.weight_scale",
                "blk.{layer}.indexer.qk_proj.weight_scale"},
    TensorAlias{"model.layers.{layer}.self_attn.indexer.k_layernorm.weight",
                "blk.{layer}.indexer.k_norm.weight"},
    TensorAlias{"model.layers.{layer}.self_attn.indexer.q_layernorm.weight",
                "blk.{layer}.indexer.q_norm.weight"},
    TensorAlias{"model.layers.{layer}.self_attn.k_norm.weight", "blk.{layer}.attn_k_norm.weight"},
    TensorAlias{"model.layers.{layer}.self_attn.k_proj.weight", "blk.{layer}.attn_k.weight"},
    TensorAlias{"model.layers.{layer}.self_attn.k_proj.weight_scale",
                "blk.{layer}.attn_k.weight_scale"},
    TensorAlias{"model.layers.{layer}.self_attn.o_proj.weight", "blk.{layer}.attn_output.weight"},
    TensorAlias{"model.layers.{layer}.self_attn.o_proj.weight_scale",
                "blk.{layer}.attn_output.weight_scale"},
    TensorAlias{"model.layers.{layer}.self_attn.q_norm.weight", "blk.{layer}.attn_q_norm.weight"},
    TensorAlias{"model.layers.{layer}.self_attn.q_proj.weight", "blk.{layer}.attn_q.weight"},
    TensorAlias{"model.layers.{layer}.self_attn.q_proj.weight_scale",
                "blk.{layer}.attn_q.weight_scale"},
    TensorAlias{"model.layers.{layer}.self_attn.v_proj.weight", "blk.{layer}.attn_v.weight"},
    TensorAlias{"model.layers.{layer}.self_attn.v_proj.weight_scale",
                "blk.{layer}.attn_v.weight_scale"},
    TensorAlias{"model.layers.1.self_attn.ple_embed.shard_{shard}.weight",
                "per_layer_token_embd.weight"},
    TensorAlias{"model.layers.1.self_attn.ple_embed.shard_{shard}.weight_scale",
                "per_layer_token_embd.weight_scale"},
    TensorAlias{"model.layers.1.self_attn.ple_embed.weight_scale_2",
                "per_layer_token_embd.weight_scale_2"},
};

// ------------------------------------------------------------ qwen4exp-mtp

constexpr std::string_view kDraftRows = "draft_head_rows";
// The rows of the draft head a pass reads (Qwen38Options::draft_vocab's
// default: 65,536, or a selected head's rows when fewer).
constexpr std::string_view kDraftVocab = "draft_vocab";
constexpr std::string_view kRecipeDraftVocab = "qwen4exp-mtp.draft_vocab";

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
    LLMP_QWEN38_BLOCK_PARAMS,
    FromKey("embedding", "eps", kEps, kReal),
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

// Autoregressive MTP passes; a selected head's token map and the draft
// vocabulary's value.
constexpr std::string_view kHeadIds = "draft_output.ids";
constexpr DraftRecipe kMtpDraft{.shape = DraftRecipe::Shape::kMtp,
                                .target_architecture = "qwen4exp",
                                .head_ids = kHeadIds,
                                .vocab_key = kRecipeDraftVocab};

constexpr Mapping kMtpMapping{
    .architecture = "qwen4exp-mtp",
    .version = 1,
    .pipeline = PipelineKind::kLlm,
    .layer_count_key = "block_count",
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
    .auxiliary = kGgufAuxiliary,
};

// The native importer's drafter (D-109): the
// checkpoint's MTP tensors in their source encodings (its BF16 linears)
// but its norms' 1 + w folded in F32 (norm.plus_one, chosen by role), its
// routed experts relaid out as the native target's (scale-atom gate-up and
// down tables, `.weight` and `.weight_scale`, and their tensor scales' F32
// vectors), and the optional selected head the importer gathered from the
// target's head rows. The mapping names no fold of its own: the source
// records the importer's (Source::folds).
constexpr auto kMtpNativeTensors = [] {
  constexpr auto ExpertScales = [](std::string_view pattern, std::string_view slot,
                                   std::string_view role) {
    auto r = Scales(pattern, slot, role);
    r.experts = true;
    return r;
  };
  constexpr auto ExpertBiases = [](std::string_view pattern, std::string_view slot,
                                   std::string_view role) {
    auto r = Biases(pattern, slot, role);
    r.experts = true;
    return r;
  };
  std::array<TensorRule, kMtpTensors.size() + 51> rules{};
  std::size_t next = 0;
  for (auto rule : kMtpTensors) {
    if (rule.pattern == "blk.{layer}.ffn_gate_up_exps.codes") {
      rule.pattern = "blk.{layer}.ffn_gate_up_exps.weight";
      rule.optional = true;
    } else if (rule.pattern == "blk.{layer}.ffn_gate_up_exps.scales") {
      rule.pattern = "blk.{layer}.ffn_gate_up_exps.weight_scale";
      rule.optional = true;
    } else if (rule.pattern == "blk.{layer}.ffn_down_exps.codes") {
      rule.pattern = "blk.{layer}.ffn_down_exps.weight";
      rule.optional = true;
    } else if (rule.pattern == "blk.{layer}.ffn_down_exps.scales") {
      rule.pattern = "blk.{layer}.ffn_down_exps.weight_scale";
      rule.optional = true;
    } else if (rule.pattern.ends_with("_exps.weight_scale_2")) {
      // ModelOpt's tensor scales; an MLX drafter's affine experts have none.
      rule.optional = true;
    }
    if (rule.experts && rule.pattern.ends_with(".weight")) rule.any_part = true;
    rules[next++] = rule;
  }
  // Linear scale and bias rules
  rules[next++] = Scales("fc_embd.weight_scale", "embedding", "fc_embd");
  rules[next++] = Biases("fc_embd.weight_bias", "embedding", "fc_embd");
  rules[next++] = Scales("fc_hidden.weight_scale", "embedding", "fc_hidden");
  rules[next++] = Biases("fc_hidden.weight_bias", "embedding", "fc_hidden");
  rules[next++] = Scales("blk.{layer}.hc_attn_down.weight_scale", "attn_mix", "down");
  rules[next++] = Biases("blk.{layer}.hc_attn_down.weight_bias", "attn_mix", "down");
  rules[next++] = Scales("blk.{layer}.hc_attn_up.weight_scale", "attn_mix", "up");
  rules[next++] = Biases("blk.{layer}.hc_attn_up.weight_bias", "attn_mix", "up");
  rules[next++] = Scales("blk.{layer}.hc_attn_inject.weight_scale", "attn_combine", "inject");
  rules[next++] = Biases("blk.{layer}.hc_attn_inject.weight_bias", "attn_combine", "inject");
  rules[next++] = Scales("blk.{layer}.hc_ffn_down.weight_scale", "ffn_mix", "down");
  rules[next++] = Biases("blk.{layer}.hc_ffn_down.weight_bias", "ffn_mix", "down");
  rules[next++] = Scales("blk.{layer}.hc_ffn_up.weight_scale", "ffn_mix", "up");
  rules[next++] = Biases("blk.{layer}.hc_ffn_up.weight_bias", "ffn_mix", "up");
  rules[next++] = Scales("blk.{layer}.hc_ffn_inject.weight_scale", "ffn_combine", "inject");
  rules[next++] = Biases("blk.{layer}.hc_ffn_inject.weight_bias", "ffn_combine", "inject");
  rules[next++] = Scales("blk.{layer}.attn_q.weight_scale", "attention", "q");
  rules[next++] = Biases("blk.{layer}.attn_q.weight_bias", "attention", "q");
  rules[next++] = Scales("blk.{layer}.attn_k.weight_scale", "attention", "k");
  rules[next++] = Biases("blk.{layer}.attn_k.weight_bias", "attention", "k");
  rules[next++] = Scales("blk.{layer}.attn_v.weight_scale", "attention", "v");
  rules[next++] = Biases("blk.{layer}.attn_v.weight_bias", "attention", "v");
  rules[next++] = Scales("blk.{layer}.attn_output.weight_scale", "attention", "output");
  rules[next++] = Biases("blk.{layer}.attn_output.weight_bias", "attention", "output");
  rules[next++] = Scales("blk.{layer}.indexer.qk_proj.weight_scale", "attention", "indexer_qk");
  rules[next++] = Biases("blk.{layer}.indexer.qk_proj.weight_bias", "attention", "indexer_qk");
  rules[next++] = Scales("blk.{layer}.ffn_gate_inp.weight_scale", "moe", "router");
  rules[next++] = Biases("blk.{layer}.ffn_gate_inp.weight_bias", "moe", "router");
  rules[next++] = Scales("blk.{layer}.ffn_gate_inp_shexp.weight_scale", "moe", "shared_gate");
  rules[next++] = Biases("blk.{layer}.ffn_gate_inp_shexp.weight_bias", "moe", "shared_gate");
  rules[next++] = Scales("blk.{layer}.ffn_gate_shexp.weight_scale", "moe", "gate");
  rules[next++] = Biases("blk.{layer}.ffn_gate_shexp.weight_bias", "moe", "gate");
  rules[next++] = Scales("blk.{layer}.ffn_up_shexp.weight_scale", "moe", "up");
  rules[next++] = Biases("blk.{layer}.ffn_up_shexp.weight_bias", "moe", "up");
  rules[next++] = Scales("blk.{layer}.ffn_down_shexp.weight_scale", "moe", "down");
  rules[next++] = Biases("blk.{layer}.ffn_down_shexp.weight_bias", "moe", "down");
  rules[next++] = Scales("output_hc_down.weight_scale", "output_mix", "down");
  rules[next++] = Biases("output_hc_down.weight_bias", "output_mix", "down");
  rules[next++] = Scales("output_hc_up.weight_scale", "output_mix", "up");
  rules[next++] = Biases("output_hc_up.weight_bias", "output_mix", "up");
  rules[next++] = Scales("draft_output.weight_scale", "selected", "weight");
  rules[next++] = Biases("draft_output.weight_bias", "selected", "weight");
  rules[next++] = Scales("output.weight_scale", "output", "weight");
  rules[next++] = Biases("output.weight_bias", "output", "weight");
  // Unstacked expert rules
  rules[next++] = TensorRule{.pattern = "blk.{layer}.ffn_gate_exps.weight",
                             .slot = "moe",
                             .role = "gate_experts",
                             .part = PartRole::kCodes,
                             .experts = true,
                             .optional = true,
                             .any_part = true};
  rules[next++] = ExpertScales("blk.{layer}.ffn_gate_exps.weight_scale", "moe", "gate_experts");
  rules[next++] = ExpertBiases("blk.{layer}.ffn_gate_exps.weight_bias", "moe", "gate_experts");
  rules[next++] = TensorRule{.pattern = "blk.{layer}.ffn_up_exps.weight",
                             .slot = "moe",
                             .role = "up_experts",
                             .part = PartRole::kCodes,
                             .experts = true,
                             .optional = true,
                             .any_part = true};
  rules[next++] = ExpertScales("blk.{layer}.ffn_up_exps.weight_scale", "moe", "up_experts");
  rules[next++] = ExpertBiases("blk.{layer}.ffn_up_exps.weight_bias", "moe", "up_experts");
  rules[next++] = ExpertBiases("blk.{layer}.ffn_down_exps.weight_bias", "moe", "down_experts");
  return rules;
}();
static_assert(kMtpNativeTensors.size() == kMtpTensors.size() + 51);
// Its norms zero-centered, as the checkpoint holds them, unless the
// importer stored 1 + w in F32 (norm.plus_one, as the converted drafter's).
template <std::size_t N>
constexpr auto ZeroCentered(const std::array<ParamRule, N>& base) {
  std::array<ParamRule, N + 19> params{};
  std::size_t i = 0;
  for (auto rule : base) {
    if (rule.slot == "moe" && rule.param == "projection_scales") continue;
    if (rule.slot == "moe" && !rule.key.empty() &&
        (rule.param == "stacked" || rule.param == "projection_scales")) {
      rule.tensor = rule.param == "stacked" ? "blk.{layer}.ffn_gate_up_exps.weight"
                                            : "blk.{layer}.ffn_gate_exps.weight_scale_2";
      rule.key = {};
    }
    if (rule.param == "eps") {
      rule.transform = Transform::kFloat32;
    } else if (rule.slot == "attention" && rule.param == "rope_dims") {
      rule.numerator_key = "attention.key_length";
      rule.divisor_key = "rope.partial_rotary_factor";
      rule.transform = Transform::kProduct;
    } else if (rule.slot == "attention" && rule.param == "indexer" && rule.key.empty()) {
      rule.value.text = "fused";
    }
    params[i++] = rule;
  }
  params[i++] = Flag("moe", "projection_scales", false);
  params[i++] = FromTensor("moe", "gate_scale", "blk.{layer}.ffn_gate_exps.weight_scale_2");
  params[i++] = Flag("moe", "gate_scale", false);
  params[i++] = FromTensor("moe", "up_scale", "blk.{layer}.ffn_up_exps.weight_scale_2");
  params[i++] = Flag("moe", "up_scale", false);
  params[i++] = FromTensor("moe", "down_scale", "blk.{layer}.ffn_down_exps.weight_scale_2");
  params[i++] = Flag("moe", "down_scale", false);
  for (std::string_view slot : {"embedding", "attn_mix", "ffn_mix", "output_mix", "attention"}) {
    params[i] = Flag(slot, "norm_zero_centered", true);
    params[i++].unless_fold = "norm.plus_one";
  }
  params[i++] = Require("mamba_ssm_dtype", "float32", true);
  params[i++] = Require("dtype", "bfloat16", true);
  params[i++] = Require("hidden_act", "silu", true);
  params[i++] = Require("output_gate_type", "sigmoid", true);
  params[i++] = Require("rope.scaling.type", "default", true);
  params[i++] = ParamRule{
      .key = "is_encoder_decoder", .type = kBool, .value = {.flag = false}, .optional = true};
  params[i++] =
      ParamRule{.key = "attention_bias", .type = kBool, .value = {.flag = false}, .optional = true};
  params[i++] = RequireInt("attention.indexer.head_count_kv", 1, true);
  params[i] = ParamRule{.key = "use_bidirectional_attention",
                        .type = kBool,
                        .value = {.flag = false},
                        .optional = true};
  return params;
}
constexpr auto kMtpNativeParams = ZeroCentered(kMtpParams);
// Missing kept metadata in published native MTP artifacts is reconstructed
// from the actual target spec and this artifact's own selected-head tensor.
constexpr std::array kMtpValues = {
    ValueRule{.key = "context_length", .origin = ValueOrigin::kContext},
    ValueRule{.key = "embedding_length", .origin = ValueOrigin::kWidth},
    ValueRule{.key = "hyper_connection.count", .origin = ValueOrigin::kStreams},
    ValueRule{.key = "hyper_connection.low_rank",
              .origin = ValueOrigin::kComponent,
              .kind = "hc.mix",
              .param = "rank"},
    ValueRule{.key = "attention.layer_norm_rms_epsilon",
              .origin = ValueOrigin::kComponent,
              .kind = "attention.qsa",
              .param = "eps"},
    ValueRule{.key = "attention.head_count",
              .origin = ValueOrigin::kComponent,
              .kind = "attention.qsa",
              .param = "heads"},
    ValueRule{.key = "attention.head_count_kv",
              .origin = ValueOrigin::kComponent,
              .kind = "attention.qsa",
              .param = "kv_heads"},
    ValueRule{.key = "attention.key_length",
              .origin = ValueOrigin::kComponent,
              .kind = "attention.qsa",
              .param = "head_dim"},
    ValueRule{.key = "attention.value_length",
              .origin = ValueOrigin::kComponent,
              .kind = "attention.qsa",
              .param = "head_dim"},
    ValueRule{.key = "rope.dimension_count",
              .origin = ValueOrigin::kComponent,
              .kind = "attention.qsa",
              .param = "rope_dims"},
    ValueRule{.key = "rope.freq_base",
              .origin = ValueOrigin::kComponent,
              .kind = "attention.qsa",
              .param = "rope_base"},
    ValueRule{.key = "rope.dimension_sections",
              .origin = ValueOrigin::kComponent,
              .kind = "attention.qsa",
              .param = "rope_sections"},
    ValueRule{.key = "attention.indexer.head_count",
              .origin = ValueOrigin::kComponent,
              .kind = "attention.qsa",
              .param = "indexer_heads"},
    ValueRule{.key = "attention.indexer.key_length",
              .origin = ValueOrigin::kComponent,
              .kind = "attention.qsa",
              .param = "indexer_dim"},
    ValueRule{.key = "attention.indexer.top_k",
              .origin = ValueOrigin::kComponent,
              .kind = "attention.qsa",
              .param = "budget"},
    ValueRule{.key = "attention.compress_ratios",
              .origin = ValueOrigin::kComponent,
              .kind = "attention.qsa",
              .param = "ratio",
              .list = true},
    ValueRule{.key = "expert_count",
              .origin = ValueOrigin::kComponent,
              .kind = "ffn.moe",
              .param = "experts"},
    ValueRule{.key = "expert_used_count",
              .origin = ValueOrigin::kComponent,
              .kind = "ffn.moe",
              .param = "used"},
    ValueRule{.key = "expert_feed_forward_length",
              .origin = ValueOrigin::kComponent,
              .kind = "ffn.moe",
              .param = "expert_hidden"},
    ValueRule{.key = "expert_shared_feed_forward_length",
              .origin = ValueOrigin::kComponent,
              .kind = "ffn.moe",
              .param = "hidden"},
    ValueRule{.key = "block_count",
              .origin = ValueOrigin::kLayerCount,
              .tensor = "blk.{layer}.attn_q.weight"},
    ValueRule{.key = kIndexer, .value = {.text = "fused"}, .type = kString},
    // The drafter's own expert layout: a relaid ModelOpt drafter is stacked
    // and scaled whatever its target's (GGUF and MLX targets are neither).
    ValueRule{.key = kStacked,
              .origin = ValueOrigin::kTensorPresent,
              .tensor = "blk.{layer}.ffn_gate_up_exps.weight",
              .type = kBool},
    ValueRule{.key = kScaled,
              .origin = ValueOrigin::kTensorPresent,
              .tensor = "blk.{layer}.ffn_gate_exps.weight_scale_2",
              .type = kBool},
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
                                      .target_architecture = "qwen4exp",
                                      .values = kMtpValues,
                                      .head_ids = kHeadIds,
                                      .vocab_key = kRecipeDraftVocab};
constexpr std::array<std::string_view, 7> kMtpNativeIgnoredPrefixes = {
    "lm_head.", "language_model.lm_head.", "model.visual.",         "vision_tower.",
    "visual.",  "model.language_model.",   "language_model.model.",
};
constexpr std::array kMtpNativeKeyAliases = {
    KeyAlias{"text_config.indexer_kv_heads", "attention.indexer.head_count_kv"},
    KeyAlias{"indexer_kv_heads", "attention.indexer.head_count_kv"},
    KeyAlias{"text_config.rope_parameters.mrope_interleaved", "rope.mrope_interleaved"},
    KeyAlias{"rope_parameters.mrope_interleaved", "rope.mrope_interleaved"},
    KeyAlias{"text_config.use_bidirectional_attention", "use_bidirectional_attention"},
    KeyAlias{"text_config.mtp_num_hidden_layers", "block_count"},
    KeyAlias{"mtp_num_hidden_layers", "block_count"},
    KeyAlias{"text_config.max_position_embeddings", "context_length"},
    KeyAlias{"max_position_embeddings", "context_length"},
    KeyAlias{"text_config.hidden_size", "embedding_length"},
    KeyAlias{"hidden_size", "embedding_length"},
    KeyAlias{"text_config.hc_count", "hyper_connection.count"},
    KeyAlias{"hc_count", "hyper_connection.count"},
    KeyAlias{"text_config.hc_lowrank", "hyper_connection.low_rank"},
    KeyAlias{"hc_lowrank", "hyper_connection.low_rank"},
    KeyAlias{"text_config.rms_norm_eps", "attention.layer_norm_rms_epsilon"},
    KeyAlias{"rms_norm_eps", "attention.layer_norm_rms_epsilon"},
    KeyAlias{"text_config.num_attention_heads", "attention.head_count"},
    KeyAlias{"num_attention_heads", "attention.head_count"},
    KeyAlias{"text_config.num_key_value_heads", "attention.head_count_kv"},
    KeyAlias{"num_key_value_heads", "attention.head_count_kv"},
    KeyAlias{"text_config.head_dim", "attention.key_length"},
    KeyAlias{"head_dim", "attention.key_length"},
    KeyAlias{"text_config.rope_parameters.rope_theta", "rope.freq_base"},
    KeyAlias{"rope_parameters.rope_theta", "rope.freq_base"},
    KeyAlias{"rope_theta", "rope.freq_base"},
    KeyAlias{"text_config.rope_parameters.mrope_section", "rope.dimension_sections"},
    KeyAlias{"rope_parameters.mrope_section", "rope.dimension_sections"},
    KeyAlias{"mrope_section", "rope.dimension_sections"},
    KeyAlias{"text_config.rope_parameters.partial_rotary_factor", "rope.partial_rotary_factor"},
    KeyAlias{"rope_parameters.partial_rotary_factor", "rope.partial_rotary_factor"},
    KeyAlias{"text_config.partial_rotary_factor", "rope.partial_rotary_factor"},
    KeyAlias{"partial_rotary_factor", "rope.partial_rotary_factor"},
    KeyAlias{"text_config.indexer_n_heads", "attention.indexer.head_count"},
    KeyAlias{"indexer_n_heads", "attention.indexer.head_count"},
    KeyAlias{"text_config.indexer_head_dim", "attention.indexer.key_length"},
    KeyAlias{"indexer_head_dim", "attention.indexer.key_length"},
    KeyAlias{"text_config.indexer_budget", "attention.indexer.top_k"},
    KeyAlias{"indexer_budget", "attention.indexer.top_k"},
    KeyAlias{"text_config.indexer_compress_ratio", "attention.compress_ratios"},
    KeyAlias{"indexer_compress_ratio", "attention.compress_ratios"},
    KeyAlias{"text_config.num_experts", "expert_count"},
    KeyAlias{"num_experts", "expert_count"},
    KeyAlias{"text_config.num_experts_per_tok", "expert_used_count"},
    KeyAlias{"num_experts_per_tok", "expert_used_count"},
    KeyAlias{"text_config.moe_intermediate_size", "expert_feed_forward_length"},
    KeyAlias{"moe_intermediate_size", "expert_feed_forward_length"},
    KeyAlias{"text_config.shared_expert_intermediate_size", "expert_shared_feed_forward_length"},
    KeyAlias{"shared_expert_intermediate_size", "expert_shared_feed_forward_length"},
    KeyAlias{"text_config.mamba_ssm_dtype", "mamba_ssm_dtype"},
    KeyAlias{"mamba_ssm_dtype", "mamba_ssm_dtype"},
    KeyAlias{"text_config.dtype", "dtype"},
    KeyAlias{"dtype", "dtype"},
    KeyAlias{"text_config.is_encoder_decoder", "is_encoder_decoder"},
    KeyAlias{"is_encoder_decoder", "is_encoder_decoder"},
    KeyAlias{"text_config.hidden_act", "hidden_act"},
    KeyAlias{"hidden_act", "hidden_act"},
    KeyAlias{"text_config.attention_bias", "attention_bias"},
    KeyAlias{"attention_bias", "attention_bias"},
    KeyAlias{"text_config.output_gate_type", "output_gate_type"},
    KeyAlias{"output_gate_type", "output_gate_type"},
    KeyAlias{"text_config.rope_parameters.rope_type", "rope.scaling.type"},
    KeyAlias{"rope_parameters.rope_type", "rope.scaling.type"},
    KeyAlias{"text_config.rope_parameters.type", "rope.scaling.type"},
    KeyAlias{"rope_parameters.type", "rope.scaling.type"},
};
constexpr std::array kQwen38TargetNativeAuxiliary = {
    AuxiliaryNamespace{"eos_token_id",
                       "generation stop framing when nested text EOS supplies the PLE hash"},
    AuxiliaryNamespace{"vision_config.", "multimodal vision encoder outside the text pipeline"},
    AuxiliaryNamespace{"quantization.", "checkpoint quantization metadata handled by reader"},
    AuxiliaryNamespace{"quantization_config.",
                       "checkpoint quantization metadata handled by reader"},
    AuxiliaryNamespace{"text_config.vocab_size", "vocabulary row count stated by tensor extent"},
    AuxiliaryNamespace{"vocab_size", "vocabulary row count stated by tensor extent"},
    AuxiliaryNamespace{"text_config.tie_word_embeddings", "output tying stated by tensor presence"},
    AuxiliaryNamespace{"tie_word_embeddings", "output tying stated by tensor presence"},
    AuxiliaryNamespace{"text_config.bos_token_id", "tokenizer framing outside model forward math"},
    AuxiliaryNamespace{"bos_token_id", "tokenizer framing outside model forward math"},
    AuxiliaryNamespace{"text_config.pad_token_id", "tokenizer framing outside model forward math"},
    AuxiliaryNamespace{"pad_token_id", "tokenizer framing outside model forward math"},
    AuxiliaryNamespace{"text_config.attention_dropout", "training dropout outside inference"},
    AuxiliaryNamespace{"attention_dropout", "training dropout outside inference"},
    AuxiliaryNamespace{"text_config.make_ngram_vocab_size_divisible_by",
                       "PLE vocab construction parameter already baked into head_vocab_sizes"},
    AuxiliaryNamespace{"make_ngram_vocab_size_divisible_by",
                       "PLE vocab construction parameter already baked into head_vocab_sizes"},
    AuxiliaryNamespace{"text_config.ngram_vocab_size_base",
                       "PLE vocab construction parameter already baked into head_vocab_sizes"},
    AuxiliaryNamespace{"ngram_vocab_size_base",
                       "PLE vocab construction parameter already baked into head_vocab_sizes"},
    AuxiliaryNamespace{"text_config.ple_embedding_dtype",
                       "PLE table dtype stated by tensor format"},
    AuxiliaryNamespace{"ple_embedding_dtype", "PLE table dtype stated by tensor format"},
    AuxiliaryNamespace{"text_config.split_ngram_parts", "PLE shard count stated by tensor names"},
    AuxiliaryNamespace{"split_ngram_parts", "PLE shard count stated by tensor names"},
    AuxiliaryNamespace{"text_config.mtp.", "MTP drafter configuration"},
    AuxiliaryNamespace{"mtp.", "MTP drafter configuration"},
    AuxiliaryNamespace{"text_config.mtp_num_hidden_layers", "MTP drafter layer count"},
    AuxiliaryNamespace{"mtp_num_hidden_layers", "MTP drafter layer count"},
    AuxiliaryNamespace{"text_config.model_type", "nested text configuration model_type"},
};
constexpr std::array kQwen38MtpNativeAuxiliary = {
    AuxiliaryNamespace{"full_attention_interval",
                       "the target stack schedule, outside the single-layer drafter"},
    AuxiliaryNamespace{"vision_config.", "multimodal vision encoder outside the text pipeline"},
    AuxiliaryNamespace{"quantization.", "checkpoint quantization metadata handled by reader"},
    AuxiliaryNamespace{"quantization_config.",
                       "checkpoint quantization metadata handled by reader"},
    AuxiliaryNamespace{"text_config.vocab_size", "vocabulary row count stated by tensor extent"},
    AuxiliaryNamespace{"vocab_size", "vocabulary row count stated by tensor extent"},
    AuxiliaryNamespace{"text_config.tie_word_embeddings", "output tying stated by tensor presence"},
    AuxiliaryNamespace{"tie_word_embeddings", "output tying stated by tensor presence"},
    AuxiliaryNamespace{"text_config.bos_token_id", "tokenizer framing outside model forward math"},
    AuxiliaryNamespace{"bos_token_id", "tokenizer framing outside model forward math"},
    AuxiliaryNamespace{"text_config.eos_token_id", "tokenizer framing outside model forward math"},
    AuxiliaryNamespace{"eos_token_id", "tokenizer framing outside model forward math"},
    AuxiliaryNamespace{"text_config.pad_token_id", "tokenizer framing outside model forward math"},
    AuxiliaryNamespace{"pad_token_id", "tokenizer framing outside model forward math"},
    AuxiliaryNamespace{"text_config.attention_dropout", "training dropout outside inference"},
    AuxiliaryNamespace{"attention_dropout", "training dropout outside inference"},
    AuxiliaryNamespace{"text_config.linear_conv_kernel_dim", "target GDN parameter"},
    AuxiliaryNamespace{"linear_conv_kernel_dim", "target GDN parameter"},
    AuxiliaryNamespace{"text_config.linear_key_head_dim", "target GDN parameter"},
    AuxiliaryNamespace{"linear_key_head_dim", "target GDN parameter"},
    AuxiliaryNamespace{"text_config.linear_num_key_heads", "target GDN parameter"},
    AuxiliaryNamespace{"linear_num_key_heads", "target GDN parameter"},
    AuxiliaryNamespace{"text_config.linear_num_value_heads", "target GDN parameter"},
    AuxiliaryNamespace{"linear_num_value_heads", "target GDN parameter"},
    AuxiliaryNamespace{"text_config.linear_value_head_dim", "target GDN parameter"},
    AuxiliaryNamespace{"linear_value_head_dim", "target GDN parameter"},
    AuxiliaryNamespace{"text_config.make_ngram_vocab_size_divisible_by", "target PLE parameter"},
    AuxiliaryNamespace{"make_ngram_vocab_size_divisible_by", "target PLE parameter"},
    AuxiliaryNamespace{"text_config.ngram_vocab_size_base", "target PLE parameter"},
    AuxiliaryNamespace{"ngram_vocab_size_base", "target PLE parameter"},
    AuxiliaryNamespace{"text_config.heads_per_ngram", "target PLE parameter"},
    AuxiliaryNamespace{"heads_per_ngram", "target PLE parameter"},
    AuxiliaryNamespace{"text_config.ngram_size", "target PLE parameter"},
    AuxiliaryNamespace{"ngram_size", "target PLE parameter"},
    AuxiliaryNamespace{"text_config.ple_conv_kernel_size", "target PLE parameter"},
    AuxiliaryNamespace{"ple_conv_kernel_size", "target PLE parameter"},
    AuxiliaryNamespace{"text_config.ple_embed_dim", "target PLE parameter"},
    AuxiliaryNamespace{"ple_embed_dim", "target PLE parameter"},
    AuxiliaryNamespace{"text_config.ple_embedding_dtype", "target PLE parameter"},
    AuxiliaryNamespace{"ple_embedding_dtype", "target PLE parameter"},
    AuxiliaryNamespace{"text_config.ple_layer_ids", "target PLE parameter"},
    AuxiliaryNamespace{"ple_layer_ids", "target PLE parameter"},
    AuxiliaryNamespace{"text_config.split_ngram_parts", "target PLE parameter"},
    AuxiliaryNamespace{"split_ngram_parts", "target PLE parameter"},
    AuxiliaryNamespace{"text_config.layer_types", "target layer types"},
    AuxiliaryNamespace{"layer_types", "target layer types"},
    AuxiliaryNamespace{"text_config.num_hidden_layers", "target layer count"},
    AuxiliaryNamespace{"num_hidden_layers", "target layer count"},
    AuxiliaryNamespace{"text_config.mtp.", "MTP drafter nested config"},
    AuxiliaryNamespace{"mtp.", "MTP drafter nested config"},
    AuxiliaryNamespace{"text_config.model_type", "nested text configuration model_type"},
};
constexpr std::array kMtpNativeTensorAliases = {
    TensorAlias{"mtp.fc_embedding.weight", "fc_embd.weight"},
    TensorAlias{"mtp.fc_hidden.weight", "fc_hidden.weight"},
    TensorAlias{"mtp.pre_fc_norm_embedding.weight", "norm_embd.weight"},
    TensorAlias{"mtp.pre_fc_norm_hidden.weight", "norm_hidden.weight"},
    TensorAlias{"mtp.hyper_connection_mixer.hc_norm.weight", "output_hc_norm.weight"},
    TensorAlias{"mtp.hyper_connection_mixer.input_mix_weight_down.weight", "output_hc_down.weight"},
    TensorAlias{"mtp.hyper_connection_mixer.input_mix_weight_up.weight", "output_hc_up.weight"},
    TensorAlias{"mtp.layers.{layer}.attn_hyper_connection.block_inject_weight.weight",
                "blk.{layer}.hc_attn_inject.weight"},
    TensorAlias{"mtp.layers.{layer}.attn_hyper_connection.hc_norm.weight",
                "blk.{layer}.hc_attn_norm.weight"},
    TensorAlias{"mtp.layers.{layer}.attn_hyper_connection.input_mix_weight_down.weight",
                "blk.{layer}.hc_attn_down.weight"},
    TensorAlias{"mtp.layers.{layer}.attn_hyper_connection.input_mix_weight_up.weight",
                "blk.{layer}.hc_attn_up.weight"},
    TensorAlias{"mtp.layers.{layer}.mlp_hyper_connection.block_inject_weight.weight",
                "blk.{layer}.hc_ffn_inject.weight"},
    TensorAlias{"mtp.layers.{layer}.mlp_hyper_connection.hc_norm.weight",
                "blk.{layer}.hc_ffn_norm.weight"},
    TensorAlias{"mtp.layers.{layer}.mlp_hyper_connection.input_mix_weight_down.weight",
                "blk.{layer}.hc_ffn_down.weight"},
    TensorAlias{"mtp.layers.{layer}.mlp_hyper_connection.input_mix_weight_up.weight",
                "blk.{layer}.hc_ffn_up.weight"},
    TensorAlias{"mtp.layers.{layer}.self_attn.q_proj.weight", "blk.{layer}.attn_q.weight"},
    TensorAlias{"mtp.layers.{layer}.self_attn.k_proj.weight", "blk.{layer}.attn_k.weight"},
    TensorAlias{"mtp.layers.{layer}.self_attn.v_proj.weight", "blk.{layer}.attn_v.weight"},
    TensorAlias{"mtp.layers.{layer}.self_attn.o_proj.weight", "blk.{layer}.attn_output.weight"},
    TensorAlias{"mtp.layers.{layer}.self_attn.q_norm.weight", "blk.{layer}.attn_q_norm.weight"},
    TensorAlias{"mtp.layers.{layer}.self_attn.k_norm.weight", "blk.{layer}.attn_k_norm.weight"},
    TensorAlias{"mtp.layers.{layer}.self_attn.indexer.index_qk_proj.weight",
                "blk.{layer}.indexer.qk_proj.weight"},
    TensorAlias{"mtp.layers.{layer}.self_attn.indexer.q_layernorm.weight",
                "blk.{layer}.indexer.q_norm.weight"},
    TensorAlias{"mtp.layers.{layer}.self_attn.indexer.k_layernorm.weight",
                "blk.{layer}.indexer.k_norm.weight"},
    TensorAlias{"mtp.layers.{layer}.mlp.gate.weight", "blk.{layer}.ffn_gate_inp.weight"},
    TensorAlias{"mtp.layers.{layer}.mlp.shared_expert_gate.weight",
                "blk.{layer}.ffn_gate_inp_shexp.weight"},
    TensorAlias{"mtp.layers.{layer}.mlp.shared_expert.gate_proj.weight",
                "blk.{layer}.ffn_gate_shexp.weight"},
    TensorAlias{"mtp.layers.{layer}.mlp.shared_expert.up_proj.weight",
                "blk.{layer}.ffn_up_shexp.weight"},
    TensorAlias{"mtp.layers.{layer}.mlp.shared_expert.down_proj.weight",
                "blk.{layer}.ffn_down_shexp.weight"},
    TensorAlias{"mtp.layers.{layer}.mlp.experts.{expert}.gate_proj.weight",
                "blk.{layer}.ffn_gate_exps.weight"},
    TensorAlias{"mtp.layers.{layer}.mlp.experts.{expert}.gate_proj.weight_scale",
                "blk.{layer}.ffn_gate_exps.weight_scale"},
    TensorAlias{"mtp.layers.{layer}.mlp.experts.{expert}.gate_proj.weight_scale_2",
                "blk.{layer}.ffn_gate_exps.weight_scale_2"},
    TensorAlias{"mtp.layers.{layer}.mlp.experts.{expert}.up_proj.weight",
                "blk.{layer}.ffn_up_exps.weight"},
    TensorAlias{"mtp.layers.{layer}.mlp.experts.{expert}.up_proj.weight_scale",
                "blk.{layer}.ffn_up_exps.weight_scale"},
    TensorAlias{"mtp.layers.{layer}.mlp.experts.{expert}.up_proj.weight_scale_2",
                "blk.{layer}.ffn_up_exps.weight_scale_2"},
    TensorAlias{"mtp.layers.{layer}.mlp.experts.{expert}.down_proj.weight",
                "blk.{layer}.ffn_down_exps.weight"},
    TensorAlias{"mtp.layers.{layer}.mlp.experts.{expert}.down_proj.weight_scale",
                "blk.{layer}.ffn_down_exps.weight_scale"},
    TensorAlias{"mtp.layers.{layer}.mlp.experts.{expert}.down_proj.weight_scale_2",
                "blk.{layer}.ffn_down_exps.weight_scale_2"},
    TensorAlias{"language_model.mtp.fc_embedding.weight", "fc_embd.weight"},
    TensorAlias{"language_model.mtp.fc_embedding.scales", "fc_embd.weight_scale"},
    TensorAlias{"language_model.mtp.fc_embedding.biases", "fc_embd.weight_bias"},
    TensorAlias{"language_model.mtp.fc_hidden.weight", "fc_hidden.weight"},
    TensorAlias{"language_model.mtp.fc_hidden.scales", "fc_hidden.weight_scale"},
    TensorAlias{"language_model.mtp.fc_hidden.biases", "fc_hidden.weight_bias"},
    TensorAlias{"language_model.mtp.pre_fc_norm_embedding.weight", "norm_embd.weight"},
    TensorAlias{"language_model.mtp.pre_fc_norm_hidden.weight", "norm_hidden.weight"},
    TensorAlias{"language_model.mtp.hyper_connection_mixer.hc_norm.weight",
                "output_hc_norm.weight"},
    TensorAlias{"language_model.mtp.hyper_connection_mixer.input_mix_weight_down.weight",
                "output_hc_down.weight"},
    TensorAlias{"language_model.mtp.hyper_connection_mixer.input_mix_weight_down.scales",
                "output_hc_down.weight_scale"},
    TensorAlias{"language_model.mtp.hyper_connection_mixer.input_mix_weight_down.biases",
                "output_hc_down.weight_bias"},
    TensorAlias{"language_model.mtp.hyper_connection_mixer.input_mix_weight_up.weight",
                "output_hc_up.weight"},
    TensorAlias{"language_model.mtp.hyper_connection_mixer.input_mix_weight_up.scales",
                "output_hc_up.weight_scale"},
    TensorAlias{"language_model.mtp.hyper_connection_mixer.input_mix_weight_up.biases",
                "output_hc_up.weight_bias"},
    TensorAlias{
        "language_model.mtp.layers.{layer}.attn_hyper_connection.block_inject_weight.weight",
        "blk.{layer}.hc_attn_inject.weight"},
    TensorAlias{
        "language_model.mtp.layers.{layer}.attn_hyper_connection.block_inject_weight.scales",
        "blk.{layer}.hc_attn_inject.weight_scale"},
    TensorAlias{
        "language_model.mtp.layers.{layer}.attn_hyper_connection.block_inject_weight.biases",
        "blk.{layer}.hc_attn_inject.weight_bias"},
    TensorAlias{"language_model.mtp.layers.{layer}.attn_hyper_connection.hc_norm.weight",
                "blk.{layer}.hc_attn_norm.weight"},
    TensorAlias{
        "language_model.mtp.layers.{layer}.attn_hyper_connection.input_mix_weight_down.weight",
        "blk.{layer}.hc_attn_down.weight"},
    TensorAlias{
        "language_model.mtp.layers.{layer}.attn_hyper_connection.input_mix_weight_down.scales",
        "blk.{layer}.hc_attn_down.weight_scale"},
    TensorAlias{
        "language_model.mtp.layers.{layer}.attn_hyper_connection.input_mix_weight_down.biases",
        "blk.{layer}.hc_attn_down.weight_bias"},
    TensorAlias{
        "language_model.mtp.layers.{layer}.attn_hyper_connection.input_mix_weight_up.weight",
        "blk.{layer}.hc_attn_up.weight"},
    TensorAlias{
        "language_model.mtp.layers.{layer}.attn_hyper_connection.input_mix_weight_up.scales",
        "blk.{layer}.hc_attn_up.weight_scale"},
    TensorAlias{
        "language_model.mtp.layers.{layer}.attn_hyper_connection.input_mix_weight_up.biases",
        "blk.{layer}.hc_attn_up.weight_bias"},
    TensorAlias{"language_model.mtp.layers.{layer}.mlp_hyper_connection.block_inject_weight.weight",
                "blk.{layer}.hc_ffn_inject.weight"},
    TensorAlias{"language_model.mtp.layers.{layer}.mlp_hyper_connection.block_inject_weight.scales",
                "blk.{layer}.hc_ffn_inject.weight_scale"},
    TensorAlias{"language_model.mtp.layers.{layer}.mlp_hyper_connection.block_inject_weight.biases",
                "blk.{layer}.hc_ffn_inject.weight_bias"},
    TensorAlias{"language_model.mtp.layers.{layer}.mlp_hyper_connection.hc_norm.weight",
                "blk.{layer}.hc_ffn_norm.weight"},
    TensorAlias{
        "language_model.mtp.layers.{layer}.mlp_hyper_connection.input_mix_weight_down.weight",
        "blk.{layer}.hc_ffn_down.weight"},
    TensorAlias{
        "language_model.mtp.layers.{layer}.mlp_hyper_connection.input_mix_weight_down.scales",
        "blk.{layer}.hc_ffn_down.weight_scale"},
    TensorAlias{
        "language_model.mtp.layers.{layer}.mlp_hyper_connection.input_mix_weight_down.biases",
        "blk.{layer}.hc_ffn_down.weight_bias"},
    TensorAlias{"language_model.mtp.layers.{layer}.mlp_hyper_connection.input_mix_weight_up.weight",
                "blk.{layer}.hc_ffn_up.weight"},
    TensorAlias{"language_model.mtp.layers.{layer}.mlp_hyper_connection.input_mix_weight_up.scales",
                "blk.{layer}.hc_ffn_up.weight_scale"},
    TensorAlias{"language_model.mtp.layers.{layer}.mlp_hyper_connection.input_mix_weight_up.biases",
                "blk.{layer}.hc_ffn_up.weight_bias"},
    TensorAlias{"language_model.mtp.layers.{layer}.self_attn.k_norm.weight",
                "blk.{layer}.attn_k_norm.weight"},
    TensorAlias{"language_model.mtp.layers.{layer}.self_attn.k_proj.weight",
                "blk.{layer}.attn_k.weight"},
    TensorAlias{"language_model.mtp.layers.{layer}.self_attn.k_proj.scales",
                "blk.{layer}.attn_k.weight_scale"},
    TensorAlias{"language_model.mtp.layers.{layer}.self_attn.k_proj.biases",
                "blk.{layer}.attn_k.weight_bias"},
    TensorAlias{"language_model.mtp.layers.{layer}.self_attn.o_proj.weight",
                "blk.{layer}.attn_output.weight"},
    TensorAlias{"language_model.mtp.layers.{layer}.self_attn.o_proj.scales",
                "blk.{layer}.attn_output.weight_scale"},
    TensorAlias{"language_model.mtp.layers.{layer}.self_attn.o_proj.biases",
                "blk.{layer}.attn_output.weight_bias"},
    TensorAlias{"language_model.mtp.layers.{layer}.self_attn.q_norm.weight",
                "blk.{layer}.attn_q_norm.weight"},
    TensorAlias{"language_model.mtp.layers.{layer}.self_attn.q_proj.weight",
                "blk.{layer}.attn_q.weight"},
    TensorAlias{"language_model.mtp.layers.{layer}.self_attn.q_proj.scales",
                "blk.{layer}.attn_q.weight_scale"},
    TensorAlias{"language_model.mtp.layers.{layer}.self_attn.q_proj.biases",
                "blk.{layer}.attn_q.weight_bias"},
    TensorAlias{"language_model.mtp.layers.{layer}.self_attn.v_proj.weight",
                "blk.{layer}.attn_v.weight"},
    TensorAlias{"language_model.mtp.layers.{layer}.self_attn.v_proj.scales",
                "blk.{layer}.attn_v.weight_scale"},
    TensorAlias{"language_model.mtp.layers.{layer}.self_attn.v_proj.biases",
                "blk.{layer}.attn_v.weight_bias"},
    TensorAlias{"language_model.mtp.layers.{layer}.self_attn.indexer.k_layernorm.weight",
                "blk.{layer}.indexer.k_norm.weight"},
    TensorAlias{"language_model.mtp.layers.{layer}.self_attn.indexer.q_layernorm.weight",
                "blk.{layer}.indexer.q_norm.weight"},
    TensorAlias{"language_model.mtp.layers.{layer}.self_attn.indexer.index_qk_proj.weight",
                "blk.{layer}.indexer.qk_proj.weight"},
    TensorAlias{"language_model.mtp.layers.{layer}.self_attn.indexer.index_qk_proj.scales",
                "blk.{layer}.indexer.qk_proj.weight_scale"},
    TensorAlias{"language_model.mtp.layers.{layer}.self_attn.indexer.index_qk_proj.biases",
                "blk.{layer}.indexer.qk_proj.weight_bias"},
    TensorAlias{"language_model.mtp.layers.{layer}.mlp.gate.weight",
                "blk.{layer}.ffn_gate_inp.weight"},
    TensorAlias{"language_model.mtp.layers.{layer}.mlp.shared_expert_gate.weight",
                "blk.{layer}.ffn_gate_inp_shexp.weight"},
    TensorAlias{"language_model.mtp.layers.{layer}.mlp.shared_expert_gate.scales",
                "blk.{layer}.ffn_gate_inp_shexp.weight_scale"},
    TensorAlias{"language_model.mtp.layers.{layer}.mlp.shared_expert_gate.biases",
                "blk.{layer}.ffn_gate_inp_shexp.weight_bias"},
    TensorAlias{"language_model.mtp.layers.{layer}.mlp.shared_expert.down_proj.weight",
                "blk.{layer}.ffn_down_shexp.weight"},
    TensorAlias{"language_model.mtp.layers.{layer}.mlp.shared_expert.down_proj.scales",
                "blk.{layer}.ffn_down_shexp.weight_scale"},
    TensorAlias{"language_model.mtp.layers.{layer}.mlp.shared_expert.down_proj.biases",
                "blk.{layer}.ffn_down_shexp.weight_bias"},
    TensorAlias{"language_model.mtp.layers.{layer}.mlp.shared_expert.gate_proj.weight",
                "blk.{layer}.ffn_gate_shexp.weight"},
    TensorAlias{"language_model.mtp.layers.{layer}.mlp.shared_expert.gate_proj.scales",
                "blk.{layer}.ffn_gate_shexp.weight_scale"},
    TensorAlias{"language_model.mtp.layers.{layer}.mlp.shared_expert.gate_proj.biases",
                "blk.{layer}.ffn_gate_shexp.weight_bias"},
    TensorAlias{"language_model.mtp.layers.{layer}.mlp.shared_expert.up_proj.weight",
                "blk.{layer}.ffn_up_shexp.weight"},
    TensorAlias{"language_model.mtp.layers.{layer}.mlp.shared_expert.up_proj.scales",
                "blk.{layer}.ffn_up_shexp.weight_scale"},
    TensorAlias{"language_model.mtp.layers.{layer}.mlp.shared_expert.up_proj.biases",
                "blk.{layer}.ffn_up_shexp.weight_bias"},
    TensorAlias{"language_model.mtp.layers.{layer}.mlp.switch_mlp.down_proj.weight",
                "blk.{layer}.ffn_down_exps.weight"},
    TensorAlias{"language_model.mtp.layers.{layer}.mlp.switch_mlp.down_proj.scales",
                "blk.{layer}.ffn_down_exps.weight_scale"},
    TensorAlias{"language_model.mtp.layers.{layer}.mlp.switch_mlp.down_proj.biases",
                "blk.{layer}.ffn_down_exps.weight_bias"},
    TensorAlias{"language_model.mtp.layers.{layer}.mlp.switch_mlp.gate_proj.weight",
                "blk.{layer}.ffn_gate_exps.weight"},
    TensorAlias{"language_model.mtp.layers.{layer}.mlp.switch_mlp.gate_proj.scales",
                "blk.{layer}.ffn_gate_exps.weight_scale"},
    TensorAlias{"language_model.mtp.layers.{layer}.mlp.switch_mlp.gate_proj.biases",
                "blk.{layer}.ffn_gate_exps.weight_bias"},
    TensorAlias{"language_model.mtp.layers.{layer}.mlp.switch_mlp.up_proj.weight",
                "blk.{layer}.ffn_up_exps.weight"},
    TensorAlias{"language_model.mtp.layers.{layer}.mlp.switch_mlp.up_proj.scales",
                "blk.{layer}.ffn_up_exps.weight_scale"},
    TensorAlias{"language_model.mtp.layers.{layer}.mlp.switch_mlp.up_proj.biases",
                "blk.{layer}.ffn_up_exps.weight_bias"},
};
// A speculative stage's routed experts take the W4A4 grouped GEMM (its
// moe_experts: activations quantized to FP4 with a per-step dynamic scale,
// row amax / (6 * 448), as v0's did) whatever its checkpoint's activation
// quantization: the target's verification decides every token, so the
// drafter's numerics change only acceptance and speed (AGENTS.md rule 11,
// speed over bit-exactness at the same quality), and on spark-b it measured
// 1.2 to 2.0% faster decode than W4A16 at equal acceptance
// (experiments/m36-numerical-audit/qwen-modelopt-v1.md). The converted
// drafter's policy (kMtpMapping's, which the mapping below keeps); a device
// without FP4 falls back to the format ladder.
constexpr auto kQwen38NativeSemantic = [] {
  std::array<std::string_view, kHfSemantic.size() + 4> out{};
  std::ranges::copy(kHfSemantic, out.begin());
  out[kHfSemantic.size()] = "linear_*";
  out[kHfSemantic.size() + 1] = "indexer_*";
  out[kHfSemantic.size() + 2] = "hc_*";
  out[kHfSemantic.size() + 3] = "zz_*";
  return out;
}();
// Text import uses equal positions on every axis, so either MRoPE axis
// ordering produces the same frequencies; section geometry still validates.
constexpr std::array kQwen38NativeRequirements = {
    ScalarRequirement{.key = "rope.mrope_interleaved", .optional = true, .boolean = true},
    ScalarRequirement{
        .key = "ssm.value_head_dim", .equals_key = "ssm.state_size", .optional = true},
    ScalarRequirement{
        .key = "attention.value_length", .equals_key = "attention.key_length", .optional = true},
};
constexpr std::array kQwen38MtpNativeRequirements = {
    ScalarRequirement{.key = "rope.mrope_interleaved", .optional = true, .boolean = true},
    ScalarRequirement{.key = "block_count", .reject_equal = 1, .require_equal = true},
};
constexpr std::array kQwen38NativeLists = {
    ListRequirement{.key = "ple.layer_ids", .size = kLayerLimit, .min_size = 1, .optional = true},
    ListRequirement{.key = "ple.layers", .size = kLayerLimit, .min_size = 1, .optional = true},
    ListRequirement{.key = "rope.dimension_sections",
                    .size = 4,
                    .min_size = 3,
                    .sections_of = "attention.key_length",
                    .sections_multiplier_key = "rope.partial_rotary_factor"},
    ListRequirement{.key = "layer_types",
                    .size = kLayerLimit,
                    .min_size = 1,
                    .optional = true,
                    .per_layer = true},
};
constexpr std::array kQwen38MtpNativeLists = {
    ListRequirement{.key = "rope.dimension_sections",
                    .size = 4,
                    .min_size = 3,
                    .sections_of = "attention.key_length",
                    .sections_multiplier_key = "rope.partial_rotary_factor"},
    ListRequirement{.key = "attention.compress_ratios",
                    .size = kLayerLimit,
                    .min_size = 1,
                    .optional = true,
                    .per_layer = true},
};
constexpr auto kMtpNativeMapping = [] {
  auto mapping = kMtpMapping;
  mapping.containers = kNativeContainers;
  mapping.source_architectures = kNativeSourceArchitectures;
  mapping.source_key_prefix = "text_config.";
  mapping.draft = &kMtpNativeDraft;
  mapping.tensors = kMtpNativeTensors;
  mapping.params = kMtpNativeParams;
  mapping.folds = kNativeFolds;
  mapping.tensor_aliases = kMtpNativeTensorAliases;
  mapping.key_aliases = kMtpNativeKeyAliases;
  mapping.ignored_tensor_prefixes = kMtpNativeIgnoredPrefixes;
  mapping.auxiliary = kQwen38MtpNativeAuxiliary;
  mapping.requirements = kQwen38MtpNativeRequirements;
  mapping.lists = kQwen38MtpNativeLists;
  mapping.semantic = kQwen38NativeSemantic;
  mapping.extended_components = true;
  return mapping;
}();
constexpr auto kNativeSlots = [] {
  auto slots = kSlots;
  for (auto& slot : slots)
    if (slot.layers_key == kPleLayers) {
      slot.layers_alias = "ple.layer_ids";
      slot.alias_index_base = 1;
    }
  return slots;
}();
constexpr auto kNativeGates = [] {
  auto gates = kGates;
  for (auto& gate : gates)
    if (gate.param == "projection_scales") gate.param = "gate_scale";
  return gates;
}();
constexpr auto kNativeMapping = [] {
  auto mapping = kMapping;
  mapping.containers = kNativeContainers;
  mapping.gates = kNativeGates;
  mapping.source_architectures = kNativeSourceArchitectures;
  mapping.source_key_prefix = "text_config.";
  mapping.context_key = "context_length";
  mapping.layer_flags_key = {};
  mapping.layer_type_key = "layer_types";
  mapping.layer_type_interval_key = "full_attention_interval";
  mapping.layer_type_codes = kNativeLayerTypeCodes;
  mapping.semantic_tensors = kNativeSemanticTensors;
  mapping.slots = kNativeSlots;
  mapping.tensors = kNativeTensors;
  mapping.params = kNativeParams;
  mapping.folds = kNativeFolds;
  mapping.policies = kNativePolicies;
  mapping.tensor_aliases = kNativeTensorAliases;
  mapping.key_aliases = kNativeKeyAliases;
  mapping.ignored_tensors = kNativeIgnoredTensors;
  mapping.ignored_tensor_prefixes = kNativeIgnoredPrefixes;
  mapping.auxiliary = kQwen38TargetNativeAuxiliary;
  mapping.requirements = kQwen38NativeRequirements;
  mapping.lists = kQwen38NativeLists;
  mapping.semantic = kQwen38NativeSemantic;
  mapping.extended_components = true;
  return mapping;
}();
#undef LLMP_QWEN38_BLOCK_PARAMS

}  // namespace

const Mapping& Qwen38Mapping() { return kMapping; }
const Mapping& Qwen38GgufMapping() { return kGgufMapping; }
const Mapping& Qwen38NativeMapping() { return kNativeMapping; }
const Mapping& Qwen38MtpMapping() { return kMtpMapping; }
const Mapping& Qwen38MtpNativeMapping() { return kMtpNativeMapping; }

}  // namespace llmp::spec::mappings
