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
#include "spec/mappings/names.h"
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
    Plain(names::kTokenEmbd.gguf, "embedding", "table"),
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
    Matrix(names::kAttnQkv.gguf, "linear", "qkv"),
    Scales("blk.{layer}.attn_qkv.weight_scale", "linear", "qkv"),
    Biases("blk.{layer}.attn_qkv.weight_bias", "linear", "qkv"),
    Matrix(names::kAttnGate.gguf, "linear", "gate"),
    Scales("blk.{layer}.attn_gate.weight_scale", "linear", "gate"),
    Biases("blk.{layer}.attn_gate.weight_bias", "linear", "gate"),
    Matrix(names::kSsmBeta.gguf, "linear", "beta"),
    Scales("blk.{layer}.ssm_beta.weight_scale", "linear", "beta"),
    Biases("blk.{layer}.ssm_beta.weight_bias", "linear", "beta"),
    Matrix(names::kSsmAlpha.gguf, "linear", "alpha"),
    Scales("blk.{layer}.ssm_alpha.weight_scale", "linear", "alpha"),
    Biases("blk.{layer}.ssm_alpha.weight_bias", "linear", "alpha"),
    Plain(names::kSsmDt.gguf, "linear", "dt_bias"),
    Plain(names::kSsmA.gguf, "linear", "a"),
    Plain(names::kSsmConv1d.gguf, "linear", "conv"),
    Plain(names::kSsmNorm.gguf, "linear", "norm"),
    Matrix(names::kSsmOut.gguf, "linear", "output"),
    Scales("blk.{layer}.ssm_out.weight_scale", "linear", "output"),
    Biases("blk.{layer}.ssm_out.weight_bias", "linear", "output"),
    // QSA: the indexer's projection fused (ModelOpt) or split (GGUF).
    Matrix(names::kAttnQ.gguf, "attention", "q"),
    Scales("blk.{layer}.attn_q.weight_scale", "attention", "q"),
    Biases("blk.{layer}.attn_q.weight_bias", "attention", "q"),
    Matrix(names::kAttnK.gguf, "attention", "k"),
    Scales("blk.{layer}.attn_k.weight_scale", "attention", "k"),
    Biases("blk.{layer}.attn_k.weight_bias", "attention", "k"),
    Matrix(names::kAttnV.gguf, "attention", "v"),
    Scales("blk.{layer}.attn_v.weight_scale", "attention", "v"),
    Biases("blk.{layer}.attn_v.weight_bias", "attention", "v"),
    Matrix(names::kAttnOutput.gguf, "attention", "output"),
    Scales("blk.{layer}.attn_output.weight_scale", "attention", "output"),
    Biases("blk.{layer}.attn_output.weight_bias", "attention", "output"),
    Plain(names::kAttnQNorm.gguf, "attention", "q_norm"),
    Plain(names::kAttnKNorm.gguf, "attention", "k_norm"),
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
    // The MoE: router, shared expert and its gate, routed experts (per-expert
    // arrays, or the scale-atom layout's stacked gate-up), their scales.
    Matrix(names::kFfnGateInp.gguf, "moe", "router"),
    Matrix(names::kFfnGateInpShexp.gguf, "moe", "shared_gate"),
    Scales("blk.{layer}.ffn_gate_inp_shexp.weight_scale", "moe", "shared_gate"),
    Biases("blk.{layer}.ffn_gate_inp_shexp.weight_bias", "moe", "shared_gate"),
    Matrix(names::kFfnGateShexp.gguf, "moe", "gate"),
    Scales("blk.{layer}.ffn_gate_shexp.weight_scale", "moe", "gate"),
    Biases("blk.{layer}.ffn_gate_shexp.weight_bias", "moe", "gate"),
    Matrix(names::kFfnUpShexp.gguf, "moe", "up"),
    Scales("blk.{layer}.ffn_up_shexp.weight_scale", "moe", "up"),
    Biases("blk.{layer}.ffn_up_shexp.weight_bias", "moe", "up"),
    Matrix(names::kFfnDownShexp.gguf, "moe", "down"),
    Scales("blk.{layer}.ffn_down_shexp.weight_scale", "moe", "down"),
    Biases("blk.{layer}.ffn_down_shexp.weight_bias", "moe", "down"),
    TensorRule{.pattern = names::kFfnGateExps.gguf,
               .slot = "moe",
               .role = "gate_experts",
               .experts = true,
               .optional = true},
    TensorRule{.pattern = names::kFfnUpExps.gguf,
               .slot = "moe",
               .role = "up_experts",
               .experts = true,
               .optional = true},
    TensorRule{.pattern = names::kFfnDownExps.gguf,
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
    Matrix(names::kOutput.gguf, "output", "weight"),
};

constexpr std::string_view kEps = "attention.layer_norm_rms_epsilon";

// The QSA, MoE and mix parameters both mappings read (the drafter's
// extraction values invert them: kMtpValues).
constexpr std::array kBlockParams = {
    Name("streams", "mix", "sigmoid"),
    FromKey("attn_mix", "eps", "attention.layer_norm_rms_epsilon", kReal),
    FromKey("attn_mix", "rank", "hyper_connection.low_rank", kInt),
    Flag("attn_mix", "inject", true),
    FromKey("ffn_mix", "eps", "attention.layer_norm_rms_epsilon", kReal),
    FromKey("ffn_mix", "rank", "hyper_connection.low_rank", kInt),
    Flag("ffn_mix", "inject", true),
    FromKey("output_mix", "eps", "attention.layer_norm_rms_epsilon", kReal),
    FromKey("output_mix", "rank", "hyper_connection.low_rank", kInt),
    Flag("output_mix", "inject", false),
    FromKey("attention", "eps", "attention.layer_norm_rms_epsilon", kReal),
    FromKey("attention", "heads", "attention.head_count", kInt),
    FromKey("attention", "kv_heads", "attention.head_count_kv", kInt),
    FromKey("attention", "head_dim", "attention.key_length", kInt),
    FromKey("attention", "rope_dims", "rope.dimension_count", kInt),
    FromKey("attention", "rope_base", "rope.freq_base", kReal),
    FromKey("attention", "rope_sections", "rope.dimension_sections", kInts),
    FromKey("attention", "indexer_heads", "attention.indexer.head_count", kInt),
    FromKey("attention", "indexer_dim", "attention.indexer.key_length", kInt),
    PerLayer("attention", "ratio", "attention.compress_ratios", kInt),
    FromKey("attention", "budget", "attention.indexer.top_k", kInt),
    FromKey("attention", "indexer", kIndexer, kString, Transform::kNone, {}, true),
    Name("attention", "indexer", "split"),
    Name("moe", "placement", "none"),
    FromKey("moe", "eps", "attention.layer_norm_rms_epsilon", kReal),
    Name("moe", "activation", "silu"),
    FromKey("moe", "hidden", "expert_shared_feed_forward_length", kInt),
    FromKey("moe", "experts", "expert_count", kInt),
    FromKey("moe", "used", "expert_used_count", kInt),
    FromKey("moe", "expert_hidden", "expert_feed_forward_length", kInt),
    Name("moe", "router", "softmax_topk"),
    Name("moe", "router_input", "stream"),
    Flag("moe", "normalize", true),
    Flag("moe", "shared_gate", true),
    ParamRule{.slot = "moe", .param = "stacked", .key = kStacked, .type = kBool, .optional = true},
    Flag("moe", "stacked", false),
    ParamRule{.slot = "moe",
              .param = "projection_scales",
              .key = kScaled,
              .type = kBool,
              .optional = true},
    Flag("moe", "projection_scales", false),
    FromKey(kStageSlot, "context", "context_length", kInt),
    FromKey(kStreamSlot, "width", "embedding_length", kInt),
    FromKey(kStreamSlot, "streams", "hyper_connection.count", kInt),
};

constexpr auto kParams =
    Concat(kBlockParams, std::array{
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
                         });

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

// The GGUF form of the canonical rules: each matrix whole of its type (its
// data part, required and of that part only), the indexer's projection
// split, the experts unscaled, the n-gram hash's constants in the metadata;
// no rule of a role the form lacks (kGgufAbsent), of another part or unbound.
struct RoleRef {
  std::string_view slot{}, role{};
};
constexpr std::array kGgufAbsent{RoleRef{"attention", "indexer_qk"}, RoleRef{"moe", "gate_scale"},
                                 RoleRef{"moe", "up_scale"}, RoleRef{"moe", "down_scale"}};
constexpr bool GgufTensor(const TensorRule& r) {
  return r.part == PartRole::kData && !r.unbound &&
         std::ranges::none_of(kGgufAbsent, [&](const RoleRef& absent) {
           return absent.slot == r.slot && absent.role == r.role;
         });
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
// Its parameters: the canonical ones but the native importer's binding facts.
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
// The native form of the canonical rules (the ModelOpt and MLX imports,
// import/normalize.h), as an overlay: a checkpoint's split parts bound to
// the same shared roles (MLX's affine experts and tables), and the
// importer's relaid-out NVFP4 experts (import/expert_relayout.h): the
// scale-atom gate-up and down tables, `.weight` and `.weight_scale`, beside
// their tensor scales' F32 [experts] vectors (the converted form's roles). A
// rule binds one role and part, so a down table's split and scale-atom forms
// share a pattern; the converted form's codes and scales tables are dropped.
constexpr std::array kNativeTensorEdits{
    TensorEdit{.slot = "moe", .role = "gate_up_experts", .drop = true},
    TensorEdit{.slot = "moe", .role = "down_experts", .part = PartRole::kCodes, .drop = true},
    TensorEdit{.slot = "moe", .role = "down_experts", .part = PartRole::kBlockScales, .drop = true},
    TensorEdit{.slot = "moe",
               .role = "gate_experts",
               .part = PartRole::kData,
               .set_part = PartRole::kCodes,
               .any_part = true},
    TensorEdit{.slot = "moe",
               .role = "up_experts",
               .part = PartRole::kData,
               .set_part = PartRole::kCodes,
               .any_part = true},
    TensorEdit{.slot = "moe",
               .role = "down_experts",
               .part = PartRole::kData,
               .set_part = PartRole::kCodes,
               .any_part = true},
    // A table's codes (split) or its rows of codes and scales (interleaved).
    TensorEdit{.slot = "embedding", .role = "table", .part = PartRole::kData, .any_part = true},
    TensorEdit{.slot = "ngram", .role = "table", .part = PartRole::kData, .any_part = true},
};
constexpr TensorRule ExpertScales(std::string_view pattern, std::string_view role) {
  auto rule = Scales(pattern, "moe", role);
  rule.experts = true;
  return rule;
}
constexpr TensorRule ExpertBiases(std::string_view pattern, std::string_view role) {
  auto rule = Biases(pattern, "moe", role);
  rule.experts = true;
  return rule;
}
constexpr std::array kNativeTensorExtras{
    TensorRule{.pattern = names::kFfnGateUpExpsFused.gguf,
               .slot = "moe",
               .role = "gate_up_experts",
               .part = PartRole::kCodes,
               .experts = true,
               .optional = true,
               .any_part = true},
    TensorRule{.pattern = "blk.{layer}.ffn_gate_up_exps.weight_scale",
               .slot = "moe",
               .role = "gate_up_experts",
               .part = PartRole::kBlockScales,
               .experts = true,
               .optional = true},
    Scales("token_embd.weight_scale", "embedding", "table"),
    Biases("token_embd.weight_bias", "embedding", "table"),
    Scales("output.weight_scale", "output", "weight"),
    Biases("output.weight_bias", "output", "weight"),
    Scales("per_layer_token_embd.weight_scale", "ngram", "table"),
    Biases("per_layer_token_embd.weight_bias", "ngram", "table"),
    ExpertScales("blk.{layer}.ffn_gate_exps.weight_scale", "gate_experts"),
    ExpertScales("blk.{layer}.ffn_up_exps.weight_scale", "up_experts"),
    ExpertScales("blk.{layer}.ffn_down_exps.weight_scale", "down_experts"),
    ExpertBiases("blk.{layer}.ffn_gate_exps.weight_bias", "gate_experts"),
    ExpertBiases("blk.{layer}.ffn_up_exps.weight_bias", "up_experts"),
    ExpertBiases("blk.{layer}.ffn_down_exps.weight_bias", "down_experts"),
};
constexpr auto kNativeTensors = Overlay<kTensors, kNativeTensorEdits, kNativeTensorExtras>();
static_assert(kNativeTensors.size() == kTensors.size() + 10);

// The native form of the block's parameters (the target's and its
// drafter's): the experts' stacking a fact of their tensors, every epsilon
// in float32 (the checkpoint's), the rotary dimensions a fraction of the
// head, the indexer fused, the projections' scales by their tensor scales.
constexpr std::array kNativeBlockEdits{
    ParamEdit{.slot = "moe", .param = "projection_scales", .drop = true},
    ParamEdit{.slot = "moe",
              .param = "stacked",
              .key = kStacked,
              .set_key = std::string_view{},
              .tensor = names::kFfnGateUpExpsFused.gguf},
    ParamEdit{.param = "eps", .transform = Transform::kFloat32},
    ParamEdit{.slot = "attention",
              .param = "rope_dims",
              .numerator_key = "attention.key_length",
              .divisor_key = "rope.partial_rotary_factor",
              .transform = Transform::kProduct},
    ParamEdit{.slot = "attention", .param = "indexer", .key = std::string_view{}, .text = "fused"},
};
constexpr auto kNativeParamEdits =
    Concat(kNativeBlockEdits,
           std::array{ParamEdit{.slot = "ngram",
                                .param = "row",
                                .optional = true,
                                .then = ParamRule{.slot = "ngram",
                                                  .param = "row",
                                                  .tensor = "per_layer_token_embd.weight",
                                                  .type = kInt}}});
// The native forms' own rules: the experts' tensor scales by their tensors,
// the checkpoint's conventions unless the importer folded them (D-108 as
// amended: the ModelOpt import stores 1 + w and -exp(A) in F32), and its
// config's requirements.
constexpr std::array kNativeExpertScales{
    Flag("moe", "projection_scales", false),
    FromTensor("moe", "gate_scale", "blk.{layer}.ffn_gate_exps.weight_scale_2"),
    Flag("moe", "gate_scale", false),
    FromTensor("moe", "up_scale", "blk.{layer}.ffn_up_exps.weight_scale_2"),
    Flag("moe", "up_scale", false),
    FromTensor("moe", "down_scale", "blk.{layer}.ffn_down_exps.weight_scale_2"),
    Flag("moe", "down_scale", false),
};
constexpr std::array kNativeRequirements{
    Require("mamba_ssm_dtype", "float32", true),
    Require("dtype", "bfloat16", true),
    Require("hidden_act", "silu", true),
    Require("output_gate_type", "sigmoid", true),
    Require("rope.scaling.type", "default", true),
    ParamRule{
        .key = "is_encoder_decoder", .type = kBool, .value = {.flag = false}, .optional = true},
    ParamRule{.key = "attention_bias", .type = kBool, .value = {.flag = false}, .optional = true},
};
constexpr std::array kNativeCausal{
    RequireInt("attention.indexer.head_count_kv", 1, true),
    ParamRule{.key = "use_bidirectional_attention",
              .type = kBool,
              .value = {.flag = false},
              .optional = true},
};
constexpr auto kNativeParamExtras = Concat(
    kNativeExpertScales,
    std::array{ZeroCenteredNorm("attn_mix"), ZeroCenteredNorm("ffn_mix"),
               ZeroCenteredNorm("output_mix"), ZeroCenteredNorm("attention"),
               ZeroCenteredNorm("ple"),
               ParamRule{.slot = "linear",
                         .param = "a_log",
                         .type = kBool,
                         .value = {.flag = true},
                         .unless_fold = "ssm.a_neg_exp"}},
    kNativeRequirements,
    std::array{
        FromKey("ngram", "declared_width", "ple.embed_dim", kInt, Transform::kNone, {}, true),
        FromKey("ple", "input_width", "ple.embed_dim", kInt, Transform::kNone, {}, true)},
    kNativeCausal);
constexpr auto kNativeParams = Overlay<kParams, kNativeParamEdits, kNativeParamExtras>();
static_assert(kNativeParams.size() == kParams.size() + 23);
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
    SemanticTensorRule{.tensor = "model.layers.{layer}.ple.ple_embedding.layer_multipliers",
                       .key = "ple.layer_multipliers",
                       .size = 0},
    SemanticTensorRule{.tensor = "model.layers.{layer}.self_attn.ple_embed.layer_multipliers",
                       .key = "ple.layer_multipliers",
                       .size = 0},
    SemanticTensorRule{.tensor = "model.layers.{layer}.ple.ple_embedding.ngram_heads_offsets",
                       .key = "ple.head_offsets",
                       .size = 0},
    SemanticTensorRule{.tensor = "model.layers.{layer}.self_attn.ple_embed.head_offsets",
                       .key = "ple.head_offsets",
                       .size = 0},
    SemanticTensorRule{.tensor = "model.layers.{layer}.ple.ple_embedding.ngram_heads_vocab_sizes",
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
    "model.layers.{layer}.mlp.experts.{expert}.down_proj.input_scale",
    "model.layers.{layer}.mlp.experts.{expert}.gate_proj.input_scale",
    "model.layers.{layer}.mlp.experts.{expert}.up_proj.input_scale",
};
constexpr std::array kNativeKeyAliases = {
    keys::kLinearValueHeadDim,
    KeyAlias{"indexer_kv_heads", "attention.indexer.head_count_kv"},
    keys::kRopeParametersMropeInterleaved,
    keys::kBidirectional,
    keys::kLayerCount,
    keys::kContext,
    keys::kWidth,
    KeyAlias{"hc_count", "hyper_connection.count"},
    KeyAlias{"hc_lowrank", "hyper_connection.low_rank"},
    keys::kRmsEps,
    keys::kHeads,
    keys::kKvHeads,
    keys::kHeadDim,
    keys::kRopeParametersTheta,
    keys::kRopeTheta,
    keys::kRopeParametersMropeSection,
    KeyAlias{"mrope_section", "rope.dimension_sections"},
    keys::kRopeParametersPartial,
    keys::kPartialRotary,
    KeyAlias{"indexer_n_heads", "attention.indexer.head_count"},
    KeyAlias{"indexer_head_dim", "attention.indexer.key_length"},
    KeyAlias{"indexer_budget", "attention.indexer.top_k"},
    KeyAlias{"indexer_compress_ratio", "attention.compress_ratios"},
    keys::kExperts,
    keys::kExpertsUsed,
    keys::kExpertWidth,
    keys::kSharedExpertWidth,
    keys::kLinearConvKernel,
    keys::kLinearKeyHeads,
    keys::kLinearValueHeads,
    keys::kLinearKeyHeadDim,
    KeyAlias{"ngram_size", "ple.ngram_size"},
    KeyAlias{"heads_per_ngram", "ple.heads_per_ngram"},
    KeyAlias{"ple_conv_kernel_size", "ple.conv_kernel"},
    KeyAlias{"ple_embed_dim", "ple.embed_dim"},
    KeyAlias{"text_config.eos_token_id", "ple.eos_token_id"},
    KeyAlias{"eos_token_id", "ple.eos_token_id", "text_config.eos_token_id"},
    KeyAlias{"ple_layer_ids", "ple.layer_ids"},
    keys::kLayerTypes,
    KeyAlias{"full_attention_interval", "full_attention_interval"},
    keys::kMambaDtype,
    KeyAlias{"dtype", "dtype"},
    keys::kEncoderDecoder,
    keys::kHiddenAct,
    keys::kAttentionBias,
    keys::kOutputGateType,
    keys::kRopeParametersType,
    KeyAlias{"rope_parameters.type", "rope.scaling.type"},
};
constexpr auto kNativeTensorAliases = Aliases(
    names::kOutput, names::kTokenEmbd,
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
    names::kSsmA, names::kSsmConv1d, names::kSsmDt, names::kSsmAlpha, names::kSsmBeta,
    names::kAttnQkv, names::kAttnGate, names::kSsmNorm, names::kSsmOut, names::kFfnDownExps,
    names::kFfnGateExps, names::kFfnUpExps, names::kFfnDownExpsSwitch, names::kFfnGateExpsSwitch,
    names::kFfnUpExpsSwitch, names::kFfnGateInp, names::kFfnDownShexp, names::kFfnGateShexp,
    names::kFfnUpShexp, names::kFfnGateInpShexp,
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
    TensorAlias{"model.layers.1.ple.ple_embedding.ngram_embedding.weight_scale_2",
                "per_layer_token_embd.weight_scale_2"},
    TensorAlias{"model.layers.{layer}.ple.value_proj.weight", "blk.{layer}.ple_value.weight"},
    TensorAlias{"model.layers.{layer}.self_attn.indexer.index_qk_proj.weight",
                "blk.{layer}.indexer.qk_proj.weight"},
    TensorAlias{"model.layers.{layer}.self_attn.indexer.k_layernorm.weight",
                "blk.{layer}.indexer.k_norm.weight"},
    TensorAlias{"model.layers.{layer}.self_attn.indexer.q_layernorm.weight",
                "blk.{layer}.indexer.q_norm.weight"},
    names::kAttnKNorm, names::kAttnK, names::kAttnOutput, names::kAttnQNorm, names::kAttnQ,
    names::kAttnV,
    TensorAlias{"model.layers.1.self_attn.ple_embed.shard_{shard}.weight",
                "per_layer_token_embd.weight"},
    TensorAlias{"model.layers.1.self_attn.ple_embed.weight_scale_2",
                "per_layer_token_embd.weight_scale_2"});

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
        .pattern = names::kTokenEmbd.gguf, .slot = "embedding", .role = "table", .target = true},
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
    Matrix(names::kAttnQ.gguf, "attention", "q"),
    Matrix(names::kAttnK.gguf, "attention", "k"),
    Matrix(names::kAttnV.gguf, "attention", "v"),
    Matrix(names::kAttnOutput.gguf, "attention", "output"),
    Plain(names::kAttnQNorm.gguf, "attention", "q_norm"),
    Plain(names::kAttnKNorm.gguf, "attention", "k_norm"),
    Matrix("blk.{layer}.indexer.qk_proj.weight", "attention", "indexer_qk"),
    Plain("blk.{layer}.indexer.q_norm.weight", "attention", "indexer_q_norm"),
    Plain("blk.{layer}.indexer.k_norm.weight", "attention", "indexer_k_norm"),
    Matrix(names::kFfnGateInp.gguf, "moe", "router"),
    Matrix(names::kFfnGateInpShexp.gguf, "moe", "shared_gate"),
    Matrix(names::kFfnGateShexp.gguf, "moe", "gate"),
    Matrix(names::kFfnUpShexp.gguf, "moe", "up"),
    Matrix(names::kFfnDownShexp.gguf, "moe", "down"),
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
    TensorRule{.pattern = names::kOutput.gguf, .slot = "output", .role = "weight", .target = true},
};

constexpr auto kMtpParams = Concat(
    kBlockParams, std::array{
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
                  });

// The drafter's: its BF16 linears' inputs rounded once past 16 rows, its
// QSA layer's and MoE's fast forms (its experts in the scale-atom layout).
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
constexpr std::array kMtpNativeTensorEdits{
    TensorEdit{.slot = "moe",
               .role = "gate_up_experts",
               .part = PartRole::kCodes,
               .pattern = names::kFfnGateUpExpsFused.gguf,
               .optional = true,
               .any_part = true},
    TensorEdit{.slot = "moe",
               .role = "gate_up_experts",
               .part = PartRole::kBlockScales,
               .pattern = "blk.{layer}.ffn_gate_up_exps.weight_scale",
               .optional = true},
    TensorEdit{.slot = "moe",
               .role = "down_experts",
               .part = PartRole::kCodes,
               .pattern = names::kFfnDownExps.gguf,
               .optional = true,
               .any_part = true},
    TensorEdit{.slot = "moe",
               .role = "down_experts",
               .part = PartRole::kBlockScales,
               .pattern = "blk.{layer}.ffn_down_exps.weight_scale",
               .optional = true},
    // ModelOpt's tensor scales; an MLX drafter's affine experts have none.
    TensorEdit{.slot = "moe", .role = "gate_scale", .optional = true},
    TensorEdit{.slot = "moe", .role = "up_scale", .optional = true},
    TensorEdit{.slot = "moe", .role = "down_scale", .optional = true},
};
// A native drafter's matrices may come with companions: their MXFP8 block
// scales and their MLX affine biases, each its matrix's `.weight` respelled.
// The drafter's: its own matrices (its rules of any part) and its target's
// head, in table order.
template <const auto& kRules>
constexpr auto MatricesOf() {
  std::array<TensorRule, std::ranges::count_if(kRules, &TensorRule::any_part)> out{};
  std::ranges::copy_if(kRules, out.begin(), &TensorRule::any_part);
  return out;
}
template <std::size_t N>
constexpr auto CompanionNames(const std::array<TensorRule, N>& matrices) {
  std::array<ComposedName, 2 * N> out{};
  for (std::size_t m = 0; m < N; ++m) {
    out[2 * m] = Respell(matrices[m].pattern, ".weight", ".weight_scale");
    out[(2 * m) + 1] = Respell(matrices[m].pattern, ".weight", ".weight_bias");
  }
  return out;
}
template <std::size_t N>
constexpr auto Companions(const std::array<TensorRule, N>& matrices,
                          const std::array<ComposedName, 2 * N>& names) {
  std::array<TensorRule, 2 * N> out{};
  for (std::size_t m = 0; m < N; ++m) {
    out[2 * m] = Scales(names[2 * m].view(), matrices[m].slot, matrices[m].role);
    out[(2 * m) + 1] = Biases(names[(2 * m) + 1].view(), matrices[m].slot, matrices[m].role);
  }
  return out;
}
constexpr auto kMtpMatrices =
    Concat(MatricesOf<kMtpTensors>(), std::array{Plain(names::kOutput.gguf, "output", "weight")});
constexpr auto kMtpCompanionNames = CompanionNames(kMtpMatrices);
// Each matrix's companions, then the unstacked experts' arrays and theirs.
constexpr auto kMtpNativeTensorExtras =
    Concat(Companions(kMtpMatrices, kMtpCompanionNames),
           std::array{
               TensorRule{.pattern = names::kFfnGateExps.gguf,
                          .slot = "moe",
                          .role = "gate_experts",
                          .part = PartRole::kCodes,
                          .experts = true,
                          .optional = true,
                          .any_part = true},
               ExpertScales("blk.{layer}.ffn_gate_exps.weight_scale", "gate_experts"),
               ExpertBiases("blk.{layer}.ffn_gate_exps.weight_bias", "gate_experts"),
               TensorRule{.pattern = names::kFfnUpExps.gguf,
                          .slot = "moe",
                          .role = "up_experts",
                          .part = PartRole::kCodes,
                          .experts = true,
                          .optional = true,
                          .any_part = true},
               ExpertScales("blk.{layer}.ffn_up_exps.weight_scale", "up_experts"),
               ExpertBiases("blk.{layer}.ffn_up_exps.weight_bias", "up_experts"),
               ExpertBiases("blk.{layer}.ffn_down_exps.weight_bias", "down_experts"),
           });
constexpr auto kMtpNativeTensors =
    Overlay<kMtpTensors, kMtpNativeTensorEdits, kMtpNativeTensorExtras>();
static_assert(kMtpNativeTensors.size() == kMtpTensors.size() + 51);
// Its parameters: the block's native form (kNativeBlockEdits), its norms
// zero-centered as the checkpoint holds them unless the importer stored
// 1 + w in F32 (norm.plus_one, as the converted drafter's), and its config's
// requirements.
constexpr auto kMtpNativeParamExtras =
    Concat(kNativeExpertScales,
           std::array{ZeroCenteredNorm("embedding"), ZeroCenteredNorm("attn_mix"),
                      ZeroCenteredNorm("ffn_mix"), ZeroCenteredNorm("output_mix"),
                      ZeroCenteredNorm("attention")},
           kNativeRequirements, kNativeCausal);
constexpr auto kMtpNativeParams = Overlay<kMtpParams, kNativeBlockEdits, kMtpNativeParamExtras>();
static_assert(kMtpNativeParams.size() == kMtpParams.size() + 19);
// Missing kept metadata in published native MTP artifacts is reconstructed
// from the actual target spec and this artifact's own tensors: the block's
// source keys from the target's components that read them (the forward
// rules inverted, kBlockParams), then the drafter's own facts.
constexpr auto kMtpBlockValues = InvertedValues<kBlockParams, kMtpSlots, kQwenRequirements>();
constexpr auto kMtpValues =
    Concat(kMtpBlockValues,
           std::array{ValueRule{.key = "block_count",
                                .origin = ValueOrigin::kLayerCount,
                                .tensor = names::kAttnQ.gguf},
                      ValueRule{.key = kIndexer, .value = {.text = "fused"}, .type = kString},
                      // The drafter's own expert layout: a relaid ModelOpt drafter is
                      // stacked and scaled whatever its target's (GGUF and MLX targets are
                      // neither).
                      ValueRule{.key = kStacked,
                                .origin = ValueOrigin::kTensorPresent,
                                .tensor = names::kFfnGateUpExpsFused.gguf,
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
                                .at_most = kServing.default_draft_vocab}});
constexpr std::array<std::string_view, 7> kMtpNativeIgnoredPrefixes = {
    "lm_head.", "language_model.lm_head.", "model.visual.",         "vision_tower.",
    "visual.",  "model.language_model.",   "language_model.model.",
};
constexpr std::array kMtpNativeKeyAliases = {
    KeyAlias{"indexer_kv_heads", "attention.indexer.head_count_kv"},
    keys::kRopeParametersMropeInterleaved,
    keys::kBidirectional,
    KeyAlias{"mtp_num_hidden_layers", "block_count"},
    keys::kContext,
    keys::kWidth,
    KeyAlias{"hc_count", "hyper_connection.count"},
    KeyAlias{"hc_lowrank", "hyper_connection.low_rank"},
    keys::kRmsEps,
    keys::kHeads,
    keys::kKvHeads,
    keys::kHeadDim,
    keys::kRopeParametersTheta,
    keys::kRopeTheta,
    keys::kRopeParametersMropeSection,
    KeyAlias{"mrope_section", "rope.dimension_sections"},
    keys::kRopeParametersPartial,
    keys::kPartialRotary,
    KeyAlias{"indexer_n_heads", "attention.indexer.head_count"},
    KeyAlias{"indexer_head_dim", "attention.indexer.key_length"},
    KeyAlias{"indexer_budget", "attention.indexer.top_k"},
    KeyAlias{"indexer_compress_ratio", "attention.compress_ratios"},
    keys::kExperts,
    keys::kExpertsUsed,
    keys::kExpertWidth,
    keys::kSharedExpertWidth,
    keys::kMambaDtype,
    KeyAlias{"dtype", "dtype"},
    keys::kEncoderDecoder,
    keys::kHiddenAct,
    keys::kAttentionBias,
    keys::kOutputGateType,
    keys::kRopeParametersType,
    KeyAlias{"rope_parameters.type", "rope.scaling.type"},
};
constexpr std::array kQwen38TargetNativeAuxiliary = {
    AuxiliaryNamespace{"eos_token_id",
                       "generation stop framing when nested text EOS supplies the PLE hash"},
    AuxiliaryNamespace{"vision_config.", "multimodal vision encoder outside the text pipeline"},
    AuxiliaryNamespace{"quantization.", "checkpoint quantization metadata handled by reader"},
    AuxiliaryNamespace{"quantization_config.",
                       "checkpoint quantization metadata handled by reader"},
    AuxiliaryNamespace{"vocab_size", "vocabulary row count stated by tensor extent"},
    AuxiliaryNamespace{"tie_word_embeddings", "output tying stated by tensor presence"},
    AuxiliaryNamespace{"bos_token_id", "tokenizer framing outside model forward math"},
    AuxiliaryNamespace{"pad_token_id", "tokenizer framing outside model forward math"},
    AuxiliaryNamespace{"attention_dropout", "training dropout outside inference"},
    AuxiliaryNamespace{"make_ngram_vocab_size_divisible_by",
                       "PLE vocab construction parameter already baked into head_vocab_sizes"},
    AuxiliaryNamespace{"ngram_vocab_size_base",
                       "PLE vocab construction parameter already baked into head_vocab_sizes"},
    AuxiliaryNamespace{"ple_embedding_dtype", "PLE table dtype stated by tensor format"},
    AuxiliaryNamespace{"split_ngram_parts", "PLE shard count stated by tensor names"},
    AuxiliaryNamespace{"mtp.", "MTP drafter configuration"},
    AuxiliaryNamespace{"mtp_num_hidden_layers", "MTP drafter layer count"},
    AuxiliaryNamespace{"text_config.model_type", "nested text configuration model_type"},
    AuxiliaryNamespace{"text_config.intermediate_size",
                       "dense FFN width; every imported layer routes its experts (a dense layer's "
                       "tensors have no rule and are refused)"},
    AuxiliaryNamespace{
        "text_config.mtp_use_dedicated_embeddings",
        "MTP drafter embedding sharing (the drafter's own import reads its tensors)"},
};
constexpr std::array kQwen38MtpNativeAuxiliary = {
    AuxiliaryNamespace{"full_attention_interval",
                       "the target stack schedule, outside the single-layer drafter"},
    AuxiliaryNamespace{"vision_config.", "multimodal vision encoder outside the text pipeline"},
    AuxiliaryNamespace{"quantization.", "checkpoint quantization metadata handled by reader"},
    AuxiliaryNamespace{"quantization_config.",
                       "checkpoint quantization metadata handled by reader"},
    AuxiliaryNamespace{"vocab_size", "vocabulary row count stated by tensor extent"},
    AuxiliaryNamespace{"tie_word_embeddings", "output tying stated by tensor presence"},
    AuxiliaryNamespace{"bos_token_id", "tokenizer framing outside model forward math"},
    AuxiliaryNamespace{"eos_token_id", "tokenizer framing outside model forward math"},
    AuxiliaryNamespace{"pad_token_id", "tokenizer framing outside model forward math"},
    AuxiliaryNamespace{"attention_dropout", "training dropout outside inference"},
    AuxiliaryNamespace{"linear_conv_kernel_dim", "target GDN parameter"},
    AuxiliaryNamespace{"linear_key_head_dim", "target GDN parameter"},
    AuxiliaryNamespace{"linear_num_key_heads", "target GDN parameter"},
    AuxiliaryNamespace{"linear_num_value_heads", "target GDN parameter"},
    AuxiliaryNamespace{"linear_value_head_dim", "target GDN parameter"},
    AuxiliaryNamespace{"make_ngram_vocab_size_divisible_by", "target PLE parameter"},
    AuxiliaryNamespace{"ngram_vocab_size_base", "target PLE parameter"},
    AuxiliaryNamespace{"heads_per_ngram", "target PLE parameter"},
    AuxiliaryNamespace{"ngram_size", "target PLE parameter"},
    AuxiliaryNamespace{"ple_conv_kernel_size", "target PLE parameter"},
    AuxiliaryNamespace{"ple_embed_dim", "target PLE parameter"},
    AuxiliaryNamespace{"ple_embedding_dtype", "target PLE parameter"},
    AuxiliaryNamespace{"ple_layer_ids", "target PLE parameter"},
    AuxiliaryNamespace{"split_ngram_parts", "target PLE parameter"},
    AuxiliaryNamespace{"layer_types", "target layer types"},
    AuxiliaryNamespace{"num_hidden_layers", "target layer count"},
    AuxiliaryNamespace{"mtp.", "MTP drafter nested config"},
    AuxiliaryNamespace{"text_config.model_type", "nested text configuration model_type"},
    AuxiliaryNamespace{"text_config.intermediate_size",
                       "dense FFN width; the drafter's block routes its experts"},
    AuxiliaryNamespace{"text_config.mtp_use_dedicated_embeddings",
                       "MTP embedding sharing: the drafter borrows its target's table"},
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
    TensorAlias{"mtp.layers.{layer}.mlp.experts.{expert}.up_proj.weight",
                "blk.{layer}.ffn_up_exps.weight"},
    TensorAlias{"mtp.layers.{layer}.mlp.experts.{expert}.down_proj.weight",
                "blk.{layer}.ffn_down_exps.weight"},
    TensorAlias{"mtp.layers.{layer}.mlp.switch_mlp.down_proj.weight",
                "blk.{layer}.ffn_down_exps.weight"},
    TensorAlias{"mtp.layers.{layer}.mlp.switch_mlp.gate_proj.weight",
                "blk.{layer}.ffn_gate_exps.weight"},
    TensorAlias{"mtp.layers.{layer}.mlp.switch_mlp.up_proj.weight",
                "blk.{layer}.ffn_up_exps.weight"},
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
constexpr auto kNativeGates = [] {
  auto gates = kGates;
  for (auto& gate : gates)
    if (gate.param == "projection_scales") gate.param = "gate_scale";
  return gates;
}();
constexpr std::array kNativeSlotSourceOverrides = {
    SourceDialect::SlotSourceOverride{
        .slot = "ngram", .layers_alias = "ple.layer_ids", .alias_index_base = 1},
    SourceDialect::SlotSourceOverride{
        .slot = "ple", .layers_alias = "ple.layer_ids", .alias_index_base = 1}};
// The GGUF target: llama.cpp's converted form (its folds asserted).
constexpr SourceDialect kGgufTargetDialect{.architecture = "qwen4exp",
                                           .container = "gguf",
                                           .execution_architecture = "qwen4exp",
                                           .layer_count_key = "block_count",
                                           .layer_types = "lllf",
                                           .layer_type_interval_key = "full_attention_interval",
                                           .layer_flags_key = "attention.compress_ratios",
                                           .layer_flag_type = 'f',
                                           .explicit_layer_rules = true,
                                           .complete_params = true,
                                           .params = kGgufParams,
                                           .tensors = kGgufTensors,
                                           .folds = kFolds,
                                           .auxiliary = kQwenAuxiliary,
                                           .requirements = kQwenRequirements,
                                           .policies = std::span(kPolicies).last(1),
                                           .gates = kGates,
                                           .lists = kQwenLists,
                                           .extended_components = true,
                                           .explicit_source_semantics = true,
                                           .row_indices = kQwenRows};
// A native target (a ModelOpt or MLX checkpoint under one of its source
// architectures' names): its config's spellings and layer types, the
// native forms of the tables, the importer's folds recorded.
constexpr SourceDialect NativeTargetDialect(std::string_view architecture,
                                            std::string_view container) {
  return {.architecture = architecture,
          .container = container,
          .execution_architecture = "qwen4exp",
          .source_architectures = kNativeSourceArchitectures,
          .source_key_prefix = "text_config.",
          .layer_count_key = "block_count",
          .context_key = "context_length",
          .layer_types = "lllf",
          .layer_type_key = "layer_types",
          .layer_type_interval_key = "full_attention_interval",
          .layer_flag_type = 'f',
          .layer_type_codes = kNativeLayerTypeCodes,
          .explicit_layer_rules = true,
          .complete_params = true,
          .params = kNativeParams,
          .tensors = kNativeTensors,
          .slots = kNativeSlotSourceOverrides,
          .folds = kNativeFolds,
          .tensor_aliases = kNativeTensorAliases,
          .key_aliases = kNativeKeyAliases,
          .auxiliary = kQwen38TargetNativeAuxiliary,
          .requirements = kQwen38NativeRequirements,
          .semantic = kQwen38NativeSemantic,
          .ignored_tensors = kNativeIgnoredTensors,
          .ignored_tensor_prefixes = kNativeIgnoredPrefixes,
          .policies = kNativePolicies,
          .gates = kNativeGates,
          .lists = kQwen38NativeLists,
          .extended_components = true,
          .explicit_source_semantics = true,
          .semantic_tensors = kNativeSemanticTensors};
}
constexpr std::array<SourceDialect, 7> kTargetDialects = {
    kGgufTargetDialect,
    NativeTargetDialect("qwen3_8_flash_next", "safetensors"),
    NativeTargetDialect("qwen3_8_flash_next", "mlx"),
    NativeTargetDialect("qwen4_exp", "safetensors"),
    NativeTargetDialect("qwen4_exp", "mlx"),
    NativeTargetDialect("qwen4exp", "safetensors"),
    NativeTargetDialect("qwen4exp", "mlx")};
constexpr auto kQwenAllTensors = WithNewIdentities<kTensors, kNativeTensors>();
constexpr auto kQwenAllParams = WithNewIdentities<kParams, kNativeParams>();
constexpr std::array<std::string_view, 3> kQwenContainers{"gguf", "safetensors", "mlx"};
constexpr std::array<std::string_view, 2> kMtpContainers{"safetensors", "mlx"};
constexpr Mapping kQwenCanonicalMapping = [] {
  auto mapping = kMapping;
  mapping.source_architectures = kNativeSourceArchitectures;
  mapping.containers = kQwenContainers;
  mapping.tensors = kQwenAllTensors;
  mapping.params = kQwenAllParams;
  mapping.dialects = kTargetDialects;
  return mapping;
}();

// The native drafter (its only containers): the native forms of its tables
// and the extraction values its kept metadata may lack.
constexpr SourceDialect NativeMtpDialect(std::string_view architecture,
                                         std::string_view container) {
  return {.architecture = architecture,
          .container = container,
          .execution_architecture = "qwen4exp-mtp",
          .source_architectures = kNativeSourceArchitectures,
          .source_key_prefix = "text_config.",
          .layer_count_key = "block_count",
          .layer_types = "f",
          .explicit_layer_rules = true,
          .complete_params = true,
          .params = kMtpNativeParams,
          .tensors = kMtpNativeTensors,
          .folds = kNativeFolds,
          .tensor_aliases = kMtpNativeTensorAliases,
          .key_aliases = kMtpNativeKeyAliases,
          .auxiliary = kQwen38MtpNativeAuxiliary,
          .requirements = kQwen38MtpNativeRequirements,
          .semantic = kQwen38NativeSemantic,
          .ignored_tensor_prefixes = kMtpNativeIgnoredPrefixes,
          .policies = kMtpPolicies,
          .lists = kQwen38MtpNativeLists,
          .extended_components = true,
          .explicit_source_semantics = true,
          .draft_values = kMtpValues};
}
constexpr std::array<SourceDialect, 6> kMtpDialects = {
    NativeMtpDialect("qwen3_8_flash_next", "safetensors"),
    NativeMtpDialect("qwen3_8_flash_next", "mlx"),
    NativeMtpDialect("qwen4_exp", "safetensors"),
    NativeMtpDialect("qwen4_exp", "mlx"),
    NativeMtpDialect("qwen4exp-mtp", "safetensors"),
    NativeMtpDialect("qwen4exp-mtp", "mlx")};
constexpr auto kMtpAllTensors = WithNewIdentities<kMtpTensors, kMtpNativeTensors>();
constexpr auto kMtpAllParams = WithNewIdentities<kMtpParams, kMtpNativeParams>();
constexpr Mapping kMtpCanonicalMapping = [] {
  auto mapping = kMtpMapping;
  mapping.source_architectures = kNativeSourceArchitectures;
  mapping.containers = kMtpContainers;
  mapping.tensors = kMtpAllTensors;
  mapping.params = kMtpAllParams;
  mapping.dialects = kMtpDialects;
  return mapping;
}();

}  // namespace

const Mapping& Qwen38Mapping() { return kQwenCanonicalMapping; }
const Mapping& Qwen38GgufMapping() { return kQwenCanonicalMapping; }
const Mapping& Qwen38NativeMapping() { return kQwenCanonicalMapping; }
const Mapping& Qwen38MtpMapping() { return kMtpCanonicalMapping; }
const Mapping& Qwen38MtpNativeMapping() { return kMtpCanonicalMapping; }

}  // namespace llmp::spec::mappings
