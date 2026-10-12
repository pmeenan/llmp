// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "spec/mappings/qwen38.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <expected>
#include <string>
#include <string_view>
#include <vector>

#include "format/encoding.h"
#include "format/layout.h"
#include "model/qwen38.h"
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
constexpr TensorRule Plain(std::string_view pattern, std::string_view slot, std::string_view role) {
  return {.pattern = pattern, .slot = slot, .role = role};
}

// The ModelOpt import's pairs: MXFP8 matrices, the n-gram table and its
// tensor scale, and the CUTLASS expert arrays.
constexpr std::array kAssembly = {
    V0Assembly{.suffixes = {".weight", ".weight_scale", ""},
               .roles = {PartRole::kCodes, PartRole::kBlockScales, PartRole::kData},
               .encoding = format::Encoding::kMxfp8,
               .layout = format::Layout::kSplit},
    V0Assembly{.stem = "per_layer_token_embd",
               .suffixes = {".weight", ".weight_scale_2", ""},
               .roles = {PartRole::kData, PartRole::kTensorScale, PartRole::kData},
               .encoding = format::Encoding::kNvfp4,
               .layout = format::Layout::kRowInterleaved},
    V0Assembly{.stem = "blk.{layer}.ffn_gate_up_exps",
               .suffixes = {".codes", ".scales", ""},
               .roles = {PartRole::kCodes, PartRole::kBlockScales, PartRole::kData},
               .encoding = format::Encoding::kNvfp4,
               .layout = format::Layout::kCutlassScaleAtoms,
               .stacked = 2,
               .experts = true},
    V0Assembly{.stem = "blk.{layer}.ffn_down_exps",
               .suffixes = {".codes", ".scales", ""},
               .roles = {PartRole::kCodes, PartRole::kBlockScales, PartRole::kData},
               .encoding = format::Encoding::kNvfp4,
               .layout = format::Layout::kCutlassScaleAtoms,
               .experts = true},
};

// The folds both converters apply: by GGUF sources, and by the ModelOpt
// import's converter identity (it records no transformation).
constexpr std::string_view kModelOpt = "modelopt_qwen38-";
constexpr std::array kFolds = {
    FoldRule{.fold = "norm.plus_one", .container = "gguf"},
    FoldRule{.fold = "ssm.a_neg_exp", .container = "gguf"},
    FoldRule{.fold = "gdn.value_heads_tiled", .container = "gguf"},
    FoldRule{.fold = "norm.plus_one", .converter = kModelOpt},
    FoldRule{.fold = "ssm.a_neg_exp", .converter = kModelOpt},
    FoldRule{.fold = "gdn.value_heads_tiled", .converter = kModelOpt},
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
    Matrix("blk.{layer}.ple_value.weight", "ple", "value"),
    Plain("blk.{layer}.ple_norm_key.weight", "ple", "norm_key"),
    Plain("blk.{layer}.ple_norm_query.weight", "ple", "norm_query"),
    Plain("blk.{layer}.ple_norm_conv.weight", "ple", "norm_conv"),
    Plain("blk.{layer}.ple_conv1d.weight", "ple", "conv"),
    Plain("blk.{layer}.hc_attn_norm.weight", "attn_mix", "norm"),
    Matrix("blk.{layer}.hc_attn_down.weight", "attn_mix", "down"),
    Matrix("blk.{layer}.hc_attn_up.weight", "attn_mix", "up"),
    Matrix("blk.{layer}.hc_attn_inject.weight", "attn_combine", "inject"),
    Plain("blk.{layer}.hc_ffn_norm.weight", "ffn_mix", "norm"),
    Matrix("blk.{layer}.hc_ffn_down.weight", "ffn_mix", "down"),
    Matrix("blk.{layer}.hc_ffn_up.weight", "ffn_mix", "up"),
    Matrix("blk.{layer}.hc_ffn_inject.weight", "ffn_combine", "inject"),
    // Gated DeltaNet.
    Matrix("blk.{layer}.attn_qkv.weight", "linear", "qkv"),
    Scales("blk.{layer}.attn_qkv.weight_scale", "linear", "qkv"),
    Matrix("blk.{layer}.attn_gate.weight", "linear", "gate"),
    Scales("blk.{layer}.attn_gate.weight_scale", "linear", "gate"),
    Matrix("blk.{layer}.ssm_beta.weight", "linear", "beta"),
    Scales("blk.{layer}.ssm_beta.weight_scale", "linear", "beta"),
    Matrix("blk.{layer}.ssm_alpha.weight", "linear", "alpha"),
    Scales("blk.{layer}.ssm_alpha.weight_scale", "linear", "alpha"),
    Plain("blk.{layer}.ssm_dt.bias", "linear", "dt_bias"),
    Plain("blk.{layer}.ssm_a", "linear", "a"),
    Plain("blk.{layer}.ssm_conv1d.weight", "linear", "conv"),
    Plain("blk.{layer}.ssm_norm.weight", "linear", "norm"),
    Matrix("blk.{layer}.ssm_out.weight", "linear", "output"),
    Scales("blk.{layer}.ssm_out.weight_scale", "linear", "output"),
    // QSA: the indexer's projection fused (ModelOpt) or split (GGUF).
    Matrix("blk.{layer}.attn_q.weight", "attention", "q"),
    Scales("blk.{layer}.attn_q.weight_scale", "attention", "q"),
    Matrix("blk.{layer}.attn_k.weight", "attention", "k"),
    Scales("blk.{layer}.attn_k.weight_scale", "attention", "k"),
    Matrix("blk.{layer}.attn_v.weight", "attention", "v"),
    Scales("blk.{layer}.attn_v.weight_scale", "attention", "v"),
    Matrix("blk.{layer}.attn_output.weight", "attention", "output"),
    Scales("blk.{layer}.attn_output.weight_scale", "attention", "output"),
    Plain("blk.{layer}.attn_q_norm.weight", "attention", "q_norm"),
    Plain("blk.{layer}.attn_k_norm.weight", "attention", "k_norm"),
    TensorRule{.pattern = "blk.{layer}.indexer.qk_proj.weight",
               .slot = "attention",
               .role = "indexer_qk",
               .optional = true,
               .any_part = true},
    Scales("blk.{layer}.indexer.qk_proj.weight_scale", "attention", "indexer_qk"),
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
    Matrix("blk.{layer}.ffn_gate_shexp.weight", "moe", "gate"),
    Scales("blk.{layer}.ffn_gate_shexp.weight_scale", "moe", "gate"),
    Matrix("blk.{layer}.ffn_up_shexp.weight", "moe", "up"),
    Scales("blk.{layer}.ffn_up_shexp.weight_scale", "moe", "up"),
    Matrix("blk.{layer}.ffn_down_shexp.weight", "moe", "down"),
    Scales("blk.{layer}.ffn_down_shexp.weight_scale", "moe", "down"),
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
    Matrix("output_hc_up.weight", "output_mix", "up"),
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

// The default planning policies (planner/policy.h, slice G3b): today's
// fast forms (kernels/ggml/qwen38_graph.h: the fused graph, not exact),
// each a form's own fusions enabled and the norms' and owners' patterns
// (which today's Qwen3.8 graphs do not take) off; the same in every
// serving envelope. The ModelOpt import with its experts in the CUTLASS
// layout (stacked), in GGML's NVFP4 blocks (scaled, not stacked), and a
// GGUF quantization's (neither).
constexpr std::array kPolicies = {
    PolicyRule{.gate = kStacked,
               .text = "max_decode_rows 16\n"
                       "enable bf16_rows\n"
                       "enable gdn_conv\n"
                       "enable gdn_gates\n"
                       "enable gdn_norm_gate\n"
                       "enable gdn_step\n"
                       "enable hc_combine\n"
                       "enable hc_fold_inputs\n"
                       "enable hc_prep\n"
                       "enable moe_experts\n"
                       "enable moe_router\n"
                       "enable mxfp8_inputs\n"
                       "enable owner_row_invariant\n"
                       "enable qsa_gate\n"
                       "enable qsa_pool\n"
                       "enable qsa_prep\n"
                       "enable qsa_select\n"
                       "disable grouped_stores\n"
                       "disable mul_mat_glu\n"
                       "disable narrow_final\n"
                       "disable owner_attention\n"
                       "disable prepared_inputs\n"
                       "disable rms_norm_mul\n"
                       "disable rms_norm_mul_add\n"
                       "disable rms_norm_mul_rope\n"
                       "disable rope_store\n"
                       "disable route_reduce\n"},
    PolicyRule{.gate = kScaled,
               .text = "max_decode_rows 16\n"
                       "enable bf16_rows\n"
                       "enable gdn_conv\n"
                       "enable gdn_gates\n"
                       "enable gdn_norm_gate\n"
                       "enable gdn_step\n"
                       "enable hc_combine\n"
                       "enable hc_fold_inputs\n"
                       "enable hc_prep\n"
                       "enable moe_combine\n"
                       "enable moe_glu\n"
                       "enable moe_router\n"
                       "enable mxfp8_inputs\n"
                       "enable owner_row_invariant\n"
                       "enable qsa_gate\n"
                       "enable qsa_pool\n"
                       "enable qsa_prep\n"
                       "enable qsa_select\n"
                       "disable grouped_stores\n"
                       "disable mul_mat_glu\n"
                       "disable narrow_final\n"
                       "disable owner_attention\n"
                       "disable prepared_inputs\n"
                       "disable rms_norm_mul\n"
                       "disable rms_norm_mul_add\n"
                       "disable rms_norm_mul_rope\n"
                       "disable rope_store\n"
                       "disable route_reduce\n"},
    PolicyRule{.text = "max_decode_rows 16\n"
                       "enable bf16_inputs\n"
                       "enable gdn_conv\n"
                       "enable gdn_norm_gate\n"
                       "enable gdn_step\n"
                       "enable hc_combine\n"
                       "enable hc_mix\n"
                       "enable moe_combine\n"
                       "enable moe_router\n"
                       "enable owner_row_invariant\n"
                       "enable qsa_pool\n"
                       "enable qsa_prep\n"
                       "enable qsa_select\n"
                       "enable vecq_glu\n"
                       "enable vecq_inputs\n"
                       "disable grouped_stores\n"
                       "disable mul_mat_glu\n"
                       "disable narrow_final\n"
                       "disable owner_attention\n"
                       "disable prepared_inputs\n"
                       "disable rms_norm_mul\n"
                       "disable rms_norm_mul_add\n"
                       "disable rms_norm_mul_rope\n"
                       "disable rope_store\n"
                       "disable route_reduce\n"},
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
    .v0_assembly = kAssembly,
};

// Closed GGUF target projection: whole encoded data parts, split indexer,
// source-owned PLE constants, and the existing GGUF planning semantics.
constexpr bool GgufTensor(const TensorRule& r) {
  return r.part == PartRole::kData && !r.unbound && !r.pattern.contains("weight_scale") &&
         !r.pattern.contains("ffn_gate_up_exps") && !r.pattern.contains("indexer.qk_proj");
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
constexpr std::array kGgufProfile = {
    RequireInt("qwen4exp.block_count", 48, false),
    RequireInt("qwen4exp.context_length", 262144, false),
    RequireInt("qwen4exp.embedding_length", 2560, false),
    RequireInt("qwen4exp.attention.head_count", 24, false),
    RequireInt("qwen4exp.attention.head_count_kv", 2, false),
    RequireInt("qwen4exp.attention.key_length", 256, false),
    RequireInt("qwen4exp.attention.value_length", 256, false),
    RequireInt("qwen4exp.rope.dimension_count", 64, false),
    RequireInt("qwen4exp.expert_count", 512, false),
    RequireInt("qwen4exp.expert_used_count", 10, false),
    RequireInt("qwen4exp.expert_feed_forward_length", 640, false),
    RequireInt("qwen4exp.expert_shared_feed_forward_length", 640, false),
    RequireInt("qwen4exp.ssm.conv_kernel", 4, false),
    RequireInt("qwen4exp.ssm.state_size", 128, false),
    RequireInt("qwen4exp.ssm.group_count", 16, false),
    RequireInt("qwen4exp.ssm.time_step_rank", 48, false),
    RequireInt("qwen4exp.ssm.inner_size", 6144, false),
    RequireInt("qwen4exp.hyper_connection.count", 4, false),
    RequireInt("qwen4exp.hyper_connection.low_rank", 320, false),
    RequireInt("qwen4exp.attention.indexer.head_count", 4, false),
    RequireInt("qwen4exp.attention.indexer.key_length", 128, false),
    RequireInt("qwen4exp.attention.indexer.top_k", 2048, false),
    RequireInt("qwen4exp.ple.ngram_size", 3, false),
    RequireInt("qwen4exp.ple.heads_per_ngram", 8, false),
    RequireInt("qwen4exp.ple.conv_kernel", 4, false),
    RequireInt("qwen4exp.ple.eos_token_id", 248044, false),
    RequireInt("qwen4exp.ple.image_token_id", 248056, false),
    RequireInt("qwen4exp.embedding_length_per_layer_input", 160, false),
    ParamRule{.key = "qwen4exp.rope.freq_base", .type = kReal, .value = {.real = 10000000.0}},
    ParamRule{.key = "qwen4exp.attention.layer_norm_rms_epsilon",
              .type = kReal,
              .value = {.real = static_cast<double>(1e-6F)}},
};
constexpr auto kGgufParams = [] {
  constexpr auto count =
      std::ranges::count_if(kParams, [](const auto& p) { return !p.key.starts_with("llmp."); });
  std::array<ParamRule, count + kGgufProfile.size()> out{};
  std::size_t n = 0;
  for (const auto& p : kParams)
    if (!p.key.starts_with("llmp.")) out[n++] = p;
  for (const auto& p : kGgufProfile) out[n++] = p;
  return out;
}();
constexpr auto kGgufMapping = [] {
  auto m = kMapping;
  m.tensors = kGgufTensors;
  m.params = kGgufParams;
  m.folds = std::span(kFolds).first(3);
  m.policies = std::span(kPolicies).last(1);
  m.v0_assembly = {};
  return m;
}();

// Native normalization binds source-preserving split parts to the same
// shared roles. Embedded per-matrix scales are not projection multipliers.
constexpr auto kNativeTensors = [] {
  // Four legacy CUTLASS alternatives are replaced by four native split-part
  // rules. Keeping both down-scale alternatives would bind one role twice.
  std::array<TensorRule, kTensors.size()> rules{};
  std::size_t next = 0;
  for (auto rule : kTensors) {
    if (rule.pattern == "blk.{layer}.ffn_gate_up_exps.codes" ||
        rule.pattern == "blk.{layer}.ffn_gate_up_exps.scales" ||
        rule.pattern == "blk.{layer}.ffn_down_exps.codes" ||
        rule.pattern == "blk.{layer}.ffn_down_exps.scales")
      continue;
    if (rule.pattern == "per_layer_token_embd.weight" ||
        rule.pattern == "blk.{layer}.ffn_gate_exps.weight" ||
        rule.pattern == "blk.{layer}.ffn_up_exps.weight" ||
        rule.pattern == "blk.{layer}.ffn_down_exps.weight") {
      rule.part = PartRole::kCodes;
      rule.any_part = false;
    }
    if (rule.role == "gate_scale" || rule.role == "up_scale" || rule.role == "down_scale") {
      rule.role = rule.role == "gate_scale" ? "gate_experts"
                  : rule.role == "up_scale" ? "up_experts"
                                            : "down_experts";
      rule.part = PartRole::kTensorScale;
      rule.experts = true;
    }
    rules[next++] = rule;
  }
  rules[next++] = Scales("per_layer_token_embd.weight_scale", "ngram", "table");
  for (const auto& [pattern, role] : std::array<std::pair<std::string_view, std::string_view>, 3>{
           {{"blk.{layer}.ffn_gate_exps.weight_scale", "gate_experts"},
            {"blk.{layer}.ffn_up_exps.weight_scale", "up_experts"},
            {"blk.{layer}.ffn_down_exps.weight_scale", "down_experts"}}}) {
    auto rule = Scales(pattern, "moe", role);
    rule.experts = true;
    rules[next++] = rule;
  }
  return rules;
}();
constexpr auto kNativeParams = [] {
  std::array<ParamRule, kParams.size() + 6> params{};
  std::ranges::copy(kParams, params.begin());
  std::size_t i = kParams.size();
  for (std::string_view slot : {"attn_mix", "ffn_mix", "output_mix", "attention", "ple"})
    params[i++] = Flag(slot, "norm_zero_centered", true);
  params[i] = Flag("linear", "a_log", true);
  return params;
}();
constexpr auto kNativeMapping = [] {
  auto mapping = kMapping;
  mapping.tensors = kNativeTensors;
  mapping.params = kNativeParams;
  mapping.folds = {};
  mapping.policies = {};
  mapping.v0_assembly = {};
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
                       "enable bf16_inputs\n"
                       "enable hc_combine\n"
                       "enable hc_fold_inputs\n"
                       "enable hc_prep\n"
                       "enable moe_experts\n"
                       "enable moe_router\n"
                       "enable owner_row_invariant\n"
                       "enable qsa_pool\n"
                       "enable qsa_prep\n"
                       "enable qsa_select\n"
                       "disable grouped_stores\n"
                       "disable mul_mat_glu\n"
                       "disable narrow_final\n"
                       "disable owner_attention\n"
                       "disable prepared_inputs\n"
                       "disable rms_norm_mul\n"
                       "disable rms_norm_mul_add\n"
                       "disable rms_norm_mul_rope\n"
                       "disable rope_store\n"
                       "disable route_reduce\n"},
};

constexpr std::array kMtpReads = {
    ReadRule{.kind = ReadKind::kTap, .name = "streams", .point = TapPoint::kStreams},
};

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
    .v0_assembly = kAssembly,
};

constexpr auto kMtpReplaySlots = [] {
  auto slots = kMtpSlots;
  for (auto& slot : slots)
    if (slot.kind == "embedding.mtp" || slot.kind == "attention.qsa") slot.version = 2;
  return slots;
}();
constexpr auto kMtpReplayParams = [] {
  std::array<ParamRule, kMtpParams.size() + 1> params{};
  std::ranges::copy(kMtpParams, params.begin());
  params.back() = Int("embedding", "token_offset", 1);
  return params;
}();
constexpr auto kMtpReplayMapping = [] {
  auto mapping = kMtpMapping;
  mapping.version = 2;
  mapping.slots = kMtpReplaySlots;
  mapping.params = kMtpReplayParams;
  return mapping;
}();

#undef LLMP_QWEN38_BLOCK_PARAMS

// The keys both mappings' blocks read, under `p`.
void BlockValues(std::vector<SourceValue>& out, const std::string& p, const model::Qwen38Profile& q,
                 std::vector<std::int64_t> ratios) {
  const auto key = [&](std::string_view k) { return p + std::string(k); };
  out.push_back(Value(key("context_length"), model::kQwen38FlashContext));
  out.push_back(Value(key("embedding_length"), q.width));
  out.push_back(Value(key("hyper_connection.count"), q.hc));
  out.push_back(Value(key("hyper_connection.low_rank"), q.hc_rank));
  out.push_back(Value(key("attention.layer_norm_rms_epsilon"), q.rms_eps));
  out.push_back(Value(key("attention.head_count"), q.heads));
  out.push_back(Value(key("attention.head_count_kv"), q.kv_heads));
  out.push_back(Value(key("attention.key_length"), q.head_dim));
  out.push_back(Value(key("attention.value_length"), q.head_dim));
  out.push_back(Value(key("rope.dimension_count"), q.rope_dims));
  out.push_back(Value(key("rope.freq_base"), q.rope_base));
  out.push_back(Value(key("rope.dimension_sections"),
                      std::vector<std::int64_t>{q.rope_sections[0], q.rope_sections[1],
                                                q.rope_sections[2], q.rope_sections[3]}));
  out.push_back(Value(key("attention.indexer.head_count"), q.indexer_heads));
  out.push_back(Value(key("attention.indexer.key_length"), q.indexer_head_dim));
  out.push_back(Value(key("attention.indexer.top_k"), q.indexer_budget));
  out.push_back(Value(key("attention.compress_ratios"), std::move(ratios)));
  out.push_back(Value(key("expert_count"), q.experts));
  out.push_back(Value(key("expert_used_count"), q.experts_used));
  out.push_back(Value(key("expert_feed_forward_length"), q.expert_ffn));
  out.push_back(Value(key("expert_shared_feed_forward_length"), q.shared_ffn));
}

}  // namespace

const Mapping& Qwen38Mapping() { return kMapping; }
const Mapping& Qwen38GgufMapping() { return kGgufMapping; }
const Mapping& Qwen38NativeMapping() { return kNativeMapping; }
const Mapping& Qwen38MtpMapping() { return kMtpMapping; }
const Mapping& Qwen38MtpReplayMapping() { return kMtpReplayMapping; }

std::expected<std::vector<SourceValue>, std::string> Qwen38V0Values(
    const model::Qwen38Profile& p, const model::Qwen38Binding& binding,
    const model::Qwen38PleHash& hash) {
  std::vector<std::int64_t> ratios;
  for (std::uint32_t il = 0; il < p.layers; ++il) {
    const bool full = kMapping.layer_types.at(il % kMapping.layer_types.size()) == 'f';
    if (p.linear(il) == full) {
      return std::unexpected("the Qwen3.8 profile's linear layers are not the mapping's");
    }
    ratios.push_back(full ? std::int64_t{p.indexer_ratio} : 0);
  }
  if (binding.layers.size() != p.layers) return std::unexpected("a binding of another profile");
  const auto list = [](const std::vector<std::uint64_t>& v) {
    return std::vector<std::int64_t>(v.begin(), v.end());
  };
  std::vector<SourceValue> out;
  BlockValues(out, "qwen4exp.", p, std::move(ratios));
  for (SourceValue v : std::vector<SourceValue>{
           Value("qwen4exp.block_count", p.layers),
           Value("qwen4exp.full_attention_interval", 4U),
           Value("qwen4exp.ssm.conv_kernel", p.conv),
           Value("qwen4exp.ssm.group_count", p.lin_k_heads),
           Value("qwen4exp.ssm.time_step_rank", p.lin_v_heads),
           Value("qwen4exp.ssm.state_size", p.lin_head_dim),
           Value("qwen4exp.ssm.inner_size", p.lin_v_width()),
           Value("qwen4exp.ple.ngram_size", p.ngram),
           Value("qwen4exp.ple.heads_per_ngram", p.heads_per_ngram),
           Value("qwen4exp.ple.conv_kernel", p.ple_conv),
           Value("qwen4exp.ple.eos_token_id", static_cast<std::uint32_t>(p.ple_eos)),
           Value("qwen4exp.embedding_length_per_layer_input", p.ple_row),
           Value(kPleLayers, std::vector<std::int64_t>{p.ple_layer}),
           Value("qwen4exp.ple.layer_multipliers", list(hash.multipliers)),
           Value("qwen4exp.ple.head_offsets", list(hash.offsets)),
           Value("qwen4exp.ple.head_vocab_sizes", list(hash.vocab)),
       }) {
    out.push_back(std::move(v));
  }
  // The ModelOpt import's form (a GGUF's when absent).
  if (!binding.gguf()) {
    out.push_back(Value(kIndexer, "fused"));
    out.push_back(FlagValue(kStacked, binding.cutlass()));
    out.push_back(FlagValue(kScaled, true));
  }
  return out;
}

std::expected<std::vector<SourceValue>, std::string> Qwen38MtpV0Values(
    const model::Qwen38Profile& p, const model::Qwen38MtpBinding& binding) {
  std::vector<SourceValue> out;
  BlockValues(out, "qwen4exp-mtp.", p, {std::int64_t{p.indexer_ratio}});
  out.push_back(Value("qwen4exp-mtp.block_count", 1U));
  out.push_back(Value(kIndexer, "fused"));
  out.push_back(FlagValue(kStacked, true));
  out.push_back(FlagValue(kScaled, true));
  std::uint64_t head_rows = p.vocab;
  if (binding.selected_head()) {
    if (binding.draft_output.ne.size() != 2) return std::unexpected("a selected head of no rows");
    out.push_back(Value(kDraftRows, static_cast<std::uint32_t>(binding.draft_output.ne[1])));
    head_rows = binding.draft_output.ne[1];
  }
  out.push_back(Value(kDraftVocab, static_cast<std::uint32_t>(
                                       std::min<std::uint64_t>(kQwen38DraftVocab, head_rows))));
  return out;
}

}  // namespace llmp::spec::mappings
