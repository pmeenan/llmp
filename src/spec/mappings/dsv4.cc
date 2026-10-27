// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "spec/mappings/dsv4.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <expected>
#include <format>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "format/encoding.h"
#include "format/tensor_format.h"
#include "spec/mapping.h"
#include "spec/mappings/names.h"
#include "spec/mappings/rules.h"
#include "spec/spec.h"

namespace llmp::spec::mappings {
namespace {

using enum ValueType;

// The compression ratios of a CSA and an HCA layer.
constexpr std::uint32_t kCsaRatio = 4;
constexpr std::uint32_t kHcaRatio = 128;

constexpr TensorRule Plain(std::string_view pattern, std::string_view slot, std::string_view role,
                           std::string_view layer_types = {}) {
  return {.pattern = pattern, .slot = slot, .role = role, .layer_types = layer_types};
}
constexpr TensorRule Experts(std::string_view pattern, std::string_view slot,
                             std::string_view role) {
  return {.pattern = pattern, .slot = slot, .role = role, .experts = true};
}

// Each layer's type by its compression ratio.
constexpr std::array kCodes = {
    LayerTypeCode{.value = 0, .type = 'w'},
    LayerTypeCode{.value = kCsaRatio, .type = 'c'},
    LayerTypeCode{.value = kHcaRatio, .type = 'h'},
};

// The block slots both mappings have: each block between its Sinkhorn mix
// and combine.
#define LLMP_DSV4_BLOCK_SLOTS                                                       \
  SlotRule{.slot = "streams", .site = Site::kMixing, .kind = "residual.hc"},        \
      SlotRule{.slot = "attn_mix", .site = Site::kLayer, .kind = "hc.mix"},         \
      SlotRule{.slot = "attention", .site = Site::kLayer, .kind = "attention.mqa"}, \
      SlotRule{.slot = "attn_combine", .site = Site::kLayer, .kind = "hc.combine"}, \
      SlotRule{.slot = "ffn_mix", .site = Site::kLayer, .kind = "hc.mix"}

// A block's tensors, but its MoE's (by slot) and the head's.
#define LLMP_DSV4_BLOCK_TENSORS                                           \
  Plain("blk.{layer}.hc_attn_fn.weight", "attn_mix", "fn"),               \
      Plain("blk.{layer}.hc_attn_base.weight", "attn_mix", "base"),       \
      Plain("blk.{layer}.hc_attn_scale.weight", "attn_mix", "scale"),     \
      Plain(names::kAttnNorm.gguf, "attn_mix", "post_norm"),              \
      Plain("blk.{layer}.attn_q_a.weight", "attention", "q_a"),           \
      Plain("blk.{layer}.attn_q_a_norm.weight", "attention", "q_norm"),   \
      Plain("blk.{layer}.attn_q_b.weight", "attention", "q_b"),           \
      Plain("blk.{layer}.attn_kv.weight", "attention", "kv"),             \
      Plain("blk.{layer}.attn_kv_a_norm.weight", "attention", "kv_norm"), \
      Plain("blk.{layer}.attn_sinks.weight", "attention", "sinks"),       \
      Plain("blk.{layer}.attn_output_a.weight", "attention", "out_a"),    \
      Plain("blk.{layer}.attn_output_b.weight", "attention", "out_b"),    \
      Plain("blk.{layer}.hc_ffn_fn.weight", "ffn_mix", "fn"),             \
      Plain("blk.{layer}.hc_ffn_base.weight", "ffn_mix", "base"),         \
      Plain("blk.{layer}.hc_ffn_scale.weight", "ffn_mix", "scale"),       \
      Plain(names::kFfnNorm.gguf, "ffn_mix", "post_norm"),                \
      Plain("output_hc_fn.weight", "output_mix", "fn"),                   \
      Plain("output_hc_base.weight", "output_mix", "base"),               \
      Plain("output_hc_scale.weight", "output_mix", "scale"),             \
      Plain(names::kOutputNorm.gguf, "output_mix", "post_norm")

// An MoE slot's tensors.
#define LLMP_DSV4_MOE_TENSORS(SLOT)                                                               \
  Plain(names::kFfnGateInp.gguf, SLOT, "router"),                                                 \
      Experts(names::kFfnGateExps.gguf, SLOT, "gate_experts"),                                    \
      Experts(names::kFfnUpExps.gguf, SLOT, "up_experts"),                                        \
      Experts(names::kFfnDownExps.gguf, SLOT, "down_experts"),                                    \
      Plain(names::kFfnGateShexp.gguf, SLOT, "gate"), Plain(names::kFfnUpShexp.gguf, SLOT, "up"), \
      Plain(names::kFfnDownShexp.gguf, SLOT, "down")

// The block parameters both mappings read.
#define LLMP_DSV4_BLOCK_PARAMS                                                                     \
  Name("streams", "mix", "sinkhorn"), Name("attn_mix", "mix", "sinkhorn"),                         \
      FromKey("attn_mix", "eps", "attention.layer_norm_rms_epsilon", kReal),                       \
      FromKey("attn_mix", "hc_eps", "hyper_connection.epsilon", kReal),                            \
      FromKey("attn_mix", "iterations", "hyper_connection.sinkhorn_iterations", kInt),             \
      Name("attn_combine", "mix", "sinkhorn"), Name("ffn_mix", "mix", "sinkhorn"),                 \
      FromKey("ffn_mix", "eps", "attention.layer_norm_rms_epsilon", kReal),                        \
      FromKey("ffn_mix", "hc_eps", "hyper_connection.epsilon", kReal),                             \
      FromKey("ffn_mix", "iterations", "hyper_connection.sinkhorn_iterations", kInt),              \
      Name("ffn_combine", "mix", "sinkhorn"), Name("output_mix", "mix", "sinkhorn"),               \
      FromKey("output_mix", "eps", "attention.layer_norm_rms_epsilon", kReal),                     \
      FromKey("output_mix", "hc_eps", "hyper_connection.epsilon", kReal),                          \
      Flag("output_mix", "inject", false),                                                         \
      FromKey("attention", "eps", "attention.layer_norm_rms_epsilon", kReal),                      \
      FromKey("attention", "heads", "attention.head_count", kInt),                                 \
      FromKey("attention", "head_dim", "attention.key_length", kInt),                              \
      FromKey("attention", "rope_dims", "rope.dimension_count", kInt),                             \
      FromKey("attention", "q_rank", "attention.q_lora_rank", kInt),                               \
      FromKey("attention", "o_rank", "attention.output_lora_rank", kInt),                          \
      FromKey("attention", "o_groups", "attention.output_group_count", kInt),                      \
      FromKey("attention", "window", "attention.sliding_window", kInt),                            \
      FromKey("attention", "rope_base", "rope.freq_base", kReal, Transform::kNone, "w"),           \
      FromKey("attention", "rope_base", "attention.compress_rope_freq_base", kReal,                \
              Transform::kNone, "ch"),                                                             \
      FromKey("attention", "rope_factor", "rope.scaling.factor", kReal, Transform::kNone, "ch"),   \
      FromKey("attention", "yarn_context", "rope.scaling.original_context_length", kInt,           \
              Transform::kNone, "ch"),                                                             \
      FromKey("attention", "beta_fast", "rope.scaling.yarn_beta_fast", kReal, Transform::kNone,    \
              "ch"),                                                                               \
      FromKey("attention", "beta_slow", "rope.scaling.yarn_beta_slow", kReal, Transform::kNone,    \
              "ch"),                                                                               \
      Int("attention", "ratio", kCsaRatio, "c"), Int("attention", "ratio", kHcaRatio, "h"),        \
      Flag("attention", "overlap", true, "c"), Flag("attention", "indexer", true, "c"),            \
      FromKey("attention", "indexer_heads", "attention.indexer.head_count", kInt,                  \
              Transform::kNone, "c"),                                                              \
      FromKey("attention", "index_dim", "attention.indexer.key_length", kInt, Transform::kNone,    \
              "c"),                                                                                \
      FromKey("attention", "top_k", "attention.indexer.top_k", kInt, Transform::kNone, "c"),       \
      FromKey(kStageSlot, "context", "context_length", kInt),                                      \
      FromKey(kStreamSlot, "width", "embedding_length", kInt),                                     \
      FromKey(kStreamSlot, "streams", "hyper_connection.count",                                    \
              kInt), /* One KV head; the rotation's YaRN where it is scaled. */                    \
      RequireInt("attention.head_count_kv", 1, false), Require("rope.scaling.type", "yarn", true), \
      RequireInt("expert_shared_count", 1, false)

// An MoE slot's parameters (its router aside). A source listing no shared
// expert clamps (antirez's GGUFs) clamps its shared expert as its routed
// ones, as the M3 profile served it.
#define LLMP_DSV4_MOE_PARAMS(SLOT)                                                                 \
  Name(SLOT, "placement", "none"),                                                                 \
      FromKey(SLOT, "eps", "attention.layer_norm_rms_epsilon", kReal),                             \
      Name(SLOT, "activation", "silu_clamp"),                                                      \
      FromKey(SLOT, "hidden", "expert_feed_forward_length", kInt),                                 \
      FromKey(SLOT, "experts", "expert_count", kInt),                                              \
      FromKey(SLOT, "used", "expert_used_count", kInt),                                            \
      FromKey(SLOT, "expert_hidden", "expert_feed_forward_length", kInt),                          \
      Name(SLOT, "router_input", "stream"),                                                        \
      FromKey(SLOT, "normalize", "expert_weights_norm", kBool),                                    \
      FromKey(SLOT, "route_scale", "expert_weights_scale", kReal), Flag(SLOT, "router_f32", true), \
      PerLayer(SLOT, "limit", "swiglu_clamp_exp", kReal),                                          \
      PerLayer(SLOT, "shared_limit", "swiglu_clamp_shexp", kReal),                                 \
      PerLayer(SLOT, "shared_limit", "swiglu_clamp_exp", kReal)

// ---------------------------------------------------------------- deepseek4

constexpr std::string_view kHashLayers = "hash_layer_count";

constexpr std::array kSlots = {
    SlotRule{.slot = "embedding", .site = Site::kEmbedding, .kind = "embedding.token"},
    // A token-hash side input a hash-routed layer.
    SlotRule{.slot = "hash",
             .site = Site::kSide,
             .kind = "side.hash",
             .count_key = kHashLayers,
             .per_layer = true},
    LLMP_DSV4_BLOCK_SLOTS,
    SlotRule{.slot = "moe_hash", .site = Site::kLayer, .kind = "ffn.moe", .count_key = kHashLayers},
    SlotRule{.slot = "moe",
             .site = Site::kLayer,
             .kind = "ffn.moe",
             .count_key = kHashLayers,
             .count_above = true},
    SlotRule{.slot = "ffn_combine", .site = Site::kLayer, .kind = "hc.combine"},
    SlotRule{.slot = "output_mix", .site = Site::kFinalNorm, .kind = "hc.mix"},
    SlotRule{.slot = "output", .site = Site::kOutput, .kind = "head.linear"},
};

constexpr std::array kTensors = {
    Plain(names::kTokenEmbd.gguf, "embedding", "table"),
    Plain("blk.{layer}.ffn_gate_tid2eid.weight", "hash", "table"),
    LLMP_DSV4_BLOCK_TENSORS,
    // The compressor (CSA and HCA layers) and the indexer (CSA's).
    Plain("blk.{layer}.attn_compressor_kv.weight", "attention", "comp_kv", "ch"),
    Plain("blk.{layer}.attn_compressor_gate.weight", "attention", "comp_gate", "ch"),
    Plain("blk.{layer}.attn_compressor_ape.weight", "attention", "comp_ape", "ch"),
    Plain("blk.{layer}.attn_compressor_norm.weight", "attention", "comp_norm", "ch"),
    Plain("blk.{layer}.indexer.attn_q_b.weight", "attention", "idx_q", "c"),
    Plain("blk.{layer}.indexer.proj.weight", "attention", "idx_proj", "c"),
    Plain("blk.{layer}.indexer_compressor_kv.weight", "attention", "idx_kv", "c"),
    Plain("blk.{layer}.indexer_compressor_gate.weight", "attention", "idx_gate", "c"),
    Plain("blk.{layer}.indexer_compressor_ape.weight", "attention", "idx_ape", "c"),
    Plain("blk.{layer}.indexer_compressor_norm.weight", "attention", "idx_norm", "c"),
    LLMP_DSV4_MOE_TENSORS("moe_hash"),
    LLMP_DSV4_MOE_TENSORS("moe"),
    Plain("blk.{layer}.exp_probs_b.bias", "moe", "router_bias"),
    Plain(names::kOutput.gguf, "output", "weight"),
};

constexpr std::array kParams = {
    LLMP_DSV4_BLOCK_PARAMS,
    Real("embedding", "scale", 1.0),
    FromKey("hash", "experts", "expert_count", kInt),
    FromKey("hash", "used", "expert_used_count", kInt),
    Name("attention", "cache", "ring"),
    LLMP_DSV4_MOE_PARAMS("moe_hash"),
    Name("moe_hash", "router", "hash"),
    Name("moe_hash", "hash", "hash_{layer}"),
    LLMP_DSV4_MOE_PARAMS("moe"),
    Name("moe", "router", "sqrt_softplus"),
    // sqrt(softplus) gating (LLAMA_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS).
    RequireInt("expert_gating_func", 4, true),
};

// Its serving: the derived shared route, in its automatic shape only (no
// qualified envelopes of its own), with kDeepSeekServingPolicy, prompt
// chunks each alone and none captured ahead; its fast plan's attention mask
// bounds a prompt chunk's rows (RE-037); 4,096-row prompt chunks (2,048 above
// 262,144 tokens: the mask's memory); at most eight owners while speculating
// (a DSpark verify takes two of a wave's sixteen rows); adaptive waves on
// lanes of their own. Its chat route thinks unless a request turns thinking
// off.
constexpr PolicyRule kServingRule{.text = kDeepSeekServingPolicy,
                                  .prompts_alone = true,
                                  .common_read = true,
                                  .capture_ahead = false};
// With DSpark: a draft-verify wave's time over a plain decode wave's,
// widths 2 to 8 (execution/adaptive_wave_mode.h): the median joined wave of
// the retired `llmp_spec_runner --check wave --slots N --wave-mode
// verify|decode` (GB10, community GGUF, wave lanes on, 2026-10-03;
// docs/experiments/deepseek-batching, "Wave lanes" and "Joined draft
// blocks"): with joined drafts, 127.5 / 61.3, 180.0 / 71.5, 231.2 / 82.3
// and 229.0 / 95.6 ms at widths 2 to 5; with lanes, 224.7 / 106.1, 258.0 /
// 116.6 and 287.3 / 129.1 ms at widths 6 to 8 (D-104; past five a joined
// draft's rows exceed a wave's, so each block runs alone). A DSpark verify
// then takes three rows a request at width 5 and two past it, so past five
// a draft-verify wave (at most two tokens a request) never pays
// (docs/experiments/request-slots). To be measured again when a wave's
// cost changes.
constexpr std::array kWaveCosts{2.08, 2.52, 2.81, 2.40, 2.12, 2.21, 2.23};
constexpr ServingRecipe kServing{.family = "DeepSeek V4",
                                 .name = "DeepSeek V4",
                                 .prefill_rows = 4096,
                                 .deep_context = 262144,
                                 .deep_prefill_rows = 2048,
                                 .draft_rows = true,
                                 .block_drafts = true,
                                 .adaptive_waves = true,
                                 // DSpark's drafts a step (its verify takes
                                 // one row more), and the state a joined
                                 // wave reads rounded as Qwen3.8's.
                                 .default_draft_rows = 3,
                                 .default_wave_costs = kWaveCosts,
                                 .default_wave_lanes = true,
                                 .default_wave_read_align = 2048,
                                 .default_thinking = true,

                                 .frozen_host_masks = false,

                                 .serving_rule = &kServingRule};

constexpr std::array kGates = {GateRule{.key = "deepseek4.expert_count", .kind = "ffn.moe"}};
constexpr std::array kDsv4Requirements{
    ScalarRequirement{.key = "attention.value_length", .equals_key = "attention.key_length"}};
constexpr std::array kDsv4Auxiliary{
    kGgufAuxiliary[0], kGgufAuxiliary[1], kGgufAuxiliary[2],
    AuxiliaryNamespace{"deepseek4.nextn_predict_layers", "MTP layers the import does not hold"},
    AuxiliaryNamespace{"deepseek4.vocab_size", "the vocabulary the token table states",
                       names::kTokenEmbd.gguf, 1}};
constexpr std::array kDsv4Lists{
    ListRequirement{.key = "attention.compress_ratios",
                    .size = kLayerLimit,
                    .min_size = 1,
                    .per_layer = true,
                    .extra_layers_key = "nextn_predict_layers",
                    .allow_unstated_extra = true},
    ListRequirement{
        .key = "swiglu_clamp_exp", .size = kLayerLimit, .min_size = 1, .per_layer = true},
    ListRequirement{.key = "swiglu_clamp_shexp",
                    .size = kLayerLimit,
                    .min_size = 1,
                    .optional = true,
                    .per_layer = true}};

constexpr Mapping kMapping{
    .architecture = "deepseek4",
    .version = 1,
    .pipeline = PipelineKind::kLlm,
    .layer_count_key = "block_count",
    .layer_types = "wch",
    .layer_type_key = "attention.compress_ratios",
    .layer_type_codes = kCodes,
    .slots = kSlots,
    .tensors = kTensors,
    .params = kParams,
    .serving = &kServing,
    .containers = kGgufContainers,
    .gates = kGates,
    .auxiliary = kDsv4Auxiliary,
    .requirements = kDsv4Requirements,
    .lists = kDsv4Lists,
    .extended_components = true,
};

// ---------------------------------------------------------------- dflash

// The drafter's ring: its window and a draft block, in attention's
// 256-cell steps.
constexpr std::int64_t kDsparkRing = 256;

constexpr std::array kDsparkSlots = {
    SlotRule{.slot = "embedding", .site = Site::kEmbedding, .kind = "embedding.inject"},
    LLMP_DSV4_BLOCK_SLOTS,
    SlotRule{.slot = "moe", .site = Site::kLayer, .kind = "ffn.moe"},
    SlotRule{.slot = "ffn_combine", .site = Site::kLayer, .kind = "hc.combine"},
    SlotRule{.slot = "output_mix", .site = Site::kFinalNorm, .kind = "hc.mix"},
    SlotRule{.slot = "output", .site = Site::kOutput, .kind = "head.linear"},
};

constexpr std::array kDsparkTensors = {
    TensorRule{
        .pattern = names::kTokenEmbd.gguf, .slot = "embedding", .role = "table", .target = true},
    Plain("fc.weight", "embedding", "fc"),
    Plain("enc.output_norm.weight", "embedding", "norm"),
    LLMP_DSV4_BLOCK_TENSORS,
    LLMP_DSV4_MOE_TENSORS("moe"),
    Plain("blk.{layer}.exp_probs_b.bias", "moe", "router_bias"),
    TensorRule{.pattern = names::kOutput.gguf, .slot = "output", .role = "weight", .target = true},
    Plain("markov_w1.weight", "output", "markov_w1"),
    Plain("markov_w2.weight", "output", "markov_w2"),
    // The confidence head: bound by today's runner, never read.
    TensorRule{.pattern = "conf_proj.weight", .slot = "output", .unbound = true},
};

constexpr std::array kDsparkParams = {
    LLMP_DSV4_BLOCK_PARAMS,
    FromKey("embedding", "eps", "attention.layer_norm_rms_epsilon", kReal),
    Name("embedding", "tap", "features"),
    Int("embedding", "taps", 3),
    Int("embedding", "cells", kDsparkRing),
    Name("attention", "cache", "injected"),
    Int("attention", "cells", kDsparkRing),
    LLMP_DSV4_MOE_PARAMS("moe"),
    Name("moe", "router", "sqrt_softplus"),
    Flag("output_mix", "gather", false),
    Flag("output", "gather", false),
    Flag("output", "markov", true),
    // Every block routed by its bias, sqrt(softplus).
    RequireInt("hash_layer_count", 0, true),
    RequireInt("expert_gating_func", 4, true),
};

// The streams' means entering the target layers it names.
constexpr std::array kDsparkReads = {
    ReadRule{.kind = ReadKind::kTap,
             .name = "features_0",
             .point = TapPoint::kStreamMean,
             .layers_key = "target_layers",
             .index = 0,
             .entering = true},
    ReadRule{.kind = ReadKind::kTap,
             .name = "features_1",
             .point = TapPoint::kStreamMean,
             .layers_key = "target_layers",
             .index = 1,
             .entering = true},
    ReadRule{.kind = ReadKind::kTap,
             .name = "features_2",
             .point = TapPoint::kStreamMean,
             .layers_key = "target_layers",
             .index = 2,
             .entering = true},
};

// A block of proposals: at most dflash.block_size drafts, filled with
// tokenizer.ggml.mask_token_id, both from the drafter's kept metadata.
constexpr DraftRecipe kDsparkDraft{.shape = DraftRecipe::Shape::kBlock,
                                   .depth_key = "dflash.block_size",
                                   .fill_key = "tokenizer.ggml.mask_token_id",
                                   .target_architecture = "deepseek4"};

constexpr std::array kDsparkPolicies = {PolicyRule{.text = kDsparkServingPolicy}};
constexpr std::array kDsparkRequirements{
    ScalarRequirement{.key = "attention.value_length", .equals_key = "attention.key_length"},
    ScalarRequirement{.key = "block_size", .positive = true}};
constexpr std::array kDsparkLists{
    ListRequirement{
        .key = "attention.compress_ratios", .size = kLayerLimit, .min_size = 1, .per_layer = true},
    ListRequirement{
        .key = "swiglu_clamp_exp", .size = kLayerLimit, .min_size = 1, .per_layer = true},
    ListRequirement{
        .key = "swiglu_clamp_shexp", .size = kLayerLimit, .min_size = 1, .per_layer = true},
    ListRequirement{.key = "target_layers", .size = kLayerLimit, .min_size = 1}};

constexpr Mapping kDsparkMapping{
    .architecture = "dflash",
    .version = 1,
    .pipeline = PipelineKind::kLlm,
    .layer_count_key = "block_count",
    .layer_types = "wch",
    .layer_type_key = "attention.compress_ratios",
    .layer_type_codes = std::span(kCodes).first(1),
    .slots = kDsparkSlots,
    .tensors = kDsparkTensors,
    .params = kDsparkParams,
    .role = StageRole::kDrafter,
    .stage = "dspark",
    .reads = kDsparkReads,
    .policies = kDsparkPolicies,
    .draft = &kDsparkDraft,
    .containers = kGgufContainers,
    .auxiliary = kGgufAuxiliary,
    .requirements = kDsparkRequirements,
    .lists = kDsparkLists,
    .extended_components = true,
};

#undef LLMP_DSV4_BLOCK_SLOTS
#undef LLMP_DSV4_BLOCK_TENSORS
#undef LLMP_DSV4_MOE_TENSORS
#undef LLMP_DSV4_BLOCK_PARAMS
#undef LLMP_DSV4_MOE_PARAMS

}  // namespace

const Mapping& Dsv4Mapping() { return kMapping; }
const Mapping& DsparkMapping() { return kDsparkMapping; }

}  // namespace llmp::spec::mappings
