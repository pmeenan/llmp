// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "spec/mappings/dsv4.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <expected>
#include <string>
#include <string_view>
#include <vector>

#include "model/dspark.h"
#include "model/dsv4.h"
#include "spec/mapping.h"
#include "spec/mappings/rules.h"
#include "spec/spec.h"

namespace llmp::spec::mappings {
namespace {

using enum ValueType;

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
    LayerTypeCode{.value = model::kDsv4CsaRatio, .type = 'c'},
    LayerTypeCode{.value = model::kDsv4HcaRatio, .type = 'h'},
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
      Plain("blk.{layer}.attn_norm.weight", "attn_mix", "post_norm"),     \
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
      Plain("blk.{layer}.ffn_norm.weight", "ffn_mix", "post_norm"),       \
      Plain("output_hc_fn.weight", "output_mix", "fn"),                   \
      Plain("output_hc_base.weight", "output_mix", "base"),               \
      Plain("output_hc_scale.weight", "output_mix", "scale"),             \
      Plain("output_norm.weight", "output_mix", "post_norm")

// An MoE slot's tensors.
#define LLMP_DSV4_MOE_TENSORS(SLOT)                                      \
  Plain("blk.{layer}.ffn_gate_inp.weight", SLOT, "router"),              \
      Experts("blk.{layer}.ffn_gate_exps.weight", SLOT, "gate_experts"), \
      Experts("blk.{layer}.ffn_up_exps.weight", SLOT, "up_experts"),     \
      Experts("blk.{layer}.ffn_down_exps.weight", SLOT, "down_experts"), \
      Plain("blk.{layer}.ffn_gate_shexp.weight", SLOT, "gate"),          \
      Plain("blk.{layer}.ffn_up_shexp.weight", SLOT, "up"),              \
      Plain("blk.{layer}.ffn_down_shexp.weight", SLOT, "down")

// The block parameters both mappings read, under a prefix's keys.
#define LLMP_DSV4_BLOCK_PARAMS(P)                                                                  \
  Name("streams", "mix", "sinkhorn"), Name("attn_mix", "mix", "sinkhorn"),                         \
      FromKey("attn_mix", "eps", P "attention.layer_norm_rms_epsilon", kReal),                     \
      FromKey("attn_mix", "hc_eps", P "hyper_connection.epsilon", kReal),                          \
      FromKey("attn_mix", "iterations", P "hyper_connection.sinkhorn_iterations", kInt),           \
      Name("attn_combine", "mix", "sinkhorn"), Name("ffn_mix", "mix", "sinkhorn"),                 \
      FromKey("ffn_mix", "eps", P "attention.layer_norm_rms_epsilon", kReal),                      \
      FromKey("ffn_mix", "hc_eps", P "hyper_connection.epsilon", kReal),                           \
      FromKey("ffn_mix", "iterations", P "hyper_connection.sinkhorn_iterations", kInt),            \
      Name("ffn_combine", "mix", "sinkhorn"), Name("output_mix", "mix", "sinkhorn"),               \
      FromKey("output_mix", "eps", P "attention.layer_norm_rms_epsilon", kReal),                   \
      FromKey("output_mix", "hc_eps", P "hyper_connection.epsilon", kReal),                        \
      Flag("output_mix", "inject", false),                                                         \
      FromKey("attention", "eps", P "attention.layer_norm_rms_epsilon", kReal),                    \
      FromKey("attention", "heads", P "attention.head_count", kInt),                               \
      FromKey("attention", "head_dim", P "attention.key_length", kInt),                            \
      FromKey("attention", "rope_dims", P "rope.dimension_count", kInt),                           \
      FromKey("attention", "q_rank", P "attention.q_lora_rank", kInt),                             \
      FromKey("attention", "o_rank", P "attention.output_lora_rank", kInt),                        \
      FromKey("attention", "o_groups", P "attention.output_group_count", kInt),                    \
      FromKey("attention", "window", P "attention.sliding_window", kInt),                          \
      FromKey("attention", "rope_base", P "rope.freq_base", kReal, Transform::kNone, "w"),         \
      FromKey("attention", "rope_base", P "attention.compress_rope_freq_base", kReal,              \
              Transform::kNone, "ch"),                                                             \
      FromKey("attention", "rope_factor", P "rope.scaling.factor", kReal, Transform::kNone, "ch"), \
      FromKey("attention", "yarn_context", P "rope.scaling.original_context_length", kInt,         \
              Transform::kNone, "ch"),                                                             \
      FromKey("attention", "beta_fast", P "rope.scaling.yarn_beta_fast", kReal, Transform::kNone,  \
              "ch"),                                                                               \
      FromKey("attention", "beta_slow", P "rope.scaling.yarn_beta_slow", kReal, Transform::kNone,  \
              "ch"),                                                                               \
      Int("attention", "ratio", model::kDsv4CsaRatio, "c"),                                        \
      Int("attention", "ratio", model::kDsv4HcaRatio, "h"),                                        \
      Flag("attention", "overlap", true, "c"), Flag("attention", "indexer", true, "c"),            \
      FromKey("attention", "indexer_heads", P "attention.indexer.head_count", kInt,                \
              Transform::kNone, "c"),                                                              \
      FromKey("attention", "index_dim", P "attention.indexer.key_length", kInt, Transform::kNone,  \
              "c"),                                                                                \
      FromKey("attention", "top_k", P "attention.indexer.top_k", kInt, Transform::kNone, "c"),     \
      FromKey(kStageSlot, "context", P "context_length", kInt),                                    \
      FromKey(kStreamSlot, "width", P "embedding_length", kInt),                                   \
      FromKey(kStreamSlot, "streams", P "hyper_connection.count",                                  \
              kInt), /* One KV head; the rotation's YaRN where it is scaled. */                    \
      RequireInt(P "attention.head_count_kv", 1, false),                                           \
      Require(P "rope.scaling.type", "yarn", true), RequireInt(P "expert_shared_count", 1, false)

// An MoE slot's parameters (its router aside).
#define LLMP_DSV4_MOE_PARAMS(P, SLOT)                                                       \
  Name(SLOT, "placement", "none"),                                                          \
      FromKey(SLOT, "eps", P "attention.layer_norm_rms_epsilon", kReal),                    \
      Name(SLOT, "activation", "silu_clamp"),                                               \
      FromKey(SLOT, "hidden", P "expert_feed_forward_length", kInt),                        \
      FromKey(SLOT, "experts", P "expert_count", kInt),                                     \
      FromKey(SLOT, "used", P "expert_used_count", kInt),                                   \
      FromKey(SLOT, "expert_hidden", P "expert_feed_forward_length", kInt),                 \
      Name(SLOT, "router_input", "stream"),                                                 \
      FromKey(SLOT, "normalize", P "expert_weights_norm", kBool),                           \
      FromKey(SLOT, "route_scale", P "expert_weights_scale", kReal),                        \
      Flag(SLOT, "router_f32", true), PerLayer(SLOT, "limit", P "swiglu_clamp_exp", kReal), \
      PerLayer(SLOT, "shared_limit", P "swiglu_clamp_shexp", kReal)

// ---------------------------------------------------------------- deepseek4

constexpr std::string_view kHashLayers = "deepseek4.hash_layer_count";

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
    Plain("token_embd.weight", "embedding", "table"),
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
    Plain("output.weight", "output", "weight"),
};

constexpr std::array kParams = {
    LLMP_DSV4_BLOCK_PARAMS("deepseek4."),
    Real("embedding", "scale", 1.0),
    FromKey("hash", "experts", "deepseek4.expert_count", kInt),
    FromKey("hash", "used", "deepseek4.expert_used_count", kInt),
    Name("attention", "cache", "ring"),
    LLMP_DSV4_MOE_PARAMS("deepseek4.", "moe_hash"),
    Name("moe_hash", "router", "hash"),
    Name("moe_hash", "hash", "hash_{layer}"),
    LLMP_DSV4_MOE_PARAMS("deepseek4.", "moe"),
    Name("moe", "router", "sqrt_softplus"),
    // sqrt(softplus) gating (LLAMA_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS).
    RequireInt("deepseek4.expert_gating_func", 4, true),
};

// Its serving: the derived shared route's automatic shape (else only the
// closed single-owner diagnostic bridge), with kDeepSeekServingPolicy, prompt
// chunks each alone and none captured ahead; its fast plan's attention mask
// bounds a prompt chunk's rows (RE-037); 4,096-row prompt chunks (2,048 above
// 262,144 tokens: the mask's memory); at most eight owners while speculating
// (a DSpark verify takes two of a wave's sixteen rows); adaptive waves on
// lanes of their own. Its chat route thinks unless a request turns thinking
// off.
constexpr PolicyRule kServingRule{
    .text = kDeepSeekServingPolicy, .prompts_alone = true, .capture_ahead = false};
constexpr ServingRecipe kServing{
    .family = "DeepSeek V4",
    .name = "DeepSeek V4",
    .prefill_rows = 4096,
    .deep_context = 262144,
    .deep_prefill_rows = 2048,
    .speculative_owners = 8,
    .draft_rows = true,
    .block_drafts = true,
    .adaptive_waves = true,
    .default_thinking = true,
    .shared_route = true,
    .closed_bridge = true,
    .most_rows =
        +[](std::uint32_t context) { return model::Dsv4MostRows(model::Dsv4Flash(), context); },
    .frozen_host_masks = false,
    .lanes = true,
    .lid_mask = true,
    .serving_rule = &kServingRule};

constexpr Mapping kMapping{
    .architecture = "deepseek4",
    .version = 1,
    .pipeline = PipelineKind::kLlm,
    .layer_count_key = "deepseek4.block_count",
    .layer_types = "wch",
    .layer_type_key = "deepseek4.attention.compress_ratios",
    .layer_type_codes = kCodes,
    .slots = kSlots,
    .tensors = kTensors,
    .params = kParams,
    .serving = &kServing,
};

// ---------------------------------------------------------------- dflash

// The drafter's ring: its window and a draft block, in attention's
// 256-cell steps (model/dspark.h DsparkDeepSeekV4Flash's).
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
        .pattern = "token_embd.weight", .slot = "embedding", .role = "table", .target = true},
    Plain("fc.weight", "embedding", "fc"),
    Plain("enc.output_norm.weight", "embedding", "norm"),
    LLMP_DSV4_BLOCK_TENSORS,
    LLMP_DSV4_MOE_TENSORS("moe"),
    Plain("blk.{layer}.exp_probs_b.bias", "moe", "router_bias"),
    TensorRule{.pattern = "output.weight", .slot = "output", .role = "weight", .target = true},
    Plain("markov_w1.weight", "output", "markov_w1"),
    Plain("markov_w2.weight", "output", "markov_w2"),
    // The confidence head: bound by today's runner, never read.
    TensorRule{.pattern = "conf_proj.weight", .slot = "output", .unbound = true},
};

constexpr std::array kDsparkParams = {
    LLMP_DSV4_BLOCK_PARAMS("dflash."),
    FromKey("embedding", "eps", "dflash.attention.layer_norm_rms_epsilon", kReal),
    Name("embedding", "tap", "features"),
    Int("embedding", "taps", 3),
    Int("embedding", "cells", kDsparkRing),
    Name("attention", "cache", "injected"),
    Int("attention", "cells", kDsparkRing),
    LLMP_DSV4_MOE_PARAMS("dflash.", "moe"),
    Name("moe", "router", "sqrt_softplus"),
    Flag("output_mix", "gather", false),
    Flag("output", "gather", false),
    Flag("output", "markov", true),
    // Every block routed by its bias, sqrt(softplus).
    RequireInt("dflash.hash_layer_count", 0, true),
    RequireInt("dflash.expert_gating_func", 4, true),
};

// The streams' means entering the target layers it names.
constexpr std::array kDsparkReads = {
    ReadRule{.kind = ReadKind::kTap,
             .name = "features_0",
             .point = TapPoint::kStreamMean,
             .layers_key = "dflash.target_layers",
             .index = 0,
             .entering = true},
    ReadRule{.kind = ReadKind::kTap,
             .name = "features_1",
             .point = TapPoint::kStreamMean,
             .layers_key = "dflash.target_layers",
             .index = 1,
             .entering = true},
    ReadRule{.kind = ReadKind::kTap,
             .name = "features_2",
             .point = TapPoint::kStreamMean,
             .layers_key = "dflash.target_layers",
             .index = 2,
             .entering = true},
};

// A block of proposals: at most dflash.block_size drafts, filled with
// tokenizer.ggml.mask_token_id, both from the drafter's kept metadata.
constexpr DraftRecipe kDsparkDraft{.shape = DraftRecipe::Shape::kBlock,
                                   .depth_key = "dflash.block_size",
                                   .fill_key = "tokenizer.ggml.mask_token_id"};

constexpr Mapping kDsparkMapping{
    .architecture = "dflash",
    .version = 1,
    .pipeline = PipelineKind::kLlm,
    .layer_count_key = "dflash.block_count",
    .layer_types = "wch",
    .layer_type_key = "dflash.attention.compress_ratios",
    .layer_type_codes = std::span(kCodes).first(1),
    .slots = kDsparkSlots,
    .tensors = kDsparkTensors,
    .params = kDsparkParams,
    .role = StageRole::kDrafter,
    .stage = "dspark",
    .reads = kDsparkReads,
    .draft = &kDsparkDraft,
};

#undef LLMP_DSV4_BLOCK_SLOTS
#undef LLMP_DSV4_BLOCK_TENSORS
#undef LLMP_DSV4_MOE_TENSORS
#undef LLMP_DSV4_BLOCK_PARAMS
#undef LLMP_DSV4_MOE_PARAMS

SourceValue Reals(std::string_view key, std::vector<double> v) {
  return {.key = std::string(key), .value = ParamValue(std::move(v))};
}

// The keys both mappings' blocks read, under `p`.
void BlockValues(std::vector<SourceValue>& out, const std::string& p, const model::Dsv4Profile& q) {
  const auto key = [&](std::string_view k) { return p + std::string(k); };
  std::vector<std::int64_t> ratios(q.compress_ratios.begin(), q.compress_ratios.end());
  for (SourceValue v : std::vector<SourceValue>{
           Value(key("block_count"), q.layers),
           Value(key("context_length"), model::kDsv4FlashContext),
           Value(key("embedding_length"), q.width),
           Value(key("attention.head_count"), q.heads),
           Value(key("attention.head_count_kv"), 1U),
           Value(key("attention.key_length"), q.head_dim),
           Value(key("attention.value_length"), q.head_dim),
           Value(key("rope.dimension_count"), q.rope_dims),
           Value(key("attention.q_lora_rank"), q.q_lora),
           Value(key("attention.output_lora_rank"), q.o_lora),
           Value(key("attention.output_group_count"), q.o_groups),
           Value(key("attention.sliding_window"), q.window),
           Value(key("expert_count"), q.experts),
           Value(key("expert_used_count"), q.experts_used),
           Value(key("expert_feed_forward_length"), q.expert_ffn),
           Value(key("expert_shared_count"), q.shared_experts),
           Value(key("hash_layer_count"), q.hash_layers),
           Value(key("hyper_connection.count"), q.hc),
           Value(key("hyper_connection.sinkhorn_iterations"), q.sinkhorn_iterations),
           Value(key("attention.indexer.head_count"), q.indexer_heads),
           Value(key("attention.indexer.key_length"), q.indexer_head_dim),
           Value(key("attention.indexer.top_k"), q.indexer_top_k),
           Value(key("rope.scaling.original_context_length"), q.yarn_original_context),
           Value(key("attention.layer_norm_rms_epsilon"), q.rms_eps),
           Value(key("hyper_connection.epsilon"), q.hc_eps),
           Value(key("rope.freq_base"), q.rope_base),
           Value(key("attention.compress_rope_freq_base"), q.compress_rope_base),
           Value(key("rope.scaling.factor"), q.rope_scale),
           Value(key("rope.scaling.type"), "yarn"),
           Value(key("rope.scaling.yarn_beta_fast"), q.yarn_beta_fast),
           Value(key("rope.scaling.yarn_beta_slow"), q.yarn_beta_slow),
           Value(key("expert_weights_scale"), q.expert_weights_scale),
           FlagValue(key("expert_weights_norm"), q.expert_weights_norm),
           Value(key("expert_gating_func"), 4U),
           Reals(key("swiglu_clamp_exp"),
                 std::vector<double>(q.layers, static_cast<double>(q.swiglu_limit))),
           Reals(key("swiglu_clamp_shexp"),
                 std::vector<double>(q.layers, static_cast<double>(q.swiglu_limit_shared))),
           Value(key("attention.compress_ratios"), std::move(ratios)),
       }) {
    out.push_back(std::move(v));
  }
}

}  // namespace

const Mapping& Dsv4Mapping() { return kMapping; }
const Mapping& DsparkMapping() { return kDsparkMapping; }

std::expected<std::vector<SourceValue>, std::string> Dsv4V0Values(const model::Dsv4Profile& p) {
  if (p.compress_ratios.size() != p.layers || p.shared_experts != 1 || p.hc < 2) {
    return std::unexpected("the DeepSeek V4 profile is not the mapping's form");
  }
  std::vector<SourceValue> out;
  BlockValues(out, "deepseek4.", p);
  return out;
}

std::expected<std::vector<SourceValue>, std::string> DsparkV0Values(const model::DsparkProfile& d) {
  const model::Dsv4Profile& p = d.blocks;
  if (p.hash_layers != 0 || d.ring != kDsparkRing || d.target_layers.size() != 3 ||
      std::ranges::any_of(p.compress_ratios, [](std::uint32_t r) { return r != 0; })) {
    return std::unexpected("the DSpark profile is not the mapping's form");
  }
  std::vector<SourceValue> out;
  BlockValues(out, "dflash.", p);
  out.push_back(Value("dflash.block_size", d.block_size));
  out.push_back(Value("dflash.target_layers",
                      std::vector<std::int64_t>(d.target_layers.begin(), d.target_layers.end())));
  return out;
}

}  // namespace llmp::spec::mappings
