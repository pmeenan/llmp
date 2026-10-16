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
#include "spec/mappings/rules.h"
#include "spec/spec.h"

namespace llmp::spec::mappings {
namespace {

using enum ValueType;

// The compression ratios of a CSA and an HCA layer.
constexpr std::uint32_t kCsaRatio = 4;
constexpr std::uint32_t kHcaRatio = 128;

// DeepSeek V4 Flash, as the deepseek4.* keys of DeepSeek-V4-Flash-0731-
// UD-Q2_K_XL (and the e3aa0d6a revision) hold it (checked by
// docs/experiments/dsv4-native/gguf_profile_check.py; the epsilons are the
// GGUF's float32 values): what its serving recipe's prompt bound and its
// DSpark drafter's import (its target's values and tensors) are made of.
struct Flash {
  std::uint32_t layers = 43, width = 4096, heads = 64, head_dim = 512, rope_dims = 64;
  std::uint32_t q_lora = 1024, o_lora = 1024, o_groups = 8, window = 128;
  std::uint32_t experts = 256, experts_used = 6, expert_ffn = 2048, shared_experts = 1;
  std::uint32_t hash_layers = 3, hc = 4, sinkhorn_iterations = 20;
  std::uint32_t indexer_heads = 64, indexer_head_dim = 128, indexer_top_k = 512;
  std::uint32_t vocab = 129280, context = 1048576, yarn_original_context = 65536;
  float rms_eps = 1e-6F, hc_eps = 1e-6F, rope_base = 10000.0F, compress_rope_base = 160000.0F;
  float rope_scale = 16.0F, yarn_beta_fast = 32.0F, yarn_beta_slow = 1.0F;
  float expert_weights_scale = 1.5F, swiglu_limit = 10.0F, swiglu_limit_shared = 10.0F;
  // Layers 0 and 1 window-only; then CSA and HCA alternate from layer 2,
  // the last layer CSA.
  std::uint32_t ratio(std::uint32_t layer) const {
    return layer < 2 ? 0 : (layer % 2 == 0 ? kCsaRatio : kHcaRatio);
  }
  std::uint64_t hc_width() const { return std::uint64_t{hc} * width; }
  std::uint64_t hc_mix() const { return std::uint64_t{2 + hc} * hc; }
};
constexpr Flash kFlash{};

// The widest prompt chunk at `context`: the fast plan's attention mask, F16
// [ring cells + compressed cells, rows], stays under 2^31 bytes (the MMA
// kernel takes its planes' strides in 32 bits, RE-037); the largest such
// rows, by bisection (the mask grows with the rows).
std::uint32_t MostRows(std::uint32_t context) {
  const auto pad = [](std::uint64_t n) { return (n + 255) / 256 * 256; };
  if (context == 0 || context > kFlash.context ||
      context > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) - 255) {
    return 0;
  }
  const std::uint64_t cells = pad(context);
  if (cells <= kFlash.window) {
    return 0;
  }
  const std::uint64_t compressed = pad((std::uint64_t{context} + kCsaRatio - 1) / kCsaRatio);
  const auto fits = [&](std::uint64_t rows) {
    const std::uint64_t ring = std::min(cells, pad(std::uint64_t{kFlash.window} + rows));
    return (ring + compressed) * rows * 2 <=
           static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max());
  };
  std::uint64_t lo = 0;
  std::uint64_t hi = std::min<std::uint64_t>(context, cells - kFlash.window);
  while (lo < hi) {
    const std::uint64_t mid = hi - ((hi - lo) / 2);
    if (fits(mid)) {
      lo = mid;
    } else {
      hi = mid - 1;
    }
  }
  return static_cast<std::uint32_t>(lo);
}

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
      Int("attention", "ratio", kCsaRatio, "c"), Int("attention", "ratio", kHcaRatio, "h"),        \
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

// An MoE slot's parameters (its router aside). A source listing no shared
// expert clamps (antirez's GGUFs) clamps its shared expert as its routed
// ones, as the M3 profile served it.
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
      PerLayer(SLOT, "shared_limit", P "swiglu_clamp_shexp", kReal),                        \
      PerLayer(SLOT, "shared_limit", P "swiglu_clamp_exp", kReal)

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

// Its serving: the derived shared route, in its automatic shape only (no
// qualified envelopes of its own), with kDeepSeekServingPolicy, prompt
// chunks each alone and none captured ahead; its fast plan's attention mask
// bounds a prompt chunk's rows (RE-037); 4,096-row prompt chunks (2,048 above
// 262,144 tokens: the mask's memory); at most eight owners while speculating
// (a DSpark verify takes two of a wave's sixteen rows); adaptive waves on
// lanes of their own. Its chat route thinks unless a request turns thinking
// off.
constexpr PolicyRule kServingRule{
    .text = kDeepSeekServingPolicy, .prompts_alone = true, .capture_ahead = false};
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
                                 .speculative_owners = 8,
                                 .draft_rows = true,
                                 .block_drafts = true,
                                 .adaptive_waves = true,
                                 // DSpark's drafts a step (its verify takes
                                 // one row more), and the state a joined
                                 // wave reads rounded as Qwen3.8's.
                                 .default_draft_rows = 3,
                                 .default_wave_costs = kWaveCosts,
                                 .default_wave_read_align = 2048,
                                 .default_thinking = true,
                                 .shared_route = true,
                                 .most_rows = &MostRows,
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

SourceValue Reals(std::string_view key, std::vector<double> v) {
  return {.key = std::string(key), .value = ParamValue(std::move(v))};
}

// Its deepseek4.* keys.
std::expected<std::vector<SourceValue>, std::string> TargetValues() {
  const Flash& q = kFlash;
  const auto key = [&](std::string_view k) { return "deepseek4." + std::string(k); };
  std::vector<std::int64_t> ratios;
  for (std::uint32_t il = 0; il < q.layers; ++il) ratios.push_back(q.ratio(il));
  return std::vector<SourceValue>{
      Value(key("block_count"), q.layers),
      Value(key("context_length"), q.context),
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
      FlagValue(key("expert_weights_norm"), true),
      Value(key("expert_gating_func"), 4U),
      Reals(key("swiglu_clamp_exp"),
            std::vector<double>(q.layers, static_cast<double>(q.swiglu_limit))),
      Reals(key("swiglu_clamp_shexp"),
            std::vector<double>(q.layers, static_cast<double>(q.swiglu_limit_shared))),
      Value(key("attention.compress_ratios"), std::move(ratios)),
  };
}

// Its tensors as its GGUF names them: BF16 matrices, F32 vectors and I32
// hash tables.
Source TargetSource() {
  const Flash& p = kFlash;
  Source s{.architecture = "deepseek4", .container = "gguf"};
  const auto add = [&](std::string name, std::vector<std::uint64_t> ne, bool experts = false,
                       std::optional<format::Encoding> enc = std::nullopt) {
    const auto encoding =
        enc.value_or(ne.size() == 1 ? format::Encoding::kF32 : format::Encoding::kBF16);
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
  add("output_hc_fn.weight", {p.hc_width(), p.hc});
  add("output_hc_base.weight", {p.hc});
  add("output_hc_scale.weight", {1});
  const std::uint64_t w = p.width;
  const std::uint64_t head = p.head_dim;
  for (std::uint32_t il = 0; il < p.layers; ++il) {
    const std::string prefix = std::format("blk.{}.", il);
    add(prefix + "attn_norm.weight", {w});
    add(prefix + "attn_sinks.weight", {p.heads});
    add(prefix + "attn_q_a.weight", {w, p.q_lora});
    add(prefix + "attn_q_a_norm.weight", {p.q_lora});
    add(prefix + "attn_q_b.weight", {p.q_lora, std::uint64_t{p.heads} * head});
    add(prefix + "attn_kv.weight", {w, head});
    add(prefix + "attn_kv_a_norm.weight", {head});
    add(prefix + "attn_output_a.weight",
        {std::uint64_t{p.heads} * head / p.o_groups, std::uint64_t{p.o_lora} * p.o_groups});
    add(prefix + "attn_output_b.weight", {std::uint64_t{p.o_lora} * p.o_groups, w});
    add(prefix + "hc_attn_fn.weight", {p.hc_width(), p.hc_mix()});
    add(prefix + "hc_attn_base.weight", {p.hc_mix()});
    add(prefix + "hc_attn_scale.weight", {3});
    add(prefix + "hc_ffn_fn.weight", {p.hc_width(), p.hc_mix()});
    add(prefix + "hc_ffn_base.weight", {p.hc_mix()});
    add(prefix + "hc_ffn_scale.weight", {3});
    const std::uint32_t ratio = p.ratio(il);
    if (ratio != 0) {
      const std::uint64_t coff = ratio == kCsaRatio ? 2 : 1;
      add(prefix + "attn_compressor_kv.weight", {w, coff * head});
      add(prefix + "attn_compressor_gate.weight", {w, coff * head});
      add(prefix + "attn_compressor_ape.weight", {coff * head, ratio});
      add(prefix + "attn_compressor_norm.weight", {head});
    }
    if (ratio == kCsaRatio) {
      const std::uint64_t ih = p.indexer_head_dim;
      add(prefix + "indexer.attn_q_b.weight", {p.q_lora, std::uint64_t{p.indexer_heads} * ih});
      add(prefix + "indexer.proj.weight", {w, p.indexer_heads});
      add(prefix + "indexer_compressor_kv.weight", {w, 2 * ih});
      add(prefix + "indexer_compressor_gate.weight", {w, 2 * ih});
      add(prefix + "indexer_compressor_ape.weight", {2 * ih, kCsaRatio});
      add(prefix + "indexer_compressor_norm.weight", {ih});
    }
    add(prefix + "ffn_norm.weight", {w});
    add(prefix + "ffn_gate_inp.weight", {w, p.experts});
    if (il < p.hash_layers) {
      add(prefix + "ffn_gate_tid2eid.weight", {p.experts_used, p.vocab}, false,
          format::Encoding::kI32);
    } else {
      add(prefix + "exp_probs_b.bias", {p.experts});
    }
    const std::uint64_t shared = std::uint64_t{p.expert_ffn} * p.shared_experts;
    add(prefix + "ffn_gate_shexp.weight", {w, shared});
    add(prefix + "ffn_up_shexp.weight", {w, shared});
    add(prefix + "ffn_down_shexp.weight", {shared, w});
    add(prefix + "ffn_gate_exps.weight", {w, p.expert_ffn, p.experts}, true);
    add(prefix + "ffn_up_exps.weight", {w, p.expert_ffn, p.experts}, true);
    add(prefix + "ffn_down_exps.weight", {p.expert_ffn, w, p.experts}, true);
  }
  return s;
}

// The DSpark drafter's import composes it with DeepSeek V4 Flash.
constexpr std::array kDsparkTargets = {
    ImportTarget{.mapping = &Dsv4Mapping, .values = &TargetValues, .source = &TargetSource},
};

// A block of proposals: at most dflash.block_size drafts, filled with
// tokenizer.ggml.mask_token_id, both from the drafter's kept metadata.
constexpr DraftRecipe kDsparkDraft{.shape = DraftRecipe::Shape::kBlock,
                                   .depth_key = "dflash.block_size",
                                   .fill_key = "tokenizer.ggml.mask_token_id",
                                   .import_targets = kDsparkTargets};

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

}  // namespace

const Mapping& Dsv4Mapping() { return kMapping; }
const Mapping& DsparkMapping() { return kDsparkMapping; }

}  // namespace llmp::spec::mappings
