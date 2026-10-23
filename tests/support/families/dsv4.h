// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// DeepSeek V4 Flash and its DSpark drafter as test fixtures: the approved
// checkpoints' facts (their GGUFs' deepseek4.* and dflash.* key/values,
// checked by docs/experiments/dsv4-native/gguf_profile_check.py), the
// source values their architecture mappings read, so that a test can
// synthesize a real family's spec (or a small variant's) from a fixture's
// tensor table without an artifact, the fixture tables' resource
// description, and the M3 runners' state layout in closed form (each
// layer's window cache, compressed caches and compressor rings; the
// drafter's rings), which the generic layouts derived from the specs'
// state contracts are checked against.
//
// Serving reads the spec a v1 artifact stores (D-118); nothing in src/
// holds these facts but the DSpark import's target data
// (spec/mappings/dsv4.cc), which must stay equal to Dsv4Flash(). A change
// here changes only what the tests synthesize.

#ifndef LLMP_TESTS_SUPPORT_FAMILIES_DSV4_H_
#define LLMP_TESTS_SUPPORT_FAMILIES_DSV4_H_

#include <algorithm>
#include <cstdint>
#include <expected>
#include <format>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "spec/mapping.h"
#include "spec/mappings/rules.h"

namespace llmp::test_support::families {

inline constexpr std::uint32_t kDsv4CsaRatio = 4;
inline constexpr std::uint32_t kDsv4HcaRatio = 128;
// The supported Flash checkpoint's trained position ceiling.
inline constexpr std::uint32_t kDsv4FlashContext = 1048576;
// The DSpark drafter's ring: the window and a block, padded to attention's
// 256-cell steps (the dflash mapping's).
inline constexpr std::uint32_t kDsparkRing = 256;

// A DeepSeek V4 checkpoint's hyperparameters (the deepseek4.* key/values).
struct Dsv4Profile {
  std::string_view name;
  std::uint32_t layers = 0;
  std::uint32_t width = 0;      // n_embd
  std::uint32_t heads = 0;      // query heads; one KV head (K = V)
  std::uint32_t head_dim = 0;   // key and value length
  std::uint32_t rope_dims = 0;  // the rotated tail of each head
  std::uint32_t q_lora = 0;     // q_lora_rank
  std::uint32_t o_lora = 0;     // output_lora_rank
  std::uint32_t o_groups = 0;   // output_group_count
  std::uint32_t window = 0;     // sliding_window
  std::uint32_t experts = 0;
  std::uint32_t experts_used = 0;
  std::uint32_t expert_ffn = 0;  // per routed and shared expert
  std::uint32_t shared_experts = 0;
  std::uint32_t hash_layers = 0;  // the first layers route by token id
  std::uint32_t hc = 0;           // hyper-connection streams
  std::uint32_t sinkhorn_iterations = 0;
  std::uint32_t indexer_heads = 0;
  std::uint32_t indexer_head_dim = 0;
  std::uint32_t indexer_top_k = 0;
  std::uint32_t vocab = 0;
  std::uint32_t yarn_original_context = 0;
  float rms_eps = 0.0F;
  float hc_eps = 0.0F;
  float rope_base = 0.0F;           // uncompressed layers
  float compress_rope_base = 0.0F;  // compressed layers, the indexer and compressed rows
  float rope_scale = 0.0F;          // YaRN factor (freq_scale = 1 / factor)
  float yarn_beta_fast = 0.0F;
  float yarn_beta_slow = 0.0F;
  float expert_weights_scale = 0.0F;
  bool expert_weights_norm = false;
  float swiglu_limit = 0.0F;         // routed experts (every layer)
  float swiglu_limit_shared = 0.0F;  // shared expert (every layer)
  // Per layer: 0 (sliding window only), 4 (CSA, with the indexer) or 128
  // (HCA).
  std::vector<std::uint32_t> compress_ratios;

  std::uint32_t hc_width() const { return hc * width; }
  std::uint32_t hc_mix() const { return (2 + hc) * hc; }
};

// DeepSeek V4 Flash (both the e3aa0d6a and 0731 GGUF revisions share it).
// The epsilons are the GGUF's float32 values (9.999999974752427e-07).
inline const Dsv4Profile& Dsv4Flash() {
  static const Dsv4Profile kProfile = [] {
    Dsv4Profile p;
    p.name = "deepseek-v4-flash";
    p.layers = 43;
    p.width = 4096;
    p.heads = 64;
    p.head_dim = 512;
    p.rope_dims = 64;
    p.q_lora = 1024;
    p.o_lora = 1024;
    p.o_groups = 8;
    p.window = 128;
    p.experts = 256;
    p.experts_used = 6;
    p.expert_ffn = 2048;
    p.shared_experts = 1;
    p.hash_layers = 3;
    p.hc = 4;
    p.sinkhorn_iterations = 20;
    p.indexer_heads = 64;
    p.indexer_head_dim = 128;
    p.indexer_top_k = 512;
    p.vocab = 129280;
    p.yarn_original_context = 65536;
    p.rms_eps = 1e-6F;
    p.hc_eps = 1e-6F;
    p.rope_base = 10000.0F;
    p.compress_rope_base = 160000.0F;
    p.rope_scale = 16.0F;
    p.yarn_beta_fast = 32.0F;
    p.yarn_beta_slow = 1.0F;
    p.expert_weights_scale = 1.5F;
    p.expert_weights_norm = true;
    p.swiglu_limit = 10.0F;
    p.swiglu_limit_shared = 10.0F;
    // Layers 0 and 1 are window-only; then CSA and HCA alternate from layer
    // 2, and the last layer is CSA.
    p.compress_ratios.assign(p.layers, 0);
    for (std::uint32_t il = 2; il < p.layers; ++il) {
      p.compress_ratios[il] = il % 2 == 0 ? kDsv4CsaRatio : kDsv4HcaRatio;
    }
    return p;
  }();
  return kProfile;
}

// The DSpark drafter: its DeepSeek V4 stages (window-only, every layer
// routed, the rest as the GGUF's dflash.* keys), the most drafts, the
// target layers it taps, its mask token, its Markov head's rank and its
// KV ring's cells.
struct DsparkProfile {
  std::string_view name;
  Dsv4Profile blocks;
  std::uint32_t block_size = 0;              // dflash.block_size: the most drafts
  std::vector<std::uint32_t> target_layers;  // dflash.target_layers
  std::int32_t mask_token = 0;               // tokenizer.ggml.mask_token_id
  std::uint32_t markov_rank = 0;             // markov_w1's rows
  std::uint32_t ring = 0;                    // the KV ring's cells
};

// DeepSeek-V4-Flash-0731's DSpark drafter (dspark-DeepSeek-V4-Flash-0731-
// Q8_0.gguf): the stages' hyperparameters equal the target's but for three
// blocks, all window-only and routed.
inline const DsparkProfile& DsparkDeepSeekV4Flash() {
  static const DsparkProfile kProfile = [] {
    DsparkProfile d;
    d.name = "deepseek-v4-flash-dspark";
    d.blocks = Dsv4Flash();
    d.blocks.name = "deepseek-v4-flash-dspark-blocks";
    d.blocks.layers = 3;
    d.blocks.hash_layers = 0;
    d.blocks.compress_ratios.assign(3, 0);
    d.block_size = 5;
    d.target_layers = {41, 42, 43};
    d.mask_token = 128799;
    d.markov_rank = 256;
    d.ring = kDsparkRing;
    return d;
  }();
  return kProfile;
}

// A fixture table's resource or expert array: its roles (an expert array's
// name), GGML type and ne (an expert array's: one slice's).
struct Dsv4Resource {
  std::vector<std::string> roles;
  std::string type;
  std::vector<std::uint64_t> ne;
  bool expert_array = false;
  std::uint32_t count = 0;  // an expert array's experts
};

// ---------------------------------------------------------------- values

namespace dsv4_detail {

inline spec::SourceValue Reals(std::string_view key, std::vector<double> v) {
  return {.key = std::string(key), .value = spec::ParamValue(std::move(v))};
}

// The keys both mappings' blocks read, under `prefix`.
inline void BlockValues(std::vector<spec::SourceValue>& out, const std::string& prefix,
                        const Dsv4Profile& q) {
  using spec::mappings::FlagValue;
  using spec::mappings::Value;
  const auto key = [&](std::string_view k) { return prefix + std::string(k); };
  std::vector<std::int64_t> ratios(q.compress_ratios.begin(), q.compress_ratios.end());
  for (spec::SourceValue v : std::vector<spec::SourceValue>{
           Value(key("block_count"), q.layers),
           Value(key("context_length"), kDsv4FlashContext),
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

}  // namespace dsv4_detail

// The profile as the deepseek4 mapping's keys. Refused where the profile
// is not the mapping's form.
inline std::expected<std::vector<spec::SourceValue>, std::string> Dsv4Values(const Dsv4Profile& p) {
  if (p.compress_ratios.size() != p.layers || p.shared_experts != 1 || p.hc < 2) {
    return std::unexpected("the DeepSeek V4 profile is not the mapping's form");
  }
  std::vector<spec::SourceValue> out;
  dsv4_detail::BlockValues(out, "deepseek4.", p);
  return out;
}

// The drafter's as the dflash mapping's keys. Refused where its profile is
// not the mapping's form (window-only routed stages, three target layers,
// the mapping's ring).
inline std::expected<std::vector<spec::SourceValue>, std::string> DsparkValues(
    const DsparkProfile& d) {
  using spec::mappings::Value;
  const Dsv4Profile& p = d.blocks;
  if (p.hash_layers != 0 || d.ring != kDsparkRing || d.target_layers.size() != 3 ||
      std::ranges::any_of(p.compress_ratios, [](std::uint32_t r) { return r != 0; })) {
    return std::unexpected("the DSpark profile is not the mapping's form");
  }
  std::vector<spec::SourceValue> out;
  dsv4_detail::BlockValues(out, "dflash.", p);
  out.push_back(Value("dflash.block_size", d.block_size));
  out.push_back(spec::SourceValue{"tokenizer.ggml.mask_token_id", std::int64_t{d.mask_token}});
  out.push_back(Value("dflash.target_layers",
                      std::vector<std::int64_t>(d.target_layers.begin(), d.target_layers.end())));
  return out;
}

// ---------------------------------------------------------------- state

// The M3 runner's state of one sequence in closed form, for a context of
// `context` positions and chunks of at most `max_rows` rows: every layer's
// tensors in one region, each at a 256-byte aligned offset. kFull: the
// window cache a cell per position (pad(context, 256) cells); kRing: a ring
// of pad(window + max_rows, 256) cells (at most the full size). The
// compressed caches hold pad(max(1, ceil(context / ratio)), 256) rows; the
// compressors' rings each layer's last 2·ratio (CSA, indexer) or ratio
// (HCA) positions' projections. Each compressed layer's compressed cache
// follows its window cache directly.
enum class Dsv4Window : std::uint8_t { kFull, kRing };
struct Dsv4StateTensor {
  enum class Kind : std::uint8_t {
    kRawK,        // F16 [head_dim, raw_cells]
    kCsaK,        // F16 [head_dim, csa_cells]
    kCsaStateKv,  // F32 [2·head_dim, 2·4]
    kCsaStateScore,
    kLidK,        // F16 [indexer_head_dim, csa_cells]
    kLidStateKv,  // F32 [2·indexer_head_dim, 2·4]
    kLidStateScore,
    kHcaK,        // F16 [head_dim, hca_cells]
    kHcaStateKv,  // F32 [head_dim, 128]
    kHcaStateScore,
  };
  Kind kind = Kind::kRawK;
  std::uint32_t layer = 0;
  bool f16 = true;
  std::uint64_t ne0 = 0;
  std::uint64_t ne1 = 0;
  std::uint64_t offset = 0;  // in the state region
  std::uint64_t bytes = 0;

  std::uint64_t row_bytes() const { return ne0 * (f16 ? 2U : 4U); }
};

struct Dsv4StateLayout {
  std::uint32_t context = 0;
  std::uint32_t max_rows = 0;
  Dsv4Window window = Dsv4Window::kFull;
  std::uint32_t raw_cells = 0;
  std::uint32_t csa_cells = 0;  // also the indexer's
  std::uint32_t hca_cells = 0;
  std::uint32_t csa_state_rows = 0;  // 2 x 4
  std::uint32_t hca_state_rows = 0;  // 128
  std::vector<Dsv4StateTensor> tensors;
  std::uint64_t bytes = 0;  // the region

  // Layer `layer`'s tensor of `kind`, or null.
  const Dsv4StateTensor* Find(std::uint32_t layer, Dsv4StateTensor::Kind kind) const {
    const auto at = std::ranges::find_if(
        tensors, [&](const Dsv4StateTensor& t) { return t.layer == layer && t.kind == kind; });
    return at == tensors.end() ? nullptr : &*at;
  }
};

namespace dsv4_detail {
inline std::uint64_t Pad(std::uint64_t n, std::uint64_t to) { return (n + to - 1) / to * to; }
}  // namespace dsv4_detail

// Refused if the context or chunk bound is zero or past the Flash context,
// or a chunk leaves the window no room in the ring.
inline std::expected<Dsv4StateLayout, std::string> Dsv4State(
    const Dsv4Profile& p, std::uint32_t context, std::uint32_t max_rows,
    Dsv4Window window = Dsv4Window::kFull) {
  using dsv4_detail::Pad;
  using K = Dsv4StateTensor::Kind;
  if (context == 0 || context > kDsv4FlashContext || max_rows == 0 || max_rows > context ||
      p.compress_ratios.size() != p.layers) {
    return std::unexpected(
        std::format("no state for {} positions in chunks of {}", context, max_rows));
  }
  Dsv4StateLayout s;
  s.context = context;
  s.max_rows = max_rows;
  s.window = window;
  s.raw_cells = static_cast<std::uint32_t>(Pad(context, 256));
  if (window == Dsv4Window::kRing) {
    s.raw_cells = static_cast<std::uint32_t>(
        std::min<std::uint64_t>(s.raw_cells, Pad(std::uint64_t{p.window} + max_rows, 256)));
  }
  if (max_rows > s.raw_cells - p.window) {
    return std::unexpected(std::format("chunks of {} rows leave the window no room in {} cells",
                                       max_rows, s.raw_cells));
  }
  const auto comp_cells = [&](std::uint32_t ratio) {
    return static_cast<std::uint32_t>(
        Pad(std::max<std::uint64_t>(1, (std::uint64_t{context} + ratio - 1) / ratio), 256));
  };
  s.csa_cells = comp_cells(kDsv4CsaRatio);
  s.hca_cells = comp_cells(kDsv4HcaRatio);
  s.csa_state_rows = 2 * kDsv4CsaRatio;
  s.hca_state_rows = kDsv4HcaRatio;
  const auto add = [&](K kind, std::uint32_t layer, bool f16, std::uint64_t ne0,
                       std::uint64_t ne1) {
    const std::uint64_t bytes = ne0 * ne1 * (f16 ? 2 : 4);
    s.tensors.push_back({.kind = kind,
                         .layer = layer,
                         .f16 = f16,
                         .ne0 = ne0,
                         .ne1 = ne1,
                         .offset = s.bytes,
                         .bytes = bytes});
    s.bytes = Pad(s.bytes + bytes, 256);
  };
  for (std::uint32_t il = 0; il < p.layers; ++il) {
    add(K::kRawK, il, true, p.head_dim, s.raw_cells);
    if (p.compress_ratios[il] == kDsv4CsaRatio) {
      add(K::kCsaK, il, true, p.head_dim, s.csa_cells);
      add(K::kCsaStateKv, il, false, 2ULL * p.head_dim, s.csa_state_rows);
      add(K::kCsaStateScore, il, false, 2ULL * p.head_dim, s.csa_state_rows);
      add(K::kLidK, il, true, p.indexer_head_dim, s.csa_cells);
      add(K::kLidStateKv, il, false, 2ULL * p.indexer_head_dim, s.csa_state_rows);
      add(K::kLidStateScore, il, false, 2ULL * p.indexer_head_dim, s.csa_state_rows);
    } else if (p.compress_ratios[il] == kDsv4HcaRatio) {
      add(K::kHcaK, il, true, p.head_dim, s.hca_cells);
      add(K::kHcaStateKv, il, false, p.head_dim, s.hca_state_rows);
      add(K::kHcaStateScore, il, false, p.head_dim, s.hca_state_rows);
    }
  }
  return s;
}

// The DSpark drafter's state in closed form: one F16 [head_dim, ring] ring
// per block, each at a 256-byte aligned offset.
struct DsparkStateLayout {
  std::uint32_t ring = 0;
  std::uint32_t max_rows = 0;
  std::vector<std::uint64_t> offsets;
  std::uint64_t bytes = 0;
};

// Refused if the ring cannot hold the window and a block beside it, in
// 256-cell steps, or `max_rows` is 0.
inline std::expected<DsparkStateLayout, std::string> DsparkState(const DsparkProfile& profile,
                                                                 std::uint32_t max_rows) {
  const Dsv4Profile& p = profile.blocks;
  if (max_rows == 0 || profile.ring == 0 || profile.ring % 256 != 0 ||
      std::uint64_t{profile.ring} < std::uint64_t{p.window} + profile.block_size) {
    return std::unexpected("the drafter's ring must hold its window and a block");
  }
  DsparkStateLayout s;
  s.ring = profile.ring;
  s.max_rows = max_rows;
  const std::uint64_t ring_bytes = std::uint64_t{p.head_dim} * profile.ring * 2;
  for (std::uint32_t il = 0; il < p.layers; ++il) {
    s.offsets.push_back(s.bytes);
    s.bytes = dsv4_detail::Pad(s.bytes + ring_bytes, 256);
  }
  return s;
}

}  // namespace llmp::test_support::families

#endif  // LLMP_TESTS_SUPPORT_FAMILIES_DSV4_H_
