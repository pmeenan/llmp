// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Gemma 2, 3 and 4 and the Gemma 4 assistant as test fixtures: the
// approved checkpoints' facts (their GGUF metadata, checked against the
// pinned files' tensor tables in tests/unit/data/gemma*), the source values
// their architecture mappings read, so that a test can synthesize a real
// family's spec (or a small variant's) from a fixture's tensor table
// without an artifact, and the fixture tables' resource descriptions.
//
// Serving reads the spec a v1 artifact stores (D-118); nothing in src/
// holds these facts. A change here changes only what the tests synthesize.

#ifndef LLMP_TESTS_SUPPORT_FAMILIES_GEMMA_H_
#define LLMP_TESTS_SUPPORT_FAMILIES_GEMMA_H_

#include <cmath>
#include <cstdint>
#include <expected>
#include <string>
#include <string_view>
#include <vector>

#include "spec/mapping.h"
#include "spec/mappings/rules.h"

namespace llmp::test_support::families {

// A fixture table's tensor as its v0 descriptor gave it (a GGML type and
// ne; Gemma 4's expert arrays one expert's ne with their count).
struct Gemma2Resource {
  std::vector<std::string> roles;
  std::string type;
  std::vector<std::uint64_t> ne;
  std::uint64_t readable = 0;
};
using Gemma3Resource = Gemma2Resource;
struct Gemma4Resource {
  std::vector<std::string> roles;
  std::string type;
  std::vector<std::uint64_t> ne;
  bool expert_array = false;
  std::uint32_t count = 0;
  std::uint64_t group_offset = 0, readable = 0;
};

struct Gemma2Profile {
  std::uint32_t layers = 0, width = 0, ffn = 0, heads = 0, kv_heads = 0;
  std::uint32_t key_dim = 0, value_dim = 0, vocab = 0, context = 0, window = 0;
  float rope_base = 0, rope_scale = 0, rms_eps = 0;
  float attention_scale = 0, attention_softcap = 0, final_softcap = 0;
  bool operator==(const Gemma2Profile&) const = default;
  // d812 gemma2.cpp load_swa_pattern(2): even layers local, odd global.
  bool local(std::uint32_t layer) const { return layer % 2 == 0; }
};
inline const Gemma2Profile& Gemma2_2B() {
  // Actual metadata plus d812 defaults: absent RoPE base/scale => 10000/1.
  // The 2B attention scale is 1/sqrt(256), not 1/sqrt(width/heads).
  static constexpr Gemma2Profile profile{26,      2304,    9216,  8,    4,        256,
                                         256,     256000,  8192,  4096, 10000.0F, 1.0F,
                                         1.0e-6F, 0.0625F, 50.0F, 30.0F};
  return profile;
}

struct Gemma3Profile {
  std::uint32_t layers = 0, width = 0, ffn = 0, heads = 0, kv_heads = 0;
  std::uint32_t key_dim = 0, value_dim = 0, vocab = 0, context = 0, window = 0;
  float rope_base = 0, rope_scale = 0, rms_eps = 0;
  std::string_view rope_scaling;
  bool operator==(const Gemma3Profile&) const = default;
  // d812 gemma3.cpp load_swa_pattern(6): the final layer33 is local.
  bool local(std::uint32_t layer) const { return (layer + 1) % 6 != 0; }
};
inline const Gemma3Profile& Gemma3_4BQat() {
  // Facts from bbcac0d0's approved Q4_0 file.
  static constexpr Gemma3Profile profile{
      34, 2560, 10240, 8, 4, 256, 256, 262208, 131072, 1024, 1000000.0F, 8.0F, 1.0e-6F, "linear"};
  return profile;
}

inline constexpr std::uint32_t kGemma4Context = 262144;
struct Gemma4Profile {
  std::string_view name;
  std::uint32_t layers = 0, width = 0, heads = 0;
  std::uint32_t local_kv_heads = 0, global_kv_heads = 0;
  std::uint32_t local_head_dim = 0, global_head_dim = 0;
  std::uint32_t local_rope_dims = 0, global_rope_dims = 0;
  std::uint32_t ffn = 0, experts = 0, experts_used = 0, expert_ffn = 0;
  std::uint32_t vocab = 0, context = 0, window = 0;
  float local_rope_base = 0, global_rope_base = 0, rms_eps = 0, final_softcap = 0;
  bool operator==(const Gemma4Profile&) const = default;
  // Both approved checkpoints have five local layers then one global, no
  // inter-layer KV sharing, and no per-layer input embedding.
  bool local(std::uint32_t layer) const { return (layer + 1) % 6 != 0; }
  std::uint32_t head_dim(std::uint32_t layer) const {
    return local(layer) ? local_head_dim : global_head_dim;
  }
  std::uint32_t kv_heads(std::uint32_t layer) const {
    return local(layer) ? local_kv_heads : global_kv_heads;
  }
};
inline const Gemma4Profile& Gemma4_26BA4B() {
  static constexpr Gemma4Profile p{.name = "gemma-4-26B-A4B-it",
                                   .layers = 30,
                                   .width = 2816,
                                   .heads = 16,
                                   .local_kv_heads = 8,
                                   .global_kv_heads = 2,
                                   .local_head_dim = 256,
                                   .global_head_dim = 512,
                                   .local_rope_dims = 256,
                                   .global_rope_dims = 512,
                                   .ffn = 2112,
                                   .experts = 128,
                                   .experts_used = 8,
                                   .expert_ffn = 704,
                                   .vocab = 262144,
                                   .context = kGemma4Context,
                                   .window = 1024,
                                   .local_rope_base = 10000,
                                   .global_rope_base = 1000000,
                                   .rms_eps = 1e-6F,
                                   .final_softcap = 30};
  return p;
}
inline const Gemma4Profile& Gemma4_31B() {
  static constexpr Gemma4Profile p{.name = "gemma-4-31B-it",
                                   .layers = 60,
                                   .width = 5376,
                                   .heads = 32,
                                   .local_kv_heads = 16,
                                   .global_kv_heads = 4,
                                   .local_head_dim = 256,
                                   .global_head_dim = 512,
                                   .local_rope_dims = 256,
                                   .global_rope_dims = 512,
                                   .ffn = 21504,
                                   .experts = 0,
                                   .experts_used = 0,
                                   .expert_ffn = 0,
                                   .vocab = 262144,
                                   .context = kGemma4Context,
                                   .window = 1024,
                                   .local_rope_base = 10000,
                                   .global_rope_base = 1000000,
                                   .rms_eps = 1e-6F,
                                   .final_softcap = 30};
  return p;
}

struct Gemma4AssistantProfile {
  std::string_view name;
  std::uint32_t target_width = 0, heads = 0, local_kv_heads = 0, global_kv_heads = 0;
  std::uint32_t local_target_layer = 0, global_target_layer = 0;
  bool operator==(const Gemma4AssistantProfile&) const = default;
};
inline constexpr std::uint32_t kGemma4AssistantWidth = 1024;
inline constexpr std::uint32_t kGemma4AssistantFfn = 8192;
inline constexpr std::uint32_t kGemma4AssistantLayers = 4;
inline const Gemma4AssistantProfile& Gemma4Assistant26() {
  static constexpr Gemma4AssistantProfile p{
      "gemma-4-26B-A4B-it-assistant-q8_0", 2816, 16, 8, 2, 28, 29};
  return p;
}
inline const Gemma4AssistantProfile& Gemma4Assistant31() {
  static constexpr Gemma4AssistantProfile p{
      "gemma-4-31B-it-assistant-q8_0", 5376, 32, 16, 4, 58, 59};
  return p;
}

namespace gemma_detail {
inline std::vector<std::int64_t> PerLayerOf(std::uint32_t layers, auto value) {
  std::vector<std::int64_t> out;
  for (std::uint32_t il = 0; il < layers; ++il) out.push_back(static_cast<std::int64_t>(value(il)));
  return out;
}
}  // namespace gemma_detail

// The profile as the GGUF keys the mapping reads. Refused where the profile
// says what the mapping cannot (an attention scale other than 1 / sqrt(key
// length), a rotation scaling, other local layers).
inline std::expected<std::vector<spec::SourceValue>, std::string> Gemma2Values(
    const Gemma2Profile& p) {
  using spec::mappings::Value;
  if (p.key_dim == 0 ||
      p.attention_scale != static_cast<float>(1.0 / std::sqrt(static_cast<double>(p.key_dim))) ||
      p.rope_scale != 1.0F)
    return std::unexpected(
        "the Gemma 2 profile's attention scale or rotation scaling is not the mapping's");
  for (std::uint32_t il = 0; il < p.layers; ++il)
    if (p.local(il) != (il % 2 == 0))
      return std::unexpected("the Gemma 2 profile's local layers are not the mapping's");
  return std::vector<spec::SourceValue>{
      Value("gemma2.context_length", p.context),
      Value("gemma2.embedding_length", p.width),
      Value("gemma2.block_count", p.layers),
      Value("gemma2.feed_forward_length", p.ffn),
      Value("gemma2.attention.head_count", p.heads),
      Value("gemma2.attention.head_count_kv", p.kv_heads),
      Value("gemma2.attention.key_length", p.key_dim),
      Value("gemma2.attention.value_length", p.value_dim),
      Value("gemma2.attention.layer_norm_rms_epsilon", p.rms_eps),
      Value("gemma2.attn_logit_softcapping", p.attention_softcap),
      Value("gemma2.final_logit_softcapping", p.final_softcap),
      Value("gemma2.attention.sliding_window", p.window),
      Value("gemma2.rope.freq_base", p.rope_base),
  };
}

// Refused for a scaling other than linear, or other local layers.
inline std::expected<std::vector<spec::SourceValue>, std::string> Gemma3Values(
    const Gemma3Profile& p) {
  using spec::mappings::Value;
  if (p.rope_scaling != "linear")
    return std::unexpected("the Gemma 3 profile's rotation scaling is not the mapping's");
  for (std::uint32_t il = 0; il < p.layers; ++il)
    if (p.local(il) != ((il + 1) % 6 != 0))
      return std::unexpected("the Gemma 3 profile's local layers are not the mapping's");
  return std::vector<spec::SourceValue>{
      Value("gemma3.context_length", p.context),
      Value("gemma3.embedding_length", p.width),
      Value("gemma3.block_count", p.layers),
      Value("gemma3.feed_forward_length", p.ffn),
      Value("gemma3.attention.head_count", p.heads),
      Value("gemma3.attention.head_count_kv", p.kv_heads),
      Value("gemma3.attention.key_length", p.key_dim),
      Value("gemma3.attention.value_length", p.value_dim),
      Value("gemma3.attention.layer_norm_rms_epsilon", p.rms_eps),
      Value("gemma3.attention.sliding_window", p.window),
      Value("gemma3.rope.freq_base", p.rope_base),
      Value("gemma3.rope.scaling.type", p.rope_scaling),
      Value("gemma3.rope.scaling.factor", p.rope_scale),
  };
}

// The per-layer lists as lists. Refused for other local layers.
inline std::expected<std::vector<spec::SourceValue>, std::string> Gemma4Values(
    const Gemma4Profile& p) {
  using gemma_detail::PerLayerOf;
  using spec::mappings::Value;
  for (std::uint32_t il = 0; il < p.layers; ++il)
    if (p.local(il) != ((il + 1) % 6 != 0))
      return std::unexpected("the Gemma 4 profile's local layers are not the mapping's");
  std::vector<spec::SourceValue> out{
      Value("gemma4.context_length", p.context),
      Value("gemma4.embedding_length", p.width),
      Value("gemma4.block_count", p.layers),
      Value("gemma4.feed_forward_length", p.ffn),
      Value("gemma4.attention.head_count", p.heads),
      Value("gemma4.attention.head_count_kv",
            PerLayerOf(p.layers, [&](std::uint32_t il) { return p.kv_heads(il); })),
      Value("gemma4.attention.key_length", p.global_head_dim),
      Value("gemma4.attention.key_length_swa", p.local_head_dim),
      Value("gemma4.attention.value_length", p.global_head_dim),
      Value("gemma4.attention.value_length_swa", p.local_head_dim),
      Value("gemma4.attention.layer_norm_rms_epsilon", p.rms_eps),
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
    out.push_back(Value("gemma4.expert_count", p.experts));
    out.push_back(Value("gemma4.expert_used_count", p.experts_used));
    out.push_back(Value("gemma4.expert_feed_forward_length", p.expert_ffn));
  }
  return out;
}

// The assistant's, with its target's, as its mapping's keys. Refused where
// its target layers are not its target's last local and global layers.
inline std::expected<std::vector<spec::SourceValue>, std::string> Gemma4AssistantValues(
    const Gemma4AssistantProfile& a, const Gemma4Profile& t) {
  using gemma_detail::PerLayerOf;
  using spec::mappings::Value;
  std::uint32_t last_local = 0;
  std::uint32_t last_global = 0;
  for (std::uint32_t il = 0; il < t.layers; ++il) (t.local(il) ? last_local : last_global) = il;
  if (a.target_width != t.width || a.local_target_layer != last_local ||
      a.global_target_layer != last_global || a.local_kv_heads != t.local_kv_heads ||
      a.global_kv_heads != t.global_kv_heads)
    return std::unexpected("the assistant's profile is not its target's");
  const std::string kA = "gemma4-assistant.";
  const auto layers = kGemma4AssistantLayers;
  const auto local = [](std::uint32_t il) { return il % 4 != 3; };  // "lllg"
  return std::vector<spec::SourceValue>{
      Value(kA + "context_length", t.context),
      Value(kA + "embedding_length", kGemma4AssistantWidth),
      Value(kA + "block_count", layers),
      Value(kA + "feed_forward_length", kGemma4AssistantFfn),
      Value(kA + "attention.head_count", a.heads),
      Value(kA + "attention.head_count_kv", PerLayerOf(layers,
                                                       [&](std::uint32_t il) {
                                                         return local(il) ? a.local_kv_heads
                                                                          : a.global_kv_heads;
                                                       })),
      Value(kA + "attention.key_length", t.global_head_dim),
      Value(kA + "attention.key_length_swa", t.local_head_dim),
      Value(kA + "attention.value_length", t.global_head_dim),
      Value(kA + "attention.value_length_swa", t.local_head_dim),
      Value(kA + "attention.layer_norm_rms_epsilon", t.rms_eps),
      Value(kA + "attention.sliding_window", t.window),
      Value(kA + "attention.sliding_window_pattern",
            PerLayerOf(layers, [&](std::uint32_t il) { return local(il) ? 1 : 0; })),
      Value(kA + "rope.freq_base", t.global_rope_base),
      Value(kA + "rope.freq_base_swa", t.local_rope_base),
      Value(kA + "rope.dimension_count", t.global_rope_dims),
      Value(kA + "rope.dimension_count_swa", t.local_rope_dims),
      Value(kA + "embedding_length_out", a.target_width),
  };
}

}  // namespace llmp::test_support::families

#endif  // LLMP_TESTS_SUPPORT_FAMILIES_GEMMA_H_
