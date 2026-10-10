// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Approved Gemma 4 26B-A4B and 31B profiles and the strict GGML artifact
// binding: the approval a prepared artifact passes before admission. The
// state layout, step inputs and graphs come from the components (D-107,
// spec/mappings/gemma4.h). Semantics follow llama.cpp b29c606e's gemma4.cpp
// and llama-kv-cache-iswa.cpp; actual pinned GGUF metadata/tensor contracts
// are retained in tests/unit/data/gemma4.
#ifndef LLMP_MODEL_GEMMA4_H_
#define LLMP_MODEL_GEMMA4_H_

#include <cstdint>
#include <expected>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace llmp::artifact {
class Artifact;
}
namespace llmp::model {

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
  // Both approved checkpoints have five local layers then one global,
  // no inter-layer KV sharing, and no per-layer input embedding.
  bool local(std::uint32_t layer) const { return (layer + 1) % 6 != 0; }
  std::uint32_t head_dim(std::uint32_t layer) const {
    return local(layer) ? local_head_dim : global_head_dim;
  }
  std::uint32_t kv_heads(std::uint32_t layer) const {
    return local(layer) ? local_kv_heads : global_kv_heads;
  }
};
const Gemma4Profile& Gemma4_26BA4B();
const Gemma4Profile& Gemma4_31B();

// Normal resources use their GGML ne; expert arrays use one expert's ne,
// with count separate, as the validated v0 artifact represents them.
struct Gemma4Resource {
  std::vector<std::string> roles;
  std::string type;
  std::vector<std::uint64_t> ne;
  bool expert_array = false;
  std::uint32_t count = 0;
  std::uint64_t group_offset = 0, readable = 0;
};
struct Gemma4Tensor {
  std::uint32_t index = 0;  // ordinary resources or expert arrays, separately
  std::string type;
  std::vector<std::uint64_t> ne;
  bool expert_array = false;
  std::uint64_t group_offset = 0, readable = 0;
  bool operator==(const Gemma4Tensor&) const = default;
};
struct Gemma4Layer {
  Gemma4Tensor attn_norm, q, k, v, out, q_norm, k_norm, attn_post_norm;
  // Globals reuse K's projection as V, before K's learned norm/RoPE.
  // K and V still have separate caches: V uses unweighted RMSNorm.
  bool tied_kv = false;
  Gemma4Tensor ffn_norm, gate, up, down, ffn_post_norm;
  std::optional<Gemma4Tensor> output_scale;
  std::optional<Gemma4Tensor> router, router_scale, ffn_pre_norm_2, ffn_post_norm_1,
      ffn_post_norm_2, gate_up_exps, gate_exps, up_exps, down_exps, expert_scale;
  bool operator==(const Gemma4Layer&) const = default;
};
struct Gemma4Binding {
  Gemma4Tensor token_embd, output, output_norm, rope_freqs;
  std::vector<Gemma4Layer> layers;
  bool operator==(const Gemma4Binding&) const = default;
};
// Missing/duplicate/unread roles, wrong shapes, integer or malformed matrix
// types, and malformed expert storage are refused. Recognition of a GGML
// representation is not a claim that an execution kernel supports it.
std::expected<Gemma4Binding, std::string> BindGemma4(const Gemma4Profile& profile,
                                                     std::string_view architecture,
                                                     std::span<const Gemma4Resource> resources);
std::expected<Gemma4Binding, std::string> BindGemma4(const Gemma4Profile& profile,
                                                     const artifact::Artifact& artifact);
// Checks public/mutable binding descriptors through the same strict role and
// storage contract as artifact admission. Execution builders call this before
// GGML sees any dimensions; tied identities must describe the same storage.
std::expected<void, std::string> CheckGemma4Binding(const Gemma4Profile& profile,
                                                    const Gemma4Binding& binding);
// Global n_rot is 512, not 128. Preserve the checkpoint's 256 F32 frequency
// factors; their large finite suffix encodes proportional RoPE.
std::expected<void, std::string> CheckGemma4RopeFactors(const Gemma4Profile& profile,
                                                        std::span<const float> factors);

inline constexpr std::uint32_t kGemma4Context = 262144;

}  // namespace llmp::model
#endif  // LLMP_MODEL_GEMMA4_H_
