// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Gemma 4 text-model foundation: approved checkpoint profiles, strict GGML
// artifact binding, bounded independent-slot KV layouts and chunk inputs.
// No graph, kernel, importer or serving runner is supplied here. Semantics
// follow llama.cpp b29c606e's gemma4.cpp and llama-kv-cache-iswa.cpp; actual
// pinned GGUF metadata/tensor contracts are retained in tests/unit/data/gemma4.
#ifndef JITLLM_MODEL_GEMMA4_H_
#define JITLLM_MODEL_GEMMA4_H_

#include <cstdint>
#include <expected>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "model/state.h"

namespace jitllm::artifact {
class Artifact;
}
namespace jitllm::model {

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
};
struct Gemma4Binding {
  Gemma4Tensor token_embd, output, output_norm, rope_freqs;
  std::vector<Gemma4Layer> layers;
};
// Missing/duplicate/unread roles, wrong shapes, integer or malformed matrix
// types, and malformed expert storage are refused. Recognition of a GGML
// representation is not a claim that an execution kernel supports it.
std::expected<Gemma4Binding, std::string> BindGemma4(const Gemma4Profile& profile,
                                                     std::string_view architecture,
                                                     std::span<const Gemma4Resource> resources);
std::expected<Gemma4Binding, std::string> BindGemma4(const Gemma4Profile& profile,
                                                     const artifact::Artifact& artifact);
// Global n_rot is 512, not 128. Preserve the checkpoint's 256 F32 frequency
// factors; their large finite suffix encodes proportional RoPE.
std::expected<void, std::string> CheckGemma4RopeFactors(const Gemma4Profile& profile,
                                                        std::span<const float> factors);

inline constexpr std::uint32_t kGemma4MaxRows = 8192;
inline constexpr std::uint32_t kGemma4MaxSlots = 16;
struct Gemma4StateTensor {
  std::uint32_t layer = 0;
  bool value = false, local = false;
  std::uint32_t width = 0, cells = 0;  // F16 [head_dim * kv_heads, cells]
  std::uint64_t offset = 0, bytes = 0;
};
struct Gemma4StateLayout {
  std::uint32_t context = 0, max_rows = 0, global_cells = 0, local_cells = 0;
  std::vector<Gemma4StateTensor> tensors;
  std::uint64_t bytes = 0;
  // Append-only local rings cannot advertise arbitrary rollback. A future
  // speculative runner must supply and qualify snapshots of overwritten rows.
  std::expected<std::vector<StateRepresentation>, std::string> Representations(
      const Gemma4Profile& profile) const;
};
// One slot, stable virtual layout; each K/V starts on a 2 MiB boundary.
// Globals append; locals hold pad(min(context, window + max_rows),256)
// cells so every query can still read its window after a complete chunk write.
std::expected<Gemma4StateLayout, std::string> Gemma4State(const Gemma4Profile& profile,
                                                          std::uint32_t context,
                                                          std::uint32_t max_rows);
// Includes padded cells actually read, to be zeroed/materialized before use.
// At zero positions nothing is initialized; locals become a bounded full ring.
std::expected<std::vector<StateRange>, std::string> Gemma4UsedState(const Gemma4Profile& profile,
                                                                    const Gemma4StateLayout& state,
                                                                    std::uint32_t positions,
                                                                    std::uint32_t read_align = 256);

struct Gemma4Segment {
  std::uint32_t slot = 0, n_past = 0;
  std::span<const std::int32_t> tokens;
};
struct Gemma4SegmentInputs {
  std::uint32_t slot = 0, first_row = 0, rows = 0, n_past = 0;
  std::uint32_t global_n_kv = 0, local_n_kv = 0;
  std::vector<std::int64_t> global_cells, local_cells;
  // Optional reference masks, F16 [n_kv, rows], in this slot's own cache.
  std::vector<std::uint16_t> global_mask, local_mask;
};
struct Gemma4ChunkInputs {
  std::vector<std::int32_t> tokens, positions, out_ids;
  std::vector<Gemma4SegmentInputs> segments;
};
// Total rows <= max_rows, unique slot IDs <16. Positions/cache cells restart
// independently at each segment's n_past; flat output IDs identify its rows.
// Default inputs are O(rows+slots), retaining causal/window descriptors for
// a future device mask. Reference masks are opt-in and bounded before growth.
std::expected<Gemma4ChunkInputs, std::string> Gemma4Chunk(const Gemma4Profile& profile,
                                                          const Gemma4StateLayout& state,
                                                          std::span<const Gemma4Segment> segments,
                                                          bool masks = false,
                                                          std::uint32_t read_align = 256);
// Heap buffer bytes, including the segment descriptor array, before growth.
// This is an envelope, not admission: the future caller must fund it first.
// Empty-vector reserve/resize allocates exactly on the pinned libstdc++; the
// builder checks actual capacities against this envelope before returning.
std::expected<std::uint64_t, std::string> Gemma4HostInputBytes(
    const Gemma4Profile& profile, const Gemma4StateLayout& state,
    std::span<const Gemma4Segment> segments, bool masks = false, std::uint32_t read_align = 256);
// The exact physical rows a completed chunk writes, relative to its slot.
// This differs from the initialized read footprint and splits wraparound.
std::expected<std::vector<StateRange>, std::string> Gemma4ChunkWrites(
    const Gemma4Profile& profile, const Gemma4StateLayout& state, std::uint32_t n_past,
    std::uint32_t rows);

}  // namespace jitllm::model
#endif  // JITLLM_MODEL_GEMMA4_H_
