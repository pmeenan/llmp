// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Approved Gemma 3 4B QAT profile, checked binding and bounded state/inputs.
// Media inputs remain outside this text profile.
#ifndef JITLLM_MODEL_GEMMA3_H_
#define JITLLM_MODEL_GEMMA3_H_

#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "model/state.h"

namespace jitllm::artifact {
class Artifact;
}
namespace jitllm::model {

struct Gemma3Profile {
  std::uint32_t layers = 0, width = 0, ffn = 0, heads = 0, kv_heads = 0;
  std::uint32_t key_dim = 0, value_dim = 0, vocab = 0, context = 0, window = 0;
  float rope_base = 0, rope_scale = 0, rms_eps = 0;
  std::string_view rope_scaling;
  bool operator==(const Gemma3Profile&) const = default;
  // d812 gemma3.cpp load_swa_pattern(6): the final layer33 is local.
  bool local(std::uint32_t layer) const { return (layer + 1) % 6 != 0; }
};
const Gemma3Profile& Gemma3_4BQat();

struct Gemma3Resource {
  std::vector<std::string> roles;
  std::string type;
  std::vector<std::uint64_t> ne;
  std::uint64_t readable = 0;
};
struct Gemma3Tensor {
  std::uint32_t index = 0;
  std::string type;
  std::vector<std::uint64_t> ne;
  std::uint64_t readable = 0;
  bool operator==(const Gemma3Tensor&) const = default;
};
struct Gemma3Layer {
  Gemma3Tensor attn_norm, q, k, v, out, q_norm, k_norm, attn_post_norm;
  Gemma3Tensor ffn_norm, gate, up, down, ffn_post_norm;
  bool operator==(const Gemma3Layer&) const = default;
};
struct Gemma3Binding {
  // The approved file has no output.weight; output aliases token_embd.
  Gemma3Tensor token_embd, output, output_norm;
  std::vector<Gemma3Layer> layers;
  bool operator==(const Gemma3Binding&) const = default;
};
// Closed to the approved profile and actual F32/Q4_0/Q8_0 tensor contract.
// This recognizes storage, not executable kernel or importer support.
std::expected<Gemma3Binding, std::string> BindGemma3(const Gemma3Profile& profile,
                                                     std::string_view architecture,
                                                     std::span<const Gemma3Resource> resources);
std::expected<Gemma3Binding, std::string> BindGemma3(const Gemma3Profile& profile,
                                                     const artifact::Artifact& artifact);
// Execution admission additionally authenticates the approved prepared source.
std::expected<Gemma3Binding, std::string> BindApprovedGemma3(const artifact::Artifact& artifact);
// Public descriptors are rechecked before any graph or placement.
std::expected<void, std::string> CheckGemma3Binding(const Gemma3Profile& profile,
                                                    const Gemma3Binding& binding);

inline constexpr std::uint32_t kGemma3Context = 131072;
inline constexpr std::uint32_t kGemma3MaxRows = 8192;
inline constexpr std::uint32_t kGemma3MaxSlots = 16;
struct Gemma3StateTensor {
  std::uint32_t layer = 0;
  bool value = false, local = false;
  std::uint32_t width = 0, cells = 0;  // F16 [head_dim * kv_heads, cells]
  std::uint64_t offset = 0, bytes = 0;
};
struct Gemma3StateLayout {
  std::uint32_t context = 0, max_rows = 0, global_cells = 0, local_cells = 0;
  std::vector<Gemma3StateTensor> tensors;
  std::uint64_t bytes = 0;
  // Append-only local rings cannot advertise arbitrary rollback. A future
  // speculative runner must supply and qualify snapshots of overwritten rows.
  std::expected<std::vector<StateRepresentation>, std::string> Representations(
      const Gemma3Profile& profile) const;
};
// One slot, stable virtual layout; each K/V starts on a 2 MiB boundary.
// Globals append; locals hold pad(min(context, window + max_rows),256)
// cells so every query can still read its window after a complete chunk write.
std::expected<Gemma3StateLayout, std::string> Gemma3State(const Gemma3Profile& profile,
                                                          std::uint32_t context,
                                                          std::uint32_t max_rows);
// Includes padded cells actually read, to be zeroed/materialized before use.
// At zero positions nothing is initialized; locals become a bounded full ring.
std::expected<std::vector<StateRange>, std::string> Gemma3UsedState(const Gemma3Profile& profile,
                                                                    const Gemma3StateLayout& state,
                                                                    std::uint32_t positions,
                                                                    std::uint32_t read_align = 256);

struct Gemma3Segment {
  std::uint32_t slot = 0, n_past = 0;
  std::span<const std::int32_t> tokens;
};
struct Gemma3SegmentInputs {
  std::uint32_t slot = 0, first_row = 0, rows = 0, n_past = 0;
  std::uint32_t global_n_kv = 0, local_n_kv = 0;
  std::vector<std::int64_t> global_cells, local_cells;
  // Optional reference masks, F16 [n_kv, rows], in this slot's own cache.
  std::vector<std::uint16_t> global_mask, local_mask;
};
struct Gemma3ChunkInputs {
  std::vector<std::int32_t> tokens, positions, out_ids;
  std::vector<Gemma3SegmentInputs> segments;
};
// Total rows <= max_rows, unique slot IDs <16. Positions/cache cells restart
// independently at each segment's n_past; flat output IDs identify its rows.
// Default inputs are O(rows+slots), retaining causal/window descriptors for
// a future device mask. Reference masks are opt-in and bounded before growth.
std::expected<Gemma3ChunkInputs, std::string> Gemma3Chunk(const Gemma3Profile& profile,
                                                          const Gemma3StateLayout& state,
                                                          std::span<const Gemma3Segment> segments,
                                                          bool masks = false,
                                                          std::uint32_t read_align = 256);
// Heap buffer bytes, including the segment descriptor array, before growth.
// This is an envelope, not admission: the future caller must fund it first.
// Empty-vector reserve/resize allocates exactly on the pinned libstdc++; the
// builder checks actual capacities against this envelope before returning.
std::expected<std::uint64_t, std::string> Gemma3HostInputBytes(
    const Gemma3Profile& profile, const Gemma3StateLayout& state,
    std::span<const Gemma3Segment> segments, bool masks = false, std::uint32_t read_align = 256);
// The exact physical rows a completed chunk writes, relative to its slot.
// This differs from the initialized read footprint and splits wraparound.
std::expected<std::vector<StateRange>, std::string> Gemma3ChunkWrites(
    const Gemma3Profile& profile, const Gemma3StateLayout& state, std::uint32_t n_past,
    std::uint32_t rows);

}  // namespace jitllm::model
#endif  // JITLLM_MODEL_GEMMA3_H_
