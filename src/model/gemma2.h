// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Approved Gemma 2 2B Q8_0 profile, checked binding and bounded state/inputs.
// No media adapter.
#ifndef LLMP_MODEL_GEMMA2_H_
#define LLMP_MODEL_GEMMA2_H_

#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "model/state.h"

namespace llmp::artifact {
class Artifact;
}
namespace llmp::model {

struct Gemma2Profile {
  std::uint32_t layers = 0, width = 0, ffn = 0, heads = 0, kv_heads = 0;
  std::uint32_t key_dim = 0, value_dim = 0, vocab = 0, context = 0, window = 0;
  float rope_base = 0, rope_scale = 0, rms_eps = 0;
  float attention_scale = 0, attention_softcap = 0, final_softcap = 0;
  bool operator==(const Gemma2Profile&) const = default;
  // d812 gemma2.cpp load_swa_pattern(2): even layers local, odd global.
  bool local(std::uint32_t layer) const { return layer % 2 == 0; }
};
const Gemma2Profile& Gemma2_2B();

struct Gemma2Resource {
  std::vector<std::string> roles;
  std::string type;
  std::vector<std::uint64_t> ne;
  std::uint64_t readable = 0;
};
struct Gemma2Tensor {
  std::uint32_t index = 0;
  std::string type;
  std::vector<std::uint64_t> ne;
  std::uint64_t readable = 0;
  bool operator==(const Gemma2Tensor&) const = default;
};
struct Gemma2Layer {
  Gemma2Tensor attn_norm, q, k, v, out, attn_post_norm;
  Gemma2Tensor ffn_norm, gate, up, down, ffn_post_norm;
  bool operator==(const Gemma2Layer&) const = default;
};
struct Gemma2Binding {
  // The approved file has no output.weight; output aliases token_embd.
  Gemma2Tensor token_embd, output, output_norm;
  std::vector<Gemma2Layer> layers;
  bool operator==(const Gemma2Binding&) const = default;
};
// Bind the approved prepared Q8_0 source identity before model admission.
std::expected<Gemma2Binding, std::string> BindApprovedGemma2(const artifact::Artifact& artifact);
// Closed to the approved profile and actual F32/Q8_0 tensor contract.
// This recognizes storage, not executable kernel or importer support.
std::expected<Gemma2Binding, std::string> BindGemma2(const Gemma2Profile& profile,
                                                     std::string_view architecture,
                                                     std::span<const Gemma2Resource> resources);
std::expected<Gemma2Binding, std::string> BindGemma2(const Gemma2Profile& profile,
                                                     const artifact::Artifact& artifact);
// Public descriptors are rechecked before any graph or placement.
std::expected<void, std::string> CheckGemma2Binding(const Gemma2Profile& profile,
                                                    const Gemma2Binding& binding);

inline constexpr std::uint32_t kGemma2Context = 8192;
inline constexpr std::uint32_t kGemma2MaxRows = 8192;
inline constexpr std::uint32_t kGemma2MaxSlots = 16;
struct Gemma2StateTensor {
  std::uint32_t layer = 0;
  bool value = false, local = false;
  std::uint32_t width = 0, cells = 0;  // F16 [head_dim * kv_heads, cells]
  std::uint64_t offset = 0, bytes = 0;
};
struct Gemma2StateLayout {
  std::uint32_t context = 0, max_rows = 0, global_cells = 0, local_cells = 0;
  std::vector<Gemma2StateTensor> tensors;
  std::uint64_t bytes = 0;
  // Append-only local rings cannot advertise arbitrary rollback. A future
  // speculative runner must supply and qualify snapshots of overwritten rows.
  std::expected<std::vector<StateRepresentation>, std::string> Representations(
      const Gemma2Profile& profile) const;
};
// One slot, stable virtual layout; each K/V starts on a 2 MiB boundary.
// Globals append; locals hold pad(min(context, window + max_rows),256)
// cells so every query can still read its window after a complete chunk write.
std::expected<Gemma2StateLayout, std::string> Gemma2State(const Gemma2Profile& profile,
                                                          std::uint32_t context,
                                                          std::uint32_t max_rows);
// Includes padded cells actually read, to be zeroed/materialized before use.
// At zero positions nothing is initialized; locals become a bounded full ring.
std::expected<std::vector<StateRange>, std::string> Gemma2UsedState(const Gemma2Profile& profile,
                                                                    const Gemma2StateLayout& state,
                                                                    std::uint32_t positions,
                                                                    std::uint32_t read_align = 256);

struct Gemma2Segment {
  std::uint32_t slot = 0, n_past = 0;
  std::span<const std::int32_t> tokens;
};
struct Gemma2SegmentInputs {
  std::uint32_t slot = 0, first_row = 0, rows = 0, n_past = 0;
  std::uint32_t global_n_kv = 0, local_n_kv = 0;
  std::vector<std::int64_t> global_cells, local_cells;
  // Optional reference masks, F16 [n_kv, rows], in this slot's own cache.
  std::vector<std::uint16_t> global_mask, local_mask;
};
struct Gemma2ChunkInputs {
  std::vector<std::int32_t> tokens, positions, out_ids;
  std::vector<Gemma2SegmentInputs> segments;
};
// Per-slot rows <= max_rows, unique slot IDs <16. Positions/cache cells restart
// independently at each segment's n_past; flat output IDs identify its rows.
// Zero max_total_rows retains the original max_rows total bound. An explicit
// larger wave bound never enlarges a slot's query/ring/checkpoint layout.
// Default inputs are O(rows+slots), retaining causal/window descriptors for
// a future device mask. Reference masks are opt-in and bounded before growth.
std::expected<Gemma2ChunkInputs, std::string> Gemma2Chunk(const Gemma2Profile& profile,
                                                          const Gemma2StateLayout& state,
                                                          std::span<const Gemma2Segment> segments,
                                                          bool masks = false,
                                                          std::uint32_t read_align = 256,
                                                          std::uint32_t max_total_rows = 0);
// Heap buffer bytes, including the segment descriptor array, before growth.
// This is an envelope, not admission: the future caller must fund it first.
// Empty-vector reserve/resize allocates exactly on the pinned libstdc++; the
// builder checks actual capacities against this envelope before returning.
std::expected<std::uint64_t, std::string> Gemma2HostInputBytes(
    const Gemma2Profile& profile, const Gemma2StateLayout& state,
    std::span<const Gemma2Segment> segments, bool masks = false, std::uint32_t read_align = 256,
    std::uint32_t max_total_rows = 0);
// The exact physical rows a completed chunk writes, relative to its slot.
// This differs from the initialized read footprint and splits wraparound.
std::expected<std::vector<StateRange>, std::string> Gemma2ChunkWrites(
    const Gemma2Profile& profile, const Gemma2StateLayout& state, std::uint32_t n_past,
    std::uint32_t rows);

}  // namespace llmp::model
#endif  // LLMP_MODEL_GEMMA2_H_
