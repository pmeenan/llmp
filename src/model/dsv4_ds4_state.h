// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Private complete-ds4 benchmark state. This representation is distinct
// from the production Dsv4StateLayout: original F32 raw rings contain QAT
// values rounded through F16, compressed primaries are FP8/FP4, and the
// original compressor frontiers retain F32. No production spill identity,
// speculative rollback or graph compatibility is implied by this layout.
#ifndef JITLLM_MODEL_DSV4_DS4_STATE_H_
#define JITLLM_MODEL_DSV4_DS4_STATE_H_

#include <cstdint>
#include <expected>
#include <string>
#include <vector>

#include "model/dsv4.h"
#include "model/state.h"

namespace jitllm::model {

enum class Ds4BaselineStateKind : std::uint8_t {
  kRaw,
  kAttentionKv,
  kAttentionScore,
  kIndexerKv,
  kIndexerScore,
  kAttentionCodes,
  kAttentionScales,
  kAttentionF32,
  kIndexerCodes,
  kIndexerScales,
  kIndexerF32,
  kLayerScalar,
  kDecodeScalar,
  kDecodeTable,
};

struct Ds4BaselineStateTensor {
  Ds4BaselineStateKind kind = Ds4BaselineStateKind::kRaw;
  std::uint32_t layer = 0;
  std::uint32_t ratio = 0;  // zero: fixed state; otherwise used rows=positions/ratio
  std::uint32_t capacity = 0;
  std::uint64_t row_bytes = 0;
  std::uint64_t offset = 0;
  std::uint64_t bytes = 0;
};

struct Ds4BaselineStateLayout {
  std::uint32_t context = 0;
  std::uint32_t max_rows = 0;
  std::uint32_t raw_cells = 0;
  bool packed_kv = true;
  bool packed_indexer = true;
  std::uint64_t granularity = 0;
  std::uint64_t fixed_bytes = 0;
  std::uint64_t virtual_bytes = 0;
  std::vector<Ds4BaselineStateTensor> tensors;
};

// Same default capacities as the pinned original: raw=min(context,
// pad256(min(context,min(128,context)+max_rows))), capped at8192; each
// compressed capacity=floor(context/ratio)+2. Defaults retain packed
// primaries; explicit F32-primary controls still require original QAT.
// Separate cache planes start on provider-granularity boundaries so a
// growable plane cannot share backing with another plane's unused tail.
// No F32 fallback copy is silently allocated for a packed primary;
// dequantized attention/indexer views are accounted planned scratch.
std::expected<Ds4BaselineStateLayout, std::string> LayoutDs4BaselineState(
    const Dsv4Profile& profile, std::uint32_t context, std::uint32_t max_rows,
    std::uint64_t granularity, bool packed_kv = true, bool packed_indexer = true);

// Logical byte ranges needed through positions. Fixed rings/frontiers and
// scalar/table resources are resident from setup. Compressed planes grow
// only with complete emitted groups. The caller rounds these ranges to
// provider backing and charges/maps them before queuing writes; logical
// capacity and committed backing are never interchangeable.
std::expected<std::vector<StateRange>, std::string> Ds4BaselineStateThrough(
    const Ds4BaselineStateLayout& layout, std::uint32_t positions);

}  // namespace jitllm::model

#endif  // JITLLM_MODEL_DSV4_DS4_STATE_H_
