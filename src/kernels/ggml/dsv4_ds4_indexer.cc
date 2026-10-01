// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "kernels/ggml/dsv4_ds4_indexer.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>

namespace jitllm::kernels::ggml {
namespace {

constexpr std::uint32_t kMaxTokens = 4096;
constexpr std::uint32_t kMaxCells = 262144;
constexpr std::uint32_t kHeads = 64;
constexpr std::uint32_t kWidth = 128;
constexpr std::uint32_t kMergeGroup = 8;
constexpr std::uint32_t kChunkCells = 4096;
constexpr std::uint64_t kScratchAlignment = 256;

std::unexpected<KernelFailure> Rejected(std::string detail) {
  return std::unexpected(
      KernelFailure{.error = KernelError::kRejected, .detail = std::move(detail)});
}

bool Absent(const Ds4CacheBuffer& buffer) { return buffer.address == 0 && buffer.bytes == 0; }

bool Holds(const Ds4CacheBuffer& buffer, std::uint64_t bytes, std::uint64_t alignment) {
  return buffer.address != 0 && buffer.address % alignment == 0 && buffer.bytes >= bytes &&
         buffer.bytes <= std::numeric_limits<std::uint64_t>::max() - buffer.address;
}

bool Overlap(const Ds4CacheBuffer& a, std::uint64_t a_bytes, const Ds4CacheBuffer& b,
             std::uint64_t b_bytes) {
  return a.address < b.address + b_bytes && b.address < a.address + a_bytes;
}

std::uint64_t Align(std::uint64_t bytes) {
  return (bytes + kScratchAlignment - 1) & ~(kScratchAlignment - 1);
}

bool Dimensions(std::uint32_t tokens, std::uint32_t cells, std::uint32_t band) {
  return tokens != 0 && tokens <= kMaxTokens && cells != 0 && cells <= kMaxCells && band >= cells &&
         band <= kMaxCells;
}

bool Mxf4Shape(const Ds4IndexerScores& desc) {
  return !desc.quality_mode && desc.causal && desc.ratio == 4 && desc.tokens >= 32 &&
         desc.cells >= 1024 && !Absent(desc.key_codes) && !Absent(desc.key_scales) &&
         Absent(desc.layer_scalars) && desc.score_band == desc.cells &&
         (Absent(desc.bank_ids) ||
          desc.consecutive_bank != std::numeric_limits<std::uint32_t>::max());
}

struct Access {
  Ds4CacheBuffer buffer;
  std::uint64_t bytes = 0;
  bool writable = false;
};

template <std::size_t N>
bool DisjointWrites(const std::array<Access, N>& accesses) {
  for (const auto& access : accesses) {
    if (!Absent(access.buffer) && !Holds(access.buffer, access.buffer.bytes, 1)) return false;
  }
  for (std::size_t i = 0; i < accesses.size(); ++i) {
    if (accesses[i].bytes == 0) continue;
    for (std::size_t j = i + 1; j < accesses.size(); ++j) {
      if (accesses[j].bytes != 0 && (accesses[i].writable || accesses[j].writable) &&
          Overlap(accesses[i].buffer, accesses[i].bytes, accesses[j].buffer, accesses[j].bytes)) {
        return false;
      }
    }
  }
  return true;
}

}  // namespace

std::expected<std::uint64_t, KernelFailure> Ds4IndexerScoreScratchBytes(
    const Ds4IndexerScores& desc) {
  if (!Dimensions(desc.tokens, desc.cells, desc.score_band))
    return Rejected("ds4 indexer score geometry exceeds the bounded model shape");
  switch (desc.kind) {
    case Ds4IndexerScoreKind::kOriginalDefault:
      if (!Mxf4Shape(desc)) return 0;
      break;
    case Ds4IndexerScoreKind::kMxf4:
      if (!Mxf4Shape(desc)) return Rejected("ds4 MXF4 score shape is ineligible");
      break;
    case Ds4IndexerScoreKind::kScalar:
    case Ds4IndexerScoreKind::kDirectOne:
    case Ds4IndexerScoreKind::kMultisequenceScalar:
    case Ds4IndexerScoreKind::kMultisequenceV5d:
    case Ds4IndexerScoreKind::kMultisequenceV5e:
    case Ds4IndexerScoreKind::kWmma16:
    case Ds4IndexerScoreKind::kWmma32:
    case Ds4IndexerScoreKind::kWmma64:
    case Ds4IndexerScoreKind::kWmma128:
      return 0;
    default:
      return Rejected("unknown ds4 indexer score kind");
  }
  const auto rows = static_cast<std::uint64_t>(desc.tokens) * kHeads;
  // Original mirror path packs four F32 power-of-two scales into one u32.
  // Requantization additionally writes64 packed bytes per query/head row.
  return Align(rows * sizeof(std::uint32_t)) +
         (Absent(desc.query_codes) ? Align(rows * kDs4IndexerPackedRowBytes) : 0);
}

std::expected<std::uint64_t, KernelFailure> Ds4IndexerSelectScratchBytes(
    const Ds4IndexerSelect& desc) {
  if (!Dimensions(desc.tokens, desc.cells, desc.score_band) || desc.top_k == 0 ||
      desc.top_k > 512 || desc.top_k > desc.score_band)
    return Rejected("ds4 indexer selection geometry is invalid");
  if (desc.kind != Ds4IndexerSelectKind::kChunkTree) return 0;
  if (desc.top_k != 512 || !Absent(desc.layer_scalars))
    return Rejected("ds4 chunked selection requires eager top512");
  std::uint32_t sets = (desc.score_band + kChunkCells - 1) / kChunkCells;
  std::uint64_t words_per_token = static_cast<std::uint64_t>(sets) * desc.top_k;
  while (sets > kMergeGroup) {
    sets = (sets + kMergeGroup - 1) / kMergeGroup;
    words_per_token += static_cast<std::uint64_t>(sets) * desc.top_k;
  }
  return words_per_token * desc.tokens * sizeof(std::uint32_t);
}

std::expected<void, KernelFailure> CheckDs4IndexerScores(const Ds4IndexerScores& desc) {
  if (!Dimensions(desc.tokens, desc.cells, desc.score_band) || desc.cells_per_bank < desc.cells ||
      desc.cells_per_bank > kMaxCells || desc.banks == 0 ||
      static_cast<std::uint64_t>(desc.banks) * desc.cells_per_bank >
          std::numeric_limits<std::uint32_t>::max() ||
      desc.tokens > std::numeric_limits<std::uint32_t>::max() - desc.first ||
      !std::isfinite(desc.scale) || desc.scale <= 0 || (desc.causal && desc.ratio == 0)) {
    return Rejected("ds4 indexer score dimensions/position/scale are invalid");
  }
  const bool packed = !Absent(desc.key_codes);
  if (packed != !Absent(desc.key_scales)) return Rejected("ds4 indexer key mirror is incomplete");
  if (!Absent(desc.query_codes) && Absent(desc.query_scales))
    return Rejected("ds4 indexer query mirror requires scales");
  if (Absent(desc.positions) != Absent(desc.bank_ids))
    return Rejected("ds4 indexer row positions and bank IDs must be paired");
  const bool per_row = !Absent(desc.bank_ids);
  const bool consecutive = desc.consecutive_bank != std::numeric_limits<std::uint32_t>::max();
  if (consecutive && (!per_row || desc.tokens < 2 || desc.consecutive_bank >= desc.banks))
    return Rejected("ds4 indexer consecutive-bank claim is invalid");
  const bool live = !Absent(desc.layer_scalars);
  if (live &&
      (desc.tokens != 1 || per_row || desc.causal || desc.score_band != desc.cells_per_bank))
    return Rejected("ds4 live indexer requires original direct-one stable capacity band");
  if (!live && desc.score_band != desc.cells)
    return Rejected("ds4 eager indexer band must equal compressed count");
  const bool independent_rows = per_row && !consecutive;
  switch (desc.kind) {
    case Ds4IndexerScoreKind::kOriginalDefault:
    case Ds4IndexerScoreKind::kMxf4:
      break;
    case Ds4IndexerScoreKind::kScalar:
      if (independent_rows || live) return Rejected("ds4 scalar indexer has no row/band overrides");
      break;
    case Ds4IndexerScoreKind::kDirectOne:
      if (desc.tokens != 1 || per_row) return Rejected("ds4 direct indexer needs one bank row");
      break;
    case Ds4IndexerScoreKind::kMultisequenceV5e:
      if (desc.tokens < 2 || desc.tokens > 8)
        return Rejected("ds4 V5E indexer only supports verify widths2..8");
      [[fallthrough]];
    case Ds4IndexerScoreKind::kMultisequenceV5d:
      if (Absent(desc.query_scales)) return Rejected("ds4 V5D/V5E requires query QAT scales");
      [[fallthrough]];
    case Ds4IndexerScoreKind::kMultisequenceScalar:
      if (!independent_rows || live)
        return Rejected("ds4 per-row indexer needs bank/position arrays");
      break;
    case Ds4IndexerScoreKind::kWmma16:
    case Ds4IndexerScoreKind::kWmma32:
    case Ds4IndexerScoreKind::kWmma64:
    case Ds4IndexerScoreKind::kWmma128:
      if (independent_rows || live)
        return Rejected("ds4 WMMA indexer requires one consecutive bank");
      break;
    default:
      return Rejected("unknown ds4 indexer score kind");
  }
  const auto scratch_bytes = Ds4IndexerScoreScratchBytes(desc);
  if (!scratch_bytes) return std::unexpected(scratch_bytes.error());
  if (desc.kind == Ds4IndexerScoreKind::kMxf4 && Absent(desc.scratch))
    return Rejected("ds4 explicit MXF4 requires paid query-preparation scratch");
  const auto rows = static_cast<std::uint64_t>(desc.tokens) * kHeads;
  const auto key_rows = static_cast<std::uint64_t>(desc.banks) * desc.cells_per_bank;
  const auto query_bytes = rows * kWidth * sizeof(float);
  const auto weight_bytes = rows * sizeof(float);
  const auto key_bytes = key_rows * kWidth * sizeof(float);
  const auto score_bytes =
      static_cast<std::uint64_t>(desc.tokens) * desc.score_band * sizeof(float);
  const auto row_bytes = static_cast<std::uint64_t>(desc.tokens) * sizeof(std::int32_t);
  if (!Holds(desc.query, query_bytes, 16) || !Holds(desc.weights, weight_bytes, 4) ||
      !Holds(desc.keys, key_bytes, 16) || !Holds(desc.scores, score_bytes, 4) ||
      !Holds(desc.diagnostics, sizeof(Ds4IndexerDiagnostics), alignof(Ds4IndexerDiagnostics)) ||
      (packed &&
       (!Holds(desc.key_codes, key_rows * 64, 16) || !Holds(desc.key_scales, key_rows * 16, 16))) ||
      (!Absent(desc.query_codes) && !Holds(desc.query_codes, rows * 64, 16)) ||
      (!Absent(desc.query_scales) && !Holds(desc.query_scales, rows * 16, 16)) ||
      (per_row && (!Holds(desc.positions, row_bytes, 4) || !Holds(desc.bank_ids, row_bytes, 4))) ||
      (live && !Holds(desc.layer_scalars, sizeof(Ds4IndexerLayerScalars), 4)) ||
      (!Absent(desc.scratch) && !Holds(desc.scratch, *scratch_bytes, kScratchAlignment))) {
    return Rejected("ds4 indexer score operand/scratch range is invalid");
  }
  const std::array accesses{Access{desc.query, query_bytes},
                            Access{desc.weights, weight_bytes},
                            Access{desc.keys, key_bytes},
                            Access{desc.key_codes, packed ? key_rows * 64 : 0},
                            Access{desc.key_scales, packed ? key_rows * 16 : 0},
                            Access{desc.query_codes, Absent(desc.query_codes) ? 0 : rows * 64},
                            Access{desc.query_scales, Absent(desc.query_scales) ? 0 : rows * 16},
                            Access{desc.positions, per_row ? row_bytes : 0},
                            Access{desc.bank_ids, per_row ? row_bytes : 0},
                            Access{desc.layer_scalars, live ? sizeof(Ds4IndexerLayerScalars) : 0},
                            Access{desc.scores, score_bytes, true},
                            Access{desc.scratch, Absent(desc.scratch) ? 0 : *scratch_bytes, true},
                            Access{desc.diagnostics, sizeof(Ds4IndexerDiagnostics), true}};
  if (!DisjointWrites(accesses)) return Rejected("ds4 indexer score writes overlap an operand");
  return {};
}

std::expected<void, KernelFailure> CheckDs4IndexerSelect(const Ds4IndexerSelect& desc) {
  const auto scratch_bytes = Ds4IndexerSelectScratchBytes(desc);
  if (!scratch_bytes) return std::unexpected(scratch_bytes.error());
  const bool live = !Absent(desc.layer_scalars);
  if ((live && desc.tokens != 1) || (!live && desc.score_band != desc.cells))
    return Rejected("ds4 indexer select band/live geometry is invalid");
  std::uint32_t ceiling = 0;
  switch (desc.kind) {
    case Ds4IndexerSelectKind::kOriginalDefault:
    case Ds4IndexerSelectKind::kInsertion:
      break;
    case Ds4IndexerSelectKind::kBitonic1024:
      ceiling = 1024;
      break;
    case Ds4IndexerSelectKind::kBitonic2048:
      ceiling = 2048;
      break;
    case Ds4IndexerSelectKind::kBitonic4096:
      ceiling = 4096;
      break;
    case Ds4IndexerSelectKind::kCub8192:
    case Ds4IndexerSelectKind::kBitonic8192:
      ceiling = 8192;
      break;
    case Ds4IndexerSelectKind::kStream512:
    case Ds4IndexerSelectKind::kChunkTree:
      if (desc.top_k != 512) return Rejected("ds4 streaming/tree selection needs top512");
      break;
    default:
      return Rejected("unknown ds4 indexer select kind");
  }
  if (ceiling != 0 && (desc.top_k != 512 || desc.score_band > ceiling))
    return Rejected("ds4 indexer selector specialization cannot hold the stable band");
  const auto scores = static_cast<std::uint64_t>(desc.tokens) * desc.score_band * sizeof(float);
  const auto selected =
      static_cast<std::uint64_t>(desc.tokens) * desc.top_k * sizeof(std::uint32_t);
  if (!Holds(desc.scores, scores, 4) || !Holds(desc.selected, selected, 4) ||
      !Holds(desc.diagnostics, sizeof(Ds4IndexerDiagnostics), alignof(Ds4IndexerDiagnostics)) ||
      (live && !Holds(desc.layer_scalars, sizeof(Ds4IndexerLayerScalars), 4)) ||
      (*scratch_bytes != 0 && !Holds(desc.scratch, *scratch_bytes, kScratchAlignment)) ||
      (*scratch_bytes == 0 && !Absent(desc.scratch)))
    return Rejected("ds4 indexer selection operand/scratch range is invalid");
  const std::array accesses{Access{desc.scores, scores}, Access{desc.selected, selected, true},
                            Access{desc.layer_scalars, live ? sizeof(Ds4IndexerLayerScalars) : 0},
                            Access{desc.scratch, *scratch_bytes, true},
                            Access{desc.diagnostics, sizeof(Ds4IndexerDiagnostics), true}};
  if (!DisjointWrites(accesses)) return Rejected("ds4 indexer selection writes overlap an operand");
  return {};
}

}  // namespace jitllm::kernels::ggml
