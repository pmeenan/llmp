// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "kernels/ggml/dsv4_ds4_attention.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>

namespace jitllm::kernels::ggml {
namespace {

constexpr std::uint32_t kMaxTokens = 4096;
constexpr std::uint32_t kMaxRawCells = 8192;
constexpr std::uint32_t kMaxCompressed = 262144;
constexpr std::uint32_t kHeadDim = 512;
constexpr std::uint32_t kHeads = 64;
constexpr std::uint64_t kF32 = 4;
constexpr std::uint64_t kF16 = 2;
constexpr std::uint64_t kAlignment = 256;

std::unexpected<KernelFailure> Rejected(std::string detail) {
  return std::unexpected(
      KernelFailure{.error = KernelError::kRejected, .detail = std::move(detail)});
}

bool Absent(const Ds4CacheBuffer& buffer) { return buffer.address == 0 && buffer.bytes == 0; }
bool Ring(const Ds4Attention& desc) {
  return desc.domain == Ds4AttentionDomain::kDecodeHeads ||
         desc.domain == Ds4AttentionDomain::kMixedRing ||
         desc.domain == Ds4AttentionDomain::kIndexedRing;
}
bool Indexed(const Ds4Attention& desc) { return desc.domain == Ds4AttentionDomain::kIndexedRing; }

bool Geometry(const Ds4Attention& desc) {
  return desc.tokens != 0 && desc.tokens <= kMaxTokens && desc.raw_cells != 0 &&
         desc.raw_cells <= kMaxRawCells && desc.raw_count != 0 &&
         desc.raw_count <= desc.raw_cells && desc.raw_start < desc.raw_cells &&
         desc.compressed_count <= desc.compressed_cells &&
         desc.compressed_cells <= kMaxCompressed && desc.banks != 0 &&
         static_cast<std::uint64_t>(desc.banks) * desc.raw_cells <=
             std::numeric_limits<std::uint32_t>::max() &&
         static_cast<std::uint64_t>(desc.banks) * desc.compressed_cells <=
             std::numeric_limits<std::uint32_t>::max() &&
         desc.tokens <= std::numeric_limits<std::uint32_t>::max() - desc.first &&
         desc.window <= 256;
}

bool Holds(const Ds4CacheBuffer& buffer, std::uint64_t bytes, std::uint64_t alignment) {
  return buffer.address != 0 && buffer.address % alignment == 0 && buffer.bytes >= bytes &&
         buffer.bytes <= std::numeric_limits<std::uint64_t>::max() - buffer.address;
}

struct Access {
  Ds4CacheBuffer buffer;
  std::uint64_t bytes = 0;
  bool writable = false;
};

std::uint64_t Aligned(std::uint64_t bytes) { return (bytes + kAlignment - 1) & ~(kAlignment - 1); }

bool TokenTileShape(const Ds4Attention& desc) {
  if (desc.tokens < 128 || desc.window != 128 || desc.ratio == 0 || !Absent(desc.mask) ||
      !Absent(desc.draft_raw_count) || !Absent(desc.decode_scalars) || !Absent(desc.layer_scalars))
    return false;
  if (Indexed(desc)) {
    return desc.top_k == 512 && desc.compressed_count != 0 &&
           desc.consecutive_first != std::numeric_limits<std::uint32_t>::max() &&
           (Absent(desc.positions) ||
            (desc.allow_multisequence_heads8 && !Absent(desc.bank_ids))) &&
           (Absent(desc.positions) == Absent(desc.bank_ids)) &&
           (!Absent(desc.positions) || desc.raw_count >= desc.tokens);
  }
  if (desc.compressed_count > 32768) return false;
  if (desc.domain == Ds4AttentionDomain::kMixedPrefill) return !desc.quality_mode;
  if (desc.domain != Ds4AttentionDomain::kMixedRing ||
      desc.consecutive_first == std::numeric_limits<std::uint32_t>::max())
    return false;
  return Absent(desc.positions) ? !desc.quality_mode && desc.raw_count >= desc.tokens
                                : desc.allow_multisequence_heads8 && !Absent(desc.bank_ids) &&
                                      desc.raw_cells >= desc.tokens + 127;
}

std::uint32_t HeadGroupSplits(std::uint32_t sms) {
  return std::clamp(((2 * sms) + 7) / 8, std::uint32_t{4}, std::uint32_t{16});
}

}  // namespace

std::expected<Ds4AttentionScratch, KernelFailure> PlanDs4AttentionScratch(const Ds4Attention& desc,
                                                                          Ds4AttentionKind selected,
                                                                          std::uint32_t device_sms,
                                                                          bool predecode) {
  if (!Geometry(desc) || device_sms == 0 || device_sms > 65535)
    return Rejected("ds4 attention scratch geometry/device shape is invalid");
  Ds4AttentionScratch layout;
  const auto append = [&layout](std::uint64_t& region, std::uint64_t bytes) {
    region = layout.bytes;
    layout.bytes = Aligned(layout.bytes + bytes);
  };
  const auto heads = static_cast<std::uint64_t>(desc.tokens) * kHeads;
  switch (selected) {
    case Ds4AttentionKind::kTokenTile: {
      if (!TokenTileShape(desc)) return Rejected("original ds4 token-tile shape is ineligible");
      layout.record_stride =
          Indexed(desc) ? std::min(4 * desc.top_k, desc.compressed_count) : desc.compressed_count;
      const auto tiles = (desc.tokens + 3) / 4;
      append(layout.records,
             std::uint64_t{tiles} * std::max(layout.record_stride, std::uint32_t{1}) * 8);
      append(layout.counts, std::uint64_t{tiles} * sizeof(std::uint32_t));
      append(layout.raw_mirror, (std::uint64_t{desc.tokens} + 127) * kHeadDim * kF16);
      append(layout.compressed_mirror, std::uint64_t{desc.compressed_count} * kHeadDim * kF16);
      return layout;
    }
    case Ds4AttentionKind::kHeadGroup:
      if (!Ring(desc) || desc.tokens > 8 || !Absent(desc.mask) || !Absent(desc.draft_raw_count) ||
          (Indexed(desc) && Absent(desc.positions)))
        return Rejected("original ds4 head-group shape is ineligible");
      layout.splits = HeadGroupSplits(device_sms);
      append(layout.partials, heads * layout.splits * (kHeadDim + 2) * kF32);
      return layout;
    case Ds4AttentionKind::kPerHeadSplit:
      if (desc.domain != Ds4AttentionDomain::kDecodeHeads || desc.tokens != 1 ||
          !Absent(desc.positions) || !Absent(desc.draft_raw_count) || device_sms <= kHeads)
        return Rejected("original ds4 per-head split shape is ineligible");
      layout.splits = std::clamp(((device_sms + (device_sms / 3)) + kHeads - 1) / kHeads,
                                 std::uint32_t{2}, std::uint32_t{8});
      append(layout.partials, heads * layout.splits * (kHeadDim + 2) * kF32);
      return layout;
    case Ds4AttentionKind::kCublas: {
      if (Ring(desc) || desc.tokens <= 1) return Rejected("original ds4 SGEMM only serves prefill");
      const auto keys =
          std::uint64_t{desc.tokens} +
          (desc.domain == Ds4AttentionDomain::kRawPrefill ? 0 : desc.compressed_count);
      if (desc.domain != Ds4AttentionDomain::kRawPrefill)
        append(layout.gemm_keys, keys * kHeadDim * kF32);
      append(layout.gemm_scores, heads * keys * kF32);
      append(layout.gemm_output, heads * kHeadDim * kF32);
      return layout;
    }
    case Ds4AttentionKind::kScalar:
      if (!Ring(desc) && desc.domain != Ds4AttentionDomain::kRawPrefill) {
        const auto raw = desc.window == 0 ? desc.tokens : std::min(desc.tokens, desc.window);
        const auto compressed =
            desc.ratio == 0 ? 0 : std::min(desc.compressed_count, desc.tokens / desc.ratio);
        if (raw + compressed > 512)
          return Rejected("original ds4 static scalar score array cannot hold this shape");
      }
      break;
    case Ds4AttentionKind::kHeads8Online:
      if (desc.tokens <= 1 && !Ring(desc))
        return Rejected("original ds4 static heads8 requires multiple rows");
      if (!Absent(desc.mask) || !Absent(desc.draft_raw_count))
        return Rejected("original ds4 online heads8 has no mask/draft span port");
      if (Indexed(desc) &&
          (desc.tokens <= 1 || !Absent(desc.decode_scalars) || !Absent(desc.layer_scalars)))
        return Rejected("original ds4 indexed heads8 has no live scalar port");
      break;
    case Ds4AttentionKind::kIndexedTwoPass:
      if (!Indexed(desc) || desc.tokens <= 1 || !Absent(desc.positions) ||
          !Absent(desc.decode_scalars) || !Absent(desc.layer_scalars))
        return Rejected("original ds4 indexed two-pass is single-sequence prefill only");
      break;
    case Ds4AttentionKind::kOriginalDefault:
    default:
      return Rejected("ds4 scratch planning needs a resolved original kernel kind");
  }
  if (Indexed(desc)) {
    if (desc.tokens > 1 && desc.top_k == 512) {
      layout.sort_ids = true;
      append(layout.sorted_ids, std::uint64_t{desc.tokens} * desc.top_k * sizeof(std::int32_t));
    }
    if (selected == Ds4AttentionKind::kScalar && predecode && !Absent(desc.compressed_codes) &&
        (Absent(desc.positions) || (Absent(desc.decode_scalars) && Absent(desc.layer_scalars)))) {
      layout.predecode = true;
      const auto rows = Absent(desc.positions) ? std::uint64_t{desc.compressed_cells}
                                               : std::uint64_t{desc.tokens} * desc.top_k;
      append(layout.predecoded, rows * kHeadDim * kF32);
    }
  }
  return layout;
}

std::expected<void, KernelFailure> CheckDs4Attention(const Ds4Attention& desc) {
  if (!Geometry(desc)) return Rejected("ds4 attention dimensions/position are invalid");
  switch (desc.domain) {
    case Ds4AttentionDomain::kRawPrefill:
      if (desc.compressed_count != 0 || desc.compressed_cells != 0)
        return Rejected("ds4 raw prefill has no compressed cells");
      [[fallthrough]];
    case Ds4AttentionDomain::kMixedPrefill:
    case Ds4AttentionDomain::kMaskedPrefill:
      if (desc.first != 0 || desc.raw_cells != desc.tokens || desc.raw_count != desc.tokens ||
          desc.raw_start != 0 || desc.banks != 1 || !Absent(desc.positions) ||
          !Absent(desc.bank_ids) || !Absent(desc.draft_raw_count) || !Absent(desc.decode_scalars) ||
          !Absent(desc.layer_scalars) ||
          (desc.domain != Ds4AttentionDomain::kRawPrefill && desc.ratio == 0))
        return Rejected("ds4 static prefill must be the original zero-prefix shape");
      break;
    case Ds4AttentionDomain::kDecodeHeads:
      if (desc.tokens != 1 || desc.ratio != 0 || desc.window != 0 || desc.raw_count > 256 ||
          !Absent(desc.positions) || !Absent(desc.bank_ids) || !Absent(desc.draft_raw_count))
        return Rejected("ds4 single decode-head entry requires original all-visible scalars");
      break;
    case Ds4AttentionDomain::kMixedRing:
    case Ds4AttentionDomain::kIndexedRing:
      if (Absent(desc.positions) &&
          static_cast<std::uint64_t>(desc.first) + desc.tokens < desc.raw_count)
        return Rejected("ds4 ring's oldest raw position would be negative");
      if (desc.compressed_count != 0 && desc.ratio == 0 &&
          (desc.domain != Ds4AttentionDomain::kMixedRing || desc.tokens != 1 ||
           !Absent(desc.positions)))
        return Rejected("ds4 compressed causal attention requires a ratio");
      break;
    default:
      return Rejected("unknown ds4 attention domain");
  }
  const bool positions = !Absent(desc.positions);
  const bool banks = !Absent(desc.bank_ids);
  if ((banks && !positions) || (!banks && desc.banks != 1) ||
      (!Absent(desc.draft_raw_count) && (!positions || Indexed(desc))))
    return Rejected("ds4 attention row/bank/draft arrays are inconsistent");
  if (desc.consecutive_first != std::numeric_limits<std::uint32_t>::max() &&
      (desc.tokens > std::numeric_limits<std::uint32_t>::max() - desc.consecutive_first ||
       (!positions && desc.consecutive_first != desc.first)))
    return Rejected("ds4 attention consecutive-run position is invalid");
  const bool indexed = Indexed(desc);
  const bool masked = !Absent(desc.mask);
  if ((indexed && (desc.compressed_count == 0 || desc.top_k == 0 || desc.top_k > 512 || masked)) ||
      (desc.domain == Ds4AttentionDomain::kMaskedPrefill && !masked) ||
      (masked && desc.domain != Ds4AttentionDomain::kMaskedPrefill &&
       desc.domain != Ds4AttentionDomain::kMixedRing &&
       desc.domain != Ds4AttentionDomain::kDecodeHeads) ||
      (!indexed && !Absent(desc.selected)))
    return Rejected("ds4 attention selected IDs/mask do not match the domain");
  const bool packed = !Absent(desc.compressed_codes);
  if (packed != !Absent(desc.compressed_scales) || packed != !Absent(desc.decode_table) ||
      (packed && desc.compressed_cells == 0))
    return Rejected("ds4 attention packed primary/table is incomplete");
  const auto rows = std::uint64_t{desc.tokens} * kHeads;
  const auto values = rows * kHeadDim * kF32;
  const auto raw = std::uint64_t{desc.banks} * desc.raw_cells * kHeadDim * kF32;
  const auto comp_rows = std::uint64_t{desc.banks} * desc.compressed_cells;
  const auto compressed = comp_rows * kHeadDim * kF32;
  const auto selected =
      indexed ? std::uint64_t{desc.tokens} * desc.top_k * sizeof(std::int32_t) : 0;
  const auto mask_cells =
      Absent(desc.layer_scalars) ? desc.compressed_count : desc.compressed_cells;
  const auto mask = masked ? std::uint64_t{desc.tokens} * mask_cells * kF32 : 0;
  const auto row_bytes = std::uint64_t{desc.tokens} * sizeof(std::int32_t);
  if (!Holds(desc.query, values, 16) || !Holds(desc.output, values, 16) ||
      !Holds(desc.sinks, kHeads * kF32, 4) || !Holds(desc.raw, raw, 16) ||
      (compressed != 0 && !Holds(desc.compressed, compressed, 16)) ||
      (packed && (!Holds(desc.compressed_codes, comp_rows * 704, 16) ||
                  !Holds(desc.compressed_scales, comp_rows * 28, 4) ||
                  !Holds(desc.decode_table, 128 * kF32, 4))) ||
      (indexed && !Holds(desc.selected, selected, 4)) || (masked && !Holds(desc.mask, mask, 4)) ||
      (positions && !Holds(desc.positions, row_bytes, 4)) ||
      (banks && !Holds(desc.bank_ids, row_bytes, 4)) ||
      (!Absent(desc.draft_raw_count) && !Holds(desc.draft_raw_count, row_bytes, 4)) ||
      (!Absent(desc.decode_scalars) &&
       !Holds(desc.decode_scalars, sizeof(Ds4AttentionDecodeScalars), 4)) ||
      (!Absent(desc.layer_scalars) &&
       !Holds(desc.layer_scalars, sizeof(Ds4IndexerLayerScalars), 4)) ||
      (!Absent(desc.scratch) && !Holds(desc.scratch, desc.scratch.bytes, kAlignment)) ||
      !Holds(desc.diagnostics, sizeof(Ds4AttentionDiagnostics), alignof(Ds4AttentionDiagnostics)))
    return Rejected("ds4 attention operand/scratch range is invalid");
  const std::array accesses{
      Access{desc.query, values},
      Access{desc.output, values, true},
      Access{desc.sinks, kHeads * kF32},
      Access{desc.raw, raw},
      Access{desc.compressed, compressed},
      Access{desc.compressed_codes, packed ? comp_rows * 704 : 0},
      Access{desc.compressed_scales, packed ? comp_rows * 28 : 0},
      Access{desc.decode_table, packed ? 128 * kF32 : 0},
      Access{desc.selected, selected},
      Access{desc.mask, mask},
      Access{desc.positions, positions ? row_bytes : 0},
      Access{desc.bank_ids, banks ? row_bytes : 0},
      Access{desc.draft_raw_count, Absent(desc.draft_raw_count) ? 0 : row_bytes},
      Access{desc.decode_scalars,
             Absent(desc.decode_scalars) ? 0 : sizeof(Ds4AttentionDecodeScalars)},
      Access{desc.layer_scalars, Absent(desc.layer_scalars) ? 0 : sizeof(Ds4IndexerLayerScalars)},
      Access{desc.scratch, desc.scratch.bytes, true},
      Access{desc.diagnostics, sizeof(Ds4AttentionDiagnostics), true}};
  for (const auto& access : accesses) {
    if (!Absent(access.buffer) && !Holds(access.buffer, access.buffer.bytes, 1))
      return Rejected("ds4 attention has a malformed provided logical view");
  }
  for (std::size_t i = 0; i < accesses.size(); ++i) {
    if (accesses[i].bytes == 0) continue;
    for (std::size_t j = i + 1; j < accesses.size(); ++j) {
      if (accesses[j].bytes != 0 && (accesses[i].writable || accesses[j].writable) &&
          accesses[i].buffer.address < accesses[j].buffer.address + accesses[j].bytes &&
          accesses[j].buffer.address < accesses[i].buffer.address + accesses[i].bytes)
        return Rejected("ds4 attention writes overlap an operand");
    }
  }
  if (desc.kind != Ds4AttentionKind::kOriginalDefault) {
    // Device-dependent split count/SM eligibility is resolved at launch.
    const auto plan = PlanDs4AttentionScratch(
        desc, desc.kind, desc.kind == Ds4AttentionKind::kPerHeadSplit ? 188 : 48);
    if (!plan) return std::unexpected(plan.error());
  }
  return {};
}

std::expected<void, KernelFailure> CheckDs4AttentionMask(const Ds4AttentionMask& desc) {
  if (desc.tokens == 0 || desc.tokens > kMaxTokens || desc.cells == 0 ||
      desc.cells > kMaxCompressed || desc.top_k == 0 || desc.top_k > 512)
    return Rejected("ds4 original selected-mask geometry is invalid");
  const auto selected = std::uint64_t{desc.tokens} * desc.top_k * sizeof(std::uint32_t);
  const auto mask = std::uint64_t{desc.tokens} * desc.cells * kF32;
  if (!Holds(desc.selected, selected, 4) || !Holds(desc.mask, mask, 4))
    return Rejected("ds4 selected-mask operand range is invalid");
  if (desc.selected.address < desc.mask.address + mask &&
      desc.mask.address < desc.selected.address + selected)
    return Rejected("ds4 selected-mask write overlaps its input");
  return {};
}

}  // namespace jitllm::kernels::ggml
