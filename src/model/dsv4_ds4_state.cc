// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "model/dsv4_ds4_state.h"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <expected>
#include <string>
#include <vector>

namespace jitllm::model {
namespace {
using Kind = Ds4BaselineStateKind;
constexpr std::uint64_t kAlignment = 256;

bool Pad(std::uint64_t value, std::uint64_t alignment, std::uint64_t& result) {
  std::uint64_t added = 0;
  if (__builtin_add_overflow(value, alignment - 1, &added)) return false;
  result = added & ~(alignment - 1);
  return true;
}

bool Add(Ds4BaselineStateLayout& layout, Kind kind, std::uint32_t layer, std::uint32_t ratio,
         std::uint32_t rows, std::uint64_t row_bytes, std::uint64_t alignment) {
  std::uint64_t bytes = 0;
  std::uint64_t offset = 0;
  std::uint64_t end = 0;
  if (__builtin_mul_overflow(std::uint64_t{rows}, row_bytes, &bytes) || bytes == 0 ||
      !Pad(layout.virtual_bytes, alignment, offset) || __builtin_add_overflow(offset, bytes, &end))
    return false;
  layout.tensors.push_back({.kind = kind,
                            .layer = layer,
                            .ratio = ratio,
                            .capacity = rows,
                            .row_bytes = row_bytes,
                            .offset = offset,
                            .bytes = bytes});
  layout.virtual_bytes = end;
  return true;
}

bool Geometry(const Dsv4Profile& profile) {
  // The pinned original numerical path is Flash-specific. Refuse another
  // geometry instead of suggesting that the original launch rules generalize.
  if (profile.layers != 43 || profile.width != 4096 || profile.hc != 4 || profile.heads != 64 ||
      profile.head_dim != 512 || profile.rope_dims != 64 || profile.window != 128 ||
      profile.indexer_head_dim != 128 || profile.indexer_heads != 64 ||
      profile.indexer_top_k != 512 || profile.compress_ratios.size() != profile.layers)
    return false;
  for (std::uint32_t layer = 0; layer < profile.layers; ++layer) {
    std::uint32_t ratio = 0;
    if (layer >= 2) ratio = layer % 2 == 0 ? 4U : 128U;
    if (profile.compress_ratios[layer] != ratio) return false;
  }
  return true;
}
}  // namespace

std::expected<Ds4BaselineStateLayout, std::string> LayoutDs4BaselineState(
    const Dsv4Profile& profile, std::uint32_t context, std::uint32_t max_rows,
    std::uint64_t granularity, bool packed_kv, bool packed_indexer) {
  if (!Geometry(profile) || context == 0 || context > kDsv4FlashContext || max_rows == 0 ||
      max_rows > 4096 || !std::has_single_bit(granularity) || granularity < kAlignment ||
      granularity > (std::uint64_t{1} << 30U)) {
    return std::unexpected(
        "ds4 baseline state requires pinned Flash geometry and bounded capacity");
  }
  const auto window = std::min(profile.window, context);
  std::uint64_t raw_padded = 0;
  const auto wanted = std::min(std::uint64_t{context}, std::uint64_t{window} + max_rows);
  if (!Pad(wanted, kAlignment, raw_padded)) {
    return std::unexpected("ds4 baseline raw capacity overflows");
  }
  const auto raw_cells = static_cast<std::uint32_t>(
      std::min({raw_padded, std::uint64_t{8192}, std::uint64_t{context}}));
  Ds4BaselineStateLayout layout{.context = context,
                                .max_rows = max_rows,
                                .raw_cells = raw_cells,
                                .packed_kv = packed_kv,
                                .packed_indexer = packed_indexer,
                                .granularity = granularity,
                                .tensors = {}};
  bool complete = true;
  for (std::uint32_t layer = 0; layer < profile.layers; ++layer) {
    complete = complete &&
               Add(layout, Kind::kRaw, layer, 0, raw_cells, std::uint64_t{512} * 4, kAlignment);
    const auto ratio = profile.compress_ratios[layer];
    if (ratio == 0) continue;
    const auto width = ratio == 4 ? 1024U : 512U;
    const auto rows = ratio == 4 ? 8U : 128U;
    complete = complete && Add(layout, Kind::kAttentionKv, layer, 0, rows, std::uint64_t{width} * 4,
                               kAlignment);
    complete = complete && Add(layout, Kind::kAttentionScore, layer, 0, rows,
                               std::uint64_t{width} * 4, kAlignment);
    if (ratio == 4) {
      complete = complete &&
                 Add(layout, Kind::kIndexerKv, layer, 0, 8, std::uint64_t{256} * 4, kAlignment);
      complete = complete &&
                 Add(layout, Kind::kIndexerScore, layer, 0, 8, std::uint64_t{256} * 4, kAlignment);
    }
  }
  for (std::uint32_t layer = 0; layer < profile.layers; ++layer) {
    complete = complete && Add(layout, Kind::kLayerScalar, layer, 0, 1, 16, kAlignment);
  }
  complete = complete && Add(layout, Kind::kDecodeScalar, 0, 0, 1, 40, kAlignment);
  complete = complete && Add(layout, Kind::kDecodeTable, 0, 0, 1, 512, kAlignment);
  complete = complete && Pad(layout.virtual_bytes, granularity, layout.virtual_bytes);
  layout.fixed_bytes = layout.virtual_bytes;
  for (std::uint32_t layer = 0; layer < profile.layers; ++layer) {
    const auto ratio = profile.compress_ratios[layer];
    if (ratio == 0) continue;
    const auto capacity = (context / ratio) + 2;
    if (packed_kv) {
      complete =
          complete && Add(layout, Kind::kAttentionCodes, layer, ratio, capacity, 704, granularity);
      complete =
          complete && Add(layout, Kind::kAttentionScales, layer, ratio, capacity, 28, granularity);
    } else {
      complete = complete && Add(layout, Kind::kAttentionF32, layer, ratio, capacity,
                                 std::uint64_t{512} * 4, granularity);
    }
    if (ratio == 4 && packed_indexer) {
      complete =
          complete && Add(layout, Kind::kIndexerCodes, layer, ratio, capacity, 64, granularity);
      complete =
          complete && Add(layout, Kind::kIndexerScales, layer, ratio, capacity, 16, granularity);
    } else if (ratio == 4) {
      complete = complete && Add(layout, Kind::kIndexerF32, layer, ratio, capacity,
                                 std::uint64_t{128} * 4, granularity);
    }
  }
  complete = complete && Pad(layout.virtual_bytes, granularity, layout.virtual_bytes);
  if (!complete) return std::unexpected("ds4 baseline state byte layout overflows");
  return layout;
}

std::expected<std::vector<StateRange>, std::string> Ds4BaselineStateThrough(
    const Ds4BaselineStateLayout& layout, std::uint32_t positions) {
  if (positions > layout.context || layout.context == 0 || layout.fixed_bytes == 0 ||
      layout.fixed_bytes > layout.virtual_bytes || !std::has_single_bit(layout.granularity)) {
    return std::unexpected("ds4 baseline state prefix or layout is invalid");
  }
  std::vector<StateRange> ranges{{.offset = 0, .bytes = layout.fixed_bytes}};
  for (const auto& tensor : layout.tensors) {
    if (tensor.ratio == 0) continue;
    if ((tensor.ratio != 4 && tensor.ratio != 128) || tensor.row_bytes == 0 ||
        tensor.offset < layout.fixed_bytes || tensor.offset % layout.granularity != 0 ||
        tensor.offset > layout.virtual_bytes ||
        tensor.bytes > layout.virtual_bytes - tensor.offset ||
        tensor.capacity != (layout.context / tensor.ratio) + 2 ||
        tensor.bytes / tensor.row_bytes != tensor.capacity ||
        tensor.bytes % tensor.row_bytes != 0) {
      return std::unexpected("ds4 baseline growing cache descriptor is invalid");
    }
    const auto rows = positions / tensor.ratio;
    const auto bytes = std::uint64_t{rows} * tensor.row_bytes;
    if (bytes != 0) ranges.push_back({.offset = tensor.offset, .bytes = bytes});
  }
  return ranges;
}

}  // namespace jitllm::model
