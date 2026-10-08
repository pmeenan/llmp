// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "artifact/layout.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

#include "artifact/error.h"
#include "base/bytes.h"

namespace llmp::artifact {
namespace {

std::unexpected<Error> Fail(Rule rule, std::string_view reason, std::uint64_t item = kNoItem) {
  return std::unexpected(Error{.rule = rule, .reason = reason, .item = item});
}

}  // namespace

std::expected<ChunkRange, Error> ChunkRangeOf(const Layout& layout, ChunkKey key) {
  if (key.group >= layout.groups.size()) {
    return Fail(Rule::kBounds, "no such group", key.group);
  }
  const Group& g = layout.groups[key.group];
  if (key.chunk >= g.chunks) {
    return Fail(Rule::kBounds, "no such chunk in the group", key.chunk);
  }
  if (g.shard >= layout.shards.size()) {
    return Fail(Rule::kBounds, "no such shard", g.shard);
  }
  const Shard& s = layout.shards[g.shard];
  // An opened artifact's layout always fits; one built by hand is checked.
  const Bytes start(std::uint64_t{key.chunk} * kChunkBytes);
  const auto base = s.data_offset.Plus(g.offset);
  const auto file_offset = base ? base->Plus(start) : std::nullopt;
  if (!file_offset || start >= g.stored) {
    return Fail(Rule::kBounds, "chunk outside the file", key.chunk);
  }
  return ChunkRange{
      .shard = g.shard,
      .file_offset = *file_offset,
      .length = Bytes(std::min(kChunkBytes, g.stored.value() - start.value())),
  };
}

std::expected<Closure, Error> ClosureOf(const Layout& layout, std::uint32_t group, Bytes offset,
                                        Bytes length) {
  if (group >= layout.groups.size()) {
    return Fail(Rule::kBounds, "no such group", group);
  }
  const Group& g = layout.groups[group];
  const auto end = offset.Plus(length);
  if (length.value() == 0 || !end || *end > g.stored) {
    return Fail(Rule::kBounds, "range outside the group", group);
  }
  return Closure{
      .group = group,
      .first = static_cast<std::uint32_t>(offset.value() / kChunkBytes),
      .last = static_cast<std::uint32_t>((end->value() - 1) / kChunkBytes),
  };
}

std::expected<std::vector<ReadRun>, Error> PlanReads(const Layout& layout,
                                                     std::span<const ChunkKey> missing,
                                                     std::span<const ChunkKey> resident,
                                                     ReadLimits limits) {
  std::vector<ChunkKey> keys(missing.begin(), missing.end());
  std::ranges::sort(keys);
  const auto [first, last] = std::ranges::unique(keys);
  keys.erase(first, last);
  std::vector<ChunkKey> held(resident.begin(), resident.end());
  std::ranges::sort(held);
  std::vector<ReadRun> runs;
  for (const ChunkKey key : keys) {
    if (std::ranges::binary_search(held, key)) {
      return Fail(Rule::kState, "a chunk both missing and resident", key.group);
    }
    auto range = ChunkRangeOf(layout, key);
    if (!range) {
      return std::unexpected(range.error());
    }
    const ReadSegment segment{.chunk = key,
                              .length = range->length,
                              .group_offset = Bytes(std::uint64_t{key.chunk} * kChunkBytes)};
    if (!runs.empty()) {
      ReadRun& run = runs.back();
      const auto run_end = run.file_offset.Plus(run.length);
      const auto grown = run.length.Plus(range->length);
      if (run.shard == range->shard && run_end && *run_end == range->file_offset && grown &&
          *grown <= limits.max_run && run.segments.size() < limits.max_segments) {
        run.length = *grown;
        run.segments.push_back(segment);
        continue;
      }
    }
    runs.push_back(ReadRun{.shard = range->shard,
                           .file_offset = range->file_offset,
                           .length = range->length,
                           .segments = {segment}});
  }
  return runs;
}

}  // namespace llmp::artifact
