// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Where a v0 artifact's groups and chunks live, and how missing chunks are
// read (docs/artifact-format.md, "Page-in contract"; D-056):
//
//   - A group is one contiguous, 4 KiB-aligned range of one shard's data
//     section. Its chunks are group-relative 2 MiB slices
//     [k * 2 MiB, min((k + 1) * 2 MiB, stored)), numbered globally in group
//     order from the group's first_chunk.
//   - A byte range of a group needs every chunk it touches (its closure).
//   - Missing chunks of one shard that are adjacent in the file coalesce
//     into one vectored direct read, with one segment per chunk, each into
//     that chunk's own destination at offset k * 2 MiB of the group's
//     backing. A run breaks at a chunk not asked for (a resident one is
//     never re-read to bridge a gap), at a shard boundary, at the run's
//     byte limit and at the iovec limit.
//
// These functions take positions from the caller (group, chunk and row
// numbers are untrusted: token IDs choose rows) and reject any out of
// range; they never clamp.

#ifndef LLMP_ARTIFACT_LAYOUT_H_
#define LLMP_ARTIFACT_LAYOUT_H_

#include <compare>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "artifact/error.h"
#include "base/bytes.h"
#include "base/sha256.h"

namespace llmp::artifact {

using base::Bytes;
using Digest = base::Sha256Digest;

// The spark-v0 profile (manifest `layout`); any other is refused.
inline constexpr std::uint64_t kFileAlignment = 4096;
inline constexpr std::uint64_t kChunkBytes = std::uint64_t{2} << 20U;
inline constexpr std::uint64_t kMemberAlignment = 256;

struct Shard {
  std::string path;        // "data/NNNNN.safetensors", beneath the artifact
  Bytes data_offset;       // where the data section starts in the file
  Bytes data_bytes;        // its length; the file is data_offset + data_bytes
  Digest header_sha256{};  // over the file's bytes before the data section
};

enum class GroupKind : std::uint8_t { kTable, kLayer, kExpert, kGlobal, kHead };

struct Group {
  GroupKind kind = GroupKind::kLayer;
  std::optional<std::uint32_t> layer;   // layer and expert groups
  std::optional<std::uint32_t> expert;  // expert groups
  std::uint32_t shard = 0;
  Bytes offset;                   // within the shard's data section
  Bytes used;                     // end of the last readable range
  Bytes stored;                   // used, rounded up to 4 KiB
  std::uint64_t first_chunk = 0;  // global number of its chunk 0
  std::uint32_t chunks = 0;       // ceil(stored / 2 MiB)
};

struct Layout {
  std::vector<Shard> shards;
  std::vector<Group> groups;
};

// A group-relative chunk.
struct ChunkKey {
  std::uint32_t group = 0;
  std::uint32_t chunk = 0;
  auto operator<=>(const ChunkKey&) const = default;
};

// Where one chunk is on disk: 4 KiB-aligned, at most 2 MiB.
struct ChunkRange {
  std::uint32_t shard = 0;
  Bytes file_offset;  // in the shard file, header included
  Bytes length;
};

// The chunks [first, last] of one group that a byte range touches.
struct Closure {
  std::uint32_t group = 0;
  std::uint32_t first = 0;
  std::uint32_t last = 0;
};

struct ReadSegment {
  ChunkKey chunk{};
  Bytes length;        // this chunk's stored bytes
  Bytes group_offset;  // destination: chunk * 2 MiB within the group's backing
};

// One vectored direct read: a contiguous file range of one shard.
struct ReadRun {
  std::uint32_t shard = 0;
  Bytes file_offset;
  Bytes length;
  std::vector<ReadSegment> segments;
};

struct ReadLimits {
  Bytes max_run{std::uint64_t{64} << 20U};  // a tuning value, not ABI
  std::size_t max_segments = 1024;          // IOV_MAX on both hosts
};

std::expected<ChunkRange, Error> ChunkRangeOf(const Layout& layout, ChunkKey key);

// The closure of [offset, offset + length) within a group; the range must
// be non-empty and inside the group's stored bytes.
std::expected<Closure, Error> ClosureOf(const Layout& layout, std::uint32_t group, Bytes offset,
                                        Bytes length);

// Coalesced reads of `missing` (deduplicated, in file order). A chunk named
// in both `missing` and `resident` is the caller's bug, refused as
// kState. A run always takes at least one chunk, whatever the limits.
std::expected<std::vector<ReadRun>, Error> PlanReads(const Layout& layout,
                                                     std::span<const ChunkKey> missing,
                                                     std::span<const ChunkKey> resident,
                                                     ReadLimits limits = {});

}  // namespace llmp::artifact

#endif  // LLMP_ARTIFACT_LAYOUT_H_
