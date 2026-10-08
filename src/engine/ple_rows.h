// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Qwen3.8's n-gram (PLE) table paged by rows on demand (D-035; M3's swap
// path, docs/experiments/fast-swap/swap.md): the table (28.8 GB of 90-byte
// rows) is never resident; before each chunk's job the rows its tokens'
// n-gram hash names are read from the artifact, and the job gathers them
// on the GPU into the chunk's row slots, which the graph's row lookup then
// reads in place of the table.
//
// - Granularity: sub-chunk direct reads. Each distinct row's 90 bytes are
//   covered by the 4 KiB blocks around them (one block, or two where the
//   row crosses a block boundary); blocks of rows that touch or overlap
//   are read together, up to `max_read` bytes a read (a read split at that
//   limit starts at its row's first block, which may repeat the last block
//   of the read before: a row lies within one read). The v0 format's reads
//   are whole 2 MiB chunks (docs/artifact-format.md); these reads are
//   smaller, inside one chunk's stored range or across two consecutive
//   chunks of the same shard, and read nothing the chunk does not store.
//   A whole-chunk policy's bytes are counted beside them (`extents`), the
//   evidence D-035 asks for a smaller-read path.
// - Validity and accounting: the rows are the chunk's alone. The landing
//   (pinned host memory, cataloged staging) and the row slots (device
//   memory, cataloged) are fixed-size and charged whole; nothing persists
//   from one chunk to the next, so no row has a residency of its own to
//   track or evict.
// - No CPU payload copy: the reads land by direct I/O and a kernel copies
//   each row into its slot (D-081's GPU copy, by an SM kernel here).

#ifndef LLMP_ENGINE_PLE_ROWS_H_
#define LLMP_ENGINE_PLE_ROWS_H_

#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace llmp::providers {
class Storage;
}

namespace llmp::engine {

// Where the table is: `rows` rows of `row_bytes` from `file_offset` in one
// shard, stored contiguously; `chunk_file_offset` is where the table's
// group (its first 2 MiB chunk) starts in that shard, for counting whole
// chunks.
struct PleTable {
  int fd = -1;
  std::uint64_t file_offset = 0;
  std::uint64_t rows = 0;
  std::uint64_t row_bytes = 0;
  std::uint64_t chunk_file_offset = 0;
  std::uint64_t file_bytes = 0;  // the end of the group's stored range: no read passes it
};

// One direct read: a 4 KiB-aligned range of the table's shard into the
// landing at `landing` (4 KiB-aligned).
struct PleRead {
  std::uint64_t file_offset = 0;
  std::uint64_t length = 0;
  std::uint64_t landing = 0;
};

struct PleRowPlan {
  // For each lookup (the chunk's ple_rows, in order), its row slot.
  std::vector<std::int32_t> slots;
  // For each slot, where its row starts in the landing.
  std::vector<std::uint32_t> sources;
  std::vector<PleRead> reads;
  std::uint64_t landing_bytes = 0;  // the reads' total
  std::uint64_t useful_bytes = 0;   // distinct rows x row bytes
  std::uint64_t extents = 0;        // distinct 2 MiB chunks the rows touch
};

inline constexpr std::uint64_t kPleBlock = 4096;
inline constexpr std::uint64_t kPleMaxRead = std::uint64_t{64} << 10U;

// Plans a chunk's rows. Refused if a row is outside the table, the table's
// placement is not 4 KiB-aligned where it must be, or the reads need more
// than `landing_capacity` bytes or more than `max_slots` slots.
std::expected<PleRowPlan, std::string> PlanPleRows(const PleTable& table,
                                                   std::span<const std::int32_t> rows,
                                                   std::uint64_t landing_capacity,
                                                   std::uint64_t max_slots,
                                                   std::uint64_t max_read = kPleMaxRead);

// The most landing bytes a chunk of `lookups` lookups can need: each row's
// two blocks.
constexpr std::uint64_t PleLandingBound(std::uint64_t lookups) { return lookups * 2 * kPleBlock; }

// Runs a plan's reads into `landing` on `storage` (the caller's own ring,
// never the storage lane's), polling (never sleeping) until every one has
// completed, an unknown submission included (it is in flight, storage.h);
// refused on a short or failed read,
// after draining the rest. Refused without draining if reads make no
// progress for 30 s: they may still land, so the caller must neither reuse
// the landing nor destroy the ring.
std::expected<void, std::string> ReadPleRows(providers::Storage& storage, int fd,
                                             const PleRowPlan& plan, std::byte* landing);

// The job's gather is a kernel of its own (kernels/paging/paging.h,
// GatherPleRows).

}  // namespace llmp::engine

#endif  // LLMP_ENGINE_PLE_ROWS_H_
