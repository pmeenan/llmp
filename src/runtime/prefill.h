// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// A turn's prefill in chunks (docs/runtime-serving.md#prefill-chunks):
// how many rows a chunk takes, and the loop over the chunks, which asks
// between chunks whether to go on. Vendor-free, so the CPU tests drive it;
// the models' runs (serving.h) supply the chunks.
//
// - The chunk: a model's `prefill_chunk` if configured, else the runtime's
//   default for the model, at most what the model's state layout allows at
//   its context (`most`) and below the context (a prompt is shorter than
//   the context, and a chunk as long as it would leave no row to generate),
//   in whole tiles of kPrefillRowTile rows.
// - Wide chunks: a chunk of kPrefillTiledFrom rows or more runs in whole
//   tiles, and its remainder (fewer rows than a tile) as a chunk of its
//   own: from 1,024 rows on, the flash-attention kernels' mask pre-pass
//   reads the mask in whole 8-row column tiles (kernels/ggml/fattn_mma.cu),
//   and the models' masks have exactly the chunk's rows (RE-036).
// - Cancellation: a chunk is the granularity. Before each chunk the loop
//   asks `go_on` with the chunk's rows (the client still there, the
//   backend making progress, a non-streaming deadline not passed, the
//   runtime not stopping; the rows set the watchdog's allowance for the
//   chunk, watchdog.h); false stops it there, and what ran is exactly
//   the chunks before, so the conversation's state holds a prefix of the
//   turn's tokens and nothing it did not process.

#ifndef LLMP_RUNTIME_PREFILL_H_
#define LLMP_RUNTIME_PREFILL_H_

#include <cstdint>
#include <expected>
#include <functional>
#include <optional>
#include <string>

namespace llmp::runtime {

inline constexpr std::uint32_t kPrefillRowTile = 8;
inline constexpr std::uint32_t kPrefillTiledFrom = 1024;

// Rows of a prefill chunk (above); 0 when none fits (a context under 2
// tokens, or a model that allows none). From kPrefillRowTile rows on, a
// multiple of it.
std::uint32_t PrefillChunkRows(std::uint32_t context, std::optional<std::uint32_t> configured,
                               std::uint32_t preferred, std::uint32_t most);

// Optional prediction for planning, capture and fresh backing preparation;
// zero rows means none. It publishes no future logical state and never evicts
// useful cache for backing. A wrong hint costs only unused optional work.
struct PrefillHint {
  std::uint32_t rows = 0;
  bool want_head = true;
  // The chunk after that one, likewise (zero rows: none or unknown).
  std::uint32_t after_rows = 0;
  bool after_want_head = true;
};
// The ordinary tiled chunk geometry, without consuming any input.
std::uint32_t PrefillRows(std::uint32_t remaining, std::uint32_t maximum);

// How a prefill's chunks ran.
struct PrefillRun {
  std::uint32_t end = 0;     // the position after the last chunk that ran
  std::uint32_t chunks = 0;  // chunks run
  bool stopped = false;      // go_on said no before a chunk
  double longest = 0;        // seconds: the longest chunk, the cancellation granularity
};

// Runs positions [from, end) in chunks of at most `rows`, in order (one of
// kPrefillTiledFrom rows or more in whole tiles, above): before each,
// `go_on` (if set) is asked with its rows, and false stops the loop with
// PrefillRun::end the position the chunks reached. A chunk's failure is
// the error, with its position; the chunks before it ran.
using PrefillGoOn = std::function<bool(std::uint32_t rows)>;
using PrefillChunk =
    std::function<std::expected<void, std::string>(std::uint32_t at, std::uint32_t rows)>;
std::expected<PrefillRun, std::string> RunPrefillChunks(std::uint32_t from, std::uint32_t end,
                                                        std::uint32_t rows,
                                                        const PrefillChunk& chunk,
                                                        const PrefillGoOn& go_on);

}  // namespace llmp::runtime

#endif  // LLMP_RUNTIME_PREFILL_H_
