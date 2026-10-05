// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "runtime/prefill.h"

#include <algorithm>
#include <chrono>
#include <format>

namespace jitllm::runtime {

std::uint32_t PrefillChunkRows(std::uint32_t context, std::optional<std::uint32_t> configured,
                               std::uint32_t preferred, std::uint32_t most) {
  if (context < 2) {
    return 0;
  }
  const std::uint32_t rows = std::min({configured.value_or(preferred), most, context - 1});
  return rows < kPrefillRowTile ? rows : rows - (rows % kPrefillRowTile);
}

std::uint32_t PrefillRows(std::uint32_t remaining, std::uint32_t maximum) {
  std::uint32_t rows = std::min(remaining, maximum);
  if (rows >= kPrefillTiledFrom) rows -= rows % kPrefillRowTile;
  return rows;
}

std::expected<PrefillRun, std::string> RunPrefillChunks(std::uint32_t from, std::uint32_t end,
                                                        std::uint32_t rows,
                                                        const PrefillChunk& chunk,
                                                        const PrefillGoOn& go_on) {
  using Clock = std::chrono::steady_clock;
  PrefillRun run{.end = from, .chunks = 0, .stopped = false, .longest = 0};
  if (rows == 0 && from < end) {
    return std::unexpected("a prefill chunk of 0 rows");
  }
  while (run.end < end) {
    const std::uint32_t n = PrefillRows(end - run.end, rows);
    if (go_on && !go_on(n)) {
      run.stopped = true;
      return run;
    }
    const auto started = Clock::now();
    if (auto r = chunk(run.end, n); !r) {
      return std::unexpected(std::format("the chunk at {}: {}", run.end, r.error()));
    }
    run.longest =
        std::max(run.longest, std::chrono::duration<double>(Clock::now() - started).count());
    run.end += n;
    ++run.chunks;
  }
  return run;
}

}  // namespace jitllm::runtime
