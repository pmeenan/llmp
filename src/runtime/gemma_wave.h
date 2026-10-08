// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef LLMP_RUNTIME_GEMMA_WAVE_H_
#define LLMP_RUNTIME_GEMMA_WAVE_H_

#include <algorithm>
#include <cstddef>
#include <expected>
#include <string>

namespace llmp::runtime {
// Callers validate all tuples before this first dispatch. A successful run
// proves its group's native completion. A clean refusal belongs only to that
// group; completed peers remain available for the caller's later publication.
template <class Run, class Usable, class Refused>
std::expected<void, std::string> RunGemmaGroups(std::size_t owners, std::size_t limit, Run&& run,
                                                Usable&& usable, Refused&& refused) {
  if (owners == 0 || owners > 12 || limit == 0 || limit > 12)
    return std::unexpected("invalid bounded Gemma group envelope");
  for (std::size_t first = 0; first < owners; first += limit) {
    const auto count = std::min(limit, owners - first);
    const auto result = run(first, count);
    if (result) continue;
    if (!usable()) return result;
    refused(first, count, result.error());
  }
  return {};
}
}  // namespace llmp::runtime
#endif  // LLMP_RUNTIME_GEMMA_WAVE_H_
