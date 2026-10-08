// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "runtime/memory_guard.h"

#include <cstdint>
#include <expected>
#include <format>
#include <string>

namespace llmp::runtime {

std::uint64_t GuardReserve(const MemoryGuard& guard) {
  return guard.host_inputs + guard.plans + guard.requests + kUncountedMargin;
}

std::expected<void, std::string> CheckMemoryGuard(const MemoryGuard& guard) {
  const auto gib = [](std::uint64_t bytes) { return static_cast<double>(bytes) / (1ULL << 30U); };
  // Subtracted term by term, so no sum can wrap.
  const std::uint64_t a = guard.available;
  const bool fits =
      guard.largest <= a && guard.host_inputs <= a - guard.largest &&
      guard.plans <= a - guard.largest - guard.host_inputs &&
      kUncountedMargin <= a - guard.largest - guard.host_inputs - guard.plans &&
      guard.requests <= a - guard.largest - guard.host_inputs - guard.plans - kUncountedMargin;
  if (a != 0 && !fits) {
    return std::unexpected(std::format(
        "the largest model's weights ({:.1f} GiB), a model's largest host-built chunk inputs "
        "({:.1f} GiB), a model step's plans ({:.1f} GiB), the request memory ({:.1f} GiB, "
        "[client] request_memory_bytes) and a {:.0f} GiB margin do not fit the {:.1f} GiB "
        "available beside this node's fixed memory ({:.1f} GiB)",
        gib(guard.largest), gib(guard.host_inputs), gib(guard.plans), gib(guard.requests),
        gib(kUncountedMargin), gib(guard.available), gib(guard.fixed)));
  }
  return {};
}

std::expected<void, std::string> CheckDiagnosticBudgetCap(const MemoryGuard& guard,
                                                          std::uint64_t dynamic_budget,
                                                          std::uint64_t cap,
                                                          std::uint64_t extra_pinned) {
  if (guard.available == 0)
    return std::unexpected("diagnostic budget cap requires known physical availability");
  if (auto checked = CheckMemoryGuard(guard); !checked) return checked;
  if (cap > dynamic_budget)
    return std::unexpected("diagnostic budget cap exceeds the ordinary dynamic budget");
  // Subtract the required terms rather than adding an overflowing minimum.
  if (guard.fixed > cap || guard.largest > cap - guard.fixed ||
      extra_pinned > cap - guard.fixed - guard.largest)
    return std::unexpected("diagnostic budget cap is below the required startup footprint");
  return {};
}

}  // namespace llmp::runtime
