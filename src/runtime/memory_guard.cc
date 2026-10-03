// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "runtime/memory_guard.h"

#include <cstdint>
#include <expected>
#include <format>
#include <string>

namespace jitllm::runtime {

std::uint64_t GuardReserve(const MemoryGuard& guard) {
  return guard.host_inputs + guard.plans + kUncountedMargin;
}

std::expected<void, std::string> CheckMemoryGuard(const MemoryGuard& guard) {
  const auto gib = [](std::uint64_t bytes) { return static_cast<double>(bytes) / (1ULL << 30U); };
  // Subtracted term by term, so no sum can wrap.
  const std::uint64_t a = guard.available;
  const bool fits = guard.largest <= a && guard.host_inputs <= a - guard.largest &&
                    guard.plans <= a - guard.largest - guard.host_inputs &&
                    kUncountedMargin <= a - guard.largest - guard.host_inputs - guard.plans;
  if (a != 0 && !fits) {
    return std::unexpected(std::format(
        "the largest model's weights ({:.1f} GiB), a model's largest host-built chunk inputs "
        "({:.1f} GiB), a model's plans and graphs ({:.1f} GiB) and a {:.0f} GiB margin do not "
        "fit the {:.1f} GiB available beside this node's fixed memory ({:.1f} GiB)",
        gib(guard.largest), gib(guard.host_inputs), gib(guard.plans), gib(kUncountedMargin),
        gib(guard.available), gib(guard.fixed)));
  }
  return {};
}

}  // namespace jitllm::runtime
