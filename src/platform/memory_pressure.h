// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Pressure on the host's memory from outside the runtime (D-004: a Spark's
// 128 GB is one budget that other processes share): the memory available
// to a new allocation and Linux's pressure-stall information
// (/proc/pressure/memory), which the runtime's watcher reads between units
// to trim what it can give back through its reclaim order (D-055 as
// amended 2026-10-02); and the C library's free heap returned to the
// system once plans are dropped.

#ifndef JITLLM_PLATFORM_MEMORY_PRESSURE_H_
#define JITLLM_PLATFORM_MEMORY_PRESSURE_H_

#include <cstdint>
#include <optional>
#include <string_view>

namespace jitllm::platform {

// One line of a pressure file: the share of the last 10 seconds in which
// some (or all) tasks stalled on memory, in percent.
struct PressureStall {
  double some_avg10 = 0;
  double full_avg10 = 0;
};

// Parses /proc/pressure/memory's text ("some avg10=... total=...\nfull
// avg10=..."); unknown if it does not hold both lines.
std::optional<PressureStall> ParsePressure(std::string_view text);

struct MemoryPressure {
  std::optional<std::uint64_t> available;  // MemAvailable, bytes
  std::optional<PressureStall> stall;      // absent without PSI
};

// Both, now.
MemoryPressure ReadMemoryPressure();

// Returns the C library's free heap pages to the system (glibc's
// malloc_trim): freed plans' memory then shows as available again.
void ReleaseFreeHeap();

}  // namespace jitllm::platform

#endif  // JITLLM_PLATFORM_MEMORY_PRESSURE_H_
