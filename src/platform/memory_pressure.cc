// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "platform/memory_pressure.h"

#include <malloc.h>

#include <charconv>
#include <cstddef>
#include <optional>
#include <string_view>
#include <system_error>

#include "platform/files.h"
#include "platform/host_probe.h"

namespace jitllm::platform {

namespace {

// The avg10 value on the line that starts with `kind` ("some", "full").
std::optional<double> Avg10(std::string_view text, std::string_view kind) {
  std::size_t at = 0;
  while (at < text.size()) {
    std::size_t end = text.find('\n', at);
    if (end == std::string_view::npos) {
      end = text.size();
    }
    const std::string_view line = text.substr(at, end - at);
    at = end + 1;
    if (!line.starts_with(kind) || line.size() <= kind.size() || line[kind.size()] != ' ') {
      continue;
    }
    constexpr std::string_view kKey = "avg10=";
    const std::size_t key = line.find(kKey);
    if (key == std::string_view::npos) {
      return std::nullopt;
    }
    const std::string_view value = line.substr(key + kKey.size());
    double parsed = 0;
    const auto [ptr, ec] = std::from_chars(value.data(), value.data() + value.size(), parsed);
    if (ec != std::errc() || ptr == value.data() || parsed < 0 || parsed > 100) {
      return std::nullopt;
    }
    return parsed;
  }
  return std::nullopt;
}

}  // namespace

std::optional<PressureStall> ParsePressure(std::string_view text) {
  const auto some = Avg10(text, "some");
  const auto full = Avg10(text, "full");
  if (!some || !full) {
    return std::nullopt;
  }
  return PressureStall{.some_avg10 = *some, .full_avg10 = *full};
}

MemoryPressure ReadMemoryPressure() {
  MemoryPressure out;
  out.available = AvailableMemoryBytes();
  if (const auto text = ReadSmallFile("/proc/pressure/memory"); text) {
    out.stall = ParsePressure(*text);
  }
  return out;
}

void ReleaseFreeHeap() { (void)::malloc_trim(0); }

}  // namespace jitllm::platform
