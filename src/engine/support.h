// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Small helpers every engine file uses (docs/engine.md): an error result,
// an address as a pointer and back, rounding up, seconds of a duration, and
// a list of problems joined into one error. Header-only.

#ifndef JITLLM_ENGINE_SUPPORT_H_
#define JITLLM_ENGINE_SUPPORT_H_

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace jitllm::engine::support {

inline std::unexpected<std::string> Error(std::string what) {
  return std::unexpected(std::move(what));
}

inline void* Pointer(std::uint64_t address) {
  return reinterpret_cast<void*>(address);  // NOLINT(performance-no-int-to-ptr)
}

inline std::uint64_t Address(const void* pointer) {
  return reinterpret_cast<std::uintptr_t>(pointer);
}

// `bytes` rounded up to a multiple of `to`.
inline std::uint64_t Round(std::uint64_t bytes, std::uint64_t to) {
  return (bytes + to - 1) / to * to;
}

inline double Seconds(std::chrono::steady_clock::duration d) {
  return std::chrono::duration<double>(d).count();
}

// Preserve the usual endpoint probes, then enumerate every positive owner-row
// composition whose total can select shared MMVQ. At eight columns there are
// at most70 additional shapes for a given owner count; no payload/state is built
// beyond the caller's ordinary measuring path.
inline std::vector<std::vector<std::uint32_t>> ChunkMeasurementRows(std::uint32_t max_rows,
                                                                    std::uint32_t budget,
                                                                    std::uint32_t owners,
                                                                    bool shared_q8) {
  std::vector<std::vector<std::uint32_t>> cases;
  if (owners == 0 || owners > budget || max_rows == 0) return cases;
  const auto add = [&](const std::vector<std::uint32_t>& rows) {
    std::uint64_t total = 0;
    for (const auto n : rows) total += n;
    if (total <= budget && std::ranges::find(cases, rows) == cases.end()) cases.push_back(rows);
  };
  for (const auto n :
       {1U, std::min(max_rows, budget / owners), std::min(max_rows, budget - owners + 1)}) {
    std::vector<std::uint32_t> rows(owners, n);
    if (n == budget - owners + 1)
      for (std::size_t i = 1; i < rows.size(); ++i) rows[i] = 1;
    add(rows);
  }
  if (!shared_q8 || owners > 8) return cases;
  std::vector<std::uint32_t> rows(owners, 1);
  const auto compose = [&](auto&& self, std::uint32_t left, std::uint32_t owner) -> void {
    if (owner == owners) {
      if (left == 0) add(rows);
      return;
    }
    const auto peers = owners - owner - 1;
    for (std::uint32_t n = 1; n <= std::min(max_rows, left - peers); ++n) {
      rows[owner] = n;
      self(self, left - n, owner + 1);
    }
  };
  for (std::uint32_t total = owners; total <= std::min(8U, budget); ++total)
    compose(compose, total, 0);
  return cases;
}

// Every problem in one error, "; "-separated; success if there is none.
inline std::expected<void, std::string> Joined(std::span<const std::string> problems) {
  if (problems.empty()) {
    return {};
  }
  std::string all;
  for (const std::string& problem : problems) {
    all += (all.empty() ? "" : "; ") + problem;
  }
  return Error(all);
}

}  // namespace jitllm::engine::support

#endif  // JITLLM_ENGINE_SUPPORT_H_
