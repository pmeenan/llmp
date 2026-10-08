// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef JITLLM_MODEL_HOST_MASK_H_
#define JITLLM_MODEL_HOST_MASK_H_

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>

namespace jitllm::model {

// Physical visible cells in ascending order; all other cells are blocked.
// A logical causal interval crosses a ring boundary at most once.
struct HostMaskIntervals {
  struct Interval {
    std::size_t begin = 0;
    std::size_t end = 0;
  };
  std::size_t width = 0;
  std::array<Interval, 2> visible{};
  std::size_t count = 0;
};

inline bool ValidHostMaskIntervals(const HostMaskIntervals& mask, std::size_t width) {
  if (mask.width != width || mask.count > mask.visible.size()) return false;
  std::size_t previous = 0;
  for (std::size_t i = 0; i < mask.count; ++i) {
    const auto& run = mask.visible[i];
    if (run.begin < previous || run.begin >= run.end || run.end > width) return false;
    previous = run.end;
  }
  return true;
}

inline std::optional<HostMaskIntervals> HostMaskPrefix(std::size_t width, std::uint64_t visible) {
  if (visible > width) return std::nullopt;
  HostMaskIntervals mask{.width = width};
  if (visible != 0) {
    mask.visible[0] = {0, static_cast<std::size_t>(visible)};
    mask.count = 1;
  }
  return mask;
}

// The whole chunk has written through end-1. Each physical cell holds its
// latest congruent position below end, even for an earlier query in that chunk.
inline std::optional<HostMaskIntervals> HostMaskRing(std::size_t width, std::uint64_t position,
                                                     std::uint64_t end, std::uint64_t capacity,
                                                     std::uint64_t window) {
  if (capacity == 0 || width > capacity || window == 0 || position >= end) return std::nullopt;
  const auto finish = position + 1;  // position < end proves no overflow.
  const auto start =
      std::max(end > capacity ? end - capacity : 0, finish > window ? finish - window : 0);
  HostMaskIntervals mask{.width = width};
  if (start >= finish) return mask;
  const auto first = start % capacity;
  const auto length = finish - start;  // At most capacity, by the end bound.
  const auto high_length = std::min(length, capacity - first);
  const auto add = [&](std::uint64_t begin, std::uint64_t stop) {
    begin = std::min<std::uint64_t>(begin, width);
    stop = std::min<std::uint64_t>(stop, width);
    if (begin < stop)
      mask.visible[mask.count++] = {static_cast<std::size_t>(begin),
                                    static_cast<std::size_t>(stop)};
  };
  add(0, length - high_length);
  add(first, first + high_length);
  return mask;
}

template <typename T>
inline bool FillHostMaskVisible(std::span<T> row, const HostMaskIntervals& mask, T value) {
  if (!ValidHostMaskIntervals(mask, row.size())) return false;
  for (std::size_t i = 0; i < mask.count; ++i) {
    const auto& run = mask.visible[i];
    std::fill(row.begin() + static_cast<std::ptrdiff_t>(run.begin),
              row.begin() + static_cast<std::ptrdiff_t>(run.end), value);
  }
  return true;
}

// Constant intervals allow a vectorized reduction, without a branch/early
// return for every public value. Every F16 bit is authenticated, including NaNs.
inline bool MatchHostMask(std::span<const std::uint16_t> row, const HostMaskIntervals& mask) {
  if (!ValidHostMaskIntervals(mask, row.size())) return false;
  std::uint32_t mismatch = 0;
  std::size_t cell = 0;
  for (std::size_t i = 0; i < mask.count; ++i) {
    const auto& run = mask.visible[i];
    for (; cell < run.begin; ++cell) mismatch |= std::uint32_t{row[cell]} ^ 0xFC00U;
    for (; cell < run.end; ++cell) mismatch |= row[cell];
  }
  for (; cell < row.size(); ++cell) mismatch |= std::uint32_t{row[cell]} ^ 0xFC00U;
  return mismatch == 0;
}

// QSA has three exact values. The first incomplete block is a deliberate
// finite dead sentinel, even though the following unwritten blocks are -inf.
inline bool FillHostQsaBias(std::span<float> row, std::size_t full, std::uint64_t visible) {
  if (full > row.size()) return false;
  const auto cut = static_cast<std::size_t>(std::min<std::uint64_t>(full, visible));
  std::fill_n(row.begin(), static_cast<std::ptrdiff_t>(cut), 0.0f);
  std::fill(row.begin() + static_cast<std::ptrdiff_t>(cut),
            row.begin() + static_cast<std::ptrdiff_t>(full), 1e9f);
  std::fill(row.begin() + static_cast<std::ptrdiff_t>(full), row.end(),
            -std::numeric_limits<float>::infinity());
  if (full < row.size()) row[full] = 1e9f;
  return true;
}

}  // namespace jitllm::model

#endif  // JITLLM_MODEL_HOST_MASK_H_
