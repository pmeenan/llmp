// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef LLMP_RUNTIME_SCORE_ROWS_H_
#define LLMP_RUNTIME_SCORE_ROWS_H_

#include <cstdint>
#include <span>

#include "base/check.h"
#include "engine/host_buffer.h"

namespace llmp::runtime {
// The paid immutable result of one retired scoring batch. State/history may
// advance farther than callback publication. Copies retain the same allocation;
// cancellation and same-process spill/resume never drop undelivered rows.
struct ScoreRows {
  engine::HostBuffer bytes;
  std::uint32_t first = 0, rows = 0, width = 0, next = 0;
  // Last supplied-token position whose score callback was invoked (0: none).
  std::uint32_t published = 0;
  bool valid() const {
    return rows != 0 && width != 0 && next <= rows && bytes.size() % sizeof(float) == 0 &&
           bytes.size() / sizeof(float) / rows == width && bytes.size() / sizeof(float) % rows == 0;
  }
  std::span<const float> row(std::uint32_t index) const {
    base::Check(valid() && index < rows, "score row belongs to its immutable paid batch");
    return {reinterpret_cast<const float*>(bytes.data()) + std::size_t{index} * width, width};
  }
  void ClearRows() {
    const auto frontier = published;
    *this = {};
    published = frontier;
  }
};
}  // namespace llmp::runtime

#endif  // LLMP_RUNTIME_SCORE_ROWS_H_
