// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef LLMP_KERNELS_GGML_ROUNDED_INDEX_SCORE_H_
#define LLMP_KERNELS_GGML_ROUNDED_INDEX_SCORE_H_

#include <array>
#include <cstdint>
#include <limits>

namespace llmp::kernels::ggml {
// Logical native operand metadata. `offset` is the root-relative byte
// offset at selection; actual validation uses the bound address instead.
// Neither form proves backing or a root's base alignment by itself.
struct ScoreOperand {
  std::array<std::uint64_t, 4> ne{};
  std::array<std::uint64_t, 4> nb{};
  std::uint64_t offset = 0;
};

// The unchanged native score body uses signed int input indices. Check
// the complete addressed index, not just each pitch separately. Output
// and fill use int64 indices, bounded here to INT32_MAX elements.
inline bool RoundedIndexScoreFits(const std::array<ScoreOperand, 5>& a) {
  constexpr auto max = static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max());
  const auto& q = a[0];
  const auto& k = a[1];
  const auto& w = a[2];
  const auto& counts = a[3];
  const auto& output = a[4];
  const auto rows = q.ne[2];
  const auto cells = k.ne[2];
  if (rows == 0 || rows > std::uint64_t{65535} * 4 || cells == 0 || cells > max / 4 ||
      rows > max / cells || q.ne != std::array<std::uint64_t, 4>{128, 64, rows, 1} ||
      k.ne != std::array<std::uint64_t, 4>{128, 1, cells, 1} ||
      w.ne != std::array<std::uint64_t, 4>{64, rows, 1, 1} ||
      counts.ne != std::array<std::uint64_t, 4>{rows, 1, 1, 1} ||
      output.ne != std::array<std::uint64_t, 4>{cells, rows, 1, 1} || q.nb[0] != 4 ||
      q.nb[1] % 8 != 0 || q.nb[2] % 8 != 0 || q.offset % 8 != 0 || k.nb[0] != 2 ||
      k.nb[2] % 16 != 0 || k.offset % 16 != 0 || w.nb[0] != 4 || w.nb[1] % 4 != 0 ||
      w.offset % 4 != 0 || counts.nb[0] != 4 || counts.offset % 4 != 0 || output.nb[0] != 4 ||
      output.nb[1] != cells * 4 || output.offset % 4 != 0)
    return false;
  const auto fits = [](std::uint64_t width, std::uint64_t pitch, std::uint64_t count) {
    return width <= max && pitch <= max && (count - 1) <= (max - width) / (pitch ? pitch : 1);
  };
  // Zero pitches are legal repeated inputs and do not advance an index.
  const auto q_head = q.nb[1] / 4;
  const auto q_row = q.nb[2] / 4;
  if (q_head > (max - 127) / 63 || q_row > max ||
      (q_row != 0 && (rows - 1) > (max - 127 - 63 * q_head) / q_row))
    return false;
  return fits(127, k.nb[2] / 2, k.nb[2] == 0 ? 1 : cells) &&
         fits(63, w.nb[1] / 4, w.nb[1] == 0 ? 1 : rows);
}
}  // namespace llmp::kernels::ggml

#endif  // LLMP_KERNELS_GGML_ROUNDED_INDEX_SCORE_H_
