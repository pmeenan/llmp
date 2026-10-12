// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Fixed immutable trajectories shared by active benchmark adapters.

#ifndef LLMP_BENCHMARKS_FP16_COMMON_H_
#define LLMP_BENCHMARKS_FP16_COMMON_H_

#include <cstdint>
#include <expected>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace llmp::benchmarks {

struct Trajectory {
  std::string name;
  std::vector<std::int32_t> tokens;
  std::vector<std::uint32_t> chunks;
  std::uint32_t cells = 0;
  // The profile's restore point (backend-proof.md, numerical profiles):
  // the tokens evaluated before the restored evaluation's restore.
  std::uint32_t restore_after = 0;
};

// `control`: the bridge's tokens.txt (76 IDs, one per line), chunks 32
// then 44 × 1, 512 cells, restore after 32. `heldout`: the declared
// held-out IDs (1,040 little-endian int64, SHA-256 6dd8da89..., the first
// 577 used), chunks 16, 17, 16 × 1, 512, 16 × 1, 1,024 cells, restore
// after 33.
std::expected<Trajectory, std::string> LoadTrajectory(std::string_view name,
                                                      const std::filesystem::path& tokens);

inline std::uint64_t RoundUp(std::uint64_t bytes, std::uint64_t to) {
  return (bytes + to - 1) / to * to;
}

}  // namespace llmp::benchmarks

#endif  // LLMP_BENCHMARKS_FP16_COMMON_H_
