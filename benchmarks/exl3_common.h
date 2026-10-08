// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// What the backend proof's native EXL3 model harnesses share
// (docs/backend-proof.md, P3): rung 3's cudaMalloc run (exl3_exec.cc) and
// rungs 4 and 5's paged run (exl3_paged.cc). Every profile builds this;
// nothing here launches.

#ifndef LLMP_BENCHMARKS_EXL3_COMMON_H_
#define LLMP_BENCHMARKS_EXL3_COMMON_H_

#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "model/qwen2_exl3.h"

namespace llmp::benchmarks {

// The trajectories' single-token steps after each prefix, and the
// reference's cache (exl3_heldout.py SUFFIX and CAPACITY).
inline constexpr int kSuffix = 16;
inline constexpr int kCells = 4096;

std::string Hex(std::span<const std::byte> bytes);
std::string HexFile(const std::filesystem::path& path);

// A .npy file (format 1.0) of little-endian values of `descr` ("<f4",
// "<f2").
std::expected<void, std::string> WriteNpy(const std::filesystem::path& path, std::string_view descr,
                                          const std::vector<std::int64_t>& shape,
                                          std::span<const std::byte> data);

// The declared held-out IDs: 1,040 little-endian int64, SHA-256
// 6dd8da89...
std::expected<std::vector<std::int32_t>, std::string> LoadIds(const std::filesystem::path& path);

// The launch table of a plan file (model_plan.py): `linear NAME K N BITS
// OUT BIAS` lines, each of which must be the artifact's linear, `case ID
// ROWS PATH ...` lines (the per-linear sweep's format), and `#` comments,
// of which `# cache SHA256` names the frozen tuning cache. The table's
// source, part of every plan identity, is that cache and the file's
// SHA-256.
std::expected<model::Exl3LaunchTable, std::string> LoadTable(const std::filesystem::path& path,
                                                             const model::Exl3Binding& binding);

// BF16 to F32, exactly.
float Bf16ToFloat(std::uint16_t value);

}  // namespace llmp::benchmarks

#endif  // LLMP_BENCHMARKS_EXL3_COMMON_H_
