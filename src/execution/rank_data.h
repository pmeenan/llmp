// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#ifndef LLMP_EXECUTION_RANK_DATA_H_
#define LLMP_EXECUTION_RANK_DATA_H_

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "base/sha256.h"
#include "format/encoding.h"
#include "format/layout.h"
#include "ir/op.h"

namespace llmp::execution {
struct CapabilityQuery;
// Checked-in measurements supplied by a backend. Every string is owned so
// registry construction can consume temporary or assembled rank data safely.
// Bounds are inclusive; zero K/N and an empty model are explicit wildcards.
// Unmeasured formats/devices/batches retain the ordinary ladder.
struct DeviceMeasuredRank {
  std::string device_class, device_model;
  std::uint32_t arch = 0, multiprocessors = 0;
  ir::OpKey op{};
  format::Encoding encoding = format::Encoding::kF32;
  format::Layout layout = format::Layout::kDense;
  std::uint64_t rows_min = 1, rows_max = 1, k = 0, n = 0;
  std::string implementation;
  std::uint32_t rank = 0;
  format::EncodingParams params{};
  std::uint8_t stacked = 1;
  // The implementation's kernel version (Implementation::version) when
  // measured; another version leaves the row inert.
  std::uint32_t version = 1;
};
// A fused implementation measured against the planner's own decomposition of
// the same node (benchmarks/fusion_ranks.cc: one graph planned with and
// without `disable <pattern>`, lowered normally, both timed), as a coarse
// class that survives unrelated code changes: the device class, the fused
// operation, the implementation and the kernel version it was measured at
// (Implementation::version), operand 0's encoding and layout, and inclusive
// bands of the query's rows and width (its k: a row's elements). A row whose
// implementation has another version is stale and decides nothing; a lookup
// never extrapolates past its bands. Timings are medians in nanoseconds.
struct FusionRank {
  std::string device_class;
  ir::OpKey op{};
  std::string implementation;
  std::uint32_t version = 1;
  format::Encoding encoding = format::Encoding::kF32;
  format::Layout layout = format::Layout::kDense;
  std::uint64_t rows_min = 1, rows_max = ~std::uint64_t{0};
  std::uint64_t width_min = 1, width_max = ~std::uint64_t{0};
  std::uint64_t fused_ns = 0, decomposition_ns = 0;
  std::uint32_t samples = 0;
  std::string provenance;  // the checked-in aggregate report
};

struct RankData {
  std::vector<DeviceMeasuredRank> records{};
  std::vector<FusionRank> fusions{};
};
// `version`: the named implementation's kernel version in this build.
std::optional<std::uint32_t> DeviceRank(const CapabilityQuery& query,
                                        std::string_view implementation,
                                        std::span<const DeviceMeasuredRank> table,
                                        std::uint32_t version = 1);
base::Sha256Digest RankDataDigest(const RankData& data);
}  // namespace llmp::execution
#endif  // LLMP_EXECUTION_RANK_DATA_H_
