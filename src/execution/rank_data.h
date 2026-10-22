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
};
struct RankData {
  std::vector<DeviceMeasuredRank> records;
};
std::optional<std::uint32_t> DeviceRank(const CapabilityQuery& query,
                                        std::string_view implementation,
                                        std::span<const DeviceMeasuredRank> table);
base::Sha256Digest RankDataDigest(const RankData& data);
}  // namespace llmp::execution
#endif  // LLMP_EXECUTION_RANK_DATA_H_
