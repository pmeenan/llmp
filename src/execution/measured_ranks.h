// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#ifndef LLMP_EXECUTION_MEASURED_RANKS_H_
#define LLMP_EXECUTION_MEASURED_RANKS_H_

#include <array>
#include <optional>
#include <string_view>

#include "execution/registry.h"

namespace llmp::execution {
// Checked-in measurements, never architecture-derived tuning. Shape bounds are
// inclusive; unmeasured formats/devices/batches retain the ordinary ladder.
struct DeviceMeasuredRank {
  std::string_view device_class, device_model;
  std::uint32_t arch, multiprocessors;
  ir::OpKey op;
  format::Encoding encoding;
  format::Layout layout;
  std::uint64_t rows_min, rows_max, k, n;
  std::string_view implementation;
  std::uint32_t rank;
  format::EncodingParams params{};
  std::uint8_t stacked = 1;
};
// Filled only from the frozen protocol in experiments/decode-ladder-r6.md.
inline constexpr std::array<DeviceMeasuredRank, 2> kDeviceMeasuredRanks{{
    {.device_class = "sm_121",
     .device_model = "NVIDIA GB10",
     .arch = 121,
     .multiprocessors = 48,
     .op = ir::Primitive(ir::Op::kMatMul),
     .encoding = format::Encoding::kMxfp8,
     .layout = format::Layout::kSplit,
     .rows_min = 1,
     .rows_max = 1,
     .k = 2560,
     .n = 10240,
     .implementation = "llmp.mxfp8.mul_mat_vec",
     .rank = 0,
     .params = {}},
    {.device_class = "sm_121",
     .device_model = "NVIDIA GB10",
     .arch = 121,
     .multiprocessors = 48,
     .op = ir::Primitive(ir::Op::kGetRows),
     .encoding = format::Encoding::kNvfp4,
     .layout = format::Layout::kRowInterleaved,
     .rows_min = 16,
     .rows_max = 16,
     .k = 160,
     .n = 1048576,
     .implementation = "llmp.nvfp4.get_rows",
     .rank = 0,
     .params = {.tensor_scale = true}},
}};

std::optional<std::uint32_t> DeviceRank(const CapabilityQuery& query,
                                        std::string_view implementation);
const base::Sha256Digest& MeasuredRanksDigest();
}  // namespace llmp::execution
#endif  // LLMP_EXECUTION_MEASURED_RANKS_H_
