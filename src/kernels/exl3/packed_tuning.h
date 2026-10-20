// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#ifndef LLMP_KERNELS_EXL3_PACKED_TUNING_H_
#define LLMP_KERNELS_EXL3_PACKED_TUNING_H_
#include <array>
#include <cstdint>
#include <expected>
#include <string>

#include "kernels/exl3/validate.h"
namespace llmp::kernels::exl3 {
// Four authenticated upstream caches at the existing reference pins.
// These identify tuning data, not architecture graphs or runtime dispatchers.
enum class FrozenCache : std::uint8_t { k40G, k45G, k40O, k45O };
// Device occupancy facts, without model dimensions: output F16/F32, rate
// 4/5/6/8, and shapes1..4. Zero means no legal launch for that kernel.
struct PackedCapacity {
  std::uint32_t arch = 0;
  int sms = 0;
  std::array<std::array<std::array<int, kShapes>, 4>, 2> gemm{};
  std::array<std::array<std::array<int, kShapes>, 4>, 2> multi{};
  bool operator==(const PackedCapacity&) const = default;
};
// CUDA only: current-device occupancy, no launches or payload allocation.
std::expected<PackedCapacity, std::string> QueryPackedCapacity(int device);
// A conservative cooperative grid from occupancy, shape divisibility and
// the pinned kernel's integer split-K arithmetic. No measured-speed claim.
std::expected<GemmPlan, std::string> DefaultGemmPlan(const PackedCapacity& capacity, int rows,
                                                     int k, int n, int bits, Output output,
                                                     bool multi = false);
std::expected<MultiGemmPlan, std::string> DefaultMultiGemmPlan(const PackedCapacity& capacity,
                                                               int rows, int k, int n, int bits,
                                                               Output output);
std::expected<GemmPlan, std::string> FrozenGemmPlan(FrozenCache cache, std::uint32_t arch, int sms,
                                                    int rows, int k, int n, int bits,
                                                    Output output);
std::expected<MultiGemmPlan, std::string> FrozenMultiGemmPlan(FrozenCache cache, std::uint32_t arch,
                                                              int sms, int rows, int k, int n,
                                                              int bits, Output output);
}  // namespace llmp::kernels::exl3
#endif  // LLMP_KERNELS_EXL3_PACKED_TUNING_H_
