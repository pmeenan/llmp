// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#ifndef LLMP_KERNELS_EXL3_PACKED_TUNING_H_
#define LLMP_KERNELS_EXL3_PACKED_TUNING_H_
#include <cstdint>
#include <expected>
#include <string>

#include "kernels/exl3/validate.h"
namespace llmp::kernels::exl3 {
// Four authenticated upstream caches at the existing reference pins.
// These identify tuning data, not architecture graphs or runtime dispatchers.
enum class FrozenCache : std::uint8_t { k40G, k45G, k40O, k45O };
std::expected<GemmPlan, std::string> FrozenGemmPlan(FrozenCache cache, std::uint32_t arch, int sms,
                                                    int rows, int k, int n, int bits,
                                                    Output output);
std::expected<MultiGemmPlan, std::string> FrozenMultiGemmPlan(FrozenCache cache, std::uint32_t arch,
                                                              int sms, int rows, int k, int n,
                                                              int bits, Output output);
}  // namespace llmp::kernels::exl3
#endif  // LLMP_KERNELS_EXL3_PACKED_TUNING_H_
