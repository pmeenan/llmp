// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef LLMP_KERNELS_EXL3_INPUT_CONVERSION_H_
#define LLMP_KERNELS_EXL3_INPUT_CONVERSION_H_

#include <cstdint>
#include <expected>
#include <string_view>

#include "kernels/exl3/validate.h"

namespace llmp::kernels::exl3 {
// Reuses the pinned GGML F32->F16 RN converter. The caller supplies stream
// accounting and keeps both disjoint, bounded ranges alive through completion.
std::expected<void, KernelFailure> LaunchInputConversion(void* stream, std::uint64_t source,
                                                         std::uint64_t destination,
                                                         std::uint64_t elements) noexcept;
std::string_view InputConversionSourceIdentity();
}  // namespace llmp::kernels::exl3

#endif  // LLMP_KERNELS_EXL3_INPUT_CONVERSION_H_
