// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef LLMP_KERNELS_EXL3_POINTER_TABLE_H_
#define LLMP_KERNELS_EXL3_POINTER_TABLE_H_

#include <cstdint>
#include <expected>
#include <span>

#include "kernels/exl3/validate.h"

namespace llmp::kernels::exl3 {

std::expected<void, KernelFailure> LaunchWritePointerTable(
    void* stream, std::uint64_t dst_device_ptr, std::span<const std::uint64_t> words) noexcept;

}  // namespace llmp::kernels::exl3

#endif  // LLMP_KERNELS_EXL3_POINTER_TABLE_H_
