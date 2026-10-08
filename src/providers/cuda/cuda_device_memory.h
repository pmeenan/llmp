// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The CUDA device-memory provider (D-006, D-033, D-034): VmmProvider over
// the driver's virtual memory management API. It retains the device's
// primary context and makes it current for each call, so any lane thread
// may call it (one at a time: the VMM lane, or the device submission lane
// without one, owns it; device_memory.h, D-048).
//
// Its classes are the device's local memory and, where the device supports
// host-NUMA VMM, GPU-accessible host memory on its NUMA node (D-034); each
// has the driver's minimum granularity, and the provider's granularity is
// their least common multiple. Host backing given access is also opened to
// the CPU, at the same address.
//
// Driver errors map to ProviderError conservatively (cuda_errors.h): an
// outcome is known only for errors the driver reports before doing
// anything; any other is unknown, a fault, not a retry.
//
// This header holds no CUDA types (AGENTS.md rule 6).

#ifndef LLMP_PROVIDERS_CUDA_CUDA_DEVICE_MEMORY_H_
#define LLMP_PROVIDERS_CUDA_CUDA_DEVICE_MEMORY_H_

#include <expected>
#include <memory>

#include "providers/device_memory.h"

namespace llmp::providers::cuda {

// Opens device `ordinal` (initializing the driver). Fails if the driver or
// device is missing or the device has no VMM.
std::expected<std::unique_ptr<VmmProvider>, Failure> OpenDeviceMemory(int ordinal);

}  // namespace llmp::providers::cuda

#endif  // LLMP_PROVIDERS_CUDA_CUDA_DEVICE_MEMORY_H_
