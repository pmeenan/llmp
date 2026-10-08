// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The CUDA device-execution provider: DeviceExecution over the driver API.
// Streams are non-blocking (they never synchronize with the legacy default
// stream); copies are asynchronous copies between VMM addresses; fences
// are timing-free events, queried with cuEventQuery. Like the device-memory
// provider it retains the device's primary context and makes it current on
// each call. Errors map as the device-memory provider's do (cuda_errors.h). This header holds no
// CUDA types.
//
// Fences' events come from a pool, made when the provider opens and kept
// when a fence is released: on the GB10 (driver 580.178.04) cuEventCreate
// blocks for as long as another thread is blocked launching into a full
// stream (RE-029), so a lane that fences its work, such as the landing
// zone's copy lane, must not create events as it goes.

#ifndef LLMP_PROVIDERS_CUDA_CUDA_DEVICE_EXECUTION_H_
#define LLMP_PROVIDERS_CUDA_CUDA_DEVICE_EXECUTION_H_

#include <cstddef>
#include <expected>
#include <memory>

#include "providers/device_execution.h"

namespace llmp::providers::cuda {

struct ExecutionSettings {
  std::size_t events_ahead = 256;  // made when the provider opens
  std::size_t events_kept = 4096;  // at most kept for reuse; beyond, released fences destroy theirs
};

std::expected<std::unique_ptr<DeviceExecution>, Failure> OpenDeviceExecution(
    int ordinal, ExecutionSettings settings = {});

}  // namespace llmp::providers::cuda

#endif  // LLMP_PROVIDERS_CUDA_CUDA_DEVICE_EXECUTION_H_
