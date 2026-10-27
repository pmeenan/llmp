// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The fake kernel set (device_backend.h) for host programs and tests, in
// every profile: the guardrail that the engine's and runtime's backend path
// needs no kernel library. Its backend has no implementations, so every plan
// over it is refused, binds a workspace of its own declared size, and
// launches over the device runtime's recorded work (recording_launch.h).
#ifndef LLMP_KERNELS_COMMON_FAKE_DEVICE_BACKEND_H_
#define LLMP_KERNELS_COMMON_FAKE_DEVICE_BACKEND_H_

#include <cstdint>

#include "kernels/common/device_backend.h"

namespace llmp::kernels {

// The workspace the fake backend asks to have bound (library_workspace_bytes).
inline constexpr std::uint64_t kFakeLibraryWorkspaceBytes = 1ULL << 20U;

const KernelSet& FakeKernelSet();

}  // namespace llmp::kernels

#endif  // LLMP_KERNELS_COMMON_FAKE_DEVICE_BACKEND_H_
