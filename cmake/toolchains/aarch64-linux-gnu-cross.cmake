# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0

# AArch64 cross build for DGX Spark from an x86-64 host (D-011, D-032,
# D-070): the SDK's Clang and LLD against its Spark sysroot, which holds the
# cross-built GCC 16.2 runtime; CUDA for sm_121 from the sbsa-linux target
# tree. Tests run under qemu-user locally or on a Spark over SSH
# (tools/run-target). The `cross` preset uses this file.
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)
set(LLMP_PROFILE cross)
set(LLMP_BUILD_ARCH x86_64)
set(LLMP_TARGET_TRIPLE aarch64-linux-gnu)
set(LLMP_TARGET_MARCH armv8-a)
set(LLMP_LINK_FLAGS -fuse-ld=lld)
set(LLMP_SYSROOT sysroot/aarch64-linux-gnu)
include("${CMAKE_CURRENT_LIST_DIR}/sdk.cmake")
