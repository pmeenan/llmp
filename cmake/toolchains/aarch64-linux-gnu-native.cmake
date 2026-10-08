# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0

# The native Spark fallback (D-032): built on a Spark with the aarch64 SDK's
# Clang and GCC 16.2 runtime, linked with the host's GNU binutils. A
# diagnostic profile; the cross build is the primary workflow (D-011). The
# `spark-native` preset uses this file.
set(LLMP_PROFILE spark-native)
set(LLMP_BUILD_ARCH aarch64)
set(LLMP_TARGET_TRIPLE aarch64-linux-gnu)
set(LLMP_TARGET_MARCH armv8-a)
set(LLMP_LINK_FLAGS -fuse-ld=bfd)
include("${CMAKE_CURRENT_LIST_DIR}/sdk.cmake")
