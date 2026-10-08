# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0

# Native x86-64 build on the workstation (D-011, D-032): the SDK's Clang
# and LLD, its GCC 16.2 runtime and the host's glibc (>= 2.39, D-070). CUDA
# code compiles for the GB10 (sm_121) and for the workstation's discrete GPU,
# an RTX 3080 Ti (sm_86, D-082), where its GPU tests can run. The `native`
# and `cpu` presets use this file.
set(LLMP_PROFILE native)
set(LLMP_BUILD_ARCH x86_64)
set(LLMP_TARGET_TRIPLE x86_64-linux-gnu)
set(LLMP_TARGET_MARCH x86-64)
set(LLMP_LINK_FLAGS -fuse-ld=lld)
if(NOT DEFINED LLMP_CUDA_DISCRETE_ARCHITECTURES)
  set(LLMP_CUDA_DISCRETE_ARCHITECTURES 86)
endif()
include("${CMAKE_CURRENT_LIST_DIR}/sdk.cmake")
