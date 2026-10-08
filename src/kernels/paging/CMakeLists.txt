# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0

# Llmpalooza's own kernels that the engine's device jobs queue around paging
# (paging.h): filling device ranges, gathering Qwen3.8's n-gram rows,
# bounding draft ids and copying a verify's snapshot ranges. Not
# registry-bound operations (D-053). One implementation per device backend,
# chosen by what a program links: the CUDA kernels (CUDA builds), and the
# host loops for the fake device runtime (every build), never both.
if(LLMP_CUDA)
  add_library(llmp_paging_kernels STATIC clamp_tokens.cu copy_ranges.cu fill_ranges.cu
                                           gather_rows.cu)
  target_include_directories(llmp_paging_kernels PUBLIC "${PROJECT_SOURCE_DIR}/src")
  target_link_libraries(llmp_paging_kernels PRIVATE llmp_warnings)
  # sm_121a, as the GGML module's units build, and the profile's discrete
  # GPUs (D-082).
  set_target_properties(llmp_paging_kernels PROPERTIES
                        CUDA_ARCHITECTURES "121a-real;${LLMP_CUDA_DISCRETE_SASS}"
                        INTERFACE_LLMP_DEVICE_BACKEND cuda
                        COMPATIBLE_INTERFACE_STRING LLMP_DEVICE_BACKEND)
endif()

add_library(llmp_host_paging_kernels STATIC host_paging.cc)
target_include_directories(llmp_host_paging_kernels PUBLIC "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(llmp_host_paging_kernels PUBLIC llmp_fake_providers PRIVATE llmp_warnings)
# CMake refuses a program that links both (providers/CMakeLists.txt).
set_target_properties(llmp_host_paging_kernels PROPERTIES INTERFACE_LLMP_DEVICE_BACKEND fake
                      COMPATIBLE_INTERFACE_STRING LLMP_DEVICE_BACKEND)
