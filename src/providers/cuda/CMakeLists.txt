# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0

# The CUDA provider (D-006, D-026): the device probe, the device-memory and
# device-execution providers and the device runtime. Outside the kernel
# modules, only this module includes CUDA's headers (docs/portability.md,
# checked by tools/llmp_boundaries.py). It uses the driver API from
# ordinary C++ units (the runtime API in the device runtime only) and links
# the NVIDIA driver's libcuda.so.1, which every host running a CUDA build
# must have (D-072).

# The driver, linked through NVIDIA's stub in the SDK: binaries record
# libcuda.so.1 as needed and the driver supplies it at run time.
set(stub "${LLMP_SDK}/cuda/targets/${LLMP_CUDA_TARGET_DIR}/lib/stubs/libcuda.so")
if(NOT EXISTS "${stub}")
  message(FATAL_ERROR "The SDK lacks the CUDA driver stub ${stub}; run `mise run setup`.")
endif()
add_library(llmp_cuda_driver SHARED IMPORTED)
set_target_properties(llmp_cuda_driver PROPERTIES IMPORTED_LOCATION "${stub}"
                                                    IMPORTED_SONAME libcuda.so.1)
# Tests that need no GPU load the stub on hosts without the driver (qemu-user,
# the reference container); the test presets put this directory on the
# library path. The stub's cuInit() fails, so doctor reports it.
file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/cuda-stub")
file(COPY_FILE "${stub}" "${PROJECT_BINARY_DIR}/cuda-stub/libcuda.so.1" ONLY_IF_DIFFERENT)

# What the probe judges, apart from the driver (cuda_facts.h).
add_library(llmp_cuda_facts STATIC cuda_report.cc)
target_include_directories(llmp_cuda_facts PUBLIC "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(llmp_cuda_facts PUBLIC llmp_platform llmp_report PRIVATE llmp_warnings)

add_library(llmp_cuda_probe STATIC cuda_probe.cc)
target_include_directories(llmp_cuda_probe SYSTEM PRIVATE
                           "${LLMP_SDK}/cuda/targets/${LLMP_CUDA_TARGET_DIR}/include")
list(JOIN CMAKE_CUDA_ARCHITECTURES "," architectures)
target_compile_definitions(llmp_cuda_probe PRIVATE "LLMP_CUDA_ARCHITECTURES=\"${architectures}\"")
target_link_libraries(llmp_cuda_probe PUBLIC llmp_cuda_facts llmp_cuda_driver
                      PRIVATE llmp_platform llmp_warnings)

# The device-memory and device-execution providers over the driver API
# (cuda_device_memory.h, cuda_device_execution.h), sharing their error
# mapping (cuda_errors.h).
add_library(llmp_cuda_device_memory STATIC cuda_device_memory.cc cuda_device_execution.cc cuda_errors.cc)
target_include_directories(llmp_cuda_device_memory SYSTEM PRIVATE
                           "${LLMP_SDK}/cuda/targets/${LLMP_CUDA_TARGET_DIR}/include")
target_link_libraries(llmp_cuda_device_memory PUBLIC llmp_device_memory llmp_cuda_driver
                      PRIVATE llmp_warnings)

# The device runtime (device_runtime.h) over the CUDA runtime API: opening
# the device, and the engine's copies, fills, timing marks, recorded work
# (CUDA graphs), pinned host memory and device facts. The runtime's static
# library comes with the CUDA link of every program that uses it (the
# engine and its kernels, CMAKE_CUDA_RUNTIME_LIBRARY).
add_library(llmp_cuda_device_runtime STATIC cuda_device_runtime.cc)
target_include_directories(llmp_cuda_device_runtime SYSTEM PRIVATE
                           "${LLMP_SDK}/cuda/targets/${LLMP_CUDA_TARGET_DIR}/include")
target_link_libraries(llmp_cuda_device_runtime PUBLIC llmp_cuda_device_memory
                      PRIVATE llmp_warnings)
