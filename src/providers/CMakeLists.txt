# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0

# The provider interfaces and their implementations
# (docs/architecture.md#providers, D-026). The device probe:
# llmp_device_probe is the CUDA provider's in a CUDA build, and in the
# CPU-only profile one that reports no device provider.
if(LLMP_CUDA)
  add_subdirectory(cuda)
  add_library(llmp_device_probe ALIAS llmp_cuda_probe)
  # The device runtime (device_runtime.h): the build's one device backend
  # defines it. The CPU-only profile has none; its programs and tests that
  # run the engine core link the fake one (llmp_fake_device_runtime).
  add_library(llmp_device_runtime ALIAS llmp_cuda_device_runtime)
else()
  add_library(llmp_device_probe STATIC cpu_only_device_probe.cc)
  target_include_directories(llmp_device_probe PUBLIC "${PROJECT_SOURCE_DIR}/src")
  target_link_libraries(llmp_device_probe PUBLIC llmp_report PRIVATE llmp_warnings)
endif()

# The device-memory interface and the rules every implementation shares
# (device_memory.h).
add_library(llmp_device_memory STATIC device_memory.cc)
target_include_directories(llmp_device_memory PUBLIC "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(llmp_device_memory PUBLIC llmp_base PRIVATE llmp_warnings)

# The storage interface (storage.h), its io_uring implementation and whole
# reads over any implementation (direct_reader.h).
add_library(llmp_storage STATIC uring_storage.cc direct_reader.cc)
target_include_directories(llmp_storage PUBLIC "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(llmp_storage PUBLIC llmp_base llmp_platform PRIVATE llmp_warnings)

# The deterministic fakes every build has (fake/).
add_library(llmp_fake_providers STATIC fake/fake_device_memory.cc fake/fake_device_execution.cc
                                           fake/fake_storage.cc)
target_include_directories(llmp_fake_providers PUBLIC "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(llmp_fake_providers PUBLIC llmp_device_memory llmp_storage
                      PRIVATE llmp_platform llmp_warnings)

# The device runtime over the fakes (fake/fake_device_runtime.h), in every
# build: what a program links instead of a device backend's runtime to run
# the engine core on the host (the CPU-only profile's tests, and the same
# tests in a CUDA build). Never linked with llmp_device_runtime's CUDA one:
# both define device_runtime.h, and static archives would let the first one
# linked win silently, so each backend library names its backend
# (LLMP_DEVICE_BACKEND) and CMake refuses a program that links two.
add_library(llmp_fake_device_runtime STATIC fake/fake_device_runtime.cc)
target_include_directories(llmp_fake_device_runtime PUBLIC "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(llmp_fake_device_runtime PUBLIC llmp_fake_providers PRIVATE llmp_warnings)
set_target_properties(llmp_fake_device_runtime PROPERTIES INTERFACE_LLMP_DEVICE_BACKEND fake
                      COMPATIBLE_INTERFACE_STRING LLMP_DEVICE_BACKEND)
