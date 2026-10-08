# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0

# Checks that a CPU-only build directory never saw a CUDA toolkit (D-026):
# no CUDA language, no NVCC and no path into the SDK's CUDA components in the
# cache, the generated build or the compile database.
#
#   cmake -DBUILD_DIR=<build directory> -P check_cpu_only.cmake
cmake_minimum_required(VERSION 4.4.3)

if(NOT IS_DIRECTORY "${BUILD_DIR}")
  message(FATAL_ERROR "set BUILD_DIR")
endif()
file(STRINGS "${BUILD_DIR}/CMakeCache.txt" sdk REGEX "^LLMP_SDK:PATH=")
string(REGEX REPLACE "^LLMP_SDK:PATH=" "" sdk "${sdk}")
if(NOT sdk)
  message(FATAL_ERROR "${BUILD_DIR}/CMakeCache.txt names no LLMP_SDK")
endif()

set(problems "")
foreach(file IN ITEMS CMakeCache.txt build.ninja compile_commands.json)
  file(READ "${BUILD_DIR}/${file}" text)
  foreach(needle IN ITEMS "${sdk}/cuda" "nvcc" "CMAKE_CUDA_COMPILER" "cudart")
    string(FIND "${text}" "${needle}" at)
    if(at GREATER_EQUAL 0)
      list(APPEND problems "${file} mentions ${needle}")
    endif()
  endforeach()
endforeach()
file(GLOB_RECURSE cuda_state "${BUILD_DIR}/CMakeFiles/*/CMakeCUDACompiler.cmake")
if(cuda_state OR EXISTS "${BUILD_DIR}/toolchain/cuda-host-clang++")
  list(APPEND problems "CMake enabled CUDA")
endif()

if(problems)
  list(JOIN problems "\n  " text)
  message(FATAL_ERROR "the CPU-only build saw CUDA:\n  ${text}")
endif()
message(STATUS "${BUILD_DIR}: no CUDA toolkit in the configuration")
