# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0

# Reference-only GGML operation harness (../README.md, ggml-ops.json): the
# pinned GGML subtree (llama.cpp b29c606e2, ggml/, unmodified) built as shared
# libraries with its own CMake, plus ggml_shim.cu, a C API that runs one GGML
# CUDA operation on device buffers owned by PyTorch. build.sh configures it
# inside the EXL3 reference container. It is an external measurement tool,
# not part of llmpalooza.
cmake_minimum_required(VERSION 3.28)
project(llmp_ggml_shim C CXX CUDA)
set(GGML_SOURCE "" CACHE PATH "The pinned llama.cpp source tree's ggml/ directory")
set(GGML_SHIM_SOURCE_ID "" CACHE STRING "Identity of the GGML source, recorded in the library")
add_subdirectory("${GGML_SOURCE}" ggml)

# The shim reads ggml-cuda's context struct, so it compiles with exactly the
# definitions ggml-cuda compiles with (directory-level and target-level).
get_directory_property(cuda_dir_defs DIRECTORY "${GGML_SOURCE}/src/ggml-cuda" COMPILE_DEFINITIONS)
get_target_property(cuda_target_defs ggml-cuda COMPILE_DEFINITIONS)
if(NOT cuda_target_defs)
  set(cuda_target_defs "")
endif()
get_directory_property(cuda_archs DIRECTORY "${GGML_SOURCE}/src/ggml-cuda" DEFINITION CMAKE_CUDA_ARCHITECTURES)
message(STATUS "ggml-cuda definitions: ${cuda_dir_defs};${cuda_target_defs}")
message(STATUS "ggml-cuda architectures: ${cuda_archs}")

find_package(CUDAToolkit REQUIRED)
add_library(ggml_shim SHARED "${CMAKE_CURRENT_LIST_DIR}/ggml_shim.cu")
set_target_properties(ggml_shim PROPERTIES CUDA_ARCHITECTURES "${cuda_archs}" CUDA_STANDARD 17)
target_compile_definitions(ggml_shim PRIVATE ${cuda_dir_defs} ${cuda_target_defs}
  GGML_SHIM_SOURCE_ID="${GGML_SHIM_SOURCE_ID}" GGML_SHIM_ARCHS="${cuda_archs}")
target_include_directories(ggml_shim PRIVATE "${GGML_SOURCE}/include" "${GGML_SOURCE}/src" "${GGML_SOURCE}/src/ggml-cuda")
target_link_libraries(ggml_shim PRIVATE ggml ggml-base ggml-cuda CUDA::cudart CUDA::cublas)
