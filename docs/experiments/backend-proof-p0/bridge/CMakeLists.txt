# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0

# The FP16 toolchain bridge (docs/backend-proof.md, oracle rung 2): the
# pinned llama.cpp source built with llmpalooza's SDK and linked with the same
# harness as the image reference (../fp16_reference.cc). bridge.sh configures
# it; it is an external reference build, not part of llmpalooza.
cmake_minimum_required(VERSION 3.28)
project(llmp_fp16_bridge C CXX CUDA)
set(LLAMA_SOURCE "" CACHE PATH "The pinned llama.cpp source tree")
add_subdirectory("${LLAMA_SOURCE}" llama.cpp EXCLUDE_FROM_ALL)
add_executable(fp16_reference "${CMAKE_CURRENT_LIST_DIR}/../fp16_reference.cc")
target_compile_features(fp16_reference PRIVATE cxx_std_23)
target_link_libraries(fp16_reference PRIVATE llama ggml)
