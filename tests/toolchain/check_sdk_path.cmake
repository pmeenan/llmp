# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0

# A relocated SDK may live in a path containing spaces. Compiler discovery
# must still compile and link with its GCC runtime, including in try_compile.
cmake_minimum_required(VERSION 4.4.3)

file(MAKE_DIRECTORY "${TEST_DIR}")
file(CREATE_LINK "${SDK}" "${TEST_DIR}/sdk with spaces" SYMBOLIC RESULT result)
if(NOT result STREQUAL "0")
  message(FATAL_ERROR "cannot create the SDK path fixture: ${result}")
endif()
execute_process(
  COMMAND "${CMAKE_COMMAND}" -S "${SOURCE_DIR}" -B "${TEST_DIR}/build" -G Ninja --fresh
          "-DCMAKE_TOOLCHAIN_FILE=${TOOLCHAIN}" "-DLLMP_SDK=${TEST_DIR}/sdk with spaces"
          -DLLMP_CUDA=OFF
  OUTPUT_VARIABLE output ERROR_VARIABLE error RESULT_VARIABLE result)
if(NOT result STREQUAL "0")
  message(FATAL_ERROR "configuration with an SDK path containing spaces failed:\n${output}\n${error}")
endif()
file(REMOVE_RECURSE "${TEST_DIR}/build")
file(REMOVE "${TEST_DIR}/sdk with spaces")
