# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 4.4.3)
string(REPLACE "|" ";" emulator "${EMULATOR}")
execute_process(COMMAND ${emulator} "${BINARY}" "--gtest_filter=MainSetupFailure.${CASE}"
                OUTPUT_VARIABLE output ERROR_VARIABLE error RESULT_VARIABLE result)
if(result STREQUAL "0" OR NOT output MATCHES "test startup: cannot disable core dumps")
  message(FATAL_ERROR "startup failure was not propagated from the death-test child (${result}):\n${output}\n${error}")
endif()
