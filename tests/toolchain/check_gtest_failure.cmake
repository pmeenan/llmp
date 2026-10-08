# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0

# Runs a GoogleTest executable's disabled negative control and checks that it
# fails the way a real failure would: exit status 1 and the failure reported.
# That holds natively, under qemu-user and over SSH to a Spark (D-059, D-061).
#
#   cmake -DBINARY=<test executable> [-DEMULATOR=<runner args joined by |>]
#         -P check_gtest_failure.cmake
cmake_minimum_required(VERSION 4.4.3)

if(NOT BINARY)
  message(FATAL_ERROR "set BINARY")
endif()
string(REPLACE "|" ";" command "${EMULATOR}")
list(APPEND command "${BINARY}" --gtest_also_run_disabled_tests "--gtest_filter=NegativeControl.*")
execute_process(COMMAND ${command} OUTPUT_VARIABLE output ERROR_VARIABLE error RESULT_VARIABLE result)
if(NOT result STREQUAL "1" OR NOT output MATCHES "\\[  FAILED  \\] NegativeControl\\.DISABLED_Fails")
  message(FATAL_ERROR "the negative control did not fail as a failing test must (exit ${result}):\n"
                      "${output}\n${error}")
endif()
message(STATUS "the negative control failed as expected (exit ${result})")
