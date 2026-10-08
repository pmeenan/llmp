# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0

# Checks that a sanitizer build's sanitizers are active (D-061): the control
# runs clean without a defect, and each defect its sanitizers cover fails the
# process with that sanitizer's report. Natively, under qemu-user and over
# SSH to a Spark, where the test preset's *SAN_OPTIONS pass through. Leak
# detection is checked unless ASAN_OPTIONS turns it off (qemu-user, RE-014).
#
#   cmake -DBINARY=<sanitizer_control> -DSANITIZERS=<sanitizers joined by |>
#         [-DEMULATOR=<runner args joined by |>] -P check_sanitizer.cmake
cmake_minimum_required(VERSION 4.4.3)

if(NOT BINARY OR NOT SANITIZERS)
  message(FATAL_ERROR "set BINARY and SANITIZERS")
endif()
string(REPLACE "|" ";" sanitizers "${SANITIZERS}")
string(REPLACE "|" ";" emulator "${EMULATOR}")

# Pairs of a control mode and the report it must produce.
set(cases "")
if("address" IN_LIST sanitizers)
  list(APPEND cases heap-overflow "ERROR: AddressSanitizer: heap-buffer-overflow")
  if("$ENV{ASAN_OPTIONS}" MATCHES "(^|:)detect_leaks=0(:|$)")
    message(STATUS "leak detection is off here (ASAN_OPTIONS); the leak is not checked")
  else()
    list(APPEND cases leak "ERROR: LeakSanitizer: detected memory leaks")
  endif()
endif()
if("undefined" IN_LIST sanitizers)
  list(APPEND cases signed-overflow "runtime error: signed integer overflow")
endif()
if("thread" IN_LIST sanitizers)
  list(APPEND cases race "WARNING: ThreadSanitizer: data race")
endif()
if(NOT cases)
  message(FATAL_ERROR "no control covers the sanitizers ${SANITIZERS}")
endif()

execute_process(COMMAND ${emulator} "${BINARY}" clean
                OUTPUT_VARIABLE output ERROR_VARIABLE error RESULT_VARIABLE result)
if(NOT result STREQUAL "0")
  message(FATAL_ERROR "the control failed without a defect (exit ${result}):\n${output}\n${error}")
endif()

set(problems "")
while(cases)
  list(POP_FRONT cases mode report)
  execute_process(COMMAND ${emulator} "${BINARY}" ${mode}
                  OUTPUT_VARIABLE output ERROR_VARIABLE error RESULT_VARIABLE result)
  string(FIND "${error}" "${report}" at)
  if(result STREQUAL "0" OR at LESS 0)
    list(APPEND problems "${mode}: exit ${result}, expected a failure reporting \"${report}\":\n${error}")
  else()
    message(STATUS "${mode}: reported, exit ${result}")
  endif()
endwhile()
if(problems)
  list(JOIN problems "\n" text)
  message(FATAL_ERROR "the sanitizers missed a defect:\n${text}")
endif()
