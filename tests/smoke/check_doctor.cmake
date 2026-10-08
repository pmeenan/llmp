# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0

# Runs `llmp doctor` and checks its report (D-026).
#
#   MODE=report  any host: a report on standard output, nothing on standard
#                error, and exit 0 exactly when the summary says "no
#                problems" (1 otherwise).
#   MODE=cpu     the CPU-only profile: exit 0 and no device provider, since
#                the host probe only ever warns.
#   MODE=gb10    a Spark: exit 0 with no GPU warning, and GPU 0 a GB10
#                (sm_121), unified, with CUDA VMM, its memory, and
#                granularities for device-local and host-NUMA backing
#                (D-033, D-034).
#   MODE=discrete  a host with a discrete GPU the build targets (D-082):
#                the same, for a discrete GPU 0 of any targeted architecture
#                but sm_121.
#
#   cmake -DBINARY=<llmp> [-DEMULATOR=<runner args joined by |>] -DMODE=<mode>
#         -P check_doctor.cmake
cmake_minimum_required(VERSION 4.4.3)

if(NOT BINARY OR NOT MODE MATCHES "^(report|cpu|gb10|discrete)$")
  message(FATAL_ERROR "set BINARY, and MODE to report, cpu, gb10 or discrete")
endif()
string(REPLACE "|" ";" command "${EMULATOR}")
list(APPEND command "${BINARY}" doctor)
execute_process(COMMAND ${command} OUTPUT_VARIABLE output ERROR_VARIABLE error RESULT_VARIABLE result)
message(STATUS "llmp doctor exited ${result}:\n${output}")

set(problems "")
if(NOT error STREQUAL "")
  list(APPEND problems "it wrote to standard error:\n${error}")
endif()
if(NOT output MATCHES "^build\n  version: ")
  list(APPEND problems "the report does not start with the build section")
endif()
if(output MATCHES "\ndoctor: no problems, [0-9]+ warnings?\n$")
  set(want 0)
elseif(output MATCHES "\ndoctor: [1-9][0-9]* problems?, [0-9]+ warnings?\n$")
  set(want 1)
else()
  set(want "")
  list(APPEND problems "the report does not end with its summary line")
endif()
if(want STREQUAL "" OR NOT result STREQUAL want)
  list(APPEND problems "exit status ${result} does not match the summary")
endif()
if(MODE STREQUAL "cpu")
  if(NOT result STREQUAL "0")
    list(APPEND problems "a CPU-only build has no device problems to report")
  endif()
  if(NOT output MATCHES "\ndevices\n  provider: none ")
    list(APPEND problems "no 'provider: none' line")
  endif()
elseif(MODE MATCHES "^(gb10|discrete)$")
  if(NOT result STREQUAL "0")
    list(APPEND problems "a targeted GPU's host must have no problems")
  endif()
  if(output MATCHES "\nwarning: GPU")
    list(APPEND problems "a targeted GPU's host must have no GPU warnings")
  endif()
  # Patterns for each line GPU 0's section must have.
  set(granularity "granularity [0-9]+ [KMG]iB minimum, [0-9]+ [KMG]iB recommended")
  if(MODE STREQUAL "gb10")
    set(device "\nGPU 0: NVIDIA GB10\n" "\n  compute capability: 12\\.1 \\(sm_121\\)\n"
               "\n  class: unified [^\n]*\n")
  else()
    # Any targeted architecture but the GB10's: no "has no code for".
    set(device "\nGPU 0: [^\n]+\n" "\n  compute capability: [0-9]+\\.[0-9] \\(sm_[0-9]+\\)\n"
               "\n  class: discrete [^\n]*\n")
    if(output MATCHES "\n  compute capability: 12\\.1 ")
      list(APPEND problems "GPU 0 is a GB10, not a discrete GPU")
    endif()
  endif()
  foreach(pattern IN LISTS device ITEMS "\n  memory: [0-9.]+ GiB\n"
                          "\n  virtual memory management: supported\n"
                          "\n  device-local backing: ${granularity}\n"
                          "\n  host NUMA node [0-9]+ backing: ${granularity}\n")
    if(NOT output MATCHES "${pattern}")
      string(STRIP "${pattern}" pattern)
      list(APPEND problems "no line matching '${pattern}'")
    endif()
  endforeach()
endif()

if(problems)
  list(JOIN problems "\n" text)
  message(FATAL_ERROR "${text}")
endif()
