# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0

# Checks that `llmp --version` reports what the build receipt records, and
# that both describe the checkout as it is now (D-062): the build derived the
# version again rather than keeping configure's. Natively, under qemu-user
# and over SSH to a Spark.
#
#   cmake -DBINARY=<llmp> [-DEMULATOR=<runner args joined by |>]
#         -DRECEIPT=<llmp-receipt.json> -DSOURCE_DIR=<checkout>
#         -DPROJECT_VERSION=<X.Y.Z> [-DGIT=<git>] -P check_version.cmake
cmake_minimum_required(VERSION 4.4.3)

foreach(var IN ITEMS BINARY RECEIPT SOURCE_DIR PROJECT_VERSION)
  if(NOT ${var})
    message(FATAL_ERROR "set ${var}")
  endif()
endforeach()
include("${CMAKE_CURRENT_LIST_DIR}/../../cmake/version/derive.cmake")

string(REPLACE "|" ";" command "${EMULATOR}")
list(APPEND command "${BINARY}" --version)
execute_process(COMMAND ${command} OUTPUT_VARIABLE output ERROR_VARIABLE error RESULT_VARIABLE result)
if(NOT result STREQUAL "0" OR NOT error STREQUAL "")
  message(FATAL_ERROR "llmp --version exited ${result}:\n${output}\n${error}")
endif()

file(READ "${RECEIPT}" receipt)
function(get out)
  string(JSON value ERROR_VARIABLE error GET "${receipt}" ${ARGN})
  if(NOT error STREQUAL "NOTFOUND")
    message(FATAL_ERROR "${RECEIPT}: ${error}")
  endif()
  set(${out} "${value}" PARENT_SCOPE)
endfunction()
get(product version product)
get(debian version debian)
get(commit version commit)
get(modified version modified)
get(origin version origin)
get(license_profile license_profile)
get(sdk sdk)
get(target target)

set(problems "")
if(commit STREQUAL "null" OR commit STREQUAL "")
  set(commit_line "unknown")
else()
  set(commit_line "${commit}")
endif()
if(modified)
  string(APPEND commit_line " (with uncommitted changes)")
endif()
string(JOIN "\n" expected
  "llmp ${product}" "commit: ${commit_line}" "license profile: ${license_profile}" "SDK: ${sdk}"
  "target: ${target}" "")
if(NOT output STREQUAL expected)
  list(APPEND problems "llmp --version printed\n${output}instead of what the receipt records:\n${expected}")
endif()
string(REPLACE "-" "~" want_debian "${product}")
if(NOT debian STREQUAL "${want_debian}-1")
  list(APPEND problems "the receipt's Debian version is ${debian}, not ${want_debian}-1")
endif()
string(REPLACE "." "\\." version_pattern "${PROJECT_VERSION}")
if(NOT product MATCHES "^${version_pattern}($|-)")
  list(APPEND problems "the receipt's version ${product} is not project(VERSION) ${PROJECT_VERSION}'s")
endif()

# The checkout now: a commit or an edit since the build means a stale binary.
llmp_version_derive(now SOURCE_DIR "${SOURCE_DIR}" PROJECT_VERSION "${PROJECT_VERSION}" GIT "${GIT}")
llmp_version_json(now_json now)
string(JSON recorded GET "${receipt}" version)
string(JSON same EQUAL "${recorded}" "${now_json}")
if(NOT same)
  list(APPEND problems "the build records ${recorded}, but the checkout is now ${now_json}; rebuild")
endif()

if(problems)
  list(JOIN problems "\n" problems)
  message(FATAL_ERROR "${problems}")
endif()
message(STATUS "llmp --version: ${product} (${origin})")
