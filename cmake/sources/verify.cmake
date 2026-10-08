# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0

# Build-time check of one component's source tree against the digest
# configure accepted (D-057): an edit after configure fails the build rather
# than being compiled in. cmake/LlmpSources.cmake runs it before each
# component's targets build.
#
#   cmake -DDIR=<tree> -DEXPECT=<digest> -DSTAMP=<file> -P verify.cmake
cmake_minimum_required(VERSION 4.4.3)
include("${CMAKE_CURRENT_LIST_DIR}/TreeDigest.cmake")

foreach(var IN ITEMS DIR EXPECT STAMP)
  if(NOT DEFINED ${var} OR "${${var}}" STREQUAL "")
    message(FATAL_ERROR "set ${var}")
  endif()
endforeach()
file(REMOVE "${STAMP}")
llmp_tree_digest(digest "${DIR}")
if(NOT digest STREQUAL EXPECT)
  message(FATAL_ERROR
    "${DIR} changed after configure (tree ${digest}, configured ${EXPECT}). Remove it and run "
    "`mise run prepare`, then build again; for deliberate local edits use LLMP_SOURCE_OVERRIDE_<ID> (D-057).")
endif()
file(TOUCH "${STAMP}")
