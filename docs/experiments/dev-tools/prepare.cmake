# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
# Script-mode source preparation (D-057/D-058). Usage:
#   cmake -DARCHIVE=/abs/googletest-1.18.0.tar.gz -DSOURCE_DIR=/abs/new-dir -P prepare.cmake
cmake_minimum_required(VERSION 4.4.3)
if(NOT CMAKE_VERSION VERSION_EQUAL "4.4.3")
  message(FATAL_ERROR "This experiment validates exactly CMake 4.4.3")
endif()
foreach(var ARCHIVE SOURCE_DIR)
  if(NOT IS_ABSOLUTE "${${var}}")
    message(FATAL_ERROR "Set ${var} to an absolute path")
  endif()
endforeach()
if(EXISTS "${SOURCE_DIR}")
  message(FATAL_ERROR "SOURCE_DIR must not exist yet: ${SOURCE_DIR}")
endif()
# Keep FetchContent bookkeeping beside SOURCE_DIR, never in the caller's cwd.
set(FETCHCONTENT_BASE_DIR "${SOURCE_DIR}-work")
include(FetchContent)
FetchContent_Populate(googletest
  URL "file://${ARCHIVE}"
  URL_HASH SHA256=6e3191c1455468b3fc35a417fb565c1c5071aee1b7e7f85e30cf48a98d37d8b5
  SOURCE_DIR "${SOURCE_DIR}"
  BINARY_DIR "${SOURCE_DIR}-work/build"
  QUIET)
