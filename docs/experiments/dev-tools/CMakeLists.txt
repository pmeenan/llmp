# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 4.4.3)
if(NOT CMAKE_VERSION VERSION_EQUAL "4.4.3")
  message(FATAL_ERROR "This experiment validates exactly CMake 4.4.3")
endif()
project(llmp_dev_tools_smoke LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_SKIP_RPATH ON)
# No C++ modules: skip the per-source dependency scan CMake adds for C++20+
# with Ninja. It also fails when a compiler wrapper hides clang-scan-deps.
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

set(GOOGLETEST_SOURCE_DIR "" CACHE PATH "Prepared GoogleTest 1.18.0 source (prepare.cmake)")
if(NOT EXISTS "${GOOGLETEST_SOURCE_DIR}/googletest/include/gtest/gtest.h")
  message(FATAL_ERROR "Prepare GoogleTest with prepare.cmake and set GOOGLETEST_SOURCE_DIR")
endif()
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
set(BUILD_GMOCK ON CACHE BOOL "" FORCE)
set(GTEST_HAS_ABSL OFF CACHE BOOL "" FORCE)
# Third-party headers are system headers for our warnings and clang-tidy.
add_subdirectory("${GOOGLETEST_SOURCE_DIR}" googletest EXCLUDE_FROM_ALL SYSTEM)

# Candidate warning set for llmpalooza-owned targets only.
set(LLMP_WARNINGS
    -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion
    -Wnon-virtual-dtor -Wold-style-cast -Wimplicit-fallthrough -Werror)

enable_testing()
include(GoogleTest)
add_executable(devtools_test devtools_test.cc)
target_compile_options(devtools_test PRIVATE ${LLMP_WARNINGS})
target_link_libraries(devtools_test PRIVATE GTest::gmock_main)
# PRE_TEST discovery lists tests when CTest runs, so a cross build never
# executes target binaries at build time.
gtest_discover_tests(devtools_test DISCOVERY_MODE PRE_TEST)

option(SMOKE_FAILING "Add a deliberately failing test (negative control)" OFF)
if(SMOKE_FAILING)
  add_executable(failing_test failing_test.cc)
  target_compile_options(failing_test PRIVATE ${LLMP_WARNINGS})
  target_link_libraries(failing_test PRIVATE GTest::gtest_main)
  gtest_discover_tests(failing_test DISCOVERY_MODE PRE_TEST)
endif()

option(SMOKE_ASAN_PROBE "Build the symbolizer probe (needs sanitizer flags)" OFF)
if(SMOKE_ASAN_PROBE)
  add_executable(asan_probe asan_probe.cc)
  target_compile_options(asan_probe PRIVATE ${LLMP_WARNINGS})
endif()
