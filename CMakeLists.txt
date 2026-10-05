# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0

# jitLLM (D-010, D-011, D-057, D-058, D-059, D-060, D-062, D-066). Configure through
# a preset in CMakePresets.json, which selects a toolchain file in
# cmake/toolchains/; `mise run build` does that with the pinned SDK. Third-party
# sources come prepared from third_party/sources.lock.json (`mise run prepare`).
cmake_minimum_required(VERSION 4.4.3)
if(NOT CMAKE_VERSION VERSION_EQUAL "4.4.3")
  message(FATAL_ERROR "jitLLM builds with the SDK's CMake 4.4.3 (D-058), not ${CMAKE_VERSION}. "
                      "Use `mise run build`, or run $JITLLM_SDK/bin/cmake.")
endif()

# The next release's version (D-062). Builds report it with their distance
# from the last release tag (cmake/JitllmVersion.cmake); the first commit
# after a release raises it.
project(jitllm VERSION 0.1.0 LANGUAGES CXX)

if(NOT JITLLM_PROFILE)
  message(FATAL_ERROR "Configure with a preset from CMakePresets.json (`mise run build` does); "
                      "its toolchain file in cmake/toolchains/ selects the SDK's compilers.")
endif()
# A change to the SDK's inputs reruns configure, which then checks the SDK again.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${JITLLM_SDK_INPUT_FILES})
message(STATUS "jitLLM profile ${JITLLM_PROFILE}, SDK ${JITLLM_SDK_IDENTITY}, CUDA ${JITLLM_CUDA}")
include(cmake/JitllmVersion.cmake)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
# jitLLM uses no C++ modules; skip the dependency scan CMake adds for C++20
# and later (D-059).
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# Executables link their runtimes statically and need no RPATH (D-060), and
# nothing builds a shared library (D-064).
set(CMAKE_SKIP_RPATH ON)
set(BUILD_SHARED_LIBS OFF)

# No exceptions anywhere in the build, tests and third-party code included
# (D-066).
add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-fno-exceptions>")

# libstdc++'s precondition checks (D-083): `error()` on a std::expected that
# holds a value, `*` on an empty optional, an index out of range and the
# like end the process instead of reading whatever is there. The test and
# development presets set it, `spark-native` included; only `cross`, which
# builds the package, does not. Every target in the build gets it, libraries,
# tests and third-party code alike, and every language (NVCC passes it to
# both of its passes): an inline function compiled both with and without it
# would leave the linker to keep either copy.
option(JITLLM_LIBSTDCXX_ASSERTIONS "Build with _GLIBCXX_ASSERTIONS (D-083)" OFF)
if(JITLLM_LIBSTDCXX_ASSERTIONS)
  message(STATUS "libstdc++ assertions: on")
  add_compile_definitions(_GLIBCXX_ASSERTIONS)
endif()

if(JITLLM_CUDA)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 23)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_EXTENSIONS OFF)
  set(CMAKE_CUDA_RUNTIME_LIBRARY Static)
  add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fno-exceptions>")
endif()

# Sanitizer builds for D-061's check tiers (the `*-asan` and `*-tsan`
# presets): every target, tests and third-party code included, C (GGML's
# base sources) as well as C++ and CUDA, is instrumented. A UBSan report
# ends the process as an ASan one does; the test presets make
# ThreadSanitizer's do so too. NVCC splits -Xcompiler
# values at commas, so each sanitizer gets its own flag.
set(JITLLM_SANITIZE "" CACHE STRING "Sanitizers to build with (D-061): address;undefined, thread, or empty")
set(_jitllm_sanitizers address undefined thread)
foreach(_jitllm_sanitizer IN LISTS JITLLM_SANITIZE)
  if(NOT _jitllm_sanitizer IN_LIST _jitllm_sanitizers)
    message(FATAL_ERROR "JITLLM_SANITIZE names ${_jitllm_sanitizer}; it takes ${_jitllm_sanitizers}")
  endif()
endforeach()
if("thread" IN_LIST JITLLM_SANITIZE AND "address" IN_LIST JITLLM_SANITIZE)
  message(FATAL_ERROR "JITLLM_SANITIZE cannot combine thread with address")
endif()
foreach(_jitllm_sanitizer IN LISTS JITLLM_SANITIZE)
  add_compile_options("$<$<COMPILE_LANGUAGE:C,CXX>:-fsanitize=${_jitllm_sanitizer}>"
                      "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fsanitize=${_jitllm_sanitizer}>")
  add_link_options("-fsanitize=${_jitllm_sanitizer}")
endforeach()
if(JITLLM_SANITIZE)
  message(STATUS "Sanitizers: ${JITLLM_SANITIZE}")
  add_compile_options("$<$<COMPILE_LANGUAGE:C,CXX>:-fno-sanitize-recover=all;-fno-omit-frame-pointer>"
                      "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fno-sanitize-recover=all,-fno-omit-frame-pointer>")
endif()

# A build directory keeps the compilers and initial flags of its first
# configure, whatever the toolchain file now says. If JITLLM_SDK names another
# SDK than that one, the two would mix: require a fresh configure.
set(_jitllm_compilers CXX "${JITLLM_SDK}/bin/clang++")
if(JITLLM_CUDA)
  list(APPEND _jitllm_compilers CUDA "${JITLLM_SDK}/cuda/bin/nvcc")
endif()
while(_jitllm_compilers)
  list(POP_FRONT _jitllm_compilers _jitllm_lang _jitllm_want)
  cmake_path(NORMAL_PATH _jitllm_want)
  cmake_path(SET _jitllm_have NORMALIZE "${CMAKE_${_jitllm_lang}_COMPILER}")
  if(NOT _jitllm_have STREQUAL _jitllm_want)
    message(FATAL_ERROR
      "${PROJECT_BINARY_DIR} was first configured with ${_jitllm_have}, not the SDK at ${JITLLM_SDK}. "
      "Configure it afresh: `mise run build -- --fresh <preset>`, or add --fresh to `cmake --preset`.")
  endif()
endwhile()

# D-059's warnings, for jitLLM's own targets only (link this to each one);
# third-party code keeps its own warning settings. NVCC's host pass sees its
# generated line markers and CUDA's own headers as ordinary source, so it
# goes without -Wold-style-cast and allows GNU line markers, and allows the
# long string literal that holds the device code in NVCC's generated stub.
add_library(jitllm_warnings INTERFACE)
target_compile_options(jitllm_warnings INTERFACE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic;-Wshadow;-Wconversion;-Wsign-conversion;-Wnon-virtual-dtor;-Wold-style-cast;-Wimplicit-fallthrough;-Werror>"
  "$<$<COMPILE_LANGUAGE:CUDA>:-Werror=all-warnings;-Xcompiler=-Wall,-Wextra,-Wpedantic,-Wno-gnu-line-marker,-Wno-overlength-strings,-Wshadow,-Wconversion,-Wsign-conversion,-Wnon-virtual-dtor,-Wimplicit-fallthrough,-Werror>")

# Marks a test executable: in a cross build, CTest runs it under qemu-user,
# or on a Spark over SSH once deployed (tools/run-target, `mise run test`).
# Only tests get this runner; a build-time tool built for the target must
# never run under emulation (D-012).
function(jitllm_test_executable target)
  if(CMAKE_CROSSCOMPILING)
    set_property(TARGET ${target} PROPERTY CROSSCOMPILING_EMULATOR
      "${PROJECT_SOURCE_DIR}/tools/run-target;--build-dir;${PROJECT_BINARY_DIR};--sysroot;${CMAKE_SYSROOT};--qemu;qemu-${CMAKE_SYSTEM_PROCESSOR}-static;--")
  endif()
endfunction()

# Third-party sources: the profile's closure from the source lock, checked
# against it before any of their CMake code runs (D-057). After the flags
# above, so they build under the same toolchain and without exceptions.
include(cmake/JitllmSources.cmake)
jitllm_sources_add()

# A GoogleTest (with gMock) test executable (D-059), with the safe reading of
# a std::expected's error (tests/support/expected_error.h). CTest lists its tests
# when it runs them (PRE_TEST), so a cross build never runs target code at
# build time; ARGN goes to gtest_discover_tests().
include(GoogleTest)
function(jitllm_gtest target)
  target_link_libraries(${target} PRIVATE GTest::gmock_main jitllm_test_expected jitllm_warnings)
  jitllm_test_executable(${target})
  gtest_discover_tests(${target} DISCOVERY_MODE PRE_TEST ${ARGN})
endfunction()

# A GoogleTest executable that needs a GPU (label `gpu`): a GB10, and for
# the tests DISCRETE names (Suite.Name, or ALL) also the discrete GPUs a
# build targets (label `gpu-discrete` as well, where the build has any;
# D-082). It may load cuBLAS (D-076). Where GPU tests are skipped it may
# not even load (qemu-user has no libgcc_s for cuBLAS), so CTest never runs
# it to list its tests: they are read from its sources (gtest_add_tests).
# It finds the SDK's cuBLAS in build/<preset>/cublas through a RUNPATH
# relative to itself, which a deployed copy of the build tree keeps.
# ENVIRONMENT sets its tests' environment; MODELS marks model-file tests.
function(jitllm_gpu_gtest target)
  cmake_parse_arguments(PARSE_ARGV 1 arg "MODELS" "" "DISCRETE;ENVIRONMENT")
  if(arg_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR "jitllm_gpu_gtest(${target}): unexpected ${arg_UNPARSED_ARGUMENTS}")
  endif()
  target_link_libraries(${target} PRIVATE GTest::gmock_main jitllm_test_expected jitllm_warnings)
  jitllm_test_executable(${target})
  file(RELATIVE_PATH runtime "${CMAKE_CURRENT_BINARY_DIR}" "${PROJECT_BINARY_DIR}/cublas")
  target_link_options(${target} PRIVATE "LINKER:--enable-new-dtags,-rpath,$ORIGIN/${runtime}")
  gtest_add_tests(TARGET ${target} TEST_PREFIX unit. TEST_LIST tests)
  set(labels gpu)
  if(arg_MODELS)
    list(APPEND labels models)
    # Each process loads its own weights into the Spark's physical budget.
    # Serialize full-model GPU tests while other checks remain parallel.
    set_tests_properties(${tests} PROPERTIES RESOURCE_LOCK jitllm_gpu_model)
  endif()
  set_tests_properties(${tests} PROPERTIES LABELS "${labels}")
  if(arg_ENVIRONMENT)
    set_tests_properties(${tests} PROPERTIES ENVIRONMENT "${arg_ENVIRONMENT}")
  endif()
  set(discrete "")
  if(arg_DISCRETE STREQUAL "ALL")
    set(discrete ${tests})
  else()
    foreach(name IN LISTS arg_DISCRETE)
      if(NOT "unit.${name}" IN_LIST tests)
        message(FATAL_ERROR "jitllm_gpu_gtest(${target}): no test ${name}")
      endif()
      list(APPEND discrete "unit.${name}")
    endforeach()
  endif()
  if(discrete AND JITLLM_CUDA_DISCRETE_SASS)
    set_tests_properties(${discrete} PROPERTIES LABELS "${labels};gpu-discrete")
  endif()
endfunction()

add_subdirectory(src)
add_subdirectory(benchmarks)
add_subdirectory(packaging)

enable_testing()
add_subdirectory(tests/toolchain)
add_subdirectory(tests/sources)
add_subdirectory(tests/version)
add_subdirectory(tests/support)
add_subdirectory(tests/unit)
add_subdirectory(tests/smoke)
add_subdirectory(tests/jobs)

# Last: checks what the whole configure looked up, then writes the receipt.
jitllm_sources_finalize(VERSION "${JITLLM_BUILD_JSON}")
