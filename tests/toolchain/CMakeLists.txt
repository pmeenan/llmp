# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0

# The toolchain contract, tested in what each profile produces (D-011, D-026,
# D-032, D-057, D-059, D-060, D-066, D-083). Every test carries the `toolchain` label; `gpu`
# marks those that need a GB10, which run only on a Spark, and `gpu-discrete`
# those that also run on a discrete GPU the build targets (D-082).

string(REGEX MATCH "^[^-]+" target_arch "${LLMP_TARGET_TRIPLE}")
# Spark binaries must run on the Sparks' glibc, which the cross build's
# sysroot pins (D-070). Native x86-64 binaries link the build host's own
# glibc, so they get no ceiling.
if(target_arch STREQUAL "aarch64")
  set(glibc_max 2.39)
else()
  set(glibc_max "")
endif()
if(LLMP_SANITIZE)
  set(sanitized ON)
else()
  set(sanitized OFF)
endif()

function(llmp_toolchain_test target)
  target_link_libraries(${target} PRIVATE llmp_warnings)
  llmp_test_executable(${target})
  add_test(NAME toolchain.${target} COMMAND ${target})
  add_test(NAME toolchain.${target}.binary
           COMMAND ${CMAKE_COMMAND} -DREADELF=${CMAKE_READELF} -DARCH=${target_arch}
                   -DGLIBC_MAX=${glibc_max} -DSANITIZED=${sanitized} -DBINARY=$<TARGET_FILE:${target}>
                   -P ${CMAKE_CURRENT_SOURCE_DIR}/check_binary.cmake)
  set_tests_properties(toolchain.${target} toolchain.${target}.binary PROPERTIES LABELS toolchain)
endfunction()

add_executable(toolchain_contract_test toolchain_contract_test.cc)
target_compile_definitions(toolchain_contract_test PRIVATE LLMP_TEST_TARGET_${target_arch}
                           "LLMP_TEST_LIBSTDCXX_ASSERTIONS=$<BOOL:${LLMP_LIBSTDCXX_ASSERTIONS}>")
llmp_toolchain_test(toolchain_contract_test)

# GoogleTest and gMock from the source lock, without exceptions (D-057,
# D-059, D-066), plus a negative control that must fail the run.
add_executable(gtest_contract_test gtest_contract_test.cc)
llmp_gtest(gtest_contract_test TEST_PREFIX toolchain.gtest. TEST_FILTER -NegativeControl.*
             PROPERTIES LABELS toolchain)
add_test(NAME toolchain.gtest_contract_test.binary
         COMMAND ${CMAKE_COMMAND} -DREADELF=${CMAKE_READELF} -DARCH=${target_arch}
                 -DGLIBC_MAX=${glibc_max} -DSANITIZED=${sanitized} -DBINARY=$<TARGET_FILE:gtest_contract_test>
                 -P ${CMAKE_CURRENT_SOURCE_DIR}/check_binary.cmake)
add_test(NAME toolchain.gtest_negative_control
         COMMAND ${CMAKE_COMMAND} -DBINARY=$<TARGET_FILE:gtest_contract_test>
                 "-DEMULATOR=$<JOIN:$<TARGET_PROPERTY:gtest_contract_test,CROSSCOMPILING_EMULATOR>,|>"
                 -P ${CMAKE_CURRENT_SOURCE_DIR}/check_gtest_failure.cmake)
set_tests_properties(toolchain.gtest_contract_test.binary toolchain.gtest_negative_control
                     PROPERTIES LABELS toolchain)

# Exercise compiler discovery with a legal SDK path that needs shell quoting.
# This configures another CPU-only tree but never runs its nested tests.
add_test(NAME toolchain.sdk_path
         COMMAND ${CMAKE_COMMAND} -DSOURCE_DIR=${PROJECT_SOURCE_DIR} -DSDK=${LLMP_SDK}
                 -DTOOLCHAIN=${CMAKE_TOOLCHAIN_FILE} -DTEST_DIR=${CMAKE_CURRENT_BINARY_DIR}/sdk-path
                 -P ${CMAKE_CURRENT_SOURCE_DIR}/check_sdk_path.cmake)
set_tests_properties(toolchain.sdk_path PROPERTIES LABELS toolchain)

# In a sanitizer build, each sanitizer must report its deliberate defect
# (D-061). The control builds everywhere, so clang-tidy sees it.
add_executable(sanitizer_control sanitizer_control.cc)
target_link_libraries(sanitizer_control PRIVATE llmp_warnings)
llmp_test_executable(sanitizer_control)
if(LLMP_SANITIZE)
  list(JOIN LLMP_SANITIZE "|" sanitizers)
  add_test(NAME toolchain.sanitizer_control
           COMMAND ${CMAKE_COMMAND} -DBINARY=$<TARGET_FILE:sanitizer_control> -DSANITIZERS=${sanitizers}
                   "-DEMULATOR=$<JOIN:$<TARGET_PROPERTY:sanitizer_control,CROSSCOMPILING_EMULATOR>,|>"
                   -P ${CMAKE_CURRENT_SOURCE_DIR}/check_sanitizer.cmake)
  set_tests_properties(toolchain.sanitizer_control PROPERTIES LABELS toolchain)
endif()

if(LLMP_CUDA)
  add_executable(cuda_contract_test cuda_contract_test.cu)
  # The compute capabilities it has SASS for, which it requires of device 0,
  # as "/121/86/" (NVCC splits a definition at commas).
  string(REPLACE ";" "/" sass "121;${LLMP_CUDA_DISCRETE_ARCHITECTURES}")
  target_compile_definitions(cuda_contract_test PRIVATE "LLMP_CUDA_SASS=\"/${sass}/\"")
  llmp_toolchain_test(cuda_contract_test)
  set_property(TEST toolchain.cuda_contract_test APPEND PROPERTY LABELS gpu)
  if(LLMP_CUDA_DISCRETE_SASS)
    set_property(TEST toolchain.cuda_contract_test APPEND PROPERTY LABELS gpu-discrete)
  endif()
else()
  add_test(NAME toolchain.cpu_only
           COMMAND ${CMAKE_COMMAND} -DBUILD_DIR=${PROJECT_BINARY_DIR}
                   -P ${CMAKE_CURRENT_SOURCE_DIR}/check_cpu_only.cmake)
  set_tests_properties(toolchain.cpu_only PROPERTIES LABELS toolchain)
endif()
