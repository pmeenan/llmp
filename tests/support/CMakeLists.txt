# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0

# Support for tests and benchmarks: reading a std::expected's error safely,
# and recording an executed plan to compare with a recorded one
# (docs/backend-proof.md, the FP16 Tier E gate;
# docs/experiments/backend-proof-p2/plan_compare.py). No production target
# may link any of it; the check at the end of this file fails the configure
# if one does.

# Reading a std::expected's error only when it holds one (expected_error.h,
# D-083), for every GoogleTest executable (llmp_gtest links it).
add_library(llmp_test_expected INTERFACE)
target_include_directories(llmp_test_expected INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}")

# Frozen digests and the published v1 store's ID map for the v1 equivalence
# proofs (v1_frozen.h).
add_library(llmp_test_v1_frozen INTERFACE)
target_include_directories(llmp_test_v1_frozen INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(llmp_test_v1_frozen INTERFACE llmp_json llmp_base)

# The recording's events and JSON lines, in every profile.
add_library(llmp_plan_record STATIC plan_record.cc)
target_include_directories(llmp_plan_record PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(llmp_plan_record PRIVATE llmp_warnings)

# plan_record_sample.txt, the JSON lines the writer must produce for its
# events, which plan_compare.py's tests read too, compiled in at build time
# so a deployed test needs no source tree.
set(sample "${CMAKE_CURRENT_BINARY_DIR}/plan_record_sample.cc")
add_custom_command(
  OUTPUT "${sample}"
  COMMAND "${CMAKE_COMMAND}" "-DINPUT=${CMAKE_CURRENT_SOURCE_DIR}/plan_record_sample.txt"
          "-DOUTPUT=${sample}" -P "${CMAKE_CURRENT_SOURCE_DIR}/plan_record_sample.cmake"
  DEPENDS plan_record_sample.txt plan_record_sample.cmake
  COMMENT "Compiling in the executed-plan recording's sample"
  VERBATIM)
add_library(llmp_plan_record_sample STATIC "${sample}")
target_link_libraries(llmp_plan_record_sample PUBLIC llmp_plan_record)

# The launch recorder (launch_recorder.h), for binaries that run GGML-derived
# kernels (llmp_ggml_kernels brings the CUDA runtime). An object library,
# so its wrappers are in every link that uses it, whatever the link order.
if(LLMP_CUDA)
  add_library(llmp_launch_recorder OBJECT launch_recorder.cc)
  target_include_directories(llmp_launch_recorder SYSTEM PRIVATE
                             "${LLMP_SDK}/cuda/targets/${LLMP_CUDA_TARGET_DIR}/include")
  # llmp_cuda_device_memory brings the NVIDIA driver.
  target_link_libraries(llmp_launch_recorder PUBLIC llmp_plan_record llmp_ggml_kernels
                                                      llmp_cuda_device_memory
                        PRIVATE llmp_base llmp_warnings)
  target_link_options(llmp_launch_recorder INTERFACE
                      "LINKER:--wrap=__cudaLaunchKernel,--wrap=cudaLaunchKernelExC"
                      "LINKER:--wrap=cudaLaunchKernel,--wrap=cudaLaunchCooperativeKernel"
                      "LINKER:--wrap=cublasLtMatmul"
                      "LINKER:--wrap=cudaMemcpyAsync,--wrap=cudaMemcpy2DAsync,--wrap=cudaMemsetAsync"
                      "LINKER:--wrap=cuMemcpyAsync"
                      "LINKER:--wrap=cublasSgemm_v2,--wrap=cublasGemmEx"
                      "LINKER:--wrap=cublasGemmBatchedEx,--wrap=cublasGemmStridedBatchedEx")
endif()

# The scheduler's task programs (src/scheduler/programs.h) under the paged
# harnesses' names, with LaunchOnlyProgram, which only tests post
# (paged_programs.h); and in CUDA builds the engine's paged node
# (src/engine/paged_node.h) under the same names (paged_node.h).
add_library(llmp_paged_programs INTERFACE)
target_include_directories(llmp_paged_programs INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(llmp_paged_programs INTERFACE llmp_scheduler llmp_memory)
if(LLMP_CUDA)
  add_library(llmp_paged_node INTERFACE)
  target_include_directories(llmp_paged_node INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}")
  target_link_libraries(llmp_paged_node INTERFACE llmp_paged_programs llmp_engine)
endif()

# A test runner's device backend made, mapped, bound and released as
# production does a model's (runner_kernels.h), for the GPU tests.
if(LLMP_CUDA)
  add_library(llmp_runner_kernels STATIC runner_kernels.cc)
  target_include_directories(llmp_runner_kernels PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
  target_link_libraries(llmp_runner_kernels PUBLIC llmp_engine PRIVATE llmp_warnings)
endif()

# A step source whose programs run as host work on the fake device and
# write deterministic bytes (fake_step_source.h), for the engine core's
# runner, wave composer and prefill tests in every profile.
add_library(llmp_fake_step_source STATIC fake_step_source.cc)
target_include_directories(llmp_fake_step_source PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(llmp_fake_step_source PUBLIC llmp_engine_core llmp_fake_providers
                      PRIVATE llmp_warnings)

# Common bounded teacher-forced scoring over the shared runner.
add_library(llmp_corpus_score STATIC corpus_score.cc)
target_include_directories(llmp_corpus_score PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(llmp_corpus_score PUBLIC llmp_engine_core PRIVATE llmp_sampling llmp_warnings)

# A test-only writer of v1 prepared artifacts and compositions
# (artifact_writer.h, D-113), in every profile: the v1 reader's fixtures.
add_library(llmp_test_artifact_writer STATIC artifact_writer.cc)
target_include_directories(llmp_test_artifact_writer PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(llmp_test_artifact_writer PUBLIC llmp_artifact llmp_format
                      PRIVATE llmp_base llmp_warnings)

# No production binary links the support libraries, directly or through
# another library. Checked once every target exists.
function(llmp_check_no_test_support)
  set(pending llmp llmp-runtime)
  set(seen "")
  while(pending)
    list(POP_FRONT pending target)
    # Anywhere in the item, so that a generator expression cannot hide it.
    if(target MATCHES "(^|[^A-Za-z0-9_])llmp_(test_expected|test_main|plan_record|plan_record_sample|launch_recorder|paged_programs|paged_node|paged_runners|fake_step_source|test_artifact_writer|corpus_score|runner_kernels)([^A-Za-z0-9_]|$)")
      message(FATAL_ERROR "A production binary links the test support library: ${target}.")
    endif()
    if(target IN_LIST seen OR NOT TARGET "${target}")
      continue()
    endif()
    list(APPEND seen "${target}")
    get_target_property(type "${target}" TYPE)
    set(links "")
    if(NOT type STREQUAL "INTERFACE_LIBRARY")
      get_target_property(direct "${target}" LINK_LIBRARIES)
      if(direct)
        list(APPEND links ${direct})
      endif()
    endif()
    get_target_property(interface "${target}" INTERFACE_LINK_LIBRARIES)
    if(interface)
      list(APPEND links ${interface})
    endif()
    foreach(link IN LISTS links)
      string(REGEX REPLACE "^\\$<LINK_ONLY:(.*)>$" "\\1" link "${link}")
      list(APPEND pending "${link}")
    endforeach()
  endwhile()
endfunction()
cmake_language(DEFER DIRECTORY "${PROJECT_SOURCE_DIR}" CALL llmp_check_no_test_support)
