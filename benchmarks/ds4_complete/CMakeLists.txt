# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0

# Temporary causal-comparison fixture. CPU builds retain checked metadata,
# ledger and refusal controls without a foreign runtime or model load.
add_library(jitllm_ds4_complete_contract STATIC executor.cc)
target_include_directories(jitllm_ds4_complete_contract PUBLIC "${PROJECT_SOURCE_DIR}")
target_link_libraries(jitllm_ds4_complete_contract PUBLIC jitllm_ggml_tensors jitllm_model
                                                PRIVATE jitllm_warnings)

if(JITLLM_CUDA)
  add_library(jitllm_ds4_complete_native STATIC executor.cc binding.cc weight_preparation.cc runner.cc)
  target_include_directories(jitllm_ds4_complete_native PUBLIC
                             "${PROJECT_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}/benchmarks")
  target_compile_definitions(jitllm_ds4_complete_native PRIVATE JITLLM_DS4_COMPLETE_CUDA=1)
  target_link_libraries(jitllm_ds4_complete_native PUBLIC jitllm_engine
                                               PRIVATE jitllm_warnings)

  add_executable(jitllm_ds4_complete main.cc)
  target_include_directories(jitllm_ds4_complete SYSTEM PRIVATE
                             "${JITLLM_SDK}/cuda/targets/${JITLLM_CUDA_TARGET_DIR}/include")
  target_link_libraries(jitllm_ds4_complete PRIVATE jitllm_ds4_complete_native jitllm_warnings)
  target_link_options(jitllm_ds4_complete PRIVATE
                      "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../../cublas")
endif()
