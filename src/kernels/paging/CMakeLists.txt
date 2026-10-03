# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0

# jitLLM's own kernels that the engine's device jobs queue around paging
# (paging.h): filling device ranges and gathering Qwen3.8's n-gram rows.
# Not registry-bound operations (D-053). CUDA builds only.
if(JITLLM_CUDA)
  add_library(jitllm_paging_kernels STATIC clamp_tokens.cu fill_ranges.cu gather_rows.cu)
  target_include_directories(jitllm_paging_kernels PUBLIC "${PROJECT_SOURCE_DIR}/src")
  target_link_libraries(jitllm_paging_kernels PRIVATE jitllm_warnings)
  # sm_121a, as the GGML module's units build, and the profile's discrete
  # GPUs (D-082).
  set_target_properties(jitllm_paging_kernels PROPERTIES
                        CUDA_ARCHITECTURES "121a-real;${JITLLM_CUDA_DISCRETE_SASS}")
endif()
