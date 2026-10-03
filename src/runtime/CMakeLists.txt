# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0

# The node runtime process (/usr/libexec/jitllm/jitllm-runtime once
# packaged, D-063): startup, readiness and stop (runtime.h), the serving
# commands' arguments (commands.h), a prefill's chunks (prefill.h), the
# start's memory guard (memory_guard.h), checkpoint ceilings (model_limits.h),
# a chat cohort's state-capacity policy (cohort_capacity.h) and schedule
# (cohort_schedule.h), when pressure from outside has it trim
# (pressure_trim.h), room for a swap (swap_room.h), and the chat route's
# HTTP, requests, binding, server and watchdog (http.h, api.h, binding.h,
# api_server.h, watchdog.h; D-097), all vendor-free.
add_library(jitllm_runtime STATIC runtime.cc commands.cc http.cc api.cc api_server.cc binding.cc
                                  cohort_capacity.cc cohort_schedule.cc completion_tokens.cc
                                  memory_guard.cc model_limits.cc prefill.cc pressure_trim.cc
                                  swap_room.cc watchdog.cc)
target_include_directories(jitllm_runtime PUBLIC "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(jitllm_runtime PUBLIC jitllm_config jitllm_platform
                      PRIVATE jitllm_build_info jitllm_report jitllm_device_probe jitllm_json
                              jitllm_sampling jitllm_tokenizer jitllm_warnings)

# The serving commands themselves (serving.h, D-096): the configured models
# on the engine's paged node, in CUDA builds; elsewhere a refusal.
if(JITLLM_CUDA)
  add_library(jitllm_runtime_serving STATIC serving.cc serve_commands.cc serve_api.cc)
  target_link_libraries(jitllm_runtime_serving PUBLIC jitllm_runtime jitllm_engine jitllm_sampling
                        PRIVATE jitllm_base jitllm_platform jitllm_warnings)
else()
  add_library(jitllm_runtime_serving STATIC serving_none.cc)
  target_link_libraries(jitllm_runtime_serving PUBLIC jitllm_runtime PRIVATE jitllm_warnings)
endif()

add_executable(jitllm-runtime main.cc)
target_link_libraries(jitllm-runtime PRIVATE jitllm_runtime jitllm_runtime_serving jitllm_platform
                                             jitllm_warnings)
if(JITLLM_CUDA)
  # cuBLAS, linked dynamically (D-076), from lib/jitllm two levels above the
  # runtime: /usr/lib/jitllm when packaged (packaging/CMakeLists.txt), and
  # here build/<preset>/lib/jitllm, hard links to the SDK's libraries.
  target_link_options(jitllm-runtime PRIVATE "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../../lib/jitllm")
  set(cublas_dir "${JITLLM_SDK}/cuda/targets/${JITLLM_CUDA_TARGET_DIR}/lib")
  file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/lib/jitllm")
  foreach(library IN ITEMS libcublas.so.13 libcublasLt.so.13)
    file(REAL_PATH "${cublas_dir}/${library}" resolved)
    file(CREATE_LINK "${resolved}" "${PROJECT_BINARY_DIR}/lib/jitllm/${library}" COPY_ON_ERROR)
  endforeach()
endif()
