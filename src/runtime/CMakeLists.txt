# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0

# The node runtime process (/usr/libexec/jitllm/jitllm-runtime once
# packaged, D-063): startup, readiness and stop (runtime.h), the serving
# commands' arguments (commands.h), a prefill's chunks (prefill.h), the
# start's memory guard (memory_guard.h), serving's choices from a binding (model_limits.h),
# a chat cohort's state-capacity policy (cohort_capacity.h) and schedule
# (cohort_schedule.h), when pressure from outside has it trim
# (pressure_trim.h), room for a swap (swap_room.h), the chat route's
# HTTP, requests, binding, server, watchdog and memory-derived limits
# (http.h, api.h, binding.h, api_server.h, watchdog.h, intake_limits.h;
# D-097, D-102), each model's settings in three layers and their
# calibration on the machine (model_settings.h, calibration.h; D-103),
# hang recovery's ladder (hang_ladder.h, D-102), and the records of
# conversations kept across a restart with their keeper (kept_record.h,
# state_keeper.h, D-105), all vendor-free.
add_library(jitllm_runtime STATIC runtime.cc commands.cc http.cc api.cc api_server.cc binding.cc
                                  calibration.cc cohort_capacity.cc cohort_schedule.cc
                                  completion_tokens.cc hang_ladder.cc intake_limits.cc
                                  gemma_profile.cc kept_record.cc memory_guard.cc model_limits.cc model_settings.cc
                                  prefill.cc pressure_trim.cc state_keeper.cc swap_room.cc
                                  watchdog.cc)
target_include_directories(jitllm_runtime PUBLIC "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(jitllm_runtime PUBLIC jitllm_artifact jitllm_config jitllm_platform
                                            jitllm_tokenizer
                      PRIVATE jitllm_base jitllm_build_info jitllm_report jitllm_device_probe jitllm_model
                              jitllm_json jitllm_sampling jitllm_warnings)

# The serving commands themselves (serving.h, D-096): the configured models
# on the engine's paged node, in CUDA builds; elsewhere a refusal.
if(JITLLM_CUDA)
  add_library(jitllm_runtime_serving STATIC serving.cc serve_commands.cc serve_api.cc)
  target_link_libraries(jitllm_runtime_serving PUBLIC jitllm_runtime jitllm_engine jitllm_sampling
                        PRIVATE jitllm_base jitllm_build_info jitllm_platform jitllm_warnings)
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
