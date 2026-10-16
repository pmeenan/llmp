# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0

# The node runtime process (/usr/libexec/llmp/llmp-runtime once
# packaged, D-063): startup, readiness and stop (runtime.h), the serving
# commands' arguments (commands.h), a prefill's chunks (prefill.h), the
# start's memory guard (memory_guard.h),
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
add_library(llmp_runtime STATIC runtime.cc commands.cc http.cc api.cc api_server.cc approvals.cc binding.cc
                                  calibration.cc cohort_capacity.cc cohort_schedule.cc
                                  completion_tokens.cc component_admission.cc hang_ladder.cc intake_limits.cc
                                  gemma_profile.cc kept_record.cc memory_guard.cc model_settings.cc
                                  prefill.cc pressure_trim.cc state_keeper.cc swap_room.cc
                                  watchdog.cc)
target_include_directories(llmp_runtime PUBLIC "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(llmp_runtime PUBLIC llmp_artifact llmp_config llmp_platform
                                            llmp_tokenizer
                      PRIVATE llmp_base llmp_build_info llmp_report llmp_device_probe llmp_model
                              llmp_json llmp_sampling llmp_spec llmp_components llmp_warnings)

# An LLM's conversations over the engine core (serving.h's Llm, llm.cc) and
# the one serving adapter per pipeline kind over the shared runner
# (engine_llm.h, engine_image.h; D-107), in every build profile: they name
# no kernel library and link no device backend, so the tests run them on
# the fake one.
add_library(llmp_runtime_llm STATIC llm.cc engine_llm.cc engine_image.cc snapshot_store.cc component_bindings.cc)
target_include_directories(llmp_runtime_llm PUBLIC "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(llmp_runtime_llm PUBLIC llmp_runtime llmp_engine_core llmp_sampling
                                                llmp_chat llmp_tokenizer llmp_spec
                      PRIVATE llmp_base llmp_platform llmp_warnings)

# The serving commands themselves (serving.h, D-096): the configured models
# on the engine's paged node, in CUDA builds; elsewhere a refusal.
if(LLMP_CUDA)
  add_library(llmp_runtime_serving STATIC serving.cc component_build.cc component_image_build.cc serve_commands.cc
                                          serve_api.cc)
  target_link_libraries(llmp_runtime_serving PUBLIC llmp_runtime llmp_runtime_llm llmp_engine
                                                      llmp_sampling
                        PRIVATE llmp_base llmp_build_info llmp_platform llmp_decode_selected
                                llmp_warnings)
else()
  add_library(llmp_runtime_serving STATIC serving_none.cc)
  target_link_libraries(llmp_runtime_serving PUBLIC llmp_runtime PRIVATE llmp_warnings)
endif()

add_executable(llmp-runtime main.cc)
target_link_libraries(llmp-runtime PRIVATE llmp_runtime llmp_runtime_serving llmp_platform
                                             llmp_warnings)
if(LLMP_CUDA)
  # cuBLAS, linked dynamically (D-076), from lib/llmp two levels above the
  # runtime: /usr/lib/llmp when packaged (packaging/CMakeLists.txt), and
  # here build/<preset>/lib/llmp, hard links to the SDK's libraries.
  target_link_options(llmp-runtime PRIVATE "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../../lib/llmp")
  set(cublas_dir "${LLMP_SDK}/cuda/targets/${LLMP_CUDA_TARGET_DIR}/lib")
  file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/lib/llmp")
  foreach(library IN ITEMS libcublas.so.13 libcublasLt.so.13)
    file(REAL_PATH "${cublas_dir}/${library}" resolved)
    file(CREATE_LINK "${resolved}" "${PROJECT_BINARY_DIR}/lib/llmp/${library}" COPY_ON_ERROR)
  endforeach()
endif()
