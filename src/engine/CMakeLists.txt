# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0

# The engine (docs/engine.md, docs/runtime-serving.md): the paged node that
# composes the providers, the resource core and the lanes on one GPU
# (paged_node.h); the skeleton every model runner is built from: its
# weights' paging (paged_weights.h), its conversation state, spill and
# verify snapshot (live_state.h), its planned shapes (planned.h), their runs
# and graphs (graph_runs.h) and its own device resources
# (runner_resources.h); and each M3 model's runner on it, which adds its
# plans, state layout and steps (dsv4_runner.h, qwen38_runner.h,
# qwen_image_runner.h). Above the kernels in the layers. It reaches the
# device only through the providers (the device runtime,
# providers/device_runtime.h, among them) and the kernels, so it is not
# given CUDA's headers (docs/portability.md); it builds where its kernels
# do, CUDA builds only. llmp-runtime links it, and so do the paged
# harnesses and their tests.
add_library(llmp_gemma4_greedy STATIC gemma4_greedy.cc)
target_include_directories(llmp_gemma4_greedy PUBLIC "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(llmp_gemma4_greedy PRIVATE llmp_sampling llmp_warnings)

if(LLMP_CUDA)
  add_library(llmp_engine STATIC paged_node.cc paged_weights.cc live_state.cc checkpoint_file.cc planned.cc
                                   graph_runs.cc runner_resources.cc ple_rows.cc request_cohort.cc
                                   dsv4_plan.cc dsv4_runner.cc gemma2_plan.cc gemma2_runner.cc gemma3_plan.cc gemma3_runner.cc gemma4_plan.cc gemma4_runner.cc qwen38_plan.cc qwen38_wave_plan.cc
                                   qwen38_runner.cc qwen_image_runner.cc)
  target_include_directories(llmp_engine PUBLIC "${PROJECT_SOURCE_DIR}/src")
  target_sources(llmp_engine PRIVATE gemma4_assistant_plan.cc gemma4_assistant.cc)
  target_link_libraries(llmp_engine PUBLIC llmp_scheduler llmp_memory llmp_storage
                                             llmp_device_runtime llmp_ggml_kernels
                                             llmp_image_kernels llmp_paging_kernels
                                             llmp_artifact llmp_model llmp_tokenizer
                                             llmp_chat
                                      PRIVATE llmp_base llmp_platform llmp_gemma4_greedy llmp_warnings)
endif()
