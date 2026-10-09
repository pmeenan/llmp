# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0

# The engine (docs/engine.md, docs/runtime-serving.md), above the kernels in
# the layers, in two libraries (D-096 as amended for M3.6):
#
# - llmp_engine_core, in every build profile: the paged node that composes
#   the providers, the resource core and the lanes on one GPU
#   (paged_node.h), and the vendor-free skeleton every model runner is built
#   from: its weights' paging (paged_weights.h), its conversation state,
#   spill and verify snapshot (live_state.h, checkpoint_file.h), its own
#   memory (runner_memory.h), what its plans hold (plan_account.h), prefill
#   lookahead, the request cohort, the shared slot lifecycle derived from
#   the stages' state contracts (slot_lifecycle.h) and Qwen3.8's n-gram
#   rows (ple_rows.h).
#   It reaches the device only through the providers, the device runtime
#   (providers/device_runtime.h) and the paging kernels (kernels/paging/),
#   and names no kernel library. It links no device backend: a program
#   links the CUDA one (llmp_engine does) or the fake device runtime and
#   the host paging kernels (llmp_fake_device_backend), never both.
# - llmp_engine, CUDA builds only: the GGML plans and graph runs
#   (planned.h, graph_runs.h), the runners' cuBLAS and launch context
#   (runner_resources.h), and each M3 model's runner on the core, with
#   their plans, state layouts and steps. It is not given CUDA's headers
#   (docs/portability.md). llmp-runtime links it, and so do the paged
#   harnesses and their tests.
add_library(llmp_gemma4_greedy STATIC gemma4_greedy.cc)
target_include_directories(llmp_gemma4_greedy PUBLIC "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(llmp_gemma4_greedy PRIVATE llmp_sampling llmp_warnings)

add_library(llmp_engine_core STATIC paged_node.cc paged_weights.cc live_state.cc checkpoint_file.cc
                                      request_cohort.cc slot_lifecycle.cc ple_rows.cc runner_memory.cc
                                      plan_account.cc)
target_include_directories(llmp_engine_core PUBLIC "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(llmp_engine_core PUBLIC llmp_scheduler llmp_memory llmp_storage
                                              llmp_device_memory llmp_artifact llmp_model
                                       PRIVATE llmp_base llmp_platform llmp_warnings)

# The fake device backend for programs that run the engine core on the
# host (its tests, in every profile).
add_library(llmp_fake_device_backend INTERFACE)
target_link_libraries(llmp_fake_device_backend INTERFACE llmp_fake_device_runtime
                                                         llmp_host_paging_kernels)

if(LLMP_CUDA)
  add_library(llmp_engine STATIC planned.cc graph_runs.cc runner_resources.cc expert_slab.cc
                                   live_state_launch.cc
                                   dsv4_plan.cc dsv4_runner.cc gemma2_plan.cc gemma2_runner.cc gemma3_plan.cc gemma3_runner.cc gemma4_plan.cc gemma4_runner.cc qwen38_plan.cc qwen38_wave_plan.cc
                                   qwen38_runner.cc qwen_image_runner.cc)
  target_include_directories(llmp_engine PUBLIC "${PROJECT_SOURCE_DIR}/src")
  target_sources(llmp_engine PRIVATE gemma4_assistant_plan.cc gemma4_assistant.cc)
  target_link_libraries(llmp_engine PUBLIC llmp_engine_core llmp_scheduler llmp_memory llmp_storage
                                             llmp_device_runtime llmp_ggml_kernels
                                             llmp_image_kernels llmp_paging_kernels
                                             llmp_artifact llmp_model llmp_tokenizer
                                             llmp_chat
                                      PRIVATE llmp_base llmp_platform llmp_gemma4_greedy llmp_warnings)
endif()
