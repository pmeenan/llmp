# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0

# Unit tests of jitLLM's modules, one GoogleTest executable per module
# (docs/architecture.md#testing). Every test carries the `unit` label.

add_executable(qwen38_vocab_test qwen38_vocab_test.cc)
target_link_libraries(qwen38_vocab_test PRIVATE jitllm_draft_vocab_contract)
jitllm_gtest(qwen38_vocab_test TEST_PREFIX unit. PROPERTIES LABELS unit)

add_executable(qwen38_reference_test qwen38_reference_test.cc)
target_link_libraries(qwen38_reference_test PRIVATE jitllm_vocab_reference jitllm_base)
jitllm_gtest(qwen38_reference_test TEST_PREFIX unit. PROPERTIES LABELS unit
             ENVIRONMENT "JITLLM_TEST_SCRATCH=${CMAKE_CURRENT_BINARY_DIR}/scratch")

add_executable(cli_test cli_test.cc)
target_link_libraries(cli_test PRIVATE jitllm_cli jitllm_surface_versions)
target_compile_definitions(cli_test PRIVATE "JITLLM_TEST_PROJECT_VERSION=\"${PROJECT_VERSION}\"")
jitllm_gtest(cli_test TEST_PREFIX unit. PROPERTIES LABELS unit
             ENVIRONMENT "JITLLM_TEST_SCRATCH=${CMAKE_CURRENT_BINARY_DIR}/scratch")

add_executable(base_test base_test.cc)
add_executable(gemma_joined_groups_test gemma_joined_groups_test.cc)
target_link_libraries(gemma_joined_groups_test PRIVATE jitllm_base)
jitllm_gtest(gemma_joined_groups_test TEST_PREFIX unit. PROPERTIES LABELS unit)
target_link_libraries(base_test PRIVATE jitllm_report jitllm_base)
jitllm_gtest(base_test TEST_PREFIX unit. PROPERTIES LABELS unit)

add_executable(platform_test platform_test.cc)
target_link_libraries(platform_test PRIVATE jitllm_platform)
jitllm_gtest(platform_test TEST_PREFIX unit. PROPERTIES LABELS unit
             ENVIRONMENT "JITLLM_TEST_SCRATCH=${CMAKE_CURRENT_BINARY_DIR}/scratch")

# The CUDA provider, apart from its device queries, which need a GPU
# (smoke.doctor in tests/smoke).
if(JITLLM_CUDA)
  add_executable(cuda_test cuda_test.cc)
  target_link_libraries(cuda_test PRIVATE jitllm_cuda_probe)
  jitllm_gtest(cuda_test TEST_PREFIX unit. PROPERTIES LABELS unit)
endif()

# The node configuration. Tests that need scratch trees (here and in
# platform_test) put them in the build tree, whose parents no other user
# shares, and remote runs map it (JITLLM_TEST_SCRATCH).
add_executable(config_test config_test.cc)
target_link_libraries(config_test PRIVATE jitllm_config jitllm_platform)
jitllm_gtest(config_test TEST_PREFIX unit. PROPERTIES LABELS unit
             ENVIRONMENT "JITLLM_TEST_SCRATCH=${CMAKE_CURRENT_BINARY_DIR}/scratch")

# The runtime's startup steps and the process services it relies on, and a
# prefill's chunks.
add_executable(runtime_test runtime_test.cc)
target_link_libraries(runtime_test PRIVATE jitllm_runtime jitllm_platform)
jitllm_gtest(runtime_test TEST_PREFIX unit. PROPERTIES LABELS unit
             ENVIRONMENT "JITLLM_TEST_SCRATCH=${CMAKE_CURRENT_BINARY_DIR}/scratch")

# A model's settings in three layers (D-103): sources, precedence, refused
# and ignored overrides, the reasoning markers, the listing.
add_executable(model_settings_test model_settings_test.cc)
target_link_libraries(model_settings_test PRIVATE jitllm_runtime jitllm_config)
jitllm_gtest(model_settings_test TEST_PREFIX unit. PROPERTIES LABELS unit)

# The calibration record (D-103): its format, key, atomic write and trusted
# read, and what a model's first uses measure.
add_executable(calibration_test calibration_test.cc)
target_link_libraries(calibration_test PRIVATE jitllm_runtime)
jitllm_gtest(calibration_test TEST_PREFIX unit. PROPERTIES LABELS unit
             ENVIRONMENT "JITLLM_TEST_SCRATCH=${CMAKE_CURRENT_BINARY_DIR}/scratch")

# A chat cohort's state-capacity policy: who waits, fails, is preempted.
add_executable(cohort_capacity_test cohort_capacity_test.cc)
target_link_libraries(cohort_capacity_test PRIVATE jitllm_runtime)
jitllm_gtest(cohort_capacity_test TEST_PREFIX unit. PROPERTIES LABELS unit)

add_executable(cohort_schedule_test cohort_schedule_test.cc)
target_link_libraries(cohort_schedule_test PRIVATE jitllm_runtime)
jitllm_gtest(cohort_schedule_test TEST_PREFIX unit. PROPERTIES LABELS unit)

# When pressure on memory from outside has the runtime trim: its target,
# hysteresis and back-off, from readings of the test's own.
add_executable(pressure_trim_test pressure_trim_test.cc)
target_link_libraries(pressure_trim_test PRIVATE jitllm_runtime)
jitllm_gtest(pressure_trim_test TEST_PREFIX unit. PROPERTIES LABELS unit)

# D-102's hang ladder: what is and is not a hang, its rungs and its last
# resort, on a clock of the test's own.
add_executable(hang_ladder_test hang_ladder_test.cc)
target_link_libraries(hang_ladder_test PRIVATE jitllm_runtime)
jitllm_gtest(hang_ladder_test TEST_PREFIX unit. PROPERTIES LABELS unit)

# D-105's conversations kept across a restart, host-side: their owner-only
# files, their record's format and checks, and the keeper that writes
# records in the background.
add_executable(kept_state_test kept_state_test.cc)
target_link_libraries(kept_state_test PRIVATE jitllm_runtime jitllm_platform jitllm_base)
jitllm_gtest(kept_state_test TEST_PREFIX unit. PROPERTIES LABELS unit
             ENVIRONMENT "JITLLM_TEST_SCRATCH=${CMAKE_CURRENT_BINARY_DIR}/scratch")

# Room for a swap's incoming model: reclaimed in whole extents and checked
# again, against a ledger whose charge moves in whole extents.
add_executable(swap_room_test swap_room_test.cc)
target_link_libraries(swap_room_test PRIVATE jitllm_runtime)
jitllm_gtest(swap_room_test TEST_PREFIX unit. PROPERTIES LABELS unit)

# The chat route (D-097): requests and their bounds, the bind resolution and
# Host guard, and the server over real loopback sockets with a fake backend.
add_executable(api_test api_test.cc)
target_link_libraries(api_test PRIVATE jitllm_runtime jitllm_json)
jitllm_gtest(api_test TEST_PREFIX unit. PROPERTIES LABELS unit)

# The actual Llm loops with fake completed chunks; no GPU work. Serving's
# Llm implementation is currently linked only in CUDA profiles.
if(JITLLM_CUDA)
  add_executable(llm_scores_test llm_scores_test.cc)
  target_link_libraries(llm_scores_test PRIVATE jitllm_runtime_serving)
  file(RELATIVE_PATH runtime "${CMAKE_CURRENT_BINARY_DIR}" "${PROJECT_BINARY_DIR}/cublas")
  target_link_options(llm_scores_test PRIVATE "LINKER:--enable-new-dtags,-rpath,$ORIGIN/${runtime}")
  jitllm_gtest(llm_scores_test TEST_PREFIX unit. PROPERTIES LABELS unit)
endif()

# The resource core: the catalog and the memory manager's policy, on no
# vendor SDK (D-026).
add_executable(catalog_test catalog_test.cc)
target_link_libraries(catalog_test PRIVATE jitllm_catalog jitllm_base)
jitllm_gtest(catalog_test TEST_PREFIX unit. PROPERTIES LABELS unit)

add_executable(memory_test memory_test.cc)
target_link_libraries(memory_test PRIVATE jitllm_memory jitllm_catalog)
jitllm_gtest(memory_test TEST_PREFIX unit. PROPERTIES LABELS unit)

add_executable(admission_test admission_test.cc)
target_link_libraries(admission_test PRIVATE jitllm_scheduler)
jitllm_gtest(admission_test TEST_PREFIX unit. PROPERTIES LABELS unit)

# D-048's lanes, completion board and task trees, with real threads.
add_executable(lanes_test lanes_test.cc)
target_link_libraries(lanes_test PRIVATE jitllm_scheduler)
jitllm_gtest(lanes_test TEST_PREFIX unit. PROPERTIES LABELS unit)

# The scheduler thread's turn loop and its lanes over the fake providers:
# deterministic turns, and threaded lost-wakeup and memory-ordering stress.
add_executable(scheduler_test scheduler_test.cc)
target_link_libraries(scheduler_test PRIVATE jitllm_scheduler jitllm_fake_providers)
jitllm_gtest(scheduler_test TEST_PREFIX unit. PROPERTIES LABELS unit)

# The runtime wake (docs/experiments/runtime-wake/): the device lanes on
# their own threads over the fake device, which completes fences when the
# test chooses; lost wakeups, the backstop, the relay and shutdown.
add_executable(wake_test wake_test.cc)
target_link_libraries(wake_test PRIVATE jitllm_scheduler jitllm_fake_providers)
jitllm_gtest(wake_test TEST_PREFIX unit. PROPERTIES LABELS unit)

# The D-081 page-in path, its evictions and kernel jobs' leases, over the
# fake providers: deterministic turns and a threaded stress.
add_executable(pagein_test pagein_test.cc)
target_link_libraries(pagein_test PRIVATE jitllm_scheduler jitllm_fake_providers)
jitllm_gtest(pagein_test TEST_PREFIX unit. PROPERTIES LABELS unit)

# BP-S3's acquisition (tests/support/paged_programs.h) over the fake
# providers: two models under one budget, each evicting only the other's
# weights.
add_executable(acquire_test acquire_test.cc)
target_link_libraries(acquire_test PRIVATE jitllm_paged_programs jitllm_scheduler
                                           jitllm_fake_providers)
jitllm_gtest(acquire_test TEST_PREFIX unit. PROPERTIES LABELS unit)

# A request's lease (M3's lease per request; paged_programs.h
# RequestProgram) over the fake providers: steps under one lease, eviction
# refused and swaps waiting while it is held, release at its end or after a
# cancelled request's step drains, and kept for good when a step is unproven.
add_executable(held_lease_test held_lease_test.cc)
target_link_libraries(held_lease_test PRIVATE jitllm_paged_programs jitllm_scheduler
                                               jitllm_fake_providers)
jitllm_gtest(held_lease_test TEST_PREFIX unit. PROPERTIES LABELS unit)

# Storage and whole reads, on the fake and on io_uring with a real file.
add_executable(storage_test storage_test.cc)
target_link_libraries(storage_test PRIVATE jitllm_fake_providers jitllm_storage jitllm_platform)
jitllm_gtest(storage_test TEST_PREFIX unit. PROPERTIES LABELS unit
             ENVIRONMENT "JITLLM_TEST_SCRATCH=${CMAKE_CURRENT_BINARY_DIR}/scratch")

# The providers' shared rules, on the deterministic fakes.
add_executable(device_memory_test device_memory_test.cc)
target_link_libraries(device_memory_test PRIVATE jitllm_fake_providers)
jitllm_gtest(device_memory_test TEST_PREFIX unit. PROPERTIES LABELS unit)

# The retained-backing replay's trace reader, reference and candidate
# designs (benchmarks/retained_backing/), on the address-only fake.
add_executable(rb_replay_test rb_replay_test.cc)
target_link_libraries(rb_replay_test PRIVATE jitllm_retained_backing)
jitllm_gtest(rb_replay_test TEST_PREFIX unit. PROPERTIES LABELS unit
             ENVIRONMENT "JITLLM_TEST_SCRATCH=${CMAKE_CURRENT_BINARY_DIR}/scratch")

add_executable(long_context_tasks_test long_context_tasks_test.cc)
target_link_libraries(long_context_tasks_test PRIVATE jitllm_long_context_tasks)
jitllm_gtest(long_context_tasks_test TEST_PREFIX unit. PROPERTIES LABELS unit)

add_executable(device_execution_test device_execution_test.cc)
target_link_libraries(device_execution_test PRIVATE jitllm_fake_providers)
jitllm_gtest(device_execution_test TEST_PREFIX unit. PROPERTIES LABELS unit)

# The CUDA device-memory and device-execution providers against a real
# device: label `gpu`, and `gpu-discrete` too, so on a Spark and on a
# discrete GPU the build targets (D-082).
if(JITLLM_CUDA)
  add_executable(cuda_device_execution_test cuda_device_execution_test.cc)
  target_include_directories(cuda_device_execution_test SYSTEM PRIVATE
                             "${JITLLM_SDK}/cuda/targets/${JITLLM_CUDA_TARGET_DIR}/include")
  target_link_libraries(cuda_device_execution_test PRIVATE jitllm_cuda_device_memory)
  jitllm_gtest(cuda_device_execution_test TEST_PREFIX unit. PROPERTIES LABELS unit)

  # The scheduler and its lanes over io_uring, CUDA host VMM and CUDA
  # streams.
  add_executable(cuda_lanes_test cuda_lanes_test.cc)
  target_link_libraries(cuda_lanes_test PRIVATE jitllm_scheduler jitllm_cuda_device_memory)
  jitllm_gpu_gtest(cuda_lanes_test DISCRETE ALL
                   ENVIRONMENT "JITLLM_TEST_SCRATCH=${CMAKE_CURRENT_BINARY_DIR}/scratch")

  # The D-081 page-in path over io_uring, a host-VMM landing zone and
  # device VMM, the same on both classes (D-082). Each test runs both
  # instantiations (the VMM work on the device lane or its own).
  # Coalesced restores (CudaCoalescing, BP-P1) are verified on the GB10
  # only so far.
  # Write-back (CudaWriteBack) is verified on the GB10 only so far.
  add_executable(cuda_pagein_test cuda_pagein_test.cc)
  target_include_directories(cuda_pagein_test SYSTEM PRIVATE
                             "${JITLLM_SDK}/cuda/targets/${JITLLM_CUDA_TARGET_DIR}/include")
  target_link_libraries(cuda_pagein_test PRIVATE jitllm_scheduler jitllm_cuda_device_memory)
  jitllm_gpu_gtest(cuda_pagein_test
                   DISCRETE "*/CudaPageIn.LandedLoadsReachDeviceVmmAndReloadIdentically/*"
                            "*/CudaPageIn.CancellingWithLoadsInFlightDrainsThem/*"
                            "*/CudaPageIn.CancellingDuringACopyWaitsForItsFence/*"
                   ENVIRONMENT "JITLLM_TEST_SCRATCH=${CMAKE_CURRENT_BINARY_DIR}/scratch")

  # BP-S3's paged node (tests/support/paged_node.h) on a GB10: two models
  # on their own streams alternating under one budget, through one zone and
  # one shared workspace.
  add_executable(cuda_paged_node_test cuda_paged_node_test.cc)
  target_include_directories(cuda_paged_node_test SYSTEM PRIVATE
                             "${JITLLM_SDK}/cuda/targets/${JITLLM_CUDA_TARGET_DIR}/include")
  target_link_libraries(cuda_paged_node_test PRIVATE jitllm_paged_node jitllm_fake_providers)
  jitllm_gpu_gtest(cuda_paged_node_test
                   ENVIRONMENT "JITLLM_TEST_SCRATCH=${CMAKE_CURRENT_BINARY_DIR}/scratch")

  # Decode graphs (D-090) on a GB10: a small GGML plan captured through the
  # launch context replays bit for bit, also after full swaps that page its
  # model back in at its pinned places; refused captures leave the context
  # as it was; a graph is one operation in its stream (RE-029).
  add_executable(cuda_graph_test cuda_graph_test.cc)
  target_include_directories(cuda_graph_test SYSTEM PRIVATE
                             "${JITLLM_SDK}/cuda/targets/${JITLLM_CUDA_TARGET_DIR}/include")
  target_link_libraries(cuda_graph_test PRIVATE jitllm_paged_node jitllm_ggml_kernels)
  jitllm_gpu_gtest(cuda_graph_test
                   ENVIRONMENT "JITLLM_TEST_SCRATCH=${CMAKE_CURRENT_BINARY_DIR}/scratch")

  # The engine's expert-slab pages (engine/paged_weights.h): a
  # host-only check, in the CUDA build that has the engine.
  add_executable(slab_layout_test slab_layout_test.cc)
  target_include_directories(slab_layout_test SYSTEM PRIVATE
                             "${JITLLM_SDK}/cuda/targets/${JITLLM_CUDA_TARGET_DIR}/include")
  target_link_libraries(slab_layout_test PRIVATE jitllm_engine)
  jitllm_gpu_gtest(slab_layout_test)

  add_executable(live_state_test live_state_test.cc)
  target_link_libraries(live_state_test PRIVATE jitllm_runtime jitllm_engine)
  jitllm_gpu_gtest(live_state_test
                   ENVIRONMENT "JITLLM_TEST_SCRATCH=${CMAKE_CURRENT_BINARY_DIR}/scratch")

  # Qwen3.8's n-gram rows read on demand (engine/ple_rows.h): the read
  # plan on the host, and the reads and the gather kernel on a GB10.
  add_executable(ple_rows_test ple_rows_test.cc)
  target_include_directories(ple_rows_test SYSTEM PRIVATE
                             "${JITLLM_SDK}/cuda/targets/${JITLLM_CUDA_TARGET_DIR}/include")
  target_link_libraries(ple_rows_test PRIVATE jitllm_engine jitllm_fake_providers)
  jitllm_gpu_gtest(ple_rows_test
                   ENVIRONMENT "JITLLM_TEST_SCRATCH=${CMAKE_CURRENT_BINARY_DIR}/scratch")

  add_executable(cuda_device_memory_test cuda_device_memory_test.cc)
  target_link_libraries(cuda_device_memory_test PRIVATE jitllm_cuda_device_memory jitllm_storage)
  jitllm_gpu_gtest(cuda_device_memory_test DISCRETE ALL
                   ENVIRONMENT "JITLLM_TEST_SCRATCH=${CMAKE_CURRENT_BINARY_DIR}/scratch")

  # The device runtime (providers/device_runtime.h), the engine's only way
  # to the device besides the providers and its kernels: opening the
  # device, its facts, copies and fills, timing marks and recorded work. It
  # links the paging kernels, whose CUDA units bring the CUDA runtime's
  # static library to the link, as the engine's kernels do.
  add_executable(device_runtime_test device_runtime_test.cc)
  target_include_directories(device_runtime_test SYSTEM PRIVATE
                             "${JITLLM_SDK}/cuda/targets/${JITLLM_CUDA_TARGET_DIR}/include")
  target_link_libraries(device_runtime_test PRIVATE jitllm_device_runtime jitllm_paging_kernels)
  jitllm_gpu_gtest(device_runtime_test DISCRETE ALL)
endif()

# The executed-plan recording's JSON lines (tests/support/plan_record.h),
# which docs/experiments/backend-proof-p2/plan_compare.py reads.
add_executable(plan_record_test plan_record_test.cc)
target_link_libraries(plan_record_test PRIVATE jitllm_plan_record jitllm_plan_record_sample)
jitllm_gtest(plan_record_test TEST_PREFIX unit. PROPERTIES LABELS unit)

# General JSON as untrusted input (base/json.h).
add_executable(json_test json_test.cc)
target_link_libraries(json_test PRIVATE jitllm_json jitllm_chat jitllm_base)
jitllm_gtest(json_test TEST_PREFIX unit. PROPERTIES LABELS unit)

# The tokenizer and chat fixtures and the pinned chat templates
# (tests/unit/data), copied into the build tree so that remote runs map them
# (JITLLM_TEST_DATA).
set(tokenizer_data "${CMAKE_CURRENT_BINARY_DIR}/data")
file(GLOB_RECURSE tokenizer_data_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}/data"
     CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/data/*.json"
     "${CMAKE_CURRENT_SOURCE_DIR}/data/*.jinja")
foreach(file IN LISTS tokenizer_data_files)
  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/data/${file}" "${tokenizer_data}/${file}" COPYONLY)
endforeach()

# Approved Gemma2 profile and actual padded-readable tensor binding only.
add_executable(gemma2_test gemma2_test.cc)
target_link_libraries(gemma2_test PRIVATE jitllm_model jitllm_artifact jitllm_json)
jitllm_gtest(gemma2_test TEST_PREFIX unit. PROPERTIES LABELS unit
             ENVIRONMENT "JITLLM_TEST_DATA=${tokenizer_data}")

# Approved Gemma3 profile, tensor storage, state/input and descriptor foundations.
add_executable(gemma3_test gemma3_test.cc)
target_link_libraries(gemma3_test PRIVATE jitllm_model jitllm_artifact jitllm_json)
jitllm_gtest(gemma3_test TEST_PREFIX unit. PROPERTIES LABELS unit
             ENVIRONMENT "JITLLM_TEST_DATA=${tokenizer_data}")

# Approved Gemma4 profiles and actual tensor contracts, independent-slot
# state and segment inputs. This foundation executes no model kernels.
add_executable(gemma4_test gemma4_test.cc)
target_link_libraries(gemma4_test PRIVATE jitllm_model jitllm_artifact jitllm_json)
jitllm_gtest(gemma4_test TEST_PREFIX unit. PROPERTIES LABELS unit
             ENVIRONMENT "JITLLM_TEST_DATA=${tokenizer_data}")

add_executable(gemma_profile_test gemma_profile_test.cc)
target_link_libraries(gemma_profile_test PRIVATE jitllm_runtime jitllm_model jitllm_json)
jitllm_gtest(gemma_profile_test TEST_PREFIX unit. PROPERTIES LABELS unit
             ENVIRONMENT "JITLLM_TEST_DATA=${tokenizer_data}")

add_executable(gemma2_graph_test gemma2_graph_test.cc)
target_link_libraries(gemma2_graph_test PRIVATE jitllm_ggml_tensors jitllm_artifact jitllm_json)
jitllm_gtest(gemma2_graph_test TEST_PREFIX unit. PROPERTIES LABELS unit
             ENVIRONMENT "JITLLM_TEST_DATA=${tokenizer_data}")
if(JITLLM_CUDA)
  add_executable(gemma2_plan_test gemma2_plan_test.cc)
  target_link_libraries(gemma2_plan_test PRIVATE jitllm_engine jitllm_json)
  jitllm_gtest(gemma2_plan_test TEST_PREFIX unit. PROPERTIES LABELS unit
               ENVIRONMENT "JITLLM_TEST_DATA=${tokenizer_data}")
endif()

add_executable(gemma3_graph_test gemma3_graph_test.cc)
target_link_libraries(gemma3_graph_test PRIVATE jitllm_ggml_tensors jitllm_artifact jitllm_json)
jitllm_gtest(gemma3_graph_test TEST_PREFIX unit. PROPERTIES LABELS unit
             ENVIRONMENT "JITLLM_TEST_DATA=${tokenizer_data}")
if(JITLLM_CUDA)
  add_executable(gemma2_checkpoint_gpu_test gemma2_checkpoint_gpu_test.cc)
  target_link_libraries(gemma2_checkpoint_gpu_test PRIVATE jitllm_engine)
  jitllm_gpu_gtest(gemma2_checkpoint_gpu_test MODELS)
  add_executable(gemma2_runner_test gemma2_runner_test.cc)
  target_link_libraries(gemma2_runner_test PRIVATE jitllm_engine)
  file(RELATIVE_PATH runtime "${CMAKE_CURRENT_BINARY_DIR}" "${PROJECT_BINARY_DIR}/cublas")
  target_link_options(gemma2_runner_test PRIVATE "LINKER:--enable-new-dtags,-rpath,$ORIGIN/${runtime}")
  jitllm_gtest(gemma2_runner_test TEST_PREFIX unit. PROPERTIES LABELS unit)
  add_executable(gemma3_serving_gpu_test gemma3_serving_gpu_test.cc)
  target_link_libraries(gemma3_serving_gpu_test PRIVATE jitllm_engine)
  jitllm_gpu_gtest(gemma3_serving_gpu_test MODELS)
  add_executable(gemma3_runner_test gemma3_runner_test.cc)
  target_link_libraries(gemma3_runner_test PRIVATE jitllm_engine)
  file(RELATIVE_PATH runtime "${CMAKE_CURRENT_BINARY_DIR}" "${PROJECT_BINARY_DIR}/cublas")
  target_link_options(gemma3_runner_test PRIVATE "LINKER:--enable-new-dtags,-rpath,$ORIGIN/${runtime}")
  jitllm_gtest(gemma3_runner_test TEST_PREFIX unit. PROPERTIES LABELS unit)
  add_executable(gemma3_plan_test gemma3_plan_test.cc)
  target_link_libraries(gemma3_plan_test PRIVATE jitllm_engine jitllm_json)
  jitllm_gtest(gemma3_plan_test TEST_PREFIX unit. PROPERTIES LABELS unit
               ENVIRONMENT "JITLLM_TEST_DATA=${tokenizer_data}")
endif()

add_executable(gemma4_graph_test gemma4_graph_test.cc)
add_executable(gemma4_greedy_test gemma4_greedy_test.cc)
target_link_libraries(gemma4_greedy_test PRIVATE jitllm_gemma4_greedy)
jitllm_gtest(gemma4_greedy_test TEST_PREFIX unit. PROPERTIES LABELS unit)

add_executable(gemma4_assistant_graph_test gemma4_assistant_graph_test.cc)
target_link_libraries(gemma4_assistant_graph_test PRIVATE jitllm_ggml_tensors jitllm_artifact jitllm_json)
jitllm_gtest(gemma4_assistant_graph_test TEST_PREFIX unit. PROPERTIES LABELS unit
             ENVIRONMENT "JITLLM_TEST_DATA=${tokenizer_data}")
target_link_libraries(gemma4_graph_test PRIVATE jitllm_ggml_tensors jitllm_artifact jitllm_json)
jitllm_gtest(gemma4_graph_test TEST_PREFIX unit. PROPERTIES LABELS unit
             ENVIRONMENT "JITLLM_TEST_DATA=${tokenizer_data}")
if(JITLLM_CUDA)
  add_executable(gemma4_exec_test gemma4_exec_test.cc)
  add_executable(gemma_norm_gpu_test gemma_norm_gpu_test.cc)
  target_include_directories(gemma_norm_gpu_test SYSTEM PRIVATE
      "${JITLLM_SDK}/cuda/targets/${JITLLM_CUDA_TARGET_DIR}/include")
  target_link_libraries(gemma_norm_gpu_test PRIVATE jitllm_ggml_kernels jitllm_cuda_device_memory)
  jitllm_gpu_gtest(gemma_norm_gpu_test)
  target_link_libraries(gemma4_exec_test PRIVATE jitllm_engine jitllm_json jitllm_cuda_device_memory)
  target_include_directories(gemma4_exec_test SYSTEM PRIVATE
      "${JITLLM_SDK}/cuda/targets/${JITLLM_CUDA_TARGET_DIR}/include")
  jitllm_gpu_gtest(gemma4_exec_test ENVIRONMENT "JITLLM_TEST_DATA=${tokenizer_data}")
  add_executable(gemma4_serving_gpu_test gemma4_serving_gpu_test.cc)
  target_link_libraries(gemma4_serving_gpu_test PRIVATE jitllm_runtime_serving)
  jitllm_gpu_gtest(gemma4_serving_gpu_test MODELS ENVIRONMENT "JITLLM_TEST_DATA=${tokenizer_data}")
  add_executable(plain_tokens_serving_gpu_test plain_tokens_serving_gpu_test.cc)
  target_link_libraries(plain_tokens_serving_gpu_test PRIVATE jitllm_runtime_serving)
  jitllm_gpu_gtest(plain_tokens_serving_gpu_test MODELS)
  add_executable(gemma_prefill_serving_gpu_test gemma_prefill_serving_gpu_test.cc)
  target_link_libraries(gemma_prefill_serving_gpu_test PRIVATE jitllm_runtime_serving)
  jitllm_gpu_gtest(gemma_prefill_serving_gpu_test MODELS)
  add_executable(gemma_joined_serving_gpu_test gemma_joined_serving_gpu_test.cc)
  target_link_libraries(gemma_joined_serving_gpu_test PRIVATE jitllm_runtime_serving)
  jitllm_gpu_gtest(gemma_joined_serving_gpu_test MODELS
                   ENVIRONMENT "JITLLM_TEST_DATA=${tokenizer_data}")
  add_executable(gemma4_assistant_gpu_test gemma4_assistant_gpu_test.cc)
  target_link_libraries(gemma4_assistant_gpu_test PRIVATE jitllm_engine)
  jitllm_gpu_gtest(gemma4_assistant_gpu_test MODELS)
  add_executable(gemma4_greedy_gpu_test gemma4_greedy_gpu_test.cc)
  target_link_libraries(gemma4_greedy_gpu_test PRIVATE jitllm_engine)
  jitllm_gpu_gtest(gemma4_greedy_gpu_test MODELS)
  add_executable(gemma4_verify_gpu_test gemma4_verify_gpu_test.cc)
  target_link_libraries(gemma4_verify_gpu_test PRIVATE jitllm_engine)
  jitllm_gpu_gtest(gemma4_verify_gpu_test MODELS)
  add_executable(gemma4_runner_gpu_test gemma4_runner_gpu_test.cc)
  target_link_libraries(gemma4_runner_gpu_test PRIVATE jitllm_engine)
  jitllm_gpu_gtest(gemma4_runner_gpu_test MODELS)
  add_executable(gemma4_runner_test gemma4_runner_test.cc)
  target_link_libraries(gemma4_runner_test PRIVATE jitllm_engine)
  file(RELATIVE_PATH runtime "${CMAKE_CURRENT_BINARY_DIR}" "${PROJECT_BINARY_DIR}/cublas")
  target_link_options(gemma4_runner_test PRIVATE "LINKER:--enable-new-dtags,-rpath,$ORIGIN/${runtime}")
  jitllm_gtest(gemma4_runner_test TEST_PREFIX unit. PROPERTIES LABELS unit)
  add_executable(gemma4_assistant_plan_test gemma4_assistant_plan_test.cc)
  target_link_libraries(gemma4_assistant_plan_test PRIVATE jitllm_engine jitllm_json)
  jitllm_gtest(gemma4_assistant_plan_test TEST_PREFIX unit. PROPERTIES LABELS unit
               ENVIRONMENT "JITLLM_TEST_DATA=${tokenizer_data}")
  add_executable(gemma_measurement_gpu_test gemma_measurement_gpu_test.cc)
  target_link_libraries(gemma_measurement_gpu_test PRIVATE jitllm_engine jitllm_json)
  jitllm_gpu_gtest(gemma_measurement_gpu_test
                   ENVIRONMENT "JITLLM_TEST_DATA=${tokenizer_data}")
  add_executable(gemma4_plan_test gemma4_plan_test.cc)
  target_link_libraries(gemma4_plan_test PRIVATE jitllm_engine jitllm_json)
  jitllm_gtest(gemma4_plan_test TEST_PREFIX unit. PROPERTIES LABELS unit
               ENVIRONMENT "JITLLM_TEST_DATA=${tokenizer_data}")
endif()

# The native tokenizer (M3) on synthetic vocabularies, in every profile.
add_executable(tokenizer_test tokenizer_test.cc)
target_link_libraries(tokenizer_test PRIVATE jitllm_tokenizer)
jitllm_gtest(tokenizer_test TEST_PREFIX unit. PROPERTIES LABELS unit)

# The chat renderers against their references' text (D-067), and JSON as
# Python prints it.
add_executable(chat_test chat_test.cc)
add_executable(turn_reuse_test turn_reuse_test.cc)
target_link_libraries(turn_reuse_test PRIVATE jitllm_runtime jitllm_chat)
jitllm_gtest(turn_reuse_test TEST_PREFIX unit. PROPERTIES LABELS unit)
target_link_libraries(chat_test PRIVATE jitllm_chat jitllm_json jitllm_base)
jitllm_gtest(chat_test TEST_PREFIX unit. PROPERTIES LABELS unit
             ENVIRONMENT "JITLLM_TEST_DATA=${tokenizer_data}")

# Python's str case operations (chat/pycase.h) against Python 3.12 over
# every code point, and against Jinja2's title filter.
add_executable(pycase_test pycase_test.cc)
target_link_libraries(pycase_test PRIVATE jitllm_chat jitllm_json jitllm_tokenizer jitllm_base)
jitllm_gtest(pycase_test TEST_PREFIX unit. PROPERTIES LABELS unit
             ENVIRONMENT "JITLLM_TEST_DATA=${tokenizer_data}")

# The Jinja-subset interpreter: snippets against transformers' text,
# provenance, and its bounds under hostile templates.
add_executable(jinja_test jinja_test.cc)
target_link_libraries(jinja_test PRIVATE jitllm_chat jitllm_json jitllm_base)
jitllm_gtest(jinja_test TEST_PREFIX unit. PROPERTIES LABELS unit
             ENVIRONMENT "JITLLM_TEST_DATA=${tokenizer_data}")

# Choosing a template's renderer (hash, probe, interpreter), interpreted
# rendering with its marked control tokens; and the real-template corpus
# against transformers (label `models`: the corpus lives on the Sparks).
add_executable(chat_template_test chat_template_test.cc)
target_link_libraries(chat_template_test PRIVATE jitllm_chat jitllm_json jitllm_base)
jitllm_gtest(chat_template_test TEST_PREFIX unit. PROPERTIES LABELS unit
             ENVIRONMENT "JITLLM_TEST_DATA=${tokenizer_data}")
# The actual approved Gemma2 template remains interpreted, including its
# system refusal and the end-of-turn stop from the kept GGUF metadata.
add_executable(gemma2_chat_assets_test gemma2_chat_assets_test.cc)
target_link_libraries(gemma2_chat_assets_test PRIVATE jitllm_runtime jitllm_chat jitllm_model jitllm_base)
jitllm_gtest(gemma2_chat_assets_test TEST_PREFIX unit. PROPERTIES LABELS "unit\\;models")

add_executable(chat_corpus_test chat_corpus_test.cc)
target_link_libraries(chat_corpus_test PRIVATE jitllm_chat jitllm_json jitllm_base)
jitllm_gtest(chat_corpus_test TEST_PREFIX unit. PROPERTIES LABELS "unit\\;models"
             ENVIRONMENT "JITLLM_TEST_DATA=${tokenizer_data}")

# The tokenizer and renderers against the references' token IDs on the real
# vocabularies, and NFC against UCD's NormalizationTest: label `models` too,
# since the files live on the Sparks; each test skips where they are absent.
add_executable(tokenizer_models_test tokenizer_models_test.cc)
target_link_libraries(tokenizer_models_test PRIVATE jitllm_chat jitllm_tokenizer jitllm_json jitllm_base)
jitllm_gtest(tokenizer_models_test TEST_PREFIX unit. PROPERTIES LABELS "unit\\;models"
             ENVIRONMENT "JITLLM_TEST_DATA=${tokenizer_data}")

# Greedy and seeded sampling (execution/sampling.h).
add_executable(adaptive_depth_test adaptive_depth_test.cc)
target_link_libraries(adaptive_depth_test PRIVATE jitllm_execution)
jitllm_gtest(adaptive_depth_test TEST_PREFIX unit. PROPERTIES LABELS unit)

add_executable(sampling_test sampling_test.cc)
target_link_libraries(sampling_test PRIVATE jitllm_sampling)
jitllm_gtest(sampling_test TEST_PREFIX unit. PROPERTIES LABELS unit)

# The implementation registry and plans (D-053), on fake declarations.
add_executable(execution_test execution_test.cc)
target_link_libraries(execution_test PRIVATE jitllm_execution)
jitllm_gtest(execution_test TEST_PREFIX unit. PROPERTIES LABELS unit)

# D-068's shape contracts: state representations and composed contexts
# (model_test), and request programs planned from them and driven through
# admission, the ledgers and the catalog over the fake device-memory
# provider (shapes_test), in every profile.
add_executable(model_test model_test.cc)
target_link_libraries(model_test PRIVATE jitllm_model)
jitllm_gtest(model_test TEST_PREFIX unit. PROPERTIES LABELS unit)

add_executable(shapes_test shapes_test.cc)
target_link_libraries(shapes_test PRIVATE jitllm_execution jitllm_scheduler jitllm_fake_providers)
jitllm_gtest(shapes_test TEST_PREFIX unit. PROPERTIES LABELS unit)

# GGML tensor descriptors over jitLLM memory, in every profile, and GGML's
# CUDA launchers and the cuBLAS path under jitLLM's launch context on a
# real device (label `gpu`, so only on a Spark). On a discrete GPU the
# build targets (`gpu-discrete`, D-082), the launchers' smoke: identical
# results across cudaMalloc, device VMM and host VMM, and the launch
# context's rules. The plan, operation and cuBLAS tests compare with GB10
# records and stay GB10-only.
add_executable(ggml_tensors_test ggml_tensors_test.cc)
target_link_libraries(ggml_tensors_test PRIVATE jitllm_ggml_tensors)
jitllm_gtest(ggml_tensors_test TEST_PREFIX unit. PROPERTIES LABELS unit)

# The Qwen2 adapter (model/qwen2.h): binding a synthetic index to the
# compiled-in profile, and a chunk's host-built inputs.
add_executable(qwen2_test qwen2_test.cc)
target_link_libraries(qwen2_test PRIVATE jitllm_model)
jitllm_gtest(qwen2_test TEST_PREFIX unit. PROPERTIES LABELS unit)

# The EXL3 Qwen2 adapter and its native operation plan (model/qwen2_exl3.h):
# binding synthetic resource lists, the recorded phase kinds, the order of
# operations, placement and the launch digest, and the pinned regions.
add_executable(qwen2_exl3_test qwen2_exl3_test.cc)
target_link_libraries(qwen2_exl3_test PRIVATE jitllm_model)
jitllm_gtest(qwen2_exl3_test TEST_PREFIX unit. PROPERTIES LABELS unit)

# The Qwen2 chunk graph, its plan with and without fusion, and activation
# placement (kernels/ggml/qwen2_graph.h, graph_plan.h).
add_executable(ggml_graph_test ggml_graph_test.cc)
target_link_libraries(ggml_graph_test PRIVATE jitllm_ggml_tensors jitllm_model)
jitllm_gtest(ggml_graph_test TEST_PREFIX unit. PROPERTIES LABELS unit)

# The DeepSeek V4 adapter: binding, state, chunk inputs, and its chunk graph's
# plan and placement (model/dsv4.h, kernels/ggml/dsv4_graph.h).
add_executable(dsv4_test dsv4_test.cc)
target_link_libraries(dsv4_test PRIVATE jitllm_ggml_tensors jitllm_model)
jitllm_gtest(dsv4_test TEST_PREFIX unit. PROPERTIES LABELS unit)

if(JITLLM_CUDA)
  add_executable(dsv4_prefill_plan_test dsv4_prefill_plan_test.cc)
  target_link_libraries(dsv4_prefill_plan_test PRIVATE jitllm_engine)
  file(RELATIVE_PATH runtime "${CMAKE_CURRENT_BINARY_DIR}" "${PROJECT_BINARY_DIR}/cublas")
  target_link_options(dsv4_prefill_plan_test PRIVATE "LINKER:--enable-new-dtags,-rpath,$ORIGIN/${runtime}")
  jitllm_gtest(dsv4_prefill_plan_test TEST_PREFIX unit. WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}" PROPERTIES LABELS unit)

  # A Qwen3.8 wave's composition (engine/qwen38_wave_plan.h), planned on
  # the host over a synthetic binding: no device, no model files.
  add_executable(qwen38_wave_plan_test qwen38_wave_plan_test.cc)
  target_link_libraries(qwen38_wave_plan_test PRIVATE jitllm_engine)
  file(RELATIVE_PATH runtime "${CMAKE_CURRENT_BINARY_DIR}" "${PROJECT_BINARY_DIR}/cublas")
  target_link_options(qwen38_wave_plan_test PRIVATE "LINKER:--enable-new-dtags,-rpath,$ORIGIN/${runtime}")
  jitllm_gtest(qwen38_wave_plan_test TEST_PREFIX unit. PROPERTIES LABELS unit)

  # The request cohort's host-only rules (engine/request_cohort.h).
  add_executable(request_cohort_test request_cohort_test.cc)
  target_link_libraries(request_cohort_test PRIVATE jitllm_engine)
  file(RELATIVE_PATH runtime "${CMAKE_CURRENT_BINARY_DIR}" "${PROJECT_BINARY_DIR}/cublas")
  target_link_options(request_cohort_test PRIVATE "LINKER:--enable-new-dtags,-rpath,$ORIGIN/${runtime}")
  jitllm_gtest(request_cohort_test TEST_PREFIX unit. PROPERTIES LABELS unit)

  # A runner's plan and graph caps (engine/planned.h), host-only.
  add_executable(plan_cache_test plan_cache_test.cc)
  target_link_libraries(plan_cache_test PRIVATE jitllm_engine)
  file(RELATIVE_PATH runtime "${CMAKE_CURRENT_BINARY_DIR}" "${PROJECT_BINARY_DIR}/cublas")
  target_link_options(plan_cache_test PRIVATE "LINKER:--enable-new-dtags,-rpath,$ORIGIN/${runtime}")
  jitllm_gtest(plan_cache_test TEST_PREFIX unit. PROPERTIES LABELS unit)

  # Releasing a partly created region (engine/paged_node.h ReleaseMapped)
  # over fake device memory, host-only.
  add_executable(release_mapped_test release_mapped_test.cc)
  target_link_libraries(release_mapped_test PRIVATE jitllm_engine jitllm_fake_providers)
  file(RELATIVE_PATH runtime "${CMAKE_CURRENT_BINARY_DIR}" "${PROJECT_BINARY_DIR}/cublas")
  target_link_options(release_mapped_test PRIVATE "LINKER:--enable-new-dtags,-rpath,$ORIGIN/${runtime}")
  jitllm_gtest(release_mapped_test TEST_PREFIX unit. PROPERTIES LABELS unit)
endif()

# DeepSeek V4's DSpark drafter and speculation's parts of the adapter and
# graph (model/dspark.h; D-092).
add_executable(dspark_test dspark_test.cc)
target_link_libraries(dspark_test PRIVATE jitllm_ggml_tensors jitllm_model)
jitllm_gtest(dspark_test TEST_PREFIX unit. PROPERTIES LABELS unit)

# The Qwen3.8 Flash Next adapter: binding, n-gram hash, state, chunk inputs
# and QSA tables, and its chunk graph's plan and placement (model/qwen38.h,
# kernels/ggml/qwen38_graph.h).
add_executable(qwen38_test qwen38_test.cc)
target_link_libraries(qwen38_test PRIVATE jitllm_ggml_tensors jitllm_model)
jitllm_gtest(qwen38_test TEST_PREFIX unit. PROPERTIES LABELS unit)

add_executable(ggml_validate_test ggml_validate_test.cc)
add_executable(gemma_norm_test gemma_norm_test.cc)
target_link_libraries(gemma_norm_test PRIVATE jitllm_ggml_tensors)
jitllm_gtest(gemma_norm_test TEST_PREFIX unit. PROPERTIES LABELS unit)
target_link_libraries(ggml_validate_test PRIVATE jitllm_ggml_tensors)
jitllm_gtest(ggml_validate_test TEST_PREFIX unit. PROPERTIES LABELS unit)

# The checks of the operations DeepSeek V4 Flash and Qwen3.8 Flash add
# (kernels/ggml/validate_ext.h), in every profile.
add_executable(ggml_ext_validate_test ggml_ext_validate_test.cc)
target_link_libraries(ggml_ext_validate_test PRIVATE jitllm_ggml_tensors)
jitllm_gtest(ggml_ext_validate_test TEST_PREFIX unit. PROPERTIES LABELS unit)

add_executable(gemma_moe_test gemma_moe_test.cc)
target_link_libraries(gemma_moe_test PRIVATE jitllm_ggml_tensors)
jitllm_gtest(gemma_moe_test TEST_PREFIX unit. PROPERTIES LABELS unit)

if(JITLLM_CUDA)
  add_executable(gemma_moe_gpu_test gemma_moe_gpu_test.cc)
  target_include_directories(gemma_moe_gpu_test SYSTEM PRIVATE
                            "${JITLLM_SDK}/cuda/targets/${JITLLM_CUDA_TARGET_DIR}/include")
  target_link_libraries(gemma_moe_gpu_test PRIVATE jitllm_ggml_kernels jitllm_cuda_device_memory)
  jitllm_gpu_gtest(gemma_moe_gpu_test)

  add_executable(ggml_kernels_test ggml_kernels_test.cc)
  target_include_directories(ggml_kernels_test SYSTEM PRIVATE
                             "${JITLLM_SDK}/cuda/targets/${JITLLM_CUDA_TARGET_DIR}/include")
  target_link_libraries(ggml_kernels_test PRIVATE jitllm_ggml_kernels jitllm_cuda_device_memory)
  jitllm_gpu_gtest(ggml_kernels_test
                   DISCRETE GgmlMemoryTest.HostAndDeviceVmmMatchCudaMallocBitForBit
                            GgmlKernelsTest.LaunchesUseTheProvidersContextAndStreamOrder
                            GgmlKernelsTest.AnUnknownSubmissionFaultsTheContext
                            GgmlKernelsTest.AnErrorPendingOnTheThreadRefusesOneContext
                            GgmlKernelsTest.WhatDoesNotFitIsRefusedAndALaunchErrorIsAFault)

  # The module's implementation identities follow every one of its source
  # files (src/kernels/ggml/module_digest.cmake). A script on the build
  # host, so it runs in every CUDA profile without a GPU.
  get_property(module_names GLOBAL PROPERTY JITLLM_GGML_MODULE_NAMES)
  add_test(NAME unit.GgmlModuleDigest
           COMMAND ${CMAKE_COMMAND} "-DBASE=${PROJECT_SOURCE_DIR}/src/kernels/ggml"
                   "-DNAMES=${module_names}"
                   "-DGENERATED=${PROJECT_BINARY_DIR}/src/kernels/ggml/module_digest.cc"
                   "-DWORK=${CMAKE_CURRENT_BINARY_DIR}/module_digest"
                   -P ${CMAKE_CURRENT_SOURCE_DIR}/ggml_module_digest_test.cmake)
  set_tests_properties(unit.GgmlModuleDigest PROPERTIES LABELS unit)

  # The remaining operations of the FP16 bridge's recorded plan. The test
  # records each launch through tests/support's launch recorder, which wraps
  # the CUDA runtime's entry points at link time, and compares it with the
  # recorded plan.
  add_executable(ggml_ops_test ggml_ops_test.cc)
  target_include_directories(ggml_ops_test SYSTEM PRIVATE
                             "${JITLLM_SDK}/cuda/targets/${JITLLM_CUDA_TARGET_DIR}/include")
  target_link_libraries(ggml_ops_test PRIVATE jitllm_ggml_kernels jitllm_cuda_device_memory
                                              jitllm_launch_recorder jitllm_plan_record_sample)
  jitllm_gpu_gtest(ggml_ops_test)

  # The GGML operations the native EXL3 plan adds (casts, the BF16
  # embedding, the forced vector attention): exact against the host or
  # close to FP64, launching what exl3-op-plan.json records.
  add_executable(ggml_exl3_ops_test ggml_exl3_ops_test.cc)
  target_include_directories(ggml_exl3_ops_test SYSTEM PRIVATE
                             "${JITLLM_SDK}/cuda/targets/${JITLLM_CUDA_TARGET_DIR}/include")
  target_link_libraries(ggml_exl3_ops_test PRIVATE jitllm_ggml_kernels jitllm_cuda_device_memory
                                                   jitllm_launch_recorder)
  # On a discrete GPU too (`gpu-discrete`, D-082): the over-read probe and
  # the host checks; the casts, embedding and attention hold to launches
  # recorded on a GB10.
  jitllm_gpu_gtest(ggml_exl3_ops_test
                   DISCRETE GgmlExl3OpsTest.OperationsReadAndWriteOnlyTheirTensors
                            GgmlExl3OpsTest.AttentionChecksRefuseWhatTheLauncherWouldNotTake)

  # The operations DeepSeek V4 Flash and Qwen3.8 Flash add
  # (kernels/ggml/ops_ext.h), against FP64 references within upstream's
  # test-backend-ops bounds, at small cases of the models' shapes.
  add_executable(ggml_ext_ops_test ggml_ext_ops_test.cc)
  target_include_directories(ggml_ext_ops_test SYSTEM PRIVATE
                             "${JITLLM_SDK}/cuda/targets/${JITLLM_CUDA_TARGET_DIR}/include")
  target_link_libraries(ggml_ext_ops_test PRIVATE jitllm_ggml_kernels jitllm_cuda_device_memory)
  jitllm_gpu_gtest(ggml_ext_ops_test)

  # Qwen3.8's formats: jitLLM's MXFP8 and NVFP4-row operations
  # (kernels/ggml/jitllm_ops.h), GGML's NVFP4 expert products over repacked
  # ModelOpt weights, and QSA's attention shape, against FP64 references.
  add_executable(qwen38_ops_test qwen38_ops_test.cc)
  target_include_directories(qwen38_ops_test SYSTEM PRIVATE
                             "${JITLLM_SDK}/cuda/targets/${JITLLM_CUDA_TARGET_DIR}/include")
  target_link_libraries(qwen38_ops_test PRIVATE jitllm_ggml_kernels jitllm_cuda_device_memory)
  jitllm_gpu_gtest(qwen38_ops_test)

  # A speculative verify's row-invariant products (D-092), the drafter's
  # argmax and the snapshot's range copies, against GGML's own one-column
  # launches bit for bit.
  add_executable(spec_rows_test spec_rows_test.cc)
  target_include_directories(spec_rows_test SYSTEM PRIVATE
                             "${JITLLM_SDK}/cuda/targets/${JITLLM_CUDA_TARGET_DIR}/include")
  target_link_libraries(spec_rows_test PRIVATE jitllm_ggml_kernels jitllm_cuda_device_memory)
  jitllm_gpu_gtest(spec_rows_test)

  # DeepSeek V4's fast plan (kernels/ggml/jitllm_ops.h, dsv4_fast.cu; D-085's
  # note): the vector products against GGML's MMVQ, every launch
  # configuration, the routing, combination, hyper-connection pre-mix and
  # compressor against FP64, and what their checks refuse.
  add_executable(dsv4_fast_test dsv4_fast_test.cc)
  target_include_directories(dsv4_fast_test SYSTEM PRIVATE
                             "${JITLLM_SDK}/cuda/targets/${JITLLM_CUDA_TARGET_DIR}/include")
  target_link_libraries(dsv4_fast_test PRIVATE jitllm_ggml_kernels jitllm_cuda_device_memory)
  jitllm_gpu_gtest(dsv4_fast_test)

  add_executable(qwen38_quant_head_test qwen38_quant_head_test.cc)
  target_include_directories(qwen38_quant_head_test SYSTEM PRIVATE
                             "${JITLLM_SDK}/cuda/targets/${JITLLM_CUDA_TARGET_DIR}/include")
  target_link_libraries(qwen38_quant_head_test PRIVATE jitllm_ggml_kernels jitllm_cuda_device_memory)
  jitllm_gpu_gtest(qwen38_quant_head_test MODELS)

  # The fast plan's ds4 prefill stage mechanisms (docs/experiments/
  # ds4-prefill-stages): each against the launches it replaces, byte for
  # byte, with its selection, refusals and registry declaration.
  add_executable(dsv4_stages_test dsv4_stages_test.cc)
  target_include_directories(dsv4_stages_test SYSTEM PRIVATE
                             "${JITLLM_SDK}/cuda/targets/${JITLLM_CUDA_TARGET_DIR}/include")
  target_link_libraries(dsv4_stages_test PRIVATE jitllm_ggml_kernels jitllm_cuda_device_memory)
  jitllm_gpu_gtest(dsv4_stages_test)

  # jitLLM's Qwen-Image kernels (kernels/image) against host references.
  add_executable(image_kernels_test image_kernels_test.cc)
  target_include_directories(image_kernels_test SYSTEM PRIVATE
                             "${JITLLM_SDK}/cuda/targets/${JITLLM_CUDA_TARGET_DIR}/include")
  target_link_libraries(image_kernels_test PRIVATE jitllm_image_kernels)
  jitllm_gpu_gtest(image_kernels_test)

  # jitLLM's fusions of Qwen3.8's hyper-connection and MoE-output nodes
  # (kernels/ggml/jitllm_ops.h) against the GGML nodes they replace, bit for
  # bit, and against FP64 references.
  add_executable(qwen38_fused_test qwen38_fused_test.cc)
  target_include_directories(qwen38_fused_test SYSTEM PRIVATE
                             "${JITLLM_SDK}/cuda/targets/${JITLLM_CUDA_TARGET_DIR}/include")
  target_link_libraries(qwen38_fused_test PRIVATE jitllm_ggml_kernels jitllm_cuda_device_memory)
  jitllm_gpu_gtest(qwen38_fused_test)

  # Qwen3.8's fast path (kernels/ggml/jitllm_ops.h, D-085): the MXFP8
  # tensor-core products, the fused hyper-connections, routing, Gated
  # DeltaNet and QSA kernels, against FP64 references and exact host
  # selections, and what their checks refuse.
  add_executable(qwen38_fast_test qwen38_fast_test.cc)
  target_include_directories(qwen38_fast_test SYSTEM PRIVATE
                             "${JITLLM_SDK}/cuda/targets/${JITLLM_CUDA_TARGET_DIR}/include")
  target_link_libraries(qwen38_fast_test PRIVATE jitllm_ggml_kernels jitllm_cuda_device_memory)
  jitllm_gpu_gtest(qwen38_fast_test)

  # Qwen3.8's routed experts over the CUTLASS layout (kernels/ggml/
  # moe_layout.h): the conversion, routing, CUTLASS's grouped GEMM, the
  # prefill path against GGML's MMQ path and FP64, decode's vector products.
  add_executable(qwen38_moe_test qwen38_moe_test.cc)
  target_include_directories(qwen38_moe_test SYSTEM PRIVATE
                             "${JITLLM_SDK}/cuda/targets/${JITLLM_CUDA_TARGET_DIR}/include")
  target_link_libraries(qwen38_moe_test PRIVATE jitllm_ggml_kernels jitllm_cuda_device_memory)
  jitllm_gpu_gtest(qwen38_moe_test)

  # A Qwen3.8 speculative verify's commit (kernels/ggml/qwen38_commit.h):
  # the kept rows' recurrence bit for bit the verify's own, the histories
  # exact, and the launcher's refusals.
  add_executable(qwen38_commit_test qwen38_commit_test.cc)
  target_include_directories(qwen38_commit_test SYSTEM PRIVATE
                             "${JITLLM_SDK}/cuda/targets/${JITLLM_CUDA_TARGET_DIR}/include")
  target_link_libraries(qwen38_commit_test PRIVATE jitllm_ggml_kernels jitllm_cuda_device_memory)
  jitllm_gpu_gtest(qwen38_commit_test)

  add_executable(ggml_cublas_test ggml_cublas_test.cc)
  target_include_directories(ggml_cublas_test SYSTEM PRIVATE
                             "${JITLLM_SDK}/cuda/targets/${JITLLM_CUDA_TARGET_DIR}/include")
  target_link_libraries(ggml_cublas_test PRIVATE jitllm_ggml_kernels jitllm_cuda_device_memory)
  jitllm_gpu_gtest(ggml_cublas_test)
endif()

# The v0 artifact reader, in every profile, against a corpus the M0
# prototype builds and judges at build time (artifact_corpus.py): its views
# of valid artifacts and its verdicts on about 190 mutated ones and 800
# seeded random mutations of their documents (with 8 divergence probes). The real
# v0 fixtures are opened where they are installed (a Spark) and skipped
# elsewhere.
set(artifact_corpus "${CMAKE_CURRENT_BINARY_DIR}/artifact-corpus")
set(artifact_prototype "${PROJECT_SOURCE_DIR}/docs/experiments/artifact-layout")
add_custom_command(
  OUTPUT "${artifact_corpus}/stamp"
  COMMAND "${JITLLM_PYTHON}" -B "${CMAKE_CURRENT_SOURCE_DIR}/artifact_corpus.py" "${artifact_corpus}"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/artifact_corpus.py" "${artifact_prototype}/layout.py"
          "${artifact_prototype}/test_layout.py"
  COMMENT "Building the artifact reader's corpus with the M0 prototype"
  VERBATIM)
add_custom_target(artifact_corpus ALL DEPENDS "${artifact_corpus}/stamp")
if(TARGET dsv4_prefill_plan_test)
  add_dependencies(dsv4_prefill_plan_test artifact_corpus)
endif()
add_executable(artifact_test artifact_test.cc)
target_link_libraries(artifact_test PRIVATE jitllm_artifact)
add_dependencies(artifact_test artifact_corpus)
# CTest runs it here, beside the corpus (remote runs map this directory).
jitllm_gtest(artifact_test TEST_PREFIX unit. WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
             PROPERTIES LABELS unit ENVIRONMENT "JITLLM_TEST_SCRATCH=${CMAKE_CURRENT_BINARY_DIR}/scratch")

# The kept GGUF metadata reader (artifact/gguf_metadata.h): values and each
# refusal on headers built in the test.
add_executable(gguf_metadata_test gguf_metadata_test.cc)
target_link_libraries(gguf_metadata_test PRIVATE jitllm_artifact)
jitllm_gtest(gguf_metadata_test TEST_PREFIX unit. PROPERTIES LABELS unit)

# The composition reader (D-089): each rule of the prototype's
# verify_composition on compositions built here, and the installed
# Qwen-Image-2.1 composition where the M3 store holds it (a Spark).
add_executable(composition_test composition_test.cc)
target_link_libraries(composition_test PRIVATE jitllm_artifact)
jitllm_gtest(composition_test TEST_PREFIX unit.
             PROPERTIES LABELS unit
                        ENVIRONMENT "JITLLM_TEST_SCRATCH=${CMAKE_CURRENT_BINARY_DIR}/scratch")

# The Qwen-Image-2.1 adapter (model/qwen_image.h): schedule, tensor lists
# and binding, the VAE's plan, rotary tables, sinusoid and pixels; the
# installed components bound where the M3 store holds them (a Spark).
add_executable(qwen_image_test qwen_image_test.cc)
target_link_libraries(qwen_image_test PRIVATE jitllm_model jitllm_artifact)
jitllm_gtest(qwen_image_test TEST_PREFIX unit. PROPERTIES LABELS unit)

# The locked ExLlamaV3 subset: the device context's sizes and the
# launchers' host checks in every profile; the kernels' tables and lookups
# link in every CUDA profile, and on a GB10 (label `gpu`) every kernel loads,
# the recorded ones use the P0 launch record's registers, and jitLLM's
# launchers run each path of the native linear. On a discrete GPU the build
# targets (`gpu-discrete`, D-082) every GEMM kernel loads from its own SASS
# and the launchers and the native plan run; what compares with GB10 records
# (registers, cuBLASLt pins, recorded launches, upstream's GB10 choices)
# stays GB10-only.
add_executable(exl3_context_test exl3_context_test.cc)
target_link_libraries(exl3_context_test PRIVATE jitllm_exl3)
jitllm_gtest(exl3_context_test TEST_PREFIX unit. PROPERTIES LABELS unit)

add_executable(exl3_validate_test exl3_validate_test.cc)
target_link_libraries(exl3_validate_test PRIVATE jitllm_exl3)
jitllm_gtest(exl3_validate_test TEST_PREFIX unit. PROPERTIES LABELS unit)

if(JITLLM_CUDA)
  # The tables and instances alone (jitllm_exl3_cuda), not the launchers:
  # this test runs in every CUDA profile, where CTest runs it to list its
  # tests, and so must load without cuBLAS (which the launchers link and
  # the workstation and qemu-user runs cannot find).
  add_executable(exl3_tables_test exl3_tables_test.cc)
  target_include_directories(exl3_tables_test SYSTEM PRIVATE
                             "${JITLLM_SDK}/cuda/targets/${JITLLM_CUDA_TARGET_DIR}/include")
  target_link_libraries(exl3_tables_test PRIVATE jitllm_exl3 jitllm_exl3_cuda)
  jitllm_gtest(exl3_tables_test TEST_PREFIX unit. PROPERTIES LABELS unit)

  add_executable(exl3_kernels_test exl3_kernels_test.cc)
  target_link_libraries(exl3_kernels_test PRIVATE jitllm_exl3_kernels)
  jitllm_gpu_gtest(exl3_kernels_test DISCRETE Exl3KernelsGpuTest.EveryKernelLoadsFromTheDevicesSass)

  # The launchers, the linear's paths and the registry on a GB10: operands
  # in cudaMalloc memory and device VMM alike, every operand flush against
  # unmapped VMM granules (the over-read probe), and every refusal.
  add_executable(exl3_linear_test exl3_linear_test.cc)
  target_link_libraries(exl3_linear_test PRIVATE jitllm_exl3_kernels jitllm_cuda_device_memory)
  # On a discrete GPU too (`gpu-discrete`, D-082), all but what runs the
  # reconstruction GEMM with its cuBLASLt algorithms pinned on a GB10.
  jitllm_gpu_gtest(exl3_linear_test
                   DISCRETE Exl3LinearTest.TheRegistryBindsEachPathToItsOwnCalls
                            Exl3LinearTest.RefusesGridsThatAreNotCoresident
                            Exl3LinearTest.NoTwoLiveContextsShareLockSlots
                            Exl3LinearTest.AFaultReturnsAsAFaultAndStopsTheContext
                            Exl3LinearTest.NoPackedKernelReadsOutsideItsOperandsAtTheFixturesShapes
                            Exl3LinearTest.TwoContextsAtTheCoresidentLimitBothComplete
                            Exl3ImplementationsTest.IdentitiesRecordTheLibraryAssertions)

  # One phase of the native EXL3 plan (kernels/exl3/qwen2.h) over zeroed
  # weights of the model's shapes: binding, refusals and a run.
  add_executable(exl3_qwen2_test exl3_qwen2_test.cc)
  target_link_libraries(exl3_qwen2_test PRIVATE jitllm_exl3_qwen2 jitllm_cuda_device_memory)
  # On a discrete GPU too (`gpu-discrete`, D-082), all but the check of
  # upstream's GEMV choice as the per-linear sweep saw it on a GB10.
  jitllm_gpu_gtest(exl3_qwen2_test
                   DISCRETE Exl3Qwen2Test.APrefillAndAStepBindAndRunOnOneStream
                            Exl3Qwen2Test.BindRefusesAMissingOrStaleImplementationAndShortMemory
                            Exl3Qwen2Test.TablesWhoseRewriteFailedPartwayRefuseTheMultiGemm)

  # The module's implementation identities follow every one of its source
  # files (src/kernels/exl3/module_digest.cmake), checked as the GGML
  # module's are.
  get_property(exl3_module_names GLOBAL PROPERTY JITLLM_EXL3_MODULE_NAMES)
  add_test(NAME unit.Exl3ModuleDigest
           COMMAND ${CMAKE_COMMAND} "-DBASE=${PROJECT_SOURCE_DIR}/src/kernels/exl3"
                   "-DNAMES=${exl3_module_names}"
                   "-DGENERATED=${PROJECT_BINARY_DIR}/src/kernels/exl3/module_digest.cc"
                   "-DWORK=${CMAKE_CURRENT_BINARY_DIR}/exl3_module_digest"
                   -P ${CMAKE_CURRENT_SOURCE_DIR}/ggml_module_digest_test.cmake)
  set_tests_properties(unit.Exl3ModuleDigest PROPERTIES LABELS unit)
endif()

add_executable(gemma_moe_fusion_test gemma_moe_fusion_test.cc)
target_link_libraries(gemma_moe_fusion_test PRIVATE jitllm_ggml_tensors jitllm_artifact jitllm_json)
jitllm_gtest(gemma_moe_fusion_test TEST_PREFIX unit. PROPERTIES LABELS unit
             ENVIRONMENT "JITLLM_TEST_DATA=${tokenizer_data}")

add_executable(gemma4_assistant_test gemma4_assistant_test.cc)
target_link_libraries(gemma4_assistant_test PRIVATE jitllm_model jitllm_artifact jitllm_json)
jitllm_gtest(gemma4_assistant_test TEST_PREFIX unit. PROPERTIES LABELS unit
             ENVIRONMENT "JITLLM_TEST_DATA=${tokenizer_data}")

add_executable(gemma4_assistant_artifact_test gemma4_assistant_artifact_test.cc)
target_link_libraries(gemma4_assistant_artifact_test PRIVATE jitllm_model jitllm_artifact jitllm_tokenizer)
jitllm_gtest(gemma4_assistant_artifact_test TEST_PREFIX unit. PROPERTIES LABELS "unit\\;models")
