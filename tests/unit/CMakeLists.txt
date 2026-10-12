# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0

# Unit tests of llmpalooza's modules, one GoogleTest executable per module
# (docs/architecture.md#testing). Every test carries the `unit` label.

add_executable(qwen38_vocab_test qwen38_vocab_test.cc)
target_link_libraries(qwen38_vocab_test PRIVATE llmp_draft_vocab_contract)
llmp_gtest(qwen38_vocab_test TEST_PREFIX unit. PROPERTIES LABELS unit)

add_executable(qwen38_reference_test qwen38_reference_test.cc)
target_link_libraries(qwen38_reference_test PRIVATE llmp_vocab_reference llmp_base)
llmp_gtest(qwen38_reference_test TEST_PREFIX unit. PROPERTIES LABELS unit
             ENVIRONMENT "LLMP_TEST_SCRATCH=${CMAKE_CURRENT_BINARY_DIR}/scratch")

add_executable(cli_test cli_test.cc)
target_link_libraries(cli_test PRIVATE llmp_cli llmp_surface_versions)
target_compile_definitions(cli_test PRIVATE "LLMP_TEST_PROJECT_VERSION=\"${PROJECT_VERSION}\"")
llmp_gtest(cli_test TEST_PREFIX unit. PROPERTIES LABELS unit
             ENVIRONMENT "LLMP_TEST_SCRATCH=${CMAKE_CURRENT_BINARY_DIR}/scratch")

add_executable(base_test base_test.cc)
target_link_libraries(base_test PRIVATE llmp_report llmp_base)
llmp_gtest(base_test TEST_PREFIX unit. PROPERTIES LABELS unit)

add_executable(platform_test platform_test.cc)
target_link_libraries(platform_test PRIVATE llmp_platform)
llmp_gtest(platform_test TEST_PREFIX unit. PROPERTIES LABELS unit
             ENVIRONMENT "LLMP_TEST_SCRATCH=${CMAKE_CURRENT_BINARY_DIR}/scratch")

# The CUDA provider, apart from its device queries, which need a GPU
# (smoke.doctor in tests/smoke).
if(LLMP_CUDA)
  add_executable(cuda_test cuda_test.cc)
  target_link_libraries(cuda_test PRIVATE llmp_cuda_probe)
  llmp_gtest(cuda_test TEST_PREFIX unit. PROPERTIES LABELS unit)
endif()

# The node configuration. Tests that need scratch trees (here and in
# platform_test) put them in the build tree, whose parents no other user
# shares, and remote runs map it (LLMP_TEST_SCRATCH).
add_executable(config_test config_test.cc)
target_link_libraries(config_test PRIVATE llmp_config llmp_platform)
llmp_gtest(config_test TEST_PREFIX unit. PROPERTIES LABELS unit
             ENVIRONMENT "LLMP_TEST_SCRATCH=${CMAKE_CURRENT_BINARY_DIR}/scratch")

# The runtime's startup steps and the process services it relies on, and a
# prefill's chunks.
add_executable(runtime_test runtime_test.cc)
target_link_libraries(runtime_test PRIVATE llmp_runtime llmp_platform)
llmp_gtest(runtime_test TEST_PREFIX unit. PROPERTIES LABELS unit
             ENVIRONMENT "LLMP_TEST_SCRATCH=${CMAKE_CURRENT_BINARY_DIR}/scratch")

# A model's settings in three layers (D-103): sources, precedence, refused
# and ignored overrides, the reasoning markers, the listing.
add_executable(model_settings_test model_settings_test.cc)
target_link_libraries(model_settings_test PRIVATE llmp_runtime llmp_config)
llmp_gtest(model_settings_test TEST_PREFIX unit. PROPERTIES LABELS unit)

# The calibration record (D-103): its format, key, atomic write and trusted
# read, and what a model's first uses measure.
add_executable(calibration_test calibration_test.cc)
target_link_libraries(calibration_test PRIVATE llmp_runtime)
llmp_gtest(calibration_test TEST_PREFIX unit. PROPERTIES LABELS unit
             ENVIRONMENT "LLMP_TEST_SCRATCH=${CMAKE_CURRENT_BINARY_DIR}/scratch")

# A chat cohort's state-capacity policy: who waits, fails, is preempted.
add_executable(cohort_capacity_test cohort_capacity_test.cc)
target_link_libraries(cohort_capacity_test PRIVATE llmp_runtime)
llmp_gtest(cohort_capacity_test TEST_PREFIX unit. PROPERTIES LABELS unit)

add_executable(cohort_schedule_test cohort_schedule_test.cc)
target_link_libraries(cohort_schedule_test PRIVATE llmp_runtime)
llmp_gtest(cohort_schedule_test TEST_PREFIX unit. PROPERTIES LABELS unit)

# When pressure on memory from outside has the runtime trim: its target,
# hysteresis and back-off, from readings of the test's own.
add_executable(pressure_trim_test pressure_trim_test.cc)
target_link_libraries(pressure_trim_test PRIVATE llmp_runtime)
llmp_gtest(pressure_trim_test TEST_PREFIX unit. PROPERTIES LABELS unit)

# D-102's hang ladder: what is and is not a hang, its rungs and its last
# resort, on a clock of the test's own.
add_executable(hang_ladder_test hang_ladder_test.cc)
target_link_libraries(hang_ladder_test PRIVATE llmp_runtime)
llmp_gtest(hang_ladder_test TEST_PREFIX unit. PROPERTIES LABELS unit)

# D-105's conversations kept across a restart, host-side: their owner-only
# files, their record's format and checks, and the keeper that writes
# records in the background.
add_executable(kept_state_test kept_state_test.cc)
target_link_libraries(kept_state_test PRIVATE llmp_runtime llmp_platform llmp_base)
llmp_gtest(kept_state_test TEST_PREFIX unit. PROPERTIES LABELS unit
             ENVIRONMENT "LLMP_TEST_SCRATCH=${CMAKE_CURRENT_BINARY_DIR}/scratch")

# Room for a swap's incoming model: reclaimed in whole extents and checked
# again, against a ledger whose charge moves in whole extents.
add_executable(swap_room_test swap_room_test.cc)
target_link_libraries(swap_room_test PRIVATE llmp_runtime)
llmp_gtest(swap_room_test TEST_PREFIX unit. PROPERTIES LABELS unit)

# The chat route (D-097): requests and their bounds, the bind resolution and
# Host guard, and the server over real loopback sockets with a fake backend.
add_executable(api_test api_test.cc)
target_link_libraries(api_test PRIVATE llmp_runtime llmp_json)
llmp_gtest(api_test TEST_PREFIX unit. PROPERTIES LABELS unit)

# The actual Llm loops with fake completed chunks; no GPU work. Serving's
# Llm implementation is currently linked only in CUDA profiles.
if(LLMP_CUDA)
  add_executable(llm_scores_test llm_scores_test.cc)
  target_link_libraries(llm_scores_test PRIVATE llmp_runtime_serving)
  file(RELATIVE_PATH runtime "${CMAKE_CURRENT_BINARY_DIR}" "${PROJECT_BINARY_DIR}/cublas")
  target_link_options(llm_scores_test PRIVATE "LINKER:--enable-new-dtags,-rpath,$ORIGIN/${runtime}")
  llmp_gtest(llm_scores_test TEST_PREFIX unit. PROPERTIES LABELS unit)
endif()

# The resource core: the catalog and the memory manager's policy, on no
# vendor SDK (D-026).
add_executable(catalog_test catalog_test.cc)
target_link_libraries(catalog_test PRIVATE llmp_catalog llmp_base)
llmp_gtest(catalog_test TEST_PREFIX unit. PROPERTIES LABELS unit)

add_executable(memory_test memory_test.cc)
target_link_libraries(memory_test PRIVATE llmp_memory llmp_catalog)
llmp_gtest(memory_test TEST_PREFIX unit. PROPERTIES LABELS unit)

add_executable(admission_test admission_test.cc)
target_link_libraries(admission_test PRIVATE llmp_scheduler)
llmp_gtest(admission_test TEST_PREFIX unit. PROPERTIES LABELS unit)

# D-048's lanes, completion board and task trees, with real threads.
add_executable(lanes_test lanes_test.cc)
target_link_libraries(lanes_test PRIVATE llmp_scheduler)
llmp_gtest(lanes_test TEST_PREFIX unit. PROPERTIES LABELS unit)

# The scheduler thread's turn loop and its lanes over the fake providers:
# deterministic turns, and threaded lost-wakeup and memory-ordering stress.
add_executable(scheduler_test scheduler_test.cc)
target_link_libraries(scheduler_test PRIVATE llmp_scheduler llmp_fake_providers)
llmp_gtest(scheduler_test TEST_PREFIX unit. PROPERTIES LABELS unit)

# The runtime wake (docs/experiments/runtime-wake/): the device lanes on
# their own threads over the fake device, which completes fences when the
# test chooses; lost wakeups, the backstop, the relay and shutdown. Its
# backstop and relay tests bound latencies in milliseconds, so the whole
# executable carries the `timing` label too (its tests are listed when
# CTest runs, so the label cannot pick single tests).
add_executable(wake_test wake_test.cc)
target_link_libraries(wake_test PRIVATE llmp_scheduler llmp_fake_providers)
llmp_gtest(wake_test TEST_PREFIX unit. PROPERTIES LABELS "unit\\;timing")

# The D-081 page-in path, its evictions and kernel jobs' leases, over the
# fake providers: deterministic turns and a threaded stress.
add_executable(pagein_test pagein_test.cc)
target_link_libraries(pagein_test PRIVATE llmp_scheduler llmp_fake_providers)
llmp_gtest(pagein_test TEST_PREFIX unit. PROPERTIES LABELS unit)

# BP-S3's acquisition (tests/support/paged_programs.h) over the fake
# providers: two models under one budget, each evicting only the other's
# weights.
add_executable(acquire_test acquire_test.cc)
target_link_libraries(acquire_test PRIVATE llmp_paged_programs llmp_scheduler
                                           llmp_fake_providers)
llmp_gtest(acquire_test TEST_PREFIX unit. PROPERTIES LABELS unit)

# A request's lease (M3's lease per request; paged_programs.h
# RequestProgram) over the fake providers: steps under one lease, eviction
# refused and swaps waiting while it is held, release at its end or after a
# cancelled request's step drains, and kept for good when a step is unproven.
add_executable(held_lease_test held_lease_test.cc)
target_link_libraries(held_lease_test PRIVATE llmp_paged_programs llmp_scheduler
                                               llmp_fake_providers)
llmp_gtest(held_lease_test TEST_PREFIX unit. PROPERTIES LABELS unit)

# Storage and whole reads, on the fake and on io_uring with a real file.
add_executable(storage_test storage_test.cc)
target_link_libraries(storage_test PRIVATE llmp_fake_providers llmp_storage llmp_platform)
llmp_gtest(storage_test TEST_PREFIX unit. PROPERTIES LABELS unit
             ENVIRONMENT "LLMP_TEST_SCRATCH=${CMAKE_CURRENT_BINARY_DIR}/scratch")

# The providers' shared rules, on the deterministic fakes.
add_executable(device_memory_test device_memory_test.cc)
target_link_libraries(device_memory_test PRIVATE llmp_fake_providers)
llmp_gtest(device_memory_test TEST_PREFIX unit. PROPERTIES LABELS unit)

# The retained-backing replay's trace reader, reference and candidate
# designs (benchmarks/retained_backing/), on the address-only fake.
add_executable(rb_replay_test rb_replay_test.cc)
target_link_libraries(rb_replay_test PRIVATE llmp_retained_backing)
llmp_gtest(rb_replay_test TEST_PREFIX unit. PROPERTIES LABELS unit
             ENVIRONMENT "LLMP_TEST_SCRATCH=${CMAKE_CURRENT_BINARY_DIR}/scratch")

add_executable(long_context_tasks_test long_context_tasks_test.cc)
target_link_libraries(long_context_tasks_test PRIVATE llmp_long_context_tasks)
llmp_gtest(long_context_tasks_test TEST_PREFIX unit. PROPERTIES LABELS unit)

add_executable(device_execution_test device_execution_test.cc)
target_link_libraries(device_execution_test PRIVATE llmp_fake_providers)
llmp_gtest(device_execution_test TEST_PREFIX unit. PROPERTIES LABELS unit)

# The CUDA device-memory and device-execution providers against a real
# device: label `gpu`, and `gpu-discrete` too, so on a Spark and on a
# discrete GPU the build targets (D-082).
if(LLMP_CUDA)
  add_executable(cuda_device_execution_test cuda_device_execution_test.cc)
  target_include_directories(cuda_device_execution_test SYSTEM PRIVATE
                             "${LLMP_SDK}/cuda/targets/${LLMP_CUDA_TARGET_DIR}/include")
  target_link_libraries(cuda_device_execution_test PRIVATE llmp_cuda_device_memory)
  llmp_gtest(cuda_device_execution_test TEST_PREFIX unit. PROPERTIES LABELS unit)

  # The scheduler and its lanes over io_uring, CUDA host VMM and CUDA
  # streams.
  add_executable(cuda_lanes_test cuda_lanes_test.cc)
  target_link_libraries(cuda_lanes_test PRIVATE llmp_scheduler llmp_cuda_device_memory)
  llmp_gpu_gtest(cuda_lanes_test DISCRETE ALL
                   ENVIRONMENT "LLMP_TEST_SCRATCH=${CMAKE_CURRENT_BINARY_DIR}/scratch")

  # The D-081 page-in path over io_uring, a host-VMM landing zone and
  # device VMM, the same on both classes (D-082). Each test runs both
  # instantiations (the VMM work on the device lane or its own).
  # Coalesced restores (CudaCoalescing, BP-P1) are verified on the GB10
  # only so far.
  # Write-back (CudaWriteBack) is verified on the GB10 only so far.
  add_executable(cuda_pagein_test cuda_pagein_test.cc)
  target_include_directories(cuda_pagein_test SYSTEM PRIVATE
                             "${LLMP_SDK}/cuda/targets/${LLMP_CUDA_TARGET_DIR}/include")
  target_link_libraries(cuda_pagein_test PRIVATE llmp_scheduler llmp_cuda_device_memory)
  llmp_gpu_gtest(cuda_pagein_test
                   DISCRETE "*/CudaPageIn.LandedLoadsReachDeviceVmmAndReloadIdentically/*"
                            "*/CudaPageIn.CancellingWithLoadsInFlightDrainsThem/*"
                            "*/CudaPageIn.CancellingDuringACopyWaitsForItsFence/*"
                   ENVIRONMENT "LLMP_TEST_SCRATCH=${CMAKE_CURRENT_BINARY_DIR}/scratch")

  # BP-S3's paged node (tests/support/paged_node.h) on a GB10: two models
  # on their own streams alternating under one budget, through one zone and
  # one shared workspace. The sleeping-steps test bounds lags in
  # milliseconds and CPU use in cores (label `timing`).
  add_executable(cuda_paged_node_test cuda_paged_node_test.cc)
  target_include_directories(cuda_paged_node_test SYSTEM PRIVATE
                             "${LLMP_SDK}/cuda/targets/${LLMP_CUDA_TARGET_DIR}/include")
  target_link_libraries(cuda_paged_node_test PRIVATE llmp_paged_node llmp_fake_providers)
  llmp_gpu_gtest(cuda_paged_node_test
                   TIMING "*/CudaPagedNodeSleep.RequestStepsSleepBetweenAndLoseNoCompletion/*"
                   ENVIRONMENT "LLMP_TEST_SCRATCH=${CMAKE_CURRENT_BINARY_DIR}/scratch")

  # Decode graphs (D-090) on a GB10: a small GGML plan captured through the
  # launch context replays bit for bit, also after full swaps that page its
  # model back in at its pinned places; refused captures leave the context
  # as it was; a graph is one operation in its stream (RE-029).
  add_executable(cuda_graph_test cuda_graph_test.cc)
  target_include_directories(cuda_graph_test SYSTEM PRIVATE
                             "${LLMP_SDK}/cuda/targets/${LLMP_CUDA_TARGET_DIR}/include")
  target_link_libraries(cuda_graph_test PRIVATE llmp_paged_node llmp_ggml_kernels)
  llmp_gpu_gtest(cuda_graph_test
                   ENVIRONMENT "LLMP_TEST_SCRATCH=${CMAKE_CURRENT_BINARY_DIR}/scratch")

  add_executable(live_state_test live_state_test.cc)
  target_link_libraries(live_state_test PRIVATE llmp_runtime llmp_engine)
  # Also the legacy runners' launch-context path (engine/live_state_launch.h).
  target_compile_definitions(live_state_test PRIVATE LLMP_LIVE_STATE_LAUNCH_TEST)
  llmp_gpu_gtest(live_state_test
                   ENVIRONMENT "LLMP_TEST_SCRATCH=${CMAKE_CURRENT_BINARY_DIR}/scratch")

  # Qwen3.8's n-gram rows read on demand (engine/ple_rows.h): the read
  # plan on the host, and the reads and the gather kernel on a GB10.
  add_executable(ple_rows_test ple_rows_test.cc)
  target_include_directories(ple_rows_test SYSTEM PRIVATE
                             "${LLMP_SDK}/cuda/targets/${LLMP_CUDA_TARGET_DIR}/include")
  target_link_libraries(ple_rows_test PRIVATE llmp_engine llmp_fake_providers)
  llmp_gpu_gtest(ple_rows_test
                   ENVIRONMENT "LLMP_TEST_SCRATCH=${CMAKE_CURRENT_BINARY_DIR}/scratch")

  add_executable(cuda_device_memory_test cuda_device_memory_test.cc)
  target_link_libraries(cuda_device_memory_test PRIVATE llmp_cuda_device_memory llmp_storage)
  llmp_gpu_gtest(cuda_device_memory_test DISCRETE ALL
                   ENVIRONMENT "LLMP_TEST_SCRATCH=${CMAKE_CURRENT_BINARY_DIR}/scratch")

  # The device runtime (providers/device_runtime.h), the engine's only way
  # to the device besides the providers and its kernels: opening the
  # device, its facts, copies and fills, timing marks and recorded work. It
  # links the paging kernels, whose CUDA units bring the CUDA runtime's
  # static library to the link, as the engine's kernels do.
  add_executable(device_runtime_test device_runtime_test.cc)
  target_include_directories(device_runtime_test SYSTEM PRIVATE
                             "${LLMP_SDK}/cuda/targets/${LLMP_CUDA_TARGET_DIR}/include")
  target_link_libraries(device_runtime_test PRIVATE llmp_device_runtime llmp_paging_kernels)
  llmp_gpu_gtest(device_runtime_test DISCRETE ALL)
endif()

# The executed-plan recording's JSON lines (tests/support/plan_record.h),
# which docs/experiments/backend-proof-p2/plan_compare.py reads.
add_executable(plan_record_test plan_record_test.cc)
target_link_libraries(plan_record_test PRIVATE llmp_plan_record llmp_plan_record_sample)
llmp_gtest(plan_record_test TEST_PREFIX unit. PROPERTIES LABELS unit)

# General JSON as untrusted input (base/json.h).
add_executable(json_test json_test.cc)
target_link_libraries(json_test PRIVATE llmp_json llmp_chat llmp_base)
llmp_gtest(json_test TEST_PREFIX unit. PROPERTIES LABELS unit)

# The tokenizer and chat fixtures and the pinned chat templates
# (tests/unit/data), copied into the build tree so that remote runs map them
# (LLMP_TEST_DATA).
set(tokenizer_data "${CMAKE_CURRENT_BINARY_DIR}/data")
file(GLOB_RECURSE tokenizer_data_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}/data"
     CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/data/*.json"
     "${CMAKE_CURRENT_SOURCE_DIR}/data/*.jinja")
foreach(file IN LISTS tokenizer_data_files)
  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/data/${file}" "${tokenizer_data}/${file}" COPYONLY)
endforeach()

# Shared exact host-mask interval construction and existing validation contract.
add_executable(host_mask_test host_mask_test.cc)
target_link_libraries(host_mask_test PRIVATE llmp_model)
llmp_gtest(host_mask_test TEST_PREFIX unit. PROPERTIES LABELS unit)

# Approved Gemma2 profile and actual padded-readable tensor binding only.
add_executable(gemma2_test gemma2_test.cc)
target_link_libraries(gemma2_test PRIVATE llmp_model llmp_artifact llmp_json)
llmp_gtest(gemma2_test TEST_PREFIX unit. PROPERTIES LABELS unit
             ENVIRONMENT "LLMP_TEST_DATA=${tokenizer_data}")

# Approved Gemma3 profile and actual tensor binding only.
add_executable(gemma3_test gemma3_test.cc)
target_link_libraries(gemma3_test PRIVATE llmp_model llmp_artifact llmp_json)
llmp_gtest(gemma3_test TEST_PREFIX unit. PROPERTIES LABELS unit
             ENVIRONMENT "LLMP_TEST_DATA=${tokenizer_data}")

# Approved Gemma4 profiles and actual tensor contracts.
add_executable(gemma4_test gemma4_test.cc)
target_link_libraries(gemma4_test PRIVATE llmp_model llmp_artifact llmp_json)
llmp_gtest(gemma4_test TEST_PREFIX unit. PROPERTIES LABELS unit
             ENVIRONMENT "LLMP_TEST_DATA=${tokenizer_data}")

add_executable(gemma_profile_test gemma_profile_test.cc)
target_link_libraries(gemma_profile_test PRIVATE llmp_runtime llmp_model llmp_json)
llmp_gtest(gemma_profile_test TEST_PREFIX unit. PROPERTIES LABELS unit
             ENVIRONMENT "LLMP_TEST_DATA=${tokenizer_data}")

if(LLMP_CUDA)
  add_executable(gemma_norm_gpu_test gemma_norm_gpu_test.cc)
  target_include_directories(gemma_norm_gpu_test SYSTEM PRIVATE
      "${LLMP_SDK}/cuda/targets/${LLMP_CUDA_TARGET_DIR}/include")
  target_link_libraries(gemma_norm_gpu_test PRIVATE llmp_ggml_kernels llmp_cuda_device_memory)
  llmp_gpu_gtest(gemma_norm_gpu_test)
  add_executable(gemma4_serving_gpu_test gemma4_serving_gpu_test.cc)
  target_link_libraries(gemma4_serving_gpu_test PRIVATE llmp_runtime_serving)
  llmp_gpu_gtest(gemma4_serving_gpu_test MODELS
                   ENVIRONMENT "LLMP_TEST_DATA=${tokenizer_data}"
                               "LLMP_TEST_SCRATCH=${CMAKE_CURRENT_BINARY_DIR}/scratch")
  add_executable(plain_tokens_serving_gpu_test plain_tokens_serving_gpu_test.cc)
  target_link_libraries(plain_tokens_serving_gpu_test PRIVATE llmp_runtime_serving)
  llmp_gpu_gtest(plain_tokens_serving_gpu_test MODELS
                   ENVIRONMENT "LLMP_TEST_SCRATCH=${CMAKE_CURRENT_BINARY_DIR}/scratch")
  # Default-off Qwen single-slot serving qualification.
  add_executable(qwen38_engine_serving_gpu_test qwen38_engine_serving_gpu_test.cc)
  target_link_libraries(qwen38_engine_serving_gpu_test PRIVATE llmp_runtime_serving)
  llmp_gpu_gtest(qwen38_engine_serving_gpu_test MODELS
                   ENVIRONMENT "LLMP_TEST_SCRATCH=${CMAKE_CURRENT_BINARY_DIR}/scratch")
  # Gemma 2 and Gemma 3 on the shared engine through the server: heads,
  # state, spill, restart, the near-8K context and the swap transactions.
  add_executable(gemma_engine_serving_gpu_test gemma_engine_serving_gpu_test.cc)
  target_link_libraries(gemma_engine_serving_gpu_test PRIVATE llmp_runtime_serving)
  llmp_gpu_gtest(gemma_engine_serving_gpu_test MODELS
                   ENVIRONMENT "LLMP_TEST_SCRATCH=${CMAKE_CURRENT_BINARY_DIR}/scratch")
  # Native v1 admission: approved Gemma3/26/31, compared with pinned v0 imports.
  add_executable(gemma_native_v1_serving_gpu_test gemma_native_v1_serving_gpu_test.cc)
  target_link_libraries(gemma_native_v1_serving_gpu_test PRIVATE llmp_runtime_serving)
  llmp_gpu_gtest(gemma_native_v1_serving_gpu_test MODELS
                   ENVIRONMENT "LLMP_TEST_SCRATCH=${CMAKE_CURRENT_BINARY_DIR}/scratch")
  # Weight files across real switches between Gemma 2 and Gemma 3: only the
  # resident model's shards open (engine/paged_node.h SetActive).
  add_executable(weight_files_serving_gpu_test weight_files_serving_gpu_test.cc)
  target_link_libraries(weight_files_serving_gpu_test PRIVATE llmp_runtime_serving llmp_artifact)
  llmp_gpu_gtest(weight_files_serving_gpu_test MODELS
                   ENVIRONMENT "LLMP_TEST_SCRATCH=${CMAKE_CURRENT_BINARY_DIR}/scratch")
endif()

# The native tokenizer (M3) on synthetic vocabularies, in every profile.
add_executable(tokenizer_test tokenizer_test.cc)
target_link_libraries(tokenizer_test PRIVATE llmp_tokenizer)
llmp_gtest(tokenizer_test TEST_PREFIX unit. PROPERTIES LABELS unit)

# The chat renderers against their references' text (D-067), and JSON as
# Python prints it.
add_executable(chat_test chat_test.cc)
add_executable(turn_reuse_test turn_reuse_test.cc)
target_link_libraries(turn_reuse_test PRIVATE llmp_runtime llmp_chat)
llmp_gtest(turn_reuse_test TEST_PREFIX unit. PROPERTIES LABELS unit)
target_link_libraries(chat_test PRIVATE llmp_chat llmp_json llmp_base)
llmp_gtest(chat_test TEST_PREFIX unit. PROPERTIES LABELS unit
             ENVIRONMENT "LLMP_TEST_DATA=${tokenizer_data}")

# Python's str case operations (chat/pycase.h) against Python 3.12 over
# every code point, and against Jinja2's title filter.
add_executable(pycase_test pycase_test.cc)
target_link_libraries(pycase_test PRIVATE llmp_chat llmp_json llmp_tokenizer llmp_base)
llmp_gtest(pycase_test TEST_PREFIX unit. PROPERTIES LABELS unit
             ENVIRONMENT "LLMP_TEST_DATA=${tokenizer_data}")

# The Jinja-subset interpreter: snippets against transformers' text,
# provenance, and its bounds under hostile templates.
add_executable(jinja_test jinja_test.cc)
target_link_libraries(jinja_test PRIVATE llmp_chat llmp_json llmp_base)
llmp_gtest(jinja_test TEST_PREFIX unit. PROPERTIES LABELS unit
             ENVIRONMENT "LLMP_TEST_DATA=${tokenizer_data}")

# Choosing a template's renderer (hash, probe, interpreter), interpreted
# rendering with its marked control tokens; and the real-template corpus
# against transformers (label `models`: the corpus lives on the Sparks).
add_executable(chat_template_test chat_template_test.cc)
target_link_libraries(chat_template_test PRIVATE llmp_chat llmp_json llmp_base)
llmp_gtest(chat_template_test TEST_PREFIX unit. PROPERTIES LABELS unit
             ENVIRONMENT "LLMP_TEST_DATA=${tokenizer_data}")
# The actual approved Gemma2 template remains interpreted, including its
# system refusal and the end-of-turn stop from the kept GGUF metadata.
add_executable(gemma2_chat_assets_test gemma2_chat_assets_test.cc)
target_link_libraries(gemma2_chat_assets_test PRIVATE llmp_runtime llmp_chat llmp_model llmp_base)
llmp_gtest(gemma2_chat_assets_test TEST_PREFIX unit. PROPERTIES LABELS "unit\\;models")

add_executable(chat_corpus_test chat_corpus_test.cc)
target_link_libraries(chat_corpus_test PRIVATE llmp_chat llmp_json llmp_base)
llmp_gtest(chat_corpus_test TEST_PREFIX unit. PROPERTIES LABELS "unit\\;models"
             ENVIRONMENT "LLMP_TEST_DATA=${tokenizer_data}")

# The tokenizer and renderers against the references' token IDs on the real
# vocabularies, and NFC against UCD's NormalizationTest: label `models` too,
# since the files live on the Sparks; each test skips where they are absent.
add_executable(tokenizer_models_test tokenizer_models_test.cc)
target_link_libraries(tokenizer_models_test PRIVATE llmp_chat llmp_tokenizer llmp_json llmp_base)
llmp_gtest(tokenizer_models_test TEST_PREFIX unit. PROPERTIES LABELS "unit\\;models"
             ENVIRONMENT "LLMP_TEST_DATA=${tokenizer_data}")

# Greedy and seeded sampling (execution/sampling.h).
add_executable(adaptive_depth_test adaptive_depth_test.cc)
target_link_libraries(adaptive_depth_test PRIVATE llmp_execution)
llmp_gtest(adaptive_depth_test TEST_PREFIX unit. PROPERTIES LABELS unit)

add_executable(sampling_test sampling_test.cc)
target_link_libraries(sampling_test PRIVATE llmp_sampling)
llmp_gtest(sampling_test TEST_PREFIX unit. PROPERTIES LABELS unit)

# The implementation registry and plans (D-053), on fake declarations, and
# the capabilities selection reads (D-108).
add_executable(execution_test execution_test.cc)
target_link_libraries(execution_test PRIVATE llmp_execution)
llmp_gtest(execution_test TEST_PREFIX unit. PROPERTIES LABELS unit)

# Tensor formats (D-108): encodings, layouts and the size rule.
add_executable(format_test format_test.cc)
target_link_libraries(format_test PRIVATE llmp_format)
llmp_gtest(format_test TEST_PREFIX unit. PROPERTIES LABELS unit)

# The CPU reference decoders (D-108), against GGML's dequantization and
# hand-computed values; the Qwen3.8 imports' experts where the M3 store
# holds them (a Spark).
add_executable(reference_decode_test reference_decode_test.cc)
target_link_libraries(reference_decode_test PRIVATE llmp_reference_kernels llmp_artifact
                                                    llmp_ggml_base)
llmp_gtest(reference_decode_test TEST_PREFIX unit. PROPERTIES LABELS unit)

# The device decoders' geometry (kernels/decode/operand.h) against the
# format layer's, in every profile.
add_executable(decode_test decode_test.cc)
target_link_libraries(decode_test PRIVATE llmp_format)
llmp_gtest(decode_test TEST_PREFIX unit. PROPERTIES LABELS unit)

# The graph IR (D-107): the builder's shape rules, the verifier, the
# canonical digest and the decompositions' structure.
add_executable(ir_test ir_test.cc)
target_link_libraries(ir_test PRIVATE llmp_ir)
llmp_gtest(ir_test TEST_PREFIX unit. PROPERTIES LABELS unit)

# The reference executor of op set v1 against hand values and independent
# formulas, composite ops against their decompositions, simulated devices.
add_executable(reference_exec_test reference_exec_test.cc)
target_link_libraries(reference_exec_test PRIVATE llmp_reference_kernels llmp_ir llmp_sampling)
llmp_gtest(reference_exec_test TEST_PREFIX unit. PROPERTIES LABELS unit)

# D-068's shape contracts: state representations and composed contexts
# (model_test), and request programs planned from them and driven through
# admission, the ledgers and the catalog over the fake device-memory
# provider (shapes_test), in every profile.
add_executable(model_test model_test.cc)
target_link_libraries(model_test PRIVATE llmp_model)
llmp_gtest(model_test TEST_PREFIX unit. PROPERTIES LABELS unit)

# State contracts (model/state_contract.h): each family's state reproduced
# byte for byte from contracts, a verify's saves, representations and the
# kept layout's identity.
add_executable(state_contract_test state_contract_test.cc)
target_link_libraries(state_contract_test PRIVATE llmp_model)
llmp_gtest(state_contract_test TEST_PREFIX unit. PROPERTIES LABELS unit)

add_executable(draft_progress_test draft_progress_test.cc)
target_link_libraries(draft_progress_test PRIVATE llmp_model)
llmp_gtest(draft_progress_test TEST_PREFIX unit. PROPERTIES LABELS unit)

add_executable(draft_progress_ledger_test draft_progress_ledger_test.cc)
target_link_libraries(draft_progress_ledger_test PRIVATE llmp_engine_core llmp_fake_device_backend)
llmp_gtest(draft_progress_ledger_test TEST_PREFIX unit. PROPERTIES LABELS unit)

add_executable(shapes_test shapes_test.cc)
target_link_libraries(shapes_test PRIVATE llmp_execution llmp_scheduler llmp_fake_providers)
llmp_gtest(shapes_test TEST_PREFIX unit. PROPERTIES LABELS unit)

# GGML tensor descriptors over llmpalooza memory, in every profile, and GGML's
# CUDA launchers and the cuBLAS path under llmpalooza's launch context on a
# real device (label `gpu`, so only on a Spark). On a discrete GPU the
# build targets (`gpu-discrete`, D-082), the launchers' smoke: identical
# results across cudaMalloc, device VMM and host VMM, and the launch
# context's rules. The plan, operation and cuBLAS tests compare with GB10
# records and stay GB10-only.
add_executable(ggml_tensors_test ggml_tensors_test.cc)
target_link_libraries(ggml_tensors_test PRIVATE llmp_ggml_tensors)
llmp_gtest(ggml_tensors_test TEST_PREFIX unit. PROPERTIES LABELS unit)

# The Qwen2 adapter (model/qwen2.h): binding a synthetic index to the
# compiled-in profile, and a chunk's host-built inputs.
add_executable(qwen2_test qwen2_test.cc)
target_link_libraries(qwen2_test PRIVATE llmp_model)
llmp_gtest(qwen2_test TEST_PREFIX unit. PROPERTIES LABELS unit)

# Immutable EXL3 source profile and resource binding (model/qwen2_exl3.h).
add_executable(qwen2_exl3_test qwen2_exl3_test.cc)
target_link_libraries(qwen2_exl3_test PRIVATE llmp_model)
llmp_gtest(qwen2_exl3_test TEST_PREFIX unit. PROPERTIES LABELS unit)

# Generic graph planning, activation placement and lane/KV controls.
add_executable(ggml_graph_test ggml_graph_test.cc)
target_link_libraries(ggml_graph_test PRIVATE llmp_ggml_tensors)
llmp_gtest(ggml_graph_test TEST_PREFIX unit. PROPERTIES LABELS unit)

# The DeepSeek V4 adapter: binding, state, chunk inputs, and its chunk graph's
# plan and placement (model/dsv4.h, kernels/ggml/dsv4_graph.h).
add_executable(dsv4_test dsv4_test.cc)
target_link_libraries(dsv4_test PRIVATE llmp_ggml_tensors llmp_model)
llmp_gtest(dsv4_test TEST_PREFIX unit. PROPERTIES LABELS unit)

if(LLMP_CUDA)
  add_executable(dsv4_prefill_plan_test dsv4_prefill_plan_test.cc)
  target_link_libraries(dsv4_prefill_plan_test PRIVATE llmp_engine)
  file(RELATIVE_PATH runtime "${CMAKE_CURRENT_BINARY_DIR}" "${PROJECT_BINARY_DIR}/cublas")
  target_link_options(dsv4_prefill_plan_test PRIVATE "LINKER:--enable-new-dtags,-rpath,$ORIGIN/${runtime}")
  llmp_gtest(dsv4_prefill_plan_test TEST_PREFIX unit. WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}" PROPERTIES LABELS unit)

  # A Qwen3.8 wave's composition (engine/qwen38_wave_plan.h), planned on
  # the host over a synthetic binding: no device, no model files.
  add_executable(qwen38_wave_plan_test qwen38_wave_plan_test.cc)
  target_link_libraries(qwen38_wave_plan_test PRIVATE llmp_engine)
  file(RELATIVE_PATH runtime "${CMAKE_CURRENT_BINARY_DIR}" "${PROJECT_BINARY_DIR}/cublas")
  target_link_options(qwen38_wave_plan_test PRIVATE "LINKER:--enable-new-dtags,-rpath,$ORIGIN/${runtime}")
  llmp_gtest(qwen38_wave_plan_test TEST_PREFIX unit. PROPERTIES LABELS unit)

  # A runner's plan and graph caps (engine/planned.h), host-only.
  add_executable(plan_cache_test plan_cache_test.cc)
  target_link_libraries(plan_cache_test PRIVATE llmp_engine)
  file(RELATIVE_PATH runtime "${CMAKE_CURRENT_BINARY_DIR}" "${PROJECT_BINARY_DIR}/cublas")
  target_link_options(plan_cache_test PRIVATE "LINKER:--enable-new-dtags,-rpath,$ORIGIN/${runtime}")
  llmp_gtest(plan_cache_test TEST_PREFIX unit. PROPERTIES LABELS unit)
endif()

# The engine core (engine/CMakeLists.txt) on the fake device backend, in
# every profile: the device runtime over host memory and the paging
# kernels as host loops (providers/fake/fake_device_runtime.h,
# kernels/paging/host_paging.cc), the core's live state on a paged node
# over them (live_state_test.cc, as on a GB10 above), the request cohort's
# host-only rules, releasing a partly created region (paged_node.h
# ReleaseMapped) and the expert slab's pages (paged_weights.h LayOutSlab).
add_executable(fake_device_runtime_test fake_device_runtime_test.cc)
target_link_libraries(fake_device_runtime_test PRIVATE llmp_fake_device_backend)
llmp_gtest(fake_device_runtime_test TEST_PREFIX unit. PROPERTIES LABELS unit)

add_executable(live_state_fake_test live_state_test.cc)
target_link_libraries(live_state_fake_test PRIVATE llmp_runtime llmp_engine_core
                                                   llmp_fake_device_backend)
llmp_gtest(live_state_fake_test TEST_PREFIX unit.fake. PROPERTIES LABELS unit
             ENVIRONMENT "LLMP_TEST_SCRATCH=${CMAKE_CURRENT_BINARY_DIR}/scratch")

# The shared slot lifecycle with a target and two drafters' state.
add_executable(slot_lifecycle_test slot_lifecycle_test.cc)
target_link_libraries(slot_lifecycle_test PRIVATE llmp_engine_core llmp_fake_device_backend
                                                  llmp_platform)
llmp_gtest(slot_lifecycle_test TEST_PREFIX unit.fake. PROPERTIES LABELS unit
             ENVIRONMENT "LLMP_TEST_SCRATCH=${CMAKE_CURRENT_BINARY_DIR}/scratch")

# The one model runner, its wave composer and prefill orchestrator over a
# fake step source (tests/support/fake_step_source.h).
add_executable(model_runner_test model_runner_test.cc)
target_link_libraries(model_runner_test PRIVATE llmp_engine_core llmp_fake_step_source
                                                llmp_fake_device_backend llmp_platform)
llmp_gtest(model_runner_test TEST_PREFIX unit.fake. PROPERTIES LABELS unit
             ENVIRONMENT "LLMP_TEST_SCRATCH=${CMAKE_CURRENT_BINARY_DIR}/scratch")

# Decoding modes with two drafters (a cascade and alternatives), verify,
# accept and rollback, over the same fake step source.
add_executable(decoding_test decoding_test.cc)
target_link_libraries(decoding_test PRIVATE llmp_engine_core llmp_fake_step_source
                                            llmp_fake_device_backend llmp_platform)
llmp_gtest(decoding_test TEST_PREFIX unit.fake. PROPERTIES LABELS unit
             ENVIRONMENT "LLMP_TEST_SCRATCH=${CMAKE_CURRENT_BINARY_DIR}/scratch")

# An anchored drafter over its target's frozen state and tapped features
# (Gemma 4's assistant's decoding), the features' sources and the computed
# inputs the runner stages, over the same fake step source.
add_executable(draft_companions_test draft_companions_test.cc)
target_link_libraries(draft_companions_test PRIVATE llmp_engine_core llmp_fake_step_source
                                                   llmp_fake_device_backend llmp_platform)
llmp_gtest(draft_companions_test TEST_PREFIX unit.fake. PROPERTIES LABELS unit
             ENVIRONMENT "LLMP_TEST_SCRATCH=${CMAKE_CURRENT_BINARY_DIR}/scratch")

add_executable(anchored_drafter_test anchored_drafter_test.cc)
target_link_libraries(anchored_drafter_test PRIVATE llmp_engine_core llmp_fake_step_source
                                                    llmp_fake_device_backend llmp_platform)
llmp_gtest(anchored_drafter_test TEST_PREFIX unit.fake. PROPERTIES LABELS unit
             ENVIRONMENT "LLMP_TEST_SCRATCH=${CMAKE_CURRENT_BINARY_DIR}/scratch")

# The LLM serving adapter over the shared runner (runtime/engine_llm.h):
# Llm's conversations, waves, speculation with two drafters, spills, turn
# checkpoints, recovery and kept conversations in a file a stage (D-105's
# record version 2), over the same fake step source.
add_executable(execution_contract_test execution_contract_test.cc)
target_link_libraries(execution_contract_test PRIVATE llmp_engine_core llmp_warnings)
llmp_gtest(execution_contract_test TEST_PREFIX unit. PROPERTIES LABELS unit)

add_executable(snapshot_store_test snapshot_store_test.cc)
target_link_libraries(snapshot_store_test PRIVATE llmp_runtime_llm llmp_fake_device_backend
                                                    llmp_warnings)
llmp_gtest(snapshot_store_test TEST_PREFIX unit.fake. PROPERTIES LABELS unit)

add_executable(engine_llm_test engine_llm_test.cc)
target_link_libraries(engine_llm_test PRIVATE llmp_runtime_llm llmp_engine_core
                                              llmp_fake_step_source llmp_fake_device_backend
                                              llmp_platform)
llmp_gtest(engine_llm_test TEST_PREFIX unit.fake. PROPERTIES LABELS unit
             ENVIRONMENT "LLMP_TEST_SCRATCH=${CMAKE_CURRENT_BINARY_DIR}/scratch")

# Weight files open only while their model is active (engine/paged_node.h
# SetActive, scheduler/files.h): copies of the artifact corpus's tiny golden
# paged on the fake device backend, the process's descriptors counted.
add_executable(weight_files_test weight_files_test.cc)
target_link_libraries(weight_files_test PRIVATE llmp_engine_core llmp_fake_device_backend
                                                llmp_platform)
add_dependencies(weight_files_test artifact_corpus)
llmp_gtest(weight_files_test TEST_PREFIX unit.fake. WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
             PROPERTIES LABELS unit
             ENVIRONMENT "LLMP_TEST_SCRATCH=${CMAKE_CURRENT_BINARY_DIR}/scratch")

add_executable(request_cohort_test request_cohort_test.cc)
target_link_libraries(request_cohort_test PRIVATE llmp_engine_core llmp_fake_device_backend)
llmp_gtest(request_cohort_test TEST_PREFIX unit. PROPERTIES LABELS unit)

add_executable(release_mapped_test release_mapped_test.cc)
target_link_libraries(release_mapped_test PRIVATE llmp_engine_core llmp_fake_device_backend)
llmp_gtest(release_mapped_test TEST_PREFIX unit. PROPERTIES LABELS unit)

add_executable(slab_layout_test slab_layout_test.cc)
target_link_libraries(slab_layout_test PRIVATE llmp_engine_core llmp_fake_device_backend)
llmp_gtest(slab_layout_test TEST_PREFIX unit. PROPERTIES LABELS unit)

# DeepSeek V4's DSpark drafter and speculation's parts of the adapter and
# graph (model/dspark.h; D-092).
add_executable(dspark_test dspark_test.cc)
target_link_libraries(dspark_test PRIVATE llmp_ggml_tensors llmp_model)
llmp_gtest(dspark_test TEST_PREFIX unit. PROPERTIES LABELS unit)

# The Qwen3.8 Flash Next adapter: binding, n-gram hash, state, chunk inputs
# and QSA tables, and its chunk graph's plan and placement (model/qwen38.h,
# kernels/ggml/qwen38_graph.h).
add_executable(qwen38_test qwen38_test.cc)
target_link_libraries(qwen38_test PRIVATE llmp_ggml_tensors llmp_model)
llmp_gtest(qwen38_test TEST_PREFIX unit. PROPERTIES LABELS unit)

add_executable(ggml_validate_test ggml_validate_test.cc)
add_executable(gemma_norm_test gemma_norm_test.cc)
target_link_libraries(gemma_norm_test PRIVATE llmp_ggml_tensors)
llmp_gtest(gemma_norm_test TEST_PREFIX unit. PROPERTIES LABELS unit)
target_link_libraries(ggml_validate_test PRIVATE llmp_ggml_tensors)
llmp_gtest(ggml_validate_test TEST_PREFIX unit. PROPERTIES LABELS unit)

# The checks of the operations DeepSeek V4 Flash and Qwen3.8 Flash add
# (kernels/ggml/validate_ext.h), in every profile.
add_executable(ggml_ext_validate_test ggml_ext_validate_test.cc)
target_link_libraries(ggml_ext_validate_test PRIVATE llmp_ggml_tensors)
llmp_gtest(ggml_ext_validate_test TEST_PREFIX unit. PROPERTIES LABELS unit)

add_executable(gemma_moe_test gemma_moe_test.cc)
target_link_libraries(gemma_moe_test PRIVATE llmp_ggml_tensors)
llmp_gtest(gemma_moe_test TEST_PREFIX unit. PROPERTIES LABELS unit)

if(LLMP_CUDA)
  add_executable(gemma_moe_gpu_test gemma_moe_gpu_test.cc)
  target_include_directories(gemma_moe_gpu_test SYSTEM PRIVATE
                            "${LLMP_SDK}/cuda/targets/${LLMP_CUDA_TARGET_DIR}/include")
  target_link_libraries(gemma_moe_gpu_test PRIVATE llmp_ggml_kernels llmp_cuda_device_memory)
  llmp_gpu_gtest(gemma_moe_gpu_test)

  add_executable(ggml_kernels_test ggml_kernels_test.cc)
  target_include_directories(ggml_kernels_test SYSTEM PRIVATE
                             "${LLMP_SDK}/cuda/targets/${LLMP_CUDA_TARGET_DIR}/include")
  target_link_libraries(ggml_kernels_test PRIVATE llmp_ggml_kernels llmp_cuda_device_memory)
  llmp_gpu_gtest(ggml_kernels_test
                   DISCRETE GgmlMemoryTest.HostAndDeviceVmmMatchCudaMallocBitForBit
                            GgmlKernelsTest.LaunchesUseTheProvidersContextAndStreamOrder
                            GgmlKernelsTest.AnUnknownSubmissionFaultsTheContext
                            GgmlKernelsTest.AnErrorPendingOnTheThreadRefusesOneContext
                            GgmlKernelsTest.WhatDoesNotFitIsRefusedAndALaunchErrorIsAFault)

  # The module's implementation identities follow every one of its source
  # files (src/kernels/ggml/module_digest.cmake). A script on the build
  # host, so it runs in every CUDA profile without a GPU.
  get_property(module_names GLOBAL PROPERTY LLMP_GGML_MODULE_NAMES)
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
                             "${LLMP_SDK}/cuda/targets/${LLMP_CUDA_TARGET_DIR}/include")
  target_link_libraries(ggml_ops_test PRIVATE llmp_ggml_kernels llmp_cuda_device_memory
                                              llmp_launch_recorder llmp_plan_record_sample)
  llmp_gpu_gtest(ggml_ops_test)

  # The GGML operations the native EXL3 plan adds (casts, the BF16
  # embedding, the forced vector attention): exact against the host or
  # close to FP64, launching what exl3-op-plan.json records.
  add_executable(ggml_exl3_ops_test ggml_exl3_ops_test.cc)
  target_include_directories(ggml_exl3_ops_test SYSTEM PRIVATE
                             "${LLMP_SDK}/cuda/targets/${LLMP_CUDA_TARGET_DIR}/include")
  target_link_libraries(ggml_exl3_ops_test PRIVATE llmp_ggml_kernels llmp_cuda_device_memory
                                                   llmp_launch_recorder)
  # On a discrete GPU too (`gpu-discrete`, D-082): the over-read probe and
  # the host checks; the casts, embedding and attention hold to launches
  # recorded on a GB10.
  llmp_gpu_gtest(ggml_exl3_ops_test
                   DISCRETE GgmlExl3OpsTest.OperationsReadAndWriteOnlyTheirTensors
                            GgmlExl3OpsTest.AttentionChecksRefuseWhatTheLauncherWouldNotTake)

  # The operations DeepSeek V4 Flash and Qwen3.8 Flash add
  # (kernels/ggml/ops_ext.h), against FP64 references within upstream's
  # test-backend-ops bounds, at small cases of the models' shapes.
  add_executable(ggml_ext_ops_test ggml_ext_ops_test.cc)
  target_include_directories(ggml_ext_ops_test SYSTEM PRIVATE
                             "${LLMP_SDK}/cuda/targets/${LLMP_CUDA_TARGET_DIR}/include")
  target_link_libraries(ggml_ext_ops_test PRIVATE llmp_ggml_kernels llmp_cuda_device_memory)
  llmp_gpu_gtest(ggml_ext_ops_test)

  # Qwen3.8's formats: llmpalooza's MXFP8 and NVFP4-row operations
  # (kernels/ggml/llmp_ops.h), GGML's NVFP4 expert products over repacked
  # ModelOpt weights, and QSA's attention shape, against FP64 references.
  add_executable(qwen38_ops_test qwen38_ops_test.cc)
  target_include_directories(qwen38_ops_test SYSTEM PRIVATE
                             "${LLMP_SDK}/cuda/targets/${LLMP_CUDA_TARGET_DIR}/include")
  target_link_libraries(qwen38_ops_test PRIVATE llmp_ggml_kernels llmp_cuda_device_memory)
  llmp_gpu_gtest(qwen38_ops_test)

  # A speculative verify's row-invariant products (D-092), the drafter's
  # argmax and the snapshot's range copies, against GGML's own one-column
  # launches bit for bit.
  add_executable(spec_rows_test spec_rows_test.cc)
  target_include_directories(spec_rows_test SYSTEM PRIVATE
                             "${LLMP_SDK}/cuda/targets/${LLMP_CUDA_TARGET_DIR}/include")
  target_link_libraries(spec_rows_test PRIVATE llmp_ggml_kernels llmp_cuda_device_memory)
  llmp_gpu_gtest(spec_rows_test)

  # DeepSeek V4's fast plan (kernels/ggml/llmp_ops.h, dsv4_fast.cu; D-085's
  # note): the vector products against GGML's MMVQ, every launch
  # configuration, the routing, combination, hyper-connection pre-mix and
  # compressor against FP64, and what their checks refuse.
  add_executable(dsv4_fast_test dsv4_fast_test.cc)
  target_include_directories(dsv4_fast_test SYSTEM PRIVATE
                             "${LLMP_SDK}/cuda/targets/${LLMP_CUDA_TARGET_DIR}/include")
  target_link_libraries(dsv4_fast_test PRIVATE llmp_ggml_kernels llmp_cuda_device_memory)
  llmp_gpu_gtest(dsv4_fast_test)

  add_executable(qwen38_quant_head_test qwen38_quant_head_test.cc)
  target_include_directories(qwen38_quant_head_test SYSTEM PRIVATE
                             "${LLMP_SDK}/cuda/targets/${LLMP_CUDA_TARGET_DIR}/include")
  target_link_libraries(qwen38_quant_head_test PRIVATE llmp_ggml_kernels llmp_cuda_device_memory)
  llmp_gpu_gtest(qwen38_quant_head_test MODELS)

  # The fast plan's ds4 prefill stage mechanisms (docs/experiments/
  # ds4-prefill-stages): each against the launches it replaces, byte for
  # byte, with its selection, refusals and registry declaration.
  add_executable(dsv4_stages_test dsv4_stages_test.cc)
  target_include_directories(dsv4_stages_test SYSTEM PRIVATE
                             "${LLMP_SDK}/cuda/targets/${LLMP_CUDA_TARGET_DIR}/include")
  target_link_libraries(dsv4_stages_test PRIVATE llmp_ggml_kernels llmp_cuda_device_memory)
  llmp_gpu_gtest(dsv4_stages_test)

  # Llmpalooza's Qwen-Image kernels (kernels/image) against host references.
  add_executable(image_kernels_test image_kernels_test.cc)
  target_include_directories(image_kernels_test SYSTEM PRIVATE
                             "${LLMP_SDK}/cuda/targets/${LLMP_CUDA_TARGET_DIR}/include")
  target_link_libraries(image_kernels_test PRIVATE llmp_image_kernels)
  llmp_gpu_gtest(image_kernels_test)

  # The block decoders under the product templates (kernels/decode,
  # D-108): every decoder under every template against the CPU reference
  # decode and an F64 product, row invariance, refusals and the
  # registry's declarations; on the GB10 and the discrete GPUs (D-082).
  add_executable(decode_gpu_test decode_gpu_test.cc)
  target_include_directories(decode_gpu_test SYSTEM PRIVATE
                             "${LLMP_SDK}/cuda/targets/${LLMP_CUDA_TARGET_DIR}/include")
  target_link_libraries(decode_gpu_test PRIVATE llmp_decode_kernels llmp_reference_kernels)
  llmp_gpu_gtest(decode_gpu_test DISCRETE ALL)

  # Block-quantized KV caches (kernels/ggml/kv_quant.h): the quantizing
  # stores byte for byte against the reference encoder, the quantized vector
  # attention against the reference executor, their checks and capabilities;
  # on a discrete GPU too (D-082).
  add_executable(kv_quant_gpu_test kv_quant_gpu_test.cc)
  target_include_directories(kv_quant_gpu_test SYSTEM PRIVATE
                             "${LLMP_SDK}/cuda/targets/${LLMP_CUDA_TARGET_DIR}/include")
  target_link_libraries(kv_quant_gpu_test PRIVATE llmp_ggml_kernels llmp_cuda_device_memory
                                                  llmp_reference_kernels)
  llmp_gpu_gtest(kv_quant_gpu_test DISCRETE ALL)

  # The format layer's first proof (D-108): Qwen3.8's native NVFP4 and
  # MXFP8 tensors, one layer's worth of each product shape, through the
  # decoders against the CPU reference, with device memory checked; meant
  # for the discrete sm_86 target. The artifacts' roots come from
  # LLMP_QWEN38_PROOF_ARTIFACT (and _GGML_ARTIFACT), by default the M3
  # store; without them it skips.
  add_executable(qwen38_decode_proof_test qwen38_decode_proof_test.cc)
  target_include_directories(qwen38_decode_proof_test SYSTEM PRIVATE
                             "${LLMP_SDK}/cuda/targets/${LLMP_CUDA_TARGET_DIR}/include")
  target_link_libraries(qwen38_decode_proof_test PRIVATE llmp_decode_kernels
                                                         llmp_reference_kernels llmp_artifact
                                                         llmp_model)
  llmp_gpu_gtest(qwen38_decode_proof_test MODELS DISCRETE ALL
                   ENVIRONMENT "CUDA_MODULE_LOADING=EAGER")

  # Llmpalooza's fusions of Qwen3.8's hyper-connection and MoE-output nodes
  # (kernels/ggml/llmp_ops.h) against the GGML nodes they replace, bit for
  # bit, and against FP64 references.
  add_executable(qwen38_fused_test qwen38_fused_test.cc)
  target_include_directories(qwen38_fused_test SYSTEM PRIVATE
                             "${LLMP_SDK}/cuda/targets/${LLMP_CUDA_TARGET_DIR}/include")
  target_link_libraries(qwen38_fused_test PRIVATE llmp_ggml_kernels llmp_cuda_device_memory)
  llmp_gpu_gtest(qwen38_fused_test)

  # Qwen3.8's fast path (kernels/ggml/llmp_ops.h, D-085): the MXFP8
  # tensor-core products, the fused hyper-connections, routing, Gated
  # DeltaNet and QSA kernels, against FP64 references and exact host
  # selections, and what their checks refuse.
  add_executable(qwen38_fast_test qwen38_fast_test.cc)
  target_include_directories(qwen38_fast_test SYSTEM PRIVATE
                             "${LLMP_SDK}/cuda/targets/${LLMP_CUDA_TARGET_DIR}/include")
  target_link_libraries(qwen38_fast_test PRIVATE llmp_ggml_kernels llmp_cuda_device_memory
                                               llmp_device_runtime)
  llmp_gpu_gtest(qwen38_fast_test)

  # Qwen3.8's routed experts over the CUTLASS layout (kernels/ggml/
  # moe_layout.h): the conversion, routing, CUTLASS's grouped GEMM, the
  # prefill path against GGML's MMQ path and FP64, decode's vector products.
  add_executable(qwen38_moe_test qwen38_moe_test.cc)
  target_include_directories(qwen38_moe_test SYSTEM PRIVATE
                             "${LLMP_SDK}/cuda/targets/${LLMP_CUDA_TARGET_DIR}/include")
  target_link_libraries(qwen38_moe_test PRIVATE llmp_ggml_kernels llmp_cuda_device_memory)
  llmp_gpu_gtest(qwen38_moe_test)

  # A Qwen3.8 speculative verify's commit (kernels/ggml/qwen38_commit.h):
  # the kept rows' recurrence bit for bit the verify's own, the histories
  # exact, and the launcher's refusals.
  add_executable(qwen38_commit_test qwen38_commit_test.cc)
  target_include_directories(qwen38_commit_test SYSTEM PRIVATE
                             "${LLMP_SDK}/cuda/targets/${LLMP_CUDA_TARGET_DIR}/include")
  target_link_libraries(qwen38_commit_test PRIVATE llmp_ggml_kernels llmp_cuda_device_memory)
  llmp_gpu_gtest(qwen38_commit_test)

  add_executable(ggml_cublas_test ggml_cublas_test.cc)
  target_include_directories(ggml_cublas_test SYSTEM PRIVATE
                             "${LLMP_SDK}/cuda/targets/${LLMP_CUDA_TARGET_DIR}/include")
  target_link_libraries(ggml_cublas_test PRIVATE llmp_ggml_kernels llmp_cuda_device_memory)
  llmp_gpu_gtest(ggml_cublas_test)
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
  COMMAND "${LLMP_PYTHON}" -B "${CMAKE_CURRENT_SOURCE_DIR}/artifact_corpus.py" "${artifact_corpus}"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/artifact_corpus.py" "${artifact_prototype}/layout.py"
          "${artifact_prototype}/test_layout.py"
  COMMENT "Building the artifact reader's corpus with the M0 prototype"
  VERBATIM)
add_custom_target(artifact_corpus ALL DEPENDS "${artifact_corpus}/stamp")
if(TARGET dsv4_prefill_plan_test)
  add_dependencies(dsv4_prefill_plan_test artifact_corpus)
endif()
add_executable(artifact_test artifact_test.cc)
target_link_libraries(artifact_test PRIVATE llmp_artifact)
add_dependencies(artifact_test artifact_corpus)
# CTest runs it here, beside the corpus (remote runs map this directory).
llmp_gtest(artifact_test TEST_PREFIX unit. WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
             PROPERTIES LABELS unit ENVIRONMENT "LLMP_TEST_SCRATCH=${CMAKE_CURRENT_BINARY_DIR}/scratch")

# The v1 artifact format (D-113) through the reader: valid artifacts the
# test-only writer makes (tests/support/artifact_writer.h), v1 copies of the
# corpus's v0 goldens placed and read as the originals, each rule refused
# with its reason, seeded document mutations, v1 compositions.
add_executable(artifact_v1_test artifact_v1_test.cc)
target_link_libraries(artifact_v1_test PRIVATE llmp_artifact llmp_test_artifact_writer)
add_dependencies(artifact_v1_test artifact_corpus)
llmp_gtest(artifact_v1_test TEST_PREFIX unit. WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
             PROPERTIES LABELS unit ENVIRONMENT "LLMP_TEST_SCRATCH=${CMAKE_CURRENT_BINARY_DIR}/scratch")

# The kept GGUF metadata reader (artifact/gguf_metadata.h): values and each
# refusal on headers built in the test.
add_executable(gguf_metadata_test gguf_metadata_test.cc)
target_link_libraries(gguf_metadata_test PRIVATE llmp_artifact)
llmp_gtest(gguf_metadata_test TEST_PREFIX unit. PROPERTIES LABELS unit)

# v0's descriptors as tensor formats (format_map.h, D-108): size parity with
# the v0 rule, pairings, the corpus and the installed real artifacts.
add_executable(artifact_format_map_test artifact_format_map_test.cc)
target_link_libraries(artifact_format_map_test PRIVATE llmp_artifact)
add_dependencies(artifact_format_map_test artifact_corpus)
llmp_gtest(artifact_format_map_test TEST_PREFIX unit. WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
             PROPERTIES LABELS unit)

# The composition reader (D-089): each rule of the prototype's
# verify_composition on compositions built here, and the installed
# Qwen-Image-2.1 composition where the M3 store holds it (a Spark).
add_executable(composition_test composition_test.cc)
target_link_libraries(composition_test PRIVATE llmp_artifact)
llmp_gtest(composition_test TEST_PREFIX unit.
             PROPERTIES LABELS unit
                        ENVIRONMENT "LLMP_TEST_SCRATCH=${CMAKE_CURRENT_BINARY_DIR}/scratch")

# The Qwen-Image-2.1 adapter (model/qwen_image.h): schedule, tensor lists
# and binding, the VAE's plan, rotary tables, sinusoid and pixels; the
# installed components bound where the M3 store holds them (a Spark).
add_executable(qwen_image_test qwen_image_test.cc)
target_link_libraries(qwen_image_test PRIVATE llmp_model llmp_artifact)
llmp_gtest(qwen_image_test TEST_PREFIX unit. PROPERTIES LABELS unit)

# The locked ExLlamaV3 subset: the device context's sizes and the
# launchers' host checks in every profile; the kernels' tables and lookups
# link in every CUDA profile, and on a GB10 (label `gpu`) every kernel loads,
# the recorded ones use the P0 launch record's registers, and llmpalooza's
# launchers run each path of the native linear. On a discrete GPU the build
# targets (`gpu-discrete`, D-082) every GEMM kernel loads from its own SASS
# and the launchers and the native plan run; what compares with GB10 records
# (registers, cuBLASLt pins, recorded launches, upstream's GB10 choices)
# stays GB10-only.
add_executable(exl3_context_test exl3_context_test.cc)
target_link_libraries(exl3_context_test PRIVATE llmp_exl3)
llmp_gtest(exl3_context_test TEST_PREFIX unit. PROPERTIES LABELS unit)

add_executable(exl3_validate_test exl3_validate_test.cc)
target_link_libraries(exl3_validate_test PRIVATE llmp_exl3)
llmp_gtest(exl3_validate_test TEST_PREFIX unit. PROPERTIES LABELS unit)

if(LLMP_CUDA)
  # The tables and instances alone (llmp_exl3_cuda), not the launchers:
  # this test runs in every CUDA profile, where CTest runs it to list its
  # tests, and so must load without cuBLAS (which the launchers link and
  # the workstation and qemu-user runs cannot find).
  add_executable(exl3_tables_test exl3_tables_test.cc)
  target_include_directories(exl3_tables_test SYSTEM PRIVATE
                             "${LLMP_SDK}/cuda/targets/${LLMP_CUDA_TARGET_DIR}/include")
  target_link_libraries(exl3_tables_test PRIVATE llmp_exl3 llmp_exl3_cuda)
  llmp_gtest(exl3_tables_test TEST_PREFIX unit. PROPERTIES LABELS unit)

  add_executable(exl3_selected_test exl3_selected_test.cc)
  target_link_libraries(exl3_selected_test PRIVATE llmp_exl3_selected llmp_model llmp_spec llmp_engine)
  # CUDA-linked descriptor controls: discover names from source without loading cuBLAS.
  llmp_gpu_gtest(exl3_selected_test
                   DISCRETE Exl3SelectedTest.AllCompletePinnedReconstructionsRetainFourPartsAndBoundScratch
                            Exl3SelectedTest.RefusesMissingDuplicateShortPartsAndWrongNumericalPath
                            Exl3SelectedTest.SharedGraphKeepsEveryWeightPartAndExactSelectedOperation
                            Exl3SelectedTest.ProfileIdentityAndGemvFactsAreExplicit
                            Exl3SelectedTest.UncapturedComponentSelectionIsExplicitAndKeyed
                            Exl3SelectedTest.AllProfilesCoverPackedPairsGemvAndReconstructionBoundaries)

  add_executable(exl3_kernels_test exl3_kernels_test.cc)
  target_link_libraries(exl3_kernels_test PRIVATE llmp_exl3_kernels)
  llmp_gpu_gtest(exl3_kernels_test DISCRETE Exl3KernelsGpuTest.EveryKernelLoadsFromTheDevicesSass)

  # The launchers, the linear's paths and the registry on a GB10: operands
  # in cudaMalloc memory and device VMM alike, every operand flush against
  # unmapped VMM granules (the over-read probe), and every refusal.
  add_executable(exl3_linear_test exl3_linear_test.cc)
  target_link_libraries(exl3_linear_test PRIVATE llmp_exl3_kernels llmp_cuda_device_memory)
  # On a discrete GPU too (`gpu-discrete`, D-082), all but what runs the
  # reconstruction GEMM with its cuBLASLt algorithms pinned on a GB10.
  llmp_gpu_gtest(exl3_linear_test
                   DISCRETE Exl3LinearTest.TheRegistryBindsEachPathToItsOwnCalls
                            Exl3LinearTest.RefusesGridsThatAreNotCoresident
                            Exl3LinearTest.NoTwoLiveContextsShareLockSlots
                            Exl3LinearTest.AFaultReturnsAsAFaultAndStopsTheContext
                            Exl3LinearTest.NoPackedKernelReadsOutsideItsOperandsAtTheFixturesShapes
                            Exl3LinearTest.TwoContextsAtTheCoresidentLimitBothComplete
                            Exl3ImplementationsTest.IdentitiesRecordTheLibraryAssertions)

  # The EXL3 trellis through the format layer (D-108): the module's
  # capabilities and Candidates(), ExLlamaV3's reconstruction kernels
  # against the CPU reference, and, where the v0 fixtures are installed
  # (spark, spark-b), every fixture linear's weights and layer 0's products
  # through each candidate. On a discrete GPU too, all but the fixtures'
  # test, which runs the GB10's pinned reconstruction GEMMs.
  add_executable(exl3_trellis_test exl3_trellis_test.cc)
  target_link_libraries(exl3_trellis_test PRIVATE llmp_exl3_kernels llmp_reference_kernels
                                                  llmp_artifact llmp_model llmp_cuda_device_memory)
  llmp_gpu_gtest(exl3_trellis_test MODELS
                   DISCRETE Exl3TrellisSelectionTest.CandidatesFollowRowsRatesShapesAndDevices
                            Exl3TrellisTest.ReconstructionIsTheReferenceDecode)

  # The module's implementation identities follow every one of its source
  # files (src/kernels/exl3/module_digest.cmake), checked as the GGML
  # module's are.
  get_property(exl3_module_names GLOBAL PROPERTY LLMP_EXL3_MODULE_NAMES)
  add_test(NAME unit.Exl3ModuleDigest
           COMMAND ${CMAKE_COMMAND} "-DBASE=${PROJECT_SOURCE_DIR}/src/kernels/exl3"
                   "-DNAMES=${exl3_module_names}"
                   "-DGENERATED=${PROJECT_BINARY_DIR}/src/kernels/exl3/module_digest.cc"
                   "-DWORK=${CMAKE_CURRENT_BINARY_DIR}/exl3_module_digest"
                   -P ${CMAKE_CURRENT_SOURCE_DIR}/ggml_module_digest_test.cmake)
  set_tests_properties(unit.Exl3ModuleDigest PROPERTIES LABELS unit)
endif()

add_executable(gemma_moe_fusion_test gemma_moe_fusion_test.cc)
target_link_libraries(gemma_moe_fusion_test PRIVATE llmp_ggml_tensors llmp_artifact llmp_json)
llmp_gtest(gemma_moe_fusion_test TEST_PREFIX unit. PROPERTIES LABELS unit
             ENVIRONMENT "LLMP_TEST_DATA=${tokenizer_data}")

add_executable(gemma4_assistant_test gemma4_assistant_test.cc)
target_link_libraries(gemma4_assistant_test PRIVATE llmp_model llmp_artifact llmp_json)
llmp_gtest(gemma4_assistant_test TEST_PREFIX unit. PROPERTIES LABELS unit
             ENVIRONMENT "LLMP_TEST_DATA=${tokenizer_data}")

add_executable(gemma4_assistant_artifact_test gemma4_assistant_artifact_test.cc)
target_link_libraries(gemma4_assistant_artifact_test PRIVATE llmp_model llmp_artifact llmp_tokenizer)
llmp_gtest(gemma4_assistant_artifact_test TEST_PREFIX unit. PROPERTIES LABELS "unit\\;models")

# Closed native-v1 admission over actual descriptor formats and stored specs.
add_executable(component_admission_test component_admission_test.cc)
target_link_libraries(component_admission_test PRIVATE llmp_runtime llmp_components llmp_json)
llmp_gtest(component_admission_test TEST_PREFIX unit. PROPERTIES LABELS unit
             ENVIRONMENT "LLMP_TEST_DATA=${tokenizer_data}")

# The model spec's canonical JSON, refusals and bound digest; the
# architecture mappings as data and applied to the Gemma 2 and Gemma 3
# tensor tables with today's profiles (spec/).
add_executable(spec_test spec_test.cc)
target_link_libraries(spec_test PRIVATE llmp_spec llmp_model llmp_artifact llmp_json)
llmp_gtest(spec_test TEST_PREFIX unit. PROPERTIES LABELS unit
             ENVIRONMENT "LLMP_TEST_DATA=${tokenizer_data}")

# The spec a v1 artifact or composition stores (spec/stored.h, D-113), on
# artifacts the test-only writer makes, and v0's goldens as before.
add_executable(spec_stored_test spec_stored_test.cc)
target_link_libraries(spec_stored_test PRIVATE llmp_spec llmp_artifact llmp_model
                                               llmp_test_artifact_writer)
add_dependencies(spec_stored_test artifact_corpus)
llmp_gtest(spec_stored_test TEST_PREFIX unit. WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
             PROPERTIES LABELS unit ENVIRONMENT "LLMP_TEST_SCRATCH=${CMAKE_CURRENT_BINARY_DIR}/scratch")

# The spec synthesized from the installed Gemma 2 and Gemma 3 v0 artifacts.
add_executable(spec_models_test spec_models_test.cc)
target_link_libraries(spec_models_test PRIVATE llmp_spec llmp_model llmp_artifact llmp_json)
llmp_gtest(spec_models_test TEST_PREFIX unit. PROPERTIES LABELS "unit\;models"
             ENVIRONMENT "LLMP_TEST_DATA=${tokenizer_data}")

# The component catalog (components/): parameters, ValidateSpec on the
# Gemma specs and its refusals, each fragment run by the reference executor
# against its scalar reference, BuildStep, the stage layout against
# Gemma's closed-form layouts and the step's inputs against its host masks.
add_executable(components_test components_test.cc)
target_link_libraries(components_test PRIVATE llmp_components llmp_reference_kernels llmp_spec
                                              llmp_model llmp_artifact llmp_json)
llmp_gtest(components_test TEST_PREFIX unit. PROPERTIES LABELS unit
             ENVIRONMENT "LLMP_TEST_DATA=${tokenizer_data}")

# The Qwen3.8 components (hyper-connections, Gated DeltaNet, QSA, the n-gram
# embedding, the MoE's hyper-connection form, the MTP drafter's input)
# against their references, verify saves and commits, the qwen4exp and
# qwen4exp-mtp mappings over every import's tensors and the steps they
# make; and the steps against today's unfused graph builder, node for node.
add_executable(components_qwen2_test components_qwen2_test.cc)
target_link_libraries(components_qwen2_test PRIVATE llmp_components llmp_spec llmp_model
                                                 llmp_planner llmp_ggml_tensors)
llmp_gtest(components_qwen2_test TEST_PREFIX unit. PROPERTIES LABELS unit)

add_executable(components_qwen2_models_test components_qwen2_models_test.cc)
target_link_libraries(components_qwen2_models_test PRIVATE llmp_components llmp_spec llmp_model
                                                        llmp_artifact)
llmp_gtest(components_qwen2_models_test TEST_PREFIX unit. PROPERTIES LABELS "unit\;models")

add_executable(components_qwen38_test components_qwen38_test.cc)
target_link_libraries(components_qwen38_test PRIVATE llmp_components llmp_reference_kernels
                                                     llmp_spec llmp_model llmp_artifact llmp_json)
llmp_gtest(components_qwen38_test TEST_PREFIX unit. PROPERTIES LABELS unit)
add_executable(components_qwen38_order_test components_qwen38_order_test.cc)
target_link_libraries(components_qwen38_order_test PRIVATE llmp_components llmp_ggml_tensors
                                                           llmp_spec llmp_model llmp_artifact
                                                           llmp_json)
llmp_gtest(components_qwen38_order_test TEST_PREFIX unit. PROPERTIES LABELS unit)
# The Qwen3.8 mappings on the installed artifacts: each import's spec, the
# fixture's equal to the ModelOpt ones, today's state layouts, the MTP
# drafter composed, every step kind built.
add_executable(components_qwen38_models_test components_qwen38_models_test.cc)
target_link_libraries(components_qwen38_models_test PRIVATE llmp_components llmp_spec llmp_model
                                                            llmp_artifact llmp_json)
llmp_gtest(components_qwen38_models_test TEST_PREFIX unit. PROPERTIES LABELS "unit\;models")

# The DeepSeek V4 components (the Sinkhorn hyper-connections, MQA with its
# compressors and lightning indexer, the DeepSeek routers and clamped
# SwiGLU, the token-hash side input, the DSpark drafter's injected input and
# Markov head) against their references, the compressors' inputs against
# today's plans, the deepseek4 and dflash mappings over the fixture's
# tensors, the steps they make and the injection run; and the steps against
# today's reference-form graph builder, node for node.
add_executable(components_dsv4_test components_dsv4_test.cc)
target_link_libraries(components_dsv4_test PRIVATE llmp_components llmp_reference_kernels
                                                   llmp_spec llmp_model llmp_artifact llmp_json)
llmp_gtest(components_dsv4_test TEST_PREFIX unit. PROPERTIES LABELS unit)
add_executable(components_dsv4_order_test components_dsv4_order_test.cc)
target_link_libraries(components_dsv4_order_test PRIVATE llmp_components llmp_ggml_tensors
                                                         llmp_spec llmp_model llmp_artifact
                                                         llmp_json)
llmp_gtest(components_dsv4_order_test TEST_PREFIX unit. PROPERTIES LABELS unit)
# The DeepSeek V4 mappings on the installed artifacts: each target's spec
# and its DSpark drafter's, the fixture's equal to them, today's state
# layouts, every step kind built.
add_executable(components_dsv4_models_test components_dsv4_models_test.cc)
target_link_libraries(components_dsv4_models_test PRIVATE llmp_components llmp_spec llmp_model
                                                          llmp_artifact llmp_json)
llmp_gtest(components_dsv4_models_test TEST_PREFIX unit. PROPERTIES LABELS "unit\;models")

# The planner (D-107): policies, patterns, selection, placement, lanes and
# plan keys on small graphs and a fake registry.
add_executable(owner_products_test owner_products_test.cc)
target_link_libraries(owner_products_test PRIVATE llmp_planner llmp_ir)
llmp_gtest(owner_products_test TEST_PREFIX unit. PROPERTIES LABELS unit)

add_executable(planner_test planner_test.cc)
target_link_libraries(planner_test PRIVATE llmp_planner llmp_ir)
llmp_gtest(planner_test TEST_PREFIX unit. PROPERTIES LABELS unit)

# Partitioning (D-107): a small Gemma-like spec and a test routed-expert
# block partitioned by tensor, pipeline and expert parallelism over three
# to eight simulated devices against the unpartitioned graph; Assign's
# choices for described topologies and against a brute-force enumeration.
add_executable(planner_partition_test planner_partition_test.cc)
target_link_libraries(planner_partition_test PRIVATE llmp_planner llmp_components
                                                     llmp_reference_kernels llmp_spec llmp_model
                                                     llmp_artifact llmp_json)
llmp_gtest(planner_partition_test TEST_PREFIX unit. PROPERTIES LABELS unit
             ENVIRONMENT "LLMP_TEST_DATA=${tokenizer_data}")

# The planner on the Gemma 2 and Gemma 3 steps under each family's
# production recipe: its fusions, shared preparations and a sound placement
# at every setup shape.
add_executable(planner_gemma_test planner_gemma_test.cc)
target_link_libraries(planner_gemma_test PRIVATE llmp_planner llmp_components llmp_ggml_tensors
                                                 llmp_spec llmp_model llmp_artifact llmp_json)
llmp_gtest(planner_gemma_test TEST_PREFIX unit. PROPERTIES LABELS unit
             ENVIRONMENT "LLMP_TEST_DATA=${tokenizer_data}")

# Bound ordinary dense owner recipes, without enabling a runtime path.
add_executable(ggml_owner_products_test ggml_owner_products_test.cc)
target_link_libraries(ggml_owner_products_test PRIVATE llmp_ggml_tensors llmp_planner
                                                     llmp_components llmp_spec llmp_model)
llmp_gtest(ggml_owner_products_test TEST_PREFIX unit. PROPERTIES LABELS unit)

# The GGML lowering (kernels/ggml/lower.h) of the Gemma 2 and Gemma 3 steps
# at every setup shape (every profile builds the lowering).
add_executable(ggml_lower_test ggml_lower_test.cc)
target_link_libraries(ggml_lower_test PRIVATE llmp_ggml_tensors llmp_planner llmp_components
                                              llmp_spec llmp_model llmp_artifact llmp_json)
llmp_gtest(ggml_lower_test TEST_PREFIX unit. PROPERTIES LABELS unit
             ENVIRONMENT "LLMP_TEST_DATA=${tokenizer_data}")

# The GGML lowering of Gemma 4's steps (26B-A4B, 31B and their assistants)
# under each serving recipe the gemma4 mapping's policy gives, structurally,
# at every setup shape.
add_executable(ggml_lower_gemma4_test ggml_lower_gemma4_test.cc)
target_link_libraries(ggml_lower_gemma4_test PRIVATE llmp_ggml_tensors llmp_planner
                                                     llmp_components llmp_spec llmp_model
                                                     llmp_artifact llmp_json)
llmp_gtest(ggml_lower_gemma4_test TEST_PREFIX unit. PROPERTIES LABELS unit
             ENVIRONMENT "LLMP_TEST_DATA=${tokenizer_data}")

# The GGML lowering of Qwen3.8's steps (each import's target, plain and with
# the MTP drafter, and the drafter's passes) against today's builder's fast
# form, launch for launch, at the runner's setup shapes.
add_executable(ggml_lower_dsv4_test ggml_lower_dsv4_test.cc)
target_link_libraries(ggml_lower_dsv4_test PRIVATE llmp_ggml_tensors llmp_planner
                                                llmp_engine_core llmp_components llmp_spec
                                                llmp_model llmp_artifact llmp_json)
llmp_gtest(ggml_lower_dsv4_test TEST_PREFIX unit. PROPERTIES LABELS unit)
if(LLMP_CUDA)
  add_executable(ggml_lower_dsv4_gpu_test ggml_lower_dsv4_gpu_test.cc)
  # Independent original score producer: test-only, same device flags/targets.
  target_sources(ggml_lower_dsv4_gpu_test PRIVATE ../support/dsv4_score_original.cu)
  target_include_directories(ggml_lower_dsv4_gpu_test PRIVATE "${PROJECT_SOURCE_DIR}")
  get_target_property(score_architectures llmp_ggml_kernels CUDA_ARCHITECTURES)
  set_target_properties(ggml_lower_dsv4_gpu_test PROPERTIES
                        CUDA_ARCHITECTURES "${score_architectures}")
  set_source_files_properties(../support/dsv4_score_original.cu PROPERTIES
                              COMPILE_OPTIONS "-use_fast_math;-extended-lambda")
  target_include_directories(ggml_lower_dsv4_gpu_test SYSTEM PRIVATE
      "${LLMP_SDK}/cuda/targets/${LLMP_CUDA_TARGET_DIR}/include")
  target_link_libraries(ggml_lower_dsv4_gpu_test PRIVATE llmp_ggml_kernels llmp_planner
                                                        llmp_components llmp_cuda_device_memory)
  llmp_gpu_gtest(ggml_lower_dsv4_gpu_test DISCRETE ALL)
endif()
add_executable(ggml_lower_qwen38_test ggml_lower_qwen38_test.cc)
target_link_libraries(ggml_lower_qwen38_test PRIVATE llmp_ggml_tensors llmp_planner
                                                     llmp_components llmp_spec llmp_model
                                                     llmp_artifact llmp_json)
llmp_gtest(ggml_lower_qwen38_test TEST_PREFIX unit. PROPERTIES LABELS unit)

# The components' state contracts through the shared slot lifecycle on the
# fake device backend.
add_executable(components_state_test components_state_test.cc)
target_link_libraries(components_state_test PRIVATE llmp_components llmp_engine_core
                                                    llmp_fake_device_backend llmp_platform)
llmp_gtest(components_state_test TEST_PREFIX unit.fake. PROPERTIES LABELS unit
             ENVIRONMENT "LLMP_TEST_SCRATCH=${CMAKE_CURRENT_BINARY_DIR}/scratch")

# The Gemma 2 and Gemma 3 components end to end on the installed artifacts,
# by the reference executor, against the served model (the server over the
# shared engine).
if(LLMP_CUDA)
  add_executable(expert_slab_test expert_slab_test.cc)
  target_link_libraries(expert_slab_test PRIVATE llmp_engine llmp_test_artifact_writer)
  llmp_gtest(expert_slab_test TEST_PREFIX unit. PROPERTIES LABELS unit
               ENVIRONMENT "LLMP_TEST_SCRATCH=${CMAKE_CURRENT_BINARY_DIR}/scratch")
endif()
if(LLMP_CUDA)
  add_executable(components_models_gpu_test components_models_gpu_test.cc)
  target_link_libraries(components_models_gpu_test PRIVATE llmp_runtime_serving llmp_components
                                                           llmp_reference_kernels llmp_spec
                                                           llmp_artifact)
  llmp_gpu_gtest(components_models_gpu_test MODELS
                   ENVIRONMENT "LLMP_TEST_SCRATCH=${CMAKE_CURRENT_BINARY_DIR}/scratch")

  # The Gemma 4 components end to end on the installed 26B-A4B and 31B, by
  # the reference executor, against the served model.
  add_executable(components_gemma4_models_gpu_test components_gemma4_models_gpu_test.cc)
  target_link_libraries(components_gemma4_models_gpu_test PRIVATE llmp_runtime_serving
                                                                  llmp_components
                                                                  llmp_reference_kernels llmp_spec
                                                                  llmp_artifact)
  llmp_gpu_gtest(components_gemma4_models_gpu_test MODELS
                   ENVIRONMENT "LLMP_TEST_SCRATCH=${CMAKE_CURRENT_BINARY_DIR}/scratch")

  # The one model runner over the component step source and the GGML
  # lowering on the installed artifacts, a scripted run finite and exact
  # from one runner to the next.
  # Real Qwen2 FP16 fixture on the shared runner against the frozen bridge.
  add_executable(component_runner_qwen2_exl3_gpu_test component_runner_qwen2_exl3_gpu_test.cc)
  target_link_libraries(component_runner_qwen2_exl3_gpu_test PRIVATE llmp_corpus_score llmp_engine llmp_exl3_selected
                                                                     llmp_runtime
                                                                     llmp_components llmp_spec
                                                                     llmp_artifact)
  llmp_gpu_gtest(component_runner_qwen2_exl3_gpu_test MODELS
                   ENVIRONMENT "LLMP_TEST_SCRATCH=${CMAKE_CURRENT_BINARY_DIR}/scratch")

  add_executable(fp16_benchmark_gpu_test fp16_benchmark_gpu_test.cc)
  target_link_libraries(fp16_benchmark_gpu_test PRIVATE llmp_paged_runners llmp_base)
  llmp_gpu_gtest(fp16_benchmark_gpu_test MODELS
                   ENVIRONMENT "LLMP_TEST_SCRATCH=${CMAKE_CURRENT_BINARY_DIR}/scratch")

  add_executable(component_runner_qwen2_gpu_test component_runner_qwen2_gpu_test.cc)
  target_link_libraries(component_runner_qwen2_gpu_test PRIVATE llmp_corpus_score llmp_engine llmp_ggml_kernels
                                                                llmp_components llmp_spec
                                                                llmp_artifact)
  llmp_gpu_gtest(component_runner_qwen2_gpu_test MODELS
                   ENVIRONMENT "LLMP_TEST_SCRATCH=${CMAKE_CURRENT_BINARY_DIR}/scratch")

  add_executable(component_runner_gpu_test component_runner_gpu_test.cc)
  target_link_libraries(component_runner_gpu_test PRIVATE llmp_engine llmp_ggml_kernels
                                                          llmp_components llmp_spec llmp_artifact)
  llmp_gpu_gtest(component_runner_gpu_test MODELS
                   ENVIRONMENT "LLMP_TEST_SCRATCH=${CMAKE_CURRENT_BINARY_DIR}/scratch")

  # The same on the Gemma 4 26B-A4B and 31B and their assistants' draft
  # passes, under each serving recipe the mapping's policy gives, and a
  # speculative verify's waves.
  add_executable(component_runner_gemma4_gpu_test component_runner_gemma4_gpu_test.cc)
  target_link_libraries(component_runner_gemma4_gpu_test PRIVATE llmp_engine llmp_ggml_kernels
                                                                 llmp_components llmp_spec
                                                                 llmp_artifact)
  llmp_gpu_gtest(component_runner_gemma4_gpu_test MODELS
                   ENVIRONMENT "LLMP_TEST_SCRATCH=${CMAKE_CURRENT_BINARY_DIR}/scratch")
  # The same on the Qwen3.8 imports (ModelOpt's two expert layouts, a
  # GGUF quantization's) against today's runner's fast form, one at a time.
  add_executable(component_runner_qwen38_gpu_test component_runner_qwen38_gpu_test.cc)
  target_link_libraries(component_runner_qwen38_gpu_test PRIVATE llmp_engine llmp_ggml_kernels
                                                                 llmp_components llmp_spec
                                                                 llmp_artifact)
  llmp_gpu_gtest(component_runner_qwen38_gpu_test MODELS
                   ENVIRONMENT "LLMP_TEST_SCRATCH=${CMAKE_CURRENT_BINARY_DIR}/scratch")
endif()

# The general GGUF header parser (base/gguf.h) and the importer's container
# readers (import/): GGUF, safetensors with config.json, MLX, EXL3 and
# diffusers pipelines on files built in the test, every refusal, the
# quantizations, which reader claims what, and the mapping sources.
add_executable(import_test import_test.cc import_readers_test.cc)
target_link_libraries(import_test PRIVATE llmp_import llmp_gguf llmp_spec llmp_json)
llmp_gtest(import_test TEST_PREFIX unit. PROPERTIES LABELS unit
             ENVIRONMENT "LLMP_TEST_SCRATCH=${CMAKE_CURRENT_BINARY_DIR}/scratch")

# Closed native ModelOpt normalization and separately funded source control.
add_executable(import_modelopt_normalize_test import_modelopt_normalize_test.cc)
target_link_libraries(import_modelopt_normalize_test PRIVATE llmp_import llmp_components llmp_model llmp_artifact llmp_json)
llmp_gtest(import_modelopt_normalize_test TEST_PREFIX unit. PROPERTIES LABELS unit)

add_executable(import_modelopt_models_test import_modelopt_models_test.cc)
target_link_libraries(import_modelopt_models_test PRIVATE llmp_cli llmp_import llmp_components llmp_platform)
llmp_gtest(import_modelopt_models_test TEST_PREFIX unit. PROPERTIES LABELS models)

add_executable(import_indexed_parts_test import_indexed_parts_test.cc)
target_link_libraries(import_indexed_parts_test PRIVATE llmp_import)
llmp_gtest(import_indexed_parts_test TEST_PREFIX unit. PROPERTIES LABELS unit)

add_executable(import_route_test import_route_test.cc)
target_link_libraries(import_route_test PRIVATE llmp_import)
llmp_gtest(import_route_test TEST_PREFIX unit. PROPERTIES LABELS unit)

# Import-time BF16 folds and complete multipart value-head permutations.
add_executable(import_transforms_test import_transforms_test.cc)
target_link_libraries(import_transforms_test PRIVATE llmp_import llmp_base)
llmp_gtest(import_transforms_test TEST_PREFIX unit. PROPERTIES LABELS unit)

add_executable(import_normalize_test import_normalize_test.cc)
target_link_libraries(import_normalize_test PRIVATE llmp_import llmp_model llmp_json llmp_base)
llmp_gtest(import_normalize_test TEST_PREFIX unit. PROPERTIES LABELS unit
             ENVIRONMENT "LLMP_TEST_DATA=${CMAKE_CURRENT_SOURCE_DIR}/data")

# The readers on the current models' checkpoints: the Gemma 2 and 3 source
# GGUFs' specs equal to their v0 artifacts', the split GGUFs, Qwen3.8's
# ModelOpt checkpoint, Qwen-Image's pipeline against its v0 components,
# the MLX checkpoint decoded against Qwen3.8's native one, the EXL3
# fixtures against their v0 imports.
add_executable(import_models_test import_models_test.cc)
target_link_libraries(import_models_test PRIVATE llmp_import llmp_spec llmp_model llmp_artifact
                                                 llmp_reference_kernels)
llmp_gtest(import_models_test TEST_PREFIX unit. PROPERTIES LABELS "unit\;models")

add_executable(import_writer_test import_writer_test.cc)
target_link_libraries(import_writer_test PRIVATE llmp_cli llmp_import llmp_components llmp_spec llmp_artifact llmp_base)
llmp_gtest(import_writer_test TEST_PREFIX unit. PROPERTIES LABELS unit)

add_executable(corpus_score_test corpus_score_test.cc)
target_link_libraries(corpus_score_test PRIVATE llmp_corpus_score llmp_fake_step_source
                         llmp_fake_device_backend)
llmp_gtest(corpus_score_test TEST_PREFIX unit. PROPERTIES LABELS unit
             ENVIRONMENT "LLMP_TEST_SCRATCH=${CMAKE_CURRENT_BINARY_DIR}/scratch")
