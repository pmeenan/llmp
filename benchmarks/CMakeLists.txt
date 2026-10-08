# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0

# Measurement harnesses: built with the rest, never run by CTest. Each
# prints aggregate tables that a report under docs/experiments/ records.

# Shared exact host mask formulas; CPU-only, no model or device acquisition.
add_executable(llmp_host_mask_probe host_mask_probe.cc)
target_link_libraries(llmp_host_mask_probe PRIVATE llmp_model llmp_warnings)

add_library(llmp_draft_vocab_contract STATIC qwen38_vocab.cc)
target_include_directories(llmp_draft_vocab_contract PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(llmp_draft_vocab_contract PRIVATE llmp_json llmp_chat
                                                       llmp_model llmp_tokenizer llmp_warnings)

add_library(llmp_vocab_reference STATIC qwen38_reference.cc)
target_include_directories(llmp_vocab_reference PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(llmp_vocab_reference PRIVATE llmp_base llmp_platform llmp_warnings)

# Offline selected-draft-head preparation with the pinned GGML quantizer.
add_executable(llmp_qwen38_quantize_head qwen38_quantize_head.cc)
target_link_libraries(llmp_qwen38_quantize_head PRIVATE llmp_ggml_base llmp_base llmp_warnings)

# D-048's lanes: wakeup latency against polling, lane throughput, and the
# completion board's round trip (docs/experiments/task-lanes/).
add_executable(llmp_lanes_bench lanes_bench.cc)
target_link_libraries(llmp_lanes_bench PRIVATE llmp_scheduler llmp_warnings)

# Host sampling's cost a draw at the served vocabularies (execution/sampling.h;
# docs/tokenizer.md#sampling).
add_executable(llmp_sampling_bench sampling_bench.cc)
target_link_libraries(llmp_sampling_bench PRIVATE llmp_sampling llmp_warnings)

# The retained-backing comparison's deterministic replay of the swap trace
# on the fake provider (docs/experiments/retained-backing/).
add_subdirectory(retained_backing)

# CPU-only frozen task construction/scoring; never loads a device or model.
add_library(llmp_long_context_tasks STATIC long_context_tasks.cc)
target_include_directories(llmp_long_context_tasks PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(llmp_long_context_tasks PUBLIC llmp_json PRIVATE llmp_warnings)
add_executable(llmp_long_context_task_prepare long_context_tasks_main.cc)
target_link_libraries(llmp_long_context_task_prepare PRIVATE llmp_long_context_tasks
                      llmp_artifact llmp_tokenizer llmp_chat llmp_base llmp_warnings)

# CPU-only arbitrary Gemma text/actual-template ID export; no device/model loading.
add_executable(llmp_gemma_tokenize gemma_tokenize.cc)
target_link_libraries(llmp_gemma_tokenize PRIVATE llmp_tokenizer llmp_chat
                                                llmp_base llmp_warnings)

# Storage queue depth and request size, direct into host VMM through the
# providers (docs/experiments/storage-queue/). Needs a CUDA device.
if(LLMP_CUDA)
  add_executable(llmp_plain_token_probe plain_token_probe.cc)
  target_link_libraries(llmp_plain_token_probe PRIVATE llmp_engine llmp_base llmp_warnings)
  target_link_options(llmp_plain_token_probe PRIVATE
                     "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../lib/llmp")

  add_executable(llmp_turn_reuse turn_reuse.cc)
  target_link_libraries(llmp_turn_reuse PRIVATE llmp_runtime_serving llmp_base llmp_warnings)
  target_link_options(llmp_turn_reuse PRIVATE
                     "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../lib/llmp")

  # Exact long-context swap controls without a second whole-state allocation.
  add_executable(llmp_gemma3_max_context gemma3_max_context.cc)
  target_link_libraries(llmp_gemma3_max_context PRIVATE llmp_runtime_serving llmp_base llmp_warnings)
  target_link_options(llmp_gemma3_max_context PRIVATE
                     "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../lib/llmp")

  add_executable(llmp_swap_pager swap_pager.cc)
  target_link_libraries(llmp_swap_pager PRIVATE llmp_runtime_serving llmp_base llmp_warnings)
  target_link_options(llmp_swap_pager PRIVATE
                     "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../lib/llmp")

  add_executable(llmp_long_swap long_swap.cc)
  target_link_libraries(llmp_long_swap PRIVATE llmp_runtime_serving llmp_base llmp_warnings)
  target_link_options(llmp_long_swap PRIVATE
                     "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../lib/llmp")

  add_executable(llmp_storage_bench storage_bench.cc)
  target_link_libraries(llmp_storage_bench PRIVATE llmp_cuda_device_memory llmp_storage llmp_warnings)

  # Page-in throughput through the scheduler and its lanes: the landing
  # zone into device VMM against direct reads in place into host VMM
  # (D-081; docs/experiments/pagein-perf/).
  add_executable(llmp_pagein_bench pagein_bench.cc)
  target_include_directories(llmp_pagein_bench SYSTEM PRIVATE
                             "${LLMP_SDK}/cuda/targets/${LLMP_CUDA_TARGET_DIR}/include")
  target_link_libraries(llmp_pagein_bench PRIVATE llmp_cuda_device_memory
                                                    llmp_scheduler llmp_storage llmp_warnings)

  # Which OS counters include VMM backing (docs/experiments/vmm-counters/).
  add_executable(llmp_vmm_counters vmm_counters.cc)
  target_include_directories(llmp_vmm_counters SYSTEM PRIVATE
                             "${LLMP_SDK}/cuda/targets/${LLMP_CUDA_TARGET_DIR}/include")
  target_link_libraries(llmp_vmm_counters PRIVATE llmp_cuda_device_memory llmp_warnings)

  # Host time per launch of RMSNorm-mul, for each implementation a plan can
  # select, through the launch context and around it (BP-F4 in backend-proof
  # P1; docs/experiments/launch-overhead/). Needs a GB10.
  add_executable(llmp_launch_bench launch_bench.cc)
  target_include_directories(llmp_launch_bench SYSTEM PRIVATE
                             "${LLMP_SDK}/cuda/targets/${LLMP_CUDA_TARGET_DIR}/include")
  target_link_libraries(llmp_launch_bench PRIVATE llmp_ggml_kernels llmp_cuda_device_memory
                                                    llmp_warnings)
  # It links cuBLAS dynamically (D-076), found as the GPU tests find it.
  file(RELATIVE_PATH runtime "${CMAKE_CURRENT_BINARY_DIR}" "${PROJECT_BINARY_DIR}/cublas")
  target_link_options(llmp_launch_bench PRIVATE "LINKER:--enable-new-dtags,-rpath,$ORIGIN/${runtime}")

  # Family-neutral sparse-attention and paired expert preparation controls.
  add_executable(llmp_prefill_transfer_bench prefill_transfer_bench.cc)
  target_include_directories(llmp_prefill_transfer_bench SYSTEM PRIVATE
                             "${LLMP_SDK}/cuda/targets/${LLMP_CUDA_TARGET_DIR}/include")
  target_link_libraries(llmp_prefill_transfer_bench PRIVATE llmp_ggml_kernels
                                                           llmp_warnings)
  target_link_options(llmp_prefill_transfer_bench PRIVATE
                      "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../cublas")

  # BP-F1: GGML kernels with their operands in cudaMalloc, host VMM or device VMM
  # memory, one timing block per process (docs/experiments/backend-proof-p1/).
  # Finds the SDK's cuBLAS in build/<preset>/cublas, as the GPU tests do.
  add_executable(llmp_ggml_vmm_bench ggml_vmm_bench.cc)
  target_include_directories(llmp_ggml_vmm_bench SYSTEM PRIVATE
                             "${LLMP_SDK}/cuda/targets/${LLMP_CUDA_TARGET_DIR}/include")
  target_link_libraries(llmp_ggml_vmm_bench PRIVATE llmp_ggml_kernels llmp_cuda_device_memory
                                                      llmp_warnings)
  target_link_options(llmp_ggml_vmm_bench PRIVATE
                      "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../cublas")

  # Backend-proof P2, rung 3: the FP16 fixture from its v0 artifact on
  # cudaMalloc memory, recorded for the FP16 gate's plan comparison or read
  # for the allocation census (docs/experiments/backend-proof-p2/). Links the
  # launch recorder; finds the SDK's cuBLAS in build/<preset>/cublas.
  add_executable(llmp_fp16_exec fp16_exec.cc)
  target_include_directories(llmp_fp16_exec SYSTEM PRIVATE
                             "${LLMP_SDK}/cuda/targets/${LLMP_CUDA_TARGET_DIR}/include")
  target_link_libraries(llmp_fp16_exec PRIVATE llmp_ggml_kernels llmp_cuda_device_memory
                                                 llmp_artifact llmp_model llmp_launch_recorder
                                                 llmp_warnings)
  target_link_options(llmp_fp16_exec PRIVATE "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../cublas")

  # llmp.vecq's launch configurations on DeepSeek V4 Flash's product shapes
  # (kernels/ggml/dsv4_fast.h; docs/experiments/dsv4-decode/).
  add_executable(llmp_vecq_bench vecq_bench.cc)
  target_include_directories(llmp_vecq_bench SYSTEM PRIVATE
                             "${LLMP_SDK}/cuda/targets/${LLMP_CUDA_TARGET_DIR}/include")
  target_link_libraries(llmp_vecq_bench PRIVATE llmp_ggml_kernels llmp_cuda_device_memory
                                                  llmp_warnings)
  target_link_options(llmp_vecq_bench PRIVATE "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../cublas")

  # Closed C2 ordinary-MMVQ preparation factor, operator-only/default-unmodified.
  add_executable(llmp_gemma31_mmvq_pair_bench EXCLUDE_FROM_ALL
                 gemma31_mmvq_pair_bench.cc gemma31_mmvq_pair.cu)
  target_include_directories(llmp_gemma31_mmvq_pair_bench SYSTEM PRIVATE
                             "${LLMP_SDK}/cuda/targets/${LLMP_CUDA_TARGET_DIR}/include")
  target_link_libraries(llmp_gemma31_mmvq_pair_bench PRIVATE llmp_ggml_kernels
                          llmp_cuda_device_memory llmp_warnings)
  set_target_properties(llmp_gemma31_mmvq_pair_bench PROPERTIES CUDA_ARCHITECTURES 121a-real)
  target_link_options(llmp_gemma31_mmvq_pair_bench PRIVATE
                      "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../cublas")

  add_executable(llmp_gemma_quant_bench gemma_quant.cc)
  target_include_directories(llmp_gemma_quant_bench SYSTEM PRIVATE
                             "${LLMP_SDK}/cuda/targets/${LLMP_CUDA_TARGET_DIR}/include")
  target_link_libraries(llmp_gemma_quant_bench PRIVATE llmp_ggml_kernels llmp_cuda_device_memory
                                                        llmp_warnings)
  target_link_options(llmp_gemma_quant_bench PRIVATE "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../cublas")

  add_executable(llmp_gemma_geglu_quant_bench gemma_geglu_quant.cc)
  target_include_directories(llmp_gemma_geglu_quant_bench SYSTEM PRIVATE
                             "${LLMP_SDK}/cuda/targets/${LLMP_CUDA_TARGET_DIR}/include")
  target_link_libraries(llmp_gemma_geglu_quant_bench PRIVATE llmp_ggml_kernels
                          llmp_cuda_device_memory llmp_warnings)
  target_link_options(llmp_gemma_geglu_quant_bench PRIVATE
                      "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../cublas")

  add_executable(llmp_gemma_attention_bench gemma_attention.cc)
  target_include_directories(llmp_gemma_attention_bench SYSTEM PRIVATE
                             "${LLMP_SDK}/cuda/targets/${LLMP_CUDA_TARGET_DIR}/include")
  target_link_libraries(llmp_gemma_attention_bench PRIVATE llmp_ggml_kernels llmp_cuda_device_memory
                                                        llmp_warnings)
  target_link_options(llmp_gemma_attention_bench PRIVATE "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../cublas")

  # M3's first model slice: DeepSeek V4 Flash, native and resident, from its
  # v0 prepared artifact (docs/experiments/dsv4-native/).
  add_executable(llmp_dsv4_exec dsv4_exec.cc)
  target_include_directories(llmp_dsv4_exec SYSTEM PRIVATE
                             "${LLMP_SDK}/cuda/targets/${LLMP_CUDA_TARGET_DIR}/include")
  target_link_libraries(llmp_dsv4_exec PRIVATE llmp_ggml_kernels llmp_cuda_device_memory
                                                 llmp_artifact llmp_model llmp_warnings
                                                 llmp_long_context_tasks)
  target_link_options(llmp_dsv4_exec PRIVATE "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../cublas")

  # M3's second model slice: Qwen3.8 Flash Next (Mia's NVFP4 and MXFP8
  # checkpoint), native and resident (docs/experiments/qwen38-native/),
  # planning through the path the paged runner shares (engine/qwen38_plan.h).
  add_executable(llmp_qwen38_exec qwen38_exec.cc)
  target_include_directories(llmp_qwen38_exec SYSTEM PRIVATE
                             "${LLMP_SDK}/cuda/targets/${LLMP_CUDA_TARGET_DIR}/include")
  target_link_libraries(llmp_qwen38_exec PRIVATE llmp_engine llmp_ggml_kernels
                                                   llmp_cuda_device_memory llmp_artifact
                                                   llmp_model llmp_warnings)
  target_link_options(llmp_qwen38_exec PRIVATE "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../cublas")

  # M3's third model slice: the Qwen-Image-2.1 BF16 pipeline, native, from
  # its component artifacts and their composition
  # (docs/experiments/qwen-image-native/), with the native tokenizer and the
  # prompt renderer.
  add_executable(llmp_qwen_image_exec qwen_image_exec.cc)
  target_include_directories(llmp_qwen_image_exec SYSTEM PRIVATE
                             "${LLMP_SDK}/cuda/targets/${LLMP_CUDA_TARGET_DIR}/include")
  target_link_libraries(llmp_qwen_image_exec PRIVATE llmp_image_kernels llmp_ggml_kernels
                                                       llmp_cuda_device_memory llmp_artifact
                                                       llmp_model llmp_tokenizer llmp_chat
                                                       llmp_warnings)
  target_link_options(llmp_qwen_image_exec PRIVATE
                      "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../cublas")
  # Its products' cuBLASLt candidates, timed, for the algorithms the image
  # kernels pin (kernels/image/gemm.h).
  add_executable(llmp_qwen_image_gemm_tune qwen_image_gemm_tune.cc)
  target_include_directories(llmp_qwen_image_gemm_tune SYSTEM PRIVATE
                             "${LLMP_SDK}/cuda/targets/${LLMP_CUDA_TARGET_DIR}/include")
  target_link_libraries(llmp_qwen_image_gemm_tune PRIVATE llmp_image_kernels llmp_warnings)
  target_link_options(llmp_qwen_image_gemm_tune PRIVATE
                      "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../cublas")

  # Qwen3.8's small-column BF16 products, with their actual output types.
  add_executable(llmp_qwen38_gemm_tune qwen38_gemm_tune.cc)
  target_include_directories(llmp_qwen38_gemm_tune SYSTEM PRIVATE
                             "${LLMP_SDK}/cuda/targets/${LLMP_CUDA_TARGET_DIR}/include")
  target_link_libraries(llmp_qwen38_gemm_tune PRIVATE llmp_ggml_kernels llmp_warnings)
  target_link_options(llmp_qwen38_gemm_tune PRIVATE
                      "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../cublas")

  # GB10-only scheduling sweep, run only by explicit benchmark invocation.
  add_executable(llmp_qwen38_mxfp8_tune qwen38_mxfp8_tune.cc qwen38_mxfp8_tune_kernels.cu)
  target_include_directories(llmp_qwen38_mxfp8_tune SYSTEM PRIVATE
                             "${LLMP_SDK}/cuda/targets/${LLMP_CUDA_TARGET_DIR}/include")
  target_link_libraries(llmp_qwen38_mxfp8_tune PRIVATE llmp_ggml_kernels llmp_warnings)
  set_target_properties(llmp_qwen38_mxfp8_tune PROPERTIES CUDA_ARCHITECTURES 121a-real)
  set_source_files_properties(qwen38_mxfp8_tune_kernels.cu PROPERTIES COMPILE_OPTIONS "-Xptxas=-v")
  target_link_options(llmp_qwen38_mxfp8_tune PRIVATE
                      "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../cublas")

  # The paged harnesses' model runners: the FP16 fixture and an EXL3
  # fixture, each a model on the engine's paged node
  # (tests/support/paged_node.h), so one process can hold either or both
  # (BP-S3), and the M3 harnesses' helpers (dsv4_common.h) over the engine's
  # M3 runners (src/engine/, under their harness names in engine_names.h).
  # Their binaries link the launch recorder themselves (an object library,
  # whose wrappers must be in the link).
  add_library(llmp_paged_runners STATIC fp16_runner.cc fp16_common.cc exl3_runner.cc
                                          exl3_common.cc dsv4_common.cc)
  target_include_directories(llmp_paged_runners SYSTEM PRIVATE
                             "${LLMP_SDK}/cuda/targets/${LLMP_CUDA_TARGET_DIR}/include")
  target_include_directories(llmp_paged_runners PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
  target_link_libraries(llmp_paged_runners PUBLIC llmp_paged_node llmp_engine
                                                    llmp_exl3_qwen2 llmp_ggml_kernels
                                                    llmp_scheduler llmp_cuda_device_memory
                                                    llmp_artifact llmp_model
                                                    llmp_plan_record llmp_storage
                                             PRIVATE llmp_warnings)

  # Backend-proof P2, rungs 4 and 5: the same fixture paged into device VMM
  # through the landing zone by the scheduler and its lanes, executed as
  # device jobs under leases, evicted and restored
  # (docs/experiments/backend-proof-p2/). Links the launch recorder.
  add_executable(llmp_fp16_paged fp16_paged.cc)
  target_link_libraries(llmp_fp16_paged PRIVATE llmp_paged_runners llmp_launch_recorder
                                                  llmp_warnings)
  target_link_options(llmp_fp16_paged PRIVATE "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../cublas")

  # BP-S3: the FP16 fixture and an EXL3 fixture alternating on one paged
  # node, under a budget that forces each to evict the other's weights
  # (docs/experiments/backend-proof/). Links the launch recorder.
  add_executable(llmp_alternate_paged alternate_paged.cc)
  target_include_directories(llmp_alternate_paged SYSTEM PRIVATE
                             "${LLMP_SDK}/cuda/targets/${LLMP_CUDA_TARGET_DIR}/include")
  target_link_libraries(llmp_alternate_paged PRIVATE llmp_paged_runners llmp_launch_recorder
                                                       llmp_warnings)
  target_link_options(llmp_alternate_paged PRIVATE
                      "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../cublas")

  # M3's swap runner: DeepSeek V4 Flash and the FP16 fixture on one paged
  # node, full swaps A→B→A with each part timed, state spilled and restored
  # (docs/experiments/fast-swap/swap.md). A harness: it runs on the test
  # harness's paged node and links the launch recorder.
  add_executable(llmp_swap_runner swap_runner.cc)
  target_link_libraries(llmp_swap_runner PRIVATE llmp_paged_runners llmp_launch_recorder
                                                   llmp_tokenizer llmp_warnings)
  target_link_options(llmp_swap_runner PRIVATE "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../cublas")

  # M3's swap pairs: two of DeepSeek V4 Flash, Qwen3.8 Flash Next and the
  # Qwen-Image-2.1 pipeline on one paged node, full swaps A→B→A with each
  # part timed (docs/experiments/fast-swap/swap.md), over the engine's
  # runners. The harness beside llmp-runtime's swap-table, which runs the
  # same protocol over the configured models in one process.
  add_executable(llmp_swap_pairs swap_pairs.cc)
  target_include_directories(llmp_swap_pairs SYSTEM PRIVATE
                             "${LLMP_SDK}/cuda/targets/${LLMP_CUDA_TARGET_DIR}/include")
  target_link_libraries(llmp_swap_pairs PRIVATE llmp_paged_runners llmp_launch_recorder
                                                  llmp_image_kernels llmp_tokenizer
                                                  llmp_chat llmp_warnings)
  target_link_options(llmp_swap_pairs PRIVATE "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../cublas")

  # M3's speculation harness: DeepSeek V4 Flash with its DSpark drafter on
  # the paged node, greedy and sampled speculation, the forced-rejection
  # checks and speed (docs/experiments/dspark/). A harness for now; it
  # links the native tokenizer and chat renderer.
  add_executable(llmp_gemma_retained_prefill gemma_retained_prefill.cc)
  target_link_libraries(llmp_gemma_retained_prefill PRIVATE llmp_engine llmp_warnings)
  target_link_options(llmp_gemma_retained_prefill PRIVATE "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../cublas")

  add_executable(llmp_gemma_prefill gemma_prefill.cc)
  target_link_libraries(llmp_gemma_prefill PRIVATE llmp_engine llmp_warnings)
  target_link_options(llmp_gemma_prefill PRIVATE "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../cublas")

  # Manual closed 26B routing keep diagnostic; production planner stays intact.
  add_executable(llmp_gemma_quality_keep28 gemma_quality_keep28.cc
                   gemma_quality_keep28_wrap.cc)
  target_link_libraries(llmp_gemma_quality_keep28 PRIVATE llmp_engine llmp_warnings)
  target_link_options(llmp_gemma_quality_keep28 PRIVATE
      "LINKER:--wrap=_ZN4llmp6engine15PlanGemma4ChunkERKNS0_11Gemma4ModelERKNS_7kernels4ggml16Gemma4ChunkShapeERKNS5_13DeviceChoicesEmmSt4spanIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEELm18446744073709551615EE"
      "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../cublas")

  # Manual fixed-capacity schedule qualification; never registered with CTest.
  add_executable(llmp_gemma_quality_schedule gemma_quality_schedule.cc)
  target_link_libraries(llmp_gemma_quality_schedule PRIVATE llmp_engine llmp_warnings)
  target_link_options(llmp_gemma_quality_schedule PRIVATE "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../cublas")

  add_executable(llmp_gemma_quality gemma_quality.cc)
  target_link_libraries(llmp_gemma_quality PRIVATE llmp_engine llmp_warnings)
  target_link_options(llmp_gemma_quality PRIVATE "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../cublas")

  add_executable(llmp_gemma26_continuation
                   ../docs/experiments/gemma26-production/native_continuation.cc)
  target_link_libraries(llmp_gemma26_continuation PRIVATE llmp_engine llmp_warnings)
  target_link_options(llmp_gemma26_continuation PRIVATE
                     "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../cublas")

  add_executable(llmp_gemma_runner gemma_runner.cc)
  # Manual native-origin first-local operand capture. Link wrapping adds no
  # graph keep, planner policy or production observer.
  add_executable(llmp_gemma_attention_capture gemma_attention_capture.cc
                   gemma_attention_capture_wrap.cc)
  target_include_directories(llmp_gemma_attention_capture SYSTEM PRIVATE
                             "${LLMP_SDK}/cuda/targets/${LLMP_CUDA_TARGET_DIR}/include")
  target_link_libraries(llmp_gemma_attention_capture PRIVATE llmp_engine llmp_warnings)
  target_link_options(llmp_gemma_attention_capture PRIVATE
      "LINKER:--wrap=_ZN4llmp7kernels4ggml15FlashAttnVec256ERNS1_13LaunchContextEP11ggml_tensor"
      "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../cublas")

  add_executable(llmp_gemma_attention_replay gemma_attention_replay.cc)
  target_link_libraries(llmp_gemma_attention_replay PRIVATE llmp_engine llmp_warnings)
  target_link_options(llmp_gemma_attention_replay PRIVATE
                      "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../cublas")

  # Separate first-global D512 carrier/replay; measured local helpers remain fixed.
  add_executable(llmp_gemma_attention_global_capture gemma_attention_global_capture.cc
                 gemma_attention_global_capture_wrap.cc)
  target_include_directories(llmp_gemma_attention_global_capture SYSTEM PRIVATE
      "${LLMP_SDK}/cuda/targets/${LLMP_CUDA_TARGET_DIR}/include")
  target_link_libraries(llmp_gemma_attention_global_capture PRIVATE llmp_engine llmp_warnings)
  target_link_options(llmp_gemma_attention_global_capture PRIVATE
    "LINKER:--wrap=_ZN4llmp7kernels4ggml12FlashAttnMmaERNS1_13LaunchContextEP11ggml_tensorb"
    "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../cublas")
  add_executable(llmp_gemma_attention_global_owner_replay gemma_attention_global_owner_replay.cc)
  target_link_libraries(llmp_gemma_attention_global_owner_replay PRIVATE llmp_engine llmp_warnings)
  target_link_options(llmp_gemma_attention_global_owner_replay PRIVATE "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../cublas")
  # Explicit owner-root MMA operand proof only; no production selector.
  add_executable(llmp_gemma_attention_owner_replay gemma_attention_owner_replay.cc)
  target_link_libraries(llmp_gemma_attention_owner_replay PRIVATE llmp_engine llmp_warnings)
  target_link_options(llmp_gemma_attention_owner_replay PRIVATE
                      "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../cublas")

  # Explicit manual packed versus independent-owner C4 factor. No default emits the op.
  foreach(_owner_target IN ITEMS llmp_gemma_owner_c4 llmp_gemma_owner_c4_metadata)
    if(_owner_target STREQUAL "llmp_gemma_owner_c4")
      set(_owner_source gemma_owner_c4.cc)
    else()
      set(_owner_source gemma_owner_c4_metadata.cc)
    endif()
    add_executable(${_owner_target} ${_owner_source} gemma_owner_c4_wrap.cc)
    target_include_directories(${_owner_target} PRIVATE "${PROJECT_SOURCE_DIR}/tests/support")
    target_link_libraries(${_owner_target} PRIVATE llmp_engine llmp_warnings)
    target_link_options(${_owner_target} PRIVATE
      "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../cublas"
      "LINKER:--wrap=_ZN4llmp7kernels4ggml16BuildGemma4GraphERNS1_11TensorArenaERKNS_5model13Gemma4ProfileERKNS4_13Gemma4BindingERKNS4_17Gemma4StateLayoutERKNS1_16Gemma4ChunkShapeERKNS1_18Gemma4GraphOptionsE"
      "LINKER:--wrap=_ZN4llmp7kernels4ggml18Gemma4GraphTensorsERKNS_5model13Gemma4ProfileEm"
      "LINKER:--wrap=_ZN4llmp7kernels4ggml19PlanFlashAttnOwnersERKNS1_13LaunchContextERKNS1_15FlashAttnOwnersE")
  endforeach()

  add_executable(llmp_gemma_attention_packed
    gemma_attention_packed.cc gemma_attention_packed_wrap.cc)
  target_link_libraries(llmp_gemma_attention_packed PRIVATE llmp_engine llmp_warnings)
  target_link_options(llmp_gemma_attention_packed PRIVATE
    "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../cublas"
    "LINKER:--wrap=_ZN4llmp7kernels4ggml16BuildGemma4GraphERNS1_11TensorArenaERKNS_5model13Gemma4ProfileERKNS4_13Gemma4BindingERKNS4_17Gemma4StateLayoutERKNS1_16Gemma4ChunkShapeERKNS1_18Gemma4GraphOptionsE"
    "LINKER:--wrap=_ZN4llmp7kernels4ggml18Gemma4GraphTensorsERKNS_5model13Gemma4ProfileEm")
  add_executable(llmp_gemma_attention_packed_metadata
    gemma_attention_packed_metadata.cc gemma_attention_packed_wrap.cc)
  target_include_directories(llmp_gemma_attention_packed_metadata PRIVATE
    "${PROJECT_SOURCE_DIR}/tests/support")
  target_link_libraries(llmp_gemma_attention_packed_metadata PRIVATE llmp_engine llmp_warnings)
  target_link_options(llmp_gemma_attention_packed_metadata PRIVATE
    "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../cublas"
    "LINKER:--wrap=_ZN4llmp7kernels4ggml16BuildGemma4GraphERNS1_11TensorArenaERKNS_5model13Gemma4ProfileERKNS4_13Gemma4BindingERKNS4_17Gemma4StateLayoutERKNS1_16Gemma4ChunkShapeERKNS1_18Gemma4GraphOptionsE"
    "LINKER:--wrap=_ZN4llmp7kernels4ggml18Gemma4GraphTensorsERKNS_5model13Gemma4ProfileEm")

if(LLMP_CUDA)
  # Manual scalar layer0 FFN acquisition/replay; no production dispatch change.
  add_executable(llmp_gemma_dense_ffn_capture gemma_dense_ffn_capture.cc
                 gemma_dense_ffn_capture_wrap.cc)
  target_include_directories(llmp_gemma_dense_ffn_capture SYSTEM PRIVATE
                             "${LLMP_SDK}/cuda/targets/${LLMP_CUDA_TARGET_DIR}/include")
  target_link_libraries(llmp_gemma_dense_ffn_capture PRIVATE llmp_engine llmp_warnings)
  target_link_options(llmp_gemma_dense_ffn_capture PRIVATE
      "LINKER:--wrap=_ZN4llmp7kernels4ggml10MulMatVecQERNS1_13LaunchContextEP11ggml_tensor"
      "LINKER:--wrap=_ZNK4llmp6engine12PagedWeights16resource_addressEj"
      "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../cublas")
  add_executable(llmp_gemma_dense_ffn_replay gemma_dense_ffn_replay.cc)
  target_link_libraries(llmp_gemma_dense_ffn_replay PRIVATE llmp_engine llmp_warnings)
  target_link_options(llmp_gemma_dense_ffn_replay PRIVATE
                      "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../cublas")
endif()
  add_executable(llmp_gemma26_attention_packed
    gemma26_attention_packed.cc gemma26_attention_packed_wrap.cc)
  target_link_libraries(llmp_gemma26_attention_packed PRIVATE llmp_engine llmp_warnings)
  target_link_options(llmp_gemma26_attention_packed PRIVATE
    "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../cublas"
    "LINKER:--wrap=_ZN4llmp7kernels4ggml16BuildGemma4GraphERNS1_11TensorArenaERKNS_5model13Gemma4ProfileERKNS4_13Gemma4BindingERKNS4_17Gemma4StateLayoutERKNS1_16Gemma4ChunkShapeERKNS1_18Gemma4GraphOptionsE"
    "LINKER:--wrap=_ZN4llmp7kernels4ggml18Gemma4GraphTensorsERKNS_5model13Gemma4ProfileEm")
  add_executable(llmp_gemma26_attention_compound
    gemma26_attention_compound.cc gemma26_attention_packed_wrap.cc)
  target_link_libraries(llmp_gemma26_attention_compound PRIVATE llmp_engine llmp_warnings)
  target_link_options(llmp_gemma26_attention_compound PRIVATE
    "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../cublas"
    "LINKER:--wrap=_ZN4llmp7kernels4ggml16BuildGemma4GraphERNS1_11TensorArenaERKNS_5model13Gemma4ProfileERKNS4_13Gemma4BindingERKNS4_17Gemma4StateLayoutERKNS1_16Gemma4ChunkShapeERKNS1_18Gemma4GraphOptionsE"
    "LINKER:--wrap=_ZN4llmp7kernels4ggml18Gemma4GraphTensorsERKNS_5model13Gemma4ProfileEm")
  # Manual common-input late-prefill MoE diagnostic; no CTest/model policy.
  add_executable(llmp_gemma26_late_moe gemma26_late_moe.cc)
  target_link_libraries(llmp_gemma26_late_moe PRIVATE llmp_engine llmp_warnings)
  target_link_options(llmp_gemma26_late_moe PRIVATE
                      "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../cublas")
  add_executable(llmp_gemma26_attention_packed_metadata
    gemma26_attention_packed_metadata.cc gemma26_attention_packed_wrap.cc)
  target_include_directories(llmp_gemma26_attention_packed_metadata PRIVATE
    "${PROJECT_SOURCE_DIR}/tests/support")
  target_link_libraries(llmp_gemma26_attention_packed_metadata PRIVATE llmp_engine llmp_warnings)
  target_link_options(llmp_gemma26_attention_packed_metadata PRIVATE
    "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../cublas"
    "LINKER:--wrap=_ZN4llmp7kernels4ggml16BuildGemma4GraphERNS1_11TensorArenaERKNS_5model13Gemma4ProfileERKNS4_13Gemma4BindingERKNS4_17Gemma4StateLayoutERKNS1_16Gemma4ChunkShapeERKNS1_18Gemma4GraphOptionsE"
    "LINKER:--wrap=_ZN4llmp7kernels4ggml18Gemma4GraphTensorsERKNS_5model13Gemma4ProfileEm")
  # Private prefix-only routing diagnostic; original joined target stays intact.
  add_executable(llmp_gemma_joined_prefix_keep gemma_joined.cc gemma_joined_prefix_keep.cc)
  target_link_libraries(llmp_gemma_joined_prefix_keep PRIVATE llmp_engine llmp_warnings)
  target_link_options(llmp_gemma_joined_prefix_keep PRIVATE
      "LINKER:--wrap=_ZN4llmp6engine15PlanGemma4ChunkERKNS0_11Gemma4ModelERKNS_7kernels4ggml16Gemma4ChunkShapeERKNS5_13DeviceChoicesEmmSt4spanIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEELm18446744073709551615EE"
      "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../cublas")
  add_executable(llmp_gemma_prefix_keep_contract gemma_joined_prefix_keep_contract.cc
                   gemma_joined_prefix_keep.cc)
  target_link_libraries(llmp_gemma_prefix_keep_contract PRIVATE llmp_engine llmp_warnings)
  add_executable(llmp_gemma_joined gemma_joined.cc)
  target_link_libraries(llmp_gemma_joined PRIVATE llmp_engine llmp_warnings)
  target_link_options(llmp_gemma_joined PRIVATE
                      "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../cublas")
  target_link_libraries(llmp_gemma_runner PRIVATE llmp_engine llmp_warnings)
  target_link_options(llmp_gemma_runner PRIVATE "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../cublas")

  add_executable(llmp_spec_runner spec_runner.cc)
  target_link_libraries(llmp_spec_runner PRIVATE llmp_paged_runners llmp_launch_recorder
                                                   llmp_tokenizer llmp_chat llmp_sampling
                                                   llmp_warnings)
  target_link_options(llmp_spec_runner PRIVATE "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../cublas")

  # M3's speculation harness for Qwen3.8 Flash Next with its MTP drafter on
  # the paged node (docs/experiments/qwen38-mtp/). A harness for now; it
  # links the native tokenizer and chat renderer.
  # Actual scalar prompt prediction, initialized state and continuation controls.
  add_executable(llmp_prefill_prediction_probe prefill_prediction_probe.cc)
  target_link_libraries(llmp_prefill_prediction_probe PRIVATE llmp_runtime_serving llmp_base llmp_warnings)
  target_link_options(llmp_prefill_prediction_probe PRIVATE "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../cublas")

  # Same-binary descriptor/funding and startup-latency checks; never loads weights.
  add_executable(llmp_startup_measurement_probe startup_measurement_probe.cc)
  target_link_libraries(llmp_startup_measurement_probe PRIVATE llmp_engine llmp_warnings)
  target_link_options(llmp_startup_measurement_probe PRIVATE "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../cublas")

  add_executable(llmp_qwen38_spec qwen38_spec.cc)
  target_link_libraries(llmp_qwen38_spec PRIVATE llmp_paged_runners llmp_launch_recorder
                                                   llmp_tokenizer llmp_chat llmp_sampling
                                                   llmp_draft_vocab_contract
                                                   llmp_vocab_reference
                                                   llmp_runtime
                                                   llmp_warnings)
  target_link_options(llmp_qwen38_spec PRIVATE "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../cublas")

  # GGUF target waves: fixed-history logits/state against solo or unjoined steps.
  add_executable(llmp_qwen38_gguf_wave qwen38_gguf_wave.cc)
  target_link_libraries(llmp_qwen38_gguf_wave PRIVATE llmp_paged_runners
                        llmp_base llmp_warnings)
  target_link_options(llmp_qwen38_gguf_wave PRIVATE "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../cublas")

  # Temporary C2 shared-row mechanism proof, no serving/default selection.
  add_executable(llmp_qwen38_batch qwen38_batch_main.cc qwen38_batch.cc
                 qwen38_batch_graph.cc qwen38_batch_generation.cc)
  target_link_libraries(llmp_qwen38_batch PRIVATE llmp_paged_runners
                        llmp_vocab_reference llmp_json llmp_base llmp_warnings)
  target_link_options(llmp_qwen38_batch PRIVATE "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../cublas")

  # Backend-proof P3, the per-linear sweep (BP-N5): every real projection of
  # an EXL3 fixture through llmpalooza's launchers at upstream's forced plans,
  # hashed for comparison with upstream's outputs
  # (docs/experiments/backend-proof-p3/). Finds the SDK's cuBLAS in
  # build/<preset>/cublas.
  add_executable(llmp_exl3_linear_sweep exl3_linear_sweep.cc)
  target_link_libraries(llmp_exl3_linear_sweep PRIVATE llmp_exl3_kernels llmp_cuda_device_memory
                                                         llmp_artifact llmp_warnings)
  target_link_options(llmp_exl3_linear_sweep PRIVATE
                      "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../cublas")

  # Backend-proof P3, the native EXL3 model: an EXL3 fixture through the
  # native operation plan, for the plan gate, Tier E (operation level) and
  # Tier C (docs/experiments/backend-proof-p3/). Links the launch recorder.
  add_executable(llmp_exl3_exec exl3_exec.cc exl3_common.cc)
  target_link_libraries(llmp_exl3_exec PRIVATE llmp_exl3_qwen2 llmp_cuda_device_memory
                                                 llmp_artifact llmp_model
                                                 llmp_launch_recorder llmp_warnings)
  target_link_options(llmp_exl3_exec PRIVATE "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../cublas")

  # Backend-proof P3, rungs 4 and 5 for EXL3: the same fixture paged into
  # device VMM through the landing zone by the scheduler and its lanes,
  # executed as device jobs under leases, evicted and restored
  # (docs/experiments/backend-proof-p3/). Links the launch recorder.
  add_executable(llmp_exl3_paged exl3_paged.cc)
  target_link_libraries(llmp_exl3_paged PRIVATE llmp_paged_runners llmp_launch_recorder
                                                  llmp_warnings)
  target_link_options(llmp_exl3_paged PRIVATE "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../cublas")

  # Why host VMM slowed BP-F1's products, and what the alternatives cost:
  # microkernels over each kind of memory, copies between them, and GGML
  # products with each buffer group placed separately
  # (docs/experiments/host-vmm-diagnosis/). Diagnostic, not a gate. Its
  # kernels are the GB10's alone, sm_121a in every profile: the diagnosis
  # is of the GB10's L2, so no discrete target builds them (D-082).
  add_executable(llmp_vmm_diag_bench vmm_diag_bench.cc vmm_diag_kernels.cu)
  target_include_directories(llmp_vmm_diag_bench SYSTEM PRIVATE
                             "${LLMP_SDK}/cuda/targets/${LLMP_CUDA_TARGET_DIR}/include")
  target_link_libraries(llmp_vmm_diag_bench PRIVATE llmp_ggml_kernels llmp_cuda_device_memory
                                                      llmp_storage llmp_warnings)
  set_target_properties(llmp_vmm_diag_bench PROPERTIES CUDA_ARCHITECTURES 121a-real)
  target_link_options(llmp_vmm_diag_bench PRIVATE
                      "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../cublas")

  # The runtime wake: from a step's fence to the next step's launch, for
  # each candidate way of waiting, and the runtime's own path on the paged
  # node (docs/experiments/runtime-wake/). Its kernels read the GB10's
  # global timer: sm_121a only, as the host-VMM diagnosis's.
  add_executable(llmp_wake_bench wake_bench.cc wake_bench_kernels.cu)
  target_include_directories(llmp_wake_bench SYSTEM PRIVATE
                             "${LLMP_SDK}/cuda/targets/${LLMP_CUDA_TARGET_DIR}/include")
  target_link_libraries(llmp_wake_bench PRIVATE llmp_paged_node llmp_warnings)
  set_target_properties(llmp_wake_bench PROPERTIES CUDA_ARCHITECTURES 121a-real)
  target_link_options(llmp_wake_bench PRIVATE "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../cublas")
endif()

if(LLMP_CUDA)
# The existing HTTP route with internal Gemma diagnostic controls only.
add_executable(llmp_gemma_joined_http gemma_joined_http.cc)
target_link_libraries(llmp_gemma_joined_http PRIVATE llmp_runtime_serving llmp_platform llmp_warnings)
target_link_options(llmp_gemma_joined_http PRIVATE "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../lib/llmp")
endif()

if(LLMP_CUDA)
  # Explicit C1 draft/verify diagnostic; no serving policy or CTest registration.
  add_executable(llmp_gemma_greedy_reference gemma_greedy_reference.cc)
  target_link_libraries(llmp_gemma_greedy_reference PRIVATE llmp_engine llmp_json llmp_warnings)
  target_link_options(llmp_gemma_greedy_reference PRIVATE "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../lib/llmp")
  # Original stage-zero immutable inputs; manual diagnostic, never CTest.
  add_executable(llmp_gemma_assistant_fixture gemma_assistant_fixture.cc)
  target_link_libraries(llmp_gemma_assistant_fixture PRIVATE llmp_engine llmp_json llmp_warnings)
  target_link_options(llmp_gemma_assistant_fixture PRIVATE "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../lib/llmp")
endif()

if(LLMP_CUDA)
  # Two-owner original stage-zero inputs, manual diagnostic only.
  add_executable(llmp_gemma_assistant_wave_fixture gemma_assistant_wave_fixture.cc)
  target_link_libraries(llmp_gemma_assistant_wave_fixture PRIVATE llmp_engine llmp_json llmp_warnings)
  target_link_options(llmp_gemma_assistant_wave_fixture PRIVATE "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../lib/llmp")
endif()

if(LLMP_CUDA)
  # Closed dense31 scalar norm screen; manual only, never CTest.
  add_executable(llmp_gemma_dense31_c1_norm gemma_dense31_c1_norm.cc)
  target_link_libraries(llmp_gemma_dense31_c1_norm PRIVATE llmp_engine llmp_warnings)
  target_link_options(llmp_gemma_dense31_c1_norm PRIVATE "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../lib/llmp")
endif()

# Optional manual observer only: NVTX headers come from the recorded external
# Nsight tool, never from a production dependency or a vendored source.
set(LLMP_BENCHMARK_NVTX_INCLUDE_DIR "" CACHE PATH "External NVTX headers for manual profiling")
if(LLMP_CUDA AND LLMP_BENCHMARK_NVTX_INCLUDE_DIR)
  if(NOT EXISTS "${LLMP_BENCHMARK_NVTX_INCLUDE_DIR}/nvtx3/nvToolsExt.h")
    message(FATAL_ERROR "Manual NVTX profiling requires the supplied public headers")
  endif()
  add_executable(llmp_gemma26_prefill_profile EXCLUDE_FROM_ALL gemma26_prefill_profile.cc)
  target_include_directories(llmp_gemma26_prefill_profile SYSTEM PRIVATE
                             "${LLMP_BENCHMARK_NVTX_INCLUDE_DIR}")
  target_link_libraries(llmp_gemma26_prefill_profile PRIVATE
                        llmp_engine llmp_warnings ${CMAKE_DL_LIBS})
  target_link_options(llmp_gemma26_prefill_profile PRIVATE
                     "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../cublas")
endif()

# Diagnostic-only manual targets. The reversible three-unit measurement
# overlay is deliberately absent from canonical production source/objects.
option(LLMP_BENCHMARK_COARSE_PREFILL "Enable isolated coarse prefill diagnostics" OFF)
if(LLMP_CUDA AND LLMP_BENCHMARK_COARSE_PREFILL)
  if(NOT EXISTS "${LLMP_BENCHMARK_NVTX_INCLUDE_DIR}/nvtx3/nvToolsExt.h")
    message(FATAL_ERROR "Coarse diagnostics require the authenticated external NVTX interface")
  endif()
  foreach(_coarse_source IN ITEMS gemma4_runner.cc planned.cc live_state.cc)
    file(READ "${PROJECT_SOURCE_DIR}/src/engine/${_coarse_source}" _coarse_text)
    string(FIND "${_coarse_text}" "llmp.prefill.coarse.patch.v1" _coarse_marker)
    if(_coarse_marker EQUAL -1)
      message(FATAL_ERROR "Apply the authenticated diagnostic overlay only in the measurement tree")
    endif()
  endforeach()
  add_executable(llmp_gemma26_prefill_coarse EXCLUDE_FROM_ALL
                 gemma26_prefill_coarse.cc gemma26_prefill_coarse_scopes.cc)
  target_include_directories(llmp_gemma26_prefill_coarse SYSTEM PRIVATE
                             "${LLMP_BENCHMARK_NVTX_INCLUDE_DIR}")
  target_link_libraries(llmp_gemma26_prefill_coarse PRIVATE llmp_engine llmp_warnings ${CMAKE_DL_LIBS})
  target_link_options(llmp_gemma26_prefill_coarse PRIVATE
                     "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../cublas")
  add_executable(llmp_gemma26_clock_probe EXCLUDE_FROM_ALL
                 gemma26_prefill_clock_probe.cc gemma26_prefill_coarse_scopes.cc)
  target_include_directories(llmp_gemma26_clock_probe SYSTEM PRIVATE
                             "${LLMP_BENCHMARK_NVTX_INCLUDE_DIR}")
  target_link_libraries(llmp_gemma26_clock_probe PRIVATE llmp_warnings ${CMAKE_DL_LIBS})
endif()

if(LLMP_CUDA)
  add_executable(llmp_gemma31_production gemma31_production.cc)
  target_link_libraries(llmp_gemma31_production PRIVATE llmp_runtime_serving llmp_platform llmp_warnings)
  target_link_options(llmp_gemma31_production PRIVATE "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../lib/llmp")

  add_executable(llmp_gemma26_frontier_control gemma31_production.cc
                   ../docs/experiments/gemma26-production/frontier_control_wrap.cc)
  target_link_libraries(llmp_gemma26_frontier_control PRIVATE
                        llmp_runtime_serving llmp_platform llmp_warnings)
  target_link_options(llmp_gemma26_frontier_control PRIVATE
      "LINKER:--wrap=_ZN4llmp6engine12Gemma4RunnerC1ERNS0_9PagedNodeENS0_13Gemma4OptionsEij"
      "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../lib/llmp")
endif()

if(LLMP_CUDA)
  add_executable(llmp_gemma2_probe gemma2_probe.cc)
  target_link_libraries(llmp_gemma2_probe PRIVATE llmp_engine llmp_platform llmp_warnings)
  target_link_options(llmp_gemma2_probe PRIVATE "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../cublas")

  add_executable(llmp_gemma3_probe gemma3_probe.cc)
  target_link_libraries(llmp_gemma3_probe PRIVATE llmp_engine llmp_platform llmp_warnings)
  target_link_options(llmp_gemma3_probe PRIVATE "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../cublas")
endif()

if(LLMP_CUDA)
  add_executable(llmp_gemma3_c2_probe gemma3_c2_probe.cc)
  target_link_libraries(llmp_gemma3_c2_probe PRIVATE llmp_engine llmp_platform llmp_warnings)
  target_link_options(llmp_gemma3_c2_probe PRIVATE "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../cublas")
  add_executable(llmp_gemma2_c2_probe gemma2_c2_probe.cc)
  target_link_libraries(llmp_gemma2_c2_probe PRIVATE llmp_engine llmp_platform llmp_warnings)
  target_link_options(llmp_gemma2_c2_probe PRIVATE "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../cublas")
endif()

if(LLMP_CUDA)
  add_executable(llmp_gemma3_batch_probe gemma3_batch_probe.cc)
  target_link_libraries(llmp_gemma3_batch_probe PRIVATE llmp_engine llmp_warnings)
  target_link_options(llmp_gemma3_batch_probe PRIVATE "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../cublas")
  add_executable(llmp_gemma3_joint_prefill_probe gemma3_joint_prefill_probe.cc)
  target_link_libraries(llmp_gemma3_joint_prefill_probe PRIVATE llmp_engine llmp_warnings)
  target_link_options(llmp_gemma3_joint_prefill_probe PRIVATE "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../cublas")
  add_executable(llmp_gemma4_common_width_probe gemma4_common_width_probe.cc)
  target_link_libraries(llmp_gemma4_common_width_probe PRIVATE llmp_engine llmp_warnings)
  target_link_options(llmp_gemma4_common_width_probe PRIVATE "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../cublas")
  add_executable(llmp_gemma3_context_probe gemma3_context_probe.cc)
  target_link_libraries(llmp_gemma3_context_probe PRIVATE llmp_engine llmp_warnings)
  target_link_options(llmp_gemma3_context_probe PRIVATE "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../cublas")
endif()

if(LLMP_CUDA)
  add_executable(llmp_gemma2_joint_prefill_probe gemma2_joint_prefill_probe.cc)
  target_link_libraries(llmp_gemma2_joint_prefill_probe PRIVATE llmp_engine llmp_platform llmp_warnings)
  target_link_options(llmp_gemma2_joint_prefill_probe PRIVATE "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../cublas")
endif()

if(LLMP_CUDA)
  add_executable(llmp_gemma3_wide_cycle gemma3_wide_cycle.cc)
  target_link_libraries(llmp_gemma3_wide_cycle PRIVATE llmp_engine llmp_warnings)
  target_link_options(llmp_gemma3_wide_cycle PRIVATE "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../cublas")

  add_executable(llmp_gemma3_wide_probe gemma3_wide_probe.cc)
  target_link_libraries(llmp_gemma3_wide_probe PRIVATE llmp_engine llmp_warnings)
  target_link_options(llmp_gemma3_wide_probe PRIVATE "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../cublas")
endif()

if(LLMP_CUDA)
  add_executable(llmp_dsv4_mask_probe dsv4_mask_probe.cc)
  target_link_libraries(llmp_dsv4_mask_probe PRIVATE llmp_engine llmp_platform llmp_warnings)
  target_link_options(llmp_dsv4_mask_probe PRIVATE "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../cublas")
endif()
