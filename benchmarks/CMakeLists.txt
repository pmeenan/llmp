# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0

# Measurement harnesses: built with the rest, never run by CTest. Each
# prints aggregate tables that a report under docs/experiments/ records.

add_library(jitllm_draft_vocab_contract STATIC qwen38_vocab.cc)
target_include_directories(jitllm_draft_vocab_contract PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(jitllm_draft_vocab_contract PRIVATE jitllm_json jitllm_chat
                                                       jitllm_model jitllm_tokenizer jitllm_warnings)

add_library(jitllm_vocab_reference STATIC qwen38_reference.cc)
target_include_directories(jitllm_vocab_reference PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(jitllm_vocab_reference PRIVATE jitllm_base jitllm_platform jitllm_warnings)

# D-048's lanes: wakeup latency against polling, lane throughput, and the
# completion board's round trip (docs/experiments/task-lanes/).
add_executable(jitllm_lanes_bench lanes_bench.cc)
target_link_libraries(jitllm_lanes_bench PRIVATE jitllm_scheduler jitllm_warnings)

# Host sampling's cost a draw at the served vocabularies (execution/sampling.h;
# docs/tokenizer.md#sampling).
add_executable(jitllm_sampling_bench sampling_bench.cc)
target_link_libraries(jitllm_sampling_bench PRIVATE jitllm_sampling jitllm_warnings)

# The retained-backing comparison's deterministic replay of the swap trace
# on the fake provider (docs/experiments/retained-backing/).
add_subdirectory(retained_backing)

# CPU-only frozen task construction/scoring; never loads a device or model.
add_library(jitllm_long_context_tasks STATIC long_context_tasks.cc)
target_include_directories(jitllm_long_context_tasks PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(jitllm_long_context_tasks PUBLIC jitllm_json PRIVATE jitllm_warnings)
add_executable(jitllm_long_context_task_prepare long_context_tasks_main.cc)
target_link_libraries(jitllm_long_context_task_prepare PRIVATE jitllm_long_context_tasks
                      jitllm_artifact jitllm_tokenizer jitllm_chat jitllm_base jitllm_warnings)

# Storage queue depth and request size, direct into host VMM through the
# providers (docs/experiments/storage-queue/). Needs a CUDA device.
if(JITLLM_CUDA)
  add_executable(jitllm_turn_reuse turn_reuse.cc)
  target_link_libraries(jitllm_turn_reuse PRIVATE jitllm_runtime_serving jitllm_base jitllm_warnings)
  target_link_options(jitllm_turn_reuse PRIVATE
                     "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../lib/jitllm")

  # Exact long-context swap controls without a second whole-state allocation.
  add_executable(jitllm_long_swap long_swap.cc)
  target_link_libraries(jitllm_long_swap PRIVATE jitllm_runtime_serving jitllm_base jitllm_warnings)
  target_link_options(jitllm_long_swap PRIVATE
                     "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../lib/jitllm")

  add_executable(jitllm_storage_bench storage_bench.cc)
  target_link_libraries(jitllm_storage_bench PRIVATE jitllm_cuda_device_memory jitllm_storage jitllm_warnings)

  # Page-in throughput through the scheduler and its lanes: the landing
  # zone into device VMM against direct reads in place into host VMM
  # (D-081; docs/experiments/pagein-perf/).
  add_executable(jitllm_pagein_bench pagein_bench.cc)
  target_include_directories(jitllm_pagein_bench SYSTEM PRIVATE
                             "${JITLLM_SDK}/cuda/targets/${JITLLM_CUDA_TARGET_DIR}/include")
  target_link_libraries(jitllm_pagein_bench PRIVATE jitllm_cuda_device_memory
                                                    jitllm_scheduler jitllm_storage jitllm_warnings)

  # Which OS counters include VMM backing (docs/experiments/vmm-counters/).
  add_executable(jitllm_vmm_counters vmm_counters.cc)
  target_include_directories(jitllm_vmm_counters SYSTEM PRIVATE
                             "${JITLLM_SDK}/cuda/targets/${JITLLM_CUDA_TARGET_DIR}/include")
  target_link_libraries(jitllm_vmm_counters PRIVATE jitllm_cuda_device_memory jitllm_warnings)

  # Host time per launch of RMSNorm-mul, for each implementation a plan can
  # select, through the launch context and around it (BP-F4 in backend-proof
  # P1; docs/experiments/launch-overhead/). Needs a GB10.
  add_executable(jitllm_launch_bench launch_bench.cc)
  target_include_directories(jitllm_launch_bench SYSTEM PRIVATE
                             "${JITLLM_SDK}/cuda/targets/${JITLLM_CUDA_TARGET_DIR}/include")
  target_link_libraries(jitllm_launch_bench PRIVATE jitllm_ggml_kernels jitllm_cuda_device_memory
                                                    jitllm_warnings)
  # It links cuBLAS dynamically (D-076), found as the GPU tests find it.
  file(RELATIVE_PATH runtime "${CMAKE_CURRENT_BINARY_DIR}" "${PROJECT_BINARY_DIR}/cublas")
  target_link_options(jitllm_launch_bench PRIVATE "LINKER:--enable-new-dtags,-rpath,$ORIGIN/${runtime}")

  # Family-neutral sparse-attention and paired expert preparation controls.
  add_executable(jitllm_prefill_transfer_bench prefill_transfer_bench.cc)
  target_include_directories(jitllm_prefill_transfer_bench SYSTEM PRIVATE
                             "${JITLLM_SDK}/cuda/targets/${JITLLM_CUDA_TARGET_DIR}/include")
  target_link_libraries(jitllm_prefill_transfer_bench PRIVATE jitllm_ggml_kernels
                                                           jitllm_warnings)
  target_link_options(jitllm_prefill_transfer_bench PRIVATE
                      "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../cublas")

  # BP-F1: GGML kernels with their operands in cudaMalloc, host VMM or device VMM
  # memory, one timing block per process (docs/experiments/backend-proof-p1/).
  # Finds the SDK's cuBLAS in build/<preset>/cublas, as the GPU tests do.
  add_executable(jitllm_ggml_vmm_bench ggml_vmm_bench.cc)
  target_include_directories(jitllm_ggml_vmm_bench SYSTEM PRIVATE
                             "${JITLLM_SDK}/cuda/targets/${JITLLM_CUDA_TARGET_DIR}/include")
  target_link_libraries(jitllm_ggml_vmm_bench PRIVATE jitllm_ggml_kernels jitllm_cuda_device_memory
                                                      jitllm_warnings)
  target_link_options(jitllm_ggml_vmm_bench PRIVATE
                      "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../cublas")

  # Backend-proof P2, rung 3: the FP16 fixture from its v0 artifact on
  # cudaMalloc memory, recorded for the FP16 gate's plan comparison or read
  # for the allocation census (docs/experiments/backend-proof-p2/). Links the
  # launch recorder; finds the SDK's cuBLAS in build/<preset>/cublas.
  add_executable(jitllm_fp16_exec fp16_exec.cc)
  target_include_directories(jitllm_fp16_exec SYSTEM PRIVATE
                             "${JITLLM_SDK}/cuda/targets/${JITLLM_CUDA_TARGET_DIR}/include")
  target_link_libraries(jitllm_fp16_exec PRIVATE jitllm_ggml_kernels jitllm_cuda_device_memory
                                                 jitllm_artifact jitllm_model jitllm_launch_recorder
                                                 jitllm_warnings)
  target_link_options(jitllm_fp16_exec PRIVATE "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../cublas")

  # jitllm.vecq's launch configurations on DeepSeek V4 Flash's product shapes
  # (kernels/ggml/dsv4_fast.h; docs/experiments/dsv4-decode/).
  add_executable(jitllm_vecq_bench vecq_bench.cc)
  target_include_directories(jitllm_vecq_bench SYSTEM PRIVATE
                             "${JITLLM_SDK}/cuda/targets/${JITLLM_CUDA_TARGET_DIR}/include")
  target_link_libraries(jitllm_vecq_bench PRIVATE jitllm_ggml_kernels jitllm_cuda_device_memory
                                                  jitllm_warnings)
  target_link_options(jitllm_vecq_bench PRIVATE "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../cublas")

  add_executable(jitllm_gemma_quant_bench gemma_quant.cc)
  target_include_directories(jitllm_gemma_quant_bench SYSTEM PRIVATE
                             "${JITLLM_SDK}/cuda/targets/${JITLLM_CUDA_TARGET_DIR}/include")
  target_link_libraries(jitllm_gemma_quant_bench PRIVATE jitllm_ggml_kernels jitllm_cuda_device_memory
                                                        jitllm_warnings)
  target_link_options(jitllm_gemma_quant_bench PRIVATE "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../cublas")

  add_executable(jitllm_gemma_geglu_quant_bench gemma_geglu_quant.cc)
  target_include_directories(jitllm_gemma_geglu_quant_bench SYSTEM PRIVATE
                             "${JITLLM_SDK}/cuda/targets/${JITLLM_CUDA_TARGET_DIR}/include")
  target_link_libraries(jitllm_gemma_geglu_quant_bench PRIVATE jitllm_ggml_kernels
                          jitllm_cuda_device_memory jitllm_warnings)
  target_link_options(jitllm_gemma_geglu_quant_bench PRIVATE
                      "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../cublas")

  add_executable(jitllm_gemma_attention_bench gemma_attention.cc)
  target_include_directories(jitllm_gemma_attention_bench SYSTEM PRIVATE
                             "${JITLLM_SDK}/cuda/targets/${JITLLM_CUDA_TARGET_DIR}/include")
  target_link_libraries(jitllm_gemma_attention_bench PRIVATE jitllm_ggml_kernels jitllm_cuda_device_memory
                                                        jitllm_warnings)
  target_link_options(jitllm_gemma_attention_bench PRIVATE "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../cublas")

  # M3's first model slice: DeepSeek V4 Flash, native and resident, from its
  # v0 prepared artifact (docs/experiments/dsv4-native/).
  add_executable(jitllm_dsv4_exec dsv4_exec.cc)
  target_include_directories(jitllm_dsv4_exec SYSTEM PRIVATE
                             "${JITLLM_SDK}/cuda/targets/${JITLLM_CUDA_TARGET_DIR}/include")
  target_link_libraries(jitllm_dsv4_exec PRIVATE jitllm_ggml_kernels jitllm_cuda_device_memory
                                                 jitllm_artifact jitllm_model jitllm_warnings
                                                 jitllm_long_context_tasks)
  target_link_options(jitllm_dsv4_exec PRIVATE "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../cublas")

  # M3's second model slice: Qwen3.8 Flash Next (Mia's NVFP4 and MXFP8
  # checkpoint), native and resident (docs/experiments/qwen38-native/),
  # planning through the path the paged runner shares (engine/qwen38_plan.h).
  add_executable(jitllm_qwen38_exec qwen38_exec.cc)
  target_include_directories(jitllm_qwen38_exec SYSTEM PRIVATE
                             "${JITLLM_SDK}/cuda/targets/${JITLLM_CUDA_TARGET_DIR}/include")
  target_link_libraries(jitllm_qwen38_exec PRIVATE jitllm_engine jitllm_ggml_kernels
                                                   jitllm_cuda_device_memory jitllm_artifact
                                                   jitllm_model jitllm_warnings)
  target_link_options(jitllm_qwen38_exec PRIVATE "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../cublas")

  # M3's third model slice: the Qwen-Image-2.1 BF16 pipeline, native, from
  # its component artifacts and their composition
  # (docs/experiments/qwen-image-native/), with the native tokenizer and the
  # prompt renderer.
  add_executable(jitllm_qwen_image_exec qwen_image_exec.cc)
  target_include_directories(jitllm_qwen_image_exec SYSTEM PRIVATE
                             "${JITLLM_SDK}/cuda/targets/${JITLLM_CUDA_TARGET_DIR}/include")
  target_link_libraries(jitllm_qwen_image_exec PRIVATE jitllm_image_kernels jitllm_ggml_kernels
                                                       jitllm_cuda_device_memory jitllm_artifact
                                                       jitllm_model jitllm_tokenizer jitllm_chat
                                                       jitllm_warnings)
  target_link_options(jitllm_qwen_image_exec PRIVATE
                      "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../cublas")
  # Its products' cuBLASLt candidates, timed, for the algorithms the image
  # kernels pin (kernels/image/gemm.h).
  add_executable(jitllm_qwen_image_gemm_tune qwen_image_gemm_tune.cc)
  target_include_directories(jitllm_qwen_image_gemm_tune SYSTEM PRIVATE
                             "${JITLLM_SDK}/cuda/targets/${JITLLM_CUDA_TARGET_DIR}/include")
  target_link_libraries(jitllm_qwen_image_gemm_tune PRIVATE jitllm_image_kernels jitllm_warnings)
  target_link_options(jitllm_qwen_image_gemm_tune PRIVATE
                      "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../cublas")

  # Qwen3.8's small-column BF16 products, with their actual output types.
  add_executable(jitllm_qwen38_gemm_tune qwen38_gemm_tune.cc)
  target_include_directories(jitllm_qwen38_gemm_tune SYSTEM PRIVATE
                             "${JITLLM_SDK}/cuda/targets/${JITLLM_CUDA_TARGET_DIR}/include")
  target_link_libraries(jitllm_qwen38_gemm_tune PRIVATE jitllm_ggml_kernels jitllm_warnings)
  target_link_options(jitllm_qwen38_gemm_tune PRIVATE
                      "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../cublas")

  # GB10-only scheduling sweep, run only by explicit benchmark invocation.
  add_executable(jitllm_qwen38_mxfp8_tune qwen38_mxfp8_tune.cc qwen38_mxfp8_tune_kernels.cu)
  target_include_directories(jitllm_qwen38_mxfp8_tune SYSTEM PRIVATE
                             "${JITLLM_SDK}/cuda/targets/${JITLLM_CUDA_TARGET_DIR}/include")
  target_link_libraries(jitllm_qwen38_mxfp8_tune PRIVATE jitllm_ggml_kernels jitllm_warnings)
  set_target_properties(jitllm_qwen38_mxfp8_tune PROPERTIES CUDA_ARCHITECTURES 121a-real)
  set_source_files_properties(qwen38_mxfp8_tune_kernels.cu PROPERTIES COMPILE_OPTIONS "-Xptxas=-v")
  target_link_options(jitllm_qwen38_mxfp8_tune PRIVATE
                      "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../cublas")

  # The paged harnesses' model runners: the FP16 fixture and an EXL3
  # fixture, each a model on the engine's paged node
  # (tests/support/paged_node.h), so one process can hold either or both
  # (BP-S3), and the M3 harnesses' helpers (dsv4_common.h) over the engine's
  # M3 runners (src/engine/, under their harness names in engine_names.h).
  # Their binaries link the launch recorder themselves (an object library,
  # whose wrappers must be in the link).
  add_library(jitllm_paged_runners STATIC fp16_runner.cc fp16_common.cc exl3_runner.cc
                                          exl3_common.cc dsv4_common.cc)
  target_include_directories(jitllm_paged_runners SYSTEM PRIVATE
                             "${JITLLM_SDK}/cuda/targets/${JITLLM_CUDA_TARGET_DIR}/include")
  target_include_directories(jitllm_paged_runners PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
  target_link_libraries(jitllm_paged_runners PUBLIC jitllm_paged_node jitllm_engine
                                                    jitllm_exl3_qwen2 jitllm_ggml_kernels
                                                    jitllm_scheduler jitllm_cuda_device_memory
                                                    jitllm_artifact jitllm_model
                                                    jitllm_plan_record jitllm_storage
                                             PRIVATE jitllm_warnings)

  # Backend-proof P2, rungs 4 and 5: the same fixture paged into device VMM
  # through the landing zone by the scheduler and its lanes, executed as
  # device jobs under leases, evicted and restored
  # (docs/experiments/backend-proof-p2/). Links the launch recorder.
  add_executable(jitllm_fp16_paged fp16_paged.cc)
  target_link_libraries(jitllm_fp16_paged PRIVATE jitllm_paged_runners jitllm_launch_recorder
                                                  jitllm_warnings)
  target_link_options(jitllm_fp16_paged PRIVATE "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../cublas")

  # BP-S3: the FP16 fixture and an EXL3 fixture alternating on one paged
  # node, under a budget that forces each to evict the other's weights
  # (docs/experiments/backend-proof/). Links the launch recorder.
  add_executable(jitllm_alternate_paged alternate_paged.cc)
  target_include_directories(jitllm_alternate_paged SYSTEM PRIVATE
                             "${JITLLM_SDK}/cuda/targets/${JITLLM_CUDA_TARGET_DIR}/include")
  target_link_libraries(jitllm_alternate_paged PRIVATE jitllm_paged_runners jitllm_launch_recorder
                                                       jitllm_warnings)
  target_link_options(jitllm_alternate_paged PRIVATE
                      "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../cublas")

  # M3's swap runner: DeepSeek V4 Flash and the FP16 fixture on one paged
  # node, full swaps A→B→A with each part timed, state spilled and restored
  # (docs/experiments/fast-swap/swap.md). A harness: it runs on the test
  # harness's paged node and links the launch recorder.
  add_executable(jitllm_swap_runner swap_runner.cc)
  target_link_libraries(jitllm_swap_runner PRIVATE jitllm_paged_runners jitllm_launch_recorder
                                                   jitllm_tokenizer jitllm_warnings)
  target_link_options(jitllm_swap_runner PRIVATE "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../cublas")

  # M3's swap pairs: two of DeepSeek V4 Flash, Qwen3.8 Flash Next and the
  # Qwen-Image-2.1 pipeline on one paged node, full swaps A→B→A with each
  # part timed (docs/experiments/fast-swap/swap.md), over the engine's
  # runners. The harness beside jitllm-runtime's swap-table, which runs the
  # same protocol over the configured models in one process.
  add_executable(jitllm_swap_pairs swap_pairs.cc)
  target_include_directories(jitllm_swap_pairs SYSTEM PRIVATE
                             "${JITLLM_SDK}/cuda/targets/${JITLLM_CUDA_TARGET_DIR}/include")
  target_link_libraries(jitllm_swap_pairs PRIVATE jitllm_paged_runners jitllm_launch_recorder
                                                  jitllm_image_kernels jitllm_tokenizer
                                                  jitllm_chat jitllm_warnings)
  target_link_options(jitllm_swap_pairs PRIVATE "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../cublas")

  # M3's speculation harness: DeepSeek V4 Flash with its DSpark drafter on
  # the paged node, greedy and sampled speculation, the forced-rejection
  # checks and speed (docs/experiments/dspark/). A harness for now; it
  # links the native tokenizer and chat renderer.
  add_executable(jitllm_gemma_prefill gemma_prefill.cc)
  target_link_libraries(jitllm_gemma_prefill PRIVATE jitllm_engine jitllm_warnings)
  target_link_options(jitllm_gemma_prefill PRIVATE "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../cublas")

  add_executable(jitllm_gemma_quality gemma_quality.cc)
  target_link_libraries(jitllm_gemma_quality PRIVATE jitllm_engine jitllm_warnings)
  target_link_options(jitllm_gemma_quality PRIVATE "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../cublas")

  add_executable(jitllm_gemma_runner gemma_runner.cc)
  # Manual native-origin first-local operand capture. Link wrapping adds no
  # graph keep, planner policy or production observer.
  add_executable(jitllm_gemma_attention_capture gemma_attention_capture.cc
                   gemma_attention_capture_wrap.cc)
  target_include_directories(jitllm_gemma_attention_capture SYSTEM PRIVATE
                             "${JITLLM_SDK}/cuda/targets/${JITLLM_CUDA_TARGET_DIR}/include")
  target_link_libraries(jitllm_gemma_attention_capture PRIVATE jitllm_engine jitllm_warnings)
  target_link_options(jitllm_gemma_attention_capture PRIVATE
      "LINKER:--wrap=_ZN6jitllm7kernels4ggml15FlashAttnVec256ERNS1_13LaunchContextEP11ggml_tensor"
      "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../cublas")

  add_executable(jitllm_gemma_attention_replay gemma_attention_replay.cc)
  target_link_libraries(jitllm_gemma_attention_replay PRIVATE jitllm_engine jitllm_warnings)
  target_link_options(jitllm_gemma_attention_replay PRIVATE
                      "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../cublas")

  add_executable(jitllm_gemma_attention_packed
    gemma_attention_packed.cc gemma_attention_packed_wrap.cc)
  target_link_libraries(jitllm_gemma_attention_packed PRIVATE jitllm_engine jitllm_warnings)
  target_link_options(jitllm_gemma_attention_packed PRIVATE
    "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../cublas"
    "LINKER:--wrap=_ZN6jitllm7kernels4ggml16BuildGemma4GraphERNS1_11TensorArenaERKNS_5model13Gemma4ProfileERKNS4_13Gemma4BindingERKNS4_17Gemma4StateLayoutERKNS1_16Gemma4ChunkShapeERKNS1_18Gemma4GraphOptionsE"
    "LINKER:--wrap=_ZN6jitllm7kernels4ggml18Gemma4GraphTensorsERKNS_5model13Gemma4ProfileEm")
  add_executable(jitllm_gemma_attention_packed_metadata
    gemma_attention_packed_metadata.cc gemma_attention_packed_wrap.cc)
  target_include_directories(jitllm_gemma_attention_packed_metadata PRIVATE
    "${PROJECT_SOURCE_DIR}/tests/support")
  target_link_libraries(jitllm_gemma_attention_packed_metadata PRIVATE jitllm_engine jitllm_warnings)
  target_link_options(jitllm_gemma_attention_packed_metadata PRIVATE
    "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../cublas"
    "LINKER:--wrap=_ZN6jitllm7kernels4ggml16BuildGemma4GraphERNS1_11TensorArenaERKNS_5model13Gemma4ProfileERKNS4_13Gemma4BindingERKNS4_17Gemma4StateLayoutERKNS1_16Gemma4ChunkShapeERKNS1_18Gemma4GraphOptionsE"
    "LINKER:--wrap=_ZN6jitllm7kernels4ggml18Gemma4GraphTensorsERKNS_5model13Gemma4ProfileEm")

if(JITLLM_CUDA)
  # Manual scalar layer0 FFN acquisition/replay; no production dispatch change.
  add_executable(jitllm_gemma_dense_ffn_capture gemma_dense_ffn_capture.cc
                 gemma_dense_ffn_capture_wrap.cc)
  target_include_directories(jitllm_gemma_dense_ffn_capture SYSTEM PRIVATE
                             "${JITLLM_SDK}/cuda/targets/${JITLLM_CUDA_TARGET_DIR}/include")
  target_link_libraries(jitllm_gemma_dense_ffn_capture PRIVATE jitllm_engine jitllm_warnings)
  target_link_options(jitllm_gemma_dense_ffn_capture PRIVATE
      "LINKER:--wrap=_ZN6jitllm7kernels4ggml10MulMatVecQERNS1_13LaunchContextEP11ggml_tensor"
      "LINKER:--wrap=_ZNK6jitllm6engine12PagedWeights16resource_addressEj"
      "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../cublas")
  add_executable(jitllm_gemma_dense_ffn_replay gemma_dense_ffn_replay.cc)
  target_link_libraries(jitllm_gemma_dense_ffn_replay PRIVATE jitllm_engine jitllm_warnings)
  target_link_options(jitllm_gemma_dense_ffn_replay PRIVATE
                      "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../cublas")
endif()
  add_executable(jitllm_gemma26_attention_packed
    gemma26_attention_packed.cc gemma26_attention_packed_wrap.cc)
  target_link_libraries(jitllm_gemma26_attention_packed PRIVATE jitllm_engine jitllm_warnings)
  target_link_options(jitllm_gemma26_attention_packed PRIVATE
    "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../cublas"
    "LINKER:--wrap=_ZN6jitllm7kernels4ggml16BuildGemma4GraphERNS1_11TensorArenaERKNS_5model13Gemma4ProfileERKNS4_13Gemma4BindingERKNS4_17Gemma4StateLayoutERKNS1_16Gemma4ChunkShapeERKNS1_18Gemma4GraphOptionsE"
    "LINKER:--wrap=_ZN6jitllm7kernels4ggml18Gemma4GraphTensorsERKNS_5model13Gemma4ProfileEm")
  add_executable(jitllm_gemma26_attention_compound
    gemma26_attention_compound.cc gemma26_attention_packed_wrap.cc)
  target_link_libraries(jitllm_gemma26_attention_compound PRIVATE jitllm_engine jitllm_warnings)
  target_link_options(jitllm_gemma26_attention_compound PRIVATE
    "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../cublas"
    "LINKER:--wrap=_ZN6jitllm7kernels4ggml16BuildGemma4GraphERNS1_11TensorArenaERKNS_5model13Gemma4ProfileERKNS4_13Gemma4BindingERKNS4_17Gemma4StateLayoutERKNS1_16Gemma4ChunkShapeERKNS1_18Gemma4GraphOptionsE"
    "LINKER:--wrap=_ZN6jitllm7kernels4ggml18Gemma4GraphTensorsERKNS_5model13Gemma4ProfileEm")
  # Manual common-input late-prefill MoE diagnostic; no CTest/model policy.
  add_executable(jitllm_gemma26_late_moe gemma26_late_moe.cc)
  target_link_libraries(jitllm_gemma26_late_moe PRIVATE jitllm_engine jitllm_warnings)
  target_link_options(jitllm_gemma26_late_moe PRIVATE
                      "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../cublas")
  add_executable(jitllm_gemma26_attention_packed_metadata
    gemma26_attention_packed_metadata.cc gemma26_attention_packed_wrap.cc)
  target_include_directories(jitllm_gemma26_attention_packed_metadata PRIVATE
    "${PROJECT_SOURCE_DIR}/tests/support")
  target_link_libraries(jitllm_gemma26_attention_packed_metadata PRIVATE jitllm_engine jitllm_warnings)
  target_link_options(jitllm_gemma26_attention_packed_metadata PRIVATE
    "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../cublas"
    "LINKER:--wrap=_ZN6jitllm7kernels4ggml16BuildGemma4GraphERNS1_11TensorArenaERKNS_5model13Gemma4ProfileERKNS4_13Gemma4BindingERKNS4_17Gemma4StateLayoutERKNS1_16Gemma4ChunkShapeERKNS1_18Gemma4GraphOptionsE"
    "LINKER:--wrap=_ZN6jitllm7kernels4ggml18Gemma4GraphTensorsERKNS_5model13Gemma4ProfileEm")
  add_executable(jitllm_gemma_joined gemma_joined.cc)
  target_link_libraries(jitllm_gemma_joined PRIVATE jitllm_engine jitllm_warnings)
  target_link_options(jitllm_gemma_joined PRIVATE
                      "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../cublas")
  target_link_libraries(jitllm_gemma_runner PRIVATE jitllm_engine jitllm_warnings)
  target_link_options(jitllm_gemma_runner PRIVATE "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../cublas")

  add_executable(jitllm_spec_runner spec_runner.cc)
  target_link_libraries(jitllm_spec_runner PRIVATE jitllm_paged_runners jitllm_launch_recorder
                                                   jitllm_tokenizer jitllm_chat jitllm_sampling
                                                   jitllm_warnings)
  target_link_options(jitllm_spec_runner PRIVATE "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../cublas")

  # M3's speculation harness for Qwen3.8 Flash Next with its MTP drafter on
  # the paged node (docs/experiments/qwen38-mtp/). A harness for now; it
  # links the native tokenizer and chat renderer.
  add_executable(jitllm_qwen38_spec qwen38_spec.cc)
  target_link_libraries(jitllm_qwen38_spec PRIVATE jitllm_paged_runners jitllm_launch_recorder
                                                   jitllm_tokenizer jitllm_chat jitllm_sampling
                                                   jitllm_draft_vocab_contract
                                                   jitllm_vocab_reference
                                                   jitllm_runtime
                                                   jitllm_warnings)
  target_link_options(jitllm_qwen38_spec PRIVATE "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../cublas")

  # GGUF target waves: fixed-history logits/state against solo or unjoined steps.
  add_executable(jitllm_qwen38_gguf_wave qwen38_gguf_wave.cc)
  target_link_libraries(jitllm_qwen38_gguf_wave PRIVATE jitllm_paged_runners
                        jitllm_base jitllm_warnings)
  target_link_options(jitllm_qwen38_gguf_wave PRIVATE "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../cublas")

  # Temporary C2 shared-row mechanism proof, no serving/default selection.
  add_executable(jitllm_qwen38_batch qwen38_batch_main.cc qwen38_batch.cc
                 qwen38_batch_graph.cc qwen38_batch_generation.cc)
  target_link_libraries(jitllm_qwen38_batch PRIVATE jitllm_paged_runners
                        jitllm_vocab_reference jitllm_json jitllm_base jitllm_warnings)
  target_link_options(jitllm_qwen38_batch PRIVATE "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../cublas")

  # Backend-proof P3, the per-linear sweep (BP-N5): every real projection of
  # an EXL3 fixture through jitLLM's launchers at upstream's forced plans,
  # hashed for comparison with upstream's outputs
  # (docs/experiments/backend-proof-p3/). Finds the SDK's cuBLAS in
  # build/<preset>/cublas.
  add_executable(jitllm_exl3_linear_sweep exl3_linear_sweep.cc)
  target_link_libraries(jitllm_exl3_linear_sweep PRIVATE jitllm_exl3_kernels jitllm_cuda_device_memory
                                                         jitllm_artifact jitllm_warnings)
  target_link_options(jitllm_exl3_linear_sweep PRIVATE
                      "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../cublas")

  # Backend-proof P3, the native EXL3 model: an EXL3 fixture through the
  # native operation plan, for the plan gate, Tier E (operation level) and
  # Tier C (docs/experiments/backend-proof-p3/). Links the launch recorder.
  add_executable(jitllm_exl3_exec exl3_exec.cc exl3_common.cc)
  target_link_libraries(jitllm_exl3_exec PRIVATE jitllm_exl3_qwen2 jitllm_cuda_device_memory
                                                 jitllm_artifact jitllm_model
                                                 jitllm_launch_recorder jitllm_warnings)
  target_link_options(jitllm_exl3_exec PRIVATE "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../cublas")

  # Backend-proof P3, rungs 4 and 5 for EXL3: the same fixture paged into
  # device VMM through the landing zone by the scheduler and its lanes,
  # executed as device jobs under leases, evicted and restored
  # (docs/experiments/backend-proof-p3/). Links the launch recorder.
  add_executable(jitllm_exl3_paged exl3_paged.cc)
  target_link_libraries(jitllm_exl3_paged PRIVATE jitllm_paged_runners jitllm_launch_recorder
                                                  jitllm_warnings)
  target_link_options(jitllm_exl3_paged PRIVATE "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../cublas")

  # Why host VMM slowed BP-F1's products, and what the alternatives cost:
  # microkernels over each kind of memory, copies between them, and GGML
  # products with each buffer group placed separately
  # (docs/experiments/host-vmm-diagnosis/). Diagnostic, not a gate. Its
  # kernels are the GB10's alone, sm_121a in every profile: the diagnosis
  # is of the GB10's L2, so no discrete target builds them (D-082).
  add_executable(jitllm_vmm_diag_bench vmm_diag_bench.cc vmm_diag_kernels.cu)
  target_include_directories(jitllm_vmm_diag_bench SYSTEM PRIVATE
                             "${JITLLM_SDK}/cuda/targets/${JITLLM_CUDA_TARGET_DIR}/include")
  target_link_libraries(jitllm_vmm_diag_bench PRIVATE jitllm_ggml_kernels jitllm_cuda_device_memory
                                                      jitllm_storage jitllm_warnings)
  set_target_properties(jitllm_vmm_diag_bench PROPERTIES CUDA_ARCHITECTURES 121a-real)
  target_link_options(jitllm_vmm_diag_bench PRIVATE
                      "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../cublas")

  # The runtime wake: from a step's fence to the next step's launch, for
  # each candidate way of waiting, and the runtime's own path on the paged
  # node (docs/experiments/runtime-wake/). Its kernels read the GB10's
  # global timer: sm_121a only, as the host-VMM diagnosis's.
  add_executable(jitllm_wake_bench wake_bench.cc wake_bench_kernels.cu)
  target_include_directories(jitllm_wake_bench SYSTEM PRIVATE
                             "${JITLLM_SDK}/cuda/targets/${JITLLM_CUDA_TARGET_DIR}/include")
  target_link_libraries(jitllm_wake_bench PRIVATE jitllm_paged_node jitllm_warnings)
  set_target_properties(jitllm_wake_bench PROPERTIES CUDA_ARCHITECTURES 121a-real)
  target_link_options(jitllm_wake_bench PRIVATE "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../cublas")
endif()

if(JITLLM_CUDA)
# The existing HTTP route with internal Gemma diagnostic controls only.
add_executable(jitllm_gemma_joined_http gemma_joined_http.cc)
target_link_libraries(jitllm_gemma_joined_http PRIVATE jitllm_runtime_serving jitllm_platform jitllm_warnings)
target_link_options(jitllm_gemma_joined_http PRIVATE "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../lib/jitllm")
endif()

if(JITLLM_CUDA)
  # Original stage-zero immutable inputs; manual diagnostic, never CTest.
  add_executable(jitllm_gemma_assistant_fixture gemma_assistant_fixture.cc)
  target_link_libraries(jitllm_gemma_assistant_fixture PRIVATE jitllm_engine jitllm_json jitllm_warnings)
  target_link_options(jitllm_gemma_assistant_fixture PRIVATE "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../lib/jitllm")
endif()

if(JITLLM_CUDA)
  # Two-owner original stage-zero inputs, manual diagnostic only.
  add_executable(jitllm_gemma_assistant_wave_fixture gemma_assistant_wave_fixture.cc)
  target_link_libraries(jitllm_gemma_assistant_wave_fixture PRIVATE jitllm_engine jitllm_json jitllm_warnings)
  target_link_options(jitllm_gemma_assistant_wave_fixture PRIVATE "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../lib/jitllm")
endif()

if(JITLLM_CUDA)
  # Closed dense31 scalar norm screen; manual only, never CTest.
  add_executable(jitllm_gemma_dense31_c1_norm gemma_dense31_c1_norm.cc)
  target_link_libraries(jitllm_gemma_dense31_c1_norm PRIVATE jitllm_engine jitllm_warnings)
  target_link_options(jitllm_gemma_dense31_c1_norm PRIVATE "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../lib/jitllm")
endif()

# Optional manual observer only: NVTX headers come from the recorded external
# Nsight tool, never from a production dependency or a vendored source.
set(JITLLM_BENCHMARK_NVTX_INCLUDE_DIR "" CACHE PATH "External NVTX headers for manual profiling")
if(JITLLM_CUDA AND JITLLM_BENCHMARK_NVTX_INCLUDE_DIR)
  if(NOT EXISTS "${JITLLM_BENCHMARK_NVTX_INCLUDE_DIR}/nvtx3/nvToolsExt.h")
    message(FATAL_ERROR "Manual NVTX profiling requires the supplied public headers")
  endif()
  add_executable(jitllm_gemma26_prefill_profile EXCLUDE_FROM_ALL gemma26_prefill_profile.cc)
  target_include_directories(jitllm_gemma26_prefill_profile SYSTEM PRIVATE
                             "${JITLLM_BENCHMARK_NVTX_INCLUDE_DIR}")
  target_link_libraries(jitllm_gemma26_prefill_profile PRIVATE
                        jitllm_engine jitllm_warnings ${CMAKE_DL_LIBS})
  target_link_options(jitllm_gemma26_prefill_profile PRIVATE
                     "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../cublas")
endif()
