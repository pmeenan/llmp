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

# The importer's container readers, and the reference decoder of what they
# describe, as a fuzzing entry (import/; D-009): a
# driver over the files it is given, or, built with -DLLMP_LIBFUZZER and
# -fsanitize=fuzzer, a libFuzzer target.
add_executable(llmp_import_fuzz import_fuzz.cc)
target_link_libraries(llmp_import_fuzz PRIVATE llmp_import llmp_reference_kernels llmp_warnings)

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

  # The decode module's templates (kernels/decode, D-108) against the tuned
  # kernels they would someday replace, one Qwen3.8 shape each; the tuned
  # kernels run on a GB10 only.
  add_executable(llmp_decode_templates decode_templates.cc)
  target_include_directories(llmp_decode_templates SYSTEM PRIVATE
                             "${LLMP_SDK}/cuda/targets/${LLMP_CUDA_TARGET_DIR}/include")
  target_link_libraries(llmp_decode_templates PRIVATE llmp_ggml_kernels llmp_decode_kernels
                                                      llmp_warnings)
  target_link_options(llmp_decode_templates PRIVATE
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

  # The FP16 swap fixture. Its binaries link the launch recorder themselves
  # (an object library, whose wrappers must be in the link).
  add_library(llmp_paged_runners STATIC fp16_runner.cc fp16_common.cc)
  target_include_directories(llmp_paged_runners SYSTEM PRIVATE
                             "${LLMP_SDK}/cuda/targets/${LLMP_CUDA_TARGET_DIR}/include")
  target_include_directories(llmp_paged_runners PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
  target_link_libraries(llmp_paged_runners PUBLIC llmp_paged_node llmp_engine
                                                    llmp_ggml_kernels
                                                    llmp_scheduler llmp_cuda_device_memory
                                                    llmp_artifact llmp_model
                                                    llmp_plan_record llmp_storage
                                             PRIVATE llmp_warnings)

  # Kernel-level replays of attention and dense-FFN operands captured from
  # the retired M3 Gemma 4 runner (docs/experiments/gemma-attention-c4,
  # gemma-dense-ffn); they name no runner.
  add_executable(llmp_gemma_attention_replay gemma_attention_replay.cc)
  target_link_libraries(llmp_gemma_attention_replay PRIVATE llmp_engine llmp_warnings)
  target_link_options(llmp_gemma_attention_replay PRIVATE
                      "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../cublas")
  add_executable(llmp_gemma_attention_global_owner_replay gemma_attention_global_owner_replay.cc)
  target_link_libraries(llmp_gemma_attention_global_owner_replay PRIVATE llmp_engine llmp_warnings)
  target_link_options(llmp_gemma_attention_global_owner_replay PRIVATE "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../cublas")
  # Explicit owner-root MMA operand proof only; no production selector.
  add_executable(llmp_gemma_attention_owner_replay gemma_attention_owner_replay.cc)
  target_link_libraries(llmp_gemma_attention_owner_replay PRIVATE llmp_engine llmp_warnings)
  target_link_options(llmp_gemma_attention_owner_replay PRIVATE
                      "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../cublas")
  add_executable(llmp_gemma_dense_ffn_replay gemma_dense_ffn_replay.cc)
  target_link_libraries(llmp_gemma_dense_ffn_replay PRIVATE llmp_engine llmp_warnings)
  target_link_options(llmp_gemma_dense_ffn_replay PRIVATE
                      "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../cublas")

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
  # The same screen on the shared engine (M3.6's assistant gate); manual, never CTest.
  add_executable(llmp_gemma_engine_greedy gemma_engine_greedy.cc)
  target_link_libraries(llmp_gemma_engine_greedy PRIVATE llmp_runtime_serving llmp_warnings)
  target_link_options(llmp_gemma_engine_greedy PRIVATE "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../lib/llmp")
endif()
