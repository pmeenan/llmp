# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0

# GGML-derived implementations (D-053; docs/backend-proof.md), over the
# locked GGML subset (third_party/sources.lock.json, component `ggml`).
#
#   jitllm_ggml_tensors  GGML tensor descriptors over jitLLM memory
#                        (tensors.h), the implementations' operand checks
#                        (validate.h), upstream's fusion gates (fusion.h),
#                        graph planning and activation placement
#                        (graph_plan.h), the Qwen2 chunk graph
#                        (qwen2_graph.h) and the DeepSeek V4 chunk graph
#                        (dsv4_graph.h). Every profile.
#   jitllm_ggml_kernels  the K-C launch context (launch.h), jitLLM's cuBLAS
#                        handle (cublas.h), the operation implementations
#                        (ops.h, mul_mat_cublas.cu, fattn.cu) and jitLLM's definitions
#                        of the ggml-cuda.cu symbols GGML's launchers use
#                        (ggml_support.cu), and the module's declarations
#                        in the implementation registry
#                        (implementations.h), and the executor of planned
#                        graphs (executor.h). CUDA builds only.
#                        The M3 models' operations add their checks
#                        (validate_ext.h) here and their implementations
#                        (ops_ext.h: ops_ext.cu, mul_mat_q.cu, mmvq_rows.cu,
#                        fattn_mma*.cu)
#                        to jitllm_ggml_kernels.
add_library(jitllm_ggml_tensors STATIC dsv4_ds4_product.cc dsv4_ds4_cache.cc dsv4_ds4_hc.cc dsv4_ds4_comp.cc dsv4_ds4_repack.cc dsv4_graph.cc fusion.cc graph_plan.cc jitllm_ops.cc
                                       qwen2_graph.cc qwen38_graph.cc tensors.cc validate.cc
                                       validate_ext.cc)
target_include_directories(jitllm_ggml_tensors PUBLIC "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(jitllm_ggml_tensors PUBLIC jitllm_base jitllm_ggml_base jitllm_execution
                                                 jitllm_model
                                          # cutlass/version.h only (jitllm_ops.cc pins the
                                          # version the CUTLASS expert layout was written
                                          # against), in every profile
                                          PRIVATE jitllm_cutlass_headers jitllm_warnings)

if(JITLLM_CUDA)
  # cuBLAS, the SDK's pinned shared library, linked dynamically (D-076):
  # binaries record libcublas.so.13 and libcublasLt.so.13, which
  # libcublas.so.13 finds beside itself (GNU ld on the Spark looks for it
  # when linking). The package ships
  # neither until a packaged binary links them; tests find them in
  # build/<preset>/cublas (jitllm_gpu_gtest), hard links to the SDK's.
  set(cublas_dir "${JITLLM_SDK}/cuda/targets/${JITLLM_CUDA_TARGET_DIR}/lib")
  add_library(jitllm_cublas SHARED IMPORTED GLOBAL)
  set_target_properties(jitllm_cublas PROPERTIES IMPORTED_LOCATION "${cublas_dir}/libcublas.so"
                                                 IMPORTED_SONAME libcublas.so.13)
  target_link_options(jitllm_cublas INTERFACE "LINKER:-rpath-link,${cublas_dir}")
  # cuBLASLt too, whose version the handle checks (cublas.h).
  add_library(jitllm_cublaslt SHARED IMPORTED GLOBAL)
  set_target_properties(jitllm_cublaslt PROPERTIES IMPORTED_LOCATION "${cublas_dir}/libcublasLt.so"
                                                   IMPORTED_SONAME libcublasLt.so.13)
  file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/cublas")
  foreach(library IN ITEMS libcublas.so.13 libcublasLt.so.13)
    file(REAL_PATH "${cublas_dir}/${library}" resolved)
    file(CREATE_LINK "${resolved}" "${PROJECT_BINARY_DIR}/cublas/${library}" COPY_ON_ERROR)
  endforeach()

  add_library(jitllm_ggml_kernels STATIC ggml_support.cu launch.cu ops.cu mul_mat_cublas.cu
                                         fattn.cu ops_ext.cu mul_mat_q.cu mul_mat_q2_d2r.cu dsv4_hca_tokentile.cu dsv4_ds4_cache.cu dsv4_ds4_hc.cu dsv4_ds4_comp.cu dsv4_ds4_repack.cu mmvq_rows.cu fattn_mma.cu
                                         fattn_mma_d128.cu fattn_mma_d256.cu fattn_mma_d512.cu
                                         jitllm_ops.cu jitllm_fused.cu jitllm_moe.cu dsv4_fast.cu dsv4_sparse.cu
                                         qsa_sparse.cu
                                         cublas.cc executor.cc implementations.cc)
  target_include_directories(jitllm_ggml_kernels PUBLIC "${PROJECT_SOURCE_DIR}/src")
  target_sources(jitllm_ggml_kernels PRIVATE dsv4_ds4_product.cu dsv4_ds4_product_raw.cu)
  target_link_libraries(jitllm_ggml_kernels PRIVATE jitllm_ds4_original_headers)
  target_include_directories(jitllm_ggml_kernels SYSTEM PRIVATE
                             "${JITLLM_SDK}/cuda/targets/${JITLLM_CUDA_TARGET_DIR}/include")
  # CUTLASS's NVFP4 grouped GEMM for the routed experts (moe_cutlass.h) and
  # its MXFP8 GEMM for Qwen3.8's dense products (mxfp8_cutlass.h): units of
  # their own, device code for sm_121a only (the SM120 block-scaled kernels
  # build for no other target; a plan names them only on that device), with
  # the flags CUTLASS's examples build with.
  add_library(jitllm_ggml_cutlass STATIC moe_cutlass.cu mxfp8_cutlass.cu)
  target_include_directories(jitllm_ggml_cutlass PUBLIC "${PROJECT_SOURCE_DIR}/src")
  target_link_libraries(jitllm_ggml_cutlass PRIVATE jitllm_cutlass_headers)
  set_target_properties(jitllm_ggml_cutlass PROPERTIES CUDA_ARCHITECTURES 121a-real)
  # 20012: __host__ __device__ on a defaulted constructor, which CUTLASS's
  # SM100 tile scheduler declares.
  target_compile_options(jitllm_ggml_cutlass PRIVATE --expt-relaxed-constexpr
                                                     --diag-suppress=20012)
  target_link_libraries(jitllm_ggml_kernels
                        PUBLIC jitllm_ggml_tensors jitllm_ggml_cuda jitllm_device_memory
                               jitllm_cublas jitllm_cublaslt jitllm_execution jitllm_ds4_d2r jitllm_ds4_hca
                        # the NVIDIA driver, for ggml_support.cu's device queries
                        PRIVATE jitllm_cuda_device_memory jitllm_warnings jitllm_ggml_cutlass)
  # sm_121a as GGML's own units build (third_party/patches/ggml/0002), and
  # the profile's discrete GPUs (D-082).
  set(architectures 121a-real ${JITLLM_CUDA_DISCRETE_SASS})
  set_target_properties(jitllm_ggml_kernels PROPERTIES CUDA_ARCHITECTURES "${architectures}")
  # What each implementation's identity records of the build (D-053): the
  # prepared GGML tree the build used, as the source receipt records it
  # (D-057), the SDK, the target, the device architectures, the build type
  # (NDEBUG, and with it GGML's device asserts) and any sanitizers.
  # A missing receipt, entry or digest fails configure: never an identity
  # without the tree.
  # The CUTLASS tree likewise, for the grouped GEMM's identity.
  get_property(sources GLOBAL PROPERTY JITLLM_SOURCES_ENTRIES)
  set(ggml_tree "")
  set(cutlass_tree "")
  set(ds4_tree "")
  string(JSON count ERROR_VARIABLE error LENGTH "${sources}")
  if(NOT error AND count GREATER 0)
    # foreach(RANGE -1) would still run, for 0 and -1.
    math(EXPR last "${count} - 1")
    foreach(i RANGE ${last})
      string(JSON id ERROR_VARIABLE error GET "${sources}" ${i} id)
      if(NOT error AND id STREQUAL "ggml")
        string(JSON ggml_tree ERROR_VARIABLE error GET "${sources}" ${i} source_tree)
      elseif(NOT error AND id STREQUAL "cutlass")
        string(JSON cutlass_tree ERROR_VARIABLE error GET "${sources}" ${i} source_tree)
      elseif(NOT error AND id STREQUAL "ds4")
        string(JSON ds4_tree ERROR_VARIABLE error GET "${sources}" ${i} source_tree)
      endif()
    endforeach()
  endif()
  foreach(tree IN ITEMS ggml_tree cutlass_tree ds4_tree)
    string(LENGTH "${${tree}}" length)
    if(NOT length EQUAL 64 OR NOT ${tree} MATCHES "^[0-9a-f]+$")
      message(FATAL_ERROR "The source receipt has no ${tree} digest for the registry's identities")
    endif()
  endforeach()
  string(REPLACE ";" "+" sanitize "${JITLLM_SANITIZE}")
  list(JOIN architectures "," architecture_text)
  set_source_files_properties(implementations.cc PROPERTIES COMPILE_DEFINITIONS
    "JITLLM_GGML_SOURCE_TREE=\"${ggml_tree}\";JITLLM_CUTLASS_SOURCE_TREE=\"${cutlass_tree}\";JITLLM_DS4_SOURCE_TREE=\"${ds4_tree}\";JITLLM_GGML_SDK=\"${JITLLM_SDK_IDENTITY}\";JITLLM_GGML_TARGET=\"${JITLLM_TARGET_TRIPLE}\";JITLLM_GGML_CUDA_ARCHITECTURES=\"${architecture_text}\";JITLLM_GGML_BUILD_TYPE=\"${CMAKE_BUILD_TYPE}\";JITLLM_GGML_SANITIZE=\"${sanitize}\"")

  # And jitLLM's own code in the module: every file in this directory, whose
  # digest a build-time command writes into a generated source
  # (module_digest.cmake). Editing any of them regenerates and recompiles
  # only that source. A file added here without being listed fails the next
  # build's configure (CONFIGURE_DEPENDS), so none escapes the identity;
  # editors' hidden, backup and autosave files (.x, x~, #x#, x.orig, x.rej)
  # are not sources and are skipped (as in ggml_module_digest_test.cmake).
  set(module_files CMakeLists.txt cublas.cc cublas.h dsv4_fast.cu dsv4_fast.h dsv4_graph.cc dsv4_sparse.cu
                   dsv4_ds4_product.h dsv4_ds4_product.cc dsv4_ds4_product.cu
                   dsv4_ds4_product_raw.h dsv4_ds4_product_raw.cu
                   dsv4_ds4_product_original.cuh dsv4_ds4_product_quantize.cuh
                   dsv4_ds4_product_f16.cuh dsv4_ds4_product_q8.cuh
                   dsv4_ds4_product_source.json dsv4_ds4_product_source.json.license
                   dsv4_graph.h dsv4_hca_tokentile.cu dsv4_ds4_cache.cc dsv4_ds4_cache.cu
                   dsv4_ds4_cache.h dsv4_ds4_cache_core.cuh dsv4_ds4_cache_core.json
                   dsv4_ds4_cache_core.json.license
                   dsv4_ds4_hc.cc dsv4_ds4_hc.cu dsv4_ds4_hc.h
                   dsv4_ds4_hc_core.cuh dsv4_ds4_hc_core.json dsv4_ds4_hc_core.json.license
                   dsv4_ds4_comp.cc dsv4_ds4_comp.cu dsv4_ds4_comp.h
                   dsv4_ds4_comp_core.cuh dsv4_ds4_comp_core.json dsv4_ds4_comp_core.json.license
                   dsv4_ds4_repack.cc dsv4_ds4_repack.cu dsv4_ds4_repack.h
                   dsv4_ds4_repack_core.cuh dsv4_ds4_repack_core.json dsv4_ds4_repack_core.json.license
                   executor.cc
                   executor.h fattn.cu fattn_mma.cu
                   fattn_mma.h fattn_mma_d128.cu fattn_mma_d256.cu fattn_mma_d512.cu
                   fattn_mma_shape.cuh fusion.cc
                   fusion.h ggml_support.cu ggml_support.h graph_plan.cc graph_plan.h implementations.cc
                   implementations.h jitllm_fused.cu jitllm_moe.cu jitllm_ops.cc jitllm_ops.cu
                   jitllm_ops.h launch.cu launch.h
                   mmvq_rows.cu module_digest.cmake moe_cutlass.cu moe_cutlass.h moe_layout.h
                   mxfp8_cutlass.cu mxfp8_cutlass.h mxfp8_quant.cuh
                   mul_mat_cublas.cu mul_mat_q.cu mul_mat_q2_d2r.cu ops.cu ops.h ops_ext.cu qsa_sparse.cu
                   ops_ext.h qwen2_graph.cc qwen2_graph.h qwen38_commit.h qwen38_graph.cc
                   qwen38_graph.h tensors.cc tensors.h validate.cc validate.h validate_ext.cc validate_ext.h
                   validate_util.h)
  file(GLOB present CONFIGURE_DEPENDS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
       "${CMAKE_CURRENT_SOURCE_DIR}/[!.]*")
  foreach(name IN LISTS present)
    if(NOT name IN_LIST module_files AND NOT name MATCHES "(^#|~$|\\.orig$|\\.rej$)")
      message(FATAL_ERROR "src/kernels/ggml/${name} is not in module_files, so the "
                          "implementations' identity would not cover it")
    endif()
  endforeach()
  string(JOIN ":" module_names ${module_files})
  set_property(GLOBAL PROPERTY JITLLM_GGML_MODULE_NAMES "${module_names}")
  list(TRANSFORM module_files PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/" OUTPUT_VARIABLE module_paths)
  set(module_digest "${CMAKE_CURRENT_BINARY_DIR}/module_digest.cc")
  add_custom_command(
    OUTPUT "${module_digest}"
    COMMAND "${CMAKE_COMMAND}" "-DBASE=${CMAKE_CURRENT_SOURCE_DIR}" "-DNAMES=${module_names}"
            "-DOUTPUT=${module_digest}" -P "${CMAKE_CURRENT_SOURCE_DIR}/module_digest.cmake"
    DEPENDS ${module_paths}
    COMMENT "Hashing the GGML module's sources for its implementation identities (D-053)"
    VERBATIM)
  target_sources(jitllm_ggml_kernels PRIVATE "${module_digest}")
  # The copy of upstream's cuBLAS path builds with the FP16 bridge's device
  # flag, as GGML's own files do (third_party/patches/ggml/0002), so that its
  # kernel's SASS compares with the bridge's. (NVCC's -O sets only the host
  # optimization level; device code is always -O3.)
  set_source_files_properties(mul_mat_cublas.cu PROPERTIES COMPILE_OPTIONS "-use_fast_math")
  # The vector flash-attention instance (fattn.cu) likewise, with the rest
  # of GGML's flags for its files, as GGML's own instance unit
  # (template-instances/fattn-vec-instance-f16-f16.cu) builds in the bridge.
  set_source_files_properties(fattn.cu PROPERTIES COMPILE_OPTIONS
                              "-use_fast_math;-extended-lambda")
  # The default-off complete-ds4 benchmark's cache/QAT and HC/norm cores
  # retain the pinned original CUDA flags. The caller owns every range,
  # including the decode table; this adds no original runtime or globals.
  set_source_files_properties(dsv4_ds4_cache.cu dsv4_ds4_hc.cu dsv4_ds4_comp.cu dsv4_ds4_repack.cu PROPERTIES COMPILE_OPTIONS
                              "-O3;--use_fast_math;-lineinfo")
  set_source_files_properties(dsv4_ds4_product.cu dsv4_ds4_product_raw.cu PROPERTIES COMPILE_OPTIONS
                              "-O3;--use_fast_math;-lineinfo;-extended-lambda")
  # The M3 units that compile or launch GGML's kernels take GGML's device
  # flags too, as GGML's own files and instance units build
  # (third_party/patches/ggml/0002): the tensor-core flash-attention
  # instances and the sparse gather, the quantized products' host code, and
  # the other launchers' wrappers. jitLLM's fusions of GGML nodes
  # (jitllm_fused.cu) build with the same flags, so that their arithmetic
  # compiles as the nodes' does.
  set_source_files_properties(fattn_mma.cu fattn_mma_d128.cu fattn_mma_d256.cu fattn_mma_d512.cu
                              mul_mat_q.cu mul_mat_q2_d2r.cu mmvq_rows.cu ops_ext.cu jitllm_fused.cu jitllm_moe.cu
                              dsv4_fast.cu dsv4_sparse.cu PROPERTIES COMPILE_OPTIONS
                              "-use_fast_math;-extended-lambda")
endif()
