# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0

# ExLlamaV3-derived implementations (D-052, D-053; docs/backend-proof.md),
# over the locked ExLlamaV3 subset (third_party/sources.lock.json, component
# `exllamav3`, MIT).
#
#   llmp_exl3          the host checks of every launch (validate.h),
#                        upstream's GEMV choice (upstream_gemv.h), the
#                        launch contract (launch_contract.h) and the locked
#                        headers, for the host constants that size the
#                        kernels' device context (quant/exl3_devctx.cuh).
#                        Every profile.
#   llmp_exl3_kernels  llmpalooza's launchers (launch.h), which replace
#                        upstream's ATen host wrappers and keep
#                        launch_contract.h; the reconstruction GEMM on
#                        cuBLASLt (recon_gemm.h); the linear's paths
#                        (linear.h); and the module's declarations in the
#                        implementation registry (implementations.h). The
#                        kernels themselves are the component's
#                        (llmp_exl3_cuda: upstream's GEMM units and
#                        Llmpalooza's instances of the rest,
#                        llmp_exl3_kernels.h), built as the P0 reference
#                        was. CUDA builds only.
#   llmp_exl3_qwen2    one phase of the native EXL3 plan (qwen2.h), with
#                        GGML's operations. CUDA builds only.
add_library(llmp_exl3 STATIC upstream_gemv.cc validate.cc)
target_include_directories(llmp_exl3 PUBLIC "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(llmp_exl3 PUBLIC llmp_exl3_headers PRIVATE llmp_warnings)

if(LLMP_CUDA)
  add_library(llmp_exl3_kernels STATIC implementations.cc launch.cc linear.cc recon_gemm.cc)
  target_include_directories(llmp_exl3_kernels SYSTEM PUBLIC
                             "${LLMP_SDK}/cuda/targets/${LLMP_CUDA_TARGET_DIR}/include")
  target_link_libraries(llmp_exl3_kernels
                        PUBLIC llmp_exl3 llmp_exl3_cuda llmp_device_memory llmp_cublaslt
                               llmp_execution
                        PRIVATE llmp_warnings)
  # What each implementation's identity records of the build (D-053): the
  # prepared ExLlamaV3 tree the build used, as the source receipt records
  # it (D-057), the SDK, the target, the kernels' device architectures, the
  # build type and any sanitizers. A missing receipt, entry or digest fails
  # configure: never an identity without the tree.
  get_property(sources GLOBAL PROPERTY LLMP_SOURCES_ENTRIES)
  set(exl3_tree "")
  string(JSON count ERROR_VARIABLE error LENGTH "${sources}")
  if(NOT error AND count GREATER 0)
    math(EXPR last "${count} - 1")
    foreach(i RANGE ${last})
      string(JSON id ERROR_VARIABLE error GET "${sources}" ${i} id)
      if(NOT error AND id STREQUAL "exllamav3")
        string(JSON exl3_tree ERROR_VARIABLE error GET "${sources}" ${i} source_tree)
      endif()
    endforeach()
  endif()
  string(LENGTH "${exl3_tree}" length)
  if(NOT length EQUAL 64 OR NOT exl3_tree MATCHES "^[0-9a-f]+$")
    message(FATAL_ERROR "The source receipt has no ExLlamaV3 tree digest for the registry's identities")
  endif()
  string(REPLACE ";" "+" sanitize "${LLMP_SANITIZE}")
  # The kernels' architectures as their target builds them: sm_121, and in
  # the native profile its discrete GPUs' too (D-082).
  get_target_property(architectures llmp_exl3_cuda CUDA_ARCHITECTURES)
  if(NOT architectures)
    message(FATAL_ERROR "llmp_exl3_cuda names no CUDA architectures for the registry's identities")
  endif()
  list(JOIN architectures "," architecture_text)
  set_source_files_properties(implementations.cc PROPERTIES COMPILE_DEFINITIONS
    "LLMP_EXL3_SOURCE_TREE=\"${exl3_tree}\";LLMP_EXL3_SDK=\"${LLMP_SDK_IDENTITY}\";LLMP_EXL3_TARGET=\"${LLMP_TARGET_TRIPLE}\";LLMP_EXL3_CUDA_ARCHITECTURES=\"${architecture_text}\";LLMP_EXL3_BUILD_TYPE=\"${CMAKE_BUILD_TYPE}\";LLMP_EXL3_SANITIZE=\"${sanitize}\"")

  # And llmpalooza's own code in the module: every file in this directory,
  # hashed at build time into a generated source (module_digest.cmake), as
  # the GGML module does. A file added here without being listed fails the
  # next configure.
  set(module_files CMakeLists.txt implementations.cc implementations.h launch.cc launch.h
                   launch_contract.h linear.cc linear.h module_digest.cmake qwen2.cc qwen2.h
                   recon_gemm.cc recon_gemm.h upstream_gemv.cc upstream_gemv.h validate.cc
                   validate.h)
  file(GLOB present CONFIGURE_DEPENDS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
       "${CMAKE_CURRENT_SOURCE_DIR}/[!.]*")
  foreach(name IN LISTS present)
    if(NOT name IN_LIST module_files AND NOT name MATCHES "(^#|~$|\\.orig$|\\.rej$)")
      message(FATAL_ERROR "src/kernels/exl3/${name} is not in module_files, so the "
                          "implementations' identity would not cover it")
    endif()
  endforeach()
  string(JOIN ":" module_names ${module_files})
  set_property(GLOBAL PROPERTY LLMP_EXL3_MODULE_NAMES "${module_names}")
  list(TRANSFORM module_files PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/" OUTPUT_VARIABLE module_paths)
  set(module_digest "${CMAKE_CURRENT_BINARY_DIR}/module_digest.cc")
  add_custom_command(
    OUTPUT "${module_digest}"
    COMMAND "${CMAKE_COMMAND}" "-DBASE=${CMAKE_CURRENT_SOURCE_DIR}" "-DNAMES=${module_names}"
            "-DOUTPUT=${module_digest}" -P "${CMAKE_CURRENT_SOURCE_DIR}/module_digest.cmake"
    DEPENDS ${module_paths}
    COMMENT "Hashing the EXL3 module's sources for its implementation identities (D-053)"
    VERBATIM)
  target_sources(llmp_exl3_kernels PRIVATE "${module_digest}")

  # One phase of the native EXL3 plan (model/qwen2_exl3.h) under llmpalooza's
  # dispatch: the module's linears and bias adds, and GGML's operations
  # (kernels/ggml, the layer's module before this one), bound through the
  # registry (qwen2.h).
  add_library(llmp_exl3_qwen2 STATIC qwen2.cc)
  target_link_libraries(llmp_exl3_qwen2 PUBLIC llmp_exl3_kernels llmp_ggml_kernels
                                                 llmp_model PRIVATE llmp_warnings)
endif()
