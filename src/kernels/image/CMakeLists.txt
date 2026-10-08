# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0

# Llmpalooza's own kernels for the Qwen-Image-2.1 pipeline's BF16 operations
# (ops.h, conv.cu), its BF16 products on cuBLAS and cuBLASLt (gemm.h), the
# module's declarations in the implementation registry (implementations.h)
# and the pipeline's phases dispatched through a bound plan (pipeline.h).
# CUDA builds only.
if(LLMP_CUDA)
  add_library(llmp_image_kernels STATIC ops.cu flash_attention.cu conv.cu gemm.cc
                                          implementations.cc pipeline.cc)
  target_include_directories(llmp_image_kernels PUBLIC "${PROJECT_SOURCE_DIR}/src")
  target_include_directories(llmp_image_kernels SYSTEM PRIVATE
                             "${LLMP_SDK}/cuda/targets/${LLMP_CUDA_TARGET_DIR}/include")
  target_link_libraries(llmp_image_kernels PUBLIC llmp_cublas llmp_cublaslt llmp_execution
                                                    llmp_model
                                             PRIVATE llmp_warnings)
  # sm_121a, as the GGML module's units build, and the profile's discrete
  # GPUs (D-082).
  set_target_properties(llmp_image_kernels PROPERTIES
                        CUDA_ARCHITECTURES "121a-real;${LLMP_CUDA_DISCRETE_SASS}")

  # What each implementation's identity records of the build (D-053): the
  # SDK, the target, the kernels' device architectures, the build type and
  # any sanitizers.
  string(REPLACE ";" "+" sanitize "${LLMP_SANITIZE}")
  get_target_property(architectures llmp_image_kernels CUDA_ARCHITECTURES)
  list(JOIN architectures "," architecture_text)
  set_source_files_properties(implementations.cc PROPERTIES COMPILE_DEFINITIONS
    "LLMP_IMAGE_SDK=\"${LLMP_SDK_IDENTITY}\";LLMP_IMAGE_TARGET=\"${LLMP_TARGET_TRIPLE}\";LLMP_IMAGE_CUDA_ARCHITECTURES=\"${architecture_text}\";LLMP_IMAGE_BUILD_TYPE=\"${CMAKE_BUILD_TYPE}\";LLMP_IMAGE_SANITIZE=\"${sanitize}\"")

  # And llmpalooza's own code in the module: every file in this directory,
  # hashed at build time into a generated source (module_digest.cmake), as
  # the GGML and EXL3 modules do. A file added here without being listed
  # fails the next configure.
  set(module_files CMakeLists.txt conv.cu flash_attention.cu gemm.cc gemm.h implementations.cc
                   implementations.h module_digest.cmake ops.cu ops.h pipeline.cc pipeline.h)
  file(GLOB present CONFIGURE_DEPENDS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
       "${CMAKE_CURRENT_SOURCE_DIR}/[!.]*")
  foreach(name IN LISTS present)
    if(NOT name IN_LIST module_files AND NOT name MATCHES "(^#|~$|\\.orig$|\\.rej$)")
      message(FATAL_ERROR "src/kernels/image/${name} is not in module_files, so the "
                          "implementations' identity would not cover it")
    endif()
  endforeach()
  string(JOIN ":" module_names ${module_files})
  list(TRANSFORM module_files PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/" OUTPUT_VARIABLE module_paths)
  set(module_digest "${CMAKE_CURRENT_BINARY_DIR}/module_digest.cc")
  add_custom_command(
    OUTPUT "${module_digest}"
    COMMAND "${CMAKE_COMMAND}" "-DBASE=${CMAKE_CURRENT_SOURCE_DIR}" "-DNAMES=${module_names}"
            "-DOUTPUT=${module_digest}" -P "${CMAKE_CURRENT_SOURCE_DIR}/module_digest.cmake"
    DEPENDS ${module_paths}
    COMMENT "Hashing the image module's sources for its implementation identities (D-053)"
    VERBATIM)
  target_sources(llmp_image_kernels PRIVATE "${module_digest}")
endif()
