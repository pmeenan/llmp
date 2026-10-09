# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0

# Block decoders under shared product templates (D-108): a decoder per
# encoding and layout (decoders.cuh), the vector, GLU, matrix, grouped and
# rows templates over them (templates.cuh), one CUDA unit per encoding
# (decode_*.cu), the checked host launch (products.h) and the module's
# declarations in the implementation registry (implementations.h). CUDA
# builds only. Device code takes the format geometry from operand.h's
# local copies, which decode_test checks against the format layer.
if(LLMP_CUDA)
  add_library(llmp_decode_kernels STATIC decode_dense.cu decode_q8_0.cu decode_nvfp4.cu
                                         decode_mxfp8.cu decode_mixed.cu products.cc
                                         implementations.cc)
  target_include_directories(llmp_decode_kernels PUBLIC "${PROJECT_SOURCE_DIR}/src")
  target_include_directories(llmp_decode_kernels SYSTEM PRIVATE
                             "${LLMP_SDK}/cuda/targets/${LLMP_CUDA_TARGET_DIR}/include")
  target_link_libraries(llmp_decode_kernels PUBLIC llmp_execution llmp_format llmp_ir
                                            PRIVATE llmp_warnings)
  # sm_121a, as the other kernel modules build, and the profile's discrete
  # GPUs (D-082).
  set_target_properties(llmp_decode_kernels PROPERTIES
                        CUDA_ARCHITECTURES "121a-real;${LLMP_CUDA_DISCRETE_SASS}")

  # What each implementation's identity records of the build (D-053).
  string(REPLACE ";" "+" sanitize "${LLMP_SANITIZE}")
  get_target_property(architectures llmp_decode_kernels CUDA_ARCHITECTURES)
  list(JOIN architectures "," architecture_text)
  set_source_files_properties(implementations.cc PROPERTIES COMPILE_DEFINITIONS
    "LLMP_DECODE_SDK=\"${LLMP_SDK_IDENTITY}\";LLMP_DECODE_TARGET=\"${LLMP_TARGET_TRIPLE}\";LLMP_DECODE_CUDA_ARCHITECTURES=\"${architecture_text}\";LLMP_DECODE_BUILD_TYPE=\"${CMAKE_BUILD_TYPE}\";LLMP_DECODE_SANITIZE=\"${sanitize}\"")

  # And llmpalooza's own code in the module: every file in this directory,
  # hashed at build time into a generated source (module_digest.cmake). A
  # file added here without being listed fails the next configure.
  set(module_files CMakeLists.txt decode_dense.cu decode_mixed.cu decode_mxfp8.cu decode_nvfp4.cu
                   decode_q8_0.cu decoders.cuh implementations.cc implementations.h launch.h
                   module_digest.cmake operand.h products.cc products.h templates.cuh)
  file(GLOB present CONFIGURE_DEPENDS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
       "${CMAKE_CURRENT_SOURCE_DIR}/[!.]*")
  foreach(name IN LISTS present)
    if(NOT name IN_LIST module_files AND NOT name MATCHES "(^#|~$|\\.orig$|\\.rej$)")
      message(FATAL_ERROR "src/kernels/decode/${name} is not in module_files, so the "
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
    COMMENT "Hashing the decode module's sources for its implementation identities (D-053)"
    VERBATIM)
  target_sources(llmp_decode_kernels PRIVATE "${module_digest}")
endif()
