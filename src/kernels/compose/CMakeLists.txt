# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0

# The CUDA build's composition root (kernel_set.h): the one library that
# links every kernel source (GGML's step backend, the decoders, EXL3) and
# registers them as the process's kernel set (kernels/common/device_backend.h).
# The sources link only the common interface, never each other.
if(LLMP_CUDA)
  add_library(llmp_kernel_set STATIC kernel_set.cc)
  target_include_directories(llmp_kernel_set PUBLIC "${PROJECT_SOURCE_DIR}/src")
  target_link_libraries(llmp_kernel_set PUBLIC llmp_step_common
                                        PRIVATE llmp_ggml_kernels llmp_decode_selected
                                                llmp_exl3_selected llmp_warnings)
endif()
