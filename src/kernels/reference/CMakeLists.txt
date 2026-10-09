# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0

# CPU references (D-108): every tensor format's decoder (decode.h), against
# which device decoders and products are checked. Every profile, no device
# code; GGML's block encodings decode through ggml-base's dequantization.
add_library(llmp_reference_kernels STATIC decode.cc)
target_include_directories(llmp_reference_kernels PUBLIC "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(llmp_reference_kernels PUBLIC llmp_format PRIVATE llmp_ggml_base llmp_warnings)
