# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0

# CPU references (D-107, D-108): every tensor format's decoder (decode.h),
# against which device decoders and products are checked, and the
# reference executor of every op of the graph IR (exec.h), the oracle of
# components and backend ports. Every profile, no device code; GGML's
# block encodings decode through ggml-base's dequantization, the EXL3
# trellis as ExLlamaV3's kernels reconstruct it (decode_exl3.cc).
add_library(llmp_reference_kernels STATIC decode.cc decode_exl3.cc exec.cc)
target_include_directories(llmp_reference_kernels PUBLIC "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(llmp_reference_kernels PUBLIC llmp_format llmp_ir
                      PRIVATE llmp_ggml_base llmp_sampling llmp_warnings)
