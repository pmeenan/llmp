# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0

# The execution layer (docs/architecture.md#layers-and-dependency-rules): so
# far the implementation registry and plans (registry.h, D-053), which
# kernel modules declare into and never the reverse, and phase kinds,
# decoding modes and request programs with their envelopes (program.h,
# D-050, D-068). Model layer: no vendor or kernel-module types.
add_library(llmp_execution STATIC program.cc registry.cc)
target_include_directories(llmp_execution PUBLIC "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(llmp_execution PUBLIC llmp_base llmp_catalog llmp_memory llmp_model
                                       PRIVATE llmp_warnings)

# Greedy and seeded sampling on the host (sampling.h), for M3's decode
# loop and speculation checks: base only.
add_library(llmp_sampling STATIC sampling.cc)
target_include_directories(llmp_sampling PUBLIC "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(llmp_sampling PRIVATE llmp_warnings)
