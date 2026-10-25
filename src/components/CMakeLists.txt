# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0

# The component catalog (D-107; docs/engine-components.md#components): each
# component's parameters, weight roles, state contract, batching, splits,
# patterns, IR fragment and scalar CPU reference (component.h), and from a
# model spec the validated instances, state layout and a step's primitive
# graph (step.h). Model layer, after spec: no planner library, kernel,
# engine or vendor type (the planner's step shape is a header-only type).
add_library(llmp_components STATIC comparison_boundary.cc numerics.cc params.cc catalog.cc step.cc state_encoding.cc attention_core.cc rotation_reference.cc attention.cc ffn.cc moe.cc
                                   blocks.cc image_blocks.cc conditioning_blocks.cc spatial.cc spatial_blocks.cc causal_first_frame.cc hc.cc gdn.cc qsa.cc ple.cc mqa.cc inject.cc)
target_include_directories(llmp_components PUBLIC "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(llmp_components PUBLIC llmp_spec llmp_ir llmp_model llmp_format llmp_base
                                      PRIVATE llmp_warnings)
