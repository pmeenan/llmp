# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0

# The planner (D-053, D-107; docs/engine-components.md#the-graph-ir): the
# step shapes and plan keys (step_shape.h, header-only, which components
# include), the policy, the pattern table, implementation selection over
# the registry's capabilities, lanes, activation placement and Plan(),
# from a step's primitive graph to a StepPlan a lowering takes; and
# partitioning over a described topology (partition.h: Assign, Partition).
# Model layer, after components: no kernel, engine or vendor type.
add_library(llmp_planner STATIC policy.cc select.cc pattern.cc lanes.cc placement.cc plan.cc
                                partition.cc)
target_include_directories(llmp_planner PUBLIC "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(llmp_planner PUBLIC llmp_components llmp_spec llmp_model llmp_ir
                                          llmp_execution llmp_format llmp_base
                                   PRIVATE llmp_warnings)
