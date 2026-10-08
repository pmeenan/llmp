# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0

# The memory manager's policy (D-050, D-055): the commitment ledger, the
# victim-selection baseline, materialization planning and the reclaim order.
# Resource core: no vendor types.
add_library(llmp_memory STATIC commitment.cc materialize.cc reclaim.cc victims.cc)
target_include_directories(llmp_memory PUBLIC "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(llmp_memory PUBLIC llmp_base llmp_catalog PRIVATE llmp_warnings)
