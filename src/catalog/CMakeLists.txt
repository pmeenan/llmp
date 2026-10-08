# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0

# The resource catalog (D-006, D-007): extents, resources, leases,
# registrations, generations and occupancy. Resource core: no vendor types.
add_library(llmp_catalog STATIC catalog.cc)
target_include_directories(llmp_catalog PUBLIC "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(llmp_catalog PUBLIC llmp_base PRIVATE llmp_warnings)
