# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0

# The graph IR (D-107; docs/engine-components.md#the-graph-ir), so far its
# op set v1 (op.h): operations, fused kinds and parameters. Model layer,
# after format and artifact: no kernel, execution or vendor type.
add_library(llmp_ir STATIC op.cc)
target_include_directories(llmp_ir PUBLIC "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(llmp_ir PUBLIC llmp_format PRIVATE llmp_warnings)
