# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0

# The graph IR (D-107; docs/engine-components.md#the-graph-ir): op set v1
# (op.h), tensors and graphs (graph.h), the builder, shape rules, verifier
# and the primitive decompositions. Model layer, after format: base and
# format only, no artifact, kernel, execution or vendor type.
add_library(llmp_ir STATIC op.cc op_traits.cc graph.cc infer.cc builder.cc verify.cc decompose.cc layout.cc)
target_include_directories(llmp_ir PUBLIC "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(llmp_ir PUBLIC llmp_format llmp_base PRIVATE llmp_warnings)
