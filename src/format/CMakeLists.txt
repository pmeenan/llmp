# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0

# Tensor formats (D-108; docs/engine-components.md#weight-and-state-formats):
# encodings, layouts, part roles, TensorFormat and its size rule. The model
# layer's first module (docs/architecture.md#layers-and-dependency-rules):
# base only, no artifact, kernel or vendor type.
add_library(llmp_format STATIC encoding.cc layout.cc tensor_format.cc)
target_include_directories(llmp_format PUBLIC "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(llmp_format PUBLIC llmp_base PRIVATE llmp_warnings)
