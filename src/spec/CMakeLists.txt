# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0

# The model spec (D-107, D-109; docs/engine-components.md#the-model-spec):
# its types, canonical JSON, digest and validation (spec.h), and the
# architecture mappings with the synthesis of a spec from a source or a v0
# artifact (mapping.h, mappings/). Model layer, after format, artifact, ir
# and model; no kernel, engine or vendor type.
add_library(llmp_spec STATIC spec.cc mapping.cc mappings/gemma2.cc mappings/gemma3.cc)
target_include_directories(llmp_spec PUBLIC "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(llmp_spec PUBLIC llmp_format llmp_base PRIVATE llmp_artifact llmp_model
                                                                    llmp_warnings)
