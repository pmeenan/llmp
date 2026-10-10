# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0

# The prepared-artifact reader (docs/artifact-format.md; D-056, and D-113's
# version 1): a strict parser of the manifest and index, their validation
# as untrusted input, and the groups, chunks, placements and direct-read
# plans of a validated artifact; compositions of artifacts (D-089). The
# model layer (docs/architecture.md#layers-and-dependency-rules): it needs
# only base, format and platform, and no vendor SDK. v0's representation
# descriptors read as tensor formats (format_map.h, D-108); v1 records the
# formats themselves (format_json.h).
add_library(llmp_artifact STATIC artifact.cc composition.cc error.cc format_json.cc format_map.cc
                                   gguf_metadata.cc json.cc layout.cc representation.cc)
target_include_directories(llmp_artifact PUBLIC "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(llmp_artifact PUBLIC llmp_base llmp_format PRIVATE llmp_gguf llmp_platform llmp_warnings)
