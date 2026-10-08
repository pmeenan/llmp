# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0

# What this build is (build_info.h): its definition is generated on every
# build from Git and the build configuration (D-062, cmake/LlmpVersion.cmake).
llmp_build_info_library(llmp_build_info)

# The versions of llmpalooza's own public surfaces (surface_versions.h).
add_library(llmp_surface_versions INTERFACE)
target_include_directories(llmp_surface_versions INTERFACE "${PROJECT_SOURCE_DIR}/src")

# Invariant checks (check.h), typed identities (ids.h), byte counts
# (bytes.h) and SHA-256 (sha256.h).
add_library(llmp_base STATIC check.cc sha256.cc)
target_include_directories(llmp_base PUBLIC "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(llmp_base PRIVATE llmp_warnings)

# Diagnostic reports (report.h), such as `llmp doctor` prints.
add_library(llmp_report STATIC report.cc)
target_include_directories(llmp_report PUBLIC "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(llmp_report PRIVATE llmp_warnings)

# General JSON (json.h): tokenizer.json, chat fixtures and later request
# bodies, parsed as untrusted input. Its own library, so that only what
# reads JSON links it (and floating-point from_chars).
add_library(llmp_json STATIC json.cc)
target_include_directories(llmp_json PUBLIC "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(llmp_json PRIVATE llmp_warnings)
