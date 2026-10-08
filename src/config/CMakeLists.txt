# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0

# The node's configuration (D-063, D-073): the strict TOML document and its
# storage roles. toml++ stays inside node_config.cc, in its TOML 1.0.0 mode
# without exceptions (D-066) or formatters.
add_library(llmp_config STATIC node_config.cc storage_roles.cc)
target_include_directories(llmp_config PUBLIC "${PROJECT_SOURCE_DIR}/src")
target_compile_definitions(llmp_config PRIVATE TOML_EXCEPTIONS=0 TOML_ENABLE_FORMATTERS=0
                                                 TOML_ENABLE_UNRELEASED_FEATURES=0 TOML_HEADER_ONLY=1)
target_link_libraries(llmp_config PUBLIC llmp_report llmp_surface_versions PRIVATE llmp_platform tomlplusplus::tomlplusplus llmp_warnings)
