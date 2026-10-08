# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0

# The `llmp` command (/usr/bin/llmp once packaged, D-063).
add_library(llmp_cli STATIC cli.cc doctor.cc)
target_include_directories(llmp_cli PUBLIC "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(llmp_cli PUBLIC llmp_build_info llmp_report
                      PRIVATE llmp_config llmp_platform llmp_device_probe llmp_warnings)

add_executable(llmp main.cc)
target_link_libraries(llmp PRIVATE llmp_cli llmp_warnings)
