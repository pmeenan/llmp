# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0

# Run by cpack: the Debian version, architecture and dependencies of the
# build being packaged, which tools/llmp_package.py wrote (control.json).
if(NOT EXISTS "${CPACK_LLMP_CONTROL}")
  message(FATAL_ERROR "${CPACK_LLMP_CONTROL} is missing: package with `mise run package`")
endif()
file(READ "${CPACK_LLMP_CONTROL}" _llmp_control)
string(JSON _llmp_version GET "${_llmp_control}" version)
string(JSON CPACK_DEBIAN_PACKAGE_ARCHITECTURE GET "${_llmp_control}" architecture)
string(JSON CPACK_DEBIAN_PACKAGE_DEPENDS GET "${_llmp_control}" depends)
# "<upstream>-<revision>"
string(REGEX MATCH "^(.+)-([^-]+)$" _llmp_match "${_llmp_version}")
set(CPACK_DEBIAN_PACKAGE_VERSION "${CMAKE_MATCH_1}")
set(CPACK_DEBIAN_PACKAGE_RELEASE "${CMAKE_MATCH_2}")
set(CPACK_PACKAGE_VERSION "${CMAKE_MATCH_1}")
