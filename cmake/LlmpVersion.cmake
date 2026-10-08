# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0

# The product version (D-062), from project(VERSION) and Git
# (cmake/version/derive.cmake has the rules).
#
# Including this derives it once at configure, which stops on a version the
# rules refuse and sets LLMP_BUILD_VERSION, LLMP_BUILD_DEBIAN,
# LLMP_BUILD_COMMIT, LLMP_BUILD_DIRTY, LLMP_BUILD_ORIGIN and
# LLMP_BUILD_GIT, and LLMP_BUILD_JSON for the build receipt.
#
#   llmp_build_info_library(<target>)  after llmp_sources_add(): a static
#       library defining base::GetBuildInfo() (src/base/build_info.h). Every
#       build derives the version again (cmake/version/stamp.cmake) and
#       updates the library's generated source and the receipt, so neither
#       goes stale when a commit or an edit needs no configure.

include_guard(GLOBAL)
include("${CMAKE_CURRENT_LIST_DIR}/version/derive.cmake")

set(_LLMP_VERSION_SCRIPTS "${CMAKE_CURRENT_LIST_DIR}/version")

llmp_version_derive(LLMP_BUILD SOURCE_DIR "${PROJECT_SOURCE_DIR}" PROJECT_VERSION "${PROJECT_VERSION}")
llmp_version_json(LLMP_BUILD_JSON LLMP_BUILD)
message(STATUS "llmpalooza version ${LLMP_BUILD_VERSION}")

function(llmp_build_info_library target)
  get_property(license_profile GLOBAL PROPERTY LLMP_SOURCES_PROFILE)
  if(NOT license_profile)
    message(FATAL_ERROR "call llmp_sources_add() before llmp_build_info_library()")
  endif()
  set(generated "${CMAKE_CURRENT_BINARY_DIR}/${target}.cc")
  add_custom_target(${target}_stamp ALL
    COMMAND "${CMAKE_COMMAND}" "-DSOURCE_DIR=${PROJECT_SOURCE_DIR}" "-DPROJECT_VERSION=${PROJECT_VERSION}"
            "-DGIT=${LLMP_BUILD_GIT}" "-DOUTPUT=${generated}"
            "-DRECEIPT=${PROJECT_BINARY_DIR}/llmp-receipt.json" "-DLICENSE_PROFILE=${license_profile}"
            "-DSDK=${LLMP_SDK_IDENTITY}" "-DTARGET=${LLMP_TARGET_TRIPLE}"
            -P "${_LLMP_VERSION_SCRIPTS}/stamp.cmake"
    BYPRODUCTS "${generated}"
    COMMENT "Deriving the llmpalooza version (D-062)"
    VERBATIM)
  add_library(${target} STATIC "${generated}")
  add_dependencies(${target} ${target}_stamp)
  target_include_directories(${target} PUBLIC "${PROJECT_SOURCE_DIR}/src")
  target_link_libraries(${target} PRIVATE llmp_warnings)
endfunction()
