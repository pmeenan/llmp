# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0

# The installed layout (D-063) and the Debian package (D-027, D-074), built
# with CPack from the cross profile by `mise run package` (tools/build
# package), which first writes the package's documents into
# build/<preset>/package/doc (tools/llmp_package.py). Paths are relative
# to the package's /usr.
set(package_doc "${PROJECT_BINARY_DIR}/package/doc")
# Directories are 0755 whatever the packager's umask.
set(directory_mode OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
set(CMAKE_INSTALL_DEFAULT_DIRECTORY_PERMISSIONS ${directory_mode})
set(CPACK_INSTALL_DEFAULT_DIRECTORY_PERMISSIONS ${directory_mode})

install(TARGETS llmp RUNTIME DESTINATION bin)
install(TARGETS llmp-runtime RUNTIME DESTINATION libexec/llmp)
install(FILES llmp.service DESTINATION lib/systemd/system)
install(FILES llmp.sysusers DESTINATION lib/sysusers.d RENAME llmp.conf)
install(FILES llmp.tmpfiles DESTINATION lib/tmpfiles.d RENAME llmp.conf)
install(FILES "${PROJECT_SOURCE_DIR}/LICENSE" "${PROJECT_SOURCE_DIR}/NOTICE" "${PROJECT_SOURCE_DIR}/CHANGELOG.md"
              "${package_doc}/copyright" "${package_doc}/THIRD-PARTY-NOTICES" "${package_doc}/llmp.spdx.json"
        DESTINATION share/doc/llmp)
install(FILES llmp.example.toml DESTINATION share/doc/llmp/examples RENAME llmp.toml)
if(LLMP_CUDA)
  # cuBLAS (D-076): the SDK's two shared libraries, as the build tree's
  # lib/llmp links them, in the runtime's private directory, which its run
  # path names (src/runtime/CMakeLists.txt). install(FILES) never strips, so
  # they ship unmodified and unstripped, as the EULA's section 2.3 requires;
  # CPACK_STRIP_FILES below strips the installed targets alone.
  install(FILES "${PROJECT_BINARY_DIR}/lib/llmp/libcublas.so.13"
                "${PROJECT_BINARY_DIR}/lib/llmp/libcublasLt.so.13"
          DESTINATION lib/llmp PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ)
endif()

set(CPACK_GENERATOR DEB)
set(CPACK_PACKAGE_NAME llmp)
set(CPACK_PACKAGE_VENDOR "llmpalooza contributors")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "just-in-time LLM inference engine with SSD paging")
string(CONCAT CPACK_PACKAGE_DESCRIPTION
       "llmpalooza keeps the useful parts of many models resident, reclaims the least valuable extents\n"
       "across models when memory is needed, and pages weights and state back from prepared on-disk\n"
       "artifacts on demand. This package holds the node runtime (llmp.service) and the llmp command.")
set(CPACK_PACKAGE_DIRECTORY "${PROJECT_BINARY_DIR}/package")
set(CPACK_PACKAGING_INSTALL_PREFIX /usr)
set(CPACK_STRIP_FILES ON)
set(CPACK_DEBIAN_FILE_NAME DEB-DEFAULT)
set(CPACK_DEBIAN_PACKAGE_SECTION misc)
set(CPACK_DEBIAN_PACKAGE_PRIORITY optional)
set(CPACK_DEBIAN_PACKAGE_HOMEPAGE "https://github.com/pmeenan/llmp")
# The owner's address (2026-09-24, D-074).
set(CPACK_DEBIAN_PACKAGE_MAINTAINER "Patrick Meenan <pmeenan@llmpalooza.com>")
set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS OFF)
set(CPACK_DEBIAN_PACKAGE_CONTROL_EXTRA "${CMAKE_CURRENT_SOURCE_DIR}/debian/postinst"
                                       "${CMAKE_CURRENT_SOURCE_DIR}/debian/prerm"
                                       "${CMAKE_CURRENT_SOURCE_DIR}/debian/postrm")
set(CPACK_DEBIAN_PACKAGE_CONTROL_STRICT_PERMISSION ON)
# The version, architecture and dependencies come from the build that is
# packaged, at package time (cpack-project.cmake reads control.json).
set(CPACK_PROJECT_CONFIG_FILE "${CMAKE_CURRENT_SOURCE_DIR}/cpack-project.cmake")
set(CPACK_LLMP_CONTROL "${PROJECT_BINARY_DIR}/package/doc/control.json")
set(CPACK_INSTALL_CMAKE_PROJECTS "${PROJECT_BINARY_DIR};llmp;ALL;/")
include(CPack)
