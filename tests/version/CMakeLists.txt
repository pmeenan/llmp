# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0

# The product version (D-062). Every test carries the `version` label.

# The `llmp` binary: the link contract (tests/toolchain), and `--version`
# reporting what the receipt records, which is this checkout as it is now.
# The product binary gets no runner of its own; a cross build runs it with
# the unit tests' (tools/run-target).
string(REGEX MATCH "^[^-]+" target_arch "${LLMP_TARGET_TRIPLE}")
if(target_arch STREQUAL "aarch64")
  set(glibc_max 2.39)
else()
  set(glibc_max "")
endif()
if(LLMP_SANITIZE)
  set(sanitized ON)
else()
  set(sanitized OFF)
endif()
add_test(NAME version.llmp.binary
         COMMAND ${CMAKE_COMMAND} -DREADELF=${CMAKE_READELF} -DARCH=${target_arch}
                 -DGLIBC_MAX=${glibc_max} -DSANITIZED=${sanitized} -DCUDA_DRIVER=${LLMP_CUDA}
                 -DBINARY=$<TARGET_FILE:llmp>
                 -P ${PROJECT_SOURCE_DIR}/tests/toolchain/check_binary.cmake)
add_test(NAME version.llmp
         COMMAND ${CMAKE_COMMAND} -DBINARY=$<TARGET_FILE:llmp>
                 "-DEMULATOR=$<JOIN:$<TARGET_PROPERTY:cli_test,CROSSCOMPILING_EMULATOR>,|>"
                 -DRECEIPT=${PROJECT_BINARY_DIR}/llmp-receipt.json -DSOURCE_DIR=${PROJECT_SOURCE_DIR}
                 -DPROJECT_VERSION=${PROJECT_VERSION} -DGIT=${LLMP_BUILD_GIT}
                 -P ${CMAKE_CURRENT_SOURCE_DIR}/check_version.cmake)
set_tests_properties(version.llmp.binary version.llmp PROPERTIES LABELS version)

# The derivation rules and the build-time stamp on synthetic repositories.
# They run on the build host, so once, in the CPU-only profile.
if(NOT CMAKE_CROSSCOMPILING AND NOT LLMP_CUDA AND NOT LLMP_SANITIZE)
  add_test(NAME version.derive
           COMMAND ${LLMP_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/derive_test.py
                   --repo ${PROJECT_SOURCE_DIR} --cmake ${CMAKE_COMMAND}
                   --work ${CMAKE_CURRENT_BINARY_DIR}/derive)
  set_tests_properties(version.derive PROPERTIES LABELS version TIMEOUT 300)
endif()
