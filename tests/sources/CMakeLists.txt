# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0

# The source mechanism (D-017, D-057), checked on this build and, in the CPU
# profile, end to end on a synthetic lock. Every test carries the `sources`
# label.

# LLMP_PYTHON comes from cmake/sources/TreeDigest.cmake.

# The whole lock is valid, every patch file included, not only the closure
# configure read (D-017, D-057).
add_test(NAME sources.lock
         COMMAND ${LLMP_PYTHON} ${PROJECT_SOURCE_DIR}/tools/prepare-sources --check)

# The receipt matches the lock and names exactly the profile's closure.
add_test(NAME sources.receipt
         COMMAND ${CMAKE_COMMAND} -DRECEIPT=${PROJECT_BINARY_DIR}/llmp-receipt.json
                 -DLOCK=${PROJECT_SOURCE_DIR}/third_party/sources.lock.json
                 -DSDK_IDENTITY=${LLMP_SDK_IDENTITY} -P ${CMAKE_CURRENT_SOURCE_DIR}/check_receipt.cmake)

# What the build actually compiled and linked is the receipt's closure, from
# allowed places only, without exceptions, and with libstdc++'s assertions
# exactly where the preset asks for them (D-083).
set(closure_flags "")
if(CMAKE_CROSSCOMPILING)
  list(APPEND closure_flags --cross)
endif()
if(LLMP_LIBSTDCXX_ASSERTIONS)
  list(APPEND closure_flags --libstdcxx-assertions)
endif()
add_test(NAME sources.closure
         COMMAND ${LLMP_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/check_closure.py
                 --build-dir ${PROJECT_BINARY_DIR} --source-dir ${PROJECT_SOURCE_DIR}
                 --sdk ${LLMP_SDK} --ninja ${CMAKE_MAKE_PROGRAM} ${closure_flags})
set_tests_properties(sources.lock sources.receipt sources.closure PROPERTIES LABELS sources)

# The mechanism's gates on fixtures. They build natively for the host, so
# they run once, in the CPU-only profile (not again in its sanitizer build).
if(NOT CMAKE_CROSSCOMPILING AND NOT LLMP_CUDA AND NOT LLMP_SANITIZE)
  add_test(NAME sources.mechanism
           COMMAND ${LLMP_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/mechanism_test.py
                   --repo ${PROJECT_SOURCE_DIR} --work ${CMAKE_CURRENT_BINARY_DIR}/mechanism
                   --cmake ${CMAKE_COMMAND} --ninja ${CMAKE_MAKE_PROGRAM} --sdk ${LLMP_SDK}
                   --toolchain ${CMAKE_TOOLCHAIN_FILE})
  set_tests_properties(sources.mechanism PROPERTIES LABELS sources TIMEOUT 600)
endif()

# Standalone Gemma routing/scaled reduction reuse the unchanged pinned math.
add_test(NAME sources.gemma_moe
         COMMAND ${LLMP_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/gemma_moe_test.py
                 ${PROJECT_BINARY_DIR})
set_tests_properties(sources.gemma_moe PROPERTIES LABELS sources)

# Focused auditing is opt-in; its no-compile controls preserve the full default audit.
add_test(NAME sources.focused_closure
         COMMAND ${LLMP_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/focused_closure_test.py)
set_tests_properties(sources.focused_closure PROPERTIES LABELS sources)
