# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0

# Preparation, configure and build use one definition of source identity,
# including owner-executable bits and rejection of unreadable/special files.
# CMake's IS_EXECUTABLE tests access, not the owner mode bit in the format.
find_program(LLMP_PYTHON NAMES python3 REQUIRED DOC "Python 3 for source validation (mise's, under mise)")
get_filename_component(_LLMP_SOURCE_INSPECTOR "${CMAKE_CURRENT_LIST_DIR}/../../tools/inspect-sources" ABSOLUTE)

# Sets <out> to the digest of <dir>, or to "invalid: <reason>".
function(llmp_tree_digest out dir)
  execute_process(COMMAND "${LLMP_PYTHON}" "${_LLMP_SOURCE_INSPECTOR}" --tree "${dir}"
                  RESULT_VARIABLE result OUTPUT_VARIABLE digest ERROR_VARIABLE error
                  OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_STRIP_TRAILING_WHITESPACE)
  if(NOT result STREQUAL "0")
    set(digest "invalid: ${error}")
  endif()
  set(${out} "${digest}" PARENT_SCOPE)
endfunction()
