# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0

# A kernel module's implementation identities follow its sources (D-053;
# src/kernels/ggml/module_digest.cmake, and the EXL3 module's copy of it):
# - the digest names every file in the module's directory;
# - the build's generated digest is the digest of the sources as they are;
# - changing any one file, in a copy, changes the digest.
#
#   cmake -DBASE=<src/kernels/ggml or exl3> -DNAMES=<a:b:...> -DGENERATED=<module_digest.cc>
#         -DWORK=<scratch dir> -P ggml_module_digest_test.cmake

foreach(variable IN ITEMS BASE NAMES GENERATED WORK)
  if(NOT ${variable})
    message(FATAL_ERROR "ggml_module_digest_test.cmake needs ${variable}")
  endif()
endforeach()
set(script "${BASE}/module_digest.cmake")
string(REPLACE ":" ";" names "${NAMES}")

# Editors' hidden, backup and autosave files are skipped, as in the module's
# CMakeLists.txt.
file(GLOB present RELATIVE "${BASE}" "${BASE}/[!.]*")
foreach(name IN LISTS present)
  if(NOT name IN_LIST names AND NOT name MATCHES "(^#|~$|\\.orig$|\\.rej$)")
    message(FATAL_ERROR "${BASE}/${name} is not covered by the module digest")
  endif()
endforeach()

function(digest_of out base)
  execute_process(COMMAND "${CMAKE_COMMAND}" "-DBASE=${base}" "-DNAMES=${NAMES}" "-DCONFIGURATION=${CONFIGURATION}" -P "${script}"
                  RESULT_VARIABLE result ERROR_VARIABLE printed)
  string(STRIP "${printed}" printed)
  if(NOT result EQUAL 0 OR NOT printed MATCHES "^[0-9a-f]+$")
    message(FATAL_ERROR "module_digest.cmake failed on ${base}: ${printed}")
  endif()
  set(${out} "${printed}" PARENT_SCOPE)
endfunction()

digest_of(current "${BASE}")
file(READ "${GENERATED}" generated)
if(NOT generated MATCHES "return \"${current}\";")
  message(FATAL_ERROR "The build's module digest is not that of the current sources (${current})")
endif()

file(REMOVE_RECURSE "${WORK}")
# The module inventory also covers shared IR sources and its data generator.
# Keep their ../../ and ../../../ relative paths within the scratch root.
set(copy_base "${WORK}/root/src/kernels/module")
file(MAKE_DIRECTORY "${copy_base}")
foreach(name IN LISTS names)
  get_filename_component(parent "${copy_base}/${name}" DIRECTORY)
  file(MAKE_DIRECTORY "${parent}")
  file(COPY_FILE "${BASE}/${name}" "${copy_base}/${name}")
endforeach()
digest_of(copied "${copy_base}")
if(NOT copied STREQUAL current)
  message(FATAL_ERROR "A copy of the sources has another digest")
endif()
foreach(name IN LISTS names)
  file(APPEND "${copy_base}/${name}" "\n")
  digest_of(changed "${copy_base}")
  if(changed STREQUAL current)
    message(FATAL_ERROR "Changing ${name} left the module digest unchanged")
  endif()
  file(COPY_FILE "${BASE}/${name}" "${copy_base}/${name}")
endforeach()
if(CONFIGURATION)
  set(original_configuration "${CONFIGURATION}")
  string(SHA256 CONFIGURATION "different compile options")
  digest_of(changed_configuration "${BASE}")
  if(changed_configuration STREQUAL current)
    message(FATAL_ERROR "Changing build configuration left the module digest unchanged")
  endif()
  set(CONFIGURATION "${original_configuration}")
endif()
file(REMOVE_RECURSE "${WORK}")
list(LENGTH names count)
message(STATUS "The module digest ${current} covers all ${count} files")
