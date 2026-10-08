# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0

# Checks one executable against the link contract (D-011, D-060, D-070): the
# target's ELF machine and dynamic loader, glibc as its only shared
# libraries (the C++ and CUDA runtimes are static), no symbol version newer
# than GLIBC_MAX (if given), and no RPATH or RUNPATH. A sanitizer build
# (SANITIZED) may also need glibc's libresolv, which the sanitizer runtimes
# use. A binary that uses the CUDA driver (CUDA_DRIVER) must also need the
# driver's libcuda.so.1, and one that does not must not (D-072).
#
#   cmake -DREADELF=<llvm-readelf> -DARCH=<x86_64|aarch64> [-DGLIBC_MAX=<x.y>]
#         [-DSANITIZED=ON] [-DCUDA_DRIVER=ON] -DBINARY=<file> -P check_binary.cmake
cmake_minimum_required(VERSION 4.4.3)

foreach(var IN ITEMS READELF ARCH BINARY)
  if(NOT ${var})
    message(FATAL_ERROR "set ${var}")
  endif()
endforeach()
if(ARCH STREQUAL "x86_64")
  set(machine "Advanced Micro Devices X86-64")
  set(loader "/lib64/ld-linux-x86-64.so.2")
elseif(ARCH STREQUAL "aarch64")
  set(machine "AArch64")
  set(loader "/lib/ld-linux-aarch64.so.1")
else()
  message(FATAL_ERROR "unknown ARCH ${ARCH}")
endif()

execute_process(
  COMMAND "${READELF}" --file-header --program-headers --dynamic --version-info "${BINARY}"
  OUTPUT_VARIABLE elf ERROR_VARIABLE error RESULT_VARIABLE result)
if(result)
  message(FATAL_ERROR "${READELF} failed on ${BINARY}: ${error}")
endif()

set(problems "")
if(NOT elf MATCHES "Class: +ELF64\n")
  list(APPEND problems "not an ELF64 file")
endif()
if(NOT elf MATCHES "Machine: +${machine}\n")
  list(APPEND problems "not built for ${ARCH}")
endif()
string(REPLACE "." "\\." loader_pattern "${loader}")
if(NOT elf MATCHES "\\[Requesting program interpreter: ${loader_pattern}\\]")
  list(APPEND problems "dynamic loader is not ${loader}")
endif()
cmake_path(GET loader FILENAME loader_name)
set(glibc_libraries "^lib(c|m)\\.so\\.6$")
if(SANITIZED)
  set(glibc_libraries "^(lib(c|m)\\.so\\.6|libresolv\\.so\\.2)$")
endif()
string(REGEX MATCHALL "\\(NEEDED\\) +Shared library: \\[[^]\n]+\\]" entries "${elf}")
set(needed "")
foreach(entry IN LISTS entries)
  string(REGEX REPLACE ".*\\[(.*)\\]" "\\1" library "${entry}")
  list(APPEND needed "${library}")
  if(library STREQUAL "libcuda.so.1")
    if(NOT CUDA_DRIVER)
      list(APPEND problems "needs libcuda.so.1, but uses no CUDA driver")
    endif()
  elseif(NOT library STREQUAL loader_name AND NOT library MATCHES "${glibc_libraries}")
    list(APPEND problems "needs ${library}, but only glibc and the CUDA driver may be dynamic (D-060)")
  endif()
endforeach()
if(CUDA_DRIVER AND NOT "libcuda.so.1" IN_LIST needed)
  list(APPEND problems "does not need libcuda.so.1, the CUDA driver (D-072)")
endif()
if(elf MATCHES "\\((RPATH|RUNPATH)\\)")
  list(APPEND problems "has an RPATH or RUNPATH")
endif()
# Numbered versions must exist in the target's glibc; GLIBC_PRIVATE is never
# a stable interface. ABI markers such as GLIBC_ABI_DT_RELR are allowed.
string(REGEX MATCHALL "GLIBC_[0-9A-Z_.]+" versions "${elf}")
list(REMOVE_DUPLICATES versions)
foreach(version IN LISTS versions)
  if(version MATCHES "^GLIBC_([0-9]+\\.[0-9]+)(\\.[0-9]+)?$")
    if(GLIBC_MAX AND CMAKE_MATCH_1 VERSION_GREATER GLIBC_MAX)
      list(APPEND problems "needs ${version}, newer than the target's glibc ${GLIBC_MAX}")
    endif()
  elseif(version STREQUAL "GLIBC_PRIVATE")
    list(APPEND problems "uses GLIBC_PRIVATE")
  endif()
endforeach()

if(problems)
  list(JOIN problems "\n  " text)
  message(FATAL_ERROR "${BINARY}:\n  ${text}")
endif()
list(JOIN needed ", " text)
message(STATUS "${BINARY}: ${ARCH}, ${loader}, needs ${text}")
