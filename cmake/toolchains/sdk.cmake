# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0

# The shared part of the profile toolchain files beside it (D-011, D-032,
# D-049, D-060, D-070). A profile sets these, then includes this file:
#
#   LLMP_PROFILE        the profile's name, for messages and the project
#   LLMP_BUILD_ARCH     the build host architecture the profile runs on
#   LLMP_TARGET_TRIPLE  the target triple Clang compiles for
#   LLMP_TARGET_MARCH   the explicit target CPU baseline (never `native`)
#   LLMP_LINK_FLAGS     how the SDK's Clang links for this target
#   LLMP_SYSROOT        cross builds only: the sysroot, relative to the SDK
#   LLMP_CUDA_DISCRETE_ARCHITECTURES
#                         x86-64 only, optional: the discrete NVIDIA GPUs a
#                         CUDA build also targets beside the GB10 (D-082), as
#                         compute capabilities without the dot (86 for sm_86)
#
# Compilers and build tools come from the SDK that `mise run setup` provisions;
# Spark-native also uses the host's declared GNU linker. LLMP_SDK names the
# SDK root, as a cache entry or environment variable (`mise run build` passes
# it), and the SDK's receipt must match this checkout's toolchain inputs. LLMP_CUDA
# (default ON) selects CUDA; with it OFF, nothing here refers to the SDK's
# CUDA components (the CPU-only profile, D-026).

cmake_path(GET CMAKE_CURRENT_LIST_DIR PARENT_PATH _llmp_cmake_dir)
cmake_path(GET _llmp_cmake_dir PARENT_PATH LLMP_SOURCE_ROOT)

# CMake re-reads the toolchain file in every try_compile project; these carry
# the settings there.
list(APPEND CMAKE_TRY_COMPILE_PLATFORM_VARIABLES LLMP_SDK LLMP_CUDA LLMP_TOOLCHAIN_DIR
     LLMP_CUDA_DISCRETE_ARCHITECTURES)

# The SDK -----------------------------------------------------------------------

if(NOT LLMP_SDK)
  set(LLMP_SDK "$ENV{LLMP_SDK}")
endif()
if(NOT LLMP_SDK)
  message(FATAL_ERROR
    "LLMP_SDK is not set. `mise run build` sets it; to configure by hand, run\n"
    "  export LLMP_SDK=$(tools/setup-toolchain --print-root)\n"
    "  $LLMP_SDK/bin/cmake --preset <name>")
endif()
set(LLMP_SDK "${LLMP_SDK}" CACHE PATH "Root of the pinned llmpalooza SDK (tools/setup-toolchain --print-root)")

set(_llmp_receipt_file "${LLMP_SDK}/sdk.json")
if(NOT EXISTS "${_llmp_receipt_file}")
  message(FATAL_ERROR "No SDK at ${LLMP_SDK} (no sdk.json). Run `mise run setup`.")
endif()
file(READ "${_llmp_receipt_file}" _llmp_receipt)
string(JSON _llmp_schema ERROR_VARIABLE _llmp_error GET "${_llmp_receipt}" schema)
string(JSON LLMP_SDK_IDENTITY ERROR_VARIABLE _llmp_error GET "${_llmp_receipt}" identity)
string(JSON _llmp_sdk_arch ERROR_VARIABLE _llmp_error GET "${_llmp_receipt}" host_arch)
string(JSON _llmp_inputs ERROR_VARIABLE _llmp_error LENGTH "${_llmp_receipt}" inputs)
if(_llmp_error OR NOT _llmp_schema EQUAL 1 OR NOT _llmp_inputs GREATER 0)
  message(FATAL_ERROR "${_llmp_receipt_file} is not a valid SDK receipt; remove the SDK and run `mise run setup`.")
endif()
if(NOT CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL LLMP_BUILD_ARCH)
  message(FATAL_ERROR
    "The ${LLMP_PROFILE} profile builds on ${LLMP_BUILD_ARCH} hosts; this host is "
    "${CMAKE_HOST_SYSTEM_PROCESSOR}. Choose another preset (CMakePresets.json).")
endif()
if(NOT _llmp_sdk_arch STREQUAL CMAKE_HOST_SYSTEM_PROCESSOR)
  message(FATAL_ERROR "${LLMP_SDK} is an SDK for ${_llmp_sdk_arch} hosts, not ${CMAKE_HOST_SYSTEM_PROCESSOR}.")
endif()

# The receipt lists the files whose bytes determined the SDK. Each must match
# this checkout, or the SDK was built from other pins.
set(LLMP_SDK_INPUT_FILES "")
math(EXPR _llmp_last "${_llmp_inputs} - 1")
foreach(_llmp_i RANGE ${_llmp_last})
  string(JSON _llmp_name MEMBER "${_llmp_receipt}" inputs ${_llmp_i})
  string(JSON _llmp_want GET "${_llmp_receipt}" inputs "${_llmp_name}")
  set(_llmp_file "${LLMP_SOURCE_ROOT}/${_llmp_name}")
  if(EXISTS "${_llmp_file}")
    file(SHA256 "${_llmp_file}" _llmp_have)
  else()
    set(_llmp_have "missing")
  endif()
  if(NOT _llmp_have STREQUAL _llmp_want)
    message(FATAL_ERROR
      "The SDK at ${LLMP_SDK} was set up from a different ${_llmp_name} than this "
      "checkout has. Run `mise run setup`, then configure afresh with the new SDK: "
      "`mise run build -- --fresh <preset>`.")
  endif()
  list(APPEND LLMP_SDK_INPUT_FILES "${_llmp_file}")
endforeach()

# Compilers and tools -----------------------------------------------------------

set(CMAKE_C_COMPILER "${LLMP_SDK}/bin/clang")
set(CMAKE_CXX_COMPILER "${LLMP_SDK}/bin/clang++")
set(CMAKE_C_COMPILER_TARGET "${LLMP_TARGET_TRIPLE}")
set(CMAKE_CXX_COMPILER_TARGET "${LLMP_TARGET_TRIPLE}")
set(CMAKE_MAKE_PROGRAM "${LLMP_SDK}/bin/ninja" CACHE FILEPATH "The SDK's Ninja (D-059)" FORCE)
foreach(_llmp_tool IN ITEMS AR RANLIB NM OBJCOPY OBJDUMP READELF STRIP)
  string(TOLOWER "${_llmp_tool}" _llmp_name)
  set(CMAKE_${_llmp_tool} "${LLMP_SDK}/bin/llvm-${_llmp_name}" CACHE FILEPATH "" FORCE)
endforeach()

# The GCC 16.2 C++ runtime (D-060): its headers for every compile, and its
# static libstdc++ and libgcc for every link. A cross build finds it inside the
# sysroot, the only place Clang searches a GCC install's libraries.
if(DEFINED LLMP_SYSROOT)
  set(CMAKE_SYSROOT "${LLMP_SDK}/${LLMP_SYSROOT}")
  set(_llmp_gcc_root "${CMAKE_SYSROOT}/opt/gcc")
  set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
  set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
  set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
  set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
else()
  set(_llmp_gcc_root "${LLMP_SDK}/gcc/${LLMP_TARGET_TRIPLE}")
endif()
set(LLMP_GCC_INSTALL_DIR "${_llmp_gcc_root}/lib/gcc/${LLMP_TARGET_TRIPLE}/16")
foreach(_llmp_path IN ITEMS "${CMAKE_CXX_COMPILER}" "${CMAKE_MAKE_PROGRAM}" "${LLMP_GCC_INSTALL_DIR}/libgcc.a")
  if(NOT EXISTS "${_llmp_path}")
    message(FATAL_ERROR "${_llmp_path} is missing from the SDK; remove ${LLMP_SDK} and run `mise run setup`.")
  endif()
endforeach()

# Quotes a value for a POSIX shell, including paths in compiler flag strings.
function(_llmp_sh_quote out value)
  string(REPLACE "'" "'\\''" value "${value}")
  set(${out} "'${value}'" PARENT_SCOPE)
endfunction()

_llmp_sh_quote(_llmp_gcc_flag "--gcc-install-dir=${LLMP_GCC_INSTALL_DIR}")
set(_llmp_target_flags "-march=${LLMP_TARGET_MARCH} ${_llmp_gcc_flag}")
set(CMAKE_C_FLAGS_INIT "${_llmp_target_flags}")
set(CMAKE_CXX_FLAGS_INIT "${_llmp_target_flags}")
# The SDK's GCC runtime has only static archives, so every link, CMake's own
# checks included, takes the runtime statically (D-060).
list(APPEND LLMP_LINK_FLAGS -static-libstdc++ -static-libgcc)
list(JOIN LLMP_LINK_FLAGS " " CMAKE_EXE_LINKER_FLAGS_INIT)

# CUDA ----------------------------------------------------------------------------

if(NOT DEFINED LLMP_CUDA)
  set(LLMP_CUDA ON CACHE BOOL "Build CUDA code for sm_121 (and the profile's discrete GPUs, D-082). OFF is the CPU-only profile, which uses no CUDA SDK (D-026).")
endif()
if(NOT LLMP_TOOLCHAIN_DIR)
  set(LLMP_TOOLCHAIN_DIR "${CMAKE_BINARY_DIR}/toolchain")
endif()

if(LLMP_CUDA)
  if(LLMP_TARGET_TRIPLE STREQUAL "x86_64-linux-gnu")
    set(LLMP_CUDA_TARGET_DIR x86_64-linux)
  else()
    set(LLMP_CUDA_TARGET_DIR sbsa-linux)
  endif()
  set(CMAKE_CUDA_COMPILER "${LLMP_SDK}/cuda/bin/nvcc")
  if(NOT EXISTS "${CMAKE_CUDA_COMPILER}" OR NOT EXISTS "${LLMP_SDK}/cuda/targets/${LLMP_CUDA_TARGET_DIR}")
    message(FATAL_ERROR "The SDK lacks CUDA for ${LLMP_CUDA_TARGET_DIR}; remove ${LLMP_SDK} and run `mise run setup`.")
  endif()
  set(CMAKE_CUDA_FLAGS_INIT "--target-directory ${LLMP_CUDA_TARGET_DIR}")
  # The GB10, and the discrete GPUs the profile names (D-082), as SASS: no
  # PTX to JIT (D-011, D-032). Each is named explicitly, never detected.
  # Spark builds stay GB10-only. LLMP_CUDA_DISCRETE_SASS is the discrete
  # part, which the kernel modules' own architecture lists add to theirs.
  set(LLMP_CUDA_DISCRETE_SASS "")
  if(LLMP_CUDA_DISCRETE_ARCHITECTURES AND NOT LLMP_TARGET_TRIPLE STREQUAL "x86_64-linux-gnu")
    message(FATAL_ERROR "The ${LLMP_PROFILE} profile targets the GB10 only; discrete GPUs "
                        "(LLMP_CUDA_DISCRETE_ARCHITECTURES) are an x86-64 target (D-082).")
  endif()
  foreach(_llmp_arch IN LISTS LLMP_CUDA_DISCRETE_ARCHITECTURES)
    if(NOT _llmp_arch MATCHES "^[1-9][0-9]+$" OR _llmp_arch STREQUAL "121")
      message(FATAL_ERROR "LLMP_CUDA_DISCRETE_ARCHITECTURES: '${_llmp_arch}' is not a "
                          "discrete GPU's compute capability, such as 86 (D-082)")
    endif()
    list(APPEND LLMP_CUDA_DISCRETE_SASS "${_llmp_arch}-real")
  endforeach()
  list(REMOVE_DUPLICATES LLMP_CUDA_DISCRETE_SASS)
  # One list for llmpalooza's code, the kernel modules, the probe that `llmp
  # doctor` reports and the test labels, so CMAKE_CUDA_ARCHITECTURES is not
  # a separate knob (try_compile projects pass on the same list).
  set(_llmp_cuda_architectures 121-real ${LLMP_CUDA_DISCRETE_SASS})
  if(DEFINED CMAKE_CUDA_ARCHITECTURES
     AND NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "${_llmp_cuda_architectures}")
    message(FATAL_ERROR "CMAKE_CUDA_ARCHITECTURES is '${CMAKE_CUDA_ARCHITECTURES}', but this "
                        "profile builds '${_llmp_cuda_architectures}': name discrete GPUs with "
                        "LLMP_CUDA_DISCRETE_ARCHITECTURES instead (D-082).")
  endif()
  set(CMAKE_CUDA_ARCHITECTURES ${_llmp_cuda_architectures})

  # NVCC takes a host compiler path but no arguments for it, and runs it for
  # preprocessing, compiling and CMake's link checks. This wrapper adds the
  # target flags, and the link flags only when linking, so that compile-only
  # passes see no unused arguments (the validated D-032/D-060 arrangement).
  set(_llmp_compile "")
  foreach(_llmp_arg IN ITEMS "${CMAKE_CXX_COMPILER}" "--target=${LLMP_TARGET_TRIPLE}"
                                "-march=${LLMP_TARGET_MARCH}" "--gcc-install-dir=${LLMP_GCC_INSTALL_DIR}")
    _llmp_sh_quote(_llmp_quoted "${_llmp_arg}")
    string(APPEND _llmp_compile " ${_llmp_quoted}")
  endforeach()
  if(CMAKE_SYSROOT)
    _llmp_sh_quote(_llmp_quoted "--sysroot=${CMAKE_SYSROOT}")
    string(APPEND _llmp_compile " ${_llmp_quoted}")
  endif()
  set(_llmp_link "")
  foreach(_llmp_arg IN LISTS LLMP_LINK_FLAGS)
    _llmp_sh_quote(_llmp_quoted "${_llmp_arg}")
    string(APPEND _llmp_link " ${_llmp_quoted}")
  endforeach()
  set(CMAKE_CUDA_HOST_COMPILER "${LLMP_TOOLCHAIN_DIR}/cuda-host-clang++")
  file(CONFIGURE OUTPUT "${CMAKE_CUDA_HOST_COMPILER}" @ONLY CONTENT [=[
#!/bin/sh
# Generated by cmake/toolchains/sdk.cmake for the @LLMP_PROFILE@ profile:
# NVCC's host compiler, the SDK's Clang with the target flags.
link=yes
for arg in "$@"; do
  case "$arg" in -c|-E|-S|-M|-MM|-fsyntax-only) link= ;; esac
done
if [ -n "$link" ]; then
  set --@_llmp_link@ "$@"
fi
exec@_llmp_compile@ "$@"
]=])
  file(CHMOD "${CMAKE_CUDA_HOST_COMPILER}" FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
       GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
endif()
