# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0

# Third-party source components from the source lock (D-017, D-057).
#
#   llmp_sources_add([LOCK <file>])  selects the profile's closure from the
#       lock, checks every selected component's prepared tree against the
#       lock, then adds each as a SYSTEM, EXCLUDE_FROM_ALL subproject.
#   llmp_sources_finalize([VERSION <json>])  after everything else: rejects
#       undeclared package lookups and FetchContent use, then writes the build
#       receipt, llmp-receipt.json in the build directory, with VERSION as
#       its `version` object (D-062) if given.
#
# Configure never fetches: tools/prepare-sources (`mise run prepare`) fetches,
# verifies and unpacks the sources into LLMP_SOURCES_DIR, and a missing or
# modified tree stops configure before any third-party CMake code runs. The
# build checks each tree again before compiling from it.
#
# Cache variables:
#   LLMP_MODULES                    optional modules to enable (default none:
#                                     the copyleft-disabled core profile, D-002)
#   LLMP_SOURCES_DIR                prepared trees (default build/sources)
#   LLMP_SOURCE_OVERRIDE_<ID>       a local development tree for component
#                                     <id> (upper case, `-` as `_`); recorded in
#                                     the receipt as modified and unofficial
#   LLMP_REQUIRE_LOCKED_SOURCES     reject overrides (check and release builds)

include_guard(GLOBAL)
include("${CMAKE_CURRENT_LIST_DIR}/sources/TreeDigest.cmake")

set(_LLMP_SOURCES_SCRIPTS "${CMAKE_CURRENT_LIST_DIR}/sources")

set(LLMP_MODULES "" CACHE STRING "Optional modules to build (D-002, D-017); empty is the core profile")
set(LLMP_SOURCES_DIR "${PROJECT_SOURCE_DIR}/build/sources" CACHE PATH
    "Prepared source trees (tools/prepare-sources, `mise run prepare`)")
option(LLMP_REQUIRE_LOCKED_SOURCES "Reject LLMP_SOURCE_OVERRIDE_<ID> trees (check and release builds)" OFF)

# Quotes a value as a JSON string.
function(_llmp_json_string out value)
  string(REPLACE "\\" "\\\\" value "${value}")
  string(REPLACE "\"" "\\\"" value "${value}")
  string(REPLACE "\n" "\\n" value "${value}")
  string(REPLACE "\t" "\\t" value "${value}")
  string(REPLACE "\r" "\\r" value "${value}")
  set(${out} "\"${value}\"" PARENT_SCOPE)
endfunction()

# Escapes glob metacharacters in a path, so file(GLOB) matches it literally.
function(_llmp_glob_escape out path)
  string(REGEX REPLACE "([][*?])" "[\\1]" path "${path}")
  set(${out} "${path}" PARENT_SCOPE)
endfunction()

# A JSON array of strings from a CMake list.
function(_llmp_json_array out)
  set(items "")
  foreach(value IN LISTS ARGN)
    _llmp_json_string(quoted "${value}")
    list(APPEND items "${quoted}")
  endforeach()
  list(JOIN items ", " joined)
  set(${out} "[${joined}]" PARENT_SCOPE)
endfunction()

# Reads a JSON value from the lock or stops configure. Sets <out> to it and,
# with TYPE, checks the value's JSON type.
function(_llmp_lock_get out)
  cmake_parse_arguments(PARSE_ARGV 1 arg "" "TYPE" "PATH")
  string(JSON value ERROR_VARIABLE error GET "${_llmp_lock_json}" ${arg_PATH})
  if(error STREQUAL "NOTFOUND" AND arg_TYPE)
    string(JSON type ERROR_VARIABLE error TYPE "${_llmp_lock_json}" ${arg_PATH})
    if(NOT type STREQUAL arg_TYPE)
      set(error "${arg_PATH} is ${type}, not ${arg_TYPE}")
    endif()
  endif()
  if(NOT error STREQUAL "NOTFOUND")
    list(JOIN arg_PATH "." where)
    message(FATAL_ERROR "${_llmp_lock_file}: ${where}: ${error}. Run `mise run prepare -- --check`.")
  endif()
  set(${out} "${value}" PARENT_SCOPE)
endfunction()

# The members of a lock object, as a list.
function(_llmp_lock_members out)
  string(JSON count ERROR_VARIABLE error LENGTH "${_llmp_lock_json}" ${ARGN})
  set(members "")
  if(error STREQUAL "NOTFOUND" AND count GREATER 0)
    math(EXPR last "${count} - 1")
    foreach(i RANGE ${last})
      string(JSON member MEMBER "${_llmp_lock_json}" ${ARGN} ${i})
      list(APPEND members "${member}")
    endforeach()
  endif()
  set(${out} "${members}" PARENT_SCOPE)
endfunction()

# The elements of a lock array of strings, as a list.
function(_llmp_lock_list out)
  _llmp_lock_get(array PATH ${ARGN} TYPE ARRAY)
  string(JSON count LENGTH "${array}")
  set(values "")
  if(count GREATER 0)
    math(EXPR last "${count} - 1")
    foreach(i RANGE ${last})
      string(JSON value GET "${array}" ${i})
      list(APPEND values "${value}")
    endforeach()
  endif()
  set(${out} "${values}" PARENT_SCOPE)
endfunction()

# Every buildsystem target defined in a directory and its subdirectories.
function(_llmp_directory_targets out dir)
  get_property(targets DIRECTORY "${dir}" PROPERTY BUILDSYSTEM_TARGETS)
  get_property(subdirs DIRECTORY "${dir}" PROPERTY SUBDIRECTORIES)
  foreach(subdir IN LISTS subdirs)
    _llmp_directory_targets(more "${subdir}")
    list(APPEND targets ${more})
  endforeach()
  set(${out} "${targets}" PARENT_SCOPE)
endfunction()

# Adds one checked component. A function of its own, so its CMake options
# are visible to its subproject (CMP0077) and to nothing else. Everything it
# uses after setting them has a `_llmp` name, which the lock's option names
# can never take (tools/llmp_sources.py rejects them, and so does this).
function(_llmp_add_component _llmp_id)
  set(_llmp_source "${_llmp_source_${_llmp_id}}")
  set(_llmp_binary "${_llmp_binary_${_llmp_id}}")
  set(_llmp_digest "${_llmp_digest_${_llmp_id}}")
  set(_llmp_scripts "${_LLMP_SOURCES_SCRIPTS}")
  set(_llmp_python "${LLMP_PYTHON}")
  set(_llmp_cmake "${CMAKE_COMMAND}")
  _llmp_lock_get(_llmp_subdirectory PATH components ${_llmp_id} cmake subdirectory TYPE STRING)
  cmake_path(APPEND _llmp_source "${_llmp_subdirectory}" OUTPUT_VARIABLE _llmp_project)
  _llmp_lock_list(_llmp_declared components ${_llmp_id} cmake targets)

  # Every lookup a subproject makes must fail rather than find something
  # outside the closure: no FetchContent population, no source overrides and
  # no package registries (llmp_sources_finalize checks what was used).
  # Subprojects that declare an old minimum version get the current
  # FetchContent and option policies anyway, so a declared download still
  # fails and the locked options still hold.
  set(FETCHCONTENT_FULLY_DISCONNECTED ON)
  set(FETCHCONTENT_UPDATES_DISCONNECTED ON)
  set(FETCHCONTENT_TRY_FIND_PACKAGE_MODE NEVER)
  set(FETCHCONTENT_BASE_DIR "${_llmp_fetchcontent_tripwire}")
  set(CMAKE_FIND_USE_PACKAGE_REGISTRY OFF)
  set(CMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY OFF)
  set(CMAKE_EXPORT_NO_PACKAGE_REGISTRY ON)
  foreach(_llmp_policy IN ITEMS CMP0077 CMP0126 CMP0168 CMP0169 CMP0170)
    set(CMAKE_POLICY_DEFAULT_${_llmp_policy} NEW)
  endforeach()

  _llmp_lock_members(_llmp_options components ${_llmp_id} cmake options)
  foreach(_llmp_option IN LISTS _llmp_options)
    if(NOT _llmp_option MATCHES "^[A-Za-z][A-Za-z0-9_]*$"
       OR _llmp_option MATCHES "^([Cc][Mm][Aa][Kk][Ee]|[Jj][Ii][Tt][Ll][Ll][Mm]|[Ff][Ee][Tt][Cc][Hh][Cc][Oo][Nn][Tt][Ee][Nn][Tt])_"
       OR _llmp_option MATCHES "^[Bb][Uu][Ii][Ll][Dd]_[Ss][Hh][Aa][Rr][Ee][Dd]_[Ll][Ii][Bb][Ss]$")
      message(FATAL_ERROR "${_llmp_id}: the lock may not set ${_llmp_option}")
    endif()
    _llmp_lock_get(_llmp_value PATH components ${_llmp_id} cmake options ${_llmp_option} TYPE STRING)
    set(${_llmp_option} "${_llmp_value}")
  endforeach()

  add_subdirectory("${_llmp_project}" "${_llmp_binary}" EXCLUDE_FROM_ALL SYSTEM)

  foreach(_llmp_target IN LISTS _llmp_declared)
    if(NOT TARGET ${_llmp_target})
      message(FATAL_ERROR "${_llmp_id} defines no target ${_llmp_target}, which the lock declares")
    endif()
  endforeach()

  # Check on every build: timestamps cannot detect added files, mode changes
  # or edits that preserve mtime. Dependencies order this check before the
  # component and its consumers without forcing their objects to rebuild.
  set(_llmp_stamp "${_llmp_binary}/llmp-source-verified.stamp")
  add_custom_target(llmp_source_verify_${_llmp_id}
    COMMAND "${_llmp_cmake}" "-DLLMP_PYTHON=${_llmp_python}"
            "-DDIR=${_llmp_source}" "-DEXPECT=${_llmp_digest}" "-DSTAMP=${_llmp_stamp}"
            -P "${_llmp_scripts}/verify.cmake"
    BYPRODUCTS "${_llmp_stamp}"
    COMMENT "Checking the prepared ${_llmp_id} source against the lock"
    VERBATIM)

  # Keep every target's outputs inside the component's binary directory,
  # which configure deletes once the component leaves the closure, and end
  # every compile with -fno-exceptions (D-066) after the component's own
  # target flags. tests/sources/check_closure.py checks the result.
  _llmp_directory_targets(_llmp_targets "${_llmp_project}")
  foreach(_llmp_target IN LISTS _llmp_targets)
    get_target_property(_llmp_type ${_llmp_target} TYPE)
    if(_llmp_type MATCHES "^(STATIC|SHARED|MODULE|OBJECT)_LIBRARY$|^EXECUTABLE$")
      set_target_properties(${_llmp_target} PROPERTIES
        ARCHIVE_OUTPUT_DIRECTORY "${_llmp_binary}/lib"
        LIBRARY_OUTPUT_DIRECTORY "${_llmp_binary}/lib"
        RUNTIME_OUTPUT_DIRECTORY "${_llmp_binary}/bin"
        PDB_OUTPUT_DIRECTORY "${_llmp_binary}/bin"
        COMPILE_PDB_OUTPUT_DIRECTORY "${_llmp_binary}/lib")
      # SHELL: keeps CMake from folding this into the directory's identical
      # flag, which would leave it before the component's own options.
      target_compile_options(${_llmp_target} PRIVATE
        "$<$<COMPILE_LANGUAGE:CXX>:SHELL:-fno-exceptions>"
        "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler=-fno-exceptions>")
    endif()
    add_dependencies(${_llmp_target} llmp_source_verify_${_llmp_id})
  endforeach()
endfunction()

function(llmp_sources_add)
  cmake_parse_arguments(PARSE_ARGV 0 arg "" "LOCK" "")
  if(NOT arg_LOCK)
    set(arg_LOCK "${PROJECT_SOURCE_DIR}/third_party/sources.lock.json")
  endif()
  set(_llmp_lock_file "${arg_LOCK}")
  set(receipt "${CMAKE_BINARY_DIR}/llmp-receipt.json")
  # A receipt describes a completed configure; remove the old one first.
  file(REMOVE "${receipt}")
  set_property(DIRECTORY "${CMAKE_SOURCE_DIR}" APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
               "${arg_LOCK}" "${_LLMP_SOURCE_INSPECTOR}"
               "${_LLMP_SOURCES_SCRIPTS}/../../tools/llmp_sources.py")

  # Unrecorded ways to swap in other source or packages (D-057).
  get_cmake_property(variables VARIABLES)
  get_cmake_property(cache_variables CACHE_VARIABLES)
  list(APPEND variables ${cache_variables})
  list(REMOVE_DUPLICATES variables)
  foreach(variable IN LISTS variables)
    if(variable MATCHES "^FETCHCONTENT_SOURCE_DIR_")
      message(FATAL_ERROR "${variable} would replace a locked source without a record; "
                          "use LLMP_SOURCE_OVERRIDE_<ID> for a local development tree (D-057)")
    endif()
    if(variable MATCHES "^CMAKE_PROJECT_(.*_)?INCLUDE(_BEFORE)?$|^CMAKE_PROJECT_TOP_LEVEL_INCLUDES$"
       AND NOT "${${variable}}" STREQUAL "")
      message(FATAL_ERROR "${variable} injects CMake code or a dependency provider; "
                          "llmpalooza builds take dependencies only from the source lock (D-057)")
    endif()
  endforeach()

  # Use preparation's complete schema/admission checks even when prepared
  # trees already exist. Editing the lock must not bypass them at configure.
  list(JOIN LLMP_MODULES "," validation_modules)
  if(LLMP_CUDA)
    set(validation_cuda on)
  else()
    set(validation_cuda off)
  endif()
  execute_process(COMMAND "${LLMP_PYTHON}" "${_LLMP_SOURCE_INSPECTOR}"
                          --lock "${arg_LOCK}" --modules "${validation_modules}" --cuda "${validation_cuda}"
                  RESULT_VARIABLE validation_result ERROR_VARIABLE validation_error)
  if(NOT validation_result STREQUAL "0")
    message(FATAL_ERROR "${validation_error}")
  endif()

  file(READ "${arg_LOCK}" _llmp_lock_json)
  file(SHA256 "${arg_LOCK}" lock_sha256)
  _llmp_lock_get(schema PATH schema TYPE NUMBER)
  if(NOT schema EQUAL 1)
    message(FATAL_ERROR "${arg_LOCK}: schema ${schema} is not 1")
  endif()
  _llmp_lock_members(modules modules)
  _llmp_lock_members(components components)

  set(enabled ${LLMP_MODULES})
  list(REMOVE_DUPLICATES enabled)
  list(SORT enabled)
  foreach(module IN LISTS enabled)
    if(NOT module IN_LIST modules)
      message(FATAL_ERROR "LLMP_MODULES names ${module}, which ${arg_LOCK} does not declare (it has: ${modules})")
    endif()
  endforeach()
  list(PREPEND enabled core)
  list(JOIN enabled "+" license_profile)
  list(POP_FRONT enabled)

  # Select the closure and check each component's classification.
  set(selected "")
  foreach(id IN LISTS components)
    if(NOT id MATCHES "^[a-z][a-z0-9-]*$")
      message(FATAL_ERROR "${arg_LOCK}: component id ${id} is not valid")
    endif()
    _llmp_lock_get(kind PATH components ${id} kind TYPE STRING)
    _llmp_lock_get(machine PATH components ${id} machine TYPE STRING)
    _llmp_lock_get(tier PATH components ${id} tier TYPE STRING)
    if(NOT kind STREQUAL "archive" OR NOT machine STREQUAL "target")
      message(FATAL_ERROR "${id}: only archive components for the target machine are supported (kind ${kind}, machine ${machine})")
    endif()
    string(JSON requires_cuda ERROR_VARIABLE condition_error GET "${_llmp_lock_json}"
                components ${id} requires_cuda)
    # The shared validator already rejects nonboolean conditions. Legacy
    # locks omit the condition and select the component in both builds.
    if(condition_error)
      set(requires_cuda FALSE)
    endif()
    if(requires_cuda AND NOT LLMP_CUDA)
      continue()
    endif()
    if(tier STREQUAL "core")
      list(APPEND selected ${id})
    elseif(tier STREQUAL "optional")
      _llmp_lock_get(module PATH components ${id} module TYPE STRING)
      if(module IN_LIST enabled)
        list(APPEND selected ${id})
      endif()
    else()
      message(FATAL_ERROR "${id}: tier ${tier} is neither core nor optional")
    endif()
  endforeach()

  # Dependency order; every dependency must be in the closure.
  set(ordered "")
  set(pending ${selected})
  while(NOT "${pending}" STREQUAL "")
    set(progress FALSE)
    foreach(id IN LISTS pending)
      _llmp_lock_list(depends components ${id} depends)
      set(ready TRUE)
      foreach(dep IN LISTS depends)
        if(NOT dep IN_LIST selected)
          message(FATAL_ERROR "${id} depends on ${dep}, which the ${license_profile} profile does not select")
        endif()
        if(NOT dep IN_LIST ordered)
          set(ready FALSE)
        endif()
      endforeach()
      if(ready)
        list(APPEND ordered ${id})
        list(REMOVE_ITEM pending ${id})
        set(progress TRUE)
      endif()
    endforeach()
    if(NOT progress)
      message(FATAL_ERROR "${arg_LOCK}: dependency cycle among ${pending}")
    endif()
  endwhile()

  # Check every selected tree before any third-party CMake code runs.
  set(missing "")
  set(official TRUE)
  set(overrides "")
  foreach(id IN LISTS components)
    string(TOUPPER "${id}" upper)
    string(REPLACE "-" "_" upper "${upper}")
    list(APPEND overrides "LLMP_SOURCE_OVERRIDE_${upper}")
  endforeach()
  foreach(variable IN LISTS variables)
    if(variable MATCHES "^LLMP_SOURCE_OVERRIDE_" AND NOT variable IN_LIST overrides
       AND NOT "${${variable}}" STREQUAL "")
      message(FATAL_ERROR "${variable} names no component in ${arg_LOCK}")
    endif()
  endforeach()
  foreach(id IN LISTS ordered)
    _llmp_lock_get(patches PATH components ${id} patches TYPE ARRAY)
    string(JSON patch_count LENGTH "${patches}")
    if(patch_count GREATER 0)
      math(EXPR last_patch "${patch_count} - 1")
      cmake_path(GET arg_LOCK PARENT_PATH lock_dir)
      foreach(i RANGE ${last_patch})
        string(JSON patch_path GET "${patches}" ${i} path)
        set_property(DIRECTORY "${CMAKE_SOURCE_DIR}" APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
                     "${lock_dir}/${patch_path}")
      endforeach()
    endif()
    _llmp_lock_get(tree PATH components ${id} tree_sha256 TYPE STRING)
    if(NOT tree MATCHES "^[0-9a-f]+$")
      message(FATAL_ERROR "${id}: tree_sha256 ${tree} is not a digest")
    endif()
    string(TOUPPER "${id}" upper)
    string(REPLACE "-" "_" upper "${upper}")
    set(override "${LLMP_SOURCE_OVERRIDE_${upper}}")
    set(_llmp_override_${id} FALSE)
    if(NOT override STREQUAL "")
      set(_llmp_override_${id} TRUE)
      if(LLMP_REQUIRE_LOCKED_SOURCES)
        message(FATAL_ERROR "LLMP_SOURCE_OVERRIDE_${upper} replaces the locked ${id} source, "
                            "which LLMP_REQUIRE_LOCKED_SOURCES forbids")
      endif()
      cmake_path(ABSOLUTE_PATH override NORMALIZE)
      set(source "${override}")
      set(official FALSE)
      message(WARNING "${id} builds from ${source}, a local override; this build's receipt is unofficial")
    else()
      string(SUBSTRING "${tree}" 0 16 short)
      set(source "${LLMP_SOURCES_DIR}/${id}-${short}")
      if(NOT IS_DIRECTORY "${source}")
        list(APPEND missing ${id})
        continue()
      endif()
    endif()
    llmp_tree_digest(digest "${source}")
    if(digest MATCHES "^invalid: ")
      message(FATAL_ERROR "${id}: ${digest}")
    endif()
    if(_llmp_override_${id})
      set(modified FALSE)
      if(NOT digest STREQUAL tree)
        set(modified TRUE)
      endif()
    elseif(NOT digest STREQUAL tree)
      message(FATAL_ERROR
        "The prepared ${id} source at ${source} no longer matches the lock (tree ${digest}, locked ${tree}). "
        "Remove that directory and run `mise run prepare`; for deliberate local edits, set "
        "LLMP_SOURCE_OVERRIDE_${upper} to a copy instead (D-057).")
    else()
      set(modified FALSE)
    endif()
    string(SUBSTRING "${digest}" 0 16 short)
    set(_llmp_source_${id} "${source}")
    set(_llmp_binary_${id} "${CMAKE_BINARY_DIR}/third_party/${id}-${short}")
    set(_llmp_digest_${id} "${digest}")
    set(_llmp_modified_${id} ${modified})
  endforeach()
  if(missing)
    list(JOIN missing ", " names)
    set(command "mise run prepare")
    if(enabled)
      list(JOIN enabled "," joined)
      string(APPEND command " -- --modules ${joined}")
    endif()
    message(FATAL_ERROR "The ${license_profile} profile needs sources that are not prepared in "
                        "${LLMP_SOURCES_DIR}: ${names}. Run `${command}` (configure never downloads, D-057).")
  endif()

  # A build directory that has built an optional module never builds a
  # profile without it: that module's generated or linked payloads could be
  # anywhere in the tree, so only a new directory proves their absence (D-002,
  # D-017). Every configure that adds components keeps this record, which
  # survives `cmake --fresh`; a directory with component outputs but no
  # record has an unknown history and is refused the same way.
  set(built_record "${CMAKE_BINARY_DIR}/llmp-modules-built.txt")
  set(built "")
  if(EXISTS "${built_record}")
    file(STRINGS "${built_record}" built REGEX "^[^#]")
  elseif(EXISTS "${CMAKE_BINARY_DIR}/third_party")
    message(FATAL_ERROR "${CMAKE_BINARY_DIR} has third-party build outputs but no ${built_record}, so the "
                        "optional modules it has built are unknown. Build in a new directory (remove this one) "
                        "(D-002, D-057).")
  endif()
  set(dropped ${built})
  if(enabled)
    list(REMOVE_ITEM dropped ${enabled})
  endif()
  if(dropped)
    list(JOIN dropped ", " names)
    message(FATAL_ERROR "${CMAKE_BINARY_DIR} has built optional module(s) ${names}, which the "
                        "${license_profile} profile leaves out. Build it in a new directory (remove this one) "
                        "so none of their payloads can be reused (D-002, D-057).")
  endif()
  list(APPEND built ${enabled})
  list(REMOVE_DUPLICATES built)
  list(SORT built)
  list(JOIN built "\n" built_text)
  file(WRITE "${built_record}"
       "# Optional modules this build directory has built; it never builds a profile without them.\n"
       "# Written by cmake/LlmpSources.cmake (D-057). Removing it makes configure refuse the directory.\n"
       "${built_text}\n")

  # Outputs of older trees of the selected components go.
  set(keep "")
  foreach(id IN LISTS ordered)
    cmake_path(GET _llmp_binary_${id} FILENAME name)
    list(APPEND keep "${name}")
  endforeach()
  _llmp_glob_escape(third_party_glob "${CMAKE_BINARY_DIR}/third_party")
  file(GLOB previous LIST_DIRECTORIES true "${third_party_glob}/*")
  foreach(path IN LISTS previous)
    cmake_path(GET path FILENAME name)
    if(NOT name IN_LIST keep)
      message(STATUS "Removing ${path}, which this configuration does not use")
      file(REMOVE_RECURSE "${path}")
    endif()
  endforeach()
  set(_llmp_fetchcontent_tripwire "${CMAKE_BINARY_DIR}/third_party/.fetchcontent-denied")

  # The receipt's component records, one JSON array (paths may hold `;`, so
  # never a CMake list).
  set(entries "[]")
  set(index 0)
  set(allowed_packages "")
  foreach(id IN LISTS ordered)
    _llmp_add_component(${id})
    set(entry "{}")
    foreach(field IN ITEMS id version tier use license archive_sha256 locked_tree source_tree source)
      if(field STREQUAL "id")
        set(value "${id}")
      elseif(field STREQUAL "license")
        _llmp_lock_get(value PATH components ${id} license expression TYPE STRING)
      elseif(field STREQUAL "archive_sha256")
        _llmp_lock_get(value PATH components ${id} archive sha256 TYPE STRING)
      elseif(field STREQUAL "locked_tree")
        _llmp_lock_get(value PATH components ${id} tree_sha256 TYPE STRING)
      elseif(field STREQUAL "source_tree")
        set(value "${_llmp_digest_${id}}")
      elseif(field STREQUAL "source")
        set(value "${_llmp_source_${id}}")
      else()
        _llmp_lock_get(value PATH components ${id} ${field} TYPE STRING)
      endif()
      _llmp_json_string(value "${value}")
      string(JSON entry SET "${entry}" ${field} "${value}")
    endforeach()
    foreach(flag IN ITEMS override modified)
      if(_llmp_${flag}_${id})
        string(JSON entry SET "${entry}" ${flag} true)
      else()
        string(JSON entry SET "${entry}" ${flag} false)
      endif()
    endforeach()
    _llmp_lock_get(patches PATH components ${id} patches TYPE ARRAY)
    _llmp_lock_get(options PATH components ${id} cmake options TYPE OBJECT)
    string(JSON entry SET "${entry}" patches "${patches}")
    string(JSON entry SET "${entry}" options "${options}")
    string(JSON entries SET "${entries}" ${index} "${entry}")
    math(EXPR index "${index} + 1")
    _llmp_lock_list(packages components ${id} cmake platform_packages)
    list(APPEND allowed_packages ${packages})
  endforeach()
  list(REMOVE_DUPLICATES allowed_packages)

  set_property(GLOBAL PROPERTY LLMP_SOURCES_ORDERED "${ordered}")
  set_property(GLOBAL PROPERTY LLMP_SOURCES_ENTRIES "${entries}")
  set_property(GLOBAL PROPERTY LLMP_SOURCES_PACKAGES "${allowed_packages}")
  set_property(GLOBAL PROPERTY LLMP_SOURCES_LOCK "${arg_LOCK}")
  set_property(GLOBAL PROPERTY LLMP_SOURCES_LOCK_SHA256 "${lock_sha256}")
  set_property(GLOBAL PROPERTY LLMP_SOURCES_PROFILE "${license_profile}")
  set_property(GLOBAL PROPERTY LLMP_SOURCES_MODULES "${enabled}")
  set_property(GLOBAL PROPERTY LLMP_SOURCES_OFFICIAL "${official}")
  set_property(GLOBAL PROPERTY LLMP_SOURCES_TRIPWIRE "${_llmp_fetchcontent_tripwire}")
  message(STATUS "llmpalooza sources: ${license_profile} profile, ${ordered}")
endfunction()

function(llmp_sources_finalize)
  cmake_parse_arguments(PARSE_ARGV 0 arg "" "VERSION" "")
  get_property(ordered GLOBAL PROPERTY LLMP_SOURCES_ORDERED)
  get_property(entries GLOBAL PROPERTY LLMP_SOURCES_ENTRIES)
  get_property(allowed GLOBAL PROPERTY LLMP_SOURCES_PACKAGES)
  get_property(lock_file GLOBAL PROPERTY LLMP_SOURCES_LOCK)
  get_property(lock_sha256 GLOBAL PROPERTY LLMP_SOURCES_LOCK_SHA256)
  get_property(profile GLOBAL PROPERTY LLMP_SOURCES_PROFILE)
  get_property(modules GLOBAL PROPERTY LLMP_SOURCES_MODULES)
  get_property(official GLOBAL PROPERTY LLMP_SOURCES_OFFICIAL)
  get_property(tripwire GLOBAL PROPERTY LLMP_SOURCES_TRIPWIRE)
  if(NOT profile)
    message(FATAL_ERROR "call llmp_sources_add() before llmp_sources_finalize()")
  endif()

  # Package lookups: only the platform packages the lock declares (D-017),
  # found or not, so no host's installed libraries can substitute.
  get_property(found GLOBAL PROPERTY PACKAGES_FOUND)
  get_property(not_found GLOBAL PROPERTY PACKAGES_NOT_FOUND)
  set(undeclared ${found} ${not_found})
  if(allowed)
    list(REMOVE_ITEM undeclared ${allowed})
  endif()
  if(undeclared)
    list(REMOVE_DUPLICATES undeclared)
    message(FATAL_ERROR "find_package() looked for ${undeclared}, which no selected component declares "
                        "in cmake.platform_packages; dependencies come from the source lock (D-017, D-057)")
  endif()
  if(EXISTS "${tripwire}")
    message(FATAL_ERROR "A dependency tried to populate sources with FetchContent (${tripwire}); "
                        "every source comes from the lock through `mise run prepare` (D-057)")
  endif()

  cmake_path(RELATIVE_PATH lock_file BASE_DIRECTORY "${PROJECT_SOURCE_DIR}" OUTPUT_VARIABLE lock_rel)
  _llmp_json_string(lock_rel "${lock_rel}")
  _llmp_json_array(modules_json ${modules})
  _llmp_json_array(found_json ${found})
  foreach(field IN ITEMS LLMP_PROFILE LLMP_TARGET_TRIPLE CMAKE_BUILD_TYPE LLMP_SDK_IDENTITY profile)
    _llmp_json_string(json_${field} "${${field}}")
  endforeach()
  if(official)
    set(official true)
  else()
    set(official false)
  endif()
  if(LLMP_CUDA)
    set(cuda true)
  else()
    set(cuda false)
  endif()
  set(version_line "")
  if(DEFINED arg_VERSION)
    set(version_line "  \"version\": ${arg_VERSION},")
  endif()
  string(JOIN "\n" receipt
    "{"
    "  \"schema\": 1,"
    "  \"profile\": ${json_LLMP_PROFILE},"
    "  \"target\": ${json_LLMP_TARGET_TRIPLE},"
    "  \"cuda\": ${cuda},"
    "  \"build_type\": ${json_CMAKE_BUILD_TYPE},"
    "  \"sdk\": ${json_LLMP_SDK_IDENTITY},"
    ${version_line}
    "  \"license_profile\": ${json_profile},"
    "  \"modules\": ${modules_json},"
    "  \"official\": ${official},"
    "  \"source_lock\": {\"path\": ${lock_rel}, \"sha256\": \"${lock_sha256}\"},"
    "  \"platform_packages\": ${found_json},"
    "  \"components\": ${entries}"
    "}"
    "")
  string(JSON ignored ERROR_VARIABLE error TYPE "${receipt}")
  if(NOT error STREQUAL "NOTFOUND")
    message(FATAL_ERROR "internal error: the build receipt is not valid JSON: ${error}")
  endif()
  file(WRITE "${CMAKE_BINARY_DIR}/llmp-receipt.json" "${receipt}")
endfunction()
