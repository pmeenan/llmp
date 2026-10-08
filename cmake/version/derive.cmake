# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0

# The product version of a checkout (D-062).
#
#   llmp_version_derive(<prefix> SOURCE_DIR <dir> PROJECT_VERSION <X.Y.Z> [GIT <git>])
#
# sets, in the caller's scope:
#   <prefix>_VERSION   SemVer 2.0.0: X.Y.Z for a clean checkout of the release
#                      tag vX.Y.Z; otherwise X.Y.Z-dev.N+g<sha> (N commits
#                      since the last release tag, or since the root with no
#                      tag), +g<sha>.dirty for a modified tree, and
#                      X.Y.Z-dev+unknown for a tree with no Git metadata
#   <prefix>_DEBIAN    the Debian package version: SemVer's `-` as `~`, so a
#                      dev build sorts before its release, and revision -1
#   <prefix>_COMMIT    the full commit ID, or empty with no Git metadata
#   <prefix>_DIRTY     TRUE when tracked files differ from the commit or
#                      untracked files Git does not ignore exist
#   <prefix>_ORIGIN    `git`, or `none` for a tree with no Git metadata
#   <prefix>_GIT       the git it ran, or empty with no Git metadata
#
#   llmp_version_json(<out> <prefix>)
#
# gives those values as the build receipt's `version` object.
#
# X.Y.Z is project(VERSION), the next release. A release tag is an annotated
# tag named vX.Y.Z, without leading zeros, on a commit HEAD contains; any
# other tag is ignored. With N > 0, X.Y.Z must be above the highest release
# tag, so no dev build sorts below a release; a clean release checkout must
# carry that tag's version. A modified release checkout may raise it before
# committing, so the next version can pass the pre-commit checks. Either
# failure, a shallow clone (whose N and tags are
# unknowable) or a Git error stops with FATAL_ERROR. A tree whose SOURCE_DIR
# has no .git (an exported copy, such as the reference build's) reports no
# commit; a .git that Git cannot read is an error, never a fallback.

include_guard(GLOBAL)

set(_LLMP_SEMVER_CORE "(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)")

# Runs git on the checkout alone: no repository, index, work tree or shallow
# file named in the environment, and no replacement objects or grafts (the
# environment's or the repository's), either of which would rewrite the
# history N counts.
set(_LLMP_GIT_ENV
  --unset=GIT_DIR --unset=GIT_WORK_TREE --unset=GIT_INDEX_FILE --unset=GIT_OBJECT_DIRECTORY
  --unset=GIT_ALTERNATE_OBJECT_DIRECTORIES --unset=GIT_COMMON_DIR --unset=GIT_NAMESPACE
  --unset=GIT_CEILING_DIRECTORIES --unset=GIT_DISCOVERY_ACROSS_FILESYSTEM --unset=GIT_SHALLOW_FILE
  GIT_NO_REPLACE_OBJECTS=1 GIT_GRAFT_FILE=/dev/null)

function(_llmp_git out)
  execute_process(COMMAND "${CMAKE_COMMAND}" -E env ${_LLMP_GIT_ENV}
                          "${_git}" -c advice.graftFileDeprecated=false --no-optional-locks
                          -C "${_source_dir}" ${ARGN}
                  OUTPUT_VARIABLE output ERROR_VARIABLE error RESULT_VARIABLE result
                  OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(NOT result STREQUAL "0")
    string(JOIN " " command ${ARGN})
    string(STRIP "${error}" error)
    message(FATAL_ERROR "The version comes from Git (D-062), but `git ${command}` failed in "
                        "${_source_dir} (${result}): ${error}")
  endif()
  set(${out} "${output}" PARENT_SCOPE)
endfunction()

function(llmp_version_derive prefix)
  cmake_parse_arguments(PARSE_ARGV 1 arg "" "SOURCE_DIR;PROJECT_VERSION;GIT" "")
  if(NOT arg_SOURCE_DIR OR NOT arg_PROJECT_VERSION)
    message(FATAL_ERROR "llmp_version_derive needs SOURCE_DIR and PROJECT_VERSION")
  endif()
  if(NOT arg_PROJECT_VERSION MATCHES "^${_LLMP_SEMVER_CORE}$")
    message(FATAL_ERROR "project(VERSION) is ${arg_PROJECT_VERSION}; the product version is "
                        "MAJOR.MINOR.PATCH without leading zeros (D-062)")
  endif()
  set(version "${arg_PROJECT_VERSION}")
  set(_source_dir "${arg_SOURCE_DIR}")

  if(NOT EXISTS "${_source_dir}/.git")
    set(${prefix}_VERSION "${version}-dev+unknown" PARENT_SCOPE)
    set(${prefix}_DEBIAN "${version}~dev+unknown-1" PARENT_SCOPE)
    set(${prefix}_COMMIT "" PARENT_SCOPE)
    set(${prefix}_DIRTY FALSE PARENT_SCOPE)
    set(${prefix}_ORIGIN none PARENT_SCOPE)
    set(${prefix}_GIT "" PARENT_SCOPE)
    return()
  endif()
  set(_git "${arg_GIT}")
  if(_git STREQUAL "")
    unset(_git)
    find_program(_git git NO_CACHE NO_CMAKE_FIND_ROOT_PATH)
    if(NOT _git)
      message(FATAL_ERROR "${_source_dir} is a Git checkout, and its version comes from Git (D-062), "
                          "but git is not installed (see toolchains/prerequisites/)")
    endif()
  endif()
  _llmp_git(shallow rev-parse --is-shallow-repository)
  if(shallow STREQUAL "true")
    message(FATAL_ERROR "${_source_dir} is a shallow clone, so its distance from the last release tag "
                        "is unknown (D-062). Fetch its full history: `git fetch --unshallow --tags`")
  endif()
  # HEAD is read once: a commit made while this runs cannot mix two commits'
  # IDs, distances or tags into one version.
  _llmp_git(commit rev-parse --verify "HEAD^{commit}")
  _llmp_git(short rev-parse --short=12 "${commit}")

  # The highest release tag HEAD contains. Lightweight tags peel to a commit,
  # annotated ones are tag objects. A tag name may hold `;`, CMake's list
  # separator, so it becomes `,` first: `v1.0.0;x` must not read as v1.0.0.
  _llmp_git(tags for-each-ref "--merged=${commit}" "--format=%(objecttype) %(refname)" refs/tags/)
  string(REPLACE ";" "," tags "${tags}")
  string(REPLACE "\n" ";" tags "${tags}")
  set(last "")
  foreach(tag IN LISTS tags)
    if(tag MATCHES "^tag refs/tags/v(${_LLMP_SEMVER_CORE})$")
      set(candidate "${CMAKE_MATCH_1}")
      if(last STREQUAL "" OR candidate VERSION_GREATER last)
        set(last "${candidate}")
      endif()
    endif()
  endforeach()
  if(NOT last STREQUAL "")
    _llmp_git(distance rev-list --count "refs/tags/v${last}^{commit}..${commit}")
  else()
    _llmp_git(distance rev-list --count "${commit}")
  endif()
  _llmp_git(status status --porcelain --untracked-files=normal)
  if(status STREQUAL "")
    set(dirty FALSE)
  else()
    set(dirty TRUE)
  endif()

  set(at_release FALSE)
  if(distance EQUAL 0 AND NOT last STREQUAL "")
    set(at_release TRUE)
  endif()
  if(at_release)
    if(dirty AND version VERSION_LESS last)
      message(FATAL_ERROR "project(VERSION) is ${version}, below the release tag v${last}; "
                          "a modified release checkout may keep its version or raise it (D-062)")
    elseif(NOT dirty AND NOT version VERSION_EQUAL last)
      message(FATAL_ERROR "HEAD is the release tag v${last}, but project(VERSION) is ${version}; "
                          "a release commit carries its own version (D-062)")
    endif()
  elseif(NOT last STREQUAL "" AND NOT version VERSION_GREATER last)
    message(FATAL_ERROR "project(VERSION) is ${version}, not above the last release tag v${last}; "
                        "the first commit after a release raises it, so no dev build sorts below "
                        "that release (D-062)")
  endif()

  if(at_release AND NOT dirty)
    set(semver "${version}")
  else()
    set(semver "${version}-dev.${distance}+g${short}")
    if(dirty)
      string(APPEND semver ".dirty")
    endif()
  endif()
  string(REPLACE "-" "~" debian "${semver}")
  set(${prefix}_VERSION "${semver}" PARENT_SCOPE)
  set(${prefix}_DEBIAN "${debian}-1" PARENT_SCOPE)
  set(${prefix}_COMMIT "${commit}" PARENT_SCOPE)
  set(${prefix}_DIRTY ${dirty} PARENT_SCOPE)
  set(${prefix}_ORIGIN git PARENT_SCOPE)
  set(${prefix}_GIT "${_git}" PARENT_SCOPE)
endfunction()

function(llmp_version_json out prefix)
  if(NOT "${${prefix}_COMMIT}" STREQUAL "")
    set(commit "\"${${prefix}_COMMIT}\"")
  else()
    set(commit null)
  endif()
  if(${prefix}_DIRTY)
    set(modified true)
  else()
    set(modified false)
  endif()
  set(${out} "{\"product\": \"${${prefix}_VERSION}\", \"debian\": \"${${prefix}_DEBIAN}\", \"commit\": ${commit}, \"modified\": ${modified}, \"origin\": \"${${prefix}_ORIGIN}\"}" PARENT_SCOPE)
endfunction()
