// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// What this build is (D-062): the product version and commit it was built
// from, and the configuration that built it. The build generates the
// definition from Git and the build configuration every time it runs
// (cmake/version/stamp.cmake) and records the same values in the build
// receipt, llmp-receipt.json.

#ifndef LLMP_BASE_BUILD_INFO_H_
#define LLMP_BASE_BUILD_INFO_H_

#include <string_view>

namespace llmp::base {

struct BuildInfo {
  // SemVer 2.0.0: X.Y.Z for a clean checkout of the release tag vX.Y.Z,
  // otherwise X.Y.Z-dev.N+g<commit>[.dirty], or X.Y.Z-dev+unknown for a
  // tree without Git metadata.
  std::string_view version;
  // The full commit ID, or empty for a tree without Git metadata.
  std::string_view commit;
  // Whether the tree had uncommitted changes or untracked files.
  bool modified = false;
  // `core`, or `core+<module>...` with the optional modules built (D-002).
  std::string_view license_profile;
  // The identity of the SDK that built it (D-049).
  std::string_view sdk;
  // The target triple, such as aarch64-linux-gnu.
  std::string_view target;
};

const BuildInfo& GetBuildInfo();

}  // namespace llmp::base

#endif  // LLMP_BASE_BUILD_INFO_H_
