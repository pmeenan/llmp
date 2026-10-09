// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Where a path leads, its links followed as the kernel follows them, and
// facts that `llmp doctor` and the runtime report about what they find
// there. Nothing here judges who could change a path: the environment is
// trusted, and filesystem access control is its owner's (D-073 as amended
// on 2026-10-09).

#ifndef LLMP_PLATFORM_PATHS_H_
#define LLMP_PLATFORM_PATHS_H_

#include <sys/stat.h>
#include <sys/types.h>

#include <expected>
#include <filesystem>
#include <string>

namespace llmp::platform {

// Where a path leads.
struct ResolvedPath {
  // The path with every existing link resolved; a missing tail is kept as
  // written.
  std::filesystem::path resolved;
  // Whether the whole path exists; if not, `existing` is its longest
  // existing prefix (resolved), and `status` describes that.
  bool exists = false;
  std::filesystem::path existing;
  struct stat status{};
};

// Resolves an absolute path. Fails, naming the component and why, when a
// component cannot be examined (a file where a directory is needed, links
// that loop, a link to nothing, no permission to search).
std::expected<ResolvedPath, std::string> ResolvePath(const std::filesystem::path& path);

// Whether a directory carries a default ACL, which what is later created
// in it inherits (an error counts as none).
bool HasDefaultAcl(const std::filesystem::path& path);

// The mode bits as four octal digits, "0755".
std::string OctalMode(mode_t mode);

}  // namespace llmp::platform

#endif  // LLMP_PLATFORM_PATHS_H_
