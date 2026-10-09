// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "platform/paths.h"

#include <sys/stat.h>
#include <sys/types.h>
#include <sys/xattr.h>

#include <cerrno>
#include <cstring>
#include <expected>
#include <filesystem>
#include <format>
#include <string>
#include <system_error>
#include <vector>

namespace llmp::platform {
namespace {

namespace fs = std::filesystem;

std::string Errno(int error) { return std::strerror(error); }  // NOLINT(concurrency-mt-unsafe)

}  // namespace

std::expected<ResolvedPath, std::string> ResolvePath(const fs::path& path) {
  if (!path.is_absolute()) {
    return std::unexpected(std::format("{} is not an absolute path", path.string()));
  }
  // The longest prefix that exists, its links followed by stat; what is
  // missing below it is kept as written.
  ResolvedPath result;
  fs::path existing = path.lexically_normal();
  std::vector<fs::path> missing;
  while (::stat(existing.c_str(), &result.status) != 0) {
    const int error = errno;
    if (error != ENOENT || !existing.has_relative_path()) {
      return std::unexpected(
          std::format("{} cannot be examined: {}", existing.string(), Errno(error)));
    }
    // A link to nothing is not a missing tail: what would be created is
    // not where the path says, and creating through it fails.
    if (struct stat link{}; ::lstat(existing.c_str(), &link) == 0 && S_ISLNK(link.st_mode)) {
      return std::unexpected(
          std::format("{} is a link to a path that does not exist", existing.string()));
    }
    if (existing.has_filename()) {
      missing.push_back(existing.filename());
    }
    existing = existing.parent_path();
  }
  std::error_code error;
  result.existing = fs::canonical(existing, error);
  if (error) {
    return std::unexpected(
        std::format("{} cannot be resolved: {}", existing.string(), error.message()));
  }
  result.exists = missing.empty();
  result.resolved = result.existing;
  for (auto part = missing.rbegin(); part != missing.rend(); ++part) {
    result.resolved /= *part;
  }
  return result;
}

bool HasDefaultAcl(const fs::path& path) {
  return ::getxattr(path.c_str(), "system.posix_acl_default", nullptr, 0) >= 0;
}

std::string OctalMode(mode_t mode) { return std::format("{:04o}", mode & 07777U); }

}  // namespace llmp::platform
