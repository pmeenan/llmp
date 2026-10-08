// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "platform/lock_file.h"

#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <expected>
#include <filesystem>
#include <format>
#include <string>
#include <utility>

#include "platform/path_trust.h"

namespace llmp::platform {
namespace {

std::string Errno(int error) { return std::strerror(error); }  // NOLINT(concurrency-mt-unsafe)

}  // namespace

LockFile::LockFile(LockFile&& other) noexcept : fd_(std::exchange(other.fd_, -1)) {}

LockFile& LockFile::operator=(LockFile&& other) noexcept {
  if (this != &other) {
    if (fd_ >= 0) {
      (void)::close(fd_);  // releases the lock
    }
    fd_ = std::exchange(other.fd_, -1);
  }
  return *this;
}

LockFile::~LockFile() {
  if (fd_ >= 0) {
    (void)::close(fd_);  // releases the lock
  }
}

std::expected<LockFile, std::string> LockFile::Acquire(const std::filesystem::path& path,
                                                       uid_t owner) {
  const int fd =
      ::open(path.c_str(), O_RDWR | O_CREAT | O_NOFOLLOW | O_CLOEXEC | O_NOCTTY | O_NONBLOCK, 0600);
  if (fd < 0) {
    const int error = errno;
    return std::unexpected(std::format("cannot open the lock file {}: {}", path.string(),
                                       error == ELOOP ? "it is a symbolic link" : Errno(error)));
  }
  LockFile lock(fd);
  struct stat status{};
  if (::fstat(fd, &status) != 0) {
    return std::unexpected(
        std::format("cannot examine the lock file {}: {}", path.string(), Errno(errno)));
  }
  if (!S_ISREG(status.st_mode) || status.st_uid != owner || (status.st_mode & 077U) != 0) {
    return std::unexpected(
        std::format("the lock file {} must be a regular file owned by uid {} with mode 0600, "
                    "not uid {} and mode {}",
                    path.string(), owner, status.st_uid, OctalMode(status.st_mode)));
  }
  while (::flock(fd, LOCK_EX | LOCK_NB) != 0) {
    const int error = errno;
    if (error == EINTR) {
      continue;
    }
    if (error == EWOULDBLOCK) {
      return std::unexpected(std::format("{} is locked: another process holds it", path.string()));
    }
    return std::unexpected(std::format("cannot lock {}: {}", path.string(), Errno(error)));
  }
  return lock;
}

}  // namespace llmp::platform
