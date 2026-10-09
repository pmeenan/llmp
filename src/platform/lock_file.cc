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

std::expected<LockFile, std::string> LockFile::Acquire(const std::filesystem::path& path) {
  // Only a regular file (or nothing yet) is opened: opening a device or a
  // FIFO for writing can have effects of its own. Through a link to
  // nothing, the open creates the link's target, empty; nothing is ever
  // written to the file, so an existing one is left as it was.
  if (struct stat seen{}; ::stat(path.c_str(), &seen) == 0 && !S_ISREG(seen.st_mode)) {
    return std::unexpected(std::format("the lock file {} is not a regular file", path.string()));
  }
  const int fd = ::open(path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC | O_NOCTTY | O_NONBLOCK, 0600);
  if (fd < 0) {
    return std::unexpected(
        std::format("cannot open the lock file {}: {}", path.string(), Errno(errno)));
  }
  LockFile lock(fd);
  struct stat status{};
  if (::fstat(fd, &status) != 0) {
    return std::unexpected(
        std::format("cannot examine the lock file {}: {}", path.string(), Errno(errno)));
  }
  if (!S_ISREG(status.st_mode)) {
    return std::unexpected(std::format("the lock file {} is not a regular file", path.string()));
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
