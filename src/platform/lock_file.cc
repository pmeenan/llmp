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
#include <vector>

#include "platform/event_loop.h"
#include "platform/files.h"

namespace llmp::platform {
namespace {

std::string Errno(int error) { return std::strerror(error); }  // NOLINT(concurrency-mt-unsafe)

// Existing files are pinned before their I/O open, for reading only: flock
// needs no write access, so a lock file the runtime cannot write (read-only,
// or another user's) still locks. Missing names are created exclusively in a
// pinned parent directory. A final dangling link is followed
// explicitly so creation keeps the lock-file policy without O_CREAT opening a
// FIFO or device that arrived in the meantime.
std::expected<int, int> OpenLockFile(const std::filesystem::path& path) {
  std::filesystem::path name = path;
  OwnedDescriptor parent;
  int links = 0;
  constexpr int kSymlinks = 40;  // Linux's pathname-resolution limit.
  for (;;) {
    auto pinned = OpenPathReference(parent.get(), name.c_str());
    if (pinned) {
      const OwnedDescriptor reference(*pinned);
      return ReopenRegularFile(reference.get(), false);
    }
    if (pinned.error() != ENOENT) return std::unexpected(pinned.error());

    const auto directory = name.has_parent_path() ? name.parent_path() : ".";
    auto at = OpenPathReference(parent.get(), directory.c_str());
    if (!at) return std::unexpected(at.error());
    OwnedDescriptor next_parent(*at);
    struct stat status{};
    if (::fstat(next_parent.get(), &status) != 0) return std::unexpected(errno);
    if (!S_ISDIR(status.st_mode)) return std::unexpected(ENOTDIR);
    name = name.filename();
    parent = std::move(next_parent);

    auto leaf = OpenPathReference(parent.get(), name.c_str(), false);
    if (leaf) {
      const OwnedDescriptor reference(*leaf);
      if (::fstat(reference.get(), &status) != 0) return std::unexpected(errno);
      if (!S_ISLNK(status.st_mode)) return ReopenRegularFile(reference.get(), false);
      if (++links > kSymlinks) return std::unexpected(ELOOP);
      // readlinkat's empty name reads this pinned symlink, even if its name
      // has been replaced. Grow only when the filesystem reports truncation.
      std::vector<char> target(256);
      for (;;) {
        const auto count = ::readlinkat(reference.get(), "", target.data(), target.size());
        if (count < 0) return std::unexpected(errno);
        if (static_cast<std::size_t>(count) < target.size()) {
          name = std::string(target.data(), static_cast<std::size_t>(count));
          break;
        }
        target.resize(target.size() * 2);
      }
      continue;
    }
    if (leaf.error() != ENOENT) return std::unexpected(leaf.error());
    const int fd =
        ::openat(parent.get(), name.c_str(), O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    if (fd < 0) {
      if (errno == EEXIST) {
        // Resolve the creation collision once. If its object disappeared
        // again (or is now a dangling link), report the race to the caller.
        auto arrived = OpenPathReference(parent.get(), name.c_str());
        if (!arrived) {
          return std::unexpected(arrived.error() == ENOENT ? EAGAIN : arrived.error());
        }
        const OwnedDescriptor reference(*arrived);
        return ReopenRegularFile(reference.get(), false);
      }
      return std::unexpected(errno);
    }
    if (::fchmod(fd, 0600) != 0) {
      const int error = errno;
      (void)::close(fd);
      return std::unexpected(error);
    }
    return fd;
  }
}

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
  auto opened = OpenLockFile(path);
  if (!opened) {
    if (opened.error() == EPERM) {
      return std::unexpected(std::format("the lock file {} is not a regular file", path.string()));
    }
    return std::unexpected(
        std::format("cannot open the lock file {}: {}", path.string(),
                    opened.error() == kNoProcfs ? "/proc is not mounted" : Errno(opened.error())));
  }
  const int fd = *opened;
  LockFile lock(fd);
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
