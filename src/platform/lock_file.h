// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// An exclusive lock held for a process's lifetime: flock(2) on a file that
// only its owner can open, so no other user can take it first. The lock
// goes with the open file description, so the kernel releases it when the
// last descriptor for it closes, however the process ends.

#ifndef LLMP_PLATFORM_LOCK_FILE_H_
#define LLMP_PLATFORM_LOCK_FILE_H_

#include <sys/types.h>

#include <expected>
#include <filesystem>
#include <string>

namespace llmp::platform {

class LockFile {
 public:
  LockFile(LockFile&& other) noexcept;
  LockFile& operator=(LockFile&& other) noexcept;
  LockFile(const LockFile&) = delete;
  LockFile& operator=(const LockFile&) = delete;
  ~LockFile();

  // Opens or creates path (mode 0600, never through a link) and takes its
  // lock without waiting. The file must be a regular file owned by
  // `owner` with no access for anyone else. A lock someone else holds is
  // an error naming the path. The descriptor is close-on-exec, so no child
  // inherits the lock.
  static std::expected<LockFile, std::string> Acquire(const std::filesystem::path& path,
                                                      uid_t owner);

  int fd() const { return fd_; }

 private:
  explicit LockFile(int fd) : fd_(fd) {}
  int fd_ = -1;
};

}  // namespace llmp::platform

#endif  // LLMP_PLATFORM_LOCK_FILE_H_
