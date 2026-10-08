// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Small reads of the kernel's text interfaces (/proc, /sys) and of
// directories, with errors as values (D-066); a file's inode generation;
// and anonymous memory files.

#ifndef LLMP_PLATFORM_FILES_H_
#define LLMP_PLATFORM_FILES_H_

#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <optional>
#include <string>
#include <system_error>
#include <vector>

namespace llmp::platform {

inline constexpr std::size_t kSmallFileLimit = std::size_t{64} * 1024;

// Reads a whole file of at most limit bytes, reading to its end rather
// than trusting its reported size, which /proc and /sys files misstate. A
// longer file is an error (std::errc::file_too_large).
std::expected<std::string, std::error_code> ReadSmallFile(const std::filesystem::path& path,
                                                          std::size_t limit = kSmallFileLimit);

// The first line of such a file, without its line ending or trailing
// whitespace.
std::expected<std::string, std::error_code> ReadFirstLine(const std::filesystem::path& path);

// The names in a directory, sorted bytewise.
std::expected<std::vector<std::string>, std::error_code> ListDirectory(
    const std::filesystem::path& path);

// The open file's inode generation, where its filesystem reports one
// (Linux's FS_IOC_GETVERSION: ext4, Btrfs and XFS do; overlayfs, tmpfs
// and NFS do not; macOS has st_gen, readable by root only). With st_dev
// and st_ino it tells a file from one that later reused its inode. The
// request's buffer is its declared size, a long: FUSE passes that size to
// its server and copies back as many bytes as the server replies with,
// and the whole buffer is the value, compared for equality only. None on
// any failure.
std::optional<std::uint64_t> FileGeneration(int fd);

// A file that lives in memory only, for a mapping several views share (the
// fake device memory's backing, providers/fake/): Linux's memfd_create;
// macOS would use shm_open and shm_unlink at once. Close-on-exec. errno on
// failure.
std::expected<int, int> OpenAnonymousMemoryFile(const char* name);

// The running executable file's identity: its size and modification time
// (nanoseconds since the epoch), which change when it is rebuilt, and its
// device, inode and inode generation (FileGeneration; unset where the
// filesystem reports none), which change when it is reinstalled: Linux's
// /proc/self/exe; macOS would use _NSGetExecutablePath. None on failure.
struct ExecutableStamp {
  std::uint64_t size = 0;
  std::int64_t modified_ns = 0;
  std::uint64_t device = 0;
  std::uint64_t inode = 0;
  std::optional<std::uint64_t> generation;
};
std::optional<ExecutableStamp> RunningExecutableStamp();

}  // namespace llmp::platform

#endif  // LLMP_PLATFORM_FILES_H_
