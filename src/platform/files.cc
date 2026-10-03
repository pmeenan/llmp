// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "platform/files.h"

#include <fcntl.h>
#include <linux/fs.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <optional>
#include <string>
#include <system_error>
#include <vector>

namespace jitllm::platform {
namespace {

std::error_code LastError() { return {errno, std::generic_category()}; }

}  // namespace

std::expected<std::string, std::error_code> ReadSmallFile(const std::filesystem::path& path,
                                                          std::size_t limit) {
  const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOCTTY | O_NONBLOCK);
  if (fd < 0) {
    return std::unexpected(LastError());
  }
  std::string text;
  std::error_code error;
  // One byte past the limit tells a file of exactly `limit` bytes from a longer one.
  text.resize(limit + 1);
  std::size_t size = 0;
  while (size < text.size()) {
    const ssize_t got = ::read(fd, text.data() + size, text.size() - size);
    if (got < 0) {
      if (errno == EINTR) {
        continue;
      }
      error = LastError();
      break;
    }
    if (got == 0) {
      break;
    }
    size += static_cast<std::size_t>(got);
  }
  (void)::close(fd);  // read-only: nothing to lose
  if (error) {
    return std::unexpected(error);
  }
  if (size > limit) {
    return std::unexpected(std::make_error_code(std::errc::file_too_large));
  }
  text.resize(size);
  return text;
}

std::expected<std::string, std::error_code> ReadFirstLine(const std::filesystem::path& path) {
  auto text = ReadSmallFile(path);
  if (!text) {
    return text;
  }
  std::string line = text->substr(0, text->find('\n'));
  const std::size_t end = line.find_last_not_of(" \t\r\n");
  line.resize(end == std::string::npos ? 0 : end + 1);
  return line;
}

std::expected<std::vector<std::string>, std::error_code> ListDirectory(
    const std::filesystem::path& path) {
  std::error_code error;
  std::filesystem::directory_iterator it(path, error);
  std::vector<std::string> names;
  for (const std::filesystem::directory_iterator end; !error && it != end; it.increment(error)) {
    names.push_back(it->path().filename().string());
  }
  if (error) {
    return std::unexpected(error);
  }
  std::ranges::sort(names);
  return names;
}

std::optional<std::uint64_t> FileGeneration(int fd) {
  long generation = 0;
  static_assert(sizeof generation >= _IOC_SIZE(FS_IOC_GETVERSION));
  if (::ioctl(fd, FS_IOC_GETVERSION, &generation) != 0) {
    return std::nullopt;
  }
  return static_cast<std::uint64_t>(generation);
}

std::expected<int, int> OpenAnonymousMemoryFile(const char* name) {
  const int fd = ::memfd_create(name, MFD_CLOEXEC);
  if (fd < 0) {
    return std::unexpected(errno);
  }
  return fd;
}

std::optional<ExecutableStamp> RunningExecutableStamp() {
  struct stat status{};
  if (::stat("/proc/self/exe", &status) != 0) {
    return std::nullopt;
  }
  return ExecutableStamp{
      .size = static_cast<std::uint64_t>(status.st_size),
      .modified_ns = (static_cast<std::int64_t>(status.st_mtim.tv_sec) * 1'000'000'000) +
                     static_cast<std::int64_t>(status.st_mtim.tv_nsec)};
}

}  // namespace jitllm::platform
