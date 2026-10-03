// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "platform/kept_files.h"

#include <dirent.h>
#include <fcntl.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "platform/files.h"

namespace jitllm::platform {
namespace {

// A descriptor closed when it goes out of scope, unless released.
class Owned {
 public:
  explicit Owned(int fd) : fd_(fd) {}
  Owned(const Owned&) = delete;
  Owned& operator=(const Owned&) = delete;
  Owned(Owned&&) = delete;
  Owned& operator=(Owned&&) = delete;
  ~Owned() {
    if (fd_ >= 0) {
      (void)::close(fd_);
    }
  }
  int get() const { return fd_; }
  int release() { return std::exchange(fd_, -1); }

 private:
  int fd_;
};

bool Ours(const struct stat& st, mode_t mode) {
  return st.st_uid == ::geteuid() && (st.st_mode & 07777) == mode;
}

}  // namespace

std::expected<int, int> OpenPrivateDirectory(int parent, const char* name) {
  const int at = parent >= 0 ? parent : AT_FDCWD;
  if (::mkdirat(at, name, 0700) != 0 && errno != EEXIST) {
    return std::unexpected(errno);
  }
  Owned fd(::openat(at, name, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC));
  if (fd.get() < 0) {
    return std::unexpected(errno);
  }
  struct stat st{};
  if (::fstat(fd.get(), &st) != 0) {
    return std::unexpected(errno);
  }
  if (!S_ISDIR(st.st_mode) || !Ours(st, 0700)) {
    return std::unexpected(EPERM);
  }
  return fd.release();
}

std::expected<PrivateFile, int> OpenPrivateFile(int dir, const char* name, PrivateOpen how) {
  int flags = (how.write ? O_RDWR : O_RDONLY) | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC | O_NOCTTY;
  if (how.create) {
    flags |= O_CREAT;
  }
  if (how.direct) {
    flags |= O_DIRECT;
  }
  Owned fd(::openat(dir, name, flags, 0600));
  if (fd.get() < 0) {
    return std::unexpected(errno);
  }
  struct stat st{};
  if (::fstat(fd.get(), &st) != 0) {
    return std::unexpected(errno);
  }
  if (!S_ISREG(st.st_mode) || !Ours(st, 0600) || st.st_nlink != 1) {
    return std::unexpected(EPERM);
  }
  // Regular files do not block; the flag only kept a FIFO from hanging the
  // open.
  if (::fcntl(fd.get(), F_SETFL, ::fcntl(fd.get(), F_GETFL) & ~O_NONBLOCK) != 0) {
    return std::unexpected(errno);
  }
  if (how.truncate && ::ftruncate(fd.get(), 0) != 0) {
    return std::unexpected(errno);
  }
  PrivateFile file;
  file.identity = {.device = static_cast<std::uint64_t>(st.st_dev),
                   .inode = static_cast<std::uint64_t>(st.st_ino),
                   .generation = FileGeneration(fd.get())};
  file.bytes = how.truncate ? 0 : static_cast<std::uint64_t>(st.st_size);
  file.fd = fd.release();
  return file;
}

std::expected<void, int> ReplacePrivateFile(int dir, const char* name, std::string_view bytes) {
  const std::string temporary = std::string(name) + ".tmp";
  // A crashed write's temporary file, if any, goes first (unlinking a link
  // removes the link, not its target); the new one is made here, never
  // opened through what another left (O_EXCL).
  (void)::unlinkat(dir, temporary.c_str(), 0);
  const auto written = [&]() -> std::expected<void, int> {
    Owned fd(::openat(dir, temporary.c_str(),
                      O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC | O_NOCTTY, 0600));
    if (fd.get() < 0) {
      return std::unexpected(errno);
    }
    struct stat st{};
    if (::fstat(fd.get(), &st) != 0) {
      return std::unexpected(errno);
    }
    if (!S_ISREG(st.st_mode) || st.st_uid != ::geteuid() || st.st_nlink != 1 ||
        ::fchmod(fd.get(), 0600) != 0) {
      return std::unexpected(EPERM);
    }
    std::size_t at = 0;
    while (at < bytes.size()) {
      const ssize_t wrote = ::write(fd.get(), bytes.data() + at, bytes.size() - at);
      if (wrote < 0) {
        if (errno == EINTR) {
          continue;
        }
        return std::unexpected(errno);
      }
      at += static_cast<std::size_t>(wrote);
    }
    if (::fsync(fd.get()) != 0) {
      return std::unexpected(errno);
    }
    if (::renameat(dir, temporary.c_str(), dir, name) != 0) {
      return std::unexpected(errno);
    }
    return {};
  }();
  if (!written) {
    (void)::unlinkat(dir, temporary.c_str(), 0);  // nothing half-written stays
    return written;
  }
  if (::fsync(dir) != 0) {
    return std::unexpected(errno);
  }
  return {};
}

std::expected<std::string, int> ReadPrivateFile(int dir, const char* name, std::size_t limit) {
  auto opened = OpenPrivateFile(dir, name, {});
  if (!opened) {
    return std::unexpected(opened.error());
  }
  const Owned fd(opened->fd);
  if (opened->bytes > limit) {
    return std::unexpected(EFBIG);
  }
  std::string text(static_cast<std::size_t>(opened->bytes), '\0');
  std::size_t at = 0;
  while (at < text.size()) {
    const ssize_t got =
        ::pread(fd.get(), text.data() + at, text.size() - at, static_cast<off_t>(at));
    if (got < 0) {
      if (errno == EINTR) {
        continue;
      }
      return std::unexpected(errno);
    }
    if (got == 0) {
      return std::unexpected(EIO);  // shorter than it said
    }
    at += static_cast<std::size_t>(got);
  }
  return text;
}

std::expected<void, int> RemovePrivate(int dir, const char* name) {
  struct stat st{};
  if (::fstatat(dir, name, &st, AT_SYMLINK_NOFOLLOW) != 0) {
    return errno == ENOENT ? std::expected<void, int>{} : std::unexpected(errno);
  }
  if (S_ISDIR(st.st_mode)) {
    Owned sub(::openat(dir, name, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC));
    if (sub.get() < 0) {
      return std::unexpected(errno);
    }
    auto entries = ListPrivateDirectory(sub.get());
    if (!entries) {
      return std::unexpected(entries.error());
    }
    for (const DirectoryEntry& entry : *entries) {
      if (auto removed = RemovePrivate(sub.get(), entry.name.c_str()); !removed) {
        return removed;
      }
    }
    if (::unlinkat(dir, name, AT_REMOVEDIR) != 0 && errno != ENOENT) {
      return std::unexpected(errno);
    }
    return {};
  }
  if (::unlinkat(dir, name, 0) != 0 && errno != ENOENT) {
    return std::unexpected(errno);
  }
  return {};
}

std::expected<std::vector<DirectoryEntry>, int> ListPrivateDirectory(int dir) {
  // A descriptor of its own for the stream: closedir closes it.
  const int copy = ::openat(dir, ".", O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  if (copy < 0) {
    return std::unexpected(errno);
  }
  DIR* stream = ::fdopendir(copy);
  if (stream == nullptr) {
    const int error = errno;
    (void)::close(copy);
    return std::unexpected(error);
  }
  std::vector<DirectoryEntry> entries;
  int error = 0;
  for (;;) {
    errno = 0;
    const dirent* entry = ::readdir(stream);  // NOLINT(concurrency-mt-unsafe): its own stream
    if (entry == nullptr) {
      error = errno;  // 0 at the end
      break;
    }
    const std::string_view name(entry->d_name);
    if (name == "." || name == "..") {
      continue;
    }
    struct stat st{};
    if (::fstatat(dir, entry->d_name, &st, AT_SYMLINK_NOFOLLOW) != 0) {
      continue;  // gone meanwhile
    }
    entries.push_back({.name = std::string(name),
                       .regular = S_ISREG(st.st_mode),
                       .directory = S_ISDIR(st.st_mode)});
  }
  (void)::closedir(stream);
  if (error != 0) {
    return std::unexpected(error);
  }
  std::ranges::sort(entries, {}, &DirectoryEntry::name);
  return entries;
}

std::expected<void, int> SyncFileData(int fd) {
  if (::fdatasync(fd) != 0) {
    return std::unexpected(errno);
  }
  return {};
}

bool LowerThreadPriority() {
  const auto tid = static_cast<id_t>(::gettid());
  const bool nice = ::setpriority(PRIO_PROCESS, tid, 19) == 0;
  // ioprio_set(IOPRIO_WHO_PROCESS, tid, IOPRIO_PRIO_VALUE(IOPRIO_CLASS_IDLE, 0)):
  // no glibc wrapper.
  constexpr int kWhoProcess = 1;
  constexpr int kClassIdle = 3;
  constexpr int kClassShift = 13;
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg)
  const bool idle = ::syscall(SYS_ioprio_set, kWhoProcess, tid, kClassIdle << kClassShift) == 0;
  return nice && idle;
}

}  // namespace jitllm::platform
