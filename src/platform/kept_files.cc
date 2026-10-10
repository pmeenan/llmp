// SPDX-FileCopyrightText: 2026 llmpalooza contributors
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
#include <format>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "platform/files.h"

namespace llmp::platform {
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

// A replacement's temporary file beside `name`.
std::string TemporaryName(const char* name) { return std::string(name) + ".tmp"; }

FileIdentity IdentityOf(const struct stat& st, int fd) {
  return {.device = static_cast<std::uint64_t>(st.st_dev),
          .inode = static_cast<std::uint64_t>(st.st_ino),
          .generation = FileGeneration(fd)};
}

// The file `name` beneath `dir` now (never through a link, never waiting
// on a FIFO), with its status in `st`; errno if it cannot be opened.
std::expected<FileIdentity, int> IdentityAt(int dir, const std::string& name, struct stat& st) {
  const Owned fd(
      ::openat(dir, name.c_str(), O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC | O_NOCTTY));
  if (fd.get() < 0) {
    return std::unexpected(errno);
  }
  if (::fstat(fd.get(), &st) != 0) {
    return std::unexpected(errno);
  }
  return IdentityOf(st, fd.get());
}

}  // namespace

std::expected<int, int> OpenPrivateDirectory(int parent, const char* name) {
  const int at = parent >= 0 ? parent : AT_FDCWD;
  const bool made = ::mkdirat(at, name, 0700) == 0;
  if (!made && errno != EEXIST) {
    return std::unexpected(errno);
  }
  // One made here is exactly 0700, not what the umask left (which could
  // leave it closed to its owner too), before it is opened; it is opened
  // as the directory made, never through a link put in its place since.
  if (made && ::fchmodat(at, name, 0700, 0) != 0) {
    return std::unexpected(errno);
  }
  // One found is used as it is, a link followed (D-105 as amended on
  // 2026-10-09). The name exists, so a missing target is a link to
  // nothing.
  Owned fd(::openat(at, name, O_RDONLY | O_DIRECTORY | O_CLOEXEC | (made ? O_NOFOLLOW : 0)));
  if (fd.get() < 0) {
    return std::unexpected(errno == ENOENT && !made ? ENOLINK : errno);
  }
  return fd.release();
}

std::expected<PrivateFile, int> OpenPrivateFile(int dir, const char* name, PrivateOpen how) {
  // The name is looked at before anything is opened, and may change before
  // the open (another writer, in a directory its owner opened up): an open
  // that finds it changed tries again, a few times.
  constexpr int kAttempts = 8;
  int error = EAGAIN;
  for (int attempt = 0; attempt < kAttempts; ++attempt) {
    // What the name is now: a device or a FIFO is not opened (even
    // non-blocking, opening can have effects), and a link to nothing is not
    // created through.
    struct stat seen{};
    bool present = true;
    if (::fstatat(dir, name, &seen, AT_SYMLINK_NOFOLLOW) != 0) {
      if (errno != ENOENT) {
        return std::unexpected(errno);
      }
      present = false;
    }
    if (present && S_ISLNK(seen.st_mode)) {
      if (::fstatat(dir, name, &seen, 0) != 0) {
        return std::unexpected(errno == ENOENT ? ENOLINK : errno);
      }
    }
    if (present && !S_ISREG(seen.st_mode)) {
      return std::unexpected(EPERM);
    }
    if (!present && !how.create) {
      return std::unexpected(ENOENT);
    }
    // Emptying gives the name a new file: what was there (a link, a file
    // with other names, or one only this name leads to) is unlinked, never
    // emptied, so nothing reached another way loses its bytes. A link count
    // looked at cannot say that safely: the name and the count are read
    // apart, so a name linked to another file and unlinked again meanwhile
    // can look like a file of one link.
    const bool replace = how.truncate && present;
    if (replace && ::unlinkat(dir, name, 0) != 0 && errno != ENOENT) {
      return std::unexpected(errno);
    }
    // A file this call makes is made here (O_EXCL: never through a link,
    // never what appeared meanwhile) and is exactly 0600.
    const bool make = !present || replace;
    int flags = (how.write ? O_RDWR : O_RDONLY) | O_NONBLOCK | O_CLOEXEC | O_NOCTTY;
    if (how.direct) {
      flags |= O_DIRECT;
    }
    if (make) {
      flags |= O_CREAT | O_EXCL;
    }
    Owned fd(::openat(dir, name, flags, 0600));
    if (fd.get() < 0) {
      if ((make && errno == EEXIST) || (!make && errno == ENOENT)) {
        error = errno;  // changed since it was looked at
        continue;
      }
      return std::unexpected(errno);
    }
    struct stat st{};
    if (::fstat(fd.get(), &st) != 0) {
      return std::unexpected(errno);
    }
    if (!S_ISREG(st.st_mode)) {
      return std::unexpected(EPERM);
    }
    // Not through the umask, which could leave it unreadable to its owner.
    if (make && ::fchmod(fd.get(), 0600) != 0) {
      return std::unexpected(errno);
    }
    // Regular files do not block; the flag only kept a FIFO put there since
    // from hanging the open.
    if (::fcntl(fd.get(), F_SETFL, ::fcntl(fd.get(), F_GETFL) & ~O_NONBLOCK) != 0) {
      return std::unexpected(errno);
    }
    PrivateFile file;
    file.identity = IdentityOf(st, fd.get());
    file.bytes = static_cast<std::uint64_t>(st.st_size);
    file.fd = fd.release();
    return file;
  }
  return std::unexpected(error);
}

std::expected<FileIdentity, int> WritePrivateReplacement(int dir, const char* name,
                                                         std::string_view bytes) {
  const std::string temporary = TemporaryName(name);
  // A crashed write's temporary file, if any, goes first (unlinking a link
  // removes the link, not its target); the new one is made here, never
  // opened through what another left (O_EXCL).
  (void)::unlinkat(dir, temporary.c_str(), 0);
  std::optional<FileIdentity> made;  // once made: removed again on failure
  const auto written = [&]() -> std::expected<FileIdentity, int> {
    Owned fd(::openat(dir, temporary.c_str(),
                      O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC | O_NOCTTY, 0600));
    if (fd.get() < 0) {
      return std::unexpected(errno);
    }
    struct stat st{};
    if (::fstat(fd.get(), &st) != 0) {
      return std::unexpected(errno);
    }
    made = IdentityOf(st, fd.get());
    if (!S_ISREG(st.st_mode) || ::fchmod(fd.get(), 0600) != 0) {
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
    return *made;
  }();
  if (!written && made) {
    AbandonPrivateReplacement(dir, name, *made);  // nothing half-written stays
  }
  return written;
}

std::expected<void, int> CommitPrivateReplacement(int dir, const char* name,
                                                  const FileIdentity& written) {
  const std::string temporary = TemporaryName(name);
  // Still the file written. Not a link nor any other kind of file (never
  // one written: removed, which leaves what a link names); not another
  // regular file put in its place since (another writer's, left to it).
  // Checked on the file itself; the rename then goes by its name. Its
  // owner, mode and links are not checked: its identity says it is the one
  // written (D-105 as amended on 2026-10-09).
  struct stat st{};
  if (::fstatat(dir, temporary.c_str(), &st, AT_SYMLINK_NOFOLLOW) != 0) {
    return std::unexpected(errno);
  }
  if (!S_ISREG(st.st_mode)) {
    (void)::unlinkat(dir, temporary.c_str(), 0);
    return std::unexpected(EPERM);
  }
  auto found = IdentityAt(dir, temporary, st);
  if (!found) {
    return std::unexpected(found.error());
  }
  if (*found != written) {
    return std::unexpected(EPERM);
  }
  if (!S_ISREG(st.st_mode)) {
    (void)::unlinkat(dir, temporary.c_str(), 0);  // the one written, changed since
    return std::unexpected(EPERM);
  }
  if (::renameat(dir, temporary.c_str(), dir, name) != 0) {
    const int error = errno;
    (void)::unlinkat(dir, temporary.c_str(), 0);
    return std::unexpected(error);
  }
  return {};
}

void AbandonPrivateReplacement(int dir, const char* name, const FileIdentity& written) {
  const std::string temporary = TemporaryName(name);
  struct stat st{};
  if (auto found = IdentityAt(dir, temporary, st); found && *found == written) {
    (void)::unlinkat(dir, temporary.c_str(), 0);
  }
}

std::expected<void, int> SyncDirectory(int dir) {
  if (::fsync(dir) != 0) {
    return std::unexpected(errno);
  }
  return {};
}

std::expected<void, int> ReplacePrivateFile(int dir, const char* name, std::string_view bytes) {
  auto written = WritePrivateReplacement(dir, name, bytes);
  if (!written) {
    return std::unexpected(written.error());
  }
  if (auto renamed = CommitPrivateReplacement(dir, name, *written); !renamed) {
    return renamed;
  }
  return SyncDirectory(dir);
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

std::string PrivacyNote(int dir) {
  std::vector<std::string> parts;
  struct stat st{};
  if (::fstat(dir, &st) != 0) {
    return {};
  }
  if (st.st_uid != ::geteuid()) {
    parts.push_back(std::format("owned by uid {}, not the runtime's user (uid {})", st.st_uid,
                                static_cast<unsigned>(::geteuid())));
  }
  if ((st.st_mode & 07777) != 0700) {
    parts.push_back(std::format("mode {:04o}, not 0700", st.st_mode & 07777U));
  }
  std::size_t open_files = 0;
  std::size_t links = 0;
  if (auto entries = ListPrivateDirectory(dir)) {
    for (const DirectoryEntry& entry : *entries) {
      if (entry.directory) {
        continue;  // its own note
      }
      if (::fstatat(dir, entry.name.c_str(), &st, AT_SYMLINK_NOFOLLOW) != 0) {
        continue;
      }
      if (S_ISLNK(st.st_mode)) {
        ++links;
      } else if (S_ISREG(st.st_mode) && (!Ours(st, 0600) || st.st_nlink != 1)) {
        ++open_files;
      }
    }
  }
  if (open_files != 0) {
    parts.push_back(
        std::format("{} file{} with another owner, a mode other than 0600 or other names",
                    open_files, open_files == 1 ? "" : "s"));
  }
  if (links != 0) {
    parts.push_back(std::format("{} symbolic link{} (followed)", links, links == 1 ? "" : "s"));
  }
  std::string note;
  for (const std::string& part : parts) {
    note += note.empty() ? part : "; " + part;
  }
  return note;
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

}  // namespace llmp::platform
