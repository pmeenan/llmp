// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Link-time interception makes pathname replacements deterministic at the real
// production open boundaries; no test callbacks enter the platform library.
#include <fcntl.h>
#include <gtest/gtest.h>
#include <sys/stat.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

#include "platform/event_loop.h"
#include "platform/files.h"
#include "platform/kept_files.h"
#include "platform/lock_file.h"

namespace {
namespace fs = std::filesystem;
namespace pf = llmp::platform;

struct Intercept {
  std::string name;
  int flags = O_PATH;
  bool before = false;
  std::function<void()> replace;
  bool fired = false;
};
thread_local Intercept* intercept = nullptr;
thread_local int special_io_opens = 0;

void InterceptOpen(const char* name, int flags, bool before) {
  if (intercept != nullptr && !intercept->fired && intercept->name == name &&
      (flags & intercept->flags) == intercept->flags && intercept->before == before) {
    intercept->fired = true;  // the replacement's own opens are not intercepted
    intercept->replace();
  }
}

void ObserveOpen(int fd, int flags) {
  const int error = errno;
  struct stat status{};
  if (fd >= 0 && (flags & O_PATH) == 0 && ::fstat(fd, &status) == 0 &&
      (S_ISFIFO(status.st_mode) || S_ISCHR(status.st_mode) || S_ISBLK(status.st_mode))) {
    ++special_io_opens;
  }
  errno = error;
}

class PrivateFileRace : public ::testing::Test {
 protected:
  void SetUp() override {
    std::string pattern = (fs::path(::testing::TempDir()) / "llmp-private-XXXXXX").string();
    ASSERT_NE(::mkdtemp(pattern.data()), nullptr);
    root = pattern;
    special_io_opens = 0;
  }
  void TearDown() override {
    intercept = nullptr;
    EXPECT_EQ(special_io_opens, 0) << "a special object was opened for I/O";
    std::error_code error;
    fs::remove_all(root, error);
    EXPECT_FALSE(error) << error.message();
  }
  static void Write(const fs::path& path, std::string_view text) {
    std::ofstream out(path);
    out << text;
    ASSERT_TRUE(out.good());
  }
  static std::string Read(int fd) {
    std::array<char, 64> bytes{};
    const auto count = ::pread(fd, bytes.data(), bytes.size(), 0);
    EXPECT_GE(count, 0);
    return count < 0 ? "" : std::string(bytes.data(), static_cast<std::size_t>(count));
  }
  static void Replace(const fs::path& path, std::string_view kind) {
    if (kind == "fifo") ASSERT_EQ(::mkfifo(path.c_str(), 0600), 0);
    if (kind == "device") fs::create_symlink("/dev/null", path);
    if (kind == "regular") Write(path, "replacement");
  }
  fs::path root;
};
}  // namespace

extern "C" int __real_openat(int dir, const char* name, int flags, ...);
extern "C" int __real_open(const char* name, int flags, ...);

// NOLINTBEGIN(cppcoreguidelines-pro-type-vararg)
extern "C" int __wrap_openat(int dir, const char* name, int flags, ...) {
  mode_t mode = 0;
  if ((flags & O_CREAT) != 0 || (flags & O_TMPFILE) == O_TMPFILE) {
    va_list args;
    va_start(args, flags);
    mode = va_arg(args, mode_t);
    va_end(args);
  }
  InterceptOpen(name, flags, true);
  const int fd = __real_openat(dir, name, flags, mode);
  const int error = errno;
  if (fd >= 0) InterceptOpen(name, flags, false);
  ObserveOpen(fd, flags);
  errno = error;
  return fd;
}
extern "C" int __wrap_open(const char* name, int flags, ...) {
  mode_t mode = 0;
  if ((flags & O_CREAT) != 0 || (flags & O_TMPFILE) == O_TMPFILE) {
    va_list args;
    va_start(args, flags);
    mode = va_arg(args, mode_t);
    va_end(args);
  }
  const int fd = __real_open(name, flags, mode);
  ObserveOpen(fd, flags);
  return fd;
}
// NOLINTEND(cppcoreguidelines-pro-type-vararg)

namespace {

TEST_F(PrivateFileRace, KeptReadAndWriteOpenThePinnedObjectAfterReplacement) {
  for (const bool write : {false, true}) {
    for (const std::string_view kind : {"regular", "fifo", "device", "absent"}) {
      SCOPED_TRACE(kind);
      SCOPED_TRACE(write);
      const auto path = root / "state";
      const auto saved = root / "selected";
      Write(path, "selected");
      auto directory = pf::OpenPathReference(-1, root.c_str());
      ASSERT_TRUE(directory);
      const pf::OwnedDescriptor dir(*directory);
      Intercept race{.name = "state", .replace = [&] {
                       fs::rename(path, saved);
                       Replace(path, kind);
                     }};
      intercept = &race;
      auto opened = pf::OpenPrivateFile(dir.get(), "state", {.write = write, .create = true});
      intercept = nullptr;
      ASSERT_TRUE(race.fired);
      ASSERT_TRUE(opened) << opened.error_or(0);
      const pf::OwnedDescriptor file(opened->fd);
      EXPECT_EQ(opened->bytes, 8U);
      EXPECT_EQ(Read(file.get()), "selected");
      EXPECT_NE(::fcntl(file.get(), F_GETFD) & FD_CLOEXEC, 0);
      EXPECT_EQ(::fcntl(file.get(), F_GETFL) & O_NONBLOCK, 0);
      struct stat status{};
      ASSERT_EQ(::stat(saved.c_str(), &status), 0);
      EXPECT_EQ(opened->identity.inode, static_cast<std::uint64_t>(status.st_ino));
      if (write) {
        ASSERT_EQ(::pwrite(file.get(), "changed!", 8, 0), 8);
        EXPECT_EQ(pf::ReadSmallFile(saved).value_or(""), "changed!");
      }
      if (kind == "regular") EXPECT_EQ(pf::ReadSmallFile(path).value_or(""), "replacement");
      fs::remove(path);
      fs::remove(saved);
    }
  }
}

TEST_F(PrivateFileRace, LockRemainsOnPinnedObjectAcrossReplacementAndRelease) {
  for (const std::string_view kind : {"regular", "fifo", "device", "absent"}) {
    SCOPED_TRACE(kind);
    const auto path = root / "process.lock";
    const auto saved = root / "selected.lock";
    Write(path, "selected");
    Intercept race{.name = path.string(), .replace = [&] {
                     fs::rename(path, saved);
                     Replace(path, kind);
                   }};
    {
      intercept = &race;
      auto held = pf::LockFile::Acquire(path);
      intercept = nullptr;
      ASSERT_TRUE(race.fired);
      ASSERT_TRUE(held);
      EXPECT_EQ(Read(held->fd()), "selected");
      EXPECT_NE(::fcntl(held->fd(), F_GETFD) & FD_CLOEXEC, 0);
      auto same = pf::LockFile::Acquire(saved);
      ASSERT_FALSE(same);
      EXPECT_NE(same.error().find("is locked"), std::string::npos);
      // A new file at the former name is a distinct lock; the original
      // object stays locked until its owning descriptor closes.
      if (kind == "regular") EXPECT_TRUE(pf::LockFile::Acquire(path));
    }
    EXPECT_TRUE(pf::LockFile::Acquire(saved));
    fs::remove(path);
    fs::remove(saved);
  }
}

TEST_F(PrivateFileRace, ExclusiveCreationNeverOpensAnArrivingSpecialObject) {
  for (const bool lock : {false, true}) {
    for (const std::string_view kind : {"regular", "fifo", "device"}) {
      SCOPED_TRACE(lock);
      SCOPED_TRACE(kind);
      const auto path = root / "new";
      auto directory = pf::OpenPathReference(-1, root.c_str());
      ASSERT_TRUE(directory);
      const pf::OwnedDescriptor dir(*directory);
      Intercept race{
          .name = "new", .flags = O_CREAT, .before = true, .replace = [&] { Replace(path, kind); }};
      intercept = &race;
      if (lock) {
        auto result = pf::LockFile::Acquire(path);
        EXPECT_EQ(result.has_value(), kind == "regular");
        if (result) EXPECT_EQ(Read(result->fd()), "replacement");
      } else {
        auto result = pf::OpenPrivateFile(dir.get(), "new", {.write = true, .create = true});
        EXPECT_EQ(result.has_value(), kind == "regular");
        if (result) {
          const pf::OwnedDescriptor file(result->fd);
          EXPECT_EQ(Read(file.get()), "replacement");
        }
      }
      intercept = nullptr;
      EXPECT_TRUE(race.fired);
      fs::remove(path);
    }
  }
}

TEST_F(PrivateFileRace, KeptTruncationReplacesTheNameWithoutEmptyingTheSelectedObject) {
  const auto path = root / "state";
  Write(path, "selected");
  const auto saved = root / "selected";
  fs::create_hard_link(path, saved);
  const auto victim = root / "victim";
  Write(victim, "preserved");
  auto directory = pf::OpenPathReference(-1, root.c_str());
  ASSERT_TRUE(directory);
  const pf::OwnedDescriptor dir(*directory);
  Intercept race{.name = "state", .replace = [&] {
                   fs::remove(path);
                   fs::create_symlink(victim, path);
                 }};
  intercept = &race;
  auto result = pf::OpenPrivateFile(dir.get(), "state", {.write = true, .truncate = true});
  intercept = nullptr;
  ASSERT_TRUE(race.fired);
  ASSERT_TRUE(result);
  const pf::OwnedDescriptor file(result->fd);
  EXPECT_EQ(result->bytes, 0U);
  EXPECT_EQ(pf::ReadSmallFile(saved).value_or(""), "selected");
  EXPECT_EQ(pf::ReadSmallFile(victim).value_or(""), "preserved");
  EXPECT_FALSE(fs::is_symlink(path));
}

TEST_F(PrivateFileRace, DanglingLinksKeepTheirDistinctKeptAndLockCreationPolicies) {
  fs::create_directory(root / "sub");
  fs::create_symlink("sub/next", root / "link");
  fs::create_symlink("../target", root / "sub/next");
  auto directory = pf::OpenPathReference(-1, root.c_str());
  ASSERT_TRUE(directory);
  const pf::OwnedDescriptor dir(*directory);
  EXPECT_EQ(pf::OpenPrivateFile(dir.get(), "link", {.write = true, .create = true}).error_or(0),
            ENOLINK);
  EXPECT_FALSE(fs::exists(root / "target"));
  auto locked = pf::LockFile::Acquire(root / "link");
  ASSERT_TRUE(locked);
  EXPECT_TRUE(fs::is_symlink(root / "link"));
  EXPECT_TRUE(fs::is_symlink(root / "sub/next"));
  struct stat status{};
  ASSERT_EQ(::stat((root / "target").c_str(), &status), 0);
  EXPECT_EQ(status.st_mode & 07777U, 0600U);
  EXPECT_EQ(status.st_size, 0);
  EXPECT_FALSE(pf::LockFile::Acquire(root / "target"));
}

TEST_F(PrivateFileRace, DanglingLockUsesThePinnedLinkWhenItsNameChanges) {
  fs::create_symlink("selected", root / "link");
  Intercept race{.name = "link", .flags = O_PATH | O_NOFOLLOW, .replace = [&] {
                   fs::remove(root / "link");
                   fs::create_symlink("replacement", root / "link");
                 }};
  intercept = &race;
  auto locked = pf::LockFile::Acquire(root / "link");
  intercept = nullptr;
  ASSERT_TRUE(race.fired);
  ASSERT_TRUE(locked);
  EXPECT_TRUE(fs::exists(root / "selected"));
  EXPECT_FALSE(fs::exists(root / "replacement"));
  EXPECT_FALSE(pf::LockFile::Acquire(root / "selected"));
}

TEST_F(PrivateFileRace, LockCreatesInThePinnedParentWhenItsNameChanges) {
  fs::create_directory(root / "parent");
  Intercept race{.name = (root / "parent").string(), .replace = [&] {
                   fs::rename(root / "parent", root / "selected");
                   fs::create_directory(root / "parent");
                   fs::create_symlink("/dev/null", root / "parent/new");
                 }};
  intercept = &race;
  auto locked = pf::LockFile::Acquire(root / "parent/new");
  intercept = nullptr;
  ASSERT_TRUE(race.fired);
  ASSERT_TRUE(locked);
  EXPECT_TRUE(fs::is_regular_file(root / "selected/new"));
  EXPECT_TRUE(fs::is_symlink(root / "parent/new"));
  EXPECT_FALSE(pf::LockFile::Acquire(root / "selected/new"));
}

TEST_F(PrivateFileRace, ReplacementIdentityChecksRefuseSubstitutedSpecialObjectsBeforeIo) {
  auto directory = pf::OpenPathReference(-1, root.c_str());
  ASSERT_TRUE(directory);
  const pf::OwnedDescriptor dir(*directory);
  for (const bool commit : {false, true}) {
    for (const std::string_view kind : {"fifo", "device"}) {
      SCOPED_TRACE(commit);
      SCOPED_TRACE(kind);
      Write(root / "record", "original");
      auto written = pf::WritePrivateReplacement(dir.get(), "record", "replacement");
      ASSERT_TRUE(written);
      Intercept race{.name = "record.tmp", .before = true, .replace = [&] {
                       fs::remove(root / "record.tmp");
                       Replace(root / "record.tmp", kind);
                     }};
      intercept = &race;
      if (commit) {
        EXPECT_EQ(pf::CommitPrivateReplacement(dir.get(), "record", *written).error_or(0), EPERM);
      } else {
        pf::AbandonPrivateReplacement(dir.get(), "record", *written);
      }
      intercept = nullptr;
      EXPECT_TRUE(race.fired);
      EXPECT_EQ(pf::ReadSmallFile(root / "record").value_or(""), "original");
      fs::remove(root / "record.tmp");
    }
  }
}

TEST_F(PrivateFileRace, LockRefusesSymlinkLoopsAndDoesNotCreateMissingParents) {
  fs::create_symlink("loop", root / "loop");
  EXPECT_FALSE(pf::LockFile::Acquire(root / "loop"));
  fs::create_symlink("missing/child", root / "dangling");
  EXPECT_FALSE(pf::LockFile::Acquire(root / "dangling"));
  EXPECT_FALSE(fs::exists(root / "missing"));
}

}  // namespace
