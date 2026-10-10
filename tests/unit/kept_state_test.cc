// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Conversations kept across a restart (D-105), host-side: the owner-only
// files they live in (platform/kept_files.h), their record's format
// (runtime/kept_record.h: written whole, read back strictly, refused when
// cut short, edited, foreign or past its retention) and the keeper that
// hashes and writes records in the background (runtime/state_keeper.h: a
// record describes its files exactly or does not exist; one invalidated
// while it is made is never written). Scratch trees go in the build tree
// (LLMP_TEST_SCRATCH), on a filesystem with direct I/O.

#include <fcntl.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <expected>
#include <filesystem>
#include <format>
#include <fstream>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "base/sha256.h"
#include "platform/direct_io.h"
#include "platform/files.h"
#include "platform/kept_files.h"
#include "runtime/kept_record.h"
#include "runtime/state_keeper.h"

namespace {

namespace fs = std::filesystem;
namespace kept = llmp::runtime::kept;
namespace pf = llmp::platform;
using llmp::runtime::HashPlaces;
using llmp::runtime::StateKeeper;
using ::testing::HasSubstr;

constexpr std::uint64_t kExtent = kept::kExtentBytes;

fs::path Scratch(std::string_view name) {
  const char* base = std::getenv("LLMP_TEST_SCRATCH");  // NOLINT(concurrency-mt-unsafe)
  const fs::path root = base != nullptr ? fs::path(base) : fs::path(::testing::TempDir());
  const fs::path path = root / std::format("kept-{}-{}", name, ::getpid());
  fs::remove_all(path);
  fs::create_directories(path.parent_path());
  return path;
}

// An open private directory, closed at the end.
struct Dir {
  explicit Dir(const fs::path& path)
      : fd(pf::OpenPrivateDirectory(-1, path.c_str()).value_or(-1)) {}
  Dir(const Dir&) = delete;
  Dir& operator=(const Dir&) = delete;
  Dir(Dir&&) = delete;
  Dir& operator=(Dir&&) = delete;
  ~Dir() {
    if (fd >= 0) {
      (void)::close(fd);
    }
  }
  int fd;
};

// Writes `extents` 2 MiB extents of a pattern to `name` beneath `dir`,
// with direct I/O: its identity.
pf::FileIdentity WriteExtents(int dir, const char* name, std::size_t extents, std::uint8_t seed) {
  auto file = pf::OpenPrivateFile(
      dir, name, {.write = true, .create = true, .truncate = true, .direct = true});
  EXPECT_TRUE(file.has_value());
  auto* raw = static_cast<std::byte*>(std::aligned_alloc(4096, kExtent));  // NOLINT
  for (std::size_t e = 0; e < extents; ++e) {
    for (std::uint64_t i = 0; i < kExtent; ++i) {
      raw[i] = static_cast<std::byte>((i * 7) + (e * 13) + seed);
    }
    EXPECT_TRUE(pf::TransferDirectFile(file->fd, e * kExtent, std::span(raw, kExtent), true));
  }
  std::free(raw);  // NOLINT(cppcoreguidelines-no-malloc)
  const pf::FileIdentity id = file->identity;
  (void)::close(file->fd);
  return id;
}

kept::FileId Id(const pf::FileIdentity& id) {
  return {.device = id.device, .inode = id.inode, .generation = id.generation};
}

kept::Record Sample() {
  kept::Record r;
  r.identity = {.build = "0.1.0;abc;sdk;aarch64;exe=1:2:3:4:5.000000006",
                .artifact = "8a355bfb",
                .drafter = "dd2d3f9c",
                .layout = "deepseek4-state/1;context=8704;dspark=1;window=ring;regions=t:1:2"};
  r.slot = 2;
  r.files = {{.name = kept::StateFileName(2),
              .id = {.device = 66306, .inode = 1234567, .generation = 99},
              .bytes = 6 * kExtent}};
  r.regions = {4 * kExtent, 2 * kExtent};
  r.extents = {{.region = 0, .index = 0, .digest = {}},
               {.region = 0, .index = 1, .digest = {}},
               {.region = 1, .index = 0, .digest = {}}};
  r.extents[1].digest[0] = 0xab;
  r.tokens = {1, 2, 3, 100, 200, 7};
  r.cursor = 1;
  r.decoding = {3, 2, 3, 0x3ff28f5c28f5c28fULL, 0, 0, 0, 5, 0x4000000000000000ULL, 9, 0};
  r.used_unix_ms = 1'790'000'000'000;
  r.checkpoints.push_back({.file = kept::CheckpointFileName(2, 7),
                           .id = {.device = 66306, .inode = 7654321, .generation = std::nullopt},
                           .file_bytes = 8192,
                           .position = 4,
                           .created_unix_ms = 1'789'999'000'000,
                           .ranges = {{.region = 0, .offset = 2 * kExtent, .bytes = 5000}},
                           .footprint = {{.region = 0, .offset = 0, .bytes = kExtent}},
                           .cursor = 0,
                           .decoding = r.decoding,
                           .digests = {kept::Digest{}}});
  return r;
}

kept::Expected Expect(const kept::Record& r) {
  return {.identity = r.identity,
          .slot = r.slot,
          .regions = r.regions,
          .layouts = {(4 * kExtent) - 4096, 2 * kExtent},
          .context = 8704,
          .vocabulary = 129280,
          .now_unix_ms = r.used_unix_ms + 1000,
          .retention_ms = 24LL * 3600 * 1000};
}

// --------------------------------------------------------------- the files

TEST(KeptFiles, WhatItMakesIsPrivateAndWhatItFindsIsUsedWithANote) {
  const fs::path root = Scratch("files");
  fs::create_directories(root);
  fs::permissions(root, fs::perms::owner_all, fs::perm_options::replace);
  // Made exactly 0700 whatever the umask.
  const mode_t old_umask = ::umask(0277);
  Dir dir(root / "conversations");
  (void)::umask(old_umask);
  ASSERT_GE(dir.fd, 0);
  struct stat st{};
  ASSERT_EQ(::stat((root / "conversations").c_str(), &st), 0);
  EXPECT_EQ(st.st_mode & 07777, 0700U);
  EXPECT_EQ(pf::PrivacyNote(dir.fd), "");

  // A directory found with other modes is used, with a note; a link to a
  // directory is followed; a link to nothing, a file and a missing parent
  // are refused by name.
  fs::create_directories(root / "open");
  fs::permissions(root / "open", fs::perms::all, fs::perm_options::replace);
  auto open = pf::OpenPrivateDirectory(-1, (root / "open").c_str());
  ASSERT_TRUE(open.has_value());
  EXPECT_THAT(pf::PrivacyNote(*open), HasSubstr("mode 0777, not 0700"));
  (void)::close(*open);
  ASSERT_EQ(::stat((root / "open").c_str(), &st), 0);
  EXPECT_EQ(st.st_mode & 07777, 0777U);  // left as it was
  fs::create_directory_symlink(root / "conversations", root / "link");
  auto followed = pf::OpenPrivateDirectory(-1, (root / "link").c_str());
  ASSERT_TRUE(followed.has_value());
  EXPECT_EQ(pf::PrivacyNote(*followed), "");
  (void)::close(*followed);
  fs::create_directory_symlink(root / "nowhere", root / "dangling");
  EXPECT_EQ(pf::OpenPrivateDirectory(-1, (root / "dangling").c_str()).error_or(0), ENOLINK);
  EXPECT_FALSE(fs::exists(root / "nowhere"));
  {
    std::ofstream(root / "plain") << "x";
  }
  EXPECT_EQ(pf::OpenPrivateDirectory(-1, (root / "plain").c_str()).error_or(0), ENOTDIR);
  if (::geteuid() == 0) {
    // Another owner can only be made as root.
    ASSERT_EQ(::chown((root / "open").c_str(), 65534, static_cast<gid_t>(-1)), 0);
    auto foreign = pf::OpenPrivateDirectory(-1, (root / "open").c_str());
    ASSERT_TRUE(foreign.has_value());
    EXPECT_THAT(pf::PrivacyNote(*foreign), HasSubstr("owned by uid 65534"));
    (void)::close(*foreign);
  }

  // A file: made 0600 (whatever the umask); one found with another mode,
  // another link or a symbolic link is used and noted; only a link to
  // nothing or what is not a regular file is refused.
  const mode_t before = ::umask(0277);
  auto made = pf::OpenPrivateFile(dir.fd, "a", {.write = true, .create = true});
  (void)::umask(before);
  ASSERT_TRUE(made.has_value());
  ASSERT_EQ(::write(made->fd, "abc", 3), 3);
  (void)::close(made->fd);
  ASSERT_EQ(::fstatat(dir.fd, "a", &st, 0), 0);
  EXPECT_EQ(st.st_mode & 07777, 0600U);
  EXPECT_EQ(pf::PrivacyNote(dir.fd), "");
  ASSERT_EQ(::symlinkat("a", dir.fd, "b"), 0);
  auto through = pf::OpenPrivateFile(dir.fd, "b", {});
  ASSERT_TRUE(through.has_value());
  EXPECT_EQ(through->bytes, 3U);
  (void)::close(through->fd);
  EXPECT_EQ(pf::PrivacyNote(dir.fd), "1 symbolic link (followed)");
  ASSERT_EQ(::unlinkat(dir.fd, "b", 0), 0);
  ASSERT_EQ(::linkat(dir.fd, "a", dir.fd, "c", 0), 0);
  auto linked = pf::OpenPrivateFile(dir.fd, "a", {});
  ASSERT_TRUE(linked.has_value());
  (void)::close(linked->fd);
  EXPECT_THAT(pf::PrivacyNote(dir.fd), HasSubstr("2 files with another owner"));
  // Emptying a file with other names makes this name a new file: the
  // other name keeps what it had.
  auto emptied = pf::OpenPrivateFile(dir.fd, "a", {.write = true, .truncate = true});
  ASSERT_TRUE(emptied.has_value());
  EXPECT_EQ(emptied->bytes, 0U);
  EXPECT_NE(emptied->identity, linked->identity);
  (void)::close(emptied->fd);
  EXPECT_EQ(pf::ReadPrivateFile(dir.fd, "c", 16).value_or(""), "abc");
  EXPECT_EQ(pf::PrivacyNote(dir.fd), "");  // "c" alone again
  ASSERT_EQ(::unlinkat(dir.fd, "c", 0), 0);
  ASSERT_EQ(::fchmodat(dir.fd, "a", 0644, 0), 0);
  auto loose = pf::OpenPrivateFile(dir.fd, "a", {.write = true});
  ASSERT_TRUE(loose.has_value());
  (void)::close(loose->fd);
  EXPECT_THAT(pf::PrivacyNote(dir.fd), HasSubstr("1 file with another owner"));
  ASSERT_EQ(::fchmodat(dir.fd, "a", 0600, 0), 0);
  // The same through a link: a name that is a link is replaced, not
  // written through, when it is emptied; the file it named is untouched.
  ASSERT_EQ(::symlinkat("a", dir.fd, "d"), 0);
  auto replaced =
      pf::OpenPrivateFile(dir.fd, "d", {.write = true, .create = true, .truncate = true});
  ASSERT_TRUE(replaced.has_value());
  ASSERT_EQ(::write(replaced->fd, "xy", 2), 2);
  (void)::close(replaced->fd);
  EXPECT_FALSE(fs::is_symlink(root / "conversations" / "d"));
  EXPECT_EQ(pf::ReadPrivateFile(dir.fd, "d", 16).value_or(""), "xy");
  EXPECT_EQ(pf::ReadPrivateFile(dir.fd, "a", 16).value_or(""), "");
  ASSERT_EQ(::unlinkat(dir.fd, "d", 0), 0);
  // A link to nothing is refused by name and never created through.
  ASSERT_EQ(::symlinkat("nowhere", dir.fd, "e"), 0);
  EXPECT_EQ(pf::OpenPrivateFile(dir.fd, "e", {}).error_or(0), ENOLINK);
  EXPECT_EQ(pf::OpenPrivateFile(dir.fd, "e", {.write = true, .create = true}).error_or(0), ENOLINK);
  EXPECT_FALSE(fs::exists(root / "conversations" / "nowhere"));
  ASSERT_EQ(::unlinkat(dir.fd, "e", 0), 0);
  // Not a regular file: refused, a FIFO without opening (it would block).
  ASSERT_EQ(::mkfifoat(dir.fd, "fifo", 0600), 0);
  EXPECT_EQ(pf::OpenPrivateFile(dir.fd, "fifo", {}).error_or(0), EPERM);
  EXPECT_EQ(pf::OpenPrivateFile(dir.fd, "fifo", {.write = true}).error_or(0), EPERM);
  ASSERT_EQ(::unlinkat(dir.fd, "fifo", 0), 0);
  ASSERT_EQ(::mkdirat(dir.fd, "adir", 0700), 0);
  EXPECT_EQ(pf::OpenPrivateFile(dir.fd, "adir", {}).error_or(0), EPERM);
  ASSERT_EQ(::unlinkat(dir.fd, "adir", AT_REMOVEDIR), 0);
  ASSERT_EQ(::symlinkat("a", dir.fd, "b"), 0);  // for the listing below
  EXPECT_TRUE(pf::OpenPrivateFile(dir.fd, "missing", {}).error_or(0) == ENOENT);

  // Replaced whole, read back; removed, a directory with all in it.
  ASSERT_TRUE(pf::ReplacePrivateFile(dir.fd, "r", "first"));
  ASSERT_TRUE(pf::ReplacePrivateFile(dir.fd, "r", "second, longer"));
  EXPECT_EQ(pf::ReadPrivateFile(dir.fd, "r", 1024).value_or(""), "second, longer");
  EXPECT_EQ(pf::ReadPrivateFile(dir.fd, "r", 4).error_or(0), EFBIG);
  Dir sub(root / "conversations" / "sub");
  ASSERT_GE(sub.fd, 0);
  ASSERT_TRUE(pf::ReplacePrivateFile(sub.fd, "inside", "x"));
  auto listed = pf::ListPrivateDirectory(dir.fd);
  ASSERT_TRUE(listed.has_value());
  std::vector<std::string> names;
  for (const auto& entry : *listed) {
    names.push_back(entry.name);
  }
  EXPECT_EQ(names, (std::vector<std::string>{"a", "b", "r", "sub"}));
  EXPECT_TRUE(pf::RemovePrivate(dir.fd, "sub"));
  EXPECT_TRUE(pf::RemovePrivate(dir.fd, "b"));  // the link, not what it names
  EXPECT_TRUE(pf::RemovePrivate(dir.fd, "gone"));
  EXPECT_TRUE(fs::exists(root / "conversations" / "a"));
  EXPECT_FALSE(fs::exists(root / "conversations" / "sub"));
  // Replaced in steps: written and synced beside its name, renamed over it
  // only when committed; abandoned, nothing of it stays.
  auto third = pf::WritePrivateReplacement(dir.fd, "r", "third");
  ASSERT_TRUE(third.has_value());
  EXPECT_TRUE(fs::exists(root / "conversations" / "r.tmp"));
  EXPECT_EQ(pf::ReadPrivateFile(dir.fd, "r", 1024).value_or(""), "second, longer");
  ASSERT_TRUE(pf::CommitPrivateReplacement(dir.fd, "r", *third));
  ASSERT_TRUE(pf::SyncDirectory(dir.fd));
  EXPECT_FALSE(fs::exists(root / "conversations" / "r.tmp"));
  EXPECT_EQ(pf::ReadPrivateFile(dir.fd, "r", 1024).value_or(""), "third");
  auto fourth = pf::WritePrivateReplacement(dir.fd, "r", "fourth");
  ASSERT_TRUE(fourth.has_value());
  pf::AbandonPrivateReplacement(dir.fd, "r", *fourth);
  EXPECT_FALSE(fs::exists(root / "conversations" / "r.tmp"));
  EXPECT_EQ(pf::ReadPrivateFile(dir.fd, "r", 1024).value_or(""), "third");
  // A temporary file that is no longer the private one written (a link put
  // in its place) is never renamed over the name, and goes.
  auto fifth = pf::WritePrivateReplacement(dir.fd, "r", "fifth");
  ASSERT_TRUE(fifth.has_value());
  ASSERT_EQ(::unlinkat(dir.fd, "r.tmp", 0), 0);
  ASSERT_EQ(::symlinkat("a", dir.fd, "r.tmp"), 0);
  EXPECT_EQ(pf::CommitPrivateReplacement(dir.fd, "r", *fifth).error_or(0), EPERM);
  EXPECT_FALSE(fs::is_symlink(root / "conversations" / "r.tmp"));
  EXPECT_EQ(pf::ReadPrivateFile(dir.fd, "r", 1024).value_or(""), "third");
  // nothing to rename
  EXPECT_FALSE(pf::CommitPrivateReplacement(dir.fd, "r", *fifth).has_value());
  // Nor another private-looking regular file put in its place (same owner,
  // mode and link count; made before the one written goes, so never its
  // inode): refused and left to whoever made it, also by an abandon; the
  // next write removes it.
  auto sixth = pf::WritePrivateReplacement(dir.fd, "r", "sixth");
  ASSERT_TRUE(sixth.has_value());
  auto other = pf::OpenPrivateFile(dir.fd, "other", {.write = true, .create = true});
  ASSERT_TRUE(other.has_value());
  ASSERT_EQ(::write(other->fd, "foreign", 7), 7);
  (void)::close(other->fd);
  ASSERT_EQ(::renameat(dir.fd, "other", dir.fd, "r.tmp"), 0);
  EXPECT_EQ(pf::CommitPrivateReplacement(dir.fd, "r", *sixth).error_or(0), EPERM);
  EXPECT_EQ(pf::ReadPrivateFile(dir.fd, "r.tmp", 1024).value_or(""), "foreign");
  EXPECT_EQ(pf::ReadPrivateFile(dir.fd, "r", 1024).value_or(""), "third");
  pf::AbandonPrivateReplacement(dir.fd, "r", *sixth);
  EXPECT_EQ(pf::ReadPrivateFile(dir.fd, "r.tmp", 1024).value_or(""), "foreign");
  auto seventh = pf::WritePrivateReplacement(dir.fd, "r", "seventh");
  ASSERT_TRUE(seventh.has_value());
  ASSERT_TRUE(pf::CommitPrivateReplacement(dir.fd, "r", *seventh));
  EXPECT_EQ(pf::ReadPrivateFile(dir.fd, "r", 1024).value_or(""), "seventh");
  EXPECT_FALSE(fs::exists(root / "conversations" / "r.tmp"));
  ASSERT_TRUE(pf::ReplacePrivateFile(dir.fd, "r", "second, longer"));
  const auto stamp = pf::RunningExecutableStamp();
  ASSERT_TRUE(stamp.has_value());
  EXPECT_NE(stamp.value_or(pf::ExecutableStamp{}).inode, 0U);
  fs::remove_all(root);
}

// The default case is what it was before D-105's amendment: the runtime's
// own directory and files, opened again, are the same files with the same
// modes and one link; emptying one gives its name a new private file (a
// new identity, which a record takes from that open); a record replaced
// over a name that is a link replaces the link, never writing what it
// names. Made exactly 0700 and 0600 even under a
// umask that would leave them closed to their owner.
TEST(KeptFiles, TheRuntimesOwnFilesAreAsBeforeWhateverTheUmask) {
  const fs::path root = Scratch("default");
  fs::create_directories(root);
  const mode_t old_umask = ::umask(0777);
  const auto made_dir = pf::OpenPrivateDirectory(-1, (root / "conversations").c_str());
  std::expected<pf::PrivateFile, int> made = std::unexpected(EBADF);
  if (made_dir) {
    made = pf::OpenPrivateFile(*made_dir, "slot-0.state",
                               {.write = true, .create = true, .truncate = true});
  }
  (void)::umask(old_umask);
  ASSERT_TRUE(made_dir.has_value()) << made_dir.error_or(0);
  const int dir = *made_dir;
  ASSERT_TRUE(made.has_value()) << made.error_or(0);
  ASSERT_EQ(::pwrite(made->fd, "state", 5, 0), 5);
  (void)::close(made->fd);
  struct stat st{};
  ASSERT_EQ(::stat((root / "conversations").c_str(), &st), 0);
  EXPECT_EQ(st.st_mode & 07777, 0700U);
  ASSERT_EQ(::fstatat(dir, "slot-0.state", &st, AT_SYMLINK_NOFOLLOW), 0);
  EXPECT_TRUE(S_ISREG(st.st_mode));
  EXPECT_EQ(st.st_mode & 07777, 0600U);
  EXPECT_EQ(st.st_nlink, 1U);
  EXPECT_EQ(st.st_uid, ::geteuid());
  // Opened again (as an adopted spill file is): the same file, its bytes.
  auto again = pf::OpenPrivateFile(dir, "slot-0.state", {.write = true, .create = true});
  ASSERT_TRUE(again.has_value());
  EXPECT_EQ(again->identity, made->identity);
  EXPECT_EQ(again->bytes, 5U);
  (void)::close(again->fd);
  // Emptied (as a fresh slot's is): a new private file in its place.
  auto emptied =
      pf::OpenPrivateFile(dir, "slot-0.state", {.write = true, .create = true, .truncate = true});
  ASSERT_TRUE(emptied.has_value());
  EXPECT_NE(emptied->identity, made->identity);
  EXPECT_EQ(emptied->bytes, 0U);
  (void)::close(emptied->fd);
  ASSERT_EQ(::fstatat(dir, "slot-0.state", &st, AT_SYMLINK_NOFOLLOW), 0);
  EXPECT_EQ(st.st_mode & 07777, 0600U);
  EXPECT_EQ(st.st_size, 0);
  EXPECT_EQ(st.st_nlink, 1U);
  // The directory opened again: the same one, unchanged.
  auto reopened = pf::OpenPrivateDirectory(-1, (root / "conversations").c_str());
  ASSERT_TRUE(reopened.has_value());
  EXPECT_EQ(pf::PrivacyNote(*reopened), "");
  (void)::close(*reopened);
  // A record: 0600, one link; replaced over a link, the link goes and
  // what it named is untouched.
  ASSERT_TRUE(pf::ReplacePrivateFile(dir, "slot-0.record", "record"));
  ASSERT_EQ(::fstatat(dir, "slot-0.record", &st, AT_SYMLINK_NOFOLLOW), 0);
  EXPECT_EQ(st.st_mode & 07777, 0600U);
  EXPECT_EQ(st.st_nlink, 1U);
  EXPECT_EQ(pf::PrivacyNote(dir), "");
  {
    std::ofstream(root / "elsewhere") << "untouched";
  }
  ASSERT_EQ(::unlinkat(dir, "slot-0.record", 0), 0);
  ASSERT_EQ(::symlinkat((root / "elsewhere").c_str(), dir, "slot-0.record"), 0);
  ASSERT_TRUE(pf::ReplacePrivateFile(dir, "slot-0.record", "new record"));
  EXPECT_FALSE(fs::is_symlink(root / "conversations" / "slot-0.record"));
  EXPECT_EQ(pf::ReadPrivateFile(dir, "slot-0.record", 64).value_or(""), "new record");
  std::ifstream elsewhere(root / "elsewhere");
  EXPECT_EQ(
      std::string((std::istreambuf_iterator<char>(elsewhere)), std::istreambuf_iterator<char>()),
      "untouched");
  (void)::close(dir);
  fs::remove_all(root);
}

// A name changed by another writer between the look and the open (a
// directory its owner opened up): an emptying open never empties a file
// reached through a link or another name put there meanwhile, a creating
// open never creates through a link to nothing put there meanwhile, and a
// FIFO put there meanwhile never blocks an open nor is used. A thread swaps
// the name while the opens run; what they reach is checked every time.
TEST(KeptFiles, ANameChangedDuringTheOpenIsLookedAtAgain) {
  const fs::path root = Scratch("racing");
  fs::create_directories(root);
  Dir dir(root / "d");
  ASSERT_GE(dir.fd, 0);
  const std::string victim_text = "another file's bytes";
  {
    std::ofstream(root / "d" / "victim") << victim_text;
  }
  const auto victim = [&] {
    std::ifstream in(root / "d" / "victim");
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  };
  constexpr int kOpens = 3000;
  enum class Swap { kLinkOrHardLink, kDanglingLink, kFifo };
  for (const Swap swap : {Swap::kLinkOrHardLink, Swap::kDanglingLink, Swap::kFifo}) {
    std::atomic<bool> stop{false};
    std::thread swapper([&] {
      const int d = dir.fd;
      for (std::uint64_t n = 0; !stop.load(std::memory_order_relaxed); ++n) {
        switch (swap) {
          case Swap::kLinkOrHardLink: {
            // A regular file of one link, then the victim by a symbolic
            // link or by another name, each renamed over the name at once.
            const int fd = ::openat(d, "fresh", O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
            if (fd >= 0) (void)::close(fd);
            (void)::renameat(d, "fresh", d, "name");
            if (n % 2 == 0) {
              (void)::symlinkat("victim", d, "link");
              (void)::renameat(d, "link", d, "name");
            } else {
              (void)::linkat(d, "victim", d, "hard", 0);
              (void)::renameat(d, "hard", d, "name");
            }
            break;
          }
          case Swap::kDanglingLink:
            (void)::symlinkat("made-through", d, "name");
            (void)::unlinkat(d, "name", 0);
            break;
          case Swap::kFifo: {
            const int fd = ::openat(d, "fresh", O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
            if (fd >= 0) (void)::close(fd);
            (void)::renameat(d, "fresh", d, "name");
            (void)::mkfifoat(d, "fifo", 0600);
            (void)::renameat(d, "fifo", d, "name");
            break;
          }
        }
      }
    });
    int opened = 0;
    for (int i = 0; i < kOpens; ++i) {
      std::expected<pf::PrivateFile, int> file;
      if (swap == Swap::kLinkOrHardLink) {
        file =
            pf::OpenPrivateFile(dir.fd, "name", {.write = true, .create = true, .truncate = true});
      } else if (swap == Swap::kDanglingLink) {
        file = pf::OpenPrivateFile(dir.fd, "name", {.write = true, .create = true});
        if (file) (void)::unlinkat(dir.fd, "name", 0);
      } else {
        file = pf::OpenPrivateFile(dir.fd, "name", {.write = i % 2 == 0});
      }
      if (file) {
        ++opened;
        struct stat st{};
        EXPECT_EQ(::fstat(file->fd, &st), 0);
        EXPECT_TRUE(S_ISREG(st.st_mode));
        (void)::close(file->fd);
      }
    }
    stop = true;
    swapper.join();
    EXPECT_EQ(victim(), victim_text);
    EXPECT_FALSE(fs::exists(root / "d" / "made-through"));
    // A FIFO's swap leaves the name a regular file only for an instant,
    // which every open may miss; the other swaps leave one most of the time.
    if (swap != Swap::kFifo) {
      EXPECT_GT(opened, 0);
    }
    for (const char* name : {"name", "fresh", "link", "hard", "fifo", "made-through"}) {
      (void)::unlinkat(dir.fd, name, 0);
    }
  }
  fs::remove_all(root);
}

// --------------------------------------------------------------- the record

TEST(KeptRecord, ARecordReadsBackAsWritten) {
  const kept::Record r = Sample();
  const std::string text = kept::Encode(r);
  auto back = kept::Decode(text);
  ASSERT_TRUE(back.has_value()) << back.error();
  EXPECT_EQ(*back, r);
  EXPECT_EQ(kept::Encode(*back), text);
  auto checked = *back;
  EXPECT_TRUE(kept::Check(checked, Expect(r)).has_value());
  EXPECT_EQ(checked, r);
}

// Cut short, edited anywhere (a token, the build, a digit of the digest),
// or with a field added, the record is refused: its digest says so.
TEST(KeptRecord, ATamperedOrCutRecordIsRefused) {
  const std::string text = kept::Encode(Sample());
  EXPECT_FALSE(kept::Decode(text.substr(0, text.size() - 1)).has_value());
  EXPECT_FALSE(kept::Decode(text.substr(0, text.size() / 2)).has_value());
  EXPECT_FALSE(kept::Decode("").has_value());
  for (const std::string_view from : {"100,200", "\"slot\":2", "abc;sdk"}) {
    std::string edited = text;
    const auto at = edited.find(from);
    ASSERT_NE(at, std::string::npos) << from;
    edited[at + 1] = edited[at + 1] == '9' ? '8' : '9';
    auto decoded = kept::Decode(edited);
    ASSERT_FALSE(decoded.has_value()) << from;
    EXPECT_NE(decoded.error().find("digest"), std::string::npos) << decoded.error();
  }
  std::string digest_edited = text;
  digest_edited[digest_edited.size() - 4] =
      digest_edited[digest_edited.size() - 4] == 'a' ? 'b' : 'a';
  EXPECT_FALSE(kept::Decode(digest_edited).has_value());
  // A field added, with a digest made to match: refused as not the format.
  std::string body = text.substr(0, text.size() - 77) + R"(,"extra":1})";
  const std::string digest = llmp::base::ToHex(llmp::base::Sha256().Update(body).Finish());
  body.pop_back();
  body += R"(,"digest":")" + digest + R"("})";
  auto extra = kept::Decode(body);
  ASSERT_FALSE(extra.has_value());
  EXPECT_NE(extra.error().find("fields"), std::string::npos) << extra.error();
}

// A foreign record (another build, model, layout, slot or region set), one
// whose state lies outside the layout, holds a token outside the
// vocabulary, or is past its retention is refused whole; a checkpoint that
// fails its own checks is dropped alone.
TEST(KeptRecord, AForeignOrExpiredRecordIsRefused) {
  constexpr std::int64_t kLeastMs = std::numeric_limits<std::int64_t>::min();
  constexpr std::int64_t kMostMs = std::numeric_limits<std::int64_t>::max();
  const kept::Record r = Sample();
  const auto refused = [&](auto change, std::string_view why) {
    kept::Record copy = r;
    kept::Expected expected = Expect(r);
    change(copy, expected);
    auto checked = kept::Check(copy, expected);
    ASSERT_FALSE(checked.has_value()) << why;
    EXPECT_NE(checked.error().find(why), std::string::npos) << checked.error();
  };
  refused([](kept::Record&, kept::Expected& e) { e.identity.build += "x"; }, "another build");
  refused([](kept::Record&, kept::Expected& e) { e.identity.artifact = "77"; }, "another model");
  refused([](kept::Record&, kept::Expected& e) { e.identity.drafter.clear(); }, "another model");
  refused([](kept::Record&, kept::Expected& e) { e.identity.layout += "x"; }, "layout");
  refused([](kept::Record&, kept::Expected& e) { e.slot = 1; }, "another request slot");
  refused([](kept::Record&, kept::Expected& e) { e.regions[1] += kExtent; }, "regions");
  refused([](kept::Record& c, kept::Expected&) { c.extents.push_back({1, 2, {}}); }, "outside");
  refused([](kept::Record& c, kept::Expected&) { std::swap(c.extents[0], c.extents[1]); },
          "in order");
  refused([](kept::Record& c, kept::Expected&) { c.extents.clear(); }, "keeps no state");
  refused([](kept::Record& c, kept::Expected&) { c.tokens.push_back(129280); }, "vocabulary");
  refused([](kept::Record&, kept::Expected& e) { e.context = 5; }, "context");
  refused([](kept::Record& c, kept::Expected&) { c.cursor = 7; }, "cursor");
  refused([](kept::Record&, kept::Expected& e) { e.now_unix_ms += e.retention_ms; }, "retention");
  // Edited times at the ends of their range overflow nothing.
  refused([](kept::Record& c, kept::Expected&) { c.used_unix_ms = kLeastMs; }, "retention");
  refused([](kept::Record& c, kept::Expected&) { c.used_unix_ms = kMostMs; }, "future");
  refused([](kept::Record& c, kept::Expected&) { c.files[0].bytes += 4096; }, "size");
  // A checkpoint past the conversation, expired or outside the layout goes alone.
  for (const auto& change : std::vector<void (*)(kept::Checkpoint&)>{
           [](kept::Checkpoint& c) { c.position = 7; },
           [](kept::Checkpoint& c) { c.created_unix_ms -= 25LL * 3600 * 1000; },
           [](kept::Checkpoint& c) { c.created_unix_ms = kLeastMs; },
           [](kept::Checkpoint& c) { c.created_unix_ms = kMostMs; },
           [](kept::Checkpoint& c) { c.ranges[0].offset = 5 * kExtent; },
           [](kept::Checkpoint& c) { c.file = kept::CheckpointFileName(3, 7); },
           [](kept::Checkpoint& c) { c.digests.clear(); },
           [](kept::Checkpoint& c) { c.cursor = 5; }}) {
    kept::Record copy = r;
    change(copy.checkpoints[0]);
    std::vector<std::string> dropped;
    EXPECT_TRUE(kept::Check(copy, Expect(r), &dropped).has_value());
    EXPECT_TRUE(copy.checkpoints.empty());
    EXPECT_EQ(dropped.size(), 1U);
  }
}

// What a record's extents do not cover of its spill file, region by
// region in file order, adjacent runs joined (across regions too).
TEST(KeptRecord, UnlistedPlacesAreWhatItsExtentsDoNotCover) {
  kept::Record r = Sample();  // regions of 4 and 2 extents; (0,0), (0,1), (1,0) listed
  const auto places = [](const kept::Record& record) {
    std::vector<std::pair<std::uint64_t, std::uint64_t>> out;
    for (const kept::Place& p : kept::UnlistedPlaces(record)) {
      out.emplace_back(p.offset, p.bytes);
    }
    return out;
  };
  using Runs = std::vector<std::pair<std::uint64_t, std::uint64_t>>;
  EXPECT_EQ(places(r), (Runs{{2 * kExtent, 2 * kExtent}, {5 * kExtent, kExtent}}));
  r.extents = {{.region = 0, .index = 0, .digest = {}}};
  EXPECT_EQ(places(r), (Runs{{kExtent, 5 * kExtent}}));
  r.extents = {{.region = 0, .index = 3, .digest = {}}, {.region = 1, .index = 1, .digest = {}}};
  EXPECT_EQ(places(r), (Runs{{0, 3 * kExtent}, {4 * kExtent, kExtent}}));
  for (std::uint32_t i = 0; i < 4; ++i) {
    r.extents.push_back({.region = 0, .index = i, .digest = {}});
  }
  r.extents.push_back({.region = 1, .index = 0, .digest = {}});
  EXPECT_TRUE(places(r).empty());
}

TEST(KeptRecord, FileNamesAreTheSlotsOwn) {
  EXPECT_EQ(kept::StateFileName(3), "slot-3.state");
  EXPECT_EQ(kept::RecordFileName(3), "slot-3.record");
  EXPECT_EQ(kept::CheckpointFileName(3, 12), "slot-3.turn-12.state");
  const auto turn = kept::ParseFileName("slot-3.turn-12.state");
  EXPECT_TRUE(turn.has_value() && turn->kind == kept::NameOf::Kind::kCheckpoint &&
              turn->slot == 3 && turn->serial == 12);
  const auto record = kept::ParseFileName("slot-0.record");
  EXPECT_TRUE(record.has_value() && record->kind == kept::NameOf::Kind::kRecord &&
              record->slot == 0);
  for (const std::string_view bad :
       {"slot-.state", "slot-03.state", "slot-3.turn-.state", "slot-3.record.tmp", "x", "slot-3"}) {
    EXPECT_FALSE(kept::ParseFileName(bad).has_value()) << bad;
  }
  EXPECT_TRUE(kept::ValidDirectoryName("8a355bfb0f"));
  EXPECT_FALSE(kept::ValidDirectoryName("../x"));
  EXPECT_FALSE(kept::ValidDirectoryName("ABC"));
  EXPECT_FALSE(kept::ValidDirectoryName(""));
}

// --------------------------------------------------------------- the keeper

TEST(StateKeeper, QuietKeepsOnlyTheLatestQueuedRecordPerSlot) {
  const fs::path root = Scratch("bounded-keeper");
  fs::create_directories(root);
  Dir model(root / "model");
  ASSERT_GE(model.fd, 0);
  const pf::FileIdentity first = WriteExtents(model.fd, "slot-1.state", 1, 5);
  const pf::FileIdentity second = WriteExtents(model.fd, "slot-2.state", 1, 9);
  StateKeeper keeper(1, {});
  const std::size_t index = keeper.AddModel(::dup(model.fd));
  const StateKeeper::Quiet quiet(&keeper);
  kept::Record r = Sample();
  r.slot = 1;
  r.files = {{.name = kept::StateFileName(1), .id = Id(first), .bytes = kExtent}};
  r.regions = {kExtent};
  r.extents = {{.region = 0, .index = 0, .digest = {}}};
  r.checkpoints.clear();
  for (int i = 0; i < 64; ++i) {
    r.tokens = {i};
    keeper.Keep(index, r);
    // One being hashed and at most one waiting, whichever the thread's
    // ordering; Quiet prevents either from finishing its hash.
    EXPECT_LE(keeper.pending(), 2U);
  }
  kept::Record other = r;
  other.slot = 2;
  other.files = {{.name = kept::StateFileName(2), .id = Id(second), .bytes = kExtent}};
  keeper.Keep(index, other);
  EXPECT_LE(keeper.pending(), 3U);
  keeper.Invalidate(index, 1);
  // The invalidated slot's queued copy is gone; another slot's stays.
  EXPECT_LE(keeper.pending(), 2U);
  r.tokens = {1234};
  keeper.Keep(index, r);
  // Drain bypasses Quiet; the latest record and the other slot both survive.
  ASSERT_TRUE(keeper.Drain(std::chrono::steady_clock::now() + std::chrono::seconds(30)));
  auto text = pf::ReadPrivateFile(model.fd, "slot-1.record", kept::kMostRecordBytes);
  ASSERT_TRUE(text.has_value());
  auto written = kept::Decode(*text);
  ASSERT_TRUE(written.has_value()) << written.error();
  EXPECT_EQ(written->tokens, r.tokens);
  EXPECT_TRUE(keeper.Kept(index, 1));
  EXPECT_TRUE(keeper.Kept(index, 2));
  EXPECT_EQ(keeper.pending(), 0U);
  fs::remove_all(root);
}

TEST(StateKeeper, ARecordDescribesItsFilesExactly) {
  const fs::path root = Scratch("keeper");
  fs::create_directories(root);
  Dir model(root / "model");
  ASSERT_GE(model.fd, 0);
  const pf::FileIdentity state = WriteExtents(model.fd, "slot-1.state", 3, 5);
  const pf::FileIdentity turn = WriteExtents(model.fd, "slot-1.turn-4.state", 1, 9);
  StateKeeper keeper(2, {});
  const std::size_t index = keeper.AddModel(::dup(model.fd));
  kept::Record r = Sample();
  r.slot = 1;
  r.files = {{.name = kept::StateFileName(1), .id = Id(state), .bytes = 3 * kExtent}};
  r.regions = {3 * kExtent};
  r.extents = {{.region = 0, .index = 0, .digest = {}}, {.region = 0, .index = 2, .digest = {}}};
  r.checkpoints[0].file = "slot-1.turn-4.state";
  r.checkpoints[0].id = Id(turn);
  r.checkpoints[0].file_bytes = kExtent;
  r.checkpoints[0].ranges = {{.region = 0, .offset = 0, .bytes = kExtent}};
  r.checkpoints[0].digests.clear();
  keeper.Keep(index, r);
  ASSERT_TRUE(keeper.Drain(std::chrono::steady_clock::now() + std::chrono::seconds(30)));
  EXPECT_TRUE(keeper.Kept(index, 1));
  auto text = pf::ReadPrivateFile(model.fd, "slot-1.record", kept::kMostRecordBytes);
  ASSERT_TRUE(text.has_value());
  auto written = kept::Decode(*text);
  ASSERT_TRUE(written.has_value()) << written.error();
  // Its digests are the files' own.
  auto file = pf::OpenPrivateFile(model.fd, "slot-1.state", {.direct = true});
  ASSERT_TRUE(file.has_value());
  const std::array<kept::Place, 2> places = {
      {{.offset = 0, .bytes = kExtent}, {.offset = 2 * kExtent, .bytes = kExtent}}};
  const auto digests = HashPlaces(file->fd, places, 2);
  (void)::close(file->fd);
  ASSERT_EQ(written->extents.size(), 2U);
  EXPECT_EQ(std::optional(written->extents[0].digest), digests[0]);
  EXPECT_EQ(std::optional(written->extents[1].digest), digests[1]);
  ASSERT_EQ(written->checkpoints.size(), 1U);
  EXPECT_EQ(written->checkpoints[0].digests.size(), 1U);
  EXPECT_EQ(keeper.stats().written, 1U);
  EXPECT_GE(keeper.stats().hashed_bytes, 3 * kExtent);

  // Invalidated: the record goes at once.
  keeper.Invalidate(index, 1);
  EXPECT_FALSE(keeper.Kept(index, 1));
  EXPECT_FALSE(fs::exists(root / "model" / "slot-1.record"));

  // A record whose file is not the one it was made for is not written.
  kept::Record wrong = r;
  wrong.files[0].id.inode += 1;
  keeper.Keep(index, wrong);
  ASSERT_TRUE(keeper.Drain(std::chrono::steady_clock::now() + std::chrono::seconds(30)));
  EXPECT_FALSE(fs::exists(root / "model" / "slot-1.record"));
  EXPECT_EQ(keeper.stats().failed, 1U);

  // Invalidated after it was queued (the file about to change): never
  // written, whatever the order the worker meets them in.
  for (int i = 0; i < 20; ++i) {
    keeper.Keep(index, r);
    keeper.Invalidate(index, 1);
  }
  ASSERT_TRUE(keeper.Drain(std::chrono::steady_clock::now() + std::chrono::seconds(60)));
  EXPECT_FALSE(fs::exists(root / "model" / "slot-1.record"));
  EXPECT_FALSE(keeper.Kept(index, 1));
  EXPECT_EQ(keeper.stats().written, 1U);
  EXPECT_GE(keeper.stats().stale, 20U);
  // A record made stale after it was written beside its name leaves no
  // temporary file.
  EXPECT_FALSE(fs::exists(root / "model" / "slot-1.record.tmp"));

  // Written again once nothing invalidates it; then the file changes
  // (Invalidate first, as the driver does) and the slot is kept again.
  keeper.Keep(index, r);
  ASSERT_TRUE(keeper.Drain(std::chrono::steady_clock::now() + std::chrono::seconds(30)));
  EXPECT_TRUE(fs::exists(root / "model" / "slot-1.record"));
  EXPECT_FALSE(fs::exists(root / "model" / "slot-1.record.tmp"));
  EXPECT_EQ(keeper.stats().written, 2U);
  keeper.Invalidate(index, 1);
  EXPECT_FALSE(fs::exists(root / "model" / "slot-1.record"));
  fs::remove_all(root);
}

// Adopting a record empties every extent of its spill file it does not
// list (D-105): a record whose removal never reached the disk can come
// back after a crash while the file's other extents hold a later
// conversation's state. The listed extents stay as they were; a file that
// is not the record's is refused.
TEST(StateKeeper, AdoptionEmptiesWhatTheRecordDoesNotList) {
  const fs::path root = Scratch("unlisted");
  fs::create_directories(root);
  Dir model(root / "model");
  ASSERT_GE(model.fd, 0);
  const pf::FileIdentity state = WriteExtents(model.fd, "slot-0.state", 4, 3);
  kept::Record r = Sample();
  r.slot = 0;
  r.files = {{.name = kept::StateFileName(0), .id = Id(state), .bytes = 4 * kExtent}};
  r.regions = {3 * kExtent, kExtent};
  r.extents = {{.region = 0, .index = 0, .digest = {}}, {.region = 0, .index = 2, .digest = {}}};
  r.checkpoints.clear();
  const auto hash = [&](std::span<const kept::Place> places) {
    auto file = pf::OpenPrivateFile(model.fd, "slot-0.state", {.direct = true});
    EXPECT_TRUE(file.has_value());
    auto digests = HashPlaces(file->fd, places, 2);
    (void)::close(file->fd);
    return digests;
  };
  const std::array<kept::Place, 4> all = {{{.offset = 0, .bytes = kExtent},
                                           {.offset = kExtent, .bytes = kExtent},
                                           {.offset = 2 * kExtent, .bytes = kExtent},
                                           {.offset = 3 * kExtent, .bytes = kExtent}}};
  const auto before = hash(all);
  // Not the file the record names: refused, nothing emptied.
  kept::Record other = r;
  other.files[0].id.inode += 1;
  EXPECT_FALSE(llmp::runtime::EmptyUnlisted(model.fd, other).empty());
  EXPECT_EQ(hash(all), before);
  ASSERT_EQ(llmp::runtime::EmptyUnlisted(model.fd, r), "");
  const auto after = hash(all);
  const std::vector<std::byte> zeros(kExtent);
  const auto zero = llmp::base::Sha256().Update(std::span<const std::byte>(zeros)).Finish();
  EXPECT_EQ(after[0], before[0]);
  EXPECT_EQ(after[1], std::optional(zero));
  EXPECT_EQ(after[2], before[2]);
  EXPECT_EQ(after[3], std::optional(zero));
  // Its size and identity stay: the record still names it.
  auto file = pf::OpenPrivateFile(model.fd, "slot-0.state", {});
  ASSERT_TRUE(file.has_value());
  EXPECT_EQ(Id(file->identity), r.files[0].id);
  EXPECT_EQ(file->bytes, 4 * kExtent);
  (void)::close(file->fd);
  fs::remove_all(root);
}

// The files' owner, modes and links are the machine owner's (D-105 as
// amended on 2026-10-09): a model's directory and files found opened up,
// with another name or reached through a link, are kept, described and
// emptied as any are, while the record's own checks hold: the file must be
// the one the record names, even through a link that leads elsewhere.
TEST(StateKeeper, FilesFoundOpenedUpAreUsedAndTheRecordsChecksHold) {
  const fs::path root = Scratch("opened");
  fs::create_directories(root);
  Dir model(root / "model");
  ASSERT_GE(model.fd, 0);
  const pf::FileIdentity state = WriteExtents(model.fd, "slot-0.state", 4, 3);
  ASSERT_EQ(::fchmodat(model.fd, "slot-0.state", 0666, 0), 0);
  ASSERT_EQ(::linkat(model.fd, "slot-0.state", model.fd, "second-name", 0), 0);
  ASSERT_EQ(::fchmod(model.fd, 0755), 0);
  EXPECT_THAT(pf::PrivacyNote(model.fd), HasSubstr("mode 0755, not 0700"));
  EXPECT_THAT(pf::PrivacyNote(model.fd), HasSubstr("2 files with another owner"));
  kept::Record r = Sample();
  r.slot = 0;
  r.files = {{.name = kept::StateFileName(0), .id = Id(state), .bytes = 4 * kExtent}};
  r.regions = {3 * kExtent, kExtent};
  r.extents = {{.region = 0, .index = 0, .digest = {}}, {.region = 0, .index = 2, .digest = {}}};
  r.checkpoints.clear();
  ASSERT_EQ(llmp::runtime::EmptyUnlisted(model.fd, r), "");
  StateKeeper keeper(1, {});
  const std::size_t index = keeper.AddModel(::dup(model.fd));
  keeper.Keep(index, r);
  ASSERT_TRUE(keeper.Drain(std::chrono::steady_clock::now() + std::chrono::seconds(30)));
  EXPECT_TRUE(keeper.Kept(index, 0));
  EXPECT_TRUE(fs::exists(root / "model" / "slot-0.record"));
  // A link in the file's place leading to another file is not the file the
  // record names.
  const pf::FileIdentity elsewhere = WriteExtents(model.fd, "elsewhere", 4, 4);
  ASSERT_NE(Id(elsewhere), Id(state));
  ASSERT_EQ(::unlinkat(model.fd, "slot-0.state", 0), 0);
  ASSERT_EQ(::symlinkat("elsewhere", model.fd, "slot-0.state"), 0);
  EXPECT_FALSE(llmp::runtime::EmptyUnlisted(model.fd, r).empty());
  fs::remove_all(root);
}

}  // namespace

TEST(StateKeeper, PendingTokenCopyKeepsItsChargeThroughInvalidationAndRetirement) {
  namespace rt = llmp::runtime;
  const fs::path root = Scratch("charged-keeper");
  fs::create_directories(root);
  Dir model(root / "model");
  ASSERT_GE(model.fd, 0);
  rt::RequestMemory memory(4096);
  {
    StateKeeper keeper(1, {});
    const std::size_t index = keeper.AddModel(::dup(model.fd));
    const StateKeeper::Quiet quiet(&keeper);
    const auto file = WriteExtents(model.fd, "slot-1.state", 1, 5);
    for (int serial = 0; serial < 32; ++serial) {
      rt::MemoryCharge tokens;
      kept::Record r = Sample();
      std::vector<std::int32_t>().swap(r.tokens);
      ASSERT_TRUE(rt::ReserveTokenStorage(r.tokens, tokens, memory, 64));
      r.tokens.assign(64, serial);
      r.slot = 1;
      r.files = {{.name = kept::StateFileName(1), .id = Id(file), .bytes = kExtent}};
      r.regions = {kExtent};
      r.extents = {{.region = 0, .index = 0, .digest = {}}};
      r.checkpoints.clear();
      keeper.Keep(index, std::move(r), std::move(tokens));
      EXPECT_LE(memory.used(), 2 * 64 * sizeof(std::int32_t));
    }
    keeper.Invalidate(index, 1);
    ASSERT_TRUE(keeper.Drain(std::chrono::steady_clock::now() + std::chrono::seconds(30)));
  }
  EXPECT_EQ(memory.used(), 0U);
  fs::remove_all(root);
}

TEST(KeptRecord, DecodingFundsIndependentTokenCapacityBeforeAllocation) {
  namespace rt = llmp::runtime;
  const auto text = kept::Encode(Sample());
  rt::RequestMemory tiny(1);
  rt::MemoryCharge refusal;
  // A refusal for memory says so (a later start may adopt the record); a
  // bad record's does not.
  bool memory_refused = false;
  EXPECT_FALSE(kept::Decode(text, &tiny, &refusal, &memory_refused));
  EXPECT_TRUE(memory_refused);
  EXPECT_EQ(tiny.used(), 0U);
  bool not_memory = false;
  EXPECT_FALSE(kept::Decode(text.substr(0, text.size() - 1), &tiny, &refusal, &not_memory));
  EXPECT_FALSE(not_memory);
  rt::RequestMemory roomy(4096);
  rt::MemoryCharge tokens;
  {
    bool refused = false;
    auto record = kept::Decode(text, &roomy, &tokens, &refused);
    ASSERT_TRUE(record);
    EXPECT_FALSE(refused);
    EXPECT_EQ(tokens.bytes(), record->tokens.capacity() * sizeof(std::int32_t));
    EXPECT_EQ(roomy.used(), tokens.bytes());
  }
  tokens.Reset();
  EXPECT_EQ(roomy.used(), 0U);
}

// ------------------------------------------------- version 2: a file a stage
//
// The shared runner's record (D-112): one spill file per stage, each its
// stage's one region; the reader takes both versions, a model holds a
// record to the version it writes, and the other is refused, not migrated.

namespace {

kept::FileId FakeId(std::uint64_t inode) {
  return {.device = 66306, .inode = inode, .generation = 7};
}

kept::Record SampleStages() {
  kept::Record r = Sample();
  r.version = kept::kStageFilesVersion;
  r.identity.layout = "llmp-state-v2:0123456789abcdef";
  r.files = {{.name = kept::StateFileName(2, 0), .id = FakeId(111), .bytes = 4 * kExtent},
             {.name = kept::StateFileName(2, 1), .id = FakeId(112), .bytes = 2 * kExtent},
             {.name = kept::StateFileName(2, 2), .id = FakeId(113), .bytes = kExtent}};
  r.regions = {4 * kExtent, 2 * kExtent, kExtent};
  r.extents = {{.region = 0, .index = 0, .digest = {}},
               {.region = 0, .index = 1, .digest = {}},
               {.region = 1, .index = 0, .digest = {}},
               {.region = 2, .index = 0, .digest = {}}};
  r.extents[3].digest[5] = 0x5c;
  // A checkpoint's ranges name the stage as their region.
  r.checkpoints[0].ranges = {{.region = 2, .offset = 0, .bytes = 5000}};
  r.checkpoints[0].footprint = {{.region = 0, .offset = 0, .bytes = kExtent},
                                {.region = 1, .offset = 0, .bytes = kExtent}};
  return r;
}

kept::Expected ExpectStages(const kept::Record& r) {
  kept::Expected e = Expect(r);
  e.version = kept::kStageFilesVersion;
  e.layouts = {(4 * kExtent) - 4096, 2 * kExtent, kExtent - 8192};
  return e;
}

// A record's text without its digest, and that text sealed with a digest
// that matches it (a forger's, so only the format's own checks refuse it).
std::string Unsealed(const std::string& text) { return text.substr(0, text.size() - 77) + "}"; }
std::string Sealed(std::string body) {
  const std::string digest = llmp::base::ToHex(llmp::base::Sha256().Update(body).Finish());
  body.pop_back();
  return body + R"(,"digest":")" + digest + R"("})";
}

}  // namespace

TEST(KeptRecordStages, AStageRecordReadsBackAsWritten) {
  const kept::Record r = SampleStages();
  const std::string text = kept::Encode(r);
  EXPECT_NE(text.find(R"("version":2,)"), std::string::npos);
  EXPECT_NE(text.find(R"("stages":[{"file":"slot-2.state",)"), std::string::npos);
  EXPECT_NE(text.find(R"({"file":"slot-2.stage-2.state",)"), std::string::npos);
  EXPECT_EQ(text.find(R"("regions")"), std::string::npos);
  auto back = kept::Decode(text);
  ASSERT_TRUE(back.has_value()) << back.error();
  EXPECT_EQ(*back, r);
  EXPECT_EQ(kept::Encode(*back), text);
  auto checked = *back;
  auto ok = kept::Check(checked, ExpectStages(r));
  EXPECT_TRUE(ok.has_value()) << (ok ? std::string() : ok.error());
  EXPECT_EQ(checked, r);
  // Its text unsealed and sealed again: the same text.
  EXPECT_EQ(Sealed(Unsealed(text)), text);
}

// A model keeps the version it writes: a record of the other version is
// refused (and so discarded), whichever way round; a version neither
// reads is refused by the reader.
TEST(KeptRecordStages, EachVersionIsRefusedWhereTheOtherIsKept) {
  kept::Record stages = SampleStages();
  kept::Expected one_file = ExpectStages(stages);
  one_file.version = kept::kOneFileVersion;
  auto refused = kept::Check(stages, one_file);
  ASSERT_FALSE(refused.has_value());
  EXPECT_NE(refused.error().find("format version 2"), std::string::npos) << refused.error();
  // An old record (one spill file a slot) where the shared runner keeps
  // version 2: decoded, then refused.
  const std::string old_text = kept::Encode(Sample());
  auto old = kept::Decode(old_text);
  ASSERT_TRUE(old.has_value()) << old.error();
  EXPECT_EQ(old->version, kept::kOneFileVersion);
  kept::Expected stage_files = Expect(Sample());
  stage_files.version = kept::kStageFilesVersion;
  auto old_refused = kept::Check(*old, stage_files);
  ASSERT_FALSE(old_refused.has_value());
  EXPECT_NE(old_refused.error().find("format version 1"), std::string::npos) << old_refused.error();
  // A third version, sealed as a writer of it would: refused unread.
  std::string body = Unsealed(kept::Encode(stages));
  const auto at = body.find(R"("version":2)");
  ASSERT_NE(at, std::string::npos);
  body[at + 10] = '3';
  auto third = kept::Decode(Sealed(body));
  ASSERT_FALSE(third.has_value());
  EXPECT_NE(third.error().find("version"), std::string::npos) << third.error();
}

// A stage record naming another slot's or stage's file, holding another
// number of stages, a file not the size of its region or an extent past
// its stages is refused whole.
TEST(KeptRecordStages, AStageRecordNamingAnotherFileOrStageIsRefused) {
  const kept::Record r = SampleStages();
  const auto refused = [&](auto change, std::string_view why) {
    kept::Record copy = r;
    kept::Expected expected = ExpectStages(r);
    change(copy, expected);
    auto checked = kept::Check(copy, expected);
    ASSERT_FALSE(checked.has_value()) << why;
    EXPECT_NE(checked.error().find(why), std::string::npos) << checked.error();
  };
  refused([](kept::Record& c, kept::Expected&) { c.files[1].name = kept::StateFileName(2, 2); },
          "stage's file");
  refused([](kept::Record& c, kept::Expected&) { c.files[0].name = kept::StateFileName(3); },
          "stage's file");
  refused([](kept::Record& c, kept::Expected&) { std::swap(c.files[1], c.files[2]); },
          "stage's file");
  refused(
      [](kept::Record& c, kept::Expected&) {
        c.files.pop_back();
        c.regions.pop_back();
      },
      "regions");
  refused([](kept::Record& c, kept::Expected&) { c.files.pop_back(); }, "stages");
  refused([](kept::Record& c, kept::Expected&) { c.files[1].bytes += kExtent; }, "its region");
  refused([](kept::Record& c, kept::Expected&) { c.extents.push_back({3, 0, {}}); }, "outside");
  refused([](kept::Record& c, kept::Expected&) { c.extents.push_back({2, 1, {}}); }, "outside");
  refused([](kept::Record&, kept::Expected& e) { e.identity.layout += "x"; }, "layout");
}

// The reader holds a stage record to its own fields: none of version 1's,
// 1 to 16 stages, each of exactly its five fields.
TEST(KeptRecordStages, AStageRecordsShapeIsStrict) {
  const kept::Record r = SampleStages();
  const std::string body = Unsealed(kept::Encode(r));
  const auto decodes = [](const std::string& text, std::string_view why) {
    auto decoded = kept::Decode(text);
    ASSERT_FALSE(decoded.has_value()) << why;
    EXPECT_NE(decoded.error().find(why), std::string::npos) << decoded.error();
  };
  {
    std::string forged = body;
    const auto at = forged.find(R"("stages":)");
    ASSERT_NE(at, std::string::npos);
    forged.insert(at, R"("file":"slot-2.state",)");
    decodes(Sealed(forged), "fields");
  }
  {
    std::string forged = body;
    const std::string first = R"("file_bytes":8388608})";
    const auto at = forged.find(first);
    ASSERT_NE(at, std::string::npos);
    forged.replace(at, first.size(), R"("file_bytes":8388608,"x":1})");
    decodes(Sealed(forged), "fields");
  }
  kept::Record none = r;
  none.files.clear();
  decodes(kept::Encode(none), "stages");
  kept::Record many = r;
  many.files.assign(kept::kMostStages + 1, r.files[2]);
  decodes(kept::Encode(many), "stages");
}

TEST(KeptRecordStages, StageFilesAreNamedAndPlacedByStage) {
  EXPECT_EQ(kept::StateFileName(3, 0), "slot-3.state");
  EXPECT_EQ(kept::StateFileName(3, 2), "slot-3.stage-2.state");
  const auto stage = kept::ParseFileName("slot-3.stage-2.state");
  EXPECT_TRUE(stage.has_value() && stage->kind == kept::NameOf::Kind::kState && stage->slot == 3 &&
              stage->stage == 2);
  const auto target = kept::ParseFileName("slot-3.state");
  EXPECT_TRUE(target.has_value() && target->kind == kept::NameOf::Kind::kState &&
              target->stage == 0);
  for (const std::string_view bad :
       {"slot-3.stage-0.state", "slot-3.stage-02.state", "slot-3.stage-.state", "slot-3.stage-2",
        "slot-3.stage-2x.state", "slot-3.stage-2.turn-1.state"}) {
    EXPECT_FALSE(kept::ParseFileName(bad).has_value()) << bad;
  }
  // Each extent in its stage's file, from 0; what each file's listed
  // extents leave, its own.
  const kept::Record r = SampleStages();
  const auto at = kept::ExtentPlace(r, {.region = 1, .index = 0, .digest = {}});
  EXPECT_EQ(at.file, 1U);
  EXPECT_EQ(at.offset, 0U);
  const auto later = kept::ExtentPlace(r, {.region = 0, .index = 1, .digest = {}});
  EXPECT_EQ(later.file, 0U);
  EXPECT_EQ(later.offset, kExtent);
  const auto places = [](const kept::Record& record, std::size_t file) {
    std::vector<std::pair<std::uint64_t, std::uint64_t>> out;
    for (const kept::Place& p : kept::UnlistedPlaces(record, file)) {
      out.emplace_back(p.offset, p.bytes);
    }
    return out;
  };
  using Runs = std::vector<std::pair<std::uint64_t, std::uint64_t>>;
  EXPECT_EQ(places(r, 0), (Runs{{2 * kExtent, 2 * kExtent}}));
  EXPECT_EQ(places(r, 1), (Runs{{kExtent, kExtent}}));
  EXPECT_TRUE(places(r, 2).empty());
  EXPECT_TRUE(places(r, 3).empty());
  // Version 1: one file, the regions one after another.
  const kept::Record one = Sample();
  const auto old = kept::ExtentPlace(one, {.region = 1, .index = 0, .digest = {}});
  EXPECT_EQ(old.file, 0U);
  EXPECT_EQ(old.offset, 4 * kExtent);
  EXPECT_TRUE(places(one, 1).empty());
}

// The keeper hashes each stage's file at its own extents; the start's
// check reads each the same way and refuses a file that changed; adoption
// empties what each file's record does not list.
TEST(StateKeeperStages, AStageRecordDescribesEachFileExactly) {
  const fs::path root = Scratch("stage-keeper");
  fs::create_directories(root);
  Dir model(root / "model");
  ASSERT_GE(model.fd, 0);
  const pf::FileIdentity target = WriteExtents(model.fd, "slot-1.state", 2, 5);
  const pf::FileIdentity drafter = WriteExtents(model.fd, "slot-1.stage-1.state", 2, 9);
  StateKeeper keeper(2, {});
  const std::size_t index = keeper.AddModel(::dup(model.fd));
  kept::Record r = SampleStages();
  r.slot = 1;
  r.files = {{.name = kept::StateFileName(1, 0), .id = Id(target), .bytes = 2 * kExtent},
             {.name = kept::StateFileName(1, 1), .id = Id(drafter), .bytes = 2 * kExtent}};
  r.regions = {2 * kExtent, 2 * kExtent};
  r.extents = {{.region = 0, .index = 0, .digest = {}},
               {.region = 0, .index = 1, .digest = {}},
               {.region = 1, .index = 1, .digest = {}}};
  r.checkpoints.clear();
  keeper.Keep(index, r);
  ASSERT_TRUE(keeper.Drain(std::chrono::steady_clock::now() + std::chrono::seconds(30)));
  auto text = pf::ReadPrivateFile(model.fd, "slot-1.record", kept::kMostRecordBytes);
  ASSERT_TRUE(text.has_value());
  auto written = kept::Decode(*text);
  ASSERT_TRUE(written.has_value()) << written.error();
  EXPECT_EQ(written->version, kept::kStageFilesVersion);
  const auto digests_of = [&](const char* name, std::span<const kept::Place> places) {
    auto file = pf::OpenPrivateFile(model.fd, name, {.direct = true});
    EXPECT_TRUE(file.has_value());
    auto digests = HashPlaces(file->fd, places, 2);
    (void)::close(file->fd);
    return digests;
  };
  const std::array<kept::Place, 2> both = {
      {{.offset = 0, .bytes = kExtent}, {.offset = kExtent, .bytes = kExtent}}};
  const auto target_digests = digests_of("slot-1.state", both);
  const auto drafter_digests = digests_of("slot-1.stage-1.state", both);
  ASSERT_EQ(written->extents.size(), 3U);
  EXPECT_EQ(std::optional(written->extents[0].digest), target_digests[0]);
  EXPECT_EQ(std::optional(written->extents[1].digest), target_digests[1]);
  EXPECT_EQ(std::optional(written->extents[2].digest), drafter_digests[1]);
  EXPECT_NE(target_digests[1], drafter_digests[1]);
  EXPECT_EQ(llmp::runtime::VerifyKeptState(model.fd, *written), "");

  // A stage's file that is not the one named, or that changed since: refused.
  kept::Record wrong = *written;
  wrong.files[1].id.inode += 1;
  EXPECT_FALSE(llmp::runtime::VerifyKeptState(model.fd, wrong).empty());
  kept::Record changed = *written;
  changed.extents[2].digest[0] = static_cast<std::uint8_t>(changed.extents[2].digest[0] ^ 1U);
  EXPECT_NE(llmp::runtime::VerifyKeptState(model.fd, changed).find("digests differ"),
            std::string::npos);

  // Adoption empties each file's unlisted extents: the drafter's first.
  ASSERT_EQ(llmp::runtime::EmptyUnlisted(model.fd, *written), "");
  const std::vector<std::byte> zeros(kExtent);
  const auto zero = llmp::base::Sha256().Update(std::span<const std::byte>(zeros)).Finish();
  EXPECT_EQ(digests_of("slot-1.state", both), target_digests);
  const auto after = digests_of("slot-1.stage-1.state", both);
  EXPECT_EQ(after[0], std::optional(zero));
  EXPECT_EQ(after[1], drafter_digests[1]);
  EXPECT_EQ(llmp::runtime::VerifyKeptState(model.fd, *written), "");
  fs::remove_all(root);
}

// ------------------------------------- review: the formats against crafts
//
// Version 1's bytes are M3's exactly (a record the M3 runtime wrote reads
// back, and this build writes the same bytes for it); a version 2 record
// a forger sealed is refused whenever it names a file the model's
// directory does not hold for that slot and stage, holds extents outside
// its files, or bends the shape the reader expects.

namespace {

// The M3 runtime's encoding of Sample() (format version 1), byte for byte.
constexpr std::string_view kVersionOneSample =
    R"({"format":"llmp-kept-conversation","version":1,)"
    R"("build":"0.1.0;abc;sdk;aarch64;exe=1:2:3:4:5.000000006","artifact":"8a355bfb",)"
    R"("drafter":"dd2d3f9c",)"
    R"("layout":"deepseek4-state/1;context=8704;dspark=1;window=ring;regions=t:1:2","slot":2,)"
    R"("file":"slot-2.state","device":66306,"inode":1234567,"generation":99,)"
    R"("file_bytes":12582912,"regions":[8388608,4194304],)"
    R"("extents":[[0,0,"0000000000000000000000000000000000000000000000000000000000000000"],)"
    R"([0,1,"ab00000000000000000000000000000000000000000000000000000000000000"],)"
    R"([1,0,"0000000000000000000000000000000000000000000000000000000000000000"]],)"
    R"("tokens":[1,2,3,100,200,7],"cursor":1,)"
    R"("decoding":["0000000000000003","0000000000000002","0000000000000003",)"
    R"("3ff28f5c28f5c28f","0000000000000000","0000000000000000","0000000000000000",)"
    R"("0000000000000005","4000000000000000","0000000000000009","0000000000000000"],)"
    R"("used_unix_ms":1790000000000,)"
    R"("checkpoints":[{"file":"slot-2.turn-7.state","device":66306,"inode":7654321,)"
    R"("generation":null,"file_bytes":8192,"position":4,"created_unix_ms":1789999000000,)"
    R"("ranges":[[0,4194304,5000]],"footprint":[[0,0,2097152]],"cursor":0,)"
    R"("decoding":["0000000000000003","0000000000000002","0000000000000003",)"
    R"("3ff28f5c28f5c28f","0000000000000000","0000000000000000","0000000000000000",)"
    R"("0000000000000005","4000000000000000","0000000000000009","0000000000000000"],)"
    R"("digests":["0000000000000000000000000000000000000000000000000000000000000000"]}],)"
    R"("digest":"170901a370c09a1b9cb691ef8ed0b4cf339c2bda4f88579fa4aabd7246b87ba6"})";

// `text` with its first `from` replaced by `to`, sealed again.
std::string Forged(const std::string& text, std::string_view from, std::string_view to) {
  std::string body = Unsealed(text);
  const auto at = body.find(from);
  EXPECT_NE(at, std::string::npos) << from;
  if (at != std::string::npos) body.replace(at, from.size(), to);
  return Sealed(body);
}

TEST(StateKeeperStages, ChangedProgressControlCannotReuseTheOldKeptRecord) {
  const fs::path root = Scratch("stage-progress-control");
  fs::create_directories(root);
  Dir model(root / "model");
  ASSERT_GE(model.fd, 0);
  const pf::FileIdentity target = WriteExtents(model.fd, "slot-1.state", 1, 5);
  const pf::FileIdentity drafter = WriteExtents(model.fd, "slot-1.stage-1.state", 1, 9);
  const auto write_control = [&](const std::array<std::int64_t, 8>& words) {
    auto file = pf::OpenPrivateFile(model.fd, "slot-1.stage-1.state", {.write = true});
    if (!file) return false;
    const bool ok = ::pwrite(file->fd, words.data(), sizeof(words), 4096) ==
                        static_cast<ssize_t>(sizeof(words)) &&
                    ::fsync(file->fd) == 0;
    (void)::close(file->fd);
    return ok;
  };
  // Repeat-pending relation 1: N=10, C=7, the three target features at 7..9.
  const std::array<std::int64_t, 8> old_control{1, 10, 7, 7, 3, 1, 1, 0};
  const std::array<std::int64_t, 8> next_control{1, 12, 10, 10, 2, 1, 1, 0};
  ASSERT_TRUE(write_control(old_control));
  kept::Record record = SampleStages();
  record.slot = 1;
  record.tokens.resize(10, 7);
  record.files = {{.name = kept::StateFileName(1, 0), .id = Id(target), .bytes = kExtent},
                  {.name = kept::StateFileName(1, 1), .id = Id(drafter), .bytes = kExtent}};
  record.regions = {kExtent, kExtent};
  record.extents = {{.region = 0, .index = 0, .digest = {}},
                    {.region = 1, .index = 0, .digest = {}}};
  record.checkpoints.clear();
  StateKeeper keeper(1, {});
  const std::size_t index = keeper.AddModel(::dup(model.fd));
  keeper.Keep(index, record);
  ASSERT_TRUE(keeper.Drain(std::chrono::steady_clock::now() + std::chrono::seconds(30)));
  const auto serialized = pf::ReadPrivateFile(model.fd, "slot-1.record", kept::kMostRecordBytes);
  ASSERT_TRUE(serialized);
  const auto original = kept::Decode(*serialized);
  ASSERT_TRUE(original);
  ASSERT_EQ(llmp::runtime::VerifyKeptState(model.fd, *original), "");

  // Accept may update numeric cache before its new control publishes. A
  // crash in that interval leaves the old 64-byte N/C record beside changed
  // numerical state. Startup must refuse the old kept record on the digest.
  auto file = pf::OpenPrivateFile(model.fd, "slot-1.stage-1.state", {.write = true});
  ASSERT_TRUE(file);
  const std::array<std::byte, 32> changed_cache{};
  ASSERT_EQ(::pwrite(file->fd, changed_cache.data(), changed_cache.size(), 8192),
            static_cast<ssize_t>(changed_cache.size()));
  ASSERT_EQ(::fsync(file->fd), 0);
  std::array<std::int64_t, 8> unchanged_control{};
  ASSERT_EQ(::pread(file->fd, unchanged_control.data(), sizeof(unchanged_control), 4096),
            static_cast<ssize_t>(sizeof(unchanged_control)));
  EXPECT_EQ(unchanged_control, old_control);
  (void)::close(file->fd);
  EXPECT_THAT(llmp::runtime::VerifyKeptState(model.fd, *original), HasSubstr("digests differ"));

  // A later control change remains covered by the same ordinary extent hash.
  ASSERT_TRUE(write_control(next_control));
  EXPECT_THAT(llmp::runtime::VerifyKeptState(model.fd, *original), HasSubstr("digests differ"));

  // The ordinary restore-before-reuse hook removes the record. A new record
  // is available only after a subsequent completed spill is hashed again.
  keeper.Invalidate(index, 1);
  EXPECT_FALSE(fs::exists(root / "model" / "slot-1.record"));
  EXPECT_FALSE(keeper.Kept(index, 1));
  record.tokens.resize(12, 8);
  keeper.Keep(index, record);
  ASSERT_TRUE(keeper.Drain(std::chrono::steady_clock::now() + std::chrono::seconds(30)));
  const auto updated_text = pf::ReadPrivateFile(model.fd, "slot-1.record", kept::kMostRecordBytes);
  ASSERT_TRUE(updated_text);
  const auto updated = kept::Decode(*updated_text);
  ASSERT_TRUE(updated);
  EXPECT_EQ(llmp::runtime::VerifyKeptState(model.fd, *updated), "");
  EXPECT_NE(updated->extents[1].digest, original->extents[1].digest);
  EXPECT_EQ(updated->version, original->version);
  fs::remove_all(root);
}

}  // namespace

TEST(KeptRecord, VersionOneBytesAreTheM3Runtimes) {
  const kept::Record r = Sample();
  EXPECT_EQ(r.version, kept::kOneFileVersion);
  EXPECT_EQ(kept::Encode(r), kVersionOneSample);
  auto back = kept::Decode(kVersionOneSample);
  ASSERT_TRUE(back.has_value()) << back.error();
  EXPECT_EQ(*back, r);
  // Version 1's spill file name, and its extents' places, as M3's.
  EXPECT_EQ(kept::StateFileName(2), "slot-2.state");
  EXPECT_EQ(kept::ExtentPlace(r, r.extents[2]).offset, 4 * kExtent);
  // Version 1 holds no stages, and version 2's shape is not version 1's.
  const std::string text(kVersionOneSample);
  auto staged = kept::Decode(Forged(text, R"("regions":)", R"("stages":)"));
  EXPECT_FALSE(staged.has_value());
  auto relabelled = kept::Decode(Forged(text, R"("version":1,)", R"("version":2,)"));
  EXPECT_FALSE(relabelled.has_value());
}

TEST(KeptRecordStages, ACraftedStageRecordIsRefused) {
  const kept::Record r = SampleStages();
  const std::string text = kept::Encode(r);
  // The reader: the version a number it reads, the stages an array of
  // exactly-shaped objects, no field of version 1's.
  for (const auto& [from, to] : std::vector<std::pair<std::string_view, std::string_view>>{
           {R"("version":2,)", R"("version":"2",)"},
           {R"("version":2,)", R"("version":2.0,)"},
           {R"("version":2,)", R"("version":-2,)"},
           {R"("version":2,)", R"("version":0,)"},
           {R"("version":2,)", R"("version":4294967298,)"},
           {R"("stages":[)", R"("regions":[8388608],"stages":[)"},
           {R"("stages":[)", R"("file":"slot-2.state","stages":[)"},
           {R"({"file":"slot-2.state","device":66306,)",
            R"({"device":66306,"file":"slot-2.state",)"},
           {R"("generation":7,"file_bytes":8388608)", R"("generation":"7","file_bytes":8388608)"},
           {R"("file_bytes":8388608})", R"("file_bytes":-8388608})"},
           {R"("file_bytes":8388608})", R"("file_bytes":18446744073709551616})"},
           {R"({"file":"slot-2.state",)", R"({"file":2,)"},
           {R"("extents":[[0,0,)", R"("extents":[[0,4294967296,)"},
           {R"("extents":[[0,0,)", R"("extents":[[-1,0,)"},
       }) {
    auto decoded = kept::Decode(Forged(text, from, to));
    EXPECT_FALSE(decoded.has_value()) << to;
  }
  // The stages an object, not an array.
  EXPECT_FALSE(kept::Decode(Forged(Forged(text, R"("stages":[)", R"("stages":{"a":[)"),
                                   R"(],"extents":)", R"(]},"extents":)"))
                   .has_value());
  // Decoded, then held to the model: a file named anywhere but the slot's
  // stage's own name in the model's directory (a path out of it, another
  // slot's, a stage named twice), a size past its region, or an extent
  // past its stage's file.
  const auto refused = [&](const std::string& forged, std::string_view why) {
    auto decoded = kept::Decode(forged);
    ASSERT_TRUE(decoded.has_value()) << decoded.error();
    auto checked = kept::Check(*decoded, ExpectStages(r));
    ASSERT_FALSE(checked.has_value()) << forged;
    EXPECT_NE(checked.error().find(why), std::string::npos) << checked.error();
  };
  refused(Forged(text, R"({"file":"slot-2.state",)", R"({"file":"../slot-2.state",)"),
          "stage's file");
  refused(Forged(text, R"({"file":"slot-2.state",)", R"({"file":"/tmp/slot-2.state",)"),
          "stage's file");
  refused(Forged(text, R"({"file":"slot-2.state",)", R"({"file":"slot-2.state/..",)"),
          "stage's file");
  {
    // A name with a NUL in it: refused by the reader or by the check, never
    // opened as the shorter name.
    auto decoded = kept::Decode(
        Forged(text, R"({"file":"slot-2.state",)", R"({"file":"slot-2.state\u0000x",)"));
    EXPECT_TRUE(!decoded.has_value() || !kept::Check(*decoded, ExpectStages(r)).has_value());
  }
  refused(Forged(text, R"({"file":"slot-2.stage-1.state",)", R"({"file":"slot-2.state",)"),
          "stage's file");
  refused(Forged(text, R"({"file":"slot-2.stage-1.state",)", R"({"file":"slot-1.stage-1.state",)"),
          "stage's file");
  refused(Forged(text, R"({"file":"slot-2.stage-1.state",)", R"({"file":"slot-2.turn-1.state",)"),
          "stage's file");
  refused(Forged(text, R"("slot":2,)", R"("slot":3,)"), "slot");
  refused(Forged(text, R"("file_bytes":8388608})", R"("file_bytes":18446744073709551615})"),
          "regions");
  refused(Forged(text, R"("file_bytes":2097152})", R"("file_bytes":2101248})"), "regions");
  // An extent past its stage's file, in a stage that is not one, or listed
  // twice; a checkpoint page past its stage goes alone.
  refused(Forged(text, R"([2,0,")", R"([2,1,")"), "outside");
  refused(Forged(text, R"([2,0,")", R"([3,0,")"), "outside");
  refused(Forged(text, R"([2,0,")", R"([4294967295,0,")"), "outside");
  refused(Forged(text, R"([2,0,")", R"([2,4294967295,")"), "outside");
  refused(Forged(text, R"([1,0,")", R"([0,1,")"), "order");
  {
    auto decoded =
        kept::Decode(Forged(text, R"("ranges":[[2,0,5000]])", R"("ranges":[[3,0,5000]])"));
    ASSERT_TRUE(decoded.has_value()) << decoded.error();
    std::vector<std::string> dropped;
    ASSERT_TRUE(kept::Check(*decoded, ExpectStages(r), &dropped).has_value());
    EXPECT_TRUE(decoded->checkpoints.empty());
    EXPECT_EQ(dropped.size(), 1U);
  }
  // Sixteen stages read (the model's count then refuses them); seventeen not.
  kept::Record sixteen = r;
  sixteen.files.clear();
  sixteen.regions.clear();
  for (std::uint32_t s = 0; s < kept::kMostStages; ++s) {
    sixteen.files.push_back(
        {.name = kept::StateFileName(2, s), .id = FakeId(200 + s), .bytes = kExtent});
    sixteen.regions.push_back(kExtent);
  }
  auto read = kept::Decode(kept::Encode(sixteen));
  ASSERT_TRUE(read.has_value()) << read.error();
  EXPECT_EQ(*read, sixteen);
  EXPECT_FALSE(kept::Check(*read, ExpectStages(r)).has_value());
}

// The files a version 2 record names, against what the directory holds: a
// stage's file missing, another slot's or the version 1 file in its place,
// or one that is not the size named, is refused by the start's check, by
// the emptying of unlisted extents and by the keeper (no record written).
TEST(StateKeeperStages, ARecordWhoseStageFilesAreNotThereIsNeverKeptOrAdopted) {
  const fs::path root = Scratch("stage-files");
  fs::create_directories(root);
  Dir model(root / "model");
  ASSERT_GE(model.fd, 0);
  const pf::FileIdentity target = WriteExtents(model.fd, "slot-1.state", 2, 5);
  const pf::FileIdentity drafter = WriteExtents(model.fd, "slot-1.stage-1.state", 2, 9);
  const pf::FileIdentity other = WriteExtents(model.fd, "slot-0.stage-1.state", 2, 3);
  kept::Record r = SampleStages();
  r.slot = 1;
  r.files = {{.name = kept::StateFileName(1, 0), .id = Id(target), .bytes = 2 * kExtent},
             {.name = kept::StateFileName(1, 1), .id = Id(drafter), .bytes = 2 * kExtent}};
  r.regions = {2 * kExtent, 2 * kExtent};
  r.extents = {{.region = 0, .index = 0, .digest = {}}, {.region = 1, .index = 1, .digest = {}}};
  r.checkpoints.clear();
  StateKeeper keeper(1, {});
  const std::size_t index = keeper.AddModel(::dup(model.fd));
  const auto never_kept = [&](const kept::Record& bad) {
    keeper.Keep(index, bad);
    ASSERT_TRUE(keeper.Drain(std::chrono::steady_clock::now() + std::chrono::seconds(30)));
    EXPECT_FALSE(fs::exists(root / "model" / "slot-1.record"));
    EXPECT_FALSE(keeper.Kept(index, 1));
  };
  // Another slot's drafter file in the stage's place (its identity named).
  kept::Record foreign = r;
  foreign.files[1].id = Id(other);
  never_kept(foreign);
  // A stage's file of another size.
  kept::Record larger = r;
  larger.files[1].bytes += kExtent;
  never_kept(larger);
  // Written whole: the record, then the drafter's file removed. The start
  // refuses it; adoption's emptying touches nothing.
  keeper.Keep(index, r);
  ASSERT_TRUE(keeper.Drain(std::chrono::steady_clock::now() + std::chrono::seconds(30)));
  auto text = pf::ReadPrivateFile(model.fd, "slot-1.record", kept::kMostRecordBytes);
  ASSERT_TRUE(text.has_value());
  auto written = kept::Decode(*text);
  ASSERT_TRUE(written.has_value()) << written.error();
  EXPECT_EQ(llmp::runtime::VerifyKeptState(model.fd, *written), "");
  ASSERT_TRUE(fs::remove(root / "model" / "slot-1.stage-1.state"));
  EXPECT_NE(llmp::runtime::VerifyKeptState(model.fd, *written).find("slot-1.stage-1.state"),
            std::string::npos);
  // The version 1 layout in its place: one file of both regions under the
  // drafter's name is not the file the record names.
  (void)WriteExtents(model.fd, "slot-1.stage-1.state", 4, 9);
  EXPECT_NE(llmp::runtime::VerifyKeptState(model.fd, *written), "");
  // Refused for the drafter's file, the emptying leaves the target's whole
  // too (its unlisted second extent still holds its pattern).
  const auto second_extent = [&] {
    auto file = pf::OpenPrivateFile(model.fd, "slot-1.state", {.direct = true});
    EXPECT_TRUE(file.has_value());
    const std::array<kept::Place, 1> at = {{{.offset = kExtent, .bytes = kExtent}}};
    auto digests = HashPlaces(file->fd, at, 1);
    (void)::close(file->fd);
    return digests.front();
  };
  const auto before = second_extent();
  EXPECT_NE(llmp::runtime::EmptyUnlisted(model.fd, *written), "");
  EXPECT_EQ(second_extent(), before);
  const std::vector<std::byte> zeros(kExtent);
  EXPECT_NE(before,
            std::optional(llmp::base::Sha256().Update(std::span<const std::byte>(zeros)).Finish()));
  // An extent naming a stage past the record's files: refused before any
  // file is read.
  kept::Record past = *written;
  past.extents.push_back({.region = 2, .index = 0, .digest = {}});
  EXPECT_NE(llmp::runtime::VerifyKeptState(model.fd, past).find("no file"), std::string::npos);
  fs::remove_all(root);
}

// A record invalidated while its files are hashed (a turn restoring the
// slot: its files about to change) is never written, whichever stage's
// file the keeper was reading; the next one is.
TEST(StateKeeperStages, AnInvalidatedStageRecordIsNeverWritten) {
  const fs::path root = Scratch("stage-invalidate");
  fs::create_directories(root);
  Dir model(root / "model");
  ASSERT_GE(model.fd, 0);
  const pf::FileIdentity target = WriteExtents(model.fd, "slot-1.state", 1, 5);
  const pf::FileIdentity drafter = WriteExtents(model.fd, "slot-1.stage-1.state", 1, 9);
  kept::Record r = SampleStages();
  r.slot = 1;
  r.files = {{.name = kept::StateFileName(1, 0), .id = Id(target), .bytes = kExtent},
             {.name = kept::StateFileName(1, 1), .id = Id(drafter), .bytes = kExtent}};
  r.regions = {kExtent, kExtent};
  r.extents = {{.region = 0, .index = 0, .digest = {}}, {.region = 1, .index = 0, .digest = {}}};
  r.checkpoints.clear();
  StateKeeper keeper(1, {});
  const std::size_t index = keeper.AddModel(::dup(model.fd));
  {
    const StateKeeper::Quiet quiet(&keeper);
    keeper.Keep(index, r);
    keeper.Invalidate(index, 1);
  }
  ASSERT_TRUE(keeper.Drain(std::chrono::steady_clock::now() + std::chrono::seconds(30)));
  EXPECT_FALSE(fs::exists(root / "model" / "slot-1.record"));
  EXPECT_FALSE(keeper.Kept(index, 1));
  keeper.Keep(index, r);
  ASSERT_TRUE(keeper.Drain(std::chrono::steady_clock::now() + std::chrono::seconds(30)));
  EXPECT_TRUE(keeper.Kept(index, 1));
  keeper.Invalidate(index, 1);
  EXPECT_FALSE(fs::exists(root / "model" / "slot-1.record"));
  fs::remove_all(root);
}
