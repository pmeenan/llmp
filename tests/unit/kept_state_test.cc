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
#include <gtest/gtest.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
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
  r.file = kept::StateFileName(2);
  r.id = {.device = 66306, .inode = 1234567, .generation = 99};
  r.regions = {4 * kExtent, 2 * kExtent};
  r.file_bytes = 6 * kExtent;
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

TEST(KeptFiles, PrivateDirectoriesAndFilesAreOwnerOnlyAndNeverLinks) {
  const fs::path root = Scratch("files");
  fs::create_directories(root);
  fs::permissions(root, fs::perms::owner_all, fs::perm_options::replace);
  Dir dir(root / "conversations");
  ASSERT_GE(dir.fd, 0);
  struct stat st{};
  ASSERT_EQ(::stat((root / "conversations").c_str(), &st), 0);
  EXPECT_EQ(st.st_mode & 07777, 0700U);
  // A directory with other permissions, or a link, is refused.
  fs::create_directories(root / "open");
  fs::permissions(root / "open", fs::perms::all, fs::perm_options::replace);
  EXPECT_EQ(pf::OpenPrivateDirectory(-1, (root / "open").c_str()).error_or(0), EPERM);
  fs::create_directory_symlink(root / "conversations", root / "link");
  EXPECT_FALSE(pf::OpenPrivateDirectory(-1, (root / "link").c_str()).has_value());

  // A file: made 0600; refused through a link, with another mode or a
  // second link.
  auto made = pf::OpenPrivateFile(dir.fd, "a", {.write = true, .create = true});
  ASSERT_TRUE(made.has_value());
  (void)::close(made->fd);
  ASSERT_EQ(::fstatat(dir.fd, "a", &st, 0), 0);
  EXPECT_EQ(st.st_mode & 07777, 0600U);
  ASSERT_EQ(::symlinkat("a", dir.fd, "b"), 0);
  EXPECT_FALSE(pf::OpenPrivateFile(dir.fd, "b", {}).has_value());
  ASSERT_EQ(::linkat(dir.fd, "a", dir.fd, "c", 0), 0);
  EXPECT_EQ(pf::OpenPrivateFile(dir.fd, "a", {}).error_or(0), EPERM);
  ASSERT_EQ(::unlinkat(dir.fd, "c", 0), 0);
  ASSERT_EQ(::fchmodat(dir.fd, "a", 0644, 0), 0);
  EXPECT_EQ(pf::OpenPrivateFile(dir.fd, "a", {}).error_or(0), EPERM);
  ASSERT_EQ(::fchmodat(dir.fd, "a", 0600, 0), 0);
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
  refused([](kept::Record& c, kept::Expected&) { c.file_bytes += 4096; }, "size");
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
  r.file = kept::StateFileName(1);
  r.id = Id(first);
  r.regions = {kExtent};
  r.file_bytes = kExtent;
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
  other.file = kept::StateFileName(2);
  other.id = Id(second);
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
  r.file = kept::StateFileName(1);
  r.id = Id(state);
  r.regions = {3 * kExtent};
  r.file_bytes = 3 * kExtent;
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
  wrong.id.inode += 1;
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
  r.file = kept::StateFileName(0);
  r.id = Id(state);
  r.regions = {3 * kExtent, kExtent};
  r.file_bytes = 4 * kExtent;
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
  other.id.inode += 1;
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
  EXPECT_EQ(Id(file->identity), r.id);
  EXPECT_EQ(file->bytes, 4 * kExtent);
  (void)::close(file->fd);
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
      r.file = kept::StateFileName(1);
      r.id = Id(file);
      r.regions = {kExtent};
      r.file_bytes = kExtent;
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
