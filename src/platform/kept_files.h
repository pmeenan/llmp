// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Owner-only files kept across a restart (D-014, D-105): a private
// directory beneath the spill role, the files in it opened for direct I/O
// (a conversation's spill file and turn checkpoints) or replaced whole and
// atomically (its record), never through a link, and checked to be what
// the runtime made: a regular file of the calling user's, mode 0600, one
// link; a directory of its, mode 0700. Linux here (O_DIRECT, the inode
// generation); another system plugs in below the same calls as
// direct_io.h's opens do (docs/portability.md).

#ifndef JITLLM_PLATFORM_KEPT_FILES_H_
#define JITLLM_PLATFORM_KEPT_FILES_H_

#include <cstddef>
#include <cstdint>
#include <expected>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace jitllm::platform {

// Which file this is: its device and inode, and the inode's generation
// where the filesystem reports one (files.h FileGeneration), so a file is
// told apart from a later one that reused its inode.
struct FileIdentity {
  std::uint64_t device = 0;
  std::uint64_t inode = 0;
  std::optional<std::uint64_t> generation;
  bool operator==(const FileIdentity&) const = default;
};

// A private directory `name` beneath the open directory `parent` (or the
// directory at `path`, with `parent` -1): created 0700 if missing; refused
// (EPERM) unless it is a directory, not a link, the calling user owns with
// mode 0700. Its descriptor, the caller's to close. errno on failure.
std::expected<int, int> OpenPrivateDirectory(int parent, const char* name);

// A private file beneath the open directory `dir`.
struct PrivateFile {
  int fd = -1;  // the caller's to close
  FileIdentity identity;
  std::uint64_t bytes = 0;  // its size when opened (after `truncate`)
};
struct PrivateOpen {
  bool write = false;     // read and write; otherwise read only
  bool create = false;    // made (0600) if missing; otherwise ENOENT
  bool truncate = false;  // emptied
  bool direct = false;    // direct I/O (O_DIRECT), as the spill role offers
};
// Opens `name` beneath `dir`, never following a final link; refused
// (EPERM) unless a regular file the calling user owns with mode 0600 and
// one link (a hard link elsewhere could outlive its deletion here).
std::expected<PrivateFile, int> OpenPrivateFile(int dir, const char* name, PrivateOpen how);

// Replaces `name` beneath `dir` with `bytes`, atomically: a temporary file
// (0600) written and synced, renamed over it, the directory synced. A
// crash leaves the old file or the new one, never part of either.
std::expected<void, int> ReplacePrivateFile(int dir, const char* name, std::string_view bytes);
// The same in its steps, for a caller that decides between them (the
// state keeper: whether its record is still wanted, under its lock, with
// no sync waited on there). WritePrivateReplacement: `name`.tmp beneath
// `dir` made here (O_EXCL, 0600, never through a link; one a crash left
// is removed first), `bytes` written and synced; removed again on
// failure. Returns its identity, which the other steps take.
// CommitPrivateReplacement: that file, still the one written and still a
// private one, renamed over `name` (atomic: on failure nothing moved).
// Refused (EPERM) when it is not: one that is not a regular file, or the
// one written but changed, is removed; another regular file put there
// since is left to whoever made it (a later Write removes it). The file
// written is removed when the rename fails. SyncDirectory: the rename (or
// a removal) reaches the device. AbandonPrivateReplacement: the temporary
// file removed, unrenamed, if it is still the one written.
std::expected<FileIdentity, int> WritePrivateReplacement(int dir, const char* name,
                                                         std::string_view bytes);
std::expected<void, int> CommitPrivateReplacement(int dir, const char* name,
                                                  const FileIdentity& written);
void AbandonPrivateReplacement(int dir, const char* name, const FileIdentity& written);
std::expected<void, int> SyncDirectory(int dir);

// Reads a private file (as OpenPrivateFile checks it) of at most `limit`
// bytes whole: EFBIG past it.
std::expected<std::string, int> ReadPrivateFile(int dir, const char* name, std::size_t limit);

// Removes the file `name` beneath `dir`; one that is not there is not an
// error. A directory there is removed with everything in it (never
// following a link).
std::expected<void, int> RemovePrivate(int dir, const char* name);

struct DirectoryEntry {
  std::string name;
  bool regular = false;    // a regular file (not a link)
  bool directory = false;  // a directory (not a link)
};
// The entries of the open directory `dir` (not "." or ".."), sorted by
// name, each typed without following a link.
std::expected<std::vector<DirectoryEntry>, int> ListPrivateDirectory(int dir);

// Data written to `fd` reaches the device before this returns (fdatasync).
std::expected<void, int> SyncFileData(int fd);

// The calling thread runs as background work from now on: the lowest CPU
// priority (Linux: nice 19 for this thread) and the idle I/O class (its
// reads wait for everyone else's where the block scheduler honours it).
// Best effort; false if either could not be set.
bool LowerThreadPriority();

}  // namespace jitllm::platform

#endif  // JITLLM_PLATFORM_KEPT_FILES_H_
