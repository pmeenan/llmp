// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Owner-only files kept across a restart (D-014, D-105): a private
// directory beneath the spill role, the files in it opened for direct I/O
// (a conversation's spill file and turn checkpoints) or replaced whole and
// atomically (its record). What the runtime creates is private: a
// directory 0700, a file 0600, the calling user's. What it finds is used,
// whoever owns it, whatever its modes and links (D-105 as amended on
// 2026-10-09: a machine's owner who changes permissions chose to; a note
// says so, PrivacyNote); links are followed. Only what the runtime could
// not use correctly is refused: a file that is not a regular file, a link
// to nothing. Linux here (O_DIRECT, the inode generation); another system
// plugs in below the same calls as direct_io.h's opens do
// (docs/portability.md).

#ifndef LLMP_PLATFORM_KEPT_FILES_H_
#define LLMP_PLATFORM_KEPT_FILES_H_

#include <cstddef>
#include <cstdint>
#include <expected>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace llmp::platform {

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
// directory at `path`, with `parent` -1): created 0700 (exactly, whatever
// the umask) if missing; an existing one is used as it is, a link to a
// directory followed. ENOLINK for a link to nothing, ENOTDIR for what is
// not a directory. Its descriptor, the caller's to close. errno on
// failure.
std::expected<int, int> OpenPrivateDirectory(int parent, const char* name);

// A private file beneath the open directory `dir`.
struct PrivateFile {
  int fd = -1;  // the caller's to close
  FileIdentity identity;
  std::uint64_t bytes = 0;  // its size when opened (0 with `truncate`)
};
struct PrivateOpen {
  bool write = false;     // read and write; otherwise read only
  bool create = false;    // made (0600) if missing; otherwise ENOENT
  bool truncate = false;  // a new empty file in the name's place
  bool direct = false;    // direct I/O (O_DIRECT), as the spill role offers
};
// Opens `name` beneath `dir`: an existing file as it is (any owner, mode
// or number of links; a link followed), a new one made 0600 (O_EXCL,
// never through a link). An existing object is pinned and checked before
// opening it for I/O, so a substituted FIFO or device is never opened.
// EPERM unless it is regular; ENOLINK for a link to nothing, which is
// never created through. `truncate` gives the name a new empty file (made
// as above): what was there is unlinked, never emptied, so a link's target
// or a file's other names keep their bytes (its identity is therefore a
// new one). Existing opens keep the pinned object across pathname changes;
// exclusive creation retries a name that appears meanwhile, with EAGAIN
// if it keeps changing.
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
// CommitPrivateReplacement: that file, still the one written, renamed over
// `name` (atomic: on failure nothing moved; a link at `name` is replaced,
// not written through). Refused (EPERM) when it is not: one that is not a
// regular file is removed; another regular file put there since is left
// to whoever made it (a later Write removes it). The file written is
// removed when the rename fails. SyncDirectory: the rename (or
// a removal) reaches the device. AbandonPrivateReplacement: the temporary
// file removed, unrenamed, if it is still the one written.
std::expected<FileIdentity, int> WritePrivateReplacement(int dir, const char* name,
                                                         std::string_view bytes);
std::expected<void, int> CommitPrivateReplacement(int dir, const char* name,
                                                  const FileIdentity& written);
void AbandonPrivateReplacement(int dir, const char* name, const FileIdentity& written);
std::expected<void, int> SyncDirectory(int dir);

// Reads a private file (as OpenPrivateFile opens it) of at most `limit`
// bytes whole: EFBIG past it.
std::expected<std::string, int> ReadPrivateFile(int dir, const char* name, std::size_t limit);

// How an existing kept directory departs from what the runtime makes, as
// text for a log line or `llmp doctor`, empty if it does not: the open
// directory `dir` owned by another user or with a mode other than 0700,
// how many regular files in it are another user's, not mode 0600 or have
// other names, and how many entries are symbolic links. Never a reason to
// refuse; callers say "used as found" and report it once per directory.
std::string PrivacyNote(int dir);

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

}  // namespace llmp::platform

#endif  // LLMP_PLATFORM_KEPT_FILES_H_
