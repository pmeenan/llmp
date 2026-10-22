// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// How the importer opens a checkpoint's files (D-009, D-054): as any
// program opens files, links followed, so a Hugging Face cache's snapshot
// links into its blobs work. Who may change a checkpoint is the
// filesystem's business (D-073 as amended on 2026-10-09); D-009's
// protection is in what the readers do with the bytes (bounds, caps,
// strict parsers, no checkpoint code run). Only a regular file is opened:
// a device or a FIFO is refused before it is opened, since opening one can
// have effects of its own. Opening pins the resolved object before checking
// its type and reopens that reference, never the original pathname. A reader
// holds a file open while it reads it
// and then closes it; the checkpoint keeps each file's path and identity
// (device and inode), and a later read opens the file again and refuses
// one that is no longer the file the reader read (checkpoint.h).
//
// One narrow rule protects what is copied, not parsed: a file the importer
// keeps verbatim (into an artifact or a composition, as the diffusers
// reader's kept files and the writer's tokenizer and configuration copies
// are) must resolve inside the checkpoint's tree (ResolvesInside). A file
// resolving elsewhere, such as a link a cloned repository planted to one
// of the user's own files, is skipped with a diagnostic
// (Checkpoint::skipped), never refused.

#ifndef LLMP_IMPORT_FILES_H_
#define LLMP_IMPORT_FILES_H_

#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <optional>
#include <span>
#include <string>

#include "artifact/artifact.h"
#include "base/sha256.h"
#include "import/error.h"

namespace llmp::import {

// Bytes interpreted by a reader, authenticated again during source streaming.
// Current readers interpret one contiguous prefix per file. Unread payload
// contributes no bookkeeping; an unparsed kept file has no receipt.
struct ParsedPrefix {
  std::uint64_t bytes = 0;
  base::Sha256Digest sha256{};
  bool operator==(const ParsedPrefix&) const = default;
};
static_assert(sizeof(ParsedPrefix) == 40);

// Publication directories: create new directories with mode 0700, leaving
// existing permissions unchanged. Persist each new directory and its entry in
// its parent before proceeding. With parents=false, never create a missing parent
// (in particular an archive root that disappeared after an unmount).
std::expected<void, Error> CreateImportDirectory(const std::filesystem::path& path, bool parents);

// An open regular file, with what identifies it.
struct OpenFile {
  artifact::FileDescriptor fd{};
  std::filesystem::path path{};  // absolute, as opened (links not resolved)
  std::uint64_t bytes = 0;
  std::uint64_t device = 0;
  std::uint64_t inode = 0;
  ParsedPrefix parsed{};
};

// The file at `path` (kFileType when it does not exist or is not a regular
// file). `file` is its index in the checkpoint's order, for errors.
std::expected<OpenFile, Error> OpenCheckpointFile(const std::filesystem::path& path,
                                                  std::uint64_t file);
// The same, or none when the path does not exist (a link to nothing is
// kFileType: a broken checkpoint, not an absent file).
std::expected<std::optional<OpenFile>, Error> OpenCheckpointFileIfPresent(
    const std::filesystem::path& path, std::uint64_t file);

// The directory at `path`, absolute with its links resolved: the path
// files beneath it are opened through (kFileType when it is not one).
std::expected<std::filesystem::path, Error> CheckpointDirectory(const std::filesystem::path& path);

// Whether `path`, its links resolved, lies inside the checkpoint tree
// `root` (resolved too); or, when `root` is or lies inside a Hugging Face
// cache snapshot (<repository>/snapshots/<revision>), inside that
// repository's blobs/ directory, where the cache's snapshot files link.
// False when either cannot be resolved. A file the importer keeps verbatim
// is kept only if this holds.
bool Beneath(const std::filesystem::path& path, const std::filesystem::path& root);
bool ResolvesInside(const std::filesystem::path& root, const std::filesystem::path& path);

// Reads exactly out.size() bytes at offset (kIo when the file ends first).
std::expected<void, Error> ReadExactly(int fd, std::uint64_t offset, std::span<std::byte> out,
                                       std::uint64_t file);

// A whole small document of at most `limit` bytes (kBounds beyond).
std::expected<std::string, Error> ReadDocument(const OpenFile& f, std::uint64_t limit,
                                               std::uint64_t file);

}  // namespace llmp::import

#endif  // LLMP_IMPORT_FILES_H_
