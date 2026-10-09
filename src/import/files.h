// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// How the importer opens a checkpoint's files (D-009, D-054): every path
// through the platform's path-trust walk (platform/path_trust.h), trusting
// root and one user, then opened without following a link and compared
// with what the walk approved by device and inode. Links on the way are
// followed as the walk resolves them, so a Hugging Face cache's snapshot
// links into its blobs work. A file must be a regular file owned by root or
// the trusted user that nobody else can write; a directory must be one
// nobody else can add entries to. The descriptors stay open for the
// checkpoint's life, so its bytes are read from the files that were
// checked; a file that shrinks later fails its read rather than being
// reopened.

#ifndef LLMP_IMPORT_FILES_H_
#define LLMP_IMPORT_FILES_H_

#include <sys/types.h>

#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <optional>
#include <span>
#include <string>

#include "artifact/artifact.h"
#include "import/error.h"

namespace llmp::import {

// An open, checked file.
struct OpenFile {
  artifact::FileDescriptor fd{};
  std::uint64_t bytes = 0;
  std::uint64_t device = 0;
  std::uint64_t inode = 0;
};

// The file at `path` (kFileType when it does not exist). `file` is its
// index in the checkpoint's order, for errors.
std::expected<OpenFile, Error> OpenTrustedFile(const std::filesystem::path& path, uid_t trusted,
                                               std::uint64_t file);
// The same, or none when the path does not exist.
std::expected<std::optional<OpenFile>, Error> OpenTrustedFileIfPresent(
    const std::filesystem::path& path, uid_t trusted, std::uint64_t file);

// The directory at `path`, its links resolved: the path files beneath it
// are opened through.
std::expected<std::filesystem::path, Error> TrustedDirectory(const std::filesystem::path& path,
                                                             uid_t trusted);

// Reads exactly out.size() bytes at offset (kIo when the file ends first).
std::expected<void, Error> ReadExactly(int fd, std::uint64_t offset, std::span<std::byte> out,
                                       std::uint64_t file);

// A whole small document of at most `limit` bytes (kBounds beyond).
std::expected<std::string, Error> ReadDocument(const OpenFile& f, std::uint64_t limit,
                                               std::uint64_t file);

}  // namespace llmp::import

#endif  // LLMP_IMPORT_FILES_H_
