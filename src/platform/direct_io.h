// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// What the storage roles that the runtime pages from and spills to must
// offer (D-034, D-054, D-055): a local block-device filesystem, writable,
// on which direct I/O works at the 4 KiB alignment of D-056's artifacts;
// and the two ways the runtime opens files for direct I/O, a shard to read
// and an unnamed spill file.
//
// Direct I/O is Linux's O_DIRECT here, probed per storage role at startup
// (ProbeDirectIo), and required: on the GB10 the page cache comes out of
// the one memory budget (D-034), so a role without it is refused rather
// than read through the cache. Another system plugs in below the same
// calls (docs/portability.md): macOS opens without O_DIRECT and sets
// F_NOCACHE, a hint rather than a guarantee, and has no O_TMPFILE (a file
// made with mkstemp and unlinked at once); Windows opens with
// FILE_FLAG_NO_BUFFERING (sector-aligned like O_DIRECT) and
// FILE_FLAG_DELETE_ON_CLOSE for the spill file. Where direct I/O is only
// preferred, OpenForDirectRead's `buffered_fallback` reopens a refused
// file through the cache and says so (as TensorFold's reader does); jitLLM
// never asks for that on Linux.

#ifndef JITLLM_PLATFORM_DIRECT_IO_H_
#define JITLLM_PLATFORM_DIRECT_IO_H_

#include <cstdint>
#include <expected>
#include <filesystem>
#include <string>

namespace jitllm::platform {

// The largest direct-I/O alignment the storage roles may require: D-056
// aligns artifact data to 4 KiB.
inline constexpr std::uint32_t kDirectIoAlignment = 4096;

struct FilesystemFacts {
  // The kernel's name for it where known ("ext4"), else its magic number.
  std::string type;
  // One of the local block-device filesystems the roles accept: ext4, XFS
  // or Btrfs. Network, FUSE, overlay and memory-backed ones are not.
  bool accepted = false;
  bool read_only = false;
};

// The filesystem holding path, which must exist.
std::expected<FilesystemFacts, std::string> DescribeFilesystem(const std::filesystem::path& path);

struct DirectIoFacts {
  // What statx reports for a file there (0 if the kernel does not say):
  // the alignment direct I/O needs for memory and for file offsets.
  std::uint32_t memory_alignment = 0;
  std::uint32_t offset_alignment = 0;
};

// D-034's probe: creates an unnamed file in directory (O_TMPFILE, so
// nothing is left behind), opened for direct I/O, then writes and reads
// back one 4 KiB-aligned block. Fails if the filesystem is not accepted,
// if direct I/O is refused or needs more than 4 KiB alignment, or if the
// data does not come back.
std::expected<DirectIoFacts, std::string> ProbeDirectIo(const std::filesystem::path& directory);

// A file opened for direct reads: the descriptor (the caller's to close),
// and whether reads bypass the cache.
struct DirectFile {
  int fd = -1;
  bool direct = false;
};

// Opens `name` beneath the directory `dir` for direct reads: read-only,
// never following a final link, close-on-exec, not a controlling terminal,
// and non-blocking (so a FIFO put in its place cannot hang the open; the
// caller clears it once it knows the file). Without `buffered_fallback`,
// a filesystem that refuses direct I/O fails the open (errno EINVAL); with
// it, the file is opened through the cache instead (`direct` false). As
// openat's errno on failure.
std::expected<DirectFile, int> OpenForDirectRead(int dir, const char* name,
                                                 bool buffered_fallback = false);

// An unnamed file in `directory` for direct reads and writes, owner-only,
// close-on-exec, gone once closed (Linux's O_TMPFILE), so nothing outlives
// the process. errno on failure.
std::expected<int, int> OpenUnnamedDirectFile(const std::filesystem::path& directory);
// Drops a spill range without changing the file length. Future reads of
// that range return zero; adjacent saved state remains intact. No I/O on
// the range may still be in flight. errno on failure.
std::expected<void, int> DiscardFileRange(int fd, std::uint64_t offset, std::uint64_t bytes);

}  // namespace jitllm::platform

#endif  // JITLLM_PLATFORM_DIRECT_IO_H_
