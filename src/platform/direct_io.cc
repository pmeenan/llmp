// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "platform/direct_io.h"

#include <fcntl.h>
#include <linux/falloc.h>
#include <sys/stat.h>
#include <sys/statfs.h>
#include <sys/statvfs.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <expected>
#include <filesystem>
#include <format>
#include <limits>
#include <memory>
#include <string>
#include <utility>

namespace llmp::platform {
namespace {

// Filesystem magic numbers (statfs(2), linux/magic.h).
constexpr std::uint64_t kExt4Magic = 0xEF53;  // ext2, ext3 and ext4 share it
constexpr std::uint64_t kXfsMagic = 0x58465342;
constexpr std::uint64_t kBtrfsMagic = 0x9123683E;
constexpr std::uint64_t kTmpfsMagic = 0x01021994;
constexpr std::uint64_t kOverlayMagic = 0x794C7630;
constexpr std::uint64_t kNfsMagic = 0x6969;
constexpr std::uint64_t kFuseMagic = 0x65735546;
constexpr std::uint64_t kSmbMagic = 0xFF534D42;
constexpr std::uint64_t kSmb2Magic = 0xFE534D42;
constexpr std::uint64_t kRamfsMagic = 0x858458F6;
constexpr std::uint64_t kNineP = 0x01021997;
constexpr std::uint64_t kZfsMagic = 0x2FC12FC1;

std::string Errno(int error) { return std::strerror(error); }  // NOLINT(concurrency-mt-unsafe)

std::string TypeName(std::uint64_t magic) {
  switch (magic) {
    case kExt4Magic:
      return "ext4";
    case kXfsMagic:
      return "xfs";
    case kBtrfsMagic:
      return "btrfs";
    case kTmpfsMagic:
      return "tmpfs";
    case kOverlayMagic:
      return "overlay";
    case kNfsMagic:
      return "nfs";
    case kFuseMagic:
      return "fuse";
    case kSmbMagic:
    case kSmb2Magic:
      return "smb";
    case kRamfsMagic:
      return "ramfs";
    case kNineP:
      return "9p";
    case kZfsMagic:
      return "zfs";
    default:
      return std::format("0x{:x}", magic);
  }
}

class Fd {
 public:
  explicit Fd(int fd) : fd_(fd) {}
  Fd(const Fd&) = delete;
  Fd& operator=(const Fd&) = delete;
  Fd(Fd&&) = delete;
  Fd& operator=(Fd&&) = delete;
  ~Fd() {
    if (fd_ >= 0) {
      (void)::close(fd_);  // an unnamed file: nothing to keep
    }
  }
  int get() const { return fd_; }

 private:
  int fd_;
};

struct FreeDeleter {
  void operator()(void* p) const { std::free(p); }  // NOLINT(cppcoreguidelines-no-malloc)
};

std::unique_ptr<std::byte, FreeDeleter> AlignedBlock() {
  return std::unique_ptr<std::byte, FreeDeleter>(
      static_cast<std::byte*>(std::aligned_alloc(kDirectIoAlignment, kDirectIoAlignment)));
}

}  // namespace

std::expected<FilesystemFacts, std::string> DescribeFilesystem(const std::filesystem::path& path) {
  struct statfs fs{};
  if (::statfs(path.c_str(), &fs) != 0) {
    return std::unexpected(
        std::format("cannot examine the filesystem of {}: {}", path.string(), Errno(errno)));
  }
  struct statvfs vfs{};
  if (::statvfs(path.c_str(), &vfs) != 0) {
    return std::unexpected(
        std::format("cannot examine the filesystem of {}: {}", path.string(), Errno(errno)));
  }
  const auto magic = static_cast<std::uint64_t>(static_cast<std::uint32_t>(fs.f_type));
  FilesystemFacts facts;
  facts.type = TypeName(magic);
  facts.accepted = magic == kExt4Magic || magic == kXfsMagic || magic == kBtrfsMagic;
  facts.read_only = (vfs.f_flag & ST_RDONLY) != 0;
  return facts;
}

std::expected<DirectFile, int> OpenForDirectRead(int dir, const char* name,
                                                 bool buffered_fallback) {
  constexpr int kFlags = O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC | O_NOCTTY;
  const int fd = ::openat(dir, name, kFlags | O_DIRECT);
  if (fd >= 0) {
    return DirectFile{.fd = fd, .direct = true};
  }
  if (errno != EINVAL || !buffered_fallback) {
    return std::unexpected(errno);
  }
  const int buffered = ::openat(dir, name, kFlags);
  if (buffered < 0) {
    return std::unexpected(errno);
  }
  return DirectFile{.fd = buffered, .direct = false};
}

std::expected<int, int> OpenUnnamedDirectFile(const std::filesystem::path& directory) {
  const int fd = ::open(directory.c_str(), O_TMPFILE | O_RDWR | O_DIRECT | O_CLOEXEC, 0600);
  if (fd < 0) {
    return std::unexpected(errno);
  }
  return fd;
}

std::expected<DirectIoFacts, std::string> ProbeDirectIo(const std::filesystem::path& directory) {
  auto filesystem = DescribeFilesystem(directory);
  if (!filesystem) {
    return std::unexpected(filesystem.error());
  }
  if (!filesystem->accepted) {
    return std::unexpected(
        std::format("{} is on {}, not a local block-device filesystem (ext4, xfs or btrfs)",
                    directory.string(), filesystem->type));
  }
  if (filesystem->read_only) {
    return std::unexpected(std::format("{} is on a read-only filesystem", directory.string()));
  }
  const Fd fd(::open(directory.c_str(), O_TMPFILE | O_RDWR | O_DIRECT | O_CLOEXEC, 0600));
  if (fd.get() < 0) {
    return std::unexpected(
        std::format("cannot create a direct-I/O file in {}: {}", directory.string(), Errno(errno)));
  }
  DirectIoFacts facts;
  struct statx sx{};
  if (::statx(fd.get(), "", AT_EMPTY_PATH, STATX_DIOALIGN, &sx) == 0 &&
      (sx.stx_mask & STATX_DIOALIGN) != 0) {
    facts.memory_alignment = sx.stx_dio_mem_align;
    facts.offset_alignment = sx.stx_dio_offset_align;
    if (facts.offset_alignment == 0) {
      return std::unexpected(std::format("{} does not support direct I/O", directory.string()));
    }
    if (facts.offset_alignment > kDirectIoAlignment ||
        facts.memory_alignment > kDirectIoAlignment) {
      return std::unexpected(std::format(
          "direct I/O in {} needs {}-byte offsets and {}-byte buffers, more than {}",
          directory.string(), facts.offset_alignment, facts.memory_alignment, kDirectIoAlignment));
    }
  }
  const auto out = AlignedBlock();
  const auto in = AlignedBlock();
  if (!out || !in) {
    return std::unexpected("cannot allocate the direct-I/O probe's buffers");
  }
  for (std::size_t i = 0; i < kDirectIoAlignment; ++i) {
    out.get()[i] = static_cast<std::byte>(((i * 131U) + 7U) & 0xFFU);
  }
  std::memset(in.get(), 0, kDirectIoAlignment);
  const ssize_t wrote = ::pwrite(fd.get(), out.get(), kDirectIoAlignment, 0);
  if (std::cmp_not_equal(wrote, kDirectIoAlignment)) {
    return std::unexpected(
        std::format("a direct write in {} failed: {}", directory.string(),
                    wrote < 0 ? Errno(errno) : std::format("wrote {} bytes", wrote)));
  }
  const ssize_t got = ::pread(fd.get(), in.get(), kDirectIoAlignment, 0);
  if (std::cmp_not_equal(got, kDirectIoAlignment)) {
    return std::unexpected(std::format("a direct read in {} failed: {}", directory.string(),
                                       got < 0 ? Errno(errno) : std::format("read {} bytes", got)));
  }
  if (std::memcmp(out.get(), in.get(), kDirectIoAlignment) != 0) {
    return std::unexpected(
        std::format("a direct read in {} did not return what was written", directory.string()));
  }
  return facts;
}

std::expected<void, int> DiscardFileRange(int fd, std::uint64_t offset, std::uint64_t bytes) {
  const auto limit = static_cast<std::uint64_t>(std::numeric_limits<off_t>::max());
  if (fd < 0 || offset > limit || bytes > limit - offset || offset % kDirectIoAlignment != 0 ||
      bytes % kDirectIoAlignment != 0) {
    return std::unexpected(EINVAL);
  }
  if (bytes != 0 && ::fallocate(fd, FALLOC_FL_PUNCH_HOLE | FALLOC_FL_KEEP_SIZE,
                                static_cast<off_t>(offset), static_cast<off_t>(bytes)) != 0) {
    return std::unexpected(errno);
  }
  return {};
}

std::expected<void, int> TransferDirectFile(int fd, std::uint64_t offset,
                                            std::span<std::byte> buffer, bool write) {
  const auto limit = static_cast<std::uint64_t>(std::numeric_limits<off_t>::max());
  if (fd < 0 || offset > limit || buffer.size() > limit - offset ||
      offset % kDirectIoAlignment != 0 || buffer.size() % kDirectIoAlignment != 0 ||
      reinterpret_cast<std::uintptr_t>(buffer.data()) % kDirectIoAlignment != 0) {
    return std::unexpected(EINVAL);
  }
  std::size_t done = 0;
  while (done < buffer.size()) {
    const std::size_t bytes = std::min<std::size_t>(buffer.size() - done, 2U << 20U);
    const auto at = static_cast<off_t>(offset + done);
    const ssize_t result = write ? ::pwrite(fd, buffer.data() + done, bytes, at)
                                 : ::pread(fd, buffer.data() + done, bytes, at);
    if (result < 0) {
      if (errno == EINTR) {
        continue;
      }
      return std::unexpected(errno);
    }
    if (result == 0 || std::cmp_greater(result, bytes) ||
        static_cast<std::uint64_t>(result) % kDirectIoAlignment != 0) {
      return std::unexpected(EIO);
    }
    done += static_cast<std::size_t>(result);
  }
  return {};
}

}  // namespace llmp::platform
