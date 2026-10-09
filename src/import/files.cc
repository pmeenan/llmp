// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "import/files.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <system_error>
#include <utility>

#include "artifact/artifact.h"
#include "import/error.h"
#include "platform/path_trust.h"

namespace llmp::import {
namespace {

std::unexpected<Error> Fail(Rule rule, std::string_view reason, std::uint64_t file) {
  return std::unexpected(Error{.rule = rule, .reason = reason, .file = file});
}

std::expected<platform::TrustedPath, Error> Walk(const std::filesystem::path& path, uid_t trusted,
                                                 std::uint64_t file) {
  std::error_code error;
  const std::filesystem::path absolute = std::filesystem::absolute(path, error);
  if (error) {
    return Fail(Rule::kIo, "cannot make a checkpoint path absolute", file);
  }
  auto walk = platform::WalkTrusted(absolute, trusted, /*final_link_ok=*/true);
  if (!walk) {
    return Fail(Rule::kUntrusted, "a checkpoint path others could change, or not a path to one",
                file);
  }
  return std::move(*walk);
}

}  // namespace

std::expected<OpenFile, Error> OpenTrustedFile(const std::filesystem::path& path, uid_t trusted,
                                               std::uint64_t file) {
  auto f = OpenTrustedFileIfPresent(path, trusted, file);
  if (!f) {
    return std::unexpected(f.error());
  }
  if (!*f) {
    return Fail(Rule::kFileType, "a checkpoint file is absent", file);
  }
  return std::move(**f);
}

std::expected<std::optional<OpenFile>, Error> OpenTrustedFileIfPresent(
    const std::filesystem::path& path, uid_t trusted, std::uint64_t file) {
  auto walk = Walk(path, trusted, file);
  if (!walk) {
    return std::unexpected(walk.error());
  }
  if (!walk->exists) {
    return std::optional<OpenFile>();
  }
  // Only what the walk saw as a regular file is opened: opening a device
  // can have effects of its own.
  if (!S_ISREG(walk->status.st_mode)) {
    return Fail(Rule::kFileType, "a checkpoint file is not a regular file", file);
  }
  OpenFile f;
  f.fd = artifact::FileDescriptor(
      ::open(walk->resolved.c_str(), O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC | O_NOCTTY));
  if (!f.fd.valid()) {
    const int error = errno;
    if (error == EMFILE || error == ENFILE || error == ENOMEM) {
      return Fail(Rule::kIo, "out of file descriptors opening a checkpoint file", file);
    }
    return Fail(Rule::kFileType, "a checkpoint file cannot be opened as a regular file", file);
  }
  struct stat status{};
  if (::fstat(f.fd.get(), &status) != 0) {
    return Fail(Rule::kIo, "fstat failed", file);
  }
  if (!S_ISREG(status.st_mode)) {
    return Fail(Rule::kFileType, "a checkpoint file is not a regular file", file);
  }
  if (status.st_dev != walk->status.st_dev || status.st_ino != walk->status.st_ino) {
    return Fail(Rule::kUntrusted, "a checkpoint file changed during the trust walk", file);
  }
  if ((status.st_uid != 0 && status.st_uid != trusted) ||
      platform::OthersCanWrite(status, trusted, f.fd.get())) {
    return Fail(Rule::kUntrusted, "a checkpoint file others could change", file);
  }
  f.bytes = static_cast<std::uint64_t>(status.st_size);
  f.device = static_cast<std::uint64_t>(status.st_dev);
  f.inode = static_cast<std::uint64_t>(status.st_ino);
  return std::optional<OpenFile>(std::move(f));
}

std::expected<std::filesystem::path, Error> TrustedDirectory(const std::filesystem::path& path,
                                                             uid_t trusted) {
  auto walk = Walk(path, trusted, kNoItem);
  if (!walk) {
    return std::unexpected(walk.error());
  }
  if (!walk->exists || !S_ISDIR(walk->status.st_mode)) {
    return Fail(Rule::kFileType, "the checkpoint is not a directory", kNoItem);
  }
  if (auto ok = platform::CheckPrivateDirectory(walk->resolved, walk->status, trusted); !ok) {
    return Fail(Rule::kUntrusted, "a checkpoint directory others could change", kNoItem);
  }
  return std::move(walk->resolved);
}

std::expected<void, Error> ReadExactly(int fd, std::uint64_t offset, std::span<std::byte> out,
                                       std::uint64_t file) {
  std::size_t done = 0;
  while (done < out.size()) {
    const ssize_t n =
        ::pread(fd, out.data() + done, out.size() - done, static_cast<off_t>(offset + done));
    if (n < 0) {
      if (errno == EINTR) {
        continue;
      }
      return Fail(Rule::kIo, "a checkpoint read failed", file);
    }
    if (n == 0) {
      return Fail(Rule::kIo, "a checkpoint file ended before its size", file);
    }
    done += static_cast<std::size_t>(n);
  }
  return {};
}

std::expected<std::string, Error> ReadDocument(const OpenFile& f, std::uint64_t limit,
                                               std::uint64_t file) {
  if (f.bytes > limit) {
    return Fail(Rule::kBounds, "a document larger than its cap", file);
  }
  std::string text(static_cast<std::size_t>(f.bytes), '\0');
  if (auto ok = ReadExactly(f.fd.get(), 0, std::as_writable_bytes(std::span(text)), file); !ok) {
    return std::unexpected(ok.error());
  }
  return text;
}

}  // namespace llmp::import
