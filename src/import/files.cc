// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "import/files.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
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
#include <vector>

#include "artifact/artifact.h"
#include "artifact/schema.h"
#include "import/error.h"
#include "platform/files.h"

namespace llmp::import {
namespace {

namespace fs = std::filesystem;

std::unexpected<Error> Fail(Rule rule, std::string_view reason, std::uint64_t file) {
  return std::unexpected(Error{.rule = rule, .reason = reason, .file = file});
}

}  // namespace

// Whether `path` lies strictly beneath `root`, both resolved.
bool Beneath(const fs::path& path, const fs::path& root) {
  const auto [in_root, in_path] = std::ranges::mismatch(root, path);
  return in_root == root.end() && in_path != path.end();
}

std::expected<void, Error> CreateImportDirectory(const fs::path& path, bool parents) {
  std::error_code error;
  auto absolute = fs::absolute(path, error);
  if (error || path.empty()) return Fail(Rule::kIo, "invalid publication directory", kNoItem);
  while (!absolute.has_filename() && absolute != absolute.root_path()) {
    absolute = absolute.parent_path();
  }
  std::vector<fs::path> missing;
  auto at = absolute;
  for (;;) {
    const auto status = fs::status(at, error);
    if (error && error != std::errc::no_such_file_or_directory)
      return Fail(Rule::kIo, "cannot examine publication directory", kNoItem);
    if (fs::exists(status)) {
      if (!fs::is_directory(status))
        return Fail(Rule::kIo, "publication parent is not a directory", kNoItem);
      break;
    }
    if ((!parents && !missing.empty()) || missing.size() == 128 || at == at.parent_path())
      return Fail(Rule::kIo, "publication directory has no available parent", kNoItem);
    missing.push_back(at);
    at = at.parent_path();
  }
  const auto sync = [](const fs::path& directory) -> std::expected<void, Error> {
    artifact::FileDescriptor fd(::open(directory.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC));
    if (!fd.valid() || ::fsync(fd.get()) != 0)
      return Fail(Rule::kIo, "cannot persist publication directory ancestry", kNoItem);
    return {};
  };
  for (auto next = missing.rbegin(); next != missing.rend(); ++next) {
    // The environment's umask may remove owner access, but must never expose
    // a newly created directory. Do not change a path another creator supplied.
    if (::mkdir(next->c_str(), 0700) == 0) {
      if (::chmod(next->c_str(), 0700) != 0) {
        (void)::rmdir(next->c_str());
        return Fail(Rule::kIo, "cannot protect publication directory", kNoItem);
      }
    } else {
      error.clear();
      if (errno != EEXIST || !fs::is_directory(*next, error) || error)
        return Fail(Rule::kIo, "cannot create publication directory", kNoItem);
    }
    if (auto ok = sync(*next); !ok) return ok;
    if (auto ok = sync(next->parent_path()); !ok) return ok;
  }
  return {};
}

std::expected<OpenFile, Error> OpenCheckpointFile(const fs::path& path, std::uint64_t file) {
  auto f = OpenCheckpointFileIfPresent(path, file);
  if (!f) {
    return std::unexpected(f.error());
  }
  if (!*f) {
    return Fail(Rule::kFileType, "a checkpoint file is absent", file);
  }
  return std::move(**f);
}

std::expected<std::optional<OpenFile>, Error> OpenCheckpointFileIfPresent(const fs::path& path,
                                                                          std::uint64_t file) {
  if (!artifact::schema::IsFileName(path.filename().string()))
    return Fail(
        Rule::kContainer,
        "unsafe checkpoint filename: expected a plain UTF-8 name without controls or separators",
        file);
  std::error_code error;
  const fs::path absolute = fs::absolute(path, error);
  if (error) {
    return Fail(Rule::kIo, "cannot make a checkpoint path absolute", file);
  }
  // Pin first: checking a pathname and then opening it could instead open a
  // device or FIFO swapped into its place. O_PATH lives in the platform seam.
  auto pinned = platform::OpenPathReference(-1, absolute.c_str());
  if (!pinned) {
    const int open_error = pinned.error();
    if (open_error == ENOENT) {
      // A link to nothing (a cache's blob not downloaded) is a broken
      // checkpoint, not a file it lacks: reading on would silently omit it.
      if (struct stat link{}; ::lstat(absolute.c_str(), &link) == 0 && S_ISLNK(link.st_mode)) {
        return Fail(Rule::kFileType, "a checkpoint file is a link to nothing", file);
      }
      return std::optional<OpenFile>();
    }
    if (open_error == EMFILE || open_error == ENFILE || open_error == ENOMEM) {
      return Fail(Rule::kIo, "out of file descriptors opening a checkpoint file", file);
    }
    return Fail(Rule::kFileType, "a checkpoint path that cannot be examined", file);
  }
  const artifact::FileDescriptor reference(*pinned);
  auto opened = platform::ReopenRegularFileForRead(reference.get());
  if (!opened) {
    const int open_error = opened.error();
    if (open_error == EMFILE || open_error == ENFILE || open_error == ENOMEM) {
      return Fail(Rule::kIo, "out of file descriptors opening a checkpoint file", file);
    }
    if (open_error == EIO || open_error == EBADF) {
      return Fail(Rule::kIo, "cannot examine a pinned checkpoint file", file);
    }
    return Fail(Rule::kFileType, "a checkpoint file cannot be opened as a regular file", file);
  }
  OpenFile f;
  f.path = absolute;
  f.fd = artifact::FileDescriptor(*opened);
  struct stat status{};
  if (::fstat(f.fd.get(), &status) != 0) {
    return Fail(Rule::kIo, "fstat failed", file);
  }
  if (!S_ISREG(status.st_mode) || status.st_size < 0) {
    return Fail(Rule::kFileType, "a checkpoint file is not a regular file", file);
  }
  f.bytes = static_cast<std::uint64_t>(status.st_size);
  f.device = static_cast<std::uint64_t>(status.st_dev);
  f.inode = static_cast<std::uint64_t>(status.st_ino);
  return std::optional<OpenFile>(std::move(f));
}

std::expected<fs::path, Error> CheckpointDirectory(const fs::path& path) {
  std::error_code error;
  fs::path resolved = fs::canonical(path, error);
  if (error) {
    return Fail(error == std::errc::no_such_file_or_directory ? Rule::kFileType : Rule::kIo,
                "the checkpoint cannot be resolved", kNoItem);
  }
  if (!fs::is_directory(resolved, error) || error) {
    return Fail(Rule::kFileType, "the checkpoint is not a directory", kNoItem);
  }
  return resolved;
}

bool ResolvesInside(const fs::path& root, const fs::path& path) {
  std::error_code error;
  const fs::path tree = fs::canonical(root, error);
  if (error) {
    return false;
  }
  const fs::path at = fs::canonical(path, error);
  if (error) {
    return false;
  }
  if (Beneath(at, tree)) {
    return true;
  }
  // A Hugging Face cache: <repository>/snapshots/<revision>/... links into
  // <repository>/blobs/ (itself not followed: a blobs/ that is a link
  // elsewhere does not count).
  for (fs::path at_or_above = tree; at_or_above.has_relative_path();
       at_or_above = at_or_above.parent_path()) {
    if (at_or_above.parent_path().filename() == "snapshots") {
      const fs::path blobs = at_or_above.parent_path().parent_path() / "blobs";
      if (!fs::is_symlink(blobs, error) && fs::is_directory(blobs, error)) {
        return Beneath(at, blobs);
      }
      return false;
    }
  }
  return false;
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
