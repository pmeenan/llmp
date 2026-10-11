// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#include "import/stream.h"

#include <sys/stat.h>

#include <algorithm>
#include <vector>

namespace llmp::import {
std::expected<base::Sha256Digest, Error> StreamSource(const SourceFile& source, std::uint64_t file,
                                                      const SourceSink& sink,
                                                      std::optional<base::Sha256Digest> expected) {
  const auto fail = [file](std::string_view reason) {
    return std::unexpected(Error{.rule = Rule::kIo, .reason = reason, .file = file});
  };
  if (source.parsed.bytes > source.bytes || source.path.empty()) {
    return fail("a source has no valid file or parsed-prefix receipt");
  }
  auto input = OpenCheckpointFile(source.path, file);
  if (!input) return std::unexpected(input.error());
  if (input->bytes != source.bytes || input->device != source.device ||
      input->inode != source.inode) {
    return fail("a source changed identity or size after parsing");
  }
  struct stat before{};
  if (::fstat(input->fd.get(), &before) != 0) return fail("cannot examine source before streaming");
  if (before.st_size < 0 || static_cast<std::uint64_t>(before.st_size) != source.bytes)
    return fail("a source changed size before streaming");
  std::vector<std::byte> buffer(1U << 20U);
  base::Sha256 full, prefix;
  for (std::uint64_t offset = 0; offset < source.bytes;) {
    auto bytes = std::span(buffer).first(
        static_cast<std::size_t>(std::min<std::uint64_t>(buffer.size(), source.bytes - offset)));
    if (auto read = ReadExactly(input->fd.get(), offset, bytes, file); !read)
      return std::unexpected(read.error());
    full.Update(bytes);
    if (offset < source.parsed.bytes)
      prefix.Update(bytes.first(static_cast<std::size_t>(
          std::min<std::uint64_t>(bytes.size(), source.parsed.bytes - offset))));
    if (auto copied = sink(offset, bytes); !copied) return std::unexpected(copied.error());
    offset += bytes.size();
  }
  struct stat after{};
  if (::fstat(input->fd.get(), &after) != 0 || after.st_size != before.st_size ||
      after.st_mtim.tv_sec != before.st_mtim.tv_sec ||
      after.st_mtim.tv_nsec != before.st_mtim.tv_nsec ||
      after.st_ctim.tv_sec != before.st_ctim.tv_sec ||
      after.st_ctim.tv_nsec != before.st_ctim.tv_nsec) {
    return fail("a source changed during streaming");
  }
  if (source.parsed.bytes != 0 && prefix.Finish() != source.parsed.sha256)
    return fail("parsed source bytes changed before streaming");
  auto digest = full.Finish();
  if (expected && digest != *expected)
    return fail("a source hash differs from its archive manifest");
  return digest;
}
}  // namespace llmp::import
