// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "fp16_common.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <format>
#include <fstream>
#include <iterator>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>

#include "base/sha256.h"

namespace llmp::benchmarks {
namespace {

std::unexpected<std::string> Error(std::string what) { return std::unexpected(std::move(what)); }

}  // namespace

std::expected<Trajectory, std::string> LoadTrajectory(std::string_view name,
                                                      const std::filesystem::path& tokens) {
  Trajectory t;
  t.name = std::string(name);
  struct File {
    int fd;
    ~File() {
      if (fd >= 0) (void)::close(fd);
    }
  } file{::open(tokens.c_str(), O_RDONLY | O_CLOEXEC | O_NONBLOCK | O_NOFOLLOW)};
  struct stat metadata{};
  if (file.fd < 0 || ::fstat(file.fd, &metadata) != 0 || !S_ISREG(metadata.st_mode) ||
      metadata.st_size <= 0 || metadata.st_size > 1040 * 8)
    return Error("trajectory must be a bounded regular fixture file");
  std::string bytes(static_cast<std::size_t>(metadata.st_size), '\0');
  std::size_t used = 0;
  while (used < bytes.size()) {
    const auto read = ::read(file.fd, bytes.data() + used, bytes.size() - used);
    if (read < 0 && errno == EINTR) continue;
    if (read <= 0) return Error("trajectory changed during its bounded read");
    used += static_cast<std::size_t>(read);
  }
  char extra = 0;
  ssize_t tail = 0;
  do {
    tail = ::read(file.fd, &extra, 1);
  } while (tail < 0 && errno == EINTR);
  if (tail != 0) return Error("trajectory changed during its bounded read");
  if (name == "control") {
    std::istringstream in(bytes);
    std::int64_t id = 0;
    while (in >> id) {
      if (id < 0 || id >= 151936 || t.tokens.size() == 76)
        return Error("control token is outside its fixed fixture domain");
      t.tokens.push_back(static_cast<std::int32_t>(id));
    }
    if (!in.eof()) return Error("control contains malformed token text");
    if (t.tokens.size() != 76) {
      return Error(std::format("control has 76 tokens, not {}", t.tokens.size()));
    }
    t.chunks.push_back(32);
    t.chunks.insert(t.chunks.end(), 44, 1);
    t.cells = 512;
    t.restore_after = 32;
  } else if (name == "heldout") {
    base::Sha256 hash;
    hash.Update(std::as_bytes(std::span(bytes)));
    const std::string digest = base::ToHex(hash.Finish());
    if (bytes.size() != std::size_t{1040} * 8 ||
        digest != "6dd8da897821f5615e7796f4d882795faf306a941f050e2eb68b619096e3fb0c") {
      return Error("not the declared held-out IDs");
    }
    t.chunks = {16, 17};
    t.chunks.insert(t.chunks.end(), 16, 1);
    t.chunks.push_back(512);
    t.chunks.insert(t.chunks.end(), 16, 1);
    std::size_t total = 0;
    for (const std::uint32_t c : t.chunks) {
      total += c;
    }
    for (std::size_t i = 0; i < total; ++i) {
      std::uint64_t id = 0;
      for (int b = 7; b >= 0; --b) {
        id = (id << 8U) | static_cast<unsigned char>(bytes[(i * 8) + static_cast<std::size_t>(b)]);
      }
      if (id >= 151936) return Error("heldout token is outside its fixed fixture domain");
      t.tokens.push_back(static_cast<std::int32_t>(id));
    }
    t.cells = 1024;
    t.restore_after = 33;
  } else {
    return Error("the trajectory is control or heldout");
  }
  return t;
}

}  // namespace llmp::benchmarks
