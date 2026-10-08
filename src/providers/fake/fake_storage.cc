// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "providers/fake/fake_storage.h"

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <utility>
#include <vector>

#include "providers/storage.h"

namespace llmp::providers::fake {

FakeStorage::FakeStorage(std::size_t depth, std::uint32_t alignment)
    : depth_(depth), alignment_(alignment) {}

int FakeStorage::AddFile(std::vector<std::byte> contents) {
  const int fd = next_fd_++;
  files_.emplace(fd, std::move(contents));
  return fd;
}

std::span<const std::byte> FakeStorage::Contents(int fd) const {
  const auto found = files_.find(fd);
  return found != files_.end() ? std::span<const std::byte>(found->second)
                               : std::span<const std::byte>();
}

bool FakeStorage::Release(std::uint64_t token) {
  const auto found = requests_.find(token);
  if (found == requests_.end() || !found->second.held) {
    return false;
  }
  found->second.held = false;
  return true;
}

Submission FakeStorage::Submit(const IoRequest& request) {
  // A request refused for room never reaches the script: the next one the
  // fake has room for takes it.
  if (requests_.size() >= depth_ || requests_.contains(request.token)) {
    return Submission::kNotStarted;
  }
  if (!request.segments.empty()) {
    // As io_uring's provider refuses one: `length` is the segments' sum.
    std::uint64_t total = 0;
    for (const IoSegment& segment : request.segments) {
      total += segment.length;
    }
    if (total != request.length) {
      return Submission::kNotStarted;
    }
  }
  Script script;
  if (!scripts_.empty()) {
    script = scripts_.front();
    scripts_.pop_front();
  }
  if (script.submission == Submission::kNotStarted) {
    return Submission::kNotStarted;
  }
  IoRequest kept = request;
  if (!request.segments.empty()) {
    // The caller's segments need outlive only Submit: keep a copy, which
    // submitted() also shows.
    kept.segments = segments_.emplace_back(request.segments.begin(), request.segments.end());
  }
  submitted_.push_back(kept);
  requests_.emplace(request.token, Pending{.request = kept,
                                           .result = script.result,
                                           .held = script.hold,
                                           .cancelled = false,
                                           .sequence = next_sequence_++});
  return script.submission;
}

Submission FakeStorage::Cancel(std::uint64_t token) {
  const auto found = requests_.find(token);
  if (found == requests_.end()) {
    return Submission::kNotStarted;
  }
  if (found->second.held) {
    found->second.cancelled = true;
    found->second.held = false;
  }
  return Submission::kAccepted;
}

std::int64_t FakeStorage::Perform(const Pending& pending) {
  const IoRequest& request = pending.request;
  if (pending.cancelled) {
    return -ECANCELED;
  }
  if (pending.result && *pending.result < 0) {
    return *pending.result;
  }
  // A plain request is one segment.
  const IoSegment whole{.memory = request.memory, .length = request.length};
  const std::span<const IoSegment> segments =
      request.segments.empty() ? std::span<const IoSegment>(&whole, 1) : request.segments;
  if (request.offset % alignment_ != 0 || segments.size() > kMaxSegments) {
    return -EINVAL;
  }
  std::uint64_t total = 0;
  for (const IoSegment& segment : segments) {
    const auto address = reinterpret_cast<std::uintptr_t>(segment.memory);
    if (address % alignment_ != 0 || segment.length % alignment_ != 0) {
      return -EINVAL;  // as the kernel refuses unaligned direct I/O
    }
    total += segment.length;
  }
  const auto file = files_.find(request.fd);
  if (file == files_.end()) {
    return -EBADF;
  }
  std::vector<std::byte>& contents = file->second;
  std::uint64_t count = total;
  if (pending.result) {
    count = std::min<std::uint64_t>(count, static_cast<std::uint64_t>(*pending.result));
  }
  if (request.kind == IoKind::kRead) {
    const std::uint64_t available =
        request.offset < contents.size() ? contents.size() - request.offset : 0;
    count = std::min(count, available);
    if (count == 0) {
      return 0;  // EOF may be arbitrarily far past the vector's end.
    }
  } else if (contents.size() < request.offset + count) {
    contents.resize(request.offset + count);
  }
  // The count fills the segments in order.
  std::uint64_t moved = 0;
  for (const IoSegment& segment : segments) {
    const std::uint64_t part = std::min<std::uint64_t>(segment.length, count - moved);
    if (part == 0) {
      break;
    }
    std::byte* file_at = contents.data() + request.offset +
                         moved;  // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    if (request.kind == IoKind::kRead) {
      std::memcpy(segment.memory, file_at, part);
    } else {
      std::memcpy(file_at, segment.memory, part);
    }
    moved += part;
  }
  return static_cast<std::int64_t>(count);
}

std::size_t FakeStorage::Harvest(std::span<IoCompletion> out, bool /*wait*/) {
  // Everything due completes, in submission order.
  std::vector<std::pair<std::uint64_t, std::uint64_t>> due;  // sequence, token
  for (const auto& [token, pending] : requests_) {
    if (!pending.held) {
      due.emplace_back(pending.sequence, token);
    }
  }
  std::ranges::sort(due);
  std::size_t produced = 0;
  for (const auto& [sequence, token] : due) {
    if (produced == out.size()) {
      break;
    }
    const auto found = requests_.find(token);
    out[produced++] = IoCompletion{.token = token, .result = Perform(found->second)};
    requests_.erase(found);
  }
  return produced;
}

}  // namespace llmp::providers::fake
