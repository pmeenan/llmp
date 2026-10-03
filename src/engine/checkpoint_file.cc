// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "engine/checkpoint_file.h"

#include <unistd.h>

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <limits>
#include <system_error>
#include <utility>

#include "engine/support.h"
#include "platform/direct_io.h"

namespace jitllm::engine {
namespace {

using Range = LiveState::Range;
constexpr std::uint64_t kAlignment = platform::kDirectIoAlignment;

auto Failure(std::string detail, bool invalid = false, bool cancelled = false) {
  return std::unexpected(CheckpointFailure{
      .detail = std::move(detail), .invalid_state = invalid, .cancelled = cancelled});
}

// Allocate enough for an aligned view even if a future provider's pinned
// allocation has less alignment than direct I/O. Keep/free the original base.
class Staging {
 public:
  explicit Staging(PagedNode& node) : node_(node) {}
  Staging(const Staging&) = delete;
  Staging& operator=(const Staging&) = delete;
  Staging(Staging&&) = delete;
  Staging& operator=(Staging&&) = delete;
  ~Staging() {
    if (base_ != nullptr) {
      if (reserved_) {
        node_.ReturnStaging(base_, !unknown_);
      } else if (unknown_) {
        node_.KeepPinned(base_);
      } else {
        if (auto freed = node_.FreePinned(base_); !freed) {
          node_.KeepPinned(base_);
        }
      }
    }
  }
  // The node's reserved staging when it is free (PagedNode::ReserveStaging:
  // a full budget never starves a checkpoint), else an allocation of its
  // own, for which the node reclaims room first.
  Status Open() {
    const std::uint64_t kBytes = CheckpointStagingBytes();
    if (void* reserved = node_.TakeStaging(kBytes); reserved != nullptr) {
      base_ = reserved;
      reserved_ = true;
    } else {
      std::vector<catalog::ExtentId> extents;
      auto pinned = node_.Pinned(kBytes, kShared, extents);
      if (!pinned) {
        return std::unexpected(pinned.error());
      }
      base_ = *pinned;
    }
    memory_ = static_cast<std::byte*>(
        support::Pointer(support::Round(reinterpret_cast<std::uintptr_t>(base_), kAlignment)));
    return {};
  }
  std::byte* memory() const { return memory_; }
  void Unknown() { unknown_ = true; }

 private:
  PagedNode& node_;
  void* base_ = nullptr;
  std::byte* memory_ = nullptr;
  bool unknown_ = false;
  bool reserved_ = false;
};

Status Validate(std::span<const Range> ranges) {
  std::uint64_t file_bytes = 0;
  for (const Range& range : ranges) {
    if (range.bytes == 0 || range.bytes > kPagedExtent ||
        range.offset > std::numeric_limits<std::uint64_t>::max() - range.bytes) {
      return support::Error("invalid checkpoint page range");
    }
    const std::uint64_t transfer = support::Round(range.bytes, kAlignment);
    if (transfer > std::numeric_limits<std::int64_t>::max() - file_bytes) {
      return support::Error("checkpoint file offsets overflow");
    }
    file_bytes += transfer;
  }
  return {};
}

}  // namespace

std::uint64_t CheckpointStagingBytes() { return kPagedExtent + kAlignment; }

std::expected<std::vector<Range>, std::string> CheckpointPages(std::span<const Range> used,
                                                               std::span<const Range> writes) {
  if (auto valid = Validate(used); !valid) {
    return std::unexpected(valid.error());
  }
  for (const Range& range : writes) {
    if (range.offset > std::numeric_limits<std::uint64_t>::max() - range.bytes) {
      return support::Error("checkpoint writes overflow their region");
    }
  }
  std::vector<Range> out;
  for (const Range& range : used) {
    if (range.offset % kPagedExtent != 0) {
      return support::Error("checkpoint used ranges are not physical pages");
    }
    const bool mutable_page = std::ranges::any_of(writes, [&](const Range& write) {
      return write.region == range.region && write.bytes != 0 &&
             write.offset < range.offset + range.bytes && range.offset < write.offset + write.bytes;
    });
    if (mutable_page) {
      out.push_back(range);
    }
  }
  return out;
}

CheckpointFile::CheckpointFile(CheckpointFile&& other) noexcept
    : fd_(std::exchange(other.fd_, -1)), bytes_(other.bytes_), ranges_(std::move(other.ranges_)) {}

CheckpointFile& CheckpointFile::operator=(CheckpointFile&& other) noexcept {
  if (this != &other) {
    if (fd_ >= 0) {
      (void)::close(fd_);
    }
    fd_ = std::exchange(other.fd_, -1);
    bytes_ = other.bytes_;
    ranges_ = std::move(other.ranges_);
  }
  return *this;
}

CheckpointFile::~CheckpointFile() {
  if (fd_ >= 0) {
    (void)::close(fd_);
  }
}

std::expected<CheckpointFile, CheckpointFailure> CheckpointFile::Capture(
    PagedNode& node, const std::filesystem::path& directory, std::span<const Range> ranges,
    const Copy& copy, const Continue& go_on) {
  if (auto valid = Validate(ranges); !valid) {
    return Failure(valid.error());
  }
  if (go_on && !go_on()) {
    return Failure("checkpoint capture cancelled", false, true);
  }
  auto opened = platform::OpenUnnamedDirectFile(directory);
  if (!opened) {
    return Failure("opening checkpoint file: " + std::system_category().message(opened.error()));
  }
  CheckpointFile checkpoint;
  checkpoint.fd_ = *opened;
  Staging staging(node);
  if (auto allocated = staging.Open(); !allocated) {
    return Failure(allocated.error());
  }
  std::uint64_t at = 0;
  for (const Range& range : ranges) {
    if (go_on && !go_on()) {
      return Failure("checkpoint capture cancelled", false, true);
    }
    if (auto copied = copy(staging.memory(), std::span(&range, 1)); !copied) {
      staging.Unknown();
      return Failure(copied.error(), true);
    }
    const std::uint64_t transfer = support::Round(range.bytes, kAlignment);
    std::memset(staging.memory() + range.bytes, 0, transfer - range.bytes);
    auto written = platform::TransferDirectFile(
        checkpoint.fd_, at, std::span(staging.memory(), static_cast<std::size_t>(transfer)), true);
    if (!written) {
      return Failure("writing checkpoint: " + std::system_category().message(written.error()));
    }
    at += transfer;
    checkpoint.bytes_ += range.bytes;
  }
  if (go_on && !go_on()) {
    return Failure("checkpoint capture cancelled", false, true);
  }
  checkpoint.ranges_.assign(ranges.begin(), ranges.end());
  return checkpoint;
}

std::expected<void, CheckpointFailure> CheckpointFile::Restore(PagedNode& node,
                                                               const Prepare& prepare,
                                                               const Copy& copy,
                                                               const Continue& go_on) const {
  if (fd_ < 0) {
    return Failure("no completed checkpoint file");
  }
  if (go_on && !go_on()) {
    return Failure("checkpoint restore cancelled", false, true);
  }
  Staging staging(node);
  if (auto allocated = staging.Open(); !allocated) {
    return Failure(allocated.error());
  }
  if (auto prepared = prepare(); !prepared) {
    return Failure(prepared.error(), true);
  }
  std::uint64_t at = 0;
  for (const Range& range : ranges_) {
    if (go_on && !go_on()) {
      return Failure("checkpoint restore cancelled", true, true);
    }
    const std::uint64_t transfer = support::Round(range.bytes, kAlignment);
    auto read = platform::TransferDirectFile(
        fd_, at, std::span(staging.memory(), static_cast<std::size_t>(transfer)), false);
    if (!read) {
      return Failure("reading checkpoint: " + std::system_category().message(read.error()), true);
    }
    if (auto copied = copy(staging.memory(), std::span(&range, 1)); !copied) {
      staging.Unknown();
      return Failure(copied.error(), true);
    }
    at += transfer;
  }
  if (go_on && !go_on()) {
    return Failure("checkpoint restore cancelled", true, true);
  }
  return {};
}

}  // namespace jitllm::engine
