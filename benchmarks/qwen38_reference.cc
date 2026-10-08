// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "qwen38_reference.h"

#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <format>
#include <limits>
#include <system_error>
#include <utility>

#include "base/sha256.h"
#include "platform/direct_io.h"

namespace llmp::benchmarks::draft_vocab {
namespace {
using Clock = std::chrono::steady_clock;
constexpr std::uint64_t kAlignment = platform::kDirectIoAlignment;

auto Failure(std::string detail, bool unknown = false) {
  return std::unexpected(ReferenceFailure{.detail = std::move(detail), .unknown_copy = unknown});
}

double Seconds(Clock::duration duration) { return std::chrono::duration<double>(duration).count(); }

std::uint64_t Rounded(std::uint64_t bytes) {
  return ((bytes + kAlignment - 1) / kAlignment) * kAlignment;
}

std::expected<std::uint64_t, ReferenceFailure> Validate(std::span<const PageRange> ranges,
                                                        std::span<std::byte> memory) {
  if (ranges.empty() || ranges.size() > ReferencePages::kMaxRanges || memory.data() == nullptr ||
      reinterpret_cast<std::uintptr_t>(memory.data()) % kAlignment != 0) {
    return Failure("invalid reference page count or aligned staging view");
  }
  std::uint64_t bytes = 0;
  std::uint64_t transfers = 0;
  const PageRange* previous = nullptr;
  for (const PageRange& range : ranges) {
    if (range.bytes == 0 || range.bytes > ReferencePages::kPage ||
        range.offset % ReferencePages::kPage != 0 ||
        range.offset > std::numeric_limits<std::uint64_t>::max() - range.bytes ||
        Rounded(range.bytes) > memory.size() || range.bytes > ReferencePages::kMaxBytes - bytes ||
        Rounded(range.bytes) > std::numeric_limits<std::int64_t>::max() - transfers ||
        (previous != nullptr && (range.region < previous->region ||
                                 (range.region == previous->region &&
                                  range.offset < previous->offset + previous->bytes)))) {
      return Failure("invalid, reordered or overlapping reference page geometry");
    }
    bytes += range.bytes;
    transfers += Rounded(range.bytes);
    previous = &range;
  }
  return bytes;
}

bool Overlap(std::span<std::byte> first, std::span<std::byte> second) {
  const auto a = reinterpret_cast<std::uintptr_t>(first.data());
  const auto b = reinterpret_cast<std::uintptr_t>(second.data());
  return a <= b ? b - a < first.size() : a - b < second.size();
}

bool FileExtent(int fd, std::uint64_t bytes) {
  struct stat info{};
  return ::fstat(fd, &info) == 0 && info.st_size >= 0 && std::cmp_equal(info.st_size, bytes);
}
}  // namespace

ReferencePages::ReferencePages(ReferencePages&& other) noexcept
    : fd_(std::exchange(other.fd_, -1)),
      bytes_(other.bytes_),
      file_bytes_(other.file_bytes_),
      cursor_(other.cursor_),
      ranges_(std::move(other.ranges_)),
      sha256_(std::move(other.sha256_)) {}

ReferencePages& ReferencePages::operator=(ReferencePages&& other) noexcept {
  if (this != &other) {
    if (fd_ >= 0) {
      (void)::close(fd_);
    }
    fd_ = std::exchange(other.fd_, -1);
    bytes_ = other.bytes_;
    file_bytes_ = other.file_bytes_;
    cursor_ = other.cursor_;
    ranges_ = std::move(other.ranges_);
    sha256_ = std::move(other.sha256_);
  }
  return *this;
}

ReferencePages::~ReferencePages() {
  if (fd_ >= 0) {
    (void)::close(fd_);
  }
}

std::expected<ReferencePages, ReferenceFailure> ReferencePages::Capture(
    const std::filesystem::path& directory, std::span<const PageRange> ranges, std::uint32_t cursor,
    std::span<std::byte> live, const Copy& copy, ReferenceStats& stats) {
  const auto started = Clock::now();
  auto valid = Validate(ranges, live);
  if (!valid) {
    return std::unexpected(valid.error());
  }
  if (!copy || cursor > 33792) {
    return Failure("missing completed reference page copy");
  }
  auto file = platform::OpenUnnamedDirectFile(directory);
  if (!file) {
    return Failure("opening reference pages: " + std::system_category().message(file.error()));
  }
  ReferencePages reference;
  reference.fd_ = *file;
  reference.cursor_ = cursor;
  base::Sha256 hash;
  std::uint64_t at = 0;
  for (const PageRange& range : ranges) {
    const auto copy_start = Clock::now();
    if (auto copied = copy(live.data(), range); !copied) {
      return Failure(copied.error(), true);
    }
    stats.capture_copy_seconds += Seconds(Clock::now() - copy_start);
    const auto sha_start = Clock::now();
    hash.Update(std::format("{}:{}:{};", range.region, range.offset, range.bytes));
    hash.Update(live.first(static_cast<std::size_t>(range.bytes)));
    stats.sha_seconds += Seconds(Clock::now() - sha_start);
    const auto transfer = static_cast<std::size_t>(Rounded(range.bytes));
    std::memset(live.data() + range.bytes, 0, transfer - range.bytes);
    const auto write_start = Clock::now();
    auto written = platform::TransferDirectFile(reference.fd_, at, live.first(transfer), true);
    if (!written) {
      return Failure("writing reference pages: " + std::system_category().message(written.error()));
    }
    stats.write_seconds += Seconds(Clock::now() - write_start);
    stats.write_bytes += transfer;
    at += transfer;
  }
  if (!FileExtent(reference.fd_, at)) {
    return Failure("reference file extent differs after complete capture");
  }
  reference.bytes_ = *valid;
  reference.file_bytes_ = at;
  reference.ranges_.assign(ranges.begin(), ranges.end());
  reference.sha256_ = base::ToHex(hash.Finish());
  stats.capture_bytes += *valid;
  ++stats.captures;
  stats.capture_seconds += Seconds(Clock::now() - started);
  return reference;
}

std::expected<void, ReferenceFailure> ReferencePages::Compare(
    std::span<const PageRange> ranges, std::uint32_t cursor, std::span<std::byte> expected,
    std::span<std::byte> live, const Copy& copy, ReferenceStats& stats) const {
  const auto started = Clock::now();
  auto valid = Validate(ranges, expected);
  if (!valid) {
    return std::unexpected(valid.error());
  }
  if (auto live_valid = Validate(ranges, live); !live_valid) {
    return std::unexpected(live_valid.error());
  }
  if (fd_ < 0 || cursor != cursor_ || !std::ranges::equal(ranges, ranges_) || !copy ||
      Overlap(expected, live)) {
    return Failure("reference cursor, complete geometry or disjoint staging views differ");
  }
  if (!FileExtent(fd_, file_bytes_)) {
    return Failure("reference file is short or has extra bytes");
  }
  std::uint64_t at = 0;
  for (const PageRange& range : ranges) {
    const auto transfer = static_cast<std::size_t>(Rounded(range.bytes));
    const auto read_start = Clock::now();
    auto read = platform::TransferDirectFile(fd_, at, expected.first(transfer), false);
    if (!read) {
      return Failure("reading reference pages: " + std::system_category().message(read.error()));
    }
    stats.read_seconds += Seconds(Clock::now() - read_start);
    stats.read_bytes += transfer;
    const auto copy_start = Clock::now();
    if (auto copied = copy(live.data(), range); !copied) {
      return Failure(copied.error(), true);
    }
    stats.comparison_copy_seconds += Seconds(Clock::now() - copy_start);
    const auto compare_start = Clock::now();
    const bool equal =
        std::memcmp(expected.data(), live.data(), static_cast<std::size_t>(range.bytes)) == 0;
    stats.byte_compare_seconds += Seconds(Clock::now() - compare_start);
    if (!equal) {
      return Failure(
          std::format("reference byte mismatch at region {} page {}", range.region, range.offset));
    }
    at += transfer;
  }
  stats.comparison_bytes += *valid;
  ++stats.comparisons;
  stats.comparison_seconds += Seconds(Clock::now() - started);
  return {};
}

}  // namespace llmp::benchmarks::draft_vocab
