// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "qwen38_reference.h"

#include <gtest/gtest.h>
#include <unistd.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <format>
#include <limits>
#include <span>
#include <string>
#include <system_error>
#include <vector>

#include "base/sha256.h"

namespace llmp::benchmarks::draft_vocab {
struct ReferencePagesTestAccess {
  static bool Resize(const ReferencePages& reference, std::uint64_t bytes) {
    return ::ftruncate(reference.fd_, static_cast<off_t>(bytes)) == 0;
  }
};
}  // namespace llmp::benchmarks::draft_vocab

namespace {
namespace dv = llmp::benchmarks::draft_vocab;

class Qwen38Reference : public ::testing::Test {
 protected:
  void SetUp() override {
    // NOLINTNEXTLINE(concurrency-mt-unsafe): immutable test environment
    const char* scratch = std::getenv("LLMP_TEST_SCRATCH");
    ASSERT_NE(scratch, nullptr);
    directory_ = std::filesystem::path(scratch) / "qwen38-reference";
    std::error_code error;
    std::filesystem::create_directories(directory_, error);
    ASSERT_FALSE(error);
  }
  static std::expected<void, std::string> Copy(void* memory, const dv::PageRange& range) {
    std::memset(memory, range.offset == 0 ? 0x35 : 0x62, static_cast<std::size_t>(range.bytes));
    return {};
  }
  const std::array<dv::PageRange, 2> ranges_ = {
      dv::PageRange{.region = 0, .offset = 0, .bytes = 37},
      dv::PageRange{.region = 1, .offset = dv::ReferencePages::kPage, .bytes = 4101}};
  alignas(4096) std::array<std::byte, 8192> expected_{};
  alignas(4096) std::array<std::byte, 8192> live_{};
  std::filesystem::path directory_;
  dv::ReferenceStats stats_;
};

TEST_F(Qwen38Reference, AuthenticatesEveryLogicalByteAndPadsOnlyTheFileTransfer) {
  auto saved = dv::ReferencePages::Capture(directory_, ranges_, 1, live_, Copy, stats_);
  ASSERT_TRUE(saved);
  EXPECT_EQ(saved->bytes(), 4138);
  llmp::base::Sha256 hash;
  for (const auto& range : ranges_) {
    ASSERT_TRUE(Copy(live_.data(), range));
    hash.Update(std::format("{}:{}:{};", range.region, range.offset, range.bytes));
    hash.Update(std::span(live_).first(static_cast<std::size_t>(range.bytes)));
  }
  EXPECT_EQ(saved->sha256(), llmp::base::ToHex(hash.Finish()));
  ASSERT_TRUE(saved->Compare(ranges_, 1, expected_, live_, Copy, stats_));
  EXPECT_EQ(stats_.captures, 1);
  EXPECT_EQ(stats_.comparisons, 1);
  EXPECT_EQ(stats_.capture_bytes, 4138);
  EXPECT_EQ(stats_.comparison_bytes, 4138);
  EXPECT_EQ(stats_.write_bytes, 12288);
  EXPECT_EQ(stats_.read_bytes, 12288);
  auto changed = [](void* memory, const dv::PageRange& range) -> std::expected<void, std::string> {
    auto copied = Copy(memory, range);
    static_cast<std::byte*>(memory)[range.bytes - 1] ^= std::byte{1};
    return copied;
  };
  const auto mismatch = saved->Compare(ranges_, 1, expected_, live_, changed, stats_);
  ASSERT_FALSE(mismatch);
  EXPECT_FALSE(mismatch.error().unknown_copy);
  EXPECT_EQ(stats_.comparisons, 1);
}

TEST_F(Qwen38Reference, SignedZeroAndFailedCompletionCannotQualify) {
  const std::array<dv::PageRange, 1> range = {dv::PageRange{.bytes = 4}};
  const auto zero = [](void* memory, const dv::PageRange&) -> std::expected<void, std::string> {
    const std::uint32_t bits = 0;
    std::memcpy(memory, &bits, sizeof(bits));
    return {};
  };
  auto saved = dv::ReferencePages::Capture(directory_, range, 0, live_, zero, stats_);
  ASSERT_TRUE(saved);
  const auto negative_zero = [](void* memory,
                                const dv::PageRange&) -> std::expected<void, std::string> {
    const std::uint32_t bits = 0x80000000;
    std::memcpy(memory, &bits, sizeof(bits));
    return {};
  };
  auto mismatch = saved->Compare(range, 0, expected_, live_, negative_zero, stats_);
  ASSERT_FALSE(mismatch);
  EXPECT_FALSE(mismatch.error().unknown_copy);
  const auto unknown = [](void* memory, const dv::PageRange&) -> std::expected<void, std::string> {
    static_cast<std::byte*>(memory)[0] = std::byte{0x11};
    return std::unexpected("submitted copy has unproved completion");
  };
  auto failed = saved->Compare(range, 0, expected_, live_, unknown, stats_);
  ASSERT_FALSE(failed);
  EXPECT_TRUE(failed.error().unknown_copy);
  auto failed_capture = dv::ReferencePages::Capture(directory_, range, 0, live_, unknown, stats_);
  ASSERT_FALSE(failed_capture);
  EXPECT_TRUE(failed_capture.error().unknown_copy);
}

TEST_F(Qwen38Reference, RejectsMissingExtraReorderedShortRangesCursorAndAliasingBeforeCopy) {
  auto saved = dv::ReferencePages::Capture(directory_, ranges_, 1, live_, Copy, stats_);
  ASSERT_TRUE(saved);
  std::uint32_t copies = 0;
  const auto counted = [&](void* memory, const dv::PageRange& range) {
    ++copies;
    return Copy(memory, range);
  };
  auto ranges = std::vector<dv::PageRange>(ranges_.begin(), ranges_.end());
  EXPECT_FALSE(saved->Compare(std::span(ranges).first(1), 1, expected_, live_, counted, stats_));
  ranges.push_back({.region = 2, .offset = 0, .bytes = 1});
  EXPECT_FALSE(saved->Compare(ranges, 1, expected_, live_, counted, stats_));
  ranges.assign(ranges_.rbegin(), ranges_.rend());
  EXPECT_FALSE(saved->Compare(ranges, 1, expected_, live_, counted, stats_));
  ranges.assign(ranges_.begin(), ranges_.end());
  --ranges.back().bytes;
  EXPECT_FALSE(saved->Compare(ranges, 1, expected_, live_, counted, stats_));
  EXPECT_FALSE(saved->Compare(ranges_, 2, expected_, live_, counted, stats_));
  EXPECT_FALSE(saved->Compare(ranges_, 1, live_, live_, counted, stats_));
  EXPECT_EQ(copies, 0);
}

TEST_F(Qwen38Reference, RejectsShortOrExtraFileBeforeCopyAndChecksAllRangeBounds) {
  auto saved = dv::ReferencePages::Capture(directory_, ranges_, 1, live_, Copy, stats_);
  ASSERT_TRUE(saved);
  ASSERT_TRUE(dv::ReferencePagesTestAccess::Resize(*saved, 4096));
  std::uint32_t copies = 0;
  const auto counted = [&](void* memory, const dv::PageRange& range) {
    ++copies;
    return Copy(memory, range);
  };
  EXPECT_FALSE(saved->Compare(ranges_, 1, expected_, live_, counted, stats_));
  ASSERT_TRUE(dv::ReferencePagesTestAccess::Resize(*saved, 16384));
  EXPECT_FALSE(saved->Compare(ranges_, 1, expected_, live_, counted, stats_));
  EXPECT_EQ(copies, 0);
  auto ranges = std::vector<dv::PageRange>(ranges_.begin(), ranges_.end());
  ranges[0].bytes = 0;
  EXPECT_FALSE(dv::ReferencePages::Capture(directory_, ranges, 0, live_, counted, stats_));
  ranges[0].bytes = dv::ReferencePages::kPage + 1;
  EXPECT_FALSE(dv::ReferencePages::Capture(directory_, ranges, 0, live_, counted, stats_));
  ranges[0] = {.region = 0, .offset = std::numeric_limits<std::uint64_t>::max() - 3, .bytes = 8};
  EXPECT_FALSE(dv::ReferencePages::Capture(directory_, ranges, 0, live_, counted, stats_));
  ranges.assign(dv::ReferencePages::kMaxRanges + 1, dv::PageRange{.bytes = 1});
  EXPECT_FALSE(dv::ReferencePages::Capture(directory_, ranges, 0, live_, counted, stats_));
  EXPECT_FALSE(dv::ReferencePages::Capture(directory_, ranges_, 33793, live_, counted, stats_));
  EXPECT_FALSE(dv::ReferencePages::Capture(directory_, ranges_, 0, std::span(live_).subspan(1),
                                           counted, stats_));
  EXPECT_FALSE(dv::ReferencePages::Capture(directory_, ranges_, 0, std::span(live_).first(4096),
                                           counted, stats_));
  EXPECT_EQ(copies, 0);
}
}  // namespace
