// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The storage provider and whole reads over it (D-034, D-048): short
// transfers, alignment, retries, cancellation that drains, and coalesced
// duplicate reads, on the scripted fake; and the io_uring provider
// against a real direct-I/O file and a pipe whose read never completes
// (skipped where there is no io_uring, as under qemu-user or a container
// policy that denies the syscall).

#include "providers/storage.h"

#include <fcntl.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <unistd.h>

#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

#include "expected_error.h"
#include "platform/direct_io.h"
#include "providers/direct_reader.h"
#include "providers/fake/fake_storage.h"
#include "providers/uring_storage.h"

#ifdef __SANITIZE_THREAD__
#define LLMP_TEST_TSAN 1
#elifdef __has_feature
#if __has_feature(thread_sanitizer)
#define LLMP_TEST_TSAN 1
#endif
#endif

namespace {

using llmp::providers::DirectReader;
using llmp::providers::FinishedRead;
using llmp::providers::IoCompletion;
using llmp::providers::IoKind;
using llmp::providers::IoRequest;
using llmp::providers::ReadError;
using llmp::providers::ReaderSettings;
using llmp::providers::ReadOutcome;
using llmp::providers::ReadSpec;
using llmp::providers::Submission;
using llmp::providers::UringStorage;
using llmp::providers::fake::FakeStorage;
using llmp::test_support::Failed;
using ::testing::ElementsAre;
using ::testing::IsEmpty;

constexpr std::uint64_t kAlignment = 4096;

std::vector<std::byte> Pattern(std::size_t size) {
  std::vector<std::byte> bytes(size);
  for (std::size_t i = 0; i < size; ++i) {
    bytes[i] = static_cast<std::byte>((i * 7) + (i >> 12));
  }
  return bytes;
}

// Aligned memory for direct I/O.
struct Buffer {
  explicit Buffer(std::size_t size) : size(size) {
    data = static_cast<std::byte*>(std::aligned_alloc(kAlignment, size));
    std::memset(data, 0xee, size);
  }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  Buffer(Buffer&&) = delete;
  Buffer& operator=(Buffer&&) = delete;
  ~Buffer() { std::free(data); }  // NOLINT(cppcoreguidelines-no-malloc)
  std::byte* data;
  std::size_t size;
};

// One request per piece of four blocks (the reader's default, no
// coalescing), as the tests of a read's own behaviour count them;
// Coalescing() lets adjacent pieces share one.
ReaderSettings Settings() {
  return ReaderSettings{.alignment = static_cast<std::uint32_t>(kAlignment),
                        .request_bytes = static_cast<std::uint32_t>(4 * kAlignment),
                        .retries = 2,
                        .reads = 4,
                        .waiters = 2,
                        .span_bytes = ReaderSettings{}.span_bytes,
                        .span_segments = llmp::providers::kMaxSegments};
}

ReaderSettings Coalescing(std::uint32_t span_blocks = 64, std::size_t segments = 64) {
  ReaderSettings settings = Settings();
  settings.reads = 16;
  settings.span_bytes = static_cast<std::uint32_t>(span_blocks * kAlignment);
  settings.span_segments = segments;
  return settings;
}

std::vector<FinishedRead> PollUntilDone(DirectReader& reader) {
  std::vector<FinishedRead> all;
  for (int i = 0; i < 16 && reader.reads() > 0; ++i) {
    for (FinishedRead& finished : reader.Poll(false)) {
      all.push_back(std::move(finished));
    }
  }
  return all;
}

class ReaderTest : public ::testing::Test {
 protected:
  FakeStorage storage_{8, static_cast<std::uint32_t>(kAlignment)};
  DirectReader reader_{storage_, Settings()};
  std::vector<std::byte> contents_ = Pattern(10 * kAlignment);
  int fd_ = storage_.AddFile(contents_);
  Buffer buffer_{10 * kAlignment};
};

TEST_F(ReaderTest, AWholeReadSplitsAndCompletes) {
  const ReadSpec spec{.fd = fd_, .offset = 0, .memory = buffer_.data, .length = 8 * kAlignment};
  EXPECT_FALSE(reader_.Read(1, spec, 100).value());
  const auto finished = PollUntilDone(reader_);
  ASSERT_EQ(finished.size(), 1U);
  EXPECT_EQ(finished[0].outcome, ReadOutcome::kComplete);
  EXPECT_EQ(finished[0].bytes, 8 * kAlignment);
  EXPECT_THAT(finished[0].waiters, ElementsAre(100));
  EXPECT_EQ(std::memcmp(buffer_.data, contents_.data(), 8 * kAlignment), 0);
  EXPECT_EQ(storage_.submitted().size(), 2U);  // two requests of four blocks
}

TEST_F(ReaderTest, ShortTransfersContinueWhereTheyStopped) {
  storage_.ScriptNext({.submission = Submission::kAccepted,
                       .result = static_cast<std::int64_t>(kAlignment),
                       .hold = false});
  const ReadSpec spec{
      .fd = fd_, .offset = kAlignment, .memory = buffer_.data, .length = 4 * kAlignment};
  ASSERT_TRUE(reader_.Read(1, spec, 100).has_value());
  const auto finished = PollUntilDone(reader_);
  ASSERT_EQ(finished.size(), 1U);
  EXPECT_EQ(finished[0].outcome, ReadOutcome::kComplete);
  ASSERT_EQ(storage_.submitted().size(), 2U);
  EXPECT_EQ(storage_.submitted()[1].offset, 2 * kAlignment);  // the remainder
  EXPECT_EQ(storage_.submitted()[1].length, 3 * kAlignment);
  EXPECT_EQ(std::memcmp(buffer_.data, contents_.data() + kAlignment, 4 * kAlignment), 0);
}

TEST_F(ReaderTest, TheEndOfTheFileEndsTheRead) {
  const ReadSpec spec{
      .fd = fd_, .offset = 8 * kAlignment, .memory = buffer_.data, .length = 4 * kAlignment};
  ASSERT_TRUE(reader_.Read(1, spec, 100).has_value());
  const auto finished = PollUntilDone(reader_);
  ASSERT_EQ(finished.size(), 1U);
  EXPECT_EQ(finished[0].outcome, ReadOutcome::kEndOfFile);
  EXPECT_EQ(finished[0].bytes, 2 * kAlignment);
}

TEST_F(ReaderTest, TransientErrorsRetryAndOthersFailAfterDraining) {
  storage_.ScriptNext({.submission = Submission::kAccepted, .result = -EINTR, .hold = false});
  const ReadSpec spec{.fd = fd_, .offset = 0, .memory = buffer_.data, .length = 4 * kAlignment};
  ASSERT_TRUE(reader_.Read(1, spec, 100).has_value());
  auto finished = PollUntilDone(reader_);
  ASSERT_EQ(finished.size(), 1U);
  EXPECT_EQ(finished[0].outcome, ReadOutcome::kComplete);

  // A hard error on one request fails the read, but only once the other
  // request in flight has completed.
  storage_.ScriptNext({.submission = Submission::kAccepted, .result = -EIO, .hold = false});
  storage_.ScriptNext({.submission = Submission::kAccepted, .result = std::nullopt, .hold = true});
  const ReadSpec two{.fd = fd_, .offset = 0, .memory = buffer_.data, .length = 8 * kAlignment};
  ASSERT_TRUE(reader_.Read(2, two, 100).has_value());
  EXPECT_THAT(reader_.Poll(false), IsEmpty());  // the held request still owns its memory
  ASSERT_TRUE(storage_.Release(storage_.submitted().back().token));
  finished = PollUntilDone(reader_);
  ASSERT_EQ(finished.size(), 1U);
  EXPECT_EQ(finished[0].outcome, ReadOutcome::kFailed);
  EXPECT_EQ(finished[0].error, EIO);
}

// Write-back's whole writes: an aligned short write continues where it
// stopped, and one that stops short at an unaligned point, or makes no
// progress, fails (never "end of file"), with the bytes it moved reported
// so the caller cannot mistake it for whole.
TEST_F(ReaderTest, AWriteContinuesAfterAnAlignedShortTransferAndOtherwiseFails) {
  const std::vector<std::byte> pattern = Pattern(4 * kAlignment);
  std::memcpy(buffer_.data, pattern.data(), pattern.size());
  storage_.ScriptNext({.submission = Submission::kAccepted,
                       .result = static_cast<std::int64_t>(kAlignment),
                       .hold = false});
  const ReadSpec spec{.fd = fd_,
                      .offset = 12 * kAlignment,  // past the file's end: the write extends it
                      .memory = buffer_.data,
                      .length = 4 * kAlignment,
                      .kind = IoKind::kWrite};
  ASSERT_TRUE(reader_.Read(1, spec, 100).has_value());
  auto finished = PollUntilDone(reader_);
  ASSERT_EQ(finished.size(), 1U);
  EXPECT_EQ(finished[0].outcome, ReadOutcome::kComplete);
  EXPECT_EQ(finished[0].bytes, 4 * kAlignment);
  ASSERT_EQ(storage_.submitted().size(), 2U);
  EXPECT_EQ(storage_.submitted()[1].kind, IoKind::kWrite);
  EXPECT_EQ(storage_.submitted()[1].offset, 13 * kAlignment);  // the remainder
  ASSERT_GE(storage_.Contents(fd_).size(), 16 * kAlignment);
  EXPECT_EQ(std::memcmp(storage_.Contents(fd_).data() + (12 * kAlignment), pattern.data(),
                        pattern.size()),
            0);

  for (const std::int64_t result : {std::int64_t{100}, std::int64_t{0}}) {
    storage_.ScriptNext({.submission = Submission::kAccepted, .result = result, .hold = false});
    ASSERT_TRUE(reader_.Read(2, spec, 100).has_value());
    finished = PollUntilDone(reader_);
    ASSERT_EQ(finished.size(), 1U);
    EXPECT_EQ(finished[0].outcome, ReadOutcome::kFailed) << result;
    EXPECT_EQ(finished[0].error, EIO);
    EXPECT_EQ(finished[0].bytes, static_cast<std::uint64_t>(result));
  }
}

TEST_F(ReaderTest, DuplicatesCoalesceAndTheLastWaiterCancels) {
  storage_.ScriptNext({.submission = Submission::kAccepted, .result = std::nullopt, .hold = true});
  const ReadSpec spec{.fd = fd_, .offset = 0, .memory = buffer_.data, .length = 4 * kAlignment};
  EXPECT_FALSE(reader_.Read(1, spec, 100).value());
  EXPECT_TRUE(reader_.Read(1, spec, 100).value());  // repeating an interest is idempotent
  EXPECT_TRUE(reader_.Read(1, spec, 101).value());  // joined: no second request
  EXPECT_EQ(Failed(reader_.Read(1, spec, 102)), ReadError::kTooManyWaiters);
  ReadSpec other = spec;
  other.offset = kAlignment;
  EXPECT_EQ(Failed(reader_.Read(1, other, 103)), ReadError::kMismatch);
  EXPECT_EQ(storage_.submitted().size(), 1U);
  ASSERT_TRUE(reader_.Withdraw(1, 100).has_value());  // one waiter leaves: the read goes on
  EXPECT_EQ(storage_.in_flight(), 1U);
  ASSERT_TRUE(reader_.Withdraw(1, 101).has_value());  // the last: cancelled, draining
  EXPECT_EQ(Failed(reader_.Read(1, spec, 104)), ReadError::kDraining);
  const auto finished = PollUntilDone(reader_);
  ASSERT_EQ(finished.size(), 1U);
  EXPECT_EQ(finished[0].outcome, ReadOutcome::kCancelled);
  EXPECT_THAT(finished[0].waiters, IsEmpty());
  EXPECT_EQ(storage_.in_flight(), 0U);
  EXPECT_EQ(Failed(reader_.Withdraw(1, 101)), ReadError::kUnknownRead);
}

TEST_F(ReaderTest, AlignmentIsCheckedAndAFullProviderWaits) {
  const ReadSpec unaligned{
      .fd = fd_, .offset = 512, .memory = buffer_.data, .length = 4 * kAlignment};
  EXPECT_EQ(Failed(reader_.Read(1, unaligned, 100)), ReadError::kUnaligned);
  const ReadSpec odd{
      .fd = fd_, .offset = 0, .memory = buffer_.data + 512, .length = 4 * kAlignment};
  EXPECT_EQ(Failed(reader_.Read(1, odd, 100)), ReadError::kUnaligned);
  // The provider refuses the first attempt; the next poll starts it.
  storage_.ScriptNext(
      {.submission = Submission::kNotStarted, .result = std::nullopt, .hold = false});
  const ReadSpec spec{.fd = fd_, .offset = 0, .memory = buffer_.data, .length = 4 * kAlignment};
  ASSERT_TRUE(reader_.Read(1, spec, 100).has_value());
  EXPECT_TRUE(storage_.submitted().empty());
  const auto finished = PollUntilDone(reader_);
  ASSERT_EQ(finished.size(), 1U);
  EXPECT_EQ(finished[0].outcome, ReadOutcome::kComplete);
}

TEST(ReaderOrderTest, ReadsStartInArrivalOrderAndContinuationsGoFirst) {
  // One request at a time, so every read after the first waits for room.
  FakeStorage storage{1, static_cast<std::uint32_t>(kAlignment)};
  DirectReader reader{storage, Settings()};
  const int fd = storage.AddFile(Pattern(16 * kAlignment));
  Buffer buffer{16 * kAlignment};
  const auto spec = [&](std::uint64_t block) {
    return ReadSpec{.fd = fd,
                    .offset = block * kAlignment,
                    .memory = buffer.data + (block * kAlignment),
                    .length = 2 * kAlignment};
  };
  // The first read's request moves only one block: its remainder must
  // start before the later reads.
  storage.ScriptNext({.submission = Submission::kAccepted,
                      .result = static_cast<std::int64_t>(kAlignment),
                      .hold = false});
  // Keys in an order unlike their arrival (the storage lane's keys are
  // mailbox indices, which are reused out of order).
  ASSERT_TRUE(reader.Read(9, spec(0), 100).has_value());
  ASSERT_TRUE(reader.Read(2, spec(2), 100).has_value());
  ASSERT_TRUE(reader.Read(7, spec(4), 100).has_value());
  ASSERT_TRUE(reader.Read(4, spec(6), 100).has_value());
  const auto finished = PollUntilDone(reader);
  ASSERT_EQ(finished.size(), 4U);
  std::vector<std::uint64_t> offsets;
  for (const auto& request : storage.submitted()) {
    offsets.push_back(request.offset / kAlignment);
  }
  EXPECT_THAT(offsets, ElementsAre(0, 1, 2, 4, 6));
  std::vector<std::uint64_t> keys;
  for (const auto& read : finished) {
    EXPECT_EQ(read.outcome, ReadOutcome::kComplete);
    keys.push_back(read.key);
  }
  EXPECT_THAT(keys, ElementsAre(9, 2, 7, 4));
}

TEST(ReaderOrderTest, AWithdrawnReadThatNeverStartedEndsWithoutHoldingUpTheOthers) {
  // One request at a time, and the first read's is held: the two later
  // reads wait, in order, and nothing of them has started.
  FakeStorage storage{1, static_cast<std::uint32_t>(kAlignment)};
  DirectReader reader{storage, Settings()};
  const int fd = storage.AddFile(Pattern(16 * kAlignment));
  Buffer buffer{16 * kAlignment};
  const auto spec = [&](std::uint64_t block) {
    return ReadSpec{.fd = fd,
                    .offset = block * kAlignment,
                    .memory = buffer.data + (block * kAlignment),
                    .length = 2 * kAlignment};
  };
  storage.ScriptNext({.submission = Submission::kAccepted, .result = std::nullopt, .hold = true});
  ASSERT_TRUE(reader.Read(1, spec(0), 100).has_value());
  ASSERT_TRUE(reader.Read(2, spec(2), 100).has_value());
  ASSERT_TRUE(reader.Read(3, spec(4), 100).has_value());
  ASSERT_EQ(storage.submitted().size(), 1U);
  // The last read, queued behind one that cannot start, is withdrawn: with
  // nothing in flight it ends at the next poll, and leaves the queue.
  ASSERT_TRUE(reader.Withdraw(3, 100).has_value());
  auto finished = reader.Poll(false);
  ASSERT_EQ(finished.size(), 1U);
  EXPECT_EQ(finished[0].key, 3U);
  EXPECT_EQ(finished[0].outcome, ReadOutcome::kCancelled);
  EXPECT_EQ(finished[0].bytes, 0U);
  EXPECT_EQ(storage.submitted().size(), 1U);
  ASSERT_TRUE(storage.Release(storage.submitted().back().token));
  finished = PollUntilDone(reader);
  std::vector<std::uint64_t> keys;
  for (const auto& read : finished) {
    EXPECT_EQ(read.outcome, ReadOutcome::kComplete);
    keys.push_back(read.key);
  }
  EXPECT_THAT(keys, ElementsAre(1, 2));
  std::vector<std::uint64_t> offsets;
  for (const auto& request : storage.submitted()) {
    offsets.push_back(request.offset / kAlignment);
  }
  EXPECT_THAT(offsets, ElementsAre(0, 2));
  EXPECT_EQ(reader.reads(), 0U);
}

// Coalescing (D-056, BP-P1): reads waiting for room that continue one
// another in a file start as one vectored request, a segment per piece
// into that piece's own memory. A provider of depth 0 holds every read
// back until the test gives it room, so the reads are all waiting.
class CoalesceTest : public ::testing::Test {
 protected:
  // Block b of the file lands at block `where(b)` of the buffer: spread
  // out of file order, as landing slots are.
  static std::uint64_t Where(std::uint64_t block) { return 63 - block; }
  ReadSpec Spec(std::uint64_t block, std::uint64_t blocks) const {
    return ReadSpec{.fd = fd_,
                    .offset = block * kAlignment,
                    .memory = buffer_.data + (Where(block + blocks - 1) * kAlignment),
                    .length = blocks * kAlignment};
  }
  // The read's memory holds its file range.
  bool Holds(const ReadSpec& spec) const {
    return std::memcmp(spec.memory, contents_.data() + spec.offset, spec.length) == 0;
  }
  // Each submitted request as (first block, segments; 0 for a plain one).
  std::vector<std::pair<std::uint64_t, std::size_t>> Requests() const {
    std::vector<std::pair<std::uint64_t, std::size_t>> requests;
    for (const IoRequest& request : storage_.submitted()) {
      requests.emplace_back(request.offset / kAlignment, request.segments.size());
    }
    return requests;
  }
  static std::map<std::uint64_t, FinishedRead> Drain(DirectReader& reader) {
    std::map<std::uint64_t, FinishedRead> by_key;
    for (FinishedRead& read : PollUntilDone(reader)) {
      by_key.emplace(read.key, std::move(read));
    }
    return by_key;
  }

  FakeStorage storage_{0, static_cast<std::uint32_t>(kAlignment)};
  std::vector<std::byte> contents_ = Pattern(64 * kAlignment);
  int fd_ = storage_.AddFile(contents_);
  Buffer buffer_{64 * kAlignment};
};

TEST_F(CoalesceTest, WaitingReadsThatContinueOneAnotherShareOneRequestInFileOrder) {
  DirectReader reader{storage_, Coalescing()};
  // Keys unlike arrival order; a gap between blocks 6 and 8 (a resident
  // chunk between two misses), and a read of another file.
  const int other = storage_.AddFile(Pattern(8 * kAlignment));
  Buffer elsewhere{2 * kAlignment};
  const ReadSpec other_file{
      .fd = other, .offset = 6 * kAlignment, .memory = elsewhere.data, .length = 2 * kAlignment};
  const std::vector<std::pair<std::uint64_t, ReadSpec>> reads = {
      {9, Spec(0, 2)},  {2, Spec(2, 2)}, {7, Spec(4, 2)}, {4, Spec(8, 2)},
      {3, Spec(10, 2)}, {8, other_file}, {6, Spec(12, 2)}};
  for (const auto& [key, spec] : reads) {
    ASSERT_TRUE(reader.Read(key, spec, 100).has_value());
  }
  EXPECT_TRUE(storage_.submitted().empty());
  storage_.SetDepth(1);  // one request at a time, so each run is visible
  const auto finished = Drain(reader);
  ASSERT_EQ(finished.size(), reads.size());
  for (const auto& [key, spec] : reads) {
    EXPECT_EQ(finished.at(key).outcome, ReadOutcome::kComplete) << key;
    EXPECT_EQ(finished.at(key).bytes, spec.length) << key;
    EXPECT_TRUE(spec.fd == other || Holds(spec)) << key;
  }
  EXPECT_EQ(std::memcmp(elsewhere.data, storage_.Contents(other).data() + (6 * kAlignment),
                        2 * kAlignment),
            0);
  // Blocks 0-6 as one, 8-12 as one (the gap breaks it), the other file's
  // read alone, then 12 alone (it does not continue the other file).
  using Run = std::pair<std::uint64_t, std::size_t>;
  EXPECT_THAT(Requests(), ElementsAre(Run{0, 3}, Run{8, 2}, Run{6, 0}, Run{12, 0}));
  // Every segment is its own read's memory.
  const IoRequest& first = storage_.submitted().front();
  ASSERT_EQ(first.segments.size(), 3U);
  EXPECT_EQ(first.length, 6 * kAlignment);
  EXPECT_EQ(first.segments[0].memory, Spec(0, 2).memory);
  EXPECT_EQ(first.segments[1].memory, Spec(2, 2).memory);
  EXPECT_EQ(first.segments[2].memory, Spec(4, 2).memory);
}

TEST_F(CoalesceTest, NothingWaitsToCoalesce) {
  // With room, each read starts as it arrives: only reads already waiting
  // for room join a span.
  storage_.SetDepth(8);
  DirectReader reader{storage_, Coalescing()};
  for (std::uint64_t block = 0; block < 8; block += 2) {
    ASSERT_TRUE(reader.Read(block, Spec(block, 2), 100).has_value());
  }
  using Run = std::pair<std::uint64_t, std::size_t>;
  EXPECT_THAT(Requests(), ElementsAre(Run{0, 0}, Run{2, 0}, Run{4, 0}, Run{6, 0}));
  EXPECT_EQ(Drain(reader).size(), 4U);
}

TEST_F(CoalesceTest, TheDefaultIsOneRequestPerPiece) {
  // Coalescing is an option, off by default (BP-P1's A/B measured it
  // slower): reads that wait and continue one another still go one by one.
  DirectReader reader{storage_, Settings()};
  for (std::uint64_t block = 0; block < 8; block += 2) {
    ASSERT_TRUE(reader.Read(block, Spec(block, 2), 100).has_value());
  }
  EXPECT_TRUE(storage_.submitted().empty());
  storage_.SetDepth(8);
  const auto finished = Drain(reader);
  ASSERT_EQ(finished.size(), 4U);
  for (const auto& [key, read] : finished) {
    EXPECT_EQ(read.outcome, ReadOutcome::kComplete) << key;
    EXPECT_TRUE(Holds(Spec(key, 2))) << key;
  }
  using Run = std::pair<std::uint64_t, std::size_t>;
  EXPECT_THAT(Requests(), ElementsAre(Run{0, 0}, Run{2, 0}, Run{4, 0}, Run{6, 0}));
}

TEST_F(CoalesceTest, ASpanOfUnknownStartIsWaitedForAndRefusedOnesStartLater) {
  DirectReader reader{storage_, Coalescing()};
  for (std::uint64_t block = 0; block < 6; block += 2) {
    ASSERT_TRUE(reader.Read(block, Spec(block, 2), 100).has_value());
  }
  // Refused once with room (no request made), then started with its
  // outcome unknown: the span is owed a completion like any other.
  storage_.ScriptNext(
      {.submission = Submission::kNotStarted, .result = std::nullopt, .hold = false});
  storage_.ScriptNext({.submission = Submission::kUnknown, .result = std::nullopt, .hold = true});
  storage_.SetDepth(4);
  // A poll starts what it can before and after harvesting: the refusal,
  // then the span.
  EXPECT_THAT(reader.Poll(false), IsEmpty());
  ASSERT_EQ(storage_.submitted().size(), 1U);
  EXPECT_EQ(storage_.submitted().front().segments.size(), 3U);
  EXPECT_EQ(reader.reads(), 3U);
  ASSERT_TRUE(storage_.Release(storage_.submitted().front().token));
  const auto finished = Drain(reader);
  ASSERT_EQ(finished.size(), 3U);
  for (const auto& [key, read] : finished) {
    EXPECT_EQ(read.outcome, ReadOutcome::kComplete) << key;
    EXPECT_TRUE(Holds(Spec(key, 2))) << key;
  }
  EXPECT_EQ(storage_.submitted().size(), 1U);
}

TEST_F(CoalesceTest, ASpanBreaksAtItsBytesItsSegmentsAndAtWrites) {
  {
    // At most five blocks a span: two reads of two blocks each. (A span
    // no larger than a piece, four blocks here, would not coalesce.)
    DirectReader reader{storage_, Coalescing(5)};
    for (std::uint64_t block = 0; block < 10; block += 2) {
      ASSERT_TRUE(reader.Read(block, Spec(block, 2), 100).has_value());
    }
    storage_.SetDepth(8);
    const auto finished = Drain(reader);
    ASSERT_EQ(finished.size(), 5U);
    using Run = std::pair<std::uint64_t, std::size_t>;
    EXPECT_THAT(Requests(), ElementsAre(Run{0, 2}, Run{4, 2}, Run{8, 0}));
  }
  storage_.SetDepth(0);
  const std::size_t before = storage_.submitted().size();
  {
    // At most three segments a span; a write between reads breaks the run
    // and is never coalesced itself.
    DirectReader reader{storage_, Coalescing(64, 3)};
    for (std::uint64_t block = 16; block < 24; ++block) {
      ReadSpec spec = Spec(block, 1);
      if (block == 20 || block == 21) {
        spec.kind = IoKind::kWrite;
      }
      ASSERT_TRUE(reader.Read(block, spec, 100).has_value());
    }
    storage_.SetDepth(8);
    const auto finished = Drain(reader);
    ASSERT_EQ(finished.size(), 8U);
    for (const auto& [key, read] : finished) {
      EXPECT_EQ(read.outcome, ReadOutcome::kComplete) << key;
    }
    std::vector<std::pair<std::uint64_t, std::size_t>> runs = Requests();
    runs.erase(runs.begin(), runs.begin() + static_cast<std::ptrdiff_t>(before));
    using Run = std::pair<std::uint64_t, std::size_t>;
    EXPECT_THAT(runs, ElementsAre(Run{16, 3}, Run{19, 0}, Run{20, 0}, Run{21, 0}, Run{22, 2}));
    EXPECT_EQ(storage_.submitted()[before + 2].kind, IoKind::kWrite);
  }
}

TEST_F(CoalesceTest, FourKibOffsetsLongReadsAndAShortLastChunkCoalesce) {
  // Groups start at 4 KiB-aligned offsets, not 2 MiB ones: a read of
  // three blocks from block 1, one of six (two pieces of four and two),
  // and a group's short last chunk of one block.
  DirectReader reader{storage_, Coalescing()};
  const std::vector<std::pair<std::uint64_t, ReadSpec>> reads = {
      {1, Spec(1, 3)}, {2, Spec(4, 6)}, {3, Spec(10, 1)}};
  for (const auto& [key, spec] : reads) {
    ASSERT_TRUE(reader.Read(key, spec, 100).has_value());
  }
  storage_.SetDepth(4);
  const auto finished = Drain(reader);
  ASSERT_EQ(finished.size(), 3U);
  for (const auto& [key, spec] : reads) {
    EXPECT_EQ(finished.at(key).outcome, ReadOutcome::kComplete) << key;
    EXPECT_TRUE(Holds(spec)) << key;
  }
  ASSERT_EQ(storage_.submitted().size(), 1U);
  const IoRequest& span = storage_.submitted().front();
  EXPECT_EQ(span.offset, kAlignment);
  EXPECT_EQ(span.length, 10 * kAlignment);
  std::vector<std::uint64_t> lengths;
  for (const auto& segment : span.segments) {
    lengths.push_back(segment.length / kAlignment);
  }
  EXPECT_THAT(lengths, ElementsAre(3, 4, 2, 1));
}

// A span's count fills its pieces in order: those it filled are done, the
// first it left short continues or ends as a lone request would, and the
// rest start again, still in file order.
TEST_F(CoalesceTest, AShortSpanFinishesWhatItFilledAndStartsTheRestAgain) {
  struct Case {
    std::int64_t result;
    // (read, outcome, bytes) for the reads at blocks 0, 2 and 4.
    std::array<ReadOutcome, 3> outcomes;
    std::array<std::uint64_t, 3> bytes;
    std::vector<std::pair<std::uint64_t, std::size_t>> requests;
  };
  const std::uint64_t whole = 2 * kAlignment;
  const std::vector<Case> cases = {
      // Mid-piece at an aligned point: the second read continues, and its
      // remainder coalesces with the third.
      {.result = static_cast<std::int64_t>(3 * kAlignment),
       .outcomes = {ReadOutcome::kComplete, ReadOutcome::kComplete, ReadOutcome::kComplete},
       .bytes = {whole, whole, whole},
       .requests = {{0, 3}, {3, 2}}},
      // At a piece's boundary: not the end of the file; both go again.
      {.result = static_cast<std::int64_t>(2 * kAlignment),
       .outcomes = {ReadOutcome::kComplete, ReadOutcome::kComplete, ReadOutcome::kComplete},
       .bytes = {whole, whole, whole},
       .requests = {{0, 3}, {2, 2}}},
      // At an unaligned point: the file ended in the second read; the third
      // starts again on its own and finds its bytes.
      {.result = static_cast<std::int64_t>((2 * kAlignment) + 100),
       .outcomes = {ReadOutcome::kComplete, ReadOutcome::kEndOfFile, ReadOutcome::kComplete},
       .bytes = {whole, 100, whole},
       .requests = {{0, 3}, {4, 0}}},
      // Nothing: the first read is at the end of the file, and the others
      // start again.
      {.result = 0,
       .outcomes = {ReadOutcome::kEndOfFile, ReadOutcome::kComplete, ReadOutcome::kComplete},
       .bytes = {0, whole, whole},
       .requests = {{0, 3}, {2, 2}}},
  };
  for (const Case& c : cases) {
    FakeStorage storage{0, static_cast<std::uint32_t>(kAlignment)};
    const int fd = storage.AddFile(contents_);
    DirectReader reader{storage, Coalescing()};
    std::array<ReadSpec, 3> specs{};
    for (std::uint64_t i = 0; i < 3; ++i) {
      specs.at(i) = Spec(2 * i, 2);
      specs.at(i).fd = fd;
      std::memset(specs.at(i).memory, 0xee, whole);
      ASSERT_TRUE(reader.Read(i, specs.at(i), 100).has_value());
    }
    storage.ScriptNext({.submission = Submission::kAccepted, .result = c.result, .hold = false});
    storage.SetDepth(4);
    std::map<std::uint64_t, FinishedRead> finished;
    for (FinishedRead& read : PollUntilDone(reader)) {
      finished.emplace(read.key, std::move(read));
    }
    ASSERT_EQ(finished.size(), 3U) << c.result;
    for (std::uint64_t i = 0; i < 3; ++i) {
      EXPECT_EQ(finished.at(i).outcome, c.outcomes.at(i)) << c.result << " read " << i;
      EXPECT_EQ(finished.at(i).bytes, c.bytes.at(i)) << c.result << " read " << i;
      if (c.outcomes.at(i) == ReadOutcome::kComplete) {
        EXPECT_TRUE(Holds(specs.at(i))) << c.result << " read " << i;
      }
    }
    std::vector<std::pair<std::uint64_t, std::size_t>> requests;
    for (const IoRequest& request : storage.submitted()) {
      requests.emplace_back(request.offset / kAlignment, request.segments.size());
    }
    EXPECT_EQ(requests, c.requests) << c.result;
  }
}

TEST_F(CoalesceTest, AnErrorInASpanLandsOnTheReadItBelongsTo) {
  DirectReader reader{storage_, Coalescing()};
  for (std::uint64_t block = 0; block < 6; block += 2) {
    ASSERT_TRUE(reader.Read(block, Spec(block, 2), 100).has_value());
  }
  // The span fails; alone, the first and last reads succeed and the
  // middle one fails again.
  storage_.ScriptNext({.submission = Submission::kAccepted, .result = -EIO, .hold = false});
  storage_.ScriptNext({.submission = Submission::kAccepted, .result = std::nullopt, .hold = false});
  storage_.ScriptNext({.submission = Submission::kAccepted, .result = -EIO, .hold = false});
  storage_.SetDepth(4);
  auto finished = Drain(reader);
  ASSERT_EQ(finished.size(), 3U);
  EXPECT_EQ(finished.at(0).outcome, ReadOutcome::kComplete);
  EXPECT_EQ(finished.at(2).outcome, ReadOutcome::kFailed);
  EXPECT_EQ(finished.at(2).error, EIO);
  EXPECT_EQ(finished.at(4).outcome, ReadOutcome::kComplete);
  EXPECT_TRUE(Holds(Spec(0, 2)));
  EXPECT_TRUE(Holds(Spec(4, 2)));
  using Run = std::pair<std::uint64_t, std::size_t>;
  EXPECT_THAT(Requests(), ElementsAre(Run{0, 3}, Run{0, 0}, Run{2, 0}, Run{4, 0}));

  // A transient error retries the span as a span.
  storage_.SetDepth(0);
  for (std::uint64_t block = 8; block < 14; block += 2) {
    ASSERT_TRUE(reader.Read(block, Spec(block, 2), 100).has_value());
  }
  storage_.ScriptNext({.submission = Submission::kAccepted, .result = -EINTR, .hold = false});
  storage_.SetDepth(4);
  finished = Drain(reader);
  ASSERT_EQ(finished.size(), 3U);
  for (const auto& [key, read] : finished) {
    EXPECT_EQ(read.outcome, ReadOutcome::kComplete) << key;
    EXPECT_TRUE(Holds(Spec(key, 2))) << key;
  }
  const auto runs = Requests();
  EXPECT_THAT(std::vector(runs.end() - 2, runs.end()), ElementsAre(Run{8, 3}, Run{8, 3}));
}

TEST_F(CoalesceTest, WithdrawingOneReadOfASpanNeverCancelsItsNeighbours) {
  DirectReader reader{storage_, Coalescing()};
  ASSERT_TRUE(reader.Read(1, Spec(0, 2), 100).has_value());
  ASSERT_TRUE(reader.Read(2, Spec(2, 2), 100).has_value());
  storage_.ScriptNext({.submission = Submission::kAccepted, .result = std::nullopt, .hold = true});
  storage_.SetDepth(1);
  EXPECT_THAT(reader.Poll(false), IsEmpty());
  ASSERT_EQ(storage_.submitted().size(), 1U);
  // The first read's last waiter leaves while the span is in flight: the
  // span is not cancelled (the fake would fail it), and the withdrawn
  // read ends only once the span has completed, since its memory is still
  // being written until then.
  ASSERT_TRUE(reader.Withdraw(1, 100).has_value());
  EXPECT_THAT(reader.Poll(false), IsEmpty());
  ASSERT_TRUE(storage_.Release(storage_.submitted().back().token));
  auto finished = Drain(reader);
  ASSERT_EQ(finished.size(), 2U);
  EXPECT_EQ(finished.at(1).outcome, ReadOutcome::kCancelled);
  EXPECT_EQ(finished.at(2).outcome, ReadOutcome::kComplete);
  EXPECT_TRUE(Holds(Spec(2, 2)));

  // Once every read in a span has left, the span is cancelled.
  storage_.SetDepth(0);
  ASSERT_TRUE(reader.Read(3, Spec(4, 2), 100).has_value());
  ASSERT_TRUE(reader.Read(4, Spec(6, 2), 100).has_value());
  storage_.ScriptNext({.submission = Submission::kAccepted, .result = std::nullopt, .hold = true});
  storage_.SetDepth(1);
  EXPECT_THAT(reader.Poll(false), IsEmpty());
  ASSERT_TRUE(reader.Withdraw(3, 100).has_value());
  ASSERT_TRUE(reader.Withdraw(4, 100).has_value());
  finished = Drain(reader);
  ASSERT_EQ(finished.size(), 2U);
  EXPECT_EQ(finished.at(3).outcome, ReadOutcome::kCancelled);
  EXPECT_EQ(finished.at(3).bytes, 0U);  // the fake cancelled it before it moved anything
  EXPECT_EQ(finished.at(4).outcome, ReadOutcome::kCancelled);
  EXPECT_EQ(storage_.in_flight(), 0U);
}

TEST(ReaderDeathTest, CoalescingSettingsAreBounded) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  FakeStorage storage(1, static_cast<std::uint32_t>(kAlignment));
  ReaderSettings too_many = Settings();
  too_many.span_segments = llmp::providers::kMaxSegments + 1;
  EXPECT_DEATH({ const DirectReader reader(storage, too_many); }, "coalesced reads need");
  ReaderSettings too_long = Settings();
  too_long.span_bytes = (1U << 30U) + 4096;
  EXPECT_DEATH({ const DirectReader reader(storage, too_long); }, "coalesced reads need");
}

TEST_F(ReaderTest, OverflowingRangesAreRefusedBeforeAnyIo) {
  const ReadSpec wrapped_file{
      .fd = fd_,
      .offset = std::numeric_limits<std::uint64_t>::max() - (kAlignment - 1),
      .memory = buffer_.data,
      .length = 2 * kAlignment};
  EXPECT_EQ(Failed(reader_.Read(1, wrapped_file, 100)), ReadError::kInvalidRange);
  const ReadSpec wrapped_memory{
      .fd = fd_,
      .offset = 0,
      .memory = reinterpret_cast<std::byte*>(  // NOLINT(performance-no-int-to-ptr)
          std::numeric_limits<std::uintptr_t>::max() - (kAlignment - 1)),
      .length = 2 * kAlignment};
  EXPECT_EQ(Failed(reader_.Read(2, wrapped_memory, 100)), ReadError::kInvalidRange);
  EXPECT_TRUE(storage_.submitted().empty());
}

TEST_F(ReaderTest, AReadStartingPastTheFileEndsWithoutTouchingMemory) {
  const ReadSpec spec{
      .fd = fd_, .offset = 100 * kAlignment, .memory = buffer_.data, .length = kAlignment};
  ASSERT_TRUE(reader_.Read(1, spec, 100).has_value());
  const auto finished = PollUntilDone(reader_);
  ASSERT_EQ(finished.size(), 1U);
  EXPECT_EQ(finished[0].outcome, ReadOutcome::kEndOfFile);
  EXPECT_EQ(finished[0].bytes, 0U);
  EXPECT_EQ(buffer_.data[0], std::byte{0xee});
}

TEST(ReaderDeathTest, AReaderCannotForgetAcceptedRequests) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  FakeStorage storage(1, static_cast<std::uint32_t>(kAlignment));
  const int fd = storage.AddFile(Pattern(kAlignment));
  Buffer buffer(kAlignment);
  storage.ScriptNext({.submission = Submission::kAccepted, .result = std::nullopt, .hold = true});
  EXPECT_DEATH(
      {
        DirectReader reader(storage, Settings());
        ASSERT_TRUE(
            reader.Read(1, {.fd = fd, .offset = 0, .memory = buffer.data, .length = kAlignment}, 1)
                .has_value());
      },
      "direct reader destroyed with requests in flight");
}

// The io_uring provider against a real file opened for direct I/O.
class UringTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto storage = UringStorage::Create(8);
    if (!storage && (storage.error() == std::errc::function_not_supported ||
                     storage.error() == std::errc::operation_not_permitted)) {
      GTEST_SKIP() << "io_uring is unavailable or denied by host policy: "
                   << storage.error().message();
    }
    ASSERT_TRUE(storage.has_value()) << storage.error().message();
    storage_ = std::move(*storage);
    const char* base = std::getenv("LLMP_TEST_SCRATCH");  // NOLINT(concurrency-mt-unsafe)
    const std::filesystem::path directory =
        base != nullptr ? std::filesystem::path(base) : std::filesystem::path(::testing::TempDir());
    std::filesystem::create_directories(directory);
    filesystem_ = llmp::platform::DescribeFilesystem(directory)
                      .value_or(llmp::platform::FilesystemFacts{})
                      .type;
    fd_ = ::open(directory.c_str(), O_TMPFILE | O_RDWR | O_DIRECT | O_CLOEXEC, 0600);
    ASSERT_GE(fd_, 0) << std::strerror(errno);  // NOLINT(concurrency-mt-unsafe)
    Buffer staging(contents_.size());
    std::memcpy(staging.data, contents_.data(), contents_.size());
    ASSERT_EQ(::pwrite(fd_, staging.data, contents_.size(), 0),
              static_cast<ssize_t>(contents_.size()));
  }
  void TearDown() override {
    if (fd_ >= 0) {
      (void)::close(fd_);
    }
  }

  std::unique_ptr<UringStorage> storage_;
  std::vector<std::byte> contents_ = Pattern(16 * kAlignment);
  int fd_ = -1;
  std::string filesystem_;
};

TEST_F(UringTest, DirectReadsLandInPlace) {
  Buffer buffer(16 * kAlignment);
  DirectReader reader(*storage_, Settings());
  const ReadSpec spec{.fd = fd_, .offset = 0, .memory = buffer.data, .length = 16 * kAlignment};
  ASSERT_TRUE(reader.Read(7, spec, 1).has_value());
  std::vector<FinishedRead> finished;
  for (int i = 0; i < 100 && finished.empty(); ++i) {
    finished = reader.Poll(true);
  }
  ASSERT_EQ(finished.size(), 1U);
  EXPECT_EQ(finished[0].outcome, ReadOutcome::kComplete);
  EXPECT_EQ(finished[0].bytes, 16 * kAlignment);
  EXPECT_EQ(std::memcmp(buffer.data, contents_.data(), contents_.size()), 0);
  // Reading past the end is a short transfer: end of file.
  const ReadSpec past{
      .fd = fd_, .offset = 12 * kAlignment, .memory = buffer.data, .length = 8 * kAlignment};
  ASSERT_TRUE(reader.Read(8, past, 1).has_value());
  finished.clear();
  for (int i = 0; i < 100 && finished.empty(); ++i) {
    finished = reader.Poll(true);
  }
  ASSERT_EQ(finished.size(), 1U);
  EXPECT_EQ(finished[0].outcome, ReadOutcome::kEndOfFile);
  EXPECT_EQ(finished[0].bytes, 4 * kAlignment);
}

// Vectored requests through the kernel (READV and WRITEV): each segment
// takes its own part of the file range, in order, wherever it is; a span
// past the file's end fills a prefix. Through the reader, adjacent pieces
// coalesce into one request and land exactly.
TEST_F(UringTest, VectoredRequestsFillEachSegmentInOrder) {
  Buffer buffer(16 * kAlignment);
  // Three segments of 2, 1 and 3 blocks, out of memory order.
  const std::array<llmp::providers::IoSegment, 3> segments = {{
      {.memory = buffer.data + (8 * kAlignment),
       .length = static_cast<std::uint32_t>(2 * kAlignment)},
      {.memory = buffer.data, .length = static_cast<std::uint32_t>(kAlignment)},
      {.memory = buffer.data + (12 * kAlignment),
       .length = static_cast<std::uint32_t>(3 * kAlignment)},
  }};
  const auto run = [&](IoKind kind, std::uint64_t offset) {
    const IoRequest request{.token = 5,
                            .kind = kind,
                            .fd = fd_,
                            .offset = offset,
                            .memory = nullptr,
                            .length = static_cast<std::uint32_t>(6 * kAlignment),
                            .segments = segments};
    EXPECT_NE(storage_->Submit(request), Submission::kNotStarted);
    std::array<IoCompletion, 4> completions{};
    std::size_t got = 0;
    for (int i = 0; i < 100 && got == 0; ++i) {
      got = storage_->Harvest(completions, true);
    }
    EXPECT_EQ(got, 1U);
    EXPECT_EQ(completions[0].token, 5U);
    return completions[0].result;
  };
  EXPECT_EQ(run(IoKind::kRead, 2 * kAlignment), static_cast<std::int64_t>(6 * kAlignment));
  EXPECT_EQ(std::memcmp(buffer.data + (8 * kAlignment), contents_.data() + (2 * kAlignment),
                        2 * kAlignment),
            0);
  EXPECT_EQ(std::memcmp(buffer.data, contents_.data() + (4 * kAlignment), kAlignment), 0);
  EXPECT_EQ(std::memcmp(buffer.data + (12 * kAlignment), contents_.data() + (5 * kAlignment),
                        3 * kAlignment),
            0);
  // From block 12 of 16: four blocks, the first two segments and one block
  // of the third.
  EXPECT_EQ(run(IoKind::kRead, 12 * kAlignment), static_cast<std::int64_t>(4 * kAlignment));
  EXPECT_EQ(std::memcmp(buffer.data + (12 * kAlignment), contents_.data() + (15 * kAlignment),
                        kAlignment),
            0);
  // Written out past the end, then read back.
  const std::vector<std::byte> before(buffer.data, buffer.data + buffer.size);
  EXPECT_EQ(run(IoKind::kWrite, 32 * kAlignment), static_cast<std::int64_t>(6 * kAlignment));
  std::memset(buffer.data, 0, buffer.size);
  EXPECT_EQ(run(IoKind::kRead, 32 * kAlignment), static_cast<std::int64_t>(6 * kAlignment));
  for (const auto& segment : segments) {
    const auto at = static_cast<std::size_t>(segment.memory - buffer.data);
    EXPECT_EQ(std::memcmp(segment.memory, before.data() + at, segment.length), 0);
  }
  EXPECT_EQ(storage_->in_flight(), 0U);
  // A length other than the segments' sum is refused before the kernel.
  EXPECT_EQ(storage_->Submit(IoRequest{.token = 6,
                                       .kind = IoKind::kRead,
                                       .fd = fd_,
                                       .offset = 0,
                                       .memory = nullptr,
                                       .length = static_cast<std::uint32_t>(4 * kAlignment),
                                       .segments = segments}),
            Submission::kNotStarted);
  EXPECT_EQ(storage_->in_flight(), 0U);

  // The reader: a read of 16 blocks is four pieces of four, one request.
  ReaderSettings settings = Coalescing();
  DirectReader reader(*storage_, settings);
  std::memset(buffer.data, 0, buffer.size);
  ASSERT_TRUE(
      reader.Read(7, {.fd = fd_, .offset = 0, .memory = buffer.data, .length = 16 * kAlignment}, 1)
          .has_value());
  std::vector<FinishedRead> finished;
  for (int i = 0; i < 100 && finished.empty(); ++i) {
    finished = reader.Poll(true);
  }
  ASSERT_EQ(finished.size(), 1U);
  EXPECT_EQ(finished[0].outcome, ReadOutcome::kComplete);
  EXPECT_EQ(finished[0].bytes, 16 * kAlignment);
  EXPECT_EQ(std::memcmp(buffer.data, contents_.data(), contents_.size()), 0);
}

TEST_F(UringTest, UnalignedDirectIoFailsAndCancellationStillCompletes) {
  Buffer buffer(8 * kAlignment);
  // The kernel refuses a misaligned direct-I/O offset; the request still
  // completes.
  ASSERT_EQ(storage_->Submit(IoRequest{.token = 1,
                                       .kind = IoKind::kRead,
                                       .fd = fd_,
                                       .offset = 1,
                                       .memory = buffer.data,
                                       .length = static_cast<std::uint32_t>(kAlignment),
                                       .segments = {}}),
            Submission::kAccepted);
  // A cancellation of an unknown token is refused; of a live one, accepted.
  EXPECT_EQ(storage_->Cancel(99), Submission::kNotStarted);
  ASSERT_NE(storage_->Submit(IoRequest{.token = 2,
                                       .kind = IoKind::kRead,
                                       .fd = fd_,
                                       .offset = 0,
                                       .memory = buffer.data,
                                       .length = 4 * kAlignment,
                                       .segments = {}}),
            Submission::kNotStarted);
  (void)storage_->Cancel(2);
  std::array<IoCompletion, 8> completions{};
  std::size_t seen = 0;
  std::int64_t unaligned = 0;
  for (int i = 0; i < 100 && storage_->in_flight() > 0; ++i) {
    const std::size_t count = storage_->Harvest(std::span(completions).subspan(seen), true);
    for (std::size_t c = seen; c < seen + count; ++c) {
      if (completions[c].token == 1) {
        unaligned = completions[c].result;
      }
    }
    seen += count;
  }
  EXPECT_EQ(storage_->in_flight(), 0U);
  EXPECT_EQ(seen, 2U);  // both originals, never the cancellation's own result
  // Btrfs serves misaligned direct I/O through the page cache instead of
  // refusing it (RE-018); ext4 and XFS refuse it.
  if (filesystem_ != "btrfs") {
    EXPECT_EQ(unaligned, -EINVAL);
  }
}

// Cancellations count as in flight until their own completions arrive, so
// an owner that drains to zero, even one slot at a time, can then destroy
// the provider.
TEST_F(UringTest, DrainingIncludesCancellations) {
  Buffer buffer(4 * kAlignment);
  ASSERT_NE(storage_->Submit(IoRequest{.token = 5,
                                       .kind = IoKind::kRead,
                                       .fd = fd_,
                                       .offset = 0,
                                       .memory = buffer.data,
                                       .length = static_cast<std::uint32_t>(4 * kAlignment),
                                       .segments = {}}),
            Submission::kNotStarted);
  if (storage_->Cancel(5) != Submission::kNotStarted) {
    EXPECT_EQ(storage_->in_flight(), 2U);
  }
  std::array<IoCompletion, 1> one{};
  std::size_t originals = 0;
  for (int i = 0; i < 100 && storage_->in_flight() > 0; ++i) {
    originals += storage_->Harvest(one, true);
  }
  EXPECT_EQ(originals, 1U);
  EXPECT_EQ(storage_->in_flight(), 0U);
  storage_.reset();  // drained: no abort
}

// A read of an empty pipe never completes on its own. Wake, from another
// thread, ends the Harvest waiting for it, so the lane can take a new
// command, such as the cancellation that then drains it.
TEST_F(UringTest, WakeEndsAHarvestWaitingForAReadThatNeverCompletes) {
  std::array<int, 2> pipe_fds{-1, -1};
  ASSERT_EQ(::pipe2(pipe_fds.data(), O_CLOEXEC), 0);
  Buffer buffer(kAlignment);
  ASSERT_NE(storage_->Submit(IoRequest{.token = 9,
                                       .kind = IoKind::kRead,
                                       .fd = pipe_fds[0],
                                       .offset = 0,
                                       .memory = buffer.data,
                                       .length = static_cast<std::uint32_t>(kAlignment),
                                       .segments = {}}),
            Submission::kNotStarted);
  std::atomic<bool> returned{false};
  std::size_t harvested = 0;
  std::array<IoCompletion, 4> completions{};
  {
    std::jthread waiter([&] {
      harvested = storage_->Harvest(completions, true);
      returned.store(true);
    });
    // Most likely waiting by now; the wake must end the wait either way.
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    storage_->Wake();
    storage_->Wake();  // coalesced
    const auto give_up = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (!returned.load() && std::chrono::steady_clock::now() < give_up) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (!returned.load()) {
      ADD_FAILURE() << "Wake did not end the Harvest";
      const std::vector<std::byte> fill(kAlignment);
      ASSERT_EQ(::write(pipe_fds[1], fill.data(), fill.size()), static_cast<ssize_t>(kAlignment));
    }
  }
  EXPECT_EQ(harvested, 0U);
  ASSERT_EQ(storage_->in_flight(), 1U);
  // The cancellation is what drains it.
  ASSERT_NE(storage_->Cancel(9), Submission::kNotStarted);
  std::int64_t result = 0;
  for (int i = 0; i < 100 && storage_->in_flight() > 0; ++i) {
    const std::size_t count = storage_->Harvest(completions, true);
    for (std::size_t c = 0; c < count; ++c) {
      if (completions.at(c).token == 9) {
        result = completions.at(c).result;
      }
    }
  }
  EXPECT_EQ(storage_->in_flight(), 0U);
  EXPECT_EQ(result, -ECANCELED);
  (void)::close(pipe_fds[0]);
  (void)::close(pipe_fds[1]);
}

// The lane's side of the wake protocol under many producers: each publishes
// a command (here a counter), then wakes; the lane waits in Harvest for a
// read that never completes, and after every return looks for commands
// before it waits again. A wake lost among the coalesced ones leaves the
// lane asleep with a command published, which shows as a producer that is
// never answered (bounded: the test then ends the read itself).
TEST_F(UringTest, NoWakeIsLostAmongManyProducers) {
#ifdef LLMP_TEST_TSAN
  constexpr std::uint64_t kCommands = 2000;  // per producer
#else
  constexpr std::uint64_t kCommands = 20000;
#endif
  constexpr std::uint64_t kProducers = 4;
  std::array<int, 2> pipe_fds{-1, -1};
  ASSERT_EQ(::pipe2(pipe_fds.data(), O_CLOEXEC), 0);
  Buffer buffer(kAlignment);
  ASSERT_NE(storage_->Submit(IoRequest{.token = 5,
                                       .kind = IoKind::kRead,
                                       .fd = pipe_fds[0],
                                       .offset = 0,
                                       .memory = buffer.data,
                                       .length = static_cast<std::uint32_t>(kAlignment),
                                       .segments = {}}),
            Submission::kNotStarted);
  std::atomic<std::uint64_t> published{0};
  std::atomic<std::uint64_t> answered{0};
  std::atomic<bool> stop{false};
  std::atomic<bool> lost{false};
  std::size_t completions_seen = 0;
  {
    std::jthread lane([&] {
      std::array<IoCompletion, 4> completions{};
      while (!stop.load()) {
        const std::uint64_t now = published.load();
        if (now > answered.load()) {
          answered.store(now);  // "took the commands"
          continue;
        }
        completions_seen += storage_->Harvest(completions, true);
      }
    });
    {
      std::vector<std::jthread> producers;
      producers.reserve(kProducers);
      for (std::uint64_t p = 0; p < kProducers; ++p) {
        producers.emplace_back([&] {
          for (std::uint64_t i = 0; i < kCommands && !lost.load(); ++i) {
            if (i % 16 == 0) {
              // Now and then, long enough for the lane to wait in the kernel.
              std::this_thread::sleep_for(std::chrono::microseconds(20));
            }
            const std::uint64_t mine = published.fetch_add(1) + 1;
            storage_->Wake();
            const auto give_up = std::chrono::steady_clock::now() + std::chrono::seconds(30);
            while (answered.load() < mine) {
              if (std::chrono::steady_clock::now() > give_up) {
                lost.store(true);
                return;
              }
              std::this_thread::yield();
            }
          }
        });
      }
    }
    stop.store(true);
    storage_->Wake();
    if (lost.load()) {
      // Release the lane so the test can end.
      const std::vector<std::byte> fill(kAlignment);
      EXPECT_EQ(::write(pipe_fds[1], fill.data(), fill.size()), static_cast<ssize_t>(kAlignment));
    }
  }
  EXPECT_FALSE(lost.load()) << "a wake was lost: the lane slept with a command published";
  if (!lost.load()) {
    EXPECT_EQ(answered.load(), kCommands * kProducers);
  }
  // Drain the read: cancelled unless the test had to end it.
  std::array<IoCompletion, 4> completions{};
  if (storage_->in_flight() > 0) {
    ASSERT_NE(storage_->Cancel(5), Submission::kNotStarted);
  }
  for (int i = 0; i < 100 && storage_->in_flight() > 0; ++i) {
    completions_seen += storage_->Harvest(completions, true);
  }
  EXPECT_EQ(storage_->in_flight(), 0U);
  EXPECT_EQ(completions_seen, 1U);
  (void)::close(pipe_fds[0]);
  (void)::close(pipe_fds[1]);
}

}  // namespace
