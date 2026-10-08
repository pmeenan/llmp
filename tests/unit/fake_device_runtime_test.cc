// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The fake device runtime (providers/fake/fake_device_runtime.h) and the
// host paging kernels (kernels/paging/host_paging.cc), the engine core's
// device in every build profile: the device opens with the fakes; fills,
// copies, kernels, timing marks and replays queued on a job's stream run
// only once the stream runs (a fence query), in order, reading their
// operands then; recorded work replays what it recorded with the data its
// inputs hold at replay; a handle no fake stream has is refused and leaves
// the thread's error state set until taken; and the paging launches keep
// CUDA's refusals.

#include "providers/fake/fake_device_runtime.h"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <utility>

#include "kernels/paging/paging.h"
#include "providers/device_execution.h"
#include "providers/device_runtime.h"

namespace {

namespace pr = llmp::providers;
namespace paging = llmp::kernels::paging;

constexpr std::size_t kBytes = 4096;

class FakeDeviceRuntimeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto device = pr::OpenDevice(0, {.fences_ahead = 4, .fences_kept = 16});
    ASSERT_TRUE(device.has_value()) << device.error().detail;
    device_ = std::move(*device);
    auto stream = device_.execution->CreateStream();
    ASSERT_TRUE(stream.has_value());
    stream_ = *stream;
    auto on_device = pr::AllocatePinned(kBytes);  // host memory either way
    auto pinned = pr::AllocatePinned(kBytes);
    ASSERT_TRUE(on_device.has_value() && pinned.has_value());
    on_device_ = static_cast<std::byte*>(*on_device);
    host_ = static_cast<std::byte*>(*pinned);
    std::memset(on_device_, 0, kBytes);
    std::memset(host_, 0, kBytes);
  }

  void TearDown() override {
    if (device_.execution != nullptr && stream_.valid()) {
      Finish();
      EXPECT_TRUE(device_.execution->DestroyStream(stream_).has_value());
    }
    pr::FreePinned(host_);
    pr::FreePinned(on_device_);
  }

  pr::NativeStream Native() {
    auto native = device_.execution->Submission(stream_);
    EXPECT_TRUE(native.has_value());
    return native.value_or(pr::NativeStream{});
  }

  // Waits for everything queued so far: the fake runs it at the query.
  void Finish() {
    auto fence = device_.execution->Record(stream_);
    ASSERT_TRUE(fence.has_value());
    auto state = device_.execution->Query(*fence);
    ASSERT_TRUE(state.has_value());
    ASSERT_EQ(*state, pr::FenceState::kComplete);
    ASSERT_TRUE(device_.execution->Release(*fence).has_value());
  }

  bool Holds(const std::byte* at, std::uint8_t value, std::size_t bytes = kBytes) const {
    for (std::size_t i = 0; i < bytes; ++i) {
      if (at[i] != std::byte{value}) {
        return false;
      }
    }
    return true;
  }

  pr::Device device_;
  pr::StreamId stream_;
  std::byte* on_device_ = nullptr;
  std::byte* host_ = nullptr;
};

TEST_F(FakeDeviceRuntimeTest, ReportsTheFakeDevice) {
  auto facts = pr::QueryDeviceFacts(0);
  ASSERT_TRUE(facts.has_value());
  EXPECT_EQ(facts->architecture, 0U);
  auto memory = pr::QueryDeviceMemory();
  ASSERT_TRUE(memory.has_value());
  EXPECT_GT(memory->total, 0U);
  EXPECT_GE(memory->total, memory->free);
  EXPECT_EQ(device_.memory->Granularity().value(), std::uint64_t{2} << 20U);
  EXPECT_FALSE(pr::OpenDevice(1).has_value());
  EXPECT_FALSE(pr::QueryDeviceFacts(1).has_value());
  EXPECT_FALSE(pr::TakeLastError().ok());  // the refused query's
  EXPECT_TRUE(pr::PeekLastError().ok());
  EXPECT_NE(pr::DeviceStatus(1).text(), nullptr);
}

TEST_F(FakeDeviceRuntimeTest, QueuedWorkRunsOnlyWhenTheStreamRunsInOrder) {
  const pr::NativeStream native = Native();
  ASSERT_TRUE(pr::FillAsync(native, on_device_, 0xAB, kBytes).ok());
  ASSERT_TRUE(pr::CopyAsync(native, host_, on_device_, kBytes, pr::CopyKind::kDeviceToHost).ok());
  // Returning queued the work; nothing ran.
  EXPECT_TRUE(Holds(on_device_, 0));
  EXPECT_TRUE(Holds(host_, 0));
  Finish();
  EXPECT_TRUE(Holds(on_device_, 0xAB));
  EXPECT_TRUE(Holds(host_, 0xAB));
  // A copy reads its source when it runs, not when it was queued.
  std::memset(host_, 0x11, kBytes);
  ASSERT_TRUE(pr::CopyAsync(native, on_device_, host_, kBytes, pr::CopyKind::kHostToDevice).ok());
  std::memset(host_, 0x3C, kBytes);
  Finish();
  EXPECT_TRUE(Holds(on_device_, 0x3C));
}

TEST_F(FakeDeviceRuntimeTest, AHandleNoStreamHasIsRefusedAndLeavesTheErrorUntilTaken) {
  const pr::NativeStream stranger{.handle = reinterpret_cast<void*>(std::uintptr_t{8})};
  EXPECT_FALSE(pr::FillAsync(stranger, on_device_, 1, kBytes).ok());
  EXPECT_FALSE(pr::PeekLastError().ok());
  EXPECT_FALSE(pr::TakeLastError().ok());
  EXPECT_TRUE(pr::TakeLastError().ok());
  EXPECT_FALSE(
      paging::ClampTokens(reinterpret_cast<std::int32_t*>(on_device_), 1, 8, stranger.handle));
  EXPECT_FALSE(pr::CopyAsync(Native(), nullptr, host_, 1, pr::CopyKind::kHostToDevice).ok());
  EXPECT_TRUE(Holds(on_device_, 0));
  (void)pr::TakeLastError();
}

TEST_F(FakeDeviceRuntimeTest, TimingMarksAreReadyOnceTheirStreamRanThem) {
  auto begin = pr::CreateTimingMark();
  auto end = pr::CreateTimingMark();
  ASSERT_TRUE(begin.has_value() && end.has_value());
  const pr::NativeStream native = Native();
  ASSERT_TRUE(pr::RecordTimingMark(*begin, native).ok());
  ASSERT_TRUE(pr::FillAsync(native, on_device_, 7, kBytes).ok());
  ASSERT_TRUE(pr::RecordTimingMark(*end, native).ok());
  EXPECT_FALSE(pr::ElapsedMilliseconds(*begin, *end).has_value());
  Finish();
  auto ms = pr::ElapsedMilliseconds(*begin, *end);
  ASSERT_TRUE(ms.has_value());
  EXPECT_GE(*ms, 0.0F);
  pr::DestroyTimingMark(*begin);
  pr::DestroyTimingMark(*end);
  (void)pr::TakeLastError();
}

TEST_F(FakeDeviceRuntimeTest, RecordedWorkReplaysWithWhatItsInputsHoldAtReplay) {
  const pr::NativeStream native = Native();
  ASSERT_TRUE(pr::BeginRecording(native).ok());
  EXPECT_FALSE(pr::BeginRecording(native).ok());  // already recording
  (void)pr::TakeLastError();
  ASSERT_TRUE(pr::CopyAsync(native, on_device_, host_, kBytes, pr::CopyKind::kHostToDevice).ok());
  // Fences cannot be recorded into a recording.
  EXPECT_FALSE(device_.execution->Record(stream_).has_value());
  auto work = pr::EndRecording(native);
  ASSERT_TRUE(work.has_value());
  ASSERT_TRUE(work->valid());
  EXPECT_FALSE(pr::EndRecording(native).has_value());
  (void)pr::TakeLastError();
  Finish();
  EXPECT_TRUE(Holds(on_device_, 0));  // recorded, not run
  for (const std::uint8_t value : std::array<std::uint8_t, 2>{0x21, 0x42}) {
    std::memset(host_, value, kBytes);
    ASSERT_TRUE(pr::Replay(*work, native).ok());
    Finish();
    EXPECT_TRUE(Holds(on_device_, value));
  }
}

TEST_F(FakeDeviceRuntimeTest, PagingKernelsReadTheirOperandsWhenTheyRun) {
  const pr::NativeStream native = Native();
  // Fill: two ranges, the descriptors written after the launch.
  auto* ranges = reinterpret_cast<std::uint64_t*>(host_);
  ASSERT_TRUE(paging::FillRanges(ranges, 2, 0x5A, native.handle));
  ranges[0] = reinterpret_cast<std::uint64_t>(on_device_);
  ranges[1] = 64;
  ranges[2] = reinterpret_cast<std::uint64_t>(on_device_ + 128);
  ranges[3] = 64;
  Finish();
  EXPECT_TRUE(Holds(on_device_, 0x5A, 64));
  EXPECT_TRUE(Holds(on_device_ + 64, 0, 64));
  EXPECT_TRUE(Holds(on_device_ + 128, 0x5A, 64));

  // Range copies: 16-byte multiples, both ends aligned.
  std::memset(on_device_, 0, kBytes);
  std::memset(on_device_ + 1024, 0x77, 256);
  auto* copies = reinterpret_cast<paging::RangeCopy*>(host_ + 2048);
  copies[0] = {.from = reinterpret_cast<std::uint64_t>(on_device_ + 1024),
               .to = reinterpret_cast<std::uint64_t>(on_device_),
               .bytes = 256};
  ASSERT_TRUE(paging::CopyRanges(copies, 1, native.handle));
  Finish();
  EXPECT_TRUE(Holds(on_device_, 0x77, 256));
  copies[0].bytes = 8;  // not a multiple of 16
  EXPECT_FALSE(paging::CopyRanges(copies, 1, native.handle));
  EXPECT_FALSE(paging::CopyRanges(nullptr, paging::kMaxRangeCopies + 1, native.handle));
  EXPECT_TRUE(paging::CopyRanges(nullptr, 0, native.handle));

  // The n-gram rows' gather: only the first *count of max_count slots.
  std::memset(on_device_, 0, kBytes);
  std::byte* const landing = host_ + 3072;
  for (std::size_t i = 0; i < 64; ++i) landing[i] = static_cast<std::byte>(i);
  auto* sources = reinterpret_cast<std::uint32_t*>(host_ + 3584);
  auto* count = reinterpret_cast<std::uint32_t*>(host_ + 3840);
  ASSERT_TRUE(paging::GatherPleRows(landing, sources, count, 3, 8, on_device_, native.handle));
  sources[0] = 16;
  sources[1] = 0;
  sources[2] = 40;
  *count = 2;
  Finish();
  EXPECT_EQ(on_device_[0], std::byte{16});
  EXPECT_EQ(on_device_[7], std::byte{23});
  EXPECT_EQ(on_device_[8], std::byte{0});
  EXPECT_EQ(on_device_[15], std::byte{7});
  EXPECT_TRUE(Holds(on_device_ + 16, 0, 8));  // slot 2 beyond *count

  // Draft ids outside [0, limit) become 0.
  constexpr std::int32_t kLimit = 129280;
  const std::array<std::int32_t, 7> ids = {
      -1, 0, 5, kLimit - 1, kLimit, kLimit + 3, std::numeric_limits<std::int32_t>::min()};
  std::memcpy(on_device_, ids.data(), sizeof(ids));
  ASSERT_TRUE(paging::ClampTokens(reinterpret_cast<std::int32_t*>(on_device_),
                                  static_cast<std::uint32_t>(ids.size()), kLimit, native.handle));
  Finish();
  std::array<std::int32_t, 7> got{};
  std::memcpy(got.data(), on_device_, sizeof(got));
  EXPECT_EQ(got, (std::array<std::int32_t, 7>{0, 0, 5, kLimit - 1, 0, 0, 0}));

  // CUDA's grid limit refuses these before anything runs.
  constexpr std::uint32_t kTooMany = std::numeric_limits<std::uint32_t>::max();
  EXPECT_FALSE(paging::FillRanges(nullptr, kTooMany, 0, native.handle));
  EXPECT_FALSE(
      paging::GatherPleRows(nullptr, nullptr, nullptr, kTooMany, 1, nullptr, native.handle));
  EXPECT_TRUE(paging::ClampTokens(nullptr, 0, kLimit, native.handle));
}

TEST_F(FakeDeviceRuntimeTest, AStreamWithUnfencedWorkIsNotDestroyed) {
  ASSERT_TRUE(pr::FillAsync(Native(), on_device_, 3, kBytes).ok());
  EXPECT_FALSE(device_.execution->DestroyStream(stream_).has_value());
  Finish();
  EXPECT_TRUE(Holds(on_device_, 3));
}

}  // namespace
