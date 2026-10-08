// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The device runtime (providers/device_runtime.h) over CUDA on a GPU: the
// device opens with its providers and reports its facts and free memory;
// fills and copies on a job's stream land in pinned host memory; timing
// marks measure the span between them; recorded work replays what it
// recorded, with the data its pinned inputs hold at replay; and the error
// state reads clear. The engine's jobs use nothing else of the device,
// besides its own paging kernels (kernels/paging/paging.h), of which the
// draft-id guard and the launchers' failure reporting are checked here.

#include "providers/device_runtime.h"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <utility>

#include "kernels/paging/paging.h"
#include "providers/device_execution.h"

namespace {

namespace pr = llmp::providers;

constexpr std::size_t kBytes = 4096;

class DeviceRuntimeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto device = pr::OpenDevice(0, {.fences_ahead = 4, .fences_kept = 16});
    ASSERT_TRUE(device.has_value()) << device.error().detail;
    device_ = std::move(*device);
    auto stream = device_.execution->CreateStream();
    ASSERT_TRUE(stream.has_value());
    stream_ = *stream;
    ASSERT_EQ(cudaMalloc(&on_device_, kBytes), cudaSuccess);
    auto pinned = pr::AllocatePinned(kBytes);
    ASSERT_TRUE(pinned.has_value()) << pinned.error().text();
    host_ = static_cast<std::byte*>(*pinned);
  }

  void TearDown() override {
    if (device_.execution != nullptr && stream_.valid()) {
      Finish();
      EXPECT_TRUE(device_.execution->DestroyStream(stream_).has_value());
    }
    pr::FreePinned(host_);
    if (on_device_ != nullptr) {
      (void)cudaFree(on_device_);
    }
  }

  pr::NativeStream Native() {
    auto native = device_.execution->Submission(stream_);
    EXPECT_TRUE(native.has_value());
    return native.value_or(pr::NativeStream{});
  }

  // Waits for everything queued so far.
  void Finish() {
    auto fence = device_.execution->Record(stream_);
    ASSERT_TRUE(fence.has_value());
    for (;;) {
      auto state = device_.execution->Query(*fence);
      ASSERT_TRUE(state.has_value());
      if (*state == pr::FenceState::kComplete) {
        break;
      }
    }
    ASSERT_TRUE(device_.execution->Release(*fence).has_value());
  }

  bool HostHolds(std::uint8_t value) const {
    for (std::size_t i = 0; i < kBytes; ++i) {
      if (host_[i] != std::byte{value}) {
        return false;
      }
    }
    return true;
  }

  pr::Device device_;
  pr::StreamId stream_;
  void* on_device_ = nullptr;
  std::byte* host_ = nullptr;
};

TEST_F(DeviceRuntimeTest, ReportsTheDevice) {
  auto facts = pr::QueryDeviceFacts(0);
  ASSERT_TRUE(facts.has_value()) << facts.error().text();
  int major = 0;
  int minor = 0;
  ASSERT_EQ(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0), cudaSuccess);
  ASSERT_EQ(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, 0), cudaSuccess);
  EXPECT_EQ(facts->architecture, static_cast<std::uint32_t>((100 * major) + (10 * minor)));
  auto memory = pr::QueryDeviceMemory();
  ASSERT_TRUE(memory.has_value()) << memory.error().text();
  EXPECT_GT(memory->free, 0U);
  EXPECT_GE(memory->total, memory->free);
  EXPECT_TRUE(pr::TakeLastError().ok());
  EXPECT_TRUE(pr::PeekLastError().ok());
  EXPECT_NE(pr::DeviceStatus(1).text(), nullptr);
  EXPECT_FALSE(pr::DeviceStatus(1).ok());
}

TEST_F(DeviceRuntimeTest, FillsAndCopiesOnAJobsStream) {
  const pr::NativeStream native = Native();
  ASSERT_TRUE(pr::FillAsync(native, on_device_, 0xAB, kBytes).ok());
  ASSERT_TRUE(pr::CopyAsync(native, host_, on_device_, kBytes, pr::CopyKind::kDeviceToHost).ok());
  Finish();
  EXPECT_TRUE(HostHolds(0xAB));
  std::memset(host_, 0x3C, kBytes);
  ASSERT_TRUE(pr::CopyAsync(native, on_device_, host_, kBytes, pr::CopyKind::kHostToDevice).ok());
  Finish();
  std::memset(host_, 0, kBytes);
  ASSERT_TRUE(pr::CopyAsync(native, host_, on_device_, kBytes, pr::CopyKind::kDeviceToHost).ok());
  Finish();
  EXPECT_TRUE(HostHolds(0x3C));
}

// The engine's draft-id guard (kernels/paging/paging.h ClampTokens): ids
// outside [0, limit) become 0, the rest are kept.
TEST_F(DeviceRuntimeTest, DraftIdsAreBoundedToTheVocabulary) {
  const pr::NativeStream native = Native();
  constexpr std::int32_t kLimit = 129280;
  const std::array<std::int32_t, 7> ids = {
      -1, 0, 5, kLimit - 1, kLimit, kLimit + 3, std::numeric_limits<std::int32_t>::min()};
  const std::size_t bytes = sizeof(ids);
  std::memcpy(host_, ids.data(), bytes);
  ASSERT_TRUE(pr::CopyAsync(native, on_device_, host_, bytes, pr::CopyKind::kHostToDevice).ok());
  ASSERT_TRUE(llmp::kernels::paging::ClampTokens(static_cast<std::int32_t*>(on_device_),
                                                 static_cast<std::uint32_t>(ids.size()), kLimit,
                                                 native.handle));
  std::memset(host_, 0xFF, bytes);
  ASSERT_TRUE(pr::CopyAsync(native, host_, on_device_, bytes, pr::CopyKind::kDeviceToHost).ok());
  Finish();
  std::array<std::int32_t, 7> got{};
  std::memcpy(got.data(), host_, bytes);
  EXPECT_EQ(got, (std::array<std::int32_t, 7>{0, 0, 5, kLimit - 1, 0, 0, 0}));
  EXPECT_TRUE(llmp::kernels::paging::ClampTokens(nullptr, 0, kLimit, native.handle));
}

// A paging launch that fails (a grid past CUDA's x limit, refused before
// anything runs, so no operand is read) reports it and leaves no error
// behind for a later check to take as its own; the next launch succeeds.
TEST_F(DeviceRuntimeTest, AFailedPagingLaunchClearsItsError) {
  namespace paging = llmp::kernels::paging;
  const pr::NativeStream native = Native();
  constexpr std::uint32_t kTooMany = std::numeric_limits<std::uint32_t>::max();
  EXPECT_FALSE(paging::FillRanges(nullptr, kTooMany, 0, native.handle));
  EXPECT_TRUE(pr::PeekLastError().ok());
  EXPECT_FALSE(
      paging::GatherPleRows(nullptr, nullptr, nullptr, kTooMany, 1, nullptr, native.handle));
  EXPECT_TRUE(pr::PeekLastError().ok());
  const std::int32_t id = -1;
  std::memcpy(host_, &id, sizeof(id));
  ASSERT_TRUE(
      pr::CopyAsync(native, on_device_, host_, sizeof(id), pr::CopyKind::kHostToDevice).ok());
  EXPECT_TRUE(paging::ClampTokens(static_cast<std::int32_t*>(on_device_), 1, 8, native.handle));
  ASSERT_TRUE(
      pr::CopyAsync(native, host_, on_device_, sizeof(id), pr::CopyKind::kDeviceToHost).ok());
  Finish();
  std::int32_t got = -1;
  std::memcpy(&got, host_, sizeof(got));
  EXPECT_EQ(got, 0);
}

TEST_F(DeviceRuntimeTest, TimingMarksMeasureTheirSpan) {
  auto begin = pr::CreateTimingMark();
  auto end = pr::CreateTimingMark();
  ASSERT_TRUE(begin.has_value() && end.has_value());
  const pr::NativeStream native = Native();
  ASSERT_TRUE(pr::RecordTimingMark(*begin, native).ok());
  ASSERT_TRUE(pr::FillAsync(native, on_device_, 1, kBytes).ok());
  ASSERT_TRUE(pr::RecordTimingMark(*end, native).ok());
  Finish();
  auto ms = pr::ElapsedMilliseconds(*begin, *end);
  ASSERT_TRUE(ms.has_value()) << ms.error().text();
  EXPECT_GE(*ms, 0.0F);
  pr::DestroyTimingMark(*begin);
  pr::DestroyTimingMark(*end);
}

TEST_F(DeviceRuntimeTest, RecordedWorkReplaysWithItsInputsAtReplay) {
  const pr::NativeStream native = Native();
  std::memset(host_, 0x11, kBytes / 2);
  std::memset(host_ + (kBytes / 2), 0, kBytes / 2);
  // Recorded: the host's first half up to the device, then the device back
  // down into the second half.
  ASSERT_TRUE(pr::BeginRecording(native).ok());
  ASSERT_TRUE(
      pr::CopyAsync(native, on_device_, host_, kBytes / 2, pr::CopyKind::kHostToDevice).ok());
  ASSERT_TRUE(pr::CopyAsync(native, host_ + (kBytes / 2), on_device_, kBytes / 2,
                            pr::CopyKind::kDeviceToHost)
                  .ok());
  auto recorded = pr::EndRecording(native);
  ASSERT_TRUE(recorded.has_value()) << recorded.error().text();
  ASSERT_TRUE(recorded->valid());
  Finish();
  // Recording ran nothing.
  EXPECT_EQ(host_[kBytes / 2], std::byte{0});
  for (const std::uint8_t value : {std::uint8_t{0x22}, std::uint8_t{0x33}}) {
    std::memset(host_, value, kBytes / 2);
    ASSERT_TRUE(pr::Replay(*recorded, native).ok());
    Finish();
    EXPECT_TRUE(HostHolds(value));
  }
  pr::RecordedWork moved = std::move(*recorded);
  EXPECT_TRUE(moved.valid());
  EXPECT_FALSE(recorded->valid());  // NOLINT(bugprone-use-after-move): moved from, by design
  moved.Reset();
  EXPECT_FALSE(moved.valid());
}

// A recording the backend invalidates (here a synchronization on the
// recording stream, which a capture refuses) ends with an error and no
// work, and the stream runs as before afterwards: the failure is not
// sticky, and the thread's error state clears once read.
TEST_F(DeviceRuntimeTest, AFailedRecordingLeavesTheStreamUsable) {
  const pr::NativeStream native = Native();
  ASSERT_TRUE(pr::BeginRecording(native).ok());
  ASSERT_TRUE(pr::FillAsync(native, on_device_, 0x5A, kBytes).ok());
  EXPECT_NE(cudaStreamSynchronize(static_cast<cudaStream_t>(native.handle)), cudaSuccess);
  auto recorded = pr::EndRecording(native);
  ASSERT_FALSE(recorded.has_value());
  (void)pr::TakeLastError();
  EXPECT_TRUE(pr::PeekLastError().ok());
  ASSERT_TRUE(pr::FillAsync(native, on_device_, 0x6B, kBytes).ok());
  ASSERT_TRUE(pr::CopyAsync(native, host_, on_device_, kBytes, pr::CopyKind::kDeviceToHost).ok());
  Finish();
  EXPECT_TRUE(HostHolds(0x6B));
}

}  // namespace
