// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// The deterministic device-execution fake: each stream is a queue of
// copies, cross-stream waits and fence markers that runs only when a test
// steps it, so completion order is the test's to choose. Copies move bytes
// between the fake device memory's addresses (which are host addresses)
// when they run, so a consumer that reads before its fence completes sees
// the old bytes. A stream's native handle is a distinct token that nothing
// dereferences. Releasing a fence before it was seen complete, or
// destroying a stream with work not yet behind a released fence, is
// refused, as the CUDA provider refuses them. Every call takes one lock, so
// a completion lane may query while the submission lane submits.

#ifndef JITLLM_PROVIDERS_FAKE_FAKE_DEVICE_EXECUTION_H_
#define JITLLM_PROVIDERS_FAKE_FAKE_DEVICE_EXECUTION_H_

#include <cstddef>
#include <cstdint>
#include <deque>
#include <expected>
#include <mutex>
#include <optional>

#include "base/bytes.h"
#include "base/ids.h"
#include "providers/device_execution.h"

namespace jitllm::providers::fake {

class FakeDeviceExecution final : public DeviceExecution {
 public:
  std::expected<StreamId, Failure> CreateStream() override;
  std::expected<void, Failure> DestroyStream(StreamId stream) override;
  std::expected<void, Failure> Copy(StreamId stream, std::uint64_t destination,
                                    std::uint64_t source, Bytes size) override;
  std::expected<void, Failure> Zero(StreamId stream, std::uint64_t destination,
                                    Bytes size) override;
  std::expected<NativeStream, Failure> Submission(StreamId stream) override;
  std::expected<void, Failure> Wait(StreamId stream, FenceId fence) override;
  std::expected<FenceId, Failure> Record(StreamId stream) override;
  std::expected<FenceState, Failure> Query(FenceId fence) override;
  std::expected<void, Failure> Release(FenceId fence) override;

  // Runs the stream's next step, if it can: false if the stream is empty
  // or waiting on a fence that is not yet complete.
  bool Step(StreamId stream);
  // Steps every stream until none can move.
  void Drain();
  // The next query of `fence` reports this failure (a device fault).
  void FailNextQuery(FenceId fence, ProviderError error) {
    const std::scoped_lock lock(mutex_);
    fault_ = {.fence = fence, .error = error, .times = 1};
  }
  // The next `times` queries of any fence report this failure; zero stops.
  void FailNextQuery(ProviderError error, std::size_t times = 1) {
    const std::scoped_lock lock(mutex_);
    fault_ = {.fence = FenceId{}, .error = error, .times = times};
  }
  // The next `times` copies (zeroing too) report this failure. A known one queues
  // nothing; kUnknown queues the copy anyway, as a fault whose effect is
  // unknown may have, and it runs when the stream is stepped. Zero stops.
  void FailNextCopy(ProviderError error, std::size_t times = 1) {
    const std::scoped_lock lock(mutex_);
    copy_fault_ = {.fence = FenceId{}, .error = error, .times = times};
  }
  // The next `times` releases of any fence report this failure and change
  // nothing; zero stops.
  void FailNextRelease(ProviderError error, std::size_t times) {
    const std::scoped_lock lock(mutex_);
    release_fault_ = {.fence = FenceId{}, .error = error, .times = times};
  }

  std::size_t streams() const {
    const std::scoped_lock lock(mutex_);
    return streams_.size();
  }
  std::size_t fences() const {
    const std::scoped_lock lock(mutex_);
    return fences_.size();
  }

 private:
  struct Queued {
    enum class Kind : std::uint8_t { kCopy, kZero, kWait, kFence } kind = Kind::kCopy;
    std::uint64_t destination = 0;
    std::uint64_t source = 0;
    Bytes size;
    FenceId fence;
  };
  struct Stream {
    std::deque<Queued> steps;
    std::size_t fences = 0;    // unreleased fences recorded on it
    bool unfenced = false;     // work queued since its last fence
    std::uint64_t native = 0;  // the token Submission hands out
  };
  struct Fence {
    StreamId stream;
    bool complete = false;
    bool seen = false;  // a query reported it complete
  };

  bool StepLocked(StreamId stream);

  mutable std::mutex mutex_;
  base::SlotTable<StreamTag, Stream> streams_;
  base::SlotTable<FenceTag, Fence> fences_;
  struct Fault {
    FenceId fence;  // invalid for any fence
    ProviderError error = ProviderError::kFailed;
    std::size_t times = 0;
  };
  // Takes one failure from `fault` for `fence`, if it applies.
  static std::optional<ProviderError> Take(Fault& fault, FenceId fence);
  Fault fault_;          // queries
  Fault release_fault_;  // releases
  Fault copy_fault_;     // copies
  std::uint64_t next_native_ = 0;
};

}  // namespace jitllm::providers::fake

#endif  // JITLLM_PROVIDERS_FAKE_FAKE_DEVICE_EXECUTION_H_
