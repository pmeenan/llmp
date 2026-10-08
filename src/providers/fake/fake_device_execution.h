// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The deterministic device-execution fake: each stream is a queue of
// copies, cross-stream waits, host work and fence markers that runs only
// when a test steps it (Progress::kStepped), so completion order is the
// test's to choose, or, for the fake device runtime
// (providers/fake/fake_device_runtime.h), whenever a fence is queried
// (Progress::kOnQuery): then every stream runs as far as it can before the
// query answers, as a device would have run it meanwhile. Copies move bytes
// between the fake device memory's addresses (which are host addresses)
// when they run, so a consumer that reads before its fence completes sees
// the old bytes. A stream's native handle is a distinct token, unique in
// the process, that nothing dereferences: Launch and the recording calls
// find the stream by it, for the fake device runtime and the host paging
// kernels (kernels/paging/host_paging.cc), which queue host work on it.
// Releasing a fence before it was seen complete, or destroying a stream
// with work not yet behind a released fence, is refused, as the CUDA
// provider refuses them. Every call takes one lock, so a completion lane
// may query while the submission lane submits; host work runs under it and
// must not call back into the fake.

#ifndef LLMP_PROVIDERS_FAKE_FAKE_DEVICE_EXECUTION_H_
#define LLMP_PROVIDERS_FAKE_FAKE_DEVICE_EXECUTION_H_

#include <cstddef>
#include <cstdint>
#include <deque>
#include <expected>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

#include "base/bytes.h"
#include "base/ids.h"
#include "providers/device_execution.h"

namespace llmp::providers::fake {

class FakeDeviceExecution final : public DeviceExecution {
 public:
  // When queued work runs: only when a test steps it, or also whenever a
  // fence is queried.
  enum class Progress : std::uint8_t { kStepped, kOnQuery };
  // Work queued on a stream as a device would run it: when the stream
  // reaches it, after everything queued before.
  using HostWork = std::move_only_function<void()>;
  // Work a stream recorded (BeginRecording to EndRecording), replayed whole.
  using Recording = std::vector<HostWork>;

  explicit FakeDeviceExecution(Progress progress = Progress::kStepped) : progress_(progress) {}
  ~FakeDeviceExecution() override;
  FakeDeviceExecution(const FakeDeviceExecution&) = delete;
  FakeDeviceExecution& operator=(const FakeDeviceExecution&) = delete;
  FakeDeviceExecution(FakeDeviceExecution&&) = delete;
  FakeDeviceExecution& operator=(FakeDeviceExecution&&) = delete;

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

  // Queues `work` on the stream whose native handle is `stream`, in any
  // fake's streams; or, while that stream records, appends it to the
  // recording. Refused (kInvalid) if no live fake stream has that handle.
  static std::expected<void, Failure> Launch(NativeStream stream, HostWork work);
  // From here until EndRecording, what is queued on the stream (Launch,
  // Copy, Zero) is recorded instead of run; Wait and Record are refused
  // meanwhile. Refused if it is already recording.
  static std::expected<void, Failure> BeginRecording(NativeStream stream);
  // Ends the recording (refused if there is none) and hands it over.
  static std::expected<std::unique_ptr<Recording>, Failure> EndRecording(NativeStream stream);

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
    enum class Kind : std::uint8_t { kCopy, kZero, kWait, kFence, kHost } kind = Kind::kCopy;
    std::uint64_t destination = 0;
    std::uint64_t source = 0;
    Bytes size;
    FenceId fence;
    HostWork work;  // kHost
  };
  struct Stream {
    std::deque<Queued> steps;
    std::size_t fences = 0;    // unreleased fences recorded on it
    bool unfenced = false;     // work queued since its last fence
    std::uint64_t native = 0;  // the token Submission hands out
    // While it records: what it queues instead of running it.
    std::unique_ptr<Recording> recording;
  };
  struct Fence {
    StreamId stream;
    bool complete = false;
    bool seen = false;  // a query reported it complete
  };

  bool StepLocked(StreamId stream);
  void DrainLocked();
  // The stream and its fake behind a native handle (the process's token
  // registry), or nothing.
  struct Owner {
    FakeDeviceExecution* execution = nullptr;
    StreamId stream;
  };
  static std::optional<Owner> OwnerOf(NativeStream stream);
  std::expected<void, Failure> Enqueue(StreamId stream, HostWork work);
  std::expected<void, Failure> Begin(StreamId stream);
  std::expected<std::unique_ptr<Recording>, Failure> End(StreamId stream);

  const Progress progress_;

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
};

}  // namespace llmp::providers::fake

#endif  // LLMP_PROVIDERS_FAKE_FAKE_DEVICE_EXECUTION_H_
