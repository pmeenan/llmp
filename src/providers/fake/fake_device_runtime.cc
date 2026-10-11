// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The device runtime (providers/device_runtime.h) over the fakes
// (fake_device_runtime.h): every call that queues work queues host work on
// the fake stream behind the job's native handle.

#include "providers/fake/fake_device_runtime.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <expected>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <utility>

#include "base/bytes.h"
#include "providers/device_execution.h"
#include "providers/device_memory.h"
#include "providers/device_runtime.h"
#include "providers/fake/fake_device_execution.h"
#include "providers/fake/fake_device_memory.h"

namespace llmp::providers {
namespace {

using fake::FakeDeviceExecution;
using fake::RuntimeError;

constexpr std::uint64_t kGranularity = std::uint64_t{2} << 20U;  // the GB10's VMM granularity
constexpr std::size_t kPinnedAlignment = 256;

fake::FakeDeviceSettings& Settings() {
  static fake::FakeDeviceSettings settings;
  return settings;
}

// The calling thread's last failed call, as the CUDA runtime keeps it.
thread_local int last_error = 0;
struct CopyFailure {
  CopyKind kind;
  std::size_t bytes;
  bool queue_first;
  std::size_t matches_before;
};
std::mutex copy_failure_mutex;
std::optional<CopyFailure> copy_failure;
struct PinnedFreeHook {
  void* pointer;
  std::function<void()> callback;
};
std::mutex pinned_free_mutex;
std::optional<PinnedFreeHook> pinned_free_hook;
std::mutex pinned_alignment_mutex;
std::optional<std::size_t> pinned_misalignment;
std::map<void*, void*> misaligned_pinned;

std::function<void()> TakePinnedFreeHook(void* pointer) {
  const std::lock_guard lock(pinned_free_mutex);
  if (!pinned_free_hook || pinned_free_hook->pointer != pointer) return {};
  auto callback = std::move(pinned_free_hook->callback);
  pinned_free_hook.reset();
  return callback;
}

std::optional<CopyFailure> TakeCopyFailure(CopyKind kind, std::size_t bytes) {
  const std::lock_guard lock(copy_failure_mutex);
  if (!copy_failure || copy_failure->kind != kind || copy_failure->bytes != bytes) return {};
  if (copy_failure->matches_before != 0) {
    --copy_failure->matches_before;
    return {};
  }
  return std::exchange(copy_failure, std::nullopt);
}

DeviceStatus Fail(RuntimeError error) {
  last_error = static_cast<int>(error);
  return DeviceStatus(last_error);
}

std::unexpected<DeviceStatus> Failed(RuntimeError error) { return std::unexpected(Fail(error)); }

DeviceStatus Queue(NativeStream stream, FakeDeviceExecution::HostWork work) {
  if (!FakeDeviceExecution::Launch(stream, std::move(work))) {
    return Fail(RuntimeError::kInvalidStream);
  }
  return {};
}

// A timing mark: the steady clock's ticks when its stream reached it, or -1.
struct Mark {
  std::atomic<std::int64_t> ticks{-1};
};

}  // namespace

const char* DeviceStatus::text() const {
  switch (code_) {
    case 0:
      return "no error";
    case static_cast<int>(RuntimeError::kInvalidStream):
      return "no fake stream has this native handle";
    case static_cast<int>(RuntimeError::kInvalidValue):
      return "an invalid argument";
    case static_cast<int>(RuntimeError::kRecording):
      return "the stream's recording was misused";
    case static_cast<int>(RuntimeError::kNotReady):
      return "the timing mark has not run";
    case static_cast<int>(RuntimeError::kNoMemory):
      return "pinned memory could not be allocated";
    case static_cast<int>(RuntimeError::kNoDevice):
      return "the fake device has only ordinal 0";
    default:
      return "an unknown fake runtime error";
  }
}

std::expected<Device, Failure> OpenDevice(int ordinal, DeviceSettings /*settings*/) {
  if (ordinal != 0) {
    return std::unexpected(
        Failure{.error = ProviderError::kInvalid, .detail = "the fake device has only ordinal 0"});
  }
  return Device{
      .memory = std::make_unique<fake::FakeDeviceMemory>(
          base::Bytes(kGranularity), Settings().capacity, fake::Contents::kPoisoned),
      .execution = std::make_unique<FakeDeviceExecution>(FakeDeviceExecution::Progress::kOnQuery)};
}

std::expected<DeviceFacts, DeviceStatus> QueryDeviceFacts(int ordinal) {
  if (ordinal != 0) {
    return Failed(RuntimeError::kNoDevice);
  }
  return DeviceFacts{.architecture = 0};
}

std::expected<DeviceMemoryInfo, DeviceStatus> QueryDeviceMemory() {
  const auto capacity = static_cast<std::size_t>(Settings().capacity.value());
  return DeviceMemoryInfo{.free = capacity, .total = capacity};
}

DeviceStatus CopyAsync(NativeStream stream, void* to, const void* from, std::size_t bytes,
                       CopyKind kind) {
  if (kind != CopyKind::kHostToDevice && kind != CopyKind::kDeviceToHost &&
      kind != CopyKind::kDeviceToDevice) {
    return Fail(RuntimeError::kInvalidValue);
  }
  if (bytes != 0 && (to == nullptr || from == nullptr)) {
    return Fail(RuntimeError::kInvalidValue);
  }
  const auto failure = TakeCopyFailure(kind, bytes);
  if (failure && !failure->queue_first) return Fail(RuntimeError::kInvalidValue);
  const auto queued = Queue(stream, [to, from, bytes] {
    if (bytes != 0) std::memmove(to, from, bytes);
  });
  return queued.ok() && failure ? Fail(RuntimeError::kInvalidValue) : queued;
}

DeviceStatus FillAsync(NativeStream stream, void* to, std::uint8_t value, std::size_t bytes) {
  if (bytes != 0 && to == nullptr) {
    return Fail(RuntimeError::kInvalidValue);
  }
  return Queue(stream, [to, value, bytes] {
    if (bytes != 0) {
      std::memset(to, value, bytes);
    }
  });
}

DeviceStatus TakeLastError() { return DeviceStatus(std::exchange(last_error, 0)); }

DeviceStatus PeekLastError() { return DeviceStatus(last_error); }

std::expected<TimingMark, DeviceStatus> CreateTimingMark() {
  return TimingMark{.handle = new Mark()};  // NOLINT(cppcoreguidelines-owning-memory)
}

DeviceStatus RecordTimingMark(TimingMark mark, NativeStream stream) {
  if (mark.handle == nullptr) {
    return Fail(RuntimeError::kInvalidValue);
  }
  auto* const m = static_cast<Mark*>(mark.handle);
  m->ticks.store(-1, std::memory_order_relaxed);
  return Queue(stream, [m] {
    m->ticks.store(std::chrono::steady_clock::now().time_since_epoch().count(),
                   std::memory_order_release);
  });
}

std::expected<float, DeviceStatus> ElapsedMilliseconds(TimingMark begin, TimingMark end) {
  if (begin.handle == nullptr || end.handle == nullptr) {
    return Failed(RuntimeError::kInvalidValue);
  }
  const std::int64_t from = static_cast<Mark*>(begin.handle)->ticks.load(std::memory_order_acquire);
  const std::int64_t to = static_cast<Mark*>(end.handle)->ticks.load(std::memory_order_acquire);
  if (from < 0 || to < 0) {
    return Failed(RuntimeError::kNotReady);
  }
  const std::chrono::steady_clock::duration span(to - from);
  return std::chrono::duration<float, std::milli>(span).count();
}

void DestroyTimingMark(TimingMark mark) {
  delete static_cast<Mark*>(mark.handle);  // NOLINT(cppcoreguidelines-owning-memory)
}

void RecordedWork::Reset() {
  // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
  delete static_cast<FakeDeviceExecution::Recording*>(executable_);
  executable_ = nullptr;
  recording_ = nullptr;
}

DeviceStatus BeginRecording(NativeStream stream) {
  if (auto begun = FakeDeviceExecution::BeginRecording(stream); !begun) {
    return Fail(RuntimeError::kRecording);
  }
  return {};
}

std::expected<RecordedWork, DeviceStatus> EndRecording(NativeStream stream) {
  auto recording = FakeDeviceExecution::EndRecording(stream);
  if (!recording) {
    return Failed(RuntimeError::kRecording);
  }
  return RecordedWork(nullptr, recording->release());
}

DeviceStatus Replay(const RecordedWork& work, NativeStream stream) {
  if (!work.valid()) {
    return Fail(RuntimeError::kInvalidValue);
  }
  // The recording must outlive the replay's run, as a CUDA graph must
  // (device_runtime.h).
  auto* const recording = static_cast<FakeDeviceExecution::Recording*>(work.executable());
  return Queue(stream, [recording] {
    for (FakeDeviceExecution::HostWork& step : *recording) {
      step();
    }
  });
}

std::expected<void*, DeviceStatus> AllocatePinned(std::size_t bytes) {
  {
    const std::lock_guard lock(pinned_alignment_mutex);
    if (pinned_misalignment && *pinned_misalignment == bytes) {
      pinned_misalignment.reset();
      constexpr std::size_t page = 4096;
      if (bytes > std::numeric_limits<std::size_t>::max() - kPinnedAlignment - (page - 1))
        return Failed(RuntimeError::kNoMemory);
      const auto rounded = (bytes + kPinnedAlignment + page - 1) & ~(page - 1);
      void* backing = std::aligned_alloc(page, rounded);  // NOLINT(cppcoreguidelines-no-malloc)
      if (backing == nullptr) return Failed(RuntimeError::kNoMemory);
      auto* pointer = static_cast<std::byte*>(backing) + kPinnedAlignment;
      misaligned_pinned.emplace(pointer, backing);
      return pointer;
    }
  }
  const std::size_t rounded = std::max<std::size_t>(
      kPinnedAlignment, (bytes + kPinnedAlignment - 1) & ~(kPinnedAlignment - 1));
  void* pointer =
      std::aligned_alloc(kPinnedAlignment, rounded);  // NOLINT(cppcoreguidelines-no-malloc)
  if (pointer == nullptr) {
    return Failed(RuntimeError::kNoMemory);
  }
  return pointer;
}

void FreePinned(void* pointer) {
  auto after_free = TakePinnedFreeHook(pointer);
  void* backing = pointer;
  {
    const std::lock_guard lock(pinned_alignment_mutex);
    if (const auto found = misaligned_pinned.find(pointer); found != misaligned_pinned.end()) {
      backing = found->second;
      misaligned_pinned.erase(found);
    }
  }
  std::free(backing);  // NOLINT(cppcoreguidelines-no-malloc)
  if (after_free) after_free();
}

}  // namespace llmp::providers

namespace llmp::providers::fake {

void FailNextCopy(CopyKind kind, std::size_t bytes, bool queue_first, std::size_t matches_before) {
  const std::lock_guard lock(copy_failure_mutex);
  copy_failure = CopyFailure{kind, bytes, queue_first, matches_before};
}
void ClearCopyFailure() {
  const std::lock_guard lock(copy_failure_mutex);
  copy_failure.reset();
}

void MisalignNextPinned(std::size_t bytes) {
  const std::lock_guard lock(pinned_alignment_mutex);
  pinned_misalignment = bytes;
}
void ClearPinnedMisalignment() {
  const std::lock_guard lock(pinned_alignment_mutex);
  pinned_misalignment.reset();
}
void AfterNextPinnedFree(void* pointer, std::function<void()> callback) {
  const std::lock_guard lock(pinned_free_mutex);
  pinned_free_hook = PinnedFreeHook{pointer, std::move(callback)};
}
void ClearPinnedFreeHook() {
  const std::lock_guard lock(pinned_free_mutex);
  pinned_free_hook.reset();
}

void ConfigureFakeDevice(FakeDeviceSettings settings) { Settings() = settings; }

}  // namespace llmp::providers::fake
