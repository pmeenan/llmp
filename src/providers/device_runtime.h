// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The device runtime (D-026, D-053; docs/portability.md): what the engine
// asks of the build's device backend beyond the pager's two interfaces
// (device_memory.h, device_execution.h). Opening the device, work a device
// job queues on the stream its lane hands it (DeviceExecution::Submission):
// copies, fills, timing marks and recorded work (a captured and
// instantiated sequence of the stream's work, replayed as one), the
// calling thread's device error state, pinned host memory, and the device's
// facts and free memory.
//
// Plain functions, not virtual: a build has one device backend (D-028,
// D-053: no runtime plugin ABI), whose provider module defines them. The
// CUDA module (providers/cuda/cuda_device_runtime.cc) wraps the CUDA
// runtime API one call for one call; another backend's module (HIP, Metal)
// would define the same functions over its own runtime. Each is one direct
// call around the backend's own, bound at link time, so a job's hot path
// pays one call and a return-code conversion more than calling the
// backend directly. The engine uses nothing else of the device, so it
// includes no vendor header; kernels (src/kernels/) receive the stream as
// NativeStream's opaque handle and unwrap it themselves.
//
// Threads: stream work, the error state and a job's reads of free memory
// run where the job runs (the device lane's submission thread); the rest
// runs on the node's driver thread, before the lanes run or between the
// jobs it waits for. A recording is
// thread-local: the recording thread queues nothing else on the device
// until it ends. This header holds no vendor types.

#ifndef LLMP_PROVIDERS_DEVICE_RUNTIME_H_
#define LLMP_PROVIDERS_DEVICE_RUNTIME_H_

#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <utility>

#include "providers/device_execution.h"
#include "providers/device_memory.h"

namespace llmp::providers {

// A backend call's outcome: 0 is success; any other value is the backend's
// own error code (a cudaError_t in a CUDA build), which text() names.
class [[nodiscard]] DeviceStatus {
 public:
  constexpr DeviceStatus() = default;
  constexpr explicit DeviceStatus(int code) : code_(code) {}
  constexpr bool ok() const { return code_ == 0; }
  constexpr int code() const { return code_; }
  // The backend's description of it, in static storage.
  const char* text() const;

 private:
  int code_ = 0;
};

// ---- the device

struct DeviceSettings {
  // Fences made when the execution provider opens, and at most kept for
  // reuse (a CUDA build's events: RE-029, cuda_device_execution.h).
  std::size_t fences_ahead = 256;
  std::size_t fences_kept = 4096;
};

struct Device {
  std::unique_ptr<VmmProvider> memory;
  std::unique_ptr<DeviceExecution> execution;
};

// Binds the calling thread to device `ordinal` and makes its context (in a
// CUDA build, the primary context, which runtime-API launches use), then
// opens the device-memory and device-execution providers on it.
std::expected<Device, Failure> OpenDevice(int ordinal, DeviceSettings settings = {});

struct DeviceFacts {
  // The architecture number kernels are chosen by: in a CUDA build
  // 100 × major + 10 × minor of the compute capability (GGML's convention;
  // 1210 on the GB10).
  std::uint32_t architecture = 0;
};
std::expected<DeviceFacts, DeviceStatus> QueryDeviceFacts(int ordinal);

// The calling thread's device's free and total memory, as the backend
// counts them.
struct DeviceMemoryInfo {
  std::size_t free = 0;
  std::size_t total = 0;
};
std::expected<DeviceMemoryInfo, DeviceStatus> QueryDeviceMemory();

// ---- work on a job's stream, after what is queued there

enum class CopyKind : std::uint8_t { kHostToDevice, kDeviceToHost, kDeviceToDevice };

// `bytes` from `from` to `to`. Host memory is pinned (AllocatePinned) for
// the copy to be asynchronous.
DeviceStatus CopyAsync(NativeStream stream, void* to, const void* from, std::size_t bytes,
                       CopyKind kind);
// `bytes` of device memory at `to` set to `value`.
DeviceStatus FillAsync(NativeStream stream, void* to, std::uint8_t value, std::size_t bytes);

// The calling thread's last device error (a failed launch reports there):
// TakeLastError reads and clears it, PeekLastError only reads it.
DeviceStatus TakeLastError();
DeviceStatus PeekLastError();

// ---- timing marks: points on a stream whose device times can be compared

struct TimingMark {
  void* handle = nullptr;
};
std::expected<TimingMark, DeviceStatus> CreateTimingMark();
DeviceStatus RecordTimingMark(TimingMark mark, NativeStream stream);
// The device time from `begin` to `end`, both recorded and complete.
std::expected<float, DeviceStatus> ElapsedMilliseconds(TimingMark begin, TimingMark end);
// Once nothing queued records it.
void DestroyTimingMark(TimingMark mark);

// ---- recorded work: a stream's work captured once, replayed as one (a CUDA
// graph in a CUDA build; D-090)

// Move-only. Destroying it (or assigning over it) releases the recording,
// which is safe only once no replay of it is still queued.
class RecordedWork {
 public:
  RecordedWork() = default;
  // For the backend's implementation: takes ownership of its handles.
  RecordedWork(void* recording, void* executable)
      : recording_(recording), executable_(executable) {}
  RecordedWork(RecordedWork&& other) noexcept
      : recording_(std::exchange(other.recording_, nullptr)),
        executable_(std::exchange(other.executable_, nullptr)) {}
  RecordedWork& operator=(RecordedWork&& other) noexcept {
    if (this != &other) {
      Reset();
      recording_ = std::exchange(other.recording_, nullptr);
      executable_ = std::exchange(other.executable_, nullptr);
    }
    return *this;
  }
  RecordedWork(const RecordedWork&) = delete;
  RecordedWork& operator=(const RecordedWork&) = delete;
  ~RecordedWork() { Reset(); }

  bool valid() const { return executable_ != nullptr; }
  void* executable() const { return executable_; }
  void Reset();

 private:
  void* recording_ = nullptr;
  void* executable_ = nullptr;
};

// From here until EndRecording, what this thread queues on `stream` is
// recorded instead of run.
DeviceStatus BeginRecording(NativeStream stream);
// Ends the recording (always, even when it fails) and makes it replayable.
std::expected<RecordedWork, DeviceStatus> EndRecording(NativeStream stream);
DeviceStatus Replay(const RecordedWork& work, NativeStream stream);

// ---- pinned host memory, which the device reads and writes in place

std::expected<void*, DeviceStatus> AllocatePinned(std::size_t bytes);
// Once nothing queued reads or writes it.
void FreePinned(void* pointer);

}  // namespace llmp::providers

#endif  // LLMP_PROVIDERS_DEVICE_RUNTIME_H_
