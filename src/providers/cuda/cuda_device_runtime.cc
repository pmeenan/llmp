// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The device runtime (providers/device_runtime.h) over the CUDA runtime
// API, one call for one call: the engine's calls moved here unchanged, so
// what reaches the driver is what the engine called before.

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <utility>

#include "providers/cuda/cuda_device_execution.h"
#include "providers/cuda/cuda_device_memory.h"
#include "providers/device_execution.h"
#include "providers/device_memory.h"
#include "providers/device_runtime.h"

namespace llmp::providers {
namespace {

static_assert(cudaSuccess == 0, "DeviceStatus takes 0 as success");

cudaStream_t Native(NativeStream stream) { return static_cast<cudaStream_t>(stream.handle); }

DeviceStatus Status(cudaError_t result) { return DeviceStatus(static_cast<int>(result)); }

std::unexpected<DeviceStatus> Failed(cudaError_t result) { return std::unexpected(Status(result)); }

constexpr cudaMemcpyKind Kind(CopyKind kind) {
  switch (kind) {
    case CopyKind::kHostToDevice:
      return cudaMemcpyHostToDevice;
    case CopyKind::kDeviceToHost:
      return cudaMemcpyDeviceToHost;
    case CopyKind::kDeviceToDevice:
      return cudaMemcpyDeviceToDevice;
  }
  return cudaMemcpyDefault;
}

}  // namespace

const char* DeviceStatus::text() const {
  return cudaGetErrorString(static_cast<cudaError_t>(code_));
}

std::expected<Device, Failure> OpenDevice(int ordinal, DeviceSettings settings) {
  if (cudaSetDevice(ordinal) != cudaSuccess || cudaFree(nullptr) != cudaSuccess) {
    return std::unexpected(
        Failure{.error = ProviderError::kFailed,
                .detail = std::format("CUDA device {} has no context", ordinal)});
  }
  auto memory = cuda::OpenDeviceMemory(ordinal);
  if (!memory) {
    return std::unexpected(Failure{.error = memory.error().error,
                                   .detail = "OpenDeviceMemory: " + memory.error().detail});
  }
  auto execution = cuda::OpenDeviceExecution(
      ordinal, {.events_ahead = settings.fences_ahead, .events_kept = settings.fences_kept});
  if (!execution) {
    return std::unexpected(Failure{.error = execution.error().error,
                                   .detail = "OpenDeviceExecution: " + execution.error().detail});
  }
  return Device{.memory = std::move(*memory), .execution = std::move(*execution)};
}

std::expected<DeviceFacts, DeviceStatus> QueryDeviceFacts(int ordinal) {
  int major = 0;
  int minor = 0;
  if (const cudaError_t r =
          cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, ordinal);
      r != cudaSuccess) {
    return Failed(r);
  }
  if (const cudaError_t r =
          cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, ordinal);
      r != cudaSuccess) {
    return Failed(r);
  }
  return DeviceFacts{.architecture = static_cast<std::uint32_t>((100 * major) + (10 * minor))};
}

std::expected<DeviceMemoryInfo, DeviceStatus> QueryDeviceMemory() {
  DeviceMemoryInfo info;
  if (const cudaError_t r = cudaMemGetInfo(&info.free, &info.total); r != cudaSuccess) {
    return Failed(r);
  }
  return info;
}

DeviceStatus CopyAsync(NativeStream stream, void* to, const void* from, std::size_t bytes,
                       CopyKind kind) {
  return Status(cudaMemcpyAsync(to, from, bytes, Kind(kind), Native(stream)));
}

DeviceStatus FillAsync(NativeStream stream, void* to, std::uint8_t value, std::size_t bytes) {
  return Status(cudaMemsetAsync(to, value, bytes, Native(stream)));
}

DeviceStatus TakeLastError() { return Status(cudaGetLastError()); }

DeviceStatus PeekLastError() { return Status(cudaPeekAtLastError()); }

std::expected<TimingMark, DeviceStatus> CreateTimingMark() {
  cudaEvent_t event = nullptr;
  if (const cudaError_t r = cudaEventCreate(&event); r != cudaSuccess) {
    return Failed(r);
  }
  return TimingMark{.handle = event};
}

DeviceStatus RecordTimingMark(TimingMark mark, NativeStream stream) {
  return Status(cudaEventRecord(static_cast<cudaEvent_t>(mark.handle), Native(stream)));
}

std::expected<float, DeviceStatus> ElapsedMilliseconds(TimingMark begin, TimingMark end) {
  float ms = 0;
  if (const cudaError_t r = cudaEventElapsedTime(&ms, static_cast<cudaEvent_t>(begin.handle),
                                                 static_cast<cudaEvent_t>(end.handle));
      r != cudaSuccess) {
    return Failed(r);
  }
  return ms;
}

void DestroyTimingMark(TimingMark mark) {
  if (mark.handle != nullptr) {
    (void)cudaEventDestroy(static_cast<cudaEvent_t>(mark.handle));
  }
}

void RecordedWork::Reset() {
  if (executable_ != nullptr) {
    (void)cudaGraphExecDestroy(static_cast<cudaGraphExec_t>(executable_));
    executable_ = nullptr;
  }
  if (recording_ != nullptr) {
    (void)cudaGraphDestroy(static_cast<cudaGraph_t>(recording_));
    recording_ = nullptr;
  }
}

DeviceStatus BeginRecording(NativeStream stream) {
  return Status(cudaStreamBeginCapture(Native(stream), cudaStreamCaptureModeThreadLocal));
}

std::expected<RecordedWork, DeviceStatus> EndRecording(NativeStream stream) {
  cudaGraph_t graph = nullptr;
  if (const cudaError_t r = cudaStreamEndCapture(Native(stream), &graph); r != cudaSuccess) {
    if (graph != nullptr) {
      (void)cudaGraphDestroy(graph);
    }
    return Failed(r);
  }
  cudaGraphExec_t executable = nullptr;
  if (const cudaError_t r = cudaGraphInstantiate(&executable, graph, 0); r != cudaSuccess) {
    (void)cudaGraphDestroy(graph);
    return Failed(r);
  }
  return RecordedWork(graph, executable);
}

DeviceStatus Replay(const RecordedWork& work, NativeStream stream) {
  return Status(cudaGraphLaunch(static_cast<cudaGraphExec_t>(work.executable()), Native(stream)));
}

std::expected<void*, DeviceStatus> AllocatePinned(std::size_t bytes) {
  void* pointer = nullptr;
  if (const cudaError_t r = cudaMallocHost(&pointer, bytes); r != cudaSuccess) {
    return Failed(r);
  }
  return pointer;
}

void FreePinned(void* pointer) {
  if (pointer != nullptr) {
    (void)cudaFreeHost(pointer);
  }
}

}  // namespace llmp::providers
