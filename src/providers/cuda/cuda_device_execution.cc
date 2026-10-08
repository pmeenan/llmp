// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "providers/cuda/cuda_device_execution.h"

#include <cuda.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <memory>
#include <mutex>
#include <vector>

#include "base/bytes.h"
#include "base/ids.h"
#include "providers/cuda/cuda_errors.h"
#include "providers/device_execution.h"

namespace llmp::providers::cuda {
namespace {

std::unexpected<Failure> Invalid(const char* detail) {
  return std::unexpected(Failure{.error = ProviderError::kInvalid, .detail = detail});
}

// Fences are events from a pool, made ahead and kept when released (the
// header): cuEventCreate, like cuStreamCreate, blocks for as long as
// another thread is blocked launching into a full stream (RE-029), while
// recording, querying and copying on other streams do not.
class CudaDeviceExecution final : public DeviceExecution {
 public:
  CudaDeviceExecution(CUdevice device, CUcontext context, ExecutionSettings settings)
      : device_(device), context_(context), settings_(settings) {}
  ~CudaDeviceExecution() override {
    if (Current()) {
      for (CUevent event : spare_) {
        (void)cuEventDestroy(event);
      }
    }
    (void)cuDevicePrimaryCtxRelease(device_);
  }

  // Makes the pool's events ahead of use; a failure leaves them to be made
  // when needed.
  void MakeEventsAhead() {
    if (!Current()) {
      return;
    }
    const std::scoped_lock lock(mutex_);
    spare_.reserve(settings_.events_kept);
    while (spare_.size() < std::min(settings_.events_ahead, settings_.events_kept)) {
      CUevent event = nullptr;
      if (cuEventCreate(&event, CU_EVENT_DISABLE_TIMING) != CUDA_SUCCESS) {
        return;
      }
      spare_.push_back(event);
    }
  }

  CudaDeviceExecution(const CudaDeviceExecution&) = delete;
  CudaDeviceExecution& operator=(const CudaDeviceExecution&) = delete;
  CudaDeviceExecution(CudaDeviceExecution&&) = delete;
  CudaDeviceExecution& operator=(CudaDeviceExecution&&) = delete;

  std::expected<StreamId, Failure> CreateStream() override {
    if (auto current = Current(); !current) {
      return std::unexpected(current.error());
    }
    CUstream stream = nullptr;
    if (const CUresult result = cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING);
        result != CUDA_SUCCESS) {
      return Error(result, "cuStreamCreate");
    }
    StreamId id;
    {
      const std::scoped_lock lock(mutex_);
      id = streams_.Insert(
          Stream{.handle = stream, .fences = 0, .unfenced = false, .faulted = false});
    }
    if (!id.valid()) {
      (void)cuStreamDestroy(stream);  // nothing was queued on it
      return std::unexpected(
          Failure{.error = ProviderError::kFailed, .detail = "the stream table is full"});
    }
    return id;
  }

  std::expected<void, Failure> DestroyStream(StreamId stream) override {
    CUstream handle = nullptr;
    {
      const std::scoped_lock lock(mutex_);
      const Stream* found = streams_.Find(stream);
      if (found == nullptr) {
        return Invalid("stale or unknown stream");
      }
      // Destroying a stream does not wait for its work: everything queued
      // on it must lie behind a fence that has been seen complete and
      // released.
      if (found->faulted) {
        return Invalid("the stream is undetermined after an unknown outcome");
      }
      if (found->fences > 0 || found->unfenced) {
        return Invalid("the stream has work not yet behind a completed fence");
      }
      handle = found->handle;
    }
    // Only the submission lane changes streams, so the entry is still ours.
    auto destroyed = Current();
    if (destroyed) {
      if (const CUresult result = cuStreamDestroy(handle); result != CUDA_SUCCESS) {
        destroyed = Error(result, "cuStreamDestroy");
      }
    }
    const std::scoped_lock lock(mutex_);
    if (!destroyed) {
      // A known failure changed nothing; an unknown one leaves the stream
      // for its owner to quarantine, never to retry.
      streams_.Find(stream)->faulted = destroyed.error().error == ProviderError::kUnknown;
      return destroyed;
    }
    (void)streams_.Erase(stream);
    return {};
  }

  std::expected<void, Failure> Copy(StreamId stream, std::uint64_t destination,
                                    std::uint64_t source, Bytes size) override {
    auto handle = Queue(stream);
    if (!handle) {
      return std::unexpected(handle.error());
    }
    if (auto current = Current(); !current) {
      return current;
    }
    if (const CUresult result =
            cuMemcpyAsync(static_cast<CUdeviceptr>(destination), static_cast<CUdeviceptr>(source),
                          size.value(), *handle);
        result != CUDA_SUCCESS) {
      return Error(result, "cuMemcpyAsync");
    }
    return {};
  }

  std::expected<void, Failure> Zero(StreamId stream, std::uint64_t destination,
                                    Bytes size) override {
    auto handle = Queue(stream);
    if (!handle) {
      return std::unexpected(handle.error());
    }
    if (auto current = Current(); !current) {
      return current;
    }
    if (const CUresult result =
            cuMemsetD8Async(static_cast<CUdeviceptr>(destination), 0, size.value(), *handle);
        result != CUDA_SUCCESS) {
      return Error(result, "cuMemsetD8Async");
    }
    return {};
  }

  std::expected<NativeStream, Failure> Submission(StreamId stream) override {
    auto handle = Queue(stream);
    if (!handle) {
      return std::unexpected(handle.error());
    }
    if (auto current = Current(); !current) {
      return std::unexpected(current.error());
    }
    return NativeStream{.handle = *handle};
  }

  std::expected<void, Failure> Wait(StreamId stream, FenceId fence) override {
    CUevent event = nullptr;
    {
      const std::scoped_lock lock(mutex_);
      const Fence* awaited = fences_.Find(fence);
      if (awaited == nullptr || awaited->faulted) {
        return Invalid("stale, unknown or undetermined fence");
      }
      event = awaited->event;
    }
    auto handle = Queue(stream);
    if (!handle) {
      return std::unexpected(handle.error());
    }
    if (auto current = Current(); !current) {
      return current;
    }
    if (const CUresult result = cuStreamWaitEvent(*handle, event, 0); result != CUDA_SUCCESS) {
      return Error(result, "cuStreamWaitEvent");
    }
    return {};
  }

  std::expected<FenceId, Failure> Record(StreamId stream) override {
    CUstream handle = nullptr;
    FenceId id;
    {
      // The table entry first, so a full table changes nothing.
      const std::scoped_lock lock(mutex_);
      const Stream* found = streams_.Find(stream);
      if (found == nullptr || found->faulted) {
        return Invalid("stale, unknown or undetermined stream");
      }
      handle = found->handle;
      id = fences_.Insert(Fence{.event = nullptr,
                                .stream = stream,
                                .seen = false,
                                .queries = 0,
                                .releasing = false,
                                .faulted = false});
      if (!id.valid()) {
        return std::unexpected(
            Failure{.error = ProviderError::kFailed, .detail = "the fence table is full"});
      }
    }
    CUevent event = nullptr;
    CUresult result = CUDA_SUCCESS;
    if (auto current = Current(); !current) {
      Forget(id);
      return std::unexpected(current.error());
    }
    {
      const std::scoped_lock lock(mutex_);
      if (!spare_.empty()) {
        event = spare_.back();
        spare_.pop_back();
      }
    }
    if (event == nullptr) {
      result = cuEventCreate(&event, CU_EVENT_DISABLE_TIMING);
    }
    if (result == CUDA_SUCCESS) {
      result = cuEventRecord(event, handle);
      if (result != CUDA_SUCCESS) {
        // CUDA permits destroying a pending event; an unknown record
        // result still leaves the caller's work awaiting recovery.
        (void)cuEventDestroy(event);
      }
    }
    if (result != CUDA_SUCCESS) {
      Forget(id);
      return Error(result, "cuEventRecord");
    }
    const std::scoped_lock lock(mutex_);
    fences_.Find(id)->event = event;
    Stream* found = streams_.Find(stream);
    ++found->fences;
    found->unfenced = false;
    return id;
  }

  std::expected<FenceState, Failure> Query(FenceId fence) override {
    CUevent event = nullptr;
    {
      const std::scoped_lock lock(mutex_);
      Fence* found = fences_.Find(fence);
      if (found == nullptr || found->event == nullptr || found->releasing || found->faulted) {
        return Invalid("stale, unknown or undetermined fence");
      }
      event = found->event;
      ++found->queries;  // the event stays valid until this query ends
    }
    auto current = Current();
    const CUresult result = current ? cuEventQuery(event) : CUDA_SUCCESS;
    const std::scoped_lock lock(mutex_);
    Fence& found = *fences_.Find(fence);
    --found.queries;
    if (!current) {
      return std::unexpected(current.error());  // its own outcome, unmasked
    }
    if (result == CUDA_ERROR_NOT_READY) {
      return FenceState::kPending;
    }
    if (result != CUDA_SUCCESS) {
      return Error(result, "cuEventQuery");
    }
    found.seen = true;
    return FenceState::kComplete;
  }

  std::expected<void, Failure> Release(FenceId fence) override {
    CUevent event = nullptr;
    {
      const std::scoped_lock lock(mutex_);
      Fence* found = fences_.Find(fence);
      if (found == nullptr || found->event == nullptr || found->faulted) {
        return Invalid("stale, unknown or undetermined fence");
      }
      if (!found->seen) {
        return Invalid("the fence has not been seen complete");
      }
      if (found->queries > 0) {
        return std::unexpected(Failure{.error = ProviderError::kFailed,
                                       .detail = "the fence is being queried; release it later"});
      }
      if (spare_.size() < settings_.events_kept) {
        // Kept for the next fence: seen complete, so nothing waits on it,
        // and no query is under way.
        spare_.push_back(found->event);
        --streams_.Find(found->stream)->fences;
        (void)fences_.Erase(fence);
        return {};
      }
      event = found->event;
      found->releasing = true;  // no new query may race cuEventDestroy
    }
    auto released = Current();
    if (released) {
      if (const CUresult result = cuEventDestroy(event); result != CUDA_SUCCESS) {
        released = Error(result, "cuEventDestroy");
      }
    }
    const std::scoped_lock lock(mutex_);
    Fence* found = fences_.Find(fence);
    if (!released) {
      found->releasing = false;
      found->faulted = released.error().error == ProviderError::kUnknown;
      return released;
    }
    --streams_.Find(found->stream)->fences;
    (void)fences_.Erase(fence);
    return {};
  }

 private:
  struct Stream {
    CUstream handle = nullptr;
    std::size_t fences = 0;  // recorded and not yet released
    bool unfenced = false;   // work queued since its last fence
    bool faulted = false;    // a destroy's outcome was unknown
  };
  struct Fence {
    CUevent event = nullptr;  // null until recorded
    StreamId stream;
    bool seen = false;
    std::uint32_t queries = 0;  // queries in progress on another lane
    bool releasing = false;     // destruction in progress on the submission lane
    bool faulted = false;       // destruction's outcome unknown: never touch it again
  };

  // The stream's handle, noting that work is being queued on it.
  std::expected<CUstream, Failure> Queue(StreamId stream) {
    const std::scoped_lock lock(mutex_);
    Stream* found = streams_.Find(stream);
    if (found == nullptr || found->faulted) {
      return Invalid("stale, unknown or undetermined stream");
    }
    found->unfenced = true;
    return found->handle;
  }

  void Forget(FenceId fence) {
    const std::scoped_lock lock(mutex_);
    (void)fences_.Erase(fence);
  }

  std::expected<void, Failure> Current() {
    if (const CUresult result = cuCtxSetCurrent(context_); result != CUDA_SUCCESS) {
      return Error(result, "cuCtxSetCurrent");
    }
    return {};
  }

  CUdevice device_;
  CUcontext context_;
  // Guards the tables only, never across a driver call: submission and
  // completion run on different lanes (D-048).
  std::mutex mutex_;
  base::SlotTable<StreamTag, Stream> streams_;
  base::SlotTable<FenceTag, Fence> fences_;
  ExecutionSettings settings_;
  std::vector<CUevent> spare_;  // events no fence holds, at most settings_.events_kept
};

}  // namespace

std::expected<std::unique_ptr<DeviceExecution>, Failure> OpenDeviceExecution(
    int ordinal, ExecutionSettings settings) {
  if (const CUresult result = cuInit(0); result != CUDA_SUCCESS) {
    return Error(result, "cuInit");
  }
  CUdevice device = 0;
  if (const CUresult result = cuDeviceGet(&device, ordinal); result != CUDA_SUCCESS) {
    return Error(result, "cuDeviceGet");
  }
  CUcontext context = nullptr;
  if (const CUresult result = cuDevicePrimaryCtxRetain(&context, device); result != CUDA_SUCCESS) {
    return Error(result, "cuDevicePrimaryCtxRetain");
  }
  auto execution = std::make_unique<CudaDeviceExecution>(device, context, settings);
  execution->MakeEventsAhead();
  return execution;
}

}  // namespace llmp::providers::cuda
