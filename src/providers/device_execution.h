// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The device-execution provider (D-026, D-048, D-053;
// docs/architecture.md#providers): llmpalooza-owned streams, copies between
// backing ranges on them, and fences recorded after a phase's last
// consumer and queried without blocking. Kernel launches come with the
// backend proof's operation contract; this is the part the pager needs.
//
// A fence names the work queued on its stream before it was recorded, and
// nothing else. Querying never blocks. A fence is released only once it
// has been observed complete: destroying a pending event would lose the
// completion, and its memory's retirement with it (docs/async-model.md).
// A stream is destroyed only when everything queued on it lies behind a
// fence that has been seen complete and released: destroying a stream does
// not wait for its work.
//
// Calls run on the device submission lane; Query may run concurrently on
// the device completion lane, independent of a blocking submission
// (D-048), and implementations are safe for that pair of callers. A
// driver's direct steps (engine/paged_node.h) also call Submission,
// Record, Query and Release, for their own stream and fences alone,
// concurrently with both lanes: implementations are safe for calls on
// distinct streams and fences from any threads. This header holds no
// vendor types.

#ifndef LLMP_PROVIDERS_DEVICE_EXECUTION_H_
#define LLMP_PROVIDERS_DEVICE_EXECUTION_H_

#include <cstdint>
#include <expected>

#include "base/bytes.h"
#include "base/ids.h"
#include "providers/device_memory.h"

namespace llmp::providers {

struct StreamTag {
  static constexpr const char* kName = "stream";
};
struct FenceTag {
  static constexpr const char* kName = "fence";
};
using StreamId = base::Id<StreamTag>;
using FenceId = base::Id<FenceTag>;

enum class FenceState : std::uint8_t { kPending, kComplete };

// A stream's native handle, opaque to the core: only a kernel module built
// for the provider's device interprets it (D-053).
struct NativeStream {
  void* handle = nullptr;
};

class DeviceExecution {
 public:
  DeviceExecution() = default;
  DeviceExecution(const DeviceExecution&) = delete;
  DeviceExecution& operator=(const DeviceExecution&) = delete;
  DeviceExecution(DeviceExecution&&) = delete;
  DeviceExecution& operator=(DeviceExecution&&) = delete;
  virtual ~DeviceExecution() = default;

  virtual std::expected<StreamId, Failure> CreateStream() = 0;
  virtual std::expected<void, Failure> DestroyStream(StreamId stream) = 0;

  // Queues a copy of `size` bytes between device addresses (backing mapped
  // with access), after everything queued on the stream before it.
  virtual std::expected<void, Failure> Copy(StreamId stream, std::uint64_t destination,
                                            std::uint64_t source, Bytes size) = 0;
  // Queues zeroing `size` bytes at a device address (backing mapped with
  // access), after everything queued on the stream before it; like Copy,
  // it is work the stream must be fenced after.
  virtual std::expected<void, Failure> Zero(StreamId stream, std::uint64_t destination,
                                            Bytes size) = 0;
  // The stream's native handle, for a kernel implementation about to queue
  // work on it (D-053). Like Copy, it notes that work is being queued, so
  // the stream needs a fence recorded after that work, seen complete and
  // released, before it can be destroyed. The handle is valid until the
  // stream is destroyed and is used on the calling thread only. The CUDA
  // provider also makes its context current on the calling thread, which
  // is the context runtime-API launches bind to.
  virtual std::expected<NativeStream, Failure> Submission(StreamId stream) = 0;
  // Makes the stream wait for a fence recorded on another stream.
  virtual std::expected<void, Failure> Wait(StreamId stream, FenceId fence) = 0;

  // A fence after everything queued on the stream so far.
  virtual std::expected<FenceId, Failure> Record(StreamId stream) = 0;
  // Never blocks. An error whose effect is unknown (a device fault) is
  // kUnknown: the fenced work's memory stays quarantined until a later
  // successful query proves completion or validated recovery proves
  // quiescence. Query does not change the event's identity and may retry;
  // an unknown destruction outcome must never be retried.
  virtual std::expected<FenceState, Failure> Query(FenceId fence) = 0;
  // Only once a query has seen it complete.
  virtual std::expected<void, Failure> Release(FenceId fence) = 0;
};

}  // namespace llmp::providers

#endif  // LLMP_PROVIDERS_DEVICE_EXECUTION_H_
