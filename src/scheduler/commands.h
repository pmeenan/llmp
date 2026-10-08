// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The commands the scheduler thread publishes to its service lanes (D-048,
// docs/async-model.md#submission-and-completion-protocol). Each names the
// operation it belongs to by its generation-tagged identity, whose mailbox
// on the completion board receives what the lane observes. A command is a
// value: it owns no memory, and the backing it names is protected by the
// leases or load the scheduler recorded before publishing it.

#ifndef LLMP_SCHEDULER_COMMANDS_H_
#define LLMP_SCHEDULER_COMMANDS_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <variant>

#include "base/bytes.h"
#include "providers/device_execution.h"
#include "providers/device_memory.h"
#include "providers/direct_reader.h"
#include "scheduler/completions.h"

namespace llmp::scheduler {

using base::Bytes;

// An operation's identity as one number: the storage lane's read key and
// waiter.
constexpr std::uint64_t KeyOf(OperationId operation) {
  return (std::uint64_t{operation.index()} << 32U) | operation.generation();
}
constexpr OperationId OperationOf(std::uint64_t key) {
  return {static_cast<std::uint32_t>(key >> 32U), static_cast<std::uint32_t>(key)};
}

// Storage lane: read a whole range into protected memory, or withdraw
// interest in a read (best-effort cancellation; the read still drains).
struct ReadCommand {
  OperationId operation;
  providers::ReadSpec spec;
};
struct CancelRead {
  OperationId operation;
};
using StorageCommand = std::variant<ReadCommand, CancelRead>;

// Device submission lane: copies queued in order on one of the lane's
// streams, then a fence after them, which the device completion lane
// watches. A `zero` copy zeroes `size` bytes at `destination` and reads
// no source.
struct DeviceCopy {
  std::uint64_t destination = 0;
  std::uint64_t source = 0;
  Bytes size;
  bool zero = false;
};
inline constexpr std::size_t kMaxDeviceCopies = 4;
struct DeviceWork {
  std::uint32_t stream = 0;  // an index into the lane's streams
  std::array<DeviceCopy, kMaxDeviceCopies> copies{};
  std::size_t count = 0;
};

// Device submission lane, VMM work (D-006, D-033): kMap creates backing
// of `size` in the provider's allocation class, maps it at `offset` of the
// reservation and gives it read-write access (the CPU's too, for host
// backing); kUnmap unmaps the backing mapped there and releases it. Both
// run at once on the lane, with nothing to fence: the lane publishes the
// result, and whether the provider's state is known, directly. A kMap
// whose later step fails undoes the earlier ones, so a known failure
// changed nothing.
//
// The handoff (D-033): a kUnmap with `retain` unmaps the backing but keeps
// it, unreleased, among the lane's handed-off backing; a kMap with `reuse`
// maps one of those (of the same class and size) instead of creating
// backing, and on a known failure puts it back; kRelease releases one of
// them. The scheduler keeps each such backing charged to an extent
// throughout (scheduler.h), so the lane never holds one the catalog does
// not count. A reuse or release that finds none of that class and size
// changes nothing and is refused as not started. A `lazy` retain leaves
// the backing mapped at its place: the reuse or release that takes it
// unmaps it there first, and a kMap at a place a kept backing still holds
// takes that one (reuse) or unmaps it first (a created backing's map).
struct BackingWork {
  enum class Kind : std::uint8_t { kMap, kUnmap, kRelease };
  Kind kind = Kind::kMap;
  providers::ReservationId reservation;
  Bytes offset;
  Bytes size;
  std::size_t allocation_class = 0;
  bool retain = false;  // kUnmap: keep the backing for a handoff
  // kUnmap with `retain`: keep it mapped where it is until a load or a
  // release takes it (the unmap then runs as that work's first step).
  bool lazy = false;
  bool reuse = false;  // kMap: map a handed-off backing
};

// Device submission lane: a job that queues kernel work on one of the
// lane's streams (D-053), given the stream's native handle; the lane then
// fences the stream, as it does copies. The job runs on the submission
// lane's thread and must only queue work, never wait for it. It says what
// it queued:
enum class JobResult : std::uint8_t {
  kNotStarted,  // nothing: refused before any launch, or its first a known refusal
  kQueued,      // everything it meant to
  kFailed,      // some work, then a known refusal: it fails once its fence completes
  kUnknown,     // a launch reported an error whose effect is unknown (a fault)
};
using DeviceJob = std::move_only_function<JobResult(providers::NativeStream)>;

// What a job reports when the provider refuses a copy or launch it tried
// to queue, `queued` telling whether it queued anything before. An unknown
// outcome stays unknown even on the first try: the work may still run and
// touch the job's closure, whose lease must then hold until the fence.
// Only a known refusal of the first queued nothing (DeviceService's own
// copies follow the same rule). A job that calls a vendor API directly
// has no provider classification and reports any error of it as unknown.
constexpr JobResult AfterRefusal(providers::ProviderError error, bool queued) {
  if (error == providers::ProviderError::kUnknown) {
    return JobResult::kUnknown;
  }
  return queued ? JobResult::kFailed : JobResult::kNotStarted;
}

struct LaunchWork {
  std::uint32_t stream = 0;  // an index into the lane's streams
  DeviceJob job;
};

struct DeviceCommand {
  OperationId operation;
  std::variant<DeviceWork, BackingWork, LaunchWork> work;
};

// CPU worker lane: bounded host work, such as verification, over memory
// the operation's lease protects. It returns its outcome and the bytes it
// covered; its access ends when it returns.
struct CpuResult {
  Outcome outcome = Outcome::kSucceeded;
  std::uint64_t bytes = 0;
};
using CpuJob = std::move_only_function<CpuResult()>;
struct CpuCommand {
  OperationId operation;
  CpuJob job;
};

}  // namespace llmp::scheduler

#endif  // LLMP_SCHEDULER_COMMANDS_H_
