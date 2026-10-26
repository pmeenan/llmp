// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The fake device runtime (fake_device_runtime.cc): providers/device_runtime.h
// over host memory, so the engine core runs, and is tested, in every build
// profile without a GPU (D-096 as amended for M3.6). A program links it
// (llmp_fake_device_runtime) instead of a device backend's runtime, with the
// host paging kernels (kernels/paging/host_paging.cc); the two are never in
// one program.
//
// OpenDevice opens the fakes: device memory over anonymous host memory at
// 2 MiB with a device and a host class (FakeDeviceMemory, poisoned), and
// execution whose queued work runs whenever a fence is queried
// (FakeDeviceExecution::Progress::kOnQuery). Copies, fills, timing marks
// and replays are host work queued on the job's stream: they run when the
// fake runs that stream, after what was queued before, and a consumer that
// reads before its fence has been seen complete reads what was there
// before. Nothing completes because a call returned, or because an object
// was destroyed. Recorded work is the stream's queued work captured as a
// list of host work, replayed whole. Pinned memory is plain host memory,
// 256-byte aligned. The device's architecture number is 0, its memory the
// configured capacity, and the thread's error state the last failed call's.
//
// Not a performance model: it proves the engine's logic, ordering and
// completion handling, not timing.

#ifndef LLMP_PROVIDERS_FAKE_FAKE_DEVICE_RUNTIME_H_
#define LLMP_PROVIDERS_FAKE_FAKE_DEVICE_RUNTIME_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>

#include "base/bytes.h"
#include "providers/device_runtime.h"

namespace llmp::providers::fake {

// The fake runtime's error codes (DeviceStatus::code; 0 is success).
enum class RuntimeError : int {
  kInvalidStream = 1,  // no fake stream has the handle
  kInvalidValue = 2,   // a null pointer, an unknown kind, a mark not made here
  kRecording = 3,      // a recording misused: begun twice, ended unbegun
  kNotReady = 4,       // a timing mark not yet run
  kNoMemory = 5,       // pinned memory not allocated
  kNoDevice = 6,       // an ordinal other than 0
};

struct FakeDeviceSettings {
  // What the fake device memory may back at once, across its classes.
  base::Bytes capacity = base::Bytes(std::uint64_t{4} << 30U);
  // Whether QueryDeviceFacts reports the device's memory as the host's
  // (DeviceFacts::unified), and what QueryDeviceMemory reports free (none:
  // the capacity).
  bool unified = true;
  std::optional<base::Bytes> free = std::nullopt;
};
// The settings the next OpenDevice uses (before it; not thread-safe with it).
void ConfigureFakeDevice(FakeDeviceSettings settings);
// One matching runtime copy fails. With queue_first its actual copy is
// queued before the failure, so callers must retain staging until proven
// retirement. Used by state-control publication/hydration fault tests.
// matches_before lets a test fail a later member of one publication batch.
void FailNextCopy(CopyKind kind, std::size_t bytes, bool queue_first = false,
                  std::size_t matches_before = 0);
void ClearCopyFailure();
// One exact-pointer free invokes this callback after physically freeing its
// allocation. The hook is taken before the free and called without its mutex,
// so a test can fault subsequent bookkeeping deterministically. No callback
// receives or may access the freed bytes. Clear it before referenced test
// state retires if the matching free did not happen.
void AfterNextPinnedFree(void* pointer, std::function<void()> callback);
void ClearPinnedFreeHook();
// The next matching allocation is 256-aligned but deliberately not 4 KiB
// aligned. FreePinned still frees its real backing; no production guarantee
// beyond the ordinary fake's 256-byte alignment is changed.
void MisalignNextPinned(std::size_t bytes);
void ClearPinnedMisalignment();

}  // namespace llmp::providers::fake

#endif  // LLMP_PROVIDERS_FAKE_FAKE_DEVICE_RUNTIME_H_
