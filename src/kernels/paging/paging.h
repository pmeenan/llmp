// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Llmpalooza's own small kernels that the engine's device jobs queue
// around the paging of weights, rows and state (engine/paged_weights.h,
// engine/ple_rows.h, engine/live_state.h): filling device ranges,
// gathering Qwen3.8's n-gram rows from their pinned landing, bounding draft
// token ids before they index a table, and copying a verify's snapshot
// ranges. Data movement and guards, not operations of a model's plan, so
// they are not registry-bound (D-053). A backend without them fills and
// gathers with its own (docs/portability.md).
//
// Two implementations, one per device backend, chosen when a program links
// (llmp_paging_kernels or llmp_host_paging_kernels): CUDA kernels
// (`*.cu`), and host loops queued on the fake device runtime's streams
// (host_paging.cc), which run when the fake runs that stream's work, as a
// device would. `stream` is the job's NativeStream handle
// (providers/device_execution.h), which only these implementations unwrap
// (a cudaStream_t, or a fake stream's token). Launches only queue work
// there; completion is the caller's. Each returns false if the launch
// failed, reading (and so clearing) the thread's last device error as
// llmpalooza's other launchers do, so a failure is not left for a later
// check to report.

#ifndef LLMP_KERNELS_PAGING_PAGING_H_
#define LLMP_KERNELS_PAGING_PAGING_H_

#include <cstddef>
#include <cstdint>

namespace llmp::kernels::paging {

// Fills each of `count` device ranges (`ranges` holds address and length
// pairs, in pinned host memory the device reads) with `value`.
bool FillRanges(const std::uint64_t* ranges, std::uint32_t count, std::uint8_t value, void* stream);

// Row slot i's `row_bytes` bytes from landing + sources[i], for
// i < *count, over a grid of `max_count` slots (at least *count). `count`,
// `sources` and `landing` are pinned host memory the device reads in
// place, so a captured graph replays the gather with whatever the host
// wrote there for the next chunk (D-090: data, not launch parameters);
// `slots` is device memory.
bool GatherPleRows(const std::byte* landing, const std::uint64_t* sources,
                   const std::uint32_t* count, std::uint32_t max_count, std::uint32_t row_bytes,
                   std::byte* slots, void* stream);

// Replaces each of `count` device token ids outside [0, limit) with 0 in
// place: a drafter's argmax over non-finite logits gives -1, which would
// index past a token table (the model runner's drafts' rows).
bool ClampTokens(std::int32_t* tokens, std::uint32_t count, std::int32_t limit, void* stream);

// Byte ranges copied device to device in one launch, not a graph node: a
// speculative verify's snapshot of the state rows it will write, and the
// restore of the rows a rejected draft wrote (D-068's truncation at a
// snapshot; engine/live_state.h). Each range is 16-byte aligned at both
// ends and a multiple of 16 bytes (RangeCopiesValid); no destination
// overlaps another range. `ranges` is memory the device reads in place
// (pinned host memory, or device memory), valid until the copy completes;
// at most kMaxRangeCopies.
struct RangeCopy {
  std::uint64_t from = 0;
  std::uint64_t to = 0;
  std::uint64_t bytes = 0;
};
inline constexpr std::uint32_t kMaxRangeCopies = 65535;

// Whether CopyRanges takes these: none at all, or at most kMaxRangeCopies,
// each with nonzero addresses, both 16-byte aligned, and a multiple of 16
// bytes. Read on the host before the launch.
inline bool RangeCopiesValid(const RangeCopy* ranges, std::uint32_t count) {
  if (count == 0) {
    return true;
  }
  if (ranges == nullptr || count > kMaxRangeCopies) {
    return false;
  }
  for (std::uint32_t i = 0; i < count; ++i) {
    const RangeCopy& r = ranges[i];
    if (r.from == 0 || r.to == 0 || r.from % 16 != 0 || r.to % 16 != 0 || r.bytes % 16 != 0) {
      return false;
    }
  }
  return true;
}

// Queues the `count` copies (RangeCopiesValid's ranges; nothing for none).
bool CopyRanges(const RangeCopy* ranges, std::uint32_t count, void* stream);

}  // namespace llmp::kernels::paging

#endif  // LLMP_KERNELS_PAGING_PAGING_H_
