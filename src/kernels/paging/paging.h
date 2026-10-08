// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Llmpalooza's own small CUDA kernels that the engine's device jobs queue
// around the paging of weights and rows (engine/paged_weights.h,
// engine/ple_rows.h): filling device ranges, gathering Qwen3.8's n-gram
// rows from their pinned landing, and bounding draft token ids before they
// index a table. Data movement and guards, not operations of
// a model's plan, so they are not registry-bound (D-053). A backend
// without them fills and gathers with its own (docs/portability.md).
//
// `stream` is the job's NativeStream handle (providers/device_execution.h),
// which only these kernels unwrap (a cudaStream_t). Launches only queue
// work there; completion is the caller's. Each returns false if the launch
// failed, reading (and so clearing) the thread's last CUDA error as llmpalooza's
// other launchers do, so a failure is not left for a later check to report.
// CUDA builds only.

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
bool GatherPleRows(const std::byte* landing, const std::uint32_t* sources,
                   const std::uint32_t* count, std::uint32_t max_count, std::uint32_t row_bytes,
                   std::byte* slots, void* stream);

// Replaces each of `count` device token ids outside [0, limit) with 0 in
// place: a drafter's argmax over non-finite logits gives -1, which would
// index past a token table (engine/dsv4_runner.cc, the drafts' rows).
bool ClampTokens(std::int32_t* tokens, std::uint32_t count, std::int32_t limit, void* stream);

}  // namespace llmp::kernels::paging

#endif  // LLMP_KERNELS_PAGING_PAGING_H_
