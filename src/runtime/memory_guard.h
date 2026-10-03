// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// The memory guard at a serving start (docs/runtime-serving.md): after each
// model's own memory is mapped, the largest model's weights, the largest
// chunk inputs a model builds on the host, the most plans one step of a
// model holds at once and a margin for what the catalog does not count
// otherwise must fit the memory then available, or the runtime refuses to
// serve. Vendor-free, so the CPU tests hold its arithmetic and its refusal.
//
// - The margin (kUncountedMargin): what the catalog and the terms below do
//   not count, the CUDA context, the driver's and cuBLAS's own allocations
//   among it. Its 6 GiB rests on the uncounted memory measured as peak
//   minus budget beside the models' host-built inputs, 4.6-5.4 GiB
//   (docs/experiments/long-context, "Memory and the guard's margin"). It
//   is meant to be consumed by what it covers, not set apart again for
//   anything the catalog already charges (a state snapshot's staging).
// - The host-built chunk inputs: what each model's chunks build on the host
//   before staging them (bounded by the staging they are sized for), which
//   grows with a model's context and chunk and so is counted on its own,
//   one model's (one model runs chunks at a time, so the most of any),
//   beside the margin rather than in it.
// - The plans: what one step of a model holds of plans at once at most
//   (Served::plan_floor_bytes, engine/planned.h), the largest model's.
//   Every plan and graph past it is charged inside the budget and given
//   back through the reclaim order (D-090 as amended 2026-10-02), so the
//   guard sets apart only this in-use minimum, not a worst case.
// - The request memory's floor (intake_limits.h): what requests hold in
//   host memory (bodies, parses, renderings, unread output) is one pool
//   charged by real size (D-102); its first 256 MiB are set apart here,
//   and what passes them is charged inside the budget as requests grow,
//   through the reclaim order, like any other need.

#ifndef JITLLM_RUNTIME_MEMORY_GUARD_H_
#define JITLLM_RUNTIME_MEMORY_GUARD_H_

#include <cstdint>
#include <expected>
#include <string>

namespace jitllm::runtime {

inline constexpr std::uint64_t kUncountedMargin = std::uint64_t{6} << 30U;

struct MemoryGuard {
  std::uint64_t largest = 0;      // the largest model's weights
  std::uint64_t host_inputs = 0;  // the most any model's chunk builds on the host
  std::uint64_t plans = 0;        // the most one step of a model holds of plans
  std::uint64_t available = 0;  // MemAvailable after the models' own memory was mapped; 0: unknown
  std::uint64_t fixed = 0;      // the node's mapped memory (reported in the refusal)
  // The request memory's floor (intake_limits.h): requests' host memory set
  // apart beside the margin; what passes it is charged inside the budget.
  std::uint64_t requests = 0;
};

// Everything the guard sets apart from the available memory beside the
// weights: the host inputs, the plans, the request memory and
// kUncountedMargin.
std::uint64_t GuardReserve(const MemoryGuard& guard);

// Refused, saying what did not fit, unless largest + host_inputs + plans +
// requests + kUncountedMargin fit `available`; an unknown `available` (0)
// passes.
std::expected<void, std::string> CheckMemoryGuard(const MemoryGuard& guard);

}  // namespace jitllm::runtime

#endif  // JITLLM_RUNTIME_MEMORY_GUARD_H_
