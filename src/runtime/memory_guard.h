// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// The memory guard at a serving start (docs/runtime-serving.md): after each
// model's own memory is mapped, the largest model's weights, the largest
// chunk inputs a model builds on the host, the largest model's bounded plans
// and graphs and a margin for what the catalog does not count otherwise
// must fit the memory then available, or the runtime refuses to serve.
// Vendor-free, so the CPU tests hold its arithmetic and its refusal.
//
// - The margin (kUncountedMargin): what the catalog and the terms below do
//   not count, the CUDA context, the driver's and cuBLAS's own allocations
//   among it. Its 6 GiB rests on the uncounted memory measured as peak
//   minus budget beside the models' host-built inputs, 4.6-5.4 GiB
//   (docs/experiments/long-context, "Memory and the guard's margin"). Those
//   runs also held their decode graphs, at most 8 a DeepSeek runner (each
//   ~20-25 MiB, docs/experiments/deepseek-batching, "Plan memory"), which
//   the plans term now counts too: at most ~0.2 GiB of the margin is
//   counted twice. It is kept until a measurement of the uncounted memory
//   without them says otherwise.
// - The host-built chunk inputs: what each model's chunks build on the host
//   before staging them (bounded by the staging they are sized for), which
//   grows with a model's context and chunk and so is counted on its own,
//   one model's (one model runs chunks at a time, so the most of any),
//   beside the margin rather than in it.
// - The plans: a model's cached plans and graphs, host and driver memory
//   bounded by its caps (Served::plan_host_bytes, engine/planned.h). Only
//   the resident model keeps them (a swap drops the outgoing model's), so
//   the largest model's, counted beside the margin so that no number of
//   cached shapes can consume it.

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
  std::uint64_t plans = 0;        // the largest model's plans' and graphs' bound
  std::uint64_t available = 0;  // MemAvailable after the models' own memory was mapped; 0: unknown
  std::uint64_t fixed = 0;      // the node's mapped memory (reported in the refusal)
};

// Everything the guard sets apart from the available memory beside the
// weights: the host inputs, the plans and kUncountedMargin.
std::uint64_t GuardReserve(const MemoryGuard& guard);

// Refused, saying what did not fit, unless largest + host_inputs + plans +
// kUncountedMargin fit `available`; an unknown `available` (0) passes.
std::expected<void, std::string> CheckMemoryGuard(const MemoryGuard& guard);

}  // namespace jitllm::runtime

#endif  // JITLLM_RUNTIME_MEMORY_GUARD_H_
