// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The memory guard at a serving start (docs/runtime-serving.md): after each
// model's own memory is mapped, the largest model's closure (its weights and
// its own prefix of the pageable shared workspace), the largest
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
//   (Served::plan_floor_bytes, engine/plan_account.h), the largest model's.
//   Every plan and graph past it is charged inside the budget and given
//   back through the reclaim order (D-090 as amended 2026-10-02), so the
//   guard sets apart only this in-use minimum, not a worst case.
// - The request memory's floor (intake_limits.h): what requests hold in
//   host memory (bodies, parses, renderings, unread output) is one pool
//   charged by real size (D-102); its first 256 MiB are set apart here,
//   and what passes them is charged inside the budget as requests grow,
//   through the reclaim order, like any other need.

#ifndef LLMP_RUNTIME_MEMORY_GUARD_H_
#define LLMP_RUNTIME_MEMORY_GUARD_H_

#include <cstdint>
#include <expected>
#include <string>

namespace llmp::runtime {

inline constexpr std::uint64_t kUncountedMargin = std::uint64_t{6} << 30U;

// What one model's closure holds resident while it is the active model (one
// model is resident at a time, M3's full swap): its weights and its own
// prefix of each region of the shared workspace, which pages with it
// (engine/paged_node.h MapWorkspace), each region in whole 2 MiB extents and
// at least one. Never what another model needs of the workspace: that is
// evicted while this one runs. Saturates rather than wraps.
std::uint64_t ActiveFootprint(std::uint64_t weights, std::uint64_t activations, std::uint64_t pool);

struct MemoryGuard {
  // The largest active footprint of any model (ActiveFootprint): its weights
  // and its workspace.
  std::uint64_t largest = 0;
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

// Initial metadata loan, independent of a file's observed size: the existing
// readers enforce their caps on opened descriptors even if files grow. At most
// target+drafter parse simultaneously; aliases resolve and set up sequentially.
// The server replaces this loan only after all parser owners have died.
std::expected<std::uint64_t, std::string> StartupBootstrapBytes(
    std::uint64_t aliases, std::uint64_t configuration_bytes = 0);

// An ordinary startup's budget (no LLMP_DIAGNOSTIC_* ceiling), from the
// node's real resources only (D-102): the parser loan the configured
// aliases' readers are bounded by (StartupBootstrapBytes) and the execution
// ceiling construction is admitted against before Start, the measured
// physical availability less kUncountedMargin in whole 2 MiB extents.
// Refused when availability is unknown (0) or leaves nothing past the
// margin. Once the node runs, its execution budget bounds growth as well.
struct StartupBudget {
  std::uint64_t parser_loan = 0;
  std::uint64_t cap = 0;
};
// That ceiling alone: `available` less kUncountedMargin in whole 2 MiB
// extents; 0 when availability is unknown or leaves no extent.
std::uint64_t MeasuredStartupCeiling(std::uint64_t available);
std::expected<StartupBudget, std::string> OrdinaryStartupBudget(std::uint64_t aliases,
                                                                std::uint64_t available);

// A default-off startup screen prepays host metadata before Setup. Unknown
// physical availability, unaligned/overflowed allowances and a cap that cannot
// leave the ordinary driver margin all refuse before opening a node.
std::expected<void, std::string> CheckStartupHostAllowance(std::uint64_t bytes, std::uint64_t cap,
                                                           std::uint64_t available);

// Internal matched controls can reduce the ordinary dynamic catalog budget,
// never raise it or bypass known physical availability. Include pinned memory
// charged after the startup guard, such as the backing handle reserve.
std::expected<void, std::string> CheckDiagnosticBudgetCap(const MemoryGuard& guard,
                                                          std::uint64_t dynamic_budget,
                                                          std::uint64_t cap,
                                                          std::uint64_t extra_pinned = 0);

}  // namespace llmp::runtime

#endif  // LLMP_RUNTIME_MEMORY_GUARD_H_
