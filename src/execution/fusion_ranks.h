// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#ifndef LLMP_EXECUTION_FUSION_RANKS_H_
#define LLMP_EXECUTION_FUSION_RANKS_H_

#include <cstdint>
#include <string_view>

#include "execution/registry.h"

namespace llmp::execution {

enum class FusionPreference { kUnmeasured, kStale, kNeutral, kFusion, kDecomposition };
// Symmetric latency band relative to the decomposition: both exact +/-5%
// endpoints are neutral (the faster wherever a gap is beyond the
// measurements' repeat noise: docs/experiments/vecq-columns/README.md).
// Zero timings are unmeasured. Shared with the harness.
FusionPreference ClassifyFusionLatency(std::uint64_t fused_ns, std::uint64_t decomposition_ns);

// The backend's coarse measured preference (rank_data.h FusionRank) for this
// fused implementation computing this query: the first row whose device
// class, operation, implementation, operand 0 encoding and layout match and
// whose row, width and output bands hold the query's rows, k and n. A coarse
// decision: within 5% keeps the incumbent; a >5% loss excludes the fusion, so
// the planner falls back to its own decomposition; a >5% win qualifies a
// fusion that needs measurement. It never grants numerical eligibility. A
// matching row measured at another kernel version is kStale and decides
// nothing.
FusionPreference FusionPreferenceFor(const Registry& registry, const CapabilityQuery& query,
                                     std::string_view implementation);

// Whether a row's bands and identity are well formed (positive timings and
// samples, ordered bands, a fused operation, provenance). Registry::Create
// refuses malformed rows; it does not require the named implementation to
// exist in a build (the stale-row test checks the shipped table).
bool ValidFusionRank(const FusionRank& rank);

}  // namespace llmp::execution
#endif  // LLMP_EXECUTION_FUSION_RANKS_H_
