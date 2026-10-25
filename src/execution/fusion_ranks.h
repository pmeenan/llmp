// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#ifndef LLMP_EXECUTION_FUSION_RANKS_H_
#define LLMP_EXECUTION_FUSION_RANKS_H_

#include <expected>
#include <optional>
#include <string>

#include "execution/registry.h"

namespace llmp::execution {

// No extrapolation: all operands (including outputs), parameters, immutable
// device facts and selection requirements must match. Operand order is not
// significant; missing/duplicate/unknown facts never qualify a measurement.
std::optional<base::Sha256Digest> FusionQueryKey(const CapabilityQuery& query);

enum class FusionPreference { kUnmeasured, kStale, kNeutral, kFusion, kDecomposition };
// Symmetric latency band relative to the decomposition: both exact +/-10%
// endpoints are neutral. Zero timings are unmeasured. Shared with the harness.
FusionPreference ClassifyFusionLatency(std::uint64_t fused_ns, std::uint64_t decomposition_ns);
// Coarse D-085 decision: within 10% retains the incumbent. A >10% loss
// excludes the fusion, allowing the planner's existing own-decomposition
// fallback. This preference cannot grant numerical eligibility.
struct NamedFusionPreference {
  std::string_view implementation;
  FusionPreference preference;
};
// Computes one exact key for all measured alternatives of this operation.
std::vector<NamedFusionPreference> FusionPreferences(const Registry& registry,
                                                     const CapabilityQuery& query);
FusionPreference FusionPreferenceFor(const Registry& registry, const CapabilityQuery& query,
                                     std::string_view implementation);

// Canonical complete data identity, including evidence and measured source
// identities. Kept separate from the implementation source identity.
base::Sha256Digest FusionComparisonsDigest(std::span<const FusionComparison> comparisons);

// Exact local-region recipe identity. The key digest hashes typed fields
// without allocating; neither helper interprets caller labels as authority.
base::Sha256Digest RecipeComparisonKeyDigest(const RecipeComparisonKey& key);
base::Sha256Digest RecipeComparisonsDigest(std::span<const RecipeComparison> comparisons);
std::expected<void, std::string> ValidateRecipeComparison(const RecipeComparison& comparison);
enum class RecipePreference : std::uint8_t { kIncumbent, kAlternative, kNeutral };
RecipePreference ClassifyRecipeLatency(std::uint64_t incumbent_ns, std::uint64_t alternative_ns);
// Requires a validated registry record; quality rejection has no latency meaning.
RecipePreference RecipeComparisonPreference(const RecipeComparison& comparison);

}  // namespace llmp::execution
#endif  // LLMP_EXECUTION_FUSION_RANKS_H_
