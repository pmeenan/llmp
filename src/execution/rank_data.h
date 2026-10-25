// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#ifndef LLMP_EXECUTION_RANK_DATA_H_
#define LLMP_EXECUTION_RANK_DATA_H_

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "base/sha256.h"
#include "format/encoding.h"
#include "format/layout.h"
#include "ir/op.h"

namespace llmp::execution {
struct CapabilityQuery;
// Checked-in measurements supplied by a backend. Every string is owned so
// registry construction can consume temporary or assembled rank data safely.
// Bounds are inclusive; zero K/N and an empty model are explicit wildcards.
// Unmeasured formats/devices/batches retain the ordinary ladder.
struct DeviceMeasuredRank {
  std::string device_class, device_model;
  std::uint32_t arch = 0, multiprocessors = 0;
  ir::OpKey op{};
  format::Encoding encoding = format::Encoding::kF32;
  format::Layout layout = format::Layout::kDense;
  std::uint64_t rows_min = 1, rows_max = 1, k = 0, n = 0;
  std::string implementation;
  std::uint32_t rank = 0;
  format::EncodingParams params{};
  std::uint8_t stacked = 1;
};
// Paired timings of a fused implementation and its own IR decomposition.
// The query key is an exact operand/parameter/device identity, not a width
// whitelist or an extrapolated interval. Implementation identities make stale
// measurements explicitly unmeasured after code/toolchain changes.
struct MeasuredImplementation {
  std::string name;
  base::Sha256Digest identity{};
};
struct FusionComparison {
  std::string shape_class;  // descriptive, never a selection predicate
  base::Sha256Digest query{};
  base::Sha256Digest selection_environment{};
  MeasuredImplementation fusion;
  std::vector<MeasuredImplementation> decomposition;
  std::uint64_t fused_ns = 0, decomposition_ns = 0;
  std::uint32_t samples = 0;
  std::string provenance;  // checked-in aggregate report and source receipt
};

enum class RecipeRunMode : std::uint8_t { kFull, kWarm };
struct RecipeArmIdentity {
  base::Sha256Digest planned_region{};
  base::Sha256Digest physical_recipe{};
  base::Sha256Digest backend_recipe{};
  base::Sha256Digest backend_build{};
  base::Sha256Digest backend_selection{};
  bool operator==(const RecipeArmIdentity&) const = default;
};
struct RecipeComparisonKey {
  std::uint32_t version = 1;
  std::uint32_t region_canonicalizer = 2;
  ir::OpKey discovery_operation{};
  RecipeRunMode mode = RecipeRunMode::kFull;
  base::Sha256Digest subject{};
  base::Sha256Digest boundary{};
  base::Sha256Digest policy{};
  base::Sha256Digest device{};
  base::Sha256Digest selection_environment{};
  base::Sha256Digest preparation_domain{};
  RecipeArmIdentity incumbent{};
  RecipeArmIdentity alternative{};
  bool operator==(const RecipeComparisonKey&) const = default;
};
enum class RecipeComparisonKind : std::uint8_t { kLatency, kRejectIncumbentQuality };
struct RecipeComparison {
  RecipeComparisonKey key{};
  std::uint64_t incumbent_ns = 0;
  std::uint64_t alternative_ns = 0;
  std::uint32_t samples = 0;
  base::Sha256Digest protocol{};
  base::Sha256Digest receipt{};
  std::string provenance;
  std::string shape_class;
  RecipeComparisonKind kind = RecipeComparisonKind::kLatency;
};
struct RankData {
  std::vector<DeviceMeasuredRank> records{};
  std::vector<FusionComparison> fusions{};
  std::vector<RecipeComparison> recipe_comparisons{};
};
std::optional<std::uint32_t> DeviceRank(const CapabilityQuery& query,
                                        std::string_view implementation,
                                        std::span<const DeviceMeasuredRank> table);
base::Sha256Digest RankDataDigest(const RankData& data);
}  // namespace llmp::execution
#endif  // LLMP_EXECUTION_RANK_DATA_H_
