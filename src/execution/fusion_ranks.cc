// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "execution/fusion_ranks.h"

#include <algorithm>
#include <cstdint>
#include <string_view>

#include "format/encoding.h"
#include "format/layout.h"

namespace llmp::execution {

FusionPreference ClassifyFusionLatency(std::uint64_t fused_ns, std::uint64_t decomposition_ns) {
  if (fused_ns == 0 || decomposition_ns == 0) return FusionPreference::kUnmeasured;
  // Difference versus floor(D/10) is exact for integer nanoseconds and avoids
  // overflow even at uint64 max. Both directions use the reference latency D.
  if (fused_ns > decomposition_ns && fused_ns - decomposition_ns > decomposition_ns / 10)
    return FusionPreference::kDecomposition;
  if (decomposition_ns > fused_ns && decomposition_ns - fused_ns > decomposition_ns / 10)
    return FusionPreference::kFusion;
  return FusionPreference::kNeutral;
}

bool ValidFusionRank(const FusionRank& r) {
  return !r.device_class.empty() && r.op.op == ir::Op::kFused &&
         r.op.fused != ir::FusedKind::kNone && !r.implementation.empty() && r.version != 0 &&
         static_cast<std::size_t>(r.encoding) < format::Encodings().size() &&
         static_cast<std::size_t>(r.layout) < format::Layouts().size() && r.rows_min != 0 &&
         r.rows_min <= r.rows_max && r.width_min != 0 && r.width_min <= r.width_max &&
         r.fused_ns != 0 && r.decomposition_ns != 0 && r.samples != 0 && !r.provenance.empty();
}

FusionPreference FusionPreferenceFor(const Registry& registry, const CapabilityQuery& query,
                                     std::string_view implementation) {
  const auto& rows = registry.ranks().fusions;
  if (rows.empty() || query.op.op != ir::Op::kFused || query.device_class.empty())
    return FusionPreference::kUnmeasured;
  const auto operand = std::ranges::find(query.operands, std::uint8_t{0}, &OperandFormat::operand);
  if (operand == query.operands.end()) return FusionPreference::kUnmeasured;
  for (const FusionRank& r : rows) {
    if (r.op != query.op || r.implementation != implementation ||
        r.device_class != query.device_class || r.encoding != operand->encoding ||
        r.layout != operand->layout || query.rows < r.rows_min || query.rows > r.rows_max ||
        query.k < r.width_min || query.k > r.width_max)
      continue;
    const auto index = registry.Find(implementation);
    if (!index || registry.at(*index).version != r.version) return FusionPreference::kStale;
    return ClassifyFusionLatency(r.fused_ns, r.decomposition_ns);
  }
  return FusionPreference::kUnmeasured;
}

}  // namespace llmp::execution
