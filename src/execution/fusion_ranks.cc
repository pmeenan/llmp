// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "execution/fusion_ranks.h"

#include <algorithm>
#include <array>
#include <format>
#include <string>
#include <vector>

#include "format/tensor_format.h"

namespace llmp::execution {
namespace {
void Field(base::Sha256& hash, std::string_view value) {
  hash.Update(std::format("{}:", value.size()));
  hash.Update(value);
}
void Digest(base::Sha256& hash, const base::Sha256Digest& value) {
  hash.Update(std::as_bytes(std::span(value)));
}
void HashImplementation(base::Sha256& hash, const MeasuredImplementation& value) {
  Field(hash, value.name);
  Digest(hash, value.identity);
}
}  // namespace

std::optional<base::Sha256Digest> FusionQueryKey(const CapabilityQuery& q) {
  if (q.op.op != ir::Op::kFused || q.op.fused == ir::FusedKind::kNone || q.device_class.empty() ||
      q.device_model.empty() || q.arch == 0 || q.multiprocessors == 0 || q.operands.empty())
    return {};
  std::vector<const OperandFormat*> operands;
  for (const auto& o : q.operands) {
    if (o.stacked == 0 || !o.memory || !o.rank || *o.rank > format::kMaxDims ||
        std::ranges::contains(o.ne, 0U) ||
        static_cast<std::size_t>(o.encoding) >= format::Encodings().size() ||
        static_cast<std::size_t>(o.layout) >= format::Layouts().size())
      return {};
    operands.push_back(&o);
  }
  std::ranges::sort(operands, {}, [](const auto* o) { return o->operand; });
  if (std::ranges::adjacent_find(
          operands, [](const auto* a, const auto* b) { return a->operand == b->operand; }) !=
          operands.end() ||
      operands.front()->operand != 0 || operands.back()->operand < kOutputOperand)
    return {};
  base::Sha256 hash;
  Field(hash, "llmp-fusion-query-v2");
  Field(hash, ir::OpKeyName(q.op));
  Field(hash, q.device_class);
  Field(hash, q.device_model);
  Field(hash, std::format("{}:{}:{}:{}:{}:{}:{}:{}:{}", q.arch, q.features, q.multiprocessors,
                          q.rows, q.k, q.n, q.row_invariant, q.capturable, q.state_span));
  Field(hash, std::format("{}:{}:{}", q.policy.max_decode_rows, q.policy.allow_lossy,
                          q.policy.vector_rows));
  // Keep each namespace separate. Explicit rank order is semantic when a
  // caller supplies repeated names (Candidates uses the first matching rank).
  Field(hash, q.policy.options ? "policy-options-present" : "policy-options-absent");
  if (q.policy.options) {
    Field(hash, std::format("{}", q.policy.options->size()));
    for (const auto value : *q.policy.options) Field(hash, value);
  }
  Field(hash, std::format("{}", q.preferred.size()));
  for (const auto& value : q.preferred) Field(hash, value);
  Field(hash, std::format("{}", q.measured.size()));
  for (const auto& r : q.measured) {
    Field(hash, r.implementation);
    Field(hash, std::format("{}", r.rank));
  }
  Field(hash, std::format("{}", q.tuned.size()));
  for (const auto& t : q.tuned) {
    Field(hash, t.implementation);
    Field(hash,
          std::format("{}:{}:{}:{}:{}:{}", t.rows, t.k, t.n, t.weight_bits,
                      t.output_encoding ? static_cast<int>(*t.output_encoding) : -1, t.all_rows));
  }
  Field(hash, std::format("{}", q.params.size()));
  hash.Update(q.params);
  Field(hash, std::format("{}", operands.size()));
  for (const auto* operand : operands) {
    const auto& o = *operand;
    Field(hash, std::format("{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}", o.operand, o.stacked,
                            static_cast<unsigned>(o.encoding), static_cast<unsigned>(o.layout),
                            o.params.k_bits, static_cast<unsigned>(o.params.codebook),
                            o.params.tensor_scale, o.params.bits, o.params.group,
                            static_cast<unsigned>(o.params.scale), o.params.zero_point));
    Field(hash, std::format("{}", *o.rank));
    for (auto n : o.ne) Field(hash, std::format("{}", n));
    for (auto n : o.memory->nb) Field(hash, std::format("{}", n));
    Field(hash, std::format("{}:{}:{}", o.memory->view_offset,
                            static_cast<std::uint64_t>(o.memory->root_alignment),
                            static_cast<bool>(o.memory->is_view)));
  }
  return hash.Finish();
}

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

namespace {
FusionPreference Preference(const Registry& registry, const CapabilityQuery& query,
                            const FusionComparison& r) {
  if (r.selection_environment != registry.SelectionDigest()) return FusionPreference::kStale;
  if (query.terminal_decomposition.empty()) return FusionPreference::kUnmeasured;
  if (query.terminal_decomposition.size() != r.decomposition.size())
    return FusionPreference::kStale;
  for (std::size_t i = 0; i < r.decomposition.size(); ++i) {
    const auto index = query.terminal_decomposition[i];
    if (index >= registry.size() || registry.at(index).name != r.decomposition[i].name ||
        registry.identity(index) != r.decomposition[i].identity)
      return FusionPreference::kStale;
  }
  const auto current = [&](const MeasuredImplementation& m) {
    const auto index = registry.Find(m.name);
    return index && registry.identity(*index) == m.identity;
  };
  if (!current(r.fusion) || !std::ranges::all_of(r.decomposition, current))
    return FusionPreference::kStale;
  return ClassifyFusionLatency(r.fused_ns, r.decomposition_ns);
}
}  // namespace

std::vector<NamedFusionPreference> FusionPreferences(const Registry& registry,
                                                     const CapabilityQuery& query) {
  if (!registry.HasFusionComparisons(query.op)) return {};
  const auto key = FusionQueryKey(query);
  if (!key) return {};
  std::vector<NamedFusionPreference> preferences;
  for (const auto& entry : registry.FusionComparisons(query.op, *key)) {
    const auto& record = *entry.comparison;
    preferences.push_back({record.fusion.name, Preference(registry, query, record)});
  }
  return preferences;
}
FusionPreference FusionPreferenceFor(const Registry& registry, const CapabilityQuery& query,
                                     std::string_view implementation) {
  const auto preferences = FusionPreferences(registry, query);
  const auto found =
      std::ranges::find(preferences, implementation, &NamedFusionPreference::implementation);
  return found == preferences.end() ? FusionPreference::kUnmeasured : found->preference;
}

base::Sha256Digest FusionComparisonsDigest(std::span<const FusionComparison> comparisons) {
  std::vector<base::Sha256Digest> records;
  for (const auto& r : comparisons) {
    base::Sha256 hash;
    Field(hash, r.shape_class);
    Digest(hash, r.query);
    Digest(hash, r.selection_environment);
    HashImplementation(hash, r.fusion);
    Field(hash, std::format("{}", r.decomposition.size()));
    for (const auto& implementation : r.decomposition) HashImplementation(hash, implementation);
    Field(hash, std::format("{}:{}:{}", r.fused_ns, r.decomposition_ns, r.samples));
    Field(hash, r.provenance);
    records.push_back(hash.Finish());
  }
  std::ranges::sort(records);
  base::Sha256 hash;
  Field(hash, "llmp-fusion-comparisons-v2");
  for (const auto& record : records) Digest(hash, record);
  return hash.Finish();
}
}  // namespace llmp::execution
