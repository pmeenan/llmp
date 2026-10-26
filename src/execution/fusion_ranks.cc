// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "execution/fusion_ranks.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <format>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

#include "format/tensor_format.h"

namespace llmp::execution {
namespace {
void Field(base::Sha256& hash, std::string_view value) {
  std::array<char, std::numeric_limits<std::size_t>::digits10 + 2> size{};
  const auto [end, error] = std::to_chars(size.data(), size.data() + size.size() - 1, value.size());
  if (error != std::errc{}) return;
  *end = ':';
  hash.Update(std::string_view(size.data(), static_cast<std::size_t>(end - size.data()) + 1));
  hash.Update(value);
}
void FieldParts(base::Sha256& hash, std::string_view a, std::string_view b, std::string_view c) {
  const std::size_t size = a.size() + b.size() + c.size();
  std::array<char, std::numeric_limits<std::size_t>::digits10 + 2> prefix{};
  const auto [end, error] = std::to_chars(prefix.data(), prefix.data() + prefix.size() - 1, size);
  if (error != std::errc{}) return;
  *end = ':';
  hash.Update(std::string_view(prefix.data(), static_cast<std::size_t>(end - prefix.data()) + 1));
  hash.Update(a);
  hash.Update(b);
  hash.Update(c);
}
template <class T>
void AppendValue(char*& at, char* end, T value) {
  static_assert(std::is_integral_v<T>);
  if constexpr (std::is_same_v<T, bool>) {
    const std::string_view text = value ? "true" : "false";
    std::copy(text.begin(), text.end(), at);
    at += text.size();
  } else {
    const auto [next, error] = std::to_chars(at, end, value);
    if (error == std::errc{}) at = next;
  }
}
template <class T>
consteval std::size_t DecimalChars() {
  static_assert(std::is_integral_v<T>);
  if constexpr (std::is_same_v<T, bool>) {
    return 5;  // The longer spelling emitted by AppendValue is "false".
  } else if constexpr (std::is_signed_v<T>) {
    return std::numeric_limits<T>::digits10 + 2;  // Decimal digits, sign, and rounding carry.
  } else {
    return std::numeric_limits<T>::digits10 + 1;  // Decimal digits and rounding carry.
  }
}
template <class... T>
void NumberField(base::Sha256& hash, T... values) {
  static_assert(sizeof...(T) > 0);
  static_assert(sizeof...(T) <= 13);
  constexpr std::size_t capacity = (DecimalChars<T>() + ...) + sizeof...(T) - 1;
  std::array<char, capacity> text{};
  char* at = text.data();
  char* const end = text.data() + text.size();
  bool first = true;
  auto append = [&](auto value) {
    if (!first) *at++ = ':';
    first = false;
    AppendValue(at, end, value);
  };
  (append(values), ...);
  Field(hash, std::string_view(text.data(), static_cast<std::size_t>(at - text.data())));
}
void Digest(base::Sha256& hash, const base::Sha256Digest& value) {
  hash.Update(std::as_bytes(std::span(value)));
}
template <class T>
void Fixed(base::Sha256& hash, T value) {
  static_assert(std::is_unsigned_v<T>);
  std::array<std::uint8_t, sizeof(T)> bytes{};
  for (std::size_t i = 0; i < bytes.size(); ++i)
    bytes[i] = static_cast<std::uint8_t>(value >> (i * 8));
  hash.Update(std::as_bytes(std::span(bytes)));
}
bool Nonzero(const base::Sha256Digest& digest) {
  return std::ranges::any_of(digest, [](std::uint8_t byte) { return byte != 0; });
}
bool ValidRecipeOp(ir::OpKey op) {
  return ir::OpName(op.op) != "unknown" && ir::FusedKindName(op.fused) != "unknown" &&
         ((op.op == ir::Op::kFused) == (op.fused != ir::FusedKind::kNone));
}
bool ValidRecipeText(std::string_view text) {
  return !text.empty() &&
         std::ranges::all_of(text, [](unsigned char c) { return c >= 0x20 && c < 0x7f; });
}
void HashRecipeArm(base::Sha256& hash, const RecipeArmIdentity& arm) {
  Digest(hash, arm.planned_region);
  Digest(hash, arm.physical_recipe);
  Digest(hash, arm.backend_recipe);
  Digest(hash, arm.backend_build);
  Digest(hash, arm.backend_selection);
}
void HashRecipeKey(base::Sha256& hash, const RecipeComparisonKey& key) {
  Fixed(hash, key.version);
  Fixed(hash, key.region_canonicalizer);
  Fixed(hash, static_cast<std::uint16_t>(key.discovery_operation.op));
  Fixed(hash, static_cast<std::uint16_t>(key.discovery_operation.fused));
  Fixed(hash, static_cast<std::uint8_t>(key.mode));
  Digest(hash, key.subject);
  Digest(hash, key.boundary);
  Digest(hash, key.policy);
  Digest(hash, key.device);
  Digest(hash, key.selection_environment);
  Digest(hash, key.preparation_domain);
  HashRecipeArm(hash, key.incumbent);
  HashRecipeArm(hash, key.alternative);
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
  std::array<const OperandFormat*, 256> operands{};
  std::size_t operand_count = 0;
  std::uint8_t last_operand = 0;
  bool has_operand = false;
  for (const auto& o : q.operands) {
    if (o.stacked == 0 || !o.memory || !o.rank || *o.rank > format::kMaxDims ||
        std::ranges::contains(o.ne, 0U) ||
        static_cast<std::size_t>(o.encoding) >= format::Encodings().size() ||
        static_cast<std::size_t>(o.layout) >= format::Layouts().size())
      return {};
    if (operands[o.operand] != nullptr) return {};
    operands[o.operand] = &o;
    ++operand_count;
    last_operand = has_operand ? std::max(last_operand, o.operand) : o.operand;
    has_operand = true;
  }
  if (operands[0] == nullptr || !has_operand || last_operand < kOutputOperand) return {};
  base::Sha256 hash;
  Field(hash, "llmp-fusion-query-v2");
  FieldParts(hash, ir::OpName(q.op.op), ".", ir::FusedKindName(q.op.fused));
  Field(hash, q.device_class);
  Field(hash, q.device_model);
  NumberField(hash, q.arch, q.features, q.multiprocessors, q.rows, q.k, q.n, q.row_invariant,
              q.capturable, q.state_span);
  NumberField(hash, q.policy.max_decode_rows, q.policy.allow_lossy, q.policy.vector_rows);
  // Keep each namespace separate. Explicit rank order is semantic when a
  // caller supplies repeated names (Candidates uses the first matching rank).
  Field(hash, q.policy.options ? "policy-options-present" : "policy-options-absent");
  if (q.policy.options) {
    NumberField(hash, q.policy.options->size());
    for (const auto value : *q.policy.options) Field(hash, value);
  }
  NumberField(hash, q.preferred.size());
  for (const auto& value : q.preferred) Field(hash, value);
  NumberField(hash, q.measured.size());
  for (const auto& r : q.measured) {
    Field(hash, r.implementation);
    NumberField(hash, r.rank);
  }
  NumberField(hash, q.tuned.size());
  for (const auto& t : q.tuned) {
    Field(hash, t.implementation);
    NumberField(hash, t.rows, t.k, t.n, t.weight_bits,
                t.output_encoding ? static_cast<int>(*t.output_encoding) : -1, t.all_rows);
  }
  NumberField(hash, q.params.size());
  hash.Update(q.params);
  NumberField(hash, operand_count);
  for (const auto* operand : operands) {
    if (operand == nullptr) continue;
    const auto& o = *operand;
    NumberField(hash, o.operand, o.stacked, static_cast<unsigned>(o.encoding),
                static_cast<unsigned>(o.layout), o.params.k_bits,
                static_cast<unsigned>(o.params.codebook), o.params.tensor_scale, o.params.bits,
                o.params.group, static_cast<unsigned>(o.params.scale), o.params.zero_point,
                o.params.block_rows, o.params.block_cols);
    NumberField(hash, *o.rank);
    for (auto n : o.ne) NumberField(hash, n);
    for (auto n : o.memory->nb) NumberField(hash, n);
    NumberField(hash, o.memory->view_offset, static_cast<std::uint64_t>(o.memory->root_alignment),
                static_cast<bool>(o.memory->is_view));
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
  if (!query.fusion_comparisons || !registry.HasFusionComparisons(query.op)) return {};
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
  if (!query.fusion_comparisons || !registry.HasFusionComparisons(query.op))
    return FusionPreference::kUnmeasured;
  const auto key = FusionQueryKey(query);
  if (!key) return FusionPreference::kUnmeasured;
  for (const auto& entry : registry.FusionComparisons(query.op, *key)) {
    const auto& record = *entry.comparison;
    if (record.fusion.name == implementation) return Preference(registry, query, record);
  }
  return FusionPreference::kUnmeasured;
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

base::Sha256Digest RecipeComparisonKeyDigest(const RecipeComparisonKey& key) {
  base::Sha256 hash;
  hash.Update("llmp-region-recipe-key-v1");
  HashRecipeKey(hash, key);
  return hash.Finish();
}

base::Sha256Digest RecipeComparisonsDigest(std::span<const RecipeComparison> comparisons) {
  std::vector<base::Sha256Digest> records;
  records.reserve(comparisons.size());
  for (const auto& comparison : comparisons) {
    base::Sha256 hash;
    hash.Update("llmp-region-recipe-record-v2");
    HashRecipeKey(hash, comparison.key);
    Fixed(hash, static_cast<std::uint8_t>(comparison.kind));
    Fixed(hash, comparison.incumbent_ns);
    Fixed(hash, comparison.alternative_ns);
    Fixed(hash, comparison.samples);
    Digest(hash, comparison.protocol);
    Digest(hash, comparison.receipt);
    Field(hash, comparison.provenance);
    Field(hash, comparison.shape_class);
    records.push_back(hash.Finish());
  }
  std::ranges::sort(records);
  base::Sha256 hash;
  hash.Update("llmp-region-recipe-comparisons-v2");
  Fixed(hash, static_cast<std::uint64_t>(records.size()));
  for (const auto& record : records) Digest(hash, record);
  return hash.Finish();
}

std::expected<void, std::string> ValidateRecipeComparison(const RecipeComparison& comparison) {
  const auto& key = comparison.key;
  if (key.version != 1 || key.region_canonicalizer != 2 ||
      !ValidRecipeOp(key.discovery_operation) ||
      static_cast<std::uint8_t>(key.mode) > static_cast<std::uint8_t>(RecipeRunMode::kWarm))
    return std::unexpected("unknown recipe comparison key version, operation, or run mode");
  if (!Nonzero(key.subject) || !Nonzero(key.boundary) || !Nonzero(key.policy) ||
      !Nonzero(key.device) || !Nonzero(key.selection_environment) ||
      !Nonzero(key.preparation_domain) || !Nonzero(key.incumbent.planned_region) ||
      !Nonzero(key.incumbent.physical_recipe) || !Nonzero(key.incumbent.backend_recipe) ||
      !Nonzero(key.incumbent.backend_build) || !Nonzero(key.incumbent.backend_selection) ||
      !Nonzero(key.alternative.planned_region) || !Nonzero(key.alternative.physical_recipe) ||
      !Nonzero(key.alternative.backend_recipe) || !Nonzero(key.alternative.backend_build) ||
      !Nonzero(key.alternative.backend_selection) || !Nonzero(comparison.protocol) ||
      !Nonzero(comparison.receipt))
    return std::unexpected("recipe comparison is missing a required nonzero identity");
  if (key.incumbent == key.alternative)
    return std::unexpected("recipe comparison arms have identical complete identities");
  switch (comparison.kind) {
    case RecipeComparisonKind::kLatency:
      if (comparison.incumbent_ns == 0 || comparison.alternative_ns == 0 || comparison.samples == 0)
        return std::unexpected("recipe comparison timing and sample counts must be positive");
      break;
    case RecipeComparisonKind::kRejectIncumbentQuality:
      if (comparison.incumbent_ns != 0 || comparison.alternative_ns != 0 || comparison.samples != 0)
        return std::unexpected("quality rejection must not contain timing or samples");
      break;
    default:
      return std::unexpected("unknown recipe comparison kind");
  }
  if (!ValidRecipeText(comparison.provenance) || comparison.provenance == "UNREVIEWED" ||
      !ValidRecipeText(comparison.shape_class))
    return std::unexpected("recipe comparison provenance or shape class is invalid");
  return {};
}

RecipePreference RecipeComparisonPreference(const RecipeComparison& comparison) {
  if (comparison.kind == RecipeComparisonKind::kRejectIncumbentQuality)
    return RecipePreference::kAlternative;
  return ClassifyRecipeLatency(comparison.incumbent_ns, comparison.alternative_ns);
}

RecipePreference ClassifyRecipeLatency(std::uint64_t incumbent_ns, std::uint64_t alternative_ns) {
  switch (ClassifyFusionLatency(incumbent_ns, alternative_ns)) {
    case FusionPreference::kDecomposition:
      return RecipePreference::kAlternative;
    case FusionPreference::kFusion:
      return RecipePreference::kIncumbent;
    case FusionPreference::kNeutral:
      return RecipePreference::kNeutral;
    case FusionPreference::kUnmeasured:
    case FusionPreference::kStale:
      return RecipePreference::kNeutral;
  }
  return RecipePreference::kNeutral;
}
}  // namespace llmp::execution
