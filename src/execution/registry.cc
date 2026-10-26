// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "execution/registry.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>
#include <format>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "base/check.h"
#include "base/sha256.h"
#include "execution/fusion_ranks.h"
#include "execution/rank_data.h"
#include "format/encoding.h"
#include "format/layout.h"
#include "ir/decompose.h"
#include "ir/op.h"

namespace llmp::execution {
namespace {

std::unexpected<PlanRejection> Rejected(PlanError error, std::size_t operation,
                                        std::string detail) {
  return std::unexpected(
      PlanRejection{.error = error, .operation = operation, .detail = std::move(detail)});
}

// Fields are length-prefixed, so no two sequences of fields hash alike by
// moving bytes from one field to the next.
void Field(base::Sha256& hash, std::string_view text) {
  std::array<std::uint8_t, 8> length{};
  for (std::size_t i = 0; i < length.size(); ++i) {
    length[i] = static_cast<std::uint8_t>(static_cast<std::uint64_t>(text.size()) >> (8 * i));
  }
  hash.Update(std::as_bytes(std::span(length)));
  hash.Update(text);
}

bool ValidName(std::string_view name) {
  return !name.empty() && name.size() <= Registry::kMaxName &&
         std::ranges::all_of(name, [](char c) {
           return (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '.' || c == '_';
         });
}

// The registry index of the implementation a plan names for its operation
// at `position`: never another implementation, whatever the registry has.
std::expected<std::size_t, PlanRejection> Lookup(const Registry& registry,
                                                 llmp::ir::OpKey operation, std::string_view name,
                                                 std::size_t position) {
  if (!ValidName(name)) {
    return Rejected(PlanError::kInvalid, position, "not an implementation name");
  }
  const auto found = registry.Find(name);
  if (!found) {
    return Rejected(PlanError::kUnsupported, position,
                    std::format("no implementation {} in this build", name));
  }
  const Implementation& implementation = registry.at(*found);
  if (implementation.operation != operation) {
    return Rejected(
        PlanError::kWrongOperation, position,
        std::format("{} implements {}, not {}", name, llmp::ir::OpKeyName(implementation.operation),
                    llmp::ir::OpKeyName(operation)));
  }
  return *found;
}

// A capability's identity text: every field but the predicate.
std::string CapabilityText(const Capability& c) {
  std::string text = std::format("op {}; operands", ir::OpKeyName(c.op));
  for (const OperandRule& rule : c.operands) {
    text += std::format(" [{}:", rule.operand);
    for (const format::Encoding e : rule.encodings) {
      text += std::format(" {}", format::Info(e).name);
    }
    text += " /";
    for (const format::Layout l : rule.layouts) {
      text += std::format(" {}", format::Info(l).name);
    }
    text += "]";
  }
  text += std::format(
      "; device {}-{} needs {}; rows {}-{} k%{} n%{}; rung {}; expansion {}; lossy {}; "
      "row_invariant {}; capturable {}; group {}; priority {}",
      c.device.min_arch, c.device.max_arch, c.device.needs, c.shape.min_rows, c.shape.max_rows,
      c.shape.k_multiple, c.shape.n_multiple, static_cast<int>(c.rung),
      static_cast<int>(c.expansion), c.lossy, c.row_invariant, c.capturable, c.group, c.priority);
  // Added only when set, so that the identities of capabilities without
  // them are unchanged.
  if (c.needs_state_span) text += "; needs contiguous state span";
  if (c.expands) text += "; concrete expansion predicate";
  if (c.fusion_needs_measurement) text += "; automatic fusion requires measurement";
  if (c.rung_for) text += "; concrete operand rung";
  if (c.product_arithmetic) text += "; authenticated product arithmetic recipe";
  if (c.pitched_output) text += "; output 0 at source 0's row pitch";
  if (c.standalone_comparator) text += "; stable standalone comparator recipe";
  if (c.needs_tuning) {
    text += "; needs tuning";
  }
  if (c.batch != 0) {
    text += std::format("; batch {}", c.batch);
  }
  if (!c.launches.empty()) {
    text += "; launches";
    for (const std::string_view l : c.launches) {
      text += std::format(" {}", l);
    }
  }
  return text;
}

// A primitive op, or kFused with the kind it names; both in the op set.
bool ValidOpKey(ir::OpKey key) {
  return ir::OpName(key.op) != "unknown" && ir::FusedKindName(key.fused) != "unknown" &&
         (key.op == ir::Op::kFused) == (key.fused != ir::FusedKind::kNone);
}

bool ValidCapability(const Capability& c) {
  if (!ValidOpKey(c.op)) {
    return false;
  }
  const bool known_rung = c.rung == Rung::kNative || c.rung == Rung::kOnChipDecode ||
                          c.rung == Rung::kBf16Decode || c.rung == Rung::kPrimitive;
  if (!known_rung || static_cast<std::uint8_t>(c.expansion) > 2 ||
      (!c.group.empty() && !ValidName(c.group)) || c.shape.min_rows > c.shape.max_rows ||
      c.shape.k_multiple == 0 || c.shape.n_multiple == 0 || c.device.min_arch > c.device.max_arch) {
    return false;
  }
  for (std::size_t i = 0; i < c.operands.size(); ++i) {
    for (std::size_t j = i + 1; j < c.operands.size(); ++j) {
      if (c.operands[i].operand == c.operands[j].operand) {
        return false;
      }
    }
    for (const format::Encoding e : c.operands[i].encodings) {
      if (static_cast<std::size_t>(e) >= format::Encodings().size()) {
        return false;
      }
    }
    for (const format::Layout l : c.operands[i].layouts) {
      if (static_cast<std::size_t>(l) >= format::Layouts().size()) {
        return false;
      }
    }
  }
  return true;
}

// Every rule that constrains an operand finds it in the query, in a format
// the rule takes: a query that leaves a constrained operand out does not
// match (an operand no rule names is unconstrained).
bool OperandsMatch(const Capability& c, std::span<const OperandFormat> operands) {
  for (const OperandRule& rule : c.operands) {
    if (rule.encodings.empty() && rule.layouts.empty()) {
      continue;
    }
    const auto operand = std::ranges::find_if(
        operands, [&](const OperandFormat& o) { return o.operand == rule.operand; });
    if (operand == operands.end() ||
        (!rule.encodings.empty() &&
         std::ranges::find(rule.encodings, operand->encoding) == rule.encodings.end()) ||
        (!rule.layouts.empty() &&
         std::ranges::find(rule.layouts, operand->layout) == rule.layouts.end())) {
      return false;
    }
  }
  return true;
}

bool Computes(const Capability& c, const CapabilityQuery& q) {
  if (c.op != q.op || !OperandsMatch(c, q.operands)) {
    return false;
  }
  if (q.arch < c.device.min_arch || q.arch > c.device.max_arch ||
      (q.features & c.device.needs) != c.device.needs) {
    return false;
  }
  if (q.rows < c.shape.min_rows || q.rows > c.shape.max_rows || q.k % c.shape.k_multiple != 0 ||
      q.n % c.shape.n_multiple != 0) {
    return false;
  }
  if (q.op.op == ir::Op::kMatMul || q.op.op == ir::Op::kMatMulId ||
      q.op == ir::Fused(ir::FusedKind::kGroupedMatMul)) {
    ir::ProductParams product{};
    if (q.params.size() >= sizeof(product)) std::memcpy(&product, q.params.data(), sizeof(product));
    if (product.arithmetic != ir::ProductArithmetic::kImplementation &&
        (!c.product_arithmetic || c.product_arithmetic(q) != product.arithmetic))
      return false;
  }
  const bool expands = Expands(c, q);
  if (expands && q.rows <= q.policy.max_decode_rows) {
    return false;
  }
  if (c.needs_state_span && !q.state_span) return false;
  if ((c.lossy && !q.policy.allow_lossy) || (q.row_invariant && !c.row_invariant) ||
      (q.capturable && !c.capturable)) {
    return false;
  }
  return c.accepts == nullptr || c.accepts(q);
}

// Explicit exact-shape or all-positive-rows algorithm qualification.
bool Tuned(const Implementation& implementation, const CapabilityQuery& q) {
  return !implementation.capability->needs_tuning ||
         std::ranges::any_of(q.tuned, [&](const TunedShape& t) {
           return t.implementation == implementation.name &&
                  (t.all_rows ? t.rows == 1 && q.rows > 0 : t.rows == q.rows) && t.k == q.k &&
                  t.n == q.n &&
                  (t.weight_bits == 0 ||
                   std::ranges::any_of(q.operands,
                                       [&](const auto& o) {
                                         return o.operand == 0 && o.params.k_bits == t.weight_bits;
                                       })) &&
                  (!t.output_encoding || std::ranges::any_of(q.operands, [&](const auto& o) {
                    return o.operand == kOutputOperand && o.encoding == *t.output_encoding;
                  }));
         });
}

}  // namespace

bool Expands(const Capability& c, const CapabilityQuery& q) {
  bool expands = c.expansion == Expansion::kTransientWorkspace;
  if (c.expansion == Expansion::kDenseF32Accumulation) {
    ir::ProductParams product{};
    if (q.params.size() >= sizeof(product)) std::memcpy(&product, q.params.data(), sizeof(product));
    const auto weight = std::ranges::find(q.operands, std::uint8_t{0}, &OperandFormat::operand);
    expands =
        product.accumulate == ir::Accumulate::kF32 && weight != q.operands.end() &&
        (weight->encoding == format::Encoding::kF16 || weight->encoding == format::Encoding::kBF16);
  }
  return expands && (!c.expands || c.expands(q));
}

base::Sha256Digest IdentityOf(const Implementation& implementation) {
  base::Sha256 hash;
  Field(hash, "llmp.implementation.v1.ir-key");
  Field(hash, implementation.name);
  Field(hash, llmp::ir::OpKeyName(implementation.operation));
  Field(hash, implementation.source);
  Field(hash, implementation.revision);
  Field(hash, implementation.build);
  Field(hash, implementation.variant);
  // Only a declared capability adds a field, so the identities of
  // implementations declared without one are unchanged.
  if (implementation.capability) {
    Field(hash, CapabilityText(*implementation.capability));
  }
  return hash.Finish();
}

Registry::Registry(std::vector<Implementation> implementations,
                   std::vector<base::Sha256Digest> identities, std::vector<std::size_t> by_name,
                   RankData ranks)
    : rank_digest_(RankDataDigest(ranks)),
      ranks_(std::move(ranks)),
      implementations_(std::move(implementations)),
      identities_(std::move(identities)),
      by_name_(std::move(by_name)) {
  for (const auto& comparison : ranks_.fusions)
    fusion_index_.push_back({at(*Find(comparison.fusion.name)).operation, &comparison});
  std::ranges::sort(fusion_index_, [](const auto& a, const auto& b) {
    if (a.operation.fused != b.operation.fused) return a.operation.fused < b.operation.fused;
    if (a.comparison->query != b.comparison->query)
      return a.comparison->query < b.comparison->query;
    return a.comparison->fusion.name < b.comparison->fusion.name;
  });
  for (const auto& comparison : ranks_.recipe_comparisons)
    recipe_comparison_index_.push_back({comparison.key.discovery_operation,
                                        RecipeComparisonKeyDigest(comparison.key), &comparison});
  std::ranges::sort(recipe_comparison_index_, [](const auto& a, const auto& b) {
    return std::pair{a.operation, a.key_digest} < std::pair{b.operation, b.key_digest};
  });
  base::Sha256 selection;
  selection.Update("llmp-terminal-selection-environment-v1");
  const auto primitive_ranks = RankDataDigest({.records = ranks_.records});
  selection.Update(std::as_bytes(std::span(primitive_ranks)));
  for (const auto index : by_name_) selection.Update(std::as_bytes(std::span(identities_[index])));
  selection_digest_ = selection.Finish();
  base::Sha256 hash;
  hash.Update("llmp-registry-set-v2");
  hash.Update(std::as_bytes(std::span(rank_digest_)));
  for (const auto index : by_name_) hash.Update(std::as_bytes(std::span(identities_[index])));
  digest_ = hash.Finish();
}

bool Registry::HasFusionComparisons(ir::OpKey operation) const {
  if (operation.op != ir::Op::kFused) return false;
  const auto found = std::ranges::lower_bound(
      fusion_index_, operation.fused, {}, [](const auto& entry) { return entry.operation.fused; });
  return found != fusion_index_.end() && found->operation == operation;
}
std::span<const Registry::IndexedFusionComparison> Registry::FusionComparisons(
    ir::OpKey operation, const base::Sha256Digest& query) const {
  if (operation.op != ir::Op::kFused) return {};
  const auto key = std::pair{operation.fused, query};
  const auto range = std::ranges::equal_range(fusion_index_, key, {}, [](const auto& entry) {
    return std::pair{entry.operation.fused, entry.comparison->query};
  });
  return {range.begin(), range.end()};
}

bool Registry::HasRecipeComparisons(ir::OpKey operation) const {
  const auto found = std::ranges::lower_bound(recipe_comparison_index_, operation, {},
                                              &IndexedRecipeComparison::operation);
  return found != recipe_comparison_index_.end() && found->operation == operation;
}

const RecipeComparison* Registry::FindRecipeComparison(const RecipeComparisonKey& key) const {
  if (key.selection_environment != selection_digest_ || key.version != 1 ||
      key.region_canonicalizer != 2 ||
      static_cast<std::uint8_t>(key.mode) > static_cast<std::uint8_t>(RecipeRunMode::kWarm) ||
      !ValidOpKey(key.discovery_operation))
    return nullptr;
  const auto digest = RecipeComparisonKeyDigest(key);
  const auto lookup = std::pair{key.discovery_operation, digest};
  const auto range =
      std::ranges::equal_range(recipe_comparison_index_, lookup, {},
                               [](const auto& e) { return std::pair{e.operation, e.key_digest}; });
  for (const auto& candidate : range)
    if (candidate.comparison->key == key) return candidate.comparison;
  return nullptr;
}

std::expected<Registry, PlanRejection> Registry::Create(std::vector<Implementation> implementations,
                                                        RankData ranks) {
  if (implementations.size() > kMaxImplementations) {
    return Rejected(PlanError::kInvalid, 0,
                    std::format("{} implementations, more than {}", implementations.size(),
                                kMaxImplementations));
  }
  std::vector<base::Sha256Digest> identities;
  identities.reserve(implementations.size());
  std::vector<std::size_t> by_name(implementations.size());
  for (std::size_t i = 0; i < implementations.size(); ++i) {
    if (!ValidOpKey(implementations[i].operation) || !ValidName(implementations[i].name)) {
      return Rejected(PlanError::kInvalid, 0,
                      "an implementation name that is empty, too long "
                      "or not lower-case letters, digits, '.' and '_'");
    }
    if (implementations[i].capability && !ValidCapability(*implementations[i].capability)) {
      return Rejected(PlanError::kInvalid, 0,
                      std::format("{} declares a malformed capability", implementations[i].name));
    }
    identities.push_back(IdentityOf(implementations[i]));
    by_name[i] = i;
  }
  std::ranges::sort(by_name, [&](std::size_t a, std::size_t b) {
    return implementations[a].name < implementations[b].name;
  });
  const auto duplicate = std::ranges::adjacent_find(by_name, [&](std::size_t a, std::size_t b) {
    return implementations[a].name == implementations[b].name;
  });
  if (duplicate != by_name.end()) {
    return Rejected(PlanError::kInvalid, 0,
                    std::format("two implementations named {}", implementations[*duplicate].name));
  }
  for (const auto& rank : ranks.records) {
    const auto implementation =
        std::ranges::find(implementations, rank.implementation, &Implementation::name);
    if (rank.device_class.empty() || rank.rows_min > rank.rows_max || rank.stacked == 0 ||
        implementation == implementations.end() || implementation->operation != rank.op ||
        static_cast<std::size_t>(rank.encoding) >= format::Encodings().size() ||
        static_cast<std::size_t>(rank.layout) >= format::Layouts().size())
      return Rejected(PlanError::kInvalid, 0, "malformed or unbound backend rank record");
  }
  for (std::size_t i = 0; i < ranks.records.size(); ++i) {
    const auto& a = ranks.records[i];
    for (std::size_t j = i + 1; j < ranks.records.size(); ++j) {
      const auto& b = ranks.records[j];
      if (a.implementation == b.implementation && a.op == b.op &&
          a.device_class == b.device_class &&
          (a.device_model.empty() || b.device_model.empty() || a.device_model == b.device_model) &&
          a.arch == b.arch && a.multiprocessors == b.multiprocessors && a.encoding == b.encoding &&
          a.layout == b.layout && a.params == b.params && a.stacked == b.stacked &&
          a.rows_min <= b.rows_max && b.rows_min <= a.rows_max &&
          (a.k == 0 || b.k == 0 || a.k == b.k) && (a.n == 0 || b.n == 0 || a.n == b.n))
        return Rejected(PlanError::kInvalid, 0, "overlapping backend rank records");
    }
  }
  for (std::size_t i = 0; i < ranks.fusions.size(); ++i) {
    const auto& r = ranks.fusions[i];
    const auto fusion = std::ranges::find(implementations, r.fusion.name, &Implementation::name);
    const auto empty_digest = [](const auto& digest) {
      return std::ranges::all_of(digest, [](auto v) { return v == 0; });
    };
    if (r.shape_class.empty() || r.provenance.empty() || r.fused_ns == 0 ||
        r.decomposition_ns == 0 || r.samples == 0 || r.decomposition.empty() ||
        empty_digest(r.query) || empty_digest(r.selection_environment) ||
        empty_digest(r.fusion.identity) || fusion == implementations.end() ||
        fusion->operation.op != ir::Op::kFused || !ir::HasDecomposition(fusion->operation))
      return Rejected(PlanError::kInvalid, 0, "malformed or unbound fusion comparison");
    for (const auto& part : r.decomposition) {
      const auto implementation =
          std::ranges::find(implementations, part.name, &Implementation::name);
      if (part.name == r.fusion.name || empty_digest(part.identity) ||
          implementation == implementations.end() ||
          ir::HasDecomposition(implementation->operation) ||
          implementation->operation.op == ir::Op::kFused)
        return Rejected(PlanError::kInvalid, 0, "unbound fusion decomposition implementation");
    }
    for (std::size_t j = 0; j < i; ++j)
      if (r.fusion.name == ranks.fusions[j].fusion.name && r.query == ranks.fusions[j].query)
        return Rejected(PlanError::kInvalid, 0, "duplicate fusion comparison");
  }
  Registry registry(std::move(implementations), std::move(identities), std::move(by_name),
                    std::move(ranks));
  for (std::size_t i = 0; i < registry.ranks_.recipe_comparisons.size(); ++i) {
    const auto& comparison = registry.ranks_.recipe_comparisons[i];
    if (auto valid = ValidateRecipeComparison(comparison); !valid)
      return Rejected(PlanError::kInvalid, 0, valid.error());
    for (std::size_t j = 0; j < i; ++j)
      if (comparison.key == registry.ranks_.recipe_comparisons[j].key)
        return Rejected(PlanError::kInvalid, 0, "duplicate recipe comparison key");
  }
  return registry;
}

std::optional<std::size_t> Registry::Find(std::string_view name) const {
  const auto found = std::ranges::lower_bound(
      by_name_, name, {},
      [this](std::size_t i) -> std::string_view { return implementations_[i].name; });
  if (found == by_name_.end() || implementations_[*found].name != name) {
    return std::nullopt;
  }
  return *found;
}

const Implementation& Registry::at(std::size_t index) const {
  base::Check(index < implementations_.size(), "registry index in range");
  return implementations_[index];
}

const base::Sha256Digest& Registry::identity(std::size_t index) const {
  base::Check(index < identities_.size(), "registry index in range");
  return identities_[index];
}

std::expected<Plan, PlanRejection> Plan::Build(const Registry& registry,
                                               std::span<const Choice> choices) {
  if (choices.empty() || choices.size() > kMaxOperations) {
    return Rejected(
        PlanError::kInvalid, 0,
        std::format("{} operations; a plan has 1 to {}", choices.size(), kMaxOperations));
  }
  std::vector<Step> steps;
  steps.reserve(choices.size());
  base::Sha256 hash;
  Field(hash, "llmp.plan.v0");
  Field(hash, std::format("{}", choices.size()));
  for (std::size_t i = 0; i < choices.size(); ++i) {
    const Choice& choice = choices[i];
    const auto found = Lookup(registry, choice.operation, choice.implementation, i);
    if (!found) {
      return std::unexpected(found.error());
    }
    const Implementation& implementation = registry.at(*found);
    const base::Sha256Digest& identity = registry.identity(*found);
    Field(hash, llmp::ir::OpKeyName(choice.operation));
    Field(hash, implementation.name);
    hash.Update(std::as_bytes(std::span(identity)));
    steps.push_back({.operation = choice.operation,
                     .implementation = implementation.name,
                     .identity = identity});
  }
  return Plan(std::move(steps), hash.Finish());
}

const Implementation& BoundPlan::at(std::size_t operation) const {
  base::Check(operation < bound_.size(), "plan operation in range");
  return *bound_[operation];
}

std::expected<BoundPlan, PlanRejection> Resolve(const Registry& registry, const Plan& plan) {
  std::vector<const Implementation*> bound;
  bound.reserve(plan.steps().size());
  for (std::size_t i = 0; i < plan.steps().size(); ++i) {
    const Plan::Step& step = plan.steps()[i];
    const auto found = Lookup(registry, step.operation, step.implementation, i);
    if (!found) {
      return std::unexpected(found.error());
    }
    const Implementation& implementation = registry.at(*found);
    if (registry.identity(*found) != step.identity) {
      return Rejected(
          PlanError::kStale, i,
          std::format("{} is {} in this build; the plan was made for {}", implementation.name,
                      base::ToHex(registry.identity(*found)), base::ToHex(step.identity)));
    }
    bound.push_back(&implementation);
  }
  return BoundPlan(std::move(bound), plan.identity());
}

std::optional<std::uint32_t> DeviceRank(const CapabilityQuery& query,
                                        std::string_view implementation,
                                        std::span<const DeviceMeasuredRank> table) {
  const auto weight = std::ranges::find(query.operands, 0, &OperandFormat::operand);
  if (weight == query.operands.end() || weight->ne[0] != query.k || weight->ne[1] != query.n ||
      weight->ne[2] != 1 || weight->ne[3] != 1)
    return {};
  const auto input = std::ranges::find(query.operands, 1, &OperandFormat::operand);
  if (input == query.operands.end()) return {};
  if (query.op == ir::Primitive(ir::Op::kMatMul) &&
      (input->ne[0] != query.k || input->ne[1] != query.rows))
    return {};
  if (query.op == ir::Primitive(ir::Op::kGetRows) && input->ne[0] != query.rows) return {};
  for (const auto& operand : query.operands)
    if (operand.ne[2] != 1 || operand.ne[3] != 1) return {};
  for (const auto& record : table) {
    if (record.implementation == implementation && record.op == query.op &&
        record.device_class == query.device_class &&
        (record.device_model.empty() || record.device_model == query.device_model) &&
        record.arch == query.arch && record.multiprocessors == query.multiprocessors &&
        record.encoding == weight->encoding && record.layout == weight->layout &&
        record.params == weight->params && record.stacked == weight->stacked &&
        query.rows >= record.rows_min && query.rows <= record.rows_max &&
        (record.k == 0 || record.k == query.k) && (record.n == 0 || record.n == query.n))
      return record.rank;
  }
  return {};
}
base::Sha256Digest RankDataDigest(const RankData& data) {
  std::vector<base::Sha256Digest> records;
  records.reserve(data.records.size());
  for (const auto& r : data.records) {
    base::Sha256 hash;
    Field(hash, r.device_class);
    Field(hash, r.device_model);
    Field(hash, r.implementation);
    Field(hash, std::format("stacked:{}", r.stacked));
    Field(hash, std::format("params:{}:{}:{}:{}:{}:{}:{}:{}:{}", r.params.k_bits,
                            static_cast<unsigned>(r.params.codebook), r.params.tensor_scale,
                            r.params.bits, r.params.group, static_cast<unsigned>(r.params.scale),
                            r.params.zero_point, r.params.block_rows, r.params.block_cols));
    Field(hash, std::format("{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}", r.arch, r.multiprocessors,
                            static_cast<unsigned>(r.op.op), static_cast<unsigned>(r.op.fused),
                            static_cast<unsigned>(r.encoding), static_cast<unsigned>(r.layout),
                            r.rows_min, r.rows_max, r.k, r.n, r.rank, 0));
    records.push_back(hash.Finish());
  }
  std::ranges::sort(records);
  base::Sha256 hash;
  Field(hash, "llmp-device-measured-ranks-v4");
  const auto fusions = FusionComparisonsDigest(data.fusions);
  hash.Update(std::as_bytes(std::span(fusions)));
  const auto recipes = RecipeComparisonsDigest(data.recipe_comparisons);
  hash.Update(std::as_bytes(std::span(recipes)));
  for (const auto& record : records) hash.Update(std::as_bytes(std::span(record)));
  return hash.Finish();
}
namespace {

bool EligibleCandidate(const Registry& registry, const CapabilityQuery& query,
                       const Implementation& implementation, FusionPreference fusion_preference) {
  if (!implementation.capability || !Computes(*implementation.capability, query) ||
      !Tuned(implementation, query))
    return false;
  if (fusion_preference == FusionPreference::kDecomposition) return false;
  const auto rank =
      std::ranges::find(query.measured, implementation.name, &MeasuredRank::implementation);
  const auto device_rank = DeviceRank(query, implementation.name, registry.ranks().records);
  return !query.automatic_fusion || !implementation.capability->fusion_needs_measurement ||
         rank != query.measured.end() || device_rank ||
         fusion_preference == FusionPreference::kFusion;
}

}  // namespace

namespace {
struct RankedCandidate {
  std::size_t index = 0;
  std::uint8_t rung = 0;
  std::uint64_t measured = 0;  // unranked last
  std::uint16_t priority = 0;
  bool preferred = false;
  std::string_view name;
};

RankedCandidate RankCandidate(const Registry& registry, const CapabilityQuery& query,
                              std::size_t index) {
  const Implementation& implementation = registry.at(index);
  const auto rank = std::ranges::find_if(query.measured, [&](const MeasuredRank& m) {
    return m.implementation == implementation.name;
  });
  const auto device_rank = DeviceRank(query, implementation.name, registry.ranks().records);
  return {.index = index,
          .rung = static_cast<std::uint8_t>(implementation.capability->rung_for
                                                ? implementation.capability->rung_for(query)
                                                : implementation.capability->rung),
          .measured = rank != query.measured.end() ? std::uint64_t{rank->rank}
                      : device_rank                ? (std::uint64_t{1} << 32U) + *device_rank
                                                   : ~std::uint64_t{0},
          .priority = implementation.capability->priority,
          .preferred = std::ranges::contains(query.preferred, implementation.name),
          .name = implementation.name};
}

bool RankedBefore(const RankedCandidate& a, const RankedCandidate& b) {
  if (a.measured != b.measured) return a.measured < b.measured;
  if (a.preferred != b.preferred) return a.preferred;
  if (a.rung != b.rung) return a.rung < b.rung;
  if (a.priority != b.priority) return a.priority > b.priority;
  return a.name < b.name;
}
}  // namespace

std::vector<std::size_t> Candidates(const Registry& registry, const CapabilityQuery& query) {
  std::vector<RankedCandidate> ranked;
  for (std::size_t i = 0; i < registry.size(); ++i) {
    const Implementation& implementation = registry.at(i);
    const auto fusion = FusionPreferenceFor(registry, query, implementation.name);
    if (!EligibleCandidate(registry, query, implementation, fusion)) continue;
    if (fusion == FusionPreference::kDecomposition) continue;
    ranked.push_back(RankCandidate(registry, query, i));
  }
  // D-053: correctness (the filter above), then measured speed; the
  // ladder's rung and the static priority order what is not measured.
  std::ranges::sort(ranked, RankedBefore);
  std::vector<std::size_t> out;
  out.reserve(ranked.size());
  for (const RankedCandidate& r : ranked) out.push_back(r.index);
  return out;
}

std::optional<std::size_t> BestCandidate(const Registry& registry, const CapabilityQuery& query) {
  return NextCandidate(registry, query, std::nullopt);
}

std::optional<std::size_t> NextCandidate(const Registry& registry, const CapabilityQuery& query,
                                         std::optional<std::size_t> after) {
  if (after && *after >= registry.size()) return std::nullopt;
  std::optional<RankedCandidate> best;
  std::optional<RankedCandidate> previous;
  if (after && *after < registry.size()) previous = RankCandidate(registry, query, *after);
  for (std::size_t i = 0; i < registry.size(); ++i) {
    const Implementation& implementation = registry.at(i);
    const auto fusion = FusionPreferenceFor(registry, query, implementation.name);
    if (!EligibleCandidate(registry, query, implementation, fusion) ||
        fusion == FusionPreference::kDecomposition)
      continue;
    const RankedCandidate current = RankCandidate(registry, query, i);
    if (previous && !RankedBefore(*previous, current)) continue;
    if (!best || RankedBefore(current, *best)) best = current;
  }
  if (!best) return std::nullopt;
  return best->index;
}

bool IsCandidate(const Registry& registry, const CapabilityQuery& query, std::size_t index) {
  if (index >= registry.size()) return false;
  const auto& implementation = registry.at(index);
  const auto fusion = FusionPreferenceFor(registry, query, implementation.name);
  return EligibleCandidate(registry, query, implementation, fusion) &&
         fusion != FusionPreference::kDecomposition;
}

bool HasCandidate(const Registry& registry, const CapabilityQuery& query) {
  for (std::size_t i = 0; i < registry.size(); ++i) {
    const Implementation& implementation = registry.at(i);
    const auto fusion = FusionPreferenceFor(registry, query, implementation.name);
    if (EligibleCandidate(registry, query, implementation, fusion)) return true;
  }
  return false;
}

}  // namespace llmp::execution
