// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#include "import/source_mapping.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <string>
#include <string_view>

#include "spec/mapping.h"
#include "spec/stage_recipe.h"

namespace llmp::import {
namespace nd = normalization_detail;
namespace {
using ErrorResult = std::unexpected<Error>;
ErrorResult Invalid() {
  return ErrorResult(Error{.rule = Rule::kUnsupported,
                           .reason = "invalid or unsupported source normalization table"});
}

bool SameShape(std::string_view a, std::string_view b) {
  return (a.find(spec::kLayerField) == std::string_view::npos) ==
         (b.find(spec::kLayerField) == std::string_view::npos);
}
std::string_view ConcreteSecondary(std::string_view pattern,
                                   std::array<char, spec::kMaxText>& buf) {
  for (const std::string_view field : {spec::kExpertField, spec::kShardField}) {
    const auto at = pattern.find(field);
    if (at == std::string_view::npos) continue;
    auto* end = std::copy_n(pattern.data(), at, buf.begin());
    *end++ = '0';
    end = std::copy(pattern.begin() + static_cast<std::ptrdiff_t>(at + field.size()), pattern.end(),
                    end);
    return {buf.data(), static_cast<std::size_t>(end - buf.data())};
  }
  return pattern;
}
std::string_view TailAfterFields(std::string_view pattern) {
  std::size_t pos = 0;
  for (const std::string_view field : {spec::kLayerField, spec::kExpertField, spec::kShardField}) {
    if (const auto at = pattern.find(field); at != std::string_view::npos) {
      pos = std::max(pos, at + field.size());
    }
  }
  return pattern.substr(pos);
}
// Bounded finite pattern language: at most 1024 decimal layer values. Prefix
// and suffix rejection makes unrelated compiled rules constant-time.
bool OverlapOneWay(std::string_view a, std::string_view b) {
  if (a == b) return true;
  std::array<char, spec::kMaxText> sec_buf{};
  const auto at = a.find(spec::kLayerField), bt = b.find(spec::kLayerField);
  if (at == std::string_view::npos)
    return nd::Match(b, ConcreteSecondary(a, sec_buf), nd::kLayerLimit).has_value();
  if (bt == std::string_view::npos)
    return nd::Match(a, ConcreteSecondary(b, sec_buf), nd::kLayerLimit).has_value();
  const auto ap = a.substr(0, at), bp = b.substr(0, bt);
  const auto as = a.substr(at + spec::kLayerField.size());
  const auto atail = TailAfterFields(a), btail = TailAfterFields(b);
  if ((!ap.starts_with(bp) && !bp.starts_with(ap)) ||
      (!atail.ends_with(btail) && !btail.ends_with(atail)))
    return false;
  const auto as_concrete = ConcreteSecondary(as, sec_buf);
  std::array<char, spec::kMaxText> text{};
  for (std::uint32_t layer = 0; layer < nd::kLayerLimit; ++layer) {
    auto* end = std::copy(ap.begin(), ap.end(), text.begin());
    end = std::to_chars(end, text.data() + text.size(), layer).ptr;
    end = std::copy(as_concrete.begin(), as_concrete.end(), end);
    if (nd::Match(b, {text.data(), static_cast<std::size_t>(end - text.data())}, nd::kLayerLimit))
      return true;
  }
  return false;
}
bool Overlap(std::string_view a, std::string_view b) {
  return OverlapOneWay(a, b) || OverlapOneWay(b, a);
}
bool Structural(std::string_view key) {
  return key == "general.architecture" || key == "general.alignment" || key == "split.no" ||
         key == "split.count" || key == "split.tensors.count" || key == "model_type";
}
bool IsName(std::string_view s) {
  return !s.empty() && s.size() <= spec::kMaxText && (s.front() >= 'a' && s.front() <= 'z') &&
         std::ranges::all_of(s, [](char c) {
           return (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_';
         });
}

}  // namespace

const spec::Mapping* FindImportMapping(std::string_view container, std::string_view architecture,
                                       std::string_view component) {
  if (component.empty()) return spec::FindMapping(architecture, container);
  const spec::Mapping* found = nullptr;
  for (const auto* mapping : spec::Mappings()) {
    const bool drafter = mapping->stage_recipe == nullptr &&
                         mapping->role == spec::StageRole::kDrafter && mapping->stage == component;
    const bool spatial = mapping->stage_recipe != nullptr && mapping->stage == component;
    if (!spec::SupportsSource(*mapping, architecture, container) || (!drafter && !spatial))
      continue;
    if (found) return nullptr;
    found = mapping;
  }
  return found;
}

namespace normalization_detail {
std::uint8_t KeyTypes(const spec::Mapping& m, std::string_view key, bool allow_unprefixed) {
  std::uint8_t types = 0;
  bool seen = false;
  const auto add = [&](std::uint8_t allowed) {
    types = seen ? static_cast<std::uint8_t>(types & allowed) : allowed;
    seen = true;
  };
  const auto matches = [&](std::string_view rule_key) {
    return !rule_key.empty() && (allow_unprefixed ? spec::MatchesCanonicalKey(m, rule_key, key)
                                                  : spec::IsCanonicalKey(m, rule_key, key));
  };
  if (m.draft && (matches(m.draft->depth_key) || matches(m.draft->fill_key))) add(kInteger);
  if (matches(m.layer_count_key)) add(kInteger);
  if (matches(m.context_key)) add(kInteger);
  if (matches(m.softcap_key)) add(kInteger | kReal);
  if (matches(m.layer_type_key)) add(kIntegers | kText);
  if (matches(m.layer_type_interval_key)) add(kInteger);
  if (matches(m.layer_flags_key)) add(kIntegers);
  for (const auto& rule : m.params) {
    if (!key.empty() && (matches(rule.numerator_key) || matches(rule.divisor_key))) {
      add(kInteger | kReal);
    }
    if (!matches(rule.key)) continue;
    switch (rule.type) {
      case spec::ValueType::kInt:
        add(rule.per_layer ? kInteger | kIntegers : kInteger);
        break;
      case spec::ValueType::kReal:
        add(rule.per_layer ? kInteger | kReal | kReals : kInteger | kReal);
        break;
      case spec::ValueType::kBool:
        add(!rule.flag_text.empty() ? kText | kFlag : !rule.value.text.empty() ? kText : kFlag);
        break;
      case spec::ValueType::kString:
        add(kText);
        break;
      case spec::ValueType::kInts:
        add(kIntegers);
        break;
    }
  }
  for (const auto& slot : m.slots)
    if (matches(slot.layers_key) || matches(slot.layers_alias)) add(kIntegers);
  for (const auto& read : m.reads)
    if (matches(read.layers_key)) add(kIntegers);
  for (const auto& rule : m.slots)
    if (matches(rule.gate) || matches(rule.count_key)) add(kInteger);
  for (const auto& rule : m.policies)
    if (matches(rule.gate)) add(kInteger);
  for (const auto& requirement : m.requirements)
    if (!seen && matches(requirement.key)) add(requirement.boolean ? kFlag : kInteger | kReal);
  for (const auto& list : m.lists) {
    if (!seen && matches(list.key)) add(kIntegers);
    if (matches(list.sections_of)) add(kInteger);
    if (matches(list.sections_multiplier_key)) add(kInteger | kReal);
  }
  return types;
}

bool Pattern(std::string_view pattern) {
  if (pattern.empty() || pattern.size() > spec::kMaxText - 4) return false;
  const auto at_layer = pattern.find(spec::kLayerField);
  const auto at_expert = pattern.find(spec::kExpertField);
  const auto at_shard = pattern.find(spec::kShardField);
  if (at_expert != std::string_view::npos && at_shard != std::string_view::npos) return false;
  const auto at_idx = at_expert != std::string_view::npos ? at_expert : at_shard;
  const auto idx_field =
      at_expert != std::string_view::npos ? spec::kExpertField : spec::kShardField;
  if (at_layer == std::string_view::npos && at_idx == std::string_view::npos)
    return spec::IsSourceKey(pattern);
  if (at_layer != std::string_view::npos &&
      pattern.find(spec::kLayerField, at_layer + spec::kLayerField.size()) !=
          std::string_view::npos)
    return false;
  if (at_idx != std::string_view::npos &&
      pattern.find(idx_field, at_idx + idx_field.size()) != std::string_view::npos)
    return false;
  if (at_layer != std::string_view::npos && at_idx != std::string_view::npos &&
      at_layer + spec::kLayerField.size() >= at_idx)
    return false;
  // IsSourceKey excludes braces; inspect the literal pieces directly.
  for (std::size_t i = 0; i < pattern.size(); ++i) {
    if (i == at_layer) {
      i += spec::kLayerField.size() - 1;
      continue;
    }
    if (i == at_idx) {
      i += idx_field.size() - 1;
      continue;
    }
    const char c = pattern[i];
    if (c <= 32 || c >= 127 || c == '"' || c == '\\' || c == '{' || c == '}') return false;
  }
  return true;
}

std::optional<NameMatch> Match(std::string_view pattern, std::string_view name,
                               std::uint32_t layers) {
  const auto parse_digits = [](std::string_view digits,
                               std::uint32_t limit) -> std::optional<std::uint32_t> {
    if (digits.empty() || digits.size() > 4 || (digits.size() > 1 && digits.front() == '0'))
      return std::nullopt;
    std::uint32_t val = 0;
    const auto parsed = std::from_chars(digits.data(), digits.data() + digits.size(), val);
    if (parsed.ec != std::errc{} || parsed.ptr != digits.data() + digits.size() || val >= limit)
      return std::nullopt;
    return val;
  };
  const auto at_layer = pattern.find(spec::kLayerField);
  const auto at_expert = pattern.find(spec::kExpertField);
  const auto at_shard = pattern.find(spec::kShardField);
  const bool is_expert = at_expert != std::string_view::npos;
  const auto at_idx = is_expert ? at_expert : at_shard;
  const auto idx_field = is_expert ? spec::kExpertField : spec::kShardField;
  if (at_layer == std::string_view::npos && at_idx == std::string_view::npos)
    return name == pattern ? std::optional<NameMatch>(NameMatch{}) : std::nullopt;
  if (at_idx == std::string_view::npos) {
    const auto prefix = pattern.substr(0, at_layer),
               suffix = pattern.substr(at_layer + spec::kLayerField.size());
    if (name.size() <= prefix.size() + suffix.size() || !name.starts_with(prefix) ||
        !name.ends_with(suffix))
      return std::nullopt;
    const auto digits = name.substr(prefix.size(), name.size() - prefix.size() - suffix.size());
    if (!parse_digits(digits, layers)) return std::nullopt;
    return NameMatch{digits};
  }
  if (at_layer == std::string_view::npos) {
    const auto prefix = pattern.substr(0, at_idx),
               suffix = pattern.substr(at_idx + idx_field.size());
    if (name.size() <= prefix.size() + suffix.size() || !name.starts_with(prefix) ||
        !name.ends_with(suffix))
      return std::nullopt;
    const auto idx_digits = name.substr(prefix.size(), name.size() - prefix.size() - suffix.size());
    const auto idx = parse_digits(idx_digits, kLayerLimit);
    if (!idx) return std::nullopt;
    return NameMatch{
        .index_digits = idx_digits, .index = *idx, .is_expert = is_expert, .is_shard = !is_expert};
  }
  if (at_layer + spec::kLayerField.size() >= at_idx) return std::nullopt;
  const auto prefix = pattern.substr(0, at_layer);
  const auto middle = pattern.substr(at_layer + spec::kLayerField.size(),
                                     at_idx - (at_layer + spec::kLayerField.size()));
  const auto suffix = pattern.substr(at_idx + idx_field.size());
  if (name.size() < prefix.size() + middle.size() + suffix.size() + 2 ||
      !name.starts_with(prefix) || !name.ends_with(suffix))
    return std::nullopt;
  const auto inner = name.substr(prefix.size(), name.size() - prefix.size() - suffix.size());
  const auto mid_pos = inner.find(middle);
  if (mid_pos == std::string_view::npos) return std::nullopt;
  const auto layer_digits = inner.substr(0, mid_pos);
  const auto idx_digits = inner.substr(mid_pos + middle.size());
  const auto layer = parse_digits(layer_digits, layers);
  const auto idx = parse_digits(idx_digits, kLayerLimit);
  if (!layer || !idx) return std::nullopt;
  return NameMatch{.digits = layer_digits,
                   .index_digits = idx_digits,
                   .index = *idx,
                   .is_expert = is_expert,
                   .is_shard = !is_expert};
}
}  // namespace normalization_detail

std::expected<void, Error> ValidateImportMapping(const spec::Mapping& table,
                                                 std::string_view container) {
  if (!table.dialects.empty()) {
    if (!spec::ValidateMapping(table)) return Invalid();
    bool found = false;
    for (const auto& dialect : table.dialects) {
      if (dialect.container != container) continue;
      auto selected = spec::ResolveSourceMapping(table, dialect.architecture, container);
      if (!selected) return Invalid();
      if (auto valid = ValidateImportMapping(selected->mapping, container); !valid) return valid;
      found = true;
    }
    return found ? std::expected<void, Error>{} : Invalid();
  }
  if (table.stage_recipe != nullptr) {
    if ((container != "gguf" && container != "safetensors" && container != "mlx") ||
        !std::ranges::contains(table.containers, container) || !spec::ValidateMapping(table) ||
        !table.slots.empty() || !table.tensors.empty() || !table.params.empty() ||
        !table.tensor_aliases.empty() || !table.key_aliases.empty() ||
        !table.requirements.empty() || !table.lists.empty() || !table.semantic_tensors.empty() ||
        !table.shapes.empty() || !table.row_indices.empty() || !table.tensor_presence.empty() ||
        !table.folds.empty() || !table.policies.empty() || !table.reads.empty() ||
        table.draft != nullptr || table.companion != nullptr ||
        table.pipeline != spec::PipelineKind::kImage || table.role != table.stage_recipe->role ||
        table.stage != table.stage_recipe->name || table.ignored_tensor_prefixes.size() > 8)
      return Invalid();
    for (const auto pattern : table.ignored_tensors)
      if (!nd::Pattern(pattern)) return Invalid();
    return {};
  }
  if ((container != "gguf" && container != "exl3" && container != "safetensors" &&
       container != "mlx") ||
      table.version == 0 || !std::ranges::contains(table.containers, container) ||
      table.tensor_aliases.size() > nd::kRuleLimit || table.key_aliases.size() > nd::kRuleLimit ||
      table.auxiliary.size() > nd::kRuleLimit || table.requirements.size() > 16 ||
      table.ignored_tensor_prefixes.size() > 8 || table.tensor_presence.size() > 8 ||
      ((container == "exl3" || container == "safetensors" || container == "mlx") &&
       table.auxiliary.empty()))
    return Invalid();
  const auto& m = table;
  if (!spec::ValidateMapping(m)) return Invalid();
  const auto tensor = [&](std::string_view name) {
    return nd::Pattern(name) && std::ranges::any_of(m.tensors, [&](const auto& rule) {
             return !rule.target && rule.pattern == name;
           });
  };
  // This retained declaration is not an ingress whitelist; readers and the
  // format/part validators determine encoding support. Registered lists are empty.
  for (const auto encoding : m.encodings)
    if (!std::ranges::contains(format::Encodings(), encoding, &format::EncodingInfo::id))
      return Invalid();
  for (const auto& rule : m.tensor_presence)
    if (!spec::IsSourceKey(rule.flag) || !tensor(rule.tensor)) return Invalid();
  for (const auto& rule : m.shapes)
    if (!tensor(rule.name) || rule.shape.empty() || rule.shape.size() > format::kMaxDims ||
        std::ranges::contains(rule.shape, std::uint64_t{0}))
      return Invalid();
  for (const auto& rule : m.row_indices)
    if (!tensor(rule.tensor) || (nd::KeyTypes(m, rule.offsets, true) & nd::kIntegers) == 0 ||
        (nd::KeyTypes(m, rule.counts, true) & nd::kIntegers) == 0)
      return Invalid();
  for (const auto& a : m.lists) {
    if (a.size > nd::kLayerLimit || a.min_size > a.size ||
        (!a.extra_layers_key.empty() && !spec::IsSourceKey(a.extra_layers_key)) ||
        (!a.sections_multiplier_key.empty() &&
         (a.sections_of.empty() ||
          (nd::KeyTypes(m, a.sections_multiplier_key, true) & (nd::kInteger | nd::kReal)) == 0)) ||
        (!a.sections_of.empty() && (nd::KeyTypes(m, a.sections_of, true) & nd::kInteger) == 0) ||
        (!a.exact.empty() && a.exact.size() != a.size) ||
        (nd::KeyTypes(m, a.key, true) & (nd::kIntegers | nd::kReals)) == 0)
      return Invalid();
  }
  for (auto key : m.semantic) {
    if (key.ends_with('*')) key.remove_suffix(1);
    if (!spec::IsSourceKey(key)) return Invalid();
  }
  for (const auto key : m.null_semantics)
    if (!spec::IsSourceKey(key)) return Invalid();
  for (const auto pattern : m.ignored_tensors)
    if (!nd::Pattern(pattern)) return Invalid();
  for (const auto prefix : m.ignored_tensor_prefixes)
    if (!spec::IsSourceKey(prefix)) return Invalid();
  const bool drafter = m.role == spec::StageRole::kDrafter;
  if ((m.role != spec::StageRole::kTarget && !drafter) ||
      (drafter ? m.stage.empty() || !IsName(m.stage) || m.stage == "target" || m.reads.size() > 8
               : !m.stage.empty() || !m.reads.empty()))
    return Invalid();
  if (m.slots.empty() || m.slots.size() > (table.extended_components ? 16U : 8U) ||
      m.tensors.size() > nd::kRuleLimit || m.params.size() > 512 || m.folds.size() > 16 ||
      m.policies.size() > 8 || m.pipeline != spec::PipelineKind::kLlm ||
      (!table.extended_components && !m.layer_type_key.empty()) ||
      !spec::IsSourceKey(spec::CanonicalKey(m, m.layer_count_key)))
    return Invalid();
  for (const auto& slot : m.slots)
    if (!table.extended_components &&
        (!slot.count_key.empty() || slot.per_layer || !slot.layers_key.empty() ||
         slot.site == spec::Site::kSide || slot.site == spec::Site::kMixing))
      return Invalid();
  for (const auto& p : m.params)
    if ((p.per_layer && p.type != spec::ValueType::kInt &&
         !(table.extended_components && p.type == spec::ValueType::kReal)) ||
        (!table.extended_components && p.type == spec::ValueType::kInts))
      return Invalid();
  for (const auto& r : m.tensors)
    if (!nd::Pattern(r.pattern) || (!drafter && r.target) || (container == "exl3" && r.any_part) ||
        (!drafter && !table.extended_components && r.unbound) || (r.target && r.unbound) ||
        (container == "gguf" && r.part != format::PartRole::kData) ||
        ((container == "safetensors" || container == "mlx") && r.part != format::PartRole::kData &&
         r.part != format::PartRole::kCodes && r.part != format::PartRole::kBlockScales &&
         r.part != format::PartRole::kTensorScale && r.part != format::PartRole::kBiases &&
         r.part != format::PartRole::kZeros) ||
        (container == "exl3" && r.part != format::PartRole::kData &&
         r.part != format::PartRole::kCodes && r.part != format::PartRole::kInputSigns &&
         r.part != format::PartRole::kOutputSigns && r.part != format::PartRole::kCodebookMarker))
      return Invalid();
  const auto count = m.tensors.size() + table.tensor_aliases.size();
  const auto spelling = [&](std::size_t i,
                            std::array<char, spec::kMaxText>& buf) -> std::string_view {
    if (i < m.tensors.size()) return m.tensors[i].pattern;
    const std::string_view s = table.tensor_aliases[i - m.tensors.size()].source;
    if (!m.tensor_prefix.empty() && !s.starts_with(m.tensor_prefix) &&
        m.tensor_prefix.size() + s.size() <= buf.size()) {
      auto* out = std::ranges::copy(m.tensor_prefix, buf.begin()).out;
      out = std::ranges::copy(s, out).out;
      return {buf.data(), static_cast<std::size_t>(out - buf.data())};
    }
    return s;
  };
  for (const auto& a : table.tensor_aliases) {
    if (!nd::Pattern(a.source) || !SameShape(a.source, a.canonical)) return Invalid();
    std::size_t matches = 0;
    for (const auto& r : m.tensors) matches += (!r.target && r.pattern == a.canonical);
    if (matches == 0) return Invalid();
  }
  // Equal canonical names can describe opposite sides of one declared gate.
  // They still identify one physical tensor. No alias may overlap any spelling.
  const auto exclusive = [&](const spec::TensorRule& a, const spec::TensorRule& b) {
    if (a.pattern != b.pattern || a.experts != b.experts) return false;
    const spec::SlotRule *left = nullptr, *right = nullptr;
    for (const auto& slot : m.slots) {
      if (slot.slot == a.slot) left = &slot;
      if (slot.slot == b.slot) right = &slot;
    }
    return left && right &&
           ((left->site == spec::Site::kLayer && right->site == spec::Site::kLayer &&
             !left->layer_types.empty() && !right->layer_types.empty() &&
             std::ranges::none_of(left->layer_types,
                                  [&](char c) { return right->layer_types.contains(c); })) ||
            (!left->gate.empty() && left->gate == right->gate &&
             left->gate_absent != right->gate_absent) ||
            (!left->count_key.empty() && left->count_key == right->count_key &&
             left->count_above != right->count_above));
  };
  const auto is_target = [&](std::size_t k) { return k < m.tensors.size() && m.tensors[k].target; };
  std::array<char, spec::kMaxText> buf_i{}, buf_j{};
  for (std::size_t i = 0; i < count; ++i)
    for (std::size_t j = 0; j < i; ++j)
      if (is_target(i) == is_target(j) && Overlap(spelling(i, buf_i), spelling(j, buf_j)) &&
          !(i < m.tensors.size() && exclusive(m.tensors[i], m.tensors[j])))
        return Invalid();
  for (std::size_t i = 0; i < table.key_aliases.size(); ++i) {
    const auto& a = table.key_aliases[i];
    if (!spec::IsSourceKey(a.source) || !nd::KeyTypes(m, a.canonical, true) ||
        (m.key_prefix.empty() && a.source != a.canonical && nd::KeyTypes(m, a.source)) ||
        Structural(a.source))
      return Invalid();
    if (!a.unless_present.empty() &&
        (!spec::IsSourceKey(a.unless_present) || a.unless_present == a.source ||
         std::ranges::none_of(table.key_aliases, [&](const auto& primary) {
           return primary.source == a.unless_present && primary.unless_present.empty() &&
                  spec::MatchesCanonicalKey(m, primary.canonical, a.canonical);
         })))
      return Invalid();
    for (std::size_t j = 0; j < i; ++j)
      if (a.source == table.key_aliases[j].source) return Invalid();
  }
  for (std::size_t i = 0; i < table.auxiliary.size(); ++i) {
    const auto& a = table.auxiliary[i];
    if (a.prefix.empty() || !spec::IsSourceKey(a.prefix) || a.reason.empty() ||
        a.reason.size() > spec::kMaxText ||
        (!a.equals_tensor.empty() &&
         (a.prefix.ends_with('.') || !spec::IsSourceKey(a.equals_tensor) || a.equals_dim >= 4)))
      return Invalid();
    const auto matches_aux_raw = [&](std::string_view key) {
      return a.prefix.ends_with('.') ? key.starts_with(a.prefix) : key == a.prefix;
    };
    const auto matches_aux = [&](std::string_view rule_key) {
      if (rule_key.empty()) return false;
      const auto pfx = spec::KeyPrefix(m);
      if (pfx.empty() || rule_key.starts_with("llmp.") || rule_key.starts_with("tokenizer.") ||
          rule_key.starts_with("general.") || rule_key.starts_with("split.") ||
          (rule_key.size() > pfx.size() && rule_key.starts_with(pfx) &&
           rule_key[pfx.size()] == '.')) {
        return matches_aux_raw(rule_key);
      }
      if (pfx.size() + 1 + rule_key.size() > buf_i.size()) return false;
      auto* out = std::ranges::copy(pfx, buf_i.begin()).out;
      *out++ = '.';
      out = std::ranges::copy(rule_key, out).out;
      return matches_aux_raw({buf_i.data(), static_cast<std::size_t>(out - buf_i.data())});
    };
    if (matches_aux(m.layer_count_key) ||
        (!m.layer_flags_key.empty() && matches_aux(m.layer_flags_key)))
      return Invalid();
    for (const auto& slot : m.slots)
      if (!slot.gate.empty() && matches_aux(slot.gate)) return Invalid();
    for (const auto& read : m.reads)
      if (!read.layers_key.empty() && matches_aux(read.layers_key)) return Invalid();
    for (const auto& policy : m.policies)
      if (!policy.gate.empty() && matches_aux(policy.gate)) return Invalid();
    for (const auto& p : m.params)
      if ((!p.key.empty() && matches_aux(p.key)) ||
          (!p.numerator_key.empty() && matches_aux(p.numerator_key)) ||
          (!p.divisor_key.empty() && matches_aux(p.divisor_key)))
        return Invalid();
    for (const auto& k : table.key_aliases)
      if (matches_aux_raw(k.source) && k.unless_present.empty()) return Invalid();
    for (std::size_t j = 0; j < i; ++j)
      if (a.prefix.starts_with(table.auxiliary[j].prefix) ||
          table.auxiliary[j].prefix.starts_with(a.prefix))
        return Invalid();
  }
  for (const auto& r : table.requirements) {
    if (!spec::IsSourceKey(r.key)) return Invalid();
    if (!r.unless_present.empty() && (!nd::KeyTypes(m, r.unless_present, true) || r.require_equal ||
                                      r.positive || !r.equals_key.empty()))
      return Invalid();
    if (!r.flag_of_positive.empty() &&
        (!r.boolean || r.require_equal || r.positive || !r.equals_key.empty() ||
         !r.unless_present.empty() || !spec::IsSourceKey(r.flag_of_positive) ||
         (nd::KeyTypes(m, r.flag_of_positive, true) & nd::kInteger) == 0))
      return Invalid();
    if (!r.require_equal && r.equals_key.empty() && !r.positive && !r.boolean &&
        nd::KeyTypes(m, r.key, true) != nd::kInteger)
      return Invalid();
    if ((!r.equals_key.empty() &&
         ((nd::KeyTypes(m, r.key, true) & (nd::kInteger | nd::kReal)) == 0 ||
          (nd::KeyTypes(m, r.equals_key, true) & (nd::kInteger | nd::kReal)) == 0)) ||
        (r.positive && (nd::KeyTypes(m, r.key, true) & (nd::kInteger | nd::kReal)) == 0) ||
        (r.require_equal && (!spec::IsSourceKey(r.key) || Structural(r.key))))
      return Invalid();
  }
  return {};
}
}  // namespace llmp::import
