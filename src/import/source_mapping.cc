// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#include "import/source_mapping.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <string_view>

#include "spec/mappings/gemma2.h"
#include "spec/mappings/gemma3.h"
#include "spec/mappings/gemma4.h"

namespace llmp::import {
namespace nd = normalization_detail;
namespace {
using ErrorResult = std::unexpected<Error>;
ErrorResult Invalid() {
  return ErrorResult(Error{.rule = Rule::kUnsupported,
                           .reason = "invalid or unsupported source normalization table"});
}
constexpr std::array kAuxiliary{
    AuxiliaryNamespace{"general.", "container provenance"},
    AuxiliaryNamespace{"tokenizer.", "tokenizer metadata retained for the writer"},
    AuxiliaryNamespace{"quantize.imatrix.", "source quantization calibration provenance"}};
constexpr std::array kGemma2Requirements{ScalarRequirement{"gemma2.block_count", 46}};
constexpr std::array kGemma3Requirements{ScalarRequirement{"gemma3.block_count", 62}};

bool SameShape(std::string_view a, std::string_view b) {
  return (a.find(spec::kLayerField) == std::string_view::npos) ==
         (b.find(spec::kLayerField) == std::string_view::npos);
}
// Bounded finite pattern language: at most 1024 decimal layer values. Prefix
// and suffix rejection makes unrelated compiled rules constant-time.
bool Overlap(std::string_view a, std::string_view b) {
  const auto at = a.find(spec::kLayerField), bt = b.find(spec::kLayerField);
  if (at == std::string_view::npos) return nd::Match(b, a, nd::kLayerLimit).has_value();
  if (bt == std::string_view::npos) return nd::Match(a, b, nd::kLayerLimit).has_value();
  const auto ap = a.substr(0, at), bp = b.substr(0, bt);
  const auto as = a.substr(at + spec::kLayerField.size());
  const auto bs = b.substr(bt + spec::kLayerField.size());
  if ((!ap.starts_with(bp) && !bp.starts_with(ap)) || (!as.ends_with(bs) && !bs.ends_with(as)))
    return false;
  std::array<char, spec::kMaxText> text{};
  for (std::uint32_t layer = 0; layer < nd::kLayerLimit; ++layer) {
    auto* end = std::copy(ap.begin(), ap.end(), text.begin());
    end = std::to_chars(end, text.data() + text.size(), layer).ptr;
    end = std::copy(as.begin(), as.end(), end);
    if (nd::Match(b, {text.data(), static_cast<std::size_t>(end - text.data())}, nd::kLayerLimit))
      return true;
  }
  return false;
}
bool Structural(std::string_view key) {
  return key == "general.architecture" || key == "general.alignment" || key == "split.no" ||
         key == "split.count" || key == "split.tensors.count";
}
}  // namespace

const SourceMapping* FindSourceMapping(std::string_view container, std::string_view architecture) {
  static const std::array tables{SourceMapping{.container = "gguf",
                                               .architecture = &spec::mappings::Gemma2Mapping(),
                                               .auxiliary = kAuxiliary,
                                               .requirements = kGemma2Requirements},
                                 SourceMapping{.container = "gguf",
                                               .architecture = &spec::mappings::Gemma3Mapping(),
                                               .auxiliary = kAuxiliary,
                                               .requirements = kGemma3Requirements},
                                 SourceMapping{.container = "gguf",
                                               .architecture = &spec::mappings::Gemma4Mapping(),
                                               .auxiliary = kAuxiliary}};
  for (const auto& table : tables)
    if (container == table.container && architecture == table.architecture->architecture)
      return &table;
  return nullptr;
}

namespace normalization_detail {
std::uint8_t KeyTypes(const spec::Mapping& m, std::string_view key) {
  std::uint8_t types = 0;
  bool seen = false;
  const auto add = [&](std::uint8_t allowed) {
    types = seen ? static_cast<std::uint8_t>(types & allowed) : allowed;
    seen = true;
  };
  if (key == m.layer_count_key) add(kInteger);
  if (!m.layer_flags_key.empty() && key == m.layer_flags_key) add(kIntegers);
  for (const auto& rule : m.params) {
    if (key != rule.key || rule.key.empty()) continue;
    switch (rule.type) {
      case spec::ValueType::kInt:
        add(rule.per_layer ? kInteger | kIntegers : kInteger);
        break;
      case spec::ValueType::kReal:
        add(kInteger | kReal);
        break;
      case spec::ValueType::kBool:
        add(kFlag);
        break;
      case spec::ValueType::kString:
        add(kText);
        break;
      case spec::ValueType::kInts:
        return 0;
    }
  }
  for (const auto& rule : m.slots)
    if ((!rule.gate.empty() && key == rule.gate) ||
        (!rule.count_key.empty() && key == rule.count_key))
      add(kInteger);
  for (const auto& rule : m.policies)
    if (!rule.gate.empty() && key == rule.gate) add(kInteger);
  return types;
}

bool Pattern(std::string_view pattern) {
  if (pattern.empty() || pattern.size() > spec::kMaxText - 4) return false;
  const auto field = pattern.find(spec::kLayerField);
  if (field == std::string_view::npos) return spec::IsSourceKey(pattern);
  if (pattern.find(spec::kLayerField, field + spec::kLayerField.size()) != std::string_view::npos)
    return false;
  // IsSourceKey excludes braces; inspect the two literal pieces directly.
  for (std::size_t i = 0; i < pattern.size(); ++i) {
    if (i == field) {
      i += spec::kLayerField.size() - 1;
      continue;
    }
    const char c = pattern[i];
    if (c <= 32 || c >= 127 || c == '"' || c == '\\' || c == '{' || c == '}') return false;
  }
  return true;
}

std::optional<NameMatch> Match(std::string_view pattern, std::string_view name,
                               std::uint32_t layers) {
  const auto at = pattern.find(spec::kLayerField);
  if (at == std::string_view::npos)
    return name == pattern ? std::optional<NameMatch>(NameMatch{}) : std::nullopt;
  const auto prefix = pattern.substr(0, at), suffix = pattern.substr(at + spec::kLayerField.size());
  if (name.size() <= prefix.size() + suffix.size() || !name.starts_with(prefix) ||
      !name.ends_with(suffix))
    return std::nullopt;
  const auto digits = name.substr(prefix.size(), name.size() - prefix.size() - suffix.size());
  if (digits.size() > 4 || (digits.size() > 1 && digits.front() == '0')) return std::nullopt;
  std::uint32_t layer = 0;
  const auto parsed = std::from_chars(digits.data(), digits.data() + digits.size(), layer);
  if (parsed.ec != std::errc{} || parsed.ptr != digits.data() + digits.size() || layer >= layers)
    return std::nullopt;
  return NameMatch{digits};
}
}  // namespace normalization_detail

std::expected<void, Error> ValidateSourceMapping(const SourceMapping& table) {
  if (table.container != "gguf" || table.version == 0 || !table.architecture ||
      table.route != RouteOperation::kCopyPart || table.tensors.size() > nd::kRuleLimit ||
      table.keys.size() > 64 || table.auxiliary.size() > 8 || table.requirements.size() > 16)
    return Invalid();
  const auto& m = *table.architecture;
  const auto* registered = FindSourceMapping(table.container, m.architecture);
  if (!registered || registered->architecture != &m) return Invalid();
  for (const auto& required : registered->requirements) {
    bool found = false;
    for (const auto& rule : table.requirements)
      found |= rule.key == required.key && rule.reject_equal == required.reject_equal;
    if (!found) return Invalid();
  }
  if (m.slots.empty() || m.slots.size() > 8 || m.tensors.size() > nd::kRuleLimit ||
      m.params.size() > 64 || m.folds.size() > 8 || m.policies.size() > 8 || !m.reads.empty() ||
      !m.v0_assembly.empty() || m.pipeline != spec::PipelineKind::kLlm ||
      m.role != spec::StageRole::kTarget || !m.layer_type_key.empty() || !m.stage.empty() ||
      !spec::IsSourceKey(m.layer_count_key))
    return Invalid();
  for (const auto& slot : m.slots)
    if (!slot.layers_key.empty() || !slot.count_key.empty() || slot.per_layer ||
        slot.site == spec::Site::kSide || slot.site == spec::Site::kMixing)
      return Invalid();
  for (const auto& p : m.params)
    if ((p.per_layer && p.type != spec::ValueType::kInt) || p.type == spec::ValueType::kInts)
      return Invalid();
  for (const auto& r : m.tensors)
    if (!nd::Pattern(r.pattern) || r.target || r.any_part || r.unbound ||
        r.part != format::PartRole::kData)
      return Invalid();
  const auto count = m.tensors.size() + table.tensors.size();
  const auto spelling = [&](std::size_t i) {
    return i < m.tensors.size() ? m.tensors[i].pattern : table.tensors[i - m.tensors.size()].source;
  };
  for (const auto& a : table.tensors) {
    if (!nd::Pattern(a.source) || !SameShape(a.source, a.canonical)) return Invalid();
    std::size_t matches = 0;
    for (const auto& r : m.tensors) matches += r.pattern == a.canonical;
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
    return left && right && !left->gate.empty() && left->gate == right->gate &&
           left->gate_absent != right->gate_absent;
  };
  for (std::size_t i = 0; i < count; ++i)
    for (std::size_t j = 0; j < i; ++j)
      if (Overlap(spelling(i), spelling(j)) &&
          !(i < m.tensors.size() && exclusive(m.tensors[i], m.tensors[j])))
        return Invalid();
  for (std::size_t i = 0; i < table.keys.size(); ++i) {
    const auto& a = table.keys[i];
    if (!spec::IsSourceKey(a.source) || !nd::KeyTypes(m, a.canonical) ||
        nd::KeyTypes(m, a.source) || Structural(a.source))
      return Invalid();
    for (std::size_t j = 0; j < i; ++j)
      if (a.source == table.keys[j].source) return Invalid();
  }
  for (std::size_t i = 0; i < table.auxiliary.size(); ++i) {
    const auto& a = table.auxiliary[i];
    if (a.prefix.empty() || !a.prefix.ends_with('.') || !spec::IsSourceKey(a.prefix) ||
        a.reason.empty() || a.reason.size() > spec::kMaxText)
      return Invalid();
    if (m.layer_count_key.starts_with(a.prefix) ||
        (!m.layer_flags_key.empty() && m.layer_flags_key.starts_with(a.prefix)))
      return Invalid();
    for (const auto& slot : m.slots)
      if (!slot.gate.empty() && slot.gate.starts_with(a.prefix)) return Invalid();
    for (const auto& policy : m.policies)
      if (!policy.gate.empty() && policy.gate.starts_with(a.prefix)) return Invalid();
    for (const auto& p : m.params)
      if (!p.key.empty() && p.key.starts_with(a.prefix)) return Invalid();
    for (const auto& k : table.keys)
      if (k.source.starts_with(a.prefix)) return Invalid();
    for (std::size_t j = 0; j < i; ++j)
      if (a.prefix.starts_with(table.auxiliary[j].prefix) ||
          table.auxiliary[j].prefix.starts_with(a.prefix))
        return Invalid();
  }
  for (const auto& r : table.requirements)
    if (nd::KeyTypes(m, r.key) != nd::kInteger) return Invalid();
  return {};
}
}  // namespace llmp::import
