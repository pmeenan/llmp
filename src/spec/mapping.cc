// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "spec/mapping.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <map>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "artifact/artifact.h"
#include "artifact/format_map.h"
#include "format/encoding.h"
#include "format/tensor_format.h"
#include "spec/mappings/gemma2.h"
#include "spec/mappings/gemma3.h"
#include "spec/spec.h"

namespace llmp::spec {
namespace {

using Status = std::expected<void, std::string>;

std::unexpected<std::string> Refuse(std::string_view where, std::string_view what) {
  return std::unexpected(std::format("{}: {}", where, what));
}

bool Lower(char c) { return c >= 'a' && c <= 'z'; }
bool Digit(char c) { return c >= '0' && c <= '9'; }

bool IsName(std::string_view s) {
  return !s.empty() && s.size() <= kMaxText && Lower(s.front()) &&
         std::ranges::all_of(s, [](char c) { return Lower(c) || Digit(c) || c == '_'; });
}

bool IsKind(std::string_view s) {
  if (s.empty() || s.size() > kMaxText) {
    return false;
  }
  std::size_t start = 0;
  while (true) {
    const std::size_t dot = s.find('.', start);
    if (!IsName(s.substr(start, dot == std::string_view::npos ? dot : dot - start))) {
      return false;
    }
    if (dot == std::string_view::npos) {
      return true;
    }
    start = dot + 1;
  }
}

// A tensor name or source key: printable ASCII without space, quotes,
// backslash or braces.
bool IsText(std::string_view s) {
  return !s.empty() && s.size() <= kMaxText && std::ranges::all_of(s, [](char c) {
    return c > ' ' && c < 0x7F && c != '"' && c != '\\' && c != '{' && c != '}';
  });
}

bool IsPseudo(std::string_view slot) {
  return slot == kStageSlot || slot == kStreamSlot || slot == kHeadsSlot;
}

// A pattern: kLayerField exactly once when `layered`, never otherwise, and
// text around it.
bool IsPattern(std::string_view p, bool layered) {
  const std::size_t at = p.find(kLayerField);
  if (!layered) {
    return at == std::string_view::npos && IsText(p);
  }
  if (at == std::string_view::npos) {
    return false;
  }
  const std::string_view before = p.substr(0, at);
  const std::string_view after = p.substr(at + kLayerField.size());
  return p.size() <= kMaxText && (before.empty() || IsText(before)) &&
         (after.empty() || IsText(after)) && !(before.empty() && after.empty());
}

// The pattern with its layer.
std::string Expand(std::string_view pattern, std::uint32_t layer) {
  const std::size_t at = pattern.find(kLayerField);
  if (at == std::string_view::npos) {
    return std::string(pattern);
  }
  return std::format("{}{}{}", pattern.substr(0, at), layer,
                     pattern.substr(at + kLayerField.size()));
}

// Whether `name` is the pattern's, and at which layer (0 for an unlayered
// pattern): a decimal number without a leading zero, below `layers`.
std::optional<std::uint32_t> Match(std::string_view pattern, std::string_view name,
                                   std::uint32_t layers) {
  const std::size_t at = pattern.find(kLayerField);
  if (at == std::string_view::npos) {
    return pattern == name ? std::optional<std::uint32_t>(0) : std::nullopt;
  }
  const std::string_view before = pattern.substr(0, at);
  const std::string_view after = pattern.substr(at + kLayerField.size());
  if (name.size() <= before.size() + after.size() || !name.starts_with(before) ||
      !name.ends_with(after)) {
    return std::nullopt;
  }
  const std::string_view digits =
      name.substr(before.size(), name.size() - before.size() - after.size());
  if (digits.size() > 6 || (digits.size() > 1 && digits.front() == '0') ||
      !std::ranges::all_of(digits, Digit)) {
    return std::nullopt;
  }
  std::uint32_t layer = 0;
  for (const char c : digits) {
    layer = (layer * 10) + static_cast<std::uint32_t>(c - '0');
  }
  return layer < layers ? std::optional<std::uint32_t>(layer) : std::nullopt;
}

bool TypesWithin(std::string_view types, std::string_view letters) {
  return std::ranges::all_of(types, [&](char c) { return letters.contains(c); });
}

bool Applies(std::string_view types, char type) { return types.empty() || types.contains(type); }

const SlotRule* FindSlot(const Mapping& m, std::string_view slot) {
  const auto found = std::ranges::find(m.slots, slot, &SlotRule::slot);
  return found == m.slots.end() ? nullptr : &*found;
}

// The pseudo-slots' parameters and their types.
std::optional<ValueType> PseudoParam(std::string_view slot, std::string_view param) {
  if (slot == kStageSlot && param == "context") {
    return ValueType::kInt;
  }
  if (slot == kStreamSlot && (param == "width" || param == "streams")) {
    return ValueType::kInt;
  }
  if (slot == kHeadsSlot && param == "softcap") {
    return ValueType::kReal;
  }
  return std::nullopt;
}

// ---------------------------------------------------------------- values

using Values = std::map<std::string_view, const ParamValue*>;

std::expected<double, std::string> Transformed(double x, Transform t, std::string_view where) {
  double y = x;
  switch (t) {
    case Transform::kNone:
      break;
    case Transform::kReciprocal:
      y = 1.0 / x;
      break;
    case Transform::kSqrt:
      y = x >= 0.0 ? std::sqrt(x) : std::nan("");
      break;
    case Transform::kRsqrt:
      y = x > 0.0 ? 1.0 / std::sqrt(x) : std::nan("");
      break;
  }
  if (!std::isfinite(y)) {
    return Refuse(where, "a value outside its transform's domain");
  }
  return y;
}

// The rule's value: its constant, or its key's converted; nothing when the
// key is absent.
std::expected<std::optional<ParamValue>, std::string> Evaluate(const ParamRule& r,
                                                               const Values& values,
                                                               std::string_view where) {
  if (r.key.empty()) {
    switch (r.type) {
      case ValueType::kInt:
        return ParamValue(r.value.integer);
      case ValueType::kReal: {
        auto y = Transformed(r.value.real, r.transform, where);
        if (!y) {
          return std::unexpected(y.error());
        }
        return ParamValue(*y);
      }
      case ValueType::kBool:
        return ParamValue(r.value.flag);
      case ValueType::kString:
        return ParamValue(std::string(r.value.text));
    }
  }
  const auto found = values.find(r.key);
  if (found == values.end()) {
    return std::optional<ParamValue>();
  }
  const ParamValue& v = *found->second;
  const std::string at = std::format("{} (from {})", where, r.key);
  switch (r.type) {
    case ValueType::kInt:
      if (const auto* i = std::get_if<std::int64_t>(&v)) {
        return ParamValue(*i);
      }
      return Refuse(at, "not an integer");
    case ValueType::kReal: {
      double x = 0.0;
      if (const auto* i = std::get_if<std::int64_t>(&v)) {
        x = static_cast<double>(*i);
      } else if (const auto* d = std::get_if<double>(&v)) {
        x = *d;
      } else {
        return Refuse(at, "not a number");
      }
      auto y = Transformed(x, r.transform, at);
      if (!y) {
        return std::unexpected(y.error());
      }
      return ParamValue(*y);
    }
    case ValueType::kBool:
      if (const auto* b = std::get_if<bool>(&v)) {
        return ParamValue(*b);
      }
      return Refuse(at, "not a flag");
    case ValueType::kString:
      if (const auto* s = std::get_if<std::string>(&v); s != nullptr && IsName(*s)) {
        return ParamValue(*s);
      }
      return Refuse(at, "not a name");
  }
  return Refuse(at, "an unknown value type");
}

// The parameter's value at a layer type (0: not a layer slot), by its
// rules in order; nothing when no rule applies, or none resolves and one
// is optional.
std::expected<std::optional<ParamValue>, std::string> Resolve(const Mapping& m,
                                                              std::string_view slot,
                                                              std::string_view param, char type,
                                                              const Values& values) {
  const std::string where = std::format("{}.{}", slot, param);
  bool applied = false;
  bool optional = false;
  for (const ParamRule& r : m.params) {
    if (r.slot != slot || r.param != param || (type != 0 && !Applies(r.layer_types, type))) {
      continue;
    }
    applied = true;
    optional = optional || r.optional;
    auto v = Evaluate(r, values, where);
    if (!v) {
      return std::unexpected(v.error());
    }
    if (*v) {
      return v;
    }
  }
  if (applied && !optional) {
    return Refuse(where, type != 0 ? std::format("no value for layer type '{}'", type)
                                   : std::string("no value"));
  }
  return std::optional<ParamValue>();
}

std::expected<void, std::string> CheckRequirements(const Mapping& m, const Values& values) {
  for (const ParamRule& r : m.params) {
    if (!r.slot.empty()) {
      continue;
    }
    auto v = Evaluate(r, values, std::format("requirement {}", r.key));
    if (!v) {
      return std::unexpected(v.error());
    }
    if (!*v) {
      if (!r.optional) {
        return Refuse(r.key, "absent from the source");
      }
      continue;
    }
    ParamRule constant = r;
    constant.key = {};
    constant.transform = Transform::kNone;
    auto want = Evaluate(constant, values, r.key);
    if (!want || **v != **want) {
      return Refuse(r.key, "not the value the mapping's components implement");
    }
  }
  return {};
}

// ---------------------------------------------------------------- synthesis

class Synthesis {
 public:
  Synthesis(const Mapping& m, const Source& s, const Values& v, std::uint32_t layers)
      : m_(m), source_(s), values_(v), layers_(layers) {}

  std::expected<void, std::string> Bind() {
    for (const SourceTensor& t : source_.tensors) {
      names_.insert(t.name);
    }
    for (const SourceTensor& t : source_.tensors) {
      std::optional<std::pair<std::size_t, std::uint32_t>> match;
      for (std::size_t i = 0; i < m_.tensors.size(); ++i) {
        const auto layer = Match(m_.tensors[i].pattern, t.name, layers_);
        if (!layer) {
          continue;
        }
        if (match) {
          return Refuse(t.name, "matches two of the mapping's tensor rules");
        }
        match = std::pair(i, *layer);
      }
      if (!match) {
        return Refuse(t.name, "matches none of the mapping's tensor rules");
      }
      const TensorRule& rule = m_.tensors[match->first];
      const SlotRule& slot = *FindSlot(m_, rule.slot);
      if (slot.site == Site::kLayer && !Applies(slot.layer_types, Type(match->second))) {
        return Refuse(t.name, std::format("layer {} has no {}", match->second, rule.slot));
      }
      if (t.part != rule.part || t.experts != rule.experts) {
        return Refuse(t.name,
                      std::format("holds {}{}, not the rule's {}{}", format::PartRoleName(t.part),
                                  t.experts ? " experts" : "", format::PartRoleName(rule.part),
                                  rule.experts ? " experts" : ""));
      }
      if (!bound_.emplace(*match, t.name).second) {
        return Refuse(t.name, "binds a role another tensor binds");
      }
    }
    return {};
  }

  char Type(std::uint32_t layer) const { return m_.layer_types.at(layer % m_.layer_types.size()); }

  // The component of a slot at a layer (0 for the others).
  std::expected<ComponentRef, std::string> Component(const SlotRule& slot, std::uint32_t layer) {
    ComponentRef c{.kind = std::string(slot.kind), .version = slot.version};
    const char type = slot.site == Site::kLayer ? Type(layer) : char{0};
    std::set<std::string_view> seen;
    for (const ParamRule& r : m_.params) {
      if (r.slot != slot.slot || (type != 0 && !Applies(r.layer_types, type)) ||
          !seen.insert(r.param).second) {
        continue;
      }
      auto v = Resolve(m_, slot.slot, r.param, type, values_);
      if (!v) {
        return std::unexpected(v.error());
      }
      if (*v) {
        c.params.push_back({.name = std::string(r.param), .value = std::move(**v)});
      }
    }
    for (std::size_t i = 0; i < m_.tensors.size(); ++i) {
      const TensorRule& rule = m_.tensors[i];
      if (rule.slot != slot.slot) {
        continue;
      }
      std::string tensor;
      if (const auto found = bound_.find({i, layer}); found != bound_.end()) {
        tensor = found->second;
      } else if (!rule.fallback.empty()) {
        tensor = Expand(rule.fallback, layer);
        if (!names_.contains(tensor)) {
          return Refuse(Expand(rule.pattern, layer),
                        std::format("absent, and so is its fallback {}", tensor));
        }
      } else if (rule.optional) {
        continue;
      } else {
        return Refuse(Expand(rule.pattern, layer), "absent from the source");
      }
      c.weights.push_back(WeightBinding{.role = std::string(rule.role),
                                        .tensor = tensor,
                                        .part = rule.part,
                                        .experts = rule.experts});
    }
    return c;
  }

  std::expected<std::optional<ComponentRef>, std::string> Only(Site site) {
    for (const SlotRule& s : m_.slots) {
      if (s.site == site) {
        auto c = Component(s, 0);
        if (!c) {
          return std::unexpected(c.error());
        }
        return std::optional<ComponentRef>(std::move(*c));
      }
    }
    return std::optional<ComponentRef>();
  }

 private:
  const Mapping& m_;
  const Source& source_;
  const Values& values_;
  std::uint32_t layers_;
  std::set<std::string_view> names_;
  std::map<std::pair<std::size_t, std::uint32_t>, std::string> bound_;  // (rule, layer)
};

std::expected<std::uint64_t, std::string> Count(const ParamValue* v, std::string_view where,
                                                std::uint64_t least, std::uint64_t most) {
  const auto* i = v != nullptr ? std::get_if<std::int64_t>(v) : nullptr;
  if (i == nullptr || *i < 0 || static_cast<std::uint64_t>(*i) < least ||
      static_cast<std::uint64_t>(*i) > most) {
    return Refuse(where, std::format("not an integer from {} to {}", least, most));
  }
  return static_cast<std::uint64_t>(*i);
}

}  // namespace

bool IsSourceKey(std::string_view key) { return IsText(key); }

std::span<const Mapping* const> Mappings() {
  static const std::array<const Mapping*, 2> all = {&mappings::Gemma2Mapping(),
                                                    &mappings::Gemma3Mapping()};
  return all;
}

const Mapping* FindMapping(std::string_view architecture) {
  for (const Mapping* m : Mappings()) {
    if (m->architecture == architecture) {
      return m;
    }
  }
  return nullptr;
}

Status ValidateMapping(const Mapping& m) {
  const std::string where = std::format("mapping {}", m.architecture);
  if (!IsName(m.architecture) || m.version == 0 ||
      static_cast<std::uint8_t>(m.pipeline) > static_cast<std::uint8_t>(PipelineKind::kDecision)) {
    return Refuse(where, "an invalid architecture, version or pipeline");
  }
  if (!IsText(m.layer_count_key) || m.layer_types.empty() ||
      m.layer_types.size() > kMaxLayerTypes || !std::ranges::all_of(m.layer_types, Lower)) {
    return Refuse(where, "an invalid layer count key or layer types");
  }
  if (m.slots.size() + m.tensors.size() + m.params.size() + m.folds.size() > kMaxMappingRules) {
    return Refuse(where, std::format("more than {} rules", kMaxMappingRules));
  }
  std::array<int, 4> sites{};
  std::set<std::string_view> slots;
  for (const SlotRule& s : m.slots) {
    const std::string at = std::format("{} slot {}", where, s.slot);
    if (!IsName(s.slot) || IsPseudo(s.slot) || !slots.insert(s.slot).second) {
      return Refuse(at, "not a name, a pseudo-slot's, or named twice");
    }
    if (!IsKind(s.kind) || s.version == 0 ||
        static_cast<std::uint8_t>(s.site) > static_cast<std::uint8_t>(Site::kOutput)) {
      return Refuse(at, "an invalid kind, version or site");
    }
    if (s.site != Site::kLayer ? !s.layer_types.empty()
                               : !TypesWithin(s.layer_types, m.layer_types)) {
      return Refuse(at, "layer types off a layer slot, or not the mapping's");
    }
    if (s.site != Site::kLayer && ++sites.at(static_cast<std::size_t>(s.site)) > 1) {
      return Refuse(at, "a second embedding, final norm or output");
    }
  }
  std::set<std::string_view> patterns;
  std::set<std::tuple<std::string_view, std::string_view, format::PartRole>> roles;
  for (const TensorRule& t : m.tensors) {
    const std::string at = std::format("{} tensor {}", where, t.pattern);
    const SlotRule* slot = FindSlot(m, t.slot);
    if (slot == nullptr) {
      return Refuse(at, "a slot the mapping does not have");
    }
    const bool layered = slot->site == Site::kLayer;
    if (!IsPattern(t.pattern, layered) || !patterns.insert(t.pattern).second) {
      return Refuse(at, "not a pattern of its slot's site, or twice");
    }
    if (!t.fallback.empty() && !IsPattern(t.fallback, layered)) {
      return Refuse(at, "a fallback that is not a pattern of its slot's site");
    }
    if (!IsName(t.role) ||
        static_cast<std::uint8_t>(t.part) > static_cast<std::uint8_t>(format::PartRole::kZeros) ||
        !roles.emplace(t.slot, t.role, t.part).second) {
      return Refuse(at, "an invalid role or part, or one bound twice");
    }
  }
  for (const ParamRule& r : m.params) {
    const std::string at = std::format("{} parameter {}.{}", where, r.slot, r.param);
    if (static_cast<std::uint8_t>(r.type) > static_cast<std::uint8_t>(ValueType::kString) ||
        static_cast<std::uint8_t>(r.transform) > static_cast<std::uint8_t>(Transform::kRsqrt) ||
        (r.transform != Transform::kNone && r.type != ValueType::kReal)) {
      return Refuse(at, "an unknown type or transform, or a transform of a non-real");
    }
    if (!r.key.empty() && !IsText(r.key)) {
      return Refuse(at, "an invalid key");
    }
    if ((r.key.empty() || r.slot.empty()) && r.type == ValueType::kString &&
        !IsName(r.value.text)) {
      return Refuse(at, "a constant string that is not a name");
    }
    if ((r.key.empty() || r.slot.empty()) && r.type == ValueType::kReal &&
        !std::isfinite(r.value.real)) {
      return Refuse(at, "a constant real that is not finite");
    }
    if (r.slot.empty()) {
      if (r.key.empty() || !r.param.empty() || !r.layer_types.empty() ||
          r.transform != Transform::kNone) {
        return Refuse(at, "a requirement is a key and its value");
      }
      continue;
    }
    if (!IsName(r.param)) {
      return Refuse(at, "not a name");
    }
    if (IsPseudo(r.slot)) {
      if (PseudoParam(r.slot, r.param) != r.type || !r.layer_types.empty()) {
        return Refuse(at, "not one of the pseudo-slot's parameters and types");
      }
      continue;
    }
    const SlotRule* slot = FindSlot(m, r.slot);
    if (slot == nullptr) {
      return Refuse(at, "a slot the mapping does not have");
    }
    if (!r.layer_types.empty() &&
        (slot->site != Site::kLayer || !TypesWithin(r.layer_types, m.layer_types))) {
      return Refuse(at, "layer types off a layer slot, or not the mapping's");
    }
  }
  for (const FoldRule& f : m.folds) {
    if (FindFold(f.fold) == nullptr || !IsName(f.container)) {
      return Refuse(std::format("{} fold {}", where, f.fold),
                    "not in the fold catalog, or an invalid container");
    }
  }
  return {};
}

std::expected<Source, std::string> SourceOfV0(const artifact::Artifact& artifact) {
  Source s;
  s.architecture = artifact.model().architecture;
  const auto sources = artifact.sources();
  const auto all = [&](auto pred) {
    return !sources.empty() && std::ranges::all_of(sources, pred);
  };
  if (all([](const artifact::Source& f) { return f.name.ends_with(".gguf"); })) {
    s.container = "gguf";
  } else if (all([](const artifact::Source& f) {
               return f.name.ends_with(".safetensors") || f.name.ends_with(".json");
             })) {
    s.container = "safetensors";
  } else {
    s.container = "other";
  }
  for (const artifact::Resource& r : artifact.resources()) {
    auto f = artifact::FormatOf(r.repr);
    if (!f) {
      return Refuse(r.name,
                    std::format("no tensor format ({})", format::FormatErrorName(f.error())));
    }
    for (const std::string& role : r.roles) {
      s.tensors.push_back(
          {.name = role, .format = *f, .part = artifact::V0PartRole(r.repr), .experts = false});
    }
  }
  for (const artifact::ExpertArray& a : artifact.expert_arrays()) {
    auto f = artifact::FormatOf(a.repr);
    if (!f || f->shape.rank > 2) {
      return Refuse(a.name, "no two-dimensional slice format");
    }
    f->shape.ne[2] = a.count;
    f->shape.rank = 3;
    s.tensors.push_back(
        {.name = a.name, .format = *f, .part = artifact::V0PartRole(a.repr), .experts = true});
  }
  if (s.tensors.size() > kMaxSourceTensors) {
    return Refuse("source", std::format("more than {} tensors", kMaxSourceTensors));
  }
  return s;
}

std::expected<ModelSpec, std::string> Synthesize(const Mapping& mapping, const Source& source,
                                                 std::span<const SourceValue> values) {
  if (auto ok = ValidateMapping(mapping); !ok) {
    return std::unexpected(ok.error());
  }
  if (source.architecture != mapping.architecture) {
    return Refuse("source", std::format("architecture {}, not the mapping's {}",
                                        source.architecture, mapping.architecture));
  }
  if (source.tensors.size() > kMaxSourceTensors || values.size() > kMaxSourceValues) {
    return Refuse("source", "too many tensors or values");
  }
  Values by_key;
  for (const SourceValue& v : values) {
    if (!IsText(v.key) || !by_key.emplace(v.key, &v.value).second) {
      return Refuse(v.key, "an invalid key, or a key twice");
    }
  }
  const auto found = by_key.find(mapping.layer_count_key);
  const auto layers = Count(found == by_key.end() ? nullptr : found->second,
                            mapping.layer_count_key, 1, kMaxLayers);
  if (!layers) {
    return std::unexpected(layers.error());
  }
  if (auto ok = CheckRequirements(mapping, by_key); !ok) {
    return std::unexpected(ok.error());
  }
  Synthesis syn(mapping, source, by_key, static_cast<std::uint32_t>(*layers));
  if (auto ok = syn.Bind(); !ok) {
    return std::unexpected(ok.error());
  }

  Stage stage{.name = "target", .role = StageRole::kTarget};
  const auto pseudo =
      [&](std::string_view slot,
          std::string_view param) -> std::expected<std::optional<ParamValue>, std::string> {
    return Resolve(mapping, slot, param, 0, by_key);
  };
  const auto count = [&](std::string_view slot, std::string_view param, std::uint64_t least,
                         std::uint64_t most,
                         std::uint64_t absent) -> std::expected<std::uint64_t, std::string> {
    auto v = pseudo(slot, param);
    if (!v) {
      return std::unexpected(v.error());
    }
    if (!*v) {
      return absent;
    }
    return Count(&**v, std::format("{}.{}", slot, param), least, most);
  };
  auto context = count(kStageSlot, "context", 1, kMaxContext, 0);
  auto width = count(kStreamSlot, "width", 1, kMaxWidth, 0);
  auto streams = count(kStreamSlot, "streams", 1, kMaxStreams, 1);
  if (!context || !width || !streams) {
    return std::unexpected(!context ? context.error() : (!width ? width.error() : streams.error()));
  }
  stage.context = static_cast<std::uint32_t>(*context);
  stage.stream.width = *width;
  stage.stream.streams = static_cast<std::uint32_t>(*streams);
  auto softcap = pseudo(kHeadsSlot, "softcap");
  if (!softcap) {
    return std::unexpected(softcap.error());
  }
  if (*softcap) {
    stage.heads.softcap = std::get<double>(**softcap);
  }

  auto embedding = syn.Only(Site::kEmbedding);
  if (!embedding) {
    return std::unexpected(embedding.error());
  }
  stage.embedding = std::move(*embedding);
  stage.layers.resize(*layers);
  for (std::uint32_t il = 0; il < *layers; ++il) {
    for (const SlotRule& s : mapping.slots) {
      if (s.site != Site::kLayer || !Applies(s.layer_types, syn.Type(il))) {
        continue;
      }
      auto c = syn.Component(s, il);
      if (!c) {
        return std::unexpected(std::format("layer {}: {}", il, c.error()));
      }
      stage.layers[il].components.push_back(std::move(*c));
    }
  }
  auto final_norm = syn.Only(Site::kFinalNorm);
  auto output = syn.Only(Site::kOutput);
  if (!final_norm || !output) {
    return std::unexpected(!final_norm ? final_norm.error() : output.error());
  }
  stage.heads.final_norm = std::move(*final_norm);
  stage.heads.output = std::move(*output);

  ModelSpec spec{.pipeline = mapping.pipeline};
  spec.stages.push_back(std::move(stage));
  for (const FoldRule& f : mapping.folds) {
    if (f.container == source.container && !std::ranges::contains(spec.folds, f.fold)) {
      spec.folds.emplace_back(f.fold);
    }
  }
  Canonicalize(spec);
  if (auto ok = Validate(spec); !ok) {
    return std::unexpected(ok.error());
  }
  return spec;
}

std::expected<ModelSpec, std::string> SynthesizeV0(const artifact::Artifact& artifact,
                                                   const Mapping& mapping,
                                                   std::span<const SourceValue> values) {
  auto source = SourceOfV0(artifact);
  if (!source) {
    return std::unexpected(source.error());
  }
  return Synthesize(mapping, *source, values);
}

}  // namespace llmp::spec
