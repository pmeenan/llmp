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
#include <limits>
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
#include "spec/mappings/dsv4.h"
#include "spec/mappings/gemma2.h"
#include "spec/mappings/gemma3.h"
#include "spec/mappings/gemma4.h"
#include "spec/mappings/qwen38.h"
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

// An architecture as GGUF names it: a name that may also hold '-'
// ("gemma4-assistant").
bool IsArchitecture(std::string_view s) {
  return !s.empty() && s.size() <= kMaxText && Lower(s.front()) &&
         std::ranges::all_of(s,
                             [](char c) { return Lower(c) || Digit(c) || c == '_' || c == '-'; });
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

// Whether a fold rule's converter names one of `identity`'s '+'-joined
// parts (Source::converter: "name version", the version's parts joined by
// '+'): the part begins with it ("modelopt_qwen38-" names the part
// "modelopt_qwen38-031ed8750c9da55d", never "xmodelopt_qwen38-…").
bool NamesPart(std::string_view identity, std::string_view rule) {
  while (true) {
    const std::size_t end = identity.find('+');
    if (identity.substr(0, end).starts_with(rule)) return true;
    if (end == std::string_view::npos) return false;
    identity.remove_prefix(end + 1);
  }
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

// Where a per-layer rule reads: the layer and the stage's layer count.
struct LayerAt {
  std::uint32_t layer = 0;
  std::uint32_t layers = 0;
};

// The rule's value: its constant, or its key's converted (a per-layer
// key's entry for the layer); nothing when the key is absent.
std::expected<std::optional<ParamValue>, std::string> Evaluate(const ParamRule& r,
                                                               const Values& values,
                                                               std::string_view where,
                                                               LayerAt layer = {}) {
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
        // A layer slot's constant may name its layer (kLayerField).
        return ParamValue(Expand(r.value.text, layer.layer));
      case ValueType::kInts:
        return Refuse(where, "a list without its key");
    }
  }
  const auto found = values.find(r.key);
  if (found == values.end()) {
    return std::optional<ParamValue>();
  }
  const std::string at = std::format("{} (from {})", where, r.key);
  // A per-layer key's entry for the layer (a list of one value a layer),
  // or its one value for every layer.
  ParamValue entry = *found->second;
  if (r.per_layer) {
    const auto pick = [&](const auto& list) -> std::expected<void, std::string> {
      if (list.size() != layer.layers || layer.layer >= list.size()) {
        return Refuse(at,
                      std::format("a list of {} values for {} layers", list.size(), layer.layers));
      }
      entry = list[layer.layer];
      return {};
    };
    if (const auto* is = std::get_if<std::vector<std::int64_t>>(&*found->second)) {
      if (auto ok = pick(*is); !ok) {
        return std::unexpected(ok.error());
      }
    } else if (const auto* ds = std::get_if<std::vector<double>>(&*found->second)) {
      if (auto ok = pick(*ds); !ok) {
        return std::unexpected(ok.error());
      }
    }
  }
  const ParamValue& v = entry;
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
    case ValueType::kInts:
      if (const auto* is = std::get_if<std::vector<std::int64_t>>(&v)) {
        return ParamValue(*is);
      }
      return Refuse(at, "not a list of integers");
  }
  return Refuse(at, "an unknown value type");
}

// The parameter's value at a layer type (0: not a layer slot), by its
// rules in order; nothing when no rule applies, or none resolves and one
// is optional.
std::expected<std::optional<ParamValue>, std::string> Resolve(const Mapping& m,
                                                              std::string_view slot,
                                                              std::string_view param, char type,
                                                              const Values& values,
                                                              LayerAt layer = {}) {
  const std::string where = std::format("{}.{}", slot, param);
  bool applied = false;
  bool optional = false;
  for (const ParamRule& r : m.params) {
    if (r.slot != slot || r.param != param || (type != 0 && !Applies(r.layer_types, type))) {
      continue;
    }
    applied = true;
    optional = optional || r.optional;
    auto v = Evaluate(r, values, where, layer);
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

// The layer a layer slot's rule without kLayerField binds under: every
// layer of its types binds that one tensor.
constexpr std::uint32_t kShared = ~std::uint32_t{0};

bool Layered(std::string_view pattern) { return pattern.contains(kLayerField); }

// Whether a gate holds for the values: the key an integer above zero, or
// (absent) the key absent or zero; an empty gate always holds.
std::expected<bool, std::string> Holds(std::string_view gate, bool absent, const Values& values) {
  if (gate.empty()) {
    return true;
  }
  std::int64_t n = 0;
  if (const auto found = values.find(gate); found != values.end()) {
    // A flag (a v0 binding's fact, FlagValue) holds when set.
    if (const auto* flag = std::get_if<bool>(found->second); flag != nullptr) {
      n = *flag ? 1 : 0;
    } else {
      const auto* i = std::get_if<std::int64_t>(found->second);
      if (i == nullptr || *i < 0) {
        return Refuse(gate, "a gate that is not a flag or an integer of at least zero");
      }
      n = *i;
    }
  }
  return absent ? n == 0 : n > 0;
}

// Two slots of which a checkpoint (or a layer) has at most one: gated by
// one key oppositely, or at the layers on either side of one count.
bool Exclusive(const SlotRule& a, const SlotRule& b) {
  return (!a.gate.empty() && a.gate == b.gate && a.gate_absent != b.gate_absent) ||
         (!a.count_key.empty() && a.count_key == b.count_key && a.count_above != b.count_above);
}

// The layers a source key names: an integer or a list of integers, each
// below `layers`.
std::expected<std::set<std::uint32_t>, std::string> LayersOf(std::string_view key,
                                                             const Values& values,
                                                             std::uint32_t layers) {
  std::set<std::uint32_t> out;
  const auto found = values.find(key);
  if (found == values.end()) return out;
  std::vector<std::int64_t> list;
  if (const auto* i = std::get_if<std::int64_t>(found->second)) {
    list.push_back(*i);
  } else if (const auto* is = std::get_if<std::vector<std::int64_t>>(found->second)) {
    list = *is;
  } else {
    return Refuse(key, "not a layer or a list of layers");
  }
  for (const std::int64_t l : list) {
    if (l < 0 || l >= static_cast<std::int64_t>(layers)) {
      return Refuse(key, std::format("a layer {} past the {} layers", l, layers));
    }
    out.insert(static_cast<std::uint32_t>(l));
  }
  return out;
}

std::expected<std::uint64_t, std::string> Count(const ParamValue* v, std::string_view where,
                                                std::uint64_t least, std::uint64_t most);

class Synthesis {
 public:
  Synthesis(const Mapping& m, const Source& s, const Values& v, std::uint32_t layers,
            std::string_view target_stage, std::string types)
      : m_(m),
        source_(s),
        values_(v),
        layers_(layers),
        target_stage_(target_stage),
        types_(std::move(types)) {}

  // The slots the source has, by their gates, and the layers a slot with a
  // layers or count key is at.
  std::expected<void, std::string> Gate() {
    for (const SlotRule& slot : m_.slots) {
      auto present = Holds(slot.gate, slot.gate_absent, values_);
      if (!present) {
        return std::unexpected(present.error());
      }
      if (*present) {
        present_.insert(slot.slot);
      }
      if (!slot.layers_key.empty()) {
        auto at = LayersOf(slot.layers_key, values_, layers_);
        if (!at) {
          return std::unexpected(at.error());
        }
        at_.emplace(slot.slot, std::move(*at));
      }
      if (!slot.count_key.empty()) {
        std::uint64_t n = 0;
        if (const auto found = values_.find(slot.count_key); found != values_.end()) {
          auto count = Count(found->second, slot.count_key, 0, layers_);
          if (!count) return std::unexpected(count.error());
          n = *count;
        }
        std::set<std::uint32_t> at;
        for (std::uint32_t il = 0; il < layers_; ++il) {
          if ((il < n) != slot.count_above) at.insert(il);
        }
        at_.emplace(slot.slot, std::move(at));
      }
    }
    return {};
  }
  bool Present(std::string_view slot) const { return present_.contains(slot); }
  // Whether a layer slot is at layer `il` (its types and its layers key).
  bool At(const SlotRule& s, std::uint32_t il) const {
    if (!Applies(s.layer_types, Type(il)) || !Present(s.slot)) {
      return false;
    }
    const auto found = at_.find(s.slot);
    return (s.layers_key.empty() && s.count_key.empty()) ||
           (found != at_.end() && found->second.contains(il));
  }
  const std::set<std::uint32_t>& Layers(std::string_view slot) const {
    static const std::set<std::uint32_t> none;
    const auto found = at_.find(slot);
    return found == at_.end() ? none : found->second;
  }

  std::expected<void, std::string> Bind() {
    for (const SourceTensor& t : source_.tensors) {
      names_.emplace(t.name, &t);
    }
    for (const SourceTensor& t : source_.tensors) {
      std::optional<std::pair<std::size_t, std::uint32_t>> match;
      for (std::size_t i = 0; i < m_.tensors.size(); ++i) {
        const TensorRule& rule = m_.tensors[i];
        if (rule.target || !Present(rule.slot)) {
          continue;  // the target's tensor, or a slot the source has not
        }
        const auto layer = Match(rule.pattern, t.name, layers_);
        if (!layer) {
          continue;
        }
        // A count's slot binds only at its own layers (its counterpart's
        // rule takes the others').
        const SlotRule& at = *FindSlot(m_, rule.slot);
        if (!at.count_key.empty() && at.site == Site::kLayer && Layered(rule.pattern) &&
            !At(at, *layer)) {
          continue;
        }
        if (match) {
          return Refuse(t.name, "matches two of the mapping's tensor rules");
        }
        match = std::pair(i, Layered(rule.pattern) ? *layer : kShared);
      }
      if (!match) {
        return Refuse(t.name, "matches none of the mapping's tensor rules of the slots it has");
      }
      const TensorRule& rule = m_.tensors[match->first];
      const SlotRule& slot = *FindSlot(m_, rule.slot);
      if (match->second != kShared && slot.site == Site::kLayer &&
          (!At(slot, match->second) || !Applies(rule.layer_types, Type(match->second)))) {
        return Refuse(t.name, std::format("layer {} of type '{}' has no {} {}", match->second,
                                          Type(match->second), rule.slot, rule.role));
      }
      // A per-layer side input's table, at a layer it is at.
      if (match->second != kShared && slot.site == Site::kSide &&
          !Layers(slot.slot).contains(match->second)) {
        return Refuse(t.name,
                      std::format("layer {} has no side input {}", match->second, rule.slot));
      }
      if ((!rule.any_part && t.part != rule.part) || t.experts != rule.experts) {
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

  char Type(std::uint32_t layer) const { return types_.at(layer % types_.size()); }

  // The component of a slot at a layer (0 for the others).
  std::expected<ComponentRef, std::string> Component(const SlotRule& slot, std::uint32_t layer) {
    ComponentRef c{.kind = std::string(slot.kind), .version = slot.version};
    const bool layered = slot.site == Site::kLayer;
    const char type = layered ? Type(layer) : char{0};
    const LayerAt at{.layer = layer, .layers = layers_};
    std::set<std::string_view> seen;
    for (const ParamRule& r : m_.params) {
      if (r.slot != slot.slot || (type != 0 && !Applies(r.layer_types, type)) ||
          !seen.insert(r.param).second) {
        continue;
      }
      auto v = Resolve(m_, slot.slot, r.param, type, values_, at);
      if (!v) {
        return std::unexpected(v.error());
      }
      if (*v) {
        c.params.push_back({.name = std::string(r.param), .value = std::move(**v)});
      }
    }
    for (std::size_t i = 0; i < m_.tensors.size(); ++i) {
      const TensorRule& rule = m_.tensors[i];
      if (rule.slot != slot.slot || rule.unbound || (layered && !Applies(rule.layer_types, type))) {
        continue;
      }
      if (rule.target) {
        // The target's tensor, as its checkpoint names it; the spec's
        // validation binds it in the target's artifact.
        c.weights.push_back(WeightBinding{.role = std::string(rule.role),
                                          .tensor = Expand(rule.pattern, layer),
                                          .part = rule.part,
                                          .experts = rule.experts,
                                          .stage = std::string(target_stage_)});
        continue;
      }
      const std::uint32_t key = Layered(rule.pattern) ? layer : kShared;
      std::string tensor;
      if (const auto found = bound_.find({i, key}); found != bound_.end()) {
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
      // The part it holds: the rule's, or with any_part the tensor's own.
      const auto held = names_.find(tensor);
      const format::PartRole part =
          rule.any_part && held != names_.end() ? held->second->part : rule.part;
      c.weights.push_back(WeightBinding{
          .role = std::string(rule.role), .tensor = tensor, .part = part, .experts = rule.experts});
    }
    return c;
  }

  std::expected<std::optional<ComponentRef>, std::string> Only(Site site) {
    for (const SlotRule& s : m_.slots) {
      if (s.site == site && Present(s.slot)) {
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
  std::string_view target_stage_;
  std::map<std::string_view, const SourceTensor*> names_;
  std::set<std::string_view> present_;
  std::map<std::string_view, std::set<std::uint32_t>> at_;
  std::map<std::pair<std::size_t, std::uint32_t>, std::string> bound_;  // (rule, layer)
  std::string types_;  // a letter a layer, or the mapping's repeating pattern
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

// Each layer's type: by the layer type key's values (layer_type_key), or
// the mapping's repeating pattern.
std::expected<std::string, std::string> LayerTypes(const Mapping& m, const Values& values,
                                                   std::uint64_t layers) {
  if (m.layer_type_key.empty()) return std::string(m.layer_types);
  const auto found = values.find(m.layer_type_key);
  const auto* list =
      found == values.end() ? nullptr : std::get_if<std::vector<std::int64_t>>(found->second);
  if (list == nullptr || list->size() < layers) {
    return Refuse(m.layer_type_key, std::format("not a value for each of {} layers", layers));
  }
  std::string types;
  for (std::uint64_t il = 0; il < layers; ++il) {
    const auto code = std::ranges::find(m.layer_type_codes, (*list)[il], &LayerTypeCode::value);
    if (code == m.layer_type_codes.end()) {
      return Refuse(m.layer_type_key, std::format("layer {}'s {} is none of the mapping's layer "
                                                  "types",
                                                  il, (*list)[il]));
    }
    types.push_back(code->type);
  }
  return types;
}

// A layer-flag list (layer_flags_key): one flag a layer, set exactly at
// the layers of layer_flag_type.
std::expected<void, std::string> CheckLayerFlags(const Mapping& m, const Values& values,
                                                 std::uint64_t layers) {
  if (m.layer_flags_key.empty()) {
    return {};
  }
  const auto found = values.find(m.layer_flags_key);
  if (found == values.end()) {
    return {};
  }
  const auto* flags = std::get_if<std::vector<std::int64_t>>(found->second);
  if (flags == nullptr || flags->size() != layers) {
    return Refuse(m.layer_flags_key, std::format("not a flag for each of {} layers", layers));
  }
  for (std::uint64_t il = 0; il < layers; ++il) {
    const bool typed = m.layer_types.at(il % m.layer_types.size()) == m.layer_flag_type;
    if (((*flags)[il] != 0) != typed) {
      return Refuse(m.layer_flags_key, std::format("layer {} is not of the mapping's layer types "
                                                   "(\"{}\")",
                                                   il, m.layer_types));
    }
  }
  return {};
}

// A stage the mapping makes of the source with `values` and `extra` (the
// composition's facts): its layers, embedding, heads and stream, its
// folds; not its name, role, artifact or reads.
struct Made {
  Stage stage;
};
std::expected<Made, std::string> MakeStage(const Mapping& mapping, const Source& source,
                                           std::span<const SourceValue> values,
                                           std::span<const SourceValue> extra,
                                           std::string_view target_stage) {
  if (auto ok = ValidateMapping(mapping); !ok) {
    return std::unexpected(ok.error());
  }
  if (source.architecture != mapping.architecture) {
    return Refuse("source", std::format("architecture {}, not the mapping's {}",
                                        source.architecture, mapping.architecture));
  }
  if (source.tensors.size() > kMaxSourceTensors ||
      values.size() + extra.size() > kMaxSourceValues) {
    return Refuse("source", "too many tensors or values");
  }
  Values by_key;
  for (const auto list : {values, extra}) {
    for (const SourceValue& v : list) {
      if (!IsText(v.key) || !by_key.emplace(v.key, &v.value).second) {
        return Refuse(v.key, "an invalid key, or a key twice");
      }
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
  if (auto ok = CheckLayerFlags(mapping, by_key, *layers); !ok) {
    return std::unexpected(ok.error());
  }
  auto types = LayerTypes(mapping, by_key, *layers);
  if (!types) return std::unexpected(types.error());
  Synthesis syn(mapping, source, by_key, static_cast<std::uint32_t>(*layers), target_stage,
                std::move(*types));
  if (auto ok = syn.Gate(); !ok) {
    return std::unexpected(ok.error());
  }
  if (auto ok = syn.Bind(); !ok) {
    return std::unexpected(ok.error());
  }

  Made made;
  Stage& stage = made.stage;
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
  auto mixing = syn.Only(Site::kMixing);
  if (!mixing) {
    return std::unexpected(mixing.error());
  }
  stage.stream.mixing = std::move(*mixing);
  for (const SlotRule& s : mapping.slots) {
    if (s.site != Site::kSide || !syn.Present(s.slot)) {
      continue;
    }
    const auto& at = syn.Layers(s.slot);
    if (s.per_layer) {
      // A side input a layer, of that layer's tensors.
      for (const std::uint32_t il : at) {
        auto c = syn.Component(s, il);
        if (!c) {
          return std::unexpected(
              std::format("side input {} of layer {}: {}", s.slot, il, c.error()));
        }
        stage.side.push_back(
            {.name = std::format("{}_{}", s.slot, il), .source = std::move(*c), .layers = {il}});
      }
      continue;
    }
    auto c = syn.Component(s, 0);
    if (!c) {
      return std::unexpected(std::format("side input {}: {}", s.slot, c.error()));
    }
    stage.side.push_back(
        {.name = std::string(s.slot), .source = std::move(*c), .layers = {at.begin(), at.end()}});
  }
  stage.layers.resize(*layers);
  for (std::uint32_t il = 0; il < *layers; ++il) {
    for (const SlotRule& s : mapping.slots) {
      if (s.site != Site::kLayer || !syn.At(s, il)) {
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
  // The stage's folds: its container converter's and its importer's (the
  // latter validated against the catalog with the spec), sorted by
  // Canonicalize.
  for (const FoldRule& f : mapping.folds) {
    // A converter rule names a v0 artifact's converter, so it never adds a
    // fold to a v1 artifact's or a checkpoint's source (empty converter).
    const bool applied =
        f.converter.empty() ? f.container == source.container
                            : !source.converter.empty() && NamesPart(source.converter, f.converter);
    if (applied && !std::ranges::contains(stage.folds, f.fold)) {
      stage.folds.emplace_back(f.fold);
    }
  }
  for (const std::string& f : source.folds) {
    if (!std::ranges::contains(stage.folds, f)) {
      stage.folds.push_back(f);
    }
  }
  return made;
}

}  // namespace

bool IsSourceKey(std::string_view key) { return IsText(key); }

std::expected<std::string_view, std::string> DefaultPolicy(const Mapping& mapping,
                                                           std::span<const SourceValue> values,
                                                           const Envelope& envelope) {
  auto rule = DefaultRule(mapping, values, envelope);
  if (!rule) {
    return std::unexpected(rule.error());
  }
  return *rule == nullptr ? std::string_view{} : (*rule)->text;
}

std::expected<const PolicyRule*, std::string> DefaultRule(const Mapping& mapping,
                                                          std::span<const SourceValue> values,
                                                          const Envelope& envelope) {
  if (auto ok = ValidateMapping(mapping); !ok) {
    return std::unexpected(ok.error());
  }
  Values by_key;
  for (const SourceValue& v : values) {
    if (!IsText(v.key) || !by_key.emplace(v.key, &v.value).second) {
      return Refuse(v.key, "an invalid key, or a key twice");
    }
  }
  for (const PolicyRule& p : mapping.policies) {
    auto holds = Holds(p.gate, p.gate_absent, by_key);
    if (!holds) {
      return std::unexpected(holds.error());
    }
    const auto within = [](std::uint32_t bound, std::uint32_t value) {
      return bound == 0 || (value != 0 && value <= bound);
    };
    if (*holds && within(p.max_context, envelope.context) && within(p.max_slots, envelope.slots)) {
      return &p;
    }
  }
  return nullptr;
}

std::span<const Mapping* const> Mappings() {
  static const std::array<const Mapping*, 8> all = {
      &mappings::Gemma2Mapping(), &mappings::Gemma3Mapping(),
      &mappings::Gemma4Mapping(), &mappings::Gemma4AssistantMapping(),
      &mappings::Qwen38Mapping(), &mappings::Qwen38MtpMapping(),
      &mappings::Dsv4Mapping(),   &mappings::DsparkMapping()};
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
  if (!IsArchitecture(m.architecture) || m.version == 0 ||
      static_cast<std::uint8_t>(m.pipeline) > static_cast<std::uint8_t>(PipelineKind::kDecision)) {
    return Refuse(where, "an invalid architecture, version or pipeline");
  }
  if (!IsText(m.layer_count_key) || m.layer_types.empty() ||
      m.layer_types.size() > kMaxLayerTypes || !std::ranges::all_of(m.layer_types, Lower)) {
    return Refuse(where, "an invalid layer count key or layer types");
  }
  if (m.layer_type_key.empty() != m.layer_type_codes.empty() ||
      (!m.layer_type_key.empty() &&
       (!IsText(m.layer_type_key) || !m.layer_flags_key.empty() ||
        !std::ranges::all_of(m.layer_type_codes, [&](const LayerTypeCode& c) {
          return m.layer_types.contains(c.type) &&
                 std::ranges::count(m.layer_type_codes, c.value, &LayerTypeCode::value) == 1;
        })))) {
    return Refuse(where, "a layer type key without codes of the mapping's types (each value once)");
  }
  if (!m.layer_flags_key.empty() &&
      (!IsText(m.layer_flags_key) || !m.layer_types.contains(m.layer_flag_type))) {
    return Refuse(where, "an invalid layer flag key, or a flag type of no layer");
  }
  if (m.role != StageRole::kTarget && m.role != StageRole::kDrafter) {
    return Refuse(where, "a role other than a target's or a drafter's");
  }
  const bool drafter = m.role == StageRole::kDrafter;
  if (drafter ? !IsName(m.stage) || m.stage == "target" : !m.stage.empty() || !m.reads.empty()) {
    return Refuse(where, "a drafter's stage name and reads, or a target's");
  }
  for (const ReadRule& r : m.reads) {
    if (!IsName(r.name) || static_cast<std::uint8_t>(r.kind) > 1 ||
        static_cast<std::uint8_t>(r.point) > static_cast<std::uint8_t>(TapPoint::kStreamMean)) {
      return Refuse(std::format("{} read {}", where, r.name), "an invalid name, kind or point");
    }
    if (!r.layers_key.empty() && (!IsText(r.layers_key) || r.kind != ReadKind::kTap ||
                                  r.point == TapPoint::kPostFinalNorm)) {
      return Refuse(std::format("{} read {}", where, r.name),
                    "a layers key of a read other than a layer's tap");
    }
    if (r.layers_key.empty() && (r.index != 0 || r.entering)) {
      return Refuse(std::format("{} read {}", where, r.name), "an entry of no layers key");
    }
  }
  // A policy's own rules are the planner's; here, printable lines.
  for (const PolicyRule& p : m.policies) {
    if (p.text.empty() || p.text.size() > kMaxPolicyText || !p.text.ends_with('\n') ||
        !std::ranges::all_of(p.text,
                             [](char ch) { return ch == '\n' || (ch >= ' ' && ch <= '~'); })) {
      return Refuse(where, "a policy text of other than printable lines");
    }
    if ((!p.gate.empty() && !IsText(p.gate)) || (p.gate.empty() && p.gate_absent)) {
      return Refuse(where, "a policy of an invalid gate");
    }
    if (p.prefill_lookahead > 2) {
      return Refuse(where, "a policy planning more than two prompt chunks ahead");
    }
  }
  if (m.slots.size() + m.tensors.size() + m.params.size() + m.folds.size() + m.reads.size() +
          m.policies.size() >
      kMaxMappingRules) {
    return Refuse(where, std::format("more than {} rules", kMaxMappingRules));
  }
  std::set<std::string_view> slots;
  for (const SlotRule& s : m.slots) {
    const std::string at = std::format("{} slot {}", where, s.slot);
    if (!IsName(s.slot) || IsPseudo(s.slot) || !slots.insert(s.slot).second) {
      return Refuse(at, "not a name, a pseudo-slot's, or named twice");
    }
    if (!IsKind(s.kind) || s.version == 0 ||
        static_cast<std::uint8_t>(s.site) > static_cast<std::uint8_t>(Site::kMixing)) {
      return Refuse(at, "an invalid kind, version or site");
    }
    if ((s.site == Site::kSide && s.layers_key.empty() && s.count_key.empty()) ||
        (!s.layers_key.empty() && !s.count_key.empty()) ||
        (!s.layers_key.empty() &&
         (!IsText(s.layers_key) || (s.site != Site::kLayer && s.site != Site::kSide))) ||
        (!s.count_key.empty() &&
         (!IsText(s.count_key) || (s.site != Site::kLayer && s.site != Site::kSide))) ||
        (s.count_key.empty() && s.count_above) || (s.per_layer && s.site != Site::kSide)) {
      return Refuse(at,
                    "a side input without its layers' key, one key of its layers, or a layers "
                    "or count key off a layer");
    }
    if (s.site != Site::kLayer ? !s.layer_types.empty()
                               : !TypesWithin(s.layer_types, m.layer_types)) {
      return Refuse(at, "layer types off a layer slot, or not the mapping's");
    }
    if ((!s.gate.empty() && !IsText(s.gate)) || (s.gate.empty() && s.gate_absent)) {
      return Refuse(at, "an invalid gate");
    }
    // An embedding, final norm or output is a checkpoint's only one: a
    // second of the site is the alternative of the first's gate.
    if (s.site != Site::kLayer && s.site != Site::kSide) {
      for (const SlotRule& o : m.slots) {
        if (&o == &s) {
          break;
        }
        if (o.site == s.site && !Exclusive(o, s)) {
          return Refuse(at, "a second embedding, final norm or output");
        }
      }
    }
  }
  std::map<std::string_view, const SlotRule*> patterns;
  std::map<std::string_view, const SlotRule*> target_patterns;
  // The layer types each (slot, role, part) is bound at so far.
  std::map<std::tuple<std::string_view, std::string_view, format::PartRole>, std::string> roles;
  for (const TensorRule& t : m.tensors) {
    const std::string at = std::format("{} tensor {}", where, t.pattern);
    const SlotRule* slot = FindSlot(m, t.slot);
    if (slot == nullptr) {
      return Refuse(at, "a slot the mapping does not have");
    }
    const bool layered = slot->site == Site::kLayer;
    // A layer slot's pattern may name a shared tensor (no layer field); a
    // per-layer side input's names its layer's.
    const bool valid = layered           ? IsPattern(t.pattern, true) || IsPattern(t.pattern, false)
                       : slot->per_layer ? IsPattern(t.pattern, true)
                                         : IsPattern(t.pattern, false);
    // A pattern twice only in two slots of which a checkpoint has one; a
    // target's tensor names its target's checkpoint, not the source's.
    auto& names = t.target ? target_patterns : patterns;
    const auto [seen, fresh] = names.emplace(t.pattern, slot);
    if (!valid || (!fresh && !Exclusive(*seen->second, *slot))) {
      return Refuse(at, "not a pattern of its slot's site, or twice");
    }
    if (!t.fallback.empty() && !IsPattern(t.fallback, Layered(t.pattern))) {
      return Refuse(at, "a fallback that is not a pattern of its rule's form");
    }
    if (layered ? !TypesWithin(t.layer_types, m.layer_types) : !t.layer_types.empty()) {
      return Refuse(at, "layer types off a layer slot, or not the mapping's");
    }
    if (t.target && (!drafter || !t.fallback.empty() || t.optional)) {
      return Refuse(at, "a target's tensor in a target's mapping, or with a fallback");
    }
    if (t.unbound) {
      // A tensor accounted for, bound to no role.
      if (!t.role.empty() || t.target || !t.fallback.empty() || t.any_part) {
        return Refuse(at, "an unbound tensor with a role, a target or a fallback");
      }
      continue;
    }
    // A role bound once a layer type: rules for one role and part may
    // split the layer types between them (no types: all of them).
    if (!IsName(t.role) ||
        static_cast<std::uint8_t>(t.part) > static_cast<std::uint8_t>(format::PartRole::kZeros)) {
      return Refuse(at, "an invalid role or part, or one bound twice");
    }
    std::string& bound = roles[{t.slot, t.role, t.part}];
    const std::string_view types = t.layer_types.empty() ? m.layer_types : t.layer_types;
    if (std::ranges::any_of(types, [&](char c) { return bound.contains(c); })) {
      return Refuse(at, "an invalid role or part, or one bound twice");
    }
    bound += types;
  }
  for (const ParamRule& r : m.params) {
    const std::string at = std::format("{} parameter {}.{}", where, r.slot, r.param);
    if (static_cast<std::uint8_t>(r.type) > static_cast<std::uint8_t>(ValueType::kInts) ||
        (r.type == ValueType::kInts && (r.key.empty() || r.per_layer)) ||
        static_cast<std::uint8_t>(r.transform) > static_cast<std::uint8_t>(Transform::kRsqrt) ||
        (r.transform != Transform::kNone && r.type != ValueType::kReal)) {
      return Refuse(at, "an unknown type or transform, or a transform of a non-real");
    }
    if (!r.key.empty() && !IsText(r.key)) {
      return Refuse(at, "an invalid key");
    }
    // A layer slot's constant may name its layer (kLayerField).
    const SlotRule* named = r.slot.empty() ? nullptr : FindSlot(m, r.slot);
    const bool layer_name = named != nullptr && named->site == Site::kLayer &&
                            IsPattern(r.value.text, true) && IsName(Expand(r.value.text, 0));
    if ((r.key.empty() || r.slot.empty()) && r.type == ValueType::kString &&
        !IsName(r.value.text) && !(r.key.empty() && layer_name)) {
      return Refuse(at, "a constant string that is not a name");
    }
    if ((r.key.empty() || r.slot.empty()) && r.type == ValueType::kReal &&
        !std::isfinite(r.value.real)) {
      return Refuse(at, "a constant real that is not finite");
    }
    if (r.slot.empty()) {
      if (r.key.empty() || !r.param.empty() || !r.layer_types.empty() ||
          r.transform != Transform::kNone || r.per_layer) {
        return Refuse(at, "a requirement is a key and its value");
      }
      continue;
    }
    if (!IsName(r.param)) {
      return Refuse(at, "not a name");
    }
    if (IsPseudo(r.slot)) {
      if (PseudoParam(r.slot, r.param) != r.type || !r.layer_types.empty() || r.per_layer) {
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
    if (r.per_layer && (slot->site != Site::kLayer || r.key.empty() ||
                        (r.type != ValueType::kInt && r.type != ValueType::kReal))) {
      return Refuse(at, "a per-layer key of a layer slot's number");
    }
  }
  for (const FoldRule& f : m.folds) {
    if (FindFold(f.fold) == nullptr || f.container.empty() == f.converter.empty() ||
        (!f.container.empty() && !IsName(f.container)) ||
        (!f.converter.empty() && (!IsText(f.converter) || f.converter.contains('+')))) {
      return Refuse(std::format("{} fold {}", where, f.fold),
                    "not in the fold catalog, or not one container or converter part");
    }
  }
  for (const V0Assembly& a : m.v0_assembly) {
    const std::string at = std::format("{} v0 assembly {}", where, a.stem);
    if ((!a.stem.empty() && !IsPattern(a.stem, true) && !IsPattern(a.stem, false)) ||
        a.suffixes[0].empty() || a.suffixes[1].empty() || a.stacked == 0) {
      return Refuse(at, "an invalid stem, fewer than two parts or no stack");
    }
    for (std::size_t i = 0; i < a.suffixes.size(); ++i) {
      if (!a.suffixes.at(i).empty() && !IsText(a.suffixes.at(i))) {
        return Refuse(at, "an invalid suffix");
      }
    }
  }
  return {};
}

std::expected<Source, std::string> SourceOfV0Tensors(std::string architecture,
                                                     std::string container, std::string converter,
                                                     std::span<const V0Tensor> tensors,
                                                     const Mapping* mapping) {
  Source s{.architecture = std::move(architecture),
           .container = std::move(container),
           .converter = std::move(converter),
           .tensors = {}};
  // The resources and arrays the mapping pairs: by name, the format of the
  // tensor they hold a part of, and the part.
  std::map<std::string, std::pair<format::TensorFormat, format::PartRole>, std::less<>> paired;
  if (mapping != nullptr) {
    std::map<std::string_view, const V0Tensor*> by_name;
    for (const V0Tensor& t : tensors) by_name.emplace(t.name, &t);
    for (const V0Assembly& a : mapping->v0_assembly) {
      for (const V0Tensor& t : tensors) {
        if ((t.count != 0) != a.experts || !t.name.ends_with(a.suffixes[0])) continue;
        const std::string_view stem =
            std::string_view(t.name).substr(0, t.name.size() - a.suffixes[0].size());
        if (!a.stem.empty() && !Match(a.stem, stem, std::numeric_limits<std::uint32_t>::max())) {
          continue;
        }
        std::vector<artifact::V0Part> parts;
        std::vector<std::string> part_names;
        std::size_t wanted = 0;
        for (std::size_t i = 0; i < a.suffixes.size(); ++i) {
          if (a.suffixes.at(i).empty()) continue;
          ++wanted;
          std::string name = std::format("{}{}", stem, a.suffixes.at(i));
          const auto found = by_name.find(name);
          if (found == by_name.end() || (found->second->count != 0) != a.experts ||
              found->second->repr == nullptr) {
            break;
          }
          parts.push_back({.role = a.roles.at(i), .rep = found->second->repr});
          part_names.push_back(std::move(name));
        }
        if (parts.size() != wanted) continue;  // not every part: left as it is
        auto f = artifact::Assemble(a.encoding, a.layout, parts, a.stacked);
        if (!f) {
          return Refuse(t.name, std::format("its parts do not assemble ({})",
                                            format::FormatErrorName(f.error())));
        }
        if (a.experts) {
          f->shape.ne[2] = t.count;
          f->shape.rank = 3;
        }
        for (std::size_t i = 0; i < parts.size(); ++i) {
          if (!paired.emplace(part_names[i], std::pair{*f, parts[i].role}).second) {
            return Refuse(part_names[i], "a part of two tensors");
          }
        }
      }
    }
  }
  for (const V0Tensor& t : tensors) {
    if (t.repr == nullptr) return Refuse(t.name, "no representation");
    auto f = artifact::FormatOf(*t.repr);
    if (!f || (t.count != 0 && f->shape.rank > 2)) {
      return Refuse(t.name,
                    f ? std::string("no two-dimensional slice format")
                      : std::format("no tensor format ({})", format::FormatErrorName(f.error())));
    }
    format::PartRole part = artifact::V0PartRole(*t.repr);
    if (const auto at = paired.find(t.name); at != paired.end()) {
      f = at->second.first;
      part = at->second.second;
    } else if (t.count != 0) {
      f->shape.ne[2] = t.count;
      f->shape.rank = 3;
    }
    if (t.count != 0) {
      s.tensors.push_back({.name = t.name, .format = *f, .part = part, .experts = true});
      continue;
    }
    for (const std::string& role : t.roles) {
      s.tensors.push_back({.name = role, .format = *f, .part = part, .experts = false});
    }
  }
  if (s.tensors.size() > kMaxSourceTensors) {
    return Refuse("source", std::format("more than {} tensors", kMaxSourceTensors));
  }
  return s;
}

std::expected<Source, std::string> SourceOfV0(const artifact::Artifact& artifact,
                                              const Mapping* mapping) {
  if (artifact.format_version() != 0) {
    // Its descriptors are empty; spec/stored.h SourceOf reads its formats.
    return Refuse(
        std::format("artifact {}", artifact.id()),
        std::format("format version {}, not 0 (read it with SourceOf)", artifact.format_version()));
  }
  const auto sources = artifact.sources();
  const auto all = [&](auto pred) {
    return !sources.empty() && std::ranges::all_of(sources, pred);
  };
  std::string container = "other";
  if (all([](const artifact::Source& f) { return f.name.ends_with(".gguf"); })) {
    container = "gguf";
  } else if (all([](const artifact::Source& f) {
               return f.name.ends_with(".safetensors") || f.name.ends_with(".json");
             })) {
    container = "safetensors";
  }
  std::vector<V0Tensor> tensors;
  for (const artifact::Resource& r : artifact.resources()) {
    tensors.push_back({.name = r.name, .roles = r.roles, .repr = &r.repr, .count = 0});
  }
  for (const artifact::ExpertArray& a : artifact.expert_arrays()) {
    tensors.push_back({.name = a.name, .roles = {a.name}, .repr = &a.repr, .count = a.count});
  }
  return SourceOfV0Tensors(
      artifact.model().architecture, std::move(container),
      std::format("{} {}", artifact.converter().name, artifact.converter().version), tensors,
      mapping);
}

std::expected<ModelSpec, std::string> Synthesize(const Mapping& mapping, const Source& source,
                                                 std::span<const SourceValue> values) {
  if (mapping.role != StageRole::kTarget) {
    return Refuse(std::format("mapping {}", mapping.architecture),
                  "a drafter's mapping, which composes with its target");
  }
  auto made = MakeStage(mapping, source, values, {}, {});
  if (!made) {
    return std::unexpected(made.error());
  }
  made->stage.name = "target";
  made->stage.role = StageRole::kTarget;
  ModelSpec spec{.pipeline = mapping.pipeline};
  spec.stages.push_back(std::move(made->stage));
  Canonicalize(spec);
  if (auto ok = Validate(spec); !ok) {
    return std::unexpected(ok.error());
  }
  return spec;
}

std::expected<ModelSpec, std::string> SynthesizeV0(const artifact::Artifact& artifact,
                                                   const Mapping& mapping,
                                                   std::span<const SourceValue> values) {
  auto source = SourceOfV0(artifact, &mapping);
  if (!source) {
    return std::unexpected(source.error());
  }
  return Synthesize(mapping, *source, values);
}

std::expected<ModelSpec, std::string> ComposeDrafter(const ModelSpec& target,
                                                     const Mapping& target_mapping,
                                                     const Mapping& drafter, const Source& source,
                                                     std::span<const SourceValue> values) {
  const std::string where = std::format("mapping {}", drafter.architecture);
  if (drafter.role != StageRole::kDrafter || target_mapping.role != StageRole::kTarget) {
    return Refuse(where, "not a drafter's mapping composed with a target's");
  }
  if (auto ok = ValidateMapping(target_mapping); !ok) {
    return std::unexpected(ok.error());
  }
  if (auto ok = Validate(target); !ok) {
    return std::unexpected(ok.error());
  }
  if (target.stages.empty() || target.stages.front().role != StageRole::kTarget ||
      target.pipeline != drafter.pipeline) {
    return Refuse(where, "a target spec whose first stage is not its target, or of another kind");
  }
  if (std::ranges::contains(target.stages, drafter.stage, &Stage::name)) {
    return Refuse(where, std::format("the target spec already has a stage {}", drafter.stage));
  }
  const Stage& head = target.stages.front();
  // The target's facts: its last layer of each of its mapping's types (a
  // repeating pattern's: types a key gives are the target's values', which
  // the composition has not).
  std::vector<SourceValue> facts;
  const auto layers = static_cast<std::uint32_t>(head.layers.size());
  for (const char t :
       target_mapping.layer_type_key.empty() ? target_mapping.layer_types : std::string_view{}) {
    const std::string key = std::format("{}{}", kTargetLastLayer, t);
    if (std::ranges::contains(facts, key, &SourceValue::key)) {
      continue;
    }
    for (std::uint32_t il = layers; il-- > 0;) {
      if (target_mapping.layer_types.at(il % target_mapping.layer_types.size()) == t) {
        facts.push_back({.key = key, .value = ParamValue(std::int64_t{il})});
        break;
      }
    }
  }
  auto made = MakeStage(drafter, source, values, facts, head.name);
  if (!made) {
    return std::unexpected(made.error());
  }
  ModelSpec spec = target;
  std::uint32_t artifact = 0;
  for (const Stage& s : spec.stages) {
    artifact = std::max(artifact, s.artifact + 1);
  }
  Stage& stage = made->stage;
  stage.name = std::string(drafter.stage);
  stage.role = StageRole::kDrafter;
  stage.artifact = artifact;
  Stage& target_stage = spec.stages.front();
  for (const ReadRule& r : drafter.reads) {
    stage.reads.push_back(
        {.stage = target_stage.name, .kind = r.kind, .name = std::string(r.name)});
    if (r.kind != ReadKind::kTap) {
      continue;
    }
    // A layer's tap is the target's last layer's, or the one the drafter's
    // source names (the streams entering it: after the layer before).
    std::uint32_t tap_layer = r.point == TapPoint::kPostFinalNorm || target_stage.layers.empty()
                                  ? 0
                                  : static_cast<std::uint32_t>(target_stage.layers.size() - 1);
    if (!r.layers_key.empty()) {
      const auto found = std::ranges::find(values, r.layers_key, &SourceValue::key);
      const auto* list =
          found == values.end() ? nullptr : std::get_if<std::vector<std::int64_t>>(&found->value);
      const auto count = static_cast<std::int64_t>(target_stage.layers.size());
      const std::int64_t entry =
          list != nullptr && r.index < list->size() ? (*list)[r.index] : std::int64_t{-1};
      const std::int64_t layer = r.entering ? entry - 1 : entry;
      if (layer < 0 || layer >= count) {
        return Refuse(where,
                      std::format("read {}: {} names no layer of the target's {} ({})", r.name,
                                  r.layers_key, count, r.entering ? "entering" : "after"));
      }
      tap_layer = static_cast<std::uint32_t>(layer);
    }
    const auto tap = std::ranges::find(target_stage.taps, r.name, &FeatureTap::name);
    if (tap == target_stage.taps.end()) {
      target_stage.taps.push_back(
          {.name = std::string(r.name), .layer = tap_layer, .point = r.point});
    } else if (tap->point != r.point || tap->layer != tap_layer) {
      return Refuse(where,
                    std::format("the target's tap {} is not the one the drafter reads", r.name));
    }
  }
  // The drafter's folds are its own artifact's, never merged with the
  // target's: the two may come from different containers.
  spec.stages.push_back(std::move(stage));
  Canonicalize(spec);
  if (auto ok = Validate(spec); !ok) {
    return std::unexpected(ok.error());
  }
  return spec;
}

std::expected<ModelSpec, std::string> ComposeDrafterV0(const ModelSpec& target,
                                                       const Mapping& target_mapping,
                                                       const artifact::Artifact& artifact,
                                                       const Mapping& drafter,
                                                       std::span<const SourceValue> values) {
  auto source = SourceOfV0(artifact, &drafter);
  if (!source) {
    return std::unexpected(source.error());
  }
  return ComposeDrafter(target, target_mapping, drafter, *source, values);
}

}  // namespace llmp::spec
