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
#include <ranges>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "format/encoding.h"
#include "format/tensor_format.h"
#include "spec/mappings/dsv4.h"
#include "spec/mappings/gemma2.h"
#include "spec/mappings/gemma3.h"
#include "spec/mappings/gemma4.h"
#include "spec/mappings/qwen2.h"
#include "spec/mappings/qwen38.h"
#include "spec/mappings/rules.h"
#include "spec/names.h"
#include "spec/spec.h"

namespace llmp::spec {
namespace {

using Status = std::expected<void, std::string>;

std::unexpected<std::string> Refuse(std::string_view where, std::string_view what) {
  return std::unexpected(std::format("{}: {}", where, what));
}

using names::Digit;
using names::IsKind;
using names::IsName;
using names::Lower;

// An architecture as GGUF names it: a name that may also hold '-'
// ("gemma4-assistant").
bool IsArchitecture(std::string_view s) {
  return !s.empty() && s.size() <= kMaxText && Lower(s.front()) &&
         std::ranges::all_of(s,
                             [](char c) { return Lower(c) || Digit(c) || c == '_' || c == '-'; });
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

const ParamValue* FindValue(const Mapping& m, const Values& values, std::string_view key) {
  if (key.empty()) return nullptr;
  if (const auto found = values.find(key); found != values.end()) return found->second;
  const std::string canonical = CanonicalKey(m, key);
  if (const auto found = values.find(canonical); found != values.end()) return found->second;
  return nullptr;
}

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
    case Transform::kFloat32:
      y = static_cast<double>(static_cast<float>(x));
      break;
    case Transform::kProduct:
      break;
  }
  if (!std::isfinite(y)) {
    return Refuse(where, "a value outside its transform's domain");
  }
  return y == 0.0 ? 0.0 : y;
}

// Where a per-layer rule reads: the layer and the stage's layer count.
struct LayerAt {
  std::uint32_t layer = 0;
  std::uint32_t layers = 0;
};

// The rule's value: its constant, or its key's converted (a per-layer
// key's entry for the layer); nothing when the key is absent.
std::expected<std::optional<ParamValue>, std::string> Evaluate(
    const Mapping& m, const ParamRule& r, const Values& values, std::string_view where,
    LayerAt layer = {}, const std::map<std::string_view, const SourceTensor*>* tensors = nullptr) {
  if (!r.tensor.empty()) {
    if (tensors != nullptr) {
      if (const auto it = tensors->find(Expand(r.tensor, layer.layer)); it != tensors->end()) {
        if (r.type == ValueType::kInt) {
          if (it->second->format.shape.rank == 0) {
            return Refuse(where, std::format("tensor '{}' has rank 0", r.tensor));
          }
          return ParamValue(static_cast<std::int64_t>(it->second->format.shape.ne[0]));
        }
        return ParamValue(true);
      }
    }
    return std::optional<ParamValue>();
  }
  const auto eval_ratio = [&]() -> std::expected<std::optional<ParamValue>, std::string> {
    const ParamValue* num_found = FindValue(m, values, r.numerator_key);
    const ParamValue* den_found = FindValue(m, values, r.divisor_key);
    if (num_found == nullptr || den_found == nullptr) {
      return std::optional<ParamValue>();
    }
    const bool is_product = r.transform == Transform::kProduct;
    const std::string at = std::format("{} (from {} {} {})", where, r.numerator_key,
                                       is_product ? "*" : "/", r.divisor_key);
    const auto* num_i = std::get_if<std::int64_t>(num_found);
    const auto* den_i = std::get_if<std::int64_t>(den_found);
    const auto* num_d = std::get_if<double>(num_found);
    const auto* den_d = std::get_if<double>(den_found);
    if (num_i == nullptr && num_d == nullptr) {
      return Refuse(at, "numerator is not a number");
    }
    if (den_i == nullptr && den_d == nullptr) {
      return Refuse(at, "divisor is not a number");
    }
    if (r.type == ValueType::kInt) {
      if (num_i != nullptr && den_i != nullptr) {
        if (is_product) {
          std::int64_t prod = 0;
          if (__builtin_mul_overflow(*num_i, *den_i, &prod)) {
            return Refuse(at, "product overflows");
          }
          return ParamValue(prod);
        }
        if (*den_i == 0) return Refuse(at, "division by zero");
        if (*num_i == std::numeric_limits<std::int64_t>::min() && *den_i == -1) {
          return Refuse(at, "ratio overflows");
        }
        if (*num_i % *den_i != 0) return Refuse(at, "ratio is not an integer");
        return ParamValue(*num_i / *den_i);
      }
      const double num_val = num_i != nullptr ? static_cast<double>(*num_i) : *num_d;
      const double den_val = den_i != nullptr ? static_cast<double>(*den_i) : *den_d;
      if (!is_product && den_val == 0.0) return Refuse(at, "division by zero");
      const double q = is_product ? (num_val * den_val) : (num_val / den_val);
      if (!std::isfinite(q) || std::floor(q) != q ||
          q < static_cast<double>(std::numeric_limits<std::int64_t>::min()) ||
          q >= 9223372036854775808.0) {
        return Refuse(at, is_product ? "product is not an integer" : "ratio is not an integer");
      }
      return ParamValue(static_cast<std::int64_t>(q));
    }
    if (r.type == ValueType::kReal) {
      const double num_val = num_i != nullptr ? static_cast<double>(*num_i) : *num_d;
      const double den_val = den_i != nullptr ? static_cast<double>(*den_i) : *den_d;
      if (!is_product && den_val == 0.0) return Refuse(at, "division by zero");
      const double raw = is_product ? (num_val * den_val) : (num_val / den_val);
      auto y = Transformed(raw, is_product ? Transform::kNone : r.transform, at);
      if (!y) return std::unexpected(y.error());
      return ParamValue(*y);
    }
    return Refuse(at, "ratio of an unsupported type");
  };

  if (r.key.empty()) {
    if (!r.numerator_key.empty() && !r.divisor_key.empty()) {
      return eval_ratio();
    }
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
  const ParamValue* found = FindValue(m, values, r.key);
  if (found == nullptr) {
    if (!r.numerator_key.empty() && !r.divisor_key.empty()) {
      return eval_ratio();
    }
    return std::optional<ParamValue>();
  }
  const std::string at = std::format("{} (from {})", where, r.key);
  // A per-layer key's entry for the layer (a list of one value a layer),
  // or its one value for every layer.
  ParamValue entry = *found;
  if (r.per_layer) {
    const auto pick = [&](const auto& list) -> std::expected<void, std::string> {
      if (list.size() != layer.layers || layer.layer >= list.size()) {
        return Refuse(at,
                      std::format("a list of {} values for {} layers", list.size(), layer.layers));
      }
      entry = list[layer.layer];
      return {};
    };
    if (const auto* is = std::get_if<std::vector<std::int64_t>>(found)) {
      if (auto ok = pick(*is); !ok) {
        return std::unexpected(ok.error());
      }
    } else if (const auto* ds = std::get_if<std::vector<double>>(found)) {
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
      if (!r.flag_text.empty()) {
        if (const auto* text = std::get_if<std::string>(&v)) {
          if (*text == r.flag_text) return ParamValue(r.value.flag);
          return Refuse(at, "not the value the mapping's components implement");
        }
      }
      if (!r.value.text.empty()) {
        const auto* s = std::get_if<std::string>(&v);
        if (s == nullptr) {
          return Refuse(at, "not a string");
        }
        if (*s == r.value.text) {
          return ParamValue(r.value.flag);
        }
        if (*s == "none" || *s == "default" ||
            std::ranges::any_of(m.params, [&](const ParamRule& other) {
              return other.type == ValueType::kBool && !other.value.text.empty() &&
                     MatchesCanonicalKey(m, other.key, r.key) && *s == other.value.text;
            })) {
          return std::optional<ParamValue>();
        }
        return Refuse(at, "not the value the mapping's components implement");
      }
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
        if (r.param == "rope_sections" && is->size() == 3) {
          std::vector<std::int64_t> padded = *is;
          padded.push_back(0);
          return ParamValue(std::move(padded));
        }
        return ParamValue(*is);
      }
      return Refuse(at, "not a list of integers");
  }
  return Refuse(at, "an unknown value type");
}

// The parameter's value at a layer type (0: not a layer slot), by its
// rules in order; nothing when no rule applies, or none resolves and one
// is optional. A rule unless a fold the source carries does not apply.
std::expected<std::optional<ParamValue>, std::string> Resolve(
    const Mapping& m, std::string_view slot, std::string_view param, char type,
    const Values& values, LayerAt layer = {}, std::span<const std::string> folds = {},
    const std::map<std::string_view, const SourceTensor*>* tensors = nullptr) {
  const std::string where = std::format("{}.{}", slot, param);
  bool applied = false;
  bool optional = false;
  for (const ParamRule& r : m.params) {
    if (r.slot != slot || r.param != param || (type != 0 && !Applies(r.layer_types, type)) ||
        (!r.unless_fold.empty() && std::ranges::contains(folds, r.unless_fold))) {
      continue;
    }
    applied = true;
    optional = optional || r.optional;
    auto v = Evaluate(m, r, values, where, layer, tensors);
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
    auto v = Evaluate(m, r, values, std::format("requirement {}", r.key));
    if (!v) {
      return std::unexpected(v.error());
    }
    if (!*v) {
      if (r.optional) {
        continue;
      }
      return Refuse(r.key, "absent from the source");
    }
    ParamRule constant = r;
    constant.key = {};
    constant.numerator_key = {};
    constant.divisor_key = {};
    constant.transform = Transform::kNone;
    auto want = Evaluate(m, constant, values, r.key);
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

bool MatchesArchitecture(const Mapping& m, std::string_view arch) {
  return m.architecture == arch || std::ranges::contains(m.source_architectures, arch);
}

std::optional<std::string> CompanionStem(std::string_view name, format::PartRole part) {
  if (part != format::PartRole::kBlockScales && part != format::PartRole::kTensorScale &&
      part != format::PartRole::kBiases && part != format::PartRole::kZeros) {
    return std::nullopt;
  }
  for (std::string_view suffix : {".weight_scale_2", ".weight_scale", ".weight_bias",
                                  ".weight_zero", ".scales", ".biases", ".zeros"}) {
    if (name.size() > suffix.size() && name.ends_with(suffix)) {
      return std::format("{}.weight", name.substr(0, name.size() - suffix.size()));
    }
  }
  if (name.ends_with(".weight")) {
    return std::string(name);
  }
  return std::nullopt;
}

// Whether a gate holds for the values: the key an integer above zero, or
// (absent) the key absent or zero; an empty gate always holds.
std::expected<bool, std::string> Holds(const Mapping& m, std::string_view gate, bool absent,
                                       const Values& values) {
  if (gate.empty()) {
    return true;
  }
  std::int64_t n = 0;
  if (const ParamValue* found = FindValue(m, values, gate); found != nullptr) {
    // A flag (an approval's gate) holds when set.
    if (const auto* flag = std::get_if<bool>(found); flag != nullptr) {
      n = *flag ? 1 : 0;
    } else {
      const auto* i = std::get_if<std::int64_t>(found);
      if (i == nullptr || *i < 0) {
        return Refuse(gate, "a gate that is not a flag or an integer of at least zero");
      }
      n = *i;
    }
  }
  return absent ? n == 0 : n > 0;
}

// Two slots of which a checkpoint (or a layer) has at most one: gated by
// one key oppositely, at the layers on either side of one count, or disjoint
// in layer types.
bool Exclusive(const SlotRule& a, const SlotRule& b) {
  if (a.site == Site::kLayer && b.site == Site::kLayer && !a.layer_types.empty() &&
      !b.layer_types.empty() &&
      std::ranges::none_of(a.layer_types, [&](char c) { return b.layer_types.contains(c); })) {
    return true;
  }
  return (!a.gate.empty() && a.gate == b.gate && a.gate_absent != b.gate_absent) ||
         (!a.count_key.empty() && a.count_key == b.count_key && a.count_above != b.count_above);
}

// The layers a source key names: an integer or a list of integers, each
// below `layers`.
std::expected<std::set<std::uint32_t>, std::string> LayersOf(const Mapping& m, std::string_view key,
                                                             const Values& values,
                                                             std::uint32_t layers,
                                                             std::string_view alias = {},
                                                             std::uint8_t alias_base = 0) {
  std::set<std::uint32_t> out;
  bool one_based = false;
  const ParamValue* found = FindValue(m, values, key);
  if (found == nullptr && !alias.empty()) {
    found = FindValue(m, values, alias);
    one_based = alias_base == 1;
  }
  if (found == nullptr) return out;
  std::vector<std::int64_t> list;
  if (const auto* i = std::get_if<std::int64_t>(found)) {
    list.push_back(*i);
  } else if (const auto* is = std::get_if<std::vector<std::int64_t>>(found)) {
    list = *is;
  } else {
    return Refuse(key, "not a layer or a list of layers");
  }
  if (!one_based && !alias.empty()) {
    if (const auto* other = FindValue(m, values, alias)) {
      std::vector<std::int64_t> alternate;
      if (const auto* i = std::get_if<std::int64_t>(other))
        alternate.push_back(*i);
      else if (const auto* alternate_values = std::get_if<std::vector<std::int64_t>>(other))
        alternate = *alternate_values;
      else
        return Refuse(alias, "not a layer selector");
      if (alternate.size() != list.size())
        return Refuse(alias, "alternate layer selector disagrees");
      for (std::size_t i = 0; i < list.size(); ++i)
        if (alternate[i] < alias_base || alternate[i] - alias_base != list[i])
          return Refuse(alias, "alternate layer selector disagrees");
    }
  }
  for (std::int64_t l : list) {
    if (one_based) {
      if (l < 1) return Refuse(key, "layer index is below its declared origin");
      --l;
    }
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
            std::string_view target_stage, std::string types, std::span<const std::string> folds)
      : m_(m),
        source_(s),
        values_(v),
        layers_(layers),
        target_stage_(target_stage),
        folds_(folds),
        types_(std::move(types)) {}

  // The slots the source has, by their gates, and the layers a slot with a
  // layers or count key is at.
  std::expected<void, std::string> Gate() {
    for (const SlotRule& slot : m_.slots) {
      auto present = Holds(m_, slot.gate, slot.gate_absent, values_);
      if (!present) {
        return std::unexpected(present.error());
      }
      if (*present) {
        present_.insert(slot.slot);
      }
      if (!slot.layers_key.empty()) {
        auto at = LayersOf(m_, slot.layers_key, values_, layers_, slot.layers_alias,
                           slot.alias_index_base);
        if (!at) {
          return std::unexpected(at.error());
        }
        at_.emplace(slot.slot, std::move(*at));
      }
      if (!slot.count_key.empty()) {
        std::uint64_t n = 0;
        if (const ParamValue* found = FindValue(m_, values_, slot.count_key); found != nullptr) {
          auto count = Count(found, slot.count_key, 0, layers_);
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
    const auto match_rule =
        [&](std::string_view tensor_name) -> std::optional<std::pair<std::size_t, std::uint32_t>> {
      std::optional<std::pair<std::size_t, std::uint32_t>> match;
      for (std::size_t i = 0; i < m_.tensors.size(); ++i) {
        const TensorRule& rule = m_.tensors[i];
        if (rule.target || !Present(rule.slot)) {
          continue;  // the target's tensor, or a slot the source has not
        }
        const auto layer = Match(rule.pattern, tensor_name, layers_);
        if (!layer) {
          continue;
        }
        const SlotRule& at = *FindSlot(m_, rule.slot);
        if (at.site == Site::kLayer && Layered(rule.pattern) &&
            (!At(at, *layer) || !Applies(rule.layer_types, Type(*layer)))) {
          continue;
        }
        if (match) {
          return std::nullopt;
        }
        match = std::pair(i, Layered(rule.pattern) ? *layer : kShared);
      }
      return match;
    };
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
        const SlotRule& at = *FindSlot(m_, rule.slot);
        if (at.site == Site::kLayer && Layered(rule.pattern) &&
            (!At(at, *layer) || !Applies(rule.layer_types, Type(*layer)))) {
          continue;
        }
        if (match) {
          return Refuse(t.name, "matches two of the mapping's tensor rules");
        }
        match = std::pair(i, Layered(rule.pattern) ? *layer : kShared);
      }
      const bool mismatched_rule =
          match &&
          ((!m_.tensors[match->first].any_part && m_.tensors[match->first].part != t.part) ||
           m_.tensors[match->first].experts != t.experts);
      if (!match || bound_.contains(*match) || mismatched_rule) {
        if (const auto stem = CompanionStem(t.name, t.part); stem && names_.contains(*stem)) {
          if (const auto stem_match = match_rule(*stem)) {
            const TensorRule& stem_rule = m_.tensors[stem_match->first];
            if (stem_rule.any_part && stem_rule.experts == t.experts) {
              auto& list = companions_[*stem];
              if (std::ranges::any_of(list,
                                      [&](const SourceTensor* c) { return c->part == t.part; })) {
                return Refuse(t.name, "binds a companion part another tensor binds");
              }
              list.push_back(&t);
              continue;
            }
          }
        }
        if (!match) {
          return Refuse(t.name, "matches none of the mapping's tensor rules of the slots it has");
        }
      }
      const TensorRule& rule = m_.tensors[match->first];
      const SlotRule& slot = *FindSlot(m_, rule.slot);
      if (match->second != kShared && slot.site == Site::kLayer &&
          (!At(slot, match->second) || !Applies(rule.layer_types, Type(match->second)))) {
        return Refuse(t.name, std::format("layer {} of type '{}' has no {} {}", match->second,
                                          Type(match->second), rule.slot, rule.role));
      }
      if (match->second == kShared && slot.site == Site::kLayer &&
          !std::ranges::any_of(std::views::iota(std::uint32_t{0}, layers_), [&](auto layer) {
            return At(slot, layer) && Applies(rule.layer_types, Type(layer));
          }))
        return Refuse(t.name, "a shared tensor has no component instance that binds it");
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
      auto v = Resolve(m_, slot.slot, r.param, type, values_, at, folds_, &names_);
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
      if (held != names_.end() && ((!rule.any_part && held->second->part != rule.part) ||
                                   held->second->experts != rule.experts))
        return Refuse(tensor, "fallback tensor has a different part or expert geometry");
      const format::PartRole part =
          rule.any_part && held != names_.end() ? held->second->part : rule.part;
      c.weights.push_back(WeightBinding{
          .role = std::string(rule.role), .tensor = tensor, .part = part, .experts = rule.experts});
      if (const auto comp_it = companions_.find(tensor); comp_it != companions_.end()) {
        for (const SourceTensor* comp : comp_it->second) {
          if (std::ranges::none_of(c.weights, [&](const WeightBinding& w) {
                return w.role == rule.role && w.part == comp->part && w.stage.empty();
              })) {
            c.weights.push_back(WeightBinding{.role = std::string(rule.role),
                                              .tensor = comp->name,
                                              .part = comp->part,
                                              .experts = comp->experts});
          }
        }
      }
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
  std::map<std::string, std::vector<const SourceTensor*>, std::less<>> companions_;
  std::span<const std::string> folds_;
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
  std::optional<std::uint64_t> interval;
  if (const auto* value = FindValue(m, values, m.layer_type_interval_key)) {
    const auto* integer = std::get_if<std::int64_t>(value);
    if (!integer || *integer <= 0)
      return Refuse(m.layer_type_interval_key, "not a positive interval");
    interval = static_cast<std::uint64_t>(*integer);
  }
  const auto finish = [&](std::string types,
                          bool explicit_types) -> std::expected<std::string, std::string> {
    if (!interval) return types;
    if (!explicit_types) types.resize(static_cast<std::size_t>(layers));
    for (std::uint64_t layer = 0; layer < layers; ++layer) {
      const char type = (layer + 1) % *interval == 0 ? m.layer_types.back() : m.layer_types.front();
      if (explicit_types && types[layer] != type)
        return Refuse(m.layer_type_interval_key, "disagrees with the explicit layer schedule");
      types[layer] = type;
    }
    return types;
  };
  const ParamValue* found = FindValue(m, values, m.layer_type_key);
  if (found == nullptr && !m.layer_flags_key.empty()) {
    if (const auto* value = FindValue(m, values, m.layer_flags_key)) {
      const auto* flags = std::get_if<std::vector<std::int64_t>>(value);
      if (flags == nullptr || flags->size() != layers)
        return Refuse(m.layer_flags_key, "not a flag for each layer");
      char other = 0;
      for (const char type : m.layer_types) {
        if (type == m.layer_flag_type) continue;
        if (other != 0 && other != type)
          return Refuse(m.layer_flags_key, "flags do not describe a two-type schedule");
        other = type;
      }
      if (other == 0) return Refuse(m.layer_flags_key, "flags do not describe a two-type schedule");
      std::string types;
      types.reserve(static_cast<std::size_t>(layers));
      for (const auto flag : *flags) types.push_back(flag != 0 ? m.layer_flag_type : other);
      return finish(std::move(types), true);
    }
  }
  if (m.layer_type_key.empty()) return finish(std::string(m.layer_types), false);
  if (found != nullptr) {
    if (const auto* str = std::get_if<std::string>(found)) {
      if (str->size() < layers) {
        return Refuse(m.layer_type_key, std::format("not a value for each of {} layers", layers));
      }
      for (std::uint64_t il = 0; il < layers; ++il) {
        if (!m.layer_types.contains((*str)[il])) {
          return Refuse(
              m.layer_type_key,
              std::format("layer {}'s '{}' is none of the mapping's layer types", il, (*str)[il]));
        }
      }
      return finish(str->substr(0, layers), true);
    }
  }
  if (found == nullptr && (interval || (!m.layer_type_codes.empty() && m.layer_types.size() > 1))) {
    return finish(std::string(m.layer_types), false);
  }
  const auto* list = found == nullptr ? nullptr : std::get_if<std::vector<std::int64_t>>(found);
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
  return finish(std::move(types), true);
}

// A layer-flag list (layer_flags_key): one flag a layer, set exactly at
// the layers of layer_flag_type.
std::expected<void, std::string> CheckLayerFlags(const Mapping& m, const Values& values,
                                                 std::string_view types, std::uint64_t layers) {
  if (m.layer_flags_key.empty()) {
    return {};
  }
  const ParamValue* found = FindValue(m, values, m.layer_flags_key);
  if (found == nullptr) {
    return {};
  }
  const auto* flags = std::get_if<std::vector<std::int64_t>>(found);
  if (flags == nullptr || flags->size() != layers) {
    return Refuse(m.layer_flags_key, std::format("not a flag for each of {} layers", layers));
  }
  for (std::uint64_t il = 0; il < layers; ++il) {
    const bool typed = types.at(il % types.size()) == m.layer_flag_type;
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
  if (!mapping.containers.empty() && !std::ranges::contains(mapping.containers, source.container))
    return Refuse("source", "container has no variant of this mapping");
  if (!MatchesArchitecture(mapping, source.architecture)) {
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
  const ParamValue* found = FindValue(mapping, by_key, mapping.layer_count_key);
  const auto layers = Count(found, mapping.layer_count_key, 1, kMaxLayers);
  if (!layers) {
    return std::unexpected(layers.error());
  }
  if (auto ok = CheckRequirements(mapping, by_key); !ok) {
    return std::unexpected(ok.error());
  }
  auto types = LayerTypes(mapping, by_key, *layers);
  if (!types) return std::unexpected(types.error());
  if (auto ok = CheckLayerFlags(mapping, by_key, *types, *layers); !ok) {
    return std::unexpected(ok.error());
  }
  std::vector<std::string> effective_folds = source.folds;
  for (const FoldRule& f : mapping.folds)
    if (!f.imported && f.container == source.container &&
        !std::ranges::contains(effective_folds, f.fold))
      effective_folds.emplace_back(f.fold);
  Synthesis syn(mapping, source, by_key, static_cast<std::uint32_t>(*layers), target_stage,
                std::move(*types), effective_folds);
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
  if (context && *context == 0 && !mapping.context_key.empty()) {
    if (const ParamValue* cv = FindValue(mapping, by_key, mapping.context_key); cv != nullptr) {
      context = Count(cv, mapping.context_key, 1, kMaxContext);
    }
  }
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
  } else if (!mapping.softcap_key.empty()) {
    if (const ParamValue* sv = FindValue(mapping, by_key, mapping.softcap_key); sv != nullptr) {
      if (const auto* d = std::get_if<double>(sv)) {
        stage.heads.softcap = *d;
      } else if (const auto* i = std::get_if<std::int64_t>(sv)) {
        stage.heads.softcap = static_cast<double>(*i);
      } else {
        return Refuse(mapping.softcap_key, "not a number");
      }
    }
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
    if (!f.imported && f.container == source.container &&
        !std::ranges::contains(stage.folds, f.fold)) {
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
    auto holds = Holds(mapping, p.gate, p.gate_absent, by_key);
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
  static const std::array all = {
      &mappings::Gemma2Mapping(),          &mappings::Gemma3Mapping(),
      &mappings::Gemma4Mapping(),          &mappings::Gemma4SafetensorsMapping(),
      &mappings::Gemma4AssistantMapping(), &mappings::Qwen38Mapping(),
      &mappings::Qwen38MtpMapping(),       &mappings::Dsv4Mapping(),
      &mappings::DsparkMapping(),          &mappings::Qwen2Mapping(),
      &mappings::Qwen35Mapping(),          &mappings::Qwen38GgufMapping(),
      &mappings::Qwen38NativeMapping(),    &mappings::Qwen38MtpNativeMapping(),
      &mappings::LlamaMapping(),           &mappings::MistralMapping()};
  return all;
}

const SourceDialect* FindSourceDialect(const Mapping& mapping, std::string_view architecture,
                                       std::string_view container) {
  const SourceDialect* found = nullptr;
  for (const auto& dialect : mapping.dialects) {
    if (dialect.architecture != architecture || dialect.container != container) continue;
    if (found) return nullptr;
    found = &dialect;
  }
  return found;
}

bool SupportsSource(const Mapping& mapping, std::string_view architecture,
                    std::string_view container) {
  if (!mapping.dialects.empty()) return FindSourceDialect(mapping, architecture, container);
  return MatchesArchitecture(mapping, architecture) &&
         std::ranges::contains(mapping.containers, container);
}

std::span<const FoldRule> SourceFolds(const Mapping& mapping, std::string_view architecture,
                                      std::string_view container) {
  if (mapping.dialects.empty()) return mapping.folds;
  const auto* dialect = FindSourceDialect(mapping, architecture, container);
  return dialect ? dialect->folds : std::span<const FoldRule>{};
}

std::expected<ResolvedMapping, std::string> ResolveSourceMapping(const Mapping& canonical,
                                                                 std::string_view architecture,
                                                                 std::string_view container) {
  ResolvedMapping out(canonical);
  if (canonical.dialects.empty()) return out;
  const auto* d = FindSourceDialect(canonical, architecture, container);
  if (!d) return Refuse(architecture, "missing or ambiguous source dialect");
  if (d->keys.size() + d->overrides.size() + d->params.size() + d->defaults.size() +
          d->tensors.size() + d->parts.size() + canonical.params.size() + canonical.tensors.size() >
      kMaxMappingRules)
    return Refuse(architecture, "too many source dialect rules");
  for (const auto& key : d->keys) {
    if (!IsText(key.canonical) || !IsText(key.source) ||
        std::ranges::count(d->keys, key.canonical, &SourceKey::canonical) != 1)
      return Refuse(architecture, "invalid or duplicate source key spelling");
    const bool scalar = key.canonical == canonical.layer_count_key ||
                        key.canonical == canonical.context_key ||
                        key.canonical == canonical.softcap_key;
    const bool parameter = std::ranges::any_of(canonical.params, [&](const auto& p) {
      return key.canonical == p.key || key.canonical == p.numerator_key ||
             key.canonical == p.divisor_key;
    });
    if (!scalar && !parameter)
      return Refuse(key.canonical, "source spelling has no supported canonical input");
  }
  const auto key = [&](std::string_view input) {
    const auto at = std::ranges::find(d->keys, input, &SourceKey::canonical);
    return at == d->keys.end() ? input : at->source;
  };
  auto& m = out.mapping;
  // Preserve the emitted source identity, including a text subarchitecture.
  m.architecture = d->architecture;
  m.source_architectures = {};
  m.containers = std::span(&d->container, 1);
  m.dialects = {};
  m.key_prefix = d->key_prefix;
  m.unprefixed_keys = d->unprefixed_keys;
  m.tensor_prefix = d->tensor_prefix;
  m.source_key_prefix = d->source_key_prefix;
  m.layer_count_key = key(m.layer_count_key);
  m.context_key = key(m.context_key);
  m.softcap_key = key(m.softcap_key);
  m.folds = d->folds;
  m.tensor_aliases = d->tensor_aliases;
  m.key_aliases = d->key_aliases;
  m.auxiliary = d->auxiliary;
  m.requirements = d->requirements;
  m.semantic = d->semantic;
  m.ignored_tensors = d->ignored_tensors;
  m.null_semantics = d->null_semantics;
  m.ignored_tensor_prefixes = d->ignored_tensor_prefixes;
  m.tensor_presence = d->tensor_presence;
  std::vector<bool> used(d->overrides.size(), false);
  for (auto p : canonical.params) {
    std::optional<std::size_t> selected;
    for (std::size_t i = 0; i < d->overrides.size(); ++i) {
      const auto& r = d->overrides[i];
      if (p.slot != r.slot || p.param != r.param ||
          (r.layer_types && p.layer_types != *r.layer_types) || (r.key && p.key != *r.key))
        continue;
      if (selected) return Refuse(architecture, "overlapping source parameter overrides");
      selected = i;
    }
    if (selected) {
      const auto& r = d->overrides[*selected];
      if (r.rule.slot != r.slot || r.rule.param != r.param)
        return Refuse(architecture, "source override changes its parameter identity");
      if (!used[*selected]) out.params.push_back(r.rule);
      used[*selected] = true;
    } else {
      p.key = key(p.key);
      p.numerator_key = key(p.numerator_key);
      p.divisor_key = key(p.divisor_key);
      out.params.push_back(p);
    }
  }
  if (std::ranges::contains(used, false))
    return Refuse(architecture, "source override has no canonical parameter");
  out.params.insert(out.params.end(), d->params.begin(), d->params.end());
  out.params.insert(out.params.end(), d->defaults.begin(), d->defaults.end());
  m.params = out.params;
  const auto tensor_rules = d->tensors.empty() ? canonical.tensors : d->tensors;
  if (!d->tensors.empty())
    for (const auto& rule : tensor_rules) {
      const bool canonical_role = std::ranges::any_of(canonical.tensors, [&](const auto& base) {
        return base.slot == rule.slot && base.role == rule.role;
      });
      if (!canonical_role)
        return Refuse(architecture, "source tensor table adds a noncanonical role");
    }
  for (const auto& alias : d->tensor_aliases) {
    const bool target_exists = std::ranges::any_of(tensor_rules, [&](const auto& rule) {
      return !rule.target && rule.pattern == alias.canonical;
    });
    if (!IsPattern(alias.source, Layered(alias.canonical)) || !target_exists ||
        std::ranges::count(d->tensor_aliases, alias.source, &TensorAlias::source) != 1)
      return Refuse(architecture, "invalid or unresolved source tensor alias");
  }
  out.tensors.assign(tensor_rules.begin(), tensor_rules.end());
  for (const auto& part : d->parts) {
    if (std::ranges::count_if(d->parts, [&](const auto& other) {
          return other.slot == part.slot && other.role == part.role;
        }) != 1)
      return Refuse(architecture, "duplicate source tensor part override");
    bool found = false;
    for (auto& tensor : out.tensors) {
      if (tensor.slot != part.slot || tensor.role != part.role) continue;
      tensor.any_part = part.any_part;
      found = true;
    }
    if (!found) return Refuse(architecture, "source part override has no canonical tensor role");
  }
  m.tensors = out.tensors;
  return out;
}

const Mapping* FindMapping(std::string_view architecture) {
  for (const Mapping* m : Mappings())
    if (m->architecture == architecture && m->version == 1) return m;
  for (const Mapping* m : Mappings())
    if (m->role == StageRole::kTarget && MatchesArchitecture(*m, architecture) && m->version == 1)
      return m;
  return nullptr;
}

const Mapping* FindMapping(std::string_view architecture, std::string_view container,
                           std::uint16_t version) {
  if (container.empty()) return nullptr;
  const Mapping* result = nullptr;
  for (const Mapping* m : Mappings()) {
    if (m->architecture != architecture || m->version != version ||
        !SupportsSource(*m, architecture, container))
      continue;
    if (result != nullptr) return nullptr;  // ambiguous registrations never pick by order
    result = m;
  }
  if (result != nullptr) return result;
  for (const Mapping* m : Mappings()) {
    if (m->role != StageRole::kTarget || !MatchesArchitecture(*m, architecture) ||
        m->version != version || !SupportsSource(*m, architecture, container))
      continue;
    if (result != nullptr) return nullptr;
    result = m;
  }
  return result;
}

std::vector<SourceValue> PolicyGates(const Mapping& mapping, const ModelSpec& model,
                                     std::string_view container) {
  std::vector<SourceValue> out;
  if (model.stages.empty()) return out;
  for (const GateRule& rule : mapping.gates) {
    if (!rule.container.empty() && rule.container != container) continue;
    bool holds = false;
    EachComponent(model.stages.front(), [&](const ComponentRef& c, const std::string&) {
      if (c.kind != rule.kind) return;
      if (rule.param.empty()) {
        holds = true;
        return;
      }
      if (const auto* p = c.Find(rule.param))
        if (const auto* flag = std::get_if<bool>(&p->value); flag && *flag) holds = true;
    });
    if (holds) out.push_back({.key = std::string(rule.key), .value = true});
  }
  return out;
}

std::expected<std::uint32_t, std::string> Vocabulary(const ModelSpec& model, const Source& source) {
  if (model.stages.empty() || !model.stages.front().heads.output)
    return Refuse("vocabulary", "target has no output head");
  std::optional<std::uint32_t> rows;
  for (const WeightBinding& w : model.stages.front().heads.output->weights) {
    if (w.role != "weight" || !w.stage.empty()) continue;
    const auto tensor = std::ranges::find(source.tensors, w.tensor, &SourceTensor::name);
    if (tensor == source.tensors.end() || tensor->format.shape.rank != 2 ||
        tensor->format.shape.ne[1] == 0 || tensor->format.shape.ne[1] > kMaxWidth)
      return Refuse("vocabulary", "output head lacks a bounded matrix");
    auto count = static_cast<std::uint32_t>(tensor->format.shape.ne[1]);
    if (rows && *rows != count) return Refuse("vocabulary", "head parts disagree on rows");
    rows = count;
  }
  if (!rows) return Refuse("vocabulary", "output head has no weight");
  const ComponentRef& head = *model.stages.front().heads.output;
  if (head.kind == "head.linear") {
    if (const Param* p = head.Find("rows")) {
      const auto* selected = std::get_if<std::int64_t>(&p->value);
      if (!selected || *selected < 0 || std::cmp_greater(*selected, *rows))
        return Refuse("vocabulary", "output head rows exceed its weight");
      if (*selected) return static_cast<std::uint32_t>(*selected);
    }
  }
  return *rows;
}

std::expected<std::vector<SourceValue>, std::string> ExtractValues(const Mapping& mapping,
                                                                   const ModelSpec& target,
                                                                   const Source& target_source,
                                                                   const Source& source) {
  if (!mapping.draft || target.stages.empty() || target.stages.front().role != StageRole::kTarget)
    return Refuse(mapping.architecture, "drafter extraction requires a target spec");
  if (auto valid = Validate(target); !valid) return std::unexpected(valid.error());
  const Stage& stage = target.stages.front();
  std::vector<SourceValue> out;
  for (const ValueRule& rule : mapping.draft->values) {
    const std::string canonical_key = CanonicalKey(mapping, rule.key);
    if (std::ranges::contains(out, canonical_key, &SourceValue::key)) continue;
    std::optional<ParamValue> value;
    switch (rule.origin) {
      case ValueOrigin::kLiteral:
        switch (rule.type) {
          case ValueType::kInt:
            value = rule.value.integer;
            break;
          case ValueType::kReal:
            value = rule.value.real;
            break;
          case ValueType::kBool:
            value = rule.value.flag;
            break;
          case ValueType::kString:
            value = std::string(rule.value.text);
            break;
          case ValueType::kInts:
            return Refuse(rule.key, "a list literal is unsupported");
        }
        break;
      case ValueOrigin::kContext:
        value = std::int64_t{stage.context};
        break;
      case ValueOrigin::kWidth:
        value = static_cast<std::int64_t>(stage.stream.width);
        break;
      case ValueOrigin::kStreams:
        value = std::int64_t{stage.stream.streams};
        break;
      case ValueOrigin::kVocabulary: {
        auto rows = Vocabulary(target, target_source);
        if (!rows) return std::unexpected(rows.error());
        value = std::int64_t{*rows};
        break;
      }
      case ValueOrigin::kComponent:
        // Last matching layer/component: MTP follows the target's last block
        // of the declared component kind. No target size or layer number is fixed.
        EachComponent(stage, [&](const ComponentRef& c, const std::string&) {
          if (c.kind == rule.kind)
            if (const Param* p = c.Find(rule.param)) value = p->value;
        });
        break;
      case ValueOrigin::kLayerCount: {
        std::set<std::uint32_t> layers;
        for (const SourceTensor& tensor : source.tensors)
          if (auto layer = Match(rule.tensor, tensor.name, kMaxLayers)) layers.insert(*layer);
        if (!layers.empty() && *layers.rbegin() + 1 == layers.size())
          value = static_cast<std::int64_t>(layers.size());
        break;
      }
      case ValueOrigin::kTensor:
        if (const auto t = std::ranges::find(source.tensors, rule.tensor, &SourceTensor::name);
            t != source.tensors.end() && rule.dimension < t->format.shape.rank)
          value = static_cast<std::int64_t>(t->format.shape.ne[rule.dimension]);
        break;
      case ValueOrigin::kTensorPresent:
        // The drafter's own layout, never the target's: a stacked or scaled
        // drafter composes with any target of its geometry.
        value = std::ranges::any_of(source.tensors, [&](const SourceTensor& tensor) {
          return Match(rule.tensor, tensor.name, kMaxLayers).has_value();
        });
        break;
    }
    if (!value) {
      if (rule.optional) continue;
      return Refuse(rule.key, "target spec or drafter tensor lacks the declared extraction fact");
    }
    if (rule.at_most) {
      auto* i = std::get_if<std::int64_t>(&*value);
      if (!i || *i <= 0) return Refuse(rule.key, "a nonpositive extraction bound");
      *i = std::min<std::int64_t>(*i, rule.at_most);
    }
    if (rule.list) {
      const auto* i = std::get_if<std::int64_t>(&*value);
      if (!i) return Refuse(rule.key, "a noninteger list extraction");
      value = std::vector<std::int64_t>{*i};
    }
    out.push_back({.key = canonical_key, .value = std::move(*value)});
  }
  return out;
}

std::vector<std::string> SourceKeys(const Mapping& m) {
  std::vector<std::string> keys;
  const auto add = [&](std::string_view key) {
    if (key.empty()) return;
    const std::string canonical = CanonicalKey(m, key);
    if (std::ranges::find(keys, canonical) == keys.end()) keys.push_back(canonical);
  };
  if (m.draft) {
    add(m.draft->depth_key);
    add(m.draft->fill_key);
  }
  add(m.layer_count_key);
  add(m.context_key);
  add(m.softcap_key);
  add(m.layer_type_key);
  add(m.layer_type_interval_key);
  add(m.layer_flags_key);
  for (const SlotRule& s : m.slots) {
    add(s.gate);
    add(s.layers_key);
    add(s.layers_alias);
    add(s.count_key);
  }
  for (const ParamRule& p : m.params) {
    add(p.key);
    add(p.numerator_key);
    add(p.divisor_key);
  }
  for (const auto& list : m.lists) {
    add(list.key);
    add(list.sections_of);
    add(list.sections_multiplier_key);
  }
  for (const ReadRule& r : m.reads) add(r.layers_key);
  for (const PolicyRule& p : m.policies) add(p.gate);
  if (m.role == StageRole::kTarget && m.companion != nullptr &&
      m.companion->role == StageRole::kDrafter) {
    for (const std::string& k : SourceKeys(*m.companion)) {
      if (std::ranges::find(keys, k) == keys.end()) keys.push_back(k);
    }
  }
  return keys;
}

Status ValidateMapping(const Mapping& m) {
  const std::string where = std::format("mapping {}", m.architecture);
  if (m.dialects.size() > kMaxMappingRules) return Refuse(where, "too many source dialects");
  for (const auto& d : m.dialects) {
    if (!MatchesArchitecture(m, d.architecture) ||
        !std::ranges::contains(m.containers, d.container))
      return Refuse(where, "source dialect outside the mapping's architecture or containers");
    auto selected = ResolveSourceMapping(m, d.architecture, d.container);
    if (!selected) return std::unexpected(selected.error());
    if (auto ok = ValidateMapping(selected->mapping); !ok) return ok;
  }
  if (!IsArchitecture(m.architecture) || m.version == 0 ||
      static_cast<std::uint8_t>(m.pipeline) > static_cast<std::uint8_t>(PipelineKind::kDecision)) {
    return Refuse(where, "an invalid architecture, version or pipeline");
  }
  for (std::size_t i = 0; i < m.source_architectures.size(); ++i)
    if (!IsArchitecture(m.source_architectures[i]) || m.source_architectures[i] == m.architecture ||
        std::ranges::count(m.source_architectures, m.source_architectures[i]) != 1)
      return Refuse(where, "invalid or duplicate source architecture");
  for (std::size_t i = 0; i < m.containers.size(); ++i)
    if (!IsName(m.containers[i]) || std::ranges::count(m.containers, m.containers[i]) != 1)
      return Refuse(where, "invalid or duplicate mapping container");
  if ((!m.context_key.empty() && !IsText(m.context_key)) ||
      (!m.softcap_key.empty() && !IsText(m.softcap_key))) {
    return Refuse(where, "an invalid context or softcap key");
  }
  if (!m.source_key_prefix.empty() &&
      (!m.source_key_prefix.ends_with('.') ||
       !IsText(m.source_key_prefix.substr(0, m.source_key_prefix.size() - 1))))
    return Refuse(where, "invalid source metadata nesting prefix");
  for (const SemanticTensorRule& s : m.semantic_tensors)
    if (!IsPattern(s.tensor, true) || !IsText(s.key) || s.size > kMaxList)
      return Refuse(where, "an invalid semantic tensor rule");
  for (const GateRule& rule : m.gates)
    if (!IsText(rule.key) || !IsKind(rule.kind) || (!rule.param.empty() && !IsName(rule.param)) ||
        (!rule.container.empty() && !IsName(rule.container)))
      return Refuse(where, "invalid derived policy gate");
  if (m.draft) {
    if (!IsArchitecture(m.draft->target_architecture))
      return Refuse(where, "drafter lacks a target architecture");
    for (const ValueRule& rule : m.draft->values)
      if (!IsText(rule.key) || rule.dimension >= format::kMaxDims ||
          static_cast<unsigned>(rule.origin) > static_cast<unsigned>(ValueOrigin::kTensorPresent) ||
          static_cast<unsigned>(rule.type) > static_cast<unsigned>(ValueType::kInts) ||
          (rule.origin == ValueOrigin::kComponent && (!IsKind(rule.kind) || !IsName(rule.param))) ||
          (rule.origin == ValueOrigin::kTensor && !IsText(rule.tensor)) ||
          (rule.origin == ValueOrigin::kLayerCount && !IsPattern(rule.tensor, true)) ||
          (rule.origin == ValueOrigin::kTensorPresent &&
           (!IsPattern(rule.tensor, true) || rule.type != ValueType::kBool)))
        return Refuse(where, "invalid drafter value extraction rule");
  }
  if (!IsText(m.layer_count_key) || m.layer_types.empty() ||
      m.layer_types.size() > kMaxLayerTypes || !std::ranges::all_of(m.layer_types, Lower)) {
    return Refuse(where, "an invalid layer count key or layer types");
  }
  if (!m.layer_type_interval_key.empty() &&
      (!IsText(m.layer_type_interval_key) || m.layer_types.size() < 2 ||
       m.layer_types.front() == m.layer_types.back() ||
       !std::ranges::all_of(m.layer_types.substr(0, m.layer_types.size() - 1),
                            [&](char type) { return type == m.layer_types.front(); })))
    return Refuse(where, "an invalid interval key or repeating layer schedule");
  if ((m.layer_type_key.empty() && !m.layer_type_codes.empty()) ||
      (!m.layer_type_key.empty() &&
       (!IsText(m.layer_type_key) ||
        !std::ranges::all_of(m.layer_type_codes, [&](const LayerTypeCode& c) {
          return m.layer_types.contains(c.type) && (c.label.empty() || IsName(c.label)) &&
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
  if (m.companion != nullptr) {
    if (m.role != StageRole::kTarget || m.companion->role != StageRole::kDrafter) {
      return Refuse(where, "a companion must be a drafter mapping on a target mapping");
    }
    if (auto ok = ValidateMapping(*m.companion); !ok) return ok;
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
    if (s.alias_index_base > 1 ||
        (!s.layers_alias.empty() && (s.layers_key.empty() || !IsText(s.layers_alias))) ||
        (s.layers_alias.empty() && s.alias_index_base != 0))
      return Refuse(at, "invalid alternate layer selector or index base");
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
  std::map<std::string_view, std::vector<const SlotRule*>> patterns;
  std::map<std::string_view, std::vector<const SlotRule*>> target_patterns;
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
    auto& seen_slots = names[t.pattern];
    const bool exclusive_with_all = std::ranges::all_of(seen_slots, [&](const SlotRule* prior) {
      if (!Exclusive(*prior, *slot)) return false;
      if (prior->site == Site::kLayer && slot->site == Site::kLayer &&
          !prior->layer_types.empty() && !slot->layer_types.empty() && !Layered(t.pattern)) {
        return false;
      }
      return true;
    });
    if (!valid || !exclusive_with_all) {
      return Refuse(at, "not a pattern of its slot's site, or twice");
    }
    seen_slots.push_back(slot);
    if (!t.fallback.empty() && !IsPattern(t.fallback, Layered(t.pattern))) {
      return Refuse(at, "a fallback that is not a pattern of its rule's form");
    }
    const std::string_view slot_types =
        slot->layer_types.empty() ? m.layer_types : slot->layer_types;
    if (layered ? (!TypesWithin(t.layer_types, m.layer_types) ||
                   !TypesWithin(t.layer_types, slot_types))
                : !t.layer_types.empty()) {
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
    const std::string_view types = t.layer_types.empty() ? slot_types : t.layer_types;
    if (std::ranges::any_of(types, [&](char c) { return bound.contains(c); })) {
      return Refuse(at, "an invalid role or part, or one bound twice");
    }
    bound += types;
  }
  for (const ParamRule& r : m.params) {
    const std::string at = std::format("{} parameter {}.{}", where, r.slot, r.param);
    if (static_cast<std::uint8_t>(r.type) > static_cast<std::uint8_t>(ValueType::kInts) ||
        (r.type == ValueType::kInts && (r.key.empty() || r.per_layer)) ||
        static_cast<std::uint8_t>(r.transform) > static_cast<std::uint8_t>(Transform::kProduct) ||
        (r.transform != Transform::kNone && r.transform != Transform::kProduct &&
         r.type != ValueType::kReal) ||
        (r.transform == Transform::kProduct &&
         (r.numerator_key.empty() || (r.type != ValueType::kInt && r.type != ValueType::kReal)))) {
      return Refuse(at, "an unknown type or transform, or a transform of a non-real");
    }
    if (!r.key.empty() && !IsText(r.key)) {
      return Refuse(at, "an invalid key");
    }
    if (!r.tensor.empty() &&
        (!IsPattern(r.tensor, r.tensor.contains(kLayerField)) || r.slot.empty() ||
         IsPseudo(r.slot) || !r.key.empty() || !r.numerator_key.empty() || !r.divisor_key.empty() ||
         (r.type != ValueType::kBool && r.type != ValueType::kInt) ||
         r.transform != Transform::kNone || (r.type == ValueType::kBool && !r.optional) ||
         r.per_layer || !r.unless_fold.empty())) {
      return Refuse(at, "an invalid tensor presence parameter rule");
    }
    if (!r.flag_text.empty() && (r.type != ValueType::kBool || !r.slot.empty() || r.key.empty() ||
                                 !r.value.text.empty() || !IsName(r.flag_text))) {
      return Refuse(at, "an invalid boolean requirement string alias");
    }
    if (!r.value.text.empty() && r.type == ValueType::kBool &&
        (r.slot.empty() || r.key.empty() || !IsName(r.value.text) || !r.optional || r.per_layer)) {
      return Refuse(at, "an invalid string-matched flag parameter rule");
    }
    if (r.numerator_key.empty() != r.divisor_key.empty()) {
      return Refuse(at, "a ratio requires both numerator and divisor keys");
    }
    if (!r.numerator_key.empty() && (!IsText(r.numerator_key) || !IsText(r.divisor_key))) {
      return Refuse(at, "an invalid numerator or divisor key");
    }
    if (!r.numerator_key.empty() && r.type != ValueType::kInt && r.type != ValueType::kReal) {
      return Refuse(at, "a ratio must be an integer or real");
    }
    if (!r.numerator_key.empty() && r.per_layer) {
      return Refuse(at, "a ratio cannot be per-layer");
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
    if (!r.unless_fold.empty() && FindFold(r.unless_fold) == nullptr) {
      return Refuse(at, "unless a fold the catalog does not have");
    }
    if (r.slot.empty()) {
      if (r.type == ValueType::kInts || r.key.empty() || !r.param.empty() ||
          !r.layer_types.empty() || !r.numerator_key.empty() || !r.divisor_key.empty() ||
          !r.tensor.empty() || r.transform != Transform::kNone || r.per_layer ||
          !r.unless_fold.empty()) {
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
    if (FindFold(f.fold) == nullptr || !IsName(f.container)) {
      return Refuse(std::format("{} fold {}", where, f.fold),
                    "not in the fold catalog, or no container");
    }
  }
  return {};
}

std::expected<ModelSpec, std::string> Synthesize(const Mapping& mapping, const Source& source,
                                                 std::span<const SourceValue> values) {
  if (!mapping.dialects.empty()) {
    auto selected = ResolveSourceMapping(mapping, source.architecture, source.container);
    if (!selected) return std::unexpected(selected.error());
    return Synthesize(selected->mapping, source, values);
  }
  if (mapping.role != StageRole::kTarget) {
    return Refuse(std::format("mapping {}", mapping.architecture),
                  "a drafter's mapping, which composes with its target");
  }
  if (mapping.companion != nullptr) {
    const std::string stage_prefix = std::format("{}.", mapping.companion->stage);
    const auto matches_rules = [](std::span<const TensorRule> rules, std::string_view name) {
      for (const TensorRule& rule : rules) {
        if (rule.target) continue;
        if (Match(rule.pattern, name, kMaxLayers).has_value() ||
            (!rule.fallback.empty() && Match(rule.fallback, name, kMaxLayers).has_value())) {
          return true;
        }
      }
      return false;
    };
    const bool has_prefixed_companion =
        std::ranges::any_of(source.tensors, [&](const SourceTensor& t) {
          return t.name.starts_with(stage_prefix) &&
                 matches_rules(mapping.companion->tensors, t.name.substr(stage_prefix.size()));
        });
    const bool has_unprefixed_companion =
        !has_prefixed_companion && std::ranges::any_of(source.tensors, [&](const SourceTensor& t) {
          return matches_rules(mapping.companion->tensors, t.name) &&
                 !matches_rules(mapping.tensors, t.name);
        });
    if (has_prefixed_companion || has_unprefixed_companion) {
      Source target_source = source;
      Source companion_source = source;
      companion_source.architecture = mapping.companion->architecture;
      target_source.tensors.clear();
      companion_source.tensors.clear();
      for (const SourceTensor& t : source.tensors) {
        if (has_prefixed_companion) {
          if (t.name.starts_with(stage_prefix) &&
              matches_rules(mapping.companion->tensors, t.name.substr(stage_prefix.size()))) {
            SourceTensor stripped = t;
            stripped.name = t.name.substr(stage_prefix.size());
            companion_source.tensors.push_back(std::move(stripped));
          } else {
            target_source.tensors.push_back(t);
          }
        } else {
          if (matches_rules(mapping.companion->tensors, t.name)) {
            companion_source.tensors.push_back(t);
          } else {
            target_source.tensors.push_back(t);
          }
        }
      }
      auto made = MakeStage(mapping, target_source, values, {}, {});
      if (!made) {
        return std::unexpected(made.error());
      }
      made->stage.name = "target";
      made->stage.role = StageRole::kTarget;
      ModelSpec target{.pipeline = mapping.pipeline};
      target.stages.push_back(std::move(made->stage));
      Canonicalize(target);
      if (auto ok = Validate(target); !ok) {
        return std::unexpected(ok.error());
      }
      auto composed = ComposeDrafter(target, mapping, *mapping.companion, companion_source, values);
      if (!composed) {
        return composed;
      }
      if (has_prefixed_companion) {
        EachComponent(composed->stages.back(), [&](ComponentRef& component, const std::string&) {
          for (WeightBinding& w : component.weights) {
            if (w.stage.empty()) {
              w.tensor = stage_prefix + w.tensor;
            }
          }
        });
      }
      composed->stages.back().artifact = 0;
      std::vector<std::string> shared_folds;
      for (const Stage& s : composed->stages) {
        if (s.artifact == 0) {
          for (const std::string& f : s.folds) {
            if (std::ranges::find(shared_folds, f) == shared_folds.end()) {
              shared_folds.push_back(f);
            }
          }
        }
      }
      std::ranges::sort(shared_folds);
      for (Stage& s : composed->stages) {
        if (s.artifact == 0) {
          s.folds = shared_folds;
        }
      }
      Canonicalize(*composed);
      if (auto ok = Validate(*composed); !ok) {
        return std::unexpected(ok.error());
      }
      return composed;
    }
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
  // A target rule names an anchor tensor, not a physical encoding part. Borrow
  // its complete logical role so split tables keep all of their source parts.
  Status borrowed;
  EachComponent(made->stage, [&](ComponentRef& component, const std::string&) {
    if (!borrowed) return;
    std::vector<WeightBinding> bindings;
    for (const WeightBinding& requested : component.weights) {
      if (requested.stage != head.name) {
        bindings.push_back(requested);
        continue;
      }
      std::optional<std::vector<WeightBinding>> found;
      EachComponent(head, [&](const ComponentRef& owner, const std::string&) {
        if (!borrowed) return;
        for (const WeightBinding& anchor : owner.weights) {
          if (anchor.tensor != requested.tensor || anchor.experts != requested.experts) continue;
          std::vector<WeightBinding> group;
          for (const WeightBinding& part : owner.weights) {
            if (part.role != anchor.role) continue;
            auto binding = part;
            binding.role = requested.role;
            if (binding.stage.empty()) binding.stage = head.name;
            group.push_back(std::move(binding));
          }
          if (found && *found != group) {
            borrowed = Refuse(requested.tensor, "ambiguous target weight role");
            return;
          }
          found = std::move(group);
        }
      });
      if (!borrowed) return;
      if (!found) {
        borrowed = Refuse(requested.tensor, "no target weight role contains this anchor");
        return;
      }
      if (bindings.size() > kMaxWeights || found->size() > kMaxWeights - bindings.size()) {
        borrowed = Refuse(requested.tensor, "borrowed target role exceeds the weight bound");
        return;
      }
      bindings.insert(bindings.end(), found->begin(), found->end());
    }
    component.weights = std::move(bindings);
  });
  if (!borrowed) return std::unexpected(borrowed.error());
  ModelSpec spec = target;
  std::uint32_t artifact = 0;
  for (const Stage& s : spec.stages) {
    artifact = std::max(artifact, s.artifact + 1);
  }
  Stage& stage = made->stage;
  stage.name = std::string(drafter.stage);
  stage.role = StageRole::kDrafter;
  stage.artifact = artifact;
  if (drafter.draft) {
    const auto& recipe = *drafter.draft;
    const auto integer = [&](std::string_view key, std::uint64_t least,
                             std::uint64_t most) -> std::expected<std::uint64_t, std::string> {
      const auto found = std::ranges::find_if(
          values, [&](const SourceValue& v) { return MatchesCanonicalKey(drafter, key, v.key); });
      return Count(found == values.end() ? nullptr : &found->value, key, least, most);
    };
    switch (recipe.shape) {
      case DraftRecipe::Shape::kAnchored:
        stage.drafting = AnchoredDraft{};
        break;
      case DraftRecipe::Shape::kBlock: {
        auto depth = integer(recipe.depth_key, 1, UINT32_MAX - 1);
        auto fill = integer(recipe.fill_key, 0, INT32_MAX);
        if (!depth) return std::unexpected(depth.error());
        if (!fill) return std::unexpected(fill.error());
        stage.drafting =
            BlockDraft{static_cast<std::uint32_t>(*depth), static_cast<std::int32_t>(*fill)};
        break;
      }
      case DraftRecipe::Shape::kMtp: {
        stage.drafting = MtpDraft{};
        break;
      }
    }
  }
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
      const auto found = std::ranges::find_if(values, [&](const SourceValue& v) {
        return MatchesCanonicalKey(drafter, r.layers_key, v.key);
      });
      const auto* list =
          found == values.end() ? nullptr : std::get_if<std::vector<std::int64_t>>(&found->value);
      const auto count = static_cast<std::int64_t>(target_stage.layers.size());
      const std::int64_t entry =
          list != nullptr && r.index < list->size() ? (*list)[r.index] : std::int64_t{-1};
      if (r.entering ? entry <= 0 || entry > count : entry < 0 || entry >= count) {
        return Refuse(where,
                      std::format("read {}: {} names no layer of the target's {} ({})", r.name,
                                  r.layers_key, count, r.entering ? "entering" : "after"));
      }
      const std::int64_t layer = r.entering ? entry - 1 : entry;
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

}  // namespace llmp::spec
