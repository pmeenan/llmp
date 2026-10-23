// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#include "import/normalize.h"

#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <format>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <type_traits>
#include <utility>

#include "base/json.h"
#include "base/sha256.h"
#include "components/component.h"
#include "components/step.h"
#include "import/expert_relayout.h"
#include "import/files.h"
#include "import/parameter_folds.h"
#include "import/readers/gguf.h"
#include "import/sampling_defaults.h"
#include "import/transforms.h"
#include "spec/stored.h"

namespace llmp::import {
namespace nd = normalization_detail;
namespace json = base::json;
namespace {
using Failure = std::unexpected<Error>;
Failure Bad(std::string_view reason, ItemKind kind = ItemKind::kNone, std::uint64_t item = kNoItem,
            Rule rule = Rule::kContainer) {
  return Failure(Error{.rule = rule, .reason = reason, .kind = kind, .item = item});
}
Failure Bad(std::string_view reason, Rule rule) {
  return Failure(Error{.rule = rule, .reason = reason});
}
struct Arithmetic {
  bool ok = true;
  std::uint64_t Add(std::uint64_t a, std::uint64_t b) {
    std::uint64_t out = 0;
    if (__builtin_add_overflow(a, b, &out)) ok = false;
    return out;
  }
  std::uint64_t Mul(std::uint64_t a, std::uint64_t b) {
    std::uint64_t out = 0;
    if (__builtin_mul_overflow(a, b, &out)) ok = false;
    return out;
  }
};
struct Key {
  MetadataClass kind = MetadataClass::kAuxiliary;
  std::string_view canonical{};
  std::uint8_t types = 0;
  std::uint32_t auxiliary = 0;
};
std::optional<Key> Classify(const spec::Mapping& t, std::string_view container,
                            std::string_view key, bool is_null = false,
                            const Checkpoint* source = nullptr) {
  const auto& m = t;
  if ((container == "gguf" &&
       (key == "general.architecture" || key == "general.alignment" || key == "split.no" ||
        key == "split.count" || key == "split.tensors.count")) ||
      ((container == "exl3" || container == "safetensors" || container == "mlx") &&
       key == "model_type"))
    return Key{.kind = MetadataClass::kStructural, .canonical = key};
  std::string_view stripped = key;
  const bool has_key_prefix = !m.key_prefix.empty() && key.size() > m.key_prefix.size() &&
                              key.starts_with(m.key_prefix) && key[m.key_prefix.size()] == '.';
  if (has_key_prefix) stripped = key.substr(m.key_prefix.size() + 1);
  if (is_null && (std::ranges::contains(t.null_semantics, key) ||
                  (has_key_prefix && std::ranges::contains(t.null_semantics, stripped))))
    return Key{.auxiliary = static_cast<std::uint32_t>(t.auxiliary.size())};
  auto canonical = key;
  bool aliased = false;
  for (const auto& alias : t.key_aliases) {
    if ((m.key_prefix.empty() && key == alias.source) ||
        (has_key_prefix && stripped == alias.source)) {
      if (!alias.unless_present.empty() && source && source->FindValue(alias.unless_present))
        continue;
      canonical = alias.canonical;
      aliased = true;
      break;
    }
  }
  const auto match_key = [&](std::string_view rule_key) {
    return aliased ? spec::MatchesCanonicalKey(m, rule_key, canonical)
                   : spec::IsCanonicalKey(m, rule_key, canonical);
  };
  const auto unprefixed = spec::UnprefixedKey(m, canonical);
  for (const auto& requirement : t.tensor_presence)
    if (canonical == requirement.flag)
      return Key{.kind = MetadataClass::kSelected, .canonical = canonical, .types = nd::kFlag};
  for (const auto& requirement : t.requirements)
    if (match_key(requirement.key) &&
        (requirement.require_equal || !requirement.equals_key.empty() || requirement.positive ||
         requirement.boolean)) {
      const std::uint8_t kt = nd::KeyTypes(m, canonical, aliased);
      return Key{.kind = MetadataClass::kSelected,
                 .canonical = unprefixed,
                 .types = static_cast<std::uint8_t>(kt != 0 ? kt : nd::kInteger)};
    }
  if (const auto types = nd::KeyTypes(m, canonical, aliased); types != 0)
    return Key{.kind = MetadataClass::kSelected, .canonical = unprefixed, .types = types};
  for (const auto& list : t.lists)
    if (match_key(list.key))
      return Key{.kind = MetadataClass::kSelected, .canonical = unprefixed, .types = nd::kIntegers};
  std::string_view nested = key;
  if (!m.source_key_prefix.empty() && nested.starts_with(m.source_key_prefix))
    nested.remove_prefix(m.source_key_prefix.size());
  const auto auxiliary_match = [&](std::string_view candidate, std::string_view prefix) {
    return prefix.ends_with('.') ? candidate.starts_with(prefix) : candidate == prefix;
  };
  for (std::size_t i = 0; i < t.auxiliary.size(); ++i)
    if (auxiliary_match(key, t.auxiliary[i].prefix))
      return Key{.auxiliary = static_cast<std::uint32_t>(i)};
  for (std::size_t i = 0; nested != key && i < t.auxiliary.size(); ++i)
    if (auxiliary_match(nested, t.auxiliary[i].prefix))
      return Key{.auxiliary = static_cast<std::uint32_t>(i)};
  const auto semantic_match = [&](std::string_view candidate, std::string_view semantic) {
    return semantic.ends_with('*') ? candidate.starts_with(semantic.substr(0, semantic.size() - 1))
           : semantic.ends_with('.')
               ? candidate.starts_with(semantic)
               : (candidate == semantic ||
                  (candidate.starts_with(semantic) && candidate.size() > semantic.size() &&
                   candidate[semantic.size()] == '.'));
  };
  for (const auto semantic : t.semantic)
    if (semantic_match(key, semantic) || semantic_match(nested, semantic)) return std::nullopt;
  // The architecture's own namespace (a GGUF's "<general.architecture>.")
  // and llmpalooza's are math-bearing wherever a table maps them.
  const std::string_view own = m.architecture;
  if ((key.size() > own.size() && key.starts_with(own) && key[own.size()] == '.') ||
      key.starts_with("llmp."))
    return std::nullopt;
  return Key{.auxiliary = static_cast<std::uint32_t>(t.auxiliary.size())};
}
bool SelectedValue(const MetaValue& value, std::uint8_t types) {
  if (std::holds_alternative<std::int64_t>(value)) return (types & nd::kInteger) != 0;
  if (const auto* real = std::get_if<double>(&value))
    return (types & nd::kReal) != 0 && std::isfinite(*real);
  if (std::holds_alternative<bool>(value)) return (types & nd::kFlag) != 0;
  if (const auto* text = std::get_if<std::string>(&value))
    return (types & nd::kText) != 0 && spec::IsSourceKey(*text);
  if (const auto* list = std::get_if<std::vector<std::int64_t>>(&value))
    return (types & nd::kIntegers) != 0 && !list->empty() && list->size() <= nd::kLayerLimit;
  if (const auto* list = std::get_if<std::vector<std::string>>(&value))
    return (types & nd::kText) != 0 && !list->empty() && list->size() <= nd::kLayerLimit;
  if (const auto* list = std::get_if<std::vector<double>>(&value))
    return (types & nd::kReals) != 0 && !list->empty() && list->size() <= nd::kLayerLimit &&
           std::ranges::all_of(*list, [](double x) { return std::isfinite(x); });
  return false;
}
// Reader accounting, with validation before scanning any array/string. No
// result allocation and no repeated vocabulary scans for parameter rules.
std::optional<std::uint64_t> MetaBytes(const MetaValue& value) {
  Arithmetic a;
  std::uint64_t bytes = 0;
  bool valid = true;
  const auto text = [&](const std::string& s) {
    if (s.size() > kMaxGgufString) {
      valid = false;
      return;
    }
    if (json::FirstInvalidUtf8(s)) valid = false;
    bytes = a.Add(bytes, a.Add(32, s.size()));
  };
  std::visit(
      [&](const auto& v) {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, std::string>)
          text(v);
        else if constexpr (std::is_same_v<T, std::vector<std::string>>) {
          if (v.size() > kMaxGgufArray) {
            valid = false;
            return;
          }
          for (const auto& s : v) {
            text(s);
            if (!valid || !a.ok || bytes > kMaxGgufMetadataBytes) {
              valid = false;
              return;
            }
          }
        } else if constexpr (std::is_same_v<T, std::vector<double>> ||
                             std::is_same_v<T, std::vector<std::int64_t>>) {
          if (v.size() > kMaxGgufArray) {
            valid = false;
            return;
          }
          bytes = a.Mul(v.size(), 8);
          if constexpr (std::is_same_v<T, std::vector<double>>)
            for (double d : v)
              if (!std::isfinite(d)) valid = false;
        } else if constexpr (std::is_same_v<T, double>) {
          valid = std::isfinite(v);
          bytes = 8;
        } else
          bytes = 8;
      },
      value);
  return valid && a.ok && bytes <= kMaxGgufMetadataBytes ? std::optional(bytes) : std::nullopt;
}

struct ResolvedName {
  std::string_view pattern{}, digits{};
  format::PartRole part = format::PartRole::kData;
  bool experts = false;
  std::optional<std::uint32_t> index{};
  bool is_expert = false, is_shard = false;
};
std::optional<ResolvedName> TensorName(const spec::Mapping& t, std::string_view container,
                                       std::string_view name, std::uint32_t layers,
                                       format::PartRole source_part = format::PartRole::kData,
                                       bool prefixed = true) {
  const auto& m = t;
  if (prefixed && !m.tensor_prefix.empty()) {
    if (!name.starts_with(m.tensor_prefix)) return std::nullopt;
    name.remove_prefix(m.tensor_prefix.size());
  }
  std::optional<ResolvedName> found;
  for (const auto& rule : m.tensors) {
    if (rule.target) continue;
    if (const auto match = nd::Match(rule.pattern, name, layers)) {
      if (found && found->pattern != rule.pattern) return std::nullopt;
      found = ResolvedName{rule.pattern,
                           match->digits,
                           (rule.any_part || container == "safetensors" || container == "mlx")
                               ? source_part
                               : rule.part,
                           rule.experts,
                           match->index,
                           match->is_expert,
                           match->is_shard};
    }
  }
  for (const auto& alias : t.tensor_aliases)
    if (const auto match = nd::Match(alias.source, name, layers)) {
      if (found) return std::nullopt;
      for (const auto& rule : m.tensors)
        if (!rule.target && rule.pattern == alias.canonical)
          found = ResolvedName{alias.canonical,
                               match->digits,
                               (rule.any_part || container == "safetensors" || container == "mlx")
                                   ? source_part
                                   : rule.part,
                               rule.experts,
                               match->index,
                               match->is_expert,
                               match->is_shard};
    }
  if (!found && source_part != format::PartRole::kData && source_part != format::PartRole::kCodes) {
    for (std::string_view suffix : {".weight_scale_2", ".weight_scale", ".weight_bias",
                                    ".weight_zero", ".scales", ".biases", ".zeros"}) {
      if (!name.ends_with(suffix)) continue;
      const std::string base = std::string(name.substr(0, name.size() - suffix.size())) + ".weight";
      if (auto weight = TensorName(t, container, base, layers, format::PartRole::kCodes, false)) {
        if (!weight->digits.empty())
          weight->digits = name.substr(
              static_cast<std::size_t>(weight->digits.data() - base.data()), weight->digits.size());
        weight->part = source_part;
        return weight;
      }
    }
  }
  return found;
}
bool IgnoredTensor(const spec::Mapping& table, std::string_view name, std::uint32_t layers) {
  if (std::ranges::any_of(table.ignored_tensor_prefixes,
                          [&](auto prefix) { return name.starts_with(prefix); }))
    return true;
  return std::ranges::any_of(table.ignored_tensors, [&](const auto pattern) {
    return nd::Match(pattern, name, layers).has_value();
  });
}
std::string Expand(ResolvedName n) {
  const auto at = n.pattern.find(spec::kLayerField);
  if (at == std::string_view::npos) return std::string(n.pattern);
  std::string out;
  out.reserve(n.pattern.size() - spec::kLayerField.size() + n.digits.size());
  out.append(n.pattern.substr(0, at));
  out.append(n.digits);
  out.append(n.pattern.substr(at + spec::kLayerField.size()));
  return out;
}

// Container part suffixes describe encoding geometry, independently of the
// architecture's tensor dialect. The weight's mapping authorizes its parts.
std::string_view PartSuffix(format::PartRole part) {
  using enum format::PartRole;
  switch (part) {
    case kBlockScales:
      return ".weight_scale";
    case kTensorScale:
      return ".weight_scale_2";
    case kBiases:
      return ".weight_bias";
    case kZeros:
      return ".weight_zero";
    default:
      return ".weight";
  }
}
std::string PartName(std::string_view name, format::PartRole part) {
  if (part == format::PartRole::kData || part == format::PartRole::kCodes) return std::string(name);
  if (name.ends_with(".weight"))
    return std::string(name.substr(0, name.size() - 7)) + std::string(PartSuffix(part));
  return std::string(name) + std::string(PartSuffix(part));
}
std::string WeightName(std::string_view name, format::PartRole part) {
  if (part == format::PartRole::kData || part == format::PartRole::kCodes) return std::string(name);
  for (std::string_view suffix : {".weight_scale_2", ".weight_scale", ".weight_bias",
                                  ".weight_zero", ".scales", ".biases", ".zeros"})
    if (name.ends_with(suffix))
      return std::string(name.substr(0, name.size() - suffix.size())) + ".weight";
  return std::string(name);
}
bool IndexedSource(const Checkpoint& source) {
  return source.container == "safetensors" || source.container == "mlx";
}
bool SourceDescriptor(const Tensor& tensor) {
  const auto bytes = SourcePartBytes(tensor);
  const auto physical = format::EncodingOfDtype(tensor.dtype);
  if (!bytes || !physical || format::Info(*physical).quantized) return false;
  const auto element = format::BlockBytes(*physical);
  if (!element) return false;
  std::uint64_t elements = 1;
  for (const auto dim : tensor.dims)
    if (dim == 0 || __builtin_mul_overflow(elements, dim, &elements)) return false;
  std::uint64_t physical_bytes = 0;
  if (__builtin_mul_overflow(elements, static_cast<std::uint64_t>(*element), &physical_bytes) ||
      physical_bytes != *bytes)
    return false;
  if (tensor.scalar_storage && *physical != *tensor.scalar_storage) return false;
  if (tensor.part == format::PartRole::kTensorScale)
    return (*physical == format::Encoding::kF32 || tensor.scalar_storage) && elements == 1 &&
           tensor.dims.size() <= 1;
  auto expected = tensor.format.shape;
  if (tensor.part == format::PartRole::kData) {
    if (*physical != tensor.format.encoding || tensor.format.layout != format::Layout::kDense)
      return false;
  } else if (tensor.part == format::PartRole::kCodes) {
    const auto bits = tensor.format.encoding == format::Encoding::kAffine
                          ? tensor.format.params.bits
                          : format::Info(tensor.format.encoding).element_bits;
    if (bits == 0 || expected.ne[0] > std::numeric_limits<std::uint64_t>::max() / bits ||
        (expected.ne[0] * bits) % (*element * 8U) != 0)
      return false;
    expected.ne[0] = expected.ne[0] * bits / (*element * 8U);
  } else if (tensor.part == format::PartRole::kBlockScales ||
             tensor.part == format::PartRole::kBiases || tensor.part == format::PartRole::kZeros) {
    const auto group = tensor.format.encoding == format::Encoding::kAffine
                           ? tensor.format.params.group
                           : format::Info(tensor.format.encoding).block_elements;
    if (group == 0 || expected.ne[0] % group != 0) return false;
    expected.ne[0] /= group;
    if (tensor.format.encoding == format::Encoding::kAffine) {
      const auto scale = tensor.format.params.scale;
      const auto encoding = scale == format::AffineScale::kBF16  ? format::Encoding::kBF16
                            : scale == format::AffineScale::kF16 ? format::Encoding::kF16
                                                                 : format::Encoding::kF32;
      if (*physical != encoding) return false;
    }
  } else
    return false;
  if (tensor.dims.size() != expected.rank) return false;
  for (std::size_t i = 0; i < tensor.dims.size(); ++i)
    if (tensor.dims[i] != expected.ne[tensor.dims.size() - 1 - i]) return false;
  return true;
}

const spec::SemanticTensorRule* SemanticTensor(const spec::Mapping& m, std::string_view name) {
  const auto found = std::ranges::find_if(m.semantic_tensors, [&](const auto& rule) {
    return nd::Match(rule.tensor, name, nd::kLayerLimit).has_value();
  });
  return found == m.semantic_tensors.end() ? nullptr : &*found;
}
struct AssemblyInventory {
  std::vector<TensorAssembly> outputs{};
  std::vector<SourceDisposition> dispositions{};
};
std::expected<AssemblyInventory, Error> Assemble(const Checkpoint& source,
                                                 const spec::Mapping& table, std::uint32_t layers,
                                                 const TensorAssembly* selection = nullptr) {
  struct Group {
    TensorAssembly output{};
    std::map<format::PartRole, std::map<std::uint32_t, std::uint32_t>> members{};
    bool experts = false, shards = false;
  };
  std::map<std::string, Group, std::less<>> groups;
  AssemblyInventory inventory;
  inventory.dispositions.resize(source.tensors.size());
  for (std::uint32_t ti = 0; ti < source.tensors.size(); ++ti) {
    const auto& tensor = source.tensors[ti];
    if (selection && std::ranges::any_of(selection->parts, [&](const auto& part) {
          return std::ranges::contains(part.sources, ti);
        }))
      continue;
    if (SemanticTensor(table, tensor.name)) {
      inventory.dispositions[ti] = {SourceDispositionKind::kSemantic, 1};
      continue;
    }
    auto name = TensorName(table, source.container, tensor.name, layers, tensor.part);
    if (!name && IgnoredTensor(table, tensor.name, layers)) {
      inventory.dispositions[ti] = {SourceDispositionKind::kAuxiliary, 1};
      continue;
    }
    if (!name) return Bad("tensor has no mapping role", ItemKind::kTensor, ti);
    const auto canonical = WeightName(Expand(*name), tensor.part);
    auto [it, inserted] = groups.try_emplace(canonical);
    auto& group = it->second;
    if (inserted) {
      group.output = {.name = canonical, .format = tensor.format};
      group.experts = name->index.has_value() && name->is_expert;
      group.shards = name->index.has_value() && name->is_shard;
    } else {
      auto basis = group.output.format;
      // Shards may have unequal row counts (the last one may be shorter).
      if (group.shards || name->is_shard) basis.shape.ne[1] = tensor.format.shape.ne[1];
      if (basis != tensor.format)
        return Bad("paired tensor parts disagree on logical format", ItemKind::kTensor, ti);
      if (name->index && !group.experts && !group.shards) {
        if (std::ranges::any_of(group.members, [](const auto& part) {
              return part.first != format::PartRole::kTensorScale;
            }))
          return Bad("indexed tensor parts mix with an unindexed matrix", ItemKind::kTensor, ti);
        group.experts = name->is_expert;
        group.shards = name->is_shard;
      }
      if ((name->index && (group.experts != name->is_expert || group.shards != name->is_shard)) ||
          (!name->index && (group.experts || group.shards) &&
           tensor.part != format::PartRole::kTensorScale))
        return Bad("paired tensor parts disagree on index kind", ItemKind::kTensor, ti);
    }
    auto& members = group.members[tensor.part];
    if (!members.emplace(name->index.value_or(0), ti).second)
      return Bad("two source tensors bind one canonical part member", ItemKind::kTensor, ti);
  }
  for (auto& [name, group] : groups) {
    auto& out = group.output;
    const auto source_parts = format::PartsOf(out.format);
    if (!source_parts) return Bad("invalid source logical format");
    std::size_t count = 0;
    if (group.experts || group.shards) {
      const auto codes = group.members.find(source_parts->parts().front().role);
      if (codes == group.members.end()) return Bad("gathered tensor has no data part");
      count = codes->second.size();
      if (out.format.shape.rank != 2 || count == 0) return Bad("only matrices may be gathered");
      if (group.experts) {
        out.format.shape.rank = 3;
        out.format.shape.ne[2] = count;
      } else {
        out.format.shape.ne[1] = 0;
        for (auto [index, ti] : codes->second) {
          (void)index;
          if (__builtin_add_overflow(out.format.shape.ne[1], source.tensors[ti].format.shape.ne[1],
                                     &out.format.shape.ne[1]))
            return Bad("gathered row count overflow", Rule::kBounds);
        }
      }
    }
    for (const auto& part : source_parts->parts()) {
      auto found = group.members.find(part.role);
      if (found == group.members.end()) return Bad("logical tensor is missing an encoded part");
      const bool shared = group.shards && part.role == format::PartRole::kTensorScale;
      if (found->second.size() != (shared ? 1U : count ? count : 1U))
        return Bad("gathered tensor part coverage differs");
      PartAssembly assembly{.part = part.role,
                            .kind = shared          ? PartAssemblyKind::kSharedTensorScale
                                    : group.shards  ? PartAssemblyKind::kRows
                                    : group.experts ? PartAssemblyKind::kMatrices
                                                    : PartAssemblyKind::kCopy};
      std::uint32_t ordinal = 0;
      for (auto [index, ti] : found->second) {
        if (index != ordinal++) return Bad("gathered tensor indices are not contiguous from zero");
        assembly.sources.push_back(ti);
      }
      out.parts.push_back(std::move(assembly));
    }
    if (group.members.size() != source_parts->size()) return Bad("unexpected logical tensor part");
    const spec::TensorRule* role = nullptr;
    for (const auto& r : table.tensors)
      if (nd::Match(r.pattern, out.name, layers)) {
        role = &r;
        break;
      }
    if (role) {
      bool reshape = false;
      if (role->role == "conv" && out.format.layout == format::Layout::kDense &&
          out.parts.size() == 1 && out.format.shape.rank == 3 &&
          (out.format.shape.ne[0] == 1 || out.format.shape.ne[1] == 1)) {
        out.format.shape = {
            .ne = {out.format.shape.ne[0] == 1 ? out.format.shape.ne[1] : out.format.shape.ne[0],
                   out.format.shape.ne[2], 1, 1},
            .rank = 2};
        reshape = true;
      } else if (role->role == "shared_gate" && out.format.shape.rank == 2 &&
                 out.format.shape.ne[1] == 1) {
        out.format.shape = {.ne = {out.format.shape.ne[0], 1, 1, 1}, .rank = 1};
        reshape = true;
      }
      if (reshape)
        for (auto& part : out.parts) part.kind = PartAssemblyKind::kDenseReshape;
    }
    inventory.outputs.push_back(std::move(out));
  }
  return inventory;
}
// The component's source convention is grouped by key head. Its runtime
// contract uses tiled value heads. Plan one lossless permutation for every
// encoded part of each role, then record the component fold as a whole.
std::expected<std::vector<std::uint32_t>, Error> TileGdnHeads(
    const spec::Stage& stage, std::vector<TensorAssembly>& outputs) {
  std::vector<std::uint32_t> changed;
  for (const auto& layer : stage.layers)
    for (const auto& ref : layer.components) {
      if (ref.kind != "attention.gdn") continue;
      const auto integer = [&](std::string_view name) -> std::uint64_t {
        const auto* p = ref.Find(name);
        const auto* n = p ? std::get_if<std::int64_t>(&p->value) : nullptr;
        return n && *n > 0 ? static_cast<std::uint64_t>(*n) : 0;
      };
      const auto keys = integer("k_heads"), values = integer("v_heads"), dim = integer("head_dim");
      if (keys == 0 || values == 0 || dim == 0) return Bad("GDN head geometry is absent");
      std::set<std::uint32_t> selected;
      for (const auto& w : ref.weights) {
        const auto at = std::ranges::find(outputs, w.tensor, &TensorAssembly::name);
        if (at == outputs.end()) continue;  // companion part: same output permutation
        const bool scalar = w.role == "a" || w.role == "dt_bias";
        const bool projected_scalar = w.role == "alpha" || w.role == "beta";
        const bool qkv = w.role == "qkv";
        const bool conv = w.role == "conv";
        const bool gate = w.role == "gate";
        const bool output = w.role == "output";
        if (!scalar && !projected_scalar && !qkv && !conv && !gate && !output) continue;
        const auto oi = static_cast<std::uint32_t>(at - outputs.begin());
        if (!selected.insert(oi).second) continue;
        const auto axis = scalar || output ? HeadAxis::kColumns : HeadAxis::kRows;
        auto permutation = HeadPermutationPlan::Make(at->format, axis, keys, values,
                                                     scalar || projected_scalar ? 1 : dim,
                                                     qkv || conv ? 2 * keys * dim : 0);
        if (!permutation) return std::unexpected(permutation.error());
        at->permutation = std::move(*permutation);
        changed.push_back(oi);
      }
    }
  return changed;
}

void CanonicalParts(const std::vector<TensorAssembly>& outputs, spec::Source& source,
                    std::vector<NormalizedPartBinding>& bindings) {
  source.tensors.clear();
  bindings.clear();
  for (std::uint32_t oi = 0; oi < outputs.size(); ++oi) {
    const auto& out = outputs[oi];
    const auto parts = format::PartsOf(out.format);
    if (!parts) continue;  // The indexed planner reports invalid output formats.
    for (const auto& part : parts->parts()) {
      bindings.push_back({oi, part.role});
      source.tensors.push_back({.name = PartName(out.name, part.role),
                                .format = out.format,
                                .part = part.role,
                                .experts = out.format.shape.rank == 3});
    }
  }
}

// An explicit bottom-up merge: at most N*ceil(log2(N)) comparisons,
// exactly two N-index arrays, with no hidden sort scratch allocation.
template <class Less>
void Sort(std::vector<std::uint32_t>& order, std::vector<std::uint32_t>& scratch, Less less) {
  for (std::size_t width = 1; width < order.size(); width *= 2) {
    for (std::size_t begin = 0; begin < order.size(); begin += 2 * width) {
      const auto middle = std::min(begin + width, order.size());
      const auto end = std::min(begin + 2 * width, order.size());
      auto left = begin, right = middle;
      for (auto dest = begin; dest < end; ++dest)
        scratch[dest] = right == end || (left < middle && !less(order[right], order[left]))
                            ? order[left++]
                            : order[right++];
    }
    order.swap(scratch);
  }
}

// Every listed object is charged separately by the source-derived ledger.
static_assert(sizeof(spec::SourceTensor) <= 256 && sizeof(spec::SourceValue) <= 256);
static_assert(sizeof(spec::ComponentRef) <= 256 && sizeof(components::Instance) <= 256);
static_assert(sizeof(spec::Param) <= 128 && sizeof(spec::WeightBinding) <= 160);
static_assert(sizeof(model::StateTensorContract) <= 256);
static_assert(sizeof(CopyRoute) <= 256 && sizeof(MetadataUse) <= 256);
static_assert(sizeof(std::string) <= 32 && sizeof(std::vector<std::uint32_t>) <= 32);
static_assert(sizeof(void*) == 8);
}  // namespace

namespace normalization_detail {
std::expected<NormalizationCost, Error> Bound(const Checkpoint& s, const spec::Mapping& t,
                                              NormalizeBoundOptions options) {
  if (!t.dialects.empty()) {
    auto selected = spec::ResolveSourceMapping(t, s.architecture, s.container);
    if (!selected) return Bad("missing, ambiguous or invalid source dialect", Rule::kUnsupported);
    return Bound(s, selected->mapping, options);
  }
  if ((!options.component.empty() || options.draft_head_rows) &&
      t.role != spec::StageRole::kDrafter)
    return Bad("draft options require a drafter mapping", Rule::kUnsupported);
  if (options.draft_head_rows > kMaxTensors)
    return Bad("draft head row count exceeds tensor bounds", Rule::kBounds);
  if (auto table = ValidateImportMapping(t, s.container); !table)
    return std::unexpected(table.error());
  const auto& m = t;
  if ((s.architecture != m.architecture &&
       !std::ranges::contains(m.source_architectures, s.architecture)) ||
      !s.role.empty() ||
      (!IndexedSource(s) &&
       s.quantization != (s.container == "exl3" ? std::string_view("exl3") : std::string_view{})) ||
      !s.components.empty())
    return Bad("source is outside the normalization domain", ItemKind::kNone, kNoItem,
               Rule::kUnsupported);
  if (s.files.empty() || s.files.size() > kMaxSourceFiles || s.metadata.size() > kMaxMetadata ||
      s.skipped_tensors.size() > kMaxTensors ||
      s.tensors.size() > kMaxTensors - s.skipped_tensors.size())
    return Bad("normalization input count exceeds its bound", ItemKind::kNone, kNoItem,
               Rule::kBounds);
  for (const auto& slot : m.slots) {
    const auto* def = components::Find(slot.kind, slot.version);
    if (!def || def->params.size() > 64 || def->roles.size() > 32)
      return Bad("component catalog exceeds normalization budget", ItemKind::kNone, kNoItem,
                 Rule::kUnsupported);
    if ((def->state != nullptr) != (def->max_state_tensors != 0))
      return Bad("component state contract has no catalog bound");
    for (const auto& p : def->params) {
      if (p.type == components::ParamType::kReals ||
          (p.type == components::ParamType::kInts && !t.extended_components))
        return Bad("component list defaults are outside normalization budget");
      if (p.name.size() > spec::kMaxText || p.choices.size() > 64)
        return Bad("component parameter names exceed normalization budget");
      for (const auto choice : p.choices)
        if (choice.size() > spec::kMaxText)
          return Bad("component choices exceed normalization budget");
    }
  }
  const auto raw_value_for = [&](std::string_view rule_key) -> const MetaValue* {
    if (const auto* v = s.FindValue(rule_key)) return v;
    for (const auto& entry : s.metadata)
      if (spec::IsCanonicalKey(m, rule_key, entry.key)) return &entry.value;
    return nullptr;
  };
  Arithmetic a;
  std::uint64_t metadata_bytes = 0, layers = 0, width = 0, selected = 0;
  std::array<std::string_view, kRuleLimit> selected_keys{};
  std::array<const MetaValue*, kRuleLimit> selected_values{};
  bool architecture = false;
  std::optional<std::int64_t> split_no, split_count, split_tensors;
  for (std::size_t i = 0; i < s.metadata.size(); ++i) {
    const auto& e = s.metadata[i];
    if (e.key.empty() || e.key.size() > kMaxKeyBytes || json::FirstInvalidUtf8(e.key) ||
        !NoControlCharacters(e.key) || (i != 0 && s.metadata[i - 1].key >= e.key))
      return Bad("metadata keys are invalid, duplicated or unordered", ItemKind::kKey, i);
    const auto bytes = MetaBytes(e.value);
    if (!bytes || !a.ok ||
        (metadata_bytes = a.Add(metadata_bytes, a.Add(e.key.size() + 64, *bytes))) >
            kMaxGgufMetadataBytes)
      return Bad("metadata exceeds the reader accounting bound", ItemKind::kKey, i, Rule::kBounds);
    const auto key =
        Classify(t, s.container, e.key, std::holds_alternative<std::nullptr_t>(e.value), &s);
    if (!key)
      return Bad("unconsumed semantic metadata changes unsupported mapping semantics",
                 ItemKind::kKey, i);
    if (key->kind == MetadataClass::kSelected) {
      if (!SelectedValue(e.value, key->types))
        return Bad("selected metadata has the wrong type", ItemKind::kKey, i);
      if (selected == selected_keys.size())
        return Bad("too many selected metadata keys", ItemKind::kKey, i);
      bool duplicate = false;
      for (std::size_t j = 0; j < selected; ++j)
        if (selected_keys[j] == key->canonical) {
          if (!IndexedSource(s) || *selected_values[j] != e.value)
            return Bad("inconsistent metadata aliases", ItemKind::kKey, i);
          duplicate = true;
        }
      if (duplicate) continue;
      selected_keys[selected] = key->canonical;
      selected_values[selected++] = &e.value;
      const auto* integer = std::get_if<std::int64_t>(&e.value);
      if (spec::MatchesCanonicalKey(m, m.layer_count_key, key->canonical) && integer &&
          *integer > 0)
        layers = static_cast<std::uint64_t>(*integer);
      for (const auto& p : m.params)
        if (p.slot == spec::kStreamSlot && p.param == "width" &&
            spec::MatchesCanonicalKey(m, p.key, key->canonical) && integer && *integer > 0)
          width = static_cast<std::uint64_t>(*integer);
      for (const auto& r : t.requirements)
        if (spec::MatchesCanonicalKey(m, r.key, key->canonical) && integer &&
            r.equals_key.empty() && !r.positive &&
            ((*integer == r.reject_equal) != r.require_equal) &&
            (r.unless_present.empty() || !raw_value_for(r.unless_present)))
          return Bad("source scalar requires unsupported mapping semantics", ItemKind::kKey, i,
                     Rule::kUnsupported);
    } else if (key->kind == MetadataClass::kStructural) {
      const auto* integer = std::get_if<std::int64_t>(&e.value);
      if (e.key == "general.architecture" || e.key == "model_type") {
        const auto* name = std::get_if<std::string>(&e.value);
        if (!name || *name != s.architecture)
          return Bad("source architecture disagrees", ItemKind::kKey, i);
        architecture = true;
      } else if (!integer)
        return Bad("structural metadata requires an integer", ItemKind::kKey, i);
      else if (e.key == "general.alignment") {
        if (*integer <= 0 || *integer > kMaxGgufAlignment ||
            !std::has_single_bit(static_cast<std::uint64_t>(*integer)))
          return Bad("invalid GGUF alignment", ItemKind::kKey, i);
      } else if (e.key == "split.no")
        split_no = *integer;
      else if (e.key == "split.count")
        split_count = *integer;
      else
        split_tensors = *integer;
    }
  }
  // Borrow the already classified source values only during preflight. This
  // preserves the common table's source aliases without constructing a map.
  const auto value_for = [&](std::string_view key) -> const MetaValue* {
    for (std::size_t i = 0; i < selected; ++i)
      if (spec::MatchesCanonicalKey(m, key, selected_keys[i])) return selected_values[i];
    return nullptr;
  };
  if (const auto* interval_value = value_for(m.layer_type_interval_key)) {
    const auto* interval = std::get_if<std::int64_t>(interval_value);
    if (!interval || *interval <= 0) return Bad("layer interval is not positive");
    if (const auto* flags_value = value_for(m.layer_flags_key)) {
      const auto* flags = std::get_if<std::vector<std::int64_t>>(flags_value);
      if (!flags || flags->size() < layers) return Bad("layer flags do not cover their layers");
      for (std::uint64_t layer = 0; layer < layers; ++layer) {
        const char type = (layer + 1) % static_cast<std::uint64_t>(*interval) == 0
                              ? m.layer_types.back()
                              : m.layer_types.front();
        if (((*flags)[layer] > 0) != (type == m.layer_flag_type))
          return Bad("layer interval disagrees with layer flags");
      }
    }
    if (const auto* schedule = value_for(m.layer_type_key)) {
      const auto* codes = std::get_if<std::vector<std::int64_t>>(schedule);
      const auto* labels = std::get_if<std::vector<std::string>>(schedule);
      const auto* text = std::get_if<std::string>(schedule);
      const auto count = codes ? codes->size() : labels ? labels->size() : text ? text->size() : 0;
      if (count < layers) return Bad("explicit layer schedule does not cover its layers");
      for (std::uint64_t layer = 0; layer < layers; ++layer) {
        char actual = text ? (*text)[layer] : '\0';
        if (!text) {
          const auto found = std::ranges::find_if(m.layer_type_codes, [&](const auto& code) {
            return codes ? code.value == (*codes)[layer] : code.label == (*labels)[layer];
          });
          if (found != m.layer_type_codes.end()) actual = found->type;
        }
        const char expected = (layer + 1) % static_cast<std::uint64_t>(*interval) == 0
                                  ? m.layer_types.back()
                                  : m.layer_types.front();
        if (actual != expected) return Bad("layer interval disagrees with explicit layer schedule");
      }
    }
  }
  const auto tensor_for = [&](std::string_view canonical) -> const Tensor* {
    const auto find_with_prefix = [&](std::string_view suffix) -> const Tensor* {
      if (m.tensor_prefix.empty()) return s.FindTensor(suffix);
      for (const auto& tensor : s.tensors)
        if (tensor.name.size() == m.tensor_prefix.size() + suffix.size() &&
            tensor.name.starts_with(m.tensor_prefix) &&
            tensor.name.substr(m.tensor_prefix.size()) == suffix)
          return &tensor;
      return nullptr;
    };
    if (const auto* found = find_with_prefix(canonical)) return found;
    for (const auto& alias : t.tensor_aliases)
      if (alias.canonical == canonical)
        if (const auto* found = find_with_prefix(alias.source)) return found;
    return nullptr;
  };
  for (const auto& rule : t.tensor_presence) {
    const MetaValue* value = nullptr;
    for (std::size_t i = 0; i < selected; ++i)
      if (selected_keys[i] == rule.flag) {
        value = selected_values[i];
        break;
      }
    if (!value) continue;
    const auto* flag = std::get_if<bool>(value);
    if (!flag || (*flag == rule.when && !tensor_for(rule.tensor)))
      return Bad("source tying flag requires an explicit output tensor");
  }
  for (const auto& rule : t.requirements) {
    const auto* value = value_for(rule.key);
    if (!value && rule.optional) continue;
    if ((rule.require_equal || !rule.equals_key.empty() || rule.positive || rule.boolean) && !value)
      return Bad("required source scalar is absent");
    if (rule.boolean && value && !std::holds_alternative<bool>(*value))
      return Bad("required source flag has wrong type");
    if (!rule.flag_of_positive.empty()) {
      const auto* related = value_for(rule.flag_of_positive);
      const auto* count = related ? std::get_if<std::int64_t>(related) : nullptr;
      const auto* flag = value ? std::get_if<bool>(value) : nullptr;
      if (!flag || (related && !count) || *flag != (count && *count > 0))
        return Bad("source flag disagrees with its related structural parameter");
    }
    const auto* integer = value ? std::get_if<std::int64_t>(value) : nullptr;
    const auto* real = value ? std::get_if<double>(value) : nullptr;
    if (rule.positive && !(integer ? *integer > 0 : real && std::isfinite(*real) && *real > 0.0))
      return Bad("required source scalar must be positive");
    if (!rule.equals_key.empty()) {
      const auto* other = value_for(rule.equals_key);
      const auto* expected = other ? std::get_if<std::int64_t>(other) : nullptr;
      if (!integer || !expected || *integer != *expected)
        return Bad("source scalar disagrees with its related structural parameter");
    }
  }
  // Profile requirements are checked allocation-free, including required
  // presence, before allocating mapping/synthesis/catalog validation.
  for (const auto& rule : m.params) {
    if (!rule.slot.empty()) continue;
    const auto* value = value_for(rule.key);
    if (!value) {
      if (rule.optional) continue;
      return Bad("required source profile key is absent");
    }
    bool equal = false;
    if (const auto* i = std::get_if<std::int64_t>(value))
      equal = (rule.type == spec::ValueType::kInt && *i == rule.value.integer) ||
              (rule.type == spec::ValueType::kReal && static_cast<double>(*i) == rule.value.real);
    if (const auto* d = std::get_if<double>(value))
      equal = rule.type == spec::ValueType::kReal && *d == rule.value.real;
    if (const auto* flag = std::get_if<bool>(value))
      equal = rule.type == spec::ValueType::kBool && *flag == rule.value.flag;
    if (const auto* text = std::get_if<std::string>(value))
      equal = (rule.type == spec::ValueType::kString && *text == rule.value.text) ||
              (rule.type == spec::ValueType::kBool && !rule.flag_text.empty() &&
               *text == rule.flag_text);
    if (!equal) return Bad("source profile requirement disagrees");
  }
  if (!architecture || layers == 0 || layers > kLayerLimit || width == 0 ||
      width > components::kMaxStreamWidth)
    return Bad("source layer count or width is outside component bounds", ItemKind::kNone, kNoItem,
               Rule::kBounds);
  std::size_t list_elements = 0;
  for (const auto& rule : t.lists) {
    const auto* value = value_for(rule.key);
    const auto* list = value ? std::get_if<std::vector<std::int64_t>>(value) : nullptr;
    const auto* reals = value ? std::get_if<std::vector<double>>(value) : nullptr;
    const auto* labels = value ? std::get_if<std::vector<std::string>>(value) : nullptr;
    const auto count = list     ? list->size()
                       : reals  ? reals->size()
                       : labels ? labels->size()
                       : rule.per_layer && value &&
                               (std::holds_alternative<std::int64_t>(*value) ||
                                std::holds_alternative<double>(*value))
                           ? layers
                           : 0;
    if (labels && (!spec::MatchesCanonicalKey(m, m.layer_type_key, rule.key) ||
                   std::ranges::any_of(*labels, [&](const auto& label) {
                     return std::ranges::none_of(
                         m.layer_type_codes, [&](const auto& code) { return code.label == label; });
                   })))
      return Bad("unknown declared layer type label");
    if (!value && rule.optional) continue;
    if ((rule.min_size == 0 ? count != rule.size : (count < rule.min_size || count > rule.size)) ||
        (!rule.exact.empty() && (!list || !std::ranges::equal(*list, rule.exact.first(count)))))
      return Bad("source numeric-list profile requirement disagrees");
    if (rule.per_layer) {
      std::uint64_t extra = 0;
      bool stated = false;
      if (!rule.extra_layers_key.empty()) {
        if (const auto* v = raw_value_for(rule.extra_layers_key)) {
          const auto* n = std::get_if<std::int64_t>(v);
          if (!n || *n < 0 || static_cast<std::uint64_t>(*n) > nd::kLayerLimit - layers)
            return Bad("invalid declared extra layer count");
          extra = static_cast<std::uint64_t>(*n);
          stated = true;
        }
      }
      if (count != layers && count != layers + extra &&
          !(rule.allow_unstated_extra && !stated && count > layers))
        return Bad("source numeric list does not cover its declared layers");
    }
    if (list && (spec::MatchesCanonicalKey(m, m.layer_type_key, rule.key) ||
                 spec::MatchesCanonicalKey(m, m.layer_flags_key, rule.key))) {
      for (const auto code : *list) {
        if (spec::MatchesCanonicalKey(m, m.layer_type_key, rule.key)
                ? std::ranges::none_of(m.layer_type_codes,
                                       [&](const auto& c) { return c.value == code; })
                : (code < 0 || code > 1024))
          return Bad("source layer ratio is outside its components' domain");
      }
    }
    if (list && !rule.sections_of.empty()) {
      std::uint64_t pairs = 0;
      for (auto n : *list) {
        if (n < 0 || n > 1024) return Bad("invalid rotary section width");
        pairs += static_cast<std::uint64_t>(n);
      }
      const auto* dims = value_for(rule.sections_of);
      const auto* n = dims ? std::get_if<std::int64_t>(dims) : nullptr;
      double multiplier = 1;
      if (!rule.sections_multiplier_key.empty()) {
        const auto* v = value_for(rule.sections_multiplier_key);
        const auto* r = v ? std::get_if<double>(v) : nullptr;
        const auto* i = v ? std::get_if<std::int64_t>(v) : nullptr;
        if (!r && !i) return Bad("rotary section multiplier is missing");
        multiplier = r ? *r : static_cast<double>(*i);
      }
      if (!n || *n < 0 || !std::isfinite(multiplier) || multiplier <= 0 ||
          static_cast<double>(pairs * 2) != static_cast<double>(*n) * multiplier)
        return Bad("rotary sections do not cover the rotary dimension");
    }
    list_elements = std::max(list_elements, count);
  }
  for (const auto& rule : t.shapes) {
    const auto* found = tensor_for(rule.name);
    if (!found || !std::ranges::equal(found->dims, rule.shape))
      return Bad("source profile tensor shape disagrees");
  }
  // The mapped side table uses signed I32 row IDs. Preserve arbitrary source
  // hash constants, but refuse a range that cannot address its encoded table.
  for (const auto& rule : t.row_indices) {
    const auto* offsets_value = value_for(rule.offsets);
    const auto* counts_value = value_for(rule.counts);
    const auto* offsets =
        offsets_value ? std::get_if<std::vector<std::int64_t>>(offsets_value) : nullptr;
    const auto* counts =
        counts_value ? std::get_if<std::vector<std::int64_t>>(counts_value) : nullptr;
    const auto* tensor = tensor_for(rule.tensor);
    if (!offsets || !counts || offsets->size() != counts->size() || !tensor ||
        tensor->dims.size() != 2)
      return Bad("source row-index metadata is incomplete");
    const auto limit =
        std::min<std::uint64_t>(tensor->dims[1], std::numeric_limits<std::int32_t>::max());
    for (std::size_t i = 0; i < offsets->size(); ++i) {
      const auto offset = (*offsets)[i], count = (*counts)[i];
      if (offset < 0 || count <= 0 || static_cast<std::uint64_t>(offset) > limit ||
          static_cast<std::uint64_t>(count) > limit - static_cast<std::uint64_t>(offset))
        return Bad("source row-index range exceeds the encoded table");
    }
  }
  if (!architecture || layers == 0 || layers > kLayerLimit || width == 0 ||
      width > components::kMaxStreamWidth)
    return Bad("source layer count or width is outside component bounds", ItemKind::kNone, kNoItem,
               Rule::kBounds);
  for (std::size_t i = 0; i < s.metadata.size(); ++i) {
    const auto key = Classify(t, s.container, s.metadata[i].key,
                              std::holds_alternative<std::nullptr_t>(s.metadata[i].value), &s);
    if (key->kind != MetadataClass::kSelected) continue;
    const auto* list = std::get_if<std::vector<std::int64_t>>(&s.metadata[i].value);
    const auto* reals = std::get_if<std::vector<double>>(&s.metadata[i].value);
    if (!list && !reals) continue;
    const auto count = list ? list->size() : reals->size();
    const bool per_layer =
        spec::MatchesCanonicalKey(m, m.layer_flags_key, key->canonical) ||
        std::ranges::any_of(m.params, [&](const auto& p) {
          return spec::MatchesCanonicalKey(m, p.key, key->canonical) && p.per_layer;
        });
    if (per_layer && count != layers)
      return Bad("selected layer list does not cover every layer", ItemKind::kKey, i);
    if (t.extended_components && !per_layer && std::ranges::none_of(t.lists, [&](const auto& rule) {
          return spec::MatchesCanonicalKey(m, rule.key, key->canonical);
        }))
      return Bad("selected list has no compiled storage/domain bound", ItemKind::kKey, i);
    for (const auto& slot : m.slots) {
      const bool alias = spec::MatchesCanonicalKey(m, slot.layers_alias, key->canonical);
      if (!spec::MatchesCanonicalKey(m, slot.layers_key, key->canonical) && !alias) continue;
      if (!list) return Bad("layer selector requires integer values");
      const auto base = alias ? slot.alias_index_base : 0;
      for (std::size_t j = 0; j < list->size(); ++j)
        if ((*list)[j] < base || static_cast<std::uint64_t>((*list)[j] - base) >= layers ||
            (j && (*list)[j - 1] >= (*list)[j]))
          return Bad("layer selector is not ordered unique in-range layers", ItemKind::kKey, i);
    }
  }
  if (s.container == "gguf" &&
      ((split_no && *split_no != 0) ||
       (split_count &&
        (*split_count <= 0 || static_cast<std::uint64_t>(*split_count) != s.files.size())) ||
       (s.files.size() > 1 && (!split_count || !split_no)) ||
       (split_tensors && (*split_tensors < 0 || static_cast<std::uint64_t>(*split_tensors) !=
                                                    s.tensors.size() + s.skipped_tensors.size()))))
    return Bad("split structure disagrees with checkpoint files", ItemKind::kNone, kNoItem,
               Rule::kShards);
  std::uint64_t maximum = 0;
  for (const auto& rule : m.tensors) {
    if (rule.target) continue;
    maximum = a.Add(maximum, rule.pattern.contains(spec::kLayerField) ? layers : 1);
  }
  const auto ignored = std::ranges::count_if(s.tensors, [&](const Tensor& tensor) {
    return IgnoredTensor(t, tensor.name, static_cast<std::uint32_t>(layers));
  });
  if (s.tensors.empty() ||
      (!IndexedSource(s) && s.tensors.size() - static_cast<std::size_t>(ignored) > maximum))
    return Bad("tensor count exceeds the complete mapping", ItemKind::kNone, kNoItem,
               Rule::kBounds);
  for (std::size_t i = 0; i < s.files.size(); ++i) {
    const auto& f = s.files[i];
    if (f.name.empty() || f.name.size() > kMaxKeyBytes || !NoControlCharacters(f.name) ||
        json::FirstInvalidUtf8(f.name) || f.path.native().size() > 4096 ||
        (f.path.empty() && (s.files.size() != 1 || f.device != 0 || f.inode != 0)) ||
        (!f.path.empty() && (!f.path.is_absolute() || f.inode == 0)))
      return Bad("invalid source file identity", ItemKind::kShard, i);
  }
  // A skipped empty role is absent from synthesis, which owns optional,
  // fallback and conditional requiredness. Readers already check all live and
  // skipped names for duplicates across shards.
  std::uint64_t payload = 0;
  for (std::size_t i = 0; i < s.tensors.size(); ++i) {
    const auto& tensor = s.tensors[i];
    if (tensor.name.empty() || tensor.name.size() > kMaxNameBytes ||
        !NoControlCharacters(tensor.name) || json::FirstInvalidUtf8(tensor.name))
      return Bad("invalid source tensor name", ItemKind::kTensor, i);
    const auto name =
        TensorName(t, s.container, tensor.name, static_cast<std::uint32_t>(layers), tensor.part);
    if ((!name && IgnoredTensor(t, tensor.name, static_cast<std::uint32_t>(layers))) ||
        SemanticTensor(m, tensor.name)) {
      const auto& extent = tensor.extent;
      const auto parts = format::PartsOf(tensor.format);
      const auto part = parts ? parts->Find(tensor.part) : std::nullopt;
      if (!part || extent.file >= s.files.size() ||
          (!SourcePartBytes(tensor) || extent.bytes != *SourcePartBytes(tensor)) ||
          extent.offset > s.files[extent.file].bytes ||
          extent.bytes > s.files[extent.file].bytes - extent.offset)
        return Bad("ignored tensor has invalid source extent", ItemKind::kTensor, i);
      continue;
    }
    if (tensor.name.empty() || tensor.name.size() > kMaxNameBytes || !name)
      return Bad("tensor has no mapping role or declared ignorable pattern", ItemKind::kTensor, i);
    const auto parts = format::PartsOf(tensor.format);
    const auto part = parts ? parts->Find(tensor.part) : std::nullopt;
    if (!part || tensor.part != name->part ||
        (tensor.experts != name->experts && !IndexedSource(s)) || tensor.format.stacked != 1 ||
        tensor.dims.size() > format::kMaxDims)
      return Bad("tensor is not a complete source part", ItemKind::kTensor, i);
    if (IndexedSource(s)) {
      if (!SourceDescriptor(tensor))
        return Bad("source physical descriptor disagrees with logical format", ItemKind::kTensor,
                   i);
    } else if (s.container == "exl3") {
      if (tensor.part == format::PartRole::kData) {
        if (parts->size() != 1 || tensor.format.layout != format::Layout::kDense ||
            tensor.format.params != format::EncodingParams{} ||
            tensor.dims.size() != tensor.format.shape.rank)
          return Bad("EXL3 dense tensor disagrees with its role format", ItemKind::kTensor, i);
        for (std::size_t d = 0; d < tensor.dims.size(); ++d)
          if (tensor.dims[d] != tensor.format.shape.ne[tensor.dims.size() - 1 - d])
            return Bad("EXL3 dense tensor shape disagrees with its format", ItemKind::kTensor, i);
      } else {
        const auto bits = tensor.format.params.k_bits;
        const auto k = tensor.format.shape.ne[0], n = tensor.format.shape.ne[1];
        if (tensor.format.encoding != format::Encoding::kExl3Trellis ||
            tensor.format.layout != format::Layout::kExl3Tiles || tensor.format.shape.rank != 2 ||
            tensor.format.params.tensor_scale)
          return Bad("EXL3 trellis tensor disagrees with its role format", ItemKind::kTensor, i);
        const bool dims_ok =
            tensor.part == format::PartRole::kCodes
                ? ((tensor.dtype == "I16" || tensor.dtype == "exl3_trellis") &&
                   tensor.dims == std::vector<std::uint64_t>{k / 16, n / 16, 16ULL * bits})
            : tensor.part == format::PartRole::kInputSigns
                ? ((tensor.dtype == "F16" || tensor.dtype == "exl3_trellis") &&
                   tensor.dims == std::vector<std::uint64_t>{k})
            : tensor.part == format::PartRole::kOutputSigns
                ? ((tensor.dtype == "F16" || tensor.dtype == "exl3_trellis") &&
                   tensor.dims == std::vector<std::uint64_t>{n})
                : (tensor.part == format::PartRole::kCodebookMarker &&
                   (tensor.dtype == "I32" || tensor.dtype == "exl3_trellis") &&
                   tensor.dims.empty());
        if (!dims_ok)
          return Bad("EXL3 trellis part dtype or dimensions disagree", ItemKind::kTensor, i);
      }
    } else {
      if (parts->size() != 1 || (*parts)[0].role != format::PartRole::kData ||
          tensor.dims.size() != tensor.format.shape.rank ||
          !format::Info(tensor.format.encoding).ggml ||
          tensor.dtype != format::Info(tensor.format.encoding).name ||
          !std::ranges::equal(tensor.dims,
                              std::span(tensor.format.shape.ne).first(tensor.dims.size())))
        return Bad("tensor is not a complete GGUF data part", ItemKind::kTensor, i);
      const auto& encoding = format::Info(tensor.format.encoding);
      const auto layout = !encoding.quantized ? format::Layout::kDense
                          : encoding.id == format::Encoding::kNvfp4
                              ? format::Layout::kGgmlNvfp4Blocks
                              : format::Layout::kGgmlBlocks;
      if (tensor.format.layout != layout)
        return Bad("tensor layout is not the GGUF encoding layout", ItemKind::kTensor, i);
    }
    const auto& extent = tensor.extent;
    if (extent.file >= s.files.size() ||
        (!SourcePartBytes(tensor) || extent.bytes != *SourcePartBytes(tensor)) ||
        extent.offset > s.files[extent.file].bytes ||
        extent.bytes > s.files[extent.file].bytes - extent.offset)
      return Bad("tensor extent disagrees with its format or file", ItemKind::kTensor, i);
    payload = a.Add(payload, extent.bytes);
  }
  for (const auto& tensor : s.tensors)
    if (const auto* rule = SemanticTensor(m, tensor.name)) {
      const auto count = tensor.format.shape.rank == 1 ? tensor.format.shape.ne[0] : 0;
      if (tensor.format.encoding != format::Encoding::kI64 ||
          tensor.part != format::PartRole::kData || count == 0 || count > spec::kMaxList ||
          (rule->size != 0 && rule->size != count))
        return Bad("semantic tensor has unsupported dimensions or encoding");
      list_elements = std::max(list_elements, static_cast<std::size_t>(count));
    }
  // Requested storage ledger: canonical data, selected scalar/layer lists,
  // synthesis and validation. Count even mutually exclusive layer slots. Conservatively
  // count all synthesis and component-validation families simultaneously.
  const auto T = std::uint64_t{s.tensors.size()}, M = std::uint64_t{s.metadata.size()};
  const auto F = std::uint64_t{s.files.size()};
  std::uint64_t C = 0;
  for (const auto& slot : m.slots) {
    if (!slot.count_key.empty()) {
      const auto* v = value_for(slot.count_key);
      const auto* count = v ? std::get_if<std::int64_t>(v) : nullptr;
      if (!count || *count < 0 || static_cast<std::uint64_t>(*count) > layers)
        return Bad("source component count exceeds its layer domain");
    }
    C = a.Add(C, slot.site == spec::Site::kLayer || slot.per_layer ? layers : 1);
  }
  const auto V = [&](std::uint64_t n, std::uint64_t z) { return a.Add(a.Mul(a.Mul(3, n), z), 64); };
  const auto Str = [&](std::uint64_t n) { return a.Mul(n, 1024); };
  const auto Tree = [&](std::uint64_t n) { return a.Mul(n, 256); };
  std::uint64_t bytes = 0;
  const auto charge = [&](std::uint64_t n) { bytes = a.Add(bytes, n); };
  charge(a.Add(a.Mul(3, V(T, 256)), a.Mul(2, Str(T))));
  charge(a.Add(a.Mul(4, V(T, 8)), V(F, 256)));
  charge(a.Mul(2, V(M, 256)));
  charge(a.Add(V(kRuleLimit, 256), a.Mul(2, Str(kRuleLimit))));
  // Owned selected lists plus overlapping ParamValue copies in Synthesis::Param.
  // Per-layer extraction leaves scalars in components, not retained layer lists.
  charge(a.Mul(68, V(std::max<std::uint64_t>(layers, list_elements), 8)));
  // Params::Of's absent optional list default is empty (params.cc). Present
  // lists can coexist in source values, synthesis, validated Instances and
  // temporary parameter values; conservatively fund all 64 per component.
  if (t.extended_components) charge(a.Mul(a.Mul(a.Mul(4, C), 64), V(list_elements, 8)));
  charge(a.Add(a.Add(a.Mul(2, Tree(T)), Str(T)), Tree(kRuleLimit)));
  charge(a.Add(a.Mul(8, Tree(1024)), a.Mul(4, Str(1024))));
  charge(a.Add(a.Add(V(layers, 32), V(C, 256)), a.Add(layers * 64, Str(C))));
  charge(a.Mul(C, a.Add(V(64, 128), a.Mul(2, Str(64)))));
  charge(a.Mul(C, a.Add(V(32, 160), a.Mul(3, Str(32)))));
  charge(a.Add(a.Add(V(layers, 32), V(C, 256)), a.Add(layers * 64, a.Mul(3, V(C, 8)))));
  charge(a.Mul(C, a.Add(a.Add(V(64, 128), a.Mul(3, Str(64))), V(64, 32))));
  std::uint64_t contracts = 0;
  for (const auto& slot : m.slots) {
    const auto* def = components::Find(slot.kind, slot.version);
    const auto instances = slot.site == spec::Site::kLayer || slot.per_layer ? layers : 1;
    contracts = a.Add(contracts, a.Mul(instances, def->max_state_tensors));
  }
  charge(a.Add(a.Add(V(contracts, 256), a.Mul(2, Str(contracts))), V(4, 256)));
  charge(a.Add(a.Mul(6, V(32, 256)), a.Mul(4, Str(32))));
  charge(64 * 256 + 16 * 8192);
  charge(a.Mul(options.draft_head_rows, 4096));
  if (m.role == spec::StageRole::kDrafter) charge(spec::kMaxSpecBytes * 32ULL);
  const bool has_imported_folds = m.role != spec::StageRole::kDrafter &&
                                  std::ranges::any_of(m.folds, [&](const spec::FoldRule& f) {
                                    return f.imported && f.container == s.container;
                                  });
  if (has_imported_folds) charge(a.Add(65536, a.Mul(T, 2048)));
  // Three classification scans plus selected-key duplicate, width and constraint
  // scans total at most 2048 bounded key comparisons per metadata entry.
  std::uint64_t work = metadata_bytes;
  const auto effort = [&](std::uint64_t n) { work = a.Add(work, n); };
  effort(a.Mul(a.Mul(M, 2048), kMaxKeyBytes));
  effort(a.Mul(a.Mul(maximum, T), 256));
  effort(a.Mul(a.Mul(a.Mul(C, 64), 128), 256));
  effort(a.Mul(a.Mul(4 * T, static_cast<std::uint64_t>(std::bit_width(T))), 512));
  effort(a.Mul(a.Mul(F, F), 4096));
  effort(a.Mul(a.Mul(C, 64), a.Mul(layers, 8)));
  effort(64ULL * 64 * 1024 * 256 + 1024ULL * 1024 * 256);
  if (has_imported_folds) effort(a.Mul(T, 4096));
  if (m.role == spec::StageRole::kTarget) {
    auto sampling = SamplingDefaultsBound(s);
    if (!sampling) return std::unexpected(sampling.error());
    charge(sampling->metadata_bytes);
    effort(sampling->work_units);
  }
  if (!a.ok) return Bad("normalization bound overflow", ItemKind::kNone, kNoItem, Rule::kBounds);
  return NormalizationCost{bytes, work, payload, static_cast<std::uint32_t>(layers)};
}

std::expected<NormalizedPlan, Error> Build(Checkpoint&& source, const spec::Mapping& table,
                                           NormalizationLimits limits,
                                           const NormalizationTarget* target,
                                           NormalizeOptions options) {
  if (!table.dialects.empty()) {
    auto selected = spec::ResolveSourceMapping(table, source.architecture, source.container);
    if (!selected) return Bad("missing, ambiguous or invalid source dialect", Rule::kUnsupported);
    return Build(std::move(source), selected->mapping, limits, target, std::move(options));
  }
  if (table.role == spec::StageRole::kDrafter && !target)
    return Bad("drafter import requires --target TARGET_ID in the selected store",
               Rule::kUnsupported);
  const auto cost = Bound(
      source, table, {options.component, options.draft_head ? options.draft_head->ids.size() : 0});
  if (!cost) return std::unexpected(cost.error());
  if (cost->metadata_bytes > limits.metadata_bytes || cost->work_units > limits.work_units)
    return Bad("normalization reservation is insufficient", ItemKind::kNone, kNoItem,
               Rule::kBounds);
  if (target != nullptr && table.role != spec::StageRole::kDrafter)
    return Bad("--target applies only to a drafter import", ItemKind::kNone, kNoItem,
               Rule::kUnsupported);
  NormalizedPlan plan;
  plan.cost_ = *cost;
  plan.architecture_version_ = table.version;
  plan.source_ = std::move(source);
  spec::SamplingDefaults sampling;
  if (table.role == spec::StageRole::kTarget) {
    auto budget = SamplingDefaultsBound(plan.source_);
    if (!budget) return std::unexpected(budget.error());
    auto defaults = ReadSamplingDefaults(plan.source_, *budget, table.source_key_prefix);
    if (!defaults) return std::unexpected(defaults.error());
    sampling = std::move(*defaults);
  }
  const auto& input = plan.source_;
  plan.canonical_.architecture = std::string(table.architecture);
  plan.mapping_architecture_ = plan.canonical_.architecture;
  plan.canonical_.container = input.container;
  plan.canonical_.tensors.reserve(input.tensors.size());
  plan.routes_.reserve(input.tensors.size());
  std::optional<spec::SourceTensor> selected_ids;
  std::optional<TensorAssembly> selected_output;
  std::vector<Tensor> head_parts;
  if (options.draft_head) {
    if (!target || !table.draft || !IndexedSource(plan.source_))
      return Bad("selected head requires a native drafter and actual target", Rule::kUnsupported);
    const auto& head = *options.draft_head;
    auto vocab = spec::Vocabulary(target->model, target->source);
    if (!vocab || head.ids.empty() || head.file.parsed.bytes != head.file.bytes ||
        head.file.bytes == 0 || head.file.path.empty() || head.file.inode == 0 ||
        std::ranges::contains(plan.source_.files, head.file.name, &SourceFile::name))
      return Bad("selected head list lacks its complete authenticated source");
    std::int32_t previous = -1;
    for (auto id : head.ids) {
      if (id <= previous || id < 0 || static_cast<std::uint32_t>(id) >= *vocab)
        return Bad("selected head IDs must be ascending in the actual target vocabulary");
      previous = id;
    }
    const auto* canonical = FindImportMapping(input.container, target->source.architecture);
    if (!canonical) return Bad("selected head has no source dialect for the target architecture");
    auto resolved =
        spec::ResolveSourceMapping(*canonical, target->source.architecture, input.container);
    if (!resolved) return Bad("selected head has an invalid target source dialect");
    const auto* dialect = &resolved->mapping;
    std::set<std::uint32_t> head_indices;
    for (std::uint32_t ti = 0; ti < input.tensors.size(); ++ti) {
      const auto& tensor = input.tensors[ti];
      const auto name =
          TensorName(*dialect, input.container, tensor.name, nd::kLayerLimit, tensor.part);
      if (!name) continue;
      const auto rule =
          std::ranges::find(dialect->tensors, name->pattern, &spec::TensorRule::pattern);
      if (rule == dialect->tensors.end()) continue;
      const auto slot = std::ranges::find(dialect->slots, rule->slot, &spec::SlotRule::slot);
      if (slot != dialect->slots.end() && slot->site == spec::Site::kOutput &&
          rule->role == "weight") {
        head_parts.push_back(tensor);
        head_indices.insert(ti);
      }
    }
    if (head_parts.empty()) return Bad("selected head requires the checkpoint output weights");
    const auto data = std::ranges::find_if(head_parts, [](const auto& tensor) {
      return tensor.part == format::PartRole::kData || tensor.part == format::PartRole::kCodes;
    });
    if (data == head_parts.end() || data->format.shape.rank != 2 ||
        data->format.shape.ne[1] != *vocab)
      return Bad("selected head output geometry differs from target vocabulary");
    std::string output_name;
    for (const auto& rule : table.tensors) {
      if (rule.target || rule.role != "weight") continue;
      const auto slot = std::ranges::find(table.slots, rule.slot, &spec::SlotRule::slot);
      if (slot != table.slots.end() && slot->site == spec::Site::kOutput) {
        output_name = rule.pattern;
        break;
      }
    }
    if (output_name.empty()) return Bad("drafter mapping has no selected output role");
    selected_output = TensorAssembly{.name = output_name, .format = data->format};
    selected_output->format.shape.ne[1] = head.ids.size();
    const auto expected_parts = format::PartsOf(data->format);
    if (!expected_parts || expected_parts->size() != head_parts.size())
      return Bad("selected head has incomplete parts");
    std::vector<Tensor> kept;
    kept.reserve(plan.source_.tensors.size());
    for (std::uint32_t ti = 0; ti < input.tensors.size(); ++ti)
      if (!head_indices.contains(ti)) kept.push_back(std::move(plan.source_.tensors[ti]));
    plan.source_.tensors = std::move(kept);
    for (const auto& part : head_parts) {
      if (part.format != data->format) return Bad("selected head parts disagree on format");
      PartAssembly assembly{.part = part.part,
                            .kind = part.part == format::PartRole::kTensorScale
                                        ? PartAssemblyKind::kSharedTensorScale
                                        : PartAssemblyKind::kRows};
      if (part.part == format::PartRole::kTensorScale) {
        assembly.sources.push_back(static_cast<std::uint32_t>(plan.source_.tensors.size()));
        plan.source_.tensors.push_back(part);
      } else {
        const auto row = format::RowBytes(part.format, part.part);
        if (!row) return Bad("selected head format has no addressable rows");
        for (std::size_t at = 0; at < head.ids.size();) {
          std::size_t end = at + 1;
          while (end < head.ids.size() && head.ids[end] == head.ids[end - 1] + 1) ++end;
          const auto first = static_cast<std::uint64_t>(head.ids[at]), rows = end - at;
          Tensor run = part;
          run.name += std::format(".selected.{}", first);
          run.format.shape.ne[1] = rows;
          run.extent.offset += first * *row;
          run.extent.bytes = rows * *row;
          if (!run.dims.empty()) run.dims[0] = rows;
          assembly.sources.push_back(static_cast<std::uint32_t>(plan.source_.tensors.size()));
          plan.source_.tensors.push_back(std::move(run));
          at = end;
        }
      }
      selected_output->parts.push_back(std::move(assembly));
    }
    // Assemble the declared drafter roles below; synthetic head runs are
    // authorized only by this validated selection, never by a name prefix.
    const auto file = static_cast<std::uint32_t>(plan.source_.files.size());
    plan.source_.files.push_back(head.file);
    selected_ids =
        spec::SourceTensor{.name = std::string(table.draft->head_ids),
                           .format = {.encoding = format::Encoding::kI32,
                                      .shape = {.ne = {1, head.ids.size(), 1, 1}, .rank = 2}},
                           .part = format::PartRole::kData};
    GeneratedPart generated{.file = file};
    for (auto id : head.ids)
      for (unsigned b = 0; b < 4; ++b)
        generated.bytes.push_back(
            static_cast<std::byte>((static_cast<std::uint32_t>(id) >> (8 * b)) & 255));
    plan.generated_.push_back(std::move(generated));
    plan.selected_rows_ = head.ids.size();
    plan.has_options_ = true;
    plan.options_ = "{\"component\":";
    json::AppendQuoted(table.stage, plan.options_);
    plan.options_ += ",\"draft_head\":";
    json::AppendQuoted(head.file.name, plan.options_);
    plan.options_ += '}';
  }
  std::optional<AssemblyInventory> assembled;
  if (IndexedSource(input)) {
    auto inventory =
        Assemble(input, table, cost->layers, selected_output ? &*selected_output : nullptr);
    if (!inventory) return std::unexpected(inventory.error());
    assembled = std::move(*inventory);
    if (selected_output) assembled->outputs.push_back(std::move(*selected_output));
    auto relaid = RelayoutExpertTables(assembled->outputs, nullptr, &table);
    if (!relaid) return std::unexpected(relaid.error());
    for (auto& out : assembled->outputs) {
      const auto role = std::ranges::find_if(table.tensors, [&](const auto& r) {
        return r.role == "table" && nd::Match(r.pattern, out.name, cost->layers).has_value();
      });
      if (role == table.tensors.end()) continue;
      const auto slot = std::ranges::find(table.slots, role->slot, &spec::SlotRule::slot);
      if (slot != table.slots.end() &&
          (slot->site == spec::Site::kSide || slot->site == spec::Site::kEmbedding)) {
        if (auto row = RelayoutRowTable(out); !row) return std::unexpected(row.error());
      }
    }
    if (table.role == spec::StageRole::kDrafter) {
      // A component publication authenticates only its contributing shards and
      // kept documents. The disposition table, not a family/name test, selects
      // tensors; all source and output indices are remapped together.
      std::vector<std::uint32_t> tensor_ids(input.tensors.size(),
                                            std::numeric_limits<std::uint32_t>::max());
      std::vector<Tensor> kept;
      std::vector<SourceDisposition> dispositions;
      for (std::uint32_t ti = 0; ti < input.tensors.size(); ++ti) {
        if (assembled->dispositions[ti].kind == SourceDispositionKind::kAuxiliary) continue;
        tensor_ids[ti] = static_cast<std::uint32_t>(kept.size());
        kept.push_back(std::move(plan.source_.tensors[ti]));
        dispositions.push_back(assembled->dispositions[ti]);
      }
      for (auto& out : assembled->outputs)
        for (auto& part : out.parts)
          for (auto& ti : part.sources) ti = tensor_ids[ti];
      plan.source_.tensors = std::move(kept);
      assembled->dispositions = std::move(dispositions);
      std::vector<bool> used(input.files.size(), false);
      for (const auto& tensor : input.tensors) used[tensor.extent.file] = true;
      for (const auto& generated : plan.generated_) used[generated.file] = true;
      std::vector<std::uint32_t> file_ids(input.files.size());
      std::vector<SourceFile> files;
      for (std::uint32_t fi = 0; fi < input.files.size(); ++fi) {
        if (!used[fi] && input.files[fi].name.ends_with(".safetensors")) continue;
        file_ids[fi] = static_cast<std::uint32_t>(files.size());
        files.push_back(std::move(plan.source_.files[fi]));
      }
      for (auto& tensor : plan.source_.tensors) tensor.extent.file = file_ids[tensor.extent.file];
      for (auto& generated : plan.generated_) generated.file = file_ids[generated.file];
      plan.source_.files = std::move(files);
    }
    CanonicalParts(assembled->outputs, plan.canonical_, plan.bindings_);
  }
  const auto add_ids = [&]() {
    if (!selected_ids) return;
    plan.generated_.front().tensor = static_cast<std::uint32_t>(plan.canonical_.tensors.size());
    plan.canonical_.tensors.push_back(*selected_ids);
  };
  add_ids();
  for (std::size_t i = 0; !assembled && i < input.tensors.size(); ++i) {
    auto& tensor = plan.source_.tensors[i];
    tensor.dtype = format::Info(tensor.format.encoding).name;
    const auto name = TensorName(table, input.container, tensor.name, cost->layers);
    if (!name && IgnoredTensor(table, tensor.name, cost->layers)) continue;
    if (!name) return Bad("tensor mapping changed after preflight", ItemKind::kTensor, i);
    plan.canonical_.tensors.push_back({.name = Expand(*name),
                                       .format = tensor.format,
                                       .part = tensor.part,
                                       .experts = tensor.experts});
    const auto parts = format::PartsOf(tensor.format);
    const auto part = parts->Find(tensor.part);
    plan.routes_.push_back({static_cast<std::uint32_t>(i), tensor.extent, part->readable});
  }
  std::vector<std::uint32_t> names(plan.canonical_.tensors.size()),
      scratch(plan.canonical_.tensors.size());
  names.reserve(input.tensors.size());
  scratch.reserve(input.tensors.size());
  std::iota(names.begin(), names.end(), 0U);
  Sort(names, scratch, [&](auto x, auto y) {
    return plan.canonical_.tensors[x].name < plan.canonical_.tensors[y].name;
  });
  for (std::size_t i = 1; i < names.size(); ++i)
    if (plan.canonical_.tensors[names[i - 1]].name == plan.canonical_.tensors[names[i]].name)
      return Bad("two source tensors bind one canonical output", ItemKind::kTensor, names[i]);
  std::vector<std::uint32_t> physical(input.files.size());
  std::iota(physical.begin(), physical.end(), 0U);
  for (std::size_t i = 0; i < input.files.size(); ++i)
    for (std::size_t j = 0; j < i; ++j) {
      const auto& x = input.files[i];
      const auto& y = input.files[j];
      if ((!x.path.empty() && x.path == y.path) ||
          (x.inode != 0 && x.device == y.device && x.inode == y.inode)) {
        if (x.bytes != y.bytes || x.inode != y.inode || x.device != y.device)
          return Bad("source file aliases disagree", ItemKind::kShard, i);
        physical[i] = physical[j];
        break;
      }
    }
  names.resize(input.tensors.size());
  scratch.resize(input.tensors.size());
  std::iota(names.begin(), names.end(), 0U);
  Sort(names, scratch, [&](auto x, auto y) {
    const auto& a = input.tensors[x].extent;
    const auto& b = input.tensors[y].extent;
    return physical[a.file] != physical[b.file] ? physical[a.file] < physical[b.file]
                                                : a.offset < b.offset;
  });
  for (std::size_t i = 1; i < names.size(); ++i) {
    const auto& a = input.tensors[names[i - 1]].extent;
    const auto& b = input.tensors[names[i]].extent;
    if (physical[a.file] == physical[b.file] && b.offset < a.offset + a.bytes)
      return Bad("tensor payloads overlap in their physical source file", ItemKind::kTensor,
                 names[i]);
  }
  plan.order_.resize(plan.routes_.size());
  scratch.resize(plan.routes_.size());
  std::iota(plan.order_.begin(), plan.order_.end(), 0U);
  Sort(plan.order_, scratch, [&](auto x, auto y) {
    const auto& a = plan.routes_[x].extent;
    const auto& b = plan.routes_[y].extent;
    return physical[a.file] != physical[b.file] ? physical[a.file] < physical[b.file]
                                                : a.offset < b.offset;
  });
  plan.values_.reserve(kRuleLimit);
  plan.metadata_.reserve(input.metadata.size());
  plan.auxiliary_.reserve(table.auxiliary.size() + 1);
  for (const auto& aux : table.auxiliary)
    plan.auxiliary_.push_back({std::string(aux.prefix), std::string(aux.reason)});
  plan.auxiliary_.push_back({"", "unread metadata retained as source provenance"});
  for (std::size_t i = 0; i < input.metadata.size(); ++i) {
    const auto& e = input.metadata[i];
    const auto key = *Classify(table, input.container, e.key,
                               std::holds_alternative<std::nullptr_t>(e.value), &input);
    auto destination = key.auxiliary;
    if (key.kind == MetadataClass::kAuxiliary && key.auxiliary < table.auxiliary.size()) {
      // An auxiliary key a tensor's extent states must agree with it.
      const auto& aux = table.auxiliary[key.auxiliary];
      if (!aux.equals_tensor.empty()) {
        const auto* value = std::get_if<std::int64_t>(&e.value);
        const auto tensor = std::ranges::find(input.tensors, aux.equals_tensor, &Tensor::name);
        if (value == nullptr || tensor == input.tensors.end() || *value < 0 ||
            static_cast<std::uint64_t>(*value) != tensor->format.shape.ne[aux.equals_dim])
          return Bad("auxiliary metadata disagrees with its tensor's extent", ItemKind::kKey, i);
      }
    }
    if (key.kind == MetadataClass::kSelected) {
      destination = static_cast<std::uint32_t>(plan.values_.size());
      spec::ParamValue value{};
      std::visit(
          [&](const auto& v) {
            using T = std::decay_t<decltype(v)>;
            if constexpr (std::is_same_v<T, std::int64_t> || std::is_same_v<T, double> ||
                          std::is_same_v<T, bool> || std::is_same_v<T, std::string> ||
                          std::is_same_v<T, std::vector<std::int64_t>> ||
                          std::is_same_v<T, std::vector<double>>)
              value = v;
          },
          e.value);
      if (const auto* labels = std::get_if<std::vector<std::string>>(&e.value)) {
        std::vector<std::int64_t> codes;
        for (const auto& label : *labels) {
          const auto found =
              std::ranges::find(table.layer_type_codes, label, &spec::LayerTypeCode::label);
          if (found == table.layer_type_codes.end()) return Bad("unknown layer type label");
          codes.push_back(found->value);
        }
        value = std::move(codes);
      }
      const bool mapped_key = nd::KeyTypes(table, key.canonical, true) != 0;
      const auto canonical_key =
          mapped_key ? spec::CanonicalKey(table, key.canonical) : std::string(key.canonical);
      const auto previous = std::ranges::find(plan.values_, canonical_key, &spec::SourceValue::key);
      if (previous != plan.values_.end()) {
        if (previous->value != value) return Bad("inconsistent metadata aliases");
        destination = static_cast<std::uint32_t>(previous - plan.values_.begin());
      } else {
        plan.values_.push_back({canonical_key, std::move(value)});
      }
    }
    plan.metadata_.push_back({static_cast<std::uint32_t>(i), key.kind, destination});
  }
  for (std::uint32_t ti = 0; ti < input.tensors.size(); ++ti) {
    const auto& tensor = input.tensors[ti];
    const auto* rule = SemanticTensor(table, tensor.name);
    if (!rule) continue;
    const auto count = tensor.format.shape.rank == 1 ? tensor.format.shape.ne[0] : 0;
    if (count == 0 || count > spec::kMaxList || (rule->size && rule->size != count) ||
        tensor.format.encoding != format::Encoding::kI64 ||
        tensor.part != format::PartRole::kData || tensor.extent.bytes != count * 8)
      return Bad("semantic tensor is not its declared I64 list", ItemKind::kTensor, ti);
    const auto key = spec::CanonicalKey(table, rule->key);
    std::vector<std::byte> bytes(count * 8);
    if (auto read = input.Read(tensor.extent, 0, bytes); !read)
      return std::unexpected(read.error());
    std::vector<std::int64_t> values;
    values.reserve(static_cast<std::size_t>(count));
    for (std::size_t i = 0; i < count; ++i) {
      std::uint64_t value = 0;
      for (unsigned b = 0; b < 8; ++b)
        value |= std::to_integer<std::uint64_t>(bytes[i * 8 + b]) << (8 * b);
      values.push_back(std::bit_cast<std::int64_t>(value));
    }
    const auto previous = std::ranges::find(plan.values_, key, &spec::SourceValue::key);
    if (previous != plan.values_.end()) {
      const auto* list = std::get_if<std::vector<std::int64_t>>(&previous->value);
      if (!list || *list != values)
        return Bad("inconsistent duplicate semantic tensor or metadata value");
    } else
      plan.values_.push_back({key, std::move(values)});
    plan.semantic_.push_back({ti, tensor.extent, base::Sha256{}.Update(bytes).Finish()});
  }
  const bool drafter = table.role == spec::StageRole::kDrafter;
  if (drafter) {
    if (target == nullptr)
      return Bad("drafter import requires --target TARGET_ID in the selected store",
                 Rule::kUnsupported);
    if (auto valid = ValidateTarget(*target, table); !valid) return std::unexpected(valid.error());
  }
  const auto synthesize = [&]() -> std::expected<spec::ModelSpec, Error> {
    if (drafter && table.draft) {
      auto values = spec::ExtractValues(table, target->model, target->source, plan.canonical_);
      if (!values) return Bad("drafter target extraction failed");
      for (auto& value : *values) {
        auto at = std::ranges::find(plan.values_, value.key, &spec::SourceValue::key);
        if (at == plan.values_.end()) plan.values_.push_back(std::move(value));
      }
    }
    auto model = drafter ? spec::ComposeDrafter(target->model, *target->mapping, table,
                                                plan.canonical_, plan.values_)
                         : spec::Synthesize(table, plan.canonical_, plan.values_);
    if (!model) {
      return Bad("source does not satisfy the architecture mapping");
    }
    if (!drafter) model->stages.front().sampling = sampling;
    return std::move(*model);
  };
  const auto validate = [&](const spec::ModelSpec& model) -> std::expected<void, Error> {
    const std::array<std::span<const spec::SourceTensor>, 2> tables{
        drafter ? std::span(target->source.tensors) : std::span(plan.canonical_.tensors),
        plan.canonical_.tensors};
    auto valid =
        drafter ? components::ValidateSpec(model, tables, components::ValidationPurpose::kImport)
                : components::ValidateSpec(model, plan.canonical_.tensors,
                                           components::ValidationPurpose::kImport);
    if (!valid) {
      return Bad("normalized tensors do not satisfy their components");
    }
    return {};
  };
  auto model = synthesize();
  if (!model) return std::unexpected(model.error());
  if (!assembled)
    if (auto valid = validate(*model); !valid) return std::unexpected(valid.error());
  plan.spec_ = std::move(*model);
  const auto own_stage = [&]() -> const spec::Stage& {
    return *std::ranges::find(plan.spec_.stages, table.role, &spec::Stage::role);
  };
  const bool has_imported_folds = std::ranges::any_of(table.folds, [&](const spec::FoldRule& f) {
    return f.imported && f.container == plan.canonical_.container;
  });
  if (has_imported_folds || assembled) {
    std::vector<TensorAssembly> outputs;
    if (assembled) outputs = std::move(assembled->outputs);
    outputs.reserve(plan.canonical_.tensors.size());
    for (std::size_t ci = 0; !assembled && ci < plan.canonical_.tensors.size(); ++ci) {
      outputs.push_back({.name = plan.canonical_.tensors[ci].name,
                         .format = plan.canonical_.tensors[ci].format,
                         .parts = {{.part = plan.canonical_.tensors[ci].part,
                                    .kind = PartAssemblyKind::kCopy,
                                    .sources = {plan.routes_[ci].source_tensor}}}});
    }
    if (assembled && std::ranges::any_of(table.folds, [&](const auto& f) {
          return f.imported && f.container == input.container && f.fold == "gdn.value_heads_tiled";
        })) {
      auto tiled = TileGdnHeads(own_stage(), outputs);
      if (!tiled) return std::unexpected(tiled.error());
      if (!tiled->empty()) {
        plan.canonical_.folds.push_back("gdn.value_heads_tiled");
        plan.planned_folds_.push_back({"gdn.value_heads_tiled", std::move(*tiled)});
      }
    }
    auto applied = FoldDerivedParameters(own_stage(), outputs);
    if (!applied) return std::unexpected(applied.error());
    if (!applied->empty() || assembled) {
      for (auto& a : *applied) {
        if (!assembled)
          for (const std::uint32_t oi : a.outputs) {
            plan.canonical_.tensors[oi].format = outputs[oi].format;
            const auto parts = format::PartsOf(outputs[oi].format);
            plan.routes_[oi].readable = parts->Find(plan.canonical_.tensors[oi].part)->readable;
          }
        plan.canonical_.folds.push_back(a.fold);
        plan.planned_folds_.push_back({std::move(a.fold), std::move(a.outputs)});
      }
      if (assembled) {
        CanonicalParts(outputs, plan.canonical_, plan.bindings_);
        add_ids();
      }
      std::ranges::sort(plan.canonical_.folds);
      auto folded_spec = synthesize();
      if (!folded_spec) return std::unexpected(folded_spec.error());
      if (auto valid = validate(*folded_spec); !valid) return std::unexpected(valid.error());
      plan.spec_ = std::move(*folded_spec);
      std::vector<SourceDisposition> dispositions(
          plan.source_.tensors.size(), SourceDisposition{SourceDispositionKind::kAuxiliary, 1U});
      if (assembled) dispositions = assembled->dispositions;
      for (const auto& route : plan.routes_)
        dispositions[route.source_tensor] = {SourceDispositionKind::kRouted, 0U};
      auto indexed_cost = IndexedPartPlan::Bound(plan.source_, outputs);
      if (!indexed_cost) return std::unexpected(indexed_cost.error());
      auto parts =
          IndexedPartPlan::Make(std::move(plan.source_), outputs, dispositions, *indexed_cost);
      if (!parts) return std::unexpected(parts.error());
      plan.parts_.emplace(std::move(*parts));
      if (!assembled) plan.bindings_.reserve(outputs.size());
      for (std::uint32_t oi = 0; !assembled && oi < outputs.size(); ++oi)
        plan.bindings_.push_back({oi, plan.canonical_.tensors[oi].part});
    }
  }
  for (auto& g : spec::PolicyGates(table, plan.spec_, plan.canonical_.container)) {
    if (std::ranges::none_of(plan.values_, [&](const auto& v) { return v.key == g.key; }))
      plan.values_.push_back(std::move(g));
  }
  plan.folds_.reserve(table.folds.size());
  for (const auto& fold : table.folds)
    if (!fold.imported && fold.container == plan.canonical_.container)
      plan.folds_.push_back({std::string(fold.fold), std::string(fold.container)});
  return plan;
}

std::expected<NormalizedPlan, Error> Build(const Checkpoint& source, const spec::Mapping& table,
                                           NormalizationLimits limits,
                                           const NormalizationTarget* target,
                                           NormalizeOptions options) {
  if (table.role == spec::StageRole::kDrafter && !target)
    return Bad("drafter import requires --target TARGET_ID in the selected store",
               Rule::kUnsupported);
  const auto cost = Bound(
      source, table, {options.component, options.draft_head ? options.draft_head->ids.size() : 0});
  if (!cost) return std::unexpected(cost.error());
  if (cost->metadata_bytes > limits.metadata_bytes || cost->work_units > limits.work_units)
    return Bad("normalization reservation is insufficient", ItemKind::kNone, kNoItem,
               Rule::kBounds);
  if (target != nullptr && table.role != spec::StageRole::kDrafter)
    return Bad("--target applies only to a drafter import", ItemKind::kNone, kNoItem,
               Rule::kUnsupported);
  Checkpoint copy = source;
  return Build(std::move(copy), table, limits, target, std::move(options));
}

}  // namespace normalization_detail

std::expected<void, Error> ValidateTarget(const NormalizationTarget& target,
                                          const spec::Mapping& drafter) {
  if (!drafter.draft || target.mapping == nullptr ||
      target.mapping != spec::FindMapping(target.source.architecture, target.source.container) ||
      target.mapping->role != spec::StageRole::kTarget ||
      drafter.draft->target_architecture != target.mapping->architecture ||
      target.model.stages.size() != 1 ||
      target.model.stages.front().role != spec::StageRole::kTarget ||
      target.model.stages.front().artifact != 0)
    return Bad("drafter target architecture/container/spec is inconsistent", ItemKind::kNone,
               kNoItem, Rule::kUnsupported);
  if (auto valid = components::ValidateSpec(target.model, target.source.tensors,
                                            components::ValidationPurpose::kImport);
      !valid)
    return Bad("drafter target spec does not satisfy its components");
  return {};
}

std::expected<DraftHeadList, Error> ReadDraftHeadList(const std::filesystem::path& path,
                                                      std::uint32_t vocab) {
  if (vocab == 0 || vocab > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()))
    return Bad("invalid draft head vocabulary", Rule::kBounds);
  const auto name = path.filename().string();
  if (name.empty() || name.size() > 255 || name == "." || name == ".." ||
      !std::ranges::all_of(name, [](char c) {
        return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') ||
               c == '_' || c == '.' || c == '-';
      }))
    return Bad("the draft head list's file name is outside the artifact vocabulary",
               Rule::kUnsupported);
  auto opened = OpenCheckpointFile(path, 0);
  if (!opened) return std::unexpected(opened.error());
  auto text = ReadDocument(*opened, 12ULL * vocab, 0);
  if (!text) return std::unexpected(text.error());
  if (text->empty()) return Bad("the draft head list is empty");
  DraftHeadList out;
  std::string_view rest = *text;
  if (rest.ends_with('\n')) rest.remove_suffix(1);
  while (true) {
    const auto end = rest.find('\n');
    auto line = rest.substr(0, end);
    if (line.ends_with('\r')) line.remove_suffix(1);
    std::uint64_t id = 0;
    const auto parsed = std::from_chars(line.data(), line.data() + line.size(), id);
    if (line.empty() || line.size() > 10 || parsed.ec != std::errc{} ||
        parsed.ptr != line.data() + line.size() ||
        !std::ranges::all_of(line, [](char c) { return c >= '0' && c <= '9'; }))
      return Bad("the draft head list needs one decimal token ID a line");
    if (id >= vocab) return Bad("a draft head token ID is outside the vocabulary");
    if (!out.ids.empty() && static_cast<std::uint64_t>(out.ids.back()) >= id)
      return Bad("draft head token IDs must be strictly ascending");
    if (out.ids.size() == vocab) return Bad("the draft head list has more IDs than the vocabulary");
    out.ids.push_back(static_cast<std::int32_t>(id));
    if (end == std::string_view::npos) break;
    rest.remove_prefix(end + 1);
  }
  out.file = {.name = name,
              .path = opened->path,
              .bytes = opened->bytes,
              .device = opened->device,
              .inode = opened->inode,
              .parsed = {opened->bytes, base::Sha256().Update(*text).Finish()}};
  return out;
}

std::expected<NormalizationCost, Error> NormalizationBound(const Checkpoint& source,
                                                           NormalizeBoundOptions options) {
  const auto* mapping = FindImportMapping(source.container, source.architecture, options.component);
  if (!mapping) return Bad("no complete normalization table for this source", Rule::kUnsupported);
  return nd::Bound(source, *mapping, options);
}

std::expected<NormalizedPlan, Error> Normalize(Checkpoint&& source, NormalizationLimits limits,
                                               const NormalizationTarget* target,
                                               NormalizeOptions options) {
  const auto* mapping = FindImportMapping(source.container, source.architecture, options.component);
  if (!mapping) return Bad("no complete normalization table for this source", Rule::kUnsupported);
  return nd::Build(std::move(source), *mapping, limits, target, std::move(options));
}

std::expected<NormalizedPlan, Error> Normalize(const Checkpoint& source, NormalizationLimits limits,
                                               const NormalizationTarget* target,
                                               NormalizeOptions options) {
  const auto* mapping = FindImportMapping(source.container, source.architecture, options.component);
  if (!mapping) return Bad("no complete normalization table for this source", Rule::kUnsupported);
  return nd::Build(source, *mapping, limits, target, std::move(options));
}

std::expected<SourceCopyCursor, Error> NormalizedPlan::CursorFor(std::size_t output) const {
  if (output >= routes_.size()) return Bad("copy output index is outside the plan");
  return SourceCopyCursor(routes_[output].extent);
}
std::expected<std::optional<SourceCopy>, Error> SourceCopyCursor::Next(std::uint64_t maximum) {
  if (maximum == 0) return Bad("copy chunk maximum is zero");
  if (next_ == extent_.bytes) return std::optional<SourceCopy>{};
  const auto count =
      std::min({maximum, extent_.bytes - next_,
                static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())});
  const SourceCopy copy{extent_.file, extent_.offset + next_, next_, count};
  next_ += count;
  return std::optional(copy);
}
}  // namespace llmp::import
