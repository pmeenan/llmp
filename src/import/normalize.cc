// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#include "import/normalize.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <limits>
#include <numeric>
#include <type_traits>
#include <utility>

#include "base/json.h"
#include "components/component.h"
#include "components/step.h"
#include "import/readers/gguf.h"

namespace llmp::import {
namespace nd = normalization_detail;
namespace json = base::json;
namespace {
using Failure = std::unexpected<Error>;
Failure Bad(std::string_view reason, ItemKind kind = ItemKind::kNone, std::uint64_t item = kNoItem,
            Rule rule = Rule::kContainer) {
  return Failure(Error{.rule = rule, .reason = reason, .kind = kind, .item = item});
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
std::optional<Key> Classify(const SourceMapping& t, std::string_view key) {
  if ((t.container == "gguf" &&
       (key == "general.architecture" || key == "general.alignment" || key == "split.no" ||
        key == "split.count" || key == "split.tensors.count")) ||
      (t.container == "exl3" && key == "model_type"))
    return Key{.kind = MetadataClass::kStructural, .canonical = key};
  auto canonical = key;
  for (const auto& alias : t.keys)
    if (key == alias.source) canonical = alias.canonical;
  for (const auto& requirement : t.requirements)
    if (canonical == requirement.key && requirement.require_equal)
      return Key{.kind = MetadataClass::kSelected, .canonical = canonical, .types = nd::kInteger};
  if (const auto types = nd::KeyTypes(*t.architecture, canonical); types != 0)
    return Key{.kind = MetadataClass::kSelected, .canonical = canonical, .types = types};
  for (std::size_t i = 0; i < t.auxiliary.size(); ++i)
    if (t.container == "gguf" ? key.starts_with(t.auxiliary[i].prefix)
                              : key == t.auxiliary[i].prefix)
      return Key{.auxiliary = static_cast<std::uint32_t>(i)};
  return std::nullopt;
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
};
std::optional<ResolvedName> TensorName(const SourceMapping& t, std::string_view name,
                                       std::uint32_t layers) {
  std::optional<ResolvedName> found;
  for (const auto& rule : t.architecture->tensors) {
    if (rule.target) continue;
    if (const auto match = nd::Match(rule.pattern, name, layers)) {
      if (found && found->pattern != rule.pattern) return std::nullopt;
      found = ResolvedName{rule.pattern, match->digits, rule.part, rule.experts};
    }
  }
  for (const auto& alias : t.tensors)
    if (const auto match = nd::Match(alias.source, name, layers)) {
      if (found) return std::nullopt;
      for (const auto& rule : t.architecture->tensors)
        if (!rule.target && rule.pattern == alias.canonical)
          found = ResolvedName{alias.canonical, match->digits, rule.part, rule.experts};
    }
  return found;
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

std::expected<NormalizationCost, Error> NormalizationBound(const Checkpoint& source) {
  const auto* mapping = FindSourceMapping(source.container, source.architecture);
  if (!mapping)
    return Bad("no complete normalization table for this source", ItemKind::kNone, kNoItem,
               Rule::kUnsupported);
  return nd::Bound(source, *mapping);
}

namespace normalization_detail {
std::expected<NormalizationCost, Error> Bound(const Checkpoint& s, const SourceMapping& t) {
  if (auto table = ValidateSourceMapping(t); !table) return std::unexpected(table.error());
  const auto& m = *t.architecture;
  if (s.container != t.container || s.architecture != m.architecture || !s.role.empty() ||
      s.quantization != (t.container == "exl3" ? std::string_view("exl3") : std::string_view{}) ||
      !s.components.empty() || !s.skipped.empty())
    return Bad("source is outside the normalization domain", ItemKind::kNone, kNoItem,
               Rule::kUnsupported);
  if (s.files.empty() || s.files.size() > kMaxGgufSplits || s.metadata.size() > kMaxMetadata ||
      s.tensors.size() > kMaxTensors)
    return Bad("normalization input count exceeds its bound", ItemKind::kNone, kNoItem,
               Rule::kBounds);
  for (const auto& slot : m.slots) {
    const auto* def = components::Find(slot.kind, slot.version);
    if (!def || def->params.size() > 64 || def->roles.size() > 32)
      return Bad("component catalog exceeds normalization budget", ItemKind::kNone, kNoItem,
                 Rule::kUnsupported);
    // The extended target's only state producers are these compiled catalog
    // contracts (MQA seven, QSA four, GDN two, PLE one). Refuse a newly stateful kind
    // before constructing Instances or allocating a StateContract.
    if (t.extended_components && def->state && slot.kind != "attention.qsa" &&
        slot.kind != "attention.gdn" && slot.kind != "residual.ple" && slot.kind != "attention.mqa")
      return Bad("component state producer is outside normalization budget");
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
  Arithmetic a;
  std::uint64_t metadata_bytes = 0, layers = 0, width = 0, selected = 0;
  std::array<std::string_view, 64> selected_keys{};
  std::array<const MetaValue*, 64> selected_values{};
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
    const auto key = Classify(t, e.key);
    if (!key) return Bad("unclassified source metadata", ItemKind::kKey, i);
    if (key->kind == MetadataClass::kSelected) {
      if (!SelectedValue(e.value, key->types))
        return Bad("selected metadata has the wrong type", ItemKind::kKey, i);
      if (selected == selected_keys.size())
        return Bad("too many selected metadata keys", ItemKind::kKey, i);
      for (std::size_t j = 0; j < selected; ++j)
        if (selected_keys[j] == key->canonical)
          return Bad("duplicate metadata aliases", ItemKind::kKey, i);
      selected_keys[selected] = key->canonical;
      selected_values[selected++] = &e.value;
      const auto* integer = std::get_if<std::int64_t>(&e.value);
      if (key->canonical == m.layer_count_key && integer && *integer > 0)
        layers = static_cast<std::uint64_t>(*integer);
      for (const auto& p : m.params)
        if (p.slot == spec::kStreamSlot && p.param == "width" && p.key == key->canonical &&
            integer && *integer > 0)
          width = static_cast<std::uint64_t>(*integer);
      for (const auto& r : t.requirements)
        if (r.key == key->canonical && integer && ((*integer == r.reject_equal) != r.require_equal))
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
    } else if (t.container == "exl3") {
      if (e.key == "architectures") {
        const auto* list = std::get_if<std::vector<std::string>>(&e.value);
        if (!list || list->size() != 1 || (*list)[0] != "Qwen2ForCausalLM")
          return Bad("EXL3 source architectures disagree", ItemKind::kKey, i);
      } else {
        const auto* text = std::get_if<std::string>(&e.value);
        if (e.key != "transformers_version" || !text || text->empty() ||
            !NoControlCharacters(*text))
          return Bad("EXL3 auxiliary metadata has the wrong type", ItemKind::kKey, i);
      }
    }
  }
  // Borrow the already classified source values only during preflight. This
  // preserves the common table's source aliases without constructing a map.
  const auto value_for = [&](std::string_view key) -> const MetaValue* {
    for (std::size_t i = 0; i < selected; ++i)
      if (selected_keys[i] == key) return selected_values[i];
    return nullptr;
  };
  const auto tensor_for = [&](std::string_view canonical) -> const Tensor* {
    if (const auto* found = s.FindTensor(canonical)) return found;
    for (const auto& alias : t.tensors)
      if (alias.canonical == canonical)
        if (const auto* found = s.FindTensor(alias.source)) return found;
    return nullptr;
  };
  for (const auto& rule : t.requirements)
    if (rule.require_equal && !value_for(rule.key)) return Bad("required source scalar is absent");
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
      equal = rule.type == spec::ValueType::kString && *text == rule.value.text;
    if (!equal) return Bad("source profile requirement disagrees");
  }
  std::size_t list_elements = 0;
  for (const auto& rule : t.lists) {
    const auto* value = value_for(rule.key);
    const auto* list = value ? std::get_if<std::vector<std::int64_t>>(value) : nullptr;
    const auto* reals = value ? std::get_if<std::vector<double>>(value) : nullptr;
    const auto count = list ? list->size() : reals ? reals->size() : 0;
    if (count != rule.size ||
        (!rule.exact.empty() && (!list || !std::ranges::equal(*list, rule.exact))))
      return Bad("source numeric-list profile requirement disagrees");
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
    const auto key = Classify(t, s.metadata[i].key);
    if (key->kind != MetadataClass::kSelected) continue;
    const auto* list = std::get_if<std::vector<std::int64_t>>(&s.metadata[i].value);
    const auto* reals = std::get_if<std::vector<double>>(&s.metadata[i].value);
    if (!list && !reals) continue;
    const auto count = list ? list->size() : reals->size();
    const bool per_layer =
        key->canonical == m.layer_flags_key || std::ranges::any_of(m.params, [&](const auto& p) {
          return p.key == key->canonical && p.per_layer;
        });
    if (per_layer && count != layers)
      return Bad("selected layer list does not cover every layer", ItemKind::kKey, i);
    if (t.extended_components &&
        std::ranges::none_of(t.lists, [&](const auto& rule) { return rule.key == key->canonical; }))
      return Bad("selected list has no compiled storage/domain bound", ItemKind::kKey, i);
    for (const auto& slot : m.slots) {
      if (slot.layers_key != key->canonical) continue;
      if (!list) return Bad("layer selector requires integer values");
      for (std::size_t j = 0; j < list->size(); ++j)
        if ((*list)[j] < 0 || static_cast<std::uint64_t>((*list)[j]) >= layers ||
            (j && (*list)[j - 1] >= (*list)[j]))
          return Bad("layer selector is not ordered unique in-range layers", ItemKind::kKey, i);
    }
  }
  if (t.container == "exl3") {
    if (s.files.size() > 2 || (s.files.size() == 2 && (s.files[0].name != "model.safetensors" ||
                                                       s.files[1].name != "config.json")))
      return Bad("EXL3 source files differ from the single-shard contract", ItemKind::kNone,
                 kNoItem, Rule::kShards);
  } else if ((split_no && *split_no != 0) ||
             (split_count &&
              (*split_count <= 0 || static_cast<std::uint64_t>(*split_count) != s.files.size())) ||
             (s.files.size() > 1 && (!split_count || !split_no)) ||
             (split_tensors && (*split_tensors < 0 ||
                                static_cast<std::uint64_t>(*split_tensors) != s.tensors.size())))
    return Bad("split structure disagrees with checkpoint files", ItemKind::kNone, kNoItem,
               Rule::kShards);
  std::uint64_t maximum = 0;
  for (const auto& rule : m.tensors) {
    if (rule.target) continue;
    maximum = a.Add(maximum, rule.pattern.contains(spec::kLayerField) ? layers : 1);
  }
  if (s.tensors.empty() || s.tensors.size() > maximum)
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
  std::uint64_t payload = 0;
  for (std::size_t i = 0; i < s.tensors.size(); ++i) {
    const auto& tensor = s.tensors[i];
    const auto name = TensorName(t, tensor.name, static_cast<std::uint32_t>(layers));
    if (tensor.name.empty() || tensor.name.size() > kMaxNameBytes || !name)
      return Bad("unclassified tensor or noncanonical layer name", ItemKind::kTensor, i);
    const auto parts = format::PartsOf(tensor.format);
    const auto part = parts ? parts->Find(tensor.part) : std::nullopt;
    if (!part || tensor.part != name->part || tensor.experts != name->experts ||
        tensor.format.stacked != 1 || tensor.dims.size() > format::kMaxDims)
      return Bad("tensor is not a complete source part", ItemKind::kTensor, i);
    if (!t.encodings.empty() && !std::ranges::contains(t.encodings, tensor.format.encoding))
      return Bad("tensor encoding is outside this source domain", ItemKind::kTensor, i);
    if (t.container == "exl3") {
      if (tensor.part == format::PartRole::kData) {
        const bool bias = tensor.name.ends_with(".bias");
        const auto expected_encoding = bias ? format::Encoding::kF16 : format::Encoding::kBF16;
        if (parts->size() != 1 || tensor.format.encoding != expected_encoding ||
            tensor.format.layout != format::Layout::kDense ||
            tensor.format.params != format::EncodingParams{} ||
            tensor.dims.size() != tensor.format.shape.rank ||
            (tensor.dtype != (bias ? "F16" : "BF16") && tensor.dtype != (bias ? "f16" : "bf16")))
          return Bad("EXL3 dense tensor disagrees with its role format", ItemKind::kTensor, i);
        for (std::size_t d = 0; d < tensor.dims.size(); ++d)
          if (tensor.dims[d] != tensor.format.shape.ne[tensor.dims.size() - 1 - d])
            return Bad("EXL3 dense tensor shape disagrees with its format", ItemKind::kTensor, i);
      } else {
        const auto bits = tensor.format.params.k_bits;
        const auto k = tensor.format.shape.ne[0], n = tensor.format.shape.ne[1];
        if (parts->size() != 4 || tensor.format.encoding != format::Encoding::kExl3Trellis ||
            tensor.format.layout != format::Layout::kExl3Tiles || tensor.format.shape.rank != 2 ||
            tensor.format.params.codebook != format::Exl3Codebook::kMcg ||
            tensor.format.params.tensor_scale || (bits != 4 && bits != 5 && bits != 6 && bits != 8))
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
    if ((t.container == "exl3" ? extent.file != 0 : extent.file >= s.files.size()) ||
        extent.bytes != part->bytes || extent.offset > s.files[extent.file].bytes ||
        extent.bytes > s.files[extent.file].bytes - extent.offset)
      return Bad("tensor extent disagrees with its format or file", ItemKind::kTensor, i);
    payload = a.Add(payload, extent.bytes);
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
  charge(a.Add(V(64, 256), a.Mul(2, Str(64))));
  // Owned selected lists plus overlapping ParamValue copies in Synthesis::Param.
  // Per-layer extraction leaves scalars in components, not retained layer lists.
  charge(a.Mul(68, V(std::max<std::uint64_t>(layers, list_elements), 8)));
  // Params::Of's absent optional list default is empty (params.cc). Present
  // lists can coexist in source values, synthesis, validated Instances and
  // temporary parameter values; conservatively fund all 64 per component.
  if (t.extended_components) charge(a.Mul(a.Mul(a.Mul(4, C), 64), V(list_elements, 8)));
  charge(a.Add(a.Add(a.Mul(2, Tree(T)), Str(T)), Tree(64)));
  charge(a.Add(a.Mul(8, Tree(1024)), a.Mul(4, Str(1024))));
  charge(a.Add(a.Add(V(layers, 32), V(C, 256)), a.Add(layers * 64, Str(C))));
  charge(a.Mul(C, a.Add(V(64, 128), a.Mul(2, Str(64)))));
  charge(a.Mul(C, a.Add(V(32, 160), a.Mul(3, Str(32)))));
  charge(a.Add(a.Add(V(layers, 32), V(C, 256)), a.Add(layers * 64, a.Mul(3, V(C, 8)))));
  charge(a.Mul(C, a.Add(a.Add(V(64, 128), a.Mul(3, Str(64))), V(64, 32))));
  const auto contracts = t.extended_components ? a.Mul(7, C) : a.Mul(2, layers);
  charge(a.Add(a.Add(V(contracts, 256), a.Mul(2, Str(contracts))), V(4, 256)));
  charge(a.Add(a.Mul(6, V(32, 256)), a.Mul(4, Str(32))));
  charge(64 * 256 + 16 * 8192);
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
  if (!a.ok) return Bad("normalization bound overflow", ItemKind::kNone, kNoItem, Rule::kBounds);
  return NormalizationCost{bytes, work, payload, static_cast<std::uint32_t>(layers)};
}

std::expected<NormalizedPlan, Error> Build(Checkpoint source, const SourceMapping& table,
                                           NormalizationLimits limits) {
  const auto cost = Bound(source, table);
  if (!cost) return std::unexpected(cost.error());
  if (cost->metadata_bytes > limits.metadata_bytes || cost->work_units > limits.work_units)
    return Bad("normalization reservation is insufficient", ItemKind::kNone, kNoItem,
               Rule::kBounds);
  if (auto mapping = spec::ValidateMapping(*table.architecture); !mapping)
    return Bad("invalid architecture mapping", ItemKind::kNone, kNoItem, Rule::kUnsupported);
  NormalizedPlan plan;
  plan.cost_ = *cost;
  plan.version_ = table.version;
  plan.architecture_version_ = table.architecture->version;
  plan.source_ = std::move(source);
  const auto& input = plan.source_;
  plan.canonical_.architecture = input.architecture;
  plan.canonical_.container = input.container;
  plan.canonical_.tensors.reserve(input.tensors.size());
  plan.routes_.reserve(input.tensors.size());
  for (std::size_t i = 0; i < input.tensors.size(); ++i) {
    auto& tensor = plan.source_.tensors[i];
    tensor.dtype = format::Info(tensor.format.encoding).name;
    const auto name = TensorName(table, tensor.name, cost->layers);
    if (!name) return Bad("tensor mapping changed after preflight", ItemKind::kTensor, i);
    plan.canonical_.tensors.push_back({.name = Expand(*name),
                                       .format = tensor.format,
                                       .part = tensor.part,
                                       .experts = tensor.experts});
    const auto parts = format::PartsOf(tensor.format);
    const auto part = parts->Find(tensor.part);
    plan.routes_.push_back({static_cast<std::uint32_t>(i), tensor.extent, part->readable});
  }
  std::vector<std::uint32_t> names(input.tensors.size()), scratch(input.tensors.size());
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
  plan.order_.resize(input.tensors.size());
  std::iota(plan.order_.begin(), plan.order_.end(), 0U);
  Sort(plan.order_, scratch, [&](auto x, auto y) {
    const auto& a = plan.routes_[x].extent;
    const auto& b = plan.routes_[y].extent;
    return physical[a.file] != physical[b.file] ? physical[a.file] < physical[b.file]
                                                : a.offset < b.offset;
  });
  for (std::size_t i = 1; i < plan.order_.size(); ++i) {
    const auto& a = plan.routes_[plan.order_[i - 1]].extent;
    const auto& b = plan.routes_[plan.order_[i]].extent;
    if (physical[a.file] == physical[b.file] && b.offset < a.offset + a.bytes)
      return Bad("tensor payloads overlap in their physical source file", ItemKind::kTensor,
                 plan.order_[i]);
  }
  plan.values_.reserve(64);
  plan.metadata_.reserve(input.metadata.size());
  plan.auxiliary_.reserve(table.auxiliary.size());
  for (const auto& aux : table.auxiliary)
    plan.auxiliary_.push_back({std::string(aux.prefix), std::string(aux.reason)});
  for (std::size_t i = 0; i < input.metadata.size(); ++i) {
    const auto& e = input.metadata[i];
    const auto key = *Classify(table, e.key);
    auto destination = key.auxiliary;
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
      plan.values_.push_back({std::string(key.canonical), std::move(value)});
    }
    plan.metadata_.push_back({static_cast<std::uint32_t>(i), key.kind, destination});
  }
  auto spec = spec::Synthesize(*table.architecture, plan.canonical_, plan.values_);
  if (!spec) return Bad("source does not satisfy the architecture mapping");
  if (auto valid = components::ValidateSpec(*spec, plan.canonical_.tensors); !valid)
    return Bad("normalized tensors do not satisfy their components");
  plan.spec_ = std::move(*spec);
  plan.folds_.reserve(table.architecture->folds.size());
  for (const auto& fold : table.architecture->folds)
    if (fold.container == input.container && fold.converter.empty())
      plan.folds_.push_back({std::string(fold.fold), std::string(fold.container)});
  return plan;
}
}  // namespace normalization_detail

std::expected<NormalizedPlan, Error> Normalize(Checkpoint source, NormalizationLimits limits) {
  const auto* mapping = FindSourceMapping(source.container, source.architecture);
  if (!mapping)
    return Bad("no complete normalization table for this source", ItemKind::kNone, kNoItem,
               Rule::kUnsupported);
  return nd::Build(std::move(source), *mapping, limits);
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
