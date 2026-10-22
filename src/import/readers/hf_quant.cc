// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "import/readers/hf_quant.h"

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "base/json.h"
#include "format/encoding.h"
#include "format/layout.h"
#include "format/tensor_format.h"
#include "import/checkpoint.h"
#include "import/error.h"

namespace llmp::import {
namespace {

namespace json = base::json;
using format::Encoding;
using format::PartRole;

constexpr std::size_t kMaxQuantizedLayers = std::size_t{1} << 18U;
constexpr std::size_t kMaxExclusions = 4096;
constexpr std::size_t kMaxPatternBytes = kMaxKeyBytes;
// The steps all exclusion matching of one checkpoint may take (a literal
// comparison and a module look-up charged by the bytes they read): a
// wildcard pattern against a name backtracks,
// quadratic in the name at worst, so many hostile patterns against many
// scaled weights could otherwise run for hours; real configurations take
// at most a few hundred million.
constexpr std::uint64_t kMatchBudget = std::uint64_t{1} << 31U;

std::unexpected<Error> Refuse(Rule rule, std::string_view reason) {
  return std::unexpected(Error{.rule = rule, .reason = reason});
}

std::unexpected<Error> RefuseTensor(Rule rule, std::string_view reason, const Tensor& t,
                                    std::size_t index) {
  return std::unexpected(Error{.rule = rule,
                               .reason = reason,
                               .file = t.extent.file,
                               .kind = ItemKind::kTensor,
                               .item = index});
}

std::optional<std::string_view> StringMember(json::Value object, std::string_view name) {
  const auto v = object.find(name);
  if (!v || !v->is_string()) {
    return std::nullopt;
  }
  return v->string();
}

// A ModelOpt algorithm's scheme and group size.
std::optional<std::pair<QuantScheme, std::int64_t>> SchemeOf(std::string_view algorithm) {
  if (algorithm == "NVFP4" || algorithm == "W4A16_NVFP4") {
    return std::pair(QuantScheme::kNvfp4, std::int64_t{16});
  }
  if (algorithm == "MXFP8") {
    return std::pair(QuantScheme::kMxfp8, std::int64_t{32});
  }
  return std::nullopt;
}

// The scheme of {quant_algo, group_size}: the group, when given, must be
// the algorithm's.
std::expected<QuantScheme, Error> LayerScheme(std::string_view algorithm,
                                              std::optional<json::Value> group) {
  const auto scheme = SchemeOf(algorithm);
  if (!scheme) {
    return Refuse(Rule::kUnsupported, "a ModelOpt algorithm the format layer cannot describe");
  }
  if (group && (!group->int64() || *group->int64() != scheme->second)) {
    return Refuse(Rule::kConfig, "a ModelOpt group size other than its algorithm's");
  }
  return scheme->first;
}

std::expected<std::vector<std::string>, Error> Patterns(json::Value object) {
  std::vector<std::string> out;
  for (const std::string_view key : {"ignore", "exclude_modules"}) {
    const auto list = object.find(key);
    if (!list || list->is_null()) {
      continue;
    }
    if (!list->is_array() || list->size() > kMaxExclusions) {
      return Refuse(Rule::kConfig, "a quantization exclusion list that is not a list of names");
    }
    for (std::size_t i = 0; i < list->size(); ++i) {
      const json::Value p = list->at(i);
      if (!p.is_string() || p.string().empty() || p.string().size() > kMaxPatternBytes ||
          p.string().contains('[')) {
        return Refuse(Rule::kConfig,
                      "a quantization exclusion that is not a name with '*' and '?' wildcards");
      }
      out.emplace_back(p.string());
    }
  }
  std::ranges::sort(out);
  const auto [first, last] = std::ranges::unique(out);
  out.erase(first, last);
  return out;
}

// ModelOpt's description: its algorithm (one, or per module) and
// exclusions, from an object holding quant_algo (config.json's
// quantization_config, or hf_quant_config.json's root or its
// `quantization` object).
std::expected<Quantization, Error> ModelOpt(json::Value object) {
  Quantization q{.method = "modelopt"};
  const auto algorithm = StringMember(object, "quant_algo");
  if (!algorithm) {
    return Refuse(Rule::kConfig, "a ModelOpt quantization without quant_algo");
  }
  q.algorithm = *algorithm;
  if (q.algorithm == "MIXED_PRECISION") {
    const auto layers = object.find("quantized_layers");
    if (!layers || !layers->is_object() || layers->size() > kMaxQuantizedLayers) {
      return Refuse(Rule::kConfig, "a MIXED_PRECISION quantization without quantized_layers");
    }
    for (std::size_t i = 0; i < layers->size(); ++i) {
      const json::Value layer = layers->member(i);
      const auto layer_algorithm =
          layer.is_object() ? StringMember(layer, "quant_algo") : std::nullopt;
      if (layers->key(i).empty() || !layer_algorithm) {
        return Refuse(Rule::kConfig, "a quantized layer without a name or quant_algo");
      }
      auto scheme = LayerScheme(*layer_algorithm, layer.find("group_size"));
      if (!scheme) {
        return std::unexpected(scheme.error());
      }
      q.layers.emplace(std::string(layers->key(i)), *scheme);
    }
  } else {
    auto scheme = LayerScheme(q.algorithm, object.find("group_size"));
    if (!scheme) {
      return std::unexpected(scheme.error());
    }
    q.scheme = *scheme;
  }
  auto exclude = Patterns(object);
  if (!exclude) {
    return std::unexpected(exclude.error());
  }
  if (q.scheme) {
    q.exclude = std::move(*exclude);
  }
  return q;
}

// One config's quantization: none for compressed-tensors' `dense`.
std::expected<std::optional<Quantization>, Error> Normalize(json::Value object) {
  if (!object.is_object()) {
    return Refuse(Rule::kConfig, "a quantization config that is not an object");
  }
  const auto method = object.find("quant_method");
  if (method) {
    if (!method->is_string()) {
      return Refuse(Rule::kConfig, "a quant_method that is not a string");
    }
    const std::string_view m = method->string();
    if (m == "modelopt") {
      auto q = ModelOpt(object);
      if (!q) {
        return std::unexpected(q.error());
      }
      return std::optional<Quantization>(std::move(*q));
    }
    if (m == "compressed-tensors") {
      if (StringMember(object, "format") == "dense") {
        return std::optional<Quantization>();
      }
      return Refuse(Rule::kUnsupported,
                    "a compressed-tensors format the reader cannot describe (its NVFP4 global "
                    "scale is the reciprocal of NVFP4's tensor scale, which needs a fold)");
    }
    if (m == "awq" || m == "gptq") {
      return Refuse(Rule::kUnsupported,
                    "an AWQ or GPTQ quantization: the affine encoding with zero points, but "
                    "packed across output columns (AWQ in its own nibble order), which no "
                    "layout describes yet");
    }
    if (m == "exl3") {
      return Refuse(Rule::kUnsupported, "an EXL3 quantization, which the EXL3 reader reads");
    }
    if (m == "fp8") {
      return Refuse(Rule::kUnsupported,
                    "a block-scaled FP8 quantization with F32 scales, which no encoding describes");
    }
    return Refuse(Rule::kUnsupported, "an unknown quantization method");
  }
  // hf_quant_config.json: ModelOpt's older form nests the description.
  if (const auto nested = object.find("quantization"); nested && nested->is_object()) {
    auto q = ModelOpt(*nested);
    if (!q) {
      return std::unexpected(q.error());
    }
    return std::optional<Quantization>(std::move(*q));
  }
  if (object.find("quant_algo")) {
    auto q = ModelOpt(object);
    if (!q) {
      return std::unexpected(q.error());
    }
    return std::optional<Quantization>(std::move(*q));
  }
  return Refuse(Rule::kUnsupported, "a quantization config of no known form");
}

// fnmatch's '*' and '?' over the whole name, each step charged to
// `budget` (none once it is spent).
std::optional<bool> Matches(std::string_view pattern, std::string_view name,
                            std::uint64_t& budget) {
  std::size_t p = 0;
  std::size_t n = 0;
  std::size_t star = std::string_view::npos;
  std::size_t resume = 0;
  while (n < name.size()) {
    if (budget == 0) {
      return std::nullopt;
    }
    --budget;
    if (p < pattern.size() && (pattern[p] == '?' || pattern[p] == name[n])) {
      ++p;
      ++n;
    } else if (p < pattern.size() && pattern[p] == '*') {
      star = p++;
      resume = n;
    } else if (star != std::string_view::npos) {
      p = star + 1;
      n = ++resume;
    } else {
      return false;
    }
  }
  while (p < pattern.size() && pattern[p] == '*') {
    ++p;
  }
  return p == pattern.size();
}

struct Covered {
  std::optional<QuantScheme> scheme{};  // none: excluded, or no module configured
};

// Charges `steps` to `budget`: false when it does not hold them.
bool Charge(std::uint64_t& budget, std::uint64_t steps) {
  if (steps > budget) {
    budget = 0;
    return false;
  }
  budget -= steps;
  return true;
}

// The configured scheme of module `m` (none when matching spent the budget).
std::optional<Covered> Configured(const Quantization& q, std::string_view m,
                                  std::uint64_t& budget) {
  if (q.scheme) {
    for (const std::string& p : q.exclude) {
      if (!p.contains('*') && !p.contains('?')) {
        // A comparison reads up to the shorter name and one byte more.
        if (!Charge(budget, std::min(p.size(), m.size()) + 1)) {
          return std::nullopt;
        }
        if (p == m) {
          return Covered{};
        }
        continue;
      }
      const auto matched = Matches(p, m, budget);
      if (!matched) {
        return std::nullopt;
      }
      if (*matched) {
        return Covered{};
      }
    }
    return Covered{.scheme = q.scheme};
  }
  // The longest configured module that is m or an ancestor of it: each
  // look-up builds the prefix and compares it down the map's depth.
  const std::uint64_t depth = static_cast<std::uint64_t>(std::bit_width(q.layers.size())) + 1;
  std::string_view prefix = m;
  while (true) {
    if (!Charge(budget, (prefix.size() + 1) * depth)) {
      return std::nullopt;
    }
    if (const auto found = q.layers.find(std::string(prefix)); found != q.layers.end()) {
      return Covered{.scheme = found->second};
    }
    const std::size_t dot = prefix.rfind('.');
    if (dot == std::string_view::npos) {
      return Covered{};
    }
    prefix = prefix.substr(0, dot);
  }
}

std::uint64_t Elements(const Tensor& t) {
  std::uint64_t n = 1;
  for (const std::uint64_t d : t.dims) {
    n *= d;  // the reader checked the product
  }
  return n;
}

bool Is(const Tensor& t, std::string_view dtype, std::size_t rank) {
  return t.dtype == dtype && t.dims.size() == rank;
}

// The scheme a scaled weight's dtypes and shapes match under ModelOpt's
// conventions.
std::optional<QuantScheme> Inferred(const Tensor& w, const Tensor& s) {
  // K = 2 · w.dims[1] values a row (a file's size bounds every dimension
  // below 2^63, so the doubling cannot wrap).
  if (Is(w, "U8", 2) && Is(s, "F8_E4M3", 2) && w.dims[0] == s.dims[0] &&
      (w.dims[1] * 2) % 16 == 0 && s.dims[1] == w.dims[1] * 2 / 16) {
    return QuantScheme::kNvfp4;
  }
  if (Is(w, "F8_E4M3", 2) && (Is(s, "U8", 2) || Is(s, "F8_E8M0", 2)) && w.dims[0] == s.dims[0] &&
      w.dims[1] % 32 == 0 && s.dims[1] == w.dims[1] / 32) {
    return QuantScheme::kMxfp8;
  }
  return std::nullopt;
}

struct Pair {
  std::size_t weight = 0;
  std::size_t scale = 0;
  std::optional<std::size_t> tensor_scale{};
};

// The pair's format under `scheme`, every part's bytes checked.
std::expected<format::TensorFormat, Error> Describe(QuantScheme scheme, const Pair& p,
                                                    const std::vector<Tensor>& tensors) {
  const Tensor& w = tensors[p.weight];
  const Tensor& s = tensors[p.scale];
  const auto refuse = [&](std::string_view reason) {
    return RefuseTensor(Rule::kQuantization, reason, w, p.weight);
  };
  if (w.dims.size() != 2 || s.dims.size() != 2) {
    return refuse("a scaled weight or its scales of other than two dimensions");
  }
  const std::uint64_t n = w.dims[0];
  format::TensorFormat f{.layout = format::Layout::kSplit};
  if (scheme == QuantScheme::kNvfp4) {
    if (w.dtype != "U8" || s.dtype != "F8_E4M3") {
      return refuse("NVFP4 codes that are not U8 or block scales that are not F8_E4M3");
    }
    f.encoding = Encoding::kNvfp4;
    f.shape = {.ne = {w.dims[1] * 2, n, 1, 1}, .rank = 2};
    if (p.tensor_scale) {
      const Tensor& s2 = tensors[*p.tensor_scale];
      if ((s2.dtype != "F32" && s2.dtype != "BF16" && s2.dtype != "F16") || Elements(s2) != 1 ||
          s2.dims.size() > 1) {
        return refuse("an NVFP4 tensor scale that is not one F32, BF16 or F16 value");
      }
      f.params.tensor_scale = true;
    }
  } else {
    if (w.dtype != "F8_E4M3" || (s.dtype != "U8" && s.dtype != "F8_E8M0")) {
      return refuse("MXFP8 codes that are not F8_E4M3 or block scales that are not E8M0 bytes");
    }
    if (p.tensor_scale) {
      return refuse("an MXFP8 weight with a tensor scale");
    }
    f.encoding = Encoding::kMxfp8;
    f.shape = {.ne = {w.dims[1], n, 1, 1}, .rank = 2};
  }
  const auto parts = format::PartsOf(f);
  if (!parts) {
    return refuse("a scaled weight whose rows are not whole blocks");
  }
  const auto bytes = [&](PartRole role) { return parts->Find(role)->bytes; };
  if (bytes(PartRole::kCodes) != w.extent.bytes ||
      bytes(PartRole::kBlockScales) != s.extent.bytes || s.dims[0] != n ||
      (p.tensor_scale && (tensors[*p.tensor_scale].dtype == "F32" ? 4U : 2U) !=
                             tensors[*p.tensor_scale].extent.bytes)) {
    return refuse("a scaled weight's parts are not the sizes its format gives them");
  }
  return f;
}

}  // namespace

std::expected<std::optional<Quantization>, Error> RecognizeQuantization(json::Value config,
                                                                        const json::Value* hf) {
  if (const auto mlx = config.find("quantization"); mlx && mlx->is_object()) {
    return Refuse(Rule::kUnsupported, "an MLX quantization, which the MLX reader reads");
  }
  const auto qc = config.find("quantization_config");
  const bool has_config = qc && !qc->is_null();
  std::optional<Quantization> from_config;
  if (has_config) {
    auto q = Normalize(*qc);
    if (!q) {
      return std::unexpected(q.error());
    }
    from_config = std::move(*q);
  }
  if (hf == nullptr) {
    return from_config;
  }
  auto from_hf = Normalize(*hf);
  if (!from_hf) {
    return std::unexpected(from_hf.error());
  }
  if (has_config && *from_hf != from_config) {
    return Refuse(Rule::kConfig,
                  "config.json and hf_quant_config.json describe different quantizations");
  }
  return from_hf;
}

std::expected<void, Error> ApplyQuantization(const std::optional<Quantization>& quantization,
                                             std::vector<Tensor>& tensors,
                                             std::span<const SkippedTensor> skipped) {
  if (!quantization) {
    return {};
  }
  const Quantization& q = *quantization;
  std::map<std::string_view, std::size_t> index;
  for (std::size_t i = 0; i < tensors.size(); ++i) {
    index.emplace(tensors[i].name, i);
  }
  const auto find = [&](const std::string& name) -> std::optional<std::size_t> {
    const auto found = index.find(name);
    return found == index.end() ? std::nullopt : std::optional(found->second);
  };
  constexpr std::string_view kScale = ".weight_scale";
  // A part recorded as empty (zero elements: skipped by the safetensors
  // reader) would read as absent, and change the pairing: an empty block
  // scale would leave its weight dense, an empty tensor scale would drop
  // it or take an ancestor's. Each is refused.
  for (const SkippedTensor& t : skipped) {
    const std::string_view name = t.name;
    const bool scale = name.ends_with(kScale) || name.ends_with(".weight_scale_2");
    const bool weight = name.ends_with(".weight") &&
                        find(std::string(name.substr(0, name.size() - 7)) + std::string(kScale));
    if (scale || weight) {
      return Refuse(Rule::kQuantization, "a quantized weight with an empty part");
    }
  }
  // An NVFP4 weight without a tensor scale of its own takes its nearest
  // ancestor module's, when that module has one and no block scales (a
  // table stored as row shards under one scale, as Qwen3.8's n-gram
  // table): the shared tensor is then the tensor-scale part of each.
  const auto inherited = [&](std::string_view module) -> std::optional<std::size_t> {
    for (std::size_t dot = module.rfind('.'); dot != std::string_view::npos && dot != 0;
         dot = module.rfind('.', dot - 1)) {
      const std::string ancestor(module.substr(0, dot));
      if (const auto s2 = find(ancestor + ".weight_scale_2")) {
        return find(ancestor + std::string(kScale)) ? std::nullopt : s2;
      }
    }
    return std::nullopt;
  };
  std::uint64_t budget = kMatchBudget;
  std::vector<bool> described(tensors.size(), false);
  for (std::size_t i = 0; i < tensors.size(); ++i) {
    const std::string_view name = tensors[i].name;
    if (!name.ends_with(kScale) || name.size() == kScale.size()) {
      continue;
    }
    const std::string module(name.substr(0, name.size() - kScale.size()));
    const auto weight = find(module + ".weight");
    if (!weight) {
      continue;  // not a weight's block scales: a tensor of its own, dense
    }
    const auto covered = Configured(q, module, budget);
    if (!covered) {
      return Refuse(Rule::kBounds, "quantization exclusions beyond the reader's matching budget");
    }
    // A module no scheme covers (excluded, or not configured) is described
    // by its dtypes and shapes, which say what its parts are.
    const auto scheme = covered->scheme ? covered->scheme : Inferred(tensors[*weight], tensors[i]);
    if (!scheme) {
      return RefuseTensor(Rule::kQuantization,
                          "a scaled weight the quantization config does not describe",
                          tensors[*weight], *weight);
    }
    Pair pair{.weight = *weight, .scale = i, .tensor_scale = find(module + ".weight_scale_2")};
    if (!pair.tensor_scale && *scheme == QuantScheme::kNvfp4) {
      pair.tensor_scale = inherited(module);
    }
    auto f = Describe(*scheme, pair, tensors);
    if (!f) {
      return std::unexpected(f.error());
    }
    const auto set = [&](std::size_t t, PartRole role) {
      tensors[t].format = *f;
      tensors[t].part = role;
      tensors[t].experts = false;
      described[t] = true;
    };
    set(*weight, PartRole::kCodes);
    set(i, PartRole::kBlockScales);
    if (pair.tensor_scale) {
      const std::size_t s2 = *pair.tensor_scale;
      if (described[s2] && tensors[s2].format != *f) {
        return RefuseTensor(Rule::kQuantization,
                            "a shared tensor scale whose weights' formats differ", tensors[s2], s2);
      }
      set(s2, PartRole::kTensorScale);
      if (tensors[s2].dtype == "BF16") tensors[s2].scalar_storage = Encoding::kBF16;
      if (tensors[s2].dtype == "F16") tensors[s2].scalar_storage = Encoding::kF16;
    }
  }
  return {};
}

}  // namespace llmp::import
