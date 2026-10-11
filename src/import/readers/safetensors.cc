// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "import/readers/safetensors.h"

#include <sys/types.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <format>
#include <functional>
#include <iterator>
#include <map>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <variant>
#include <vector>

#include "base/json.h"
#include "format/encoding.h"
#include "format/layout.h"
#include "format/tensor_format.h"
#include "import/checkpoint.h"
#include "import/error.h"
#include "import/files.h"
#include "import/readers/hf_quant.h"
#include "import/readers/mlx.h"

namespace llmp::import {
namespace {

namespace json = base::json;
using format::Encoding;

constexpr std::string_view kConfigName = "config.json";
constexpr std::string_view kHfQuantName = "hf_quant_config.json";
// The weights' base names: a transformers model's, a diffusers component's.
constexpr std::array<std::string_view, 2> kWeightBases = {"model", "diffusion_pytorch_model"};
constexpr std::string_view kShardSuffix = ".safetensors";
constexpr std::string_view kIndexSuffix = ".safetensors.index.json";
constexpr std::size_t kMaxShardName = 255;
constexpr std::size_t kMaxVariant = 64;
// Entries listed looking for a variant's weights.
constexpr std::size_t kMaxDirectoryEntries = 4 * kMaxShards;

struct Dtype {
  std::string_view name;
  Encoding encoding;
  std::uint64_t bytes;
};

constexpr std::array<Dtype, 16> kDtypes = {{
    {"BOOL", Encoding::kBool, 1},
    {"U8", Encoding::kU8, 1},
    {"I8", Encoding::kI8, 1},
    {"F8_E5M2", Encoding::kF8E5M2, 1},
    {"F8_E4M3", Encoding::kF8E4M3, 1},
    {"F8_E8M0", Encoding::kE8M0, 1},
    {"I16", Encoding::kI16, 2},
    {"U16", Encoding::kU16, 2},
    {"F16", Encoding::kF16, 2},
    {"BF16", Encoding::kBF16, 2},
    {"I32", Encoding::kI32, 4},
    {"U32", Encoding::kU32, 4},
    {"F32", Encoding::kF32, 4},
    {"I64", Encoding::kI64, 8},
    {"U64", Encoding::kU64, 8},
    {"F64", Encoding::kF64, 8},
}};

std::unexpected<Error> Fail(Rule rule, std::string_view reason, std::uint64_t file = kNoItem,
                            ItemKind kind = ItemKind::kNone, std::uint64_t item = kNoItem) {
  return std::unexpected(
      Error{.rule = rule, .reason = reason, .file = file, .kind = kind, .item = item});
}

const Dtype* FindDtype(std::string_view name) {
  const Dtype* const found = std::ranges::find(kDtypes, name, &Dtype::name);
  return found == kDtypes.end() ? nullptr : &*found;
}

bool PlainName(std::string_view name) {
  return !name.empty() && name.size() <= kMaxShardName && name != "." && name != ".." &&
         !name.contains('/') && !name.contains('\0');
}

bool TensorName(std::string_view name) {
  return !name.empty() && name.size() <= kMaxNameBytes && NoControlCharacters(name);
}

std::optional<std::uint64_t> NonNegative(json::Value v) {
  if (!v.is_number() || !v.is_integer()) {
    return std::nullopt;
  }
  const auto i = v.int64();
  if (!i || *i < 0) {
    return std::nullopt;
  }
  return static_cast<std::uint64_t>(*i);
}

// ------------------------------------------------------------- config.json

std::expected<void, Error> Insert(std::string key, MetaValue value,
                                  std::map<std::string, MetaValue, std::less<>>& out) {
  if (key.size() > kMaxKeyBytes) {
    return Fail(Rule::kBounds, "a config.json key longer than the reader's cap");
  }
  if (out.size() >= kMaxMetadata) {
    return Fail(Rule::kBounds, "more config.json values than the reader's cap");
  }
  if (!out.emplace(std::move(key), std::move(value)).second) {
    return Fail(Rule::kConfig, "config.json flattens two members to one key");
  }
  return {};
}

MetaValue ListOf(json::Value list) {
  const std::size_t n = list.size();
  bool integers = true;
  bool numbers = true;
  bool strings = true;
  for (std::size_t i = 0; i < n; ++i) {
    const json::Value v = list.at(i);
    integers = integers && v.is_number() && v.is_integer() && v.int64().has_value();
    numbers = numbers && v.is_number() && v.float64() && std::isfinite(*v.float64());
    strings = strings && v.is_string();
  }
  if (integers) {
    std::vector<std::int64_t> out;
    out.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
      out.push_back(list.at(i).int64().value_or(0));  // every one an integer (above)
    }
    return out;
  }
  if (numbers) {
    std::vector<double> out;
    out.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
      out.push_back(list.at(i).float64().value_or(0.0));  // every one finite (above)
    }
    return out;
  }
  if (strings) {
    std::vector<std::string> out;
    out.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
      out.emplace_back(list.at(i).string());
    }
    return out;
  }
  return {};
}

std::expected<void, Error> Flatten(json::Value v, const std::string& key,
                                   std::span<const std::string_view> skip,
                                   std::map<std::string, MetaValue, std::less<>>& out) {
  switch (v.kind()) {
    case json::Kind::kObject:
      for (std::size_t i = 0; i < v.size(); ++i) {
        if (key.empty() && std::ranges::contains(skip, v.key(i))) {
          continue;
        }
        std::string member =
            key.empty() ? std::string(v.key(i)) : key + "." + std::string(v.key(i));
        if (member.size() > kMaxKeyBytes) {
          return Fail(Rule::kBounds, "a config.json key longer than the reader's cap");
        }
        if (auto ok = Flatten(v.member(i), member, skip, out); !ok) {
          return ok;
        }
      }
      return {};
    case json::Kind::kArray:
      return Insert(key, ListOf(v), out);
    case json::Kind::kNumber: {
      if (v.is_integer()) {
        if (const auto i = v.int64()) {
          return Insert(key, *i, out);
        }
      }
      const auto d = v.float64();
      return Insert(key, d && std::isfinite(*d) ? MetaValue(*d) : MetaValue(), out);
    }
    case json::Kind::kString:
      return Insert(key, std::string(v.string()), out);
    case json::Kind::kTrue:
    case json::Kind::kFalse:
      return Insert(key, v.boolean(), out);
    case json::Kind::kNull:
      return Insert(key, MetaValue(), out);
  }
  return {};
}

json::Limits DocumentLimits(std::uint64_t bytes, std::size_t depth, std::size_t values) {
  return {.max_bytes = static_cast<std::size_t>(bytes),
          .max_depth = depth,
          .max_values = values,
          .max_string_bytes = static_cast<std::size_t>(bytes)};
}

// A config document (config.json, hf_quant_config.json) within its cap.
std::expected<json::Document, Error> ParseConfigDocument(std::string_view text,
                                                         std::string_view large,
                                                         std::string_view invalid) {
  if (text.size() > kMaxConfigBytes) {
    return Fail(Rule::kBounds, large);
  }
  auto doc =
      json::Parse(text, DocumentLimits(kMaxConfigBytes, kMaxConfigDepth, std::size_t{1} << 20U));
  if (!doc) {
    return Fail(Rule::kConfig, invalid, kNoItem, ItemKind::kByte, doc.error().offset);
  }
  return std::move(*doc);
}

bool ArchitectureName(json::Value v) {
  return v.is_string() && !v.string().empty() && v.string().size() <= kMaxArchitecture &&
         NoControlCharacters(v.string());
}

// The root members config.json's metadata leaves out: the quantization
// the readers describe as tensor formats.
constexpr std::array<std::string_view, 2> kQuantizationMembers = {"quantization_config",
                                                                  "quantization"};

// config.json and hf_quant_config.json parsed, and what config.json says
// of every checkpoint: its architecture and metadata.
struct Config {
  json::Document document{};
  std::optional<json::Document> hf{};
  std::string architecture{};
  std::vector<MetaEntry> metadata{};
};

std::expected<Config, Error> ParseConfig(std::string_view text,
                                         std::optional<std::string_view> hf_text) {
  auto doc = ParseConfigDocument(text, "a config.json larger than its cap",
                                 "config.json is not valid JSON within the reader's limits");
  if (!doc) {
    return std::unexpected(doc.error());
  }
  Config c{.document = std::move(*doc)};
  const json::Value root = c.document.root();
  if (!root.is_object()) {
    return Fail(Rule::kConfig, "config.json is not an object");
  }
  // A transformers model's model_type, or a diffusers component's
  // _class_name when it has none.
  const auto model_type = root.find("model_type");
  const auto name = model_type ? model_type : root.find("_class_name");
  if (!name || !ArchitectureName(*name)) {
    return Fail(Rule::kConfig,
                "config.json has no model_type (or _class_name) string of 1 to 200 bytes");
  }
  c.architecture = std::string(name->string());
  if (hf_text) {
    auto hf =
        ParseConfigDocument(*hf_text, "an hf_quant_config.json larger than its cap",
                            "hf_quant_config.json is not valid JSON within the reader's limits");
    if (!hf) {
      return std::unexpected(hf.error());
    }
    c.hf = std::move(*hf);
  }
  auto metadata = FlattenMetadata(root, kQuantizationMembers);
  if (!metadata) {
    return std::unexpected(metadata.error());
  }
  c.metadata = std::move(*metadata);
  return c;
}

// ------------------------------------------------------------- shards

// What a shard's `__metadata__.format` says wrote it: nothing, MLX
// ("mlx"), or another library ("pt", "tf", ...).
enum class ShardFormat : std::uint8_t { kNone, kMlx, kOther };

// The tensors of one shard's header (`header`, which follows its 8-byte
// length), whose data section starts at `data` in a file of `size` bytes,
// and the format its metadata names.
std::expected<std::vector<Tensor>, Error> ParseHeader(std::string_view header, std::uint64_t data,
                                                      std::uint64_t size, std::uint32_t file,
                                                      ShardFormat& format) {
  format = ShardFormat::kNone;
  // Depth 4: an entry's members may hold one more level, which is ignored.
  const auto doc = json::Parse(header, DocumentLimits(header.size(), 4, std::size_t{1} << 23U));
  if (!doc) {
    return Fail(Rule::kContainer, "a safetensors header that is not valid JSON within the limits",
                file, ItemKind::kByte, doc.error().offset + 8);
  }
  const json::Value root = doc->root();
  if (!root.is_object()) {
    return Fail(Rule::kContainer, "a safetensors header that is not an object", file);
  }
  if (root.size() > kMaxTensors + 1) {
    return Fail(Rule::kBounds, "more tensors than the reader's cap", file);
  }
  struct Entry {
    std::uint64_t begin = 0;
    std::uint64_t end = 0;
    std::size_t member = 0;
  };
  std::vector<Entry> entries;
  std::vector<Tensor> tensors;
  entries.reserve(root.size());
  tensors.reserve(root.size());
  for (std::size_t i = 0; i < root.size(); ++i) {
    const std::string_view name = root.key(i);
    const json::Value v = root.member(i);
    const auto refuse = [&](Rule rule, std::string_view reason) {
      return Fail(rule, reason, file, ItemKind::kTensor, i);
    };
    if (name == "__metadata__") {
      // Free text, but for the writer's `format`: MLX's arrays keep their
      // own axis order (readers/mlx.h).
      const auto f = v.is_object() ? v.find("format") : std::nullopt;
      if (f && f->is_string()) {
        format = f->string() == "mlx" ? ShardFormat::kMlx : ShardFormat::kOther;
      }
      continue;
    }
    if (!TensorName(name)) {
      return refuse(Rule::kContainer, "a tensor name empty, too long or with control characters");
    }
    const auto dtype_v = v.is_object() ? v.find("dtype") : std::nullopt;
    const auto shape_v = v.is_object() ? v.find("shape") : std::nullopt;
    const auto offsets_v = v.is_object() ? v.find("data_offsets") : std::nullopt;
    if (!dtype_v || !shape_v || !offsets_v || !dtype_v->is_string() || !shape_v->is_array() ||
        !offsets_v->is_array() || offsets_v->size() != 2) {
      return refuse(Rule::kContainer,
                    "a tensor entry without a dtype, a shape and two data offsets");
    }
    const Dtype* dtype = FindDtype(dtype_v->string());
    if (dtype == nullptr) {
      return refuse(Rule::kUnsupported, "a dtype the reader does not know");
    }
    if (shape_v->size() > kMaxRank) {
      return refuse(Rule::kBounds, "a tensor of more dimensions than the reader's cap");
    }
    Tensor t{.name = std::string(name), .dtype = dtype->name};
    std::uint64_t elements = 1;
    for (std::size_t d = 0; d < shape_v->size(); ++d) {
      const auto n = NonNegative(shape_v->at(d));
      if (!n) {
        return refuse(Rule::kContainer, "a dimension that is not a non-negative integer");
      }
      if (__builtin_mul_overflow(elements, *n, &elements)) {
        return refuse(Rule::kBounds, "a tensor beyond 64-bit sizes");
      }
      t.dims.push_back(*n);
    }
    if (elements == 0) {
      return refuse(Rule::kUnsupported, "a tensor with no elements");
    }
    std::uint64_t bytes = 0;
    if (__builtin_mul_overflow(elements, dtype->bytes, &bytes)) {
      return refuse(Rule::kBounds, "a tensor beyond 64-bit sizes");
    }
    const auto begin = NonNegative(offsets_v->at(0));
    const auto end = NonNegative(offsets_v->at(1));
    if (!begin || !end || *end < *begin || *end - *begin != bytes) {
      return refuse(Rule::kContainer, "data offsets that do not span the tensor's bytes");
    }
    // GGML's order, dimensions past the fourth folded into ne[3].
    format::TensorFormat& f = t.format;
    f.encoding = dtype->encoding;
    f.layout = format::Layout::kDense;
    const std::size_t rank = t.dims.size();
    f.shape.rank = static_cast<std::uint8_t>(std::min(rank, format::kMaxDims));
    for (std::size_t d = 0; d < rank; ++d) {
      std::uint64_t& ne = f.shape.ne.at(std::min(d, format::kMaxDims - 1));
      const std::uint64_t dim = t.dims[rank - 1 - d];
      ne = d < format::kMaxDims ? dim : ne * dim;  // the product was checked
    }
    const auto parts = format::PartsOf(f);
    if (!parts || (*parts)[0].bytes != bytes) {
      return refuse(Rule::kContainer, "a tensor the size rule does not size as its dtype");
    }
    t.experts = rank == 3 && HasComponent(t.name, "experts");
    entries.push_back({.begin = *begin, .end = *end, .member = tensors.size()});
    tensors.push_back(std::move(t));
  }
  std::ranges::stable_sort(entries, {}, &Entry::begin);
  // Every tensor inside the data section and none overlapping another
  // (safetensors' own reader also wants them to tile it with no gap and no
  // byte left over; gaps and trailing bytes leave every tensor described).
  const std::uint64_t section = size - data;
  std::uint64_t previous = 0;
  std::vector<Tensor> ordered;
  ordered.reserve(tensors.size());
  for (const Entry& e : entries) {
    if (e.end > section) {
      return Fail(Rule::kContainer, "a tensor's data past the end of its file", file,
                  ItemKind::kTensor, e.member);
    }
    if (e.begin < previous) {
      return Fail(Rule::kContainer, "two tensors' data overlap", file, ItemKind::kTensor, e.member);
    }
    previous = e.end;
    Tensor& t = tensors[e.member];
    t.extent = {.file = file, .offset = data + e.begin, .bytes = e.end - e.begin};
    ordered.push_back(std::move(t));
  }
  return ordered;
}

std::uint64_t HeaderLength(std::span<const std::byte, 8> length) {
  std::uint64_t n = 0;
  for (std::size_t i = 0; i < length.size(); ++i) {
    n |= static_cast<std::uint64_t>(std::to_integer<std::uint8_t>(length[i])) << (8U * i);
  }
  return n;
}

// One shard's tensors from its open file, and its format.
std::expected<std::vector<Tensor>, Error> ReadShard(OpenFile& f, std::uint32_t file,
                                                    ShardFormat& format) {
  std::array<std::byte, 8> length{};
  if (f.bytes < length.size()) {
    return Fail(Rule::kContainer, "a safetensors file shorter than its header length", file);
  }
  if (auto ok = ReadExactly(f.fd.get(), 0, length, file); !ok) {
    return std::unexpected(ok.error());
  }
  const std::uint64_t n = HeaderLength(length);
  if (n > kMaxHeaderBytes) {
    return Fail(Rule::kBounds, "a safetensors header beyond its cap", file);
  }
  if (n > f.bytes - 8) {
    return Fail(Rule::kContainer, "a safetensors header past the end of its file", file);
  }
  std::string header(static_cast<std::size_t>(n), '\0');
  if (auto ok = ReadExactly(f.fd.get(), 8, std::as_writable_bytes(std::span(header)), file); !ok) {
    return std::unexpected(ok.error());
  }
  f.parsed = {8 + n, base::Sha256().Update(length).Update(header).Finish()};
  return ParseHeader(header, 8 + n, f.bytes, file, format);
}

// The index's weight map: tensor name to shard name.
std::expected<std::map<std::string, std::string, std::less<>>, Error> ParseIndex(
    std::string_view text) {
  const auto doc = json::Parse(text, DocumentLimits(kMaxIndexBytes, 8, std::size_t{1} << 22U));
  if (!doc) {
    return Fail(Rule::kConfig, "the shard index is not valid JSON within the reader's limits",
                kNoItem, ItemKind::kByte, doc.error().offset);
  }
  const json::Value root = doc->root();
  // Other members (`metadata`, a writer's own) describe nothing read here.
  const auto map = root.is_object() ? root.find("weight_map") : std::nullopt;
  if (!map || !map->is_object()) {
    return Fail(Rule::kConfig, "the shard index has no weight_map object");
  }
  if (map->size() > kMaxTensors) {
    return Fail(Rule::kBounds, "more tensors than the reader's cap");
  }
  std::map<std::string, std::string, std::less<>> out;
  for (std::size_t i = 0; i < map->size(); ++i) {
    const json::Value shard = map->member(i);
    if (!TensorName(map->key(i)) || !shard.is_string() || !PlainName(shard.string()) ||
        !shard.string().ends_with(kShardSuffix)) {
      return Fail(Rule::kConfig, "a shard index entry that is not a tensor and a shard file name",
                  kNoItem, ItemKind::kKey, i);
    }
    out.emplace(std::string(map->key(i)), std::string(shard.string()));
  }
  return out;
}

// The weights' files beneath a directory under one base name and variant:
// its index, its single file, either or both, by name.
struct Weights {
  std::string index_name{};
  std::string single_name{};
  std::optional<OpenFile> index{};
  std::optional<OpenFile> single{};
};

// A variant's name (diffusers' and transformers' `variant`: "fp16", "bf16",
// "non_ema"): letters, digits and '_', which a sharded file's
// "-00001-of-00002" never is.
bool VariantName(std::string_view v) {
  return !v.empty() && v.size() <= kMaxVariant && std::ranges::all_of(v, [](char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_';
  });
}

// The variant `name` holds weights of under `base`: <base>.<v>.safetensors
// or <base>.safetensors.index.<v>.json (both libraries' `_add_variant`,
// the variant before the last extension), or none.
std::optional<std::string_view> VariantOf(std::string_view name, std::string_view base) {
  if (!name.starts_with(base) || !name.substr(base.size()).starts_with('.')) {
    return std::nullopt;
  }
  name.remove_prefix(base.size() + 1);
  constexpr std::string_view kIndexStem = "safetensors.index.";
  constexpr std::string_view kJson = ".json";
  std::string_view v;
  if (name.size() > kIndexStem.size() + kJson.size() && name.starts_with(kIndexStem) &&
      name.ends_with(kJson)) {
    v = name.substr(kIndexStem.size(), name.size() - kIndexStem.size() - kJson.size());
  } else if (name.size() > kShardSuffix.size() && name.ends_with(kShardSuffix)) {
    v = name.substr(0, name.size() - kShardSuffix.size());
  } else {
    return std::nullopt;
  }
  return VariantName(v) ? std::optional(v) : std::nullopt;
}

// The files of `base` under `variant` (none: the default names), opened
// when present.
std::expected<Weights, Error> OpenWeights(const std::filesystem::path& root, std::string_view base,
                                          std::string_view variant) {
  Weights w;
  if (variant.empty()) {
    w.index_name = std::string(base) + std::string(kIndexSuffix);
    w.single_name = std::string(base) + std::string(kShardSuffix);
  } else {
    w.index_name = std::format("{}.safetensors.index.{}.json", base, variant);
    w.single_name = std::format("{}.{}{}", base, variant, kShardSuffix);
  }
  auto index = OpenCheckpointFileIfPresent(root / w.index_name, kNoItem);
  if (!index) {
    return std::unexpected(index.error());
  }
  auto single = OpenCheckpointFileIfPresent(root / w.single_name, 0);
  if (!single) {
    return std::unexpected(single.error());
  }
  w.index = std::move(*index);
  w.single = std::move(*single);
  return w;
}

// The weights a directory holds: under the default names of one base, or,
// with none, under the one variant of one base it holds (as a library
// loading it without a variant would, or could only, read). Weights under
// both bases, or several variants and no default, are ambiguous.
std::expected<Weights, Error> FindWeights(const std::filesystem::path& root) {
  std::optional<Weights> found;
  for (const std::string_view base : kWeightBases) {
    auto w = OpenWeights(root, base, {});
    if (!w) {
      return std::unexpected(w.error());
    }
    if (!w->index && !w->single) {
      continue;
    }
    if (found) {
      return Fail(Rule::kShards, "weights under both model and diffusion_pytorch_model");
    }
    found = std::move(*w);
  }
  if (found) {
    return std::move(*found);
  }
  std::error_code error;
  std::filesystem::directory_iterator it(root, error);
  if (error) {
    return Fail(Rule::kIo, "cannot list a checkpoint's directory");
  }
  std::set<std::pair<std::string_view, std::string>> variants;  // base, variant
  std::size_t entries = 0;
  while (it != std::filesystem::directory_iterator()) {
    if (++entries > kMaxDirectoryEntries) {
      return Fail(Rule::kBounds, "a checkpoint directory of more entries than the reader's cap");
    }
    const std::string name = it->path().filename().string();
    for (const std::string_view base : kWeightBases) {
      if (const auto v = VariantOf(name, base)) {
        variants.emplace(base, std::string(*v));
      }
    }
    it.increment(error);
    if (error) {
      return Fail(Rule::kIo, "cannot list a checkpoint's directory");
    }
  }
  if (variants.empty()) {
    return Fail(Rule::kFileType, "no model.safetensors and no shard index");
  }
  if (variants.size() > 1) {
    return Fail(Rule::kShards, "weights under several variants and no default, or both bases");
  }
  return OpenWeights(root, variants.begin()->first, variants.begin()->second);
}

}  // namespace

bool HasComponent(std::string_view name, std::string_view component) {
  std::size_t start = 0;
  while (true) {
    const std::size_t dot = name.find('.', start);
    if (name.substr(start, dot == std::string_view::npos ? dot : dot - start) == component) {
      return true;
    }
    if (dot == std::string_view::npos) {
      return false;
    }
    start = dot + 1;
  }
}

std::expected<std::vector<MetaEntry>, Error> FlattenMetadata(
    json::Value root, std::span<const std::string_view> skip) {
  if (!root.is_object()) {
    return Fail(Rule::kConfig, "a configuration document that is not an object");
  }
  std::map<std::string, MetaValue, std::less<>> flat;
  if (auto ok = Flatten(root, {}, skip, flat); !ok) {
    return std::unexpected(ok.error());
  }
  std::vector<MetaEntry> out;
  out.reserve(flat.size());
  for (auto& [key, value] : flat) {
    out.push_back({.key = key, .value = std::move(value)});
  }
  return out;
}

std::optional<json::Document> PeekConfig(const std::filesystem::path& directory) {
  auto f = OpenCheckpointFileIfPresent(directory / kConfigName, kNoItem);
  if (!f || !f->has_value()) {
    return std::nullopt;
  }
  auto text = ReadDocument(**f, kMaxConfigBytes, kNoItem);
  if (!text) {
    return std::nullopt;
  }
  auto doc = ParseConfigDocument(*text, {}, {});
  if (!doc || !doc->root().is_object()) {
    return std::nullopt;
  }
  return std::move(*doc);
}

std::expected<HfDirectory, Error> ReadHfDirectory(const std::filesystem::path& directory) {
  auto root = CheckpointDirectory(directory);
  if (!root) {
    return std::unexpected(root.error());
  }
  std::vector<SourceFile> sources;
  // A document beside the shards, when present; it follows them in the
  // checkpoint's file order. Each file is open only while it is read.
  const auto keep = [&](std::string_view name, const OpenFile& opened,
                        std::uint64_t limit) -> std::expected<std::string, Error> {
    auto text = ReadDocument(opened, limit, kNoItem);
    if (!text) {
      return std::unexpected(text.error());
    }
    sources.push_back({.name = std::string(name),
                       .path = opened.path,
                       .bytes = opened.bytes,
                       .device = opened.device,
                       .inode = opened.inode,
                       .parsed = {opened.bytes, base::Sha256().Update(*text).Finish()}});
    return std::move(*text);
  };
  const auto document =
      [&](std::string_view name,
          std::uint64_t limit) -> std::expected<std::optional<std::string>, Error> {
    auto f = OpenCheckpointFileIfPresent(*root / name, kNoItem);
    if (!f) {
      return std::unexpected(f.error());
    }
    if (!f->has_value()) {
      return std::optional<std::string>();
    }
    auto text = keep(name, **f, limit);
    if (!text) {
      return std::unexpected(text.error());
    }
    return std::optional<std::string>(std::move(*text));
  };
  auto config_text = document(kConfigName, kMaxConfigBytes);
  if (!config_text) {
    return std::unexpected(config_text.error());
  }
  if (!config_text->has_value()) {
    return Fail(Rule::kConfig, "the checkpoint has no config.json");
  }
  auto hf_text = document(kHfQuantName, kMaxConfigBytes);
  if (!hf_text) {
    return std::unexpected(hf_text.error());
  }
  std::optional<std::string_view> hf;
  if (hf_text->has_value()) {
    hf = **hf_text;
  }
  auto config = ParseConfig(**config_text, hf);
  if (!config) {
    return std::unexpected(config.error());
  }
  auto weights = FindWeights(*root);
  if (!weights) {
    return std::unexpected(weights.error());
  }
  const std::string& single_name = weights->single_name;
  std::optional<std::map<std::string, std::string, std::less<>>> index;
  std::vector<std::string> shards;
  if (weights->index) {
    auto text = keep(weights->index_name, *weights->index, kMaxIndexBytes);
    weights->index.reset();  // closed
    if (!text) {
      return std::unexpected(text.error());
    }
    auto parsed = ParseIndex(*text);
    if (!parsed) {
      return std::unexpected(parsed.error());
    }
    index = std::move(*parsed);
    std::set<std::string> names;
    for (const auto& [tensor, shard] : *index) {
      names.insert(shard);
    }
    if (names.size() > kMaxShards) {
      return Fail(Rule::kBounds, "more shards than the reader's cap");
    }
    shards.assign(names.begin(), names.end());
  }
  std::optional<OpenFile>& single = weights->single;
  if (single && index) {
    // The single file beside an index is one of its shards, or a stray
    // that makes the checkpoint ambiguous.
    if (!std::ranges::binary_search(shards, single_name)) {
      return Fail(Rule::kShards, "both model.safetensors and a shard index that does not name it");
    }
    single.reset();
  }
  if (single) {
    shards.push_back(single_name);
  }
  std::vector<Tensor> tensors;
  std::set<std::string_view> seen;
  bool mlx = false;
  bool other = false;
  // The shards come first in the checkpoint's file order; each is open
  // only while its header is read.
  std::vector<SourceFile> shard_files;
  for (std::size_t s = 0; s < shards.size(); ++s) {
    const auto file = static_cast<std::uint32_t>(s);
    // The single file was opened above; an index's shards are opened here.
    std::expected<OpenFile, Error> f = single ? std::expected<OpenFile, Error>(std::move(*single))
                                              : OpenCheckpointFile(*root / shards[s], file);
    if (!f) {
      return std::unexpected(f.error());
    }
    ShardFormat format = ShardFormat::kNone;
    auto shard = ReadShard(*f, file, format);
    if (!shard) {
      return std::unexpected(shard.error());
    }
    mlx = mlx || format == ShardFormat::kMlx;
    other = other || format == ShardFormat::kOther;
    if (mlx && other) {
      return Fail(Rule::kShards, "shards in MLX's format and in another's", file);
    }
    if (tensors.size() + shard->size() > kMaxTensors) {
      return Fail(Rule::kBounds, "more tensors than the reader's cap", file);
    }
    for (std::size_t t = 0; t < shard->size(); ++t) {
      const std::string& name = (*shard)[t].name;
      if (index) {
        const auto found = index->find(name);
        if (found == index->end() || found->second != shards[s]) {
          return Fail(Rule::kShards, "a shard holds a tensor the index does not place in it", file,
                      ItemKind::kTensor, t);
        }
      }
    }
    std::ranges::move(*shard, std::back_inserter(tensors));
    shard_files.push_back({.name = shards[s],
                           .path = f->path,
                           .bytes = f->bytes,
                           .device = f->device,
                           .inode = f->inode,
                           .parsed = f->parsed});
  }
  for (std::size_t t = 0; t < tensors.size(); ++t) {
    if (!seen.insert(tensors[t].name).second) {
      return Fail(Rule::kShards, "a tensor name in two shards", tensors[t].extent.file,
                  ItemKind::kTensor, t);
    }
  }
  if (index && index->size() != tensors.size()) {
    return Fail(Rule::kShards, "the shard index names a tensor no shard holds");
  }
  HfDirectory out{
      .config = std::move(config->document), .hf_quant = std::move(config->hf), .mlx = mlx};
  Checkpoint& c = out.checkpoint;
  c.container = "safetensors";
  c.architecture = std::move(config->architecture);
  c.metadata = std::move(config->metadata);
  c.tensors = std::move(tensors);
  // Shards, then the documents (their extents name only shards).
  c.files = std::move(shard_files);
  std::ranges::move(sources, std::back_inserter(c.files));
  return out;
}

std::expected<HfDirectory, Error> ParseHfShard(std::span<const std::byte> shard,
                                               std::string_view config,
                                               std::string_view hf_quant_config) {
  auto parsed = ParseConfig(config, hf_quant_config.empty()
                                        ? std::nullopt
                                        : std::optional<std::string_view>(hf_quant_config));
  if (!parsed) {
    return std::unexpected(parsed.error());
  }
  if (shard.size() < 8) {
    return Fail(Rule::kContainer, "a safetensors file shorter than its header length", 0);
  }
  const std::uint64_t n = HeaderLength(shard.first<8>());
  if (n > kMaxHeaderBytes) {
    return Fail(Rule::kBounds, "a safetensors header beyond its cap", 0);
  }
  if (n > shard.size() - 8) {
    return Fail(Rule::kContainer, "a safetensors header past the end of its file", 0);
  }
  const std::string_view header(reinterpret_cast<const char*>(shard.data() + 8),
                                static_cast<std::size_t>(n));
  ShardFormat format = ShardFormat::kNone;
  auto tensors = ParseHeader(header, 8 + n, shard.size(), 0, format);
  if (!tensors) {
    return std::unexpected(tensors.error());
  }
  HfDirectory out{.config = std::move(parsed->document),
                  .hf_quant = std::move(parsed->hf),
                  .mlx = format == ShardFormat::kMlx};
  Checkpoint& c = out.checkpoint;
  c.container = "safetensors";
  c.architecture = std::move(parsed->architecture);
  c.metadata = std::move(parsed->metadata);
  c.tensors = std::move(*tensors);
  c.files.push_back({.name = "memory", .bytes = shard.size()});
  return out;
}

namespace {

// The checkpoint with the quantization config.json and
// hf_quant_config.json describe applied (hf_quant.h).
std::expected<Checkpoint, Error> Quantized(HfDirectory dir) {
  const std::optional<json::Value> hf =
      dir.hf_quant ? std::optional<json::Value>(dir.hf_quant->root()) : std::nullopt;
  auto quantization = RecognizeQuantization(dir.config.root(), hf ? &*hf : nullptr);
  if (!quantization) {
    return std::unexpected(quantization.error());
  }
  Checkpoint& c = dir.checkpoint;
  if (auto ok = ApplyQuantization(*quantization, c.tensors); !ok) {
    return std::unexpected(ok.error());
  }
  if (*quantization) {
    c.quantization = (*quantization)->method + " " + (*quantization)->algorithm;
  }
  return std::move(c);
}

}  // namespace

std::expected<Checkpoint, Error> ReadSafetensors(const std::filesystem::path& directory) {
  auto dir = ReadHfDirectory(directory);
  if (!dir) {
    return std::unexpected(dir.error());
  }
  if (dir->mlx) {
    return MlxCheckpoint(std::move(*dir));  // MLX's, though unquantized
  }
  return Quantized(std::move(*dir));
}

std::expected<Checkpoint, Error> ParseSafetensors(std::span<const std::byte> shard,
                                                  std::string_view config,
                                                  std::string_view hf_quant_config) {
  auto dir = ParseHfShard(shard, config, hf_quant_config);
  if (!dir) {
    return std::unexpected(dir.error());
  }
  if (dir->mlx) {
    return MlxCheckpoint(std::move(*dir));
  }
  return Quantized(std::move(*dir));
}

}  // namespace llmp::import
