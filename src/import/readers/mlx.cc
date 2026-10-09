// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "import/readers/mlx.h"

#include <sys/types.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <functional>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "base/json.h"
#include "format/encoding.h"
#include "format/layout.h"
#include "format/tensor_format.h"
#include "import/checkpoint.h"
#include "import/error.h"
#include "import/readers/safetensors.h"

namespace llmp::import {
namespace {

namespace json = base::json;
using format::Encoding;
using format::PartRole;

constexpr std::size_t kMaxOverrides = std::size_t{1} << 18U;
constexpr std::int64_t kMaxGroup = std::int64_t{1} << 30U;

enum class Mode : std::uint8_t { kAffine, kMxfp4, kNvfp4, kMxfp8 };

struct ModeInfo {
  std::string_view name;
  Mode mode;
  Encoding encoding;
  std::uint32_t bits;   // fixed for the fp modes; 0 for affine
  std::uint32_t group;  // likewise
};

constexpr std::array<ModeInfo, 4> kModes = {{
    {"affine", Mode::kAffine, Encoding::kAffine, 0, 0},
    {"mxfp4", Mode::kMxfp4, Encoding::kMxfp4, 4, 32},
    {"nvfp4", Mode::kNvfp4, Encoding::kNvfp4, 4, 16},
    {"mxfp8", Mode::kMxfp8, Encoding::kMxfp8, 8, 32},
}};

struct Scheme {
  const ModeInfo* mode = nullptr;
  std::uint32_t bits = 0;
  std::uint32_t group = 0;
};

// A module's override: not quantized (false), the default (true), or a
// scheme of its own.
struct Override {
  bool quantized = true;
  std::optional<Scheme> scheme{};
};

struct Quantization {
  std::optional<Scheme> fallback{};  // the default, when config gives one
  std::string_view mode{};           // the default's mode name
  std::map<std::string, Override, std::less<>> modules{};
};

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

const ModeInfo* FindMode(std::string_view name) {
  for (const ModeInfo& m : kModes) {
    if (m.name == name) {
      return &m;
    }
  }
  return nullptr;
}

// A positive integer member up to `limit`, or none when absent.
std::expected<std::optional<std::uint32_t>, Error> Count(json::Value object, std::string_view name,
                                                         std::int64_t limit) {
  const auto v = object.find(name);
  if (!v) {
    return std::optional<std::uint32_t>();
  }
  const auto i = v->int64();
  if (!v->is_integer() || !i || *i < 1 || *i > limit) {
    return Refuse(Rule::kConfig, "an MLX group_size or bits that is not a positive integer");
  }
  return std::optional(static_cast<std::uint32_t>(*i));
}

// A scheme object's mode, bits and group: the fp modes' fixed values when
// absent (and refused when others), affine's required; none when an
// affine scheme gives neither (a default with no scheme of its own).
std::expected<std::optional<Scheme>, Error> SchemeOf(json::Value object, bool required) {
  const ModeInfo* mode = &kModes[0];
  if (const auto m = object.find("mode")) {
    mode = m->is_string() ? FindMode(m->string()) : nullptr;
    if (mode == nullptr) {
      return Refuse(Rule::kUnsupported,
                    "an MLX quantization mode other than affine, mxfp4, "
                    "nvfp4 and mxfp8");
    }
  }
  auto bits = Count(object, "bits", 8);
  if (!bits) {
    return std::unexpected(bits.error());
  }
  auto group = Count(object, "group_size", kMaxGroup);
  if (!group) {
    return std::unexpected(group.error());
  }
  if (mode->mode != Mode::kAffine) {
    if ((*bits && **bits != mode->bits) || (*group && **group != mode->group)) {
      return Refuse(Rule::kConfig, "an MLX fp mode with other bits or group_size than its own");
    }
    return Scheme{.mode = mode, .bits = mode->bits, .group = mode->group};
  }
  if (!*bits && !*group && !required) {
    return std::optional<Scheme>();
  }
  if (!*bits || !*group) {
    return Refuse(Rule::kConfig, "an MLX affine scheme without both bits and group_size");
  }
  return Scheme{.mode = mode, .bits = **bits, .group = **group};
}

bool SchemeMember(std::string_view key) {
  return key == "group_size" || key == "bits" || key == "mode";
}

std::expected<Quantization, Error> ParseQuantization(json::Value q) {
  if (!q.is_object()) {
    return Refuse(Rule::kConfig, "an MLX quantization that is not an object");
  }
  Quantization out;
  auto fallback = SchemeOf(q, false);
  if (!fallback) {
    return std::unexpected(fallback.error());
  }
  out.fallback = *fallback;
  out.mode = out.fallback ? out.fallback->mode->name : kModes[0].name;
  if (q.size() > kMaxOverrides + 3) {
    return Refuse(Rule::kBounds, "more MLX quantization overrides than the reader's cap");
  }
  for (std::size_t i = 0; i < q.size(); ++i) {
    const std::string_view key = q.key(i);
    if (SchemeMember(key)) {
      continue;
    }
    if (key.empty() || key.size() > kMaxKeyBytes) {
      return Refuse(Rule::kConfig, "an MLX quantization override without a module path");
    }
    const json::Value v = q.member(i);
    Override o;
    if (v.is_bool()) {
      o.quantized = v.boolean();
    } else if (v.is_object()) {
      auto scheme = SchemeOf(v, true);
      if (!scheme) {
        return std::unexpected(scheme.error());
      }
      o.scheme = *scheme;
    } else {
      return Refuse(Rule::kConfig,
                    "an MLX quantization override that is not an object, true or false");
    }
    out.modules.emplace(std::string(key), o);  // keys are unique (base/json.h)
  }
  return out;
}

std::optional<std::uint64_t> UnsignedBytes(std::string_view dtype) {
  if (dtype == "U8") {
    return 1;
  }
  if (dtype == "U16") {
    return 2;
  }
  if (dtype == "U32") {
    return 4;
  }
  if (dtype == "U64") {
    return 8;
  }
  return std::nullopt;
}

std::optional<format::AffineScale> AffineScaleOf(std::string_view dtype) {
  if (dtype == "F16") {
    return format::AffineScale::kF16;
  }
  if (dtype == "BF16") {
    return format::AffineScale::kBF16;
  }
  if (dtype == "F32") {
    return format::AffineScale::kF32;
  }
  return std::nullopt;
}

bool FpScaleDtype(Mode mode, std::string_view dtype) {
  return dtype == "U8" || (mode == Mode::kNvfp4 ? dtype == "F8_E4M3" : dtype == "F8_E8M0");
}

// The format of a quantized module's codes `w`, scales `s` and, for
// affine, biases `b`, every part's bytes checked.
std::expected<format::TensorFormat, Error> Describe(const Scheme& scheme,
                                                    const std::vector<Tensor>& tensors,
                                                    std::size_t w_index, std::size_t s_index,
                                                    std::optional<std::size_t> b_index) {
  const Tensor& w = tensors[w_index];
  const Tensor& s = tensors[s_index];
  const auto refuse = [&](std::string_view reason) {
    return RefuseTensor(Rule::kQuantization, reason, w, w_index);
  };
  const std::size_t rank = w.dims.size();
  if ((rank != 2 && rank != 3) || s.dims.size() != rank) {
    return refuse("MLX codes or scales of other than two or three dimensions");
  }
  for (std::size_t d = 0; d + 1 < rank; ++d) {
    if (w.dims[d] != s.dims[d]) {
      return refuse("MLX codes and scales of different rows or experts");
    }
  }
  if (!UnsignedBytes(w.dtype)) {
    return refuse("MLX codes that are not of an unsigned integer dtype");
  }
  std::uint64_t k = 0;
  if (__builtin_mul_overflow(s.dims[rank - 1], std::uint64_t{scheme.group}, &k)) {
    return refuse("an MLX row beyond 64-bit sizes");
  }
  format::TensorFormat f{.layout = format::Layout::kSplit};
  f.shape = {.ne = {k, w.dims[rank - 2], rank == 3 ? w.dims[0] : 1, 1},
             .rank = static_cast<std::uint8_t>(rank)};
  const Mode mode = scheme.mode->mode;
  if (mode == Mode::kAffine) {
    const auto scale = AffineScaleOf(s.dtype);
    if (!scale) {
      return refuse("MLX affine scales that are not F16, BF16 or F32");
    }
    if (!b_index) {
      return refuse("an MLX affine weight without its biases");
    }
    const Tensor& b = tensors[*b_index];
    if (b.dtype != s.dtype || b.dims != s.dims) {
      return refuse("MLX biases of another dtype or shape than their scales");
    }
    f.encoding = Encoding::kAffine;
    f.params = {.bits = static_cast<std::uint8_t>(scheme.bits),
                .group = scheme.group,
                .scale = *scale,
                .zero_point = false};
  } else {
    if (!FpScaleDtype(mode, s.dtype)) {
      return refuse("MLX fp scales that are not bytes of their mode's scale format");
    }
    f.encoding = scheme.mode->encoding;
  }
  const auto parts = format::PartsOf(f);
  if (!parts) {
    return refuse("an MLX weight whose rows are not whole groups of its scheme");
  }
  const auto bytes = [&](PartRole role) { return parts->Find(role)->bytes; };
  if (bytes(PartRole::kCodes) != w.extent.bytes ||
      bytes(PartRole::kBlockScales) != s.extent.bytes ||
      (mode == Mode::kAffine && bytes(PartRole::kBiases) != tensors[*b_index].extent.bytes)) {
    return refuse("an MLX weight's parts are not the sizes its scheme gives them");
  }
  return f;
}

std::expected<void, Error> Apply(const Quantization& q, std::vector<Tensor>& tensors) {
  std::unordered_map<std::string_view, std::size_t> index;
  index.reserve(tensors.size());
  for (std::size_t i = 0; i < tensors.size(); ++i) {
    index.emplace(tensors[i].name, i);
  }
  const auto find = [&](const std::string& name) -> std::optional<std::size_t> {
    const auto found = index.find(name);
    return found == index.end() ? std::nullopt : std::optional(found->second);
  };
  constexpr std::string_view kScales = ".scales";
  for (std::size_t i = 0; i < tensors.size(); ++i) {
    const std::string_view name = tensors[i].name;
    if (!name.ends_with(kScales) || name.size() == kScales.size()) {
      continue;
    }
    const std::string module(name.substr(0, name.size() - kScales.size()));
    const auto weight = find(module + ".weight");
    if (!weight) {
      continue;  // not a quantized module's scales: a tensor of its own
    }
    std::optional<Scheme> scheme = q.fallback;
    if (const auto o = q.modules.find(module); o != q.modules.end()) {
      if (!o->second.quantized) {
        return RefuseTensor(Rule::kQuantization,
                            "an MLX module the config leaves unquantized with scales",
                            tensors[*weight], *weight);
      }
      if (o->second.scheme) {
        scheme = o->second.scheme;
      }
    }
    if (!scheme) {
      return RefuseTensor(Rule::kConfig, "an MLX quantized module with no scheme in the config",
                          tensors[*weight], *weight);
    }
    const auto biases =
        scheme->mode->mode == Mode::kAffine ? find(module + ".biases") : std::nullopt;
    auto f = Describe(*scheme, tensors, *weight, i, biases);
    if (!f) {
      return std::unexpected(f.error());
    }
    const auto set = [&](std::size_t t, PartRole role) {
      tensors[t].format = *f;
      tensors[t].part = role;
      tensors[t].experts = f->shape.rank == 3;
    };
    set(*weight, PartRole::kCodes);
    set(i, PartRole::kBlockScales);
    if (biases) {
      set(*biases, PartRole::kBiases);
    }
  }
  return {};
}

// mlx-lm's expert arrays (SwitchGLU's switch_mlp), dense ones included.
void MarkExpertArrays(std::vector<Tensor>& tensors) {
  for (Tensor& t : tensors) {
    if (t.format.layout == format::Layout::kDense && t.dims.size() == 3 &&
        HasComponent(t.name, "switch_mlp")) {
      t.experts = true;
    }
  }
}

}  // namespace

std::expected<Checkpoint, Error> MlxCheckpoint(HfDirectory dir) {
  Checkpoint& c = dir.checkpoint;
  c.container = "mlx";
  const auto q = dir.config.root().find("quantization");
  if (!q || q->is_null()) {
    MarkExpertArrays(c.tensors);  // an unquantized MLX checkpoint: every tensor dense
    return std::move(c);
  }
  auto quantization = ParseQuantization(*q);
  if (!quantization) {
    return std::unexpected(quantization.error());
  }
  if (auto ok = Apply(*quantization, c.tensors); !ok) {
    return std::unexpected(ok.error());
  }
  MarkExpertArrays(c.tensors);
  c.quantization = "mlx " + std::string(quantization->mode);
  return std::move(c);
}

bool IsMlxConfig(json::Value root) {
  const auto q = root.is_object() ? root.find("quantization") : std::nullopt;
  return q && q->is_object();
}

std::expected<Checkpoint, Error> ReadMlx(const std::filesystem::path& directory) {
  auto dir = ReadHfDirectory(directory);
  if (!dir) {
    return std::unexpected(dir.error());
  }
  return MlxCheckpoint(std::move(*dir));
}

std::expected<Checkpoint, Error> ParseMlx(std::span<const std::byte> shard,
                                          std::string_view config) {
  auto dir = ParseHfShard(shard, config);
  if (!dir) {
    return std::unexpected(dir.error());
  }
  return MlxCheckpoint(std::move(*dir));
}

}  // namespace llmp::import
