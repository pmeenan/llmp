// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "import/readers/exl3.h"

#include <sys/types.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
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
#include "import/readers/safetensors.h"

namespace llmp::import {
namespace {

namespace json = base::json;
using format::Encoding;
using format::Exl3Codebook;
using format::PartRole;

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

std::uint64_t Elements(const Tensor& t) {
  std::uint64_t n = 1;
  for (const std::uint64_t d : t.dims) {
    n *= d;  // the reader checked the product
  }
  return n;
}

// An EXL3 part (a trellis, its sign vectors or a codebook marker) the
// safetensors reader recorded as empty: present, so its linear's decode
// would change with it read as absent (an empty marker as 3inst).
bool EmptyPart(std::span<const SkippedTensor> skipped) {
  return std::ranges::any_of(skipped, [](const SkippedTensor& t) {
    return std::ranges::any_of(
        std::array<std::string_view, 5>{".trellis", ".suh", ".svh", ".mcg", ".mul1"},
        [&](std::string_view suffix) { return t.name.ends_with(suffix); });
  });
}

// config.json's `codebook` (exllamav3's one for every linear), when it
// gives one.
std::expected<std::optional<Exl3Codebook>, Error> StatedCodebook(json::Value root) {
  const auto q = root.find("quantization_config");
  const auto codebook = q && q->is_object() ? q->find("codebook") : std::nullopt;
  if (!codebook) {
    return std::optional<Exl3Codebook>();
  }
  const std::string_view name = codebook->is_string() ? codebook->string() : std::string_view{};
  if (name == "mcg") {
    return Exl3Codebook::kMcg;
  }
  if (name == "mul1") {
    return Exl3Codebook::kMul1;
  }
  if (name == "3inst") {
    return Exl3Codebook::k3Inst;
  }
  return Refuse(Rule::kUnsupported, "an EXL3 codebook other than mcg, mul1 and 3inst");
}

std::expected<void, Error> Apply(std::vector<Tensor>& tensors,
                                 std::span<const SkippedTensor> skipped,
                                 std::optional<Exl3Codebook> stated) {
  if (EmptyPart(skipped)) {
    return Refuse(Rule::kQuantization, "an EXL3 linear with an empty part");
  }
  std::map<std::string_view, std::size_t> index;
  for (std::size_t i = 0; i < tensors.size(); ++i) {
    index.emplace(tensors[i].name, i);
  }
  const auto find = [&](const std::string& name) -> std::optional<std::size_t> {
    const auto found = index.find(name);
    return found == index.end() ? std::nullopt : std::optional(found->second);
  };
  constexpr std::string_view kTrellis = ".trellis";
  for (std::size_t i = 0; i < tensors.size(); ++i) {
    const std::string_view name = tensors[i].name;
    if (!name.ends_with(kTrellis) || name.size() == kTrellis.size()) {
      continue;
    }
    const Tensor& trellis = tensors[i];
    const auto refuse = [&](Rule rule, std::string_view reason) {
      return RefuseTensor(rule, reason, trellis, i);
    };
    const std::string module(name.substr(0, name.size() - kTrellis.size()));
    if (trellis.dtype != "I16" || trellis.dims.size() != 3) {
      return refuse(Rule::kQuantization, "an EXL3 trellis that is not I16 of three dimensions");
    }
    const std::uint64_t last = trellis.dims[2];
    if (last % 16 != 0 || last / 16 < 1 || last / 16 > 8) {
      return refuse(Rule::kUnsupported,
                    "an EXL3 trellis rate other than a whole 1 to 8 bits a weight");
    }
    // The dimensions multiply to below 2^63 (the file holds their I16s), so
    // 16 times either cannot wrap.
    const std::uint64_t k = trellis.dims[0] * 16;
    const std::uint64_t n = trellis.dims[1] * 16;
    const auto suh = find(module + ".suh");
    const auto svh = find(module + ".svh");
    if (!suh || !svh) {
      const bool packed = find(module + ".su") || find(module + ".sv");
      return refuse(packed ? Rule::kUnsupported : Rule::kQuantization,
                    packed ? "EXL3 packed sign vectors (su, sv), which need an unpacking fold"
                           : "an EXL3 trellis without its sign vectors (suh, svh)");
    }
    const auto mcg = find(module + ".mcg");
    const auto mul1 = find(module + ".mul1");
    if (mcg && mul1) {
      return refuse(Rule::kQuantization, "an EXL3 linear with both mcg and mul1 markers");
    }
    const std::optional<std::size_t> marker = mcg ? mcg : mul1;
    const Exl3Codebook codebook = mcg    ? Exl3Codebook::kMcg
                                  : mul1 ? Exl3Codebook::kMul1
                                         : Exl3Codebook::k3Inst;
    if (stated && *stated != codebook) {
      return refuse(Rule::kQuantization,
                    "an EXL3 linear whose codebook marker is not the config's codebook");
    }
    format::TensorFormat f{.encoding = Encoding::kExl3Trellis,
                           .params = {.k_bits = static_cast<std::uint8_t>(last / 16),
                                      .codebook = codebook,
                                      .tensor_scale = false},
                           .layout = format::Layout::kExl3Tiles,
                           .shape = {.ne = {k, n, 1, 1}, .rank = 2}};
    const auto parts = format::PartsOf(f);
    if (!parts) {
      return refuse(Rule::kQuantization, "an EXL3 trellis the size rule does not take");
    }
    const Tensor& su = tensors[*suh];
    const Tensor& sv = tensors[*svh];
    if (su.dtype != "F16" || sv.dtype != "F16" || Elements(su) != k || Elements(sv) != n) {
      return refuse(Rule::kQuantization,
                    "EXL3 sign vectors that are not F16 of the trellis's inputs and outputs");
    }
    const auto bytes = [&](PartRole role) { return parts->Find(role)->bytes; };
    if (bytes(PartRole::kCodes) != trellis.extent.bytes ||
        bytes(PartRole::kInputSigns) != su.extent.bytes ||
        bytes(PartRole::kOutputSigns) != sv.extent.bytes ||
        (marker && bytes(PartRole::kCodebookMarker) != tensors[*marker].extent.bytes)) {
      return refuse(Rule::kQuantization,
                    "an EXL3 linear's parts are not the sizes its trellis gives them");
    }
    const auto set = [&](std::size_t t, PartRole role) {
      tensors[t].format = f;
      tensors[t].part = role;
      tensors[t].experts = false;
    };
    set(i, PartRole::kCodes);
    set(*suh, PartRole::kInputSigns);
    set(*svh, PartRole::kOutputSigns);
    if (marker) {
      set(*marker, PartRole::kCodebookMarker);
    }
  }
  return {};
}

std::expected<Checkpoint, Error> Quantized(HfDirectory dir) {
  if (!IsExl3Config(dir.config.root())) {
    return Refuse(Rule::kConfig, "config.json's quantization_config is not EXL3's");
  }
  // MLX's quantization object beside it describes the tensors otherwise
  // (mlx-lm overwrites quantization_config with its copy, so no MLX
  // checkpoint names EXL3): which describes them is ambiguous.
  if (const auto mlx = dir.config.root().find("quantization"); mlx && mlx->is_object()) {
    return Refuse(Rule::kConfig, "config.json has both EXL3's and MLX's quantization");
  }
  // ModelOpt's description beside EXL3's: which describes the tensors is
  // ambiguous.
  if (dir.hf_quant) {
    return Refuse(Rule::kConfig, "an EXL3 checkpoint with an hf_quant_config.json");
  }
  const auto stated = StatedCodebook(dir.config.root());
  if (!stated) {
    return std::unexpected(stated.error());
  }
  Checkpoint& c = dir.checkpoint;
  c.container = "exl3";
  if (auto ok = Apply(c.tensors, c.skipped_tensors, *stated); !ok) {
    return std::unexpected(ok.error());
  }
  c.quantization = "exl3";
  return std::move(c);
}

}  // namespace

bool IsExl3Config(json::Value root) {
  const auto q = root.is_object() ? root.find("quantization_config") : std::nullopt;
  const auto method = q && q->is_object() ? q->find("quant_method") : std::nullopt;
  return method && method->is_string() && method->string() == "exl3";
}

std::expected<Checkpoint, Error> ReadExl3(const std::filesystem::path& directory) {
  auto dir = ReadHfDirectory(directory);
  if (!dir) {
    return std::unexpected(dir.error());
  }
  return Quantized(std::move(*dir));
}

std::expected<Checkpoint, Error> ParseExl3(std::span<const std::byte> shard,
                                           std::string_view config) {
  auto dir = ParseHfShard(shard, config);
  if (!dir) {
    return std::unexpected(dir.error());
  }
  return Quantized(std::move(*dir));
}

}  // namespace llmp::import
