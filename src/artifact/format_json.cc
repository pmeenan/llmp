// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "artifact/format_json.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include "artifact/error.h"
#include "artifact/json.h"
#include "artifact/schema.h"
#include "format/encoding.h"
#include "format/layout.h"
#include "format/tensor_format.h"

namespace llmp::artifact {
namespace {

using format::AffineScale;
using format::Exl3Codebook;
using format::PartRole;
using schema::Fail;

constexpr std::array<std::pair<std::string_view, Exl3Codebook>, 3> kCodebooks = {{
    {"3inst", Exl3Codebook::k3Inst},
    {"mcg", Exl3Codebook::kMcg},
    {"mul1", Exl3Codebook::kMul1},
}};

constexpr std::array<std::pair<std::string_view, AffineScale>, 3> kScales = {{
    {"bf16", AffineScale::kBF16},
    {"f16", AffineScale::kF16},
    {"f32", AffineScale::kF32},
}};

constexpr std::array kPartRoles = {
    PartRole::kData,           PartRole::kCodes,      PartRole::kBlockScales,
    PartRole::kTensorScale,    PartRole::kInputSigns, PartRole::kOutputSigns,
    PartRole::kCodebookMarker, PartRole::kBiases,     PartRole::kZeros,
};

template <typename T, std::size_t N>
std::optional<T> ByName(const std::array<std::pair<std::string_view, T>, N>& table,
                        std::string_view name) {
  const auto found = std::ranges::find_if(table, [&](const auto& e) { return e.first == name; });
  if (found == table.end()) {
    return std::nullopt;
  }
  return found->second;
}

template <typename T, std::size_t N>
std::string_view NameOf(const std::array<std::pair<std::string_view, T>, N>& table, T value) {
  const auto found = std::ranges::find_if(table, [&](const auto& e) { return e.second == value; });
  return found == table.end() ? std::string_view() : found->first;
}

// A parameter's integer: 1 to hi (0 is its default, which is not written).
std::expected<std::uint64_t, Error> Positive(json::Value v, std::int64_t hi) {
  if (schema::IsInt(v, 0)) {
    return Fail(Rule::kCanonical, "a format parameter written at its default");
  }
  return schema::Int(v, 1, hi, "format.params value");
}

// A flag parameter: only `true` is written.
std::expected<bool, Error> Flag(json::Value v) {
  if (v.kind() == json::Kind::kFalse) {
    return Fail(Rule::kCanonical, "a format parameter written at its default");
  }
  if (v.kind() != json::Kind::kTrue) {
    return Fail(Rule::kSchema, "format.params value");
  }
  return true;
}

std::expected<void, Error> ReadParams(json::Value v, format::EncodingParams& p) {
  if (!v.is_object()) {
    return Fail(Rule::kSchema, "format.params");
  }
  for (std::size_t i = 0; i < v.size(); ++i) {
    const std::string_view key = v.key(i);
    const json::Value value = v.member(i);
    if (key == "k_bits" || key == "bits") {
      auto n = Positive(value, std::numeric_limits<std::uint8_t>::max());
      if (!n) {
        return std::unexpected(n.error());
      }
      (key == "k_bits" ? p.k_bits : p.bits) = static_cast<std::uint8_t>(*n);
    } else if (key == "group") {
      auto n = Positive(value, std::numeric_limits<std::uint32_t>::max());
      if (!n) {
        return std::unexpected(n.error());
      }
      p.group = static_cast<std::uint32_t>(*n);
    } else if (key == "tensor_scale" || key == "zero_point") {
      auto flag = Flag(value);
      if (!flag) {
        return std::unexpected(flag.error());
      }
      (key == "tensor_scale" ? p.tensor_scale : p.zero_point) = true;
    } else if (key == "codebook") {
      if (!value.is_string()) {
        return Fail(Rule::kSchema, "format.params value");
      }
      const auto codebook = ByName(kCodebooks, value.string());
      if (!codebook) {
        return Fail(Rule::kRepr, "an unknown EXL3 codebook");
      }
      p.codebook = *codebook;
    } else if (key == "scale") {
      if (!value.is_string()) {
        return Fail(Rule::kSchema, "format.params value");
      }
      const auto scale = ByName(kScales, value.string());
      if (!scale) {
        return Fail(Rule::kRepr, "an unknown affine scale type");
      }
      p.scale = *scale;
    } else {
      return Fail(Rule::kSchema, "format.params keys");
    }
  }
  return {};
}

void Quoted(std::string& out, std::string_view s) {
  out += '"';
  out += s;
  out += '"';
}

}  // namespace

std::expected<format::TensorFormat, Error> ParseFormat(json::Value v) {
  if (auto ok =
          schema::Object(v, {"encoding", "layout", "ne", "params", "stacked"}, {}, "format keys");
      !ok) {
    return std::unexpected(ok.error());
  }
  const json::Value encoding_name = schema::Get(v, "encoding");
  const json::Value layout_name = schema::Get(v, "layout");
  if (!encoding_name.is_string() || !layout_name.is_string()) {
    return Fail(Rule::kSchema, "format.encoding or format.layout");
  }
  format::TensorFormat f;
  const auto encoding = format::EncodingByName(encoding_name.string());
  if (!encoding) {
    return Fail(Rule::kRepr, "an unknown encoding");
  }
  f.encoding = *encoding;
  const auto layout = format::LayoutByName(layout_name.string());
  if (!layout) {
    return Fail(Rule::kRepr, "an unknown layout");
  }
  f.layout = *layout;
  const json::Value ne = schema::Get(v, "ne");
  if (!ne.is_array() || ne.size() > format::kMaxDims) {
    return Fail(Rule::kSchema, "format.ne");
  }
  for (std::size_t i = 0; i < ne.size(); ++i) {
    auto d = schema::Int(ne.at(i), 1, schema::kMaxInt, "format.ne");
    if (!d) {
      return std::unexpected(d.error());
    }
    f.shape.ne.at(i) = *d;
  }
  f.shape.rank = static_cast<std::uint8_t>(ne.size());
  auto stacked = schema::Int(schema::Get(v, "stacked"), 1, static_cast<std::int64_t>(kMaxStacked),
                             "format.stacked");
  if (!stacked) {
    return std::unexpected(stacked.error());
  }
  f.stacked = static_cast<std::uint8_t>(*stacked);
  if (auto ok = ReadParams(schema::Get(v, "params"), f.params); !ok) {
    return std::unexpected(ok.error());
  }
  if (auto ok = format::Validate(f); !ok) {
    return Fail(Rule::kRepr, format::FormatErrorName(ok.error()));
  }
  return f;
}

void AppendFormatJson(const format::TensorFormat& f, std::string& out) {
  out += R"({"encoding":)";
  Quoted(out, format::Info(f.encoding).name);
  out += R"(,"layout":)";
  Quoted(out, format::Info(f.layout).name);
  out += R"(,"ne":[)";
  for (std::size_t i = 0; i < f.shape.rank && i < format::kMaxDims; ++i) {
    out += i > 0 ? "," : "";
    out += std::to_string(f.shape.ne.at(i));
  }
  out += R"(],"params":{)";
  // Keys in canonical (sorted) order, each only when not at its default.
  const format::EncodingParams& p = f.params;
  bool first = true;
  const auto key = [&](std::string_view name) {
    out += first ? "" : ",";
    first = false;
    Quoted(out, name);
    out += ':';
  };
  if (p.bits != 0) {
    key("bits");
    out += std::to_string(p.bits);
  }
  if (p.codebook != Exl3Codebook::kNone) {
    key("codebook");
    Quoted(out, NameOf(kCodebooks, p.codebook));
  }
  if (p.group != 0) {
    key("group");
    out += std::to_string(p.group);
  }
  if (p.k_bits != 0) {
    key("k_bits");
    out += std::to_string(p.k_bits);
  }
  if (p.scale != AffineScale::kNone) {
    key("scale");
    Quoted(out, NameOf(kScales, p.scale));
  }
  if (p.tensor_scale) {
    key("tensor_scale");
    out += "true";
  }
  if (p.zero_point) {
    key("zero_point");
    out += "true";
  }
  out += R"(},"stacked":)";
  out += std::to_string(f.stacked);
  out += '}';
}

std::optional<format::PartRole> PartRoleByName(std::string_view name) {
  const auto found =
      std::ranges::find_if(kPartRoles, [&](PartRole r) { return format::PartRoleName(r) == name; });
  if (found == kPartRoles.end()) {
    return std::nullopt;
  }
  return *found;
}

ContainerView ContainerViewOf(const format::TensorFormat& f, format::PartRole part,
                              std::uint64_t bytes) {
  if (f.layout == format::Layout::kDense && part == PartRole::kData) {
    if (const auto dtype = format::DtypeOfEncoding(f.encoding)) {
      ContainerView view{.dtype = *dtype, .shape = {}};
      for (std::size_t i = f.shape.rank; i-- > 0;) {
        view.shape.push_back(f.shape.ne.at(i));
      }
      return view;
    }
  }
  return ContainerView{.dtype = "U8", .shape = {bytes}};
}

}  // namespace llmp::artifact
