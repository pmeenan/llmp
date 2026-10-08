// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "artifact/representation.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <limits>
#include <optional>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

#include "artifact/error.h"
#include "artifact/json.h"
#include "artifact/schema.h"

namespace llmp::artifact {
namespace {

using schema::Fail;

constexpr std::uint64_t kSaturated = std::numeric_limits<std::uint64_t>::max();
constexpr std::uint32_t kQk = 256;  // QK_K

// Pinned from llama.cpp b29c606e2 gguf-py/gguf/constants.py GGML_QUANT_SIZES,
// as the prototype (layout.py GGML_TYPES) records them.
constexpr std::array kGgmlTypes = {
    GgmlType{"F32", 0, 1, 4},
    GgmlType{"F16", 1, 1, 2},
    GgmlType{"Q4_0", 2, 32, 18},
    GgmlType{"Q4_1", 3, 32, 20},
    GgmlType{"Q5_0", 6, 32, 22},
    GgmlType{"Q5_1", 7, 32, 24},
    GgmlType{"Q8_0", 8, 32, 34},
    GgmlType{"Q8_1", 9, 32, 40},
    GgmlType{"Q2_K", 10, kQk, 2 + 2 + (kQk / 16) + (kQk / 4)},
    GgmlType{"Q3_K", 11, kQk, 2 + (kQk / 4) + (kQk / 8) + 12},
    GgmlType{"Q4_K", 12, kQk, 2 + 2 + (kQk / 2) + 12},
    GgmlType{"Q5_K", 13, kQk, 2 + 2 + (kQk / 2) + (kQk / 8) + 12},
    GgmlType{"Q6_K", 14, kQk, 2 + (kQk / 2) + (kQk / 4) + (kQk / 16)},
    GgmlType{"Q8_K", 15, kQk, 4 + kQk + (kQk / 8)},
    GgmlType{"IQ2_XXS", 16, kQk, 2 + (kQk / 4)},
    GgmlType{"IQ2_XS", 17, kQk, 2 + (kQk / 4) + (kQk / 32)},
    GgmlType{"IQ3_XXS", 18, kQk, 2 + (kQk / 4) + (kQk / 8)},
    GgmlType{"IQ1_S", 19, kQk, 2 + (kQk / 8) + (kQk / 16)},
    GgmlType{"IQ4_NL", 20, 32, 18},
    GgmlType{"IQ3_S", 21, kQk, 2 + (kQk / 4) + (kQk / 8) + (kQk / 32) + 4},
    GgmlType{"IQ2_S", 22, kQk, 2 + (kQk / 4) + (kQk / 16)},
    GgmlType{"IQ4_XS", 23, kQk, 2 + 2 + (kQk / 2) + (kQk / 64)},
    GgmlType{"I8", 24, 1, 1},
    GgmlType{"I16", 25, 1, 2},
    GgmlType{"I32", 26, 1, 4},
    GgmlType{"I64", 27, 1, 8},
    GgmlType{"F64", 28, 1, 8},
    GgmlType{"IQ1_M", 29, kQk, (kQk / 8) + (kQk / 16) + (kQk / 32)},
    GgmlType{"BF16", 30, 1, 2},
    GgmlType{"TQ1_0", 34, kQk, 2 + (4 * 13)},
    GgmlType{"TQ2_0", 35, kQk, 2 + 64},
    GgmlType{"MXFP4", 39, 32, 17},
    GgmlType{"NVFP4", 40, 64, 36},
    GgmlType{"Q1_0", 41, 128, 18},
    GgmlType{"Q2_0", 42, 64, 18},
};

// layout.py ST_SIZES.
constexpr std::array kDtypes = {
    Dtype{"BOOL", 1}, Dtype{"U8", 1},  Dtype{"I8", 1},  Dtype{"F8_E5M2", 1}, Dtype{"F8_E4M3", 1},
    Dtype{"I16", 2},  Dtype{"U16", 2}, Dtype{"F16", 2}, Dtype{"BF16", 2},    Dtype{"I32", 4},
    Dtype{"U32", 4},  Dtype{"F32", 4}, Dtype{"I64", 8}, Dtype{"U64", 8},     Dtype{"F64", 8},
};

// layout.py ST_NATIVE: GGML type names that are also safetensors dtypes.
constexpr std::array<std::string_view, 8> kNative = {"F32", "F16", "BF16", "I8",
                                                     "I16", "I32", "I64",  "F64"};

constexpr std::array<std::string_view, 4> kExl3Roles = {"trellis", "suh", "svh", "mcg"};
// EXL3 part suffixes (layout.py EXL3_PART).
constexpr std::array<std::string_view, 8> kExl3Parts = {"trellis", "suh", "svh",  "su",
                                                        "sv",      "mcg", "mul1", "bias"};
constexpr std::string_view kCodebookMcg = "mcg";

std::uint64_t Mul(std::uint64_t a, std::uint64_t b) {
  std::uint64_t out = 0;
  return __builtin_mul_overflow(a, b, &out) ? kSaturated : out;
}

std::uint64_t Add(std::uint64_t a, std::uint64_t b) {
  std::uint64_t out = 0;
  return __builtin_add_overflow(a, b, &out) ? kSaturated : out;
}

// The exact product, or saturated; a zero factor makes it zero whatever the
// others are, as exact arithmetic would.
std::uint64_t Product(std::span<const std::uint64_t> values) {
  std::uint64_t out = 1;
  for (const std::uint64_t v : values) {
    if (v == 0) {
      return 0;
    }
    out = Mul(out, v);
  }
  return out;
}

std::uint64_t GgmlRowBytes(const GgmlType& type, std::uint64_t elements) {
  return Mul(elements / type.block_elements, type.block_bytes);
}

// A static copy of a known name, so a representation never points into the
// artifact's bytes.
template <typename Table>
std::string_view Intern(const Table& table, std::string_view name) {
  for (const std::string_view known : table) {
    if (known == name) {
      return known;
    }
  }
  return {};
}

std::expected<Representation, Error> ParseGgml(json::Value v) {
  if (auto ok = schema::Object(v, {"family", "type", "ne"}, {}, "ggml representation keys"); !ok) {
    return std::unexpected(ok.error());
  }
  const json::Value type_value = schema::Get(v, "type");
  const GgmlType* type = type_value.is_string() ? FindGgmlType(type_value.string()) : nullptr;
  if (type == nullptr) {
    return Fail(Rule::kRepr, "unknown ggml type");
  }
  const json::Value ne = schema::Get(v, "ne");
  if (auto ok = schema::List(ne, 1, "ggml ne must be a list"); !ok) {
    return std::unexpected(ok.error());
  }
  if (ne.size() > kMaxGgmlDims) {
    return Fail(Rule::kRepr, "ggml ne has more than 4 dimensions");
  }
  Representation rep;
  rep.family = Family::kGgml;
  rep.type = type->name;
  for (std::size_t i = 0; i < ne.size(); ++i) {
    const auto n = ne.at(i).integer();
    if (!n || *n < 1) {
      return Fail(Rule::kRepr, "ggml ne must be positive integers");
    }
    rep.dims.push_back(static_cast<std::uint64_t>(*n));
  }
  if (rep.dims[0] % type->block_elements != 0) {
    return Fail(Rule::kRepr, "ggml row not a multiple of the block");
  }
  rep.bytes = Mul(GgmlRowBytes(*type, rep.dims[0]), Product(std::span(rep.dims).subspan(1)));
  rep.readable = rep.bytes;
  if (type->block_elements > 1 && rep.dims[0] % kGgmlRowPadding != 0) {
    rep.readable =
        Add(rep.bytes, GgmlRowBytes(*type, kGgmlRowPadding - (rep.dims[0] % kGgmlRowPadding)));
  }
  return rep;
}

// Whether v is an integer exactly 16 * n. 16n may not fit 64 bits, so v's
// digits are divided by 16 instead: an integer has at most 20 digits, so
// every prefix's sixteenth fits (the parser refuses leading zeros, so the
// digits are the value's).
static_assert(json::kMaxIntegerChars <= 20,
              "a longer integer's sixteenth could wrap, matching a small n");
bool IsSixteenTimes(json::Value v, std::uint64_t n) {
  const std::string_view digits = v.raw_integer();
  if (digits.empty() || digits.front() == '-') {
    return false;  // not an integer, or negative ("-0" is never canonical)
  }
  std::uint64_t quotient = 0;
  std::uint64_t remainder = 0;
  for (const char c : digits) {
    const std::uint64_t current = (remainder * 10) + static_cast<std::uint64_t>(c - '0');
    quotient = (quotient * 10) + (current / 16);
    remainder = current % 16;
  }
  return remainder == 0 && quotient == n;
}

// Whether an EXL3 part is one v0 supports, consistently described; fills a
// trellis's fields.
bool Exl3Consistent(Representation& rep, json::Value v) {
  const auto& shape = rep.dims;
  if (rep.role == "trellis") {
    const auto k = schema::Get(v, "k_bits").integer();
    const json::Value codebook = schema::Get(v, "codebook");
    if (rep.type != "I16" || shape.size() != 3 || !k || codebook.string() != kCodebookMcg ||
        (*k != 4 && *k != 5 && *k != 6 && *k != 8) ||
        shape[2] != 16 * static_cast<std::uint64_t>(*k)) {
      return false;
    }
    // _exact(in_features, 16 * shape[0]), on exact integers: a product
    // beyond 64 bits still matches its digits, so that such a tensor is
    // refused by the rule that follows (its bytes, or an expert array's
    // family), as in the prototype.
    if (!IsSixteenTimes(schema::Get(v, "in_features"), shape[0]) ||
        !IsSixteenTimes(schema::Get(v, "out_features"), shape[1])) {
      return false;
    }
    rep.k_bits = static_cast<std::uint32_t>(*k);
    rep.in_features = Mul(16, shape[0]);  // saturated beyond 64 bits
    rep.out_features = Mul(16, shape[1]);
    rep.codebook = kCodebookMcg;
    return true;
  }
  if (rep.role == "suh" || rep.role == "svh") {
    return rep.type == "F16" && shape.size() == 1;
  }
  if (rep.role == "mcg") {
    return rep.type == "I32" && shape.empty();
  }
  return false;  // su/sv packed signs, mul1 and 3inst need their own fixtures
}

std::expected<Representation, Error> ParseTensor(json::Value v, Family family) {
  const bool exl3 = family == Family::kExl3;
  const auto role_value = v.find("role");
  const bool trellis = role_value && schema::IsString(*role_value, "trellis");
  std::expected<void, Error> keys;
  if (!exl3) {
    keys = schema::Object(v, {"family", "dtype", "shape"}, {}, "plain representation keys");
  } else if (trellis) {
    keys = schema::Object(
        v,
        {"family", "dtype", "shape", "role", "k_bits", "in_features", "out_features", "codebook"},
        {}, "exl3 representation keys");
  } else {
    keys = schema::Object(v, {"family", "dtype", "shape", "role"}, {}, "exl3 representation keys");
  }
  if (!keys) {
    return std::unexpected(keys.error());
  }
  const json::Value dtype_value = schema::Get(v, "dtype");
  const Dtype* dtype = dtype_value.is_string() ? FindDtype(dtype_value.string()) : nullptr;
  if (dtype == nullptr) {
    return Fail(Rule::kRepr, "unknown dtype");
  }
  Representation rep;
  rep.family = family;
  rep.type = dtype->name;
  if (exl3) {
    const json::Value role = schema::Get(v, "role");
    const auto codebook = v.find("codebook");
    if (!role.is_string() || (codebook && !codebook->is_string())) {
      return Fail(Rule::kRepr, "exl3 role and codebook must be strings");
    }
    rep.role = Intern(kExl3Roles, role.string());
  }
  const json::Value shape = schema::Get(v, "shape");
  if (auto ok = schema::List(shape, 0, "shape must be a list"); !ok) {
    return std::unexpected(ok.error());
  }
  if (shape.size() > kMaxShapeDims) {
    return Fail(Rule::kRepr, "shape has more than 8 dimensions");
  }
  for (std::size_t i = 0; i < shape.size(); ++i) {
    const auto n = shape.at(i).integer();
    if (!n || *n < 0) {
      return Fail(Rule::kRepr, "shape must be non-negative integers");
    }
    rep.dims.push_back(static_cast<std::uint64_t>(*n));
  }
  rep.bytes = Mul(Product(rep.dims), dtype->bytes);
  rep.readable = rep.bytes;  // EXL3 over-read is not established; plain has none
  if (exl3) {
    if (rep.role.empty() || !Exl3Consistent(rep, v)) {
      return Fail(Rule::kRepr, "unsupported or inconsistent exl3 part");
    }
  }
  if (rep.bytes == 0) {
    return Fail(Rule::kRepr, "empty tensor");
  }
  return rep;
}

// The EXL3 part suffix of a name: (prefix, part), or nothing.
std::optional<std::pair<std::string_view, std::string_view>> Exl3Part(std::string_view name) {
  const std::size_t dot = name.rfind('.');
  if (dot == std::string_view::npos || dot == 0) {
    return std::nullopt;
  }
  const std::string_view part = name.substr(dot + 1);
  if (std::ranges::find(kExl3Parts, part) == kExl3Parts.end()) {
    return std::nullopt;
  }
  return std::pair{name.substr(0, dot), part};
}

}  // namespace

std::string_view FamilyName(Family family) {
  switch (family) {
    case Family::kGgml:
      return "ggml";
    case Family::kExl3:
      return "exl3";
    case Family::kPlain:
      return "plain";
  }
  return "unknown";
}

std::span<const GgmlType> GgmlTypes() { return kGgmlTypes; }

const GgmlType* FindGgmlType(std::string_view name) {
  const auto* found =
      std::ranges::find_if(kGgmlTypes, [&](const GgmlType& t) { return t.name == name; });
  return found == kGgmlTypes.end() ? nullptr : found;
}

std::span<const Dtype> Dtypes() { return kDtypes; }

const Dtype* FindDtype(std::string_view name) {
  const auto* found = std::ranges::find_if(kDtypes, [&](const Dtype& d) { return d.name == name; });
  return found == kDtypes.end() ? nullptr : found;
}

bool IsNativeContainerType(std::string_view ggml_type) {
  return std::ranges::find(kNative, ggml_type) != kNative.end();
}

std::expected<Representation, Error> ParseRepresentation(json::Value v) {
  if (!v.is_object()) {
    return Fail(Rule::kRepr, "representation is not an object");
  }
  const auto family = v.find("family");
  if (!family || !family->is_string()) {
    return Fail(Rule::kRepr, "representation family must be a string");
  }
  if (family->string() == "ggml") {
    return ParseGgml(v);
  }
  if (family->string() == "exl3") {
    return ParseTensor(v, Family::kExl3);
  }
  if (family->string() == "plain") {
    return ParseTensor(v, Family::kPlain);
  }
  return Fail(Rule::kRepr, "unknown representation family");
}

ContainerView ContainerViewOf(const Representation& rep) {
  if (rep.family == Family::kGgml) {
    if (IsNativeContainerType(rep.type)) {
      return {.dtype = rep.type, .shape = {rep.dims.rbegin(), rep.dims.rend()}};
    }
    return {.dtype = "U8", .shape = {rep.bytes}};
  }
  return {.dtype = rep.type, .shape = rep.dims};
}

std::optional<RowGeometry> RowGeometryOf(const Representation& rep) {
  if (rep.dims.size() != 2) {
    return std::nullopt;
  }
  if (rep.family == Family::kGgml) {
    const GgmlType* type = FindGgmlType(rep.type);
    return RowGeometry{.rows = rep.dims[1], .row_bytes = GgmlRowBytes(*type, rep.dims[0])};
  }
  if (rep.dims[0] < 1) {
    return std::nullopt;
  }
  return RowGeometry{.rows = rep.dims[0], .row_bytes = rep.bytes / rep.dims[0]};
}

bool Exl3ClosureHolds(std::span<const std::pair<std::string_view, const Representation*>> bound) {
  struct Part {
    std::string_view prefix;
    std::string_view part;
    const Representation* rep;
  };
  std::vector<Part> parts;
  for (const auto& [name, rep] : bound) {
    const auto split = Exl3Part(name);
    if (split && split->second != "bias" && rep->family != Family::kExl3) {
      return false;
    }
    if (rep->family == Family::kExl3) {
      if (!split || split->second != rep->role) {
        return false;
      }
      parts.push_back({.prefix = split->first, .part = split->second, .rep = rep});
    }
  }
  std::ranges::sort(parts, [](const Part& a, const Part& b) {
    return std::pair(a.prefix, a.part) < std::pair(b.prefix, b.part);
  });
  for (std::size_t i = 0; i < parts.size();) {
    std::size_t end = i;
    const Representation* trellis = nullptr;
    const Representation* suh = nullptr;
    const Representation* svh = nullptr;
    const Representation* mcg = nullptr;
    std::size_t count = 0;
    while (end < parts.size() && parts[end].prefix == parts[i].prefix) {
      const Part& p = parts[end];
      if (p.part == "trellis") {
        trellis = p.rep;
      } else if (p.part == "suh") {
        suh = p.rep;
      } else if (p.part == "svh") {
        svh = p.rep;
      } else if (p.part == "mcg") {
        mcg = p.rep;
      }
      ++count;
      ++end;
    }
    // Exactly {trellis, suh, svh, <codebook>}; v0's only codebook is mcg.
    if (trellis == nullptr || suh == nullptr || svh == nullptr || mcg == nullptr || count != 4 ||
        trellis->codebook != kCodebookMcg) {
      return false;
    }
    if (suh->dims != std::vector<std::uint64_t>{trellis->in_features} ||
        svh->dims != std::vector<std::uint64_t>{trellis->out_features}) {
      return false;
    }
    i = end;
  }
  return true;
}

}  // namespace llmp::artifact
