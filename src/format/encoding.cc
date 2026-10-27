// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "format/encoding.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>

#include "base/check.h"

namespace llmp::format {
namespace {

constexpr std::uint32_t kQk = 256;  // QK_K

constexpr std::array kStateEncodings = {Encoding::kF32,  Encoding::kF16,  Encoding::kBF16,
                                        Encoding::kQ4_0, Encoding::kQ4_1, Encoding::kQ5_0,
                                        Encoding::kQ5_1, Encoding::kQ8_0};

constexpr std::array kNvfp4Parts = {
    PartSpec{.role = PartRole::kCodes, .element = Encoding::kU8, .required = true},
    PartSpec{.role = PartRole::kBlockScales, .element = Encoding::kF8E4M3, .required = true},
    PartSpec{.role = PartRole::kTensorScale, .element = Encoding::kF32, .required = false},
};
constexpr std::array kMxfp4Parts = {
    PartSpec{.role = PartRole::kCodes, .element = Encoding::kU8, .required = true},
    PartSpec{.role = PartRole::kBlockScales, .element = Encoding::kE8M0, .required = true},
};
constexpr std::array kMxfp8Parts = {
    PartSpec{.role = PartRole::kCodes, .element = Encoding::kF8E4M3, .required = true},
    PartSpec{.role = PartRole::kBlockScales, .element = Encoding::kE8M0, .required = true},
};
constexpr std::array kExl3Parts = {
    PartSpec{.role = PartRole::kCodes, .element = Encoding::kI16, .required = true},
    PartSpec{.role = PartRole::kInputSigns, .element = Encoding::kF16, .required = true},
    PartSpec{.role = PartRole::kOutputSigns, .element = Encoding::kF16, .required = true},
    PartSpec{.role = PartRole::kCodebookMarker, .element = Encoding::kI32, .required = false},
};
constexpr std::array kAffineParts = {
    PartSpec{.role = PartRole::kCodes, .element = Encoding::kU8, .required = true},
    PartSpec{.role = PartRole::kBlockScales, .element = Encoding::kF16, .required = true},
    PartSpec{.role = PartRole::kBiases, .element = Encoding::kF16, .required = false},
    PartSpec{.role = PartRole::kZeros, .element = Encoding::kU8, .required = false},
};
constexpr std::array kFp8ScaledParts = {
    PartSpec{.role = PartRole::kCodes, .element = Encoding::kF8E4M3, .required = true},
    PartSpec{.role = PartRole::kBlockScales, .element = Encoding::kF32, .required = true},
};

constexpr EncodingInfo Dense(Encoding id, std::string_view name, std::uint32_t bits, bool floating,
                             std::optional<std::uint32_t> ggml = std::nullopt) {
  EncodingInfo info{.id = id,
                    .name = name,
                    .block_elements = 1,
                    .element_bits = bits,
                    .quantized = false,
                    .floating = floating,
                    .parts = {},
                    .ggml = std::nullopt};
  if (ggml) {
    info.ggml = GgmlBlock{.type = *ggml, .block_elements = 1, .block_bytes = bits / 8};
  }
  return info;
}

constexpr EncodingInfo Ggml(Encoding id, std::string_view name, std::uint32_t type,
                            std::uint32_t block_elements, std::uint32_t block_bytes) {
  return {.id = id,
          .name = name,
          .block_elements = block_elements,
          .element_bits = 0,
          .quantized = true,
          .floating = false,
          .parts = {},
          .ggml = GgmlBlock{
              .type = type, .block_elements = block_elements, .block_bytes = block_bytes}};
}

// Enum order; block sizes as ggml-common.h at the pin, and as
// artifact/representation.cc's table (which the tests compare) but for
// Q8_1: v0's table records gguf-py's 40 bytes, GGML's block_q8_1 is 36.
constexpr std::array kEncodings = {
    Dense(Encoding::kF32, "f32", 32, true, 0),
    Dense(Encoding::kF16, "f16", 16, true, 1),
    Dense(Encoding::kBF16, "bf16", 16, true, 30),
    Dense(Encoding::kF64, "f64", 64, true, 28),
    Dense(Encoding::kI8, "i8", 8, false, 24),
    Dense(Encoding::kI16, "i16", 16, false, 25),
    Dense(Encoding::kI32, "i32", 32, false, 26),
    Dense(Encoding::kI64, "i64", 64, false, 27),
    Dense(Encoding::kU8, "u8", 8, false),
    Dense(Encoding::kU16, "u16", 16, false),
    Dense(Encoding::kU32, "u32", 32, false),
    Dense(Encoding::kU64, "u64", 64, false),
    Dense(Encoding::kBool, "bool", 8, false),
    Dense(Encoding::kF8E4M3, "f8_e4m3", 8, true),
    Dense(Encoding::kF8E5M2, "f8_e5m2", 8, true),
    Dense(Encoding::kE8M0, "e8m0", 8, true),
    Ggml(Encoding::kQ1_0, "q1_0", 41, 128, 18),
    Ggml(Encoding::kQ2_0, "q2_0", 42, 64, 18),
    Ggml(Encoding::kQ4_0, "q4_0", 2, 32, 18),
    Ggml(Encoding::kQ4_1, "q4_1", 3, 32, 20),
    Ggml(Encoding::kQ5_0, "q5_0", 6, 32, 22),
    Ggml(Encoding::kQ5_1, "q5_1", 7, 32, 24),
    Ggml(Encoding::kQ8_0, "q8_0", 8, 32, 34),
    Ggml(Encoding::kQ8_1, "q8_1", 9, 32, 36),
    Ggml(Encoding::kQ2_K, "q2_k", 10, kQk, 2 + 2 + (kQk / 16) + (kQk / 4)),
    Ggml(Encoding::kQ3_K, "q3_k", 11, kQk, 2 + (kQk / 4) + (kQk / 8) + 12),
    Ggml(Encoding::kQ4_K, "q4_k", 12, kQk, 2 + 2 + (kQk / 2) + 12),
    Ggml(Encoding::kQ5_K, "q5_k", 13, kQk, 2 + 2 + (kQk / 2) + (kQk / 8) + 12),
    Ggml(Encoding::kQ6_K, "q6_k", 14, kQk, 2 + (kQk / 2) + (kQk / 4) + (kQk / 16)),
    Ggml(Encoding::kQ8_K, "q8_k", 15, kQk, 4 + kQk + (kQk / 8)),
    Ggml(Encoding::kIQ1_S, "iq1_s", 19, kQk, 2 + (kQk / 8) + (kQk / 16)),
    Ggml(Encoding::kIQ1_M, "iq1_m", 29, kQk, (kQk / 8) + (kQk / 16) + (kQk / 32)),
    Ggml(Encoding::kIQ2_XXS, "iq2_xxs", 16, kQk, 2 + (kQk / 4)),
    Ggml(Encoding::kIQ2_XS, "iq2_xs", 17, kQk, 2 + (kQk / 4) + (kQk / 32)),
    Ggml(Encoding::kIQ2_S, "iq2_s", 22, kQk, 2 + (kQk / 4) + (kQk / 16)),
    Ggml(Encoding::kIQ3_XXS, "iq3_xxs", 18, kQk, 2 + (kQk / 4) + (kQk / 8)),
    Ggml(Encoding::kIQ3_S, "iq3_s", 21, kQk, 2 + (kQk / 4) + (kQk / 8) + (kQk / 32) + 4),
    Ggml(Encoding::kIQ4_NL, "iq4_nl", 20, 32, 18),
    Ggml(Encoding::kIQ4_XS, "iq4_xs", 23, kQk, 2 + 2 + (kQk / 2) + (kQk / 64)),
    Ggml(Encoding::kTQ1_0, "tq1_0", 34, kQk, 2 + (4 * 13)),
    Ggml(Encoding::kTQ2_0, "tq2_0", 35, kQk, 2 + 64),
    // GGML's block_mxfp4 (an E8M0 byte, then 16 code bytes, byte j holding
    // values j and j + 16) is one layout of MXFP4.
    EncodingInfo{.id = Encoding::kMxfp4,
                 .name = "mxfp4",
                 .block_elements = 32,
                 .element_bits = 4,
                 .quantized = true,
                 .floating = false,
                 .parts = kMxfp4Parts,
                 .ggml = GgmlBlock{.type = 39, .block_elements = 32, .block_bytes = 17}},
    EncodingInfo{.id = Encoding::kMxfp8,
                 .name = "mxfp8",
                 .block_elements = 32,
                 .element_bits = 8,
                 .quantized = true,
                 .floating = false,
                 .parts = kMxfp8Parts,
                 .ggml = std::nullopt},
    // GGML's block_nvfp4 (64 values: 4 scales, then 32 code bytes) is one
    // layout of NVFP4 (layout.h kGgmlNvfp4Blocks).
    EncodingInfo{.id = Encoding::kNvfp4,
                 .name = "nvfp4",
                 .block_elements = 16,
                 .element_bits = 4,
                 .quantized = true,
                 .floating = false,
                 .parts = kNvfp4Parts,
                 .ggml = GgmlBlock{.type = 40, .block_elements = 64, .block_bytes = 36}},
    EncodingInfo{.id = Encoding::kExl3Trellis,
                 .name = "exl3_trellis",
                 .block_elements = 16,
                 .element_bits = 0,
                 .quantized = true,
                 .floating = false,
                 .parts = kExl3Parts,
                 .ggml = std::nullopt},
    EncodingInfo{.id = Encoding::kAffine,
                 .name = "affine",
                 .block_elements = 8,
                 .element_bits = 0,
                 .quantized = true,
                 .floating = false,
                 .parts = kAffineParts,
                 .ggml = std::nullopt},
    EncodingInfo{.id = Encoding::kFp8Scaled,
                 .name = "fp8_scaled",
                 .block_elements = 1,
                 .element_bits = 8,
                 .quantized = true,
                 .floating = false,
                 .parts = kFp8ScaledParts,
                 .ggml = std::nullopt},
};

constexpr bool InEnumOrder() {
  for (std::size_t i = 0; i < kEncodings.size(); ++i) {
    if (static_cast<std::size_t>(kEncodings[i].id) != i) {
      return false;
    }
  }
  return kEncodings.back().id == Encoding::kFp8Scaled;
}
static_assert(InEnumOrder(), "the table lists every encoding in enum order");

// A dense encoding's exact value set, for Lossless: a float's significand
// bits (the implicit one included), largest exponent and smallest positive
// value's exponent (its last subnormal, or its least normal where it has
// none), and its specials; an integer's magnitude bits and sign.
struct ValueSet {
  enum class Kind : std::uint8_t { kFloat, kInteger, kOther } kind = Kind::kOther;
  int precision = 0;
  int max_exponent = 0;
  int min_exponent = 0;  // of the smallest positive value
  bool has_zero = true;
  bool has_negative = true;
  bool has_infinity = false;
  bool has_nan = false;
};

constexpr ValueSet Float(int precision, int max_exponent, int min_exponent, bool infinity,
                         bool nan) {
  return {.kind = ValueSet::Kind::kFloat,
          .precision = precision,
          .max_exponent = max_exponent,
          .min_exponent = min_exponent,
          .has_zero = true,
          .has_negative = true,
          .has_infinity = infinity,
          .has_nan = nan};
}

// The largest exponent is a signed type's most negative value, -2^n, or
// an unsigned type's top bit.
constexpr ValueSet Integer(int magnitude_bits, bool negative) {
  return {.kind = ValueSet::Kind::kInteger,
          .precision = magnitude_bits,
          .max_exponent = negative ? magnitude_bits : magnitude_bits - 1,
          .min_exponent = 0,
          .has_zero = true,
          .has_negative = negative,
          .has_infinity = false,
          .has_nan = false};
}

ValueSet ValuesOf(Encoding e) {
  switch (e) {
    case Encoding::kF64:
      return Float(53, 1023, -1074, true, true);
    case Encoding::kF32:
      return Float(24, 127, -149, true, true);
    case Encoding::kF16:
      return Float(11, 15, -24, true, true);
    case Encoding::kBF16:
      return Float(8, 127, -133, true, true);
    case Encoding::kF8E4M3:
      return Float(4, 8, -9, false, true);
    case Encoding::kF8E5M2:
      return Float(3, 15, -16, true, true);
    case Encoding::kE8M0: {
      ValueSet v = Float(1, 127, -127, false, true);
      v.has_zero = false;
      v.has_negative = false;
      return v;
    }
    case Encoding::kI8:
      return Integer(7, true);
    case Encoding::kI16:
      return Integer(15, true);
    case Encoding::kI32:
      return Integer(31, true);
    case Encoding::kI64:
      return Integer(63, true);
    case Encoding::kU8:
      return Integer(8, false);
    case Encoding::kU16:
      return Integer(16, false);
    case Encoding::kU32:
      return Integer(32, false);
    case Encoding::kU64:
      return Integer(64, false);
    case Encoding::kBool:
      return Integer(1, false);
    default:
      return {};
  }
}

}  // namespace

std::string_view PartRoleName(PartRole role) {
  switch (role) {
    case PartRole::kData:
      return "data";
    case PartRole::kCodes:
      return "codes";
    case PartRole::kBlockScales:
      return "block_scales";
    case PartRole::kTensorScale:
      return "tensor_scale";
    case PartRole::kInputSigns:
      return "input_signs";
    case PartRole::kOutputSigns:
      return "output_signs";
    case PartRole::kCodebookMarker:
      return "codebook_marker";
    case PartRole::kBiases:
      return "biases";
    case PartRole::kZeros:
      return "zeros";
  }
  return "unknown";
}

Encoding AffineScaleEncoding(AffineScale scale) {
  switch (scale) {
    case AffineScale::kF16:
      return Encoding::kF16;
    case AffineScale::kBF16:
      return Encoding::kBF16;
    case AffineScale::kNone:
    case AffineScale::kF32:
      break;
  }
  return Encoding::kF32;
}

std::span<const EncodingInfo> Encodings() { return kEncodings; }

const EncodingInfo& Info(Encoding encoding) {
  const auto index = static_cast<std::size_t>(encoding);
  base::Check(index < kEncodings.size(), "format::Info of an encoding outside the enum");
  return kEncodings[index];
}

std::optional<Encoding> EncodingByName(std::string_view name) {
  const auto* found =
      std::ranges::find_if(kEncodings, [&](const EncodingInfo& e) { return e.name == name; });
  if (found == kEncodings.end()) {
    return std::nullopt;
  }
  return found->id;
}

namespace {

bool EqualsIgnoreCase(std::string_view a, std::string_view b) {
  if (a.size() != b.size()) {
    return false;
  }
  for (std::size_t i = 0; i < a.size(); ++i) {
    const auto ca = static_cast<unsigned char>(a[i]);
    const auto cb = static_cast<unsigned char>(b[i]);
    if (std::tolower(ca) != std::tolower(cb)) {
      return false;
    }
  }
  return true;
}

}  // namespace

std::optional<std::uint32_t> BlockBytes(Encoding encoding) {
  const EncodingInfo& info = Info(encoding);
  if (info.ggml.has_value()) {
    return info.ggml->block_bytes;
  }
  if (info.parts.empty() && info.block_elements == 1 && info.element_bits != 0 &&
      info.element_bits % 8 == 0) {
    return info.element_bits / 8;
  }
  return std::nullopt;
}

std::optional<BlockStruct> BlockStructOf(Encoding encoding) {
  const EncodingInfo& info = Info(encoding);
  if (!info.ggml.has_value()) return std::nullopt;
  return BlockStruct{.elements = info.ggml->block_elements, .bytes = info.ggml->block_bytes};
}

std::optional<Encoding> EncodingByTypeName(std::string_view name) {
  for (const EncodingInfo& info : Encodings()) {
    if (EqualsIgnoreCase(name, info.name)) {
      return info.id;
    }
  }
  return std::nullopt;
}

namespace {

constexpr std::array<std::pair<std::string_view, Encoding>, 34> kGgmlNames = {{
    {"F32", Encoding::kF32},         {"F16", Encoding::kF16},         {"Q4_0", Encoding::kQ4_0},
    {"Q4_1", Encoding::kQ4_1},       {"Q5_0", Encoding::kQ5_0},       {"Q5_1", Encoding::kQ5_1},
    {"Q8_0", Encoding::kQ8_0},       {"Q2_K", Encoding::kQ2_K},       {"Q3_K", Encoding::kQ3_K},
    {"Q4_K", Encoding::kQ4_K},       {"Q5_K", Encoding::kQ5_K},       {"Q6_K", Encoding::kQ6_K},
    {"Q8_K", Encoding::kQ8_K},       {"IQ2_XXS", Encoding::kIQ2_XXS}, {"IQ2_XS", Encoding::kIQ2_XS},
    {"IQ3_XXS", Encoding::kIQ3_XXS}, {"IQ1_S", Encoding::kIQ1_S},     {"IQ4_NL", Encoding::kIQ4_NL},
    {"IQ3_S", Encoding::kIQ3_S},     {"IQ2_S", Encoding::kIQ2_S},     {"IQ4_XS", Encoding::kIQ4_XS},
    {"I8", Encoding::kI8},           {"I16", Encoding::kI16},         {"I32", Encoding::kI32},
    {"I64", Encoding::kI64},         {"F64", Encoding::kF64},         {"IQ1_M", Encoding::kIQ1_M},
    {"BF16", Encoding::kBF16},       {"TQ1_0", Encoding::kTQ1_0},     {"TQ2_0", Encoding::kTQ2_0},
    {"MXFP4", Encoding::kMxfp4},     {"NVFP4", Encoding::kNvfp4},     {"Q1_0", Encoding::kQ1_0},
    {"Q2_0", Encoding::kQ2_0},
}};

constexpr std::array<std::pair<std::string_view, Encoding>, 15> kDtypeNames = {{
    {"BOOL", Encoding::kBool},
    {"U8", Encoding::kU8},
    {"I8", Encoding::kI8},
    {"F8_E5M2", Encoding::kF8E5M2},
    {"F8_E4M3", Encoding::kF8E4M3},
    {"I16", Encoding::kI16},
    {"U16", Encoding::kU16},
    {"F16", Encoding::kF16},
    {"BF16", Encoding::kBF16},
    {"I32", Encoding::kI32},
    {"U32", Encoding::kU32},
    {"F32", Encoding::kF32},
    {"I64", Encoding::kI64},
    {"U64", Encoding::kU64},
    {"F64", Encoding::kF64},
}};

std::optional<Encoding> LookupName(std::span<const std::pair<std::string_view, Encoding>> table,
                                   std::string_view name) {
  const auto found = std::ranges::find_if(table, [&](const auto& e) { return e.first == name; });
  if (found == table.end()) {
    return std::nullopt;
  }
  return found->second;
}

}  // namespace

std::optional<Encoding> EncodingOfGgmlType(std::string_view type) {
  return LookupName(kGgmlNames, type);
}

std::optional<std::string_view> GgmlTypeName(std::uint32_t type_id) {
  if (type_id == 9U) {
    return "Q8_1";
  }
  for (const auto& [name, enc] : kGgmlNames) {
    const EncodingInfo& info = Info(enc);
    if (info.ggml && info.ggml->type == type_id) {
      return name;
    }
  }
  return std::nullopt;
}

std::optional<Encoding> EncodingOfDtype(std::string_view dtype) {
  return LookupName(kDtypeNames, dtype);
}

std::optional<std::string_view> DtypeOfEncoding(Encoding encoding) {
  const auto* found =
      std::ranges::find_if(kDtypeNames, [&](const auto& e) { return e.second == encoding; });
  if (found == kDtypeNames.end()) {
    return std::nullopt;
  }
  return found->first;
}

bool Lossless(Encoding from, Encoding to) {
  if (from == to) {
    return true;
  }
  const ValueSet a = ValuesOf(from);
  const ValueSet b = ValuesOf(to);
  if (a.kind == ValueSet::Kind::kOther || b.kind == ValueSet::Kind::kOther) {
    return false;  // quantized: exactness is not established
  }
  if ((a.has_negative && !b.has_negative) || (a.has_zero && !b.has_zero) ||
      (a.has_infinity && !b.has_infinity) || (a.has_nan && !b.has_nan)) {
    return false;
  }
  if (b.kind == ValueSet::Kind::kInteger) {
    // Only integers fit integers: magnitudes, and a signed source's most
    // negative value (a power of two) in a signed target of more bits.
    return a.kind == ValueSet::Kind::kInteger && a.precision <= b.precision;
  }
  // Into a float: every value's significand fits, its exponent is in
  // range, and its last bit is no finer than the target's smallest value
  // (an integer's last bit is 1).
  if (a.precision > b.precision || a.max_exponent > b.max_exponent) {
    return false;
  }
  return a.min_exponent >= b.min_exponent;
}

std::span<const Encoding> StateEncodings() { return kStateEncodings; }

bool IsStateEncoding(Encoding encoding) {
  return std::ranges::find(kStateEncodings, encoding) != kStateEncodings.end();
}

ScaleCode ScaleCodeOf(Encoding encoding) {
  switch (encoding) {
    case Encoding::kNvfp4:
      return ScaleCode::kUe4m3;
    case Encoding::kMxfp4:
    case Encoding::kMxfp8:
      return ScaleCode::kE8m0;
    default:
      return ScaleCode::kNone;
  }
}

}  // namespace llmp::format
