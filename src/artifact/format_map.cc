// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "artifact/format_map.h"

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

#include "artifact/representation.h"
#include "format/encoding.h"
#include "format/layout.h"
#include "format/tensor_format.h"

namespace llmp::artifact {
namespace {

using format::Encoding;
using format::FormatError;
using format::Layout;
using format::PartRole;
using format::TensorFormat;
using Result = std::expected<TensorFormat, FormatError>;

constexpr std::uint64_t kSaturated = std::numeric_limits<std::uint64_t>::max();

// GGML type names (representation.cc's table) and their encodings; not
// Q8_1, whose size v0's table records as gguf-py's 40 bytes where GGML's
// block at the pin is 36 (an activation type no artifact stores).
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

std::optional<Encoding> Lookup(std::span<const std::pair<std::string_view, Encoding>> table,
                               std::string_view name) {
  const auto found = std::ranges::find_if(table, [&](const auto& e) { return e.first == name; });
  if (found == table.end()) {
    return std::nullopt;
  }
  return found->second;
}

Result Checked(const TensorFormat& f) {
  if (auto ok = format::Validate(f); !ok) {
    return std::unexpected(ok.error());
  }
  return f;
}

Result GgmlFormat(const Representation& rep) {
  const auto encoding = EncodingOfGgmlType(rep.type);
  const auto shape = format::ShapeOf(rep.dims);
  if (!encoding || !shape) {
    return std::unexpected(!encoding ? FormatError::kUnknown : FormatError::kShape);
  }
  TensorFormat f{.encoding = *encoding, .shape = *shape};
  const format::EncodingInfo& info = format::Info(*encoding);
  if (!info.quantized) {
    f.layout = Layout::kDense;
  } else if (*encoding == Encoding::kNvfp4) {
    f.layout = Layout::kGgmlNvfp4Blocks;
  } else {
    f.layout = Layout::kGgmlBlocks;
  }
  return Checked(f);
}

// A container shape (outermost first) in GGML's order, its outer
// dimensions past the fourth folded into ne[3].
Result DenseFormat(const Representation& rep) {
  const auto encoding = EncodingOfDtype(rep.type);
  if (!encoding) {
    return std::unexpected(FormatError::kUnknown);
  }
  TensorFormat f{.encoding = *encoding, .layout = Layout::kDense};
  const std::size_t rank = rep.dims.size();
  f.shape.rank = static_cast<std::uint8_t>(std::min(rank, format::kMaxDims));
  for (std::size_t i = 0; i < rank; ++i) {
    const std::uint64_t d = rep.dims[rank - 1 - i];
    std::uint64_t& ne = f.shape.ne.at(std::min(i, format::kMaxDims - 1));
    if (i < format::kMaxDims) {
      ne = d;
    } else if (__builtin_mul_overflow(ne, d, &ne)) {
      return std::unexpected(FormatError::kOverflow);
    }
  }
  return Checked(f);
}

Result TrellisFormat(const Representation& rep) {
  if (rep.in_features == kSaturated || rep.out_features == kSaturated) {
    return std::unexpected(FormatError::kOverflow);
  }
  format::Exl3Codebook codebook = format::Exl3Codebook::kNone;
  if (rep.codebook == "mcg") {
    codebook = format::Exl3Codebook::kMcg;
  }
  return Checked({.encoding = Encoding::kExl3Trellis,
                  .params = {.k_bits = static_cast<std::uint8_t>(rep.k_bits),
                             .codebook = codebook,
                             .tensor_scale = false},
                  .layout = Layout::kExl3Tiles,
                  .shape = {.ne = {rep.in_features, rep.out_features, 1, 1}, .rank = 2},
                  .stacked = 1});
}

// A resource of one-byte elements as rows: its 2-D row geometry.
std::optional<RowGeometry> ByteRows(const Representation& rep) {
  const auto f = FormatOf(rep);
  if (!f || f->layout != Layout::kDense || format::Info(f->encoding).element_bits != 8 ||
      f->shape.rank != 2) {
    return std::nullopt;
  }
  return RowGeometryOf(rep);
}

const Representation* Find(std::span<const V0Part> parts, PartRole role) {
  const auto found = std::ranges::find_if(parts, [&](const V0Part& p) { return p.role == role; });
  return found == parts.end() ? nullptr : found->rep;
}

// The tensor's format from its parts, before the bytes are compared.
Result Shaped(Encoding encoding, Layout layout, std::span<const V0Part> parts,
              std::uint8_t stacked) {
  const format::EncodingInfo& info = format::Info(encoding);
  TensorFormat f{.encoding = encoding, .layout = layout, .stacked = stacked};
  f.params.tensor_scale = Find(parts, PartRole::kTensorScale) != nullptr;
  switch (layout) {
    case Layout::kSplit:
    case Layout::kScaleAtoms:
    case Layout::kRowInterleaved: {
      if (info.element_bits == 0 || !info.quantized) {
        return std::unexpected(FormatError::kLayout);  // not block-scaled codes
      }
      const PartRole role = layout == Layout::kRowInterleaved ? PartRole::kData : PartRole::kCodes;
      const Representation* rep = Find(parts, role);
      const auto rows = rep != nullptr ? ByteRows(*rep) : std::nullopt;
      if (!rows) {
        return std::unexpected(FormatError::kParts);
      }
      // A row's bytes: K · bits / 8 codes, and interleaved, K / block scales.
      const std::uint64_t per_block = (info.block_elements * info.element_bits / 8) +
                                      (layout == Layout::kRowInterleaved ? 1 : 0);
      if (rows->row_bytes % per_block != 0) {
        return std::unexpected(FormatError::kBlock);
      }
      f.shape = {.ne = {rows->row_bytes / per_block * info.block_elements, rows->rows, 1, 1},
                 .rank = 2};
      return f;
    }
    case Layout::kExl3Tiles: {
      const Representation* trellis = Find(parts, PartRole::kCodes);
      if (trellis == nullptr || trellis->family != Family::kExl3 || trellis->role != "trellis") {
        return std::unexpected(FormatError::kParts);
      }
      return TrellisFormat(*trellis);
    }
    case Layout::kDense:
    case Layout::kGgmlBlocks:
    case Layout::kGgmlNvfp4Blocks: {
      const Representation* rep = Find(parts, PartRole::kData);
      if (rep == nullptr) {
        return std::unexpected(FormatError::kParts);
      }
      auto own = FormatOf(*rep);
      if (!own) {
        return own;
      }
      if (own->encoding != encoding || own->layout != layout) {
        return std::unexpected(FormatError::kLayout);
      }
      f.shape = own->shape;
      return f;
    }
  }
  return std::unexpected(FormatError::kUnknown);
}

}  // namespace

std::optional<format::Encoding> EncodingOfGgmlType(std::string_view type) {
  return Lookup(kGgmlNames, type);
}

std::optional<format::Encoding> EncodingOfDtype(std::string_view dtype) {
  return Lookup(kDtypeNames, dtype);
}

std::optional<std::string_view> DtypeOfEncoding(format::Encoding encoding) {
  const auto found =
      std::ranges::find_if(kDtypeNames, [&](const auto& e) { return e.second == encoding; });
  if (found == kDtypeNames.end()) {
    return std::nullopt;
  }
  return found->first;
}

std::expected<format::TensorFormat, format::FormatError> FormatOf(const Representation& rep) {
  switch (rep.family) {
    case Family::kGgml:
      return GgmlFormat(rep);
    case Family::kExl3:
      if (rep.role == "trellis") {
        return TrellisFormat(rep);
      }
      return DenseFormat(rep);
    case Family::kPlain:
      return DenseFormat(rep);
  }
  return std::unexpected(FormatError::kUnknown);
}

format::PartRole V0PartRole(const Representation& rep) {
  if (rep.family != Family::kExl3) {
    return PartRole::kData;
  }
  if (rep.role == "trellis") {
    return PartRole::kCodes;
  }
  if (rep.role == "suh") {
    return PartRole::kInputSigns;
  }
  if (rep.role == "svh") {
    return PartRole::kOutputSigns;
  }
  return PartRole::kCodebookMarker;
}

std::expected<format::TensorFormat, format::FormatError> Assemble(format::Encoding encoding,
                                                                  format::Layout layout,
                                                                  std::span<const V0Part> parts,
                                                                  std::uint8_t stacked) {
  if (static_cast<std::size_t>(encoding) >= format::Encodings().size() ||
      static_cast<std::size_t>(layout) >= format::Layouts().size()) {
    return std::unexpected(FormatError::kUnknown);
  }
  for (std::size_t i = 0; i < parts.size(); ++i) {
    if (parts[i].rep == nullptr) {
      return std::unexpected(FormatError::kParts);
    }
    for (std::size_t j = i + 1; j < parts.size(); ++j) {
      if (parts[i].role == parts[j].role) {
        return std::unexpected(FormatError::kParts);
      }
    }
  }
  const auto f = Shaped(encoding, layout, parts, stacked);
  if (!f) {
    return f;
  }
  const auto extents = format::PartsOf(*f);
  if (!extents) {
    return std::unexpected(extents.error());
  }
  if (extents->size() != parts.size()) {
    return std::unexpected(FormatError::kParts);
  }
  for (const format::PartExtent& extent : extents->parts()) {
    const Representation* rep = Find(parts, extent.role);
    if (rep == nullptr || rep->bytes != extent.bytes) {
      return std::unexpected(FormatError::kParts);
    }
    if (extent.role == PartRole::kTensorScale && EncodingOfDtype(rep->type) != Encoding::kF32) {
      return std::unexpected(FormatError::kParts);
    }
  }
  return f;
}

}  // namespace llmp::artifact
