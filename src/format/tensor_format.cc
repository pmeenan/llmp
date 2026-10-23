// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "format/tensor_format.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <optional>
#include <span>
#include <string_view>

#include "base/check.h"
#include "format/encoding.h"
#include "format/layout.h"

namespace llmp::format {
namespace {

using Size = std::expected<std::uint64_t, FormatError>;

Size Mul(std::uint64_t a, std::uint64_t b) {
  std::uint64_t out = 0;
  if (__builtin_mul_overflow(a, b, &out)) {
    return std::unexpected(FormatError::kOverflow);
  }
  return out;
}

Size Add(std::uint64_t a, std::uint64_t b) {
  std::uint64_t out = 0;
  if (__builtin_add_overflow(a, b, &out)) {
    return std::unexpected(FormatError::kOverflow);
  }
  return out;
}

bool KnownEncoding(Encoding e) { return static_cast<std::size_t>(e) < Encodings().size(); }

bool KnownLayout(Layout l) { return static_cast<std::size_t>(l) < Layouts().size(); }

bool HasPart(const EncodingInfo& info, PartRole role) {
  return std::ranges::any_of(info.parts, [&](const PartSpec& p) { return p.role == role; });
}

// MXFP4, MXFP8 and NVFP4: uniform codes with a one-byte scale per block.
bool BlockScaled(const EncodingInfo& info) {
  const auto scale = std::ranges::find_if(
      info.parts, [](const PartSpec& p) { return p.role == PartRole::kBlockScales; });
  return info.element_bits > 0 && scale != info.parts.end() &&
         Info(scale->element).element_bits == 8;
}

std::expected<void, FormatError> CheckShape(const TensorFormat& f) {
  const Shape& s = f.shape;
  if (s.rank > kMaxDims) {
    return std::unexpected(FormatError::kShape);
  }
  for (std::size_t i = 0; i < kMaxDims; ++i) {
    if (s.ne.at(i) == 0 || (i >= s.rank && s.ne.at(i) != 1)) {
      return std::unexpected(FormatError::kShape);
    }
  }
  if (f.stacked == 0 || s.ne[1] % f.stacked != 0) {
    return std::unexpected(FormatError::kShape);
  }
  return {};
}

std::expected<void, FormatError> CheckParams(const TensorFormat& f, const EncodingInfo& info) {
  const EncodingParams& p = f.params;
  const bool exl3 = f.encoding == Encoding::kExl3Trellis;
  if (exl3 != (p.k_bits != 0) || exl3 != (p.codebook != Exl3Codebook::kNone) || p.k_bits > 8 ||
      static_cast<std::uint8_t>(p.codebook) > static_cast<std::uint8_t>(Exl3Codebook::k3Inst)) {
    return std::unexpected(FormatError::kParams);
  }
  if (p.tensor_scale && !HasPart(info, PartRole::kTensorScale)) {
    return std::unexpected(FormatError::kParams);
  }
  const bool fp8_scaled = f.encoding == Encoding::kFp8Scaled;
  if (fp8_scaled != (p.block_rows != 0 && p.block_cols != 0) ||
      (!fp8_scaled && (p.block_rows != 0 || p.block_cols != 0))) {
    return std::unexpected(FormatError::kParams);
  }
  if (fp8_scaled && (f.shape.rank < 2 || f.shape.rank > 4 || f.layout != Layout::kSplit)) {
    return std::unexpected(FormatError::kLayout);
  }
  const bool affine = f.encoding == Encoding::kAffine;
  if (!affine) {
    if (p.bits != 0 || p.group != 0 || p.scale != AffineScale::kNone || p.zero_point) {
      return std::unexpected(FormatError::kParams);
    }
    return {};
  }
  // Groups of whole bytes of codes at every width, and whole groups a row.
  if (p.bits == 0 || p.bits > 8 || p.group == 0 || p.group % 8 != 0 ||
      p.scale == AffineScale::kNone ||
      static_cast<std::uint8_t>(p.scale) > static_cast<std::uint8_t>(AffineScale::kF32)) {
    return std::unexpected(FormatError::kParams);
  }
  return {};
}

// The affine encoding's parts of one row: K · bits / 8 code bytes, K /
// group scales (and biases) of the scale type, and K / group zero points
// of `bits` bits, padded to a whole byte.
struct AffineRow {
  std::uint64_t codes = 0;
  std::uint64_t scales = 0;
  std::uint64_t zeros = 0;
};

AffineRow AffineRowOf(const TensorFormat& f) {
  const EncodingParams& p = f.params;
  const std::uint64_t k = f.shape.ne[0];
  const std::uint64_t groups = k / p.group;
  const std::uint64_t scale_bytes = Info(AffineScaleEncoding(p.scale)).element_bits / 8;
  // With bits at most 8 and groups of at least 8 (K a multiple of 8), each
  // is at most K, which cannot wrap.
  return {.codes = k / 8 * p.bits,
          .scales = groups * scale_bytes,
          .zeros = ((groups * p.bits) + 7) / 8};
}

// Whether the layout holds the encoding.
bool Holds(Layout layout, const EncodingInfo& info) {
  switch (layout) {
    case Layout::kDense:
      return !info.quantized && info.block_elements == 1;
    case Layout::kGgmlBlocks:
      return info.quantized && info.ggml.has_value() && info.id != Encoding::kNvfp4;
    case Layout::kGgmlNvfp4Blocks:
      return info.id == Encoding::kNvfp4;
    case Layout::kSplit:
      return BlockScaled(info) || info.id == Encoding::kAffine || info.id == Encoding::kFp8Scaled;
    case Layout::kRowInterleaved:
    case Layout::kScaleAtoms:
      return BlockScaled(info);
    case Layout::kExl3Tiles:
      return info.id == Encoding::kExl3Trellis;
  }
  return false;
}

// For a layout known to hold the encoding. This is also the geometry used
// by the size rule, so neutral consumers cannot drift from stored bytes.
std::optional<AddressUnit> UnitOf(Layout layout, const EncodingInfo& info) {
  switch (layout) {
    case Layout::kDense:
      return AddressUnit{.elements = 1, .bytes = info.element_bits / 8};
    case Layout::kGgmlBlocks:
    case Layout::kGgmlNvfp4Blocks:
      return AddressUnit{.elements = info.ggml->block_elements, .bytes = info.ggml->block_bytes};
    default:
      return std::nullopt;
  }
}

// Bytes a row's codes take (K a multiple of the block, so of 2 for 4-bit
// codes).
std::uint64_t CodeRowBytes(std::uint64_t k, const EncodingInfo& info) {
  return k / 8 * info.element_bits + (k % 8) * info.element_bits / 8;
}

// The parts of one matrix (readable equal to bytes); the caller scales
// them by the matrices and adds the over-read.
std::expected<PartList, FormatError> MatrixParts(const TensorFormat& f, const EncodingInfo& info) {
  const std::uint64_t k = f.shape.ne[0];
  const std::uint64_t n = f.shape.ne[1];
  PartList parts;
  auto add = [&](PartRole role, Size bytes) -> std::expected<void, FormatError> {
    if (!bytes) {
      return std::unexpected(bytes.error());
    }
    parts.Add({.role = role, .bytes = *bytes, .readable = *bytes});
    return {};
  };
  std::expected<void, FormatError> ok;
  switch (f.layout) {
    case Layout::kDense:
    case Layout::kGgmlBlocks:
    case Layout::kGgmlNvfp4Blocks: {
      const AddressUnit unit = *UnitOf(f.layout, info);
      if (k % unit.elements != 0) {
        return std::unexpected(FormatError::kBlock);
      }
      const Size row = Mul(k / unit.elements, unit.bytes);
      ok = add(PartRole::kData, row ? Mul(*row, n) : row);
      break;
    }
    case Layout::kSplit:
    case Layout::kRowInterleaved:
    case Layout::kScaleAtoms: {
      if (f.encoding == Encoding::kAffine) {  // kSplit only (Holds)
        if (k % f.params.group != 0) {
          return std::unexpected(FormatError::kBlock);
        }
        const AffineRow row = AffineRowOf(f);
        ok = add(PartRole::kCodes, Mul(row.codes, n));
        if (ok) {
          ok = add(PartRole::kBlockScales, Mul(row.scales, n));
        }
        if (ok) {
          ok = f.params.zero_point ? add(PartRole::kZeros, Mul(row.zeros, n))
                                   : add(PartRole::kBiases, Mul(row.scales, n));
        }
        break;
      }
      if (f.encoding == Encoding::kFp8Scaled) {
        auto grid = Fp8ScaleShapeOf(f);
        if (!grid) return std::unexpected(grid.error());
        Size codes = Mul(k, n);
        Size scales = Mul(grid->ne[0], grid->ne[1]);
        if (scales) scales = Mul(*scales, 4);
        ok = add(PartRole::kCodes, codes);
        if (ok) ok = add(PartRole::kBlockScales, scales);
        break;
      }
      if (k % info.block_elements != 0) {
        return std::unexpected(FormatError::kBlock);
      }
      const std::uint64_t blocks = k / info.block_elements;
      const std::uint64_t codes = CodeRowBytes(k, info);
      if (f.layout == Layout::kRowInterleaved) {
        const Size row = Add(codes, blocks);
        ok = add(PartRole::kData, row ? Mul(*row, n) : row);
        break;
      }
      ok = add(PartRole::kCodes, Mul(codes, n));
      if (!ok) {
        break;
      }
      if (f.layout == Layout::kSplit) {
        ok = add(PartRole::kBlockScales, Mul(blocks, n));
        break;
      }
      if (blocks % kScaleAtomBlocks != 0) {
        return std::unexpected(FormatError::kBlock);
      }
      const std::uint64_t atoms = (n / kScaleAtomRows) + (n % kScaleAtomRows != 0 ? 1 : 0);
      const Size rows = Mul(atoms, kScaleAtomRows);
      ok = add(PartRole::kBlockScales, rows ? Mul(*rows, blocks) : rows);
      break;
    }
    case Layout::kExl3Tiles: {
      if (k % 16 != 0 || n % 16 != 0) {
        return std::unexpected(FormatError::kBlock);
      }
      // [K / 16, N / 16, 16 · k_bits] I16: K · N · k_bits / 8 bytes.
      const Size tiles = Mul(k / 16, n / 16);
      ok = add(PartRole::kCodes, tiles ? Mul(*tiles, 32U * f.params.k_bits) : tiles);
      if (ok) {
        ok = add(PartRole::kInputSigns, Mul(k, 2));
      }
      if (ok) {
        ok = add(PartRole::kOutputSigns, Mul(n, 2));
      }
      // The marker that names mcg or mul1; a 3inst trellis has none.
      if (ok && f.params.codebook != Exl3Codebook::k3Inst) {
        ok = add(PartRole::kCodebookMarker, Size(std::uint64_t{4}));
      }
      break;
    }
  }
  if (!ok) {
    return std::unexpected(ok.error());
  }
  if (f.params.tensor_scale) {
    parts.Add({.role = PartRole::kTensorScale,
               .bytes = std::uint64_t{4} * f.stacked,
               .readable = std::uint64_t{4} * f.stacked});
  }
  return parts;
}

}  // namespace

std::optional<Shape> ShapeOf(std::span<const std::uint64_t> ne) {
  if (ne.size() > kMaxDims) {
    return std::nullopt;
  }
  Shape s;
  std::ranges::copy(ne, s.ne.begin());
  s.rank = static_cast<std::uint8_t>(ne.size());
  return s;
}

std::expected<Shape, FormatError> Fp8ScaleShapeOf(const TensorFormat& f) {
  if (f.encoding != Encoding::kFp8Scaled) return std::unexpected(FormatError::kParams);
  if (auto ok = CheckShape(f); !ok) return std::unexpected(ok.error());
  if (auto ok = CheckParams(f, Info(f.encoding)); !ok) return std::unexpected(ok.error());
  if (f.layout != Layout::kSplit) return std::unexpected(FormatError::kLayout);
  const std::uint64_t rows_per_projection = f.shape.ne[1] / f.stacked;
  const std::uint64_t col_blocks =
      f.shape.ne[0] / f.params.block_cols + (f.shape.ne[0] % f.params.block_cols != 0);
  const std::uint64_t row_blocks =
      rows_per_projection / f.params.block_rows + (rows_per_projection % f.params.block_rows != 0);
  std::uint64_t stacked_rows = 0;
  if (__builtin_mul_overflow(row_blocks, static_cast<std::uint64_t>(f.stacked), &stacked_rows))
    return std::unexpected(FormatError::kOverflow);
  std::uint64_t elements = 0;
  if (__builtin_mul_overflow(col_blocks, stacked_rows, &elements) ||
      __builtin_mul_overflow(elements, f.shape.ne[2], &elements) ||
      __builtin_mul_overflow(elements, f.shape.ne[3], &elements) ||
      __builtin_mul_overflow(elements, std::uint64_t{4}, &elements))
    return std::unexpected(FormatError::kOverflow);
  return Shape{.ne = {col_blocks, stacked_rows, f.shape.ne[2], f.shape.ne[3]},
               .rank = f.shape.rank};
}

std::string_view FormatErrorName(FormatError error) {
  switch (error) {
    case FormatError::kUnknown:
      return "unknown encoding or layout";
    case FormatError::kShape:
      return "bad shape";
    case FormatError::kBlock:
      return "rows not whole blocks";
    case FormatError::kLayout:
      return "layout does not hold the encoding";
    case FormatError::kParams:
      return "bad encoding parameters";
    case FormatError::kParts:
      return "no such part";
    case FormatError::kOverflow:
      return "size beyond 64 bits";
  }
  return "unknown";
}

std::optional<PartExtent> PartList::Find(PartRole role) const {
  for (const PartExtent& p : parts()) {
    if (p.role == role) {
      return p;
    }
  }
  return std::nullopt;
}

void PartList::Add(PartExtent part) {
  base::Check(count_ < kMax, "format::PartList holds at most four parts");
  parts_.at(count_++) = part;
}

std::expected<void, FormatError> Validate(const TensorFormat& format) {
  const auto parts = PartsOf(format);
  if (!parts) {
    return std::unexpected(parts.error());
  }
  return {};
}

std::expected<PartList, FormatError> PartsOf(const TensorFormat& format) {
  if (!KnownEncoding(format.encoding) || !KnownLayout(format.layout)) {
    return std::unexpected(FormatError::kUnknown);
  }
  const EncodingInfo& info = Info(format.encoding);
  if (auto ok = CheckShape(format); !ok) {
    return std::unexpected(ok.error());
  }
  if (auto ok = CheckParams(format, info); !ok) {
    return std::unexpected(ok.error());
  }
  if (!Holds(format.layout, info)) {
    return std::unexpected(FormatError::kLayout);
  }
  const auto matrix = MatrixParts(format, info);
  if (!matrix) {
    return std::unexpected(matrix.error());
  }
  const Size matrices = Mul(format.shape.ne[2], format.shape.ne[3]);
  if (!matrices) {
    return std::unexpected(matrices.error());
  }
  PartList parts;
  for (const PartExtent& m : matrix->parts()) {
    const Size bytes = Mul(m.bytes, *matrices);
    if (!bytes) {
      return std::unexpected(bytes.error());
    }
    PartExtent p{.role = m.role, .bytes = *bytes, .readable = *bytes};
    // GGML's over-read past the tensor (representation.cc's rule).
    const std::uint64_t step = Info(format.layout).over_read_row_step;
    const std::uint64_t k = format.shape.ne[0];
    if (m.role == PartRole::kData && step != 0 && k % step != 0) {
      const GgmlBlock& block = *info.ggml;
      const Size over = Mul((step - (k % step)) / block.block_elements, block.block_bytes);
      const Size readable = over ? Add(p.bytes, *over) : over;
      if (!readable) {
        return std::unexpected(readable.error());
      }
      p.readable = *readable;
    }
    parts.Add(p);
  }
  return parts;
}

std::optional<AddressUnit> AddressUnitOf(const TensorFormat& format) {
  if (!KnownEncoding(format.encoding) || !KnownLayout(format.layout) ||
      format.params.tensor_scale) {
    return std::nullopt;
  }
  const EncodingInfo& info = Info(format.encoding);
  if (!Holds(format.layout, info) || !CheckParams(format, info)) {
    return std::nullopt;
  }
  return UnitOf(format.layout, info);
}

std::expected<std::uint64_t, FormatError> RowBytes(const TensorFormat& format, PartRole role) {
  const auto parts = PartsOf(format);
  if (!parts) {
    return std::unexpected(parts.error());
  }
  if (!parts->Find(role)) {
    return std::unexpected(FormatError::kParts);
  }
  const EncodingInfo& info = Info(format.encoding);
  const std::uint64_t k = format.shape.ne[0];
  if (role == PartRole::kTensorScale) {
    return std::unexpected(FormatError::kLayout);  // one value a matrix, not a row
  }
  switch (format.layout) {
    case Layout::kDense:
    case Layout::kGgmlBlocks:
    case Layout::kGgmlNvfp4Blocks: {
      const AddressUnit unit = *UnitOf(format.layout, info);
      return k / unit.elements * unit.bytes;
    }
    case Layout::kSplit:
      if (format.encoding == Encoding::kAffine) {
        const AffineRow row = AffineRowOf(format);
        return role == PartRole::kCodes   ? row.codes
               : role == PartRole::kZeros ? row.zeros
                                          : row.scales;
      }
      if (format.encoding == Encoding::kFp8Scaled) {
        if (role == PartRole::kCodes) return k;
        if (format.params.block_rows != 1) return std::unexpected(FormatError::kLayout);
        return (k / format.params.block_cols + (k % format.params.block_cols != 0)) * 4;
      }
      return role == PartRole::kCodes ? CodeRowBytes(k, info) : k / info.block_elements;
    case Layout::kRowInterleaved:
      return CodeRowBytes(k, info) + (k / info.block_elements);
    case Layout::kScaleAtoms:
      if (role == PartRole::kCodes) {
        return CodeRowBytes(k, info);
      }
      return std::unexpected(FormatError::kLayout);
    case Layout::kExl3Tiles:
      return std::unexpected(FormatError::kLayout);
  }
  return std::unexpected(FormatError::kUnknown);
}

std::uint64_t Matrices(const TensorFormat& format) {
  return format.shape.ne[2] * format.shape.ne[3];
}

ScaleBytes ScaleBytesOf(const TensorFormat& format, PartRole part) {
  const ScaleCode code = ScaleCodeOf(format.encoding);
  if (code == ScaleCode::kNone || !Validate(format)) return {};
  const EncodingInfo& info = Info(format.encoding);
  switch (format.layout) {
    case Layout::kSplit:
    case Layout::kScaleAtoms:
      if (part != PartRole::kBlockScales) return {};
      return {.code = code, .period = 1, .offset = 0, .count = 1};
    case Layout::kRowInterleaved: {
      if (part != PartRole::kData) return {};
      const std::uint64_t k = format.shape.ne[0];
      const std::uint64_t codes = CodeRowBytes(k, info);
      const std::uint64_t blocks = k / info.block_elements;
      return {.code = code, .period = codes + blocks, .offset = codes, .count = blocks};
    }
    case Layout::kGgmlBlocks:
    case Layout::kGgmlNvfp4Blocks: {
      // block_mxfp4: its E8M0 scale, then 16 bytes; block_nvfp4: four
      // UE4M3 scales, then 32 bytes.
      if (part != PartRole::kData || !info.ggml) return {};
      if (format.encoding == Encoding::kMxfp4 && format.layout == Layout::kGgmlBlocks)
        return {.code = code, .period = info.ggml->block_bytes, .offset = 0, .count = 1};
      if (format.encoding == Encoding::kNvfp4 && format.layout == Layout::kGgmlNvfp4Blocks)
        return {.code = code, .period = info.ggml->block_bytes, .offset = 0, .count = 4};
      return {};
    }
    default:
      return {};
  }
}

std::optional<std::uint64_t> FirstUndefinedScale(const ScaleBytes& where, std::uint64_t at,
                                                 std::span<const std::byte> bytes) {
  if (where.code == ScaleCode::kNone || where.period == 0 || where.count == 0 ||
      where.offset >= where.period || where.count > where.period - where.offset)
    return std::nullopt;
  // Run by run of scale bytes, skipping the bytes between.
  std::uint64_t i = 0;
  while (i < bytes.size()) {
    const std::uint64_t in = (at + i) % where.period;
    if (in < where.offset) {
      i += where.offset - in;
      continue;
    }
    if (in >= where.offset + where.count) {
      i += where.period - in;
      continue;
    }
    const std::uint64_t end =
        std::min<std::uint64_t>(bytes.size(), i + (where.offset + where.count - in));
    for (; i < end; ++i)
      if (!DefinedScale(where.code, std::to_integer<std::uint8_t>(bytes[i]))) return i;
  }
  return std::nullopt;
}

}  // namespace llmp::format
