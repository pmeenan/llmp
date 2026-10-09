// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "kernels/reference/decode.h"

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>
#include <limits>
#include <optional>
#include <span>
#include <string_view>

#include "format/encoding.h"
#include "format/layout.h"
#include "format/tensor_format.h"
#include "ggml.h"
#include "kernels/reference/decode_exl3.h"

namespace llmp::kernels::reference {
namespace {

using format::Encoding;
using format::Layout;
using format::PartRole;
using format::TensorFormat;
using Bytes = std::span<const std::byte>;

std::uint8_t U8(Bytes b, std::uint64_t at) { return std::to_integer<std::uint8_t>(b[at]); }

template <typename T>
T Load(Bytes b, std::uint64_t at) {
  T v{};
  std::memcpy(&v, b.data() + at, sizeof(T));  // both targets are little-endian
  return v;
}

std::uint8_t Nibble(std::uint8_t byte, bool high) {
  return high ? static_cast<std::uint8_t>(byte >> 4U) : static_cast<std::uint8_t>(byte & 15U);
}

float DenseValue(Encoding e, Bytes row, std::uint64_t j) {
  switch (e) {
    case Encoding::kF32:
      return Load<float>(row, 4 * j);
    case Encoding::kF16:
      return F16ToF32(Load<std::uint16_t>(row, 2 * j));
    case Encoding::kBF16:
      return Bf16ToF32(Load<std::uint16_t>(row, 2 * j));
    case Encoding::kF64:
      return static_cast<float>(Load<double>(row, 8 * j));
    case Encoding::kI8:
      return static_cast<float>(Load<std::int8_t>(row, j));
    case Encoding::kI16:
      return static_cast<float>(Load<std::int16_t>(row, 2 * j));
    case Encoding::kI32:
      return static_cast<float>(Load<std::int32_t>(row, 4 * j));
    case Encoding::kI64:
      return static_cast<float>(Load<std::int64_t>(row, 8 * j));
    case Encoding::kU8:
      return static_cast<float>(U8(row, j));
    case Encoding::kU16:
      return static_cast<float>(Load<std::uint16_t>(row, 2 * j));
    case Encoding::kU32:
      return static_cast<float>(Load<std::uint32_t>(row, 4 * j));
    case Encoding::kU64:
      return static_cast<float>(Load<std::uint64_t>(row, 8 * j));
    case Encoding::kBool:
      return U8(row, j) != 0 ? 1.0f : 0.0f;
    case Encoding::kF8E4M3:
      return E4m3ToF32(U8(row, j));
    case Encoding::kF8E5M2:
      return E5m2ToF32(U8(row, j));
    case Encoding::kE8M0:
      return E8m0ToF32(U8(row, j));
    default:
      return std::numeric_limits<float>::quiet_NaN();  // not dense: never reached
  }
}

// A block-scaled encoding's code j of a row of codes.
float Code(const format::EncodingInfo& info, Bytes codes, std::uint64_t j) {
  if (info.element_bits == 8) {
    return E4m3ToF32(U8(codes, j));  // MXFP8
  }
  return E2m1ToF32(Nibble(U8(codes, j / 2), j % 2 != 0));
}

float BlockScale(Encoding e, std::uint8_t bits) {
  return e == Encoding::kNvfp4 ? E4m3ToF32(bits) : E8m0ToF32(bits);
}

// Where one matrix's share of each part starts.
struct Parts {
  Bytes data, codes, scales, tensor_scale, input_signs, output_signs, biases, zeros;
};

std::expected<Parts, DecodeError> Bind(const TensorFormat& f, const format::PartList& extents,
                                       std::span<const PartView> views, std::uint64_t matrix) {
  if (views.size() != extents.size()) {
    return std::unexpected(DecodeError::kParts);
  }
  const std::uint64_t matrices = format::Matrices(f);
  Parts parts;
  for (const format::PartExtent& extent : extents.parts()) {
    const PartView* view = nullptr;
    for (const PartView& v : views) {
      if (v.role == extent.role) {
        if (view != nullptr) {
          return std::unexpected(DecodeError::kParts);
        }
        view = &v;
      }
    }
    if (view == nullptr || view->bytes.size() < extent.bytes) {
      return std::unexpected(DecodeError::kParts);
    }
    const std::uint64_t share = extent.bytes / matrices;
    const Bytes mine = view->bytes.subspan(matrix * share, share);
    switch (extent.role) {
      case PartRole::kData:
        parts.data = mine;
        break;
      case PartRole::kCodes:
        parts.codes = mine;
        break;
      case PartRole::kBlockScales:
        parts.scales = mine;
        break;
      case PartRole::kTensorScale:
        parts.tensor_scale = mine;
        break;
      case PartRole::kInputSigns:
        parts.input_signs = mine;
        break;
      case PartRole::kOutputSigns:
        parts.output_signs = mine;
        break;
      case PartRole::kCodebookMarker:
        break;  // its value means nothing (artifact-format.md): the codebook is a parameter
      case PartRole::kBiases:
        parts.biases = mine;
        break;
      case PartRole::kZeros:
        parts.zeros = mine;
        break;
    }
  }
  return parts;
}

// GGML's activation-side blocks, which its type traits do not dequantize:
// d · q, as ggml-quants.c's dequantize_row_q8_K.
void Q8Row(Encoding e, Bytes row, std::uint64_t k, float* y) {
  const bool q8_1 = e == Encoding::kQ8_1;  // an F16 d, else an F32 d
  const format::GgmlBlock& g = *format::Info(e).ggml;
  const std::uint64_t block = g.block_elements;
  const std::uint64_t bytes = g.block_bytes;
  for (std::uint64_t b = 0; b < k / block; ++b) {
    const Bytes x = row.subspan(b * bytes, bytes);
    const float d = q8_1 ? F16ToF32(Load<std::uint16_t>(x, 0)) : Load<float>(x, 0);
    for (std::uint64_t j = 0; j < block; ++j) {
      y[(b * block) + j] = d * static_cast<float>(Load<std::int8_t>(x, 4 + j));
    }
  }
}

// `bits` bits at bit `at` of a little-endian bit stream (layout.h kSplit).
std::uint32_t Bits(Bytes stream, std::uint64_t at, std::uint32_t bits) {
  std::uint32_t v = 0;
  for (std::uint32_t i = 0; i < bits; ++i) {
    const std::uint64_t bit = at + i;
    v |= static_cast<std::uint32_t>((U8(stream, bit / 8) >> (bit % 8)) & 1U) << i;
  }
  return v;
}

float AffineScaleValue(format::AffineScale type, Bytes row, std::uint64_t g) {
  switch (type) {
    case format::AffineScale::kF16:
      return F16ToF32(Load<std::uint16_t>(row, 2 * g));
    case format::AffineScale::kBF16:
      return Bf16ToF32(Load<std::uint16_t>(row, 2 * g));
    case format::AffineScale::kF32:
    case format::AffineScale::kNone:  // not a valid format: never reached
      break;
  }
  return Load<float>(row, 4 * g);
}

// The affine encoding's row r: s · q + b rounded once (a fused
// multiply-add), or s · (q − z), q − z exact.
void AffineRow(const TensorFormat& f, const Parts& p, std::uint64_t r, float* y) {
  const format::EncodingParams& a = f.params;
  const std::uint64_t k = f.shape.ne[0];
  const std::uint64_t groups = k / a.group;
  const std::uint64_t code_row = k / 8 * a.bits;
  const std::uint64_t scale_row =
      groups * (format::Info(format::AffineScaleEncoding(a.scale)).element_bits / 8);
  const std::uint64_t zero_row = ((groups * a.bits) + 7) / 8;
  const Bytes codes = p.codes.subspan(r * code_row, code_row);
  const Bytes scales = p.scales.subspan(r * scale_row, scale_row);
  for (std::uint64_t g = 0; g < groups; ++g) {
    const float s = AffineScaleValue(a.scale, scales, g);
    const float b = a.zero_point
                        ? 0.0f
                        : AffineScaleValue(a.scale, p.biases.subspan(r * scale_row, scale_row), g);
    const auto z = a.zero_point ? static_cast<std::int32_t>(Bits(
                                      p.zeros.subspan(r * zero_row, zero_row), g * a.bits, a.bits))
                                : 0;
    for (std::uint64_t j = g * a.group; j < (g + 1) * a.group; ++j) {
      const auto q = static_cast<std::int32_t>(Bits(codes, j * a.bits, a.bits));
      y[j] = a.zero_point ? s * static_cast<float>(q - z) : std::fma(s, static_cast<float>(q), b);
    }
  }
}

std::expected<void, DecodeError> DecodeRow(const TensorFormat& f, const Parts& p, std::uint64_t r,
                                           float* y) {
  const format::EncodingInfo& info = format::Info(f.encoding);
  const std::uint64_t k = f.shape.ne[0];
  switch (f.layout) {
    case Layout::kDense: {
      const std::uint64_t row_bytes = k * (info.element_bits / 8);
      const Bytes row = p.data.subspan(r * row_bytes, row_bytes);
      for (std::uint64_t j = 0; j < k; ++j) {
        y[j] = DenseValue(f.encoding, row, j);
      }
      return {};
    }
    case Layout::kGgmlBlocks: {
      const format::GgmlBlock& g = *info.ggml;
      const std::uint64_t row_bytes = k / g.block_elements * g.block_bytes;
      const Bytes row = p.data.subspan(r * row_bytes, row_bytes);
      if (f.encoding == Encoding::kMxfp4) {
        // block_mxfp4: an E8M0 scale, then 16 bytes, byte j holding values
        // j (low nibble) and j + 16.
        for (std::uint64_t j = 0; j < k; ++j) {
          const Bytes x = row.subspan(j / 32 * 17, 17);
          const std::uint64_t w = j % 32;
          y[j] = E2m1ToF32(Nibble(U8(x, 1 + (w % 16)), w >= 16)) * E8m0ToF32(U8(x, 0));
        }
        return {};
      }
      if (f.encoding == Encoding::kQ8_1 || f.encoding == Encoding::kQ8_K) {
        Q8Row(f.encoding, row, k, y);
        return {};
      }
      const ggml_type_traits* traits = ggml_get_type_traits(static_cast<ggml_type>(g.type));
      if (traits == nullptr || traits->to_float == nullptr) {
        return std::unexpected(DecodeError::kUnsupported);
      }
      traits->to_float(row.data(), y, static_cast<std::int64_t>(k));
      return {};
    }
    case Layout::kGgmlNvfp4Blocks: {
      // block_nvfp4: four E4M3 scales, then 32 bytes, byte j of a 16-value
      // sub-block holding values j (low nibble) and j + 8.
      const std::uint64_t row_bytes = k / 64 * 36;
      const Bytes row = p.data.subspan(r * row_bytes, row_bytes);
      for (std::uint64_t j = 0; j < k; ++j) {
        const Bytes x = row.subspan(j / 64 * 36, 36);
        const std::uint64_t s = (j % 64) / 16;
        const std::uint64_t t = j % 16;
        y[j] = E2m1ToF32(Nibble(U8(x, 4 + (s * 8) + (t % 8)), t >= 8)) * E4m3ToF32(U8(x, s));
      }
      return {};
    }
    case Layout::kSplit:
    case Layout::kRowInterleaved:
    case Layout::kCutlassScaleAtoms: {
      if (f.encoding == Encoding::kAffine) {
        AffineRow(f, p, r, y);
        return {};
      }
      const std::uint64_t blocks = k / info.block_elements;
      const std::uint64_t code_bytes = k * info.element_bits / 8;
      Bytes codes;
      Bytes scales;
      if (f.layout == Layout::kRowInterleaved) {
        const Bytes row = p.data.subspan(r * (code_bytes + blocks), code_bytes + blocks);
        codes = row.first(code_bytes);
        scales = row.subspan(code_bytes);
      } else {
        codes = p.codes.subspan(r * code_bytes, code_bytes);
        if (f.layout == Layout::kSplit) {
          scales = p.scales.subspan(r * blocks, blocks);
        }
      }
      for (std::uint64_t j = 0; j < k; ++j) {
        const std::uint64_t b = j / info.block_elements;
        const std::uint8_t s = f.layout == Layout::kCutlassScaleAtoms
                                   ? U8(p.scales, format::ScaleAtomOffset(r, b, blocks))
                                   : U8(scales, b);
        y[j] = Code(info, codes, j) * BlockScale(f.encoding, s);
      }
      return {};
    }
    case Layout::kExl3Tiles:
      break;  // a block of rows at a time (DecodeRows)
  }
  return std::unexpected(DecodeError::kUnsupported);
}

}  // namespace

std::string_view DecodeErrorName(DecodeError error) {
  switch (error) {
    case DecodeError::kUnsupported:
      return "unsupported format";
    case DecodeError::kFormat:
      return "invalid format";
    case DecodeError::kParts:
      return "parts do not match the format";
    case DecodeError::kRange:
      return "rows or output out of range";
  }
  return "unknown";
}

float F16ToF32(std::uint16_t bits) {
  const float sign = (bits & 0x8000U) != 0 ? -1.0f : 1.0f;
  const unsigned exponent = (bits >> 10U) & 31U;
  const unsigned mantissa = bits & 1023U;
  if (exponent == 31) {
    return mantissa != 0 ? std::numeric_limits<float>::quiet_NaN()
                         : sign * std::numeric_limits<float>::infinity();
  }
  if (exponent == 0) {
    return sign * std::ldexp(static_cast<float>(mantissa), -24);
  }
  return sign * std::ldexp(static_cast<float>(1024U + mantissa), static_cast<int>(exponent) - 25);
}

std::uint16_t F32ToF16(float value) {
  const auto x = std::bit_cast<std::uint32_t>(value);
  const auto sign = static_cast<std::uint16_t>((x >> 16U) & 0x8000U);
  const std::uint32_t magnitude = x & 0x7FFFFFFFU;
  if (magnitude >= 0x7F800000U) {  // infinity or NaN (quieted)
    return static_cast<std::uint16_t>(sign | (magnitude > 0x7F800000U ? 0x7E00U : 0x7C00U));
  }
  if (magnitude >= 0x477FF000U) {  // rounds to 65520 or beyond
    return static_cast<std::uint16_t>(sign | 0x7C00U);
  }
  if (magnitude < 0x38800000U) {  // below 2^-14: an F16 subnormal, in units of 2^-24
    const std::uint32_t exponent = magnitude >> 23U;
    const std::uint32_t shift = 126U - exponent;
    if (exponent == 0 || shift > 24) {
      return sign;  // below 2^-25: zero
    }
    const std::uint32_t significand = (magnitude & 0x7FFFFFU) | 0x800000U;
    std::uint32_t q = significand >> shift;
    const std::uint32_t rest = significand & ((1U << shift) - 1U);
    const std::uint32_t half = 1U << (shift - 1U);
    if (rest > half || (rest == half && (q & 1U) != 0)) {
      ++q;  // may carry into the least normal, which is its encoding
    }
    return static_cast<std::uint16_t>(sign | q);
  }
  const std::uint32_t rounded = magnitude + 0xFFFU + ((magnitude >> 13U) & 1U);
  return static_cast<std::uint16_t>(sign | ((rounded - 0x38000000U) >> 13U));
}

float Bf16ToF32(std::uint16_t bits) { return std::bit_cast<float>(std::uint32_t{bits} << 16U); }

std::uint16_t F32ToBf16(float value) {
  const auto x = std::bit_cast<std::uint32_t>(value);
  if ((x & 0x7FFFFFFFU) > 0x7F800000U) {
    return static_cast<std::uint16_t>((x >> 16U) | 0x40U);
  }
  return static_cast<std::uint16_t>((x + 0x7FFFU + ((x >> 16U) & 1U)) >> 16U);
}

float E4m3ToF32(std::uint8_t bits) {
  const float sign = (bits & 0x80U) != 0 ? -1.0f : 1.0f;
  const unsigned exponent = (bits >> 3U) & 15U;
  const unsigned mantissa = bits & 7U;
  if (exponent == 15 && mantissa == 7) {
    return std::numeric_limits<float>::quiet_NaN();
  }
  if (exponent == 0) {
    return sign * std::ldexp(static_cast<float>(mantissa), -9);
  }
  return sign * std::ldexp(static_cast<float>(8U + mantissa), static_cast<int>(exponent) - 10);
}

float E5m2ToF32(std::uint8_t bits) {
  return F16ToF32(static_cast<std::uint16_t>(std::uint16_t{bits} << 8U));
}

float E8m0ToF32(std::uint8_t bits) {
  if (bits == 0xFF) {
    return std::numeric_limits<float>::quiet_NaN();
  }
  return std::ldexp(1.0f, static_cast<int>(bits) - 127);
}

float E2m1ToF32(std::uint8_t code) {
  constexpr std::array<float, 8> kValues = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  const float v = kValues.at(code & 7U);
  return (code & 8U) != 0 ? -v : v;
}

std::expected<void, DecodeError> DecodeRows(const format::TensorFormat& format,
                                            std::span<const PartView> parts, std::uint64_t matrix,
                                            std::uint64_t first, std::uint64_t count,
                                            std::span<float> out) {
  const auto extents = format::PartsOf(format);
  if (!extents) {
    return std::unexpected(DecodeError::kFormat);
  }
  if (format.layout == Layout::kExl3Tiles && !exl3::Transformable(format)) {
    return std::unexpected(DecodeError::kUnsupported);
  }
  const std::uint64_t k = format.shape.ne[0];
  const std::uint64_t n = format.shape.ne[1];
  std::uint64_t end = 0;
  std::uint64_t values = 0;
  if (matrix >= format::Matrices(format) || __builtin_add_overflow(first, count, &end) || end > n ||
      __builtin_mul_overflow(count, k, &values) || values != out.size()) {
    return std::unexpected(DecodeError::kRange);
  }
  const auto bound = Bind(format, *extents, parts, matrix);
  if (!bound) {
    return std::unexpected(bound.error());
  }
  if (format.layout == Layout::kExl3Tiles) {
    exl3::Rows(format,
               {.codes = bound->codes, .suh = bound->input_signs, .svh = bound->output_signs},
               first, count, out);
    return {};
  }
  for (std::uint64_t r = first; r < end; ++r) {
    float* y = out.data() + ((r - first) * k);
    if (auto ok = DecodeRow(format, *bound, r, y); !ok) {
      return ok;
    }
    if (format.params.tensor_scale) {
      const std::uint64_t index = r / (n / format.stacked);
      const float scale = Load<float>(bound->tensor_scale, 4 * index);
      for (std::uint64_t j = 0; j < k; ++j) {
        y[j] *= scale;
      }
    }
  }
  return {};
}

std::expected<void, DecodeError> DecodeExl3Rotated(const format::TensorFormat& format,
                                                   std::span<const PartView> parts,
                                                   std::uint64_t matrix, std::uint64_t first,
                                                   std::uint64_t count,
                                                   std::span<std::uint16_t> out) {
  const auto extents = format::PartsOf(format);
  if (!extents) {
    return std::unexpected(DecodeError::kFormat);
  }
  if (format.layout != Layout::kExl3Tiles) {
    return std::unexpected(DecodeError::kUnsupported);
  }
  const std::uint64_t k = format.shape.ne[0];
  std::uint64_t end = 0;
  std::uint64_t values = 0;
  if (matrix >= format::Matrices(format) || first % 16 != 0 || count % 16 != 0 ||
      __builtin_add_overflow(first, count, &end) || end > format.shape.ne[1] ||
      __builtin_mul_overflow(count, k, &values) || values != out.size()) {
    return std::unexpected(DecodeError::kRange);
  }
  const auto bound = Bind(format, *extents, parts, matrix);
  if (!bound) {
    return std::unexpected(bound.error());
  }
  exl3::Rotated(format,
                {.codes = bound->codes, .suh = bound->input_signs, .svh = bound->output_signs},
                first, count, out);
  return {};
}

std::expected<void, DecodeError> EncodeRows(const format::TensorFormat& format,
                                            std::uint64_t matrix, std::uint64_t first,
                                            std::span<const float> rows,
                                            std::span<std::byte> data) {
  const auto extents = format::PartsOf(format);
  if (!extents) {
    return std::unexpected(DecodeError::kFormat);
  }
  if (extents->size() != 1 || (*extents)[0].role != PartRole::kData ||
      (format.layout != Layout::kDense && format.layout != Layout::kGgmlBlocks &&
       format.layout != Layout::kGgmlNvfp4Blocks)) {
    return std::unexpected(DecodeError::kUnsupported);
  }
  const std::uint64_t k = format.shape.ne[0];
  const std::uint64_t n = format.shape.ne[1];
  const std::uint64_t count = rows.size() / k;
  std::uint64_t end = 0;
  if (matrix >= format::Matrices(format) || rows.size() % k != 0 ||
      __builtin_add_overflow(first, count, &end) || end > n || data.size() < (*extents)[0].bytes) {
    return std::unexpected(DecodeError::kRange);
  }
  const std::uint64_t share = (*extents)[0].bytes / format::Matrices(format);
  const std::uint64_t row_bytes = share / n;
  const std::span<std::byte> out =
      data.subspan((matrix * share) + (first * row_bytes), count * row_bytes);
  switch (format.encoding) {
    case Encoding::kF32:
      std::memcpy(out.data(), rows.data(), rows.size_bytes());
      return {};
    case Encoding::kF16:
    case Encoding::kBF16:
      for (std::size_t i = 0; i < rows.size(); ++i) {
        const std::uint16_t v =
            format.encoding == Encoding::kF16 ? F32ToF16(rows[i]) : F32ToBf16(rows[i]);
        std::memcpy(out.data() + (2 * i), &v, 2);
      }
      return {};
    default:
      break;
  }
  const format::EncodingInfo& info = format::Info(format.encoding);
  // GGML's quantize_chunk takes neither activation-side block.
  if (!info.quantized || !info.ggml || format.encoding == Encoding::kQ8_1 ||
      format.encoding == Encoding::kQ8_K) {
    return std::unexpected(DecodeError::kUnsupported);
  }
  const auto type = static_cast<ggml_type>(info.ggml->type);
  const ggml_type_traits* traits = ggml_get_type_traits(type);
  if (traits == nullptr || traits->from_float_ref == nullptr ||
      ggml_quantize_requires_imatrix(type)) {
    return std::unexpected(DecodeError::kUnsupported);
  }
  // From the output's first row: GGML places a chunk at its start row.
  const std::size_t written =
      ggml_quantize_chunk(type, rows.data(), out.data(), 0, static_cast<std::int64_t>(count),
                          static_cast<std::int64_t>(k), nullptr);
  if (written != out.size()) {
    return std::unexpected(DecodeError::kUnsupported);
  }
  return {};
}

}  // namespace llmp::kernels::reference
