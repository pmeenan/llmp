// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "import/transforms.h"

#include <algorithm>
#include <bit>
#include <cfenv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <functional>
#include <limits>
#include <optional>
#include <span>
#include <string_view>

#include "format/encoding.h"
#include "format/layout.h"
#include "format/tensor_format.h"
#include "import/error.h"

namespace llmp::import {
namespace {

static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);
static_assert(sizeof(double) == 8 && std::numeric_limits<double>::is_iec559);

std::unexpected<Error> Refuse(std::string_view reason) {
  return std::unexpected(Error{.rule = Rule::kBounds, .reason = reason});
}

std::uint32_t InputBits(std::span<const std::byte> input, std::size_t offset) {
  return (std::to_integer<std::uint32_t>(input[offset]) << 16U) |
         (std::to_integer<std::uint32_t>(input[offset + 1]) << 24U);
}

// Construct binary64 directly. Even a subnormal BF16 is a normal
// binary64; no F32 load/conversion can flush a source value to zero.
double Bf16Double(std::uint32_t bits) {
  const std::uint64_t sign = static_cast<std::uint64_t>(bits & 0x80000000U) << 32U;
  const std::uint32_t exponent = (bits >> 23U) & 0xFFU;
  const std::uint32_t fraction = (bits >> 16U) & 0x7FU;
  std::uint64_t encoded = sign;
  if (exponent == 255) {
    encoded |= 0x7FF0000000000000ULL;
    if (fraction != 0) encoded |= (static_cast<std::uint64_t>(fraction) << 45U) | (1ULL << 51U);
  } else if (exponent != 0) {
    encoded |= (static_cast<std::uint64_t>(exponent + 896U) << 52U) |
               (static_cast<std::uint64_t>(fraction) << 45U);
  } else if (fraction != 0) {
    const auto high = static_cast<unsigned>(31 - std::countl_zero(fraction));
    encoded |= (static_cast<std::uint64_t>(high + 890U) << 52U) |
               (static_cast<std::uint64_t>(fraction ^ (1U << high)) << (52U - high));
  }
  return std::bit_cast<double>(encoded);
}

std::uint64_t RoundRight(std::uint64_t significand, unsigned shift) {
  const std::uint64_t whole = significand >> shift;
  const std::uint64_t remainder = significand & ((1ULL << shift) - 1U);
  const std::uint64_t halfway = 1ULL << (shift - 1U);
  return whole + static_cast<std::uint64_t>(remainder > halfway ||
                                            (remainder == halfway && (whole & 1U) != 0));
}

// Integer binary64 -> binary32 nearest-even encoding. A floating cast
// under FTZ can erase a nonzero F32 subnormal even with FE_TONEAREST.
// Normal binary64 exp results cover every nonzero F32 result, so the
// subnormal bits below never pass through floating hardware as F32.
std::uint32_t F32Bits(double value) {
  const std::uint64_t bits = std::bit_cast<std::uint64_t>(value);
  const auto sign = static_cast<std::uint32_t>(bits >> 32U) & 0x80000000U;
  const auto encoded_exp = static_cast<unsigned>((bits >> 52U) & 0x7FFU);
  const std::uint64_t fraction = bits & 0x000FFFFFFFFFFFFFULL;
  if (encoded_exp == 2047) {
    return sign | 0x7F800000U |
           (fraction == 0 ? 0U : (static_cast<std::uint32_t>(fraction >> 29U) | 0x00400000U));
  }
  if (encoded_exp == 0) return sign;  // a double subnormal is below half a float subnormal
  int exponent = static_cast<int>(encoded_exp) - 1023;
  if (exponent > 127) return sign | 0x7F800000U;
  if (exponent < -150) return sign;
  const std::uint64_t significand = (1ULL << 52U) | fraction;
  if (exponent < -126) {
    return sign | static_cast<std::uint32_t>(
                      RoundRight(significand, static_cast<unsigned>(-exponent - 97)));
  }
  std::uint64_t rounded = RoundRight(significand, 29);
  if (rounded == (1ULL << 24U)) {
    rounded >>= 1U;
    ++exponent;
  }
  if (exponent > 127) return sign | 0x7F800000U;
  return sign | (static_cast<std::uint32_t>(exponent + 127) << 23U) |
         (static_cast<std::uint32_t>(rounded) & 0x007FFFFFU);
}

// Python's float loads quiet a signaling BF16 NaN, preserving payload and
// sign; neg_exp's final negation flips its sign. Handle it without relying
// on a compiler's NaN propagation through arithmetic/libm.
std::uint32_t FoldBits(ScalarFold fold, std::uint32_t bits) {
  if ((bits & 0x7F800000U) == 0x7F800000U && (bits & 0x007FFFFFU) != 0) {
    return (bits | 0x00400000U) ^ (fold == ScalarFold::kANegExp ? 0x80000000U : 0U);
  }
  const double value = Bf16Double(bits);
  const double result = fold == ScalarFold::kNormPlusOne ? value + 1.0 : -std::exp(value);
  return F32Bits(result);
}

bool ProductFits(std::uint64_t a, std::uint64_t b) {
  return b == 0 || a <= std::numeric_limits<std::uint64_t>::max() / b;
}

}  // namespace

std::expected<void, Error> FoldBf16(ScalarFold fold, std::span<const std::byte> input,
                                    std::span<std::byte> output) {
  if (fold != ScalarFold::kNormPlusOne && fold != ScalarFold::kANegExp) {
    return Refuse("unknown scalar fold");
  }
  if (input.size() % 2 != 0 || input.size() > std::numeric_limits<std::size_t>::max() / 2 ||
      output.size() != input.size() * 2) {
    return Refuse("scalar fold spans do not hold whole BF16 and F32 values");
  }
  if (std::fegetround() != FE_TONEAREST)
    return Refuse("scalar folds require nearest-even rounding");
  if (input.empty()) return {};
  const std::less<const std::byte*> less;
  if (less(input.data(), output.data() + output.size()) &&
      less(output.data(), input.data() + input.size())) {
    return Refuse("scalar fold spans overlap");
  }
  // Preflight the entire call: a later overflowing A_log must not leave
  // earlier output changed. This rereads caller RAM, never checkpoint I/O.
  if (fold == ScalarFold::kANegExp) {
    for (std::size_t i = 0; i < input.size(); i += 2) {
      const double value = Bf16Double(InputBits(input, i));
      if (std::isfinite(value) && std::isinf(std::exp(value))) {
        return Refuse("negative exponential overflows binary64");
      }
    }
  }
  for (std::size_t i = 0; i < input.size(); i += 2) {
    const std::uint32_t bits = FoldBits(fold, InputBits(input, i));
    for (std::size_t j = 0; j < 4; ++j) {
      output[i * 2 + j] = static_cast<std::byte>((bits >> (j * 8)) & 0xFFU);
    }
  }
  return {};
}

std::expected<HeadPermutationPlan, Error> HeadPermutationPlan::Make(
    const format::TensorFormat& format, HeadAxis axis, std::uint64_t key_heads,
    std::uint64_t value_heads, std::uint64_t head_elements, std::uint64_t first_value_row) {
  using format::Encoding;
  using format::Layout;
  using format::PartRole;
  const auto parts = format::PartsOf(format);
  if (!parts || format.stacked != 1 || format.shape.rank < 1 || format.shape.rank > 2 ||
      !((format.encoding == Encoding::kF32 && format.layout == Layout::kDense) ||
        (format.encoding == Encoding::kMxfp8 && format.layout == Layout::kSplit &&
         !format.params.tensor_scale))) {
    return Refuse("head permutation needs unstacked dense F32 or split MXFP8");
  }
  if ((format.encoding == Encoding::kF32 &&
       (parts->size() != 1 || !parts->Find(PartRole::kData))) ||
      (format.encoding == Encoding::kMxfp8 &&
       (parts->size() != 2 || !parts->Find(PartRole::kCodes) ||
        !parts->Find(PartRole::kBlockScales)))) {
    return Refuse("head permutation lacks complete format parts");
  }
  if (key_heads == 0 || value_heads == 0 || value_heads % key_heads != 0 || head_elements == 0 ||
      !ProductFits(value_heads, head_elements)) {
    return Refuse("invalid value-head counts or width");
  }
  const std::uint64_t extent = value_heads * head_elements;
  if (axis == HeadAxis::kRows) {
    if (format.shape.rank != 2 || first_value_row > format.shape.ne[1] ||
        extent > format.shape.ne[1] - first_value_row) {
      return Refuse("value heads are outside the matrix rows");
    }
  } else if (axis == HeadAxis::kColumns) {
    if (first_value_row != 0 || extent != format.shape.ne[0] ||
        (format.encoding == Encoding::kMxfp8 && head_elements % 32 != 0)) {
      return Refuse("column heads must cover the row and whole scale blocks");
    }
  } else {
    return Refuse("unknown head-permutation axis");
  }
  // PartsOf checked the whole byte products. Each part is row-addressable;
  // every derived head/prefix interval is bounded by that whole part.
  for (const format::PartExtent& part : parts->parts()) {
    if (!format::RowBytes(format, part.role)) return Refuse("head part has no addressable rows");
  }
  HeadPermutationPlan plan;
  plan.format_ = format;
  plan.parts_ = *parts;
  plan.axis_ = axis;
  plan.key_heads_ = key_heads;
  plan.value_heads_ = value_heads;
  plan.head_elements_ = head_elements;
  plan.first_row_ = first_value_row;
  return plan;
}

std::expected<HeadPermutationCursor, Error> HeadPermutationPlan::CursorFor(
    format::PartRole part) const {
  const auto extent = parts_.Find(part);
  if (!extent) return Refuse("head permutation cursor names an absent part");
  const auto row_bytes = format::RowBytes(format_, part);
  if (!row_bytes) return Refuse("head part has no addressable rows");
  HeadPermutationCursor cursor;
  cursor.axis_ = axis_;
  cursor.part_ = part;
  cursor.bytes_ = extent->bytes;
  cursor.row_bytes_ = *row_bytes;
  cursor.key_heads_ = key_heads_;
  cursor.per_key_ = value_heads_ / key_heads_;
  cursor.head_bytes_ =
      axis_ == HeadAxis::kRows ? head_elements_ * *row_bytes : *row_bytes / value_heads_;
  cursor.first_ = first_row_ * *row_bytes;
  cursor.end_ = cursor.first_ + value_heads_ * cursor.head_bytes_;
  return cursor;
}

std::expected<std::optional<CopyRun>, Error> HeadPermutationCursor::Next(
    std::uint64_t maximum_bytes) {
  if (maximum_bytes == 0) return Refuse("a head-permutation run must hold bytes");
  if (next_ == bytes_) return std::optional<CopyRun>();
  CopyRun run{.part = part_, .source = next_, .destination = next_, .bytes = bytes_ - next_};
  if (axis_ == HeadAxis::kRows && next_ < first_) {
    run.bytes = first_ - next_;
  } else if (axis_ == HeadAxis::kColumns || next_ < end_) {
    const std::uint64_t base =
        axis_ == HeadAxis::kRows ? first_ : (next_ / row_bytes_) * row_bytes_;
    const std::uint64_t local = next_ - base;
    const std::uint64_t head = local / head_bytes_;
    const std::uint64_t within = local % head_bytes_;
    const std::uint64_t destination_head = (head % per_key_) * key_heads_ + head / per_key_;
    run.destination = base + destination_head * head_bytes_ + within;
    run.bytes = head_bytes_ - within;
  }
  run.bytes = std::min(run.bytes, maximum_bytes);
  next_ += run.bytes;
  return std::optional(run);
}

}  // namespace llmp::import
