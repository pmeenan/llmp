// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// A tensor's format (D-108): its encoding with that encoding's parameters,
// its layout, and its logical shape; and the size rule, which says what
// bytes each of its parts holds. Weights and state carry a TensorFormat
// wherever they are described: in the graph IR's tensors, a state
// contract, a prepared artifact's index (read from v0's descriptors by
// artifact/format_map.h until the importer writes them).
//
// The size rule is representation.cc's for the formats v0 describes,
// rule for rule (checked arithmetic; an overflow is refused here rather
// than saturated), and extends to the split, interleaved and tiled
// arrangements v0 stores as plain bytes.

#ifndef LLMP_FORMAT_TENSOR_FORMAT_H_
#define LLMP_FORMAT_TENSOR_FORMAT_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <optional>
#include <span>
#include <string_view>

#include "format/encoding.h"
#include "format/layout.h"

namespace llmp::format {

inline constexpr std::size_t kMaxDims = 4;

// A logical shape in GGML's order: ne[0] values to a row, ne[1] rows,
// ne[2] · ne[3] matrices. Dimensions from `rank` on are 1; a scalar has
// rank 0.
struct Shape {
  std::array<std::uint64_t, kMaxDims> ne{1, 1, 1, 1};
  std::uint8_t rank = 0;
  bool operator==(const Shape&) const = default;
};

// ne in GGML's order (innermost first), at most kMaxDims of them.
std::optional<Shape> ShapeOf(std::span<const std::uint64_t> ne);

struct TensorFormat {
  Encoding encoding = Encoding::kF32;
  EncodingParams params{};
  Layout layout = Layout::kDense;
  Shape shape{};
  // Logical matrices stored as one along ne[1] (gate and up as 2): each
  // has ne[1] / stacked rows and, with a tensor scale, its own scale.
  std::uint8_t stacked = 1;
  bool operator==(const TensorFormat&) const = default;
};

enum class FormatError : std::uint8_t {
  kUnknown,   // an encoding or layout outside the enums
  kShape,     // a rank over kMaxDims, a zero or stray dimension, a bad stack
  kBlock,     // a row that is not whole blocks (or tiles, or scale atoms)
  kLayout,    // the layout does not hold this encoding, or the part's rows
  kParams,    // parameters the encoding does not take, or lacks
  kParts,     // a part the format does not have
  kOverflow,  // a size beyond 64 bits
};
std::string_view FormatErrorName(FormatError error);

// One part's bytes, and its readable range: the bytes plus what kernels
// may read past them (layout.h; zero padding the store reserves).
struct PartExtent {
  PartRole role = PartRole::kData;
  std::uint64_t bytes = 0;
  std::uint64_t readable = 0;
  bool operator==(const PartExtent&) const = default;
};

// A format's parts, in a fixed order: kData, or kCodes then kBlockScales;
// then kTensorScale; EXL3's kCodes, kInputSigns, kOutputSigns and
// kCodebookMarker. Each holds the format's matrices one after the other,
// so a matrix's share of a part is bytes / Matrices().
class PartList {
 public:
  static constexpr std::size_t kMax = 4;
  std::span<const PartExtent> parts() const { return std::span(parts_).first(count_); }
  std::size_t size() const { return count_; }
  const PartExtent& operator[](std::size_t i) const { return parts_.at(i); }
  std::optional<PartExtent> Find(PartRole role) const;
  void Add(PartExtent part);  // at most kMax

 private:
  std::array<PartExtent, kMax> parts_{};
  std::size_t count_ = 0;
};

// Whether the format is one the size rule takes: a known encoding and
// layout that holds it, parameters it takes, and a shape of whole blocks.
std::expected<void, FormatError> Validate(const TensorFormat& format);

// The size rule: each part's bytes and readable range.
std::expected<PartList, FormatError> PartsOf(const TensorFormat& format);

// The bytes of one row of a part, for a layout whose rows are addressable
// (kLayout otherwise; kParts for a part the format does not have).
std::expected<std::uint64_t, FormatError> RowBytes(const TensorFormat& format, PartRole role);

// ne[2] · ne[3] (of a valid format, never overflowing: its parts' sizes
// are multiples of it).
std::uint64_t Matrices(const TensorFormat& format);

}  // namespace llmp::format

#endif  // LLMP_FORMAT_TENSOR_FORMAT_H_
