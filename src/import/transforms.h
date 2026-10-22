// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Bounded import transforms (D-109). The scalar folds reproduce the BF16
// norm/A conversion in the historical ModelOpt converter, not a general
// weight promotion. Head permutations move bytes without changing encoding.
#ifndef LLMP_IMPORT_TRANSFORMS_H_
#define LLMP_IMPORT_TRANSFORMS_H_

#include <cstddef>
#include <cstdint>
#include <expected>
#include <optional>
#include <span>

#include "format/tensor_format.h"
#include "import/error.h"

namespace llmp::import {

enum class ScalarFold : std::uint8_t { kNormPlusOne, kANegExp, kIdentity };

// Little-endian BF16/F16/F32 -> F32. Uses binary64 arithmetic then integer
// nearest-even F32 encoding, without hardware F32 flush-to-zero on conversion.
// Refusal leaves all output unchanged (including on a later input's exp
// overflow). No allocation. Spans must be disjoint and stable for the call.
// Empty spans are permitted.
double F16Double(std::uint16_t bits) noexcept;

std::expected<void, Error> FoldScalar(ScalarFold fold, format::Encoding source_encoding,
                                      std::span<const std::byte> input,
                                      std::span<std::byte> output);

std::expected<void, Error> FoldBf16(ScalarFold fold, std::span<const std::byte> input,
                                    std::span<std::byte> output);

enum class HeadAxis : std::uint8_t { kRows, kColumns };

struct CopyRun {
  format::PartRole part = format::PartRole::kData;
  std::uint64_t source = 0;
  std::uint64_t destination = 0;
  std::uint64_t bytes = 0;
};

class HeadPermutationPlan;

// Enumerates a validated part from offset zero, in source order. A zero
// maximum is refused without advancing. Finished cursors return nullopt.
// This describes copies; exhaustion does not acknowledge a successful write.
class HeadPermutationCursor {
 public:
  std::expected<std::optional<CopyRun>, Error> Next(std::uint64_t maximum_bytes);

 private:
  friend class HeadPermutationPlan;
  HeadPermutationCursor() = default;
  HeadAxis axis_ = HeadAxis::kRows;
  format::PartRole part_ = format::PartRole::kData;
  std::uint64_t bytes_ = 0;
  std::uint64_t row_bytes_ = 0;
  std::uint64_t head_bytes_ = 0;
  std::uint64_t first_ = 0;
  std::uint64_t end_ = 0;
  std::uint64_t key_heads_ = 0;
  std::uint64_t per_key_ = 0;
  std::uint64_t next_ = 0;
};

// Grouped-to-tiled heads: destination j*K+g receives source g*(V/K)+j.
// Dense F32/BF16, split MXFP8 (codes AND block scales), or split affine
// 4/8-bit, groups 32/64/128, BF16/F16/F32 scales (codes AND block scales
// AND biases), all unstacked.
// Affine is rank 2 only; dense/MXFP8 remain rank 1 or 2. Rows are rank 2,
// with unchanged prefix/suffix; columns cover a whole row. Quantized column
// heads are whole encoding groups and packed code bytes. No decoding,
// allocation, format/shape change or source head-order authority is implied.
class HeadPermutationPlan {
 public:
  static std::expected<HeadPermutationPlan, Error> Make(const format::TensorFormat& format,
                                                        HeadAxis axis, std::uint64_t key_heads,
                                                        std::uint64_t value_heads,
                                                        std::uint64_t head_elements,
                                                        std::uint64_t first_value_row = 0);

  // One independent monotone cursor per actual part: any source part
  // may arrive first. No caller-provided initial offsets or coverage.
  std::expected<HeadPermutationCursor, Error> CursorFor(format::PartRole part) const;
  const format::TensorFormat& format() const { return format_; }
  std::span<const format::PartExtent> parts() const { return parts_.parts(); }

 private:
  HeadPermutationPlan() = default;
  format::TensorFormat format_{};
  format::PartList parts_{};
  HeadAxis axis_ = HeadAxis::kRows;
  std::uint64_t key_heads_ = 0;
  std::uint64_t value_heads_ = 0;
  std::uint64_t head_elements_ = 0;
  std::uint64_t first_row_ = 0;
};

}  // namespace llmp::import
#endif  // LLMP_IMPORT_TRANSFORMS_H_
