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

enum class ScalarFold : std::uint8_t { kNormPlusOne, kANegExp };

// Little-endian BF16 -> F32, N whole scalars -> exactly 4*N bytes. Uses
// binary64 arithmetic then integer nearest-even F32 encoding, like the
// historical converter, without hardware F32 flush-to-zero on conversion.
// Refusal leaves all output unchanged (including on a later input's exp
// overflow). No allocation. Spans must be disjoint and stable for the call.
// No other source dtype is accepted by this API. Empty spans are permitted.
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
// Only dense F32/BF16 or split MXFP8 (codes AND block scales), unstacked, rank
// 1 or 2. Rows are rank 2, with unchanged prefix/suffix; columns cover a
// whole row (rank-1 dense input is one row). Column MXFP8 heads are whole 32-value
// blocks. Neither format, shape nor any payload byte changes.
class HeadPermutationPlan {
 public:
  static std::expected<HeadPermutationPlan, Error> Make(const format::TensorFormat& format,
                                                        HeadAxis axis, std::uint64_t key_heads,
                                                        std::uint64_t value_heads,
                                                        std::uint64_t head_elements,
                                                        std::uint64_t first_value_row = 0);

  // One independent monotone cursor per actual part: either source part
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
