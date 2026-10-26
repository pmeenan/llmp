// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#ifndef LLMP_IMPORT_ROUTE_H_
#define LLMP_IMPORT_ROUTE_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <functional>
#include <optional>
#include <span>

#include "import/transforms.h"

namespace llmp::import {
// Sink offsets are relative to one private output part. Borrowed bytes must
// be consumed synchronously. Publication also requires source authentication;
// route completion alone is not a successful import receipt.
using PartSink =
    std::function<std::expected<void, Error>(std::uint64_t, std::span<const std::byte>)>;
class PartRoutePlan;
class PartRouteExecution {
 public:
  PartRouteExecution(const PartRouteExecution&) = delete;
  PartRouteExecution& operator=(const PartRouteExecution&) = delete;
  PartRouteExecution(PartRouteExecution&& other) noexcept;
  PartRouteExecution& operator=(PartRouteExecution&& other) noexcept;

  // Exactly the next source-relative offset, with no gaps or duplicates.
  // Every Consume error is terminal, including a sink error after partial private
  // writes. Workspace and input must be disjoint and stable through the call.
  std::expected<void, Error> Consume(std::uint64_t offset, std::span<const std::byte> input,
                                     const PartSink& sink);
  // Observational completeness check; an incomplete Finish does not poison
  // the cursor. No source authentication/publication is implied by success.
  std::expected<void, Error> Finish() const;
  void Cancel() { failed_ = true; }
  std::uint64_t received() const { return received_; }

 private:
  friend class PartRoutePlan;
  PartRouteExecution() = default;
  std::uint64_t input_bytes_ = 0, output_bytes_ = 0, received_ = 0, acknowledged_ = 0;
  std::optional<HeadPermutationCursor> permutation_{};
  std::optional<ScalarFold> fold_{};
  format::Encoding source_encoding_ = format::Encoding::kBF16;
  std::uint64_t atom_blocks_ = 0;  // a scale-atom route's scales a row
  // A strided route's row bytes, output row stride and offset in a row.
  std::uint64_t row_bytes_ = 0, stride_ = 0, lane_ = 0;
  std::span<std::byte> workspace_{};
  std::array<std::byte, 4> carry_{};
  std::uint8_t carried_ = 0;
  bool failed_ = false, busy_ = false;
};

// Constant-size checked geometry, without a source identity or permission to
// transform any model. Only an owned normalization mapping may bind a route
// to an authenticated source part and canonical output. Current mappings
// still issue copy routes only.
class PartRoutePlan {
 public:
  static std::expected<PartRoutePlan, Error> Copy(std::uint64_t bytes);
  static std::expected<PartRoutePlan, Error> Permute(const HeadPermutationPlan& plan,
                                                     format::PartRole part);
  // Scalar fold to F32 (kBF16, kF16, or kF32 source). Reuses FoldScalar's
  // exact numerical and refusal contract.
  static std::expected<PartRoutePlan, Error> Fold(
      ScalarFold fold, std::uint64_t elements,
      format::Encoding source_encoding = format::Encoding::kBF16);
  // A fold under a head permutation of the source (`plan`'s format):
  // a small part (at most 8192 elements) buffered whole in the workspace,
  // then each permuted run folded to its F32 place (the fold commutes with
  // the reorder).
  static std::expected<PartRoutePlan, Error> PermuteFold(const HeadPermutationPlan& plan,
                                                         format::PartRole part, ScalarFold fold);
  // A matrix's row-major one-byte block scales (`rows` rows of `blocks`)
  // into the scale atoms of format::Layout::kScaleAtoms, losslessly and at
  // the same size: rows whole atoms (a multiple of 128) and blocks a
  // multiple of 4 (a 128-row band of 128 * blocks bytes is the unit). A
  // band's atoms occupy the band's own byte range, so the route buffers
  // one band in its workspace and emits it whole.
  static std::expected<PartRoutePlan, Error> ScaleAtoms(std::uint64_t rows, std::uint64_t blocks);
  // Row-major rows of `row_bytes` into every `stride`-th byte run at `lane`
  // in its row (format::Layout::kRowInterleaved: one part's rows beside
  // another's, each source routed into its lane), written as they arrive.
  // The route writes rows · row_bytes bytes over a span of rows · stride;
  // its sibling lanes cover the rest of the span.
  static std::expected<PartRoutePlan, Error> Strided(std::uint64_t rows, std::uint64_t row_bytes,
                                                     std::uint64_t stride, std::uint64_t lane);
  std::uint64_t input_bytes() const { return input_bytes_; }
  // The bytes it writes, and the output range they lie in (a strided
  // route's rows · stride; otherwise the same).
  std::uint64_t output_bytes() const { return output_bytes_; }
  std::uint64_t span() const {
    return stride_ != 0 ? input_bytes_ / row_bytes_ * stride_ : output_bytes_;
  }
  std::uint64_t lane() const { return lane_; }
  // No allocation: caller owns/funds workspace and keeps it alive through
  // execution. A fold needs >=4 bytes and uses at most 64 KiB; a permuted
  // fold its input and output (4 * 16 KiB = 64 KiB); a scale-atom
  // route two bands (256 * blocks bytes) and keeps a partial band there between
  // Consume calls, so no other route may use the workspace until it
  // finishes (the writer streams one route at a time); other routes 0.
  std::expected<PartRouteExecution, Error> Start(std::span<std::byte> workspace = {}) const;
  // The least workspace Start accepts (above).
  std::uint64_t workspace_bytes() const;
  format::Encoding source_encoding() const { return source_encoding_; }

 private:
  PartRoutePlan() = default;
  std::uint64_t input_bytes_ = 0, output_bytes_ = 0;
  std::optional<HeadPermutationCursor> permutation_{};
  std::optional<ScalarFold> fold_{};
  format::Encoding source_encoding_ = format::Encoding::kBF16;
  std::uint64_t atom_blocks_ = 0;
  std::uint64_t row_bytes_ = 0, stride_ = 0, lane_ = 0;
};
}  // namespace llmp::import
#endif  // LLMP_IMPORT_ROUTE_H_
