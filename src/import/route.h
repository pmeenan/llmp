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
  std::span<std::byte> workspace_{};
  std::array<std::byte, 2> carry_{};
  bool carried_ = false, failed_ = false, busy_ = false;
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
  // Historical BF16->F32 conversion, not a native resident-weight expansion
  // policy. Reuses FoldBf16's exact numerical and refusal contract.
  static std::expected<PartRoutePlan, Error> Fold(ScalarFold fold, std::uint64_t elements);
  std::uint64_t input_bytes() const { return input_bytes_; }
  std::uint64_t output_bytes() const { return output_bytes_; }
  // No allocation: caller owns/funds workspace and keeps it alive through
  // execution. A fold needs >=4 bytes and uses at most64KiB; other routes 0.
  std::expected<PartRouteExecution, Error> Start(std::span<std::byte> workspace = {}) const;

 private:
  PartRoutePlan() = default;
  std::uint64_t input_bytes_ = 0, output_bytes_ = 0;
  std::optional<HeadPermutationCursor> permutation_{};
  std::optional<ScalarFold> fold_{};
};
}  // namespace llmp::import
#endif  // LLMP_IMPORT_ROUTE_H_
