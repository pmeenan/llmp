// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef LLMP_ENGINE_BATCH_INVENTORY_H_
#define LLMP_ENGINE_BATCH_INVENTORY_H_

#include <cstddef>
#include <cstdint>
#include <expected>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "engine/wave_composer.h"

namespace llmp::engine {

// A paid, immutable resource preview, never an execution/ledger permission.
// Includes every actual pre/target/post composition and its singleton failure
// alternatives. Actual minting must still authenticate each ledger unit.
class PromptBatchInventory {
 public:
  static std::expected<PromptBatchInventory, kernels::ResourceFailure> Preview(
      ModelRunner& runner, const WaveComposer& composer, std::span<const PendingWork> target);
  PromptBatchInventory() = default;
  PromptBatchInventory(const PromptBatchInventory&) = delete;
  PromptBatchInventory& operator=(const PromptBatchInventory&) = delete;
  PromptBatchInventory(PromptBatchInventory&& other) noexcept { Swap(other); }
  PromptBatchInventory& operator=(PromptBatchInventory&& other) noexcept {
    PromptBatchInventory replacement(std::move(other));
    Swap(replacement);
    return *this;
  }
  std::span<const planner::StepShape> shapes() const { return shapes_; }
  bool fallback(std::size_t index) const {
    return index < fallback_.size() && fallback_[index] != 0;
  }
  // No state is changed and no rights are minted. A competing operation or
  // changed source requires a new preview.
  Status Revalidate() const;
  Status CheckSource() const;
  std::uint64_t paid_bytes() const { return payment_.bytes(); }

 private:
  void Swap(PromptBatchInventory& other) noexcept {
    std::swap(payment_, other.payment_);
    std::swap(owner_, other.owner_);
    std::swap(identity_, other.identity_);
    fallback_.swap(other.fallback_);
    shapes_.swap(other.shapes_);
    target_.swap(other.target_);
  }
  // Every copied descriptor/segment is destroyed before its original payment.
  HostPayment payment_;
  RunnerOwner owner_;
  std::optional<base::Sha256Digest> identity_;
  std::vector<std::uint8_t> fallback_;
  std::vector<planner::StepShape> shapes_;
  std::vector<PendingWork> target_;
};
}  // namespace llmp::engine
#endif  // LLMP_ENGINE_BATCH_INVENTORY_H_
