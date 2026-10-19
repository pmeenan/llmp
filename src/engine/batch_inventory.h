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
    return index < 64 && ((fallback_mask_ >> index) & 1U) != 0;
  }
  // No state is changed and no rights are minted. A competing operation or
  // changed source requires a new preview before any compound Begin/progress.
  Status Revalidate() const;
  Status CheckSource() const;
  std::expected<model::DraftRows, std::string> CompanionRows(std::uint32_t slot, std::size_t stage,
                                                             std::uint32_t past,
                                                             std::uint32_t rows) const;
  std::uint64_t paid_bytes() const { return payment_.bytes(); }

 private:
  struct Consumer {
    std::uint32_t slot = 0, past = 0, rows = 0;
    std::size_t stage = 0;
    DraftProgressLedger::State state = DraftProgressLedger::State::kReady;
    model::DraftProgress completed, after_owed;
    model::DraftCompanions companions;
  };
  void Swap(PromptBatchInventory& other) noexcept {
    std::swap(payment_, other.payment_);
    std::swap(owner_, other.owner_);
    std::swap(identity_, other.identity_);
    std::swap(fallback_mask_, other.fallback_mask_);
    shapes_.swap(other.shapes_);
    consumers_.swap(other.consumers_);
    target_.swap(other.target_);
  }
  // Every copied descriptor/segment is destroyed before its original payment.
  HostPayment payment_;
  RunnerOwner owner_;
  std::optional<base::Sha256Digest> identity_;
  std::uint64_t fallback_mask_ = 0;
  std::vector<planner::StepShape> shapes_;
  std::vector<Consumer> consumers_;
  std::vector<PendingWork> target_;
};
}  // namespace llmp::engine
#endif  // LLMP_ENGINE_BATCH_INVENTORY_H_
