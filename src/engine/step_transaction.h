// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef LLMP_ENGINE_STEP_TRANSACTION_H_
#define LLMP_ENGINE_STEP_TRANSACTION_H_

#include <array>
#include <cstdint>
#include <expected>
#include <limits>
#include <memory>
#include <span>
#include <utility>
#include <vector>

#include "engine/model_runner.h"

namespace llmp::engine {

// One synchronous step's permission and admission owner. Submitted witnesses
// remain linked to the runner until positive retirement or its drained Release.
// Destruction releases paid metadata; it never rolls back submitted state.
class StepTransaction {
 public:
  static std::expected<std::shared_ptr<StepTransaction>, StepRefusal> Begin(
      ModelRunner& runner, const StepRequest& request);
  static Status PrepareBatch(PromptBatch& batch, std::span<const PendingWork> pending,
                             std::span<const SegmentWork> work,
                             std::span<const std::span<const std::int32_t>> histories,
                             std::span<const std::byte> step_bytes, const BatchAhead& ahead,
                             bool* over_budget);
  StepTransaction(const StepTransaction&) = delete;
  StepTransaction& operator=(const StepTransaction&) = delete;
  ~StepTransaction();

  std::span<const StepAuthorization> authorization() const {
    return {permits_.get(), permits_count_};
  }
  std::expected<StepReport, StepRefusal> Run(const StepRequest& request);
  std::expected<void, StepRefusal> Admit(const StepRequest& request);
  std::expected<void, StepRefusal> Revalidate(const StepRequest& request) const {
    return ValidateAuthorization(request, authorization());
  }
  Status FinishPreparations();
  void Queued() {
    if (retirement_ != Retirement::kUnproven) retirement_ = Retirement::kQueued;
  }
  void JobRetired(const JobReport& report);
  bool unproven() const { return retirement_ == Retirement::kUnproven; }
  bool releasable() const {
    return retirement_ == Retirement::kUnused || retirement_ == Retirement::kRetired;
  }

 private:
  struct Frame;
  struct Lookahead;
  struct LookaheadDelete {
    void operator()(Lookahead*) const;
  };
  std::expected<void, StepRefusal> Plan(Frame& frame);
  std::expected<void, StepRefusal> PlanRows(Frame& frame);
  std::expected<void, StepRefusal> Stage(Frame& frame);
  std::expected<void, StepRefusal> PrepareLookahead(Frame& frame);
  std::expected<void, StepRefusal> PrepareState(Frame& frame);
  std::expected<void, StepRefusal> PrepareFutureState(Frame& frame);
  void Queue(Frame& frame);
  std::expected<void, StepRefusal> Settle(Frame& frame);
  std::expected<StepReport, StepRefusal> Publish(Frame& frame);
  enum class Retirement : std::uint8_t { kUnused, kQueued, kRetired, kUnproven };
  explicit StepTransaction(ModelRunner& runner);
  bool PreparedInputsMatch(const StepRequest& request) const;
  // The side-table rows `input` reads for `request`'s rows, derived from
  // each owner's token history (what a step reads of its histories).
  std::expected<void, StepRefusal> DeriveSideRows(const StepRequest& request,
                                                  const kernels::InputSlot& input,
                                                  std::vector<std::int32_t>& ids) const;
  std::expected<void, StepRefusal> Validate(const StepRequest& request) const;
  std::expected<void, StepRefusal> ValidateAuthorization(
      const StepRequest& request, std::span<const StepAuthorization> authorization) const;

  ModelRunner& runner_;
  HostPayment frame_payment_;
  PromptBatch batch_;
  std::unique_ptr<Frame> prepared_;
  std::size_t permits_capacity_ = 0;
  std::size_t batch_index_ = 0;
  bool ran_ = false;
  std::unique_ptr<Lookahead, LookaheadDelete> lookahead_;
  std::uint64_t charge_ = 0;
  std::size_t permits_count_ = 0;
  std::unique_ptr<StepAuthorization[]> permits_;
  // Existing routed executions authenticate a complete environment owner.
  // Other runners retain their existing drained-Release lifetime contract.
  std::shared_ptr<const void> routed_owner_;
  Retirement retirement_ = Retirement::kUnused;
};

}  // namespace llmp::engine

#endif  // LLMP_ENGINE_STEP_TRANSACTION_H_
