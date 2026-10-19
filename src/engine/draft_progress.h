// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// One slot/stage's host mirror of its ordinary drafter progress control.
// SlotLifecycle owns this ledger and its transport: a method named
// Completed is called only after the corresponding device work retires.
// No queueing method publishes a completed record or makes it exportable.

#ifndef LLMP_ENGINE_DRAFT_PROGRESS_H_
#define LLMP_ENGINE_DRAFT_PROGRESS_H_

#include <cstdint>
#include <expected>
#include <optional>
#include <string>

#include "model/draft_progress.h"

namespace llmp::engine {

class DraftProgressLedger {
 public:
  using Status = std::expected<void, std::string>;
  enum class State : std::uint8_t {
    kReady,
    kVerify,
    kDirty,
    kCompound,
    kReplacing,
    kClearing,
    kQuarantined
  };

  // A transient capability tied to this owner/stage and logical unit.
  // It contains no persisted identity and cannot be built by the caller.
  class Unit {
   public:
    Unit() = default;

   private:
    friend class DraftProgressLedger;
    const DraftProgressLedger* owner_ = nullptr;
    std::uint64_t epoch_ = 0;
  };

  // The exact pending publication, replacement or clear whose completion
  // is being reported. A late completion from an earlier generation must
  // not publish a newer transition, even at the same target position.
  class Transition {
   public:
    Transition() = default;

   private:
    friend class DraftProgressLedger;
    const DraftProgressLedger* owner_ = nullptr;
    std::uint64_t epoch_ = 0;
  };

  explicit DraftProgressLedger(model::DraftProgressBounds bounds) : bounds_(bounds) {}
  DraftProgressLedger(const DraftProgressLedger&) = delete;
  DraftProgressLedger& operator=(const DraftProgressLedger&) = delete;
  DraftProgressLedger(DraftProgressLedger&&) = delete;
  DraftProgressLedger& operator=(DraftProgressLedger&&) = delete;

  State state() const { return state_; }
  bool settled() const { return state_ == State::kReady; }
  bool owed() const { return state_ == State::kDirty; }
  // The completed record, also during a compound flow. Being able to read
  // this mirror does not authorize exporting unsettled device state.
  const model::DraftProgress& completed() const { return completed_; }
  // For preparing a step whose job first settles all owed work. This is
  // not publication/readiness: the pending record becomes usable only
  // after the ordered data/control copies complete.
  const model::DraftProgress& after_owed() const { return pending_ ? *pending_ : completed_; }

  std::expected<Unit, std::string> BeginUnit();
  Status CheckUnit(Unit unit) const;
  struct Successor {
    Unit unit;
    Transition predecessor;
    model::DraftRows replay;
  };
  // Reserve the next replay against the unpublished record without making
  // it ready. It may execute only behind this exact publication prefix.
  std::expected<Successor, std::string> BeginSuccessor();
  Status CheckSuccessor(Unit unit, Transition predecessor, const model::DraftRows& rows) const;
  // Before its ordered prefix/own job is submitted. Cancellation then needs
  // real retirement; a queued successor cannot be revoked as unused.
  Status SuccessorQueued(Unit unit, Transition predecessor, const model::DraftRows& rows);
  // Only before any successor work is queued. Leaves the predecessor owed.
  Status CancelSuccessor(Unit unit);

  Status ReplayCompleted(Unit unit, const model::DraftRows& rows);
  Status CanVerify(Unit unit, std::uint32_t rows) const;
  // Before target verification can overwrite its consumers' old taps.
  Status BeginVerify(Unit unit, std::uint32_t rows);
  // Prevalidate every stage's candidate before mutating any stage on Accept.
  std::expected<model::DraftProgress, std::string> AcceptCandidate(Unit unit,
                                                                   std::uint32_t keep) const;
  Status Accept(Unit unit, std::uint32_t keep);
  // The old tap bytes have been restored. Retry requires a new unit/replay.
  Status CheckDiscard(Unit unit) const;
  Status DiscardCompleted(Unit unit);

  struct Compound {
    Unit unit;
    model::DraftCompanions shapes;
  };
  // Acquires compound ownership before any pre companion runs. An owed
  // predecessor stays unpublished until this unit's first useful job retires.
  std::expected<model::DraftCompanions, std::string> PreviewCompound(std::uint32_t rows) const;
  std::expected<Compound, std::string> BeginCompound(std::uint32_t rows);
  Status CheckCompound(Unit unit) const;
  bool compound_owned() const { return state_ == State::kCompound || compound_successor_; }
  bool CompoundNeedsPrefix(Unit unit) const;
  Status CompoundPrefixQueued(Unit unit);
  Status PreQueued(Unit unit);
  Status CheckPreCompletion(Unit unit) const;
  Status PreCompleted(Unit unit);
  // Before the first target or leading-carry write, not just its kernel.
  Status TargetQueued(Unit unit);
  Status CheckTargetCompletion(Unit unit) const;
  Status TargetCompleted(Unit unit);
  // Only the compound's private post companion may read these new taps.
  std::optional<model::DraftProgress> compound_tap() const;
  Status CheckPostCompletion(Unit unit) const;
  Status PostCompleted(Unit unit);
  Status FinishCompound(Unit unit);
  // Only known completed/no pre work and zero target/tap writes permit
  // retry. Other failures quarantine; this bounded flow has no rollback.
  Status RefuseCompound(Unit unit);

  struct PublicationCopy {
    Transition ticket;
    model::DraftProgressWords words;
  };
  std::expected<PublicationCopy, std::string> Publication() const;
  enum class CompoundClose : std::uint8_t { kTarget, kPost, kEarlier };
  // A tail is prepared before its final useful job, with a separate identity
  // from any predecessor publication in the same job. Preparation is not
  // completion, and never changes the completed record.
  std::expected<model::DraftProgress, std::string> CompoundTailCandidate(Unit unit,
                                                                         CompoundClose close) const;
  std::expected<PublicationCopy, std::string> PrepareCompoundTail(Unit unit, CompoundClose close);
  Status CheckCompoundTailQueued(Unit unit, Transition ticket, CompoundClose close) const;
  Status CheckCompoundTailRetired(Unit unit, Transition ticket, CompoundClose close) const;
  std::expected<std::uint32_t, std::string> CompoundTailPositions(Unit unit,
                                                                  Transition ticket) const;
  Status CheckCompoundTailCursor(Unit unit, Transition ticket, std::uint32_t positions) const;
  Status CompleteCompoundTail(Unit unit, Transition ticket, CompoundClose close);
  // Completion of every data restore/commit, then the control copy.
  Status CheckPublication(Transition ticket) const;
  Status PublicationCompleted(Transition ticket);
  void Quarantine();

  // After replacement admission succeeds, before copying any new bytes.
  // A pre-admission refusal leaves the completed mirror untouched.
  std::expected<Transition, std::string> BeginReplacement();
  // The exact pending replacement; this never authorizes execution.
  bool Replacing(Transition ticket) const;
  // Only after actual materialization + the leased D2H control read retire.
  Status RestoreCompleted(Transition ticket, const model::DraftProgressWords& words,
                          std::uint32_t positions);
  // Before a clear that the lifecycle has admitted and can retire. A
  // verify/compound operation still owns the state and cannot be cleared.
  std::expected<Transition, std::string> BeginClear();
  // After the existing clear/zero job completed. Does not mean spill.
  Status ClearCompleted(Transition ticket);

 private:
  enum class Work : std::uint8_t { kNone, kQueued, kComplete };
  bool Current(Unit unit) const;
  bool Current(Transition ticket) const;
  Transition Ticket() const;
  Status NextUnit();
  Status NextTransition();
  void InvalidateUnit();
  Status Need(Unit unit, State state) const;
  Status NeedCompoundExecution(Unit unit) const;

  model::DraftProgressBounds bounds_;
  model::DraftProgress completed_{};
  std::optional<model::DraftProgress> pending_;
  State state_ = State::kReady;
  std::uint64_t unit_epoch_ = 0, transition_epoch_ = 0, next_transition_epoch_ = 0;
  std::optional<model::DraftProgress> tail_candidate_;
  std::uint64_t tail_epoch_ = 0;
  CompoundClose tail_close_ = CompoundClose::kTarget;
  bool successor_ = false, successor_queued_ = false;
  bool compound_successor_ = false, compound_prefix_queued_ = false;
  Transition compound_predecessor_;
  model::DraftProgress compound_base_{};
  bool unit_active_ = false;
  bool replayed_ = false;
  std::uint32_t verify_rows_ = 0;
  std::uint32_t compound_rows_ = 0;
  model::DraftCompanions companions_{};
  Work pre_ = Work::kNone, target_ = Work::kNone, post_ = Work::kNone;
};

}  // namespace llmp::engine

#endif  // LLMP_ENGINE_DRAFT_PROGRESS_H_
