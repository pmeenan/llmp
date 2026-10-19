// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "engine/draft_progress.h"

#include <algorithm>
#include <limits>

namespace llmp::engine {
namespace {

std::unexpected<std::string> Refused(const char* what) {
  return std::unexpected(std::string(what));
}

}  // namespace

bool DraftProgressLedger::Current(Unit unit) const {
  return unit_active_ && unit.owner_ == this && unit.epoch_ == unit_epoch_;
}

bool DraftProgressLedger::Current(Transition ticket) const {
  return ticket.owner_ == this && ticket.epoch_ == transition_epoch_;
}

bool DraftProgressLedger::Replacing(Transition ticket) const {
  return state_ == State::kReplacing && Current(ticket);
}

DraftProgressLedger::Transition DraftProgressLedger::Ticket() const {
  Transition ticket;
  ticket.owner_ = this;
  ticket.epoch_ = transition_epoch_;
  return ticket;
}

DraftProgressLedger::Status DraftProgressLedger::NextUnit() {
  if (unit_epoch_ == std::numeric_limits<std::uint64_t>::max())
    return Refused("drafter unit identity cannot be represented");
  ++unit_epoch_;
  return {};
}

DraftProgressLedger::Status DraftProgressLedger::NextTransition() {
  if (next_transition_epoch_ == std::numeric_limits<std::uint64_t>::max())
    return Refused("drafter transition identity cannot be represented");
  transition_epoch_ = ++next_transition_epoch_;
  return {};
}

void DraftProgressLedger::InvalidateUnit() {
  successor_ = false;
  successor_queued_ = false;
  compound_successor_ = false;
  compound_prefix_queued_ = false;
  tail_candidate_.reset();
  tail_epoch_ = 0;
  unit_active_ = false;
  replayed_ = false;
  verify_rows_ = 0;
}

DraftProgressLedger::Status DraftProgressLedger::Need(Unit unit, State state) const {
  if (state_ != state || !Current(unit)) return Refused("drafter progress unit or state differs");
  return {};
}

std::expected<DraftProgressLedger::Unit, std::string> DraftProgressLedger::BeginUnit() {
  if (state_ != State::kReady) return Refused("drafter progress is not ready for a unit");
  if (auto checked = model::CheckDraftProgress(completed_, bounds_); !checked)
    return std::unexpected(checked.error());
  if (auto next = NextUnit(); !next) return std::unexpected(next.error());
  unit_active_ = true;
  replayed_ = false;
  Unit unit;
  unit.owner_ = this;
  unit.epoch_ = unit_epoch_;
  return unit;
}

DraftProgressLedger::Status DraftProgressLedger::CheckUnit(Unit unit) const {
  return Need(unit, State::kReady);
}

std::expected<DraftProgressLedger::Successor, std::string> DraftProgressLedger::BeginSuccessor() {
  if (state_ != State::kDirty || !pending_ || successor_ || compound_successor_)
    return Refused("drafter successor needs one unowned publication prefix");
  auto replay = model::DraftFirstPass(*pending_, bounds_);
  if (!replay) return std::unexpected(replay.error());
  if (auto next = NextUnit(); !next) return std::unexpected(next.error());
  unit_active_ = true;
  successor_ = true;
  replayed_ = false;
  Unit unit;
  unit.owner_ = this;
  unit.epoch_ = unit_epoch_;
  return Successor{.unit = unit, .predecessor = Ticket(), .replay = *replay};
}

DraftProgressLedger::Status DraftProgressLedger::CheckSuccessor(
    Unit unit, Transition predecessor, const model::DraftRows& rows) const {
  if (!successor_ || !Current(unit)) return Refused("drafter successor unit differs");
  if (auto publication = CheckPublication(predecessor); !publication) return publication;
  auto replay = model::DraftFirstPass(*pending_, bounds_);
  if (!replay) return std::unexpected(replay.error());
  if (rows != *replay) return Refused("drafter successor differs from the pending replay");
  return {};
}

DraftProgressLedger::Status DraftProgressLedger::SuccessorQueued(Unit unit, Transition predecessor,
                                                                 const model::DraftRows& rows) {
  if (auto checked = CheckSuccessor(unit, predecessor, rows); !checked) return checked;
  if (successor_queued_) return Refused("drafter successor was already queued");
  successor_queued_ = true;
  return {};
}

DraftProgressLedger::Status DraftProgressLedger::CancelSuccessor(Unit unit) {
  if (state_ != State::kDirty || !successor_ || successor_queued_ || !Current(unit))
    return Refused("drafter successor cancellation differs or has submitted work");
  InvalidateUnit();
  return {};
}

DraftProgressLedger::Status DraftProgressLedger::ReplayCompleted(Unit unit,
                                                                 const model::DraftRows& rows) {
  if (auto ready = Need(unit, State::kReady); !ready) return ready;
  auto first = model::DraftFirstPass(completed_, bounds_);
  if (!first) return std::unexpected(first.error());
  if (rows != *first) return Refused("completed replay differs from the whole pending schedule");
  replayed_ = true;
  return {};
}

DraftProgressLedger::Status DraftProgressLedger::CanVerify(Unit unit, std::uint32_t rows) const {
  if (auto ready = Need(unit, State::kReady); !ready) return ready;
  if (completed_.positions != 0 && !replayed_)
    return Refused("target verification would lose unreplayed feature rows");
  if (rows == 0 || rows > bounds_.max_rows || rows > bounds_.context - completed_.positions)
    return Refused("target verification exceeds drafter progress bounds");
  return {};
}

DraftProgressLedger::Status DraftProgressLedger::BeginVerify(Unit unit, std::uint32_t rows) {
  if (auto checked = CanVerify(unit, rows); !checked) return checked;
  verify_rows_ = rows;
  state_ = State::kVerify;
  return {};
}

std::expected<model::DraftProgress, std::string> DraftProgressLedger::AcceptCandidate(
    Unit unit, std::uint32_t keep) const {
  if (auto ready = Need(unit, State::kVerify); !ready) return std::unexpected(ready.error());
  if (keep == 0 || keep > verify_rows_) return Refused("accepted drafter progress rows differ");
  const std::uint32_t a = bounds_.relation == model::DraftRelation::kTokenAfterCache ? 1U : 0U;
  // The empty target has no prior pending row. Both relations start with
  // their first valid cache coordinate, 1-a.
  const std::uint32_t next = completed_.positions + 1 - a;
  model::DraftProgress p{.positions = completed_.positions + keep,
                         .pending_first = next,
                         .tap_first = completed_.positions,
                         .tap_rows = keep,
                         .preceding = std::min(completed_.tap_rows, bounds_.carry)};
  if (auto checked = model::CheckDraftProgress(p, bounds_); !checked)
    return std::unexpected(checked.error());
  return p;
}

DraftProgressLedger::Status DraftProgressLedger::Accept(Unit unit, std::uint32_t keep) {
  auto next = AcceptCandidate(unit, keep);
  if (!next) return std::unexpected(next.error());
  if (auto identity = NextTransition(); !identity) return identity;
  pending_ = *next;
  state_ = State::kDirty;
  InvalidateUnit();
  return {};
}

DraftProgressLedger::Status DraftProgressLedger::CheckDiscard(Unit unit) const {
  return Need(unit, State::kVerify);
}

DraftProgressLedger::Status DraftProgressLedger::DiscardCompleted(Unit unit) {
  if (auto ready = CheckDiscard(unit); !ready) return ready;
  state_ = State::kReady;
  InvalidateUnit();
  return {};
}

std::expected<model::DraftCompanions, std::string> DraftProgressLedger::PreviewCompound(
    std::uint32_t rows) const {
  const bool behind_prefix =
      state_ == State::kDirty && pending_ && !successor_ && !compound_successor_;
  if (state_ != State::kReady && !behind_prefix)
    return Refused("drafter progress is not ready for an unowned prompt");
  return model::DraftPrefillCompanions(behind_prefix ? *pending_ : completed_, bounds_, rows);
}

std::expected<DraftProgressLedger::Compound, std::string> DraftProgressLedger::BeginCompound(
    std::uint32_t rows) {
  const bool behind_prefix =
      state_ == State::kDirty && pending_ && !successor_ && !compound_successor_;
  const auto base = behind_prefix ? *pending_ : completed_;
  auto shapes = PreviewCompound(rows);
  if (!shapes) return std::unexpected(shapes.error());
  Unit unit;
  if (behind_prefix) {
    if (auto next = NextUnit(); !next) return std::unexpected(next.error());
    unit_active_ = true;
    replayed_ = false;
    unit.owner_ = this;
    unit.epoch_ = unit_epoch_;
    compound_predecessor_ = Ticket();
  } else {
    auto begun = BeginUnit();
    if (!begun) return std::unexpected(begun.error());
    unit = *begun;
  }
  companions_ = *shapes;
  compound_base_ = base;
  compound_rows_ = rows;
  pre_ = shapes->before.count == 0 ? Work::kComplete : Work::kNone;
  target_ = Work::kNone;
  post_ = shapes->after.count == 0 ? Work::kComplete : Work::kNone;
  compound_successor_ = behind_prefix;
  compound_prefix_queued_ = false;
  if (!behind_prefix) state_ = State::kCompound;
  return Compound{.unit = unit, .shapes = *shapes};
}

DraftProgressLedger::Status DraftProgressLedger::CheckCompound(Unit unit) const {
  if (state_ == State::kCompound) return Need(unit, State::kCompound);
  if (state_ != State::kDirty || !compound_successor_ || !Current(unit))
    return Refused("compound progress unit or predecessor differs");
  return CheckPublication(compound_predecessor_);
}

bool DraftProgressLedger::CompoundNeedsPrefix(Unit unit) const {
  return compound_successor_ && Current(unit);
}

DraftProgressLedger::Status DraftProgressLedger::CompoundPrefixQueued(Unit unit) {
  if (auto checked = CheckCompound(unit); !checked) return checked;
  if (!compound_successor_) return {};
  if (compound_prefix_queued_) return Refused("compound predecessor was already queued");
  compound_prefix_queued_ = true;
  return {};
}

DraftProgressLedger::Status DraftProgressLedger::NeedCompoundExecution(Unit unit) const {
  if (auto checked = CheckCompound(unit); !checked) return checked;
  if (compound_successor_ && !compound_prefix_queued_)
    return Refused("compound numeric work has no fully queued predecessor");
  return {};
}

DraftProgressLedger::Status DraftProgressLedger::PreQueued(Unit unit) {
  if (auto ready = NeedCompoundExecution(unit); !ready) return ready;
  if (pre_ != Work::kNone || target_ != Work::kNone) return Refused("pre companion is not next");
  pre_ = Work::kQueued;
  return {};
}

DraftProgressLedger::Status DraftProgressLedger::CheckPreCompletion(Unit unit) const {
  if (auto ready = Need(unit, State::kCompound); !ready) return ready;
  if (pre_ != Work::kQueued) return Refused("pre companion was not queued");
  return {};
}

DraftProgressLedger::Status DraftProgressLedger::PreCompleted(Unit unit) {
  if (auto checked = CheckPreCompletion(unit); !checked) return checked;
  pre_ = Work::kComplete;
  return {};
}

DraftProgressLedger::Status DraftProgressLedger::TargetQueued(Unit unit) {
  if (auto ready = NeedCompoundExecution(unit); !ready) return ready;
  if (pre_ != Work::kComplete || target_ != Work::kNone)
    return Refused("target must follow its completed pre companion");
  target_ = Work::kQueued;
  return {};
}

DraftProgressLedger::Status DraftProgressLedger::CheckTargetCompletion(Unit unit) const {
  if (auto ready = Need(unit, State::kCompound); !ready) return ready;
  if (target_ != Work::kQueued) return Refused("compound target was not queued");
  return {};
}

DraftProgressLedger::Status DraftProgressLedger::TargetCompleted(Unit unit) {
  if (auto checked = CheckTargetCompletion(unit); !checked) return checked;
  target_ = Work::kComplete;
  return {};
}

std::optional<model::DraftProgress> DraftProgressLedger::compound_tap() const {
  if (state_ != State::kCompound || target_ != Work::kComplete) return std::nullopt;
  return model::DraftProgress{.positions = compound_base_.positions + compound_rows_,
                              .pending_first = companions_.final_pending_first,
                              .tap_first = compound_base_.positions,
                              .tap_rows = compound_rows_,
                              .preceding = std::min(compound_base_.tap_rows, bounds_.carry)};
}

DraftProgressLedger::Status DraftProgressLedger::CheckPostCompletion(Unit unit) const {
  if (auto ready = Need(unit, State::kCompound); !ready) return ready;
  if (target_ != Work::kComplete || post_ != Work::kNone)
    return Refused("post companion must follow its completed target");
  return {};
}

DraftProgressLedger::Status DraftProgressLedger::PostCompleted(Unit unit) {
  if (auto checked = CheckPostCompletion(unit); !checked) return checked;
  post_ = Work::kComplete;
  return {};
}

DraftProgressLedger::Status DraftProgressLedger::FinishCompound(Unit unit) {
  if (auto ready = Need(unit, State::kCompound); !ready) return ready;
  if (tail_candidate_) return Refused("compound has a separately reserved final publication");
  if (pre_ != Work::kComplete || target_ != Work::kComplete || post_ != Work::kComplete)
    return Refused("compound prompt has incomplete work");
  model::DraftProgress p{.positions = compound_base_.positions + compound_rows_,
                         .pending_first = companions_.final_pending_first,
                         .tap_first = compound_base_.positions,
                         .tap_rows = compound_rows_,
                         .preceding = std::min(compound_base_.tap_rows, bounds_.carry)};
  if (auto checked = model::CheckDraftProgress(p, bounds_); !checked)
    return std::unexpected(checked.error());
  if (auto identity = NextTransition(); !identity) return identity;
  pending_ = p;
  state_ = State::kDirty;
  InvalidateUnit();
  return {};
}

DraftProgressLedger::Status DraftProgressLedger::RefuseCompound(Unit unit) {
  if (auto ready = CheckCompound(unit); !ready) return ready;
  if (pre_ == Work::kQueued || target_ != Work::kNone ||
      (compound_successor_ && compound_prefix_queued_)) {
    Quarantine();
    return Refused("compound refusal follows submitted prefix, pre work or target/tap mutation");
  }
  // An unused successor leaves its exact old publication owed. A settled
  // compound can return to its completed base without undoing numeric work.
  if (!compound_successor_) state_ = State::kReady;
  InvalidateUnit();
  return {};
}

std::expected<DraftProgressLedger::PublicationCopy, std::string> DraftProgressLedger::Publication()
    const {
  if (state_ != State::kDirty || !pending_) return Refused("no drafter control publication owed");
  auto words = model::EncodeDraftProgress(*pending_, bounds_);
  if (!words) return std::unexpected(words.error());
  return PublicationCopy{.ticket = Ticket(), .words = *words};
}

std::expected<model::DraftProgress, std::string> DraftProgressLedger::CompoundTailCandidate(
    Unit unit, CompoundClose close) const {
  if (auto owned = CheckCompound(unit); !owned) return std::unexpected(owned.error());
  const bool closing_target = close == CompoundClose::kTarget;
  const bool closing_post = close == CompoundClose::kPost;
  if (pre_ != Work::kComplete || target_ != (closing_target ? Work::kNone : Work::kComplete) ||
      post_ != (closing_post ? Work::kNone : Work::kComplete))
    return Refused("compound final publication is not behind its last useful work");
  const model::DraftProgress candidate{
      .positions = compound_base_.positions + compound_rows_,
      .pending_first = companions_.final_pending_first,
      .tap_first = compound_base_.positions,
      .tap_rows = compound_rows_,
      .preceding = std::min(compound_base_.tap_rows, bounds_.carry)};
  if (auto valid = model::CheckDraftProgress(candidate, bounds_); !valid)
    return std::unexpected(valid.error());
  return candidate;
}

std::expected<DraftProgressLedger::PublicationCopy, std::string>
DraftProgressLedger::PrepareCompoundTail(Unit unit, CompoundClose close) {
  auto candidate = CompoundTailCandidate(unit, close);
  if (!candidate) return std::unexpected(candidate.error());
  if (tail_candidate_) return Refused("compound final publication already has an owner");
  auto words = model::EncodeDraftProgress(*candidate, bounds_);
  if (!words) return std::unexpected(words.error());
  if (next_transition_epoch_ == std::numeric_limits<std::uint64_t>::max())
    return Refused("drafter final publication identity cannot be represented");
  // Issuance cannot invalidate an old publication still queued as our prefix.
  tail_epoch_ = ++next_transition_epoch_;
  tail_candidate_ = *candidate;
  tail_close_ = close;
  Transition ticket;
  ticket.owner_ = this;
  ticket.epoch_ = tail_epoch_;
  return PublicationCopy{.ticket = ticket, .words = *words};
}

DraftProgressLedger::Status DraftProgressLedger::CheckCompoundTailQueued(
    Unit unit, Transition ticket, CompoundClose close) const {
  if (auto owned = NeedCompoundExecution(unit); !owned) return owned;
  if (!tail_candidate_ || ticket.owner_ != this || ticket.epoch_ != tail_epoch_ ||
      close != tail_close_)
    return Refused("compound final publication ticket or closing work differs");
  const bool closing_target = close == CompoundClose::kTarget;
  const bool closing_post = close == CompoundClose::kPost;
  if (pre_ != Work::kComplete || target_ != (closing_target ? Work::kQueued : Work::kComplete) ||
      post_ != (closing_post ? Work::kNone : Work::kComplete))
    return Refused("compound final control cannot precede its numeric work");
  return {};
}

DraftProgressLedger::Status DraftProgressLedger::CheckCompoundTailRetired(
    Unit unit, Transition ticket, CompoundClose close) const {
  if (auto owned = Need(unit, State::kCompound); !owned) return owned;
  return CheckCompoundTailQueued(unit, ticket, close);
}

std::expected<std::uint32_t, std::string> DraftProgressLedger::CompoundTailPositions(
    Unit unit, Transition ticket) const {
  if (auto owned = CheckCompound(unit); !owned) return std::unexpected(owned.error());
  if (!tail_candidate_ || ticket.owner_ != this || ticket.epoch_ != tail_epoch_)
    return Refused("compound final positions have a different publication owner");
  return tail_candidate_->positions;
}

DraftProgressLedger::Status DraftProgressLedger::CheckCompoundTailCursor(
    Unit unit, Transition ticket, std::uint32_t positions) const {
  if (auto final = CompoundTailPositions(unit, ticket); !final)
    return std::unexpected(final.error());
  if (positions != compound_base_.positions)
    return Refused("compound final publication cursor differs from its original base");
  return {};
}

DraftProgressLedger::Status DraftProgressLedger::CompleteCompoundTail(Unit unit, Transition ticket,
                                                                      CompoundClose close) {
  if (auto retired = CheckCompoundTailRetired(unit, ticket, close); !retired) return retired;
  // Every owner/consumer was preflighted before this deterministic transition.
  if (close == CompoundClose::kTarget) target_ = Work::kComplete;
  if (close == CompoundClose::kPost) post_ = Work::kComplete;
  completed_ = *tail_candidate_;
  transition_epoch_ = tail_epoch_;
  pending_.reset();
  state_ = State::kReady;
  InvalidateUnit();
  return {};
}

DraftProgressLedger::Status DraftProgressLedger::CheckPublication(Transition ticket) const {
  if (state_ != State::kDirty || !pending_ || !Current(ticket))
    return Refused("drafter publication completion differs from its transition");
  return {};
}

DraftProgressLedger::Status DraftProgressLedger::PublicationCompleted(Transition ticket) {
  if (auto checked = CheckPublication(ticket); !checked) return checked;
  // A reserved successor keeps its distinct unit identity. Publication
  // alone never witnesses its replay or makes a verify legal.
  successor_ = false;
  completed_ = *pending_;
  pending_.reset();
  state_ = compound_successor_ ? State::kCompound : State::kReady;
  compound_successor_ = false;
  compound_prefix_queued_ = false;
  return {};
}

void DraftProgressLedger::Quarantine() {
  state_ = State::kQuarantined;
  InvalidateUnit();
}

std::expected<DraftProgressLedger::Transition, std::string>
DraftProgressLedger::BeginReplacement() {
  if (state_ != State::kReady) return Refused("drafter progress replacement needs settled state");
  if (auto next = NextTransition(); !next) return std::unexpected(next.error());
  state_ = State::kReplacing;
  InvalidateUnit();
  return Ticket();
}

DraftProgressLedger::Status DraftProgressLedger::RestoreCompleted(
    Transition ticket, const model::DraftProgressWords& words, std::uint32_t positions) {
  if (state_ != State::kReplacing || !Current(ticket))
    return Refused("drafter restore completion differs from its transition");
  auto restored = model::DecodeDraftProgress(words, positions, bounds_);
  if (!restored) {
    Quarantine();
    return std::unexpected(restored.error());
  }
  completed_ = *restored;
  state_ = State::kReady;
  return {};
}

std::expected<DraftProgressLedger::Transition, std::string> DraftProgressLedger::BeginClear() {
  if (state_ == State::kVerify || state_ == State::kCompound || state_ == State::kClearing ||
      successor_ || compound_successor_)
    return Refused("drafter clear conflicts with an owned transition");
  if (auto next = NextTransition(); !next) return std::unexpected(next.error());
  state_ = State::kClearing;
  InvalidateUnit();
  return Ticket();
}

DraftProgressLedger::Status DraftProgressLedger::ClearCompleted(Transition ticket) {
  if (state_ != State::kClearing || !Current(ticket))
    return Refused("drafter clear completion differs from its transition");
  if (auto checked = model::CheckDraftProgress({}, bounds_); !checked)
    return std::unexpected(checked.error());
  completed_ = {};
  pending_.reset();
  state_ = State::kReady;
  InvalidateUnit();
  return {};
}

}  // namespace llmp::engine
