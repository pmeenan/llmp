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
  return unit_active_ && unit.owner_ == this && unit.epoch_ == epoch_;
}

bool DraftProgressLedger::Current(Transition ticket) const {
  return ticket.owner_ == this && ticket.epoch_ == epoch_;
}

DraftProgressLedger::Transition DraftProgressLedger::Ticket() const {
  Transition ticket;
  ticket.owner_ = this;
  ticket.epoch_ = epoch_;
  return ticket;
}

DraftProgressLedger::Status DraftProgressLedger::NextEpoch() {
  if (epoch_ == std::numeric_limits<std::uint64_t>::max())
    return Refused("drafter transition identity cannot be represented");
  ++epoch_;
  return {};
}

void DraftProgressLedger::InvalidateUnit() {
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
  if (auto next = NextEpoch(); !next) return std::unexpected(next.error());
  unit_active_ = true;
  replayed_ = false;
  Unit unit;
  unit.owner_ = this;
  unit.epoch_ = epoch_;
  return unit;
}

DraftProgressLedger::Status DraftProgressLedger::CheckUnit(Unit unit) const {
  return Need(unit, State::kReady);
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
  pending_ = *next;
  state_ = State::kDirty;
  InvalidateUnit();
  return {};
}

DraftProgressLedger::Status DraftProgressLedger::DiscardCompleted(Unit unit) {
  if (auto ready = Need(unit, State::kVerify); !ready) return ready;
  state_ = State::kReady;
  InvalidateUnit();
  return {};
}

std::expected<DraftProgressLedger::Compound, std::string> DraftProgressLedger::BeginCompound(
    std::uint32_t rows) {
  if (state_ != State::kReady) return Refused("drafter progress is not ready for a prompt");
  auto shapes = model::DraftPrefillCompanions(completed_, bounds_, rows);
  if (!shapes) return std::unexpected(shapes.error());
  auto unit = BeginUnit();
  if (!unit) return std::unexpected(unit.error());
  companions_ = *shapes;
  compound_rows_ = rows;
  pre_ = shapes->before.count == 0 ? Work::kComplete : Work::kNone;
  target_ = Work::kNone;
  post_ = shapes->after.count == 0 ? Work::kComplete : Work::kNone;
  state_ = State::kCompound;
  return Compound{.unit = *unit, .shapes = *shapes};
}

DraftProgressLedger::Status DraftProgressLedger::PreQueued(Unit unit) {
  if (auto ready = Need(unit, State::kCompound); !ready) return ready;
  if (pre_ != Work::kNone || target_ != Work::kNone) return Refused("pre companion is not next");
  pre_ = Work::kQueued;
  return {};
}

DraftProgressLedger::Status DraftProgressLedger::PreCompleted(Unit unit) {
  if (auto ready = Need(unit, State::kCompound); !ready) return ready;
  if (pre_ != Work::kQueued) return Refused("pre companion was not queued");
  pre_ = Work::kComplete;
  return {};
}

DraftProgressLedger::Status DraftProgressLedger::TargetQueued(Unit unit) {
  if (auto ready = Need(unit, State::kCompound); !ready) return ready;
  if (pre_ != Work::kComplete || target_ != Work::kNone)
    return Refused("target must follow its completed pre companion");
  target_ = Work::kQueued;
  return {};
}

DraftProgressLedger::Status DraftProgressLedger::TargetCompleted(Unit unit) {
  if (auto ready = Need(unit, State::kCompound); !ready) return ready;
  if (target_ != Work::kQueued) return Refused("compound target was not queued");
  target_ = Work::kComplete;
  return {};
}

std::optional<model::DraftProgress> DraftProgressLedger::compound_tap() const {
  if (state_ != State::kCompound || target_ != Work::kComplete) return std::nullopt;
  return model::DraftProgress{.positions = completed_.positions + compound_rows_,
                              .pending_first = companions_.final_pending_first,
                              .tap_first = completed_.positions,
                              .tap_rows = compound_rows_,
                              .preceding = std::min(completed_.tap_rows, bounds_.carry)};
}

DraftProgressLedger::Status DraftProgressLedger::PostCompleted(Unit unit) {
  if (auto ready = Need(unit, State::kCompound); !ready) return ready;
  if (target_ != Work::kComplete || post_ != Work::kNone)
    return Refused("post companion must follow its completed target");
  post_ = Work::kComplete;
  return {};
}

DraftProgressLedger::Status DraftProgressLedger::FinishCompound(Unit unit) {
  if (auto ready = Need(unit, State::kCompound); !ready) return ready;
  if (pre_ != Work::kComplete || target_ != Work::kComplete || post_ != Work::kComplete)
    return Refused("compound prompt has incomplete work");
  model::DraftProgress p{.positions = completed_.positions + compound_rows_,
                         .pending_first = companions_.final_pending_first,
                         .tap_first = completed_.positions,
                         .tap_rows = compound_rows_,
                         .preceding = std::min(completed_.tap_rows, bounds_.carry)};
  if (auto checked = model::CheckDraftProgress(p, bounds_); !checked)
    return std::unexpected(checked.error());
  pending_ = p;
  state_ = State::kDirty;
  InvalidateUnit();
  return {};
}

DraftProgressLedger::Status DraftProgressLedger::RefuseCompound(Unit unit) {
  if (auto ready = Need(unit, State::kCompound); !ready) return ready;
  if (pre_ == Work::kQueued || target_ != Work::kNone) {
    Quarantine();
    return Refused("compound refusal follows unproven pre work or target/tap mutation");
  }
  state_ = State::kReady;
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

DraftProgressLedger::Status DraftProgressLedger::PublicationCompleted(Transition ticket) {
  if (state_ != State::kDirty || !pending_ || !Current(ticket))
    return Refused("drafter publication completion differs from its transition");
  completed_ = *pending_;
  pending_.reset();
  state_ = State::kReady;
  return {};
}

void DraftProgressLedger::Quarantine() {
  state_ = State::kQuarantined;
  InvalidateUnit();
}

std::expected<DraftProgressLedger::Transition, std::string>
DraftProgressLedger::BeginReplacement() {
  if (state_ != State::kReady) return Refused("drafter progress replacement needs settled state");
  if (auto next = NextEpoch(); !next) return std::unexpected(next.error());
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
  if (state_ == State::kVerify || state_ == State::kCompound || state_ == State::kClearing)
    return Refused("drafter clear conflicts with an owned transition");
  if (auto next = NextEpoch(); !next) return std::unexpected(next.error());
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
