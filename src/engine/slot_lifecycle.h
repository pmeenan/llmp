// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The shared slot lifecycle (docs/engine.md, "The slot lifecycle";
// docs/engine-components.md, "Execution"): a runner's request slots and
// every verb that moves their state, written once, from the stages' state
// contracts (model/state_contract.h), where each family's runner wrote its
// own copy. Where those copies disagreed, the behaviour here is the
// reconciled one, recorded in engine.md.
//
// A slot holds one LiveState per stage: the target's (stage 0) and each
// drafter's, each a growing region of its stage's derived layout with its
// own spill file. At this interface a LiveState::Range's `region` names the
// stage. The slot also keeps its completed positions, whether it is
// spilled, adopted from a kept record (D-105) or mid-restore, a clean
// capacity refusal, and per stage its borrows (a reader holding that
// stage's state frozen: a drafter reading the target's KV) and what its
// spill file holds (SpillTrack, an incremental spill).
//
// Positions are the slot's committed positions: the target's, by which
// every stage's state is indexed. A step on the target (prefill, decode,
// score, a verify) starts at them; a drafter's step (a draft, which never
// commits) writes only its own stage, from wherever its catch-up starts,
// and never past one chunk bound of its stage beyond them (model_runner.h
// Admit). A verify's rows await Accept, which commits the kept ones;
// DiscardVerify undoes it whole; Settle runs what an accepted one owes.
// Closures come from the request cohort (request_cohort.h) over every
// stage's extents; a spilled slot is in none.
//
// What the runner (or the step orchestration) owes it:
// - RecordWrite before posting any job that may write a slot's state (the
//   first position it writes from, or the ranges a copy writes): an
//   incremental spill writes back only what a record covers.
// - SetPositions after a step that completed positions retired.
// - CheckActive before a slot's work; Fault / CheckFailedJob after a job
//   whose effect it cannot prove local (request_cohort.h's rules).
//
// Part of the engine core: no kernel library, every build profile.

#ifndef LLMP_ENGINE_SLOT_LIFECYCLE_H_
#define LLMP_ENGINE_SLOT_LIFECYCLE_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "base/sha256.h"
#include "catalog/catalog.h"
#include "engine/draft_progress.h"
#include "engine/live_state.h"
#include "engine/paged_node.h"
#include "engine/request_cohort.h"
#include "model/state_contract.h"

namespace llmp::engine {

class PagedWeights;

// One stage's state: the target's or a drafter's.
struct StageState {
  std::string name;  // "target", "mtp", "dspark", "assistant"
  model::StateLayoutDerived layout;
  // Its steps' own read alignment (0: the options'); its context is its
  // layout's.
  std::uint32_t read_align = 0;
};

struct SlotLifecycleOptions {
  std::string model;               // names it in refusals ("Gemma2")
  std::uint32_t slots = 1;         // 1 to kMaxRequestSlots
  std::vector<StageState> stages;  // [0] the target, then each drafter
  base::Sha256Digest spec{};       // the model spec's digest (kept_layout_id)
  std::uint32_t read_align = 256;  // ReserveStateThrough's default
  // Where each slot's stage spills: unnamed files in `spill_directory`
  // unless given.
  std::function<LiveState::SpillPlace(std::uint32_t slot, std::uint32_t stage)> spill_place;
  std::filesystem::path spill_directory;
  // A slot's state no longer matches what its runner derived from it
  // (retained features, a speculation cursor, pending rows): cleared,
  // spilled, restored from a checkpoint, adopted, overwritten by a copy or
  // discarded by a recovery.
  std::function<void(std::uint32_t slot)> invalidate;
  // Every plan and graph is to be dropped: a place moved, a recovery.
  std::function<void()> drop_plans;
  // Logical state was replaced or cleared, rather than merely spilled.
  // Position-to-cell metadata survives an ordinary spill/restore.
  std::function<void(std::uint32_t slot)> content_changed = {};
};

class SlotLifecycle;

// Opaque permission for the first reader behind one exact accepted prefix.
// It grants no ordinary frozen view and cannot be minted by a caller.
class StateBorrowPermit {
 private:
  friend class SlotLifecycle;
  const SlotLifecycle* owner_ = nullptr;
  std::uint32_t slot_ = 0;
  std::size_t reader_ = 0;
  std::uint64_t identity_ = 0;
};

// A borrow of one slot's stage (SlotLifecycle::Borrow): move-only, returned
// once (Return, or its destruction), and before its lifecycle is
// destroyed. A borrow from before a recovery (RecoverInPlace drops every
// borrow) returns nothing.
class StateBorrow {
 public:
  StateBorrow() = default;
  StateBorrow(const StateBorrow&) = delete;
  StateBorrow& operator=(const StateBorrow&) = delete;
  StateBorrow(StateBorrow&& other) noexcept { *this = std::move(other); }
  StateBorrow& operator=(StateBorrow&& other) noexcept;
  ~StateBorrow() { Return(); }
  void Return();
  bool held() const { return owner_ != nullptr; }
  std::uint32_t slot() const { return slot_; }
  std::size_t stage() const { return stage_; }
  std::optional<StateBorrowPermit> permit() const { return permit_; }

 private:
  friend class SlotLifecycle;
  SlotLifecycle* owner_ = nullptr;
  std::uint32_t slot_ = 0;
  std::size_t stage_ = 0;
  std::uint64_t epoch_ = 0;
  std::optional<std::size_t> ordered_reader_;
  std::optional<StateBorrowPermit> permit_;
};

class SlotLifecycle {
 public:
  using Range = LiveState::Range;  // `region`: the stage
  using CopyRetirement = LiveState::CopyRetirement;

  SlotLifecycle(PagedNode& node, SlotLifecycleOptions options, int owner, std::uint32_t stream);
  SlotLifecycle(const SlotLifecycle&) = delete;
  SlotLifecycle& operator=(const SlotLifecycle&) = delete;
  SlotLifecycle(SlotLifecycle&&) = delete;
  SlotLifecycle& operator=(SlotLifecycle&&) = delete;
  // Nothing is released here: Release, after the node's teardown.
  ~SlotLifecycle();

  // ------------------------------------------------------- the runner's life

  // Before the weights' places: every slot's every stage's region.
  Status Setup();
  // Once after setup and before registration, bind the prepared source recipe
  // into the same opaque layout used by checkpoints and durable kept records.
  // A null identity preserves the legacy layout byte for byte. No allocation.
  Status SealExecutionIdentity(const std::optional<base::Sha256Digest>& identity);
  // After the node's Start: each region's spill file. The caller pins
  // reserved_extents() with its weights' (Scheduler::PinPlaces).
  Status Register();
  std::vector<catalog::ExtentId> reserved_extents() const;
  // After the workspace: what every job leases beside the state (the
  // weights, the workspace, the runner's own memory), and the weights
  // CheckPlaces checks. Builds the closures.
  Status Bind(std::vector<catalog::ExtentId> shared, std::vector<const PagedWeights*> weights);
  // Stable-address workspace growth, before any new state/model work. New
  // cataloged extents become shared only after atomic request admission.
  // Capacity refusal preserves the old closures; unknown replacement faults
  // the cohort. No existing shared extent or cached address is removed.
  Status TryExtendShared(std::span<const catalog::ExtentId> additional,
                         bool* over_budget = nullptr);
  // Every region released (problems appended). Once, after teardown.
  void Release(providers::VmmProvider& memory, std::vector<std::string>& problems);

  // ---------------------------------------------------------------- views

  const catalog::Closure& everything() const { return everything_; }
  const catalog::Closure& fence() const { return fence_; }
  const catalog::Closure& execution() const { return execution_; }
  const catalog::Closure& slot_fence(std::uint32_t slot) const;
  std::uint32_t slots() const { return options_.slots; }
  std::size_t stages() const { return options_.stages.size(); }
  const model::StateLayoutDerived& layout(std::size_t stage) const {
    return options_.stages.at(stage).layout;
  }
  // The target's own read alignment, and a stage's (ReserveStateThrough's
  // default for a step on it).
  std::uint32_t read_align() const { return read_align(0); }
  std::uint32_t read_align(std::size_t stage) const {
    const std::uint32_t own = options_.stages.at(stage).read_align;
    return own != 0 ? own : options_.read_align;
  }
  std::uint32_t context(std::size_t stage) const { return layout(stage).context; }
  SlotMask active() const { return cohort_.active(); }
  bool faulted() const { return cohort_.faulted(); }
  // D-105: what a kept record's layout must be (model::KeptLayoutId).
  const std::string& kept_layout_id() const { return kept_layout_id_; }
  // A slot's stage's live state, for its steps (Use is through
  // ReserveStateThrough; a verify's snapshot is the stage's own).
  LiveState& live(std::uint32_t slot, std::size_t stage);
  const LiveState& live(std::uint32_t slot, std::size_t stage) const;
  // None for an ordinary stage. The lifecycle owns transport and readiness;
  // orchestration reports the exact completed replay/compound operations.
  DraftProgressLedger* progress(std::uint32_t slot, std::size_t stage);
  const DraftProgressLedger* progress(std::uint32_t slot, std::size_t stage) const;
  std::expected<DraftProgressLedger::Unit, std::string> BeginDraftUnit(std::uint32_t slot,
                                                                       std::size_t stage);
  std::expected<DraftProgressLedger::Successor, std::string> BeginDraftSuccessor(std::uint32_t slot,
                                                                                 std::size_t stage);
  Status CancelDraftSuccessor(std::uint32_t slot, std::size_t stage,
                              DraftProgressLedger::Unit unit);
  // Check every owner before beginning any owner, after host preflight.
  Status CheckProgressVerify(std::uint32_t slot, std::uint32_t rows) const;
  Status BeginProgressVerify(std::uint32_t slot, std::uint32_t rows);
  // A failed/discarded verify owes restoring the old tap before this
  // logical transition can be forgotten. Unknown writes quarantine it.
  void ProgressVerifyFailed(std::uint32_t slot);
  // Every slot's used extents, kept zeroed backing, and the extents of
  // resident slots an incremental write-back may skip (unchanged since
  // their spill file last held them).
  std::vector<catalog::ExtentId> state() const;
  std::vector<catalog::ExtentId> kept_state() const;
  std::vector<catalog::ExtentId> unchanged_state() const;

  std::uint32_t positions(std::uint32_t slot) const;
  bool spilled(std::uint32_t slot) const;
  bool adopted(std::uint32_t slot) const;
  bool restoring(std::uint32_t slot) const;
  // Some stage is borrowed; the stage is.
  bool borrowed(std::uint32_t slot) const;
  bool borrowed(std::uint32_t slot, std::size_t stage) const;
  // A verify awaits its Accept in some stage; its rows (0: none).
  bool verify_pending(std::uint32_t slot) const;
  std::uint32_t verify_rows(std::uint32_t slot) const;
  // A verify's restore or commit is still owed in some stage.
  bool owed(std::uint32_t slot) const;
  // The last growth or restore was a clean capacity refusal: the state is
  // as it was (with any fresh zero pages), and may be retried once room is
  // made (Llm::StateRefusedFor).
  bool state_refused(std::uint32_t slot) const;
  void ForgetRefusal(std::uint32_t slot);
  // No stage quarantined, the cohort healthy, no restore pending.
  bool usable(std::uint32_t slot) const;
  // A healthy spilled prefix may be hydrated, including a kept adoption's
  // exact pending progress replacement. It is still not execution-ready.
  bool restorable(std::uint32_t slot) const;
  // The request open on the stream leases its state now.
  bool held(std::uint32_t slot) const;
  // D-105: its spill files hold all of its state, settled, as written last
  // (nothing recorded since a spill, a write-back or an adopted restore).
  bool kept_whole(std::uint32_t slot) const;
  std::uint64_t used_bytes(std::uint32_t slot) const;
  std::uint64_t refused_bytes(std::uint32_t slot) const;
  // What its spill files hold while spilled (whole extents).
  std::uint64_t spilled_bytes(std::uint32_t slot) const;
  // What its spill files occupy on disk while it is not spilled: an
  // incremental spill keeps them past a restore (0 while spilled).
  std::uint64_t retained_spill_bytes(std::uint32_t slot) const;
  // What a Spill now would write (0 while spilled).
  std::uint64_t spill_write_bytes(std::uint32_t slot) const;

  // ---------------------------------------------------------------- verbs

  // Between completed units. Refused if it drops a slot whose state has
  // work in flight that needs its closure: a pending restore, a borrow, a
  // verify awaiting its Accept. An unchanged selection inside an open
  // request keeps its closure.
  Status SelectSlots(std::span<const std::uint32_t> slots);
  // Within its own request (several active slots: the held one): its
  // state zeroed for reuse, its peers protected, then discarded; positions
  // and every pending restore, adoption and refusal reset.
  Status Clear(std::uint32_t slot);
  // A slot not leased by the open request: its state discarded.
  Status ClearIdle(std::uint32_t slot);
  // Out of every closure, then what changed since its spill files last
  // held it written back and its backing released. Refused while a
  // restore is pending, a verify awaits its Accept or it is borrowed. A
  // slot whose verify still owes a restore or commit (in any stage) is
  // written whole by the next spill (what runs it may record no write).
  Status Spill(std::uint32_t slot);
  // A slot not spilled gives back the disk its spill files hold
  // (LiveState::ReleaseSpillFile); its state stays, and its next spill
  // writes it whole. Refused while spilled, restoring, borrowed or while a
  // verify awaits its Accept.
  Status ReleaseSpillFiles(std::uint32_t slot);
  // A spilled slot's state back, at the generations it was written at, or
  // an adopted one read from its kept files; a clean capacity refusal
  // leaves it spilled (state_refused). Refused while a restore is pending.
  Status Restore(std::uint32_t slot);
  // Backs what a step on `stage` reads through `positions` positions: a
  // target's step (stage 0), which may write every stage (a drafter's
  // feature taps), backs every stage, the target at `read_align` (0: its
  // own) and each drafter at its own, through its context at most; a
  // drafter's step backs its own stage alone. Refused while a restore is
  // pending, a verify awaits its Accept, or a stage it grows is borrowed.
  Status ReserveStateThrough(std::uint32_t slot, std::uint32_t positions,
                             std::uint32_t read_align = 0, std::size_t stage = 0);
  // Optional growth for a known next chunk (LiveState::Prepare), submitted
  // before the current job: the fresh zero extents of the stages a step on
  // `stage` grows (as ReserveStateThrough) through `positions` positions
  // at `read_align` (0: the stage's own), never published as used
  // (ReserveStateThrough's Use alone does), never recorded as written.
  // True if any stage submitted some; false (nothing submitted) when the
  // slot may not grow now or nothing is eligible. FinishPreparations
  // drains it after the current job's fence, before anything else changes
  // the state.
  std::expected<bool, std::string> PrepareThrough(std::uint32_t slot, std::uint32_t positions,
                                                  std::uint32_t read_align = 0,
                                                  std::size_t stage = 0);
  // After the current job's fence: every slot's every stage's preparation
  // drained (cancelled if `cancel`); the first problem. One that cannot
  // prove its retirement keeps its ticket (LiveState::FinishPreparation).
  Status FinishPreparations(bool cancel = false);
  // A turn checkpoint's pages: every used page later positions may write
  // (CheckpointPages over each stage's CheckpointWrites). Refused unless
  // the slot is active, usable and settled (no verify pending or owed).
  std::expected<std::vector<Range>, std::string> CheckpointRanges(std::uint32_t slot,
                                                                  std::uint32_t positions) const;
  // A restored footprint: whole extents of the stages' layouts in
  // (stage, offset) order, covering what `positions` positions read, empty
  // at none.
  Status ValidateFootprint(std::uint32_t positions, std::span<const Range> footprint) const;
  // A checkpoint's restore, in three steps: Prepare (the layout ID and
  // footprint checked; refused until a verify's owed restore or commit is
  // settled, which would otherwise run over the checkpoint's bytes; the
  // state made exactly the footprint, its peers protected; a clean capacity
  // refusal leaves it as it was, invalidating nothing), Copy each page in
  // (CopyState to the device), then Complete (refused until every page the
  // checkpoint holds was copied in, in order).
  Status PrepareRestore(std::uint32_t slot, std::uint32_t positions,
                        std::span<const Range> footprint, std::string_view source_layout);
  Status CompleteRestore(std::uint32_t slot, std::uint32_t positions);
  // Packed copies between the slot's state and pinned host memory: to the
  // host outside a restore, to the device only within one; refused while a
  // verify awaits its Accept or owes a restore or commit. A failed copy
  // to the device quarantines the slot; one of unproven completion faults
  // the cohort (the caller keeps `pinned` then).
  Status CopyState(std::uint32_t slot, void* pinned, std::span<const Range> ranges, bool to_host,
                   CopyRetirement* retirement = nullptr);
  // A kept conversation (D-105) into an empty, idle slot: its layout ID
  // and footprint checked; spilled until Restore reads it.
  Status Adopt(std::uint32_t slot, std::uint32_t positions, std::span<const Range> footprint,
               std::string_view source_layout);
  // Every weight and state extent still pinned at its registered place
  // (D-090); refused, every plan dropped, if one moved.
  Status CheckPlaces();
  // A swap's write-back completed (`whole`: every resident slot's state is
  // in its spill files), or did not. A slot mid-restore, unusable, or with
  // a verify awaiting its Accept or owing a restore or commit in any stage
  // is not counted as held by its files.
  void StateWrittenBack(bool whole);
  // D-102's hang recovery, rung 2, after the stream was fenced and the
  // launch's retirement proven by the caller, with no request open: each
  // spilled slot untouched since is kept, every slot the failure may have
  // touched discarded (`before_discard` told first: its kept record goes),
  // and so is every slot whose verify awaits its Accept or whose checkpoint
  // restore is pending (the failure may have skipped them), the fault
  // lifted, every borrow dropped, the default request selected, plans
  // dropped. The slots discarded.
  std::expected<SlotMask, std::string> RecoverInPlace(
      const std::function<void(std::uint32_t slot)>& before_discard = {});
  // An incremental spill's split of a stage's extents.
  void SplitForSpill(std::uint32_t slot, std::size_t stage, std::vector<catalog::ExtentId>& written,
                     std::vector<catalog::ExtentId>& unchanged) const;

  // ------------------------------------------------- what steps tell it

  // Before a slot's work: bound, provisioned, resident, active (several
  // only under the held request), every stage usable, no checkpoint
  // restore between its Prepare and Complete.
  Status CheckActive(std::uint32_t slot) const;
  // Before posting a job that may write the slot's state from `position`
  // on (every stage), or a copy that writes `ranges`. Refused for a
  // spilled (or adopted, unread) slot, which no job may write.
  Status RecordWrite(std::uint32_t slot, std::uint32_t position);
  Status RecordWrite(std::uint32_t slot, std::span<const Range> ranges);
  // After a step's completed positions retired. Refused mid-restore or
  // while spilled.
  Status CheckPositions(std::uint32_t slot, std::uint32_t positions) const;
  Status SetPositions(std::uint32_t slot, std::uint32_t positions);
  // Validate every cursor before publishing any owner.
  Status SetPositions(std::span<const std::pair<std::uint32_t, std::uint32_t>> positions);
  // A reader (an assistant drafting from the target's KV) holds a slot's
  // stage frozen until the borrow returns: no growth of that stage, and no
  // clear, spill, restore or copy in of the slot, nor a selection dropping
  // it; its other stages may grow (the reader's own). Refused until a
  // verify is accepted and what it owes has run (the state is not yet the
  // kept rows').
  std::expected<StateBorrow, std::string> Borrow(std::uint32_t slot, std::size_t stage = 0);

  // Reserve the target guard now, but authorize a reader only behind this
  // exact successor's data/control prefix. Ordinary borrowed(stage) remains
  // false until its positive, atomic publication. No fresh allocation.
  std::expected<StateBorrow, std::string> BorrowAfterPrefix(
      std::uint32_t slot, std::size_t reader, const DraftProgressLedger::Successor& successor);
  Status CheckBorrowAfterPrefix(std::uint32_t slot, std::size_t reader,
                                DraftProgressLedger::Unit unit,
                                DraftProgressLedger::Transition predecessor) const;

  std::expected<StateBorrow, std::string> BorrowAfterOwed(std::uint32_t slot, std::size_t reader,
                                                          std::uint32_t past, std::uint32_t rows);
  std::expected<StateBorrow, std::string> BorrowAfterCompoundPrefix(std::uint32_t slot,
                                                                    std::size_t reader,
                                                                    DraftProgressLedger::Unit unit,
                                                                    std::uint32_t past,
                                                                    std::uint32_t rows);
  Status CompanionPrefixQueued(std::uint32_t slot, std::size_t reader,
                               DraftProgressLedger::Unit unit);
  Status CheckBorrowPermit(const StateBorrowPermit& permit, std::uint32_t slot, std::size_t reader,
                           std::uint32_t past, std::uint32_t rows) const;

  // ------------------------------------------------------------ a verify
  //
  // A verify's step (ModelRunner::RunStep, phase kVerify) saves what it
  // writes in each stage (model::WritesFor) and leaves its rows awaiting
  // Accept in every stage it writes (the target, a drafter's feature taps)
  // or that commits, whether or not it saved anything there.

  // Its first `keep` rows (1 to its rows) stay in every stage: the rest's
  // saved bytes are owed back, then each stage's commit, queued before the
  // next job's own work (or now, by Settle); the positions advance by
  // `keep`. Refused, with nothing changed, unless every verifying stage is
  // usable and awaits a verify of the same rows.
  Status Accept(std::uint32_t slot, std::uint32_t keep);
  // The verify undone whole in every stage (nothing kept, nothing
  // committed), now, as a job of its own; the positions stay.
  Status DiscardVerify(std::uint32_t slot);
  // What an accepted verify owes (its restores, then its commits), now, as
  // a job of its own over the execution closure. A failure checks the job
  // (CheckFailedJob); a stage it quarantined faults the cohort (the shared
  // stream's work is unproven).
  Status Settle(std::uint32_t slot);
  Status FlushOwed(std::span<const std::uint32_t> slots, std::string_view boundary);
  // Common boundary for RunStep and standalone Settle. A PlanStep remains
  // alive across prepare/queue/retirement; private control staging cannot
  // be reused until CompleteOwed. All stages' numeric work precedes controls.
  Status PrepareOwed(std::uint32_t slot);
  struct OwedQueued {
    LiveState::Queued result;
    bool wrote = false;
  };
  OwedQueued QueueOwed(std::uint32_t slot, providers::NativeStream stream);
  Status CompleteOwed(std::uint32_t slot, bool success,
                      CopyRetirement retirement = CopyRetirement::kProven);
  // Prevalidate every completed consumer/owner before publishing any mirror.
  Status CompleteOwed(std::span<const std::uint32_t> slots, bool success,
                      CopyRetirement retirement = CopyRetirement::kProven);
  using CompoundConsumer = std::pair<std::size_t, DraftProgressLedger::Unit>;
  struct CompoundTailOwner {
    std::uint32_t slot = 0;
    std::uint16_t closing_stage = 0;
    std::span<const CompoundConsumer> consumers;
  };
  Status PrepareCompoundTail(const CompoundTailOwner& owner);
  Status QueueCompoundTail(const CompoundTailOwner& owner, providers::NativeStream native);
  Status CompleteCompoundTails(std::span<const CompoundTailOwner> owners,
                               JobReport::Retirement retirement);
  // Quarantine exactly these admitted owners after partial/invalid publication.
  void QuarantineOwners(std::span<const std::uint32_t> slots);
  // Every slot quarantined until RecoverInPlace.
  void Fault();
  void AcknowledgeFence() noexcept;
  // After a failed job: request_cohort.h's CheckFailedJob.
  void CheckFailedJob();
  // The closures rebuilt, the held request refreshed to the execution
  // closure over `protect` (default: the active slots).
  Status RefreshClosures(SlotMask protect);
  Status RefreshClosures() { return RefreshClosures(cohort_.active()); }

 private:
  friend class StateBorrow;
  friend class PromptCompanions;
  struct Stage {
    std::unique_ptr<LiveState> live;
    SpillTrack track;
    std::uint32_t borrows = 0;  // includes guards reserved behind an owed prefix
    struct OrderedBorrow {
      std::optional<DraftProgressLedger::Successor> successor;
      std::optional<DraftProgressLedger::Unit> compound;
      std::uint64_t identity = 0, prefix_epoch = 0;
      std::uint32_t positions = 0, past = 0, rows = 0;
      bool active = false, prefix_queued = false;
    };
    std::optional<OrderedBorrow> ordered_borrow;  // this stage reads target stage 0
    // Kept from the process before (D-105): what its spill file holds,
    // until Restore reads it.
    std::vector<Range> adopted;
    std::unique_ptr<DraftProgressLedger> progress;
    std::uint64_t control_offset = 0;
    model::DraftProgressWords* control_publish = nullptr;
    model::DraftProgressWords* control_hydrate = nullptr;
    model::DraftProgressWords* control_tail = nullptr;
    std::optional<DraftProgressLedger::Transition> tail_ticket;
    bool tail_attempted = false, tail_queued = false;
    std::optional<DraftProgressLedger::Transition> control_ticket;
    std::optional<DraftProgressLedger::Transition> restore_ticket;
    std::optional<DraftProgressLedger::Unit> progress_unit;
    bool control_queued = false;
    bool progress_discard = false;
  };
  struct Slot {
    std::uint32_t index = 0;
    std::vector<Stage> stages;
    catalog::Closure fence;
    std::uint32_t positions = 0;
    std::uint64_t accepted_epoch = 0;
    bool spilled = false;
    bool state_refused = false;
    // Set only around the shared compound helper's synchronous steps.
    bool compound_access = false;
    std::uint64_t adopted_bytes = 0;
    std::optional<std::uint32_t> restoring;  // its positions
    std::vector<Range> restore_needed;       // region: the stage
    std::vector<std::uint64_t> restored_bytes;
  };

  std::expected<Slot*, std::string> Find(std::uint32_t slot);
  std::expected<const Slot*, std::string> Find(std::uint32_t slot) const;
  std::vector<LiveState*> AllStates();
  std::vector<LiveState*> StatesOf(Slot& slot);
  // The execution closure and every extent of `slot`'s stages, protected
  // while one stage grows beside another's fresh extents.
  Status KeepFor(const Slot& slot, catalog::Closure& keep);
  void Quarantine(Slot& slot);
  Status ResetSlot(Slot& slot);
  std::unexpected<std::string> Refuse(std::string_view what) const;
  bool Usable(const Slot& slot) const;
  bool Borrowed(const Slot& slot) const;
  // The stages a step on `stage` grows (every one for the target's).
  std::vector<std::size_t> GrownBy(std::size_t stage) const;
  std::expected<StateBorrow, std::string> ReserveOrderedBorrow(
      std::uint32_t slot, std::size_t reader, std::uint32_t past, std::uint32_t rows,
      std::optional<DraftProgressLedger::Successor> successor,
      std::optional<DraftProgressLedger::Unit> compound = std::nullopt);
  void Return(std::uint32_t slot, std::size_t stage, std::uint64_t epoch,
              std::optional<std::size_t> ordered_reader);
  // No stage's verify awaits its Accept or owes a restore or commit.
  bool Settled(const Slot& slot) const;
  bool ProgressBusy(const Slot& slot) const;
  Status BeginProgressReplacement(Slot& slot);
  Status HydrateProgress(Slot& slot, std::uint32_t positions);
  // CheckActive, or within a checkpoint restore (its copies in and its
  // completion).
  Status Ready(std::uint32_t slot, bool within_restore) const;

  PagedNode& node_;
  SlotLifecycleOptions options_;
  int owner_;
  std::uint32_t stream_;
  // Every candidate closure, including the per-slot fences, dies before
  // this replacement payment is refunded.
  std::shared_ptr<const void> shared_payment_;
  RequestCohort cohort_;
  std::vector<Slot> slots_;
  std::vector<catalog::ExtentId> shared_;
  std::vector<const PagedWeights*> weights_;
  std::vector<catalog::ExtentId> progress_staging_;
  catalog::Closure everything_, fence_, execution_;
  std::optional<std::uint64_t> places_clean_;
  std::string kept_layout_id_;
  std::uint64_t borrow_identity_ = 0;
  std::uint64_t borrow_epoch_ = 1;  // bumped by RecoverInPlace
  bool setup_ = false, registered_ = false, bound_ = false, released_ = false;
  bool execution_sealed_ = false;
};

}  // namespace llmp::engine

#endif  // LLMP_ENGINE_SLOT_LIFECYCLE_H_
