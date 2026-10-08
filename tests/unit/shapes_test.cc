// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// D-068's shapes expressed, not executed, in every build profile: request
// programs planned from phase kinds, state representations, decoding modes
// and composed contexts (execution/program.h, model/), then driven through
// the real admission, commitment ledger, catalog and materialization code,
// with the fake device-memory provider backing every extent the catalog
// counts. No kernel runs: a phase leases its closure and working set,
// changes state and output as its outcome says, and ends.
//
// The scenarios:
//   - a stored draft layer: drafts rejected in part and rolled back, the
//     committed prefix never truncated; a draft paused or cancelled
//     before its verify commits nothing;
//   - paged KV, snapshot-only recurrent state and a drafter's own KV
//     through full verify cycles at every acceptance count, paused after
//     one, narrowing to the output bound, and a failed verify rolled back
//     through the snapshot at the prefix;
//   - a companion drafter in a second artifact: one closure over both,
//     its shared embeddings leased and charged once;
//   - a 256-position canvas denoised over several steps, paused for an
//     interactive request between steps: the canvas stays in R(G), so the
//     substitute's phase fits and nothing waits in a circle; charged to
//     the phase instead, the same pause would deadlock;
//   - block output committed all or none, and a failed commit on state
//     that cannot truncate ending its request.
// After every step each asserts D-050's guarantee: the ledger's inequality
// holds; the catalog's occupancy equals the fake's backing and stays within
// the budget; everything but reclaimable cache fits within the
// commitments and the envelopes of the phase kinds in flight; each
// request's retained state (blocks and snapshots) stays within its
// itemized R_i, never below its committed prefix's blocks, and its running
// phase within its kind's envelope.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <initializer_list>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "base/bytes.h"
#include "base/sha256.h"
#include "catalog/catalog.h"
#include "execution/program.h"
#include "expected_error.h"
#include "memory/commitment.h"
#include "memory/materialize.h"
#include "model/context.h"
#include "model/state.h"
#include "providers/device_memory.h"
#include "providers/fake/fake_device_memory.h"
#include "scheduler/admission.h"

namespace {

using llmp::base::Bytes;
using llmp::base::Sha256Digest;
using llmp::catalog::Catalog;
using llmp::catalog::Closure;
using llmp::catalog::DomainId;
using llmp::catalog::ExtentId;
using llmp::catalog::LeaseId;
using llmp::catalog::MemoryClass;
using llmp::catalog::Recovery;
using llmp::catalog::ResourceId;
using llmp::execution::DecodingMode;
using llmp::execution::OutputBuffer;
using llmp::execution::OutputUnit;
using llmp::execution::PhaseKind;
using llmp::execution::PhaseWidth;
using llmp::execution::PlanProgram;
using llmp::execution::ProgramContract;
using llmp::execution::ProgramCursor;
using llmp::execution::ProgramError;
using llmp::execution::ProgramPlan;
using llmp::execution::ProgramRejection;
using llmp::execution::ProgramRequest;
using llmp::execution::Repeat;
using llmp::execution::WidthRule;
using llmp::model::ComponentRole;
using llmp::model::ModelContext;
using llmp::model::StateCapability;
using llmp::model::StateCursor;
using llmp::model::StateError;
using llmp::model::StateRepresentation;
using llmp::scheduler::Admission;
using llmp::scheduler::Decision;
using llmp::scheduler::PolicySettings;
using llmp::scheduler::RequestClass;
using llmp::scheduler::RequestId;
using llmp::scheduler::RequestSpec;
using llmp::scheduler::RequestState;
using llmp::scheduler::Tick;
using llmp::test_support::Failed;
using llmp::test_support::FailedCode;
using ::testing::ElementsAre;
using ::testing::IsEmpty;

// The fake's backing granularity; every size here is whole units of it.
constexpr std::uint64_t kUnit = std::uint64_t{64} << 10U;
constexpr Bytes U(std::uint64_t units) { return Bytes(units * kUnit); }
constexpr Bytes kWire(64);             // worst-case wire bytes per position
constexpr std::uint64_t kBlock = 256;  // a canvas

Bytes Add(Bytes a, Bytes b) {
  const std::optional<Bytes> sum = a.Plus(b);
  EXPECT_TRUE(sum.has_value());
  return sum.value_or(Bytes());
}

Bytes Wire(std::uint64_t positions) { return Bytes(positions * kWire.value()); }

// Whole units of the fake's granularity.
Bytes RoundUp(Bytes bytes) { return U((bytes.value() + kUnit - 1) / kUnit); }

// The error a refused plan names, or nothing if it was admitted (an admitted
// plan's first bytes would read as ProgramError::kInvalid through .error()).
std::optional<ProgramError> Refused(
    const std::expected<ProgramPlan, llmp::execution::ProgramRejection>& plan) {
  return FailedCode(plan);
}

class Driver;

// One memory domain: the fake provider's backing, the catalog, the
// commitment ledger and admission over it, and fixed overhead F as a
// pinned runtime extent.
class Node {
 public:
  Node(std::uint64_t budget_units, std::uint64_t fixed_units, PolicySettings settings = {})
      : memory_(U(1), U(budget_units)),
        budget_(U(budget_units)),
        fixed_(U(fixed_units)),
        domain_(catalog_.AddDomain("spark")),
        ledger_(LedgerFor(domain_, budget_)),
        admission_(ledger_, domain_, settings) {
    const ExtentId runtime = catalog_
                                 .AddExtent({.domain = domain_,
                                             .memory_class = MemoryClass::kRuntime,
                                             .recovery = Recovery::kPinned,
                                             .size = fixed_,
                                             .content = {}},
                                            /*resident=*/true)
                                 .value();
    backing_[runtime] = memory_.Create(0, fixed_).value();
    EXPECT_TRUE(admission_.AddFixed(fixed_, now_).has_value());
  }

  Node(const Node&) = delete;
  Node& operator=(const Node&) = delete;
  Node(Node&&) = delete;
  Node& operator=(Node&&) = delete;
  ~Node() = default;

  Catalog& catalog() { return catalog_; }
  Admission& admission() { return admission_; }
  DomainId domain() const { return domain_; }
  Bytes budget() const { return budget_; }
  Bytes fixed() const { return fixed_; }
  // The request may have alone: B - F.
  Bytes available() const { return Bytes(budget_.value() - fixed_.value()); }
  Tick Tock() { return ++now_; }
  Tick now() const { return now_; }
  std::size_t evictions() const { return evictions_; }

  // An artifact's weights, one resource in an extent of its own.
  ResourceId Weights(std::uint8_t artifact, std::uint32_t group, std::uint64_t units) {
    llmp::catalog::ContentKey content;
    content.artifact[0] = artifact;
    content.group = group;
    const ExtentId extent = catalog_
                                .AddExtent({.domain = domain_,
                                            .memory_class = MemoryClass::kWeights,
                                            .recovery = Recovery::kFromArtifact,
                                            .size = U(units),
                                            .content = content})
                                .value();
    const std::array ranges = {
        llmp::catalog::Range{.extent = extent, .offset = Bytes(), .length = U(units)}};
    return catalog_.AddResource(ranges).value();
  }

  // Brings a closure's missing extents in, evicting reclaimable cache the
  // closure does not hold; false if even that cannot make room.
  bool Materialize(const Closure& closure) {
    const llmp::memory::MaterializationPlan plan =
        llmp::memory::PlanMaterialization(catalog_, domain_, budget_, closure);
    EXPECT_THAT(plan.stale, IsEmpty());
    EXPECT_THAT(plan.quarantined, IsEmpty());
    if (!plan.feasible) {
      return false;
    }
    for (const llmp::memory::Victim& victim : plan.victims.victims) {
      EXPECT_EQ(victim.memory_class, MemoryClass::kWeights);  // only idle cache gives way
      Evict(victim.extent);
      ++evictions_;
    }
    for (const ExtentId extent : plan.load) {
      Load(extent);
    }
    return true;
  }

  // New backing for a request's own allocation. It is within the
  // request's commitments, so reclaimable cache gives way to it.
  ExtentId Allocate(MemoryClass memory_class, Recovery recovery, Bytes size) {
    const ExtentId extent = catalog_
                                .AddExtent({.domain = domain_,
                                            .memory_class = memory_class,
                                            .recovery = recovery,
                                            .size = size,
                                            .content = {}})
                                .value();
    const std::array extents = {extent};
    EXPECT_TRUE(Materialize(catalog_.ClosureOfExtents(extents).value()));
    return extent;
  }

  // Releases a request's own allocation: mutable contents are invalidated
  // first, never silently dropped.
  void Free(ExtentId extent) {
    const auto view = catalog_.Describe(extent).value();
    switch (view.descriptor.recovery) {
      case Recovery::kPreserve:
        EXPECT_TRUE(catalog_.InvalidateContents(extent).has_value());
        Evict(extent);
        break;
      case Recovery::kDiscardable:
        Evict(extent);
        break;
      case Recovery::kPinned:
        EXPECT_TRUE(catalog_.ReleasePinned(extent).has_value());
        EXPECT_TRUE(memory_.Release(backing_.at(extent)).has_value());
        backing_.erase(extent);
        break;
      case Recovery::kFromArtifact:
        ADD_FAILURE() << "weights are evicted, not freed";
        return;
    }
    EXPECT_TRUE(catalog_.RemoveExtent(extent).has_value());
  }

  void Evict(ExtentId extent) {
    const auto ticket = catalog_.BeginEvict(extent);
    ASSERT_TRUE(ticket.has_value()) << llmp::catalog::ToString(ticket.error());
    EXPECT_TRUE(memory_.Release(backing_.at(extent)).has_value());
    backing_.erase(extent);
    EXPECT_TRUE(catalog_.CompleteEvict(*ticket).has_value());
  }

  void Enroll(const Driver* run) { runs_.push_back(run); }
  void Forget(const Driver* run) { std::erase(runs_, run); }

  // D-050's guarantee and the ledgers' invariants, after every step.
  void Check() const;

  // Occupancy that cannot give way: everything but reclaimable cache.
  Bytes Protected() const {
    Bytes reclaimable;
    for (const auto& [extent, view] : catalog_.ExtentsOf(domain_)) {
      if (Catalog::Evictable(view)) {
        reclaimable = Add(reclaimable, view.descriptor.size);
      }
    }
    return catalog_.OccupancyOf(domain_).Total().Minus(reclaimable).value_or(Bytes());
  }
  // What may be protected at a completed boundary, with no phase running:
  // F + R(G) + J.
  Bytes AtBoundary() const {
    const llmp::memory::CommitmentTotals totals = admission_.Totals();
    return Add(Add(totals.fixed, totals.background), totals.retained);
  }

  // Everything the requests held is gone: only fixed overhead and idle
  // weight cache remain, with no commitment beyond F.
  void ExpectDrained() const {
    const llmp::memory::CommitmentTotals totals = admission_.Totals();
    EXPECT_EQ(totals.retained, Bytes());
    EXPECT_EQ(totals.max_phase, Bytes());
    EXPECT_EQ(totals.required, fixed_);
    const llmp::catalog::Occupancy occupancy = catalog_.OccupancyOf(domain_);
    EXPECT_EQ(occupancy.held, Bytes());
    EXPECT_EQ(occupancy.pinned, fixed_);
    EXPECT_EQ(occupancy.loading, Bytes());
    EXPECT_EQ(occupancy.evicting, Bytes());
    EXPECT_EQ(occupancy.quarantined, Bytes());
    EXPECT_EQ(occupancy.by_class.at(static_cast<std::size_t>(MemoryClass::kLiveState)), Bytes());
    EXPECT_EQ(occupancy.by_class.at(static_cast<std::size_t>(MemoryClass::kScratch)), Bytes());
    EXPECT_EQ(occupancy.Total(), memory_.in_use());
    EXPECT_EQ(Add(occupancy.idle, fixed_), memory_.in_use());
  }

 private:
  static llmp::memory::CommitmentLedger LedgerFor(DomainId domain, Bytes budget) {
    llmp::memory::CommitmentLedger ledger;
    EXPECT_TRUE(ledger.AddDomain(domain, budget).has_value());
    return ledger;
  }

  void Load(ExtentId extent) {
    const auto ticket = catalog_.BeginLoad(extent, budget_);
    ASSERT_TRUE(ticket.has_value()) << llmp::catalog::ToString(ticket.error());
    const Bytes size = catalog_.Describe(extent).value().descriptor.size;
    // The catalog checked occupancy against B, so the fake has room.
    backing_[extent] = memory_.Create(0, size).value();
    EXPECT_TRUE(catalog_.CompleteLoad(*ticket).has_value());
  }

  llmp::providers::fake::FakeDeviceMemory memory_;
  Bytes budget_;
  Bytes fixed_;
  Catalog catalog_;
  DomainId domain_;
  llmp::memory::CommitmentLedger ledger_;
  Admission admission_;
  std::map<ExtentId, llmp::providers::BackingId> backing_;
  std::vector<const Driver*> runs_;
  Tick now_ = 0;
  std::size_t evictions_ = 0;
};

// One admitted request carrying out its program: its live states and their
// blocks, its working states, its output buffer, and the phase in flight
// (the lease on its closure and working set).
class Driver {
 public:
  Driver(Node& node, const ProgramContract& contract, const ModelContext& context, ProgramPlan plan)
      : node_(node),
        contract_(contract),
        context_(context),
        plan_(std::move(plan)),
        cursor_(plan_),
        output_(plan_.output_capacity, plan_.output) {
    for (const StateRepresentation& representation : contract_.states) {
      states_.push_back(StateCursor::Create(representation, plan_.state_bound).value());
      blocks_.emplace_back();
      snapshots_.emplace_back();
    }
    node_.Enroll(this);
  }
  Driver(const Driver&) = delete;
  Driver& operator=(const Driver&) = delete;
  Driver(Driver&&) = delete;
  Driver& operator=(Driver&&) = delete;
  ~Driver() { node_.Forget(this); }

  RequestState Submit(RequestClass request_class) {
    const RequestSpec spec{.request_class = request_class,
                           .envelope = plan_.envelope,
                           .work = plan_.max_phases,
                           .switch_cost = 1,
                           .deadline = std::nullopt};
    const auto submitted = node_.admission().Submit(spec, node_.Tock());
    EXPECT_TRUE(submitted.has_value());
    id_ = submitted.has_value() ? submitted->id : RequestId{};
    node_.Check();
    return node_.admission().StateOf(id_).value_or(RequestState::kRetired);
  }

  // The request holds the slot: its output buffer and working states come
  // into being, charged to R_i.
  void Start() {
    ASSERT_EQ(node_.admission().StateOf(id_), RequestState::kRunning);
    output_extent_ =
        node_.Allocate(MemoryClass::kRuntime, Recovery::kPinned, plan_.Retained("output")->bytes);
    for (const auto& working : contract_.working) {
      working_[working.name] = node_.Allocate(MemoryClass::kLiveState, Recovery::kPreserve,
                                              plan_.Retained(working.name)->bytes);
    }
    node_.Check();
  }

  // Starts a phase: its closure over the context's components, its working
  // set at `width`, and `also` (working state it reads), all leased.
  bool Begin(std::string_view kind, std::uint64_t width, std::span<const ExtentId> also = {}) {
    EXPECT_EQ(node_.admission().StateOf(id_), RequestState::kRunning);
    if (auto begun = cursor_.Begin(kind, width); !begun) {
      ADD_FAILURE() << kind << ": " << llmp::execution::ToString(begun.error());
      return false;
    }
    const PhaseKind& phase = *std::ranges::find(contract_.phases, kind, &PhaseKind::name);
    const Closure weights = context_.ClosureOf(node_.catalog(), phase.components).value();
    // The working set's physical backing: whole units of the provider's
    // granularity, as the plan charges it.
    const Bytes working =
        RoundUp(std::ranges::find(phase.widths, width, &PhaseWidth::positions)->working);
    scratch_ = node_.catalog()
                   .AddExtent({.domain = node_.domain(),
                               .memory_class = MemoryClass::kScratch,
                               .recovery = Recovery::kDiscardable,
                               .size = working,
                               .content = {}})
                   .value();
    std::vector<ExtentId> extents{also.begin(), also.end()};
    extents.push_back(*scratch_);
    for (const auto& [extent, generation] : weights.extents) {
      extents.push_back(extent);
    }
    const Closure closure = node_.catalog().ClosureOfExtents(extents).value();
    if (!node_.Materialize(closure)) {
      ADD_FAILURE() << kind << ": its closure cannot be materialized";
      return false;
    }
    lease_ = node_.catalog().AcquireLease(closure).value();
    EXPECT_TRUE(node_.catalog().RecordUse(*lease_, node_.now()).has_value());
    phase_bytes_ = Add(weights.bytes_by_domain.contains(node_.domain())
                           ? weights.bytes_by_domain.at(node_.domain())
                           : Bytes(),
                       working);
    node_.Check();
    return true;
  }

  // The phase reached its completed boundary: its lease ends and its
  // working set goes; what survives is retained state.
  void End() {
    if (!lease_ || !scratch_) {
      ADD_FAILURE() << "no phase to end";
      return;
    }
    EXPECT_TRUE(node_.catalog().ReleaseLease(*lease_).has_value());
    lease_.reset();
    node_.Free(*scratch_);
    scratch_.reset();
    phase_bytes_ = Bytes();
    EXPECT_TRUE(cursor_.End().has_value());
    node_.Check();
  }

  void Append(std::size_t state, std::uint64_t positions) {
    ASSERT_TRUE(states_.at(state).Append(positions).has_value());
    Sync(state);
  }
  void Commit(std::size_t state, std::uint64_t positions) {
    ASSERT_TRUE(states_.at(state).Commit(positions).has_value());
    Sync(state);  // snapshots below the new prefix go
  }
  void Truncate(std::size_t state, std::uint64_t position) {
    ASSERT_TRUE(states_.at(state).Truncate(position).has_value());
    Sync(state);
  }
  void Snapshot(std::size_t state) {
    ASSERT_TRUE(states_.at(state).Snapshot().has_value());
    Sync(state);
  }

  // Retires or terminates the request, whatever its state: the phase in
  // flight drains, and everything it holds is released.
  Decision Retire() {
    if (!cursor_.AtBoundary()) {
      End();
    }
    for (std::vector<ExtentId>& blocks : blocks_) {
      for (const ExtentId block : blocks) {
        node_.Free(block);
      }
      blocks.clear();
    }
    for (std::vector<ExtentId>& snapshots : snapshots_) {
      for (const ExtentId snapshot : snapshots) {
        node_.Free(snapshot);
      }
      snapshots.clear();
    }
    for (const auto& [name, extent] : working_) {
      node_.Free(extent);
    }
    working_.clear();
    if (output_extent_) {
      node_.Free(*output_extent_);
      output_extent_.reset();
    }
    const auto decision = node_.admission().Retire(id_, node_.Tock());
    EXPECT_TRUE(decision.has_value());
    node_.Forget(this);
    node_.Check();
    return decision.value_or(Decision{});
  }

  // The switching policy at this completed boundary (D-069).
  Decision Boundary() {
    EXPECT_TRUE(cursor_.AtBoundary());  // never mid-phase
    const auto decision = node_.admission().Boundary(id_, cursor_.phases_run(), node_.Tock());
    EXPECT_TRUE(decision.has_value());
    node_.Check();
    return decision.value_or(Decision{});
  }

  // Its retained state within R_i, item by item, and its phase within E_i.
  void CheckBounds() const {
    Bytes retained;
    for (std::size_t i = 0; i < states_.size(); ++i) {
      Bytes bytes;
      for (const ExtentId extent : blocks_[i]) {
        bytes = Add(bytes, Size(extent));
      }
      for (const ExtentId extent : snapshots_[i]) {
        bytes = Add(bytes, Size(extent));
      }
      EXPECT_LE(bytes, plan_.Retained(contract_.states[i].name)->bytes) << contract_.states[i].name;
      // A rollback never frees the committed prefix's blocks.
      const StateRepresentation& representation = contract_.states[i];
      if (states_[i].committed() > 0) {
        EXPECT_GE(blocks_[i].size(),
                  llmp::model::BlocksFor(representation, states_[i].committed()).value_or(0))
            << representation.name;
      }
      retained = Add(retained, bytes);
    }
    for (const auto& [name, extent] : working_) {
      const Bytes bytes = Size(extent);
      EXPECT_LE(bytes, plan_.Retained(name)->bytes) << name;
      retained = Add(retained, bytes);
    }
    if (output_extent_) {
      EXPECT_LE(Size(*output_extent_), plan_.Retained("output")->bytes);
      retained = Add(retained, Size(*output_extent_));
    }
    EXPECT_LE(retained, plan_.envelope.retained);
    EXPECT_LE(output_.used(), output_.capacity());
    if (!cursor_.AtBoundary()) {
      EXPECT_LE(phase_bytes_, plan_.Phase(cursor_.current())->envelope) << cursor_.current();
      EXPECT_LE(phase_bytes_, plan_.envelope.phase);
    }
  }

  RequestId id() const { return id_; }
  bool InPhase() const { return !cursor_.AtBoundary(); }
  const std::string& phase() const { return cursor_.current(); }
  const ProgramPlan& plan() const { return plan_; }
  ProgramCursor& cursor() { return cursor_; }
  OutputBuffer& output() { return output_; }
  StateCursor& state(std::size_t state) { return states_.at(state); }
  std::size_t state_count() const { return states_.size(); }
  std::size_t blocks(std::size_t state) const { return blocks_.at(state).size(); }
  ExtentId working(const std::string& name) const { return working_.at(name); }
  // Blocks a truncation released.
  std::size_t freed_blocks() const { return freed_blocks_; }
  Node& node() { return node_; }

 private:
  Bytes Size(ExtentId extent) const {
    return node_.catalog().Describe(extent).value().descriptor.size;
  }

  // The state's backing follows its blocks and snapshots: new ones
  // materialize, and those a truncation or commit drops are invalidated and
  // released. (Which snapshot backing holds which position is the
  // adapter's business; only the count matters here.)
  void Sync(std::size_t state) {
    std::vector<ExtentId>& blocks = blocks_.at(state);
    const StateRepresentation& representation = contract_.states.at(state);
    while (blocks.size() < states_.at(state).blocks()) {
      blocks.push_back(
          node_.Allocate(MemoryClass::kLiveState, Recovery::kPreserve, representation.block_bytes));
    }
    while (blocks.size() > states_.at(state).blocks()) {
      node_.Free(blocks.back());
      blocks.pop_back();
      ++freed_blocks_;
    }
    std::vector<ExtentId>& snapshots = snapshots_.at(state);
    while (snapshots.size() < states_.at(state).snapshots().size()) {
      snapshots.push_back(node_.Allocate(MemoryClass::kLiveState, Recovery::kPreserve,
                                         representation.snapshot_bytes));
    }
    while (snapshots.size() > states_.at(state).snapshots().size()) {
      node_.Free(snapshots.back());
      snapshots.pop_back();
    }
    node_.Check();
  }

  Node& node_;
  const ProgramContract& contract_;
  const ModelContext& context_;
  ProgramPlan plan_;
  ProgramCursor cursor_;
  OutputBuffer output_;
  RequestId id_;
  std::vector<StateCursor> states_;
  std::vector<std::vector<ExtentId>> blocks_;
  std::vector<std::vector<ExtentId>> snapshots_;
  std::map<std::string, ExtentId> working_;
  std::optional<ExtentId> output_extent_;
  std::optional<LeaseId> lease_;
  std::optional<ExtentId> scratch_;
  Bytes phase_bytes_;
  std::size_t freed_blocks_ = 0;
};

void Node::Check() const {
  const llmp::memory::CommitmentTotals totals = admission_.Totals();
  EXPECT_LE(totals.required, totals.budget);
  const llmp::catalog::Occupancy occupancy = catalog_.OccupancyOf(domain_);
  EXPECT_EQ(occupancy.Total(), memory_.in_use());
  EXPECT_LE(occupancy.Total(), budget_);
  // Everything but reclaimable cache fits within fixed overhead, every
  // admitted request's retained state and the envelope of each phase kind
  // in flight, not merely within the ledger's max E_i: state that outlives
  // a completed boundary outside R(G) is caught at that boundary, and a
  // phase that holds more than its own kind's envelope is caught even when
  // a wider kind would cover it.
  Bytes in_flight;
  for (const Driver* run : runs_) {
    if (run->InPhase()) {
      in_flight = Add(in_flight, run->plan().Phase(run->phase())->envelope);
    }
  }
  EXPECT_LE(Protected(), Add(AtBoundary(), in_flight));
  EXPECT_LE(Protected(), totals.required);
  for (const Driver* run : runs_) {
    run->CheckBounds();
  }
}

// ---- Contracts -------------------------------------------------------------

Sha256Digest Artifact(std::uint8_t tag) {
  Sha256Digest digest{};
  digest[0] = tag;
  return digest;
}

// A paged KV cache of one-unit blocks.
StateRepresentation Kv(std::string name, std::uint64_t positions) {
  return {.name = std::move(name),
          .block_positions = positions,
          .block_bytes = U(1),
          .capabilities = StateCapability::kAppend | StateCapability::kTruncate,
          .max_snapshots = 0,
          .snapshot_bytes = {}};
}

// Validated widths: (positions, working units).
std::vector<PhaseWidth> Widths(
    std::initializer_list<std::pair<std::uint64_t, std::uint64_t>> list) {
  std::vector<PhaseWidth> widths;
  for (const auto& [positions, units] : list) {
    widths.push_back({.positions = positions, .working = U(units)});
  }
  return widths;
}

PhaseKind Kind(std::string name, std::vector<ComponentRole> components,
               std::vector<PhaseWidth> widths, WidthRule rule, Repeat repeat, bool appends,
               bool emits) {
  return {.name = std::move(name),
          .components = std::move(components),
          .widths = std::move(widths),
          .rule = rule,
          .repeat = repeat,
          .appends = appends,
          .emits = emits};
}

DecodingMode Autoregressive() {
  return {.name = "autoregressive",
          .unit = OutputUnit::kToken,
          .max_draft = 0,
          .block = 0,
          .max_steps = 0,
          .wire_per_position = kWire};
}

DecodingMode Speculative(std::uint32_t max_draft) {
  return {.name = "speculative",
          .unit = OutputUnit::kAcceptedRun,
          .max_draft = max_draft,
          .block = 0,
          .max_steps = 0,
          .wire_per_position = kWire};
}

DecodingMode BlockDiffusion() {
  return {.name = "block_diffusion",
          .unit = OutputUnit::kBlock,
          .max_draft = 0,
          .block = kBlock,
          .max_steps = 48,
          .wire_per_position = kWire};
}

constexpr std::array kMain = {ComponentRole::kMain};
constexpr std::array kDrafter = {ComponentRole::kDrafter};
constexpr std::array kBoth = {ComponentRole::kMain, ComponentRole::kDrafter};

std::vector<ComponentRole> Roles(std::span<const ComponentRole> roles) {
  return {roles.begin(), roles.end()};
}

// A dense model with a stored draft layer: drafts of up to four, verified
// up to five wide. What passes between the phases across their completed
// boundaries is a working state: the drafted positions awaiting their
// verify, and the verify's last hidden state, which the next draft reads.
ProgramContract StoredDraftContract() {
  return {.context_limit = 4096,
          .states = {Kv("kv", 16)},
          .working = {{.name = "draft.tokens", .bytes = U(1)}},
          .phases = {Kind("prefill", Roles(kMain), Widths({{16, 1}, {64, 3}}), WidthRule::kChunk,
                          Repeat::kPromptChunks, true, false),
                     Kind("draft", Roles(kDrafter), Widths({{1, 1}, {2, 1}, {3, 1}, {4, 1}}),
                          WidthRule::kDraft, Repeat::kOutputSteps, false, false),
                     Kind("verify", Roles(kMain), Widths({{1, 2}, {2, 2}, {3, 2}, {4, 2}, {5, 2}}),
                          WidthRule::kVerify, Repeat::kOutputSteps, true, true)},
          .mode = Speculative(4)};
}

// A target and a companion drafter with KV of its own, drafting and
// verifying in one phase over both.
ProgramContract CompanionContract() {
  return {
      .context_limit = 4096,
      .states = {Kv("kv", 16), Kv("drafter.kv", 16)},
      .working = {},
      .phases = {Kind("prefill", Roles(kBoth), Widths({{64, 3}}), WidthRule::kChunk,
                      Repeat::kPromptChunks, true, false),
                 Kind("speculate", Roles(kBoth), Widths({{1, 3}, {2, 3}, {3, 3}, {4, 3}, {5, 3}}),
                      WidthRule::kVerify, Repeat::kOutputSteps, true, true)},
      .mode = Speculative(4)};
}

// Block diffusion over a causal prefix: a 256-position canvas kept as a
// working state across its denoising steps, then committed. With
// `canvas_in_phase`, the canvas is charged to the phases instead, which
// the contract must not do (the negative control).
ProgramContract DiffusionContract(bool canvas_in_phase = false) {
  const std::uint64_t canvas = canvas_in_phase ? 8 : 0;
  ProgramContract contract{.context_limit = 2048,
                           .states = {Kv("kv", 64)},
                           .working = {},
                           .phases = {Kind("prefill", Roles(kMain), Widths({{64, 2}}),
                                           WidthRule::kChunk, Repeat::kPromptChunks, true, false),
                                      Kind("denoise", Roles(kMain), Widths({{kBlock, 6 + canvas}}),
                                           WidthRule::kBlock, Repeat::kBlockSteps, false, false),
                                      Kind("commit", Roles(kMain), Widths({{kBlock, 4 + canvas}}),
                                           WidthRule::kBlock, Repeat::kBlocks, true, true)},
                           .mode = BlockDiffusion()};
  if (!canvas_in_phase) {
    contract.working.push_back({.name = "canvas", .bytes = U(8)});
  }
  return contract;
}

// An ordinary autoregressive model.
ProgramContract AutoregressiveContract(std::uint64_t working_units) {
  return {.context_limit = 4096,
          .states = {Kv("kv", 64)},
          .working = {},
          .phases = {Kind("prefill", Roles(kMain), Widths({{32, working_units}}), WidthRule::kChunk,
                          Repeat::kPromptChunks, true, false),
                     Kind("decode", Roles(kMain), Widths({{1, working_units}}), WidthRule::kOne,
                          Repeat::kOutputSteps, true, true)},
          .mode = Autoregressive()};
}

// A main model of embeddings, trunk and head, in `artifact`.
struct DenseWeights {
  ResourceId embeddings;
  ResourceId trunk;
  ResourceId head;
};

DenseWeights Dense(Node& node, std::uint8_t artifact, std::uint64_t embeddings, std::uint64_t trunk,
                   std::uint64_t head) {
  return {.embeddings = node.Weights(artifact, 0, embeddings),
          .trunk = node.Weights(artifact, 1, trunk),
          .head = node.Weights(artifact, 2, head)};
}

ModelContext Single(std::uint8_t artifact, const DenseWeights& weights) {
  return ModelContext::Create({{.role = ComponentRole::kMain,
                                .artifact = Artifact(artifact),
                                .resources = {weights.embeddings, weights.trunk, weights.head}}})
      .value();
}

ProgramPlan Plan(Node& node, const ProgramContract& contract, const ModelContext& context,
                 const ProgramRequest& request) {
  auto plan = PlanProgram(contract, context, node.catalog(), node.domain(), request,
                          node.available(), U(1));
  EXPECT_TRUE(plan.has_value()) << (plan ? "" : llmp::execution::ToString(plan.error()));
  return plan.value_or(ProgramPlan{});
}

// ---- Program steps -----------------------------------------------------------

// The prompt, in chunks of the planned width (a short last chunk runs
// padded at the narrowest validated width that holds it).
void Prefill(Driver& run) {
  const auto* planned = run.plan().Phase("prefill");
  std::uint64_t left = run.plan().prompt;
  while (left > 0) {
    const std::uint64_t chunk = std::min(left, planned->width);
    const std::uint64_t width =
        *std::ranges::find_if(planned->widths, [&](std::uint64_t w) { return w >= chunk; });
    ASSERT_TRUE(run.Begin("prefill", width));
    for (std::size_t i = 0; i < run.state_count(); ++i) {
      run.Append(i, chunk);
      run.Commit(i, chunk);
    }
    run.End();
    left -= chunk;
  }
}

// A draft phase: it writes the drafted positions into the working state
// that carries them to their verify.
void Draft(Driver& run, std::uint64_t width) {
  const ExtentId tokens = run.working("draft.tokens");
  const std::array also = {tokens};
  ASSERT_TRUE(run.Begin("draft", width, also));
  EXPECT_TRUE(run.node().catalog().ReplaceContents(tokens).has_value());
  run.End();
}

// One speculative step: draft up to `depth`, clamped so written positions
// never pass the bound, verify every draft plus one, keep `accepted`
// drafts and the verify's own token, and roll the rest back. With a
// separate draft phase, the draft is its own phase; otherwise one phase
// drafts and verifies. Returns the positions committed.
std::uint64_t SpeculativeStep(Driver& run, std::uint32_t depth, std::uint32_t accepted,
                              bool separate_draft, std::string_view verify_kind) {
  const std::uint64_t width = std::min<std::uint64_t>(depth + 1, run.state(0).room());
  if (width == 0) {
    return 0;
  }
  const std::uint64_t drafted = width - 1;
  if (separate_draft && drafted > 0) {
    Draft(run, drafted);
  }
  EXPECT_TRUE(run.output().Reserve(Wire(width), width).has_value());
  std::vector<ExtentId> also;
  if (separate_draft) {
    also.push_back(run.working("draft.tokens"));  // the verify reads the drafted positions
  }
  EXPECT_TRUE(run.Begin(verify_kind, width, also));
  for (std::size_t i = 0; i < run.state_count(); ++i) {
    run.Append(i, width);  // tentative
  }
  if (separate_draft) {
    // The last hidden state for the next draft, kept across the boundary.
    EXPECT_TRUE(run.node().catalog().ReplaceContents(also.front()).has_value());
  }
  const std::uint64_t keep = std::min<std::uint64_t>(accepted, drafted) + 1;
  for (std::size_t i = 0; i < run.state_count(); ++i) {
    run.Commit(i, keep);
    run.Truncate(i, run.state(i).committed());
  }
  EXPECT_TRUE(run.output().Publish(Wire(keep), keep).has_value());
  run.End();
  EXPECT_TRUE(run.output().Drain(Wire(keep)).has_value());
  return keep;
}

// Autoregressive decode to the output bound.
void Decode(Driver& run) {
  while (run.state(0).room() > 0) {
    ASSERT_TRUE(run.output().Reserve(Wire(1), 1).has_value());
    ASSERT_TRUE(run.Begin("decode", 1));
    run.Append(0, 1);
    run.Commit(0, 1);
    EXPECT_TRUE(run.output().Publish(Wire(1), 1).has_value());
    run.End();
    EXPECT_TRUE(run.output().Drain(Wire(1)).has_value());
  }
}

// One denoising step: the phase holds the canvas's only lease and rewrites
// it in place.
void Denoise(Node& node, Driver& run, ExtentId canvas) {
  const std::uint64_t before = node.catalog().Describe(canvas).value().content_generation;
  const std::array also = {canvas};
  ASSERT_TRUE(run.Begin("denoise", kBlock, also));
  EXPECT_TRUE(node.catalog().ReplaceContents(canvas).has_value());
  run.End();
  EXPECT_EQ(node.catalog().Describe(canvas).value().content_generation, before + 1);
}

// Commits the canvas: every position, clamped to the output bound, lands
// in the KV and the output together, or the phase does not start.
std::expected<std::uint64_t, ProgramError> CommitBlock(Driver& run, ExtentId canvas) {
  const std::uint64_t positions = std::min(kBlock, run.state(0).room());
  if (auto reserved = run.output().Reserve(Wire(positions), positions); !reserved) {
    return std::unexpected(reserved.error());
  }
  const std::array also = {canvas};
  EXPECT_TRUE(run.Begin("commit", kBlock, also));
  run.Append(0, positions);
  run.Commit(0, positions);
  EXPECT_TRUE(run.output().Publish(Wire(positions), positions).has_value());
  run.End();
  return positions;
}

// ---- Plans: widths, envelopes and rejections -----------------------------------

TEST(ProgramPlanTest, ExplainsItsEnvelopeItemByItem) {
  Node node(42, 2);
  const DenseWeights main = Dense(node, 1, 4, 20, 4);
  const ResourceId draft_layer = node.Weights(1, 3, 3);
  const ModelContext context =
      ModelContext::Create({{.role = ComponentRole::kMain,
                             .artifact = Artifact(1),
                             .resources = {main.embeddings, main.trunk, main.head}},
                            {.role = ComponentRole::kDrafter,
                             .artifact = Artifact(1),
                             .resources = {draft_layer, main.embeddings, main.head}}})
          .value();
  const ProgramContract contract = StoredDraftContract();
  const ProgramPlan plan =
      Plan(node, contract, context, {.prompt = 40, .max_output = 64, .draft_depth = 4, .steps = 0});
  // R_i: KV through 104 positions (seven blocks), the drafted positions
  // and a five-position output buffer (320 bytes, one unit).
  EXPECT_EQ(plan.state_bound, 104U);
  EXPECT_EQ(plan.Retained("kv")->bytes, U(7));
  EXPECT_EQ(plan.Retained("draft.tokens")->bytes, U(1));
  EXPECT_EQ(plan.Retained("output")->bytes, U(1));
  EXPECT_EQ(plan.output_capacity, Wire(5));
  EXPECT_EQ(plan.envelope.retained, U(9));
  // E_i: the widest prefill that fits; the draft reads the stored layer
  // and the shared embeddings and head; the verify is five wide.
  const auto* prefill = plan.Phase("prefill");
  EXPECT_EQ(prefill->width, 64U);
  EXPECT_EQ(prefill->envelope, U(31));
  EXPECT_EQ(prefill->max_count, 3U);  // 40 positions in chunks of at least 16
  const auto* draft = plan.Phase("draft");
  EXPECT_THAT(draft->widths, ElementsAre(1, 2, 3, 4));
  EXPECT_EQ(draft->closure, U(11));
  const auto* verify = plan.Phase("verify");
  EXPECT_EQ(verify->width, 5U);
  EXPECT_THAT(verify->widths, ElementsAre(1, 2, 3, 4, 5));
  EXPECT_EQ(verify->envelope, U(30));
  EXPECT_EQ(verify->max_count, 64U);
  EXPECT_EQ(plan.envelope.phase, U(31));
  EXPECT_EQ(plan.max_phases, 3U + 64 + 64);

  // Less room: the prefill narrows to 16; less still, the plan is refused,
  // naming the phase kind, its width, the bytes and the shortfall.
  const ProgramRequest request{.prompt = 40, .max_output = 64, .draft_depth = 4, .steps = 0};
  const auto narrow =
      PlanProgram(contract, context, node.catalog(), node.domain(), request, U(39), U(1)).value();
  EXPECT_EQ(narrow.Phase("prefill")->width, 16U);
  EXPECT_EQ(narrow.envelope.phase, U(30));
  const auto refused_plan =
      PlanProgram(contract, context, node.catalog(), node.domain(), request, U(37), U(1));
  ASSERT_FALSE(refused_plan.has_value());
  const ProgramRejection& refused = refused_plan.error();
  EXPECT_EQ(refused.error, ProgramError::kDoesNotFit);
  EXPECT_EQ(refused.phase, "prefill");
  EXPECT_EQ(refused.width, 16U);
  EXPECT_EQ(refused.required, U(38));
  EXPECT_EQ(refused.shortfall, U(1));
  EXPECT_EQ(llmp::execution::ToString(refused),
            "a phase envelope does not fit above the retained state: phase prefill at width 16 "
            "needs 2490368 bytes, 65536 more than available");
}

// D-050's "grant with a full useful cache; lease release without pressure"
// (M2): a grant is a promise, not bytes, so admitting a request over a
// cache that fills the budget evicts nothing. The request's first use
// brings in only what it lacks, evicting idle cache only until that fits,
// and releasing its lease leaves its contents resident.
TEST(ShapeScenarioTest, AGrantOverAFullUsefulCacheEvictsNothing) {
  Node node(40, 2);
  const DenseWeights cached = Dense(node, 1, 4, 24, 4);  // 32 units: with F, 34 of 40
  const std::array cached_resources = {cached.embeddings, cached.trunk, cached.head};
  ASSERT_TRUE(node.Materialize(node.catalog().ClosureOf(cached_resources).value()));
  const DenseWeights fresh = Dense(node, 2, 2, 6, 2);  // 10 units
  const Bytes before = node.catalog().OccupancyOf(node.domain()).Total();
  const ProgramPlan plan = Plan(node, AutoregressiveContract(1), Single(2, fresh),
                                {.prompt = 32, .max_output = 8, .draft_depth = 0, .steps = 0});
  const auto admitted = node.admission().Submit(
      llmp::scheduler::RequestSpec{.request_class = RequestClass::kInteractive,
                                   .envelope = plan.envelope,
                                   .work = 10,
                                   .switch_cost = 1,
                                   .deadline = std::nullopt},
      node.Tock());
  ASSERT_TRUE(admitted.has_value());
  EXPECT_EQ(node.admission().StateOf(admitted->id), RequestState::kRunning);
  EXPECT_EQ(node.evictions(), 0U);
  EXPECT_EQ(node.catalog().OccupancyOf(node.domain()).Total(), before);
  // On use: the fresh weights load, and idle cache gives way only for the
  // shortfall (34 + 10 - 40 = 4 units).
  const std::array fresh_resources = {fresh.embeddings, fresh.trunk, fresh.head};
  const Closure closure = node.catalog().ClosureOf(fresh_resources).value();
  ASSERT_TRUE(node.Materialize(closure));
  EXPECT_GE(node.evictions(), 1U);
  EXPECT_LE(node.catalog().OccupancyOf(node.domain()).Total(), node.budget());
  std::size_t cached_resident = 0;
  for (const ResourceId resource : cached_resources) {
    const std::array one = {resource};
    const auto extents = node.catalog().ClosureOf(one).value().extents;
    cached_resident += node.catalog().Describe(extents.front().first).value().state ==
                               llmp::catalog::ExtentState::kResident
                           ? 1
                           : 0;
  }
  EXPECT_GE(cached_resident, 1U);  // not the whole cache
  const LeaseId lease = node.catalog().AcquireLease(closure).value();
  const Bytes held = node.catalog().OccupancyOf(node.domain()).Total();
  ASSERT_TRUE(node.catalog().ReleaseLease(lease).has_value());
  EXPECT_EQ(node.catalog().OccupancyOf(node.domain()).Total(), held);
  for (const auto& [extent, generation] : closure.extents) {
    EXPECT_EQ(node.catalog().Describe(extent).value().state, llmp::catalog::ExtentState::kResident);
  }
}

// BP-V3 with the EXL3 fixture's own numbers: its weights (292 extents of
// 2 MiB, the 4.0 bpw artifact), its KV (24 layers, 128-wide K and V, F16:
// 12,288 bytes a position) in blocks of 512 positions, and each recorded
// phase kind's working set as the proof measured it: the pre-registered
// activation region (Qwen2Exl3Test.RegionsAreThePreRegisteredBufferPlan)
// plus the GGML pool's bound. The largest reconstruction phase is the
// 1,023-row prefill, larger than the 1,024-row one. At a budget of exactly
// R_i plus its envelope the widest prefill is admitted, since a prompt may
// end in a 1,023-row chunk; a byte less and the plan narrows, explained;
// below the narrowest prefill it is refused as impossible, naming the
// phase kind, the width and the shortfall.
TEST(ProgramPlanTest, ATightBudgetAdmitsTheLargestReconstructionPhaseOrRefusesThePlan) {
  constexpr std::uint64_t kGranule = std::uint64_t{2} << 20U;  // D-033
  Catalog catalog;
  const DomainId domain = catalog.AddDomain("spark");
  const ExtentId extent = catalog
                              .AddExtent({.domain = domain,
                                          .memory_class = MemoryClass::kWeights,
                                          .recovery = Recovery::kFromArtifact,
                                          .size = Bytes(292 * kGranule),
                                          .content = {}})
                              .value();
  const std::array ranges = {
      llmp::catalog::Range{.extent = extent, .offset = Bytes(), .length = Bytes(292 * kGranule)}};
  const ResourceId weights = catalog.AddResource(ranges).value();
  const ModelContext context =
      ModelContext::Create(
          {{.role = ComponentRole::kMain, .artifact = Artifact(40), .resources = {weights}}})
          .value();
  const auto phase = [](std::uint64_t region, std::uint64_t pool) { return Bytes(region + pool); };
  const Bytes w1023 = phase(373'247'744, 11'343'104);
  const ProgramContract contract{
      .context_limit = 8192,
      .states = {{.name = "kv",
                  .block_positions = 512,
                  .block_bytes = Bytes(std::uint64_t{512} * 12'288),
                  .capabilities = StateCapability::kAppend | StateCapability::kTruncate,
                  .max_snapshots = 0,
                  .snapshot_bytes = {}}},
      .working = {},
      .phases = {{.name = "prefill",
                  .components = Roles(kMain),
                  .widths = {{.positions = 32, .working = phase(9'838'592, 354'816)},
                             {.positions = 144, .working = phase(44'273'664, 1'596'672)},
                             {.positions = 145, .working = phase(103'301'376, 1'607'936)},
                             {.positions = 1023, .working = w1023},
                             {.positions = 1024, .working = phase(371'720'192, 11'356'160)}},
                  .rule = WidthRule::kChunk,
                  .repeat = Repeat::kPromptChunks,
                  .appends = true,
                  .emits = false},
                 {.name = "step",
                  .components = Roles(kMain),
                  .widths = {{.positions = 1, .working = phase(307'456, 48'128)}},
                  .rule = WidthRule::kOne,
                  .repeat = Repeat::kOutputSteps,
                  .appends = true,
                  .emits = true}},
      .mode = Autoregressive()};
  const ProgramRequest request{.prompt = 1023, .max_output = 16, .draft_depth = 0, .steps = 0};
  const auto plan_at = [&](Bytes available) {
    return PlanProgram(contract, context, catalog, domain, request, available, Bytes(kGranule));
  };
  const auto roomy = plan_at(Bytes(std::uint64_t{64} << 30U));
  ASSERT_TRUE(roomy.has_value()) << llmp::execution::ToString(roomy.error());
  const Bytes closure(292 * kGranule);
  const Bytes widest((w1023.value() + kGranule - 1) / kGranule * kGranule);
  EXPECT_EQ(roomy->Phase("prefill")->width, 1024U);
  EXPECT_EQ(roomy->Phase("prefill")->working, widest);  // the 1,023-row phase's, rounded
  EXPECT_EQ(roomy->Phase("prefill")->envelope, Add(closure, widest));
  const Bytes retained = roomy->envelope.retained;
  // 1,039 positions; output.
  EXPECT_EQ(retained, Add(Bytes(std::uint64_t{3} * 512 * 12'288), Bytes(kGranule)));

  // Exactly R_i plus the largest reconstruction phase's envelope: admitted.
  const Bytes exact = Add(retained, Add(closure, widest));
  const auto tight = plan_at(exact);
  ASSERT_TRUE(tight.has_value()) << llmp::execution::ToString(tight.error());
  EXPECT_EQ(tight->Phase("prefill")->width, 1024U);
  // A byte less: the prefill narrows to the widest width whose chunks fit.
  const auto narrower = plan_at(Bytes(exact.value() - 1));
  ASSERT_TRUE(narrower.has_value()) << llmp::execution::ToString(narrower.error());
  EXPECT_EQ(narrower->Phase("prefill")->width, 145U);
  // Less than even the narrowest prefill needs: refused, explained.
  const Bytes narrowest = Add(retained, Add(closure, Bytes(5 * kGranule)));  // 32 rows: 10.2 MB
  const auto refused = plan_at(Bytes(narrowest.value() - 1));
  ASSERT_FALSE(refused.has_value());
  EXPECT_EQ(refused.error().error, ProgramError::kDoesNotFit);
  EXPECT_EQ(refused.error().phase, "prefill");
  EXPECT_EQ(refused.error().width, 32U);
  EXPECT_EQ(refused.error().required, narrowest);
  EXPECT_EQ(refused.error().shortfall, Bytes(1));
}

TEST(ProgramPlanTest, SpeculationNeedsTruncationAndValidatedWidths) {
  Node node(64, 2);
  const DenseWeights main = Dense(node, 1, 4, 20, 4);
  const ResourceId draft_layer = node.Weights(1, 3, 3);
  const ModelContext context =
      ModelContext::Create({{.role = ComponentRole::kMain,
                             .artifact = Artifact(1),
                             .resources = {main.embeddings, main.trunk, main.head}},
                            {.role = ComponentRole::kDrafter,
                             .artifact = Artifact(1),
                             .resources = {draft_layer, main.embeddings}}})
          .value();
  const auto plan = [&](const ProgramContract& contract, std::uint32_t depth) {
    return PlanProgram(contract, context, node.catalog(), node.domain(),
                       {.prompt = 40, .max_output = 64, .draft_depth = depth, .steps = 0},
                       node.available(), U(1));
  };
  ProgramContract contract = StoredDraftContract();
  // An append-only state cannot roll a rejected draft back.
  contract.states[0].capabilities = static_cast<std::uint8_t>(StateCapability::kAppend);
  EXPECT_EQ(Refused(plan(contract, 4)), ProgramError::kUnsupported);
  // Snapshot-only state needs a snapshot at every drafted position.
  contract.states[0] = {
      .name = "recurrent",
      .block_positions = 0,
      .block_bytes = U(1),
      .capabilities = StateCapability::kAppend | StateCapability::kTruncateAtSnapshot,
      .max_snapshots = 4,
      .snapshot_bytes = U(1)};
  EXPECT_EQ(Refused(plan(contract, 4)), ProgramError::kUnsupported);
  EXPECT_TRUE(plan(contract, 3).has_value());
  contract.states[0].max_snapshots = 5;
  EXPECT_EQ(plan(contract, 4).value().Retained("recurrent")->bytes, U(1 + 5));
  // No deeper than the mode drafts, and a clamped verify needs every
  // narrower width validated.
  contract = StoredDraftContract();
  // A shallower draft plans, and runs, no wider than its own verify.
  const ProgramPlan shallow = plan(contract, 2).value();
  EXPECT_THAT(shallow.Phase("verify")->widths, ElementsAre(1, 2, 3));
  ProgramCursor cursor(shallow);
  EXPECT_EQ(Failed(cursor.Begin("verify", 4)), ProgramError::kUnsupported);
  EXPECT_EQ(Refused(plan(contract, 5)), ProgramError::kUnsupported);
  EXPECT_EQ(Refused(plan(contract, 0)), ProgramError::kUnsupported);
  contract.phases[2].widths = Widths({{1, 2}, {2, 2}, {5, 2}});
  const auto missing_plan = plan(contract, 4);
  ASSERT_FALSE(missing_plan.has_value());
  const ProgramRejection& missing = missing_plan.error();
  EXPECT_EQ(missing.error, ProgramError::kUnsupported);
  EXPECT_EQ(missing.phase, "verify");
  EXPECT_EQ(missing.width, 3U);
  // A draft kind only in a speculative mode.
  contract = StoredDraftContract();
  contract.mode = Autoregressive();
  EXPECT_EQ(Refused(plan(contract, 0)), ProgramError::kInvalid);
}

TEST(ProgramPlanTest, BlockOutputEndsOnABlockTheContextHolds) {
  Node node(128, 2);
  const DenseWeights main = Dense(node, 3, 4, 20, 4);
  const ModelContext context = Single(3, main);
  ProgramContract contract = DiffusionContract();
  const auto plan = [&](std::uint64_t prompt, std::uint64_t output, std::uint32_t steps) {
    return PlanProgram(contract, context, node.catalog(), node.domain(),
                       {.prompt = prompt, .max_output = output, .draft_depth = 0, .steps = steps},
                       node.available(), U(1));
  };
  const ProgramPlan full = plan(64, 500, 4).value();
  EXPECT_EQ(full.blocks, 2U);
  EXPECT_EQ(full.output, 500U);  // the last commit clamps to it
  EXPECT_EQ(full.Phase("denoise")->width, kBlock);
  EXPECT_EQ(full.Phase("denoise")->max_count, 8U);
  EXPECT_EQ(full.Phase("commit")->max_count, 2U);
  EXPECT_EQ(full.Retained("canvas")->bytes, U(8));  // retained, not in the phase
  EXPECT_EQ(full.Retained("kv")->bytes, U(9));      // 564 positions
  EXPECT_EQ(full.output_capacity, Wire(kBlock));
  EXPECT_EQ(full.envelope.retained, U(18));
  EXPECT_EQ(full.envelope.phase, U(34));
  // The context holds seven whole blocks after this prompt: a longer
  // output bound is lowered to end on the seventh.
  const ProgramPlan lowered = plan(64, 5000, 4).value();
  EXPECT_EQ(lowered.blocks, 7U);
  EXPECT_EQ(lowered.output, 7 * kBlock);
  EXPECT_EQ(Refused(plan(1900, 100, 4)), ProgramError::kContextExhausted);
  EXPECT_EQ(Refused(plan(64, 500, 49)), ProgramError::kUnsupported);
  EXPECT_EQ(Refused(plan(64, 500, 0)), ProgramError::kUnsupported);
  // Charged to the phases instead, the canvas leaves R_i and swells E_i.
  contract = DiffusionContract(/*canvas_in_phase=*/true);
  const ProgramPlan naive = plan(64, 500, 4).value();
  EXPECT_EQ(naive.envelope.retained, U(10));
  EXPECT_EQ(naive.envelope.phase, U(42));
  // An autoregressive output bound past the context is refused, not cut.
  const ProgramContract autoregressive = AutoregressiveContract(1);
  EXPECT_EQ(Refused(PlanProgram(autoregressive, context, node.catalog(), node.domain(),
                                {.prompt = 4000, .max_output = 100, .draft_depth = 0, .steps = 0},
                                node.available(), U(1))),
            ProgramError::kContextExhausted);
}

TEST(ProgramPlanTest, CursorAndOutputBufferHoldTheProgramToItsBounds) {
  Node node(64, 2);
  const DenseWeights main = Dense(node, 3, 4, 20, 4);
  const ModelContext context = Single(3, main);
  const ProgramContract contract = DiffusionContract();
  const ProgramPlan plan = Plan(node, contract, context,
                                {.prompt = 64, .max_output = 200, .draft_depth = 0, .steps = 2});
  ProgramCursor cursor(plan);
  EXPECT_EQ(Failed(cursor.End()), ProgramError::kWrongState);
  EXPECT_EQ(Failed(cursor.Begin("denoise", 128)), ProgramError::kUnsupported);
  EXPECT_EQ(Failed(cursor.Begin("unknown", 1)), ProgramError::kInvalid);
  for (int step = 0; step < 2; ++step) {
    ASSERT_TRUE(cursor.Begin("denoise", kBlock).has_value());
    EXPECT_FALSE(cursor.AtBoundary());
    EXPECT_EQ(Failed(cursor.Begin("commit", kBlock)), ProgramError::kWrongState);
    ASSERT_TRUE(cursor.End().has_value());
  }
  // Two steps of one block: a third is past the admitted program.
  EXPECT_EQ(Failed(cursor.Begin("denoise", kBlock)), ProgramError::kBeyondProgram);
  ASSERT_TRUE(cursor.Begin("commit", kBlock).has_value());
  ASSERT_TRUE(cursor.End().has_value());
  EXPECT_EQ(Failed(cursor.Begin("commit", kBlock)), ProgramError::kBeyondProgram);
  EXPECT_EQ(cursor.phases_run(), 3U);

  OutputBuffer output(Wire(kBlock), 1000);
  EXPECT_EQ(Failed(output.Reserve(Wire(kBlock + 1), kBlock + 1)), ProgramError::kOutputFull);
  ASSERT_TRUE(output.Reserve(Wire(kBlock), kBlock).has_value());
  EXPECT_EQ(Failed(output.Reserve(Wire(1), 1)), ProgramError::kWrongState);
  EXPECT_EQ(Failed(output.Publish(Wire(kBlock + 1), kBlock + 1)), ProgramError::kInvalid);
  ASSERT_TRUE(output.Publish(Wire(200), 200).has_value());
  EXPECT_EQ(Failed(output.Publish(Wire(1), 1)), ProgramError::kWrongState);
  EXPECT_EQ(Failed(output.Reserve(Wire(kBlock), kBlock)), ProgramError::kOutputFull);
  ASSERT_TRUE(output.Reserve(Wire(56), 56).has_value());
  output.Cancel();
  EXPECT_EQ(output.used(), Wire(200));
  EXPECT_EQ(output.published_positions(), 200U);
  EXPECT_EQ(Failed(output.Drain(Wire(201))), ProgramError::kInvalid);
  ASSERT_TRUE(output.Drain(Wire(200)).has_value());
}

// Contracts that could only fail once running are refused at planning: no
// phase that emits, or an output unit of no wire bytes (the output buffer
// would be empty, so no phase could ever reserve room); R_i items whose
// names collide, so that one item's allowance could be read for another's;
// and a closure that reaches a memory domain the plan does not check.
TEST(ProgramPlanTest, RefusesContractsThatCouldOnlyFailOnceRunning) {
  Node node(64, 2);
  const DenseWeights main = Dense(node, 3, 4, 20, 4);
  const ModelContext context = Single(3, main);
  const ProgramRequest request{.prompt = 32, .max_output = 16, .draft_depth = 0, .steps = 0};
  const auto plan = [&](const ProgramContract& contract) {
    return PlanProgram(contract, context, node.catalog(), node.domain(), request, node.available(),
                       U(1));
  };
  ASSERT_TRUE(plan(AutoregressiveContract(1)).has_value());

  ProgramContract silent = AutoregressiveContract(1);
  silent.phases[1].emits = false;
  EXPECT_EQ(Refused(plan(silent)), ProgramError::kInvalid);
  ProgramContract wireless = AutoregressiveContract(1);
  wireless.mode.wire_per_position = Bytes();
  EXPECT_EQ(Refused(plan(wireless)), ProgramError::kInvalid);

  ProgramContract twice = AutoregressiveContract(1);
  twice.states.push_back(Kv("kv", 16));
  EXPECT_EQ(Refused(plan(twice)), ProgramError::kInvalid);
  ProgramContract shadow = AutoregressiveContract(1);
  shadow.working.push_back({.name = "kv", .bytes = U(1)});
  EXPECT_EQ(Refused(plan(shadow)), ProgramError::kInvalid);
  ProgramContract output = AutoregressiveContract(1);
  output.working.push_back({.name = "output", .bytes = U(8)});
  EXPECT_EQ(Refused(plan(output)), ProgramError::kInvalid);
  // Phase kinds are named apart from R_i's items: sharing a name with one
  // is no collision.
  ProgramContract apart = AutoregressiveContract(1);
  apart.phases[0].name = "kv";
  apart.phases[1].name = "output";
  EXPECT_TRUE(plan(apart).has_value());

  // State blocks are whole backing units.
  ProgramContract ragged = AutoregressiveContract(1);
  ragged.states[0].block_bytes = Bytes(kUnit + 1);
  EXPECT_EQ(Refused(plan(ragged)), ProgramError::kInvalid);

  // The head's backing lives in a second domain: its bytes are in no
  // envelope this plan computes, so the plan is refused, not partial.
  const DomainId host = node.catalog().AddDomain("host");
  const ExtentId remote = node.catalog()
                              .AddExtent({.domain = host,
                                          .memory_class = MemoryClass::kWeights,
                                          .recovery = Recovery::kFromArtifact,
                                          .size = U(4),
                                          .content = {}})
                              .value();
  const std::array ranges = {
      llmp::catalog::Range{.extent = remote, .offset = Bytes(), .length = U(4)}};
  const ResourceId head = node.catalog().AddResource(ranges).value();
  const ModelContext split =
      ModelContext::Create({{.role = ComponentRole::kMain,
                             .artifact = Artifact(3),
                             .resources = {main.embeddings, main.trunk, head}}})
          .value();
  const auto refused = PlanProgram(AutoregressiveContract(1), split, node.catalog(), node.domain(),
                                   request, node.available(), U(1));
  ASSERT_FALSE(refused.has_value());
  EXPECT_EQ(Refused(refused), ProgramError::kUnsupported);
  EXPECT_EQ(refused.error().phase, "prefill");
}

// The output buffer holds the program to its admitted output bound: a
// phase whose output would pass it cannot reserve room, so a verify wider
// than the output left never starts, and a publication never exceeds what
// its phase reserved.
TEST(ProgramPlanTest, OutputBufferRefusesOutputPastTheBound) {
  OutputBuffer output(Wire(5), 12);
  EXPECT_EQ(Failed(output.Reserve(Wire(5), 13)), ProgramError::kBeyondProgram);
  for (int step = 0; step < 2; ++step) {
    ASSERT_TRUE(output.Reserve(Wire(5), 5).has_value());
    ASSERT_TRUE(output.Publish(Wire(5), 5).has_value());
    ASSERT_TRUE(output.Drain(Wire(5)).has_value());
  }
  // Two positions are left: a five-wide verify is refused before it starts.
  EXPECT_EQ(Failed(output.Reserve(Wire(5), 5)), ProgramError::kBeyondProgram);
  ASSERT_TRUE(output.Reserve(Wire(2), 2).has_value());
  EXPECT_EQ(Failed(output.Publish(Wire(2), 3)), ProgramError::kInvalid);
  ASSERT_TRUE(output.Publish(Wire(1), 1).has_value());
  ASSERT_TRUE(output.Reserve(Wire(1), 1).has_value());
  ASSERT_TRUE(output.Publish(Wire(1), 1).has_value());
  EXPECT_EQ(output.published_positions(), 12U);
  EXPECT_EQ(Failed(output.Reserve(Wire(1), 1)), ProgramError::kBeyondProgram);
}

// A live state is charged through its bound's last position: a bound one
// past a whole block takes a block more, and the program, run to that
// bound, fills it within R_i.
TEST(ProgramPlanTest, RetainedStateCoversItsLastPosition) {
  Node node(64, 2);
  const DenseWeights main = Dense(node, 3, 4, 20, 4);
  const ModelContext context = Single(3, main);
  const ProgramContract contract = AutoregressiveContract(1);
  const ProgramPlan plan =
      Plan(node, contract, context, {.prompt = 32, .max_output = 33, .draft_depth = 0, .steps = 0});
  EXPECT_EQ(plan.state_bound, 65U);
  EXPECT_EQ(plan.Retained("kv")->bytes, U(2));
  Driver run(node, contract, context, plan);
  ASSERT_EQ(run.Submit(RequestClass::kInteractive), RequestState::kRunning);
  run.Start();
  Prefill(run);
  Decode(run);
  EXPECT_EQ(run.state(0).committed(), plan.state_bound);
  EXPECT_EQ(run.blocks(0), 2U);
  (void)run.Retire();
  node.ExpectDrained();
}

// ---- Scenario 1: a stored draft layer ---------------------------------------

struct StoredDraft {
  explicit StoredDraft(Node& node)
      : main(Dense(node, 1, 4, 20, 4)),
        draft_layer(node.Weights(1, 3, 3)),
        context(ModelContext::Create({{.role = ComponentRole::kMain,
                                       .artifact = Artifact(1),
                                       .resources = {main.embeddings, main.trunk, main.head}},
                                      {.role = ComponentRole::kDrafter,
                                       .artifact = Artifact(1),
                                       .resources = {draft_layer, main.embeddings, main.head}}})
                    .value()) {}
  DenseWeights main;
  ResourceId draft_layer;
  ModelContext context;
  ProgramContract contract = StoredDraftContract();
  ProgramRequest request{.prompt = 40, .max_output = 64, .draft_depth = 4, .steps = 0};
};

// At a budget of exactly F + R_i + E_i, drafts of four are verified five
// wide and accepted in part; the rejected positions roll back, freeing
// whole blocks, never the committed prefix; the draft and verify closures
// take turns in memory; the program ends at its bound with every position
// accounted for, and nothing leaks.
TEST(ShapeScenarioTest, StoredDraftRejectedAndRolledBack) {
  Node node(42, 2);
  StoredDraft model(node);
  const ProgramPlan plan = Plan(node, model.contract, model.context, model.request);
  ASSERT_EQ(Add(Add(node.fixed(), plan.envelope.retained), plan.envelope.phase), node.budget());
  Driver run(node, model.contract, model.context, plan);
  ASSERT_EQ(run.Submit(RequestClass::kInteractive), RequestState::kRunning);
  run.Start();
  Prefill(run);
  EXPECT_EQ(run.state(0).committed(), 40U);
  EXPECT_EQ(run.blocks(0), 3U);

  const std::array accepted = {3U, 0U, 4U, 2U, 1U, 4U, 0U, 3U};
  std::size_t step = 0;
  std::uint64_t committed = run.state(0).committed();
  while (run.state(0).room() > 0) {
    const std::uint64_t kept =
        SpeculativeStep(run, 4, accepted.at(step % accepted.size()), true, "verify");
    ++step;
    EXPECT_GE(kept, 1U);
    EXPECT_EQ(run.state(0).committed(), committed + kept);
    committed = run.state(0).committed();
    EXPECT_EQ(run.state(0).written(), committed);  // nothing rejected survives
    // Rolling back below the committed prefix is refused and changes nothing.
    EXPECT_EQ(Failed(run.state(0).Truncate(committed - 1)), StateError::kBelowCommitted);
    EXPECT_EQ(run.output().published_positions(), committed - plan.prompt);
  }
  // Some verifies wrote into a new block that their rollback released.
  EXPECT_GT(run.freed_blocks(), 0U);
  EXPECT_EQ(run.state(0).committed(), plan.state_bound);
  EXPECT_EQ(run.output().published_positions(), plan.output);
  EXPECT_EQ(run.blocks(0), 7U);
  EXPECT_LE(run.cursor().CountOf("verify"), plan.Phase("verify")->max_count);
  EXPECT_GT(node.evictions(), 0U);  // the closures took turns at this budget
  // The program is over: no position is left to verify.
  EXPECT_EQ(Failed(run.state(0).Append(1)), StateError::kExceedsBound);
  (void)run.Retire();
  node.ExpectDrained();
}

// A background speculative request is paused after its draft, before the
// verify: nothing of the draft commits while it waits, its drafted
// positions stay resident and protected, and it resumes to verify them. A
// draft cancelled before its verify retires all its work.
TEST(ShapeScenarioTest, DraftPausedOrCancelledBeforeVerifyCommitsNothing) {
  // F + R_A + R_B + E_A = 2 + 9 + 2 + 31.
  Node node(44, 2);
  StoredDraft model(node);
  const DenseWeights small = Dense(node, 2, 1, 4, 1);
  const ModelContext other = Single(2, small);
  const ProgramContract autoregressive = AutoregressiveContract(1);
  const ProgramPlan plan_a = Plan(node, model.contract, model.context, model.request);
  const ProgramPlan plan_b = Plan(node, autoregressive, other,
                                  {.prompt = 32, .max_output = 16, .draft_depth = 0, .steps = 0});
  ASSERT_EQ(plan_b.envelope.retained, U(2));

  Driver a(node, model.contract, model.context, plan_a);
  ASSERT_EQ(a.Submit(RequestClass::kBackground), RequestState::kRunning);
  a.Start();
  Prefill(a);
  (void)SpeculativeStep(a, 4, 2, true, "verify");
  const std::uint64_t committed = a.state(0).committed();
  const std::uint64_t published = a.output().published_positions();
  Draft(a, 4);

  Driver b(node, autoregressive, other, plan_b);
  ASSERT_EQ(b.Submit(RequestClass::kInteractive), RequestState::kWaiting);
  const Decision pause = a.Boundary();
  ASSERT_EQ(pause.paused, a.id());
  ASSERT_EQ(pause.run, b.id());
  b.Start();
  Prefill(b);
  // While A waits, its draft commits nothing and stays protected.
  EXPECT_EQ(a.state(0).committed(), committed);
  EXPECT_EQ(a.state(0).written(), committed);
  EXPECT_EQ(a.output().published_positions(), published);
  const auto tokens = node.catalog().Describe(a.working("draft.tokens")).value();
  EXPECT_FALSE(Catalog::Evictable(tokens));
  Decode(b);
  EXPECT_EQ(b.Retire().run, a.id());  // A resumes next

  // The verify of the paused draft.
  EXPECT_TRUE(a.output().Reserve(Wire(5), 5).has_value());
  const std::array tokens_also = {a.working("draft.tokens")};
  ASSERT_TRUE(a.Begin("verify", 5, tokens_also));
  a.Append(0, 5);
  a.Commit(0, 2);
  a.Truncate(0, a.state(0).committed());
  EXPECT_TRUE(a.output().Publish(Wire(2), 2).has_value());
  a.End();
  EXPECT_EQ(a.state(0).committed(), committed + 2);

  // Cancelled between draft and verify: the request retires with its
  // draft uncommitted.
  const std::uint64_t before_cancel = a.state(0).committed();
  Draft(a, 4);
  EXPECT_EQ(a.state(0).committed(), before_cancel);
  (void)a.Retire();
  node.ExpectDrained();
}

// ---- Scenario 1b: paged and snapshot-only state, and a drafter's own KV ------

// A hybrid model (attention layers with paged KV, recurrent layers whose
// state truncates only through snapshots) with a stored draft layer that
// keeps KV of its own, which its draft phase appends to. The draft phase
// first catches up on the verify's own token, so it may run a position
// wider than the draft. Sizes are deliberately ragged: the draft's working
// set and the handoff are not whole units, and the verify's working set is
// largest at its narrowest width.
ProgramContract MixedContract() {
  const auto draft_widths = [] {
    std::vector<PhaseWidth> widths;
    for (std::uint64_t w = 1; w <= 5; ++w) {
      widths.push_back({.positions = w, .working = Bytes(kUnit + 1)});
    }
    return widths;
  };
  return {
      .context_limit = 4096,
      .states = {Kv("kv", 16),
                 {.name = "ssm",
                  .block_positions = 0,
                  .block_bytes = U(2),
                  .capabilities = StateCapability::kAppend | StateCapability::kTruncateAtSnapshot,
                  .max_snapshots = 5,
                  .snapshot_bytes = U(2)},
                 Kv("draft.kv", 16)},
      .working = {{.name = "draft.handoff", .bytes = Bytes(100)}},
      .phases = {Kind("prefill", Roles(kBoth), Widths({{16, 1}, {64, 3}}), WidthRule::kChunk,
                      Repeat::kPromptChunks, true, false),
                 Kind("draft", Roles(kDrafter), draft_widths(), WidthRule::kVerify,
                      Repeat::kOutputSteps, true, false),
                 Kind("verify", Roles(kMain), Widths({{1, 5}, {2, 2}, {3, 2}, {4, 2}, {5, 2}}),
                      WidthRule::kVerify, Repeat::kOutputSteps, true, true)},
      .mode = Speculative(4)};
}

constexpr std::size_t kKv = 0;
constexpr std::size_t kSsm = 1;
constexpr std::size_t kDraftKv = 2;

// Begins the verify: a snapshot of the recurrent state at the committed
// prefix and after every drafted position (`width` snapshots, never one
// after the last position, which acceptance of every draft never needs),
// with every position written tentatively.
void BeginMixedVerify(Driver& run, std::uint64_t width) {
  const std::array also = {run.working("draft.handoff")};
  ASSERT_TRUE(run.output().Reserve(Wire(width), width).has_value());
  ASSERT_TRUE(run.Begin("verify", width, also));
  run.Append(kKv, width);
  run.Snapshot(kSsm);
  for (std::uint64_t position = 0; position < width; ++position) {
    run.Append(kSsm, 1);
    if (position + 1 < width) {
      run.Snapshot(kSsm);
    }
  }
  EXPECT_EQ(run.state(kSsm).snapshots().size(), width);
}

// One cycle: the draft phase catches the drafter's KV up and drafts, the
// verify writes KV and recurrent state tentatively, `accepted` drafts and
// the verify's own token are kept, and every state rolls back to its new
// prefix. Returns the positions kept.
std::uint64_t MixedStep(Driver& run, std::uint32_t depth, std::uint32_t accepted) {
  const std::uint64_t width = std::min<std::uint64_t>(depth + 1, run.state(kKv).room());
  if (width == 0) {
    return 0;
  }
  const std::uint64_t drafted = width - 1;
  const std::uint64_t committed = run.state(kKv).committed();
  const ExtentId handoff = run.working("draft.handoff");
  const std::array also = {handoff};
  if (drafted > 0) {
    const std::uint64_t lag = committed - run.state(kDraftKv).committed();
    EXPECT_LE(lag, 1U);
    EXPECT_TRUE(run.Begin("draft", lag + drafted, also));
    run.Append(kDraftKv, lag + drafted);
    EXPECT_TRUE(run.node().catalog().ReplaceContents(handoff).has_value());
    run.End();
  }
  BeginMixedVerify(run, width);
  EXPECT_TRUE(run.node().catalog().ReplaceContents(handoff).has_value());
  const std::uint64_t keep = std::min<std::uint64_t>(accepted, drafted) + 1;
  for (const std::size_t state : {kKv, kSsm}) {
    run.Commit(state, keep);
    run.Truncate(state, run.state(state).committed());
  }
  // The drafter keeps what the verify accepted, up to what it wrote.
  const std::uint64_t target = std::min(committed + keep, run.state(kDraftKv).written());
  run.Commit(kDraftKv, target - run.state(kDraftKv).committed());
  run.Truncate(kDraftKv, target);
  EXPECT_TRUE(run.output().Publish(Wire(keep), keep).has_value());
  run.End();
  EXPECT_TRUE(run.output().Drain(Wire(keep)).has_value());
  // Nothing tentative survives the boundary; the recurrent state keeps at
  // most the snapshot at its prefix.
  EXPECT_EQ(run.state(kKv).written(), committed + keep);
  EXPECT_EQ(run.state(kSsm).written(), committed + keep);
  EXPECT_EQ(run.state(kDraftKv).written(), run.state(kDraftKv).committed());
  EXPECT_LE(run.state(kSsm).snapshots().size(), 1U);
  return keep;
}

struct Mixed {
  explicit Mixed(Node& node)
      : main(Dense(node, 1, 4, 20, 4)),
        draft_layer(node.Weights(1, 3, 3)),
        context(ModelContext::Create({{.role = ComponentRole::kMain,
                                       .artifact = Artifact(1),
                                       .resources = {main.embeddings, main.trunk, main.head}},
                                      {.role = ComponentRole::kDrafter,
                                       .artifact = Artifact(1),
                                       .resources = {draft_layer, main.embeddings, main.head}}})
                    .value()) {}
  DenseWeights main;
  ResourceId draft_layer;
  ModelContext context;
  ProgramContract contract = MixedContract();
  ProgramRequest request{.prompt = 40, .max_output = 64, .draft_depth = 4, .steps = 0};
};

// Full verify cycles over paged KV, snapshot-only recurrent state and the
// drafter's own KV, at every acceptance count from none to all, cycles
// after all were accepted, a pause right after one (with the handoff and
// the snapshot at the prefix protected in R_i), and verifies narrowing to
// one position at the output bound, where the verify's working set is at
// its largest. The budget is exactly F + R_A + R_B + E_A.
TEST(ShapeScenarioTest, MixedStateSpeculationAtEveryAcceptanceCount) {
  Node node(66, 2);
  Mixed model(node);
  const ProgramPlan plan = Plan(node, model.contract, model.context, model.request);
  // R_i: KV and drafter KV through 104 positions (seven blocks each), the
  // recurrent state and its five snapshots, the ragged handoff and output.
  EXPECT_EQ(plan.Retained("ssm")->bytes, U(2 + 10));
  EXPECT_EQ(plan.Retained("draft.handoff")->bytes, U(1));
  EXPECT_EQ(plan.envelope.retained, U(7 + 12 + 7 + 1 + 1));
  // E_i: the prefill over both components is widest; the verify is charged
  // its narrowest width's working set, and the ragged draft working set
  // whole units.
  EXPECT_EQ(plan.Phase("verify")->envelope, U(28 + 5));
  EXPECT_EQ(plan.Phase("draft")->envelope, U(11 + 2));
  EXPECT_EQ(plan.envelope.phase, U(34));
  const DenseWeights small = Dense(node, 2, 1, 4, 1);
  const ModelContext other = Single(2, small);
  const ProgramContract autoregressive = AutoregressiveContract(1);
  const ProgramPlan plan_b = Plan(node, autoregressive, other,
                                  {.prompt = 32, .max_output = 16, .draft_depth = 0, .steps = 0});
  ASSERT_EQ(Add(Add(Add(node.fixed(), plan.envelope.retained), plan_b.envelope.retained),
                plan.envelope.phase),
            node.budget());

  Driver a(node, model.contract, model.context, plan);
  ASSERT_EQ(a.Submit(RequestClass::kBackground), RequestState::kRunning);
  a.Start();
  Prefill(a);
  for (const std::size_t state : {kKv, kSsm, kDraftKv}) {
    EXPECT_EQ(a.state(state).committed(), 40U);
  }
  for (const std::uint32_t accepted : {0U, 1U, 2U, 3U, 4U, 4U, 0U}) {
    EXPECT_EQ(MixedStep(a, 4, accepted), accepted + 1U);
  }
  // Paused right after a verify that accepted every draft: the drafter's
  // KV is a position behind, the recurrent state holds no snapshot, and
  // the handoff holds the verify's last hidden state.
  EXPECT_EQ(MixedStep(a, 4, 4), 5U);
  const std::uint64_t committed = a.state(kKv).committed();
  EXPECT_EQ(a.state(kDraftKv).committed(), committed - 1);
  EXPECT_THAT(a.state(kSsm).snapshots(), IsEmpty());
  const auto handoff = node.catalog().Describe(a.working("draft.handoff")).value();
  Driver b(node, autoregressive, other, plan_b);
  ASSERT_EQ(b.Submit(RequestClass::kInteractive), RequestState::kWaiting);
  ASSERT_EQ(a.Boundary().paused, a.id());
  b.Start();
  Prefill(b);
  Decode(b);
  const auto held = node.catalog().Describe(a.working("draft.handoff")).value();
  EXPECT_EQ(held.state, llmp::catalog::ExtentState::kResident);
  EXPECT_EQ(held.content_generation, handoff.content_generation);
  EXPECT_EQ(a.state(kKv).committed(), committed);
  EXPECT_EQ(b.Retire().run, a.id());

  // Resumed: the next draft catches up and every acceptance count recurs,
  // then the verifies narrow, one position at a time, to the bound.
  std::size_t step = 0;
  const std::array accepted = {1U, 4U, 3U, 0U, 2U};
  while (a.state(kKv).room() > 5) {
    (void)MixedStep(a, 4, accepted.at(step++ % accepted.size()));
  }
  while (a.state(kKv).room() > 0) {
    EXPECT_EQ(MixedStep(a, 4, 0), 1U);
  }
  for (const std::size_t state : {kKv, kSsm}) {
    EXPECT_EQ(a.state(state).committed(), plan.state_bound);
  }
  EXPECT_EQ(a.output().published_positions(), plan.output);
  EXPECT_EQ(Failed(a.output().Reserve(Wire(1), 1)), ProgramError::kBeyondProgram);
  (void)a.Retire();
  node.ExpectDrained();
}

// A verify that fails after writing its positions rolls every state back
// to its committed prefix, the recurrent state through the snapshot taken
// at that prefix; nothing is published, and the request ends explicitly.
TEST(ShapeScenarioTest, MixedVerifyFailureRollsBackThroughTheSnapshot) {
  Node node(64, 2);
  Mixed model(node);
  const ProgramPlan plan = Plan(node, model.contract, model.context, model.request);
  Driver run(node, model.contract, model.context, plan);
  ASSERT_EQ(run.Submit(RequestClass::kInteractive), RequestState::kRunning);
  run.Start();
  Prefill(run);
  (void)MixedStep(run, 4, 2);
  const std::uint64_t committed = run.state(kKv).committed();
  const std::uint64_t published = run.output().published_positions();
  const std::array also = {run.working("draft.handoff")};
  ASSERT_TRUE(run.Begin("draft", 4, also));
  run.Append(kDraftKv, 4);
  run.End();
  BeginMixedVerify(run, 5);
  for (const std::size_t state : {kKv, kSsm, kDraftKv}) {
    run.Truncate(state, run.state(state).committed());
  }
  run.output().Cancel();
  EXPECT_EQ(run.state(kSsm).written(), committed);
  EXPECT_THAT(run.state(kSsm).snapshots(), ElementsAre(committed));
  EXPECT_EQ(run.state(kKv).written(), committed);
  EXPECT_EQ(run.output().published_positions(), published);
  (void)run.Retire();
  node.ExpectDrained();
}

// ---- Scenario 2: a companion drafter ------------------------------------------

// A target and a companion drafter from two artifacts, the drafter sharing
// the target's embeddings: one closure over both is leased and charged
// once, idle cache of another model gives way to it, a lease over a closure
// with an extent reclaimed in between is refused whole, and both live
// states roll back together after a rejected draft.
TEST(ShapeScenarioTest, CompanionDrafterSharesOneClosure) {
  // F + R + E = 2 + 11 + 35.
  Node node(48, 2);
  const DenseWeights target = Dense(node, 1, 4, 20, 4);
  const ResourceId drafter_layers = node.Weights(2, 0, 3);
  const ResourceId drafter_head = node.Weights(2, 1, 1);
  const ModelContext context =
      ModelContext::Create({{.role = ComponentRole::kMain,
                             .artifact = Artifact(1),
                             .resources = {target.embeddings, target.trunk, target.head}},
                            {.role = ComponentRole::kDrafter,
                             .artifact = Artifact(2),
                             .resources = {drafter_layers, drafter_head, target.embeddings}}})
          .value();
  EXPECT_EQ(context.SharedBytes(node.catalog(), node.domain()).value(), U(4));
  const ProgramContract contract = CompanionContract();
  const ProgramPlan plan =
      Plan(node, contract, context, {.prompt = 40, .max_output = 32, .draft_depth = 4, .steps = 0});
  EXPECT_EQ(plan.Phase("speculate")->closure, U(32));  // not 28 + 8
  EXPECT_EQ(plan.envelope.retained, U(5 + 5 + 1));
  EXPECT_EQ(plan.envelope.phase, U(35));

  // Another model's weights sit idle in memory.
  const DenseWeights cached = Dense(node, 9, 2, 4, 2);
  const std::array cached_resources = {cached.embeddings, cached.trunk, cached.head};
  ASSERT_TRUE(node.Materialize(node.catalog().ClosureOf(cached_resources).value()));

  Driver run(node, contract, context, plan);
  ASSERT_EQ(run.Submit(RequestClass::kInteractive), RequestState::kRunning);
  run.Start();
  Prefill(run);
  EXPECT_EQ(run.state(1).committed(), 40U);
  EXPECT_GT(node.evictions(), 0U);  // the other model's idle weights gave way

  // The drafter's layers are reclaimed while no lease holds them: a lease
  // on the closure taken before is refused whole, holding nothing.
  const Closure closure = context.ClosureOf(node.catalog(), kBoth).value();
  const auto layers = node.catalog().ClosureOf(std::array{drafter_layers}).value();
  node.Evict(layers.extents.front().first);
  EXPECT_EQ(Failed(node.catalog().AcquireLease(closure)),
            llmp::catalog::CatalogError::kNotResident);
  EXPECT_EQ(node.catalog().OccupancyOf(node.domain()).held, Bytes());

  // Inside the phase the union is held once: 32 units of weights and its
  // 3 of working set, and the shared embeddings' backing exists once.
  EXPECT_TRUE(run.output().Reserve(Wire(5), 5).has_value());
  ASSERT_TRUE(run.Begin("speculate", 5));
  EXPECT_EQ(node.catalog().OccupancyOf(node.domain()).held, U(35));
  run.Append(0, 5);
  run.Append(1, 5);
  run.Commit(0, 2);  // one draft accepted, and the verify's token
  run.Commit(1, 2);
  run.Truncate(0, run.state(0).committed());
  run.Truncate(1, run.state(1).committed());
  EXPECT_TRUE(run.output().Publish(Wire(2), 2).has_value());
  run.End();
  EXPECT_TRUE(run.output().Drain(Wire(2)).has_value());
  EXPECT_EQ(run.state(0).written(), 42U);
  EXPECT_EQ(run.state(1).written(), 42U);
  // Release is not eviction: the closure stays resident, now idle.
  EXPECT_EQ(node.catalog().OccupancyOf(node.domain()).held, Bytes());
  EXPECT_TRUE(
      node.catalog()
          .AcquireLease(context.ClosureOf(node.catalog(), kBoth).value())
          .transform([&](LeaseId lease) { return node.catalog().ReleaseLease(lease).has_value(); })
          .value_or(false));

  const std::array accepted = {4U, 0U, 1U, 3U};
  std::size_t step = 0;
  while (run.state(0).room() > 0) {
    (void)SpeculativeStep(run, 4, accepted.at(step++ % accepted.size()), false, "speculate");
    EXPECT_EQ(run.state(0).committed(), run.state(1).committed());
  }
  EXPECT_EQ(run.output().published_positions(), plan.output);
  (void)run.Retire();
  node.ExpectDrained();
}

// ---- Scenario 3: a canvas across a pause ----------------------------------------

struct Diffusion {
  explicit Diffusion(Node& node) : weights(Dense(node, 3, 4, 20, 4)), context(Single(3, weights)) {}
  DenseWeights weights;
  ModelContext context;
  ProgramRequest request{.prompt = 64, .max_output = 500, .draft_depth = 0, .steps = 4};
};

// A background diffusion request, paused between two denoising steps of
// its first block for an interactive request (D-069). Its canvas is in
// R_i, so it stays resident and unchanged through the pause while the
// substitute's phases evict only the paused request's idle weights; no
// phase waits on another's allowance. It resumes, finishes the block, and
// commits the second block clamped to the output bound. Nothing commits
// while it is paused.
TEST(ShapeScenarioTest, CanvasHeldAcrossAPause) {
  // F + R_A + R_B + E_A = 2 + 18 + 2 + 34.
  Node node(56, 2);
  Diffusion model(node);
  const ProgramContract contract = DiffusionContract();
  const ProgramPlan plan = Plan(node, contract, model.context, model.request);
  const DenseWeights other_weights = Dense(node, 4, 2, 16, 2);
  const ModelContext other = Single(4, other_weights);
  const ProgramContract autoregressive = AutoregressiveContract(1);
  const ProgramPlan plan_b = Plan(node, autoregressive, other,
                                  {.prompt = 32, .max_output = 16, .draft_depth = 0, .steps = 0});
  EXPECT_EQ(plan_b.envelope.phase, U(21));

  Driver a(node, contract, model.context, plan);
  ASSERT_EQ(a.Submit(RequestClass::kBackground), RequestState::kRunning);
  a.Start();
  Prefill(a);
  const ExtentId canvas = a.working("canvas");
  Denoise(node, a, canvas);
  Denoise(node, a, canvas);

  Driver b(node, autoregressive, other, plan_b);
  ASSERT_EQ(b.Submit(RequestClass::kInteractive), RequestState::kWaiting);
  const Decision pause = a.Boundary();
  ASSERT_EQ(pause.paused, a.id());
  ASSERT_EQ(pause.run, b.id());
  const auto held = node.catalog().Describe(canvas).value();
  const std::size_t evictions = node.evictions();
  b.Start();
  Prefill(b);
  Decode(b);
  EXPECT_GT(node.evictions(), evictions);  // A's idle weights made room
  const auto after = node.catalog().Describe(canvas).value();
  EXPECT_EQ(after.state, llmp::catalog::ExtentState::kResident);
  EXPECT_EQ(after.content_generation, held.content_generation);
  EXPECT_EQ(after.backing_generation, held.backing_generation);
  EXPECT_EQ(a.state(0).committed(), 64U);  // the paused canvas committed nothing
  EXPECT_EQ(a.output().published_positions(), 0U);
  EXPECT_EQ(b.Retire().run, a.id());

  Denoise(node, a, canvas);
  Denoise(node, a, canvas);
  EXPECT_EQ(CommitBlock(a, canvas).value_or(0), kBlock);
  EXPECT_TRUE(a.output().Drain(Wire(kBlock)).has_value());
  for (int step = 0; step < 4; ++step) {
    Denoise(node, a, canvas);
  }
  EXPECT_EQ(CommitBlock(a, canvas).value_or(0), 500 - kBlock);  // clamped, still whole
  EXPECT_EQ(a.state(0).committed(), plan.state_bound);
  EXPECT_EQ(a.output().published_positions(), 500U);
  EXPECT_EQ(Failed(a.cursor().Begin("denoise", kBlock)), ProgramError::kBeyondProgram);
  (void)a.Retire();
  node.ExpectDrained();
}

// The negative control: a contract that charged the canvas to its phases
// would be admitted beside a large interactive request at this budget, and
// pausing it after its first block would leave the substitute unable to
// materialize its first phase: the canvas and the paused KV outlive the
// boundary outside R(G). The substitute would wait for memory only the
// paused request frees, which waits for the substitute: a circular wait.
// With the canvas in R_i, the same request is not admitted beside it, so
// it waits in the queue holding nothing and runs once the canvas retires.
TEST(ShapeScenarioTest, CanvasChargedToThePhaseWouldDeadlockAPause) {
  Node node(56, 2);
  Diffusion model(node);
  const DenseWeights large_weights = Dense(node, 5, 2, 36, 2);
  const ModelContext large = Single(5, large_weights);
  const ProgramContract autoregressive = AutoregressiveContract(2);
  const ProgramPlan plan_b = Plan(node, autoregressive, large,
                                  {.prompt = 32, .max_output = 16, .draft_depth = 0, .steps = 0});
  ASSERT_EQ(plan_b.envelope.phase, U(42));
  const RequestSpec spec_b{.request_class = RequestClass::kInteractive,
                           .envelope = plan_b.envelope,
                           .work = plan_b.max_phases,
                           .switch_cost = 1,
                           .deadline = std::nullopt};

  {
    // Charged to the phase: R 10, E 42, and the ledger admits both.
    const ProgramContract naive = DiffusionContract(/*canvas_in_phase=*/true);
    const ProgramPlan plan = Plan(node, naive, model.context, model.request);
    const RequestSpec spec_a{.request_class = RequestClass::kBackground,
                             .envelope = plan.envelope,
                             .work = plan.max_phases,
                             .switch_cost = 1,
                             .deadline = std::nullopt};
    Admission& admission = node.admission();
    const RequestId a = admission.Submit(spec_a, node.Tock()).value().id;
    ASSERT_EQ(admission.StateOf(a), RequestState::kRunning);
    // After its first block: KV through 320 positions, the canvas it keeps
    // between steps, its output buffer, and its weights, now idle.
    std::vector<ExtentId> kept;
    kept.reserve(7);
    for (int block = 0; block < 5; ++block) {
      kept.push_back(node.Allocate(MemoryClass::kLiveState, Recovery::kPreserve, U(1)));
    }
    kept.push_back(node.Allocate(MemoryClass::kLiveState, Recovery::kPreserve, U(8)));
    kept.push_back(node.Allocate(MemoryClass::kRuntime, Recovery::kPinned, U(1)));
    ASSERT_TRUE(node.Materialize(model.context.ClosureOf(node.catalog(), kMain).value()));

    const RequestId b = admission.Submit(spec_b, node.Tock()).value().id;
    ASSERT_EQ(admission.StateOf(b), RequestState::kWaiting);  // 2 + 10 + 2 + 42 = 56
    ASSERT_EQ(admission.Boundary(a, 6, node.Tock()).value().paused, a);
    // The harness's invariant catches it at this boundary: the canvas
    // outlives it outside R(G), protecting 2 + 14 against 2 + 10 + 2.
    EXPECT_EQ(node.Protected(), U(16));
    EXPECT_EQ(node.AtBoundary(), U(14));
    const ExtentId b_output = node.Allocate(MemoryClass::kRuntime, Recovery::kPinned, U(1));
    const Closure closure = large.ClosureOf(node.catalog(), kMain).value();
    const auto materialization =
        llmp::memory::PlanMaterialization(node.catalog(), node.domain(), node.budget(), closure);
    EXPECT_FALSE(materialization.feasible);  // even with all of A's weights evicted
    EXPECT_FALSE(materialization.victims.sufficient);

    // Unwind the construction.
    node.Free(b_output);
    for (const ExtentId extent : kept) {
      node.Free(extent);
    }
    ASSERT_TRUE(admission.Retire(a, node.Tock()).has_value());  // b now runs
    ASSERT_TRUE(admission.Retire(b, node.Tock()).has_value());
    node.ExpectDrained();
  }

  // The contract: R 18, E 34. B does not fit beside it (2 + 18 + 2 + 42),
  // so it is queued without a grant, never a pause's substitute.
  const ProgramContract contract = DiffusionContract();
  const ProgramPlan plan = Plan(node, contract, model.context, model.request);
  Driver a(node, contract, model.context, plan);
  ASSERT_EQ(a.Submit(RequestClass::kBackground), RequestState::kRunning);
  a.Start();
  Prefill(a);
  Driver b(node, autoregressive, large, plan_b);
  ASSERT_EQ(b.Submit(RequestClass::kInteractive), RequestState::kQueued);
  const ExtentId canvas = a.working("canvas");
  for (int block = 0; block < 2; ++block) {
    for (int step = 0; step < 4; ++step) {
      Denoise(node, a, canvas);
      EXPECT_FALSE(a.Boundary().paused.has_value());
      EXPECT_EQ(node.admission().StateOf(b.id()), RequestState::kQueued);
    }
    ASSERT_TRUE(CommitBlock(a, canvas).has_value());
    EXPECT_TRUE(a.output().Drain(a.output().used()).has_value());
  }
  EXPECT_EQ(a.Retire().run, b.id());
  b.Start();
  Prefill(b);
  Decode(b);
  (void)b.Retire();
  node.ExpectDrained();
}

// A paused canvas that is cancelled never commits: its request retires
// with the canvas discarded, and the substitute runs on.
TEST(ShapeScenarioTest, CancelledPausedCanvasNeverCommits) {
  Node node(56, 2);
  Diffusion model(node);
  const ProgramContract contract = DiffusionContract();
  const ProgramPlan plan = Plan(node, contract, model.context, model.request);
  const DenseWeights other_weights = Dense(node, 4, 2, 16, 2);
  const ModelContext other = Single(4, other_weights);
  const ProgramContract autoregressive = AutoregressiveContract(1);
  const ProgramPlan plan_b = Plan(node, autoregressive, other,
                                  {.prompt = 32, .max_output = 16, .draft_depth = 0, .steps = 0});

  Driver a(node, contract, model.context, plan);
  ASSERT_EQ(a.Submit(RequestClass::kBackground), RequestState::kRunning);
  a.Start();
  Prefill(a);
  Denoise(node, a, a.working("canvas"));
  Driver b(node, autoregressive, other, plan_b);
  ASSERT_EQ(b.Submit(RequestClass::kInteractive), RequestState::kWaiting);
  ASSERT_EQ(a.Boundary().paused, a.id());
  b.Start();
  Prefill(b);
  const Decision cancelled = a.Retire();
  EXPECT_FALSE(cancelled.run.has_value());  // the substitute runs on
  EXPECT_EQ(a.state(0).committed(), 64U);
  EXPECT_EQ(a.output().published_positions(), 0U);
  EXPECT_EQ(node.admission().Running(), std::vector<RequestId>{b.id()});
  Decode(b);
  (void)b.Retire();
  node.ExpectDrained();
}

// ---- Scenario 4: block output, all or none ---------------------------------------

// A block commit starts only when the output buffer has room for the whole
// block; one that fails mid-phase rolls its positions back and publishes
// nothing, and the request ends explicitly with nothing leaked.
TEST(ShapeScenarioTest, BlockCommitIsAllOrNone) {
  Node node(56, 2);
  Diffusion model(node);
  const ProgramContract contract = DiffusionContract();
  const ProgramPlan plan = Plan(node, contract, model.context, model.request);
  Driver run(node, contract, model.context, plan);
  ASSERT_EQ(run.Submit(RequestClass::kInteractive), RequestState::kRunning);
  run.Start();
  Prefill(run);
  const ExtentId canvas = run.working("canvas");
  for (int step = 0; step < 4; ++step) {
    Denoise(node, run, canvas);
  }
  ASSERT_EQ(CommitBlock(run, canvas).value_or(0), kBlock);
  EXPECT_EQ(run.output().published_positions(), kBlock);
  for (int step = 0; step < 4; ++step) {
    Denoise(node, run, canvas);
  }
  // The client has not read the first block: the second waits at the
  // boundary with nothing started.
  const std::size_t blocks = run.blocks(0);
  EXPECT_EQ(Failed(CommitBlock(run, canvas)), ProgramError::kOutputFull);
  EXPECT_TRUE(run.cursor().AtBoundary());
  EXPECT_EQ(run.state(0).committed(), 64 + kBlock);
  EXPECT_EQ(run.blocks(0), blocks);
  ASSERT_TRUE(run.output().Drain(Wire(kBlock)).has_value());

  // The commit fails after writing its positions (an operation error with
  // known completion): they roll back and nothing is published.
  const std::uint64_t positions = std::min(kBlock, run.state(0).room());
  ASSERT_TRUE(run.output().Reserve(Wire(positions), positions).has_value());
  const std::array also = {canvas};
  ASSERT_TRUE(run.Begin("commit", kBlock, also));
  run.Append(0, positions);
  EXPECT_GT(run.blocks(0), blocks);
  run.Truncate(0, run.state(0).committed());
  run.output().Cancel();
  EXPECT_EQ(run.blocks(0), blocks);
  EXPECT_EQ(run.state(0).committed(), 64 + kBlock);
  EXPECT_EQ(run.output().published_positions(), kBlock);
  EXPECT_EQ(run.output().used(), Bytes());
  (void)run.Retire();  // drains the phase and ends the request
  node.ExpectDrained();
}

// Block output does not require truncation: a commit that fails ends its
// request (D-050's explicit unwind), so an append-only state that cannot
// roll the failed block back is discarded whole at retirement, with its
// tentative positions never committed and nothing published.
TEST(ShapeScenarioTest, FailedBlockCommitOnAppendOnlyStateEndsTheRequest) {
  Node node(56, 2);
  Diffusion model(node);
  ProgramContract contract = DiffusionContract();
  contract.states[0].capabilities = static_cast<std::uint8_t>(StateCapability::kAppend);
  const ProgramPlan plan = Plan(node, contract, model.context, model.request);
  Driver run(node, contract, model.context, plan);
  ASSERT_EQ(run.Submit(RequestClass::kInteractive), RequestState::kRunning);
  run.Start();
  Prefill(run);
  const ExtentId canvas = run.working("canvas");
  for (int step = 0; step < 4; ++step) {
    Denoise(node, run, canvas);
  }
  ASSERT_TRUE(run.output().Reserve(Wire(kBlock), kBlock).has_value());
  const std::array also = {canvas};
  ASSERT_TRUE(run.Begin("commit", kBlock, also));
  run.Append(0, kBlock);
  // The commit fails: nothing can roll back, so nothing is committed or
  // published, and the request ends.
  EXPECT_EQ(Failed(run.state(0).Truncate(64)), StateError::kUnsupported);
  run.output().Cancel();
  EXPECT_EQ(run.state(0).committed(), 64U);
  EXPECT_EQ(run.output().published_positions(), 0U);
  (void)run.Retire();
  node.ExpectDrained();
}

}  // namespace
