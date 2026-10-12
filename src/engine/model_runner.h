// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The one model runner (docs/engine.md, "The model runner";
// docs/engine-components.md, "Execution"; D-107): the shared skeleton
// (weights, the slot lifecycle with its live states, plans, step runs, the
// runner's own memory) composed once, over a step source that turns step
// shapes into step programs. It replaces each family's runner as the
// families migrate; no family code is here.
//
// Its life (the order every runner had): Setup (the slots' state regions
// first, then the weights' places, then every setup shape measured over
// placeless activations to size the workspace, the staging and the plan
// floor, and the staging pinned), Register (each source, the places
// pinned), Bind (the launch, the plan account, the closures), steps, and
// Release after the node's teardown (the plans and their recordings first,
// then the launch, then the memory they name).
//
// A step (RunStep) runs one composed shape: each owner checked active and
// at its positions, its state backed through what the step reads, the
// plan found or built (StepSource::Build, then StepProgram::Bind and the
// coverage check, BP-A1), its inputs staged, a write recorded for every
// owner, one device job queueing what the owners' states owe, the run and
// the output copies, and after its completion each owner's positions and
// its own outputs. A verify (D-068) first saves what it writes in every
// stage, from the stages' state contracts (model::WritesFor), into each
// stage's snapshot, queued in its job before its run; its rows then await
// the slot's Accept (SlotLifecycle::Accept), and a failed one is undone
// whole. The decoding modes that drive drafts and verifies are
// decoding.h's. Around it, what a caller predicts (StepAhead): up to two
// future shapes built while the device runs this one (the prefill
// lookahead), the next one's recording captured beside this run, and the
// next one's fresh state prepared before the job. Policy (which futures,
// which captures, what joins) is the orchestrators' (wave_composer.h,
// prefill.h); the mechanics are here.
//
// Part of the engine core: no kernel library, every build profile.

#ifndef LLMP_ENGINE_MODEL_RUNNER_H_
#define LLMP_ENGINE_MODEL_RUNNER_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "catalog/catalog.h"
#include "engine/coverage.h"
#include "engine/execution_contract.h"
#include "engine/paged_node.h"
#include "engine/paged_weights.h"
#include "engine/plan_account.h"
#include "engine/plan_cache.h"
#include "engine/ple_rows.h"
#include "engine/run_stats.h"
#include "engine/runner_memory.h"
#include "engine/slot_lifecycle.h"
#include "engine/step_runs.h"
#include "format/encoding.h"
#include "format/tensor_format.h"
#include "ir/graph.h"
#include "kernels/common/routed_program.h"
#include "kernels/common/step_program.h"
#include "memory/reclaim.h"
#include "planner/step_shape.h"

namespace llmp::test_support {
class RoutedComponentInspection;
}

namespace llmp::engine {

// What the runner asks for a step program: one virtual call per plan miss
// (the planner and a lowering behind it: ComponentStepSource once the
// components exist; FakeStepSource in the tests).
struct CommitBuffer {
  std::uint32_t layer = 0;
  std::uint16_t component = 0;
  std::string what;
  std::uint64_t row_bytes = 0;
};

class StepSource {
 public:
  StepSource() = default;
  StepSource(const StepSource&) = delete;
  StepSource& operator=(const StepSource&) = delete;
  StepSource(StepSource&&) = delete;
  StepSource& operator=(StepSource&&) = delete;
  virtual ~StepSource() = default;
  // What every plan of this source shares in its key (the spec's digest,
  // the pattern set, the policy, the partition); the runner sets the shape.
  // Asked once, at Bind.
  virtual planner::PlanKey KeyBase() const = 0;
  // After paid kernel prepare, before measuring any source shape. Required
  // sources publish one immutable whole-composition identity here.
  virtual Status PrepareExecution() { return {}; }
  virtual Status CheckExecutionUse(NumericalUse) const { return {}; }
  virtual Status CheckExecutionShape(const planner::StepShape&) const { return {}; }
  virtual std::expected<std::optional<base::Sha256Digest>, std::string> ExecutionIdentity() const {
    return std::optional<base::Sha256Digest>{};
  }
  // Only baked graph dimensions that depend on positions; empty when
  // positions affect input data alone. Pure, also called by lookahead.
  virtual std::vector<std::uint64_t> ShapeParameters(const planner::StepShape&) const { return {}; }
  // Immutable numerical qualification for this stage's contribution to a
  // multi-owner repeat-pending submission (including target, companions,
  // head-bearing replay and continuation). The runner requires every
  // stage's source to qualify the entire shape domain it accepts. This is
  // a backend proof, never a caller batching preference. Component sources
  // remain unqualified until joined arithmetic has its exact GPU proof.
  virtual bool JoinedRepeatPending(std::uint16_t /*stage*/) const { return false; }
  // A side-table component's row IDs from the owner's complete token
  // history. Pure host work before any state admission or device writes.
  virtual bool UsesSideRows(std::uint16_t /*stage*/) const { return false; }
  virtual Status SideRows(std::uint16_t /*stage*/, std::uint16_t /*component*/,
                          std::span<const std::int32_t> /*history*/, std::uint32_t /*position*/,
                          std::span<std::int32_t> /*out*/) const {
    return std::unexpected("the step source cannot derive side rows");
  }
  // The intermediates a stage's verify stores for its commit, per owner.
  // Empty for a source without commit-based state. Queried once at setup.
  virtual std::expected<std::vector<CommitBuffer>, std::string> CommitBuffers(
      std::uint16_t /*stage*/, std::uint32_t /*rows*/) const {
    return std::vector<CommitBuffer>{};
  }
  // A program for `shape` at `addresses`. Host-only: it may run on the
  // driver while the device runs the current step (a lookahead), so it
  // never calls the node, the launch or the device; the runner binds it
  // (StepProgram::Bind) after. A measuring build (addresses.measuring())
  // only reports what it needs. The program serves every step with an equal
  // shape and ShapeParameters, whatever its segments' `past`: positions
  // reach it as input data, or, for a program that is not capturable,
  // through StepProgram::Position before each run; nothing it bakes may
  // follow `past` unless included in ShapeParameters.
  virtual std::expected<std::unique_ptr<kernels::StepProgram>, std::string> Build(
      const planner::StepShape& shape, const kernels::Addresses& addresses) = 0;
};

// One stage's weights, opened by the caller; Setup reserves their places.
struct StageWeights {
  std::unique_ptr<PagedWeights> weights;
  std::vector<GroupPlace> places;  // one per group
  std::vector<SlabSpec> slabs;
};

struct RunnerConfig {
  // The slots and each stage's state (the target's, then each drafter's).
  // The runner chains its own plan drop into `drop_plans`.
  SlotLifecycleOptions slots;
  std::vector<StageWeights> weights;
  // The widest shapes a step may take, measured at Setup over placeless
  // activations: the workspace, the staging and the plan floor are their
  // largest needs with a quarter's margin. At least one.
  std::vector<planner::StepShape> setup_shapes;
  // Graphs on (D-090), and the phases a shape is captured in on its second
  // run (bit 1 << Phase; a hinted repeat or a capture ahead is the step's
  // own request). Qwen3.8 and DeepSeek capture prefill only from a hint.
  bool graphs = true;
  std::uint32_t capture_phases = 0x1F;
  // Device memory the step source's kernels hold beside the node's
  // workspace (a cuBLAS workspace): each mapped at Setup before any
  // measurement, of the runtime class, pinned and in every closure as the
  // runner's own memory. `prepare` then makes what uses it (given each
  // one's base, in order: a cuBLAS handle on the model's stream), and
  // `retire` unmakes it at Release (also after a `prepare` that failed part
  // way), after the plans, their recordings and the launch are gone and
  // before the memory is released.
  struct KernelMemory {
    std::string name;
    std::uint64_t bytes = 0;
  };
  std::vector<KernelMemory> kernel_memory;
  std::function<std::expected<void, std::string>(std::span<const std::uint64_t> bases)> prepare;
  std::function<void()> retire;
  // The step launch, made at Bind (after the workspace is mapped).
  std::function<std::expected<std::unique_ptr<kernels::StepLaunch>, std::string>()> launch;
  // A weight's address by role and part (the spec's binding to the
  // weights); unset: 0, for sources that bind none.
  std::function<std::uint64_t(const ir::WeightRef&, format::PartRole)> weight_address;
  // Tables whose rows are read into bounded device slots before a step.
  // All row-addressable parts of each weight must be declared exactly once.
  // Their groups must contain only that resource and be placed kNone; a
  // matrix-wide tensor scale stays resident through ordinary weight binding.
  struct RowTable {
    ir::WeightRef weight;
    std::uint32_t resource = 0;
    format::TensorFormat format;
    format::PartRole part = format::PartRole::kData;
    std::uint64_t rows_per_token = 0;
    // Complete logical v0 assembly, including non-row tensor scales. Empty
    // retains ordinary resources whose recorded format already is logical.
    std::vector<std::pair<format::PartRole, std::uint32_t>> source_parts{};
  };
  std::vector<RowTable> row_tables;
  // Explicit routed execution support. The weak owner names the caller's
  // complete node/source/runner environment; it grants lifetime, not source
  // or residency authority. A step locks it before effects and retains it on
  // uncertain retirement. Metadata uses an admitted catalog allowance.
  std::weak_ptr<const void> routed_owner;
  std::shared_ptr<ExecutionMetadataAllowance> routed_metadata;
  // Speculation: the most rows a verify takes (0: none; no snapshot is
  // mapped and a verify is refused). Setup maps each slot's stages'
  // snapshots, sized by model::VerifySaveBound, for the stages a target's
  // verify writes (the target, a drafter's feature taps) or that commit,
  // and refuses a feature tap ring of fewer cells than a verify's rows
  // (each verify row's feature has a cell of its own).
  std::uint32_t max_verify_rows = 0;
  // A stage's commit of a verify's kept rows (LiveState::Commit: Qwen3.8's
  // recurrent state), per slot; unset or empty: none.
  std::function<LiveState::Commit(std::uint32_t slot, std::size_t stage)> commit;
};

// Source validation with fixed part scratch, before row-table storage is created.
Status CheckRowTableSource(const RunnerConfig::RowTable& table, std::uint32_t version,
                           std::span<const artifact::Resource> resources);

// Where a segment's rows read their feature (a drafter's rows: the step
// program's kFeatures input, components/step.h kFeatures), copied on the
// device before the step's run, outside any recording (it moves with the
// positions).
struct FeatureFrom {
  enum class Kind : std::uint8_t {
    kNone,
    // The step's stage's feature tap (model::FeatureTap: the target's
    // features by position, which its steps write): row r reads position
    // `position + r`'s committed cell in the ring or currently held chunk.
    // A chunk tap retains its preceding tail through ordinary spill/restore;
    // checkpoint replacement or adoption requires a validated durable
    // repeat-pending control before lookup. Without that policy, replacement
    // clears the transient span until a target step writes fresh features.
    kTap,
    // The slot's carry on the step's stage: the next features the last
    // drafter step with features of that slot on that stage returned (its
    // segment's rows, one a row; RunStep carries them after every such
    // step and forgets them when the slot's state changes otherwise).
    kCarry,
  };
  Kind kind = Kind::kNone;
  std::uint32_t position = 0;
};

// One segment's host data for a step, by the shape's segment order.
struct SegmentWork {
  std::span<const std::int32_t> tokens{};  // kTokens inputs: one a row
  std::span<const std::byte> bytes{};      // kSegmentBytes inputs
  FeatureFrom feature{};                   // kFeatures inputs (a drafter's rows)
  // Optional full history for declarative side-row lookups. The rows at
  // the declared token coordinate (cache past + relation offset) must equal
  // tokens; an explicit StepRequest side_rows
  // remains authoritative and is never partially replaced from history.
  std::span<const std::int32_t> history{};
};

// What the caller predicts beyond the step: never work, and a wrong
// prediction costs only an unused plan, recording or zero page.
struct StepAhead {
  // The next step's shape and the one after it, from this step's owners
  // continuing (the prefill lookahead's near and far futures).
  std::array<std::optional<planner::StepShape>, 2> shapes{};
  // Distinct missing futures built while the device runs this step (0 to
  // 2; a threaded node only), each under its own optional allowance.
  std::uint32_t build = 0;
  // The next shape's recording captured beside this run when it differs
  // and is planned; or, when it repeats this shape, this shape captured on
  // its first run (prefill reads grow every alignment, so a second-run
  // capture of a chunk shape would never replay).
  bool capture_ahead = false;
  // The next shape's fresh state prepared before this job (a threaded node
  // only; drained after it).
  bool prepare = false;
};

struct StepRequest {
  planner::StepShape shape{};
  std::span<const SegmentWork> work{};      // one per segment
  std::span<const std::byte> step_bytes{};  // kStepBytes inputs
  // kSideRows inputs: the rows of a side table each row reads (a source's
  // lookup from the token history, components/component.h side_rows), the
  // step's rows in order.
  std::span<const std::int32_t> side_rows{};
  // After success each owner's positions become past + rows (prefill,
  // decode, score); a verify or a draft leaves them to its Accept (a
  // verify or a drafter's step that would advance is refused).
  bool advance = true;
  StepAhead ahead;
};

struct StepReport {
  RunPath path = RunPath::kEager;
  std::vector<std::vector<std::byte>> outputs;  // by segment: its own output slots, in order
  std::vector<std::byte> joined;                // the step's joined outputs
  bool captured_first = false;                  // captured on its first run (a hinted repeat)
  std::uint32_t built_ahead = 0, cached_ahead = 0;
  bool captured_ahead = false;
  bool prepared_ahead = false;
};

// Why a step did not run. `clean`: nothing was queued and no owner's state
// changed beyond clean growth, so its owners stay usable and the step (or
// its peers without `slot`) may run again; otherwise every owner the step
// may have written is quarantined. `slot`: the one owner that refused (its
// state could not grow, it is not active), whose peers are untouched.
struct StepRefusal {
  std::string what{};
  bool clean = true;
  std::optional<std::uint32_t> slot = std::nullopt;
  bool state_refused = false;  // `slot`'s growth was a clean capacity refusal
  // `slot` was refused only backing a joined step's wider reads (its
  // joined alignment, a common read): what it reads alone is backed, so it
  // may step alone with no growth.
  bool widening = false;
};

struct StepLookaheadStats {
  std::uint64_t attempted = 0, built = 0, cached = 0, refused = 0;
  std::uint64_t captured_ahead = 0, captured_first = 0, dropped_ahead = 0, prepared = 0;
  double build_seconds = 0;
};

class ModelRunner final : public PagedModel {
 public:
  using Plans = BasicPlanCache<planner::PlanKey, StepPlanned, 1, StepPlanRuns>;

  static std::expected<std::unique_ptr<ModelRunner>, std::string> Create(
      PagedNode& node, RunnerConfig config, std::unique_ptr<StepSource> source, int owner,
      std::uint32_t stream);
  ModelRunner(const ModelRunner&) = delete;
  ModelRunner& operator=(const ModelRunner&) = delete;
  ModelRunner(ModelRunner&&) = delete;
  ModelRunner& operator=(ModelRunner&&) = delete;
  ~ModelRunner() override;

  // ------------------------------------------------------------ its life

  // Before the node's workspace and Start.
  Status Setup();
  // After the node's Start, before Run.
  Status Register();
  // After the node runs.
  Status Bind();
  // After the node's teardown: plans and recordings, the launch, the
  // runner's memory, the slots' state, the weights. Once.
  Status Release() override;

  std::uint32_t stream() const override { return stream_; }
  const catalog::Closure& fence_closure() const override { return slots_.fence(); }
  std::vector<catalog::ExtentId> managed_extents() const override;

  // What Setup measured (for the node's workspace and the start's guard):
  // the activations and pool the widest step needs, the host bytes a step
  // copies its inputs and outputs through, and what one step's plans hold
  // at once (the lookahead's allowance too).
  std::uint64_t activations_needed() const { return activations_needed_; }
  std::uint64_t pool_needed() const { return pool_needed_; }
  std::uint64_t host_input_bytes() const { return host_input_bytes_; }
  Status CheckExecutionUse(NumericalUse use) const { return source_->CheckExecutionUse(use); }
  std::expected<std::optional<base::Sha256Digest>, std::string> ExecutionIdentity() const {
    return source_->ExecutionIdentity();
  }
  std::uint64_t plan_floor_bytes() const { return plan_floor_bytes_; }

  SlotLifecycle& slots() { return slots_; }
  const SlotLifecycle& slots() const { return slots_; }
  // Every stage's weight extents, and what one full load of them reads
  // (a serving adapter's view of the weights, D-107).
  std::vector<catalog::ExtentId> weights() const { return weight_extents(); }
  std::uint64_t weight_read_bytes() const;
  // The most rows a verify takes (RunnerConfig::max_verify_rows).
  std::uint32_t max_verify_rows() const { return config_.max_verify_rows; }

  // ------------------------------------------------------------- steps

  // Pure preflight for an entire submitted operation, before settlement,
  // unit ownership, side inputs or admission. An unqualified repeat-pending
  // model accepts one owner per submission, including target-only work.
  Status CheckOwners(std::size_t owners) const;
  std::expected<StepReport, StepRefusal> RunStep(const StepRequest& request);
  // A segment's owner admitted to a step on `stage`: active, its rows where
  // the stage's steps may run, and its state backed through every cell the
  // step reads (`past + rows`, or `read_cells` when wider: a common read)
  // at `read_align` (0: the stage's own). Where rows may run:
  // - on the target (stage 0), advancing or not (a verify), from the
  //   slot's committed positions (`past` == SlotLifecycle::positions);
  // - on a drafter, never advancing (drafts commit nothing): from anywhere
  //   its catch-up starts, ending within one chunk bound of its stage
  //   (its layout's max_rows) past the committed positions; a drafter
  //   whose contract borrows the target's state only while the target's
  //   stage is borrowed (SlotLifecycle::Borrow).
  // Refusal names the slot; peers are untouched. RunStep admits a joined
  // step's owners through what each reads alone first (its rows at its
  // stage's alignment, SlotLifecycle::read_align(stage), as a lone step's
  // segment), then through the step's reads.
  std::expected<void, StepRefusal> Admit(const planner::Segment& segment, std::uint16_t stage,
                                         std::uint32_t read_align, bool advance);

  // ------------------------------------------------- plans and reclaim

  void ReclaimCandidates(std::uint32_t owner, bool running,
                         std::vector<memory::ReclaimCandidate>& out);
  std::uint64_t Reclaim(memory::ReclaimKind kind, std::uint64_t id);
  // Every plan and recording dropped, only between jobs.
  void DropPlans();
  const Plans& plans() const { return plans_; }
  const GraphStats& graph_stats() const { return graph_stats_; }
  const Coverage& coverage() const { return coverage_; }
  const StepLookaheadStats& lookahead_stats() const { return lookahead_; }
  bool graphs() const { return runs_.graphs(); }
  void set_graphs(bool on) { runs_.set_graphs(on); }
  // Its counts, as one JSON object (the runtime's report).
  std::string report_json() const;

 private:
  friend class llmp::test_support::RoutedComponentInspection;
  friend class PromptCompanions;
  friend class DecodingOrchestrator;
  struct AuthorizedDraft {
    std::uint32_t slot = 0;
    std::uint16_t stage = 0;
    std::uint32_t past = 0, rows = 0;
    FeatureFrom feature;
    DraftProgressLedger::Unit unit;
  };
  std::vector<AuthorizedDraft> authorized_drafts_;
  enum class CompanionPart : std::uint8_t { kPre, kTarget, kPost };
  struct CompanionStep {
    std::uint32_t slot = 0;
    std::uint16_t stage = 0;
    std::uint32_t past = 0, rows = 0;
    CompanionPart part = CompanionPart::kTarget;
    std::vector<std::pair<std::size_t, DraftProgressLedger::Unit>> units;
  };
  // Set only by the shared helper. Public steps cannot acquire a compound
  // operation's state or overwrite an opted-in consumer's pending taps.
  std::vector<CompanionStep> companion_steps_;
  const CompanionStep* CompanionFor(std::uint32_t slot, std::uint16_t stage) const;
  ModelRunner(PagedNode& node, RunnerConfig config, std::unique_ptr<StepSource> source, int owner,
              std::uint32_t stream);
  static SlotLifecycleOptions LifecycleOptions(ModelRunner* self, SlotLifecycleOptions options);
  // Each slot's stages' verify snapshots and commits (max_verify_rows).
  Status SetupSnapshots();
  Status SetupCommits();
  Status SetupRowTables();
  Status PrepareCommit(std::uint32_t slot, std::uint16_t stage, std::uint32_t keep);
  LiveState::Queued QueueCommit(std::uint32_t slot, std::uint16_t stage,
                                providers::NativeStream stream);
  // Each slot's carried features on each stage whose steps return them
  // (`carry_bytes` by stage, from the setup shapes' programs).
  Status SetupCarry(std::span<const std::uint64_t> carry_bytes);
  // The device copies filling a step's kFeatures input from where each
  // segment's work names it.
  std::expected<std::vector<RunCopy>, std::string> FeatureCopies(
      const kernels::StepProgram& program, const StepRequest& request) const;
  // The device copies carrying each segment's next features (a drafter
  // head's) to its slot's carry on the step's stage.
  std::expected<std::vector<RunCopy>, std::string> CarryCopies(
      const kernels::StepProgram& program, const planner::StepShape& shape) const;
  std::uint64_t CarryAddress(std::uint32_t slot, std::size_t stage) const;
  void ForgetCarry(std::uint32_t slot);
  struct TapRows {
    std::uint32_t first = 0, count = 0, preceding = 0;
  };
  TapRows HeldTapRows(std::uint32_t slot, std::size_t stage) const;
  std::expected<std::vector<RunCopy>, std::string> TapCarries(
      const planner::StepShape& shape) const;

  planner::PlanKey KeyOf(const planner::StepShape& shape) const;
  kernels::Addresses AddressesFor(const planner::StepShape& shape, bool measuring) const;
  std::expected<void, std::string> CheckShape(const planner::StepShape& shape) const;
  // A plan found, or built, bound, checked and cached.
  std::expected<Plans::Entry*, std::string> Planned(const planner::StepShape& shape);
  // A plan built ahead (or now) bound, checked and cached; `transfer` hands
  // its lookahead grant to the cache's charge.
  std::expected<Plans::Entry*, std::string> Install(const planner::StepShape& shape,
                                                    std::unique_ptr<StepPlanned> planned,
                                                    double seconds,
                                                    const std::function<void()>& transfer);
  std::expected<Copies, std::string> OutputCopies(const kernels::StepProgram& program) const;
  std::vector<catalog::ExtentId> weight_extents() const;
  struct RoutedState;
  struct RoutedFlight;
  Status BeginRouted(kernels::StepProgram& program, StepPlanRuns& runs, const Copies& inputs,
                     const Copies& outputs, const Copies& carries, bool capture);
  void ForgetUnstartedRouted();
  void ForgetCompletedRouting();
  // Private test-support observation of already completed retained host copies.
  // This grants no routed access token or execution authority.
  std::expected<std::pair<std::uint64_t, std::span<const std::int32_t>>, std::string>
  CompletedRoute(const kernels::RoutedProgram&, std::size_t group) const;
  StepQueued QueueRoutedPrefix(providers::NativeStream native);
  Status ContinueRouted(RunPath& path);
  Status MeasureRouting(const kernels::StepProgram& program);
  Status FinishRoutingSetup();
  std::uint64_t RoutedPinnedBytes() const;
  Status CheckRouting(const kernels::StepProgram& program) const;
  Status CheckRoutingSource() const;
  Status CheckRoutedWeightBindings(std::span<const kernels::RoutedWeight> reads) const;
  std::vector<catalog::ExtentId> base_weight_extents() const;
  std::shared_ptr<RoutedState> routed_;
  std::shared_ptr<RoutedFlight> routed_flight_;
  // Whether a target's verify saves in, or commits, a slot's stage.
  bool Verifies(std::uint32_t slot, std::size_t stage) const;
  // The cells an owner of `past` positions and `rows` rows reads alone on
  // `stage` (a stage that reads frozen, through `past` only).
  std::uint32_t LoneRead(std::size_t stage, std::uint32_t past, std::uint32_t rows) const;
  // Its steps read only the cells before their rows (model::ReadsFrozen).
  bool Frozen(std::size_t stage) const { return frozen_.at(stage); }

  PagedNode& node_;
  RunnerConfig config_;
  std::unique_ptr<StepSource> source_;
  int owner_;
  std::uint32_t stream_;
  SlotLifecycle slots_;
  RunnerMemory memory_;
  StepRuns runs_;
  std::unique_ptr<kernels::StepLaunch> launch_;
  PlanAccount account_;
  Plans plans_;
  planner::PlanKey key_base_;

  std::uint64_t activations_needed_ = 0;
  std::uint64_t pool_needed_ = 0;
  std::uint64_t host_input_bytes_ = 0;
  std::uint64_t plan_floor_bytes_ = 0;
  void* inputs_staging_ = nullptr;
  std::uint64_t inputs_staging_bytes_ = 0;
  std::byte* outputs_staging_ = nullptr;
  std::uint64_t outputs_staging_bytes_ = 0;
  Mapped snapshots_;  // every slot's stages' verify snapshots
  struct CommitPlace {
    CommitBuffer buffer;
    std::uint64_t offset = 0;
  };
  std::vector<std::vector<CommitPlace>> commit_places_;  // by stage
  Mapped commit_saves_;
  void* commit_staging_ = nullptr;
  std::uint64_t commit_staging_stride_ = 0;
  struct PreparedCommit {
    Plans::Entry* plan = nullptr;  // protected by the job's PlanStep
    Copies copies;
    bool capture = false;
  };
  std::vector<PreparedCommit> prepared_commits_;  // by slot, then stage
  struct RowTableState {
    RunnerConfig::RowTable config;
    PleTable table;
    std::uint32_t shard = 0;
    std::uint64_t capacity = 0;
    Mapped slots;
    void* landing_allocation = nullptr;  // raw owner retained if a read stalls
    std::byte* landing = nullptr;
    std::uint32_t* sources = nullptr;
    std::uint32_t* count = nullptr;
    std::unique_ptr<providers::Storage> storage;
    bool stalled = false;
  };
  std::vector<std::unique_ptr<RowTableState>> row_tables_;
  // Every slot's carried features (by slot, then stage: carry_offset_ and
  // carry_bytes_ by stage, carry_stride_ a slot) and the rows each holds.
  Mapped carry_;
  std::vector<std::uint64_t> carry_offset_, carry_bytes_;
  std::uint64_t carry_stride_ = 0;
  std::vector<std::uint32_t> carry_rows_;              // by slot, then stage
  std::vector<TapRows> tap_rows_, previous_tap_rows_;  // per-row taps; by slot, then stage
  std::vector<bool> frozen_;                           // by stage
  // RunnerConfig::kernel_memory's mappings (each at a stable address).
  std::vector<std::unique_ptr<Mapped>> kernel_mapped_;
  bool prepared_ = false;      // RunnerConfig::prepare was called
  std::vector<bool> commits_;  // by slot, then stage: a commit is set

  GraphStats graph_stats_;
  Coverage coverage_;
  StepLookaheadStats lookahead_;
  bool setup_ = false, registered_ = false, bound_ = false, released_ = false;
};

}  // namespace llmp::engine

#endif  // LLMP_ENGINE_MODEL_RUNNER_H_
