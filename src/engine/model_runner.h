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
#include "engine/paged_node.h"
#include "engine/paged_weights.h"
#include "engine/plan_account.h"
#include "engine/plan_cache.h"
#include "engine/run_stats.h"
#include "engine/runner_memory.h"
#include "engine/slot_lifecycle.h"
#include "engine/step_runs.h"
#include "format/encoding.h"
#include "ir/graph.h"
#include "kernels/common/step_program.h"
#include "memory/reclaim.h"
#include "planner/step_shape.h"

namespace llmp::engine {

// What the runner asks for a step program: one virtual call per plan miss
// (the planner and a lowering behind it: ComponentStepSource once the
// components exist; FakeStepSource in the tests).
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
  // A program for `shape` at `addresses`. Host-only: it may run on the
  // driver while the device runs the current step (a lookahead), so it
  // never calls the node, the launch or the device; the runner binds it
  // (StepProgram::Bind) after. A measuring build (addresses.measuring())
  // only reports what it needs. The program serves every step of an equal
  // shape, whatever its segments' `past` (not part of the key): positions
  // reach it as input data, or, for a program that is not capturable,
  // through StepProgram::Position before each run; nothing it bakes may
  // follow `past`.
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
  // The step launch, made at Bind (after the workspace is mapped).
  std::function<std::expected<std::unique_ptr<kernels::StepLaunch>, std::string>()> launch;
  // A weight's address by role and part (the spec's binding to the
  // weights); unset: 0, for sources that bind none.
  std::function<std::uint64_t(const ir::WeightRef&, format::PartRole)> weight_address;
  // Speculation: the most rows a verify takes (0: none; no snapshot is
  // mapped and a verify is refused). Setup maps each slot's stages'
  // snapshots, sized by model::VerifySaveBound, for the stages a target's
  // verify writes (the target, a drafter's feature taps) or that commit.
  std::uint32_t max_verify_rows = 0;
  // A stage's commit of a verify's kept rows (LiveState::Commit: Qwen3.8's
  // recurrent state), per slot; unset or empty: none.
  std::function<LiveState::Commit(std::uint32_t slot, std::size_t stage)> commit;
};

// One segment's host data for a step, by the shape's segment order.
struct SegmentWork {
  std::span<const std::int32_t> tokens{};  // kTokens inputs: one a row
  std::span<const std::byte> bytes{};      // kSegmentBytes inputs
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
  std::uint64_t plan_floor_bytes() const { return plan_floor_bytes_; }

  SlotLifecycle& slots() { return slots_; }
  const SlotLifecycle& slots() const { return slots_; }
  // The most rows a verify takes (RunnerConfig::max_verify_rows).
  std::uint32_t max_verify_rows() const { return config_.max_verify_rows; }

  // ------------------------------------------------------------- steps

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
  ModelRunner(PagedNode& node, RunnerConfig config, std::unique_ptr<StepSource> source, int owner,
              std::uint32_t stream);
  static SlotLifecycleOptions LifecycleOptions(ModelRunner* self, SlotLifecycleOptions options);
  // Each slot's stages' verify snapshots and commits (max_verify_rows).
  Status SetupSnapshots();

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
  // Whether a target's verify saves in, or commits, a slot's stage.
  bool Verifies(std::uint32_t slot, std::size_t stage) const;
  // The cells an owner of `past` positions and `rows` rows reads alone on
  // `stage`.
  std::uint32_t LoneRead(std::size_t stage, std::uint32_t past, std::uint32_t rows) const;

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
  Mapped snapshots_;           // every slot's stages' verify snapshots
  std::vector<bool> commits_;  // by slot, then stage: a commit is set

  GraphStats graph_stats_;
  Coverage coverage_;
  StepLookaheadStats lookahead_;
  bool setup_ = false, registered_ = false, bound_ = false, released_ = false;
};

}  // namespace llmp::engine

#endif  // LLMP_ENGINE_MODEL_RUNNER_H_
