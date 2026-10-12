// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// How a runner's step programs run (docs/engine.md; D-090): GraphRuns
// (graph_runs.h) over the vendor-free StepProgram
// (kernels/common/step_program.h), in the engine core. Each run's inputs
// are written by the host into the model's pinned staging, slot by slot,
// then queued in a device job as the input copies, anything the model
// queues between them and the program, the program's run and the outputs'
// copies; a shape that has run once launch by launch (or earlier when its
// runner asks: beside its first run, or ahead of it with CaptureAhead) is
// recorded whole through the program's Record and replayed as one launch
// from then on.
//
// A recording keeps every address it was made with, so whatever varies
// between runs of a shape is data it copies in from the staging at the
// offsets staged at its capture, checked at each replay (a changed staging
// refuses the replay rather than run the wrong inputs), never a launch
// parameter. The places it names are pinned for the model's life (D-090):
// the weights' and the state's in the scheduler, the workspace and the
// staging. A refused recording leaves that plan launch by launch.
// Recordings are destroyed with their plans, before the launch and before
// the memory they name (ModelRunner::Release), and only between jobs.

#ifndef LLMP_ENGINE_STEP_RUNS_H_
#define LLMP_ENGINE_STEP_RUNS_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <functional>
#include <memory>
#include <span>
#include <string>

#include "engine/run_stats.h"
#include "kernels/common/routed_program.h"
#include "kernels/common/step_program.h"
#include "providers/device_execution.h"

namespace llmp::engine {

// One step program's runs: how often it ran launch by launch, and its
// recording once captured with the input copies the recording holds
// (PlanRuns' counterpart).
struct StepPlanRuns {
  std::uint32_t eager_runs = 0;
  bool uncapturable = false;  // a capture was refused
  // A capture ahead of the plan's first run was refused: not asked again
  // ahead, while a capture beside a run (which may succeed) still may be.
  bool ahead_refused = false;
  std::unique_ptr<kernels::Recorded> graph;
  std::vector<std::unique_ptr<kernels::RoutedRecording>> routed_graphs;
  Copies copies;
  Copies routed_outputs;
  double seconds = 0;                // its capture and instantiation: what capturing it again costs
  std::uint64_t measured_bytes = 0;  // the device's free memory it took

  bool has_routed_graph() const {
    for (const auto& recording : routed_graphs)
      if (recording) return true;
    return false;
  }
  bool has_graph() const { return graph != nullptr || has_routed_graph(); }
  // With graphs on: no graph yet, none refused, and one run launch by
  // launch first (D-090: a shape is captured on its second run).
  bool CaptureDue(bool graphs) const {
    return graphs && !has_graph() && !uncapturable && eager_runs > 0;
  }
  // Destroys the recording, only between jobs (nothing in flight replays it).
  void DropGraph() {
    graph.reset();
    for (auto& segment : routed_graphs) segment.reset();
    copies.clear();
    routed_outputs.clear();
  }
};

// A planned step: its program, what the plan cache counts for it, and its
// measured planning time.
struct StepPlanned {
  std::unique_ptr<kernels::StepProgram> program;
  bool bound = false;       // Bind ran
  bool positioned = false;  // never captured: Position before each run
};
// What a planned step holds on the host, as the plan budget counts it (the
// lookahead's allowance check finds it by argument-dependent lookup).
std::uint64_t PlannedHostBytes(const StepPlanned& planned);

// What queueing a run did.
struct StepQueued {
  RunPath path = RunPath::kEager;
  bool before = false;  // work queued before any failure
  std::expected<void, kernels::StepFailure> result;
};

class StepRuns {
 public:
  explicit StepRuns(bool graphs) : graphs_(graphs) {}

  // The pinned staging (at setup) and the launch (at bind).
  void SetStaging(void* staging, std::uint64_t bytes) {
    staging_ = static_cast<std::byte*>(staging);
    staging_bytes_ = bytes;
  }
  void SetLaunch(kernels::StepLaunch* launch) { launch_ = launch; }
  std::byte* staging() const { return staging_; }
  std::uint64_t staging_bytes() const { return staging_bytes_; }
  kernels::StepLaunch* launch() const { return launch_; }

  // Graphs on or off for the next runs; captured graphs are kept.
  bool graphs() const { return graphs_; }
  void set_graphs(bool on) { graphs_ = on; }

  // Writes each input slot's bytes into the staging from `base`, each at a
  // 256-byte boundary (`fill` writes slot i's into its span, which is
  // exactly the slot's bytes): the input copies a run queues. On the host,
  // before the job queues anything that reads the staging. A kFeatures slot
  // is not staged (the runner copies it in on the device before the run).
  using Fill =
      std::function<std::expected<void, std::string>(std::size_t slot, std::span<std::byte> out)>;
  std::expected<Copies, std::string> Stage(std::span<const kernels::InputSlot> inputs,
                                           const Fill& fill, std::uint64_t base) const;
  // The input copies Stage would return for `inputs`, staging nothing:
  // what a graph captured ahead of its run copies in.
  std::expected<Copies, std::string> Layout(std::span<const kernels::InputSlot> inputs,
                                            std::uint64_t base) const;

  // Queues a run of `program` on `native`: the input copies, `between` (if
  // set; false is a failure of unknown effect), the program's run and the
  // outputs' copies. `runs`' recording replayed if it has one and graphs
  // are on; else, if `capture`, run launch by launch with the recording
  // made beside the run, or (with `between`) recorded first and replayed
  // once (a refusal leaves the plan launch by launch, counted in `stats`);
  // else launch by launch. The caller counts the path once its job ran.
  StepQueued Queue(StepPlanRuns& runs, const Copies& inputs,
                   const std::function<bool(providers::NativeStream)>& between,
                   const kernels::StepProgram& program, std::span<const RunCopy> outputs,
                   bool capture, GraphStats& stats, providers::NativeStream native) const;
  // One part of an owned dependency-discovery step. Prefix copies host inputs
  // and the complete route readback; continuation copies final outputs. Only
  // ModelRunner can supply its fresh access token. Both recordings stay in the
  // same plan account; an ordinary whole-step recording is never used here.
  StepQueued QueueRouted(StepPlanRuns& runs, const kernels::RoutedAccess& access,
                         kernels::RoutedProgram& program, const Copies& inputs,
                         std::span<const RunCopy> outputs, std::span<std::byte> route_staging,
                         bool capture, GraphStats& stats, providers::NativeStream native) const;
  // Records `runs`' graph of `program` without running it, beside work
  // already queued on `native`: the plan's next run replays it. A refusal
  // (false) sets ahead_refused only; only a fault (kUnknown) is an error.
  // Not counted as a run.
  std::expected<bool, kernels::StepFailure> CaptureAhead(StepPlanRuns& runs, const Copies& inputs,
                                                         const kernels::StepProgram& program,
                                                         std::span<const RunCopy> outputs,
                                                         GraphStats& stats,
                                                         providers::NativeStream native) const;

 private:
  std::expected<void, kernels::StepFailure> Record(
      const Copies& inputs, const std::function<bool(providers::NativeStream)>& between,
      const kernels::StepProgram& program, std::span<const RunCopy> outputs,
      providers::NativeStream native) const;
  std::expected<bool, kernels::StepFailure> CaptureInto(
      StepPlanRuns& runs, const Copies& inputs,
      const std::function<bool(providers::NativeStream)>& between,
      const kernels::StepProgram& program, std::span<const RunCopy> outputs, GraphStats& stats,
      providers::NativeStream native) const;

  bool graphs_ = true;
  std::byte* staging_ = nullptr;
  std::uint64_t staging_bytes_ = 0;
  kernels::StepLaunch* launch_ = nullptr;
};

}  // namespace llmp::engine

#endif  // LLMP_ENGINE_STEP_RUNS_H_
