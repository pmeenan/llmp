// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// How a runner's bound plans run (docs/engine.md; D-090): each run's inputs
// staged by the host into the model's pinned staging, then queued in a
// device job as the input copies, anything the model queues between them
// and the plan (Qwen3.8's n-gram row gather), the plan's steps and the
// outputs' copies; and for a shape that has run once launch by launch (or
// earlier when its runner asks: beside its first run, or ahead of it with
// CaptureAhead), that whole sequence captured as one graph through the K-C
// launch context and replayed as one launch from then on.
//
// A graph keeps every address it was captured with, so whatever varies
// between runs of a shape is data the graph copies in from the staging at
// the offsets the host staged it at the capture, checked at each replay
// (a changed staging refuses the replay rather than run the wrong inputs),
// never a launch parameter. The places a graph names are pinned for the
// model's life (D-090): the weights' and the state's in the scheduler, the
// workspace, the pool, the cuBLAS workspace and the staging mapped for the
// model's life. A capture the runtime refuses leaves that plan launch by
// launch, and a backend without recorded work runs every plan that way.
// Per step, not per kernel: nothing here is on a kernel's hot path.

#ifndef JITLLM_ENGINE_GRAPH_RUNS_H_
#define JITLLM_ENGINE_GRAPH_RUNS_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "ggml.h"
#include "kernels/ggml/executor.h"
#include "kernels/ggml/launch.h"
#include "providers/device_execution.h"

namespace jitllm::engine {

// How a run went (D-090).
enum class RunPath : std::uint8_t {
  kEager,     // launch by launch
  kCaptured,  // captured: beside a launch-by-launch run, or then replayed once
  kReplayed,  // one launch of a graph captured earlier
};

// A model's runs of one kind (its target's, or its drafter's), summed.
struct GraphStats {
  std::uint64_t eager = 0;
  std::uint64_t captured = 0;
  std::uint64_t replayed = 0;
  std::uint64_t refused = 0;       // captures refused: those plans stay launch by launch
  std::uint64_t dropped = 0;       // graphs destroyed to stay within the model's cap
  double capture_seconds = 0;      // capturing
  double instantiate_seconds = 0;  // instantiating and uploading
  std::uint64_t nodes = 0;         // in every graph captured
  std::int64_t memory_bytes = 0;   // the drop in the device's free memory across captures
  std::string first_refusal;
};

// Counts a run that went `path`.
void Count(GraphStats& stats, RunPath path);

// Copies a run queues, three addresses or sizes each: an input's device
// address, its bytes and its offset in the staging (Stage's); an output's
// host address, device address and bytes.
using RunCopy = std::array<std::uint64_t, 3>;
using Copies = std::vector<RunCopy>;

// One plan's runs: how often it ran launch by launch, and its graph once
// captured with the input copies the graph holds.
struct PlanRuns {
  std::uint32_t eager_runs = 0;
  bool uncapturable = false;  // a capture was refused
  // A capture ahead of the plan's first run was refused: not asked again
  // ahead, while a capture beside a run (which may succeed) still may be.
  bool ahead_refused = false;
  std::optional<kernels::ggml::CapturedGraph> graph;
  Copies copies;
  double seconds = 0;  // its graph's capture and instantiation: what capturing it again costs
  // What the capture took of the device's free memory (on a GB10 the
  // node's one memory, so host and driver memory alike; other processes'
  // use in the meantime counts too): a check on kGraphNodeHostBytes.
  std::uint64_t measured_bytes = 0;

  // With graphs on: no graph yet, none refused, and one run launch by
  // launch first (D-090: a shape is captured on its second run).
  bool CaptureDue(bool graphs) const {
    return graphs && !graph.has_value() && !uncapturable && eager_runs > 0;
  }
  // Destroys the graph, only between jobs (nothing in flight replays it).
  void DropGraph() {
    graph.reset();
    copies.clear();
  }
};

// What queueing a run did.
struct Queued {
  RunPath path = RunPath::kEager;
  bool before = false;  // work queued before any failure
  std::expected<void, kernels::ggml::KernelFailure> result;
};

// A model's staging and launch context, and whether its graphs are on.
class GraphRuns {
 public:
  explicit GraphRuns(bool graphs) : graphs_(graphs) {}

  // The pinned staging (at setup) and the launch context (at bind).
  void SetStaging(void* staging, std::uint64_t bytes) {
    staging_ = static_cast<std::byte*>(staging);
    staging_bytes_ = bytes;
  }
  void SetLaunch(kernels::ggml::LaunchContext* launch) { launch_ = launch; }
  std::byte* staging() const { return staging_; }
  std::uint64_t staging_bytes() const { return staging_bytes_; }

  // Graphs on or off for the next runs; captured graphs are kept.
  bool graphs() const { return graphs_; }
  void set_graphs(bool on) { graphs_ = on; }

  // Copies `sources`' bytes into the staging from `base`, each input at a
  // 256-byte boundary: the input copies a run queues. On the host, before
  // the job queues anything that reads the staging.
  std::expected<Copies, std::string> Stage(
      std::span<const std::pair<ggml_tensor*, const void*>> sources, std::uint64_t base) const;
  // The input copies Stage would return for `inputs` in this order, without
  // staging anything: what a graph captured ahead of its run copies in.
  std::expected<Copies, std::string> Layout(std::span<ggml_tensor* const> inputs,
                                            std::uint64_t base) const;

  // Queues a run of `bound` on `native`, the launch context's stream: the
  // input copies, `between` (if set; false is a failure of unknown
  // effect), the plan's steps and the outputs' copies. `runs`' graph
  // replayed if it has one and graphs are on; else, if `capture`, run
  // launch by launch with its graph captured beside the run (then
  // instantiated and uploaded while the device works), or with `between`
  // captured first and replayed once (a refusal leaves the plan launch by
  // launch, counted in `stats`); else launch by launch. The caller counts
  // the path once its job has run.
  Queued Queue(PlanRuns& runs, const Copies& inputs,
               const std::function<bool(void* stream)>& between, kernels::ggml::BoundGraph& bound,
               std::span<const RunCopy> outputs, bool capture, GraphStats& stats,
               providers::NativeStream native) const;
  // Captures `runs`' graph of `bound` without running it, beside work
  // already queued on `native` (a graph records, it queues nothing): the
  // plan's next run replays it. A refusal (false) sets ahead_refused only;
  // only a fault (kUnknown) is an error. Not counted as a run.
  std::expected<bool, kernels::ggml::KernelFailure> CaptureAhead(
      PlanRuns& runs, const Copies& inputs, kernels::ggml::BoundGraph& bound,
      std::span<const RunCopy> outputs, GraphStats& stats, providers::NativeStream native) const;

 private:
  // Queues (or, inside a capture, records) the input copies, `between`, the
  // plan's steps and the outputs' copies.
  std::expected<void, kernels::ggml::KernelFailure> Record(
      kernels::ggml::LaunchContext& launch, const Copies& inputs,
      const std::function<bool(void* stream)>& between, kernels::ggml::BoundGraph& bound,
      std::span<const RunCopy> outputs, providers::NativeStream native) const;
  // Captures what Record queues as `runs`' graph, with its costs in
  // `stats`; refused captures mark the plan uncapturable.
  std::expected<bool, kernels::ggml::KernelFailure> CaptureInto(
      PlanRuns& runs, const Copies& inputs, const std::function<bool(void* stream)>& between,
      kernels::ggml::BoundGraph& bound, std::span<const RunCopy> outputs, GraphStats& stats,
      providers::NativeStream native) const;

  bool graphs_ = true;
  std::byte* staging_ = nullptr;
  std::uint64_t staging_bytes_ = 0;
  kernels::ggml::LaunchContext* launch_ = nullptr;
};

}  // namespace jitllm::engine

#endif  // JITLLM_ENGINE_GRAPH_RUNS_H_
