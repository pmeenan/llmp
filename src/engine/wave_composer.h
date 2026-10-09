// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The wave composer (docs/engine.md, "The model runner";
// docs/engine-components.md, "Execution"): the request slots' pending work
// made into step shapes (planner/step_shape.h) by the model's batching
// capability (model/batching.h), and RunWave, which runs them on a model
// runner with every owner's outcome its own.
//
// Composition. Work of one kind (stage, phase, row invariance, device
// choice and, unless the capability allows mixing, head intent) joins in
// slot order while the step's owners, rows and (for prompt rows) equal
// widths allow; within a step, consecutive owners form joined groups while
// their rows and number fit the group caps. A joined step reads at the
// joined alignment (its key repeats as its owners grow), and with a common
// read every owner reads the step's widest cells. Nothing here is a family
// constant: every cap is the capability's.
//
// Isolation (RunWave). One owner's refusal never fails its peers: an
// owner that is not active, not at its positions, or whose state cannot
// grow is refused alone and the rest are composed again without it (one
// out of capacity only for the joined step's wider reads, at a joined
// alignment or through a common read, first steps alone at its own, which
// the runner backed first: StepRefusal::widening); a joined step refused
// before anything ran (its plan, its staging) runs its owners alone. Only a
// job that failed after it may have written its owners' state fails those
// owners together (their states quarantined, the shared work unproven).
//
// Part of the engine core.

#ifndef LLMP_ENGINE_WAVE_COMPOSER_H_
#define LLMP_ENGINE_WAVE_COMPOSER_H_

#include <cstddef>
#include <cstdint>
#include <expected>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "engine/model_runner.h"
#include "engine/run_stats.h"
#include "model/batching.h"
#include "planner/step_shape.h"

namespace llmp::engine {

// One slot's work for the next step.
struct PendingWork {
  std::uint32_t slot = 0;
  std::uint16_t stage = 0;
  planner::Phase phase = planner::Phase::kDecode;
  std::uint32_t past = 0;
  std::uint32_t rows = 1;
  std::uint32_t outputs = 1;  // 0 with want_head false
  bool want_head = true;
  bool greedy = false;
  bool row_invariant = false;
  bool alone = false;  // steps by itself (RunWave's fallback)
};

struct ComposedStep {
  planner::StepShape shape{};
  std::vector<std::size_t> work{};  // the pending work's index, by segment
};

struct Composition {
  std::vector<ComposedStep> steps;
  std::vector<std::pair<std::size_t, std::string>> refused;  // malformed work, by index
};

class WaveComposer {
 public:
  // `context` the state's positions, `read_align` an owner's own read
  // alignment (its state's: RunWave refuses a runner whose
  // SlotLifecycle::read_align differs).
  static std::expected<WaveComposer, std::string> Make(model::Batching caps, std::uint32_t context,
                                                       std::uint32_t read_align);

  const model::Batching& caps() const { return caps_; }
  std::uint32_t context() const { return context_; }
  std::uint32_t read_align() const { return read_align_; }
  // A step's read alignment by its owner count.
  std::uint32_t AlignFor(std::size_t owners) const {
    return owners > 1 && caps_.joined_read_align != 0 ? caps_.joined_read_align : read_align_;
  }
  // The cells an owner of `past` positions and `rows` rows reads alone.
  std::uint32_t LoneRead(std::uint32_t past, std::uint32_t rows) const;

  Composition Compose(std::span<const PendingWork> pending) const;

 private:
  WaveComposer(model::Batching caps, std::uint32_t context, std::uint32_t read_align)
      : caps_(caps), context_(context), read_align_(read_align) {}
  // Fills in the shape's alignment, reads and groups from its segments.
  void Finish(planner::StepShape& shape) const;

  model::Batching caps_;
  std::uint32_t context_ = 0;
  std::uint32_t read_align_ = 256;
};

// One pending work's outcome.
struct WaveOutcome {
  std::optional<StepRefusal> refusal{};  // none: it ran
  std::vector<std::byte> outputs{};      // its own output slots
  std::vector<std::byte> joined{};       // its step's joined outputs
  RunPath path = RunPath::kEager;
  std::uint32_t step = 0;  // the step (in run order) it ran in
  // Where it sat in that step, to find its share of `joined` without
  // composing again: its segment, and the rows and outputs of the
  // segments before it.
  std::uint32_t segment = 0;
  std::uint32_t first_row = 0;
  std::uint32_t first_output = 0;
};

struct WaveReport {
  std::vector<WaveOutcome> outcomes;  // by pending index
  std::uint32_t steps = 0;            // steps run
  std::uint32_t recomposed = 0;       // compositions after a refusal
};

// What a step predicts beyond itself (the prefill orchestrator's
// futures), given the step and its segments' pending indices.
using AheadFor =
    std::function<StepAhead(const ComposedStep& step, std::span<const std::size_t> pending)>;

// Runs `pending` (with `work`, its host data, by index) on `runner`,
// composing, admitting and isolating as above; every step advances its
// owners' positions (`advance`).
WaveReport RunWave(ModelRunner& runner, const WaveComposer& composer,
                   std::span<const PendingWork> pending, std::span<const SegmentWork> work,
                   bool advance = true, const AheadFor& ahead = {});

}  // namespace llmp::engine

#endif  // LLMP_ENGINE_WAVE_COMPOSER_H_
