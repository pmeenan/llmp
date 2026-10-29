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
// read every owner reads the step's widest cells; a row-invariant step (an
// exact verify, D-092) reads what each owner reads alone, its stage's own
// alignment and no common read, so no row's arithmetic follows a peer's
// reads. Each stage (the target, each drafter) has its own context and
// read alignment. Nothing here is a family constant: every cap is the
// capability's.
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

// One stage's reads: its state's positions, an owner's own read alignment
// and the read classes per doubling (execution::ReadCellsFor; 0: every
// alignment).
struct StageReads {
  std::uint32_t context = 0;
  std::uint32_t read_align = 256;
  std::uint32_t read_buckets = 0;
  // Its steps read only the cells before their rows (model::ReadsFrozen:
  // a drafter over its target's frozen state), so an owner reads through
  // its past positions, not past its rows.
  bool frozen = false;
};

class WaveComposer {
 public:
  // `stages` by stage (the target's first): each one's context, own read
  // alignment and read buckets (its state's: RunWave refuses a runner
  // whose SlotLifecycle::read_align, read_buckets or context differs). The
  // three-argument form composes the target alone.
  static std::expected<WaveComposer, std::string> Make(model::Batching caps,
                                                       std::vector<StageReads> stages);
  static std::expected<WaveComposer, std::string> Make(model::Batching caps, std::uint32_t context,
                                                       std::uint32_t read_align) {
    return Make(caps, std::vector<StageReads>{{.context = context, .read_align = read_align}});
  }
  // Every stage of a runner's slots.
  static std::expected<WaveComposer, std::string> For(model::Batching caps,
                                                      const SlotLifecycle& slots);

  const model::Batching& caps() const { return caps_; }
  std::size_t stages() const { return stages_.size(); }
  std::uint32_t context(std::size_t stage = 0) const { return stages_.at(stage).context; }
  std::uint32_t read_align(std::size_t stage = 0) const { return stages_.at(stage).read_align; }
  std::uint32_t read_buckets(std::size_t stage = 0) const { return stages_.at(stage).read_buckets; }
  bool frozen(std::size_t stage) const { return stages_.at(stage).frozen; }
  // A step's read alignment by its owner count (a row-invariant step's is
  // always its stage's own).
  std::uint32_t AlignFor(std::size_t owners) const { return AlignFor(0, owners); }
  std::uint32_t AlignFor(std::size_t stage, std::size_t owners, bool row_invariant = false) const {
    return owners > 1 && caps_.joined_read_align != 0 && !row_invariant ? caps_.joined_read_align
                                                                        : read_align(stage);
  }
  // The cells an owner of `past` positions and `rows` rows reads alone (on
  // a frozen stage, through `past`): their read class at the stage's own
  // alignment (execution::ReadCellsFor), as ModelRunner's.
  std::uint32_t LoneRead(std::uint32_t past, std::uint32_t rows) const {
    return LoneRead(0, past, rows);
  }
  std::uint32_t LoneRead(std::size_t stage, std::uint32_t past, std::uint32_t rows) const;

  // Runtime session eligibility uses this same pairwise rule; no second
  // equal-width/head/read-alignment policy lives in the serving adapter.
  bool CanJoinPrefill(std::uint32_t past, std::uint32_t rows, bool want_head,
                      std::uint32_t peer_past, std::uint32_t peer_rows, bool peer_want_head) const;
  Composition Compose(std::span<const PendingWork> pending) const;
  enum class PredictionError : std::uint8_t { kExtent, kWork };
  // Allocation-free composition into already paid shape/segment capacities.
  // Uses the same joining and read geometry as authoritative composition.
  std::expected<std::size_t, PredictionError> ComposePrediction(
      std::span<const PendingWork> pending, std::span<planner::StepShape> into,
      std::span<std::size_t> order) const;
  static StepAuthorization Authorize(const ModelRunner& runner, const PendingWork& work);

 private:
  WaveComposer(model::Batching caps, std::vector<StageReads> stages)
      : caps_(caps), stages_(std::move(stages)) {}
  // Fills in the shape's alignment, reads and groups from its segments.
  void Finish(planner::StepShape& shape) const;
  void Finish(std::uint16_t stage, bool row_invariant, std::span<planner::Segment> segments,
              std::uint32_t& read_align) const;

  model::Batching caps_;
  std::vector<StageReads> stages_;
};

// One pending work's outcome.
struct WaveOutcome {
  std::optional<StepRefusal> refusal{};  // none: it ran
  HostBuffer outputs{};                  // its own output slots
  HostBuffer joined{};                   // its step's joined outputs
  planner::Phase phase = planner::Phase::kDecode;
  std::uint32_t rows = 0;
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
  WaveReport() = default;
  WaveReport(WaveReport&& other) noexcept { Swap(other); }
  WaveReport& operator=(WaveReport&& other) noexcept {
    WaveReport replaced(std::move(other));
    Swap(replaced);
    return *this;
  }
  void Swap(WaveReport& other) noexcept {
    std::swap(payment_, other.payment_);
    outcomes.swap(other.outcomes);
    std::swap(steps, other.steps);
    std::swap(recomposed, other.recomposed);
  }
  // The copied result descriptors die before their retained payment. Payloads
  // carry their own independent host receipts through any longer retention.
 private:
  HostPayment payment_;

 public:
  bool HoldPayment(HostPayment paid) {
    if (payment_.bytes() != 0) return false;
    payment_ = std::move(paid);
    return true;
  }
  std::uint64_t paid_bytes() const { return payment_.bytes(); }
  std::vector<WaveOutcome> outcomes;  // by pending index
  std::uint32_t steps = 0;            // steps run
  std::uint32_t recomposed = 0;       // compositions after a refusal
};

// What a step predicts beyond itself (the prefill orchestrator's
// futures), given the step and its segments' pending indices.
using AheadFor =
    std::function<StepAhead(const ComposedStep& step, std::span<const std::size_t> pending)>;

// Head rows read back for greedy pending work (HeadRowReadback), by its
// pending index, with its rows' tokens (its SegmentWork's): each step's
// right after that step, before the wave's next.
struct WaveReadback {
  std::function<std::optional<std::uint32_t>(std::size_t pending,
                                             std::span<const std::int32_t> tokens,
                                             std::span<const std::byte> outputs)>
      want;
  std::function<void(std::size_t pending, std::uint32_t row,
                     std::expected<std::span<const std::byte>, std::string> logits)>
      take;
};

// Runs `pending` (with `work`, its host data, by index) on `runner`,
// composing, admitting and isolating as above; every step advances its
// owners' positions (`advance`; a verify's and a draft's do not).
WaveReport RunWave(ModelRunner& runner, const WaveComposer& composer,
                   std::span<const PendingWork> pending, std::span<const SegmentWork> work,
                   bool advance = true, const AheadFor& ahead = {},
                   std::span<const StepAuthorization> authorization = {},
                   const PromptBatch& batch = {}, const WaveReadback& readback = {});

}  // namespace llmp::engine

#endif  // LLMP_ENGINE_WAVE_COMPOSER_H_
