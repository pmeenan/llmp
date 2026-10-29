// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The prefill orchestrator (docs/engine.md, "The model runner";
// docs/engine-components.md, "Execution"): prompts run as chunks, one
// completed unit at a time, on the shared runner, with what every family's
// prefill did around a chunk written once.
//
// - Chunking for throughput: each owner's next chunk is the configured
//   rows (the model's setting), bounded by the capability's per-owner
//   rows, the prompt left and a boundary the caller names (a turn
//   checkpoint); no time cap (owner direction).
// - State-only cuts: a chunk that is not its prompt's last skips the head
//   (want_head false, no outputs) unless the options keep it; the last
//   returns the caller's head (its frontier row, or every row's).
// - Joined prompt waves where the capability allows (wave_composer.h),
//   with each owner's refusal its own.
// - The prefill lookahead: the next unit's shape and the one after it,
//   from the same owners continuing (an owner whose prompt ends drops out),
//   built while the device runs this chunk (up to `lookahead`, each under
//   its own allowance), the next one's recording captured beside this run
//   or, when it repeats this shape, this shape captured on its first run,
//   and the next chunk's fresh state prepared before the job
//   (LiveState::Prepare through SlotLifecycle::PrepareThrough).
// - Production continuation policy receives actual session budgets, sampling
//   intent and ready peers; semantic boundaries come from the owned source.
//   Candidate, key and build capacities are paid from actual resource counts.
// - The lifecycle's ledger: a write recorded from each owner's first row
//   before the job, its positions set to the chunk's end after it.
//
// Part of the engine core.

#ifndef LLMP_ENGINE_PREFILL_H_
#define LLMP_ENGINE_PREFILL_H_

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
#include "engine/prefill_lookahead.h"
#include "engine/run_stats.h"
#include "engine/wave_composer.h"

namespace llmp::engine {

struct PrefillOptions {
  std::uint32_t chunk_rows = 128;  // an owner's rows a unit
  std::uint32_t lookahead = 2;     // independent optional compiler grants
  // Explicit optional search budget, not a model/compressor interval.
  std::uint32_t specialization_lookahead = 1;
  bool capture_ahead = true;
  bool prepare_ahead = true;
  bool state_only = true;  // nonfinal chunks skip the head
};

// One prompt's work: the tokens not yet prefilled, after its slot's
// positions.
struct PromptWork {
  std::uint32_t slot = 0;
  // A prefill advances its slot's positions: the target's (a drafter's
  // step never commits, ModelRunner::Admit, so another stage is refused).
  std::uint16_t stage = 0;
  std::span<const std::int32_t> tokens{};
  std::uint32_t past = 0;
  bool want_head = true;     // the last chunk returns its head
  bool all_outputs = false;  // every row's outputs (scoring)
  // Rows into `tokens` no chunk crosses (a checkpoint's boundary); 0: none.
  std::uint32_t boundary = 0;
  // Full history through these prompt tokens, required by stages whose
  // steps read their owners' histories (side-row lookups).
  std::span<const std::int32_t> history{};
};

// Production input: an exact chunk and the session's actual next geometry.
// Histories/tokens are borrowed only for the synchronous StepPrepared call.
struct PrefillUnit {
  PendingWork current;
  SegmentWork work;
  PrefillHint next;
  // Full session prompt extent; absent for a caller that supplies only chunks.
  // A zero near hint may instead mean a checkpoint or unknown geometry.
  std::optional<std::uint32_t> prompt_end{};
  // Actual future target input rows remaining in this generation session.
  // Absence never supplies an invented peer lifetime.
  std::optional<std::uint32_t> remaining{};
  bool decode_greedy = false, verify_greedy = false, all_outputs = false;
  bool generation_expected = false, sampling = false;
};
struct PrefillPeer {
  PendingWork next;  // exact next decode coordinate, including its output mode
  bool ready = false, verify_greedy = false, sampling = false;
  std::optional<std::uint32_t> remaining{};
};
struct PrefillContinuation {
  bool speculate = false, row_invariant = false;
  std::uint32_t verify_rows = 1, min_verify_rows = 1;
  // Pure choice from the same configured adaptive policy used for decoding.
  // Empty means plain unless speculation is requested.
  std::function<bool(std::uint32_t, bool)> plain{};
  // A drafter stage whose prompt rows are rewritten after each prompt chunk
  // (DecodingOrchestrator::CatchUpPrompt: its rows after the chunk's first),
  // whose plans are then predicted beside the chunk's (0: none).
  std::uint16_t catch_up_stage = 0;
};

struct PrefillOutcome {
  std::optional<StepRefusal> refusal{};
  planner::Phase phase = planner::Phase::kPrefill;
  std::uint32_t rows = 0;  // prefilled by this unit
  bool last = false;       // the prompt's last chunk
  HostBuffer outputs{};
  // Its step's joined outputs, and where it sat in that step (its
  // segment, the rows and outputs of the segments before it): its share of
  // them (WaveOutcome's).
  HostBuffer joined{};
  std::uint32_t segment = 0;
  std::uint32_t first_row = 0;
  std::uint32_t first_output = 0;
  RunPath path = RunPath::kEager;
};

class PrefillReport {
 public:
  PrefillReport() = default;
  PrefillReport(const PrefillReport&) = delete;
  PrefillReport& operator=(const PrefillReport&) = delete;
  PrefillReport(PrefillReport&& other) noexcept { Swap(other); }
  PrefillReport& operator=(PrefillReport&& other) noexcept {
    PrefillReport replacement(std::move(other));
    Swap(replacement);
    return *this;
  }
  std::span<const PrefillOutcome> outcomes() const { return outcomes_; }
  std::optional<StepRefusal> refusal{};  // required dispatch metadata refused before work
  std::uint32_t steps = 0;
  std::uint32_t prediction_refused = 0;

 private:
  friend class PrefillOrchestrator;
  void Swap(PrefillReport& other) noexcept {
    std::swap(refusal, other.refusal);
    std::swap(steps, other.steps);
    std::swap(prediction_refused, other.prediction_refused);
    std::swap(payment_, other.payment_);
    outcomes_.swap(other.outcomes_);
  }
  // Private lifetime: capacity is destroyed before its payment, including
  // replacement after move assignment. Immutable views cannot reset receipts.
  HostPayment payment_;
  std::vector<PrefillOutcome> outcomes_;
};

class PrefillOrchestrator {
 public:
  PrefillOrchestrator(ModelRunner& runner, const WaveComposer& composer, PrefillOptions options)
      : runner_(runner), composer_(composer), options_(options) {}

  // The rows a prompt's chunk starting `at` rows into its tokens takes (0
  // when none is left).
  std::uint32_t ChunkRows(const PromptWork& prompt, std::uint32_t at) const;
  // The pending work of that chunk.
  PendingWork ChunkAt(const PromptWork& prompt, std::uint32_t at) const;

  // One completed unit: every prompt's next chunk.
  PrefillReport Step(std::span<const PromptWork> prompts);
  PrefillReport StepPrepared(std::span<const PrefillUnit> units,
                             std::span<const PrefillPeer> peers = {},
                             const PrefillContinuation& continuation = {});
  // Pure optional policy over explicit session facts. All candidate/workspace
  // capacity is admitted before composition. Refusal never changes real work.
  std::expected<StepAhead, std::string> Predict(std::span<const PrefillUnit> units,
                                                std::span<const PrefillPeer> peers,
                                                const PrefillContinuation& continuation,
                                                std::span<const std::size_t> owners) const;

 private:
  ModelRunner& runner_;
  const WaveComposer& composer_;
  PrefillOptions options_;
};

}  // namespace llmp::engine

#endif  // LLMP_ENGINE_PREFILL_H_
