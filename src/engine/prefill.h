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
// - The lifecycle's ledger: a write recorded from each owner's first row
//   before the job, its positions set to the chunk's end after it.
//
// Part of the engine core.

#ifndef LLMP_ENGINE_PREFILL_H_
#define LLMP_ENGINE_PREFILL_H_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include "engine/model_runner.h"
#include "engine/run_stats.h"
#include "engine/wave_composer.h"

namespace llmp::engine {

struct PrefillOptions {
  std::uint32_t chunk_rows = 128;  // an owner's rows a unit
  std::uint32_t lookahead = 2;     // futures built ahead (0 to 2)
  bool capture_ahead = true;
  bool prepare_ahead = true;
  bool state_only = true;  // nonfinal chunks skip the head
};

// One prompt's work: the tokens not yet prefilled, after its slot's
// positions.
struct PromptWork {
  std::uint32_t slot = 0;
  std::uint16_t stage = 0;
  std::span<const std::int32_t> tokens{};
  std::uint32_t past = 0;
  bool want_head = true;     // the last chunk returns its head
  bool all_outputs = false;  // every row's outputs (scoring)
  // Rows into `tokens` no chunk crosses (a checkpoint's boundary); 0: none.
  std::uint32_t boundary = 0;
};

struct PrefillOutcome {
  std::optional<StepRefusal> refusal{};
  std::uint32_t rows = 0;  // prefilled by this unit
  bool last = false;       // the prompt's last chunk
  std::vector<std::byte> outputs{};
  RunPath path = RunPath::kEager;
};

struct PrefillReport {
  std::vector<PrefillOutcome> outcomes;  // by prompt
  std::uint32_t steps = 0;
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

 private:
  ModelRunner& runner_;
  const WaveComposer& composer_;
  PrefillOptions options_;
};

}  // namespace llmp::engine

#endif  // LLMP_ENGINE_PREFILL_H_
