// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "engine/prefill.h"

#include <algorithm>

#include "engine/draft_companions.h"

namespace llmp::engine {

namespace pl = llmp::planner;

std::uint32_t PrefillOrchestrator::ChunkRows(const PromptWork& prompt, std::uint32_t at) const {
  const auto size = static_cast<std::uint32_t>(prompt.tokens.size());
  if (at >= size || options_.chunk_rows == 0 || prompt.stage >= composer_.stages()) return 0;
  std::uint32_t rows = std::min(options_.chunk_rows, size - at);
  if (composer_.caps().max_segment_rows != 0) {
    rows = std::min(rows, composer_.caps().max_segment_rows);
  }
  const std::uint32_t context = composer_.context(prompt.stage);
  const std::uint64_t start = std::uint64_t{prompt.past} + at;
  if (start >= context) return 0;
  rows = static_cast<std::uint32_t>(std::min<std::uint64_t>(rows, context - start));
  if (prompt.boundary > at) rows = std::min(rows, prompt.boundary - at);
  return rows;
}

PendingWork PrefillOrchestrator::ChunkAt(const PromptWork& prompt, std::uint32_t at) const {
  const std::uint32_t rows = ChunkRows(prompt, at);
  const bool last = at + rows == prompt.tokens.size();
  const bool head = last ? prompt.want_head : !options_.state_only;
  return {.slot = prompt.slot,
          .stage = prompt.stage,
          .phase = pl::Phase::kPrefill,
          .past = prompt.past + at,
          .rows = rows,
          .outputs = !head                ? 0U
                     : prompt.all_outputs ? rows
                                          : 1U,
          .want_head = head,
          .greedy = false,
          .row_invariant = false,
          .alone = false};
}

PrefillReport PrefillOrchestrator::Step(std::span<const PromptWork> prompts) {
  PrefillReport report;
  report.outcomes.resize(prompts.size());
  if (auto qualified = runner_.CheckOwners(prompts.size()); !qualified) {
    for (auto& out : report.outcomes) out.refusal = StepRefusal{.what = qualified.error()};
    return report;
  }
  std::vector<PendingWork> pending;
  std::vector<SegmentWork> work;
  std::vector<std::span<const std::int32_t>> histories;
  std::vector<std::size_t> of;  // the prompt of each pending work
  for (std::size_t i = 0; i < prompts.size(); ++i) {
    const PendingWork chunk = ChunkAt(prompts[i], 0);
    if (chunk.rows == 0) {
      report.outcomes[i].refusal = StepRefusal{.what = "the prompt has no rows left to prefill"};
      continue;
    }
    pending.push_back(chunk);
    work.push_back({.tokens = prompts[i].tokens.first(chunk.rows), .bytes = {}});
    histories.push_back(prompts[i].history.empty() && prompts[i].past == 0 ? prompts[i].tokens
                                                                           : prompts[i].history);
    of.push_back(i);
  }
  // The same owners' next and following chunks, composed: a future is a
  // step only when they compose into one.
  const auto future = [&](std::span<const std::size_t> owners,
                          std::size_t depth) -> std::optional<pl::StepShape> {
    std::vector<PendingWork> next;
    for (const std::size_t j : owners) {
      const PromptWork& prompt = prompts[of[j]];
      std::uint32_t at = 0;
      for (std::size_t d = 0; d <= depth; ++d) {
        const std::uint32_t rows = ChunkRows(prompt, at);
        if (rows == 0) break;
        at += rows;
      }
      const PendingWork chunk = ChunkAt(prompt, at);
      if (chunk.rows != 0) next.push_back(chunk);
    }
    if (next.empty()) return std::nullopt;
    Composition composed = composer_.Compose(next);
    if (composed.steps.size() != 1 || !composed.refused.empty()) return std::nullopt;
    return std::move(composed.steps.front().shape);
  };
  const AheadFor ahead = [&](const ComposedStep& /*step*/, std::span<const std::size_t> owners) {
    StepAhead a;
    if (options_.lookahead == 0 && !options_.capture_ahead && !options_.prepare_ahead) return a;
    a.shapes[0] = future(owners, 0);
    if (a.shapes[0]) a.shapes[1] = future(owners, 1);
    a.build = std::min<std::uint32_t>(options_.lookahead, 2);
    a.capture_ahead = options_.capture_ahead;
    a.prepare = options_.prepare_ahead;
    return a;
  };
  const WaveReport wave = RunPromptWave(runner_, composer_, pending, work, histories, ahead);
  report.steps = wave.steps;
  for (std::size_t j = 0; j < pending.size(); ++j) {
    PrefillOutcome& outcome = report.outcomes[of[j]];
    const WaveOutcome& ran = wave.outcomes[j];
    outcome.refusal = ran.refusal;
    if (ran.refusal) continue;
    outcome.rows = pending[j].rows;
    outcome.last =
        pending[j].past + pending[j].rows == prompts[of[j]].past + prompts[of[j]].tokens.size();
    outcome.outputs = ran.outputs;
    outcome.joined = ran.joined;
    outcome.segment = ran.segment;
    outcome.first_row = ran.first_row;
    outcome.first_output = ran.first_output;
    outcome.path = ran.path;
  }
  return report;
}

}  // namespace llmp::engine
