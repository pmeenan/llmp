// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "engine/wave_composer.h"

#include <algorithm>
#include <format>
#include <tuple>

#include "engine/request_cohort.h"
#include "engine/support.h"
#include "model/state_contract.h"

namespace llmp::engine {

namespace {

namespace pl = llmp::planner;
using support::Error;
using support::Round;

bool AlignValid(std::uint32_t align) { return align != 0 && align % 256 == 0; }

pl::Phase DispatchPhase(const model::Batching& caps, const PendingWork& p) {
  return caps.mixed_causal && caps.join_prefill && caps.max_segments > 1 && p.stage == 0 &&
                 !p.row_invariant &&
                 (p.phase == pl::Phase::kPrefill || (p.phase == pl::Phase::kDecode && p.rows == 1))
             ? pl::Phase::kPrefill
             : p.phase;
}
auto Kind(const model::Batching& caps, const PendingWork& p) {
  return std::make_tuple(p.stage, static_cast<int>(DispatchPhase(caps, p)), p.row_invariant,
                         caps.mixed_greedy ? false : p.greedy, caps.mixed_heads || p.want_head);
}
bool Join(const model::Batching& caps, const PendingWork& first, const PendingWork& next,
          std::size_t owners, std::uint32_t rows, std::uint32_t first_read,
          std::uint32_t next_read) {
  return Kind(caps, first) == Kind(caps, next) && !first.alone && !next.alone &&
         caps.max_segments > 1 &&
         (DispatchPhase(caps, next) != pl::Phase::kPrefill || caps.join_prefill) &&
         owners < caps.max_segments && std::uint64_t{rows} + next.rows <= UINT32_MAX &&
         (caps.max_step_rows == 0 || std::uint64_t{rows} + next.rows <= caps.max_step_rows) &&
         (DispatchPhase(caps, next) != pl::Phase::kPrefill || !caps.equal_prefill ||
          (caps.mixed_causal && first.phase != next.phase) ||
          (next.rows == first.rows && next_read == first_read));
}

}  // namespace

std::expected<WaveComposer, std::string> WaveComposer::Make(model::Batching caps,
                                                            std::vector<StageReads> stages) {
  if (stages.empty()) return Error("a wave composer needs a stage");
  for (const StageReads& stage : stages) {
    if (stage.context == 0) return Error("a wave composer needs a context");
    if (!AlignValid(stage.read_align)) {
      return Error("a read alignment is a positive multiple of 256");
    }
  }
  if (caps.joined_read_align != 0 && !AlignValid(caps.joined_read_align)) {
    return Error("a joined read alignment is a positive multiple of 256");
  }
  if (caps.max_segments == 0 || caps.max_segments > kMaxRequestSlots) {
    return Error(std::format("a step joins 1 to {} owners", kMaxRequestSlots));
  }
  return WaveComposer(caps, std::move(stages));
}

std::expected<WaveComposer, std::string> WaveComposer::For(model::Batching caps,
                                                           const SlotLifecycle& slots) {
  std::vector<StageReads> stages;
  for (std::size_t s = 0; s < slots.stages(); ++s) {
    stages.push_back({.context = slots.context(s),
                      .read_align = slots.read_align(s),
                      .frozen = s != 0 && model::ReadsFrozen(slots.layout(s))});
  }
  return Make(caps, std::move(stages));
}

std::uint32_t WaveComposer::LoneRead(std::size_t stage, std::uint32_t past,
                                     std::uint32_t rows) const {
  const std::uint64_t end = frozen(stage) ? past : std::uint64_t{past} + rows;
  return static_cast<std::uint32_t>(
      std::min<std::uint64_t>(Round(end, read_align(stage)), context(stage)));
}

void WaveComposer::Finish(pl::StepShape& shape) const {
  if (shape.phase == pl::Phase::kPrefill && !shape.segments.empty() &&
      std::ranges::all_of(shape.segments, [](const auto& segment) {
        return segment.origin_phase == pl::Phase::kDecode;
      })) {
    shape.phase = pl::Phase::kDecode;
    for (auto& segment : shape.segments) segment.origin_phase.reset();
  }
  Finish(shape.stage, shape.row_invariant, shape.segments, shape.read_align);
}

void WaveComposer::Finish(std::uint16_t stage, bool row_invariant, std::span<pl::Segment> segments,
                          std::uint32_t& read_align) const {
  // A row-invariant step reads each owner's own cells (D-092): its stage's
  // alignment, no common read.
  read_align = AlignFor(stage, segments.size(), row_invariant);
  std::uint32_t widest = 0;
  for (pl::Segment& s : segments) {
    const std::uint64_t end = frozen(stage) ? s.past : std::uint64_t{s.past} + s.rows;
    s.read_cells =
        static_cast<std::uint32_t>(std::min<std::uint64_t>(Round(end, read_align), context(stage)));
    widest = std::max(widest, s.read_cells);
  }
  if (caps_.common_read && segments.size() > 1 && !row_invariant) {
    for (pl::Segment& s : segments) s.read_cells = widest;
  }
  const std::uint32_t group_segments = row_invariant && caps_.invariant_group_segments != 0
                                           ? caps_.invariant_group_segments
                                           : caps_.group_segments;
  std::uint8_t group = 0;
  std::uint32_t rows = 0, owners = 0;
  for (pl::Segment& s : segments) {
    const bool rows_full = caps_.group_rows != 0 && rows + s.rows > caps_.group_rows;
    const bool owners_full = group_segments != 0 && owners + 1 > group_segments;
    if (owners != 0 && (rows_full || owners_full)) {
      ++group;
      rows = 0;
      owners = 0;
    }
    s.group = group;
    rows += s.rows;
    ++owners;
  }
}

bool WaveComposer::CanJoinPrefill(std::uint32_t past, std::uint32_t rows, bool want_head,
                                  std::uint32_t peer_past, std::uint32_t peer_rows,
                                  bool peer_want_head) const {
  const auto valid = [&](std::uint32_t at, std::uint32_t count) {
    return count != 0 && at <= context() && count <= context() - at &&
           (caps_.max_segment_rows == 0 || count <= caps_.max_segment_rows);
  };
  if (!valid(past, rows) || !valid(peer_past, peer_rows)) return false;
  const PendingWork first{.slot = 0,
                          .stage = 0,
                          .phase = pl::Phase::kPrefill,
                          .past = past,
                          .rows = rows,
                          .outputs = want_head ? 1U : 0U,
                          .want_head = want_head};
  const PendingWork peer{.slot = 1,
                         .stage = 0,
                         .phase = pl::Phase::kPrefill,
                         .past = peer_past,
                         .rows = peer_rows,
                         .outputs = peer_want_head ? 1U : 0U,
                         .want_head = peer_want_head};
  return Join(caps_, first, peer, 1, rows, LoneRead(past, rows), LoneRead(peer_past, peer_rows));
}

Composition WaveComposer::Compose(std::span<const PendingWork> pending) const {
  Composition out;
  std::vector<std::size_t> order;
  order.reserve(pending.size());
  out.steps.reserve(pending.size());
  out.refused.reserve(pending.size());
  for (std::size_t i = 0; i < pending.size(); ++i) {
    const PendingWork& p = pending[i];
    std::string why;
    const std::uint32_t context = p.stage < stages_.size() ? stages_[p.stage].context : 0;
    if (p.stage >= stages_.size()) {
      why = "it names no stage";
    } else if (p.rows == 0 || p.past > context || p.rows > context - p.past) {
      why = "its rows are outside the context";
    } else if (caps_.max_segment_rows != 0 && p.rows > caps_.max_segment_rows) {
      why = std::format("its {} rows exceed an owner's {}", p.rows, caps_.max_segment_rows);
    } else if (p.outputs > p.rows || (!p.want_head && p.outputs != 0)) {
      why = "its outputs exceed its rows";
    } else if (p.slot >= kMaxRequestSlots) {
      why = "it names no slot";
    } else if (std::ranges::any_of(order, [&](std::size_t j) {
                 return pending[j].slot == p.slot && pending[j].stage == p.stage;
               })) {
      why = "its slot already has work in this wave";
    }
    if (!why.empty()) {
      out.refused.emplace_back(i, std::format("slot {}: {}", p.slot, why));
      continue;
    }
    order.push_back(i);
  }
  // Kinds together, slots ascending within a kind, and work that steps
  // alone after the rest of its kind (between them, it would split their
  // joined step in two).
  const auto kind = [&](const PendingWork& p) { return Kind(caps_, p); };
  std::ranges::stable_sort(order, [&](std::size_t a, std::size_t b) {
    return std::tuple_cat(kind(pending[a]), std::make_tuple(pending[a].alone, pending[a].slot)) <
           std::tuple_cat(kind(pending[b]), std::make_tuple(pending[b].alone, pending[b].slot));
  });
  const auto lone_read = [&](const PendingWork& p) { return LoneRead(p.stage, p.past, p.rows); };
  ComposedStep step;
  std::uint32_t step_rows = 0;
  const auto close = [&] {
    if (step.work.empty()) return;
    Finish(step.shape);
    out.steps.push_back(std::move(step));
    step = {};
    step_rows = 0;
  };
  for (const std::size_t i : order) {
    const PendingWork& p = pending[i];
    bool join = !step.work.empty();
    if (join) {
      const PendingWork& first = pending[step.work.front()];
      join = Join(caps_, first, p, step.work.size(), step_rows, lone_read(first), lone_read(p));
    }
    if (!join) close();
    if (step.work.empty()) {
      step.work.reserve(pending.size());
      step.shape.segments.reserve(pending.size());
      step.shape.stage = p.stage;
      step.shape.phase = DispatchPhase(caps_, p);
      step.shape.row_invariant = p.row_invariant;
    }
    step.shape.segments.push_back({.slot = static_cast<std::uint8_t>(p.slot),
                                   .rows = p.rows,
                                   .past = p.past,
                                   .read_cells = 0,
                                   .outputs = p.outputs,
                                   .group = 0,
                                   .want_head = p.want_head,
                                   .greedy = p.greedy,
                                   .origin_phase = DispatchPhase(caps_, p) != p.phase
                                                       ? std::optional(p.phase)
                                                       : std::nullopt});
    step.work.push_back(i);
    step_rows += p.rows;
  }
  close();
  return out;
}

std::expected<std::size_t, WaveComposer::PredictionError> WaveComposer::ComposePrediction(
    std::span<const PendingWork> pending, std::span<pl::StepShape> into,
    std::span<std::size_t> order) const {
  if (order.size() < pending.size() || into.size() < pending.size())
    return std::unexpected(PredictionError::kExtent);
  for (std::size_t i = 0; i < pending.size(); ++i) {
    const auto& p = pending[i];
    if (p.stage >= stages_.size() || p.rows == 0 || p.past > context(p.stage) ||
        p.rows > context(p.stage) - p.past ||
        (caps_.max_segment_rows && p.rows > caps_.max_segment_rows) || p.outputs > p.rows ||
        (!p.want_head && p.outputs != 0) || p.slot >= kMaxRequestSlots)
      return std::unexpected(PredictionError::kWork);
    for (std::size_t j = 0; j < i; ++j)
      if (pending[j].slot == p.slot && pending[j].stage == p.stage)
        return std::unexpected(PredictionError::kWork);
    order[i] = i;
    if (into[i].segments.capacity() < pending.size())
      return std::unexpected(PredictionError::kExtent);
    into[i].segments.clear();
  }
  order = order.first(pending.size());
  std::ranges::sort(order, [&](std::size_t a, std::size_t b) {
    return std::tuple_cat(Kind(caps_, pending[a]),
                          std::make_tuple(pending[a].alone, pending[a].slot, a)) <
           std::tuple_cat(Kind(caps_, pending[b]),
                          std::make_tuple(pending[b].alone, pending[b].slot, b));
  });
  std::size_t count = 0, first = 0;
  std::uint32_t rows = 0;
  for (const auto i : order) {
    const auto& p = pending[i];
    if (count && !into[count - 1].segments.empty() &&
        !Join(caps_, pending[first], p, into[count - 1].segments.size(), rows,
              LoneRead(pending[first].stage, pending[first].past, pending[first].rows),
              LoneRead(p.stage, p.past, p.rows))) {
      Finish(into[count - 1]);
      rows = 0;
      ++count;
    }
    if (!count) ++count;
    auto& shape = into[count - 1];
    if (shape.segments.empty()) {
      first = i;
      shape.stage = p.stage;
      shape.phase = DispatchPhase(caps_, p);
      shape.row_invariant = p.row_invariant;
    }
    shape.segments.push_back({.slot = static_cast<std::uint8_t>(p.slot),
                              .rows = p.rows,
                              .past = p.past,
                              .read_cells = 0,
                              .outputs = p.outputs,
                              .group = 0,
                              .want_head = p.want_head,
                              .greedy = p.greedy,
                              .origin_phase = DispatchPhase(caps_, p) != p.phase
                                                  ? std::optional(p.phase)
                                                  : std::nullopt});
    rows += p.rows;
  }
  if (count) Finish(into[count - 1]);
  return count;
}

StepAuthorization WaveComposer::Authorize(const ModelRunner& runner, const PendingWork& work) {
  StepAuthorization permit;
  permit.runner_ = &runner;
  permit.slot_ = work.slot;
  permit.stage_ = work.stage;
  permit.phase_ = work.phase;
  permit.past_ = work.past;
  permit.rows_ = work.rows;
  permit.outputs_ = work.outputs;
  permit.want_head_ = work.want_head;
  permit.greedy_ = work.greedy;
  permit.part_ = StepAuthorization::Part::kOrigin;
  return permit;
}

WaveReport RunWave(ModelRunner& runner, const WaveComposer& composer,
                   std::span<const PendingWork> pending, std::span<const SegmentWork> work,
                   bool advance, const AheadFor& ahead,
                   std::span<const StepAuthorization> authorization, const PromptBatch& batch,
                   const WaveReadback& readback) {
  WaveReport report;
  report.outcomes.resize(pending.size());
  for (std::size_t i = 0; i < pending.size(); ++i) {
    report.outcomes[i].phase = pending[i].phase;
    report.outcomes[i].rows = pending[i].rows;
  }
  std::vector<bool> settled(pending.size());
  const auto settle = [&](std::size_t i, StepRefusal refusal) {
    report.outcomes[i].refusal = std::move(refusal);
    settled[i] = true;
  };
  // An owner's lone read is the composer's alignment; what the runner backs
  // first for each owner of a joined step is its state's. Equal, an owner
  // whose widening is refused steps alone with no growth.
  const SlotLifecycle& slots = runner.slots();
  bool differs = composer.stages() > slots.stages();
  for (std::size_t s = 0; s < composer.stages() && !differs; ++s) {
    differs = composer.read_align(s) != slots.read_align(s) ||
              composer.context(s) != slots.context(s) ||
              composer.frozen(s) != (s != 0 && model::ReadsFrozen(slots.layout(s)));
  }
  const char* mismatch = !authorization.empty() && authorization.size() != pending.size()
                             ? "the wave authorization does not match its pending work"
                         : work.size() != pending.size()
                             ? "the wave's work does not match its pending work"
                         : differs ? "the composer's read alignment or context is not the state's"
                                   : nullptr;
  if (mismatch != nullptr) {
    for (std::size_t i = 0; i < pending.size(); ++i) settle(i, {.what = mismatch});
    return report;
  }
  std::vector<PendingWork> live(pending.begin(), pending.end());
  bool first = true;
  for (;;) {
    std::vector<std::size_t> open;
    std::vector<PendingWork> subset;
    open.reserve(live.size());
    subset.reserve(live.size());
    for (std::size_t i = 0; i < live.size(); ++i) {
      if (settled[i]) continue;
      open.push_back(i);
      subset.push_back(live[i]);
    }
    if (open.empty()) break;
    if (!first) ++report.recomposed;
    first = false;
    const Composition composition = composer.Compose(subset);
    for (const auto& [index, why] : composition.refused) settle(open[index], {.what = why});
    bool restart = false;
    for (const ComposedStep& composed : composition.steps) {
      std::vector<SegmentWork> segments;
      std::vector<StepAuthorization> permits;
      std::vector<std::size_t> indices;
      segments.reserve(composed.work.size());
      permits.reserve(composed.work.size());
      indices.reserve(composed.work.size());
      const bool mixed_origin = std::ranges::any_of(
          composed.shape.segments, [](const pl::Segment& s) { return s.origin_phase.has_value(); });
      for (const std::size_t j : composed.work) {
        segments.push_back(work[open[j]]);
        if (!authorization.empty()) {
          permits.push_back(authorization[open[j]]);
        } else if (mixed_origin) {
          permits.push_back(WaveComposer::Authorize(runner, live[open[j]]));
        }
        indices.push_back(open[j]);
      }
      // The step's segments' readbacks, as their pending work's.
      const HeadRowReadback readbacks{
          .want =
              [&](std::size_t s, std::span<const std::byte> outputs) {
                return readback.want(indices[s], segments[s].tokens, outputs);
              },
          .take =
              [&](std::size_t s, std::uint32_t row,
                  std::expected<std::span<const std::byte>, std::string> logits) {
                readback.take(indices[s], row, std::move(logits));
              }};
      const StepRequest request{.shape = composed.shape,
                                .work = segments,
                                .step_bytes = {},
                                .advance = advance,
                                .ahead = batch ? batch.Ahead(composed.shape)
                                               : (ahead ? ahead(composed, indices) : StepAhead{}),
                                .authorization = permits,
                                .batch = batch,
                                .readback = readback.want && readback.take ? &readbacks : nullptr};
      auto ran = runner.RunStep(request);
      if (ran) {
        std::uint32_t rows = 0, outputs = 0;
        for (std::size_t s = 0; s < composed.work.size(); ++s) {
          WaveOutcome& outcome = report.outcomes[open[composed.work[s]]];
          outcome.phase = composed.shape.segments[s].origin_phase.value_or(composed.shape.phase);
          outcome.rows = composed.shape.segments[s].rows;
          outcome.outputs = ran->outputs[s];
          outcome.joined = ran->joined;
          outcome.path = ran->path;
          outcome.step = report.steps;
          outcome.segment = static_cast<std::uint32_t>(s);
          outcome.first_row = rows;
          outcome.first_output = outputs;
          rows += composed.shape.segments[s].rows;
          outputs += composed.shape.segments[s].outputs;
          settled[open[composed.work[s]]] = true;
        }
        ++report.steps;
        continue;
      }
      StepRefusal& refusal = ran.error();
      const auto refused_at = std::ranges::find_if(composed.work, [&](std::size_t j) {
        return refusal.slot.has_value() && subset[j].slot == *refusal.slot;
      });
      if (refused_at != composed.work.end()) {
        // One owner refused: its peers compose again. Out of state capacity
        // only for the joined step's wider reads (a joined alignment, or a
        // common read through a peer's cells), it steps alone (its own
        // reads are backed) before it is refused; else a long peer would
        // refuse it in every wave. Refused for its own reads, alone needs
        // the same: refused now.
        PendingWork& refused = live[open[*refused_at]];
        if (refusal.state_refused && refusal.widening && !refused.alone &&
            composed.work.size() > 1) {
          refused.alone = true;
          if (batch)
            for (auto& item : live) item.alone = true;
          restart = true;
          break;
        }
        settle(open[*refused_at], std::move(refusal));
        if (batch)
          for (auto& item : live) item.alone = true;
        restart = true;
        break;
      }
      if (refusal.clean && composed.work.size() > 1) {
        // The joined step was refused before anything ran: its owners alone.
        if (batch) {
          for (auto& item : live) item.alone = true;
        } else {
          for (const std::size_t j : composed.work) live[open[j]].alone = true;
        }
        restart = true;
        break;
      }
      for (const std::size_t j : composed.work) settle(open[j], refusal);
    }
    if (!restart) break;
  }
  return report;
}

}  // namespace llmp::engine
