// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "engine/wave_composer.h"

#include <algorithm>
#include <format>
#include <tuple>

#include "engine/request_cohort.h"
#include "engine/support.h"

namespace llmp::engine {

namespace {

namespace pl = llmp::planner;
using support::Error;
using support::Round;

bool AlignValid(std::uint32_t align) { return align != 0 && align % 256 == 0; }

}  // namespace

std::expected<WaveComposer, std::string> WaveComposer::Make(model::Batching caps,
                                                            std::uint32_t context,
                                                            std::uint32_t read_align) {
  if (context == 0) return Error("a wave composer needs a context");
  if (!AlignValid(read_align)) return Error("a read alignment is a positive multiple of 256");
  if (caps.joined_read_align != 0 && !AlignValid(caps.joined_read_align)) {
    return Error("a joined read alignment is a positive multiple of 256");
  }
  if (caps.max_segments == 0 || caps.max_segments > kMaxRequestSlots) {
    return Error(std::format("a step joins 1 to {} owners", kMaxRequestSlots));
  }
  return WaveComposer(caps, context, read_align);
}

std::uint32_t WaveComposer::LoneRead(std::uint32_t past, std::uint32_t rows) const {
  return static_cast<std::uint32_t>(
      std::min<std::uint64_t>(Round(std::uint64_t{past} + rows, read_align_), context_));
}

void WaveComposer::Finish(pl::StepShape& shape) const {
  shape.read_align = AlignFor(shape.segments.size());
  std::uint32_t widest = 0;
  for (pl::Segment& s : shape.segments) {
    s.read_cells = static_cast<std::uint32_t>(
        std::min<std::uint64_t>(Round(std::uint64_t{s.past} + s.rows, shape.read_align), context_));
    widest = std::max(widest, s.read_cells);
  }
  if (caps_.common_read && shape.segments.size() > 1) {
    for (pl::Segment& s : shape.segments) s.read_cells = widest;
  }
  const std::uint32_t group_segments = shape.row_invariant && caps_.invariant_group_segments != 0
                                           ? caps_.invariant_group_segments
                                           : caps_.group_segments;
  std::uint8_t group = 0;
  std::uint32_t rows = 0, owners = 0;
  for (pl::Segment& s : shape.segments) {
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

Composition WaveComposer::Compose(std::span<const PendingWork> pending) const {
  Composition out;
  std::vector<std::size_t> order;
  for (std::size_t i = 0; i < pending.size(); ++i) {
    const PendingWork& p = pending[i];
    std::string why;
    if (p.rows == 0 || p.past > context_ || p.rows > context_ - p.past) {
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
  const auto kind = [&](const PendingWork& p) {
    return std::make_tuple(p.stage, static_cast<int>(p.phase), p.row_invariant, p.greedy,
                           caps_.mixed_heads || p.want_head);
  };
  std::ranges::stable_sort(order, [&](std::size_t a, std::size_t b) {
    return std::tuple_cat(kind(pending[a]), std::make_tuple(pending[a].alone, pending[a].slot)) <
           std::tuple_cat(kind(pending[b]), std::make_tuple(pending[b].alone, pending[b].slot));
  });
  const auto lone_read = [&](const PendingWork& p) { return LoneRead(p.past, p.rows); };
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
      join = kind(first) == kind(p) && !first.alone && !p.alone && caps_.max_segments > 1 &&
             (p.phase != pl::Phase::kPrefill || caps_.join_prefill) &&
             step.work.size() < caps_.max_segments &&
             (caps_.max_step_rows == 0 || step_rows + p.rows <= caps_.max_step_rows) &&
             (p.phase != pl::Phase::kPrefill || !caps_.equal_prefill ||
              (p.rows == first.rows && lone_read(p) == lone_read(first)));
    }
    if (!join) close();
    if (step.work.empty()) {
      step.shape.stage = p.stage;
      step.shape.phase = p.phase;
      step.shape.row_invariant = p.row_invariant;
    }
    step.shape.segments.push_back({.slot = static_cast<std::uint8_t>(p.slot),
                                   .rows = p.rows,
                                   .past = p.past,
                                   .read_cells = 0,
                                   .outputs = p.outputs,
                                   .group = 0,
                                   .want_head = p.want_head,
                                   .greedy = p.greedy});
    step.work.push_back(i);
    step_rows += p.rows;
  }
  close();
  return out;
}

WaveReport RunWave(ModelRunner& runner, const WaveComposer& composer,
                   std::span<const PendingWork> pending, std::span<const SegmentWork> work,
                   bool advance, const AheadFor& ahead) {
  WaveReport report;
  report.outcomes.resize(pending.size());
  std::vector<bool> settled(pending.size());
  const auto settle = [&](std::size_t i, StepRefusal refusal) {
    report.outcomes[i].refusal = std::move(refusal);
    settled[i] = true;
  };
  // An owner's lone read is the composer's alignment; what the runner backs
  // first for each owner of a joined step is its state's. Equal, an owner
  // whose widening is refused steps alone with no growth.
  const char* mismatch = work.size() != pending.size()
                             ? "the wave's work does not match its pending work"
                         : composer.read_align() != runner.slots().read_align()
                             ? "the composer's read alignment is not the state's"
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
      std::vector<std::size_t> indices;
      for (const std::size_t j : composed.work) {
        segments.push_back(work[open[j]]);
        indices.push_back(open[j]);
      }
      const StepRequest request{.shape = composed.shape,
                                .work = segments,
                                .step_bytes = {},
                                .advance = advance,
                                .ahead = ahead ? ahead(composed, indices) : StepAhead{}};
      auto ran = runner.RunStep(request);
      if (ran) {
        std::uint32_t rows = 0, outputs = 0;
        for (std::size_t s = 0; s < composed.work.size(); ++s) {
          WaveOutcome& outcome = report.outcomes[open[composed.work[s]]];
          outcome.outputs = std::move(ran->outputs[s]);
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
          restart = true;
          break;
        }
        settle(open[*refused_at], std::move(refusal));
        restart = true;
        break;
      }
      if (refusal.clean && composed.work.size() > 1) {
        // The joined step was refused before anything ran: its owners alone.
        for (const std::size_t j : composed.work) live[open[j]].alone = true;
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
