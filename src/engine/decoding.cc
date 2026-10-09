// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "engine/decoding.h"

#include <algorithm>
#include <cmath>
#include <format>
#include <utility>

#include "engine/support.h"

namespace llmp::engine {

namespace {

namespace pl = llmp::planner;
using support::Error;
using Kind = DecodingMode::Kind;

StepRefusal Refusal(std::string what, std::optional<std::uint32_t> slot = std::nullopt) {
  return {.what = std::move(what), .clean = true, .slot = slot};
}

// The step's outputs and where its owner sat in it.
void Place(DecodeOutcome& out, WaveOutcome& ran) {
  out.outputs = std::move(ran.outputs);
  out.joined = std::move(ran.joined);
  out.segment = ran.segment;
  out.first_row = ran.first_row;
  out.first_output = ran.first_output;
  out.path = ran.path;
}

}  // namespace

std::uint32_t DecodingMode::verify_rows() const {
  std::uint32_t rows = 0;
  for (const DrafterSpec& d : drafters) {
    rows = combine == Drafters::kCascade ? rows + d.max_depth : std::max(rows, d.max_depth);
  }
  return rows + 1;
}

// --------------------------------------------------------------- the choice

DrafterChoice::DrafterChoice(std::vector<double> costs) {
  stats_.resize(costs.size());
  for (std::size_t i = 0; i < costs.size(); ++i) {
    stats_[i].cost = std::isfinite(costs[i]) && costs[i] > 0 ? costs[i] : 1.0;
  }
}

std::size_t DrafterChoice::Choose() const {
  for (std::size_t i = 0; i < stats_.size(); ++i) {
    if (stats_[i].seen < kExplore) return i;
  }
  return preferred_;
}

void DrafterChoice::Observe(std::size_t drafter, std::uint32_t committed) {
  if (drafter >= stats_.size() || committed == 0) return;
  Stats& s = stats_[drafter];
  ++s.seen;
  s.kept = s.seen == 1 ? committed : s.kept + ((committed - s.kept) / 8);
  if (std::ranges::any_of(stats_, [](const Stats& t) { return t.seen < kExplore; })) {
    preferred_ = 0;
    return;
  }
  const auto worth = [&](std::size_t i) { return stats_[i].kept / stats_[i].cost; };
  std::size_t best = preferred_;
  for (std::size_t i = 0; i < stats_.size(); ++i) {
    if (worth(i) > worth(best)) best = i;
  }
  // Only past a 3% margin, as the adaptive depth's.
  if (best != preferred_ && worth(best) > 1.03 * worth(preferred_)) preferred_ = best;
}

// ------------------------------------------------------------- orchestrator

DecodingOrchestrator::DecodingOrchestrator(ModelRunner& runner, const WaveComposer& composer,
                                           DecodingMode mode, DecodingPolicy policy)
    : runner_(&runner), composer_(&composer), mode_(std::move(mode)), policy_(std::move(policy)) {
  const std::uint32_t slots = runner.slots().slots();
  cursor_.assign(std::size_t{slots} * mode_.drafters.size(), 0);
  std::vector<double> costs;
  for (const DrafterSpec& d : mode_.drafters) costs.push_back(d.unit_cost);
  for (std::uint32_t slot = 0; slot < slots; ++slot) {
    for (const DrafterSpec& d : mode_.drafters) {
      adaptive_.emplace_back(d.adaptive ? d.max_depth : 1, d.depth_cost);
    }
    choice_.emplace_back(costs);
  }
}

std::expected<DecodingOrchestrator, std::string> DecodingOrchestrator::Make(
    ModelRunner& runner, const WaveComposer& composer, DecodingMode mode, DecodingPolicy policy) {
  const SlotLifecycle& slots = runner.slots();
  if (mode.kind == Kind::kBlockDiffusion) {
    return Error("block diffusion is a declared decoding mode, not built yet");
  }
  if (composer.stages() != slots.stages())
    return Error("the composer's stages are not the runner's");
  for (std::size_t s = 0; s < slots.stages(); ++s) {
    if (composer.read_align(s) != slots.read_align(s) || composer.context(s) != slots.context(s)) {
      return Error(std::format("the composer's stage {} reads otherwise than its state", s));
    }
  }
  if (mode.kind != Kind::kSpeculative) {
    if (!mode.drafters.empty()) return Error("only speculation drafts");
    return DecodingOrchestrator(runner, composer, std::move(mode), std::move(policy));
  }
  if (mode.drafters.empty()) return Error("speculation needs a drafter");
  if (!policy.propose || !policy.judge) return Error("speculation needs proposals and a judge");
  const std::uint32_t verify = mode.verify_rows();
  std::vector<bool> seen(slots.stages(), false);
  for (const DrafterSpec& d : mode.drafters) {
    if (d.stage == 0 || d.stage >= slots.stages() || seen[d.stage]) {
      return Error(std::format("a drafter's stage {} is not a drafter stage of its own", d.stage));
    }
    seen[d.stage] = true;
    const model::StateLayoutDerived& layout = slots.layout(d.stage);
    if (d.max_depth == 0 || (d.adaptive && d.max_depth > 7)) {
      return Error(
          std::format("stage {} drafts 1 to {} rows", d.stage, d.adaptive ? 7 : 1U << 31U));
    }
    // Its rows end within its chunk bound past the committed positions
    // (ModelRunner::Admit): a cascade's last drafter's at the verify's.
    if (verify > layout.max_rows) {
      return Error(std::format("a verify of {} rows passes stage {}'s chunk bound of {}", verify,
                               d.stage, layout.max_rows));
    }
    if (!layout.borrowed.empty() && !d.borrows_target) {
      return Error(std::format("stage {} reads the target's state without borrowing it", d.stage));
    }
  }
  if (verify > runner.max_verify_rows() || verify > slots.layout(0).max_rows) {
    return Error(std::format("a verify of {} rows passes the runner's {}", verify,
                             std::min(runner.max_verify_rows(), slots.layout(0).max_rows)));
  }
  return DecodingOrchestrator(runner, composer, std::move(mode), std::move(policy));
}

void DecodingOrchestrator::Reset(std::uint32_t slot) {
  const std::uint32_t positions = runner_->slots().positions(slot);
  for (std::size_t d = 0; d < mode_.drafters.size(); ++d) cursor_.at(At(slot, d)) = positions;
}

std::uint32_t DecodingOrchestrator::cursor(std::uint32_t slot, std::size_t drafter) const {
  return cursor_.at(At(slot, drafter));
}

const execution::AdaptiveDepth& DecodingOrchestrator::depth(std::uint32_t slot,
                                                            std::size_t drafter) const {
  return adaptive_.at(At(slot, drafter));
}

void DecodingOrchestrator::SetDepth(std::uint32_t slot, std::size_t drafter,
                                    const execution::AdaptiveDepth& depth) {
  adaptive_.at(At(slot, drafter)) = depth;
}

DecodeReport DecodingOrchestrator::Step(std::span<const DecodeWork> work) {
  if (mode_.kind != Kind::kSpeculative) {
    return Single(work, mode_.kind == Kind::kScored, mode_.kind == Kind::kPlain);
  }
  // The wave's form: speculation, or a plain decode row each (the target's
  // step still writes the drafters' taps).
  if (policy_.speculate && !policy_.speculate(static_cast<std::uint32_t>(work.size()))) {
    return Single(work, false, true);
  }
  return Speculate(work);
}

DecodeReport DecodingOrchestrator::Single(std::span<const DecodeWork> work, bool scored,
                                          bool greedy) {
  DecodeReport report;
  report.outcomes.resize(work.size());
  const SlotLifecycle& slots = runner_->slots();
  std::vector<PendingWork> pending;
  std::vector<SegmentWork> segments;
  std::vector<std::size_t> of;
  for (std::size_t i = 0; i < work.size(); ++i) {
    const DecodeWork& w = work[i];
    if (w.slot >= slots.slots()) {
      report.outcomes[i].refusal = Refusal(std::format("no slot {}", w.slot));
      continue;
    }
    const std::uint32_t positions = slots.positions(w.slot);
    // One row: the anchor at the slot's positions; scored: every given
    // token from them.
    const std::span<const std::int32_t> tokens = scored ? w.score
                                                 : w.history.size() == std::size_t{positions} + 1
                                                     ? w.history.last(1)
                                                     : std::span<const std::int32_t>{};
    if (tokens.empty()) {
      report.outcomes[i].refusal = Refusal(
          std::format("slot {}'s work does not continue its {} positions", w.slot, positions),
          w.slot);
      continue;
    }
    const auto rows = static_cast<std::uint32_t>(tokens.size());
    pending.push_back({.slot = w.slot,
                       .stage = 0,
                       .phase = scored ? pl::Phase::kScore : pl::Phase::kDecode,
                       .past = positions,
                       .rows = rows,
                       .outputs = scored ? rows : 1U,
                       .want_head = true,
                       .greedy = greedy && w.greedy,
                       .row_invariant = false,
                       .alone = false});
    segments.push_back({.tokens = tokens, .bytes = {}});
    of.push_back(i);
  }
  WaveReport wave = RunWave(*runner_, *composer_, pending, segments, /*advance=*/true);
  report.steps = wave.steps;
  for (std::size_t j = 0; j < pending.size(); ++j) {
    DecodeOutcome& out = report.outcomes[of[j]];
    WaveOutcome& ran = wave.outcomes[j];
    if (ran.refusal) {
      out.refusal = std::move(ran.refusal);
      continue;
    }
    out.kept = pending[j].rows;
    Place(out, ran);
  }
  return report;
}

DecodeReport DecodingOrchestrator::Speculate(std::span<const DecodeWork> work) {
  SlotLifecycle& slots = runner_->slots();
  const std::size_t drafters = mode_.drafters.size();
  const bool cascade = mode_.combine == DecodingMode::Drafters::kCascade;
  DecodeReport report;
  report.outcomes.resize(work.size());
  report.speculated = true;
  struct Unit {
    bool live = false;
    std::uint32_t positions = 0;
    std::vector<std::int32_t> proposed;
    std::vector<std::uint32_t> depth;        // by drafter: asked (0: not run)
    std::vector<bool> full;                  // by drafter: its depth was not cut short
    std::vector<std::uint32_t> first;        // by drafter: its first proposal's index
    std::vector<bool> caught_up;             // by drafter: its catch-up pass ran
    std::vector<std::size_t> run;            // by drafter: its DraftRun's index
    std::optional<std::size_t> alternative;  // the drafter chosen
  };
  std::vector<Unit> units(work.size());
  const std::uint32_t target_context = slots.context(0);

  // Each slot's unit: its drafters and their depths, within the verify's
  // rows (the reply's end, the context, the runner's verify).
  for (std::size_t i = 0; i < work.size(); ++i) {
    const DecodeWork& w = work[i];
    DecodeOutcome& out = report.outcomes[i];
    Unit& u = units[i];
    if (w.slot >= slots.slots() ||
        std::ranges::any_of(work.first(i), [&](const DecodeWork& o) { return o.slot == w.slot; })) {
      out.refusal = Refusal(std::format("slot {} has no unit of its own here", w.slot));
      continue;
    }
    u.positions = slots.positions(w.slot);
    if (w.history.size() != std::size_t{u.positions} + 1 || w.left == 0 ||
        u.positions >= target_context) {
      out.refusal = Refusal(
          std::format("slot {}'s work does not continue its {} positions", w.slot, u.positions),
          w.slot);
      continue;
    }
    u.live = true;
    u.depth.assign(drafters, 0);
    u.full.assign(drafters, false);
    u.first.assign(drafters, 0);
    u.caught_up.assign(drafters, false);
    u.run.assign(drafters, 0);
    std::uint32_t budget = static_cast<std::uint32_t>(std::min<std::uint64_t>(
                               {w.left, target_context - u.positions, mode_.verify_rows()})) -
                           1;
    std::vector<std::size_t> chosen;
    if (cascade) {
      for (std::size_t d = 0; d < drafters; ++d) chosen.push_back(d);
    } else {
      std::optional<std::size_t> pick = policy_.choose ? policy_.choose(w.slot) : std::nullopt;
      if (!pick || *pick >= drafters) pick = choice_.at(w.slot).Choose();
      u.alternative = *pick;
      chosen.push_back(*pick);
    }
    std::uint32_t proposals = 0;
    for (const std::size_t d : chosen) {
      const DrafterSpec& spec = mode_.drafters[d];
      const std::uint32_t want =
          spec.adaptive ? adaptive_.at(At(w.slot, d)).Choose() : spec.max_depth;
      // Its rows end inside its stage's context.
      const std::uint64_t end = std::uint64_t{u.positions} + proposals;
      const std::uint32_t context = slots.context(spec.stage);
      const std::uint32_t room = end < context ? static_cast<std::uint32_t>(context - end) : 0;
      u.depth[d] = std::min({want, budget, room});
      u.full[d] = u.depth[d] == want;
      budget -= u.depth[d];
      proposals += u.depth[d];
    }
  }

  // Each drafter in order: a cascade's drafter extends what the ones
  // before it proposed.
  for (std::size_t d = 0; d < drafters; ++d) {
    const DrafterSpec& spec = mode_.drafters[d];
    const bool block = spec.shape == DrafterSpec::Shape::kBlock;
    const std::uint32_t chunk = slots.layout(spec.stage).max_rows;
    std::vector<StateBorrow> borrows(work.size());  // returned as this drafter ends
    std::vector<bool> drafting(work.size(), false);
    std::uint32_t passes = 0;
    for (std::size_t i = 0; i < work.size(); ++i) {
      Unit& u = units[i];
      if (!u.live || u.depth[d] == 0) continue;
      DecodeOutcome& out = report.outcomes[i];
      u.first[d] = static_cast<std::uint32_t>(u.proposed.size());
      u.run[d] = out.drafts.size();
      out.drafts.push_back({.drafter = d, .depth = u.depth[d]});
      if (spec.borrows_target) {
        // The target's state frozen for its passes, once what the last
        // verify owes has run.
        const std::uint32_t slot = work[i].slot;
        Status settled = slots.owed(slot) ? slots.Settle(slot) : Status{};
        auto borrowed =
            settled ? slots.Borrow(slot, 0)
                    : std::expected<StateBorrow, std::string>(std::unexpected(settled.error()));
        if (!borrowed) {
          out.drafts.back().refusal = Refusal(borrowed.error(), slot);
          continue;
        }
        borrows[i] = std::move(*borrowed);
      }
      drafting[i] = true;
      passes = std::max(passes, block ? 1U : u.depth[d]);
    }
    for (std::uint32_t pass = 0; pass < passes; ++pass) {
      std::vector<PendingWork> pending;
      std::vector<std::vector<std::int32_t>> tokens;
      std::vector<std::size_t> of;
      tokens.reserve(work.size());
      for (std::size_t i = 0; i < work.size(); ++i) {
        Unit& u = units[i];
        if (!drafting[i] || pass >= (block ? 1U : u.depth[d])) continue;
        const DecodeWork& w = work[i];
        // The token at position q: the history's, then the proposals'.
        const auto token = [&](std::uint32_t q) {
          return q <= u.positions ? w.history[q] : u.proposed[q - u.positions - 1];
        };
        const auto last = static_cast<std::uint32_t>(u.positions + u.proposed.size());
        std::uint32_t past = last;
        std::uint32_t rows = block ? u.depth[d] : 1;
        if (!block && pass == 0 && spec.catch_up) {
          // Its catch-up: from where its state stops being the committed
          // tokens', through the last token known. One that fell further
          // behind than its chunk bound (an alternative not chosen for some
          // units) catches up its last chunk's rows: its older rows stay
          // stale, which costs acceptance, never what the verify keeps.
          past = std::min(cursor_.at(At(w.slot, d)), last);
          if (last - past + 1 > chunk) past = last + 1 - chunk;
          rows = last - past + 1;
        }
        std::vector<std::int32_t> in;
        if (block) {
          in.assign(rows, spec.fill);
          in.front() = token(last);
        } else {
          for (std::uint32_t q = past; q <= last; ++q) in.push_back(token(q));
        }
        tokens.push_back(std::move(in));
        pending.push_back({.slot = w.slot,
                           .stage = spec.stage,
                           .phase = pl::Phase::kDraft,
                           .past = past,
                           .rows = rows,
                           .outputs = block ? rows : 1U,
                           .want_head = true,
                           .greedy = false,
                           .row_invariant = false,
                           .alone = false});
        of.push_back(i);
      }
      if (pending.empty()) break;
      std::vector<SegmentWork> segments;
      for (const auto& t : tokens) segments.push_back({.tokens = t, .bytes = {}});
      WaveReport wave = RunWave(*runner_, *composer_, pending, segments, /*advance=*/false);
      report.steps += wave.steps;
      for (std::size_t j = 0; j < pending.size(); ++j) {
        const std::size_t i = of[j];
        Unit& u = units[i];
        DraftRun& run = report.outcomes[i].drafts[u.run[d]];
        WaveOutcome& ran = wave.outcomes[j];
        if (ran.refusal) {
          run.refusal = std::move(ran.refusal);
          drafting[i] = false;
          continue;
        }
        ++run.steps;
        if (pass == 0 && !block && spec.catch_up) u.caught_up[d] = true;
        const std::uint32_t want = block ? u.depth[d] : 1;
        auto proposed = policy_.propose(work[i].slot, d, ran.outputs, want);
        if (!proposed) {
          run.refusal = Refusal(proposed.error(), work[i].slot);
          drafting[i] = false;
          continue;
        }
        const auto take = static_cast<std::uint32_t>(std::min<std::size_t>(proposed->size(), want));
        u.proposed.insert(u.proposed.end(), proposed->begin(), proposed->begin() + take);
        run.proposed += take;
        if (take < want || block) drafting[i] = false;
      }
    }
  }

  // One verify of every slot's anchor and proposals.
  std::vector<PendingWork> pending;
  std::vector<std::vector<std::int32_t>> verified;
  std::vector<std::size_t> of;
  verified.reserve(work.size());
  for (std::size_t i = 0; i < work.size(); ++i) {
    Unit& u = units[i];
    if (!u.live) continue;
    std::vector<std::int32_t> rows = {work[i].history.back()};
    rows.insert(rows.end(), u.proposed.begin(), u.proposed.end());
    report.outcomes[i].proposed = u.proposed;
    pending.push_back({.slot = work[i].slot,
                       .stage = 0,
                       .phase = pl::Phase::kVerify,
                       .past = u.positions,
                       .rows = static_cast<std::uint32_t>(rows.size()),
                       .outputs = static_cast<std::uint32_t>(rows.size()),
                       .want_head = true,
                       .greedy = false,
                       .row_invariant = mode_.row_invariant,
                       .alone = false});
    verified.push_back(std::move(rows));
    of.push_back(i);
  }
  std::vector<SegmentWork> segments;
  for (const auto& v : verified) segments.push_back({.tokens = v, .bytes = {}});
  WaveReport wave = RunWave(*runner_, *composer_, pending, segments, /*advance=*/false);
  report.steps += wave.steps;
  for (std::size_t j = 0; j < pending.size(); ++j) {
    const std::size_t i = of[j];
    const std::uint32_t slot = work[i].slot;
    Unit& u = units[i];
    DecodeOutcome& out = report.outcomes[i];
    WaveOutcome& ran = wave.outcomes[j];
    if (ran.refusal) {
      out.refusal = std::move(ran.refusal);
      continue;
    }
    Place(out, ran);
    // A drafter that caught up rewrote its state from the committed tokens
    // up to the verify's first row; from there its rows were proposals.
    for (std::size_t d = 0; d < drafters; ++d) {
      if (u.caught_up[d]) cursor_.at(At(slot, d)) = u.positions;
    }
    const auto rows = static_cast<std::uint32_t>(verified[j].size());
    auto verdict = policy_.judge(slot, verified[j], out.outputs);
    std::string refused;
    if (!verdict) {
      refused = verdict.error();
    } else if (verdict->keep == 0 || verdict->keep > rows) {
      refused = std::format("a verdict keeps {} of {} rows", verdict->keep, rows);
    } else if (auto accepted = slots.Accept(slot, verdict->keep); !accepted) {
      refused = accepted.error();
    }
    if (!refused.empty()) {
      // Nothing kept: the verify undone whole.
      out.discarded = true;
      if (auto discarded = slots.DiscardVerify(slot); !discarded) {
        refused += "; " + discarded.error();
      }
      out.refusal = Refusal(std::move(refused), slot);
      continue;
    }
    out.kept = verdict->keep;
    out.tokens = std::move(verdict->tokens);
    // What each drafter's proposals earned: a cascade's drafter counts only
    // when every proposal before its own was accepted (else its own were
    // never judged), and only a complete step teaches its depth.
    const std::uint32_t accepted = verdict->keep - 1;
    for (std::size_t d = 0; d < drafters; ++d) {
      if (u.depth[d] == 0) continue;
      const DraftRun& run = out.drafts[u.run[d]];
      const bool judged = accepted >= u.first[d];
      const std::uint32_t own = judged ? std::min(accepted - u.first[d], run.proposed) : 0;
      if (mode_.drafters[d].adaptive) {
        const bool complete = u.full[d] && judged && !run.refusal && run.proposed == u.depth[d];
        adaptive_.at(At(slot, d)).Observe(u.depth[d], own + 1, complete);
      }
    }
    if (u.alternative) choice_.at(slot).Observe(*u.alternative, verdict->keep);
  }
  return report;
}

}  // namespace llmp::engine
