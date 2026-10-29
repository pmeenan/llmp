// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "engine/decoding.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <format>
#include <utility>

#include "engine/prompt_wave.h"
#include "engine/support.h"
#include "model/state_contract.h"

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

std::expected<std::uint32_t, std::string> DrafterHeadRows(const DrafterSpec& drafter,
                                                          std::uint32_t target_vocabulary) {
  if (target_vocabulary == 0 || target_vocabulary > std::uint32_t{INT32_MAX})
    return Error("a drafter head has no supported target vocabulary");
  const auto rows = drafter.head_rows ? drafter.head_rows : target_vocabulary;
  if (rows > target_vocabulary ||
      (drafter.head_ids && !drafter.head_ids->empty() && drafter.head_ids->size() != rows))
    return Error("a drafter head width differs from its source vocabulary map");
  return rows;
}

std::expected<std::int32_t, std::string> DrafterHeadToken(const DrafterSpec& drafter,
                                                          std::uint32_t row,
                                                          std::uint32_t target_vocabulary) {
  auto rows = DrafterHeadRows(drafter, target_vocabulary);
  if (!rows) return Error(rows.error());
  if (row >= *rows) return Error("a draft head row is out of range");
  if (!drafter.head_ids || drafter.head_ids->empty()) return static_cast<std::int32_t>(row);
  const auto token = drafter.head_ids->at(row);
  if (token < 0 || std::cmp_greater_equal(token, target_vocabulary))
    return Error("a selected draft head token is outside the target vocabulary");
  return token;
}

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
  const auto use = mode.kind == Kind::kSpeculative      ? NumericalUse::kSpeculative
                   : mode.kind == Kind::kScored         ? NumericalUse::kScore
                   : mode.kind == Kind::kBlockDiffusion ? NumericalUse::kBlock
                                                        : NumericalUse::kDecode;
  if (auto declared = runner.CheckExecutionUse(use); !declared) return Error(declared.error());
  const SlotLifecycle& slots = runner.slots();
  if (mode.kind == Kind::kBlockDiffusion) {
    return Error("block diffusion is a declared decoding mode, not built yet");
  }
  if (composer.stages() != slots.stages())
    return Error("the composer's stages are not the runner's");
  for (std::size_t s = 0; s < slots.stages(); ++s) {
    if (composer.read_align(s) != slots.read_align(s) ||
        composer.read_buckets(s) != slots.read_buckets() ||
        composer.context(s) != slots.context(s)) {
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
    if (d.draft_before_limit && mode.drafters.size() != 1)
      return Error("complete passes before verify truncation need one drafter");
    seen[d.stage] = true;
    const model::StateLayoutDerived& layout = slots.layout(d.stage);
    if (d.max_depth == 0 || (d.adaptive && d.max_depth > 7)) {
      return Error(
          std::format("stage {} drafts 1 to {} rows", d.stage, d.adaptive ? 7 : 1U << 31U));
    }
    if (d.shared_wave_depth > d.max_depth ||
        (d.shared_wave_depth != 0 && d.shape != DrafterSpec::Shape::kAutoregressive)) {
      return Error("a shared-wave draft depth must bound an autoregressive drafter");
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
    if (d.anchored && (d.shape != DrafterSpec::Shape::kAutoregressive || d.catch_up)) {
      return Error(
          std::format("stage {}'s anchored drafter is autoregressive, without catch-up", d.stage));
    }
    if (d.features && model::FeatureTap(layout) == nullptr) {
      return Error(std::format("stage {} reads features but has no feature tap", d.stage));
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
  const auto use = mode_.kind == Kind::kSpeculative ? NumericalUse::kSpeculative
                   : mode_.kind == Kind::kScored    ? NumericalUse::kScore
                                                    : NumericalUse::kDecode;
  auto qualified = runner_->CheckExecutionUse(use);
  if (!qualified) {
    DecodeReport report;
    report.outcomes.resize(work.size());
    for (auto& out : report.outcomes) out.refusal = Refusal(qualified.error());
    return report;
  }
  if (std::ranges::any_of(work, [](const auto& unit) { return !unit.certain.empty(); })) {
    auto partition = runner_->PayHostMetadata(
        work.size() * (2 * sizeof(DecodeWork) + 2 * sizeof(std::size_t) + sizeof(DecodeOutcome)));
    if (!partition) {
      DecodeReport report;
      report.outcomes.resize(work.size());
      for (auto& out : report.outcomes) out.refusal = Refusal(partition.error());
      return report;
    }
    std::vector<DecodeWork> known, ordinary;
    known.reserve(work.size());
    ordinary.reserve(work.size());
    std::vector<std::size_t> ki, oi;
    ki.reserve(work.size());
    oi.reserve(work.size());
    DecodeReport result;
    result.outcomes.resize(work.size());
    for (std::size_t i = 0; i < work.size(); ++i) {
      if (std::ranges::any_of(work.first(i),
                              [&](const DecodeWork& o) { return o.slot == work[i].slot; })) {
        result.outcomes[i].refusal =
            Refusal(std::format("slot {} has no unit of its own here", work[i].slot));
        continue;
      }
      if (work[i].certain.empty()) {
        ordinary.push_back(work[i]);
        oi.push_back(i);
      } else {
        known.push_back(work[i]);
        ki.push_back(i);
      }
    }
    const auto merge = [&](DecodeReport report, const auto& indexes) {
      result.steps += report.steps;
      result.speculated = result.speculated || report.speculated;
      for (std::size_t i = 0; i < indexes.size(); ++i)
        result.outcomes[indexes[i]] = std::move(report.outcomes[i]);
    };
    if (!known.empty()) merge(Certain(known), ki);
    if (!ordinary.empty()) merge(Step(ordinary), oi);
    return result;
  }
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

WaveReadback DecodingOrchestrator::Readback(std::span<const PendingWork> pending) const {
  if (!policy_.readback || !policy_.read) return {};
  return {.want =
              [this, pending](std::size_t j, std::span<const std::int32_t> tokens,
                              std::span<const std::byte> outputs) {
                return policy_.readback(pending[j].slot, tokens, outputs);
              },
          .take =
              [this, pending](std::size_t j, std::uint32_t row,
                              std::expected<std::span<const std::byte>, std::string> logits) {
                policy_.read(pending[j].slot, row, std::move(logits));
              }};
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
    if (std::ranges::any_of(work.first(i), [&](const DecodeWork& o) { return o.slot == w.slot; })) {
      report.outcomes[i].refusal =
          Refusal(std::format("slot {} has no unit of its own here", w.slot));
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
  if (!scored) {
    if (!pending.empty()) {
      if (auto allowed = runner_->CheckExecutionUse(NumericalUse::kDecode); !allowed) {
        for (const auto idx : of)
          report.outcomes[idx].refusal = StepRefusal{.what = allowed.error()};
        return report;
      }
    }
    for (std::size_t j = 0; j < pending.size(); ++j) segments[j].history = work[of[j]].history;
    WaveReport wave =
        RunWave(*runner_, *composer_, pending, segments, true, {}, {}, {}, Readback(pending));
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
  std::vector<std::vector<std::int32_t>> full(pending.size());
  std::vector<std::span<const std::int32_t>> histories;
  for (std::size_t j = 0; j < pending.size(); ++j) {
    const auto& w = work[of[j]];
    if (!scored) {
      histories.push_back(w.history);
    } else if (w.history.size() >= pending[j].past) {
      full[j].assign(w.history.begin(), w.history.begin() + pending[j].past);
      full[j].insert(full[j].end(), w.score.begin(), w.score.end());
      histories.push_back(full[j]);
    } else {
      histories.emplace_back();
    }
  }
  WaveReport wave = RunPromptWave(*runner_, *composer_, pending, segments, histories);
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

DecodeReport DecodingOrchestrator::Certain(std::span<const DecodeWork> work) {
  DecodeReport report;
  report.outcomes.resize(work.size());
  // The rows' tokens (the anchor, then the certain ones), contiguous; the
  // whole history before them only for a target whose steps read it.
  const bool whole = runner_->ReadsHistory(0);
  std::uint64_t bytes = 0;
  for (const auto& unit : work)
    bytes += ((whole ? unit.history.size() : 1) + unit.certain.size()) * sizeof(std::int32_t) +
             sizeof(PendingWork) + sizeof(SegmentWork) + 256;
  auto payment = runner_->PayHostMetadata(bytes);
  if (!payment) {
    for (auto& out : report.outcomes) out.refusal = Refusal(payment.error());
    return report;
  }
  std::vector<std::vector<std::int32_t>> histories(work.size());
  std::vector<std::span<const std::int32_t>> views;
  std::vector<PendingWork> pending;
  std::vector<SegmentWork> segments;
  std::vector<std::size_t> indexes;
  const auto& slots = runner_->slots();
  views.reserve(work.size());
  pending.reserve(work.size());
  segments.reserve(work.size());
  indexes.reserve(work.size());
  for (std::size_t i = 0; i < work.size(); ++i) {
    const auto& unit = work[i];
    if (std::ranges::any_of(work.first(i),
                            [&](const DecodeWork& o) { return o.slot == unit.slot; })) {
      report.outcomes[i].refusal =
          Refusal(std::format("slot {} has no unit of its own here", unit.slot));
      continue;
    }
    if (unit.slot >= slots.slots() || unit.certain.empty() ||
        unit.history.size() != std::size_t{slots.positions(unit.slot)} + 1 ||
        unit.certain.size() >= unit.left || !unit.score.empty()) {
      report.outcomes[i].refusal =
          Refusal("certain inputs do not continue one anchor within the reply budget");
      continue;
    }
    const auto past = slots.positions(unit.slot);
    const auto kept = whole ? unit.history : unit.history.last(1);
    histories[i].reserve(kept.size() + unit.certain.size());
    histories[i].assign(kept.begin(), kept.end());
    histories[i].insert(histories[i].end(), unit.certain.begin(), unit.certain.end());
    const auto rows = static_cast<std::uint32_t>(1 + unit.certain.size());
    pending.push_back({.slot = unit.slot,
                       .stage = 0,
                       .phase = unit.certain_scores ? pl::Phase::kScore : pl::Phase::kPrefill,
                       .past = past,
                       .rows = rows,
                       .outputs = unit.certain_scores ? rows : 1U,
                       .want_head = true,
                       .greedy = unit.greedy,
                       .row_invariant = false,
                       .alone = false});
    segments.push_back(
        {.tokens = std::span<const std::int32_t>(histories[i]).subspan(whole ? past : 0),
         .bytes = {}});
    views.push_back(whole ? std::span<const std::int32_t>(histories[i])
                          : std::span<const std::int32_t>{});
    indexes.push_back(i);
  }
  // One target step consumes every known row, writing the drafters' tapped
  // features as any target step does. No drafter runs and no acceptance is
  // drawn; prompt numerical permission covers the causal rows.
  auto wave = RunPromptWave(*runner_, *composer_, pending, segments, views);
  report.steps = wave.steps;
  for (std::size_t j = 0; j < indexes.size(); ++j) {
    auto& out = report.outcomes[indexes[j]];
    auto& ran = wave.outcomes[j];
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
    std::vector<std::uint32_t> depth;         // by drafter: asked (0: not run)
    std::vector<bool> full;                   // by drafter: its depth was not cut short
    std::vector<std::uint32_t> first;         // by drafter: its first proposal's index
    std::vector<bool> caught_up;              // by drafter: its catch-up pass ran
    std::vector<std::size_t> run;             // by drafter: its DraftRun's index
    std::optional<std::uint32_t> full_depth;  // checked before any replay unit begins
    bool full_cut = false;                    // full_depth was cut at the context's end
    std::optional<std::size_t> alternative;   // the drafter chosen
    // The history through the proposals so far, copied once a unit, for
    // stages whose steps read it (ModelRunner::ReadsHistory); else empty.
    std::vector<std::int32_t> history;
  };
  std::vector<Unit> units(work.size());
  const std::uint32_t target_context = slots.context(0);
  const bool target_history = runner_->ReadsHistory(0);
  bool any_history = target_history;
  for (const DrafterSpec& d : mode_.drafters)
    any_history = any_history || runner_->ReadsHistory(d.stage);

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
    if (drafters == 1 && mode_.drafters[0].draft_before_limit) {
      const auto& spec = mode_.drafters[0];
      auto want = policy_.depth   ? policy_.depth(w.slot, 0)
                  : spec.adaptive ? adaptive_.at(At(w.slot, 0)).Choose()
                                  : spec.max_depth;
      const auto context = slots.context(spec.stage);
      if (want == 0 || want > spec.max_depth) {
        out.refusal = Refusal("the requested draft depth exceeds its declared bounds", w.slot);
        continue;
      }
      if (work.size() > 1 && spec.shared_wave_depth != 0) {
        want = std::min(want, spec.shared_wave_depth);
      }
      // Near the context's end the schedule shrinks to the rows left (its
      // stage's, and the target's), and a unit with none left verifies its
      // anchor alone; such a unit teaches nothing (full_cut).
      const std::uint32_t room =
          u.positions < context ? std::min(target_context, context) - u.positions : 0;
      u.full_cut = want > room;
      u.full_depth = std::min(want, room);
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
      // A first pass reads the features before its rows: none at position 0.
      if ((spec.anchored || spec.features) && u.positions == 0) continue;
      if (u.full_depth == 0U) continue;  // no row left before the context's end
      std::uint32_t want = u.full_depth    ? *u.full_depth
                           : policy_.depth ? policy_.depth(w.slot, d)
                           : spec.adaptive ? adaptive_.at(At(w.slot, d)).Choose()
                                           : spec.max_depth;
      if (want == 0 || want > spec.max_depth) {
        u.live = false;
        out.refusal = Refusal("the requested draft depth exceeds its declared bounds", w.slot);
        break;
      }
      if (work.size() > 1 && spec.shared_wave_depth != 0) {
        want = std::min(want, spec.shared_wave_depth);
      }
      // Its rows end inside its stage's context.
      const std::uint64_t end = std::uint64_t{u.positions} + proposals;
      const std::uint32_t context = slots.context(spec.stage);
      const std::uint32_t room = end < context ? static_cast<std::uint32_t>(context - end) : 0;
      u.depth[d] = spec.draft_before_limit ? want : std::min({want, budget, room});
      u.full[d] = u.depth[d] == want && !u.full_cut;
      budget -= std::min(budget, u.depth[d]);
      proposals += u.depth[d];
    }
    if (u.live && any_history) {
      u.history.reserve(w.history.size() + proposals);
      u.history.assign(w.history.begin(), w.history.end());
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
      if (spec.borrows_target && !slots.owed(work[i].slot)) {
        auto borrowed = slots.Borrow(work[i].slot, 0);
        if (!borrowed) {
          out.drafts.back().refusal = Refusal(borrowed.error(), work[i].slot);
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
      std::vector<std::span<const std::int32_t>> histories;
      std::vector<FeatureFrom> features;
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
        // An anchored pass stays at the committed positions.
        std::uint32_t past = spec.anchored ? u.positions : last;
        std::uint32_t rows = block ? u.depth[d] : 1;
        std::uint32_t token_first = past;
        std::uint32_t feature_first = past == 0 ? 0 : past - 1;
        if (!block && pass == 0 && spec.catch_up) {
          // Its catch-up: from where its state stops being the committed
          // tokens', through the last token known. One that fell further
          // behind than its chunk bound (an alternative not chosen for some
          // units) catches up its last chunk's rows: its older rows stay
          // stale, which costs acceptance, never what the verify keeps.
          past = std::min(cursor_.at(At(w.slot, d)), last);
          // A plain step since its last verify (a forced or certain span, a
          // plain wave) committed rows its cursor never followed: it catches
          // up those the target's tap still holds, from the oldest held
          // feature; older rows stay stale, which, as below, costs
          // acceptance only.
          if (spec.features) {
            if (const auto oldest = runner_->OldestTapFeature(w.slot, spec.stage);
                oldest && past < *oldest + 1)
              past = std::min(*oldest + 1, last);
          }
          if (last - past + 1 > chunk) past = last + 1 - chunk;
          rows = last - past + 1;
          token_first = past;
          feature_first = past == 0 ? 0 : past - 1;
        }
        if (spec.borrows_target && pass == 0 && !borrows[i].held()) {
          // Coordinates are now concrete for this reader. A guard reserves
          // the exact accepted prefix without publishing a frozen view.
          auto borrowed = slots.BorrowAfterOwed(w.slot, spec.stage, past, rows);
          if (!borrowed) {
            report.outcomes[i].drafts[u.run[d]].refusal = Refusal(borrowed.error(), w.slot);
            drafting[i] = false;
            continue;
          }
          borrows[i] = std::move(*borrowed);
        }
        std::vector<std::int32_t> in;
        if (block) {
          in.assign(rows, spec.fill);
          in.front() = token(last);
        } else if (spec.anchored) {
          in.push_back(token(last));
        } else {
          for (std::uint32_t r = 0; r < rows; ++r) in.push_back(token(token_first + r));
        }
        // Its features: the target's before its rows on the unit's first
        // pass, then its own last pass's.
        if (spec.features) {
          features.push_back(
              pass == 0 ? FeatureFrom{.kind = FeatureFrom::Kind::kTap, .position = feature_first}
                        : FeatureFrom{.kind = FeatureFrom::Kind::kCarry});
        } else {
          features.emplace_back();
        }
        histories.push_back(runner_->ReadsHistory(spec.stage)
                                ? std::span<const std::int32_t>(u.history)
                                : std::span<const std::int32_t>{});
        tokens.push_back(std::move(in));
        pending.push_back({.slot = w.slot,
                           .stage = spec.stage,
                           .phase = pl::Phase::kDraft,
                           .past = past,
                           .rows = rows,
                           .outputs = block ? rows : 1U,
                           .want_head = true,
                           .greedy = spec.device_tokens,
                           .row_invariant = false,
                           .alone = false});
        of.push_back(i);
      }
      if (pending.empty()) break;
      if (spec.draft_wave_max > 0 && pending.size() > spec.draft_wave_max) {
        for (PendingWork& w : pending) w.alone = true;
      }
      std::vector<SegmentWork> segments;
      for (std::size_t j = 0; j < tokens.size(); ++j) {
        segments.push_back(
            {.tokens = tokens[j], .bytes = {}, .feature = features[j], .history = histories[j]});
      }
      std::vector<StepAuthorization> authorization(pending.size());
      for (std::size_t j = 0; j < pending.size(); ++j) {
        const auto borrowed = borrows[of[j]].permit();
        if (borrowed) {
          auto& permit = authorization[j];
          permit.runner_ = runner_;
          permit.slot_ = pending[j].slot;
          permit.stage_ = pending[j].stage;
          permit.phase_ = pending[j].phase;
          permit.past_ = pending[j].past;
          permit.rows_ = pending[j].rows;
          permit.outputs_ = pending[j].outputs;
          permit.want_head_ = pending[j].want_head;
          permit.greedy_ = pending[j].greedy;
          permit.feature_ = segments[j].feature;
          permit.borrowed_ = borrowed;
        }
      }
      WaveReport wave =
          RunWave(*runner_, *composer_, pending, segments, /*advance=*/false, {}, authorization);
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
        if (pass == 0 && !block && spec.catch_up) {
          u.caught_up[d] = true;
        }
        const std::uint32_t want = block ? u.depth[d] : 1;
        auto proposed = policy_.propose(work[i].slot, d, ran.outputs, want);
        if (!proposed) {
          run.refusal = Refusal(proposed.error(), work[i].slot);
          drafting[i] = false;
          continue;
        }
        const auto take = static_cast<std::uint32_t>(std::min<std::size_t>(proposed->size(), want));
        u.proposed.insert(u.proposed.end(), proposed->begin(), proposed->begin() + take);
        if (any_history)
          u.history.insert(u.history.end(), proposed->begin(), proposed->begin() + take);
        run.proposed += take;
        if (take < want || block) drafting[i] = false;
      }
    }
  }

  // One verify of every slot's anchor and proposals.
  std::vector<PendingWork> pending;
  std::vector<std::vector<std::int32_t>> verified;
  std::vector<std::span<const std::int32_t>> histories;
  std::vector<std::size_t> of;
  verified.reserve(work.size());
  const bool full_block_verify = drafters == 1 &&
                                 mode_.drafters[0].shape == DrafterSpec::Shape::kBlock &&
                                 mode_.drafters[0].draft_before_limit;
  for (std::size_t i = 0; i < work.size(); ++i) {
    Unit& u = units[i];
    if (!u.live) continue;
    const auto count =
        (full_block_verify && std::uint64_t{u.positions} + 1 + u.proposed.size() <= target_context)
            ? u.proposed.size()
            : std::min<std::size_t>(u.proposed.size(),
                                    std::min(work[i].left, target_context - u.positions) - 1U);
    std::vector<std::int32_t> rows = {work[i].history.back()};
    rows.insert(rows.end(), u.proposed.begin(),
                u.proposed.begin() + static_cast<std::ptrdiff_t>(count));
    report.outcomes[i].proposed.assign(u.proposed.begin(),
                                       u.proposed.begin() + static_cast<std::ptrdiff_t>(count));
    pending.push_back({.slot = work[i].slot,
                       .stage = 0,
                       .phase = pl::Phase::kVerify,
                       .past = u.positions,
                       .rows = static_cast<std::uint32_t>(rows.size()),
                       .outputs = static_cast<std::uint32_t>(rows.size()),
                       .want_head = true,
                       .greedy = work[i].greedy,
                       .row_invariant = mode_.row_invariant,
                       .alone = false});
    histories.push_back(target_history ? std::span<const std::int32_t>(u.history).first(
                                             work[i].history.size() + count)
                                       : std::span<const std::int32_t>{});
    verified.push_back(std::move(rows));
    of.push_back(i);
  }
  std::vector<SegmentWork> segments;
  for (std::size_t i = 0; i < verified.size(); ++i)
    segments.push_back({.tokens = verified[i], .bytes = {}, .history = histories[i]});
  WaveReport wave = RunWave(*runner_, *composer_, pending, segments, /*advance=*/false, {}, {}, {},
                            Readback(pending));
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
    // A full-block verify may hold rows past what the reply may still
    // emit: only those within `left` are judged (each row's outputs alike).
    std::span<const std::int32_t> judged_rows = verified[j];
    std::span<const std::byte> judged_outputs = out.outputs;
    if (full_block_verify && rows > work[i].left && work[i].left > 0 &&
        out.outputs.size() % rows == 0) {
      judged_rows = judged_rows.first(work[i].left);
      judged_outputs = judged_outputs.first(out.outputs.size() / rows * work[i].left);
    }
    auto verdict = policy_.judge(slot, judged_rows, judged_outputs);
    if (verdict && full_block_verify && verdict->keep > work[i].left) {
      verdict->keep = work[i].left;
      if (verdict->tokens.size() > verdict->keep) verdict->tokens.resize(verdict->keep);
    }
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
      const bool complete = u.full[d] && judged && !run.refusal && run.proposed == u.depth[d] &&
                            rows == u.proposed.size() + 1;
      if (policy_.observe)
        policy_.observe(slot, d, u.depth[d], own + 1, complete);
      else if (mode_.drafters[d].adaptive)
        adaptive_.at(At(slot, d)).Observe(u.depth[d], own + 1, complete);
    }
    if (u.alternative) choice_.at(slot).Observe(*u.alternative, verdict->keep);
  }

  for (std::size_t i = 0; i < report.outcomes.size(); ++i)
    if (auto& refusal = report.outcomes[i].refusal; refusal && work[i].slot < slots.slots())
      refusal->clean = refusal->clean && slots.usable(work[i].slot);
  return report;
}

}  // namespace llmp::engine
